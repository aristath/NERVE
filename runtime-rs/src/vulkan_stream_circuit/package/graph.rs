#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VulkanResidentPackageCircuitGraph {
    pub topology: String,
    pub edges: Vec<crate::stream_circuit::StreamCircuitGraphEdge>,
    pub boundary: StreamCircuitGraphBoundary,
    #[serde(default)]
    pub architecture: Value,
    #[serde(default)]
    pub dimensions: Value,
    #[serde(default)]
    pub input_transducer: Value,
    #[serde(default)]
    pub output_transducer: Value,
    #[serde(default)]
    pub components: Vec<VulkanResidentPackageComponentCircuit>,
}

impl VulkanResidentPackageCircuitGraph {
    fn with_kernel_runtime_contracts(
        &self,
        executions: &[VulkanResidentComponentExecutionSpec],
    ) -> Result<Self, VulkanResidentTokenModelPackageError> {
        let execution_by_component = executions
            .iter()
            .map(|execution| (execution.component_id.as_str(), execution))
            .collect::<BTreeMap<_, _>>();
        let mut graph = self.clone();
        for component in &mut graph.components {
            if !component.runtime_role.is_signal_processor() {
                continue;
            }
            let execution = execution_by_component
                .get(component.component_id.as_str())
                .ok_or_else(|| {
                    VulkanResidentTokenModelPackageError::new(format!(
                        "signal processor {:?} has no compiled kernel execution contract",
                        component.component_id
                    ))
                })?;
            for node in &mut component.circuit.nodes {
                let kernel = execution
                    .kernels
                    .iter()
                    .find(|kernel| kernel.node_id == node.id && kernel.op == node.op)
                    .ok_or_else(|| {
                        VulkanResidentTokenModelPackageError::new(format!(
                            "signal processor {} node {} has no matching compiled kernel contract",
                            component.component_id, node.id
                        ))
                    })?;
                let attrs = node.attrs.as_object_mut().ok_or_else(|| {
                    VulkanResidentTokenModelPackageError::new(format!(
                        "signal processor {} node {} attributes are not an object",
                        component.component_id, node.id
                    ))
                })?;
                attrs.insert(
                    "stream_control_binding".to_string(),
                    kernel
                        .stream_control_binding
                        .map_or(Value::Null, |binding| Value::Number(binding.into())),
                );
            }
        }
        Ok(graph)
    }

    pub(crate) fn to_resolved_lowered_execution_graph(
        &self,
        package_root: impl Into<PathBuf>,
    ) -> Result<ResolvedLoweredExecutionGraph, VulkanResidentTokenModelPackageError> {
        let mut operator_counts = BTreeMap::new();
        let mut circuit_refs = Vec::with_capacity(self.components.len());
        let mut circuits = Vec::with_capacity(self.components.len());

        for component in &self.components {
            *operator_counts
                .entry(component.operator_type.clone())
                .or_insert(0) += 1;
            let circuit_ref = LoweredCircuitRef {
                id: component.component_id.clone(),
                operator_type: component.operator_type.clone(),
                runtime_role: component.runtime_role,
                circuit: format!("package://{}/circuit", component.component_id),
                params: format!("package://{}/params", component.component_id),
                state: format!("package://{}/state", component.component_id),
                implementation: component.implementation.clone(),
                behavioral_role: component.behavioral_role.clone(),
            };
            let resolved = ResolvedCircuitArtifact {
                component: circuit_ref.clone(),
                circuit: component.circuit.clone(),
                params: component.params.clone(),
                state: component.state.clone(),
            };
            resolved.validate().map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "resident package circuit graph component {} is invalid: {error}",
                    component.component_id
                ))
            })?;
            circuit_refs.push(circuit_ref);
            circuits.push(resolved);
        }

        let index = LoweredExecutionGraph {
            schema: LOWERED_EXECUTION_GRAPH_SCHEMA.to_string(),
            source: LoweredExecutionGraphSource {
                format: VULKAN_RESIDENT_MODEL_PACKAGE_MANIFEST_SCHEMA.to_string(),
                artifact_root: "package".to_string(),
            },
            architecture: self.architecture.clone(),
            dimensions: self.dimensions.clone(),
            graph: LoweredExecutionGraphGraph {
                topology: self.topology.clone(),
                circuits: circuit_refs,
                edges: self.edges.clone(),
                boundary: self.boundary.clone(),
                input_transducer: self.input_transducer.clone(),
                output_transducer: self.output_transducer.clone(),
            },
            summary: LoweredExecutionGraphSummary {
                circuit_count: self.components.len(),
                operator_counts,
            },
            notes: vec!["resolved from resident model package manifest".to_string()],
        };
        index.validate_index().map_err(|error| {
            VulkanResidentTokenModelPackageError::new(format!(
                "resident package circuit graph is invalid: {error}"
            ))
        })?;

        Ok(ResolvedLoweredExecutionGraph {
            artifact_root: package_root.into(),
            index,
            circuits,
        })
    }

    pub(super) fn to_signal_processor_graph(
        &self,
        package_root: impl Into<PathBuf>,
    ) -> Result<ResolvedLoweredExecutionGraph, VulkanResidentTokenModelPackageError> {
        let full = self.to_resolved_lowered_execution_graph(package_root)?;
        let input_transducer = runtime_transducer_parameter_metadata(
            &full,
            CircuitRuntimeRole::InputTransducer,
        )?;
        let output_transducer = runtime_transducer_parameter_metadata(
            &full,
            CircuitRuntimeRole::OutputTransducer,
        )?;
        let processor_ids = full
            .circuits
            .iter()
            .filter(|artifact| artifact.circuit.runtime_role.is_signal_processor())
            .map(|artifact| artifact.component.id.as_str())
            .collect::<BTreeSet<_>>();
        if processor_ids.is_empty() {
            return Err(VulkanResidentTokenModelPackageError::new(
                "resident package execution_graph contains no signal processors",
            ));
        }
        let circuits = full
            .circuits
            .iter()
            .filter(|artifact| processor_ids.contains(artifact.component.id.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        let circuit_refs = circuits
            .iter()
            .map(|artifact| artifact.component.clone())
            .collect::<Vec<_>>();
        let edges = full
            .index
            .graph
            .edges
            .iter()
            .filter(|edge| {
                edge.connection.is_instantaneous()
                    && processor_ids.contains(edge.source.component_id.as_str())
                    && processor_ids.contains(edge.destination.component_id.as_str())
            })
            .cloned()
            .collect::<Vec<_>>();
        let external_inputs = execution_boundary_inputs(&full, &processor_ids);
        let public_outputs = execution_boundary_outputs(&full, &processor_ids);
        let mut operator_counts = BTreeMap::new();
        for artifact in &circuits {
            *operator_counts
                .entry(artifact.component.operator_type.clone())
                .or_insert(0) += 1;
        }
        let mut index = full.index.clone();
        if let Some(input_transducer) = input_transducer {
            index.graph.input_transducer = input_transducer;
        }
        if let Some(output_transducer) = output_transducer {
            index.graph.output_transducer = output_transducer;
        }
        index.graph.circuits = circuit_refs;
        index.graph.edges = edges;
        index.graph.boundary = StreamCircuitGraphBoundary {
            external_inputs,
            public_outputs,
        };
        index.summary = LoweredExecutionGraphSummary {
            circuit_count: circuits.len(),
            operator_counts,
        };
        index.validate_index().map_err(|error| {
            VulkanResidentTokenModelPackageError::new(format!(
                "resident package signal-processor graph is invalid: {error}"
            ))
        })?;
        Ok(ResolvedLoweredExecutionGraph {
            artifact_root: full.artifact_root,
            index,
            circuits,
        })
    }

    pub(super) fn signal_processor_placement(
        &self,
        placement: &StreamCircuitPlacementSpec,
    ) -> StreamCircuitPlacementSpec {
        let processor_ids = self
            .components
            .iter()
            .filter(|component| component.runtime_role.is_signal_processor())
            .map(|component| component.component_id.as_str())
            .collect::<BTreeSet<_>>();
        StreamCircuitPlacementSpec {
            schema: placement.schema.clone(),
            default_device_id: placement.default_device_id.clone(),
            node_devices: placement
                .node_devices
                .iter()
                .filter(|(component_id, _)| processor_ids.contains(component_id.as_str()))
                .map(|(component_id, device_id)| (component_id.clone(), device_id.clone()))
                .collect(),
            component_shard_devices: placement
                .component_shard_devices
                .iter()
                .filter(|(component_id, _)| processor_ids.contains(component_id.as_str()))
                .map(|(component_id, device_ids)| (component_id.clone(), device_ids.clone()))
                .collect(),
        }
    }

    pub(super) fn signal_processor_endpoint_component_ids(
        &self,
    ) -> Result<(String, String), VulkanResidentTokenModelPackageError> {
        let graph = self.to_signal_processor_graph(PathBuf::from("."))?;
        let [input] = graph.index.graph.boundary.external_inputs.as_slice() else {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "Vulkan generation currently requires exactly one signal-processor input boundary, found {}",
                graph.index.graph.boundary.external_inputs.len()
            )));
        };
        let [output] = graph.index.graph.boundary.public_outputs.as_slice() else {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "Vulkan generation currently requires exactly one signal-processor output boundary, found {}",
                graph.index.graph.boundary.public_outputs.len()
            )));
        };
        Ok((
            input.endpoint.component_id.clone(),
            output.endpoint.component_id.clone(),
        ))
    }

    pub(super) fn signal_processor_device_ids(
        &self,
        placement: &StreamCircuitPlacementSpec,
    ) -> Vec<String> {
        let processor_ids = self
            .components
            .iter()
            .filter(|component| component.runtime_role.is_signal_processor())
            .map(|component| component.component_id.as_str())
            .collect::<BTreeSet<_>>();
        processor_ids
            .iter()
            .map(|component_id| placement.device_for_component(component_id).to_string())
            .chain(
                placement
                    .component_shard_devices
                    .iter()
                    .filter(|(component_id, _)| processor_ids.contains(component_id.as_str()))
                    .flat_map(|(_, device_ids)| device_ids.iter().cloned()),
            )
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    pub(super) fn signal_processor_owner_device_ids(
        &self,
        placement: &StreamCircuitPlacementSpec,
    ) -> Vec<String> {
        self.components
            .iter()
            .filter(|component| component.runtime_role.is_signal_processor())
            .map(|component| {
                placement
                    .device_for_component(&component.component_id)
                    .to_string()
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }
}

fn runtime_transducer_parameter_metadata(
    graph: &ResolvedLoweredExecutionGraph,
    role: CircuitRuntimeRole,
) -> Result<Option<Value>, VulkanResidentTokenModelPackageError> {
    let matching = graph
        .circuits
        .iter()
        .filter(|artifact| artifact.component.runtime_role == role)
        .collect::<Vec<_>>();
    let artifact = match matching.as_slice() {
        [] => return Ok(None),
        [artifact] => *artifact,
        _ => {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "resident package has multiple {role:?} components: {}",
            matching.len()
        )));
        }
    };
    let params = artifact
        .params
        .refs
        .iter()
        .map(|(parameter_id, parameter)| {
            (
                parameter_id.clone(),
                serde_json::json!({"tensor": parameter.tensor}),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    Ok(Some(serde_json::json!({
        "id": artifact.component.id,
        "params": params,
    })))
}

pub(crate) fn execution_boundary_inputs(
    graph: &ResolvedLoweredExecutionGraph,
    processor_ids: &BTreeSet<&str>,
) -> Vec<crate::stream_circuit::StreamCircuitGraphBoundaryPort> {
    let mut ports = graph
        .index
        .graph
        .boundary
        .external_inputs
        .iter()
        .filter(|port| processor_ids.contains(port.endpoint.component_id.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    ports.extend(
        graph
            .index
            .graph
            .edges
            .iter()
            .filter(|edge| {
                edge.connection.is_instantaneous()
                    && !processor_ids.contains(edge.source.component_id.as_str())
                    && processor_ids.contains(edge.destination.component_id.as_str())
            })
            .map(
                |edge| crate::stream_circuit::StreamCircuitGraphBoundaryPort {
                    id: format!("{}_input", edge.id),
                    endpoint: edge.destination.clone(),
                    source_tap: None,
                },
            ),
    );
    ports
}

pub(crate) fn execution_boundary_outputs(
    graph: &ResolvedLoweredExecutionGraph,
    processor_ids: &BTreeSet<&str>,
) -> Vec<crate::stream_circuit::StreamCircuitGraphBoundaryPort> {
    let mut ports = graph
        .index
        .graph
        .boundary
        .public_outputs
        .iter()
        .filter(|port| processor_ids.contains(port.endpoint.component_id.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    ports.extend(
        graph
            .index
            .graph
            .edges
            .iter()
            .filter(|edge| {
                edge.connection.is_instantaneous()
                    && processor_ids.contains(edge.source.component_id.as_str())
                    && !processor_ids.contains(edge.destination.component_id.as_str())
            })
            .map(
                |edge| crate::stream_circuit::StreamCircuitGraphBoundaryPort {
                    id: format!("{}_output", edge.id),
                    endpoint: edge.source.clone(),
                    source_tap: None,
                },
            ),
    );
    ports
}
