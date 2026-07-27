#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct VulkanRuntimeComponentOverlay {
    schema: String,
    source_component_id: String,
    component: VulkanResidentPackageComponentCircuit,
    execution: VulkanResidentComponentExecutionSpec,
}

impl crate::RuntimeSelectionRequest {
    pub fn from_vulkan_runtime_model(
        runtime_model: &VulkanResidentRuntimeModel,
        profiles_by_logical_device: &BTreeMap<
            String,
            crate::HardwareProcessProfile,
        >,
        execution: crate::RuntimeExecutionEnvelope,
        exact_baseline_compatible: bool,
    ) -> Result<Self, VulkanResidentTokenModelPackageError> {
        let source_roles = runtime_model
            .package
            .circuit_graph
            .components
            .iter()
            .map(|component| {
                (
                    component.component_id.as_str(),
                    component.runtime_role,
                )
            })
            .collect::<BTreeMap<_, _>>();
        let mut instances = runtime_model
            .runtime_graph
            .instances
            .iter()
            .filter(|instance| {
                source_roles
                    .get(instance.source_component_id.as_str())
                    .is_some_and(|role| role.is_signal_processor())
            })
            .map(|instance| {
                let mut logical_device_ids = vec![
                    runtime_model
                        .placement
                        .device_for_component(&instance.instance_id)
                        .to_string(),
                ];
                if let Some(shards) = runtime_model
                    .placement
                    .component_shard_devices
                    .get(&instance.instance_id)
                {
                    logical_device_ids.extend(shards.iter().cloned());
                }
                logical_device_ids.sort();
                logical_device_ids.dedup();
                crate::RuntimeSelectionInstance {
                    instance_id: instance.instance_id.clone(),
                    source_component_id: instance
                        .source_component_id
                        .clone(),
                    logical_device_ids,
                }
            })
            .collect::<Vec<_>>();
        instances.sort_by(|left, right| {
            left.instance_id.cmp(&right.instance_id)
        });
        let included_instances = instances
            .iter()
            .map(|instance| instance.instance_id.as_str())
            .collect::<BTreeSet<_>>();
        let mut edges = runtime_model
            .runtime_graph
            .effective_edges()
            .map_err(|error| {
                VulkanResidentTokenModelPackageError::new(
                    error.to_string(),
                )
            })?
            .into_iter()
            .filter(|edge| {
                included_instances
                    .contains(edge.source.component_id.as_str())
                    && included_instances
                        .contains(edge.destination.component_id.as_str())
            })
            .map(|edge| crate::RuntimeSelectionEdge {
                source_instance_id: edge.source.component_id,
                destination_instance_id: edge.destination.component_id,
            })
            .collect::<Vec<_>>();
        edges.sort_by(|left, right| {
            (
                left.source_instance_id.as_str(),
                left.destination_instance_id.as_str(),
            )
                .cmp(&(
                    right.source_instance_id.as_str(),
                    right.destination_instance_id.as_str(),
                ))
        });
        let devices = profiles_by_logical_device
            .iter()
            .map(|(logical_device_id, profile)| {
                crate::RuntimeSelectionDevice {
                    logical_device_id: logical_device_id.clone(),
                    physical_device_id: profile
                        .hardware_identity
                        .stable_device_id
                        .clone(),
                    profile: profile.clone(),
                }
            })
            .collect();
        Ok(Self {
            execution,
            devices,
            instances,
            edges,
            exact_baseline_compatible,
        })
    }
}

impl VulkanResidentRuntimeModel {
    pub fn select_runtime_implementations(
        &self,
        package_root: impl AsRef<Path>,
        profiles_by_logical_device: &BTreeMap<
            String,
            crate::HardwareProcessProfile,
        >,
        execution: crate::RuntimeExecutionEnvelope,
    ) -> io::Result<crate::RuntimeImplementationSelectionReport> {
        let catalog = self
            .package
            .implementation_catalog(package_root)?;
        let request =
            crate::RuntimeSelectionRequest::from_vulkan_runtime_model(
                self,
                profiles_by_logical_device,
                execution,
                true,
            )
            .map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    error.to_string(),
                )
            })?;
        catalog.select(&request)
    }

    pub fn select_and_apply_runtime_implementations(
        self,
        package_root: impl AsRef<Path>,
        profiles_by_logical_device: &BTreeMap<
            String,
            crate::HardwareProcessProfile,
        >,
        execution: crate::RuntimeExecutionEnvelope,
    ) -> Result<
        (
            Self,
            crate::RuntimeImplementationSelectionReport,
        ),
        VulkanResidentTokenModelPackageError,
    > {
        let package_root = package_root
            .as_ref()
            .canonicalize()
            .map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to resolve runtime package root: {error}"
                ))
            })?;
        let catalog = self
            .package
            .implementation_catalog(&package_root)
            .map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to load runtime implementation catalog: {error}"
                ))
            })?;
        let request = crate::RuntimeSelectionRequest::from_vulkan_runtime_model(
            &self,
            profiles_by_logical_device,
            execution,
            true,
        )?;
        let selection = catalog.select(&request).map_err(|error| {
            VulkanResidentTokenModelPackageError::new(format!(
                "failed to select runtime implementations: {error}"
            ))
        })?;
        let mounted = self.apply_runtime_implementation_catalog_selection(
            &package_root,
            &catalog,
            selection.clone(),
        )?;
        Ok((mounted, selection))
    }

    fn apply_runtime_implementation_catalog_selection(
        mut self,
        package_root: &Path,
        catalog: &crate::RuntimeImplementationCatalog,
        selection: crate::RuntimeImplementationSelectionReport,
    ) -> Result<Self, VulkanResidentTokenModelPackageError> {
        if selection.package_id != self.package.package_id {
            return Err(VulkanResidentTokenModelPackageError::new(
                "runtime implementation selection belongs to another package",
            ));
        }
        let expected_totals = (
            checked_selection_total(
                selection
                    .selected
                    .iter()
                    .map(|item| item.estimated_saved_ns),
                "estimated saved time",
            )?,
            checked_selection_total(
                selection
                    .selected
                    .iter()
                    .map(|item| item.conversion_ns),
                "conversion time",
            )?,
            checked_selection_total(
                selection
                    .selected
                    .iter()
                    .map(|item| item.conversion_bytes),
                "conversion bytes",
            )?,
            checked_selection_total(
                selection
                    .selected
                    .iter()
                    .map(|item| item.boundary_count),
                "representation boundary count",
            )?,
        );
        if expected_totals
            != (
                selection.total_estimated_saved_ns,
                selection.total_conversion_ns,
                selection.total_conversion_bytes,
                selection.total_boundary_count,
            )
        {
            return Err(VulkanResidentTokenModelPackageError::new(
                "runtime implementation selection totals are inconsistent",
            ));
        }
        let implementations = catalog
            .implementations
            .iter()
            .map(|loaded| {
                (
                    loaded.implementation.implementation_id.as_str(),
                    loaded,
                )
            })
            .collect::<BTreeMap<_, _>>();
        let runtime_instances = self
            .runtime_graph
            .instances
            .iter()
            .map(|instance| {
                (instance.instance_id.clone(), instance.clone())
            })
            .collect::<BTreeMap<_, _>>();
        let source_components = self
            .package
            .circuit_graph
            .components
            .iter()
            .map(|component| {
                (component.component_id.clone(), component.clone())
            })
            .collect::<BTreeMap<_, _>>();
        let effective_edges = self
            .runtime_graph
            .effective_edges()
            .map_err(|error| {
                VulkanResidentTokenModelPackageError::new(error.to_string())
            })?;
        let mut mounted_instances = BTreeSet::new();
        let mut loaded_tensor_fragments = BTreeSet::new();

        for selected in &selection.selected {
            let loaded = implementations
                .get(selected.implementation_id.as_str())
                .ok_or_else(|| {
                    VulkanResidentTokenModelPackageError::new(format!(
                        "selected implementation {:?} is absent from the package",
                        selected.implementation_id
                    ))
                })?;
            validate_selected_implementation(selected, loaded)?;
            let island_instances = selected
                .instance_ids
                .iter()
                .map(|instance_id| {
                    let instance = runtime_instances
                        .get(instance_id)
                        .cloned()
                        .ok_or_else(|| {
                            VulkanResidentTokenModelPackageError::new(
                                format!(
                                    "selected runtime instance {instance_id:?} does not exist"
                                ),
                            )
                        })?;
                    if !mounted_instances.insert(instance_id.clone()) {
                        return Err(
                            VulkanResidentTokenModelPackageError::new(
                                format!(
                                    "runtime instance {instance_id:?} has overlapping selected implementations"
                                ),
                            ),
                        );
                    }
                    Ok((instance_id.as_str(), instance))
                })
                .collect::<Result<BTreeMap<_, _>, _>>()?;
            let island_ids = island_instances
                .keys()
                .copied()
                .collect::<BTreeSet<_>>();

            for replacement in &loaded.mount_plan.component_replacements {
                let matching_instances = island_instances
                    .values()
                    .filter(|instance| {
                        instance.source_component_id
                            == replacement.source_component_id
                    })
                    .collect::<Vec<_>>();
                if matching_instances.len() != 1 {
                    return Err(
                        VulkanResidentTokenModelPackageError::new(format!(
                            "implementation {:?} does not map source component {:?} exactly once in runtime island {:?}",
                            selected.implementation_id,
                            replacement.source_component_id,
                            selected.instance_ids,
                        )),
                    );
                }
                let source = source_components
                    .get(&replacement.source_component_id)
                    .cloned()
                    .ok_or_else(|| {
                        VulkanResidentTokenModelPackageError::new(format!(
                            "implementation references unknown source component {:?}",
                            replacement.source_component_id
                        ))
                    })?;
                let overlay_path = contained_candidate_artifact(
                    &loaded.candidate_root,
                    &replacement.overlay_ref,
                    "runtime component overlay",
                )?;
                let bytes = fs::read(&overlay_path).map_err(|error| {
                    VulkanResidentTokenModelPackageError::new(format!(
                        "failed to read runtime component overlay {overlay_path:?}: {error}"
                    ))
                })?;
                let mut overlay: VulkanRuntimeComponentOverlay =
                    serde_json::from_slice(&bytes).map_err(|error| {
                        VulkanResidentTokenModelPackageError::new(format!(
                            "invalid runtime component overlay {overlay_path:?}: {error}"
                        ))
                    })?;
                validate_runtime_component_overlay(
                    &overlay,
                    &source,
                    matching_instances[0].instance_id.as_str(),
                    &island_ids,
                    &effective_edges,
                    &self.runtime_graph.boundary,
                )?;
                rebase_overlay_shader_paths(
                    &mut overlay.execution,
                    &loaded.candidate_root,
                )?;
                mount_runtime_component_overlay(
                    &mut self,
                    matching_instances[0].instance_id.as_str(),
                    overlay,
                )?;
            }
            for reference in &loaded.mount_plan.tensor_index_refs {
                let index_path = contained_candidate_artifact(
                    &loaded.candidate_root,
                    reference,
                    "runtime tensor-index fragment",
                )?;
                if loaded_tensor_fragments.insert(index_path.clone()) {
                    self.tensor_index_fragments.push(
                        VulkanRuntimeTensorIndexFragment {
                            index_path,
                            candidate_root: loaded
                                .candidate_root
                                .clone(),
                        },
                    );
                }
            }
        }
        let source_roles = self
            .package
            .circuit_graph
            .components
            .iter()
            .map(|component| {
                (
                    component.component_id.as_str(),
                    component.runtime_role,
                )
            })
            .collect::<BTreeMap<_, _>>();
        let mut expected_exact_instances = self
            .runtime_graph
            .instances
            .iter()
            .filter(|instance| {
                source_roles
                    .get(instance.source_component_id.as_str())
                    .is_some_and(|role| role.is_signal_processor())
            })
            .map(|instance| instance.instance_id.clone())
            .filter(|instance_id| !mounted_instances.contains(instance_id))
            .collect::<Vec<_>>();
        expected_exact_instances.sort();
        if selection.exact_instance_ids != expected_exact_instances {
            return Err(VulkanResidentTokenModelPackageError::new(
                "runtime implementation selection exact coverage is inconsistent",
            ));
        }
        self.tensor_index_fragments.sort_by(|left, right| {
            left.index_path.cmp(&right.index_path)
        });
        let graph = self
            .circuit_graph
            .to_resolved_lowered_execution_graph(PathBuf::from("."))?;
        validate_component_executions_against_graph(
            &self.package.package_id,
            &self.component_executions,
            &graph,
        )?;
        validate_generation_execution_contract(&self.package, &self.circuit_graph)?;
        self.implementation_selection = Some(selection);
        self.load_runtime_tensor_index(package_root)?;
        Ok(self)
    }

    pub fn load_runtime_tensor_index(
        &self,
        package_root: impl AsRef<Path>,
    ) -> Result<TensorIndex, VulkanResidentTokenModelPackageError> {
        let package_root = package_root.as_ref();
        let tensor_index_path = resolve_resident_model_package_path(
            package_root,
            &self.package.tensor_index_path,
        );
        let mut tensor_index =
            TensorIndex::from_package_json_file(&tensor_index_path).map_err(
                |error| {
                    VulkanResidentTokenModelPackageError::new(format!(
                        "failed to load tensor index {tensor_index_path:?}: {error}"
                    ))
                },
            )?;
        for fragment in &self.tensor_index_fragments {
            let loaded = TensorIndex::from_package_fragment_json_file(
                &fragment.index_path,
                &fragment.candidate_root,
            )
            .map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to load runtime tensor-index fragment {:?}: {error}",
                    fragment.index_path
                ))
            })?;
            tensor_index.merge_fragment(loaded).map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to merge runtime tensor-index fragment {:?}: {error}",
                    fragment.index_path
                ))
            })?;
        }
        Ok(tensor_index)
    }
}

fn checked_selection_total(
    mut values: impl Iterator<Item = u64>,
    label: &str,
) -> Result<u64, VulkanResidentTokenModelPackageError> {
    values.try_fold(0u64, |total, value| {
        total.checked_add(value).ok_or_else(|| {
            VulkanResidentTokenModelPackageError::new(format!(
                "runtime implementation selection {label} overflows"
            ))
        })
    })
}

fn validate_selected_implementation(
    selected: &crate::RuntimeSelectedImplementation,
    loaded: &crate::LoadedRuntimeImplementation,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    if selected.candidate_id != loaded.implementation.candidate_id
        || selected.scope_ids != loaded.implementation.scope_ids
        || selected.predicate != loaded.implementation.runtime_predicate
        || selected.mount_adapter_id != loaded.mount_plan.adapter_id
        || selected.representation != loaded.implementation.representation
        || selected.provenance != loaded.implementation.provenance
        || selected.benchmark_id
            != loaded.implementation.comparison.benchmark_id
        || selected.validation_id
            != loaded.implementation.comparison.validation_id
        || selected.validation_status != "passed"
        || selected.decision_reason != loaded.implementation.decision_reason
        || loaded.mount_plan.adapter_id
            != crate::VULKAN_STREAM_CIRCUIT_OVERLAY_ADAPTER
    {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "selected implementation {:?} does not match its verified package contract",
            selected.implementation_id
        )));
    }
    Ok(())
}

fn contained_candidate_artifact(
    candidate_root: &Path,
    reference: &str,
    label: &str,
) -> Result<PathBuf, VulkanResidentTokenModelPackageError> {
    let relative = Path::new(reference);
    if reference.is_empty()
        || relative.is_absolute()
        || relative.components().any(|component| {
            !matches!(component, std::path::Component::Normal(_))
        })
        || relative.to_string_lossy() != reference
    {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "{label} reference is not a canonical candidate-relative path"
        )));
    }
    let path = candidate_root
        .join(relative)
        .canonicalize()
        .map_err(|error| {
            VulkanResidentTokenModelPackageError::new(format!(
                "{label} is missing or unreadable: {error}"
            ))
        })?;
    if !path.starts_with(candidate_root) || !path.is_file() {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "{label} escapes its candidate bundle"
        )));
    }
    Ok(path)
}

fn validate_runtime_component_overlay(
    overlay: &VulkanRuntimeComponentOverlay,
    source: &VulkanResidentPackageComponentCircuit,
    runtime_instance_id: &str,
    island_instance_ids: &BTreeSet<&str>,
    effective_edges: &[crate::stream_circuit::StreamCircuitGraphEdge],
    graph_boundary: &StreamCircuitGraphBoundary,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    if overlay.schema != crate::VULKAN_COMPONENT_OVERLAY_SCHEMA
        || overlay.source_component_id != source.component_id
        || overlay.component.component_id != source.component_id
        || overlay.component.circuit.source.component_id
            != source.component_id
        || overlay.execution.component_id != source.component_id
        || overlay.component.operator_type != source.operator_type
        || overlay.execution.operator_type != source.operator_type
        || overlay.component.runtime_role != source.runtime_role
        || overlay.component.circuit.runtime_role != source.runtime_role
    {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "runtime component overlay for {:?} changes its logical source identity",
            source.component_id
        )));
    }
    let source_inputs = source
        .circuit
        .boundary
        .inputs
        .iter()
        .map(|port| (port.id.as_str(), port))
        .collect::<BTreeMap<_, _>>();
    let overlay_inputs = overlay
        .component
        .circuit
        .boundary
        .inputs
        .iter()
        .map(|port| (port.id.as_str(), port))
        .collect::<BTreeMap<_, _>>();
    let source_outputs = source
        .circuit
        .boundary
        .outputs
        .iter()
        .map(|port| (port.id.as_str(), port))
        .collect::<BTreeMap<_, _>>();
    let overlay_outputs = overlay
        .component
        .circuit
        .boundary
        .outputs
        .iter()
        .map(|port| (port.id.as_str(), port))
        .collect::<BTreeMap<_, _>>();
    if source_inputs.keys().ne(overlay_inputs.keys())
        || source_outputs.keys().ne(overlay_outputs.keys())
    {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "runtime component overlay for {:?} changes its logical port identities",
            source.component_id
        )));
    }
    let external_inputs = effective_edges
        .iter()
        .filter(|edge| {
            edge.destination.component_id == runtime_instance_id
                && !island_instance_ids
                    .contains(edge.source.component_id.as_str())
        })
        .map(|edge| edge.destination.port_id.as_str())
        .chain(
            graph_boundary
                .external_inputs
                .iter()
                .filter(|port| {
                    port.endpoint.component_id == runtime_instance_id
                })
                .map(|port| port.endpoint.port_id.as_str()),
        )
        .collect::<BTreeSet<_>>();
    let external_outputs = effective_edges
        .iter()
        .filter(|edge| {
            edge.source.component_id == runtime_instance_id
                && !island_instance_ids
                    .contains(edge.destination.component_id.as_str())
        })
        .map(|edge| edge.source.port_id.as_str())
        .chain(
            graph_boundary
                .public_outputs
                .iter()
                .filter(|port| {
                    port.endpoint.component_id == runtime_instance_id
                })
                .map(|port| port.endpoint.port_id.as_str()),
        )
        .collect::<BTreeSet<_>>();
    for port_id in external_inputs {
        if source_inputs.get(port_id) != overlay_inputs.get(port_id) {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "runtime component overlay for {:?} changes external input port {port_id:?}",
                source.component_id
            )));
        }
    }
    for port_id in external_outputs {
        if source_outputs.get(port_id) != overlay_outputs.get(port_id) {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "runtime component overlay for {:?} changes external output port {port_id:?}",
                source.component_id
            )));
        }
    }
    Ok(())
}

fn rebase_overlay_shader_paths(
    execution: &mut VulkanResidentComponentExecutionSpec,
    candidate_root: &Path,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    for kernel in &mut execution.kernels {
        kernel.shader_path = contained_candidate_artifact(
            candidate_root,
            &kernel.shader_path,
            "runtime implementation shader",
        )?
        .to_string_lossy()
        .into_owned();
        for implementation in &mut kernel.batch_implementations {
            for stage in &mut implementation.stages {
                stage.shader_path = contained_candidate_artifact(
                    candidate_root,
                    &stage.shader_path,
                    "runtime implementation batch shader",
                )?
                .to_string_lossy()
                .into_owned();
            }
        }
    }
    Ok(())
}

fn mount_runtime_component_overlay(
    runtime_model: &mut VulkanResidentRuntimeModel,
    runtime_instance_id: &str,
    mut overlay: VulkanRuntimeComponentOverlay,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    overlay.component.component_id = runtime_instance_id.to_string();
    overlay.component.circuit.source.component_id =
        runtime_instance_id.to_string();
    overlay.execution.component_id = runtime_instance_id.to_string();
    let component = runtime_model
        .circuit_graph
        .components
        .iter_mut()
        .find(|component| component.component_id == runtime_instance_id)
        .ok_or_else(|| {
            VulkanResidentTokenModelPackageError::new(format!(
                "runtime implementation cannot find component {runtime_instance_id:?}"
            ))
        })?;
    *component = overlay.component;
    let execution = runtime_model
        .component_executions
        .iter_mut()
        .find(|execution| execution.component_id == runtime_instance_id)
        .ok_or_else(|| {
            VulkanResidentTokenModelPackageError::new(format!(
                "runtime implementation cannot find execution for {runtime_instance_id:?}"
            ))
        })?;
    *execution = overlay.execution;
    Ok(())
}
