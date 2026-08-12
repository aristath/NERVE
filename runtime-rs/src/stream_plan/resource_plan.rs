#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedParameterResource {
    pub tensor: String,
    pub uses: Vec<PlannedParameterUse>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedParameterUse {
    pub component_id: String,
    pub circuit_id: String,
    pub param_id: String,
    pub role: Option<String>,
    pub layout: String,
    pub storage: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedStateResource {
    pub component_id: String,
    pub circuit_id: String,
    pub state_id: String,
    pub state_type: String,
    pub shape: Option<Vec<usize>>,
    pub elements_per_activation: Option<usize>,
    pub max_dynamic_activations: Option<usize>,
    pub update: Option<String>,
    pub growth: Option<String>,
    pub sharing: Option<String>,
    pub owner: Option<String>,
    pub layout: Option<String>,
    pub source_layout: Option<String>,
    pub dtype: Option<String>,
    pub element_bytes: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedSelectionDomain {
    pub component_id: String,
    pub circuit_id: String,
    pub node_id: String,
    pub domain_id: String,
    pub resource_count: usize,
    pub selection_count_per_activation: usize,
}

pub(crate) fn circuit_dtype_bytes(dtype: &str) -> Result<usize, CircuitPlanError> {
    match dtype {
        "U8" | "I8" | "FP8_E4M3" | "FP8_E5M2" => Ok(1),
        "BF16" | "F16" => Ok(2),
        "F32" | "U32" | "I32" => Ok(4),
        unsupported => Err(CircuitPlanError(format!(
            "unsupported circuit dtype {unsupported:?}"
        ))),
    }
}

fn node_output_element_bytes(
    component_id: &str,
    node: &CircuitNode,
    output_index: usize,
) -> Result<Option<usize>, CircuitPlanError> {
    let Some(raw_widths) = node
        .attrs
        .as_object()
        .and_then(|attrs| attrs.get("output_element_bytes"))
    else {
        return Ok(None);
    };
    let widths = raw_widths.as_array().ok_or_else(|| {
        CircuitPlanError(format!(
            "{component_id} node {} output_element_bytes must be an array",
            node.id
        ))
    })?;
    if widths.len() != node.outputs.len() {
        return Err(CircuitPlanError(format!(
            "{component_id} node {} declares {} output byte widths for {} outputs",
            node.id,
            widths.len(),
            node.outputs.len()
        )));
    }
    let width = widths[output_index].as_u64().ok_or_else(|| {
        CircuitPlanError(format!(
            "{component_id} node {} output byte width {} must be a positive integer",
            node.id, output_index
        ))
    })?;
    let width = usize::try_from(width).map_err(|_| {
        CircuitPlanError(format!(
            "{component_id} node {} output byte width {} exceeds usize",
            node.id, output_index
        ))
    })?;
    if width == 0 {
        return Err(CircuitPlanError(format!(
            "{component_id} node {} output byte width {} must be positive",
            node.id, output_index
        )));
    }
    Ok(Some(width))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedActivationSlotBank {
    pub component_id: String,
    pub circuit_id: String,
    pub temporary_signal_count: usize,
    pub state_view_signal_count: usize,
    pub slot_count: usize,
    pub slots: Vec<PlannedActivationSlot>,
    pub assignments: Vec<SignalSlotAssignment>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedActivationSlot {
    pub slot: usize,
    pub signal_ids: Vec<String>,
    pub max_elements: Option<usize>,
    pub max_bytes: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CircuitActivationPlan {
    pub component_id: String,
    pub circuit_id: String,
    pub input_ports: Vec<PlannedPort>,
    pub output_ports: Vec<PlannedPort>,
    pub state_ports: Vec<PlannedStatePort>,
    pub parameter_refs: Vec<String>,
    pub nodes: Vec<PlannedNode>,
    pub signals: BTreeMap<String, PlannedSignal>,
    pub temporary_signals: Vec<String>,
    pub state_view_signals: Vec<String>,
}

impl CircuitActivationPlan {
    pub fn from_artifact(artifact: &ResolvedCircuitArtifact) -> Result<Self, CircuitPlanError> {
        Self::from_circuit(&artifact.component.id, &artifact.circuit)
    }

    pub fn from_artifact_with_tensor_index(
        artifact: &ResolvedCircuitArtifact,
        tensor_index: &TensorIndex,
    ) -> Result<Self, CircuitPlanError> {
        Self::from_circuit_with_tensor_index(&artifact.component.id, &artifact.circuit, tensor_index)
    }

    pub fn from_circuit(
        component_id: impl Into<String>,
        circuit: &StreamCircuit,
    ) -> Result<Self, CircuitPlanError> {
        Self::from_circuit_with_optional_tensor_index(component_id, circuit, None)
    }

    pub fn from_circuit_with_tensor_index(
        component_id: impl Into<String>,
        circuit: &StreamCircuit,
        tensor_index: &TensorIndex,
    ) -> Result<Self, CircuitPlanError> {
        Self::from_circuit_with_optional_tensor_index(component_id, circuit, Some(tensor_index))
    }

    fn from_circuit_with_optional_tensor_index(
        component_id: impl Into<String>,
        circuit: &StreamCircuit,
        tensor_index: Option<&TensorIndex>,
    ) -> Result<Self, CircuitPlanError> {
        let component_id = component_id.into();
        let state_ids: BTreeSet<_> = circuit.state_ports.iter().map(|state| &state.id).collect();
        let param_ids: BTreeSet<_> = circuit.parameters.refs.keys().collect();
        let boundary_output_sources: BTreeSet<_> = circuit
            .boundary
            .outputs
            .iter()
            .map(|port| port.source.as_ref().unwrap_or(&port.id).clone())
            .collect();

        let mut available = BTreeSet::new();
        let mut signals = BTreeMap::new();
        for input in &circuit.boundary.inputs {
            available.insert(input.id.clone());
            signals.insert(
                input.id.clone(),
                PlannedSignal {
                    id: input.id.clone(),
                    producer: SignalProducer::BoundaryInput,
                    consumers: Vec::new(),
                    shape: Some(input.shape.clone()),
                    element_bytes: input
                        .dtype
                        .as_deref()
                        .map(circuit_dtype_bytes)
                        .transpose()?,
                    storage: SignalStorage::Boundary,
                    is_boundary_output: false,
                },
            );
        }
        for control in &circuit.boundary.controls {
            if !available.insert(control.id.clone()) {
                return Err(CircuitPlanError(format!(
                    "{} runtime control {:?} collides with another boundary signal",
                    component_id, control.id
                )));
            }
            signals.insert(
                control.id.clone(),
                PlannedSignal {
                    id: control.id.clone(),
                    producer: SignalProducer::RuntimeControl {
                        runtime_source: control.runtime_source.clone(),
                    },
                    consumers: Vec::new(),
                    shape: Some(control.shape.clone()),
                    element_bytes: Some(4),
                    storage: SignalStorage::RuntimeControl,
                    is_boundary_output: false,
                },
            );
        }
        for state in &circuit.state_ports {
            available.insert(state.id.clone());
            signals.insert(
                state.id.clone(),
                PlannedSignal {
                    id: state.id.clone(),
                    producer: SignalProducer::StatePort,
                    consumers: Vec::new(),
                    shape: state.shape.clone(),
                    element_bytes: None,
                    storage: SignalStorage::State,
                    is_boundary_output: false,
                },
            );
        }

        let mut planned_nodes = Vec::with_capacity(circuit.nodes.len());
        for (index, node) in circuit.nodes.iter().enumerate() {
            validate_node_dependencies(&component_id, node, &available, &state_ids, &param_ids)?;
            let output_shapes = infer_node_output_shapes(
                &component_id,
                node,
                &signals,
                &circuit.parameters.refs,
                tensor_index,
            )?;

            for input in &node.inputs {
                let signal = signals.get_mut(input).ok_or_else(|| {
                    CircuitPlanError(format!(
                        "{} node {} input {:?} is not in the planned signal table",
                        component_id, node.id, input
                    ))
                })?;
                signal.consumers.push(node.id.clone());
            }

            for (output_index, output) in node.outputs.iter().enumerate() {
                if available.contains(output) {
                    return Err(CircuitPlanError(format!(
                        "{} node {} output {:?} is already available",
                        component_id, node.id, output
                    )));
                }
                available.insert(output.clone());
                signals.insert(
                    output.clone(),
                    PlannedSignal {
                        id: output.clone(),
                        producer: SignalProducer::Node {
                            node_id: node.id.clone(),
                        },
                        consumers: Vec::new(),
                        shape: output_shapes.get(output_index).cloned().unwrap_or(None),
                        element_bytes: node_output_element_bytes(
                            &component_id,
                            node,
                            output_index,
                        )?,
                        storage: node_output_storage(node),
                        is_boundary_output: boundary_output_sources.contains(output),
                    },
                );
            }

        planned_nodes.push(PlannedNode::from_node(&component_id, index, node)?);
        }

        for output in &circuit.boundary.outputs {
            let source = output.source.as_ref().unwrap_or(&output.id);
            let signal = signals.get_mut(source).ok_or_else(|| {
                CircuitPlanError(format!(
                    "{} boundary output {} source {:?} is not planned",
                    component_id, output.id, source
                ))
            })?;
            signal.is_boundary_output = true;
            signal
                .consumers
                .push(format!("boundary.output:{}", output.id));
        }

        let temporary_signals = signals
            .values()
            .filter(|signal| {
                matches!(signal.producer, SignalProducer::Node { .. })
                    && signal.storage == SignalStorage::Activation
                    && !signal.is_boundary_output
            })
            .map(|signal| signal.id.clone())
            .collect();
        let state_view_signals = signals
            .values()
            .filter(|signal| {
                matches!(signal.producer, SignalProducer::Node { .. })
                    && signal.storage == SignalStorage::StateView
            })
            .map(|signal| signal.id.clone())
            .collect();

        Ok(Self {
            component_id,
            circuit_id: circuit.id.clone(),
            input_ports: circuit
                .boundary
                .inputs
                .iter()
                .map(PlannedPort::from_port)
                .collect::<Result<Vec<_>, _>>()?,
            output_ports: circuit
                .boundary
                .outputs
                .iter()
                .map(PlannedPort::from_port)
                .collect::<Result<Vec<_>, _>>()?,
            state_ports: circuit
                .state_ports
                .iter()
                .map(PlannedStatePort::from_state_port)
                .collect(),
            parameter_refs: circuit.parameters.refs.keys().cloned().collect(),
            nodes: planned_nodes,
            signals,
            temporary_signals,
            state_view_signals,
        })
    }

    pub fn produced_signal_count(&self) -> usize {
        self.signals
            .values()
            .filter(|signal| matches!(signal.producer, SignalProducer::Node { .. }))
            .count()
    }

    pub fn signal(&self, signal_id: &str) -> Option<&PlannedSignal> {
        self.signals.get(signal_id)
    }

    pub fn activation_frame_plan(&self) -> ActivationFramePlan {
        let liveness = self.signal_liveness();
        let mut slot_free_after: Vec<usize> = Vec::new();
        let mut assignments = Vec::with_capacity(liveness.len());

        for live in &liveness {
            let reusable_slot = slot_free_after
                .iter()
                .position(|free_after| *free_after < live.produced_at);
            let slot = if let Some(slot) = reusable_slot {
                slot_free_after[slot] = live.last_consumed_at;
                slot
            } else {
                let slot = slot_free_after.len();
                slot_free_after.push(live.last_consumed_at);
                slot
            };
            assignments.push(SignalSlotAssignment {
                signal_id: live.signal_id.clone(),
                slot,
                produced_at: live.produced_at,
                last_consumed_at: live.last_consumed_at,
            });
        }

        ActivationFramePlan {
            liveness,
            assignments,
            slot_count: slot_free_after.len(),
        }
    }

    fn signal_liveness(&self) -> Vec<SignalLiveness> {
        let temporary_signals: BTreeSet<_> = self.temporary_signals.iter().cloned().collect();
        let node_indices: BTreeMap<_, _> = self
            .nodes
            .iter()
            .map(|node| (node.id.as_str(), node.index))
            .collect();
        let mut liveness = Vec::new();

        for node in &self.nodes {
            for output in &node.outputs {
                if !temporary_signals.contains(output) {
                    continue;
                }
                let signal = self
                    .signals
                    .get(output)
                    .expect("temporary signal is in the planned signal table");
                let consumer_indices: Vec<_> = signal
                    .consumers
                    .iter()
                    .map(|consumer| {
                        if consumer.starts_with("boundary.output:") {
                            self.nodes.len()
                        } else {
                            *node_indices.get(consumer.as_str()).unwrap_or_else(|| {
                                panic!("unknown consumer {consumer:?} for signal {output:?}")
                            })
                        }
                    })
                    .collect();
                let last_consumed_at = consumer_indices.iter().copied().max().unwrap_or(node.index);
                liveness.push(SignalLiveness {
                    signal_id: output.clone(),
                    produced_by: node.id.clone(),
                    produced_at: node.index,
                    consumers: signal.consumers.clone(),
                    consumer_indices,
                    last_consumed_at,
                });
            }
        }

        liveness
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedPort {
    pub id: String,
    pub signal: String,
    pub shape: Vec<usize>,
    pub dtype: Option<String>,
    pub element_bytes: Option<usize>,
    pub source: Option<String>,
}

impl PlannedPort {
    fn from_port(port: &CircuitPort) -> Result<Self, CircuitPlanError> {
        Ok(Self {
            id: port.id.clone(),
            signal: port.signal.clone(),
            shape: port.shape.clone(),
            dtype: port.dtype.clone(),
            element_bytes: port
                .dtype
                .as_deref()
                .map(circuit_dtype_bytes)
                .transpose()?,
            source: port.source.clone(),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedStatePort {
    pub id: String,
    pub state_type: String,
    pub shape: Option<Vec<usize>>,
    pub elements_per_activation: Option<usize>,
    pub max_dynamic_activations: Option<usize>,
    pub dtype: Option<String>,
}

impl PlannedStatePort {
    fn from_state_port(state: &StatePort) -> Self {
        Self {
            id: state.id.clone(),
            state_type: state.state_type.clone(),
            shape: state.shape.clone(),
            elements_per_activation: state.elements_per_activation(),
            max_dynamic_activations: state.dynamic_activation_capacity(),
            dtype: state.dtype.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedNode {
    pub index: usize,
    pub id: String,
    pub op: String,
    pub specialization: String,
    pub stream_control_binding: Option<u32>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub params: Vec<String>,
    pub state_reads: Vec<String>,
    pub state_writes: Vec<String>,
    pub selection_domain: Option<PlannedNodeSelectionDomain>,
    pub selected_parameter_accesses: Vec<PlannedSelectedParameterAccess>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedNodeSelectionDomain {
    pub domain_id: String,
    pub resource_count: usize,
    pub selection_signal: String,
    pub encoding: PlannedSelectionEncoding,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlannedSelectionElementType {
    U32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedSelectionEncoding {
    pub element_type: PlannedSelectionElementType,
    pub selection_count_per_activation: usize,
    pub index_shift: u32,
    pub index_mask: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedSelectedParameterAccess {
    pub selection_signal: String,
    pub layout: PlannedSelectedParameterLayout,
    pub parameter_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlannedSelectedParameterLayout {
    Partitioned { partition_axis: usize },
    Independent {
        resource_count: usize,
        parameters_per_resource: usize,
    },
}

impl PlannedNode {
    fn from_node(
        component_id: &str,
        index: usize,
        node: &CircuitNode,
    ) -> Result<Self, CircuitPlanError> {
        Ok(Self {
            index,
            id: node.id.clone(),
            op: node.op.clone(),
            specialization: if node.attrs.is_null()
                || node
                    .attrs
                    .as_object()
                    .is_some_and(serde_json::Map::is_empty)
            {
                String::new()
            } else {
                serde_json::to_string(&node.attrs)
                    .expect("circuit node attributes must serialize as JSON")
            },
            stream_control_binding: planned_node_stream_control_binding(
                component_id,
                node,
            )?,
            inputs: node.inputs.clone(),
            outputs: node.outputs.clone(),
            params: node.params.clone(),
            state_reads: node.state_reads.clone(),
            state_writes: node.state_writes.clone(),
            selection_domain: planned_node_selection_domain(component_id, node)?,
            selected_parameter_accesses:
                planned_node_selected_parameter_accesses(component_id, node)?,
        })
    }
}

fn planned_node_stream_control_binding(
    component_id: &str,
    node: &CircuitNode,
) -> Result<Option<u32>, CircuitPlanError> {
    let Some(value) = node
        .attrs
        .as_object()
        .and_then(|attrs| attrs.get("stream_control_binding"))
    else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let binding = value.as_u64().ok_or_else(|| {
        CircuitPlanError(format!(
            "{component_id} node {} stream_control_binding must be a non-negative integer or null",
            node.id
        ))
    })?;
    u32::try_from(binding).map(Some).map_err(|_| {
        CircuitPlanError(format!(
            "{component_id} node {} stream_control_binding exceeds u32",
            node.id
        ))
    })
}

fn planned_node_selection_domain(
    component_id: &str,
    node: &CircuitNode,
) -> Result<Option<PlannedNodeSelectionDomain>, CircuitPlanError> {
    let Some(domain) = node
        .attrs
        .as_object()
        .and_then(|attrs| attrs.get("selection_domain"))
    else {
        return Ok(None);
    };
    let domain = domain.as_object().ok_or_else(|| {
        CircuitPlanError(format!(
            "{component_id} node {} selection_domain must be an object",
            node.id
        ))
    })?;
    if domain.len() != 4
        || !["id", "resource_count", "selection_signal", "encoding"]
            .iter()
            .all(|field| domain.contains_key(*field))
    {
        return Err(CircuitPlanError(format!(
            "{component_id} node {} selection_domain has ambiguous fields",
            node.id
        )));
    }
    let domain_id = domain
        .get("id")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            CircuitPlanError(format!(
                "{component_id} node {} selection_domain.id must be a non-empty string",
                node.id
            ))
        })?
        .to_string();
    let resource_count = domain
        .get("resource_count")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value > 0)
        .ok_or_else(|| {
            CircuitPlanError(format!(
                "{component_id} node {} selection_domain.resource_count must be a positive integer",
                node.id
            ))
        })?;
    let selection_signal = domain
        .get("selection_signal")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .filter(|value| node.outputs.iter().any(|output| output == *value))
        .ok_or_else(|| {
            CircuitPlanError(format!(
                "{component_id} node {} selection_domain.selection_signal must name a node output",
                node.id
            ))
        })?
        .to_string();
    let encoding = domain
        .get("encoding")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            CircuitPlanError(format!(
                "{component_id} node {} selection_domain.encoding must be an object",
                node.id
            ))
        })?;
    if encoding.len() != 4
        || ![
            "element_type",
            "selection_count_per_activation",
            "index_shift",
            "index_mask",
        ]
        .iter()
        .all(|field| encoding.contains_key(*field))
    {
        return Err(CircuitPlanError(format!(
            "{component_id} node {} selection_domain.encoding has ambiguous fields",
            node.id
        )));
    }
    let element_type = match encoding
        .get("element_type")
        .and_then(serde_json::Value::as_str)
    {
        Some("u32") => PlannedSelectionElementType::U32,
        _ => {
            return Err(CircuitPlanError(format!(
                "{component_id} node {} selection_domain.encoding.element_type is unsupported",
                node.id
            )));
        }
    };
    let selection_count_per_activation = encoding
        .get("selection_count_per_activation")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value > 0)
        .ok_or_else(|| {
            CircuitPlanError(format!(
                "{component_id} node {} selection_domain.encoding.selection_count_per_activation must be a positive integer",
                node.id
            ))
        })?;
    let index_shift = encoding
        .get("index_shift")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| *value < u32::BITS)
        .ok_or_else(|| {
            CircuitPlanError(format!(
                "{component_id} node {} selection_domain.encoding.index_shift must be below 32",
                node.id
            ))
        })?;
    let index_mask = encoding
        .get("index_mask")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| {
            *value != 0
                && *value <= u32::MAX >> index_shift
                && (*value == u32::MAX
                    || *value & (*value + 1) == 0)
        })
        .filter(|value| {
            u32::try_from(resource_count - 1)
                .is_ok_and(|maximum| maximum & *value == maximum)
        })
        .ok_or_else(|| {
            CircuitPlanError(format!(
                "{component_id} node {} selection_domain.encoding.index_mask cannot represent its domain",
                node.id
            ))
        })?;
    Ok(Some(PlannedNodeSelectionDomain {
        domain_id,
        resource_count,
        selection_signal,
        encoding: PlannedSelectionEncoding {
            element_type,
            selection_count_per_activation,
            index_shift,
            index_mask,
        },
    }))
}

fn planned_node_selected_parameter_accesses(
    component_id: &str,
    node: &CircuitNode,
) -> Result<Vec<PlannedSelectedParameterAccess>, CircuitPlanError> {
    let Some(accesses) = node
        .attrs
        .as_object()
        .and_then(|attrs| attrs.get("selected_parameter_accesses"))
    else {
        return Ok(Vec::new());
    };
    let accesses = accesses.as_array().ok_or_else(|| {
        CircuitPlanError(format!(
            "{component_id} node {} selected_parameter_accesses must be an array",
            node.id
        ))
    })?;
    let mut planned = Vec::with_capacity(accesses.len());
    let mut seen_signals = std::collections::BTreeSet::new();
    for access in accesses {
        let access = access.as_object().ok_or_else(|| {
            CircuitPlanError(format!(
                "{component_id} node {} selected parameter access must be an object",
                node.id
            ))
        })?;
        let partitioned = access.len() == 3
            && ["selection_signal", "partition_axis", "parameter_ids"]
                .iter()
                .all(|field| access.contains_key(*field));
        let independent = access.len() == 2
            && ["selection_signal", "mapping"]
                .iter()
                .all(|field| access.contains_key(*field));
        if !partitioned && !independent {
            return Err(CircuitPlanError(format!(
                "{component_id} node {} selected parameter access has ambiguous fields",
                node.id
            )));
        }
        let selection_signal = access
            .get("selection_signal")
            .and_then(serde_json::Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .filter(|value| node.inputs.iter().any(|input| input == *value))
            .ok_or_else(|| {
                CircuitPlanError(format!(
                    "{component_id} node {} selected parameter access signal must name a node input",
                    node.id
                ))
            })?
            .to_string();
        if !seen_signals.insert(selection_signal.clone()) {
            return Err(CircuitPlanError(format!(
                "{component_id} node {} repeats a selected parameter access signal",
                node.id
            )));
        }
        let (layout, parameter_ids) = if partitioned {
            let partition_axis = access
                .get("partition_axis")
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| {
                    CircuitPlanError(format!(
                        "{component_id} node {} selected parameter partition axis must be a non-negative integer",
                        node.id
                    ))
                })?;
            let parameter_ids = selected_parameter_ids(
                component_id,
                node,
                access.get("parameter_ids"),
            )?;
            (
                PlannedSelectedParameterLayout::Partitioned { partition_axis },
                parameter_ids,
            )
        } else {
            let mapping = access
                .get("mapping")
                .and_then(serde_json::Value::as_array)
                .filter(|mapping| !mapping.is_empty())
                .ok_or_else(|| {
                    CircuitPlanError(format!(
                        "{component_id} node {} independent selected parameter mapping must be a non-empty array",
                        node.id
                    ))
                })?;
            let mut parameter_ids = Vec::new();
            let mut parameters_per_resource = None;
            let mut seen_parameters = BTreeSet::new();
            for (expected_selector, entry) in mapping.iter().enumerate() {
                let entry = entry.as_object().filter(|entry| {
                    entry.len() == 2
                        && entry.contains_key("selector")
                        && entry.contains_key("parameter_ids")
                });
                let Some(entry) = entry else {
                    return Err(CircuitPlanError(format!(
                        "{component_id} node {} independent selected parameter mapping has ambiguous fields",
                        node.id
                    )));
                };
                if entry
                    .get("selector")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|value| usize::try_from(value).ok())
                    != Some(expected_selector)
                {
                    return Err(CircuitPlanError(format!(
                        "{component_id} node {} independent selected parameter selectors must be contiguous and ordered",
                        node.id
                    )));
                }
                let selected = selected_parameter_ids(
                    component_id,
                    node,
                    entry.get("parameter_ids"),
                )?;
                if parameters_per_resource
                    .replace(selected.len())
                    .is_some_and(|count| count != selected.len())
                    || selected
                        .iter()
                        .any(|parameter| !seen_parameters.insert(parameter.clone()))
                {
                    return Err(CircuitPlanError(format!(
                        "{component_id} node {} independent selected parameter mapping is not rectangular and unique",
                        node.id
                    )));
                }
                parameter_ids.extend(selected);
            }
            (
                PlannedSelectedParameterLayout::Independent {
                    resource_count: mapping.len(),
                    parameters_per_resource: parameters_per_resource.unwrap(),
                },
                parameter_ids,
            )
        };
        planned.push(PlannedSelectedParameterAccess {
            selection_signal,
            layout,
            parameter_ids,
        });
    }
    Ok(planned)
}

fn selected_parameter_ids(
    component_id: &str,
    node: &CircuitNode,
    value: Option<&serde_json::Value>,
) -> Result<Vec<String>, CircuitPlanError> {
    let parameter_ids = value
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            CircuitPlanError(format!(
                "{component_id} node {} selected parameter ids must be an array",
                node.id
            ))
        })?
        .iter()
        .map(|value| {
            value
                .as_str()
                .filter(|value| !value.trim().is_empty())
                .filter(|value| node.params.iter().any(|parameter| parameter == *value))
                .map(str::to_string)
                .ok_or_else(|| {
                    CircuitPlanError(format!(
                        "{component_id} node {} selected parameter id must name a node parameter",
                        node.id
                    ))
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if parameter_ids.is_empty()
        || parameter_ids
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
    {
        return Err(CircuitPlanError(format!(
            "{component_id} node {} selected parameter ids must be non-empty, unique, and sorted",
            node.id
        )));
    }
    Ok(parameter_ids)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedSignal {
    pub id: String,
    pub producer: SignalProducer,
    pub consumers: Vec<String>,
    pub shape: Option<Vec<usize>>,
    pub element_bytes: Option<usize>,
    pub storage: SignalStorage,
    pub is_boundary_output: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SignalProducer {
    BoundaryInput,
    RuntimeControl { runtime_source: String },
    StatePort,
    Node { node_id: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SignalStorage {
    Boundary,
    RuntimeControl,
    State,
    Activation,
    StateView,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActivationFramePlan {
    pub liveness: Vec<SignalLiveness>,
    pub assignments: Vec<SignalSlotAssignment>,
    pub slot_count: usize,
}

impl ActivationFramePlan {
    pub fn slot_for(&self, signal_id: &str) -> Option<usize> {
        self.assignments
            .iter()
            .find(|assignment| assignment.signal_id == signal_id)
            .map(|assignment| assignment.slot)
    }

    pub fn liveness_for(&self, signal_id: &str) -> Option<&SignalLiveness> {
        self.liveness
            .iter()
            .find(|liveness| liveness.signal_id == signal_id)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignalLiveness {
    pub signal_id: String,
    pub produced_by: String,
    pub produced_at: usize,
    pub consumers: Vec<String>,
    pub consumer_indices: Vec<usize>,
    pub last_consumed_at: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignalSlotAssignment {
    pub signal_id: String,
    pub slot: usize,
    pub produced_at: usize,
    pub last_consumed_at: usize,
}

#[cfg(test)]
mod selection_domain_tests {
    use super::*;

    fn valid_selection_domain() -> serde_json::Value {
        serde_json::json!({
            "id": "addressable_resources",
            "resource_count": 256,
            "selection_signal": "selected",
            "encoding": {
                "element_type": "u32",
                "selection_count_per_activation": 8,
                "index_shift": 0,
                "index_mask": 0xffff
            }
        })
    }

    fn node_with_selection_domain(selection_domain: serde_json::Value) -> CircuitNode {
        CircuitNode {
            id: "selector".to_string(),
            op: "top_k".to_string(),
            inputs: Vec::new(),
            outputs: vec!["selected".to_string()],
            params: Vec::new(),
            state_reads: Vec::new(),
            state_writes: Vec::new(),
            attrs: serde_json::json!({
                "selection_domain": selection_domain,
            }),
        }
    }

    #[test]
    fn selection_domain_contract_accepts_a_positive_resource_domain() {
        let node = node_with_selection_domain(valid_selection_domain());

        assert_eq!(
            planned_node_selection_domain("component", &node).unwrap(),
            Some(PlannedNodeSelectionDomain {
                domain_id: "addressable_resources".to_string(),
                resource_count: 256,
                selection_signal: "selected".to_string(),
                encoding: PlannedSelectionEncoding {
                    element_type: PlannedSelectionElementType::U32,
                    selection_count_per_activation: 8,
                    index_shift: 0,
                    index_mask: 0xffff,
                },
            })
        );
    }

    #[test]
    fn selected_parameter_access_contract_is_preserved_in_the_stream_plan() {
        let node = CircuitNode {
            id: "selected_compute".to_string(),
            op: "generic_compute".to_string(),
            inputs: vec!["activation".to_string(), "selected".to_string()],
            outputs: vec!["output".to_string()],
            params: vec!["bank".to_string(), "scale".to_string()],
            state_reads: Vec::new(),
            state_writes: Vec::new(),
            attrs: serde_json::json!({
                "selected_parameter_accesses": [{
                    "selection_signal": "selected",
                    "partition_axis": 0,
                    "parameter_ids": ["bank", "scale"]
                }]
            }),
        };

        assert_eq!(
            planned_node_selected_parameter_accesses("component", &node).unwrap(),
            vec![PlannedSelectedParameterAccess {
                selection_signal: "selected".to_string(),
                layout: PlannedSelectedParameterLayout::Partitioned {
                    partition_axis: 0,
                },
                parameter_ids: vec!["bank".to_string(), "scale".to_string()],
            }]
        );
    }

    #[test]
    fn independent_selected_parameter_access_preserves_explicit_resource_slots() {
        let node = CircuitNode {
            id: "expert_compute".to_string(),
            op: "generic_compute".to_string(),
            inputs: vec!["activation".to_string(), "selected".to_string()],
            outputs: vec!["output".to_string()],
            params: vec![
                "expert_0_scale".to_string(),
                "expert_0_weight".to_string(),
                "expert_1_scale".to_string(),
                "expert_1_weight".to_string(),
            ],
            state_reads: Vec::new(),
            state_writes: Vec::new(),
            attrs: serde_json::json!({
                "selected_parameter_accesses": [{
                    "selection_signal": "selected",
                    "mapping": [
                        {"selector": 0, "parameter_ids": ["expert_0_scale", "expert_0_weight"]},
                        {"selector": 1, "parameter_ids": ["expert_1_scale", "expert_1_weight"]}
                    ]
                }]
            }),
        };

        assert_eq!(
            planned_node_selected_parameter_accesses("component", &node).unwrap(),
            vec![PlannedSelectedParameterAccess {
                selection_signal: "selected".to_string(),
                layout: PlannedSelectedParameterLayout::Independent {
                    resource_count: 2,
                    parameters_per_resource: 2,
                },
                parameter_ids: node.params.clone(),
            }]
        );
    }

    #[test]
    fn independent_selected_parameter_access_rejects_ragged_slots() {
        let node = CircuitNode {
            id: "expert_compute".to_string(),
            op: "generic_compute".to_string(),
            inputs: vec!["activation".to_string(), "selected".to_string()],
            outputs: vec!["output".to_string()],
            params: vec![
                "expert_0_scale".to_string(),
                "expert_0_weight".to_string(),
                "expert_1_weight".to_string(),
            ],
            state_reads: Vec::new(),
            state_writes: Vec::new(),
            attrs: serde_json::json!({
                "selected_parameter_accesses": [{
                    "selection_signal": "selected",
                    "mapping": [
                        {"selector": 0, "parameter_ids": ["expert_0_scale", "expert_0_weight"]},
                        {"selector": 1, "parameter_ids": ["expert_1_weight"]}
                    ]
                }]
            }),
        };

        let error = planned_node_selected_parameter_accesses("component", &node)
            .unwrap_err();

        assert!(error
            .to_string()
            .contains("not rectangular and unique"));
    }

    #[test]
    fn selected_parameter_access_contract_rejects_nonphysical_metadata() {
        let node = CircuitNode {
            id: "selected_compute".to_string(),
            op: "generic_compute".to_string(),
            inputs: vec!["activation".to_string(), "selected".to_string()],
            outputs: vec!["output".to_string()],
            params: vec!["bank".to_string(), "scale".to_string()],
            state_reads: Vec::new(),
            state_writes: Vec::new(),
            attrs: serde_json::json!({
                "selected_parameter_accesses": [{
                    "selection_signal": "selected",
                    "partition_axis": 0,
                    "parameter_ids": ["scale", "bank"]
                }]
            }),
        };

        let error =
            planned_node_selected_parameter_accesses("component", &node)
                .unwrap_err();
        assert!(error
            .to_string()
            .contains("must be non-empty, unique, and sorted"));
    }

    #[test]
    fn selection_domain_contract_rejects_malformed_metadata() {
        let mut missing_id = valid_selection_domain();
        missing_id.as_object_mut().unwrap().remove("id");
        let mut blank_id = valid_selection_domain();
        blank_id["id"] = serde_json::json!(" ");
        let mut zero_resources = valid_selection_domain();
        zero_resources["resource_count"] = serde_json::json!(0);
        let mut wrong_signal = valid_selection_domain();
        wrong_signal["selection_signal"] = serde_json::json!("not_an_output");
        let mut missing_encoding = valid_selection_domain();
        missing_encoding.as_object_mut().unwrap().remove("encoding");
        let mut wrong_element = valid_selection_domain();
        wrong_element["encoding"]["element_type"] = serde_json::json!("u16");
        let mut zero_selections = valid_selection_domain();
        zero_selections["encoding"]["selection_count_per_activation"] =
            serde_json::json!(0);
        let mut excessive_shift = valid_selection_domain();
        excessive_shift["encoding"]["index_shift"] = serde_json::json!(32);
        let mut inadequate_mask = valid_selection_domain();
        inadequate_mask["encoding"]["index_mask"] = serde_json::json!(0x7f);

        let invalid = [
            (
                serde_json::json!("addressable_resources"),
                "must be an object",
            ),
            (missing_id, "ambiguous fields"),
            (blank_id, "id must be a non-empty string"),
            (zero_resources, "resource_count must be a positive integer"),
            (wrong_signal, "selection_signal must name a node output"),
            (missing_encoding, "ambiguous fields"),
            (wrong_element, "element_type is unsupported"),
            (
                zero_selections,
                "selection_count_per_activation must be a positive integer",
            ),
            (excessive_shift, "index_shift must be below 32"),
            (inadequate_mask, "index_mask cannot represent its domain"),
        ];

        for (selection_domain, expected) in invalid {
            let node = node_with_selection_domain(selection_domain);
            let error = planned_node_selection_domain("component", &node).unwrap_err();
            assert!(
                error.to_string().contains(expected),
                "expected {expected:?} in {error}"
            );
        }
    }
}
