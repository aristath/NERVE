pub const VULKAN_RUNTIME_RESIDENCY_PLAN_SCHEMA: &str =
    "nerve.vulkan_runtime_residency_plan.v7";

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VulkanRuntimeResidencyPlan {
    pub schema: String,
    pub package_id: String,
    pub residency_policy: ResourceResidencyPolicy,
    pub context_capacity_activations: usize,
    pub speculative_draft_tokens: usize,
    pub device_plans: Vec<VulkanRuntimeDeviceResidencyPlan>,
    pub total_initial_device_resident_bytes: usize,
    pub total_current_resident_parameter_bytes: usize,
    pub total_maximum_addressable_parameter_bytes: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct VulkanRuntimeDeviceResidencyBreakdown {
    pub stream_state_bytes: usize,
    pub state_transaction_bytes: usize,
    pub activation_slot_bytes: usize,
    pub selection_telemetry_bytes: usize,
    pub boundary_buffer_bytes: usize,
    pub edge_buffer_bytes: usize,
    pub stream_control_bytes: usize,
    pub output_transducer_workspace_bytes: usize,
    pub sampler_workspace_bytes: usize,
    pub feedback_workspace_bytes: usize,
    pub speculative_decoder_state_bytes: usize,
    pub speculative_decoder_activation_bytes: usize,
    pub speculative_decoder_workspace_bytes: usize,
    pub causal_verification_snapshot_bytes: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct VulkanRuntimeWorkingSetBytes {
    pub transient_state_bytes: usize,
    pub activation_headroom_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VulkanRuntimeDeviceResidencyPlan {
    pub device_id: String,
    pub parameter_residency: VulkanRuntimeParameterResidencyBytes,
    pub resource_store: VulkanCompiledResourceStoreResidencyBytes,
    pub working_set: VulkanRuntimeWorkingSetBytes,
    pub breakdown: VulkanRuntimeDeviceResidencyBreakdown,
    pub resident_stream_device_allocations: Vec<VulkanRuntimeResidentStreamAllocation>,
    pub initial_device_resident_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct VulkanRuntimeResidentStreamAllocation {
    pub kind: VulkanRuntimeResidentStreamAllocationKind,
    pub byte_capacity: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VulkanRuntimeResidentBufferClass {
    OutputTransducerWorkspace,
    SamplerWorkspace,
    FeedbackWorkspace,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum VulkanRuntimeResidentStreamAllocationKind {
    State {
        component_id: String,
        state_id: String,
    },
    StateTransaction {
        component_id: String,
        state_id: String,
    },
    CausalVerificationSnapshot {
        component_id: String,
        state_id: String,
    },
    SelectionTelemetry {
        component_id: String,
        node_id: String,
        domain_id: String,
    },
    ActivationSlot {
        component_id: String,
        slot: usize,
    },
    BoundaryInput {
        component_id: String,
        signal_id: String,
    },
    BoundaryOutput {
        component_id: String,
        signal_id: String,
    },
    EdgeProducedPort {
        component_id: String,
        port_id: String,
        edge_indices: Vec<usize>,
    },
    EdgeIncoming {
        edge_index: usize,
    },
    EdgeStagingReplica {
        component_id: String,
        port_id: String,
        edge_indices: Vec<usize>,
    },
    RuntimeBuffer {
        class: VulkanRuntimeResidentBufferClass,
        scope_id: String,
        buffer_id: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanRuntimeResidencyPlanError(pub String);

impl Display for VulkanRuntimeResidencyPlanError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for VulkanRuntimeResidencyPlanError {}

/// Plans the buffers owned by one resident placed stream without creating a
/// Vulkan instance or logical device. The plan deliberately uses the compiled
/// graph and the runtime's physical transient-state layout rather than a
/// model-family estimate.
pub fn plan_vulkan_runtime_residency(
    manifest_dir: impl AsRef<Path>,
    runtime_model: &VulkanResidentRuntimeModel,
    tensor_index: &TensorIndex,
    context_capacity_activations: usize,
    speculative_draft_tokens: usize,
    residency_policy: ResourceResidencyPolicy,
) -> Result<VulkanRuntimeResidencyPlan, VulkanRuntimeResidencyPlanError> {
    let resource_contract =
        instantiate_runtime_resource_contract(runtime_model)
            .map_err(|error| VulkanRuntimeResidencyPlanError(error.to_string()))?;
    plan_vulkan_runtime_residency_with_contract(
        manifest_dir,
        runtime_model,
        tensor_index,
        context_capacity_activations,
        speculative_draft_tokens,
        residency_policy,
        &resource_contract,
    )
}

#[allow(clippy::too_many_arguments)]
fn plan_vulkan_runtime_residency_with_contract(
    manifest_dir: impl AsRef<Path>,
    runtime_model: &VulkanResidentRuntimeModel,
    tensor_index: &TensorIndex,
    context_capacity_activations: usize,
    speculative_draft_tokens: usize,
    residency_policy: ResourceResidencyPolicy,
    resource_contract: &CompiledResourceResidencyContract,
) -> Result<VulkanRuntimeResidencyPlan, VulkanRuntimeResidencyPlanError> {
    let mount_speculative_decoders = speculative_draft_tokens > 0;
    if context_capacity_activations == 0 {
        return Err(VulkanRuntimeResidencyPlanError(
            "runtime residency context capacity must be positive".to_string(),
        ));
    }
    if context_capacity_activations > runtime_model.package.max_context_activations {
        return Err(VulkanRuntimeResidencyPlanError(format!(
            "runtime residency context capacity {context_capacity_activations} exceeds package maximum {}",
            runtime_model.package.max_context_activations
        )));
    }
    let manifest_dir = manifest_dir.as_ref();
    let (input_component_id, output_component_id) = runtime_model
        .circuit_graph
        .signal_processor_endpoint_component_ids()
        .map_err(residency_package_error)?;
    let input_device_id = runtime_model
        .placement
        .device_for_component(&input_component_id)
        .to_string();
    let output_device_id = runtime_model
        .placement
        .device_for_component(&output_component_id)
        .to_string();
    let device_ids = runtime_model
        .circuit_graph
        .signal_processor_owner_device_ids(&runtime_model.placement);
    if device_ids.is_empty() {
        return Err(VulkanRuntimeResidencyPlanError(
            "runtime residency plan has no owner devices".to_string(),
        ));
    }

    let (_resource_plan, _placement_plan, _first_placed_plan) =
        plan_resident_package_placed_stream_circuit_with_tensor_index(
            &device_ids[0],
            &runtime_model.placement,
            &runtime_model.circuit_graph,
            manifest_dir,
            tensor_index,
            runtime_model.package.activation_element_bytes,
        )
        .map_err(residency_package_error)?;

    let mut by_device = device_ids
        .iter()
        .map(|device_id| {
            (
                device_id.clone(),
                VulkanRuntimeDeviceResidencyBreakdown::default(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut parameter_residency_by_device =
        plan_compiled_parameter_residency(
            runtime_model,
            resource_contract,
            &input_device_id,
            &output_device_id,
            &device_ids,
            mount_speculative_decoders,
            residency_policy,
        )?;
    let compiled_resource_layout = VulkanCompiledResourceAddressLayout::from_contract(
        resource_contract,
    )
    .map_err(|error| VulkanRuntimeResidencyPlanError(error.to_string()))?;
    let minimum_upload_alignment =
        compiled_resource_contract_minimum_upload_alignment(resource_contract)?;
    let mut resident_stream_device_allocations_by_device = BTreeMap::new();

    for device_id in &device_ids {
        let (_resources, _placement, placed_plan) =
            plan_resident_package_placed_stream_circuit_with_tensor_index(
                device_id,
                &runtime_model.placement,
                &runtime_model.circuit_graph,
                manifest_dir,
                tensor_index,
                runtime_model.package.activation_element_bytes,
            )
            .map_err(residency_package_error)?;
        let breakdown = by_device
            .get_mut(device_id)
            .expect("owner device was indexed above");
        let stream = plan_stream_circuit_residency(
            &placed_plan,
            context_capacity_activations,
            mount_speculative_decoders,
            speculative_draft_tokens,
        )?;
        breakdown.stream_state_bytes = stream.state_bytes;
        breakdown.state_transaction_bytes = stream.transaction_bytes;
        breakdown.activation_slot_bytes = stream.activation_bytes;
        breakdown.selection_telemetry_bytes = stream.selection_telemetry_bytes;
        breakdown.boundary_buffer_bytes = stream.boundary_bytes;
        breakdown.edge_buffer_bytes = stream.edge_bytes;
        breakdown.causal_verification_snapshot_bytes =
            stream.causal_verification_snapshot_bytes;
        resident_stream_device_allocations_by_device
            .insert(device_id.clone(), stream.allocations);
        // The placed runtime aliases a single stream-control allocation across
        // devices when possible. Charging every importing device is a safe
        // admission bound and remains negligible.
        breakdown.stream_control_bytes = VULKAN_STREAM_CONTROL_BYTE_CAPACITY;
    }

    let output = by_device.get_mut(&output_device_id).ok_or_else(|| {
        VulkanRuntimeResidencyPlanError(format!(
            "output device {output_device_id:?} has no resident component slice"
        ))
    })?;
    let output_transducer_allocations = vec![
        runtime_buffer_allocation(
            VulkanRuntimeResidentBufferClass::OutputTransducerWorkspace,
            &runtime_model.package.output_transducer.spec.transducer_id,
            "normalized_frame",
            runtime_model
                .package
                .output_transducer
                .spec
                .normalized_frame_byte_capacity,
        )?,
        runtime_buffer_allocation(
            VulkanRuntimeResidentBufferClass::OutputTransducerWorkspace,
            &runtime_model.package.output_transducer.spec.transducer_id,
            "logits",
            runtime_model.package.output_transducer.spec.logits_byte_capacity,
        )?,
    ];
    output.output_transducer_workspace_bytes =
        runtime_buffer_allocation_total(&output_transducer_allocations)?;
    let sampler_allocations = sampler_workspace_allocations(
        &runtime_model.package.sampler.spec,
        context_capacity_activations,
        false,
    )?;
    output.sampler_workspace_bytes = runtime_buffer_allocation_total(&sampler_allocations)?;
    let feedback_allocations = main_feedback_workspace_allocations(
        runtime_model,
        context_capacity_activations,
        mount_speculative_decoders,
    )?;
    output.feedback_workspace_bytes = runtime_buffer_allocation_total(&feedback_allocations)?;
    resident_stream_device_allocations_by_device
        .get_mut(&output_device_id)
        .expect("output owner allocation ledger was indexed above")
        .extend(
            output_transducer_allocations
                .into_iter()
                .chain(sampler_allocations)
                .chain(feedback_allocations),
        );

    if mount_speculative_decoders {
        for decoder in &runtime_model.package.speculative_decoders {
            plan_speculative_decoder_residency(
                &mut by_device,
                manifest_dir,
                runtime_model,
                tensor_index,
                decoder,
                &output_device_id,
                context_capacity_activations,
            )?;
        }
    }

    let mut total_initial_device_resident_bytes = 0usize;
    let mut total_current_resident_parameter_bytes = 0usize;
    let mut total_maximum_addressable_parameter_bytes = 0usize;
    let mut device_plans = Vec::with_capacity(by_device.len());
    for (device_id, breakdown) in by_device {
        let parameter_residency = parameter_residency_by_device
            .remove(&device_id)
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(format!(
                    "runtime parameter plan omitted device {device_id:?}"
                ))
            })?;
        let resource_store = if parameter_residency.maximum_addressable_bytes
            > parameter_residency.always_resident_bytes
        {
            let logical_device_ids = BTreeSet::from([device_id.clone()]);
            let allowed_selector_ids = compiled_resource_selector_ids_for_device_set(
                runtime_model,
                resource_contract,
                &input_device_id,
                &output_device_id,
                &logical_device_ids,
                mount_speculative_decoders,
            )?;
            plan_compiled_resource_store_residency(
                resource_contract,
                &compiled_resource_layout,
                &allowed_selector_ids,
                parameter_residency.staging_headroom_bytes,
                minimum_upload_alignment,
            )?
        } else {
            VulkanCompiledResourceStoreResidencyBytes::default()
        };
        let working_set = VulkanRuntimeWorkingSetBytes {
            transient_state_bytes:
                sum_transient_state_breakdown(&breakdown)?,
            activation_headroom_bytes:
                sum_activation_headroom_breakdown(&breakdown)?,
        };
        let initial_resource_store_bytes = if residency_policy == ResourceResidencyPolicy::Eager {
            resource_store.maximum_extra_device_bytes()?
        } else {
            resource_store.fixed_device_bytes()?
        };
        let initial_device_resident_bytes = [
            parameter_residency.current_resident_bytes,
            initial_resource_store_bytes,
            working_set.transient_state_bytes,
            working_set.activation_headroom_bytes,
        ]
        .into_iter()
        .try_fold(0usize, |total, bytes| {
            checked_residency_add(
                total,
                bytes,
                "initial device residency total",
            )
        })?;
        total_initial_device_resident_bytes = checked_residency_add(
            total_initial_device_resident_bytes,
            initial_device_resident_bytes,
            "runtime initial residency total",
        )?;
        total_current_resident_parameter_bytes = checked_residency_add(
            total_current_resident_parameter_bytes,
            parameter_residency.current_resident_bytes,
            "runtime current parameter total",
        )?;
        total_maximum_addressable_parameter_bytes = checked_residency_add(
            total_maximum_addressable_parameter_bytes,
            parameter_residency.maximum_addressable_bytes,
            "runtime maximum addressable parameter total",
        )?;
        let resident_stream_device_allocations =
            resident_stream_device_allocations_by_device
                .remove(&device_id)
                .ok_or_else(|| {
                    VulkanRuntimeResidencyPlanError(format!(
                        "runtime residency lost the resident allocation ledger for {device_id:?}",
                    ))
                })?;
        device_plans.push(VulkanRuntimeDeviceResidencyPlan {
            device_id,
            parameter_residency,
            resource_store,
            working_set,
            breakdown,
            resident_stream_device_allocations,
            initial_device_resident_bytes,
        });
    }
    if !parameter_residency_by_device.is_empty() {
        return Err(VulkanRuntimeResidencyPlanError(
            "runtime parameter plan contains unknown devices".to_string(),
        ));
    }
    Ok(VulkanRuntimeResidencyPlan {
        schema: VULKAN_RUNTIME_RESIDENCY_PLAN_SCHEMA.to_string(),
        package_id: runtime_model.package.package_id.clone(),
        residency_policy,
        context_capacity_activations,
        speculative_draft_tokens,
        device_plans,
        total_initial_device_resident_bytes,
        total_current_resident_parameter_bytes,
        total_maximum_addressable_parameter_bytes,
    })
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct StreamCircuitResidencyBytes {
    state_bytes: usize,
    transaction_bytes: usize,
    activation_bytes: usize,
    selection_telemetry_bytes: usize,
    boundary_bytes: usize,
    edge_bytes: usize,
    causal_verification_snapshot_bytes: usize,
    allocations: Vec<VulkanRuntimeResidentStreamAllocation>,
}

fn plan_stream_circuit_residency(
    placed_plan: &VulkanPlacedStreamCircuitPlan,
    context_capacity_activations: usize,
    transactional: bool,
    speculative_draft_tokens: usize,
) -> Result<StreamCircuitResidencyBytes, VulkanRuntimeResidencyPlanError> {
    plan_stream_circuit_residency_for_component(
        placed_plan,
        None,
        context_capacity_activations,
        transactional,
        speculative_draft_tokens,
    )
}

fn plan_component_stream_circuit_residency(
    placed_plan: &VulkanPlacedStreamCircuitPlan,
    component_id: &str,
    context_capacity_activations: usize,
    transactional: bool,
    speculative_draft_tokens: usize,
) -> Result<StreamCircuitResidencyBytes, VulkanRuntimeResidencyPlanError> {
    plan_stream_circuit_residency_for_component(
        placed_plan,
        Some(component_id),
        context_capacity_activations,
        transactional,
        speculative_draft_tokens,
    )
}

fn plan_stream_circuit_residency_for_component(
    placed_plan: &VulkanPlacedStreamCircuitPlan,
    component_id: Option<&str>,
    context_capacity_activations: usize,
    transactional: bool,
    speculative_draft_tokens: usize,
) -> Result<StreamCircuitResidencyBytes, VulkanRuntimeResidencyPlanError> {
    let resident = &placed_plan.placed_resident_plan.resident_plan;
    let mut state_bytes = 0usize;
    let mut transaction_bytes = 0usize;
    let mut allocations = Vec::new();
    let verification_lane_capacity = if speculative_draft_tokens == 0 {
        0
    } else {
        causal_component_block_lane_capacity(
            speculative_draft_tokens.checked_add(1).ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(
                    "speculative verification width overflowed".to_string(),
                )
            })?,
        )
        .map_err(residency_display_error)?
    };
    let mut causal_verification_snapshot_bytes = 0usize;
    for state in resident
        .stream_state_buffers
        .iter()
        .filter(|state| component_id.is_none_or(|component| state.component_id == component))
    {
        let layout = VulkanTransientStateBufferLayout::for_state(
            state,
            context_capacity_activations,
        )
        .map_err(residency_display_error)?;
        state_bytes =
            checked_residency_add(state_bytes, layout.byte_capacity, "stream state bytes")?;
        allocations.push(VulkanRuntimeResidentStreamAllocation {
            kind: VulkanRuntimeResidentStreamAllocationKind::State {
                component_id: state.component_id.clone(),
                state_id: state.state_id.clone(),
            },
            byte_capacity: layout.byte_capacity,
        });
        if transactional && state.static_bytes.is_some() {
            // Speculative verification uses one transactional slot and one
            // baseline, exactly matching new_transactional(..., 1).
            let byte_capacity = checked_residency_mul(
                layout.byte_capacity,
                2,
                "state transaction bytes",
            )?;
            transaction_bytes = checked_residency_add(
                transaction_bytes,
                byte_capacity,
                "state transaction bytes",
            )?;
            allocations.push(VulkanRuntimeResidentStreamAllocation {
                kind: VulkanRuntimeResidentStreamAllocationKind::StateTransaction {
                    component_id: state.component_id.clone(),
                    state_id: state.state_id.clone(),
                },
                byte_capacity,
            });
        }
        if let Some(static_bytes) = state.static_bytes
            && verification_lane_capacity > 0
        {
            let byte_capacity = checked_residency_mul(
                static_bytes,
                verification_lane_capacity,
                "causal verification snapshots",
            )?;
            causal_verification_snapshot_bytes = checked_residency_add(
                causal_verification_snapshot_bytes,
                byte_capacity,
                "causal verification snapshots",
            )?;
            allocations.push(VulkanRuntimeResidentStreamAllocation {
                kind: VulkanRuntimeResidentStreamAllocationKind::CausalVerificationSnapshot {
                    component_id: state.component_id.clone(),
                    state_id: state.state_id.clone(),
                },
                byte_capacity,
            });
        }
    }
    let mut selection_telemetry_bytes = 0usize;
    for telemetry in resident.selection_telemetry.iter().filter(|telemetry| {
        component_id.is_none_or(|component| telemetry.component_id == component)
    }) {
        selection_telemetry_bytes = checked_residency_add(
            selection_telemetry_bytes,
            telemetry.byte_capacity,
            "selection telemetry bytes",
        )?;
        allocations.push(VulkanRuntimeResidentStreamAllocation {
            kind: VulkanRuntimeResidentStreamAllocationKind::SelectionTelemetry {
                component_id: telemetry.component_id.clone(),
                node_id: telemetry.node_id.clone(),
                domain_id: telemetry.domain_id.clone(),
            },
            byte_capacity: telemetry.byte_capacity,
        });
    }
    let mut activation_bytes = 0usize;
    for bank in resident
        .activation_banks
        .iter()
        .filter(|bank| component_id.is_none_or(|component| bank.component_id == component))
    {
        for slot in &bank.slots {
            let label = format!(
                "activation slot {}.slot_{} for signals {:?}",
                bank.component_id, slot.slot, slot.signal_ids
            );
            let byte_capacity = required_optional_bytes(slot.bytes, &label)?;
            activation_bytes = checked_residency_add(
                activation_bytes,
                byte_capacity,
                "activation slot bytes",
            )?;
            allocations.push(VulkanRuntimeResidentStreamAllocation {
                kind: VulkanRuntimeResidentStreamAllocationKind::ActivationSlot {
                    component_id: bank.component_id.clone(),
                    slot: slot.slot,
                },
                byte_capacity,
            });
        }
    }
    let boundary_plan = VulkanModelBoundaryBufferPlan::from_placed_plan(placed_plan)
        .map_err(residency_display_error)?;
    let selected_inputs = boundary_plan
        .inputs
        .iter()
        .filter(|input| component_id.is_none_or(|component| input.component_id == component))
        .collect::<Vec<_>>();
    let mut boundary_bytes = 0usize;
    for input in &selected_inputs {
        let byte_capacity =
            required_optional_bytes(input.byte_capacity, "model input boundary buffer")?;
        boundary_bytes = checked_residency_add(
            boundary_bytes,
            byte_capacity,
            "model boundary buffers",
        )?;
        allocations.push(VulkanRuntimeResidentStreamAllocation {
            kind: VulkanRuntimeResidentStreamAllocationKind::BoundaryInput {
                component_id: input.component_id.clone(),
                signal_id: input.signal_id.clone(),
            },
            byte_capacity,
        });
    }
    for output in boundary_plan
        .outputs
        .iter()
        .filter(|output| component_id.is_none_or(|component| output.component_id == component))
    {
        let aliases_input = output.source_signal_id.is_some()
            && selected_inputs.iter().any(|input| {
                input.component_id == output.component_id
                    && input.signal_id == output.signal_id
                    && input.shape == output.shape
            });
        if aliases_input {
            continue;
        }
        let byte_capacity =
            required_optional_bytes(output.byte_capacity, "model output boundary buffer")?;
        boundary_bytes = checked_residency_add(
            boundary_bytes,
            byte_capacity,
            "model boundary buffers",
        )?;
        allocations.push(VulkanRuntimeResidentStreamAllocation {
            kind: VulkanRuntimeResidentStreamAllocationKind::BoundaryOutput {
                component_id: output.component_id.clone(),
                signal_id: output.signal_id.clone(),
            },
            byte_capacity,
        });
    }
    let edge_plan = VulkanPlacedEdgeIoPlan::from_placed_resident_plan(
        &placed_plan.placed_resident_plan,
    )
    .map_err(residency_display_error)?;
    let (edge_bytes, edge_allocations) = plan_edge_residency_allocations(
        placed_plan,
        &edge_plan,
        &boundary_plan,
        component_id,
    )?;
    allocations.extend(edge_allocations);
    Ok(StreamCircuitResidencyBytes {
        state_bytes,
        transaction_bytes,
        activation_bytes,
        selection_telemetry_bytes,
        boundary_bytes,
        edge_bytes,
        causal_verification_snapshot_bytes,
        allocations,
    })
}

fn plan_edge_residency_allocations(
    placed_plan: &VulkanPlacedStreamCircuitPlan,
    plan: &VulkanPlacedEdgeIoPlan,
    boundary_plan: &VulkanModelBoundaryBufferPlan,
    component_id: Option<&str>,
) -> Result<(usize, Vec<VulkanRuntimeResidentStreamAllocation>), VulkanRuntimeResidencyPlanError> {
    let passthrough_produced_ports = placed_plan
        .binding_plan
        .circuits
        .iter()
        .flat_map(|circuit| {
            circuit.output_ports.iter().filter_map(|output| {
                let source_signal_id = output.source.as_deref()?;
                boundary_plan
                    .inputs
                    .iter()
                    .any(|input| {
                        input.component_id == circuit.component_id
                            && input.signal_id == source_signal_id
                            && input.shape == output.shape
                    })
                    .then(|| (circuit.component_id.clone(), output.id.clone()))
            })
        })
        .collect::<BTreeSet<_>>();
    plan_edge_residency_allocations_with_passthrough(
        plan,
        &passthrough_produced_ports,
        component_id,
    )
}

fn plan_edge_residency_allocations_with_passthrough(
    plan: &VulkanPlacedEdgeIoPlan,
    passthrough_produced_ports: &BTreeSet<(String, String)>,
    component_id: Option<&str>,
) -> Result<(usize, Vec<VulkanRuntimeResidentStreamAllocation>), VulkanRuntimeResidencyPlanError> {
    let mut produced_ports = BTreeMap::<(String, String), (usize, BTreeSet<usize>)>::new();
    for (producer_component_id, producer_port_id, edge_index, byte_capacity) in plan
        .local_edges
        .iter()
        .map(|edge| {
            (
                edge.source_component_id.as_str(),
                edge.source_port_id.as_str(),
                edge.edge_index,
                edge.byte_capacity,
            )
        })
        .chain(
            plan.endpoints
                .iter()
                .filter(|endpoint| endpoint.direction == VulkanPlacedEdgeDirection::Outgoing)
                .map(|endpoint| {
                    (
                        endpoint.local_component_id.as_str(),
                        endpoint.local_port_id.as_str(),
                        endpoint.edge_index,
                        endpoint.byte_capacity,
                    )
                }),
        )
    {
        let byte_capacity =
            required_optional_bytes(byte_capacity, "placed produced-port buffer")?;
        let key = (
            producer_component_id.to_string(),
            producer_port_id.to_string(),
        );
        match produced_ports.entry(key) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert((byte_capacity, BTreeSet::from([edge_index])));
            }
            std::collections::btree_map::Entry::Occupied(mut entry) => {
                let (existing_capacity, edge_indices) = entry.get_mut();
                if *existing_capacity != byte_capacity {
                    return Err(VulkanRuntimeResidencyPlanError(format!(
                        "placed produced port {producer_component_id}.{producer_port_id} has incompatible capacities {existing_capacity} and {byte_capacity}",
                    )));
                }
                edge_indices.insert(edge_index);
            }
        }
    }

    let mut total = 0usize;
    let mut allocations = Vec::new();
    for ((producer_component_id, port_id), (byte_capacity, edge_indices)) in produced_ports {
        if component_id.is_some_and(|component| component != producer_component_id)
            || passthrough_produced_ports
                .contains(&(producer_component_id.clone(), port_id.clone()))
        {
            continue;
        }
        total = checked_residency_add(total, byte_capacity, "placed edge buffers")?;
        allocations.push(VulkanRuntimeResidentStreamAllocation {
            kind: VulkanRuntimeResidentStreamAllocationKind::EdgeProducedPort {
                component_id: producer_component_id,
                port_id,
                edge_indices: edge_indices.into_iter().collect(),
            },
            byte_capacity,
        });
    }
    for endpoint in &plan.endpoints {
        if endpoint.direction != VulkanPlacedEdgeDirection::Incoming
            || component_id.is_some_and(|component| component != endpoint.local_component_id)
        {
            continue;
        }
        let byte_capacity =
            required_optional_bytes(endpoint.byte_capacity, "placed incoming edge buffer")?;
        total = checked_residency_add(total, byte_capacity, "placed edge buffers")?;
        allocations.push(VulkanRuntimeResidentStreamAllocation {
            kind: VulkanRuntimeResidentStreamAllocationKind::EdgeIncoming {
                edge_index: endpoint.edge_index,
            },
            byte_capacity,
        });
    }
    Ok((total, allocations))
}

#[allow(clippy::too_many_arguments)]
fn plan_speculative_decoder_residency(
    by_device: &mut BTreeMap<String, VulkanRuntimeDeviceResidencyBreakdown>,
    manifest_dir: &Path,
    target_runtime_model: &VulkanResidentRuntimeModel,
    tensor_index: &TensorIndex,
    decoder: &VulkanResidentSpeculativeDecoderPackageSpec,
    output_device_id: &str,
    context_capacity_activations: usize,
) -> Result<(), VulkanRuntimeResidencyPlanError> {
    let draft_runtime_model =
        speculative_decoder_runtime_model(target_runtime_model, decoder, output_device_id);
    let (_resources, _placement, placed_plan) =
        plan_resident_package_placed_stream_circuit_with_tensor_index(
            output_device_id,
            &draft_runtime_model.placement,
            &draft_runtime_model.circuit_graph,
            manifest_dir,
            tensor_index,
            draft_runtime_model.package.activation_element_bytes,
        )
        .map_err(residency_package_error)?;
    let breakdown = by_device.get_mut(output_device_id).ok_or_else(|| {
        VulkanRuntimeResidencyPlanError(format!(
            "speculative decoder device {output_device_id:?} has no resident component slice"
        ))
    })?;
    let stream =
        plan_stream_circuit_residency(&placed_plan, context_capacity_activations, true, 0)?;
    let state_bytes = [
        stream.state_bytes,
        stream.transaction_bytes,
        VULKAN_STREAM_CONTROL_BYTE_CAPACITY,
    ]
    .into_iter()
    .try_fold(0usize, |total, bytes| {
        checked_residency_add(total, bytes, "speculative decoder state bytes")
    })?;
    breakdown.speculative_decoder_state_bytes = checked_residency_add(
        breakdown.speculative_decoder_state_bytes,
        state_bytes,
        "speculative decoder state bytes",
    )?;
    let activation_bytes = [
        stream.activation_bytes,
        stream.boundary_bytes,
        stream.edge_bytes,
    ]
    .into_iter()
    .try_fold(0usize, |total, bytes| {
        checked_residency_add(
            total,
            bytes,
            "speculative decoder activation bytes",
        )
    })?;
    breakdown.speculative_decoder_activation_bytes =
        checked_residency_add(
            breakdown.speculative_decoder_activation_bytes,
            activation_bytes,
            "speculative decoder activation bytes",
        )?;

    let (output_workspace, sampler_workspace, auxiliary_workspace) = match (
        decoder.dedicated_input_adapter(),
        decoder.dedicated_output_transducer(),
    ) {
        (Some(input), Some(output)) => (
            checked_residency_add(
                output.output_hidden_byte_capacity,
                output.logits_byte_capacity,
                "speculative decoder output workspace",
            )?,
            sampler_workspace_bytes(
                &target_runtime_model.package.sampler.spec,
                context_capacity_activations,
                true,
            )?,
            checked_residency_add(
                input.target_hidden_byte_capacity,
                checked_residency_mul(
                    VULKAN_BACKEND_LOOP_MAX_WINDOW,
                    VULKAN_STREAM_CONTROL_BYTE_CAPACITY,
                    "speculative catch-up controls",
                )?,
                "speculative decoder auxiliary workspace",
            )?,
        ),
        (None, None) => (0, 0, 0),
        _ => {
            return Err(VulkanRuntimeResidencyPlanError(format!(
                "speculative decoder {:?} has incomplete dedicated I/O",
                decoder.id
            )))
        }
    };
    breakdown.speculative_decoder_workspace_bytes = checked_residency_add(
        breakdown.speculative_decoder_workspace_bytes,
        [output_workspace, sampler_workspace, auxiliary_workspace]
            .into_iter()
            .try_fold(0usize, |total, bytes| {
                checked_residency_add(
                    total,
                    bytes,
                    "speculative decoder workspace bytes",
                )
            })?,
        "speculative decoder workspace bytes",
    )?;
    Ok(())
}

fn speculative_decoder_runtime_model(
    target: &VulkanResidentRuntimeModel,
    decoder: &VulkanResidentSpeculativeDecoderPackageSpec,
    device_id: &str,
) -> VulkanResidentRuntimeModel {
    let mut circuit_graph = decoder.circuit_graph.clone();
    let compiled_component_ids = decoder
        .component_executions
        .iter()
        .map(|execution| execution.component_id.as_str())
        .collect::<BTreeSet<_>>();
    for component in &mut circuit_graph.components {
        let has_compiled_execution =
            compiled_component_ids.contains(component.component_id.as_str());
        component.runtime_role = speculative_decoder_planning_role(
            component.runtime_role,
            has_compiled_execution,
        );
        component.circuit.runtime_role = speculative_decoder_planning_role(
            component.circuit.runtime_role,
            has_compiled_execution,
        );
    }
    let mut package = target.package.clone();
    package.package_id = format!("{}::{}", package.package_id, decoder.id);
    package.circuit_graph = circuit_graph.clone();
    package.component_executions = decoder.component_executions.clone();
    package.speculative_decoders.clear();
    VulkanResidentRuntimeModel {
        execution_scope: format!("draft:{}", decoder.id),
        package,
        runtime_graph: target.runtime_graph.clone(),
        placement: StreamCircuitPlacementSpec::new(device_id),
        circuit_graph,
        component_executions: decoder.component_executions.clone(),
        tensor_index_fragments: Vec::new(),
        implementation_selection: None,
    }
}

fn speculative_decoder_planning_role(
    role: CircuitRuntimeRole,
    has_compiled_execution: bool,
) -> CircuitRuntimeRole {
    match (role, has_compiled_execution) {
        (
            CircuitRuntimeRole::DraftInputAdapter
            | CircuitRuntimeRole::DraftProcessor
            | CircuitRuntimeRole::DraftOutputTransducer,
            true,
        ) => CircuitRuntimeRole::SignalProcessor,
        (role, _) => role,
    }
}

fn sampler_workspace_bytes(
    spec: &VulkanResidentSamplerSpec,
    context_capacity_activations: usize,
    private_feedback_control: bool,
) -> Result<usize, VulkanRuntimeResidencyPlanError> {
    runtime_buffer_allocation_total(&sampler_workspace_allocations(
        spec,
        context_capacity_activations,
        private_feedback_control,
    )?)
}

fn sampler_workspace_allocations(
    spec: &VulkanResidentSamplerSpec,
    context_capacity_activations: usize,
    private_feedback_control: bool,
) -> Result<Vec<VulkanRuntimeResidentStreamAllocation>, VulkanRuntimeResidencyPlanError> {
    let vocabulary_size = spec.logits_byte_capacity / std::mem::size_of::<f32>();
    let token_state_is_active =
        spec.repetition_penalty != 1.0 || spec.presence_penalty != 0.0;
    let history_byte_capacity = checked_residency_add(
        checked_residency_mul(
            context_capacity_activations,
            VULKAN_SAMPLER_HISTORY_RECORD_BYTE_CAPACITY,
            "sampler history",
        )?,
        spec.output_byte_capacity,
        "sampler history",
    )?;
    let mut allocations = vec![runtime_buffer_allocation(
        VulkanRuntimeResidentBufferClass::SamplerWorkspace,
        &spec.sampler_id,
        "history_and_output",
        history_byte_capacity,
    )?];
    if sampler_method_uses_randomness(&spec.method) {
        allocations.push(runtime_buffer_allocation(
            VulkanRuntimeResidentBufferClass::SamplerWorkspace,
            &spec.sampler_id,
            "scratch",
            spec.scratch_byte_capacity,
        )?);
        allocations.push(runtime_buffer_allocation(
            VulkanRuntimeResidentBufferClass::SamplerWorkspace,
            &spec.sampler_id,
            "random_seed",
            4,
        )?);
    }
    let seen_token_bytes = vocabulary_size
        .div_ceil(u32::BITS as usize)
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| {
            VulkanRuntimeResidencyPlanError("sampler token-state size overflowed".to_string())
        })?;
    if token_state_is_active || spec.runtime_parameterized {
        allocations.push(runtime_buffer_allocation(
            VulkanRuntimeResidentBufferClass::SamplerWorkspace,
            &spec.sampler_id,
            "seen_token_state",
            seen_token_bytes,
        )?);
    }
    if token_state_is_active {
        allocations.push(runtime_buffer_allocation(
            VulkanRuntimeResidentBufferClass::SamplerWorkspace,
            &spec.sampler_id,
            "seen_token_snapshot",
            seen_token_bytes,
        )?);
        allocations.push(runtime_buffer_allocation(
            VulkanRuntimeResidentBufferClass::SamplerWorkspace,
            &spec.sampler_id,
            "seen_token_batch",
            VULKAN_BACKEND_LOOP_MAX_WINDOW * std::mem::size_of::<u32>(),
        )?);
    }
    if spec.runtime_parameterized {
        allocations.push(runtime_buffer_allocation(
            VulkanRuntimeResidentBufferClass::SamplerWorkspace,
            &spec.sampler_id,
            "runtime_parameters",
            6 * std::mem::size_of::<u32>(),
        )?);
    }
    if private_feedback_control {
        allocations.push(runtime_buffer_allocation(
            VulkanRuntimeResidentBufferClass::SamplerWorkspace,
            &spec.sampler_id,
            "private_feedback_control",
            (VULKAN_FEEDBACK_CONTROL_HEADER_WORD_COUNT + 1) * std::mem::size_of::<u32>(),
        )?);
    }
    Ok(allocations)
}

fn main_feedback_workspace_bytes(
    runtime_model: &VulkanResidentRuntimeModel,
    context_capacity_activations: usize,
    mount_speculative_decoders: bool,
) -> Result<usize, VulkanRuntimeResidencyPlanError> {
    runtime_buffer_allocation_total(&main_feedback_workspace_allocations(
        runtime_model,
        context_capacity_activations,
        mount_speculative_decoders,
    )?)
}

fn main_feedback_workspace_allocations(
    runtime_model: &VulkanResidentRuntimeModel,
    context_capacity_activations: usize,
    mount_speculative_decoders: bool,
) -> Result<Vec<VulkanRuntimeResidentStreamAllocation>, VulkanRuntimeResidencyPlanError> {
    let vocabulary_size =
        runtime_model.package.sampler.spec.logits_byte_capacity / std::mem::size_of::<f32>();
    let stop_mask_words = vocabulary_size.div_ceil(u32::BITS as usize);
    // This upper bound includes every declared component and sampler kernel;
    // actual mounted feedback dispatches are a subset.
    let dispatch_capacity = runtime_model
        .component_executions
        .iter()
        .try_fold(3usize, |total, execution| {
            checked_residency_add(
                total,
                execution.kernels.len(),
                "feedback dispatch capacity",
            )
        })?
        .checked_add(runtime_model.package.sampler.kernels.len())
        .ok_or_else(|| {
            VulkanRuntimeResidencyPlanError(
                "feedback dispatch capacity overflowed".to_string(),
            )
        })?;
    let control_bytes = VULKAN_FEEDBACK_CONTROL_HEADER_WORD_COUNT
        .checked_add(stop_mask_words)
        .and_then(|words| {
            dispatch_capacity
                .checked_mul(VULKAN_RESIDENT_INDIRECT_DISPATCH_BYTE_COUNT)
                .and_then(|dispatch_bytes| {
                    words
                        .checked_mul(std::mem::size_of::<u32>())
                        .and_then(|header_bytes| header_bytes.checked_add(dispatch_bytes))
                })
        })
        .ok_or_else(|| {
            VulkanRuntimeResidencyPlanError(
                "feedback control workspace overflowed".to_string(),
            )
        })?;
    let mut allocations = vec![runtime_buffer_allocation(
        VulkanRuntimeResidentBufferClass::FeedbackWorkspace,
        &runtime_model.package.package_id,
        "control",
        control_bytes,
    )?];
    let retains_normalized_frames = mount_speculative_decoders
        && runtime_model
            .package
            .speculative_decoders
            .iter()
            .any(|decoder| decoder.execution_contract.uses_dedicated_autoregressive_io());
    if retains_normalized_frames {
        allocations.push(runtime_buffer_allocation(
            VulkanRuntimeResidentBufferClass::FeedbackWorkspace,
            &runtime_model.package.package_id,
            "speculative_target_frame_history",
            checked_residency_mul(
                runtime_model
                    .package
                    .output_transducer
                    .spec
                    .normalized_frame_byte_capacity,
                context_capacity_activations.min(VULKAN_BACKEND_LOOP_MAX_WINDOW),
                "speculative target-frame history",
            )?,
        )?);
    }
    Ok(allocations)
}

fn runtime_buffer_allocation(
    class: VulkanRuntimeResidentBufferClass,
    scope_id: &str,
    buffer_id: &str,
    byte_capacity: usize,
) -> Result<VulkanRuntimeResidentStreamAllocation, VulkanRuntimeResidencyPlanError> {
    if scope_id.is_empty() || buffer_id.is_empty() || byte_capacity == 0 {
        return Err(VulkanRuntimeResidencyPlanError(
            "runtime buffer allocation requires a class, scope, buffer identity, and positive capacity"
                .to_string(),
        ));
    }
    Ok(VulkanRuntimeResidentStreamAllocation {
        kind: VulkanRuntimeResidentStreamAllocationKind::RuntimeBuffer {
            class,
            scope_id: scope_id.to_string(),
            buffer_id: buffer_id.to_string(),
        },
        byte_capacity,
    })
}

fn runtime_buffer_allocation_total(
    allocations: &[VulkanRuntimeResidentStreamAllocation],
) -> Result<usize, VulkanRuntimeResidencyPlanError> {
    allocations.iter().try_fold(0usize, |total, allocation| {
        checked_residency_add(total, allocation.byte_capacity, "runtime buffer allocations")
    })
}

fn sum_transient_state_breakdown(
    breakdown: &VulkanRuntimeDeviceResidencyBreakdown,
) -> Result<usize, VulkanRuntimeResidencyPlanError> {
    [
        breakdown.stream_state_bytes,
        breakdown.state_transaction_bytes,
        breakdown.stream_control_bytes,
        breakdown.speculative_decoder_state_bytes,
        breakdown.causal_verification_snapshot_bytes,
    ]
    .into_iter()
    .try_fold(0usize, |total, bytes| {
        checked_residency_add(total, bytes, "transient state total")
    })
}

fn sum_activation_headroom_breakdown(
    breakdown: &VulkanRuntimeDeviceResidencyBreakdown,
) -> Result<usize, VulkanRuntimeResidencyPlanError> {
    [
        breakdown.activation_slot_bytes,
        breakdown.selection_telemetry_bytes,
        breakdown.boundary_buffer_bytes,
        breakdown.edge_buffer_bytes,
        breakdown.output_transducer_workspace_bytes,
        breakdown.sampler_workspace_bytes,
        breakdown.feedback_workspace_bytes,
        breakdown.speculative_decoder_activation_bytes,
        breakdown.speculative_decoder_workspace_bytes,
    ]
    .into_iter()
    .try_fold(0usize, |total, bytes| {
        checked_residency_add(total, bytes, "activation headroom total")
    })
}

fn required_optional_bytes(
    bytes: Option<usize>,
    label: &str,
) -> Result<usize, VulkanRuntimeResidencyPlanError> {
    bytes.ok_or_else(|| {
        VulkanRuntimeResidencyPlanError(format!(
            "runtime residency cannot resolve {label} byte capacity"
        ))
    })
}

fn checked_residency_add(
    left: usize,
    right: usize,
    label: &str,
) -> Result<usize, VulkanRuntimeResidencyPlanError> {
    left.checked_add(right)
        .ok_or_else(|| VulkanRuntimeResidencyPlanError(format!("{label} overflowed")))
}

fn checked_residency_mul(
    left: usize,
    right: usize,
    label: &str,
) -> Result<usize, VulkanRuntimeResidencyPlanError> {
    left.checked_mul(right)
        .ok_or_else(|| VulkanRuntimeResidencyPlanError(format!("{label} overflowed")))
}

fn residency_package_error(
    error: VulkanResidentTokenModelPackageError,
) -> VulkanRuntimeResidencyPlanError {
    VulkanRuntimeResidencyPlanError(error.to_string())
}

fn residency_display_error(error: impl Display) -> VulkanRuntimeResidencyPlanError {
    VulkanRuntimeResidencyPlanError(error.to_string())
}
