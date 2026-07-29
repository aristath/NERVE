pub const VULKAN_RUNTIME_RESIDENCY_PLAN_SCHEMA: &str =
    "nerve.vulkan_runtime_residency_plan.v1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VulkanRuntimeResidencyPlan {
    pub schema: String,
    pub package_id: String,
    pub context_capacity_activations: usize,
    pub speculative_decoders_mounted: bool,
    pub device_plans: Vec<VulkanRuntimeDeviceResidencyPlan>,
    pub total_device_resident_bytes: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct VulkanRuntimeDeviceResidencyBreakdown {
    pub component_parameter_bytes: usize,
    pub transducer_parameter_bytes: usize,
    pub stream_state_bytes: usize,
    pub state_transaction_bytes: usize,
    pub activation_slot_bytes: usize,
    pub boundary_buffer_bytes: usize,
    pub edge_buffer_bytes: usize,
    pub stream_control_bytes: usize,
    pub output_transducer_workspace_bytes: usize,
    pub sampler_workspace_bytes: usize,
    pub feedback_workspace_bytes: usize,
    pub speculative_decoder_parameter_bytes: usize,
    pub speculative_decoder_stream_bytes: usize,
    pub speculative_decoder_workspace_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VulkanRuntimeDeviceResidencyPlan {
    pub device_id: String,
    pub breakdown: VulkanRuntimeDeviceResidencyBreakdown,
    pub total_device_resident_bytes: usize,
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
    mount_speculative_decoders: bool,
) -> Result<VulkanRuntimeResidencyPlan, VulkanRuntimeResidencyPlanError> {
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
    if !runtime_model.placement.component_shard_devices.is_empty() {
        return Err(VulkanRuntimeResidencyPlanError(
            "runtime residency planning refuses internal component sharding until its \
             hardware-dependent distributed allocation plan is supplied"
                .to_string(),
        ));
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

    let (resource_plan, _placement_plan, _first_placed_plan) =
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
        breakdown.component_parameter_bytes = required_optional_bytes(
            VulkanPermanentParameterBufferPlan::from_placed_resident_plan(
                &placed_plan.placed_resident_plan,
            )
            .map_err(residency_display_error)?
            .total_byte_capacity,
            "component parameters",
        )?;
        let stream = plan_stream_circuit_residency(
            &placed_plan,
            context_capacity_activations,
            mount_speculative_decoders,
        )?;
        breakdown.stream_state_bytes = stream.state_bytes;
        breakdown.state_transaction_bytes = stream.transaction_bytes;
        breakdown.activation_slot_bytes = stream.activation_bytes;
        breakdown.boundary_buffer_bytes = stream.boundary_bytes;
        breakdown.edge_buffer_bytes = stream.edge_bytes;
        // The placed runtime aliases a single stream-control allocation across
        // devices when possible. Charging every importing device is a safe
        // admission bound and remains negligible.
        breakdown.stream_control_bytes = VULKAN_STREAM_CONTROL_BYTE_CAPACITY;
    }

    plan_transducer_parameters(
        &mut by_device,
        &resource_plan,
        tensor_index,
        &input_device_id,
        &output_device_id,
    )?;
    let output = by_device.get_mut(&output_device_id).ok_or_else(|| {
        VulkanRuntimeResidencyPlanError(format!(
            "output device {output_device_id:?} has no resident component slice"
        ))
    })?;
    output.output_transducer_workspace_bytes = checked_residency_add(
        runtime_model
            .package
            .output_transducer
            .spec
            .normalized_frame_byte_capacity,
        runtime_model.package.output_transducer.spec.logits_byte_capacity,
        "output transducer workspace",
    )?;
    output.sampler_workspace_bytes = sampler_workspace_bytes(
        &runtime_model.package.sampler.spec,
        context_capacity_activations,
        false,
    )?;
    output.feedback_workspace_bytes = main_feedback_workspace_bytes(
        runtime_model,
        context_capacity_activations,
        mount_speculative_decoders,
    )?;

    if mount_speculative_decoders {
        let target_output_tensors = transducer_parameter_tensors(
            &resource_plan,
            "output_transducer",
        );
        for decoder in &runtime_model.package.speculative_decoders {
            plan_speculative_decoder_residency(
                &mut by_device,
                manifest_dir,
                runtime_model,
                tensor_index,
                decoder,
                &output_device_id,
                context_capacity_activations,
                &target_output_tensors,
            )?;
        }
    }

    let mut total_device_resident_bytes = 0usize;
    let mut device_plans = Vec::with_capacity(by_device.len());
    for (device_id, breakdown) in by_device {
        let total = sum_residency_breakdown(&breakdown)?;
        total_device_resident_bytes = checked_residency_add(
            total_device_resident_bytes,
            total,
            "runtime residency total",
        )?;
        device_plans.push(VulkanRuntimeDeviceResidencyPlan {
            device_id,
            breakdown,
            total_device_resident_bytes: total,
        });
    }
    Ok(VulkanRuntimeResidencyPlan {
        schema: VULKAN_RUNTIME_RESIDENCY_PLAN_SCHEMA.to_string(),
        package_id: runtime_model.package.package_id.clone(),
        context_capacity_activations,
        speculative_decoders_mounted: mount_speculative_decoders,
        device_plans,
        total_device_resident_bytes,
    })
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct StreamCircuitResidencyBytes {
    state_bytes: usize,
    transaction_bytes: usize,
    activation_bytes: usize,
    boundary_bytes: usize,
    edge_bytes: usize,
}

fn plan_stream_circuit_residency(
    placed_plan: &VulkanPlacedStreamCircuitPlan,
    context_capacity_activations: usize,
    transactional: bool,
) -> Result<StreamCircuitResidencyBytes, VulkanRuntimeResidencyPlanError> {
    let resident = &placed_plan.placed_resident_plan.resident_plan;
    let mut state_bytes = 0usize;
    let mut transaction_bytes = 0usize;
    for state in &resident.stream_state_buffers {
        let layout = VulkanTransientStateBufferLayout::for_state(
            state,
            context_capacity_activations,
        )
        .map_err(residency_display_error)?;
        state_bytes =
            checked_residency_add(state_bytes, layout.byte_capacity, "stream state bytes")?;
        if transactional && state.static_bytes.is_some() {
            // Speculative verification uses one transactional slot and one
            // baseline, exactly matching new_transactional(..., 1).
            transaction_bytes = checked_residency_add(
                transaction_bytes,
                checked_residency_mul(
                    layout.byte_capacity,
                    2,
                    "state transaction bytes",
                )?,
                "state transaction bytes",
            )?;
        }
    }
    let activation_bytes = resident.activation_banks.iter().try_fold(
        0usize,
        |total, bank| {
            bank.slots.iter().try_fold(total, |total, slot| {
                checked_residency_add(
                    total,
                    required_optional_bytes(slot.bytes, "activation slot")?,
                    "activation slot bytes",
                )
            })
        },
    )?;
    let boundary_bytes = required_optional_bytes(
        VulkanModelBoundaryBufferPlan::from_placed_plan(placed_plan)
            .map_err(residency_display_error)?
            .total_byte_capacity,
        "model boundary buffers",
    )?;
    let edge_bytes = required_optional_bytes(
        VulkanPlacedEdgeIoPlan::from_placed_resident_plan(
            &placed_plan.placed_resident_plan,
        )
        .map_err(residency_display_error)?
        .total_byte_capacity,
        "placed edge buffers",
    )?;
    Ok(StreamCircuitResidencyBytes {
        state_bytes,
        transaction_bytes,
        activation_bytes,
        boundary_bytes,
        edge_bytes,
    })
}

fn plan_transducer_parameters(
    by_device: &mut BTreeMap<String, VulkanRuntimeDeviceResidencyBreakdown>,
    resource_plan: &StreamCircuitResourcePlan,
    tensor_index: &TensorIndex,
    input_device_id: &str,
    output_device_id: &str,
) -> Result<(), VulkanRuntimeResidencyPlanError> {
    if input_device_id == output_device_id {
        let bytes = required_optional_bytes(
            VulkanPermanentParameterBufferPlan::from_transducer_parameters(
                input_device_id,
                resource_plan,
                Some(tensor_index),
            )
            .map_err(residency_display_error)?
            .total_byte_capacity,
            "shared transducer parameters",
        )?;
        by_device
            .get_mut(input_device_id)
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(format!(
                    "transducer device {input_device_id:?} has no resident component slice"
                ))
            })?
            .transducer_parameter_bytes = bytes;
        return Ok(());
    }
    for (device_id, transducer_id) in [
        (input_device_id, "input_transducer"),
        (output_device_id, "output_transducer"),
    ] {
        let bytes = required_optional_bytes(
            VulkanPermanentParameterBufferPlan::from_transducer_parameters_for(
                device_id,
                resource_plan,
                Some(tensor_index),
                transducer_id,
            )
            .map_err(residency_display_error)?
            .total_byte_capacity,
            "transducer parameters",
        )?;
        by_device
            .get_mut(device_id)
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(format!(
                    "transducer device {device_id:?} has no resident component slice"
                ))
            })?
            .transducer_parameter_bytes = bytes;
    }
    Ok(())
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
    target_output_tensors: &BTreeSet<String>,
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
    let draft_parameter_bytes = required_optional_bytes(
        VulkanPermanentParameterBufferPlan::from_placed_resident_plan(
            &placed_plan.placed_resident_plan,
        )
        .map_err(residency_display_error)?
        .total_byte_capacity,
        "speculative decoder parameters",
    )?;
    let additional_parameter_bytes = speculative_decoder_additional_parameter_tensors(
        &target_runtime_model.package.input_transducer.spec,
        decoder,
        |tensor| target_output_tensors.contains(tensor),
    )
    .into_iter()
    .try_fold(0usize, |total, tensor| {
        let byte_count = tensor_index
            .tensors
            .get(tensor)
            .and_then(|metadata| metadata.byte_count)
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(format!(
                    "speculative decoder parameter {tensor:?} has no byte count"
                ))
            })?;
        checked_residency_add(
            total,
            byte_count,
            "speculative decoder additional parameters",
        )
    })?;
    breakdown.speculative_decoder_parameter_bytes = checked_residency_add(
        breakdown.speculative_decoder_parameter_bytes,
        checked_residency_add(
            draft_parameter_bytes,
            additional_parameter_bytes,
            "speculative decoder parameter bytes",
        )?,
        "speculative decoder parameter bytes",
    )?;

    let stream =
        plan_stream_circuit_residency(&placed_plan, context_capacity_activations, true)?;
    let stream_bytes = [
        stream.state_bytes,
        stream.transaction_bytes,
        stream.activation_bytes,
        stream.boundary_bytes,
        stream.edge_bytes,
        VULKAN_STREAM_CONTROL_BYTE_CAPACITY,
    ]
    .into_iter()
    .try_fold(0usize, |total, bytes| {
        checked_residency_add(total, bytes, "speculative decoder stream bytes")
    })?;
    breakdown.speculative_decoder_stream_bytes = checked_residency_add(
        breakdown.speculative_decoder_stream_bytes,
        stream_bytes,
        "speculative decoder stream bytes",
    )?;

    let output_workspace = checked_residency_add(
        decoder.output_transducer.output_hidden_byte_capacity,
        decoder.output_transducer.logits_byte_capacity,
        "speculative decoder output workspace",
    )?;
    let sampler_workspace = sampler_workspace_bytes(
        &target_runtime_model.package.sampler.spec,
        context_capacity_activations,
        true,
    )?;
    let auxiliary_workspace = checked_residency_add(
        decoder.input_adapter.target_hidden_byte_capacity,
        checked_residency_mul(
            VULKAN_BACKEND_LOOP_MAX_WINDOW,
            VULKAN_STREAM_CONTROL_BYTE_CAPACITY,
            "speculative catch-up controls",
        )?,
        "speculative decoder auxiliary workspace",
    )?;
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
    for component in &mut circuit_graph.components {
        if matches!(
            component.runtime_role,
            CircuitRuntimeRole::DraftInputAdapter | CircuitRuntimeRole::DraftProcessor
        ) {
            component.runtime_role = CircuitRuntimeRole::SignalProcessor;
            component.circuit.runtime_role = CircuitRuntimeRole::SignalProcessor;
        }
    }
    let mut package = target.package.clone();
    package.package_id = format!("{}::{}", package.package_id, decoder.id);
    package.circuit_graph = circuit_graph.clone();
    package.component_executions = decoder.component_executions.clone();
    package.speculative_decoders.clear();
    VulkanResidentRuntimeModel {
        package,
        runtime_graph: target.runtime_graph.clone(),
        placement: StreamCircuitPlacementSpec::new(device_id),
        circuit_graph,
        component_executions: decoder.component_executions.clone(),
        tensor_index_fragments: Vec::new(),
        implementation_selection: None,
    }
}

fn speculative_decoder_additional_parameter_tensors<'a>(
    input_embedding: &'a VulkanResidentInputEmbeddingTransducerSpec,
    decoder: &'a VulkanResidentSpeculativeDecoderPackageSpec,
    mut target_has_tensor: impl FnMut(&str) -> bool,
) -> Vec<&'a str> {
    [
        input_embedding.parameter_tensor.as_str(),
        decoder.output_transducer.norm_parameter_tensor.as_str(),
        decoder
            .output_transducer
            .projection_parameter_tensor
            .as_str(),
    ]
    .into_iter()
    .filter(|tensor| !target_has_tensor(tensor))
    .collect::<BTreeSet<_>>()
    .into_iter()
    .collect()
}

fn sampler_workspace_bytes(
    spec: &VulkanResidentSamplerSpec,
    context_capacity_activations: usize,
    private_feedback_control: bool,
) -> Result<usize, VulkanRuntimeResidencyPlanError> {
    let vocabulary_size = spec.logits_byte_capacity / std::mem::size_of::<f32>();
    let token_state_is_active =
        spec.repetition_penalty != 1.0 || spec.presence_penalty != 0.0;
    let mut total = checked_residency_add(
        checked_residency_mul(
            context_capacity_activations,
            VULKAN_SAMPLER_HISTORY_RECORD_BYTE_CAPACITY,
            "sampler history",
        )?,
        spec.output_byte_capacity,
        "sampler history",
    )?;
    if spec.method == "temperature_top_k_top_p" {
        total = checked_residency_add(total, spec.scratch_byte_capacity, "sampler scratch")?;
        total = checked_residency_add(total, 4, "sampler seed")?;
    }
    let seen_token_bytes = vocabulary_size
        .div_ceil(u32::BITS as usize)
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| {
            VulkanRuntimeResidencyPlanError("sampler token-state size overflowed".to_string())
        })?;
    if token_state_is_active || spec.runtime_parameterized {
        total =
            checked_residency_add(total, seen_token_bytes, "sampler token-state bytes")?;
    }
    if token_state_is_active {
        total =
            checked_residency_add(total, seen_token_bytes, "sampler token snapshot bytes")?;
        total = checked_residency_add(
            total,
            VULKAN_BACKEND_LOOP_MAX_WINDOW * std::mem::size_of::<u32>(),
            "sampler token batch bytes",
        )?;
    }
    if spec.runtime_parameterized {
        total = checked_residency_add(total, 6 * std::mem::size_of::<u32>(), "sampler parameters")?;
    }
    if private_feedback_control {
        total = checked_residency_add(
            total,
            (VULKAN_FEEDBACK_CONTROL_HEADER_WORD_COUNT + 1) * std::mem::size_of::<u32>(),
            "sampler feedback control",
        )?;
    }
    Ok(total)
}

fn main_feedback_workspace_bytes(
    runtime_model: &VulkanResidentRuntimeModel,
    context_capacity_activations: usize,
    mount_speculative_decoders: bool,
) -> Result<usize, VulkanRuntimeResidencyPlanError> {
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
    if !mount_speculative_decoders || runtime_model.package.speculative_decoders.is_empty() {
        return Ok(control_bytes);
    }
    checked_residency_add(
        control_bytes,
        checked_residency_mul(
            runtime_model
                .package
                .output_transducer
                .spec
                .normalized_frame_byte_capacity,
            context_capacity_activations.min(VULKAN_BACKEND_LOOP_MAX_WINDOW),
            "speculative target-frame history",
        )?,
        "feedback workspace",
    )
}

fn transducer_parameter_tensors(
    resource_plan: &StreamCircuitResourcePlan,
    transducer_id: &str,
) -> BTreeSet<String> {
    resource_plan
        .transducer_parameters
        .iter()
        .filter(|parameter| {
            parameter
                .uses
                .iter()
                .any(|parameter_use| parameter_use.circuit_id == transducer_id)
        })
        .map(|parameter| parameter.tensor.clone())
        .collect()
}

fn sum_residency_breakdown(
    breakdown: &VulkanRuntimeDeviceResidencyBreakdown,
) -> Result<usize, VulkanRuntimeResidencyPlanError> {
    [
        breakdown.component_parameter_bytes,
        breakdown.transducer_parameter_bytes,
        breakdown.stream_state_bytes,
        breakdown.state_transaction_bytes,
        breakdown.activation_slot_bytes,
        breakdown.boundary_buffer_bytes,
        breakdown.edge_buffer_bytes,
        breakdown.stream_control_bytes,
        breakdown.output_transducer_workspace_bytes,
        breakdown.sampler_workspace_bytes,
        breakdown.feedback_workspace_bytes,
        breakdown.speculative_decoder_parameter_bytes,
        breakdown.speculative_decoder_stream_bytes,
        breakdown.speculative_decoder_workspace_bytes,
    ]
    .into_iter()
    .try_fold(0usize, |total, bytes| {
        checked_residency_add(total, bytes, "device residency total")
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
