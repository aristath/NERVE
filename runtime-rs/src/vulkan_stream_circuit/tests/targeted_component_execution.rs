#[test]
fn targeted_component_quanta_cover_decode_work_exactly() {
    let quanta = targeted_execution_quanta(8_194, 1).unwrap();
    assert_eq!(quanta.len(), 129);
    assert!(quanta[..128].iter().all(|repetitions| *repetitions == 64));
    assert_eq!(quanta[128], 2);
    assert_eq!(quanta.iter().sum::<usize>(), 8_194);
}

#[test]
fn targeted_component_quanta_cover_prefill_work_exactly() {
    let quanta = targeted_execution_quanta(4_096, 64).unwrap();
    assert_eq!(quanta, vec![1; 64]);
    assert_eq!(
        quanta.iter().sum::<usize>() * 64,
        4_096,
    );
}

#[test]
fn targeted_output_prefill_microbenchmark_yields_two_windows() {
    let quanta = targeted_execution_quanta(128, 4).unwrap();
    assert_eq!(quanta, vec![16, 16]);
}

#[test]
fn targeted_component_quanta_reject_partial_activation_batches() {
    let error = targeted_execution_quanta(65, 64).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("is not divisible by activation width"),
        "{error}"
    );
}

#[test]
fn targeted_component_fixture_is_deterministic_bounded_bf16() {
    let first = targeted_fixture_bytes(4_096, 17, 2);
    let repeated = targeted_fixture_bytes(4_096, 17, 2);
    let other_seed = targeted_fixture_bytes(4_096, 18, 2);
    let other_binding = targeted_fixture_bytes(4_096, 17, 3);
    assert_eq!(first, repeated);
    assert_ne!(first, other_seed);
    assert_ne!(first, other_binding);
    assert_eq!(first.len(), 4_096);
    for bytes in first.chunks_exact(2) {
        let bf16 = u16::from_le_bytes([bytes[0], bytes[1]]);
        let value = f32::from_bits(u32::from(bf16) << 16);
        assert!(value.is_finite());
        assert!(value.abs() <= 4.031_25, "{value}");
    }
}

fn targeted_test_dispatch(
    dispatch_index: usize,
    component_id: &str,
    node_id: &str,
) -> VulkanMountedPlacedBoundDispatch {
    VulkanMountedPlacedBoundDispatch {
        dispatch_index,
        kernel_id: format!("kernel_{dispatch_index}"),
        component_id: component_id.to_string(),
        circuit_id: format!("{component_id}_circuit"),
        node_index: dispatch_index,
        node_id: node_id.to_string(),
        op: "fixture".to_string(),
        reusable_family_id: format!("family_{dispatch_index}"),
        artifact_path: format!("shader_{dispatch_index}.spv"),
        entry_point: "main".to_string(),
        local_size_x: 1,
        descriptors: Vec::new(),
        push_constants: Vec::new(),
        stream_control_binding: None,
    }
}

fn targeted_test_bound_plan(
    dispatches: Vec<VulkanMountedPlacedBoundDispatch>,
) -> VulkanMountedPlacedBoundDispatchPlan {
    VulkanMountedPlacedBoundDispatchPlan {
        backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
        device_id: "amd0".to_string(),
        dispatches,
        total_descriptor_count: 0,
        resident_descriptor_count: 0,
        model_boundary_descriptor_count: 0,
        local_edge_descriptor_count: 0,
        edge_endpoint_descriptor_count: 0,
        incoming_edge_descriptor_count: 0,
        outgoing_edge_descriptor_count: 0,
    }
}

#[test]
fn targeted_component_prefix_executes_only_the_real_causal_prefix() {
    let first = targeted_test_dispatch(2, "layer_00", "first");
    let foreign = targeted_test_dispatch(3, "layer_01", "foreign");
    let target = targeted_test_dispatch(4, "layer_00", "target");
    let later = targeted_test_dispatch(5, "layer_00", "later");
    let plan = targeted_test_bound_plan(vec![
        first.clone(),
        foreign,
        target.clone(),
        later,
    ]);

    let prefix = targeted_component_execution_dispatches(
        &plan,
        "layer_00",
        &target,
        VulkanTargetedComponentExecutionPhase::Decode,
        VulkanTargetedComponentExecutionScope::DecodeComponentPrefix,
    )
    .unwrap();

    assert_eq!(
        prefix
            .iter()
            .map(|dispatch| dispatch.node_id.as_str())
            .collect::<Vec<_>>(),
        vec!["first", "target"],
    );
}

#[test]
fn targeted_component_prefix_rejects_prefill_instead_of_faking_internal_inputs() {
    let target = targeted_test_dispatch(1, "layer_00", "target");
    let plan = targeted_test_bound_plan(vec![target.clone()]);

    let error = targeted_component_execution_dispatches(
        &plan,
        "layer_00",
        &target,
        VulkanTargetedComponentExecutionPhase::Prefill {
            activation_batch_width: 64,
        },
        VulkanTargetedComponentExecutionScope::DecodeComponentPrefix,
    )
    .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("component-prefix targeted execution requires decode phase"),
        "{error}",
    );
}

#[test]
fn targeted_capture_uses_the_logical_signal_extent_not_its_reused_slot() {
    let descriptor = VulkanMountedPlacedBoundDescriptor {
        binding: 0,
        usage: VulkanKernelDescriptorUsage::OutputSignal,
        name: "quantized_activation".to_string(),
        target: VulkanMountedPlacedBoundDescriptorTarget::Resident {
            target: VulkanBoundDescriptorTarget::ActivationSlot {
                buffer_index: 3,
                component_id: "layer_00".to_string(),
                signal_id: "quantized_activation".to_string(),
                circuit_id: "layer_00_circuit".to_string(),
                slot: 4,
                byte_capacity: 32_768,
                signal_byte_capacity: 4_096,
            },
        },
    };

    assert_eq!(targeted_signal_byte_count(&descriptor, 32_768), 4_096);
}

#[test]
fn targeted_fixture_only_mutates_true_signal_buffers() {
    let input = VulkanMountedPlacedBoundDescriptor {
        binding: 1,
        usage: VulkanKernelDescriptorUsage::InputSignal,
        name: "local_kv_memory".to_string(),
        target: VulkanMountedPlacedBoundDescriptorTarget::Resident {
            target: VulkanBoundDescriptorTarget::StreamStateBuffer {
                buffer_index: 0,
                component_id: "layer_00".to_string(),
                state_id: "local_kv_memory".to_string(),
                state_type: "rolling_attention_memory".to_string(),
                byte_capacity: 131_328,
                static_bytes: None,
                bytes_per_activation: Some(1_024),
            },
        },
    };
    let output = VulkanMountedPlacedBoundDescriptor {
        binding: 2,
        usage: VulkanKernelDescriptorUsage::OutputSignal,
        name: "local_kv_values".to_string(),
        target: VulkanMountedPlacedBoundDescriptorTarget::Resident {
            target: VulkanBoundDescriptorTarget::StreamStateView {
                buffer_index: 0,
                component_id: "layer_00".to_string(),
                state_id: "local_kv_memory".to_string(),
                state_type: "rolling_attention_memory".to_string(),
                byte_capacity: 131_328,
                static_bytes: None,
                bytes_per_activation: Some(1_024),
            },
        },
    };
    let runtime_control = VulkanMountedPlacedBoundDescriptor {
        binding: 1,
        usage: VulkanKernelDescriptorUsage::InputSignal,
        name: "token_id".to_string(),
        target: VulkanMountedPlacedBoundDescriptorTarget::Resident {
            target: VulkanBoundDescriptorTarget::RuntimeControl {
                runtime_source: "input_token_id".to_string(),
                byte_capacity: 4,
            },
        },
    };
    let activation = VulkanMountedPlacedBoundDescriptor {
        binding: 0,
        usage: VulkanKernelDescriptorUsage::InputSignal,
        name: "operator_input".to_string(),
        target: VulkanMountedPlacedBoundDescriptorTarget::Resident {
            target: VulkanBoundDescriptorTarget::ActivationSlot {
                buffer_index: 0,
                component_id: "layer_00".to_string(),
                signal_id: "operator_input".to_string(),
                circuit_id: "layer_00_circuit".to_string(),
                slot: 0,
                byte_capacity: 32_768,
                signal_byte_capacity: 8_192,
            },
        },
    };

    assert!(!targeted_signal_accepts_fixture_mutation(&input));
    assert!(!targeted_signal_accepts_fixture_mutation(&output));
    assert!(!targeted_signal_accepts_fixture_mutation(&runtime_control));
    assert!(targeted_signal_accepts_fixture_mutation(&activation));
}

#[test]
fn targeted_output_identity_remains_an_artifact_digest() {
    let digest = targeted_finalized_artifact_digest(&[0xAB; 32]);

    assert_eq!(
        digest,
        format!("nerve.optimizer.artifact_sha256.v1:{}", "ab".repeat(32))
    );
}

#[test]
fn targeted_prefill_accepts_truthful_stateless_causal_scan_metadata() {
    assert!(targeted_prefill_batch_mode_is_supported(
        VulkanResidentComponentKernelBatchMode::CausalScan,
    ));
    assert!(targeted_prefill_batch_mode_is_supported(
        VulkanResidentComponentKernelBatchMode::WeightShared,
    ));
    assert!(!targeted_prefill_batch_mode_is_supported(
        VulkanResidentComponentKernelBatchMode::SerialLanes,
    ));
}
