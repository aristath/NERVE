#[test]
fn targeted_component_quanta_cover_decode_work_exactly() {
    let quanta = targeted_execution_quanta(8_194, 1, 2).unwrap();
    assert_eq!(quanta.len(), 129);
    assert!(quanta[..128].iter().all(|repetitions| *repetitions == 64));
    assert_eq!(quanta[128], 2);
    assert_eq!(quanta.iter().sum::<usize>(), 8_194);
}

#[test]
fn targeted_component_quanta_cover_prefill_work_exactly() {
    let quanta = targeted_execution_quanta(4_096, 64, 2).unwrap();
    assert_eq!(quanta, vec![1; 64]);
    assert_eq!(
        quanta.iter().sum::<usize>() * 64,
        4_096,
    );
}

#[test]
fn targeted_output_prefill_microbenchmark_yields_two_windows() {
    let quanta = targeted_execution_quanta(128, 4, 2).unwrap();
    assert_eq!(quanta, vec![16, 16]);
}

#[test]
fn targeted_component_quanta_honor_requested_sustained_windows() {
    let quanta = targeted_execution_quanta(2, 1, 2).unwrap();
    assert_eq!(quanta, vec![1, 1]);
}

#[test]
fn targeted_component_quanta_reject_more_windows_than_activation_batches() {
    let error = targeted_execution_quanta(1, 1, 2).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("sustained windows exceed useful activation batches"),
        "{error}"
    );
}

#[test]
fn targeted_component_quanta_reject_partial_activation_batches() {
    let error = targeted_execution_quanta(65, 64, 1).unwrap_err();
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

#[test]
fn targeted_state_fixture_preserves_declared_numeric_type() {
    let bf16 = targeted_state_fixture_bytes(32, 17, 2, "BF16").unwrap();
    let f32 = targeted_state_fixture_bytes(64, 17, 2, "F32").unwrap();
    let u32s = targeted_state_fixture_bytes(64, 17, 2, "U32").unwrap();

    assert_eq!(bf16, targeted_state_fixture_bytes(32, 17, 2, "BF16").unwrap());
    assert_ne!(bf16, vec![0; bf16.len()]);
    assert!(bf16.chunks_exact(2).all(|bytes| {
        f32::from_bits(u32::from(u16::from_le_bytes(bytes.try_into().unwrap())) << 16).is_finite()
    }));
    assert_ne!(f32, vec![0; f32.len()]);
    assert!(f32.chunks_exact(4).all(|bytes| {
        f32::from_le_bytes(bytes.try_into().unwrap()).is_finite()
    }));
    assert!(u32s.chunks_exact(4).all(|bytes| {
        u32::from_le_bytes(bytes.try_into().unwrap()) < 1_024
    }));
    assert!(
        targeted_state_fixture_bytes(3, 17, 2, "BF16")
            .unwrap_err()
            .to_string()
            .contains("aligned")
    );
    assert!(
        targeted_state_fixture_bytes(16, 17, 2, "unknown")
            .unwrap_err()
            .to_string()
            .contains("unsupported state dtype")
    );
}

#[test]
fn targeted_causal_fixture_uses_declared_nonempty_history() {
    assert_eq!(
        targeted_prefill_start_stream_tick(
            VulkanResidentComponentKernelBatchMode::WeightShared,
            3,
            4_096,
        )
        .unwrap(),
        0,
    );
    assert_eq!(
        targeted_prefill_start_stream_tick(
            VulkanResidentComponentKernelBatchMode::CausalScan,
            3,
            4_096,
        )
        .unwrap(),
        4_093,
    );
    assert_eq!(
        targeted_prefill_start_stream_tick(
            VulkanResidentComponentKernelBatchMode::CausalScan,
            3,
            8,
        )
        .unwrap(),
        5,
    );
    assert!(
        targeted_prefill_start_stream_tick(
            VulkanResidentComponentKernelBatchMode::CausalScan,
            9,
            8,
        )
        .unwrap_err()
        .to_string()
        .contains("exceeds dynamic-state capacity")
    );
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
        None,
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
        None,
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
fn targeted_component_prefix_completes_the_containing_residency_checkpoint() {
    let before = targeted_test_dispatch(9, "layer_00", "before");
    let select = targeted_test_dispatch(10, "layer_00", "select");
    let compute_a = targeted_test_dispatch(11, "layer_00", "compute_a");
    let target = targeted_test_dispatch(12, "layer_00", "compute_target");
    let compute_b = targeted_test_dispatch(13, "layer_00", "compute_b");
    let continuation = targeted_test_dispatch(14, "layer_00", "reduce");
    let after = targeted_test_dispatch(15, "layer_00", "after");
    let plan = targeted_test_bound_plan(vec![
        before,
        select,
        compute_a,
        target.clone(),
        compute_b,
        continuation,
        after,
    ]);
    let schedule = VulkanPhysicalResidencySchedule {
        execution_scope: "layer_00".to_string(),
        checkpoints: vec![VulkanPhysicalResidencyCheckpoint {
            id: "experts".to_string(),
            execution_scope: "layer_00".to_string(),
            component_id: "layer_00".to_string(),
            selector_ids: vec!["routes".to_string()],
            selection_dispatch_index: 10,
            selected_computation_dispatch_indices: vec![11, 12, 13],
            selected_result_continuation_dispatch_index: Some(14),
        }],
    };

    let prefix = targeted_component_execution_dispatches(
        &plan,
        "layer_00",
        &target,
        VulkanTargetedComponentExecutionPhase::Decode,
        VulkanTargetedComponentExecutionScope::DecodeComponentPrefix,
        Some(&schedule),
    )
    .unwrap();

    assert_eq!(
        prefix
            .iter()
            .map(|dispatch| dispatch.node_id.as_str())
            .collect::<Vec<_>>(),
        vec![
            "before",
            "select",
            "compute_a",
            "compute_target",
            "compute_b",
            "reduce",
        ],
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
fn targeted_demand_mount_owns_only_its_component_selectors() {
    let selector = |id: &str, scope: &str, component: &str| {
        CompiledResourceSelector {
            id: id.to_string(),
            execution_scope: scope.to_string(),
            component_id: component.to_string(),
            node_id: "route".to_string(),
            domain_id: "experts".to_string(),
            resource_count: 256,
            selection_signal: "selected_experts".to_string(),
            execution_signal: "selected_experts".to_string(),
            execution_calibration_word_base: 0,
            encoding: CompiledResourceSelectionEncoding {
                element_type: CompiledResourceSelectionElementType::U32,
                selection_count_per_activation: 6,
                index_shift: 0,
                index_mask: 0xffff,
                calibration_word_base: 0,
            },
            mapping: CompiledResourceSelectorMapping::GroupTable {
                atomic_group_ids: Vec::new(),
            },
        }
    };
    let selectors = vec![
        selector("layer-0", "target", "layer_00"),
        selector("layer-1", "target", "layer_01"),
        selector("draft-layer-0", "draft:0", "layer_00"),
    ];

    assert_eq!(
        targeted_demand_selector_ids(&selectors, "target", "layer_00"),
        BTreeSet::from(["layer-0".to_string()]),
    );
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
fn targeted_prefill_accepts_truthful_causal_scan_metadata() {
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
