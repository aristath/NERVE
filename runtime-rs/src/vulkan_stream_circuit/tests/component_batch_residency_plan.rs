fn fixture_component_batch_residency_plans(
) -> (VulkanPlacedStreamCircuitPlan, VulkanPreparedDispatchPlan) {
    let runtime_model = fixture_model_runtime_model();
    let tensor_index = TensorIndex::from_json_file(fixture_model_tensor_index_path()).unwrap();
    let device_id = runtime_model.placement.default_device_id.clone();
    let (_, _, placed_plan) = plan_resident_package_placed_stream_circuit_with_tensor_index(
        &device_id,
        &runtime_model.placement,
        &runtime_model.circuit_graph,
        &tiny_model_dir(),
        &tensor_index,
        runtime_model.package.activation_element_bytes,
    )
    .unwrap();
    let manifest = resident_package_reusable_kernel_manifest(&placed_plan);
    let prepared_plan = placed_plan
        .prepared_dispatch_plan(&manifest, 128)
        .unwrap();
    (placed_plan, prepared_plan)
}

fn fixture_component_batch_demand_residency_plans() -> (
    VulkanPlacedStreamCircuitPlan,
    VulkanPreparedDispatchPlan,
    CompiledResourceResidencyContract,
    VulkanCompiledResourceAddressLayout,
    VulkanPhysicalResidencySchedule,
) {
    let runtime_model = fixture_model_runtime_model_with_one_dynamic_group();
    let contract = instantiate_runtime_resource_contract(&runtime_model).unwrap();
    let tensor_index = TensorIndex::from_json_file(fixture_model_tensor_index_path()).unwrap();
    let device_id = runtime_model.placement.default_device_id.clone();
    let (_, _, placed_plan) = plan_resident_package_placed_stream_circuit_with_tensor_index(
        &device_id,
        &runtime_model.placement,
        &runtime_model.circuit_graph,
        &tiny_model_dir(),
        &tensor_index,
        runtime_model.package.activation_element_bytes,
    )
    .unwrap();
    let manifest = resident_package_reusable_kernel_manifest(&placed_plan);
    let prepared_plan = placed_plan.prepared_dispatch_plan(&manifest, 128).unwrap();
    let layout = VulkanCompiledResourceAddressLayout::from_contract(&contract).unwrap();
    let schedule = VulkanPhysicalResidencySchedule::from_prepared_dispatch_plan(
        &contract,
        runtime_model.execution_scope,
        &prepared_plan,
    )
    .unwrap();
    (placed_plan, prepared_plan, contract, layout, schedule)
}

#[test]
fn component_batch_residency_plan_names_every_scalar_fallback_allocation() {
    let (placed_plan, prepared_plan) = fixture_component_batch_residency_plans();
    let lane_capacity = 4;
    let plan = VulkanComponentBatchResidentAllocationPlan::for_single_device(
        &placed_plan,
        &prepared_plan,
        &[],
        lane_capacity,
        VulkanComponentBatchExecutionMode::CausalSequence,
        &VulkanComponentBatchExecutionScope::all(),
        &BTreeSet::new(),
        false,
        None,
    )
    .unwrap();

    let signal_bytes = plan
        .signal_buffer_plan
        .iter()
        .map(|signal| signal.frame_byte_capacity * lane_capacity)
        .sum::<usize>();
    let control_bytes = component_batch_control_payloads()
        .into_iter()
        .map(|payload| payload.byte_count() as usize)
        .sum::<usize>();
    assert_eq!(
        plan.total_byte_capacity,
        signal_bytes
            + lane_capacity * VULKAN_STREAM_CONTROL_BYTE_CAPACITY
            + lane_capacity * size_of::<u32>()
            + control_bytes
            + size_of::<u32>(),
    );
    assert_eq!(
        plan.allocations
            .iter()
            .filter(|allocation| matches!(
                allocation.kind,
                VulkanComponentBatchResidentAllocationKind::SignalBuffer { .. }
            ))
            .count(),
        plan.signal_buffer_plan.len(),
    );
    assert_eq!(
        plan.allocations
            .iter()
            .filter(|allocation| matches!(
                allocation.kind,
                VulkanComponentBatchResidentAllocationKind::StreamControl { .. }
            ))
            .count(),
        lane_capacity,
    );
    assert!(plan.allocations.iter().any(|allocation| matches!(
        allocation.kind,
        VulkanComponentBatchResidentAllocationKind::CausalStateSnapshotDummy
    )));
    assert!(!plan.allocations.iter().any(|allocation| matches!(
        allocation.kind,
        VulkanComponentBatchResidentAllocationKind::CausalStateSnapshot { .. }
            | VulkanComponentBatchResidentAllocationKind::DemandPipelinePredicate
    )));
}

#[test]
fn component_batch_residency_plan_tracks_selected_snapshot_state_exactly() {
    let (mut placed_plan, mut prepared_plan) = fixture_component_batch_residency_plans();
    let static_byte_capacity = 64;
    placed_plan
        .placed_resident_plan
        .resident_plan
        .stream_state_buffers
        .push(VulkanResidentStateBuffer {
            component_id: "snapshot_component".to_string(),
            state_id: "snapshot_state".to_string(),
            state_type: "recurrent".to_string(),
            dtype: Some("U32".to_string()),
            layout: Some("flat".to_string()),
            static_elements: Some(static_byte_capacity / size_of::<u32>()),
            elements_per_activation: None,
            max_dynamic_activations: None,
            static_bytes: Some(static_byte_capacity),
            bytes_per_activation: None,
            clone_from: None,
        });
    let dispatch = prepared_plan
        .dispatches
        .iter_mut()
        .find(|dispatch| dispatch.push_constants.is_empty())
        .unwrap();
    let source_binding = dispatch.descriptors.len();
    dispatch.descriptors.push(VulkanResolvedDescriptorBinding {
        binding: source_binding,
        usage: VulkanKernelDescriptorUsage::StateRead,
        name: "snapshot_state".to_string(),
        resource: VulkanDescriptorResourceAddress::StateBuffer {
            component_id: "snapshot_component".to_string(),
            state_id: "snapshot_state".to_string(),
            state_type: "recurrent".to_string(),
            byte_capacity: static_byte_capacity,
            static_bytes: Some(static_byte_capacity),
            bytes_per_activation: None,
        },
    });
    let component_id = dispatch.component_id.clone();
    let node_id = dispatch.node_id.clone();
    let artifact = VulkanResidentComponentBatchKernelArtifact {
        component_id: component_id.clone(),
        node_id: node_id.clone(),
        execution_domain: VulkanResidentComponentKernelExecutionDomain::Prefill,
        batch_mode: VulkanResidentComponentKernelBatchMode::CausalScan,
        lane_tile_width: 8,
        selection_priority: 0,
        independent_candidate_compatible: false,
        causal_sequence_compatible: true,
        parallel_block_compatible: false,
        device_requirements: VulkanResidentVulkanDeviceRequirements::default(),
        stages: vec![VulkanResidentComponentBatchStageArtifact {
            shader_path: "fixture-snapshot.spv".to_string(),
            spirv_words: Vec::new(),
            local_size_x: 64,
            workgroup_count_x: 1,
            descriptor_bindings: Vec::new(),
            state_snapshot_binding: Some(31),
            state_snapshot_source_binding: Some(source_binding as u32),
            control: VulkanResidentComponentBatchControlSpec::StorageBuffer {
                byte_count: VulkanResidentComponentBatchControlPayload::TemporalStateSnapshots
                    .byte_count(),
                binding: 30,
                payload: VulkanResidentComponentBatchControlPayload::TemporalStateSnapshots,
                access: VulkanResidentComponentBatchControlAccess::Read,
            },
            indirect_dispatch_byte_offset: None,
            dispatch_y_from_batch_width: false,
        }],
    };
    let scope = VulkanComponentBatchExecutionScope::nodes(BTreeMap::from([(
        component_id,
        BTreeSet::from([node_id]),
    )]))
    .unwrap();
    let lane_capacity = 4;
    let plan = VulkanComponentBatchResidentAllocationPlan::for_single_device(
        &placed_plan,
        &prepared_plan,
        &[artifact],
        lane_capacity,
        VulkanComponentBatchExecutionMode::CausalSequence,
        &scope,
        &BTreeSet::new(),
        false,
        None,
    )
    .unwrap();

    let snapshots = plan
        .allocations
        .iter()
        .filter(|allocation| matches!(
            allocation.kind,
            VulkanComponentBatchResidentAllocationKind::CausalStateSnapshot { .. }
        ))
        .collect::<Vec<_>>();
    assert_eq!(snapshots.len(), 1);
    assert_eq!(
        snapshots[0].byte_capacity,
        static_byte_capacity * lane_capacity
    );
    assert!(!plan.allocations.iter().any(|allocation| matches!(
        allocation.kind,
        VulkanComponentBatchResidentAllocationKind::DemandPipelinePredicate
    )));
}

#[test]
fn component_batch_demand_segment_plan_accounts_exact_dspark_gate_geometry() {
    let lane_capacity = 7;
    let geometries = vec![VulkanComponentBatchDemandGateGeometry {
        checkpoint_id: "draft_state_ingestion".to_string(),
        selector_id: "expert_selector".to_string(),
        selection_count_per_activation: 6,
        selection_index_shift: 0,
        selection_index_mask: 0xffff,
        address_mapping: VulkanCompiledSelectorAddressMapping::GroupTable {
            resource_address_slots: vec![0, 1, 2, 3],
            resource_address_slot_offsets: vec![0, 2, 4],
        },
    }];

    let owned = component_batch_demand_segment_allocations(
        &geometries,
        lane_capacity,
        true,
    )
    .unwrap();
    let miss_queue = owned
        .iter()
        .find(|allocation| {
            allocation.kind == VulkanComponentBatchResidentAllocationKind::DemandMissQueue
        })
        .unwrap();
    assert_eq!(miss_queue.byte_capacity, 352);
    assert!(miss_queue.host_visible);
    assert_eq!(
        owned
            .iter()
            .filter(|allocation| matches!(
                allocation.kind,
                VulkanComponentBatchResidentAllocationKind::DemandPipelinePredicate
            ))
            .count(),
        1,
    );
    assert_eq!(
        owned
            .iter()
            .filter(|allocation| matches!(
                allocation.kind,
                VulkanComponentBatchResidentAllocationKind::DemandGateConfiguration { .. }
                    | VulkanComponentBatchResidentAllocationKind::DemandGateResourceGroups { .. }
                    | VulkanComponentBatchResidentAllocationKind::DemandGateAddressSlots { .. }
                    | VulkanComponentBatchResidentAllocationKind::DemandGateResolvedAddresses { .. }
            ))
            .count(),
        4,
    );

    let shared_predicate = component_batch_demand_segment_allocations(
        &geometries,
        lane_capacity,
        false,
    )
    .unwrap();
    assert_eq!(shared_predicate.len() + 1, owned.len());
    assert_eq!(
        shared_predicate.iter().map(|allocation| allocation.byte_capacity).sum::<usize>()
            + size_of::<u32>(),
        owned.iter().map(|allocation| allocation.byte_capacity).sum::<usize>(),
    );
}

#[test]
fn component_batch_demand_plan_rejects_checkpoint_crossing_selected_scope() {
    let (placed_plan, prepared_plan, contract, layout, schedule) =
        fixture_component_batch_demand_residency_plans();
    let selector = contract.selectors.first().unwrap();
    let demand_context = VulkanComponentBatchDemandResidencyPlanContext {
        schedule: &schedule,
        contract: &contract,
        layout: &layout,
    };
    let complete = VulkanComponentBatchResidentAllocationPlan::for_single_device(
        &placed_plan,
        &prepared_plan,
        &[],
        7,
        VulkanComponentBatchExecutionMode::CausalSequence,
        &VulkanComponentBatchExecutionScope::all(),
        &BTreeSet::new(),
        false,
        Some(demand_context),
    )
    .unwrap();
    assert!(complete.allocations.iter().any(|allocation| matches!(
        allocation.kind,
        VulkanComponentBatchResidentAllocationKind::DemandMissQueue
    )));
    assert!(complete.allocations.iter().any(|allocation| matches!(
        allocation.kind,
        VulkanComponentBatchResidentAllocationKind::DemandGateResolvedAddresses { .. }
    )));
    let selection_only_scope = VulkanComponentBatchExecutionScope::nodes(BTreeMap::from([(
        selector.component_id.clone(),
        BTreeSet::from([selector.node_id.clone()]),
    )]))
    .unwrap();

    let error = VulkanComponentBatchResidentAllocationPlan::for_single_device(
        &placed_plan,
        &prepared_plan,
        &[],
        7,
        VulkanComponentBatchExecutionMode::CausalSequence,
        &selection_only_scope,
        &BTreeSet::new(),
        false,
        Some(demand_context),
    )
    .unwrap_err();

    assert!(error.0.contains("selected computation crosses the scope boundary"));
}

#[test]
fn component_batch_residency_plan_rejects_missing_snapshot_source_binding() {
    let (placed_plan, prepared_plan) = fixture_component_batch_residency_plans();
    let dispatch = prepared_plan
        .dispatches
        .iter()
        .find(|dispatch| dispatch.push_constants.is_empty())
        .unwrap();
    let artifact = VulkanResidentComponentBatchKernelArtifact {
        component_id: dispatch.component_id.clone(),
        node_id: dispatch.node_id.clone(),
        execution_domain: VulkanResidentComponentKernelExecutionDomain::Prefill,
        batch_mode: VulkanResidentComponentKernelBatchMode::CausalScan,
        lane_tile_width: 8,
        selection_priority: 0,
        independent_candidate_compatible: false,
        causal_sequence_compatible: true,
        parallel_block_compatible: false,
        device_requirements: VulkanResidentVulkanDeviceRequirements::default(),
        stages: vec![VulkanResidentComponentBatchStageArtifact {
            shader_path: "fixture-invalid-snapshot.spv".to_string(),
            spirv_words: Vec::new(),
            local_size_x: 64,
            workgroup_count_x: 1,
            descriptor_bindings: Vec::new(),
            state_snapshot_binding: Some(31),
            state_snapshot_source_binding: Some(u32::MAX),
            control: VulkanResidentComponentBatchControlSpec::StorageBuffer {
                byte_count: VulkanResidentComponentBatchControlPayload::TemporalStateSnapshots
                    .byte_count(),
                binding: 30,
                payload: VulkanResidentComponentBatchControlPayload::TemporalStateSnapshots,
                access: VulkanResidentComponentBatchControlAccess::Read,
            },
            indirect_dispatch_byte_offset: None,
            dispatch_y_from_batch_width: false,
        }],
    };
    let scope = VulkanComponentBatchExecutionScope::nodes(BTreeMap::from([(
        dispatch.component_id.clone(),
        BTreeSet::from([dispatch.node_id.clone()]),
    )]))
    .unwrap();

    let error = VulkanComponentBatchResidentAllocationPlan::for_single_device(
        &placed_plan,
        &prepared_plan,
        &[artifact],
        4,
        VulkanComponentBatchExecutionMode::CausalSequence,
        &scope,
        &BTreeSet::new(),
        false,
        None,
    )
    .unwrap_err();

    assert!(error.0.contains("absent descriptor binding"));
}
