#[test]
fn runtime_residency_plan_uses_physical_transient_layout_without_opening_vulkan() {
    let runtime_model = fixture_model_runtime_model();
    let package_root = tiny_model_dir();
    let tensor_index = runtime_model
        .load_runtime_tensor_index(&package_root)
        .unwrap();

    let short = plan_vulkan_runtime_residency(
        &package_root,
        &runtime_model,
        &tensor_index,
        16,
        0,
        ResourceResidencyPolicy::Eager,
    )
    .unwrap();
    let long = plan_vulkan_runtime_residency(
        &package_root,
        &runtime_model,
        &tensor_index,
        64,
        0,
        ResourceResidencyPolicy::Eager,
    )
    .unwrap();

    assert_eq!(short.schema, VULKAN_RUNTIME_RESIDENCY_PLAN_SCHEMA);
    assert_eq!(short.device_plans.len(), 1);
    assert_eq!(long.device_plans.len(), 1);
    let short_device = &short.device_plans[0];
    let long_device = &long.device_plans[0];
    assert!(
        short_device
            .parameter_residency
            .always_resident_bytes
            > 0
    );
    assert_eq!(
        short_device
            .parameter_residency
            .current_resident_bytes,
        short_device
            .parameter_residency
            .maximum_addressable_bytes,
    );
    assert!(
        long_device.breakdown.stream_state_bytes
            > short_device.breakdown.stream_state_bytes
    );
    assert!(
        long_device.initial_device_resident_bytes
            > short_device.initial_device_resident_bytes
    );
    assert_eq!(
        short.total_initial_device_resident_bytes,
        short_device.initial_device_resident_bytes
    );
    assert_eq!(
        short_device
            .parameter_residency
            .current_resident_bytes
            + short_device.resource_store.maximum_extra_device_bytes().unwrap()
            + short_device.working_set.transient_state_bytes
            + short_device.working_set.activation_headroom_bytes,
        short_device.initial_device_resident_bytes
    );
    let ledger_bytes = short_device
        .resident_stream_device_allocations
        .iter()
        .map(|allocation| allocation.byte_capacity)
        .sum::<usize>();
    let covered_breakdown_bytes = [
        short_device.breakdown.stream_state_bytes,
        short_device.breakdown.state_transaction_bytes,
        short_device.breakdown.causal_verification_snapshot_bytes,
        short_device.breakdown.selection_telemetry_bytes,
        short_device.breakdown.activation_slot_bytes,
        short_device.breakdown.boundary_buffer_bytes,
        short_device.breakdown.edge_buffer_bytes,
        short_device.breakdown.output_transducer_workspace_bytes,
        short_device.breakdown.sampler_workspace_bytes,
        short_device.breakdown.feedback_workspace_bytes,
    ]
    .into_iter()
    .sum::<usize>();
    assert_eq!(ledger_bytes, covered_breakdown_bytes);
}

#[test]
fn runtime_residency_preserves_output_sampler_and_feedback_allocation_identities() {
    let runtime_model = fixture_model_runtime_model();
    let package_root = tiny_model_dir();
    let tensor_index = runtime_model
        .load_runtime_tensor_index(&package_root)
        .unwrap();
    let plan = plan_vulkan_runtime_residency(
        &package_root,
        &runtime_model,
        &tensor_index,
        16,
        0,
        ResourceResidencyPolicy::Eager,
    )
    .unwrap();
    let device = &plan.device_plans[0];
    let runtime_buffers = device
        .resident_stream_device_allocations
        .iter()
        .filter_map(|allocation| match &allocation.kind {
            VulkanRuntimeResidentStreamAllocationKind::RuntimeBuffer {
                class,
                buffer_id,
                ..
            } => Some((*class, buffer_id.as_str(), allocation.byte_capacity)),
            _ => None,
        })
        .collect::<Vec<_>>();

    assert!(runtime_buffers.contains(&(
        VulkanRuntimeResidentBufferClass::OutputTransducerWorkspace,
        "normalized_frame",
        runtime_model
            .package
            .output_transducer
            .spec
            .normalized_frame_byte_capacity,
    )));
    assert!(runtime_buffers.contains(&(
        VulkanRuntimeResidentBufferClass::OutputTransducerWorkspace,
        "logits",
        runtime_model.package.output_transducer.spec.logits_byte_capacity,
    )));
    assert!(runtime_buffers.iter().any(|(class, buffer_id, _)| {
        *class == VulkanRuntimeResidentBufferClass::SamplerWorkspace
            && *buffer_id == "history_and_output"
    }));
    assert!(runtime_buffers.iter().any(|(class, buffer_id, _)| {
        *class == VulkanRuntimeResidentBufferClass::FeedbackWorkspace
            && *buffer_id == "control"
    }));
    for (class, expected) in [
        (
            VulkanRuntimeResidentBufferClass::OutputTransducerWorkspace,
            device.breakdown.output_transducer_workspace_bytes,
        ),
        (
            VulkanRuntimeResidentBufferClass::SamplerWorkspace,
            device.breakdown.sampler_workspace_bytes,
        ),
        (
            VulkanRuntimeResidentBufferClass::FeedbackWorkspace,
            device.breakdown.feedback_workspace_bytes,
        ),
    ] {
        assert_eq!(
            runtime_buffers
                .iter()
                .filter(|(candidate, _, _)| *candidate == class)
                .map(|(_, _, bytes)| *bytes)
                .sum::<usize>(),
            expected,
        );
    }
}

#[test]
fn sampler_residency_tracks_random_scratch_and_seed_for_every_random_method() {
    let mut spec = fixture_model_runtime_model().package.sampler.spec;
    spec.method = "temperature_top_p".to_string();
    spec.scratch_byte_capacity = 4_096;
    let random = sampler_workspace_allocations(&spec, 16, false).unwrap();
    let random_ids = random
        .iter()
        .filter_map(|allocation| match &allocation.kind {
            VulkanRuntimeResidentStreamAllocationKind::RuntimeBuffer { buffer_id, .. } => {
                Some(buffer_id.as_str())
            }
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    assert!(random_ids.contains("scratch"));
    assert!(random_ids.contains("random_seed"));

    spec.method = "greedy".to_string();
    let greedy = sampler_workspace_allocations(&spec, 16, false).unwrap();
    assert!(greedy.iter().all(|allocation| {
        !matches!(
            &allocation.kind,
            VulkanRuntimeResidentStreamAllocationKind::RuntimeBuffer { buffer_id, .. }
                if buffer_id == "scratch" || buffer_id == "random_seed"
        )
    }));
}

#[test]
fn runtime_residency_accounts_selection_telemetry_as_a_physical_allocation() {
    let runtime_model = fixture_model_runtime_model_with_remote_middle();
    let graph = runtime_model
        .circuit_graph
        .to_signal_processor_graph(tiny_model_dir())
        .unwrap();
    let tensor_index = TensorIndex::from_json_file(fixture_model_tensor_index_path()).unwrap();
    let execution_plan =
        StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index).unwrap();
    let mut resource_plan =
        StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
    resource_plan
        .selection_domains
        .push(crate::stream_plan::PlannedSelectionDomain {
            component_id: "layer_00".to_string(),
            circuit_id: "layer_00".to_string(),
            node_id: "selector".to_string(),
            domain_id: "experts".to_string(),
            resource_count: 256,
            selection_count_per_activation: 8,
            predictable_dependency: None,
        });
    let placement_plan = graph.placement_plan(&runtime_model.placement).unwrap();
    let resident = VulkanPlacedStreamCircuitResidentPlan::from_resource_plan_for_device(
        &resource_plan,
        &placement_plan,
        "gpu0",
        Some(&tensor_index),
        Some(2),
    )
    .unwrap();
    let placed =
        VulkanPlacedStreamCircuitPlan::from_plans(&execution_plan, &resource_plan, resident)
            .unwrap();

    let residency = plan_stream_circuit_residency(&placed, 16, true, 0).unwrap();
    let telemetry = residency
        .allocations
        .iter()
        .filter(|allocation| {
            matches!(
                &allocation.kind,
                VulkanRuntimeResidentStreamAllocationKind::SelectionTelemetry { .. }
            )
        })
        .collect::<Vec<_>>();

    assert_eq!(telemetry.len(), 1);
    assert_eq!(residency.selection_telemetry_bytes, telemetry[0].byte_capacity);
    assert!(residency.selection_telemetry_bytes > 0);
}

#[test]
fn resident_edge_ledger_allocates_fanout_once_and_honors_boundary_aliases() {
    let plan = mixed_fanout_edge_plan();
    let (bytes, allocations) = plan_edge_residency_allocations_with_passthrough(
        &plan,
        &BTreeSet::new(),
        None,
    )
    .unwrap();

    assert_eq!(bytes, 8_192);
    assert_eq!(allocations.len(), 1);
    assert_eq!(allocations[0].byte_capacity, 8_192);
    assert_eq!(
        allocations[0].kind,
        VulkanRuntimeResidentStreamAllocationKind::EdgeProducedPort {
            component_id: "input_adapter".to_string(),
            port_id: "shared_context".to_string(),
            edge_indices: vec![4, 5, 6],
        }
    );

    let (aliased_bytes, aliased) = plan_edge_residency_allocations_with_passthrough(
        &plan,
        &BTreeSet::from([(
            "input_adapter".to_string(),
            "shared_context".to_string(),
        )]),
        None,
    )
    .unwrap();
    assert_eq!(aliased_bytes, 0);
    assert!(aliased.is_empty());

    let mut incompatible = plan;
    incompatible.local_edges[0].byte_capacity = Some(4_096);
    let error = plan_edge_residency_allocations_with_passthrough(
        &incompatible,
        &BTreeSet::new(),
        None,
    )
    .unwrap_err();
    assert!(error.to_string().contains("incompatible capacities"));
}

#[test]
fn resident_edge_ledger_scopes_incoming_storage_to_its_component() {
    let outgoing = outgoing_fanout_endpoint(0, 5, "gpu1", "draft_01");
    let incoming = incoming_fanout_endpoint(&outgoing);
    let plan = VulkanPlacedEdgeIoPlan {
        backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
        device_id: "gpu1".to_string(),
        signal_element_bytes: Some(2),
        local_edges: Vec::new(),
        endpoints: vec![incoming],
        local_edge_count: 0,
        incoming_endpoint_count: 1,
        outgoing_endpoint_count: 0,
        total_buffer_count: 1,
        total_endpoint_count: 1,
        total_byte_capacity: Some(8_192),
        unresolved_byte_edges: Vec::new(),
    };

    let (bytes, allocations) = plan_edge_residency_allocations_with_passthrough(
        &plan,
        &BTreeSet::new(),
        Some("draft_01"),
    )
    .unwrap();
    assert_eq!(bytes, 8_192);
    assert_eq!(
        allocations[0].kind,
        VulkanRuntimeResidentStreamAllocationKind::EdgeIncoming { edge_index: 5 }
    );

    let (other_bytes, other) = plan_edge_residency_allocations_with_passthrough(
        &plan,
        &BTreeSet::new(),
        Some("other"),
    )
    .unwrap();
    assert_eq!(other_bytes, 0);
    assert!(other.is_empty());
}

#[test]
fn runtime_residency_plan_sizes_verification_snapshots_to_the_requested_window() {
    let mut runtime_model = fixture_model_runtime_model();
    let component = runtime_model
        .circuit_graph
        .components
        .iter_mut()
        .find(|component| component.runtime_role.is_signal_processor())
        .unwrap();
    for state in component
        .state
        .state_ports
        .iter_mut()
        .chain(component.circuit.state_ports.iter_mut())
    {
        state.shape = Some(vec![128]);
        state.shape_per_token = None;
        state.key_shape_per_token = None;
        state.value_shape_per_token = None;
        state.capacity = None;
        state.dtype = Some("F32".to_string());
        state.max_dynamic_activations = None;
    }
    let package_root = tiny_model_dir();
    let tensor_index = runtime_model
        .load_runtime_tensor_index(&package_root)
        .unwrap();

    let one_draft = plan_vulkan_runtime_residency(
        &package_root,
        &runtime_model,
        &tensor_index,
        64,
        1,
        ResourceResidencyPolicy::Eager,
    )
    .unwrap();
    let two_drafts = plan_vulkan_runtime_residency(
        &package_root,
        &runtime_model,
        &tensor_index,
        64,
        2,
        ResourceResidencyPolicy::Eager,
    )
    .unwrap();

    assert_eq!(one_draft.speculative_draft_tokens, 1);
    assert_eq!(two_drafts.speculative_draft_tokens, 2);
    let one_snapshot_bytes = one_draft.device_plans[0]
        .breakdown
        .causal_verification_snapshot_bytes;
    let two_snapshot_bytes = two_drafts.device_plans[0]
        .breakdown
        .causal_verification_snapshot_bytes;
    assert!(one_snapshot_bytes > 0);
    assert_eq!(two_snapshot_bytes, one_snapshot_bytes * 2);
    assert!(
        two_drafts.device_plans[0].initial_device_resident_bytes
            > one_draft.device_plans[0].initial_device_resident_bytes
    );
}

#[test]
fn runtime_residency_plan_keeps_internal_shards_out_of_component_ownership() {
    let runtime_model = fixture_model_runtime_model()
        .with_component_shard_devices(
            "layer_00",
            vec![
                RUNTIME_DEFAULT_LOGICAL_DEVICE_ID.to_string(),
                "gpu1".to_string(),
            ],
        )
        .unwrap();
    let package_root = tiny_model_dir();
    let tensor_index = runtime_model
        .load_runtime_tensor_index(&package_root)
        .unwrap();

    let plan = plan_vulkan_runtime_residency(
        &package_root,
        &runtime_model,
        &tensor_index,
        16,
        0,
        ResourceResidencyPolicy::Eager,
    )
    .unwrap();

    assert_eq!(plan.device_plans.len(), 1);
    assert_eq!(
        plan.device_plans[0].device_id,
        RUNTIME_DEFAULT_LOGICAL_DEVICE_ID
    );
    assert!(
        plan.device_plans[0]
            .parameter_residency
            .maximum_addressable_bytes
            > 0
    );
}

#[test]
fn runtime_resource_contract_instantiates_duplicates_without_copying_resources() {
    let mut manifest = fixture_model_package_manifest();
    let dynamic_group_id =
        manifest.resource_residency.atomic_groups[0].id.clone();
    let dynamic_resource_ids = manifest.resource_residency.atomic_groups[0]
        .resource_ids
        .clone();
    manifest.resource_residency.atomic_groups[0].lifetime =
        CompiledResourceLifetime::Dynamic;
    for resource in &mut manifest.resource_residency.resources {
        if dynamic_resource_ids.contains(&resource.id) {
            resource.lifetime = CompiledResourceLifetime::Dynamic;
        }
    }
    let mut selector = CompiledResourceSelector {
        id: String::new(),
        execution_scope: "target".to_string(),
        component_id: "layer_00".to_string(),
        node_id: "operator_norm".to_string(),
        domain_id: "test_domain".to_string(),
        resource_count: 1,
        selection_signal: "operator_norm_out".to_string(),
        execution_signal: "operator_norm_out".to_string(),
        execution_calibration_word_base: 0,
        encoding: CompiledResourceSelectionEncoding {
            element_type: CompiledResourceSelectionElementType::U32,
            selection_count_per_activation: 1,
            index_shift: 0,
            index_mask: 1,
            calibration_word_base: 0,
        },
        mapping: CompiledResourceSelectorMapping::GroupTable {
            atomic_group_ids: vec![dynamic_group_id],
        },
    };
    selector.id = package::compiled_selector_identity(&selector).unwrap();
    let mut checkpoint = CompiledResidencyCheckpoint {
        id: String::new(),
        execution_scope: "target".to_string(),
        component_id: "layer_00".to_string(),
        after_node_id: "operator_norm".to_string(),
        resume_node_id: "q_projection".to_string(),
        selector_ids: vec![selector.id.clone()],
    };
    checkpoint.id =
        package::compiled_checkpoint_identity(&checkpoint).unwrap();
    manifest.resource_residency.selectors = vec![selector];
    manifest.resource_residency.checkpoints = vec![checkpoint];
    let package_root = tiny_model_dir();
    let source_graph = manifest
        .circuit_graph
        .to_resolved_lowered_execution_graph(&package_root)
        .unwrap();
    let runtime_graph =
        StreamCircuitRuntimeGraph::from_source_series(&source_graph, "gpu0")
            .unwrap()
            .duplicate_after_instance(
                &source_graph,
                "layer_00",
                "layer_00_repeat",
            )
            .unwrap()
            .with_instance_device("layer_00_repeat", "gpu1")
            .unwrap();
    let runtime_model = manifest
        .clone()
        .mount_runtime_graph(&runtime_graph)
        .unwrap();
    let source = &manifest.resource_residency;
    let runtime =
        instantiate_runtime_resource_contract(&runtime_model).unwrap();

    assert_eq!(runtime.resources, source.resources);
    assert_eq!(runtime.atomic_groups, source.atomic_groups);
    assert_eq!(runtime.partition_templates, source.partition_templates);
    assert_eq!(
        runtime.bindings.len(),
        source.bindings.len()
            + source
                .bindings
                .iter()
                .filter(|binding| {
                    binding.execution_scope == "target"
                        && binding.component_id == "layer_00"
                })
                .count()
    );
    assert_eq!(
        runtime.selectors.len(),
        source.selectors.len()
            + source
                .selectors
                .iter()
                .filter(|selector| {
                    selector.execution_scope == "target"
                        && selector.component_id == "layer_00"
                })
                .count()
    );
    assert_eq!(
        runtime.checkpoints.len(),
        source.checkpoints.len()
            + source
                .checkpoints
                .iter()
                .filter(|checkpoint| {
                    checkpoint.execution_scope == "target"
                        && checkpoint.component_id == "layer_00"
                })
                .count()
    );
    assert_eq!(
        runtime
            .selectors
            .iter()
            .map(|selector| selector.component_id.as_str())
            .collect::<BTreeSet<_>>(),
        BTreeSet::from(["layer_00", "layer_00_repeat"])
    );
    assert_eq!(
        VulkanCompiledResourceAddressLayout::from_contract(&runtime)
            .unwrap()
            .slot_count(),
        VulkanCompiledResourceAddressLayout::from_contract(source)
            .unwrap()
            .slot_count()
    );

    let tensor_index = runtime_model
        .load_runtime_tensor_index(&package_root)
        .unwrap();
    let plan = plan_vulkan_runtime_residency(
        &package_root,
        &runtime_model,
        &tensor_index,
        16,
        0,
        ResourceResidencyPolicy::DemandRetained,
    )
    .unwrap();
    assert_eq!(
        plan.device_plans
            .iter()
            .map(|device| device.device_id.as_str())
            .collect::<BTreeSet<_>>(),
        BTreeSet::from(["gpu0", "gpu1"])
    );
    assert!(plan.device_plans.iter().all(|device| {
        device.parameter_residency.initial_dynamic_bytes == 0
            && device.parameter_residency.maximum_addressable_bytes > 0
    }));
    let paged = plan_vulkan_runtime_residency(
        &package_root,
        &runtime_model,
        &tensor_index,
        16,
        0,
        ResourceResidencyPolicy::DemandPaged,
    )
    .expect("runtime paging must reuse the package demand-loading contract");
    assert_eq!(
        paged.residency_policy,
        ResourceResidencyPolicy::DemandPaged
    );
    assert_eq!(paged.device_plans, plan.device_plans);
}

#[test]
fn runtime_resource_contract_rewires_source_semantics_to_new_instance_ids() {
    let manifest = fixture_model_package_manifest();
    let package_root = tiny_model_dir();
    let source_graph = manifest
        .circuit_graph
        .to_resolved_lowered_execution_graph(&package_root)
        .unwrap();
    let runtime_graph =
        StreamCircuitRuntimeGraph::from_source_series(&source_graph, "gpu0")
            .unwrap()
            .with_signal_processor_chain(
                &source_graph,
                &[("rewired".to_string(), "layer_00".to_string())],
            )
            .unwrap();
    let runtime_model = manifest.mount_runtime_graph(&runtime_graph).unwrap();
    let runtime =
        instantiate_runtime_resource_contract(&runtime_model).unwrap();

    assert!(!runtime.bindings.is_empty());
    assert!(!runtime
        .bindings
        .iter()
        .any(|binding| binding.component_id == "layer_00"));
    assert!(runtime
        .bindings
        .iter()
        .any(|binding| binding.component_id == "rewired"));
    assert!(!runtime
        .selectors
        .iter()
        .any(|selector| selector.component_id == "layer_00"));
    assert!(!runtime
        .checkpoints
        .iter()
        .any(|checkpoint| checkpoint.component_id == "layer_00"));
}

#[test]
fn compiled_resource_placement_coalesces_logical_aliases_on_one_physical_device() {
    let device = selected_test_vulkan_device()
        .expect("explicit AMD Vulkan test device must open");
    let groups = group_compiled_resource_logical_devices_by_physical(&[
        ("graph_a".to_string(), &device),
        ("graph_b".to_string(), &device),
    ])
    .unwrap();

    assert_eq!(
        groups,
        vec![vec!["graph_a".to_string(), "graph_b".to_string()]]
    );
}

#[test]
fn compiled_resource_placement_rejects_distinct_logical_devices_for_one_physical_gpu() {
    let first = selected_test_vulkan_device()
        .expect("first explicit AMD Vulkan test device must open");
    let second = selected_test_vulkan_device()
        .expect("second explicit AMD Vulkan test device must open");
    assert!(first.shares_physical_device_with(&second));
    assert!(!first.shares_logical_device_with(&second));

    let error = group_compiled_resource_logical_devices_by_physical(&[
        ("graph_a".to_string(), &first),
        ("graph_b".to_string(), &second),
    ])
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("through different Vulkan logical devices")
    );
}

#[test]
fn compiled_resource_placement_separates_distinct_physical_devices() {
    let Some((owner, peer)) = selected_test_vulkan_device_pair() else {
        panic!("explicit Vulkan owner and peer devices are required");
    };
    let groups = group_compiled_resource_logical_devices_by_physical(&[
        ("graph_a".to_string(), owner.as_ref()),
        ("graph_b".to_string(), peer.as_ref()),
    ])
    .unwrap();

    assert_eq!(
        groups,
        vec![
            vec!["graph_a".to_string()],
            vec!["graph_b".to_string()],
        ]
    );
}

#[test]
fn compiled_resource_cross_device_access_requires_an_explicit_choice() {
    let request = VulkanCompiledResourceCrossDeviceAccessRequest {
        selector_id: "selector".to_string(),
        execution_physical_device_id:
            "vulkan-uuid:11111111111111111111111111111111".to_string(),
        resident_physical_device_ids: vec![
            "vulkan-uuid:00000000000000000000000000000000".to_string(),
        ],
    };
    let error =
        require_explicit_compiled_resource_cross_device_choice(&request, None)
            .unwrap_err();
    assert!(error.to_string().contains("choose remote execution"));
    assert_eq!(
        require_explicit_compiled_resource_cross_device_choice(
            &request,
            Some(
                VulkanCompiledResourceCrossDeviceAccessChoice::PeerTransfer,
            ),
        )
        .unwrap(),
        VulkanCompiledResourceCrossDeviceAccessChoice::PeerTransfer
    );
}

#[test]
fn demand_plan_does_not_allocate_its_maximum_parameter_address_space() {
    let runtime_model = fixture_model_runtime_model_with_dynamic_partition(1_000, 64);
    let package_root = tiny_model_dir();
    let tensor_index = runtime_model
        .load_runtime_tensor_index(&package_root)
        .unwrap();

    let demand = plan_vulkan_runtime_residency(
        &package_root,
        &runtime_model,
        &tensor_index,
        16,
        0,
        ResourceResidencyPolicy::DemandRetained,
    )
    .unwrap();
    let eager = plan_vulkan_runtime_residency(
        &package_root,
        &runtime_model,
        &tensor_index,
        16,
        0,
        ResourceResidencyPolicy::Eager,
    )
    .unwrap();
    let demand_device = &demand.device_plans[0];
    let eager_device = &eager.device_plans[0];

    assert_eq!(
        demand_device
            .parameter_residency
            .initial_dynamic_bytes,
        0
    );
    assert_eq!(
        demand_device
            .parameter_residency
            .staging_headroom_bytes,
        64
    );
    assert_eq!(
        demand_device
            .parameter_residency
            .maximum_addressable_bytes
            - demand_device
                .parameter_residency
                .current_resident_bytes,
        64_000
    );
    assert_eq!(
        eager_device
            .parameter_residency
            .initial_dynamic_bytes,
        64_000
    );
    assert_eq!(
        eager_device.initial_device_resident_bytes
            - demand_device.initial_device_resident_bytes,
        64_000
            + eager_device
                .resource_store
                .maximum_dynamic_allocation_padding_bytes
    );
    assert_eq!(demand_device.resource_store.transfer_staging_slot_count, 2);
    assert_eq!(
        demand_device.resource_store.transfer_staging_device_bytes,
        demand_device
            .resource_store
            .transfer_staging_slot_byte_capacity
            * 2
    );
    assert!(demand_device.resource_store.address_table_device_bytes > 0);
    assert!(demand_device.resource_store.metadata_device_bytes > 0);
    assert_eq!(
        demand_device.initial_device_resident_bytes,
        demand_device.parameter_residency.current_resident_bytes
            + demand_device.resource_store.fixed_device_bytes().unwrap()
            + demand_device.working_set.transient_state_bytes
            + demand_device.working_set.activation_headroom_bytes
    );
    assert_eq!(
        vulkan_runtime_maximum_device_resident_bytes(demand_device).unwrap(),
        demand_device.parameter_residency.maximum_addressable_bytes
            + demand_device.resource_store.maximum_extra_device_bytes().unwrap()
            + demand_device.working_set.transient_state_bytes
            + demand_device.working_set.activation_headroom_bytes
    );

    let safe_capacity = demand_device.initial_device_resident_bytes + 64;
    let admission = admit_vulkan_runtime_residency_growth(
        demand_device,
        demand_device.parameter_residency.current_resident_bytes,
        64,
        safe_capacity,
    )
    .unwrap();
    assert_eq!(
        admission.projected_resident_parameter_bytes,
        demand_device.parameter_residency.current_resident_bytes + 64
    );
    assert_eq!(admission.projected_device_resident_bytes, safe_capacity);

    let capacity_error = admit_vulkan_runtime_residency_growth(
        demand_device,
        demand_device.parameter_residency.current_resident_bytes,
        64,
        safe_capacity - 1,
    )
    .unwrap_err();
    assert!(capacity_error.to_string().contains("safe capacity"));

    let maximum_error = admit_vulkan_runtime_residency_growth(
        demand_device,
        demand_device
            .parameter_residency
            .maximum_addressable_bytes,
        64,
        usize::MAX,
    )
    .unwrap_err();
    assert!(maximum_error.to_string().contains("maximum addressable"));

    let staging_error = admit_vulkan_runtime_residency_growth(
        demand_device,
        demand_device.parameter_residency.current_resident_bytes,
        65,
        usize::MAX,
    )
    .unwrap_err();
    assert!(staging_error.to_string().contains("staging headroom"));

    let current_error = admit_vulkan_runtime_residency_growth(
        demand_device,
        demand_device.parameter_residency.current_resident_bytes - 1,
        64,
        usize::MAX,
    )
    .unwrap_err();
    assert!(current_error.to_string().contains("outside the planned range"));
}

#[test]
fn initial_runtime_residency_aggregates_logical_slices_on_one_physical_device() {
    let plan = VulkanRuntimeResidencyPlan {
        schema: VULKAN_RUNTIME_RESIDENCY_PLAN_SCHEMA.to_string(),
        package_id: "package".to_string(),
        residency_policy: ResourceResidencyPolicy::DemandRetained,
        context_capacity_activations: 1,
        speculative_draft_tokens: 0,
        device_plans: vec![
            VulkanRuntimeDeviceResidencyPlan {
                device_id: "logical-a".to_string(),
                parameter_residency: VulkanRuntimeParameterResidencyBytes::default(),
                resource_store: VulkanCompiledResourceStoreResidencyBytes::default(),
                working_set: VulkanRuntimeWorkingSetBytes::default(),
                breakdown: VulkanRuntimeDeviceResidencyBreakdown::default(),
                resident_stream_device_allocations: Vec::new(),
                initial_device_resident_bytes: 600,
            },
            VulkanRuntimeDeviceResidencyPlan {
                device_id: "logical-b".to_string(),
                parameter_residency: VulkanRuntimeParameterResidencyBytes::default(),
                resource_store: VulkanCompiledResourceStoreResidencyBytes::default(),
                working_set: VulkanRuntimeWorkingSetBytes::default(),
                breakdown: VulkanRuntimeDeviceResidencyBreakdown::default(),
                resident_stream_device_allocations: Vec::new(),
                initial_device_resident_bytes: 401,
            },
        ],
        total_initial_device_resident_bytes: 1_001,
        total_current_resident_parameter_bytes: 0,
        total_maximum_addressable_parameter_bytes: 0,
    };
    let physical = BTreeMap::from([
        ("logical-a".to_string(), "physical".to_string()),
        ("logical-b".to_string(), "physical".to_string()),
    ]);

    let admitted = admit_vulkan_runtime_initial_residency_by_physical_device(
        &plan,
        &physical,
        &BTreeMap::from([("physical".to_string(), 1_001)]),
    )
    .unwrap();
    assert_eq!(admitted["physical"], 1_001);

    let error = admit_vulkan_runtime_initial_residency_by_physical_device(
        &plan,
        &physical,
        &BTreeMap::from([("physical".to_string(), 1_000)]),
    )
    .unwrap_err();
    assert!(error.to_string().contains("needs 1001 initial device bytes"));
    assert!(error.to_string().contains("stable safe capacity is 1000"));
}
