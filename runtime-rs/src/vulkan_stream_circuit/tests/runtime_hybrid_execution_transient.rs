#[test]
fn runtime_hybrid_shared_host_ledger_extends_without_collapsing_allocations() {
    let mut normal = VulkanRuntimeHybridExecutionTransientPlan::default();
    normal
        .add_shared_host_allocation(
            VulkanRuntimeSharedHostTransientAllocationMode::Always,
            "gpu0",
            ["gpu0".to_string(), "gpu1".to_string()],
            17,
            "normal signal",
        )
        .unwrap();
    let mut verification = VulkanRuntimeHybridExecutionTransientPlan::default();
    verification
        .add_shared_host_allocation(
            VulkanRuntimeSharedHostTransientAllocationMode::CrossDeviceTimelineStaging,
            "gpu1",
            ["gpu0".to_string(), "gpu1".to_string()],
            29,
            "verification edge",
        )
        .unwrap();

    normal
        .extend(verification.into_allocation_class(
            VulkanRuntimeStreamAllocationClass::VerificationRunner,
        ))
        .unwrap();

    assert_eq!(normal.host_bytes(), 46);
    assert_eq!(normal.shared_host_allocations.len(), 2);
    assert_eq!(normal.shared_host_allocations[0].byte_capacity, 17);
    assert_eq!(normal.shared_host_allocations[1].byte_capacity, 29);
    assert_eq!(
        normal.shared_host_allocations[0].allocation_class,
        VulkanRuntimeStreamAllocationClass::Permanent,
    );
    assert_eq!(
        normal.shared_host_allocations[1].allocation_class,
        VulkanRuntimeStreamAllocationClass::VerificationRunner,
    );
    let original = normal.clone();
    let malformed = normal
        .add_shared_host_allocation(
            VulkanRuntimeSharedHostTransientAllocationMode::Always,
            "gpu0",
            ["gpu1".to_string()],
            1,
            "missing owner",
        )
        .unwrap_err();
    assert!(malformed.to_string().contains("owner participant"));
    assert_eq!(normal, original);
}

#[test]
fn speculative_catch_up_transient_matches_one_canonical_component_batch_and_embedding() {
    let package_root = tiny_model_dir();
    let mut model = fixture_model_runtime_model();
    let logical_device_id = model.placement.default_device_id.clone();
    let tensor_index = model.load_runtime_tensor_index(&package_root).unwrap();
    let contract = instantiate_runtime_resource_contract(&model).unwrap();
    let slice = VulkanResidentModelPackageDeviceSlicePlan::prepare_for_physical_planning(
        &package_root,
        &model,
        &contract,
        &tensor_index,
        &logical_device_id,
        64,
    )
    .unwrap();
    let lane_capacity = speculative_catch_up_lane_capacity(7).unwrap();
    let component_plan = VulkanComponentBatchResidentAllocationPlan::for_single_device(
        &slice.placed_plan,
        &slice.prepared_plan,
        &slice.batch_kernels,
        lane_capacity,
        VulkanComponentBatchExecutionMode::CausalSequence,
        &VulkanComponentBatchExecutionScope::all(),
        &BTreeSet::new(),
        false,
        false,
    )
    .unwrap();
    let mut decoder = parallel_speculative_decoder_with_default("draft", 7);
    decoder.execution_contract =
        VulkanResidentSpeculativeExecutionContract::AutoregressiveFeedback {
            processor_schedule: "one_token_per_tick".to_string(),
            output_schedule: "dedicated_token_transducer".to_string(),
        };
    model.package.speculative_decoders.push(decoder);
    let slice_plans = BTreeMap::from([("draft".to_string(), slice)]);

    let planned = exact_vulkan_runtime_speculative_catch_up_transient_plan(
        &model,
        &slice_plans,
        4,
        7,
        ResourceResidencyPolicy::Eager,
    )
    .unwrap();
    let embedding_bytes = lane_capacity * size_of::<u32>()
        + VULKAN_COMPONENT_BATCH_WIDTH_CONTROL_BYTE_CAPACITY as usize;
    assert_eq!(
        planned.device_bytes_by_logical_device[&logical_device_id],
        component_plan.total_byte_capacity + embedding_bytes,
    );
    assert_eq!(
        planned.device_allocations.len(),
        component_plan.allocations.len() + 2,
    );
    assert!(planned.device_allocations.iter().all(|allocation| {
        allocation.allocation_class == VulkanRuntimeStreamAllocationClass::CatchUpRunner
    }));
    assert!(planned.shared_host_allocations.is_empty());

    let prompt_lane_capacity = speculative_catch_up_execution_lane_capacity(17, 7).unwrap();
    let prompt_component_plan = VulkanComponentBatchResidentAllocationPlan::for_single_device(
        &slice_plans["draft"].placed_plan,
        &slice_plans["draft"].prepared_plan,
        &slice_plans["draft"].batch_kernels,
        prompt_lane_capacity,
        VulkanComponentBatchExecutionMode::CausalSequence,
        &VulkanComponentBatchExecutionScope::all(),
        &BTreeSet::new(),
        false,
        false,
    )
    .unwrap();
    let prompt_planned = exact_vulkan_runtime_speculative_catch_up_transient_plan(
        &model,
        &slice_plans,
        17,
        7,
        ResourceResidencyPolicy::Eager,
    )
    .unwrap();
    assert_eq!(prompt_lane_capacity, 32);
    assert_eq!(
        prompt_planned.device_bytes_by_logical_device[&logical_device_id],
        prompt_component_plan.total_byte_capacity
            + prompt_lane_capacity * size_of::<u32>()
            + VULKAN_COMPONENT_BATCH_WIDTH_CONTROL_BYTE_CAPACITY as usize,
    );
    assert!(
        prompt_planned.device_bytes_by_logical_device[&logical_device_id]
            > planned.device_bytes_by_logical_device[&logical_device_id],
    );
}

#[test]
fn speculative_catch_up_transient_is_absent_for_demand_residency_and_fails_closed() {
    let mut model = fixture_model_runtime_model();
    let mut decoder = parallel_speculative_decoder_with_default("draft", 7);
    decoder.execution_contract =
        VulkanResidentSpeculativeExecutionContract::AutoregressiveFeedback {
            processor_schedule: "one_token_per_tick".to_string(),
            output_schedule: "dedicated_token_transducer".to_string(),
        };
    model.package.speculative_decoders.push(decoder);

    let demand = exact_vulkan_runtime_speculative_catch_up_transient_plan(
        &model,
        &BTreeMap::new(),
        4,
        7,
        ResourceResidencyPolicy::DemandRetained,
    )
    .unwrap();
    assert_eq!(demand, VulkanRuntimeHybridExecutionTransientPlan::default());

    let error = exact_vulkan_runtime_speculative_catch_up_transient_plan(
        &model,
        &BTreeMap::new(),
        4,
        7,
        ResourceResidencyPolicy::Eager,
    )
    .unwrap_err();
    assert!(error.to_string().contains("no prepared catch-up slice"));
}

fn fixture_parallel_speculative_runtime_model(
    source_device_id: &str,
) -> (
    VulkanResidentRuntimeModel,
    BTreeMap<String, VulkanResidentModelPackageDeviceSlicePlan>,
    Vec<VulkanRuntimeSelectedResourceMountDevice>,
) {
    let package_root = tiny_model_dir();
    let mut model = fixture_model_runtime_model();
    let decoder_device_id = model.placement.default_device_id.clone();
    let mut decoder = parallel_speculative_decoder_with_default("parallel_draft", 7);
    decoder.circuit_graph = model.package.circuit_graph.clone();
    let layer_component = model
        .package
        .circuit_graph
        .components
        .iter()
        .find(|component| component.component_id == "layer_00")
        .unwrap()
        .clone();
    let layer_execution = model.package.component_executions[0].clone();
    let draft_component = |component_id: &str, runtime_role| {
        let mut component = layer_component.clone();
        component.component_id = component_id.to_string();
        component.runtime_role = runtime_role;
        component.circuit.source.component_id = component_id.to_string();
        component.circuit.runtime_role = runtime_role;
        component
    };
    let mut input_component =
        draft_component("draft_input", CircuitRuntimeRole::DraftInputAdapter);
    let mut anchor_port = input_component.circuit.boundary.inputs[0].clone();
    anchor_port.id = "anchor_frame".to_string();
    anchor_port.component_port = Some("anchor".to_string());
    input_component.circuit.boundary.inputs.push(anchor_port);
    decoder.circuit_graph.components = vec![
        input_component,
        draft_component("draft_processor", CircuitRuntimeRole::DraftProcessor),
        draft_component(
            "draft_output",
            CircuitRuntimeRole::DraftOutputTransducer,
        ),
    ];
    decoder.component_executions = ["draft_input", "draft_processor", "draft_output"]
        .into_iter()
        .map(|component_id| {
            let mut execution = layer_execution.clone();
            execution.component_id = component_id.to_string();
            execution
        })
        .collect();
    decoder.circuit_graph.edges = vec![
        crate::stream_circuit::StreamCircuitGraphEdge {
            id: "committed_context".to_string(),
            source: crate::stream_circuit::StreamCircuitEdgeEndpoint {
                component_id: "draft_input".to_string(),
                port_id: "output_frame".to_string(),
            },
            destination: crate::stream_circuit::StreamCircuitEdgeEndpoint {
                component_id: "draft_processor".to_string(),
                port_id: "input_frame".to_string(),
            },
            connection: StreamCircuitConnection::SharedContext {
                state_update: "committed_target_only".to_string(),
            },
        },
        crate::stream_circuit::StreamCircuitGraphEdge {
            id: "proposal_output".to_string(),
            source: crate::stream_circuit::StreamCircuitEdgeEndpoint {
                component_id: "draft_processor".to_string(),
                port_id: "output_frame".to_string(),
            },
            destination: crate::stream_circuit::StreamCircuitEdgeEndpoint {
                component_id: "draft_output".to_string(),
                port_id: "input_frame".to_string(),
            },
            connection: StreamCircuitConnection::Forward,
        },
    ];
    decoder.circuit_graph.boundary = crate::stream_circuit::StreamCircuitGraphBoundary {
        external_inputs: vec![
            crate::stream_circuit::StreamCircuitGraphBoundaryPort {
                id: "input_frame".to_string(),
                endpoint: crate::stream_circuit::StreamCircuitEdgeEndpoint {
                    component_id: "draft_input".to_string(),
                    port_id: "input_frame".to_string(),
                },
                source_tap: Some(StreamCircuitGraphSourceTap {
                    component_id: "layer_00".to_string(),
                    port_id: "output_frame".to_string(),
                    instance_selection:
                        StreamCircuitGraphSourceTapInstanceSelection::LastInExecutionOrder,
                }),
            },
            crate::stream_circuit::StreamCircuitGraphBoundaryPort {
                id: "anchor_frame".to_string(),
                endpoint: crate::stream_circuit::StreamCircuitEdgeEndpoint {
                    component_id: "draft_input".to_string(),
                    port_id: "anchor_frame".to_string(),
                },
                source_tap: None,
            },
        ],
        public_outputs: vec![crate::stream_circuit::StreamCircuitGraphBoundaryPort {
            id: "draft_output".to_string(),
            endpoint: crate::stream_circuit::StreamCircuitEdgeEndpoint {
                component_id: "draft_output".to_string(),
                port_id: "output_frame".to_string(),
            },
            source_tap: None,
        }],
    };
    model
        .runtime_graph
        .instances
        .iter_mut()
        .find(|instance| instance.source_component_id == "layer_00")
        .expect("fixture layer instance exists")
        .device_id = source_device_id.to_string();
    model.package.speculative_decoders.push(decoder.clone());

    let tensor_index = model.load_runtime_tensor_index(&package_root).unwrap();
    let draft_runtime_model =
        speculative_decoder_runtime_model(&model, &decoder, &decoder_device_id);
    let draft_contract = instantiate_runtime_resource_contract(&draft_runtime_model).unwrap();
    let slice = VulkanResidentModelPackageDeviceSlicePlan::prepare_for_physical_planning(
        &package_root,
        &draft_runtime_model,
        &draft_contract,
        &tensor_index,
        &decoder_device_id,
        64,
    )
    .unwrap();
    let devices = [
        (decoder_device_id.clone(), "physical0".to_string()),
        (source_device_id.to_string(), "physical1".to_string()),
    ]
    .into_iter()
    .collect::<BTreeMap<_, _>>()
    .into_iter()
    .map(
        |(logical_device_id, physical_device_id)| VulkanRuntimeSelectedResourceMountDevice {
            execution_identity: hybrid_test_device(&physical_device_id),
            logical_device_id,
            physical_device_id,
            live_safe_capacity_bytes: usize::MAX,
            upload_alignment: 8,
        },
    )
    .collect::<Vec<_>>();
    (
        model,
        BTreeMap::from([("parallel_draft".to_string(), slice)]),
        devices,
    )
}

#[test]
fn parallel_speculative_processor_transient_matches_mounted_permanent_allocations() {
    let (model, slices, devices) = fixture_parallel_speculative_runtime_model("source_gpu");
    let decoder = &model.package.speculative_decoders[0];
    let slice = &slices[&decoder.id];
    let scopes = parallel_speculative_execution_scopes(decoder).unwrap();
    let proposal = VulkanComponentBatchResidentAllocationPlan::for_single_device(
        &slice.placed_plan,
        &slice.prepared_plan,
        &slice.batch_kernels,
        7,
        VulkanComponentBatchExecutionMode::ParallelBlock,
        &VulkanComponentBatchExecutionScope::nodes(scopes.proposal_node_ids_by_component)
            .unwrap(),
        &BTreeSet::new(),
        false,
        false,
    )
    .unwrap();
    let committed = VulkanComponentBatchResidentAllocationPlan::for_single_device(
        &slice.placed_plan,
        &slice.prepared_plan,
        &slice.batch_kernels,
        1,
        VulkanComponentBatchExecutionMode::ParallelBlock,
        &VulkanComponentBatchExecutionScope::nodes(scopes.state_node_ids_by_component).unwrap(),
        &BTreeSet::new(),
        false,
        false,
    )
    .unwrap();

    let planned = exact_vulkan_runtime_parallel_speculative_processor_transient_plan(
        &model,
        &slices,
        &devices,
        7,
        ResourceResidencyPolicy::Eager,
    )
    .unwrap();
    let readback_bytes = 7 * (size_of::<u32>() + size_of::<f32>());
    let source_tap_bytes = model.package.activation_element_bytes.unwrap() * 16;
    assert_eq!(
        planned.device_bytes_by_logical_device[&slice.device_id],
        proposal.total_byte_capacity
            + committed.total_byte_capacity
            + readback_bytes
            + source_tap_bytes,
    );
    assert_eq!(
        planned.device_bytes_by_logical_device["source_gpu"],
        source_tap_bytes,
    );
    assert_eq!(
        planned.device_allocations.len(),
        proposal.allocations.len() + committed.allocations.len() + 3,
    );
    assert!(planned.device_allocations.iter().all(|allocation| {
        allocation.allocation_class == VulkanRuntimeStreamAllocationClass::Permanent
    }));

    let (same_model, same_slices, mut same_devices) =
        fixture_parallel_speculative_runtime_model("source_gpu");
    same_devices
        .iter_mut()
        .find(|device| device.logical_device_id == "source_gpu")
        .unwrap()
        .physical_device_id = "physical0".to_string();
    let same = exact_vulkan_runtime_parallel_speculative_processor_transient_plan(
        &same_model,
        &same_slices,
        &same_devices,
        7,
        ResourceResidencyPolicy::Eager,
    )
    .unwrap();
    assert!(!same
        .device_bytes_by_logical_device
        .contains_key("source_gpu"));
    assert_eq!(
        same.device_bytes_by_logical_device[&slice.device_id],
        proposal.total_byte_capacity + committed.total_byte_capacity + readback_bytes,
    );
}

#[test]
fn parallel_speculative_processor_transient_fails_closed_on_incomplete_mount_identity() {
    let (model, slices, devices) = fixture_parallel_speculative_runtime_model("source_gpu");
    let disabled = exact_vulkan_runtime_parallel_speculative_processor_transient_plan(
        &model,
        &BTreeMap::new(),
        &[],
        0,
        ResourceResidencyPolicy::Eager,
    )
    .unwrap();
    assert_eq!(disabled, VulkanRuntimeHybridExecutionTransientPlan::default());

    let missing_slice = exact_vulkan_runtime_parallel_speculative_processor_transient_plan(
        &model,
        &BTreeMap::new(),
        &devices,
        7,
        ResourceResidencyPolicy::Eager,
    )
    .unwrap_err();
    assert!(missing_slice.to_string().contains("has no prepared slice"));

    let missing_source = exact_vulkan_runtime_parallel_speculative_processor_transient_plan(
        &model,
        &slices,
        &devices
            .into_iter()
            .filter(|device| device.logical_device_id != "source_gpu")
            .collect::<Vec<_>>(),
        7,
        ResourceResidencyPolicy::Eager,
    )
    .unwrap_err();
    assert!(missing_source.to_string().contains("has no physical identity"));
}

#[test]
fn parallel_speculative_state_ingestion_accounts_both_cached_lane_classes() {
    let (model, slices, devices) = fixture_parallel_speculative_runtime_model("source_gpu");
    let decoder = &model.package.speculative_decoders[0];
    let slice = &slices[&decoder.id];
    let scopes = parallel_speculative_execution_scopes(decoder).unwrap();
    let execution_scope = VulkanComponentBatchExecutionScope::nodes(
        scopes.state_ingestion_node_ids_by_component,
    )
    .unwrap();
    let normal_lane_capacity = causal_component_block_lane_capacity(3).unwrap();
    let verification_lane_capacity = speculative_catch_up_lane_capacity(7).unwrap();
    let component_plan = |lane_capacity, uses_demand_residency| {
        VulkanComponentBatchResidentAllocationPlan::for_single_device(
            &slice.placed_plan,
            &slice.prepared_plan,
            &slice.batch_kernels,
            lane_capacity,
            VulkanComponentBatchExecutionMode::CausalSequence,
            &execution_scope,
            &BTreeSet::new(),
            false,
            uses_demand_residency,
        )
        .unwrap()
    };
    let normal = component_plan(normal_lane_capacity, false);
    let verification = component_plan(verification_lane_capacity, false);
    let source_frame_bytes = model.package.activation_element_bytes.unwrap() * 16;

    let planned = exact_vulkan_runtime_parallel_speculative_state_ingestion_transient_plan(
        &model,
        &slices,
        &devices,
        3,
        7,
        ResourceResidencyPolicy::Eager,
    )
    .unwrap();
    let staging_bytes = source_frame_bytes * (normal_lane_capacity + verification_lane_capacity);
    assert_eq!(
        planned.device_bytes_by_logical_device[&slice.device_id],
        normal.total_byte_capacity + verification.total_byte_capacity + staging_bytes,
    );
    assert_eq!(
        planned.device_bytes_by_logical_device["source_gpu"],
        staging_bytes,
    );
    assert_eq!(
        planned.device_allocations.len(),
        normal.allocations.len() + verification.allocations.len() + 4,
    );
    assert!(planned.device_allocations.iter().any(|allocation| {
        allocation.concern.contains("normal prefill")
            && allocation.allocation_class == VulkanRuntimeStreamAllocationClass::PromptRunner
    }));
    assert!(planned.device_allocations.iter().any(|allocation| {
        allocation.concern.contains("causal verification")
            && allocation.allocation_class
                == VulkanRuntimeStreamAllocationClass::VerificationRunner
    }));
    assert!(planned.device_allocations.iter().all(|allocation| matches!(
        allocation.allocation_class,
        VulkanRuntimeStreamAllocationClass::PromptRunner
            | VulkanRuntimeStreamAllocationClass::VerificationRunner
    )));

    let demand = exact_vulkan_runtime_parallel_speculative_state_ingestion_transient_plan(
        &model,
        &slices,
        &devices,
        3,
        7,
        ResourceResidencyPolicy::DemandRetained,
    )
    .unwrap();
    assert_eq!(
        demand.device_allocations.len(),
        planned.device_allocations.len() + 2,
        "each demand-resident causal runner needs its own predicate",
    );

    let mut colocated_devices = devices;
    colocated_devices
        .iter_mut()
        .find(|device| device.logical_device_id == "source_gpu")
        .unwrap()
        .physical_device_id = "physical0".to_string();
    let colocated = exact_vulkan_runtime_parallel_speculative_state_ingestion_transient_plan(
        &model,
        &slices,
        &colocated_devices,
        3,
        7,
        ResourceResidencyPolicy::Eager,
    )
    .unwrap();
    assert!(!colocated
        .device_bytes_by_logical_device
        .contains_key("source_gpu"));
    assert_eq!(
        colocated.device_bytes_by_logical_device[&slice.device_id],
        normal.total_byte_capacity + verification.total_byte_capacity,
    );
}

#[test]
fn parallel_speculative_state_ingestion_fails_closed_without_its_slice() {
    let (model, _, devices) = fixture_parallel_speculative_runtime_model("source_gpu");
    let disabled = exact_vulkan_runtime_parallel_speculative_state_ingestion_transient_plan(
        &model,
        &BTreeMap::new(),
        &[],
        0,
        0,
        ResourceResidencyPolicy::Eager,
    )
    .unwrap();
    assert_eq!(disabled, VulkanRuntimeHybridExecutionTransientPlan::default());

    let error = exact_vulkan_runtime_parallel_speculative_state_ingestion_transient_plan(
        &model,
        &BTreeMap::new(),
        &devices,
        4,
        7,
        ResourceResidencyPolicy::Eager,
    )
    .unwrap_err();
    assert!(error.to_string().contains("no prepared state-ingestion slice"));
}

#[test]
fn physical_mount_admits_parallel_speculative_processor_allocations() {
    let (model, _, _) = fixture_parallel_speculative_runtime_model("runtime_default");
    let physical = VulkanRuntimePhysicalExecutionPlan::uniform(&model);
    let devices = physical
        .device_ids(&model)
        .into_iter()
        .map(|logical_device_id| physical_mount_test_device(&logical_device_id))
        .collect::<Vec<_>>();
    let planned = plan_vulkan_runtime_physical_mount(
        tiny_model_dir(),
        &model,
        &physical,
        None,
        64,
        7,
        ResourceResidencyPolicy::Eager,
        &devices,
        usize::MAX,
    )
    .unwrap()
    .unwrap();
    let allocations = planned
        .physical_execution_residency_plan
        .device_plans
        .iter()
        .flat_map(|device| &device.execution_transient_device_allocations)
        .collect::<Vec<_>>();
    assert!(allocations
        .iter()
        .any(|allocation| allocation.concern.contains("proposal")
            && allocation.allocation_class == VulkanRuntimeStreamAllocationClass::Permanent));
    assert!(allocations
        .iter()
        .any(|allocation| allocation.concern.contains("committed context")
            && allocation.allocation_class == VulkanRuntimeStreamAllocationClass::Permanent));
    assert!(allocations
        .iter()
        .any(|allocation| allocation.concern.contains("output readback")
            && allocation.allocation_class == VulkanRuntimeStreamAllocationClass::Permanent));
    assert!(allocations
        .iter()
        .any(|allocation| allocation.concern.contains("normal prefill state ingestion")
            && allocation.allocation_class
                == VulkanRuntimeStreamAllocationClass::PromptRunner));
    assert!(allocations.iter().any(|allocation| allocation
        .concern
        .contains("causal verification state ingestion")
        && allocation.allocation_class
            == VulkanRuntimeStreamAllocationClass::VerificationRunner));
}

#[test]
fn runtime_hybrid_exact_prefill_transient_scales_only_lane_residency() {
    let package_root = tiny_model_dir();
    let model = fixture_model_runtime_model();
    let logical_device_id = model.placement.default_device_id.clone();
    let tensor_index = model.load_runtime_tensor_index(&package_root).unwrap();
    let contract = instantiate_runtime_resource_contract(&model).unwrap();
    let slice = VulkanResidentModelPackageDeviceSlicePlan::prepare_for_physical_planning(
        &package_root,
        &model,
        &contract,
        &tensor_index,
        &logical_device_id,
        64,
    )
    .unwrap();
    let execution_plan = VulkanDistributedExecutionPlan {
        device_ids: Vec::new(),
        storage_buffer_offset_alignment: 16,
        dispatches: Vec::new(),
        execution_islands: Vec::new(),
        shared_activation_route: VulkanSharedResidentBufferRoute::SharedHost,
        shared_input_byte_capacity: 0,
        shared_output_byte_capacity: 0,
        distributed_parameter_byte_count: 0,
    };
    let components = BTreeSet::from(["layer_00".to_string()]);
    let one = exact_vulkan_runtime_hybrid_prefill_transient_plan(
        &model,
        &components,
        std::slice::from_ref(&slice),
        &execution_plan,
        1,
        1,
        &contract,
        &VulkanCompiledResourceAddressLayout::from_contract(&contract).unwrap(),
        ResourceResidencyPolicy::Eager,
        false,
        false,
    )
    .unwrap();
    let four = exact_vulkan_runtime_hybrid_prefill_transient_plan(
        &model,
        &components,
        std::slice::from_ref(&slice),
        &execution_plan,
        4,
        4,
        &contract,
        &VulkanCompiledResourceAddressLayout::from_contract(&contract).unwrap(),
        ResourceResidencyPolicy::Eager,
        false,
        false,
    )
    .unwrap();
    let three_in_four = exact_vulkan_runtime_hybrid_prefill_transient_plan(
        &model,
        &components,
        std::slice::from_ref(&slice),
        &execution_plan,
        3,
        4,
        &contract,
        &VulkanCompiledResourceAddressLayout::from_contract(&contract).unwrap(),
        ResourceResidencyPolicy::Eager,
        false,
        false,
    )
    .unwrap();
    let fixed = exact_vulkan_component_batch_fixed_control_bytes().unwrap()
        + size_of::<u32>();
    let one_bytes = one.device_bytes_by_logical_device[&logical_device_id];
    let four_bytes = four.device_bytes_by_logical_device[&logical_device_id];
    assert!(one_bytes > fixed, "the fixture must require lane storage");
    assert_eq!(four_bytes, fixed + 4 * (one_bytes - fixed));
    assert_eq!(
        three_in_four.device_bytes_by_logical_device[&logical_device_id],
        four_bytes,
        "the allocation ledger must reserve the rounded runner capacity, not only the active width",
    );
    assert_eq!(
        one.device_allocations
            .iter()
            .map(|allocation| allocation.byte_capacity)
            .sum::<usize>(),
        one_bytes
    );
    assert_eq!(
        four
            .device_allocations
            .iter()
            .map(|allocation| allocation.byte_capacity)
            .sum::<usize>(),
        four_bytes
    );
    assert_eq!(
        four
            .device_allocations
            .iter()
            .filter(|allocation| allocation.concern == "component-batch lane stream-control")
            .count(),
        4
    );
    assert_eq!(one.device_bytes_by_logical_device.len(), 1);
    assert_eq!(four.device_bytes_by_logical_device.len(), 1);

    let undersized = exact_vulkan_runtime_hybrid_prefill_transient_plan(
        &model,
        &components,
        std::slice::from_ref(&slice),
        &execution_plan,
        4,
        3,
        &contract,
        &VulkanCompiledResourceAddressLayout::from_contract(&contract).unwrap(),
        ResourceResidencyPolicy::Eager,
        false,
        false,
    )
    .unwrap_err();
    assert!(undersized.to_string().contains("covering the active width"));
}

#[test]
fn runtime_hybrid_exact_speculative_prefill_accounts_both_cached_runners() {
    let package_root = tiny_model_dir();
    let model = fixture_model_runtime_model();
    let logical_device_id = model.placement.default_device_id.clone();
    let tensor_index = model.load_runtime_tensor_index(&package_root).unwrap();
    let contract = instantiate_runtime_resource_contract(&model).unwrap();
    let layout = VulkanCompiledResourceAddressLayout::from_contract(&contract).unwrap();
    let slice = VulkanResidentModelPackageDeviceSlicePlan::prepare_for_physical_planning(
        &package_root,
        &model,
        &contract,
        &tensor_index,
        &logical_device_id,
        64,
    )
    .unwrap();
    let execution_plan = VulkanDistributedExecutionPlan {
        device_ids: Vec::new(),
        storage_buffer_offset_alignment: 16,
        dispatches: Vec::new(),
        execution_islands: Vec::new(),
        shared_activation_route: VulkanSharedResidentBufferRoute::SharedHost,
        shared_input_byte_capacity: 0,
        shared_output_byte_capacity: 0,
        distributed_parameter_byte_count: 0,
    };
    let components = BTreeSet::from(["layer_00".to_string()]);
    let normal = exact_vulkan_runtime_hybrid_prefill_transient_plan(
        &model,
        &components,
        std::slice::from_ref(&slice),
        &execution_plan,
        4,
        4,
        &contract,
        &layout,
        ResourceResidencyPolicy::Eager,
        false,
        false,
    )
    .unwrap();
    let verification = exact_vulkan_runtime_hybrid_prefill_transient_plan(
        &model,
        &components,
        std::slice::from_ref(&slice),
        &execution_plan,
        8,
        8,
        &contract,
        &layout,
        ResourceResidencyPolicy::Eager,
        false,
        true,
    )
    .unwrap();
    let expected = normal.device_bytes_by_logical_device[&logical_device_id]
        + verification.device_bytes_by_logical_device[&logical_device_id];
    let mut combined = normal;
    combined.extend(verification).unwrap();
    assert_eq!(combined.device_bytes_by_logical_device[&logical_device_id], expected);
}

#[test]
fn runtime_hybrid_exact_speculative_source_tap_resolves_the_last_runtime_instance() {
    let package_root = tiny_model_dir();
    let mut model = fixture_model_runtime_model_with_remote_middle();
    let output_port_id = model.circuit_graph.components[0].circuit.boundary.outputs[0]
        .id
        .clone();
    let mut decoder = parallel_speculative_decoder_with_default("draft", 7);
    decoder.circuit_graph.boundary.external_inputs[0].source_tap =
        Some(StreamCircuitGraphSourceTap {
            component_id: "layer_00".to_string(),
            port_id: output_port_id,
            instance_selection:
                StreamCircuitGraphSourceTapInstanceSelection::LastInExecutionOrder,
        });
    model.package.speculative_decoders.push(decoder);
    let tensor_index = model.load_runtime_tensor_index(&package_root).unwrap();
    let contract = instantiate_runtime_resource_contract(&model).unwrap();
    let slices = ["gpu0", "gpu1"]
        .into_iter()
        .map(|device_id| {
            VulkanResidentModelPackageDeviceSlicePlan::prepare_for_physical_planning(
                &package_root,
                &model,
                &contract,
                &tensor_index,
                device_id,
                64,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let components = BTreeSet::from([
        "layer_00".to_string(),
        "layer_00_remote".to_string(),
        "layer_00_tail".to_string(),
    ]);
    let retained = exact_vulkan_speculative_source_tap_signal_keys_by_device(
        &model,
        &components,
        &slices,
    )
    .unwrap();
    assert_eq!(retained.len(), 1);
    assert_eq!(retained["gpu0"].len(), 1);
    assert!(retained["gpu0"].iter().any(|key| match key {
        VulkanComponentBatchSignalKey::ModelOutput(_) => true,
        VulkanComponentBatchSignalKey::ProducedPort { component_id, .. } => {
            component_id == "layer_00_tail"
        }
        _ => false,
    }));
}

#[test]
fn runtime_hybrid_exact_prefill_candidate_replaces_sampled_transient_geometry() {
    let package_root = tiny_model_dir();
    let model = fixture_model_runtime_model();
    let phase = VulkanTargetedComponentExecutionPhase::Prefill {
        activation_batch_width: 4,
    };
    let signature = vulkan_runtime_placement_calibration_target_for_component(
        &model,
        "layer_00",
        phase,
    )
    .unwrap()
    .signature_id;
    let execution_case = hybrid_test_observation(
        hybrid_test_behavior_for_phase(
            &signature,
            nerve_execution_contracts::ExecutionPhase::Prefill,
            4,
        ),
        "gpu0",
        1,
    )
    .execution_case;
    let bindings = BTreeMap::from([("gpu0".to_string(), "gpu0".to_string())]);
    let planning_devices = [VulkanRuntimePhysicalPlanningDevice {
        logical_device_id: "gpu0".to_string(),
        identity: hybrid_test_device("gpu0"),
        safe_capacity_bytes: usize::MAX,
        storage_buffer_offset_alignment: 8,
    }];
    let planner = VulkanRuntimeHybridExactCandidateResourcePlanner {
        package_root: &package_root,
        logical_device_id_by_physical_device: &bindings,
        planning_devices: &planning_devices,
        context_capacity_activations: 64,
        speculative_draft_tokens: 0,
        residency_policy: ResourceResidencyPolicy::Eager,
    };
    let tensor_index = model.load_runtime_tensor_index(&package_root).unwrap();
    let contract = instantiate_runtime_resource_contract(&model).unwrap();
    let layout = VulkanCompiledResourceAddressLayout::from_contract(&contract).unwrap();
    let requirements = planner
        .resource_requirements(
            &model,
            phase,
            &["layer_00".to_string()],
            &execution_case,
            None,
            &tensor_index,
            &contract,
            &layout,
        )
        .unwrap();
    let transient = requirements
        .direct_claims
        .iter()
        .filter(|claim| claim.class == VulkanHybridResourceClass::ExecutionTransient)
        .collect::<Vec<_>>();
    assert_eq!(transient.len(), 1);
    assert_eq!(transient[0].target, VulkanHybridResourceTarget::Device(hybrid_test_device("gpu0")));
    assert!(transient[0].byte_count > 0);
    assert!(transient[0].claim_id.contains("prefill-width:4"));
}

#[test]
fn runtime_hybrid_exact_prefill_transient_is_authoritative_only_for_complete_route() {
    let package_root = tiny_model_dir();
    let model = fixture_model_runtime_model();
    let phase = VulkanTargetedComponentExecutionPhase::Prefill {
        activation_batch_width: 4,
    };
    let mut catalog = VulkanPlacementCalibrationCatalog::default();
    record_hybrid_phase_candidates(&model, &mut catalog, phase, 1, 2);
    let bindings = BTreeMap::from([
        ("gpu0".to_string(), "gpu0".to_string()),
        ("gpu1".to_string(), "gpu1".to_string()),
    ]);
    let planning_devices = [
        VulkanRuntimePhysicalPlanningDevice {
            logical_device_id: "gpu0".to_string(),
            identity: hybrid_test_device("gpu0"),
            safe_capacity_bytes: usize::MAX,
            storage_buffer_offset_alignment: 8,
        },
        VulkanRuntimePhysicalPlanningDevice {
            logical_device_id: "gpu1".to_string(),
            identity: hybrid_test_device("gpu1"),
            safe_capacity_bytes: usize::MAX,
            storage_buffer_offset_alignment: 8,
        },
    ];
    let planner = VulkanRuntimeHybridExactCandidateResourcePlanner {
        package_root: &package_root,
        logical_device_id_by_physical_device: &bindings,
        planning_devices: &planning_devices,
        context_capacity_activations: 64,
        speculative_draft_tokens: 0,
        residency_policy: ResourceResidencyPolicy::Eager,
    };
    let candidates = runtime_hybrid_candidate_graph(
        &model,
        &catalog,
        phase,
        None,
        VulkanRuntimeHybridComponentStrategyFilter::AnyMeasured,
        Some(&planner),
    )
    .unwrap();
    assert!(!candidates
        .authoritative_resource_classes
        .contains(&VulkanHybridResourceClass::ExecutionTransient));
    for candidate in &candidates.region_candidates {
        let resources = &candidates.resource_catalog.region_resources_by_candidate_id
            [&candidate.candidate_id];
        let claims = resources
            .claims
            .iter()
            .filter(|claim| claim.class == VulkanHybridResourceClass::ExecutionTransient)
            .collect::<Vec<_>>();
        assert_eq!(claims.len(), 1);
        assert!(claims[0].byte_count > 2, "sampled two-byte transient survived");
        assert!(claims[0].claim_id.contains("prefill-width:4"));
    }
    let unbounded = VulkanPlacementCapacityEnvelope {
        available_bytes_by_device: BTreeMap::from([
            (hybrid_test_device("gpu0"), usize::MAX),
            (hybrid_test_device("gpu1"), usize::MAX),
        ]),
        host_available_bytes: usize::MAX,
    };
    let plan = try_plan_vulkan_hybrid_ordered_graph_with_resources(
        &catalog,
        candidates.component_ids.len(),
        &candidates.region_candidates,
        &candidates.boundary_candidates,
        &candidates.resource_catalog,
        &unbounded,
    )
    .unwrap()
    .unwrap();
    let placement = runtime_hybrid_ordered_placement_from_plan(&candidates, plan);
    let route_claims = planner
        .route_execution_transient_claims(&model, phase, &placement)
        .unwrap();
    assert_eq!(route_claims.len(), 1);
    assert!(route_claims[0].byte_count > 2);
    let mut constrained = unbounded.clone();
    constrained.available_bytes_by_device.insert(
        hybrid_test_device("gpu0"),
        route_claims[0].byte_count - 1,
    );
    let route = VulkanHybridPlacementRoute {
        steps: placement.plan.steps.clone(),
        predicted_duration_ns_per_activation: placement
            .plan
            .predicted_duration_ns_per_activation,
        calibration_resource_reservations: VulkanHybridResourceReservations::default(),
        authoritative_resource_reservations: VulkanHybridResourceReservations::default(),
    };
    assert!(!runtime_hybrid_route_execution_transient_fits(
        Some(&planner),
        &model,
        phase,
        &placement,
        &route,
        &constrained,
    )
    .unwrap());
    let speculative_planner = VulkanRuntimeHybridExactCandidateResourcePlanner {
        speculative_draft_tokens: 7,
        ..planner
    };
    let speculative_route_claims = speculative_planner
        .route_execution_transient_claims(&model, phase, &placement)
        .unwrap();
    assert_eq!(speculative_route_claims.len(), 1);
    assert!(speculative_route_claims[0].byte_count > route_claims[0].byte_count);
}

#[test]
fn runtime_hybrid_exact_gate_plan_distinguishes_eager_and_demand_residency() {
    let model = fixture_model_runtime_model_with_one_dynamic_group();
    let contract = instantiate_runtime_resource_contract(&model).unwrap();
    let layout = VulkanCompiledResourceAddressLayout::from_contract(&contract).unwrap();
    let execution_plan = VulkanDistributedExecutionPlan {
        device_ids: Vec::new(),
        storage_buffer_offset_alignment: 16,
        dispatches: Vec::new(),
        execution_islands: Vec::new(),
        shared_activation_route: VulkanSharedResidentBufferRoute::SharedHost,
        shared_input_byte_capacity: 0,
        shared_output_byte_capacity: 0,
        distributed_parameter_byte_count: 0,
    };
    let components = BTreeSet::from(["layer_00".to_string()]);
    let owners = BTreeMap::from([("layer_00".to_string(), "gpu0".to_string())]);
    let eager = exact_vulkan_runtime_hybrid_gate_device_bytes(
        &components,
        &owners,
        &execution_plan,
        1,
        &contract,
        &layout,
        ResourceResidencyPolicy::Eager,
    )
    .unwrap();
    assert!(eager.is_empty());

    let one = exact_vulkan_runtime_hybrid_gate_device_bytes(
        &components,
        &owners,
        &execution_plan,
        1,
        &contract,
        &layout,
        ResourceResidencyPolicy::DemandRetained,
    )
    .unwrap();
    let four = exact_vulkan_runtime_hybrid_gate_device_bytes(
        &components,
        &owners,
        &execution_plan,
        4,
        &contract,
        &layout,
        ResourceResidencyPolicy::DemandRetained,
    )
    .unwrap();
    assert!(one["gpu0"] > size_of::<u32>());
    assert!(four["gpu0"] > one["gpu0"]);
    assert_eq!(one.len(), 1);
    assert_eq!(four.len(), 1);
}

#[test]
fn runtime_hybrid_exact_prefill_accounts_cross_device_batch_staging_once() {
    let package_root = tiny_model_dir();
    let model = fixture_model_runtime_model_with_remote_middle();
    let tensor_index = model.load_runtime_tensor_index(&package_root).unwrap();
    let contract = instantiate_runtime_resource_contract(&model).unwrap();
    let layout = VulkanCompiledResourceAddressLayout::from_contract(&contract).unwrap();
    let slice_plans = ["gpu0", "gpu1"]
        .into_iter()
        .map(|device_id| {
            VulkanResidentModelPackageDeviceSlicePlan::prepare_for_physical_planning(
                &package_root,
                &model,
                &contract,
                &tensor_index,
                device_id,
                64,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let execution_plan = VulkanDistributedExecutionPlan {
        device_ids: Vec::new(),
        storage_buffer_offset_alignment: 16,
        dispatches: Vec::new(),
        execution_islands: Vec::new(),
        shared_activation_route: VulkanSharedResidentBufferRoute::SharedHost,
        shared_input_byte_capacity: 0,
        shared_output_byte_capacity: 0,
        distributed_parameter_byte_count: 0,
    };
    let components = BTreeSet::from([
        "layer_00".to_string(),
        "layer_00_remote".to_string(),
        "layer_00_tail".to_string(),
    ]);
    let one = exact_vulkan_runtime_hybrid_prefill_transient_plan(
        &model,
        &components,
        &slice_plans,
        &execution_plan,
        1,
        1,
        &contract,
        &layout,
        ResourceResidencyPolicy::Eager,
        false,
        false,
    )
    .unwrap();
    let four = exact_vulkan_runtime_hybrid_prefill_transient_plan(
        &model,
        &components,
        &slice_plans,
        &execution_plan,
        4,
        4,
        &contract,
        &layout,
        ResourceResidencyPolicy::Eager,
        false,
        false,
    )
    .unwrap();
    assert!(one.host_bytes() > 0);
    assert_eq!(four.host_bytes(), 4 * one.host_bytes());
    assert!(!one.shared_host_allocations.is_empty());
    assert_eq!(
        one.shared_host_allocations
            .iter()
            .map(|allocation| allocation.byte_capacity)
            .sum::<usize>(),
        one.host_bytes()
    );
    assert_eq!(
        four.shared_host_allocations.len(),
        one.shared_host_allocations.len()
    );
    for (one, four) in one
        .shared_host_allocations
        .iter()
        .zip(&four.shared_host_allocations)
    {
        assert_eq!(four.mode, one.mode);
        assert_eq!(four.owner_device_id, one.owner_device_id);
        assert_eq!(four.participant_device_ids, one.participant_device_ids);
        assert_eq!(four.byte_capacity, 4 * one.byte_capacity);
        assert!(one
            .participant_device_ids
            .contains(&one.owner_device_id));
    }
    assert_eq!(one.device_bytes_by_logical_device.len(), 2);
    assert_eq!(four.device_bytes_by_logical_device.len(), 2);
}

#[test]
fn runtime_hybrid_exact_boundary_resources_match_mounted_route_backing() {
    let model = fixture_model_runtime_model_with_three_layer_series("gpu0");
    let frame_bytes = vulkan_runtime_placement_boundaries(&model).unwrap()[0].transfers[0]
        .byte_count;
    let external_case = hybrid_test_boundary_case(
        "gpu0",
        "gpu1",
        frame_bytes,
        "external_device_local",
    );
    let staged_case = hybrid_test_boundary_case(
        "gpu0",
        "gpu1",
        frame_bytes,
        "device_local_staging",
    );
    let mut catalog = VulkanPlacementCalibrationCatalog::default();
    for execution_case in [&external_case, &staged_case] {
        catalog
            .record_reference(VulkanPlacementCanonicalReference {
                behavior: execution_case.behavior.clone(),
                output_digest: "output".to_string(),
                output_artifact: None,
                state_digest: "state".to_string(),
            })
            .unwrap();
        catalog
            .record_observation(VulkanPlacementCalibrationObservation {
                execution_case: execution_case.clone(),
                warmup_call_count: 1,
                measured_call_count: 1,
                complete_transaction: true,
                duration_ns: 1,
                useful_activation_count: 1,
                output_digest: "output".to_string(),
                output_artifact: None,
                output_equivalence: VulkanPlacementOutputEquivalenceEvidence::BitExact,
                state_digest: "state".to_string(),
                resident_bytes_by_physical_device: BTreeMap::from([
                    ("gpu0".to_string(), 999),
                    ("gpu1".to_string(), 999),
                ]),
                transient_peak_bytes_by_physical_device: BTreeMap::from([
                    ("gpu0".to_string(), 999),
                    ("gpu1".to_string(), 999),
                ]),
                host_resident_bytes: 999,
                host_transient_peak_bytes: 999,
            })
            .unwrap();
    }
    let candidates = vec![
        VulkanHybridBoundaryCandidate {
            boundary_index: 0,
            byte_count: frame_bytes,
            execution_case: external_case.clone(),
        },
        VulkanHybridBoundaryCandidate {
            boundary_index: 0,
            byte_count: frame_bytes,
            execution_case: staged_case.clone(),
        },
    ];
    let mut resources =
        VulkanHybridCandidateResourceCatalog::from_calibration(&catalog, &[], &candidates)
            .unwrap();
    apply_runtime_hybrid_exact_boundary_resources(
        &mut resources,
        &model,
        &candidates,
        VulkanTargetedComponentExecutionPhase::Decode,
    )
    .unwrap();

    let external = &resources.boundary_resources_by_case[&(0, external_case)];
    assert!(
        external
            .claims
            .iter()
            .all(|claim| !runtime_hybrid_exact_authoritative_resource_classes()
                .contains(&claim.class)),
        "an external-device-local boundary reuses its source produced-port backing",
    );
    let staged = &resources.boundary_resources_by_case[&(0, staged_case)];
    let mutable = staged
        .claims
        .iter()
        .filter(|claim| claim.class == VulkanHybridResourceClass::MutableState)
        .collect::<Vec<_>>();
    assert_eq!(mutable.len(), 2);
    assert_eq!(mutable.iter().map(|claim| claim.byte_count).sum::<usize>(), 2 * frame_bytes);
    assert!(mutable.iter().any(|claim| {
        claim.target == VulkanHybridResourceTarget::Device(hybrid_test_device("gpu1"))
    }));
    assert!(
        mutable
            .iter()
            .any(|claim| claim.target == VulkanHybridResourceTarget::Host)
    );
}

#[test]
fn runtime_hybrid_exact_prefill_boundary_adds_only_real_width_scaled_staging() {
    let model = fixture_model_runtime_model_with_three_layer_series("gpu0");
    let width = 4;
    let frame_bytes = vulkan_runtime_placement_boundaries(&model).unwrap()[0].transfers[0]
        .byte_count;
    let byte_count = frame_bytes * width;
    let external_case = hybrid_test_boundary_case_for_phase(
        "gpu0",
        "gpu1",
        byte_count,
        "external_device_local",
        nerve_execution_contracts::ExecutionPhase::Prefill,
        width,
    );
    let staged_case = hybrid_test_boundary_case_for_phase(
        "gpu0",
        "gpu1",
        byte_count,
        "device_local_staging",
        nerve_execution_contracts::ExecutionPhase::Prefill,
        width,
    );
    let mut catalog = VulkanPlacementCalibrationCatalog::default();
    for execution_case in [&external_case, &staged_case] {
        catalog
            .record_reference(VulkanPlacementCanonicalReference {
                behavior: execution_case.behavior.clone(),
                output_digest: "output".to_string(),
                output_artifact: None,
                state_digest: "state".to_string(),
            })
            .unwrap();
        catalog
            .record_observation(VulkanPlacementCalibrationObservation {
                execution_case: execution_case.clone(),
                warmup_call_count: 1,
                measured_call_count: 1,
                complete_transaction: true,
                duration_ns: 1,
                useful_activation_count: width,
                output_digest: "output".to_string(),
                output_artifact: None,
                output_equivalence: VulkanPlacementOutputEquivalenceEvidence::BitExact,
                state_digest: "state".to_string(),
                resident_bytes_by_physical_device: BTreeMap::from([
                    ("gpu0".to_string(), 999),
                    ("gpu1".to_string(), 999),
                ]),
                transient_peak_bytes_by_physical_device: BTreeMap::from([
                    ("gpu0".to_string(), 999),
                    ("gpu1".to_string(), 999),
                ]),
                host_resident_bytes: 999,
                host_transient_peak_bytes: 999,
            })
            .unwrap();
    }
    let candidates = vec![
        VulkanHybridBoundaryCandidate {
            boundary_index: 0,
            byte_count,
            execution_case: external_case.clone(),
        },
        VulkanHybridBoundaryCandidate {
            boundary_index: 0,
            byte_count,
            execution_case: staged_case.clone(),
        },
    ];
    let mut resources =
        VulkanHybridCandidateResourceCatalog::from_calibration(&catalog, &[], &candidates)
            .unwrap();
    apply_runtime_hybrid_exact_boundary_resources(
        &mut resources,
        &model,
        &candidates,
        VulkanTargetedComponentExecutionPhase::Prefill {
            activation_batch_width: width,
        },
    )
    .unwrap();

    let external = &resources.boundary_resources_by_case[&(0, external_case)];
    assert!(
        external
            .claims
            .iter()
            .all(|claim| claim.class != VulkanHybridResourceClass::ExecutionTransient)
    );
    let staged = &resources.boundary_resources_by_case[&(0, staged_case)];
    let transient = staged
        .claims
        .iter()
        .filter(|claim| claim.class == VulkanHybridResourceClass::ExecutionTransient)
        .collect::<Vec<_>>();
    assert_eq!(transient.len(), 1);
    assert_eq!(transient[0].target, VulkanHybridResourceTarget::Host);
    assert_eq!(transient[0].byte_count, byte_count);
    assert!(!runtime_hybrid_exact_authoritative_resource_classes()
        .contains(&VulkanHybridResourceClass::ExecutionTransient));
}

#[test]
fn runtime_hybrid_exact_region_accounts_internal_staged_boundaries() {
    let package_root = tiny_model_dir();
    let model = fixture_model_runtime_model_with_remote_middle();
    let phase = VulkanTargetedComponentExecutionPhase::Decode;
    let mut catalog = VulkanPlacementCalibrationCatalog::default();
    let region_case =
        record_hybrid_test_serialized_region(&model, &mut catalog, phase, 1);
    let bindings = BTreeMap::from([
        ("gpu0".to_string(), "gpu0".to_string()),
        ("gpu1".to_string(), "gpu1".to_string()),
    ]);
    let planning_devices = [
        VulkanRuntimePhysicalPlanningDevice {
            logical_device_id: "gpu0".to_string(),
            identity: hybrid_test_device("gpu0"),
            safe_capacity_bytes: usize::MAX,
            storage_buffer_offset_alignment: 8,
        },
        VulkanRuntimePhysicalPlanningDevice {
            logical_device_id: "gpu1".to_string(),
            identity: hybrid_test_device("gpu1"),
            safe_capacity_bytes: usize::MAX,
            storage_buffer_offset_alignment: 8,
        },
    ];
    let planner = VulkanRuntimeHybridExactCandidateResourcePlanner {
        package_root: &package_root,
        logical_device_id_by_physical_device: &bindings,
        planning_devices: &planning_devices,
        context_capacity_activations: 64,
        speculative_draft_tokens: 0,
        residency_policy: ResourceResidencyPolicy::Eager,
    };
    let candidates = runtime_hybrid_candidate_graph(
        &model,
        &catalog,
        phase,
        None,
        VulkanRuntimeHybridComponentStrategyFilter::AnyMeasured,
        Some(&planner),
    )
    .unwrap();
    let region = candidates
        .region_candidates
        .iter()
        .find(|candidate| candidate.execution_case == region_case)
        .unwrap();
    let resources = &candidates.resource_catalog.region_resources_by_candidate_id
        [&region.candidate_id];
    let host_mutable = resources
        .claims
        .iter()
        .filter(|claim| {
            claim.class == VulkanHybridResourceClass::MutableState
                && claim.target == VulkanHybridResourceTarget::Host
        })
        .collect::<Vec<_>>();
    let expected = vulkan_runtime_placement_boundaries(&model)
        .unwrap()
        .iter()
        .map(|boundary| boundary.transfers[0].byte_count)
        .sum::<usize>();
    assert_eq!(host_mutable.len(), 2);
    assert_eq!(host_mutable.iter().map(|claim| claim.byte_count).sum::<usize>(), expected);
    assert!(host_mutable.iter().all(|claim| claim.shared));
}
