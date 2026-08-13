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

    normal.extend(verification).unwrap();

    assert_eq!(normal.host_bytes(), 46);
    assert_eq!(normal.shared_host_allocations.len(), 2);
    assert_eq!(normal.shared_host_allocations[0].byte_capacity, 17);
    assert_eq!(normal.shared_host_allocations[1].byte_capacity, 29);
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
