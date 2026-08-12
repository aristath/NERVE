fn hybrid_test_digest(byte: char) -> String {
    format!("sha256:{}", byte.to_string().repeat(64))
}

fn hybrid_test_behavior(signature: &str) -> VulkanPlacementBehaviorIdentity {
    VulkanPlacementBehaviorIdentity {
        compiled_execution_signature: signature.to_string(),
        contract_ids: vec!["contract".to_string()],
        implementation_digests: vec![hybrid_test_digest('a')],
        artifact_digest: hybrid_test_digest('b'),
        execution_graph_digest: hybrid_test_digest('c'),
        runtime_implementation_fingerprint: "runtime".to_string(),
        phase: nerve_execution_contracts::ExecutionPhase::Decode,
        shape: VulkanPlacementShapeClass {
            activation_batch_width: 1,
            input_byte_capacity: 16,
            output_byte_capacity: 16,
            operations: vec![VulkanPlacementOperationGeometry::Dispatch {
                geometry: VulkanPlacementDispatchGeometry {
                    contract_id: "contract".to_string(),
                    logical_extent: 8,
                    sampled_extent: 8,
                    input_width: 8,
                    workgroup_count_x: 1,
                    local_size_x: 64,
                },
            }],
        },
        input_fixture_digest: hybrid_test_digest('d'),
        equivalence: VulkanPlacementEquivalenceIdentity::bit_exact(),
    }
}

fn hybrid_test_behavior_for_phase(
    signature: &str,
    phase: nerve_execution_contracts::ExecutionPhase,
    activation_batch_width: usize,
) -> VulkanPlacementBehaviorIdentity {
    let mut behavior = hybrid_test_behavior(signature);
    behavior.phase = phase;
    behavior.shape.activation_batch_width = activation_batch_width;
    behavior
}

fn hybrid_test_device(id: &str) -> VulkanPlacementDeviceExecutionIdentity {
    VulkanPlacementDeviceExecutionIdentity {
        physical_device_id: id.to_string(),
        api_version: 1,
        driver_version: 2,
    }
}

fn hybrid_test_observation(
    behavior: VulkanPlacementBehaviorIdentity,
    device_id: &str,
    duration_ns: u64,
) -> VulkanPlacementCalibrationObservation {
    let useful_activation_count = behavior.shape.activation_batch_width;
    let device = hybrid_test_device(device_id);
    VulkanPlacementCalibrationObservation {
        execution_case: VulkanPlacementExecutionCaseIdentity {
            behavior,
            strategy: VulkanPlacementExecutionStrategy::SingleDevice,
            devices: vec![device],
            shards: Vec::new(),
            input_physical_device_id: device_id.to_string(),
            output_physical_device_id: device_id.to_string(),
            owner_physical_device_id: device_id.to_string(),
            transports: Vec::new(),
        },
        warmup_call_count: 1,
        measured_call_count: 1,
        complete_transaction: true,
        duration_ns,
        useful_activation_count,
        output_digest: "output".to_string(),
        output_artifact: None,
        output_equivalence: VulkanPlacementOutputEquivalenceEvidence::BitExact,
        state_digest: "state".to_string(),
        resident_bytes_by_physical_device: BTreeMap::from([(device_id.to_string(), 10)]),
        transient_peak_bytes_by_physical_device: BTreeMap::from([(device_id.to_string(), 2)]),
        host_resident_bytes: 0,
        host_transient_peak_bytes: 0,
    }
}

fn record_hybrid_phase_candidates(
    model: &VulkanResidentRuntimeModel,
    catalog: &mut VulkanPlacementCalibrationCatalog,
    phase: VulkanTargetedComponentExecutionPhase,
    gpu0_duration_ns: u64,
    gpu1_duration_ns: u64,
) {
    let execution_phase = match phase {
        VulkanTargetedComponentExecutionPhase::Decode => {
            nerve_execution_contracts::ExecutionPhase::Decode
        }
        VulkanTargetedComponentExecutionPhase::Prefill { .. } => {
            nerve_execution_contracts::ExecutionPhase::Prefill
        }
    };
    let mut signatures = model
        .circuit_graph
        .components
        .iter()
        .filter(|component| component.runtime_role.is_signal_processor())
        .map(|component| {
            vulkan_runtime_placement_calibration_target_for_component(
                model,
                &component.component_id,
                phase,
            )
            .unwrap()
            .signature_id
        })
        .collect::<Vec<_>>();
    signatures.sort();
    signatures.dedup();
    for signature in signatures {
        let behavior = hybrid_test_behavior_for_phase(
            &signature,
            execution_phase,
            phase.activation_batch_width(),
        );
        catalog
            .record_reference(VulkanPlacementCanonicalReference {
                behavior: behavior.clone(),
                output_digest: "output".to_string(),
                output_artifact: None,
                state_digest: "state".to_string(),
            })
            .unwrap();
        catalog
            .record_observation(hybrid_test_observation(
                behavior.clone(),
                "gpu0",
                gpu0_duration_ns,
            ))
            .unwrap();
        catalog
            .record_observation(hybrid_test_observation(
                behavior,
                "gpu1",
                gpu1_duration_ns,
            ))
            .unwrap();
    }
}

fn hybrid_test_distributed_observation(
    behavior: VulkanPlacementBehaviorIdentity,
    duration_ns: u64,
) -> VulkanPlacementCalibrationObservation {
    VulkanPlacementCalibrationObservation {
        execution_case: VulkanPlacementExecutionCaseIdentity {
            behavior,
            strategy: VulkanPlacementExecutionStrategy::TensorParallel,
            devices: vec![hybrid_test_device("gpu0"), hybrid_test_device("gpu1")],
            shards: vec![
                VulkanPlacementShardIdentity {
                    dispatch_ordinal: 0,
                    participant_ordinal: 0,
                    physical_device_id: "gpu0".to_string(),
                    distribution: "output_rows".to_string(),
                    logical_start: 0,
                    logical_count: 4,
                    selected_resource_indices_by_partition: BTreeMap::new(),
                    parameter_bytes: 5,
                },
                VulkanPlacementShardIdentity {
                    dispatch_ordinal: 0,
                    participant_ordinal: 1,
                    physical_device_id: "gpu1".to_string(),
                    distribution: "output_rows".to_string(),
                    logical_start: 4,
                    logical_count: 4,
                    selected_resource_indices_by_partition: BTreeMap::new(),
                    parameter_bytes: 5,
                },
            ],
            input_physical_device_id: "gpu0".to_string(),
            output_physical_device_id: "gpu0".to_string(),
            owner_physical_device_id: "gpu0".to_string(),
            transports: Vec::new(),
        },
        warmup_call_count: 1,
        measured_call_count: 1,
        complete_transaction: true,
        duration_ns,
        useful_activation_count: 1,
        output_digest: "output".to_string(),
        output_artifact: None,
        output_equivalence: VulkanPlacementOutputEquivalenceEvidence::BitExact,
        state_digest: "state".to_string(),
        resident_bytes_by_physical_device: BTreeMap::from([
            ("gpu0".to_string(), 5),
            ("gpu1".to_string(), 5),
        ]),
        transient_peak_bytes_by_physical_device: BTreeMap::from([
            ("gpu0".to_string(), 2),
            ("gpu1".to_string(), 2),
        ]),
        host_resident_bytes: 0,
        host_transient_peak_bytes: 0,
    }
}

fn hybrid_test_catalog(model: &VulkanResidentRuntimeModel) -> VulkanPlacementCalibrationCatalog {
    let mut catalog = VulkanPlacementCalibrationCatalog::default();
    let mut signatures = model
        .circuit_graph
        .components
        .iter()
        .filter(|component| component.runtime_role.is_signal_processor())
        .map(|component| {
            vulkan_runtime_placement_calibration_target_for_component(
                model,
                &component.component_id,
                VulkanTargetedComponentExecutionPhase::Decode,
            )
            .unwrap()
            .signature_id
        })
        .collect::<Vec<_>>();
    signatures.sort();
    signatures.dedup();
    for signature in signatures {
        let behavior = hybrid_test_behavior(&signature);
        catalog
            .record_reference(VulkanPlacementCanonicalReference {
                behavior: behavior.clone(),
                output_digest: "output".to_string(),
                output_artifact: None,
                state_digest: "state".to_string(),
            })
            .unwrap();
        catalog
            .record_observation(hybrid_test_observation(behavior.clone(), "gpu0", 10))
            .unwrap();
        catalog
            .record_observation(hybrid_test_observation(behavior, "gpu1", 12))
            .unwrap();
    }
    catalog
}

#[test]
fn runtime_hybrid_planner_maps_compiler_signatures_to_every_component_instance() {
    let model = fixture_model_runtime_model_with_three_layer_series("gpu0");
    let catalog = hybrid_test_catalog(&model);
    let capacity = VulkanPlacementCapacityEnvelope {
        available_bytes_by_device: BTreeMap::from([
            (hybrid_test_device("gpu0"), 100),
            (hybrid_test_device("gpu1"), 100),
        ]),
        host_available_bytes: 100,
    };

    let placement = plan_vulkan_runtime_hybrid_ordered_graph(
        &model,
        &catalog,
        &capacity,
        VulkanTargetedComponentExecutionPhase::Decode,
    )
    .unwrap();

    assert_eq!(placement.component_ids.len(), 3);
    assert_eq!(placement.plan.steps.len(), 3);
    assert_eq!(placement.plan.predicted_duration_ns_per_activation, 30);
    assert!(placement.plan.steps.iter().all(|step| matches!(
        step,
        VulkanHybridScheduledStep::Region { execution_case, .. }
            if execution_case.owner_physical_device_id == "gpu0"
    )));
}

#[test]
fn runtime_hybrid_planner_rejects_missing_or_ambiguous_behavior_evidence() {
    let model = fixture_model_runtime_model_with_three_layer_series("gpu0");
    let capacity = VulkanPlacementCapacityEnvelope {
        available_bytes_by_device: BTreeMap::from([(hybrid_test_device("gpu0"), 100)]),
        host_available_bytes: 100,
    };
    let missing = plan_vulkan_runtime_hybrid_ordered_graph(
        &model,
        &VulkanPlacementCalibrationCatalog::default(),
        &capacity,
        VulkanTargetedComponentExecutionPhase::Decode,
    )
    .unwrap_err();
    assert!(missing.0.contains("0 exact calibration behavior cohorts"));

    let mut catalog = hybrid_test_catalog(&model);
    let target = vulkan_runtime_placement_calibration_target_for_component(
        &model,
        &model.component_executions[0].component_id,
        VulkanTargetedComponentExecutionPhase::Decode,
    )
    .unwrap();
    let mut second_behavior = hybrid_test_behavior(&target.signature_id);
    let VulkanPlacementOperationGeometry::Dispatch { geometry } =
        &mut second_behavior.shape.operations[0]
    else {
        panic!("hybrid fixture operation must be a dispatch");
    };
    geometry.sampled_extent = 4;
    second_behavior.input_fixture_digest = hybrid_test_digest('e');
    catalog
        .record_reference(VulkanPlacementCanonicalReference {
            behavior: second_behavior.clone(),
            output_digest: "output".to_string(),
            output_artifact: None,
            state_digest: "state".to_string(),
        })
        .unwrap();
    catalog
        .record_observation(hybrid_test_observation(second_behavior, "gpu0", 9))
        .unwrap();

    let ambiguous = plan_vulkan_runtime_hybrid_ordered_graph(
        &model,
        &catalog,
        &capacity,
        VulkanTargetedComponentExecutionPhase::Decode,
    )
    .unwrap_err();
    assert!(ambiguous.0.contains("2 exact calibration behavior cohorts"));
}

#[test]
fn runtime_hybrid_lowering_keeps_internal_shards_phase_local() {
    let model = fixture_model_runtime_model_with_three_layer_series("gpu0");
    let mut catalog = VulkanPlacementCalibrationCatalog::default();
    let mut signatures = model
        .circuit_graph
        .components
        .iter()
        .filter(|component| component.runtime_role.is_signal_processor())
        .map(|component| {
            vulkan_runtime_placement_calibration_target_for_component(
                &model,
                &component.component_id,
                VulkanTargetedComponentExecutionPhase::Decode,
            )
            .unwrap()
            .signature_id
        })
        .collect::<Vec<_>>();
    signatures.sort();
    signatures.dedup();
    for signature in signatures {
        let behavior = hybrid_test_behavior(&signature);
        catalog
            .record_reference(VulkanPlacementCanonicalReference {
                behavior: behavior.clone(),
                output_digest: "output".to_string(),
                output_artifact: None,
                state_digest: "state".to_string(),
            })
            .unwrap();
        catalog
            .record_observation(hybrid_test_distributed_observation(behavior, 8))
            .unwrap();
    }
    let capacity = VulkanPlacementCapacityEnvelope {
        available_bytes_by_device: BTreeMap::from([
            (hybrid_test_device("gpu0"), 100),
            (hybrid_test_device("gpu1"), 100),
        ]),
        host_available_bytes: 100,
    };
    let placement = plan_vulkan_runtime_hybrid_ordered_graph(
        &model,
        &catalog,
        &capacity,
        VulkanTargetedComponentExecutionPhase::Decode,
    )
    .unwrap();
    let bindings = BTreeMap::from([
        ("gpu0".to_string(), "logical-owner".to_string()),
        ("gpu1".to_string(), "logical-helper".to_string()),
    ]);
    let lowered =
        lower_vulkan_runtime_hybrid_phase_placement(&model, &placement, &bindings).unwrap();

    assert_eq!(
        lowered.execution_phase,
        nerve_execution_contracts::ExecutionPhase::Decode
    );
    assert_eq!(lowered.activation_batch_width, 1);
    assert_eq!(lowered.execution_cases_by_component.len(), 3);
    assert!(
        lowered
            .component_device_pools
            .values()
            .all(|devices| devices == &["logical-owner", "logical-helper"])
    );
    assert!(
        lowered
            .runtime_model
            .placement
            .component_shard_devices
            .is_empty()
    );
    assert!(
        lowered
            .runtime_model
            .circuit_graph
            .components
            .iter()
            .filter(|component| component.runtime_role.is_signal_processor())
            .all(|component| {
                lowered
                    .runtime_model
                    .placement
                    .device_for_component(&component.component_id)
                    == "logical-owner"
            })
    );

    let mut stale = placement.clone();
    let VulkanHybridScheduledStep::Region { execution_case, .. } = &mut stale.plan.steps[0] else {
        panic!("first hybrid step must be a component region");
    };
    execution_case.behavior.compiled_execution_signature = "stale-signature".to_string();
    assert!(
        lower_vulkan_runtime_hybrid_phase_placement(&model, &stale, &bindings)
            .unwrap_err()
            .0
            .contains("does not match compiled component")
    );

    let mut conflicting_order = placement.clone();
    let VulkanHybridScheduledStep::Region { execution_case, .. } =
        &mut conflicting_order.plan.steps[0]
    else {
        panic!("first hybrid step must be a component region");
    };
    execution_case.shards[1].participant_ordinal = 0;
    assert!(
        lower_vulkan_runtime_hybrid_phase_placement(&model, &conflicting_order, &bindings)
            .unwrap_err()
            .0
            .contains("conflicting calibrated participant order")
    );

    let mut incomplete = placement;
    let VulkanHybridScheduledStep::Region { execution_case, .. } = &mut incomplete.plan.steps[0]
    else {
        panic!("first hybrid step must be a component region");
    };
    execution_case.shards.clear();
    assert!(
        lower_vulkan_runtime_hybrid_phase_placement(&model, &incomplete, &bindings)
            .unwrap_err()
            .0
            .contains("exact shard coverage")
    );

    let missing_binding = BTreeMap::from([("gpu0".to_string(), "logical-owner".to_string())]);
    assert!(
        lower_vulkan_runtime_hybrid_phase_placement(
            &model,
            &plan_vulkan_runtime_hybrid_ordered_graph(
                &model,
                &catalog,
                &capacity,
                VulkanTargetedComponentExecutionPhase::Decode,
            )
            .unwrap(),
            &missing_binding,
        )
        .unwrap_err()
        .0
        .contains("unbound physical device")
    );
}

#[test]
fn runtime_hybrid_phase_set_keeps_decode_owners_while_optimizing_prefill() {
    let mut model = fixture_model_runtime_model_with_three_layer_series("gpu0");
    for execution in &mut model.component_executions {
        let mut prefill_terminal = execution.kernels.last().unwrap().clone();
        prefill_terminal.execution_index += 1;
        prefill_terminal.node_id = format!("{}_prefill", prefill_terminal.node_id);
        prefill_terminal.execution_domain =
            VulkanResidentComponentKernelExecutionDomain::Prefill;
        execution.kernels.push(prefill_terminal);
    }
    let mut catalog = VulkanPlacementCalibrationCatalog::default();
    record_hybrid_phase_candidates(
        &model,
        &mut catalog,
        VulkanTargetedComponentExecutionPhase::Decode,
        5,
        10,
    );
    record_hybrid_phase_candidates(
        &model,
        &mut catalog,
        VulkanTargetedComponentExecutionPhase::Prefill {
            activation_batch_width: 4,
        },
        20,
        1,
    );
    let capacity = VulkanPlacementCapacityEnvelope {
        available_bytes_by_device: BTreeMap::from([
            (hybrid_test_device("gpu0"), 100),
            (hybrid_test_device("gpu1"), 100),
        ]),
        host_available_bytes: 100,
    };

    let phase_set = plan_vulkan_runtime_hybrid_phase_set(
        &model,
        &catalog,
        &capacity,
        Some(4),
    )
    .unwrap();
    assert!(phase_set.decode.plan.steps.iter().all(|step| matches!(
        step,
        VulkanHybridScheduledStep::Region { execution_case, .. }
            if execution_case.owner_physical_device_id == "gpu0"
    )));
    assert!(phase_set
        .prefill
        .as_ref()
        .unwrap()
        .plan
        .steps
        .iter()
        .all(|step| matches!(
            step,
            VulkanHybridScheduledStep::Region { execution_case, .. }
                if execution_case.owner_physical_device_id == "gpu0"
        )));

    let bindings = BTreeMap::from([
        ("gpu0".to_string(), "logical0".to_string()),
        ("gpu1".to_string(), "logical1".to_string()),
    ]);
    let (stable_model, physical_plan) =
        lower_vulkan_runtime_hybrid_phase_set(&model, &phase_set, &bindings).unwrap();
    assert!(stable_model
        .circuit_graph
        .components
        .iter()
        .filter(|component| component.runtime_role.is_signal_processor())
        .all(|component| stable_model
            .placement
            .device_for_component(&component.component_id)
            == "logical0"));
    assert_eq!(physical_plan.decode_execution_cases_by_component.len(), 3);
    assert_eq!(physical_plan.prefill_execution_cases_by_component.len(), 3);
    assert!(physical_plan.component_device_pools.decode.is_empty());
    assert!(physical_plan.component_device_pools.prefill.is_empty());
}
