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
        useful_activation_count: 1,
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
