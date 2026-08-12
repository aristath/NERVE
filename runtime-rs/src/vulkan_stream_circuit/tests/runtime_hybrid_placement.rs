fn hybrid_test_digest(byte: char) -> String {
    format!("sha256:{}", byte.to_string().repeat(64))
}

fn hybrid_test_behavior(signature: &str) -> VulkanPlacementBehaviorIdentity {
    VulkanPlacementBehaviorIdentity {
        compiled_execution_signature: signature.to_string(),
        runtime_implementation_fingerprint: crate::RUNTIME_IMPLEMENTATION_FINGERPRINT.to_string(),
        phase: nerve_execution_contracts::ExecutionPhase::Decode,
        shape: VulkanPlacementShapeClass {
            activation_batch_width: 1,
            input_byte_capacity: 16,
            output_byte_capacity: 16,
        },
        input_fixture_digest: hybrid_test_digest('d'),
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
            contract_ids: vec!["contract".to_string()],
            implementation_digests: vec![hybrid_test_digest('a')],
            artifact_digest: hybrid_test_digest('b'),
            execution_graph_digest: hybrid_test_digest('c'),
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
            equivalence: VulkanPlacementEquivalenceIdentity::bit_exact(),
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
            .record_observation(hybrid_test_observation(behavior, "gpu1", gpu1_duration_ns))
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
            contract_ids: vec!["contract".to_string()],
            implementation_digests: vec![hybrid_test_digest('a')],
            artifact_digest: hybrid_test_digest('b'),
            execution_graph_digest: hybrid_test_digest('c'),
            operations: vec![VulkanPlacementOperationGeometry::Dispatch {
                geometry: VulkanPlacementDispatchGeometry {
                    contract_id: "contract".to_string(),
                    logical_extent: 8,
                    sampled_extent: 8,
                    input_width: 8,
                    workgroup_count_x: 2,
                    local_size_x: 64,
                },
            }],
            equivalence: VulkanPlacementEquivalenceIdentity::bit_exact(),
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
                    selected_resource_fragments_by_partition: BTreeMap::new(),
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
                    selected_resource_fragments_by_partition: BTreeMap::new(),
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

fn hybrid_test_serialized_observation(
    behavior: VulkanPlacementBehaviorIdentity,
    duration_ns: u64,
) -> VulkanPlacementCalibrationObservation {
    let mut observation = hybrid_test_distributed_observation(behavior, duration_ns);
    observation.execution_case.strategy = VulkanPlacementExecutionStrategy::Serialized;
    observation.execution_case.shards.clear();
    observation
}

fn hybrid_test_boundary_case(
    source_device_id: &str,
    destination_device_id: &str,
    byte_count: usize,
    route: &str,
) -> VulkanPlacementExecutionCaseIdentity {
    hybrid_test_boundary_case_for_phase(
        source_device_id,
        destination_device_id,
        byte_count,
        route,
        nerve_execution_contracts::ExecutionPhase::Decode,
        1,
    )
}

fn hybrid_test_boundary_case_for_phase(
    source_device_id: &str,
    destination_device_id: &str,
    byte_count: usize,
    route: &str,
    phase: nerve_execution_contracts::ExecutionPhase,
    activation_batch_width: usize,
) -> VulkanPlacementExecutionCaseIdentity {
    let mut devices = vec![
        hybrid_test_device(source_device_id),
        hybrid_test_device(destination_device_id),
    ];
    devices.sort();
    VulkanPlacementExecutionCaseIdentity {
        behavior: VulkanPlacementBehaviorIdentity {
            compiled_execution_signature: hybrid_test_digest('f'),
            runtime_implementation_fingerprint: crate::RUNTIME_IMPLEMENTATION_FINGERPRINT
                .to_string(),
            phase,
            shape: VulkanPlacementShapeClass {
                activation_batch_width,
                input_byte_capacity: byte_count,
                output_byte_capacity: byte_count,
            },
            input_fixture_digest: hybrid_test_digest('1'),
        },
        contract_ids: vec!["boundary".to_string()],
        implementation_digests: vec![hybrid_test_digest('2')],
        artifact_digest: hybrid_test_digest('3'),
        execution_graph_digest: hybrid_test_digest('f'),
        operations: vec![VulkanPlacementOperationGeometry::DirectedTransfer {
            contract_id: "boundary".to_string(),
            byte_count,
        }],
        equivalence: VulkanPlacementEquivalenceIdentity::bit_exact(),
        strategy: VulkanPlacementExecutionStrategy::DirectedBoundary,
        devices,
        shards: Vec::new(),
        input_physical_device_id: source_device_id.to_string(),
        output_physical_device_id: destination_device_id.to_string(),
        owner_physical_device_id: source_device_id.to_string(),
        transports: vec![VulkanPlacementTransportIdentity {
            source_physical_device_id: source_device_id.to_string(),
            destination_physical_device_id: destination_device_id.to_string(),
            byte_capacity: byte_count,
            route: route.to_string(),
        }],
    }
}

fn record_hybrid_test_serialized_region(
    model: &VulkanResidentRuntimeModel,
    catalog: &mut VulkanPlacementCalibrationCatalog,
    phase: VulkanTargetedComponentExecutionPhase,
    duration_ns: u64,
) -> VulkanPlacementExecutionCaseIdentity {
    let execution_phase = runtime_hybrid_execution_phase(phase).unwrap();
    let activation_batch_width = phase.activation_batch_width();
    let component_ids = model
        .circuit_graph
        .components
        .iter()
        .filter(|component| component.runtime_role.is_signal_processor())
        .map(|component| component.component_id.clone())
        .collect::<Vec<_>>();
    let physical_devices = ["gpu0", "gpu1", "gpu0"];
    assert_eq!(component_ids.len(), physical_devices.len());
    let component_cases = component_ids
        .iter()
        .zip(physical_devices)
        .map(|(component_id, physical_device_id)| {
            let target = vulkan_runtime_placement_calibration_target_for_component(
                model,
                component_id,
                phase,
            )
            .unwrap();
            hybrid_test_observation(
                hybrid_test_behavior_for_phase(
                    &target.signature_id,
                    execution_phase,
                    activation_batch_width,
                ),
                physical_device_id,
                10,
            )
            .execution_case
        })
        .collect::<Vec<_>>();
    let graph_boundaries = vulkan_runtime_placement_boundaries(model).unwrap();
    let boundary_byte_counts = graph_boundaries
        .iter()
        .map(|boundary| {
            let [transfer] = boundary.transfers.as_slice() else {
                panic!("fixture boundary must contain one transfer");
            };
            assert!(transfer.source_in_prefix);
            transfer.byte_count * activation_batch_width
        })
        .collect::<Vec<_>>();
    let boundary_cases = vec![
        VulkanPlacementRegionBoundaryExecutionCase {
            boundary_ordinal: 0,
            execution_case: hybrid_test_boundary_case_for_phase(
                "gpu0",
                "gpu1",
                boundary_byte_counts[0],
                "device_local_staging",
                execution_phase,
                activation_batch_width,
            ),
        },
        VulkanPlacementRegionBoundaryExecutionCase {
            boundary_ordinal: 1,
            execution_case: hybrid_test_boundary_case_for_phase(
                "gpu1",
                "gpu0",
                boundary_byte_counts[1],
                "device_local_staging",
                execution_phase,
                activation_batch_width,
            ),
        },
    ];
    let signature = vulkan_placement_region_compiled_execution_signature(
        &component_cases
            .iter()
            .map(|case| case.behavior.compiled_execution_signature.clone())
            .collect::<Vec<_>>(),
        &boundary_byte_counts,
    )
    .unwrap();
    let behavior = hybrid_test_behavior_for_phase(
        &signature,
        execution_phase,
        activation_batch_width,
    );
    let mut contract_implementations = BTreeMap::new();
    let mut operations = Vec::new();
    let mut transports = Vec::new();
    for ordinal in 0..component_cases.len() {
        let component = &component_cases[ordinal];
        for (contract, implementation) in component
            .contract_ids
            .iter()
            .cloned()
            .zip(component.implementation_digests.iter().cloned())
        {
            contract_implementations.insert(contract, implementation);
        }
        operations.extend(component.operations.iter().cloned());
        if let Some(boundary) = boundary_cases
            .iter()
            .find(|boundary| boundary.boundary_ordinal == ordinal)
        {
            for (contract, implementation) in boundary
                .execution_case
                .contract_ids
                .iter()
                .cloned()
                .zip(
                    boundary
                        .execution_case
                        .implementation_digests
                        .iter()
                        .cloned(),
                )
            {
                contract_implementations.insert(contract, implementation);
            }
            operations.extend(boundary.execution_case.operations.iter().cloned());
            transports.extend(boundary.execution_case.transports.iter().cloned());
        }
    }
    transports.sort();
    let execution_case = VulkanPlacementExecutionCaseIdentity {
        behavior: behavior.clone(),
        contract_ids: contract_implementations.keys().cloned().collect(),
        implementation_digests: contract_implementations.values().cloned().collect(),
        artifact_digest: hybrid_test_digest('8'),
        execution_graph_digest: hybrid_test_digest('9'),
        operations,
        equivalence: VulkanPlacementEquivalenceIdentity::bit_exact(),
        strategy: VulkanPlacementExecutionStrategy::SerializedRegion,
        devices: vec![hybrid_test_device("gpu0"), hybrid_test_device("gpu1")],
        shards: Vec::new(),
        input_physical_device_id: "gpu0".to_string(),
        output_physical_device_id: "gpu0".to_string(),
        owner_physical_device_id: "gpu0".to_string(),
        transports,
    };
    catalog
        .record_reference(VulkanPlacementCanonicalReference {
            behavior: behavior.clone(),
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
            duration_ns,
            useful_activation_count: activation_batch_width,
            output_digest: "output".to_string(),
            output_artifact: None,
            output_equivalence: VulkanPlacementOutputEquivalenceEvidence::BitExact,
            state_digest: "state".to_string(),
            resident_bytes_by_physical_device: BTreeMap::from([
                ("gpu0".to_string(), 10),
                ("gpu1".to_string(), 10),
            ]),
            transient_peak_bytes_by_physical_device: BTreeMap::from([
                ("gpu0".to_string(), 2),
                ("gpu1".to_string(), 2),
            ]),
            host_resident_bytes: 0,
            host_transient_peak_bytes: 0,
        })
        .unwrap();
    catalog
        .record_region_execution(VulkanPlacementRegionExecutionCalibration {
            execution_case: execution_case.clone(),
            boundary_byte_counts,
            component_cases,
            boundary_cases,
        })
        .unwrap();
    execution_case
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

fn hybrid_test_distributed_catalog(
    model: &VulkanResidentRuntimeModel,
) -> VulkanPlacementCalibrationCatalog {
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
            .record_observation(hybrid_test_distributed_observation(behavior, 8))
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
fn runtime_serialized_reference_planner_excludes_faster_tp_candidates() {
    let model = fixture_model_runtime_model_with_three_layer_series("gpu0");
    let mut catalog = hybrid_test_catalog(&model);
    let behaviors = model
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
        .collect::<BTreeSet<_>>();
    for signature in behaviors {
        let behavior = catalog
            .candidate_behaviors_for_compiled_execution(
                &signature,
                crate::RUNTIME_IMPLEMENTATION_FINGERPRINT,
                nerve_execution_contracts::ExecutionPhase::Decode,
            )[0]
            .clone();
        catalog
            .record_observation(hybrid_test_distributed_observation(behavior, 1))
            .unwrap();
    }
    let capacity = VulkanPlacementCapacityEnvelope {
        available_bytes_by_device: BTreeMap::from([
            (hybrid_test_device("gpu0"), 100),
            (hybrid_test_device("gpu1"), 100),
        ]),
        host_available_bytes: 100,
    };

    let hybrid = plan_vulkan_runtime_hybrid_ordered_graph(
        &model,
        &catalog,
        &capacity,
        VulkanTargetedComponentExecutionPhase::Decode,
    )
    .unwrap();
    let serialized = plan_vulkan_runtime_serialized_ordered_graph(
        &model,
        &catalog,
        &capacity,
        VulkanTargetedComponentExecutionPhase::Decode,
    )
    .unwrap();

    assert!(hybrid.plan.steps.iter().all(|step| matches!(
        step,
        VulkanHybridScheduledStep::Region { execution_case, .. }
            if execution_case.strategy == VulkanPlacementExecutionStrategy::TensorParallel
    )));
    assert!(serialized.plan.steps.iter().all(|step| matches!(
        step,
        VulkanHybridScheduledStep::Region { execution_case, .. }
            if execution_case.strategy == VulkanPlacementExecutionStrategy::SingleDevice
    )));
    assert!(
        hybrid.plan.predicted_duration_ns_per_activation
            < serialized.plan.predicted_duration_ns_per_activation
    );
}

#[test]
fn runtime_hybrid_planner_ignores_unreplayable_serialized_regions() {
    let model = fixture_model_runtime_model_with_three_layer_series("gpu0");
    let mut catalog = hybrid_test_catalog(&model);
    let behaviors = catalog
        .candidate_behaviors_for_compiled_execution(
            &vulkan_runtime_placement_calibration_target_for_component(
                &model,
                &model.component_executions[0].component_id,
                VulkanTargetedComponentExecutionPhase::Decode,
            )
            .unwrap()
            .signature_id,
            crate::RUNTIME_IMPLEMENTATION_FINGERPRINT,
            nerve_execution_contracts::ExecutionPhase::Decode,
        )
        .into_iter()
        .cloned()
        .collect::<Vec<_>>();
    for behavior in behaviors {
        catalog
            .record_observation(hybrid_test_serialized_observation(behavior, 1))
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

    assert!(placement.plan.steps.iter().all(|step| matches!(
        step,
        VulkanHybridScheduledStep::Region { execution_case, .. }
            if execution_case.strategy != VulkanPlacementExecutionStrategy::Serialized
    )));
}

#[test]
fn runtime_hybrid_planner_selects_one_complete_exact_serialized_region() {
    let model = fixture_model_runtime_model_with_three_layer_series("gpu0");
    let mut catalog = VulkanPlacementCalibrationCatalog::default();
    let region_case = record_hybrid_test_serialized_region(
        &model,
        &mut catalog,
        VulkanTargetedComponentExecutionPhase::Decode,
        1,
    );
    let capacity = VulkanPlacementCapacityEnvelope {
        available_bytes_by_device: BTreeMap::from([
            (hybrid_test_device("gpu0"), 100),
            (hybrid_test_device("gpu1"), 100),
        ]),
        host_available_bytes: 100,
    };

    assert!(
        vulkan_runtime_hybrid_phase_is_calibrated(
            &model,
            &catalog,
            VulkanTargetedComponentExecutionPhase::Decode,
        )
        .unwrap()
    );

    let placement = plan_vulkan_runtime_hybrid_ordered_graph(
        &model,
        &catalog,
        &capacity,
        VulkanTargetedComponentExecutionPhase::Decode,
    )
    .unwrap();

    assert_eq!(placement.plan.predicted_duration_ns_per_activation, 1);
    assert_eq!(placement.plan.steps.len(), 1);
    assert!(matches!(
        &placement.plan.steps[0],
        VulkanHybridScheduledStep::Region {
            component_start: 0,
            component_end: 3,
            execution_case,
            ..
        } if execution_case == &region_case
    ));
    let bindings = BTreeMap::from([
        ("gpu0".to_string(), "logical0".to_string()),
        ("gpu1".to_string(), "logical1".to_string()),
    ]);
    let lowered =
        lower_vulkan_runtime_hybrid_phase_placement(&model, &placement, &bindings).unwrap();
    assert_eq!(lowered.execution_cases_by_component.len(), 3);
    assert_eq!(lowered.boundary_executions.len(), 2);
    assert_eq!(
        lowered
            .runtime_model
            .placement
            .device_for_component(&placement.component_ids[1]),
        "logical1",
    );
}

#[test]
fn runtime_hybrid_prefill_discovers_and_lowers_region_only_evidence() {
    let mut model = fixture_model_runtime_model_with_three_layer_series("gpu0");
    for execution in &mut model.component_executions {
        let mut prefill_terminal = execution.kernels.last().unwrap().clone();
        prefill_terminal.execution_index += 1;
        prefill_terminal.node_id = format!("{}_prefill", prefill_terminal.node_id);
        prefill_terminal.execution_domain = VulkanResidentComponentKernelExecutionDomain::Prefill;
        execution.kernels.push(prefill_terminal);
    }
    let phase = VulkanTargetedComponentExecutionPhase::Prefill {
        activation_batch_width: 4,
    };
    let mut catalog = VulkanPlacementCalibrationCatalog::default();
    record_hybrid_test_serialized_region(&model, &mut catalog, phase, 4);
    let capacity = VulkanPlacementCapacityEnvelope {
        available_bytes_by_device: BTreeMap::from([
            (hybrid_test_device("gpu0"), 100),
            (hybrid_test_device("gpu1"), 100),
        ]),
        host_available_bytes: 100,
    };

    assert_eq!(
        vulkan_runtime_hybrid_calibrated_prefill_widths(&model, &catalog).unwrap(),
        [4]
    );
    let placement =
        plan_vulkan_runtime_hybrid_ordered_graph(&model, &catalog, &capacity, phase).unwrap();
    let bindings = BTreeMap::from([
        ("gpu0".to_string(), "logical0".to_string()),
        ("gpu1".to_string(), "logical1".to_string()),
    ]);
    let lowered =
        lower_vulkan_runtime_hybrid_phase_placement(&model, &placement, &bindings).unwrap();
    assert_eq!(lowered.activation_batch_width, 4);
    assert_eq!(lowered.execution_cases_by_component.len(), 3);
    assert_eq!(lowered.boundary_executions.len(), 2);
}

#[test]
fn runtime_hybrid_planner_rejects_missing_or_ambiguous_behavior_evidence() {
    let model = fixture_model_runtime_model_with_three_layer_series("gpu0");
    let capacity = VulkanPlacementCapacityEnvelope {
        available_bytes_by_device: BTreeMap::from([(hybrid_test_device("gpu0"), 100)]),
        host_available_bytes: 100,
    };
    assert!(
        !vulkan_runtime_hybrid_phase_is_calibrated(
            &model,
            &VulkanPlacementCalibrationCatalog::default(),
            VulkanTargetedComponentExecutionPhase::Decode,
        )
        .unwrap()
    );
    let missing = plan_vulkan_runtime_hybrid_ordered_graph(
        &model,
        &VulkanPlacementCalibrationCatalog::default(),
        &capacity,
        VulkanTargetedComponentExecutionPhase::Decode,
    )
    .unwrap_err();
    assert!(missing.0.contains("no exact measured runtime hybrid placement"));

    let mut catalog = hybrid_test_catalog(&model);
    let target = vulkan_runtime_placement_calibration_target_for_component(
        &model,
        &model.component_executions[0].component_id,
        VulkanTargetedComponentExecutionPhase::Decode,
    )
    .unwrap();
    let mut second_behavior = hybrid_test_behavior(&target.signature_id);
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
    let catalog = hybrid_test_distributed_catalog(&model);
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

    let mut stale_runtime = placement.clone();
    let VulkanHybridScheduledStep::Region { execution_case, .. } =
        &mut stale_runtime.plan.steps[0]
    else {
        panic!("first hybrid step must be a component region");
    };
    execution_case.behavior.runtime_implementation_fingerprint =
        "stale-runtime".to_string();
    assert!(
        lower_vulkan_runtime_hybrid_phase_placement(&model, &stale_runtime, &bindings)
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
fn runtime_hybrid_physical_resolution_carries_measured_tp_into_normal_execution() {
    let model = fixture_model_runtime_model_with_three_layer_series("gpu0");
    let catalog = hybrid_test_distributed_catalog(&model);
    let capacity = VulkanPlacementCapacityEnvelope {
        available_bytes_by_device: BTreeMap::from([
            (hybrid_test_device("gpu0"), 100),
            (hybrid_test_device("gpu1"), 100),
        ]),
        host_available_bytes: 100,
    };
    let bindings = BTreeMap::from([
        ("gpu0".to_string(), "logical-owner".to_string()),
        ("gpu1".to_string(), "logical-helper".to_string()),
    ]);

    let resolution = resolve_vulkan_runtime_hybrid_physical_execution(
        &model,
        &catalog,
        &capacity,
        128,
        &bindings,
    )
    .unwrap()
    .expect("complete exact TP evidence must resolve");

    assert_eq!(resolution.decode_predicted_duration_ns_per_activation, 24);
    assert_eq!(resolution.prefill_activation_batch_width, None);
    assert_eq!(resolution.prefill_predicted_duration_ns_per_activation, None);
    assert_eq!(
        resolution
            .physical_execution_plan
            .decode_execution_cases_by_component
            .len(),
        3,
    );
    assert!(
        resolution
            .physical_execution_plan
            .decode_execution_cases_by_component
            .values()
            .all(|case| case.strategy == VulkanPlacementExecutionStrategy::TensorParallel)
    );
    assert!(
        resolution
            .physical_execution_plan
            .component_device_pools
            .decode
            .values()
            .all(|pool| pool == &["logical-owner", "logical-helper"])
    );
    assert!(
        resolution
            .runtime_model
            .circuit_graph
            .components
            .iter()
            .filter(|component| component.runtime_role.is_signal_processor())
            .all(|component| resolution
                .runtime_model
                .placement
                .device_for_component(&component.component_id)
                == "logical-owner")
    );

    assert!(
        resolve_vulkan_runtime_hybrid_physical_execution(
            &model,
            &VulkanPlacementCalibrationCatalog::default(),
            &capacity,
            128,
            &bindings,
        )
        .unwrap()
        .is_none()
    );
    let missing_helper = BTreeMap::from([(
        "gpu0".to_string(),
        "logical-owner".to_string(),
    )]);
    assert!(
        resolve_vulkan_runtime_hybrid_physical_execution(
            &model,
            &catalog,
            &capacity,
            128,
            &missing_helper,
        )
        .unwrap_err()
        .0
        .contains("unbound physical device")
    );
}

#[test]
fn runtime_hybrid_lowering_preserves_exact_boundary_transport_for_mount() {
    let model = fixture_model_runtime_model_with_three_layer_series("gpu0");
    let catalog = hybrid_test_catalog(&model);
    let capacity = VulkanPlacementCapacityEnvelope {
        available_bytes_by_device: BTreeMap::from([
            (hybrid_test_device("gpu0"), 100),
            (hybrid_test_device("gpu1"), 100),
        ]),
        host_available_bytes: 100,
    };
    let mut placement = plan_vulkan_runtime_hybrid_ordered_graph(
        &model,
        &catalog,
        &capacity,
        VulkanTargetedComponentExecutionPhase::Decode,
    )
    .unwrap();
    for step in placement.plan.steps.iter_mut().skip(1) {
        let VulkanHybridScheduledStep::Region { execution_case, .. } = step else {
            panic!("fixture plan contains only component regions");
        };
        execution_case.devices = vec![hybrid_test_device("gpu1")];
        execution_case.input_physical_device_id = "gpu1".to_string();
        execution_case.output_physical_device_id = "gpu1".to_string();
        execution_case.owner_physical_device_id = "gpu1".to_string();
    }
    let frame_byte_count = vulkan_runtime_placement_boundaries(&model).unwrap()[0].transfers[0]
        .byte_count;
    placement.plan.steps.insert(
        1,
        VulkanHybridScheduledStep::Boundary {
            boundary_index: 0,
            execution_case: hybrid_test_boundary_case(
                "gpu0",
                "gpu1",
                frame_byte_count,
                "external_device_local",
            ),
        },
    );
    let bindings = BTreeMap::from([
        ("gpu0".to_string(), "logical0".to_string()),
        ("gpu1".to_string(), "logical1".to_string()),
    ]);

    let lowered =
        lower_vulkan_runtime_hybrid_phase_placement(&model, &placement, &bindings).unwrap();
    let boundary = &lowered.boundary_executions[&0];
    assert_eq!(boundary.source_device_id, "logical0");
    assert_eq!(boundary.destination_device_id, "logical1");
    assert_eq!(boundary.frame_byte_count, frame_byte_count);
    assert_eq!(
        boundary.execution_case.transports[0].route,
        "external_device_local"
    );

    let physical_plan = VulkanRuntimePhysicalExecutionPlan {
        component_device_pools: VulkanDistributedPhaseComponentDevicePools::uniform(
            &lowered
                .runtime_model
                .placement
                .component_shard_devices,
        ),
        decode_execution_cases_by_component: lowered.execution_cases_by_component,
        decode_boundary_executions: lowered.boundary_executions,
        ..VulkanRuntimePhysicalExecutionPlan::default()
    };
    physical_plan.validate(&lowered.runtime_model).unwrap();
    let mut missing_boundary = physical_plan;
    missing_boundary.decode_boundary_executions.clear();
    assert!(
        missing_boundary
            .validate(&lowered.runtime_model)
            .unwrap_err()
            .0
            .contains("must cover every and only cross-device component boundary")
    );
}

#[test]
fn runtime_hybrid_phase_set_keeps_decode_owners_while_optimizing_prefill() {
    let mut model = fixture_model_runtime_model_with_three_layer_series("gpu0");
    for execution in &mut model.component_executions {
        let mut prefill_terminal = execution.kernels.last().unwrap().clone();
        prefill_terminal.execution_index += 1;
        prefill_terminal.node_id = format!("{}_prefill", prefill_terminal.node_id);
        prefill_terminal.execution_domain = VulkanResidentComponentKernelExecutionDomain::Prefill;
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

    let phase_set =
        plan_vulkan_runtime_hybrid_phase_set(&model, &catalog, &capacity, Some(4)).unwrap();
    assert!(phase_set.decode.plan.steps.iter().all(|step| matches!(
        step,
        VulkanHybridScheduledStep::Region { execution_case, .. }
            if execution_case.owner_physical_device_id == "gpu0"
    )));
    assert!(
        phase_set
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
            ))
    );

    let bindings = BTreeMap::from([
        ("gpu0".to_string(), "logical0".to_string()),
        ("gpu1".to_string(), "logical1".to_string()),
    ]);
    let (stable_model, physical_plan) =
        lower_vulkan_runtime_hybrid_phase_set(&model, &phase_set, &bindings).unwrap();
    assert!(
        stable_model
            .circuit_graph
            .components
            .iter()
            .filter(|component| component.runtime_role.is_signal_processor())
            .all(|component| stable_model
                .placement
                .device_for_component(&component.component_id)
                == "logical0")
    );
    assert_eq!(physical_plan.decode_execution_cases_by_component.len(), 3);
    assert_eq!(physical_plan.prefill_execution_cases_by_component.len(), 3);
    assert!(physical_plan.component_device_pools.decode.is_empty());
    assert!(physical_plan.component_device_pools.prefill.is_empty());
}

#[test]
fn runtime_hybrid_try_phase_set_preserves_decode_when_prefill_cannot_keep_its_owners() {
    let mut model = fixture_model_runtime_model_with_three_layer_series("gpu0");
    for execution in &mut model.component_executions {
        let mut prefill_terminal = execution.kernels.last().unwrap().clone();
        prefill_terminal.execution_index += 1;
        prefill_terminal.node_id = format!("{}_prefill", prefill_terminal.node_id);
        prefill_terminal.execution_domain = VulkanResidentComponentKernelExecutionDomain::Prefill;
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
    let mut prefill_signatures = model
        .circuit_graph
        .components
        .iter()
        .filter(|component| component.runtime_role.is_signal_processor())
        .map(|component| {
            vulkan_runtime_placement_calibration_target_for_component(
                &model,
                &component.component_id,
                VulkanTargetedComponentExecutionPhase::Prefill {
                    activation_batch_width: 4,
                },
            )
            .unwrap()
            .signature_id
        })
        .collect::<Vec<_>>();
    prefill_signatures.sort();
    prefill_signatures.dedup();
    for signature in prefill_signatures {
        let behavior = hybrid_test_behavior_for_phase(
            &signature,
            nerve_execution_contracts::ExecutionPhase::Prefill,
            4,
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
            .record_observation(hybrid_test_observation(behavior, "gpu1", 1))
            .unwrap();
    }
    let capacity = VulkanPlacementCapacityEnvelope {
        available_bytes_by_device: BTreeMap::from([
            (hybrid_test_device("gpu0"), 100),
            (hybrid_test_device("gpu1"), 100),
        ]),
        host_available_bytes: 100,
    };

    let phase_set = try_plan_vulkan_runtime_hybrid_phase_set(&model, &catalog, &capacity, Some(4))
        .unwrap()
        .unwrap();

    assert!(phase_set.decode.plan.steps.iter().all(|step| matches!(
        step,
        VulkanHybridScheduledStep::Region { execution_case, .. }
            if execution_case.owner_physical_device_id == "gpu0"
    )));
    assert!(phase_set.prefill.is_none());
    let strict_error =
        plan_vulkan_runtime_hybrid_phase_set(&model, &catalog, &capacity, Some(4)).unwrap_err();
    assert!(strict_error.0.contains("prefill placement preserves"));
}

#[test]
fn runtime_hybrid_prefill_keeps_distinct_batch_width_cohorts() {
    let mut model = fixture_model_runtime_model_with_three_layer_series("gpu0");
    for execution in &mut model.component_executions {
        let mut prefill_terminal = execution.kernels.last().unwrap().clone();
        prefill_terminal.execution_index += 1;
        prefill_terminal.node_id = format!("{}_prefill", prefill_terminal.node_id);
        prefill_terminal.execution_domain = VulkanResidentComponentKernelExecutionDomain::Prefill;
        execution.kernels.push(prefill_terminal);
    }
    let mut catalog = VulkanPlacementCalibrationCatalog::default();
    for width in [4, 8] {
        record_hybrid_phase_candidates(
            &model,
            &mut catalog,
            VulkanTargetedComponentExecutionPhase::Prefill {
                activation_batch_width: width,
            },
            10,
            20,
        );
    }
    let capacity = VulkanPlacementCapacityEnvelope {
        available_bytes_by_device: BTreeMap::from([
            (hybrid_test_device("gpu0"), 100),
            (hybrid_test_device("gpu1"), 100),
        ]),
        host_available_bytes: 100,
    };

    assert_eq!(
        vulkan_runtime_hybrid_calibrated_prefill_widths(&model, &catalog).unwrap(),
        [4, 8]
    );
    assert!(
        vulkan_runtime_hybrid_phase_is_calibrated(
            &model,
            &catalog,
            VulkanTargetedComponentExecutionPhase::Prefill {
                activation_batch_width: 4,
            },
        )
        .unwrap()
    );
    let placement = plan_vulkan_runtime_hybrid_ordered_graph(
        &model,
        &catalog,
        &capacity,
        VulkanTargetedComponentExecutionPhase::Prefill {
            activation_batch_width: 4,
        },
    )
    .unwrap();
    assert_eq!(placement.activation_batch_width, 4);
}
