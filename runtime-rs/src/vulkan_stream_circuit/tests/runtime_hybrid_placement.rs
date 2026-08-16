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

#[test]
fn speculative_batch_owner_capability_rejects_device_reentry() {
    let components = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let contiguous = BTreeMap::from([
        ("a".to_string(), "gpu0".to_string()),
        ("b".to_string(), "gpu0".to_string()),
        ("c".to_string(), "gpu1".to_string()),
    ]);
    let reentrant = BTreeMap::from([
        ("a".to_string(), "gpu0".to_string()),
        ("b".to_string(), "gpu1".to_string()),
        ("c".to_string(), "gpu0".to_string()),
    ]);

    assert!(runtime_hybrid_owner_segments_are_contiguous(&components, &contiguous).unwrap());
    assert!(!runtime_hybrid_owner_segments_are_contiguous(&components, &reentrant).unwrap());
    assert!(runtime_hybrid_owner_segments_are_contiguous(
        &components,
        &BTreeMap::from([
            ("a".to_string(), "gpu0".to_string()),
            ("b".to_string(), "gpu0".to_string()),
        ]),
    )
    .is_err());
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
    let mut targets = model
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
        })
        .collect::<Vec<_>>();
    targets.sort_by(|left, right| left.signature_id.cmp(&right.signature_id));
    targets.dedup_by(|left, right| left.signature_id == right.signature_id);
    for target in targets {
        let behavior = canonical_component_boundary_behavior(model, &target, phase).unwrap();
        assert_eq!(behavior.phase, execution_phase);
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
    hybrid_test_distributed_observation_with_strategy(
        behavior,
        duration_ns,
        VulkanPlacementExecutionStrategy::TensorParallel,
    )
}

fn hybrid_test_distributed_observation_with_strategy(
    behavior: VulkanPlacementBehaviorIdentity,
    duration_ns: u64,
    strategy: VulkanPlacementExecutionStrategy,
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
            strategy,
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
    let mut targets = model
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
        })
        .collect::<Vec<_>>();
    targets.sort_by(|left, right| left.signature_id.cmp(&right.signature_id));
    targets.dedup_by(|left, right| left.signature_id == right.signature_id);
    for target in targets {
        let behavior = canonical_component_boundary_behavior(
            model,
            &target,
            VulkanTargetedComponentExecutionPhase::Decode,
        )
        .unwrap();
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

fn hybrid_test_selected_representation(
    instance_ids: &[&str],
) -> crate::RuntimeSelectedImplementation {
    crate::RuntimeSelectedImplementation {
        implementation_id: "int4_representation".to_string(),
        candidate_id: "int4_candidate".to_string(),
        instance_ids: instance_ids
            .iter()
            .map(|instance_id| (*instance_id).to_string())
            .collect(),
        scope_ids: vec!["source_scope".to_string()],
        source_contract_digests: vec!["source_contract".to_string()],
        mount_adapter_id: crate::VULKAN_STREAM_CIRCUIT_OVERLAY_ADAPTER.to_string(),
        predicate: crate::RuntimeImplementationPredicate {
            schema: crate::RUNTIME_IMPLEMENTATION_PREDICATE_SCHEMA.to_string(),
            predicate_id: "fixture".to_string(),
            hardware: crate::RuntimeHardwarePredicate {
                measured_profile_ids: Vec::new(),
                capability_classes: Vec::new(),
                device_kinds: Vec::new(),
                apis: Vec::new(),
                required_processes: Vec::new(),
                required_features: Vec::new(),
            },
            execution: crate::RuntimeExecutionPredicate {
                phases: vec!["decode".to_string()],
                alternative_phases: vec!["decode".to_string()],
                source_retained_phases: Vec::new(),
                activation_batch: crate::RuntimeInclusiveRange {
                    minimum: 1,
                    maximum: 1,
                },
                context_activations: crate::RuntimeInclusiveRange {
                    minimum: 0,
                    maximum: 128,
                },
                state_activations: crate::RuntimeInclusiveRange {
                    minimum: 0,
                    maximum: 128,
                },
                speculative_draft_token_counts: vec![0],
                residency_policies: vec!["eager".to_string()],
            },
            placement: crate::RuntimePlacementPredicate {
                mode: "local".to_string(),
                minimum_device_count: 1,
                maximum_device_count: 1,
                required_interconnects: Vec::new(),
            },
        },
        representation: serde_json::json!({"kind": "int4"}),
        provenance: serde_json::json!({"fixture": true}),
        benchmark_id: "benchmark".to_string(),
        validation_id: "validation".to_string(),
        validation_status: "passed".to_string(),
        speedup_ppm: 500_000,
        estimated_saved_ns: 5,
        conversion_ns: 0,
        conversion_bytes: 0,
        boundary_count: 0,
        resource_load_count: 0,
        resource_reload_count: 0,
        resource_physical_read_bytes: 0,
        resource_resident_bytes_produced: 0,
        resource_uploaded_bytes: 0,
        resource_read_ns: 0,
        resource_derivation_ns: 0,
        resource_upload_ns: 0,
        resource_blocking_ns: 0,
        decision_reason: "fixture".to_string(),
    }
}

fn hybrid_test_representation_application(
    runtime_model: VulkanResidentRuntimeModel,
    instance_ids: &[&str],
) -> VulkanRuntimeHybridRepresentationApplication {
    let selected = hybrid_test_selected_representation(instance_ids);
    VulkanRuntimeHybridRepresentationApplication {
        runtime_model,
        semantic_contract_id: "source-contract".to_string(),
        selection: crate::RuntimeImplementationSelectionReport {
            package_id: "fixture".to_string(),
            execution: crate::RuntimeExecutionEnvelope {
                phases: vec!["decode".to_string()],
                activation_batch: crate::RuntimeInclusiveRange {
                    minimum: 1,
                    maximum: 1,
                },
                context_activations: crate::RuntimeInclusiveRange {
                    minimum: 0,
                    maximum: 128,
                },
                state_activations: crate::RuntimeInclusiveRange {
                    minimum: 0,
                    maximum: 128,
                },
                speculative_draft_tokens: 0,
                residency_policy: "eager".to_string(),
            },
            selected: vec![selected],
            exact_instance_ids: Vec::new(),
            rejected: Vec::new(),
            total_estimated_saved_ns: 5,
            total_conversion_ns: 0,
            total_conversion_bytes: 0,
            total_boundary_count: 0,
            total_resource_load_count: 0,
            total_resource_reload_count: 0,
            total_resource_physical_read_bytes: 0,
            total_resource_resident_bytes_produced: 0,
            total_resource_uploaded_bytes: 0,
            total_resource_read_ns: 0,
            total_resource_derivation_ns: 0,
            total_resource_upload_ns: 0,
            total_resource_blocking_ns: 0,
        },
    }
}

fn hybrid_test_predicate_for_profile(
    profile: &crate::HardwareProcessProfile,
) -> crate::RuntimeImplementationPredicate {
    crate::RuntimeImplementationPredicate {
        schema: crate::RUNTIME_IMPLEMENTATION_PREDICATE_SCHEMA.to_string(),
        predicate_id: "hybrid_fixture".to_string(),
        hardware: crate::RuntimeHardwarePredicate {
            measured_profile_ids: vec![profile.profile_id.clone()],
            capability_classes: vec![profile.capability_class.clone()],
            device_kinds: vec![profile.hardware_identity.device_kind.as_str().to_string()],
            apis: vec![profile.provenance.api.clone()],
            required_processes: Vec::new(),
            required_features: Vec::new(),
        },
        execution: crate::RuntimeExecutionPredicate {
            phases: vec!["decode".to_string()],
            alternative_phases: vec!["decode".to_string()],
            source_retained_phases: Vec::new(),
            activation_batch: crate::RuntimeInclusiveRange {
                minimum: 1,
                maximum: 1,
            },
            context_activations: crate::RuntimeInclusiveRange {
                minimum: 0,
                maximum: 128,
            },
            state_activations: crate::RuntimeInclusiveRange {
                minimum: 0,
                maximum: 128,
            },
            speculative_draft_token_counts: vec![0],
            residency_policies: vec!["eager".to_string()],
        },
        placement: crate::RuntimePlacementPredicate {
            mode: "local".to_string(),
            minimum_device_count: 1,
            maximum_device_count: 1,
            required_interconnects: Vec::new(),
        },
    }
}

fn hybrid_test_implementation_catalog(
    package_root: &Path,
    runtime_model: &VulkanResidentRuntimeModel,
    profile: &crate::HardwareProcessProfile,
) -> (crate::RuntimeImplementationCatalog, String) {
    let source_component = runtime_model
        .package
        .circuit_graph
        .components
        .iter()
        .find(|component| component.component_id == "layer_00")
        .unwrap()
        .clone();
    let source_execution = runtime_model
        .component_executions
        .iter()
        .find(|execution| execution.component_id == "layer_00")
        .unwrap()
        .clone();
    let candidate_root = package_root.join("optimization/hybrid-fixture/candidate");
    let candidate_shader_ref = "kernels/alternative.spv";
    let candidate_shader = candidate_root.join(candidate_shader_ref);
    std::fs::create_dir_all(candidate_shader.parent().unwrap()).unwrap();
    let source_shader = Path::new(&source_execution.kernels[0].shader_path);
    let source_shader = if source_shader.is_absolute() {
        source_shader.to_path_buf()
    } else {
        package_root.join(source_shader)
    };
    std::fs::copy(source_shader, &candidate_shader).unwrap();
    let mut component = source_component;
    component.implementation = "hybrid_alternative".to_string();
    component.circuit.implementation = "hybrid_alternative".to_string();
    let mut execution = source_execution;
    execution.implementation = "hybrid_alternative".to_string();
    execution.kernels[0].shader_path = candidate_shader_ref.to_string();
    let overlay_ref = "overlays/layer_00.json";
    let overlay_path = candidate_root.join(overlay_ref);
    std::fs::create_dir_all(overlay_path.parent().unwrap()).unwrap();
    std::fs::write(
        &overlay_path,
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema": crate::VULKAN_COMPONENT_OVERLAY_SCHEMA,
            "source_component_id": "layer_00",
            "component": component,
            "execution": execution,
            "resident_derivations": [],
        }))
        .unwrap(),
    )
    .unwrap();
    let predicate = hybrid_test_predicate_for_profile(profile);
    let implementation = crate::RuntimeImplementation {
        implementation_id: "hybrid_implementation".to_string(),
        candidate_id: "hybrid_candidate".to_string(),
        scope_ids: vec!["layer_scope".to_string()],
        source_contract_digests: vec!["layer_source_contract".to_string()],
        representation: serde_json::json!({"kind": "hybrid_fixture"}),
        behavioral_contract: serde_json::json!({"mode": "exact"}),
        runtime_predicate: predicate,
        artifact_bundle: crate::RuntimeImplementationArtifactBundle {
            root_ref: "optimization/hybrid-fixture".to_string(),
            candidate_integrity_ref: "unused".to_string(),
            mount_plan_ref: "unused".to_string(),
            candidate_integrity_digest: "fixture".to_string(),
            artifact_count: 1,
        },
        evidence: crate::RuntimeImplementationEvidence {
            promotion_decision_ref: "promotion".to_string(),
            candidate_contract_ref: "candidate".to_string(),
            construction_record_ref: "construction".to_string(),
            prebenchmark_record_ref: "prebenchmark".to_string(),
            benchmark_record_ref: "benchmark".to_string(),
            validation_record_ref: "validation".to_string(),
            analysis_run_refs: Vec::new(),
            hardware_profile_refs: Vec::new(),
        },
        provenance: serde_json::json!({"fixture": true}),
        comparison: crate::RuntimeImplementationComparison {
            exact_implementation_id: "exact".to_string(),
            exact_contract_digest: "exact".to_string(),
            benchmark_id: "benchmark".to_string(),
            benchmark_decision: "materially_faster".to_string(),
            workloads: Vec::new(),
            validation_id: "validation".to_string(),
            validation_status: "passed".to_string(),
            behavioral_contract: serde_json::json!({"mode": "exact"}),
        },
        decision_reason: "measured faster fixture".to_string(),
    };
    let loaded = crate::LoadedRuntimeImplementation {
        source_component_ids: vec!["layer_00".to_string()],
        workload_metrics: vec![crate::RuntimeImplementationWorkloadMetrics {
            workload_id: "decode".to_string(),
            phase: "decode".to_string(),
            activation_batch_width: 1,
            context_activations: 128,
            state_activations: 128,
            reference_latency_ns: 10,
            candidate_latency_ns: 4,
            conversion_ns: 0,
            conversion_bytes: 0,
            boundary_count: 0,
            resource_load_count: 0,
            resource_reload_count: 0,
            resource_physical_read_bytes: 0,
            resource_resident_bytes_produced: 0,
            resource_uploaded_bytes: 0,
            resource_read_ns: 0,
            resource_derivation_ns: 0,
            resource_upload_ns: 0,
            resource_blocking_ns: 0,
            speedup_ppm: 600_000,
        }],
        candidate_root,
        mount_plan: crate::RuntimeMountPlan {
            schema: crate::RUNTIME_MOUNT_PLAN_SCHEMA.to_string(),
            candidate_id: implementation.candidate_id.clone(),
            adapter_id: crate::VULKAN_STREAM_CIRCUIT_OVERLAY_ADAPTER.to_string(),
            regions: vec![crate::RuntimeMountRegion {
                replacements: vec![crate::RuntimeReplacement::Component {
                    source_component_id: "layer_00".to_string(),
                    overlay_ref: overlay_ref.to_string(),
                }],
            }],
            tensor_index_refs: Vec::new(),
        },
        implementation,
    };
    (
        crate::RuntimeImplementationCatalog {
            package_id: runtime_model.package.package_id.clone(),
            package_root: package_root.to_path_buf(),
            stage_status: "optimized".to_string(),
            exact_baseline: crate::RuntimeExactImplementation {
                artifact_ref: "lowered/execution_graph.circuits.json".to_string(),
                contract_digest: "exact".to_string(),
                mutable: false,
            },
            scopes: BTreeMap::new(),
            implementations: vec![loaded],
        },
        "hybrid_alternative".to_string(),
    )
}

fn hybrid_test_distributed_catalog(
    model: &VulkanResidentRuntimeModel,
) -> VulkanPlacementCalibrationCatalog {
    hybrid_test_distributed_catalog_with_strategy(
        model,
        VulkanPlacementExecutionStrategy::TensorParallel,
    )
}

fn hybrid_test_distributed_catalog_with_strategy(
    model: &VulkanResidentRuntimeModel,
    strategy: VulkanPlacementExecutionStrategy,
) -> VulkanPlacementCalibrationCatalog {
    let mut catalog = VulkanPlacementCalibrationCatalog::default();
    let mut targets = model
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
        })
        .collect::<Vec<_>>();
    targets.sort_by(|left, right| left.signature_id.cmp(&right.signature_id));
    targets.dedup_by(|left, right| left.signature_id == right.signature_id);
    for target in targets {
        let behavior = canonical_component_boundary_behavior(
            model,
            &target,
            VulkanTargetedComponentExecutionPhase::Decode,
        )
        .unwrap();
        catalog
            .record_reference(VulkanPlacementCanonicalReference {
                behavior: behavior.clone(),
                output_digest: "output".to_string(),
                output_artifact: None,
                state_digest: "state".to_string(),
            })
            .unwrap();
        catalog
            .record_observation(hybrid_test_distributed_observation_with_strategy(
                behavior,
                8,
                strategy,
            ))
            .unwrap();
    }
    catalog
}

#[test]
fn runtime_hybrid_planner_accepts_measured_expert_parallel_families() {
    let model = fixture_model_runtime_model_with_three_layer_series("gpu0");
    let capacity = VulkanPlacementCapacityEnvelope {
        available_bytes_by_device: BTreeMap::from([
            (hybrid_test_device("gpu0"), 100),
            (hybrid_test_device("gpu1"), 100),
        ]),
        host_available_bytes: 100,
    };

    for strategy in [
        VulkanPlacementExecutionStrategy::WholeExpertParallel,
        VulkanPlacementExecutionStrategy::IntraExpertTensorParallel,
    ] {
        let catalog = hybrid_test_distributed_catalog_with_strategy(&model, strategy);
        let placement = plan_vulkan_runtime_hybrid_ordered_graph(
            &model,
            &catalog,
            &capacity,
            VulkanTargetedComponentExecutionPhase::Decode,
        )
        .unwrap();

        assert_eq!(placement.plan.predicted_duration_ns_per_activation, 24);
        assert!(placement.plan.steps.iter().all(|step| matches!(
            step,
            VulkanHybridScheduledStep::Region { execution_case, .. }
                if execution_case.strategy == strategy
        )));
    }
}

#[test]
fn runtime_hybrid_planner_maps_compiler_signatures_to_every_component_instance() {
    let model = fixture_model_runtime_model_with_three_layer_series("gpu0");
    let catalog = hybrid_test_catalog(&model);
    assert_eq!(
        catalog.observation_count(),
        2,
        "one measurement per physical target must cover every equivalent component instance",
    );
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
    let model = fixture_model_runtime_model_with_three_layer_series("gpu0");
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

    let overridden_component = resolution
        .runtime_model
        .circuit_graph
        .components
        .iter()
        .find(|component| component.runtime_role.is_signal_processor())
        .unwrap()
        .component_id
        .clone();
    let mixed_plan = resolution
        .physical_execution_plan
        .clone()
        .with_explicit_distributed_overrides(
            &resolution.runtime_model,
            &BTreeMap::from([(
                overridden_component.clone(),
                vec!["logical-owner".to_string(), "logical-helper".to_string()],
            )]),
            &BTreeMap::from([(
                overridden_component.clone(),
                nerve_execution_contracts::ExecutionStrategy::TensorParallel,
            )]),
        )
        .unwrap();
    assert_eq!(mixed_plan.decode_execution_cases_by_component.len(), 2);
    assert!(!mixed_plan
        .decode_execution_cases_by_component
        .contains_key(&overridden_component));
    assert!(mixed_plan
        .decode_contract_ids_by_component
        .contains_key(&overridden_component));
    mixed_plan.validate(&resolution.runtime_model).unwrap();

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
fn terminal_hybrid_mount_reserves_exact_physical_stream_overhead() {
    let model = fixture_model_runtime_model();
    let physical = VulkanRuntimePhysicalExecutionPlan::uniform(&model);
    let [logical_device_id] = physical.device_ids(&model).try_into().unwrap();
    let package_root = tiny_model_dir();
    let catalog = VulkanPlacementCalibrationCatalog::default();
    let mut device = physical_mount_test_device(&logical_device_id);
    device.safe_capacity_bytes = 1usize << 40;
    let baseline = plan_vulkan_runtime_physical_mount(
        &package_root,
        &model,
        &physical,
        Some(&catalog),
        64,
        0,
        ResourceResidencyPolicy::DemandPaged,
        std::slice::from_ref(&device),
        usize::MAX,
    )
    .unwrap()
    .unwrap();
    let cache_quota = baseline
        .selected_resource_cache_quota_bytes_by_logical_device
        .values()
        .copied()
        .sum::<usize>();
    let physical_device_id = device.identity.physical_device_id.clone();
    let overhead = 4096usize;
    let resolver = |plan: &VulkanRuntimePhysicalExecutionResidencyPlan| {
        let logical_stream_bytes = plan
            .device_plans
            .iter()
            .map(|device| device.stream_device_local_bytes)
            .sum::<usize>();
        Ok(BTreeMap::from([(
            physical_device_id.clone(),
            logical_stream_bytes + overhead,
        )]))
    };
    let planning = VulkanRuntimeHybridMountPlanningContext {
        devices: std::slice::from_ref(&device),
        physical_stream_requirement_resolver: Some(&resolver),
        speculative_draft_tokens: 0,
        residency_policy: ResourceResidencyPolicy::DemandPaged,
        host_safe_capacity_bytes: usize::MAX,
    };

    let corrected = plan_vulkan_runtime_physical_mount_with_exact_stream_requirements(
        &package_root,
        &model,
        &physical,
        &catalog,
        64,
        &planning,
        std::slice::from_ref(&device),
    )
    .unwrap()
    .unwrap();
    let corrected_cache_quota = corrected
        .selected_resource_cache_quota_bytes_by_logical_device
        .values()
        .copied()
        .sum::<usize>();
    assert_eq!(corrected_cache_quota + overhead, cache_quota);
}

#[test]
fn runtime_hybrid_representation_routes_honor_stable_owner_constraints() {
    let model = fixture_model_runtime_model_with_three_layer_series("gpu0");
    let catalog = hybrid_test_distributed_catalog(&model);
    let capacity = VulkanPlacementCapacityEnvelope {
        available_bytes_by_device: BTreeMap::from([
            (hybrid_test_device("gpu0"), 100),
            (hybrid_test_device("gpu1"), 100),
        ]),
        host_available_bytes: 100,
    };
    let required_owners = model
        .circuit_graph
        .components
        .iter()
        .filter(|component| component.runtime_role.is_signal_processor())
        .map(|component| (component.component_id.clone(), "gpu0".to_string()))
        .collect::<BTreeMap<_, _>>();

    let constrained = visit_runtime_hybrid_representation_placements_by_duration(
        &model,
        &[],
        &BTreeSet::new(),
        &catalog,
        &capacity,
        VulkanTargetedComponentExecutionPhase::Decode,
        Some(&required_owners),
        None,
        |placement| Ok(Some(placement)),
    )
    .unwrap()
    .expect("the required-owner route is present in the measured catalog");
    assert!(
        runtime_hybrid_physical_owners(&constrained.ordered_placement)
            .unwrap()
            .values()
            .all(|owner| owner == "gpu0")
    );

    let impossible_owners = required_owners
        .keys()
        .map(|component_id| (component_id.clone(), "gpu1".to_string()))
        .collect::<BTreeMap<_, _>>();
    assert!(
        visit_runtime_hybrid_representation_placements_by_duration(
            &model,
            &[],
            &BTreeSet::new(),
            &catalog,
            &capacity,
            VulkanTargetedComponentExecutionPhase::Decode,
            Some(&impossible_owners),
            None,
            |placement| Ok(Some(placement)),
        )
        .unwrap()
        .is_none()
    );
}

#[test]
fn explicit_tp_overlay_retains_exact_routes_for_serialized_outer_boundaries() {
    let baseline = fixture_model_runtime_model_with_three_layer_series("gpu0");
    let component_ids = baseline
        .circuit_graph
        .components
        .iter()
        .filter(|component| component.runtime_role.is_signal_processor())
        .map(|component| component.component_id.clone())
        .collect::<Vec<_>>();
    let model = vulkan_runtime_model_with_component_placement(
        &baseline,
        "gpu0",
        &BTreeMap::from([
            (component_ids[0].clone(), "gpu0".to_string()),
            (component_ids[1].clone(), "gpu0".to_string()),
            (component_ids[2].clone(), "gpu1".to_string()),
        ]),
    )
    .unwrap();
    let graph_boundaries = vulkan_runtime_placement_boundaries(&model).unwrap();
    let frame_byte_count = graph_boundaries[1].transfers[0].byte_count;
    let digest = hybrid_test_digest('a');
    let report = VulkanRuntimePlacementTransferCalibrationReport {
        source_device_id: "gpu0".to_string(),
        source_api_version: 1,
        source_driver_version: 2,
        target_device_id: "gpu1".to_string(),
        target_api_version: 1,
        target_driver_version: 2,
        phase: nerve_execution_contracts::ExecutionPhase::Decode,
        activation_batch_width: 1,
        frame_byte_count,
        byte_count: frame_byte_count,
        route: VulkanPlacedEdgeTransferRoute::DeviceLocalStaging,
        warmup_ns: 10,
        measured_ns: 8,
        fixture_digest: digest.clone(),
        output_digest: digest,
    };
    let mut catalog = VulkanPlacementCalibrationCatalog::default();
    record_vulkan_runtime_transfer_calibration_report(&mut catalog, &report).unwrap();
    let identities = BTreeMap::from([
        ("gpu0".to_string(), hybrid_test_device("gpu0")),
        ("gpu1".to_string(), hybrid_test_device("gpu1")),
    ]);

    let missing_evidence = VulkanRuntimePhysicalExecutionPlan::uniform(&model)
        .with_exact_cross_device_boundary_routes(&model, None, &identities)
        .unwrap_err();
    assert!(
        missing_evidence
            .0
            .contains("requires exact directed-transfer calibration evidence")
    );

    let tp_component_id = component_ids[0].clone();
    let plan = VulkanRuntimePhysicalExecutionPlan::uniform(&model)
        .with_explicit_distributed_overrides(
            &model,
            &BTreeMap::from([(
                tp_component_id.clone(),
                vec!["gpu0".to_string(), "gpu1".to_string()],
            )]),
            &BTreeMap::from([(
                tp_component_id.clone(),
                nerve_execution_contracts::ExecutionStrategy::TensorParallel,
            )]),
        )
        .unwrap()
        .with_exact_cross_device_boundary_routes(&model, Some(&catalog), &identities)
        .unwrap();

    assert_eq!(plan.decode_boundary_executions.len(), 1);
    assert_eq!(
        plan.decode_boundary_executions[&1].source_device_id,
        "gpu0"
    );
    assert_eq!(
        plan.decode_boundary_executions[&1].destination_device_id,
        "gpu1"
    );
    assert_eq!(
        plan.decode_boundary_executions[&1].edge_index, 1,
        "physical routes use the mounted signal-only graph's edge space",
    );
    let full_graph_edge_index = model
        .circuit_graph
        .edges
        .iter()
        .position(|edge| {
            edge.source.component_id == component_ids[1]
                && edge.destination.component_id == component_ids[2]
        })
        .unwrap();
    assert_ne!(
        full_graph_edge_index, plan.decode_boundary_executions[&1].edge_index,
        "the fixture must retain a non-signal edge that exposes positional aliasing",
    );
    assert_eq!(
        plan.component_device_pools.decode[&tp_component_id],
        ["gpu0", "gpu1"]
    );
    plan.validate(&model).unwrap();

    // The fixture's tiny legacy TP contract intentionally cannot execute its
    // residual reduction geometry. Exercise the real outer-route mount with
    // the otherwise identical serialized plan; the TP composition above
    // proves that adding the island preserves that route record.
    let serialized_plan = VulkanRuntimePhysicalExecutionPlan::uniform(&model)
        .with_exact_cross_device_boundary_routes(&model, Some(&catalog), &identities)
        .unwrap();

    let devices = [
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
    assert!(
        plan_vulkan_runtime_physical_mount(
            tiny_model_dir(),
            &model,
            &serialized_plan,
            Some(&catalog),
            64,
            0,
            ResourceResidencyPolicy::Eager,
            &devices,
            usize::MAX,
        )
        .unwrap()
        .is_some(),
        "the workload-free planner must bind the exact outer route around a TP island",
    );
}

#[test]
fn runtime_hybrid_exact_candidate_resources_prune_before_terminal_mount() {
    reset_resident_package_planning_basis_preparation_count();
    let model = fixture_model_runtime_model();
    let catalog = hybrid_test_catalog(&model);
    let package_root = tiny_model_dir();
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
        VulkanTargetedComponentExecutionPhase::Decode,
        None,
        VulkanRuntimeHybridComponentStrategyFilter::AnyMeasured,
        Some(&planner),
    )
    .unwrap();
    assert_eq!(
        resident_package_planning_basis_preparation_count(),
        1,
        "candidate discovery must resolve the placement-invariant model plan once",
    );
    assert_eq!(
        candidates.authoritative_resource_classes,
        BTreeSet::from([
            VulkanHybridResourceClass::Permanent,
            VulkanHybridResourceClass::MutableState,
            VulkanHybridResourceClass::CacheQuota,
            VulkanHybridResourceClass::AtomicLoadWave,
        ])
    );

    let select_first = |capacity: &VulkanPlacementCapacityEnvelope| {
        visit_vulkan_hybrid_ordered_graph_routes_by_duration(
            &catalog,
            candidates.component_ids.len(),
            &candidates.region_candidates,
            &candidates.boundary_candidates,
            &candidates.resource_catalog,
            &candidates.authoritative_resource_classes,
            capacity,
            |route| Ok(Some(route.clone())),
        )
        .unwrap()
    };
    let accepted = select_first(&VulkanPlacementCapacityEnvelope {
        available_bytes_by_device: BTreeMap::from([(
            hybrid_test_device("gpu0"),
            usize::MAX,
        )]),
        host_available_bytes: usize::MAX,
    })
    .expect("unbounded exact parameter capacity must retain the route");
    let exact_parameter_bytes = accepted.authoritative_resource_reservations.device_bytes
        [&hybrid_test_device("gpu0")]
        .permanent_bytes;
    let exact_required_bytes = accepted.authoritative_resource_reservations.device_bytes
        [&hybrid_test_device("gpu0")]
        .required_capacity_bytes()
        .unwrap();
    assert!(exact_parameter_bytes > 100);
    assert!(
        accepted.authoritative_resource_reservations.device_bytes
            [&hybrid_test_device("gpu0")]
            .mutable_state_bytes
            > 0
    );
    let tensor_index = model.load_runtime_tensor_index(&package_root).unwrap();
    let full_residency = plan_vulkan_runtime_residency(
        &package_root,
        &model,
        &tensor_index,
        64,
        0,
        ResourceResidencyPolicy::Eager,
    )
    .unwrap();
    let [full_device] = full_residency.device_plans.as_slice() else {
        panic!("the one-device fixture must have one residency plan");
    };
    assert_eq!(
        exact_parameter_bytes,
        full_device.parameter_residency.current_resident_bytes,
        "candidate admission must include graph-owned endpoint parameters before terminal mount",
    );
    let expected_candidate_state = full_device
        .working_set
        .transient_state_bytes
        .checked_add(full_device.working_set.activation_headroom_bytes)
        .unwrap();
    assert_eq!(
        accepted.authoritative_resource_reservations.device_bytes
            [&hybrid_test_device("gpu0")]
            .mutable_state_bytes,
        expected_candidate_state,
        "candidate state must match the full workload-free residency plan",
    );
    assert!(
        select_first(&VulkanPlacementCapacityEnvelope {
            available_bytes_by_device: BTreeMap::from([(
                hybrid_test_device("gpu0"),
                exact_required_bytes - 1,
            )]),
            host_available_bytes: usize::MAX,
        })
        .is_none(),
        "sampled ten-byte calibration residency must not admit an oversized real component",
    );
    assert!(
        select_first(&VulkanPlacementCapacityEnvelope {
            available_bytes_by_device: BTreeMap::from([(
                hybrid_test_device("gpu0"),
                exact_required_bytes,
            )]),
            host_available_bytes: usize::MAX,
        })
        .is_some(),
        "sampled transient memory is not exact enough to prune before terminal mount",
    );
}

#[test]
fn runtime_hybrid_graph_parameters_follow_their_signal_endpoints() {
    let model = fixture_model_runtime_model_with_three_layer_series("gpu0");
    let signal_ids = model
        .circuit_graph
        .components
        .iter()
        .filter(|component| component.runtime_role.is_signal_processor())
        .map(|component| component.component_id.as_str())
        .collect::<Vec<_>>();
    let [first, middle, last] = signal_ids.as_slice() else {
        panic!("the fixture must contain three signal processors");
    };

    let first_anchors =
        exact_vulkan_runtime_hybrid_graph_parameter_anchor_ids(&model, first).unwrap();
    let middle_anchors =
        exact_vulkan_runtime_hybrid_graph_parameter_anchor_ids(&model, middle).unwrap();
    let last_anchors =
        exact_vulkan_runtime_hybrid_graph_parameter_anchor_ids(&model, last).unwrap();
    let roles = |ids: &BTreeSet<String>| {
        model
            .circuit_graph
            .components
            .iter()
            .filter(|component| ids.contains(&component.component_id))
            .map(|component| component.runtime_role)
            .collect::<Vec<_>>()
    };

    let first_roles = roles(&first_anchors);
    assert_eq!(first_roles.len(), 2);
    assert!(first_roles.contains(&CircuitRuntimeRole::InputTransducer));
    assert!(first_roles.contains(&CircuitRuntimeRole::SignalProcessor));

    let middle_roles = roles(&middle_anchors);
    assert_eq!(middle_roles, [CircuitRuntimeRole::SignalProcessor]);

    let last_roles = roles(&last_anchors);
    assert_eq!(last_roles.len(), 3);
    assert!(last_roles.contains(&CircuitRuntimeRole::SignalProcessor));
    assert!(last_roles.contains(&CircuitRuntimeRole::OutputTransducer));
    assert!(last_roles.contains(&CircuitRuntimeRole::Sampler));
}

#[test]
fn runtime_hybrid_exact_candidate_parameters_require_complete_physical_bindings() {
    let model = fixture_model_runtime_model();
    let catalog = hybrid_test_catalog(&model);
    let package_root = tiny_model_dir();
    let bindings = BTreeMap::from([("gpu0".to_string(), "gpu0".to_string())]);
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

    let error = match runtime_hybrid_candidate_graph(
        &model,
        &catalog,
        VulkanTargetedComponentExecutionPhase::Decode,
        None,
        VulkanRuntimeHybridComponentStrategyFilter::AnyMeasured,
        Some(&planner),
    ) {
        Ok(_) => panic!("incomplete physical bindings must be rejected"),
        Err(error) => error,
    };

    assert!(
        error
            .0
            .contains("planning device \"gpu1\" is not bound to logical device \"gpu1\"")
    );
}

#[test]
fn runtime_hybrid_exact_state_deduplicates_same_device_component_edges() {
    let model = fixture_model_runtime_model_with_three_layer_series("gpu0");
    let catalog = hybrid_test_catalog(&model);
    let package_root = tiny_model_dir();
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
        VulkanTargetedComponentExecutionPhase::Decode,
        None,
        VulkanRuntimeHybridComponentStrategyFilter::AnyMeasured,
        Some(&planner),
    )
    .unwrap();
    let route = visit_vulkan_hybrid_ordered_graph_routes_by_duration(
        &catalog,
        candidates.component_ids.len(),
        &candidates.region_candidates,
        &candidates.boundary_candidates,
        &candidates.resource_catalog,
        &candidates.authoritative_resource_classes,
        &VulkanPlacementCapacityEnvelope {
            available_bytes_by_device: BTreeMap::from([(
                hybrid_test_device("gpu0"),
                usize::MAX,
            )]),
            host_available_bytes: usize::MAX,
        },
        |route| Ok(Some(route.clone())),
    )
    .unwrap()
    .expect("the all-gpu0 exact route must be available");

    let tensor_index = model.load_runtime_tensor_index(&package_root).unwrap();
    let full_residency = plan_vulkan_runtime_residency(
        &package_root,
        &model,
        &tensor_index,
        64,
        0,
        ResourceResidencyPolicy::Eager,
    )
    .unwrap();
    let [full_device] = full_residency.device_plans.as_slice() else {
        panic!("the all-gpu0 fixture must have one residency plan");
    };
    assert_eq!(
        route.authoritative_resource_reservations.device_bytes
            [&hybrid_test_device("gpu0")]
            .mutable_state_bytes,
        full_device
            .working_set
            .transient_state_bytes
            .checked_add(full_device.working_set.activation_headroom_bytes)
            .unwrap(),
        "component-local edge views must canonicalize to the full graph allocation",
    );
}

#[test]
fn runtime_hybrid_exact_state_is_scoped_to_the_candidate_component() {
    let model = fixture_model_runtime_model_with_three_layer_series("gpu0");
    let package_root = tiny_model_dir();
    let tensor_index = model.load_runtime_tensor_index(&package_root).unwrap();
    let placed_model = vulkan_runtime_model_with_component_placement(
        &model,
        "gpu0",
        &BTreeMap::new(),
    )
    .unwrap();
    let graph = placed_model.executable_circuit_graph().unwrap();
    let (_, _, placed_plan) = plan_resident_package_placed_stream_circuit_with_tensor_index(
        "gpu0",
        &placed_model.placement,
        &graph,
        &package_root,
        &tensor_index,
        placed_model.package.activation_element_bytes,
    )
    .unwrap();
    let component_id = "layer_00";
    let full = plan_stream_circuit_residency(&placed_plan, 64, true, false, 0).unwrap();
    let component =
        plan_component_stream_circuit_residency(&placed_plan, component_id, 64, true, false, 0)
            .unwrap();
    assert!(component.state_bytes < full.state_bytes);
    assert!(component.activation_bytes < full.activation_bytes);
    assert!(component.edge_bytes < full.edge_bytes);
    assert!(component.boundary_bytes < full.boundary_bytes);

    let execution_case = hybrid_test_observation(
        hybrid_test_behavior("component-scoped-state"),
        "gpu0",
        1,
    )
    .execution_case;
    let requirements = exact_vulkan_runtime_hybrid_component_state_requirements(
        &package_root,
        &model,
        &placed_model,
        component_id,
        &execution_case,
        "gpu0",
        &placed_plan,
        64,
        0,
        &tensor_index,
    )
    .unwrap();
    let expected = [
        component.state_bytes,
        component.transaction_checkpoint_bytes,
        component.transaction_bytes,
        component.activation_bytes,
        component.boundary_bytes,
        component.edge_bytes,
        component.causal_verification_snapshot_bytes,
        VULKAN_STREAM_CONTROL_BYTE_CAPACITY,
    ]
    .into_iter()
    .sum::<usize>();
    assert_eq!(
        requirements
            .iter()
            .map(|requirement| requirement.byte_count)
            .sum::<usize>(),
        expected,
    );
    assert!(requirements.iter().all(|requirement| {
        !requirement.resource_identity.contains("layer_00_tail")
            && !requirement.resource_identity.contains("layer_00_remote")
    }));
}

#[test]
fn runtime_hybrid_exact_state_accounts_distributed_shared_and_private_activations() {
    let model = fixture_model_runtime_model();
    let identities = BTreeMap::from([
        ("gpu0".to_string(), hybrid_test_device("gpu0")),
        ("gpu1".to_string(), hybrid_test_device("gpu1")),
    ]);
    let activation_plan = |route| VulkanDistributedActivationBufferPlan {
        allocations: vec![VulkanDistributedActivationBufferAllocation {
            storage: VulkanDistributedActivationStorage::ActivationSlot,
            owner_device_id: "gpu0".to_string(),
            component_id: "component".to_string(),
            slot: 0,
            byte_capacity: 64,
            signal_ids: vec!["input".to_string()],
            device_ids: vec!["gpu0".to_string(), "gpu1".to_string()],
            input_use_count: 1,
            output_use_count: 0,
        }],
        reduction_allocations: vec![VulkanDistributedReductionBufferAllocation {
            owner_device_id: "gpu0".to_string(),
            dispatch_index: 1,
            component_id: "component".to_string(),
            node_id: "down".to_string(),
            plane_byte_capacity: 48,
            byte_capacity: 96,
            device_ids: vec!["gpu0".to_string(), "gpu1".to_string()],
        }],
        private_intermediate_allocations: vec![
            VulkanDistributedPrivateIntermediateBufferAllocation {
                producer_dispatch_index: 0,
                consumer_dispatch_index: 1,
                component_id: "component".to_string(),
                signal_id: "activated".to_string(),
                devices: vec![
                    VulkanDistributedPrivateIntermediateDeviceAllocation {
                        device_id: "gpu0".to_string(),
                        byte_capacity: 16,
                    },
                    VulkanDistributedPrivateIntermediateDeviceAllocation {
                        device_id: "gpu1".to_string(),
                        byte_capacity: 24,
                    },
                ],
            },
        ],
        allocation_count: 4,
        import_count: 6,
        reference_count: 6,
        total_shared_byte_capacity: 160,
        total_private_byte_capacity: 40,
        route,
    };
    let reservations = |route| {
        let requirements =
            exact_vulkan_runtime_hybrid_distributed_activation_requirements_from_plan(
                &model,
                &activation_plan(route),
                &identities,
            )
            .unwrap();
        let resources = canonical_vulkan_hybrid_shared_range_resources(&BTreeMap::from([(
            "candidate".to_string(),
            requirements,
        )]))
        .unwrap();
        VulkanHybridResourceReservations::default()
            .reserve(
                &resources["candidate"],
                &VulkanPlacementCapacityEnvelope {
                    available_bytes_by_device: BTreeMap::from([
                        (hybrid_test_device("gpu0"), usize::MAX),
                        (hybrid_test_device("gpu1"), usize::MAX),
                    ]),
                    host_available_bytes: usize::MAX,
                },
            )
            .unwrap()
            .unwrap()
    };

    let shared_host = reservations(VulkanSharedResidentBufferRoute::SharedHost);
    assert_eq!(shared_host.host_bytes.mutable_state_bytes, 160);
    assert_eq!(
        shared_host.device_bytes[&hybrid_test_device("gpu0")].mutable_state_bytes,
        16
    );
    assert_eq!(
        shared_host.device_bytes[&hybrid_test_device("gpu1")].mutable_state_bytes,
        24
    );

    let device_local = reservations(VulkanSharedResidentBufferRoute::ExternalDeviceLocal);
    assert_eq!(device_local.host_bytes.mutable_state_bytes, 0);
    assert_eq!(
        device_local.device_bytes[&hybrid_test_device("gpu0")].mutable_state_bytes,
        176
    );
    assert_eq!(
        device_local.device_bytes[&hybrid_test_device("gpu1")].mutable_state_bytes,
        24
    );
}

fn exact_candidate_resources_for_model(
    model: &VulkanResidentRuntimeModel,
    policy: ResourceResidencyPolicy,
) -> VulkanHybridCandidateResources {
    let catalog = hybrid_test_catalog(model);
    let package_root = tiny_model_dir();
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
        residency_policy: policy,
    };
    let candidates = runtime_hybrid_candidate_graph(
        model,
        &catalog,
        VulkanTargetedComponentExecutionPhase::Decode,
        None,
        VulkanRuntimeHybridComponentStrategyFilter::AnyMeasured,
        Some(&planner),
    )
    .unwrap();
    let candidate = candidates
        .region_candidates
        .iter()
        .find(|candidate| candidate.execution_case.owner_physical_device_id == "gpu0")
        .unwrap();
    candidates.resource_catalog.region_resources_by_candidate_id[&candidate.candidate_id].clone()
}

fn exact_candidate_reservations_for_model(
    model: &VulkanResidentRuntimeModel,
    policy: ResourceResidencyPolicy,
) -> VulkanHybridResourceReservations {
    let resources = exact_candidate_resources_for_model(model, policy);
    VulkanHybridResourceReservations::default()
        .reserve(
            &resources,
            &VulkanPlacementCapacityEnvelope {
                available_bytes_by_device: BTreeMap::from([
                    (hybrid_test_device("gpu0"), usize::MAX),
                    (hybrid_test_device("gpu1"), usize::MAX),
                ]),
                host_available_bytes: usize::MAX,
            },
        )
        .unwrap()
        .unwrap()
}

fn exact_dynamic_candidate_reservations(
    policy: ResourceResidencyPolicy,
) -> VulkanHybridResourceReservations {
    exact_candidate_reservations_for_model(
        &fixture_model_runtime_model_with_dynamic_partition(1_000, 64),
        policy,
    )
}

#[test]
fn runtime_hybrid_exact_dynamic_resources_reject_one_byte_below_capacity() {
    let model = fixture_model_runtime_model_with_dynamic_partition(1_000, 64);
    let resources = exact_candidate_resources_for_model(
        &model,
        ResourceResidencyPolicy::DemandRetained,
    );
    let unbounded = VulkanPlacementCapacityEnvelope {
        available_bytes_by_device: BTreeMap::from([
            (hybrid_test_device("gpu0"), usize::MAX),
            (hybrid_test_device("gpu1"), usize::MAX),
        ]),
        host_available_bytes: usize::MAX,
    };
    let exact = VulkanHybridResourceReservations::default()
        .reserve(&resources, &unbounded)
        .unwrap()
        .unwrap();
    let required_gpu0 = exact.device_bytes[&hybrid_test_device("gpu0")]
        .required_capacity_bytes()
        .unwrap();
    let exact_capacity = VulkanPlacementCapacityEnvelope {
        available_bytes_by_device: exact
            .device_bytes
            .iter()
            .map(|(device, bytes)| {
                (device.clone(), bytes.required_capacity_bytes().unwrap())
            })
            .collect(),
        host_available_bytes: exact.host_bytes.required_capacity_bytes().unwrap(),
    };
    assert!(
        VulkanHybridResourceReservations::default()
            .reserve(&resources, &exact_capacity)
            .unwrap()
            .is_some(),
        "the exact physical capacity must admit the candidate",
    );
    let mut insufficient = exact_capacity;
    insufficient
        .available_bytes_by_device
        .insert(hybrid_test_device("gpu0"), required_gpu0 - 1);
    assert!(
        VulkanHybridResourceReservations::default()
            .reserve(&resources, &insufficient)
            .unwrap()
            .is_none(),
        "one byte below the exact retained plus transient requirement must reject before terminal mount",
    );
}

#[test]
fn runtime_hybrid_exact_cache_distinguishes_paged_and_retained_residency() {
    let paged = exact_dynamic_candidate_reservations(ResourceResidencyPolicy::DemandPaged);
    let retained = exact_dynamic_candidate_reservations(ResourceResidencyPolicy::DemandRetained);
    let eager = exact_dynamic_candidate_reservations(ResourceResidencyPolicy::Eager);
    let paged = &paged.device_bytes[&hybrid_test_device("gpu0")];
    let retained = &retained.device_bytes[&hybrid_test_device("gpu0")];
    let eager = &eager.device_bytes[&hybrid_test_device("gpu0")];

    assert_eq!(paged.atomic_load_wave_bytes, 64);
    assert_eq!(paged.cache_quota_bytes, 64);
    assert_eq!(retained.atomic_load_wave_bytes, 64);
    assert_eq!(retained.cache_quota_bytes, 64_000);
    assert_eq!(eager.atomic_load_wave_bytes, 64);
    assert_eq!(eager.cache_quota_bytes, 64_000);
    assert!(paged.mutable_state_bytes > 0);
    assert_eq!(paged.mutable_state_bytes, retained.mutable_state_bytes);
    assert_eq!(retained.mutable_state_bytes, eager.mutable_state_bytes);

    let model = fixture_model_runtime_model_with_dynamic_partition(1_000, 64);
    let package_root = tiny_model_dir();
    let tensor_index = model.load_runtime_tensor_index(&package_root).unwrap();
    let plan = plan_vulkan_runtime_residency(
        &package_root,
        &model,
        &tensor_index,
        64,
        0,
        ResourceResidencyPolicy::DemandRetained,
    )
    .unwrap();
    let [device] = plan.device_plans.as_slice() else {
        panic!("the dynamic fixture must have one logical residency device");
    };
    let expected_mutable = device
        .resource_store
        .maximum_extra_device_bytes()
        .unwrap()
        .checked_add(device.working_set.transient_state_bytes)
        .and_then(|bytes| bytes.checked_add(device.working_set.activation_headroom_bytes))
        .unwrap();
    assert_eq!(retained.mutable_state_bytes, expected_mutable);
    let static_model = fixture_model_runtime_model();
    let static_reservations = exact_candidate_reservations_for_model(
        &static_model,
        ResourceResidencyPolicy::DemandRetained,
    );
    let static_bytes = &static_reservations.device_bytes[&hybrid_test_device("gpu0")];
    let static_tensor_index = static_model.load_runtime_tensor_index(&package_root).unwrap();
    let selected_tensor = static_model
        .circuit_graph
        .components
        .iter()
        .find(|component| component.component_id == "layer_00")
        .unwrap()
        .params
        .refs["ffn_down"]
        .tensor
        .as_ref()
        .unwrap();
    let selected_bytes = static_tensor_index.tensors[selected_tensor]
        .byte_count
        .unwrap();
    assert_eq!(
        static_bytes.permanent_bytes - retained.permanent_bytes,
        selected_bytes,
        "selected expert payload must move from permanent storage into cache quota exactly once",
    );
}

#[test]
fn runtime_hybrid_exact_cache_reserves_a_component_representation_wave() {
    let mut model = fixture_model_runtime_model_with_dynamic_partition(1_000, 64);
    model.package.resource_residency.partition_templates[0].member_templates[0]
        .resident_derivation = Some(CompiledResourceResidentDerivation {
        schema: RESIDENT_DERIVATION_SCHEMA.to_string(),
        kind: CompiledResourceResidentDerivationKind::Mxfp4E2m1ToFp8E4m3,
        source_byte_count: 64,
        resident_byte_count: 128,
        required_features: vec![
            "shader_float8".to_string(),
            "shader_int8".to_string(),
            "shader_mixed_float_dot_product_float8_acc_float32".to_string(),
        ],
    });

    let paged = exact_candidate_reservations_for_model(
        &model,
        ResourceResidencyPolicy::DemandPaged,
    );
    let retained = exact_candidate_reservations_for_model(
        &model,
        ResourceResidencyPolicy::DemandRetained,
    );
    let paged = &paged.device_bytes[&hybrid_test_device("gpu0")];
    let retained = &retained.device_bytes[&hybrid_test_device("gpu0")];

    assert_eq!(paged.atomic_load_wave_bytes, 64);
    assert_eq!(paged.cache_quota_bytes, 64 + 128 + 126);
    assert_eq!(retained.atomic_load_wave_bytes, 64);
    assert_eq!(retained.cache_quota_bytes, 64_000 + 128 + 126);
}

#[test]
fn runtime_hybrid_jointly_selects_a_faster_compatible_representation() {
    let model = fixture_model_runtime_model_with_three_layer_series("gpu0");
    let mut alternative = model.clone();
    alternative
        .component_executions
        .iter_mut()
        .find(|execution| execution.component_id == "layer_00")
        .unwrap()
        .implementation = "int4_representation".to_string();
    let alternative_signature = vulkan_runtime_placement_calibration_target_for_component(
        &alternative,
        "layer_00",
        VulkanTargetedComponentExecutionPhase::Decode,
    )
    .unwrap()
    .signature_id;
    let mut catalog = hybrid_test_catalog(&model);
    let behavior = hybrid_test_behavior(&alternative_signature);
    catalog
        .record_reference(VulkanPlacementCanonicalReference {
            behavior: behavior.clone(),
            output_digest: "output".to_string(),
            output_artifact: None,
            state_digest: "state".to_string(),
        })
        .unwrap();
    catalog
        .record_observation(hybrid_test_observation(behavior, "gpu0", 4))
        .unwrap();
    let application = hybrid_test_representation_application(alternative, &["layer_00"]);
    let capacity = VulkanPlacementCapacityEnvelope {
        available_bytes_by_device: BTreeMap::from([
            (hybrid_test_device("gpu0"), 100),
            (hybrid_test_device("gpu1"), 100),
        ]),
        host_available_bytes: 100,
    };

    let placement = try_plan_vulkan_runtime_hybrid_ordered_graph_with_representations(
        &model,
        &[application],
        &BTreeSet::from(["layer_00".to_string()]),
        &catalog,
        &capacity,
        VulkanTargetedComponentExecutionPhase::Decode,
    )
    .unwrap()
    .expect("the alternative must cover the incompatible baseline");

    assert_eq!(placement.ordered_placement.plan.predicted_duration_ns_per_activation, 24);
    assert_eq!(placement.selected_implementations.len(), 1);
    assert_eq!(
        placement.selected_implementations[0].instance_ids,
        ["layer_00"]
    );
    assert!(matches!(
        &placement.ordered_placement.plan.steps[0],
        VulkanHybridScheduledStep::Region { candidate_id, .. }
            if candidate_id.starts_with("representation:0:")
    ));
}

#[test]
fn runtime_hybrid_route_visitor_reaches_a_sampled_dominated_representation() {
    let model = fixture_model_runtime_model_with_three_layer_series("gpu0");
    let mut fast_model = model.clone();
    fast_model
        .component_executions
        .iter_mut()
        .find(|execution| execution.component_id == "layer_00")
        .unwrap()
        .implementation = "fast_large_representation".to_string();
    let mut slow_model = model.clone();
    slow_model
        .component_executions
        .iter_mut()
        .find(|execution| execution.component_id == "layer_00")
        .unwrap()
        .implementation = "slow_small_representation".to_string();
    let mut catalog = hybrid_test_catalog(&model);
    for (alternative, duration) in [(&fast_model, 4), (&slow_model, 6)] {
        let signature = vulkan_runtime_placement_calibration_target_for_component(
            alternative,
            "layer_00",
            VulkanTargetedComponentExecutionPhase::Decode,
        )
        .unwrap()
        .signature_id;
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
            .record_observation(hybrid_test_observation(behavior, "gpu0", duration))
            .unwrap();
    }
    let applications = [
        hybrid_test_representation_application(fast_model, &["layer_00"]),
        hybrid_test_representation_application(slow_model, &["layer_00"]),
    ];
    let capacity = VulkanPlacementCapacityEnvelope {
        available_bytes_by_device: BTreeMap::from([
            (hybrid_test_device("gpu0"), 100),
            (hybrid_test_device("gpu1"), 100),
        ]),
        host_available_bytes: 100,
    };
    let mut visited = Vec::new();

    let selected = visit_runtime_hybrid_representation_placements_by_duration(
        &model,
        &applications,
        &BTreeSet::from(["layer_00".to_string()]),
        &catalog,
        &capacity,
        VulkanTargetedComponentExecutionPhase::Decode,
        None,
        None,
        |placement| {
            let candidate_id = placement
                .ordered_placement
                .plan
                .steps
                .iter()
                .find_map(|step| match step {
                    VulkanHybridScheduledStep::Region {
                        candidate_id,
                        component_start: 0,
                        ..
                    } => Some(candidate_id.clone()),
                    _ => None,
                })
                .unwrap();
            visited.push(candidate_id.clone());
            Ok(candidate_id
                .starts_with("representation:1:")
                .then_some(placement))
        },
    )
    .unwrap()
    .expect("the terminal verifier accepts the slower representation");

    assert!(visited[0].starts_with("representation:0:"));
    assert!(visited[1].starts_with("representation:1:"));
    assert_eq!(visited.len(), 2);
    assert_eq!(selected.selected_implementations.len(), 1);
}

#[test]
fn runtime_hybrid_mounts_the_jointly_selected_representation_and_physical_plan_once() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "nerve-runtime-hybrid-mount-{}-{unique}",
        std::process::id(),
    ));
    let package_root = root.join("package");
    copy_runtime_implementation_fixture_tree(&tiny_model_dir(), &package_root);
    let runtime_model = fixture_model_runtime_model_with_three_layer_series("gpu0");
    let profile = runtime_compatibility_hardware_profile("gpu0", true);
    let (implementation_catalog, alternative_implementation) =
        hybrid_test_implementation_catalog(&package_root, &runtime_model, &profile);
    let execution = crate::RuntimeExecutionEnvelope {
        phases: vec!["decode".to_string()],
        activation_batch: crate::RuntimeInclusiveRange {
            minimum: 1,
            maximum: 1,
        },
        context_activations: crate::RuntimeInclusiveRange {
            minimum: 0,
            maximum: 64,
        },
        state_activations: crate::RuntimeInclusiveRange {
            minimum: 0,
            maximum: 64,
        },
        speculative_draft_tokens: 0,
        residency_policy: "eager".to_string(),
    };
    let request = crate::RuntimeSelectionRequest::from_vulkan_runtime_model(
        &runtime_model,
        &BTreeMap::from([("gpu0".to_string(), profile.clone())]),
        execution.clone(),
        BTreeSet::new(),
    )
    .unwrap();
    let mut independently_coverable_request = request.clone();
    independently_coverable_request.exact_baseline_incompatible_instance_ids = BTreeSet::from([
        "layer_00".to_string(),
        "layer_00_remote".to_string(),
        "layer_00_tail".to_string(),
    ]);
    assert_eq!(
        runtime_model
            .hybrid_signal_representation_applications_from_catalog(
                &package_root,
                &implementation_catalog,
                &independently_coverable_request,
            )
            .unwrap()
            .len(),
        3,
        "independent candidates must be composed by the joint solver rather than each being forced to cover every incompatible layer",
    );
    let applications = runtime_model
        .hybrid_signal_representation_applications_from_catalog(
            &package_root,
            &implementation_catalog,
            &request,
        )
        .unwrap();
    let alternative_model = &applications
        .iter()
        .find(|application| {
            application.selection.selected[0].instance_ids == ["layer_00"]
        })
        .unwrap()
        .runtime_model;
    let alternative_signature = vulkan_runtime_placement_calibration_target_for_component(
        alternative_model,
        "layer_00",
        VulkanTargetedComponentExecutionPhase::Decode,
    )
    .unwrap()
    .signature_id;
    let mut placement_catalog = hybrid_test_catalog(&runtime_model);
    let alternative_behavior = hybrid_test_behavior(&alternative_signature);
    placement_catalog
        .record_reference(VulkanPlacementCanonicalReference {
            behavior: alternative_behavior.clone(),
            output_digest: "output".to_string(),
            output_artifact: None,
            state_digest: "state".to_string(),
        })
        .unwrap();
    placement_catalog
        .record_observation(hybrid_test_observation(
            alternative_behavior,
            "gpu0",
            4,
        ))
        .unwrap();
    let capacity = VulkanPlacementCapacityEnvelope {
        available_bytes_by_device: BTreeMap::from([(
            hybrid_test_device("gpu0"),
            usize::MAX,
        )]),
        host_available_bytes: usize::MAX,
    };
    let physical_mount_devices = [VulkanRuntimePhysicalPlanningDevice {
        logical_device_id: "gpu0".to_string(),
        identity: hybrid_test_device("gpu0"),
        safe_capacity_bytes: usize::MAX,
        storage_buffer_offset_alignment: 8,
    }];

    let resolution = resolve_vulkan_runtime_hybrid_physical_execution_with_catalog(
        &package_root,
        &runtime_model,
        &BTreeMap::from([("gpu0".to_string(), profile)]),
        execution,
        &implementation_catalog,
        &placement_catalog,
        &capacity,
        64,
        &BTreeMap::from([("gpu0".to_string(), "gpu0".to_string())]),
        Some(VulkanRuntimeHybridMountPlanningContext {
            devices: &physical_mount_devices,
            physical_stream_requirement_resolver: None,
            speculative_draft_tokens: 0,
            residency_policy: ResourceResidencyPolicy::Eager,
            host_safe_capacity_bytes: usize::MAX,
        }),
        None,
    )
    .unwrap()
    .expect("measured alternative has a complete route");

    let selection = resolution
        .runtime_model
        .implementation_selection
        .as_ref()
        .expect("joint resolution must mount one canonical selection");
    assert_eq!(selection.selected.len(), 3);
    assert!(
        selection
            .selected
            .iter()
            .all(|selected| selected.implementation_id == "hybrid_implementation")
    );
    assert!(selection.exact_instance_ids.contains(&"output_transducer".to_string()));
    assert!(
        resolution
            .runtime_model
            .component_executions
            .iter()
            .filter(|execution| {
                resolution
                    .runtime_model
                    .circuit_graph
                    .components
                    .iter()
                    .find(|component| component.component_id == execution.component_id)
                    .is_some_and(|component| component.runtime_role.is_signal_processor())
            })
            .all(|execution| execution.implementation == alternative_implementation)
    );
    resolution
        .physical_execution_plan
        .validate(&resolution.runtime_model)
        .unwrap();
    assert!(
        resolution.physical_mount_plan.is_some(),
        "joint selection must pass exact full-context physical mount planning",
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn runtime_hybrid_rejects_noncontiguous_representation_applications() {
    let model = fixture_model_runtime_model_with_three_layer_series("gpu0");
    let application = hybrid_test_representation_application(
        model.clone(),
        &["layer_00", "layer_00_tail"],
    );
    let catalog = hybrid_test_catalog(&model);
    let capacity = VulkanPlacementCapacityEnvelope {
        available_bytes_by_device: BTreeMap::from([(hybrid_test_device("gpu0"), 100)]),
        host_available_bytes: 100,
    };

    let error = try_plan_vulkan_runtime_hybrid_ordered_graph_with_representations(
        &model,
        &[application],
        &BTreeSet::new(),
        &catalog,
        &capacity,
        VulkanTargetedComponentExecutionPhase::Decode,
    )
    .unwrap_err();

    assert!(error.0.contains("contiguous ordered graph region"));
}

#[test]
fn runtime_hybrid_cannot_retain_an_uncovered_incompatible_baseline() {
    let model = fixture_model_runtime_model_with_three_layer_series("gpu0");
    let catalog = hybrid_test_catalog(&model);
    let capacity = VulkanPlacementCapacityEnvelope {
        available_bytes_by_device: BTreeMap::from([(hybrid_test_device("gpu0"), 100)]),
        host_available_bytes: 100,
    };

    let placement = try_plan_vulkan_runtime_hybrid_ordered_graph_with_representations(
        &model,
        &[],
        &BTreeSet::from(["layer_00".to_string()]),
        &catalog,
        &capacity,
        VulkanTargetedComponentExecutionPhase::Decode,
    )
    .unwrap();

    assert!(placement.is_none());
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
    let model = fixture_model_runtime_model_with_three_layer_series("gpu0");
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
    assert_eq!(physical_plan.prefill_activation_batch_width, Some(4));
    assert!(physical_plan.component_device_pools.decode.is_empty());
    assert!(physical_plan.component_device_pools.prefill.is_empty());

    let mut missing_geometry = physical_plan.clone();
    missing_geometry.prefill_activation_batch_width = None;
    assert!(
        missing_geometry
            .validate(&stable_model)
            .unwrap_err()
            .0
            .contains("geometry and component cases must be declared together")
    );
    let mut scalar_geometry = physical_plan;
    scalar_geometry.prefill_activation_batch_width = Some(1);
    assert!(
        scalar_geometry
            .validate(&stable_model)
            .unwrap_err()
            .0
            .contains("lane capacity must be at least two")
    );
}

#[test]
fn runtime_hybrid_prefill_route_visitor_reaches_a_sampled_dominated_alternative() {
    let model = fixture_model_runtime_model_with_three_layer_series("gpu0");
    let phase = VulkanTargetedComponentExecutionPhase::Prefill {
        activation_batch_width: 4,
    };
    let mut signatures = model
        .circuit_graph
        .components
        .iter()
        .filter(|component| component.runtime_role.is_signal_processor())
        .map(|component| {
            vulkan_runtime_placement_calibration_target_for_component(
                &model,
                &component.component_id,
                phase,
            )
            .unwrap()
            .signature_id
        })
        .collect::<Vec<_>>();
    signatures.sort();
    signatures.dedup();

    let slow_artifact_digest = hybrid_test_digest('f');
    let mut catalog = VulkanPlacementCalibrationCatalog::default();
    for signature in signatures {
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
            .record_observation(hybrid_test_observation(behavior.clone(), "gpu0", 5))
            .unwrap();
        let mut slow = hybrid_test_observation(behavior, "gpu0", 8);
        slow.execution_case.implementation_digests = vec![hybrid_test_digest('e')];
        slow.execution_case.artifact_digest = slow_artifact_digest.clone();
        slow.resident_bytes_by_physical_device
            .insert("gpu0".to_string(), 20);
        catalog.record_observation(slow).unwrap();
    }
    let capacity = VulkanPlacementCapacityEnvelope {
        available_bytes_by_device: BTreeMap::from([(hybrid_test_device("gpu0"), 100)]),
        host_available_bytes: 100,
    };
    let required_owners = model
        .circuit_graph
        .components
        .iter()
        .filter(|component| component.runtime_role.is_signal_processor())
        .map(|component| (component.component_id.clone(), "gpu0".to_string()))
        .collect::<BTreeMap<_, _>>();
    let route_uses_slow_artifact = |placement: &VulkanRuntimeHybridOrderedPlacement| {
        placement.plan.steps.iter().any(|step| {
            matches!(
                step,
                VulkanHybridScheduledStep::Region { execution_case, .. }
                    if execution_case.artifact_digest == slow_artifact_digest
            )
        })
    };

    let pareto = try_plan_vulkan_runtime_hybrid_ordered_graph_with_owners(
        &model,
        &catalog,
        &capacity,
        phase,
        Some(&required_owners),
    )
    .unwrap()
    .expect("the sampled Pareto planner has a complete fast route");
    assert!(!route_uses_slow_artifact(&pareto));

    let mut visited_slow_artifact = Vec::new();
    let selected = visit_vulkan_runtime_hybrid_ordered_graph_with_owners_by_duration(
        &model,
        &catalog,
        &capacity,
        phase,
        Some(&required_owners),
        None,
        |placement| {
            let uses_slow_artifact = route_uses_slow_artifact(&placement);
            visited_slow_artifact.push(uses_slow_artifact);
            Ok(uses_slow_artifact.then_some(placement))
        },
    )
    .unwrap()
    .expect("the terminal verifier accepts a sampled-dominated prefill route");

    assert_eq!(visited_slow_artifact, [false, true]);
    assert!(route_uses_slow_artifact(&selected));
    assert_eq!(selected.activation_batch_width, 4);
}

#[test]
fn runtime_hybrid_try_phase_set_preserves_decode_when_prefill_cannot_keep_its_owners() {
    let model = fixture_model_runtime_model_with_three_layer_series("gpu0");
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
    let model = fixture_model_runtime_model_with_three_layer_series("gpu0");
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
