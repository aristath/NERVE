#[cfg(test)]
mod runtime_region_placement_calibration_tests {
    use super::*;

    fn digest(byte: char) -> String {
        format!("sha256:{}", byte.to_string().repeat(64))
    }

    fn device(id: &str) -> VulkanPlacementDeviceExecutionIdentity {
        VulkanPlacementDeviceExecutionIdentity {
            physical_device_id: id.to_string(),
            api_version: 1,
            driver_version: 2,
        }
    }

    fn component(
        id: &str,
        input_device: &str,
        output_device: &str,
        strategy: VulkanPlacementExecutionStrategy,
        digest_byte: char,
    ) -> VulkanPlacementExecutionCaseIdentity {
        VulkanPlacementExecutionCaseIdentity {
            behavior: VulkanPlacementBehaviorIdentity {
                compiled_execution_signature: format!("signature:{id}"),
                runtime_implementation_fingerprint: "runtime".to_string(),
                phase: nerve_execution_contracts::ExecutionPhase::Decode,
                shape: VulkanPlacementShapeClass {
                    activation_batch_width: 1,
                    input_byte_capacity: 16,
                    output_byte_capacity: 16,
                },
                input_fixture_digest: digest(digest_byte),
            },
            contract_ids: vec![format!("contract:{id}")],
            implementation_digests: vec![digest(digest_byte)],
            artifact_digest: digest(digest_byte),
            execution_graph_digest: digest(digest_byte),
            operations: vec![VulkanPlacementOperationGeometry::Dispatch {
                geometry: VulkanPlacementDispatchGeometry {
                    contract_id: format!("contract:{id}"),
                    logical_extent: 8,
                    sampled_extent: 8,
                    input_width: 8,
                    workgroup_count_x: 1,
                    local_size_x: 64,
                },
            }],
            equivalence: VulkanPlacementEquivalenceIdentity::bit_exact(),
            strategy,
            devices: [input_device, output_device]
                .into_iter()
                .map(device)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect(),
            shards: Vec::new(),
            input_physical_device_id: input_device.to_string(),
            output_physical_device_id: output_device.to_string(),
            owner_physical_device_id: input_device.to_string(),
            transports: Vec::new(),
        }
    }

    fn boundary() -> VulkanPlacementRegionBoundaryExecutionCase {
        VulkanPlacementRegionBoundaryExecutionCase {
            boundary_ordinal: 0,
            execution_case: VulkanPlacementExecutionCaseIdentity {
                behavior: VulkanPlacementBehaviorIdentity {
                    compiled_execution_signature: "boundary".to_string(),
                    runtime_implementation_fingerprint: "runtime".to_string(),
                    phase: nerve_execution_contracts::ExecutionPhase::Decode,
                    shape: VulkanPlacementShapeClass {
                        activation_batch_width: 1,
                        input_byte_capacity: 16,
                        output_byte_capacity: 16,
                    },
                    input_fixture_digest: digest('c'),
                },
                contract_ids: vec!["boundary".to_string()],
                implementation_digests: vec![digest('c')],
                artifact_digest: digest('c'),
                execution_graph_digest: digest('c'),
                operations: vec![VulkanPlacementOperationGeometry::DirectedTransfer {
                    contract_id: "boundary".to_string(),
                    byte_count: 16,
                }],
                equivalence: VulkanPlacementEquivalenceIdentity::bit_exact(),
                strategy: VulkanPlacementExecutionStrategy::DirectedBoundary,
                devices: vec![device("gpu0"), device("gpu1")],
                shards: Vec::new(),
                input_physical_device_id: "gpu0".to_string(),
                output_physical_device_id: "gpu1".to_string(),
                owner_physical_device_id: "gpu0".to_string(),
                transports: vec![VulkanPlacementTransportIdentity {
                    source_physical_device_id: "gpu0".to_string(),
                    destination_physical_device_id: "gpu1".to_string(),
                    byte_capacity: 16,
                    route: "device_local_staging".to_string(),
                }],
            },
        }
    }

    #[test]
    fn region_target_preserves_one_ordered_mixed_physical_transaction() {
        let first = component(
            "a",
            "gpu0",
            "gpu0",
            VulkanPlacementExecutionStrategy::TensorParallel,
            'a',
        );
        let second = component(
            "b",
            "gpu1",
            "gpu1",
            VulkanPlacementExecutionStrategy::SingleDevice,
            'b',
        );
        let target = vulkan_runtime_region_placement_calibration_target(
            vec!["a".to_string(), "b".to_string()],
            vec![first.clone(), second.clone()],
            vec![16],
            vec![boundary()],
            VulkanPlacementScalarFormat::Bf16,
        )
        .unwrap();

        assert_eq!(
            target.execution_case.strategy,
            VulkanPlacementExecutionStrategy::HybridRegion
        );
        assert_eq!(target.execution_case.operations.len(), 3);
        assert_eq!(target.execution_case.input_physical_device_id, "gpu0");
        assert_eq!(target.execution_case.output_physical_device_id, "gpu1");
        assert_eq!(target.execution_case.transports.len(), 1);
        assert!(target.execution_case.shards.is_empty());
        assert_eq!(target.component_cases, [first, second]);
    }

    #[test]
    fn region_target_rejects_missing_boundary_evidence() {
        let error = vulkan_runtime_region_placement_calibration_target(
            vec!["a".to_string(), "b".to_string()],
            vec![
                component(
                    "a",
                    "gpu0",
                    "gpu0",
                    VulkanPlacementExecutionStrategy::SingleDevice,
                    'a',
                ),
                component(
                    "b",
                    "gpu1",
                    "gpu1",
                    VulkanPlacementExecutionStrategy::SingleDevice,
                    'b',
                ),
            ],
            vec![16],
            Vec::new(),
            VulkanPlacementScalarFormat::Bf16,
        )
        .unwrap_err();

        assert!(error.0.contains("physical boundary 0"));
    }

    #[test]
    fn region_target_uses_strictest_tolerance_for_direct_outer_validation() {
        let mut first = component(
            "a",
            "gpu0",
            "gpu0",
            VulkanPlacementExecutionStrategy::SingleDevice,
            'a',
        );
        first.equivalence = VulkanPlacementEquivalenceIdentity {
            output: VulkanPlacementEquivalenceKind::AbsoluteRelativeTolerance,
            state: VulkanPlacementEquivalenceKind::BitExact,
            absolute_tolerance_bits: Some(0.01f64.to_bits()),
            relative_tolerance_bits: Some(0.01f64.to_bits()),
            output_scalar_format: Some(VulkanPlacementScalarFormat::Bf16),
        };
        let target = vulkan_runtime_region_placement_calibration_target(
            vec!["a".to_string(), "b".to_string()],
            vec![
                first,
                component(
                    "b",
                    "gpu0",
                    "gpu0",
                    VulkanPlacementExecutionStrategy::SingleDevice,
                    'b',
                ),
            ],
            vec![16],
            Vec::new(),
            VulkanPlacementScalarFormat::Bf16,
        )
        .unwrap();

        assert_eq!(
            target.execution_case.equivalence.output,
            VulkanPlacementEquivalenceKind::AbsoluteRelativeTolerance
        );
        assert_eq!(
            target.execution_case.equivalence.absolute_tolerance(),
            Some(0.01)
        );
        assert_eq!(
            target.execution_case.equivalence.output_scalar_format,
            Some(VulkanPlacementScalarFormat::Bf16)
        );
    }

    #[test]
    fn measured_outer_transaction_records_reference_observation_and_exact_replay() {
        let target = vulkan_runtime_region_placement_calibration_target(
            vec!["a".to_string(), "b".to_string()],
            vec![
                component(
                    "a",
                    "gpu0",
                    "gpu0",
                    VulkanPlacementExecutionStrategy::SingleDevice,
                    'a',
                ),
                component(
                    "b",
                    "gpu1",
                    "gpu1",
                    VulkanPlacementExecutionStrategy::SingleDevice,
                    'b',
                ),
            ],
            vec![16],
            vec![boundary()],
            VulkanPlacementScalarFormat::Bf16,
        )
        .unwrap();
        let captured_output_artifact = VulkanPlacementOutputArtifact {
            scalar_format: VulkanPlacementScalarFormat::Bf16,
            segments: vec![VulkanPlacementOutputSegment {
                binding: 0,
                name: "output".to_string(),
                bytes: vec![0, 0],
            }],
        };
        let output_digest =
            vulkan_placement_output_artifact_digest(&captured_output_artifact).unwrap();
        let report = VulkanRuntimeRegionPlacementCalibrationReport {
            target: target.clone(),
            warmup_execution_ns: 20,
            measured_execution_ns: 10,
            measured_ns_per_activation: 10,
            warmup_call_count: 1,
            measured_call_count: 1,
            useful_activation_count: 1,
            output_digest,
            captured_output_artifact,
            output_artifact: None,
            state_digest: "state".to_string(),
            resident_bytes_by_physical_device: BTreeMap::from([
                ("gpu0".to_string(), 100),
                ("gpu1".to_string(), 100),
            ]),
            transient_peak_bytes_by_physical_device: BTreeMap::from([
                ("gpu0".to_string(), 10),
                ("gpu1".to_string(), 10),
            ]),
            host_transient_peak_bytes: 16,
        };
        let mut catalog = VulkanPlacementCalibrationCatalog::default();

        record_vulkan_runtime_region_placement_calibration_report(&mut catalog, &report)
            .unwrap();

        assert_eq!(catalog.reference_count(), 1);
        assert_eq!(catalog.observation_count(), 1);
        assert_eq!(catalog.region_execution_count(), 1);
        assert_eq!(
            catalog
                .region_execution(&target.execution_case)
                .unwrap()
                .component_cases,
            target.component_cases,
        );
        catalog.validate().unwrap();
    }

    #[test]
    fn same_device_multi_component_region_records_as_serialized_region() {
        let target = vulkan_runtime_region_placement_calibration_target(
            vec!["a".to_string(), "b".to_string()],
            vec![
                component(
                    "a",
                    "gpu0",
                    "gpu0",
                    VulkanPlacementExecutionStrategy::SingleDevice,
                    'a',
                ),
                component(
                    "b",
                    "gpu0",
                    "gpu0",
                    VulkanPlacementExecutionStrategy::SingleDevice,
                    'b',
                ),
            ],
            vec![16],
            Vec::new(),
            VulkanPlacementScalarFormat::Bf16,
        )
        .unwrap();
        assert_eq!(
            target.execution_case.strategy,
            VulkanPlacementExecutionStrategy::SerializedRegion
        );
        assert_eq!(target.execution_case.devices, vec![device("gpu0")]);

        let captured_output_artifact = VulkanPlacementOutputArtifact {
            scalar_format: VulkanPlacementScalarFormat::Bf16,
            segments: vec![VulkanPlacementOutputSegment {
                binding: 0,
                name: "output".to_string(),
                bytes: vec![0, 0],
            }],
        };
        let report = VulkanRuntimeRegionPlacementCalibrationReport {
            target: target.clone(),
            warmup_execution_ns: 20,
            measured_execution_ns: 10,
            measured_ns_per_activation: 10,
            warmup_call_count: 1,
            measured_call_count: 1,
            useful_activation_count: 1,
            output_digest: vulkan_placement_output_artifact_digest(
                &captured_output_artifact,
            )
            .unwrap(),
            captured_output_artifact,
            output_artifact: None,
            state_digest: "state".to_string(),
            resident_bytes_by_physical_device: BTreeMap::from([(
                "gpu0".to_string(),
                100,
            )]),
            transient_peak_bytes_by_physical_device: BTreeMap::from([(
                "gpu0".to_string(),
                10,
            )]),
            host_transient_peak_bytes: 0,
        };
        let mut catalog = VulkanPlacementCalibrationCatalog::default();

        record_vulkan_runtime_region_placement_calibration_report(&mut catalog, &report)
            .unwrap();

        assert_eq!(catalog.reference_count(), 1);
        assert_eq!(catalog.observation_count(), 1);
        assert_eq!(catalog.region_execution_count(), 1);
        catalog.validate().unwrap();
    }

    #[test]
    fn tolerant_region_candidate_is_checked_against_numeric_serialized_reference() {
        let artifact = |value: f32| VulkanPlacementOutputArtifact {
            scalar_format: VulkanPlacementScalarFormat::Bf16,
            segments: vec![VulkanPlacementOutputSegment {
                binding: 0,
                name: "output".to_string(),
                bytes: (((value.to_bits() >> 16) as u16).to_le_bytes()).to_vec(),
            }],
        };
        let serialized_target = vulkan_runtime_region_placement_calibration_target(
            vec!["a".to_string(), "b".to_string()],
            vec![
                component(
                    "a",
                    "gpu0",
                    "gpu0",
                    VulkanPlacementExecutionStrategy::SingleDevice,
                    'a',
                ),
                component(
                    "b",
                    "gpu0",
                    "gpu0",
                    VulkanPlacementExecutionStrategy::SingleDevice,
                    'b',
                ),
            ],
            vec![16],
            Vec::new(),
            VulkanPlacementScalarFormat::Bf16,
        )
        .unwrap();
        let mut tolerant_second = component(
            "b",
            "gpu1",
            "gpu1",
            VulkanPlacementExecutionStrategy::SingleDevice,
            'b',
        );
        tolerant_second.equivalence = VulkanPlacementEquivalenceIdentity {
            output: VulkanPlacementEquivalenceKind::AbsoluteRelativeTolerance,
            state: VulkanPlacementEquivalenceKind::BitExact,
            absolute_tolerance_bits: Some(0.01f64.to_bits()),
            relative_tolerance_bits: Some(0.01f64.to_bits()),
            output_scalar_format: Some(VulkanPlacementScalarFormat::Bf16),
        };
        let hybrid_target = vulkan_runtime_region_placement_calibration_target(
            vec!["a".to_string(), "b".to_string()],
            vec![
                component(
                    "a",
                    "gpu0",
                    "gpu0",
                    VulkanPlacementExecutionStrategy::SingleDevice,
                    'a',
                ),
                tolerant_second,
            ],
            vec![16],
            vec![boundary()],
            VulkanPlacementScalarFormat::Bf16,
        )
        .unwrap();
        assert_eq!(
            serialized_target.execution_case.behavior,
            hybrid_target.execution_case.behavior
        );

        let reference_artifact = artifact(1.0);
        let candidate_artifact = artifact(1.0078125);
        let report = |target: VulkanRuntimeRegionPlacementCalibrationTarget,
                      captured_output_artifact: VulkanPlacementOutputArtifact,
                      output_artifact: Option<VulkanPlacementOutputArtifact>| {
            let devices = target
                .execution_case
                .devices
                .iter()
                .map(|device| (device.physical_device_id.clone(), 100))
                .collect::<BTreeMap<_, _>>();
            VulkanRuntimeRegionPlacementCalibrationReport {
                target,
                warmup_execution_ns: 20,
                measured_execution_ns: 10,
                measured_ns_per_activation: 10,
                warmup_call_count: 1,
                measured_call_count: 1,
                useful_activation_count: 1,
                output_digest: vulkan_placement_output_artifact_digest(
                    &captured_output_artifact,
                )
                .unwrap(),
                captured_output_artifact,
                output_artifact,
                state_digest: "state".to_string(),
                resident_bytes_by_physical_device: devices.clone(),
                transient_peak_bytes_by_physical_device: devices,
                host_transient_peak_bytes: 16,
            }
        };
        let serialized_report = report(serialized_target, reference_artifact, None);
        let hybrid_report = report(
            hybrid_target.clone(),
            candidate_artifact.clone(),
            Some(candidate_artifact),
        );
        let mut catalog = VulkanPlacementCalibrationCatalog::default();

        record_vulkan_runtime_region_placement_calibration_report(
            &mut catalog,
            &serialized_report,
        )
        .unwrap();
        record_vulkan_runtime_region_placement_calibration_report(&mut catalog, &hybrid_report)
            .unwrap();

        assert!(
            catalog
                .canonical_reference(&hybrid_target.execution_case.behavior)
                .unwrap()
                .output_artifact
                .is_some()
        );
        assert!(matches!(
            catalog
                .exact_observation(&hybrid_target.execution_case)
                .unwrap()
                .output_equivalence,
            VulkanPlacementOutputEquivalenceEvidence::AbsoluteRelativeTolerance { .. }
        ));
        catalog.validate().unwrap();
    }

    #[test]
    fn region_memory_evidence_excludes_opened_nonparticipants() {
        let before = BTreeMap::from([
            ("gpu0".to_string(), 1_000),
            ("gpu1".to_string(), 2_000),
            ("unused".to_string(), 9_000),
        ]);
        let after_package = BTreeMap::from([
            ("gpu0".to_string(), 1_100),
            ("gpu1".to_string(), 2_200),
            ("unused".to_string(), 9_500),
        ]);
        let peak = BTreeMap::from([
            ("gpu0".to_string(), 1_125),
            ("gpu1".to_string(), 2_250),
            ("unused".to_string(), 9_750),
        ]);

        let (resident, transient) = runtime_region_device_memory_evidence(
            &[device("gpu0"), device("gpu1")],
            &before,
            &after_package,
            &peak,
        )
        .unwrap();

        assert_eq!(
            resident,
            BTreeMap::from([("gpu0".to_string(), 100), ("gpu1".to_string(), 200)])
        );
        assert_eq!(
            transient,
            BTreeMap::from([("gpu0".to_string(), 25), ("gpu1".to_string(), 50)])
        );
        assert!(!resident.contains_key("unused"));
        assert!(!transient.contains_key("unused"));
    }

    #[test]
    fn region_memory_evidence_rejects_duplicate_or_underflowed_participants() {
        let before = BTreeMap::from([("gpu0".to_string(), 100)]);
        let after_package = BTreeMap::from([("gpu0".to_string(), 90)]);
        let peak = BTreeMap::from([("gpu0".to_string(), 110)]);

        assert!(
            runtime_region_device_memory_evidence(
                &[device("gpu0")],
                &before,
                &after_package,
                &peak,
            )
            .unwrap_err()
            .to_string()
            .contains("underflowed")
        );

        let after_package = BTreeMap::from([("gpu0".to_string(), 100)]);
        assert!(
            runtime_region_device_memory_evidence(
                &[device("gpu0"), device("gpu0")],
                &before,
                &after_package,
                &peak,
            )
            .unwrap_err()
            .to_string()
            .contains("repeats")
        );
    }
}
