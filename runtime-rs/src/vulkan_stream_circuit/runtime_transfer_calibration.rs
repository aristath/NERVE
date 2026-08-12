const VULKAN_DIRECTED_TRANSFER_CONTRACT_ID: &str = "nerve.physical_activation_boundary.copy.v1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanRuntimePlacementTransferCalibrationReport {
    pub source_device_id: String,
    pub source_api_version: u32,
    pub source_driver_version: u32,
    pub target_device_id: String,
    pub target_api_version: u32,
    pub target_driver_version: u32,
    pub phase: nerve_execution_contracts::ExecutionPhase,
    pub activation_batch_width: usize,
    pub frame_byte_count: usize,
    pub byte_count: usize,
    pub route: VulkanSharedResidentBufferRoute,
    pub warmup_ns: u64,
    pub measured_ns: u64,
    pub fixture_digest: String,
    pub output_digest: String,
}

impl VulkanRuntimePlacementTransferCalibrationReport {
    pub fn canonical_reference(
        &self,
    ) -> Result<VulkanPlacementCanonicalReference, VulkanPlacementCalibrationCatalogError> {
        self.validate_geometry()?;
        Ok(VulkanPlacementCanonicalReference {
            behavior: self.behavior_identity(),
            output_digest: self.fixture_digest.clone(),
            output_artifact: None,
            state_digest: runtime_transfer_calibration_state_digest(),
        })
    }

    pub fn calibration_observation(
        &self,
    ) -> Result<VulkanPlacementCalibrationObservation, VulkanPlacementCalibrationCatalogError> {
        self.validate_geometry()?;
        let mut devices = vec![
            VulkanPlacementDeviceExecutionIdentity {
                physical_device_id: self.source_device_id.clone(),
                api_version: self.source_api_version,
                driver_version: self.source_driver_version,
            },
            VulkanPlacementDeviceExecutionIdentity {
                physical_device_id: self.target_device_id.clone(),
                api_version: self.target_api_version,
                driver_version: self.target_driver_version,
            },
        ];
        devices.sort();
        let source_transient = self
            .byte_count
            .checked_mul(match self.route {
                VulkanSharedResidentBufferRoute::ExternalDeviceLocal => 2,
                VulkanSharedResidentBufferRoute::SharedHost => 1,
            })
            .ok_or_else(|| {
                VulkanPlacementCalibrationCatalogError(
                    "directed transfer transient byte accounting overflowed".to_string(),
                )
            })?;
        Ok(VulkanPlacementCalibrationObservation {
            execution_case: VulkanPlacementExecutionCaseIdentity {
                behavior: self.behavior_identity(),
                strategy: VulkanPlacementExecutionStrategy::DirectedBoundary,
                devices,
                shards: Vec::new(),
                input_physical_device_id: self.source_device_id.clone(),
                output_physical_device_id: self.target_device_id.clone(),
                owner_physical_device_id: self.source_device_id.clone(),
                transports: vec![VulkanPlacementTransportIdentity {
                    source_physical_device_id: self.source_device_id.clone(),
                    destination_physical_device_id: self.target_device_id.clone(),
                    byte_capacity: self.byte_count,
                    route: runtime_transfer_calibration_route_name(self.route).to_string(),
                }],
            },
            warmup_call_count: 1,
            measured_call_count: 2,
            complete_transaction: true,
            duration_ns: self.measured_ns,
            useful_activation_count: 1,
            output_digest: self.output_digest.clone(),
            output_artifact: None,
            output_equivalence: VulkanPlacementOutputEquivalenceEvidence::BitExact,
            state_digest: runtime_transfer_calibration_state_digest(),
            resident_bytes_by_physical_device: BTreeMap::from([
                (self.source_device_id.clone(), 0),
                (self.target_device_id.clone(), 0),
            ]),
            transient_peak_bytes_by_physical_device: BTreeMap::from([
                (self.source_device_id.clone(), source_transient),
                (self.target_device_id.clone(), self.byte_count),
            ]),
            host_resident_bytes: 0,
            host_transient_peak_bytes: match self.route {
                VulkanSharedResidentBufferRoute::ExternalDeviceLocal => 0,
                VulkanSharedResidentBufferRoute::SharedHost => self.byte_count,
            },
        })
    }

    fn behavior_identity(&self) -> VulkanPlacementBehaviorIdentity {
        let implementation_digest = runtime_transfer_calibration_digest(
            b"nerve.directed_transfer.implementation.v1",
            &[crate::RUNTIME_IMPLEMENTATION_FINGERPRINT.as_bytes()],
        );
        let artifact_digest = runtime_transfer_calibration_digest(
            b"nerve.directed_transfer.artifact.v1",
            &[
                crate::RUNTIME_IMPLEMENTATION_FINGERPRINT.as_bytes(),
                VULKAN_DIRECTED_TRANSFER_CONTRACT_ID.as_bytes(),
            ],
        );
        let execution_graph_digest = runtime_transfer_calibration_digest(
            b"nerve.directed_transfer.graph.v1",
            &[
                VULKAN_DIRECTED_TRANSFER_CONTRACT_ID.as_bytes(),
                &self.byte_count.to_le_bytes(),
            ],
        );
        VulkanPlacementBehaviorIdentity {
            compiled_execution_signature: execution_graph_digest.clone(),
            contract_ids: vec![VULKAN_DIRECTED_TRANSFER_CONTRACT_ID.to_string()],
            implementation_digests: vec![implementation_digest],
            artifact_digest,
            execution_graph_digest,
            runtime_implementation_fingerprint: crate::RUNTIME_IMPLEMENTATION_FINGERPRINT
                .to_string(),
            phase: self.phase,
            shape: VulkanPlacementShapeClass {
                activation_batch_width: self.activation_batch_width,
                input_byte_capacity: self.byte_count,
                output_byte_capacity: self.byte_count,
                operations: vec![VulkanPlacementOperationGeometry::DirectedTransfer {
                    contract_id: VULKAN_DIRECTED_TRANSFER_CONTRACT_ID.to_string(),
                    byte_count: self.byte_count,
                }],
            },
            input_fixture_digest: self.fixture_digest.clone(),
            equivalence: VulkanPlacementEquivalenceIdentity::bit_exact(),
        }
    }

    fn validate_geometry(&self) -> Result<(), VulkanPlacementCalibrationCatalogError> {
        let expected_byte_count = self
            .frame_byte_count
            .checked_mul(self.activation_batch_width)
            .ok_or_else(|| {
                VulkanPlacementCalibrationCatalogError(
                    "directed transfer batch geometry overflows".to_string(),
                )
            })?;
        if self.frame_byte_count == 0
            || self.activation_batch_width == 0
            || self.byte_count != expected_byte_count
            || (self.phase == nerve_execution_contracts::ExecutionPhase::Decode
                && self.activation_batch_width != 1)
        {
            return Err(VulkanPlacementCalibrationCatalogError(
                "directed transfer report has inconsistent phase, frame, batch, or payload geometry"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

pub fn record_vulkan_runtime_transfer_calibration_report(
    catalog: &mut VulkanPlacementCalibrationCatalog,
    report: &VulkanRuntimePlacementTransferCalibrationReport,
) -> Result<(), VulkanPlacementCalibrationCatalogError> {
    let reference = report.canonical_reference()?;
    let observation = report.calibration_observation()?;
    let mut updated = catalog.clone();
    updated.record_reference(reference)?;
    updated.record_observation(observation)?;
    *catalog = updated;
    Ok(())
}

pub fn vulkan_runtime_placement_transfer_byte_counts(
    runtime_model: &VulkanResidentRuntimeModel,
) -> Result<Vec<usize>, VulkanRuntimeResidencyPlanError> {
    Ok(
        vulkan_runtime_placement_boundary_byte_counts(runtime_model)?
            .into_iter()
            .collect(),
    )
}

/// Measures the exact activation payload sizes and physical route that the
/// mounted cross-device circuit will use. One cold replay is discarded and
/// two device-timestamped replays answer the deliberately binary placement
/// question without turning startup into a benchmark suite.
pub fn calibrate_vulkan_runtime_placement_transfers(
    source_device_id: &str,
    source: &VulkanComputeDevice,
    target_device_id: &str,
    target: &VulkanComputeDevice,
    byte_counts: &[usize],
) -> Result<Vec<VulkanRuntimePlacementTransferCalibrationReport>, VulkanError> {
    calibrate_vulkan_runtime_placement_phase_transfers(
        source_device_id,
        source,
        target_device_id,
        target,
        nerve_execution_contracts::ExecutionPhase::Decode,
        1,
        byte_counts,
    )
}

/// Measures the contiguous activation batch used by the real phase-specific
/// cross-device edge. `frame_byte_counts` are compiler-emitted per-activation
/// capacities; the transfer transaction scales them exactly once by the
/// mounted batch width.
pub fn calibrate_vulkan_runtime_placement_phase_transfers(
    source_device_id: &str,
    source: &VulkanComputeDevice,
    target_device_id: &str,
    target: &VulkanComputeDevice,
    phase: nerve_execution_contracts::ExecutionPhase,
    activation_batch_width: usize,
    frame_byte_counts: &[usize],
) -> Result<Vec<VulkanRuntimePlacementTransferCalibrationReport>, VulkanError> {
    if source_device_id.is_empty()
        || target_device_id.is_empty()
        || source_device_id == target_device_id
        || source.shares_logical_device_with(target)
    {
        return Err(VulkanError(
            "runtime transfer calibration requires two distinct named physical devices".to_string(),
        ));
    }
    if activation_batch_width == 0
        || (phase == nerve_execution_contracts::ExecutionPhase::Decode
            && activation_batch_width != 1)
    {
        return Err(VulkanError(
            "runtime transfer calibration requires decode width one or a positive prefill width"
                .to_string(),
        ));
    }
    let unique_frame_byte_counts = frame_byte_counts.iter().copied().collect::<BTreeSet<_>>();
    if unique_frame_byte_counts.len() != frame_byte_counts.len()
        || unique_frame_byte_counts
            .iter()
            .any(|byte_count| *byte_count == 0)
    {
        return Err(VulkanError(
            "runtime transfer calibration requires unique positive frame payload sizes".to_string(),
        ));
    }
    unique_frame_byte_counts
        .into_iter()
        .map(|frame_byte_count| {
            let byte_count = frame_byte_count
                .checked_mul(activation_batch_width)
                .ok_or_else(|| {
                    VulkanError(
                        "runtime transfer calibration batch payload size overflowed".to_string(),
                    )
                })?;
            calibrate_vulkan_runtime_placement_transfer(
                source_device_id,
                source,
                target_device_id,
                target,
                phase,
                activation_batch_width,
                frame_byte_count,
                byte_count,
            )
        })
        .collect()
}

fn calibrate_vulkan_runtime_placement_transfer(
    source_device_id: &str,
    source: &VulkanComputeDevice,
    target_device_id: &str,
    target: &VulkanComputeDevice,
    phase: nerve_execution_contracts::ExecutionPhase,
    activation_batch_width: usize,
    frame_byte_count: usize,
    byte_count: usize,
) -> Result<VulkanRuntimePlacementTransferCalibrationReport, VulkanError> {
    let fixture = runtime_transfer_calibration_fixture(byte_count);
    let fixture_digest = format!("sha256:{:x}", Sha256::digest(&fixture));
    let source_buffer = source.create_resident_buffer(byte_count)?;
    source_buffer.write_bytes(&fixture)?;
    let target_buffer = target.create_resident_buffer(byte_count)?;
    let shared = if source.supports_opaque_fd_timeline_semaphores()
        && target.supports_opaque_fd_timeline_semaphores()
    {
        source.create_shared_resident_buffers(&[target], byte_count)?
    } else {
        let allocation = source.create_shared_host_allocation(&[target], byte_count)?;
        VulkanSharedResidentBufferSet {
            route: VulkanSharedResidentBufferRoute::SharedHost,
            buffers: vec![
                Arc::new(source.import_shared_host_buffer(Arc::clone(&allocation))?),
                Arc::new(target.import_shared_host_buffer(allocation)?),
            ],
            external_device_local_error: Some(
                "cross-device timeline semaphores are unavailable".to_string(),
            ),
        }
    };
    let source_shared = shared
        .buffers
        .first()
        .ok_or_else(|| VulkanError("transfer calibration omitted its source view".to_string()))?;
    let target_shared = shared
        .buffers
        .get(1)
        .ok_or_else(|| VulkanError("transfer calibration omitted its target view".to_string()))?;
    let source_copy = source.create_timestamped_resident_buffer_copy(
        &source_buffer,
        source_shared,
        byte_count,
    )?;
    let target_copy = target.create_timestamped_resident_buffer_copy(
        target_shared,
        &target_buffer,
        byte_count,
    )?;
    let measure = || -> Result<u64, VulkanError> {
        source_copy
            .run_with_device_duration(byte_count)?
            .checked_add(target_copy.run_with_device_duration(byte_count)?)
            .ok_or_else(|| VulkanError("runtime transfer calibration time overflowed".to_string()))
    };
    let warmup_ns = measure()?;
    let measured_ns = measure()?.min(measure()?).max(1);
    let output = target_buffer.read_bytes(byte_count)?;
    validate_runtime_transfer_calibration_output(&fixture, &output)?;
    let output_digest = format!("sha256:{:x}", Sha256::digest(&output));
    Ok(VulkanRuntimePlacementTransferCalibrationReport {
        source_device_id: source_device_id.to_string(),
        source_api_version: source.api_version(),
        source_driver_version: source.driver_version(),
        target_device_id: target_device_id.to_string(),
        target_api_version: target.api_version(),
        target_driver_version: target.driver_version(),
        phase,
        activation_batch_width,
        frame_byte_count,
        byte_count,
        route: shared.route,
        warmup_ns,
        measured_ns,
        fixture_digest,
        output_digest,
    })
}

fn runtime_transfer_calibration_route_name(route: VulkanSharedResidentBufferRoute) -> &'static str {
    match route {
        VulkanSharedResidentBufferRoute::ExternalDeviceLocal => "external_device_local",
        VulkanSharedResidentBufferRoute::SharedHost => "shared_host",
    }
}

fn runtime_transfer_calibration_digest(domain: &[u8], fields: &[&[u8]]) -> String {
    let mut digest = Sha256::new();
    digest.update(domain);
    digest.update([0]);
    for field in fields {
        digest.update((field.len() as u64).to_le_bytes());
        digest.update(field);
    }
    format!("sha256:{:x}", digest.finalize())
}

fn runtime_transfer_calibration_state_digest() -> String {
    runtime_transfer_calibration_digest(b"nerve.directed_transfer.stateless.v1", &[])
}

fn runtime_transfer_calibration_fixture(byte_count: usize) -> Vec<u8> {
    (0..byte_count)
        .map(|index| {
            let index = index as u64;
            index
                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                .rotate_left((index & 31) as u32) as u8
                ^ 0xa5
        })
        .collect()
}

fn validate_runtime_transfer_calibration_output(
    fixture: &[u8],
    output: &[u8],
) -> Result<(), VulkanError> {
    if fixture != output {
        return Err(VulkanError(
            "runtime transfer calibration produced invalid destination bytes".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod runtime_transfer_calibration_validation_tests {
    use super::*;

    #[test]
    fn fixture_is_nonuniform_and_deterministic() {
        let first = runtime_transfer_calibration_fixture(4096);
        let second = runtime_transfer_calibration_fixture(4096);
        assert_eq!(first, second);
        assert!(first.windows(2).any(|pair| pair[0] != pair[1]));
        assert_ne!(&first[..1024], &first[1024..2048]);
    }

    #[test]
    fn validation_rejects_corruption_at_every_boundary() {
        let fixture = runtime_transfer_calibration_fixture(257);
        validate_runtime_transfer_calibration_output(&fixture, &fixture).unwrap();
        for index in [0, fixture.len() / 2, fixture.len() - 1] {
            let mut corrupt = fixture.clone();
            corrupt[index] ^= 1;
            assert!(validate_runtime_transfer_calibration_output(&fixture, &corrupt).is_err());
        }
        assert!(validate_runtime_transfer_calibration_output(&fixture, &fixture[..256]).is_err());
    }

    fn report(
        route: VulkanSharedResidentBufferRoute,
    ) -> VulkanRuntimePlacementTransferCalibrationReport {
        let fixture = runtime_transfer_calibration_fixture(257);
        let digest = format!("sha256:{:x}", Sha256::digest(&fixture));
        VulkanRuntimePlacementTransferCalibrationReport {
            source_device_id: "gpu-a".to_string(),
            source_api_version: 1,
            source_driver_version: 2,
            target_device_id: "gpu-b".to_string(),
            target_api_version: 3,
            target_driver_version: 4,
            phase: nerve_execution_contracts::ExecutionPhase::Decode,
            activation_batch_width: 1,
            frame_byte_count: fixture.len(),
            byte_count: fixture.len(),
            route,
            warmup_ns: 11,
            measured_ns: 10,
            fixture_digest: digest.clone(),
            output_digest: digest,
        }
    }

    #[test]
    fn directed_boundary_report_records_an_exact_typed_observation() {
        let report = report(VulkanSharedResidentBufferRoute::SharedHost);
        let mut catalog = VulkanPlacementCalibrationCatalog::default();
        record_vulkan_runtime_transfer_calibration_report(&mut catalog, &report).unwrap();
        let observation = report.calibration_observation().unwrap();
        assert_eq!(
            catalog.exact_observation(&observation.execution_case),
            Some(&observation),
        );
        assert_eq!(
            observation.execution_case.strategy,
            VulkanPlacementExecutionStrategy::DirectedBoundary,
        );
        assert!(matches!(
            observation
                .execution_case
                .behavior
                .shape
                .operations
                .as_slice(),
            [VulkanPlacementOperationGeometry::DirectedTransfer {
                byte_count: 257,
                ..
            }],
        ));
    }

    #[test]
    fn route_or_driver_change_creates_a_distinct_boundary_case() {
        let shared = report(VulkanSharedResidentBufferRoute::SharedHost)
            .calibration_observation()
            .unwrap();
        let mut external_report = report(VulkanSharedResidentBufferRoute::ExternalDeviceLocal);
        external_report.target_driver_version += 1;
        let external = external_report.calibration_observation().unwrap();
        assert_eq!(
            shared.execution_case.behavior,
            external.execution_case.behavior
        );
        assert_ne!(shared.execution_case, external.execution_case);
    }

    #[test]
    fn prefill_boundary_identity_preserves_batch_geometry() {
        let mut report = report(VulkanSharedResidentBufferRoute::SharedHost);
        report.phase = nerve_execution_contracts::ExecutionPhase::Prefill;
        report.activation_batch_width = 64;
        report.frame_byte_count = 257;
        report.byte_count = 257 * 64;
        let observation = report.calibration_observation().unwrap();
        assert_eq!(
            observation
                .execution_case
                .behavior
                .shape
                .activation_batch_width,
            64,
        );
        assert_eq!(
            observation
                .execution_case
                .behavior
                .shape
                .input_byte_capacity,
            257 * 64,
        );
        assert!(matches!(
            observation.execution_case.behavior.shape.operations.as_slice(),
            [VulkanPlacementOperationGeometry::DirectedTransfer { byte_count, .. }]
                if *byte_count == 257 * 64
        ));
    }

    #[test]
    fn inconsistent_boundary_geometry_is_rejected_transactionally() {
        let mut report = report(VulkanSharedResidentBufferRoute::SharedHost);
        report.phase = nerve_execution_contracts::ExecutionPhase::Prefill;
        report.activation_batch_width = 64;
        let mut catalog = VulkanPlacementCalibrationCatalog::default();
        let before = catalog.clone();
        assert!(
            record_vulkan_runtime_transfer_calibration_report(&mut catalog, &report)
                .unwrap_err()
                .to_string()
                .contains("inconsistent")
        );
        assert_eq!(catalog, before);

        report.byte_count = report.frame_byte_count * 64;
        record_vulkan_runtime_transfer_calibration_report(&mut catalog, &report).unwrap();
        assert_eq!(catalog.observation_count(), 1);
    }
}
