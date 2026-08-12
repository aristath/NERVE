const VULKAN_DIRECTED_TRANSFER_CONTRACT_ID: &str =
    "nerve.physical_activation_boundary.transaction.v2";

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
    pub route: VulkanPlacedEdgeTransferRoute,
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
        let implementation_digest = runtime_transfer_calibration_digest(
            b"nerve.directed_transfer.implementation.v2",
            &[crate::RUNTIME_IMPLEMENTATION_FINGERPRINT.as_bytes()],
        );
        let artifact_digest = runtime_transfer_calibration_digest(
            b"nerve.directed_transfer.artifact.v2",
            &[
                crate::RUNTIME_IMPLEMENTATION_FINGERPRINT.as_bytes(),
                VULKAN_DIRECTED_TRANSFER_CONTRACT_ID.as_bytes(),
            ],
        );
        let execution_graph_digest = self.execution_graph_digest();
        Ok(VulkanPlacementCalibrationObservation {
            execution_case: VulkanPlacementExecutionCaseIdentity {
                behavior: self.behavior_identity(),
                contract_ids: vec![VULKAN_DIRECTED_TRANSFER_CONTRACT_ID.to_string()],
                implementation_digests: vec![implementation_digest],
                artifact_digest,
                execution_graph_digest,
                operations: vec![VulkanPlacementOperationGeometry::DirectedTransfer {
                    contract_id: VULKAN_DIRECTED_TRANSFER_CONTRACT_ID.to_string(),
                    byte_count: self.byte_count,
                }],
                equivalence: VulkanPlacementEquivalenceIdentity::bit_exact(),
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
            resident_bytes_by_physical_device: match self.route {
                VulkanPlacedEdgeTransferRoute::ExternalDeviceLocal => BTreeMap::from([
                    (self.source_device_id.clone(), self.byte_count),
                    (self.target_device_id.clone(), 0),
                ]),
                VulkanPlacedEdgeTransferRoute::DeviceLocalStaging => BTreeMap::from([
                    (self.source_device_id.clone(), self.byte_count),
                    (self.target_device_id.clone(), self.byte_count),
                ]),
                _ => unreachable!("transfer report route was validated"),
            },
            transient_peak_bytes_by_physical_device: BTreeMap::from([
                (self.source_device_id.clone(), 0),
                (self.target_device_id.clone(), 0),
            ]),
            host_resident_bytes: match self.route {
                VulkanPlacedEdgeTransferRoute::ExternalDeviceLocal => 0,
                VulkanPlacedEdgeTransferRoute::DeviceLocalStaging => self.byte_count,
                _ => unreachable!("transfer report route was validated"),
            },
            host_transient_peak_bytes: 0,
        })
    }

    fn behavior_identity(&self) -> VulkanPlacementBehaviorIdentity {
        VulkanPlacementBehaviorIdentity {
            compiled_execution_signature: self.execution_graph_digest(),
            runtime_implementation_fingerprint: crate::RUNTIME_IMPLEMENTATION_FINGERPRINT
                .to_string(),
            phase: self.phase,
            shape: VulkanPlacementShapeClass {
                activation_batch_width: self.activation_batch_width,
                input_byte_capacity: self.byte_count,
                output_byte_capacity: self.byte_count,
            },
            input_fixture_digest: self.fixture_digest.clone(),
        }
    }

    fn execution_graph_digest(&self) -> String {
        runtime_transfer_calibration_digest(
            b"nerve.directed_transfer.graph.v2",
            &[
                VULKAN_DIRECTED_TRANSFER_CONTRACT_ID.as_bytes(),
                &self.byte_count.to_le_bytes(),
            ],
        )
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
            || !matches!(
                self.route,
                VulkanPlacedEdgeTransferRoute::ExternalDeviceLocal
                    | VulkanPlacedEdgeTransferRoute::DeviceLocalStaging
            )
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
    if !source.supports_opaque_fd_timeline_semaphores()
        || !target.supports_opaque_fd_timeline_semaphores()
    {
        return Err(VulkanError(
            "runtime transfer calibration requires cross-device timeline semaphores because a host-synchronized route is not resident replayable"
                .to_string(),
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
    let shared = source.create_shared_resident_buffers(&[target], byte_count)?;
    let transaction = VulkanRuntimeTransferCalibrationTransaction::new(
        source,
        target,
        shared,
        byte_count,
        &fixture,
    )?;
    let warmup_ns = transaction.measure(source, target, 1)?;
    let measured_ns = transaction
        .measure(source, target, 2)?
        .min(transaction.measure(source, target, 3)?)
        .max(1);
    let output = transaction.output.read_bytes(byte_count)?;
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
        route: transaction.route,
        warmup_ns,
        measured_ns,
        fixture_digest,
        output_digest,
    })
}

struct VulkanRuntimeTransferCalibrationTransaction {
    route: VulkanPlacedEdgeTransferRoute,
    output: Arc<VulkanResidentBuffer>,
    source_copy: Option<VulkanResidentBufferCopy>,
    destination_copy: Option<VulkanResidentBufferCopy>,
    source_signal: VulkanTimelineSemaphore,
    destination_wait: VulkanTimelineSemaphore,
    completion: VulkanTimelineSemaphore,
    _buffers: Vec<Arc<VulkanResidentBuffer>>,
}

impl VulkanRuntimeTransferCalibrationTransaction {
    fn new(
        source: &VulkanComputeDevice,
        target: &VulkanComputeDevice,
        shared: VulkanSharedResidentBufferSet,
        byte_count: usize,
        fixture: &[u8],
    ) -> Result<Self, VulkanError> {
        let source_signal = source.create_opaque_fd_exportable_timeline_semaphore(0)?;
        let destination_wait = target.create_timeline_semaphore(0)?;
        target.import_timeline_semaphore_opaque_fd(
            &destination_wait,
            source.export_timeline_semaphore_opaque_fd(&source_signal)?,
        )?;
        let completion = target.create_timeline_semaphore(0)?;
        let source_shared = shared
            .buffers
            .first()
            .cloned()
            .ok_or_else(|| VulkanError("transfer calibration omitted its source view".to_string()))?;
        let target_shared = shared
            .buffers
            .get(1)
            .cloned()
            .ok_or_else(|| VulkanError("transfer calibration omitted its target view".to_string()))?;
        match shared.route {
            VulkanSharedResidentBufferRoute::ExternalDeviceLocal => {
                source_shared.write_bytes(fixture)?;
                Ok(Self {
                    route: VulkanPlacedEdgeTransferRoute::ExternalDeviceLocal,
                    output: Arc::clone(&target_shared),
                    source_copy: None,
                    destination_copy: None,
                    source_signal,
                    destination_wait,
                    completion,
                    _buffers: shared.buffers,
                })
            }
            VulkanSharedResidentBufferRoute::SharedHost => {
                let source_buffer = Arc::new(source.create_resident_buffer(byte_count)?);
                source_buffer.write_bytes(fixture)?;
                let target_buffer = Arc::new(target.create_resident_buffer(byte_count)?);
                let source_copy = source.create_resident_buffer_copy(
                    &source_buffer,
                    &source_shared,
                    byte_count,
                )?;
                let destination_copy = target.create_resident_buffer_copy(
                    &target_shared,
                    &target_buffer,
                    byte_count,
                )?;
                let mut buffers = shared.buffers;
                buffers.push(source_buffer);
                buffers.push(Arc::clone(&target_buffer));
                Ok(Self {
                    route: VulkanPlacedEdgeTransferRoute::DeviceLocalStaging,
                    output: target_buffer,
                    source_copy: Some(source_copy),
                    destination_copy: Some(destination_copy),
                    source_signal,
                    destination_wait,
                    completion,
                    _buffers: buffers,
                })
            }
        }
    }

    fn measure(
        &self,
        source: &VulkanComputeDevice,
        target: &VulkanComputeDevice,
        timeline_value: u64,
    ) -> Result<u64, VulkanError> {
        let source_signal = VulkanTimelineSemaphorePoint::new(&self.source_signal, timeline_value);
        let destination_wait =
            VulkanTimelineSemaphorePoint::new(&self.destination_wait, timeline_value);
        let completion = VulkanTimelineSemaphorePoint::new(&self.completion, timeline_value);
        let started = Instant::now();
        match (&self.source_copy, &self.destination_copy) {
            (None, None) => {
                source.submit_timeline_semaphore_bridge(&[], &[source_signal])?;
                target.submit_timeline_semaphore_bridge(&[destination_wait], &[completion])?;
            }
            (Some(source_copy), Some(destination_copy)) => {
                source.submit_resident_buffer_copy_with_timeline_semaphores(
                    source_copy,
                    &[],
                    &[source_signal],
                )?;
                target.submit_resident_buffer_copy_with_timeline_semaphores(
                    destination_copy,
                    &[destination_wait],
                    &[completion],
                )?;
            }
            _ => {
                return Err(VulkanError(
                    "transfer calibration transaction has an incomplete staging route"
                        .to_string(),
                ));
            }
        }
        target.wait_timeline_semaphore_value(&self.completion, timeline_value)?;
        u64::try_from(started.elapsed().as_nanos())
            .map(|duration| duration.max(1))
            .map_err(|_| VulkanError("runtime transfer calibration time overflowed".to_string()))
    }
}

fn runtime_transfer_calibration_route_name(route: VulkanPlacedEdgeTransferRoute) -> &'static str {
    match route {
        VulkanPlacedEdgeTransferRoute::ExternalDeviceLocal => "external_device_local",
        VulkanPlacedEdgeTransferRoute::DeviceLocalStaging => "device_local_staging",
        _ => unreachable!("transfer report route was validated"),
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
        route: VulkanPlacedEdgeTransferRoute,
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
        let report = report(VulkanPlacedEdgeTransferRoute::DeviceLocalStaging);
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
                .operations
                .as_slice(),
            [VulkanPlacementOperationGeometry::DirectedTransfer {
                byte_count: 257,
                ..
            }],
        ));
        assert_eq!(
            observation.execution_case.transports[0].route,
            "device_local_staging"
        );
        assert_eq!(observation.resident_bytes_by_physical_device["gpu-a"], 257);
        assert_eq!(observation.resident_bytes_by_physical_device["gpu-b"], 257);
        assert_eq!(observation.host_resident_bytes, 257);
        assert_eq!(observation.host_transient_peak_bytes, 0);
    }

    #[test]
    fn route_or_driver_change_creates_a_distinct_boundary_case() {
        let shared = report(VulkanPlacedEdgeTransferRoute::DeviceLocalStaging)
            .calibration_observation()
            .unwrap();
        let mut external_report = report(VulkanPlacedEdgeTransferRoute::ExternalDeviceLocal);
        external_report.target_driver_version += 1;
        let external = external_report.calibration_observation().unwrap();
        assert_eq!(
            shared.execution_case.behavior,
            external.execution_case.behavior
        );
        assert_ne!(shared.execution_case, external.execution_case);
        assert_eq!(external.resident_bytes_by_physical_device["gpu-a"], 257);
        assert_eq!(external.resident_bytes_by_physical_device["gpu-b"], 0);
        assert_eq!(external.host_resident_bytes, 0);
    }

    #[test]
    fn report_rejects_a_route_that_the_resident_boundary_cannot_mount() {
        let invalid = report(VulkanPlacedEdgeTransferRoute::SharedHost);
        assert!(
            invalid
                .calibration_observation()
                .unwrap_err()
                .to_string()
                .contains("inconsistent")
        );
    }

    #[test]
    fn prefill_boundary_identity_preserves_batch_geometry() {
        let mut report = report(VulkanPlacedEdgeTransferRoute::DeviceLocalStaging);
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
            observation.execution_case.operations.as_slice(),
            [VulkanPlacementOperationGeometry::DirectedTransfer { byte_count, .. }]
                if *byte_count == 257 * 64
        ));
    }

    #[test]
    fn inconsistent_boundary_geometry_is_rejected_transactionally() {
        let mut report = report(VulkanPlacedEdgeTransferRoute::DeviceLocalStaging);
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
