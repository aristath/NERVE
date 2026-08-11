#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanRuntimeLoadWaveCalibrationReport {
    pub physical_device_id: String,
    pub api_version: u32,
    pub driver_version: u32,
    pub component_id: String,
    pub selector_id: String,
    pub resource_indices: Vec<usize>,
    pub group_ids: Vec<String>,
    pub phase: nerve_execution_contracts::ExecutionPhase,
    pub activation_batch_width: usize,
    pub loaded_group_count: usize,
    pub loaded_resource_count: usize,
    pub loaded_byte_count: usize,
    pub warmup_ns: u64,
    pub measured_ns: u64,
    pub output_digest: String,
    pub state_digest: String,
    pub resident_device_bytes: usize,
    pub transient_peak_device_bytes: usize,
    pub transient_host_bytes: usize,
    pub execution_case: VulkanPlacementExecutionCaseIdentity,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanRuntimeLoadWaveCalibrationTarget {
    pub component_id: String,
    pub selector_id: String,
    pub resource_indices: Vec<usize>,
    pub phase: nerve_execution_contracts::ExecutionPhase,
    pub activation_batch_width: usize,
}

impl VulkanRuntimeLoadWaveCalibrationReport {
    fn validate(&self) -> Result<(), VulkanPlacementCalibrationCatalogError> {
        let shape = &self.execution_case.behavior.shape;
        let operation_matches = matches!(
            shape.operations.as_slice(),
            [VulkanPlacementOperationGeometry::LazyLoadWave {
                contract_id,
                resource_count,
                byte_count,
            }] if contract_id == &self.selector_id
                && *resource_count == self.loaded_group_count
                && *byte_count == self.loaded_byte_count
        );
        let device_matches = matches!(
            self.execution_case.devices.as_slice(),
            [device]
                if device.physical_device_id == self.physical_device_id
                    && device.api_version == self.api_version
                    && device.driver_version == self.driver_version
        );
        if self.component_id.is_empty()
            || self.selector_id.is_empty()
            || self.resource_indices.is_empty()
            || self.resource_indices.windows(2).any(|pair| pair[0] >= pair[1])
            || self.group_ids.is_empty()
            || self.group_ids.windows(2).any(|pair| pair[0] >= pair[1])
            || self.loaded_group_count != self.group_ids.len()
            || self.loaded_resource_count == 0
            || self.loaded_byte_count == 0
            || self.warmup_ns == 0
            || self.measured_ns == 0
            || self.output_digest.is_empty()
            || self.state_digest.is_empty()
            || self.transient_peak_device_bytes < self.resident_device_bytes
            || self.activation_batch_width == 0
            || (self.phase == nerve_execution_contracts::ExecutionPhase::Decode
                && self.activation_batch_width != 1)
            || self.execution_case.strategy != VulkanPlacementExecutionStrategy::LazyLoadWave
            || self.execution_case.behavior.phase != self.phase
            || shape.activation_batch_width != self.activation_batch_width
            || shape.input_byte_capacity != self.loaded_byte_count
            || shape.output_byte_capacity != self.loaded_byte_count
            || self.execution_case.behavior.contract_ids != [self.selector_id.clone()]
            || !operation_matches
            || !device_matches
            || self.execution_case.input_physical_device_id != self.physical_device_id
            || self.execution_case.output_physical_device_id != self.physical_device_id
            || self.execution_case.owner_physical_device_id != self.physical_device_id
            || !self.execution_case.shards.is_empty()
            || !self.execution_case.transports.is_empty()
        {
            return Err(VulkanPlacementCalibrationCatalogError(
                "load-wave calibration report is internally inconsistent".to_string(),
            ));
        }
        Ok(())
    }

    fn canonical_reference(&self) -> VulkanPlacementCanonicalReference {
        VulkanPlacementCanonicalReference {
            behavior: self.execution_case.behavior.clone(),
            output_digest: self.output_digest.clone(),
            state_digest: self.state_digest.clone(),
        }
    }

    fn calibration_observation(&self) -> VulkanPlacementCalibrationObservation {
        VulkanPlacementCalibrationObservation {
            execution_case: self.execution_case.clone(),
            warmup_call_count: 1,
            measured_call_count: 1,
            complete_transaction: true,
            duration_ns: self.measured_ns,
            useful_activation_count: 1,
            output_digest: self.output_digest.clone(),
            state_digest: self.state_digest.clone(),
            resident_bytes_by_physical_device: BTreeMap::from([(
                self.physical_device_id.clone(),
                self.resident_device_bytes,
            )]),
            transient_peak_bytes_by_physical_device: BTreeMap::from([(
                self.physical_device_id.clone(),
                self.transient_peak_device_bytes,
            )]),
            host_resident_bytes: 0,
            host_transient_peak_bytes: self.transient_host_bytes,
        }
    }
}

pub fn record_vulkan_runtime_load_wave_calibration_report(
    catalog: &mut VulkanPlacementCalibrationCatalog,
    report: &VulkanRuntimeLoadWaveCalibrationReport,
) -> Result<(), VulkanPlacementCalibrationCatalogError> {
    report.validate()?;
    let mut updated = catalog.clone();
    updated.record_reference(report.canonical_reference())?;
    updated.record_observation(report.calibration_observation())?;
    *catalog = updated;
    Ok(())
}

struct VulkanRuntimeLoadWaveRun {
    duration_ns: u64,
    loaded_group_count: usize,
    validation: VulkanCompiledResourceReadbackValidation,
    before: VulkanCompiledResourceStoreReport,
    after: VulkanCompiledResourceStoreReport,
}

pub fn calibrate_vulkan_runtime_load_wave(
    physical_device_id: &str,
    device: Rc<VulkanComputeDevice>,
    manifest_dir: impl AsRef<Path>,
    runtime_model: &VulkanResidentRuntimeModel,
    target: &VulkanRuntimeLoadWaveCalibrationTarget,
) -> Result<VulkanRuntimeLoadWaveCalibrationReport, VulkanResidentTokenModelPackageError> {
    let manifest_dir = manifest_dir.as_ref();
    if physical_device_id.is_empty()
        || device.physical_device_id() != physical_device_id
        || target.component_id.is_empty()
        || target.selector_id.is_empty()
        || target.resource_indices.is_empty()
        || target
            .resource_indices
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        || target.activation_batch_width == 0
        || (target.phase == nerve_execution_contracts::ExecutionPhase::Decode
            && target.activation_batch_width != 1)
    {
        return load_wave_calibration_error(
            "load-wave calibration requires an exact device, component, selector, phase, and sorted unique resource indices",
        );
    }
    let contract = instantiate_runtime_resource_contract(runtime_model)
        .map_err(|error| load_wave_calibration_error_value(error.to_string()))?;
    let selector = contract
        .selectors
        .iter()
        .find(|selector| selector.id == target.selector_id)
        .ok_or_else(|| {
            load_wave_calibration_error_value(format!(
                "load-wave calibration selector {:?} is absent",
                target.selector_id,
            ))
        })?;
    if selector.execution_scope != runtime_model.execution_scope
        || selector.component_id != target.component_id
        || target
            .resource_indices
            .iter()
            .any(|resource_index| *resource_index >= selector.resource_count)
        || target.resource_indices.len() > selector.encoding.selection_count_per_activation
    {
        return load_wave_calibration_error(
            "load-wave calibration selector does not own the requested component resources",
        );
    }

    let logical_device_id = "calibration:load_wave";
    let placed_model = vulkan_runtime_model_with_component_placement(
        runtime_model,
        "calibration:unmounted",
        &BTreeMap::from([(
            target.component_id.clone(),
            logical_device_id.to_string(),
        )]),
    )
    .map_err(|error| load_wave_calibration_error_value(error.to_string()))?;
    let calibration_started = Instant::now();
    let warmup = run_vulkan_runtime_load_wave_once(
        Rc::clone(&device),
        manifest_dir,
        &placed_model,
        &target.component_id,
        &target.selector_id,
        &target.resource_indices,
        logical_device_id,
    )?;
    if calibration_started.elapsed() >= VULKAN_RUNTIME_PLACEMENT_CALIBRATION_MAXIMUM_DURATION {
        return load_wave_calibration_error(
            "load-wave calibration warmup exceeded the complete one-minute bound",
        );
    }
    let measured = run_vulkan_runtime_load_wave_once(
        Rc::clone(&device),
        manifest_dir,
        &placed_model,
        &target.component_id,
        &target.selector_id,
        &target.resource_indices,
        logical_device_id,
    )?;
    if calibration_started.elapsed() > VULKAN_RUNTIME_PLACEMENT_CALIBRATION_MAXIMUM_DURATION {
        return load_wave_calibration_error(
            "load-wave calibration exceeded the complete one-minute bound",
        );
    }
    if warmup.loaded_group_count != measured.loaded_group_count
        || warmup.validation != measured.validation
    {
        return load_wave_calibration_error(
            "load-wave calibration changed its exact resident payload between warmup and measurement",
        );
    }
    let resident_device_bytes = measured
        .after
        .current_device_bytes
        .saturating_sub(measured.before.current_device_bytes);
    let transient_peak_device_bytes = measured
        .after
        .high_water_device_bytes
        .saturating_sub(measured.before.current_device_bytes)
        .max(resident_device_bytes);
    let transient_host_bytes = usize::try_from(
        measured
            .after
            .physical_bytes_read
            .saturating_sub(measured.before.physical_bytes_read),
    )
    .unwrap_or(usize::MAX)
    .saturating_add(measured.after.transfer_staging_host_bytes);
    let behavior = load_wave_behavior_identity(
        runtime_model,
        selector,
        &target.resource_indices,
        &measured.validation,
        target.phase,
        target.activation_batch_width,
    )?;
    let state_digest = load_wave_digest(
        b"nerve.lazy_load_wave.state.v1",
        &[
            target.selector_id.as_bytes(),
            serde_json::to_vec(&measured.validation.group_ids)
                .map_err(|error| load_wave_calibration_error_value(error.to_string()))?
                .as_slice(),
            &measured.validation.byte_count.to_le_bytes(),
        ],
    );
    let execution_case = VulkanPlacementExecutionCaseIdentity {
        behavior,
        strategy: VulkanPlacementExecutionStrategy::LazyLoadWave,
        devices: vec![VulkanPlacementDeviceExecutionIdentity {
            physical_device_id: physical_device_id.to_string(),
            api_version: device.api_version(),
            driver_version: device.driver_version(),
        }],
        shards: Vec::new(),
        input_physical_device_id: physical_device_id.to_string(),
        output_physical_device_id: physical_device_id.to_string(),
        owner_physical_device_id: physical_device_id.to_string(),
        transports: Vec::new(),
    };
    Ok(VulkanRuntimeLoadWaveCalibrationReport {
        physical_device_id: physical_device_id.to_string(),
        api_version: device.api_version(),
        driver_version: device.driver_version(),
        component_id: target.component_id.clone(),
        selector_id: target.selector_id.clone(),
        resource_indices: target.resource_indices.clone(),
        group_ids: measured.validation.group_ids.clone(),
        phase: target.phase,
        activation_batch_width: target.activation_batch_width,
        loaded_group_count: measured.loaded_group_count,
        loaded_resource_count: measured.validation.resource_count,
        loaded_byte_count: measured.validation.byte_count,
        warmup_ns: warmup.duration_ns,
        measured_ns: measured.duration_ns.max(1),
        output_digest: measured.validation.output_digest,
        state_digest,
        resident_device_bytes,
        transient_peak_device_bytes,
        transient_host_bytes,
        execution_case,
    })
}

fn run_vulkan_runtime_load_wave_once(
    device: Rc<VulkanComputeDevice>,
    manifest_dir: &Path,
    runtime_model: &VulkanResidentRuntimeModel,
    component_id: &str,
    selector_id: &str,
    resource_indices: &[usize],
    logical_device_id: &str,
) -> Result<VulkanRuntimeLoadWaveRun, VulkanResidentTokenModelPackageError> {
    let parameter_pool = VulkanResidentBufferPool::default();
    parameter_pool
        .register_device(logical_device_id, Rc::clone(&device))
        .map_err(|error| load_wave_calibration_error_value(error.to_string()))?;
    let execution_result = (|| {
        let targeted =
            VulkanResidentTargetedModelPackageDeviceSlice::from_runtime_model_for_device_with_parameter_pool(
                &device,
                manifest_dir,
                runtime_model.clone(),
                component_id,
                logical_device_id,
                Some(1),
                &parameter_pool,
            )?;
        let context = targeted.demand_context.as_ref().ok_or_else(|| {
            load_wave_calibration_error_value(
                "load-wave calibration component has no demand-residency context",
            )
        })?;
        if !context.store.allowed_selector_ids().contains(selector_id) {
            return load_wave_calibration_error(
                "load-wave calibration selector is outside the component store",
            );
        }
        let before = context
            .store
            .residency_report()
            .map_err(|error| load_wave_calibration_error_value(error.to_string()))?;
        let started = Instant::now();
        let loaded_group_count = context
            .store
            .load_selector_resources(
                &device,
                selector_id,
                resource_indices,
                context.owner.clone(),
            )
            .map_err(|error| load_wave_calibration_error_value(error.to_string()))?;
        let duration_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        if loaded_group_count > context.store.maximum_load_wave_group_count() {
            return load_wave_calibration_error(format!(
                "load-wave calibration requested {loaded_group_count} groups but one physical wave admits only {}",
                context.store.maximum_load_wave_group_count(),
            ));
        }
        let after = context
            .store
            .residency_report()
            .map_err(|error| load_wave_calibration_error_value(error.to_string()))?;
        let validation = context
            .store
            .validate_selector_resources_readback(&device, selector_id, resource_indices)
            .map_err(|error| load_wave_calibration_error_value(error.to_string()))?;
        if loaded_group_count != validation.group_ids.len()
            || after.resident_unit_count
                < before
                    .resident_unit_count
                    .saturating_add(loaded_group_count)
        {
            return load_wave_calibration_error(
                "load-wave calibration did not publish every loaded group as resident",
            );
        }
        Ok(VulkanRuntimeLoadWaveRun {
            duration_ns: duration_ns.max(1),
            loaded_group_count,
            validation,
            before,
            after,
        })
    })();
    let mut cleanup_errors = [
        device.quiesce().err().map(|error| error.to_string()),
        parameter_pool
            .release_device(logical_device_id)
            .err()
            .map(|error| error.to_string()),
        device.quiesce().err().map(|error| error.to_string()),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
    let pool_stats = parameter_pool.stats();
    if parameter_pool.registered_device_count() != 0
        || pool_stats.resident_allocation_count != 0
        || pool_stats.resident_buffer_count != 0
        || pool_stats.resident_bytes != 0
    {
        cleanup_errors.push("load-wave calibration retained parameter-pool state".to_string());
    }
    match (execution_result, cleanup_errors.is_empty()) {
        (Ok(run), true) => Ok(run),
        (Err(error), true) => Err(error),
        (Ok(_), false) => load_wave_calibration_error(cleanup_errors.join("; ")),
        (Err(error), false) => load_wave_calibration_error(format!(
            "{error}; cleanup also failed: {}",
            cleanup_errors.join("; "),
        )),
    }
}

fn load_wave_behavior_identity(
    runtime_model: &VulkanResidentRuntimeModel,
    selector: &CompiledResourceSelector,
    resource_indices: &[usize],
    validation: &VulkanCompiledResourceReadbackValidation,
    phase: nerve_execution_contracts::ExecutionPhase,
    activation_batch_width: usize,
) -> Result<VulkanPlacementBehaviorIdentity, VulkanResidentTokenModelPackageError> {
    let selector_bytes = serde_json::to_vec(selector)
        .map_err(|error| load_wave_calibration_error_value(error.to_string()))?;
    let artifact_bytes = serde_json::to_vec(&runtime_model.package.artifact_integrity)
        .map_err(|error| load_wave_calibration_error_value(error.to_string()))?;
    let graph_bytes = serde_json::to_vec(&(
        &runtime_model.runtime_graph,
        &runtime_model.execution_scope,
        &selector.component_id,
        &selector.id,
        resource_indices,
    ))
    .map_err(|error| load_wave_calibration_error_value(error.to_string()))?;
    let indices_bytes = serde_json::to_vec(resource_indices)
        .map_err(|error| load_wave_calibration_error_value(error.to_string()))?;
    Ok(VulkanPlacementBehaviorIdentity {
        contract_ids: vec![selector.id.clone()],
        implementation_digests: vec![load_wave_digest(
            b"nerve.lazy_load_wave.implementation.v1",
            &[
                crate::RUNTIME_IMPLEMENTATION_FINGERPRINT.as_bytes(),
                &selector_bytes,
            ],
        )],
        artifact_digest: load_wave_digest(
            b"nerve.lazy_load_wave.artifact.v1",
            &[&artifact_bytes, &selector_bytes],
        ),
        execution_graph_digest: load_wave_digest(
            b"nerve.lazy_load_wave.graph.v1",
            &[&graph_bytes],
        ),
        runtime_implementation_fingerprint: crate::RUNTIME_IMPLEMENTATION_FINGERPRINT.to_string(),
        phase,
        shape: VulkanPlacementShapeClass {
            activation_batch_width,
            input_byte_capacity: validation.byte_count,
            output_byte_capacity: validation.byte_count,
            operations: vec![VulkanPlacementOperationGeometry::LazyLoadWave {
                contract_id: selector.id.clone(),
                resource_count: validation.group_ids.len(),
                byte_count: validation.byte_count,
            }],
        },
        input_fixture_digest: load_wave_digest(
            b"nerve.lazy_load_wave.request.v1",
            &[selector.id.as_bytes(), &indices_bytes],
        ),
    })
}

fn load_wave_digest(domain: &[u8], fields: &[&[u8]]) -> String {
    let mut digest = Sha256::new();
    digest.update(domain);
    digest.update([0]);
    for field in fields {
        digest.update((field.len() as u64).to_le_bytes());
        digest.update(field);
    }
    format!("sha256:{:x}", digest.finalize())
}

fn load_wave_calibration_error<T>(
    message: impl Into<String>,
) -> Result<T, VulkanResidentTokenModelPackageError> {
    Err(load_wave_calibration_error_value(message))
}

fn load_wave_calibration_error_value(
    message: impl Into<String>,
) -> VulkanResidentTokenModelPackageError {
    VulkanResidentTokenModelPackageError::new(message)
}

#[cfg(test)]
mod runtime_load_wave_calibration_validation_tests {
    use super::*;

    fn digest(byte: u8) -> String {
        format!("sha256:{}", format!("{byte:02x}").repeat(32))
    }

    fn report() -> VulkanRuntimeLoadWaveCalibrationReport {
        let behavior = VulkanPlacementBehaviorIdentity {
            contract_ids: vec![digest(1)],
            implementation_digests: vec![digest(2)],
            artifact_digest: digest(3),
            execution_graph_digest: digest(4),
            runtime_implementation_fingerprint: "runtime".to_string(),
            phase: nerve_execution_contracts::ExecutionPhase::Decode,
            shape: VulkanPlacementShapeClass {
                activation_batch_width: 1,
                input_byte_capacity: 4096,
                output_byte_capacity: 4096,
                operations: vec![VulkanPlacementOperationGeometry::LazyLoadWave {
                    contract_id: digest(1),
                    resource_count: 1,
                    byte_count: 4096,
                }],
            },
            input_fixture_digest: digest(5),
        };
        let physical_device_id = "vulkan-uuid:00112233445566778899aabbccddeeff".to_string();
        VulkanRuntimeLoadWaveCalibrationReport {
            physical_device_id: physical_device_id.clone(),
            api_version: 1,
            driver_version: 2,
            component_id: "component".to_string(),
            selector_id: digest(1),
            resource_indices: vec![0, 1, 2, 3, 4, 5],
            group_ids: vec![digest(6)],
            phase: nerve_execution_contracts::ExecutionPhase::Decode,
            activation_batch_width: 1,
            loaded_group_count: 1,
            loaded_resource_count: 6,
            loaded_byte_count: 4096,
            warmup_ns: 100,
            measured_ns: 90,
            output_digest: digest(7),
            state_digest: digest(8),
            resident_device_bytes: 4096,
            transient_peak_device_bytes: 8192,
            transient_host_bytes: 4096,
            execution_case: VulkanPlacementExecutionCaseIdentity {
                behavior,
                strategy: VulkanPlacementExecutionStrategy::LazyLoadWave,
                devices: vec![VulkanPlacementDeviceExecutionIdentity {
                    physical_device_id: physical_device_id.clone(),
                    api_version: 1,
                    driver_version: 2,
                }],
                shards: Vec::new(),
                input_physical_device_id: physical_device_id.clone(),
                output_physical_device_id: physical_device_id.clone(),
                owner_physical_device_id: physical_device_id,
                transports: Vec::new(),
            },
        }
    }

    #[test]
    fn load_wave_report_records_a_complete_resource_vector() {
        let report = report();
        let mut catalog = VulkanPlacementCalibrationCatalog::default();
        record_vulkan_runtime_load_wave_calibration_report(&mut catalog, &report).unwrap();
        let observation = report.calibration_observation();
        assert_eq!(
            catalog.exact_observation(&report.execution_case),
            Some(&observation),
        );
        assert_eq!(observation.warmup_call_count, 1);
        assert_eq!(observation.measured_call_count, 1);
        assert_eq!(observation.resident_bytes_by_physical_device[&report.physical_device_id], 4096);
        assert_eq!(observation.transient_peak_bytes_by_physical_device[&report.physical_device_id], 8192);
        assert_eq!(observation.host_transient_peak_bytes, 4096);
    }

    #[test]
    fn load_wave_catalog_recording_is_transactional() {
        let mut report = report();
        report.execution_case.behavior.shape.operations.clear();
        let mut catalog = VulkanPlacementCalibrationCatalog::default();
        let before = catalog.clone();
        assert!(record_vulkan_runtime_load_wave_calibration_report(&mut catalog, &report).is_err());
        assert_eq!(catalog, before);
    }

    #[test]
    fn load_wave_report_rejects_tensor_count_as_group_geometry() {
        let mut report = report();
        let VulkanPlacementOperationGeometry::LazyLoadWave { resource_count, .. } =
            &mut report.execution_case.behavior.shape.operations[0]
        else {
            unreachable!()
        };
        *resource_count = report.loaded_resource_count;
        let mut catalog = VulkanPlacementCalibrationCatalog::default();
        assert!(
            record_vulkan_runtime_load_wave_calibration_report(&mut catalog, &report)
                .unwrap_err()
                .to_string()
                .contains("inconsistent")
        );
        assert_eq!(catalog.observation_count(), 0);
    }
}
