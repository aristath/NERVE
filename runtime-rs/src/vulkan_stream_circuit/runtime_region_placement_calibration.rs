#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanRuntimeRegionPlacementCalibrationTarget {
    pub component_ids: Vec<String>,
    pub component_cases: Vec<VulkanPlacementExecutionCaseIdentity>,
    pub boundary_byte_counts: Vec<usize>,
    pub boundary_cases: Vec<VulkanPlacementRegionBoundaryExecutionCase>,
    pub execution_case: VulkanPlacementExecutionCaseIdentity,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VulkanRuntimeRegionPlacementCalibrationPlan {
    pub runtime_model: VulkanResidentRuntimeModel,
    pub physical_execution_plan: VulkanRuntimePhysicalExecutionPlan,
    pub target: VulkanRuntimeRegionPlacementCalibrationTarget,
}

pub fn plan_vulkan_runtime_region_placement_calibration(
    runtime_model: &VulkanResidentRuntimeModel,
    catalog: &VulkanPlacementCalibrationCatalog,
    capacity: &VulkanPlacementCapacityEnvelope,
    phase: VulkanTargetedComponentExecutionPhase,
    logical_device_id_by_physical_device: &BTreeMap<String, String>,
) -> Result<
    VulkanRuntimeRegionPlacementCalibrationPlan,
    VulkanRuntimeHybridPlacementError,
> {
    try_plan_vulkan_runtime_region_placement_calibration(
        runtime_model,
        catalog,
        capacity,
        phase,
        logical_device_id_by_physical_device,
    )?
    .ok_or_else(|| {
        VulkanRuntimeHybridPlacementError(
            "no exact measured runtime hybrid placement is available for region calibration"
                .to_string(),
        )
    })
}

pub fn try_plan_vulkan_runtime_region_placement_calibration(
    runtime_model: &VulkanResidentRuntimeModel,
    catalog: &VulkanPlacementCalibrationCatalog,
    capacity: &VulkanPlacementCapacityEnvelope,
    phase: VulkanTargetedComponentExecutionPhase,
    logical_device_id_by_physical_device: &BTreeMap<String, String>,
) -> Result<
    Option<VulkanRuntimeRegionPlacementCalibrationPlan>,
    VulkanRuntimeHybridPlacementError,
> {
    try_plan_vulkan_runtime_hybrid_ordered_graph(runtime_model, catalog, capacity, phase)?
        .map(|placement| {
            lower_vulkan_runtime_region_placement_calibration(
                runtime_model,
                placement,
                phase,
                logical_device_id_by_physical_device,
            )
        })
        .transpose()
}

pub fn plan_vulkan_runtime_serialized_region_placement_calibration(
    runtime_model: &VulkanResidentRuntimeModel,
    catalog: &VulkanPlacementCalibrationCatalog,
    capacity: &VulkanPlacementCapacityEnvelope,
    phase: VulkanTargetedComponentExecutionPhase,
    logical_device_id_by_physical_device: &BTreeMap<String, String>,
) -> Result<
    VulkanRuntimeRegionPlacementCalibrationPlan,
    VulkanRuntimeHybridPlacementError,
> {
    try_plan_vulkan_runtime_serialized_region_placement_calibration(
        runtime_model,
        catalog,
        capacity,
        phase,
        logical_device_id_by_physical_device,
    )?
    .ok_or_else(|| {
        VulkanRuntimeHybridPlacementError(
            "no exact measured serialized placement is available for region calibration"
                .to_string(),
        )
    })
}

pub fn try_plan_vulkan_runtime_serialized_region_placement_calibration(
    runtime_model: &VulkanResidentRuntimeModel,
    catalog: &VulkanPlacementCalibrationCatalog,
    capacity: &VulkanPlacementCapacityEnvelope,
    phase: VulkanTargetedComponentExecutionPhase,
    logical_device_id_by_physical_device: &BTreeMap<String, String>,
) -> Result<
    Option<VulkanRuntimeRegionPlacementCalibrationPlan>,
    VulkanRuntimeHybridPlacementError,
> {
    try_plan_vulkan_runtime_serialized_ordered_graph(runtime_model, catalog, capacity, phase)?
        .map(|placement| {
            lower_vulkan_runtime_region_placement_calibration(
                runtime_model,
                placement,
                phase,
                logical_device_id_by_physical_device,
            )
        })
        .transpose()
}

fn lower_vulkan_runtime_region_placement_calibration(
    runtime_model: &VulkanResidentRuntimeModel,
    placement: VulkanRuntimeHybridOrderedPlacement,
    phase: VulkanTargetedComponentExecutionPhase,
    logical_device_id_by_physical_device: &BTreeMap<String, String>,
) -> Result<
    VulkanRuntimeRegionPlacementCalibrationPlan,
    VulkanRuntimeHybridPlacementError,
> {
    let target = vulkan_runtime_region_placement_calibration_target_for_placement(
        runtime_model,
        &placement,
    )?;
    let lowered = lower_vulkan_runtime_hybrid_phase_placement(
        runtime_model,
        &placement,
        logical_device_id_by_physical_device,
    )?;
    let mut physical_execution_plan = VulkanRuntimePhysicalExecutionPlan::default();
    match phase {
        VulkanTargetedComponentExecutionPhase::Decode => {
            physical_execution_plan.component_device_pools.decode =
                lowered.component_device_pools;
            physical_execution_plan.decode_execution_cases_by_component =
                lowered.execution_cases_by_component;
            physical_execution_plan.decode_boundary_executions = lowered.boundary_executions;
        }
        VulkanTargetedComponentExecutionPhase::Prefill { .. } => {
            physical_execution_plan.component_device_pools.prefill =
                lowered.component_device_pools;
            physical_execution_plan.prefill_execution_cases_by_component =
                lowered.execution_cases_by_component;
            physical_execution_plan.prefill_boundary_executions = lowered.boundary_executions;
        }
    }
    physical_execution_plan.validate(&lowered.runtime_model)?;
    Ok(VulkanRuntimeRegionPlacementCalibrationPlan {
        runtime_model: lowered.runtime_model,
        physical_execution_plan,
        target,
    })
}

fn vulkan_runtime_region_placement_calibration_target_for_placement(
    runtime_model: &VulkanResidentRuntimeModel,
    placement: &VulkanRuntimeHybridOrderedPlacement,
) -> Result<
    VulkanRuntimeRegionPlacementCalibrationTarget,
    VulkanRuntimeHybridPlacementError,
> {
    if placement.component_ids.len() < 2 {
        return runtime_hybrid_error(
            "runtime region calibration requires at least two ordered components",
        );
    }
    let graph_boundaries = vulkan_runtime_placement_boundaries(runtime_model)
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    if graph_boundaries.len() + 1 != placement.component_ids.len() {
        return runtime_hybrid_error(
            "runtime region calibration requires one exact graph boundary between each ordered component",
        );
    }
    let boundary_byte_counts = graph_boundaries
        .iter()
        .map(|boundary| {
            let [transfer] = boundary.transfers.as_slice() else {
                return runtime_hybrid_error(
                    "runtime region calibration requires one transfer per ordered boundary",
                );
            };
            if !transfer.source_in_prefix {
                return runtime_hybrid_error(
                    "runtime region calibration cannot flatten a reverse graph boundary",
                );
            }
            transfer
                .byte_count
                .checked_mul(placement.activation_batch_width)
                .ok_or_else(|| {
                    VulkanRuntimeHybridPlacementError(
                        "runtime region boundary byte count overflowed".to_string(),
                    )
                })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut component_cases = Vec::with_capacity(placement.component_ids.len());
    let mut boundary_cases = BTreeMap::new();
    for step in &placement.plan.steps {
        match step {
            VulkanHybridScheduledStep::Region {
                component_start,
                component_end,
                execution_case,
                ..
            } => {
                if *component_start != component_cases.len() {
                    return runtime_hybrid_error(
                        "runtime region calibration placement is not contiguous",
                    );
                }
                let replay = runtime_hybrid_step_component_cases(
                    placement,
                    *component_start,
                    *component_end,
                    execution_case,
                )?;
                component_cases.extend(replay.into_iter().cloned());
                if component_end > &(component_start + 1) {
                    let calibration = placement
                        .region_executions_by_case
                        .get(execution_case)
                        .ok_or_else(|| {
                            VulkanRuntimeHybridPlacementError(
                                "runtime region calibration has no nested exact region replay"
                                    .to_string(),
                            )
                        })?;
                    for boundary in &calibration.boundary_cases {
                        let ordinal = component_start + boundary.boundary_ordinal;
                        if boundary_cases
                            .insert(ordinal, boundary.execution_case.clone())
                            .is_some()
                        {
                            return runtime_hybrid_error(
                                "runtime region calibration repeats a nested physical boundary",
                            );
                        }
                    }
                }
            }
            VulkanHybridScheduledStep::Boundary {
                boundary_index,
                execution_case,
            } => {
                if boundary_cases
                    .insert(*boundary_index, execution_case.clone())
                    .is_some()
                {
                    return runtime_hybrid_error(
                        "runtime region calibration repeats a physical boundary",
                    );
                }
            }
        }
    }
    if component_cases.len() != placement.component_ids.len() {
        return runtime_hybrid_error(
            "runtime region calibration placement does not cover the complete ordered graph",
        );
    }
    let boundary_cases = boundary_cases
        .into_iter()
        .map(
            |(boundary_ordinal, execution_case)| VulkanPlacementRegionBoundaryExecutionCase {
                boundary_ordinal,
                execution_case,
            },
        )
        .collect();
    vulkan_runtime_region_placement_calibration_target(
        placement.component_ids.clone(),
        component_cases,
        boundary_byte_counts,
        boundary_cases,
        runtime_region_output_scalar_format(runtime_model)?,
    )
    .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))
}

fn vulkan_runtime_region_placement_calibration_target(
    component_ids: Vec<String>,
    component_cases: Vec<VulkanPlacementExecutionCaseIdentity>,
    boundary_byte_counts: Vec<usize>,
    mut boundary_cases: Vec<VulkanPlacementRegionBoundaryExecutionCase>,
    output_scalar_format: VulkanPlacementScalarFormat,
) -> Result<
    VulkanRuntimeRegionPlacementCalibrationTarget,
    VulkanPlacementCalibrationCatalogError,
> {
    if component_ids.len() < 2
        || component_ids.len() != component_cases.len()
        || component_ids.iter().any(String::is_empty)
        || component_ids.iter().collect::<BTreeSet<_>>().len() != component_ids.len()
        || boundary_byte_counts.len() + 1 != component_cases.len()
        || boundary_byte_counts.contains(&0)
    {
        return Err(VulkanPlacementCalibrationCatalogError(
            "runtime region calibration requires at least two distinct ordered components and exact boundary geometry"
                .to_string(),
        ));
    }
    boundary_cases.sort_by_key(|boundary| boundary.boundary_ordinal);
    if boundary_cases.windows(2).any(|pair| {
        pair[0].boundary_ordinal >= pair[1].boundary_ordinal
    }) || boundary_cases
        .iter()
        .any(|boundary| boundary.boundary_ordinal >= component_cases.len() - 1)
    {
        return Err(VulkanPlacementCalibrationCatalogError(
            "runtime region calibration boundaries are duplicate or out of range".to_string(),
        ));
    }

    let first = component_cases.first().expect("component cases are nonempty");
    let last = component_cases.last().expect("component cases are nonempty");
    let phase = first.behavior.phase;
    let activation_batch_width = first.behavior.shape.activation_batch_width;
    let runtime_fingerprint = first.behavior.runtime_implementation_fingerprint.as_str();
    if component_cases.iter().any(|component| {
        component.behavior.phase != phase
            || component.behavior.shape.activation_batch_width != activation_batch_width
            || component.behavior.runtime_implementation_fingerprint != runtime_fingerprint
            || !matches!(
                component.strategy,
                VulkanPlacementExecutionStrategy::SingleDevice
                    | VulkanPlacementExecutionStrategy::TensorParallel
                    | VulkanPlacementExecutionStrategy::WholeExpertParallel
                    | VulkanPlacementExecutionStrategy::IntraExpertTensorParallel
                    | VulkanPlacementExecutionStrategy::Hybrid
            )
    }) {
        return Err(VulkanPlacementCalibrationCatalogError(
            "runtime region calibration components do not share one executable phase and shape"
                .to_string(),
        ));
    }
    if component_cases.iter().any(|component| {
        component.equivalence.state != VulkanPlacementEquivalenceKind::BitExact
    }) {
        return Err(VulkanPlacementCalibrationCatalogError(
            "runtime region calibration cannot validate tolerant state without a typed compiled state layout"
                .to_string(),
        ));
    }
    let equivalence = runtime_region_output_equivalence(
        &component_cases,
        output_scalar_format,
    )?;

    let boundary_by_ordinal = boundary_cases
        .iter()
        .map(|boundary| (boundary.boundary_ordinal, &boundary.execution_case))
        .collect::<BTreeMap<_, _>>();
    for boundary_ordinal in 0..component_cases.len() - 1 {
        let source = &component_cases[boundary_ordinal];
        let destination = &component_cases[boundary_ordinal + 1];
        let crosses_devices = source.output_physical_device_id
            != destination.input_physical_device_id;
        match (crosses_devices, boundary_by_ordinal.get(&boundary_ordinal)) {
            (false, None) => {}
            (true, Some(boundary))
                if boundary.strategy == VulkanPlacementExecutionStrategy::DirectedBoundary
                    && boundary.input_physical_device_id
                        == source.output_physical_device_id
                    && boundary.output_physical_device_id
                        == destination.input_physical_device_id
                    && boundary.behavior.phase == phase
                    && boundary.behavior.shape.activation_batch_width
                        == activation_batch_width
                    && boundary.behavior.shape.input_byte_capacity
                        == boundary_byte_counts[boundary_ordinal]
                    && boundary.behavior.shape.output_byte_capacity
                        == boundary_byte_counts[boundary_ordinal] => {}
            _ => {
                return Err(VulkanPlacementCalibrationCatalogError(format!(
                    "runtime region calibration does not exactly cover physical boundary {boundary_ordinal}",
                )));
            }
        }
    }

    let component_signatures = component_cases
        .iter()
        .map(|component| component.behavior.compiled_execution_signature.clone())
        .collect::<Vec<_>>();
    let compiled_execution_signature =
        vulkan_placement_region_compiled_execution_signature(
            &component_signatures,
            &boundary_byte_counts,
        )?;
    let shape = VulkanPlacementShapeClass {
        activation_batch_width,
        input_byte_capacity: first.behavior.shape.input_byte_capacity,
        output_byte_capacity: last.behavior.shape.output_byte_capacity,
    };
    let input_fixture_digest = distributed_calibration_fixture_identity(phase, &shape, 0)
        .map_err(|error| VulkanPlacementCalibrationCatalogError(error.to_string()))?;

    let mut nested_cases = Vec::with_capacity(component_cases.len() + boundary_cases.len());
    for (ordinal, component) in component_cases.iter().enumerate() {
        nested_cases.push(component);
        if let Some(boundary) = boundary_by_ordinal.get(&ordinal) {
            nested_cases.push(*boundary);
        }
    }
    let mut contract_implementations = BTreeMap::<String, String>::new();
    let mut operations = Vec::new();
    let mut transports = Vec::new();
    let mut devices = BTreeSet::new();
    for nested in &nested_cases {
        for (contract_id, implementation_digest) in nested
            .contract_ids
            .iter()
            .cloned()
            .zip(nested.implementation_digests.iter().cloned())
        {
            if contract_implementations
                .insert(contract_id, implementation_digest.clone())
                .is_some_and(|existing| existing != implementation_digest)
            {
                return Err(VulkanPlacementCalibrationCatalogError(
                    "runtime region calibration repeats a contract with conflicting implementations"
                        .to_string(),
                ));
            }
        }
        operations.extend(nested.operations.iter().cloned());
        transports.extend(nested.transports.iter().cloned());
        devices.extend(nested.devices.iter().cloned());
    }
    transports.sort();
    transports.dedup();
    let artifact_digest = runtime_region_digest(
        "nerve.vulkan_region_artifacts.v1",
        nested_cases.iter().map(|case| case.artifact_digest.as_str()),
    )?;
    let execution_graph_digest = runtime_region_digest(
        "nerve.vulkan_region_execution_graph.v1",
        component_ids
            .iter()
            .map(String::as_str)
            .chain(
                nested_cases
                    .iter()
                    .map(|case| case.execution_graph_digest.as_str()),
            ),
    )?;
    let strategy = if component_cases.iter().all(|component| {
        component.strategy == VulkanPlacementExecutionStrategy::SingleDevice
    }) {
        VulkanPlacementExecutionStrategy::SerializedRegion
    } else {
        VulkanPlacementExecutionStrategy::HybridRegion
    };
    let execution_case = VulkanPlacementExecutionCaseIdentity {
        behavior: VulkanPlacementBehaviorIdentity {
            compiled_execution_signature,
            runtime_implementation_fingerprint: runtime_fingerprint.to_string(),
            phase,
            shape,
            input_fixture_digest,
        },
        contract_ids: contract_implementations.keys().cloned().collect(),
        implementation_digests: contract_implementations.values().cloned().collect(),
        artifact_digest,
        execution_graph_digest,
        operations,
        equivalence,
        strategy,
        devices: devices.into_iter().collect(),
        shards: Vec::new(),
        input_physical_device_id: first.input_physical_device_id.clone(),
        output_physical_device_id: last.output_physical_device_id.clone(),
        owner_physical_device_id: first.owner_physical_device_id.clone(),
        transports,
    };
    Ok(VulkanRuntimeRegionPlacementCalibrationTarget {
        component_ids,
        component_cases,
        boundary_byte_counts,
        boundary_cases,
        execution_case,
    })
}

fn runtime_region_output_scalar_format(
    runtime_model: &VulkanResidentRuntimeModel,
) -> Result<VulkanPlacementScalarFormat, VulkanRuntimeHybridPlacementError> {
    match runtime_model.package.activation_element_bytes {
        Some(2) => Ok(VulkanPlacementScalarFormat::Bf16),
        Some(4) => Ok(VulkanPlacementScalarFormat::F32),
        _ => runtime_hybrid_error(
            "runtime region calibration requires a typed BF16 or F32 model output boundary",
        ),
    }
}

fn runtime_region_output_equivalence(
    component_cases: &[VulkanPlacementExecutionCaseIdentity],
    output_scalar_format: VulkanPlacementScalarFormat,
) -> Result<VulkanPlacementEquivalenceIdentity, VulkanPlacementCalibrationCatalogError> {
    let tolerant = component_cases
        .iter()
        .filter(|component| {
            component.equivalence.output
                == VulkanPlacementEquivalenceKind::AbsoluteRelativeTolerance
        })
        .collect::<Vec<_>>();
    if tolerant.is_empty() {
        return Ok(VulkanPlacementEquivalenceIdentity::bit_exact());
    }
    let absolute_tolerances = tolerant
        .iter()
        .map(|component| component.equivalence.absolute_tolerance())
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            VulkanPlacementCalibrationCatalogError(
                "runtime region tolerant output has incomplete absolute tolerance evidence"
                    .to_string(),
            )
        })?;
    let absolute_tolerance = absolute_tolerances
        .into_iter()
        .reduce(f64::min)
        .ok_or_else(|| {
            VulkanPlacementCalibrationCatalogError(
                "runtime region tolerant output has no absolute tolerance".to_string(),
            )
        })?;
    let relative_tolerances = tolerant
        .iter()
        .map(|component| component.equivalence.relative_tolerance())
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            VulkanPlacementCalibrationCatalogError(
                "runtime region tolerant output has incomplete relative tolerance evidence"
                    .to_string(),
            )
        })?;
    let relative_tolerance = relative_tolerances
        .into_iter()
        .reduce(f64::min)
        .ok_or_else(|| {
            VulkanPlacementCalibrationCatalogError(
                "runtime region tolerant output has no relative tolerance".to_string(),
            )
        })?;
    Ok(VulkanPlacementEquivalenceIdentity {
        output: VulkanPlacementEquivalenceKind::AbsoluteRelativeTolerance,
        state: VulkanPlacementEquivalenceKind::BitExact,
        absolute_tolerance_bits: Some(absolute_tolerance.to_bits()),
        relative_tolerance_bits: Some(relative_tolerance.to_bits()),
        output_scalar_format: Some(output_scalar_format),
    })
}

fn runtime_region_digest<'a>(
    domain: &str,
    values: impl IntoIterator<Item = &'a str>,
) -> Result<String, VulkanPlacementCalibrationCatalogError> {
    let values = values.into_iter().collect::<Vec<_>>();
    if domain.is_empty() || values.is_empty() || values.contains(&"") {
        return Err(VulkanPlacementCalibrationCatalogError(
            "runtime region digest requires a domain and nonempty values".to_string(),
        ));
    }
    let payload = serde_json::to_vec(&(domain, values)).map_err(|error| {
        VulkanPlacementCalibrationCatalogError(format!(
            "could not encode runtime region digest: {error}",
        ))
    })?;
    Ok(format!("sha256:{:x}", Sha256::digest(payload)))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanRuntimeRegionPlacementCalibrationReport {
    pub target: VulkanRuntimeRegionPlacementCalibrationTarget,
    pub warmup_execution_ns: u64,
    pub measured_execution_ns: u64,
    pub measured_ns_per_activation: u64,
    pub warmup_call_count: usize,
    pub measured_call_count: usize,
    pub useful_activation_count: usize,
    pub output_digest: String,
    pub captured_output_artifact: VulkanPlacementOutputArtifact,
    pub output_artifact: Option<VulkanPlacementOutputArtifact>,
    pub state_digest: String,
    pub resident_bytes_by_physical_device: BTreeMap<String, usize>,
    pub transient_peak_bytes_by_physical_device: BTreeMap<String, usize>,
    pub host_transient_peak_bytes: usize,
}

pub fn record_vulkan_runtime_region_placement_calibration_report(
    catalog: &mut VulkanPlacementCalibrationCatalog,
    report: &VulkanRuntimeRegionPlacementCalibrationReport,
) -> Result<(), VulkanPlacementCalibrationCatalogError> {
    let behavior = report.target.execution_case.behavior.clone();
    if catalog.canonical_reference(&behavior).is_none() {
        catalog.record_reference(VulkanPlacementCanonicalReference {
            behavior: behavior.clone(),
            output_digest: report.output_digest.clone(),
            output_artifact: Some(report.captured_output_artifact.clone()),
            state_digest: report.state_digest.clone(),
        })?;
    }
    let reference = catalog
        .canonical_reference(&behavior)
        .expect("region reference was present or inserted");
    let output_equivalence = validate_vulkan_placement_output_equivalence(
        &report.target.execution_case.equivalence,
        &reference.output_digest,
        reference.output_artifact.as_ref(),
        &report.output_digest,
        report.output_artifact.as_ref(),
    )?;
    catalog.record_observation(VulkanPlacementCalibrationObservation {
        execution_case: report.target.execution_case.clone(),
        warmup_call_count: report.warmup_call_count,
        measured_call_count: report.measured_call_count,
        complete_transaction: true,
        duration_ns: report.measured_execution_ns,
        useful_activation_count: report.useful_activation_count,
        output_digest: report.output_digest.clone(),
        output_artifact: report.output_artifact.clone(),
        output_equivalence,
        state_digest: report.state_digest.clone(),
        resident_bytes_by_physical_device: report
            .resident_bytes_by_physical_device
            .clone(),
        transient_peak_bytes_by_physical_device: report
            .transient_peak_bytes_by_physical_device
            .clone(),
        host_resident_bytes: 0,
        host_transient_peak_bytes: report.host_transient_peak_bytes,
    })?;
    catalog.record_region_execution(VulkanPlacementRegionExecutionCalibration {
        execution_case: report.target.execution_case.clone(),
        boundary_byte_counts: report.target.boundary_byte_counts.clone(),
        component_cases: report.target.component_cases.clone(),
        boundary_cases: report.target.boundary_cases.clone(),
    })
}

struct VulkanRuntimeRegionPlacementExecution {
    duration_ns: u64,
    output_digest: String,
    captured_output_artifact: VulkanPlacementOutputArtifact,
    output_artifact: Option<VulkanPlacementOutputArtifact>,
    state_digest: String,
}

/// Failure-path guard for a partially mounted calibration transaction. The
/// successful session owns explicit, error-reporting teardown; this guard only
/// exists so constructor failures cannot strand pool registrations or compiled
/// resource stores before a session value exists.
struct VulkanRuntimeRegionPlacementMountGuard<'a> {
    devices: &'a BTreeMap<String, Rc<VulkanComputeDevice>>,
    package: Option<Arc<VulkanResidentInProcessPlacedModelPackage>>,
    parameter_pool: Option<VulkanResidentBufferPool>,
    armed: bool,
}

impl Drop for VulkanRuntimeRegionPlacementMountGuard<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        if let Some(package) = self.package.take() {
            package.teardown_compiled_resources();
            drop(package);
        }
        if let Some(parameter_pool) = self.parameter_pool.take() {
            for device_id in self.devices.keys() {
                let _ = parameter_pool.release_device(device_id);
            }
        }
        for device in self.devices.values() {
            let _ = device.quiesce();
        }
    }
}

struct VulkanRuntimeRegionPlacementCalibrationSession {
    devices: BTreeMap<String, Rc<VulkanComputeDevice>>,
    package: Arc<VulkanResidentInProcessPlacedModelPackage>,
    processor: VulkanResidentInProcessPlacedStreamProcessor,
    prefill_runner: Option<VulkanResidentPlacedComponentBatchRunner>,
    pipeline: Vec<usize>,
    phase: VulkanTargetedComponentExecutionPhase,
    output_scalar_format: VulkanPlacementScalarFormat,
    target: VulkanRuntimeRegionPlacementCalibrationTarget,
    parameter_pool: VulkanResidentBufferPool,
    tracked_before_mount: BTreeMap<String, usize>,
    tracked_after_package_mount: BTreeMap<String, usize>,
    tracked_peak: BTreeMap<String, usize>,
}

impl VulkanRuntimeRegionPlacementCalibrationSession {
    fn mount(
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        manifest_dir: &Path,
        plan: &VulkanRuntimeRegionPlacementCalibrationPlan,
        catalog: &VulkanPlacementCalibrationCatalog,
        resource_residency_policy: ResourceResidencyPolicy,
    ) -> Result<Self, VulkanResidentTokenModelPackageError> {
        validate_runtime_region_bound_devices(devices, &plan.target.execution_case)?;
        let tracked_before_mount = runtime_region_tracked_bytes(devices)?;
        let output_scalar_format = runtime_region_output_scalar_format(&plan.runtime_model)
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let mut mount_guard = VulkanRuntimeRegionPlacementMountGuard {
            devices,
            package: None,
            parameter_pool: Some(VulkanResidentBufferPool::default()),
            armed: true,
        };
        let activation_batch_width = plan
            .target
            .execution_case
            .behavior
            .shape
            .activation_batch_width;
        let dynamic_state_capacity = VULKAN_RUNTIME_PLACEMENT_CALIBRATION_MAXIMUM_STATE_ACTIVATIONS
            .max(activation_batch_width)
            .min(plan.runtime_model.package.max_context_activations);
        if dynamic_state_capacity < activation_batch_width {
            return distributed_calibration_error(
                "runtime region calibration width exceeds the package context capacity",
            );
        }
        let package = Arc::new(
            VulkanResidentInProcessPlacedModelPackage::from_runtime_model_for_bound_devices_with_physical_execution_plan(
                devices,
                manifest_dir,
                plan.runtime_model.clone(),
                plan.physical_execution_plan.clone(),
                Some(catalog),
                Some(dynamic_state_capacity),
                0,
                resource_residency_policy,
                mount_guard
                    .parameter_pool
                    .as_ref()
                    .expect("mount guard owns its parameter pool"),
                None,
            )
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?,
        );
        mount_guard.package = Some(Arc::clone(&package));
        let tracked_after_package_mount = runtime_region_tracked_bytes(devices)?;
        let processor = package
            .create_stream_processor_for_bound_devices(devices, 0)
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let prefill_runner = (plan.target.execution_case.behavior.phase
            == nerve_execution_contracts::ExecutionPhase::Prefill)
            .then(|| {
                VulkanResidentPlacedComponentBatchRunner::new_for_components(
                    devices,
                    &processor.device_slices,
                    &package.runtime_execution_identity,
                    &processor.execution_quantum_calibrators,
                    activation_batch_width,
                    VulkanComponentBatchExecutionMode::CausalSequence,
                    &BTreeMap::new(),
                    true,
                    package.prefill_distributed_execution_plan(),
                    &package.distributed_parameter_buffers,
                    &package.distributed_dynamic_resource_buffers,
                    &package.compiled_resource_device_stores,
                    plan.target.component_ids.iter().cloned().collect(),
                )
                .map_err(|error| distributed_calibration_error_value(error.to_string()))
            })
            .transpose()?;
        let pipeline = processor
            .linear_pipeline_device_indices()
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let tracked_peak = runtime_region_tracked_bytes(devices)?;
        mount_guard.package = None;
        let parameter_pool = mount_guard
            .parameter_pool
            .take()
            .expect("successful mount transfers its parameter pool");
        mount_guard.armed = false;
        Ok(Self {
            devices: devices.clone(),
            package,
            processor,
            prefill_runner,
            pipeline,
            phase: match plan.target.execution_case.behavior.phase {
                nerve_execution_contracts::ExecutionPhase::Decode => {
                    VulkanTargetedComponentExecutionPhase::Decode
                }
                nerve_execution_contracts::ExecutionPhase::Prefill => {
                    VulkanTargetedComponentExecutionPhase::Prefill {
                        activation_batch_width,
                    }
                }
            },
            output_scalar_format,
            target: plan.target.clone(),
            parameter_pool,
            tracked_before_mount,
            tracked_after_package_mount,
            tracked_peak,
        })
    }

    fn execute_calls(
        &mut self,
        call_count: usize,
        maximum_duration: Duration,
    ) -> Result<VulkanRuntimeRegionPlacementExecution, VulkanResidentTokenModelPackageError> {
        if call_count == 0 || maximum_duration.is_zero() {
            return distributed_calibration_error(
                "runtime region calibration requires positive call and duration bounds",
            );
        }
        let started = Instant::now();
        let mut total_duration_ns = 0u64;
        let mut expected_output = None;
        let mut expected_state = None;
        let mut last = None;
        for _ in 0..call_count {
            if started.elapsed() >= maximum_duration {
                return distributed_calibration_error(
                    "runtime region calibration exceeded its configured duration",
                );
            }
            let execution = self.execute_once(0)?;
            if expected_output
                .as_ref()
                .is_some_and(|digest| digest != &execution.output_digest)
                || expected_state
                    .as_ref()
                    .is_some_and(|digest| digest != &execution.state_digest)
            {
                return distributed_calibration_error(
                    "runtime region calibration changed deterministic output or state between calls",
                );
            }
            expected_output.get_or_insert_with(|| execution.output_digest.clone());
            expected_state.get_or_insert_with(|| execution.state_digest.clone());
            total_duration_ns = total_duration_ns
                .checked_add(execution.duration_ns)
                .ok_or_else(|| {
                    distributed_calibration_error_value(
                        "runtime region measured duration overflowed",
                    )
                })?;
            last = Some(execution);
            runtime_region_update_peak(&mut self.tracked_peak, &self.devices)?;
        }
        if started.elapsed() > maximum_duration {
            return distributed_calibration_error(
                "runtime region calibration exceeded its configured duration",
            );
        }
        let mut last = last.expect("positive call count produced an execution");
        last.duration_ns = total_duration_ns.max(1);
        Ok(last)
    }

    fn execute_once(
        &self,
        seed: u32,
    ) -> Result<VulkanRuntimeRegionPlacementExecution, VulkanResidentTokenModelPackageError> {
        self.prepare_fixture(seed)?;
        let activation_batch_width = self.phase.activation_batch_width();
        let capacity = u32::try_from(self.package.dynamic_state_capacity_activations)
            .map_err(|_| {
                distributed_calibration_error_value(
                    "runtime region state capacity exceeds u32",
                )
            })?;
        let start_stream_tick = u64::from(capacity)
            .checked_sub(u64::try_from(activation_batch_width).unwrap_or(u64::MAX))
            .ok_or_else(|| {
                distributed_calibration_error_value(
                    "runtime region batch exceeds its mounted state capacity",
                )
            })?;
        let input_token_ids = vec![0u32; activation_batch_width];
        let started = Instant::now();
        if self.phase == VulkanTargetedComponentExecutionPhase::Decode {
            self.processor
                .run_stream_tick_on_bound_devices_in_process(
                    &self.devices,
                    start_stream_tick,
                )
                .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
            let duration_ns = elapsed_nanoseconds(started).max(1);
            let captured_output_artifact = self.output_artifact()?;
            let output_digest =
                vulkan_placement_output_artifact_digest(&captured_output_artifact)
                .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
            let output_artifact = (self.target.execution_case.equivalence.output
                == VulkanPlacementEquivalenceKind::AbsoluteRelativeTolerance)
                .then_some(captured_output_artifact.clone());
            return Ok(VulkanRuntimeRegionPlacementExecution {
                duration_ns,
                output_digest,
                captured_output_artifact,
                output_artifact,
                state_digest: self.state_digest()?,
            });
        }
        let runner = self.prefill_runner.as_ref().ok_or_else(|| {
            distributed_calibration_error_value(
                "runtime region prefill has no mounted component runner",
            )
        })?;
        for (pipeline_ordinal, device_index) in self.pipeline.iter().copied().enumerate() {
            let slice = &self.processor.device_slices[device_index];
            runner
                .run_causal_sequence(
                    &self.devices,
                    device_index,
                    &slice.device_id,
                    &slice.mounted,
                    &input_token_ids,
                    start_stream_tick,
                    capacity,
                    VulkanComponentBatchCompletionMode::Blocking,
                )
                .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
            if let Some(destination_index) = self.pipeline.get(pipeline_ordinal + 1).copied() {
                let [outgoing] = slice.mounted.edge_io.outgoing_buffers.as_slice() else {
                    return distributed_calibration_error(format!(
                        "runtime region device {:?} has {} outgoing edges; expected one",
                        slice.device_id,
                        slice.mounted.edge_io.outgoing_buffers.len(),
                    ));
                };
                runner
                    .transfer_edge(
                        device_index,
                        destination_index,
                        outgoing.endpoint.edge_index,
                    )
                    .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
            }
        }
        if !runner
            .commit_causal_state_prefix(activation_batch_width)
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?
        {
            return distributed_calibration_error(
                "runtime region calibration could not commit causal state",
            );
        }
        let duration_ns = elapsed_nanoseconds(started).max(1);
        let captured_output_artifact = self.output_artifact()?;
        let output_digest = vulkan_placement_output_artifact_digest(&captured_output_artifact)
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let output_artifact = (self.target.execution_case.equivalence.output
            == VulkanPlacementEquivalenceKind::AbsoluteRelativeTolerance)
            .then_some(captured_output_artifact.clone());
        Ok(VulkanRuntimeRegionPlacementExecution {
            duration_ns,
            output_digest,
            captured_output_artifact,
            output_artifact,
            state_digest: self.state_digest()?,
        })
    }

    fn prepare_fixture(
        &self,
        seed: u32,
    ) -> Result<(), VulkanResidentTokenModelPackageError> {
        for slice in &self.processor.device_slices {
            slice
                .mounted
                .buffers
                .zero_state_buffers()
                .and_then(|_| slice.mounted.buffers.apply_clone_state_policies())
                .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
            if matches!(
                self.phase,
                VulkanTargetedComponentExecutionPhase::Prefill { .. }
            ) {
                targeted_write_prefill_state_fixture(
                    &slice.mounted,
                    &slice.mounted_bound.dispatches,
                    seed,
                )?;
            }
        }
        if let Some(runner) = &self.prefill_runner {
            for slice in &runner.slices {
                for signal in &slice.signal_buffers {
                    signal
                        .buffer
                        .write_bytes(&vec![0; signal.buffer.byte_capacity()])
                        .map_err(|error| {
                            distributed_calibration_error_value(error.to_string())
                        })?;
                }
            }
        }
        let first_device_index = *self.pipeline.first().ok_or_else(|| {
            distributed_calibration_error_value("runtime region pipeline is empty")
        })?;
        let (input, frame_byte_capacity) = match &self.prefill_runner {
            Some(runner) => {
                let input_key = VulkanComponentBatchSignalKey::ModelInput(
                    self.package.input_transducer_spec.output_signal_id.clone(),
                );
                let input = runner
                    .slice(first_device_index)
                    .map_err(|error| distributed_calibration_error_value(error.to_string()))?
                    .signal_buffer(&input_key)
                    .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
                (Arc::clone(&input.buffer), input.frame_byte_capacity)
            }
            None => {
                let input = self.processor.device_slices[first_device_index]
                    .mounted
                    .boundary_io
                    .input_buffer(&self.package.input_transducer_spec.output_signal_id)
                    .ok_or_else(|| {
                        distributed_calibration_error_value(
                            "runtime region decode has no mounted model input buffer",
                        )
                    })?;
                (Arc::clone(&input.buffer), input.byte_capacity)
            }
        };
        let byte_count = frame_byte_capacity
            .checked_mul(self.phase.activation_batch_width())
            .ok_or_else(|| {
                distributed_calibration_error_value(
                    "runtime region input fixture byte count overflowed",
                )
            })?;
        input
            .write_bytes(&targeted_fixture_bytes(byte_count, seed, 0))
            .map_err(|error| distributed_calibration_error_value(error.to_string()))
    }

    fn output_artifact(
        &self,
    ) -> Result<VulkanPlacementOutputArtifact, VulkanResidentTokenModelPackageError> {
        let last_device_index = *self.pipeline.last().ok_or_else(|| {
            distributed_calibration_error_value("runtime region pipeline is empty")
        })?;
        let (output, frame_byte_capacity) = match &self.prefill_runner {
            Some(runner) => {
                let output_key = VulkanComponentBatchSignalKey::ModelOutput(
                    self.package.output_transducer_spec.input_signal_id.clone(),
                );
                let output = runner
                    .slice(last_device_index)
                    .map_err(|error| distributed_calibration_error_value(error.to_string()))?
                    .signal_buffer(&output_key)
                    .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
                (Arc::clone(&output.buffer), output.frame_byte_capacity)
            }
            None => {
                let output = self.processor.device_slices[last_device_index]
                    .mounted
                    .boundary_io
                    .output_buffer(&self.package.output_transducer_spec.input_signal_id)
                    .ok_or_else(|| {
                        distributed_calibration_error_value(
                            "runtime region decode has no mounted model output buffer",
                        )
                    })?;
                (Arc::clone(&output.buffer), output.byte_capacity)
            }
        };
        let byte_count = frame_byte_capacity
            .checked_mul(self.phase.activation_batch_width())
            .ok_or_else(|| {
                distributed_calibration_error_value(
                    "runtime region output artifact byte count overflowed",
                )
            })?;
        Ok(VulkanPlacementOutputArtifact {
            scalar_format: self
                .output_scalar_format,
            segments: vec![VulkanPlacementOutputSegment {
                binding: 0,
                name: self.package.output_transducer_spec.input_signal_id.clone(),
                bytes: output
                    .read_bytes(byte_count)
                    .map_err(|error| distributed_calibration_error_value(error.to_string()))?,
            }],
        })
    }

    fn state_digest(&self) -> Result<String, VulkanResidentTokenModelPackageError> {
        let mut digest = Sha256::new();
        let mut states = Vec::new();
        for slice in &self.processor.device_slices {
            for state in &slice.mounted.buffers.state_buffers {
                states.push((
                    state.component_id.clone(),
                    state.state_id.clone(),
                    state
                        .buffer
                        .read_bytes(state.buffer.byte_capacity())
                        .map_err(|error| {
                            distributed_calibration_error_value(error.to_string())
                        })?,
                ));
            }
        }
        states.sort_by(|left, right| {
            (&left.0, &left.1).cmp(&(&right.0, &right.1))
        });
        if states.windows(2).any(|pair| {
            (&pair[0].0, &pair[0].1) == (&pair[1].0, &pair[1].1)
        }) {
            return distributed_calibration_error(
                "runtime region state appears on more than one physical owner",
            );
        }
        for (component_id, state_id, bytes) in states {
            digest.update(component_id.as_bytes());
            digest.update(state_id.as_bytes());
            digest.update(bytes);
        }
        Ok(targeted_finalized_artifact_digest(
            digest.finalize().as_slice(),
        ))
    }

    fn memory_evidence(
        &self,
    ) -> Result<
        (
            BTreeMap<String, usize>,
            BTreeMap<String, usize>,
            usize,
        ),
        VulkanResidentTokenModelPackageError,
    > {
        let (resident, transient) = runtime_region_device_memory_evidence(
            &self.target.execution_case.devices,
            &self.tracked_before_mount,
            &self.tracked_after_package_mount,
            &self.tracked_peak,
        )?;
        let host_transient = self
            .package
            .physical_execution_residency_plan
            .total_stream_shared_host_bytes
            .checked_add(
                self.target
                    .boundary_cases
                    .iter()
                    .flat_map(|boundary| &boundary.execution_case.transports)
                    .filter(|transport| {
                        matches!(
                            transport.route.as_str(),
                            "device_local_staging" | "shared_host"
                        )
                    })
                    .try_fold(0usize, |total, transport| {
                        total.checked_add(transport.byte_capacity)
                    })
                    .ok_or_else(|| {
                        distributed_calibration_error_value(
                            "runtime region host transient accounting overflowed",
                        )
                    })?,
            )
            .ok_or_else(|| {
                distributed_calibration_error_value(
                    "runtime region host transient accounting overflowed",
                )
            })?;
        Ok((resident, transient, host_transient))
    }

    fn cleanup(self) -> Result<(), VulkanResidentTokenModelPackageError> {
        let Self {
            devices,
            package,
            processor,
            prefill_runner,
            pipeline: _,
            phase: _,
            output_scalar_format: _,
            target: _,
            parameter_pool,
            tracked_before_mount,
            tracked_after_package_mount: _,
            tracked_peak: _,
        } = self;
        let mut errors = Vec::new();
        drop(prefill_runner);
        drop(processor);
        let teardown = package.teardown_compiled_resources();
        if !teardown.complete {
            errors.push("compiled resource teardown was incomplete".to_string());
        }
        drop(package);
        for device_id in devices.keys() {
            if let Err(error) = parameter_pool.release_device(device_id) {
                errors.push(error.to_string());
            }
        }
        for device in devices.values() {
            if let Err(error) = device.quiesce() {
                errors.push(error.to_string());
            }
        }
        match runtime_region_tracked_bytes(&devices) {
            Ok(after) if after == tracked_before_mount => {}
            Ok(after) => errors.push(format!(
                "runtime region teardown did not restore tracked allocations: before={tracked_before_mount:?}, after={after:?}",
            )),
            Err(error) => errors.push(error.to_string()),
        }
        if errors.is_empty() {
            Ok(())
        } else {
            distributed_calibration_error(format!(
                "runtime region calibration cleanup failed: {}",
                errors.join("; "),
            ))
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn calibrate_vulkan_runtime_region_placement_with_policy(
    devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
    manifest_dir: impl AsRef<Path>,
    plan: &VulkanRuntimeRegionPlacementCalibrationPlan,
    catalog: &VulkanPlacementCalibrationCatalog,
    resource_residency_policy: ResourceResidencyPolicy,
    policy: VulkanRuntimePlacementCalibrationPolicy,
) -> Result<VulkanRuntimeRegionPlacementCalibrationReport, VulkanResidentTokenModelPackageError> {
    if policy.warmup_units == 0
        || policy.measured_units == 0
        || policy.warmup_units > 2
        || policy.measured_units > 2
        || policy.maximum_duration.is_zero()
        || policy.maximum_duration > VULKAN_RUNTIME_PLACEMENT_CALIBRATION_MAXIMUM_DURATION
        || policy.maximum_total_resident_parameter_bytes == 0
        || policy
            .maximum_resident_parameter_bytes_by_physical_device
            .iter()
            .any(|(physical_id, capacity)| physical_id.is_empty() || *capacity == 0)
    {
        return distributed_calibration_error(
            "runtime region calibration requires one or two warmup and measured calls, nonzero capacity, and at most one minute",
        );
    }
    validate_runtime_region_policy_capacity(&plan.target, catalog, &policy)?;
    let started = Instant::now();
    let mut session = VulkanRuntimeRegionPlacementCalibrationSession::mount(
        devices,
        manifest_dir.as_ref(),
        plan,
        catalog,
        resource_residency_policy,
    )?;
    let execution_result = (|| {
        let warmup = session.execute_calls(
            policy.warmup_units,
            remaining_calibration_duration(started, policy.maximum_duration)?,
        )?;
        let measured = session.execute_calls(
            policy.measured_units,
            remaining_calibration_duration(started, policy.maximum_duration)?,
        )?;
        if warmup.output_digest != measured.output_digest
            || warmup.state_digest != measured.state_digest
        {
            return distributed_calibration_error(
                "runtime region warmup and measured transactions changed output or state",
            );
        }
        let (resident, transient, host_transient) = session.memory_evidence()?;
        let useful_activation_count = policy
            .measured_units
            .checked_mul(plan.target.execution_case.behavior.shape.activation_batch_width)
            .ok_or_else(|| {
                distributed_calibration_error_value(
                    "runtime region useful activation count overflowed",
                )
            })?;
        let useful_activation_count_u64 = u64::try_from(useful_activation_count).map_err(|_| {
            distributed_calibration_error_value(
                "runtime region useful activation count exceeds u64",
            )
        })?;
        Ok(VulkanRuntimeRegionPlacementCalibrationReport {
            target: plan.target.clone(),
            warmup_execution_ns: warmup.duration_ns,
            measured_execution_ns: measured.duration_ns,
            measured_ns_per_activation: measured
                .duration_ns
                .saturating_add(useful_activation_count_u64 / 2)
                / useful_activation_count_u64,
            warmup_call_count: policy.warmup_units,
            measured_call_count: policy.measured_units,
            useful_activation_count,
            output_digest: measured.output_digest,
            captured_output_artifact: measured.captured_output_artifact,
            output_artifact: measured.output_artifact,
            state_digest: measured.state_digest,
            resident_bytes_by_physical_device: resident,
            transient_peak_bytes_by_physical_device: transient,
            host_transient_peak_bytes: host_transient,
        })
    })();
    let cleanup_result = session.cleanup();
    match (execution_result, cleanup_result) {
        (Ok(report), Ok(())) => Ok(report),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Err(execution), Err(cleanup)) => distributed_calibration_error(format!(
            "{execution}; cleanup also failed: {cleanup}",
        )),
    }
}

fn validate_runtime_region_bound_devices(
    devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
    execution_case: &VulkanPlacementExecutionCaseIdentity,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    let mut actual = BTreeMap::new();
    for device in devices.values() {
        let identity = VulkanPlacementDeviceExecutionIdentity {
            physical_device_id: device.physical_device_id().to_string(),
            api_version: device.api_version(),
            driver_version: device.driver_version(),
        };
        if actual
            .insert(identity.physical_device_id.clone(), identity)
            .is_some()
        {
            return distributed_calibration_error(
                "runtime region calibration requires one logical binding per physical device",
            );
        }
    }
    if execution_case.devices.iter().any(|expected| {
        actual.get(&expected.physical_device_id) != Some(expected)
    }) {
        return distributed_calibration_error(
            "runtime region calibration target was measured for different devices or drivers",
        );
    }
    Ok(())
}

fn validate_runtime_region_policy_capacity(
    target: &VulkanRuntimeRegionPlacementCalibrationTarget,
    catalog: &VulkanPlacementCalibrationCatalog,
    policy: &VulkanRuntimePlacementCalibrationPolicy,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    let mut required_by_device = BTreeMap::<String, usize>::new();
    for case in target.component_cases.iter().chain(
        target
            .boundary_cases
            .iter()
            .map(|boundary| &boundary.execution_case),
    ) {
        let observation = catalog.exact_observation(case).ok_or_else(|| {
            distributed_calibration_error_value(
                "runtime region calibration target has no exact nested observation",
            )
        })?;
        for device in &case.devices {
            let required = observation
                .resident_bytes_by_physical_device
                .get(&device.physical_device_id)
                .copied()
                .unwrap_or(0)
                .checked_add(
                    observation
                        .transient_peak_bytes_by_physical_device
                        .get(&device.physical_device_id)
                        .copied()
                        .unwrap_or(0),
                )
                .ok_or_else(|| {
                    distributed_calibration_error_value(
                        "runtime region nested capacity overflowed",
                    )
                })?;
            let total = required_by_device
                .entry(device.physical_device_id.clone())
                .or_default();
            *total = total.checked_add(required).ok_or_else(|| {
                distributed_calibration_error_value(
                    "runtime region aggregate capacity overflowed",
                )
            })?;
        }
    }
    let required_total = required_by_device.values().try_fold(0usize, |total, bytes| {
        total.checked_add(*bytes)
    }).ok_or_else(|| {
        distributed_calibration_error_value("runtime region total capacity overflowed")
    })?;
    if required_total > policy.maximum_total_resident_parameter_bytes {
        return distributed_calibration_error(
            "runtime region calibration exceeds its total resident capacity bound",
        );
    }
    for (device_id, required) in required_by_device {
        if required > policy.parameter_capacity_for_physical_device(&device_id)? {
            return distributed_calibration_error(format!(
                "runtime region calibration exceeds the capacity bound for {device_id:?}",
            ));
        }
    }
    Ok(())
}

fn runtime_region_tracked_bytes(
    devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
) -> Result<BTreeMap<String, usize>, VulkanResidentTokenModelPackageError> {
    let mut tracked = BTreeMap::new();
    for device in devices.values() {
        let physical_id = device.physical_device_id().to_string();
        let bytes = usize::try_from(
            device
                .device_local_memory_accounting()
                .map_err(|error| distributed_calibration_error_value(error.to_string()))?
                .tracked_allocation_bytes,
        )
        .map_err(|_| {
            distributed_calibration_error_value(
                "runtime region tracked allocation bytes exceed usize",
            )
        })?;
        if tracked.insert(physical_id, bytes).is_some() {
            return distributed_calibration_error(
                "runtime region allocation accounting repeats a physical device",
            );
        }
    }
    Ok(tracked)
}

fn runtime_region_device_memory_evidence(
    participants: &[VulkanPlacementDeviceExecutionIdentity],
    before: &BTreeMap<String, usize>,
    after_package_mount: &BTreeMap<String, usize>,
    peak: &BTreeMap<String, usize>,
) -> Result<
    (BTreeMap<String, usize>, BTreeMap<String, usize>),
    VulkanResidentTokenModelPackageError,
> {
    let mut resident = BTreeMap::new();
    let mut transient = BTreeMap::new();
    for participant in participants {
        let physical_id = &participant.physical_device_id;
        let before = before.get(physical_id).copied().ok_or_else(|| {
            distributed_calibration_error_value(
                "runtime region memory evidence has no pre-mount participant snapshot",
            )
        })?;
        let package = after_package_mount
            .get(physical_id)
            .copied()
            .ok_or_else(|| {
                distributed_calibration_error_value(
                    "runtime region memory evidence has no package participant snapshot",
                )
            })?;
        let peak = peak.get(physical_id).copied().ok_or_else(|| {
            distributed_calibration_error_value(
                "runtime region memory evidence has no peak participant snapshot",
            )
        })?;
        let resident_bytes = package.checked_sub(before).ok_or_else(|| {
            distributed_calibration_error_value(
                "runtime region persistent allocation accounting underflowed",
            )
        })?;
        let transient_bytes = peak.checked_sub(package).ok_or_else(|| {
            distributed_calibration_error_value(
                "runtime region transient allocation accounting underflowed",
            )
        })?;
        if resident
            .insert(physical_id.clone(), resident_bytes)
            .is_some()
            || transient
                .insert(physical_id.clone(), transient_bytes)
                .is_some()
        {
            return distributed_calibration_error(
                "runtime region memory evidence repeats a physical participant",
            );
        }
    }
    if resident.is_empty() {
        return distributed_calibration_error(
            "runtime region memory evidence requires a physical participant",
        );
    }
    Ok((resident, transient))
}

fn runtime_region_update_peak(
    peak: &mut BTreeMap<String, usize>,
    devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    for (physical_id, current) in runtime_region_tracked_bytes(devices)? {
        let recorded = peak.get_mut(&physical_id).ok_or_else(|| {
            distributed_calibration_error_value(
                "runtime region peak accounting found an unknown device",
            )
        })?;
        *recorded = (*recorded).max(current);
    }
    Ok(())
}

#[cfg(test)]
include!("tests/runtime_region_placement_calibration.rs");
