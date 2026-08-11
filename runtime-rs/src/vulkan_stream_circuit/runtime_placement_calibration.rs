const VULKAN_RUNTIME_PLACEMENT_CALIBRATION_WARMUP_UNITS: usize = 1;
const VULKAN_RUNTIME_PLACEMENT_CALIBRATION_MEASURED_UNITS: usize = 1;
const VULKAN_RUNTIME_PLACEMENT_CALIBRATION_MAXIMUM_STATE_ACTIVATIONS: usize = 128;
const VULKAN_RUNTIME_PLACEMENT_CALIBRATION_LOGICAL_DEVICE_ID: &str =
    "calibration:physical_candidate";
pub const VULKAN_RUNTIME_PLACEMENT_CALIBRATION_MAXIMUM_DURATION: Duration = Duration::from_secs(60);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanRuntimePlacementCalibrationTarget {
    pub signature_id: String,
    pub component_id: String,
    pub component_ids: Vec<String>,
    pub terminal_node_id: String,
    pub implementation: String,
    pub estimated_decode_work_units: u64,
    pub planned_resident_parameter_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VulkanRuntimePlacementCalibrationPolicy {
    pub warmup_units: usize,
    pub measured_units: usize,
    pub maximum_duration: Duration,
    pub maximum_resident_parameter_bytes: usize,
}

impl Default for VulkanRuntimePlacementCalibrationPolicy {
    fn default() -> Self {
        Self {
            warmup_units: VULKAN_RUNTIME_PLACEMENT_CALIBRATION_WARMUP_UNITS,
            measured_units: VULKAN_RUNTIME_PLACEMENT_CALIBRATION_MEASURED_UNITS,
            maximum_duration: VULKAN_RUNTIME_PLACEMENT_CALIBRATION_MAXIMUM_DURATION,
            maximum_resident_parameter_bytes: usize::MAX,
        }
    }
}

#[cfg(test)]
mod runtime_placement_calibration_policy_tests {
    use super::*;

    #[test]
    fn default_calibration_is_one_warmup_and_one_measured_call() {
        let policy = VulkanRuntimePlacementCalibrationPolicy::default();
        assert_eq!(policy.warmup_units, 1);
        assert_eq!(policy.measured_units, 1);
        assert!(policy.maximum_duration <= Duration::from_secs(60));
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanRuntimePlacementCalibrationReport {
    pub physical_device_id: String,
    pub target: VulkanRuntimePlacementCalibrationTarget,
    pub phase: String,
    pub activation_batch_width: usize,
    pub shared_prepare_ns: u64,
    pub slice_plan_prepare_ns: u64,
    pub slice_materialize_ns: u64,
    pub session_mount_ns: u64,
    pub warmup_execution_ns: u64,
    pub measured_execution_ns: u64,
    pub measured_ns_per_activation: u64,
    pub measured_windows: Vec<VulkanTargetedComponentThroughputWindow>,
    pub physical_dispatch_count: usize,
    pub output_digest: String,
    pub state_digest: String,
    pub resident_parameter_bytes: usize,
    pub resident_transient_bytes: usize,
}

struct VulkanRuntimePlacementCalibrationSource {
    runtime_model: VulkanResidentRuntimeModel,
    residency_plan: VulkanRuntimeResidencyPlan,
}

#[derive(Clone)]
struct VulkanRuntimePlacementCalibrationCachedPlan {
    plan: VulkanResidentTargetedModelPackageDeviceSlicePlan,
}

pub struct VulkanRuntimePlacementCalibrationSuite {
    targets: Vec<VulkanRuntimePlacementCalibrationTarget>,
    tensor_index: Arc<TensorIndex>,
    contract: Arc<CompiledResourceResidencyContract>,
    sources: Vec<VulkanRuntimePlacementCalibrationSource>,
    dynamic_state_capacity_activations: usize,
    plans_by_capability_class: BTreeMap<String, Vec<VulkanRuntimePlacementCalibrationCachedPlan>>,
    shared_prepare_ns: u64,
    shared_prepare_reported: bool,
}

impl VulkanRuntimePlacementCalibrationSuite {
    pub fn prepare(
        manifest_dir: impl AsRef<Path>,
        runtime_model: &VulkanResidentRuntimeModel,
        context_capacity_activations: usize,
    ) -> Result<Self, VulkanResidentTokenModelPackageError> {
        let started = Instant::now();
        let manifest_dir = manifest_dir.as_ref();
        let mut targets = vulkan_runtime_placement_calibration_targets(runtime_model)
            .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;
        let tensor_index = Arc::new(runtime_model.load_runtime_tensor_index(manifest_dir)?);
        let contract = Arc::new(
            instantiate_runtime_resource_contract(runtime_model).map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to instantiate runtime placement calibration contract: {error}",
                ))
            })?,
        );
        let capacity = context_capacity_activations
            .max(1)
            .min(VULKAN_RUNTIME_PLACEMENT_CALIBRATION_MAXIMUM_STATE_ACTIVATIONS);
        let mut sources = Vec::with_capacity(targets.len());
        for target in &mut targets {
            let calibration_model = vulkan_runtime_model_with_component_placement(
                runtime_model,
                "calibration:unmounted",
                &BTreeMap::from([(
                    target.component_id.clone(),
                    VULKAN_RUNTIME_PLACEMENT_CALIBRATION_LOGICAL_DEVICE_ID.to_string(),
                )]),
            )
            .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;
            let residency_plan = plan_vulkan_runtime_residency_with_contract(
                manifest_dir,
                &calibration_model,
                &tensor_index,
                capacity,
                0,
                ResourceResidencyPolicy::DemandRetained,
                &contract,
            )
            .map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to plan runtime placement calibration residency: {error}",
                ))
            })?;
            target.planned_resident_parameter_bytes = residency_plan
                .device_plans
                .iter()
                .try_fold(0usize, |total, plan| {
                    total
                        .checked_add(plan.parameter_residency.current_resident_bytes)
                        .and_then(|total| {
                            total.checked_add(plan.resource_store.maximum_load_wave_payload_bytes)
                        })
                })
                .ok_or_else(|| {
                    VulkanResidentTokenModelPackageError::new(
                        "runtime placement calibration transaction bytes overflowed",
                    )
                })?;
            sources.push(VulkanRuntimePlacementCalibrationSource {
                runtime_model: calibration_model,
                residency_plan,
            });
        }
        Ok(Self {
            targets,
            tensor_index,
            contract,
            sources,
            dynamic_state_capacity_activations: capacity,
            plans_by_capability_class: BTreeMap::new(),
            shared_prepare_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            shared_prepare_reported: false,
        })
    }

    pub fn targets(&self) -> &[VulkanRuntimePlacementCalibrationTarget] {
        &self.targets
    }

    fn plans_for_device(
        &mut self,
        device: &VulkanComputeDevice,
        capability_class: &str,
        manifest_dir: &Path,
    ) -> Result<
        (Vec<VulkanRuntimePlacementCalibrationCachedPlan>, u64, u64),
        VulkanResidentTokenModelPackageError,
    > {
        if capability_class.is_empty() {
            return Err(VulkanResidentTokenModelPackageError::new(
                "runtime placement calibration requires a hardware capability class",
            ));
        }
        let shared_prepare_ns = if self.shared_prepare_reported {
            0
        } else {
            self.shared_prepare_reported = true;
            self.shared_prepare_ns
        };
        if let Some(plans) = self.plans_by_capability_class.get(capability_class) {
            return Ok((plans.clone(), shared_prepare_ns, 0));
        }
        let started = Instant::now();
        let plans = self
            .targets
            .iter()
            .zip(&self.sources)
            .map(|(target, source)| {
                VulkanResidentTargetedModelPackageDeviceSlicePlan::prepare(
                    device,
                    manifest_dir,
                    &source.runtime_model,
                    &target.component_id,
                    VULKAN_RUNTIME_PLACEMENT_CALIBRATION_LOGICAL_DEVICE_ID,
                    self.dynamic_state_capacity_activations,
                    Arc::clone(&self.tensor_index),
                    Arc::clone(&self.contract),
                    source.residency_plan.clone(),
                )
                .map(|plan| VulkanRuntimePlacementCalibrationCachedPlan { plan })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let prepare_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.plans_by_capability_class
            .insert(capability_class.to_string(), plans.clone());
        Ok((plans, shared_prepare_ns, prepare_ns))
    }
}

/// Discovers the distinct complete decode transactions in graph order. The
/// signature deliberately excludes component and semantic labels while
/// retaining the physical dispatch contract: implementation, shader,
/// operation, execution domain, launch geometry, batching, stream controls,
/// and selected resource representation. Identical compiled transactions are
/// calibrated once and their result is applied to every matching component.
pub fn vulkan_runtime_placement_calibration_targets(
    runtime_model: &VulkanResidentRuntimeModel,
) -> Result<Vec<VulkanRuntimePlacementCalibrationTarget>, VulkanRuntimeResidencyPlanError> {
    let signal_component_ids = runtime_model
        .circuit_graph
        .components
        .iter()
        .filter(|component| component.runtime_role.is_signal_processor())
        .map(|component| component.component_id.clone())
        .collect::<Vec<_>>();
    if signal_component_ids.is_empty() {
        return Err(VulkanRuntimeResidencyPlanError(
            "runtime placement calibration found no signal processor".to_string(),
        ));
    }
    let executions = runtime_model
        .component_executions
        .iter()
        .map(|execution| (execution.component_id.as_str(), execution))
        .collect::<BTreeMap<_, _>>();
    let mut signature_target_indices = BTreeMap::<String, usize>::new();
    let mut targets = Vec::<VulkanRuntimePlacementCalibrationTarget>::new();

    for component_id in signal_component_ids {
        let execution = executions.get(component_id.as_str()).ok_or_else(|| {
            VulkanRuntimeResidencyPlanError(format!(
                "runtime placement calibration found no execution for signal processor {component_id:?}",
            ))
        })?;
        let mut decode_kernels = execution
            .kernels
            .iter()
            .filter(|kernel| kernel.execution_domain.supports_decode())
            .collect::<Vec<_>>();
        decode_kernels.sort_by_key(|kernel| kernel.execution_index);
        let terminal = decode_kernels.last().ok_or_else(|| {
            VulkanRuntimeResidencyPlanError(format!(
                "runtime placement calibration found no decode kernel for signal processor {component_id:?}",
            ))
        })?;
        let signature_payload = serde_json::to_vec(&(
            execution.operator_type.as_str(),
            execution.implementation.as_str(),
            decode_kernels
                .iter()
                .map(|kernel| {
                    (
                        kernel.execution_index,
                        kernel.op.as_str(),
                        &kernel.execution_domain,
                        kernel.stream_control_binding,
                        kernel.shader_path.as_str(),
                        kernel.local_size_x,
                        kernel.workgroup_count_x,
                        &kernel.batch_mode,
                        &kernel.batch_implementations,
                        &kernel.resource_representation_dispatch,
                    )
                })
                .collect::<Vec<_>>(),
        ))
        .map_err(|error| {
            VulkanRuntimeResidencyPlanError(format!(
                "runtime placement calibration could not encode execution signature for {component_id:?}: {error}",
            ))
        })?;
        let signature_id = format!("{:x}", Sha256::digest(&signature_payload));
        if let Some(target_index) = signature_target_indices.get(&signature_id).copied() {
            targets[target_index].component_ids.push(component_id);
            continue;
        }
        let estimated_decode_work_units = decode_kernels.iter().fold(0u64, |total, kernel| {
            total.saturating_add(
                u64::from(kernel.local_size_x).saturating_mul(u64::from(kernel.workgroup_count_x)),
            )
        });
        signature_target_indices.insert(signature_id.clone(), targets.len());
        targets.push(VulkanRuntimePlacementCalibrationTarget {
            signature_id,
            component_id: component_id.clone(),
            component_ids: vec![component_id],
            terminal_node_id: terminal.node_id.clone(),
            implementation: execution.implementation.clone(),
            estimated_decode_work_units,
            planned_resident_parameter_bytes: 0,
        });
    }
    Ok(targets)
}

/// Measures every distinct compiled decode transaction on one physical device
/// while retaining a single device context and parameter pool. The entire
/// candidate probe is bounded; cleanup is verified against NERVE's own tracked
/// allocations rather than transient driver pipeline caches.
pub fn calibrate_vulkan_runtime_placement_candidate(
    device: Rc<VulkanComputeDevice>,
    manifest_dir: impl AsRef<Path>,
    capability_class: &str,
    suite: &mut VulkanRuntimePlacementCalibrationSuite,
) -> Result<Vec<VulkanRuntimePlacementCalibrationReport>, VulkanResidentTokenModelPackageError> {
    calibrate_vulkan_runtime_placement_candidate_with_policy(
        device,
        manifest_dir,
        capability_class,
        suite,
        VulkanRuntimePlacementCalibrationPolicy::default(),
    )
}

pub fn calibrate_vulkan_runtime_placement_candidate_with_policy(
    device: Rc<VulkanComputeDevice>,
    manifest_dir: impl AsRef<Path>,
    capability_class: &str,
    suite: &mut VulkanRuntimePlacementCalibrationSuite,
    policy: VulkanRuntimePlacementCalibrationPolicy,
) -> Result<Vec<VulkanRuntimePlacementCalibrationReport>, VulkanResidentTokenModelPackageError> {
    calibrate_vulkan_runtime_placement_phase_candidate_with_policy(
        device,
        manifest_dir.as_ref(),
        capability_class,
        suite,
        VulkanTargetedComponentExecutionPhase::Decode,
        VulkanTargetedComponentExecutionScope::DecodeComponentPrefix,
        policy,
    )
}

pub fn calibrate_vulkan_runtime_prefill_placement_candidate_with_policy(
    device: Rc<VulkanComputeDevice>,
    manifest_dir: impl AsRef<Path>,
    capability_class: &str,
    suite: &mut VulkanRuntimePlacementCalibrationSuite,
    activation_batch_width: usize,
    policy: VulkanRuntimePlacementCalibrationPolicy,
) -> Result<Vec<VulkanRuntimePlacementCalibrationReport>, VulkanResidentTokenModelPackageError> {
    if activation_batch_width == 0 {
        return Err(VulkanResidentTokenModelPackageError::new(
            "runtime prefill placement calibration requires a positive activation batch width",
        ));
    }
    calibrate_vulkan_runtime_placement_phase_candidate_with_policy(
        device,
        manifest_dir.as_ref(),
        capability_class,
        suite,
        VulkanTargetedComponentExecutionPhase::Prefill {
            activation_batch_width,
        },
        VulkanTargetedComponentExecutionScope::Component,
        policy,
    )
}

fn calibrate_vulkan_runtime_placement_phase_candidate_with_policy(
    device: Rc<VulkanComputeDevice>,
    manifest_dir: &Path,
    capability_class: &str,
    suite: &mut VulkanRuntimePlacementCalibrationSuite,
    phase: VulkanTargetedComponentExecutionPhase,
    scope: VulkanTargetedComponentExecutionScope,
    policy: VulkanRuntimePlacementCalibrationPolicy,
) -> Result<Vec<VulkanRuntimePlacementCalibrationReport>, VulkanResidentTokenModelPackageError> {
    if policy.warmup_units == 0
        || policy.measured_units == 0
        || policy.maximum_duration.is_zero()
        || policy.maximum_resident_parameter_bytes == 0
    {
        return Err(VulkanResidentTokenModelPackageError::new(
            "runtime placement calibration policy has invalid zero bounds",
        ));
    }
    if suite.targets.is_empty() {
        return Err(VulkanResidentTokenModelPackageError::new(
            "runtime placement calibration requires at least one execution signature",
        ));
    }
    let (plans, shared_prepare_ns, slice_plan_prepare_ns) =
        suite.plans_for_device(&device, capability_class, manifest_dir)?;
    let targets = suite.targets.clone();
    let physical_device_id = device.physical_device_id().to_string();
    let logical_device_id = VULKAN_RUNTIME_PLACEMENT_CALIBRATION_LOGICAL_DEVICE_ID.to_string();
    let parameter_pool = VulkanResidentBufferPool::default();
    parameter_pool
        .register_device(&logical_device_id, Rc::clone(&device))
        .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;

    let execution_result = (|| {
        let mut reports = Vec::with_capacity(targets.len());
        for (target_index, (target, cached_plan)) in targets.iter().zip(&plans).enumerate() {
            if target.planned_resident_parameter_bytes
                > policy.maximum_resident_parameter_bytes
            {
                continue;
            }
            let case_started = Instant::now();
            let remaining = policy
                .maximum_duration
                .checked_sub(case_started.elapsed())
                .ok_or_else(|| {
                    VulkanResidentTokenModelPackageError::new(
                        "runtime placement calibration exceeded its configured duration",
                    )
                })?;
            let slice_materialize_started = Instant::now();
            let slice = cached_plan
                .plan
                .materialize(&device, manifest_dir, &parameter_pool)?;
            let slice_materialize_ns =
                u64::try_from(slice_materialize_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            let session_mount_started = Instant::now();
            let session = VulkanResidentTargetedExecutionSession::from_targeted_device_slice(
                &device,
                slice,
                &target.component_id,
                &target.terminal_node_id,
                phase,
                scope,
                false,
            )?;
            let session_mount_ns =
                u64::try_from(session_mount_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            let activation_batch_width = phase.activation_batch_width();
            let warmup_useful_units = policy
                .warmup_units
                .checked_mul(activation_batch_width)
                .ok_or_else(|| VulkanResidentTokenModelPackageError::new(
                    "runtime placement calibration warmup work overflowed",
                ))?;
            let measured_useful_units = policy
                .measured_units
                .checked_mul(activation_batch_width)
                .ok_or_else(|| VulkanResidentTokenModelPackageError::new(
                    "runtime placement calibration measured work overflowed",
                ))?;
            let warmup = session.execute(
                &device,
                warmup_useful_units,
                1,
                0,
                remaining,
            )?;
            let remaining = policy
                .maximum_duration
                .checked_sub(case_started.elapsed())
                .ok_or_else(|| {
                    VulkanResidentTokenModelPackageError::new(
                        "runtime placement calibration exceeded its configured duration",
                    )
                })?;
            let measured = session.execute(
                &device,
                measured_useful_units,
                policy.measured_units,
                0,
                remaining,
            )?;
            if warmup_useful_units == measured_useful_units
                && (warmup.output_digest != measured.output_digest
                    || warmup.state_digest != measured.state_digest)
            {
                return targeted_component_error(
                    "runtime placement calibration changed deterministic component output or state",
                );
            }
            reports.push(VulkanRuntimePlacementCalibrationReport {
                physical_device_id: physical_device_id.clone(),
                target: target.clone(),
                phase: measured.phase,
                activation_batch_width: measured.activation_batch_width,
                shared_prepare_ns: (target_index == 0)
                    .then_some(shared_prepare_ns)
                    .unwrap_or(0),
                slice_plan_prepare_ns: (target_index == 0)
                    .then_some(slice_plan_prepare_ns)
                    .unwrap_or(0),
                slice_materialize_ns,
                session_mount_ns,
                warmup_execution_ns: warmup.execution_ns,
                measured_execution_ns: measured.execution_ns,
                measured_ns_per_activation: measured.execution_ns.saturating_add(
                    (measured_useful_units / 2) as u64,
                ) / measured_useful_units as u64,
                measured_windows: measured.throughput_windows,
                physical_dispatch_count: measured.physical_dispatch_count,
                output_digest: measured.output_digest,
                state_digest: measured.state_digest,
                resident_parameter_bytes: measured.resident_parameter_bytes,
                resident_transient_bytes: measured.resident_transient_bytes,
            });
        }
        Ok(reports)
    })();

    let mut cleanup_errors = [
        device.quiesce().err().map(|error| error.to_string()),
        parameter_pool
            .release_device(&logical_device_id)
            .err()
            .map(|error| error.to_string()),
        device.quiesce().err().map(|error| error.to_string()),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
    match device.device_local_memory_accounting() {
        Ok(accounting)
            if accounting.tracked_allocation_bytes == 0
                && accounting.pending_reservation_bytes == 0 => {}
        Ok(accounting) => cleanup_errors.push(format!(
            "device accounting retained {} tracked and {} pending bytes",
            accounting.tracked_allocation_bytes, accounting.pending_reservation_bytes,
        )),
        Err(error) => cleanup_errors.push(error.to_string()),
    }
    let pool_stats = parameter_pool.stats();
    if parameter_pool.registered_device_count() != 0
        || pool_stats.resident_allocation_count != 0
        || pool_stats.resident_buffer_count != 0
        || pool_stats.resident_bytes != 0
    {
        cleanup_errors.push(format!(
            "resident parameter pool retained {} devices, {} allocations, {} buffers, and {} bytes",
            parameter_pool.registered_device_count(),
            pool_stats.resident_allocation_count,
            pool_stats.resident_buffer_count,
            pool_stats.resident_bytes,
        ));
    }
    match (execution_result, cleanup_errors.is_empty()) {
        (Ok(reports), true) => Ok(reports),
        (Ok(_), false) => Err(VulkanResidentTokenModelPackageError::new(format!(
            "runtime placement calibration cleanup failed: {}",
            cleanup_errors.join("; "),
        ))),
        (Err(error), true) => Err(error),
        (Err(error), false) => Err(VulkanResidentTokenModelPackageError::new(format!(
            "{error}; runtime placement calibration cleanup also failed: {}",
            cleanup_errors.join("; "),
        ))),
    }
}
