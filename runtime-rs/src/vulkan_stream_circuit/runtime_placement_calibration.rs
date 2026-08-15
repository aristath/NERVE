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
    pub planned_resident_parameter_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanRuntimePlacementCalibrationPolicy {
    pub warmup_units: usize,
    pub measured_units: usize,
    pub maximum_duration: Duration,
    pub maximum_total_resident_parameter_bytes: usize,
    pub maximum_resident_parameter_bytes_by_physical_device: BTreeMap<String, usize>,
}

impl Default for VulkanRuntimePlacementCalibrationPolicy {
    fn default() -> Self {
        Self {
            warmup_units: VULKAN_RUNTIME_PLACEMENT_CALIBRATION_WARMUP_UNITS,
            measured_units: VULKAN_RUNTIME_PLACEMENT_CALIBRATION_MEASURED_UNITS,
            maximum_duration: VULKAN_RUNTIME_PLACEMENT_CALIBRATION_MAXIMUM_DURATION,
            maximum_total_resident_parameter_bytes: usize::MAX,
            maximum_resident_parameter_bytes_by_physical_device: BTreeMap::new(),
        }
    }
}

impl VulkanRuntimePlacementCalibrationPolicy {
    fn parameter_capacity_for_physical_device(
        &self,
        physical_device_id: &str,
    ) -> Result<usize, VulkanResidentTokenModelPackageError> {
        if self
            .maximum_resident_parameter_bytes_by_physical_device
            .is_empty()
        {
            return Ok(self.maximum_total_resident_parameter_bytes);
        }
        self.maximum_resident_parameter_bytes_by_physical_device
            .get(physical_device_id)
            .copied()
            .filter(|capacity| *capacity > 0)
            .ok_or_else(|| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "runtime placement calibration has no positive parameter capacity for physical device {physical_device_id:?}",
                ))
            })
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
        assert_eq!(policy.maximum_total_resident_parameter_bytes, usize::MAX);
        assert!(
            policy
                .maximum_resident_parameter_bytes_by_physical_device
                .is_empty()
        );
    }

    #[test]
    fn calibration_capacity_is_bound_to_exact_physical_devices() {
        let policy = VulkanRuntimePlacementCalibrationPolicy {
            maximum_total_resident_parameter_bytes: 300,
            maximum_resident_parameter_bytes_by_physical_device: BTreeMap::from([
                ("gpu-a".to_string(), 100),
                ("gpu-b".to_string(), 200),
            ]),
            ..VulkanRuntimePlacementCalibrationPolicy::default()
        };

        assert_eq!(
            policy
                .parameter_capacity_for_physical_device("gpu-a")
                .unwrap(),
            100,
        );
        assert!(
            policy
                .parameter_capacity_for_physical_device("gpu-c")
                .unwrap_err()
                .to_string()
                .contains("no positive parameter capacity")
        );
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
    pub captured_outputs: Vec<VulkanTargetedCapturedOutput>,
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
    plans_by_capability_and_targets:
        BTreeMap<(String, Vec<usize>), Vec<VulkanRuntimePlacementCalibrationCachedPlan>>,
    shared_prepare_ns: u64,
    shared_prepare_reported: bool,
}

impl VulkanRuntimePlacementCalibrationSuite {
    pub fn prepare(
        manifest_dir: impl AsRef<Path>,
        runtime_model: &VulkanResidentRuntimeModel,
        context_capacity_activations: usize,
    ) -> Result<Self, VulkanResidentTokenModelPackageError> {
        let targets = vulkan_runtime_placement_calibration_targets(runtime_model)
            .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;
        Self::prepare_targets(
            manifest_dir.as_ref(),
            runtime_model,
            context_capacity_activations,
            targets,
        )
    }

    fn prepare_target(
        manifest_dir: &Path,
        runtime_model: &VulkanResidentRuntimeModel,
        context_capacity_activations: usize,
        target: VulkanRuntimePlacementCalibrationTarget,
    ) -> Result<Self, VulkanResidentTokenModelPackageError> {
        Self::prepare_targets(
            manifest_dir,
            runtime_model,
            context_capacity_activations,
            vec![target],
        )
    }

    fn prepare_targets(
        manifest_dir: &Path,
        runtime_model: &VulkanResidentRuntimeModel,
        context_capacity_activations: usize,
        mut targets: Vec<VulkanRuntimePlacementCalibrationTarget>,
    ) -> Result<Self, VulkanResidentTokenModelPackageError> {
        let started = Instant::now();
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
            plans_by_capability_and_targets: BTreeMap::new(),
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
        target_indices: &[usize],
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
        if target_indices.is_empty()
            || target_indices.windows(2).any(|pair| pair[0] >= pair[1])
            || target_indices
                .last()
                .is_some_and(|index| *index >= self.targets.len())
        {
            return Err(VulkanResidentTokenModelPackageError::new(
                "runtime placement calibration target subset is empty, unordered, or out of range",
            ));
        }
        let cache_key = (capability_class.to_string(), target_indices.to_vec());
        if let Some(plans) = self.plans_by_capability_and_targets.get(&cache_key) {
            return Ok((plans.clone(), shared_prepare_ns, 0));
        }
        let started = Instant::now();
        let plans = target_indices
            .iter()
            .map(|target_index| {
                let target = &self.targets[*target_index];
                let source = &self.sources[*target_index];
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
        self.plans_by_capability_and_targets
            .insert(cache_key, plans.clone());
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
    vulkan_runtime_placement_calibration_targets_for_phase(
        runtime_model,
        VulkanTargetedComponentExecutionPhase::Decode,
    )
}

/// Resolves the exact compiler-emitted execution for one component and phase.
/// This deliberately does not substitute a representative from a decode-only
/// equivalence class: two components may share decode kernels while using
/// different prefill artifacts or terminal dispatches.
pub fn vulkan_runtime_placement_calibration_target_for_component(
    runtime_model: &VulkanResidentRuntimeModel,
    component_id: &str,
    phase: VulkanTargetedComponentExecutionPhase,
) -> Result<VulkanRuntimePlacementCalibrationTarget, VulkanRuntimeResidencyPlanError> {
    if component_id.is_empty() {
        return Err(VulkanRuntimeResidencyPlanError(
            "runtime placement calibration requires a component ID".to_string(),
        ));
    }
    if matches!(
        phase,
        VulkanTargetedComponentExecutionPhase::Prefill {
            activation_batch_width: 0
        }
    ) {
        return Err(VulkanRuntimeResidencyPlanError(
            "runtime placement calibration requires a positive prefill batch width".to_string(),
        ));
    }
    if !runtime_model
        .circuit_graph
        .components
        .iter()
        .any(|component| {
            component.component_id == component_id && component.runtime_role.is_signal_processor()
        })
    {
        return Err(VulkanRuntimeResidencyPlanError(format!(
            "runtime placement calibration found no signal processor {component_id:?}",
        )));
    }
    let execution = runtime_model
        .component_executions
        .iter()
        .find(|execution| execution.component_id == component_id)
        .ok_or_else(|| {
            VulkanRuntimeResidencyPlanError(format!(
                "runtime placement calibration found no execution for signal processor {component_id:?}",
            ))
        })?;
    vulkan_runtime_placement_calibration_target_from_execution(component_id, execution, phase)
}

/// Discovers one representative for every distinct compiler-emitted
/// transaction in the requested execution phase. Component instance names
/// are deliberately excluded from equivalence; all exact executable and
/// geometry facts remain part of each signature.
pub fn vulkan_runtime_placement_calibration_targets_for_phase(
    runtime_model: &VulkanResidentRuntimeModel,
    phase: VulkanTargetedComponentExecutionPhase,
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
    let mut signature_target_indices = BTreeMap::<String, usize>::new();
    let mut targets = Vec::<VulkanRuntimePlacementCalibrationTarget>::new();

    for component_id in signal_component_ids {
        if matches!(phase, VulkanTargetedComponentExecutionPhase::Prefill { .. }) {
            let execution = runtime_model
                .component_executions
                .iter()
                .find(|execution| execution.component_id == component_id)
                .ok_or_else(|| {
                    VulkanRuntimeResidencyPlanError(format!(
                        "runtime placement calibration found no execution for signal processor {component_id:?}",
                    ))
                })?;
            if !vulkan_runtime_placement_execution_supports_phase(execution, phase) {
                // Prefill is an end-to-end transaction. A width unsupported by
                // even one signal processor cannot be represented by a partial
                // calibration set.
                return Ok(Vec::new());
            }
        }
        let target = vulkan_runtime_placement_calibration_target_for_component(
            runtime_model,
            &component_id,
            phase,
        )?;
        if let Some(target_index) = signature_target_indices.get(&target.signature_id).copied() {
            targets[target_index].component_ids.push(component_id);
            continue;
        }
        signature_target_indices.insert(target.signature_id.clone(), targets.len());
        targets.push(target);
    }
    Ok(targets)
}

fn vulkan_runtime_placement_calibration_target_from_execution(
    component_id: &str,
    execution: &VulkanResidentComponentExecutionSpec,
    phase: VulkanTargetedComponentExecutionPhase,
) -> Result<VulkanRuntimePlacementCalibrationTarget, VulkanRuntimeResidencyPlanError> {
    if !vulkan_runtime_placement_execution_supports_phase(execution, phase) {
        let phase_name = match phase {
            VulkanTargetedComponentExecutionPhase::Decode => "decode",
            VulkanTargetedComponentExecutionPhase::Prefill { .. } => "causal prefill",
        };
        return Err(VulkanRuntimeResidencyPlanError(format!(
            "runtime placement calibration found no complete {phase_name} transaction for signal processor {component_id:?}",
        )));
    }
    let mut kernels = execution.kernels.iter().collect::<Vec<_>>();
    kernels.sort_by_key(|kernel| kernel.execution_index);
    let phase_name = match phase {
        VulkanTargetedComponentExecutionPhase::Decode => "decode",
        VulkanTargetedComponentExecutionPhase::Prefill { .. } => "prefill",
    };
    let terminal = kernels.last().ok_or_else(|| {
        VulkanRuntimeResidencyPlanError(format!(
            "runtime placement calibration found no {phase_name} kernel for signal processor {component_id:?}",
        ))
    })?;
    let kernel_signatures = kernels
        .iter()
        .map(|kernel| {
                let selected_batch_implementation = match phase {
                    VulkanTargetedComponentExecutionPhase::Decode => None,
                    VulkanTargetedComponentExecutionPhase::Prefill {
                        activation_batch_width,
                    } => vulkan_runtime_placement_prefill_implementation(
                        kernel,
                        activation_batch_width,
                    )
                    .expect("phase support was validated before signature construction"),
                };
                let mut physical_contracts = kernel
                    .physical_execution_contracts
                    .iter()
                    .map(vulkan_runtime_placement_structural_contract_fingerprint)
                    .collect::<Result<Vec<_>, _>>()?;
                physical_contracts.sort_unstable();
                Ok((
                    kernel.execution_index,
                    kernel.op.as_str(),
                    &kernel.execution_domain,
                    kernel.stream_control_binding,
                    kernel.shader_path.as_str(),
                    kernel.local_size_x,
                    kernel.workgroup_count_x,
                    &kernel.batch_mode,
                    selected_batch_implementation,
                    &kernel.resource_representation_dispatch,
                    physical_contracts,
                ))
            })
            .collect::<Result<Vec<_>, VulkanRuntimeResidencyPlanError>>()?;
    let signature_payload = serde_json::to_vec(&(
        phase,
        execution.operator_type.as_str(),
        execution.implementation.as_str(),
        kernel_signatures,
    ))
    .map_err(|error| {
        VulkanRuntimeResidencyPlanError(format!(
            "runtime placement calibration could not encode {phase_name} execution signature for {component_id:?}: {error}",
        ))
    })?;
    Ok(VulkanRuntimePlacementCalibrationTarget {
        signature_id: format!("{:x}", Sha256::digest(&signature_payload)),
        component_id: component_id.to_string(),
        component_ids: vec![component_id.to_string()],
        terminal_node_id: terminal.node_id.clone(),
        implementation: execution.implementation.clone(),
        planned_resident_parameter_bytes: 0,
    })
}

/// Produces the performance identity of a physical contract without retaining
/// the identity of the component instance that owns it. The compiler seals
/// exact resource and node names into `implementation_digest`; those names are
/// required for replay, but do not change the work performed by an otherwise
/// identical transaction. Calibration cohorts must therefore preserve the
/// executable structure and resource relationships while replacing instance
/// names with deterministic local aliases.
fn vulkan_runtime_placement_structural_contract_fingerprint(
    contract: &nerve_execution_contracts::PhysicalExecutionContract,
) -> Result<String, VulkanRuntimeResidencyPlanError> {
    let mut structural = contract.clone();
    structural.contract_id.clear();
    structural.implementation_digest.clear();
    structural.member_node_ids = structural
        .member_node_ids
        .iter()
        .enumerate()
        .map(|(ordinal, _)| format!("member:{ordinal}"))
        .collect();
    for (ordinal, artifact) in structural.artifacts.iter_mut().enumerate() {
        artifact.path = format!("artifact:{ordinal}");
    }

    let mut resource_aliases = BTreeMap::<String, String>::new();
    for partition in &mut structural.parameter_partitions {
        let next_alias = format!("parameter:binding:{}", partition.binding);
        let alias = resource_aliases
            .entry(partition.resource.clone())
            .or_insert(next_alias)
            .clone();
        partition.resource = alias;
    }
    for (ordinal, resource) in structural.resources.iter_mut().enumerate() {
        let next_alias = resource
            .binding
            .map(|binding| format!("resource:binding:{binding}"))
            .unwrap_or_else(|| format!("resource:ordinal:{ordinal}"));
        let alias = resource_aliases
            .entry(resource.resource.clone())
            .or_insert(next_alias)
            .clone();
        resource.resource = alias;
    }

    let mut atomic_group_aliases = BTreeMap::<String, String>::new();
    for resource in &mut structural.resources {
        let Some(atomic_group) = resource.atomic_group.as_ref() else {
            continue;
        };
        let next_alias = format!("atomic_group:{}", atomic_group_aliases.len());
        let alias = atomic_group_aliases
            .entry(atomic_group.clone())
            .or_insert(next_alias)
            .clone();
        resource.atomic_group = Some(alias);
    }
    for (ordinal, partition) in structural
        .selected_resource_partitions
        .iter_mut()
        .enumerate()
    {
        partition.selection_signal = format!("selection_signal:{ordinal}");
    }
    for intermediate in &mut structural.local_intermediates {
        intermediate.signal = format!(
            "local_intermediate:{}:{}",
            intermediate.producer_binding, intermediate.consumer_binding,
        );
    }

    let payload = serde_json::to_vec(&structural).map_err(|error| {
        VulkanRuntimeResidencyPlanError(format!(
            "runtime placement calibration could not encode structural physical contract: {error}",
        ))
    })?;
    Ok(format!("{:x}", Sha256::digest(payload)))
}

#[cfg(test)]
mod runtime_placement_signature_tests {
    use super::*;
    use crate::test_support::tiny_model_package_manifest_path;

    fn fixture_execution() -> VulkanResidentComponentExecutionSpec {
        VulkanResidentModelPackageManifest::from_json_file(tiny_model_package_manifest_path())
            .unwrap()
            .component_executions
            .into_iter()
            .next()
            .expect("tiny model fixture must contain a component execution")
    }

    #[test]
    fn calibration_signature_reuses_equivalent_component_contract_labels() {
        let first = fixture_execution();
        let mut relabelled = first.clone();
        for (kernel_ordinal, kernel) in relabelled.kernels.iter_mut().enumerate() {
            for (contract_ordinal, contract) in
                kernel.physical_execution_contracts.iter_mut().enumerate()
            {
                contract.contract_id =
                    format!("component-b:{kernel_ordinal}:{contract_ordinal}");
            }
        }

        let first_target = vulkan_runtime_placement_calibration_target_from_execution(
            "component-a",
            &first,
            VulkanTargetedComponentExecutionPhase::Decode,
        )
        .unwrap();
        let relabelled_target = vulkan_runtime_placement_calibration_target_from_execution(
            "component-b",
            &relabelled,
            VulkanTargetedComponentExecutionPhase::Decode,
        )
        .unwrap();

        assert_eq!(first_target.signature_id, relabelled_target.signature_id);
    }

    #[test]
    fn calibration_signature_reuses_contracts_with_instance_specific_resources() {
        let first = fixture_execution();
        let mut changed = first.clone();
        let contract = &mut changed.kernels[0].physical_execution_contracts[0];
        contract.contract_id = format!("sha256:{}", "e".repeat(64));
        contract.implementation_digest = format!("sha256:{}", "f".repeat(64));
        for partition in &mut contract.parameter_partitions {
            partition.resource = format!("another.instance.{}", partition.binding);
        }
        for resource in &mut contract.resources {
            resource.resource = format!("another.instance.{:?}", resource.binding);
            resource.atomic_group = resource
                .atomic_group
                .as_ref()
                .map(|_| "another.instance.atomic_group".to_string());
        }

        let first_target = vulkan_runtime_placement_calibration_target_from_execution(
            "component-a",
            &first,
            VulkanTargetedComponentExecutionPhase::Decode,
        )
        .unwrap();
        let changed_target = vulkan_runtime_placement_calibration_target_from_execution(
            "component-b",
            &changed,
            VulkanTargetedComponentExecutionPhase::Decode,
        )
        .unwrap();

        assert_eq!(first_target.signature_id, changed_target.signature_id);
    }

    #[test]
    fn calibration_signature_rejects_changed_contract_artifact() {
        let first = fixture_execution();
        let mut changed = first.clone();
        changed.kernels[0].physical_execution_contracts[0].artifacts[0].sha256 =
            format!("sha256:{}", "f".repeat(64));

        let first_target = vulkan_runtime_placement_calibration_target_from_execution(
            "component-a",
            &first,
            VulkanTargetedComponentExecutionPhase::Decode,
        )
        .unwrap();
        let changed_target = vulkan_runtime_placement_calibration_target_from_execution(
            "component-b",
            &changed,
            VulkanTargetedComponentExecutionPhase::Decode,
        )
        .unwrap();

        assert_ne!(first_target.signature_id, changed_target.signature_id);
    }
}

fn vulkan_runtime_placement_execution_supports_phase(
    execution: &VulkanResidentComponentExecutionSpec,
    phase: VulkanTargetedComponentExecutionPhase,
) -> bool {
    !execution.kernels.is_empty()
        && execution.kernels.iter().all(|kernel| match phase {
            VulkanTargetedComponentExecutionPhase::Decode => {
                kernel.execution_domain.supports_decode()
            }
            VulkanTargetedComponentExecutionPhase::Prefill {
                activation_batch_width,
            } => vulkan_runtime_placement_prefill_implementation(kernel, activation_batch_width)
                .is_ok(),
        })
}

fn vulkan_runtime_placement_prefill_implementation(
    kernel: &VulkanResidentComponentKernelSpec,
    activation_batch_width: usize,
) -> Result<Option<&VulkanResidentComponentBatchImplementationSpec>, ()> {
    if activation_batch_width == 0 {
        return Err(());
    }
    if kernel.batch_mode == VulkanResidentComponentKernelBatchMode::SerialLanes {
        return Ok(None);
    }
    let selected = kernel
        .batch_implementations
        .iter()
        .filter(|implementation| {
            implementation.execution_domain.supports_prefill()
                && implementation.causal_sequence_compatible
        })
        .min_by_key(|implementation| {
            let lane_tile_width = implementation.lane_tile_width as usize;
            let priority = usize::MAX - implementation.selection_priority as usize;
            if kernel.batch_mode == VulkanResidentComponentKernelBatchMode::CausalScan {
                (0usize, priority, usize::MAX - lane_tile_width)
            } else if lane_tile_width >= activation_batch_width {
                (1usize, priority, lane_tile_width)
            } else {
                (2usize, usize::MAX - lane_tile_width, priority)
            }
        });
    let Some(selected) = selected else {
        // The component batch runner executes the scalar primary dispatch once
        // per causal lane when no compatible batch artifact is available.
        return Ok(None);
    };
    if kernel.batch_mode == VulkanResidentComponentKernelBatchMode::CausalScan
        && activation_batch_width > selected.lane_tile_width as usize
    {
        return Err(());
    }
    Ok(Some(selected))
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

/// Measures only the exact execution signatures whose component occurrences
/// are compatible with this physical target. A partially compatible target is
/// therefore available to placement for the components it can execute instead
/// of being discarded because an unrelated component needs another device.
pub fn calibrate_vulkan_runtime_placement_candidate_components(
    device: Rc<VulkanComputeDevice>,
    manifest_dir: impl AsRef<Path>,
    capability_class: &str,
    suite: &mut VulkanRuntimePlacementCalibrationSuite,
    compatible_component_ids: &BTreeSet<String>,
) -> Result<Vec<VulkanRuntimePlacementCalibrationReport>, VulkanResidentTokenModelPackageError> {
    calibrate_vulkan_runtime_placement_phase_candidate_with_policy(
        device,
        manifest_dir.as_ref(),
        capability_class,
        suite,
        VulkanTargetedComponentExecutionPhase::Decode,
        VulkanTargetedComponentExecutionScope::DecodeComponentPrefix,
        VulkanRuntimePlacementCalibrationPolicy::default(),
        Some(compatible_component_ids),
    )
}

fn compatible_runtime_placement_calibration_target_indices(
    targets: &[VulkanRuntimePlacementCalibrationTarget],
    compatible_component_ids: &BTreeSet<String>,
) -> Result<Vec<usize>, VulkanResidentTokenModelPackageError> {
    if compatible_component_ids.is_empty() {
        return Err(VulkanResidentTokenModelPackageError::new(
            "runtime placement calibration compatibility set is empty",
        ));
    }
    let target_component_ids = targets
        .iter()
        .flat_map(|target| target.component_ids.iter().cloned())
        .collect::<BTreeSet<_>>();
    let unknown_component_ids = compatible_component_ids
        .difference(&target_component_ids)
        .cloned()
        .collect::<Vec<_>>();
    if !unknown_component_ids.is_empty() {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "runtime placement calibration compatibility references unknown component instances {}",
            unknown_component_ids
                .iter()
                .map(|component_id| format!("{component_id:?}"))
                .collect::<Vec<_>>()
                .join(", "),
        )));
    }
    let mut selected = Vec::new();
    for (index, target) in targets.iter().enumerate() {
        let compatible_occurrences = target
            .component_ids
            .iter()
            .filter(|component_id| compatible_component_ids.contains(*component_id))
            .count();
        if compatible_occurrences == 0 {
            continue;
        }
        if compatible_occurrences != target.component_ids.len() {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "runtime placement calibration signature {} has mixed device compatibility across equivalent component instances",
                target.signature_id,
            )));
        }
        selected.push(index);
    }
    if selected.is_empty() {
        return Err(VulkanResidentTokenModelPackageError::new(
            "runtime placement calibration target cannot execute any compiled component signature",
        ));
    }
    Ok(selected)
}

#[cfg(test)]
mod runtime_placement_compatibility_tests {
    use super::*;

    fn target(
        signature_id: &str,
        component_ids: &[&str],
    ) -> VulkanRuntimePlacementCalibrationTarget {
        VulkanRuntimePlacementCalibrationTarget {
            signature_id: signature_id.to_string(),
            component_id: component_ids[0].to_string(),
            component_ids: component_ids
                .iter()
                .map(|component_id| (*component_id).to_string())
                .collect(),
            terminal_node_id: "terminal".to_string(),
            implementation: "fixture".to_string(),
            planned_resident_parameter_bytes: 1,
        }
    }

    #[test]
    fn calibration_compatibility_selects_only_complete_execution_signatures() {
        let targets = [target("shared", &["a", "b"]), target("other", &["c"])];

        assert_eq!(
            compatible_runtime_placement_calibration_target_indices(
                &targets,
                &BTreeSet::from(["c".to_string()]),
            )
            .unwrap(),
            [1],
        );
    }

    #[test]
    fn calibration_compatibility_rejects_mixed_equivalent_instances() {
        let targets = [target("shared", &["a", "b"])];

        let error = compatible_runtime_placement_calibration_target_indices(
            &targets,
            &BTreeSet::from(["a".to_string()]),
        )
        .unwrap_err();

        assert!(error.to_string().contains("mixed device compatibility"));
    }

    #[test]
    fn calibration_compatibility_rejects_source_ids_at_the_instance_boundary() {
        let targets = [target("shared", &["layer_00_repeat"])];

        let error = compatible_runtime_placement_calibration_target_indices(
            &targets,
            &BTreeSet::from(["layer_00".to_string()]),
        )
        .unwrap_err();

        assert!(error.to_string().contains("unknown component instances"));
        assert!(error.to_string().contains("layer_00"));
    }
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
        None,
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
        None,
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
    compatible_component_ids: Option<&BTreeSet<String>>,
) -> Result<Vec<VulkanRuntimePlacementCalibrationReport>, VulkanResidentTokenModelPackageError> {
    if policy.warmup_units == 0
        || policy.measured_units == 0
        || policy.maximum_duration.is_zero()
        || policy.maximum_total_resident_parameter_bytes == 0
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
    let target_indices = compatible_component_ids.map_or_else(
        || Ok((0..suite.targets.len()).collect::<Vec<_>>()),
        |compatible| {
            compatible_runtime_placement_calibration_target_indices(&suite.targets, compatible)
        },
    )?;
    let (plans, shared_prepare_ns, slice_plan_prepare_ns) =
        suite.plans_for_device(&device, capability_class, manifest_dir, &target_indices)?;
    let targets = target_indices
        .iter()
        .map(|index| suite.targets[*index].clone())
        .collect::<Vec<_>>();
    let physical_device_id = device.physical_device_id().to_string();
    let logical_device_id = VULKAN_RUNTIME_PLACEMENT_CALIBRATION_LOGICAL_DEVICE_ID.to_string();
    let parameter_pool = VulkanResidentBufferPool::default();
    parameter_pool
        .register_device(&logical_device_id, Rc::clone(&device))
        .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;

    let execution_result = (|| {
        let mut reports = Vec::with_capacity(targets.len());
        for (target_index, (target, cached_plan)) in targets.iter().zip(&plans).enumerate() {
            let physical_parameter_capacity = policy
                .parameter_capacity_for_physical_device(&physical_device_id)?
                .min(policy.maximum_total_resident_parameter_bytes);
            if target.planned_resident_parameter_bytes > physical_parameter_capacity {
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
                true,
            )?;
            let session_mount_ns =
                u64::try_from(session_mount_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            let activation_batch_width = phase.activation_batch_width();
            let warmup_useful_units = policy
                .warmup_units
                .checked_mul(activation_batch_width)
                .ok_or_else(|| {
                    VulkanResidentTokenModelPackageError::new(
                        "runtime placement calibration warmup work overflowed",
                    )
                })?;
            let measured_useful_units = policy
                .measured_units
                .checked_mul(activation_batch_width)
                .ok_or_else(|| {
                VulkanResidentTokenModelPackageError::new(
                    "runtime placement calibration measured work overflowed",
                )
            })?;
            let warmup = session.execute(&device, warmup_useful_units, 1, 0, remaining)?;
            validate_canonical_demand_prefill_warmup(
                phase,
                cached_plan
                    .plan
                    .slice_plan
                    .physical_residency_schedule
                    .checkpoints
                    .len(),
                warmup.resource_loading,
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
                measured_ns_per_activation: measured
                    .execution_ns
                    .saturating_add((measured_useful_units / 2) as u64)
                    / measured_useful_units as u64,
                measured_windows: measured.throughput_windows,
                physical_dispatch_count: measured.physical_dispatch_count,
                output_digest: measured.output_digest,
                captured_outputs: measured.captured_outputs.ok_or_else(|| {
                    VulkanResidentTokenModelPackageError::new(
                        "runtime placement calibration did not capture canonical outputs",
                    )
                })?,
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

fn validate_canonical_demand_prefill_warmup(
    phase: VulkanTargetedComponentExecutionPhase,
    checkpoint_count: usize,
    loading: VulkanCompiledResourceLoadStatistics,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    if matches!(phase, VulkanTargetedComponentExecutionPhase::Prefill { .. })
        && checkpoint_count > 0
        && loading.load_count == 0
    {
        return Err(VulkanResidentTokenModelPackageError::new(
            "canonical demand-resident prefill warmup executed without loading any selected resources",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod canonical_demand_prefill_warmup_tests {
    use super::*;

    #[test]
    fn demand_prefill_rejects_a_reference_that_skipped_selected_resources() {
        let error = validate_canonical_demand_prefill_warmup(
            VulkanTargetedComponentExecutionPhase::Prefill {
                activation_batch_width: 1,
            },
            1,
            VulkanCompiledResourceLoadStatistics::default(),
        )
        .unwrap_err();

        assert!(error.to_string().contains("without loading"), "{error}");
    }

    #[test]
    fn demand_prefill_accepts_a_warmup_that_loaded_selected_resources() {
        validate_canonical_demand_prefill_warmup(
            VulkanTargetedComponentExecutionPhase::Prefill {
                activation_batch_width: 1,
            },
            1,
            VulkanCompiledResourceLoadStatistics {
                load_count: 1,
                ..VulkanCompiledResourceLoadStatistics::default()
            },
        )
        .unwrap();
    }
}
