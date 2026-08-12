const VULKAN_SELECTED_RESOURCE_MOUNT_PLACEMENT_MAXIMUM_ITERATIONS: usize = 8;

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanRuntimeSelectedResourceMountDevice {
    logical_device_id: String,
    physical_device_id: String,
    execution_identity: VulkanPlacementDeviceExecutionIdentity,
    live_safe_capacity_bytes: usize,
    upload_alignment: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanRuntimeDistributedMountPlans {
    execution_plans: VulkanDistributedExecutionPlanSet,
    activation_plan: VulkanDistributedActivationBufferPlan,
    parameter_allocation_plan: VulkanDistributedParameterAllocationPlan,
    parameter_exclusion_plan: VulkanDistributedParameterExclusionPlan,
    selected_resource_execution_ownership_plan:
        VulkanDistributedSelectedResourceStorePlan,
    selected_resource_store_plan: VulkanDistributedSelectedResourceStorePlan,
    physical_execution_residency_plan: VulkanRuntimePhysicalExecutionResidencyPlan,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanRuntimeSelectedResourceMountResolution {
    plans: VulkanRuntimeDistributedMountPlans,
    placements: Vec<VulkanSelectedResourcePlacementPlan>,
}

impl VulkanRuntimeDistributedMountPlans {
    fn derive(
        execution_plans: VulkanDistributedExecutionPlanSet,
        residency_plan: &VulkanRuntimeResidencyPlan,
        logical_device_ids: &[String],
        prepared_plans: &[(&str, &VulkanPreparedDispatchPlan)],
        tensor_index: &TensorIndex,
        residency_policy: ResourceResidencyPolicy,
    ) -> Result<Self, VulkanResidentTokenModelPackageError> {
        let activation_plan = VulkanDistributedActivationBufferPlan::from_execution_plan_set(
            &execution_plans,
        )
        .map_err(|error| selected_resource_mount_error("activation planning", error))?;
        let parameter_allocation_plan =
            VulkanDistributedParameterAllocationPlan::from_execution_plan_set(
                &execution_plans,
                tensor_index,
            )
            .map_err(|error| selected_resource_mount_error("parameter planning", error))?;
        let parameter_exclusion_plan =
            VulkanDistributedParameterExclusionPlan::from_execution_plan_set(
                &execution_plans,
                prepared_plans,
                tensor_index,
            )
            .map_err(|error| selected_resource_mount_error("parameter exclusion planning", error))?;
        let selected_resource_execution_ownership_plan =
            VulkanDistributedSelectedResourceStorePlan::from_execution_plan_set(&execution_plans)
                .map_err(|error| {
                    selected_resource_mount_error("selected-resource ownership planning", error)
                })?;
        let selected_resource_store_plan = if residency_policy.is_demand_loaded() {
            selected_resource_execution_ownership_plan
                .with_whole_resource_addressability_envelope()
                .map_err(|error| {
                    selected_resource_mount_error(
                        "selected-resource addressability planning",
                        error,
                    )
                })?
        } else {
            selected_resource_execution_ownership_plan.clone()
        };
        let physical_execution_residency_plan =
            VulkanRuntimePhysicalExecutionResidencyPlan::plan(
                residency_plan,
                logical_device_ids,
                &parameter_allocation_plan,
                &parameter_exclusion_plan,
                &activation_plan,
            )
            .map_err(|error| selected_resource_mount_error("physical residency planning", error))?;
        Ok(Self {
            execution_plans,
            activation_plan,
            parameter_allocation_plan,
            parameter_exclusion_plan,
            selected_resource_execution_ownership_plan,
            selected_resource_store_plan,
            physical_execution_residency_plan,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn resolve_vulkan_runtime_selected_resource_mount(
    runtime_model: &VulkanResidentRuntimeModel,
    resource_contract: &CompiledResourceResidencyContract,
    loaded_manifest: &VulkanLoadedKernelArtifactCatalog,
    baseline_execution_plans: VulkanDistributedExecutionPlanSet,
    residency_plan: &VulkanRuntimeResidencyPlan,
    logical_device_ids: &[String],
    prepared_plans: &[(&str, &VulkanPreparedDispatchPlan)],
    tensor_index: &TensorIndex,
    devices: &[VulkanRuntimeSelectedResourceMountDevice],
    input_device_id: &str,
    output_device_id: &str,
    mount_speculative_decoders: bool,
    residency_policy: ResourceResidencyPolicy,
    catalog: Option<&VulkanPlacementCalibrationCatalog>,
    telemetry: Option<&VulkanSelectionTelemetrySnapshot>,
) -> Result<VulkanRuntimeSelectedResourceMountResolution, VulkanResidentTokenModelPackageError> {
    let baseline = VulkanRuntimeDistributedMountPlans::derive(
        baseline_execution_plans.clone(),
        residency_plan,
        logical_device_ids,
        prepared_plans,
        tensor_index,
        residency_policy,
    )?;
    let Some(catalog) = catalog else {
        return Ok(VulkanRuntimeSelectedResourceMountResolution {
            plans: baseline,
            placements: Vec::new(),
        });
    };
    let requirements = vulkan_runtime_selected_resource_execution_requirements(
        runtime_model,
        resource_contract,
        loaded_manifest,
        &baseline.execution_plans.decode,
        VulkanTargetedComponentExecutionPhase::Decode,
    )?;
    if requirements.is_empty() {
        return Ok(VulkanRuntimeSelectedResourceMountResolution {
            plans: baseline,
            placements: Vec::new(),
        });
    }

    let mut current = baseline;
    let mut seen_assignments = BTreeSet::new();
    let mut best_feasible = None::<VulkanRuntimeSelectedResourceMountResolution>;
    for _ in 0..VULKAN_SELECTED_RESOURCE_MOUNT_PLACEMENT_MAXIMUM_ITERATIONS {
        let Some(capacities) = selected_resource_mount_capacities(
            runtime_model,
            resource_contract,
            &current,
            devices,
            input_device_id,
            output_device_id,
            mount_speculative_decoders,
            residency_policy,
        )? else {
            break;
        };
        let Some(placements) = try_plan_vulkan_runtime_selected_resource_placements(
            &baseline_execution_plans.decode,
            &requirements,
            catalog,
            &capacities,
            telemetry,
            residency_policy,
            nerve_execution_contracts::ExecutionPhase::Decode,
        )? else {
            break;
        };
        let assignment = selected_resource_mount_assignment_key(&placements);
        if !seen_assignments.insert(assignment.clone()) {
            break;
        }
        if !selected_resource_placements_fit_phase_participants(
            &baseline_execution_plans,
            &placements,
        )
        .map_err(|error| selected_resource_mount_error("phase compatibility", error))?
        {
            break;
        }
        let mut candidate_execution_plans = baseline_execution_plans.clone();
        candidate_execution_plans
            .apply_selected_resource_placements(&placements)
            .map_err(|error| selected_resource_mount_error("placement replay", error))?;
        let candidate = VulkanRuntimeDistributedMountPlans::derive(
            candidate_execution_plans,
            residency_plan,
            logical_device_ids,
            prepared_plans,
            tensor_index,
            residency_policy,
        )?;
        let Some(candidate_capacities) = selected_resource_mount_capacities(
            runtime_model,
            resource_contract,
            &candidate,
            devices,
            input_device_id,
            output_device_id,
            mount_speculative_decoders,
            residency_policy,
        )? else {
            current = candidate;
            continue;
        };
        if selected_resource_placements_fit(
            &placements,
            &candidate_capacities,
            residency_policy,
        )? {
            let resolution = VulkanRuntimeSelectedResourceMountResolution {
                plans: candidate.clone(),
                placements: placements.clone(),
            };
            if best_feasible.as_ref().is_none_or(|best| {
                selected_resource_mount_objective(&placements)
                    < selected_resource_mount_objective(&best.placements)
            }) {
                best_feasible = Some(resolution);
            }
        }
        current = candidate;
    }
    Ok(best_feasible.unwrap_or(VulkanRuntimeSelectedResourceMountResolution {
        plans: VulkanRuntimeDistributedMountPlans::derive(
            baseline_execution_plans,
            residency_plan,
            logical_device_ids,
            prepared_plans,
            tensor_index,
            residency_policy,
        )?,
        placements: Vec::new(),
    }))
}

#[allow(clippy::too_many_arguments)]
fn selected_resource_mount_capacities(
    runtime_model: &VulkanResidentRuntimeModel,
    resource_contract: &CompiledResourceResidencyContract,
    plans: &VulkanRuntimeDistributedMountPlans,
    devices: &[VulkanRuntimeSelectedResourceMountDevice],
    input_device_id: &str,
    output_device_id: &str,
    mount_speculative_decoders: bool,
    residency_policy: ResourceResidencyPolicy,
) -> Result<Option<Vec<VulkanPlacementSelectedResourceDeviceCapacity>>, VulkanResidentTokenModelPackageError>
{
    if devices.is_empty() {
        return Err(VulkanResidentTokenModelPackageError::new(
            "selected-resource mount capacity has no devices",
        ));
    }
    let logical_ids = devices
        .iter()
        .map(|device| device.logical_device_id.as_str())
        .collect::<BTreeSet<_>>();
    let physical_ids = devices
        .iter()
        .map(|device| device.physical_device_id.as_str())
        .collect::<BTreeSet<_>>();
    if logical_ids.len() != devices.len()
        || physical_ids.len() != devices.len()
        || devices.iter().any(|device| {
            device.logical_device_id.is_empty()
                || device.physical_device_id.is_empty()
                || device.execution_identity.physical_device_id != device.physical_device_id
                || !device.upload_alignment.is_power_of_two()
        })
    {
        return Err(VulkanResidentTokenModelPackageError::new(
            "selected-resource mount capacity requires one exact logical binding per physical device",
        ));
    }
    let layout = VulkanCompiledResourceAddressLayout::from_contract(resource_contract)
        .map_err(|error| selected_resource_mount_error("resource address planning", error))?;
    let residency_by_device = plans
        .physical_execution_residency_plan
        .device_plans
        .iter()
        .map(|plan| (plan.device_id.as_str(), plan))
        .collect::<BTreeMap<_, _>>();
    let mut capacities = Vec::with_capacity(devices.len());
    for device in devices {
        let physical = residency_by_device
            .get(device.logical_device_id.as_str())
            .ok_or_else(|| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "selected-resource capacity has no physical residency for {:?}",
                    device.logical_device_id,
                ))
            })?;
        let logical_device_set = BTreeSet::from([device.logical_device_id.clone()]);
        let ownership = compiled_resource_selector_ownership_for_device_set(
            runtime_model,
            resource_contract,
            input_device_id,
            output_device_id,
            &logical_device_set,
            mount_speculative_decoders,
            &plans.selected_resource_store_plan,
        )
        .map_err(|error| selected_resource_mount_error("resource ownership sizing", error))?;
        let mut fixed_bytes = physical
            .mount_device_local_bytes
            .checked_add(physical.stream_device_local_bytes)
            .ok_or_else(|| {
                VulkanResidentTokenModelPackageError::new(
                    "selected-resource fixed mount capacity overflowed",
                )
            })?;
        if let Some(ownership) = ownership {
            let parameters =
                plan_compiled_parameter_residency_for_device_set_with_selector_ownership(
                    runtime_model,
                    resource_contract,
                    input_device_id,
                    output_device_id,
                    &logical_device_set,
                    mount_speculative_decoders,
                    residency_policy,
                    &ownership,
                )
                .map_err(|error| selected_resource_mount_error("resource payload sizing", error))?;
            if parameters.staging_headroom_bytes == 0 {
                return Ok(None);
            }
            let store = plan_compiled_resource_store_residency_for_ownership(
                resource_contract,
                &layout,
                &ownership,
                parameters.staging_headroom_bytes,
                device.upload_alignment,
            )
            .map_err(|error| selected_resource_mount_error("resource store sizing", error))?;
            fixed_bytes = fixed_bytes
                .checked_add(store.fixed_device_bytes().map_err(|error| {
                    selected_resource_mount_error("resource store fixed sizing", error)
                })?)
                .and_then(|bytes| {
                    bytes.checked_add(store.maximum_dynamic_allocation_padding_bytes)
                })
                .ok_or_else(|| {
                    VulkanResidentTokenModelPackageError::new(
                        "selected-resource store capacity overflowed",
                    )
                })?;
        }
        let remaining = device.live_safe_capacity_bytes.saturating_sub(fixed_bytes);
        if remaining == 0 {
            return Ok(None);
        }
        capacities.push(VulkanPlacementSelectedResourceDeviceCapacity {
            device_id: device.logical_device_id.clone(),
            identity: device.execution_identity.clone(),
            resident_payload_capacity_bytes: remaining,
        });
    }
    Ok(Some(capacities))
}

fn selected_resource_placements_fit(
    placements: &[VulkanSelectedResourcePlacementPlan],
    capacities: &[VulkanPlacementSelectedResourceDeviceCapacity],
    residency_policy: ResourceResidencyPolicy,
) -> Result<bool, VulkanResidentTokenModelPackageError> {
    let capacity_by_device = capacities
        .iter()
        .map(|capacity| {
            (
                capacity.device_id.as_str(),
                capacity.resident_payload_capacity_bytes,
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut required_by_device = BTreeMap::<&str, usize>::new();
    for placement in placements {
        for load in &placement.device_loads {
            let capacity = capacity_by_device
                .get(load.device_id.as_str())
                .copied()
                .ok_or_else(|| {
                    VulkanResidentTokenModelPackageError::new(format!(
                        "selected-resource placement references missing capacity {:?}",
                        load.device_id,
                    ))
                })?;
            let required = required_by_device.entry(load.device_id.as_str()).or_default();
            *required = match residency_policy {
                ResourceResidencyPolicy::Eager | ResourceResidencyPolicy::DemandRetained => {
                    required.checked_add(load.addressable_bytes).ok_or_else(|| {
                        VulkanResidentTokenModelPackageError::new(
                            "selected-resource retained capacity overflowed",
                        )
                    })?
                }
                ResourceResidencyPolicy::DemandPaged => {
                    (*required).max(load.maximum_load_wave_bytes)
                }
            };
            if *required > capacity {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn selected_resource_mount_assignment_key(
    placements: &[VulkanSelectedResourcePlacementPlan],
) -> Vec<(String, Vec<(usize, String)>)> {
    placements
        .iter()
        .map(|placement| {
            (
                placement.selector_id.clone(),
                placement
                    .assignments
                    .iter()
                    .map(|assignment| {
                        (assignment.resource_index, assignment.device_id.clone())
                    })
                    .collect(),
            )
        })
        .collect()
}

fn selected_resource_mount_objective(
    placements: &[VulkanSelectedResourcePlacementPlan],
) -> (u128, u128, Vec<(String, Vec<(usize, String)>)>) {
    let second = placements
        .iter()
        .map(|placement| placement.maximum_second_moment_ns2)
        .max()
        .unwrap_or(0);
    let first = placements
        .iter()
        .map(|placement| placement.maximum_first_moment_ns)
        .max()
        .unwrap_or(0);
    (
        second,
        first,
        selected_resource_mount_assignment_key(placements),
    )
}

fn selected_resource_mount_error(
    stage: &str,
    error: impl Display,
) -> VulkanResidentTokenModelPackageError {
    VulkanResidentTokenModelPackageError::new(format!(
        "failed selected-resource mount {stage}: {error}",
    ))
}

#[cfg(test)]
mod runtime_selected_resource_mount_planning_tests {
    use super::*;

    fn capacity(device_id: &str, bytes: usize) -> VulkanPlacementSelectedResourceDeviceCapacity {
        VulkanPlacementSelectedResourceDeviceCapacity {
            device_id: device_id.to_string(),
            identity: VulkanPlacementDeviceExecutionIdentity {
                physical_device_id: format!("physical-{device_id}"),
                api_version: 1,
                driver_version: 1,
            },
            resident_payload_capacity_bytes: bytes,
        }
    }

    fn placement(addressable: usize, wave: usize) -> VulkanSelectedResourcePlacementPlan {
        VulkanSelectedResourcePlacementPlan {
            selector_id: "experts".to_string(),
            assignments: vec![crate::vulkan_distributed::VulkanSelectedResourceAssignment {
                resource_index: 0,
                device_id: "gpu0".to_string(),
            }],
            device_loads: vec![crate::vulkan_distributed::VulkanSelectedResourceDeviceLoad {
                device_id: "gpu0".to_string(),
                addressable_bytes: addressable,
                maximum_load_wave_bytes: wave,
                first_moment_ns: 1,
                second_moment_ns2: 1,
                owned_resource_indices: vec![0],
            }],
            maximum_first_moment_ns: 1,
            maximum_second_moment_ns2: 1,
        }
    }

    #[test]
    fn retained_mount_requires_the_complete_owned_payload() {
        assert!(
            !selected_resource_placements_fit(
                &[placement(101, 16)],
                &[capacity("gpu0", 100)],
                ResourceResidencyPolicy::DemandRetained,
            )
            .unwrap()
        );
    }

    #[test]
    fn paged_mount_requires_only_one_complete_load_wave() {
        assert!(
            selected_resource_placements_fit(
                &[placement(1_000, 64)],
                &[capacity("gpu0", 64)],
                ResourceResidencyPolicy::DemandPaged,
            )
            .unwrap()
        );
    }

    #[test]
    fn mount_capacity_rejects_an_unaccounted_device() {
        let mut plan = placement(32, 16);
        plan.device_loads[0].device_id = "gpu1".to_string();
        assert!(
            selected_resource_placements_fit(
                &[plan],
                &[capacity("gpu0", 100)],
                ResourceResidencyPolicy::Eager,
            )
            .unwrap_err()
            .to_string()
            .contains("missing capacity")
        );
    }

    #[test]
    fn retained_mount_aggregates_payload_across_selectors() {
        let mut second = placement(60, 16);
        second.selector_id = "experts-second-layer".to_string();
        assert!(
            !selected_resource_placements_fit(
                &[placement(60, 16), second],
                &[capacity("gpu0", 100)],
                ResourceResidencyPolicy::DemandRetained,
            )
            .unwrap()
        );
    }

    #[test]
    fn paged_mount_uses_the_largest_cross_selector_wave() {
        let mut second = placement(1_000, 96);
        second.selector_id = "experts-second-layer".to_string();
        assert!(
            selected_resource_placements_fit(
                &[placement(1_000, 64), second],
                &[capacity("gpu0", 96)],
                ResourceResidencyPolicy::DemandPaged,
            )
            .unwrap()
        );
    }
}
