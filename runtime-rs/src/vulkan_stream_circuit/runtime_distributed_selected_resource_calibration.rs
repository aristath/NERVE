struct VulkanCalibrationSelectedResourceMount {
    stores: BTreeMap<String, Arc<VulkanCompiledResourceDeviceStore>>,
    dynamic_buffers: BTreeMap<String, Arc<VulkanDynamicResourceBuffers>>,
    transaction_predicates: BTreeMap<String, Arc<VulkanResidentBuffer>>,
    resident_transient_bytes_by_device: BTreeMap<String, usize>,
    resident_host_transient_bytes: usize,
}

impl VulkanCalibrationSelectedResourceMount {
    fn empty() -> Self {
        Self {
            stores: BTreeMap::new(),
            dynamic_buffers: BTreeMap::new(),
            transaction_predicates: BTreeMap::new(),
            resident_transient_bytes_by_device: BTreeMap::new(),
            resident_host_transient_bytes: 0,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn mount_distributed_calibration_selected_resources(
    manifest_dir: &Path,
    execution_scope: &str,
    contract: &Arc<CompiledResourceResidencyContract>,
    execution_plan: &VulkanDistributedExecutionPlan,
    logical_devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
    maximum_total_payload_bytes: usize,
    maximum_payload_bytes_by_device: &BTreeMap<String, usize>,
) -> Result<Option<VulkanCalibrationSelectedResourceMount>, VulkanResidentTokenModelPackageError> {
    let has_selected_resources = execution_plan
        .dispatches
        .iter()
        .any(|dispatch| !dispatch.selected_resource_partitions.is_empty());
    if !has_selected_resources {
        return Ok(Some(VulkanCalibrationSelectedResourceMount::empty()));
    }
    let store_plan = VulkanDistributedSelectedResourceStorePlan::from_execution_plan(execution_plan)
        .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
    if store_plan.devices.len() != logical_devices.len()
        || maximum_payload_bytes_by_device.len() != logical_devices.len()
        || store_plan
            .devices
            .iter()
            .any(|plan| {
                !logical_devices.contains_key(&plan.device_id)
                    || !maximum_payload_bytes_by_device.contains_key(&plan.device_id)
            })
    {
        return distributed_calibration_error(
            "distributed selected-resource calibration does not cover every participant",
        );
    }
    let minimum_load_wave_bytes = store_plan.devices.iter().try_fold(
        0usize,
        |total, plan| total.checked_add(plan.maximum_load_wave_bytes),
    ).ok_or_else(|| {
        distributed_calibration_error_value(
            "distributed selected-resource minimum load-wave bytes overflowed",
        )
    })?;
    if minimum_load_wave_bytes > maximum_total_payload_bytes {
        return Ok(None);
    }

    let layout = Arc::new(
        VulkanCompiledResourceAddressLayout::from_contract(contract)
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?,
    );
    let maximum_ranges_per_group = compiled_resource_maximum_ranges_per_group(contract)
        .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
    let mut remaining_payload_budget = maximum_total_payload_bytes;
    let mut remaining_minimum_wave_bytes = minimum_load_wave_bytes;
    let mut mounted = VulkanCalibrationSelectedResourceMount::empty();

    for device_plan in &store_plan.devices {
        remaining_minimum_wave_bytes = remaining_minimum_wave_bytes
            .checked_sub(device_plan.maximum_load_wave_bytes)
            .expect("selected-resource minimum wave was accumulated above");
        let device = logical_devices.get(&device_plan.device_id).ok_or_else(|| {
            distributed_calibration_error_value(format!(
                "distributed selected-resource calibration has no device {:?}",
                device_plan.device_id,
            ))
        })?;
        let resources_by_selector = device_plan
            .selectors
            .iter()
            .map(|selector| {
                if selector.execution_scope != execution_scope {
                    return distributed_calibration_error(format!(
                        "distributed selector {:?} belongs to execution scope {:?}, expected {execution_scope:?}",
                        selector.selector_id, selector.execution_scope,
                    ));
                }
                Ok((
                    selector.selector_id.clone(),
                    selector
                        .owned_resource_indices
                        .iter()
                        .copied()
                        .collect::<BTreeSet<_>>(),
                ))
            })
            .collect::<Result<BTreeMap<_, _>, VulkanResidentTokenModelPackageError>>()?;
        let ownership = VulkanCompiledResourceSelectorOwnership::from_resource_indices(
            contract,
            resources_by_selector,
        )
        .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let upload_alignment = compiled_resource_upload_alignment(contract, device)
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let store_residency = plan_compiled_resource_store_residency_for_ownership(
            contract,
            &layout,
            &ownership,
            device_plan.maximum_atomic_group_bytes,
            upload_alignment,
        )
        .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        if store_residency.maximum_load_wave_payload_bytes
            != device_plan.maximum_load_wave_bytes
        {
            return distributed_calibration_error(format!(
                "distributed selected-resource load-wave contract changed from {} to {} bytes on {:?}",
                device_plan.maximum_load_wave_bytes,
                store_residency.maximum_load_wave_payload_bytes,
                device_plan.device_id,
            ));
        }
        let fixed_device_bytes = store_residency
            .fixed_device_bytes()
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let admission = device
            .admit_device_local_memory(u64::try_from(fixed_device_bytes).unwrap_or(u64::MAX))
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let safe_dynamic_bytes = usize::try_from(admission.allocatable_bytes).unwrap_or(usize::MAX);
        let addressable_slot_count = layout
            .addressable_slot_count_for_ownership(&ownership)
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let maximum_alignment_padding = addressable_slot_count
            .checked_mul(upload_alignment.saturating_sub(1))
            .ok_or_else(|| {
                distributed_calibration_error_value(
                    "distributed selected-resource alignment capacity overflowed",
                )
            })?;
        let payload_budget_for_device = remaining_payload_budget
            .checked_sub(remaining_minimum_wave_bytes)
            .expect("minimum wave budget was validated above");
        let device_payload_budget = maximum_payload_bytes_by_device[&device_plan.device_id];
        let resident_payload_capacity = device_plan
            .total_addressable_bytes
            .min(safe_dynamic_bytes.saturating_sub(maximum_alignment_padding))
            .min(payload_budget_for_device);
        let resident_payload_capacity = resident_payload_capacity.min(device_payload_budget);
        if resident_payload_capacity < device_plan.maximum_load_wave_bytes {
            return Ok(None);
        }
        let allocation_capacity = resident_payload_capacity
            .checked_add(maximum_alignment_padding)
            .ok_or_else(|| {
                distributed_calibration_error_value(
                    "distributed selected-resource allocation capacity overflowed",
                )
            })?;
        let store_id = format!(
            "calibration:selected_resources:{}:{}",
            device.physical_device_id(), device_plan.device_id,
        );
        let store = Arc::new(
            VulkanCompiledResourceDeviceStore::new_tiered_with_selector_ownership(
                device,
                ResourceResidencyPolicy::DemandPaged,
                store_id,
                device.physical_device_id(),
                vec![device_plan.device_id.clone()],
                manifest_dir,
                Arc::clone(contract),
                Arc::clone(&layout),
                ownership,
                resident_payload_capacity,
                resident_payload_capacity,
                0,
                allocation_capacity,
                device_plan.maximum_atomic_group_bytes,
                maximum_ranges_per_group,
                0,
                0,
                store_residency.metadata_device_bytes,
                None,
            )
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?,
        );
        store
            .register_device_memory_reclaimer(device)
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let component_ids = device_plan
            .selectors
            .iter()
            .map(|selector| selector.component_id.clone())
            .collect::<BTreeSet<_>>();
        let dynamic_buffers = store
            .dynamic_buffers_for_components(device, execution_scope, &component_ids)
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        store
            .mark_mount_complete()
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let predicate = Arc::new(
            device
                .create_conditional_resident_buffer(size_of::<u32>())
                .map_err(|error| distributed_calibration_error_value(error.to_string()))?,
        );
        predicate
            .write_bytes(&1u32.to_le_bytes())
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let store_report = store
            .residency_report()
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let device_transient_bytes = store_report
            .metadata_device_bytes
            .checked_add(predicate.byte_capacity())
            .ok_or_else(|| {
                distributed_calibration_error_value(
                    "distributed selected-resource transient bytes overflowed",
                )
            })?;
        mounted.resident_host_transient_bytes = mounted
            .resident_host_transient_bytes
            .checked_add(store_report.transfer_staging_host_bytes)
            .ok_or_else(|| {
                distributed_calibration_error_value(
                    "distributed selected-resource host transient bytes overflowed",
                )
            })?;
        remaining_payload_budget = remaining_payload_budget
            .checked_sub(resident_payload_capacity)
            .expect("resident payload was capped by the remaining budget");
        mounted
            .stores
            .insert(device_plan.device_id.clone(), Arc::clone(&store));
        mounted
            .dynamic_buffers
            .insert(device_plan.device_id.clone(), dynamic_buffers);
        mounted
            .transaction_predicates
            .insert(device_plan.device_id.clone(), predicate);
        mounted
            .resident_transient_bytes_by_device
            .insert(device_plan.device_id.clone(), device_transient_bytes);
    }
    Ok(Some(mounted))
}
