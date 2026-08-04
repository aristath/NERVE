pub struct VulkanResidentTargetedModelPackageDeviceSlice {
    slice: VulkanResidentModelPackageDeviceSlice,
    demand_context: Option<VulkanDemandResidencyExecutionContext>,
}

impl VulkanResidentTargetedModelPackageDeviceSlice {
    pub fn from_runtime_model_for_device_with_parameter_pool(
        device: &VulkanComputeDevice,
        manifest_dir: impl AsRef<Path>,
        runtime_model: VulkanResidentRuntimeModel,
        component_id: impl AsRef<str>,
        device_id: impl AsRef<str>,
        dynamic_state_capacity_activations: Option<usize>,
        parameter_pool: &VulkanResidentBufferPool,
    ) -> Result<Self, VulkanResidentTokenModelPackageError> {
        let manifest_dir = manifest_dir.as_ref();
        let component_id = component_id.as_ref();
        let device_id = device_id.as_ref();
        let execution_scope = runtime_model.execution_scope.clone();
        let capacity = dynamic_state_capacity_activations
            .unwrap_or(runtime_model.package.max_context_activations);
        let tensor_index = runtime_model.load_runtime_tensor_index(manifest_dir)?;
        let contract = Arc::new(
            instantiate_runtime_resource_contract(&runtime_model).map_err(|error| {
                targeted_component_error_value(format!(
                    "failed to instantiate targeted runtime resource contract: {error}"
                ))
            })?,
        );
        let residency_plan = plan_vulkan_runtime_residency_with_contract(
            manifest_dir,
            &runtime_model,
            &tensor_index,
            capacity,
            false,
            ResourceResidencyPolicy::DemandRetained,
            &contract,
        )
        .map_err(|error| {
            targeted_component_error_value(format!(
                "failed to plan targeted demand residency: {error}"
            ))
        })?;
        let mut slice = VulkanResidentModelPackageDeviceSlice::
            from_runtime_model_for_device_with_parameter_pool(
                device,
                manifest_dir,
                runtime_model,
                device_id,
                Some(capacity),
                parameter_pool,
            )?;
        if slice.physical_residency_schedule().checkpoints.is_empty() {
            return Ok(Self {
                slice,
                demand_context: None,
            });
        }

        let allowed_selector_ids =
            targeted_demand_selector_ids(&contract.selectors, &execution_scope, component_id);
        if allowed_selector_ids.is_empty() {
            return targeted_component_error(format!(
                "targeted demand-resident component {component_id:?} has no owned selectors"
            ));
        }
        let layout = Arc::new(
            VulkanCompiledResourceAddressLayout::from_contract(&contract).map_err(|error| {
                targeted_component_error_value(format!(
                    "failed to lower targeted compiled-resource addresses: {error}"
                ))
            })?,
        );
        let parameter_residency = residency_plan
            .device_plans
            .iter()
            .find(|plan| plan.device_id == device_id)
            .ok_or_else(|| {
                targeted_component_error_value(format!(
                    "targeted residency plan omitted device {device_id:?}"
                ))
            })?;
        let maximum_dynamic_bytes = parameter_residency
            .parameter_residency
            .maximum_addressable_bytes
            .checked_sub(
                parameter_residency
                    .parameter_residency
                    .always_resident_bytes,
            )
            .ok_or_else(|| {
                targeted_component_error_value("targeted dynamic parameter accounting underflowed")
            })?;
        if maximum_dynamic_bytes == 0 {
            return targeted_component_error(
                "targeted demand-resident component has no dynamic parameter bytes",
            );
        }
        let maximum_group_bytes = parameter_residency
            .parameter_residency
            .staging_headroom_bytes;
        if maximum_group_bytes == 0 {
            return targeted_component_error(
                "targeted demand-resident component has no upload staging headroom",
            );
        }
        let component_ids = BTreeSet::from([component_id.to_string()]);
        let metadata_bytes = layout
            .address_table_byte_count()
            .and_then(|address_bytes| {
                layout
                    .parameter_slot_table_byte_count_for_components(
                        &execution_scope,
                        &component_ids,
                    )
                    .and_then(|slot_bytes| {
                        address_bytes.checked_add(slot_bytes).ok_or_else(|| {
                            VulkanCompiledResourceAddressLayoutError(
                                "targeted resource metadata accounting overflowed".to_string(),
                            )
                        })
                    })
            })
            .map_err(|error| targeted_component_error_value(error.to_string()))?;
        let working_set_bytes = parameter_residency
            .working_set
            .transient_state_bytes
            .checked_add(parameter_residency.working_set.activation_headroom_bytes)
            .ok_or_else(|| {
                targeted_component_error_value("targeted working-set accounting overflowed")
            })?;
        let pending_fixed_bytes = working_set_bytes
            .checked_add(maximum_group_bytes)
            .and_then(|bytes| bytes.checked_add(metadata_bytes))
            .ok_or_else(|| {
                targeted_component_error_value("targeted fixed-residency accounting overflowed")
            })?;
        let admission = device
            .admit_device_local_memory(u64::try_from(pending_fixed_bytes).unwrap_or(u64::MAX))
            .map_err(|error| targeted_component_error_value(error.to_string()))?;
        let safe_dynamic_bytes =
            usize::try_from(admission.allocatable_bytes).unwrap_or(usize::MAX);
        let upload_alignment = compiled_resource_upload_alignment(&contract, device)
            .map_err(|error| targeted_component_error_value(error.to_string()))?;
        let addressable_slot_count = layout
            .addressable_slot_count_for_selectors(&allowed_selector_ids)
            .map_err(|error| targeted_component_error_value(error.to_string()))?;
        let maximum_alignment_padding = addressable_slot_count
            .checked_mul(upload_alignment.saturating_sub(1))
            .ok_or_else(|| {
                targeted_component_error_value(
                    "targeted dynamic-resource alignment capacity overflowed",
                )
            })?;
        let resident_payload_capacity =
            maximum_dynamic_bytes.min(safe_dynamic_bytes.saturating_sub(maximum_alignment_padding));
        if resident_payload_capacity < maximum_group_bytes {
            return targeted_component_error(format!(
                "targeted demand residency can admit {resident_payload_capacity} payload bytes but one selected group requires {maximum_group_bytes}"
            ));
        }
        let maximum_ranges_per_group = compiled_resource_maximum_ranges_per_group(&contract)
            .map_err(|error| targeted_component_error_value(error.to_string()))?;
        let store_id = format!(
            "{}:targeted:{device_id}:{execution_scope}:{component_id}",
            slice.package_id,
        );
        let store = Arc::new(
            VulkanCompiledResourceDeviceStore::new(
                device,
                ResourceResidencyPolicy::DemandRetained,
                store_id.clone(),
                device.physical_device_id(),
                vec![device_id.to_string()],
                manifest_dir,
                Arc::clone(&contract),
                Arc::clone(&layout),
                allowed_selector_ids,
                resident_payload_capacity,
                safe_dynamic_bytes,
                maximum_group_bytes,
                maximum_ranges_per_group,
                parameter_residency
                    .parameter_residency
                    .always_resident_bytes,
                working_set_bytes,
                metadata_bytes,
            )
            .map_err(|error| {
                targeted_component_error_value(format!(
                    "failed to create targeted compiled-resource store: {error}"
                ))
            })?,
        );
        store
            .register_device_memory_reclaimer(device)
            .map_err(|error| {
                targeted_component_error_value(format!(
                    "failed to register targeted compiled-resource capacity reclamation: {error}"
                ))
            })?;
        slice.dynamic_resource_buffers = Some(
            store
                .dynamic_buffers_for_components(device, &execution_scope, &component_ids)
                .map_err(|error| {
                    targeted_component_error_value(format!(
                        "failed to bind targeted dynamic resources: {error}"
                    ))
                })?,
        );
        store.mark_mount_complete().map_err(|error| {
            targeted_component_error_value(format!(
                "failed to seal targeted dynamic-resource store: {error}"
            ))
        })?;
        let demand_context = VulkanDemandResidencyExecutionContext {
            execution_scope,
            contract,
            layout,
            store,
            owner: DeviceResourceResidencyOwnerId::new(format!("{store_id}:session"))
                .map_err(|error| targeted_component_error_value(error.to_string()))?,
        };
        Ok(Self {
            slice,
            demand_context: Some(demand_context),
        })
    }
}

fn targeted_demand_selector_ids(
    selectors: &[CompiledResourceSelector],
    execution_scope: &str,
    component_id: &str,
) -> BTreeSet<String> {
    selectors
        .iter()
        .filter(|selector| {
            selector.execution_scope == execution_scope && selector.component_id == component_id
        })
        .map(|selector| selector.id.clone())
        .collect()
}
