pub struct VulkanResidentTargetedModelPackageDeviceSlice {
    slice: VulkanResidentModelPackageDeviceSlice,
    demand_context: Option<VulkanDemandResidencyExecutionContext>,
}

#[derive(Clone)]
struct VulkanResidentTargetedModelPackageDeviceSlicePlan {
    component_id: String,
    device_id: String,
    execution_scope: String,
    tensor_index: Arc<TensorIndex>,
    contract: Arc<CompiledResourceResidencyContract>,
    residency_plan: VulkanRuntimeResidencyPlan,
    slice_plan: VulkanResidentModelPackageDeviceSlicePlan,
}

impl VulkanResidentTargetedModelPackageDeviceSlicePlan {
    #[allow(clippy::too_many_arguments)]
    fn prepare(
        device: &VulkanComputeDevice,
        manifest_dir: &Path,
        runtime_model: &VulkanResidentRuntimeModel,
        component_id: &str,
        device_id: &str,
        capacity: usize,
        tensor_index: Arc<TensorIndex>,
        contract: Arc<CompiledResourceResidencyContract>,
        residency_plan: VulkanRuntimeResidencyPlan,
    ) -> Result<Self, VulkanResidentTokenModelPackageError> {
        let slice_plan = VulkanResidentModelPackageDeviceSlicePlan::prepare(
            device,
            manifest_dir,
            runtime_model,
            &contract,
            &tensor_index,
            device_id,
            capacity,
        )?;
        Ok(Self {
            component_id: component_id.to_string(),
            device_id: device_id.to_string(),
            execution_scope: runtime_model.execution_scope.clone(),
            tensor_index,
            contract,
            residency_plan,
            slice_plan,
        })
    }

    fn materialize(
        &self,
        device: &VulkanComputeDevice,
        manifest_dir: &Path,
        parameter_pool: &VulkanResidentBufferPool,
    ) -> Result<VulkanResidentTargetedModelPackageDeviceSlice, VulkanResidentTokenModelPackageError>
    {
        self.materialize_excluding_tensors(device, manifest_dir, parameter_pool, &BTreeSet::new())
    }

    fn materialize_excluding_tensors(
        &self,
        device: &VulkanComputeDevice,
        manifest_dir: &Path,
        parameter_pool: &VulkanResidentBufferPool,
        excluded_tensors: &BTreeSet<String>,
    ) -> Result<VulkanResidentTargetedModelPackageDeviceSlice, VulkanResidentTokenModelPackageError>
    {
        let mut slice = self.slice_plan.clone().materialize(
            device,
            &self.tensor_index,
            excluded_tensors,
            Some(parameter_pool),
        )?;
        if slice.physical_residency_schedule().checkpoints.is_empty() {
            return Ok(VulkanResidentTargetedModelPackageDeviceSlice {
                slice,
                demand_context: None,
            });
        }

        let allowed_selector_ids = targeted_demand_selector_ids(
            &self.contract.selectors,
            &self.execution_scope,
            &self.component_id,
        );
        if allowed_selector_ids.is_empty() {
            return targeted_component_error(format!(
                "targeted demand-resident component {:?} has no owned selectors",
                self.component_id,
            ));
        }
        let layout = Arc::new(
            VulkanCompiledResourceAddressLayout::from_contract(&self.contract).map_err(
                |error| {
                    targeted_component_error_value(format!(
                        "failed to lower targeted compiled-resource addresses: {error}"
                    ))
                },
            )?,
        );
        let parameter_residency = self
            .residency_plan
            .device_plans
            .iter()
            .find(|plan| plan.device_id == self.device_id)
            .ok_or_else(|| {
                targeted_component_error_value(format!(
                    "targeted residency plan omitted device {:?}",
                    self.device_id,
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
        let upload_alignment = compiled_resource_upload_alignment(&self.contract, device)
            .map_err(|error| targeted_component_error_value(error.to_string()))?;
        let store_residency = plan_compiled_resource_store_residency(
            &self.contract,
            &layout,
            &allowed_selector_ids,
            maximum_group_bytes,
            upload_alignment,
        )
        .map_err(|error| targeted_component_error_value(error.to_string()))?;
        let component_ids = BTreeSet::from([self.component_id.clone()]);
        let working_set_bytes = parameter_residency
            .working_set
            .transient_state_bytes
            .checked_add(parameter_residency.working_set.activation_headroom_bytes)
            .ok_or_else(|| {
                targeted_component_error_value("targeted working-set accounting overflowed")
            })?;
        let pending_fixed_bytes = working_set_bytes
            .checked_add(
                store_residency
                    .fixed_device_bytes()
                    .map_err(|error| targeted_component_error_value(error.to_string()))?,
            )
            .ok_or_else(|| {
                targeted_component_error_value("targeted fixed-residency accounting overflowed")
            })?;
        let admission = device
            .admit_device_local_memory(u64::try_from(pending_fixed_bytes).unwrap_or(u64::MAX))
            .map_err(|error| targeted_component_error_value(error.to_string()))?;
        let safe_dynamic_bytes = usize::try_from(admission.allocatable_bytes).unwrap_or(usize::MAX);
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
        let maximum_ranges_per_group =
            compiled_resource_maximum_ranges_per_group(&self.contract)
                .map_err(|error| targeted_component_error_value(error.to_string()))?;
        let store_id = format!(
            "{}:targeted:{}:{}:{}",
            slice.package_id, self.device_id, self.execution_scope, self.component_id,
        );
        let store = Arc::new(
            VulkanCompiledResourceDeviceStore::new(
                device,
                ResourceResidencyPolicy::DemandRetained,
                store_id.clone(),
                device.physical_device_id(),
                vec![self.device_id.clone()],
                manifest_dir,
                Arc::clone(&self.contract),
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
                store_residency.metadata_device_bytes,
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
                .dynamic_buffers_for_components(device, &self.execution_scope, &component_ids)
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
            execution_scope: self.execution_scope.clone(),
            contract: Arc::clone(&self.contract),
            layout,
            store,
            owner: DeviceResourceResidencyOwnerId::new(format!("{store_id}:session"))
                .map_err(|error| targeted_component_error_value(error.to_string()))?,
        };
        Ok(VulkanResidentTargetedModelPackageDeviceSlice {
            slice,
            demand_context: Some(demand_context),
        })
    }
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
        let capacity = dynamic_state_capacity_activations
            .unwrap_or(runtime_model.package.max_context_activations);
        let tensor_index = Arc::new(runtime_model.load_runtime_tensor_index(manifest_dir)?);
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
            0,
            ResourceResidencyPolicy::DemandRetained,
            &contract,
        )
        .map_err(|error| {
            targeted_component_error_value(format!(
                "failed to plan targeted demand residency: {error}"
            ))
        })?;
        VulkanResidentTargetedModelPackageDeviceSlicePlan::prepare(
            device,
            manifest_dir,
            &runtime_model,
            component_id,
            device_id,
            capacity,
            tensor_index,
            contract,
            residency_plan,
        )?
        .materialize(device, manifest_dir, parameter_pool)
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
