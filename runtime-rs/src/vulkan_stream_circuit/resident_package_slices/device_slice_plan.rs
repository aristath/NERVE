struct VulkanResidentModelPackageDeviceSlicePlan {
    package_id: String,
    device_id: String,
    dynamic_state_capacity_activations: usize,
    hosted_component_count: usize,
    incoming_edge_count: usize,
    outgoing_edge_count: usize,
    placed_plan: VulkanPlacedStreamCircuitPlan,
    prepared_plan: VulkanPreparedDispatchPlan,
    physical_residency_schedule: VulkanPhysicalResidencySchedule,
    loaded_manifest: VulkanLoadedReusableKernelArtifactManifest,
    batch_kernels: Vec<VulkanResidentComponentBatchKernelArtifact>,
}

impl VulkanResidentModelPackageDeviceSlicePlan {
    fn prepare(
        device: &VulkanComputeDevice,
        manifest_dir: &Path,
        runtime_model: &VulkanResidentRuntimeModel,
        tensor_index: &TensorIndex,
        device_id: &str,
        capacity: usize,
    ) -> Result<Self, VulkanResidentTokenModelPackageError> {
        if capacity == 0 {
            return Err(VulkanResidentTokenModelPackageError::new(
                "resident dynamic state capacity must be at least 1 activation",
            ));
        }
        let missing_device_extensions = runtime_model
            .package
            .required_vulkan_device_extensions
            .iter()
            .filter(|extension| !device.has_enabled_device_extension(extension))
            .cloned()
            .collect::<Vec<_>>();
        if !missing_device_extensions.is_empty() {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "resident model package {:?} requires unavailable Vulkan device extensions: {}",
                runtime_model.package.package_id,
                missing_device_extensions.join(", ")
            )));
        }
        let missing_device_features = runtime_model
            .package
            .required_vulkan_features
            .iter()
            .filter(|feature| !device.has_enabled_shader_feature(**feature))
            .map(|feature| feature.label())
            .collect::<Vec<_>>();
        if !missing_device_features.is_empty() {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "resident model package {:?} requires Vulkan features that are not enabled on the logical device: {}",
                runtime_model.package.package_id,
                missing_device_features.join(", ")
            )));
        }
        let missing_subgroup_operations = runtime_model
            .package
            .required_vulkan_subgroup_operations
            .iter()
            .filter(|operation| !device.supports_subgroup_operation(**operation))
            .map(|operation| operation.label())
            .collect::<Vec<_>>();
        if !missing_subgroup_operations.is_empty() {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "resident model package {:?} requires unsupported Vulkan subgroup operations: {}",
                runtime_model.package.package_id,
                missing_subgroup_operations.join(", ")
            )));
        }
        validate_component_executions(
            &runtime_model.package.package_id,
            &runtime_model.component_executions,
        )?;

        let (_resource_plan, _placement_plan, placed_plan) =
            plan_resident_package_placed_stream_circuit_with_tensor_index(
                device_id,
                &runtime_model.placement,
                &runtime_model.circuit_graph,
                manifest_dir,
                tensor_index,
                runtime_model.package.activation_element_bytes,
            )?;
        let hosted_component_count = placed_plan.binding_plan.circuits.len();
        if hosted_component_count == 0 {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "resident model package {:?} has no components assigned to device {device_id:?}",
                runtime_model.package.package_id
            )));
        }
        let reusable_manifest = resident_package_reusable_kernel_manifest(&placed_plan);
        let prepared_plan = placed_plan
            .prepared_dispatch_plan(&reusable_manifest, capacity)
            .map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to prepare Vulkan dispatch plan for device {device_id:?}: {error}"
                ))
            })?;
        validate_component_executions_cover_prepared_dispatches(
            &runtime_model.package.package_id,
            &runtime_model.component_executions,
            &prepared_plan,
        )?;
        let physical_residency_schedule =
            VulkanPhysicalResidencySchedule::from_prepared_dispatch_plan(
                &runtime_model.package.resource_residency,
                "target",
                &prepared_plan,
            )
            .map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to lower physical residency checkpoints for device {device_id:?}: {error}"
                ))
            })?;
        let component_kernel_shaders =
            resident_package_component_kernel_shader_refs_for_prepared_dispatches(
                &runtime_model.component_executions,
                &prepared_plan,
            );
        let loaded_manifest = loaded_kernel_pack_from_package_shader_refs(
            manifest_dir,
            &placed_plan,
            &prepared_plan,
            &component_kernel_shaders,
        )?;
        let batch_kernels = load_resident_component_batch_kernels(
            device,
            manifest_dir,
            &runtime_model.component_executions,
            &prepared_plan,
        )?;

        Ok(Self {
            package_id: runtime_model.package.package_id.clone(),
            device_id: device_id.to_string(),
            dynamic_state_capacity_activations: capacity,
            hosted_component_count,
            incoming_edge_count: placed_plan.placed_resident_plan.incoming_edges.len(),
            outgoing_edge_count: placed_plan.placed_resident_plan.outgoing_edges.len(),
            placed_plan,
            prepared_plan,
            physical_residency_schedule,
            loaded_manifest,
            batch_kernels,
        })
    }

    fn materialize(
        self,
        device: &VulkanComputeDevice,
        tensor_index: &TensorIndex,
        excluded_tensors: &BTreeSet<String>,
        parameter_pool: Option<&VulkanResidentBufferPool>,
    ) -> Result<VulkanResidentModelPackageDeviceSlice, VulkanResidentTokenModelPackageError> {
        let dynamically_addressed_tensors = self
            .placed_plan
            .binding_plan
            .selected_parameter_tensors()
            .map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to separate permanent and dynamic parameters for device {:?}: {error}",
                    self.device_id
                ))
            })?;
        let excluded_tensors = excluded_tensors
            .union(&dynamically_addressed_tensors)
            .cloned()
            .collect::<BTreeSet<_>>();
        let parameter_buffer_plan =
            VulkanPermanentParameterBufferPlan::from_placed_resident_plan_excluding_tensors(
                &self.placed_plan.placed_resident_plan,
                &excluded_tensors,
            )
            .map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to create resident parameter buffer plan for device {:?}: {error}",
                    self.device_id
                ))
            })?;
        let parameter_buffers = Arc::new(match parameter_pool {
            Some(pool) => parameter_buffer_plan
                .allocate_and_load_from_pool(
                    tensor_index,
                    pool,
                )
                .map_err(|error| {
                    VulkanResidentTokenModelPackageError::new(
                        format!(
                            "failed to acquire pooled resident model parameters for device {:?}: {error}",
                            self.device_id
                        ),
                    )
                })?,
            None => {
                let buffers = parameter_buffer_plan
                    .allocate_buffers(device)
                    .map_err(|error| {
                        VulkanResidentTokenModelPackageError::new(
                            format!(
                                "failed to allocate resident parameter buffers for device {:?}: {error}",
                                self.device_id
                            ),
                        )
                    })?;
                buffers
                    .load_from_tensor_index(tensor_index)
                    .map_err(|error| {
                        VulkanResidentTokenModelPackageError::new(
                            format!(
                                "failed to load resident model parameters for device {:?}: {error}",
                                self.device_id
                            ),
                        )
                    })?;
                buffers
            }
        });

        Ok(VulkanResidentModelPackageDeviceSlice {
            package_id: self.package_id,
            device_id: self.device_id,
            dynamic_state_capacity_activations: self.dynamic_state_capacity_activations,
            hosted_component_count: self.hosted_component_count,
            incoming_edge_count: self.incoming_edge_count,
            outgoing_edge_count: self.outgoing_edge_count,
            permanent_parameter_count: parameter_buffers.plan.parameter_count,
            permanent_parameter_bytes: parameter_buffers.total_byte_capacity,
            reusable_kernel_word_count: self.loaded_manifest.total_word_count,
            physical_residency_schedule: self.physical_residency_schedule,
            placed_plan: self.placed_plan,
            prepared_plan: self.prepared_plan,
            loaded_manifest: self.loaded_manifest,
            batch_kernels: self.batch_kernels,
            parameter_buffers,
            dynamic_resource_buffers: None,
        })
    }
}
