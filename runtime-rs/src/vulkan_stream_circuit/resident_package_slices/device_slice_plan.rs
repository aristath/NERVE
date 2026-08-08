#[derive(Clone)]
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
    targeted_output: Option<VulkanResidentTargetedOutputTransducerPlan>,
}

#[derive(Clone)]
struct VulkanResidentTargetedOutputTransducerPlan {
    parameter_plan: VulkanPermanentParameterBufferPlan,
    embedding_norm_spirv_words: Vec<u32>,
    embedding_norm_batch_spirv_words: Vec<u32>,
    projection_spirv_words: Vec<u32>,
    projection_batch_spirv_words: Vec<u32>,
    embedding_norm_batch_lane_tile_width: u32,
    projection_batch_lane_tile_width: u32,
    spec: VulkanResidentOutputTransducerSpec,
}

impl VulkanResidentModelPackageDeviceSlicePlan {
    fn prepare(
        device: &VulkanComputeDevice,
        manifest_dir: &Path,
        runtime_model: &VulkanResidentRuntimeModel,
        resource_contract: &CompiledResourceResidencyContract,
        tensor_index: &TensorIndex,
        device_id: &str,
        capacity: usize,
    ) -> Result<Self, VulkanResidentTokenModelPackageError> {
        if capacity == 0 {
            return Err(VulkanResidentTokenModelPackageError::new(
                "resident dynamic state capacity must be at least 1 activation",
            ));
        }
        // The package-level requirement sets are the union of every mandatory
        // shader in the self-contained package.  They are useful for package
        // inspection, but they are not a placement constraint: a heterogeneous
        // device only has to execute the shaders assigned to its slice.  Every
        // primary, batch, and transducer SPIR-V module is validated against the
        // actual device when its compute pipeline is created below.  Applying
        // the package union here would incorrectly reject compatible slices
        // (for example, a BF16 input transducer on a device that does not expose
        // the FP8 extensions needed by layers hosted elsewhere).
        validate_component_executions(
            &runtime_model.package.package_id,
            &runtime_model.component_executions,
        )?;
        let executable_circuit_graph = runtime_model.executable_circuit_graph()?;

        let (resource_plan, _placement_plan, placed_plan) =
            plan_resident_package_placed_stream_circuit_with_tensor_index(
                device_id,
                &runtime_model.placement,
                &executable_circuit_graph,
                manifest_dir,
                tensor_index,
                runtime_model.package.activation_element_bytes,
            )?;
        let hosted_component_count = placed_plan.binding_plan.circuits.len();
        let output_component_id = runtime_model
            .package
            .output_transducer
            .spec
            .transducer_id
            .as_str();
        let hosts_targeted_output = runtime_model
            .placement
            .device_for_component(output_component_id)
            == device_id;
        if hosted_component_count == 0 && !hosts_targeted_output {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "resident model package {:?} has no executable runtime boundary assigned to device {device_id:?}",
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
                resource_contract,
                runtime_model.execution_scope.clone(),
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
        let targeted_output = hosts_targeted_output
            .then(|| {
                Ok(VulkanResidentTargetedOutputTransducerPlan {
                    parameter_plan:
                        VulkanPermanentParameterBufferPlan::from_transducer_parameters_for(
                            device_id,
                            &resource_plan,
                            Some(tensor_index),
                            output_component_id,
                        )
                        .map_err(|error| {
                            VulkanResidentTokenModelPackageError::new(format!(
                                "failed to plan targeted output-transducer parameters: {error}"
                            ))
                        })?,
                    embedding_norm_spirv_words: load_required_resident_model_package_shader(
                        manifest_dir,
                        &runtime_model
                            .package
                            .output_transducer
                            .embedding_norm_shader_path,
                    )?,
                    embedding_norm_batch_spirv_words: load_required_resident_model_package_shader(
                        manifest_dir,
                        &runtime_model
                            .package
                            .output_transducer
                            .embedding_norm_batch_shader_path,
                    )?,
                    projection_spirv_words: load_required_resident_model_package_shader(
                        manifest_dir,
                        &runtime_model
                            .package
                            .output_transducer
                            .projection_shader_path,
                    )?,
                    projection_batch_spirv_words: load_required_resident_model_package_shader(
                        manifest_dir,
                        &runtime_model
                            .package
                            .output_transducer
                            .projection_batch_shader_path,
                    )?,
                    embedding_norm_batch_lane_tile_width: runtime_model
                        .package
                        .output_transducer
                        .embedding_norm_batch_lane_tile_width,
                    projection_batch_lane_tile_width: runtime_model
                        .package
                        .output_transducer
                        .projection_batch_lane_tile_width,
                    spec: runtime_model.package.output_transducer.spec.clone(),
                })
            })
            .transpose()?;

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
            targeted_output,
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
        let targeted_output = self
            .targeted_output
            .map(|output| {
                let parameter_buffers = Arc::new(match parameter_pool {
                    Some(pool) => output
                        .parameter_plan
                        .allocate_and_load_from_pool(tensor_index, pool)
                        .map_err(|error| {
                            VulkanResidentTokenModelPackageError::new(format!(
                                "failed to acquire pooled targeted output-transducer parameters: {error}"
                            ))
                        })?,
                    None => {
                        let buffers = output
                            .parameter_plan
                            .allocate_buffers(device)
                            .map_err(|error| {
                                VulkanResidentTokenModelPackageError::new(format!(
                                    "failed to allocate targeted output-transducer parameters: {error}"
                                ))
                            })?;
                        buffers
                            .load_from_tensor_index(tensor_index)
                            .map_err(|error| {
                                VulkanResidentTokenModelPackageError::new(format!(
                                    "failed to load targeted output-transducer parameters: {error}"
                                ))
                            })?;
                        buffers
                    }
                });
                Ok(VulkanResidentTargetedOutputTransducerResources {
                    parameter_buffers,
                    embedding_norm_spirv_words:
                        output.embedding_norm_spirv_words,
                    embedding_norm_batch_spirv_words:
                        output.embedding_norm_batch_spirv_words,
                    projection_spirv_words:
                        output.projection_spirv_words,
                    projection_batch_spirv_words:
                        output.projection_batch_spirv_words,
                    embedding_norm_batch_lane_tile_width:
                        output.embedding_norm_batch_lane_tile_width,
                    projection_batch_lane_tile_width:
                        output.projection_batch_lane_tile_width,
                    spec: output.spec,
                })
            })
            .transpose()?;

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
            targeted_output,
        })
    }
}
