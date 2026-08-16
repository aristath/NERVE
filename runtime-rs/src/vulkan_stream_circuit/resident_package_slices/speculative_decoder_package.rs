pub struct VulkanResidentSpeculativeDecoderModelPackage {
    pub id: String,
    pub device_id: String,
    pub hosted_component_count: usize,
    package: VulkanResidentSpeculativeDecoderPackageSpec,
    device_slice: Arc<VulkanResidentModelPackageDeviceSlice>,
    execution: VulkanResidentSpeculativeDecoderModelExecution,
    demand_residency_context: Option<VulkanDemandResidencyExecutionContext>,
}

enum VulkanResidentSpeculativeDecoderModelExecution {
    Autoregressive {
        additional_parameter_buffers: Option<Arc<VulkanPermanentParameterBuffers>>,
        input_embedding_spec: VulkanResidentInputEmbeddingTransducerSpec,
        input_embedding_spirv_words: Vec<u32>,
        input_embedding_batch_spirv_words: Vec<u32>,
        input_embedding_batch_control: VulkanResidentComponentBatchControlSpec,
        output_norm_spirv_words: Vec<u32>,
        output_projection_spirv_words: Vec<u32>,
    },
    ParallelBlock {
        block_width: usize,
        source_context_tick_offset: i64,
    },
}

struct VulkanResidentSpeculativeDecoderLoadContext<'a> {
    manifest_dir: &'a Path,
    runtime_model: &'a VulkanResidentRuntimeModel,
    capacity: usize,
    tensor_index: &'a TensorIndex,
    shared_additional_parameters: Option<Arc<VulkanPermanentParameterBuffers>>,
    input_embedding_spec: &'a VulkanResidentInputEmbeddingTransducerSpec,
    input_embedding_spirv_words: &'a [u32],
    input_embedding_batch_spirv_words: &'a [u32],
    input_embedding_batch_control: VulkanResidentComponentBatchControlSpec,
    compiled_resource_device_stores:
        &'a BTreeMap<String, Arc<VulkanCompiledResourceDeviceStore>>,
    resource_residency_policy: ResourceResidencyPolicy,
}

fn speculative_decoder_additional_parameter_tensors<'a>(
    input_embedding: &'a VulkanResidentInputEmbeddingTransducerSpec,
    output: &'a VulkanResidentDraftOutputTransducerPackageSpec,
    mut target_has_tensor: impl FnMut(&str) -> bool,
) -> Vec<&'a str> {
    std::iter::once(input_embedding.parameter_tensor.as_str())
        .chain(speculative_decoder_output_parameter_tensors(output))
        .filter(|tensor| !target_has_tensor(tensor))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn speculative_decoder_shared_additional_parameter_tensors<'a>(
    runtime_model: &'a VulkanResidentRuntimeModel,
    mut target_has_tensor: impl FnMut(&str) -> bool,
) -> Vec<&'a str> {
    runtime_model
        .package
        .speculative_decoders
        .iter()
        .filter_map(VulkanResidentSpeculativeDecoderPackageSpec::dedicated_output_transducer)
        .flat_map(|output| {
            speculative_decoder_additional_parameter_tensors(
                &runtime_model.package.input_transducer.spec,
                output,
                &mut target_has_tensor,
            )
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn speculative_decoder_output_parameter_tensors(
    output: &VulkanResidentDraftOutputTransducerPackageSpec,
) -> impl Iterator<Item = &str> {
    [
        Some(output.norm_parameter_tensor.as_str()),
        Some(output.projection_parameter_tensor.as_str()),
        output.projection_scale_parameter_tensor.as_deref(),
    ]
    .into_iter()
    .flatten()
}

impl VulkanResidentSpeculativeDecoderModelPackage {
    #[allow(clippy::too_many_arguments)]
    fn prepare_device_slice_for_physical_planning(
        manifest_dir: &Path,
        target_runtime_model: &VulkanResidentRuntimeModel,
        tensor_index: &TensorIndex,
        decoder: &VulkanResidentSpeculativeDecoderPackageSpec,
        device_id: &str,
        capacity: usize,
    ) -> Result<VulkanResidentModelPackageDeviceSlicePlan, VulkanResidentTokenModelPackageError>
    {
        let draft_runtime_model =
            speculative_decoder_runtime_model(target_runtime_model, decoder, device_id);
        let resource_contract = instantiate_runtime_resource_contract(&draft_runtime_model)
            .map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to instantiate speculative decoder {:?} resource contract: {error}",
                    decoder.id,
                ))
            })?;
        VulkanResidentModelPackageDeviceSlicePlan::prepare_for_physical_planning(
            manifest_dir,
            &draft_runtime_model,
            &resource_contract,
            tensor_index,
            device_id,
            capacity,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_device_slice(
        device: &VulkanComputeDevice,
        manifest_dir: &Path,
        target_runtime_model: &VulkanResidentRuntimeModel,
        tensor_index: &TensorIndex,
        decoder: &VulkanResidentSpeculativeDecoderPackageSpec,
        device_id: &str,
        capacity: usize,
    ) -> Result<VulkanResidentModelPackageDeviceSlicePlan, VulkanResidentTokenModelPackageError>
    {
        let draft_runtime_model =
            speculative_decoder_runtime_model(target_runtime_model, decoder, device_id);
        let resource_contract = instantiate_runtime_resource_contract(&draft_runtime_model)
            .map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to instantiate speculative decoder {:?} resource contract: {error}",
                    decoder.id,
                ))
            })?;
        VulkanResidentModelPackageDeviceSlicePlan::prepare(
            device,
            manifest_dir,
            &draft_runtime_model,
            &resource_contract,
            tensor_index,
            device_id,
            capacity,
        )
    }

    fn from_runtime_model(
        device: &VulkanComputeDevice,
        decoder: &VulkanResidentSpeculativeDecoderPackageSpec,
        device_id: &str,
        device_slice_plan: VulkanResidentModelPackageDeviceSlicePlan,
        context: &VulkanResidentSpeculativeDecoderLoadContext<'_>,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        if device_slice_plan.device_id != device_id
            || device_slice_plan.dynamic_state_capacity_activations != context.capacity
        {
            return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "speculative decoder {:?} prepared slice does not match device {device_id:?} and capacity {}",
                    decoder.id, context.capacity,
                )),
            ));
        }
        let mut device_slice = device_slice_plan
            .materialize(device, context.tensor_index, &BTreeSet::new(), None)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?;
        let selected_tensors = device_slice
            .placed_plan
            .binding_plan
            .selected_parameter_tensors()
            .map_err(|error| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(format!(
                        "failed to inspect speculative decoder {:?} selected parameters: {error}",
                        decoder.id
                    )),
                )
            })?;
        if !selected_tensors.is_empty() {
            let store = context
                .compiled_resource_device_stores
                .get(device_id)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::Package(
                        VulkanResidentTokenModelPackageError::new(format!(
                            "speculative decoder {:?} has no compiled resource store on device {device_id:?}",
                            decoder.id
                        )),
                    )
                })?;
            let component_ids = device_slice
                .placed_plan
                .binding_plan
                .circuits
                .iter()
                .map(|circuit| circuit.component_id.clone())
                .collect::<BTreeSet<_>>();
            let execution_scope = format!("draft:{}", decoder.id);
            device_slice.dynamic_resource_buffers = Some(
                store
                    .dynamic_buffers_for_components(
                        device,
                        &execution_scope,
                        &component_ids,
                    )
                    .map_err(|error| {
                        VulkanResidentInProcessPlacedRuntimeError::Package(
                            VulkanResidentTokenModelPackageError::new(format!(
                                "failed to bind speculative decoder {:?} dynamic resources: {error}",
                                decoder.id
                            )),
                        )
                    })?,
            );
        }
        let device_slice = Arc::new(device_slice);

        let execution = match &decoder.execution_contract {
            VulkanResidentSpeculativeExecutionContract::AutoregressiveFeedback { .. } => {
                let input = decoder.dedicated_input_adapter().ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::Package(
                        VulkanResidentTokenModelPackageError::new(format!(
                            "speculative decoder {:?} has no dedicated input adapter",
                            decoder.id
                        )),
                    )
                })?;
                let output = decoder.dedicated_output_transducer().ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::Package(
                        VulkanResidentTokenModelPackageError::new(format!(
                            "speculative decoder {:?} has no dedicated output transducer",
                            decoder.id
                        )),
                    )
                })?;
                let additional_parameter_buffers = context.shared_additional_parameters.clone();
                let output_norm_spirv_words = load_required_resident_model_package_shader(
                    context.manifest_dir,
                    &output.norm_shader_path,
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?;
                let output_projection_spirv_words = load_required_resident_model_package_shader(
                    context.manifest_dir,
                    &output.projection_shader_path,
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?;
                let mut input_embedding_spec = context.input_embedding_spec.clone();
                input_embedding_spec.transducer_id = input.component_id.clone();
                input_embedding_spec.output_signal_id = input.token_embedding_signal_id.clone();
                VulkanResidentSpeculativeDecoderModelExecution::Autoregressive {
                    additional_parameter_buffers,
                    input_embedding_spec,
                    input_embedding_spirv_words: context.input_embedding_spirv_words.to_vec(),
                    input_embedding_batch_spirv_words: context
                        .input_embedding_batch_spirv_words
                        .to_vec(),
                    input_embedding_batch_control: context.input_embedding_batch_control,
                    output_norm_spirv_words,
                    output_projection_spirv_words,
                }
            }
            VulkanResidentSpeculativeExecutionContract::ParallelBlock {
                block_width,
                source_context_tick_offset,
                ..
            } => {
                VulkanResidentSpeculativeDecoderModelExecution::ParallelBlock {
                    block_width: *block_width,
                    source_context_tick_offset: *source_context_tick_offset,
                }
            }
        };

        let demand_residency_context =
            if context.resource_residency_policy.is_demand_loaded() {
                let store = context
                    .compiled_resource_device_stores
                    .get(device_id)
                    .cloned()
                    .ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::Package(
                            VulkanResidentTokenModelPackageError::new(format!(
                                "speculative decoder {:?} has no demand resource store on device {device_id:?}",
                                decoder.id
                            )),
                        )
                    })?;
                Some(VulkanDemandResidencyExecutionContext {
                    execution_scope: format!("draft:{}", decoder.id),
                    contract: Arc::clone(&store.contract),
                    layout: Arc::clone(&store.layout),
                    store,
                    owner: DeviceResourceResidencyOwnerId::new(format!(
                        "{}:{device_id}:draft:{}",
                        context.runtime_model.package.package_id, decoder.id
                    ))
                    .map_err(|error| {
                        VulkanResidentInProcessPlacedRuntimeError::Package(
                            VulkanResidentTokenModelPackageError::new(error.to_string()),
                        )
                    })?,
                })
            } else {
                None
            };
        Ok(Self {
            id: decoder.id.clone(),
            device_id: device_id.to_string(),
            hosted_component_count: device_slice.hosted_component_count,
            package: decoder.clone(),
            device_slice,
            execution,
            demand_residency_context,
        })
    }

    fn parameter<'a>(
        &'a self,
        target_output_parameters: &'a VulkanPermanentParameterBuffers,
        tensor: &str,
    ) -> Option<&'a VulkanPermanentParameterBufferAllocation> {
        let VulkanResidentSpeculativeDecoderModelExecution::Autoregressive {
            additional_parameter_buffers,
            ..
        } = &self.execution
        else {
            return None;
        };
        additional_parameter_buffers
            .as_deref()
            .and_then(|buffers| buffers.parameter_buffer(tensor))
            .or_else(|| target_output_parameters.parameter_buffer(tensor))
    }

    fn output_transducer_spec(
        &self,
        input_signal_id: String,
    ) -> Result<VulkanResidentOutputTransducerSpec, VulkanResidentTokenModelPackageError> {
        let output = self.package.dedicated_output_transducer().ok_or_else(|| {
            VulkanResidentTokenModelPackageError::new(format!(
                "speculative decoder {:?} does not use a dedicated output transducer",
                self.id
            ))
        })?;
        let component = self
            .package
            .circuit_graph
            .components
            .iter()
            .find(|component| component.component_id == output.component_id)
            .ok_or_else(|| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "speculative decoder {:?} has no output transducer component {:?}",
                    self.id, output.component_id
                ))
            })?;
        let node_ids = component
            .circuit
            .nodes
            .iter()
            .map(|node| node.id.clone())
            .collect::<Vec<_>>();
        if node_ids.len() != 2 {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "speculative decoder {:?} output transducer requires two nodes, found {}",
                self.id,
                node_ids.len()
            )));
        }
        Ok(VulkanResidentOutputTransducerSpec {
            transducer_id: output.component_id.clone(),
            input_signal_id,
            node_ids,
            norm_parameter_tensor: output.norm_parameter_tensor.clone(),
            norm_parameter_dtype: output.norm_parameter_dtype.clone(),
            norm_parameter_shape: output.norm_parameter_shape.clone(),
            norm_parameter_byte_capacity: output.norm_parameter_byte_capacity,
            projection_parameter_tensor: output.projection_parameter_tensor.clone(),
            projection_parameter_dtype: output.projection_parameter_dtype.clone(),
            projection_parameter_shape: output.projection_parameter_shape.clone(),
            projection_parameter_byte_capacity: output.projection_parameter_byte_capacity,
            projection_scale_parameter_tensor: output.projection_scale_parameter_tensor.clone(),
            projection_scale_parameter_dtype: output.projection_scale_parameter_dtype.clone(),
            projection_scale_parameter_shape: output.projection_scale_parameter_shape.clone(),
            projection_scale_parameter_byte_capacity: output
                .projection_scale_parameter_byte_capacity,
            input_frame_byte_capacity: output.input_frame_byte_capacity,
            normalized_frame_byte_capacity: output.output_hidden_byte_capacity,
            logits_byte_capacity: output.logits_byte_capacity,
            projection_workgroup_count_x: output.projection_workgroup_count_x,
            norm_local_size_x: output.norm_local_size_x,
            projection_local_size_x: output.projection_local_size_x,
        })
    }
}
