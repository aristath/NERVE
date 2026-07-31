pub struct VulkanResidentOutputTransducerRunner {
    pub transducer_id: String,
    pub input_signal_id: String,
    pub logits_byte_capacity: usize,
    pub dispatch_count: usize,
    pub total_descriptor_count: usize,
    pub total_push_constant_byte_count: u32,
    normalized_frame_buffer: VulkanResidentBuffer,
    logits_buffer: VulkanResidentBuffer,
    embedding_norm_dispatch: VulkanResidentKernelDispatch,
    tied_projection_dispatch: VulkanResidentKernelDispatch,
    sequence: VulkanResidentKernelSequence,
    node_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanResidentOutputTransducerSpec {
    pub transducer_id: String,
    pub input_signal_id: String,
    pub node_ids: Vec<String>,
    pub norm_parameter_tensor: String,
    pub norm_parameter_dtype: String,
    pub norm_parameter_shape: Vec<usize>,
    pub norm_parameter_byte_capacity: usize,
    pub projection_parameter_tensor: String,
    pub projection_parameter_dtype: String,
    pub projection_parameter_shape: Vec<usize>,
    pub projection_parameter_byte_capacity: usize,
    pub projection_scale_parameter_tensor: Option<String>,
    pub projection_scale_parameter_dtype: Option<String>,
    pub projection_scale_parameter_shape: Option<Vec<usize>>,
    pub projection_scale_parameter_byte_capacity: Option<usize>,
    pub input_frame_byte_capacity: usize,
    pub normalized_frame_byte_capacity: usize,
    pub logits_byte_capacity: usize,
    pub projection_workgroup_count_x: u32,
    pub norm_local_size_x: u32,
    pub projection_local_size_x: u32,
}

impl VulkanResidentOutputTransducerRunner {
    pub fn from_mounted_output_transducer(
        device: &VulkanComputeDevice,
        mounted: &VulkanMountedPlacedStreamCircuit,
        transducer_parameter_buffers: &VulkanPermanentParameterBuffers,
        embedding_norm_spirv_words: &[u32],
        tied_projection_spirv_words: &[u32],
        spec: &VulkanResidentOutputTransducerSpec,
    ) -> Result<Self, VulkanResidentOutputTransducerRunnerError> {
        let output_frame = mounted
            .boundary_io
            .output_buffer(&spec.input_signal_id)
            .ok_or_else(
                || VulkanResidentOutputTransducerRunnerError::MissingModelOutputBuffer {
                    signal_id: spec.input_signal_id.clone(),
                },
            )?;
        Self::from_input_frame(
            device,
            &output_frame.buffer,
            output_frame.byte_capacity,
            transducer_parameter_buffers,
            embedding_norm_spirv_words,
            tied_projection_spirv_words,
            spec,
        )
    }

    pub(crate) fn from_input_frame(
        device: &VulkanComputeDevice,
        input_frame: &VulkanResidentBuffer,
        input_frame_byte_capacity: usize,
        transducer_parameter_buffers: &VulkanPermanentParameterBuffers,
        embedding_norm_spirv_words: &[u32],
        tied_projection_spirv_words: &[u32],
        spec: &VulkanResidentOutputTransducerSpec,
    ) -> Result<Self, VulkanResidentOutputTransducerRunnerError> {
        if input_frame_byte_capacity != spec.input_frame_byte_capacity {
            return Err(
                VulkanResidentOutputTransducerRunnerError::InvalidInputFrameByteCapacity {
                    signal_id: spec.input_signal_id.clone(),
                    byte_capacity: input_frame_byte_capacity,
                    expected_byte_capacity: spec.input_frame_byte_capacity,
                },
            );
        }
        let embedding_norm_weight = transducer_parameter_buffers
            .parameter_buffer(&spec.norm_parameter_tensor)
            .ok_or_else(|| {
                VulkanResidentOutputTransducerRunnerError::MissingTransducerParameterBuffer {
                    tensor: spec.norm_parameter_tensor.clone(),
                }
            })?;
        let embedding_weight = transducer_parameter_buffers
            .parameter_buffer(&spec.projection_parameter_tensor)
            .ok_or_else(|| {
                VulkanResidentOutputTransducerRunnerError::MissingTransducerParameterBuffer {
                    tensor: spec.projection_parameter_tensor.clone(),
                }
            })?;
        let embedding_scale = projection_scale_parameter_buffer(transducer_parameter_buffers, spec)?;
        Self::from_input_frame_with_parameter_allocations(
            device,
            input_frame,
            embedding_norm_weight,
            embedding_weight,
            embedding_scale,
            embedding_norm_spirv_words,
            tied_projection_spirv_words,
            spec,
        )
    }

    fn from_input_frame_with_parameter_allocations(
        device: &VulkanComputeDevice,
        input_frame: &VulkanResidentBuffer,
        embedding_norm_weight: &VulkanPermanentParameterBufferAllocation,
        embedding_weight: &VulkanPermanentParameterBufferAllocation,
        embedding_scale: Option<&VulkanPermanentParameterBufferAllocation>,
        embedding_norm_spirv_words: &[u32],
        tied_projection_spirv_words: &[u32],
        spec: &VulkanResidentOutputTransducerSpec,
    ) -> Result<Self, VulkanResidentOutputTransducerRunnerError> {
        validate_output_projection_weight(embedding_weight, spec)?;
        validate_output_projection_scale(embedding_scale, spec)?;
        validate_output_embedding_norm_weight(embedding_norm_weight, spec)?;

        let normalized_frame_buffer =
            device.create_resident_buffer(spec.normalized_frame_byte_capacity)?;
        let logits_buffer = device.create_resident_buffer(spec.logits_byte_capacity)?;

        let embedding_norm_bindings = [
            VulkanResidentKernelBufferBinding::new(
                0,
                input_frame,
                spec.input_frame_byte_capacity,
            )
            .with_access(VulkanResidentKernelBufferAccess::Read),
            VulkanResidentKernelBufferBinding::new(
                1,
                &normalized_frame_buffer,
                spec.normalized_frame_byte_capacity,
            )
            .with_access(VulkanResidentKernelBufferAccess::Write),
            VulkanResidentKernelBufferBinding::new(
                2,
                &embedding_norm_weight.buffer,
                embedding_norm_weight.byte_capacity,
            )
            .with_access(VulkanResidentKernelBufferAccess::Read),
        ];
        let embedding_norm_dispatch = device.create_resident_kernel_dispatch(
            embedding_norm_spirv_words,
            &embedding_norm_bindings,
            1,
            spec.norm_local_size_x,
            0,
        )?;

        let projection_workgroup_count_x = spec.projection_workgroup_count_x;
        if projection_workgroup_count_x == 0 {
            return Err(VulkanResidentOutputTransducerRunnerError::InvalidProjectionWorkgroupCount);
        }
        let mut tied_projection_bindings = vec![
            VulkanResidentKernelBufferBinding::new(
                0,
                &normalized_frame_buffer,
                spec.normalized_frame_byte_capacity,
            )
            .with_access(VulkanResidentKernelBufferAccess::Read),
            VulkanResidentKernelBufferBinding::new(
                1,
                &embedding_weight.buffer,
                embedding_weight.byte_capacity,
            )
            .with_access(VulkanResidentKernelBufferAccess::Read),
            VulkanResidentKernelBufferBinding::new(2, &logits_buffer, spec.logits_byte_capacity)
                .with_access(VulkanResidentKernelBufferAccess::Write),
        ];
        if let Some(scale) = embedding_scale {
            tied_projection_bindings.push(
                VulkanResidentKernelBufferBinding::new(3, &scale.buffer, scale.byte_capacity)
                    .with_access(VulkanResidentKernelBufferAccess::Read),
            );
        }
        let tied_projection_dispatch = device.create_resident_kernel_dispatch(
            tied_projection_spirv_words,
            &tied_projection_bindings,
            projection_workgroup_count_x,
            spec.projection_local_size_x,
            0,
        )?;

        let total_descriptor_count = embedding_norm_dispatch
            .descriptor_count()
            .checked_add(tied_projection_dispatch.descriptor_count())
            .ok_or(VulkanResidentOutputTransducerRunnerError::DescriptorCountOverflow)?;
        let total_push_constant_byte_count = embedding_norm_dispatch
            .push_constant_byte_count()
            .checked_add(tied_projection_dispatch.push_constant_byte_count())
            .ok_or(VulkanResidentOutputTransducerRunnerError::PushConstantByteCountOverflow)?;

        Ok(Self {
            transducer_id: spec.transducer_id.clone(),
            input_signal_id: spec.input_signal_id.clone(),
            logits_byte_capacity: spec.logits_byte_capacity,
            dispatch_count: 2,
            total_descriptor_count,
            total_push_constant_byte_count,
            normalized_frame_buffer,
            logits_buffer,
            embedding_norm_dispatch,
            tied_projection_dispatch,
            sequence: device.create_resident_kernel_sequence()?,
            node_ids: spec.node_ids.clone(),
        })
    }

    fn from_mounted_output_transducer_with_parameter_allocations(
        device: &VulkanComputeDevice,
        mounted: &VulkanMountedPlacedStreamCircuit,
        embedding_norm_weight: &VulkanPermanentParameterBufferAllocation,
        embedding_weight: &VulkanPermanentParameterBufferAllocation,
        embedding_scale: Option<&VulkanPermanentParameterBufferAllocation>,
        embedding_norm_spirv_words: &[u32],
        tied_projection_spirv_words: &[u32],
        spec: &VulkanResidentOutputTransducerSpec,
    ) -> Result<Self, VulkanResidentOutputTransducerRunnerError> {
        let output_frame = mounted
            .boundary_io
            .output_buffer(&spec.input_signal_id)
            .ok_or_else(
                || VulkanResidentOutputTransducerRunnerError::MissingModelOutputBuffer {
                    signal_id: spec.input_signal_id.clone(),
                },
            )?;
        if output_frame.byte_capacity != spec.input_frame_byte_capacity {
            return Err(
                VulkanResidentOutputTransducerRunnerError::InvalidInputFrameByteCapacity {
                    signal_id: spec.input_signal_id.clone(),
                    byte_capacity: output_frame.byte_capacity,
                    expected_byte_capacity: spec.input_frame_byte_capacity,
                },
            );
        }
        Self::from_input_frame_with_parameter_allocations(
            device,
            &output_frame.buffer,
            embedding_norm_weight,
            embedding_weight,
            embedding_scale,
            embedding_norm_spirv_words,
            tied_projection_spirv_words,
            spec,
        )
    }

    pub fn run(
        &self,
        device: &VulkanComputeDevice,
    ) -> Result<VulkanResidentOutputTransducerRun, VulkanResidentOutputTransducerRunnerError> {
        device.run_resident_kernel_sequence(
            &self.sequence,
            &[
                VulkanResidentKernelSequenceStep::new(&self.embedding_norm_dispatch, &[]),
                VulkanResidentKernelSequenceStep::new(&self.tied_projection_dispatch, &[]),
            ],
        )?;
        Ok(self.completed_run())
    }

    fn completed_run(&self) -> VulkanResidentOutputTransducerRun {
        VulkanResidentOutputTransducerRun {
            transducer_id: self.transducer_id.clone(),
            input_signal_id: self.input_signal_id.clone(),
            dispatch_count: self.dispatch_count,
            node_ids: self.node_ids.clone(),
            descriptor_counts: vec![
                self.embedding_norm_dispatch.descriptor_count(),
                self.tied_projection_dispatch.descriptor_count(),
            ],
            workgroup_counts_x: vec![
                self.embedding_norm_dispatch.workgroup_count_x(),
                self.tied_projection_dispatch.workgroup_count_x(),
            ],
            push_constant_byte_counts: vec![
                self.embedding_norm_dispatch.push_constant_byte_count(),
                self.tied_projection_dispatch.push_constant_byte_count(),
            ],
            logits_byte_capacity: self.logits_byte_capacity,
        }
    }

    pub fn read_logits_bytes(&self, len: usize) -> Result<Vec<u8>, VulkanError> {
        self.logits_buffer.read_bytes(len)
    }

    pub fn logits_buffer(&self) -> &VulkanResidentBuffer {
        &self.logits_buffer
    }

    pub fn normalized_frame_buffer(&self) -> &VulkanResidentBuffer {
        &self.normalized_frame_buffer
    }

    pub fn read_normalized_frame_bytes(&self, len: usize) -> Result<Vec<u8>, VulkanError> {
        self.normalized_frame_buffer.read_bytes(len)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentOutputTransducerRun {
    pub transducer_id: String,
    pub input_signal_id: String,
    pub dispatch_count: usize,
    pub node_ids: Vec<String>,
    pub descriptor_counts: Vec<usize>,
    pub workgroup_counts_x: Vec<u32>,
    pub push_constant_byte_counts: Vec<u32>,
    pub logits_byte_capacity: usize,
}

#[derive(Debug)]
pub enum VulkanResidentOutputTransducerRunnerError {
    MissingTransducerParameterBuffer {
        tensor: String,
    },
    InvalidEmbeddingWeight {
        tensor: String,
        dtype: Option<String>,
        shape: Option<Vec<usize>>,
        byte_capacity: usize,
    },
    InvalidEmbeddingNormWeight {
        tensor: String,
        dtype: Option<String>,
        shape: Option<Vec<usize>>,
        byte_capacity: usize,
    },
    MissingProjectionScaleParameterBuffer {
        tensor: String,
    },
    InvalidProjectionScaleWeight {
        tensor: String,
        dtype: Option<String>,
        shape: Option<Vec<usize>>,
        byte_capacity: usize,
    },
    InvalidProjectionScaleSpec,
    MissingModelOutputBuffer {
        signal_id: String,
    },
    InvalidInputFrameByteCapacity {
        signal_id: String,
        byte_capacity: usize,
        expected_byte_capacity: usize,
    },
    InvalidProjectionWorkgroupCount,
    DescriptorCountOverflow,
    PushConstantByteCountOverflow,
    Vulkan(VulkanError),
}

impl Display for VulkanResidentOutputTransducerRunnerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingTransducerParameterBuffer { tensor } => {
                write!(
                    f,
                    "missing transducer parameter buffer for tensor {tensor:?}"
                )
            }
            Self::InvalidEmbeddingWeight {
                tensor,
                dtype,
                shape,
                byte_capacity,
            } => write!(
                f,
                "output projection tensor {tensor:?} has dtype {dtype:?}, shape {shape:?}, and {byte_capacity} bytes"
            ),
            Self::InvalidEmbeddingNormWeight {
                tensor,
                dtype,
                shape,
                byte_capacity,
            } => write!(
                f,
                "output embedding norm tensor {tensor:?} has dtype {dtype:?}, shape {shape:?}, and {byte_capacity} bytes"
            ),
            Self::MissingProjectionScaleParameterBuffer { tensor } => {
                write!(
                    f,
                    "missing output projection scale parameter buffer for tensor {tensor:?}"
                )
            }
            Self::InvalidProjectionScaleWeight {
                tensor,
                dtype,
                shape,
                byte_capacity,
            } => write!(
                f,
                "output projection scale tensor {tensor:?} has dtype {dtype:?}, shape {shape:?}, and {byte_capacity} bytes"
            ),
            Self::InvalidProjectionScaleSpec => f.write_str(
                "output projection scale tensor declaration must provide either all metadata fields or none",
            ),
            Self::MissingModelOutputBuffer { signal_id } => {
                write!(f, "missing model output boundary buffer {signal_id:?}")
            }
            Self::InvalidInputFrameByteCapacity {
                signal_id,
                byte_capacity,
                expected_byte_capacity,
            } => write!(
                f,
                "model output boundary buffer {signal_id:?} has {byte_capacity} bytes, expected {expected_byte_capacity}"
            ),
            Self::InvalidProjectionWorkgroupCount => {
                f.write_str("output transducer projection workgroup count must be nonzero")
            }
            Self::DescriptorCountOverflow => {
                f.write_str("output transducer descriptor count overflowed")
            }
            Self::PushConstantByteCountOverflow => {
                f.write_str("output transducer push constant byte count overflowed")
            }
            Self::Vulkan(error) => Display::fmt(error, f),
        }
    }
}

impl Error for VulkanResidentOutputTransducerRunnerError {}

impl From<VulkanError> for VulkanResidentOutputTransducerRunnerError {
    fn from(error: VulkanError) -> Self {
        Self::Vulkan(error)
    }
}

fn validate_output_projection_weight(
    allocation: &VulkanPermanentParameterBufferAllocation,
    spec: &VulkanResidentOutputTransducerSpec,
) -> Result<(), VulkanResidentOutputTransducerRunnerError> {
    if allocation.parameter.dtype.as_deref() != Some(spec.projection_parameter_dtype.as_str())
        || allocation.parameter.shape.as_deref() != Some(spec.projection_parameter_shape.as_slice())
        || allocation.byte_capacity != spec.projection_parameter_byte_capacity
    {
        return Err(
            VulkanResidentOutputTransducerRunnerError::InvalidEmbeddingWeight {
                tensor: allocation.parameter.tensor.clone(),
                dtype: allocation.parameter.dtype.clone(),
                shape: allocation.parameter.shape.clone(),
                byte_capacity: allocation.byte_capacity,
            },
        );
    }
    Ok(())
}

fn projection_scale_parameter_buffer<'a>(
    buffers: &'a VulkanPermanentParameterBuffers,
    spec: &VulkanResidentOutputTransducerSpec,
) -> Result<
    Option<&'a VulkanPermanentParameterBufferAllocation>,
    VulkanResidentOutputTransducerRunnerError,
> {
    let Some(tensor) = spec.projection_scale_parameter_tensor.as_ref() else {
        return Ok(None);
    };
    buffers.parameter_buffer(tensor).map(Some).ok_or_else(|| {
        VulkanResidentOutputTransducerRunnerError::MissingProjectionScaleParameterBuffer {
            tensor: tensor.clone(),
        }
    })
}

fn validate_output_projection_scale(
    allocation: Option<&VulkanPermanentParameterBufferAllocation>,
    spec: &VulkanResidentOutputTransducerSpec,
) -> Result<(), VulkanResidentOutputTransducerRunnerError> {
    match (
        spec.projection_scale_parameter_tensor.as_ref(),
        spec.projection_scale_parameter_dtype.as_ref(),
        spec.projection_scale_parameter_shape.as_ref(),
        spec.projection_scale_parameter_byte_capacity,
    ) {
        (Some(tensor), Some(dtype), Some(shape), Some(byte_capacity)) => {
            let allocation = allocation.ok_or_else(|| {
                VulkanResidentOutputTransducerRunnerError::MissingProjectionScaleParameterBuffer {
                    tensor: tensor.clone(),
                }
            })?;
            if allocation.parameter.dtype.as_ref() != Some(dtype)
                || allocation.parameter.shape.as_ref() != Some(shape)
                || allocation.byte_capacity != byte_capacity
            {
                return Err(
                    VulkanResidentOutputTransducerRunnerError::InvalidProjectionScaleWeight {
                        tensor: allocation.parameter.tensor.clone(),
                        dtype: allocation.parameter.dtype.clone(),
                        shape: allocation.parameter.shape.clone(),
                        byte_capacity: allocation.byte_capacity,
                    },
                );
            }
        }
        (None, None, None, None) => {
            if let Some(allocation) = allocation {
                return Err(
                    VulkanResidentOutputTransducerRunnerError::InvalidProjectionScaleWeight {
                        tensor: allocation.parameter.tensor.clone(),
                        dtype: allocation.parameter.dtype.clone(),
                        shape: allocation.parameter.shape.clone(),
                        byte_capacity: allocation.byte_capacity,
                    },
                );
            }
        }
        _ => {
            return Err(
                VulkanResidentOutputTransducerRunnerError::InvalidProjectionScaleSpec,
            );
        }
    }
    Ok(())
}

fn validate_output_embedding_norm_weight(
    allocation: &VulkanPermanentParameterBufferAllocation,
    spec: &VulkanResidentOutputTransducerSpec,
) -> Result<(), VulkanResidentOutputTransducerRunnerError> {
    if allocation.parameter.dtype.as_deref() != Some(spec.norm_parameter_dtype.as_str())
        || allocation.parameter.shape.as_deref() != Some(spec.norm_parameter_shape.as_slice())
        || allocation.byte_capacity != spec.norm_parameter_byte_capacity
    {
        return Err(
            VulkanResidentOutputTransducerRunnerError::InvalidEmbeddingNormWeight {
                tensor: allocation.parameter.tensor.clone(),
                dtype: allocation.parameter.dtype.clone(),
                shape: allocation.parameter.shape.clone(),
                byte_capacity: allocation.byte_capacity,
            },
        );
    }
    Ok(())
}
