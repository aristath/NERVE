struct VulkanComponentBatchDispatchStep {
    dispatch: VulkanResidentKernelDispatch,
    push_constants: Vec<VulkanKernelScalarBinding>,
    lane_index: Option<usize>,
    commits_state: bool,
    dispatch_control: VulkanComponentBatchDispatchControl,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VulkanComponentBatchDispatchControl {
    Fixed,
    BatchWidthY,
    Indirect {
        payload: VulkanResidentComponentBatchControlPayload,
        byte_offset: usize,
    },
}

fn component_batch_dispatch_control(
    stage: &VulkanResidentComponentBatchStageArtifact,
) -> Result<VulkanComponentBatchDispatchControl, VulkanError> {
    match (
        stage.dispatch_y_from_batch_width,
        stage.indirect_dispatch_byte_offset,
    ) {
        (true, Some(_)) => Err(VulkanError(format!(
            "component batch stage {:?} cannot combine a batch-width dispatch with an indirect dispatch",
            stage.shader_path
        ))),
        (true, None) => Ok(VulkanComponentBatchDispatchControl::BatchWidthY),
        (false, Some(byte_offset)) => Ok(VulkanComponentBatchDispatchControl::Indirect {
            payload: stage.control.storage_buffer().2,
            byte_offset: byte_offset as usize,
        }),
        (false, None) => Ok(VulkanComponentBatchDispatchControl::Fixed),
    }
}

fn component_batch_control_buffer_access(
    control: VulkanResidentComponentBatchControlSpec,
) -> VulkanResidentKernelBufferAccess {
    match control.access() {
        VulkanResidentComponentBatchControlAccess::Read => {
            VulkanResidentKernelBufferAccess::Read
        }
        VulkanResidentComponentBatchControlAccess::ReadWrite => {
            VulkanResidentKernelBufferAccess::ReadWrite
        }
    }
}

fn component_batch_descriptors_commit_state<'a>(
    usages: impl IntoIterator<Item = &'a VulkanKernelDescriptorUsage>,
) -> bool {
    usages.into_iter().any(|usage| {
        matches!(
            usage,
            VulkanKernelDescriptorUsage::StateWrite | VulkanKernelDescriptorUsage::StateView
        )
    })
}

fn component_batch_stages_replace_push_constants(
    stages: &[VulkanResidentComponentBatchStageArtifact],
    push_constants: &[VulkanKernelScalarBinding],
) -> bool {
    push_constants.iter().all(|binding| {
        binding.name == "expert_start"
            && binding.scalar_type == "u32"
            && binding.source == VulkanKernelScalarSource::PushConstant
            && !stages.is_empty()
            && stages.iter().all(|stage| {
                matches!(
                    stage.control.storage_buffer().2,
                    VulkanResidentComponentBatchControlPayload::WidthExpertStart
                        | VulkanResidentComponentBatchControlPayload::WidthExpertRangeIndirect
                )
            })
    })
}

fn component_batch_stage_bindings<'a>(
    parent: &[VulkanResidentKernelBufferBinding<'a>],
    descriptor_bindings: &[VulkanResidentComponentBatchDescriptorBindingSpec],
    control_binding: u32,
) -> Result<Vec<VulkanResidentKernelBufferBinding<'a>>, VulkanError> {
    if descriptor_bindings.is_empty() {
        if parent.iter().any(|binding| binding.binding == control_binding) {
            return Err(VulkanError(format!(
                "component batch control binding {control_binding} collides with the parent descriptor interface"
            )));
        }
        return Ok(parent
            .iter()
            .map(|binding| VulkanResidentKernelBufferBinding {
                binding: binding.binding,
                buffer: binding.buffer,
                byte_offset: binding.byte_offset,
                byte_len: binding.byte_len,
                access: binding.access,
            })
            .collect());
    }

    let mut source_bindings = BTreeSet::new();
    let mut stage_bindings = BTreeSet::new();
    descriptor_bindings
        .iter()
        .map(|mapping| {
            if !source_bindings.insert(mapping.source_binding) {
                return Err(VulkanError(format!(
                    "component batch stage maps parent descriptor {} more than once",
                    mapping.source_binding
                )));
            }
            if mapping.binding == control_binding {
                return Err(VulkanError(format!(
                    "component batch stage descriptor binding {} collides with its control binding",
                    mapping.binding
                )));
            }
            if !stage_bindings.insert(mapping.binding) {
                return Err(VulkanError(format!(
                    "component batch stage descriptor binding {} is mapped more than once",
                    mapping.binding
                )));
            }
            let source = parent
                .iter()
                .find(|binding| binding.binding == mapping.source_binding)
                .ok_or_else(|| {
                    VulkanError(format!(
                        "component batch stage references absent parent descriptor {}",
                        mapping.source_binding
                    ))
                })?;
            Ok(VulkanResidentKernelBufferBinding {
                binding: mapping.binding,
                buffer: source.buffer,
                byte_offset: source.byte_offset,
                byte_len: source.byte_len,
                access: source.access,
            })
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VulkanComponentBatchExecutionMode {
    IndependentStreams,
    CausalSequence,
    ParallelBlock,
}

fn select_component_batch_kernel_artifact<'a>(
    artifacts: &'a [VulkanResidentComponentBatchKernelArtifact],
    component_id: &str,
    node_id: &str,
    execution_mode: VulkanComponentBatchExecutionMode,
    lane_capacity: usize,
) -> Option<&'a VulkanResidentComponentBatchKernelArtifact> {
    select_component_batch_kernel_artifact_where(
        artifacts,
        component_id,
        node_id,
        execution_mode,
        lane_capacity,
        |_| true,
    )
}

fn select_component_batch_kernel_artifact_where<'a>(
    artifacts: &'a [VulkanResidentComponentBatchKernelArtifact],
    component_id: &str,
    node_id: &str,
    execution_mode: VulkanComponentBatchExecutionMode,
    lane_capacity: usize,
    compatible: impl Fn(&VulkanResidentComponentBatchKernelArtifact) -> bool,
) -> Option<&'a VulkanResidentComponentBatchKernelArtifact> {
    artifacts
        .iter()
        .filter(|artifact| {
            artifact.component_id == component_id
                && artifact.node_id == node_id
                && artifact
                    .execution_domain
                    .supports_batch_mode(execution_mode)
                && (artifact.batch_mode == VulkanResidentComponentKernelBatchMode::WeightShared
                    || execution_mode == VulkanComponentBatchExecutionMode::CausalSequence)
                && artifact.is_compatible_with(execution_mode)
                && compatible(artifact)
        })
        .min_by_key(|artifact| {
            if artifact.batch_mode == VulkanResidentComponentKernelBatchMode::CausalScan {
                (0usize, usize::MAX - artifact.lane_tile_width)
            } else if artifact.lane_tile_width >= lane_capacity {
                (0usize, artifact.lane_tile_width)
            } else {
                (1usize, usize::MAX - artifact.lane_tile_width)
            }
        })
}
