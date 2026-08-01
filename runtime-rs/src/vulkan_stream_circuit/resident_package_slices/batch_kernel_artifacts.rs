struct VulkanResidentComponentBatchKernelArtifact {
    component_id: String,
    node_id: String,
    execution_domain: VulkanResidentComponentKernelExecutionDomain,
    batch_mode: VulkanResidentComponentKernelBatchMode,
    lane_tile_width: usize,
    independent_candidate_compatible: bool,
    causal_sequence_compatible: bool,
    parallel_block_compatible: bool,
    device_requirements: VulkanResidentVulkanDeviceRequirements,
    stages: Vec<VulkanResidentComponentBatchStageArtifact>,
}

impl VulkanResidentComponentBatchKernelArtifact {
    fn is_compatible_with(&self, mode: VulkanComponentBatchExecutionMode) -> bool {
        match mode {
            VulkanComponentBatchExecutionMode::IndependentStreams => {
                self.independent_candidate_compatible
            }
            VulkanComponentBatchExecutionMode::CausalSequence => {
                self.causal_sequence_compatible
            }
            VulkanComponentBatchExecutionMode::ParallelBlock => {
                self.parallel_block_compatible
            }
        }
    }
}

struct VulkanResidentComponentBatchStageArtifact {
    shader_path: String,
    spirv_words: Vec<u32>,
    local_size_x: u32,
    workgroup_count_x: u32,
    descriptor_bindings: Vec<VulkanResidentComponentBatchDescriptorBindingSpec>,
    state_snapshot_binding: Option<u32>,
    control: VulkanResidentComponentBatchControlSpec,
    indirect_dispatch_byte_offset: Option<u32>,
    dispatch_y_from_batch_width: bool,
}
