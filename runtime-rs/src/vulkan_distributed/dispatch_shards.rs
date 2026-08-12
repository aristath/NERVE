#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedDispatchShard {
    pub device_id: String,
    /// Exact selector-local resources owned by this shard. An empty map means
    /// the dispatch uses ordinary contiguous tensor partitioning.
    pub selected_resource_indices: BTreeMap<String, Vec<usize>>,
    /// Logical fragments of selector-local resources owned by this shard.
    /// These are distinct from whole-resource ownership above: every selected
    /// resource may contribute one non-overlapping fragment to every TP shard.
    pub selected_resource_fragments:
        BTreeMap<String, Vec<VulkanDistributedSelectedResourceFragmentPlan>>,
    pub row_start: usize,
    pub row_count: usize,
    pub workgroup_count_x: u32,
    pub base_workgroup_z: u32,
    pub input_range: VulkanDistributedActivationRange,
    pub auxiliary_input_ranges: Vec<VulkanDistributedActivationRange>,
    pub output_byte_offset: usize,
    pub output_byte_count: usize,
    pub parameters: Vec<VulkanDistributedParameterFragment>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedActivationRange {
    pub byte_offset: usize,
    pub byte_count: usize,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedParameterFragment {
    pub binding: usize,
    pub tensor: String,
    pub byte_offset: usize,
    pub byte_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedSelectedResourceFragmentPlan {
    pub resource_index: usize,
    pub atomic_group_id: String,
    pub logical_start: usize,
    pub logical_count: usize,
    pub parameters: Vec<VulkanDistributedSelectedResourceParameterFragmentPlan>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedSelectedResourceParameterFragmentPlan {
    pub parameter_slot: usize,
    pub resource_id: String,
    pub resource_byte_count: usize,
    pub byte_offset: usize,
    pub byte_count: usize,
}
