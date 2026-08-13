use crate::vulkan_compute::{
    VulkanResidentDistributedExecutionKind, VulkanResidentDistributedExecutionPhase,
    record_vulkan_resident_distributed_execution_submission,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VulkanPhysicalExecutionIslandKind {
    TensorParallel,
    WholeExpertParallel,
    IntraExpertTensorParallel,
    Hybrid,
}

pub(crate) fn physical_execution_island_kind(
    island: &VulkanPhysicalExecutionIslandPlan,
) -> Option<VulkanPhysicalExecutionIslandKind> {
    use nerve_execution_contracts::ExecutionStrategy;

    let saw_tensor_parallel = island
        .dispatches
        .iter()
        .any(|dispatch| dispatch.execution_strategy == ExecutionStrategy::TensorParallel);
    let saw_whole_expert = island
        .dispatches
        .iter()
        .any(|dispatch| dispatch.execution_strategy == ExecutionStrategy::ExpertParallel);
    let saw_intra_expert = island.dispatches.iter().any(|dispatch| {
        dispatch.execution_strategy == ExecutionStrategy::TensorParallelExpert
    });
    match (saw_tensor_parallel, saw_whole_expert, saw_intra_expert) {
        (true, false, false) => Some(VulkanPhysicalExecutionIslandKind::TensorParallel),
        (false, true, false) => Some(VulkanPhysicalExecutionIslandKind::WholeExpertParallel),
        (false, false, true) => {
            Some(VulkanPhysicalExecutionIslandKind::IntraExpertTensorParallel)
        }
        (false, false, false) => None,
        _ => Some(VulkanPhysicalExecutionIslandKind::Hybrid),
    }
}

pub(crate) fn record_vulkan_physical_execution_island_submission(
    phase: VulkanResidentDistributedExecutionPhase,
    island: &VulkanPhysicalExecutionIslandPlan,
) {
    let kind = match physical_execution_island_kind(island) {
        Some(VulkanPhysicalExecutionIslandKind::TensorParallel) => {
            VulkanResidentDistributedExecutionKind::TensorParallel
        }
        Some(VulkanPhysicalExecutionIslandKind::WholeExpertParallel) => {
            VulkanResidentDistributedExecutionKind::WholeExpertParallel
        }
        Some(VulkanPhysicalExecutionIslandKind::IntraExpertTensorParallel) => {
            VulkanResidentDistributedExecutionKind::IntraExpertTensorParallel
        }
        Some(VulkanPhysicalExecutionIslandKind::Hybrid) => {
            VulkanResidentDistributedExecutionKind::Hybrid
        }
        None => return,
    };
    record_vulkan_resident_distributed_execution_submission(
        phase,
        kind,
        island.leader().shards.len(),
    );
}
