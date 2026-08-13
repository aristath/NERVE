use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::io;
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use smallvec::SmallVec;

use crate::critical_path::{
    RuntimeCriticalPathPhase, record_runtime_critical_path_device_duration,
    runtime_critical_path_device_detail_enabled, runtime_critical_path_device_detail_sample_scope,
    runtime_critical_path_device_phase_scope, runtime_critical_path_span,
};
#[cfg(test)]
use crate::critical_path::{reset_runtime_critical_path_counters, runtime_critical_path_report};
use crate::execution_schedule::{
    RuntimeExecutionCost, RuntimeExecutionQuantumCalibrator, RuntimeExecutionRegion,
};
use crate::stream_circuit::{
    CapacityPackedPlacementComponent, CapacityPackedPlacementDevice, CircuitNode,
    CircuitParamsArtifact, CircuitPort, CircuitRuntimeRole, CircuitStateArtifact,
    ComponentEdgePlacement, EdgeTransport, LOWERED_EXECUTION_GRAPH_SCHEMA, LoweredCircuitRef,
    LoweredExecutionGraph, LoweredExecutionGraphGraph, LoweredExecutionGraphSource,
    LoweredExecutionGraphSummary, RUNTIME_DEFAULT_LOGICAL_DEVICE_ID, ResolvedCircuitArtifact,
    ResolvedLoweredExecutionGraph, RuntimeSelectedResourceCountReport,
    RuntimeSelectionCoverageReport, RuntimeSelectionDomainCoverageReport,
    RuntimeSparseMoeWorkReport, StreamCircuit, StreamCircuitConnection, StreamCircuitGraphBoundary,
    StreamCircuitGraphSourceTap, StreamCircuitGraphSourceTapInstanceSelection,
    StreamCircuitNodeInstanceStatePolicy, StreamCircuitPlacementPlan, StreamCircuitPlacementSpec,
    StreamCircuitRuntimeGraph, capacity_packed_component_placement,
};
use crate::stream_plan::{
    CircuitActivationPlan, PlannedNode, PlannedParameterResource, PlannedPort,
    PlannedPredictableResourceSelection, PlannedSelectedParameterLayout, PlannedSelectionEncoding,
    SignalProducer, SignalStorage, StreamCircuitExecutionPlan, StreamCircuitResourcePlan,
    TensorIndex,
};
use crate::stream_prefix_cache::{RuntimePrefixStateCacheInsert, RuntimePrefixStateCacheKey};
use crate::stream_runtime::{
    RuntimeStreamActivation, RuntimeStreamActivationBatch, RuntimeStreamActivationBatchKind,
    RuntimeStreamActivationKind, RuntimeStreamActivationOutcome, RuntimeStreamInputEvent,
    RuntimeStreamScheduler, RuntimeStreamSchedulerBudget, RuntimeStreamSchedulerError,
    RuntimeStreamSchedulerSnapshot, RuntimeStreamStateCheckpoint, RuntimeStreamStateReservation,
    RuntimeStreamStatus,
};
use crate::stream_state::{
    TransientStateBlockId, TransientStateBlockShape, TransientStateKey, TransientStateRetention,
    TransientStateSlot, TransientStateTableSnapshot,
};
use crate::tensor_storage::TensorStorage;
use crate::vulkan::{DEFAULT_COMPUTE_LOCAL_SIZE_X, DEFAULT_SPIRV_ENTRY_POINT, read_spirv_words};
use crate::vulkan_compute::{
    VULKAN_RESIDENT_INDIRECT_DISPATCH_BYTE_COUNT, VulkanComputeDevice,
    VulkanDeviceLocalMemoryPermit, VulkanDeviceLocalMemoryReclaimer,
    VulkanDeviceLocalMemoryReclaimerRegistration, VulkanError, VulkanGpuResidencyAddressMapping,
    VulkanGpuResidencyGate, VulkanGpuResidencyGateConfig, VulkanGpuResidencyMissQueue,
    VulkanGpuResidencyMissingRequest, VulkanGpuResidencyMissingSnapshot, VulkanMemoryAdmission,
    VulkanResidentBuffer, VulkanResidentBufferCopy, VulkanResidentBufferCopyBatch,
    VulkanResidentBufferPool, VulkanResidentBufferPoolAllocation, VulkanResidentBufferPoolKey,
    VulkanResidentBufferRangeCopy, VulkanResidentBufferReadRange,
    VulkanResidentBufferReadbackBinding, VulkanResidentBufferWriteRange,
    VulkanResidentDistributedExecutionPhase, VulkanResidentExecutionQuantumMeasurement,
    VulkanResidentKernelBufferAccess, VulkanResidentKernelBufferBinding,
    VulkanResidentKernelDispatch, VulkanResidentKernelSequence,
    VulkanResidentKernelSequenceInputCopy, VulkanResidentKernelSequenceSnapshotCopy,
    VulkanResidentKernelSequenceStep, VulkanResidentMappedBufferCopy,
    VulkanResidentQueueSubmissionBatch, VulkanResidentQueueSubmissionTemplate,
    VulkanResidentTransferStream, VulkanShaderFeature, VulkanSharedResidentBufferRoute,
    VulkanSharedResidentBufferSet, VulkanStableResourceAddressPublication,
    VulkanStableResourceAddressTable, VulkanStableResourceAllocation, VulkanStableResourceArena,
    VulkanStableResourceArenaConfig, VulkanStableResourceGroupLayout, VulkanSubgroupOperation,
    VulkanTimelineSemaphore, VulkanTimelineSemaphorePoint, VulkanTimelineSemaphoreReplayState,
    record_vulkan_demand_sequence_device_duration, record_vulkan_execution_quantum_measurement,
    record_vulkan_resident_component_sequence_device_duration, semantic_label_field,
    vulkan_gpu_residency_gate_spirv_words, vulkan_spirv_requirements,
};
#[cfg(test)]
use crate::vulkan_compute::{
    VulkanComputeDeviceCatalog, VulkanStableResourceArenaStats,
    reset_vulkan_resident_execution_counters, vulkan_resident_execution_counters,
};
use crate::vulkan_distributed::{
    VulkanDistributedActivationBufferAllocation, VulkanDistributedActivationBufferPlan,
    VulkanDistributedActivationBuffers, VulkanDistributedActivationRange,
    VulkanDistributedActivationSlot, VulkanDistributedActivationStorage,
    VulkanDistributedDependencyClock, VulkanDistributedDispatchDistribution,
    VulkanDistributedDispatchPlan, VulkanDistributedDispatchRunnerError,
    VulkanDistributedDispatchRunners, VulkanDistributedDispatchSequenceKind,
    VulkanDistributedDispatchShard, VulkanDistributedDispatchSubmission,
    VulkanDistributedEquivalenceKind, VulkanDistributedExecutionPlan,
    VulkanDistributedExecutionPlanSet, VulkanDistributedParameterAllocationPlan,
    VulkanDistributedParameterBuffers, VulkanDistributedParameterExclusionPlan,
    VulkanDistributedPhaseComponentDevicePools, VulkanDistributedQueueSynchronization,
    VulkanDistributedReductionBuffer, VulkanDistributedReductionFinalizationPlan,
    VulkanDistributedReductionPlan, VulkanDistributedReductionRunner,
    VulkanDistributedSelectedResourceDevicePlan, VulkanDistributedSelectedResourceFragmentPlan,
    VulkanDistributedSelectedResourcePartitionPlan, VulkanDistributedSelectedResourceStorePlan,
    VulkanPhysicalExecutionIslandKind, VulkanPhysicalExecutionIslandPlan,
    VulkanPhysicalExecutionTransportKind, VulkanSelectedResourcePlacementPlan,
    VulkanSelectedResourceReconfigurationPlan, allocate_distributed_shared_buffer,
    create_distributed_reduction_runner_for_buffers, distributed_residency_replay_schedule,
    distributed_shard_push_constants, physical_execution_island_kind,
    record_vulkan_physical_execution_island_submission, replay_exact_execution_cases_to_phase,
    resolved_physical_execution_islands, selected_resource_activation,
    selected_resource_placements_fit_phase_participants,
    selected_resource_placements_from_execution_plan, try_plan_selected_resource_placement,
    try_plan_warm_selected_resource_reconfiguration,
    validate_selected_resource_execution_ownership_replacement,
    vulkan_distributed_placement_strategy,
};
#[cfg(test)]
use crate::vulkan_distributed::{
    VulkanDistributedDeviceParameterExclusions, VulkanDistributedParameterAllocation,
    VulkanDistributedPrivateIntermediateBufferAllocation,
    VulkanDistributedPrivateIntermediateDeviceAllocation,
    VulkanDistributedReductionBufferAllocation,
};

mod package;
pub use package::*;

pub const VULKAN_STREAM_CIRCUIT_BACKEND_ID: &str = "vulkan_stream_circuit_ir";
pub const VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA: &str =
    "nerve.vulkan_reusable_kernel_artifacts.v2";
pub const VULKAN_RESIDENT_MODEL_PACKAGE_MANIFEST_SCHEMA: &str =
    "nerve.vulkan_resident_model_package.v13";
const CONTRACT_DIGEST_ALGORITHM: &str = "nerve.json_tree_sha256.v1";
const VULKAN_STREAM_CONTROL_BYTE_CAPACITY: usize = 5 * std::mem::size_of::<u32>();
const VULKAN_STREAM_CONTROL_TOKEN_BYTE_CAPACITY: usize = std::mem::size_of::<u32>();
const VULKAN_STREAM_CONTROL_METADATA_OFFSET: usize = VULKAN_STREAM_CONTROL_TOKEN_BYTE_CAPACITY;
const VULKAN_COMPONENT_BATCH_WIDTH_CONTROL_BYTE_CAPACITY: u32 = std::mem::size_of::<u32>() as u32;
const VULKAN_COMPONENT_BATCH_CONTROL_BYTE_CAPACITY: u32 = 4 * std::mem::size_of::<u32>() as u32;
const VULKAN_SAMPLER_HISTORY_RECORD_BYTE_CAPACITY: usize = 4 * std::mem::size_of::<u32>();
pub const VULKAN_BACKEND_LOOP_MAX_WINDOW: usize = 64;
const VULKAN_BACKEND_LOOP_MIN_TRANSACTION_BUDGET_BYTES: usize = 64 * 1024 * 1024;
const VULKAN_BACKEND_LOOP_TRANSACTION_HEAP_FRACTION_DIVISOR: usize = 8;

include!("vulkan_stream_circuit/resident_plan_buffers.rs");
include!("vulkan_stream_circuit/transient_state_buffer_layout.rs");
include!("vulkan_stream_circuit/transient_state_pages.rs");
include!("vulkan_stream_circuit/edge_plan.rs");
include!("vulkan_stream_circuit/edge_buffers.rs");
include!("vulkan_stream_circuit/edge_transport.rs");
include!("vulkan_stream_circuit/circuit_binding.rs");
include!("vulkan_stream_circuit/dynamic_resource_buffers.rs");
include!("vulkan_stream_circuit/compiled_resource_selector_ownership.rs");
include!("vulkan_stream_circuit/compiled_resource_address_layout.rs");
include!("vulkan_stream_circuit/circuit_mount.rs");
include!("vulkan_stream_circuit/input_transducer.rs");
include!("vulkan_stream_circuit/output_transducer.rs");
include!("vulkan_stream_circuit/sampler.rs");
include!("vulkan_stream_circuit/resident_feedback_control.rs");
include!("vulkan_stream_circuit/batched_output_projection.rs");
include!("vulkan_stream_circuit/multi_stream_batch_runner.rs");
include!("vulkan_stream_circuit/single_token_tick.rs");
include!("vulkan_stream_circuit/feedback_loop.rs");
include!("vulkan_stream_circuit/speculative_decode.rs");
include!("vulkan_stream_circuit/state_transaction.rs");
include!("vulkan_stream_circuit/causal_state_snapshots.rs");
include!("vulkan_stream_circuit/component_batch_execution_scope.rs");
include!("vulkan_stream_circuit/speculative_source_taps.rs");
include!("vulkan_stream_circuit/component_batch_buffers.rs");
include!("vulkan_stream_circuit/component_batch_kernel_selection.rs");
include!("vulkan_stream_circuit/component_batch_slice_runner.rs");
include!("vulkan_stream_circuit/component_batch_distributed.rs");
include!("vulkan_stream_circuit/component_batch_distributed_residency.rs");
include!("vulkan_stream_circuit/component_batch_input_columns.rs");
include!("vulkan_stream_circuit/component_batch_output_rows.rs");
include!("vulkan_stream_circuit/component_batch_temporal.rs");
include!("vulkan_stream_circuit/placed_component_batch_runner.rs");
include!("vulkan_stream_circuit/stream_processor.rs");
include!("vulkan_stream_circuit/token_stream.rs");
include!("vulkan_stream_circuit/token_runtime.rs");
include!("vulkan_stream_circuit/token_engine.rs");
include!("vulkan_stream_circuit/resident_package_slices.rs");
include!("vulkan_stream_circuit/parallel_speculative_state_ingestion.rs");
include!("vulkan_stream_circuit/parallel_speculative_feedback_state.rs");
include!("vulkan_stream_circuit/targeted_component_mount.rs");
include!("vulkan_stream_circuit/targeted_component_execution.rs");
include!("vulkan_stream_circuit/placed_edge_routing.rs");
include!("vulkan_stream_circuit/placed_feedback_devices.rs");
include!("vulkan_stream_circuit/runtime_physical_execution_plan.rs");
include!("vulkan_stream_circuit/runtime_execution_identity.rs");
include!("vulkan_stream_circuit/runtime_device_compatibility.rs");
include!("vulkan_stream_circuit/runtime_implementation_selection.rs");
include!("vulkan_stream_circuit/runtime_resident_derivations.rs");
include!("vulkan_stream_circuit/runtime_resource_contract.rs");
include!("vulkan_stream_circuit/compiled_resource_store_residency.rs");
include!("vulkan_stream_circuit/sparse_moe_execution.rs");
include!("vulkan_stream_circuit/selection_telemetry.rs");
include!("vulkan_stream_circuit/host_memory_capacity.rs");
include!("vulkan_stream_circuit/stream_memory_admission.rs");
include!("vulkan_stream_circuit/placed_model_package_constructors.rs");
include!("vulkan_stream_circuit/placed_model_package_loader.rs");
include!("vulkan_stream_circuit/placed_stream_processor.rs");
include!("vulkan_stream_circuit/placed_prompt_event.rs");
include!("vulkan_stream_circuit/placed_prompt_session.rs");
include!("vulkan_stream_circuit/placed_prompt_stream.rs");
include!("vulkan_stream_circuit/placed_prompt_scheduled_activation.rs");
include!("vulkan_stream_circuit/placed_prefix_state_cache.rs");
include!("vulkan_stream_circuit/placed_prompt_engine.rs");
include!("vulkan_stream_circuit/placed_prompt_engine_shutdown.rs");
include!("vulkan_stream_circuit/placed_stream_transaction.rs");
include!("vulkan_stream_circuit/placed_prompt_device.rs");
include!("vulkan_stream_circuit/placed_runtime_error.rs");
include!("vulkan_stream_circuit/resident_model_package.rs");
include!("vulkan_stream_circuit/resident_package_execution_contract.rs");
include!("vulkan_stream_circuit/resident_package_planning.rs");
include!("vulkan_stream_circuit/compiled_resource_residency_plan.rs");
include!("vulkan_stream_circuit/compiled_resource_physical_placement.rs");
include!("vulkan_stream_circuit/physical_residency_checkpoint.rs");
include!("vulkan_stream_circuit/residency_backpressure_scheduler.rs");
include!("vulkan_stream_circuit/runtime_residency_plan.rs");
include!("vulkan_stream_circuit/runtime_physical_execution_residency.rs");
include!("vulkan_stream_circuit/runtime_auto_placement.rs");
include!("vulkan_stream_circuit/runtime_placement_calibration.rs");
include!("vulkan_stream_circuit/placement_equivalence.rs");
include!("vulkan_stream_circuit/placement_calibration_catalog.rs");
include!("vulkan_stream_circuit/region_placement_calibration_catalog.rs");
include!("vulkan_stream_circuit/selected_resource_calibration_catalog.rs");
include!("vulkan_stream_circuit/package_placement_catalog.rs");
include!("vulkan_stream_circuit/hybrid_placement_resources.rs");
include!("vulkan_stream_circuit/runtime_hybrid_parameter_resources.rs");
include!("vulkan_stream_circuit/hybrid_placement_optimizer.rs");
include!("vulkan_stream_circuit/runtime_hybrid_execution_transient.rs");
include!("vulkan_stream_circuit/runtime_hybrid_candidate_resources.rs");
include!("vulkan_stream_circuit/runtime_hybrid_placement.rs");
include!("vulkan_stream_circuit/runtime_distributed_selected_resource_planning.rs");
include!("vulkan_stream_circuit/runtime_selected_resource_mount_planning.rs");
include!("vulkan_stream_circuit/runtime_physical_mount_planning.rs");
include!("vulkan_stream_circuit/runtime_selected_resource_cache_arbiter.rs");
include!("vulkan_stream_circuit/runtime_selected_resource_reconfiguration.rs");
include!("vulkan_stream_circuit/runtime_distributed_selected_resource_calibration.rs");
include!("vulkan_stream_circuit/runtime_distributed_contract_candidates.rs");
include!("vulkan_stream_circuit/runtime_distributed_execution_identity.rs");
include!("vulkan_stream_circuit/runtime_distributed_placement_calibration.rs");
include!("vulkan_stream_circuit/runtime_region_placement_calibration.rs");
include!("vulkan_stream_circuit/runtime_canonical_placement_calibration.rs");
include!("vulkan_stream_circuit/runtime_staged_placement_calibration.rs");
include!("vulkan_stream_circuit/runtime_transfer_calibration.rs");
include!("vulkan_stream_circuit/resource_backing_store.rs");
include!("vulkan_stream_circuit/device_resource_residency.rs");
include!("vulkan_stream_circuit/device_resource_residency_cohort.rs");
include!("vulkan_stream_circuit/compiled_resource_distributed_cohorts.rs");
include!("vulkan_stream_circuit/compiled_resource_device_upload.rs");
include!("vulkan_stream_circuit/compiled_resource_residency_report.rs");
include!("vulkan_stream_circuit/compiled_resource_memory_plan.rs");
include!("vulkan_stream_circuit/compiled_resource_shared_host_cache.rs");
include!("vulkan_stream_circuit/compiled_resource_device_store.rs");
include!("vulkan_stream_circuit/compiled_resource_readback_validation.rs");
include!("vulkan_stream_circuit/runtime_load_wave_calibration.rs");
include!("vulkan_stream_circuit/compiled_resource_wave.rs");
include!("vulkan_stream_circuit/compiled_resource_retiering.rs");
include!("vulkan_stream_circuit/runtime_working_set_pressure.rs");
include!("vulkan_stream_circuit/runtime_working_set_rebalance.rs");
include!("vulkan_stream_circuit/compiled_resource_representation_cache.rs");
include!("vulkan_stream_circuit/compiled_resource_teardown.rs");
include!("vulkan_stream_circuit/demand_residency_dispatch_chain.rs");
include!("vulkan_stream_circuit/demand_residency_batch_chain.rs");
include!("vulkan_stream_circuit/distributed_selected_resource_gate.rs");
include!("vulkan_stream_circuit/demand_resident_feedback.rs");
include!("vulkan_stream_circuit/resident_package_resource_loading.rs");
include!("vulkan_stream_circuit/resident_package_kernel_loading.rs");
include!("vulkan_stream_circuit/token_engine_codec.rs");
include!("vulkan_stream_circuit/mounted_component.rs");
include!("vulkan_stream_circuit/mounted_execution_graph_runner.rs");
include!("vulkan_stream_circuit/dispatch_segment_runner.rs");
include!("vulkan_stream_circuit/stream_tick_execution_plan.rs");
include!("vulkan_stream_circuit/stream_control_bytes.rs");
include!("vulkan_stream_circuit/kernel_interface.rs");
include!("vulkan_stream_circuit/descriptor_resources.rs");
include!("vulkan_stream_circuit/reusable_kernels.rs");
include!("vulkan_stream_circuit/physical_kernel_artifacts.rs");
include!("vulkan_stream_circuit/kernel_artifact_catalog.rs");
include!("vulkan_stream_circuit/dispatch_binding_plan.rs");
include!("vulkan_stream_circuit/tick_plan.rs");
include!("vulkan_stream_circuit/tick_cursor.rs");
include!("vulkan_stream_circuit/in_process_submission.rs");
include!("vulkan_stream_circuit/placed_tick_execution.rs");
include!("vulkan_stream_circuit/stream_tick_errors.rs");
include!("vulkan_stream_circuit/bound_dispatch.rs");
include!("vulkan_stream_circuit/kernel_descriptor_signature.rs");
include!("vulkan_stream_circuit/circuit_binding_builder.rs");
include!("vulkan_stream_circuit/resident_plan_math.rs");

#[cfg(test)]
mod tests;
