use crate::execution_schedule::{
    RuntimeExecutionQuantum, RuntimeExecutionQuantumBudget, RuntimeExecutionQuantumCalibrator,
    RuntimeExecutionRegion, RuntimeExecutionSchedule,
};

include!("vulkan_compute/features.rs");
include!("vulkan_compute/device_fault.rs");
include!("vulkan_compute/device_types.rs");
include!("vulkan_compute/resident_buffers.rs");
include!("vulkan_compute/resident_buffer_pool.rs");
include!("vulkan_compute/kernel_sequence.rs");
include!("vulkan_compute/buffer_copies.rs");
include!("vulkan_compute/resident_transfer_stream.rs");
include!("vulkan_compute/stable_resource_address_space.rs");
include!("vulkan_compute/gpu_residency_gate.rs");
include!("vulkan_compute/device_catalog.rs");
include!("vulkan_compute/compute_device_construction.rs");
include!("vulkan_compute/compute_device_memory.rs");
include!("vulkan_compute/compute_device_dispatch.rs");
include!("vulkan_compute/compute_device_sequence.rs");
include!("vulkan_compute/compute_device_pipelines.rs");
include!("vulkan_compute/physical_device_capabilities.rs");
include!("vulkan_compute/hardware_profile.rs");
include!("vulkan_compute/calibration_specialized.rs");
#[cfg(test)]
include!("vulkan_compute/stable_resource_address_space_tests.rs");
#[cfg(test)]
include!("vulkan_compute/gpu_residency_gate_tests.rs");
#[cfg(test)]
include!("vulkan_compute/mxfp4_tests.rs");
#[cfg(test)]
include!("vulkan_compute/hyper_connection_tests.rs");
include!("vulkan_compute/tests.rs");
