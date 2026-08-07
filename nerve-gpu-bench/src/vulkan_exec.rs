use std::ffi::CString;
use std::hint::black_box;
use std::mem;
use std::ptr;
use std::time::Instant;

use ash::{Entry, vk};

use crate::benchmark::{
    activation_bytes_for_payload, format_workload_id, output_bytes_for_payload,
    single_target_status_measurement, single_target_status_measurements,
};
use crate::model::{GroupMeasurement, Measurement, PairMeasurement, Sample, Summary, Target};

const F32_TRANSFORM_SHADER_SPV: &[u32] = &[
    119734787, 65536, 851979, 47, 0, 131089, 1, 393227, 1, 1280527431, 1685353262, 808793134, 0,
    196622, 0, 1, 393231, 5, 4, 1852399981, 0, 11, 393232, 4, 17, 64, 1, 1, 196611, 2, 450, 655364,
    1197427783, 1279741775, 1885560645, 1953718128, 1600482425, 1701734764, 1919509599, 1769235301,
    25974, 524292, 1197427783, 1279741775, 1852399429, 1685417059, 1768185701, 1952671090, 6649449,
    262149, 4, 1852399981, 0, 196613, 8, 7890025, 524293, 11, 1197436007, 1633841004, 1986939244,
    1952539503, 1231974249, 68, 262149, 17, 1752397136, 0, 262150, 17, 0, 7234924, 262149, 19,
    1752397168, 0, 262149, 31, 1635017028, 0, 327686, 31, 0, 1970037110, 29541, 262149, 33,
    1635017060, 0, 262215, 11, 11, 28, 196679, 17, 2, 327752, 17, 0, 35, 0, 262215, 30, 6, 4,
    196679, 31, 3, 327752, 31, 0, 35, 0, 262215, 33, 33, 0, 262215, 33, 34, 0, 262215, 46, 11, 25,
    131091, 2, 196641, 3, 2, 262165, 6, 32, 0, 262176, 7, 7, 6, 262167, 9, 6, 3, 262176, 10, 1, 9,
    262203, 10, 11, 1, 262187, 6, 12, 0, 262176, 13, 1, 6, 196638, 17, 6, 262176, 18, 9, 17,
    262203, 18, 19, 9, 262165, 20, 32, 1, 262187, 20, 21, 0, 262176, 22, 9, 6, 131092, 25, 196630,
    29, 32, 196637, 30, 29, 196638, 31, 30, 262176, 32, 2, 31, 262203, 32, 33, 2, 262176, 36, 2,
    29, 262187, 29, 39, 1065354055, 262187, 29, 41, 1048576000, 262187, 6, 44, 64, 262187, 6, 45,
    1, 393260, 9, 46, 44, 45, 45, 327734, 2, 4, 0, 3, 131320, 5, 262203, 7, 8, 7, 327745, 13, 14,
    11, 12, 262205, 6, 15, 14, 196670, 8, 15, 262205, 6, 16, 8, 327745, 22, 23, 19, 21, 262205, 6,
    24, 23, 327856, 25, 26, 16, 24, 196855, 28, 0, 262394, 26, 27, 28, 131320, 27, 262205, 6, 34,
    8, 262205, 6, 35, 8, 393281, 36, 37, 33, 21, 35, 262205, 29, 38, 37, 327813, 29, 40, 38, 39,
    327809, 29, 42, 40, 41, 393281, 36, 43, 33, 21, 34, 196670, 43, 42, 131321, 28, 131320, 28,
    65789, 65592,
];

const PACKED_U32_TRANSFORM_SHADER_SPV: &[u32] = &[
    119734787, 65536, 851979, 54, 0, 131089, 1, 393227, 1, 1280527431, 1685353262, 808793134, 0,
    196622, 0, 1, 393231, 5, 4, 1852399981, 0, 11, 393232, 4, 17, 64, 1, 1, 196611, 2, 450, 655364,
    1197427783, 1279741775, 1885560645, 1953718128, 1600482425, 1701734764, 1919509599, 1769235301,
    25974, 524292, 1197427783, 1279741775, 1852399429, 1685417059, 1768185701, 1952671090, 6649449,
    262149, 4, 1852399981, 0, 196613, 8, 7890025, 524293, 11, 1197436007, 1633841004, 1986939244,
    1952539503, 1231974249, 68, 262149, 17, 1752397136, 0, 262150, 17, 0, 7234924, 262149, 19,
    1752397168, 0, 196613, 29, 120, 262149, 31, 1635017028, 0, 327686, 31, 0, 1970037110, 29541,
    262149, 33, 1635017060, 0, 262215, 11, 11, 28, 196679, 17, 2, 327752, 17, 0, 35, 0, 262215, 30,
    6, 4, 196679, 31, 3, 327752, 31, 0, 35, 0, 262215, 33, 33, 0, 262215, 33, 34, 0, 262215, 53,
    11, 25, 131091, 2, 196641, 3, 2, 262165, 6, 32, 0, 262176, 7, 7, 6, 262167, 9, 6, 3, 262176,
    10, 1, 9, 262203, 10, 11, 1, 262187, 6, 12, 0, 262176, 13, 1, 6, 196638, 17, 6, 262176, 18, 9,
    17, 262203, 18, 19, 9, 262165, 20, 32, 1, 262187, 20, 21, 0, 262176, 22, 9, 6, 131092, 25,
    196637, 30, 6, 196638, 31, 30, 262176, 32, 2, 31, 262203, 32, 33, 2, 262176, 35, 2, 6, 262187,
    6, 39, 1664525, 262187, 6, 41, 1013904223, 262187, 20, 45, 16, 262187, 6, 51, 64, 262187, 6,
    52, 1, 393260, 9, 53, 51, 52, 52, 327734, 2, 4, 0, 3, 131320, 5, 262203, 7, 8, 7, 262203, 7,
    29, 7, 327745, 13, 14, 11, 12, 262205, 6, 15, 14, 196670, 8, 15, 262205, 6, 16, 8, 327745, 22,
    23, 19, 21, 262205, 6, 24, 23, 327856, 25, 26, 16, 24, 196855, 28, 0, 262394, 26, 27, 28,
    131320, 27, 262205, 6, 34, 8, 393281, 35, 36, 33, 21, 34, 262205, 6, 37, 36, 196670, 29, 37,
    262205, 6, 38, 29, 327812, 6, 40, 38, 39, 327808, 6, 42, 40, 41, 196670, 29, 42, 262205, 6, 43,
    29, 262205, 6, 44, 29, 327874, 6, 46, 44, 45, 327878, 6, 47, 43, 46, 196670, 29, 47, 262205, 6,
    48, 8, 262205, 6, 49, 29, 393281, 35, 50, 33, 21, 48, 196670, 50, 49, 131321, 28, 131320, 28,
    65789, 65592,
];

const F32_MOE_EXPERT_SHADER_SPV: &[u8] = include_bytes!("shaders/f32_moe_expert.spv");
const F32_ROUTER_REDUCTION_SHADER_SPV: &[u8] = include_bytes!("shaders/f32_router_reduction.spv");
const PACKED_MOE_EXPERT_SHADER_SPV: &[u8] = include_bytes!("shaders/packed_moe_expert.spv");
const PACKED_ROUTER_REDUCTION_SHADER_SPV: &[u8] =
    include_bytes!("shaders/packed_router_reduction.spv");

#[derive(Clone, Copy)]
enum ShaderCode {
    Words(&'static [u32]),
    Bytes(&'static [u8]),
}

#[derive(Clone)]
struct DenseFormatKernel {
    format: String,
    shader: ShaderCode,
    bytes_per_storage_element: usize,
    logical_elements_per_storage_element: u64,
    operations_per_storage_element: u64,
    pattern: &'static str,
}

fn workload_format_kernel(workload_class: &str, format: &str) -> Option<DenseFormatKernel> {
    match workload_class {
        "dense_projection" => dense_format_kernel(format),
        "moe_expert" => moe_expert_kernel(format),
        "router_reduction" => router_reduction_kernel(format),
        _ => None,
    }
}

fn dense_format_kernel(format: &str) -> Option<DenseFormatKernel> {
    match format {
        "f32" => Some(DenseFormatKernel {
            format: "f32".to_string(),
            shader: ShaderCode::Words(F32_TRANSFORM_SHADER_SPV),
            bytes_per_storage_element: mem::size_of::<f32>(),
            logical_elements_per_storage_element: 1,
            operations_per_storage_element: 2,
            pattern: "single_target_compute",
        }),
        "f16" => Some(packed_dense_kernel("f16", 2)),
        "bf16" => Some(packed_dense_kernel("bf16", 2)),
        "fp8" | "fp8_e4m3" | "fp8_e5m2" => Some(packed_dense_kernel(format, 4)),
        "int8" | "q8_0" => Some(packed_dense_kernel(format, 4)),
        "q6_k" => Some(packed_dense_kernel(format, 5)),
        "q5_0" | "q5_1" | "q5_k" => Some(packed_dense_kernel(format, 6)),
        "int4" | "q4_0" | "q4_1" | "q4_k" | "iq4_nl" | "iq4_xs" => {
            Some(packed_dense_kernel(format, 8))
        }
        "q3_k" | "iq3_s" => Some(packed_dense_kernel(format, 10)),
        "q2_k" | "iq2_xs" => Some(packed_dense_kernel(format, 16)),
        "fp4" => Some(packed_dense_kernel("fp4", 8)),
        "mxfp4" | "nvfp4" => Some(packed_dense_kernel(format, 8)),
        _ => None,
    }
}

fn packed_dense_kernel(
    format: &str,
    logical_elements_per_storage_element: u64,
) -> DenseFormatKernel {
    DenseFormatKernel {
        format: format.to_string(),
        shader: ShaderCode::Words(PACKED_U32_TRANSFORM_SHADER_SPV),
        bytes_per_storage_element: mem::size_of::<u32>(),
        logical_elements_per_storage_element,
        operations_per_storage_element: 4,
        pattern: "single_target_packed_emulated_compute",
    }
}

fn moe_expert_kernel(format: &str) -> Option<DenseFormatKernel> {
    match format {
        "f32" => Some(DenseFormatKernel {
            format: "f32".to_string(),
            shader: ShaderCode::Bytes(F32_MOE_EXPERT_SHADER_SPV),
            bytes_per_storage_element: mem::size_of::<f32>(),
            logical_elements_per_storage_element: 1,
            operations_per_storage_element: 10,
            pattern: "moe_expert_compute",
        }),
        _ => packed_workload_kernel(
            format,
            PACKED_MOE_EXPERT_SHADER_SPV,
            "moe_expert_packed_emulated_compute",
            12,
        ),
    }
}

fn router_reduction_kernel(format: &str) -> Option<DenseFormatKernel> {
    match format {
        "f32" => Some(DenseFormatKernel {
            format: "f32".to_string(),
            shader: ShaderCode::Bytes(F32_ROUTER_REDUCTION_SHADER_SPV),
            bytes_per_storage_element: mem::size_of::<f32>(),
            logical_elements_per_storage_element: 1,
            operations_per_storage_element: 8,
            pattern: "router_reduction_compute",
        }),
        _ => packed_workload_kernel(
            format,
            PACKED_ROUTER_REDUCTION_SHADER_SPV,
            "router_reduction_packed_emulated_compute",
            8,
        ),
    }
}

fn packed_workload_kernel(
    format: &str,
    shader: &'static [u8],
    pattern: &'static str,
    operations_per_storage_element: u64,
) -> Option<DenseFormatKernel> {
    dense_format_kernel(format).map(|base| DenseFormatKernel {
        shader: ShaderCode::Bytes(shader),
        operations_per_storage_element,
        pattern,
        ..base
    })
}

pub fn run_vulkan_single_target_measurements(
    target: &Target,
    payload_bytes: usize,
    samples: usize,
    formats: &[String],
    workloads: &[String],
) -> Vec<Measurement> {
    match open_compute_device(target) {
        Ok(device) => vulkan_measurements(
            &device,
            &target.stable_target_id,
            payload_bytes,
            samples,
            formats,
            workloads,
        ),
        Err(message) => single_target_status_measurements(
            &target.stable_target_id,
            payload_bytes,
            formats,
            workloads,
            "failed",
            &message,
        ),
    }
}

pub fn run_vulkan_pair_measurements(
    targets: &[&Target],
    payload_bytes: usize,
    samples: usize,
    formats: &[String],
    workloads: &[String],
) -> Vec<PairMeasurement> {
    let mut measurements = Vec::new();
    for (source_index, source) in targets.iter().enumerate() {
        for (destination_index, destination) in targets.iter().enumerate() {
            if source.stable_target_id == destination.stable_target_id {
                continue;
            }
            measurements.extend(run_vulkan_ordered_transfer_measurements(
                source,
                destination,
                payload_bytes,
                samples,
                formats,
                workloads,
            ));
            measurements.extend(run_vulkan_ordered_serial_pair_measurements(
                source,
                destination,
                payload_bytes,
                samples,
                formats,
                workloads,
            ));
            if source_index < destination_index {
                measurements.extend(run_vulkan_parallel_pair_measurements(
                    source,
                    destination,
                    payload_bytes,
                    samples,
                    formats,
                    workloads,
                ));
            }
        }
    }
    measurements
}

fn run_vulkan_ordered_transfer_measurements(
    source: &Target,
    destination: &Target,
    payload_bytes: usize,
    samples: usize,
    formats: &[String],
    workloads: &[String],
) -> Vec<PairMeasurement> {
    let source_device = match open_compute_device(source) {
        Ok(device) => device,
        Err(message) => {
            return failed_transfer_measurements(
                &source.stable_target_id,
                &destination.stable_target_id,
                payload_bytes,
                formats,
                workloads,
                &message,
            );
        }
    };
    let destination_device = match open_compute_device(destination) {
        Ok(device) => device,
        Err(message) => {
            return failed_transfer_measurements(
                &source.stable_target_id,
                &destination.stable_target_id,
                payload_bytes,
                formats,
                workloads,
                &message,
            );
        }
    };
    formats
        .iter()
        .flat_map(|format| {
            workloads.iter().map(|workload| {
                match run_host_staged_transfer(
                    &source_device,
                    &destination_device,
                    &source.stable_target_id,
                    &destination.stable_target_id,
                    payload_bytes,
                    samples,
                    workload,
                    format,
                ) {
                    Ok(measurement) => measurement,
                    Err(message) => failed_transfer_measurement(
                        &source.stable_target_id,
                        &destination.stable_target_id,
                        payload_bytes,
                        workload,
                        format,
                        &message,
                    ),
                }
            })
        })
        .collect()
}

fn run_vulkan_ordered_serial_pair_measurements(
    source: &Target,
    destination: &Target,
    payload_bytes: usize,
    samples: usize,
    formats: &[String],
    workloads: &[String],
) -> Vec<PairMeasurement> {
    let source_device = match open_compute_device(source) {
        Ok(device) => device,
        Err(message) => {
            return failed_dense_pair_measurements(
                &source.stable_target_id,
                &destination.stable_target_id,
                "synthetic_layer_split_small_payload",
                "two_target_serial",
                payload_bytes,
                formats,
                workloads,
                &message,
            );
        }
    };
    let destination_device = match open_compute_device(destination) {
        Ok(device) => device,
        Err(message) => {
            return failed_dense_pair_measurements(
                &source.stable_target_id,
                &destination.stable_target_id,
                "synthetic_layer_split_small_payload",
                "two_target_serial",
                payload_bytes,
                formats,
                workloads,
                &message,
            );
        }
    };
    formats
        .iter()
        .flat_map(|format| {
            workloads.iter().filter_map(|workload| {
                workload_format_kernel(workload, format).map(|kernel| {
                    match run_vulkan_dense_serial_pair(
                        &source_device,
                        &destination_device,
                        &source.stable_target_id,
                        &destination.stable_target_id,
                        payload_bytes,
                        samples,
                        workload,
                        kernel,
                    ) {
                        Ok(measurement) => measurement,
                        Err(message) => failed_dense_pair_measurement(
                            &source.stable_target_id,
                            &destination.stable_target_id,
                            "synthetic_layer_split_small_payload",
                            "two_target_serial",
                            payload_bytes,
                            workload,
                            format,
                            &message,
                        ),
                    }
                })
            })
        })
        .collect()
}

fn run_vulkan_parallel_pair_measurements(
    left: &Target,
    right: &Target,
    payload_bytes: usize,
    samples: usize,
    formats: &[String],
    workloads: &[String],
) -> Vec<PairMeasurement> {
    let left_device = match open_compute_device(left) {
        Ok(device) => device,
        Err(message) => {
            return failed_dense_pair_measurements(
                &left.stable_target_id,
                &right.stable_target_id,
                "synthetic_tensor_split_small_payload",
                "two_target_parallel",
                payload_bytes,
                formats,
                workloads,
                &message,
            );
        }
    };
    let right_device = match open_compute_device(right) {
        Ok(device) => device,
        Err(message) => {
            return failed_dense_pair_measurements(
                &left.stable_target_id,
                &right.stable_target_id,
                "synthetic_tensor_split_small_payload",
                "two_target_parallel",
                payload_bytes,
                formats,
                workloads,
                &message,
            );
        }
    };
    formats
        .iter()
        .flat_map(|format| {
            workloads.iter().filter_map(|workload| {
                workload_format_kernel(workload, format).map(|kernel| {
                    match run_vulkan_dense_parallel_pair(
                        &left_device,
                        &right_device,
                        &left.stable_target_id,
                        &right.stable_target_id,
                        payload_bytes,
                        samples,
                        workload,
                        kernel,
                    ) {
                        Ok(measurement) => measurement,
                        Err(message) => failed_dense_pair_measurement(
                            &left.stable_target_id,
                            &right.stable_target_id,
                            "synthetic_tensor_split_small_payload",
                            "two_target_parallel",
                            payload_bytes,
                            workload,
                            format,
                            &message,
                        ),
                    }
                })
            })
        })
        .collect()
}

pub fn run_vulkan_group_measurements(
    targets: &[&Target],
    payload_bytes: usize,
    samples: usize,
    formats: &[String],
    workloads: &[String],
) -> Vec<GroupMeasurement> {
    let mut measurements = Vec::new();
    for first_index in 0..targets.len() {
        for second_index in (first_index + 1)..targets.len() {
            for third_index in (second_index + 1)..targets.len() {
                measurements.extend(run_vulkan_triplet_measurements(
                    targets[first_index],
                    targets[second_index],
                    targets[third_index],
                    payload_bytes,
                    samples,
                    formats,
                    workloads,
                ));
            }
        }
    }
    measurements
}

fn run_vulkan_triplet_measurements(
    first: &Target,
    second: &Target,
    third: &Target,
    payload_bytes: usize,
    samples: usize,
    formats: &[String],
    workloads: &[String],
) -> Vec<GroupMeasurement> {
    let target_ids = [
        first.stable_target_id.clone(),
        second.stable_target_id.clone(),
        third.stable_target_id.clone(),
    ];
    let first_device = match open_compute_device(first) {
        Ok(device) => device,
        Err(message) => {
            return failed_triplet_measurements(
                &target_ids,
                payload_bytes,
                formats,
                workloads,
                &message,
            );
        }
    };
    let second_device = match open_compute_device(second) {
        Ok(device) => device,
        Err(message) => {
            return failed_triplet_measurements(
                &target_ids,
                payload_bytes,
                formats,
                workloads,
                &message,
            );
        }
    };
    let third_device = match open_compute_device(third) {
        Ok(device) => device,
        Err(message) => {
            return failed_triplet_measurements(
                &target_ids,
                payload_bytes,
                formats,
                workloads,
                &message,
            );
        }
    };

    formats
        .iter()
        .flat_map(|format| {
            workloads.iter().filter_map(|workload| {
                workload_format_kernel(workload, format).map(|kernel| {
                    let serial = run_vulkan_dense_serial_triplet(
                        [&first_device, &second_device, &third_device],
                        &target_ids,
                        payload_bytes,
                        samples,
                        workload,
                        kernel.clone(),
                    )
                    .unwrap_or_else(|message| {
                        failed_triplet_measurement(
                            &target_ids,
                            "synthetic_layer_split_group_small_payload",
                            "three_target_serial",
                            payload_bytes,
                            workload,
                            format,
                            &message,
                        )
                    });
                    let parallel = run_vulkan_dense_parallel_triplet(
                        [&first_device, &second_device, &third_device],
                        &target_ids,
                        payload_bytes,
                        samples,
                        workload,
                        kernel,
                    )
                    .unwrap_or_else(|message| {
                        failed_triplet_measurement(
                            &target_ids,
                            "synthetic_tensor_split_group_small_payload",
                            "three_target_parallel",
                            payload_bytes,
                            workload,
                            format,
                            &message,
                        )
                    });
                    [serial, parallel]
                })
            })
        })
        .flatten()
        .collect()
}

fn vulkan_measurements(
    device: &OpenVulkanComputeDevice,
    target_id: &str,
    payload_bytes: usize,
    samples: usize,
    formats: &[String],
    workloads: &[String],
) -> Vec<Measurement> {
    formats
        .iter()
        .flat_map(|format| {
            workloads.iter().filter_map(move |workload| {
                workload_format_kernel(workload, format).map(|kernel| {
                    match run_vulkan_dense_projection(
                        device,
                        target_id,
                        payload_bytes,
                        samples,
                        workload,
                        kernel,
                    ) {
                        Ok(measurement) => measurement,
                        Err(message) => single_target_status_measurement(
                            target_id,
                            payload_bytes,
                            workload,
                            format,
                            "failed",
                            &message,
                        ),
                    }
                })
            })
        })
        .collect()
}

#[cfg(test)]
fn skipped_vulkan_axes<'a>(
    formats: &'a [String],
    workloads: &'a [String],
) -> Vec<(&'a str, &'a str)> {
    formats
        .iter()
        .flat_map(|format| {
            workloads.iter().filter_map(move |workload| {
                if workload_format_kernel(workload, format).is_some() {
                    None
                } else {
                    Some((format.as_str(), workload.as_str()))
                }
            })
        })
        .collect()
}

struct OpenVulkanComputeDevice {
    device: ash::Device,
    instance: ash::Instance,
    compute_queue_family_index: u32,
    queue: vk::Queue,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    timestamp_period_ns: f32,
    timestamp_valid_bits: u32,
}

impl Drop for OpenVulkanComputeDevice {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

struct VulkanBuffer {
    device: ash::Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
}

impl Drop for VulkanBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

struct TransferContext {
    device: ash::Device,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
}

impl Drop for TransferContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_fence(self.fence, None);
            self.device
                .free_command_buffers(self.command_pool, &[self.command_buffer]);
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}

struct DenseComputeContext {
    device: ash::Device,
    resources: ComputeResources,
    upload: VulkanBuffer,
    readback: VulkanBuffer,
    storage: VulkanBuffer,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    query_pool: vk::QueryPool,
    fence: vk::Fence,
    kernel: DenseFormatKernel,
    storage_elements: usize,
    buffer_size: vk::DeviceSize,
    dispatch_groups: u32,
}

impl Drop for DenseComputeContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_query_pool(self.query_pool, None);
            self.device
                .free_command_buffers(self.command_pool, &[self.command_buffer]);
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}

struct HostStagedTransfer {
    source_storage: VulkanBuffer,
    source_readback: VulkanBuffer,
    destination_upload: VulkanBuffer,
    destination_storage: VulkanBuffer,
    source_transfer: TransferContext,
    destination_transfer: TransferContext,
    host_stage: Vec<u8>,
    activation_bytes: usize,
    buffer_size: vk::DeviceSize,
}

struct TransferSampleMetrics {
    duration_ns: u128,
    bytes_read: u64,
    bytes_written: u64,
}

fn run_host_staged_transfer(
    source_device: &OpenVulkanComputeDevice,
    destination_device: &OpenVulkanComputeDevice,
    source_id: &str,
    destination_id: &str,
    payload_bytes: usize,
    samples: usize,
    workload_class: &str,
    format: &str,
) -> Result<PairMeasurement, String> {
    let activation_bytes = activation_bytes_for_payload(payload_bytes);
    let mut transfer =
        create_host_staged_transfer(source_device, destination_device, activation_bytes)?;
    let mut measured_samples = Vec::with_capacity(samples);
    for sample_index in 0..samples {
        let metrics =
            run_host_staged_transfer_sample(source_device, destination_device, &mut transfer)?;
        measured_samples.push(Sample {
            sample_index,
            duration_ns: metrics.duration_ns,
            iterations: 1,
            bytes_read: metrics.bytes_read,
            bytes_written: metrics.bytes_written,
            operations: 0,
        });
    }

    let summary = summarize_samples(&measured_samples);
    Ok(PairMeasurement {
        workload_id: format_workload_id("ordered_activation_transfer", workload_class, format),
        comparison_group: "small_payload_placement_comparison".to_string(),
        workload_class: workload_class.to_string(),
        placement_strategy: "activation_transfer_only".to_string(),
        source_target_id: source_id.to_string(),
        destination_target_id: destination_id.to_string(),
        pattern: "ordered_activation_transfer".to_string(),
        operation_family: "activation_transfer".to_string(),
        regime: "small_payload".to_string(),
        format: format.to_string(),
        status: "completed".to_string(),
        reason: None,
        payload_bytes,
        source_payload_bytes: activation_bytes,
        destination_payload_bytes: activation_bytes,
        activation_bytes,
        output_bytes: 0,
        samples: measured_samples,
        summary,
    })
}

fn create_host_staged_transfer(
    source_device: &OpenVulkanComputeDevice,
    destination_device: &OpenVulkanComputeDevice,
    activation_bytes: usize,
) -> Result<HostStagedTransfer, String> {
    let buffer_size = activation_bytes as vk::DeviceSize;
    let source_upload = create_buffer(
        source_device,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let source_storage = create_buffer(
        source_device,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let source_readback = create_buffer(
        source_device,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let destination_upload = create_buffer(
        destination_device,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let destination_storage = create_buffer(
        destination_device,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    write_pattern_bytes(&source_device.device, &source_upload)?;
    let source_transfer = create_transfer_context(source_device)?;
    let destination_transfer = create_transfer_context(destination_device)?;
    submit_copy_buffer(
        source_device,
        &source_transfer,
        source_upload.buffer,
        source_storage.buffer,
        buffer_size,
    )?;
    Ok(HostStagedTransfer {
        source_storage,
        source_readback,
        destination_upload,
        destination_storage,
        source_transfer,
        destination_transfer,
        host_stage: vec![0_u8; activation_bytes],
        activation_bytes,
        buffer_size,
    })
}

fn run_host_staged_transfer_sample(
    source_device: &OpenVulkanComputeDevice,
    destination_device: &OpenVulkanComputeDevice,
    transfer: &mut HostStagedTransfer,
) -> Result<TransferSampleMetrics, String> {
    let started = Instant::now();
    submit_copy_buffer(
        source_device,
        &transfer.source_transfer,
        transfer.source_storage.buffer,
        transfer.source_readback.buffer,
        transfer.buffer_size,
    )?;
    read_buffer_bytes(
        &source_device.device,
        &transfer.source_readback,
        &mut transfer.host_stage,
    )?;
    write_buffer_bytes(
        &destination_device.device,
        &transfer.destination_upload,
        &transfer.host_stage,
    )?;
    submit_copy_buffer(
        destination_device,
        &transfer.destination_transfer,
        transfer.destination_upload.buffer,
        transfer.destination_storage.buffer,
        transfer.buffer_size,
    )?;
    let duration = started.elapsed();
    black_box(transfer.host_stage.first().copied().unwrap_or_default());
    Ok(TransferSampleMetrics {
        duration_ns: duration.as_nanos(),
        bytes_read: transfer.activation_bytes as u64,
        bytes_written: transfer.activation_bytes as u64,
    })
}

fn create_transfer_context(
    compute_device: &OpenVulkanComputeDevice,
) -> Result<TransferContext, String> {
    let command_pool = unsafe {
        compute_device.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(compute_device.compute_queue_family_index),
            None,
        )
    }
    .map_err(|error| format!("could not create Vulkan transfer command pool: {error:?}"))?;
    let command_buffer = unsafe {
        compute_device.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
    }
    .map_err(|error| format!("could not allocate Vulkan transfer command buffer: {error:?}"))?[0];
    let fence = unsafe {
        compute_device
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
    }
    .map_err(|error| format!("could not create Vulkan transfer fence: {error:?}"))?;
    Ok(TransferContext {
        device: compute_device.device.clone(),
        command_pool,
        command_buffer,
        fence,
    })
}

fn submit_copy_buffer(
    compute_device: &OpenVulkanComputeDevice,
    context: &TransferContext,
    source: vk::Buffer,
    destination: vk::Buffer,
    size: vk::DeviceSize,
) -> Result<(), String> {
    unsafe {
        compute_device
            .device
            .reset_command_buffer(context.command_buffer, vk::CommandBufferResetFlags::empty())
            .map_err(|error| {
                format!("could not reset Vulkan transfer command buffer: {error:?}")
            })?;
        compute_device
            .device
            .begin_command_buffer(
                context.command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(|error| {
                format!("could not begin Vulkan transfer command buffer: {error:?}")
            })?;
        compute_device.device.cmd_copy_buffer(
            context.command_buffer,
            source,
            destination,
            &[vk::BufferCopy::default().size(size)],
        );
        compute_device
            .device
            .end_command_buffer(context.command_buffer)
            .map_err(|error| format!("could not end Vulkan transfer command buffer: {error:?}"))?;
        compute_device
            .device
            .reset_fences(&[context.fence])
            .map_err(|error| format!("could not reset Vulkan transfer fence: {error:?}"))?;
        compute_device
            .device
            .queue_submit(
                compute_device.queue,
                &[vk::SubmitInfo::default().command_buffers(&[context.command_buffer])],
                context.fence,
            )
            .map_err(|error| format!("could not submit Vulkan transfer work: {error:?}"))?;
        compute_device
            .device
            .wait_for_fences(&[context.fence], true, u64::MAX)
            .map_err(|error| format!("could not wait for Vulkan transfer work: {error:?}"))?;
    }
    Ok(())
}

fn failed_transfer_measurements(
    source_id: &str,
    destination_id: &str,
    payload_bytes: usize,
    formats: &[String],
    workloads: &[String],
    reason: &str,
) -> Vec<PairMeasurement> {
    formats
        .iter()
        .flat_map(|format| {
            workloads.iter().map(|workload| {
                failed_transfer_measurement(
                    source_id,
                    destination_id,
                    payload_bytes,
                    workload,
                    format,
                    reason,
                )
            })
        })
        .collect()
}

fn failed_transfer_measurement(
    source_id: &str,
    destination_id: &str,
    payload_bytes: usize,
    workload_class: &str,
    format: &str,
    reason: &str,
) -> PairMeasurement {
    let activation_bytes = activation_bytes_for_payload(payload_bytes);
    PairMeasurement {
        workload_id: format_workload_id("ordered_activation_transfer", workload_class, format),
        comparison_group: "small_payload_placement_comparison".to_string(),
        workload_class: workload_class.to_string(),
        placement_strategy: "activation_transfer_only".to_string(),
        source_target_id: source_id.to_string(),
        destination_target_id: destination_id.to_string(),
        pattern: "ordered_activation_transfer".to_string(),
        operation_family: "activation_transfer".to_string(),
        regime: "small_payload".to_string(),
        format: format.to_string(),
        status: "failed".to_string(),
        reason: Some(reason.to_string()),
        payload_bytes,
        source_payload_bytes: activation_bytes,
        destination_payload_bytes: activation_bytes,
        activation_bytes,
        output_bytes: 0,
        samples: Vec::new(),
        summary: None,
    }
}

fn run_vulkan_dense_projection(
    compute_device: &OpenVulkanComputeDevice,
    target_id: &str,
    payload_bytes: usize,
    samples: usize,
    workload_class: &str,
    kernel: DenseFormatKernel,
) -> Result<Measurement, String> {
    let context = create_dense_compute_context(compute_device, payload_bytes, kernel)?;
    let mut measured_samples = Vec::with_capacity(samples);
    for sample_index in 0..samples {
        let sample = submit_dense_compute_sample(compute_device, &context, sample_index)?;
        black_box(read_first_storage_word(
            &compute_device.device,
            &context.readback,
            &context.kernel,
        )?);
        measured_samples.push(sample);
    }

    Ok(Measurement {
        workload_id: format!(
            "single_target_small_payload:{workload_class}:{}",
            context.kernel.format
        ),
        comparison_group: "small_payload_placement_comparison".to_string(),
        workload_class: workload_class.to_string(),
        placement_strategy: "single_target_serial".to_string(),
        target_id: target_id.to_string(),
        pattern: context.kernel.pattern.to_string(),
        operation_family: workload_class.to_string(),
        regime: "small_payload".to_string(),
        format: context.kernel.format.to_string(),
        status: "completed".to_string(),
        reason: None,
        payload_bytes,
        working_set_bytes: context.buffer_size as usize,
        summary: summarize_samples(&measured_samples),
        samples: measured_samples,
    })
}

fn create_dense_compute_context(
    compute_device: &OpenVulkanComputeDevice,
    payload_bytes: usize,
    kernel: DenseFormatKernel,
) -> Result<DenseComputeContext, String> {
    if compute_device.timestamp_valid_bits == 0 || compute_device.timestamp_period_ns <= 0.0 {
        return Err("selected Vulkan compute queue does not expose usable timestamps".to_string());
    }

    let storage_elements = (payload_bytes / kernel.bytes_per_storage_element).max(1);
    let buffer_size = (storage_elements * kernel.bytes_per_storage_element) as vk::DeviceSize;
    let upload = create_buffer(
        compute_device,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let readback = create_buffer(
        compute_device,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let storage = create_buffer(
        compute_device,
        buffer_size,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    fill_upload_buffer(&compute_device.device, &upload, storage_elements, &kernel)?;

    let resources =
        create_compute_resources(compute_device, storage.buffer, buffer_size, kernel.shader)?;
    let command_pool = unsafe {
        compute_device.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(compute_device.compute_queue_family_index),
            None,
        )
    }
    .map_err(|error| format!("could not create Vulkan command pool: {error:?}"))?;
    let command_buffer = unsafe {
        compute_device.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
    }
    .map_err(|error| format!("could not allocate Vulkan command buffer: {error:?}"))?[0];
    let query_pool = unsafe {
        compute_device.device.create_query_pool(
            &vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(2),
            None,
        )
    }
    .map_err(|error| format!("could not create Vulkan timestamp query pool: {error:?}"))?;
    let fence = unsafe {
        compute_device
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
    }
    .map_err(|error| format!("could not create Vulkan fence: {error:?}"))?;
    Ok(DenseComputeContext {
        device: compute_device.device.clone(),
        resources,
        upload,
        readback,
        storage,
        command_pool,
        command_buffer,
        query_pool,
        fence,
        kernel,
        storage_elements,
        buffer_size,
        dispatch_groups: storage_elements.div_ceil(64) as u32,
    })
}

fn submit_dense_compute_sample(
    compute_device: &OpenVulkanComputeDevice,
    context: &DenseComputeContext,
    sample_index: usize,
) -> Result<Sample, String> {
    record_compute_dispatch(
        compute_device,
        &context.resources,
        context.command_buffer,
        context.query_pool,
        context.upload.buffer,
        context.storage.buffer,
        context.readback.buffer,
        context.buffer_size,
        context.storage_elements as u32,
        context.dispatch_groups,
    )?;
    unsafe {
        compute_device
            .device
            .reset_fences(&[context.fence])
            .map_err(|error| format!("could not reset Vulkan fence: {error:?}"))?;
        compute_device
            .device
            .queue_submit(
                compute_device.queue,
                &[vk::SubmitInfo::default().command_buffers(&[context.command_buffer])],
                context.fence,
            )
            .map_err(|error| format!("could not submit Vulkan compute work: {error:?}"))?;
        compute_device
            .device
            .wait_for_fences(&[context.fence], true, u64::MAX)
            .map_err(|error| format!("could not wait for Vulkan compute work: {error:?}"))?;
    }
    Ok(Sample {
        sample_index,
        duration_ns: read_timestamp_duration_ns(compute_device, context.query_pool)?,
        iterations: 1,
        bytes_read: context.buffer_size as u64,
        bytes_written: context.buffer_size as u64,
        operations: (context.storage_elements as u64)
            * context.kernel.logical_elements_per_storage_element
            * context.kernel.operations_per_storage_element,
    })
}

fn run_vulkan_dense_serial_pair(
    source_device: &OpenVulkanComputeDevice,
    destination_device: &OpenVulkanComputeDevice,
    source_id: &str,
    destination_id: &str,
    payload_bytes: usize,
    samples: usize,
    workload_class: &str,
    kernel: DenseFormatKernel,
) -> Result<PairMeasurement, String> {
    let source_payload_bytes = payload_bytes / 2;
    let destination_payload_bytes = payload_bytes - source_payload_bytes;
    let activation_bytes = activation_bytes_for_payload(payload_bytes);
    let source_context =
        create_dense_compute_context(source_device, source_payload_bytes, kernel.clone())?;
    let destination_context = create_dense_compute_context(
        destination_device,
        destination_payload_bytes,
        kernel.clone(),
    )?;
    let mut transfer =
        create_host_staged_transfer(source_device, destination_device, activation_bytes)?;
    let mut measured_samples = Vec::with_capacity(samples);
    for sample_index in 0..samples {
        let started = Instant::now();
        let source_sample =
            submit_dense_compute_sample(source_device, &source_context, sample_index)?;
        let transfer_sample =
            run_host_staged_transfer_sample(source_device, destination_device, &mut transfer)?;
        let destination_sample =
            submit_dense_compute_sample(destination_device, &destination_context, sample_index)?;
        let duration = started.elapsed();
        black_box(read_first_storage_word(
            &destination_device.device,
            &destination_context.readback,
            &destination_context.kernel,
        )?);
        measured_samples.push(Sample {
            sample_index,
            duration_ns: duration.as_nanos(),
            iterations: 1,
            bytes_read: source_sample.bytes_read
                + destination_sample.bytes_read
                + transfer_sample.bytes_read,
            bytes_written: source_sample.bytes_written
                + destination_sample.bytes_written
                + transfer_sample.bytes_written,
            operations: source_sample.operations + destination_sample.operations,
        });
    }

    let summary = summarize_samples(&measured_samples);
    Ok(PairMeasurement {
        workload_id: format_workload_id(
            "synthetic_layer_split_small_payload",
            workload_class,
            &kernel.format,
        ),
        comparison_group: "small_payload_placement_comparison".to_string(),
        workload_class: workload_class.to_string(),
        placement_strategy: "two_target_serial".to_string(),
        source_target_id: source_id.to_string(),
        destination_target_id: destination_id.to_string(),
        pattern: "synthetic_layer_split_small_payload".to_string(),
        operation_family: workload_class.to_string(),
        regime: "small_payload".to_string(),
        format: kernel.format,
        status: "completed".to_string(),
        reason: None,
        payload_bytes,
        source_payload_bytes,
        destination_payload_bytes,
        activation_bytes,
        output_bytes: output_bytes_for_payload(payload_bytes),
        samples: measured_samples,
        summary,
    })
}

fn run_vulkan_dense_parallel_pair(
    left_device: &OpenVulkanComputeDevice,
    right_device: &OpenVulkanComputeDevice,
    left_id: &str,
    right_id: &str,
    payload_bytes: usize,
    samples: usize,
    workload_class: &str,
    kernel: DenseFormatKernel,
) -> Result<PairMeasurement, String> {
    let left_payload_bytes = payload_bytes / 2;
    let right_payload_bytes = payload_bytes - left_payload_bytes;
    let activation_bytes = activation_bytes_for_payload(payload_bytes);
    let output_bytes = output_bytes_for_payload(payload_bytes);
    let left_context =
        create_dense_compute_context(left_device, left_payload_bytes, kernel.clone())?;
    let right_context =
        create_dense_compute_context(right_device, right_payload_bytes, kernel.clone())?;
    let mut left_output = vec![0_u8; output_bytes.min(left_context.readback.size as usize)];
    let mut right_output = vec![0_u8; output_bytes.min(right_context.readback.size as usize)];
    let mut measured_samples = Vec::with_capacity(samples);
    for sample_index in 0..samples {
        record_compute_dispatch(
            left_device,
            &left_context.resources,
            left_context.command_buffer,
            left_context.query_pool,
            left_context.upload.buffer,
            left_context.storage.buffer,
            left_context.readback.buffer,
            left_context.buffer_size,
            left_context.storage_elements as u32,
            left_context.dispatch_groups,
        )?;
        record_compute_dispatch(
            right_device,
            &right_context.resources,
            right_context.command_buffer,
            right_context.query_pool,
            right_context.upload.buffer,
            right_context.storage.buffer,
            right_context.readback.buffer,
            right_context.buffer_size,
            right_context.storage_elements as u32,
            right_context.dispatch_groups,
        )?;
        let started = Instant::now();
        unsafe {
            left_device
                .device
                .reset_fences(&[left_context.fence])
                .map_err(|error| format!("could not reset Vulkan left-pair fence: {error:?}"))?;
            right_device
                .device
                .reset_fences(&[right_context.fence])
                .map_err(|error| format!("could not reset Vulkan right-pair fence: {error:?}"))?;
            left_device
                .device
                .queue_submit(
                    left_device.queue,
                    &[vk::SubmitInfo::default().command_buffers(&[left_context.command_buffer])],
                    left_context.fence,
                )
                .map_err(|error| format!("could not submit Vulkan left-pair work: {error:?}"))?;
            right_device
                .device
                .queue_submit(
                    right_device.queue,
                    &[vk::SubmitInfo::default().command_buffers(&[right_context.command_buffer])],
                    right_context.fence,
                )
                .map_err(|error| format!("could not submit Vulkan right-pair work: {error:?}"))?;
            left_device
                .device
                .wait_for_fences(&[left_context.fence], true, u64::MAX)
                .map_err(|error| format!("could not wait for Vulkan left-pair work: {error:?}"))?;
            right_device
                .device
                .wait_for_fences(&[right_context.fence], true, u64::MAX)
                .map_err(|error| format!("could not wait for Vulkan right-pair work: {error:?}"))?;
        }
        let left_duration = read_timestamp_duration_ns(left_device, left_context.query_pool)?;
        let right_duration = read_timestamp_duration_ns(right_device, right_context.query_pool)?;
        read_buffer_bytes(
            &left_device.device,
            &left_context.readback,
            &mut left_output,
        )?;
        read_buffer_bytes(
            &right_device.device,
            &right_context.readback,
            &mut right_output,
        )?;
        let duration = started.elapsed();
        black_box(checksum_bytes(&left_output));
        black_box(checksum_bytes(&right_output));
        black_box(read_first_storage_word(
            &left_device.device,
            &left_context.readback,
            &left_context.kernel,
        )?);
        black_box(read_first_storage_word(
            &right_device.device,
            &right_context.readback,
            &right_context.kernel,
        )?);
        measured_samples.push(Sample {
            sample_index,
            duration_ns: duration.as_nanos().max(left_duration).max(right_duration),
            iterations: 1,
            bytes_read: left_context.buffer_size as u64
                + right_context.buffer_size as u64
                + activation_bytes as u64
                + output_bytes as u64,
            bytes_written: left_context.buffer_size as u64
                + right_context.buffer_size as u64
                + output_bytes as u64,
            operations: (left_context.storage_elements as u64
                + right_context.storage_elements as u64)
                * kernel.logical_elements_per_storage_element
                * kernel.operations_per_storage_element,
        });
    }

    let summary = summarize_samples(&measured_samples);
    Ok(PairMeasurement {
        workload_id: format_workload_id(
            "synthetic_tensor_split_small_payload",
            workload_class,
            &kernel.format,
        ),
        comparison_group: "small_payload_placement_comparison".to_string(),
        workload_class: workload_class.to_string(),
        placement_strategy: "two_target_parallel".to_string(),
        source_target_id: left_id.to_string(),
        destination_target_id: right_id.to_string(),
        pattern: "synthetic_tensor_split_small_payload".to_string(),
        operation_family: workload_class.to_string(),
        regime: "small_payload".to_string(),
        format: kernel.format,
        status: "completed".to_string(),
        reason: None,
        payload_bytes,
        source_payload_bytes: left_payload_bytes,
        destination_payload_bytes: right_payload_bytes,
        activation_bytes,
        output_bytes,
        samples: measured_samples,
        summary,
    })
}

fn failed_dense_pair_measurements(
    source_id: &str,
    destination_id: &str,
    workload_prefix: &str,
    placement_strategy: &str,
    payload_bytes: usize,
    formats: &[String],
    workloads: &[String],
    reason: &str,
) -> Vec<PairMeasurement> {
    formats
        .iter()
        .flat_map(|format| {
            workloads.iter().filter_map(move |workload| {
                if workload_format_kernel(workload, format).is_none() {
                    return None;
                }
                Some(failed_dense_pair_measurement(
                    source_id,
                    destination_id,
                    workload_prefix,
                    placement_strategy,
                    payload_bytes,
                    workload,
                    format,
                    reason,
                ))
            })
        })
        .collect()
}

fn failed_dense_pair_measurement(
    source_id: &str,
    destination_id: &str,
    workload_prefix: &str,
    placement_strategy: &str,
    payload_bytes: usize,
    workload_class: &str,
    format: &str,
    reason: &str,
) -> PairMeasurement {
    let source_payload_bytes = payload_bytes / 2;
    let destination_payload_bytes = payload_bytes - source_payload_bytes;
    PairMeasurement {
        workload_id: format_workload_id(workload_prefix, workload_class, format),
        comparison_group: "small_payload_placement_comparison".to_string(),
        workload_class: workload_class.to_string(),
        placement_strategy: placement_strategy.to_string(),
        source_target_id: source_id.to_string(),
        destination_target_id: destination_id.to_string(),
        pattern: workload_prefix.to_string(),
        operation_family: workload_class.to_string(),
        regime: "small_payload".to_string(),
        format: format.to_string(),
        status: "failed".to_string(),
        reason: Some(reason.to_string()),
        payload_bytes,
        source_payload_bytes,
        destination_payload_bytes,
        activation_bytes: activation_bytes_for_payload(payload_bytes),
        output_bytes: output_bytes_for_payload(payload_bytes),
        samples: Vec::new(),
        summary: None,
    }
}

fn run_vulkan_dense_serial_triplet(
    devices: [&OpenVulkanComputeDevice; 3],
    target_ids: &[String; 3],
    payload_bytes: usize,
    samples: usize,
    workload_class: &str,
    kernel: DenseFormatKernel,
) -> Result<GroupMeasurement, String> {
    let payload_split = split_three_payload_bytes(payload_bytes);
    let activation_bytes = activation_bytes_for_payload(payload_bytes);
    let contexts = [
        create_dense_compute_context(devices[0], payload_split[0], kernel.clone())?,
        create_dense_compute_context(devices[1], payload_split[1], kernel.clone())?,
        create_dense_compute_context(devices[2], payload_split[2], kernel.clone())?,
    ];
    let mut first_transfer = create_host_staged_transfer(devices[0], devices[1], activation_bytes)?;
    let mut second_transfer =
        create_host_staged_transfer(devices[1], devices[2], activation_bytes)?;
    let mut measured_samples = Vec::with_capacity(samples);
    for sample_index in 0..samples {
        let started = Instant::now();
        let first_sample = submit_dense_compute_sample(devices[0], &contexts[0], sample_index)?;
        let first_transfer_sample =
            run_host_staged_transfer_sample(devices[0], devices[1], &mut first_transfer)?;
        let second_sample = submit_dense_compute_sample(devices[1], &contexts[1], sample_index)?;
        let second_transfer_sample =
            run_host_staged_transfer_sample(devices[1], devices[2], &mut second_transfer)?;
        let third_sample = submit_dense_compute_sample(devices[2], &contexts[2], sample_index)?;
        let duration = started.elapsed();
        black_box(read_first_storage_word(
            &devices[2].device,
            &contexts[2].readback,
            &contexts[2].kernel,
        )?);
        measured_samples.push(Sample {
            sample_index,
            duration_ns: duration.as_nanos(),
            iterations: 1,
            bytes_read: first_sample.bytes_read
                + second_sample.bytes_read
                + third_sample.bytes_read
                + first_transfer_sample.bytes_read
                + second_transfer_sample.bytes_read,
            bytes_written: first_sample.bytes_written
                + second_sample.bytes_written
                + third_sample.bytes_written
                + first_transfer_sample.bytes_written
                + second_transfer_sample.bytes_written,
            operations: first_sample.operations
                + second_sample.operations
                + third_sample.operations,
        });
    }

    let summary = summarize_samples(&measured_samples);
    Ok(GroupMeasurement {
        workload_id: format_workload_id(
            "synthetic_layer_split_group_small_payload",
            workload_class,
            &kernel.format,
        ),
        comparison_group: "small_payload_placement_comparison".to_string(),
        workload_class: workload_class.to_string(),
        placement_strategy: "three_target_serial".to_string(),
        target_ids: target_ids.to_vec(),
        pattern: "synthetic_layer_split_group_small_payload".to_string(),
        operation_family: workload_class.to_string(),
        regime: "small_payload".to_string(),
        format: kernel.format,
        status: "completed".to_string(),
        reason: None,
        participant_count: 3,
        payload_bytes,
        payload_bytes_per_participant: payload_split,
        activation_bytes,
        output_bytes: output_bytes_for_payload(payload_bytes),
        samples: measured_samples,
        summary,
    })
}

fn run_vulkan_dense_parallel_triplet(
    devices: [&OpenVulkanComputeDevice; 3],
    target_ids: &[String; 3],
    payload_bytes: usize,
    samples: usize,
    workload_class: &str,
    kernel: DenseFormatKernel,
) -> Result<GroupMeasurement, String> {
    let payload_split = split_three_payload_bytes(payload_bytes);
    let activation_bytes = activation_bytes_for_payload(payload_bytes);
    let output_bytes = output_bytes_for_payload(payload_bytes);
    let contexts = [
        create_dense_compute_context(devices[0], payload_split[0], kernel.clone())?,
        create_dense_compute_context(devices[1], payload_split[1], kernel.clone())?,
        create_dense_compute_context(devices[2], payload_split[2], kernel.clone())?,
    ];
    let mut outputs = contexts
        .iter()
        .map(|context| vec![0_u8; output_bytes.min(context.readback.size as usize)])
        .collect::<Vec<_>>();
    let mut measured_samples = Vec::with_capacity(samples);
    for sample_index in 0..samples {
        for index in 0..3 {
            record_compute_dispatch(
                devices[index],
                &contexts[index].resources,
                contexts[index].command_buffer,
                contexts[index].query_pool,
                contexts[index].upload.buffer,
                contexts[index].storage.buffer,
                contexts[index].readback.buffer,
                contexts[index].buffer_size,
                contexts[index].storage_elements as u32,
                contexts[index].dispatch_groups,
            )?;
        }
        let started = Instant::now();
        for index in 0..3 {
            unsafe {
                devices[index]
                    .device
                    .reset_fences(&[contexts[index].fence])
                    .map_err(|error| {
                        format!("could not reset Vulkan triplet fence {index}: {error:?}")
                    })?;
                devices[index]
                    .device
                    .queue_submit(
                        devices[index].queue,
                        &[vk::SubmitInfo::default()
                            .command_buffers(&[contexts[index].command_buffer])],
                        contexts[index].fence,
                    )
                    .map_err(|error| {
                        format!("could not submit Vulkan triplet work {index}: {error:?}")
                    })?;
            }
        }
        for index in 0..3 {
            unsafe {
                devices[index]
                    .device
                    .wait_for_fences(&[contexts[index].fence], true, u64::MAX)
                    .map_err(|error| {
                        format!("could not wait for Vulkan triplet work {index}: {error:?}")
                    })?;
            }
        }
        let timestamp_durations = [
            read_timestamp_duration_ns(devices[0], contexts[0].query_pool)?,
            read_timestamp_duration_ns(devices[1], contexts[1].query_pool)?,
            read_timestamp_duration_ns(devices[2], contexts[2].query_pool)?,
        ];
        for index in 0..3 {
            read_buffer_bytes(
                &devices[index].device,
                &contexts[index].readback,
                &mut outputs[index],
            )?;
            black_box(checksum_bytes(&outputs[index]));
        }
        let duration = started.elapsed();
        measured_samples.push(Sample {
            sample_index,
            duration_ns: timestamp_durations
                .into_iter()
                .fold(duration.as_nanos(), u128::max),
            iterations: 1,
            bytes_read: contexts
                .iter()
                .map(|context| context.buffer_size as u64)
                .sum::<u64>()
                + activation_bytes as u64
                + output_bytes as u64,
            bytes_written: contexts
                .iter()
                .map(|context| context.buffer_size as u64)
                .sum::<u64>()
                + output_bytes as u64,
            operations: contexts
                .iter()
                .map(|context| {
                    (context.storage_elements as u64)
                        * context.kernel.logical_elements_per_storage_element
                        * context.kernel.operations_per_storage_element
                })
                .sum(),
        });
    }

    let summary = summarize_samples(&measured_samples);
    Ok(GroupMeasurement {
        workload_id: format_workload_id(
            "synthetic_tensor_split_group_small_payload",
            workload_class,
            &kernel.format,
        ),
        comparison_group: "small_payload_placement_comparison".to_string(),
        workload_class: workload_class.to_string(),
        placement_strategy: "three_target_parallel".to_string(),
        target_ids: target_ids.to_vec(),
        pattern: "synthetic_tensor_split_group_small_payload".to_string(),
        operation_family: workload_class.to_string(),
        regime: "small_payload".to_string(),
        format: kernel.format,
        status: "completed".to_string(),
        reason: None,
        participant_count: 3,
        payload_bytes,
        payload_bytes_per_participant: payload_split,
        activation_bytes,
        output_bytes,
        samples: measured_samples,
        summary,
    })
}

fn failed_triplet_measurements(
    target_ids: &[String; 3],
    payload_bytes: usize,
    formats: &[String],
    workloads: &[String],
    reason: &str,
) -> Vec<GroupMeasurement> {
    formats
        .iter()
        .flat_map(|format| {
            workloads.iter().filter_map(move |workload| {
                if workload_format_kernel(workload, format).is_none() {
                    return None;
                }
                Some([
                    failed_triplet_measurement(
                        target_ids,
                        "synthetic_layer_split_group_small_payload",
                        "three_target_serial",
                        payload_bytes,
                        workload,
                        format,
                        reason,
                    ),
                    failed_triplet_measurement(
                        target_ids,
                        "synthetic_tensor_split_group_small_payload",
                        "three_target_parallel",
                        payload_bytes,
                        workload,
                        format,
                        reason,
                    ),
                ])
            })
        })
        .flatten()
        .collect()
}

fn failed_triplet_measurement(
    target_ids: &[String; 3],
    workload_prefix: &str,
    placement_strategy: &str,
    payload_bytes: usize,
    workload_class: &str,
    format: &str,
    reason: &str,
) -> GroupMeasurement {
    GroupMeasurement {
        workload_id: format_workload_id(workload_prefix, workload_class, format),
        comparison_group: "small_payload_placement_comparison".to_string(),
        workload_class: workload_class.to_string(),
        placement_strategy: placement_strategy.to_string(),
        target_ids: target_ids.to_vec(),
        pattern: workload_prefix.to_string(),
        operation_family: workload_class.to_string(),
        regime: "small_payload".to_string(),
        format: format.to_string(),
        status: "failed".to_string(),
        reason: Some(reason.to_string()),
        participant_count: 3,
        payload_bytes,
        payload_bytes_per_participant: split_three_payload_bytes(payload_bytes),
        activation_bytes: activation_bytes_for_payload(payload_bytes),
        output_bytes: output_bytes_for_payload(payload_bytes),
        samples: Vec::new(),
        summary: None,
    }
}

fn split_three_payload_bytes(payload_bytes: usize) -> Vec<usize> {
    let first = payload_bytes.div_ceil(3);
    let remaining = payload_bytes - first;
    let second = remaining.div_ceil(2);
    vec![first, second, remaining - second]
}

struct ComputeResources {
    device: ash::Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_set: vk::DescriptorSet,
    pipeline_layout: vk::PipelineLayout,
    descriptor_pool: vk::DescriptorPool,
    pipeline: vk::Pipeline,
}

impl Drop for ComputeResources {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

fn create_compute_resources(
    compute_device: &OpenVulkanComputeDevice,
    storage_buffer: vk::Buffer,
    buffer_size: vk::DeviceSize,
    shader: ShaderCode,
) -> Result<ComputeResources, String> {
    let device = &compute_device.device;
    let binding = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)];
    let descriptor_set_layout = unsafe {
        device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&binding),
            None,
        )
    }
    .map_err(|error| format!("could not create Vulkan descriptor set layout: {error:?}"))?;
    let push_ranges = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(mem::size_of::<u32>() as u32)];
    let set_layouts = [descriptor_set_layout];
    let pipeline_layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&set_layouts)
                .push_constant_ranges(&push_ranges),
            None,
        )
    }
    .map_err(|error| format!("could not create Vulkan pipeline layout: {error:?}"))?;
    let shader_words = shader_words(shader)?;
    let shader_module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&shader_words),
            None,
        )
    }
    .map_err(|error| format!("could not create Vulkan shader module: {error:?}"))?;
    let entry_name = CString::new("main").expect("static string has no nul");
    let shader_stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(&entry_name);
    let pipeline = unsafe {
        device.create_compute_pipelines(
            vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo::default()
                .stage(shader_stage)
                .layout(pipeline_layout)],
            None,
        )
    }
    .map_err(|(_, error)| format!("could not create Vulkan compute pipeline: {error:?}"))?[0];
    unsafe { device.destroy_shader_module(shader_module, None) };
    let pool_sizes = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)];
    let descriptor_pool = unsafe {
        device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&pool_sizes),
            None,
        )
    }
    .map_err(|error| format!("could not create Vulkan descriptor pool: {error:?}"))?;
    let descriptor_set = unsafe {
        device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&set_layouts),
        )
    }
    .map_err(|error| format!("could not allocate Vulkan descriptor set: {error:?}"))?[0];
    let buffer_info = [vk::DescriptorBufferInfo::default()
        .buffer(storage_buffer)
        .offset(0)
        .range(buffer_size)];
    unsafe {
        device.update_descriptor_sets(
            &[vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buffer_info)],
            &[],
        );
    }
    Ok(ComputeResources {
        device: device.clone(),
        descriptor_set_layout,
        descriptor_set,
        pipeline_layout,
        descriptor_pool,
        pipeline,
    })
}

fn shader_words(shader: ShaderCode) -> Result<Vec<u32>, String> {
    match shader {
        ShaderCode::Words(words) => Ok(words.to_vec()),
        ShaderCode::Bytes(bytes) => {
            if bytes.len() % mem::size_of::<u32>() != 0 {
                return Err("SPIR-V bytecode length is not u32-aligned".to_string());
            }
            Ok(bytes
                .chunks_exact(mem::size_of::<u32>())
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect())
        }
    }
}

fn record_compute_dispatch(
    compute_device: &OpenVulkanComputeDevice,
    resources: &ComputeResources,
    command_buffer: vk::CommandBuffer,
    query_pool: vk::QueryPool,
    upload_buffer: vk::Buffer,
    storage_buffer: vk::Buffer,
    readback_buffer: vk::Buffer,
    buffer_size: vk::DeviceSize,
    elements: u32,
    dispatch_groups: u32,
) -> Result<(), String> {
    let device = &compute_device.device;
    unsafe {
        device
            .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
            .map_err(|error| format!("could not reset Vulkan command buffer: {error:?}"))?;
        device
            .begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(|error| format!("could not begin Vulkan command buffer: {error:?}"))?;
        device.cmd_reset_query_pool(command_buffer, query_pool, 0, 2);
        device.cmd_copy_buffer(
            command_buffer,
            upload_buffer,
            storage_buffer,
            &[vk::BufferCopy::default().size(buffer_size)],
        );
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                .buffer(storage_buffer)
                .offset(0)
                .size(buffer_size)],
            &[],
        );
        device.cmd_write_timestamp(
            command_buffer,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            query_pool,
            0,
        );
        device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            resources.pipeline,
        );
        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            resources.pipeline_layout,
            0,
            &[resources.descriptor_set],
            &[],
        );
        let push_bytes = std::slice::from_raw_parts(
            (&elements as *const u32).cast::<u8>(),
            mem::size_of::<u32>(),
        );
        device.cmd_push_constants(
            command_buffer,
            resources.pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            push_bytes,
        );
        device.cmd_dispatch(command_buffer, dispatch_groups, 1, 1);
        device.cmd_write_timestamp(
            command_buffer,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            query_pool,
            1,
        );
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .buffer(storage_buffer)
                .offset(0)
                .size(buffer_size)],
            &[],
        );
        device.cmd_copy_buffer(
            command_buffer,
            storage_buffer,
            readback_buffer,
            &[vk::BufferCopy::default().size(buffer_size)],
        );
        device
            .end_command_buffer(command_buffer)
            .map_err(|error| format!("could not end Vulkan command buffer: {error:?}"))?;
    }
    Ok(())
}

fn read_timestamp_duration_ns(
    compute_device: &OpenVulkanComputeDevice,
    query_pool: vk::QueryPool,
) -> Result<u128, String> {
    let mut timestamps = [0_u64; 2];
    unsafe {
        compute_device
            .device
            .get_query_pool_results(
                query_pool,
                0,
                &mut timestamps,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
            .map_err(|error| format!("could not read Vulkan timestamp queries: {error:?}"))?;
    }
    let mask = if compute_device.timestamp_valid_bits >= 64 {
        u64::MAX
    } else {
        (1_u64 << compute_device.timestamp_valid_bits) - 1
    };
    let ticks = timestamps[1].wrapping_sub(timestamps[0]) & mask;
    Ok((ticks as f64 * compute_device.timestamp_period_ns as f64).round() as u128)
}

fn create_buffer(
    compute_device: &OpenVulkanComputeDevice,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<VulkanBuffer, String> {
    let device = &compute_device.device;
    let buffer = unsafe {
        device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )
    }
    .map_err(|error| format!("could not create Vulkan buffer: {error:?}"))?;
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let memory_type_index = memory_type_index(
        &compute_device.memory_properties,
        requirements.memory_type_bits,
        properties,
    )
    .ok_or_else(|| format!("could not find Vulkan memory type with flags {properties:?}"))?;
    let memory = unsafe {
        device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type_index),
            None,
        )
    }
    .map_err(|error| format!("could not allocate Vulkan buffer memory: {error:?}"))?;
    unsafe {
        device
            .bind_buffer_memory(buffer, memory, 0)
            .map_err(|error| format!("could not bind Vulkan buffer memory: {error:?}"))?;
    }
    Ok(VulkanBuffer {
        device: device.clone(),
        buffer,
        memory,
        size,
    })
}

fn fill_upload_buffer(
    device: &ash::Device,
    buffer: &VulkanBuffer,
    elements: usize,
    kernel: &DenseFormatKernel,
) -> Result<(), String> {
    let ptr = unsafe {
        device
            .map_memory(buffer.memory, 0, buffer.size, vk::MemoryMapFlags::empty())
            .map_err(|error| format!("could not map Vulkan upload buffer: {error:?}"))?
    };
    if kernel.format == "f32" {
        let values = ptr.cast::<f32>();
        for index in 0..elements {
            unsafe {
                values
                    .add(index)
                    .write(((index % 1024) as f32) * 0.001 + 1.0);
            }
        }
    } else {
        let values = ptr.cast::<u32>();
        for index in 0..elements {
            unsafe {
                values
                    .add(index)
                    .write((index as u32).wrapping_mul(2_654_435_761) ^ 0xa5a5_5a5a);
            }
        }
    }
    unsafe { device.unmap_memory(buffer.memory) };
    Ok(())
}

fn write_pattern_bytes(device: &ash::Device, buffer: &VulkanBuffer) -> Result<(), String> {
    let mut bytes = vec![0_u8; buffer.size as usize];
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = (index as u8).wrapping_mul(31).wrapping_add(17);
    }
    write_buffer_bytes(device, buffer, &bytes)
}

fn write_buffer_bytes(
    device: &ash::Device,
    buffer: &VulkanBuffer,
    bytes: &[u8],
) -> Result<(), String> {
    if bytes.len() > buffer.size as usize {
        return Err("host write exceeds Vulkan buffer size".to_string());
    }
    let mapped = unsafe {
        device
            .map_memory(buffer.memory, 0, buffer.size, vk::MemoryMapFlags::empty())
            .map_err(|error| format!("could not map Vulkan host-write buffer: {error:?}"))?
    };
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.cast::<u8>(), bytes.len());
        device.unmap_memory(buffer.memory);
    }
    Ok(())
}

fn read_buffer_bytes(
    device: &ash::Device,
    buffer: &VulkanBuffer,
    bytes: &mut [u8],
) -> Result<(), String> {
    if bytes.len() > buffer.size as usize {
        return Err("host read exceeds Vulkan buffer size".to_string());
    }
    let mapped = unsafe {
        device
            .map_memory(buffer.memory, 0, buffer.size, vk::MemoryMapFlags::empty())
            .map_err(|error| format!("could not map Vulkan host-read buffer: {error:?}"))?
    };
    unsafe {
        ptr::copy_nonoverlapping(mapped.cast::<u8>(), bytes.as_mut_ptr(), bytes.len());
        device.unmap_memory(buffer.memory);
    }
    Ok(())
}

fn checksum_bytes(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0_u64, |sum, byte| {
        sum.rotate_left(5).wrapping_add(u64::from(*byte))
    })
}

fn read_first_storage_word(
    device: &ash::Device,
    buffer: &VulkanBuffer,
    kernel: &DenseFormatKernel,
) -> Result<u32, String> {
    let ptr = unsafe {
        device
            .map_memory(
                buffer.memory,
                0,
                kernel.bytes_per_storage_element as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )
            .map_err(|error| format!("could not map Vulkan readback buffer: {error:?}"))?
    };
    let value = if kernel.format == "f32" {
        unsafe { ptr.cast::<f32>().read().to_bits() }
    } else {
        unsafe { ptr.cast::<u32>().read() }
    };
    unsafe { device.unmap_memory(buffer.memory) };
    Ok(value)
}

fn memory_type_index(
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    allowed_types: u32,
    required_flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    memory_properties.memory_types[..memory_properties.memory_type_count as usize]
        .iter()
        .enumerate()
        .find(|(index, memory_type)| {
            (allowed_types & (1 << index)) != 0
                && memory_type.property_flags.contains(required_flags)
        })
        .map(|(index, _)| index as u32)
}

fn summarize_samples(samples: &[Sample]) -> Option<Summary> {
    if samples.is_empty() {
        return None;
    }
    let mut durations = samples
        .iter()
        .map(|sample| sample.duration_ns)
        .collect::<Vec<_>>();
    durations.sort_unstable();
    let total_bytes = samples
        .iter()
        .map(|sample| sample.bytes_read + sample.bytes_written)
        .sum::<u64>() as f64;
    let total_operations = samples.iter().map(|sample| sample.operations).sum::<u64>() as f64;
    let total_seconds = samples
        .iter()
        .map(|sample| sample.duration_ns as f64 / 1_000_000_000.0)
        .sum::<f64>();
    Some(Summary {
        min_duration_ns: durations[0],
        median_duration_ns: durations[durations.len() / 2],
        bytes_per_second: total_bytes / total_seconds,
        operations_per_second: total_operations / total_seconds,
    })
}

fn open_compute_device(target: &Target) -> Result<OpenVulkanComputeDevice, String> {
    let vulkan = target
        .vulkan
        .as_ref()
        .ok_or_else(|| "target has no Vulkan physical-device metadata".to_string())?;
    let entry = unsafe { Entry::load() }
        .map_err(|error| format!("could not load Vulkan loader: {error}"))?;
    let app_name = CString::new("nerve-gpu-bench").expect("static string has no nul");
    let engine_name = CString::new("nerve").expect("static string has no nul");
    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(1)
        .engine_name(&engine_name)
        .engine_version(1)
        .api_version(vk::make_api_version(0, 1, 3, 0));
    let instance = unsafe {
        entry.create_instance(
            &vk::InstanceCreateInfo::default().application_info(&app_info),
            None,
        )
    }
    .map_err(|error| format!("could not create Vulkan instance: {error:?}"))?;
    let result =
        unsafe { open_compute_device_from_instance(instance, vulkan.physical_device_index) };
    match result {
        Ok(device) => Ok(device),
        Err((instance, message)) => {
            unsafe { instance.destroy_instance(None) };
            Err(message)
        }
    }
}

unsafe fn open_compute_device_from_instance(
    instance: ash::Instance,
    physical_device_index: usize,
) -> Result<OpenVulkanComputeDevice, (ash::Instance, String)> {
    let physical_devices = unsafe { instance.enumerate_physical_devices() }.map_err(|error| {
        (
            instance.clone(),
            format!("could not enumerate Vulkan physical devices: {error:?}"),
        )
    })?;
    let physical_device = *physical_devices.get(physical_device_index).ok_or_else(|| {
        (
            instance.clone(),
            format!("Vulkan physical device index {physical_device_index} is no longer available"),
        )
    })?;
    let queue_families =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let compute_queue_family_index =
        compute_queue_family_index(&queue_families).ok_or_else(|| {
            (
                instance.clone(),
                format!(
                    "Vulkan physical device index {physical_device_index} has no compute queue"
                ),
            )
        })?;
    let timestamp_valid_bits =
        queue_families[compute_queue_family_index as usize].timestamp_valid_bits;
    let physical_device_properties =
        unsafe { instance.get_physical_device_properties(physical_device) };
    let memory_properties =
        unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let priorities = [1.0_f32];
    let queue_info = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(compute_queue_family_index)
        .queue_priorities(&priorities)];
    let device_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_info);
    let device = unsafe { instance.create_device(physical_device, &device_info, None) }.map_err(
        |error| {
            (
                instance.clone(),
                format!("could not create Vulkan logical device: {error:?}"),
            )
        },
    )?;
    let queue = unsafe { device.get_device_queue(compute_queue_family_index, 0) };
    Ok(OpenVulkanComputeDevice {
        device,
        instance,
        compute_queue_family_index,
        queue,
        memory_properties,
        timestamp_period_ns: physical_device_properties.limits.timestamp_period,
        timestamp_valid_bits,
    })
}

fn compute_queue_family_index(queue_families: &[vk::QueueFamilyProperties]) -> Option<u32> {
    queue_families
        .iter()
        .enumerate()
        .filter(|(_, family)| {
            family.queue_count > 0 && family.queue_flags.contains(vk::QueueFlags::COMPUTE)
        })
        .min_by_key(|(_, family)| family.queue_flags.contains(vk::QueueFlags::GRAPHICS))
        .map(|(index, _)| index as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefers_compute_only_queue_family() {
        let queue_families = [
            vk::QueueFamilyProperties {
                queue_flags: vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
                queue_count: 1,
                ..Default::default()
            },
            vk::QueueFamilyProperties {
                queue_flags: vk::QueueFlags::COMPUTE,
                queue_count: 1,
                ..Default::default()
            },
        ];
        assert_eq!(compute_queue_family_index(&queue_families), Some(1));
    }

    #[test]
    fn ignores_queue_families_without_compute_or_queues() {
        let queue_families = [
            vk::QueueFamilyProperties {
                queue_flags: vk::QueueFlags::GRAPHICS,
                queue_count: 1,
                ..Default::default()
            },
            vk::QueueFamilyProperties {
                queue_flags: vk::QueueFlags::COMPUTE,
                queue_count: 0,
                ..Default::default()
            },
        ];
        assert_eq!(compute_queue_family_index(&queue_families), None);
    }

    #[test]
    fn finds_memory_type_with_required_flags() {
        let mut memory_properties = vk::PhysicalDeviceMemoryProperties {
            memory_type_count: 2,
            ..Default::default()
        };
        memory_properties.memory_types[0].property_flags = vk::MemoryPropertyFlags::HOST_VISIBLE;
        memory_properties.memory_types[1].property_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        assert_eq!(
            memory_type_index(
                &memory_properties,
                0b11,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ),
            Some(1)
        );
    }

    #[test]
    fn maps_model_storage_formats_to_packed_dense_kernel() {
        for (format, logical_elements) in [
            ("f16", 2),
            ("bf16", 2),
            ("fp8_e4m3", 4),
            ("fp8_e5m2", 4),
            ("mxfp4", 8),
            ("nvfp4", 8),
            ("int4", 8),
            ("q5_1", 6),
            ("q4_k", 8),
            ("iq4_xs", 8),
            ("iq2_xs", 16),
        ] {
            let kernel = dense_format_kernel(format).unwrap();
            assert_eq!(kernel.format, format);
            assert_eq!(kernel.pattern, "single_target_packed_emulated_compute");
            assert_eq!(
                kernel.logical_elements_per_storage_element,
                logical_elements
            );
        }
    }

    #[test]
    fn maps_model_storage_formats_to_workload_specific_kernels() {
        for workload in ["dense_projection", "moe_expert", "router_reduction"] {
            let f32_kernel = workload_format_kernel(workload, "f32").unwrap();
            assert_eq!(f32_kernel.format, "f32");
            assert!(f32_kernel.operations_per_storage_element > 0);

            let packed_kernel = workload_format_kernel(workload, "mxfp4").unwrap();
            assert_eq!(packed_kernel.format, "mxfp4");
            assert_eq!(packed_kernel.logical_elements_per_storage_element, 8);
            assert!(packed_kernel.pattern.contains("compute"));
        }
        assert!(workload_format_kernel("unknown_workload", "f32").is_none());
        assert!(workload_format_kernel("dense_projection", "unknown_format").is_none());
    }

    #[test]
    fn unknown_vulkan_axes_are_skipped() {
        let formats = vec!["mxfp4".to_string(), "unknown_format".to_string()];
        let workloads = vec![
            "dense_projection".to_string(),
            "moe_expert".to_string(),
            "router_reduction".to_string(),
        ];
        let skipped = skipped_vulkan_axes(&formats, &workloads);
        assert!(!skipped.contains(&("mxfp4", "dense_projection")));
        assert!(!skipped.contains(&("mxfp4", "moe_expert")));
        assert!(!skipped.contains(&("mxfp4", "router_reduction")));
        assert!(skipped.contains(&("unknown_format", "dense_projection")));
        assert!(skipped.contains(&("unknown_format", "moe_expert")));
        assert!(skipped.contains(&("unknown_format", "router_reduction")));
    }

    #[test]
    fn failed_transfer_measurement_preserves_ordered_pair_metadata() {
        let measurement = failed_transfer_measurement(
            "vulkan:pci:0000:01:00.0",
            "vulkan:pci:0000:02:00.0",
            5 * 1024 * 1024,
            "dense_projection",
            "mxfp4",
            "boom",
        );
        assert_eq!(
            measurement.workload_id,
            "ordered_activation_transfer:dense_projection:mxfp4"
        );
        assert_eq!(measurement.placement_strategy, "activation_transfer_only");
        assert_eq!(measurement.source_target_id, "vulkan:pci:0000:01:00.0");
        assert_eq!(measurement.destination_target_id, "vulkan:pci:0000:02:00.0");
        assert_eq!(measurement.status, "failed");
        assert_eq!(measurement.activation_bytes, 256 * 1024);
        assert!(measurement.samples.is_empty());
    }

    #[test]
    fn dense_pair_failure_rows_cover_only_executable_axes() {
        let formats = vec![
            "f32".to_string(),
            "mxfp4".to_string(),
            "unknown_format".to_string(),
        ];
        let workloads = vec![
            "dense_projection".to_string(),
            "router_reduction".to_string(),
        ];
        let measurements = failed_dense_pair_measurements(
            "left",
            "right",
            "synthetic_tensor_split_small_payload",
            "two_target_parallel",
            64 * 1024,
            &formats,
            &workloads,
            "no device",
        );
        let ids = measurements
            .iter()
            .map(|measurement| measurement.workload_id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            ids,
            [
                "synthetic_tensor_split_small_payload:dense_projection:f32",
                "synthetic_tensor_split_small_payload:router_reduction:f32",
                "synthetic_tensor_split_small_payload:dense_projection:mxfp4",
                "synthetic_tensor_split_small_payload:router_reduction:mxfp4",
            ]
        );
        assert!(measurements.iter().all(|measurement| {
            measurement.placement_strategy == "two_target_parallel"
                && measurement.status == "failed"
                && measurement.source_target_id == "left"
                && measurement.destination_target_id == "right"
        }));
    }

    #[test]
    fn triplet_payload_split_preserves_total_bytes() {
        assert_eq!(split_three_payload_bytes(10), [4, 3, 3]);
        assert_eq!(split_three_payload_bytes(11), [4, 4, 3]);
        assert_eq!(split_three_payload_bytes(12), [4, 4, 4]);
    }

    #[test]
    fn failed_triplet_measurements_cover_serial_and_parallel_axes() {
        let target_ids = ["a".to_string(), "b".to_string(), "c".to_string()];
        let formats = vec!["f32".to_string(), "unknown_format".to_string()];
        let workloads = vec!["moe_expert".to_string()];
        let measurements =
            failed_triplet_measurements(&target_ids, 10, &formats, &workloads, "no device");
        let strategies = measurements
            .iter()
            .map(|measurement| measurement.placement_strategy.as_str())
            .collect::<Vec<_>>();
        assert_eq!(strategies, ["three_target_serial", "three_target_parallel"]);
        assert!(measurements.iter().all(|measurement| {
            measurement.target_ids == target_ids
                && measurement.participant_count == 3
                && measurement.payload_bytes_per_participant == [4, 3, 3]
                && measurement.status == "failed"
        }));
    }

    #[test]
    fn summarizes_vulkan_samples() {
        let samples = [
            Sample {
                sample_index: 0,
                duration_ns: 10,
                iterations: 1,
                bytes_read: 4,
                bytes_written: 4,
                operations: 2,
            },
            Sample {
                sample_index: 1,
                duration_ns: 20,
                iterations: 1,
                bytes_read: 4,
                bytes_written: 4,
                operations: 2,
            },
        ];
        let summary = summarize_samples(&samples).unwrap();
        assert_eq!(summary.min_duration_ns, 10);
        assert_eq!(summary.median_duration_ns, 20);
        assert!(summary.bytes_per_second > 0.0);
        assert!(summary.operations_per_second > 0.0);
    }
}
