use std::ffi::CString;
use std::hint::black_box;
use std::mem;

use ash::{Entry, vk};

use crate::benchmark::{single_target_status_measurement, single_target_status_measurements};
use crate::model::{Measurement, Sample, Summary, Target};

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

struct DenseFormatKernel {
    format: String,
    shader: &'static [u32],
    bytes_per_storage_element: usize,
    logical_elements_per_storage_element: u64,
    operations_per_storage_element: u64,
    pattern: &'static str,
}

fn dense_format_kernel(format: &str) -> Option<DenseFormatKernel> {
    match format {
        "f32" => Some(DenseFormatKernel {
            format: "f32".to_string(),
            shader: F32_TRANSFORM_SHADER_SPV,
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
        shader: PACKED_U32_TRANSFORM_SHADER_SPV,
        bytes_per_storage_element: mem::size_of::<u32>(),
        logical_elements_per_storage_element,
        operations_per_storage_element: 4,
        pattern: "single_target_packed_emulated_compute",
    }
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
            workloads
                .iter()
                .map(move |workload| match (format.as_str(), workload.as_str()) {
                    (_, "dense_projection") => {
                        if let Some(kernel) = dense_format_kernel(format) {
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
                        } else {
                            unsupported_vulkan_format(target_id, payload_bytes, workload, format)
                        }
                    }
                    _ => {
                        if dense_format_kernel(format).is_some() {
                            single_target_status_measurement(
                                target_id,
                                payload_bytes,
                                workload,
                                format,
                                "unmeasured",
                                "vulkan_kernel_not_implemented_for_workload",
                            )
                        } else {
                            unsupported_vulkan_format(target_id, payload_bytes, workload, format)
                        }
                    }
                })
        })
        .collect()
}

fn unsupported_vulkan_format(
    target_id: &str,
    payload_bytes: usize,
    workload: &str,
    format: &str,
) -> Measurement {
    single_target_status_measurement(
        target_id,
        payload_bytes,
        workload,
        format,
        "unsupported",
        "vulkan_execution_backend_has_no_kernel_for_format",
    )
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

fn run_vulkan_dense_projection(
    compute_device: &OpenVulkanComputeDevice,
    target_id: &str,
    payload_bytes: usize,
    samples: usize,
    workload_class: &str,
    kernel: DenseFormatKernel,
) -> Result<Measurement, String> {
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

    let dispatch_groups = storage_elements.div_ceil(64) as u32;
    let mut measured_samples = Vec::with_capacity(samples);
    for sample_index in 0..samples {
        record_compute_dispatch(
            compute_device,
            &resources,
            command_buffer,
            query_pool,
            upload.buffer,
            storage.buffer,
            readback.buffer,
            buffer_size,
            storage_elements as u32,
            dispatch_groups,
        )?;
        unsafe {
            compute_device
                .device
                .reset_fences(&[fence])
                .map_err(|error| format!("could not reset Vulkan fence: {error:?}"))?;
            compute_device
                .device
                .queue_submit(
                    compute_device.queue,
                    &[vk::SubmitInfo::default().command_buffers(&[command_buffer])],
                    fence,
                )
                .map_err(|error| format!("could not submit Vulkan compute work: {error:?}"))?;
            compute_device
                .device
                .wait_for_fences(&[fence], true, u64::MAX)
                .map_err(|error| format!("could not wait for Vulkan compute work: {error:?}"))?;
        }
        measured_samples.push(Sample {
            sample_index,
            duration_ns: read_timestamp_duration_ns(compute_device, query_pool)?,
            iterations: 1,
            bytes_read: buffer_size as u64,
            bytes_written: buffer_size as u64,
            operations: (storage_elements as u64)
                * kernel.logical_elements_per_storage_element
                * kernel.operations_per_storage_element,
        });
        black_box(read_first_storage_word(
            &compute_device.device,
            &readback,
            &kernel,
        )?);
    }

    unsafe {
        compute_device.device.destroy_fence(fence, None);
        compute_device.device.destroy_query_pool(query_pool, None);
        compute_device
            .device
            .free_command_buffers(command_pool, &[command_buffer]);
        compute_device
            .device
            .destroy_command_pool(command_pool, None);
    }

    Ok(Measurement {
        workload_id: format!(
            "single_target_small_payload:{workload_class}:{}",
            kernel.format
        ),
        comparison_group: "small_payload_placement_comparison".to_string(),
        workload_class: workload_class.to_string(),
        placement_strategy: "single_target_serial".to_string(),
        target_id: target_id.to_string(),
        pattern: kernel.pattern.to_string(),
        operation_family: workload_class.to_string(),
        regime: "small_payload".to_string(),
        format: kernel.format.to_string(),
        status: "completed".to_string(),
        reason: None,
        payload_bytes,
        working_set_bytes: buffer_size as usize,
        summary: summarize_samples(&measured_samples),
        samples: measured_samples,
    })
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
    shader: &[u32],
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
    let shader_module = unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(shader), None)
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
