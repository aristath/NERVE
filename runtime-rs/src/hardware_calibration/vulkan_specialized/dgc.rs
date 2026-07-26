use super::{SpecializedBuffer, SpecializedVulkanContext, SpecializedVulkanResources};
use ash::vk;
use sha2::{Digest, Sha256};
use std::ffi::{CStr, c_void};
use std::rc::Rc;

pub(crate) struct PreparedDeviceGeneratedCommands {
    context: Rc<SpecializedVulkanContext>,
    functions: DeviceGeneratedCommandsFunctions,
    resources: SpecializedVulkanResources,
    _indirect: SpecializedBuffer,
    _preprocess: SpecializedBuffer,
    output: SpecializedBuffer,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    shader_module: vk::ShaderModule,
    pipeline: vk::Pipeline,
    indirect_layout: u64,
}

impl PreparedDeviceGeneratedCommands {
    pub(in crate::hardware_calibration) fn new(
        context: Rc<SpecializedVulkanContext>,
        spirv: &[u32],
        dispatch_count: u32,
    ) -> Result<Self, String> {
        if dispatch_count == 0 {
            return Err("device-generated command count must be nonzero".to_string());
        }
        let functions = DeviceGeneratedCommandsFunctions::load(&context)?;
        let output = context.create_buffer(
            4096,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL
                | vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
            true,
        )?;
        output.write(&vec![0; 4096])?;
        let binding = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];
        let descriptor_set_layout = unsafe {
            context.device().create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&binding),
                None,
            )
        }
        .map_err(|error| format!("could not create DGC descriptor layout: {error:?}"))?;
        let set_layouts = [descriptor_set_layout];
        let pipeline_layout = unsafe {
            context.device().create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
                None,
            )
        }
        .map_err(|error| format!("could not create DGC pipeline layout: {error:?}"))?;
        let shader_module = unsafe {
            context
                .device()
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spirv), None)
        }
        .map_err(|error| format!("could not create DGC shader: {error:?}"))?;
        let main = CStr::from_bytes_with_nul(b"main\0").expect("static entry has nul");
        let pipeline = unsafe {
            context.device().create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::COMPUTE)
                            .module(shader_module)
                            .name(main),
                    )
                    .layout(pipeline_layout)],
                None,
            )
        }
        .map_err(|(_, error)| format!("could not create DGC compute pipeline: {error:?}"))?
        .remove(0);
        let descriptor_pool = unsafe {
            context.device().create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&[vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::STORAGE_BUFFER,
                        descriptor_count: 1,
                    }]),
                None,
            )
        }
        .map_err(|error| format!("could not create DGC descriptor pool: {error:?}"))?;
        let descriptor_set = unsafe {
            context.device().allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&set_layouts),
            )
        }
        .map_err(|error| format!("could not allocate DGC descriptor set: {error:?}"))?
        .remove(0);
        let output_info = [vk::DescriptorBufferInfo::default()
            .buffer(output.buffer)
            .offset(0)
            .range(4096)];
        unsafe {
            context.device().update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&output_info)],
                &[],
            );
        }
        let token = IndirectCommandsLayoutTokenExt {
            s_type: STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT,
            p_next: std::ptr::null(),
            token_type: INDIRECT_COMMANDS_TOKEN_TYPE_DISPATCH_EXT,
            data: IndirectCommandsTokenDataExt {
                pointer: std::ptr::null(),
            },
            offset: 0,
        };
        let layout_info = IndirectCommandsLayoutCreateInfoExt {
            s_type: STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_CREATE_INFO_EXT,
            p_next: std::ptr::null(),
            flags: 0,
            shader_stages: vk::ShaderStageFlags::COMPUTE.as_raw(),
            indirect_stride: 12,
            pipeline_layout,
            token_count: 1,
            tokens: std::ptr::from_ref(&token),
        };
        let mut indirect_layout = 0u64;
        let create_result = unsafe {
            (functions.create_indirect_commands_layout)(
                context.device().handle(),
                &layout_info,
                std::ptr::null(),
                &mut indirect_layout,
            )
        };
        if create_result != 0 || indirect_layout == 0 {
            return Err(format!(
                "vkCreateIndirectCommandsLayoutEXT failed with {create_result}"
            ));
        }
        let mut commands = Vec::with_capacity(dispatch_count as usize * 12);
        for _ in 0..dispatch_count {
            commands.extend_from_slice(&1u32.to_le_bytes());
            commands.extend_from_slice(&1u32.to_le_bytes());
            commands.extend_from_slice(&1u32.to_le_bytes());
        }
        let indirect = context.create_initialized_device_buffer(
            &commands,
            vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )?;
        let pipeline_info = GeneratedCommandsPipelineInfoExt {
            s_type: STRUCTURE_TYPE_GENERATED_COMMANDS_PIPELINE_INFO_EXT,
            p_next: std::ptr::null_mut(),
            pipeline,
        };
        let memory_info = GeneratedCommandsMemoryRequirementsInfoExt {
            s_type: STRUCTURE_TYPE_GENERATED_COMMANDS_MEMORY_REQUIREMENTS_INFO_EXT,
            p_next: std::ptr::from_ref(&pipeline_info).cast(),
            indirect_execution_set: 0,
            indirect_commands_layout: indirect_layout,
            max_sequence_count: dispatch_count,
            max_draw_count: 0,
        };
        let mut preprocess_requirements = vk::MemoryRequirements2::default();
        unsafe {
            (functions.get_generated_commands_memory_requirements)(
                context.device().handle(),
                &memory_info,
                &mut preprocess_requirements,
            );
        }
        let preprocess_size = preprocess_requirements.memory_requirements.size.max(256);
        let preprocess = context.create_buffer(
            preprocess_size,
            vk::BufferUsageFlags::from_raw(PREPROCESS_BUFFER_USAGE_EXT)
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            false,
        )?;
        let command_info = GeneratedCommandsInfoExt {
            s_type: STRUCTURE_TYPE_GENERATED_COMMANDS_INFO_EXT,
            p_next: std::ptr::from_ref(&pipeline_info).cast(),
            shader_stages: vk::ShaderStageFlags::COMPUTE.as_raw(),
            indirect_execution_set: 0,
            indirect_commands_layout: indirect_layout,
            indirect_address: indirect.device_address(),
            indirect_address_size: indirect.size,
            preprocess_address: preprocess.device_address(),
            preprocess_size,
            max_sequence_count: dispatch_count,
            sequence_count_address: 0,
            max_draw_count: 0,
        };
        let resources =
            SpecializedVulkanResources::new(Rc::clone(&context), context.compute_queue_family()?)?;
        resources.begin()?;
        unsafe {
            context.device().cmd_bind_pipeline(
                resources.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline,
            );
            context.device().cmd_bind_descriptor_sets(
                resources.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            (functions.cmd_execute_generated_commands)(resources.command_buffer, 0, &command_info);
            context.device().cmd_pipeline_barrier2(
                resources.command_buffer,
                &vk::DependencyInfo::default().buffer_memory_barriers(&[
                    vk::BufferMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                        .dst_access_mask(vk::AccessFlags2::HOST_READ)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .buffer(output.buffer)
                        .offset(0)
                        .size(4096),
                ]),
            );
        }
        resources.finish_recording()?;
        Ok(Self {
            context,
            functions,
            resources,
            _indirect: indirect,
            _preprocess: preprocess,
            output,
            descriptor_pool,
            descriptor_set_layout,
            pipeline_layout,
            shader_module,
            pipeline,
            indirect_layout,
        })
    }

    pub(in crate::hardware_calibration) fn run(&self) -> Result<u64, String> {
        self.resources.run(1_000_000_000)
    }

    pub(in crate::hardware_calibration) fn observed_digest(&self) -> Result<String, String> {
        let bytes = self.output.read(4096)?;
        if bytes.iter().all(|byte| *byte == 0) {
            return Err("device-generated commands produced unchanged output".to_string());
        }
        Ok(format!(
            "nerve.calibration_output_sha256.v1:{:x}",
            Sha256::digest(bytes)
        ))
    }
}

impl Drop for PreparedDeviceGeneratedCommands {
    fn drop(&mut self) {
        unsafe {
            let device = self.context.device();
            let _ = device.device_wait_idle();
            (self.functions.destroy_indirect_commands_layout)(
                device.handle(),
                self.indirect_layout,
                std::ptr::null(),
            );
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_shader_module(self.shader_module, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

pub(in crate::hardware_calibration) fn device_generated_commands_shader() -> &'static str {
    r#"#version 460
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) buffer OutputWords { uint words[]; } output_words;
void main() {
    uint lane = gl_LocalInvocationIndex;
    atomicAdd(output_words.words[lane & 1023u], lane + 1u);
}
"#
}

struct DeviceGeneratedCommandsFunctions {
    get_generated_commands_memory_requirements: GetGeneratedCommandsMemoryRequirements,
    cmd_execute_generated_commands: CmdExecuteGeneratedCommands,
    create_indirect_commands_layout: CreateIndirectCommandsLayout,
    destroy_indirect_commands_layout: DestroyIndirectCommandsLayout,
}

impl DeviceGeneratedCommandsFunctions {
    fn load(context: &SpecializedVulkanContext) -> Result<Self, String> {
        validate_dgc_abi()?;
        Ok(Self {
            get_generated_commands_memory_requirements: load_function(
                context,
                c"vkGetGeneratedCommandsMemoryRequirementsEXT",
            )?,
            cmd_execute_generated_commands: load_function(
                context,
                c"vkCmdExecuteGeneratedCommandsEXT",
            )?,
            create_indirect_commands_layout: load_function(
                context,
                c"vkCreateIndirectCommandsLayoutEXT",
            )?,
            destroy_indirect_commands_layout: load_function(
                context,
                c"vkDestroyIndirectCommandsLayoutEXT",
            )?,
        })
    }
}

fn validate_dgc_abi() -> Result<(), String> {
    let layouts = [
        (
            "VkGeneratedCommandsMemoryRequirementsInfoEXT",
            std::mem::size_of::<GeneratedCommandsMemoryRequirementsInfoExt>(),
            40,
        ),
        (
            "VkGeneratedCommandsInfoEXT",
            std::mem::size_of::<GeneratedCommandsInfoExt>(),
            96,
        ),
        (
            "VkGeneratedCommandsPipelineInfoEXT",
            std::mem::size_of::<GeneratedCommandsPipelineInfoExt>(),
            24,
        ),
        (
            "VkIndirectCommandsLayoutTokenEXT",
            std::mem::size_of::<IndirectCommandsLayoutTokenExt>(),
            40,
        ),
        (
            "VkIndirectCommandsLayoutCreateInfoEXT",
            std::mem::size_of::<IndirectCommandsLayoutCreateInfoExt>(),
            56,
        ),
    ];
    for (name, observed, expected) in layouts {
        if observed != expected {
            return Err(format!(
                "{name} ABI size is {observed}, Vulkan 1.4 requires {expected}"
            ));
        }
    }
    if std::mem::offset_of!(GeneratedCommandsInfoExt, sequence_count_address) != 80 {
        return Err("VkGeneratedCommandsInfoEXT ABI offset mismatch".to_string());
    }
    Ok(())
}

fn load_function<T: Copy>(
    context: &SpecializedVulkanContext,
    name: &'static CStr,
) -> Result<T, String> {
    let address = context.device_proc_address(name);
    if address.is_null() {
        return Err(format!(
            "Vulkan device did not expose {}",
            name.to_string_lossy()
        ));
    }
    if std::mem::size_of::<T>() != std::mem::size_of::<*const c_void>() {
        return Err("Vulkan function-pointer representation mismatch".to_string());
    }
    Ok(unsafe { std::mem::transmute_copy(&address) })
}

type GetGeneratedCommandsMemoryRequirements = unsafe extern "system" fn(
    vk::Device,
    *const GeneratedCommandsMemoryRequirementsInfoExt,
    *mut vk::MemoryRequirements2<'static>,
);
type CmdExecuteGeneratedCommands =
    unsafe extern "system" fn(vk::CommandBuffer, u32, *const GeneratedCommandsInfoExt);
type CreateIndirectCommandsLayout = unsafe extern "system" fn(
    vk::Device,
    *const IndirectCommandsLayoutCreateInfoExt,
    *const vk::AllocationCallbacks<'static>,
    *mut u64,
) -> i32;
type DestroyIndirectCommandsLayout =
    unsafe extern "system" fn(vk::Device, u64, *const vk::AllocationCallbacks<'static>);

#[repr(C)]
struct GeneratedCommandsMemoryRequirementsInfoExt {
    s_type: i32,
    p_next: *const c_void,
    indirect_execution_set: u64,
    indirect_commands_layout: u64,
    max_sequence_count: u32,
    max_draw_count: u32,
}

#[repr(C)]
struct GeneratedCommandsInfoExt {
    s_type: i32,
    p_next: *const c_void,
    shader_stages: u32,
    indirect_execution_set: u64,
    indirect_commands_layout: u64,
    indirect_address: u64,
    indirect_address_size: u64,
    preprocess_address: u64,
    preprocess_size: u64,
    max_sequence_count: u32,
    sequence_count_address: u64,
    max_draw_count: u32,
}

#[repr(C)]
struct GeneratedCommandsPipelineInfoExt {
    s_type: i32,
    p_next: *mut c_void,
    pipeline: vk::Pipeline,
}

#[repr(C)]
union IndirectCommandsTokenDataExt {
    pointer: *const c_void,
}

#[repr(C)]
struct IndirectCommandsLayoutTokenExt {
    s_type: i32,
    p_next: *const c_void,
    token_type: i32,
    data: IndirectCommandsTokenDataExt,
    offset: u32,
}

#[repr(C)]
struct IndirectCommandsLayoutCreateInfoExt {
    s_type: i32,
    p_next: *const c_void,
    flags: u32,
    shader_stages: u32,
    indirect_stride: u32,
    pipeline_layout: vk::PipelineLayout,
    token_count: u32,
    tokens: *const IndirectCommandsLayoutTokenExt,
}

const STRUCTURE_TYPE_GENERATED_COMMANDS_MEMORY_REQUIREMENTS_INFO_EXT: i32 = 1_000_572_002;
const STRUCTURE_TYPE_GENERATED_COMMANDS_INFO_EXT: i32 = 1_000_572_004;
const STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_CREATE_INFO_EXT: i32 = 1_000_572_006;
const STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT: i32 = 1_000_572_007;
const STRUCTURE_TYPE_GENERATED_COMMANDS_PIPELINE_INFO_EXT: i32 = 1_000_572_013;
const INDIRECT_COMMANDS_TOKEN_TYPE_DISPATCH_EXT: i32 = 9;
const PREPROCESS_BUFFER_USAGE_EXT: u32 = 0x8000_0000;
