use super::{
    SpecializedBuffer, SpecializedImage, SpecializedVulkanContext, SpecializedVulkanResources,
};
use ash::vk;
use sha2::{Digest, Sha256};
use std::ffi::CStr;
use std::rc::Rc;

pub(crate) struct PreparedFixedGraphics {
    context: Rc<SpecializedVulkanContext>,
    resources: SpecializedVulkanResources,
    _color_image: SpecializedImage,
    _depth_image: Option<SpecializedImage>,
    output: SpecializedBuffer,
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    pipeline_layout: vk::PipelineLayout,
    vertex_shader: vk::ShaderModule,
    fragment_shader: vk::ShaderModule,
    pipeline: vk::Pipeline,
}

impl PreparedFixedGraphics {
    pub(in crate::hardware_calibration) fn new(
        context: Rc<SpecializedVulkanContext>,
        operation: &str,
        vertex_spirv: &[u32],
        fragment_spirv: &[u32],
        width: u32,
        height: u32,
        overdraw: u32,
    ) -> Result<Self, String> {
        if !matches!(
            operation,
            "rasterization" | "fixed_function_interpolation" | "depth_stencil" | "blending"
        ) {
            return Err(format!(
                "fixed graphics calibrator does not implement {operation:?}"
            ));
        }
        if width == 0 || height == 0 || overdraw == 0 {
            return Err("fixed graphics dimensions and overdraw must be nonzero".to_string());
        }
        let device = context.device();
        let queue_family = context.graphics_queue_family()?;
        let resources = SpecializedVulkanResources::new(Rc::clone(&context), queue_family)?;
        let extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };
        let color_format = vk::Format::R16G16B16A16_SFLOAT;
        let depth_format = vk::Format::D32_SFLOAT;
        let color_image = context.create_image(
            extent,
            color_format,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::ImageAspectFlags::COLOR,
        )?;
        let use_depth = operation == "depth_stencil";
        let depth_image = use_depth
            .then(|| {
                context.create_image(
                    extent,
                    depth_format,
                    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                    vk::ImageAspectFlags::DEPTH,
                )
            })
            .transpose()?;
        let output = context.create_buffer(
            4096,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            true,
        )?;
        output.write(&vec![0; 4096])?;
        let render_pass =
            create_render_pass(device, color_format, use_depth.then_some(depth_format))?;
        let attachments = depth_image
            .as_ref()
            .map(|depth| vec![color_image.view, depth.view])
            .unwrap_or_else(|| vec![color_image.view]);
        let framebuffer = unsafe {
            device.create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(width)
                    .height(height)
                    .layers(1),
                None,
            )
        }
        .map_err(|error| format!("could not create graphics framebuffer: {error:?}"))?;
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default(), None)
        }
        .map_err(|error| format!("could not create graphics pipeline layout: {error:?}"))?;
        let vertex_shader = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(vertex_spirv),
                None,
            )
        }
        .map_err(|error| format!("could not create graphics vertex shader: {error:?}"))?;
        let fragment_shader = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(fragment_spirv),
                None,
            )
        }
        .map_err(|error| format!("could not create graphics fragment shader: {error:?}"))?;
        let pipeline = create_pipeline(
            device,
            render_pass,
            pipeline_layout,
            vertex_shader,
            fragment_shader,
            width,
            height,
            operation == "blending",
            use_depth,
        )?;
        record_graphics_commands(
            &context,
            &resources,
            render_pass,
            framebuffer,
            pipeline,
            width,
            height,
            overdraw,
            use_depth,
            color_image.image,
            output.buffer,
        )?;
        Ok(Self {
            context,
            resources,
            _color_image: color_image,
            _depth_image: depth_image,
            output,
            render_pass,
            framebuffer,
            pipeline_layout,
            vertex_shader,
            fragment_shader,
            pipeline,
        })
    }

    pub(in crate::hardware_calibration) fn run(&self) -> Result<u64, String> {
        self.resources.run(1_000_000_000)
    }

    pub(in crate::hardware_calibration) fn observed_digest(&self) -> Result<String, String> {
        let bytes = self.output.read(4096)?;
        if bytes.iter().all(|byte| *byte == 0) {
            return Err("graphics calibration produced unchanged output".to_string());
        }
        Ok(format!(
            "nerve.calibration_output_sha256.v1:{:x}",
            Sha256::digest(bytes)
        ))
    }
}

impl Drop for PreparedFixedGraphics {
    fn drop(&mut self) {
        unsafe {
            let device = self.context.device();
            let _ = device.device_wait_idle();
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_shader_module(self.fragment_shader, None);
            device.destroy_shader_module(self.vertex_shader, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_framebuffer(self.framebuffer, None);
            device.destroy_render_pass(self.render_pass, None);
        }
    }
}

pub(in crate::hardware_calibration) fn fixed_graphics_vertex_shader() -> &'static str {
    r#"#version 460
layout(location = 0) out vec4 interpolated_value;
const vec2 positions[6] = vec2[](
    vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2(-1.0,  1.0),
    vec2(-1.0,  1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0)
);
void main() {
    vec2 position = positions[gl_VertexIndex];
    float depth = mix(0.75, 0.25, float(gl_InstanceIndex & 1));
    gl_Position = vec4(position, depth, 1.0);
    interpolated_value = vec4(
        position * 0.25 + vec2(0.5),
        float((gl_VertexIndex + gl_InstanceIndex) & 7) / 7.0,
        0.25
    );
}
"#
}

pub(in crate::hardware_calibration) fn fixed_graphics_fragment_shader() -> &'static str {
    r#"#version 460
layout(location = 0) in vec4 interpolated_value;
layout(location = 0) out vec4 output_color;
void main() {
    output_color = vec4(
        interpolated_value.xyz * vec3(0.75, 0.5, 0.25) + vec3(0.03125),
        interpolated_value.w
    );
}
"#
}

fn create_render_pass(
    device: &ash::Device,
    color_format: vk::Format,
    depth_format: Option<vk::Format>,
) -> Result<vk::RenderPass, String> {
    let mut attachments = vec![
        vk::AttachmentDescription::default()
            .format(color_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
    ];
    if let Some(depth_format) = depth_format {
        attachments.push(
            vk::AttachmentDescription::default()
                .format(depth_format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
        );
    }
    let color_reference = [vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];
    let depth_reference = vk::AttachmentReference {
        attachment: 1,
        layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };
    let mut subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_reference);
    if depth_format.is_some() {
        subpass = subpass.depth_stencil_attachment(&depth_reference);
    }
    let subpasses = [subpass];
    let dependencies = [vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .dst_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        )];
    unsafe {
        device.create_render_pass(
            &vk::RenderPassCreateInfo::default()
                .attachments(&attachments)
                .subpasses(&subpasses)
                .dependencies(&dependencies),
            None,
        )
    }
    .map_err(|error| format!("could not create calibration render pass: {error:?}"))
}

#[allow(clippy::too_many_arguments)]
fn create_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vertex_shader: vk::ShaderModule,
    fragment_shader: vk::ShaderModule,
    width: u32,
    height: u32,
    blending: bool,
    depth_test: bool,
) -> Result<vk::Pipeline, String> {
    let main = CStr::from_bytes_with_nul(b"main\0").expect("static entry is nul terminated");
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_shader)
            .name(main),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_shader)
            .name(main),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewports = [vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: width as f32,
        height: height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    }];
    let scissors = [vk::Rect2D {
        offset: vk::Offset2D::default(),
        extent: vk::Extent2D { width, height },
    }];
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewports(&viewports)
        .scissors(&scissors);
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let color_attachment = [if blending {
        vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ONE)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA)
    } else {
        vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
    }];
    let color_blend =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_attachment);
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(depth_test)
        .depth_write_enable(depth_test)
        .depth_compare_op(vk::CompareOp::LESS);
    unsafe {
        device.create_graphics_pipelines(
            vk::PipelineCache::null(),
            &[vk::GraphicsPipelineCreateInfo::default()
                .stages(&stages)
                .vertex_input_state(&vertex_input)
                .input_assembly_state(&input_assembly)
                .viewport_state(&viewport)
                .rasterization_state(&rasterization)
                .multisample_state(&multisample)
                .depth_stencil_state(&depth_stencil)
                .color_blend_state(&color_blend)
                .layout(layout)
                .render_pass(render_pass)
                .subpass(0)],
            None,
        )
    }
    .map_err(|(_, error)| format!("could not create graphics calibration pipeline: {error:?}"))
    .map(|mut pipelines| pipelines.remove(0))
}

#[allow(clippy::too_many_arguments)]
fn record_graphics_commands(
    context: &SpecializedVulkanContext,
    resources: &SpecializedVulkanResources,
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    pipeline: vk::Pipeline,
    width: u32,
    height: u32,
    overdraw: u32,
    use_depth: bool,
    color_image: vk::Image,
    output: vk::Buffer,
) -> Result<(), String> {
    resources.begin()?;
    let device = context.device();
    let clear_values = if use_depth {
        vec![
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.125, 0.0625, 0.03125, 0.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ]
    } else {
        vec![vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.125, 0.0625, 0.03125, 0.0],
            },
        }]
    };
    unsafe {
        device.cmd_begin_render_pass(
            resources.command_buffer,
            &vk::RenderPassBeginInfo::default()
                .render_pass(render_pass)
                .framebuffer(framebuffer)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent: vk::Extent2D { width, height },
                })
                .clear_values(&clear_values),
            vk::SubpassContents::INLINE,
        );
        device.cmd_bind_pipeline(
            resources.command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline,
        );
        device.cmd_draw(resources.command_buffer, 6, overdraw, 0, 0);
        device.cmd_end_render_pass(resources.command_buffer);
        device.cmd_copy_image_to_buffer(
            resources.command_buffer,
            color_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            output,
            &[vk::BufferImageCopy::default()
                .buffer_offset(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: 512,
                    height: 1,
                    depth: 1,
                })],
        );
        device.cmd_pipeline_barrier2(
            resources.command_buffer,
            &vk::DependencyInfo::default().buffer_memory_barriers(&[
                vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                    .dst_access_mask(vk::AccessFlags2::HOST_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(output)
                    .offset(0)
                    .size(4096),
            ]),
        );
    }
    resources.finish_recording()
}
