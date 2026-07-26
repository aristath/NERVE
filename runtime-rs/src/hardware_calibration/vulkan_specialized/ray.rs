use super::{SpecializedBuffer, SpecializedVulkanContext, SpecializedVulkanResources};
use ash::vk;
use sha2::{Digest, Sha256};
use std::ffi::CStr;
use std::mem::size_of;
use std::rc::Rc;

pub(crate) struct PreparedRayCalibration {
    scene: PreparedRayScene,
    query: Option<PreparedRayQuery>,
}

struct PreparedRayScene {
    context: Rc<SpecializedVulkanContext>,
    resources: SpecializedVulkanResources,
    geometry: SpecializedBuffer,
    _instances: SpecializedBuffer,
    _blas_storage: SpecializedBuffer,
    _tlas_storage: SpecializedBuffer,
    _scratch: SpecializedBuffer,
    blas: vk::AccelerationStructureKHR,
    tlas: vk::AccelerationStructureKHR,
    blas_size: u64,
    tlas_size: u64,
}

struct PreparedRayQuery {
    context: Rc<SpecializedVulkanContext>,
    resources: SpecializedVulkanResources,
    output: SpecializedBuffer,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    shader_module: vk::ShaderModule,
    pipeline: vk::Pipeline,
}

impl PreparedRayCalibration {
    pub(in crate::hardware_calibration) fn new(
        context: Rc<SpecializedVulkanContext>,
        operation: &str,
        primitive_count: u32,
        ray_count: u32,
        query_spirv: Option<&[u32]>,
    ) -> Result<Self, String> {
        let scene = PreparedRayScene::new(Rc::clone(&context), primitive_count)?;
        scene.resources.run(1_000_000_000)?;
        let query = match operation {
            "build_acceleration_structure" => None,
            "ray_query_traversal" => Some(PreparedRayQuery::new(
                context,
                scene.tlas,
                ray_count,
                query_spirv.ok_or_else(|| {
                    "ray-query workload did not provide a compiled shader".to_string()
                })?,
            )?),
            other => return Err(format!("ray calibrator does not implement {other:?}")),
        };
        Ok(Self { scene, query })
    }

    pub(in crate::hardware_calibration) fn run(&self) -> Result<u64, String> {
        if let Some(query) = &self.query {
            query.resources.run(1_000_000_000)
        } else {
            self.scene.resources.run(1_000_000_000)
        }
    }

    pub(in crate::hardware_calibration) fn observed_digest(&self) -> Result<String, String> {
        if let Some(query) = &self.query {
            let bytes = query.output.read(query.output.size.min(4096) as usize)?;
            if bytes.iter().all(|byte| *byte == 0) {
                return Err("ray-query calibration produced unchanged output".to_string());
            }
            return Ok(format!(
                "nerve.calibration_output_sha256.v1:{:x}",
                Sha256::digest(bytes)
            ));
        }
        let geometry = self
            .scene
            .geometry
            .read(self.scene.geometry.size.min(4096) as usize)?;
        let mut digest = Sha256::new();
        digest.update(self.scene.blas_size.to_le_bytes());
        digest.update(self.scene.tlas_size.to_le_bytes());
        digest.update(geometry);
        Ok(format!(
            "nerve.calibration_output_sha256.v1:{:x}",
            digest.finalize()
        ))
    }
}

impl PreparedRayScene {
    fn new(context: Rc<SpecializedVulkanContext>, primitive_count: u32) -> Result<Self, String> {
        if primitive_count == 0 {
            return Err("ray scene requires at least one primitive".to_string());
        }
        let geometry_bytes = aabb_geometry(primitive_count);
        let geometry = context.create_initialized_device_buffer(
            &geometry_bytes,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )?;
        let aabb_data = vk::AccelerationStructureGeometryAabbsDataKHR::default()
            .data(vk::DeviceOrHostAddressConstKHR {
                device_address: geometry.device_address(),
            })
            .stride(size_of::<vk::AabbPositionsKHR>() as u64);
        let geometries = [vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::AABBS)
            .geometry(vk::AccelerationStructureGeometryDataKHR { aabbs: aabb_data })
            .flags(vk::GeometryFlagsKHR::OPAQUE)];
        let blas_geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(
                vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
                    | vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE,
            )
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geometries);
        let mut blas_sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            context
                .acceleration_structure()?
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &blas_geometry_info,
                    &[primitive_count],
                    &mut blas_sizes,
                );
        }
        let blas_storage = context.create_buffer(
            blas_sizes.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            false,
        )?;
        let blas = unsafe {
            context
                .acceleration_structure()?
                .create_acceleration_structure(
                    &vk::AccelerationStructureCreateInfoKHR::default()
                        .buffer(blas_storage.buffer)
                        .offset(0)
                        .size(blas_sizes.acceleration_structure_size)
                        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL),
                    None,
                )
        }
        .map_err(|error| format!("could not create calibration BLAS: {error:?}"))?;
        let blas_address = unsafe {
            context
                .acceleration_structure()?
                .get_acceleration_structure_device_address(
                    &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                        .acceleration_structure(blas),
                )
        };
        if blas_address == 0 {
            return Err("calibration BLAS has no device address".to_string());
        }
        let instance = vk::AccelerationStructureInstanceKHR {
            transform: vk::TransformMatrixKHR {
                matrix: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            },
            instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xff),
            instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                0,
                vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE
                    .as_raw()
                    .try_into()
                    .map_err(|_| "geometry-instance flags exceed 8 bits".to_string())?,
            ),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                device_handle: blas_address,
            },
        };
        let instance_bytes = unsafe {
            std::slice::from_raw_parts(
                std::ptr::from_ref(&instance).cast::<u8>(),
                size_of::<vk::AccelerationStructureInstanceKHR>(),
            )
        };
        let instances = context.create_initialized_device_buffer(
            instance_bytes,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )?;
        let instance_data = vk::AccelerationStructureGeometryInstancesDataKHR::default()
            .array_of_pointers(false)
            .data(vk::DeviceOrHostAddressConstKHR {
                device_address: instances.device_address(),
            });
        let tlas_geometries = [vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: instance_data,
            })];
        let tlas_geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&tlas_geometries);
        let mut tlas_sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            context
                .acceleration_structure()?
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &tlas_geometry_info,
                    &[1],
                    &mut tlas_sizes,
                );
        }
        let tlas_storage = context.create_buffer(
            tlas_sizes.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            false,
        )?;
        let tlas = unsafe {
            context
                .acceleration_structure()?
                .create_acceleration_structure(
                    &vk::AccelerationStructureCreateInfoKHR::default()
                        .buffer(tlas_storage.buffer)
                        .offset(0)
                        .size(tlas_sizes.acceleration_structure_size)
                        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL),
                    None,
                )
        }
        .map_err(|error| format!("could not create calibration TLAS: {error:?}"))?;
        let scratch_size = blas_sizes
            .build_scratch_size
            .max(tlas_sizes.build_scratch_size);
        let scratch = context.create_buffer(
            scratch_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            false,
        )?;
        let queue_family = context.compute_queue_family()?;
        let resources = SpecializedVulkanResources::new(Rc::clone(&context), queue_family)?;
        resources.begin()?;
        let blas_build = blas_geometry_info
            .dst_acceleration_structure(blas)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch.device_address(),
            });
        let tlas_build = tlas_geometry_info
            .dst_acceleration_structure(tlas)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch.device_address(),
            });
        let blas_range =
            [vk::AccelerationStructureBuildRangeInfoKHR::default()
                .primitive_count(primitive_count)];
        let tlas_range = [vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(1)];
        unsafe {
            context
                .acceleration_structure()?
                .cmd_build_acceleration_structures(
                    resources.command_buffer,
                    &[blas_build],
                    &[&blas_range],
                );
            context.device().cmd_pipeline_barrier2(
                resources.command_buffer,
                &vk::DependencyInfo::default().memory_barriers(&[vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
                    .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
                    .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
                    .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR)]),
            );
            context
                .acceleration_structure()?
                .cmd_build_acceleration_structures(
                    resources.command_buffer,
                    &[tlas_build],
                    &[&tlas_range],
                );
        }
        resources.finish_recording()?;
        Ok(Self {
            context,
            resources,
            geometry,
            _instances: instances,
            _blas_storage: blas_storage,
            _tlas_storage: tlas_storage,
            _scratch: scratch,
            blas,
            tlas,
            blas_size: blas_sizes.acceleration_structure_size,
            tlas_size: tlas_sizes.acceleration_structure_size,
        })
    }
}

impl Drop for PreparedRayScene {
    fn drop(&mut self) {
        unsafe {
            let _ = self.context.device().device_wait_idle();
            if let Ok(loader) = self.context.acceleration_structure() {
                loader.destroy_acceleration_structure(self.tlas, None);
                loader.destroy_acceleration_structure(self.blas, None);
            }
        }
    }
}

impl PreparedRayQuery {
    fn new(
        context: Rc<SpecializedVulkanContext>,
        tlas: vk::AccelerationStructureKHR,
        ray_count: u32,
        spirv: &[u32],
    ) -> Result<Self, String> {
        if ray_count == 0 {
            return Err("ray query requires at least one ray".to_string());
        }
        let output_size = u64::from(ray_count) * 4;
        let output = context.create_buffer(
            output_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL
                | vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
            true,
        )?;
        output.write(&vec![0; output_size as usize])?;
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let descriptor_set_layout = unsafe {
            context.device().create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }
        .map_err(|error| format!("could not create ray-query descriptor layout: {error:?}"))?;
        let set_layouts = [descriptor_set_layout];
        let push_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(4)];
        let pipeline_layout = unsafe {
            context.device().create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&push_ranges),
                None,
            )
        }
        .map_err(|error| format!("could not create ray-query pipeline layout: {error:?}"))?;
        let shader_module = unsafe {
            context
                .device()
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spirv), None)
        }
        .map_err(|error| format!("could not create ray-query shader: {error:?}"))?;
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
        .map_err(|(_, error)| format!("could not create ray-query pipeline: {error:?}"))?
        .remove(0);
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                descriptor_count: 1,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
            },
        ];
        let descriptor_pool = unsafe {
            context.device().create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&pool_sizes),
                None,
            )
        }
        .map_err(|error| format!("could not create ray-query descriptor pool: {error:?}"))?;
        let descriptor_set = unsafe {
            context.device().allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&set_layouts),
            )
        }
        .map_err(|error| format!("could not allocate ray-query descriptor set: {error:?}"))?
        .remove(0);
        let structures = [tlas];
        let mut acceleration_write = vk::WriteDescriptorSetAccelerationStructureKHR::default()
            .acceleration_structures(&structures);
        let output_info = [vk::DescriptorBufferInfo::default()
            .buffer(output.buffer)
            .offset(0)
            .range(output_size)];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .push_next(&mut acceleration_write),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&output_info),
        ];
        unsafe { context.device().update_descriptor_sets(&writes, &[]) };
        let resources =
            SpecializedVulkanResources::new(Rc::clone(&context), context.compute_queue_family()?)?;
        resources.begin()?;
        unsafe {
            context.device().cmd_pipeline_barrier2(
                resources.command_buffer,
                &vk::DependencyInfo::default().memory_barriers(&[vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
                    .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(
                        vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR
                            | vk::AccessFlags2::SHADER_WRITE,
                    )]),
            );
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
            context.device().cmd_push_constants(
                resources.command_buffer,
                pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &ray_count.to_le_bytes(),
            );
            context
                .device()
                .cmd_dispatch(resources.command_buffer, ray_count.div_ceil(256), 1, 1);
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
                        .size(output_size),
                ]),
            );
        }
        resources.finish_recording()?;
        Ok(Self {
            context,
            resources,
            output,
            descriptor_pool,
            descriptor_set_layout,
            pipeline_layout,
            shader_module,
            pipeline,
        })
    }
}

impl Drop for PreparedRayQuery {
    fn drop(&mut self) {
        unsafe {
            let device = self.context.device();
            let _ = device.device_wait_idle();
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_shader_module(self.shader_module, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

pub(in crate::hardware_calibration) fn ray_query_shader() -> &'static str {
    r#"#version 460
#extension GL_EXT_ray_query : require
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) uniform accelerationStructureEXT scene;
layout(set = 0, binding = 1) writeonly buffer OutputWords { uint words[]; } output_words;
layout(push_constant) uniform Control { uint ray_count; } control;
void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= control.ray_count) { return; }
    uint grid = 1024u;
    vec2 xy = vec2(index % grid, (index / grid) % grid) / float(grid);
    vec3 origin = vec3(xy * 32.0, -2.0);
    rayQueryEXT query;
    rayQueryInitializeEXT(
        query,
        scene,
        gl_RayFlagsOpaqueEXT,
        0xffu,
        origin,
        0.0,
        vec3(0.0, 0.0, 1.0),
        8.0
    );
    uint candidates = 0u;
    while (rayQueryProceedEXT(query)) {
        candidates++;
        if (rayQueryGetIntersectionTypeEXT(query, false) == gl_RayQueryCandidateIntersectionAABBEXT) {
            rayQueryGenerateIntersectionEXT(query, 2.0);
        }
    }
    float distance = rayQueryGetIntersectionTEXT(query, true);
    output_words.words[index] = floatBitsToUint(distance) ^ candidates ^ (index * 0x9e3779b9u);
}
"#
}

fn aabb_geometry(primitive_count: u32) -> Vec<u8> {
    let mut geometry =
        Vec::with_capacity(primitive_count as usize * size_of::<vk::AabbPositionsKHR>());
    for index in 0..primitive_count {
        let x = (index % 1024) as f32 / 32.0;
        let y = ((index / 1024) % 1024) as f32 / 32.0;
        let z = ((index / (1024 * 1024)) % 8) as f32 * 0.25;
        let aabb = vk::AabbPositionsKHR {
            min_x: x,
            min_y: y,
            min_z: z,
            max_x: x + 0.025,
            max_y: y + 0.025,
            max_z: z + 0.125,
        };
        let bytes = unsafe {
            std::slice::from_raw_parts(
                std::ptr::from_ref(&aabb).cast::<u8>(),
                size_of::<vk::AabbPositionsKHR>(),
            )
        };
        geometry.extend_from_slice(bytes);
    }
    geometry
}
