use crate::hardware_profile::{
    HardwareDeviceKind, HardwareIdentity, HardwareInterconnect, HardwareMemoryDomain,
    HardwareProcessAvailability, HardwareProcessCapability, HardwareProcessCategory,
    HardwareProcessProfile, HardwareProcessProfileDefinition,
    HardwareProcessProgrammability, HardwareProfileProvenance,
};
use serde_json::json;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanHardwareFacts {
    pub api_version: u32,
    pub driver_version: u32,
    pub driver_name: String,
    pub driver_info: String,
    pub driver_id: String,
    pub queue_family_count: u64,
    pub compute_queue_count: u64,
    pub graphics_queue_count: u64,
    pub transfer_queue_count: u64,
    pub transfer_only_queue_count: u64,
    pub video_decode_queue_count: u64,
    pub video_encode_queue_count: u64,
    pub extension_names: BTreeSet<String>,
    pub sampled_formats: BTreeSet<String>,
    pub storage_image_formats: BTreeSet<String>,
    pub linear_filter_formats: BTreeSet<String>,
    pub cooperative_matrix_variants: BTreeSet<String>,
    pub max_compute_work_group_count_x: u32,
    pub max_compute_shared_memory_size: u32,
    pub max_storage_buffer_range: u64,
    pub max_uniform_buffer_range: u64,
    pub min_storage_buffer_offset_alignment: u64,
    pub min_uniform_buffer_offset_alignment: u64,
    pub max_image_dimension_1d: u32,
    pub max_image_dimension_2d: u32,
    pub max_image_dimension_3d: u32,
    pub max_bound_descriptor_sets: u32,
    pub timestamp_compute_and_graphics: bool,
    pub timestamp_period_bits: u32,
    pub shared_host_memory_alignment: Option<u64>,
    pub shared_device_memory_supported: bool,
    pub external_timeline_semaphore_supported: bool,
    pub timeline_semaphore_supported: bool,
    pub synchronization2_supported: bool,
    pub memory_types: Vec<VulkanMemoryTypeFacts>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VulkanMemoryTypeFacts {
    pub type_index: u32,
    pub heap_index: u32,
    pub host_visible: bool,
    pub device_local: bool,
    pub coherent: bool,
    pub cached: bool,
    pub lazily_allocated: bool,
}

impl VulkanComputeDeviceCatalog {
    pub fn available_hardware_profiles(
        &self,
    ) -> Result<Vec<HardwareProcessProfile>, VulkanError> {
        let targets = self.available_target_capabilities()?;
        self.available_devices
            .iter()
            .zip(targets)
            .map(|(device, target)| {
                let physical_device = self.physical_devices[device.physical_device_index];
                let facts = inspect_vulkan_hardware_facts(
                    &self.context._entry,
                    &self.context.instance,
                    physical_device,
                )?;
                build_vulkan_hardware_profile(device, target, facts)
                    .map_err(VulkanError)
            })
            .collect()
    }
}

fn inspect_vulkan_hardware_facts(
    entry: &Entry,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<VulkanHardwareFacts, VulkanError> {
    let properties = unsafe { instance.get_physical_device_properties(physical_device) };
    let mut driver_properties = vk::PhysicalDeviceDriverProperties::default();
    let mut properties2 =
        vk::PhysicalDeviceProperties2::default().push_next(&mut driver_properties);
    unsafe {
        instance.get_physical_device_properties2(physical_device, &mut properties2);
    }
    let driver_name = unsafe {
        CStr::from_ptr(driver_properties.driver_name.as_ptr())
            .to_string_lossy()
            .into_owned()
    };
    let driver_info = unsafe {
        CStr::from_ptr(driver_properties.driver_info.as_ptr())
            .to_string_lossy()
            .into_owned()
    };
    let queue_families =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let extension_names = unsafe {
        instance
            .enumerate_device_extension_properties(physical_device)
            .map_err(|error| {
                VulkanError(format!(
                    "failed to enumerate Vulkan device extensions: {error:?}"
                ))
            })?
    }
    .into_iter()
    .map(|extension| unsafe {
        CStr::from_ptr(extension.extension_name.as_ptr())
            .to_string_lossy()
            .into_owned()
    })
    .collect::<BTreeSet<_>>();
    let format_support = inspect_vulkan_format_support(instance, physical_device);
    let cooperative_matrix_variants =
        if extension_names.contains("VK_KHR_cooperative_matrix") {
            inspect_cooperative_matrix_variants(entry, instance, physical_device)?
        } else {
            BTreeSet::new()
        };
    let shared_host_memory_alignment = if extension_names.contains("VK_EXT_external_memory_host")
        && physical_device_supports_shared_host_buffer(instance, physical_device)
    {
        Some(
            u64::try_from(physical_device_shared_host_memory_alignment(
                instance,
                physical_device,
            )?)
            .map_err(|_| VulkanError("shared-host alignment exceeds u64".to_string()))?,
        )
    } else {
        None
    };
    let shared_device_memory_supported =
        extension_names.contains("VK_KHR_external_memory_fd")
            && extension_names.contains("VK_EXT_external_memory_dma_buf")
            && physical_device_supports_shared_device_buffer(instance, physical_device);
    let external_timeline_semaphore_supported =
        extension_names.contains("VK_KHR_external_semaphore_fd")
            && physical_device_supports_opaque_fd_timeline_semaphore(
                instance,
                physical_device,
            );
    let (timeline_semaphore_supported, synchronization2_supported) =
        physical_device_supports_modern_submission(instance, physical_device);
    let memory_properties =
        unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let memory_types = (0..memory_properties.memory_type_count)
        .map(|type_index| {
        let memory_type = memory_properties.memory_types[type_index as usize];
            VulkanMemoryTypeFacts {
                type_index,
                heap_index: memory_type.heap_index,
                host_visible: memory_type
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_VISIBLE),
                device_local: memory_type
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL),
                coherent: memory_type
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_COHERENT),
                cached: memory_type
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_CACHED),
                lazily_allocated: memory_type
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::LAZILY_ALLOCATED),
            }
        })
        .collect();

    Ok(VulkanHardwareFacts {
        api_version: properties.api_version,
        driver_version: properties.driver_version,
        driver_name,
        driver_info,
        driver_id: format!("{:?}", driver_properties.driver_id),
        queue_family_count: queue_families.len() as u64,
        compute_queue_count: queue_count(&queue_families, vk::QueueFlags::COMPUTE),
        graphics_queue_count: queue_count(&queue_families, vk::QueueFlags::GRAPHICS),
        transfer_queue_count: queue_count(&queue_families, vk::QueueFlags::TRANSFER),
        transfer_only_queue_count: queue_families
            .iter()
            .filter(|family| {
                family.queue_flags.contains(vk::QueueFlags::TRANSFER)
                    && !family
                        .queue_flags
                        .intersects(vk::QueueFlags::COMPUTE | vk::QueueFlags::GRAPHICS)
            })
            .map(|family| u64::from(family.queue_count))
            .sum(),
        video_decode_queue_count: queue_count(
            &queue_families,
            vk::QueueFlags::VIDEO_DECODE_KHR,
        ),
        video_encode_queue_count: queue_count(
            &queue_families,
            vk::QueueFlags::VIDEO_ENCODE_KHR,
        ),
        extension_names,
        sampled_formats: format_support.sampled,
        storage_image_formats: format_support.storage,
        linear_filter_formats: format_support.linear_filter,
        cooperative_matrix_variants,
        max_compute_work_group_count_x: properties.limits.max_compute_work_group_count[0],
        max_compute_shared_memory_size: properties.limits.max_compute_shared_memory_size,
        max_storage_buffer_range: u64::from(properties.limits.max_storage_buffer_range),
        max_uniform_buffer_range: u64::from(properties.limits.max_uniform_buffer_range),
        min_storage_buffer_offset_alignment: properties
            .limits
            .min_storage_buffer_offset_alignment,
        min_uniform_buffer_offset_alignment: properties
            .limits
            .min_uniform_buffer_offset_alignment,
        max_image_dimension_1d: properties.limits.max_image_dimension1_d,
        max_image_dimension_2d: properties.limits.max_image_dimension2_d,
        max_image_dimension_3d: properties.limits.max_image_dimension3_d,
        max_bound_descriptor_sets: properties.limits.max_bound_descriptor_sets,
        timestamp_compute_and_graphics: properties.limits.timestamp_compute_and_graphics
            == vk::TRUE,
        timestamp_period_bits: properties.limits.timestamp_period.to_bits(),
        shared_host_memory_alignment,
        shared_device_memory_supported,
        external_timeline_semaphore_supported,
        timeline_semaphore_supported,
        synchronization2_supported,
        memory_types,
    })
}

fn queue_count(families: &[vk::QueueFamilyProperties], required: vk::QueueFlags) -> u64 {
    families
        .iter()
        .filter(|family| family.queue_flags.contains(required))
        .map(|family| u64::from(family.queue_count))
        .sum()
}

#[derive(Default)]
struct VulkanFormatSupport {
    sampled: BTreeSet<String>,
    storage: BTreeSet<String>,
    linear_filter: BTreeSet<String>,
}

fn inspect_vulkan_format_support(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> VulkanFormatSupport {
    let formats = [
        ("r8_sint", vk::Format::R8_SINT),
        ("r8_uint", vk::Format::R8_UINT),
        ("r8_unorm", vk::Format::R8_UNORM),
        ("r8g8b8a8_unorm", vk::Format::R8G8B8A8_UNORM),
        ("r16_sfloat", vk::Format::R16_SFLOAT),
        ("r16g16_sfloat", vk::Format::R16G16_SFLOAT),
        ("r32_sfloat", vk::Format::R32_SFLOAT),
        ("r32_uint", vk::Format::R32_UINT),
        ("r32g32_sfloat", vk::Format::R32G32_SFLOAT),
        ("r32g32b32a32_sfloat", vk::Format::R32G32B32A32_SFLOAT),
    ];
    let mut support = VulkanFormatSupport::default();
    for (name, format) in formats {
        let properties =
            unsafe { instance.get_physical_device_format_properties(physical_device, format) };
        if properties
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE)
        {
            support.sampled.insert(name.to_string());
        }
        if properties
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::STORAGE_IMAGE)
        {
            support.storage.insert(name.to_string());
        }
        if properties
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
        {
            support.linear_filter.insert(name.to_string());
        }
    }
    support
}

fn inspect_cooperative_matrix_variants(
    entry: &Entry,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<BTreeSet<String>, VulkanError> {
    let cooperative_matrix =
        ash::khr::cooperative_matrix::Instance::new(entry, instance);
    let properties = unsafe {
        cooperative_matrix
            .get_physical_device_cooperative_matrix_properties(physical_device)
            .map_err(|error| {
                VulkanError(format!(
                    "failed to query cooperative-matrix properties: {error:?}"
                ))
            })?
    };
    Ok(properties
        .into_iter()
        .map(|property| {
            format!(
                "a={:?};b={:?};c={:?};result={:?};scope={:?};m={};n={};k={};saturating={}",
                property.a_type,
                property.b_type,
                property.c_type,
                property.result_type,
                property.scope,
                property.m_size,
                property.n_size,
                property.k_size,
                property.saturating_accumulation == vk::TRUE,
            )
        })
        .collect())
}

pub fn build_vulkan_hardware_profile(
    device: &VulkanComputeDeviceInfo,
    target: VulkanComputeTargetCapabilities,
    facts: VulkanHardwareFacts,
) -> Result<HardwareProcessProfile, String> {
    if device.physical_device_index != target.physical_device_index
        || device.physical_device_id != target.physical_device_id
    {
        return Err("Vulkan hardware facts and compiler target identify different devices"
            .to_string());
    }
    if facts.compute_queue_count == 0 {
        return Err("Vulkan compiler target contains no compute queue".to_string());
    }
    let identity = HardwareIdentity {
        device_kind: HardwareDeviceKind::Gpu,
        stable_device_id: device.physical_device_id.clone(),
        name: device.device_name.clone(),
        vendor_id: format!("0x{:04x}", device.vendor_id),
        device_id: format!("0x{:04x}", device.device_id),
        architecture: format!(
            "vulkan_vendor_{:04x}_device_{:04x}",
            device.vendor_id, device.device_id
        ),
        physical_location: device.physical_device_id.clone(),
    };
    let processes = vulkan_processes(&target, &facts);
    let memory_domains = vulkan_memory_domains(device, &facts)?;
    let interconnects = vulkan_interconnects(&facts);
    let provenance = HardwareProfileProvenance {
        api: "vulkan".to_string(),
        api_version: vulkan_version(facts.api_version),
        driver: facts.driver_name.clone(),
        driver_version: facts.driver_version.to_string(),
        compiler: env!("NERVE_HARDWARE_DISCOVERY_FINGERPRINT").to_string(),
        operating_system: std::env::consts::OS.to_string(),
        discovery_backend: "vulkan_physical_device_queries".to_string(),
    };
    let mut compiler_capabilities = serde_json::to_value(&target)
        .map_err(|error| format!("could not serialize Vulkan target: {error}"))?;
    let compiler_capability_object = compiler_capabilities
        .as_object_mut()
        .ok_or_else(|| "serialized Vulkan compiler target is not an object".to_string())?;
    for identity_field in [
        "physical_device_index",
        "physical_device_id",
        "device_name",
        "device_type",
        "vendor_id",
        "device_id",
    ] {
        compiler_capability_object.remove(identity_field);
    }
    HardwareProcessProfile::create(HardwareProcessProfileDefinition {
        hardware_identity: identity,
        processes,
        memory_domains,
        interconnects,
        provenance,
        capability_extensions: BTreeMap::from([
            (
                "vulkan_compiler_capabilities".to_string(),
                compiler_capabilities,
            ),
            (
                "vulkan_device".to_string(),
                json!({
                    "api_version": vulkan_version(facts.api_version),
                    "driver_version": facts.driver_version,
                    "driver_id": facts.driver_id,
                    "driver_info": facts.driver_info,
                    "extensions": facts.extension_names,
                    "format_support": {
                        "sampled": facts.sampled_formats,
                        "storage_image": facts.storage_image_formats,
                        "linear_filter": facts.linear_filter_formats,
                    },
                    "queue_family_count": facts.queue_family_count,
                }),
            ),
        ]),
        identity_extensions: BTreeMap::new(),
        runtime_bindings: BTreeMap::from([(
            "vulkan_runtime_binding".to_string(),
            json!({
                "physical_device_index": device.physical_device_index,
                "physical_device_id": device.physical_device_id,
                "device_name": device.device_name,
                "device_type": device.device_type,
                "vendor_id": device.vendor_id,
                "device_id": device.device_id,
            }),
        )]),
    })
}

fn vulkan_processes(
    target: &VulkanComputeTargetCapabilities,
    facts: &VulkanHardwareFacts,
) -> Vec<HardwareProcessCapability> {
    vec![
        vulkan_scalar_process(target),
        vulkan_vector_process(target),
        vulkan_packed_dot_process(target),
        vulkan_matrix_process(target, facts),
        vulkan_subgroup_process(target),
        vulkan_register_process(),
        vulkan_shared_memory_process(target, facts),
        vulkan_occupancy_process(target, facts),
        vulkan_cache_process(),
        vulkan_memory_bandwidth_process(),
        vulkan_texture_process(facts),
        conditional_process(
            "rasterization",
            HardwareProcessCategory::Graphics,
            facts.graphics_queue_count > 0,
            &["clipping", "fragment_generation", "triangle_rasterization"],
            &[],
            &[],
        ),
        conditional_process(
            "fixed_function_interpolation",
            HardwareProcessCategory::Graphics,
            facts.graphics_queue_count > 0,
            &["barycentric_interpolation", "perspective_interpolation"],
            &["f16", "f32"],
            &[],
        ),
        conditional_process(
            "depth_stencil",
            HardwareProcessCategory::Graphics,
            facts.graphics_queue_count > 0,
            &["compare", "depth_test", "stencil_test"],
            &["d16", "d24", "d32"],
            &[],
        ),
        conditional_process(
            "blending",
            HardwareProcessCategory::Graphics,
            facts.graphics_queue_count > 0,
            &["add", "logic_op", "min_max", "multiply"],
            &["f16", "f32", "unorm8"],
            &[],
        ),
        conditional_process(
            "acceleration_structure_construction",
            HardwareProcessCategory::RayTraversal,
            facts
                .extension_names
                .contains("VK_KHR_acceleration_structure"),
            &["build", "compact", "refit", "serialize"],
            &[],
            &["VK_KHR_acceleration_structure"],
        ),
        conditional_process(
            "ray_traversal",
            HardwareProcessCategory::RayTraversal,
            facts.extension_names.contains("VK_KHR_ray_query")
                || facts
                    .extension_names
                    .contains("VK_KHR_ray_tracing_pipeline"),
            &["intersection", "ray_query", "traversal"],
            &["f32"],
            &["VK_KHR_ray_query"],
        ),
        vulkan_atomic_process(target),
        vulkan_collective_algorithm_process(target),
        conditional_process(
            "indirect_work_generation",
            HardwareProcessCategory::Scheduling,
            true,
            &["dispatch_indirect", "draw_indirect"],
            &["u32"],
            &[],
        ),
        conditional_process(
            "device_generated_commands",
            HardwareProcessCategory::Scheduling,
            facts
                .extension_names
                .contains("VK_EXT_device_generated_commands"),
            &["command_preprocess", "generated_dispatch", "generated_draw"],
            &["u32"],
            &["VK_EXT_device_generated_commands"],
        ),
        conditional_process(
            "execution_graphs",
            HardwareProcessCategory::Scheduling,
            facts.extension_names.contains("VK_AMDX_shader_enqueue")
                || facts.extension_names.contains("VK_ARM_data_graph"),
            &["device_side_graph_dispatch", "execution_graph"],
            &[],
            &[],
        ),
        vulkan_queue_process(facts),
        conditional_process(
            "resident_command_replay",
            HardwareProcessCategory::Scheduling,
            true,
            &["command_buffer_replay", "persistent_resource_binding"],
            &[],
            &[],
        ),
        vulkan_copy_process(facts),
        vulkan_sync_process(facts),
        conditional_process(
            "video_decode",
            HardwareProcessCategory::Media,
            facts.video_decode_queue_count > 0,
            &["decode_bitstream", "reconstruct_frame"],
            &["video_profile_dependent"],
            &["VK_KHR_video_queue", "VK_KHR_video_decode_queue"],
        ),
        conditional_process(
            "video_encode",
            HardwareProcessCategory::Media,
            facts.video_encode_queue_count > 0,
            &["encode_bitstream", "motion_estimation"],
            &["video_profile_dependent"],
            &["VK_KHR_video_queue", "VK_KHR_video_encode_queue"],
        ),
    ]
}

fn vulkan_scalar_process(
    target: &VulkanComputeTargetCapabilities,
) -> HardwareProcessCapability {
    let mut process = available_vulkan_process(
        "shader_scalar",
        HardwareProcessCategory::Arithmetic,
        &["add", "bitwise", "compare", "divide", "fused_multiply_add", "multiply"],
    );
    process.numeric_formats = shader_numeric_formats(target);
    process
}

fn vulkan_vector_process(
    target: &VulkanComputeTargetCapabilities,
) -> HardwareProcessCapability {
    let mut process = available_vulkan_process(
        "shader_vector",
        HardwareProcessCategory::Arithmetic,
        &["componentwise_arithmetic", "dot", "vector_load", "vector_store"],
    );
    process.numeric_formats = shader_numeric_formats(target);
    process
}

fn vulkan_packed_dot_process(
    target: &VulkanComputeTargetCapabilities,
) -> HardwareProcessCapability {
    let available = target
        .shader_features
        .contains(&VulkanShaderFeature::ShaderIntegerDotProduct)
        || target
            .shader_features
            .contains(&VulkanShaderFeature::ShaderBfloat16DotProduct)
        || target.shader_features.contains(
            &VulkanShaderFeature::ShaderMixedFloatDotProductFloat16AccFloat32,
        )
        || target.shader_features.contains(
            &VulkanShaderFeature::ShaderMixedFloatDotProductFloat16AccFloat16,
        )
        || target
            .shader_features
            .contains(&VulkanShaderFeature::ShaderMixedFloatDotProductFloat8AccFloat32);
    let mut process = conditional_process(
        "packed_dot_product",
        HardwareProcessCategory::Arithmetic,
        available,
        &["dot", "mixed_dot_accumulate", "packed_dot_accumulate"],
        &[],
        &[],
    );
    process.numeric_formats = shader_numeric_formats(target);
    process.required_features = target
        .shader_features
        .iter()
        .filter(|feature| {
            matches!(
                feature,
                VulkanShaderFeature::ShaderIntegerDotProduct
                    | VulkanShaderFeature::ShaderMixedFloatDotProductFloat16AccFloat32
                    | VulkanShaderFeature::ShaderMixedFloatDotProductFloat16AccFloat16
                    | VulkanShaderFeature::ShaderBfloat16DotProduct
                    | VulkanShaderFeature::ShaderMixedFloatDotProductBfloat16Acc
                    | VulkanShaderFeature::ShaderMixedFloatDotProductFloat8AccFloat32
            )
        })
        .map(|feature| feature.label().to_string())
        .collect();
    process
}

fn vulkan_matrix_process(
    target: &VulkanComputeTargetCapabilities,
    facts: &VulkanHardwareFacts,
) -> HardwareProcessCapability {
    let available = target
        .shader_features
        .contains(&VulkanShaderFeature::CooperativeMatrix);
    let mut process = conditional_process(
        "cooperative_matrix",
        HardwareProcessCategory::Arithmetic,
        available,
        &["matrix_multiply_accumulate"],
        &[],
        &["VK_KHR_cooperative_matrix"],
    );
    let mut matrix_formats = BTreeSet::new();
    if !target.cooperative_float16_shapes.is_empty() {
        matrix_formats.insert("f16".to_string());
    }
    if !target.cooperative_bfloat16_shapes.is_empty() {
        matrix_formats.insert("bf16".to_string());
    }
    if !target.cooperative_float8_e4m3_shapes.is_empty() {
        matrix_formats.insert("f8_e4m3".to_string());
    }
    process.numeric_formats = matrix_formats.into_iter().collect();
    process.required_features = target
        .shader_features
        .iter()
        .filter(|feature| {
            matches!(
                feature,
                VulkanShaderFeature::CooperativeMatrix
                    | VulkanShaderFeature::ShaderBfloat16CooperativeMatrix
                    | VulkanShaderFeature::ShaderFloat8CooperativeMatrix
            )
        })
        .map(|feature| feature.label().to_string())
        .collect();
    process.properties.insert(
        "float16_shapes".to_string(),
        matrix_shapes(&target.cooperative_float16_shapes),
    );
    process.properties.insert(
        "bfloat16_shapes".to_string(),
        matrix_shapes(&target.cooperative_bfloat16_shapes),
    );
    process.properties.insert(
        "float8_e4m3_shapes".to_string(),
        matrix_shapes(&target.cooperative_float8_e4m3_shapes),
    );
    process.properties.insert(
        "all_reported_variants".to_string(),
        facts.cooperative_matrix_variants
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .join("|"),
    );
    process
}

fn vulkan_subgroup_process(
    target: &VulkanComputeTargetCapabilities,
) -> HardwareProcessCapability {
    let available = target.subgroup_compute_supported && target.subgroup_size > 0;
    let mut process = conditional_process(
        "subgroup_collectives",
        HardwareProcessCategory::ControlFlow,
        available,
        &[],
        &[],
        &[],
    );
    process.operations = target
        .subgroup_operations
        .iter()
        .map(|operation| operation.label().to_string())
        .collect();
    process
        .limits
        .insert("subgroup_size".to_string(), u64::from(target.subgroup_size));
    process
}

fn vulkan_register_process() -> HardwareProcessCapability {
    let mut process = available_vulkan_process(
        "register_file",
        HardwareProcessCategory::Memory,
        &["compiler_allocated_thread_local_storage", "scalar_register", "vector_register"],
    );
    process.programmability = HardwareProcessProgrammability::Indirect;
    process.properties.insert(
        "capacity_visibility".to_string(),
        "opaque_to_vulkan".to_string(),
    );
    process
}

fn vulkan_shared_memory_process(
    target: &VulkanComputeTargetCapabilities,
    facts: &VulkanHardwareFacts,
) -> HardwareProcessCapability {
    let mut process = available_vulkan_process(
        "workgroup_shared_memory",
        HardwareProcessCategory::Memory,
        &["barrier_coordinated_access", "shared_load", "shared_store"],
    );
    process.limits.insert(
        "capacity_bytes".to_string(),
        u64::from(facts.max_compute_shared_memory_size),
    );
    process.limits.insert(
        "max_workgroup_invocations".to_string(),
        u64::from(target.max_compute_work_group_invocations),
    );
    process
}

fn vulkan_cache_process() -> HardwareProcessCapability {
    let mut process = available_vulkan_process(
        "device_cache_hierarchy",
        HardwareProcessCategory::Memory,
        &["cached_buffer_access", "cached_image_access"],
    );
    process.programmability = HardwareProcessProgrammability::Indirect;
    process
        .properties
        .insert("topology_visibility".to_string(), "opaque_to_vulkan".to_string());
    process
}

fn vulkan_memory_bandwidth_process() -> HardwareProcessCapability {
    let mut process = available_vulkan_process(
        "device_memory_bandwidth",
        HardwareProcessCategory::Memory,
        &[
            "coalesced_read",
            "coalesced_write",
            "streaming_copy",
        ],
    );
    process.programmability = HardwareProcessProgrammability::Indirect;
    process.properties.insert(
        "realized_bandwidth".to_string(),
        "requires_hardware_calibration".to_string(),
    );
    process
}

fn vulkan_occupancy_process(
    target: &VulkanComputeTargetCapabilities,
    facts: &VulkanHardwareFacts,
) -> HardwareProcessCapability {
    let mut process = HardwareProcessCapability::new(
        "occupancy_constraints",
        HardwareProcessCategory::Scheduling,
        HardwareProcessAvailability::Opaque,
        HardwareProcessProgrammability::Indirect,
        "vulkan",
    );
    process.limits.insert(
        "max_compute_shared_memory_size".to_string(),
        u64::from(facts.max_compute_shared_memory_size),
    );
    process.limits.insert(
        "max_compute_work_group_invocations".to_string(),
        u64::from(target.max_compute_work_group_invocations),
    );
    process.limits.insert(
        "max_compute_work_group_size_x".to_string(),
        u64::from(target.max_compute_work_group_size_x),
    );
    process.properties.insert(
        "register_and_wave_occupancy_model".to_string(),
        "not_exposed_by_core_vulkan".to_string(),
    );
    process
}

fn vulkan_texture_process(facts: &VulkanHardwareFacts) -> HardwareProcessCapability {
    let available = !facts.sampled_formats.is_empty();
    let mut process = conditional_process(
        "texture_sampling",
        HardwareProcessCategory::Sampling,
        available,
        &[
            "addressing",
            "format_conversion",
            "gather",
            "linear_interpolation",
            "nearest_sampling",
        ],
        &[],
        &[],
    );
    process.numeric_formats = facts.sampled_formats.iter().cloned().collect();
    process.properties.insert(
        "linear_filter_formats".to_string(),
        facts
            .linear_filter_formats
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .join(","),
    );
    process.limits.insert(
        "max_image_dimension_1d".to_string(),
        u64::from(facts.max_image_dimension_1d),
    );
    process.limits.insert(
        "max_image_dimension_2d".to_string(),
        u64::from(facts.max_image_dimension_2d),
    );
    process.limits.insert(
        "max_image_dimension_3d".to_string(),
        u64::from(facts.max_image_dimension_3d),
    );
    process
}

fn vulkan_atomic_process(
    target: &VulkanComputeTargetCapabilities,
) -> HardwareProcessCapability {
    let mut process = available_vulkan_process(
        "shader_atomics",
        HardwareProcessCategory::Synchronization,
        &["add", "compare_exchange", "exchange", "min_max"],
    );
    process.numeric_formats = vec!["i32".to_string(), "u32".to_string()];
    if target
        .shader_features
        .contains(&VulkanShaderFeature::ShaderInt64)
    {
        process.numeric_formats.extend(["i64".to_string(), "u64".to_string()]);
    }
    process
}

fn vulkan_collective_algorithm_process(
    target: &VulkanComputeTargetCapabilities,
) -> HardwareProcessCapability {
    let available = target.subgroup_compute_supported
        && target
            .subgroup_operations
            .contains(&VulkanSubgroupOperation::Arithmetic);
    conditional_process(
        "parallel_collective_algorithms",
        HardwareProcessCategory::Arithmetic,
        available,
        &["compaction", "prefix_scan", "reduction"],
        &["f32", "i32", "u32"],
        &[],
    )
}

fn vulkan_queue_process(facts: &VulkanHardwareFacts) -> HardwareProcessCapability {
    let mut process = available_vulkan_process(
        "command_queues",
        HardwareProcessCategory::Scheduling,
        &["command_buffer", "compute_dispatch", "queue_submit"],
    );
    process
        .limits
        .insert("compute_queue_count".to_string(), facts.compute_queue_count);
    process
        .limits
        .insert("graphics_queue_count".to_string(), facts.graphics_queue_count);
    process
        .limits
        .insert("queue_family_count".to_string(), facts.queue_family_count);
    process.limits.insert(
        "max_compute_work_group_count_x".to_string(),
        u64::from(facts.max_compute_work_group_count_x),
    );
    process.limits.insert(
        "max_bound_descriptor_sets".to_string(),
        u64::from(facts.max_bound_descriptor_sets),
    );
    process
}

fn vulkan_copy_process(facts: &VulkanHardwareFacts) -> HardwareProcessCapability {
    let mut process = conditional_process(
        "copy_engines",
        HardwareProcessCategory::Transfer,
        facts.transfer_queue_count > 0,
        &["buffer_copy", "buffer_fill", "buffer_image_copy", "image_copy"],
        &[],
        &[],
    );
    process
        .limits
        .insert("transfer_queue_count".to_string(), facts.transfer_queue_count);
    process.limits.insert(
        "transfer_only_queue_count".to_string(),
        facts.transfer_only_queue_count,
    );
    process
}

fn vulkan_sync_process(facts: &VulkanHardwareFacts) -> HardwareProcessCapability {
    let available = facts.timeline_semaphore_supported && facts.synchronization2_supported;
    let mut process = conditional_process(
        "synchronization",
        HardwareProcessCategory::Synchronization,
        available,
        &["barrier", "fence", "timeline_semaphore"],
        &[],
        &[],
    );
    process.required_features = vec![
        "synchronization2".to_string(),
        "timeline_semaphore".to_string(),
    ];
    process.properties.insert(
        "timestamp_compute_and_graphics".to_string(),
        facts.timestamp_compute_and_graphics.to_string(),
    );
    process.properties.insert(
        "timestamp_period_f32_bits".to_string(),
        facts.timestamp_period_bits.to_string(),
    );
    process
}

fn vulkan_memory_domains(
    device: &VulkanComputeDeviceInfo,
    facts: &VulkanHardwareFacts,
) -> Result<Vec<HardwareMemoryDomain>, String> {
    let alignment = facts
        .min_storage_buffer_offset_alignment
        .max(1)
        .next_power_of_two();
    facts
        .memory_types
        .iter()
        .map(|memory_type| {
            let heap = device
                .memory_heaps
                .iter()
                .find(|heap| heap.heap_index == memory_type.heap_index)
                .ok_or_else(|| {
                    format!(
                        "Vulkan memory type {} references missing heap {}",
                        memory_type.type_index, memory_type.heap_index
                    )
                })?;
            Ok(HardwareMemoryDomain {
                name: format!("vulkan_memory_type_{:03}", memory_type.type_index),
                kind: if memory_type.device_local {
                    "device_local_memory_type".to_string()
                } else {
                    "host_backed_memory_type".to_string()
                },
                capacity_bytes: heap.size_bytes,
                host_visible: memory_type.host_visible,
                device_local: memory_type.device_local,
                coherent: memory_type.coherent,
                cached: memory_type.cached,
                minimum_alignment_bytes: alignment,
                properties: BTreeMap::from([
                    (
                        "capacity_scope".to_string(),
                        "shared_vulkan_heap".to_string(),
                    ),
                    ("heap_index".to_string(), heap.heap_index.to_string()),
                    (
                        "max_storage_buffer_range".to_string(),
                        facts.max_storage_buffer_range.to_string(),
                    ),
                    (
                        "max_uniform_buffer_range".to_string(),
                        facts.max_uniform_buffer_range.to_string(),
                    ),
                    (
                        "min_uniform_buffer_offset_alignment".to_string(),
                        facts.min_uniform_buffer_offset_alignment.to_string(),
                    ),
                    (
                        "lazily_allocated".to_string(),
                        memory_type.lazily_allocated.to_string(),
                    ),
                ]),
            })
        })
        .collect()
}

fn vulkan_interconnects(facts: &VulkanHardwareFacts) -> Vec<HardwareInterconnect> {
    vec![
        HardwareInterconnect {
            name: "external_device_memory".to_string(),
            kind: "dma_buf".to_string(),
            availability: availability(facts.shared_device_memory_supported),
            api: "vulkan".to_string(),
            operations: operations_if(
                facts.shared_device_memory_supported,
                &["cross_device_alias", "export", "import"],
            ),
            properties: BTreeMap::new(),
        },
        HardwareInterconnect {
            name: "external_host_memory".to_string(),
            kind: "host_pointer_import".to_string(),
            availability: availability(facts.shared_host_memory_alignment.is_some()),
            api: "vulkan".to_string(),
            operations: operations_if(
                facts.shared_host_memory_alignment.is_some(),
                &["host_gpu_shared_allocation", "import"],
            ),
            properties: BTreeMap::from([(
                "minimum_alignment_bytes".to_string(),
                facts.shared_host_memory_alignment.unwrap_or(0).to_string(),
            )]),
        },
        HardwareInterconnect {
            name: "external_timeline_semaphore".to_string(),
            kind: "opaque_fd_timeline_semaphore".to_string(),
            availability: availability(facts.external_timeline_semaphore_supported),
            api: "vulkan".to_string(),
            operations: operations_if(
                facts.external_timeline_semaphore_supported,
                &["cross_device_signal", "cross_device_wait", "export", "import"],
            ),
            properties: BTreeMap::new(),
        },
        HardwareInterconnect {
            name: "host_staging_transfer".to_string(),
            kind: "host_staging".to_string(),
            availability: availability(facts.transfer_queue_count > 0),
            api: "vulkan".to_string(),
            operations: operations_if(
                facts.transfer_queue_count > 0,
                &["device_to_host", "host_to_device"],
            ),
            properties: BTreeMap::new(),
        },
    ]
}

fn available_vulkan_process(
    name: &str,
    category: HardwareProcessCategory,
    operations: &[&str],
) -> HardwareProcessCapability {
    let mut process = HardwareProcessCapability::new(
        name,
        category,
        HardwareProcessAvailability::Available,
        HardwareProcessProgrammability::Direct,
        "vulkan",
    );
    process.operations = operations.iter().map(|value| (*value).to_string()).collect();
    process
}

fn conditional_process(
    name: &str,
    category: HardwareProcessCategory,
    available: bool,
    operations: &[&str],
    formats: &[&str],
    extensions: &[&str],
) -> HardwareProcessCapability {
    let mut process = HardwareProcessCapability::new(
        name,
        category,
        availability(available),
        if available {
            HardwareProcessProgrammability::Direct
        } else {
            HardwareProcessProgrammability::None
        },
        "vulkan",
    );
    if available {
        process.operations = operations.iter().map(|value| (*value).to_string()).collect();
        process.numeric_formats = formats.iter().map(|value| (*value).to_string()).collect();
        process.required_extensions =
            extensions.iter().map(|value| (*value).to_string()).collect();
    }
    process
}

fn availability(value: bool) -> HardwareProcessAvailability {
    if value {
        HardwareProcessAvailability::Available
    } else {
        HardwareProcessAvailability::Unavailable
    }
}

fn operations_if(condition: bool, operations: &[&str]) -> Vec<String> {
    if condition {
        operations.iter().map(|value| (*value).to_string()).collect()
    } else {
        Vec::new()
    }
}

fn shader_numeric_formats(target: &VulkanComputeTargetCapabilities) -> Vec<String> {
    let mut formats = BTreeSet::from([
        "f32".to_string(),
        "i32".to_string(),
        "u32".to_string(),
    ]);
    for feature in &target.shader_features {
        match feature {
            VulkanShaderFeature::ShaderFloat16 => {
                formats.insert("f16".to_string());
            }
            VulkanShaderFeature::ShaderFloat64 => {
                formats.insert("f64".to_string());
            }
            VulkanShaderFeature::ShaderInt8 => {
                formats.extend(["i8".to_string(), "u8".to_string()]);
            }
            VulkanShaderFeature::ShaderInt16 => {
                formats.extend(["i16".to_string(), "u16".to_string()]);
            }
            VulkanShaderFeature::ShaderInt64 => {
                formats.extend(["i64".to_string(), "u64".to_string()]);
            }
            VulkanShaderFeature::ShaderFloat8 => {
                formats.insert("f8_e4m3".to_string());
            }
            VulkanShaderFeature::ShaderBfloat16Type => {
                formats.insert("bf16".to_string());
            }
            _ => {}
        }
    }
    formats.into_iter().collect()
}

fn matrix_shapes(shapes: &BTreeSet<(u32, u32, u32)>) -> String {
    shapes
        .iter()
        .map(|(m, n, k)| format!("{m}x{n}x{k}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn vulkan_version(version: u32) -> String {
    format!(
        "{}.{}.{}",
        vk::api_version_major(version),
        vk::api_version_minor(version),
        vk::api_version_patch(version)
    )
}
