#[test]
fn synthetic_vulkan_hardware_profile_covers_exposed_and_unavailable_processes() {
    let device = VulkanComputeDeviceInfo {
        physical_device_index: 2,
        physical_device_id: "vulkan-uuid:00112233445566778899aabbccddeeff".to_string(),
        device_uuid: [7; vk::UUID_SIZE],
        device_name: "synthetic GPU".to_string(),
        pci_address: Some("0000:02:00.0".to_string()),
        device_type: "discrete_gpu".to_string(),
        vendor_id: 0x1002,
        device_id: 0xabcd,
        api_version: vk::make_api_version(0, 1, 4, 0),
        driver_version: 42,
        compute_queue_family_indices: vec![0],
        memory_heaps: vec![VulkanMemoryHeapInfo {
            heap_index: 0,
            size_bytes: 16 * 1024 * 1024 * 1024,
            device_local: true,
        }],
        selected_by_default: true,
    };
    let target = VulkanComputeTargetCapabilities {
        physical_device_index: 2,
        physical_device_id: device.physical_device_id.clone(),
        device_name: device.device_name.clone(),
        device_type: device.device_type.clone(),
        vendor_id: device.vendor_id,
        device_id: device.device_id,
        shader_features: BTreeSet::from([
            VulkanShaderFeature::ShaderFloat16,
            VulkanShaderFeature::ShaderInt8,
            VulkanShaderFeature::ShaderIntegerDotProduct,
            VulkanShaderFeature::CooperativeMatrix,
        ]),
        subgroup_operations: BTreeSet::from([
            VulkanSubgroupOperation::Basic,
            VulkanSubgroupOperation::Arithmetic,
            VulkanSubgroupOperation::Shuffle,
        ]),
        subgroup_compute_supported: true,
        subgroup_size: 64,
        max_compute_work_group_invocations: 1024,
        max_compute_work_group_size_x: 1024,
        cooperative_float16_shapes: BTreeSet::from([(16, 16, 16)]),
        cooperative_bfloat16_shapes: BTreeSet::new(),
        cooperative_float8_e4m3_shapes: BTreeSet::new(),
        cooperative_sint8_shapes: BTreeSet::from([(16, 16, 32)]),
    };
    let facts = VulkanHardwareFacts {
        api_version: device.api_version,
        driver_version: device.driver_version,
        driver_name: "synthetic Vulkan driver".to_string(),
        driver_info: "synthetic driver info".to_string(),
        driver_id: "MESA_RADV".to_string(),
        queue_family_count: 3,
        compute_queue_count: 2,
        graphics_queue_count: 1,
        transfer_queue_count: 3,
        transfer_only_queue_count: 1,
        video_decode_queue_count: 1,
        video_encode_queue_count: 0,
        extension_names: BTreeSet::from([
            "VK_KHR_acceleration_structure".to_string(),
            "VK_KHR_ray_query".to_string(),
            "VK_KHR_video_decode_queue".to_string(),
            "VK_KHR_video_queue".to_string(),
        ]),
        sampled_formats: BTreeSet::from([
            "r16_sfloat".to_string(),
            "r32_sfloat".to_string(),
        ]),
        storage_image_formats: BTreeSet::from(["r32_sfloat".to_string()]),
        linear_filter_formats: BTreeSet::from(["r16_sfloat".to_string()]),
        cooperative_matrix_variants: BTreeSet::from([
            "a=FLOAT16;b=FLOAT16;c=FLOAT32;result=FLOAT32;scope=SUBGROUP;m=16;n=16;k=16;saturating=false".to_string(),
        ]),
        max_compute_work_group_count_x: 65_535,
        max_compute_shared_memory_size: 65_536,
        max_storage_buffer_range: u64::from(u32::MAX),
        max_uniform_buffer_range: 65_536,
        min_storage_buffer_offset_alignment: 32,
        min_uniform_buffer_offset_alignment: 256,
        max_image_dimension_1d: 16_384,
        max_image_dimension_2d: 16_384,
        max_image_dimension_3d: 2_048,
        max_bound_descriptor_sets: 8,
        timestamp_compute_and_graphics: true,
        timestamp_period_bits: 1.0_f32.to_bits(),
        shared_host_memory_alignment: Some(4096),
        shared_device_memory_supported: true,
        external_timeline_semaphore_supported: true,
        timeline_semaphore_supported: true,
        synchronization2_supported: true,
        memory_types: vec![
            VulkanMemoryTypeFacts {
                type_index: 0,
                heap_index: 0,
                host_visible: true,
                device_local: true,
                coherent: true,
                cached: false,
                lazily_allocated: false,
            },
            VulkanMemoryTypeFacts {
                type_index: 1,
                heap_index: 0,
                host_visible: false,
                device_local: true,
                coherent: false,
                cached: false,
                lazily_allocated: false,
            },
        ],
    };

    let profile =
        build_vulkan_hardware_profile(&device, target.clone(), facts.clone()).unwrap();
    profile.validate().unwrap();
    let process = |name: &str| {
        profile
            .processes
            .iter()
            .find(|process| process.name == name)
            .unwrap()
    };
    for required in [
        "acceleration_structure_construction",
        "blending",
        "command_queues",
        "cooperative_matrix",
        "copy_engines",
        "depth_stencil",
        "device_cache_hierarchy",
        "device_generated_commands",
        "device_memory_bandwidth",
        "execution_graphs",
        "fixed_function_interpolation",
        "indirect_work_generation",
        "occupancy_constraints",
        "packed_dot_product",
        "parallel_collective_algorithms",
        "rasterization",
        "ray_traversal",
        "register_file",
        "resident_command_replay",
        "shader_atomics",
        "shader_scalar",
        "shader_vector",
        "subgroup_collectives",
        "synchronization",
        "texture_sampling",
        "video_decode",
        "video_encode",
        "workgroup_shared_memory",
    ] {
        process(required);
    }
    assert_eq!(
        process("cooperative_matrix").availability,
        HardwareProcessAvailability::Available
    );
    assert!(
        !process("cooperative_matrix").properties["all_reported_variants"]
            .is_empty()
    );
    assert!(process("cooperative_matrix")
        .numeric_formats
        .contains(&"i8".to_string()));
    assert_eq!(
        process("cooperative_matrix").properties["sint8_shapes"],
        "16x16x32"
    );
    assert_eq!(
        process("ray_traversal").availability,
        HardwareProcessAvailability::Available
    );
    assert_eq!(
        process("video_encode").availability,
        HardwareProcessAvailability::Unavailable
    );
    assert_eq!(
        process("register_file")
            .properties
            .get("capacity_visibility")
            .map(String::as_str),
        Some("opaque_to_vulkan")
    );
    assert_eq!(
        profile.runtime_bindings["vulkan_runtime_binding"]["physical_device_id"],
        target.physical_device_id
    );
    assert_eq!(
        profile.capability_extensions["vulkan_compiler_capabilities"]
            ["subgroup_size"],
        target.subgroup_size
    );
    assert!(
        profile.capability_extensions["vulkan_compiler_capabilities"]
            .get("physical_device_id")
            .is_none()
    );
    assert!(profile.memory_domains[0].host_visible);
    assert!(profile.memory_domains[0].coherent);
    assert!(!profile.memory_domains[1].host_visible);

    let mut invalid_memory = facts.clone();
    invalid_memory.memory_types[0].heap_index = 99;
    assert!(
        build_vulkan_hardware_profile(&device, target.clone(), invalid_memory)
            .unwrap_err()
            .contains("references missing heap")
    );

    let mut peer_device = device.clone();
    peer_device.physical_device_index = 3;
    peer_device.physical_device_id =
        "vulkan-uuid:ffeeddccbbaa99887766554433221100".to_string();
    peer_device.device_uuid = [8; vk::UUID_SIZE];
    let mut peer_target = target.clone();
    peer_target.physical_device_index = peer_device.physical_device_index;
    peer_target.physical_device_id = peer_device.physical_device_id.clone();
    let peer =
        build_vulkan_hardware_profile(&peer_device, peer_target, facts.clone()).unwrap();
    assert_eq!(profile.capability_class, peer.capability_class);
    assert_ne!(profile.profile_id, peer.profile_id);

    let rebuilt = build_vulkan_hardware_profile(&device, target, facts).unwrap();
    assert_eq!(profile.profile_id, rebuilt.profile_id);
    assert_eq!(profile.capability_class, rebuilt.capability_class);
}
#[test]
fn physical_device_allowlist_preserves_real_indices_and_excludes_other_devices() {
    let first = "vulkan-uuid:00000000070000000000000000000000".to_string();
    let second = "vulkan-uuid:000000000a0000000000000000000000".to_string();
    let forbidden = "vulkan-uuid:ffffffffffffffffffffffffffffffff".to_string();
    let discovered = vec![(2, first.clone()), (5, forbidden), (9, second.clone())];

    assert_eq!(
        allowed_physical_device_indices(
            &discovered,
            Some(&BTreeSet::from([first.clone(), second.clone()])),
        )
        .unwrap(),
        vec![2, 9]
    );
    assert_eq!(
        allowed_physical_device_indices(&discovered, None).unwrap(),
        vec![2, 5, 9]
    );

    let missing = allowed_physical_device_indices(
        &discovered,
        Some(&BTreeSet::from([
            first,
            "vulkan-uuid:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee".to_string(),
        ])),
    )
    .unwrap_err();
    assert!(missing.0.contains("are not present"));
}
#[test]
fn memory_type_selection_rejects_implicit_amd_coherent_memory() {
    let mut properties = vk::PhysicalDeviceMemoryProperties {
        memory_type_count: 2,
        memory_heap_count: 1,
        ..Default::default()
    };
    properties.memory_heaps[0].size = 16 * 1024 * 1024;
    properties.memory_types[0] = vk::MemoryType {
        property_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL
            | vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT,
        heap_index: 0,
    };
    properties.memory_types[1] = vk::MemoryType {
        property_flags: properties.memory_types[0].property_flags
            | vk::MemoryPropertyFlags::DEVICE_COHERENT_AMD
            | vk::MemoryPropertyFlags::DEVICE_UNCACHED_AMD,
        heap_index: 0,
    };

    assert_eq!(
        select_memory_type_index(
            &properties,
            0b11,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
        ),
        Some(0)
    );
}

#[test]
fn staging_memory_selection_excludes_device_local_bar_memory() {
    let mut properties = vk::PhysicalDeviceMemoryProperties {
        memory_type_count: 2,
        memory_heap_count: 2,
        ..Default::default()
    };
    properties.memory_heaps[0].size = 32 * 1024 * 1024;
    properties.memory_heaps[1].size = 128 * 1024 * 1024;
    properties.memory_types[0] = vk::MemoryType {
        property_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL
            | vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT,
        heap_index: 1,
    };
    properties.memory_types[1] = vk::MemoryType {
        property_flags: vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT
            | vk::MemoryPropertyFlags::HOST_CACHED,
        heap_index: 0,
    };

    assert_eq!(
        select_memory_type_index_excluding(
            &properties,
            0b11,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::HOST_CACHED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ),
        Some(1)
    );
    assert_eq!(
        select_memory_type_index_excluding(
            &properties,
            0b01,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::HOST_CACHED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ),
        None
    );
    let staging_bits = safe_staging_memory_type_bits(&properties);
    let preferred_bits = staging_bits & cached_memory_type_bits(&properties);
    assert_eq!(staging_bits, 0b10);
    assert_eq!(preferred_bits, 0b10);
    assert_eq!(
        select_compatible_staging_memory_type_index(0b11, staging_bits, preferred_bits),
        Some(1)
    );
    assert_eq!(
        select_compatible_staging_memory_type_index(0b01, staging_bits, preferred_bits),
        None
    );
}
