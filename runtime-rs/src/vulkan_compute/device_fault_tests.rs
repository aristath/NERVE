#[test]
fn device_fault_address_registry_rejects_overlap_and_resolves_boundaries() {
    let mut registry = VulkanDeviceAddressRegistry::default();
    registry
        .register(7, 0x1_0000, 0x2000, "first allocation")
        .unwrap();
    registry
        .register(8, 0x2_0000, 0x1000, "second allocation")
        .unwrap();
    registry
        .register_annotation(70, 0x1_0800, 0x400, "stable resource slot=17")
        .unwrap();

    let first = registry.resolve(0x1_1abc).unwrap();
    assert_eq!(first.label, "first allocation");
    assert_eq!(first.byte_offset, 0x1abc);
    assert_eq!(first.byte_capacity, 0x2000);
    let annotated = registry.resolve(0x1_0920).unwrap();
    assert_eq!(annotated.label, "stable resource slot=17");
    assert_eq!(annotated.byte_offset, 0x120);
    assert_eq!(annotated.byte_capacity, 0x400);
    assert!(registry.resolve(0x1_2000).is_none());

    let overlap = registry
        .register(9, 0x1_1000, 0x2000, "overlapping allocation")
        .unwrap_err();
    assert!(overlap.0.contains("overlaps"));
    let annotation_overlap = registry
        .register_annotation(71, 0x1_0900, 0x400, "overlapping slot")
        .unwrap_err();
    assert!(annotation_overlap.0.contains("overlaps"));
    let annotation_outside = registry
        .register_annotation(72, 0x1_1f00, 0x200, "outside allocation")
        .unwrap_err();
    assert!(annotation_outside.0.contains("contained"));

    registry.unregister_annotation(70, 0x1_0800).unwrap();
    assert_eq!(registry.resolve(0x1_0920).unwrap().label, "first allocation");
    registry.unregister(7, 0x1_0000).unwrap();
    assert!(registry.resolve(0x1_1abc).is_none());
    assert_eq!(registry.resolve(0x2_0000).unwrap().label, "second allocation");
}
#[test]
fn device_fault_address_registry_resolves_sign_extended_device_addresses() {
    let mut registry = VulkanDeviceAddressRegistry::default();
    registry
        .register(
            7,
            0x8000_c74c_0000,
            0x10_0000,
            "addressable allocation",
        )
        .unwrap();
    registry
        .register_annotation(
            70,
            0x8000_c74c_6000,
            0x20_000,
            "stable resource slot=17",
        )
        .unwrap();

    let (canonical, resolved) = registry
        .resolve_reported_fault_address(0xffff_8000_c74c_7000)
        .unwrap();
    assert_eq!(canonical, 0x8000_c74c_7000);
    assert_eq!(resolved.label, "stable resource slot=17");
    assert_eq!(resolved.byte_offset, 0x1000);
    assert_eq!(resolved.byte_capacity, 0x20_000);
}

#[test]
fn device_fault_address_registry_attributes_nearest_sign_extended_range() {
    let mut registry = VulkanDeviceAddressRegistry::default();
    registry
        .register(
            7,
            0x8000_c740_0000,
            0xc_0000,
            "addressable allocation",
        )
        .unwrap();
    registry
        .register_annotation(
            70,
            0x8000_c74a_0000,
            0x20_000,
            "stable resource slot=17",
        )
        .unwrap();

    let nearest = registry
        .nearest_reported_fault_address(0xffff_8000_c74c_7000)
        .unwrap();
    assert_eq!(nearest.canonical_address, 0x8000_c74c_7000);
    assert_eq!(nearest.label, "stable resource slot=17");
    assert_eq!(nearest.signed_byte_offset, 0x2_7000);
    assert_eq!(nearest.byte_capacity, 0x20_000);
    assert_eq!(nearest.gap_bytes, 0x7001);
}

#[test]
fn device_fault_address_registry_attributes_retired_sign_extended_range() {
    let mut registry = VulkanDeviceAddressRegistry::default();
    registry
        .register(
            7,
            0x8000_c740_0000,
            0x10_0000,
            "retired addressable allocation",
        )
        .unwrap();
    registry
        .register_annotation(
            70,
            0x8000_c74c_0000,
            0x20_000,
            "retired stable resource slot=17",
        )
        .unwrap();
    registry
        .unregister_annotation(70, 0x8000_c74c_0000)
        .unwrap();
    registry.unregister(7, 0x8000_c740_0000).unwrap();

    assert!(registry.resolve(0x8000_c74c_7000).is_none());
    let (canonical, retired) = registry
        .resolve_retired_reported_fault_address(0xffff_8000_c74c_7000)
        .unwrap();
    assert_eq!(canonical, 0x8000_c74c_7000);
    assert_eq!(retired.label, "retired stable resource slot=17");
    assert_eq!(retired.byte_offset, 0x7000);
    assert_eq!(retired.byte_capacity, 0x20_000);
}

#[test]
fn device_fault_error_context_is_shared_by_detached_queue_objects() {
    let registry = Arc::new(Mutex::new(VulkanDeviceAddressRegistry::default()));

    let ordinary = vulkan_operation_error_with_device_fault(
        "copy submit",
        vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
        None,
        &registry,
    );
    assert_eq!(
        ordinary.0,
        "copy submit: ERROR_OUT_OF_DEVICE_MEMORY".to_string()
    );

    let lost = vulkan_operation_error_with_device_fault(
        "copy submit",
        vk::Result::ERROR_DEVICE_LOST,
        None,
        &registry,
    );
    assert!(lost.0.contains("copy submit: ERROR_DEVICE_LOST"));
    assert!(lost.0.contains("VK_EXT_device_fault is unavailable"));
}
