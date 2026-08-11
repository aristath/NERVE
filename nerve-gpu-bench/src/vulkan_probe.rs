use std::ffi::{CStr, CString};

use ash::{Entry, vk};

use crate::model::{
    FormatCapability, Target, VulkanDeviceInfo, VulkanMemoryHeap, VulkanQueueFamily,
};
use crate::vulkan_features::{
    EXTERNAL_MEMORY_DMA_BUF_FEATURE, EXTERNAL_MEMORY_HOST_FEATURE,
    EXTERNAL_TIMELINE_SEMAPHORE_FEATURE, NATIVE_FP8_DOT_FEATURE,
    extension_names_include_dma_buf_shared_memory, extension_names_include_host_shared_memory,
    external_timeline_semaphore_supported, native_fp8_dot_supported,
};

pub fn discover_vulkan_targets() -> Vec<Target> {
    match try_discover_vulkan_targets() {
        Ok(targets) => targets,
        Err(message) => vec![vulkan_unavailable_target(message)],
    }
}

fn try_discover_vulkan_targets() -> Result<Vec<Target>, String> {
    let entry = unsafe { Entry::load() }
        .map_err(|error| format!("could not load Vulkan loader: {error}"))?;
    let app_name = CString::new("nerve-gpu-bench").expect("static string has no nul");
    let engine_name = CString::new("nerve").expect("static string has no nul");
    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(1)
        .engine_name(&engine_name)
        .engine_version(1)
        .api_version(vk::make_api_version(0, 1, 4, 0));
    let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance = unsafe { entry.create_instance(&instance_info, None) }
        .map_err(|error| format!("could not create Vulkan instance: {error:?}"))?;
    let result = unsafe { discover_physical_devices(&instance) };
    unsafe { instance.destroy_instance(None) };
    result
}

unsafe fn discover_physical_devices(instance: &ash::Instance) -> Result<Vec<Target>, String> {
    let physical_devices = unsafe { instance.enumerate_physical_devices() }
        .map_err(|error| format!("could not enumerate Vulkan physical devices: {error:?}"))?;
    let mut targets = Vec::new();
    for (index, physical_device) in physical_devices.iter().copied().enumerate() {
        let properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let device_uuid = unsafe { vulkan_device_uuid(instance, physical_device) };
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let queue_properties =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let extensions = unsafe { instance.enumerate_device_extension_properties(physical_device) }
            .map_err(|error| {
                format!("could not enumerate Vulkan device extensions for index {index}: {error:?}")
            })?;
        let extension_names = extension_names(&extensions);
        let feature_flags =
            unsafe { vulkan_feature_flags(instance, physical_device, &extension_names) };
        let pci_address = unsafe { vulkan_pci_address(instance, physical_device, &extensions) };
        targets.push(vulkan_target(
            index,
            device_uuid,
            properties,
            &memory_properties,
            &queue_properties,
            &extensions,
            feature_flags,
            pci_address,
        ));
    }
    Ok(targets)
}

fn vulkan_target(
    index: usize,
    device_uuid: [u8; vk::UUID_SIZE],
    properties: vk::PhysicalDeviceProperties,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    queue_properties: &[vk::QueueFamilyProperties],
    extensions: &[vk::ExtensionProperties],
    feature_flags: Vec<String>,
    pci_address: Option<String>,
) -> Target {
    let device_name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    let vendor_id = format!("0x{:04x}", properties.vendor_id);
    let device_id = format!("0x{:04x}", properties.device_id);
    let device_type = device_type(properties.device_type).to_string();
    let api_version = api_version(properties.api_version);
    let extension_names = extension_names(extensions);
    let format_capabilities = vulkan_format_capabilities(&extension_names, &feature_flags);
    let memory_heaps = vulkan_memory_heaps(memory_properties);
    let queue_families = vulkan_queue_families(queue_properties);
    let stable_target_id = vulkan_stable_target_id(&device_uuid);
    let has_compute = queue_families
        .iter()
        .any(|family| family.flags.iter().any(|flag| flag == "compute"));
    let mut capabilities = vec![
        format!("vulkan_device_type={device_type}"),
        format!("vulkan_api_version={api_version}"),
        format!("vulkan_driver_version={}", properties.driver_version),
        format!("vulkan_memory_heap_count={}", memory_heaps.len()),
        format!("vulkan_queue_family_count={}", queue_families.len()),
    ];
    if has_compute {
        capabilities.push("vulkan_compute_queue".to_string());
    }
    capabilities.extend(
        feature_flags
            .iter()
            .map(|feature| format!("vulkan_feature={feature}")),
    );

    Target {
        stable_target_id: stable_target_id.clone(),
        backend: "vulkan".to_string(),
        kind: vulkan_target_kind(properties.device_type).to_string(),
        name: device_name.clone(),
        vendor_id: Some(vendor_id.clone()),
        vendor_name: Some(vendor_name(properties.vendor_id).to_string()),
        device_id: Some(device_id.clone()),
        pci_address: pci_address.clone(),
        physical_location: Some(
            pci_address
                .as_deref()
                .map(|address| format!("pci:{address}"))
                .unwrap_or_else(|| stable_target_id.clone()),
        ),
        numa_node: None,
        boot_vga: None,
        pci_link: None,
        vulkan: Some(VulkanDeviceInfo {
            physical_device_index: index,
            device_name,
            device_type,
            api_version,
            driver_version: properties.driver_version,
            vendor_id,
            device_id,
            memory_heaps,
            queue_families,
            extension_names,
            feature_flags,
        }),
        capabilities,
        format_capabilities,
        diagnostics: vec![
            "vulkan_physical_device_order_is_driver_defined".to_string(),
            "vulkan_probe_does_not_create_logical_device_or_run_workloads".to_string(),
        ],
    }
}

unsafe fn vulkan_feature_flags(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    extension_names: &[String],
) -> Vec<String> {
    let mut shader_float16_int8 = vk::PhysicalDeviceShaderFloat16Int8Features::default();
    let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut shader_float16_int8);
    unsafe {
        instance.get_physical_device_features2(physical_device, &mut features);
    }
    let mut flags = Vec::new();
    if shader_float16_int8.shader_float16 != 0 {
        flags.push("shader_float16".to_string());
    }
    if shader_float16_int8.shader_int8 != 0 {
        flags.push("shader_int8".to_string());
    }
    if unsafe { native_fp8_dot_supported(instance, physical_device, extension_names) } {
        flags.push(NATIVE_FP8_DOT_FEATURE.to_string());
    }
    if unsafe { external_timeline_semaphore_supported(instance, physical_device, extension_names) }
    {
        flags.push(EXTERNAL_TIMELINE_SEMAPHORE_FEATURE.to_string());
    }
    if extension_names_include_dma_buf_shared_memory(extension_names) {
        flags.push(EXTERNAL_MEMORY_DMA_BUF_FEATURE.to_string());
    }
    if extension_names_include_host_shared_memory(extension_names) {
        flags.push(EXTERNAL_MEMORY_HOST_FEATURE.to_string());
    }
    flags
}

unsafe fn vulkan_pci_address(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    extensions: &[vk::ExtensionProperties],
) -> Option<String> {
    let supports_pci_info = extensions.iter().any(|extension| {
        (unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) })
            == ash::ext::pci_bus_info::NAME
    });
    if !supports_pci_info {
        return None;
    }
    let mut pci = vk::PhysicalDevicePCIBusInfoPropertiesEXT::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut pci);
    unsafe {
        instance.get_physical_device_properties2(physical_device, &mut properties);
    }
    Some(format!(
        "{:04x}:{:02x}:{:02x}.{:x}",
        pci.pci_domain, pci.pci_bus, pci.pci_device, pci.pci_function
    ))
}

unsafe fn vulkan_device_uuid(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> [u8; vk::UUID_SIZE] {
    let mut id = vk::PhysicalDeviceIDProperties::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
    unsafe {
        instance.get_physical_device_properties2(physical_device, &mut properties);
    }
    id.device_uuid
}

fn vulkan_stable_target_id(device_uuid: &[u8; vk::UUID_SIZE]) -> String {
    let mut id = String::with_capacity("vulkan-uuid:".len() + vk::UUID_SIZE * 2);
    id.push_str("vulkan-uuid:");
    for byte in device_uuid {
        use std::fmt::Write as _;
        write!(id, "{byte:02x}").expect("writing to a string cannot fail");
    }
    id
}

fn vulkan_unavailable_target(message: String) -> Target {
    Target {
        stable_target_id: "vulkan:unavailable".to_string(),
        backend: "vulkan".to_string(),
        kind: "unavailable".to_string(),
        name: "Vulkan unavailable".to_string(),
        vendor_id: None,
        vendor_name: None,
        device_id: None,
        pci_address: None,
        physical_location: Some("vulkan".to_string()),
        numa_node: None,
        boot_vga: None,
        pci_link: None,
        vulkan: None,
        capabilities: Vec::new(),
        format_capabilities: Vec::new(),
        diagnostics: vec![message],
    }
}

fn vulkan_memory_heaps(
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
) -> Vec<VulkanMemoryHeap> {
    (0..memory_properties.memory_heap_count)
        .map(|heap_index| {
            let heap = memory_properties.memory_heaps[heap_index as usize];
            VulkanMemoryHeap {
                heap_index,
                size_bytes: heap.size,
                flags: memory_heap_flags(heap.flags),
            }
        })
        .collect()
}

fn vulkan_queue_families(queue_properties: &[vk::QueueFamilyProperties]) -> Vec<VulkanQueueFamily> {
    queue_properties
        .iter()
        .enumerate()
        .map(|(index, family)| VulkanQueueFamily {
            family_index: index as u32,
            queue_count: family.queue_count,
            flags: queue_flags(family.queue_flags),
        })
        .collect()
}

fn extension_names(extensions: &[vk::ExtensionProperties]) -> Vec<String> {
    let mut names = extensions
        .iter()
        .map(|extension| unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) })
        .map(|name| name.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    names.sort();
    names
}

fn vulkan_format_capabilities(
    _extension_names: &[String],
    feature_flags: &[String],
) -> Vec<FormatCapability> {
    let supports_f16 = feature_flags
        .iter()
        .any(|feature| feature == "shader_float16");
    let supports_int8 = feature_flags.iter().any(|feature| feature == "shader_int8");
    let supports_native_fp8 = feature_flags
        .iter()
        .any(|feature| feature == NATIVE_FP8_DOT_FEATURE);
    vec![
        format_capability("f32", "native", "vulkan_core", "32-bit float shader path"),
        format_capability(
            "f16",
            if supports_f16 { "native" } else { "fallback" },
            "vulkan_feature_chain",
            if supports_f16 {
                "shaderFloat16 feature bit is set"
            } else {
                "shaderFloat16 is unavailable; packed F16 is decoded into F32"
            },
        ),
        format_capability(
            "bf16",
            "fallback",
            "vulkan_format_dequant_kernel",
            "BF16 storage is decoded by the format-specific dequant benchmark path",
        ),
        fp8_format_capability("fp8_e4m3", supports_native_fp8),
        fp8_format_capability("fp8_e5m2", false),
        format_capability(
            "int8",
            "fallback",
            "vulkan_feature_chain",
            if supports_int8 {
                "router reduction can use shaderInt8; dense and MoE paths decode packed INT8 into F32"
            } else {
                "packed INT8 is decoded into F32"
            },
        ),
        format_dequant_capability("int4"),
        format_dequant_capability("fp4"),
        format_capability(
            "mxfp4",
            if supports_native_fp8 {
                "native"
            } else {
                "fallback"
            },
            if supports_native_fp8 {
                "vulkan_mixed_float_dot_product"
            } else {
                "vulkan_format_dequant_kernel"
            },
            if supports_native_fp8 {
                "compact E2M1 values expand to E4M3 and use native FP8 dot products"
            } else {
                "compact E2M1 values and E8M0 scales are decoded into F32 arithmetic"
            },
        ),
        format_dequant_capability("q8_0"),
    ]
}

fn fp8_format_capability(format: &str, native: bool) -> FormatCapability {
    format_capability(
        format,
        if native { "native" } else { "fallback" },
        if native {
            "vulkan_mixed_float_dot_product"
        } else {
            "vulkan_format_dequant_kernel"
        },
        if native {
            "block-scaled E4M3 weights use native FP8 dot products with F32 accumulation"
        } else {
            "block-scaled FP8 storage is decoded into F32 arithmetic by the measured path"
        },
    )
}

fn format_dequant_capability(format: &str) -> FormatCapability {
    format_capability(
        format,
        "fallback",
        "vulkan_format_dequant_kernel",
        "format-specific dequant benchmark path can run",
    )
}

fn format_capability(format: &str, support: &str, source: &str, notes: &str) -> FormatCapability {
    FormatCapability {
        format: format.to_string(),
        support: support.to_string(),
        source: source.to_string(),
        notes: notes.to_string(),
    }
}

fn memory_heap_flags(flags: vk::MemoryHeapFlags) -> Vec<String> {
    let mut labels = Vec::new();
    if flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
        labels.push("device_local".to_string());
    }
    if flags.contains(vk::MemoryHeapFlags::MULTI_INSTANCE) {
        labels.push("multi_instance".to_string());
    }
    labels
}

fn queue_flags(flags: vk::QueueFlags) -> Vec<String> {
    let mut labels = Vec::new();
    if flags.contains(vk::QueueFlags::GRAPHICS) {
        labels.push("graphics".to_string());
    }
    if flags.contains(vk::QueueFlags::COMPUTE) {
        labels.push("compute".to_string());
    }
    if flags.contains(vk::QueueFlags::TRANSFER) {
        labels.push("transfer".to_string());
    }
    if flags.contains(vk::QueueFlags::SPARSE_BINDING) {
        labels.push("sparse_binding".to_string());
    }
    if flags.contains(vk::QueueFlags::PROTECTED) {
        labels.push("protected".to_string());
    }
    labels
}

fn device_type(device_type: vk::PhysicalDeviceType) -> &'static str {
    match device_type {
        vk::PhysicalDeviceType::INTEGRATED_GPU => "integrated_gpu",
        vk::PhysicalDeviceType::DISCRETE_GPU => "discrete_gpu",
        vk::PhysicalDeviceType::VIRTUAL_GPU => "virtual_gpu",
        vk::PhysicalDeviceType::CPU => "cpu",
        _ => "other",
    }
}

fn vulkan_target_kind(device_type: vk::PhysicalDeviceType) -> &'static str {
    match device_type {
        vk::PhysicalDeviceType::INTEGRATED_GPU => "integrated_gpu",
        vk::PhysicalDeviceType::DISCRETE_GPU => "discrete_gpu",
        vk::PhysicalDeviceType::VIRTUAL_GPU => "virtual_gpu",
        vk::PhysicalDeviceType::CPU => "cpu",
        _ => "gpu",
    }
}

fn api_version(version: u32) -> String {
    format!(
        "{}.{}.{}",
        vk::api_version_major(version),
        vk::api_version_minor(version),
        vk::api_version_patch(version)
    )
}

fn vendor_name(vendor_id: u32) -> &'static str {
    match vendor_id {
        0x1002 => "AMD",
        0x10de => "NVIDIA",
        0x8086 => "Intel",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_target_ids_match_the_runtime_device_uuid_identity() {
        let uuid = [
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc,
            0xdd, 0xee, 0xff,
        ];
        assert_eq!(
            vulkan_stable_target_id(&uuid),
            "vulkan-uuid:00112233445566778899aabbccddeeff"
        );
    }

    #[test]
    fn api_versions_are_human_readable() {
        assert_eq!(api_version(vk::make_api_version(0, 1, 3, 268)), "1.3.268");
    }

    #[test]
    fn format_capabilities_use_feature_bits_for_f16() {
        let capabilities = vulkan_format_capabilities(&[], &["shader_float16".to_string()]);
        let f16 = capabilities
            .iter()
            .find(|capability| capability.format == "f16")
            .unwrap();
        assert_eq!(f16.support, "native");

        let capabilities = vulkan_format_capabilities(&[], &[]);
        let f16 = capabilities
            .iter()
            .find(|capability| capability.format == "f16")
            .unwrap();
        assert_eq!(f16.support, "fallback");
    }

    #[test]
    fn bf16_and_fp8_use_format_dequant_capabilities() {
        let capabilities = vulkan_format_capabilities(&[], &[]);
        for format in ["bf16", "fp8_e4m3", "fp8_e5m2"] {
            let capability = capabilities
                .iter()
                .find(|capability| capability.format == format)
                .unwrap();
            assert_eq!(capability.support, "fallback");
            assert_eq!(capability.source, "vulkan_format_dequant_kernel");
        }
    }

    #[test]
    fn native_fp8_dot_capability_selects_e4m3_and_mxfp4_native_paths() {
        let capabilities = vulkan_format_capabilities(&[], &[NATIVE_FP8_DOT_FEATURE.to_string()]);
        for format in ["fp8_e4m3", "mxfp4"] {
            let capability = capabilities
                .iter()
                .find(|capability| capability.format == format)
                .unwrap();
            assert_eq!(capability.support, "native");
            assert_eq!(capability.source, "vulkan_mixed_float_dot_product");
        }
        let e5m2 = capabilities
            .iter()
            .find(|capability| capability.format == "fp8_e5m2")
            .unwrap();
        assert_eq!(e5m2.support, "fallback");
    }

    #[test]
    fn int4_uses_format_dequant_without_native_int8() {
        let capabilities = vulkan_format_capabilities(&[], &["shader_int8".to_string()]);
        let int4 = capabilities
            .iter()
            .find(|capability| capability.format == "int4")
            .unwrap();
        assert_eq!(int4.support, "fallback");

        let capabilities = vulkan_format_capabilities(&[], &[]);
        let int4 = capabilities
            .iter()
            .find(|capability| capability.format == "int4")
            .unwrap();
        assert_eq!(int4.support, "fallback");
    }

    #[test]
    fn int8_capability_does_not_overstate_router_only_native_support() {
        let capabilities = vulkan_format_capabilities(&[], &["shader_int8".to_string()]);
        let int8 = capabilities
            .iter()
            .find(|capability| capability.format == "int8")
            .unwrap();
        assert_eq!(int8.support, "fallback");
        assert!(int8.notes.contains("router reduction"));
        assert!(int8.notes.contains("dense and MoE"));
    }
}
