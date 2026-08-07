use std::ffi::{CStr, CString};

use ash::{Entry, vk};

use crate::model::{
    FormatCapability, Target, VulkanDeviceInfo, VulkanMemoryHeap, VulkanQueueFamily,
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
        .api_version(vk::make_api_version(0, 1, 3, 0));
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
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let queue_properties =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let extensions = unsafe { instance.enumerate_device_extension_properties(physical_device) }
            .map_err(|error| {
                format!("could not enumerate Vulkan device extensions for index {index}: {error:?}")
            })?;
        let pci_address = unsafe { vulkan_pci_address(instance, physical_device, &extensions) };
        targets.push(vulkan_target(
            index,
            properties,
            &memory_properties,
            &queue_properties,
            &extensions,
            pci_address,
        ));
    }
    Ok(targets)
}

fn vulkan_target(
    index: usize,
    properties: vk::PhysicalDeviceProperties,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    queue_properties: &[vk::QueueFamilyProperties],
    extensions: &[vk::ExtensionProperties],
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
    let format_capabilities = vulkan_format_capabilities(&extension_names);
    let memory_heaps = vulkan_memory_heaps(memory_properties);
    let queue_families = vulkan_queue_families(queue_properties);
    let stable_target_id = vulkan_stable_target_id(
        index,
        &vendor_id,
        &device_id,
        &device_name,
        pci_address.as_deref(),
    );
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

    Target {
        stable_target_id,
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
                .unwrap_or_else(|| format!("vulkan:{index}")),
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
        }),
        capabilities,
        format_capabilities,
        diagnostics: vec![
            "vulkan_physical_device_order_is_driver_defined".to_string(),
            "vulkan_probe_does_not_create_logical_device_or_run_workloads".to_string(),
        ],
    }
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

fn vulkan_stable_target_id(
    index: usize,
    vendor_id: &str,
    device_id: &str,
    device_name: &str,
    pci_address: Option<&str>,
) -> String {
    pci_address
        .map(|address| format!("vulkan:pci:{address}"))
        .unwrap_or_else(|| {
            format!(
                "vulkan:{index}:{vendor_id}:{device_id}:{}",
                stable_name_fragment(device_name)
            )
        })
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

fn vulkan_format_capabilities(extension_names: &[String]) -> Vec<FormatCapability> {
    let supports_bf16 = extension_names
        .iter()
        .any(|extension| extension == "VK_KHR_shader_bfloat16");
    let supports_fp8 = extension_names
        .iter()
        .any(|extension| extension == "VK_EXT_shader_float8");
    vec![
        format_capability("f32", "native", "vulkan_core", "32-bit float shader path"),
        format_capability(
            "f16",
            "unmeasured",
            "vulkan_probe",
            "requires feature-chain probe before native/emulated classification",
        ),
        format_capability(
            "bf16",
            if supports_bf16 {
                "unmeasured"
            } else {
                "unsupported"
            },
            "vulkan_device_extensions",
            if supports_bf16 {
                "VK_KHR_shader_bfloat16 advertised; feature bits not queried yet"
            } else {
                "VK_KHR_shader_bfloat16 not advertised"
            },
        ),
        format_capability(
            "fp8",
            if supports_fp8 {
                "unmeasured"
            } else {
                "unsupported"
            },
            "vulkan_device_extensions",
            if supports_fp8 {
                "VK_EXT_shader_float8 advertised; feature bits not queried yet"
            } else {
                "VK_EXT_shader_float8 not advertised"
            },
        ),
        format_capability(
            "int4",
            "unmeasured",
            "vulkan_probe",
            "requires packed integer benchmark path",
        ),
        format_capability(
            "fp4",
            "unmeasured",
            "vulkan_probe",
            "requires packed float benchmark path",
        ),
    ]
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

fn stable_name_fragment(name: &str) -> String {
    let mut fragment = name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>();
    while fragment.contains("--") {
        fragment = fragment.replace("--", "-");
    }
    fragment.trim_matches('-').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_name_fragments_are_target_id_safe() {
        assert_eq!(
            stable_name_fragment("AMD Radeon RX 7900 XTX"),
            "amd-radeon-rx-7900-xtx"
        );
    }

    #[test]
    fn stable_target_ids_prefer_pci_addresses() {
        assert_eq!(
            vulkan_stable_target_id(2, "0x1002", "0x744c", "AMD Radeon", Some("0000:03:00.0")),
            "vulkan:pci:0000:03:00.0"
        );
        assert_eq!(
            vulkan_stable_target_id(2, "0x1002", "0x744c", "AMD Radeon", None),
            "vulkan:2:0x1002:0x744c:amd-radeon"
        );
    }

    #[test]
    fn api_versions_are_human_readable() {
        assert_eq!(api_version(vk::make_api_version(0, 1, 3, 268)), "1.3.268");
    }
}
