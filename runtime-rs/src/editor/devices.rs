pub fn discover_runtime_devices(
    default_device_id: &str,
    selected_vulkan_device_index: Option<usize>,
) -> Vec<RuntimeAvailableDevice> {
    match VulkanComputeDeviceCatalog::discover() {
        Ok(catalog)
            if catalog.available_compute_devices().is_empty() =>
        {
            vec![unavailable_device(
                default_device_id,
                "no compute-capable Vulkan physical devices were found",
                None,
            )]
        }
        Ok(catalog) => {
            let profiles = match catalog.available_hardware_profiles() {
                Ok(profiles) => profiles
                    .into_iter()
                    .map(|profile| {
                        (
                            profile
                                .hardware_identity
                                .stable_device_id
                                .clone(),
                            profile,
                        )
                    })
                    .collect::<BTreeMap<_, _>>(),
                Err(error) => {
                    return vec![unavailable_device(
                        default_device_id,
                        "Vulkan hardware-profile discovery failed",
                        Some(error.to_string()),
                    )];
                }
            };
            runtime_devices_from_compute_devices(
                default_device_id,
                selected_vulkan_device_index,
                catalog.available_compute_devices(),
            )
            .into_iter()
            .map(|mut device| {
                device.hardware_profile = device
                    .physical_device_id
                    .as_ref()
                    .and_then(|physical_id| {
                        profiles.get(physical_id).cloned()
                    });
                device
            })
            .collect()
        }
        Err(error) => vec![unavailable_device(
            default_device_id,
            "Vulkan device discovery failed",
            Some(error.to_string()),
        )],
    }
}

pub fn runtime_devices_from_compute_devices(
    default_device_id: &str,
    selected_vulkan_device_index: Option<usize>,
    devices: &[VulkanComputeDeviceInfo],
) -> Vec<RuntimeAvailableDevice> {
    let mut cpu_device_ordinal = 0usize;
    devices
        .iter()
        .map(|device| {
            let selected_by_runtime = selected_vulkan_device_index
                .map(|index| index == device.physical_device_index)
                .unwrap_or(device.selected_by_default);
            let cpu_runtime_device_id = if device.device_type == "cpu" {
                let runtime_device_id = format!("cpu{cpu_device_ordinal}");
                cpu_device_ordinal += 1;
                Some(runtime_device_id)
            } else {
                None
            };
            let runtime_device_id = selected_by_runtime
                .then(|| default_device_id.to_string())
                .or(cpu_runtime_device_id.clone());
            let device_id = runtime_device_id
                .clone()
                .unwrap_or_else(|| device.physical_device_id.clone());
            RuntimeAvailableDevice {
                device_id,
                backend: "vulkan_compute".to_string(),
                available: true,
                hardware_profile: None,
                runtime_device_id,
                physical_device_id: Some(device.physical_device_id.clone()),
                physical_device_index: Some(device.physical_device_index),
                device_name: Some(device.device_name.clone()),
                device_type: Some(device.device_type.clone()),
                vendor_id: Some(device.vendor_id),
                raw_device_id: Some(device.device_id),
                api_version: Some(device.api_version),
                driver_version: Some(device.driver_version),
                compute_queue_family_indices: Some(device.compute_queue_family_indices.clone()),
                memory_heaps: Some(
                    device
                        .memory_heaps
                        .iter()
                        .map(|heap| RuntimeAvailableMemoryHeap {
                            heap_index: heap.heap_index,
                            size_bytes: heap.size_bytes,
                            device_local: heap.device_local,
                        })
                        .collect(),
                ),
                selected_by_default: Some(device.selected_by_default),
                selected_by_runtime: Some(selected_by_runtime),
                runtime_binding: Some(if selected_by_runtime {
                    "default_local_vulkan_target".to_string()
                } else {
                    "inventory_only".to_string()
                }),
                can_host_runtime_components_on_physical_device: Some(true),
                notes: if selected_by_runtime {
                    vec!["default target for unbound node instances".to_string()]
                } else if let Some(cpu_runtime_device_id) = cpu_runtime_device_id {
                    vec![format!(
                        "CPU runtime target {cpu_runtime_device_id} backed by {}",
                        device.physical_device_id
                    )]
                } else {
                    vec!["available runtime placement target".to_string()]
                },
                error: None,
            }
        })
        .collect()
}

fn unavailable_device(
    device_id: &str,
    note: &str,
    error: Option<String>,
) -> RuntimeAvailableDevice {
    RuntimeAvailableDevice {
        device_id: device_id.to_string(),
        backend: "vulkan_compute".to_string(),
        available: false,
        hardware_profile: None,
        runtime_device_id: None,
        physical_device_id: None,
        physical_device_index: None,
        device_name: None,
        device_type: None,
        vendor_id: None,
        raw_device_id: None,
        api_version: None,
        driver_version: None,
        compute_queue_family_indices: None,
        memory_heaps: None,
        selected_by_default: None,
        selected_by_runtime: None,
        runtime_binding: None,
        can_host_runtime_components_on_physical_device: None,
        notes: vec![note.to_string()],
        error,
    }
}

#[cfg(test)]
mod device_mapping_tests {
    use super::*;

    fn device(
        index: usize,
        kind: &str,
        selected_by_default: bool,
    ) -> VulkanComputeDeviceInfo {
        VulkanComputeDeviceInfo {
            physical_device_index: index,
            physical_device_id: format!("physical:{index}"),
            device_uuid: [index as u8; 16],
            device_name: format!("device {index}"),
            pci_address: None,
            device_type: kind.to_string(),
            vendor_id: 0x1002,
            device_id: index as u32,
            api_version: 1,
            driver_version: 2,
            compute_queue_family_indices: vec![3],
            memory_heaps: vec![crate::VulkanMemoryHeapInfo {
                heap_index: 0,
                size_bytes: 8 * 1024 * 1024 * 1024,
                device_local: true,
            }],
            selected_by_default,
        }
    }

    #[test]
    fn runtime_device_mapping_preserves_physical_identity_and_allocates_cpu_targets() {
        let devices = vec![
            device(2, "discrete_gpu", true),
            device(3, "integrated_gpu", false),
            device(4, "cpu", false),
        ];
        let mapped = runtime_devices_from_compute_devices("runtime_default", None, &devices);
        assert_eq!(
            mapped
                .iter()
                .map(|device| device.device_id.as_str())
                .collect::<Vec<_>>(),
            ["runtime_default", "physical:3", "cpu0"]
        );
        assert_eq!(mapped[0].physical_device_id.as_deref(), Some("physical:2"));
        assert_eq!(mapped[0].runtime_binding.as_deref(), Some("default_local_vulkan_target"));
        assert_eq!(mapped[1].runtime_binding.as_deref(), Some("inventory_only"));
        assert_eq!(mapped[2].runtime_device_id.as_deref(), Some("cpu0"));
        assert_eq!(mapped[2].memory_heaps.as_ref().unwrap()[0].size_bytes, 8 * 1024 * 1024 * 1024);
    }

    #[test]
    fn explicit_physical_selection_overrides_catalog_default_without_losing_inventory() {
        let devices = vec![
            device(2, "discrete_gpu", true),
            device(3, "discrete_gpu", false),
        ];
        let mapped = runtime_devices_from_compute_devices("runtime_default", Some(3), &devices);
        assert_eq!(mapped[0].device_id, "physical:2");
        assert_eq!(mapped[0].selected_by_runtime, Some(false));
        assert_eq!(mapped[1].device_id, "runtime_default");
        assert_eq!(mapped[1].selected_by_runtime, Some(true));
        assert!(mapped.iter().all(|device| device.available));
    }
}
