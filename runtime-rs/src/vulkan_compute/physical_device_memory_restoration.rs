pub const VULKAN_PHYSICAL_DEVICE_MEMORY_RESTORATION_SCHEMA: &str =
    "nerve.runtime.physical_device_memory_restoration.v1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VulkanPhysicalDeviceMemoryObservation {
    pub physical_device_id: String,
    pub device_name: String,
    pub pci_address: Option<String>,
    pub api_version: u32,
    pub driver_version: u32,
    pub heap_index: u32,
    pub physical_heap_bytes: u64,
    pub memory_budget_supported: bool,
    pub budget_bytes: Option<u64>,
    pub usage_bytes: Option<u64>,
    pub available_bytes: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VulkanPhysicalDeviceMemoryRestorationDeviceReport {
    pub physical_device_id: String,
    pub restored: bool,
    pub usage_counter_tolerance_bytes: u64,
    pub before: Option<VulkanPhysicalDeviceMemoryObservation>,
    pub after: Option<VulkanPhysicalDeviceMemoryObservation>,
    pub errors: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VulkanPhysicalDeviceMemoryRestorationReport {
    pub schema: &'static str,
    pub complete: bool,
    pub physical_device_count: usize,
    pub restored_device_count: usize,
    pub devices: Vec<VulkanPhysicalDeviceMemoryRestorationDeviceReport>,
    pub errors: Vec<String>,
}

pub fn capture_vulkan_physical_device_memory_observations(
    catalog: &VulkanComputeDeviceCatalog,
    physical_device_ids: &BTreeSet<String>,
) -> Result<Vec<VulkanPhysicalDeviceMemoryObservation>, VulkanError> {
    if physical_device_ids.is_empty() {
        return Err(VulkanError(
            "physical device restoration requires at least one selected target".to_string(),
        ));
    }
    let info_by_id = catalog
        .available_compute_devices()
        .iter()
        .map(|info| (info.physical_device_id.as_str(), info))
        .collect::<BTreeMap<_, _>>();
    let memory_by_id = catalog
        .device_local_memory_snapshots()?
        .into_iter()
        .map(|snapshot| (snapshot.physical_device_id.clone(), snapshot))
        .collect::<BTreeMap<_, _>>();
    physical_device_ids
        .iter()
        .map(|physical_device_id| {
            let info = info_by_id.get(physical_device_id.as_str()).ok_or_else(|| {
                VulkanError(format!(
                    "selected physical device {physical_device_id:?} disappeared during restoration proof",
                ))
            })?;
            let memory = memory_by_id.get(physical_device_id).ok_or_else(|| {
                VulkanError(format!(
                    "selected physical device {physical_device_id:?} has no memory observation",
                ))
            })?;
            Ok(VulkanPhysicalDeviceMemoryObservation {
                physical_device_id: physical_device_id.clone(),
                device_name: info.device_name.clone(),
                pci_address: info.pci_address.clone(),
                api_version: info.api_version,
                driver_version: info.driver_version,
                heap_index: memory.heap_index,
                physical_heap_bytes: memory.physical_heap_bytes,
                memory_budget_supported: memory.memory_budget_supported,
                budget_bytes: memory.budget_bytes,
                usage_bytes: memory.usage_bytes,
                available_bytes: memory.available_bytes,
            })
        })
        .collect()
}

pub fn verify_vulkan_physical_device_memory_restoration(
    before: &[VulkanPhysicalDeviceMemoryObservation],
    after: &[VulkanPhysicalDeviceMemoryObservation],
) -> VulkanPhysicalDeviceMemoryRestorationReport {
    let before_by_id = before
        .iter()
        .map(|observation| (observation.physical_device_id.as_str(), observation))
        .collect::<BTreeMap<_, _>>();
    let after_by_id = after
        .iter()
        .map(|observation| (observation.physical_device_id.as_str(), observation))
        .collect::<BTreeMap<_, _>>();
    let physical_device_ids = before_by_id
        .keys()
        .chain(after_by_id.keys())
        .copied()
        .collect::<BTreeSet<_>>();
    let mut errors = Vec::new();
    if before.is_empty() {
        errors.push("physical device restoration has no selected targets".to_string());
    }
    if before_by_id.len() != before.len() || after_by_id.len() != after.len() {
        errors.push("physical device restoration observations repeat a target".to_string());
    }
    if before_by_id.keys().collect::<BTreeSet<_>>()
        != after_by_id.keys().collect::<BTreeSet<_>>()
    {
        errors.push("selected physical device set changed during execution".to_string());
    }
    let mut devices = Vec::with_capacity(physical_device_ids.len());
    for physical_device_id in physical_device_ids {
        let before = before_by_id.get(physical_device_id).copied();
        let after = after_by_id.get(physical_device_id).copied();
        let mut device_errors = Vec::new();
        let usage_counter_tolerance_bytes = before
            .map(|observation| {
                vulkan_device_local_memory_budget_from_available_bytes(
                    observation
                        .available_bytes
                        .unwrap_or(observation.physical_heap_bytes),
                )
                .counter_tolerance_bytes
            })
            .unwrap_or(0);
        if let (Some(before), Some(after)) = (before, after) {
            if before.device_name != after.device_name
                || before.pci_address != after.pci_address
                || before.api_version != after.api_version
                || before.driver_version != after.driver_version
                || before.heap_index != after.heap_index
                || before.physical_heap_bytes != after.physical_heap_bytes
                || before.memory_budget_supported != after.memory_budget_supported
            {
                device_errors.push("physical device, driver, or memory heap changed".to_string());
            }
            if !before.memory_budget_supported || !after.memory_budget_supported {
                device_errors.push(
                    "cannot prove physical memory restoration without VK_EXT_memory_budget"
                        .to_string(),
                );
            }
            match (before.usage_bytes, after.usage_bytes) {
                (Some(before_bytes), Some(after_bytes))
                    if before_bytes.abs_diff(after_bytes) > usage_counter_tolerance_bytes =>
                {
                    device_errors.push(format!(
                        "did not restore usage bytes: before={before_bytes}, after={after_bytes}, tolerance={usage_counter_tolerance_bytes}",
                    ));
                }
                (Some(_), Some(_)) => {}
                _ => device_errors.push(
                    "cannot prove restored usage bytes without VK_EXT_memory_budget".to_string(),
                ),
            }
        } else {
            device_errors.push("physical device is absent from one restoration stage".to_string());
        }
        errors.extend(
            device_errors
                .iter()
                .map(|error| format!("physical device {physical_device_id:?}: {error}")),
        );
        devices.push(VulkanPhysicalDeviceMemoryRestorationDeviceReport {
            physical_device_id: physical_device_id.to_string(),
            restored: device_errors.is_empty(),
            usage_counter_tolerance_bytes,
            before: before.cloned(),
            after: after.cloned(),
            errors: device_errors,
        });
    }
    let restored_device_count = devices.iter().filter(|device| device.restored).count();
    VulkanPhysicalDeviceMemoryRestorationReport {
        schema: VULKAN_PHYSICAL_DEVICE_MEMORY_RESTORATION_SCHEMA,
        complete: errors.is_empty() && restored_device_count == devices.len(),
        physical_device_count: devices.len(),
        restored_device_count,
        devices,
        errors,
    }
}
