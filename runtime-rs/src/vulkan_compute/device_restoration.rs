pub const VULKAN_DEVICE_LOCAL_MEMORY_RESTORATION_SCHEMA: &str =
    "nerve.runtime.device_local_memory_restoration.v1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VulkanDeviceLocalMemoryRestorationSnapshot {
    pub physical_device_id: String,
    pub device_name: String,
    pub pci_address: Option<String>,
    pub api_version: u32,
    pub driver_version: u32,
    pub memory_budget: VulkanDeviceLocalMemoryBudget,
    pub memory_accounting: VulkanDeviceLocalMemoryAccounting,
    pub memory_pressure: VulkanDeviceLocalMemoryPressure,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VulkanDeviceLocalMemoryRestorationDeviceReport {
    pub physical_device_id: String,
    pub restored: bool,
    pub before: Option<VulkanDeviceLocalMemoryRestorationSnapshot>,
    pub after: Option<VulkanDeviceLocalMemoryRestorationSnapshot>,
    pub errors: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VulkanDeviceLocalMemoryRestorationReport {
    pub schema: &'static str,
    pub complete: bool,
    pub physical_device_count: usize,
    pub restored_device_count: usize,
    pub devices: Vec<VulkanDeviceLocalMemoryRestorationDeviceReport>,
    pub errors: Vec<String>,
}

impl VulkanDeviceLocalMemoryRestorationReport {
    fn failed(before: &[VulkanDeviceLocalMemoryRestorationSnapshot], error: String) -> Self {
        Self {
            schema: VULKAN_DEVICE_LOCAL_MEMORY_RESTORATION_SCHEMA,
            complete: false,
            physical_device_count: before.len(),
            restored_device_count: 0,
            devices: Vec::new(),
            errors: vec![error],
        }
    }
}

pub fn capture_vulkan_device_local_memory_restoration_snapshots<'a>(
    devices: impl IntoIterator<Item = &'a VulkanComputeDevice>,
) -> Result<Vec<VulkanDeviceLocalMemoryRestorationSnapshot>, VulkanError> {
    let devices = canonical_unique_vulkan_restoration_devices(devices)?;
    devices
        .into_iter()
        .map(vulkan_device_local_memory_restoration_snapshot)
        .collect()
}

pub fn quiesce_and_verify_vulkan_device_local_memory_restoration<'a>(
    devices: impl IntoIterator<Item = &'a VulkanComputeDevice>,
    before: &[VulkanDeviceLocalMemoryRestorationSnapshot],
) -> VulkanDeviceLocalMemoryRestorationReport {
    let devices = match canonical_unique_vulkan_restoration_devices(devices) {
        Ok(devices) => devices,
        Err(error) => {
            return VulkanDeviceLocalMemoryRestorationReport::failed(before, error.to_string());
        }
    };
    for device in &devices {
        if let Err(error) = device.quiesce() {
            return VulkanDeviceLocalMemoryRestorationReport::failed(
                before,
                format!(
                    "could not quiesce physical device {:?} before restoration proof: {error}",
                    device.physical_device_id(),
                ),
            );
        }
    }
    let after = match devices
        .into_iter()
        .map(vulkan_device_local_memory_restoration_snapshot)
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(after) => after,
        Err(error) => {
            return VulkanDeviceLocalMemoryRestorationReport::failed(
                before,
                format!("could not capture post-workload device state: {error}"),
            );
        }
    };
    verify_vulkan_device_local_memory_restoration(before, &after)
}

pub fn verify_vulkan_device_local_memory_restoration(
    before: &[VulkanDeviceLocalMemoryRestorationSnapshot],
    after: &[VulkanDeviceLocalMemoryRestorationSnapshot],
) -> VulkanDeviceLocalMemoryRestorationReport {
    let mut errors = Vec::new();
    if before.is_empty() {
        errors.push("device restoration proof has no selected physical devices".to_string());
    }
    let before_by_id = restoration_snapshots_by_id("before", before, &mut errors);
    let after_by_id = restoration_snapshots_by_id("after", after, &mut errors);
    let physical_device_ids = before_by_id
        .keys()
        .chain(after_by_id.keys())
        .cloned()
        .collect::<BTreeSet<_>>();
    if before_by_id.keys().collect::<BTreeSet<_>>()
        != after_by_id.keys().collect::<BTreeSet<_>>()
    {
        errors.push("selected physical device set changed during execution".to_string());
    }

    let devices = physical_device_ids
        .into_iter()
        .map(|physical_device_id| {
            let before = before_by_id.get(&physical_device_id).copied();
            let after = after_by_id.get(&physical_device_id).copied();
            let mut device_errors = Vec::new();
            match (before, after) {
                (Some(before), Some(after)) => {
                    verify_vulkan_restoration_device(before, after, &mut device_errors);
                }
                (None, Some(_)) => device_errors.push(
                    "physical device was absent from the pre-workload snapshot".to_string(),
                ),
                (Some(_), None) => device_errors.push(
                    "physical device was absent from the post-workload snapshot".to_string(),
                ),
                (None, None) => unreachable!("device identity came from at least one snapshot"),
            }
            VulkanDeviceLocalMemoryRestorationDeviceReport {
                physical_device_id,
                restored: device_errors.is_empty(),
                before: before.cloned(),
                after: after.cloned(),
                errors: device_errors,
            }
        })
        .collect::<Vec<_>>();
    for device in &devices {
        errors.extend(device.errors.iter().map(|error| {
            format!("physical device {:?}: {error}", device.physical_device_id)
        }));
    }
    let restored_device_count = devices.iter().filter(|device| device.restored).count();
    let complete = errors.is_empty()
        && restored_device_count == before.len()
        && before.len() == after.len()
        && devices.len() == before.len();
    VulkanDeviceLocalMemoryRestorationReport {
        schema: VULKAN_DEVICE_LOCAL_MEMORY_RESTORATION_SCHEMA,
        complete,
        physical_device_count: before.len(),
        restored_device_count,
        devices,
        errors,
    }
}

fn canonical_unique_vulkan_restoration_devices<'a>(
    devices: impl IntoIterator<Item = &'a VulkanComputeDevice>,
) -> Result<Vec<&'a VulkanComputeDevice>, VulkanError> {
    let mut by_id = BTreeMap::<String, &'a VulkanComputeDevice>::new();
    for device in devices {
        let physical_device_id = device.physical_device_id().to_string();
        if let Some(existing) = by_id.get(&physical_device_id) {
            if !std::ptr::eq(*existing, device) {
                return Err(VulkanError(format!(
                    "physical device {physical_device_id:?} is represented by multiple logical Vulkan handles"
                )));
            }
            continue;
        }
        by_id.insert(physical_device_id, device);
    }
    if by_id.is_empty() {
        return Err(VulkanError(
            "device restoration proof has no selected physical devices".to_string(),
        ));
    }
    Ok(by_id.into_values().collect())
}

fn vulkan_device_local_memory_restoration_snapshot(
    device: &VulkanComputeDevice,
) -> Result<VulkanDeviceLocalMemoryRestorationSnapshot, VulkanError> {
    Ok(VulkanDeviceLocalMemoryRestorationSnapshot {
        physical_device_id: device.physical_device_id().to_string(),
        device_name: device.device_name().to_string(),
        pci_address: device.pci_address().map(str::to_string),
        api_version: device.api_version(),
        driver_version: device.driver_version(),
        memory_budget: device.device_local_memory_budget(),
        memory_accounting: device.device_local_memory_accounting()?,
        memory_pressure: device.device_local_memory_pressure()?,
    })
}

fn restoration_snapshots_by_id<'a>(
    stage: &str,
    snapshots: &'a [VulkanDeviceLocalMemoryRestorationSnapshot],
    errors: &mut Vec<String>,
) -> BTreeMap<String, &'a VulkanDeviceLocalMemoryRestorationSnapshot> {
    let mut by_id = BTreeMap::new();
    for snapshot in snapshots {
        if snapshot.physical_device_id.is_empty() {
            errors.push(format!(
                "{stage} device restoration snapshot has an empty physical identity"
            ));
            continue;
        }
        if by_id
            .insert(snapshot.physical_device_id.clone(), snapshot)
            .is_some()
        {
            errors.push(format!(
                "{stage} device restoration snapshot repeats physical device {:?}",
                snapshot.physical_device_id,
            ));
        }
    }
    by_id
}

fn verify_vulkan_restoration_device(
    before: &VulkanDeviceLocalMemoryRestorationSnapshot,
    after: &VulkanDeviceLocalMemoryRestorationSnapshot,
    errors: &mut Vec<String>,
) {
    if before.device_name != after.device_name
        || before.pci_address != after.pci_address
        || before.api_version != after.api_version
        || before.driver_version != after.driver_version
    {
        errors.push("physical device or driver identity changed".to_string());
    }
    if before.memory_budget != after.memory_budget {
        errors.push("device-local memory budget changed".to_string());
    }
    let tolerance = before.memory_budget.counter_tolerance_bytes;
    let accounting = [
        (
            "accounting baseline",
            before.memory_accounting.baseline_available_bytes,
            after.memory_accounting.baseline_available_bytes,
            0,
        ),
        (
            "accounting reservable",
            before.memory_accounting.reservable_bytes,
            after.memory_accounting.reservable_bytes,
            0,
        ),
        (
            "tracked allocation",
            before.memory_accounting.tracked_allocation_bytes,
            after.memory_accounting.tracked_allocation_bytes,
            0,
        ),
        (
            "pending reservation",
            before.memory_accounting.pending_reservation_bytes,
            after.memory_accounting.pending_reservation_bytes,
            0,
        ),
        (
            "untracked acquired",
            before.memory_accounting.untracked_acquired_bytes,
            after.memory_accounting.untracked_acquired_bytes,
            tolerance,
        ),
        (
            "available device-local",
            before.memory_accounting.currently_available_bytes,
            after.memory_accounting.currently_available_bytes,
            tolerance,
        ),
        (
            "remaining reservable",
            before.memory_accounting.remaining_bytes,
            after.memory_accounting.remaining_bytes,
            tolerance,
        ),
        (
            "admissible remaining",
            before.memory_accounting.admissible_remaining_bytes,
            after.memory_accounting.admissible_remaining_bytes,
            tolerance,
        ),
    ];
    for (name, before_bytes, after_bytes, allowed_difference) in accounting {
        if before_bytes.abs_diff(after_bytes) > allowed_difference {
            errors.push(format!(
                "did not restore {name} bytes: before={before_bytes}, after={after_bytes}, tolerance={allowed_difference}"
            ));
        }
    }
    if before.memory_pressure.active != after.memory_pressure.active
        || before.memory_pressure.episode != after.memory_pressure.episode
    {
        errors.push(format!(
            "memory-pressure state changed: active={}->{}, episode={}->{}",
            before.memory_pressure.active,
            after.memory_pressure.active,
            before.memory_pressure.episode,
            after.memory_pressure.episode,
        ));
    }
}
