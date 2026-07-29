use std::time::Instant;

pub(super) fn elapsed_ns(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

pub(super) fn maximum_cpu_temperature_millidegrees() -> Option<u64> {
    let entries = std::fs::read_dir("/sys/class/thermal").ok()?;
    entries
        .filter_map(Result::ok)
        .filter_map(|entry| std::fs::read_to_string(entry.path().join("temp")).ok())
        .filter_map(|value| value.trim().parse::<u64>().ok())
        .max()
}

#[cfg(feature = "vulkan")]
pub(super) fn maximum_pci_temperature_millidegrees(pci_address: Option<&str>) -> Option<u64> {
    let pci_address = pci_address?;
    let entries = std::fs::read_dir(format!("/sys/bus/pci/devices/{pci_address}/hwmon")).ok()?;
    entries
        .filter_map(Result::ok)
        .flat_map(|entry| {
            std::fs::read_dir(entry.path())
                .into_iter()
                .flatten()
                .filter_map(Result::ok)
        })
        .filter(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            name.starts_with("temp") && name.ends_with("_input")
        })
        .filter_map(|entry| std::fs::read_to_string(entry.path()).ok())
        .filter_map(|value| value.trim().parse::<u64>().ok())
        .max()
}
