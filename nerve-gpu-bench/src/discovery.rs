use std::fs;
use std::path::{Path, PathBuf};

use crate::model::{FormatCapability, PciLink, Target};

const MODEL_BENCHMARK_FORMATS: &[&str] = &[
    "f32", "f16", "bf16", "fp8_e4m3", "fp8_e5m2", "fp4", "mxfp4", "nvfp4", "int8", "int4", "q8_0",
    "q6_k", "q5_0", "q5_1", "q5_k", "q4_0", "q4_1", "q4_k", "q3_k", "q2_k", "iq4_nl", "iq4_xs",
    "iq3_s", "iq2_xs",
];

pub fn discover_targets() -> Vec<Target> {
    let mut targets = vec![discover_cpu_target()];
    targets.extend(discover_pci_targets(Path::new("/sys/bus/pci/devices")));
    targets.extend(crate::vulkan_probe::discover_vulkan_targets());
    targets.sort_by(|left, right| left.stable_target_id.cmp(&right.stable_target_id));
    targets
}

fn discover_cpu_target() -> Target {
    let cpu_count = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    let name = read_cpu_model_name().unwrap_or_else(|| "Host CPU".to_string());
    Target {
        stable_target_id: "cpu:host".to_string(),
        backend: "cpu".to_string(),
        kind: "cpu".to_string(),
        name,
        vendor_id: None,
        vendor_name: None,
        device_id: None,
        pci_address: None,
        physical_location: Some("host".to_string()),
        numa_node: None,
        boot_vga: None,
        pci_link: None,
        vulkan: None,
        capabilities: vec![
            format!("logical_cpus={cpu_count}"),
            "f32".to_string(),
            "u8_copy".to_string(),
        ],
        format_capabilities: cpu_format_capabilities(),
        diagnostics: Vec::new(),
    }
}

fn read_cpu_model_name() -> Option<String> {
    let cpuinfo = fs::read_to_string("/proc/cpuinfo").ok()?;
    cpuinfo
        .lines()
        .find_map(|line| line.strip_prefix("model name"))
        .and_then(|suffix| suffix.split_once(':').map(|(_, value)| value.trim()))
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn cpu_format_capabilities() -> Vec<FormatCapability> {
    let mut capabilities = vec![
        format_capability(
            "u8",
            "native",
            "cpu_baseline",
            "byte movement benchmark path",
        ),
        format_capability(
            "f32",
            "native",
            "cpu_baseline",
            "scalar/vector CPU reference path",
        ),
    ];
    capabilities.extend(
        MODEL_BENCHMARK_FORMATS
            .iter()
            .filter(|format| **format != "f32")
            .map(|format| {
                format_capability(
                    format,
                    "unmeasured",
                    "not_probed",
                    "requires explicit CPU feature/backend probe",
                )
            }),
    );
    capabilities
}

fn discover_pci_targets(root: &Path) -> Vec<Target> {
    let Ok(entries) = fs::read_dir(root) else {
        return Vec::new();
    };
    let mut targets = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(address) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some(class) = read_trimmed(path.join("class")) else {
            continue;
        };
        if !is_accelerator_or_gpu(&class) {
            continue;
        }
        targets.push(pci_target(address, &path, &class));
    }
    targets
}

fn pci_target(address: &str, path: &Path, class: &str) -> Target {
    let vendor_id = read_trimmed(path.join("vendor"));
    let device_id = read_trimmed(path.join("device"));
    let vendor_name = vendor_id.as_deref().map(vendor_name).map(str::to_string);
    let boot_vga = read_trimmed(path.join("boot_vga")).and_then(|value| match value.as_str() {
        "0" => Some(false),
        "1" => Some(true),
        _ => None,
    });
    let numa_node = read_trimmed(path.join("numa_node"))
        .and_then(|value| value.parse::<i64>().ok())
        .filter(|node| *node >= 0);

    let current_link_speed = read_trimmed(path.join("current_link_speed"));
    let current_link_width = read_trimmed(path.join("current_link_width"));
    let max_link_speed = read_trimmed(path.join("max_link_speed"));
    let max_link_width = read_trimmed(path.join("max_link_width"));
    let kind = classify_target_kind(class, vendor_id.as_deref(), current_link_width.as_deref());
    let pci_link = pci_link(
        current_link_speed.clone(),
        current_link_width.clone(),
        max_link_speed.clone(),
        max_link_width.clone(),
    );
    let mut capabilities = Vec::new();
    capabilities.push(format!("pci_class={class}"));
    if let Some(speed) = &current_link_speed {
        capabilities.push(format!("current_link_speed={speed}"));
    }
    if let Some(width) = &current_link_width {
        capabilities.push(format!("current_link_width={width}"));
    }
    if let Some(max_speed) = &max_link_speed {
        capabilities.push(format!("max_link_speed={max_speed}"));
    }
    if let Some(max_width) = &max_link_width {
        capabilities.push(format!("max_link_width={max_width}"));
    }
    if let Some(true) = boot_vga {
        capabilities.push("boot_vga".to_string());
    }

    Target {
        stable_target_id: format!("pci:{address}"),
        backend: "pci".to_string(),
        kind,
        name: pci_name(address, vendor_id.as_deref(), device_id.as_deref()),
        vendor_id,
        vendor_name,
        device_id,
        pci_address: Some(address.to_string()),
        physical_location: Some(format!("pci:{address}")),
        numa_node,
        boot_vga,
        pci_link,
        vulkan: None,
        capabilities,
        format_capabilities: passive_pci_format_capabilities(),
        diagnostics: Vec::new(),
    }
}

fn pci_link(
    current_link_speed: Option<String>,
    current_link_width: Option<String>,
    max_link_speed: Option<String>,
    max_link_width: Option<String>,
) -> Option<PciLink> {
    let current_width = current_link_width.as_deref().and_then(parse_link_width);
    let max_width = max_link_width.as_deref().and_then(parse_link_width);
    let current_one_way_bytes_per_second = current_link_speed
        .as_deref()
        .and_then(parse_link_speed_gtps)
        .zip(current_width)
        .map(|(gtps, width)| estimate_pcie_one_way_bytes_per_second(gtps, width));
    let max_one_way_bytes_per_second = max_link_speed
        .as_deref()
        .and_then(parse_link_speed_gtps)
        .zip(max_width)
        .map(|(gtps, width)| estimate_pcie_one_way_bytes_per_second(gtps, width));
    if current_link_speed.is_none()
        && current_width.is_none()
        && max_link_speed.is_none()
        && max_width.is_none()
    {
        return None;
    }
    Some(PciLink {
        current_link_speed,
        current_link_width: current_width,
        current_one_way_bytes_per_second,
        max_link_speed,
        max_link_width: max_width,
        max_one_way_bytes_per_second,
        notes: vec![
            "passive_sysfs_estimate_not_measured_peer_bandwidth".to_string(),
            "one_way_bytes_per_second_uses_pcie_encoding_efficiency".to_string(),
        ],
    })
}

fn passive_pci_format_capabilities() -> Vec<FormatCapability> {
    MODEL_BENCHMARK_FORMATS
        .iter()
        .map(|format| {
            format_capability(
                format,
                "unmeasured",
                "passive_pci_discovery",
                "backend has not probed this target",
            )
        })
        .collect()
}

fn format_capability(format: &str, support: &str, source: &str, notes: &str) -> FormatCapability {
    FormatCapability {
        format: format.to_string(),
        support: support.to_string(),
        source: source.to_string(),
        notes: notes.to_string(),
    }
}

fn read_trimmed(path: PathBuf) -> Option<String> {
    fs::read_to_string(path)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn parse_link_width(value: &str) -> Option<u32> {
    value.trim().parse::<u32>().ok().filter(|width| *width > 0)
}

fn parse_link_speed_gtps(value: &str) -> Option<f64> {
    let number = value.split_whitespace().next()?;
    number.parse::<f64>().ok().filter(|speed| *speed > 0.0)
}

fn estimate_pcie_one_way_bytes_per_second(gtps: f64, width: u32) -> u64 {
    let encoding_efficiency = if gtps <= 5.0 { 0.8 } else { 128.0 / 130.0 };
    ((gtps * 1_000_000_000.0 * f64::from(width) * encoding_efficiency) / 8.0).round() as u64
}

fn is_accelerator_or_gpu(class: &str) -> bool {
    let class = class.trim_start_matches("0x");
    if class.len() < 2 {
        return false;
    }
    matches!(&class[0..2], "03" | "12")
}

fn classify_target_kind(
    class: &str,
    vendor_id: Option<&str>,
    current_link_width: Option<&str>,
) -> String {
    let class = class.trim_start_matches("0x");
    if class.starts_with("12") {
        return "accelerator".to_string();
    }
    if current_link_width.is_some() {
        "discrete_gpu".to_string()
    } else if vendor_id == Some("0x8086") {
        "integrated_gpu".to_string()
    } else {
        "gpu".to_string()
    }
}

fn vendor_name(vendor_id: &str) -> &'static str {
    match vendor_id {
        "0x1002" => "AMD",
        "0x10de" => "NVIDIA",
        "0x8086" => "Intel",
        _ => "unknown",
    }
}

fn pci_name(address: &str, vendor_id: Option<&str>, device_id: Option<&str>) -> String {
    match (vendor_id, device_id) {
        (Some(vendor), Some(device)) => format!("PCI accelerator {vendor}:{device} at {address}"),
        _ => format!("PCI accelerator at {address}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_display_and_accelerator_classes() {
        assert!(is_accelerator_or_gpu("0x030000"));
        assert!(is_accelerator_or_gpu("0x120000"));
        assert!(!is_accelerator_or_gpu("0x020000"));
    }

    #[test]
    fn does_not_encode_vendor_bans() {
        assert_eq!(
            classify_target_kind("0x030000", Some("0x10de"), Some("4")),
            "discrete_gpu"
        );
        assert_eq!(
            classify_target_kind("0x030000", Some("0x8086"), None),
            "integrated_gpu"
        );
        assert_eq!(
            classify_target_kind("0x030000", Some("0x1002"), None),
            "gpu"
        );
    }

    #[test]
    fn parses_pcie_link_bandwidth_estimates() {
        assert_eq!(parse_link_width("4"), Some(4));
        assert_eq!(parse_link_speed_gtps("16.0 GT/s PCIe"), Some(16.0));
        assert_eq!(
            estimate_pcie_one_way_bytes_per_second(16.0, 4),
            7_876_923_077
        );
        assert_eq!(
            estimate_pcie_one_way_bytes_per_second(5.0, 4),
            2_000_000_000
        );
    }
}
