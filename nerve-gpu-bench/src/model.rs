use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

pub const RUN_SCHEMA: &str = "nerve.gpu_benchmark_run.v1";
pub const TARGET_LIST_SCHEMA: &str = "nerve.gpu_benchmark_targets.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Target {
    pub stable_target_id: String,
    pub backend: String,
    pub kind: String,
    pub name: String,
    pub vendor_id: Option<String>,
    pub vendor_name: Option<String>,
    pub device_id: Option<String>,
    pub pci_address: Option<String>,
    pub physical_location: Option<String>,
    pub numa_node: Option<i64>,
    pub boot_vga: Option<bool>,
    pub capabilities: Vec<String>,
    pub diagnostics: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RunPolicy {
    pub payload_bytes: usize,
    pub samples: usize,
    pub include_targets: Vec<String>,
    pub exclude_targets: Vec<String>,
    pub exclude_pci: Vec<String>,
    pub exclude_kinds: Vec<String>,
    pub pair_measurements: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Selection {
    pub selected_target_ids: Vec<String>,
    pub skipped_targets: Vec<SkippedTarget>,
    pub diagnostics: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SkippedTarget {
    pub stable_target_id: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BenchmarkRun {
    pub schema: String,
    pub started_at_unix_ms: u128,
    pub finished_at_unix_ms: u128,
    pub implementation: Implementation,
    pub policy: RunPolicy,
    pub discovered_targets: Vec<Target>,
    pub selected_target_ids: Vec<String>,
    pub skipped_targets: Vec<SkippedTarget>,
    pub workload_specs: Vec<WorkloadSpec>,
    pub measurements: Vec<Measurement>,
    pub pair_measurements: Vec<PairMeasurement>,
    pub diagnostics: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Implementation {
    pub name: String,
    pub version: String,
    pub backend_status: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WorkloadSpec {
    pub workload_id: String,
    pub pattern: String,
    pub format: String,
    pub participant_count: usize,
    pub payload_bytes: usize,
    pub parameter_bytes_per_participant: usize,
    pub activation_bytes: usize,
    pub output_bytes: usize,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Measurement {
    pub workload_id: String,
    pub target_id: String,
    pub pattern: String,
    pub operation_family: String,
    pub regime: String,
    pub format: String,
    pub status: String,
    pub reason: Option<String>,
    pub payload_bytes: usize,
    pub working_set_bytes: usize,
    pub samples: Vec<Sample>,
    pub summary: Option<Summary>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PairMeasurement {
    pub workload_id: String,
    pub source_target_id: String,
    pub destination_target_id: String,
    pub pattern: String,
    pub operation_family: String,
    pub regime: String,
    pub format: String,
    pub status: String,
    pub reason: Option<String>,
    pub payload_bytes: usize,
    pub source_payload_bytes: usize,
    pub destination_payload_bytes: usize,
    pub activation_bytes: usize,
    pub output_bytes: usize,
    pub samples: Vec<Sample>,
    pub summary: Option<Summary>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Sample {
    pub sample_index: usize,
    pub duration_ns: u128,
    pub iterations: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub operations: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Summary {
    pub min_duration_ns: u128,
    pub median_duration_ns: u128,
    pub bytes_per_second: f64,
    pub operations_per_second: f64,
}

#[derive(Debug, Serialize)]
struct TargetListDocument<'a> {
    schema: &'static str,
    targets: &'a [Target],
}

pub fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

pub fn targets_to_json(targets: &[Target]) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(&TargetListDocument {
        schema: TARGET_LIST_SCHEMA,
        targets,
    })
}

impl BenchmarkRun {
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

impl Implementation {
    pub fn current() -> Self {
        Self {
            name: "nerve-gpu-bench".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            backend_status: "cpu_benchmarks_only_gpu_backend_unmeasured".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_list_serializes_as_json() {
        let targets = [Target {
            stable_target_id: "cpu:host".to_string(),
            backend: "cpu".to_string(),
            kind: "cpu".to_string(),
            name: "Host CPU".to_string(),
            vendor_id: None,
            vendor_name: None,
            device_id: None,
            pci_address: None,
            physical_location: Some("host".to_string()),
            numa_node: None,
            boot_vga: None,
            capabilities: vec!["f32".to_string()],
            diagnostics: Vec::new(),
        }];
        let document = targets_to_json(&targets).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&document).unwrap();
        assert_eq!(parsed["schema"], TARGET_LIST_SCHEMA);
        assert_eq!(parsed["targets"][0]["stable_target_id"], "cpu:host");
    }
}
