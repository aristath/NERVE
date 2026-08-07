use std::time::{SystemTime, UNIX_EPOCH};

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

pub const RUN_SCHEMA: &str = "nerve.gpu_benchmark_run.v1";
pub const TARGET_LIST_SCHEMA: &str = "nerve.gpu_benchmark_targets.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunPolicy {
    pub payload_bytes: usize,
    pub samples: usize,
    pub include_targets: Vec<String>,
    pub exclude_targets: Vec<String>,
    pub exclude_pci: Vec<String>,
    pub exclude_kinds: Vec<String>,
    pub pair_measurements: bool,
    pub max_group_size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Selection {
    pub selected_target_ids: Vec<String>,
    pub skipped_targets: Vec<SkippedTarget>,
    pub diagnostics: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkippedTarget {
    pub stable_target_id: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    pub group_measurements: Vec<GroupMeasurement>,
    pub diagnostics: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Implementation {
    pub name: String,
    pub version: String,
    pub backend_status: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkloadSpec {
    pub workload_id: String,
    pub comparison_group: String,
    pub placement_strategy: String,
    pub pattern: String,
    pub format: String,
    pub participant_count: usize,
    pub payload_bytes: usize,
    pub parameter_bytes_per_participant: usize,
    pub activation_bytes: usize,
    pub output_bytes: usize,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Measurement {
    pub workload_id: String,
    pub comparison_group: String,
    pub placement_strategy: String,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairMeasurement {
    pub workload_id: String,
    pub comparison_group: String,
    pub placement_strategy: String,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroupMeasurement {
    pub workload_id: String,
    pub comparison_group: String,
    pub placement_strategy: String,
    pub target_ids: Vec<String>,
    pub pattern: String,
    pub operation_family: String,
    pub regime: String,
    pub format: String,
    pub status: String,
    pub reason: Option<String>,
    pub participant_count: usize,
    pub payload_bytes: usize,
    pub payload_bytes_per_participant: Vec<usize>,
    pub activation_bytes: usize,
    pub output_bytes: usize,
    pub samples: Vec<Sample>,
    pub summary: Option<Summary>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sample {
    pub sample_index: usize,
    pub duration_ns: u128,
    pub iterations: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub operations: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Summary {
    pub min_duration_ns: u128,
    pub median_duration_ns: u128,
    pub bytes_per_second: f64,
    pub operations_per_second: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchmarkRunSummary {
    pub discovered_target_count: usize,
    pub selected_target_count: usize,
    pub skipped_target_count: usize,
    pub single_measurement_count: usize,
    pub pair_measurement_count: usize,
    pub group_measurement_count: usize,
    pub completed_count: usize,
    pub unmeasured_count: usize,
    pub failed_count: usize,
    pub unsupported_count: usize,
    pub skipped_count: usize,
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

    pub fn validate_basic(&self) -> Result<(), String> {
        if self.schema != RUN_SCHEMA {
            return Err(format!(
                "unsupported benchmark run schema {:?}",
                self.schema
            ));
        }
        if self.finished_at_unix_ms < self.started_at_unix_ms {
            return Err("finished_at_unix_ms is before started_at_unix_ms".to_string());
        }
        if self.policy.payload_bytes == 0 {
            return Err("policy.payload_bytes must be greater than zero".to_string());
        }
        if self.policy.samples == 0 {
            return Err("policy.samples must be greater than zero".to_string());
        }
        if self.policy.max_group_size == 0 || self.policy.max_group_size > 3 {
            return Err("policy.max_group_size must be between 1 and 3".to_string());
        }

        let discovered_ids = strictly_unique_target_ids(&self.discovered_targets)?;
        for selected_id in &self.selected_target_ids {
            if !discovered_ids.contains(selected_id) {
                return Err(format!(
                    "selected target {selected_id:?} is not in discovered_targets"
                ));
            }
        }
        for skipped in &self.skipped_targets {
            if !discovered_ids.contains(&skipped.stable_target_id) {
                return Err(format!(
                    "skipped target {:?} is not in discovered_targets",
                    skipped.stable_target_id
                ));
            }
            if skipped.reason.is_empty() {
                return Err(format!(
                    "skipped target {:?} has an empty reason",
                    skipped.stable_target_id
                ));
            }
        }
        let workload_ids = self
            .workload_specs
            .iter()
            .map(|spec| spec.workload_id.as_str())
            .collect::<BTreeSet<_>>();
        if workload_ids.len() != self.workload_specs.len() {
            return Err("workload_specs contain duplicate workload IDs".to_string());
        }
        for spec in &self.workload_specs {
            if spec.comparison_group.is_empty() {
                return Err(format!(
                    "workload spec {:?} has an empty comparison_group",
                    spec.workload_id
                ));
            }
            if spec.placement_strategy.is_empty() {
                return Err(format!(
                    "workload spec {:?} has an empty placement_strategy",
                    spec.workload_id
                ));
            }
            if spec.payload_bytes != self.policy.payload_bytes {
                return Err(format!(
                    "workload spec {:?} payload differs from policy",
                    spec.workload_id
                ));
            }
            if spec.participant_count == 0 || spec.participant_count > 3 {
                return Err(format!(
                    "workload spec {:?} has invalid participant_count",
                    spec.workload_id
                ));
            }
        }
        for measurement in &self.measurements {
            if measurement.comparison_group.is_empty() || measurement.placement_strategy.is_empty()
            {
                return Err(format!(
                    "measurement {:?} has empty comparison metadata",
                    measurement.workload_id
                ));
            }
            if !discovered_ids.contains(&measurement.target_id) {
                return Err(format!(
                    "measurement {:?} references unknown target {:?}",
                    measurement.workload_id, measurement.target_id
                ));
            }
            validate_status(&measurement.status)?;
        }
        for measurement in &self.pair_measurements {
            if measurement.comparison_group.is_empty() || measurement.placement_strategy.is_empty()
            {
                return Err(format!(
                    "pair measurement {:?} has empty comparison metadata",
                    measurement.workload_id
                ));
            }
            if !discovered_ids.contains(&measurement.source_target_id) {
                return Err(format!(
                    "pair measurement {:?} references unknown source {:?}",
                    measurement.workload_id, measurement.source_target_id
                ));
            }
            if !discovered_ids.contains(&measurement.destination_target_id) {
                return Err(format!(
                    "pair measurement {:?} references unknown destination {:?}",
                    measurement.workload_id, measurement.destination_target_id
                ));
            }
            if measurement.source_target_id == measurement.destination_target_id {
                return Err(format!(
                    "pair measurement {:?} uses the same source and destination",
                    measurement.workload_id
                ));
            }
            validate_status(&measurement.status)?;
        }
        for measurement in &self.group_measurements {
            if measurement.comparison_group.is_empty() || measurement.placement_strategy.is_empty()
            {
                return Err(format!(
                    "group measurement {:?} has empty comparison metadata",
                    measurement.workload_id
                ));
            }
            if measurement.participant_count != measurement.target_ids.len() {
                return Err(format!(
                    "group measurement {:?} participant_count does not match target_ids",
                    measurement.workload_id
                ));
            }
            if measurement.payload_bytes_per_participant.len() != measurement.target_ids.len() {
                return Err(format!(
                    "group measurement {:?} payload split does not match target_ids",
                    measurement.workload_id
                ));
            }
            let unique = measurement.target_ids.iter().collect::<BTreeSet<_>>();
            if unique.len() != measurement.target_ids.len() {
                return Err(format!(
                    "group measurement {:?} contains duplicate target IDs",
                    measurement.workload_id
                ));
            }
            for target_id in &measurement.target_ids {
                if !discovered_ids.contains(target_id) {
                    return Err(format!(
                        "group measurement {:?} references unknown target {:?}",
                        measurement.workload_id, target_id
                    ));
                }
            }
            validate_status(&measurement.status)?;
        }
        Ok(())
    }

    pub fn summary(&self) -> BenchmarkRunSummary {
        let mut summary = BenchmarkRunSummary {
            discovered_target_count: self.discovered_targets.len(),
            selected_target_count: self.selected_target_ids.len(),
            skipped_target_count: self.skipped_targets.len(),
            single_measurement_count: self.measurements.len(),
            pair_measurement_count: self.pair_measurements.len(),
            group_measurement_count: self.group_measurements.len(),
            completed_count: 0,
            unmeasured_count: 0,
            failed_count: 0,
            unsupported_count: 0,
            skipped_count: 0,
        };
        for status in self
            .measurements
            .iter()
            .map(|measurement| measurement.status.as_str())
            .chain(
                self.pair_measurements
                    .iter()
                    .map(|measurement| measurement.status.as_str()),
            )
            .chain(
                self.group_measurements
                    .iter()
                    .map(|measurement| measurement.status.as_str()),
            )
        {
            match status {
                "completed" => summary.completed_count += 1,
                "unmeasured" => summary.unmeasured_count += 1,
                "failed" => summary.failed_count += 1,
                "unsupported" => summary.unsupported_count += 1,
                "skipped" => summary.skipped_count += 1,
                _ => {}
            }
        }
        summary
    }
}

pub fn parse_benchmark_run_json(input: &str) -> Result<BenchmarkRun, serde_json::Error> {
    serde_json::from_str(input)
}

fn strictly_unique_target_ids(targets: &[Target]) -> Result<BTreeSet<&String>, String> {
    let mut ids = BTreeSet::new();
    for target in targets {
        if target.stable_target_id.is_empty() {
            return Err("discovered target has an empty stable_target_id".to_string());
        }
        if !ids.insert(&target.stable_target_id) {
            return Err(format!(
                "duplicate discovered target ID {:?}",
                target.stable_target_id
            ));
        }
    }
    Ok(ids)
}

fn validate_status(status: &str) -> Result<(), String> {
    match status {
        "completed" | "unmeasured" | "failed" | "unsupported" | "skipped" => Ok(()),
        other => Err(format!("unsupported measurement status {other:?}")),
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

    #[test]
    fn validates_minimal_run_document() {
        let run = BenchmarkRun {
            schema: RUN_SCHEMA.to_string(),
            started_at_unix_ms: 1,
            finished_at_unix_ms: 2,
            implementation: Implementation::current(),
            policy: RunPolicy {
                payload_bytes: 1024,
                samples: 1,
                include_targets: Vec::new(),
                exclude_targets: Vec::new(),
                exclude_pci: Vec::new(),
                exclude_kinds: Vec::new(),
                pair_measurements: false,
                max_group_size: 1,
            },
            discovered_targets: vec![Target {
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
                capabilities: Vec::new(),
                diagnostics: Vec::new(),
            }],
            selected_target_ids: vec!["cpu:host".to_string()],
            skipped_targets: Vec::new(),
            workload_specs: Vec::new(),
            measurements: Vec::new(),
            pair_measurements: Vec::new(),
            group_measurements: Vec::new(),
            diagnostics: Vec::new(),
        };
        let encoded = run.to_json_pretty().unwrap();
        let parsed = parse_benchmark_run_json(&encoded).unwrap();
        parsed.validate_basic().unwrap();
    }

    #[test]
    fn summarizes_run_status_counts() {
        let mut run = BenchmarkRun {
            schema: RUN_SCHEMA.to_string(),
            started_at_unix_ms: 1,
            finished_at_unix_ms: 2,
            implementation: Implementation::current(),
            policy: RunPolicy {
                payload_bytes: 1024,
                samples: 1,
                include_targets: Vec::new(),
                exclude_targets: Vec::new(),
                exclude_pci: Vec::new(),
                exclude_kinds: Vec::new(),
                pair_measurements: false,
                max_group_size: 1,
            },
            discovered_targets: vec![Target {
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
                capabilities: Vec::new(),
                diagnostics: Vec::new(),
            }],
            selected_target_ids: vec!["cpu:host".to_string()],
            skipped_targets: Vec::new(),
            workload_specs: Vec::new(),
            measurements: Vec::new(),
            pair_measurements: Vec::new(),
            group_measurements: Vec::new(),
            diagnostics: Vec::new(),
        };
        run.measurements.push(Measurement {
            workload_id: "work".to_string(),
            comparison_group: "test".to_string(),
            placement_strategy: "single_target_serial".to_string(),
            target_id: "cpu:host".to_string(),
            pattern: "single".to_string(),
            operation_family: "test".to_string(),
            regime: "small_payload".to_string(),
            format: "u8".to_string(),
            status: "completed".to_string(),
            reason: None,
            payload_bytes: 1024,
            working_set_bytes: 1024,
            samples: Vec::new(),
            summary: None,
        });
        let summary = run.summary();
        assert_eq!(summary.discovered_target_count, 1);
        assert_eq!(summary.completed_count, 1);
        assert_eq!(summary.unmeasured_count, 0);
    }
}
