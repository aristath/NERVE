use std::time::{SystemTime, UNIX_EPOCH};

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

pub const RUN_SCHEMA: &str = "nerve.gpu_benchmark_run.v1";
pub const PLAN_SCHEMA: &str = "nerve.gpu_benchmark_plan.v1";
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
    pub pci_link: Option<PciLink>,
    pub vulkan: Option<VulkanDeviceInfo>,
    pub capabilities: Vec<String>,
    pub format_capabilities: Vec<FormatCapability>,
    pub diagnostics: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PciLink {
    pub current_link_speed: Option<String>,
    pub current_link_width: Option<u32>,
    pub current_one_way_bytes_per_second: Option<u64>,
    pub max_link_speed: Option<String>,
    pub max_link_width: Option<u32>,
    pub max_one_way_bytes_per_second: Option<u64>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanDeviceInfo {
    pub physical_device_index: usize,
    pub device_name: String,
    pub device_type: String,
    pub api_version: String,
    pub driver_version: u32,
    pub vendor_id: String,
    pub device_id: String,
    pub memory_heaps: Vec<VulkanMemoryHeap>,
    pub queue_families: Vec<VulkanQueueFamily>,
    pub extension_names: Vec<String>,
    pub feature_flags: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanMemoryHeap {
    pub heap_index: u32,
    pub size_bytes: u64,
    pub flags: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanQueueFamily {
    pub family_index: u32,
    pub queue_count: u32,
    pub flags: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FormatCapability {
    pub format: String,
    pub support: String,
    pub source: String,
    pub notes: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunPolicy {
    pub payload_bytes: usize,
    pub samples: usize,
    pub benchmark_formats: Vec<String>,
    pub benchmark_workloads: Vec<String>,
    pub include_targets: Vec<String>,
    pub exclude_targets: Vec<String>,
    pub exclude_pci: Vec<String>,
    pub exclude_kinds: Vec<String>,
    pub pair_measurements: bool,
    pub max_group_size: usize,
    pub execute: bool,
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
    pub comparison_sets: Vec<ComparisonSet>,
    pub measurements: Vec<Measurement>,
    pub pair_measurements: Vec<PairMeasurement>,
    pub group_measurements: Vec<GroupMeasurement>,
    pub diagnostics: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkPlan {
    pub schema: String,
    pub created_at_unix_ms: u128,
    pub policy: RunPolicy,
    pub discovered_target_count: usize,
    pub selected_target_ids: Vec<String>,
    pub skipped_targets: Vec<SkippedTarget>,
    pub requested_format_count: usize,
    pub requested_workload_count: usize,
    pub estimated_single_measurement_count: usize,
    pub estimated_pair_measurement_count: usize,
    pub estimated_group_measurement_count: usize,
    pub estimated_comparison_set_count: usize,
    pub estimated_measurement_count: usize,
    pub max_payload_bytes_per_measurement: usize,
    pub diagnostics: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComparisonSet {
    pub comparison_id: String,
    pub comparison_group: String,
    pub workload_class: String,
    pub regime: String,
    pub format: String,
    pub target_ids: Vec<String>,
    pub candidates: Vec<ComparisonCandidate>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComparisonCandidate {
    pub candidate_id: String,
    pub placement_strategy: String,
    pub measurement_kind: String,
    pub workload_id: String,
    pub target_ids: Vec<String>,
    pub notes: String,
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
    pub workload_class: String,
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
    pub workload_class: String,
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
    pub workload_class: String,
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
    pub workload_class: String,
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
    pub comparison_set_count: usize,
    pub single_measurement_count: usize,
    pub pair_measurement_count: usize,
    pub group_measurement_count: usize,
    pub completed_count: usize,
    pub unmeasured_count: usize,
    pub failed_count: usize,
    pub unsupported_count: usize,
    pub skipped_count: usize,
    pub strategy_statuses: Vec<StrategyStatusSummary>,
    pub candidate_statuses: Vec<ComparisonCandidateStatus>,
    pub coverage_warnings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrategyStatusSummary {
    pub comparison_group: String,
    pub workload_class: String,
    pub placement_strategy: String,
    pub format: String,
    pub completed_count: usize,
    pub unmeasured_count: usize,
    pub failed_count: usize,
    pub unsupported_count: usize,
    pub skipped_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComparisonCandidateStatus {
    pub comparison_id: String,
    pub candidate_id: String,
    pub workload_class: String,
    pub placement_strategy: String,
    pub measurement_kind: String,
    pub format: String,
    pub status: String,
    pub matched_measurement_count: usize,
    pub best_min_duration_ns: Option<u128>,
    pub best_median_duration_ns: Option<u128>,
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
        if self.policy.benchmark_formats.is_empty()
            || self.policy.benchmark_formats.iter().any(String::is_empty)
        {
            return Err("policy.benchmark_formats must contain non-empty formats".to_string());
        }
        if self.policy.benchmark_workloads.is_empty()
            || self.policy.benchmark_workloads.iter().any(String::is_empty)
        {
            return Err("policy.benchmark_workloads must contain non-empty workloads".to_string());
        }
        if self.policy.max_group_size == 0 || self.policy.max_group_size > 3 {
            return Err("policy.max_group_size must be between 1 and 3".to_string());
        }

        let discovered_ids = strictly_unique_target_ids(&self.discovered_targets)?;
        for target in &self.discovered_targets {
            validate_format_capabilities(target)?;
        }
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
            if spec.comparison_group.is_empty() || spec.workload_class.is_empty() {
                return Err(format!(
                    "workload spec {:?} has empty comparison metadata",
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
        let comparison_ids = self
            .comparison_sets
            .iter()
            .map(|comparison| comparison.comparison_id.as_str())
            .collect::<BTreeSet<_>>();
        if comparison_ids.len() != self.comparison_sets.len() {
            return Err("comparison_sets contain duplicate comparison IDs".to_string());
        }
        for comparison in &self.comparison_sets {
            if comparison.comparison_group.is_empty()
                || comparison.workload_class.is_empty()
                || comparison.candidates.is_empty()
            {
                return Err(format!(
                    "comparison set {:?} is missing group or candidates",
                    comparison.comparison_id
                ));
            }
            for target_id in &comparison.target_ids {
                if !discovered_ids.contains(target_id) {
                    return Err(format!(
                        "comparison set {:?} references unknown target {:?}",
                        comparison.comparison_id, target_id
                    ));
                }
            }
            let candidate_ids = comparison
                .candidates
                .iter()
                .map(|candidate| candidate.candidate_id.as_str())
                .collect::<BTreeSet<_>>();
            if candidate_ids.len() != comparison.candidates.len() {
                return Err(format!(
                    "comparison set {:?} contains duplicate candidate IDs",
                    comparison.comparison_id
                ));
            }
            for candidate in &comparison.candidates {
                if candidate.placement_strategy.is_empty() || candidate.workload_id.is_empty() {
                    return Err(format!(
                        "comparison candidate {:?} is missing strategy or workload",
                        candidate.candidate_id
                    ));
                }
                validate_candidate_measurement_kind(candidate)?;
                if !workload_ids.contains(candidate.workload_id.as_str()) {
                    return Err(format!(
                        "comparison candidate {:?} references unknown workload {:?}",
                        candidate.candidate_id, candidate.workload_id
                    ));
                }
                for target_id in &candidate.target_ids {
                    if !discovered_ids.contains(target_id) {
                        return Err(format!(
                            "comparison candidate {:?} references unknown target {:?}",
                            candidate.candidate_id, target_id
                        ));
                    }
                }
            }
        }
        for measurement in &self.measurements {
            if measurement.comparison_group.is_empty()
                || measurement.workload_class.is_empty()
                || measurement.placement_strategy.is_empty()
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
            if measurement.comparison_group.is_empty()
                || measurement.workload_class.is_empty()
                || measurement.placement_strategy.is_empty()
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
            if measurement.comparison_group.is_empty()
                || measurement.workload_class.is_empty()
                || measurement.placement_strategy.is_empty()
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
            comparison_set_count: self.comparison_sets.len(),
            single_measurement_count: self.measurements.len(),
            pair_measurement_count: self.pair_measurements.len(),
            group_measurement_count: self.group_measurements.len(),
            completed_count: 0,
            unmeasured_count: 0,
            failed_count: 0,
            unsupported_count: 0,
            skipped_count: 0,
            strategy_statuses: Vec::new(),
            candidate_statuses: Vec::new(),
            coverage_warnings: Vec::new(),
        };
        let mut strategy_statuses =
            BTreeMap::<(String, String, String, String), StrategyStatusSummary>::new();
        for measurement in &self.measurements {
            increment_summary_status(&mut summary, &measurement.status);
            increment_strategy_status(
                &mut strategy_statuses,
                &measurement.comparison_group,
                &measurement.workload_class,
                &measurement.placement_strategy,
                &measurement.format,
                &measurement.status,
            );
        }
        for measurement in &self.pair_measurements {
            increment_summary_status(&mut summary, &measurement.status);
            increment_strategy_status(
                &mut strategy_statuses,
                &measurement.comparison_group,
                &measurement.workload_class,
                &measurement.placement_strategy,
                &measurement.format,
                &measurement.status,
            );
        }
        for measurement in &self.group_measurements {
            increment_summary_status(&mut summary, &measurement.status);
            increment_strategy_status(
                &mut strategy_statuses,
                &measurement.comparison_group,
                &measurement.workload_class,
                &measurement.placement_strategy,
                &measurement.format,
                &measurement.status,
            );
        }
        let candidate_statuses = self.comparison_candidate_statuses();
        summary.coverage_warnings =
            coverage_warnings_for_run(self, &strategy_statuses, &candidate_statuses);
        summary.strategy_statuses = strategy_statuses.into_values().collect();
        summary.candidate_statuses = candidate_statuses;
        summary
    }

    fn comparison_candidate_statuses(&self) -> Vec<ComparisonCandidateStatus> {
        let mut statuses = Vec::new();
        for comparison in &self.comparison_sets {
            for candidate in &comparison.candidates {
                let matched_measurements = self.matched_candidate_measurements(candidate);
                let matched_statuses = matched_measurements
                    .iter()
                    .map(|measurement| measurement.status)
                    .collect::<Vec<_>>();
                let best_summary = matched_measurements
                    .iter()
                    .filter(|measurement| measurement.status == "completed")
                    .filter_map(|measurement| measurement.summary)
                    .min_by_key(|summary| summary.median_duration_ns);
                statuses.push(ComparisonCandidateStatus {
                    comparison_id: comparison.comparison_id.clone(),
                    candidate_id: candidate.candidate_id.clone(),
                    workload_class: comparison.workload_class.clone(),
                    placement_strategy: candidate.placement_strategy.clone(),
                    measurement_kind: candidate.measurement_kind.clone(),
                    format: comparison.format.clone(),
                    status: combined_candidate_status(&matched_statuses),
                    matched_measurement_count: matched_statuses.len(),
                    best_min_duration_ns: best_summary.map(|summary| summary.min_duration_ns),
                    best_median_duration_ns: best_summary.map(|summary| summary.median_duration_ns),
                });
            }
        }
        statuses
    }

    fn matched_candidate_measurements<'a>(
        &'a self,
        candidate: &ComparisonCandidate,
    ) -> Vec<CandidateMeasurementMatch<'a>> {
        match candidate.measurement_kind.as_str() {
            "single" => self
                .measurements
                .iter()
                .filter(|measurement| {
                    candidate.target_ids.len() == 1
                        && measurement.workload_id == candidate.workload_id
                        && measurement.placement_strategy == candidate.placement_strategy
                        && measurement.target_id == candidate.target_ids[0]
                })
                .map(|measurement| CandidateMeasurementMatch {
                    status: measurement.status.as_str(),
                    summary: measurement.summary.as_ref(),
                })
                .collect(),
            "pair" => self
                .pair_measurements
                .iter()
                .filter(|measurement| {
                    candidate.target_ids.len() == 2
                        && measurement.workload_id == candidate.workload_id
                        && measurement.placement_strategy == candidate.placement_strategy
                        && measurement.source_target_id == candidate.target_ids[0]
                        && measurement.destination_target_id == candidate.target_ids[1]
                })
                .map(|measurement| CandidateMeasurementMatch {
                    status: measurement.status.as_str(),
                    summary: measurement.summary.as_ref(),
                })
                .collect(),
            "group" => self
                .group_measurements
                .iter()
                .filter(|measurement| {
                    measurement.workload_id == candidate.workload_id
                        && measurement.placement_strategy == candidate.placement_strategy
                        && measurement.target_ids == candidate.target_ids
                })
                .map(|measurement| CandidateMeasurementMatch {
                    status: measurement.status.as_str(),
                    summary: measurement.summary.as_ref(),
                })
                .collect(),
            _ => Vec::new(),
        }
    }
}

struct CandidateMeasurementMatch<'a> {
    status: &'a str,
    summary: Option<&'a Summary>,
}

impl BenchmarkPlan {
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn validate_basic(&self) -> Result<(), String> {
        if self.schema != PLAN_SCHEMA {
            return Err(format!(
                "unsupported benchmark plan schema {:?}",
                self.schema
            ));
        }
        if self.policy.payload_bytes == 0 {
            return Err("policy.payload_bytes must be greater than zero".to_string());
        }
        if self.policy.samples == 0 {
            return Err("policy.samples must be greater than zero".to_string());
        }
        if self.policy.benchmark_formats.is_empty()
            || self.policy.benchmark_formats.iter().any(String::is_empty)
        {
            return Err("policy.benchmark_formats must contain non-empty formats".to_string());
        }
        if self.policy.benchmark_workloads.is_empty()
            || self.policy.benchmark_workloads.iter().any(String::is_empty)
        {
            return Err("policy.benchmark_workloads must contain non-empty workloads".to_string());
        }
        if self.requested_format_count != self.policy.benchmark_formats.len() {
            return Err("requested_format_count does not match policy".to_string());
        }
        if self.requested_workload_count != self.policy.benchmark_workloads.len() {
            return Err("requested_workload_count does not match policy".to_string());
        }
        let expected_total = self.estimated_single_measurement_count
            + self.estimated_pair_measurement_count
            + self.estimated_group_measurement_count;
        if self.estimated_measurement_count != expected_total {
            return Err("estimated_measurement_count does not match component counts".to_string());
        }
        if self.selected_target_ids.len() + self.skipped_targets.len()
            > self.discovered_target_count
        {
            return Err("plan selected/skipped counts exceed discovered targets".to_string());
        }
        Ok(())
    }
}

fn combined_candidate_status(statuses: &[&str]) -> String {
    if statuses.is_empty() {
        "missing".to_string()
    } else if statuses.contains(&"completed") {
        "completed".to_string()
    } else if statuses.contains(&"failed") {
        "failed".to_string()
    } else if statuses.contains(&"unsupported") {
        "unsupported".to_string()
    } else if statuses.contains(&"unmeasured") {
        "unmeasured".to_string()
    } else if statuses.contains(&"skipped") {
        "skipped".to_string()
    } else {
        "unknown".to_string()
    }
}

fn increment_summary_status(summary: &mut BenchmarkRunSummary, status: &str) {
    match status {
        "completed" => summary.completed_count += 1,
        "unmeasured" => summary.unmeasured_count += 1,
        "failed" => summary.failed_count += 1,
        "unsupported" => summary.unsupported_count += 1,
        "skipped" => summary.skipped_count += 1,
        _ => {}
    }
}

fn increment_strategy_status(
    statuses: &mut BTreeMap<(String, String, String, String), StrategyStatusSummary>,
    comparison_group: &str,
    workload_class: &str,
    placement_strategy: &str,
    format: &str,
    status: &str,
) {
    let entry = statuses
        .entry((
            comparison_group.to_string(),
            workload_class.to_string(),
            placement_strategy.to_string(),
            format.to_string(),
        ))
        .or_insert_with(|| StrategyStatusSummary {
            comparison_group: comparison_group.to_string(),
            workload_class: workload_class.to_string(),
            placement_strategy: placement_strategy.to_string(),
            format: format.to_string(),
            completed_count: 0,
            unmeasured_count: 0,
            failed_count: 0,
            unsupported_count: 0,
            skipped_count: 0,
        });
    match status {
        "completed" => entry.completed_count += 1,
        "unmeasured" => entry.unmeasured_count += 1,
        "failed" => entry.failed_count += 1,
        "unsupported" => entry.unsupported_count += 1,
        "skipped" => entry.skipped_count += 1,
        _ => {}
    }
}

fn coverage_warnings_for_run(
    run: &BenchmarkRun,
    strategy_statuses: &BTreeMap<(String, String, String, String), StrategyStatusSummary>,
    candidate_statuses: &[ComparisonCandidateStatus],
) -> Vec<String> {
    const SMALL_GROUP: &str = "small_payload_placement_comparison";
    let has_small_payload_group = run
        .workload_specs
        .iter()
        .any(|spec| spec.comparison_group == SMALL_GROUP)
        || strategy_statuses
            .keys()
            .any(|(comparison_group, _, _, _)| comparison_group == SMALL_GROUP);
    if !has_small_payload_group {
        return Vec::new();
    }

    let mut expected = vec!["single_target_serial"];
    if run.policy.pair_measurements
        && run.policy.max_group_size >= 2
        && run.selected_target_ids.len() >= 2
    {
        expected.push("two_target_serial");
        expected.push("two_target_parallel");
    }
    if run.policy.pair_measurements
        && run.policy.max_group_size >= 3
        && run.selected_target_ids.len() >= 3
    {
        expected.push("three_target_serial");
        expected.push("three_target_parallel");
    }

    let mut warnings = Vec::new();
    let expected_axes = run
        .policy
        .benchmark_workloads
        .iter()
        .flat_map(|workload| {
            run.policy
                .benchmark_formats
                .iter()
                .map(move |format| (workload.as_str(), format.as_str()))
        })
        .collect::<Vec<_>>();
    for (workload_class, format) in expected_axes {
        for placement_strategy in &expected {
            let key = (
                SMALL_GROUP.to_string(),
                workload_class.to_string(),
                (*placement_strategy).to_string(),
                format.to_string(),
            );
            match strategy_statuses.get(&key) {
                Some(status) if status.completed_count > 0 => {}
                Some(status) => warnings.push(format!(
                    "placement strategy {placement_strategy:?} has no completed measurements for workload={workload_class} format={format}: unmeasured={} failed={} unsupported={} skipped={}",
                    status.unmeasured_count,
                    status.failed_count,
                    status.unsupported_count,
                    status.skipped_count
                )),
                None => warnings.push(format!(
                    "placement strategy {placement_strategy:?} is missing from small_payload_placement_comparison for workload={workload_class} format={format}"
                )),
            }
        }
    }
    for candidate in candidate_statuses {
        if candidate.status != "completed" {
            warnings.push(format!(
                "comparison candidate {:?} has no completed measurement: workload={} format={} strategy={} kind={} status={} matches={}",
                candidate.candidate_id,
                candidate.workload_class,
                candidate.format,
                candidate.placement_strategy,
                candidate.measurement_kind,
                candidate.status,
                candidate.matched_measurement_count
            ));
        }
    }
    warnings
}

pub fn parse_benchmark_run_json(input: &str) -> Result<BenchmarkRun, serde_json::Error> {
    serde_json::from_str(input)
}

pub fn parse_benchmark_plan_json(input: &str) -> Result<BenchmarkPlan, serde_json::Error> {
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

fn validate_candidate_measurement_kind(candidate: &ComparisonCandidate) -> Result<(), String> {
    let expected_target_count = match candidate.measurement_kind.as_str() {
        "single" => 1,
        "pair" => 2,
        "group" => 3,
        other => {
            return Err(format!(
                "comparison candidate {:?} has unsupported measurement kind {other:?}",
                candidate.candidate_id
            ));
        }
    };
    if candidate.target_ids.len() != expected_target_count {
        return Err(format!(
            "comparison candidate {:?} has measurement kind {:?} but {} target IDs",
            candidate.candidate_id,
            candidate.measurement_kind,
            candidate.target_ids.len()
        ));
    }
    Ok(())
}

fn validate_format_capabilities(target: &Target) -> Result<(), String> {
    let mut formats = BTreeSet::new();
    for capability in &target.format_capabilities {
        if capability.format.is_empty() || capability.support.is_empty() {
            return Err(format!(
                "target {:?} has malformed format capability",
                target.stable_target_id
            ));
        }
        if !formats.insert(capability.format.as_str()) {
            return Err(format!(
                "target {:?} has duplicate format capability {:?}",
                target.stable_target_id, capability.format
            ));
        }
        match capability.support.as_str() {
            "native" | "emulated" | "fallback" | "unsupported" | "unmeasured" => {}
            other => {
                return Err(format!(
                    "target {:?} has unsupported format capability status {other:?}",
                    target.stable_target_id
                ));
            }
        }
    }
    Ok(())
}

impl Implementation {
    pub fn current() -> Self {
        Self {
            name: "nerve-gpu-bench".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            backend_status: "cpu_reference_plus_opt_in_vulkan_single_pair_and_group_execution"
                .to_string(),
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
            pci_link: None,
            vulkan: None,
            capabilities: vec!["f32".to_string()],
            format_capabilities: Vec::new(),
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
                benchmark_formats: vec!["f32".to_string()],
                benchmark_workloads: vec!["dense_projection".to_string()],
                include_targets: Vec::new(),
                exclude_targets: Vec::new(),
                exclude_pci: Vec::new(),
                exclude_kinds: Vec::new(),
                pair_measurements: false,
                max_group_size: 1,
                execute: false,
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
                pci_link: None,
                vulkan: None,
                capabilities: Vec::new(),
                format_capabilities: Vec::new(),
                diagnostics: Vec::new(),
            }],
            selected_target_ids: vec!["cpu:host".to_string()],
            skipped_targets: Vec::new(),
            workload_specs: Vec::new(),
            comparison_sets: Vec::new(),
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
    fn validates_minimal_plan_document() {
        let plan = BenchmarkPlan {
            schema: PLAN_SCHEMA.to_string(),
            created_at_unix_ms: 1,
            policy: RunPolicy {
                payload_bytes: 1024,
                samples: 1,
                benchmark_formats: vec!["f32".to_string()],
                benchmark_workloads: vec!["dense_projection".to_string()],
                include_targets: Vec::new(),
                exclude_targets: Vec::new(),
                exclude_pci: Vec::new(),
                exclude_kinds: Vec::new(),
                pair_measurements: false,
                max_group_size: 1,
                execute: false,
            },
            discovered_target_count: 1,
            selected_target_ids: vec!["cpu:host".to_string()],
            skipped_targets: Vec::new(),
            requested_format_count: 1,
            requested_workload_count: 1,
            estimated_single_measurement_count: 6,
            estimated_pair_measurement_count: 0,
            estimated_group_measurement_count: 0,
            estimated_comparison_set_count: 0,
            estimated_measurement_count: 6,
            max_payload_bytes_per_measurement: 1024,
            diagnostics: Vec::new(),
        };
        let encoded = plan.to_json_pretty().unwrap();
        let parsed = parse_benchmark_plan_json(&encoded).unwrap();
        parsed.validate_basic().unwrap();
    }

    #[test]
    fn plan_validation_rejects_mismatched_totals() {
        let plan = BenchmarkPlan {
            schema: PLAN_SCHEMA.to_string(),
            created_at_unix_ms: 1,
            policy: RunPolicy {
                payload_bytes: 1024,
                samples: 1,
                benchmark_formats: vec!["f32".to_string()],
                benchmark_workloads: vec!["dense_projection".to_string()],
                include_targets: Vec::new(),
                exclude_targets: Vec::new(),
                exclude_pci: Vec::new(),
                exclude_kinds: Vec::new(),
                pair_measurements: false,
                max_group_size: 1,
                execute: false,
            },
            discovered_target_count: 1,
            selected_target_ids: vec!["cpu:host".to_string()],
            skipped_targets: Vec::new(),
            requested_format_count: 1,
            requested_workload_count: 1,
            estimated_single_measurement_count: 6,
            estimated_pair_measurement_count: 0,
            estimated_group_measurement_count: 0,
            estimated_comparison_set_count: 0,
            estimated_measurement_count: 5,
            max_payload_bytes_per_measurement: 1024,
            diagnostics: Vec::new(),
        };
        let error = plan.validate_basic().unwrap_err();
        assert!(error.contains("estimated_measurement_count"));
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
                benchmark_formats: vec!["f32".to_string()],
                benchmark_workloads: vec!["dense_projection".to_string()],
                include_targets: Vec::new(),
                exclude_targets: Vec::new(),
                exclude_pci: Vec::new(),
                exclude_kinds: Vec::new(),
                pair_measurements: false,
                max_group_size: 1,
                execute: false,
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
                pci_link: None,
                vulkan: None,
                capabilities: Vec::new(),
                format_capabilities: Vec::new(),
                diagnostics: Vec::new(),
            }],
            selected_target_ids: vec!["cpu:host".to_string()],
            skipped_targets: Vec::new(),
            workload_specs: Vec::new(),
            comparison_sets: Vec::new(),
            measurements: Vec::new(),
            pair_measurements: Vec::new(),
            group_measurements: Vec::new(),
            diagnostics: Vec::new(),
        };
        run.measurements.push(Measurement {
            workload_id: "work".to_string(),
            comparison_group: "test".to_string(),
            workload_class: "dense_projection".to_string(),
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
        assert_eq!(summary.strategy_statuses.len(), 1);
        assert_eq!(
            summary.strategy_statuses[0].placement_strategy,
            "single_target_serial"
        );
        assert_eq!(summary.strategy_statuses[0].completed_count, 1);
        assert!(summary.coverage_warnings.is_empty());
        assert!(summary.candidate_statuses.is_empty());
    }

    #[test]
    fn warns_when_required_pair_strategy_has_no_completed_measurement() {
        let mut run = BenchmarkRun {
            schema: RUN_SCHEMA.to_string(),
            started_at_unix_ms: 1,
            finished_at_unix_ms: 2,
            implementation: Implementation::current(),
            policy: RunPolicy {
                payload_bytes: 1024,
                samples: 1,
                benchmark_formats: vec!["f32".to_string()],
                benchmark_workloads: vec!["dense_projection".to_string()],
                include_targets: Vec::new(),
                exclude_targets: Vec::new(),
                exclude_pci: Vec::new(),
                exclude_kinds: Vec::new(),
                pair_measurements: true,
                max_group_size: 2,
                execute: false,
            },
            discovered_targets: vec![
                Target {
                    stable_target_id: "gpu:a".to_string(),
                    backend: "test".to_string(),
                    kind: "discrete_gpu".to_string(),
                    name: "gpu:a".to_string(),
                    vendor_id: None,
                    vendor_name: None,
                    device_id: None,
                    pci_address: None,
                    physical_location: None,
                    numa_node: None,
                    boot_vga: None,
                    pci_link: None,
                    vulkan: None,
                    capabilities: Vec::new(),
                    format_capabilities: Vec::new(),
                    diagnostics: Vec::new(),
                },
                Target {
                    stable_target_id: "gpu:b".to_string(),
                    backend: "test".to_string(),
                    kind: "discrete_gpu".to_string(),
                    name: "gpu:b".to_string(),
                    vendor_id: None,
                    vendor_name: None,
                    device_id: None,
                    pci_address: None,
                    physical_location: None,
                    numa_node: None,
                    boot_vga: None,
                    pci_link: None,
                    vulkan: None,
                    capabilities: Vec::new(),
                    format_capabilities: Vec::new(),
                    diagnostics: Vec::new(),
                },
            ],
            selected_target_ids: vec!["gpu:a".to_string(), "gpu:b".to_string()],
            skipped_targets: Vec::new(),
            workload_specs: Vec::new(),
            comparison_sets: Vec::new(),
            measurements: Vec::new(),
            pair_measurements: Vec::new(),
            group_measurements: Vec::new(),
            diagnostics: Vec::new(),
        };
        run.measurements.push(Measurement {
            workload_id: "single".to_string(),
            comparison_group: "small_payload_placement_comparison".to_string(),
            workload_class: "dense_projection".to_string(),
            placement_strategy: "single_target_serial".to_string(),
            target_id: "gpu:a".to_string(),
            pattern: "single".to_string(),
            operation_family: "test".to_string(),
            regime: "small_payload".to_string(),
            format: "backend_selected".to_string(),
            status: "unmeasured".to_string(),
            reason: Some("backend missing".to_string()),
            payload_bytes: 1024,
            working_set_bytes: 1024,
            samples: Vec::new(),
            summary: None,
        });
        let summary = run.summary();
        assert!(
            summary
                .coverage_warnings
                .iter()
                .any(|warning| warning.contains("two_target_parallel"))
        );
    }

    #[test]
    fn resolves_comparison_candidate_measurement_statuses() {
        let run = BenchmarkRun {
            schema: RUN_SCHEMA.to_string(),
            started_at_unix_ms: 1,
            finished_at_unix_ms: 2,
            implementation: Implementation::current(),
            policy: RunPolicy {
                payload_bytes: 1024,
                samples: 1,
                benchmark_formats: vec!["f32".to_string()],
                benchmark_workloads: vec!["dense_projection".to_string()],
                include_targets: Vec::new(),
                exclude_targets: Vec::new(),
                exclude_pci: Vec::new(),
                exclude_kinds: Vec::new(),
                pair_measurements: true,
                max_group_size: 2,
                execute: false,
            },
            discovered_targets: vec![test_target("gpu:a"), test_target("gpu:b")],
            selected_target_ids: vec!["gpu:a".to_string(), "gpu:b".to_string()],
            skipped_targets: Vec::new(),
            workload_specs: Vec::new(),
            comparison_sets: vec![ComparisonSet {
                comparison_id: "cmp".to_string(),
                comparison_group: "small_payload_placement_comparison".to_string(),
                workload_class: "dense_projection".to_string(),
                regime: "small_payload".to_string(),
                format: "backend_selected".to_string(),
                target_ids: vec!["gpu:a".to_string(), "gpu:b".to_string()],
                candidates: vec![
                    ComparisonCandidate {
                        candidate_id: "cmp:single".to_string(),
                        placement_strategy: "single_target_serial".to_string(),
                        measurement_kind: "single".to_string(),
                        workload_id: "single_target_small_payload".to_string(),
                        target_ids: vec!["gpu:a".to_string()],
                        notes: String::new(),
                    },
                    ComparisonCandidate {
                        candidate_id: "cmp:parallel".to_string(),
                        placement_strategy: "two_target_parallel".to_string(),
                        measurement_kind: "pair".to_string(),
                        workload_id: "synthetic_tensor_split_small_payload".to_string(),
                        target_ids: vec!["gpu:a".to_string(), "gpu:b".to_string()],
                        notes: String::new(),
                    },
                ],
            }],
            measurements: vec![Measurement {
                workload_id: "single_target_small_payload".to_string(),
                comparison_group: "small_payload_placement_comparison".to_string(),
                workload_class: "dense_projection".to_string(),
                placement_strategy: "single_target_serial".to_string(),
                target_id: "gpu:a".to_string(),
                pattern: "single".to_string(),
                operation_family: "test".to_string(),
                regime: "small_payload".to_string(),
                format: "backend_selected".to_string(),
                status: "completed".to_string(),
                reason: None,
                payload_bytes: 1024,
                working_set_bytes: 1024,
                samples: Vec::new(),
                summary: None,
            }],
            pair_measurements: Vec::new(),
            group_measurements: Vec::new(),
            diagnostics: Vec::new(),
        };
        let summary = run.summary();
        assert_eq!(summary.candidate_statuses.len(), 2);
        assert_eq!(summary.candidate_statuses[0].status, "completed");
        assert_eq!(summary.candidate_statuses[0].best_median_duration_ns, None);
        assert_eq!(summary.candidate_statuses[1].status, "missing");
        assert_eq!(summary.candidate_statuses[1].best_median_duration_ns, None);
        assert!(
            summary
                .coverage_warnings
                .iter()
                .any(|warning| warning.contains("cmp:parallel"))
        );
    }

    #[test]
    fn validation_rejects_candidate_with_unknown_workload_spec() {
        let run = BenchmarkRun {
            schema: RUN_SCHEMA.to_string(),
            started_at_unix_ms: 1,
            finished_at_unix_ms: 2,
            implementation: Implementation::current(),
            policy: RunPolicy {
                payload_bytes: 1024,
                samples: 1,
                benchmark_formats: vec!["f32".to_string()],
                benchmark_workloads: vec!["dense_projection".to_string()],
                include_targets: Vec::new(),
                exclude_targets: Vec::new(),
                exclude_pci: Vec::new(),
                exclude_kinds: Vec::new(),
                pair_measurements: true,
                max_group_size: 2,
                execute: false,
            },
            discovered_targets: vec![test_target("gpu:a"), test_target("gpu:b")],
            selected_target_ids: vec!["gpu:a".to_string(), "gpu:b".to_string()],
            skipped_targets: Vec::new(),
            workload_specs: vec![WorkloadSpec {
                workload_id: "single_target_small_payload:dense_projection:f32".to_string(),
                comparison_group: "small_payload_placement_comparison".to_string(),
                workload_class: "dense_projection".to_string(),
                placement_strategy: "single_target_serial".to_string(),
                pattern: "single_target_compute".to_string(),
                format: "f32".to_string(),
                participant_count: 1,
                payload_bytes: 1024,
                parameter_bytes_per_participant: 1024,
                activation_bytes: 1024,
                output_bytes: 1024,
                description: "test".to_string(),
            }],
            comparison_sets: vec![ComparisonSet {
                comparison_id: "cmp".to_string(),
                comparison_group: "small_payload_placement_comparison".to_string(),
                workload_class: "dense_projection".to_string(),
                regime: "small_payload".to_string(),
                format: "f32".to_string(),
                target_ids: vec!["gpu:a".to_string(), "gpu:b".to_string()],
                candidates: vec![ComparisonCandidate {
                    candidate_id: "cmp:bad".to_string(),
                    placement_strategy: "two_target_parallel".to_string(),
                    measurement_kind: "pair".to_string(),
                    workload_id: "missing:dense_projection:f32".to_string(),
                    target_ids: vec!["gpu:a".to_string(), "gpu:b".to_string()],
                    notes: String::new(),
                }],
            }],
            measurements: Vec::new(),
            pair_measurements: Vec::new(),
            group_measurements: Vec::new(),
            diagnostics: Vec::new(),
        };
        let error = run.validate_basic().unwrap_err();
        assert!(error.contains("references unknown workload"));
    }

    #[test]
    fn validation_rejects_candidate_kind_target_count_mismatch() {
        let run = BenchmarkRun {
            schema: RUN_SCHEMA.to_string(),
            started_at_unix_ms: 1,
            finished_at_unix_ms: 2,
            implementation: Implementation::current(),
            policy: RunPolicy {
                payload_bytes: 1024,
                samples: 1,
                benchmark_formats: vec!["f32".to_string()],
                benchmark_workloads: vec!["dense_projection".to_string()],
                include_targets: Vec::new(),
                exclude_targets: Vec::new(),
                exclude_pci: Vec::new(),
                exclude_kinds: Vec::new(),
                pair_measurements: true,
                max_group_size: 2,
                execute: false,
            },
            discovered_targets: vec![test_target("gpu:a"), test_target("gpu:b")],
            selected_target_ids: vec!["gpu:a".to_string(), "gpu:b".to_string()],
            skipped_targets: Vec::new(),
            workload_specs: vec![WorkloadSpec {
                workload_id: "synthetic_tensor_split_small_payload:dense_projection:f32"
                    .to_string(),
                comparison_group: "small_payload_placement_comparison".to_string(),
                workload_class: "dense_projection".to_string(),
                placement_strategy: "two_target_parallel".to_string(),
                pattern: "synthetic_tensor_split_small_payload".to_string(),
                format: "f32".to_string(),
                participant_count: 2,
                payload_bytes: 1024,
                parameter_bytes_per_participant: 512,
                activation_bytes: 1024,
                output_bytes: 1024,
                description: "test".to_string(),
            }],
            comparison_sets: vec![ComparisonSet {
                comparison_id: "cmp".to_string(),
                comparison_group: "small_payload_placement_comparison".to_string(),
                workload_class: "dense_projection".to_string(),
                regime: "small_payload".to_string(),
                format: "f32".to_string(),
                target_ids: vec!["gpu:a".to_string(), "gpu:b".to_string()],
                candidates: vec![ComparisonCandidate {
                    candidate_id: "cmp:bad".to_string(),
                    placement_strategy: "two_target_parallel".to_string(),
                    measurement_kind: "pair".to_string(),
                    workload_id: "synthetic_tensor_split_small_payload:dense_projection:f32"
                        .to_string(),
                    target_ids: vec!["gpu:a".to_string()],
                    notes: String::new(),
                }],
            }],
            measurements: Vec::new(),
            pair_measurements: Vec::new(),
            group_measurements: Vec::new(),
            diagnostics: Vec::new(),
        };
        let error = run.validate_basic().unwrap_err();
        assert!(error.contains("measurement kind"));
    }
}

#[cfg(test)]
fn test_target(id: &str) -> Target {
    Target {
        stable_target_id: id.to_string(),
        backend: "test".to_string(),
        kind: "discrete_gpu".to_string(),
        name: id.to_string(),
        vendor_id: None,
        vendor_name: None,
        device_id: None,
        pci_address: None,
        physical_location: None,
        numa_node: None,
        boot_vga: None,
        pci_link: None,
        vulkan: None,
        capabilities: Vec::new(),
        format_capabilities: Vec::new(),
        diagnostics: Vec::new(),
    }
}
