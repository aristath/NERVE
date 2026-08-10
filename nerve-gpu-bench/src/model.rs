use std::time::{SystemTime, UNIX_EPOCH};

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

pub const RUN_SCHEMA: &str = "nerve.gpu_benchmark_run.v1";
pub const PLAN_SCHEMA: &str = "nerve.gpu_benchmark_plan.v1";
pub const TARGET_LIST_SCHEMA: &str = "nerve.gpu_benchmark_targets.v1";
pub const PLACEMENT_SCHEMA: &str = "nerve.placement_bench";

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
pub struct PlacementBenchmark {
    pub schema: String,
    pub payload_bytes: usize,
    pub samples: usize,
    pub results: Vec<PlacementResult>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlacementResult {
    pub kind: String,
    pub strategy: String,
    pub targets: Vec<String>,
    pub workload: String,
    pub format: String,
    pub participants: usize,
    pub payload_bytes: usize,
    pub shard_bytes: Vec<usize>,
    pub activation_bytes: usize,
    pub output_bytes: usize,
    pub iters: u64,
    pub ns: u128,
    pub total_ns: u128,
    pub bytes: u64,
    pub work_ops: u64,
    pub bps: f64,
    pub ops: f64,
    pub transport: String,
    pub collective: String,
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
    pub fn to_placement_benchmark(&self) -> PlacementBenchmark {
        let mut results = Vec::new();
        results.extend(self.measurements.iter().filter_map(placement_single));
        results.extend(self.pair_measurements.iter().filter_map(placement_pair));
        results.extend(self.group_measurements.iter().filter_map(placement_group));
        PlacementBenchmark {
            schema: PLACEMENT_SCHEMA.to_string(),
            payload_bytes: self.policy.payload_bytes,
            samples: self.policy.samples,
            results,
        }
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

impl PlacementBenchmark {
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn validate_basic(&self) -> Result<(), String> {
        if self.schema != PLACEMENT_SCHEMA {
            return Err(format!(
                "unsupported placement benchmark schema {:?}",
                self.schema
            ));
        }
        if self.payload_bytes == 0 {
            return Err("payload_bytes must be greater than zero".to_string());
        }
        if self.samples == 0 {
            return Err("samples must be greater than zero".to_string());
        }
        for result in &self.results {
            validate_placement_result(result)?;
        }
        Ok(())
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

struct PlacementMetrics {
    ns: u128,
    total_ns: u128,
    iters: u64,
    bytes: u64,
    work_ops: u64,
    bps: f64,
    ops: f64,
}

fn placement_single(measurement: &Measurement) -> Option<PlacementResult> {
    placement_metrics(
        &measurement.status,
        measurement.summary.as_ref(),
        &measurement.samples,
    )
    .map(|metrics| PlacementResult {
        kind: "single".to_string(),
        strategy: placement_strategy(&measurement.placement_strategy).to_string(),
        targets: vec![measurement.target_id.clone()],
        workload: measurement.workload_class.clone(),
        format: measurement.format.clone(),
        participants: 1,
        payload_bytes: measurement.payload_bytes,
        shard_bytes: vec![measurement.payload_bytes],
        activation_bytes: 0,
        output_bytes: 0,
        iters: metrics.iters,
        ns: metrics.ns,
        total_ns: metrics.total_ns,
        bytes: metrics.bytes,
        work_ops: metrics.work_ops,
        bps: metrics.bps,
        ops: metrics.ops,
        transport: placement_transport(&measurement.placement_strategy).to_string(),
        collective: placement_collective(
            &measurement.placement_strategy,
            &measurement.workload_class,
        )
        .to_string(),
    })
}

fn placement_pair(measurement: &PairMeasurement) -> Option<PlacementResult> {
    placement_metrics(
        &measurement.status,
        measurement.summary.as_ref(),
        &measurement.samples,
    )
    .map(|metrics| PlacementResult {
        kind: "pair".to_string(),
        strategy: placement_strategy(&measurement.placement_strategy).to_string(),
        targets: vec![
            measurement.source_target_id.clone(),
            measurement.destination_target_id.clone(),
        ],
        workload: measurement.workload_class.clone(),
        format: measurement.format.clone(),
        participants: 2,
        payload_bytes: measurement.payload_bytes,
        shard_bytes: vec![
            measurement.source_payload_bytes,
            measurement.destination_payload_bytes,
        ],
        activation_bytes: measurement.activation_bytes,
        output_bytes: measurement.output_bytes,
        iters: metrics.iters,
        ns: metrics.ns,
        total_ns: metrics.total_ns,
        bytes: metrics.bytes,
        work_ops: metrics.work_ops,
        bps: metrics.bps,
        ops: metrics.ops,
        transport: placement_transport(&measurement.placement_strategy).to_string(),
        collective: placement_collective(
            &measurement.placement_strategy,
            &measurement.workload_class,
        )
        .to_string(),
    })
}

fn placement_group(measurement: &GroupMeasurement) -> Option<PlacementResult> {
    placement_metrics(
        &measurement.status,
        measurement.summary.as_ref(),
        &measurement.samples,
    )
    .map(|metrics| PlacementResult {
        kind: "group".to_string(),
        strategy: placement_strategy(&measurement.placement_strategy).to_string(),
        targets: measurement.target_ids.clone(),
        workload: measurement.workload_class.clone(),
        format: measurement.format.clone(),
        participants: measurement.participant_count,
        payload_bytes: measurement.payload_bytes,
        shard_bytes: measurement.payload_bytes_per_participant.clone(),
        activation_bytes: measurement.activation_bytes,
        output_bytes: measurement.output_bytes,
        iters: metrics.iters,
        ns: metrics.ns,
        total_ns: metrics.total_ns,
        bytes: metrics.bytes,
        work_ops: metrics.work_ops,
        bps: metrics.bps,
        ops: metrics.ops,
        transport: placement_transport(&measurement.placement_strategy).to_string(),
        collective: placement_collective(
            &measurement.placement_strategy,
            &measurement.workload_class,
        )
        .to_string(),
    })
}

fn placement_metrics(
    status: &str,
    summary: Option<&Summary>,
    samples: &[Sample],
) -> Option<PlacementMetrics> {
    let _summary = (status == "completed").then_some(summary).flatten()?;
    if samples.is_empty() {
        return None;
    }
    let mut rows = samples
        .iter()
        .map(|sample| {
            let iters = sample.iterations.max(1);
            let ns = sample.duration_ns / u128::from(iters);
            let bytes = (sample.bytes_read + sample.bytes_written) / iters;
            let work_ops = sample.operations / iters;
            (ns, sample.duration_ns, iters, bytes, work_ops)
        })
        .collect::<Vec<_>>();
    rows.sort_unstable_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.cmp(&right.2))
            .then_with(|| left.3.cmp(&right.3))
            .then_with(|| left.4.cmp(&right.4))
    });
    let (ns, total_ns, iters, bytes, work_ops) = rows[rows.len() / 2];
    let seconds = ns as f64 / 1_000_000_000.0;
    Some(PlacementMetrics {
        ns,
        total_ns,
        iters,
        bytes,
        work_ops,
        bps: bytes as f64 / seconds.max(f64::EPSILON),
        ops: work_ops as f64 / seconds.max(f64::EPSILON),
    })
}

fn placement_strategy(strategy: &str) -> &str {
    match strategy {
        "single_target_serial" => "single",
        "activation_transfer_only" => "transfer",
        "two_target_serial" | "three_target_serial" => "serial",
        "two_target_tensor_parallel" | "three_target_tensor_parallel" => "tp",
        other => other,
    }
}

fn placement_transport(strategy: &str) -> &'static str {
    match strategy {
        "single_target_serial" => "none",
        "activation_transfer_only"
        | "two_target_serial"
        | "three_target_serial"
        | "two_target_tensor_parallel"
        | "three_target_tensor_parallel" => "host_staged",
        "two_stage_serial_reference" | "two_shard_parallel_reference" => "host",
        _ => "unknown",
    }
}

fn placement_collective(strategy: &str, workload: &str) -> &'static str {
    match strategy {
        "single_target_serial" => "none",
        "activation_transfer_only" => "copy",
        "two_target_serial" | "three_target_serial" => "pipeline",
        "two_target_tensor_parallel" | "three_target_tensor_parallel" => {
            tensor_parallel_collective_name(workload)
        }
        "two_stage_serial_reference" => "pipeline",
        "two_shard_parallel_reference" => "reference_shard_merge",
        _ => "unknown",
    }
}

fn tensor_parallel_collective_name(workload: &str) -> &'static str {
    match workload {
        "dense_projection" => "all_reduce_output",
        "moe_expert" => "expert_activation_gather",
        "router_reduction" => "router_score_reduce",
        _ => "all_reduce_output",
    }
}

fn validate_placement_result(result: &PlacementResult) -> Result<(), String> {
    if result.kind.is_empty()
        || result.strategy.is_empty()
        || result.workload.is_empty()
        || result.format.is_empty()
        || result.targets.is_empty()
        || result.transport.is_empty()
        || result.collective.is_empty()
        || result.shard_bytes.is_empty()
    {
        return Err("placement result contains empty identity fields".to_string());
    }
    if result.participants == 0 {
        return Err("placement result participants must be greater than zero".to_string());
    }
    if result.participants != result.targets.len() {
        return Err("placement result participants must match target count".to_string());
    }
    if result.participants != result.shard_bytes.len() {
        return Err("placement result participants must match shard count".to_string());
    }
    if result.payload_bytes == 0 {
        return Err("placement result payload_bytes must be greater than zero".to_string());
    }
    if result.iters == 0 {
        return Err("placement result iters must be greater than zero".to_string());
    }
    if result.ns == 0 {
        return Err("placement result ns must be greater than zero".to_string());
    }
    if result.total_ns < result.ns {
        return Err("placement result total_ns must not be smaller than ns".to_string());
    }
    if !result.bps.is_finite() || !result.ops.is_finite() {
        return Err("placement result throughput fields must be finite".to_string());
    }
    Ok(())
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
        expected.push("two_target_tensor_parallel");
    }
    if run.policy.pair_measurements
        && run.policy.max_group_size >= 3
        && run.selected_target_ids.len() >= 3
    {
        expected.push("three_target_serial");
        expected.push("three_target_tensor_parallel");
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

pub fn parse_placement_benchmark_json(
    input: &str,
) -> Result<PlacementBenchmark, serde_json::Error> {
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
        let encoded = serde_json::to_string_pretty(&run).unwrap();
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
    fn placement_benchmark_contains_only_completed_rankable_results() {
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
                execute: true,
            },
            discovered_targets: vec![test_target("gpu:a"), test_target("gpu:b")],
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
            workload_id: "single_target_small_payload:dense_projection:f32".to_string(),
            comparison_group: "small_payload_placement_comparison".to_string(),
            workload_class: "dense_projection".to_string(),
            placement_strategy: "single_target_serial".to_string(),
            target_id: "gpu:a".to_string(),
            pattern: "single".to_string(),
            operation_family: "dense_projection".to_string(),
            regime: "small_payload".to_string(),
            format: "f32".to_string(),
            status: "completed".to_string(),
            reason: None,
            payload_bytes: 1024,
            working_set_bytes: 1024,
            samples: vec![Sample {
                sample_index: 0,
                duration_ns: 12,
                iterations: 1,
                bytes_read: 64,
                bytes_written: 64,
                operations: 16,
            }],
            summary: Some(Summary {
                min_duration_ns: 10,
                median_duration_ns: 12,
                bytes_per_second: 100.0,
                operations_per_second: 200.0,
            }),
        });
        run.measurements.push(Measurement {
            workload_id: "single_target_small_payload:dense_projection:bf16".to_string(),
            comparison_group: "small_payload_placement_comparison".to_string(),
            workload_class: "dense_projection".to_string(),
            placement_strategy: "single_target_serial".to_string(),
            target_id: "gpu:a".to_string(),
            pattern: "single".to_string(),
            operation_family: "dense_projection".to_string(),
            regime: "small_payload".to_string(),
            format: "bf16".to_string(),
            status: "unsupported".to_string(),
            reason: Some("not available".to_string()),
            payload_bytes: 1024,
            working_set_bytes: 1024,
            samples: Vec::new(),
            summary: None,
        });
        run.pair_measurements.push(PairMeasurement {
            workload_id: "synthetic_tensor_parallel_small_payload:dense_projection:f32".to_string(),
            comparison_group: "small_payload_placement_comparison".to_string(),
            workload_class: "dense_projection".to_string(),
            placement_strategy: "two_target_tensor_parallel".to_string(),
            source_target_id: "gpu:a".to_string(),
            destination_target_id: "gpu:b".to_string(),
            pattern: "synthetic_tensor_parallel_small_payload".to_string(),
            operation_family: "dense_projection".to_string(),
            regime: "small_payload".to_string(),
            format: "f32".to_string(),
            status: "completed".to_string(),
            reason: None,
            payload_bytes: 1024,
            source_payload_bytes: 512,
            destination_payload_bytes: 512,
            activation_bytes: 1024,
            output_bytes: 64,
            samples: vec![Sample {
                sample_index: 0,
                duration_ns: 100,
                iterations: 4,
                bytes_read: 400,
                bytes_written: 200,
                operations: 80,
            }],
            summary: Some(Summary {
                min_duration_ns: 100,
                median_duration_ns: 100,
                bytes_per_second: 300.0,
                operations_per_second: 400.0,
            }),
        });

        let placement = run.to_placement_benchmark();
        placement.validate_basic().unwrap();
        assert_eq!(placement.schema, PLACEMENT_SCHEMA);
        assert_eq!(placement.results.len(), 2);
        assert_eq!(placement.results[0].kind, "single");
        assert_eq!(placement.results[0].strategy, "single");
        assert_eq!(placement.results[0].targets, ["gpu:a"]);
        assert_eq!(placement.results[0].participants, 1);
        assert_eq!(placement.results[0].payload_bytes, 1024);
        assert_eq!(placement.results[0].shard_bytes, [1024]);
        assert_eq!(placement.results[0].activation_bytes, 0);
        assert_eq!(placement.results[0].output_bytes, 0);
        assert_eq!(placement.results[0].ns, 12);
        assert_eq!(placement.results[0].total_ns, 12);
        assert_eq!(placement.results[0].iters, 1);
        assert_eq!(placement.results[0].bytes, 128);
        assert_eq!(placement.results[0].work_ops, 16);
        assert!(placement.results[0].bps > 0.0);
        assert!(placement.results[0].ops > 0.0);
        assert_eq!(placement.results[0].transport, "none");
        assert_eq!(placement.results[0].collective, "none");
        assert_eq!(placement.results[1].kind, "pair");
        assert_eq!(placement.results[1].strategy, "tp");
        assert_eq!(placement.results[1].targets, ["gpu:a", "gpu:b"]);
        assert_eq!(placement.results[1].participants, 2);
        assert_eq!(placement.results[1].payload_bytes, 1024);
        assert_eq!(placement.results[1].shard_bytes, [512, 512]);
        assert_eq!(placement.results[1].activation_bytes, 1024);
        assert_eq!(placement.results[1].output_bytes, 64);
        assert_eq!(placement.results[1].ns, 25);
        assert_eq!(placement.results[1].total_ns, 100);
        assert_eq!(placement.results[1].iters, 4);
        assert_eq!(placement.results[1].bytes, 150);
        assert_eq!(placement.results[1].work_ops, 20);
        assert!(placement.results[1].bps > 0.0);
        assert!(placement.results[1].ops > 0.0);
        assert_eq!(placement.results[1].transport, "host_staged");
        assert_eq!(placement.results[1].collective, "all_reduce_output");

        let encoded = placement.to_json().unwrap();
        assert!(!encoded.contains("discovered_targets"));
        assert!(!encoded.contains("unsupported"));
        parse_placement_benchmark_json(&encoded)
            .unwrap()
            .validate_basic()
            .unwrap();
    }

    #[test]
    fn placement_collectives_describe_strategy_and_workload() {
        assert_eq!(
            placement_collective("two_target_tensor_parallel", "dense_projection"),
            "all_reduce_output"
        );
        assert_eq!(
            placement_collective("two_target_tensor_parallel", "moe_expert"),
            "expert_activation_gather"
        );
        assert_eq!(
            placement_collective("three_target_tensor_parallel", "router_reduction"),
            "router_score_reduce"
        );
        assert_eq!(
            placement_collective("two_target_serial", "moe_expert"),
            "pipeline"
        );
        assert_eq!(
            placement_transport("three_target_tensor_parallel"),
            "host_staged"
        );
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
                .any(|warning| warning.contains("two_target_tensor_parallel"))
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
                        candidate_id: "cmp:tp".to_string(),
                        placement_strategy: "two_target_tensor_parallel".to_string(),
                        measurement_kind: "pair".to_string(),
                        workload_id: "synthetic_tensor_parallel_small_payload".to_string(),
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
                .any(|warning| warning.contains("cmp:tp"))
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
                    placement_strategy: "two_target_tensor_parallel".to_string(),
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
                workload_id: "synthetic_tensor_parallel_small_payload:dense_projection:f32"
                    .to_string(),
                comparison_group: "small_payload_placement_comparison".to_string(),
                workload_class: "dense_projection".to_string(),
                placement_strategy: "two_target_tensor_parallel".to_string(),
                pattern: "synthetic_tensor_parallel_small_payload".to_string(),
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
                    placement_strategy: "two_target_tensor_parallel".to_string(),
                    measurement_kind: "pair".to_string(),
                    workload_id: "synthetic_tensor_parallel_small_payload:dense_projection:f32"
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
