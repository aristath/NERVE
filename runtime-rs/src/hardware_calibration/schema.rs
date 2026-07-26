use crate::hardware_profile::stable_hardware_id;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

pub const HARDWARE_CALIBRATION_PLAN_SCHEMA: &str = "nerve.optimizer.hardware_calibration_plan.v1";
pub const HARDWARE_CALIBRATION_RUN_SCHEMA: &str = "nerve.optimizer.hardware_calibration_run.v1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareCalibrationPlan {
    pub schema: String,
    pub plan_id: String,
    pub hardware_profile_id: String,
    pub capability_class: String,
    pub implementation: CalibrationImplementation,
    pub policy: HardwareCalibrationPolicy,
    pub workloads: Vec<HardwareCalibrationWorkload>,
    pub excluded_processes: Vec<ExcludedHardwareProcess>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationImplementation {
    pub name: String,
    pub version: String,
    pub fingerprint: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareCalibrationPolicy {
    pub minimum_warmup_samples: usize,
    pub maximum_warmup_samples: usize,
    pub warmup_stability_window_samples: usize,
    pub minimum_warmup_duration_ns: u64,
    pub maximum_warmup_relative_shift_ppm: u64,
    pub minimum_steady_samples: usize,
    pub maximum_steady_samples: usize,
    pub minimum_sample_duration_ns: u64,
    pub sustained_window_duration_ms: u64,
    pub sustained_window_count: usize,
    pub confidence_level_ppm: u64,
    pub maximum_relative_ci_width_ppm: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareCalibrationWorkload {
    pub workload_id: String,
    pub process_names: Vec<String>,
    pub executor: CalibrationExecutor,
    pub operation: String,
    pub regime: BTreeMap<String, String>,
    pub work: CalibrationUsefulWork,
    pub artifacts: Vec<CalibrationArtifactDeclaration>,
    pub validation: CalibrationValidationContract,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationExecutor {
    Cpu,
    VulkanCompute,
    VulkanDgc,
    VulkanGraphics,
    VulkanRay,
    VulkanSynchronization,
    VulkanTransfer,
    VulkanVideo,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationUsefulWork {
    pub items_per_iteration: u64,
    pub operations_per_iteration: u64,
    pub bytes_read_per_iteration: u64,
    pub bytes_written_per_iteration: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationArtifactDeclaration {
    pub name: String,
    pub kind: String,
    pub digest: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationValidationContract {
    pub mode: String,
    pub expected_digest: Option<String>,
    pub maximum_error_ppm: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExcludedHardwareProcess {
    pub process_name: String,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareCalibrationRun {
    pub schema: String,
    pub run_id: String,
    pub plan_id: String,
    pub hardware_profile_id: String,
    pub status: CalibrationRunStatus,
    pub started_at: String,
    pub finished_at: String,
    pub workloads: Vec<HardwareCalibrationWorkloadResult>,
    pub diagnostics: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationRunStatus {
    Completed,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareCalibrationWorkloadResult {
    pub workload_id: String,
    pub status: CalibrationRunStatus,
    pub construction_duration_ns: u64,
    pub artifacts: Vec<CalibrationArtifactRecord>,
    pub samples: Vec<HardwareCalibrationSample>,
    pub validation: CalibrationValidationResult,
    pub counters: BTreeMap<String, u64>,
    pub diagnostics: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationArtifactRecord {
    pub name: String,
    pub kind: String,
    pub digest: String,
    pub byte_length: u64,
    pub relative_path: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareCalibrationSample {
    pub sample_index: usize,
    pub phase: CalibrationSamplePhase,
    pub duration_ns: u64,
    pub device_duration_ns: Option<u64>,
    pub iterations: u64,
    pub window_index: Option<usize>,
    pub thermal_millidegrees_celsius: Option<u64>,
    pub valid: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationSamplePhase {
    Cold,
    Warmup,
    Steady,
    Sustained,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationValidationResult {
    pub status: CalibrationValidationStatus,
    pub observed_digest: Option<String>,
    pub maximum_error_ppm: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationValidationStatus {
    Passed,
    Failed,
    NotRun,
}

impl HardwareCalibrationPlan {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != HARDWARE_CALIBRATION_PLAN_SCHEMA {
            return Err(format!(
                "unsupported hardware-calibration plan schema {:?}",
                self.schema
            ));
        }
        validate_stable_id(&self.hardware_profile_id, "hardware_profile")?;
        validate_stable_id(&self.capability_class, "hardware_capability")?;
        self.implementation.validate()?;
        self.policy.validate()?;
        if self.workloads.is_empty() {
            return Err("hardware-calibration plan contains no workloads".to_string());
        }
        let workload_ids = self
            .workloads
            .iter()
            .map(|workload| workload.workload_id.as_str())
            .collect::<Vec<_>>();
        if !strictly_sorted_unique(&workload_ids) {
            return Err(
                "hardware-calibration workloads must have unique sorted identities".to_string(),
            );
        }
        let mut calibrated_processes = BTreeSet::new();
        for workload in &self.workloads {
            workload.validate()?;
            calibrated_processes.extend(workload.process_names.iter().cloned());
        }
        let exclusion_names = self
            .excluded_processes
            .iter()
            .map(|exclusion| exclusion.process_name.as_str())
            .collect::<Vec<_>>();
        if !strictly_sorted_unique(&exclusion_names) {
            return Err("excluded calibration processes must have unique sorted names".to_string());
        }
        for exclusion in &self.excluded_processes {
            exclusion.validate()?;
            if calibrated_processes.contains(&exclusion.process_name) {
                return Err(format!(
                    "hardware process {:?} is both calibrated and excluded",
                    exclusion.process_name
                ));
            }
        }
        let workload_values = self
            .workloads
            .iter()
            .map(serde_json::to_value)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("could not serialize calibration workloads: {error}"))?;
        let exclusion_values = self
            .excluded_processes
            .iter()
            .map(serde_json::to_value)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("could not serialize excluded processes: {error}"))?;
        let expected_id = stable_hardware_id(
            "calibration_plan",
            &[
                Value::String(self.hardware_profile_id.clone()),
                Value::String(self.capability_class.clone()),
                serde_json::to_value(&self.implementation)
                    .map_err(|error| format!("could not serialize implementation: {error}"))?,
                serde_json::to_value(&self.policy)
                    .map_err(|error| format!("could not serialize policy: {error}"))?,
                Value::Array(workload_values),
                Value::Array(exclusion_values),
            ],
        )?;
        if self.plan_id != expected_id {
            return Err("calibration plan identity does not match its content".to_string());
        }
        Ok(())
    }
}

impl CalibrationImplementation {
    fn validate(&self) -> Result<(), String> {
        if self.name.is_empty() || self.version.is_empty() || !is_digest(&self.fingerprint) {
            return Err("hardware-calibration implementation identity is invalid".to_string());
        }
        Ok(())
    }
}

impl HardwareCalibrationPolicy {
    fn validate(&self) -> Result<(), String> {
        if self.minimum_warmup_samples == 0
            || self.maximum_warmup_samples < self.minimum_warmup_samples
            || self.maximum_warmup_samples < self.warmup_stability_window_samples.saturating_mul(2)
            || self.warmup_stability_window_samples == 0
            || self.minimum_warmup_duration_ns == 0
            || self.maximum_warmup_relative_shift_ppm == 0
            || self.maximum_warmup_relative_shift_ppm >= 1_000_000
            || self.minimum_steady_samples < 5
            || self.maximum_steady_samples < self.minimum_steady_samples
            || self.minimum_sample_duration_ns == 0
            || self.sustained_window_duration_ms == 0
            || self.sustained_window_count == 0
            || self.confidence_level_ppm == 0
            || self.confidence_level_ppm >= 1_000_000
            || self.maximum_relative_ci_width_ppm == 0
            || self.maximum_relative_ci_width_ppm >= 1_000_000
        {
            return Err("hardware-calibration policy is invalid".to_string());
        }
        Ok(())
    }
}

impl HardwareCalibrationWorkload {
    fn validate(&self) -> Result<(), String> {
        validate_stable_id(&self.workload_id, "calibration_workload")?;
        if self.operation.is_empty()
            || self.process_names.is_empty()
            || !sorted_unique(&self.process_names)
        {
            return Err(format!(
                "hardware-calibration workload {:?} is incomplete",
                self.workload_id
            ));
        }
        if self.work.items_per_iteration == 0
            && self.work.operations_per_iteration == 0
            && self.work.bytes_read_per_iteration == 0
            && self.work.bytes_written_per_iteration == 0
        {
            return Err(format!(
                "hardware-calibration workload {:?} declares no useful work",
                self.workload_id
            ));
        }
        let artifact_names = self
            .artifacts
            .iter()
            .map(|artifact| artifact.name.as_str())
            .collect::<Vec<_>>();
        if !strictly_sorted_unique(&artifact_names) {
            return Err(format!(
                "hardware-calibration workload {:?} artifacts must be sorted and unique",
                self.workload_id
            ));
        }
        for artifact in &self.artifacts {
            artifact.validate()?;
        }
        self.validation.validate()?;
        let expected_id = stable_hardware_id(
            "calibration_workload",
            &[
                serde_json::to_value(&self.process_names)
                    .map_err(|error| format!("could not serialize process names: {error}"))?,
                serde_json::to_value(self.executor)
                    .map_err(|error| format!("could not serialize executor: {error}"))?,
                Value::String(self.operation.clone()),
                serde_json::to_value(&self.regime)
                    .map_err(|error| format!("could not serialize regime: {error}"))?,
                serde_json::to_value(&self.work)
                    .map_err(|error| format!("could not serialize useful work: {error}"))?,
                serde_json::to_value(&self.artifacts)
                    .map_err(|error| format!("could not serialize artifacts: {error}"))?,
                serde_json::to_value(&self.validation)
                    .map_err(|error| format!("could not serialize validation: {error}"))?,
            ],
        )?;
        if self.workload_id != expected_id {
            return Err(format!(
                "hardware-calibration workload {:?} identity does not match its content",
                self.workload_id
            ));
        }
        Ok(())
    }
}

impl CalibrationArtifactDeclaration {
    fn validate(&self) -> Result<(), String> {
        if self.name.is_empty()
            || self.kind.is_empty()
            || self
                .digest
                .as_ref()
                .is_some_and(|digest| !is_digest(digest))
        {
            return Err("hardware-calibration artifact declaration is invalid".to_string());
        }
        Ok(())
    }
}

impl CalibrationValidationContract {
    fn validate(&self) -> Result<(), String> {
        if !matches!(self.mode.as_str(), "digest" | "exact" | "tolerance")
            || self
                .expected_digest
                .as_ref()
                .is_some_and(|digest| !is_digest(digest))
            || matches!(self.mode.as_str(), "digest" | "exact") && self.maximum_error_ppm != 0
        {
            return Err("hardware-calibration validation contract is invalid".to_string());
        }
        Ok(())
    }
}

impl ExcludedHardwareProcess {
    fn validate(&self) -> Result<(), String> {
        if self.process_name.is_empty()
            || !matches!(
                self.reason.as_str(),
                "unavailable" | "not_programmable" | "not_exposed_by_selected_api"
            )
        {
            return Err("excluded hardware-calibration process is invalid".to_string());
        }
        Ok(())
    }
}

impl HardwareCalibrationRun {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != HARDWARE_CALIBRATION_RUN_SCHEMA {
            return Err(format!(
                "unsupported hardware-calibration run schema {:?}",
                self.schema
            ));
        }
        validate_stable_id(&self.plan_id, "calibration_plan")?;
        validate_stable_id(&self.hardware_profile_id, "hardware_profile")?;
        if self.started_at.is_empty() || self.finished_at.is_empty() {
            return Err("hardware-calibration run timestamps are missing".to_string());
        }
        let workload_ids = self
            .workloads
            .iter()
            .map(|workload| workload.workload_id.as_str())
            .collect::<Vec<_>>();
        if !strictly_sorted_unique(&workload_ids) {
            return Err(
                "hardware-calibration results must have unique sorted identities".to_string(),
            );
        }
        for workload in &self.workloads {
            workload.validate()?;
        }
        if self.status == CalibrationRunStatus::Completed
            && self
                .workloads
                .iter()
                .any(|workload| workload.status != CalibrationRunStatus::Completed)
        {
            return Err(
                "completed hardware-calibration run contains incomplete workloads".to_string(),
            );
        }
        let expected_id = stable_hardware_id(
            "calibration_run",
            &[
                Value::String(self.plan_id.clone()),
                Value::String(self.hardware_profile_id.clone()),
                Value::String(self.started_at.clone()),
            ],
        )?;
        if self.run_id != expected_id {
            return Err("hardware-calibration run identity does not match its content".to_string());
        }
        Ok(())
    }
}

impl HardwareCalibrationWorkloadResult {
    fn validate(&self) -> Result<(), String> {
        validate_stable_id(&self.workload_id, "calibration_workload")?;
        let artifact_names = self
            .artifacts
            .iter()
            .map(|artifact| artifact.name.as_str())
            .collect::<Vec<_>>();
        if !strictly_sorted_unique(&artifact_names)
            || self.artifacts.iter().any(|artifact| {
                artifact.name.is_empty()
                    || artifact.kind.is_empty()
                    || !is_digest(&artifact.digest)
                    || artifact.byte_length == 0
                    || artifact.relative_path.is_empty()
                    || artifact.relative_path.starts_with('/')
                    || artifact.relative_path.split('/').any(|part| part == "..")
            })
        {
            return Err(format!(
                "hardware-calibration workload {:?} has invalid artifacts",
                self.workload_id
            ));
        }
        for (index, sample) in self.samples.iter().enumerate() {
            if sample.sample_index != index || sample.duration_ns == 0 || sample.iterations == 0 {
                return Err(format!(
                    "hardware-calibration workload {:?} has an invalid sample",
                    self.workload_id
                ));
            }
        }
        if self.status == CalibrationRunStatus::Completed
            && (self.samples.is_empty()
                || self.validation.status != CalibrationValidationStatus::Passed)
        {
            return Err(format!(
                "completed hardware-calibration workload {:?} is not validated",
                self.workload_id
            ));
        }
        Ok(())
    }
}

fn validate_stable_id(value: &str, prefix: &str) -> Result<(), String> {
    let expected_prefix = format!("{prefix}_");
    let suffix = value.strip_prefix(&expected_prefix).ok_or_else(|| {
        format!("stable identity {value:?} does not begin with {expected_prefix:?}")
    })?;
    if suffix.len() != 32 || !suffix.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("stable identity {value:?} is malformed"));
    }
    Ok(())
}

fn is_digest(value: &str) -> bool {
    value.rsplit_once(':').is_some_and(|(schema, digest)| {
        !schema.is_empty()
            && digest.len() == 64
            && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
    })
}

fn sorted_unique(values: &[String]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

fn strictly_sorted_unique(values: &[&str]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}
