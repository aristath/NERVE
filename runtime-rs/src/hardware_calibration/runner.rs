use super::cpu::CpuCalibrationWorkload;
use super::schema::{
    CalibrationArtifactRecord, CalibrationExecutor, CalibrationRunStatus, CalibrationSamplePhase,
    CalibrationValidationResult, CalibrationValidationStatus, HARDWARE_CALIBRATION_RUN_SCHEMA,
    HardwareCalibrationPlan, HardwareCalibrationRun, HardwareCalibrationSample,
    HardwareCalibrationWorkload, HardwareCalibrationWorkloadResult,
};
#[cfg(feature = "vulkan")]
use super::vulkan_compute::VulkanComputeCalibrationExecutor;
#[cfg(feature = "vulkan")]
use super::vulkan_graphics::VulkanGraphicsCalibrationExecutor;
#[cfg(feature = "vulkan")]
use super::vulkan_transfer::VulkanTransferCalibrationExecutor;
use crate::hardware_profile::stable_hardware_id;
#[cfg(feature = "vulkan")]
use crate::vulkan_compute::VulkanComputeDevice;
use chrono::{SecondsFormat, Utc};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;
use std::path::PathBuf;
#[cfg(feature = "vulkan")]
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct CalibrationRunnerOptions {
    pub cancelled: Arc<AtomicBool>,
    pub artifact_directory: PathBuf,
    #[cfg(feature = "vulkan")]
    pub vulkan_physical_device_index: Option<usize>,
}

impl Default for CalibrationRunnerOptions {
    fn default() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
            artifact_directory: std::env::temp_dir().join(format!(
                "nerve-hardware-calibration-artifacts-{}",
                std::process::id()
            )),
            #[cfg(feature = "vulkan")]
            vulkan_physical_device_index: None,
        }
    }
}

pub fn run_calibration_plan(
    plan: &HardwareCalibrationPlan,
    options: &CalibrationRunnerOptions,
) -> Result<HardwareCalibrationRun, String> {
    plan.validate()?;
    let started_at = Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true);
    let run_id = stable_hardware_id(
        "calibration_run",
        &[
            Value::String(plan.plan_id.clone()),
            Value::String(plan.hardware_profile_id.clone()),
            Value::String(started_at.clone()),
        ],
    )?;
    let mut results = Vec::with_capacity(plan.workloads.len());
    #[cfg(feature = "vulkan")]
    let uses_vulkan = plan
        .workloads
        .iter()
        .any(|workload| workload.executor != CalibrationExecutor::Cpu);
    #[cfg(feature = "vulkan")]
    let vulkan_device = if uses_vulkan {
        let physical_device_index = options.vulkan_physical_device_index.ok_or_else(|| {
            "a Vulkan calibration plan requires an explicit physical device index".to_string()
        })?;
        Some(Rc::new(
            VulkanComputeDevice::new_for_physical_device_index(physical_device_index)
                .map_err(|error| format!("could not open Vulkan calibration device: {error}"))?,
        ))
    } else {
        None
    };
    #[cfg(feature = "vulkan")]
    let mut vulkan_compute = if plan
        .workloads
        .iter()
        .any(|workload| workload.executor == CalibrationExecutor::VulkanCompute)
    {
        Some(VulkanComputeCalibrationExecutor::new(
            Rc::clone(
                vulkan_device
                    .as_ref()
                    .expect("Vulkan calibration device was initialized"),
            ),
            options.artifact_directory.clone(),
        )?)
    } else {
        None
    };
    #[cfg(feature = "vulkan")]
    let mut vulkan_transfer = if plan
        .workloads
        .iter()
        .any(|workload| workload.executor == CalibrationExecutor::VulkanTransfer)
    {
        Some(VulkanTransferCalibrationExecutor::new(Rc::clone(
            vulkan_device
                .as_ref()
                .expect("Vulkan calibration device was initialized"),
        )))
    } else {
        None
    };
    #[cfg(feature = "vulkan")]
    let mut vulkan_graphics = if plan
        .workloads
        .iter()
        .any(|workload| workload.executor == CalibrationExecutor::VulkanGraphics)
    {
        Some(VulkanGraphicsCalibrationExecutor::new(
            Rc::clone(
                vulkan_device
                    .as_ref()
                    .expect("Vulkan calibration device was initialized"),
            ),
            options.artifact_directory.clone(),
        )?)
    } else {
        None
    };
    for workload in &plan.workloads {
        if options.cancelled.load(Ordering::Relaxed) {
            return Ok(HardwareCalibrationRun {
                schema: HARDWARE_CALIBRATION_RUN_SCHEMA.to_string(),
                run_id,
                plan_id: plan.plan_id.clone(),
                hardware_profile_id: plan.hardware_profile_id.clone(),
                status: CalibrationRunStatus::Cancelled,
                started_at,
                finished_at: Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true),
                workloads: results,
                diagnostics: vec!["calibration was cancelled".to_string()],
            });
        }
        let result = match workload.executor {
            CalibrationExecutor::Cpu => run_cpu_workload(plan, workload, options)?,
            #[cfg(feature = "vulkan")]
            CalibrationExecutor::VulkanCompute => vulkan_compute
                .as_mut()
                .expect("Vulkan executor was initialized for the plan")
                .run(plan, workload, &options.cancelled)?,
            #[cfg(feature = "vulkan")]
            CalibrationExecutor::VulkanTransfer => vulkan_transfer
                .as_mut()
                .expect("Vulkan transfer executor was initialized for the plan")
                .run(plan, workload, &options.cancelled)?,
            #[cfg(feature = "vulkan")]
            CalibrationExecutor::VulkanGraphics => vulkan_graphics
                .as_mut()
                .expect("Vulkan graphics executor was initialized for the plan")
                .run(plan, workload, &options.cancelled)?,
            executor => {
                return Err(format!(
                    "calibration executor {executor:?} is not implemented by this build"
                ));
            }
        };
        results.push(result);
    }
    let run = HardwareCalibrationRun {
        schema: HARDWARE_CALIBRATION_RUN_SCHEMA.to_string(),
        run_id,
        plan_id: plan.plan_id.clone(),
        hardware_profile_id: plan.hardware_profile_id.clone(),
        status: CalibrationRunStatus::Completed,
        started_at,
        finished_at: Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true),
        workloads: results,
        diagnostics: Vec::new(),
    };
    run.validate()?;
    Ok(run)
}

pub fn read_calibration_plan(path: &Path) -> Result<HardwareCalibrationPlan, String> {
    let payload = fs::read(path)
        .map_err(|error| format!("could not read calibration plan {path:?}: {error}"))?;
    let plan = serde_json::from_slice::<HardwareCalibrationPlan>(&payload)
        .map_err(|error| format!("could not parse calibration plan {path:?}: {error}"))?;
    plan.validate()?;
    Ok(plan)
}

fn run_cpu_workload(
    plan: &HardwareCalibrationPlan,
    workload: &HardwareCalibrationWorkload,
    options: &CalibrationRunnerOptions,
) -> Result<HardwareCalibrationWorkloadResult, String> {
    let construction_started = Instant::now();
    let mut state = CpuCalibrationWorkload::prepare(workload)?;
    let construction_duration_ns = elapsed_ns(construction_started);
    let mut samples = Vec::new();
    for _ in 0..plan.policy.warmup_iterations {
        samples.push(measure_minimum_duration(
            &mut state,
            CalibrationSamplePhase::Warmup,
            None,
            plan.policy.minimum_sample_duration_ns,
            options,
            samples.len(),
        )?);
    }
    for _ in 0..plan.policy.steady_iterations {
        samples.push(measure_minimum_duration(
            &mut state,
            CalibrationSamplePhase::Steady,
            None,
            plan.policy.minimum_sample_duration_ns,
            options,
            samples.len(),
        )?);
    }
    let sustained_duration_ns = plan
        .policy
        .sustained_window_duration_ms
        .saturating_mul(1_000_000);
    for window_index in 0..plan.policy.sustained_window_count {
        samples.push(measure_minimum_duration(
            &mut state,
            CalibrationSamplePhase::Sustained,
            Some(window_index),
            sustained_duration_ns,
            options,
            samples.len(),
        )?);
    }
    let observed_digest = state.observed_digest();
    let validation_passed = workload
        .validation
        .expected_digest
        .as_ref()
        .is_none_or(|expected| expected == &observed_digest);
    let iterations = samples.iter().map(|sample| sample.iterations).sum();
    Ok(HardwareCalibrationWorkloadResult {
        workload_id: workload.workload_id.clone(),
        status: if validation_passed {
            CalibrationRunStatus::Completed
        } else {
            CalibrationRunStatus::Failed
        },
        construction_duration_ns,
        artifacts: cpu_artifact_records(workload, &state, &options.artifact_directory)?,
        samples,
        validation: CalibrationValidationResult {
            status: if validation_passed {
                CalibrationValidationStatus::Passed
            } else {
                CalibrationValidationStatus::Failed
            },
            observed_digest: Some(observed_digest),
            maximum_error_ppm: 0,
        },
        counters: [("logical_iterations".to_string(), iterations)]
            .into_iter()
            .collect(),
        diagnostics: if validation_passed {
            Vec::new()
        } else {
            vec!["calibration output digest did not match its plan".to_string()]
        },
    })
}

fn cpu_artifact_records(
    workload: &HardwareCalibrationWorkload,
    state: &CpuCalibrationWorkload,
    artifact_directory: &Path,
) -> Result<Vec<CalibrationArtifactRecord>, String> {
    if workload.artifacts.is_empty() {
        return Ok(Vec::new());
    }
    let (bytes, kind) = state.generated_artifact().ok_or_else(|| {
        format!(
            "CPU workload {} declared artifacts but produced no physical artifact",
            workload.workload_id
        )
    })?;
    if workload.artifacts.len() != 1 || workload.artifacts[0].kind != kind {
        return Err(format!(
            "CPU workload {} artifact declaration does not match generated {kind}",
            workload.workload_id
        ));
    }
    fs::create_dir_all(artifact_directory).map_err(|error| {
        format!(
            "could not create CPU calibration artifact directory {artifact_directory:?}: {error}"
        )
    })?;
    let file_name = format!(
        "{}_{}.bin",
        sanitize_artifact_name(&workload.artifacts[0].name),
        workload.workload_id
    );
    let path = artifact_directory.join(&file_name);
    fs::write(&path, bytes)
        .map_err(|error| format!("could not write generated-code artifact {path:?}: {error}"))?;
    Ok(vec![CalibrationArtifactRecord {
        name: workload.artifacts[0].name.clone(),
        kind: kind.to_string(),
        digest: format!(
            "nerve.calibration_artifact_sha256.v1:{:x}",
            Sha256::digest(bytes)
        ),
        byte_length: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
        relative_path: file_name,
    }])
}

fn sanitize_artifact_name(name: &str) -> String {
    name.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '-' || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect()
}

fn measure_minimum_duration(
    state: &mut CpuCalibrationWorkload,
    phase: CalibrationSamplePhase,
    window_index: Option<usize>,
    minimum_duration_ns: u64,
    options: &CalibrationRunnerOptions,
    sample_index: usize,
) -> Result<HardwareCalibrationSample, String> {
    let target = Duration::from_nanos(minimum_duration_ns);
    let started = Instant::now();
    let mut iterations = 0u64;
    while started.elapsed() < target || iterations == 0 {
        if options.cancelled.load(Ordering::Relaxed) {
            return Err("calibration was cancelled during a sample".to_string());
        }
        state.execute_once()?;
        iterations = iterations.saturating_add(1);
        if iterations == u64::MAX {
            return Err("calibration sample iteration counter overflowed".to_string());
        }
    }
    Ok(HardwareCalibrationSample {
        sample_index,
        phase,
        duration_ns: elapsed_ns(started),
        device_duration_ns: None,
        iterations,
        window_index,
        thermal_millidegrees_celsius: maximum_thermal_millidegrees(),
        valid: true,
    })
}

fn elapsed_ns(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn maximum_thermal_millidegrees() -> Option<u64> {
    let entries = fs::read_dir("/sys/class/thermal").ok()?;
    entries
        .filter_map(Result::ok)
        .filter_map(|entry| fs::read_to_string(entry.path().join("temp")).ok())
        .filter_map(|value| value.trim().parse::<u64>().ok())
        .max()
}
