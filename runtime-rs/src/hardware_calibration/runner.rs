use super::cpu::CpuCalibrationWorkload;
use super::schema::{
    CalibrationArtifactRecord, CalibrationExecutor, CalibrationRunStatus, CalibrationSamplePhase,
    CalibrationValidationResult, CalibrationValidationStatus, HARDWARE_CALIBRATION_RUN_SCHEMA,
    HardwareCalibrationPlan, HardwareCalibrationRun, HardwareCalibrationSample,
    HardwareCalibrationWorkload, HardwareCalibrationWorkloadResult,
};
use super::telemetry::{elapsed_ns, maximum_cpu_temperature_millidegrees};
#[cfg(feature = "vulkan")]
use super::vulkan_compute::VulkanComputeCalibrationExecutor;
#[cfg(feature = "vulkan")]
use super::vulkan_dgc::VulkanDgcCalibrationExecutor;
#[cfg(feature = "vulkan")]
use super::vulkan_graphics::VulkanGraphicsCalibrationExecutor;
#[cfg(feature = "vulkan")]
use super::vulkan_ray::VulkanRayCalibrationExecutor;
#[cfg(feature = "vulkan")]
use super::vulkan_synchronization::VulkanSynchronizationCalibrationExecutor;
#[cfg(feature = "vulkan")]
use super::vulkan_transfer::VulkanTransferCalibrationExecutor;
#[cfg(feature = "vulkan")]
use super::vulkan_video::VulkanVideoCalibrationExecutor;
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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

static CALIBRATION_ARTIFACT_DIRECTORY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug)]
pub struct CalibrationRunnerOptions {
    pub cancelled: Arc<AtomicBool>,
    pub artifact_directory: PathBuf,
    #[cfg(feature = "vulkan")]
    pub vulkan_physical_device_index: Option<usize>,
}

impl Default for CalibrationRunnerOptions {
    fn default() -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or_default();
        let sequence = CALIBRATION_ARTIFACT_DIRECTORY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
            artifact_directory: std::env::temp_dir().join(format!(
                "nerve-hardware-calibration-artifacts-{}-{timestamp}-{sequence}",
                std::process::id(),
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
    let vulkan_physical_device_index = if uses_vulkan {
        Some(options.vulkan_physical_device_index.ok_or_else(|| {
            "a Vulkan calibration plan requires an explicit physical device index".to_string()
        })?)
    } else {
        None
    };
    for workload in &plan.workloads {
        if options.cancelled.load(Ordering::Relaxed) {
            return Ok(cancelled_run(
                plan,
                run_id,
                started_at,
                results,
                "calibration was cancelled".to_string(),
            ));
        }
        let result = (|| -> Result<HardwareCalibrationWorkloadResult, String> {
            Ok(match workload.executor {
                CalibrationExecutor::Cpu => run_cpu_workload(plan, workload, options)?,
                #[cfg(feature = "vulkan")]
                CalibrationExecutor::VulkanCompute => {
                    let construction_started = Instant::now();
                    let device = Rc::new(
                        VulkanComputeDevice::new_for_physical_device_index(
                            vulkan_physical_device_index
                                .expect("Vulkan physical device index was validated"),
                        )
                        .map_err(|error| {
                            format!("could not open Vulkan calibration device: {error}")
                        })?,
                    );
                    let executor = VulkanComputeCalibrationExecutor::new(
                        device,
                        options.artifact_directory.clone(),
                    )?;
                    let executor_construction_ns = elapsed_ns(construction_started);
                    include_executor_construction(
                        executor.run(plan, workload, &options.cancelled)?,
                        executor_construction_ns,
                    )
                }
                #[cfg(feature = "vulkan")]
                CalibrationExecutor::VulkanDgc => {
                    let construction_started = Instant::now();
                    let executor = VulkanDgcCalibrationExecutor::new(
                        vulkan_physical_device_index
                            .expect("Vulkan physical device index was validated"),
                        options.artifact_directory.clone(),
                    )?;
                    let executor_construction_ns = elapsed_ns(construction_started);
                    include_executor_construction(
                        executor.run(plan, workload, &options.cancelled)?,
                        executor_construction_ns,
                    )
                }
                #[cfg(feature = "vulkan")]
                CalibrationExecutor::VulkanTransfer => {
                    let construction_started = Instant::now();
                    let device = Rc::new(
                        VulkanComputeDevice::new_for_physical_device_index(
                            vulkan_physical_device_index
                                .expect("Vulkan physical device index was validated"),
                        )
                        .map_err(|error| {
                            format!("could not open Vulkan calibration device: {error}")
                        })?,
                    );
                    let executor = VulkanTransferCalibrationExecutor::new(device);
                    let executor_construction_ns = elapsed_ns(construction_started);
                    include_executor_construction(
                        executor.run(plan, workload, &options.cancelled)?,
                        executor_construction_ns,
                    )
                }
                #[cfg(feature = "vulkan")]
                CalibrationExecutor::VulkanGraphics => {
                    let construction_started = Instant::now();
                    let device = Rc::new(
                        VulkanComputeDevice::new_for_physical_device_index(
                            vulkan_physical_device_index
                                .expect("Vulkan physical device index was validated"),
                        )
                        .map_err(|error| {
                            format!("could not open Vulkan calibration device: {error}")
                        })?,
                    );
                    let executor = VulkanGraphicsCalibrationExecutor::new(
                        device,
                        vulkan_physical_device_index
                            .expect("Vulkan physical device index was validated"),
                        workload.operation != "texture_sampling",
                        options.artifact_directory.clone(),
                    )?;
                    let executor_construction_ns = elapsed_ns(construction_started);
                    include_executor_construction(
                        executor.run(plan, workload, &options.cancelled)?,
                        executor_construction_ns,
                    )
                }
                #[cfg(feature = "vulkan")]
                CalibrationExecutor::VulkanRay => {
                    let construction_started = Instant::now();
                    let executor = VulkanRayCalibrationExecutor::new(
                        vulkan_physical_device_index
                            .expect("Vulkan physical device index was validated"),
                        options.artifact_directory.clone(),
                    )?;
                    let executor_construction_ns = elapsed_ns(construction_started);
                    include_executor_construction(
                        executor.run(plan, workload, &options.cancelled)?,
                        executor_construction_ns,
                    )
                }
                #[cfg(feature = "vulkan")]
                CalibrationExecutor::VulkanSynchronization => {
                    let construction_started = Instant::now();
                    let executor = VulkanSynchronizationCalibrationExecutor::new(
                        vulkan_physical_device_index
                            .expect("Vulkan physical device index was validated"),
                        workload,
                    )?;
                    let executor_construction_ns = elapsed_ns(construction_started);
                    include_executor_construction(
                        executor.run(plan, workload, &options.cancelled)?,
                        executor_construction_ns,
                    )
                }
                #[cfg(feature = "vulkan")]
                CalibrationExecutor::VulkanVideo => {
                    let construction_started = Instant::now();
                    let executor = VulkanVideoCalibrationExecutor::new(
                        vulkan_physical_device_index
                            .expect("Vulkan physical device index was validated"),
                        options.artifact_directory.clone(),
                    )?;
                    let executor_construction_ns = elapsed_ns(construction_started);
                    include_executor_construction(
                        executor.run(plan, workload, &options.cancelled)?,
                        executor_construction_ns,
                    )
                }
                #[cfg(not(feature = "vulkan"))]
                executor => {
                    return Err(format!(
                        "calibration executor {executor:?} is not implemented by this build"
                    ));
                }
            })
        })();
        match result {
            Ok(result) => results.push(result),
            Err(error) if options.cancelled.load(Ordering::Relaxed) => {
                return Ok(cancelled_run(
                    plan,
                    run_id,
                    started_at,
                    results,
                    format!(
                        "calibration was cancelled during {}: {error}",
                        workload.workload_id
                    ),
                ));
            }
            Err(error) => return Err(error),
        }
    }
    let failed_workloads = results
        .iter()
        .filter(|result| result.status == CalibrationRunStatus::Failed)
        .map(|result| result.workload_id.clone())
        .collect::<Vec<_>>();
    let run = HardwareCalibrationRun {
        schema: HARDWARE_CALIBRATION_RUN_SCHEMA.to_string(),
        run_id,
        plan_id: plan.plan_id.clone(),
        hardware_profile_id: plan.hardware_profile_id.clone(),
        status: if failed_workloads.is_empty() {
            CalibrationRunStatus::Completed
        } else {
            CalibrationRunStatus::Failed
        },
        started_at,
        finished_at: Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true),
        workloads: results,
        diagnostics: if failed_workloads.is_empty() {
            Vec::new()
        } else {
            vec![format!(
                "calibration validation failed for workloads: {}",
                failed_workloads.join(", ")
            )]
        },
    };
    run.validate()?;
    Ok(run)
}

fn cancelled_run(
    plan: &HardwareCalibrationPlan,
    run_id: String,
    started_at: String,
    workloads: Vec<HardwareCalibrationWorkloadResult>,
    diagnostic: String,
) -> HardwareCalibrationRun {
    HardwareCalibrationRun {
        schema: HARDWARE_CALIBRATION_RUN_SCHEMA.to_string(),
        run_id,
        plan_id: plan.plan_id.clone(),
        hardware_profile_id: plan.hardware_profile_id.clone(),
        status: CalibrationRunStatus::Cancelled,
        started_at,
        finished_at: Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true),
        workloads,
        diagnostics: vec![diagnostic],
    }
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
        thermal_millidegrees_celsius: maximum_cpu_temperature_millidegrees(),
        valid: true,
    })
}

#[cfg(feature = "vulkan")]
fn include_executor_construction(
    mut result: HardwareCalibrationWorkloadResult,
    executor_construction_ns: u64,
) -> HardwareCalibrationWorkloadResult {
    result.construction_duration_ns = result
        .construction_duration_ns
        .saturating_add(executor_construction_ns);
    result
}
