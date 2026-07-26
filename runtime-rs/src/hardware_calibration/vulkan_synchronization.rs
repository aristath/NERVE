use super::schema::{
    CalibrationRunStatus, CalibrationSamplePhase, CalibrationValidationResult,
    CalibrationValidationStatus, HardwareCalibrationPlan, HardwareCalibrationSample,
    HardwareCalibrationWorkload, HardwareCalibrationWorkloadResult,
};
use super::vulkan_specialized::{
    PreparedSynchronizationCalibration, SpecializedVulkanContext, SpecializedVulkanRequirements,
};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub(super) struct VulkanSynchronizationCalibrationExecutor {
    context: Rc<SpecializedVulkanContext>,
}

impl VulkanSynchronizationCalibrationExecutor {
    pub(super) fn new(physical_device_index: usize) -> Result<Self, String> {
        Ok(Self {
            context: SpecializedVulkanContext::new(
                physical_device_index,
                SpecializedVulkanRequirements {
                    compute: true,
                    ..Default::default()
                },
            )?,
        })
    }

    pub(super) fn run(
        &self,
        plan: &HardwareCalibrationPlan,
        workload: &HardwareCalibrationWorkload,
        cancelled: &Arc<AtomicBool>,
    ) -> Result<HardwareCalibrationWorkloadResult, String> {
        if workload.operation != "synchronization_round_trip" {
            return Err(format!(
                "synchronization calibrator does not implement {:?}",
                workload.operation
            ));
        }
        if !workload.artifacts.is_empty() {
            return Err("synchronization workloads must not declare shader artifacts".to_string());
        }
        let primitive = workload
            .regime
            .get("primitive")
            .ok_or_else(|| "synchronization workload has no primitive".to_string())?;
        let round_trips = workload
            .regime
            .get("round_trips")
            .ok_or_else(|| "synchronization workload has no round_trips".to_string())?
            .parse::<u32>()
            .map_err(|error| format!("invalid synchronization round_trips: {error}"))?;
        let construction_started = Instant::now();
        let prepared = PreparedSynchronizationCalibration::new(
            Rc::clone(&self.context),
            primitive,
            round_trips,
        )?;
        let construction_duration_ns = elapsed_ns(construction_started);
        let prepared = PreparedSynchronizationState {
            prepared,
            pci_address: self.context.pci_address().map(str::to_string),
        };
        let mut samples = Vec::new();
        for _ in 0..plan.policy.warmup_iterations {
            samples.push(prepared.measure(
                CalibrationSamplePhase::Warmup,
                None,
                plan.policy.minimum_sample_duration_ns,
                cancelled,
                samples.len(),
            )?);
        }
        for _ in 0..plan.policy.steady_iterations {
            samples.push(prepared.measure(
                CalibrationSamplePhase::Steady,
                None,
                plan.policy.minimum_sample_duration_ns,
                cancelled,
                samples.len(),
            )?);
        }
        for window_index in 0..plan.policy.sustained_window_count {
            samples.push(
                prepared.measure(
                    CalibrationSamplePhase::Sustained,
                    Some(window_index),
                    plan.policy
                        .sustained_window_duration_ms
                        .saturating_mul(1_000_000),
                    cancelled,
                    samples.len(),
                )?,
            );
        }
        let observed_digest = prepared.prepared.observed_digest()?;
        let validation_passed = workload
            .validation
            .expected_digest
            .as_ref()
            .is_none_or(|expected| expected == &observed_digest);
        Ok(HardwareCalibrationWorkloadResult {
            workload_id: workload.workload_id.clone(),
            status: if validation_passed {
                CalibrationRunStatus::Completed
            } else {
                CalibrationRunStatus::Failed
            },
            construction_duration_ns,
            artifacts: Vec::new(),
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
            counters: Default::default(),
            diagnostics: Vec::new(),
        })
    }
}

struct PreparedSynchronizationState {
    prepared: PreparedSynchronizationCalibration,
    pci_address: Option<String>,
}

impl PreparedSynchronizationState {
    fn measure(
        &self,
        phase: CalibrationSamplePhase,
        window_index: Option<usize>,
        minimum_duration_ns: u64,
        cancelled: &Arc<AtomicBool>,
        sample_index: usize,
    ) -> Result<HardwareCalibrationSample, String> {
        let started = Instant::now();
        let target = Duration::from_nanos(minimum_duration_ns);
        let mut iterations = 0u64;
        let mut device_duration_ns = 0u64;
        while started.elapsed() < target || iterations == 0 {
            if cancelled.load(Ordering::Relaxed) {
                return Err("calibration was cancelled during synchronization".to_string());
            }
            device_duration_ns = device_duration_ns.saturating_add(self.prepared.run()?);
            iterations = iterations.saturating_add(1);
        }
        Ok(HardwareCalibrationSample {
            sample_index,
            phase,
            duration_ns: elapsed_ns(started),
            device_duration_ns: Some(device_duration_ns),
            iterations,
            window_index,
            thermal_millidegrees_celsius: maximum_device_temperature_millidegrees(
                self.pci_address.as_deref(),
            ),
            valid: true,
        })
    }
}

fn elapsed_ns(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn maximum_device_temperature_millidegrees(pci_address: Option<&str>) -> Option<u64> {
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
