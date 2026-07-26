use super::sampling::collect_adaptive_samples;
use super::schema::{
    CalibrationRunStatus, CalibrationSamplePhase, CalibrationValidationResult,
    CalibrationValidationStatus, HardwareCalibrationPlan, HardwareCalibrationSample,
    HardwareCalibrationWorkload, HardwareCalibrationWorkloadResult,
};
use super::telemetry::{elapsed_ns, maximum_pci_temperature_millidegrees};
use super::vulkan_specialized::{
    PreparedQueueContention, PreparedSynchronizationCalibration, SpecializedVulkanContext,
    SpecializedVulkanRequirements,
};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub(super) struct VulkanSynchronizationCalibrationExecutor {
    context: Rc<SpecializedVulkanContext>,
}

impl VulkanSynchronizationCalibrationExecutor {
    pub(super) fn new(
        physical_device_index: usize,
        workload: &HardwareCalibrationWorkload,
    ) -> Result<Self, String> {
        let compute_queue_count = workload
            .regime
            .get("queue_count")
            .map(|value| {
                value
                    .parse::<u32>()
                    .map_err(|error| format!("invalid synchronization queue_count: {error}"))
            })
            .transpose()?
            .unwrap_or(1);
        Ok(Self {
            context: SpecializedVulkanContext::new(
                physical_device_index,
                SpecializedVulkanRequirements {
                    compute: true,
                    compute_queue_count,
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
        if !matches!(
            workload.operation.as_str(),
            "synchronization_round_trip" | "queue_contention"
        ) {
            return Err(format!(
                "synchronization calibrator does not implement {:?}",
                workload.operation
            ));
        }
        if !workload.artifacts.is_empty() {
            return Err("synchronization workloads must not declare shader artifacts".to_string());
        }
        let construction_started = Instant::now();
        let prepared = if workload.operation == "queue_contention" {
            let queue_count = regime_u32(workload, "queue_count")?;
            let streams = regime_u32(workload, "streams")?;
            PreparedSynchronizationWorkload::Queue(PreparedQueueContention::new(
                Rc::clone(&self.context),
                queue_count,
                streams,
            )?)
        } else {
            PreparedSynchronizationWorkload::Primitive(PreparedSynchronizationCalibration::new(
                Rc::clone(&self.context),
                workload
                    .regime
                    .get("primitive")
                    .ok_or_else(|| "synchronization workload has no primitive".to_string())?,
                regime_u32(workload, "round_trips")?,
            )?)
        };
        let construction_duration_ns = elapsed_ns(construction_started);
        let prepared = PreparedSynchronizationState {
            prepared,
            pci_address: self.context.pci_address().map(str::to_string),
        };
        let mut samples = Vec::new();
        collect_adaptive_samples(&mut samples, &plan.policy, |phase, sample_index| {
            prepared.measure(
                phase,
                None,
                plan.policy.minimum_sample_duration_ns,
                cancelled,
                sample_index,
            )
        })?;
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
    prepared: PreparedSynchronizationWorkload,
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
            thermal_millidegrees_celsius: maximum_pci_temperature_millidegrees(
                self.pci_address.as_deref(),
            ),
            valid: true,
        })
    }
}

enum PreparedSynchronizationWorkload {
    Primitive(PreparedSynchronizationCalibration),
    Queue(PreparedQueueContention),
}

impl PreparedSynchronizationWorkload {
    fn run(&self) -> Result<u64, String> {
        match self {
            Self::Primitive(prepared) => prepared.run(),
            Self::Queue(prepared) => prepared.run(),
        }
    }

    fn observed_digest(&self) -> Result<String, String> {
        match self {
            Self::Primitive(prepared) => prepared.observed_digest(),
            Self::Queue(prepared) => prepared.observed_digest(),
        }
    }
}

fn regime_u32(workload: &HardwareCalibrationWorkload, name: &str) -> Result<u32, String> {
    workload
        .regime
        .get(name)
        .ok_or_else(|| format!("synchronization workload has no {name}"))?
        .parse::<u32>()
        .map_err(|error| format!("invalid synchronization {name}: {error}"))
}
