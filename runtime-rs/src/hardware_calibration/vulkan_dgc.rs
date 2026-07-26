use super::schema::{
    CalibrationRunStatus, CalibrationSamplePhase, CalibrationValidationResult,
    CalibrationValidationStatus, HardwareCalibrationPlan, HardwareCalibrationSample,
    HardwareCalibrationWorkload, HardwareCalibrationWorkloadResult,
};
use super::shader_compiler::compile_calibration_shader;
use super::telemetry::{elapsed_ns, maximum_pci_temperature_millidegrees};
use super::vulkan_specialized::{
    PreparedDeviceGeneratedCommands, SpecializedVulkanContext, SpecializedVulkanRequirements,
    device_generated_commands_shader,
};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub(super) struct VulkanDgcCalibrationExecutor {
    context: Rc<SpecializedVulkanContext>,
    artifact_directory: PathBuf,
}

impl VulkanDgcCalibrationExecutor {
    pub(super) fn new(
        physical_device_index: usize,
        artifact_directory: PathBuf,
    ) -> Result<Self, String> {
        let context = SpecializedVulkanContext::new(
            physical_device_index,
            SpecializedVulkanRequirements {
                device_generated_commands: true,
                ..Default::default()
            },
        )?;
        Ok(Self {
            context,
            artifact_directory,
        })
    }

    pub(super) fn run(
        &self,
        plan: &HardwareCalibrationPlan,
        workload: &HardwareCalibrationWorkload,
        cancelled: &Arc<AtomicBool>,
    ) -> Result<HardwareCalibrationWorkloadResult, String> {
        if workload.operation != "device_generated_commands" {
            return Err(format!(
                "DGC calibrator does not implement {:?}",
                workload.operation
            ));
        }
        if workload.artifacts.len() != 1 {
            return Err("DGC workload must declare exactly one shader artifact".to_string());
        }
        let dispatch_count = workload
            .regime
            .get("dispatch_count")
            .ok_or_else(|| "DGC workload has no dispatch_count".to_string())?
            .parse::<u32>()
            .map_err(|error| format!("invalid DGC dispatch count: {error}"))?;
        let construction_started = Instant::now();
        let (spirv, artifact) = compile_calibration_shader(
            workload,
            0,
            device_generated_commands_shader(),
            "comp",
            &self.artifact_directory,
        )?;
        let prepared =
            PreparedDeviceGeneratedCommands::new(Rc::clone(&self.context), &spirv, dispatch_count)?;
        let construction_duration_ns = elapsed_ns(construction_started);
        let prepared = PreparedDgcState {
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
            artifacts: vec![artifact],
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

struct PreparedDgcState {
    prepared: PreparedDeviceGeneratedCommands,
    pci_address: Option<String>,
}

impl PreparedDgcState {
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
                return Err("calibration was cancelled during DGC execution".to_string());
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
