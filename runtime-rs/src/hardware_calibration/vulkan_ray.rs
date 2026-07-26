use super::sampling::collect_adaptive_samples;
use super::schema::{
    CalibrationRunStatus, CalibrationSamplePhase, CalibrationValidationResult,
    CalibrationValidationStatus, HardwareCalibrationPlan, HardwareCalibrationSample,
    HardwareCalibrationWorkload, HardwareCalibrationWorkloadResult,
};
use super::shader_compiler::{compile_calibration_shader, persist_calibration_artifact};
use super::telemetry::{elapsed_ns, maximum_pci_temperature_millidegrees};
use super::vulkan_specialized::{
    PreparedRayCalibration, SpecializedVulkanContext, SpecializedVulkanRequirements,
    ray_query_shader,
};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub(super) struct VulkanRayCalibrationExecutor {
    context: Rc<SpecializedVulkanContext>,
    artifact_directory: PathBuf,
}

impl VulkanRayCalibrationExecutor {
    pub(super) fn new(
        physical_device_index: usize,
        artifact_directory: PathBuf,
    ) -> Result<Self, String> {
        let context = SpecializedVulkanContext::new(
            physical_device_index,
            SpecializedVulkanRequirements {
                ray_query: true,
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
        let primitive_count = regime_u32(workload, "primitives")?;
        let ray_count = regime_u32(workload, "rays")?;
        let construction_started = Instant::now();
        let scene_index = workload
            .artifacts
            .iter()
            .position(|artifact| artifact.kind == "procedural_ray_scene")
            .ok_or_else(|| "ray workload has no procedural scene artifact".to_string())?;
        let scene_document = serde_json::to_vec(&serde_json::json!({
            "schema": "nerve.calibration.procedural_ray_scene.v1",
            "primitive": "axis_aligned_bounding_box",
            "primitive_count": primitive_count,
            "generator": "deterministic_grid_32",
        }))
        .map_err(|error| format!("could not serialize ray-scene artifact: {error}"))?;
        let scene_artifact = persist_calibration_artifact(
            workload,
            scene_index,
            "json",
            &scene_document,
            &self.artifact_directory,
        )?;
        let (query_spirv, query_artifact) = if workload.operation == "ray_query_traversal" {
            let shader_index = workload
                .artifacts
                .iter()
                .position(|artifact| artifact.kind == "spirv_compute")
                .ok_or_else(|| "ray-query workload has no compute artifact".to_string())?;
            let (spirv, artifact) = compile_calibration_shader(
                workload,
                shader_index,
                ray_query_shader(),
                "comp",
                &self.artifact_directory,
            )?;
            (Some(spirv), Some(artifact))
        } else {
            (None, None)
        };
        let prepared = PreparedRayCalibration::new(
            Rc::clone(&self.context),
            &workload.operation,
            primitive_count,
            ray_count,
            query_spirv.as_deref(),
        )?;
        let construction_duration_ns = elapsed_ns(construction_started);
        let mut prepared = PreparedRayState {
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
        let mut artifacts = vec![scene_artifact];
        artifacts.extend(query_artifact);
        artifacts.sort_by(|left, right| left.name.cmp(&right.name));
        Ok(HardwareCalibrationWorkloadResult {
            workload_id: workload.workload_id.clone(),
            status: if validation_passed {
                CalibrationRunStatus::Completed
            } else {
                CalibrationRunStatus::Failed
            },
            construction_duration_ns,
            artifacts,
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

struct PreparedRayState {
    prepared: PreparedRayCalibration,
    pci_address: Option<String>,
}

impl PreparedRayState {
    fn measure(
        &mut self,
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
                return Err("calibration was cancelled during ray execution".to_string());
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

fn regime_u32(workload: &HardwareCalibrationWorkload, name: &str) -> Result<u32, String> {
    workload
        .regime
        .get(name)
        .ok_or_else(|| format!("ray workload has no {name} regime"))?
        .parse::<u32>()
        .map_err(|error| format!("invalid ray {name}: {error}"))
}
