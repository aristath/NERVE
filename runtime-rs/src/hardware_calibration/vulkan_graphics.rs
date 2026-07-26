use super::schema::{
    CalibrationRunStatus, CalibrationSamplePhase, CalibrationValidationResult,
    CalibrationValidationStatus, HardwareCalibrationPlan, HardwareCalibrationSample,
    HardwareCalibrationWorkload, HardwareCalibrationWorkloadResult,
};
use super::shader_compiler::compile_calibration_shader;
use crate::vulkan_compute::{VulkanComputeDevice, VulkanTextureCalibration};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub(super) struct VulkanGraphicsCalibrationExecutor {
    device: Rc<VulkanComputeDevice>,
    artifact_directory: PathBuf,
}

impl VulkanGraphicsCalibrationExecutor {
    pub(super) fn new(
        device: Rc<VulkanComputeDevice>,
        artifact_directory: PathBuf,
    ) -> Result<Self, String> {
        std::fs::create_dir_all(&artifact_directory).map_err(|error| {
            format!(
                "could not create graphics calibration artifacts {artifact_directory:?}: {error}"
            )
        })?;
        Ok(Self {
            device,
            artifact_directory,
        })
    }

    pub(super) fn run(
        &self,
        plan: &HardwareCalibrationPlan,
        workload: &HardwareCalibrationWorkload,
        cancelled: &Arc<AtomicBool>,
    ) -> Result<HardwareCalibrationWorkloadResult, String> {
        if workload.operation != "texture_sampling" {
            return Err(format!(
                "Vulkan graphics calibrator does not implement {:?}",
                workload.operation
            ));
        }
        let construction_started = Instant::now();
        let source = texture_shader_source();
        let (spirv, artifact) =
            compile_calibration_shader(workload, 0, &source, "comp", &self.artifact_directory)?;
        let output_count = u32::try_from(workload.work.items_per_iteration)
            .map_err(|_| "texture calibration item count exceeds u32".to_string())?;
        let texture = self
            .device
            .create_texture_calibration(
                &spirv,
                workload
                    .regime
                    .get("filter")
                    .is_some_and(|value| value == "linear"),
                4096,
                4096,
                output_count,
            )
            .map_err(|error| format!("could not construct texture calibration: {error}"))?;
        let construction_duration_ns = elapsed_ns(construction_started);
        let mut prepared = PreparedTexture { texture };
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
        let observed_digest = prepared.observed_digest()?;
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

struct PreparedTexture {
    texture: VulkanTextureCalibration,
}

impl PreparedTexture {
    fn measure(
        &mut self,
        phase: CalibrationSamplePhase,
        window_index: Option<usize>,
        minimum_duration_ns: u64,
        cancelled: &Arc<AtomicBool>,
        sample_index: usize,
    ) -> Result<HardwareCalibrationSample, String> {
        let target = Duration::from_nanos(minimum_duration_ns);
        let started = Instant::now();
        let mut iterations = 0u64;
        while started.elapsed() < target || iterations == 0 {
            if cancelled.load(Ordering::Relaxed) {
                return Err("calibration was cancelled during texture sampling".to_string());
            }
            self.texture
                .run_for(Duration::from_secs(1))
                .map_err(|error| format!("texture calibration failed: {error}"))?;
            iterations = iterations.saturating_add(1);
        }
        Ok(HardwareCalibrationSample {
            sample_index,
            phase,
            duration_ns: elapsed_ns(started),
            device_duration_ns: None,
            iterations,
            window_index,
            thermal_millidegrees_celsius: None,
            valid: true,
        })
    }

    fn observed_digest(&self) -> Result<String, String> {
        let output = self
            .texture
            .output_bytes(4096)
            .map_err(|error| format!("could not validate sampled texture output: {error}"))?;
        if output.iter().all(|byte| *byte == 0) {
            return Err("texture calibration produced an unchanged output".to_string());
        }
        Ok(format!(
            "nerve.calibration_output_sha256.v1:{:x}",
            Sha256::digest(output)
        ))
    }
}

fn texture_shader_source() -> String {
    r#"#version 460
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) uniform sampler2D source_texture;
layout(set = 0, binding = 1) writeonly buffer OutputWords { uint words[]; } output_words;
layout(push_constant) uniform Control { uint output_count; } control;
uint mix_bits(uint value) {
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    return value ^ (value >> 16u);
}
void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= control.output_count) { return; }
    uint x = mix_bits(index);
    uint y = mix_bits(index ^ 0x9e3779b9u);
    vec2 coordinates = (vec2(x & 4095u, y & 4095u) + vec2(0.375, 0.625)) / 4096.0;
    vec4 sampled = texture(source_texture, coordinates);
    output_words.words[index] = floatBitsToUint(dot(sampled, vec4(1.0, 2.0, 3.0, 4.0)));
}
"#
    .to_string()
}

fn elapsed_ns(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}
