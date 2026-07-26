use super::schema::{
    CalibrationRunStatus, CalibrationSamplePhase, CalibrationValidationResult,
    CalibrationValidationStatus, HardwareCalibrationPlan, HardwareCalibrationSample,
    HardwareCalibrationWorkload, HardwareCalibrationWorkloadResult,
};
use super::shader_compiler::compile_calibration_shader;
use super::telemetry::{elapsed_ns, maximum_pci_temperature_millidegrees};
use super::vulkan_specialized::{
    PreparedFixedGraphics, SpecializedVulkanContext, SpecializedVulkanRequirements,
    fixed_graphics_fragment_shader, fixed_graphics_vertex_shader,
};
use crate::vulkan_compute::{VulkanComputeDevice, VulkanTextureCalibration};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub(super) struct VulkanGraphicsCalibrationExecutor {
    device: Option<Rc<VulkanComputeDevice>>,
    fixed_graphics_context: Option<Rc<SpecializedVulkanContext>>,
    artifact_directory: PathBuf,
}

impl VulkanGraphicsCalibrationExecutor {
    pub(super) fn new(
        device: Option<Rc<VulkanComputeDevice>>,
        physical_device_index: usize,
        needs_fixed_graphics: bool,
        artifact_directory: PathBuf,
    ) -> Result<Self, String> {
        std::fs::create_dir_all(&artifact_directory).map_err(|error| {
            format!(
                "could not create graphics calibration artifacts {artifact_directory:?}: {error}"
            )
        })?;
        let fixed_graphics_context = needs_fixed_graphics
            .then(|| {
                SpecializedVulkanContext::new(
                    physical_device_index,
                    SpecializedVulkanRequirements {
                        graphics: true,
                        ..Default::default()
                    },
                )
            })
            .transpose()?;
        Ok(Self {
            device,
            fixed_graphics_context,
            artifact_directory,
        })
    }

    pub(super) fn run(
        &self,
        plan: &HardwareCalibrationPlan,
        workload: &HardwareCalibrationWorkload,
        cancelled: &Arc<AtomicBool>,
    ) -> Result<HardwareCalibrationWorkloadResult, String> {
        if workload.operation == "texture_sampling" {
            self.run_texture(plan, workload, cancelled)
        } else {
            self.run_fixed_graphics(plan, workload, cancelled)
        }
    }

    fn run_texture(
        &self,
        plan: &HardwareCalibrationPlan,
        workload: &HardwareCalibrationWorkload,
        cancelled: &Arc<AtomicBool>,
    ) -> Result<HardwareCalibrationWorkloadResult, String> {
        let construction_started = Instant::now();
        let device = self.device.as_ref().ok_or_else(|| {
            "texture workload reached an executor without a compute device".to_string()
        })?;
        let source = texture_shader_source();
        let (spirv, artifact) =
            compile_calibration_shader(workload, 0, &source, "comp", &self.artifact_directory)?;
        let output_count = u32::try_from(workload.work.items_per_iteration)
            .map_err(|_| "texture calibration item count exceeds u32".to_string())?;
        let texture = device
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
        let mut prepared = PreparedTexture {
            texture,
            pci_address: device.pci_address().map(str::to_string),
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

    fn run_fixed_graphics(
        &self,
        plan: &HardwareCalibrationPlan,
        workload: &HardwareCalibrationWorkload,
        cancelled: &Arc<AtomicBool>,
    ) -> Result<HardwareCalibrationWorkloadResult, String> {
        if workload.artifacts.len() != 2 {
            return Err(format!(
                "fixed graphics workload {} must declare vertex and fragment artifacts",
                workload.workload_id
            ));
        }
        let context = self.fixed_graphics_context.as_ref().ok_or_else(|| {
            "fixed graphics workload reached an executor without a graphics context".to_string()
        })?;
        let construction_started = Instant::now();
        let vertex_index = workload
            .artifacts
            .iter()
            .position(|artifact| artifact.kind == "spirv_vertex")
            .ok_or_else(|| "fixed graphics workload has no vertex artifact".to_string())?;
        let fragment_index = workload
            .artifacts
            .iter()
            .position(|artifact| artifact.kind == "spirv_fragment")
            .ok_or_else(|| "fixed graphics workload has no fragment artifact".to_string())?;
        let (vertex_spirv, vertex_artifact) = compile_calibration_shader(
            workload,
            vertex_index,
            fixed_graphics_vertex_shader(),
            "vert",
            &self.artifact_directory,
        )?;
        let (fragment_spirv, fragment_artifact) = compile_calibration_shader(
            workload,
            fragment_index,
            fixed_graphics_fragment_shader(),
            "frag",
            &self.artifact_directory,
        )?;
        let (width, height) = parse_extent(
            workload
                .regime
                .get("render_target")
                .map(String::as_str)
                .unwrap_or("4096x4096"),
        )?;
        let overdraw = workload
            .regime
            .get("overdraw")
            .map(String::as_str)
            .unwrap_or("1")
            .parse::<u32>()
            .map_err(|error| format!("invalid fixed graphics overdraw: {error}"))?;
        let prepared = PreparedFixedGraphics::new(
            Rc::clone(context),
            &workload.operation,
            &vertex_spirv,
            &fragment_spirv,
            width,
            height,
            overdraw,
        )?;
        let construction_duration_ns = elapsed_ns(construction_started);
        let mut prepared = PreparedFixedGraphicsState {
            prepared,
            pci_address: context.pci_address().map(str::to_string),
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
        let mut artifacts = vec![vertex_artifact, fragment_artifact];
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

struct PreparedTexture {
    texture: VulkanTextureCalibration,
    pci_address: Option<String>,
}

struct PreparedFixedGraphicsState {
    prepared: PreparedFixedGraphics,
    pci_address: Option<String>,
}

impl PreparedFixedGraphicsState {
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
        let mut device_duration_ns = 0u64;
        while started.elapsed() < target || iterations == 0 {
            if cancelled.load(Ordering::Relaxed) {
                return Err("calibration was cancelled during fixed graphics".to_string());
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
            thermal_millidegrees_celsius: maximum_pci_temperature_millidegrees(
                self.pci_address.as_deref(),
            ),
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

fn parse_extent(value: &str) -> Result<(u32, u32), String> {
    let (width, height) = value
        .split_once('x')
        .ok_or_else(|| format!("invalid graphics extent {value:?}"))?;
    let width = width
        .parse::<u32>()
        .map_err(|error| format!("invalid graphics width: {error}"))?;
    let height = height
        .parse::<u32>()
        .map_err(|error| format!("invalid graphics height: {error}"))?;
    if width == 0 || height == 0 {
        return Err("graphics extent must be nonzero".to_string());
    }
    Ok((width, height))
}
