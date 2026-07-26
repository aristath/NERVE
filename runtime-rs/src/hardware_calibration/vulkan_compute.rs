use super::sampling::collect_adaptive_samples;
use super::schema::{
    CalibrationArtifactRecord, CalibrationRunStatus, CalibrationSamplePhase,
    CalibrationValidationResult, CalibrationValidationStatus, HardwareCalibrationPlan,
    HardwareCalibrationSample, HardwareCalibrationWorkload, HardwareCalibrationWorkloadResult,
};
use super::shader_compiler::compile_calibration_shader;
use super::telemetry::{elapsed_ns, maximum_pci_temperature_millidegrees};
use super::vulkan_compute_shaders::compute_shader_source;
use crate::vulkan_compute::{
    VulkanComputeDevice, VulkanResidentBuffer, VulkanResidentKernelBufferAccess,
    VulkanResidentKernelBufferBinding, VulkanResidentKernelDispatch, VulkanResidentKernelSequence,
    VulkanResidentKernelSequenceStep,
};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub(super) struct VulkanComputeCalibrationExecutor {
    device: Rc<VulkanComputeDevice>,
    artifact_directory: PathBuf,
}

impl VulkanComputeCalibrationExecutor {
    pub(super) fn new(
        device: Rc<VulkanComputeDevice>,
        artifact_directory: PathBuf,
    ) -> Result<Self, String> {
        fs::create_dir_all(&artifact_directory).map_err(|error| {
            format!(
                "could not create Vulkan calibration artifact directory {artifact_directory:?}: {error}"
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
        let construction_started = Instant::now();
        let mut prepared =
            PreparedVulkanComputeWorkload::new(&self.device, workload, &self.artifact_directory)?;
        let construction_duration_ns = elapsed_ns(construction_started);
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
        let sustained_duration_ns = plan
            .policy
            .sustained_window_duration_ms
            .saturating_mul(1_000_000);
        for window_index in 0..plan.policy.sustained_window_count {
            samples.push(prepared.measure(
                CalibrationSamplePhase::Sustained,
                Some(window_index),
                sustained_duration_ns,
                cancelled,
                samples.len(),
            )?);
        }
        let observed_digest = prepared.observed_digest()?;
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
            artifacts: prepared.artifacts,
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
                vec!["Vulkan calibration output digest did not match its plan".to_string()]
            },
        })
    }
}

struct PreparedVulkanComputeWorkload<'a> {
    device: &'a VulkanComputeDevice,
    _input: VulkanResidentBuffer,
    output: VulkanResidentBuffer,
    _dispatch: VulkanResidentKernelDispatch,
    _indirect: Option<VulkanResidentBuffer>,
    sequence: VulkanResidentKernelSequence,
    artifacts: Vec<CalibrationArtifactRecord>,
}

impl<'a> PreparedVulkanComputeWorkload<'a> {
    fn new(
        device: &'a VulkanComputeDevice,
        workload: &HardwareCalibrationWorkload,
        artifact_directory: &Path,
    ) -> Result<Self, String> {
        let source = compute_shader_source(workload)?;
        if workload.artifacts.len() != 1 {
            return Err(format!(
                "Vulkan compute workload {} must declare exactly one shader artifact",
                workload.workload_id
            ));
        }
        let (spirv_words, artifact) =
            compile_calibration_shader(workload, 0, &source, "comp", artifact_directory)?;
        let artifacts = vec![artifact];
        let (input_bytes, output_bytes, output_count, workgroup_count) =
            workload_buffer_shape(workload)?;
        let input = device
            .create_resident_buffer(input_bytes)
            .map_err(|error| format!("could not allocate calibration input: {error}"))?;
        let output = device
            .create_resident_buffer(output_bytes)
            .map_err(|error| format!("could not allocate calibration output: {error}"))?;
        input
            .write_bytes(&deterministic_bytes(input_bytes))
            .map_err(|error| format!("could not initialize calibration input: {error}"))?;
        output
            .write_bytes(&vec![0; output_bytes])
            .map_err(|error| format!("could not initialize calibration output: {error}"))?;
        let dispatch = device
            .create_resident_kernel_dispatch(
                &spirv_words,
                &[
                    VulkanResidentKernelBufferBinding::new(0, &input, input_bytes)
                        .with_access(VulkanResidentKernelBufferAccess::Read),
                    VulkanResidentKernelBufferBinding::new(1, &output, output_bytes)
                        .with_access(VulkanResidentKernelBufferAccess::ReadWrite),
                ],
                workgroup_count,
                if workload.operation == "cooperative_matrix_multiply" {
                    64
                } else if is_scheduling_operation(&workload.operation) {
                    1
                } else {
                    256
                },
                4,
            )
            .map_err(|error| format!("could not construct calibration dispatch: {error}"))?;
        let sequence = device
            .create_timestamped_resident_kernel_sequence()
            .map_err(|error| format!("could not construct calibration sequence: {error}"))?;
        let indirect = if workload.operation == "indirect_work_generation" {
            let buffer = device
                .create_resident_buffer(12)
                .map_err(|error| format!("could not allocate indirect-dispatch buffer: {error}"))?;
            let mut command = Vec::with_capacity(12);
            command.extend_from_slice(&workgroup_count.to_le_bytes());
            command.extend_from_slice(&1u32.to_le_bytes());
            command.extend_from_slice(&1u32.to_le_bytes());
            buffer
                .write_bytes(&command)
                .map_err(|error| format!("could not initialize indirect dispatch: {error}"))?;
            Some(buffer)
        } else {
            None
        };
        let dispatch_count = workload
            .regime
            .get("dispatch_count")
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1);
        let push_constants = output_count.to_le_bytes();
        let steps = (0..dispatch_count)
            .map(|_| {
                if let Some(indirect) = &indirect {
                    VulkanResidentKernelSequenceStep::new_indirect(
                        &dispatch,
                        &push_constants,
                        indirect,
                        0,
                    )
                    .map_err(|error| format!("could not bind indirect dispatch: {error}"))
                } else {
                    Ok(VulkanResidentKernelSequenceStep::new(
                        &dispatch,
                        &push_constants,
                    ))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        device
            .record_resident_kernel_sequence(&sequence, &steps)
            .map_err(|error| format!("could not record calibration sequence: {error}"))?;
        Ok(Self {
            device,
            _input: input,
            output,
            _dispatch: dispatch,
            _indirect: indirect,
            sequence,
            artifacts,
        })
    }

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
                return Err("calibration was cancelled during a Vulkan sample".to_string());
            }
            let device_duration = self
                .device
                .run_timestamped_recorded_resident_kernel_sequence_for(
                    &self.sequence,
                    Duration::from_secs(1),
                )
                .map_err(|error| format!("Vulkan calibration dispatch failed: {error}"))?;
            device_duration_ns = device_duration_ns.saturating_add(device_duration);
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
                self.device.pci_address(),
            ),
            valid: true,
        })
    }

    fn observed_digest(&self) -> Result<String, String> {
        let sample_length = self.output.byte_capacity().min(4096);
        let bytes = self
            .output
            .read_bytes(sample_length)
            .map_err(|error| format!("could not validate calibration output: {error}"))?;
        if bytes.iter().all(|byte| *byte == 0) {
            return Err("Vulkan calibration produced an unchanged all-zero output".to_string());
        }
        Ok(format!(
            "nerve.calibration_output_sha256.v1:{:x}",
            Sha256::digest(bytes)
        ))
    }
}

fn workload_buffer_shape(
    workload: &HardwareCalibrationWorkload,
) -> Result<(usize, usize, u32, u32), String> {
    if is_scheduling_operation(&workload.operation) {
        return Ok((4, 4, 1, 1));
    }
    if workload.operation == "cooperative_matrix_multiply" {
        let tiles = u32::try_from(workload.work.items_per_iteration)
            .map_err(|_| "cooperative-matrix tile count exceeds u32".to_string())?;
        let format_bytes = match workload.regime.get("format").map(String::as_str) {
            Some("f8_e4m3") => 1usize,
            Some("f16" | "bf16") => 2usize,
            other => return Err(format!("unsupported cooperative-matrix format {other:?}")),
        };
        let input_bytes = usize::try_from(tiles)
            .ok()
            .and_then(|tiles| tiles.checked_mul(512))
            .and_then(|elements| elements.checked_mul(format_bytes))
            .ok_or_else(|| "cooperative-matrix input size overflowed".to_string())?;
        let output_bytes = usize::try_from(tiles)
            .ok()
            .and_then(|tiles| tiles.checked_mul(256))
            .and_then(|elements| elements.checked_mul(4))
            .ok_or_else(|| "cooperative-matrix output size overflowed".to_string())?;
        return Ok((input_bytes, output_bytes, tiles, tiles));
    }
    let output_count = u32::try_from(workload.work.items_per_iteration)
        .map_err(|_| "Vulkan workload item count exceeds u32".to_string())?;
    let output_bytes = usize::try_from(output_count)
        .ok()
        .and_then(|count| count.checked_mul(4))
        .ok_or_else(|| "Vulkan calibration output size overflowed".to_string())?;
    let input_words_per_output = match workload.operation.as_str() {
        "shader_scalar" | "shader_vector" => 4,
        "packed_dot_product" => 2,
        _ => 1,
    };
    let input_bytes = output_bytes
        .checked_mul(input_words_per_output)
        .ok_or_else(|| "Vulkan calibration input size overflowed".to_string())?;
    let workgroup_count = output_count.div_ceil(256);
    Ok((input_bytes, output_bytes, output_count, workgroup_count))
}

fn deterministic_bytes(byte_count: usize) -> Vec<u8> {
    (0..byte_count)
        .map(|index| {
            let value = (index as u64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (value >> 32) as u8
        })
        .collect()
}

fn is_scheduling_operation(operation: &str) -> bool {
    matches!(
        operation,
        "command_queues" | "indirect_work_generation" | "resident_command_replay"
    )
}
