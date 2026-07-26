use super::sampling::collect_adaptive_samples;
use super::schema::{
    CalibrationRunStatus, CalibrationSamplePhase, CalibrationValidationResult,
    CalibrationValidationStatus, HardwareCalibrationPlan, HardwareCalibrationSample,
    HardwareCalibrationWorkload, HardwareCalibrationWorkloadResult,
};
use super::shader_compiler::persist_calibration_artifact;
use super::telemetry::elapsed_ns;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub(super) struct VulkanVideoCalibrationExecutor {
    physical_device_index: usize,
    artifact_directory: PathBuf,
    ffmpeg_path: PathBuf,
    ffmpeg_version: String,
    ffmpeg_digest: String,
}

impl VulkanVideoCalibrationExecutor {
    pub(super) fn new(
        physical_device_index: usize,
        artifact_directory: PathBuf,
    ) -> Result<Self, String> {
        let ffmpeg_path = find_executable("ffmpeg")
            .ok_or_else(|| "Vulkan video calibration requires ffmpeg in PATH".to_string())?;
        let ffmpeg_bytes = std::fs::read(&ffmpeg_path)
            .map_err(|error| format!("could not fingerprint ffmpeg {ffmpeg_path:?}: {error}"))?;
        let ffmpeg_digest = format!(
            "nerve.external_tool_sha256.v1:{:x}",
            Sha256::digest(ffmpeg_bytes)
        );
        let version = Command::new(&ffmpeg_path)
            .arg("-version")
            .output()
            .map_err(|error| format!("could not execute ffmpeg: {error}"))?;
        if !version.status.success() {
            return Err(format!(
                "ffmpeg -version failed: {}",
                String::from_utf8_lossy(&version.stderr)
            ));
        }
        let ffmpeg_version = String::from_utf8_lossy(&version.stdout)
            .lines()
            .next()
            .unwrap_or("ffmpeg version unavailable")
            .to_string();
        std::fs::create_dir_all(&artifact_directory).map_err(|error| {
            format!(
                "could not create Vulkan video artifact directory {artifact_directory:?}: {error}"
            )
        })?;
        Ok(Self {
            physical_device_index,
            artifact_directory,
            ffmpeg_path,
            ffmpeg_version,
            ffmpeg_digest,
        })
    }

    pub(super) fn run(
        &self,
        plan: &HardwareCalibrationPlan,
        workload: &HardwareCalibrationWorkload,
        cancelled: &Arc<AtomicBool>,
    ) -> Result<HardwareCalibrationWorkloadResult, String> {
        if !matches!(workload.operation.as_str(), "video_decode" | "video_encode") {
            return Err(format!(
                "Vulkan video calibrator does not implement {:?}",
                workload.operation
            ));
        }
        if workload.regime.get("codec").map(String::as_str) != Some("av1") {
            return Err("Vulkan video calibrator currently requires an AV1 regime".to_string());
        }
        let construction_started = Instant::now();
        let width_height = workload
            .regime
            .get("resolution")
            .map(String::as_str)
            .ok_or_else(|| "video workload has no resolution".to_string())?;
        parse_resolution(width_height)?;
        let frames = regime_u32(workload, "frames")?;
        let timeout = Duration::from_millis(u64::from(regime_u32(workload, "timeout_ms")?));
        let backend_index = artifact_index(workload, "external_backend_manifest")?;
        let fixture_index = artifact_index(workload, "video_fixture_av1")?;
        let backend_document = serde_json::to_vec(&serde_json::json!({
            "schema": "nerve.calibration.external_video_backend.v1",
            "executable": self.ffmpeg_path,
            "executable_digest": self.ffmpeg_digest,
            "version": self.ffmpeg_version,
            "codec": "av1",
            "implementation": "ffmpeg-vulkan-video",
            "physical_device_index": self.physical_device_index,
        }))
        .map_err(|error| format!("could not serialize video backend manifest: {error}"))?;
        let backend_artifact = persist_calibration_artifact(
            workload,
            backend_index,
            "json",
            &backend_document,
            &self.artifact_directory,
        )?;
        let fixture_temporary = self
            .artifact_directory
            .join(format!("{}.construction.ivf", workload.workload_id));
        let fixture_args = encode_arguments(
            self.physical_device_index,
            width_height,
            frames,
            Some(&fixture_temporary),
        );
        run_ffmpeg(&self.ffmpeg_path, &fixture_args, cancelled, timeout, false)?;
        let fixture_bytes = std::fs::read(&fixture_temporary).map_err(|error| {
            format!("could not read generated AV1 fixture {fixture_temporary:?}: {error}")
        })?;
        let _ = std::fs::remove_file(&fixture_temporary);
        if fixture_bytes.len() < 64 {
            return Err("Vulkan AV1 fixture is unexpectedly small".to_string());
        }
        let fixture_artifact = persist_calibration_artifact(
            workload,
            fixture_index,
            "ivf",
            &fixture_bytes,
            &self.artifact_directory,
        )?;
        let fixture_path = self
            .artifact_directory
            .join(&fixture_artifact.relative_path);
        let construction_duration_ns = elapsed_ns(construction_started);
        let prepared = PreparedVideo {
            operation: workload.operation.clone(),
            physical_device_index: self.physical_device_index,
            resolution: width_height.to_string(),
            frames,
            timeout,
            ffmpeg_path: self.ffmpeg_path.clone(),
            fixture_path,
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
        let observed_digest = prepared.validate(cancelled)?;
        let validation_passed = workload
            .validation
            .expected_digest
            .as_ref()
            .is_none_or(|expected| expected == &observed_digest);
        let launches = samples.iter().map(|sample| sample.iterations).sum();
        let mut artifacts = vec![backend_artifact, fixture_artifact];
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
            counters: [("external_process_launches".to_string(), launches)]
                .into_iter()
                .collect(),
            diagnostics: Vec::new(),
        })
    }
}

struct PreparedVideo {
    operation: String,
    physical_device_index: usize,
    resolution: String,
    frames: u32,
    timeout: Duration,
    ffmpeg_path: PathBuf,
    fixture_path: PathBuf,
}

impl PreparedVideo {
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
        while started.elapsed() < target || iterations == 0 {
            let arguments = if self.operation == "video_encode" {
                encode_arguments(
                    self.physical_device_index,
                    &self.resolution,
                    self.frames,
                    None,
                )
            } else {
                decode_arguments(
                    self.physical_device_index,
                    &self.fixture_path,
                    self.frames,
                    false,
                )
            };
            run_ffmpeg(
                &self.ffmpeg_path,
                &arguments,
                cancelled,
                self.timeout,
                false,
            )?;
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

    fn validate(&self, cancelled: &Arc<AtomicBool>) -> Result<String, String> {
        if self.operation == "video_encode" {
            let bytes = std::fs::read(&self.fixture_path)
                .map_err(|error| format!("could not validate AV1 fixture: {error}"))?;
            return Ok(format!(
                "nerve.calibration_output_sha256.v1:{:x}",
                Sha256::digest(bytes)
            ));
        }
        let arguments = decode_arguments(
            self.physical_device_index,
            &self.fixture_path,
            self.frames,
            true,
        );
        let output = run_ffmpeg(&self.ffmpeg_path, &arguments, cancelled, self.timeout, true)?;
        if output.stdout.is_empty() {
            return Err("Vulkan AV1 validation produced no frame checksums".to_string());
        }
        Ok(format!(
            "nerve.calibration_output_sha256.v1:{:x}",
            Sha256::digest(output.stdout)
        ))
    }
}

fn encode_arguments(
    device_index: usize,
    resolution: &str,
    frames: u32,
    output: Option<&Path>,
) -> Vec<String> {
    let mut arguments = vec![
        "-hide_banner".to_string(),
        "-loglevel".to_string(),
        "error".to_string(),
        "-nostdin".to_string(),
        "-init_hw_device".to_string(),
        format!("vulkan=vk:{device_index}"),
        "-filter_hw_device".to_string(),
        "vk".to_string(),
        "-f".to_string(),
        "lavfi".to_string(),
        "-i".to_string(),
        format!("testsrc2=size={resolution}:rate=30"),
        "-frames:v".to_string(),
        frames.to_string(),
        "-vf".to_string(),
        "format=nv12,hwupload".to_string(),
        "-c:v".to_string(),
        "av1_vulkan".to_string(),
    ];
    if let Some(output) = output {
        arguments.extend([
            "-f".to_string(),
            "ivf".to_string(),
            "-y".to_string(),
            output.display().to_string(),
        ]);
    } else {
        arguments.extend(["-f".to_string(), "null".to_string(), "-".to_string()]);
    }
    arguments
}

fn decode_arguments(
    device_index: usize,
    fixture: &Path,
    frames: u32,
    checksums: bool,
) -> Vec<String> {
    let mut arguments = vec![
        "-hide_banner".to_string(),
        "-loglevel".to_string(),
        "error".to_string(),
        "-nostdin".to_string(),
        "-init_hw_device".to_string(),
        format!("vulkan=vk:{device_index}"),
        "-hwaccel".to_string(),
        "vulkan".to_string(),
        "-hwaccel_device".to_string(),
        "vk".to_string(),
        "-hwaccel_output_format".to_string(),
        "vulkan".to_string(),
        "-i".to_string(),
        fixture.display().to_string(),
        "-frames:v".to_string(),
        frames.to_string(),
    ];
    if checksums {
        arguments.extend([
            "-vf".to_string(),
            "hwdownload,format=nv12".to_string(),
            "-f".to_string(),
            "framemd5".to_string(),
            "-".to_string(),
        ]);
    } else {
        arguments.extend(["-f".to_string(), "null".to_string(), "-".to_string()]);
    }
    arguments
}

fn run_ffmpeg(
    executable: &Path,
    arguments: &[String],
    cancelled: &Arc<AtomicBool>,
    timeout: Duration,
    capture_stdout: bool,
) -> Result<Output, String> {
    let mut command = Command::new(executable);
    command
        .args(arguments)
        .stdin(Stdio::null())
        .stdout(if capture_stdout {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .map_err(|error| format!("could not start Vulkan ffmpeg workload: {error}"))?;
    let started = Instant::now();
    loop {
        if cancelled.load(Ordering::Relaxed) {
            let _ = child.kill();
            let _ = child.wait();
            return Err("calibration was cancelled during Vulkan video execution".to_string());
        }
        if started.elapsed() > timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!(
                "Vulkan video workload exceeded bounded wait of {} ms",
                timeout.as_millis()
            ));
        }
        if child
            .try_wait()
            .map_err(|error| format!("could not poll Vulkan ffmpeg workload: {error}"))?
            .is_some()
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    let output = child
        .wait_with_output()
        .map_err(|error| format!("could not collect Vulkan ffmpeg output: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "Vulkan ffmpeg workload failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(output)
}

fn artifact_index(workload: &HardwareCalibrationWorkload, kind: &str) -> Result<usize, String> {
    workload
        .artifacts
        .iter()
        .position(|artifact| artifact.kind == kind)
        .ok_or_else(|| format!("video workload has no {kind} artifact"))
}

fn regime_u32(workload: &HardwareCalibrationWorkload, name: &str) -> Result<u32, String> {
    workload
        .regime
        .get(name)
        .ok_or_else(|| format!("video workload has no {name} regime"))?
        .parse::<u32>()
        .map_err(|error| format!("invalid video {name}: {error}"))
}

fn parse_resolution(value: &str) -> Result<(u32, u32), String> {
    let (width, height) = value
        .split_once('x')
        .ok_or_else(|| format!("invalid video resolution {value:?}"))?;
    let width = width
        .parse::<u32>()
        .map_err(|error| format!("invalid video width: {error}"))?;
    let height = height
        .parse::<u32>()
        .map_err(|error| format!("invalid video height: {error}"))?;
    if width == 0 || height == 0 {
        return Err("video resolution must be nonzero".to_string());
    }
    Ok((width, height))
}

fn find_executable(name: &str) -> Option<PathBuf> {
    std::env::var_os("PATH")
        .into_iter()
        .flat_map(|path| std::env::split_paths(&path).collect::<Vec<_>>())
        .map(|directory| directory.join(name))
        .find(|candidate| candidate.is_file())
}
