use super::schema::{
    CalibrationRunStatus, CalibrationSamplePhase, CalibrationValidationResult,
    CalibrationValidationStatus, HardwareCalibrationPlan, HardwareCalibrationSample,
    HardwareCalibrationWorkload, HardwareCalibrationWorkloadResult,
};
use super::telemetry::{elapsed_ns, maximum_pci_temperature_millidegrees};
use crate::vulkan_compute::{VulkanComputeDevice, VulkanResidentBuffer, VulkanResidentBufferCopy};
use sha2::{Digest, Sha256};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub(super) struct VulkanTransferCalibrationExecutor {
    device: Rc<VulkanComputeDevice>,
}

impl VulkanTransferCalibrationExecutor {
    pub(super) fn new(device: Rc<VulkanComputeDevice>) -> Self {
        Self { device }
    }

    pub(super) fn run(
        &self,
        plan: &HardwareCalibrationPlan,
        workload: &HardwareCalibrationWorkload,
        cancelled: &Arc<AtomicBool>,
    ) -> Result<HardwareCalibrationWorkloadResult, String> {
        let construction_started = Instant::now();
        let mut prepared = PreparedTransfer::new(&self.device, workload)?;
        let construction_duration_ns = elapsed_ns(construction_started);
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

enum PreparedTransfer {
    HostToDevice {
        destination: VulkanResidentBuffer,
        source: Vec<u8>,
        pci_address: Option<String>,
    },
    DeviceToHost {
        source: VulkanResidentBuffer,
        last_read: Vec<u8>,
        pci_address: Option<String>,
    },
    DeviceToDevice {
        device: Rc<VulkanComputeDevice>,
        _source: VulkanResidentBuffer,
        destination: VulkanResidentBuffer,
        binding: VulkanResidentBufferCopy,
        byte_count: usize,
        pci_address: Option<String>,
    },
}

impl PreparedTransfer {
    fn new(
        device: &Rc<VulkanComputeDevice>,
        workload: &HardwareCalibrationWorkload,
    ) -> Result<Self, String> {
        if workload.operation != "buffer_copy" {
            return Err(format!(
                "Vulkan transfer executor does not implement {:?}",
                workload.operation
            ));
        }
        let byte_count = workload
            .regime
            .get("bytes")
            .ok_or_else(|| "Vulkan transfer workload has no byte count".to_string())?
            .parse::<usize>()
            .map_err(|error| format!("invalid Vulkan transfer byte count: {error}"))?;
        let pattern = deterministic_bytes(byte_count);
        let pci_address = device.pci_address().map(str::to_string);
        match workload.regime.get("direction").map(String::as_str) {
            Some("host_to_device") => Ok(Self::HostToDevice {
                destination: device
                    .create_resident_buffer(byte_count)
                    .map_err(|error| format!("could not allocate transfer destination: {error}"))?,
                source: pattern,
                pci_address,
            }),
            Some("device_to_host") => {
                let source = device
                    .create_resident_buffer(byte_count)
                    .map_err(|error| format!("could not allocate transfer source: {error}"))?;
                source
                    .write_bytes(&pattern)
                    .map_err(|error| format!("could not initialize transfer source: {error}"))?;
                Ok(Self::DeviceToHost {
                    source,
                    last_read: Vec::new(),
                    pci_address,
                })
            }
            Some("device_to_device") => {
                let source = device
                    .create_resident_buffer(byte_count)
                    .map_err(|error| format!("could not allocate transfer source: {error}"))?;
                let destination = device
                    .create_resident_buffer(byte_count)
                    .map_err(|error| format!("could not allocate transfer destination: {error}"))?;
                source
                    .write_bytes(&pattern)
                    .map_err(|error| format!("could not initialize transfer source: {error}"))?;
                let binding = device
                    .create_resident_buffer_copy(&source, &destination, byte_count)
                    .map_err(|error| format!("could not record device buffer copy: {error}"))?;
                Ok(Self::DeviceToDevice {
                    device: Rc::clone(device),
                    _source: source,
                    destination,
                    binding,
                    byte_count,
                    pci_address,
                })
            }
            direction => Err(format!(
                "Vulkan transfer workload has unsupported direction {direction:?}"
            )),
        }
    }

    fn execute_once(&mut self) -> Result<(), String> {
        match self {
            Self::HostToDevice {
                destination,
                source,
                ..
            } => destination
                .write_bytes(source)
                .map_err(|error| format!("host-to-device transfer failed: {error}")),
            Self::DeviceToHost {
                source, last_read, ..
            } => {
                *last_read = source
                    .read_bytes(source.byte_capacity())
                    .map_err(|error| format!("device-to-host transfer failed: {error}"))?;
                Ok(())
            }
            Self::DeviceToDevice {
                device,
                binding,
                byte_count,
                ..
            } => device
                .run_resident_buffer_copy(binding, *byte_count)
                .map_err(|error| format!("device-to-device transfer failed: {error}")),
        }
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
        while started.elapsed() < target || iterations == 0 {
            if cancelled.load(Ordering::Relaxed) {
                return Err("calibration was cancelled during a transfer sample".to_string());
            }
            self.execute_once()?;
            iterations = iterations.saturating_add(1);
        }
        Ok(HardwareCalibrationSample {
            sample_index,
            phase,
            duration_ns: elapsed_ns(started),
            device_duration_ns: None,
            iterations,
            window_index,
            thermal_millidegrees_celsius: maximum_pci_temperature_millidegrees(self.pci_address()),
            valid: true,
        })
    }

    fn observed_digest(&mut self) -> Result<String, String> {
        let bytes = match self {
            Self::HostToDevice { destination, .. } => destination
                .read_bytes(destination.byte_capacity().min(4096))
                .map_err(|error| format!("could not validate host-to-device transfer: {error}"))?,
            Self::DeviceToHost {
                source, last_read, ..
            } => {
                if last_read.is_empty() {
                    *last_read = source.read_bytes(source.byte_capacity()).map_err(|error| {
                        format!("could not validate device-to-host transfer: {error}")
                    })?;
                }
                last_read[..last_read.len().min(4096)].to_vec()
            }
            Self::DeviceToDevice { destination, .. } => destination
                .read_bytes(destination.byte_capacity().min(4096))
                .map_err(|error| format!("could not validate device copy: {error}"))?,
        };
        Ok(format!(
            "nerve.calibration_output_sha256.v1:{:x}",
            Sha256::digest(bytes)
        ))
    }

    fn pci_address(&self) -> Option<&str> {
        match self {
            Self::HostToDevice { pci_address, .. }
            | Self::DeviceToHost { pci_address, .. }
            | Self::DeviceToDevice { pci_address, .. } => pci_address.as_deref(),
        }
    }
}

fn deterministic_bytes(byte_count: usize) -> Vec<u8> {
    (0..byte_count)
        .map(|index| (index.wrapping_mul(131).wrapping_add(17) & 0xff) as u8)
        .collect()
}
