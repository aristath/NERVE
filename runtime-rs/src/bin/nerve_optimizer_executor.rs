use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use nerve_runtime::{
    RuntimeStagedCandidate, VulkanComputeDeviceCatalog, VulkanResidentModelPackageDeviceSlice,
    VulkanResidentModelPackageManifest, VulkanResidentTargetedComponentSession,
    VulkanTargetedComponentExecutionPhase,
};
use serde::Deserialize;
use serde_json::{Value, json};

const COMMAND_SCHEMA: &str = "nerve.optimizer.executor_command.v1";
const RESPONSE_SCHEMA: &str = "nerve.optimizer.executor_response.v2";
const AMD_VENDOR_ID: u32 = 0x1002;
const UNMOUNTED_LOGICAL_DEVICE_ID: &str = "optimizer:unmounted";

#[derive(Debug, Deserialize)]
#[serde(tag = "command", rename_all = "snake_case", deny_unknown_fields)]
enum ExecutorCommand {
    Mount {
        schema: String,
        request_id: String,
        package_manifest: PathBuf,
        candidate_root: Option<PathBuf>,
        candidate_id: Option<String>,
        component_id: String,
        physical_node_id: String,
        phase: String,
        activation_batch_width: usize,
        logical_device_id: String,
        physical_device_id: String,
        dynamic_state_capacity_activations: usize,
        maximum_quantum_wait_ns: u64,
    },
    Execute {
        schema: String,
        request_id: String,
        useful_units: usize,
        seed: u32,
    },
    Close {
        schema: String,
        request_id: String,
    },
}

struct MountCommand {
    request_id: String,
    package_manifest: PathBuf,
    candidate_root: Option<PathBuf>,
    candidate_id: Option<String>,
    component_id: String,
    physical_node_id: String,
    phase: VulkanTargetedComponentExecutionPhase,
    logical_device_id: String,
    physical_device_id: String,
    dynamic_state_capacity_activations: usize,
    maximum_quantum_wait: Duration,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("nerve-optimizer-executor error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let stdin = io::stdin();
    let mut input = BufReader::new(stdin.lock());
    let stdout = io::stdout();
    let mut output = BufWriter::new(stdout.lock());
    let command = read_command(&mut input)?
        .ok_or_else(|| invalid_input("executor input ended before mount"))?;
    let mount = MountCommand::from_command(command)?;
    let mount_started = Instant::now();
    let package_manifest = mount.package_manifest.canonicalize()?;
    if !package_manifest.is_file() {
        return Err(invalid_input("package manifest is not a regular file").into());
    }
    let package_root = package_manifest
        .parent()
        .ok_or_else(|| invalid_input("package manifest has no package root"))?
        .to_path_buf();

    let allowlist = BTreeSet::from([mount.physical_device_id.clone()]);
    let catalog = VulkanComputeDeviceCatalog::discover_allowed_physical_device_ids(&allowlist)?;
    let device_info = catalog
        .available_compute_devices()
        .iter()
        .find(|device| device.physical_device_id == mount.physical_device_id)
        .cloned()
        .ok_or_else(|| invalid_input("allowed physical device is unavailable"))?;
    if device_info.vendor_id != AMD_VENDOR_ID {
        return Err(invalid_input(format!(
            "optimizer execution requires an AMD GPU, but {:?} reports vendor 0x{:04x}",
            device_info.device_name, device_info.vendor_id
        ))
        .into());
    }
    let device = catalog.open_physical_device_index(device_info.physical_device_index)?;

    let manifest = VulkanResidentModelPackageManifest::from_json_file(&package_manifest)?;
    let node_devices =
        BTreeMap::from([(mount.component_id.clone(), mount.logical_device_id.clone())]);
    let mut runtime_model = manifest.mount_runtime_graph_controls(
        Some(UNMOUNTED_LOGICAL_DEVICE_ID),
        &node_devices,
        &[],
        None,
    )?;
    if let Some(candidate_root) = &mount.candidate_root {
        let candidate = RuntimeStagedCandidate::load(&package_root, candidate_root)?;
        if mount.candidate_id.as_deref() != Some(candidate.candidate_id.as_str()) {
            return Err(invalid_input(
                "mounted staged candidate does not match requested candidate_id",
            )
            .into());
        }
        runtime_model = runtime_model.apply_staged_runtime_candidate(&package_root, &candidate)?;
    } else if mount.candidate_id.is_some() {
        return Err(invalid_input("candidate_id requires a sealed candidate_root").into());
    }
    let placed_device = runtime_model
        .placement
        .device_for_component(&mount.component_id);
    if placed_device != mount.logical_device_id {
        return Err(invalid_input(format!(
            "targeted component {:?} resolved to logical device {placed_device:?}, expected {:?}",
            mount.component_id, mount.logical_device_id
        ))
        .into());
    }
    let package_id = runtime_model.package.package_id.clone();
    let slice = VulkanResidentModelPackageDeviceSlice::from_runtime_model_for_device(
        &device,
        &package_root,
        runtime_model,
        &mount.logical_device_id,
        Some(mount.dynamic_state_capacity_activations),
    )?;
    let session = VulkanResidentTargetedComponentSession::from_device_slice(
        &device,
        slice,
        &mount.component_id,
        &mount.physical_node_id,
        mount.phase,
    )?;
    let mounted_digest = mounted_state_digest(
        &package_id,
        mount.candidate_id.as_deref(),
        &mount.logical_device_id,
        &mount.physical_device_id,
        session.resident_parameter_bytes(),
        session.resident_transient_bytes(),
    );
    write_response(
        &mut output,
        &mount.request_id,
        "mounted",
        json!({
            "package_id": package_id,
            "candidate_id": mount.candidate_id,
            "component_id": mount.component_id,
            "physical_node_id": mount.physical_node_id,
            "logical_device_id": mount.logical_device_id,
            "physical_device_id": mount.physical_device_id,
            "device_name": device_info.device_name,
            "mount_duration_ns": nonzero_elapsed_ns(mount_started),
            "resident_parameter_bytes": session.resident_parameter_bytes(),
            "resident_transient_bytes": session.resident_transient_bytes(),
            "mounted_state_digest": mounted_digest,
        }),
    )?;

    let close_request_id = loop {
        let command = read_command(&mut input)?.ok_or_else(|| {
            invalid_input("executor input ended without an explicit close command")
        })?;
        match command {
            ExecutorCommand::Execute {
                schema,
                request_id,
                useful_units,
                seed,
            } => {
                require_schema(&schema)?;
                let report =
                    session.execute(&device, useful_units, seed, mount.maximum_quantum_wait)?;
                write_response(
                    &mut output,
                    &request_id,
                    "completed",
                    serde_json::to_value(report)?,
                )?;
            }
            ExecutorCommand::Close { schema, request_id } => {
                require_schema(&schema)?;
                break request_id;
            }
            ExecutorCommand::Mount { .. } => {
                return Err(invalid_input("executor cannot mount a second session").into());
            }
        }
    };
    let release_started = Instant::now();
    drop(session);
    drop(device);
    drop(catalog);
    write_response(
        &mut output,
        &close_request_id,
        "released",
        json!({
            "released": true,
            "release_duration_ns": nonzero_elapsed_ns(release_started),
            "mounted_state_digest": mounted_digest,
        }),
    )?;
    Ok(())
}

impl MountCommand {
    fn from_command(command: ExecutorCommand) -> Result<Self, Box<dyn Error>> {
        let ExecutorCommand::Mount {
            schema,
            request_id,
            package_manifest,
            candidate_root,
            candidate_id,
            component_id,
            physical_node_id,
            phase,
            activation_batch_width,
            logical_device_id,
            physical_device_id,
            dynamic_state_capacity_activations,
            maximum_quantum_wait_ns,
        } = command
        else {
            return Err(invalid_input("the first executor command must be mount").into());
        };
        require_schema(&schema)?;
        for (name, value) in [
            ("request_id", request_id.as_str()),
            ("component_id", component_id.as_str()),
            ("physical_node_id", physical_node_id.as_str()),
            ("logical_device_id", logical_device_id.as_str()),
            ("physical_device_id", physical_device_id.as_str()),
        ] {
            if value.is_empty() {
                return Err(
                    invalid_input(format!("executor mount {name} must not be empty")).into(),
                );
            }
        }
        if !physical_device_id.starts_with("vulkan-uuid:") {
            return Err(invalid_input(
                "executor mount requires a stable vulkan-uuid physical_device_id",
            )
            .into());
        }
        if dynamic_state_capacity_activations == 0 {
            return Err(invalid_input("executor dynamic-state capacity must be positive").into());
        }
        if maximum_quantum_wait_ns == 0 {
            return Err(invalid_input("executor maximum quantum wait must be positive").into());
        }
        let phase = match phase.as_str() {
            "decode" if activation_batch_width == 1 => {
                VulkanTargetedComponentExecutionPhase::Decode
            }
            "prefill" if activation_batch_width > 0 => {
                VulkanTargetedComponentExecutionPhase::Prefill {
                    activation_batch_width,
                }
            }
            "decode" => {
                return Err(
                    invalid_input("decode execution requires activation_batch_width=1").into(),
                );
            }
            _ => {
                return Err(invalid_input(
                    "executor phase must be decode or prefill with a positive width",
                )
                .into());
            }
        };
        Ok(Self {
            request_id,
            package_manifest,
            candidate_root,
            candidate_id,
            component_id,
            physical_node_id,
            phase,
            logical_device_id,
            physical_device_id,
            dynamic_state_capacity_activations,
            maximum_quantum_wait: Duration::from_nanos(maximum_quantum_wait_ns),
        })
    }
}

fn read_command(input: &mut impl BufRead) -> Result<Option<ExecutorCommand>, Box<dyn Error>> {
    let mut line = String::new();
    let byte_count = input.read_line(&mut line)?;
    if byte_count == 0 {
        return Ok(None);
    }
    if byte_count > 1024 * 1024 {
        return Err(invalid_input("executor command exceeds 1 MiB").into());
    }
    if line.trim().is_empty() {
        return Err(invalid_input("executor command must not be empty").into());
    }
    Ok(Some(serde_json::from_str(&line)?))
}

fn write_response(
    output: &mut impl Write,
    request_id: &str,
    status: &str,
    payload: Value,
) -> Result<(), Box<dyn Error>> {
    if request_id.is_empty() {
        return Err(invalid_input("executor response request_id is empty").into());
    }
    serde_json::to_writer(
        &mut *output,
        &json!({
            "schema": RESPONSE_SCHEMA,
            "request_id": request_id,
            "status": status,
            "payload": payload,
        }),
    )?;
    output.write_all(b"\n")?;
    output.flush()?;
    Ok(())
}

fn require_schema(schema: &str) -> Result<(), Box<dyn Error>> {
    if schema != COMMAND_SCHEMA {
        return Err(
            invalid_input(format!("unsupported executor command schema {schema:?}")).into(),
        );
    }
    Ok(())
}

fn mounted_state_digest(
    package_id: &str,
    candidate_id: Option<&str>,
    logical_device_id: &str,
    physical_device_id: &str,
    resident_parameter_bytes: usize,
    resident_transient_bytes: usize,
) -> String {
    use sha2::{Digest, Sha256};

    let identity = json!({
        "package_id": package_id,
        "candidate_id": candidate_id,
        "logical_device_id": logical_device_id,
        "physical_device_id": physical_device_id,
        "resident_parameter_bytes": resident_parameter_bytes,
        "resident_transient_bytes": resident_transient_bytes,
    });
    let digest = Sha256::digest(
        serde_json::to_vec(&identity).expect("mounted state identity is serializable"),
    );
    format!(
        "nerve.optimizer.device_state_sha256.v1:{}",
        digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    )
}

fn nonzero_elapsed_ns(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos())
        .unwrap_or(u64::MAX)
        .max(1)
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mount_command(phase: &str, width: usize) -> ExecutorCommand {
        ExecutorCommand::Mount {
            schema: COMMAND_SCHEMA.to_string(),
            request_id: "mount-1".to_string(),
            package_manifest: PathBuf::from("/package/manifest.json"),
            candidate_root: None,
            candidate_id: None,
            component_id: "block_1".to_string(),
            physical_node_id: "norm".to_string(),
            phase: phase.to_string(),
            activation_batch_width: width,
            logical_device_id: "optimizer:amd0".to_string(),
            physical_device_id: format!("vulkan-uuid:{}", "0".repeat(32)),
            dynamic_state_capacity_activations: 64,
            maximum_quantum_wait_ns: 1_000_000_000,
        }
    }

    #[test]
    fn optimizer_executor_mount_requires_phase_width_consistency() {
        let error = MountCommand::from_command(mount_command("decode", 64))
            .err()
            .expect("invalid decode width must fail");
        assert!(
            error.to_string().contains("activation_batch_width=1"),
            "{error}"
        );
        let prefill = MountCommand::from_command(mount_command("prefill", 64)).unwrap();
        assert_eq!(
            prefill.phase,
            VulkanTargetedComponentExecutionPhase::Prefill {
                activation_batch_width: 64,
            }
        );
    }

    #[test]
    fn optimizer_executor_protocol_rejects_unknown_fields() {
        let input = format!(
            "{{\"command\":\"close\",\"schema\":\"{COMMAND_SCHEMA}\",\"request_id\":\"close-1\",\"surprise\":true}}\n"
        );
        let error = read_command(&mut input.as_bytes()).unwrap_err();
        assert!(error.to_string().contains("unknown field"), "{error}");
    }

    #[test]
    fn optimizer_executor_mounted_identity_is_content_bound() {
        let exact = mounted_state_digest(
            "package",
            None,
            "optimizer:amd0",
            "vulkan-uuid:0000",
            10,
            20,
        );
        let repeated = mounted_state_digest(
            "package",
            None,
            "optimizer:amd0",
            "vulkan-uuid:0000",
            10,
            20,
        );
        let candidate = mounted_state_digest(
            "package",
            Some("candidate_123"),
            "optimizer:amd0",
            "vulkan-uuid:0000",
            10,
            20,
        );
        assert_eq!(exact, repeated);
        assert_ne!(exact, candidate);
        assert!(exact.starts_with("nerve.optimizer.device_state_sha256.v1:"));
    }
}
