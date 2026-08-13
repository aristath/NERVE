use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::{Duration, Instant};

use nerve_runtime::{
    RuntimeStagedCandidate, VulkanCompiledResourceLoadStatistics,
    VulkanCompiledResourceRepresentationReport, VulkanComputeDevice, VulkanComputeDeviceCatalog,
    VulkanDeviceLocalMemorySnapshot, VulkanResidentBufferPool, VulkanResidentModelPackageManifest,
    VulkanResidentRuntimeModel, VulkanResidentTargetedExecutionSession,
    VulkanResidentTargetedModelPackageDeviceSlice, VulkanTargetedComponentExecutionPhase,
    VulkanTargetedComponentExecutionScope,
};
use serde::Deserialize;
use serde_json::{Value, json};

const COMMAND_SCHEMA: &str = "nerve.optimizer.executor_command.v5";
const RESPONSE_SCHEMA: &str = "nerve.optimizer.executor_response.v5";
const MEMORY_SNAPSHOT_SCHEMA: &str = "nerve.runtime.vulkan_memory_snapshot.v1";
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
        execution_scope: String,
        activation_batch_width: usize,
        logical_device_id: String,
        physical_device_id: String,
        dynamic_state_capacity_activations: usize,
        maximum_quantum_wait_ns: u64,
        capture_output_values: bool,
    },
    Execute {
        schema: String,
        request_id: String,
        measurement_phase: ExecutorMeasurementPhase,
        useful_units: usize,
        sustained_window_count: usize,
        seed: u32,
    },
    Close {
        schema: String,
        request_id: String,
    },
    Shutdown {
        schema: String,
        request_id: String,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ExecutorMeasurementPhase {
    Warmup,
    Measured,
    Validation,
}

#[derive(Default)]
struct ExecutorMeasurementState {
    prepared_after_warmup: bool,
    terminal_execution_completed: bool,
    pending_preparation: Option<VulkanCompiledResourceRepresentationReport>,
    pending_resource_loading: VulkanCompiledResourceLoadStatistics,
}

struct MountCommand {
    request_id: String,
    package_manifest: PathBuf,
    candidate_root: Option<PathBuf>,
    candidate_id: Option<String>,
    component_id: String,
    physical_node_id: String,
    phase: VulkanTargetedComponentExecutionPhase,
    execution_scope: VulkanTargetedComponentExecutionScope,
    logical_device_id: String,
    physical_device_id: String,
    dynamic_state_capacity_activations: usize,
    maximum_quantum_wait: Duration,
    capture_output_values: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct PreparedRuntimeModelKey {
    package_manifest: PathBuf,
    candidate_root: Option<PathBuf>,
    candidate_id: Option<String>,
    component_id: String,
    logical_device_id: String,
}

#[derive(Default)]
struct ExecutorHost {
    manifests: BTreeMap<PathBuf, VulkanResidentModelPackageManifest>,
    prepared_runtime_models: BTreeMap<PreparedRuntimeModelKey, VulkanResidentRuntimeModel>,
    devices: BTreeMap<String, Rc<VulkanComputeDevice>>,
    logical_by_physical: BTreeMap<String, String>,
    parameter_pool: VulkanResidentBufferPool,
}

fn main() {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    if arguments
        .first()
        .is_some_and(|argument| argument == "--runtime-vulkan-memory-snapshot")
    {
        if let Err(error) = print_runtime_vulkan_memory_snapshot(&arguments[1..]) {
            eprintln!("nerve-optimizer-executor error: {error}");
            std::process::exit(1);
        }
        return;
    }
    if std::env::args()
        .skip(1)
        .eq(["--runtime-implementation-fingerprint"])
    {
        println!("{}", nerve_runtime::RUNTIME_IMPLEMENTATION_FINGERPRINT);
        return;
    }
    if std::env::args()
        .skip(1)
        .eq(["--runtime-device-local-memory-policy"])
    {
        println!(
            "{}",
            serde_json::to_string(&nerve_runtime::vulkan_device_local_memory_policy())
                .expect("runtime device-local memory policy must serialize")
        );
        return;
    }
    if let Err(error) = run() {
        eprintln!("nerve-optimizer-executor error: {error}");
        std::process::exit(1);
    }
}

fn print_runtime_vulkan_memory_snapshot(
    physical_device_ids: &[String],
) -> Result<(), Box<dyn Error>> {
    if physical_device_ids.is_empty()
        || physical_device_ids
            .iter()
            .any(|device_id| !device_id.starts_with("vulkan-uuid:"))
    {
        return Err(invalid_input(
            "--runtime-vulkan-memory-snapshot requires one or more stable Vulkan device IDs",
        )
        .into());
    }
    let allowed = physical_device_ids.iter().cloned().collect::<BTreeSet<_>>();
    if allowed.len() != physical_device_ids.len() {
        return Err(
            invalid_input("--runtime-vulkan-memory-snapshot device IDs must be unique").into(),
        );
    }
    let catalog = VulkanComputeDeviceCatalog::discover_allowed_physical_device_ids(&allowed)?;
    let mut snapshots = catalog.device_local_memory_snapshots()?;
    snapshots.sort_by(|left, right| left.physical_device_id.cmp(&right.physical_device_id));
    println!(
        "{}",
        serde_json::to_string(&runtime_vulkan_memory_snapshot_payload(snapshots))?
    );
    Ok(())
}

fn runtime_vulkan_memory_snapshot_payload(devices: Vec<VulkanDeviceLocalMemorySnapshot>) -> Value {
    json!({
        "schema": MEMORY_SNAPSHOT_SCHEMA,
        "devices": devices,
    })
}

fn run() -> Result<(), Box<dyn Error>> {
    let stdin = io::stdin();
    let mut input = BufReader::new(stdin.lock());
    let stdout = io::stdout();
    let mut output = BufWriter::new(stdout.lock());
    let mut host = ExecutorHost::default();
    let mut shutdown_completed = false;
    while let Some(command) = read_command(&mut input)? {
        match command {
            ExecutorCommand::Shutdown { schema, request_id } => {
                require_schema(&schema)?;
                let report = host.shutdown()?;
                write_response(&mut output, &request_id, "shutdown_complete", report)?;
                shutdown_completed = true;
                break;
            }
            command @ ExecutorCommand::Mount { .. } => {
                let mount = MountCommand::from_command(command)?;
                execute_session(&mut host, &mut input, &mut output, mount)?;
            }
            ExecutorCommand::Execute { .. } | ExecutorCommand::Close { .. } => {
                return Err(invalid_input(
                    "executor command outside a mounted session must be mount or shutdown",
                )
                .into());
            }
        }
    }
    if !shutdown_completed {
        return Err(invalid_input("executor input ended without an acknowledged shutdown").into());
    }
    Ok(())
}

fn execute_session(
    host: &mut ExecutorHost,
    input: &mut impl BufRead,
    output: &mut impl Write,
    mount: MountCommand,
) -> Result<(), Box<dyn Error>> {
    let mount_started = Instant::now();
    let package_manifest = mount.package_manifest.canonicalize()?;
    if !package_manifest.is_file() {
        return Err(invalid_input("package manifest is not a regular file").into());
    }
    let package_root = package_manifest
        .parent()
        .ok_or_else(|| invalid_input("package manifest has no package root"))?
        .to_path_buf();

    let manifest = host.manifest(&package_manifest)?;
    let (runtime_model, prepared_runtime_model_cache_hit) =
        host.prepared_runtime_model(&package_manifest, &package_root, manifest, &mount)?;
    let device = host.device(&mount.physical_device_id, &mount.logical_device_id)?;
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
    let slice = VulkanResidentTargetedModelPackageDeviceSlice::
        from_runtime_model_for_device_with_parameter_pool(
            &device,
            &package_root,
            runtime_model,
            &mount.component_id,
            &mount.logical_device_id,
            Some(mount.dynamic_state_capacity_activations),
            &host.parameter_pool,
        )?;
    let session = VulkanResidentTargetedExecutionSession::from_targeted_device_slice(
        &device,
        slice,
        &mount.component_id,
        &mount.physical_node_id,
        mount.phase,
        mount.execution_scope,
        mount.capture_output_values,
    )?;
    let mounted_digest = mounted_state_digest(
        &package_id,
        mount.candidate_id.as_deref(),
        &mount.logical_device_id,
        &mount.physical_device_id,
        session.resident_parameter_bytes(),
        session.resident_transient_bytes(),
    );
    let pool_stats = host.parameter_pool.stats();
    write_response(
        output,
        &mount.request_id,
        "mounted",
        json!({
            "package_id": package_id,
            "candidate_id": mount.candidate_id,
            "component_id": mount.component_id,
            "physical_node_id": mount.physical_node_id,
            "execution_scope": match mount.execution_scope {
                VulkanTargetedComponentExecutionScope::Node => "node",
                VulkanTargetedComponentExecutionScope::Component => "component",
                VulkanTargetedComponentExecutionScope::DecodeComponentPrefix => "decode_component_prefix",
            },
            "logical_device_id": mount.logical_device_id,
            "physical_device_id": mount.physical_device_id,
            "device_name": device.device_name(),
            "mount_duration_ns": nonzero_elapsed_ns(mount_started),
            "resident_parameter_bytes": session.resident_parameter_bytes(),
            "resident_transient_bytes": session.resident_transient_bytes(),
            "resident_asset_pool_bytes": pool_stats.resident_bytes,
            "resident_asset_pool_buffers": pool_stats.resident_buffer_count,
            "resident_asset_pool_hits": pool_stats.hit_count,
            "resident_asset_pool_misses": pool_stats.miss_count,
            "prepared_runtime_model_cache_hit": prepared_runtime_model_cache_hit,
            "prepared_runtime_model_cache_entries": host.prepared_runtime_models.len(),
            "mounted_state_digest": mounted_digest,
        }),
    )?;

    let mut measurement_state = ExecutorMeasurementState::default();
    let close_request_id = loop {
        let command = read_command(input)?.ok_or_else(|| {
            invalid_input("executor input ended without an explicit close command")
        })?;
        match command {
            ExecutorCommand::Execute {
                schema,
                request_id,
                measurement_phase,
                useful_units,
                sustained_window_count,
                seed,
            } => {
                require_schema(&schema)?;
                let report = execute_targeted_measurement(
                    &session,
                    &device,
                    measurement_phase,
                    useful_units,
                    sustained_window_count,
                    seed,
                    mount.maximum_quantum_wait,
                    &mut measurement_state,
                )?;
                write_response(output, &request_id, "completed", report)?;
            }
            ExecutorCommand::Close { schema, request_id } => {
                require_schema(&schema)?;
                break request_id;
            }
            ExecutorCommand::Mount { .. } => {
                return Err(
                    invalid_input("executor cannot mount another session before close").into(),
                );
            }
            ExecutorCommand::Shutdown { .. } => {
                return Err(
                    invalid_input("executor cannot shut down before closing its session").into(),
                );
            }
        }
    };
    let release_started = Instant::now();
    drop(session);
    let pool_stats = host.parameter_pool.stats();
    write_response(
        output,
        &close_request_id,
        "released",
        json!({
            "released": true,
            "release_duration_ns": nonzero_elapsed_ns(release_started),
            "mounted_state_digest": mounted_digest,
            "resident_asset_pool_bytes": pool_stats.resident_bytes,
            "resident_asset_pool_buffers": pool_stats.resident_buffer_count,
            "resident_asset_pool_hits": pool_stats.hit_count,
            "resident_asset_pool_misses": pool_stats.miss_count,
        }),
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute_targeted_measurement(
    session: &VulkanResidentTargetedExecutionSession,
    device: &VulkanComputeDevice,
    measurement_phase: ExecutorMeasurementPhase,
    useful_units: usize,
    sustained_window_count: usize,
    seed: u32,
    maximum_quantum_wait: Duration,
    state: &mut ExecutorMeasurementState,
) -> Result<Value, Box<dyn Error>> {
    if state.terminal_execution_completed {
        return Err(invalid_input(
            "executor session cannot execute after its measured or validation call",
        )
        .into());
    }
    match measurement_phase {
        ExecutorMeasurementPhase::Warmup => {
            let mut report = session.execute(
                device,
                useful_units,
                sustained_window_count,
                seed,
                maximum_quantum_wait,
            )?;
            if !state.prepared_after_warmup {
                state.pending_preparation =
                    Some(session.prepare_loaded_representations_for_measurement(device)?);
                state.prepared_after_warmup = true;
            }
            state
                .pending_resource_loading
                .checked_accumulate(report.resource_loading)?;
            report.resource_loading = VulkanCompiledResourceLoadStatistics::default();
            execution_report_payload(
                report,
                VulkanCompiledResourceRepresentationReport::default(),
            )
        }
        ExecutorMeasurementPhase::Measured => {
            let mut initial = session.execute(
                device,
                useful_units,
                sustained_window_count,
                seed,
                maximum_quantum_wait,
            )?;
            let (report, preparation) = if state.prepared_after_warmup {
                initial
                    .resource_loading
                    .checked_accumulate(state.pending_resource_loading)?;
                state.pending_resource_loading = VulkanCompiledResourceLoadStatistics::default();
                (
                    initial,
                    state.pending_preparation.take().unwrap_or_default(),
                )
            } else {
                let preparation = session.prepare_loaded_representations_for_measurement(device)?;
                let report = if preparation.promoted_group_count > 0 {
                    let mut replay = session.execute(
                        device,
                        useful_units,
                        sustained_window_count,
                        seed,
                        maximum_quantum_wait,
                    )?;
                    replay
                        .resource_loading
                        .checked_accumulate(initial.resource_loading)?;
                    replay
                } else {
                    initial
                };
                (report, preparation)
            };
            state.terminal_execution_completed = true;
            execution_report_payload(report, preparation)
        }
        ExecutorMeasurementPhase::Validation => {
            if state.prepared_after_warmup {
                return Err(invalid_input(
                    "validation execution cannot follow benchmark warmup in one session",
                )
                .into());
            }
            let initial = session.execute(
                device,
                useful_units,
                sustained_window_count,
                seed,
                maximum_quantum_wait,
            )?;
            let preparation = session.prepare_loaded_representations_for_measurement(device)?;
            let report = if preparation.promoted_group_count > 0 {
                let mut replay = session.execute(
                    device,
                    useful_units,
                    sustained_window_count,
                    seed,
                    maximum_quantum_wait,
                )?;
                replay
                    .resource_loading
                    .checked_accumulate(initial.resource_loading)?;
                replay
            } else {
                initial
            };
            state.terminal_execution_completed = true;
            execution_report_payload(report, preparation)
        }
    }
}

fn execution_report_payload(
    report: nerve_runtime::VulkanTargetedComponentExecutionReport,
    preparation: VulkanCompiledResourceRepresentationReport,
) -> Result<Value, Box<dyn Error>> {
    let conversion_bytes = preparation
        .promoted_source_bytes
        .checked_add(preparation.promoted_resident_bytes)
        .ok_or_else(|| invalid_input("representation conversion byte count overflowed"))?;
    let resident_parameter_bytes = report
        .resident_parameter_bytes
        .checked_add(preparation.promoted_resident_bytes)
        .ok_or_else(|| invalid_input("resident representation byte count overflowed"))?;
    let mut payload = serde_json::to_value(report)?;
    let object = payload
        .as_object_mut()
        .ok_or_else(|| invalid_input("targeted execution report did not serialize as an object"))?;
    object.insert(
        "resident_parameter_bytes".to_string(),
        json!(resident_parameter_bytes),
    );
    object.insert(
        "representation_conversion_bytes".to_string(),
        json!(conversion_bytes),
    );
    object.insert(
        "representation_conversion_ns".to_string(),
        json!(preparation.elapsed_ns),
    );
    object.insert(
        "representation_boundary_count".to_string(),
        json!(preparation.promoted_group_count),
    );
    Ok(payload)
}

impl ExecutorHost {
    fn shutdown(&mut self) -> Result<Value, Box<dyn Error>> {
        if self.devices.is_empty() {
            return Err(invalid_input("executor cannot shut down before mounting a device").into());
        }
        let shutdown_started = Instant::now();
        self.prepared_runtime_models.clear();
        let physical_device_ids = self.devices.keys().cloned().collect::<Vec<_>>();
        let pre_release_quiesce_started = Instant::now();
        for physical_device_id in &physical_device_ids {
            self.devices
                .get(physical_device_id)
                .expect("enumerated executor device remains present")
                .quiesce()?;
        }
        let pre_release_quiesce_duration_ns = nonzero_elapsed_ns(pre_release_quiesce_started);
        let mut device_releases = Vec::with_capacity(physical_device_ids.len());
        for physical_device_id in &physical_device_ids {
            let release_started = Instant::now();
            let logical_device_id = self
                .logical_by_physical
                .remove(physical_device_id)
                .ok_or_else(|| {
                    invalid_input(format!(
                        "executor lost the logical binding for \
                         {physical_device_id:?}"
                    ))
                })?;
            let device = self
                .devices
                .get(physical_device_id)
                .cloned()
                .expect("enumerated executor device remains present");
            device.quiesce()?;
            let released = self.parameter_pool.release_device(&logical_device_id)?;
            device.quiesce()?;
            let owned_device = self
                .devices
                .remove(physical_device_id)
                .expect("enumerated executor device remains present");
            drop(owned_device);
            drop(device);
            device_releases.push(json!({
                "physical_device_id": physical_device_id,
                "logical_device_id": logical_device_id,
                "released_buffer_count": released.resident_buffer_count,
                "released_buffer_bytes": released.resident_bytes,
                "quiesced": true,
                "device_context_destroyed": true,
                "release_duration_ns": nonzero_elapsed_ns(release_started),
            }));
        }
        let residual_pool = self.parameter_pool.stats();
        if !self.devices.is_empty()
            || !self.logical_by_physical.is_empty()
            || !self.prepared_runtime_models.is_empty()
            || self.parameter_pool.registered_device_count() != 0
            || residual_pool.resident_buffer_count != 0
            || residual_pool.resident_bytes != 0
        {
            return Err(invalid_input(
                "executor serialized shutdown left resident device resources",
            )
            .into());
        }
        Ok(executor_shutdown_payload(
            physical_device_ids,
            pre_release_quiesce_duration_ns,
            device_releases,
            nonzero_elapsed_ns(shutdown_started),
        ))
    }

    fn prepared_runtime_model(
        &mut self,
        package_manifest: &Path,
        package_root: &Path,
        manifest: VulkanResidentModelPackageManifest,
        mount: &MountCommand,
    ) -> Result<(VulkanResidentRuntimeModel, bool), Box<dyn Error>> {
        let key = PreparedRuntimeModelKey::new(
            package_manifest,
            mount.candidate_root.as_deref(),
            mount.candidate_id.as_deref(),
            &mount.component_id,
            &mount.logical_device_id,
        )?;
        if let Some(runtime_model) = self.prepared_runtime_models.get(&key) {
            return Ok((runtime_model.clone(), true));
        }

        let node_devices =
            BTreeMap::from([(mount.component_id.clone(), mount.logical_device_id.clone())]);
        let mut runtime_model = manifest.mount_runtime_graph_controls(
            Some(UNMOUNTED_LOGICAL_DEVICE_ID),
            &node_devices,
            &[],
            None,
        )?;
        if let Some(candidate_root) = &mount.candidate_root {
            let candidate = RuntimeStagedCandidate::load(package_root, candidate_root)?;
            if mount.candidate_id.as_deref() != Some(candidate.candidate_id.as_str()) {
                return Err(invalid_input(
                    "mounted staged candidate does not match requested candidate_id",
                )
                .into());
            }
            runtime_model = runtime_model.apply_staged_runtime_candidate_for_target(
                package_root,
                &candidate,
                &mount.component_id,
            )?;
        }
        let previous = self
            .prepared_runtime_models
            .insert(key, runtime_model.clone());
        debug_assert!(previous.is_none());
        Ok((runtime_model, false))
    }

    fn manifest(
        &mut self,
        path: &PathBuf,
    ) -> Result<VulkanResidentModelPackageManifest, Box<dyn Error>> {
        if !self.manifests.contains_key(path) {
            self.manifests.insert(
                path.clone(),
                VulkanResidentModelPackageManifest::from_json_file(path)?,
            );
        }
        Ok(self
            .manifests
            .get(path)
            .expect("executor manifest was inserted")
            .clone())
    }

    fn device(
        &mut self,
        physical_device_id: &str,
        logical_device_id: &str,
    ) -> Result<Rc<VulkanComputeDevice>, Box<dyn Error>> {
        if !self.devices.contains_key(physical_device_id) {
            let allowlist = BTreeSet::from([physical_device_id.to_string()]);
            let catalog =
                VulkanComputeDeviceCatalog::discover_allowed_physical_device_ids(&allowlist)?;
            let device_info = catalog
                .available_compute_devices()
                .iter()
                .find(|device| device.physical_device_id == physical_device_id)
                .cloned()
                .ok_or_else(|| invalid_input("allowed physical device is unavailable"))?;
            self.devices.insert(
                physical_device_id.to_string(),
                Rc::new(catalog.open_physical_device_index(device_info.physical_device_index)?),
            );
        }
        let device = self
            .devices
            .get(physical_device_id)
            .expect("executor device was inserted")
            .clone();
        if let Some(existing) = self.logical_by_physical.get(physical_device_id) {
            if existing != logical_device_id {
                return Err(invalid_input(format!(
                    "one executor cannot bind physical device \
                     {physical_device_id:?} to both {existing:?} and \
                     {logical_device_id:?}"
                ))
                .into());
            }
        } else {
            self.logical_by_physical.insert(
                physical_device_id.to_string(),
                logical_device_id.to_string(),
            );
        }
        self.parameter_pool
            .register_device(logical_device_id, device.clone())?;
        Ok(device)
    }
}

fn executor_shutdown_payload(
    physical_device_ids: Vec<String>,
    pre_release_quiesce_duration_ns: u64,
    device_releases: Vec<Value>,
    shutdown_duration_ns: u64,
) -> Value {
    json!({
        "released": true,
        "physical_device_ids": physical_device_ids,
        "pre_release_quiesce_duration_ns": pre_release_quiesce_duration_ns,
        "device_releases": device_releases,
        "shutdown_duration_ns": shutdown_duration_ns,
    })
}

impl PreparedRuntimeModelKey {
    fn new(
        package_manifest: &Path,
        candidate_root: Option<&Path>,
        candidate_id: Option<&str>,
        component_id: &str,
        logical_device_id: &str,
    ) -> io::Result<Self> {
        let (candidate_root, candidate_id) = match (candidate_root, candidate_id) {
            (Some(candidate_root), Some(candidate_id)) => {
                if fs::symlink_metadata(candidate_root)?
                    .file_type()
                    .is_symlink()
                {
                    return Err(invalid_input(
                        "staged candidate root must not be a symbolic link",
                    ));
                }
                (
                    Some(candidate_root.canonicalize()?),
                    Some(candidate_id.to_string()),
                )
            }
            (None, None) => (None, None),
            (Some(_), None) => {
                return Err(invalid_input(
                    "sealed candidate_root requires a candidate_id",
                ));
            }
            (None, Some(_)) => {
                return Err(invalid_input(
                    "candidate_id requires a sealed candidate_root",
                ));
            }
        };
        Ok(Self {
            package_manifest: package_manifest.to_path_buf(),
            candidate_root,
            candidate_id,
            component_id: component_id.to_string(),
            logical_device_id: logical_device_id.to_string(),
        })
    }
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
            execution_scope,
            activation_batch_width,
            logical_device_id,
            physical_device_id,
            dynamic_state_capacity_activations,
            maximum_quantum_wait_ns,
            capture_output_values,
        } = command
        else {
            return Err(invalid_input("an executor session must begin with mount").into());
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
        let execution_scope = match execution_scope.as_str() {
            "node" => VulkanTargetedComponentExecutionScope::Node,
            "component" => VulkanTargetedComponentExecutionScope::Component,
            "decode_component_prefix" if phase == VulkanTargetedComponentExecutionPhase::Decode => {
                VulkanTargetedComponentExecutionScope::DecodeComponentPrefix
            }
            "decode_component_prefix" => {
                return Err(invalid_input(
                    "decode_component_prefix execution scope requires decode phase",
                )
                .into());
            }
            _ => {
                return Err(invalid_input(
                    "executor execution_scope must be node, component, or decode_component_prefix",
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
            execution_scope,
            logical_device_id,
            physical_device_id,
            dynamic_state_capacity_activations,
            maximum_quantum_wait: Duration::from_nanos(maximum_quantum_wait_ns),
            capture_output_values,
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
            execution_scope: "node".to_string(),
            activation_batch_width: width,
            logical_device_id: "optimizer:gpu0".to_string(),
            physical_device_id: format!("vulkan-uuid:{}", "0".repeat(32)),
            dynamic_state_capacity_activations: 64,
            maximum_quantum_wait_ns: 1_000_000_000,
            capture_output_values: false,
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
    fn optimizer_executor_component_prefix_is_explicitly_decode_only() {
        let mut decode = mount_command("decode", 1);
        let ExecutorCommand::Mount {
            execution_scope, ..
        } = &mut decode
        else {
            unreachable!("fixture is a mount command");
        };
        *execution_scope = "decode_component_prefix".to_string();
        assert_eq!(
            MountCommand::from_command(decode).unwrap().execution_scope,
            VulkanTargetedComponentExecutionScope::DecodeComponentPrefix,
        );

        let mut prefill = mount_command("prefill", 64);
        let ExecutorCommand::Mount {
            execution_scope, ..
        } = &mut prefill
        else {
            unreachable!("fixture is a mount command");
        };
        *execution_scope = "decode_component_prefix".to_string();
        let error = MountCommand::from_command(prefill)
            .err()
            .expect("prefill prefix must fail");
        assert!(
            error.to_string().contains("requires decode phase"),
            "{error}",
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
    fn optimizer_executor_protocol_requires_a_typed_measurement_phase() {
        let valid = format!(
            "{{\"command\":\"execute\",\"schema\":\"{COMMAND_SCHEMA}\",\"request_id\":\"execute-1\",\"measurement_phase\":\"warmup\",\"useful_units\":2,\"sustained_window_count\":1,\"seed\":7}}\n"
        );
        assert!(matches!(
            read_command(&mut valid.as_bytes()).unwrap(),
            Some(ExecutorCommand::Execute {
                measurement_phase: ExecutorMeasurementPhase::Warmup,
                ..
            }),
        ));

        let missing = format!(
            "{{\"command\":\"execute\",\"schema\":\"{COMMAND_SCHEMA}\",\"request_id\":\"execute-1\",\"useful_units\":2,\"sustained_window_count\":1,\"seed\":7}}\n"
        );
        let error = read_command(&mut missing.as_bytes()).unwrap_err();
        assert!(error.to_string().contains("measurement_phase"), "{error}");

        let unknown = format!(
            "{{\"command\":\"execute\",\"schema\":\"{COMMAND_SCHEMA}\",\"request_id\":\"execute-1\",\"measurement_phase\":\"benchmark\",\"useful_units\":2,\"sustained_window_count\":1,\"seed\":7}}\n"
        );
        let error = read_command(&mut unknown.as_bytes()).unwrap_err();
        assert!(error.to_string().contains("unknown variant"), "{error}");
    }

    #[test]
    fn optimizer_executor_memory_snapshot_is_typed_and_preserves_missing_budget() {
        let payload = runtime_vulkan_memory_snapshot_payload(vec![
            VulkanDeviceLocalMemorySnapshot {
                physical_device_id: "vulkan-uuid:gpu0".to_string(),
                device_name: "fixture GPU".to_string(),
                pci_address: Some("0000:03:00.0".to_string()),
                heap_index: 1,
                physical_heap_bytes: 32 * 1024 * 1024 * 1024,
                memory_budget_supported: true,
                budget_bytes: Some(31 * 1024 * 1024 * 1024),
                usage_bytes: Some(3 * 1024 * 1024 * 1024),
                available_bytes: Some(28 * 1024 * 1024 * 1024),
            },
            VulkanDeviceLocalMemorySnapshot {
                physical_device_id: "vulkan-uuid:gpu1".to_string(),
                device_name: "unbudgeted GPU".to_string(),
                pci_address: None,
                heap_index: 0,
                physical_heap_bytes: 8 * 1024 * 1024 * 1024,
                memory_budget_supported: false,
                budget_bytes: None,
                usage_bytes: None,
                available_bytes: None,
            },
        ]);

        assert_eq!(payload["schema"], MEMORY_SNAPSHOT_SCHEMA);
        assert_eq!(payload["devices"][0]["heap_index"], 1);
        assert_eq!(
            payload["devices"][0]["available_bytes"],
            28_u64 * 1024 * 1024 * 1024
        );
        assert_eq!(payload["devices"][1]["memory_budget_supported"], false);
        assert_eq!(payload["devices"][1]["budget_bytes"], Value::Null);
    }

    #[test]
    fn optimizer_executor_reports_real_representation_lifecycle_resources() {
        let report = nerve_runtime::VulkanTargetedComponentExecutionReport {
            component_id: "layer_1".to_string(),
            node_id: "expert".to_string(),
            op: "linear".to_string(),
            phase: "decode".to_string(),
            activation_batch_width: 1,
            useful_units: 2,
            execution_ns: 11,
            output_digest: "output".to_string(),
            output_values_f32_le_hex: None,
            captured_outputs: None,
            state_digest: "state".to_string(),
            throughput_windows: vec![nerve_runtime::VulkanTargetedComponentThroughputWindow {
                index: 0,
                start_unit: 0,
                end_unit: 2,
                duration_ns: 11,
            }],
            resident_parameter_bytes: 1_024,
            resident_transient_bytes: 256,
            resource_loading: VulkanCompiledResourceLoadStatistics {
                load_count: 2,
                reload_count: 1,
                physical_read_bytes: 128,
                resident_bytes_produced: 512,
                uploaded_bytes: 512,
                read_ns: 3,
                derivation_ns: 5,
                upload_ns: 7,
                blocking_ns: 17,
            },
            physical_dispatch_count: 1,
            queue_submission_count: 1,
            synchronization_wait_count: 1,
            synchronization_wait_ns: 3,
            queue_wait_ns: 2,
        };
        let payload = execution_report_payload(
            report,
            VulkanCompiledResourceRepresentationReport {
                considered_group_count: 2,
                promoted_group_count: 2,
                promoted_source_bytes: 128,
                promoted_resident_bytes: 512,
                skipped_unstable_load_interval: false,
                skipped_capacity_bytes: 0,
                elapsed_ns: 17,
            },
        )
        .unwrap();

        assert_eq!(payload["resident_parameter_bytes"], 1_536);
        assert_eq!(payload["representation_conversion_bytes"], 640);
        assert_eq!(payload["representation_conversion_ns"], 17);
        assert_eq!(payload["representation_boundary_count"], 2);
        assert_eq!(
            payload["resource_loading"],
            json!({
                "load_count": 2,
                "reload_count": 1,
                "physical_read_bytes": 128,
                "resident_bytes_produced": 512,
                "uploaded_bytes": 512,
                "read_ns": 3,
                "derivation_ns": 5,
                "upload_ns": 7,
                "blocking_ns": 17,
            })
        );
    }

    #[test]
    fn optimizer_executor_mounted_identity_is_content_bound() {
        let exact = mounted_state_digest(
            "package",
            None,
            "optimizer:gpu0",
            "vulkan-uuid:0000",
            10,
            20,
        );
        let repeated = mounted_state_digest(
            "package",
            None,
            "optimizer:gpu0",
            "vulkan-uuid:0000",
            10,
            20,
        );
        let candidate = mounted_state_digest(
            "package",
            Some("candidate_123"),
            "optimizer:gpu0",
            "vulkan-uuid:0000",
            10,
            20,
        );
        assert_eq!(exact, repeated);
        assert_ne!(exact, candidate);
        assert!(exact.starts_with("nerve.optimizer.device_state_sha256.v1:"));
    }

    #[test]
    fn optimizer_executor_reuses_only_semantically_identical_prepared_models() {
        let package_manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test-fixtures/tiny_model/vulkan_resident_package.json")
            .canonicalize()
            .unwrap();
        let package_root = package_manifest.parent().unwrap().to_path_buf();
        let mut mount = MountCommand::from_command(mount_command("decode", 1)).unwrap();
        mount.component_id = "layer_00".to_string();
        let mut host = ExecutorHost::default();

        let manifest = host.manifest(&package_manifest).unwrap();
        let (first, first_hit) = host
            .prepared_runtime_model(&package_manifest, &package_root, manifest, &mount)
            .unwrap();
        assert!(!first_hit);
        assert_eq!(host.prepared_runtime_models.len(), 1);

        let manifest = host.manifest(&package_manifest).unwrap();
        let (second, second_hit) = host
            .prepared_runtime_model(&package_manifest, &package_root, manifest, &mount)
            .unwrap();
        assert!(second_hit);
        assert_eq!(first, second);
        assert_eq!(host.prepared_runtime_models.len(), 1);

        let mut different_placement = mount;
        different_placement.logical_device_id = "optimizer:gpu1".to_string();
        let manifest = host.manifest(&package_manifest).unwrap();
        let (_, different_placement_hit) = host
            .prepared_runtime_model(
                &package_manifest,
                &package_root,
                manifest,
                &different_placement,
            )
            .unwrap();
        assert!(!different_placement_hit);
        assert_eq!(host.prepared_runtime_models.len(), 2);
    }

    #[test]
    fn optimizer_executor_prepared_model_key_is_candidate_custody_bound() {
        let package_manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test-fixtures/tiny_model/vulkan_resident_package.json")
            .canonicalize()
            .unwrap();
        let candidate_root = package_manifest.parent().unwrap();
        let exact = PreparedRuntimeModelKey::new(
            &package_manifest,
            None,
            None,
            "layer_00",
            "optimizer:gpu0",
        )
        .unwrap();
        let candidate_a = PreparedRuntimeModelKey::new(
            &package_manifest,
            Some(candidate_root),
            Some("candidate_a"),
            "layer_00",
            "optimizer:gpu0",
        )
        .unwrap();
        let candidate_b = PreparedRuntimeModelKey::new(
            &package_manifest,
            Some(candidate_root),
            Some("candidate_b"),
            "layer_00",
            "optimizer:gpu0",
        )
        .unwrap();
        assert_ne!(exact, candidate_a);
        assert_ne!(candidate_a, candidate_b);
        assert!(
            PreparedRuntimeModelKey::new(
                &package_manifest,
                Some(candidate_root),
                None,
                "layer_00",
                "optimizer:gpu0",
            )
            .unwrap_err()
            .to_string()
            .contains("requires a candidate_id")
        );
    }

    #[test]
    fn optimizer_executor_shutdown_proof_keeps_its_strict_protocol_shape() {
        let payload = executor_shutdown_payload(
            vec!["vulkan-uuid:gpu0".to_string()],
            1,
            vec![json!({"device": "proof"})],
            2,
        );
        assert_eq!(
            payload
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([
                "device_releases".to_string(),
                "physical_device_ids".to_string(),
                "pre_release_quiesce_duration_ns".to_string(),
                "released".to_string(),
                "shutdown_duration_ns".to_string(),
            ]),
        );
    }
}
