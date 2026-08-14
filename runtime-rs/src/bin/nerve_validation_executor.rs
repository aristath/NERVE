use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;

use nerve_runtime::{
    ResourceResidencyPolicy, RuntimeAssistantStreamProtocolAction,
    RuntimeChatGeneratedOutputControl, RuntimeChatSession, RuntimeStagedCandidate,
    VulkanCompiledResourceResidencyTotalsReport, VulkanComputeDevice, VulkanComputeDeviceCatalog,
    VulkanPlacedPromptEngineShutdownReport, VulkanResidentBufferPool,
    VulkanResidentExecutionCounters, VulkanResidentHfTokenizerTextCodec,
    VulkanResidentInProcessPlacedModelPackage, VulkanResidentInProcessPlacedPromptEngine,
    VulkanResidentInProcessPlacedPromptStream, VulkanResidentModelPackageManifest,
    VulkanResidentOutputControl, VulkanResidentRuntimeModel, VulkanResidentSamplerRuntimeConfig,
    VulkanResidentTokenInputEvent, VulkanResidentTokenTextCodec, chat_stop_token_ids_from_manifest,
    chat_transcript_codec, execute_vulkan_resident_chat_transaction,
    reset_vulkan_resident_execution_counters, vulkan_resident_execution_counters,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const COMMAND_SCHEMA: &str = "nerve.optimizer.validation_executor_command.v8";
const RESPONSE_SCHEMA: &str = "nerve.optimizer.validation_executor_response.v8";
const PROGRESS_SCHEMA: &str = "nerve.optimizer.executor_progress.v1";
const STREAM_ID: &str = "validation";
const COMPONENT_ACTIVATIONS_STEP_UNIT: &str = "component_activations";

#[derive(Debug, Deserialize)]
#[serde(tag = "command", rename_all = "snake_case", deny_unknown_fields)]
enum ExecutorCommand {
    Mount {
        schema: String,
        request_id: String,
        package_manifest: PathBuf,
        candidate_root: Option<PathBuf>,
        candidate_id: Option<String>,
        physical_device_ids: Vec<String>,
        component_placement: BTreeMap<String, String>,
        context_capacity: ValidationContextCapacity,
        validation_turns: Vec<String>,
        teacher_forced_assistant_turns: Vec<String>,
        execution_mode: String,
        conversation_set_policy: ConversationSetPolicy,
        speculative_draft_tokens: usize,
        random_seed: u32,
        sampler_config: VulkanResidentSamplerRuntimeConfig,
        residency_policy: ResourceResidencyPolicy,
        enable_thinking: bool,
        graph_operation: String,
        graph_target_component_id: Option<String>,
    },
    Execute {
        schema: String,
        request_id: String,
        turns: Vec<String>,
        teacher_forced_assistant_turns: Vec<String>,
        execution_mode: String,
        conversation_set_policy: ConversationSetPolicy,
        step_unit: String,
        max_output_tokens: Option<usize>,
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

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum ValidationContextCapacity {
    Declared { activations: usize },
    FixtureExact,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ConversationSetPolicy {
    minimum_sets: usize,
    maximum_sets: usize,
    repeat_second_set_if_residency_loaded: bool,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
struct ConversationResidencyActivity {
    load_required_count: u64,
    physical_bytes_read: u64,
    reload_count: u64,
    uploaded_bytes: u64,
}

impl ConversationResidencyActivity {
    fn between(
        before: &VulkanCompiledResourceResidencyTotalsReport,
        after: &VulkanCompiledResourceResidencyTotalsReport,
    ) -> Result<Self, io::Error> {
        fn delta(after: u64, before: u64, label: &str) -> Result<u64, io::Error> {
            after.checked_sub(before).ok_or_else(|| {
                invalid_input(format!("conversation residency counter {label} decreased"))
            })
        }
        Ok(Self {
            load_required_count: delta(
                after.residency_load_required_count,
                before.residency_load_required_count,
                "load_required_count",
            )?,
            physical_bytes_read: delta(
                after.physical_bytes_read,
                before.physical_bytes_read,
                "physical_bytes_read",
            )?,
            reload_count: delta(after.reload_count, before.reload_count, "reload_count")?,
            uploaded_bytes: delta(
                after.uploaded_bytes,
                before.uploaded_bytes,
                "uploaded_bytes",
            )?,
        })
    }

    fn loaded_residency(self) -> bool {
        self.load_required_count > 0
            || self.physical_bytes_read > 0
            || self.reload_count > 0
            || self.uploaded_bytes > 0
    }
}

fn should_run_another_conversation_set(
    policy: &ConversationSetPolicy,
    completed_sets: usize,
    activity: ConversationResidencyActivity,
) -> bool {
    completed_sets < policy.maximum_sets
        && (completed_sets < policy.minimum_sets
            || (policy.repeat_second_set_if_residency_loaded
                && completed_sets == 2
                && activity.loaded_residency()))
}

fn validate_conversation_set_policy(
    policy: &ConversationSetPolicy,
    execution_mode: &str,
) -> Result<(), io::Error> {
    let valid = match execution_mode {
        "conversation" => {
            policy.minimum_sets > 0
                && policy.minimum_sets <= policy.maximum_sets
                && policy.maximum_sets <= 3
                && (!policy.repeat_second_set_if_residency_loaded
                    || (policy.minimum_sets == 2 && policy.maximum_sets == 3))
        }
        _ => {
            policy.minimum_sets == 0
                && policy.maximum_sets == 0
                && !policy.repeat_second_set_if_residency_loaded
        }
    };
    if !valid {
        return Err(invalid_input(format!(
            "invalid conversation set policy for execution mode {execution_mode:?}"
        )));
    }
    Ok(())
}

struct MountedValidation {
    package_key: ValidationPackageKey,
    package_id: String,
    candidate_id: Option<String>,
    physical_device_ids: Vec<String>,
    engine: VulkanResidentInProcessPlacedPromptEngine,
    chat: RuntimeChatSession,
    transcript_codec: VulkanResidentHfTokenizerTextCodec,
    stop_token_ids: Vec<u32>,
    signal_processor_component_count: usize,
    mounted_state_digest: String,
    random_seed: u32,
    validation_fixture_digest: String,
    execution_mode: String,
    conversation_set_policy: ConversationSetPolicy,
    executed: bool,
}

struct ValidationDevicePool {
    physical_device_ids: Vec<String>,
    parameter_pool: VulkanResidentBufferPool,
    devices: BTreeMap<String, Rc<VulkanComputeDevice>>,
    physical_to_logical: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq)]
struct ValidationPackageKey {
    package_manifest: PathBuf,
    candidate_id: Option<String>,
    physical_device_ids: Vec<String>,
    component_placement: BTreeMap<String, String>,
    context_capacity: usize,
    speculative_draft_tokens: usize,
    sampler_config: VulkanResidentSamplerRuntimeConfig,
    residency_policy: ResourceResidencyPolicy,
    graph_operation: String,
    graph_target_component_id: Option<String>,
}

fn main() {
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
        eprintln!("nerve-validation-executor error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let stdin = io::stdin();
    let mut input = BufReader::new(stdin.lock());
    let stdout = io::stdout();
    let mut output = BufWriter::new(stdout.lock());
    let mut devices = None;
    let mut reusable_role = None;
    let mut mounted: Option<MountedValidation> = None;
    let mut shutdown_completed = false;
    while let Some(command) = read_command(&mut input)? {
        match command {
            command @ ExecutorCommand::Mount { .. } => {
                if mounted.is_some() {
                    return Err(invalid_input(
                        "executor cannot mount overlapping validation roles",
                    )
                    .into());
                }
                let mount_started = Instant::now();
                let (mount_request_id, mut next, reused) =
                    mount(command, &mut devices, &mut reusable_role)?;
                let mounted_state_digest = if reused {
                    next.mounted_state_digest.clone()
                } else {
                    next.engine.stream_resident_state_digest(STREAM_ID)?
                };
                next.mounted_state_digest = mounted_state_digest.clone();
                write_response(
                    &mut output,
                    &mount_request_id,
                    "mounted",
                    json!({
                        "package_id": next.package_id,
                        "candidate_id": next.candidate_id,
                        "physical_device_ids": next.physical_device_ids,
                        "context_capacity": next
                            .engine
                            .stream(STREAM_ID)
                            .expect("mounted stream exists")
                            .package()
                            .dynamic_state_capacity_activations,
                        "mounted_state_digest": mounted_state_digest,
                        "mount_duration_ns": nonzero_elapsed_ns(mount_started),
                    }),
                )?;
                mounted = Some(next);
            }
            ExecutorCommand::Execute {
                schema,
                request_id,
                turns,
                teacher_forced_assistant_turns,
                execution_mode,
                conversation_set_policy,
                step_unit,
                max_output_tokens,
            } => {
                require_schema(&schema)?;
                let role = mounted.as_mut().ok_or_else(|| {
                    invalid_input("executor cannot execute without a mounted role")
                })?;
                let mut progress_sequence = 0usize;
                let report = role.execute(
                    turns,
                    teacher_forced_assistant_turns,
                    &execution_mode,
                    &conversation_set_policy,
                    &step_unit,
                    max_output_tokens,
                    &mut |payload| {
                        write_progress(&mut output, &request_id, progress_sequence, payload)?;
                        progress_sequence = progress_sequence.saturating_add(1);
                        Ok(())
                    },
                )?;
                write_response(&mut output, &request_id, "completed", report)?;
            }
            ExecutorCommand::Close { schema, request_id } => {
                require_schema(&schema)?;
                let mut role = mounted
                    .take()
                    .ok_or_else(|| invalid_input("executor cannot close without a mounted role"))?;
                let release_started = Instant::now();
                let mounted_state_digest = role.mounted_state_digest.clone();
                let reset_started = Instant::now();
                role.engine
                    .reset_stream_for_new_session(STREAM_ID, role.random_seed)?;
                let reset_duration_ns = nonzero_elapsed_ns(reset_started);
                let proof_started = Instant::now();
                let reset_state_digest = role.engine.stream_resident_state_digest(STREAM_ID)?;
                let state_proof_duration_ns = nonzero_elapsed_ns(proof_started);
                if reset_state_digest != mounted_state_digest {
                    return Err(invalid_input(
                        "released validation role did not return to its exact initial state",
                    )
                    .into());
                }
                let released_device_ids = role
                    .engine
                    .stream(STREAM_ID)
                    .expect("reset stream remains resident")
                    .package()
                    .device_ids
                    .clone();
                reusable_role = Some(role);
                write_response(
                    &mut output,
                    &request_id,
                    "released",
                    json!({
                        "released": true,
                        "mounted_state_digest": mounted_state_digest,
                        "released_device_ids": released_device_ids,
                        "reset_duration_ns": reset_duration_ns,
                        "state_proof_duration_ns": state_proof_duration_ns,
                        "release_duration_ns": nonzero_elapsed_ns(release_started),
                    }),
                )?;
            }
            ExecutorCommand::Shutdown { schema, request_id } => {
                require_schema(&schema)?;
                if mounted.is_some() {
                    return Err(invalid_input(
                        "executor cannot shut down with a mounted validation role",
                    )
                    .into());
                }
                let report = shutdown_validation_resources(&mut reusable_role, &mut devices)?;
                write_response(&mut output, &request_id, "shutdown_complete", report)?;
                shutdown_completed = true;
                break;
            }
        }
    }
    if mounted.is_some() {
        return Err(invalid_input("validation executor input ended with a mounted role").into());
    }
    if !shutdown_completed {
        return Err(invalid_input(
            "validation executor input ended without an acknowledged shutdown",
        )
        .into());
    }
    Ok(())
}

fn shutdown_validation_resources(
    reusable_role: &mut Option<MountedValidation>,
    devices: &mut Option<ValidationDevicePool>,
) -> Result<Value, Box<dyn Error>> {
    let shutdown_started = Instant::now();
    let pool = devices.as_ref().ok_or_else(|| {
        invalid_input("validation executor cannot shut down before mounting a device topology")
    })?;
    if reusable_role.is_none() {
        return Err(invalid_input(
            "validation executor shutdown requires a normally released reusable role",
        )
        .into());
    }
    let physical_device_ids = pool.physical_device_ids.clone();
    let pre_release_quiesce_started = Instant::now();
    for physical_device_id in &physical_device_ids {
        let logical_device_id = pool
            .physical_to_logical
            .get(physical_device_id)
            .ok_or_else(|| {
                invalid_input(format!(
                    "validation device pool lost physical device {physical_device_id:?}"
                ))
            })?;
        pool.devices
            .get(logical_device_id)
            .ok_or_else(|| {
                invalid_input(format!(
                    "validation device pool lost logical device {logical_device_id:?}"
                ))
            })?
            .quiesce()?;
    }
    let pre_release_quiesce_duration_ns = nonzero_elapsed_ns(pre_release_quiesce_started);

    // The mounted engine owns transient buffers and compiled-resource
    // residency. It must acknowledge serialized release before the pool can
    // prove exclusive ownership of every permanent allocation.
    let role_release_started = Instant::now();
    let engine_shutdown = shutdown_validation_role(
        reusable_role
            .take()
            .expect("validated reusable role remains present"),
    )?;
    let role_release_duration_ns = nonzero_elapsed_ns(role_release_started);

    let mut pool = devices
        .take()
        .expect("validated shutdown device pool exists");
    let mut device_releases = Vec::with_capacity(physical_device_ids.len());
    for physical_device_id in &physical_device_ids {
        let release_started = Instant::now();
        let logical_device_id = pool
            .physical_to_logical
            .remove(physical_device_id)
            .ok_or_else(|| {
                invalid_input(format!(
                    "validation device pool lost physical device {physical_device_id:?}"
                ))
            })?;
        let device = pool
            .devices
            .get(&logical_device_id)
            .cloned()
            .ok_or_else(|| {
                invalid_input(format!(
                    "validation device pool lost logical device {logical_device_id:?}"
                ))
            })?;
        device.quiesce()?;
        let released = pool.parameter_pool.release_device(&logical_device_id)?;
        device.quiesce()?;
        let owned_device = pool
            .devices
            .remove(&logical_device_id)
            .expect("validated logical validation device remains present");
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
    let residual_pool = pool.parameter_pool.stats();
    if !pool.devices.is_empty()
        || !pool.physical_to_logical.is_empty()
        || pool.parameter_pool.registered_device_count() != 0
        || residual_pool.resident_buffer_count != 0
        || residual_pool.resident_bytes != 0
    {
        return Err(invalid_input(
            "validation executor serialized shutdown left resident device resources",
        )
        .into());
    }
    drop(pool);
    Ok(json!({
        "released": true,
        "physical_device_ids": physical_device_ids,
        "pre_release_quiesce_duration_ns": pre_release_quiesce_duration_ns,
        "role_release_duration_ns": role_release_duration_ns,
        "engine_shutdown": engine_shutdown,
        "device_releases": device_releases,
        "shutdown_duration_ns": nonzero_elapsed_ns(shutdown_started),
    }))
}

fn mount(
    command: ExecutorCommand,
    devices: &mut Option<ValidationDevicePool>,
    reusable_role: &mut Option<MountedValidation>,
) -> Result<(String, MountedValidation, bool), Box<dyn Error>> {
    let ExecutorCommand::Mount {
        schema,
        request_id,
        package_manifest,
        candidate_root,
        candidate_id,
        physical_device_ids,
        component_placement,
        context_capacity,
        validation_turns,
        teacher_forced_assistant_turns,
        execution_mode,
        conversation_set_policy,
        speculative_draft_tokens,
        random_seed,
        sampler_config,
        residency_policy,
        enable_thinking,
        graph_operation,
        graph_target_component_id,
    } = command
    else {
        return Err(invalid_input("the first executor command must be mount").into());
    };
    require_schema(&schema)?;
    if request_id.is_empty() {
        return Err(invalid_input("mount request_id is empty").into());
    }
    if physical_device_ids.is_empty()
        || physical_device_ids.iter().any(String::is_empty)
        || physical_device_ids.iter().collect::<BTreeSet<_>>().len() != physical_device_ids.len()
    {
        return Err(invalid_input("mount requires unique non-empty physical_device_ids").into());
    }
    let package_manifest = package_manifest.canonicalize()?;
    if !package_manifest.is_file() {
        return Err(invalid_input("package manifest is not a regular file").into());
    }
    let package_root = package_manifest
        .parent()
        .ok_or_else(|| invalid_input("package manifest has no package root"))?
        .to_path_buf();
    let manifest = VulkanResidentModelPackageManifest::from_json_file(&package_manifest)?;
    let tokenizer_dir = resolve_package_path(&package_root, &manifest.tokenizer.path);
    let chat_variables =
        BTreeMap::from([("enable_thinking".to_string(), Value::Bool(enable_thinking))]);
    let chat = RuntimeChatSession::from_tokenizer_dir(&tokenizer_dir, &chat_variables)?;
    let transcript_codec = chat_transcript_codec(&tokenizer_dir)?;
    let context_capacity = resolve_context_capacity(
        context_capacity,
        manifest.max_context_activations,
        &chat,
        &transcript_codec,
        &validation_turns,
        &teacher_forced_assistant_turns,
        &execution_mode,
    )?;
    validate_conversation_set_policy(&conversation_set_policy, &execution_mode)?;
    let validation_fixture_digest = execution_fixture_digest(
        &validation_turns,
        &teacher_forced_assistant_turns,
        &execution_mode,
        &conversation_set_policy,
    );
    let (bound_devices, physical_to_logical) = bind_devices(devices, &physical_device_ids)?;
    let default_logical_device = physical_to_logical
        .get(&physical_device_ids[0])
        .expect("declared physical device has logical binding");
    let node_devices = component_placement
        .iter()
        .map(|(component_id, physical_device_id)| {
            let logical = physical_to_logical
                .get(physical_device_id)
                .ok_or_else(|| {
                    invalid_input(format!(
                        "component {component_id:?} references undeclared physical device {physical_device_id:?}"
                    ))
                })?
                .clone();
            Ok((component_id.clone(), logical))
        })
        .collect::<Result<BTreeMap<_, _>, io::Error>>()?;
    let mut runtime_model = runtime_model_for_graph_operation(
        &manifest,
        &package_root,
        default_logical_device,
        &node_devices,
        &graph_operation,
        graph_target_component_id.as_deref(),
    )?;
    if let Some(candidate_root) = &candidate_root {
        let candidate = RuntimeStagedCandidate::load(&package_root, candidate_root)?;
        if candidate_id.as_deref() != Some(candidate.candidate_id.as_str()) {
            return Err(
                invalid_input("staged candidate does not match requested candidate_id").into(),
            );
        }
        runtime_model = runtime_model.apply_staged_runtime_candidate(&package_root, &candidate)?;
    } else if candidate_id.is_some() {
        return Err(invalid_input("candidate_id requires a sealed candidate_root").into());
    }
    runtime_model.package.sampler.spec =
        sampler_config.apply_to(&runtime_model.package.sampler.spec)?;
    validate_runtime_placement(&runtime_model, &physical_to_logical)?;
    let signal_processor_component_count = signal_processor_component_count(&runtime_model)?;
    let stop_token_ids = chat_stop_token_ids_from_manifest(
        &package_root,
        &tokenizer_dir,
        &runtime_model.package,
        &chat.formatter,
    )?;
    let package_key = ValidationPackageKey {
        package_manifest,
        candidate_id: candidate_id.clone(),
        physical_device_ids: physical_device_ids.clone(),
        component_placement,
        context_capacity,
        speculative_draft_tokens,
        sampler_config,
        residency_policy,
        graph_operation,
        graph_target_component_id,
    };
    if reusable_role
        .as_ref()
        .is_some_and(|role| role.package_key == package_key)
    {
        let mut role = reusable_role.take().expect("matching reusable role exists");
        role.engine.set_stream_random_seed(STREAM_ID, random_seed)?;
        role.chat = chat;
        role.transcript_codec = transcript_codec;
        role.stop_token_ids = stop_token_ids;
        role.random_seed = random_seed;
        role.validation_fixture_digest = validation_fixture_digest;
        role.execution_mode = execution_mode;
        role.conversation_set_policy = conversation_set_policy;
        role.executed = false;
        return Ok((request_id, role, true));
    }
    let placement_moves_parameters = reusable_role
        .as_ref()
        .is_some_and(|role| replacement_moves_parameter_residency(&role.package_key, &package_key));
    retire_reusable_role(
        reusable_role,
        devices
            .as_ref()
            .expect("bound validation device pool exists"),
        placement_moves_parameters,
    )?;
    let parameter_pool = &devices
        .as_ref()
        .expect("bound validation device pool exists")
        .parameter_pool;
    let package = Arc::new(
        VulkanResidentInProcessPlacedModelPackage::
            from_runtime_model_for_bound_devices_with_parameter_pool(
                &bound_devices,
                &package_root,
                runtime_model,
                Some(context_capacity),
                speculative_draft_tokens,
                residency_policy,
                parameter_pool,
            )?,
    );
    // The newly mounted package now owns every immutable buffer it needs.
    // Remove pooled buffers which belong only to a previous graph or
    // placement variant so validation cannot accumulate one model copy per
    // variant. Buffers shared with the new package remain referenced and are
    // therefore retained.
    evict_idle_validation_parameters(
        devices
            .as_ref()
            .expect("bound validation device pool exists"),
    )?;
    let stream =
        VulkanResidentInProcessPlacedPromptStream::new(package, bound_devices, random_seed)?
            .with_speculative_draft_tokens(speculative_draft_tokens)?;
    let package_id = stream.package().package_id.clone();
    let mut engine = VulkanResidentInProcessPlacedPromptEngine::new();
    engine.add_stream(STREAM_ID, stream)?;
    Ok((
        request_id,
        MountedValidation {
            package_key,
            package_id,
            candidate_id,
            physical_device_ids,
            engine,
            chat,
            transcript_codec,
            stop_token_ids,
            signal_processor_component_count,
            mounted_state_digest: String::new(),
            random_seed,
            validation_fixture_digest,
            execution_mode,
            conversation_set_policy,
            executed: false,
        },
        false,
    ))
}

fn replacement_moves_parameter_residency(
    previous: &ValidationPackageKey,
    next: &ValidationPackageKey,
) -> bool {
    previous.physical_device_ids != next.physical_device_ids
        || previous.component_placement != next.component_placement
}

fn retire_reusable_role(
    reusable_role: &mut Option<MountedValidation>,
    devices: &ValidationDevicePool,
    evict_before_replacement: bool,
) -> Result<(), Box<dyn Error>> {
    let Some(role) = reusable_role.take() else {
        return Ok(());
    };
    quiesce_validation_devices(devices)?;
    shutdown_validation_role(role)?;
    if evict_before_replacement {
        evict_idle_validation_parameters(devices)?;
    }
    Ok(())
}

fn shutdown_validation_role(
    role: MountedValidation,
) -> Result<VulkanPlacedPromptEngineShutdownReport, Box<dyn Error>> {
    let report = role.engine.shutdown();
    if !report.complete {
        return Err(invalid_input(format!(
            "validation engine teardown failed: {:?}",
            report.errors,
        ))
        .into());
    }
    Ok(report)
}

fn quiesce_validation_devices(devices: &ValidationDevicePool) -> Result<(), Box<dyn Error>> {
    for physical_device_id in &devices.physical_device_ids {
        let logical_device_id = devices
            .physical_to_logical
            .get(physical_device_id)
            .ok_or_else(|| {
                invalid_input(format!(
                    "validation device pool lost physical device \
                     {physical_device_id:?}"
                ))
            })?;
        devices
            .devices
            .get(logical_device_id)
            .ok_or_else(|| {
                invalid_input(format!(
                    "validation device pool lost logical device \
                     {logical_device_id:?}"
                ))
            })?
            .quiesce()?;
    }
    Ok(())
}

fn evict_idle_validation_parameters(devices: &ValidationDevicePool) -> Result<(), Box<dyn Error>> {
    for physical_device_id in &devices.physical_device_ids {
        let logical_device_id = devices
            .physical_to_logical
            .get(physical_device_id)
            .ok_or_else(|| {
                invalid_input(format!(
                    "validation device pool lost physical device \
                     {physical_device_id:?}"
                ))
            })?;
        let device = devices.devices.get(logical_device_id).ok_or_else(|| {
            invalid_input(format!(
                "validation device pool lost logical device \
                     {logical_device_id:?}"
            ))
        })?;
        device.quiesce()?;
        devices
            .parameter_pool
            .evict_unreferenced_device(logical_device_id)?;
        device.quiesce()?;
    }
    Ok(())
}

fn resolve_context_capacity(
    request: ValidationContextCapacity,
    package_limit: usize,
    chat: &RuntimeChatSession,
    transcript_codec: &VulkanResidentHfTokenizerTextCodec,
    turns: &[String],
    assistant_turns: &[String],
    execution_mode: &str,
) -> Result<usize, Box<dyn Error>> {
    let context_capacity = match request {
        ValidationContextCapacity::Declared { activations } => activations,
        ValidationContextCapacity::FixtureExact => {
            if !matches!(
                execution_mode,
                "teacher_forced" | "lifecycle_teacher_forced"
            ) {
                return Err(invalid_input(
                    "fixture-exact context capacity requires a bounded teacher-forced execution",
                )
                .into());
            }
            teacher_forced_fixture_context_capacity(
                chat,
                transcript_codec,
                turns,
                assistant_turns,
                execution_mode == "lifecycle_teacher_forced",
            )?
        }
    };
    if context_capacity == 0 {
        return Err(invalid_input("resolved context_capacity must be positive").into());
    }
    if context_capacity > package_limit {
        return Err(invalid_input(format!(
            "requested context capacity {context_capacity} exceeds package limit {package_limit}"
        ))
        .into());
    }
    Ok(context_capacity)
}

fn teacher_forced_fixture_context_capacity<C>(
    chat: &RuntimeChatSession,
    transcript_codec: &C,
    turns: &[String],
    assistant_turns: &[String],
    reserve_lifecycle_activation: bool,
) -> Result<usize, Box<dyn Error>>
where
    C: VulkanResidentTokenTextCodec,
{
    if turns.is_empty()
        || turns.len() != assistant_turns.len()
        || turns.iter().any(|turn| turn.trim().is_empty())
        || assistant_turns.iter().any(|turn| turn.trim().is_empty())
    {
        return Err(invalid_input(
            "fixture-exact capacity requires matching non-empty teacher-forced turns",
        )
        .into());
    }
    let mut simulated_chat = chat.clone();
    let mut required = 0usize;
    for (user_content, assistant_content) in turns.iter().zip(assistant_turns) {
        let prepared = simulated_chat.prepare_user_turn(user_content, transcript_codec)?;
        required = required.max(prepared.canonical_user_token_ids.len());
        let assistant_message = json!({
            "role": "assistant",
            "content": assistant_content,
        });
        let (_, canonical_committed_token_ids) = simulated_chat
            .render_assistant_commit_token_delta(
                &prepared,
                user_content,
                &assistant_message,
                transcript_codec,
            )?;
        required = required.max(canonical_committed_token_ids.len());
        simulated_chat.commit_assistant_turn(
            user_content,
            &assistant_message,
            canonical_committed_token_ids,
        );
    }
    if reserve_lifecycle_activation {
        required = required
            .checked_add(1)
            .ok_or_else(|| invalid_input("fixture-exact lifecycle context capacity overflowed"))?;
    }
    Ok(required.max(1))
}

fn execution_fixture_digest(
    turns: &[String],
    assistant_turns: &[String],
    execution_mode: &str,
    conversation_set_policy: &ConversationSetPolicy,
) -> String {
    artifact_digest(
        &serde_json::to_vec(&json!({
            "turns": turns,
            "teacher_forced_assistant_turns": assistant_turns,
            "execution_mode": execution_mode,
            "conversation_set_policy": conversation_set_policy,
        }))
        .expect("validation fixture is serializable"),
    )
}

fn positive_output_allowance(value: Option<usize>) -> Result<usize, Box<dyn Error>> {
    value.filter(|value| *value > 0).ok_or_else(|| {
        invalid_input("free-running validation requires a positive output allowance").into()
    })
}

impl MountedValidation {
    fn execute<F>(
        &mut self,
        turns: Vec<String>,
        teacher_forced_assistant_turns: Vec<String>,
        execution_mode: &str,
        conversation_set_policy: &ConversationSetPolicy,
        step_unit: &str,
        max_output_tokens: Option<usize>,
        on_progress: &mut F,
    ) -> Result<Value, Box<dyn Error>>
    where
        F: FnMut(Value) -> Result<(), Box<dyn Error>>,
    {
        if self.executed {
            return Err(
                invalid_input("validation role can execute only once before release").into(),
            );
        }
        self.executed = true;
        if step_unit != COMPONENT_ACTIVATIONS_STEP_UNIT {
            return Err(
                invalid_input(format!("unsupported validation step unit {step_unit:?}")).into(),
            );
        }
        if execution_mode != self.execution_mode
            || conversation_set_policy != &self.conversation_set_policy
            || execution_fixture_digest(
                &turns,
                &teacher_forced_assistant_turns,
                execution_mode,
                conversation_set_policy,
            ) != self.validation_fixture_digest
        {
            return Err(invalid_input(
                "validation execute request differs from the fixture bound at mount",
            )
            .into());
        }
        let mut report = match execution_mode {
            "conversation" => self.execute_conversation_sets(
                turns,
                positive_output_allowance(max_output_tokens)?,
                conversation_set_policy,
                on_progress,
            ),
            "teacher_forced" => {
                self.execute_teacher_forced(turns, teacher_forced_assistant_turns, on_progress)
            }
            "lifecycle_teacher_forced" => self.execute_lifecycle_teacher_forced(
                turns,
                teacher_forced_assistant_turns,
                on_progress,
            ),
            _ => Err(invalid_input(format!(
                "unsupported validation execution_mode {execution_mode:?}"
            ))
            .into()),
        }?;
        if execution_mode != "conversation" {
            report["conversation_sets"] = json!([]);
        }
        Ok(report)
    }

    fn conversation_residency_totals(
        &self,
    ) -> Result<VulkanCompiledResourceResidencyTotalsReport, Box<dyn Error>> {
        let stream = self
            .engine
            .stream(STREAM_ID)
            .ok_or_else(|| invalid_input("validation engine lost its mounted stream"))?;
        let coverage = stream.selection_telemetry_snapshot()?.report();
        Ok(stream
            .package()
            .compiled_resource_residency_report(&coverage)?
            .totals)
    }

    fn execute_conversation_sets<F>(
        &mut self,
        turns: Vec<String>,
        max_output_tokens: usize,
        policy: &ConversationSetPolicy,
        on_progress: &mut F,
    ) -> Result<Value, Box<dyn Error>>
    where
        F: FnMut(Value) -> Result<(), Box<dyn Error>>,
    {
        validate_conversation_set_policy(policy, "conversation")?;
        let initial_chat = self.chat.clone();
        let mut conversation_sets = Vec::with_capacity(policy.maximum_sets);
        let mut measured_report = None;
        for set_index in 0..policy.maximum_sets {
            let before = self.conversation_residency_totals()?;
            let mut set_progress = |mut payload: Value| {
                let object = payload
                    .as_object_mut()
                    .ok_or_else(|| invalid_input("validation progress payload is not an object"))?;
                object.insert("conversation_set_index".to_string(), json!(set_index));
                on_progress(payload)
            };
            let report =
                self.execute_one_conversation(turns.clone(), max_output_tokens, &mut set_progress)?;
            let after = self.conversation_residency_totals()?;
            let activity = ConversationResidencyActivity::between(&before, &after)?;
            conversation_sets.push(json!({
                "set_index": set_index,
                "disposition": "pending",
                "residency_activity": activity,
                "turns": report["turns"].clone(),
            }));
            let completed_sets = set_index + 1;
            if !should_run_another_conversation_set(policy, completed_sets, activity) {
                measured_report = Some(report);
                break;
            }
            self.engine
                .reset_stream_for_new_session(STREAM_ID, self.random_seed)?;
            self.chat = initial_chat.clone();
            let reset_state_digest = self.engine.stream_resident_state_digest(STREAM_ID)?;
            if reset_state_digest != self.mounted_state_digest {
                return Err(invalid_input(
                    "warmup conversation reset did not restore the exact initial stream state",
                )
                .into());
            }
        }
        let mut measured_report = measured_report.ok_or_else(|| {
            invalid_input("conversation set policy produced no measured conversation")
        })?;
        let measured_index = conversation_sets.len().saturating_sub(1);
        for (set_index, conversation_set) in conversation_sets.iter_mut().enumerate() {
            conversation_set["disposition"] = json!(if set_index == measured_index {
                "measured"
            } else {
                "discarded_warmup"
            });
        }
        measured_report["conversation_sets"] = Value::Array(conversation_sets);
        Ok(measured_report)
    }

    fn execute_one_conversation<F>(
        &mut self,
        turns: Vec<String>,
        max_output_tokens: usize,
        on_progress: &mut F,
    ) -> Result<Value, Box<dyn Error>>
    where
        F: FnMut(Value) -> Result<(), Box<dyn Error>>,
    {
        if turns.is_empty() || turns.iter().any(|turn| turn.trim().is_empty()) {
            return Err(invalid_input("validation conversation requires non-empty turns").into());
        }
        if max_output_tokens == 0 {
            return Err(invalid_input("validation max_output_tokens must be positive").into());
        }
        let started = Instant::now();
        let mut traces = Vec::with_capacity(turns.len());
        let mut behavioral_outputs = Vec::with_capacity(turns.len());
        let mut total_component_activations = 0usize;
        let mut total_scheduler_steps = 0usize;
        let mut total_execution_counters = VulkanResidentExecutionCounters::default();
        for (turn_index, user_content) in turns.iter().enumerate() {
            let turn_started = Instant::now();
            let prepared = self
                .chat
                .prepare_user_turn(user_content, &self.transcript_codec)?;
            let mut protocol_validator = self
                .chat
                .formatter
                .assistant_stream_protocol_validator(&self.transcript_codec)?;
            let mut progress_error: Option<Box<dyn Error>> = None;
            let transaction = execute_vulkan_resident_chat_transaction(
                &mut self.engine,
                STREAM_ID,
                &self.chat,
                &self.transcript_codec,
                &self.stop_token_ids,
                turn_index,
                user_content,
                &prepared,
                max_output_tokens,
                |event| {
                    if let Some(validator) = protocol_validator.as_mut() {
                        match validator.observe(event.output_event.token_id)? {
                            RuntimeAssistantStreamProtocolAction::Continue => {}
                            RuntimeAssistantStreamProtocolAction::TerminateAndTrim {
                                token_count,
                            } => {
                                return Ok(RuntimeChatGeneratedOutputControl::TerminateAndTrim {
                                    token_count,
                                });
                            }
                        }
                    }
                    let generated_tokens = event.output_event.output_index.saturating_add(1);
                    if (generated_tokens == 1 || generated_tokens % 32 == 0)
                        && progress_error.is_none()
                    {
                        if let Err(error) = on_progress(json!({
                            "phase": "generation",
                            "turn_index": turn_index,
                            "generated_tokens": generated_tokens,
                            "token_id": event.output_event.token_id,
                            "selected_logit_bits": event.output_event.selected_logit_bits,
                            "elapsed_ns": nonzero_elapsed_ns(turn_started),
                        })) {
                            progress_error = Some(error);
                        }
                    }
                    Ok(RuntimeChatGeneratedOutputControl::Continue)
                },
                |_, _| Ok(()),
            )?;
            if let Some(error) = progress_error {
                return Err(error);
            }
            let engine_runs = [
                Some(&transaction.user_run.engine_run),
                Some(&transaction.generation_run.engine_run),
                transaction
                    .canonical_commit_run
                    .as_ref()
                    .map(|run| &run.engine_run),
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
            let model_activations = engine_runs
                .iter()
                .map(|run| {
                    run.prefill_activation_count
                        .saturating_add(run.decode_activation_count)
                })
                .sum::<usize>();
            let component_activations = component_activation_count(
                model_activations,
                self.signal_processor_component_count,
            )?;
            let scheduler_steps = engine_runs
                .iter()
                .map(|run| run.scheduler_step_count)
                .sum::<usize>();
            total_component_activations =
                total_component_activations.saturating_add(component_activations);
            total_scheduler_steps = total_scheduler_steps.saturating_add(scheduler_steps);
            add_counters(
                &mut total_execution_counters,
                transaction.execution_counters,
            );
            let generation = &transaction
                .generation_run
                .engine_run
                .input_runs
                .iter()
                .find(|input_run| {
                    input_run.stream_id == STREAM_ID
                        && input_run.submitted_run.input_event.id == transaction.generation_event_id
                })
                .ok_or_else(|| {
                    invalid_input("validation engine did not return the generation event run")
                })?
                .submitted_run
                .session_run
                .run;
            let speculative = &generation.speculative_decode;
            let feedback = generation.resident_feedback;
            let transport = &generation.transport_stats;
            let turn_state_digest =
                validation_state_digest(&self.engine.stream_resident_state_digest(STREAM_ID)?);
            let stop_reason = validation_conversation_stop_reason(
                &transaction.generated_token_ids,
                &self.stop_token_ids,
                max_output_tokens,
                transaction.generation_terminated_by_protocol,
            )?;
            traces.push(json!({
                "turn_index": turn_index,
                "user": user_content,
                "assistant": transaction.assistant_content,
                "generated_token_ids": transaction.generated_token_ids,
                "canonical_committed_token_ids": transaction.canonical_committed_token_ids,
                "canonical_commit_mode": transaction.canonical_commit_mode,
                "stop_reason": stop_reason,
                "state_digest": turn_state_digest,
                "model_activations": model_activations,
                "component_activations": component_activations,
                "scheduler_steps": scheduler_steps,
                "elapsed_ns": transaction.elapsed_ns,
                "execution_counters": transaction.execution_counters,
                "speculative": {
                    "cycle_count": speculative.cycle_count,
                    "rollback_cycle_count": speculative.rollback_cycle_count,
                    "proposed_draft_tokens": speculative.proposed_draft_token_count,
                    "accepted_draft_tokens": speculative.accepted_draft_token_count,
                    "emitted_tokens": speculative.emitted_token_count,
                    "draft_time_ns": speculative.draft_time_ns,
                    "target_verification_time_ns": speculative.target_verification_time_ns,
                    "draft_catch_up_time_ns": speculative.draft_catch_up_time_ns,
                    "total_time_ns": speculative.total_time_ns,
                },
                "resident_feedback": {
                    "window_count": feedback.window_count,
                    "planned_tick_count": feedback.planned_tick_count,
                    "submitted_tick_count": feedback.submitted_tick_count,
                    "executed_tick_count": feedback.executed_tick_count,
                    "retained_tick_count": feedback.retained_tick_count,
                    "sampled_tick_count": feedback.sampled_tick_count,
                    "discarded_tick_count": feedback.discarded_tick_count,
                    "template_record_count": feedback.template_record_count,
                    "template_replay_count": feedback.template_replay_count,
                    "queue_submission_count": feedback.queue_submission_count,
                    "host_queue_submit_count": feedback.host_queue_submit_count,
                    "maximum_host_queue_submit_count_per_window": feedback.maximum_host_queue_submit_count_per_window,
                    "asynchronous_submission_count": feedback.asynchronous_submission_count,
                    "completion_poll_count": feedback.completion_poll_count,
                    "bounded_wait_count": feedback.bounded_wait_count,
                    "bounded_wait_timeout_count": feedback.bounded_wait_timeout_count,
                },
                "transport": {
                    "published_packet_count": transport.published_packet_count,
                    "published_byte_count": transport.published_byte_count,
                    "received_packet_count": transport.received_packet_count,
                    "received_byte_count": transport.received_byte_count,
                    "direct_copy_count": transport.direct_copy_count,
                    "direct_copy_byte_count": transport.direct_copy_byte_count,
                    "direct_receive_count": transport.direct_receive_count,
                    "direct_receive_byte_count": transport.direct_receive_byte_count,
                },
            }));
            behavioral_outputs.push(json!({
                "generated_token_ids": transaction.generated_token_ids,
                "canonical_committed_token_ids": transaction.canonical_committed_token_ids,
            }));
            self.chat.commit_assistant_turn(
                user_content,
                &transaction.assistant_message,
                transaction.canonical_committed_token_ids,
            );
            on_progress(json!({
                "phase": "turn_completed",
                "turn_index": turn_index,
                "generated_tokens": transaction.generated_token_ids.len(),
                "elapsed_ns": transaction.elapsed_ns,
                "component_activations": component_activations,
                "scheduler_steps": scheduler_steps,
            }))?;
        }
        let resident_state_digest = self.engine.stream_resident_state_digest(STREAM_ID)?;
        let state_digest = validation_state_digest(&resident_state_digest);
        let output_digest = artifact_digest(
            &serde_json::to_vec(&behavioral_outputs).expect("conversation output is serializable"),
        );
        Ok(json!({
            "output_digest": output_digest,
            "state_digest": state_digest,
            "steps": total_component_activations,
            "step_unit": COMPONENT_ACTIVATIONS_STEP_UNIT,
            "scheduler_steps": total_scheduler_steps,
            "elapsed_ns": nonzero_elapsed_ns(started),
            "turns": traces,
            "execution_counters": total_execution_counters,
        }))
    }

    fn execute_teacher_forced<F>(
        &mut self,
        turns: Vec<String>,
        assistant_turns: Vec<String>,
        on_progress: &mut F,
    ) -> Result<Value, Box<dyn Error>>
    where
        F: FnMut(Value) -> Result<(), Box<dyn Error>>,
    {
        if turns.is_empty()
            || turns.len() != assistant_turns.len()
            || turns.iter().any(|turn| turn.trim().is_empty())
            || assistant_turns.iter().any(|turn| turn.trim().is_empty())
        {
            return Err(invalid_input(
                "teacher-forced validation requires matching non-empty user and assistant turns",
            )
            .into());
        }
        let started = Instant::now();
        let mut traces = Vec::with_capacity(turns.len());
        let mut behavioral_outputs = Vec::with_capacity(turns.len());
        let mut total_component_activations = 0usize;
        let mut total_scheduler_steps = 0usize;
        let mut total_execution_counters = VulkanResidentExecutionCounters::default();
        for (turn_index, (user_content, assistant_content)) in
            turns.iter().zip(&assistant_turns).enumerate()
        {
            reset_vulkan_resident_execution_counters();
            let turn_started = Instant::now();
            let prepared = self
                .chat
                .prepare_user_turn(user_content, &self.transcript_codec)?;
            let assistant_message = json!({
                "role": "assistant",
                "content": assistant_content,
            });
            let (assistant_token_delta, canonical_committed_token_ids) =
                self.chat.render_assistant_commit_token_delta(
                    &prepared,
                    user_content,
                    &assistant_message,
                    &self.transcript_codec,
                )?;
            let mut canonical_turn_token_delta = prepared.user_token_delta.clone();
            canonical_turn_token_delta.extend_from_slice(&assistant_token_delta);
            let canonical_commit_run = self.engine.submit_input_event_until_idle(
                STREAM_ID,
                VulkanResidentTokenInputEvent::new(
                    format!("teacher_{turn_index}_canonical_turn"),
                    canonical_turn_token_delta,
                    0,
                )
                .with_origin("validation_teacher_forced_canonical_turn"),
            )?;
            let engine_runs = [&canonical_commit_run.engine_run];
            let model_activations = engine_runs
                .iter()
                .map(|run| {
                    run.prefill_activation_count
                        .saturating_add(run.decode_activation_count)
                })
                .sum::<usize>();
            let component_activations = component_activation_count(
                model_activations,
                self.signal_processor_component_count,
            )?;
            let scheduler_steps = engine_runs
                .iter()
                .map(|run| run.scheduler_step_count)
                .sum::<usize>();
            let counters = vulkan_resident_execution_counters();
            add_counters(&mut total_execution_counters, counters);
            total_component_activations =
                total_component_activations.saturating_add(component_activations);
            total_scheduler_steps = total_scheduler_steps.saturating_add(scheduler_steps);
            let turn_state_digest =
                validation_state_digest(&self.engine.stream_resident_state_digest(STREAM_ID)?);
            traces.push(json!({
                "turn_index": turn_index,
                "user": user_content,
                "assistant": assistant_content,
                "generated_token_ids": [],
                "canonical_committed_token_ids": canonical_committed_token_ids,
                "stop_reason": "fixture_completed",
                "state_digest": turn_state_digest,
                "model_activations": model_activations,
                "component_activations": component_activations,
                "scheduler_steps": scheduler_steps,
                "elapsed_ns": nonzero_elapsed_ns(turn_started),
                "execution_counters": counters,
                "speculative": zero_speculative_statistics(),
                "resident_feedback": zero_resident_feedback_statistics(),
                "transport": zero_transport_statistics(),
                "teacher_forced": true,
            }));
            behavioral_outputs.push(json!({
                "canonical_committed_token_ids": canonical_committed_token_ids,
            }));
            self.chat.commit_assistant_turn(
                user_content,
                &assistant_message,
                canonical_committed_token_ids,
            );
            on_progress(json!({
                "phase": "teacher_forced_turn_completed",
                "turn_index": turn_index,
                "generated_tokens": 0,
                "elapsed_ns": nonzero_elapsed_ns(turn_started),
                "component_activations": component_activations,
                "scheduler_steps": scheduler_steps,
            }))?;
        }
        Ok(json!({
            "output_digest": artifact_digest(
                &serde_json::to_vec(&behavioral_outputs)
                    .expect("teacher-forced output is serializable"),
            ),
            "state_digest": validation_state_digest(
                &self.engine.stream_resident_state_digest(STREAM_ID)?,
            ),
            "steps": total_component_activations,
            "step_unit": COMPONENT_ACTIVATIONS_STEP_UNIT,
            "scheduler_steps": total_scheduler_steps,
            "elapsed_ns": nonzero_elapsed_ns(started),
            "turns": traces,
            "execution_counters": total_execution_counters,
        }))
    }

    fn execute_lifecycle_teacher_forced<F>(
        &mut self,
        turns: Vec<String>,
        assistant_turns: Vec<String>,
        on_progress: &mut F,
    ) -> Result<Value, Box<dyn Error>>
    where
        F: FnMut(Value) -> Result<(), Box<dyn Error>>,
    {
        let mut report = self.execute_teacher_forced(turns, assistant_turns, on_progress)?;
        let lifecycle_started = Instant::now();
        reset_vulkan_resident_execution_counters();
        let before_fork = self.engine.stream_resident_state_digest(STREAM_ID)?;
        self.engine
            .fork_stream(STREAM_ID, "validation_fork", 0x7f31_2a09)?;
        let fork_digest = self
            .engine
            .stream_resident_state_digest("validation_fork")?;
        if fork_digest != before_fork {
            return Err(invalid_input("forked stream did not preserve resident state").into());
        }
        self.engine.remove_stream("validation_fork")?;

        let replay_token = *self
            .chat
            .committed_token_ids
            .last()
            .ok_or_else(|| invalid_input("lifecycle validation has no committed token"))?;
        let before_rollback = self.engine.stream_resident_state_digest(STREAM_ID)?;
        let rollback = self
            .engine
            .submit_input_event_transactionally_until_idle_with_output(
                STREAM_ID,
                VulkanResidentTokenInputEvent::new("validation_rollback", vec![replay_token], 1)
                    .with_origin("validation_transaction_rollback"),
                |_| VulkanResidentOutputControl::Continue,
            )?;
        let after_rollback = self.engine.stream_resident_state_digest(STREAM_ID)?;
        if before_rollback != after_rollback {
            return Err(
                invalid_input("transactional validation did not restore resident state").into(),
            );
        }

        self.engine.enqueue_input_event(
            STREAM_ID,
            VulkanResidentTokenInputEvent::new(
                "validation_interrupt_resume",
                vec![replay_token],
                0,
            )
            .with_origin("validation_interrupt_resume"),
        )?;
        let interrupt = self
            .engine
            .interrupt_stream(STREAM_ID, "validation interruption")?;
        let resumed = self.engine.run_until_idle_bounded(1)?;
        if !self.engine.snapshot().idle {
            return Err(invalid_input("interrupted stream did not resume to idle").into());
        }
        let additional_model_activations = rollback
            .engine_run
            .prefill_activation_count
            .saturating_add(rollback.engine_run.decode_activation_count)
            .saturating_add(resumed.prefill_activation_count)
            .saturating_add(resumed.decode_activation_count);
        let additional_component_activations = component_activation_count(
            additional_model_activations,
            self.signal_processor_component_count,
        )?;
        let additional_steps = rollback
            .engine_run
            .scheduler_step_count
            .saturating_add(resumed.scheduler_step_count);
        let object = report
            .as_object_mut()
            .expect("conversation report is an object");
        let mut execution_counters: VulkanResidentExecutionCounters =
            serde_json::from_value(object["execution_counters"].clone())?;
        add_counters(
            &mut execution_counters,
            vulkan_resident_execution_counters(),
        );
        object.insert(
            "execution_counters".to_string(),
            serde_json::to_value(execution_counters)?,
        );
        object.insert(
            "steps".to_string(),
            json!(object["steps"].as_u64().unwrap_or_default().saturating_add(
                u64::try_from(additional_component_activations).unwrap_or(u64::MAX),
            )),
        );
        object.insert(
            "scheduler_steps".to_string(),
            json!(
                object["scheduler_steps"]
                    .as_u64()
                    .unwrap_or_default()
                    .saturating_add(u64::try_from(additional_steps).unwrap_or(u64::MAX),)
            ),
        );
        let lifecycle = json!({
            "fork_state_preserved": true,
            "transaction_state_restored": true,
            "interrupt_state_preserved": interrupt
                .stream_control_run
                .control_event
                .state_preserved,
            "resumed_to_idle": true,
            "rollback_generated_token_ids": rollback.generated_token_ids,
        });
        let previous_output_digest = object["output_digest"].as_str().unwrap_or_default();
        object.insert(
            "output_digest".to_string(),
            json!(artifact_digest(
                &serde_json::to_vec(&json!({
                    "conversation": previous_output_digest,
                    "lifecycle": lifecycle,
                }))
                .expect("lifecycle output is serializable"),
            )),
        );
        object.insert(
            "state_digest".to_string(),
            json!(validation_state_digest(
                &self.engine.stream_resident_state_digest(STREAM_ID)?,
            )),
        );
        object
            .get_mut("turns")
            .and_then(Value::as_array_mut)
            .expect("conversation report has turns")
            .last_mut()
            .and_then(Value::as_object_mut)
            .expect("conversation report has a final turn")
            .insert("lifecycle".to_string(), lifecycle);
        on_progress(json!({
            "phase": "lifecycle_completed",
            "elapsed_ns": nonzero_elapsed_ns(lifecycle_started),
            "component_activations": additional_component_activations,
            "scheduler_steps": additional_steps,
        }))?;
        Ok(report)
    }
}

fn validation_conversation_stop_reason(
    generated_token_ids: &[u32],
    stop_token_ids: &[u32],
    output_allowance: usize,
    generation_terminated_by_protocol: bool,
) -> Result<&'static str, io::Error> {
    if generation_terminated_by_protocol {
        Ok("protocol_boundary")
    } else if generated_token_ids
        .last()
        .is_some_and(|token_id| stop_token_ids.contains(token_id))
    {
        Ok("eos")
    } else if generated_token_ids.len() == output_allowance {
        Ok("output_allowance")
    } else {
        Err(invalid_input(
            "validation conversation ended without an ordinary stop boundary",
        ))
    }
}

fn runtime_model_for_graph_operation(
    manifest: &VulkanResidentModelPackageManifest,
    package_root: &Path,
    default_logical_device: &str,
    node_devices: &BTreeMap<String, String>,
    operation: &str,
    target_component_id: Option<&str>,
) -> Result<VulkanResidentRuntimeModel, Box<dyn Error>> {
    let source = manifest.resolved_source_graph(package_root)?;
    let mut runtime_graph = manifest.runtime_graph_from_controls(
        Some(default_logical_device),
        node_devices,
        &[],
        None,
    )?;
    runtime_graph = match operation {
        "none" | "restore" => runtime_graph,
        "duplicate" => {
            let target = required_graph_target(target_component_id)?;
            runtime_graph.duplicate_after_instance(
                &source,
                target,
                format!("{target}__validation_duplicate"),
            )?
        }
        "bypass" => {
            let anchor = required_graph_target(target_component_id)?;
            let source_by_id = source
                .circuits
                .iter()
                .map(|artifact| {
                    (
                        artifact.component.id.as_str(),
                        artifact.component.runtime_role,
                    )
                })
                .collect::<BTreeMap<_, _>>();
            let chain = runtime_graph
                .instances
                .iter()
                .filter(|instance| {
                    source_by_id
                        .get(instance.source_component_id.as_str())
                        .is_some_and(|role| role.is_signal_processor())
                })
                .map(|instance| instance.instance_id.clone())
                .collect::<Vec<_>>();
            let anchor_index = chain
                .iter()
                .position(|instance_id| instance_id == anchor)
                .ok_or_else(|| {
                    invalid_input(format!(
                        "bypass anchor {anchor:?} is not a signal processor"
                    ))
                })?;
            if chain.len() < 2 {
                return Err(invalid_input(
                    "bypass validation requires at least two signal processors",
                )
                .into());
            }
            let peer_index = if anchor_index + 1 < chain.len() {
                anchor_index + 1
            } else {
                anchor_index - 1
            };
            runtime_graph.with_instance_enabled(&chain[peer_index], false)?
        }
        "rewire" => {
            let target = required_graph_target(target_component_id)?;
            let source_by_id = source
                .circuits
                .iter()
                .map(|artifact| {
                    (
                        artifact.component.id.as_str(),
                        artifact.component.runtime_role,
                    )
                })
                .collect::<BTreeMap<_, _>>();
            let mut chain = runtime_graph
                .instances
                .iter()
                .filter(|instance| {
                    source_by_id
                        .get(instance.source_component_id.as_str())
                        .is_some_and(|role| role.is_signal_processor())
                })
                .map(|instance| {
                    (
                        instance.instance_id.clone(),
                        instance.source_component_id.clone(),
                    )
                })
                .collect::<Vec<_>>();
            let target_index = chain
                .iter()
                .position(|(instance_id, _)| instance_id == target)
                .ok_or_else(|| {
                    invalid_input(format!(
                        "rewire target {target:?} is not a signal processor"
                    ))
                })?;
            if chain.len() < 2 {
                return Err(invalid_input(
                    "rewire validation requires at least two signal processors",
                )
                .into());
            }
            let peer_index = if target_index + 1 < chain.len() {
                target_index + 1
            } else {
                target_index - 1
            };
            chain.swap(target_index, peer_index);
            runtime_graph.with_signal_processor_chain(&source, &chain)?
        }
        _ => {
            return Err(invalid_input(format!(
                "unsupported validation graph operation {operation:?}"
            ))
            .into());
        }
    };
    Ok(manifest.clone().mount_runtime_graph(&runtime_graph)?)
}

fn required_graph_target(target_component_id: Option<&str>) -> Result<&str, io::Error> {
    target_component_id
        .filter(|value| !value.is_empty())
        .ok_or_else(|| invalid_input("graph operation requires graph_target_component_id"))
}

fn open_devices(
    physical_device_ids: &[String],
) -> Result<
    (
        BTreeMap<String, Rc<VulkanComputeDevice>>,
        BTreeMap<String, String>,
    ),
    Box<dyn Error>,
> {
    let allowlist = physical_device_ids.iter().cloned().collect::<BTreeSet<_>>();
    let catalog = VulkanComputeDeviceCatalog::discover_allowed_physical_device_ids(&allowlist)?;
    let available = catalog.available_compute_devices();
    let mut physical_to_logical = BTreeMap::new();
    let mut devices = BTreeMap::new();
    for (ordinal, physical_device_id) in physical_device_ids.iter().enumerate() {
        let info = available
            .iter()
            .find(|device| device.physical_device_id == *physical_device_id)
            .ok_or_else(|| {
                invalid_input(format!(
                    "allowed physical device {physical_device_id:?} is unavailable"
                ))
            })?;
        let logical_device_id = format!("optimizer:device:{ordinal}");
        let device = Rc::new(catalog.open_physical_device_index(info.physical_device_index)?);
        physical_to_logical.insert(physical_device_id.clone(), logical_device_id.clone());
        devices.insert(logical_device_id, device);
    }
    Ok((devices, physical_to_logical))
}

fn bind_devices(
    pool: &mut Option<ValidationDevicePool>,
    physical_device_ids: &[String],
) -> Result<
    (
        BTreeMap<String, Rc<VulkanComputeDevice>>,
        BTreeMap<String, String>,
    ),
    Box<dyn Error>,
> {
    if let Some(existing) = pool {
        if existing.physical_device_ids != physical_device_ids {
            return Err(invalid_input(
                "one validation stage cannot change its physical device topology",
            )
            .into());
        }
        return Ok((
            existing.devices.clone(),
            existing.physical_to_logical.clone(),
        ));
    }
    let (devices, physical_to_logical) = open_devices(physical_device_ids)?;
    let parameter_pool = VulkanResidentBufferPool::default();
    for (physical_device_id, logical_device_id) in &physical_to_logical {
        let device = devices.get(logical_device_id).ok_or_else(|| {
            invalid_input(format!(
                "opened physical device {physical_device_id:?} has no \
                 logical validation binding"
            ))
        })?;
        parameter_pool.register_device(logical_device_id, device.clone())?;
    }
    *pool = Some(ValidationDevicePool {
        physical_device_ids: physical_device_ids.to_vec(),
        parameter_pool,
        devices: devices.clone(),
        physical_to_logical: physical_to_logical.clone(),
    });
    Ok((devices, physical_to_logical))
}

fn validate_runtime_placement(
    runtime_model: &VulkanResidentRuntimeModel,
    physical_to_logical: &BTreeMap<String, String>,
) -> Result<(), Box<dyn Error>> {
    let allowed = physical_to_logical
        .values()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let used = runtime_model.placement_device_ids();
    if used.iter().any(|device| !allowed.contains(device.as_str())) {
        return Err(invalid_input(format!(
            "runtime placement uses undeclared logical devices: {used:?}"
        ))
        .into());
    }
    Ok(())
}

fn signal_processor_component_count(
    runtime_model: &VulkanResidentRuntimeModel,
) -> Result<usize, Box<dyn Error>> {
    let count = runtime_model
        .circuit_graph
        .components
        .iter()
        .filter(|component| component.runtime_role.is_signal_processor())
        .count();
    if count == 0 {
        return Err(invalid_input("mounted validation graph contains no signal processors").into());
    }
    Ok(count)
}

fn component_activation_count(
    model_activation_count: usize,
    signal_processor_component_count: usize,
) -> Result<usize, io::Error> {
    if signal_processor_component_count == 0 {
        return Err(invalid_input(
            "component activation accounting requires signal processors",
        ));
    }
    model_activation_count
        .checked_mul(signal_processor_component_count)
        .ok_or_else(|| invalid_input("component activation count overflowed"))
}

fn zero_speculative_statistics() -> Value {
    json!({
        "cycle_count": 0,
        "rollback_cycle_count": 0,
        "proposed_draft_tokens": 0,
        "accepted_draft_tokens": 0,
        "emitted_tokens": 0,
        "draft_time_ns": 0,
        "target_verification_time_ns": 0,
        "draft_catch_up_time_ns": 0,
        "total_time_ns": 0,
    })
}

fn zero_resident_feedback_statistics() -> Value {
    json!({
        "window_count": 0,
        "planned_tick_count": 0,
        "submitted_tick_count": 0,
        "executed_tick_count": 0,
        "retained_tick_count": 0,
        "sampled_tick_count": 0,
        "discarded_tick_count": 0,
        "template_record_count": 0,
        "template_replay_count": 0,
        "queue_submission_count": 0,
        "host_queue_submit_count": 0,
        "maximum_host_queue_submit_count_per_window": 0,
        "asynchronous_submission_count": 0,
        "completion_poll_count": 0,
        "bounded_wait_count": 0,
        "bounded_wait_timeout_count": 0,
    })
}

fn zero_transport_statistics() -> Value {
    json!({
        "published_packet_count": 0,
        "published_byte_count": 0,
        "received_packet_count": 0,
        "received_byte_count": 0,
        "direct_copy_count": 0,
        "direct_copy_byte_count": 0,
        "direct_receive_count": 0,
        "direct_receive_byte_count": 0,
    })
}

fn add_counters(
    total: &mut VulkanResidentExecutionCounters,
    value: VulkanResidentExecutionCounters,
) {
    total.saturating_accumulate(value);
}

fn resolve_package_path(root: &Path, raw: &str) -> PathBuf {
    let path = Path::new(raw);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
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

fn write_progress(
    output: &mut impl Write,
    request_id: &str,
    sequence: usize,
    payload: Value,
) -> Result<(), Box<dyn Error>> {
    serde_json::to_writer(
        &mut *output,
        &json!({
            "schema": PROGRESS_SCHEMA,
            "request_id": request_id,
            "sequence": sequence,
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

fn artifact_digest(bytes: &[u8]) -> String {
    format!(
        "nerve.optimizer.artifact_sha256.v1:{:x}",
        Sha256::digest(bytes)
    )
}

fn validation_state_digest(device_state_digest: &str) -> String {
    artifact_digest(device_state_digest.as_bytes())
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

    struct ByteCodec;

    impl VulkanResidentTokenTextCodec for ByteCodec {
        fn encode_text(
            &self,
            text: &str,
        ) -> Result<Vec<u32>, nerve_runtime::VulkanResidentTokenTextCodecError> {
            Ok(text.bytes().map(u32::from).collect())
        }

        fn decode_tokens(
            &self,
            token_ids: &[u32],
        ) -> Result<String, nerve_runtime::VulkanResidentTokenTextCodecError> {
            token_ids
                .iter()
                .map(|token| {
                    char::from_u32(*token).ok_or_else(|| {
                        nerve_runtime::VulkanResidentTokenTextCodecError::new("invalid byte token")
                    })
                })
                .collect()
        }
    }

    fn fixture_chat() -> RuntimeChatSession {
        RuntimeChatSession {
            formatter: nerve_runtime::RuntimeChatFormatter {
                template_source: concat!(
                    "{% for message in messages %}",
                    "{{ message.role }}:{{ message.content }}\\n",
                    "{% endfor %}",
                    "{% if add_generation_prompt %}assistant:{% endif %}",
                )
                .to_string(),
                template_variables: serde_json::Map::new(),
                render_time: chrono::Local::now().fixed_offset(),
                compiled_codec: None,
            },
            messages: Vec::new(),
            committed_token_ids: Vec::new(),
        }
    }

    #[test]
    fn validation_executor_protocol_rejects_unknown_fields() {
        let input = format!(
            "{{\"command\":\"close\",\"schema\":\"{COMMAND_SCHEMA}\",\"request_id\":\"close-1\",\"surprise\":true}}\n"
        );
        let error = read_command(&mut input.as_bytes()).unwrap_err();
        assert!(error.to_string().contains("unknown field"), "{error}");
    }

    #[test]
    fn product_conversation_sets_discard_loading_residency_before_measurement() {
        let policy = ConversationSetPolicy {
            minimum_sets: 2,
            maximum_sets: 3,
            repeat_second_set_if_residency_loaded: true,
        };
        let quiet = ConversationResidencyActivity {
            load_required_count: 0,
            physical_bytes_read: 0,
            reload_count: 0,
            uploaded_bytes: 0,
        };
        let loading = ConversationResidencyActivity {
            load_required_count: 1,
            physical_bytes_read: 4_194_304,
            reload_count: 0,
            uploaded_bytes: 4_194_304,
        };

        assert!(should_run_another_conversation_set(&policy, 1, quiet));
        assert!(!should_run_another_conversation_set(&policy, 2, quiet));
        assert!(should_run_another_conversation_set(&policy, 2, loading));
        assert!(!should_run_another_conversation_set(&policy, 3, loading));
    }

    #[test]
    fn conversation_set_policy_rejects_unbounded_or_nonconversation_warmups() {
        assert!(
            validate_conversation_set_policy(
                &ConversationSetPolicy {
                    minimum_sets: 2,
                    maximum_sets: 4,
                    repeat_second_set_if_residency_loaded: true,
                },
                "conversation",
            )
            .is_err()
        );
        assert!(
            validate_conversation_set_policy(
                &ConversationSetPolicy {
                    minimum_sets: 1,
                    maximum_sets: 1,
                    repeat_second_set_if_residency_loaded: false,
                },
                "teacher_forced",
            )
            .is_err()
        );
    }

    #[test]
    fn validation_output_digest_is_content_bound() {
        let first = artifact_digest(b"first");
        assert_eq!(first, artifact_digest(b"first"));
        assert_ne!(first, artifact_digest(b"second"));
        assert!(first.starts_with("nerve.optimizer.artifact_sha256.v1:"));
    }

    #[test]
    fn validation_conversation_completion_requires_eos_or_output_allowance() {
        assert_eq!(
            validation_conversation_stop_reason(&[7, 99], &[99], 2, false).unwrap(),
            "eos"
        );
        assert_eq!(
            validation_conversation_stop_reason(&[7, 8], &[99], 2, false).unwrap(),
            "output_allowance"
        );
        assert_eq!(
            validation_conversation_stop_reason(&[7], &[99], 2, true).unwrap(),
            "protocol_boundary"
        );
        assert!(validation_conversation_stop_reason(&[7], &[99], 2, false).is_err());
    }

    #[test]
    fn validation_progress_is_line_delimited_and_request_bound() {
        let mut output = Vec::new();
        write_progress(
            &mut output,
            "request-1",
            3,
            json!({"phase": "generation", "generated_tokens": 32}),
        )
        .unwrap();
        let document: Value = serde_json::from_slice(
            output
                .strip_suffix(b"\n")
                .expect("progress is newline terminated"),
        )
        .unwrap();
        assert_eq!(document["schema"], PROGRESS_SCHEMA);
        assert_eq!(document["request_id"], "request-1");
        assert_eq!(document["sequence"], 3);
        assert_eq!(document["payload"]["generated_tokens"], 32);
    }

    #[test]
    fn validation_response_uses_current_line_delimited_schema() {
        let mut output = Vec::new();
        write_response(
            &mut output,
            "request-1",
            "completed",
            json!({"resident_feedback": zero_resident_feedback_statistics()}),
        )
        .unwrap();
        let document: Value = serde_json::from_slice(
            output
                .strip_suffix(b"\n")
                .expect("response is newline terminated"),
        )
        .unwrap();
        assert_eq!(document["schema"], RESPONSE_SCHEMA);
        assert_eq!(
            document["schema"],
            "nerve.optimizer.validation_executor_response.v8"
        );
        assert_eq!(document["request_id"], "request-1");
        assert_eq!(document["status"], "completed");
        assert_eq!(
            document["payload"]["resident_feedback"]["maximum_host_queue_submit_count_per_window"],
            0
        );
    }

    #[test]
    fn validation_state_digest_content_binds_device_state_identity() {
        let first = validation_state_digest("nerve.optimizer.device_state_sha256.v1:first");
        assert_eq!(
            first,
            validation_state_digest("nerve.optimizer.device_state_sha256.v1:first",)
        );
        assert_ne!(
            first,
            validation_state_digest("nerve.optimizer.device_state_sha256.v1:second",)
        );
        assert!(first.starts_with("nerve.optimizer.artifact_sha256.v1:"));
    }

    #[test]
    fn validation_component_activation_horizon_counts_mounted_processors() {
        assert_eq!(component_activation_count(39, 64).unwrap(), 2_496);
        assert!(component_activation_count(1, 0).is_err());
        assert!(component_activation_count(usize::MAX, 2).is_err());
    }

    #[test]
    fn fixture_context_capacity_is_derived_from_the_exact_teacher_forced_transcript() {
        let chat = fixture_chat();
        let turns = vec!["Who are you?".to_string(), "Capital of Greece?".to_string()];
        let assistants = vec!["A model.".to_string(), "Athens.".to_string()];
        let exact =
            teacher_forced_fixture_context_capacity(&chat, &ByteCodec, &turns, &assistants, false)
                .unwrap();
        let lifecycle =
            teacher_forced_fixture_context_capacity(&chat, &ByteCodec, &turns, &assistants, true)
                .unwrap();
        let first_turn_only = teacher_forced_fixture_context_capacity(
            &chat,
            &ByteCodec,
            &turns[..1],
            &assistants[..1],
            false,
        )
        .unwrap();

        assert!(exact > first_turn_only);
        assert_eq!(lifecycle, exact + 1);
        assert!(
            teacher_forced_fixture_context_capacity(
                &chat,
                &ByteCodec,
                &turns,
                &assistants[..1],
                false,
            )
            .is_err()
        );
    }

    #[test]
    fn validation_package_identity_includes_residency_affecting_stream_configuration() {
        let key = ValidationPackageKey {
            package_manifest: PathBuf::from("/package/manifest.json"),
            candidate_id: Some("candidate".to_string()),
            physical_device_ids: vec!["device-a".to_string()],
            component_placement: BTreeMap::from([(
                "component".to_string(),
                "device-a".to_string(),
            )]),
            context_capacity: 131_072,
            speculative_draft_tokens: 3,
            sampler_config: VulkanResidentSamplerRuntimeConfig::default(),
            residency_policy: ResourceResidencyPolicy::DemandRetained,
            graph_operation: "none".to_string(),
            graph_target_component_id: None,
        };
        assert_eq!(key, key.clone());
        assert_ne!(
            key,
            ValidationPackageKey {
                context_capacity: 65_536,
                ..key.clone()
            }
        );
        assert_ne!(
            key,
            ValidationPackageKey {
                candidate_id: None,
                ..key.clone()
            }
        );
        assert_ne!(
            key,
            ValidationPackageKey {
                speculative_draft_tokens: 0,
                ..key.clone()
            }
        );
        assert_ne!(
            key,
            ValidationPackageKey {
                residency_policy: ResourceResidencyPolicy::Eager,
                ..key.clone()
            }
        );
        assert_ne!(
            key,
            ValidationPackageKey {
                sampler_config: VulkanResidentSamplerRuntimeConfig {
                    top_k: Some(1),
                    ..VulkanResidentSamplerRuntimeConfig::default()
                },
                ..key.clone()
            }
        );
    }

    #[test]
    fn validation_role_replacement_pre_evicts_only_moved_parameter_residency() {
        let key = ValidationPackageKey {
            package_manifest: PathBuf::from("/package/manifest.json"),
            candidate_id: Some("candidate-a".to_string()),
            physical_device_ids: vec!["device-a".to_string(), "device-b".to_string()],
            component_placement: BTreeMap::from([
                ("component-a".to_string(), "device-a".to_string()),
                ("component-b".to_string(), "device-b".to_string()),
            ]),
            context_capacity: 131_072,
            speculative_draft_tokens: 3,
            sampler_config: VulkanResidentSamplerRuntimeConfig::default(),
            residency_policy: ResourceResidencyPolicy::DemandRetained,
            graph_operation: "none".to_string(),
            graph_target_component_id: None,
        };
        assert!(!replacement_moves_parameter_residency(
            &key,
            &ValidationPackageKey {
                candidate_id: Some("candidate-b".to_string()),
                ..key.clone()
            },
        ));
        assert!(!replacement_moves_parameter_residency(
            &key,
            &ValidationPackageKey {
                context_capacity: 65_536,
                graph_operation: "duplicate".to_string(),
                ..key.clone()
            },
        ));
        assert!(replacement_moves_parameter_residency(
            &key,
            &ValidationPackageKey {
                component_placement: BTreeMap::from([
                    ("component-a".to_string(), "device-b".to_string(),),
                    ("component-b".to_string(), "device-a".to_string(),),
                ]),
                ..key.clone()
            },
        ));
    }
}
