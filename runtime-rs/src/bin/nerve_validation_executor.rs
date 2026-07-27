use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;

use nerve_runtime::{
    RuntimeChatSession, RuntimeStagedCandidate, VulkanComputeDevice, VulkanComputeDeviceCatalog,
    VulkanResidentExecutionCounters, VulkanResidentHfTokenizerTextCodec,
    VulkanResidentInProcessPlacedPromptEngine, VulkanResidentInProcessPlacedPromptStream,
    VulkanResidentModelPackageManifest, VulkanResidentRuntimeModel,
    VulkanResidentSamplerRuntimeConfig, VulkanResidentTokenInputEvent,
    chat_stop_token_ids_from_manifest, chat_transcript_codec,
    execute_vulkan_resident_chat_transaction, reset_vulkan_resident_execution_counters,
    vulkan_resident_execution_counters,
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const COMMAND_SCHEMA: &str = "nerve.optimizer.validation_executor_command.v1";
const RESPONSE_SCHEMA: &str = "nerve.optimizer.validation_executor_response.v2";
const AMD_VENDOR_ID: u32 = 0x1002;
const STREAM_ID: &str = "validation";

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
        context_capacity: Option<usize>,
        random_seed: u32,
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
        max_output_tokens: usize,
    },
    Close {
        schema: String,
        request_id: String,
    },
}

struct MountedValidation {
    package_id: String,
    candidate_id: Option<String>,
    physical_device_ids: Vec<String>,
    engine: VulkanResidentInProcessPlacedPromptEngine,
    chat: RuntimeChatSession,
    transcript_codec: VulkanResidentHfTokenizerTextCodec,
    stop_token_ids: Vec<u32>,
    mounted_state_digest: String,
    executed: bool,
}

fn main() {
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
    let command = read_command(&mut input)?
        .ok_or_else(|| invalid_input("executor input ended before mount"))?;
    let mount_started = Instant::now();
    let (mount_request_id, mut mounted) = mount(command)?;
    let mounted_state_digest = mounted.engine.stream_resident_state_digest(STREAM_ID)?;
    mounted.mounted_state_digest = mounted_state_digest.clone();
    write_response(
        &mut output,
        &mount_request_id,
        "mounted",
        json!({
            "package_id": mounted.package_id,
            "candidate_id": mounted.candidate_id,
            "physical_device_ids": mounted.physical_device_ids,
            "context_capacity": mounted
                .engine
                .stream(STREAM_ID)
                .expect("mounted stream exists")
                .package()
                .dynamic_state_capacity_activations,
            "mounted_state_digest": mounted_state_digest,
            "mount_duration_ns": nonzero_elapsed_ns(mount_started),
        }),
    )?;

    let close_request_id = loop {
        let command = read_command(&mut input)?.ok_or_else(|| {
            invalid_input("validation executor input ended without explicit close")
        })?;
        match command {
            ExecutorCommand::Execute {
                schema,
                request_id,
                turns,
                teacher_forced_assistant_turns,
                execution_mode,
                max_output_tokens,
            } => {
                require_schema(&schema)?;
                let report = mounted.execute(
                    turns,
                    teacher_forced_assistant_turns,
                    &execution_mode,
                    max_output_tokens,
                )?;
                write_response(&mut output, &request_id, "completed", report)?;
            }
            ExecutorCommand::Close { schema, request_id } => {
                require_schema(&schema)?;
                break request_id;
            }
            ExecutorCommand::Mount { .. } => {
                return Err(invalid_input("executor cannot mount a second role").into());
            }
        }
    };
    let release_started = Instant::now();
    let mounted_state_digest = mounted.mounted_state_digest.clone();
    let removed = mounted.engine.remove_stream(STREAM_ID)?;
    drop(mounted);
    write_response(
        &mut output,
        &close_request_id,
        "released",
        json!({
            "released": true,
            "mounted_state_digest": mounted_state_digest,
            "released_device_ids": removed.device_ids,
            "release_duration_ns": nonzero_elapsed_ns(release_started),
        }),
    )?;
    Ok(())
}

fn mount(command: ExecutorCommand) -> Result<(String, MountedValidation), Box<dyn Error>> {
    let ExecutorCommand::Mount {
        schema,
        request_id,
        package_manifest,
        candidate_root,
        candidate_id,
        physical_device_ids,
        component_placement,
        context_capacity,
        random_seed,
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
    let context_capacity = context_capacity.unwrap_or(manifest.max_context_activations);
    if context_capacity == 0 {
        return Err(invalid_input("resolved context_capacity must be positive").into());
    }
    if context_capacity > manifest.max_context_activations {
        return Err(invalid_input(format!(
            "requested context capacity {context_capacity} exceeds package limit {}",
            manifest.max_context_activations
        ))
        .into());
    }
    let tokenizer_dir = resolve_package_path(&package_root, &manifest.tokenizer.path);
    let (devices, physical_to_logical) = open_amd_devices(&physical_device_ids)?;
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
    validate_runtime_placement(&runtime_model, &physical_to_logical)?;
    let chat_variables =
        BTreeMap::from([("enable_thinking".to_string(), Value::Bool(enable_thinking))]);
    let chat = RuntimeChatSession::from_tokenizer_dir(&tokenizer_dir, &chat_variables)?;
    let stop_token_ids = chat_stop_token_ids_from_manifest(
        &package_root,
        &tokenizer_dir,
        &runtime_model.package,
        &chat.formatter,
    )?;
    let transcript_codec = chat_transcript_codec(&tokenizer_dir)?;
    let stream =
        VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices_with_sampler_config(
            devices,
            &package_root,
            runtime_model,
            Some(context_capacity),
            random_seed,
            0,
            VulkanResidentSamplerRuntimeConfig::default(),
        )?;
    let package_id = stream.package().package_id.clone();
    let mut engine = VulkanResidentInProcessPlacedPromptEngine::new();
    engine.add_stream(STREAM_ID, stream)?;
    Ok((
        request_id,
        MountedValidation {
            package_id,
            candidate_id,
            physical_device_ids,
            engine,
            chat,
            transcript_codec,
            stop_token_ids,
            mounted_state_digest: String::new(),
            executed: false,
        },
    ))
}

impl MountedValidation {
    fn execute(
        &mut self,
        turns: Vec<String>,
        teacher_forced_assistant_turns: Vec<String>,
        execution_mode: &str,
        max_output_tokens: usize,
    ) -> Result<Value, Box<dyn Error>> {
        if self.executed {
            return Err(
                invalid_input("validation role can execute only once before release").into(),
            );
        }
        self.executed = true;
        match execution_mode {
            "conversation" => self.execute_conversation(turns, max_output_tokens),
            "teacher_forced" => self.execute_teacher_forced(turns, teacher_forced_assistant_turns),
            "lifecycle" => self.execute_lifecycle(turns, max_output_tokens),
            _ => Err(invalid_input(format!(
                "unsupported validation execution_mode {execution_mode:?}"
            ))
            .into()),
        }
    }

    fn execute_conversation(
        &mut self,
        turns: Vec<String>,
        max_output_tokens: usize,
    ) -> Result<Value, Box<dyn Error>> {
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
            let prepared = self
                .chat
                .prepare_user_turn(user_content, &self.transcript_codec)?;
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
                |_| {},
            )?;
            let engine_runs = [
                &transaction.user_run.engine_run,
                &transaction.generation_run.engine_run,
                &transaction.commit_run.engine_run,
            ];
            let component_activations = engine_runs
                .iter()
                .map(|run| {
                    run.prefill_activation_count
                        .saturating_add(run.decode_activation_count)
                })
                .sum::<usize>();
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
            traces.push(json!({
                "turn_index": turn_index,
                "user": user_content,
                "assistant": transaction.assistant_content,
                "generated_token_ids": transaction.generated_token_ids,
                "canonical_committed_token_ids": transaction.canonical_committed_token_ids,
                "component_activations": component_activations,
                "scheduler_steps": scheduler_steps,
                "elapsed_ns": transaction.elapsed_ns,
                "execution_counters": transaction.execution_counters,
            }));
            behavioral_outputs.push(json!({
                "generated_token_ids": transaction.generated_token_ids,
                "canonical_committed_token_ids": transaction.canonical_committed_token_ids,
            }));
            self.chat.commit_assistant_turn(
                user_content,
                &transaction.assistant_content,
                transaction.canonical_committed_token_ids,
            );
        }
        let state_digest = self.engine.stream_resident_state_digest(STREAM_ID)?;
        let output_digest = artifact_digest(
            &serde_json::to_vec(&behavioral_outputs).expect("conversation output is serializable"),
        );
        Ok(json!({
            "output_digest": output_digest,
            "state_digest": state_digest,
            "steps": total_component_activations,
            "scheduler_steps": total_scheduler_steps,
            "elapsed_ns": nonzero_elapsed_ns(started),
            "turns": traces,
            "execution_counters": total_execution_counters,
        }))
    }

    fn execute_teacher_forced(
        &mut self,
        turns: Vec<String>,
        assistant_turns: Vec<String>,
    ) -> Result<Value, Box<dyn Error>> {
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
            let user_run = self.engine.submit_input_event_until_idle(
                STREAM_ID,
                VulkanResidentTokenInputEvent::new(
                    format!("teacher_{turn_index}_user"),
                    prepared.user_token_delta.clone(),
                    0,
                )
                .with_origin("validation_teacher_forced_user"),
            )?;
            let (assistant_commit_token_ids, canonical_committed_token_ids) =
                self.chat.render_assistant_commit_token_delta(
                    &prepared,
                    user_content,
                    assistant_content,
                    &self.transcript_codec,
                )?;
            let commit_run = self.engine.submit_input_event_until_idle(
                STREAM_ID,
                VulkanResidentTokenInputEvent::new(
                    format!("teacher_{turn_index}_assistant"),
                    assistant_commit_token_ids,
                    0,
                )
                .with_origin("validation_teacher_forced_assistant"),
            )?;
            let engine_runs = [&user_run.engine_run, &commit_run.engine_run];
            let component_activations = engine_runs
                .iter()
                .map(|run| {
                    run.prefill_activation_count
                        .saturating_add(run.decode_activation_count)
                })
                .sum::<usize>();
            let scheduler_steps = engine_runs
                .iter()
                .map(|run| run.scheduler_step_count)
                .sum::<usize>();
            let counters = vulkan_resident_execution_counters();
            add_counters(&mut total_execution_counters, counters);
            total_component_activations =
                total_component_activations.saturating_add(component_activations);
            total_scheduler_steps = total_scheduler_steps.saturating_add(scheduler_steps);
            traces.push(json!({
                "turn_index": turn_index,
                "user": user_content,
                "assistant": assistant_content,
                "generated_token_ids": [],
                "canonical_committed_token_ids": canonical_committed_token_ids,
                "component_activations": component_activations,
                "scheduler_steps": scheduler_steps,
                "elapsed_ns": nonzero_elapsed_ns(turn_started),
                "execution_counters": counters,
                "teacher_forced": true,
            }));
            behavioral_outputs.push(json!({
                "canonical_committed_token_ids": canonical_committed_token_ids,
            }));
            self.chat.commit_assistant_turn(
                user_content,
                assistant_content,
                canonical_committed_token_ids,
            );
        }
        Ok(json!({
            "output_digest": artifact_digest(
                &serde_json::to_vec(&behavioral_outputs)
                    .expect("teacher-forced output is serializable"),
            ),
            "state_digest": self
                .engine
                .stream_resident_state_digest(STREAM_ID)?,
            "steps": total_component_activations,
            "scheduler_steps": total_scheduler_steps,
            "elapsed_ns": nonzero_elapsed_ns(started),
            "turns": traces,
            "execution_counters": total_execution_counters,
        }))
    }

    fn execute_lifecycle(
        &mut self,
        turns: Vec<String>,
        max_output_tokens: usize,
    ) -> Result<Value, Box<dyn Error>> {
        let mut report = self.execute_conversation(turns, max_output_tokens)?;
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
                |_| {},
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
        let additional_activations = rollback
            .engine_run
            .prefill_activation_count
            .saturating_add(rollback.engine_run.decode_activation_count)
            .saturating_add(resumed.prefill_activation_count)
            .saturating_add(resumed.decode_activation_count);
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
            json!(
                object["steps"]
                    .as_u64()
                    .unwrap_or_default()
                    .saturating_add(u64::try_from(additional_activations).unwrap_or(u64::MAX),)
            ),
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
            json!(self.engine.stream_resident_state_digest(STREAM_ID)?),
        );
        object
            .get_mut("turns")
            .and_then(Value::as_array_mut)
            .expect("conversation report has turns")
            .last_mut()
            .and_then(Value::as_object_mut)
            .expect("conversation report has a final turn")
            .insert("lifecycle".to_string(), lifecycle);
        Ok(report)
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

fn open_amd_devices(
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
        if info.vendor_id != AMD_VENDOR_ID {
            return Err(invalid_input(format!(
                "validation execution requires AMD GPUs, but {:?} reports vendor 0x{:04x}",
                info.device_name, info.vendor_id
            ))
            .into());
        }
        let logical_device_id = format!("optimizer:device:{ordinal}");
        let device = Rc::new(catalog.open_physical_device_index(info.physical_device_index)?);
        physical_to_logical.insert(physical_device_id.clone(), logical_device_id.clone());
        devices.insert(logical_device_id, device);
    }
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

fn add_counters(
    total: &mut VulkanResidentExecutionCounters,
    value: VulkanResidentExecutionCounters,
) {
    total.resident_sequence_prepare_calls = total
        .resident_sequence_prepare_calls
        .saturating_add(value.resident_sequence_prepare_calls);
    total.resident_sequence_recorded_command_buffers = total
        .resident_sequence_recorded_command_buffers
        .saturating_add(value.resident_sequence_recorded_command_buffers);
    total.resident_sequence_reused_command_buffers = total
        .resident_sequence_reused_command_buffers
        .saturating_add(value.resident_sequence_reused_command_buffers);
    total.resident_sequence_queue_submits = total
        .resident_sequence_queue_submits
        .saturating_add(value.resident_sequence_queue_submits);
    total.resident_sequence_fence_waits = total
        .resident_sequence_fence_waits
        .saturating_add(value.resident_sequence_fence_waits);
    total.resident_queue_batch_submits = total
        .resident_queue_batch_submits
        .saturating_add(value.resident_queue_batch_submits);
    total.resident_queue_batch_commands = total
        .resident_queue_batch_commands
        .saturating_add(value.resident_queue_batch_commands);
    total.resident_copy_queue_submits = total
        .resident_copy_queue_submits
        .saturating_add(value.resident_copy_queue_submits);
    total.resident_copy_waits = total
        .resident_copy_waits
        .saturating_add(value.resident_copy_waits);
    total.execution_quantum_count = total
        .execution_quantum_count
        .saturating_add(value.execution_quantum_count);
    total.execution_quantum_region_count = total
        .execution_quantum_region_count
        .saturating_add(value.execution_quantum_region_count);
    total.execution_quantum_forced_yield_count = total
        .execution_quantum_forced_yield_count
        .saturating_add(value.execution_quantum_forced_yield_count);
    total.execution_quantum_estimated_work_units = total
        .execution_quantum_estimated_work_units
        .saturating_add(value.execution_quantum_estimated_work_units);
    total.execution_quantum_estimated_memory_bytes = total
        .execution_quantum_estimated_memory_bytes
        .saturating_add(value.execution_quantum_estimated_memory_bytes);
    total.execution_quantum_dispatch_count = total
        .execution_quantum_dispatch_count
        .saturating_add(value.execution_quantum_dispatch_count);
    total.execution_quantum_predicted_duration_ns = total
        .execution_quantum_predicted_duration_ns
        .saturating_add(value.execution_quantum_predicted_duration_ns);
    total.execution_quantum_actual_duration_ns = total
        .execution_quantum_actual_duration_ns
        .saturating_add(value.execution_quantum_actual_duration_ns);
    total.execution_quantum_max_region_count = total
        .execution_quantum_max_region_count
        .max(value.execution_quantum_max_region_count);
    total.execution_quantum_max_actual_duration_ns = total
        .execution_quantum_max_actual_duration_ns
        .max(value.execution_quantum_max_actual_duration_ns);
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

    #[test]
    fn validation_executor_protocol_rejects_unknown_fields() {
        let input = format!(
            "{{\"command\":\"close\",\"schema\":\"{COMMAND_SCHEMA}\",\"request_id\":\"close-1\",\"surprise\":true}}\n"
        );
        let error = read_command(&mut input.as_bytes()).unwrap_err();
        assert!(error.to_string().contains("unknown field"), "{error}");
    }

    #[test]
    fn validation_output_digest_is_content_bound() {
        let first = artifact_digest(b"first");
        assert_eq!(first, artifact_digest(b"first"));
        assert_ne!(first, artifact_digest(b"second"));
        assert!(first.starts_with("nerve.optimizer.artifact_sha256.v1:"));
    }
}
