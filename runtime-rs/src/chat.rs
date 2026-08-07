use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::fs;
use std::io;
use std::path::Path;
use std::time::Instant;

use chrono::{DateTime, FixedOffset, Local};
use minijinja::{Environment, Error as TemplateError, ErrorKind as TemplateErrorKind};
use serde::Serialize;

use crate::{
    RuntimeCriticalPathPhase, RuntimeCriticalPathReport, VulkanResidentExecutionCounters,
    VulkanResidentHfTokenizerTextCodec, VulkanResidentInProcessPlacedPromptEngine,
    VulkanResidentInProcessPlacedPromptEngineSubmittedInputRun, VulkanResidentModelPackageManifest,
    VulkanResidentOutputControl, VulkanResidentTokenInputEvent,
    VulkanResidentTokenRuntimeSchedulerOutputEvent, VulkanResidentTokenTextCodec,
    reset_runtime_critical_path_counters, reset_vulkan_resident_execution_counters,
    runtime_critical_path_device_phase_scope, runtime_critical_path_report,
    runtime_critical_path_span, vulkan_resident_execution_counters,
};

mod compiled_codec;
pub use compiled_codec::*;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RuntimeChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug)]
pub struct RuntimeChatSession {
    pub formatter: RuntimeChatFormatter,
    pub messages: Vec<serde_json::Value>,
    pub committed_token_ids: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimePreparedChatTurn {
    pub canonical_user_token_ids: Vec<u32>,
    pub user_token_delta: Vec<u32>,
    pub generation_prompt_token_delta: Vec<u32>,
}

pub struct VulkanResidentChatTransactionRun {
    pub generated_token_ids: Vec<u32>,
    pub assistant_content: String,
    pub assistant_message: serde_json::Value,
    pub canonical_committed_token_ids: Vec<u32>,
    pub assistant_token_delta: Vec<u32>,
    pub canonical_turn_token_delta: Vec<u32>,
    pub generation_event_id: String,
    pub user_run: VulkanResidentInProcessPlacedPromptEngineSubmittedInputRun,
    pub generation_run: VulkanResidentInProcessPlacedPromptEngineSubmittedInputRun,
    pub canonical_commit_run: VulkanResidentInProcessPlacedPromptEngineSubmittedInputRun,
    pub execution_counters: VulkanResidentExecutionCounters,
    pub critical_path: RuntimeCriticalPathReport,
    pub generation_terminated_by_protocol: bool,
    pub elapsed_ns: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeChatGeneratedOutputControl {
    Continue,
    TerminateAndTrim { token_count: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VulkanResidentChatTransactionPhase {
    UserCommitted,
    GenerationBranchCompleted,
    CanonicalTurnCommitted,
}

#[derive(Debug)]
pub struct RuntimeRecoverableChatTurnError {
    stage: &'static str,
    source: Box<dyn Error>,
}

impl RuntimeRecoverableChatTurnError {
    pub fn new(stage: &'static str, source: Box<dyn Error>) -> Self {
        Self { stage, source }
    }
}

impl fmt::Display for RuntimeRecoverableChatTurnError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.stage, self.source)
    }
}

impl Error for RuntimeRecoverableChatTurnError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(self.source.as_ref())
    }
}

fn recoverable_chat_turn_error(stage: &'static str, source: Box<dyn Error>) -> Box<dyn Error> {
    Box::new(RuntimeRecoverableChatTurnError::new(stage, source))
}

pub fn execute_vulkan_resident_chat_transaction<T, F, P>(
    engine: &mut VulkanResidentInProcessPlacedPromptEngine,
    stream_id: &str,
    chat_session: &RuntimeChatSession,
    transcript_codec: &T,
    stop_token_ids: &[u32],
    turn_index: usize,
    user_content: &str,
    prepared: &RuntimePreparedChatTurn,
    max_new_tokens: usize,
    mut on_output_event: F,
    mut on_phase_completed: P,
) -> Result<VulkanResidentChatTransactionRun, Box<dyn Error>>
where
    T: VulkanResidentTokenTextCodec,
    F: FnMut(
        VulkanResidentTokenRuntimeSchedulerOutputEvent,
    ) -> Result<RuntimeChatGeneratedOutputControl, Box<dyn Error>>,
    P: FnMut(
        VulkanResidentChatTransactionPhase,
        &VulkanResidentInProcessPlacedPromptEngine,
    ) -> Result<(), Box<dyn Error>>,
{
    reset_vulkan_resident_execution_counters();
    reset_runtime_critical_path_counters();
    let started = Instant::now();
    let protocol_span = runtime_critical_path_span(RuntimeCriticalPathPhase::Protocol);
    let outer_transaction = {
        let _state_commit = runtime_critical_path_span(RuntimeCriticalPathPhase::StateCommit);
        engine.begin_stream_transaction(stream_id)?
    };
    let transaction = (|| -> Result<VulkanResidentChatTransactionRun, Box<dyn Error>> {
        let user_run = engine.submit_input_event_until_idle(
            stream_id,
            VulkanResidentTokenInputEvent::new(
                format!("chat_{turn_index}_user"),
                prepared.user_token_delta.clone(),
                0,
            )
            .with_origin("runtime_chat_canonical_user"),
        )?;
        {
            let _telemetry = runtime_critical_path_span(RuntimeCriticalPathPhase::Telemetry);
            on_phase_completed(VulkanResidentChatTransactionPhase::UserCommitted, engine)?;
        }

        let mut generation_event = VulkanResidentTokenInputEvent::new(
            format!("chat_{turn_index}_generation_branch"),
            prepared.generation_prompt_token_delta.clone(),
            max_new_tokens,
        )
        .with_origin("runtime_chat_generation_branch");
        let generation_event_id = generation_event.id.clone();
        if !stop_token_ids.is_empty() {
            generation_event = generation_event.with_stop_tokens(stop_token_ids.to_vec());
        }
        let mut output_error: Option<Box<dyn Error>> = None;
        let mut observed_generated_token_count = 0usize;
        let mut protocol_retained_token_count = None;
        let generation_run = engine.submit_input_event_transactionally_until_idle_with_output(
            stream_id,
            generation_event,
            |event| {
                observed_generated_token_count = observed_generated_token_count.saturating_add(1);
                if output_error.is_some() || protocol_retained_token_count.is_some() {
                    return VulkanResidentOutputControl::Abort;
                }
                let _protocol = runtime_critical_path_span(RuntimeCriticalPathPhase::Protocol);
                match on_output_event(event) {
                    Ok(RuntimeChatGeneratedOutputControl::Continue) => {
                        VulkanResidentOutputControl::Continue
                    }
                    Ok(RuntimeChatGeneratedOutputControl::TerminateAndTrim { token_count }) => {
                        match observed_generated_token_count.checked_sub(token_count) {
                            Some(retained_token_count) if token_count > 0 => {
                                protocol_retained_token_count = Some(retained_token_count);
                            }
                            _ => {
                                output_error = Some(Box::new(io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    format!(
                                        "stream protocol requested trimming {token_count} token(s) after observing {observed_generated_token_count} generated token(s)",
                                    ),
                                )));
                            }
                        }
                        VulkanResidentOutputControl::Abort
                    }
                    Err(error) => {
                        output_error = Some(error);
                        VulkanResidentOutputControl::Abort
                    }
                }
            },
        )?;
        {
            let _telemetry = runtime_critical_path_span(RuntimeCriticalPathPhase::Telemetry);
            on_phase_completed(
                VulkanResidentChatTransactionPhase::GenerationBranchCompleted,
                engine,
            )?;
        }
        if let Some(error) = output_error {
            return Err(recoverable_chat_turn_error(
                "streaming generated assistant output failed before canonical commit",
                error,
            ));
        }
        let mut generated_token_ids = generation_run.generated_token_ids.clone();
        normalize_generated_tokens_at_protocol_boundary(
            &mut generated_token_ids,
            protocol_retained_token_count,
        )
        .map_err(|error| {
            recoverable_chat_turn_error(
                "normalizing generated assistant protocol boundary failed before canonical commit",
                Box::new(error),
            )
        })?;
        let generation_terminated_by_protocol = protocol_retained_token_count.is_some();
        let assistant_stopped = generation_terminated_by_protocol
            || generated_token_ids
                .last()
                .is_some_and(|token_id| stop_token_ids.contains(token_id));
        let assistant_content = transcript_codec
            .decode_tokens(assistant_content_token_ids(
                &generated_token_ids,
                stop_token_ids,
            ))
            .map_err(|error| {
                recoverable_chat_turn_error(
                    "decoding generated assistant output failed before canonical commit",
                    Box::new(error),
                )
            })?;
        let assistant_message = chat_session
            .formatter
            .parse_assistant_completion(&assistant_content, assistant_stopped)
            .map_err(|error| {
                recoverable_chat_turn_error(
                    "generated assistant protocol validation failed before canonical commit",
                    error,
                )
            })?;
        let (assistant_token_delta, canonical_committed_token_ids) = chat_session
            .render_assistant_commit_token_delta(
                prepared,
                user_content,
                &assistant_message,
                transcript_codec,
            )
            .map_err(|error| {
                recoverable_chat_turn_error(
                    "rendering the canonical assistant turn failed before commit",
                    error,
                )
            })?;
        let mut canonical_turn_token_delta = prepared.user_token_delta.clone();
        canonical_turn_token_delta.extend_from_slice(&assistant_token_delta);
        let canonical_commit_run = {
            let _state_commit = runtime_critical_path_span(RuntimeCriticalPathPhase::StateCommit);
            let _device_state_commit =
                runtime_critical_path_device_phase_scope(RuntimeCriticalPathPhase::StateCommit);
            engine.submit_input_event_until_idle(
                stream_id,
                VulkanResidentTokenInputEvent::new(
                    format!("chat_{turn_index}_canonical_assistant"),
                    assistant_token_delta.clone(),
                    0,
                )
                .with_origin("runtime_chat_canonical_assistant"),
            )?
        };
        {
            let _telemetry = runtime_critical_path_span(RuntimeCriticalPathPhase::Telemetry);
            on_phase_completed(
                VulkanResidentChatTransactionPhase::CanonicalTurnCommitted,
                engine,
            )?;
        }
        Ok(VulkanResidentChatTransactionRun {
            generated_token_ids,
            assistant_content,
            assistant_message,
            canonical_committed_token_ids,
            assistant_token_delta,
            canonical_turn_token_delta,
            generation_event_id,
            user_run,
            generation_run,
            canonical_commit_run,
            execution_counters: vulkan_resident_execution_counters(),
            critical_path: RuntimeCriticalPathReport::default(),
            generation_terminated_by_protocol,
            elapsed_ns: 0,
        })
    })();

    match transaction {
        Ok(mut transaction) => {
            {
                let _state_commit =
                    runtime_critical_path_span(RuntimeCriticalPathPhase::StateCommit);
                engine.commit_stream_transaction(outer_transaction)?;
            }
            drop(protocol_span);
            transaction.elapsed_ns = u64::try_from(started.elapsed().as_nanos())
                .unwrap_or(u64::MAX)
                .max(1);
            transaction.critical_path = runtime_critical_path_report(transaction.elapsed_ns);
            Ok(transaction)
        }
        Err(error) => match {
            let _state_commit = runtime_critical_path_span(RuntimeCriticalPathPhase::StateCommit);
            engine.restore_stream_transaction(outer_transaction)
        } {
            Ok(()) => Err(error),
            Err(restore_error) => Err(Box::new(io::Error::other(format!(
                "chat turn failed ({error}) and canonical state rollback also failed ({restore_error})",
            )))),
        },
    }
}

pub fn assistant_content_token_ids<'a>(
    generated_token_ids: &'a [u32],
    stop_token_ids: &[u32],
) -> &'a [u32] {
    let mut content_len = generated_token_ids.len();
    while content_len > 0 && stop_token_ids.contains(&generated_token_ids[content_len - 1]) {
        content_len -= 1;
    }
    &generated_token_ids[..content_len]
}

pub fn normalize_generated_tokens_at_protocol_boundary(
    generated_token_ids: &mut Vec<u32>,
    retained_token_count: Option<usize>,
) -> io::Result<()> {
    let Some(retained_token_count) = retained_token_count else {
        return Ok(());
    };
    if retained_token_count >= generated_token_ids.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "stream protocol retained {retained_token_count} token(s), but the completed generation contains only {} token(s)",
                generated_token_ids.len(),
            ),
        ));
    }
    generated_token_ids.truncate(retained_token_count);
    Ok(())
}

pub fn chat_transcript_codec(
    tokenizer_dir: &Path,
) -> Result<VulkanResidentHfTokenizerTextCodec, Box<dyn Error>> {
    Ok(
        VulkanResidentHfTokenizerTextCodec::from_model_dir(tokenizer_dir)?
            .with_add_special_tokens(false)
            .with_skip_special_tokens(false),
    )
}

impl RuntimeChatSession {
    pub fn from_tokenizer_dir(
        tokenizer_dir: &Path,
        template_variables: &BTreeMap<String, serde_json::Value>,
    ) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            formatter: RuntimeChatFormatter::from_tokenizer_dir(tokenizer_dir, template_variables)?,
            messages: Vec::new(),
            committed_token_ids: Vec::new(),
        })
    }

    pub fn prepare_user_turn<C>(
        &self,
        user_content: &str,
        codec: &C,
    ) -> Result<RuntimePreparedChatTurn, Box<dyn Error>>
    where
        C: VulkanResidentTokenTextCodec,
    {
        let mut messages = self.messages.clone();
        messages.push(serde_json::json!({
            "role": "user",
            "content": user_content,
        }));
        let canonical_user_text = self
            .formatter
            .format_structured_messages(&messages, false)?;
        let generation_prompt_text = self.formatter.format_structured_messages(&messages, true)?;
        let canonical_user_token_ids = codec.encode_text(&canonical_user_text)?;
        let generation_prompt_token_ids = codec.encode_text(&generation_prompt_text)?;
        if !canonical_user_token_ids.starts_with(&self.committed_token_ids) {
            return Err(Box::new(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "chat template rewrote {} already committed token(s) before the new user turn",
                    self.committed_token_ids.len(),
                ),
            )));
        }
        if !generation_prompt_token_ids.starts_with(&canonical_user_token_ids) {
            return Err(Box::new(io::Error::new(
                io::ErrorKind::InvalidData,
                "chat template generation prompt rewrote the canonical user-turn prefix",
            )));
        }
        let user_token_delta = canonical_user_token_ids[self.committed_token_ids.len()..].to_vec();
        let generation_prompt_token_delta =
            generation_prompt_token_ids[canonical_user_token_ids.len()..].to_vec();
        if user_token_delta.is_empty() {
            return Err(Box::new(io::Error::new(
                io::ErrorKind::InvalidData,
                "chat template produced an empty user-turn token delta",
            )));
        }
        if generation_prompt_token_delta.is_empty() {
            return Err(Box::new(io::Error::new(
                io::ErrorKind::InvalidData,
                "chat template produced an empty assistant generation prompt",
            )));
        }
        Ok(RuntimePreparedChatTurn {
            canonical_user_token_ids,
            user_token_delta,
            generation_prompt_token_delta,
        })
    }

    pub fn render_assistant_commit_token_delta<C>(
        &self,
        prepared: &RuntimePreparedChatTurn,
        user_content: &str,
        assistant_message: &serde_json::Value,
        codec: &C,
    ) -> Result<(Vec<u32>, Vec<u32>), Box<dyn Error>>
    where
        C: VulkanResidentTokenTextCodec,
    {
        if assistant_message
            .get("role")
            .and_then(serde_json::Value::as_str)
            != Some("assistant")
        {
            return Err(Box::new(io::Error::new(
                io::ErrorKind::InvalidData,
                "parsed assistant message must have role assistant",
            )));
        }
        let mut messages = self.messages.clone();
        messages.push(serde_json::json!({
            "role": "user",
            "content": user_content,
        }));
        messages.push(assistant_message.clone());
        let render_with_next_user_probe = |probe: &str| -> Result<Vec<u32>, Box<dyn Error>> {
            let mut probed_messages = messages.clone();
            probed_messages.push(serde_json::json!({
                "role": "user",
                "content": probe,
            }));
            let rendered = self
                .formatter
                .format_structured_messages(&probed_messages, false)?;
            Ok(codec.encode_text(&rendered)?)
        };
        let left_probe = render_with_next_user_probe("NERVE_NEXT_USER_LEFT_PROBE_3EAF96A1")?;
        let right_probe = render_with_next_user_probe("ZERVE_NEXT_USER_RIGHT_PROBE_8D4C217B")?;
        let stable_prefix_len = left_probe
            .iter()
            .zip(&right_probe)
            .take_while(|(left, right)| left == right)
            .count();
        let canonical_turn_token_ids = left_probe[..stable_prefix_len].to_vec();
        if !canonical_turn_token_ids.starts_with(&prepared.canonical_user_token_ids) {
            return Err(Box::new(io::Error::new(
                io::ErrorKind::InvalidData,
                "chat template rewrote the canonical user-turn prefix while committing the assistant turn",
            )));
        }
        let assistant_token_delta =
            canonical_turn_token_ids[prepared.canonical_user_token_ids.len()..].to_vec();
        if assistant_token_delta.is_empty() {
            return Err(Box::new(io::Error::new(
                io::ErrorKind::InvalidData,
                "chat template produced an empty canonical assistant turn",
            )));
        }
        Ok((assistant_token_delta, canonical_turn_token_ids))
    }

    pub fn commit_assistant_turn(
        &mut self,
        user_content: &str,
        assistant_message: &serde_json::Value,
        canonical_token_ids: Vec<u32>,
    ) {
        debug_assert_eq!(
            assistant_message
                .get("role")
                .and_then(serde_json::Value::as_str),
            Some("assistant"),
        );
        self.messages.push(serde_json::json!({
            "role": "user",
            "content": user_content,
        }));
        self.messages.push(assistant_message.clone());
        self.committed_token_ids = canonical_token_ids;
    }
}

#[derive(Clone, Debug)]
pub struct RuntimeChatFormatter {
    pub template_source: String,
    pub template_variables: serde_json::Map<String, serde_json::Value>,
    pub render_time: DateTime<FixedOffset>,
    pub compiled_codec: Option<CompiledChatCodec>,
}

impl RuntimeChatFormatter {
    pub fn from_tokenizer_dir(
        tokenizer_dir: &Path,
        variable_overrides: &BTreeMap<String, serde_json::Value>,
    ) -> Result<Self, Box<dyn Error>> {
        let template_path = tokenizer_dir.join("chat_template.jinja");
        let mut template_variables = tokenizer_template_variables(tokenizer_dir)?;
        template_variables.extend(
            variable_overrides
                .iter()
                .map(|(name, value)| (name.clone(), value.clone())),
        );
        let codec_path = tokenizer_dir.join(COMPILED_CHAT_CODEC_FILE);
        let (template_source, compiled_codec) = if template_path.is_file() {
            (
                normalize_chat_template_for_runtime(&fs::read_to_string(&template_path)?),
                None,
            )
        } else if codec_path.is_file() {
            (
                String::new(),
                Some(CompiledChatCodec::from_path(&codec_path)?),
            )
        } else {
            return Err(Box::new(io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "chat mode requires a compiled chat codec or supported chat template in {tokenizer_dir:?}",
                ),
            )));
        };
        let formatter = Self {
            template_source,
            template_variables,
            render_time: Local::now().fixed_offset(),
            compiled_codec,
        };
        formatter.format_messages(
            &[RuntimeChatMessage {
                role: "user".to_string(),
                content: "template validation".to_string(),
            }],
            true,
        )?;
        Ok(formatter)
    }

    pub fn format_messages(
        &self,
        messages: &[RuntimeChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String, Box<dyn Error>> {
        let messages = messages
            .iter()
            .map(serde_json::to_value)
            .collect::<Result<Vec<_>, _>>()?;
        self.format_structured_messages(&messages, add_generation_prompt)
    }

    pub fn format_structured_messages(
        &self,
        messages: &[serde_json::Value],
        add_generation_prompt: bool,
    ) -> Result<String, Box<dyn Error>> {
        if let Some(codec) = &self.compiled_codec {
            return Ok(codec.format_messages(
                messages,
                &self.template_variables,
                add_generation_prompt,
            )?);
        }
        let mut environment = Environment::new();
        environment
            .set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        environment.add_function(
            "raise_exception",
            |message: String| -> Result<String, TemplateError> {
                Err(TemplateError::new(
                    TemplateErrorKind::InvalidOperation,
                    message,
                ))
            },
        );
        let render_time = self.render_time;
        environment.add_function("strftime_now", move |format: String| {
            render_time.format(&format).to_string()
        });
        environment.add_template("chat", &self.template_source)?;

        let mut context = self.template_variables.clone();
        context.insert(
            "messages".to_string(),
            serde_json::Value::Array(messages.to_vec()),
        );
        context.insert(
            "add_generation_prompt".to_string(),
            serde_json::Value::Bool(add_generation_prompt),
        );
        Ok(environment.get_template("chat")?.render(context)?)
    }

    pub fn parse_assistant_completion(
        &self,
        assistant_content: &str,
        assistant_stopped: bool,
    ) -> Result<serde_json::Value, Box<dyn Error>> {
        if let Some(codec) = &self.compiled_codec {
            return Ok(codec.parse_generated_content(
                assistant_content,
                assistant_stopped,
                &self.template_variables,
            )?);
        }
        Ok(serde_json::json!({
            "role": "assistant",
            "content": assistant_content,
        }))
    }

    pub fn assistant_stream_protocol_validator<C>(
        &self,
        codec: &C,
    ) -> Result<Option<RuntimeAssistantStreamProtocolValidator>, Box<dyn Error>>
    where
        C: VulkanResidentTokenTextCodec,
    {
        let Some(compiled_codec) = &self.compiled_codec else {
            return Ok(None);
        };
        Ok(compiled_codec.assistant_stream_protocol_validator(codec, &self.template_variables)?)
    }
}

pub fn normalize_chat_template_for_runtime(source: &str) -> String {
    let mut normalized = String::with_capacity(source.len());
    let mut cursor = 0usize;
    while let Some(relative_start) = source[cursor..].find("{%") {
        let start = cursor + relative_start;
        let tag_body_start = start + 2;
        let Some(relative_end) = source[tag_body_start..].find("%}") else {
            break;
        };
        let end = tag_body_start + relative_end;
        let tag_body = &source[tag_body_start..end];
        let statement = tag_body.trim().trim_matches('-').trim();
        normalized.push_str(&source[cursor..start]);
        if matches!(statement, "generation" | "endgeneration") {
            normalized.push_str(if tag_body.starts_with('-') {
                "{#-"
            } else {
                "{#"
            });
            normalized.push_str(statement);
            normalized.push_str(if tag_body.ends_with('-') { "-#}" } else { "#}" });
        } else {
            normalized.push_str(&source[start..end + 2]);
        }
        cursor = end + 2;
    }
    normalized.push_str(&source[cursor..]);
    normalized
}

pub fn chat_stop_token_ids_from_manifest(
    manifest_dir: &Path,
    tokenizer_dir: &Path,
    manifest: &VulkanResidentModelPackageManifest,
    formatter: &RuntimeChatFormatter,
) -> Result<Vec<u32>, Box<dyn Error>> {
    let config_path = manifest_dir.join(&manifest.config_path);
    let eos_values = if config_path.is_file() {
        let config: serde_json::Value = serde_json::from_slice(&fs::read(&config_path)?)?;
        let raw_eos = config.get("eos_token_id");
        if let Some(id) = raw_eos.and_then(serde_json::Value::as_u64) {
            vec![id]
        } else if let Some(ids) = raw_eos.and_then(serde_json::Value::as_array) {
            ids.iter()
                .filter_map(serde_json::Value::as_u64)
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };
    let mut stop_token_ids = eos_values
        .into_iter()
        .map(|id| {
            u32::try_from(id).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("eos_token_id {id} does not fit in u32"),
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(Box::<dyn Error>::from)?;

    let tokenizer_config_path = tokenizer_dir.join("tokenizer_config.json");
    if tokenizer_config_path.is_file() {
        let tokenizer_config: serde_json::Value =
            serde_json::from_slice(&fs::read(&tokenizer_config_path)?)?;
        let eos_token = tokenizer_config.get("eos_token").and_then(|value| {
            value
                .as_str()
                .or_else(|| value.get("content").and_then(serde_json::Value::as_str))
        });
        if let Some(eos_token) = eos_token {
            let stop_codec = VulkanResidentHfTokenizerTextCodec::from_model_dir(tokenizer_dir)?
                .with_add_special_tokens(false)
                .with_skip_special_tokens(false);
            let encoded = stop_codec.encode_text(eos_token)?;
            let [token_id] = encoded.as_slice() else {
                return Err(Box::new(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "chat tokenizer eos_token {eos_token:?} must encode to exactly one token, got {encoded:?}",
                    ),
                )));
            };
            if !stop_token_ids.contains(token_id) {
                stop_token_ids.push(*token_id);
            }
        }
    }
    if let Some(token_id) = model_owned_assistant_turn_stop_token_id(tokenizer_dir, formatter)?
        && !stop_token_ids.contains(&token_id)
    {
        stop_token_ids.push(token_id);
    }
    Ok(stop_token_ids)
}

pub fn model_owned_assistant_turn_stop_token_id(
    tokenizer_dir: &Path,
    formatter: &RuntimeChatFormatter,
) -> Result<Option<u32>, Box<dyn Error>> {
    const ASSISTANT_SENTINEL: &str = "NERVE_ASSISTANT_TURN_CONTENT_SENTINEL_7F3A9C";
    let rendered = formatter.format_messages(
        &[
            RuntimeChatMessage {
                role: "user".to_string(),
                content: "Discover the model-owned assistant turn delimiter.".to_string(),
            },
            RuntimeChatMessage {
                role: "assistant".to_string(),
                content: ASSISTANT_SENTINEL.to_string(),
            },
        ],
        false,
    )?;
    let sentinel_start = rendered.rfind(ASSISTANT_SENTINEL).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "chat template did not preserve the synthetic assistant content used to discover its turn delimiter",
        )
    })?;
    let suffix = &rendered[sentinel_start + ASSISTANT_SENTINEL.len()..];
    let codec = VulkanResidentHfTokenizerTextCodec::from_model_dir(tokenizer_dir)?
        .with_add_special_tokens(false)
        .with_skip_special_tokens(false);
    let suffix_token_ids = codec.encode_text(suffix)?;
    let special_token_ids = tokenizer_special_token_ids(tokenizer_dir)?;
    Ok(first_special_token_id(
        &suffix_token_ids,
        &special_token_ids,
    ))
}

fn tokenizer_template_variables(
    tokenizer_dir: &Path,
) -> Result<serde_json::Map<String, serde_json::Value>, Box<dyn Error>> {
    let path = tokenizer_dir.join("tokenizer_config.json");
    if !path.is_file() {
        return Ok(serde_json::Map::new());
    }
    let config: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
    let object = config.as_object().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("tokenizer config {path:?} must contain a JSON object"),
        )
    })?;
    Ok(object
        .iter()
        .map(|(key, value)| {
            let value = if key.ends_with("_token") {
                value
                    .get("content")
                    .and_then(serde_json::Value::as_str)
                    .map(|content| serde_json::Value::String(content.to_string()))
                    .unwrap_or_else(|| value.clone())
            } else {
                value.clone()
            };
            (key.clone(), value)
        })
        .collect())
}

fn tokenizer_special_token_ids(tokenizer_dir: &Path) -> Result<BTreeSet<u32>, Box<dyn Error>> {
    let path = tokenizer_dir.join("tokenizer.json");
    let tokenizer: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
    tokenizer
        .get("added_tokens")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter(|token| {
            token
                .get("special")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
        })
        .filter_map(|token| token.get("id").and_then(serde_json::Value::as_u64))
        .map(|id| {
            u32::try_from(id).map_err(|_| {
                Box::<dyn Error>::from(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("special token id {id} in {path:?} does not fit in u32",),
                ))
            })
        })
        .collect()
}

fn first_special_token_id(token_ids: &[u32], special_token_ids: &BTreeSet<u32>) -> Option<u32> {
    token_ids
        .iter()
        .copied()
        .find(|token_id| special_token_ids.contains(token_id))
}
