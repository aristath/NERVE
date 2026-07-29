use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs;
use std::io;
use std::path::Path;
use std::time::Instant;

use chrono::{DateTime, FixedOffset, Local};
use minijinja::{Environment, Error as TemplateError, ErrorKind as TemplateErrorKind};
use serde::Serialize;

use crate::{
    VulkanResidentExecutionCounters, VulkanResidentHfTokenizerTextCodec,
    VulkanResidentInProcessPlacedPromptEngine,
    VulkanResidentInProcessPlacedPromptEngineSubmittedInputRun, VulkanResidentModelPackageManifest,
    VulkanResidentTokenInputEvent, VulkanResidentTokenRuntimeSchedulerOutputEvent,
    VulkanResidentTokenTextCodec, reset_vulkan_resident_execution_counters,
    vulkan_resident_execution_counters,
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RuntimeChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug)]
pub struct RuntimeChatSession {
    pub formatter: RuntimeChatFormatter,
    pub messages: Vec<RuntimeChatMessage>,
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
    pub canonical_committed_token_ids: Vec<u32>,
    pub assistant_commit_token_ids: Vec<u32>,
    pub generation_event_id: String,
    pub user_run: VulkanResidentInProcessPlacedPromptEngineSubmittedInputRun,
    pub generation_run: VulkanResidentInProcessPlacedPromptEngineSubmittedInputRun,
    pub commit_run: VulkanResidentInProcessPlacedPromptEngineSubmittedInputRun,
    pub execution_counters: VulkanResidentExecutionCounters,
    pub elapsed_ns: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VulkanResidentChatTransactionPhase {
    UserCommitted,
    GenerationCompleted,
    AssistantCommitted,
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
    F: FnMut(VulkanResidentTokenRuntimeSchedulerOutputEvent),
    P: FnMut(
        VulkanResidentChatTransactionPhase,
        &VulkanResidentInProcessPlacedPromptEngine,
    ) -> Result<(), Box<dyn Error>>,
{
    reset_vulkan_resident_execution_counters();
    let started = Instant::now();
    let user_run = engine.submit_input_event_until_idle(
        stream_id,
        VulkanResidentTokenInputEvent::new(
            format!("chat_{turn_index}_user"),
            prepared.user_token_delta.clone(),
            0,
        )
        .with_origin("runtime_chat_canonical_user"),
    )?;
    on_phase_completed(VulkanResidentChatTransactionPhase::UserCommitted, engine)?;
    let mut generation_event = VulkanResidentTokenInputEvent::new(
        format!("chat_{turn_index}_generation"),
        prepared.generation_prompt_token_delta.clone(),
        max_new_tokens,
    )
    .with_origin("runtime_chat_generation_branch");
    let generation_event_id = generation_event.id.clone();
    if !stop_token_ids.is_empty() {
        generation_event = generation_event.with_stop_tokens(stop_token_ids.to_vec());
    }
    let generation_run = engine.submit_input_event_transactionally_until_idle_with_output(
        stream_id,
        generation_event,
        &mut on_output_event,
    )?;
    on_phase_completed(
        VulkanResidentChatTransactionPhase::GenerationCompleted,
        engine,
    )?;
    let assistant_content = transcript_codec.decode_tokens(assistant_content_token_ids(
        &generation_run.generated_token_ids,
        stop_token_ids,
    ))?;
    let (assistant_commit_token_ids, canonical_committed_token_ids) = chat_session
        .render_assistant_commit_token_delta(
            prepared,
            user_content,
            &assistant_content,
            transcript_codec,
        )?;
    let commit_run = engine.submit_input_event_until_idle(
        stream_id,
        VulkanResidentTokenInputEvent::new(
            format!("chat_{turn_index}_assistant_commit"),
            assistant_commit_token_ids.clone(),
            0,
        )
        .with_origin("runtime_chat_canonical_assistant"),
    )?;
    on_phase_completed(
        VulkanResidentChatTransactionPhase::AssistantCommitted,
        engine,
    )?;
    Ok(VulkanResidentChatTransactionRun {
        generated_token_ids: generation_run.generated_token_ids.clone(),
        assistant_content,
        canonical_committed_token_ids,
        assistant_commit_token_ids,
        generation_event_id,
        user_run,
        generation_run,
        commit_run,
        execution_counters: vulkan_resident_execution_counters(),
        elapsed_ns: u64::try_from(started.elapsed().as_nanos())
            .unwrap_or(u64::MAX)
            .max(1),
    })
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
        messages.push(RuntimeChatMessage {
            role: "user".to_string(),
            content: user_content.to_string(),
        });
        let canonical_user_text = self.formatter.format_messages(&messages, false)?;
        let generation_prompt_text = self.formatter.format_messages(&messages, true)?;
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
        assistant_content: &str,
        codec: &C,
    ) -> Result<(Vec<u32>, Vec<u32>), Box<dyn Error>>
    where
        C: VulkanResidentTokenTextCodec,
    {
        let mut messages = self.messages.clone();
        messages.push(RuntimeChatMessage {
            role: "user".to_string(),
            content: user_content.to_string(),
        });
        messages.push(RuntimeChatMessage {
            role: "assistant".to_string(),
            content: assistant_content.to_string(),
        });
        let render_with_next_user_probe = |probe: &str| -> Result<Vec<u32>, Box<dyn Error>> {
            let mut probed_messages = messages.clone();
            probed_messages.push(RuntimeChatMessage {
                role: "user".to_string(),
                content: probe.to_string(),
            });
            let rendered = self.formatter.format_messages(&probed_messages, false)?;
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
        assistant_content: &str,
        canonical_token_ids: Vec<u32>,
    ) {
        self.messages.push(RuntimeChatMessage {
            role: "user".to_string(),
            content: user_content.to_string(),
        });
        self.messages.push(RuntimeChatMessage {
            role: "assistant".to_string(),
            content: assistant_content.to_string(),
        });
        self.committed_token_ids = canonical_token_ids;
    }
}

#[derive(Clone, Debug)]
pub struct RuntimeChatFormatter {
    pub template_source: String,
    pub template_variables: serde_json::Map<String, serde_json::Value>,
    pub render_time: DateTime<FixedOffset>,
}

impl RuntimeChatFormatter {
    pub fn from_tokenizer_dir(
        tokenizer_dir: &Path,
        variable_overrides: &BTreeMap<String, serde_json::Value>,
    ) -> Result<Self, Box<dyn Error>> {
        let template_path = tokenizer_dir.join("chat_template.jinja");
        let template = fs::read_to_string(&template_path).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!(
                    "chat mode requires a supported chat template; failed to read {:?}: {error}",
                    template_path,
                ),
            )
        })?;
        let mut template_variables = tokenizer_template_variables(tokenizer_dir)?;
        template_variables.extend(
            variable_overrides
                .iter()
                .map(|(name, value)| (name.clone(), value.clone())),
        );
        let formatter = Self {
            template_source: normalize_chat_template_for_runtime(&template),
            template_variables,
            render_time: Local::now().fixed_offset(),
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
        context.insert("messages".to_string(), serde_json::to_value(messages)?);
        context.insert(
            "add_generation_prompt".to_string(),
            serde_json::Value::Bool(add_generation_prompt),
        );
        Ok(environment.get_template("chat")?.render(context)?)
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
