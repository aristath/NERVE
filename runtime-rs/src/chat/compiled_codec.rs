use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::RuntimeChatMessage;
use crate::VulkanResidentTokenTextCodec;

pub const COMPILED_CHAT_CODEC_FILE: &str = "chat_codec.json";
const COMPILED_CHAT_CODEC_SCHEMA: &str = "nerve.chat_codec.v1";
const STRUCTURED_CODEC_KIND: &str = "role_delimited_interleaved_reasoning";

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CompiledChatCodec {
    schema: String,
    kind: String,
    tokens: CompiledChatTokens,
    templates: CompiledChatTemplates,
    reasoning: CompiledChatReasoning,
    tools: CompiledChatTools,
    tasks: BTreeMap<String, String>,
    response_parser: CompiledResponseParser,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CompiledChatTokens {
    bos: String,
    assistant_stop: String,
    thinking_start: String,
    thinking_end: String,
    tool_markup: String,
    user: String,
    assistant: String,
    latest_reminder: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CompiledChatTemplates {
    system: String,
    user: String,
    latest_reminder: String,
    assistant: String,
    assistant_without_stop: String,
    thinking: String,
    response_format: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CompiledChatReasoning {
    default_effort: String,
    effort_prompts: BTreeMap<String, String>,
    drop_previous_by_default: bool,
    preserve_when_tools_are_present: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CompiledChatTools {
    instructions_template: String,
    call_template: String,
    calls_template: String,
    calls_block_name: String,
    output_template: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CompiledResponseParser {
    kind: String,
    reject_special_tokens_in_content: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RuntimeAssistantStreamProtocolTokenKind {
    Forbidden(&'static str),
    ThinkingEnd,
    ToolMarkup,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RuntimeAssistantStreamProtocolToken {
    token_ids: Vec<u32>,
    kind: RuntimeAssistantStreamProtocolTokenKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeAssistantStreamProtocolValidator {
    thinking: bool,
    thinking_end_at: Option<usize>,
    observed_token_count: usize,
    recent_token_ids: Vec<u32>,
    maximum_protocol_token_length: usize,
    protocol_tokens: Vec<RuntimeAssistantStreamProtocolToken>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeAssistantStreamProtocolAction {
    Continue,
    TerminateAndTrim { token_count: usize },
}

impl RuntimeAssistantStreamProtocolValidator {
    pub fn observe(&mut self, token_id: u32) -> io::Result<RuntimeAssistantStreamProtocolAction> {
        self.observed_token_count = self.observed_token_count.saturating_add(1);
        self.recent_token_ids.push(token_id);
        if self.recent_token_ids.len() > self.maximum_protocol_token_length {
            let discard = self
                .recent_token_ids
                .len()
                .saturating_sub(self.maximum_protocol_token_length);
            self.recent_token_ids.drain(..discard);
        }
        let matched = self
            .protocol_tokens
            .iter()
            .find(|protocol| self.recent_token_ids.ends_with(&protocol.token_ids))
            .map(|protocol| (protocol.kind, protocol.token_ids.len()));
        match matched {
            Some((RuntimeAssistantStreamProtocolTokenKind::Forbidden(name), _)) => {
                Err(invalid_data(format!(
                    "generated assistant emitted reserved {name} token ending at generated token {}",
                    self.observed_token_count,
                )))
            }
            Some((RuntimeAssistantStreamProtocolTokenKind::ThinkingEnd, token_count)) => {
                if !self.thinking {
                    return Err(invalid_data(format!(
                        "generated assistant emitted reserved thinking_end token in non-thinking content ending at generated token {}",
                        self.observed_token_count,
                    )));
                }
                if let Some(first_end) = self.thinking_end_at {
                    let final_content_token_count = self
                        .observed_token_count
                        .saturating_sub(first_end)
                        .saturating_sub(token_count);
                    if final_content_token_count > 0 {
                        return Ok(RuntimeAssistantStreamProtocolAction::TerminateAndTrim {
                            token_count,
                        });
                    }
                    return Err(invalid_data(format!(
                        "generated assistant emitted adjacent thinking_end tokens ending at generated token {}",
                        self.observed_token_count,
                    )));
                }
                self.thinking_end_at = Some(self.observed_token_count);
                Ok(RuntimeAssistantStreamProtocolAction::Continue)
            }
            Some((RuntimeAssistantStreamProtocolTokenKind::ToolMarkup, _))
                if self.thinking && self.thinking_end_at.is_none() =>
            {
                Err(invalid_data(format!(
                    "generated assistant emitted reserved tool markup inside reasoning ending at generated token {}",
                    self.observed_token_count,
                )))
            }
            Some((RuntimeAssistantStreamProtocolTokenKind::ToolMarkup, _)) | None => {
                Ok(RuntimeAssistantStreamProtocolAction::Continue)
            }
        }
    }
}

impl CompiledChatCodec {
    pub fn from_path(path: &Path) -> io::Result<Self> {
        let codec: Self = serde_json::from_slice(&fs::read(path)?).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("compiled chat codec {path:?} is invalid: {error}"),
            )
        })?;
        codec.validate()?;
        Ok(codec)
    }

    fn validate(&self) -> io::Result<()> {
        if self.schema != COMPILED_CHAT_CODEC_SCHEMA || self.kind != STRUCTURED_CODEC_KIND {
            return Err(invalid_data("unsupported compiled chat codec"));
        }
        if self.tokens.bos.is_empty()
            || self.tokens.assistant_stop.is_empty()
            || self.tokens.thinking_start.is_empty()
            || self.tokens.thinking_end.is_empty()
            || self.tokens.user.is_empty()
            || self.tokens.assistant.is_empty()
            || self.reasoning.default_effort.is_empty()
            || !self
                .reasoning
                .effort_prompts
                .contains_key(&self.reasoning.default_effort)
        {
            return Err(invalid_data("compiled chat codec is incomplete"));
        }
        if self.response_parser.kind != "reasoning_content_and_typed_tool_calls" {
            return Err(invalid_data("compiled chat response parser is unsupported"));
        }
        Ok(())
    }

    pub fn format_text_messages(
        &self,
        messages: &[RuntimeChatMessage],
        variables: &Map<String, Value>,
        add_generation_prompt: bool,
    ) -> io::Result<String> {
        let messages = messages
            .iter()
            .map(|message| {
                serde_json::json!({
                    "role": message.role,
                    "content": message.content,
                })
            })
            .collect::<Vec<_>>();
        self.format_messages(&messages, variables, add_generation_prompt)
    }

    pub fn parse_generated_content(
        &self,
        content_without_stop: &str,
        stopped: bool,
        variables: &Map<String, Value>,
    ) -> io::Result<Value> {
        if !stopped {
            return Err(invalid_data(
                "structured assistant completion reached its token limit without a stop token",
            ));
        }
        let mut completion = content_without_stop.to_string();
        completion.push_str(&self.tokens.assistant_stop);
        let thinking = variables
            .get("enable_thinking")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        self.parse_completion(&completion, thinking)
    }

    pub fn assistant_stream_protocol_validator<C>(
        &self,
        codec: &C,
        variables: &Map<String, Value>,
    ) -> io::Result<Option<RuntimeAssistantStreamProtocolValidator>>
    where
        C: VulkanResidentTokenTextCodec,
    {
        if !self.response_parser.reject_special_tokens_in_content {
            return Ok(None);
        }
        let mut protocol_tokens = Vec::new();
        for (token, kind) in [
            (
                self.tokens.bos.as_str(),
                RuntimeAssistantStreamProtocolTokenKind::Forbidden("bos"),
            ),
            (
                self.tokens.thinking_start.as_str(),
                RuntimeAssistantStreamProtocolTokenKind::Forbidden("thinking_start"),
            ),
            (
                self.tokens.thinking_end.as_str(),
                RuntimeAssistantStreamProtocolTokenKind::ThinkingEnd,
            ),
            (
                self.tokens.tool_markup.as_str(),
                RuntimeAssistantStreamProtocolTokenKind::ToolMarkup,
            ),
        ] {
            if token.is_empty() {
                continue;
            }
            let token_ids = codec.encode_text(token).map_err(|error| {
                invalid_data(format!(
                    "could not encode compiled assistant protocol token {token:?}: {error}",
                ))
            })?;
            if token_ids.is_empty() {
                return Err(invalid_data(format!(
                    "compiled assistant protocol token {token:?} encoded to no tokens",
                )));
            }
            protocol_tokens.push(RuntimeAssistantStreamProtocolToken { token_ids, kind });
        }
        let maximum_protocol_token_length = protocol_tokens
            .iter()
            .map(|protocol| protocol.token_ids.len())
            .max()
            .unwrap_or(1);
        Ok(Some(RuntimeAssistantStreamProtocolValidator {
            thinking: variables
                .get("enable_thinking")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            thinking_end_at: None,
            observed_token_count: 0,
            recent_token_ids: Vec::with_capacity(maximum_protocol_token_length),
            maximum_protocol_token_length,
            protocol_tokens,
        }))
    }

    pub fn format_messages(
        &self,
        messages: &[Value],
        variables: &Map<String, Value>,
        add_generation_prompt: bool,
    ) -> io::Result<String> {
        self.validate()?;
        let thinking = variables
            .get("enable_thinking")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        let reasoning_effort = variables
            .get("reasoning_effort")
            .and_then(Value::as_str)
            .unwrap_or(&self.reasoning.default_effort);
        let mut drop_thinking = variables
            .get("drop_thinking")
            .and_then(Value::as_bool)
            .unwrap_or(self.reasoning.drop_previous_by_default);
        let mut messages = merge_tool_messages(messages)?;
        sort_tool_results(&mut messages);
        if self.reasoning.preserve_when_tools_are_present
            && messages
                .iter()
                .any(|message| message.get("tools").is_some())
        {
            drop_thinking = false;
        }
        if thinking && drop_thinking {
            messages = drop_previous_reasoning(messages);
        }
        let mut prompt = self.tokens.bos.clone();
        for index in 0..messages.len() {
            prompt.push_str(&self.render_message(
                index,
                &messages,
                thinking,
                drop_thinking,
                reasoning_effort,
                add_generation_prompt,
            )?);
        }
        Ok(prompt)
    }

    fn render_message(
        &self,
        index: usize,
        messages: &[Value],
        thinking: bool,
        drop_thinking: bool,
        reasoning_effort: &str,
        add_generation_prompt: bool,
    ) -> io::Result<String> {
        let message = object(&messages[index], "chat message")?;
        let role = string_field(message, "role")?;
        let content = message.get("content").and_then(Value::as_str).unwrap_or("");
        let mut rendered = if index == 0 && thinking {
            self.reasoning
                .effort_prompts
                .get(reasoning_effort)
                .ok_or_else(|| invalid_data("unsupported reasoning effort"))?
                .clone()
        } else {
            String::new()
        };
        match role {
            "system" => {
                rendered.push_str(&substitute(&self.templates.system, &[("content", content)]));
                if let Some(tools) = message.get("tools").and_then(Value::as_array) {
                    rendered.push_str("\n\n");
                    rendered.push_str(&self.render_tools(tools)?);
                }
                if let Some(format) = message.get("response_format") {
                    rendered.push_str("\n\n");
                    rendered.push_str(&substitute(
                        &self.templates.response_format,
                        &[("schema", &json_with_spaces(format))],
                    ));
                }
            }
            "developer" => {
                if content.is_empty() {
                    return Err(invalid_data("developer messages require content"));
                }
                let mut developer = format!("{}{}", self.tokens.user, content);
                if let Some(tools) = message.get("tools").and_then(Value::as_array) {
                    developer.push_str("\n\n");
                    developer.push_str(&self.render_tools(tools)?);
                }
                rendered.push_str(&substitute(
                    &self.templates.user,
                    &[("content", &developer)],
                ));
            }
            "user" => {
                rendered.push_str(&self.tokens.user);
                rendered.push_str(&self.render_user_content(message)?);
            }
            "latest_reminder" => {
                rendered.push_str(&self.tokens.latest_reminder);
                rendered.push_str(&substitute(
                    &self.templates.latest_reminder,
                    &[("content", content)],
                ));
            }
            "assistant" => {
                let normalized = self.normalize_assistant_message(message, thinking)?;
                let assistant_content = normalized
                    .get("content")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let reasoning_content = normalized
                    .get("reasoning_content")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let previous_has_task = index > 0 && messages[index - 1].get("task").is_some();
                let reasoning = if thinking
                    && !previous_has_task
                    && (!drop_thinking || index as isize > last_user_index(messages))
                {
                    format!(
                        "{}{}",
                        substitute(
                            &self.templates.thinking,
                            &[("reasoning_content", reasoning_content)],
                        ),
                        self.tokens.thinking_end,
                    )
                } else {
                    String::new()
                };
                let tool_calls = normalized
                    .get("tool_calls")
                    .and_then(Value::as_array)
                    .map(|calls| self.render_tool_calls(calls))
                    .transpose()?
                    .unwrap_or_default();
                let template = if message
                    .get("wo_eos")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    &self.templates.assistant_without_stop
                } else {
                    &self.templates.assistant
                };
                rendered.push_str(&substitute(
                    template,
                    &[
                        ("reasoning", &reasoning),
                        ("content", assistant_content),
                        ("tool_calls", &tool_calls),
                    ],
                ));
            }
            "tool" => return Err(invalid_data("tool messages were not preprocessed")),
            _ => return Err(invalid_data(format!("unsupported chat role {role:?}"))),
        }

        let followed_by_continuation = index + 1 < messages.len()
            && !matches!(
                messages[index + 1].get("role").and_then(Value::as_str),
                Some("assistant" | "latest_reminder")
            );
        if followed_by_continuation {
            return Ok(rendered);
        }
        if let Some(task) = message.get("task").and_then(Value::as_str) {
            let task_token = self
                .tasks
                .get(task)
                .ok_or_else(|| invalid_data("unsupported quick task"))?;
            if task == "action" {
                rendered.push_str(&self.tokens.assistant);
                rendered.push_str(if thinking {
                    &self.tokens.thinking_start
                } else {
                    &self.tokens.thinking_end
                });
            }
            rendered.push_str(task_token);
        } else if matches!(role, "user" | "developer")
            && (index + 1 < messages.len() || add_generation_prompt)
        {
            rendered.push_str(&self.tokens.assistant);
            if thinking && (!drop_thinking || index as isize >= last_user_index(messages)) {
                rendered.push_str(&self.tokens.thinking_start);
            } else {
                rendered.push_str(&self.tokens.thinking_end);
            }
        }
        Ok(rendered)
    }

    fn normalize_assistant_message(
        &self,
        message: &Map<String, Value>,
        thinking: bool,
    ) -> io::Result<Map<String, Value>> {
        if message.contains_key("reasoning_content") || message.contains_key("tool_calls") {
            return Ok(message.clone());
        }
        let content = message.get("content").and_then(Value::as_str).unwrap_or("");
        let has_protocol = content.contains(&self.tokens.thinking_end)
            || content.contains(&format!(
                "<{}{}>",
                self.tokens.tool_markup, self.tools.calls_block_name
            ));
        if !has_protocol {
            return Ok(message.clone());
        }
        let mut completion = content.to_string();
        completion.push_str(&self.tokens.assistant_stop);
        let parsed = self.parse_completion(&completion, thinking)?;
        object(&parsed, "parsed assistant message").cloned()
    }

    fn render_user_content(&self, message: &Map<String, Value>) -> io::Result<String> {
        let Some(blocks) = message.get("content_blocks").and_then(Value::as_array) else {
            return Ok(message
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string());
        };
        let mut parts = Vec::new();
        for block in blocks {
            let block = object(block, "user content block")?;
            match block.get("type").and_then(Value::as_str) {
                Some("text") => parts.push(
                    block
                        .get("text")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                ),
                Some("tool_result") => {
                    let content = match block.get("content") {
                        Some(Value::Array(items)) => items
                            .iter()
                            .map(|item| match item.get("type").and_then(Value::as_str) {
                                Some("text") => item
                                    .get("text")
                                    .and_then(Value::as_str)
                                    .unwrap_or("")
                                    .to_string(),
                                kind => format!("[Unsupported {}]", kind.unwrap_or("unknown")),
                            })
                            .collect::<Vec<_>>()
                            .join("\n\n"),
                        Some(Value::String(content)) => content.clone(),
                        Some(value) => json_with_spaces(value),
                        None => String::new(),
                    };
                    parts.push(substitute(
                        &self.tools.output_template,
                        &[("content", &content)],
                    ));
                }
                kind => parts.push(format!("[Unsupported {}]", kind.unwrap_or("unknown"))),
            }
        }
        Ok(parts.join("\n\n"))
    }

    fn render_tools(&self, tools: &[Value]) -> io::Result<String> {
        let schemas = tools
            .iter()
            .map(|tool| {
                tool.get("function")
                    .ok_or_else(|| invalid_data("tool has no function schema"))
                    .map(json_with_spaces)
            })
            .collect::<io::Result<Vec<_>>>()?
            .join("\n");
        Ok(substitute(
            &self.tools.instructions_template,
            &[
                ("tool_schemas", &schemas),
                ("dsml_token", &self.tokens.tool_markup),
                ("thinking_start_token", &self.tokens.thinking_start),
                ("thinking_end_token", &self.tokens.thinking_end),
            ],
        ))
    }

    fn render_tool_calls(&self, calls: &[Value]) -> io::Result<String> {
        if calls.is_empty() {
            return Ok(String::new());
        }
        let mut rendered_calls = Vec::new();
        for call in calls {
            let function = call.get("function").unwrap_or(call);
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid_data("tool call has no name"))?;
            let arguments = function
                .get("arguments")
                .and_then(Value::as_str)
                .unwrap_or("{}");
            let parsed: Value = serde_json::from_str(arguments)
                .unwrap_or_else(|_| serde_json::json!({"arguments": arguments}));
            let arguments = parsed
                .as_object()
                .cloned()
                .unwrap_or_else(|| Map::from_iter([("arguments".to_string(), parsed)]));
            let parameters = arguments
                .iter()
                .map(|(key, value)| {
                    let (is_string, rendered) = match value {
                        Value::String(value) => (true, value.clone()),
                        value => (false, json_with_spaces(value)),
                    };
                    format!(
                        "<{}parameter name=\"{}\" string=\"{}\">{}</{}parameter>",
                        self.tokens.tool_markup, key, is_string, rendered, self.tokens.tool_markup,
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            rendered_calls.push(substitute(
                &self.tools.call_template,
                &[
                    ("dsml_token", &self.tokens.tool_markup),
                    ("name", name),
                    ("arguments", &parameters),
                ],
            ));
        }
        Ok(format!(
            "\n\n{}",
            substitute(
                &self.tools.calls_template,
                &[
                    ("dsml_token", &self.tokens.tool_markup),
                    ("tool_calls", &rendered_calls.join("\n")),
                    ("tc_block_name", &self.tools.calls_block_name),
                ],
            )
        ))
    }

    pub fn parse_completion(&self, text: &str, thinking: bool) -> io::Result<Value> {
        let (reasoning, remainder) = if thinking {
            text.split_once(&self.tokens.thinking_end)
                .ok_or_else(|| invalid_data("missing reasoning end token"))?
        } else {
            ("", text)
        };
        let tool_open = format!(
            "\n\n<{}{}>\n",
            self.tokens.tool_markup, self.tools.calls_block_name
        );
        let (content, tool_calls) = if let Some((content, payload)) =
            remainder.split_once(&tool_open)
        {
            let suffix = format!(
                "\n</{}{}>{}",
                self.tokens.tool_markup, self.tools.calls_block_name, self.tokens.assistant_stop,
            );
            let payload = payload
                .strip_suffix(&suffix)
                .ok_or_else(|| invalid_data("malformed tool-call block or missing stop token"))?;
            (content, self.parse_tool_calls(payload)?)
        } else {
            let content = remainder
                .strip_suffix(&self.tokens.assistant_stop)
                .ok_or_else(|| invalid_data("missing assistant stop token"))?;
            (content, Vec::new())
        };
        if self.response_parser.reject_special_tokens_in_content {
            for (name, token) in [
                ("bos", &self.tokens.bos),
                ("assistant_stop", &self.tokens.assistant_stop),
                ("thinking_start", &self.tokens.thinking_start),
                ("thinking_end", &self.tokens.thinking_end),
                ("tool_markup", &self.tokens.tool_markup),
            ] {
                if token.is_empty() {
                    continue;
                }
                if let Some(offset) = content.find(token) {
                    return Err(invalid_data(format!(
                        "assistant content contains reserved {name} token at byte {offset}",
                    )));
                }
                if let Some(offset) = reasoning.find(token) {
                    return Err(invalid_data(format!(
                        "assistant reasoning contains reserved {name} token at byte {offset}",
                    )));
                }
            }
        }
        Ok(serde_json::json!({
            "role": "assistant",
            "content": content,
            "reasoning_content": reasoning,
            "tool_calls": tool_calls,
        }))
    }

    fn parse_tool_calls(&self, payload: &str) -> io::Result<Vec<Value>> {
        let invoke_open = format!("<{}invoke", self.tokens.tool_markup);
        let invoke_close = format!("</{}invoke>", self.tokens.tool_markup);
        let parameter_open = format!("<{}parameter", self.tokens.tool_markup);
        let parameter_close = format!("</{}parameter>", self.tokens.tool_markup);
        let mut remaining = payload;
        let mut calls = Vec::new();
        while !remaining.is_empty() {
            remaining = remaining
                .strip_prefix(&invoke_open)
                .ok_or_else(|| invalid_data("malformed tool invocation"))?;
            let (header, after_header) = remaining
                .split_once(">\n")
                .ok_or_else(|| invalid_data("malformed tool invocation header"))?;
            let name = quoted_attribute(header, "name")?;
            remaining = after_header;
            let mut arguments = Map::new();
            while !remaining.starts_with(&invoke_close) {
                remaining = remaining
                    .strip_prefix(&parameter_open)
                    .ok_or_else(|| invalid_data("malformed tool parameter"))?;
                let (header, after_header) = remaining
                    .split_once('>')
                    .ok_or_else(|| invalid_data("malformed tool parameter header"))?;
                let name = quoted_attribute(header, "name")?.to_string();
                let is_string = quoted_attribute(header, "string")?;
                if arguments.contains_key(&name) {
                    return Err(invalid_data("duplicate tool parameter"));
                }
                let (raw, after_value) = after_header
                    .split_once(&parameter_close)
                    .ok_or_else(|| invalid_data("unterminated tool parameter"))?;
                let value = if is_string == "true" {
                    Value::String(raw.to_string())
                } else if is_string == "false" {
                    serde_json::from_str(raw)
                        .map_err(|_| invalid_data("invalid JSON tool parameter"))?
                } else {
                    return Err(invalid_data("invalid tool parameter string flag"));
                };
                arguments.insert(name, value);
                remaining = after_value.strip_prefix('\n').unwrap_or(after_value);
            }
            remaining = remaining
                .strip_prefix(&invoke_close)
                .expect("prefix was checked");
            calls.push(serde_json::json!({
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json_with_spaces(&Value::Object(arguments)),
                }
            }));
            remaining = remaining.strip_prefix('\n').unwrap_or(remaining);
        }
        Ok(calls)
    }
}

fn merge_tool_messages(messages: &[Value]) -> io::Result<Vec<Value>> {
    let mut merged: Vec<Value> = Vec::new();
    for message in messages {
        let message = object(message, "chat message")?;
        match message.get("role").and_then(Value::as_str) {
            Some("tool") => {
                let block = serde_json::json!({
                    "type": "tool_result",
                    "tool_use_id": message.get("tool_call_id").cloned().unwrap_or(Value::String(String::new())),
                    "content": message.get("content").cloned().unwrap_or(Value::String(String::new())),
                });
                if let Some(previous) = merged.last_mut().and_then(Value::as_object_mut)
                    && previous.get("role").and_then(Value::as_str) == Some("user")
                    && let Some(blocks) = previous
                        .get_mut("content_blocks")
                        .and_then(Value::as_array_mut)
                {
                    blocks.push(block);
                } else {
                    merged.push(serde_json::json!({"role": "user", "content_blocks": [block]}));
                }
            }
            Some("user") => {
                let block = serde_json::json!({
                    "type": "text",
                    "text": message.get("content").cloned().unwrap_or(Value::String(String::new())),
                });
                if let Some(previous) = merged.last_mut().and_then(Value::as_object_mut)
                    && previous.get("role").and_then(Value::as_str) == Some("user")
                    && previous.get("task").is_none()
                    && let Some(blocks) = previous
                        .get_mut("content_blocks")
                        .and_then(Value::as_array_mut)
                {
                    blocks.push(block);
                } else {
                    let mut copied = Map::new();
                    copied.insert("role".to_string(), Value::String("user".to_string()));
                    copied.insert(
                        "content".to_string(),
                        message
                            .get("content")
                            .cloned()
                            .unwrap_or(Value::String(String::new())),
                    );
                    copied.insert("content_blocks".to_string(), Value::Array(vec![block]));
                    for key in ["task", "wo_eos", "mask"] {
                        if let Some(value) = message.get(key) {
                            copied.insert(key.to_string(), value.clone());
                        }
                    }
                    merged.push(Value::Object(copied));
                }
            }
            Some(_) => merged.push(Value::Object(message.clone())),
            None => return Err(invalid_data("chat message has no role")),
        }
    }
    Ok(merged)
}

fn sort_tool_results(messages: &mut [Value]) {
    let mut order = BTreeMap::new();
    for message in messages {
        let Some(message) = message.as_object_mut() else {
            continue;
        };
        if message.get("role").and_then(Value::as_str) == Some("assistant") {
            if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
                order.clear();
                for (index, call) in calls.iter().enumerate() {
                    if let Some(id) = call.get("id").and_then(Value::as_str) {
                        order.insert(id.to_string(), index);
                    }
                }
            }
            continue;
        }
        let Some(blocks) = message
            .get_mut("content_blocks")
            .and_then(Value::as_array_mut)
        else {
            continue;
        };
        let mut tools = blocks
            .iter()
            .filter(|block| block.get("type").and_then(Value::as_str) == Some("tool_result"))
            .cloned()
            .collect::<Vec<_>>();
        if tools.len() < 2 || order.is_empty() {
            continue;
        }
        tools.sort_by_key(|block| {
            block
                .get("tool_use_id")
                .and_then(Value::as_str)
                .and_then(|id| order.get(id))
                .copied()
                .unwrap_or(0)
        });
        let mut tools = tools.into_iter();
        for block in blocks {
            if block.get("type").and_then(Value::as_str) == Some("tool_result") {
                *block = tools.next().expect("tool result count is stable");
            }
        }
    }
}

fn drop_previous_reasoning(messages: Vec<Value>) -> Vec<Value> {
    let last_user = last_user_index(&messages);
    messages
        .into_iter()
        .enumerate()
        .filter_map(|(index, mut message)| {
            let role = message.get("role").and_then(Value::as_str);
            if matches!(
                role,
                Some("user" | "system" | "tool" | "latest_reminder" | "direct_search_results")
            ) || index as isize >= last_user
            {
                Some(message)
            } else if role == Some("assistant") {
                message.as_object_mut()?.remove("reasoning_content");
                Some(message)
            } else {
                None
            }
        })
        .collect()
}

fn last_user_index(messages: &[Value]) -> isize {
    messages
        .iter()
        .rposition(|message| {
            matches!(
                message.get("role").and_then(Value::as_str),
                Some("user" | "developer")
            )
        })
        .map(|index| index as isize)
        .unwrap_or(-1)
}

fn object<'a>(value: &'a Value, label: &str) -> io::Result<&'a Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| invalid_data(format!("{label} must be an object")))
}

fn string_field<'a>(object: &'a Map<String, Value>, key: &str) -> io::Result<&'a str> {
    object
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid_data(format!("chat message has no {key}")))
}

fn quoted_attribute<'a>(header: &'a str, name: &str) -> io::Result<&'a str> {
    let prefix = format!(" {name}=\"");
    let start = header
        .find(&prefix)
        .ok_or_else(|| invalid_data(format!("missing {name} attribute")))?
        + prefix.len();
    let end = header[start..]
        .find('"')
        .ok_or_else(|| invalid_data(format!("unterminated {name} attribute")))?
        + start;
    Ok(&header[start..end])
}

fn substitute(template: &str, fields: &[(&str, &str)]) -> String {
    fields
        .iter()
        .fold(template.to_string(), |result, (name, value)| {
            result.replace(&format!("{{{name}}}"), value)
        })
}

fn json_with_spaces(value: &Value) -> String {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {
            serde_json::to_string(value).expect("JSON scalar serialization cannot fail")
        }
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(json_with_spaces)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(key, value)| format!(
                    "{}: {}",
                    serde_json::to_string(key).expect("JSON key serialization cannot fail"),
                    json_with_spaces(value),
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use serde_json::{Map, Value, json};

    use super::{
        COMPILED_CHAT_CODEC_FILE, COMPILED_CHAT_CODEC_SCHEMA, CompiledChatCodec,
        CompiledChatReasoning, CompiledChatTemplates, CompiledChatTokens, CompiledChatTools,
        CompiledResponseParser, RuntimeAssistantStreamProtocolAction, STRUCTURED_CODEC_KIND,
    };
    use crate::{
        RuntimeChatFormatter, RuntimeChatMessage, VulkanResidentTokenTextCodec,
        VulkanResidentTokenTextCodecError,
    };

    struct ByteCodec;

    impl VulkanResidentTokenTextCodec for ByteCodec {
        fn encode_text(&self, text: &str) -> Result<Vec<u32>, VulkanResidentTokenTextCodecError> {
            Ok(text.bytes().map(u32::from).collect())
        }

        fn decode_tokens(
            &self,
            token_ids: &[u32],
        ) -> Result<String, VulkanResidentTokenTextCodecError> {
            let bytes = token_ids
                .iter()
                .map(|token_id| {
                    u8::try_from(*token_id)
                        .map_err(|_| VulkanResidentTokenTextCodecError::new("token is not a byte"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            String::from_utf8(bytes)
                .map_err(|error| VulkanResidentTokenTextCodecError::new(error.to_string()))
        }
    }

    fn fixture_codec() -> CompiledChatCodec {
        CompiledChatCodec {
            schema: COMPILED_CHAT_CODEC_SCHEMA.to_string(),
            kind: STRUCTURED_CODEC_KIND.to_string(),
            tokens: CompiledChatTokens {
                bos: "<B>".to_string(),
                assistant_stop: "<E>".to_string(),
                thinking_start: "<T>".to_string(),
                thinking_end: "</T>".to_string(),
                tool_markup: "X".to_string(),
                user: "<U>".to_string(),
                assistant: "<A>".to_string(),
                latest_reminder: "<R>".to_string(),
            },
            templates: CompiledChatTemplates {
                system: "<S>{content}".to_string(),
                user: "{content}".to_string(),
                latest_reminder: "{content}".to_string(),
                assistant: "<A>{reasoning}{content}{tool_calls}<E>".to_string(),
                assistant_without_stop: "<A>{reasoning}{content}{tool_calls}".to_string(),
                thinking: "{reasoning_content}".to_string(),
                response_format: "{schema}".to_string(),
            },
            reasoning: CompiledChatReasoning {
                default_effort: "low".to_string(),
                effort_prompts: BTreeMap::from([
                    ("low".to_string(), "[LOW]".to_string()),
                    ("high".to_string(), "[HIGH]".to_string()),
                    ("max".to_string(), "[MAX]".to_string()),
                ]),
                drop_previous_by_default: true,
                preserve_when_tools_are_present: true,
            },
            tools: CompiledChatTools {
                instructions_template: "tools:{tool_schemas}".to_string(),
                call_template:
                    "<{dsml_token}invoke name=\"{name}\">\n{arguments}\n</{dsml_token}invoke>"
                        .to_string(),
                calls_template:
                    "<{dsml_token}{tc_block_name}>\n{tool_calls}\n</{dsml_token}{tc_block_name}>"
                        .to_string(),
                calls_block_name: "calls".to_string(),
                output_template: "result:{content}".to_string(),
            },
            tasks: BTreeMap::new(),
            response_parser: CompiledResponseParser {
                kind: "reasoning_content_and_typed_tool_calls".to_string(),
                reject_special_tokens_in_content: true,
            },
        }
    }

    fn observe_text(
        validator: &mut super::RuntimeAssistantStreamProtocolValidator,
        text: &str,
    ) -> std::io::Result<RuntimeAssistantStreamProtocolAction> {
        let mut action = RuntimeAssistantStreamProtocolAction::Continue;
        for token_id in ByteCodec.encode_text(text).unwrap() {
            action = validator.observe(token_id)?;
            if action != RuntimeAssistantStreamProtocolAction::Continue {
                break;
            }
        }
        Ok(action)
    }

    #[test]
    fn compiled_codec_stream_validator_rejects_adjacent_reasoning_boundaries() {
        let mut validator = fixture_codec()
            .assistant_stream_protocol_validator(
                &ByteCodec,
                &Map::from_iter([("enable_thinking".to_string(), Value::Bool(true))]),
            )
            .unwrap()
            .unwrap();

        observe_text(&mut validator, "reasoning</T>").unwrap();
        let error = observe_text(&mut validator, "</T>").unwrap_err();
        assert!(
            error
                .to_string()
                .contains("emitted adjacent thinking_end tokens")
        );
    }

    #[test]
    fn compiled_codec_stream_validator_terminates_at_a_redundant_close_after_final_content() {
        let mut validator = fixture_codec()
            .assistant_stream_protocol_validator(
                &ByteCodec,
                &Map::from_iter([("enable_thinking".to_string(), Value::Bool(true))]),
            )
            .unwrap()
            .unwrap();

        observe_text(&mut validator, "reasoning</T>final answer").unwrap();
        assert_eq!(
            observe_text(&mut validator, "</T>").unwrap(),
            RuntimeAssistantStreamProtocolAction::TerminateAndTrim { token_count: 4 },
        );
    }

    #[test]
    fn compiled_codec_stream_validator_enforces_reserved_tokens_across_token_boundaries() {
        let mut thinking = fixture_codec()
            .assistant_stream_protocol_validator(
                &ByteCodec,
                &Map::from_iter([("enable_thinking".to_string(), Value::Bool(true))]),
            )
            .unwrap()
            .unwrap();
        observe_text(&mut thinking, "rea").unwrap();
        let tool_error = observe_text(&mut thinking, "soningX").unwrap_err();
        assert!(
            tool_error
                .to_string()
                .contains("reserved tool markup inside reasoning")
        );

        let mut chat = fixture_codec()
            .assistant_stream_protocol_validator(
                &ByteCodec,
                &Map::from_iter([("enable_thinking".to_string(), Value::Bool(false))]),
            )
            .unwrap()
            .unwrap();
        let boundary_error = observe_text(&mut chat, "answer</T>").unwrap_err();
        assert!(
            boundary_error
                .to_string()
                .contains("in non-thinking content")
        );

        let mut forbidden = fixture_codec()
            .assistant_stream_protocol_validator(&ByteCodec, &Map::new())
            .unwrap()
            .unwrap();
        let start_error = observe_text(&mut forbidden, "reasoning<T>").unwrap_err();
        assert!(
            start_error
                .to_string()
                .contains("reserved thinking_start token")
        );
    }

    #[test]
    fn compiled_codec_drives_runtime_formatter_without_jinja() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "nerve-compiled-codec-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(&directory).unwrap();
        fs::write(
            directory.join(COMPILED_CHAT_CODEC_FILE),
            serde_json::to_vec(&fixture_codec()).unwrap(),
        )
        .unwrap();

        let formatter = RuntimeChatFormatter::from_tokenizer_dir(
            &directory,
            &BTreeMap::from([("reasoning_effort".to_string(), json!("high"))]),
        )
        .unwrap();
        let rendered = formatter
            .format_messages(
                &[
                    RuntimeChatMessage {
                        role: "user".to_string(),
                        content: "first".to_string(),
                    },
                    RuntimeChatMessage {
                        role: "assistant".to_string(),
                        content: "private reasoning</T>answer".to_string(),
                    },
                    RuntimeChatMessage {
                        role: "user".to_string(),
                        content: "second".to_string(),
                    },
                ],
                true,
            )
            .unwrap();

        let parsed = formatter
            .parse_assistant_completion("fresh reasoning</T>fresh answer", true)
            .unwrap();
        let truncated = formatter
            .parse_assistant_completion("fresh reasoning</T>truncated", false)
            .unwrap_err();

        fs::remove_dir_all(&directory).unwrap();
        assert_eq!(
            rendered,
            "<B>[HIGH]<U>first<A></T><A>answer<E><U>second<A><T>"
        );
        assert_eq!(parsed["reasoning_content"], "fresh reasoning");
        assert_eq!(parsed["content"], "fresh answer");
        assert!(truncated.to_string().contains("without a stop token"));
    }

    #[test]
    fn compiled_codec_formats_tool_calls_and_results_as_structured_messages() {
        let codec = fixture_codec();
        let rendered = codec
            .format_messages(
                &[
                    json!({
                        "role": "system",
                        "content": "Use tools.",
                        "tools": [{"function": {"name": "lookup", "parameters": {"type": "object"}}}],
                    }),
                    json!({"role": "user", "content": "Find Athens."}),
                    json!({
                        "role": "assistant",
                        "reasoning_content": "Need lookup.",
                        "content": "",
                        "tool_calls": [{
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{\"city\":\"Athens\"}"},
                        }],
                    }),
                    json!({"role": "tool", "tool_call_id": "call-1", "content": "Greece"}),
                ],
                &Map::new(),
                true,
            )
            .unwrap();

        assert!(rendered.contains("tools:{\"name\": \"lookup\""));
        assert!(rendered.contains("<Xinvoke name=\"lookup\">"));
        assert!(rendered.contains("<Xparameter name=\"city\" string=\"true\">Athens</Xparameter>"));
        assert!(rendered.contains("result:Greece"));
        assert!(rendered.ends_with("<A><T>"));
    }

    #[test]
    fn compiled_codec_parses_typed_tool_arguments_and_rejects_malformed_protocol() {
        let codec = fixture_codec();
        let parsed = codec
            .parse_completion(
                "reasoning</T>answer\n\n<Xcalls>\n<Xinvoke name=\"lookup\">\n<Xparameter name=\"city\" string=\"true\">Athens</Xparameter>\n<Xparameter name=\"days\" string=\"false\">3</Xparameter>\n</Xinvoke>\n</Xcalls><E>",
                true,
            )
            .unwrap();
        assert_eq!(
            parsed,
            json!({
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "reasoning",
                "tool_calls": [{
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": "{\"city\": \"Athens\", \"days\": 3}",
                    }
                }],
            })
        );
        assert!(
            codec
                .parse_completion("reasoning without boundary", true)
                .is_err()
        );
        assert!(codec.parse_completion("reasoning</T>answer", true).is_err());
        let reserved = codec
            .parse_completion("reasoning</T>answer<T><E>", true)
            .unwrap_err();
        assert_eq!(
            reserved.to_string(),
            "assistant content contains reserved thinking_start token at byte 6"
        );
    }

    #[test]
    fn compiled_codec_rejects_unsupported_reasoning_effort() {
        let error = fixture_codec()
            .format_messages(
                &[json!({"role": "user", "content": "hello"})],
                &Map::from_iter([(
                    "reasoning_effort".to_string(),
                    Value::String("medium".to_string()),
                )]),
                true,
            )
            .unwrap_err();
        assert!(error.to_string().contains("unsupported reasoning effort"));
    }
}
