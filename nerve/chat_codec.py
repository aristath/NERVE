from __future__ import annotations

import ast
import json
import re
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any

from nerve.compilation import Json, ModelCompileError, write_json


CHAT_CODEC_SCHEMA = "nerve.chat_codec.v1"
CHAT_CODEC_FILE = "chat_codec.json"
STRUCTURED_CODEC_KIND = "role_delimited_interleaved_reasoning"
JINJA_CODEC_KIND = "jinja_template"

_REQUIRED_FUNCTIONS = frozenset(
    ("encode_messages", "parse_message_from_completion_text")
)
_REQUIRED_CONSTANTS = frozenset(
    (
        "bos_token",
        "eos_token",
        "thinking_start_token",
        "thinking_end_token",
        "dsml_token",
        "USER_SP_TOKEN",
        "ASSISTANT_SP_TOKEN",
        "LATEST_REMINDER_SP_TOKEN",
        "DS_TASK_SP_TOKENS",
        "system_msg_template",
        "user_msg_template",
        "latest_reminder_msg_template",
        "assistant_msg_template",
        "assistant_msg_wo_eos_template",
        "thinking_template",
        "response_format_template",
        "tool_call_template",
        "tool_calls_template",
        "tool_calls_block_name",
        "tool_output_template",
        "REASONING_EFFORT_PROMPTS",
        "DEFAULT_REASONING_EFFORT",
        "TOOLS_TEMPLATE",
    )
)


class ChatCodecCompileError(ModelCompileError):
    pass


def discover_model_chat_interface(model_dir: Path) -> str | None:
    candidates = _discover_structural_encoding_modules(model_dir)
    if len(candidates) > 1:
        raise ChatCodecCompileError(
            "source contains multiple structural chat codecs: "
            + ", ".join(str(path.relative_to(model_dir)) for path in candidates)
        )
    if candidates:
        return STRUCTURED_CODEC_KIND
    if _discover_jinja_template(model_dir) is not None:
        return JINJA_CODEC_KIND
    return None


def compile_model_chat_codec(model_dir: Path, destination_dir: Path) -> Json | None:
    """Compile model-owned chat behavior into a portable data artifact."""

    model_dir = model_dir.expanduser()
    interface = discover_model_chat_interface(model_dir)
    candidates = _discover_structural_encoding_modules(model_dir)
    if interface == STRUCTURED_CODEC_KIND:
        artifact = _compile_structured_codec(model_dir, candidates[0])
        artifact["validation"] = _validate_model_owned_vectors(model_dir, artifact)
    elif interface == JINJA_CODEC_KIND:
        template = _discover_jinja_template(model_dir)
        assert template is not None
        artifact = {
            "schema": CHAT_CODEC_SCHEMA,
            "kind": JINJA_CODEC_KIND,
            "template_file": "chat_template.jinja",
            "response_parser": {"kind": "raw_assistant_text"},
        }
    else:
        return None

    destination_dir.mkdir(parents=True, exist_ok=True)
    write_json(destination_dir / CHAT_CODEC_FILE, artifact)
    return artifact


def _discover_structural_encoding_modules(model_dir: Path) -> list[Path]:
    encoding_dir = model_dir / "encoding"
    if not encoding_dir.is_dir():
        return []
    candidates = []
    for path in sorted(encoding_dir.glob("*.py")):
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except (OSError, SyntaxError):
            continue
        functions = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        if _REQUIRED_FUNCTIONS <= functions:
            candidates.append(path)
    return candidates


def _discover_jinja_template(model_dir: Path) -> str | None:
    template_path = model_dir / "chat_template.jinja"
    if template_path.is_file():
        return template_path.read_text()
    tokenizer_config = model_dir / "tokenizer_config.json"
    if not tokenizer_config.is_file():
        return None
    config = json.loads(tokenizer_config.read_text())
    template = config.get("chat_template") if isinstance(config, dict) else None
    return template if isinstance(template, str) and template else None


def _compile_structured_codec(model_dir: Path, source_path: Path) -> Json:
    source = source_path.read_text()
    tree = ast.parse(source, filename=str(source_path))
    values = _module_literal_assignments(tree)
    missing = sorted(_REQUIRED_CONSTANTS.difference(values))
    if missing:
        raise ChatCodecCompileError(
            f"structural chat codec {source_path!s} is missing constants: "
            + ", ".join(missing)
        )
    _require_string_constants(
        values,
        _REQUIRED_CONSTANTS
        - {
            "DS_TASK_SP_TOKENS",
            "REASONING_EFFORT_PROMPTS",
        },
    )
    task_tokens = _string_mapping(values["DS_TASK_SP_TOKENS"], "task tokens")
    effort_prompts = _string_mapping(
        values["REASONING_EFFORT_PROMPTS"], "reasoning effort prompts"
    )
    default_effort = str(values["DEFAULT_REASONING_EFFORT"])
    if default_effort not in effort_prompts:
        raise ChatCodecCompileError(
            "default reasoning effort is absent from the effort prompt mapping"
        )
    relative_source = str(source_path.relative_to(model_dir))
    return {
        "schema": CHAT_CODEC_SCHEMA,
        "kind": STRUCTURED_CODEC_KIND,
        "source": {
            "path": relative_source,
            "sha256": sha256(source.encode("utf-8")).hexdigest(),
        },
        "tokens": {
            "bos": values["bos_token"],
            "assistant_stop": values["eos_token"],
            "thinking_start": values["thinking_start_token"],
            "thinking_end": values["thinking_end_token"],
            "tool_markup": values["dsml_token"],
            "user": values["USER_SP_TOKEN"],
            "assistant": values["ASSISTANT_SP_TOKEN"],
            "latest_reminder": values["LATEST_REMINDER_SP_TOKEN"],
        },
        "templates": {
            "system": values["system_msg_template"],
            "user": values["user_msg_template"],
            "latest_reminder": values["latest_reminder_msg_template"],
            "assistant": values["assistant_msg_template"],
            "assistant_without_stop": values["assistant_msg_wo_eos_template"],
            "thinking": values["thinking_template"],
            "response_format": values["response_format_template"],
        },
        "reasoning": {
            "default_effort": default_effort,
            "effort_prompts": effort_prompts,
            "drop_previous_by_default": True,
            "preserve_when_tools_are_present": True,
        },
        "tools": {
            "instructions_template": values["TOOLS_TEMPLATE"],
            "call_template": values["tool_call_template"],
            "calls_template": values["tool_calls_template"],
            "calls_block_name": values["tool_calls_block_name"],
            "output_template": values["tool_output_template"],
        },
        "tasks": task_tokens,
        "response_parser": {
            "kind": "reasoning_content_and_typed_tool_calls",
            "reject_special_tokens_in_content": True,
        },
    }


def _module_literal_assignments(tree: ast.Module) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for node in tree.body:
        name: str | None = None
        expression: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                name = target.id
                expression = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            expression = node.value
        if name is None or expression is None:
            continue
        try:
            values[name] = _literal_expression(expression, values)
        except ChatCodecCompileError:
            if name in _REQUIRED_CONSTANTS:
                raise
    return values


def _literal_expression(node: ast.expr, values: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in values:
        return values[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _literal_expression(node.left, values) + _literal_expression(
            node.right, values
        )
    if isinstance(node, ast.Dict):
        return {
            _literal_expression(key, values): _literal_expression(value, values)
            for key, value in zip(node.keys, node.values, strict=True)
            if key is not None
        }
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        constructor = list if isinstance(node, ast.List) else tuple
        result = [_literal_expression(value, values) for value in node.elts]
        return set(result) if isinstance(node, ast.Set) else constructor(result)
    raise ChatCodecCompileError(
        f"chat codec constant uses unsupported expression {ast.dump(node)}"
    )


def _require_string_constants(
    values: dict[str, Any], names: set[str] | frozenset[str]
) -> None:
    invalid = sorted(name for name in names if not isinstance(values.get(name), str))
    if invalid:
        raise ChatCodecCompileError(
            "chat codec constants must be strings: " + ", ".join(invalid)
        )


def _string_mapping(value: Any, label: str) -> dict[str, str]:
    if (
        not isinstance(value, dict)
        or not value
        or any(
            not isinstance(key, str) or not isinstance(item, str)
            for key, item in value.items()
        )
    ):
        raise ChatCodecCompileError(f"{label} must map strings to strings")
    return dict(value)


def render_compiled_chat_messages(
    artifact: Json,
    messages: list[Json],
    *,
    thinking_mode: str,
    context: list[Json] | None = None,
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
    reasoning_effort: str | None = None,
) -> str:
    _require_structured_artifact(artifact)
    if thinking_mode not in {"chat", "thinking"}:
        raise ValueError(f"invalid thinking mode {thinking_mode!r}")
    context = deepcopy(context or [])
    messages = _merge_tool_messages(deepcopy(messages))
    context = _merge_tool_messages(context)
    combined = _sort_tool_results(context + messages)
    context = combined[: len(context)]
    messages = combined[len(context) :]
    full_messages = context + messages
    prompt = artifact["tokens"]["bos"] if add_default_bos_token and not context else ""
    effective_drop = drop_thinking
    if artifact["reasoning"]["preserve_when_tools_are_present"] and any(
        message.get("tools") for message in full_messages
    ):
        effective_drop = False
    if thinking_mode == "thinking" and effective_drop:
        dropped_context = _drop_previous_reasoning(context)
        full_messages = _drop_previous_reasoning(full_messages)
        context_len = len(dropped_context)
        messages_to_render = full_messages[context_len:]
    else:
        context_len = len(context)
        messages_to_render = full_messages[context_len:]
    for offset, _message in enumerate(messages_to_render):
        prompt += _render_message(
            artifact,
            context_len + offset,
            full_messages,
            thinking_mode=thinking_mode,
            drop_thinking=effective_drop,
            reasoning_effort=reasoning_effort,
        )
    return prompt


def _render_message(
    artifact: Json,
    index: int,
    messages: list[Json],
    *,
    thinking_mode: str,
    drop_thinking: bool,
    reasoning_effort: str | None,
) -> str:
    message = messages[index]
    role = message.get("role")
    tokens = artifact["tokens"]
    templates = artifact["templates"]
    effort = reasoning_effort or artifact["reasoning"]["default_effort"]
    effort_prompts = artifact["reasoning"]["effort_prompts"]
    if effort not in effort_prompts:
        raise ValueError(
            f"invalid reasoning effort {effort!r}; expected {sorted(effort_prompts)}"
        )
    prompt = (
        effort_prompts[effort] if index == 0 and thinking_mode == "thinking" else ""
    )
    content = message.get("content")
    tools = message.get("tools")
    response_format = message.get("response_format")
    if role == "system":
        prompt += templates["system"].format(content=content or "")
        if tools:
            prompt += "\n\n" + _render_tools(artifact, tools)
        if response_format:
            prompt += "\n\n" + templates["response_format"].format(
                schema=_to_json(response_format)
            )
    elif role == "developer":
        if not content:
            raise ValueError("developer messages require content")
        developer_content = tokens["user"] + str(content)
        if tools:
            developer_content += "\n\n" + _render_tools(artifact, tools)
        if response_format:
            developer_content += "\n\n" + templates["response_format"].format(
                schema=_to_json(response_format)
            )
        prompt += templates["user"].format(content=developer_content)
    elif role == "user":
        prompt += tokens["user"]
        content_blocks = message.get("content_blocks")
        if content_blocks:
            parts = []
            for block in content_blocks:
                block_type = block.get("type")
                if block_type == "text":
                    parts.append(block.get("text", ""))
                elif block_type == "tool_result":
                    tool_content = block.get("content", "")
                    if isinstance(tool_content, list):
                        tool_content = "\n\n".join(
                            item.get("text", "")
                            if item.get("type") == "text"
                            else f"[Unsupported {item.get('type')}]"
                            for item in tool_content
                        )
                    parts.append(
                        artifact["tools"]["output_template"].format(
                            content=tool_content
                        )
                    )
                else:
                    parts.append(f"[Unsupported {block_type}]")
            prompt += "\n\n".join(parts)
        else:
            prompt += content or ""
    elif role == "latest_reminder":
        prompt += tokens["latest_reminder"] + templates["latest_reminder"].format(
            content=content
        )
    elif role == "tool":
        raise ValueError("tool messages must be merged before rendering")
    elif role == "assistant":
        reasoning_part = ""
        call_content = _render_tool_calls(artifact, message.get("tool_calls") or [])
        previous_has_task = index > 0 and messages[index - 1].get("task") is not None
        if thinking_mode == "thinking" and not previous_has_task:
            last_user = _last_user_index(messages)
            if not drop_thinking or index > last_user:
                reasoning_part = (
                    templates["thinking"].format(
                        reasoning_content=message.get("reasoning_content") or ""
                    )
                    + tokens["thinking_end"]
                )
        template = (
            templates["assistant_without_stop"]
            if message.get("wo_eos", False)
            else templates["assistant"]
        )
        prompt += template.format(
            reasoning=reasoning_part,
            content=content or "",
            tool_calls=call_content,
        )
    else:
        raise ValueError(f"unsupported chat role {role!r}")

    if index + 1 < len(messages) and messages[index + 1].get("role") not in {
        "assistant",
        "latest_reminder",
    }:
        return prompt
    task = message.get("task")
    if task is not None:
        task_token = artifact["tasks"].get(task)
        if task_token is None:
            raise ValueError(f"unsupported quick task {task!r}")
        if task == "action":
            prompt += tokens["assistant"]
            prompt += (
                tokens["thinking_start"]
                if thinking_mode == "thinking"
                else tokens["thinking_end"]
            )
        prompt += task_token
    elif role in {"user", "developer"}:
        prompt += tokens["assistant"]
        if thinking_mode == "thinking" and (
            not drop_thinking or index >= _last_user_index(messages)
        ):
            prompt += tokens["thinking_start"]
        else:
            prompt += tokens["thinking_end"]
    return prompt


def _render_tools(artifact: Json, tools: list[Json]) -> str:
    definitions = [tool["function"] for tool in tools]
    return artifact["tools"]["instructions_template"].format(
        tool_schemas="\n".join(_to_json(definition) for definition in definitions),
        dsml_token=artifact["tokens"]["tool_markup"],
        thinking_start_token=artifact["tokens"]["thinking_start"],
        thinking_end_token=artifact["tokens"]["thinking_end"],
    )


def _render_tool_calls(artifact: Json, tool_calls: list[Json]) -> str:
    if not tool_calls:
        return ""
    rendered = []
    for call in tool_calls:
        function = call.get("function", call)
        arguments = function.get("arguments", "{}")
        try:
            values = json.loads(arguments)
        except (TypeError, json.JSONDecodeError):
            values = {"arguments": arguments}
        if not isinstance(values, dict):
            values = {"arguments": values}
        parameters = []
        marker = artifact["tokens"]["tool_markup"]
        for key, value in values.items():
            is_string = isinstance(value, str)
            parameters.append(
                f'<{marker}parameter name="{key}" '
                f'string="{str(is_string).lower()}">'
                f"{value if is_string else _to_json(value)}"
                f"</{marker}parameter>"
            )
        rendered.append(
            artifact["tools"]["call_template"].format(
                dsml_token=marker,
                name=function.get("name"),
                arguments="\n".join(parameters),
            )
        )
    return "\n\n" + artifact["tools"]["calls_template"].format(
        dsml_token=artifact["tokens"]["tool_markup"],
        tool_calls="\n".join(rendered),
        tc_block_name=artifact["tools"]["calls_block_name"],
    )


def _merge_tool_messages(messages: list[Json]) -> list[Json]:
    merged: list[Json] = []
    for message in messages:
        role = message.get("role")
        if role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": message.get("tool_call_id", ""),
                "content": message.get("content", ""),
            }
            if (
                merged
                and merged[-1].get("role") == "user"
                and "content_blocks" in merged[-1]
            ):
                merged[-1]["content_blocks"].append(block)
            else:
                merged.append({"role": "user", "content_blocks": [block]})
        elif role == "user":
            block = {"type": "text", "text": message.get("content", "")}
            if (
                merged
                and merged[-1].get("role") == "user"
                and "content_blocks" in merged[-1]
                and merged[-1].get("task") is None
            ):
                merged[-1]["content_blocks"].append(block)
            else:
                copied = {
                    "role": "user",
                    "content": message.get("content", ""),
                    "content_blocks": [block],
                }
                for key in ("task", "wo_eos", "mask"):
                    if key in message:
                        copied[key] = message[key]
                merged.append(copied)
        else:
            merged.append(message)
    return merged


def _sort_tool_results(messages: list[Json]) -> list[Json]:
    call_order: dict[str, int] = {}
    for message in messages:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            call_order = {}
            for index, call in enumerate(message["tool_calls"]):
                call_id = call.get("id") or call.get("function", {}).get("id", "")
                if call_id:
                    call_order[call_id] = index
        elif message.get("role") == "user" and message.get("content_blocks"):
            blocks = message["content_blocks"]
            tool_blocks = [
                block for block in blocks if block.get("type") == "tool_result"
            ]
            if len(tool_blocks) > 1 and call_order:
                ordered = sorted(
                    tool_blocks,
                    key=lambda block: call_order.get(block.get("tool_use_id", ""), 0),
                )
                iterator = iter(ordered)
                message["content_blocks"] = [
                    next(iterator) if block.get("type") == "tool_result" else block
                    for block in blocks
                ]
    return messages


def _drop_previous_reasoning(messages: list[Json]) -> list[Json]:
    last_user = _last_user_index(messages)
    kept = []
    for index, message in enumerate(messages):
        role = message.get("role")
        if (
            role
            in {"user", "system", "tool", "latest_reminder", "direct_search_results"}
            or index >= last_user
        ):
            kept.append(message)
        elif role == "assistant":
            copied = dict(message)
            copied.pop("reasoning_content", None)
            kept.append(copied)
    return kept


def _last_user_index(messages: list[Json]) -> int:
    return next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if messages[index].get("role") in {"user", "developer"}
        ),
        -1,
    )


def parse_compiled_chat_completion(
    artifact: Json, text: str, *, thinking_mode: str
) -> Json:
    _require_structured_artifact(artifact)
    if thinking_mode not in {"chat", "thinking"}:
        raise ValueError(f"invalid thinking mode {thinking_mode!r}")
    tokens = artifact["tokens"]
    remainder = text
    reasoning = ""
    if thinking_mode == "thinking":
        boundary = tokens["thinking_end"]
        if boundary not in remainder:
            raise ValueError("missing reasoning end token")
        reasoning, remainder = remainder.split(boundary, 1)
    marker = tokens["tool_markup"]
    block_name = artifact["tools"]["calls_block_name"]
    tool_open = f"\n\n<{marker}{block_name}>\n"
    tool_calls = []
    if tool_open in remainder:
        content, tool_payload = remainder.split(tool_open, 1)
        expected_suffix = f"\n</{marker}{block_name}>{tokens['assistant_stop']}"
        if not tool_payload.endswith(expected_suffix):
            raise ValueError(
                "malformed tool-call block or missing assistant stop token"
            )
        tool_payload = tool_payload[: -len(expected_suffix)]
        tool_calls = _parse_typed_tool_calls(marker, tool_payload)
    else:
        stop = tokens["assistant_stop"]
        if not remainder.endswith(stop):
            raise ValueError("missing assistant stop token")
        content = remainder[: -len(stop)]
    if artifact["response_parser"]["reject_special_tokens_in_content"]:
        forbidden = {
            tokens["bos"],
            tokens["assistant_stop"],
            tokens["thinking_start"],
            tokens["thinking_end"],
            tokens["tool_markup"],
        }
        if any(
            token and token in value
            for token in forbidden
            for value in (content, reasoning)
        ):
            raise ValueError("assistant content contains a reserved protocol token")
    return {
        "role": "assistant",
        "content": content,
        "reasoning_content": reasoning,
        "tool_calls": tool_calls,
    }


def _parse_typed_tool_calls(marker: str, payload: str) -> list[Json]:
    invoke_open = f"<{marker}invoke"
    invoke_close = f"</{marker}invoke>"
    parameter_open = f"<{marker}parameter"
    parameter_close = f"</{marker}parameter>"
    cursor = 0
    calls = []
    while cursor < len(payload):
        if not payload.startswith(invoke_open, cursor):
            raise ValueError("malformed tool invocation")
        header_end = payload.find(">\n", cursor + len(invoke_open))
        if header_end < 0:
            raise ValueError("malformed tool invocation header")
        header = payload[cursor + len(invoke_open) : header_end]
        match = re.fullmatch(r' name="(.*?)"', header, flags=re.DOTALL)
        if match is None or not match.group(1):
            raise ValueError("malformed tool name")
        name = match.group(1)
        cursor = header_end + 2
        arguments: dict[str, Any] = {}
        while not payload.startswith(invoke_close, cursor):
            if not payload.startswith(parameter_open, cursor):
                raise ValueError("malformed tool parameter")
            parameter_header_end = payload.find(">", cursor + len(parameter_open))
            if parameter_header_end < 0:
                raise ValueError("malformed tool parameter header")
            parameter_header = payload[
                cursor + len(parameter_open) : parameter_header_end
            ]
            parameter_match = re.fullmatch(
                r' name="(.*?)" string="(true|false)"', parameter_header
            )
            if parameter_match is None or not parameter_match.group(1):
                raise ValueError("malformed tool parameter header")
            key, is_string = parameter_match.groups()
            if key in arguments:
                raise ValueError(f"duplicate tool parameter {key!r}")
            value_start = parameter_header_end + 1
            value_end = payload.find(parameter_close, value_start)
            if value_end < 0:
                raise ValueError("unterminated tool parameter")
            raw_value = payload[value_start:value_end]
            if is_string == "true":
                value: Any = raw_value
            else:
                try:
                    value = json.loads(raw_value)
                except json.JSONDecodeError as error:
                    raise ValueError("invalid JSON tool parameter") from error
            arguments[key] = value
            cursor = value_end + len(parameter_close)
            if payload.startswith("\n", cursor):
                cursor += 1
        cursor += len(invoke_close)
        calls.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": _to_json(arguments),
                },
            }
        )
        if cursor < len(payload):
            if not payload.startswith("\n", cursor):
                raise ValueError("unexpected content between tool invocations")
            cursor += 1
    return calls


def _require_structured_artifact(artifact: Json) -> None:
    if artifact.get("schema") != CHAT_CODEC_SCHEMA:
        raise ValueError("unsupported compiled chat codec schema")
    if artifact.get("kind") != STRUCTURED_CODEC_KIND:
        raise ValueError("chat operation requires a structured compiled codec")


def _validate_model_owned_vectors(model_dir: Path, artifact: Json) -> Json:
    tests_dir = model_dir / "encoding" / "tests"
    cases = []
    if not tests_dir.is_dir():
        return {"method": "structural_contract", "cases": cases}
    for input_path in sorted(tests_dir.glob("test_input_*.json")):
        suffix = input_path.name.removeprefix("test_input_").removesuffix(".json")
        output_path = tests_dir / f"test_output_{suffix}.txt"
        if not output_path.is_file():
            raise ChatCodecCompileError(
                f"chat codec vector {input_path.name!r} has no expected output"
            )
        source = json.loads(input_path.read_text())
        messages = source.get("messages") if isinstance(source, dict) else source
        if not isinstance(messages, list):
            raise ChatCodecCompileError(
                f"chat codec vector {input_path.name!r} has no message list"
            )
        messages = deepcopy(messages)
        if isinstance(source, dict) and source.get("tools"):
            messages[0]["tools"] = source["tools"]
        expected = output_path.read_text()
        matching_modes = [
            mode
            for mode in ("chat", "thinking")
            if render_compiled_chat_messages(artifact, messages, thinking_mode=mode)
            == expected
        ]
        if len(matching_modes) != 1:
            raise ChatCodecCompileError(
                f"compiled chat codec does not uniquely reproduce vector {suffix!r}"
            )
        cases.append(
            {
                "id": suffix,
                "thinking_mode": matching_modes[0],
                "input_sha256": sha256(input_path.read_bytes()).hexdigest(),
                "output_sha256": sha256(expected.encode("utf-8")).hexdigest(),
            }
        )
    return {"method": "model_owned_golden_vectors", "cases": cases}


def _to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)
