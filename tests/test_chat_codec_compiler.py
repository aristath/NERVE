from __future__ import annotations

import json
from pathlib import Path

import pytest

from nerve.chat_codec import (
    ChatCodecCompileError,
    compile_model_chat_codec,
    discover_model_chat_interface,
    parse_compiled_chat_completion,
    render_compiled_chat_messages,
)
from nerve.model_package_assets import copy_tokenizer_package


MODEL_DIR = Path("/mnt/models/models/deepseek-v4/flash-0731/safetensors")


def test_discovers_model_owned_structural_chat_interface() -> None:
    assert (
        discover_model_chat_interface(MODEL_DIR)
        == "role_delimited_interleaved_reasoning"
    )


@pytest.mark.parametrize("case", (1, 2, 3, 4))
def test_compiled_codec_matches_model_owned_golden_prompts(
    tmp_path: Path, case: int
) -> None:
    artifact = compile_model_chat_codec(MODEL_DIR, tmp_path)
    assert artifact is not None
    fixture_dir = MODEL_DIR / "encoding" / "tests"
    source = json.loads((fixture_dir / f"test_input_{case}.json").read_text())
    messages = source["messages"] if isinstance(source, dict) else source
    if isinstance(source, dict) and source.get("tools"):
        messages[0]["tools"] = source["tools"]

    rendered = render_compiled_chat_messages(
        artifact,
        messages,
        thinking_mode="chat" if case == 4 else "thinking",
    )

    assert rendered == (fixture_dir / f"test_output_{case}.txt").read_text()


def test_compiled_codec_preserves_reasoning_effort_as_model_data(
    tmp_path: Path,
) -> None:
    artifact = compile_model_chat_codec(MODEL_DIR, tmp_path)
    assert artifact is not None
    messages = [{"role": "user", "content": "Solve this."}]

    low = render_compiled_chat_messages(
        artifact, messages, thinking_mode="thinking", reasoning_effort="low"
    )
    high = render_compiled_chat_messages(
        artifact, messages, thinking_mode="thinking", reasoning_effort="high"
    )
    maximum = render_compiled_chat_messages(
        artifact, messages, thinking_mode="thinking", reasoning_effort="max"
    )

    bos = artifact["tokens"]["bos"]
    low_body = low.removeprefix(bos)
    assert high == bos + artifact["reasoning"]["effort_prompts"]["high"] + low_body
    assert maximum == (bos + artifact["reasoning"]["effort_prompts"]["max"] + low_body)


def test_compiled_codec_parses_reasoning_content_and_typed_tool_arguments(
    tmp_path: Path,
) -> None:
    artifact = compile_model_chat_codec(MODEL_DIR, tmp_path)
    assert artifact is not None
    completion = (
        "Need the tool.</think>"
        "\n\n<｜DSML｜tool_calls>\n"
        '<｜DSML｜invoke name="lookup">\n'
        '<｜DSML｜parameter name="city" string="true">Athens</｜DSML｜parameter>\n'
        '<｜DSML｜parameter name="days" string="false">3</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls><｜end▁of▁sentence｜>"
    )

    parsed = parse_compiled_chat_completion(
        artifact, completion, thinking_mode="thinking"
    )

    assert parsed == {
        "role": "assistant",
        "content": "",
        "reasoning_content": "Need the tool.",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": '{"city": "Athens", "days": 3}',
                },
            }
        ],
    }


@pytest.mark.parametrize(
    ("completion", "message"),
    (
        ("reasoning without boundary", "missing reasoning end token"),
        ("reasoning</think>answer without eos", "missing assistant stop token"),
        (
            "reasoning</think>\n\n<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="x">\n'
            '<｜DSML｜parameter name="a" string="false">not-json</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n</｜DSML｜tool_calls><｜end▁of▁sentence｜>",
            "invalid JSON tool parameter",
        ),
    ),
)
def test_compiled_codec_rejects_malformed_assistant_protocol(
    tmp_path: Path, completion: str, message: str
) -> None:
    artifact = compile_model_chat_codec(MODEL_DIR, tmp_path)
    assert artifact is not None

    with pytest.raises(ValueError, match=message):
        parse_compiled_chat_completion(artifact, completion, thinking_mode="thinking")


def test_rejects_ambiguous_model_owned_encoding_modules(tmp_path: Path) -> None:
    (tmp_path / "tokenizer.json").write_text("{}")
    encoding = tmp_path / "encoding"
    encoding.mkdir()
    source = """
def encode_messages(): pass
def parse_message_from_completion_text(): pass
"""
    (encoding / "first.py").write_text(source)
    (encoding / "second.py").write_text(source)

    with pytest.raises(ChatCodecCompileError, match="multiple structural chat codecs"):
        compile_model_chat_codec(tmp_path, tmp_path / "compiled")


def test_chat_codec_compiler_has_no_model_family_switches() -> None:
    source = (Path(__file__).parents[1] / "nerve" / "chat_codec.py").read_text()
    assert "deepseek" not in source.lower()
    assert "dsv4" not in source.lower()


def test_tokenizer_package_owns_compiled_chat_codec(tmp_path: Path) -> None:
    destination = tmp_path / "tokenizer"

    manifest = copy_tokenizer_package(MODEL_DIR, destination)

    assert manifest["chat_codec"] == "chat_codec.json"
    assert "chat_codec.json" in manifest["files"]
    artifact = json.loads((destination / "chat_codec.json").read_text())
    assert artifact["schema"] == "nerve.chat_codec.v1"
    assert artifact["source"]["path"].startswith("encoding/")
    assert len(artifact["validation"]["cases"]) == 4
