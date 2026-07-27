from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from hashlib import sha256
from pathlib import Path

from nerve.compilation import Json, ModelCompileError


_SHADER_ROOT = Path(__file__).resolve().parents[4] / "runtime-rs" / "shaders"


def source_template_sha256(*, temporal: bool) -> str:
    path = _source_template_path(temporal=temporal)
    return sha256(path.read_bytes()).hexdigest()


def render_codebook_shader(attrs: Json, *, temporal: bool) -> str:
    path = _source_template_path(temporal=temporal)
    rendered = _replace_weight_lookup(path.read_text(), temporal=temporal)
    branches = attrs["branches"]
    norms = [branch["norm"] for branch in branches]
    ropes = [branch["rope"] for branch in branches]
    if (
        len(branches) != 2
        or any(
            norms[0][field] != norms[1][field]
            for field in ("head_width", "eps", "weight_offset")
        )
        or any(
            ropes[0].get(field) != ropes[1].get(field)
            for field in (
                "head_width",
                "rotary_width",
                "theta",
                "rope_type",
                "interleaved",
                "scaling",
            )
        )
    ):
        raise ModelCompileError(
            "codebook shader branches do not share fused numerical geometry"
        )
    rope = ropes[0]
    scaling = rope.get("scaling")
    yarn = rope.get("rope_type", "default") == "yarn"
    if yarn and (not isinstance(scaling, dict) or scaling.get("type") != "yarn"):
        raise ModelCompileError("codebook YaRN shader has no scaling profile")
    replacements = {
        "BRANCH_A_HEADS": str(int(norms[0]["head_count"])),
        "BRANCH_B_HEADS": str(int(norms[1]["head_count"])),
        "HEAD_WIDTH": str(int(norms[0]["head_width"])),
        "ROTARY_WIDTH": str(int(rope["rotary_width"])),
        "NORM_EPS": _shader_float(float(norms[0]["eps"])),
        "WEIGHT_OFFSET": _shader_float(float(norms[0]["weight_offset"])),
        "ROPE_THETA": _shader_float(float(rope["theta"])),
        "ROPE_INTERLEAVED": _shader_bool(bool(rope.get("interleaved"))),
        "ROPE_PROPORTIONAL": _shader_bool(rope.get("rope_type") == "proportional"),
        "ROPE_YARN": _shader_bool(yarn),
        "ROPE_FACTOR": _shader_float(float(scaling["factor"]) if yarn else 1.0),
        "ROPE_CORRECTION_LOW": _shader_float(
            float(scaling["correction_low"]) if yarn else 0.0
        ),
        "ROPE_CORRECTION_HIGH": _shader_float(
            float(scaling["correction_high"]) if yarn else 1.0
        ),
        "ROPE_ATTENTION_FACTOR": _shader_float(
            float(scaling["attention_factor"]) if yarn else 1.0
        ),
    }
    if temporal:
        replacements["BATCH_CONTROL_BINDING"] = "7"
    for name, value in replacements.items():
        rendered = rendered.replace(f"{{{{{name}}}}}", value)
    unresolved = sorted(set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", rendered)))
    if unresolved:
        raise ModelCompileError(
            f"codebook shader has unresolved template values: {unresolved}"
        )
    return rendered


def compile_spirv(source: str, filename: str) -> bytes:
    compiler = shutil.which("glslangValidator")
    if compiler is None:
        raise ModelCompileError("codebook Vulkan lowering requires glslangValidator")
    with tempfile.TemporaryDirectory(prefix="nerve-codebook-shader-") as root:
        source_path = Path(root) / filename
        output_path = source_path.with_suffix(".spv")
        source_path.write_text(source)
        completed = subprocess.run(
            [
                compiler,
                "-V",
                "--target-env",
                "vulkan1.4",
                str(source_path),
                "-o",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            diagnostic = (completed.stderr or completed.stdout).strip()
            raise ModelCompileError(f"codebook shader compilation failed: {diagnostic}")
        payload = output_path.read_bytes()
    if len(payload) < 20 or payload[:4] != b"\x03\x02#\x07":
        raise ModelCompileError("codebook shader compiler produced invalid SPIR-V")
    return payload


def _source_template_path(*, temporal: bool) -> Path:
    template_name = (
        "parallel_head_norm_rope_2way_temporal_bf16.comp.template"
        if temporal
        else "parallel_head_norm_rope_2way_bf16.comp.template"
    )
    path = _SHADER_ROOT / template_name
    if not path.is_file():
        raise ModelCompileError(f"missing source shader template {path}")
    return path


def _replace_weight_lookup(source: str, *, temporal: bool) -> str:
    control = (
        "layout(set = 0, binding = {{BATCH_CONTROL_BINDING}}) "
        "readonly buffer BatchControl"
        if temporal
        else "layout(set = 0, binding = 6) readonly buffer StreamControl"
    )
    replacement_control = (
        "layout(set = 0, binding = 6) readonly buffer Codebook {\n"
        "    uint words[];\n"
        "} codebook;\n\n"
        + (
            control
            if temporal
            else "layout(set = 0, binding = 7) readonly buffer StreamControl"
        )
    )
    if source.count(control) != 1:
        raise ModelCompileError(
            "source head-normalization shader control binding changed"
        )
    source = source.replace(control, replacement_control)
    start = source.find("float read_branch_weight(uint branch, uint index) {")
    end = source.find("\nfloat read_normalized(uint dim) {", start)
    if start < 0 or end < 0:
        raise ModelCompileError(
            "source head-normalization shader weight lookup changed"
        )
    lookup = """uint read_branch_address(uint branch, uint index) {
    uint packed = branch == 0u
        ? weight_a.words[index >> 2]
        : weight_b.words[index >> 2];
    return (packed >> ((index & 3u) * 8u)) & 0xffu;
}

float read_branch_weight(uint branch, uint index) {
    uint address = read_branch_address(branch, index);
    uint packed = codebook.words[address >> 1];
    uint value = ((address & 1u) == 0u)
        ? (packed & 0xffffu)
        : (packed >> 16);
    return bf16_to_f32(value);
}
"""
    return source[:start] + lookup + source[end + 1 :]


def _shader_float(value: float) -> str:
    text = format(value, ".9g")
    return text if any(character in text for character in ".eE") else f"{text}.0"


def _shader_bool(value: bool) -> str:
    return "true" if value else "false"
