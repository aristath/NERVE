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


def render_embedded_parameter_shader(
    attrs: Json,
    *,
    branch_values: tuple[tuple[int, ...], tuple[int, ...]],
    temporal: bool,
    retain_parameter_abi: bool = False,
) -> str:
    head_width = int(attrs["branches"][0]["norm"]["head_width"])
    if (
        len(branch_values) != 2
        or any(len(values) != head_width for values in branch_values)
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value > 0xFFFF
            for values in branch_values
            for value in values
        )
    ):
        raise ModelCompileError(
            "embedded parameter program does not cover both BF16 branches"
        )
    path = _source_template_path(temporal=temporal)
    source = path.read_text()
    if not retain_parameter_abi:
        source = _remove_embedded_parameter_buffers(
            source,
            temporal=temporal,
        )
    packed_weights = _packed_u16_words(
        tuple(value for values in branch_values for value in values)
    )
    declarations = _const_u32_array(
        "EMBEDDED_WEIGHT_WORDS",
        packed_weights,
    )
    anchor = "float read_branch_weight(uint branch, uint index) {"
    start = source.find(anchor)
    end = source.find("\nfloat read_normalized(uint dim) {", start)
    if start < 0 or end < 0:
        raise ModelCompileError(
            "source head-normalization shader weight lookup changed"
        )
    lookup = """uint read_branch_weight_pair(uint branch, uint pair_index) {
    uint element_index = branch * HEAD_WIDTH + pair_index;
    return EMBEDDED_WEIGHT_WORDS[element_index >> 1u];
}
"""
    rendered = source[:start] + declarations + "\n\n" + lookup + source[end + 1 :]
    pair_loop_start = (
        "    for (uint pair_dim = lane * 2u; "
        "pair_dim < HEAD_WIDTH; pair_dim += 128u) {\n"
    )
    if rendered.count(pair_loop_start) != 1:
        raise ModelCompileError(
            "source head-normalization shader weight-pair loop changed"
        )
    rendered = rendered.replace(
        pair_loop_start,
        pair_loop_start + "        uint packed_weight = "
        "read_branch_weight_pair(branch, pair_dim);\n",
    )
    weight_expressions = {
        "read_branch_weight(branch, pair_dim)": (
            "bf16_to_f32(packed_weight & 0xffffu)"
        ),
        "read_branch_weight(branch, pair_dim + 1u)": (
            "bf16_to_f32(packed_weight >> 16u)"
        ),
    }
    for original, replacement in weight_expressions.items():
        if rendered.count(original) != 1:
            raise ModelCompileError(
                "source head-normalization shader weight expression changed"
            )
        rendered = rendered.replace(original, replacement)
    if temporal:
        rendered = _hoist_temporal_weight_pairs(
            rendered,
            head_width=head_width,
        )
    replacements = _shader_replacements(attrs, temporal=temporal)
    for name, value in replacements.items():
        rendered = rendered.replace(f"{{{{{name}}}}}", value)
    unresolved = sorted(set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", rendered)))
    if unresolved:
        raise ModelCompileError(
            f"embedded parameter shader has unresolved template values: {unresolved}"
        )
    return rendered


def _hoist_temporal_weight_pairs(
    source: str,
    *,
    head_width: int,
) -> str:
    if head_width <= 0 or head_width % 2:
        raise ModelCompileError(
            "embedded temporal parameter program requires an even head width"
        )
    position_loop = (
        "    for (uint position = 0u; "
        "position < batch_control.batch_width; position++) {\n"
    )
    if source.count(position_loop) != 1:
        raise ModelCompileError(
            "source temporal head-normalization position loop changed"
        )
    offsets = tuple(range(0, head_width, 128))
    declarations = ""
    for index, offset in enumerate(offsets):
        declarations += (
            "    uint embedded_weight_pair_"
            f"{index} = read_branch_weight_pair("
            f"branch, lane * 2u + {offset}u);\n"
            f"    float embedded_weight_lo_{index} = "
            f"bf16_to_f32(embedded_weight_pair_{index} & 0xffffu);\n"
            f"    float embedded_weight_hi_{index} = "
            f"bf16_to_f32(embedded_weight_pair_{index} >> 16u);\n"
        )
    source = source.replace(position_loop, declarations + "\n" + position_loop)

    pair_loop = (
        "        for (uint pair_dim = lane * 2u; "
        "pair_dim < HEAD_WIDTH; pair_dim += 128u) {\n"
    )
    pair_start = source.find(pair_loop)
    close_marker = "        }\n        barrier();"
    pair_end = source.find(close_marker, pair_start)
    if pair_start < 0 or pair_end < 0 or source.find(pair_loop, pair_start + 1) >= 0:
        raise ModelCompileError(
            "source temporal head-normalization weight-pair loop changed"
        )
    body_start = pair_start + len(pair_loop)
    body = source[body_start:pair_end]
    dynamic_lookup = (
        "        uint packed_weight = read_branch_weight_pair(branch, pair_dim);\n"
    )
    if body.count(dynamic_lookup) != 1:
        raise ModelCompileError(
            "source temporal head-normalization pair lookup changed"
        )
    body = body.replace(dynamic_lookup, "")
    packed_weight_expressions = (
        "bf16_to_f32(packed_weight & 0xffffu)",
        "bf16_to_f32(packed_weight >> 16u)",
    )
    if any(body.count(expression) != 1 for expression in packed_weight_expressions):
        raise ModelCompileError(
            "source temporal head-normalization packed-weight use changed"
        )
    unrolled = ""
    for index, offset in enumerate(offsets):
        specialized_body = body.replace(
            "bf16_to_f32(packed_weight & 0xffffu)",
            f"embedded_weight_lo_{index}",
        ).replace(
            "bf16_to_f32(packed_weight >> 16u)",
            f"embedded_weight_hi_{index}",
        )
        unrolled += (
            "        {\n"
            f"            uint pair_dim = lane * 2u + {offset}u;\n"
            "            if (pair_dim < HEAD_WIDTH) {\n"
            f"{specialized_body}"
            "            }\n"
            "        }\n"
        )
    return source[:pair_start] + unrolled + source[pair_end + len("        }\n") :]


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


def _shader_replacements(attrs: Json, *, temporal: bool) -> dict[str, str]:
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
            "embedded parameter shader branches do not share fused numerical geometry"
        )
    rope = ropes[0]
    scaling = rope.get("scaling")
    yarn = rope.get("rope_type", "default") == "yarn"
    if yarn and (not isinstance(scaling, dict) or scaling.get("type") != "yarn"):
        raise ModelCompileError("embedded parameter YaRN shader has no scaling profile")
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
        replacements["BATCH_CONTROL_BINDING"] = "4"
    return replacements


def _remove_embedded_parameter_buffers(
    source: str,
    *,
    temporal: bool,
) -> str:
    if temporal:
        declarations = (
            "layout(set = 0, binding = 4) readonly buffer WeightA "
            "{ uint words[]; } weight_a;\n",
            "layout(set = 0, binding = 5) readonly buffer WeightB "
            "{ uint words[]; } weight_b;\n",
        )
    else:
        declarations = (
            "layout(set = 0, binding = 4) readonly buffer WeightA {\n"
            "    uint words[];\n"
            "} weight_a;\n\n",
            "layout(set = 0, binding = 5) readonly buffer WeightB {\n"
            "    uint words[];\n"
            "} weight_b;\n\n",
        )
    for declaration in declarations:
        if source.count(declaration) != 1:
            raise ModelCompileError(
                "source head-normalization shader parameter ABI changed"
            )
        source = source.replace(declaration, "")
    if not temporal:
        old_control = "layout(set = 0, binding = 6) readonly buffer StreamControl"
        new_control = "layout(set = 0, binding = 4) readonly buffer StreamControl"
        if source.count(old_control) != 1:
            raise ModelCompileError(
                "source head-normalization shader control ABI changed"
            )
        source = source.replace(old_control, new_control)
    return source


def _packed_u16_words(values: tuple[int, ...]) -> tuple[int, ...]:
    padded = (*values, *((0,) if len(values) % 2 else ()))
    return tuple(
        padded[index] | (padded[index + 1] << 16) for index in range(0, len(padded), 2)
    )


def _const_u32_array(name: str, values: tuple[int, ...]) -> str:
    if not values:
        raise ModelCompileError(f"embedded parameter array {name} is empty")
    literals = ",\n    ".join(f"0x{value:08x}u" for value in values)
    return (
        f"const uint {name}[{len(values)}] = uint[{len(values)}](\n    {literals}\n);"
    )


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
