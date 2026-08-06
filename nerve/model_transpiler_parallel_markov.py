from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from nerve.model_transpiler_hyper_connections import discover_stream_mixer
from nerve.model_transpiler_quantization import (
    attach_block_quantization_scales,
    attach_packed_linear_quantization,
)
from nerve.model_transpiler_types import (
    DraftExecutionGraphStructure,
    Json,
    LayerStructure,
    ModelTranspileError,
)


PARALLEL_MARKOV_DRAFT_TYPE = "parallel_backbone_markov"
_NUMBERED_COMPONENT = re.compile(r"^(?P<root>.+)\.(?P<index>\d+)\.(?P<suffix>.+)$")


def discover_parallel_markov_drafts(
    *,
    tensors: dict[str, Json],
    model_dir: Path,
    decoder_config: Json,
    primary_layer_root: str,
    main_layer_count: int,
    hidden_size: int,
    vocabulary_size: int,
    output_projection: str,
    discover_layer: Callable[[str, int, Json], LayerStructure],
) -> tuple[DraftExecutionGraphStructure, ...]:
    roots: dict[str, set[int]] = {}
    for name in tensors:
        match = _NUMBERED_COMPONENT.match(name)
        if match is not None:
            roots.setdefault(match.group("root"), set()).add(int(match.group("index")))

    discovered = []
    for root in sorted(roots):
        if root == primary_layer_root:
            continue
        indices = sorted(roots[root])
        if indices != list(range(len(indices))) or not indices:
            continue
        first = f"{root}.{indices[0]}"
        last = f"{root}.{indices[-1]}"
        marker_names = (
            f"{first}.main_proj.weight",
            f"{last}.markov_head.markov_w1.weight",
            f"{last}.markov_head.markov_w2.weight",
            f"{last}.confidence_head.proj.weight",
        )
        if not any(name in tensors for name in marker_names):
            continue
        required = {
            "target_projection": f"{first}.main_proj.weight",
            "target_norm": f"{first}.main_norm.weight",
            "head_function": f"{last}.hc_head_fn",
            "head_base": f"{last}.hc_head_base",
            "head_scale": f"{last}.hc_head_scale",
            "norm": f"{last}.norm.weight",
            "projection": output_projection,
            "markov_embedding": f"{last}.markov_head.markov_w1.weight",
            "markov_projection": f"{last}.markov_head.markov_w2.weight",
            "confidence_projection": f"{last}.confidence_head.proj.weight",
        }
        missing = sorted(name for name in required.values() if name not in tensors)
        if missing:
            raise ModelTranspileError(
                f"parallel Markov draft stack {root!r} is incomplete: {missing}"
            )

        target_layers = _unique_config_int_list(
            decoder_config,
            suffix="target_layer_ids",
            role="parallel Markov target layers",
        )
        if not target_layers or any(
            index < 0 or index >= main_layer_count for index in target_layers
        ):
            raise ModelTranspileError(
                "parallel Markov target layers must identify existing main layers"
            )
        block_size = _unique_config_int(
            decoder_config,
            suffix="block_size",
            role="parallel Markov block size",
        )
        noise_token_id = _unique_config_int(
            decoder_config,
            suffix="noise_token_id",
            role="parallel Markov noise token",
        )
        if block_size <= 0:
            raise ModelTranspileError("parallel Markov block size must be positive")
        default_draft_tokens = discover_recommended_draft_token_count(
            model_dir, minimum=block_size
        )
        execution_block_size = max(block_size, default_draft_tokens)
        if noise_token_id < 0 or noise_token_id >= vocabulary_size:
            raise ModelTranspileError(
                "parallel Markov noise token must belong to the model vocabulary"
            )

        _require_shape(
            tensors,
            required["target_projection"],
            [hidden_size, hidden_size * len(target_layers)],
            "target projection",
        )
        _require_shape(tensors, required["target_norm"], [hidden_size], "target norm")
        _require_shape(tensors, required["norm"], [hidden_size], "output norm")
        markov_shape = _matrix_shape(tensors, required["markov_embedding"])
        if (
            len(markov_shape) != 2
            or markov_shape[0] != vocabulary_size
            or markov_shape[1] <= 0
        ):
            raise ModelTranspileError(
                "parallel Markov embedding must be a vocabulary-by-rank matrix"
            )
        _require_shape(
            tensors,
            required["markov_projection"],
            markov_shape,
            "Markov projection",
        )
        rank = markov_shape[1]
        _require_shape(
            tensors,
            required["confidence_projection"],
            [1, hidden_size + rank],
            "confidence projection",
        )
        draft_stream_mixer = discover_stream_mixer(
            tensors,
            decoder_config,
            hidden_size=hidden_size,
            prefix=last,
        )
        if draft_stream_mixer is None:
            raise ModelTranspileError(
                "parallel Markov draft has no output stream reduction contract"
            )
        layers = tuple(
            discover_layer(root, index, draft_stream_mixer) for index in indices
        )
        attach_block_quantization_scales(tensors, required)
        attach_packed_linear_quantization(tensors, required)
        discovered.append(
            DraftExecutionGraphStructure(
                id=f"draft_{len(discovered):02d}",
                prefix=root,
                tensors=required,
                layers=layers,
                draft_type=PARALLEL_MARKOV_DRAFT_TYPE,
                attributes={
                    "target_features": {
                        "layer_indices": target_layers,
                        "lane_reduction": "mean",
                        "concatenation_order": "declared_layer_order",
                    },
                    "proposal_contract": {
                        "schedule": "parallel_backbone_then_sequential_markov",
                        "configured_block_size": block_size,
                        "execution_block_size": execution_block_size,
                        "minimum_draft_tokens": 1,
                        "default_draft_tokens": default_draft_tokens,
                        "noise_token_id": noise_token_id,
                        "sampling": "greedy",
                        "confidence_prefix": ("first_sigmoid_below_runtime_threshold"),
                        "verification": ("lossless_target_longest_matching_prefix"),
                    },
                    "markov_rank": rank,
                    "stream_mixer": draft_stream_mixer,
                },
            )
        )
    return tuple(discovered)


def discover_recommended_draft_token_count(model_dir: Path, *, minimum: int) -> int:
    model_card = model_dir / "README.md"
    if not model_card.is_file():
        return minimum
    try:
        source = model_card.read_text(errors="replace")
    except OSError:
        return minimum
    recommendations = {
        int(match.group("count"))
        for match in re.finditer(
            r'["\']num_speculative_tokens["\']\s*:\s*(?P<count>\d+)',
            source,
        )
    }
    eligible = sorted(count for count in recommendations if count >= minimum)
    return eligible[0] if eligible else minimum


def _config_values_with_suffix(value: Json, suffix: str) -> list[object]:
    found = []
    for key, nested in value.items():
        if key.endswith(suffix):
            found.append(nested)
        if isinstance(nested, dict):
            found.extend(_config_values_with_suffix(nested, suffix))
    return found


def _unique_config_int(config: Json, *, suffix: str, role: str) -> int:
    values = _config_values_with_suffix(config, suffix)
    parsed = {
        int(value)
        for value in values
        if isinstance(value, int) and not isinstance(value, bool)
    }
    if len(parsed) != 1:
        raise ModelTranspileError(f"{role} is missing or ambiguous")
    return parsed.pop()


def _unique_config_int_list(config: Json, *, suffix: str, role: str) -> list[int]:
    values = _config_values_with_suffix(config, suffix)
    parsed = {
        tuple(int(item) for item in value)
        for value in values
        if isinstance(value, list)
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    }
    if len(parsed) != 1:
        raise ModelTranspileError(f"{role} are missing or ambiguous")
    return list(parsed.pop())


def _matrix_shape(tensors: dict[str, Json], name: str) -> list[int]:
    return [int(value) for value in tensors[name].get("shape", [])]


def _require_shape(
    tensors: dict[str, Json], name: str, expected: list[int], role: str
) -> None:
    actual = _matrix_shape(tensors, name)
    if actual != expected:
        raise ModelTranspileError(
            f"parallel Markov {role} tensor {name!r} has shape {actual}, "
            f"expected {expected}"
        )
