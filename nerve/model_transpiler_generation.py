from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from nerve.model_transpiler_discovery import discover_sampling_policy
from nerve.model_transpiler_types import ModelTranspileError


Json = dict[str, Any]

_SAMPLING_KEYS = (
    "do_sample",
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "presence_penalty",
    "repetition_penalty",
)
_INTEGER_KEYS = {"top_k"}
_BOOLEAN_KEYS = {"do_sample"}
_DOCUMENTED_ARGUMENTS = (
    re.compile(r"--gen_kwargs\s+([\"'])(?P<body>[^\r\n]*?)\1"),
    re.compile(r"generation_parameters\s*=\s*\{(?P<body>[^{}]*)\}"),
)


def load_source_generation_config(model_dir: Path) -> Json:
    """Load explicit generation metadata, then conservatively recover omissions.

    Some derivative checkpoints copy token IDs into ``generation_config.json`` but
    omit the sampling policy used and documented by their producer. A documented
    policy is safe to promote only when every machine-readable example in the
    source model agrees. Conflicting task-specific profiles remain runtime choices.
    """

    generation_path = model_dir / "generation_config.json"
    generation = (
        json.loads(generation_path.read_text(encoding="utf-8"))
        if generation_path.is_file()
        else {}
    )
    if not isinstance(generation, dict):
        raise ValueError(f"{generation_path} must contain a JSON object")
    if any(key in generation for key in _SAMPLING_KEYS):
        return generation

    documented = _discover_unambiguous_documented_policy(model_dir)
    if documented is None:
        return generation
    return {**generation, **documented}


def _discover_unambiguous_documented_policy(model_dir: Path) -> Json | None:
    policies: dict[str, Json] = {}
    for path in sorted(model_dir.glob("*.md")):
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in _DOCUMENTED_ARGUMENTS:
            for match in pattern.finditer(source):
                policy = _parse_documented_policy(match.group("body"))
                if policy is None:
                    continue
                canonical = json.dumps(policy, sort_keys=True, separators=(",", ":"))
                policies[canonical] = policy
                if len(policies) > 1:
                    return None
    return next(iter(policies.values()), None)


def _parse_documented_policy(source: str) -> Json | None:
    values: Json = {}
    for field in source.split(","):
        match = re.fullmatch(
            r"\s*(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*(?P<value>.*?)\s*",
            field,
        )
        if match is None:
            continue
        key = match.group("key")
        if key not in _SAMPLING_KEYS:
            continue
        try:
            values[key] = _parse_scalar(key, match.group("value"))
        except ValueError:
            return None

    if not values:
        return None
    values.setdefault("do_sample", any(key != "do_sample" for key in values))
    try:
        discover_sampling_policy(values)
    except (TypeError, ValueError, ModelTranspileError):
        return None
    return values


def _parse_scalar(key: str, source: str) -> bool | int | float:
    if key in _BOOLEAN_KEYS:
        normalized = source.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
        raise ValueError(f"invalid boolean {source!r}")
    if key in _INTEGER_KEYS:
        return int(source)
    return float(source)
