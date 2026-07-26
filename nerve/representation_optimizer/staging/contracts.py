from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.contracts import (
    ContractValidationError,
    canonical_json_bytes,
)


CANDIDATE_BUILD_PLAN_SCHEMA = "nerve.optimizer.candidate_build_plan.v1"
STAGED_ARTIFACT_DIGEST_SCHEMA = "nerve.optimizer.artifact_sha256.v1"
SOURCE_PACKAGE_SEAL_SCHEMA = "nerve.optimizer.source_package_seal.v1"
CONSTRUCTION_PHASES = (
    "semantic_construction",
    "ordinary_lowering",
    "physical_optimization",
)
_ARTIFACT_LIFETIMES = frozenset({"compile", "mount", "residency", "dynamic"})


@dataclass(frozen=True)
class CandidateBuildPlan:
    _document: Json

    @classmethod
    def from_json(cls, document: Json) -> CandidateBuildPlan:
        normalized = deepcopy(document)
        validate_candidate_build_plan(normalized)
        return cls(normalized)

    @property
    def source_inputs(self) -> tuple[Json, ...]:
        return tuple(deepcopy(self._document["source_inputs"]))

    @property
    def outputs(self) -> tuple[Json, ...]:
        return tuple(deepcopy(self._document["outputs"]))

    @property
    def output_paths(self) -> tuple[str, ...]:
        return tuple(output["path"] for output in self._document["outputs"])

    def outputs_for_phase(self, phase: str) -> tuple[Json, ...]:
        return tuple(
            deepcopy(output)
            for output in self._document["outputs"]
            if output["producer_phase"] == phase
        )

    def to_json(self) -> Json:
        return deepcopy(self._document)


def staged_artifact_digest(payload: bytes) -> str:
    return f"{STAGED_ARTIFACT_DIGEST_SCHEMA}:{sha256(payload).hexdigest()}"


def staged_file_digest(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    if (
        isinstance(chunk_bytes, bool)
        or not isinstance(chunk_bytes, int)
        or chunk_bytes <= 0
    ):
        raise ModelCompileError("artifact digest chunk size must be positive")
    digest = sha256()
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            while chunk := stream.read(chunk_bytes):
                digest.update(chunk)
    except OSError as error:
        raise ModelCompileError(
            f"artifact cannot be hashed as a regular file: {path}"
        ) from error
    return f"{STAGED_ARTIFACT_DIGEST_SCHEMA}:{digest.hexdigest()}"


def validate_candidate_build_plan(document: Json) -> None:
    canonical_json_bytes(document)
    _fields(
        document,
        {"schema", "phases", "source_inputs", "outputs", "resource_limits"},
        "candidate build plan",
    )
    if document["schema"] != CANDIDATE_BUILD_PLAN_SCHEMA:
        raise ContractValidationError(
            f"unsupported candidate build plan schema {document['schema']!r}"
        )
    if document["phases"] != list(CONSTRUCTION_PHASES):
        raise ContractValidationError(
            "candidate build plan must run semantic construction, ordinary "
            "lowering, and physical optimization in that order"
        )
    source_paths: list[str] = []
    for index, raw in enumerate(_list(document["source_inputs"], "source_inputs")):
        record = _object(raw, f"source_inputs[{index}]")
        _fields(record, {"path", "digest"}, f"source_inputs[{index}]")
        path = _safe_relative_path(record["path"], f"source_inputs[{index}].path")
        _artifact_digest(record["digest"], f"source_inputs[{index}].digest")
        source_paths.append(path)
    if source_paths != sorted(set(source_paths)):
        raise ContractValidationError(
            "candidate build plan source_inputs must be sorted and unique"
        )

    output_paths: list[str] = []
    for index, raw in enumerate(_list(document["outputs"], "outputs")):
        record = _object(raw, f"outputs[{index}]")
        _fields(
            record,
            {
                "path",
                "kind",
                "lifetime",
                "producer_phase",
                "resident_bytes",
                "validator_id",
                "validation_contract",
            },
            f"outputs[{index}]",
        )
        path = _safe_relative_path(record["path"], f"outputs[{index}].path")
        if path == "integrity.json" or path.startswith("contracts/"):
            raise ContractValidationError(
                f"outputs[{index}].path uses a staging-engine reserved path"
            )
        _text(record["kind"], f"outputs[{index}].kind")
        if record["lifetime"] not in _ARTIFACT_LIFETIMES:
            raise ContractValidationError(
                f"outputs[{index}].lifetime is unsupported"
            )
        if record["producer_phase"] not in CONSTRUCTION_PHASES:
            raise ContractValidationError(
                f"outputs[{index}].producer_phase is unsupported"
            )
        _nonnegative_integer(
            record["resident_bytes"], f"outputs[{index}].resident_bytes"
        )
        _text(record["validator_id"], f"outputs[{index}].validator_id")
        _object(
            record["validation_contract"],
            f"outputs[{index}].validation_contract",
        )
        output_paths.append(path)
    if not output_paths:
        raise ContractValidationError("candidate build plan outputs must not be empty")
    if output_paths != sorted(set(output_paths)):
        raise ContractValidationError(
            "candidate build plan outputs must be sorted and unique"
        )

    limits = _object(document["resource_limits"], "resource_limits")
    _fields(
        limits,
        {
            "maximum_construction_time_ns",
            "maximum_temporary_bytes",
            "maximum_staging_bytes",
        },
        "resource_limits",
    )
    for field, value in limits.items():
        if value is not None:
            _positive_integer(value, f"resource_limits.{field}")


def _safe_relative_path(value: Any, path: str) -> str:
    text = _text(value, path)
    relative = Path(text)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "." in relative.parts
        or relative.as_posix() != text
    ):
        raise ContractValidationError(f"{path} must be a normalized relative path")
    return text


def _artifact_digest(value: Any, path: str) -> str:
    text = _text(value, path)
    prefix = f"{STAGED_ARTIFACT_DIGEST_SCHEMA}:"
    hexadecimal = text.removeprefix(prefix)
    if (
        not text.startswith(prefix)
        or len(hexadecimal) != 64
        or any(character not in "0123456789abcdef" for character in hexadecimal)
    ):
        raise ContractValidationError(f"{path} is not a staged artifact digest")
    return text


def _fields(record: Json, expected: set[str], path: str) -> None:
    actual = set(record)
    if actual != expected:
        raise ContractValidationError(
            f"{path} fields are invalid: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _object(value: Any, path: str) -> Json:
    if not isinstance(value, dict):
        raise ContractValidationError(f"{path} must be an object")
    return value


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ContractValidationError(f"{path} must be a list")
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractValidationError(f"{path} must be a non-empty string")
    return value


def _nonnegative_integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractValidationError(f"{path} must be a non-negative integer")
    return value


def _positive_integer(value: Any, path: str) -> int:
    result = _nonnegative_integer(value, path)
    if result == 0:
        raise ContractValidationError(f"{path} must be positive")
    return result
