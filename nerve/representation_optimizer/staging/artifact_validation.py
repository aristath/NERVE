from __future__ import annotations

import json
import struct
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.staging.contracts import CandidateBuildPlan


ArtifactValidator = Callable[[Path, Json], Json]


@dataclass
class ArtifactValidatorRegistry:
    _validators: dict[str, ArtifactValidator] = field(default_factory=dict)

    @classmethod
    def with_builtin_validators(cls) -> ArtifactValidatorRegistry:
        registry = cls()
        registry.register("json_contract", _validate_json_contract)
        registry.register("nonempty_binary", _validate_nonempty_binary)
        registry.register("spirv_module", _validate_spirv_module)
        return registry

    def register(self, validator_id: str, validator: ArtifactValidator) -> None:
        if not validator_id or not callable(validator):
            raise ModelCompileError("artifact validator registration is invalid")
        if validator_id in self._validators:
            raise ModelCompileError(
                f"artifact validator {validator_id!r} is already registered"
            )
        self._validators[validator_id] = validator

    def validate_artifacts(
        self,
        root: Path,
        build_plan: CandidateBuildPlan,
    ) -> dict[str, Json]:
        results = {}
        for declaration in build_plan.outputs:
            validator_id = declaration["validator_id"]
            validator = self._validators.get(validator_id)
            if validator is None:
                raise ModelCompileError(
                    f"candidate artifact {declaration['path']!r} requires "
                    f"unregistered validator {validator_id!r}"
                )
            path = root / declaration["path"]
            if path.is_symlink() or not path.is_file():
                raise ModelCompileError(
                    f"candidate artifact is not a regular file: "
                    f"{declaration['path']!r}"
                )
            facts = validator(
                path,
                deepcopy(declaration["validation_contract"]),
            )
            if not isinstance(facts, dict):
                raise ModelCompileError(
                    f"artifact validator {validator_id!r} returned non-object facts"
                )
            results[declaration["path"]] = {
                "validator_id": validator_id,
                "status": "passed",
                "facts": deepcopy(facts),
            }
        return results


def _validate_nonempty_binary(path: Path, contract: Json) -> Json:
    _fields(contract, {"minimum_byte_count", "byte_multiple"}, "binary validator")
    minimum = _nonnegative_integer(
        contract["minimum_byte_count"], "minimum_byte_count"
    )
    multiple = _positive_integer(contract["byte_multiple"], "byte_multiple")
    byte_count = path.stat().st_size
    if byte_count < minimum or byte_count % multiple:
        raise ModelCompileError(
            "candidate binary artifact violates its length contract"
        )
    return {"byte_count": byte_count, "byte_multiple": multiple}


def _validate_json_contract(path: Path, contract: Json) -> Json:
    _fields(contract, {"schema", "object_required"}, "JSON validator")
    schema = contract["schema"]
    if schema is not None and (not isinstance(schema, str) or not schema):
        raise ModelCompileError("JSON artifact schema constraint is invalid")
    if not isinstance(contract["object_required"], bool):
        raise ModelCompileError("JSON object_required constraint must be boolean")
    try:
        with path.open("rb") as stream:
            document = json.load(stream)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelCompileError("candidate JSON artifact is malformed") from error
    if contract["object_required"] and not isinstance(document, dict):
        raise ModelCompileError("candidate JSON artifact must be an object")
    if schema is not None and (
        not isinstance(document, dict) or document.get("schema") != schema
    ):
        raise ModelCompileError(
            f"candidate JSON artifact does not use schema {schema!r}"
        )
    return {
        "json_type": (
            "object"
            if isinstance(document, dict)
            else "array"
            if isinstance(document, list)
            else type(document).__name__
        ),
        "schema": document.get("schema") if isinstance(document, dict) else None,
    }


def _validate_spirv_module(path: Path, contract: Json) -> Json:
    _fields(contract, {"minimum_version"}, "SPIR-V validator")
    minimum_version = _positive_integer(
        contract["minimum_version"], "minimum_version"
    )
    byte_count = path.stat().st_size
    if byte_count < 20 or byte_count % 4:
        raise ModelCompileError(
            "candidate SPIR-V artifact must contain an aligned five-word header"
        )
    with path.open("rb") as stream:
        header = stream.read(20)
    magic, version, generator, bound, reserved = struct.unpack("<5I", header)
    if magic != 0x07230203:
        raise ModelCompileError("candidate SPIR-V artifact has invalid magic")
    if version < minimum_version:
        raise ModelCompileError(
            "candidate SPIR-V artifact is older than its declared minimum"
        )
    if bound == 0 or reserved != 0:
        raise ModelCompileError("candidate SPIR-V artifact header is invalid")
    return {
        "version": version,
        "generator": generator,
        "id_bound": bound,
        "word_count": byte_count // 4,
    }


def _fields(record: Json, expected: set[str], label: str) -> None:
    if not isinstance(record, dict) or set(record) != expected:
        raise ModelCompileError(f"{label} fields are invalid")


def _nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelCompileError(f"{label} must be a non-negative integer")
    return value


def _positive_integer(value: object, label: str) -> int:
    result = _nonnegative_integer(value, label)
    if result == 0:
        raise ModelCompileError(f"{label} must be positive")
    return result
