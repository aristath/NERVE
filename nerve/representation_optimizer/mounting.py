from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.contracts import (
    ContractValidationError,
    canonical_json_bytes,
)
from nerve.representation_optimizer.staging.contracts import CandidateBuildPlan
from nerve.resident_representations import validate_resident_derivation


RUNTIME_MOUNT_PLAN_SCHEMA = "nerve.optimizer.runtime_mount_plan.v3"
VULKAN_STREAM_CIRCUIT_OVERLAY_ADAPTER = (
    "vulkan_stream_circuit_overlay.v2"
)
VULKAN_COMPONENT_OVERLAY_SCHEMA = (
    "nerve.optimizer.vulkan_component_overlay.v2"
)
VULKAN_COMPONENT_REGION_OVERLAY_SCHEMA = (
    "nerve.optimizer.vulkan_component_region_overlay.v2"
)
VULKAN_OUTPUT_TRANSDUCER_OVERLAY_SCHEMA = (
    "nerve.optimizer.vulkan_output_transducer_overlay.v1"
)


@dataclass(frozen=True)
class RuntimeMountPlan:
    _document: Json

    @classmethod
    def from_json(
        cls,
        document: Json,
        *,
        candidate_id: str,
        build_plan: CandidateBuildPlan,
    ) -> RuntimeMountPlan:
        normalized = deepcopy(document)
        validate_runtime_mount_plan(
            normalized,
            candidate_id=candidate_id,
            build_plan=build_plan,
        )
        return cls(normalized)

    def to_json(self) -> Json:
        return deepcopy(self._document)


def validate_runtime_mount_plan(
    document: Json,
    *,
    candidate_id: str,
    build_plan: CandidateBuildPlan,
) -> None:
    canonical_json_bytes(document)
    _fields(
        document,
        {
            "schema",
            "candidate_id",
            "adapter_id",
            "regions",
            "tensor_index_refs",
        },
        "runtime mount plan",
    )
    if document["schema"] != RUNTIME_MOUNT_PLAN_SCHEMA:
        raise ContractValidationError(
            f"unsupported runtime mount plan schema {document['schema']!r}"
        )
    if document["candidate_id"] != candidate_id:
        raise ContractValidationError(
            "runtime mount plan belongs to a different candidate"
        )
    adapter_id = _text(
        document["adapter_id"],
        "runtime mount plan adapter_id",
    )
    if adapter_id != VULKAN_STREAM_CIRCUIT_OVERLAY_ADAPTER:
        raise ContractValidationError(
            f"unsupported runtime mount adapter {adapter_id!r}"
        )

    declared_outputs = {
        output["path"]: output
        for output in build_plan.outputs
    }
    regions = _list(document["regions"], "runtime mount plan regions")
    if not regions:
        raise ContractValidationError(
            "runtime mount plan must declare at least one semantic region"
        )
    source_component_ids: list[str] = []
    referenced_outputs: list[str] = []
    source_regions: list[list[str]] = []
    for region_index, raw_region in enumerate(regions):
        region_label = f"runtime mount plan regions[{region_index}]"
        region = _object(raw_region, region_label)
        _fields(region, {"replacements"}, region_label)
        replacements = _list(
            region["replacements"],
            f"{region_label}.replacements",
        )
        region_sources: list[str] = []
        for replacement_index, raw_replacement in enumerate(replacements):
            label = (
                f"{region_label}.replacements"
                f"[{replacement_index}]"
            )
            replacement = _object(raw_replacement, label)
            _fields(
                replacement,
                {
                    "kind",
                    "source_component_id",
                    "overlay_ref",
                },
                label,
            )
            replacement_kind = _text(
                replacement["kind"],
                f"{label}.kind",
            )
            if replacement_kind not in {
                "component",
                "component_region",
                "output_transducer",
            }:
                raise ContractValidationError(
                    f"{label}.kind is unsupported: {replacement_kind!r}"
                )
            source_component_id = _text(
                replacement["source_component_id"],
                f"{label}.source_component_id",
            )
            region_sources.append(source_component_id)
            source_component_ids.append(source_component_id)
            referenced_outputs.append(
                _declared_mount_artifact(
                    replacement["overlay_ref"],
                    f"{label}.overlay_ref",
                    declared_outputs,
                    expected_kind="runtime_overlay",
                )
            )
        if (
            not region_sources
            or region_sources != sorted(set(region_sources))
        ):
            raise ContractValidationError(
                f"{region_label} replacements must be non-empty, sorted, "
                "and unique by source component"
            )
        source_regions.append(region_sources)
    if source_regions != sorted(source_regions):
        raise ContractValidationError(
            "runtime mount-plan regions must be sorted by source components"
        )
    if len(source_component_ids) != len(set(source_component_ids)):
        raise ContractValidationError(
            "runtime mount-plan source components must occur in exactly one "
            "semantic region"
        )

    tensor_index_refs = [
        _declared_mount_artifact(
            value,
            f"runtime mount plan tensor_index_refs[{index}]",
            declared_outputs,
            expected_kind="tensor_index_fragment",
        )
        for index, value in enumerate(
            _list(
                document["tensor_index_refs"],
                "runtime mount plan tensor_index_refs",
            )
        )
    ]
    if tensor_index_refs != sorted(set(tensor_index_refs)):
        raise ContractValidationError(
            "runtime mount-plan tensor index references must be sorted and unique"
        )
    referenced_outputs.extend(tensor_index_refs)
    if len(referenced_outputs) != len(set(referenced_outputs)):
        raise ContractValidationError(
            "runtime mount-plan artifact references must be unique"
        )


def validate_runtime_mount_artifacts(
    root: Path,
    mount_plan: RuntimeMountPlan,
) -> None:
    document = mount_plan.to_json()
    for region in document["regions"]:
        for replacement in region["replacements"]:
            overlay = _read_object(root / replacement["overlay_ref"])
            kind = replacement["kind"]
            if kind == "component":
                label = "Vulkan component overlay"
                expected_fields = {
                    "schema",
                    "source_component_id",
                    "component",
                    "execution",
                    "resident_derivations",
                }
                expected_schema = VULKAN_COMPONENT_OVERLAY_SCHEMA
            elif kind == "component_region":
                label = "Vulkan component-region overlay"
                expected_fields = {
                    "schema",
                    "source_component_id",
                    "source",
                    "replacement",
                }
                expected_schema = VULKAN_COMPONENT_REGION_OVERLAY_SCHEMA
            else:
                label = "Vulkan output-transducer overlay"
                expected_fields = {
                    "schema",
                    "source_component_id",
                    "component",
                    "output_transducer",
                    "speculative_output_transducers",
                }
                expected_schema = VULKAN_OUTPUT_TRANSDUCER_OVERLAY_SCHEMA
            _fields(overlay, expected_fields, label)
            if overlay["schema"] != expected_schema:
                raise ContractValidationError(
                    f"{label} schema is unsupported"
                )
            if (
                overlay["source_component_id"]
                != replacement["source_component_id"]
            ):
                raise ContractValidationError(
                    f"{label} source component disagrees "
                    "with its mount plan"
                )
            if kind == "component":
                _object(
                    overlay["component"],
                    f"{label} component",
                )
                _object(
                    overlay["execution"],
                    "Vulkan component overlay execution",
                )
                derivations = _list(
                    overlay["resident_derivations"],
                    "Vulkan component overlay resident derivations",
                )
                derivation_keys = []
                for index, derivation in enumerate(derivations):
                    derivation = _object(
                        derivation,
                        f"Vulkan component overlay resident derivations[{index}]",
                    )
                    _fields(
                        derivation,
                        {"node_id", "parameter_id", "derivation"},
                        f"Vulkan component overlay resident derivations[{index}]",
                    )
                    derivation_keys.append(
                        (
                            _text(derivation["node_id"], "resident derivation node_id"),
                            _text(
                                derivation["parameter_id"],
                                "resident derivation parameter_id",
                            ),
                        )
                    )
                    contract = _object(
                        derivation["derivation"],
                        "resident derivation contract",
                    )
                    try:
                        validate_resident_derivation(
                            contract,
                            source_byte_count=contract.get("source_byte_count"),
                            label="resident derivation contract",
                        )
                    except ModelCompileError as error:
                        raise ContractValidationError(str(error)) from error
                if derivation_keys != sorted(set(derivation_keys)):
                    raise ContractValidationError(
                        "Vulkan component overlay resident derivations must be sorted and unique"
                    )
            elif kind == "component_region":
                _validate_component_region_overlay(overlay, label)
            else:
                _object(
                    overlay["component"],
                    f"{label} component",
                )
                _object(
                    overlay["output_transducer"],
                    "Vulkan output-transducer package",
                )
                drafts = _list(
                    overlay["speculative_output_transducers"],
                    "Vulkan speculative output transducers",
                )
                decoder_ids = []
                for index, draft in enumerate(drafts):
                    draft = _object(
                        draft,
                        f"Vulkan speculative output transducers[{index}]",
                    )
                    _fields(
                        draft,
                        {"decoder_id", "output_transducer"},
                        f"Vulkan speculative output transducers[{index}]",
                    )
                    decoder_ids.append(
                        _text(
                            draft["decoder_id"],
                            f"Vulkan speculative output transducers[{index}].decoder_id",
                        )
                    )
                    _object(
                        draft["output_transducer"],
                        f"Vulkan speculative output transducers[{index}].output_transducer",
                    )
                if decoder_ids != sorted(set(decoder_ids)):
                    raise ContractValidationError(
                        "Vulkan speculative output transducers must be sorted "
                        "and unique by decoder_id"
                    )
    for reference in document["tensor_index_refs"]:
        fragment = _read_object(root / reference)
        if fragment.get("schema") != "nerve.tensor_index.v1":
            raise ContractValidationError(
                "runtime tensor-index fragment schema is unsupported"
            )
        _object(
            fragment.get("tensors"),
            "runtime tensor-index fragment tensors",
        )


def _validate_component_region_overlay(overlay: Json, label: str) -> None:
    source = _object(overlay["source"], f"{label} source")
    replacement = _object(
        overlay["replacement"],
        f"{label} replacement",
    )
    _fields(
        source,
        {"nodes", "kernels", "parameter_refs"},
        f"{label} source",
    )
    _fields(
        replacement,
        {"nodes", "kernels", "parameter_refs"},
        f"{label} replacement",
    )
    source_node_ids = _region_record_ids(
        source["nodes"],
        f"{label} source nodes",
    )
    source_kernel_ids = _region_record_ids(
        source["kernels"],
        f"{label} source kernels",
    )
    replacement_node_ids = _region_record_ids(
        replacement["nodes"],
        f"{label} replacement nodes",
    )
    replacement_kernel_ids = _region_record_ids(
        replacement["kernels"],
        f"{label} replacement kernels",
    )
    if set(source_node_ids) != set(source_kernel_ids):
        raise ContractValidationError(
            f"{label} source nodes and kernels must cover the same region"
        )
    if set(replacement_node_ids) != set(replacement_kernel_ids):
        raise ContractValidationError(
            f"{label} replacement nodes and kernels must cover the same region"
        )
    _validate_region_parameter_refs(source, f"{label} source")
    _validate_region_parameter_refs(replacement, f"{label} replacement")


def _validate_region_parameter_refs(region: Json, label: str) -> None:
    refs = _object(region["parameter_refs"], f"{label} parameter_refs")
    used = {
        _text(parameter_id, f"{label} node parameter")
        for index, raw_node in enumerate(_list(region["nodes"], f"{label} nodes"))
        for parameter_id in _list(
            _object(raw_node, f"{label} nodes[{index}]").get("params", []),
            f"{label} nodes[{index}].params",
        )
    }
    for parameter_id, raw_ref in refs.items():
        parameter_id = _text(parameter_id, f"{label} parameter ref id")
        _object(raw_ref, f"{label} parameter_refs[{parameter_id!r}]")
        if parameter_id not in used:
            raise ContractValidationError(
                f"{label} parameter ref {parameter_id!r} is not used by its nodes"
            )


def _region_record_ids(value: object, label: str) -> list[str]:
    records = _list(value, label)
    if not records:
        raise ContractValidationError(f"{label} must not be empty")
    identifiers = []
    for index, raw_record in enumerate(records):
        record = _object(raw_record, f"{label}[{index}]")
        identifiers.append(_text(record.get("node_id", record.get("id")), f"{label}[{index}] id"))
    if len(identifiers) != len(set(identifiers)):
        raise ContractValidationError(f"{label} must have unique ids")
    return identifiers


def _declared_mount_artifact(
    value: Any,
    label: str,
    declared_outputs: dict[str, Json],
    *,
    expected_kind: str,
) -> str:
    reference = _safe_relative_path(value, label)
    output = declared_outputs.get(reference)
    if output is None:
        raise ContractValidationError(
            f"{label} does not reference a declared candidate output"
        )
    if output["lifetime"] not in {"mount", "residency"}:
        raise ContractValidationError(
            f"{label} must reference a mount- or residency-lifetime artifact"
        )
    if output["kind"] != expected_kind:
        raise ContractValidationError(
            f"{label} must reference a {expected_kind!r} output"
        )
    return reference


def _safe_relative_path(value: Any, label: str) -> str:
    text = _text(value, label)
    relative = Path(text)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "." in relative.parts
        or relative.as_posix() != text
    ):
        raise ContractValidationError(
            f"{label} must be a normalized relative path"
        )
    return text


def _fields(record: Json, expected: set[str], label: str) -> None:
    actual = set(record)
    if actual != expected:
        raise ContractValidationError(
            f"{label} fields are invalid: "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _object(value: Any, label: str) -> Json:
    if not isinstance(value, dict):
        raise ContractValidationError(f"{label} must be an object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ContractValidationError(f"{label} must be a list")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractValidationError(f"{label} must be a non-empty string")
    return value


def _read_object(path: Path) -> Json:
    try:
        document = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ContractValidationError(
            f"runtime mount artifact is unreadable: {path}"
        ) from error
    return _object(document, f"runtime mount artifact {path}")
