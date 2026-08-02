from __future__ import annotations

import json
from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any

from nerve.compilation import Json, ModelCompileError
from nerve.resource_range_integrity import (
    ConcreteResourceInterval,
    PartitionRangeSeries,
    ResolvedResourceRange,
    read_verified_resource_range,
    resolve_partition_range,
    validate_partition_range_storage,
)


RESOURCE_RESIDENCY_SCHEMA = "nerve.compiled_resource_residency.v2"
RESOURCE_IDENTITY_ALGORITHM = "nerve.resource_identity_sha256.v1"
RESOURCE_STATE_MACHINE_SCHEMA = "nerve.resource_residency_state_machine.v1"
SUPPORTED_RESIDENCY_POLICIES = ("demand_retained", "eager")
RESOURCE_LIFETIMES = frozenset(("always_resident", "dynamic"))
RESOURCE_STATES = frozenset(("absent", "requested", "loading", "resident", "failed"))
SHA256_INTEGRITY_ALGORITHM = "sha256"
RESOLVED_PARTITION_GROUP_SCHEMA = "nerve.resolved_partition_group.v1"

_CONTRACT_FIELDS = frozenset(
    (
        "schema",
        "identity_algorithm",
        "state_machine_schema",
        "supported_policies",
        "resources",
        "atomic_groups",
        "partition_templates",
        "bindings",
        "selectors",
        "checkpoints",
    )
)
_RESOURCE_FIELDS = frozenset(
    ("id", "lifetime", "ranges", "dependencies", "compatibility")
)
_RANGE_FIELDS = frozenset(
    ("artifact_path", "byte_offset", "byte_count", "alignment_bytes", "integrity")
)
_INTEGRITY_FIELDS = frozenset(("algorithm", "digest"))
_COMPATIBILITY_FIELDS = frozenset(
    ("device_api", "storage_class", "read_only", "required_features")
)
_ATOMIC_GROUP_FIELDS = frozenset(
    ("id", "lifetime", "resource_ids", "dependencies")
)
_BINDING_FIELDS = frozenset(
    (
        "execution_scope",
        "component_id",
        "node_id",
        "parameter_id",
        "mapping",
    )
)
_ATOMIC_GROUP_BINDING_FIELDS = frozenset(
    ("kind", "atomic_group_id", "resource_id")
)
_SELECTED_ATOMIC_GROUP_BINDING_FIELDS = frozenset(
    (
        "kind",
        "atomic_group_id",
        "resource_id",
        "selection_signal",
        "selector_index",
        "parameter_slot",
    )
)
_PARTITION_MEMBER_BINDING_FIELDS = frozenset(
    ("kind", "partition_template_id", "resource_identity_seed")
)
_PARTITION_TEMPLATE_FIELDS = frozenset(
    (
        "id",
        "partition_count",
        "lifetime",
        "group_identity_seed",
        "member_templates",
        "dependencies",
    )
)
_PARTITION_MEMBER_FIELDS = frozenset(
    ("resource_identity_seed", "range_templates", "compatibility")
)
_RANGE_TEMPLATE_FIELDS = frozenset(
    (
        "artifact_path",
        "base_byte_offset",
        "stride_bytes",
        "byte_count",
        "alignment_bytes",
        "integrity",
    )
)
_INTEGRITY_TEMPLATE_FIELDS = frozenset(
    (
        "algorithm",
        "digest_table_path",
        "digest_table_byte_offset",
        "digest_stride_bytes",
        "table_sha256",
    )
)
_SELECTOR_FIELDS = frozenset(
    (
        "id",
        "execution_scope",
        "component_id",
        "node_id",
        "domain_id",
        "resource_count",
        "selection_signal",
        "encoding",
        "mapping",
    )
)
_SELECTION_ENCODING_FIELDS = frozenset(
    (
        "element_type",
        "selection_count_per_activation",
        "index_shift",
        "index_mask",
    )
)
_CHECKPOINT_FIELDS = frozenset(
    (
        "id",
        "execution_scope",
        "component_id",
        "after_node_id",
        "resume_node_id",
        "selector_ids",
    )
)


def residency_content_id(kind: str, payload: Json) -> str:
    canonical = json.dumps(
        {
            "algorithm": RESOURCE_IDENTITY_ALGORITHM,
            "kind": kind,
            "payload": payload,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return f"sha256:{sha256(canonical).hexdigest()}"


def resource_identity(resource: Json) -> str:
    return residency_content_id(
        "resource",
        {
            "lifetime": resource["lifetime"],
            "ranges": [
                {
                    "byte_count": byte_range["byte_count"],
                    "alignment_bytes": byte_range["alignment_bytes"],
                    "integrity": byte_range["integrity"],
                }
                for byte_range in resource["ranges"]
            ],
            "dependencies": resource["dependencies"],
            "compatibility": resource["compatibility"],
        },
    )


def atomic_group_identity(group: Json) -> str:
    return residency_content_id(
        "atomic_group",
        {
            "lifetime": group["lifetime"],
            "resource_ids": group["resource_ids"],
            "dependencies": group["dependencies"],
        },
    )


def partition_template_identity(template: Json) -> str:
    return residency_content_id(
        "partition_template",
        {
            "partition_count": template["partition_count"],
            "lifetime": template["lifetime"],
            "group_identity_seed": template["group_identity_seed"],
            "member_templates": [
                {
                    "resource_identity_seed": member["resource_identity_seed"],
                    "range_templates": [
                        {
                            "base_byte_offset": byte_range["base_byte_offset"],
                            "stride_bytes": byte_range["stride_bytes"],
                            "byte_count": byte_range["byte_count"],
                            "alignment_bytes": byte_range["alignment_bytes"],
                            "integrity": {
                                "algorithm": byte_range["integrity"]["algorithm"],
                                "digest_stride_bytes": byte_range["integrity"][
                                    "digest_stride_bytes"
                                ],
                                "table_sha256": byte_range["integrity"]["table_sha256"],
                            },
                        }
                        for byte_range in member["range_templates"]
                    ],
                    "compatibility": member["compatibility"],
                }
                for member in template["member_templates"]
            ],
            "dependencies": template["dependencies"],
        },
    )


def partition_group_identity_seed(
    partition_count: int, resource_identity_seeds: list[str]
) -> str:
    partition_count = _require_positive_int(
        partition_count, "partition group partition count"
    )
    seeds = [
        _require_content_id(seed, "partition group resource identity seed")
        for seed in resource_identity_seeds
    ]
    if seeds != sorted(set(seeds)):
        raise ModelCompileError(
            "partition group resource identity seeds must be unique and sorted"
        )
    return residency_content_id(
        "partition_group_seed",
        {
            "partition_count": partition_count,
            "resource_identity_seeds": seeds,
        },
    )


def selector_identity(selector: Json) -> str:
    return residency_content_id(
        "selector",
        {
            field: selector[field]
            for field in (
                "execution_scope",
                "component_id",
                "node_id",
                "domain_id",
                "resource_count",
                "selection_signal",
                "encoding",
                "mapping",
            )
        },
    )


def checkpoint_identity(checkpoint: Json) -> str:
    return residency_content_id(
        "checkpoint",
        {
            field: checkpoint[field]
            for field in (
                "execution_scope",
                "component_id",
                "after_node_id",
                "resume_node_id",
                "selector_ids",
            )
        },
    )


def derived_partition_identity(identity_seed: str, partition_index: int) -> str:
    _require_content_id(identity_seed, "partition identity seed")
    if (
        not isinstance(partition_index, int)
        or isinstance(partition_index, bool)
        or partition_index < 0
    ):
        raise ModelCompileError("partition identity index must be a non-negative integer")
    return residency_content_id(
        "partition",
        {"identity_seed": identity_seed, "partition_index": partition_index},
    )


def resolve_partition_atomic_group(
    package_dir: Path,
    contract: Json,
    *,
    partition_template_id: str,
    partition_index: int,
) -> Json:
    template_id = _require_content_id(
        partition_template_id, "partition template id"
    )
    templates = _require_object_list(
        contract.get("partition_templates"), "partition templates"
    )
    template = next(
        (
            candidate
            for candidate in templates
            if candidate.get("id") == template_id
        ),
        None,
    )
    if template is None:
        raise ModelCompileError(
            f"unknown partition template {partition_template_id!r}"
        )
    partition_count = _require_positive_int(
        template.get("partition_count"), "partition count"
    )
    if (
        not isinstance(partition_index, int)
        or isinstance(partition_index, bool)
        or partition_index < 0
        or partition_index >= partition_count
    ):
        raise ModelCompileError(
            f"partition index {partition_index!r} is outside [0, {partition_count})"
        )

    resources = []
    for member in _require_object_list(
        template.get("member_templates"), "partition member templates"
    ):
        seed = _require_content_id(
            member.get("resource_identity_seed"),
            "partition resource identity seed",
        )
        resolved_ranges = []
        for range_template in _require_object_list(
            member.get("range_templates"), "partition member ranges"
        ):
            integrity = _require_object(
                range_template.get("integrity"),
                "partition range integrity",
            )
            resolved = resolve_partition_range(
                package_dir,
                artifact_path=_require_safe_relative_path(
                    range_template.get("artifact_path"),
                    "partition range artifact",
                ),
                base_byte_offset=_require_non_negative_int(
                    range_template.get("base_byte_offset"),
                    "partition range base",
                ),
                stride_bytes=_require_positive_int(
                    range_template.get("stride_bytes"),
                    "partition range stride",
                ),
                byte_count=_require_positive_int(
                    range_template.get("byte_count"),
                    "partition range byte count",
                ),
                alignment_bytes=_require_power_of_two(
                    range_template.get("alignment_bytes"),
                    "partition range alignment",
                ),
                digest_table_path=_require_safe_relative_path(
                    integrity.get("digest_table_path"),
                    "partition digest table",
                ),
                digest_table_byte_offset=_require_non_negative_int(
                    integrity.get("digest_table_byte_offset"),
                    "partition digest offset",
                ),
                digest_stride_bytes=_require_positive_int(
                    integrity.get("digest_stride_bytes"),
                    "partition digest stride",
                ),
                partition_index=partition_index,
            )
            resolved_ranges.append(
                {
                    "artifact_path": resolved.artifact_path,
                    "byte_offset": resolved.byte_offset,
                    "byte_count": resolved.byte_count,
                    "alignment_bytes": resolved.alignment_bytes,
                    "integrity": {
                        "algorithm": SHA256_INTEGRITY_ALGORITHM,
                        "digest": resolved.sha256,
                    },
                }
            )
        resources.append(
            {
                "id": derived_partition_identity(seed, partition_index),
                "resource_identity_seed": seed,
                "lifetime": "dynamic",
                "ranges": resolved_ranges,
                "compatibility": deepcopy(member.get("compatibility")),
            }
        )
    resources.sort(key=lambda resource: resource["id"])
    group_seed = _require_content_id(
        template.get("group_identity_seed"), "partition group identity seed"
    )
    return {
        "schema": RESOLVED_PARTITION_GROUP_SCHEMA,
        "partition_template_id": template_id,
        "partition_index": partition_index,
        "atomic_group": {
            "id": derived_partition_identity(group_seed, partition_index),
            "resource_ids": [resource["id"] for resource in resources],
            "dependencies": list(template.get("dependencies", [])),
        },
        "resources": resources,
    }


def read_verified_partition_atomic_group(
    package_dir: Path, resolved_group: Json
) -> dict[str, list[bytes]]:
    if resolved_group.get("schema") != RESOLVED_PARTITION_GROUP_SCHEMA:
        raise ModelCompileError("resolved partition group schema is invalid")
    resources = _require_object_list(
        resolved_group.get("resources"), "resolved partition resources"
    )
    atomic_group = _require_object(
        resolved_group.get("atomic_group"),
        "resolved partition atomic group",
    )
    declared_resource_ids = atomic_group.get("resource_ids")
    if (
        not isinstance(declared_resource_ids, list)
        or not all(isinstance(value, str) for value in declared_resource_ids)
        or len(declared_resource_ids) != len(set(declared_resource_ids))
    ):
        raise ModelCompileError(
            "resolved partition atomic group resource ids are invalid"
        )
    actual_resource_ids = [
        _require_content_id(
            resource.get("id"), "resolved partition resource id"
        )
        for resource in resources
    ]
    if (
        actual_resource_ids != declared_resource_ids
        or len(actual_resource_ids) != len(set(actual_resource_ids))
    ):
        raise ModelCompileError(
            "resolved partition atomic group membership is inconsistent"
        )
    loaded: dict[str, list[bytes]] = {}
    for resource, resource_id in zip(
        resources, actual_resource_ids, strict=True
    ):
        payloads = []
        for byte_range in _require_object_list(
            resource.get("ranges"), "resolved partition ranges"
        ):
            integrity = _require_object(
                byte_range.get("integrity"),
                "resolved partition range integrity",
            )
            if integrity.get("algorithm") != SHA256_INTEGRITY_ALGORITHM:
                raise ModelCompileError(
                    "resolved partition range integrity must use SHA-256"
                )
            payloads.append(
                read_verified_resource_range(
                    package_dir,
                    ResolvedResourceRange(
                        artifact_path=_require_safe_relative_path(
                            byte_range.get("artifact_path"),
                            "resolved partition artifact",
                        ),
                        byte_offset=_require_non_negative_int(
                            byte_range.get("byte_offset"),
                            "resolved partition byte offset",
                        ),
                        byte_count=_require_positive_int(
                            byte_range.get("byte_count"),
                            "resolved partition byte count",
                        ),
                        alignment_bytes=_require_power_of_two(
                            byte_range.get("alignment_bytes"),
                            "resolved partition alignment",
                        ),
                        sha256=_require_sha256(
                            integrity.get("digest"),
                            "resolved partition SHA-256",
                        ),
                    ),
                )
            )
        loaded[resource_id] = payloads
    return loaded


def residency_state_transition_allowed(
    current: str,
    following: str,
    *,
    explicit_lifecycle: bool = False,
) -> bool:
    if current not in RESOURCE_STATES or following not in RESOURCE_STATES:
        return False
    if current == "absent":
        return following == "requested"
    if current == "requested":
        return following in ("loading", "failed", "absent")
    if current == "loading":
        return following in ("resident", "failed", "absent")
    if current in ("resident", "failed"):
        return explicit_lifecycle and following == "absent"
    return False


def build_eager_resource_residency_contract(
    *,
    package_dir: Path,
    tensor_index: Json,
    manifest: Json,
) -> Json:
    """Describe the current eager physical plan without model-family knowledge.

    Every parameter region actually referenced by a compiled circuit becomes an
    independently identified immutable resource. The next compiler milestone
    may combine or partition these regions after access-pattern analysis; this
    baseline contract is deliberately exact rather than predictive.
    """

    tensor_bindings = compiled_parameter_bindings(manifest)
    tensors = tensor_index.get("tensors")
    if not isinstance(tensors, dict):
        raise ModelCompileError("tensor index has no tensor mapping")

    source_headers, artifact_byte_counts = compiled_resource_artifact_metadata(
        package_dir, tensor_index
    )
    resources_by_id: dict[str, Json] = {}
    resource_id_by_tensor: dict[str, str] = {}
    for tensor_name in sorted(tensor_bindings):
        resource = compiled_immutable_resource(
            package_dir=package_dir,
            tensor_index=tensor_index,
            tensor_name=tensor_name,
            lifetime="always_resident",
            source_headers=source_headers,
            artifact_byte_counts=artifact_byte_counts,
        )
        resource_id = resource["id"]
        resource_id_by_tensor[tensor_name] = resource_id
        existing = resources_by_id.get(resource_id)
        if existing is None or _range_location_key(resource) < _range_location_key(existing):
            resources_by_id[resource_id] = resource
    if not resources_by_id:
        raise ModelCompileError(
            "compiled package has no immutable parameter resources"
        )
    eager_spine_group = {
        "id": "",
        "lifetime": "always_resident",
        "resource_ids": sorted(resources_by_id),
        "dependencies": [],
    }
    eager_spine_group["id"] = atomic_group_identity(eager_spine_group)

    bindings = []
    for tensor_name, uses in tensor_bindings.items():
        for use in uses:
            bindings.append(
                {
                    **use,
                    "mapping": {
                        "kind": "atomic_group",
                        "atomic_group_id": eager_spine_group["id"],
                        "resource_id": resource_id_by_tensor[tensor_name],
                    },
                }
            )
    bindings.sort(key=_binding_key)

    contract = {
        "schema": RESOURCE_RESIDENCY_SCHEMA,
        "identity_algorithm": RESOURCE_IDENTITY_ALGORITHM,
        "state_machine_schema": RESOURCE_STATE_MACHINE_SCHEMA,
        "supported_policies": list(SUPPORTED_RESIDENCY_POLICIES),
        "resources": sorted(resources_by_id.values(), key=lambda resource: resource["id"]),
        "atomic_groups": [eager_spine_group],
        "partition_templates": [],
        "bindings": bindings,
        "selectors": [],
        "checkpoints": [],
    }
    validate_resource_residency_contract(package_dir, contract, manifest)
    return contract


def compiled_immutable_resource(
    *,
    package_dir: Path,
    tensor_index: Json,
    tensor_name: str,
    lifetime: str,
    source_headers: dict[str, int],
    artifact_byte_counts: dict[str, int],
) -> Json:
    tensors = tensor_index.get("tensors")
    metadata = tensors.get(tensor_name) if isinstance(tensors, dict) else None
    if not isinstance(metadata, dict):
        raise ModelCompileError(
            f"compiled circuit references tensor {tensor_name!r} absent from the index"
        )
    source_file = _require_safe_relative_path(
        metadata.get("source_file"), f"tensor {tensor_name!r} source"
    )
    offsets = metadata.get("data_offsets")
    header_bytes = metadata.get(
        "safetensors_header_bytes",
        source_headers.get(source_file),
    )
    byte_count = metadata.get("byte_count")
    digest = metadata.get("data_sha256")
    if (
        lifetime not in RESOURCE_LIFETIMES
        or not isinstance(offsets, list)
        or len(offsets) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in offsets
        )
        or offsets[1] < offsets[0]
        or not isinstance(header_bytes, int)
        or isinstance(header_bytes, bool)
        or header_bytes <= 0
        or not isinstance(byte_count, int)
        or isinstance(byte_count, bool)
        or byte_count <= 0
        or offsets[1] - offsets[0] != byte_count
        or not _is_lower_hex_sha256(digest)
    ):
        raise ModelCompileError(
            f"compiled tensor {tensor_name!r} cannot form a bounded residency range"
        )
    absolute_offset = 8 + header_bytes + offsets[0]
    artifact_bytes = artifact_byte_counts.get(source_file)
    if artifact_bytes is None:
        raise ModelCompileError(
            f"compiled tensor {tensor_name!r} source has no inspected artifact size"
        )
    if absolute_offset + byte_count > artifact_bytes:
        raise ModelCompileError(
            f"compiled tensor {tensor_name!r} range exceeds {source_file!r}"
        )
    resource = {
        "id": "",
        "lifetime": lifetime,
        "ranges": [
            {
                "artifact_path": source_file,
                "byte_offset": absolute_offset,
                "byte_count": byte_count,
                "alignment_bytes": _largest_power_of_two_divisor(absolute_offset),
                "integrity": {
                    "algorithm": SHA256_INTEGRITY_ALGORITHM,
                    "digest": digest,
                },
            }
        ],
        "dependencies": [],
        "compatibility": {
            "device_api": "vulkan",
            "storage_class": "storage_buffer",
            "read_only": True,
            "required_features": [],
        },
    }
    resource["id"] = resource_identity(resource)
    return resource


def compiled_resource_artifact_metadata(
    package_dir: Path,
    tensor_index: Json,
) -> tuple[dict[str, int], dict[str, int]]:
    source_headers = {
        record["path"]: record["safetensors_header_bytes"]
        for record in tensor_index.get("source", {}).get("weights_files", [])
        if isinstance(record, dict)
        and isinstance(record.get("path"), str)
        and isinstance(record.get("safetensors_header_bytes"), int)
        and not isinstance(record.get("safetensors_header_bytes"), bool)
    }
    tensors = tensor_index.get("tensors")
    if not isinstance(tensors, dict):
        raise ModelCompileError("tensor index has no tensor mapping")
    source_files = {
        metadata.get("source_file")
        for metadata in tensors.values()
        if isinstance(metadata, dict)
        and isinstance(metadata.get("source_file"), str)
    }
    artifact_byte_counts: dict[str, int] = {}
    for source_file in source_files:
        try:
            artifact_byte_counts[source_file] = (package_dir / source_file).stat().st_size
        except OSError as error:
            raise ModelCompileError(
                f"compiled resource source {source_file!r} cannot be inspected: {error}"
            ) from error
    return source_headers, artifact_byte_counts


def validate_resource_residency_contract(
    package_dir: Path,
    contract: Any,
    manifest: Json,
) -> None:
    contract = _require_object(contract, "compiled resource residency contract")
    _require_exact_fields(contract, _CONTRACT_FIELDS, "compiled resource residency contract")
    if contract["schema"] != RESOURCE_RESIDENCY_SCHEMA:
        raise ModelCompileError(
            f"unsupported compiled resource residency schema {contract['schema']!r}"
        )
    if contract["identity_algorithm"] != RESOURCE_IDENTITY_ALGORITHM:
        raise ModelCompileError("compiled resource residency identity algorithm is invalid")
    if contract["state_machine_schema"] != RESOURCE_STATE_MACHINE_SCHEMA:
        raise ModelCompileError("compiled resource residency state machine is invalid")
    if contract["supported_policies"] != list(SUPPORTED_RESIDENCY_POLICIES):
        raise ModelCompileError(
            "compiled resource residency policies must declare demand_retained and eager"
        )

    resources = _require_object_list(contract["resources"], "resources")
    groups = _require_object_list(contract["atomic_groups"], "atomic groups")
    templates = _require_object_list(
        contract["partition_templates"], "partition templates"
    )
    bindings = _require_object_list(contract["bindings"], "resource bindings")
    selectors = _require_object_list(contract["selectors"], "resource selectors")
    checkpoints = _require_object_list(
        contract["checkpoints"], "residency checkpoints"
    )

    resource_by_id = _validate_resources(package_dir, resources)
    group_by_id = _validate_atomic_groups(groups, resource_by_id)
    template_by_id, partition_series = _validate_partition_templates(
        package_dir, templates, group_by_id
    )
    validate_partition_range_storage(
        package_dir,
        concrete_intervals=[
            ConcreteResourceInterval(
                artifact_path=byte_range["artifact_path"],
                byte_offset=byte_range["byte_offset"],
                byte_count=byte_range["byte_count"],
                resource_id=resource["id"],
            )
            for resource in resources
            for byte_range in resource["ranges"]
        ],
        partition_series=partition_series,
    )
    parameter_semantics, component_nodes = _compiled_semantics(manifest)
    _validate_bindings(
        bindings,
        group_by_id,
        template_by_id,
        selectors,
        parameter_semantics,
    )
    selector_by_id = _validate_selectors(
        selectors,
        group_by_id,
        template_by_id,
        component_nodes,
    )
    _validate_checkpoints(checkpoints, selector_by_id, component_nodes)


def _validate_resources(package_dir: Path, resources: list[Json]) -> dict[str, Json]:
    _require_sorted_unique_ids(resources, "resources")
    resource_by_id: dict[str, Json] = {}
    artifact_intervals: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for resource in resources:
        _require_exact_fields(resource, _RESOURCE_FIELDS, "resource")
        resource_id = _require_content_id(resource["id"], "resource id")
        lifetime = resource["lifetime"]
        if lifetime not in RESOURCE_LIFETIMES:
            raise ModelCompileError(f"resource {resource_id!r} has invalid lifetime")
        dependencies = _require_sorted_content_ids(
            resource["dependencies"], f"resource {resource_id!r} dependencies"
        )
        compatibility = _validate_compatibility(
            resource["compatibility"], f"resource {resource_id!r} compatibility"
        )
        byte_ranges = _require_object_list(
            resource["ranges"], f"resource {resource_id!r} ranges"
        )
        if not byte_ranges:
            raise ModelCompileError(f"resource {resource_id!r} has no byte ranges")
        range_keys = []
        for byte_range in byte_ranges:
            _require_exact_fields(byte_range, _RANGE_FIELDS, "resource byte range")
            artifact_path = _require_safe_relative_path(
                byte_range["artifact_path"], "resource artifact"
            )
            byte_offset = _require_non_negative_int(
                byte_range["byte_offset"], "resource byte offset"
            )
            byte_count = _require_positive_int(
                byte_range["byte_count"], "resource byte count"
            )
            alignment = _require_power_of_two(
                byte_range["alignment_bytes"], "resource alignment"
            )
            if byte_offset % alignment:
                raise ModelCompileError(
                    f"resource range at {artifact_path}:{byte_offset} violates alignment {alignment}"
                )
            integrity = _validate_integrity(
                byte_range["integrity"], "resource range integrity"
            )
            path = package_dir / artifact_path
            try:
                artifact_bytes = path.stat().st_size
            except OSError as error:
                raise ModelCompileError(
                    f"resource artifact {artifact_path!r} cannot be inspected: {error}"
                ) from error
            end = byte_offset + byte_count
            if end > artifact_bytes:
                raise ModelCompileError(
                    f"resource range {artifact_path}:{byte_offset}+{byte_count} exceeds its artifact"
                )
            range_keys.append((artifact_path, byte_offset, byte_count))
            artifact_intervals[artifact_path].append((byte_offset, end, resource_id))
            if integrity["algorithm"] != SHA256_INTEGRITY_ALGORITHM:
                raise ModelCompileError(
                    f"resource {resource_id!r} uses unsupported range integrity"
                )
        if range_keys != sorted(set(range_keys)):
            raise ModelCompileError(
                f"resource {resource_id!r} ranges must be unique and physically sorted"
            )
        if dependencies.count(resource_id):
            raise ModelCompileError(f"resource {resource_id!r} depends on itself")
        if resource_identity(resource) != resource_id:
            raise ModelCompileError(
                f"resource {resource_id!r} identity does not match its content contract"
            )
        resource_by_id[resource_id] = {
            **resource,
            "dependencies": dependencies,
            "compatibility": compatibility,
        }

    for resource_id, resource in resource_by_id.items():
        unknown = set(resource["dependencies"]).difference(resource_by_id)
        if unknown:
            raise ModelCompileError(
                f"resource {resource_id!r} depends on unknown resources {sorted(unknown)}"
            )
    _reject_dependency_cycles(
        {resource_id: resource["dependencies"] for resource_id, resource in resource_by_id.items()},
        "resource",
    )
    for artifact_path, intervals in artifact_intervals.items():
        intervals.sort()
        for previous, current in zip(intervals, intervals[1:]):
            if current[0] < previous[1]:
                raise ModelCompileError(
                    f"resource ranges overlap in artifact {artifact_path!r}"
                )
    return resource_by_id


def _validate_atomic_groups(
    groups: list[Json], resource_by_id: dict[str, Json]
) -> dict[str, Json]:
    _require_sorted_unique_ids(groups, "atomic groups")
    group_by_id: dict[str, Json] = {}
    membership = Counter()
    for group in groups:
        _require_exact_fields(group, _ATOMIC_GROUP_FIELDS, "atomic group")
        group_id = _require_content_id(group["id"], "atomic group id")
        lifetime = group["lifetime"]
        if lifetime not in RESOURCE_LIFETIMES:
            raise ModelCompileError(f"atomic group {group_id!r} has invalid lifetime")
        resource_ids = _require_sorted_content_ids(
            group["resource_ids"], f"atomic group {group_id!r} resources"
        )
        if not resource_ids:
            raise ModelCompileError(f"atomic group {group_id!r} has no resources")
        unknown_resources = set(resource_ids).difference(resource_by_id)
        if unknown_resources:
            raise ModelCompileError(
                f"atomic group {group_id!r} contains unknown resources {sorted(unknown_resources)}"
            )
        if any(resource_by_id[item]["lifetime"] != lifetime for item in resource_ids):
            raise ModelCompileError(
                f"atomic group {group_id!r} lifetime disagrees with a member"
            )
        dependencies = _require_sorted_content_ids(
            group["dependencies"], f"atomic group {group_id!r} dependencies"
        )
        if dependencies.count(group_id):
            raise ModelCompileError(f"atomic group {group_id!r} depends on itself")
        if atomic_group_identity(group) != group_id:
            raise ModelCompileError(
                f"atomic group {group_id!r} identity does not match its content contract"
            )
        membership.update(resource_ids)
        group_by_id[group_id] = group

    missing_membership = set(resource_by_id).difference(membership)
    repeated_membership = sorted(
        resource_id for resource_id, count in membership.items() if count != 1
    )
    if missing_membership or repeated_membership:
        raise ModelCompileError(
            "concrete resources must belong to exactly one atomic group"
        )
    for group_id, group in group_by_id.items():
        unknown = set(group["dependencies"]).difference(group_by_id)
        if unknown:
            raise ModelCompileError(
                f"atomic group {group_id!r} depends on unknown groups {sorted(unknown)}"
            )
    _reject_dependency_cycles(
        {group_id: group["dependencies"] for group_id, group in group_by_id.items()},
        "atomic group",
    )
    return group_by_id


def _validate_partition_templates(
    package_dir: Path,
    templates: list[Json],
    group_by_id: dict[str, Json],
) -> tuple[dict[str, Json], list[PartitionRangeSeries]]:
    _require_sorted_unique_ids(templates, "partition templates")
    template_by_id: dict[str, Json] = {}
    partition_series: list[PartitionRangeSeries] = []
    for template in templates:
        _require_exact_fields(
            template, _PARTITION_TEMPLATE_FIELDS, "partition template"
        )
        template_id = _require_content_id(template["id"], "partition template id")
        partition_count = _require_positive_int(
            template["partition_count"], "partition count"
        )
        if template["lifetime"] != "dynamic":
            raise ModelCompileError(
                f"partition template {template_id!r} must have dynamic lifetime"
            )
        group_identity_seed = _require_content_id(
            template["group_identity_seed"], "partition group identity seed"
        )
        dependencies = _require_sorted_content_ids(
            template["dependencies"],
            f"partition template {template_id!r} dependencies",
        )
        unknown_dependencies = set(dependencies).difference(group_by_id)
        if unknown_dependencies:
            raise ModelCompileError(
                f"partition template {template_id!r} depends on unknown groups"
            )
        members = _require_object_list(
            template["member_templates"],
            f"partition template {template_id!r} members",
        )
        if not members:
            raise ModelCompileError(
                f"partition template {template_id!r} has no member templates"
            )
        member_seeds = []
        for member in members:
            _require_exact_fields(
                member, _PARTITION_MEMBER_FIELDS, "partition member template"
            )
            seed = _require_content_id(
                member["resource_identity_seed"], "partition resource identity seed"
            )
            member_seeds.append(seed)
            _validate_compatibility(
                member["compatibility"], "partition member compatibility"
            )
            ranges = _require_object_list(
                member["range_templates"], "partition member ranges"
            )
            if not ranges:
                raise ModelCompileError(
                    f"partition member {seed!r} has no range templates"
                )
            for byte_range in ranges:
                partition_series.append(
                    _validate_range_template(
                        package_dir,
                        byte_range,
                        partition_count,
                        template_id=template_id,
                        resource_identity_seed=seed,
                    )
                )
        if member_seeds != sorted(set(member_seeds)):
            raise ModelCompileError(
                f"partition template {template_id!r} member seeds must be unique and sorted"
            )
        if (
            partition_group_identity_seed(partition_count, member_seeds)
            != group_identity_seed
        ):
            raise ModelCompileError(
                f"partition template {template_id!r} group identity seed "
                "does not exactly cover its members"
            )
        if partition_template_identity(template) != template_id:
            raise ModelCompileError(
                f"partition template {template_id!r} identity does not match its contract"
            )
        template_by_id[template_id] = template
    return template_by_id, partition_series


def _validate_range_template(
    package_dir: Path,
    byte_range: Json,
    partition_count: int,
    *,
    template_id: str,
    resource_identity_seed: str,
) -> PartitionRangeSeries:
    _require_exact_fields(
        byte_range, _RANGE_TEMPLATE_FIELDS, "partition range template"
    )
    artifact_path = _require_safe_relative_path(
        byte_range["artifact_path"], "partition range artifact"
    )
    base = _require_non_negative_int(
        byte_range["base_byte_offset"], "partition range base"
    )
    stride = _require_positive_int(
        byte_range["stride_bytes"], "partition range stride"
    )
    byte_count = _require_positive_int(
        byte_range["byte_count"], "partition range byte count"
    )
    alignment = _require_power_of_two(
        byte_range["alignment_bytes"], "partition range alignment"
    )
    if base % alignment or stride % alignment:
        raise ModelCompileError("partition range base and stride violate alignment")
    if stride < byte_count:
        raise ModelCompileError(
            "partition range stride overlaps adjacent resources"
        )
    last_end = base + (partition_count - 1) * stride + byte_count
    try:
        artifact_bytes = (package_dir / artifact_path).stat().st_size
    except OSError as error:
        raise ModelCompileError(
            f"partition range artifact {artifact_path!r} cannot be inspected: {error}"
        ) from error
    if last_end > artifact_bytes:
        raise ModelCompileError(
            f"partition range template exceeds artifact {artifact_path!r}"
        )
    integrity = _require_object(
        byte_range["integrity"], "partition range integrity"
    )
    _require_exact_fields(
        integrity, _INTEGRITY_TEMPLATE_FIELDS, "partition range integrity"
    )
    if integrity["algorithm"] != "sha256_table":
        raise ModelCompileError("partition range integrity must use sha256_table")
    digest_path = _require_safe_relative_path(
        integrity["digest_table_path"], "partition digest table"
    )
    digest_offset = _require_non_negative_int(
        integrity["digest_table_byte_offset"], "partition digest table offset"
    )
    digest_stride = _require_positive_int(
        integrity["digest_stride_bytes"], "partition digest stride"
    )
    if digest_stride != 32 or digest_offset % 32:
        raise ModelCompileError(
            "partition SHA-256 table must use aligned 32-byte entries"
        )
    table_sha256 = integrity["table_sha256"]
    if not _is_lower_hex_sha256(table_sha256):
        raise ModelCompileError("partition digest table SHA-256 is invalid")
    try:
        table_bytes = (package_dir / digest_path).stat().st_size
    except OSError as error:
        raise ModelCompileError(
            f"partition digest table {digest_path!r} cannot be inspected: {error}"
        ) from error
    if digest_offset + (partition_count - 1) * digest_stride + 32 > table_bytes:
        raise ModelCompileError("partition digest table is too small")
    return PartitionRangeSeries(
        template_id=template_id,
        resource_identity_seed=resource_identity_seed,
        artifact_path=artifact_path,
        base_byte_offset=base,
        stride_bytes=stride,
        byte_count=byte_count,
        partition_count=partition_count,
        digest_table_path=digest_path,
        digest_table_byte_offset=digest_offset,
        digest_stride_bytes=digest_stride,
        table_sha256=table_sha256,
    )


def _validate_bindings(
    bindings: list[Json],
    group_by_id: dict[str, Json],
    template_by_id: dict[str, Json],
    selectors: list[Json],
    parameter_semantics: set[tuple[str, str, str, str]],
) -> None:
    resource_ids_by_group = {
        group_id: set(group["resource_ids"])
        for group_id, group in group_by_id.items()
    }
    member_seeds_by_template = {
        template_id: {
            member["resource_identity_seed"]
            for member in template["member_templates"]
        }
        for template_id, template in template_by_id.items()
    }
    keys = []
    bound_semantics = []
    bound_concrete_resources: set[str] = set()
    bound_partition_members: set[tuple[str, str]] = set()
    selected_slots: dict[
        tuple[str, str, str, str], list[tuple[str, int, int]]
    ] = (
        defaultdict(list)
    )
    for binding in bindings:
        _require_exact_fields(binding, _BINDING_FIELDS, "resource binding")
        for field in ("execution_scope", "component_id", "node_id", "parameter_id"):
            _require_non_empty_string(binding[field], f"resource binding {field}")
        mapping = _require_object(binding["mapping"], "resource binding mapping")
        if mapping.get("kind") == "atomic_group":
            _require_exact_fields(
                mapping,
                _ATOMIC_GROUP_BINDING_FIELDS,
                "atomic group resource binding",
            )
            group_id = _require_content_id(
                mapping["atomic_group_id"], "resource binding atomic group id"
            )
            if group_id not in group_by_id:
                raise ModelCompileError(
                    f"resource binding references unknown atomic group {group_id!r}"
                )
            resource_id = _require_content_id(
                mapping["resource_id"], "resource binding resource id"
            )
            if resource_id not in resource_ids_by_group[group_id]:
                raise ModelCompileError(
                    "resource binding maps a resource outside its atomic group"
                )
            bound_concrete_resources.add(resource_id)
        elif mapping.get("kind") == "selected_atomic_group":
            _require_exact_fields(
                mapping,
                _SELECTED_ATOMIC_GROUP_BINDING_FIELDS,
                "selected atomic group resource binding",
            )
            group_id = _require_content_id(
                mapping["atomic_group_id"],
                "resource binding selected atomic group id",
            )
            if group_id not in group_by_id:
                raise ModelCompileError(
                    f"resource binding references unknown atomic group {group_id!r}"
                )
            resource_id = _require_content_id(
                mapping["resource_id"], "resource binding selected resource id"
            )
            if resource_id not in resource_ids_by_group[group_id]:
                raise ModelCompileError(
                    "selected resource binding maps a resource outside its atomic group"
                )
            _require_non_negative_int(
                mapping["selector_index"], "selected resource selector index"
            )
            _require_non_negative_int(
                mapping["parameter_slot"], "selected resource parameter slot"
            )
            _require_non_empty_string(
                mapping["selection_signal"], "selected resource selection signal"
            )
            bound_concrete_resources.add(resource_id)
            selected_slots[
                (
                    binding["execution_scope"],
                    binding["component_id"],
                    binding["node_id"],
                    mapping["selection_signal"],
                )
            ].append(
                (
                    group_id,
                    mapping["selector_index"],
                    mapping["parameter_slot"],
                )
            )
        elif mapping.get("kind") == "partition_template_member":
            _require_exact_fields(
                mapping,
                _PARTITION_MEMBER_BINDING_FIELDS,
                "partition member resource binding",
            )
            template_id = _require_content_id(
                mapping["partition_template_id"],
                "resource binding partition template id",
            )
            resource_seed = _require_content_id(
                mapping["resource_identity_seed"],
                "resource binding partition resource seed",
            )
            template = template_by_id.get(template_id)
            if template is None or resource_seed not in member_seeds_by_template[template_id]:
                raise ModelCompileError(
                    "resource binding references an unknown partition template member"
                )
            bound_partition_members.add((template_id, resource_seed))
        else:
            raise ModelCompileError(
                f"resource binding has unsupported mapping {mapping.get('kind')!r}"
            )
        keys.append(_binding_key(binding))
        bound_semantics.append(_binding_key(binding)[:4])
    if keys != sorted(set(keys)):
        raise ModelCompileError("resource bindings must be unique and sorted")
    if Counter(bound_semantics) != Counter(parameter_semantics):
        raise ModelCompileError(
            "resource bindings must exactly cover compiled parameter semantics"
        )
    for (scope, component_id, node_id, selection_signal), slots in selected_slots.items():
        candidates = []
        for selector in selectors:
            mapping = selector.get("mapping")
            if (
                selector.get("execution_scope") != scope
                or selector.get("component_id") != component_id
                or selector.get("selection_signal") != selection_signal
                or not isinstance(mapping, dict)
                or mapping.get("kind") != "group_table"
                or not isinstance(mapping.get("atomic_group_ids"), list)
            ):
                continue
            group_ids = mapping["atomic_group_ids"]
            if all(
                selector_index < len(group_ids)
                and group_ids[selector_index] == group_id
                for group_id, selector_index, _parameter_slot in slots
            ):
                candidates.append(selector)
        if len(candidates) != 1:
            raise ModelCompileError(
                f"selected resource bindings for {scope} {component_id}.{node_id} "
                "do not map exactly one group-table selector"
            )
        resource_count = candidates[0].get("resource_count")
        slots_by_selector: dict[int, set[int]] = defaultdict(set)
        for _group_id, selector_index, parameter_slot in slots:
            if parameter_slot in slots_by_selector[selector_index]:
                raise ModelCompileError(
                    f"selected resource bindings for {scope} {component_id}.{node_id} "
                    "repeat a selector parameter slot"
                )
            slots_by_selector[selector_index].add(parameter_slot)
        if (
            not isinstance(resource_count, int)
            or isinstance(resource_count, bool)
            or set(slots_by_selector) != set(range(resource_count))
            or not slots_by_selector
        ):
            raise ModelCompileError(
                f"selected resource bindings for {scope} {component_id}.{node_id} "
                "do not cover every selector index"
            )
        parameter_slots = set(range(len(next(iter(slots_by_selector.values())))))
        if any(slots != parameter_slots for slots in slots_by_selector.values()):
            raise ModelCompileError(
                f"selected resource bindings for {scope} {component_id}.{node_id} "
                "do not define one contiguous parameter-slot layout"
            )
    expected_concrete_resources = {
        resource_id
        for group in group_by_id.values()
        for resource_id in group["resource_ids"]
    }
    expected_partition_members = {
        (template_id, member["resource_identity_seed"])
        for template_id, template in template_by_id.items()
        for member in template["member_templates"]
    }
    if (
        bound_concrete_resources != expected_concrete_resources
        or bound_partition_members != expected_partition_members
    ):
        raise ModelCompileError(
            "resource bindings do not completely cover atomic resource membership"
        )


def _validate_selectors(
    selectors: list[Json],
    group_by_id: dict[str, Json],
    template_by_id: dict[str, Json],
    component_nodes: dict[tuple[str, str], dict[str, Json]],
) -> dict[str, Json]:
    _require_sorted_unique_ids(selectors, "selectors")
    selector_by_id = {}
    selected_dynamic_groups: set[str] = set()
    selected_templates: set[str] = set()
    for selector in selectors:
        _require_exact_fields(selector, _SELECTOR_FIELDS, "resource selector")
        selector_id = _require_content_id(selector["id"], "resource selector id")
        for field in (
            "execution_scope",
            "component_id",
            "node_id",
            "domain_id",
            "selection_signal",
        ):
            _require_non_empty_string(selector[field], f"selector {field}")
        nodes = component_nodes.get(
            (selector["execution_scope"], selector["component_id"])
        )
        node = None if nodes is None else nodes.get(selector["node_id"])
        selection_domain = (
            node.get("attrs", {}).get("selection_domain")
            if isinstance(node, dict) and isinstance(node.get("attrs", {}), dict)
            else None
        )
        if (
            not isinstance(selection_domain, dict)
            or set(selection_domain)
            != {"id", "resource_count", "selection_signal", "encoding"}
            or selection_domain.get("id") != selector["domain_id"]
            or selection_domain.get("resource_count") != selector["resource_count"]
            or selection_domain.get("selection_signal")
            != selector["selection_signal"]
            or selection_domain.get("encoding") != selector["encoding"]
        ):
            raise ModelCompileError(
                f"selector {selector_id!r} does not match compiled node semantics"
            )
        if selector["selection_signal"] not in node.get("outputs", []):
            raise ModelCompileError(
                f"selector {selector_id!r} selection signal is not a node output"
            )
        resource_count = _require_positive_int(
            selector["resource_count"], "selector resource count"
        )
        encoding = _require_object(
            selector["encoding"], "selector selection encoding"
        )
        _require_exact_fields(
            encoding,
            _SELECTION_ENCODING_FIELDS,
            "selector selection encoding",
        )
        if encoding["element_type"] != "u32":
            raise ModelCompileError(
                f"selector {selector_id!r} has unsupported selection element type"
            )
        _require_positive_int(
            encoding["selection_count_per_activation"],
            "selector selection count per activation",
        )
        index_shift = _require_non_negative_int(
            encoding["index_shift"], "selector selection index shift"
        )
        index_mask = _require_positive_int(
            encoding["index_mask"], "selector selection index mask"
        )
        if (
            index_shift >= 32
            or index_mask > 0xFFFFFFFF
            or index_mask > 0xFFFFFFFF >> index_shift
            or index_mask & (index_mask + 1) != 0
            or (resource_count - 1) & index_mask != resource_count - 1
        ):
            raise ModelCompileError(
                f"selector {selector_id!r} has invalid selection index encoding"
            )
        mapping = _require_object(selector["mapping"], "selector mapping")
        kind = mapping.get("kind")
        if kind == "group_table":
            _require_exact_fields(
                mapping,
                frozenset(("kind", "atomic_group_ids")),
                "selector group table",
            )
            group_ids = _require_content_ids(
                mapping["atomic_group_ids"], "selector group table"
            )
            if len(group_ids) != resource_count:
                raise ModelCompileError(
                    f"selector {selector_id!r} resource count disagrees with its group table"
                )
            unknown = set(group_ids).difference(group_by_id)
            if unknown:
                raise ModelCompileError(
                    f"selector {selector_id!r} maps unknown atomic groups"
                )
            if any(group_by_id[group_id]["lifetime"] != "dynamic" for group_id in group_ids):
                raise ModelCompileError(
                    f"selector {selector_id!r} maps an always-resident group"
                )
            selected_dynamic_groups.update(group_ids)
        elif kind == "partition_template":
            _require_exact_fields(
                mapping,
                frozenset(("kind", "partition_template_id")),
                "selector partition mapping",
            )
            template_id = _require_content_id(
                mapping["partition_template_id"], "selector partition template id"
            )
            template = template_by_id.get(template_id)
            if template is None:
                raise ModelCompileError(
                    f"selector {selector_id!r} maps an unknown partition template"
                )
            if template["partition_count"] != resource_count:
                raise ModelCompileError(
                    f"selector {selector_id!r} resource count disagrees with its template"
                )
            selected_templates.add(template_id)
        else:
            raise ModelCompileError(
                f"selector {selector_id!r} has unsupported mapping {kind!r}"
            )
        if selector_identity(selector) != selector_id:
            raise ModelCompileError(
                f"selector {selector_id!r} identity does not match its semantics"
            )
        selector_by_id[selector_id] = selector

    declared_dynamic_groups = {
        group_id
        for group_id, group in group_by_id.items()
        if group["lifetime"] == "dynamic"
    }
    if selected_dynamic_groups != declared_dynamic_groups:
        raise ModelCompileError(
            "every concrete dynamic atomic group must be mapped by a selector"
        )
    if selected_templates != set(template_by_id):
        raise ModelCompileError(
            "every partition template must be mapped by at least one selector"
        )
    return selector_by_id


def _validate_checkpoints(
    checkpoints: list[Json],
    selector_by_id: dict[str, Json],
    component_nodes: dict[tuple[str, str], dict[str, Json]],
) -> None:
    _require_sorted_unique_ids(checkpoints, "residency checkpoints")
    selector_owners = Counter()
    for checkpoint in checkpoints:
        _require_exact_fields(checkpoint, _CHECKPOINT_FIELDS, "residency checkpoint")
        checkpoint_id = _require_content_id(
            checkpoint["id"], "residency checkpoint id"
        )
        for field in (
            "execution_scope",
            "component_id",
            "after_node_id",
            "resume_node_id",
        ):
            _require_non_empty_string(checkpoint[field], f"checkpoint {field}")
        selector_ids = _require_sorted_content_ids(
            checkpoint["selector_ids"], "checkpoint selectors"
        )
        if not selector_ids:
            raise ModelCompileError(
                f"residency checkpoint {checkpoint_id!r} has no selectors"
            )
        nodes = component_nodes.get(
            (checkpoint["execution_scope"], checkpoint["component_id"])
        )
        if nodes is None:
            raise ModelCompileError(
                f"residency checkpoint {checkpoint_id!r} has an unknown component"
            )
        node_order = list(nodes)
        if (
            checkpoint["after_node_id"] not in nodes
            or checkpoint["resume_node_id"] not in nodes
            or node_order.index(checkpoint["after_node_id"])
            >= node_order.index(checkpoint["resume_node_id"])
        ):
            raise ModelCompileError(
                f"residency checkpoint {checkpoint_id!r} has an invalid physical resume boundary"
            )
        for selector_id in selector_ids:
            selector = selector_by_id.get(selector_id)
            if selector is None:
                raise ModelCompileError(
                    f"residency checkpoint {checkpoint_id!r} has an unknown selector"
                )
            if (
                selector["execution_scope"] != checkpoint["execution_scope"]
                or selector["component_id"] != checkpoint["component_id"]
                or selector["node_id"] != checkpoint["after_node_id"]
            ):
                raise ModelCompileError(
                    f"residency checkpoint {checkpoint_id!r} crosses an execution boundary"
                )
            selector_owners[selector_id] += 1
        if checkpoint_identity(checkpoint) != checkpoint_id:
            raise ModelCompileError(
                f"residency checkpoint {checkpoint_id!r} identity does not match its semantics"
            )
    if selector_owners != Counter({selector_id: 1 for selector_id in selector_by_id}):
        raise ModelCompileError(
            "every resource selector must belong to exactly one physical checkpoint"
        )


def compiled_parameter_bindings(manifest: Json) -> dict[str, list[Json]]:
    by_tensor: dict[str, list[Json]] = defaultdict(list)

    def collect(scope: str, graph: Any) -> None:
        graph = _require_object(graph, f"{scope} circuit graph")
        components = _require_object_list(
            graph.get("components"), f"{scope} circuit graph components"
        )
        for component in components:
            component_id = _require_non_empty_string(
                component.get("component_id"), "component id"
            )
            circuit = _require_object(component.get("circuit"), "component circuit")
            parameters = _require_object(
                component.get("params"), "component parameters"
            ).get("refs")
            parameters = _require_object(parameters, "component parameter references")
            nodes = _require_object_list(circuit.get("nodes"), "component nodes")
            for node in nodes:
                node_id = _require_non_empty_string(node.get("id"), "node id")
                node_params = node.get("params", [])
                if not isinstance(node_params, list):
                    raise ModelCompileError(
                        f"component {component_id!r} node {node_id!r} parameters are invalid"
                    )
                for parameter_id in node_params:
                    parameter_id = _require_non_empty_string(
                        parameter_id, "node parameter id"
                    )
                    parameter = parameters.get(parameter_id)
                    if not isinstance(parameter, dict):
                        raise ModelCompileError(
                            f"component {component_id!r} node {node_id!r} references unknown parameter {parameter_id!r}"
                        )
                    tensor_name = _require_non_empty_string(
                        parameter.get("tensor"), "parameter tensor"
                    )
                    by_tensor[tensor_name].append(
                        {
                            "execution_scope": scope,
                            "component_id": component_id,
                            "node_id": node_id,
                            "parameter_id": parameter_id,
                        }
                    )

    collect("target", manifest.get("circuit_graph"))
    decoders = manifest.get("speculative_decoders", [])
    if not isinstance(decoders, list):
        raise ModelCompileError("compiled speculative decoders are invalid")
    for decoder in decoders:
        decoder = _require_object(decoder, "compiled speculative decoder")
        decoder_id = _require_non_empty_string(
            decoder.get("id"), "compiled speculative decoder id"
        )
        collect(f"draft:{decoder_id}", decoder.get("circuit_graph"))
    for uses in by_tensor.values():
        uses.sort(key=_binding_key)
    return dict(by_tensor)


def _compiled_semantics(
    manifest: Json,
) -> tuple[
    set[tuple[str, str, str, str]],
    dict[tuple[str, str], dict[str, Json]],
]:
    parameter_semantics: set[tuple[str, str, str, str]] = set()
    component_nodes: dict[tuple[str, str], dict[str, Json]] = {}

    def collect(scope: str, graph: Any) -> None:
        graph = _require_object(graph, f"{scope} circuit graph")
        for component in _require_object_list(
            graph.get("components"), f"{scope} circuit graph components"
        ):
            component_id = _require_non_empty_string(
                component.get("component_id"), "component id"
            )
            circuit = _require_object(component.get("circuit"), "component circuit")
            refs = _require_object(
                _require_object(
                    component.get("params"), "component parameters"
                ).get("refs"),
                "component parameter references",
            )
            nodes = _require_object_list(circuit.get("nodes"), "component nodes")
            nodes_by_id: dict[str, Json] = {}
            for node in nodes:
                node_id = _require_non_empty_string(node.get("id"), "node id")
                if node_id in nodes_by_id:
                    raise ModelCompileError(
                        f"component {component_id!r} repeats node {node_id!r}"
                    )
                nodes_by_id[node_id] = node
                node_params = node.get("params", [])
                if not isinstance(node_params, list):
                    raise ModelCompileError(
                        f"component {component_id!r} node {node_id!r} parameters are invalid"
                    )
                for parameter_id in node_params:
                    parameter_id = _require_non_empty_string(
                        parameter_id, "node parameter id"
                    )
                    if parameter_id not in refs:
                        raise ModelCompileError(
                            f"component {component_id!r} node {node_id!r} references an unknown parameter"
                        )
                    parameter_semantics.add(
                        (scope, component_id, node_id, parameter_id)
                    )
            component_nodes[(scope, component_id)] = nodes_by_id

    collect("target", manifest.get("circuit_graph"))
    decoders = manifest.get("speculative_decoders", [])
    if not isinstance(decoders, list):
        raise ModelCompileError("compiled speculative decoders are invalid")
    for decoder in decoders:
        decoder = _require_object(decoder, "compiled speculative decoder")
        decoder_id = _require_non_empty_string(
            decoder.get("id"), "compiled speculative decoder id"
        )
        collect(f"draft:{decoder_id}", decoder.get("circuit_graph"))
    return parameter_semantics, component_nodes


def _range_location_key(resource: Json) -> tuple[str, int, int]:
    byte_range = resource["ranges"][0]
    return (
        byte_range["artifact_path"],
        byte_range["byte_offset"],
        byte_range["byte_count"],
    )


def _binding_key(binding: Json) -> tuple[str, str, str, str, str]:
    mapping = binding.get("mapping")
    if not isinstance(mapping, dict):
        mapping = {}
    mapping_key = "|".join(
        str(mapping.get(field, ""))
        for field in (
            "kind",
            "atomic_group_id",
            "resource_id",
            "selection_signal",
            "selector_index",
            "parameter_slot",
            "partition_template_id",
            "resource_identity_seed",
        )
    )
    return (
        str(binding["execution_scope"]),
        str(binding["component_id"]),
        str(binding["node_id"]),
        str(binding["parameter_id"]),
        mapping_key,
    )


def _validate_compatibility(value: Any, label: str) -> Json:
    compatibility = _require_object(value, label)
    _require_exact_fields(compatibility, _COMPATIBILITY_FIELDS, label)
    if compatibility["device_api"] != "vulkan":
        raise ModelCompileError(f"{label} has unsupported device API")
    if compatibility["storage_class"] != "storage_buffer":
        raise ModelCompileError(f"{label} has unsupported storage class")
    if compatibility["read_only"] is not True:
        raise ModelCompileError(f"{label} must describe immutable resources")
    features = compatibility["required_features"]
    if (
        not isinstance(features, list)
        or any(not isinstance(item, str) or not item for item in features)
        or features != sorted(set(features))
    ):
        raise ModelCompileError(f"{label} required features must be unique and sorted")
    return compatibility


def _validate_integrity(value: Any, label: str) -> Json:
    integrity = _require_object(value, label)
    _require_exact_fields(integrity, _INTEGRITY_FIELDS, label)
    if integrity["algorithm"] != SHA256_INTEGRITY_ALGORITHM:
        raise ModelCompileError(f"{label} has unsupported algorithm")
    if not _is_lower_hex_sha256(integrity["digest"]):
        raise ModelCompileError(f"{label} has invalid SHA-256")
    return integrity


def _reject_dependency_cycles(graph: dict[str, list[str]], label: str) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ModelCompileError(f"{label} dependencies contain a cycle")
        if node in visited:
            return
        visiting.add(node)
        for dependency in graph[node]:
            visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        visit(node)


def _require_exact_fields(value: Json, fields: frozenset[str], label: str) -> None:
    actual = frozenset(value)
    if actual != fields:
        missing = sorted(fields.difference(actual))
        unknown = sorted(actual.difference(fields))
        raise ModelCompileError(
            f"{label} fields are invalid; missing={missing}, unknown={unknown}"
        )


def _require_object(value: Any, label: str) -> Json:
    if not isinstance(value, dict):
        raise ModelCompileError(f"{label} must be an object")
    return value


def _require_object_list(value: Any, label: str) -> list[Json]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ModelCompileError(f"{label} must be a list of objects")
    return value


def _require_non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelCompileError(f"{label} must be a non-empty string")
    return value


def _require_non_negative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ModelCompileError(f"{label} must be a non-negative integer")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ModelCompileError(f"{label} must be a positive integer")
    return value


def _require_power_of_two(value: Any, label: str) -> int:
    value = _require_positive_int(value, label)
    if value & (value - 1):
        raise ModelCompileError(f"{label} must be a power of two")
    return value


def _require_content_id(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or not _is_lower_hex_sha256(value.removeprefix("sha256:"))
    ):
        raise ModelCompileError(f"{label} must be a content-addressed SHA-256 id")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not _is_lower_hex_sha256(value):
        raise ModelCompileError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_content_ids(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise ModelCompileError(f"{label} must be a list")
    return [_require_content_id(item, label) for item in value]


def _require_sorted_content_ids(value: Any, label: str) -> list[str]:
    content_ids = _require_content_ids(value, label)
    if content_ids != sorted(set(content_ids)):
        raise ModelCompileError(f"{label} must be unique and sorted")
    return content_ids


def _require_sorted_unique_ids(values: list[Json], label: str) -> None:
    ids = [_require_content_id(value.get("id"), f"{label} id") for value in values]
    if ids != sorted(set(ids)):
        raise ModelCompileError(f"{label} must have unique sorted ids")


def _require_safe_relative_path(value: Any, label: str) -> str:
    value = _require_non_empty_string(value, label)
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ModelCompileError(f"{label} must stay inside the compiled package")
    return value


def _is_lower_hex_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _largest_power_of_two_divisor(value: int) -> int:
    if value == 0:
        return 1
    return min(value & -value, 4096)
