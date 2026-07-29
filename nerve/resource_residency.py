from __future__ import annotations

import json
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from nerve.compilation import Json, ModelCompileError


RESOURCE_RESIDENCY_SCHEMA = "nerve.compiled_resource_residency.v1"
RESOURCE_IDENTITY_ALGORITHM = "nerve.resource_identity_sha256.v1"
RESOURCE_STATE_MACHINE_SCHEMA = "nerve.resource_residency_state_machine.v1"
SUPPORTED_RESIDENCY_POLICIES = ("demand_retained", "eager")
RESOURCE_LIFETIMES = frozenset(("always_resident", "dynamic"))
RESOURCE_STATES = frozenset(("absent", "requested", "loading", "resident", "failed"))
SHA256_INTEGRITY_ALGORITHM = "sha256"

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
        "atomic_group_id",
    )
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
        "mapping",
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

    tensor_bindings = _compiled_parameter_bindings(manifest)
    tensors = tensor_index.get("tensors")
    if not isinstance(tensors, dict):
        raise ModelCompileError("tensor index has no tensor mapping")

    compatibility = {
        "device_api": "vulkan",
        "storage_class": "storage_buffer",
        "read_only": True,
        "required_features": [],
    }
    resources_by_id: dict[str, Json] = {}
    source_headers = {
        record["path"]: record["safetensors_header_bytes"]
        for record in tensor_index.get("source", {}).get("weights_files", [])
        if isinstance(record, dict)
        and isinstance(record.get("path"), str)
        and isinstance(record.get("safetensors_header_bytes"), int)
        and not isinstance(record.get("safetensors_header_bytes"), bool)
    }

    for tensor_name in sorted(tensor_bindings):
        metadata = tensors.get(tensor_name)
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
            not isinstance(offsets, list)
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
        source_path = package_dir / source_file
        try:
            artifact_bytes = source_path.stat().st_size
        except OSError as error:
            raise ModelCompileError(
                f"compiled tensor {tensor_name!r} source cannot be inspected: {error}"
            ) from error
        if absolute_offset + byte_count > artifact_bytes:
            raise ModelCompileError(
                f"compiled tensor {tensor_name!r} range exceeds {source_file!r}"
            )
        byte_range = {
            "artifact_path": source_file,
            "byte_offset": absolute_offset,
            "byte_count": byte_count,
            "alignment_bytes": _largest_power_of_two_divisor(absolute_offset),
            "integrity": {
                "algorithm": SHA256_INTEGRITY_ALGORITHM,
                "digest": digest,
            },
        }
        resource = {
            "id": "",
            "lifetime": "always_resident",
            "ranges": [byte_range],
            "dependencies": [],
            "compatibility": compatibility,
        }
        resource["id"] = resource_identity(resource)
        resource_id = resource["id"]
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
                    "atomic_group_id": eager_spine_group["id"],
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
    template_by_id = _validate_partition_templates(package_dir, templates, group_by_id)
    parameter_semantics, component_nodes = _compiled_semantics(manifest)
    _validate_bindings(bindings, group_by_id, parameter_semantics)
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
) -> dict[str, Json]:
    _require_sorted_unique_ids(templates, "partition templates")
    template_by_id: dict[str, Json] = {}
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
        _require_content_id(
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
                _validate_range_template(package_dir, byte_range, partition_count)
        if member_seeds != sorted(set(member_seeds)):
            raise ModelCompileError(
                f"partition template {template_id!r} member seeds must be unique and sorted"
            )
        if partition_template_identity(template) != template_id:
            raise ModelCompileError(
                f"partition template {template_id!r} identity does not match its contract"
            )
        template_by_id[template_id] = template
    return template_by_id


def _validate_range_template(
    package_dir: Path, byte_range: Json, partition_count: int
) -> None:
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
    if digest_stride < 32:
        raise ModelCompileError("partition digest stride cannot hold SHA-256")
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


def _validate_bindings(
    bindings: list[Json],
    group_by_id: dict[str, Json],
    parameter_semantics: set[tuple[str, str, str, str]],
) -> None:
    keys = []
    bound_semantics = []
    for binding in bindings:
        _require_exact_fields(binding, _BINDING_FIELDS, "resource binding")
        for field in ("execution_scope", "component_id", "node_id", "parameter_id"):
            _require_non_empty_string(binding[field], f"resource binding {field}")
        group_id = _require_content_id(
            binding["atomic_group_id"], "resource binding atomic group id"
        )
        if group_id not in group_by_id:
            raise ModelCompileError(
                f"resource binding references unknown atomic group {group_id!r}"
            )
        keys.append(_binding_key(binding))
        bound_semantics.append(_binding_key(binding)[:4])
    if keys != sorted(set(keys)):
        raise ModelCompileError("resource bindings must be unique and sorted")
    if Counter(bound_semantics) != Counter(parameter_semantics):
        raise ModelCompileError(
            "resource bindings must exactly cover compiled parameter semantics"
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
            or selection_domain.get("id") != selector["domain_id"]
            or selection_domain.get("resource_count") != selector["resource_count"]
        ):
            raise ModelCompileError(
                f"selector {selector_id!r} does not match compiled node semantics"
            )
        resource_count = _require_positive_int(
            selector["resource_count"], "selector resource count"
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
            if template_id in selected_templates:
                raise ModelCompileError(
                    f"partition template {template_id!r} has multiple selectors"
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
            "every partition template must be mapped by exactly one selector"
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


def _compiled_parameter_bindings(manifest: Json) -> dict[str, list[Json]]:
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
    return (
        str(binding["execution_scope"]),
        str(binding["component_id"]),
        str(binding["node_id"]),
        str(binding["parameter_id"]),
        str(binding.get("atomic_group_id", "")),
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
