from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.contracts import stable_contract_id
from nerve.representation_optimizer.providers.source_artifacts import (
    SourceTensorArtifact,
)
from nerve.representation_optimizer.providers.types import ProviderContext


GROUP_SIZE = 32


@dataclass(frozen=True)
class GroupScaledInt4Opportunity:
    scope_id: str
    source_contract_digest: str
    component_id: str
    node_id: str
    evidence_ids: tuple[str, ...]
    source_artifact_refs: tuple[str, ...]
    manifest_ref: str
    circuit_ref: str
    tensor_index_ref: str
    source_weight_ref_id: str
    source_weight_ref: Json
    source_weight: SourceTensorArtifact
    input_features: int
    output_features: int
    group_size: int
    max_context_activations: int
    compiler_device: Json
    performance_signature: str

    @property
    def region_id(self) -> str:
        return stable_contract_id(
            "group_scaled_int4_region",
            self.component_id,
            self.node_id,
        )

    @property
    def candidate_weight_name(self) -> str:
        return f"nerve.optimizer.group_scaled_int4.{self.region_id}.weight"

    @property
    def candidate_scale_name(self) -> str:
        return f"nerve.optimizer.group_scaled_int4.{self.region_id}.scales"

    @property
    def replacement_weight_ref_id(self) -> str:
        return f"{self.source_weight_ref_id}__{self.region_id}"

    @property
    def replacement_scale_ref_id(self) -> str:
        return f"{self.replacement_weight_ref_id}_scales"

    @property
    def packed_shape(self) -> tuple[int, int]:
        return (self.output_features, self.input_features // 8)

    @property
    def scale_shape(self) -> tuple[int, int]:
        return (self.output_features, self.input_features // self.group_size)


@dataclass(frozen=True)
class DiscoveryResult:
    opportunities: tuple[GroupScaledInt4Opportunity, ...]
    reasons: tuple[str, ...]
    evidence_ids: tuple[str, ...] = ()


def is_group_scaled_int4_scope(scope: Json, source_contract: Json) -> bool:
    members = scope.get("members", {})
    roles = scope.get("extensions", {}).get("semantic_roles", [])
    return (
        scope.get("kind") == "operator"
        and len(members.get("component_ids", [])) == 1
        and len(members.get("source_node_ids", [])) == 1
        and roles == ["linear"]
        and source_contract.get("semantic_role") == "linear"
    )


def discover_group_scaled_int4_linears(
    context: ProviderContext,
) -> tuple[GroupScaledInt4Opportunity, ...]:
    key = "group_scaled_int4.v1:" + ",".join(context.scope_ids)
    result = context.memoized(key, lambda: _discover(context).opportunities)
    return result  # type: ignore[return-value]


def discovery_result(context: ProviderContext) -> DiscoveryResult:
    key = "group_scaled_int4.result.v1:" + ",".join(context.scope_ids)
    result = context.memoized(key, lambda: _discover(context))
    return result  # type: ignore[return-value]


def source_inputs(
    context: ProviderContext,
    opportunity: GroupScaledInt4Opportunity,
) -> list[Json]:
    artifacts = {
        source["path"]: source
        for source in opportunity.source_weight.source_inputs
    }
    for path in opportunity.source_artifact_refs:
        artifact = context.source_artifacts.resolve_path(path)
        artifacts[artifact.path] = artifact.source_input()
    return [artifacts[path] for path in sorted(artifacts)]


def _discover(context: ProviderContext) -> DiscoveryResult:
    context.checkpoint()
    capability_reason = _capability_reason(context.hardware_profile)
    if capability_reason is not None:
        return DiscoveryResult((), (capability_reason,))

    contracts = {
        str(contract["scope_id"]): contract for contract in context.source_contracts
    }
    eligible = [
        (scope, contracts[str(scope["scope_id"])])
        for scope in context.scopes
        if is_group_scaled_int4_scope(
            scope,
            contracts[str(scope["scope_id"])],
        )
    ]
    if not eligible:
        return DiscoveryResult((), ("no plain linear operator scopes",))

    evidence_by_scope = {
        str(scope["scope_id"]): tuple(
            sorted(
                str(record["evidence_id"])
                for record in context.evidence
                if record["scope_id"] == scope["scope_id"]
                and any(
                    claim.get("status") == "supported"
                    for claim in record.get("claims", [])
                )
            )
        )
        for scope, _contract in eligible
    }
    resolver = context.source_artifacts
    manifest_ref = "vulkan_resident_package.json"
    manifest = _json_object(resolver.read_path(manifest_ref), manifest_ref)
    tensor_index_ref = str(manifest.get("tensor_index_path", ""))
    if tensor_index_ref != "tensors.json":
        return DiscoveryResult((), ("compiled package has no canonical tensor index",))
    components = _unique_by_id(
        manifest.get("circuit_graph", {}).get("components"),
        "component_id",
        "resident components",
    )
    executions = _unique_by_id(
        manifest.get("component_executions"),
        "component_id",
        "component executions",
    )
    compiler_device = _compiler_device(context.hardware_profile)
    max_context = _positive_integer(
        manifest.get("max_context_activations"),
        "compiled package max_context_activations",
    )

    opportunities = []
    rejected = []
    for scope, contract in eligible:
        context.checkpoint()
        try:
            opportunity = _scope_opportunity(
                scope=scope,
                contract=contract,
                evidence_ids=evidence_by_scope[str(scope["scope_id"])],
                resolver=resolver,
                components=components,
                executions=executions,
                manifest_ref=manifest_ref,
                tensor_index_ref=tensor_index_ref,
                compiler_device=compiler_device,
                max_context_activations=max_context,
            )
        except ModelCompileError as error:
            rejected.append(str(error))
            continue
        except (KeyError, TypeError, ValueError) as error:
            rejected.append(
                f"scope {scope.get('scope_id')!r} has malformed linear "
                f"metadata: {error}"
            )
            continue
        if opportunity is not None:
            opportunities.append(opportunity)

    if not opportunities:
        return DiscoveryResult(
            (),
            tuple(rejected[:8])
            or ("no private BF16 linear parameter can use group-scaled INT4",),
        )
    opportunities.sort(key=lambda item: (item.component_id, item.node_id))
    return DiscoveryResult(
        tuple(opportunities),
        (
            f"discovered {len(opportunities)} private BF16 linear regions with "
            f"group-{GROUP_SIZE} INT4 execution contracts",
        ),
        tuple(
            sorted(
                {
                    evidence_id
                    for opportunity in opportunities
                    for evidence_id in opportunity.evidence_ids
                }
            )
        ),
    )


def _scope_opportunity(
    *,
    scope: Json,
    contract: Json,
    evidence_ids: tuple[str, ...],
    resolver,
    components: dict[str, Json],
    executions: dict[str, Json],
    manifest_ref: str,
    tensor_index_ref: str,
    compiler_device: Json,
    max_context_activations: int,
) -> GroupScaledInt4Opportunity | None:
    if not evidence_ids:
        return None
    component_id = str(scope["members"]["component_ids"][0])
    qualified_node_id = str(scope["members"]["source_node_ids"][0])
    prefix = f"{component_id}/"
    if not qualified_node_id.startswith(prefix):
        raise ModelCompileError(
            f"component {component_id!r} linear scope crosses components"
        )
    node_id = qualified_node_id.removeprefix(prefix)
    component = components.get(component_id)
    execution = executions.get(component_id)
    if component is None or execution is None:
        raise ModelCompileError(
            f"component {component_id!r} has no complete resident execution"
        )
    compiled_nodes = _unique_by_id(
        component.get("circuit", {}).get("nodes"),
        "id",
        f"component {component_id!r} nodes",
    )
    kernels = _unique_by_id(
        execution.get("kernels"),
        "node_id",
        f"component {component_id!r} kernels",
    )
    compiled_node = compiled_nodes.get(node_id)
    source_kernel = kernels.get(node_id)
    if compiled_node is None or source_kernel is None:
        return None

    circuit_refs = {
        str(path)
        for path in contract.get("exact_reference", {}).get("artifact_refs", [])
        if isinstance(path, str) and path.endswith("/circuit.json")
    }
    if len(circuit_refs) != 1:
        raise ModelCompileError(
            f"component {component_id!r} scope {node_id!r} has no unique exact circuit"
        )
    circuit_ref = next(iter(circuit_refs))
    source_circuit = _json_object(resolver.read_path(circuit_ref), circuit_ref)
    if source_circuit.get("source", {}).get("component_id") != component_id:
        raise ModelCompileError(
            f"component {component_id!r} exact circuit identity drifted"
        )
    source_nodes = _unique_by_id(
        source_circuit.get("nodes"),
        "id",
        f"component {component_id!r} exact nodes",
    )
    source_node = source_nodes.get(node_id)
    if source_node is None:
        return None
    if not _plain_linear(source_node, compiled_node):
        return None

    source_weight_ref_id = str(source_node["params"][0])
    source_refs = source_circuit.get("parameters", {}).get("refs", {})
    compiled_refs = component.get("circuit", {}).get("parameters", {}).get("refs", {})
    source_weight_ref = source_refs.get(source_weight_ref_id)
    if (
        not isinstance(source_weight_ref, dict)
        or compiled_refs.get(source_weight_ref_id) != source_weight_ref
        or sum(
            source_weight_ref_id in node.get("params", [])
            for node in component["circuit"]["nodes"]
        )
        != 1
    ):
        return None
    tensor_name = source_weight_ref.get("tensor")
    if not isinstance(tensor_name, str) or not tensor_name:
        return None
    weight = resolver.resolve_tensor(tensor_name)
    shape = weight.metadata.get("shape")
    if (
        weight.metadata.get("dtype") != "BF16"
        or weight.metadata.get("layout") != "row_major"
        or not isinstance(shape, list)
        or len(shape) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in shape
        )
    ):
        return None
    output_features, input_features = map(int, shape)
    if (
        input_features % GROUP_SIZE
        or output_features % 2
        or weight.payload_byte_count != output_features * input_features * 2
    ):
        return None
    source_artifact_refs = {
        manifest_ref,
        tensor_index_ref,
        circuit_ref,
        *(
            str(path)
            for path in contract["exact_reference"]["artifact_refs"]
            if isinstance(path, str)
        ),
    }
    return GroupScaledInt4Opportunity(
        scope_id=str(scope["scope_id"]),
        source_contract_digest=str(contract["contract_digest"]),
        component_id=component_id,
        node_id=node_id,
        evidence_ids=evidence_ids,
        source_artifact_refs=tuple(sorted(source_artifact_refs)),
        manifest_ref=manifest_ref,
        circuit_ref=circuit_ref,
        tensor_index_ref=tensor_index_ref,
        source_weight_ref_id=source_weight_ref_id,
        source_weight_ref=deepcopy(source_weight_ref),
        source_weight=weight,
        input_features=input_features,
        output_features=output_features,
        group_size=GROUP_SIZE,
        max_context_activations=max_context_activations,
        compiler_device=deepcopy(compiler_device),
        performance_signature=stable_contract_id(
            "group_scaled_int4_performance_class",
            {
                "input_features": input_features,
                "output_features": output_features,
                "group_size": GROUP_SIZE,
                "source_dtype": "BF16",
                "candidate_storage": "I32_ROW_MAJOR_PACKED_SIGNED_INT4",
                "scale_dtype": "BF16",
            },
        ),
    )


def _plain_linear(source_node: Json, compiled_node: Json) -> bool:
    attrs = compiled_node.get("attrs", {})
    return (
        source_node.get("op") == "linear"
        and compiled_node.get("op") == "linear"
        and len(source_node.get("inputs", [])) == 1
        and len(source_node.get("outputs", [])) == 1
        and len(source_node.get("params", [])) == 1
        and not source_node.get("state_reads")
        and not source_node.get("state_writes")
        and compiled_node.get("inputs") == source_node.get("inputs")
        and compiled_node.get("outputs") == source_node.get("outputs")
        and compiled_node.get("params") == source_node.get("params")
        and attrs.get("output_element_bytes") == [2]
        and "physical_input_contract" not in attrs
    )


def _capability_reason(profile: Json) -> str | None:
    if (
        profile.get("hardware_identity", {}).get("device_kind") != "gpu"
        or profile.get("provenance", {}).get("api") != "vulkan"
    ):
        return "group-scaled INT4 linear execution requires a Vulkan GPU"
    try:
        device = _compiler_device(profile)
    except ModelCompileError:
        return "target has no complete Vulkan compiler capability contract"
    max_invocations = device.get("max_compute_work_group_invocations")
    max_size_x = device.get("max_compute_work_group_size_x")
    subgroup_size = device.get("subgroup_size")
    if (
        isinstance(max_invocations, bool)
        or not isinstance(max_invocations, int)
        or max_invocations < 64
        or isinstance(max_size_x, bool)
        or not isinstance(max_size_x, int)
        or max_size_x < 64
        or "basic" not in device.get("subgroup_operations", [])
        or "arithmetic" not in device.get("subgroup_operations", [])
        or isinstance(subgroup_size, bool)
        or not isinstance(subgroup_size, int)
        or subgroup_size <= 0
        or 64 % subgroup_size
        or device.get("subgroup_compute_supported") is not True
    ):
        return "target cannot execute the group-scaled INT4 reduction kernel"
    return None


def _compiler_device(profile: Json) -> Json:
    value = profile.get("capability_extensions", {}).get(
        "vulkan_compiler_capabilities"
    )
    if not isinstance(value, dict):
        raise ModelCompileError("hardware profile has no Vulkan compiler capabilities")
    return value


def _json_object(payload: bytes, label: str) -> Json:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelCompileError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ModelCompileError(f"{label} must contain a JSON object")
    return value


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelCompileError(f"{label} must be a positive integer")
    return value


def _unique_by_id(records: object, field: str, label: str) -> dict[str, Json]:
    if not isinstance(records, list):
        raise ModelCompileError(f"{label} are missing")
    indexed = {
        str(record[field]): record
        for record in records
        if isinstance(record, dict)
        and isinstance(record.get(field), str)
        and record[field]
    }
    if len(indexed) != len(records):
        raise ModelCompileError(f"{label} contain invalid or duplicate identities")
    return indexed


__all__ = [
    "GROUP_SIZE",
    "DiscoveryResult",
    "GroupScaledInt4Opportunity",
    "discover_group_scaled_int4_linears",
    "discovery_result",
    "is_group_scaled_int4_scope",
    "source_inputs",
]
