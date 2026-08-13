from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.contracts import stable_contract_id
from nerve.representation_optimizer.providers.types import ProviderContext


_HEAD_GROUPS = (2, 4, 8)
_SOURCE_SHADER = re.compile(
    r"shaders/indexed_sparse_attention_main"
    r"(?:_score_pipeline|_tile_overlap)?_bf16_"
    r"(?P<suffix>q(?P<query_heads>\d+)_kv(?P<kv_heads>\d+)_"
    r"d(?P<head_width>\d+)_w(?P<window>\d+)_"
    r"r(?P<compression_ratio>\d+)_k(?P<max_indices>\d+)_"
    r"scale[0-9eE+.-]+__sc\d+)\.spv"
)


@dataclass(frozen=True)
class AttentionHeadGroupingOpportunity:
    scope_id: str
    source_contract_digest: str
    component_id: str
    source_node_id: str
    physical_node_id: str
    terminal_node_id: str
    evidence_ids: tuple[str, ...]
    source_artifact_refs: tuple[str, ...]
    manifest_ref: str
    circuit_ref: str
    tensor_index_ref: str
    query_heads: int
    key_value_heads: int
    head_width: int
    local_window: int
    compression_ratio: int
    max_compressed_indices: int
    head_group: int
    shader_suffix: str
    max_context_activations: int
    compiler_device: Json
    performance_signature: str

    @property
    def decode_shader_file(self) -> str:
        return (
            "indexed_sparse_attention_main_head_grouped_tile_overlap_"
            f"hg{self.head_group}_bf16_{self.shader_suffix}.comp"
        )

    @property
    def prefill_shader_file(self) -> str:
        batch_suffix = re.sub(r"__sc(\d+)$", r"__pbc\1", self.shader_suffix)
        return (
            "indexed_sparse_attention_main_head_grouped_tile_overlap_"
            f"hg{self.head_group}_temporal_parallel_bf16_"
            f"{batch_suffix}.comp"
        )


@dataclass(frozen=True)
class DiscoveryResult:
    opportunities: tuple[AttentionHeadGroupingOpportunity, ...]
    reasons: tuple[str, ...]
    evidence_ids: tuple[str, ...] = ()


def is_attention_operator_scope(scope: Json, source_contract: Json) -> bool:
    members = scope.get("members", {})
    roles = scope.get("extensions", {}).get("semantic_roles", [])
    return (
        scope.get("kind") == "semantic_module"
        and len(members.get("component_ids", [])) == 1
        and len(members.get("source_node_ids", [])) == 1
        and "operator" in scope.get("extensions", {}).get("classifications", [])
        and (
            "indexed_sparse_attention" in roles
            or "indexed_sparse_attention"
            in str(source_contract.get("semantic_role", ""))
        )
    )


def discover_attention_head_groupings(
    context: ProviderContext,
) -> tuple[AttentionHeadGroupingOpportunity, ...]:
    key = "attention_head_grouping.v1:" + ",".join(context.scope_ids)
    return context.memoized(key, lambda: _discover(context).opportunities)  # type: ignore[return-value]


def discovery_result(context: ProviderContext) -> DiscoveryResult:
    key = "attention_head_grouping.result.v1:" + ",".join(context.scope_ids)
    return context.memoized(key, lambda: _discover(context))  # type: ignore[return-value]


def source_inputs(
    context: ProviderContext,
    opportunity: AttentionHeadGroupingOpportunity,
) -> list[Json]:
    return [
        context.source_artifacts.resolve_path(path).source_input()
        for path in opportunity.source_artifact_refs
    ]


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
        if is_attention_operator_scope(
            scope,
            contracts[str(scope["scope_id"])],
        )
    ]
    if not eligible:
        return DiscoveryResult((), ("no standalone indexed-attention scopes",))

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
    tensor_index_ref = "tensors.json"
    manifest = _json_object(resolver.read_path(manifest_ref), manifest_ref)
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
    opportunities = []
    rejected = []
    for scope, contract in sorted(eligible, key=lambda item: str(item[0]["scope_id"])):
        context.checkpoint()
        supported_evidence = evidence_by_scope[str(scope["scope_id"])]
        if not supported_evidence:
            continue
        try:
            opportunities.extend(
                _scope_opportunities(
                    scope=scope,
                    contract=contract,
                    evidence_ids=supported_evidence,
                    resolver=resolver,
                    manifest=manifest,
                    components=components,
                    executions=executions,
                    manifest_ref=manifest_ref,
                    tensor_index_ref=tensor_index_ref,
                    compiler_device=compiler_device,
                )
            )
        except ModelCompileError as error:
            rejected.append(str(error))
        except (KeyError, TypeError, ValueError) as error:
            rejected.append(
                f"scope {scope.get('scope_id')!r} has malformed attention metadata: {error}"
            )
    evidence_ids = tuple(
        sorted(
            {
                evidence_id
                for opportunity in opportunities
                for evidence_id in opportunity.evidence_ids
            }
        )
    )
    if opportunities:
        return DiscoveryResult(
            tuple(opportunities),
            (
                f"discovered {len(opportunities)} exact grouped-head attention alternatives",
            ),
            evidence_ids,
        )
    return DiscoveryResult(
        (),
        tuple(rejected[:8]) or ("no structurally valid grouped-head attention",),
    )


def _scope_opportunities(
    *,
    scope: Json,
    contract: Json,
    evidence_ids: tuple[str, ...],
    resolver,
    manifest: Json,
    components: dict[str, Json],
    executions: dict[str, Json],
    manifest_ref: str,
    tensor_index_ref: str,
    compiler_device: Json,
) -> tuple[AttentionHeadGroupingOpportunity, ...]:
    component_id = str(scope["members"]["component_ids"][0])
    qualified_source_id = str(scope["members"]["source_node_ids"][0])
    prefix = f"{component_id}/"
    if not qualified_source_id.startswith(prefix):
        raise ModelCompileError(
            f"attention scope {scope['scope_id']!r} crosses its component boundary"
        )
    source_node_id = qualified_source_id.removeprefix(prefix)
    component = components.get(component_id)
    execution = executions.get(component_id)
    if component is None or execution is None:
        raise ModelCompileError(
            f"component {component_id!r} has no complete resident execution"
        )
    circuit_refs = {
        str(path)
        for path in contract.get("exact_reference", {}).get("artifact_refs", [])
        if isinstance(path, str) and path.endswith("/circuit.json")
    }
    matching_circuits = []
    for path in sorted(circuit_refs):
        circuit = _json_object(resolver.read_path(path), path)
        if circuit.get("source", {}).get("component_id") == component_id:
            matching_circuits.append((path, circuit))
    if len(matching_circuits) != 1:
        raise ModelCompileError(
            f"component {component_id!r} has no unique exact attention circuit"
        )
    circuit_ref, source_circuit = matching_circuits[0]
    source_nodes = _unique_by_id(
        source_circuit.get("nodes"),
        "id",
        f"component {component_id!r} exact nodes",
    )
    compiled_nodes = _unique_by_id(
        component.get("circuit", {}).get("nodes"),
        "id",
        f"component {component_id!r} compiled nodes",
    )
    kernels = _unique_by_id(
        execution.get("kernels"),
        "node_id",
        f"component {component_id!r} kernels",
    )
    source_node = source_nodes.get(source_node_id)
    compiled_node = compiled_nodes.get(source_node_id)
    kernel = kernels.get(source_node_id)
    if (
        source_node is None
        or source_node.get("op") != "indexed_sparse_attention"
        or compiled_node is None
        or compiled_node.get("op") != "indexed_sparse_attention"
        or kernel is None
    ):
        raise ModelCompileError(
            f"component {component_id!r} attention source did not lower one-to-one"
        )
    shader_match = _SOURCE_SHADER.fullmatch(str(kernel.get("shader_path", "")))
    attrs = compiled_node.get("attrs", {})
    if shader_match is None or not isinstance(attrs, dict):
        raise ModelCompileError(
            f"component {component_id!r} attention has no supported source shader"
        )
    query_heads = int(shader_match.group("query_heads"))
    key_value_heads = int(shader_match.group("kv_heads"))
    head_width = int(shader_match.group("head_width"))
    local_window = int(shader_match.group("window"))
    compression_ratio = int(shader_match.group("compression_ratio"))
    max_compressed_indices = int(shader_match.group("max_indices"))
    if (
        query_heads != attrs.get("query_heads")
        or key_value_heads != attrs.get("key_value_heads")
        or head_width != attrs.get("head_width")
        or local_window != attrs.get("window_size")
        or key_value_heads != 1
        or head_width != 512
        or query_heads <= key_value_heads
        or (compression_ratio == 0) != (max_compressed_indices == 0)
    ):
        raise ModelCompileError(
            f"component {component_id!r} attention geometry is not group-shareable"
        )
    batch = kernel.get("batch_implementations")
    if (
        not isinstance(batch, list)
        or len(batch) != 1
        or batch[0].get("execution_domain") != "prefill"
        or batch[0].get("causal_sequence_compatible") is not True
        or len(batch[0].get("stages", [])) != 1
    ):
        raise ModelCompileError(
            f"component {component_id!r} attention has no unique causal prefill path"
        )
    terminal_node_id = str(component["circuit"]["nodes"][-1]["id"])
    if terminal_node_id not in kernels:
        raise ModelCompileError(
            f"component {component_id!r} has no executable terminal node"
        )
    max_context_activations = _positive_integer(
        manifest.get("max_context_activations"),
        "compiled package max_context_activations",
    )
    source_refs = tuple(
        sorted(
            {
                manifest_ref,
                tensor_index_ref,
                circuit_ref,
                *(
                    str(path)
                    for path in contract["exact_reference"]["artifact_refs"]
                ),
            }
        )
    )
    shader_suffix = str(shader_match.group("suffix"))
    common_performance = {
        "source_kernel": _kernel_performance_record(kernel),
        "query_heads": query_heads,
        "key_value_heads": key_value_heads,
        "head_width": head_width,
        "local_window": local_window,
        "compression_ratio": compression_ratio,
        "max_compressed_indices": max_compressed_indices,
    }
    return tuple(
        AttentionHeadGroupingOpportunity(
            scope_id=str(scope["scope_id"]),
            source_contract_digest=str(contract["contract_digest"]),
            component_id=component_id,
            source_node_id=source_node_id,
            physical_node_id=str(compiled_node["id"]),
            terminal_node_id=terminal_node_id,
            evidence_ids=tuple(sorted(evidence_ids)),
            source_artifact_refs=source_refs,
            manifest_ref=manifest_ref,
            circuit_ref=circuit_ref,
            tensor_index_ref=tensor_index_ref,
            query_heads=query_heads,
            key_value_heads=key_value_heads,
            head_width=head_width,
            local_window=local_window,
            compression_ratio=compression_ratio,
            max_compressed_indices=max_compressed_indices,
            head_group=head_group,
            shader_suffix=shader_suffix,
            max_context_activations=max_context_activations,
            compiler_device=deepcopy(compiler_device),
            performance_signature=stable_contract_id(
                "attention_head_grouping_performance_class",
                common_performance,
                head_group,
            ),
        )
        for head_group in _HEAD_GROUPS
        if query_heads % head_group == 0
    )


def _capability_reason(profile: Json) -> str | None:
    if (
        profile.get("hardware_identity", {}).get("device_kind") != "gpu"
        or profile.get("provenance", {}).get("api") != "vulkan"
    ):
        return "grouped-head attention requires a Vulkan GPU"
    try:
        device = _compiler_device(profile)
    except ModelCompileError:
        return "target has no complete Vulkan compiler capability contract"
    subgroup_size = device.get("subgroup_size")
    max_invocations = device.get("max_compute_work_group_invocations")
    max_size_x = device.get("max_compute_work_group_size_x")
    if (
        isinstance(subgroup_size, bool)
        or not isinstance(subgroup_size, int)
        or subgroup_size < 4
        or 512 % subgroup_size
        or device.get("subgroup_compute_supported") is not True
        or not {"basic", "arithmetic"}
        <= set(device.get("subgroup_operations", []))
        or isinstance(max_invocations, bool)
        or not isinstance(max_invocations, int)
        or max_invocations < 1024
        or isinstance(max_size_x, bool)
        or not isinstance(max_size_x, int)
        or max_size_x < 1024
    ):
        return "target cannot execute grouped-head attention"
    return None


def _compiler_device(profile: Json) -> Json:
    value = profile.get("capability_extensions", {}).get(
        "vulkan_compiler_capabilities"
    )
    if not isinstance(value, dict):
        raise ModelCompileError("hardware profile has no Vulkan compiler capabilities")
    return value


def _kernel_performance_record(kernel: Json) -> Json:
    return {
        "shader_path": kernel.get("shader_path"),
        "local_size_x": kernel.get("local_size_x"),
        "workgroup_count_x": kernel.get("workgroup_count_x"),
        "batch": [
            {
                "execution_domain": implementation.get("execution_domain"),
                "lane_tile_width": implementation.get("lane_tile_width"),
                "stages": [
                    {
                        "shader_path": stage.get("shader_path"),
                        "local_size_x": stage.get("local_size_x"),
                        "workgroup_count_x": stage.get("workgroup_count_x"),
                    }
                    for stage in implementation.get("stages", [])
                ],
            }
            for implementation in kernel.get("batch_implementations", [])
        ],
    }


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
        if isinstance(record, dict) and isinstance(record.get(field), str)
    }
    if len(indexed) != len(records):
        raise ModelCompileError(f"{label} contain invalid or duplicate identities")
    return indexed
