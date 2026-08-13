from __future__ import annotations

import json
from dataclasses import dataclass

from nerve.compilation import Json, ModelCompileError
from nerve.physical_representations import FP8_E8M0_PREQUANTIZATION_CONTRACT
from nerve.representation_optimizer.contracts import stable_contract_id
from nerve.representation_optimizer.providers.types import ProviderContext


_REQUIRED_FEATURES = frozenset(("shader_float8", "shader_int8"))


@dataclass(frozen=True)
class ParallelProjectionRegion:
    scope_ids: tuple[str, ...]
    source_contract_digests: tuple[str, ...]
    semantic_source_node_ids: tuple[str, ...]
    linear_node_ids: tuple[str, ...]
    quantizer_node_id: str

    @property
    def fused_node_id(self) -> str:
        return "__".join(self.linear_node_ids)

    @property
    def source_node_ids(self) -> tuple[str, ...]:
        return (self.quantizer_node_id, *self.linear_node_ids)


@dataclass(frozen=True)
class ParallelProjectionFusionOpportunity:
    component_id: str
    region: ParallelProjectionRegion
    evidence_ids: tuple[str, ...]
    source_artifact_refs: tuple[str, ...]
    manifest_ref: str
    circuit_ref: str
    tensor_index_ref: str
    hidden_size: int
    max_context_activations: int
    compiler_device: Json
    performance_signature: str

    @property
    def scope_ids(self) -> tuple[str, ...]:
        return self.region.scope_ids

    @property
    def source_contract_digests(self) -> tuple[str, ...]:
        return self.region.source_contract_digests

    @property
    def physical_node_id(self) -> str:
        return self.region.fused_node_id


@dataclass(frozen=True)
class DiscoveryResult:
    opportunities: tuple[ParallelProjectionFusionOpportunity, ...]
    reasons: tuple[str, ...]
    evidence_ids: tuple[str, ...] = ()


def is_parallel_projection_scope(scope: Json, source_contract: Json) -> bool:
    members = scope.get("members", {})
    roles = scope.get("extensions", {}).get("semantic_roles", [])
    return (
        scope.get("kind") == "operator"
        and len(members.get("component_ids", [])) == 1
        and len(members.get("source_node_ids", [])) == 1
        and roles == ["linear"]
        and source_contract.get("semantic_role") == "linear"
    )


def discover_parallel_projection_fusions(
    context: ProviderContext,
) -> tuple[ParallelProjectionFusionOpportunity, ...]:
    key = "parallel_projection_fusion.v1:" + ",".join(context.scope_ids)
    return context.memoized(
        key,
        lambda: _discover(context).opportunities,
    )  # type: ignore[return-value]


def discovery_result(context: ProviderContext) -> DiscoveryResult:
    key = "parallel_projection_fusion.result.v1:" + ",".join(context.scope_ids)
    return context.memoized(key, lambda: _discover(context))  # type: ignore[return-value]


def source_inputs(
    context: ProviderContext,
    opportunity: ParallelProjectionFusionOpportunity,
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
        if is_parallel_projection_scope(scope, contracts[str(scope["scope_id"])])
    ]
    if not eligible:
        return DiscoveryResult((), ("no exact linear operator scopes",))

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
    by_component: dict[str, list[tuple[Json, Json]]] = {}
    for scope, contract in eligible:
        component_id = str(scope["members"]["component_ids"][0])
        by_component.setdefault(component_id, []).append((scope, contract))

    opportunities = []
    rejected = []
    for component_id in sorted(by_component):
        context.checkpoint()
        try:
            opportunities.extend(
                _component_opportunities(
                    component_id=component_id,
                    scoped_contracts=tuple(by_component[component_id]),
                    evidence_by_scope=evidence_by_scope,
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
                f"component {component_id!r} has malformed projection metadata: {error}"
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
                f"discovered {len(opportunities)} capability-scoped exact "
                "shared-input projection alternatives",
            ),
            evidence_ids,
        )
    return DiscoveryResult(
        (),
        tuple(rejected[:8]) or ("no structurally valid shared-input projections",),
    )


def _component_opportunities(
    *,
    component_id: str,
    scoped_contracts: tuple[tuple[Json, Json], ...],
    evidence_by_scope: dict[str, tuple[str, ...]],
    resolver,
    manifest: Json,
    components: dict[str, Json],
    executions: dict[str, Json],
    manifest_ref: str,
    tensor_index_ref: str,
    compiler_device: Json,
) -> tuple[ParallelProjectionFusionOpportunity, ...]:
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
    scope_by_node: dict[str, tuple[Json, Json]] = {}
    circuit_refs = set()
    for scope, contract in scoped_contracts:
        source_id = str(scope["members"]["source_node_ids"][0])
        prefix = f"{component_id}/"
        if not source_id.startswith(prefix):
            raise ModelCompileError(
                f"component {component_id!r} operator scope crosses components"
            )
        node_id = source_id.removeprefix(prefix)
        if node_id in scope_by_node:
            raise ModelCompileError(
                f"component {component_id!r} has duplicate operator scope {node_id!r}"
            )
        scope_by_node[node_id] = (scope, contract)
        circuit_refs.update(
            path
            for path in contract.get("exact_reference", {}).get("artifact_refs", [])
            if isinstance(path, str) and path.endswith("/circuit.json")
        )
    if len(circuit_refs) != 1:
        raise ModelCompileError(
            f"component {component_id!r} has no unique exact circuit reference"
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
    positions = {
        str(node["id"]): index
        for index, node in enumerate(component["circuit"]["nodes"])
    }
    max_context_activations = _positive_integer(
        manifest.get("max_context_activations"),
        "compiled package max_context_activations",
    )
    hidden_size = _component_hidden_size(component)
    results = []
    for helper in compiled_nodes.values():
        helper_attrs = helper.get("attrs", {})
        consumers = helper_attrs.get("consumer_node_ids")
        if (
            helper.get("op") != "quantize_fp8_e4m3_e8m0"
            or helper_attrs.get("physical_representation_contract")
            != FP8_E8M0_PREQUANTIZATION_CONTRACT
            or not isinstance(consumers, list)
            or len(consumers) not in {2, 3}
            or len(consumers) != len(set(consumers))
            or any(not isinstance(node_id, str) for node_id in consumers)
        ):
            continue
        linears = [compiled_nodes.get(node_id) for node_id in consumers]
        if any(node is None for node in linears):
            continue
        linear_nodes = [node for node in linears if node is not None]
        if not _compatible_linear_group(helper, linear_nodes):
            continue
        semantic_ids = tuple(
            str(node_id) for node_id in helper_attrs.get("semantic_source_node_ids", [])
        )
        if (
            len(semantic_ids) != len(linear_nodes)
            or set(semantic_ids) != set(consumers)
            or any(node_id not in scope_by_node for node_id in semantic_ids)
            or any(node_id not in source_nodes for node_id in semantic_ids)
            or any(source_nodes[node_id].get("op") != "linear" for node_id in semantic_ids)
            or any(node_id not in kernels for node_id in (helper["id"], *consumers))
        ):
            continue
        scoped = [scope_by_node[node_id] for node_id in semantic_ids]
        if any(not evidence_by_scope[str(scope["scope_id"])] for scope, _ in scoped):
            continue
        linear_by_id = {str(node["id"]): node for node in linear_nodes}
        ordered = sorted(
            (
                (node_id, linear_by_id[node_id], scope_by_node[node_id])
                for node_id in semantic_ids
            ),
            key=lambda item: positions[str(item[1]["id"])],
        )
        semantic_ids = tuple(item[0] for item in ordered)
        linear_nodes = [item[1] for item in ordered]
        scoped = [item[2] for item in ordered]
        linear_ids = tuple(str(node["id"]) for node in linear_nodes)
        scope_ids = tuple(str(scope["scope_id"]) for scope, _ in scoped)
        source_digests = tuple(str(contract["contract_digest"]) for _, contract in scoped)
        evidence_ids = tuple(
            sorted(
                {
                    evidence_id
                    for scope, _contract in scoped
                    for evidence_id in evidence_by_scope[str(scope["scope_id"])]
                }
            )
        )
        region = ParallelProjectionRegion(
            scope_ids=scope_ids,
            source_contract_digests=source_digests,
            semantic_source_node_ids=semantic_ids,
            linear_node_ids=linear_ids,
            quantizer_node_id=str(helper["id"]),
        )
        source_refs = {
            manifest_ref,
            tensor_index_ref,
            circuit_ref,
            *(
                path
                for _scope, contract in scoped
                for path in contract["exact_reference"]["artifact_refs"]
            ),
        }
        results.append(
            ParallelProjectionFusionOpportunity(
                component_id=component_id,
                region=region,
                evidence_ids=evidence_ids,
                source_artifact_refs=tuple(sorted(source_refs)),
                manifest_ref=manifest_ref,
                circuit_ref=circuit_ref,
                tensor_index_ref=tensor_index_ref,
                hidden_size=hidden_size,
                max_context_activations=max_context_activations,
                compiler_device=compiler_device,
                performance_signature=stable_contract_id(
                    "parallel_projection_performance_class",
                    {
                        "hidden_size": hidden_size,
                        "helper": _kernel_performance_record(kernels[helper["id"]]),
                        "branches": [
                            {
                                "params": len(node.get("params", [])),
                                "output_element_bytes": node.get("attrs", {}).get(
                                    "output_element_bytes"
                                ),
                                "kernel": _kernel_performance_record(kernels[node["id"]]),
                            }
                            for node in linear_nodes
                        ],
                    },
                ),
            )
        )
    source_ids = [node_id for item in results for node_id in item.region.source_node_ids]
    if len(source_ids) != len(set(source_ids)):
        raise ModelCompileError(
            f"component {component_id!r} shared-input projection regions overlap"
        )
    return tuple(sorted(results, key=lambda item: positions[item.region.quantizer_node_id]))


def _compatible_linear_group(helper: Json, linears: list[Json]) -> bool:
    attrs = [node.get("attrs", {}) for node in linears]
    outputs = [node.get("outputs", []) for node in linears]
    produced = {output for values in outputs for output in values}
    return (
        len(linears) in {2, 3}
        and len(helper.get("outputs", [])) == 2
        and all(
            node.get("op") == "linear"
            and node.get("inputs") == helper.get("outputs")
            and len(node.get("outputs", [])) == 1
            and len(node.get("params", [])) == 2
            and not node.get("state_reads")
            and not node.get("state_writes")
            for node in linears
        )
        and all(
            value.get("physical_input_contract")
            == FP8_E8M0_PREQUANTIZATION_CONTRACT
            and value.get("physical_input_provider_id") == helper.get("id")
            and value.get("physical_logical_inputs") == helper.get("inputs")
            and value.get("output_element_bytes") == [2]
            for value in attrs
        )
        and all(not produced.intersection(node.get("inputs", [])) for node in linears)
    )


def _component_hidden_size(component: Json) -> int:
    inputs = component.get("circuit", {}).get("boundary", {}).get("inputs", [])
    widths = {
        int(record["shape"][-1])
        for record in inputs
        if isinstance(record, dict)
        and isinstance(record.get("shape"), list)
        and record["shape"]
        and isinstance(record["shape"][-1], int)
        and not isinstance(record["shape"][-1], bool)
        and record["shape"][-1] > 0
    }
    if len(widths) != 1:
        raise ModelCompileError("component has no unique positive boundary width")
    return widths.pop()


def _capability_reason(profile: Json) -> str | None:
    if (
        profile.get("hardware_identity", {}).get("device_kind") != "gpu"
        or profile.get("provenance", {}).get("api") != "vulkan"
    ):
        return "exact parallel projection fusion requires a Vulkan GPU"
    try:
        device = _compiler_device(profile)
    except ModelCompileError:
        return "target has no complete Vulkan compiler capability contract"
    max_invocations = device.get("max_compute_work_group_invocations")
    max_size_x = device.get("max_compute_work_group_size_x")
    subgroup_size = device.get("subgroup_size")
    if (
        not _REQUIRED_FEATURES <= set(device.get("shader_features", []))
        or isinstance(max_invocations, bool)
        or not isinstance(max_invocations, int)
        or max_invocations < 1024
        or isinstance(max_size_x, bool)
        or not isinstance(max_size_x, int)
        or max_size_x < 1024
        or "arithmetic" not in device.get("subgroup_operations", [])
        or isinstance(subgroup_size, bool)
        or not isinstance(subgroup_size, int)
        or subgroup_size != 64
        or device.get("subgroup_compute_supported") is not True
    ):
        return "target cannot execute the exact fused parallel projection"
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


def _kernel_performance_record(kernel: Json) -> Json:
    return {
        "shader_path": kernel.get("shader_path"),
        "local_size_x": kernel.get("local_size_x"),
        "workgroup_count_x": kernel.get("workgroup_count_x"),
        "batch": [
            {
                "execution_domain": implementation.get("execution_domain"),
                "lane_tile_width": implementation.get("lane_tile_width"),
                "selection_priority": implementation.get("selection_priority"),
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
