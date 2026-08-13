from __future__ import annotations

import json
from dataclasses import dataclass

from nerve.compilation import Json, ModelCompileError
from nerve.physical_representations import FP8_E8M0_PREQUANTIZATION_CONTRACT
from nerve.representation_optimizer.contracts import stable_contract_id
from nerve.representation_optimizer.providers.adjacent_boundaries import (
    exact_adjacent_boundary_records,
    is_adjacent_representation_scope,
)
from nerve.representation_optimizer.providers.types import ProviderContext


_HYPER_OPS = {"hyper_connection_pre", "hyper_connection_post_pre"}
_REQUIRED_FEATURES = frozenset(("shader_float8", "shader_int8"))


@dataclass(frozen=True)
class HyperNormRegion:
    scope_id: str
    source_contract_digest: str
    semantic_source_node_ids: tuple[str, str]
    hyper_node_id: str
    norm_node_id: str
    quantizer_node_id: str
    boundary_scope_ids: tuple[str, ...] = ()
    boundary_source_contract_digests: tuple[str, ...] = ()

    @property
    def source_node_ids(self) -> tuple[str, str, str]:
        return (
            self.hyper_node_id,
            self.norm_node_id,
            self.quantizer_node_id,
        )


@dataclass(frozen=True)
class HyperNormFusionOpportunity:
    component_id: str
    regions: tuple[HyperNormRegion, ...]
    evidence_ids: tuple[str, ...]
    source_artifact_refs: tuple[str, ...]
    manifest_ref: str
    circuit_ref: str
    tensor_index_ref: str
    terminal_node_id: str
    hidden_size: int
    max_context_activations: int
    compiler_device: Json
    performance_signature: str

    @property
    def source_scope_contracts(self) -> tuple[tuple[str, str], ...]:
        records = []
        for region in self.regions:
            records.append((region.scope_id, region.source_contract_digest))
            records.extend(
                zip(
                    region.boundary_scope_ids,
                    region.boundary_source_contract_digests,
                    strict=True,
                )
            )
        by_scope: dict[str, str] = {}
        for scope_id, digest in records:
            previous = by_scope.setdefault(scope_id, digest)
            if previous != digest:
                raise ModelCompileError(
                    "hyper/RMS alternative contains conflicting source scopes"
                )
        return tuple(sorted(by_scope.items()))

    @property
    def scope_ids(self) -> tuple[str, ...]:
        return tuple(scope_id for scope_id, _digest in self.source_scope_contracts)

    @property
    def source_contract_digests(self) -> tuple[str, ...]:
        return tuple(digest for _scope_id, digest in self.source_scope_contracts)


@dataclass(frozen=True)
class DiscoveryResult:
    opportunities: tuple[HyperNormFusionOpportunity, ...]
    reasons: tuple[str, ...]
    evidence_ids: tuple[str, ...] = ()


def is_hyper_norm_scope(scope: Json, source_contract: Json) -> bool:
    return is_adjacent_representation_scope(scope, source_contract)


def discover_hyper_norm_fusions(
    context: ProviderContext,
) -> tuple[HyperNormFusionOpportunity, ...]:
    key = "hyper_norm_fusion.v2:" + ",".join(context.scope_ids)
    return context.memoized(
        key,
        lambda: _discover(context).opportunities,
    )  # type: ignore[return-value]


def discovery_result(context: ProviderContext) -> DiscoveryResult:
    key = "hyper_norm_fusion.result.v2:" + ",".join(context.scope_ids)
    return context.memoized(key, lambda: _discover(context))  # type: ignore[return-value]


def source_inputs(
    context: ProviderContext,
    opportunity: HyperNormFusionOpportunity,
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
        if is_hyper_norm_scope(scope, contracts[str(scope["scope_id"])])
    ]
    if not eligible:
        return DiscoveryResult((), ("no adjacent representation-island scopes",))

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
            component_opportunities = _component_opportunities(
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
        except ModelCompileError as error:
            rejected.append(str(error))
            continue
        except (KeyError, TypeError, ValueError) as error:
            rejected.append(
                f"component {component_id!r} has malformed fusion metadata: {error}"
            )
            continue
        opportunities.extend(component_opportunities)
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
                "hyper/RMS component alternatives",
            ),
            evidence_ids,
        )
    return DiscoveryResult(
        (),
        tuple(rejected[:8]) or ("no structurally valid hyper/RMS fusion",),
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
) -> tuple[HyperNormFusionOpportunity, ...]:
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
    circuit_refs = {
        path
        for _scope, contract in scoped_contracts
        for path in contract.get("exact_reference", {}).get("artifact_refs", [])
        if isinstance(path, str) and path.endswith("/circuit.json")
    }
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

    region_records: list[
        tuple[HyperNormRegion, Json, tuple[str, ...], tuple[Json, ...]]
    ] = []
    hidden_size: int | None = None
    for scope, contract in sorted(
        scoped_contracts,
        key=lambda item: str(item[0]["scope_id"]),
    ):
        scope_id = str(scope["scope_id"])
        supported_evidence = evidence_by_scope[scope_id]
        if not supported_evidence:
            continue
        raw_ids = tuple(
            str(value).removeprefix(f"{component_id}/")
            for value in scope["members"]["source_node_ids"]
        )
        if len(raw_ids) != 2 or len(set(raw_ids)) != 2:
            continue
        raw = [source_nodes.get(node_id) for node_id in raw_ids]
        if any(node is None for node in raw):
            continue
        reduce_nodes = [node for node in raw if node.get("op") == "hyper_connection_reduce"]
        norm_nodes = [node for node in raw if node.get("op") == "rms_norm"]
        if len(reduce_nodes) != 1 or len(norm_nodes) != 1:
            continue
        reduce = reduce_nodes[0]
        source_norm = norm_nodes[0]
        if source_norm.get("inputs") != reduce.get("outputs"):
            continue
        hyper_matches = [
            node
            for node in compiled_nodes.values()
            if node.get("op") in _HYPER_OPS
            and reduce["id"] in node.get("attrs", {}).get("compiled_from", [])
        ]
        norm_matches = [
            node
            for node in compiled_nodes.values()
            if node.get("id") == source_norm["id"] and node.get("op") == "rms_norm"
        ]
        if len(hyper_matches) != 1 or len(norm_matches) != 1:
            continue
        hyper = hyper_matches[0]
        norm = norm_matches[0]
        helper_matches = [
            node
            for node in compiled_nodes.values()
            if node.get("op") == "quantize_fp8_e4m3_e8m0"
            and node.get("inputs") == norm.get("outputs")
        ]
        if len(helper_matches) != 1:
            continue
        helper = helper_matches[0]
        attrs = helper.get("attrs", {})
        hyper_attrs = hyper.get("attrs", {})
        if not isinstance(attrs, dict) or not isinstance(hyper_attrs, dict):
            continue
        region_hidden = attrs.get("element_count")
        block_columns = attrs.get("block_columns")
        multiplicity = hyper_attrs.get("multiplicity")
        consumer_node_ids = attrs.get("consumer_node_ids")
        semantic_consumer_node_ids = attrs.get("semantic_source_node_ids")
        if (
            attrs.get("physical_representation_contract")
            != FP8_E8M0_PREQUANTIZATION_CONTRACT
            or isinstance(region_hidden, bool)
            or not isinstance(region_hidden, int)
            or region_hidden != 4096
            or isinstance(block_columns, bool)
            or not isinstance(block_columns, int)
            or block_columns != 128
            or isinstance(multiplicity, bool)
            or not isinstance(multiplicity, int)
            or multiplicity != 4
            or not isinstance(consumer_node_ids, list)
            or not consumer_node_ids
            or len(consumer_node_ids) != len(set(consumer_node_ids))
            or any(
                not isinstance(consumer_node_id, str) or not consumer_node_id
                for consumer_node_id in consumer_node_ids
            )
            or not isinstance(semantic_consumer_node_ids, list)
            or not semantic_consumer_node_ids
            or len(semantic_consumer_node_ids)
            != len(set(semantic_consumer_node_ids))
            or any(
                not isinstance(consumer_node_id, str) or not consumer_node_id
                for consumer_node_id in semantic_consumer_node_ids
            )
            or len(helper.get("outputs", [])) != 2
            or any(
                node_id not in kernels
                for node_id in (hyper["id"], norm["id"], helper["id"])
            )
            or positions[norm["id"]] != positions[hyper["id"]] + 1
            or positions[helper["id"]] <= positions[norm["id"]]
        ):
            continue
        if hidden_size is None:
            hidden_size = region_hidden
        elif hidden_size != region_hidden:
            raise ModelCompileError(
                f"component {component_id!r} hyper/RMS regions disagree on width"
            )
        boundary_records = exact_adjacent_boundary_records(
            component_id=component_id,
            producer_node_id=str(source_norm["id"]),
            consumer_node_ids=tuple(semantic_consumer_node_ids),
            scoped_contracts=scoped_contracts,
            evidence_by_scope=evidence_by_scope,
            require_all=False,
        )
        if boundary_records is None:
            continue
        boundary_scopes = tuple(record[0] for record in boundary_records)
        boundary_contracts = tuple(record[1] for record in boundary_records)
        region_evidence = tuple(
            sorted(
                set(supported_evidence).union(
                    evidence_id
                    for _scope, _contract, evidence_ids in boundary_records
                    for evidence_id in evidence_ids
                )
            )
        )
        region = HyperNormRegion(
            scope_id=scope_id,
            source_contract_digest=str(contract["contract_digest"]),
            semantic_source_node_ids=(str(reduce["id"]), str(source_norm["id"])),
            hyper_node_id=str(hyper["id"]),
            norm_node_id=str(norm["id"]),
            quantizer_node_id=str(helper["id"]),
            boundary_scope_ids=tuple(
                str(boundary_scope["scope_id"])
                for boundary_scope in boundary_scopes
            ),
            boundary_source_contract_digests=tuple(
                str(boundary_contract["contract_digest"])
                for boundary_contract in boundary_contracts
            ),
        )
        region_records.append(
            (
                region,
                {
                    "hyper": {
                        "op": hyper["op"],
                        "attrs": hyper["attrs"],
                        "kernel": _kernel_performance_record(kernels[hyper["id"]]),
                    },
                    "norm": {
                        "op": norm["op"],
                        "attrs": norm["attrs"],
                        "kernel": _kernel_performance_record(kernels[norm["id"]]),
                    },
                    "quantizer": {
                        "op": helper["op"],
                        "attrs": helper["attrs"],
                        "kernel": _kernel_performance_record(
                            kernels[helper["id"]]
                        ),
                    },
                },
                region_evidence,
                (contract, *boundary_contracts),
            )
        )
    if not region_records:
        return ()
    region_records.sort(key=lambda record: positions[record[0].hyper_node_id])
    regions = [record[0] for record in region_records]
    region_ids = [node_id for region in regions for node_id in region.source_node_ids]
    if len(region_ids) != len(set(region_ids)):
        raise ModelCompileError(
            f"component {component_id!r} hyper/RMS regions overlap"
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
    return tuple(
        HyperNormFusionOpportunity(
            component_id=component_id,
            regions=(region,),
            evidence_ids=tuple(sorted(supported_evidence)),
            source_artifact_refs=tuple(
                sorted(
                    {
                        manifest_ref,
                        tensor_index_ref,
                        circuit_ref,
                        *(
                            path
                            for source_contract in contracts
                            for path in source_contract["exact_reference"][
                                "artifact_refs"
                            ]
                        ),
                    }
                )
            ),
            manifest_ref=manifest_ref,
            circuit_ref=circuit_ref,
            tensor_index_ref=tensor_index_ref,
            terminal_node_id=terminal_node_id,
            hidden_size=int(hidden_size),
            max_context_activations=max_context_activations,
            compiler_device=compiler_device,
            performance_signature=stable_contract_id(
                "hyper_norm_performance_class",
                {
                    "hidden_size": hidden_size,
                    "region": performance_record,
                },
            ),
        )
        for region, performance_record, supported_evidence, contracts in region_records
    )


def _capability_reason(profile: Json) -> str | None:
    if (
        profile.get("hardware_identity", {}).get("device_kind") != "gpu"
        or profile.get("provenance", {}).get("api") != "vulkan"
    ):
        return "exact hyper/RMS fusion requires a Vulkan GPU"
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
        return "target cannot execute the exact fused hyper/RMS transaction"
    return None


def _compiler_device(profile: Json) -> Json:
    value = (
        profile.get("capability_extensions", {})
        .get("vulkan_compiler_capabilities")
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
