from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from nerve.behavioral_compiler import prove_exact_circuit_candidate
from nerve.compilation import Json, ModelCompileError
from nerve.physical_representations import FP8_E8M0_PREQUANTIZATION_CONTRACT
from nerve.representation_optimizer.providers.parallel_projection_fusion.discovery import (
    ParallelProjectionFusionOpportunity,
)


@dataclass(frozen=True)
class FusedComponent:
    circuit: Json
    source_nodes: tuple[Json, ...]
    replacement_nodes: tuple[Json, ...]
    proof: Json


def fuse_component(
    *,
    opportunity: ParallelProjectionFusionOpportunity,
    source_circuit: Json,
    compiled_circuit: Json,
    tensor_index: Json,
) -> FusedComponent:
    del tensor_index
    original_nodes = deepcopy(compiled_circuit.get("nodes"))
    if not isinstance(original_nodes, list):
        raise ModelCompileError(
            "compiled parallel projection component has no node list"
        )
    by_id = _index_nodes(original_nodes, "compiled parallel projection")
    region = opportunity.region
    helper = by_id.get(region.quantizer_node_id)
    linears = [by_id.get(node_id) for node_id in region.linear_node_ids]
    if helper is None or any(node is None for node in linears):
        raise ModelCompileError(
            f"parallel projection region {region.fused_node_id!r} drifted before fusion"
        )
    linear_nodes = [node for node in linears if node is not None]
    fused = _fused_node(opportunity, helper, linear_nodes)
    replacement_helper = _replacement_helper(opportunity, helper, fused)

    helper_index = next(
        index
        for index, node in enumerate(original_nodes)
        if node["id"] == helper["id"]
    )
    linear_indices = [
        next(
            index
            for index, node in enumerate(original_nodes)
            if node["id"] == linear["id"]
        )
        for linear in linear_nodes
    ]
    if helper_index >= min(linear_indices):
        raise ModelCompileError(
            "parallel projection quantizer must precede every fused consumer"
        )
    first_linear_id = original_nodes[min(linear_indices)]["id"]
    source_ids = set(region.source_node_ids)
    candidate_nodes = []
    replacement_nodes = []
    for node in original_nodes:
        node_id = str(node["id"])
        if node_id == helper["id"]:
            candidate_nodes.append(deepcopy(replacement_helper))
            replacement_nodes.append(deepcopy(replacement_helper))
        elif node_id == first_linear_id:
            candidate_nodes.append(deepcopy(fused))
            replacement_nodes.append(deepcopy(fused))
        elif node_id in region.linear_node_ids:
            continue
        else:
            candidate_nodes.append(deepcopy(node))
    source_nodes = tuple(
        deepcopy(node) for node in original_nodes if node["id"] in source_ids
    )
    if {str(node["id"]) for node in source_nodes} != source_ids:
        raise ModelCompileError("parallel projection source region is incomplete")
    if len(replacement_nodes) != 2:
        raise ModelCompileError(
            "parallel projection replacement must contain one helper and one projection"
        )
    expected_count = len(original_nodes) - len(linear_nodes) + 1
    if len(candidate_nodes) != expected_count:
        raise ModelCompileError(
            "parallel projection fusion changed an unexpected node count"
        )
    candidate = deepcopy(compiled_circuit)
    candidate["nodes"] = candidate_nodes
    proof = prove_exact_circuit_candidate(
        component_id=opportunity.component_id,
        source=source_circuit,
        candidate=candidate,
    )
    if proof.get("candidate_kind") != "exact_reference":
        raise ModelCompileError(
            "parallel projection fusion lost exact-reference semantics"
        )
    rewrite = [
        record
        for record in proof.get("rewrites", [])
        if record.get("candidate_node") == region.fused_node_id
    ]
    if (
        len(rewrite) != 1
        or rewrite[0].get("proof_contract") != "parallel_linear_exact_bf16.v1"
        or rewrite[0].get("source_nodes") != list(region.semantic_source_node_ids)
    ):
        raise ModelCompileError(
            "parallel projection proof did not cover the exact source branches"
        )
    return FusedComponent(
        circuit=candidate,
        source_nodes=source_nodes,
        replacement_nodes=tuple(replacement_nodes),
        proof=proof,
    )


def _fused_node(
    opportunity: ParallelProjectionFusionOpportunity,
    helper: Json,
    linears: list[Json],
) -> Json:
    region = opportunity.region
    helper_outputs = helper.get("outputs")
    helper_attrs = helper.get("attrs")
    linear_attrs = [node.get("attrs") for node in linears]
    if (
        not isinstance(helper_outputs, list)
        or len(helper_outputs) != 2
        or not isinstance(helper_attrs, dict)
        or helper_attrs.get("physical_representation_contract")
        != FP8_E8M0_PREQUANTIZATION_CONTRACT
        or set(helper_attrs.get("consumer_node_ids", []))
        != set(region.linear_node_ids)
        or set(helper_attrs.get("semantic_source_node_ids", []))
        != set(region.semantic_source_node_ids)
        or any(not isinstance(attrs, dict) for attrs in linear_attrs)
    ):
        raise ModelCompileError("parallel projection helper contract drifted")
    attrs = [value for value in linear_attrs if isinstance(value, dict)]
    shared_contract = attrs[0].get("physical_input_contract")
    shared_provider = attrs[0].get("physical_input_provider_id")
    shared_logical_inputs = attrs[0].get("physical_logical_inputs")
    passthrough = attrs[0].get("physical_passthrough_inputs")
    if (
        len(linears) not in {2, 3}
        or any(node.get("op") != "linear" for node in linears)
        or any(node.get("inputs") != helper_outputs for node in linears)
        or any(len(node.get("outputs", [])) != 1 for node in linears)
        or any(len(node.get("params", [])) != 2 for node in linears)
        or any(node.get("state_reads") or node.get("state_writes") for node in linears)
        or shared_contract != FP8_E8M0_PREQUANTIZATION_CONTRACT
        or shared_provider != helper["id"]
        or any(value.get("physical_input_contract") != shared_contract for value in attrs)
        or any(value.get("physical_input_provider_id") != shared_provider for value in attrs)
        or any(value.get("physical_logical_inputs") != shared_logical_inputs for value in attrs)
        or any(value.get("physical_passthrough_inputs") != passthrough for value in attrs)
        or any(value.get("output_element_bytes") != [2] for value in attrs)
    ):
        raise ModelCompileError("parallel projection branch contract drifted")
    fused_attrs: Json = {
        "compiled_from": list(region.semantic_source_node_ids),
        "branch_count": len(linears),
        "branch_parameter_counts": [len(node["params"]) for node in linears],
        "output_element_bytes": [2] * len(linears),
        "physical_input_contract": shared_contract,
        "physical_input_provider_id": shared_provider,
        "physical_input_source_node_ids": list(region.semantic_source_node_ids),
        "physical_logical_inputs": deepcopy(shared_logical_inputs),
    }
    if passthrough is not None:
        fused_attrs["physical_passthrough_inputs"] = deepcopy(passthrough)
    return {
        "id": region.fused_node_id,
        "op": f"parallel_linear_{len(linears)}way",
        "inputs": deepcopy(helper_outputs),
        "outputs": [str(node["outputs"][0]) for node in linears],
        "params": [parameter for node in linears for parameter in node["params"]],
        "attrs": fused_attrs,
    }


def _replacement_helper(
    opportunity: ParallelProjectionFusionOpportunity,
    helper: Json,
    fused: Json,
) -> Json:
    replacement = deepcopy(helper)
    attrs = replacement.get("attrs")
    if not isinstance(attrs, dict):
        raise ModelCompileError("parallel projection helper has no attributes")
    attrs["consumer_node_ids"] = [fused["id"]]
    attrs["semantic_source_node_ids"] = list(
        opportunity.region.semantic_source_node_ids
    )
    return replacement


def _index_nodes(nodes: list[Json], label: str) -> dict[str, Json]:
    indexed = {
        str(node["id"]): node
        for node in nodes
        if isinstance(node, dict)
        and isinstance(node.get("id"), str)
        and node["id"]
    }
    if len(indexed) != len(nodes):
        raise ModelCompileError(f"{label} nodes contain invalid or duplicate ids")
    return indexed
