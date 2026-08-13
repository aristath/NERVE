from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from nerve.behavioral_compiler import prove_exact_circuit_candidate
from nerve.circuit_optimizer import fuse_hyper_connection_rms_norm_regions
from nerve.compilation import Json, ModelCompileError
from nerve.model_package_manifest import can_fuse_hyper_connection_rms_norm
from nerve.physical_representations import FP8_E8M0_PREQUANTIZATION_CONTRACT
from nerve.representation_optimizer.providers.hyper_norm_fusion.discovery import (
    HyperNormFusionOpportunity,
)


@dataclass(frozen=True)
class FusedComponent:
    circuit: Json
    source_nodes: tuple[Json, ...]
    replacement_nodes: tuple[Json, ...]
    proof: Json


def fuse_component(
    *,
    opportunity: HyperNormFusionOpportunity,
    source_circuit: Json,
    compiled_circuit: Json,
    tensor_index: Json,
) -> FusedComponent:
    candidate = deepcopy(compiled_circuit)
    original_nodes = deepcopy(compiled_circuit.get("nodes"))
    if not isinstance(original_nodes, list):
        raise ModelCompileError("compiled hyper/RMS component has no node list")
    nodes = deepcopy(original_nodes)
    target_pairs = {
        (region.hyper_node_id, region.norm_node_id): region
        for region in opportunity.regions
    }
    for region in opportunity.regions:
        by_id = _index_nodes(nodes, "candidate")
        hyper = by_id.get(region.hyper_node_id)
        norm = by_id.get(region.norm_node_id)
        helper = by_id.get(region.quantizer_node_id)
        if hyper is None or norm is None or helper is None:
            raise ModelCompileError(
                f"hyper/RMS region {region.scope_id!r} drifted before fusion"
            )
        _absorb_quantizer(norm, helper)
        nodes = [node for node in nodes if node["id"] != region.quantizer_node_id]
    candidate["nodes"] = nodes
    boundary_outputs = {
        output.get("source", output["id"])
        for output in candidate.get("boundary", {}).get("outputs", [])
    }

    def can_fuse(hyper: Json, norm: Json) -> bool:
        if (hyper.get("id"), norm.get("id")) not in target_pairs:
            return False
        return can_fuse_hyper_connection_rms_norm(
            candidate,
            hyper,
            norm,
            tensor_index,
            hidden_size=opportunity.hidden_size,
            compiler_target={"devices": [opportunity.compiler_device]},
        )

    fused_nodes = fuse_hyper_connection_rms_norm_regions(
        nodes,
        can_fuse,
        boundary_outputs,
    )
    replacement_nodes = []
    for region in opportunity.regions:
        generated_id = f"{region.hyper_node_id}__{region.norm_node_id}"
        matches = [node for node in fused_nodes if node.get("id") == generated_id]
        if len(matches) != 1:
            raise ModelCompileError(
                f"hyper/RMS region {region.scope_id!r} did not fuse exactly once"
            )
        fused = matches[0]
        fused["id"] = region.quantizer_node_id
        replacement_nodes.append(deepcopy(fused))
        for node in fused_nodes:
            attrs = node.get("attrs", {})
            if attrs.get("physical_input_provider_id") == generated_id:
                attrs["physical_input_provider_id"] = region.quantizer_node_id
    candidate["nodes"] = fused_nodes
    expected_source_ids = {
        node_id for region in opportunity.regions for node_id in region.source_node_ids
    }
    source_nodes = tuple(
        deepcopy(node)
        for node in original_nodes
        if node["id"] in expected_source_ids
    )
    if {node["id"] for node in source_nodes} != expected_source_ids:
        raise ModelCompileError("hyper/RMS source region is incomplete")
    if len(candidate["nodes"]) != len(original_nodes) - 2 * len(opportunity.regions):
        raise ModelCompileError("hyper/RMS fusion changed an unexpected node count")
    proof = prove_exact_circuit_candidate(
        component_id=opportunity.component_id,
        source=source_circuit,
        candidate=candidate,
    )
    if proof.get("candidate_kind") != "exact_reference":
        raise ModelCompileError("hyper/RMS fusion lost exact-reference semantics")
    return FusedComponent(
        circuit=candidate,
        source_nodes=source_nodes,
        replacement_nodes=tuple(replacement_nodes),
        proof=proof,
    )


def _absorb_quantizer(norm: Json, helper: Json) -> None:
    norm_outputs = norm.get("outputs")
    helper_outputs = helper.get("outputs")
    norm_attrs = norm.get("attrs")
    helper_attrs = helper.get("attrs")
    if (
        not isinstance(norm_outputs, list)
        or len(norm_outputs) != 1
        or not isinstance(helper_outputs, list)
        or len(helper_outputs) != 2
        or helper.get("inputs") != norm_outputs
        or not isinstance(norm_attrs, dict)
        or not isinstance(helper_attrs, dict)
        or helper_attrs.get("physical_representation_contract")
        != FP8_E8M0_PREQUANTIZATION_CONTRACT
    ):
        raise ModelCompileError("hyper/RMS quantizer boundary drifted")
    logical_signal = str(norm_outputs[0])
    helper_output_bytes = helper_attrs.get("output_element_bytes")
    norm_output_bytes = norm_attrs.get("output_element_bytes", [2])
    if (
        not isinstance(helper_output_bytes, list)
        or len(helper_output_bytes) != 2
        or not isinstance(norm_output_bytes, list)
        or len(norm_output_bytes) != 1
    ):
        raise ModelCompileError("hyper/RMS output representation widths drifted")
    norm["outputs"] = [logical_signal, *deepcopy(helper_outputs)]
    norm_attrs["output_element_bytes"] = [
        norm_output_bytes[0],
        *deepcopy(helper_output_bytes),
    ]
    norm_attrs["physical_output_representations"] = [
        {
            "contract": FP8_E8M0_PREQUANTIZATION_CONTRACT,
            "logical_signal": logical_signal,
            "outputs": deepcopy(helper_outputs),
            "consumer_node_ids": deepcopy(helper_attrs["consumer_node_ids"]),
            "element_count": int(helper_attrs["element_count"]),
            "block_columns": int(helper_attrs["block_columns"]),
        }
    ]


def _index_nodes(nodes: list[Json], label: str) -> dict[str, Json]:
    indexed = {
        str(node["id"]): node
        for node in nodes
        if isinstance(node, dict) and isinstance(node.get("id"), str)
    }
    if len(indexed) != len(nodes):
        raise ModelCompileError(f"{label} nodes contain invalid or duplicate ids")
    return indexed
