from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass

from nerve.behavioral_compiler import prove_exact_circuit_candidate
from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.providers.attention_head_grouping.artifacts import (
    kernel_artifact_path,
)
from nerve.representation_optimizer.providers.attention_head_grouping.discovery import (
    AttentionHeadGroupingOpportunity,
)
from nerve.representation_optimizer.providers.types import ProviderContext


@dataclass(frozen=True)
class ShaderArtifact:
    artifact_path: str
    template_name: str


@dataclass(frozen=True)
class PreparedGroupedAttention:
    circuit: Json
    source_nodes: tuple[Json, ...]
    replacement_nodes: tuple[Json, ...]
    source_kernels: tuple[Json, ...]
    replacement_kernels: tuple[Json, ...]
    shader_artifacts: tuple[ShaderArtifact, ...]
    proof: Json


def prepare_grouped_attention(
    context: ProviderContext,
    opportunity: AttentionHeadGroupingOpportunity,
) -> PreparedGroupedAttention:
    key = (
        "attention_head_grouping.prepared.v1:"
        f"{opportunity.component_id}:{opportunity.performance_signature}:"
        f"{opportunity.scope_id}:{context.hardware_profile['capability_class']}"
    )
    return context.memoized(
        key,
        lambda: _prepare(context, opportunity),
    )  # type: ignore[return-value]


def _prepare(
    context: ProviderContext,
    opportunity: AttentionHeadGroupingOpportunity,
) -> PreparedGroupedAttention:
    context.checkpoint()
    return prepare_grouped_attention_from_documents(
        opportunity=opportunity,
        manifest=_source_json(context, opportunity.manifest_ref),
        source_circuit=_source_json(context, opportunity.circuit_ref),
    )


def prepare_grouped_attention_from_documents(
    *,
    opportunity: AttentionHeadGroupingOpportunity,
    manifest: Json,
    source_circuit: Json,
) -> PreparedGroupedAttention:
    component = _unique(
        manifest.get("circuit_graph", {}).get("components"),
        "component_id",
        opportunity.component_id,
    )
    execution = _unique(
        manifest.get("component_executions"),
        "component_id",
        opportunity.component_id,
    )
    compiled_circuit = deepcopy(component.get("circuit"))
    if not isinstance(compiled_circuit, dict):
        raise ModelCompileError("grouped-attention source circuit is missing")
    source_node = _unique(
        compiled_circuit.get("nodes"),
        "id",
        opportunity.physical_node_id,
    )
    source_kernel = _unique(
        execution.get("kernels"),
        "node_id",
        opportunity.physical_node_id,
    )
    if (
        source_node.get("op") != "indexed_sparse_attention"
        or source_kernel.get("op") != "indexed_sparse_attention"
    ):
        raise ModelCompileError(
            "grouped-attention source no longer contains indexed sparse attention"
        )
    replacement_node = deepcopy(source_node)
    replacement_kernel = deepcopy(source_kernel)
    decode = ShaderArtifact(
        kernel_artifact_path(opportunity.decode_shader_file),
        opportunity.decode_shader_file,
    )
    prefill = ShaderArtifact(
        kernel_artifact_path(opportunity.prefill_shader_file),
        opportunity.prefill_shader_file,
    )
    replacement_kernel["shader_path"] = decode.artifact_path
    replacement_kernel["local_size_x"] = opportunity.head_width * 2
    replacement_kernel["workgroup_count_x"] = (
        opportunity.query_heads // opportunity.head_group
    )
    batch = replacement_kernel.get("batch_implementations")
    if not isinstance(batch, list) or len(batch) != 1:
        raise ModelCompileError(
            "grouped-attention source has no unique prefill implementation"
        )
    stages = batch[0].get("stages")
    if not isinstance(stages, list) or len(stages) != 1:
        raise ModelCompileError(
            "grouped-attention source has no unique prefill stage"
        )
    stages[0]["shader_path"] = prefill.artifact_path
    stages[0]["local_size_x"] = opportunity.head_width * 2
    stages[0]["workgroup_count_x"] = (
        opportunity.query_heads // opportunity.head_group
    )
    replacement_kernel.pop("physical_implementations", None)
    proof = prove_exact_circuit_candidate(
        component_id=opportunity.component_id,
        source=source_circuit,
        candidate=compiled_circuit,
    )
    if proof.get("candidate_kind") != "exact_reference":
        raise ModelCompileError(
            "grouped-attention scheduling lost exact source semantics"
        )
    return PreparedGroupedAttention(
        circuit=compiled_circuit,
        source_nodes=(deepcopy(source_node),),
        replacement_nodes=(replacement_node,),
        source_kernels=(deepcopy(source_kernel),),
        replacement_kernels=(replacement_kernel,),
        shader_artifacts=(decode, prefill),
        proof=proof,
    )


def _source_json(context: ProviderContext, path: str) -> Json:
    key = f"attention_head_grouping.source_json.v1:{path}"

    def load() -> Json:
        try:
            value = json.loads(context.source_artifacts.read_path(path))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ModelCompileError(f"{path} is not valid JSON") from error
        if not isinstance(value, dict):
            raise ModelCompileError(f"{path} must contain a JSON object")
        return value

    return context.memoized(key, load)  # type: ignore[return-value]


def _unique(records: object, field: str, value: str) -> Json:
    if not isinstance(records, list):
        raise ModelCompileError(f"grouped-attention source {field} records are missing")
    matches = [
        record
        for record in records
        if isinstance(record, dict) and record.get(field) == value
    ]
    if len(matches) != 1:
        raise ModelCompileError(
            f"grouped-attention source has no unique {field}={value!r}"
        )
    return matches[0]
