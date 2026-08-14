from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

from nerve.compilation import Json, ModelCompileError
from nerve.model_package_shader_templates import render_shader_source
from nerve.model_package_spirv_requirements import (
    spirv_vulkan_requirements_from_payloads,
)
from nerve.physical_execution_contracts import (
    build_kernel_physical_execution_contracts,
)
from nerve.representation_optimizer.automation.target import CandidateToolchain
from nerve.representation_optimizer.contracts import contract_digest
from nerve.representation_optimizer.providers.attention_head_grouping.artifacts import (
    COMPONENT_FIXTURE_PATH,
    CONVERSATION_FIXTURE_PATH,
    MODEL_LIMITS_PATH,
    PRODUCT_CONVERSATION_FIXTURE_PATH,
    PROOF_PATH,
    component_fixture,
    conversation_fixture,
    model_limits_fixture,
    product_conversation_fixture,
)
from nerve.representation_optimizer.providers.attention_head_grouping.contracts import (
    PROOF_SCHEMA,
    TARGET_LOWERING_SCHEMA,
)
from nerve.representation_optimizer.providers.attention_head_grouping.discovery import (
    AttentionHeadGroupingOpportunity,
)
from nerve.representation_optimizer.providers.attention_head_grouping.physical import (
    PreparedGroupedAttention,
    prepare_grouped_attention_from_documents,
)
from nerve.representation_optimizer.providers.codebook.shaders import compile_spirv
from nerve.representation_optimizer.providers.types import ProviderCandidatePlan
from nerve.representation_optimizer.staging.workspace import (
    CandidateConstructionContext,
)


_SHADER_ROOT = Path(__file__).resolve().parents[4] / "runtime-rs" / "shaders"


class AttentionHeadGroupingToolchainResolver:
    def resolve(self, plan: ProviderCandidatePlan) -> CandidateToolchain:
        if (
            plan.provider.provider_id != "nerve.exact_attention_head_grouping"
            or plan.target_lowering.get("schema") != TARGET_LOWERING_SCHEMA
        ):
            raise ModelCompileError(
                "grouped-attention toolchain cannot construct provider "
                f"{plan.provider.provider_id!r}"
            )
        return CandidateToolchain(
            semantic_constructor=AttentionHeadGroupingSemanticConstructor(),
            ordinary_relowerer=AttentionHeadGroupingOrdinaryRelowerer(),
            physical_optimizer=AttentionHeadGroupingPhysicalOptimizer(),
        )


class AttentionHeadGroupingSemanticConstructor:
    def construct_semantic_artifacts(
        self,
        context: CandidateConstructionContext,
    ) -> None:
        opportunity = opportunities_from_lowering(_lowering(context))[0]
        context.write_json_artifact(
            COMPONENT_FIXTURE_PATH,
            component_fixture(
                component_id=opportunity.component_id,
                physical_node_id=opportunity.physical_node_id,
                query_heads=opportunity.query_heads,
                head_width=opportunity.head_width,
                local_window=opportunity.local_window,
                max_compressed_indices=opportunity.max_compressed_indices,
            ),
        )
        context.write_json_artifact(CONVERSATION_FIXTURE_PATH, conversation_fixture())
        context.write_json_artifact(
            PRODUCT_CONVERSATION_FIXTURE_PATH,
            product_conversation_fixture(),
        )
        context.write_json_artifact(
            MODEL_LIMITS_PATH,
            model_limits_fixture(opportunity.max_context_activations),
        )
        context.account_transient_bytes(0)


class AttentionHeadGroupingOrdinaryRelowerer:
    def run_ordinary_lowering(
        self,
        context: CandidateConstructionContext,
    ) -> None:
        _lowering(context)
        context.account_transient_bytes(0)


class AttentionHeadGroupingPhysicalOptimizer:
    def optimize_physical_artifacts(
        self,
        context: CandidateConstructionContext,
    ) -> None:
        lowering = _lowering(context)
        opportunities = opportunities_from_lowering(lowering)
        required_documents = {
            item.manifest_ref for item in opportunities
        } | {
            item.tensor_index_ref for item in opportunities
        } | {
            item.circuit_ref for item in opportunities
        }
        documents = {}
        for path in sorted(
            {
                path
                for item in opportunities
                for path in item.source_artifact_refs
            }
        ):
            payload = context.read_source_artifact(path)
            if path in required_documents:
                documents[path] = _json_object(payload, path)
        prepared = [
            prepare_grouped_attention_from_documents(
                opportunity=item,
                manifest=documents[item.manifest_ref],
                source_circuit=documents[item.circuit_ref],
            )
            for item in opportunities
        ]
        expected_shaders = tuple(
            (record["artifact_path"], record["template_name"])
            for record in lowering["shader_artifacts"]
        )
        for component in prepared:
            actual = tuple(
                (shader.artifact_path, shader.template_name)
                for shader in component.shader_artifacts
            )
            if actual != expected_shaders:
                raise ModelCompileError(
                    "performance-equivalent attention components lower to different shaders"
                )
        artifact_payloads = {}
        for artifact_path, template_name in expected_shaders:
            payload = compile_spirv(
                render_shader_source(_SHADER_ROOT, template_name),
                template_name,
            )
            artifact_payloads[artifact_path] = payload
            context.write_artifact(artifact_path, payload)

        proof_components = []
        for opportunity, component in zip(opportunities, prepared, strict=True):
            finalized = tuple(
                finalize_grouped_attention_kernel(
                    kernel,
                    component=component,
                    tensor_index=documents[opportunity.tensor_index_ref],
                    artifact_payloads=artifact_payloads,
                )
                for kernel in component.replacement_kernels
            )
            overlay = {
                "schema": "nerve.optimizer.vulkan_component_region_overlay.v2",
                "source_component_id": opportunity.component_id,
                "source": {
                    "nodes": list(component.source_nodes),
                    "kernels": list(component.source_kernels),
                    "parameter_refs": {},
                },
                "replacement": {
                    "nodes": list(component.replacement_nodes),
                    "kernels": list(finalized),
                    "parameter_refs": {},
                },
            }
            context.write_json_artifact(
                _lowered_component(lowering, opportunity.component_id)["overlay_path"],
                overlay,
            )
            proof_components.append(
                {
                    "component_id": opportunity.component_id,
                    "scope_id": opportunity.scope_id,
                    "head_group": opportunity.head_group,
                    "source_region_digest": contract_digest(overlay["source"]),
                    "replacement_region_digest": contract_digest(
                        overlay["replacement"]
                    ),
                    "exact_rewrite": component.proof,
                }
            )
        context.write_json_artifact(
            PROOF_PATH,
            {
                "schema": PROOF_SCHEMA,
                "candidate_id": lowering["candidate_id"],
                "scope_ids": lowering["scope_ids"],
                "components": proof_components,
                "shader_artifacts": [
                    {
                        "path": path,
                        "sha256": f"sha256:{sha256(payload).hexdigest()}",
                    }
                    for path, payload in sorted(artifact_payloads.items())
                ],
            },
        )
        context.account_transient_bytes(0)


def finalize_grouped_attention_kernel(
    source_kernel: Json,
    *,
    component: PreparedGroupedAttention,
    tensor_index: Json,
    artifact_payloads: dict[str, bytes],
) -> Json:
    kernel = deepcopy(source_kernel)
    for implementation in kernel.get("batch_implementations", []):
        paths = {
            str(stage["shader_path"]) for stage in implementation.get("stages", [])
        }
        features, subgroup_operations = spirv_vulkan_requirements_from_payloads(
            {path: artifact_payloads[path] for path in paths}
        )
        requirements = implementation["device_requirements"]
        requirements["vulkan_features"] = features
        requirements["subgroup_operations"] = subgroup_operations
    node = component.replacement_nodes[0]
    kernel["physical_execution_contracts"] = build_kernel_physical_execution_contracts(
        node=node,
        circuit=component.circuit,
        tensor_index=tensor_index,
        kernel=kernel,
        package_dir=Path("."),
        artifact_payloads=artifact_payloads,
    )
    kernel.pop("physical_implementations", None)
    return kernel


def opportunities_from_lowering(
    lowering: Json,
) -> tuple[AttentionHeadGroupingOpportunity, ...]:
    records = lowering.get("regions")
    if not isinstance(records, list) or not records:
        raise ModelCompileError("grouped-attention lowering has no component regions")
    opportunities = tuple(_opportunity(record) for record in records)
    component_ids = [item.component_id for item in opportunities]
    if component_ids != sorted(set(component_ids)):
        raise ModelCompileError(
            "grouped-attention lowering components must be sorted and unique"
        )
    signatures = {item.performance_signature for item in opportunities}
    if len(signatures) != 1:
        raise ModelCompileError(
            "grouped-attention lowering crosses performance classes"
        )
    return opportunities


def _opportunity(record: Json) -> AttentionHeadGroupingOpportunity:
    return AttentionHeadGroupingOpportunity(
        scope_id=str(record["scope_id"]),
        source_contract_digest=str(record["source_contract_digest"]),
        component_id=str(record["component_id"]),
        source_node_id=str(record["source_node_id"]),
        physical_node_id=str(record["physical_node_id"]),
        terminal_node_id=str(record["terminal_node_id"]),
        evidence_ids=tuple(record["evidence_ids"]),
        source_artifact_refs=tuple(record["source_artifact_refs"]),
        manifest_ref=str(record["manifest_ref"]),
        circuit_ref=str(record["circuit_ref"]),
        tensor_index_ref=str(record["tensor_index_ref"]),
        query_heads=int(record["query_heads"]),
        key_value_heads=int(record["key_value_heads"]),
        head_width=int(record["head_width"]),
        local_window=int(record["local_window"]),
        compression_ratio=int(record["compression_ratio"]),
        max_compressed_indices=int(record["max_compressed_indices"]),
        head_group=int(record["head_group"]),
        shader_suffix=str(record["shader_suffix"]),
        max_context_activations=int(record["max_context_activations"]),
        compiler_device=deepcopy(record["compiler_device"]),
        performance_signature=str(record["performance_signature"]),
    )


def _lowered_component(lowering: Json, component_id: str) -> Json:
    matches = [
        record
        for record in lowering["regions"]
        if record.get("component_id") == component_id
    ]
    if len(matches) != 1:
        raise ModelCompileError(
            f"grouped-attention lowering has no unique component {component_id!r}"
        )
    return matches[0]


def _lowering(context: CandidateConstructionContext) -> Json:
    lowering = context.target_lowering
    if lowering.get("schema") != TARGET_LOWERING_SCHEMA:
        raise ModelCompileError(
            "grouped-attention toolchain received incompatible lowering"
        )
    if lowering.get("candidate_id") != context.candidate["candidate_id"]:
        raise ModelCompileError(
            "grouped-attention lowering belongs to another candidate"
        )
    return lowering


def _json_object(payload: bytes, label: str) -> Json:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelCompileError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ModelCompileError(f"{label} must contain a JSON object")
    return value
