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
from nerve.representation_optimizer.providers.codebook.shaders import compile_spirv
from nerve.representation_optimizer.providers.parallel_projection_fusion.artifacts import (
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
from nerve.representation_optimizer.providers.parallel_projection_fusion.contracts import (
    PROOF_SCHEMA,
    TARGET_LOWERING_SCHEMA,
)
from nerve.representation_optimizer.providers.parallel_projection_fusion.discovery import (
    ParallelProjectionFusionOpportunity,
    ParallelProjectionRegion,
)
from nerve.representation_optimizer.providers.hyper_norm_fusion.discovery import (
    HyperNormFusionOpportunity,
    HyperNormRegion,
)
from nerve.representation_optimizer.providers.parallel_projection_fusion.physical import (
    PreparedFusedComponent,
    prepare_fused_component_from_documents,
)
from nerve.representation_optimizer.providers.types import ProviderCandidatePlan
from nerve.representation_optimizer.staging.workspace import (
    CandidateConstructionContext,
)


_SHADER_ROOT = Path(__file__).resolve().parents[4] / "runtime-rs" / "shaders"


class ParallelProjectionFusionToolchainResolver:
    def resolve(self, plan: ProviderCandidatePlan) -> CandidateToolchain:
        if (
            plan.provider.provider_id != "nerve.exact_parallel_projection_fusion"
            or plan.target_lowering.get("schema") != TARGET_LOWERING_SCHEMA
        ):
            raise ModelCompileError(
                "parallel projection fusion toolchain cannot construct provider "
                f"{plan.provider.provider_id!r}"
            )
        return CandidateToolchain(
            semantic_constructor=ParallelProjectionFusionSemanticConstructor(),
            ordinary_relowerer=ParallelProjectionFusionOrdinaryRelowerer(),
            physical_optimizer=ParallelProjectionFusionPhysicalOptimizer(),
        )


class ParallelProjectionFusionSemanticConstructor:
    def construct_semantic_artifacts(
        self,
        context: CandidateConstructionContext,
    ) -> None:
        lowering = _lowering(context)
        opportunity = opportunities_from_lowering(lowering)[0]
        context.write_json_artifact(
            COMPONENT_FIXTURE_PATH,
            component_fixture(
                component_id=opportunity.component_id,
                physical_node_id=opportunity.physical_node_id,
                hidden_size=opportunity.hidden_size,
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


class ParallelProjectionFusionOrdinaryRelowerer:
    def run_ordinary_lowering(
        self,
        context: CandidateConstructionContext,
    ) -> None:
        _lowering(context)
        context.account_transient_bytes(0)


class ParallelProjectionFusionPhysicalOptimizer:
    def optimize_physical_artifacts(
        self,
        context: CandidateConstructionContext,
    ) -> None:
        lowering = _lowering(context)
        opportunities = opportunities_from_lowering(lowering)
        required_documents = {
            opportunity.manifest_ref
            for opportunity in opportunities
        } | {
            opportunity.tensor_index_ref
            for opportunity in opportunities
        } | {
            opportunity.circuit_ref
            for opportunity in opportunities
        }
        documents = {}
        declared_paths = sorted(
            {
                path
                for opportunity in opportunities
                for path in opportunity.source_artifact_refs
            }
        )
        for path in declared_paths:
            payload = context.read_source_artifact(path)
            if path in required_documents:
                documents[path] = _json_object(payload, path)

        prepared = [
            prepare_fused_component_from_documents(
                opportunity=opportunity,
                manifest=documents[opportunity.manifest_ref],
                tensor_index=documents[opportunity.tensor_index_ref],
                source_circuit=documents[opportunity.circuit_ref],
            )
            for opportunity in opportunities
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
                    "parallel projection performance-equivalent components lower to different shaders"
                )

        artifact_payloads = {}
        for artifact_path, template_name in expected_shaders:
            source = render_shader_source(_SHADER_ROOT, template_name)
            payload = compile_spirv(source, template_name)
            artifact_payloads[artifact_path] = payload
            context.write_artifact(artifact_path, payload)

        proof_components = []
        for opportunity, component in zip(opportunities, prepared, strict=True):
            finalized_kernels = tuple(
                finalize_fused_kernel(
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
                    "nodes": list(component.transformed.source_nodes),
                    "kernels": list(component.source_kernels),
                    "parameter_refs": {},
                },
                "replacement": {
                    "nodes": list(component.transformed.replacement_nodes),
                    "kernels": list(finalized_kernels),
                    "parameter_refs": {},
                },
            }
            context.write_json_artifact(
                _lowered_component(
                    lowering,
                    opportunity.component_id,
                    opportunity.physical_node_id,
                )["overlay_path"],
                overlay,
            )
            proof_components.append(
                {
                    "component_id": opportunity.component_id,
                    "physical_node_id": opportunity.physical_node_id,
                    "scope_ids": list(opportunity.scope_ids),
                    "source_region_digest": contract_digest(overlay["source"]),
                    "replacement_region_digest": contract_digest(
                        overlay["replacement"]
                    ),
                    "exact_rewrite": component.transformed.proof,
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


def finalize_fused_kernel(
    source_kernel: Json,
    *,
    component: PreparedFusedComponent,
    tensor_index: Json,
    artifact_payloads: dict[str, bytes],
) -> Json:
    kernel = deepcopy(source_kernel)
    producer = component.transformed.replacement_nodes[0]
    if (
        kernel["node_id"] == producer["id"]
        and producer.get("op") == "quantize_fp8_e4m3_e8m0"
    ):
        return kernel
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
    node = next(
        node
        for node in component.transformed.replacement_nodes
        if node["id"] == kernel["node_id"]
    )
    kernel["physical_execution_contracts"] = build_kernel_physical_execution_contracts(
        node=node,
        circuit=component.transformed.circuit,
        tensor_index=tensor_index,
        kernel=kernel,
        package_dir=Path("."),
        artifact_payloads=artifact_payloads,
    )
    kernel.pop("physical_implementations", None)
    return kernel


def opportunities_from_lowering(
    lowering: Json,
) -> tuple[ParallelProjectionFusionOpportunity, ...]:
    records = lowering.get("regions")
    if not isinstance(records, list) or not records:
        raise ModelCompileError("parallel projection lowering has no component regions")
    opportunities = tuple(_opportunity(record) for record in records)
    region_ids = [
        (opportunity.component_id, opportunity.physical_node_id)
        for opportunity in opportunities
    ]
    if region_ids != sorted(set(region_ids)):
        raise ModelCompileError(
            "parallel projection lowering regions must be sorted and unique"
        )
    return opportunities


def _opportunity(record: Json) -> ParallelProjectionFusionOpportunity:
    raw_region = record["region"]
    region = ParallelProjectionRegion(
        scope_ids=tuple(raw_region["scope_ids"]),  # type: ignore[arg-type]
        source_contract_digests=tuple(  # type: ignore[arg-type]
            raw_region["source_contract_digests"]
        ),
        semantic_source_node_ids=tuple(  # type: ignore[arg-type]
            raw_region["semantic_source_node_ids"]
        ),
        linear_node_ids=tuple(raw_region["linear_node_ids"]),  # type: ignore[arg-type]
        quantizer_node_id=str(raw_region["quantizer_node_id"]),
        boundary_scope_ids=tuple(raw_region.get("boundary_scope_ids", ())),
        boundary_source_contract_digests=tuple(
            raw_region.get("boundary_source_contract_digests", ())
        ),
    )
    upstream_record = record.get("upstream_hyper_fusion")
    upstream = None
    if upstream_record is not None:
        if not isinstance(upstream_record, dict):
            raise ModelCompileError(
                "parallel projection lowering has malformed upstream fusion metadata"
            )
        upstream = HyperNormFusionOpportunity(
            component_id=str(upstream_record["component_id"]),
            regions=tuple(
                HyperNormRegion(
                    scope_id=str(region["scope_id"]),
                    source_contract_digest=str(
                        region["source_contract_digest"]
                    ),
                    semantic_source_node_ids=tuple(  # type: ignore[arg-type]
                        region["semantic_source_node_ids"]
                    ),
                    hyper_node_id=str(region["hyper_node_id"]),
                    norm_node_id=str(region["norm_node_id"]),
                    quantizer_node_id=str(region["quantizer_node_id"]),
                    boundary_scope_ids=tuple(
                        region.get("boundary_scope_ids", ())
                    ),
                    boundary_source_contract_digests=tuple(
                        region.get("boundary_source_contract_digests", ())
                    ),
                )
                for region in upstream_record["regions"]
            ),
            evidence_ids=tuple(upstream_record["evidence_ids"]),
            source_artifact_refs=tuple(
                upstream_record["source_artifact_refs"]
            ),
            manifest_ref=str(upstream_record["manifest_ref"]),
            circuit_ref=str(upstream_record["circuit_ref"]),
            tensor_index_ref=str(upstream_record["tensor_index_ref"]),
            terminal_node_id=str(upstream_record["terminal_node_id"]),
            hidden_size=int(upstream_record["hidden_size"]),
            max_context_activations=int(
                upstream_record["max_context_activations"]
            ),
            compiler_device=deepcopy(upstream_record["compiler_device"]),
            performance_signature=str(
                upstream_record["performance_signature"]
            ),
        )
    return ParallelProjectionFusionOpportunity(
        component_id=str(record["component_id"]),
        region=region,
        evidence_ids=tuple(record["evidence_ids"]),
        source_artifact_refs=tuple(record["source_artifact_refs"]),
        manifest_ref=str(record["manifest_ref"]),
        circuit_ref=str(record["circuit_ref"]),
        tensor_index_ref=str(record["tensor_index_ref"]),
        hidden_size=int(record["hidden_size"]),
        max_context_activations=int(record["max_context_activations"]),
        compiler_device=deepcopy(record["compiler_device"]),
        performance_signature=str(record["performance_signature"]),
        upstream_hyper_fusion=upstream,
    )


def _lowered_component(
    lowering: Json,
    component_id: str,
    physical_node_id: str,
) -> Json:
    matches = [
        record
        for record in lowering["regions"]
        if record.get("component_id") == component_id
        and record.get("physical_node_id") == physical_node_id
    ]
    if len(matches) != 1:
        raise ModelCompileError(
            "parallel projection lowering has no unique region "
            f"{component_id!r}/{physical_node_id!r}"
        )
    return matches[0]


def _lowering(context: CandidateConstructionContext) -> Json:
    lowering = context.target_lowering
    if lowering.get("schema") != TARGET_LOWERING_SCHEMA:
        raise ModelCompileError("parallel projection toolchain received incompatible lowering")
    if lowering.get("candidate_id") != context.candidate["candidate_id"]:
        raise ModelCompileError("parallel projection lowering belongs to another candidate")
    return lowering


def _json_object(payload: bytes, label: str) -> Json:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelCompileError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ModelCompileError(f"{label} must contain a JSON object")
    return value
