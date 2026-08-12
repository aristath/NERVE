from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from nerve.compilation import Json, ModelCompileError
from nerve.model_package_shader_templates import render_shader_source
from nerve.physical_representations import (
    independent_expert_resource_representation_dispatch,
)
from nerve.quantized_transforms import MXFP4_E2M1_FP8_E4M3_BITS
from nerve.representation_optimizer.automation.target import CandidateToolchain
from nerve.representation_optimizer.contracts import contract_digest
from nerve.representation_optimizer.providers.codebook.shaders import compile_spirv
from nerve.representation_optimizer.providers.resident_expansion.artifacts import (
    COMPONENT_FIXTURE_PATH,
    CONVERSATION_FIXTURE_PATH,
    MODEL_LIMITS_PATH,
    PRODUCT_CONVERSATION_FIXTURE_PATH,
    PROOF_PATH,
    component_fixture,
    component_overlay_path,
    conversation_fixture,
    model_limits_fixture,
    product_conversation_fixture,
)
from nerve.representation_optimizer.providers.resident_expansion.contracts import (
    PROOF_SCHEMA,
    TARGET_LOWERING_SCHEMA,
)
from nerve.representation_optimizer.providers.types import ProviderCandidatePlan
from nerve.representation_optimizer.staging.workspace import (
    CandidateConstructionContext,
)
from nerve.resident_representations import mxfp4_to_fp8_resident_derivation


_SHADER_ROOT = Path(__file__).resolve().parents[4] / "runtime-rs" / "shaders"


class ResidentExpansionToolchainResolver:
    def resolve(self, plan: ProviderCandidatePlan) -> CandidateToolchain:
        if (
            plan.provider.provider_id
            != "nerve.exact_resident_expert_parameter_expansion"
            or plan.target_lowering.get("schema") != TARGET_LOWERING_SCHEMA
        ):
            raise ModelCompileError(
                "resident expansion toolchain cannot construct provider "
                f"{plan.provider.provider_id!r}"
            )
        return CandidateToolchain(
            semantic_constructor=ResidentExpansionSemanticConstructor(),
            ordinary_relowerer=ResidentExpansionOrdinaryRelowerer(),
            physical_optimizer=ResidentExpansionPhysicalOptimizer(),
        )


class ResidentExpansionSemanticConstructor:
    def construct_semantic_artifacts(
        self,
        context: CandidateConstructionContext,
    ) -> None:
        lowering = _lowering(context)
        tensor_index = _json_object(
            context.read_source_artifact("tensors.json"),
            "tensors.json",
        )
        tensors = tensor_index.get("tensors")
        if not isinstance(tensors, dict):
            raise ModelCompileError("resident expansion tensor index has no map")
        for region in _regions(lowering):
            for record in region["resident_derivations"]:
                tensor = tensors.get(record["tensor_name"])
                if not isinstance(tensor, dict):
                    raise ModelCompileError(
                        f"resident expansion tensor {record['tensor_name']!r} is missing"
                    )
                derived = mxfp4_to_fp8_resident_derivation(
                    tensor,
                    {
                        "devices": [
                            {
                                "shader_features": record["derivation"][
                                    "required_features"
                                ]
                            }
                        ]
                    },
                )
                if (
                    derived != record["derivation"]
                    or tensor.get("byte_count") != record["source_byte_count"]
                ):
                    raise ModelCompileError(
                        f"resident expansion tensor {record['tensor_name']!r} drifted"
                    )
        context.write_json_artifact(
            PROOF_PATH,
            {
                "schema": PROOF_SCHEMA,
                "candidate_id": lowering["candidate_id"],
                "scope_ids": lowering["scope_ids"],
                "mapping": [
                    {
                        "source_nibble": nibble,
                        "resident_e4m3_bits": bits,
                    }
                    for nibble, bits in enumerate(MXFP4_E2M1_FP8_E4M3_BITS)
                ],
                "regions": [
                    {
                        "component_id": region["source"]["component_id"],
                        "derivation_count": len(region["resident_derivations"]),
                        "derivations_digest": contract_digest(
                            {"resident_derivations": region["resident_derivations"]}
                        ),
                        "source_weight_bytes": sum(
                            int(record["source_byte_count"])
                            for record in region["resident_derivations"]
                        ),
                        "resident_weight_bytes": sum(
                            int(record["derivation"]["resident_byte_count"])
                            for record in region["resident_derivations"]
                        ),
                    }
                    for region in _regions(lowering)
                ],
            },
        )
        representative = _regions(lowering)[0]
        geometry = representative["geometry"]
        context.write_json_artifact(
            COMPONENT_FIXTURE_PATH,
            component_fixture(
                component_id=representative["source"]["component_id"],
                node_ids=tuple(representative["source"]["node_ids"]),
                hidden_size=int(geometry["hidden_size"]),
                intermediate_size=int(geometry["intermediate_size"]),
                expert_count=int(geometry["expert_count"]),
                experts_per_token=int(geometry["experts_per_token"]),
            ),
        )
        context.write_json_artifact(
            CONVERSATION_FIXTURE_PATH,
            conversation_fixture(),
        )
        context.write_json_artifact(
            PRODUCT_CONVERSATION_FIXTURE_PATH,
            product_conversation_fixture(),
        )
        context.write_json_artifact(
            MODEL_LIMITS_PATH,
            model_limits_fixture(int(lowering["runtime"]["max_context_activations"])),
        )
        context.account_transient_bytes(0)


class ResidentExpansionOrdinaryRelowerer:
    def run_ordinary_lowering(
        self,
        context: CandidateConstructionContext,
    ) -> None:
        lowering = _lowering(context)
        manifests = {}
        for region in _regions(lowering):
            manifest_ref = region["source"]["manifest_ref"]
            manifest = manifests.get(manifest_ref)
            if manifest is None:
                manifest = _json_object(
                    context.read_source_artifact(manifest_ref), manifest_ref
                )
                manifests[manifest_ref] = manifest
            _write_region_overlay(context, manifest, region)
        context.account_transient_bytes(0)


class ResidentExpansionPhysicalOptimizer:
    def optimize_physical_artifacts(
        self,
        context: CandidateConstructionContext,
    ) -> None:
        lowering = _lowering(context)
        emitted = set()
        for region in _regions(lowering):
            for replacement in region["shader_replacements"]:
                path = replacement["artifact_path"]
                if path in emitted:
                    continue
                source = render_shader_source(
                    _SHADER_ROOT,
                    replacement["template_name"],
                )
                context.write_artifact(
                    path,
                    compile_spirv(source, replacement["template_name"]),
                )
                emitted.add(path)
        expected = {
            replacement["artifact_path"]
            for region in _regions(lowering)
            for replacement in region["shader_replacements"]
        }
        if emitted != expected:
            raise ModelCompileError(
                "resident expansion did not emit every lowered shader"
            )
        context.account_transient_bytes(0)


def _write_region_overlay(
    context: CandidateConstructionContext,
    manifest: Json,
    region: Json,
) -> None:
    component_id = region["source"]["component_id"]
    component = deepcopy(
        _unique(
            manifest["circuit_graph"]["components"],
            "component_id",
            component_id,
        )
    )
    execution = deepcopy(
        _unique(
            manifest["component_executions"],
            "component_id",
            component_id,
        )
    )
    implementation = "optimized_exact_resident_parameter_expansion_v2"
    component["implementation"] = implementation
    component["circuit"]["implementation"] = implementation
    execution["implementation"] = implementation
    resident_node_ids = sorted(
        {record["node_id"] for record in region["resident_derivations"]}
    )
    if resident_node_ids != sorted(
        {replacement["node_id"] for replacement in region["shader_replacements"]}
    ):
        raise ModelCompileError(
            "resident expansion derivations and shader replacements cover "
            "different execution nodes"
        )
    for node_id in resident_node_ids:
        kernel = _unique(execution["kernels"], "node_id", node_id)
        source_shader_path = str(kernel.get("shader_path", ""))
        if kernel.get("resource_representation_dispatch") != (
            independent_expert_resource_representation_dispatch(source_shader_path)
        ):
            raise ModelCompileError(
                f"resident expansion source kernel {node_id!r} has no exact "
                "MXFP4 resource-representation contract"
            )
        kernel["resource_representation_dispatch"] = (
            independent_expert_resource_representation_dispatch(
                source_shader_path,
                adaptive=True,
            )
        )
    replacement_counts = {}
    for replacement in region["shader_replacements"]:
        source_path = replacement["source_path"]
        target_path = context.artifact_reference(replacement["artifact_path"])
        kernel = _unique(
            execution["kernels"],
            "node_id",
            replacement["node_id"],
        )
        count = 0
        if replacement["execution_kind"] == "scalar":
            if kernel.get("shader_path") == source_path:
                kernel["shader_path"] = target_path
                count = 1
        elif replacement["execution_kind"] == "batch":
            for batch in kernel.get("batch_implementations", []):
                for stage in batch.get("stages", []):
                    if stage.get("shader_path") == source_path:
                        stage["shader_path"] = target_path
                        count += 1
        else:
            raise ModelCompileError(
                "resident expansion lowering has an invalid execution kind"
            )
        if count == 0:
            raise ModelCompileError(
                f"resident expansion source shader {source_path!r} drifted"
            )
        replacement_counts[source_path] = count
    if len(replacement_counts) != len(region["shader_replacements"]):
        raise ModelCompileError(
            "resident expansion lowering contains duplicate shader replacements"
        )
    derivations = [
        {
            "node_id": record["node_id"],
            "parameter_id": record["parameter_id"],
            "derivation": record["derivation"],
        }
        for record in region["resident_derivations"]
    ]
    derivations.sort(key=lambda item: (item["node_id"], item["parameter_id"]))
    overlay_path = region["artifacts"]["overlay_path"]
    if overlay_path != component_overlay_path(component_id):
        raise ModelCompileError(
            "resident expansion overlay path is not component-derived"
        )
    context.write_json_artifact(
        overlay_path,
        {
            "schema": "nerve.optimizer.vulkan_component_overlay.v2",
            "source_component_id": component_id,
            "component": component,
            "execution": execution,
            "resident_derivations": derivations,
        },
    )


def _regions(lowering: Json) -> list[Json]:
    regions = lowering.get("regions")
    if not isinstance(regions, list) or not regions:
        raise ModelCompileError("resident expansion lowering has no component regions")
    component_ids = [
        region.get("source", {}).get("component_id")
        if isinstance(region, dict)
        else None
        for region in regions
    ]
    if any(
        not isinstance(component_id, str) or not component_id
        for component_id in component_ids
    ) or component_ids != sorted(set(component_ids)):
        raise ModelCompileError(
            "resident expansion lowering regions must have sorted unique components"
        )
    return regions


def _lowering(context: CandidateConstructionContext) -> Json:
    lowering = context.target_lowering
    if lowering.get("schema") != TARGET_LOWERING_SCHEMA:
        raise ModelCompileError(
            "resident expansion toolchain received incompatible lowering"
        )
    if lowering.get("candidate_id") != context.candidate["candidate_id"]:
        raise ModelCompileError(
            "resident expansion lowering belongs to another candidate"
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


def _unique(records: object, field: str, value: str) -> Json:
    if not isinstance(records, list):
        raise ModelCompileError(f"resident source {field} records are missing")
    matches = [
        record
        for record in records
        if isinstance(record, dict) and record.get(field) == value
    ]
    if len(matches) != 1:
        raise ModelCompileError(f"resident source has no unique {field}={value!r}")
    return matches[0]
