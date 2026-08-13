from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from nerve.compilation import ModelCompileError
from nerve.representation_optimizer.benchmarking.contracts import BenchmarkWorkload
from nerve.representation_optimizer.contracts import contract_digest, stable_contract_id
from nerve.representation_optimizer.mounting import RuntimeMountPlan
from nerve.representation_optimizer.providers.builtin import (
    BuiltinCandidateToolchainResolver,
    load_builtin_provider_registry,
)
from nerve.representation_optimizer.providers.hyper_norm_fusion.discovery import (
    HyperNormFusionOpportunity,
    HyperNormRegion,
)
from nerve.representation_optimizer.providers.parallel_projection_fusion.artifacts import (
    PROOF_PATH,
    component_overlay_path,
    kernel_artifact_path,
)
from nerve.representation_optimizer.providers.parallel_projection_fusion.contracts import (
    PROOF_VERIFIER_ID,
)
from nerve.representation_optimizer.providers.parallel_projection_fusion.discovery import (
    ParallelProjectionFusionOpportunity,
    ParallelProjectionRegion,
    discover_parallel_projection_fusions,
    discovery_result,
)
from nerve.representation_optimizer.providers.parallel_projection_fusion.physical import (
    ShaderArtifact,
    prepare_fused_component,
    prepare_fused_component_from_documents,
)
from nerve.representation_optimizer.providers.parallel_projection_fusion.proof import (
    ExactParallelProjectionFusionProofVerifier,
)
from nerve.representation_optimizer.providers.parallel_projection_fusion.transformation import (
    fuse_component,
)
from nerve.representation_optimizer.providers.parallel_projection_fusion.provider import (
    ExactParallelProjectionFusionProvider,
)
from nerve.representation_optimizer.providers.parallel_projection_fusion.toolchain import (
    ParallelProjectionFusionToolchainResolver,
)
from nerve.representation_optimizer.providers.source_artifacts import (
    PackageSourceArtifactResolver,
)
from nerve.representation_optimizer.providers.types import EvidenceAssessment
from nerve.representation_optimizer.qualification import QualificationRegime
from nerve.representation_optimizer.representation_ir.contracts import (
    RepresentationGraphDocument,
)
from nerve.representation_optimizer.staging.contracts import (
    CandidateBuildPlan,
    staged_file_digest,
)
from nerve.representation_optimizer.staging.workspace import (
    CandidateConstructionContext,
)
from nerve.representation_optimizer.validation.contracts import (
    ValidationRequirements,
)
from nerve.representation_optimizer.validation.protocols import ProofRequest


_ARTIFACT_DIGEST = f"nerve.optimizer.artifact_sha256.v1:{'0' * 64}"


class _Context(SimpleNamespace):
    def __init__(self, **values) -> None:
        super().__init__(**values)
        self._cache = {}

    def checkpoint(self) -> None:
        return None

    def memoized(self, key, factory):
        if key not in self._cache:
            self._cache[key] = factory()
        return self._cache[key]


def _region(component_id: str, index: int) -> ParallelProjectionRegion:
    scope_ids = tuple(
        stable_contract_id("scope", component_id, index, branch)
        for branch in range(2)
    )
    return ParallelProjectionRegion(
        scope_ids=scope_ids,
        source_contract_digests=tuple(
            contract_digest(
                {"component_id": component_id, "region": index, "branch": branch}
            )
            for branch in range(2)
        ),
        semantic_source_node_ids=(f"wide_{index}", f"narrow_{index}"),
        linear_node_ids=(f"wide_{index}", f"narrow_{index}"),
        quantizer_node_id=f"quantizer_{index}",
    )


def _opportunity(
    component_id: str,
    *,
    region_index: int = 0,
    performance_signature: str = "performance_class_shared",
) -> ParallelProjectionFusionOpportunity:
    return ParallelProjectionFusionOpportunity(
        component_id=component_id,
        region=_region(component_id, region_index),
        evidence_ids=(stable_contract_id("evidence", component_id, region_index),),
        source_artifact_refs=(
            f"lowered/{component_id}/circuit.json",
            "tensors.json",
            "vulkan_resident_package.json",
        ),
        manifest_ref="vulkan_resident_package.json",
        circuit_ref=f"lowered/{component_id}/circuit.json",
        tensor_index_ref="tensors.json",
        hidden_size=4096,
        max_context_activations=131_072,
        compiler_device={
            "shader_features": ["shader_float8", "shader_int8"],
            "max_compute_work_group_invocations": 1024,
            "max_compute_work_group_size_x": 1024,
            "subgroup_operations": ["arithmetic"],
            "subgroup_size": 64,
            "subgroup_compute_supported": True,
        },
        performance_signature=performance_signature,
    )


def _three_way_opportunity(component_id: str) -> ParallelProjectionFusionOpportunity:
    node_ids = ("wide_0", "narrow_0", "auxiliary_0")
    region = ParallelProjectionRegion(
        scope_ids=tuple(
            stable_contract_id("scope", component_id, 0, branch)
            for branch in range(3)
        ),
        source_contract_digests=tuple(
            contract_digest(
                {"component_id": component_id, "region": 0, "branch": branch}
            )
            for branch in range(3)
        ),
        semantic_source_node_ids=node_ids,
        linear_node_ids=node_ids,
        quantizer_node_id="quantizer_0",
    )
    base = _opportunity(component_id)
    return ParallelProjectionFusionOpportunity(
        component_id=component_id,
        region=region,
        evidence_ids=base.evidence_ids,
        source_artifact_refs=base.source_artifact_refs,
        manifest_ref=base.manifest_ref,
        circuit_ref=base.circuit_ref,
        tensor_index_ref=base.tensor_index_ref,
        hidden_size=base.hidden_size,
        max_context_activations=base.max_context_activations,
        compiler_device=base.compiler_device,
        performance_signature="performance_class_three_way",
    )


def _combined_opportunity(component_id: str) -> ParallelProjectionFusionOpportunity:
    projection = _opportunity(component_id)
    boundary_scope_ids = tuple(
        stable_contract_id("projection_boundary", component_id, branch)
        for branch in range(2)
    )
    boundary_digests = tuple(
        contract_digest(
            {"component_id": component_id, "boundary": branch}
        )
        for branch in range(2)
    )
    projection_region = ParallelProjectionRegion(
        scope_ids=projection.region.scope_ids,
        source_contract_digests=projection.region.source_contract_digests,
        semantic_source_node_ids=projection.region.semantic_source_node_ids,
        linear_node_ids=projection.region.linear_node_ids,
        quantizer_node_id=projection.region.quantizer_node_id,
        boundary_scope_ids=boundary_scope_ids,
        boundary_source_contract_digests=boundary_digests,
    )
    region = HyperNormRegion(
        scope_id=stable_contract_id("hyper_scope", component_id),
        source_contract_digest=contract_digest(
            {"component_id": component_id, "region": "hyper_norm"}
        ),
        semantic_source_node_ids=("hyper_reduce", "operator_norm"),
        hyper_node_id="hyper_function__hyper_sinkhorn__hyper_reduce",
        norm_node_id="operator_norm",
        quantizer_node_id="quantizer_0",
        boundary_scope_ids=boundary_scope_ids,
        boundary_source_contract_digests=boundary_digests,
    )
    upstream = HyperNormFusionOpportunity(
        component_id=component_id,
        regions=(region,),
        evidence_ids=(stable_contract_id("hyper_evidence", component_id),),
        source_artifact_refs=projection.source_artifact_refs,
        manifest_ref=projection.manifest_ref,
        circuit_ref=projection.circuit_ref,
        tensor_index_ref=projection.tensor_index_ref,
        terminal_node_id="narrow_0",
        hidden_size=projection.hidden_size,
        max_context_activations=projection.max_context_activations,
        compiler_device=projection.compiler_device,
        performance_signature="hyper_performance_class",
    )
    return ParallelProjectionFusionOpportunity(
        component_id=component_id,
        region=projection_region,
        evidence_ids=tuple(
            sorted(set(projection.evidence_ids).union(upstream.evidence_ids))
        ),
        source_artifact_refs=projection.source_artifact_refs,
        manifest_ref=projection.manifest_ref,
        circuit_ref=projection.circuit_ref,
        tensor_index_ref=projection.tensor_index_ref,
        hidden_size=projection.hidden_size,
        max_context_activations=projection.max_context_activations,
        compiler_device=projection.compiler_device,
        performance_signature="combined_performance_class",
        upstream_hyper_fusion=upstream,
    )


def _products(monkeypatch, opportunities):
    prepared = SimpleNamespace(
        shader_artifacts=(
            ShaderArtifact(
                "kernels/parallel_projection/fused_decode.spv",
                "fused_decode.comp",
            ),
            ShaderArtifact(
                "kernels/parallel_projection/fused_batch.spv",
                "fused_batch.comp",
            ),
        )
    )
    monkeypatch.setattr(
        "nerve.representation_optimizer.providers.parallel_projection_fusion.provider."
        "discover_parallel_projection_fusions",
        lambda _context: tuple(opportunities),
    )
    monkeypatch.setattr(
        "nerve.representation_optimizer.providers.parallel_projection_fusion.provider."
        "prepare_fused_component",
        lambda _context, _opportunity: prepared,
    )
    monkeypatch.setattr(
        "nerve.representation_optimizer.providers.parallel_projection_fusion.provider."
        "source_inputs",
        lambda _context, opportunity: [
            {"path": path, "digest": _ARTIFACT_DIGEST}
            for path in opportunity.source_artifact_refs
        ],
    )
    context = _Context(
        hardware_profile={"capability_class": "capability_fixture"},
        qualification_regime=QualificationRegime(),
    )
    provider = ExactParallelProjectionFusionProvider()
    evidence = EvidenceAssessment(
        accepted=True,
        evidence_ids=tuple(
            sorted(
                {
                    evidence_id
                    for opportunity in opportunities
                    for evidence_id in opportunity.evidence_ids
                }
            )
        ),
        facts={"exact": True},
        reasons=("fixture evidence",),
    )
    candidates = provider.synthesize_candidates(context, evidence)
    return provider, context, candidates


def _noncontiguous_projection_circuits():
    source = {
        "schema": "nerve.stream_circuit.v1",
        "source": {"component_id": "block_alpha"},
        "boundary": {
            "inputs": [{"id": "input", "source": "x", "shape": [4, 4096]}],
            "outputs": [
                {"id": "wide", "source": "wide_normed"},
                {"id": "narrow", "source": "narrow"},
            ],
        },
        "state_ports": [],
        "parameters": {
            "refs": {
                "wide_weight": {"tensor": "wide.weight"},
                "wide_scale": {"tensor": "wide.scale"},
                "narrow_weight": {"tensor": "narrow.weight"},
                "narrow_scale": {"tensor": "narrow.scale"},
                "norm_weight": {"tensor": "norm.weight"},
            }
        },
        "behavioral_error_contract": {"mode": "source_reference_circuit"},
        "nodes": [
            {
                "id": "wide_0",
                "op": "linear",
                "inputs": ["x"],
                "outputs": ["wide"],
                "params": ["wide_weight", "wide_scale"],
            },
            {
                "id": "wide_consumer",
                "op": "rms_norm",
                "inputs": ["wide"],
                "outputs": ["wide_normed"],
                "params": ["norm_weight"],
                "attrs": {"eps": 1e-6, "weight_offset": 0.0},
            },
            {
                "id": "narrow_0",
                "op": "linear",
                "inputs": ["x"],
                "outputs": ["narrow"],
                "params": ["narrow_weight", "narrow_scale"],
            },
        ],
    }
    compiled = deepcopy(source)
    physical_contract = "bf16_blockwise_fp8_e4m3_e8m0_scale_f32.v1"
    compiled["nodes"] = [
        {
            "id": "quantizer_0",
            "op": "quantize_fp8_e4m3_e8m0",
            "inputs": ["x"],
            "outputs": ["x_fp8", "x_scale"],
            "attrs": {
                "physical_representation_contract": physical_contract,
                "consumer_node_ids": ["wide_0", "narrow_0"],
                "semantic_source_node_ids": ["wide_0", "narrow_0"],
                "element_count": 4096,
                "block_columns": 128,
                "output_element_bytes": [1, 4],
            },
        },
        {
            "id": "wide_0",
            "op": "linear",
            "inputs": ["x_fp8", "x_scale"],
            "outputs": ["wide"],
            "params": ["wide_weight", "wide_scale"],
            "attrs": {
                "output_element_bytes": [2],
                "physical_input_contract": physical_contract,
                "physical_input_provider_id": "quantizer_0",
                "physical_input_source_node_ids": ["wide_0"],
                "physical_logical_inputs": ["x"],
            },
        },
        deepcopy(source["nodes"][1]),
        {
            "id": "narrow_0",
            "op": "linear",
            "inputs": ["x_fp8", "x_scale"],
            "outputs": ["narrow"],
            "params": ["narrow_weight", "narrow_scale"],
            "attrs": {
                "output_element_bytes": [2],
                "physical_input_contract": physical_contract,
                "physical_input_provider_id": "quantizer_0",
                "physical_input_source_node_ids": ["narrow_0"],
                "physical_logical_inputs": ["x"],
            },
        },
    ]
    return source, compiled


def _three_way_projection_circuits():
    source, compiled = _noncontiguous_projection_circuits()
    source["parameters"]["refs"].update(
        {
            "auxiliary_weight": {"tensor": "auxiliary.weight"},
            "auxiliary_scale": {"tensor": "auxiliary.scale"},
        }
    )
    source["boundary"]["outputs"].append(
        {"id": "auxiliary", "source": "auxiliary"}
    )
    source["nodes"].append(
        {
            "id": "auxiliary_0",
            "op": "linear",
            "inputs": ["x"],
            "outputs": ["auxiliary"],
            "params": ["auxiliary_weight", "auxiliary_scale"],
        }
    )
    compiled["parameters"] = deepcopy(source["parameters"])
    compiled["boundary"] = deepcopy(source["boundary"])
    helper = compiled["nodes"][0]
    helper["attrs"]["consumer_node_ids"].append("auxiliary_0")
    helper["attrs"]["semantic_source_node_ids"].append("auxiliary_0")
    physical_contract = helper["attrs"]["physical_representation_contract"]
    compiled["nodes"].append(
        {
            "id": "auxiliary_0",
            "op": "linear",
            "inputs": ["x_fp8", "x_scale"],
            "outputs": ["auxiliary"],
            "params": ["auxiliary_weight", "auxiliary_scale"],
            "attrs": {
                "output_element_bytes": [2],
                "physical_input_contract": physical_contract,
                "physical_input_provider_id": "quantizer_0",
                "physical_input_source_node_ids": ["auxiliary_0"],
                "physical_logical_inputs": ["x"],
            },
        }
    )
    return source, compiled


def _combined_projection_documents():
    physical_contract = "bf16_blockwise_fp8_e4m3_e8m0_scale_f32.v1"
    parameters = {
        "refs": {
            "hyper_function": {"tensor": "hyper.function"},
            "hyper_scale": {"tensor": "hyper.scale"},
            "hyper_base": {"tensor": "hyper.base"},
            "operator_norm_weight": {"tensor": "operator_norm.weight"},
            "wide_weight": {"tensor": "wide.weight"},
            "wide_scale": {"tensor": "wide.scale"},
            "narrow_weight": {"tensor": "narrow.weight"},
            "narrow_scale": {"tensor": "narrow.scale"},
            "norm_weight": {"tensor": "norm.weight"},
        }
    }
    boundary = {
        "inputs": [{"id": "input", "source": "x", "shape": [4, 4096]}],
        "outputs": [
            {"id": "wide", "source": "wide_normed"},
            {"id": "narrow", "source": "narrow"},
            {"id": "hyper_post", "source": "hyper_post"},
            {"id": "hyper_combination", "source": "hyper_combination"},
        ],
    }
    source = {
        "schema": "nerve.stream_circuit.v1",
        "source": {"component_id": "block_alpha"},
        "boundary": deepcopy(boundary),
        "state_ports": [],
        "parameters": deepcopy(parameters),
        "behavioral_error_contract": {"mode": "source_reference_circuit"},
        "nodes": [
            {
                "id": "hyper_function",
                "op": "normalized_linear",
                "inputs": ["x"],
                "outputs": ["hyper_mixes"],
                "params": ["hyper_function"],
                "attrs": {
                    "normalization": "root_mean_square",
                    "normalization_epsilon": 1e-6,
                    "multiplicity": 4,
                    "output_element_bytes": [4],
                },
            },
            {
                "id": "hyper_sinkhorn",
                "op": "hyper_connection_sinkhorn",
                "inputs": ["hyper_mixes"],
                "outputs": ["hyper_pre", "hyper_post", "hyper_combination"],
                "params": ["hyper_scale", "hyper_base"],
                "attrs": {
                    "multiplicity": 4,
                    "sinkhorn_iterations": 20,
                    "epsilon": 1e-6,
                    "output_element_bytes": [4, 4, 4],
                },
            },
            {
                "id": "hyper_reduce",
                "op": "hyper_connection_reduce",
                "inputs": ["x", "hyper_pre"],
                "outputs": ["operator_input"],
                "attrs": {"multiplicity": 4, "output_element_bytes": [2]},
            },
            {
                "id": "operator_norm",
                "op": "rms_norm",
                "inputs": ["operator_input"],
                "outputs": ["operator_norm_out"],
                "params": ["operator_norm_weight"],
                "attrs": {"eps": 1e-6, "weight_offset": 0.0},
            },
            {
                "id": "wide_0",
                "op": "linear",
                "inputs": ["operator_norm_out"],
                "outputs": ["wide"],
                "params": ["wide_weight", "wide_scale"],
            },
            {
                "id": "wide_consumer",
                "op": "rms_norm",
                "inputs": ["wide"],
                "outputs": ["wide_normed"],
                "params": ["norm_weight"],
                "attrs": {"eps": 1e-6, "weight_offset": 0.0},
            },
            {
                "id": "narrow_0",
                "op": "linear",
                "inputs": ["operator_norm_out"],
                "outputs": ["narrow"],
                "params": ["narrow_weight", "narrow_scale"],
            },
        ],
    }
    compiled = deepcopy(source)
    compiled["nodes"] = [
        {
            "id": "hyper_function__hyper_sinkhorn__hyper_reduce",
            "op": "hyper_connection_pre",
            "inputs": ["x"],
            "outputs": ["operator_input", "hyper_post", "hyper_combination"],
            "params": ["hyper_function", "hyper_scale", "hyper_base"],
            "attrs": {
                "compiled_from": [
                    "hyper_function",
                    "hyper_sinkhorn",
                    "hyper_reduce",
                ],
                "multiplicity": 4,
                "normalization_epsilon": 1e-6,
                "sinkhorn_iterations": 20,
                "epsilon": 1e-6,
                "intermediate_rounding": "BF16",
                "output_element_bytes": [2, 4, 4],
            },
        },
        {
            "id": "operator_norm",
            "op": "rms_norm",
            "inputs": ["operator_input"],
            "outputs": ["operator_norm_out"],
            "params": ["operator_norm_weight"],
            "attrs": {
                "eps": 1e-6,
                "weight_offset": 0.0,
                "output_element_bytes": [2],
            },
        },
        {
            "id": "quantizer_0",
            "op": "quantize_fp8_e4m3_e8m0",
            "inputs": ["operator_norm_out"],
            "outputs": ["x_fp8", "x_scale"],
            "attrs": {
                "physical_representation_contract": physical_contract,
                "consumer_node_ids": ["wide_0", "narrow_0"],
                "semantic_source_node_ids": ["wide_0", "narrow_0"],
                "element_count": 4096,
                "block_columns": 128,
                "output_element_bytes": [1, 4],
            },
        },
        {
            "id": "wide_0",
            "op": "linear",
            "inputs": ["x_fp8", "x_scale"],
            "outputs": ["wide"],
            "params": ["wide_weight", "wide_scale"],
            "attrs": {
                "output_element_bytes": [2],
                "physical_input_contract": physical_contract,
                "physical_input_provider_id": "quantizer_0",
                "physical_input_source_node_ids": ["wide_0"],
                "physical_logical_inputs": ["operator_norm_out"],
            },
        },
        deepcopy(source["nodes"][5]),
        {
            "id": "narrow_0",
            "op": "linear",
            "inputs": ["x_fp8", "x_scale"],
            "outputs": ["narrow"],
            "params": ["narrow_weight", "narrow_scale"],
            "attrs": {
                "output_element_bytes": [2],
                "physical_input_contract": physical_contract,
                "physical_input_provider_id": "quantizer_0",
                "physical_input_source_node_ids": ["narrow_0"],
                "physical_logical_inputs": ["operator_norm_out"],
            },
        },
    ]
    tensor_index = {
        "tensors": {
            "hyper.function": {
                "dtype": "F32",
                "shape": [24, 16384],
                "layout": "row_major",
            },
            "hyper.scale": {
                "dtype": "F32",
                "shape": [3],
                "layout": "row_major",
            },
            "hyper.base": {
                "dtype": "F32",
                "shape": [24],
                "layout": "row_major",
            },
            "operator_norm.weight": {
                "dtype": "BF16",
                "shape": [4096],
                "layout": "row_major",
            },
            "wide.weight": {
                "dtype": "F8_E4M3",
                "shape": [1024, 4096],
                "layout": "row_major",
            },
            "wide.scale": {
                "dtype": "F8_E8M0",
                "shape": [8, 32],
                "layout": "row_major",
            },
            "narrow.weight": {
                "dtype": "F8_E4M3",
                "shape": [512, 4096],
                "layout": "row_major",
            },
            "narrow.scale": {
                "dtype": "F8_E8M0",
                "shape": [4, 32],
                "layout": "row_major",
            },
            "norm.weight": {
                "dtype": "BF16",
                "shape": [1024],
                "layout": "row_major",
            },
        }
    }
    manifest = {
        "tensor_index_path": "tensors.json",
        "circuit_graph": {
            "components": [{"component_id": "block_alpha", "circuit": compiled}]
        },
        "component_executions": [
            {
                "component_id": "block_alpha",
                "kernels": [
                    {
                        "node_id": node["id"],
                        "execution_index": index,
                        "op": node["op"],
                        "shader_path": f"shaders/{node['id']}.spv",
                        "local_size_x": 1024,
                        "workgroup_count_x": 1,
                    }
                    for index, node in enumerate(compiled["nodes"])
                ],
            }
        ],
    }
    return source, compiled, manifest, tensor_index


class _MemoryArtifact:
    def __init__(self, path: str) -> None:
        self.path = path

    def source_input(self):
        return {"path": self.path, "digest": _ARTIFACT_DIGEST}


class _MemoryResolver:
    def __init__(self, documents) -> None:
        self.documents = documents

    def read_path(self, path):
        return json.dumps(self.documents[path]).encode()

    def resolve_path(self, path):
        if path not in self.documents:
            raise ModelCompileError(f"unknown fixture artifact {path}")
        return _MemoryArtifact(path)


def _discovery_context(
    *,
    reverse_consumers: bool = False,
    three_way: bool = False,
):
    source, compiled = (
        _three_way_projection_circuits()
        if three_way
        else _noncontiguous_projection_circuits()
    )
    linear_node_ids = (
        ("wide_0", "narrow_0", "auxiliary_0")
        if three_way
        else ("wide_0", "narrow_0")
    )
    if reverse_consumers:
        compiled["nodes"][0]["attrs"]["consumer_node_ids"].reverse()
    source["parameters"]["refs"]["input_norm_weight"] = {
        "tensor": "input_norm.weight"
    }
    source["nodes"].insert(
        0,
        {
            "id": "input_norm",
            "op": "rms_norm",
            "inputs": ["x"],
            "outputs": ["x_norm"],
            "params": ["input_norm_weight"],
            "attrs": {"eps": 1e-6, "weight_offset": 0.0},
        },
    )
    for node in source["nodes"]:
        if node["id"] in linear_node_ids:
            node["inputs"] = ["x_norm"]
    compiled["parameters"] = deepcopy(source["parameters"])
    compiled["nodes"].insert(0, deepcopy(source["nodes"][0]))
    helper = next(node for node in compiled["nodes"] if node["id"] == "quantizer_0")
    helper["inputs"] = ["x_norm"]
    for node in compiled["nodes"]:
        if node["id"] in linear_node_ids:
            node["attrs"]["physical_logical_inputs"] = ["x_norm"]
    manifest = {
        "tensor_index_path": "tensors.json",
        "max_context_activations": 131_072,
        "circuit_graph": {
            "components": [
                {"component_id": "block_alpha", "circuit": compiled}
            ]
        },
        "component_executions": [
            {
                "component_id": "block_alpha",
                "kernels": [
                    {
                        "node_id": node_id,
                        "execution_index": index,
                        "shader_path": f"shaders/{node_id}.spv",
                        "local_size_x": 1024,
                        "workgroup_count_x": 64,
                        "batch_implementations": [],
                    }
                    for index, node_id in enumerate(
                        ("quantizer_0", *linear_node_ids)
                    )
                ],
            }
        ],
    }
    operator_scopes = tuple(
        {
            "scope_id": f"scope_{node_id}",
            "kind": "operator",
            "members": {
                "component_ids": ["block_alpha"],
                "source_node_ids": [f"block_alpha/{node_id}"],
            },
            "extensions": {"semantic_roles": ["linear"]},
        }
        for node_id in linear_node_ids
    )
    boundary_scopes = tuple(
        {
            "scope_id": f"scope_input_norm_to_{node_id}",
            "kind": "representation_island",
            "members": {
                "component_ids": ["block_alpha"],
                "source_node_ids": [
                    "block_alpha/input_norm",
                    f"block_alpha/{node_id}",
                ],
            },
            "extensions": {},
        }
        for node_id in linear_node_ids
    )
    scopes = (*operator_scopes, *boundary_scopes)
    contracts = tuple(
        {
            "scope_id": scope["scope_id"],
            "semantic_role": (
                "linear"
                if scope["kind"] == "operator"
                else "adjacent semantic representation boundary"
            ),
            "contract_digest": contract_digest(scope),
            "exact_reference": {
                "artifact_refs": ["lowered/block_alpha/circuit.json"]
            },
        }
        for scope in scopes
    )
    evidence = tuple(
        {
            "scope_id": scope["scope_id"],
            "evidence_id": f"evidence_{scope['scope_id']}",
            "claims": [{"status": "supported"}],
        }
        for scope in scopes
    )
    return _Context(
        scopes=scopes,
        source_contracts=contracts,
        evidence=evidence,
        hardware_profile={
            "hardware_identity": {"device_kind": "gpu"},
            "provenance": {"api": "vulkan"},
            "capability_extensions": {
                "vulkan_compiler_capabilities": {
                    "shader_features": ["shader_float8", "shader_int8"],
                    "max_compute_work_group_invocations": 1024,
                    "max_compute_work_group_size_x": 1024,
                    "subgroup_operations": ["arithmetic"],
                    "subgroup_size": 64,
                    "subgroup_compute_supported": True,
                }
            },
        },
        source_artifacts=_MemoryResolver(
            {
                "vulkan_resident_package.json": manifest,
                "tensors.json": {"tensors": {}},
                "lowered/block_alpha/circuit.json": source,
            }
        ),
        scope_ids=tuple(scope["scope_id"] for scope in scopes),
    )


def _boundary_input_discovery_context():
    context = _discovery_context()
    source = context.source_artifacts.documents[
        "lowered/block_alpha/circuit.json"
    ]
    source["nodes"] = [
        node for node in source["nodes"] if node["id"] != "input_norm"
    ]
    for node in source["nodes"]:
        if node["id"] in {"wide_0", "narrow_0"}:
            node["inputs"] = ["x"]
    compiled = context.source_artifacts.documents[
        "vulkan_resident_package.json"
    ]["circuit_graph"]["components"][0]["circuit"]
    compiled["nodes"] = [
        node for node in compiled["nodes"] if node["id"] != "input_norm"
    ]
    for node in compiled["nodes"]:
        if node["id"] == "quantizer_0":
            node["inputs"] = ["x"]
        elif node["id"] in {"wide_0", "narrow_0"}:
            node["attrs"]["physical_logical_inputs"] = ["x"]
    retained_scope_ids = {
        scope["scope_id"] for scope in context.scopes if scope["kind"] == "operator"
    }
    context.scopes = tuple(
        scope for scope in context.scopes if scope["scope_id"] in retained_scope_ids
    )
    context.source_contracts = tuple(
        contract
        for contract in context.source_contracts
        if contract["scope_id"] in retained_scope_ids
    )
    context.evidence = tuple(
        record
        for record in context.evidence
        if record["scope_id"] in retained_scope_ids
    )
    context.scope_ids = tuple(scope["scope_id"] for scope in context.scopes)
    return context


def test_discovery_is_structural_and_normalizes_consumer_order() -> None:
    opportunities = discover_parallel_projection_fusions(
        _discovery_context(reverse_consumers=True)
    )

    assert len(opportunities) == 1
    assert opportunities[0].component_id == "block_alpha"
    assert opportunities[0].region.linear_node_ids == ("wide_0", "narrow_0")
    assert opportunities[0].region.semantic_source_node_ids == (
        "wide_0",
        "narrow_0",
    )
    assert len(opportunities[0].region.boundary_scope_ids) == 2


def test_discovery_keeps_true_component_input_fanout_without_ceremonial_boundary() -> None:
    opportunities = discover_parallel_projection_fusions(
        _boundary_input_discovery_context()
    )

    assert len(opportunities) == 1
    assert opportunities[0].region.boundary_scope_ids == ()


def test_discovery_owns_every_catalogued_upstream_boundary() -> None:
    context = _discovery_context()
    missing_scope_id = "scope_input_norm_to_narrow_0"
    context.scopes = tuple(
        scope for scope in context.scopes if scope["scope_id"] != missing_scope_id
    )
    context.source_contracts = tuple(
        contract
        for contract in context.source_contracts
        if contract["scope_id"] != missing_scope_id
    )
    context.evidence = tuple(
        record
        for record in context.evidence
        if record["scope_id"] != missing_scope_id
    )
    context.scope_ids = tuple(scope["scope_id"] for scope in context.scopes)

    opportunities = discover_parallel_projection_fusions(context)

    assert len(opportunities) == 1
    assert opportunities[0].region.boundary_scope_ids == (
        "scope_input_norm_to_wide_0",
    )


def test_discovery_rejects_ambiguous_catalogued_boundary_ownership() -> None:
    context = _discovery_context()
    original = next(
        scope
        for scope in context.scopes
        if scope["scope_id"] == "scope_input_norm_to_wide_0"
    )
    duplicate = deepcopy(original)
    duplicate["scope_id"] = "scope_duplicate_input_norm_to_wide_0"
    duplicate_contract = {
        "scope_id": duplicate["scope_id"],
        "semantic_role": "adjacent semantic representation boundary",
        "contract_digest": contract_digest(duplicate),
        "exact_reference": {
            "artifact_refs": ["lowered/block_alpha/circuit.json"]
        },
    }
    context.scopes = (*context.scopes, duplicate)
    context.source_contracts = (*context.source_contracts, duplicate_contract)
    context.evidence = (
        *context.evidence,
        {
            "scope_id": duplicate["scope_id"],
            "evidence_id": "evidence_duplicate_boundary",
            "claims": [{"status": "supported"}],
        },
    )
    context.scope_ids = tuple(scope["scope_id"] for scope in context.scopes)

    assert discover_parallel_projection_fusions(context) == ()


def test_discovery_admits_a_complete_three_way_fanout() -> None:
    opportunities = discover_parallel_projection_fusions(
        _discovery_context(three_way=True)
    )

    assert len(opportunities) == 1
    assert opportunities[0].region.linear_node_ids == (
        "wide_0",
        "narrow_0",
        "auxiliary_0",
    )
    assert opportunities[0].physical_node_id == "wide_0__narrow_0__auxiliary_0"


def test_discovery_joins_an_exact_upstream_producer_without_model_identity(
    monkeypatch,
) -> None:
    context = _discovery_context()
    combined = _combined_opportunity("block_alpha")
    boundary_records = [
        (scope, contract)
        for scope, contract in zip(
            context.scopes,
            context.source_contracts,
            strict=True,
        )
        if scope["kind"] == "representation_island"
    ]
    upstream_region = replace(
        combined.upstream_hyper_fusion.regions[0],
        boundary_scope_ids=tuple(
            scope["scope_id"] for scope, _contract in boundary_records
        ),
        boundary_source_contract_digests=tuple(
            contract["contract_digest"] for _scope, contract in boundary_records
        ),
    )
    upstream = replace(
        combined.upstream_hyper_fusion,
        regions=(upstream_region,),
    )
    monkeypatch.setattr(
        "nerve.representation_optimizer.providers.parallel_projection_fusion."
        "discovery.discover_hyper_norm_fusions",
        lambda _context: (upstream,),
    )

    opportunities = discover_parallel_projection_fusions(context)

    assert len(opportunities) == 2
    ordinary = next(
        opportunity
        for opportunity in opportunities
        if not opportunity.combines_upstream_producer
    )
    joined = next(
        opportunity
        for opportunity in opportunities
        if opportunity.combines_upstream_producer
    )
    assert ordinary.region == joined.region
    assert joined.upstream_hyper_fusion == upstream
    assert len(ordinary.scope_ids) == 4
    assert len(joined.scope_ids) == 5
    assert set(ordinary.region.boundary_scope_ids) <= set(
        joined.upstream_hyper_fusion.scope_ids
    )
    assert joined.performance_signature != ordinary.performance_signature


def test_discovery_rejects_an_ambiguous_upstream_producer(monkeypatch) -> None:
    combined = _combined_opportunity("block_alpha")
    monkeypatch.setattr(
        "nerve.representation_optimizer.providers.parallel_projection_fusion."
        "discovery.discover_hyper_norm_fusions",
        lambda _context: (
            combined.upstream_hyper_fusion,
            combined.upstream_hyper_fusion,
        ),
    )

    with pytest.raises(ModelCompileError, match="ambiguous upstream fusion"):
        discover_parallel_projection_fusions(_discovery_context())


def test_discovery_rejects_a_branch_without_the_shared_physical_contract() -> None:
    context = _discovery_context()
    manifest = context.source_artifacts.documents["vulkan_resident_package.json"]
    narrow = next(
        node
        for node in manifest["circuit_graph"]["components"][0]["circuit"]["nodes"]
        if node["id"] == "narrow_0"
    )
    narrow["attrs"]["physical_input_contract"] = "different_contract"

    assert discover_parallel_projection_fusions(context) == ()


def test_fusion_replaces_noncontiguous_shared_input_linears_exactly() -> None:
    source, compiled = _noncontiguous_projection_circuits()

    fused = fuse_component(
        opportunity=_opportunity("block_alpha"),
        source_circuit=source,
        compiled_circuit=compiled,
        tensor_index={},
    )

    assert [node["id"] for node in fused.source_nodes] == [
        "quantizer_0",
        "wide_0",
        "narrow_0",
    ]
    assert [node["id"] for node in fused.replacement_nodes] == [
        "quantizer_0",
        "wide_0__narrow_0",
    ]
    assert [node["id"] for node in fused.circuit["nodes"]] == [
        "quantizer_0",
        "wide_0__narrow_0",
        "wide_consumer",
    ]
    assert fused.replacement_nodes[0]["attrs"]["consumer_node_ids"] == [
        "wide_0__narrow_0"
    ]
    assert fused.proof["candidate_kind"] == "exact_reference"
    assert fused.proof["status"] == "passed"


def test_fusion_rejects_incomplete_shared_input_consumer_set() -> None:
    source, compiled = _noncontiguous_projection_circuits()
    compiled["nodes"][0]["attrs"]["consumer_node_ids"] = ["wide_0"]

    with pytest.raises(ModelCompileError, match="helper contract drifted"):
        fuse_component(
            opportunity=_opportunity("block_alpha"),
            source_circuit=source,
            compiled_circuit=compiled,
            tensor_index={},
        )


def test_physical_lowering_emits_exact_scalar_and_causal_batch_kernels() -> None:
    source, compiled = _noncontiguous_projection_circuits()
    manifest = {
        "circuit_graph": {
            "components": [
                {"component_id": "block_alpha", "circuit": compiled}
            ]
        },
        "component_executions": [
            {
                "component_id": "block_alpha",
                "kernels": [
                    {
                        "node_id": node_id,
                        "execution_index": index,
                        "op": next(
                            node["op"]
                            for node in compiled["nodes"]
                            if node["id"] == node_id
                        ),
                        "shader_path": f"shaders/{node_id}.spv",
                        "local_size_x": 1024,
                        "workgroup_count_x": 1,
                    }
                    for index, node_id in enumerate(
                        ("quantizer_0", "wide_0", "narrow_0")
                    )
                ],
            }
        ],
    }
    tensor_index = {
        "tensors": {
            "wide.weight": {
                "dtype": "F8_E4M3",
                "shape": [128, 4096],
                "layout": "row_major",
            },
            "wide.scale": {
                "dtype": "F8_E8M0",
                "shape": [1, 32],
                "layout": "row_major",
            },
            "narrow.weight": {
                "dtype": "F8_E4M3",
                "shape": [256, 4096],
                "layout": "row_major",
            },
            "narrow.scale": {
                "dtype": "F8_E8M0",
                "shape": [2, 32],
                "layout": "row_major",
            },
            "norm.weight": {
                "dtype": "BF16",
                "shape": [128],
                "layout": "row_major",
            },
        }
    }

    prepared = prepare_fused_component_from_documents(
        opportunity=_opportunity("block_alpha"),
        manifest=manifest,
        tensor_index=tensor_index,
        source_circuit=source,
    )

    fused_kernel = prepared.replacement_kernels[1]
    assert fused_kernel["node_id"] == "wide_0__narrow_0"
    assert fused_kernel["shader_path"].endswith(
        "parallel_linear_2way_prequant_fp8_e4m3_se8m0_"
        "b128x128_4096x128_256.spv"
    )
    assert [
        implementation["lane_tile_width"]
        for implementation in fused_kernel["batch_implementations"]
    ] == [2, 4, 8, 16]
    assert all(
        stage["shader_path"].startswith("kernels/parallel_projection/")
        for implementation in fused_kernel["batch_implementations"]
        for stage in implementation["stages"]
    )


def test_three_way_physical_lowering_preserves_every_independent_branch() -> None:
    source, compiled = _three_way_projection_circuits()
    manifest = {
        "circuit_graph": {
            "components": [{"component_id": "block_alpha", "circuit": compiled}]
        },
        "component_executions": [
            {
                "component_id": "block_alpha",
                "kernels": [
                    {
                        "node_id": node_id,
                        "execution_index": index,
                        "op": next(
                            node["op"]
                            for node in compiled["nodes"]
                            if node["id"] == node_id
                        ),
                        "shader_path": f"shaders/{node_id}.spv",
                        "local_size_x": 1024,
                        "workgroup_count_x": 1,
                    }
                    for index, node_id in enumerate(
                        ("quantizer_0", "wide_0", "narrow_0", "auxiliary_0")
                    )
                ],
            }
        ],
    }
    tensor_index = {
        "tensors": {
            "wide.weight": {
                "dtype": "F8_E4M3",
                "shape": [128, 4096],
                "layout": "row_major",
            },
            "wide.scale": {
                "dtype": "F8_E8M0",
                "shape": [1, 32],
                "layout": "row_major",
            },
            "narrow.weight": {
                "dtype": "F8_E4M3",
                "shape": [256, 4096],
                "layout": "row_major",
            },
            "narrow.scale": {
                "dtype": "F8_E8M0",
                "shape": [2, 32],
                "layout": "row_major",
            },
            "auxiliary.weight": {
                "dtype": "F8_E4M3",
                "shape": [384, 4096],
                "layout": "row_major",
            },
            "auxiliary.scale": {
                "dtype": "F8_E8M0",
                "shape": [3, 32],
                "layout": "row_major",
            },
            "norm.weight": {
                "dtype": "BF16",
                "shape": [128],
                "layout": "row_major",
            },
        }
    }

    prepared = prepare_fused_component_from_documents(
        opportunity=_three_way_opportunity("block_alpha"),
        manifest=manifest,
        tensor_index=tensor_index,
        source_circuit=source,
    )

    fused = prepared.transformed.replacement_nodes[1]
    assert fused["op"] == "parallel_linear_3way"
    assert fused["outputs"] == ["wide", "narrow", "auxiliary"]
    assert prepared.transformed.proof["status"] == "passed"
    kernel = prepared.replacement_kernels[1]
    assert kernel["shader_path"].endswith(
        "parallel_linear_3way_prequant_fp8_e4m3_se8m0_"
        "b128x128_4096x128_256_384.spv"
    )
    assert [
        implementation["lane_tile_width"]
        for implementation in kernel["batch_implementations"]
    ] == [2, 4, 8, 16]


def test_combined_upstream_and_projection_island_is_one_exact_overlay() -> None:
    source, _compiled, manifest, tensor_index = _combined_projection_documents()
    opportunity = _combined_opportunity("block_alpha")

    prepared = prepare_fused_component_from_documents(
        opportunity=opportunity,
        manifest=manifest,
        tensor_index=tensor_index,
        source_circuit=source,
    )

    assert [node["id"] for node in prepared.transformed.source_nodes] == [
        "hyper_function__hyper_sinkhorn__hyper_reduce",
        "operator_norm",
        "quantizer_0",
        "wide_0",
        "narrow_0",
    ]
    producer, projections = prepared.transformed.replacement_nodes
    assert producer["id"] == "quantizer_0"
    assert producer["op"] == "hyper_connection_pre_rms_norm"
    assert projections["op"] == "parallel_linear_2way"
    assert projections["outputs"] == ["wide", "narrow"]
    assert producer["attrs"]["physical_output_representations"][0][
        "consumer_node_ids"
    ] == [projections["id"]]
    rewrites = {
        record["candidate_node"]: record["proof_contract"]
        for record in prepared.transformed.proof["rewrites"]
    }
    assert rewrites[producer["id"]] == (
        "hyper_connection_pre_rms_norm_exact_bf16.v1"
    )
    assert rewrites[projections["id"]] == "parallel_linear_exact_bf16.v1"
    assert [kernel["node_id"] for kernel in prepared.replacement_kernels] == [
        producer["id"],
        projections["id"],
    ]
    assert all(
        kernel["shader_path"].startswith("kernels/parallel_projection/")
        for kernel in prepared.replacement_kernels
    )
    assert all(
        [
            implementation["lane_tile_width"]
            for implementation in kernel["batch_implementations"]
        ]
        == [2, 4, 8, 16]
        for kernel in prepared.replacement_kernels
    )


@pytest.mark.parametrize("combined", (False, True), ids=("projection", "combined"))
def test_constructed_candidate_proof_reconstructs_artifacts_and_rejects_tampering(
    tmp_path,
    monkeypatch,
    combined,
) -> None:
    if combined:
        source, compiled, manifest, tensor_index = _combined_projection_documents()
        opportunity = _combined_opportunity("block_alpha")
    else:
        source, compiled = _noncontiguous_projection_circuits()
        manifest = {
            "tensor_index_path": "tensors.json",
            "circuit_graph": {
                "components": [
                    {"component_id": "block_alpha", "circuit": compiled}
                ]
            },
            "component_executions": [
                {
                    "component_id": "block_alpha",
                    "kernels": [
                        {
                            "node_id": node_id,
                            "execution_index": index,
                            "op": next(
                                node["op"]
                                for node in compiled["nodes"]
                                if node["id"] == node_id
                            ),
                            "shader_path": f"shaders/{node_id}.spv",
                            "local_size_x": 1024,
                            "workgroup_count_x": 1,
                        }
                        for index, node_id in enumerate(
                            ("quantizer_0", "wide_0", "narrow_0")
                        )
                    ],
                }
            ],
        }
        tensor_index = {
            "tensors": {
                "wide.weight": {
                    "dtype": "F8_E4M3",
                    "shape": [128, 4096],
                    "layout": "row_major",
                },
                "wide.scale": {
                    "dtype": "F8_E8M0",
                    "shape": [1, 32],
                    "layout": "row_major",
                },
                "narrow.weight": {
                    "dtype": "F8_E4M3",
                    "shape": [256, 4096],
                    "layout": "row_major",
                },
                "narrow.scale": {
                    "dtype": "F8_E8M0",
                    "shape": [2, 32],
                    "layout": "row_major",
                },
                "norm.weight": {
                    "dtype": "BF16",
                    "shape": [128],
                    "layout": "row_major",
                },
            }
        }
        opportunity = _opportunity("block_alpha")
    package = tmp_path / "package"
    (package / "lowered" / "block_alpha").mkdir(parents=True)
    (package / "vulkan_resident_package.json").write_text(json.dumps(manifest))
    (package / "tensors.json").write_text(json.dumps(tensor_index))
    (package / "lowered" / "block_alpha" / "circuit.json").write_text(
        json.dumps(source)
    )
    context = _Context(
        hardware_profile={"capability_class": "capability_fixture"},
        qualification_regime=QualificationRegime(),
        source_artifacts=PackageSourceArtifactResolver(package),
    )
    monkeypatch.setattr(
        "nerve.representation_optimizer.providers.parallel_projection_fusion."
        "provider.discover_parallel_projection_fusions",
        lambda _context: (opportunity,),
    )
    provider = ExactParallelProjectionFusionProvider()
    evidence = EvidenceAssessment(
        accepted=True,
        evidence_ids=opportunity.evidence_ids,
        facts={"exact": True},
        reasons=("fixture evidence",),
    )
    candidate = provider.synthesize_candidates(context, evidence)[0]
    representation = provider.emit_representation_ir(context, candidate)
    lowering = provider.lower_for_target(context, candidate, representation)
    build_plan = CandidateBuildPlan.from_json(
        provider.construction_requirements(context, candidate)
    )
    workspace = tmp_path / "workspace"
    root = workspace / "ready" / candidate["candidate_id"]
    root.mkdir(parents=True)
    construction = CandidateConstructionContext(
        package_dir=package,
        staging_dir=root,
        candidate=candidate,
        representation_graph=representation,
        target_lowering=lowering,
        build_plan=build_plan,
        source_artifacts=context.source_artifacts,
        started_ns=time.monotonic_ns(),
        cancel_requested=None,
    )
    toolchain = ParallelProjectionFusionToolchainResolver().resolve(
        SimpleNamespace(provider=provider.identity, target_lowering=lowering)
    )
    for phase, service in (
        (
            "semantic_construction",
            toolchain.semantic_constructor.construct_semantic_artifacts,
        ),
        ("ordinary_lowering", toolchain.ordinary_relowerer.run_ordinary_lowering),
        (
            "physical_optimization",
            toolchain.physical_optimizer.optimize_physical_artifacts,
        ),
    ):
        construction.begin_phase(phase)
        service(construction)
        construction.end_phase()
    construction.validate_complete()
    construction.write_internal_contract("target_lowering.json", lowering)

    proof_digest = staged_file_digest(root / PROOF_PATH)
    verifier = ExactParallelProjectionFusionProofVerifier(
        source_artifacts=context.source_artifacts,
        candidate_workspace_root=workspace,
    )
    request_values = {
        "plan_id": stable_contract_id("validation_plan", candidate["candidate_id"]),
        "candidate_id": candidate["candidate_id"],
        "verifier_id": PROOF_VERIFIER_ID,
        "source_contract_digests": tuple(
            sorted(candidate["source_contract_digests"])
        ),
        "construction_record_digest": contract_digest(
            {"construction": "fixture"}
        ),
        "reference_implementation": {
            "implementation_id": "source",
            "contract_digest": contract_digest({"source": "fixture"}),
            "artifact_refs": [],
        },
        "candidate_implementation": {
            "implementation_id": (
                f"staged-representation:{candidate['candidate_id']}"
            ),
            "contract_digest": contract_digest({"candidate": "fixture"}),
            "artifact_refs": [{"path": PROOF_PATH, "digest": proof_digest}],
        },
    }
    for obligation in candidate["behavioral_contract"]["proof_obligations"]:
        result = verifier.verify(
            ProofRequest(obligation=obligation, **request_values)
        )
        assert result["status"] == "proven"
        assert result["facts"]["region_count"] == 1

    overlay_path = root / component_overlay_path(
        opportunity.component_id,
        opportunity.physical_node_id,
    )
    overlay = json.loads(overlay_path.read_text())
    overlay["replacement"]["nodes"][0]["attrs"]["consumer_node_ids"] = [
        "tampered"
    ]
    overlay_path.write_text(json.dumps(overlay))
    tampered = verifier.verify(
        ProofRequest(
            obligation=candidate["behavioral_contract"]["proof_obligations"][0],
            **request_values,
        )
    )
    assert tampered["status"] == "inconclusive"
    assert "changed outside its proof" in tampered["diagnostics"][0]


def test_provider_builds_one_model_independent_candidate_for_equivalent_components(
    monkeypatch,
) -> None:
    opportunities = (_opportunity("block_alpha"), _opportunity("block_beta"))
    provider, context, candidates = _products(monkeypatch, opportunities)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["representation"]["topology"]["component_region_ids"] == [
        {
            "component_id": "block_alpha",
            "physical_node_id": "wide_0__narrow_0",
            "combines_upstream_producer": False,
        },
        {
            "component_id": "block_beta",
            "physical_node_id": "wide_0__narrow_0",
            "combines_upstream_producer": False,
        },
    ]
    assert "model" not in candidate["target_predicate"]
    assert candidate["target_predicate"]["compiler_capabilities"] == (
        opportunities[0].compiler_device
    )
    representation = provider.emit_representation_ir(context, candidate)
    RepresentationGraphDocument.from_json(representation)
    lowering = provider.lower_for_target(context, candidate, representation)
    build_plan = CandidateBuildPlan.from_json(
        provider.construction_requirements(context, candidate)
    )
    RuntimeMountPlan.from_json(
        provider.mount_requirements(context, candidate),
        candidate_id=candidate["candidate_id"],
        build_plan=build_plan,
    )
    validation = ValidationRequirements.from_json(
        provider.validation_requirements(context, candidate)
    )
    workloads = tuple(
        BenchmarkWorkload.from_json(workload)
        for workload in provider.benchmark_workloads(context, candidate)
    )
    workload_documents = [workload.to_json() for workload in workloads]

    assert len(lowering["regions"]) == 2
    assert len(build_plan.output_paths) == 9
    assert [
        workload["regime"]["activation_batch_width"]
        for workload in workload_documents
    ] == [
        1,
        4,
    ]
    assert all(
        workload["useful_work"]["minimum_units"] == 2
        for workload in workload_documents
    )
    assert all(
        workload["useful_work"]["sustained_window_count"] == 2
        for workload in workload_documents
    )
    assert all(
        workload["controls"]["physical_execution_scope"] == "component"
        for workload in workload_documents
    )
    assert any(
        check["product_performance"] is True for check in validation.checks
    )


def test_provider_exposes_combined_producer_as_one_nonoverlapping_candidate(
    monkeypatch,
) -> None:
    opportunity = _combined_opportunity("block_alpha")
    provider, context, candidates = _products(monkeypatch, (opportunity,))

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["representation"]["kind"] == (
        "exact_capability_scoped_upstream_parallel_projection_fusion"
    )
    assert candidate["representation"]["topology"]["component_region_ids"] == [
        {
            "component_id": "block_alpha",
            "physical_node_id": "wide_0__narrow_0",
            "combines_upstream_producer": True,
        }
    ]
    assert len(candidate["scope_ids"]) == 5
    assert candidate["behavioral_contract"]["proof_obligations"][-1] == (
        "combined_upstream_producer_preserves_hyper_norm_and_prequant_order"
    )
    representation = provider.emit_representation_ir(context, candidate)
    operation = representation["nodes"][0]["operation"]
    assert operation == "exact_upstream_hyper_norm_parallel_projection_island"
    estimate = provider.estimate_static_cost(
        context,
        candidate,
        representation,
        provider.lower_for_target(context, candidate, representation),
    )
    assert estimate.steady_state_work["source_dispatch_count"] == 5
    assert estimate.steady_state_work["candidate_dispatch_count"] == 2


def test_combined_candidate_is_the_exact_union_of_conflicting_partial_rewrites() -> None:
    combined = _combined_opportunity("block_alpha")
    projection = replace(combined, upstream_hyper_fusion=None)
    upstream = combined.upstream_hyper_fusion
    assert upstream is not None

    shared = set(projection.scope_ids).intersection(upstream.scope_ids)
    assert shared == set(projection.region.boundary_scope_ids)
    assert set(combined.scope_ids) == set(projection.scope_ids).union(
        upstream.scope_ids
    )
    assert len(combined.scope_ids) == len(set(combined.scope_ids))


def test_provider_does_not_share_measurements_across_physical_classes(
    monkeypatch,
) -> None:
    opportunities = (
        _opportunity("block_alpha"),
        _opportunity(
            "block_alpha",
            region_index=1,
            performance_signature="different_geometry",
        ),
    )
    _provider, _context, candidates = _products(monkeypatch, opportunities)

    assert len(candidates) == 2
    assert {
        tuple(
            record["component_id"]
            for record in candidate["representation"]["topology"][
                "component_region_ids"
            ]
        )
        for candidate in candidates
    } == {("block_alpha",)}
    assert len({candidate["candidate_id"] for candidate in candidates}) == 2


def test_prepared_component_cache_is_bound_to_exact_region_family(monkeypatch) -> None:
    first = _opportunity("block_alpha")
    second = _opportunity(
        "block_alpha",
        region_index=1,
        performance_signature="different_geometry",
    )
    calls = []
    monkeypatch.setattr(
        "nerve.representation_optimizer.providers.parallel_projection_fusion.physical._prepare",
        lambda _context, opportunity: calls.append(opportunity.scope_ids) or opportunity,
    )
    context = _Context(
        hardware_profile={"capability_class": "capability_fixture"},
    )

    assert prepare_fused_component(context, first) is first
    assert prepare_fused_component(context, second) is second
    assert calls == [first.scope_ids, second.scope_ids]


def test_provider_rejects_candidate_source_contract_drift(monkeypatch) -> None:
    provider, context, candidates = _products(
        monkeypatch,
        (_opportunity("block_alpha"),),
    )
    candidate = deepcopy(candidates[0])
    candidate["source_contract_digests"][0] = contract_digest({"drift": True})

    with pytest.raises(ModelCompileError, match="source contracts drifted"):
        provider.emit_representation_ir(context, candidate)


def test_discovery_rejects_non_vulkan_or_incomplete_target_before_source_access() -> None:
    for hardware_profile, reason in (
        (
            {
                "hardware_identity": {"device_kind": "cpu"},
                "provenance": {"api": "host"},
            },
            "requires a Vulkan GPU",
        ),
        (
            {
                "hardware_identity": {"device_kind": "gpu"},
                "provenance": {"api": "vulkan"},
                "capability_extensions": {
                    "vulkan_compiler_capabilities": {
                        "shader_features": ["shader_int8"],
                        "max_compute_work_group_invocations": 1024,
                        "max_compute_work_group_size_x": 1024,
                        "subgroup_operations": ["arithmetic"],
                        "subgroup_size": 64,
                        "subgroup_compute_supported": True,
                    }
                },
            },
            "cannot execute",
        ),
    ):
        result = discovery_result(
            _Context(hardware_profile=hardware_profile, scope_ids=())
        )
        assert not result.opportunities
        assert reason in result.reasons[0]


def test_builtin_registry_and_toolchain_include_parallel_projection_fusion(monkeypatch) -> None:
    opportunity = _opportunity("block_alpha")
    _provider, _context, candidates = _products(monkeypatch, (opportunity,))
    provider_ids = {
        provider.identity.provider_id
        for provider in load_builtin_provider_registry().providers
    }
    assert "nerve.exact_parallel_projection_fusion" in provider_ids

    lowering = {"schema": "nerve.optimizer.parallel_projection_fusion_vulkan_lowering.v1"}
    toolchain = BuiltinCandidateToolchainResolver().resolve(
        SimpleNamespace(
            provider=ExactParallelProjectionFusionProvider.identity,
            target_lowering=lowering,
        )
    )
    assert candidates
    assert toolchain.physical_optimizer is not None


def test_fused_kernel_artifacts_are_confined_to_candidate_namespace() -> None:
    assert kernel_artifact_path("fused_decode.comp") == (
        "kernels/parallel_projection/fused_decode.spv"
    )
    for unsafe in ("../escape.comp", "/tmp/escape.comp", "nested/fused.comp", "x.spv"):
        with pytest.raises(ModelCompileError):
            kernel_artifact_path(unsafe)
