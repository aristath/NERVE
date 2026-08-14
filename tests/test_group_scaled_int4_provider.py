from __future__ import annotations

import json
import struct
import time
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest

from nerve.compilation import ModelCompileError
from nerve.model_package_manifest import component_kernel_spec
from nerve.model_package_shader_selection import (
    local_size_x_for_shader_file,
    shader_file_for_node,
    workgroup_count_x_for_node,
)
from nerve.model_package_tensors import (
    bf16_bytes_to_f32,
    compiled_safetensors_header,
    f32_to_bf16_bytes,
)
from nerve.representation_optimizer.benchmarking.contracts import BenchmarkWorkload
from nerve.representation_optimizer.contracts import (
    contract_digest,
    stable_contract_id,
    validate_contract,
)
from nerve.representation_optimizer.mounting import (
    RuntimeMountPlan,
    validate_runtime_mount_artifacts,
)
from nerve.representation_optimizer.providers.group_scaled_int4 import (
    provider as group_scaled_int4_provider,
)
from nerve.representation_optimizer.providers.group_scaled_int4.artifacts import (
    TENSOR_FRAGMENT_PATH,
    component_overlay_path,
    scale_artifact_path,
    weight_artifact_path,
)
from nerve.representation_optimizer.providers.group_scaled_int4.discovery import (
    discover_group_scaled_int4_linears,
    discovery_result,
)
from nerve.representation_optimizer.providers.group_scaled_int4.provider import (
    GroupScaledInt4LinearProvider,
)
from nerve.representation_optimizer.providers.group_scaled_int4.toolchain import (
    GroupScaledInt4ToolchainResolver,
)
from nerve.representation_optimizer.providers.source_artifacts import (
    PackageSourceArtifactResolver,
)
from nerve.representation_optimizer.providers.types import EvidenceAssessment
from nerve.representation_optimizer.qualification import QualificationRegime
from nerve.representation_optimizer.representation_ir.contracts import (
    RepresentationGraphDocument,
)
from nerve.representation_optimizer.staging.contracts import CandidateBuildPlan
from nerve.representation_optimizer.staging.workspace import (
    CandidateConstructionContext,
)
from nerve.representation_optimizer.validation.contracts import (
    ValidationRequirements,
)


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


def _hardware_profile() -> dict:
    return {
        "capability_class": "hardware_capability_group_scaled_int4_fixture",
        "hardware_identity": {"device_kind": "gpu"},
        "provenance": {"api": "vulkan"},
        "processes": [
            {
                "name": "shader_vector",
                "availability": "available",
                "programmability": "shader",
                "numeric_formats": ["bf16", "f32"],
            }
        ],
        "capability_extensions": {
            "vulkan_compiler_capabilities": {
                "shader_features": [],
                "max_compute_work_group_invocations": 1024,
                "max_compute_work_group_size_x": 1024,
                "subgroup_operations": ["arithmetic", "basic"],
                "subgroup_size": 64,
                "subgroup_compute_supported": True,
            }
        },
    }


def _fixture_context(tmp_path, *, shared_weight: bool = False, prequantized=False):
    package = tmp_path / "package"
    (package / "lowered" / "block_alpha").mkdir(parents=True)
    (package / "weights").mkdir()
    values = np.linspace(-2.0, 2.0, 4 * 32, dtype=np.float32).reshape(4, 32)
    weight_payload = f32_to_bf16_bytes(values.reshape(-1))
    weight_name = "block.router.weight"
    weight_header = compiled_safetensors_header(
        weight_name,
        dtype="BF16",
        shape=[4, 32],
        byte_count=len(weight_payload),
        layout="row_major",
    )
    weight_path = package / "weights" / "model.safetensors"
    weight_path.write_bytes(
        struct.pack("<Q", len(weight_header)) + weight_header + weight_payload
    )
    tensor_index = {
        "schema": "nerve.tensor_index.v1",
        "source": {
            "weights_files": [
                {
                    "path": "weights/model.safetensors",
                    "safetensors_header_bytes": len(weight_header),
                }
            ]
        },
        "tensors": {
            weight_name: {
                "dtype": "BF16",
                "shape": [4, 32],
                "data_offsets": [0, len(weight_payload)],
                "parameter_count": 128,
                "byte_count": len(weight_payload),
                "source_file": "weights/model.safetensors",
                "data_sha256": sha256(weight_payload).hexdigest(),
                "safetensors_header_bytes": len(weight_header),
                "layout": "row_major",
            }
        },
    }
    source_node = {
        "id": "router_projection",
        "op": "linear",
        "inputs": ["input"],
        "outputs": ["logits"],
        "params": ["router_weight"],
    }
    source_nodes = [source_node]
    if shared_weight:
        source_nodes.append(
            {
                "id": "other_projection",
                "op": "linear",
                "inputs": ["input"],
                "outputs": ["other"],
                "params": ["router_weight"],
            }
        )
    source_circuit = {
        "schema": "nerve.stream_circuit.v1",
        "source": {"component_id": "block_alpha"},
        "boundary": {
            "inputs": [{"id": "input", "source": "input", "shape": [32]}],
            "outputs": [{"id": "logits", "source": "logits", "shape": [4]}],
        },
        "state_ports": [],
        "parameters": {
            "refs": {
                "router_weight": {
                    "tensor": weight_name,
                    "role": "routing_projection",
                }
            }
        },
        "behavioral_error_contract": {"mode": "source_reference_circuit"},
        "nodes": source_nodes,
    }
    compiled_circuit = deepcopy(source_circuit)
    compiled_circuit["nodes"][0]["attrs"] = {"output_element_bytes": [2]}
    if prequantized:
        compiled_circuit["nodes"][0]["attrs"]["physical_input_contract"] = (
            "bf16_blockwise_fp8_e4m3_e8m0_scale_f32.v1"
        )
    compiler_target = {
        "devices": [
            _hardware_profile()["capability_extensions"][
                "vulkan_compiler_capabilities"
            ]
        ]
    }
    source_shader = shader_file_for_node(
        compiled_circuit,
        compiled_circuit["nodes"][0],
        tensor_index,
        {"hidden_size": 32},
        compiler_target=compiler_target,
    )
    source_kernel = component_kernel_spec(
        execution_index=0,
        node=compiled_circuit["nodes"][0],
        circuit=compiled_circuit,
        shader_file=source_shader,
        local_size_x=local_size_x_for_shader_file(
            source_shader,
            compiled_circuit["nodes"][0],
        ),
        workgroup_count_x=workgroup_count_x_for_node(
            compiled_circuit,
            compiled_circuit["nodes"][0],
            tensor_index,
            dimensions={"hidden_size": 32},
        ),
        tensor_index=tensor_index,
    )
    _compiled_shader_paths(source_kernel)
    manifest = {
        "tensor_index_path": "tensors.json",
        "max_context_activations": 131_072,
        "circuit_graph": {
            "components": [
                {
                    "component_id": "block_alpha",
                    "circuit": compiled_circuit,
                    "params": deepcopy(compiled_circuit["parameters"]),
                }
            ]
        },
        "component_executions": [
            {
                "component_id": "block_alpha",
                "kernels": [source_kernel],
            }
        ],
    }
    (package / "tensors.json").write_text(json.dumps(tensor_index))
    (package / "vulkan_resident_package.json").write_text(json.dumps(manifest))
    circuit_path = package / "lowered" / "block_alpha" / "circuit.json"
    circuit_path.write_text(json.dumps(source_circuit))
    scope_id = stable_contract_id("scope", "block_alpha", "router_projection")
    scope = {
        "scope_id": scope_id,
        "kind": "operator",
        "members": {
            "component_ids": ["block_alpha"],
            "source_node_ids": ["block_alpha/router_projection"],
        },
        "extensions": {"semantic_roles": ["linear"]},
    }
    contract = {
        "scope_id": scope_id,
        "semantic_role": "linear",
        "contract_digest": contract_digest(scope),
        "exact_reference": {
            "artifact_refs": ["lowered/block_alpha/circuit.json"]
        },
    }
    evidence_id = stable_contract_id("evidence", scope_id)
    evidence = {
        "evidence_id": evidence_id,
        "scope_id": scope_id,
        "claims": [{"status": "supported"}],
    }
    context = _Context(
        scopes=(scope,),
        source_contracts=(contract,),
        evidence=(evidence,),
        scope_ids=(scope_id,),
        hardware_profile=_hardware_profile(),
        qualification_regime=QualificationRegime(),
        source_artifacts=PackageSourceArtifactResolver(package),
    )
    return context, package, values


def _compiled_shader_paths(kernel: dict) -> None:
    kernel["shader_path"] = kernel["shader_path"].replace(".comp", ".spv")
    for implementation in kernel.get("batch_implementations", []):
        for stage in implementation.get("stages", []):
            stage["shader_path"] = stage["shader_path"].replace(".comp", ".spv")
    for implementation in kernel.get("physical_implementations", []):
        implementation["shader_path"] = implementation["shader_path"].replace(
            ".comp", ".spv"
        )


def test_discovery_requires_private_plain_bf16_parameter(tmp_path) -> None:
    context, _package, _values = _fixture_context(tmp_path / "accepted")
    opportunities = discover_group_scaled_int4_linears(context)
    assert len(opportunities) == 1
    opportunity = opportunities[0]
    assert opportunity.source_weight_ref_id == "router_weight"
    assert opportunity.packed_shape == (4, 4)
    assert opportunity.scale_shape == (4, 1)

    shared, _package, _values = _fixture_context(
        tmp_path / "shared",
        shared_weight=True,
    )
    assert discover_group_scaled_int4_linears(shared) == ()

    prequantized, _package, _values = _fixture_context(
        tmp_path / "prequantized",
        prequantized=True,
    )
    assert discover_group_scaled_int4_linears(prequantized) == ()


def test_discovery_rejects_incompatible_subgroup_contract(tmp_path) -> None:
    context, _package, _values = _fixture_context(tmp_path)
    context.hardware_profile["capability_extensions"][
        "vulkan_compiler_capabilities"
    ]["subgroup_size"] = 128
    result = discovery_result(context)
    assert result.opportunities == ()
    assert result.reasons == (
        "target cannot execute the group-scaled INT4 reduction kernel",
    )


def test_provider_contracts_and_toolchain_construct_mountable_candidate(
    tmp_path,
) -> None:
    context, package, source_values = _fixture_context(tmp_path)
    result = discovery_result(context)
    provider = GroupScaledInt4LinearProvider()
    evidence = EvidenceAssessment(
        accepted=True,
        evidence_ids=result.evidence_ids,
        facts={"private_bf16": True},
        reasons=("fixture evidence",),
    )
    candidate = provider.synthesize_candidates(context, evidence)[0]
    validate_contract(candidate)
    representation = provider.emit_representation_ir(context, candidate)
    RepresentationGraphDocument.from_json(representation)
    lowering = provider.lower_for_target(context, candidate, representation)
    estimate = provider.estimate_static_cost(
        context,
        candidate,
        representation,
        lowering,
    )
    assert estimate.permanent_bytes == 72
    assert estimate.steady_state_work["parameter_byte_ratio"] == pytest.approx(
        72 / 256
    )
    build_plan = CandidateBuildPlan.from_json(
        provider.construction_requirements(context, candidate)
    )
    mount_plan = RuntimeMountPlan.from_json(
        provider.mount_requirements(context, candidate),
        candidate_id=candidate["candidate_id"],
        build_plan=build_plan,
    )
    assert all(
        BenchmarkWorkload.from_json(workload)
        for workload in provider.benchmark_workloads(context, candidate)
    )
    ValidationRequirements.from_json(
        provider.validation_requirements(context, candidate)
    )

    root = tmp_path / "staging" / candidate["candidate_id"]
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
    toolchain = GroupScaledInt4ToolchainResolver().resolve(
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
    validate_runtime_mount_artifacts(root, mount_plan)

    opportunity = discover_group_scaled_int4_linears(context)[0]
    fragment = json.loads((root / TENSOR_FRAGMENT_PATH).read_text())
    weight = fragment["tensors"][opportunity.candidate_weight_name]
    scale = fragment["tensors"][opportunity.candidate_scale_name]
    assert weight["dtype"] == "I32"
    assert weight["logical_shape"] == [4, 32]
    assert weight["quantization"] == {
        "format": "compressed_tensors_pack_quantized",
        "bits": 4,
        "group_size": 32,
        "symmetric": True,
        "signed_offset": 8,
        "scales": opportunity.candidate_scale_name,
    }
    packed_payload = _safetensors_payload(
        root / weight_artifact_path("block_alpha", "router_projection")
    )
    scale_payload = _safetensors_payload(
        root / scale_artifact_path("block_alpha", "router_projection")
    )
    packed = np.frombuffer(packed_payload, dtype="<u4").reshape(4, 4)
    encoded = np.stack(
        [((packed >> np.uint32(shift)) & np.uint32(15)) for shift in range(0, 32, 4)],
        axis=2,
    ).reshape(4, 32)
    stored_scales = bf16_bytes_to_f32(scale_payload, [4, 1])
    reconstructed = (encoded.astype(np.int16) - 8).astype(np.float32)
    reconstructed *= stored_scales
    relative_rms = np.sqrt(
        np.sum((reconstructed - source_values) ** 2)
        / np.sum(source_values**2)
    )
    assert relative_rms < 0.12
    assert scale["data_sha256"] == sha256(scale_payload).hexdigest()

    overlay = json.loads(
        (
            root
            / component_overlay_path("block_alpha", "router_projection")
        ).read_text()
    )
    assert overlay["schema"] == (
        "nerve.optimizer.vulkan_component_region_overlay.v2"
    )
    assert overlay["source"]["parameter_refs"] == {
        "router_weight": {
            "tensor": "block.router.weight",
            "role": "routing_projection",
        }
    }
    replacement_params = overlay["replacement"]["nodes"][0]["params"]
    assert replacement_params == [
        opportunity.replacement_weight_ref_id,
        opportunity.replacement_scale_ref_id,
    ]
    assert set(overlay["replacement"]["parameter_refs"]) == set(
        replacement_params
    )
    assert overlay["replacement"]["kernels"][0][
        "physical_execution_contracts"
    ]


def test_equivalent_regions_remain_independently_selectable(
    tmp_path,
    monkeypatch,
) -> None:
    context, _package, _values = _fixture_context(tmp_path)
    first = discover_group_scaled_int4_linears(context)[0]
    second = replace(
        first,
        scope_id=stable_contract_id("scope", "block_alpha", "second_projection"),
        source_contract_digest=stable_contract_id("contract", "second_projection"),
        node_id="second_projection",
        source_weight_ref_id="second_weight",
        source_weight_ref={
            "tensor": first.source_weight.tensor_name,
            "role": "routing_projection",
        },
    )
    monkeypatch.setattr(
        group_scaled_int4_provider,
        "discover_group_scaled_int4_linears",
        lambda _context: (first, second),
    )
    monkeypatch.setattr(
        group_scaled_int4_provider,
        "_prepare",
        lambda _context, _opportunity: SimpleNamespace(shader_artifacts=()),
    )
    provider = GroupScaledInt4LinearProvider()
    candidates = provider.synthesize_candidates(
        context,
        EvidenceAssessment(
            accepted=True,
            evidence_ids=first.evidence_ids,
            facts={"fixture": True},
            reasons=("fixture",),
        ),
    )

    assert len(candidates) == 2
    assert [candidate["scope_ids"] for candidate in candidates] == [
        [first.scope_id],
        [second.scope_id],
    ]
    for candidate in candidates:
        build_plan = CandidateBuildPlan.from_json(
            provider.construction_requirements(context, candidate)
        )
        mount = RuntimeMountPlan.from_json(
            provider.mount_requirements(context, candidate),
            candidate_id=candidate["candidate_id"],
            build_plan=build_plan,
        ).to_json()
        assert len(mount["regions"]) == 1
        assert len(mount["regions"][0]["replacements"]) == 1


def test_toolchain_rejects_nonfinite_source_before_publishing(tmp_path) -> None:
    context, package, _values = _fixture_context(tmp_path)
    weight_path = package / "weights" / "model.safetensors"
    payload = bytearray(weight_path.read_bytes())
    header_bytes = struct.unpack("<Q", payload[:8])[0]
    payload[8 + header_bytes : 10 + header_bytes] = b"\x80\x7f"
    weight_path.write_bytes(payload)
    tensor_index_path = package / "tensors.json"
    tensor_index = json.loads(tensor_index_path.read_text())
    tensor_index["tensors"]["block.router.weight"]["data_sha256"] = sha256(
        bytes(payload[8 + header_bytes :])
    ).hexdigest()
    tensor_index_path.write_text(json.dumps(tensor_index))
    context._cache.clear()
    context.source_artifacts = PackageSourceArtifactResolver(package)
    with pytest.raises(ModelCompileError, match="not finite"):
        _construct_semantic_only(context, tmp_path / "nonfinite")


def _construct_semantic_only(context, root) -> None:
    result = discovery_result(context)
    provider = GroupScaledInt4LinearProvider()
    candidate = provider.synthesize_candidates(
        context,
        EvidenceAssessment(
            accepted=True,
            evidence_ids=result.evidence_ids,
            facts={"fixture": True},
            reasons=("fixture",),
        ),
    )[0]
    representation = provider.emit_representation_ir(context, candidate)
    lowering = provider.lower_for_target(context, candidate, representation)
    build_plan = CandidateBuildPlan.from_json(
        provider.construction_requirements(context, candidate)
    )
    root.mkdir(parents=True)
    construction = CandidateConstructionContext(
        package_dir=context.source_artifacts.package_root,
        staging_dir=root,
        candidate=candidate,
        representation_graph=representation,
        target_lowering=lowering,
        build_plan=build_plan,
        source_artifacts=context.source_artifacts,
        started_ns=time.monotonic_ns(),
        cancel_requested=None,
    )
    constructor = GroupScaledInt4ToolchainResolver().resolve(
        SimpleNamespace(
            provider=provider.identity,
            target_lowering=lowering,
        )
    ).semantic_constructor
    construction.begin_phase("semantic_construction")
    constructor.construct_semantic_artifacts(construction)


def _safetensors_payload(path) -> bytes:
    payload = path.read_bytes()
    header_bytes = struct.unpack("<Q", payload[:8])[0]
    return payload[8 + header_bytes :]
