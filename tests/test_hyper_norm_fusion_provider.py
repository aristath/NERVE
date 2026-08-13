from __future__ import annotations

from copy import deepcopy
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
from nerve.representation_optimizer.providers.hyper_norm_fusion.artifacts import (
    kernel_artifact_path,
)
from nerve.representation_optimizer.providers.hyper_norm_fusion.discovery import (
    HyperNormFusionOpportunity,
    HyperNormRegion,
    discovery_result,
)
from nerve.representation_optimizer.providers.hyper_norm_fusion.physical import (
    ShaderArtifact,
    prepare_fused_component,
)
from nerve.representation_optimizer.providers.hyper_norm_fusion.provider import (
    ExactHyperNormFusionProvider,
)
from nerve.representation_optimizer.providers.types import EvidenceAssessment
from nerve.representation_optimizer.qualification import QualificationRegime
from nerve.representation_optimizer.representation_ir.contracts import (
    RepresentationGraphDocument,
)
from nerve.representation_optimizer.staging.contracts import CandidateBuildPlan
from nerve.representation_optimizer.validation.contracts import (
    ValidationRequirements,
)


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


def _region(component_id: str, index: int) -> HyperNormRegion:
    scope_id = stable_contract_id("scope", component_id, index)
    return HyperNormRegion(
        scope_id=scope_id,
        source_contract_digest=contract_digest(
            {"component_id": component_id, "region": index}
        ),
        semantic_source_node_ids=(f"reduce_{index}", f"norm_{index}"),
        hyper_node_id=f"hyper_{index}",
        norm_node_id=f"norm_{index}",
        quantizer_node_id=f"quantizer_{index}",
    )


def _opportunity(
    component_id: str,
    *,
    region_index: int = 0,
    performance_signature: str = "performance_class_shared",
) -> HyperNormFusionOpportunity:
    regions = (_region(component_id, region_index),)
    return HyperNormFusionOpportunity(
        component_id=component_id,
        regions=regions,
        evidence_ids=(stable_contract_id("evidence", component_id, region_index),),
        source_artifact_refs=(
            f"lowered/{component_id}/circuit.json",
            "tensors.json",
            "vulkan_resident_package.json",
        ),
        manifest_ref="vulkan_resident_package.json",
        circuit_ref=f"lowered/{component_id}/circuit.json",
        tensor_index_ref="tensors.json",
        terminal_node_id="component_output",
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


def _products(monkeypatch, opportunities):
    prepared = SimpleNamespace(
        shader_artifacts=(
            ShaderArtifact(
                "kernels/hyper_norm/fused_decode.spv",
                "fused_decode.comp",
            ),
            ShaderArtifact(
                "kernels/hyper_norm/fused_batch.spv",
                "fused_batch.comp",
            ),
        )
    )
    monkeypatch.setattr(
        "nerve.representation_optimizer.providers.hyper_norm_fusion.provider."
        "discover_hyper_norm_fusions",
        lambda _context: tuple(opportunities),
    )
    monkeypatch.setattr(
        "nerve.representation_optimizer.providers.hyper_norm_fusion.provider."
        "prepare_fused_component",
        lambda _context, _opportunity: prepared,
    )
    monkeypatch.setattr(
        "nerve.representation_optimizer.providers.hyper_norm_fusion.provider."
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
    provider = ExactHyperNormFusionProvider()
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


def test_provider_builds_one_model_independent_candidate_for_equivalent_components(
    monkeypatch,
) -> None:
    opportunities = (_opportunity("block_alpha"), _opportunity("block_beta"))
    provider, context, candidates = _products(monkeypatch, opportunities)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["representation"]["topology"]["component_ids"] == [
        "block_alpha",
        "block_beta",
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
        tuple(candidate["representation"]["topology"]["component_ids"])
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
        "nerve.representation_optimizer.providers.hyper_norm_fusion.physical._prepare",
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


def test_builtin_registry_and_toolchain_include_hyper_norm_fusion(monkeypatch) -> None:
    opportunity = _opportunity("block_alpha")
    _provider, _context, candidates = _products(monkeypatch, (opportunity,))
    provider_ids = {
        provider.identity.provider_id
        for provider in load_builtin_provider_registry().providers
    }
    assert "nerve.exact_hyper_norm_fusion" in provider_ids

    lowering = {"schema": "nerve.optimizer.hyper_norm_fusion_vulkan_lowering.v1"}
    toolchain = BuiltinCandidateToolchainResolver().resolve(
        SimpleNamespace(
            provider=ExactHyperNormFusionProvider.identity,
            target_lowering=lowering,
        )
    )
    assert candidates
    assert toolchain.physical_optimizer is not None


def test_fused_kernel_artifacts_are_confined_to_candidate_namespace() -> None:
    assert kernel_artifact_path("fused_decode.comp") == (
        "kernels/hyper_norm/fused_decode.spv"
    )
    for unsafe in ("../escape.comp", "/tmp/escape.comp", "nested/fused.comp", "x.spv"):
        with pytest.raises(ModelCompileError):
            kernel_artifact_path(unsafe)
