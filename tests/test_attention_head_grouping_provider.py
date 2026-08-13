from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from nerve.compilation import ModelCompileError
from nerve.representation_optimizer.benchmarking.contracts import BenchmarkWorkload
from nerve.representation_optimizer.contracts import contract_digest, stable_contract_id
from nerve.representation_optimizer.mounting import RuntimeMountPlan
from nerve.representation_optimizer.providers.attention_head_grouping.artifacts import (
    kernel_artifact_path,
)
from nerve.representation_optimizer.providers.attention_head_grouping.discovery import (
    AttentionHeadGroupingOpportunity,
    discovery_result,
)
from nerve.representation_optimizer.providers.attention_head_grouping.physical import (
    ShaderArtifact,
    prepare_grouped_attention,
)
from nerve.representation_optimizer.providers.attention_head_grouping.proof import (
    _require_candidate_artifact,
)
from nerve.representation_optimizer.providers.attention_head_grouping.provider import (
    ExactAttentionHeadGroupingProvider,
)
from nerve.representation_optimizer.providers.builtin import (
    BuiltinCandidateToolchainResolver,
    load_builtin_provider_registry,
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


def _opportunity(
    component_id: str,
    *,
    head_group: int = 4,
    compression_ratio: int = 4,
    max_compressed_indices: int = 512,
    performance_signature: str = "attention_performance_class_shared",
) -> AttentionHeadGroupingOpportunity:
    scope_id = stable_contract_id("attention_scope", component_id)
    return AttentionHeadGroupingOpportunity(
        scope_id=scope_id,
        source_contract_digest=contract_digest(
            {"component_id": component_id, "op": "indexed_sparse_attention"}
        ),
        component_id=component_id,
        source_node_id="attention_read",
        physical_node_id="attention_read",
        terminal_node_id="component_output",
        evidence_ids=(stable_contract_id("evidence", component_id),),
        source_artifact_refs=(
            f"lowered/{component_id}/circuit.json",
            "tensors.json",
            "vulkan_resident_package.json",
        ),
        manifest_ref="vulkan_resident_package.json",
        circuit_ref=f"lowered/{component_id}/circuit.json",
        tensor_index_ref="tensors.json",
        query_heads=64,
        key_value_heads=1,
        head_width=512,
        local_window=128,
        compression_ratio=compression_ratio,
        max_compressed_indices=max_compressed_indices,
        head_group=head_group,
        shader_suffix=(
            "q64_kv1_d512_w128_"
            f"r{compression_ratio}_k{max_compressed_indices}_"
            "scale0.0441941738__sc8"
        ),
        max_context_activations=131_072,
        compiler_device={
            "max_compute_work_group_invocations": 1024,
            "max_compute_work_group_size_x": 1024,
            "subgroup_operations": ["basic", "arithmetic"],
            "subgroup_size": 64,
            "subgroup_compute_supported": True,
        },
        performance_signature=performance_signature,
    )


def _products(monkeypatch, opportunities):
    prepared = SimpleNamespace(
        shader_artifacts=(
            ShaderArtifact(
                "kernels/attention_head_grouping/grouped_decode.spv",
                "grouped_decode.comp",
            ),
            ShaderArtifact(
                "kernels/attention_head_grouping/grouped_prefill.spv",
                "grouped_prefill.comp",
            ),
        )
    )
    monkeypatch.setattr(
        "nerve.representation_optimizer.providers.attention_head_grouping.provider."
        "discover_attention_head_groupings",
        lambda _context: tuple(opportunities),
    )
    monkeypatch.setattr(
        "nerve.representation_optimizer.providers.attention_head_grouping.provider."
        "prepare_grouped_attention",
        lambda _context, _opportunity: prepared,
    )
    monkeypatch.setattr(
        "nerve.representation_optimizer.providers.attention_head_grouping.provider."
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
    provider = ExactAttentionHeadGroupingProvider()
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


def test_provider_builds_one_candidate_for_equivalent_attention_components(
    monkeypatch,
) -> None:
    opportunities = (_opportunity("block_alpha"), _opportunity("block_beta"))
    provider, context, candidates = _products(monkeypatch, opportunities)

    assert len(candidates) == 1
    candidate = candidates[0]
    topology = candidate["representation"]["topology"]
    assert topology["component_ids"] == ["block_alpha", "block_beta"]
    assert topology["head_group"] == 4
    assert "model" not in candidate["target_predicate"]
    representation = provider.emit_representation_ir(context, candidate)
    RepresentationGraphDocument.from_json(representation)
    assert [item["id"] for item in representation["nodes"][0]["inputs"]] == [
        "compressed_indices",
        "compressed_state",
        "local_state",
        "query",
    ]
    logical_dtypes = {
        item["signal"]: item["dtype"]
        for item in representation["logical_contracts"]
    }
    assert logical_dtypes["local_attention_state"] == "BF16"
    assert logical_dtypes["compressed_attention_state"] == "BF16"
    assert logical_dtypes["compressed_attention_indices"] == "U32"
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

    assert len(lowering["regions"]) == 2
    assert len(build_plan.output_paths) == 9
    assert [
        workload.to_json()["regime"]["activation_batch_width"]
        for workload in workloads
    ] == [1, 4]
    assert all(
        workload.to_json()["useful_work"]["minimum_units"] == 2
        for workload in workloads
    )
    assert any(check["product_performance"] is True for check in validation.checks)


def test_provider_keeps_head_groups_and_physical_classes_separate(monkeypatch) -> None:
    opportunities = (
        _opportunity("block_alpha", head_group=2),
        _opportunity(
            "block_alpha",
            head_group=4,
            performance_signature="attention_performance_class_hg4",
        ),
        _opportunity(
            "block_beta",
            head_group=4,
            performance_signature="attention_performance_class_hg4",
        ),
    )
    _provider, _context, candidates = _products(monkeypatch, opportunities)

    assert len(candidates) == 2
    assert {
        (
            candidate["representation"]["topology"]["head_group"],
            tuple(candidate["representation"]["topology"]["component_ids"]),
        )
        for candidate in candidates
    } == {(2, ("block_alpha",)), (4, ("block_alpha", "block_beta"))}


def test_local_only_attention_boundary_omits_compressed_inputs(monkeypatch) -> None:
    opportunity = _opportunity(
        "block_alpha",
        compression_ratio=0,
        max_compressed_indices=0,
        performance_signature="attention_local_only",
    )
    provider, context, candidates = _products(monkeypatch, (opportunity,))

    representation = provider.emit_representation_ir(context, candidates[0])
    RepresentationGraphDocument.from_json(representation)
    assert [item["id"] for item in representation["nodes"][0]["inputs"]] == [
        "local_state",
        "query",
    ]
    assert not any(
        "compressed" in item["id"]
        for item in representation["logical_contracts"]
    )


def test_prepared_attention_cache_is_bound_to_exact_schedule(monkeypatch) -> None:
    first = _opportunity("block_alpha", head_group=2)
    second = _opportunity(
        "block_alpha",
        head_group=4,
        performance_signature="attention_performance_class_hg4",
    )
    calls = []
    monkeypatch.setattr(
        "nerve.representation_optimizer.providers.attention_head_grouping.physical."
        "_prepare",
        lambda _context, opportunity: calls.append(opportunity.head_group)
        or opportunity,
    )
    context = _Context(
        hardware_profile={"capability_class": "capability_fixture"},
    )

    assert prepare_grouped_attention(context, first) is first
    assert prepare_grouped_attention(context, first) is first
    assert prepare_grouped_attention(context, second) is second
    assert calls == [2, 4]


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
                        "max_compute_work_group_invocations": 512,
                        "max_compute_work_group_size_x": 512,
                        "subgroup_operations": ["basic", "arithmetic"],
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


def test_builtin_registry_and_toolchain_include_attention_head_grouping(
    monkeypatch,
) -> None:
    opportunity = _opportunity("block_alpha")
    _provider, _context, candidates = _products(monkeypatch, (opportunity,))
    provider_ids = {
        provider.identity.provider_id
        for provider in load_builtin_provider_registry().providers
    }
    assert "nerve.exact_attention_head_grouping" in provider_ids
    toolchain = BuiltinCandidateToolchainResolver().resolve(
        SimpleNamespace(
            provider=ExactAttentionHeadGroupingProvider.identity,
            target_lowering={
                "schema": (
                    "nerve.optimizer.attention_head_grouping_vulkan_lowering.v1"
                )
            },
        )
    )
    assert candidates
    assert toolchain.physical_optimizer is not None


def test_grouped_attention_kernel_artifacts_are_confined() -> None:
    assert kernel_artifact_path("grouped_decode.comp") == (
        "kernels/attention_head_grouping/grouped_decode.spv"
    )
    for unsafe in ("../escape.comp", "/tmp/escape.comp", "nested/x.comp", "x.spv"):
        with pytest.raises(ModelCompileError):
            kernel_artifact_path(unsafe)


def test_proof_seal_uses_canonical_implementation_artifact_refs() -> None:
    digest = "nerve.optimizer.artifact_sha256.v1:" + "1" * 64
    implementation = {
        "implementation_id": "staged-representation:candidate_fixture",
        "contract_digest": contract_digest({"implementation": "fixture"}),
        "artifact_refs": [{"path": "proofs/proof.json", "digest": digest}],
    }

    _require_candidate_artifact(
        implementation,
        "proofs/proof.json",
        digest,
    )

    for corrupted in (
        {**implementation, "artifact_refs": []},
        {
            **implementation,
            "artifact_refs": [
                {"path": "proofs/proof.json", "digest": _ARTIFACT_DIGEST}
            ],
        },
        {
            **implementation,
            "artifact_refs": [
                {"path": "proofs/other.json", "digest": digest}
            ],
        },
    ):
        with pytest.raises(ModelCompileError, match="does not seal proof artifact"):
            _require_candidate_artifact(
                corrupted,
                "proofs/proof.json",
                digest,
            )
