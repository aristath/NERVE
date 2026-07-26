from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from nerve.representation_optimizer.benchmarking.planning import (
    create_benchmark_workload,
)
from nerve.representation_optimizer.contracts import (
    ContractValidationError,
    representation_candidate_id,
    validate_contract,
)
from nerve.representation_optimizer.descriptor_registry import (
    load_builtin_representation_descriptors,
)
from nerve.representation_optimizer.providers import (
    EvidenceAssessment,
    MatchAssessment,
    ProviderIdentity,
    ProviderProblem,
    ProviderRegistry,
    StaticEstimate,
)
from nerve.representation_optimizer.staging.contracts import (
    staged_artifact_digest,
)
from tests.representation_graph_fixtures import exact_representation_graph
from tests.test_representation_optimizer_contracts import (
    contract_fixtures,
    hardware_profile_contract,
)


@dataclass
class FixtureProvider:
    provider_id: str
    descriptor_id: str
    semantic_match: bool = True
    structural_match: bool = True
    accept_evidence: bool = True
    fail_at: str | None = None
    omit_structural_evidence: bool = False
    mutate_scope_copy: bool = False
    calls: list[str] = field(default_factory=list)

    @property
    def identity(self) -> ProviderIdentity:
        return ProviderIdentity(self.provider_id, "1")

    def _called(self, name: str) -> None:
        self.calls.append(name)
        if self.fail_at == name:
            raise RuntimeError(f"fixture failure at {name}")

    def match_semantics(self, context):
        self._called("match_semantics")
        if self.mutate_scope_copy:
            context.scopes[0]["kind"] = "mutated"
        return MatchAssessment(
            matched=self.semantic_match,
            reasons=("semantic fixture decision",),
        )

    def match_structure(self, context):
        self._called("match_structure")
        evidence_ids = (
            ()
            if self.omit_structural_evidence
            else (context.evidence[0]["evidence_id"],)
        )
        return MatchAssessment(
            matched=self.structural_match,
            reasons=("structural fixture decision",),
            evidence_ids=evidence_ids,
        )

    def analyze_evidence(self, context):
        self._called("analyze_evidence")
        return EvidenceAssessment(
            accepted=self.accept_evidence,
            evidence_ids=(context.evidence[0]["evidence_id"],),
            facts={"structural_claim": "fixture"},
            reasons=("evidence fixture decision",),
        )

    def synthesize_candidates(self, context, evidence):
        self._called("synthesize_candidates")
        candidate = {
            "schema": "nerve.optimizer.representation_candidate.v1",
            "candidate_id": "",
            "scope_ids": list(context.scope_ids),
            "source_contract_digests": list(context.source_contract_digests),
            "provider": self.identity.to_json(),
            "descriptor_id": self.descriptor_id,
            "evidence_refs": list(evidence.evidence_ids),
            "representation": {
                "kind": "fixture_structured_transform",
                "signal_formats": [{"name": "fixture_signal"}],
                "parameter_format": {"kind": "fixture_parameters"},
                "state_format": {"kind": "source_state"},
                "topology": {"kind": "fixture_pipeline"},
            },
            "target_predicate": {
                "capability_class": context.hardware_profile["capability_class"]
            },
            "behavioral_contract": {
                "mode": "exact",
                "proof_obligations": ["fixture_reconstruction"],
                "error_contract": None,
            },
            "artifact_declarations": [
                {"path": "codebooks/table.bin"},
                {"path": "corrections/residual.bin"},
                {"path": "fields/samples.bin"},
                {"path": "geometry/basis.bin"},
                {"path": "graphs/events.bin"},
                {"path": "indexes/search.bin"},
                {"path": "kernels/native_island.spv"},
                {"path": "parameters/sparse_weights.bin"},
                {"path": "programs/evaluator.bin"},
                {"path": "state/compact_layout.json"},
                {"path": "topology/events.bin"},
            ],
        }
        candidate["candidate_id"] = representation_candidate_id(candidate)
        return (candidate,)

    def emit_representation_ir(self, context, candidate):
        self._called("emit_representation_ir")
        return exact_representation_graph(
            candidate_id=candidate["candidate_id"],
            scope_ids=context.scope_ids,
            source_contract_digests=context.source_contract_digests,
            evidence_ref=context.evidence[0]["evidence_id"],
        )

    def lower_for_target(self, context, candidate, representation_ir):
        self._called("lower_for_target")
        return {
            "schema": "nerve.optimizer.fixture_target_lowering.v1",
            "capability_class": context.hardware_profile["capability_class"],
            "representation_schema": representation_ir["schema"],
        }

    def estimate_static_cost(
        self,
        context,
        candidate,
        representation_ir,
        target_lowering,
    ):
        self._called("estimate_static_cost")
        return StaticEstimate(
            feasible=True,
            permanent_bytes=128,
            transient_bytes=64,
            construction_nanoseconds=1_000,
            steady_state_work={"operations": 4},
            reasons=("fixture target is feasible",),
        )

    def construction_requirements(self, context, candidate):
        self._called("construction_requirements")
        return {
            "schema": "nerve.optimizer.candidate_build_plan.v1",
            "phases": [
                "semantic_construction",
                "ordinary_lowering",
                "physical_optimization",
            ],
            "source_inputs": [],
            "outputs": [
                {
                    "path": "codebooks/table.bin",
                    "kind": "codebook",
                    "lifetime": "residency",
                    "producer_phase": "semantic_construction",
                    "resident_bytes": 16,
                    "validator_id": "nonempty_binary",
                    "validation_contract": {
                        "minimum_byte_count": 1,
                        "byte_multiple": 1,
                    },
                },
                {
                    "path": "corrections/residual.bin",
                    "kind": "correction_artifact",
                    "lifetime": "residency",
                    "producer_phase": "semantic_construction",
                    "resident_bytes": 16,
                    "validator_id": "nonempty_binary",
                    "validation_contract": {
                        "minimum_byte_count": 1,
                        "byte_multiple": 1,
                    },
                },
                {
                    "path": "fields/samples.bin",
                    "kind": "sampled_field",
                    "lifetime": "residency",
                    "producer_phase": "semantic_construction",
                    "resident_bytes": 16,
                    "validator_id": "nonempty_binary",
                    "validation_contract": {
                        "minimum_byte_count": 1,
                        "byte_multiple": 1,
                    },
                },
                {
                    "path": "geometry/basis.bin",
                    "kind": "geometry",
                    "lifetime": "residency",
                    "producer_phase": "semantic_construction",
                    "resident_bytes": 16,
                    "validator_id": "nonempty_binary",
                    "validation_contract": {
                        "minimum_byte_count": 1,
                        "byte_multiple": 1,
                    },
                },
                {
                    "path": "graphs/events.bin",
                    "kind": "graph",
                    "lifetime": "residency",
                    "producer_phase": "ordinary_lowering",
                    "resident_bytes": 16,
                    "validator_id": "nonempty_binary",
                    "validation_contract": {
                        "minimum_byte_count": 1,
                        "byte_multiple": 1,
                    },
                },
                {
                    "path": "indexes/search.bin",
                    "kind": "index",
                    "lifetime": "residency",
                    "producer_phase": "semantic_construction",
                    "resident_bytes": 16,
                    "validator_id": "nonempty_binary",
                    "validation_contract": {
                        "minimum_byte_count": 1,
                        "byte_multiple": 1,
                    },
                },
                {
                    "path": "kernels/native_island.spv",
                    "kind": "spirv",
                    "lifetime": "mount",
                    "producer_phase": "physical_optimization",
                    "resident_bytes": 0,
                    "validator_id": "spirv_module",
                    "validation_contract": {"minimum_version": 65536},
                },
                {
                    "path": "parameters/sparse_weights.bin",
                    "kind": "sparse_parameter",
                    "lifetime": "residency",
                    "producer_phase": "semantic_construction",
                    "resident_bytes": 128,
                    "validator_id": "nonempty_binary",
                    "validation_contract": {
                        "minimum_byte_count": 1,
                        "byte_multiple": 1,
                    },
                },
                {
                    "path": "programs/evaluator.bin",
                    "kind": "generated_program",
                    "lifetime": "residency",
                    "producer_phase": "semantic_construction",
                    "resident_bytes": 16,
                    "validator_id": "nonempty_binary",
                    "validation_contract": {
                        "minimum_byte_count": 1,
                        "byte_multiple": 1,
                    },
                },
                {
                    "path": "state/compact_layout.json",
                    "kind": "state_layout",
                    "lifetime": "mount",
                    "producer_phase": "semantic_construction",
                    "resident_bytes": 32,
                    "validator_id": "json_contract",
                    "validation_contract": {
                        "schema": "fixture.state_layout.v1",
                        "object_required": True,
                    },
                },
                {
                    "path": "topology/events.bin",
                    "kind": "event_topology",
                    "lifetime": "residency",
                    "producer_phase": "ordinary_lowering",
                    "resident_bytes": 64,
                    "validator_id": "nonempty_binary",
                    "validation_contract": {
                        "minimum_byte_count": 1,
                        "byte_multiple": 1,
                    },
                },
            ],
            "resource_limits": {
                "maximum_construction_time_ns": None,
                "maximum_temporary_bytes": None,
                "maximum_staging_bytes": None,
            },
        }

    def mount_requirements(self, context, candidate):
        self._called("mount_requirements")
        return {
            "schema": "nerve.optimizer.fixture_mount_requirements.v1",
            "resident_bytes": 128,
        }

    def proof_or_error_contract(self, context, candidate):
        self._called("proof_or_error_contract")
        return candidate["behavioral_contract"]

    def benchmark_workloads(self, context, candidate):
        self._called("benchmark_workloads")
        return (
            create_benchmark_workload(
                name="fixture decode",
                execution_phase="decode",
                activation_batch_width=1,
                context_size=4096,
                state_size=4096,
                stream_count=1,
                mount_mode="resident_reuse",
                boundary_mode="local",
                input_artifact={
                    "path": "fixtures/decode-input.bin",
                    "digest": staged_artifact_digest(b"fixture input"),
                },
                initial_state_artifact={
                    "path": "fixtures/decode-state.bin",
                    "digest": staged_artifact_digest(b"fixture state"),
                },
                controls={"sampler": "greedy"},
                randomness_algorithm="fixture-counter",
                seeds=(1, 2),
                deterministic_replay_required=True,
                permit_sampling_variance=False,
                permit_numerical_nondeterminism=False,
                permit_speculative_schedule_variance=False,
                useful_work_unit="tokens",
                minimum_useful_work_units=128,
                completion_condition="semantic_stop_or_allowance",
                output_allowance=65_536,
                output_allowance_basis={
                    "kind": "declared_model_limit",
                    "artifact": {
                        "path": "fixtures/model-limits.json",
                        "digest": staged_artifact_digest(
                            b'{"max_output_tokens":65536}'
                        ),
                    },
                    "json_pointer": "/max_output_tokens",
                    "declared_limit": 65_536,
                },
                sustained_window_count=4,
            ).to_json(),
            create_benchmark_workload(
                name="fixture prefill wide multi-stream cross-device",
                execution_phase="prefill",
                activation_batch_width=8,
                context_size=32_768,
                state_size=8_192,
                stream_count=4,
                mount_mode="cold",
                boundary_mode="cross_device",
                input_artifact={
                    "path": "fixtures/prefill-input.bin",
                    "digest": staged_artifact_digest(b"fixture prefill input"),
                },
                initial_state_artifact={
                    "path": "fixtures/prefill-state.bin",
                    "digest": staged_artifact_digest(b"fixture prefill state"),
                },
                controls={"scheduler": "multi_stream"},
                randomness_algorithm="fixture-counter",
                seeds=(1, 2),
                deterministic_replay_required=True,
                permit_sampling_variance=False,
                permit_numerical_nondeterminism=False,
                permit_speculative_schedule_variance=False,
                useful_work_unit="activation_rows",
                minimum_useful_work_units=1_024,
                completion_condition="all_prefill_rows_committed",
                output_allowance=None,
                output_allowance_basis={"kind": "unlimited"},
                sustained_window_count=4,
            ).to_json(),
        )

    def validation_requirements(self, context, candidate):
        self._called("validation_requirements")
        return {
            "schema": "nerve.optimizer.fixture_validation_requirements.v1",
            "checks": ["exact_output"],
        }


def _descriptors():
    return load_builtin_representation_descriptors()


def _descriptor_id() -> str:
    for descriptor in _descriptors().descriptors:
        document = descriptor.to_json()
        if document["identity"]["name"] == "structured_transform_with_exceptions":
            return str(document["descriptor_id"])
    raise AssertionError("fixture representation descriptor is missing")


def _problem() -> ProviderProblem:
    fixtures = contract_fixtures()
    return ProviderProblem.from_documents(
        package_id="fixture_package",
        scopes=[fixtures[0]],
        source_contracts=[fixtures[1]],
        evidence=[fixtures[2]],
        hardware_profile=hardware_profile_contract(),
    )


def test_provider_registry_executes_the_complete_interface_deterministically():
    provider = FixtureProvider("fixture.provider", _descriptor_id())
    registry = ProviderRegistry.from_providers(
        descriptors=_descriptors(),
        providers=[provider],
    )

    first = registry.run(_problem())
    second = registry.run(_problem())

    assert [evaluation.status for evaluation in first.evaluations] == ["completed"]
    assert len(first.candidates) == 1
    assert first.candidates[0].candidate_id == second.candidates[0].candidate_id
    assert first.candidates[0].static_estimate.feasible is True
    assert (
        first.candidates[0].representation_ir.to_json()["candidate_id"]
        == first.candidates[0].candidate_id
    )
    expected_calls = {
        "match_semantics",
        "match_structure",
        "analyze_evidence",
        "synthesize_candidates",
        "emit_representation_ir",
        "lower_for_target",
        "estimate_static_cost",
        "construction_requirements",
        "mount_requirements",
        "proof_or_error_contract",
        "benchmark_workloads",
        "validation_requirements",
    }
    assert set(provider.calls) == expected_calls


def test_provider_can_decline_without_error_or_candidate_construction():
    provider = FixtureProvider(
        "fixture.declines",
        _descriptor_id(),
        structural_match=False,
    )
    report = ProviderRegistry.from_providers(
        descriptors=_descriptors(),
        providers=[provider],
    ).run(_problem())

    assert report.evaluations[0].status == "declined"
    assert report.candidates == ()
    assert provider.calls == ["match_semantics", "match_structure"]


def test_hardware_availability_without_structural_evidence_cannot_make_candidate():
    provider = FixtureProvider(
        "fixture.no_evidence",
        _descriptor_id(),
        omit_structural_evidence=True,
    )
    report = ProviderRegistry.from_providers(
        descriptors=_descriptors(),
        providers=[provider],
    ).run(_problem())

    assert report.evaluations[0].status == "failed"
    assert "must cite evidence" in report.evaluations[0].error["message"]
    assert report.candidates == ()
    assert "synthesize_candidates" not in provider.calls


def test_provider_failures_are_isolated_from_other_registered_providers():
    failing = FixtureProvider(
        "a.fixture.failing",
        _descriptor_id(),
        fail_at="analyze_evidence",
    )
    healthy = FixtureProvider("b.fixture.healthy", _descriptor_id())
    report = ProviderRegistry.from_providers(
        descriptors=_descriptors(),
        providers=[healthy, failing],
    ).run(_problem())

    assert [evaluation.status for evaluation in report.evaluations] == [
        "failed",
        "completed",
    ]
    assert report.evaluations[0].error == {
        "type": "RuntimeError",
        "message": "fixture failure at analyze_evidence",
    }
    assert len(report.candidates) == 1
    assert report.candidates[0].provider == healthy.identity


def test_semantically_duplicate_candidates_are_eliminated_across_providers():
    first = FixtureProvider("a.fixture.provider", _descriptor_id())
    second = FixtureProvider("b.fixture.provider", _descriptor_id())
    forward = ProviderRegistry.from_providers(
        descriptors=_descriptors(),
        providers=[first, second],
    ).run(_problem())
    reverse = ProviderRegistry.from_providers(
        descriptors=_descriptors(),
        providers=[second, first],
    ).run(_problem())

    assert len(forward.candidates) == 1
    assert len(forward.duplicate_candidates) == 1
    assert forward.candidates[0].candidate_id == reverse.candidates[0].candidate_id
    assert forward.duplicate_candidates == reverse.duplicate_candidates


def test_provider_context_returns_copies_and_preserves_problem_contracts():
    provider = FixtureProvider(
        "fixture.mutates_copy",
        _descriptor_id(),
        mutate_scope_copy=True,
    )
    problem = _problem()
    report = ProviderRegistry.from_providers(
        descriptors=_descriptors(),
        providers=[provider],
    ).run(problem)

    assert report.evaluations[0].status == "completed"
    assert (
        problem.bind_descriptor(_descriptors().get(_descriptor_id())).scopes[0]["kind"]
        == "semantic_module"
    )


def test_registration_requires_complete_provider_and_registered_descriptor():
    provider = FixtureProvider("fixture.provider", "missing_descriptor")
    with pytest.raises(
        ContractValidationError,
        match="unregistered descriptor",
    ):
        ProviderRegistry(descriptors=_descriptors()).register(provider)

    class IncompleteProvider:
        identity = ProviderIdentity("fixture.incomplete", "1")
        descriptor_id = _descriptor_id()

    with pytest.raises(
        ContractValidationError,
        match="required methods",
    ):
        ProviderRegistry(descriptors=_descriptors()).register(IncompleteProvider())


def test_candidate_content_mutation_invalidates_deterministic_identity():
    provider = FixtureProvider("fixture.provider", _descriptor_id())
    report = ProviderRegistry.from_providers(
        descriptors=_descriptors(),
        providers=[provider],
    ).run(_problem())
    candidate = report.candidates[0].candidate.to_json()
    candidate["representation"]["kind"] = "mutated"

    with pytest.raises(
        ContractValidationError,
        match="canonical representation candidate",
    ):
        validate_contract(candidate)


def test_provider_cannot_emit_opaque_or_malformed_representation_ir():
    class OpaqueProvider(FixtureProvider):
        def emit_representation_ir(self, context, candidate):
            self._called("emit_representation_ir")
            return {
                "schema": "nerve.optimizer.provider_private_graph.v1",
                "candidate_id": candidate["candidate_id"],
            }

    provider = OpaqueProvider("fixture.opaque", _descriptor_id())
    report = ProviderRegistry.from_providers(
        descriptors=_descriptors(),
        providers=[provider],
    ).run(_problem())

    assert report.candidates == ()
    assert report.evaluations[0].status == "failed"
    assert "fields are invalid" in report.evaluations[0].error["message"]


def test_representation_graph_must_bind_to_candidate_scope_and_evidence():
    class MisboundProvider(FixtureProvider):
        def emit_representation_ir(self, context, candidate):
            self._called("emit_representation_ir")
            return exact_representation_graph(
                candidate_id="candidate_not_the_synthesized_candidate",
                scope_ids=context.scope_ids,
                source_contract_digests=context.source_contract_digests,
                evidence_ref=context.evidence[0]["evidence_id"],
            )

    provider = MisboundProvider("fixture.misbound", _descriptor_id())
    report = ProviderRegistry.from_providers(
        descriptors=_descriptors(),
        providers=[provider],
    ).run(_problem())

    assert report.candidates == ()
    assert report.evaluations[0].status == "failed"
    assert "candidate_id does not match" in report.evaluations[0].error["message"]
