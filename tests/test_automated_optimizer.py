from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from nerve.compilation import (
    ModelCompileCancelled,
    ModelCompileError,
    check_compile_cancelled,
)
from nerve.model_package_validation import validate_compiled_package
from nerve.representation_optimizer.automation import (
    CapacityLeaseState,
    CandidateToolchain,
    OptimizationBudget,
    OptimizationTarget,
    VerifiedCapacityLeaseManager,
    run_automated_optimizer,
    validate_report_directory,
)
from nerve.representation_optimizer.contracts import (
    canonical_json_bytes,
    contract_digest,
    device_state_digest,
    representation_candidate_id,
)
from nerve.representation_optimizer.providers import ProviderRegistry, StaticEstimate
from nerve.representation_optimizer.providers.source_artifacts import (
    PackageSourceArtifactResolver,
)
from nerve.representation_optimizer.qualification import QualificationRegime
from nerve.representation_optimizer.validation.proofs import (
    ProofVerifierRegistry,
)
from tests.test_candidate_benchmarking import (
    AdapterBehavior,
    FixtureExecutionAdapter,
)
from tests.test_candidate_staging import (
    CancellingStreamConstructor,
    CompletePhysicalOptimizer,
    CompleteRelowerer,
    CompleteSemanticConstructor,
)
from tests.test_candidate_validation import (
    FixtureProofVerifier,
    FixtureValidationAdapter,
    ValidationBehavior,
)
from tests.test_package_integrity import minimal_package
from tests.test_representation_optimizer_contracts import (
    hardware_profile_contract,
)
from tests.test_representation_providers import (
    FixtureProvider,
    _descriptor_id,
    _descriptors,
)


class CompleteToolchains:
    def resolve(self, plan):
        return CandidateToolchain(
            semantic_constructor=CompleteSemanticConstructor([]),
            ordinary_relowerer=CompleteRelowerer([]),
            physical_optimizer=CompletePhysicalOptimizer([]),
        )


class FailingToolchains:
    def resolve(self, plan):
        return CandidateToolchain(
            semantic_constructor=CompleteSemanticConstructor([]),
            ordinary_relowerer=CompleteRelowerer([], fail=True),
            physical_optimizer=CompletePhysicalOptimizer([]),
        )


class SelectiveToolchains:
    def resolve(self, plan):
        if plan.provider.provider_id == "fixture.bad-construction":
            return FailingToolchains().resolve(plan)
        return CompleteToolchains().resolve(plan)


class CancellingToolchains:
    def __init__(self, cancel_state: dict[str, bool]) -> None:
        self.cancel_state = cancel_state

    def resolve(self, plan):
        return CandidateToolchain(
            semantic_constructor=CancellingStreamConstructor(self.cancel_state),
            ordinary_relowerer=CompleteRelowerer([]),
            physical_optimizer=CompletePhysicalOptimizer([]),
        )


class CancellingExecutionSession:
    def __init__(self, delegate, cancel_state: dict[str, bool]) -> None:
        self.delegate = delegate
        self.cancel_state = cancel_state

    @property
    def mount_event(self):
        return self.delegate.mount_event

    def execute(self, request):
        observation = self.delegate.execute(request)
        self.cancel_state["requested"] = True
        return observation

    def close(self):
        return self.delegate.close()


class CancellingExecutionAdapter(FixtureExecutionAdapter):
    def __init__(self, cancel_state: dict[str, bool]) -> None:
        super().__init__()
        self.cancel_state = cancel_state

    def open_session(self, request):
        return CancellingExecutionSession(
            super().open_session(request),
            self.cancel_state,
        )


class DistinctFixtureProvider(FixtureProvider):
    def synthesize_candidates(self, context, evidence):
        candidates = super().synthesize_candidates(context, evidence)
        candidate = candidates[0]
        candidate["representation"]["kind"] = (
            f"fixture_structured_transform:{self.provider_id}"
        )
        candidate["candidate_id"] = representation_candidate_id(candidate)
        return (candidate,)


class UncalibratedConstructionProvider(FixtureProvider):
    def estimate_static_cost(
        self,
        context,
        candidate,
        representation_ir,
        target_lowering,
    ):
        estimate = super().estimate_static_cost(
            context,
            candidate,
            representation_ir,
            target_lowering,
        )
        return StaticEstimate(
            feasible=estimate.feasible,
            permanent_bytes=estimate.permanent_bytes,
            transient_bytes=estimate.transient_bytes,
            construction_nanoseconds=None,
            steady_state_work=estimate.steady_state_work,
            reasons=("construction duration has not been calibrated for this target",),
        )


@dataclass
class CountingLeaseManager:
    active: int = 0
    maximum_active: int = 0
    acquisitions: int = 0
    releases: int = 0

    @contextmanager
    def acquire(self, target):
        assert self.active == 0
        self.active += 1
        self.acquisitions += 1
        self.maximum_active = max(self.maximum_active, self.active)
        try:
            yield
        finally:
            self.active -= 1
            self.releases += 1


def _package(tmp_path: Path) -> Path:
    package = tmp_path / "package"
    manifest = minimal_package(package)
    (package / "vulkan_resident_package.json").write_bytes(
        canonical_json_bytes(manifest) + b"\n"
    )
    return package


def _target(
    *,
    toolchains=None,
    lease=None,
    benchmark_behavior: AdapterBehavior | None = None,
    validation_behavior: ValidationBehavior | None = None,
) -> tuple[OptimizationTarget, CountingLeaseManager]:
    profile = hardware_profile_contract()
    lease = lease or CountingLeaseManager()
    conditions = {
        "devices": [
            {
                "device_id": profile["hardware_identity"]["stable_device_id"],
                "hardware_profile_digest": contract_digest(profile),
                "capability_class": profile["capability_class"],
                "api": profile["provenance"]["api"],
            }
        ],
        "placement": {"fixture_scope": "vulkan:fixture"},
        "controls": {
            "scheduler": "normal",
            "speculative_draft_tokens": 0,
        },
        "environment": {"power_profile": "matched"},
        "capacity_reservation_digest": device_state_digest({"fixture_state": "capacity_available"}),
        "residency_scope": "capacity_partition",
    }
    return (
        OptimizationTarget(
            target_id="fixture-vulkan-target",
            synthesis_profile=profile,
            hardware_profiles=(profile,),
            matched_conditions=conditions,
            qualification_regime=QualificationRegime(),
            requires_device_lease=True,
            toolchains=toolchains or CompleteToolchains(),
            benchmark_adapter=FixtureExecutionAdapter(benchmark_behavior),
            validation_adapter=FixtureValidationAdapter(validation_behavior),
            proof_verifiers=ProofVerifierRegistry.from_verifiers(
                (FixtureProofVerifier(),)
            ),
            lease_manager=lease,
            estimate_execution_nanoseconds=lambda _plan, _policy: 1_000_000,
        ),
        lease,
    )


def _budget(*, maximum_candidates: int = 1) -> OptimizationBudget:
    return OptimizationBudget(
        maximum_scopes=10,
        maximum_candidates=maximum_candidates,
        maximum_permanent_bytes=1_000_000,
        maximum_transient_bytes=1_000_000,
        maximum_construction_nanoseconds=1_000_000_000,
        maximum_execution_nanoseconds=1_000_000_000,
        maximum_experiment_invocations=maximum_candidates * 4_000,
    )


def _providers(*providers) -> ProviderRegistry:
    return ProviderRegistry.from_providers(
        descriptors=_descriptors(),
        providers=providers
        or (FixtureProvider("fixture.automation", _descriptor_id()),),
    )


def test_unattended_loop_publishes_only_faster_fully_valid_candidate(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)
    target, lease = _target()

    outcome = run_automated_optimizer(
        package_dir=package,
        source_artifacts=PackageSourceArtifactResolver(package),
        output_package_dir=tmp_path / "optimized",
        run_root=tmp_path / "run",
        providers=_providers(
            FixtureProvider(
                "fixture.broken-provider",
                _descriptor_id(),
                fail_at="analyze_evidence",
            ),
            FixtureProvider("fixture.automation", _descriptor_id()),
        ),
        targets=(target,),
        budget=_budget(),
    )

    assert outcome.output_package_dir == tmp_path / "optimized"
    assert outcome.report["status"] == "completed"
    assert outcome.report["summary"]["promotion_count"] == 1
    assert outcome.report["summary"]["materially_faster_count"] == 1
    assert outcome.report["summary"]["provider_failure_count"] >= 1
    assert outcome.report["summary"]["analysis_failure_count"] >= 1
    assert outcome.report["candidates"][0]["status"] == "published"
    assert outcome.report["candidates"][0]["validation_status"] == "passed"
    assert lease.maximum_active == 1
    assert lease.acquisitions == 3
    assert lease.releases == 3
    assert lease.active == 0
    structures = [
        structure
        for scope in outcome.report["scopes"]
        for structure in scope["structures"]
    ]
    assert structures
    assert all("claims" not in structure for structure in structures)
    assert all(
        structure["claim_summary"]["total"] > 0
        and (tmp_path / "run" / structure["evidence_ref"]).is_file()
        for structure in structures
    )
    assert validate_report_directory(tmp_path / "run") == outcome.report
    manifest = validate_compiled_package(
        tmp_path / "optimized",
        __import__("json").loads(
            (tmp_path / "optimized" / "vulkan_resident_package.json").read_text()
        ),
    )
    assert manifest is None
    assert not (package / "optimization" / "implementations").exists()


def test_scope_routing_skips_irrelevant_scopes_before_analysis_and_budget(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)
    target, _lease = _target()

    outcome = run_automated_optimizer(
        package_dir=package,
        source_artifacts=PackageSourceArtifactResolver(package),
        output_package_dir=tmp_path / "optimized",
        run_root=tmp_path / "run",
        providers=_providers(
            FixtureProvider(
                "fixture.sampler-only",
                _descriptor_id(),
                accepted_scope_kinds=("sampler",),
            )
        ),
        targets=(target,),
        budget=_budget(),
    )

    analyzed = [
        scope
        for scope in outcome.report["scopes"]
        if scope["status"] == "analyzed"
    ]
    skipped = [
        scope
        for scope in outcome.report["scopes"]
        if scope["status"] == "provider_skipped"
    ]
    assert len(analyzed) == 1
    assert analyzed[0]["kind"] == "sampler"
    assert len(skipped) == 9
    assert outcome.report["budget_usage"]["scopes"] == 1
    assert outcome.report["summary"]["analyzed_scope_count"] == 1
    assert {
        path.name for path in (tmp_path / "run" / "analysis").iterdir()
    } == {analyzed[0]["scope_id"]}


def test_scope_analysis_runs_only_the_matching_provider_requirements(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)
    target, _lease = _target()

    outcome = run_automated_optimizer(
        package_dir=package,
        source_artifacts=PackageSourceArtifactResolver(package),
        output_package_dir=tmp_path / "optimized",
        run_root=tmp_path / "run",
        providers=_providers(
            FixtureProvider(
                "fixture.graph-evidence-only",
                _descriptor_id(),
                accepted_scope_kinds=("sampler",),
                analyzer_ids=("semantic_graph_structure",),
            )
        ),
        targets=(target,),
        budget=_budget(),
    )

    analyzed = [
        scope for scope in outcome.report["scopes"] if scope["status"] == "analyzed"
    ]
    assert len(analyzed) == 1
    assert [
        structure["analyzer"]["id"] for structure in analyzed[0]["structures"]
    ] == ["semantic_graph_structure"]
    event = next(
        json.loads(line)
        for line in (tmp_path / "run" / "events.jsonl").read_text().splitlines()
        if json.loads(line)["phase"] == "analysis"
        and json.loads(line)["status"] == "completed"
    )
    assert event["details"]["analyzer_ids"] == ["semantic_graph_structure"]


def test_scope_analysis_rejects_an_unavailable_provider_analyzer(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)
    target, lease = _target()

    outcome = run_automated_optimizer(
        package_dir=package,
        source_artifacts=PackageSourceArtifactResolver(package),
        output_package_dir=tmp_path / "unused-output",
        run_root=tmp_path / "run",
        providers=_providers(
            FixtureProvider(
                "fixture.unknown-analyzer",
                _descriptor_id(),
                accepted_scope_kinds=("sampler",),
                analyzer_ids=("nonexistent_structure",),
            )
        ),
        targets=(target,),
        budget=_budget(),
    )

    assert outcome.report["status"] == "completed_no_changes"
    failed = [
        scope for scope in outcome.report["scopes"] if scope["status"] == "failed"
    ]
    assert len(failed) == 1
    assert "unavailable" in failed[0]["reason"]
    assert outcome.report["summary"]["analysis_failure_count"] == 1
    assert lease.acquisitions == 0
    assert not (tmp_path / "unused-output").exists()


def test_scope_routing_rejects_ambiguous_provider_analyzer_requirements(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)
    target, lease = _target()

    with pytest.raises(
        ModelCompileError,
        match="automated optimizer failed safely",
    ):
        run_automated_optimizer(
            package_dir=package,
            source_artifacts=PackageSourceArtifactResolver(package),
            output_package_dir=tmp_path / "unused-output",
            run_root=tmp_path / "run",
            providers=_providers(
                FixtureProvider(
                    "fixture.duplicate-analyzers",
                    _descriptor_id(),
                    accepted_scope_kinds=("sampler",),
                    analyzer_ids=(
                        "semantic_graph_structure",
                        "semantic_graph_structure",
                    ),
                )
            ),
            targets=(target,),
            budget=_budget(),
        )

    report = json.loads((tmp_path / "run" / "report.json").read_text())
    assert report["status"] == "failed"
    assert "sorted, unique" in report["publication"]["reason"]
    assert lease.acquisitions == 0


def test_optimizer_completes_without_changes_when_no_provider_accepts_a_scope(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)
    target, lease = _target()

    outcome = run_automated_optimizer(
        package_dir=package,
        source_artifacts=PackageSourceArtifactResolver(package),
        output_package_dir=tmp_path / "unused-output",
        run_root=tmp_path / "run",
        providers=_providers(
            FixtureProvider(
                "fixture.no-static-match",
                _descriptor_id(),
                accepted_scope_kinds=("nonexistent_scope_kind",),
            )
        ),
        targets=(target,),
        budget=_budget(),
    )

    assert outcome.report["status"] == "completed_no_changes"
    assert outcome.output_package_dir == package
    assert outcome.report["budget_usage"]["scopes"] == 0
    assert outcome.report["summary"]["analyzed_scope_count"] == 0
    assert all(
        scope["status"] == "provider_skipped"
        for scope in outcome.report["scopes"]
    )
    assert outcome.report["provider_evaluations"] == []
    assert lease.acquisitions == 0
    assert not (tmp_path / "unused-output").exists()


def test_budget_rejects_whole_candidate_without_running_partial_experiment(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)
    target, lease = _target()

    outcome = run_automated_optimizer(
        package_dir=package,
        source_artifacts=PackageSourceArtifactResolver(package),
        output_package_dir=tmp_path / "unused-output",
        run_root=tmp_path / "run",
        providers=_providers(),
        targets=(target,),
        budget=_budget(maximum_candidates=0),
    )

    candidate = outcome.report["candidates"][0]
    assert outcome.report["status"] == "completed_no_changes"
    assert outcome.output_package_dir == package
    assert candidate["status"] == "rejected"
    assert "maximum_candidates" in candidate["rejection_reasons"][0]
    assert candidate["construction_status"] is None
    assert lease.acquisitions == 0
    assert not (tmp_path / "unused-output").exists()


def test_bounded_construction_budget_rejects_uncalibrated_estimate(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)
    target, lease = _target()

    outcome = run_automated_optimizer(
        package_dir=package,
        source_artifacts=PackageSourceArtifactResolver(package),
        output_package_dir=tmp_path / "unused-output",
        run_root=tmp_path / "run",
        providers=_providers(
            UncalibratedConstructionProvider(
                "fixture.uncalibrated-construction",
                _descriptor_id(),
            )
        ),
        targets=(target,),
        budget=_budget(),
    )

    candidate = outcome.report["candidates"][0]
    budget = json.loads(
        (tmp_path / "run" / candidate["budget_decision_ref"]).read_text()
    )
    assert candidate["status"] == "rejected"
    assert budget["cost"]["construction_nanoseconds"] is None
    assert any(
        "no calibrated construction-cost estimate" in reason
        for reason in candidate["rejection_reasons"]
    )
    assert candidate["construction_status"] is None
    assert lease.acquisitions == 0


def test_experiment_budget_counts_every_planned_role_execution(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)
    target, lease = _target()
    budget = replace(
        _budget(),
        maximum_experiment_invocations=29,
    )

    outcome = run_automated_optimizer(
        package_dir=package,
        source_artifacts=PackageSourceArtifactResolver(package),
        output_package_dir=tmp_path / "unused-output",
        run_root=tmp_path / "run",
        providers=_providers(),
        targets=(target,),
        budget=budget,
    )

    candidate = outcome.report["candidates"][0]
    decision = json.loads(
        (tmp_path / "run" / candidate["budget_decision_ref"]).read_text()
    )
    assert decision["cost"]["experiment_invocations"] == 30
    assert candidate["status"] == "rejected"
    assert any(
        "maximum_experiment_invocations: 30 > 29" in reason
        for reason in candidate["rejection_reasons"]
    )
    assert lease.acquisitions == 0


def test_candidate_failure_is_audited_and_releases_every_lease(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)
    target, lease = _target(toolchains=FailingToolchains())

    outcome = run_automated_optimizer(
        package_dir=package,
        source_artifacts=PackageSourceArtifactResolver(package),
        output_package_dir=tmp_path / "unused-output",
        run_root=tmp_path / "run",
        providers=_providers(),
        targets=(target,),
        budget=_budget(),
    )

    candidate = outcome.report["candidates"][0]
    assert outcome.report["status"] == "completed_no_changes"
    assert candidate["status"] == "failed"
    assert candidate["construction_status"] == "failed"
    assert "ordinary lowering fixture failure" in candidate["rejection_reasons"][0]
    assert lease.active == 0
    assert lease.acquisitions == 0
    candidate_staging = tmp_path / "run" / "workspaces" / "candidates" / ".staging"
    assert not any(candidate_staging.iterdir())


def test_failed_candidate_does_not_prevent_independent_candidate_promotion(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)
    target, lease = _target(toolchains=SelectiveToolchains())

    outcome = run_automated_optimizer(
        package_dir=package,
        source_artifacts=PackageSourceArtifactResolver(package),
        output_package_dir=tmp_path / "optimized",
        run_root=tmp_path / "run",
        providers=_providers(
            DistinctFixtureProvider(
                "fixture.bad-construction",
                _descriptor_id(),
            ),
            DistinctFixtureProvider(
                "fixture.good-construction",
                _descriptor_id(),
            ),
        ),
        targets=(target,),
        budget=_budget(maximum_candidates=2),
    )

    candidates = {
        candidate["provider"]["id"]: candidate
        for candidate in outcome.report["candidates"]
    }
    assert candidates["fixture.bad-construction"]["status"] == "failed"
    assert candidates["fixture.good-construction"]["status"] == "published"
    assert outcome.report["summary"]["promotion_count"] == 1
    assert lease.maximum_active == 1
    assert lease.acquisitions == lease.releases == 3
    lifecycle_states = {
        candidate["candidate_id"]: candidate["state"]
        for candidate in outcome.report["session"]["candidates"]
    }
    assert lifecycle_states == {
        candidate["candidate_id"]: candidate["status"]
        for candidate in outcome.report["candidates"]
    }
    assert (
        tmp_path
        / "run"
        / "promotions"
        / f"{candidates['fixture.good-construction']['promotion_id']}.json"
    ).is_file()


def test_device_target_cannot_use_noop_lease_manager(tmp_path: Path) -> None:
    from nerve.representation_optimizer.automation import NoDeviceLeaseManager

    package = _package(tmp_path)
    target, _ = _target(lease=NoDeviceLeaseManager())

    outcome = run_automated_optimizer(
        package_dir=package,
        source_artifacts=PackageSourceArtifactResolver(package),
        output_package_dir=tmp_path / "unused-output",
        run_root=tmp_path / "run",
        providers=_providers(),
        targets=(target,),
        budget=_budget(),
    )

    assert outcome.report["candidates"][0]["status"] == "failed"
    assert (
        "requires a real device lease manager"
        in (outcome.report["candidates"][0]["failure"]["message"])
    )


def test_materially_faster_but_inaccurate_candidate_is_not_published(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)
    target, lease = _target(
        validation_behavior=ValidationBehavior(invalid_stage="full_local")
    )

    outcome = run_automated_optimizer(
        package_dir=package,
        source_artifacts=PackageSourceArtifactResolver(package),
        output_package_dir=tmp_path / "unused-output",
        run_root=tmp_path / "run",
        providers=_providers(),
        targets=(target,),
        budget=_budget(),
    )

    candidate = outcome.report["candidates"][0]
    assert outcome.report["status"] == "completed_no_changes"
    assert outcome.report["summary"]["materially_faster_count"] == 1
    assert outcome.report["summary"]["faster_but_invalid_count"] == 1
    assert candidate["benchmark_decision"] == "materially_faster"
    assert candidate["validation_status"] == "rejected"
    assert candidate["status"] == "rejected"
    assert not (tmp_path / "unused-output").exists()
    assert lease.acquisitions == lease.releases == 3


def test_slower_candidate_is_rejected_without_full_behavioral_execution(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)
    target, lease = _target(
        benchmark_behavior=AdapterBehavior(candidate_duration_ns=1_100_000)
    )

    outcome = run_automated_optimizer(
        package_dir=package,
        source_artifacts=PackageSourceArtifactResolver(package),
        output_package_dir=tmp_path / "unused-output",
        run_root=tmp_path / "run",
        providers=_providers(),
        targets=(target,),
        budget=_budget(),
    )

    candidate = outcome.report["candidates"][0]
    assert candidate["benchmark_decision"] == "not_materially_faster"
    assert candidate["validation_status"] == "rejected"
    assert candidate["status"] == "rejected"
    assert outcome.report["summary"]["materially_faster_count"] == 0
    assert lease.acquisitions == lease.releases == 2


def test_report_detects_truncated_event_journal(tmp_path: Path) -> None:
    package = _package(tmp_path)
    target, _ = _target()
    run_automated_optimizer(
        package_dir=package,
        source_artifacts=PackageSourceArtifactResolver(package),
        output_package_dir=tmp_path / "unused-output",
        run_root=tmp_path / "run",
        providers=_providers(),
        targets=(target,),
        budget=_budget(maximum_candidates=0),
    )
    events = tmp_path / "run" / "events.jsonl"
    lines = events.read_text().splitlines()
    events.write_text("\n".join(lines[:-1]) + "\n")

    with pytest.raises(ModelCompileError, match="event count"):
        validate_report_directory(tmp_path / "run")


def test_report_detects_missing_event_evidence(tmp_path: Path) -> None:
    package = _package(tmp_path)
    target, _ = _target()
    outcome = run_automated_optimizer(
        package_dir=package,
        source_artifacts=PackageSourceArtifactResolver(package),
        output_package_dir=tmp_path / "unused-output",
        run_root=tmp_path / "run",
        providers=_providers(),
        targets=(target,),
        budget=_budget(maximum_candidates=0),
    )
    decision = tmp_path / "run" / outcome.report["candidates"][0]["budget_decision_ref"]
    decision.unlink()

    with pytest.raises(ModelCompileError, match="evidence is missing"):
        validate_report_directory(tmp_path / "run")


def test_report_structure_index_is_checked_against_canonical_evidence(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)
    target, _ = _target()
    outcome = run_automated_optimizer(
        package_dir=package,
        source_artifacts=PackageSourceArtifactResolver(package),
        output_package_dir=tmp_path / "unused-output",
        run_root=tmp_path / "run",
        providers=_providers(),
        targets=(target,),
        budget=_budget(maximum_candidates=0),
    )
    structure = next(
        structure
        for scope in outcome.report["scopes"]
        for structure in scope["structures"]
    )
    evidence_path = tmp_path / "run" / structure["evidence_ref"]
    evidence = json.loads(evidence_path.read_text())
    evidence["claims"][0]["facts"]["tampered"] = True
    evidence_path.write_text(json.dumps(evidence))

    with pytest.raises(
        ModelCompileError,
        match="analysis.*digest|identity|evidence_id",
    ):
        validate_report_directory(tmp_path / "run")


def test_publication_failure_leaves_source_and_destination_unambiguous(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nerve.representation_optimizer.automation.orchestrator as orchestrator

    package = _package(tmp_path)
    source_files = {
        path.relative_to(package).as_posix(): path.read_bytes()
        for path in package.rglob("*")
        if path.is_file()
    }
    target, lease = _target()

    def fail_publication(**_kwargs):
        raise RuntimeError("fixture publication failure")

    monkeypatch.setattr(
        orchestrator,
        "publish_promoted_package",
        fail_publication,
    )
    with pytest.raises(ModelCompileError, match="failed safely"):
        run_automated_optimizer(
            package_dir=package,
            source_artifacts=PackageSourceArtifactResolver(package),
            output_package_dir=tmp_path / "optimized",
            run_root=tmp_path / "run",
            providers=_providers(),
            targets=(target,),
            budget=_budget(),
        )

    report = validate_report_directory(tmp_path / "run")
    assert report["status"] == "failed"
    assert report["publication"]["status"] == "failed"
    assert not (tmp_path / "optimized").exists()
    assert source_files == {
        path.relative_to(package).as_posix(): path.read_bytes()
        for path in package.rglob("*")
        if path.is_file()
    }
    assert lease.acquisitions == lease.releases


def test_cancellation_before_analysis_publishes_terminal_report(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)
    target, lease = _target()

    with pytest.raises(ModelCompileCancelled, match="cancelled safely"):
        run_automated_optimizer(
            package_dir=package,
            source_artifacts=PackageSourceArtifactResolver(package),
            output_package_dir=tmp_path / "optimized",
            run_root=tmp_path / "run",
            providers=_providers(),
            targets=(target,),
            budget=_budget(),
            cancel_requested=lambda: True,
        )

    report = validate_report_directory(tmp_path / "run")
    assert report["status"] == "cancelled"
    assert report["publication"]["status"] == "cancelled"
    assert report["scopes"] == []
    assert report["candidates"] == []
    assert report["output_package"] == str(package)
    assert not (tmp_path / "optimized").exists()
    assert lease.acquisitions == lease.releases == 0


def test_cancellation_during_construction_stops_the_whole_run(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)
    cancel_state = {"requested": False}
    target, lease = _target(
        toolchains=CancellingToolchains(cancel_state),
    )

    with pytest.raises(ModelCompileCancelled, match="cancelled safely"):
        run_automated_optimizer(
            package_dir=package,
            source_artifacts=PackageSourceArtifactResolver(package),
            output_package_dir=tmp_path / "optimized",
            run_root=tmp_path / "run",
            providers=_providers(),
            targets=(target,),
            budget=_budget(),
            cancel_requested=lambda: cancel_state["requested"],
        )

    report = validate_report_directory(tmp_path / "run")
    assert report["status"] == "cancelled"
    assert report["candidates"][0]["status"] == "cancelled"
    assert not (tmp_path / "optimized").exists()
    staging = tmp_path / "run" / "workspaces" / "candidates" / ".staging"
    assert not any(staging.iterdir())
    assert lease.acquisitions == lease.releases == 0


def test_cancellation_during_benchmark_releases_lease_and_stops_run(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)
    cancel_state = {"requested": False}
    target, lease = _target()
    adapter = CancellingExecutionAdapter(cancel_state)
    target = replace(target, benchmark_adapter=adapter)

    with pytest.raises(ModelCompileCancelled, match="cancelled safely"):
        run_automated_optimizer(
            package_dir=package,
            source_artifacts=PackageSourceArtifactResolver(package),
            output_package_dir=tmp_path / "optimized",
            run_root=tmp_path / "run",
            providers=_providers(),
            targets=(target,),
            budget=_budget(),
            cancel_requested=lambda: cancel_state["requested"],
        )

    report = validate_report_directory(tmp_path / "run")
    candidate = report["candidates"][0]
    assert report["status"] == "cancelled"
    assert candidate["status"] == "cancelled"
    assert candidate["prebenchmark_status"] == "passed"
    assert candidate["benchmark_decision"] is None
    assert adapter.closed_sessions == len(adapter.mount_requests)
    assert lease.active == 0
    assert lease.acquisitions == lease.releases == 2
    assert not (tmp_path / "optimized").exists()


def test_cancellation_before_publication_never_commits_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nerve.representation_optimizer.automation.orchestrator as orchestrator

    package = _package(tmp_path)
    target, lease = _target()
    cancel_state = {"requested": False}

    def cancel_publication(*, cancel_requested, **_kwargs):
        cancel_state["requested"] = True
        check_compile_cancelled(cancel_requested)
        raise AssertionError("cancellation checkpoint did not stop publication")

    monkeypatch.setattr(
        orchestrator,
        "publish_promoted_package",
        cancel_publication,
    )
    with pytest.raises(ModelCompileCancelled, match="cancelled safely"):
        run_automated_optimizer(
            package_dir=package,
            source_artifacts=PackageSourceArtifactResolver(package),
            output_package_dir=tmp_path / "optimized",
            run_root=tmp_path / "run",
            providers=_providers(),
            targets=(target,),
            budget=_budget(),
            cancel_requested=lambda: cancel_state["requested"],
        )

    report = validate_report_directory(tmp_path / "run")
    assert report["status"] == "cancelled"
    assert report["publication"]["status"] == "cancelled"
    assert report["summary"]["promotion_count"] == 1
    assert report["candidates"][0]["status"] == "promotable"
    assert not (tmp_path / "optimized").exists()
    assert lease.acquisitions == lease.releases


def test_verified_capacity_lease_checks_reservation_before_and_after(
    tmp_path: Path,
) -> None:
    digest = device_state_digest({"fixture_state": "capacity_available"})
    probe_results = [
        _capacity_lease_state(digest=digest, used_vram_bytes=100),
        _capacity_lease_state(digest=digest, used_vram_bytes=105),
    ]
    target, _ = _target(
        lease=VerifiedCapacityLeaseManager(
            lock_root=tmp_path / "device-locks",
            probe_capacity_reservation_state=lambda _target: probe_results.pop(0),
        )
    )

    with target.lease_manager.acquire(target):
        assert len(probe_results) == 1

    assert probe_results == []


def test_verified_capacity_lease_reports_post_execution_residency_leak(
    tmp_path: Path,
) -> None:
    digest = device_state_digest({"fixture_state": "capacity_available"})
    probe_results = [
        _capacity_lease_state(digest=digest, used_vram_bytes=100),
        _capacity_lease_state(digest=digest, used_vram_bytes=111),
    ]
    target, _ = _target(
        lease=VerifiedCapacityLeaseManager(
            lock_root=tmp_path / "device-locks",
            probe_capacity_reservation_state=lambda _target: probe_results.pop(0),
        )
    )

    with pytest.raises(ModelCompileError, match="did not restore"):
        with target.lease_manager.acquire(target):
            pass

    target, _ = _target(
        lease=VerifiedCapacityLeaseManager(
            lock_root=tmp_path / "device-locks",
            probe_capacity_reservation_state=lambda _target: _capacity_lease_state(
                digest=digest,
                used_vram_bytes=100,
            ),
        )
    )
    with target.lease_manager.acquire(target):
        pass


def test_verified_capacity_lease_waits_for_driver_release_settlement(
    tmp_path: Path,
) -> None:
    digest = device_state_digest({"fixture_state": "capacity_available"})
    probe_results = [
        _capacity_lease_state(
            digest=digest,
            used_vram_bytes=100,
            settle_timeout_ns=1_000_000_000,
        ),
        _capacity_lease_state(
            digest=digest,
            used_vram_bytes=111,
            settle_timeout_ns=1_000_000_000,
        ),
        _capacity_lease_state(
            digest=digest,
            used_vram_bytes=105,
            settle_timeout_ns=1_000_000_000,
        ),
    ]
    now = [0]
    sleeps: list[float] = []

    def settle(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += int(seconds * 1_000_000_000)

    target, _ = _target(
        lease=VerifiedCapacityLeaseManager(
            lock_root=tmp_path / "device-locks",
            probe_capacity_reservation_state=lambda _target: probe_results.pop(0),
            monotonic_ns=lambda: now[0],
            sleep=settle,
        )
    )

    with target.lease_manager.acquire(target):
        pass

    assert probe_results == []
    assert sleeps == [0.1]


def test_verified_capacity_lease_rejects_leak_after_settlement_deadline(
    tmp_path: Path,
) -> None:
    digest = device_state_digest({"fixture_state": "capacity_available"})
    before = _capacity_lease_state(
        digest=digest,
        used_vram_bytes=100,
        settle_timeout_ns=200_000_000,
    )
    leaked = _capacity_lease_state(
        digest=digest,
        used_vram_bytes=111,
        settle_timeout_ns=200_000_000,
    )
    probe_results = [before, leaked, leaked, leaked]
    now = [0]
    sleeps: list[float] = []

    def settle(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += int(seconds * 1_000_000_000)

    target, _ = _target(
        lease=VerifiedCapacityLeaseManager(
            lock_root=tmp_path / "device-locks",
            probe_capacity_reservation_state=lambda _target: probe_results.pop(0),
            monotonic_ns=lambda: now[0],
            sleep=settle,
        )
    )

    with pytest.raises(ModelCompileError, match="retained 11 VRAM bytes"):
        with target.lease_manager.acquire(target):
            pass

    assert probe_results == []
    assert sleeps == [0.1, 0.1]


def test_verified_capacity_lease_preserves_workloads_present_at_acquisition(
    tmp_path: Path,
) -> None:
    digest = device_state_digest({"fixture_state": "capacity_available"})
    probe_results = [
        _capacity_lease_state(
            digest=digest,
            used_vram_bytes=500,
            resident_pids=(42,),
        ),
        _capacity_lease_state(
            digest=digest,
            used_vram_bytes=100,
            resident_pids=(),
        ),
    ]
    target, _ = _target(
        lease=VerifiedCapacityLeaseManager(
            lock_root=tmp_path / "device-locks",
            probe_capacity_reservation_state=lambda _target: probe_results.pop(0),
        )
    )

    with pytest.raises(ModelCompileError, match="lost pre-existing resident"):
        with target.lease_manager.acquire(target):
            pass


def _capacity_lease_state(
    *,
    digest: str,
    used_vram_bytes: int,
    resident_pids: tuple[int, ...] = (),
    settle_timeout_ns: int = 0,
) -> CapacityLeaseState:
    return CapacityLeaseState(
        reservation_digest=digest,
        observations=(
            {
                "device_id": "fixture-device",
                "vram_total_bytes": 1_000,
                "vram_used_bytes": used_vram_bytes,
                "resident_processes": [
                    {"pid": pid}
                    for pid in resident_pids
                ],
            },
        ),
        release_vram_tolerance_bytes=10,
        release_settle_timeout_ns=settle_timeout_ns,
        release_poll_interval_ns=100_000_000,
    )
