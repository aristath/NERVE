from __future__ import annotations

import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import pytest

import nerve.representation_optimizer.promotion.publication as publication
from nerve.compilation import ModelCompileCancelled, ModelCompileError, read_json
from nerve.model_package_validation import validate_compiled_package
from nerve.representation_optimizer.analysis.evidence import (
    build_analysis_run,
    build_evidence,
    write_analysis_run,
)
from nerve.representation_optimizer.benchmarking.orchestrator import (
    benchmark_candidate,
)
from nerve.representation_optimizer.benchmarking.planning import (
    build_benchmark_plan,
)
from nerve.representation_optimizer.contracts import (
    canonical_json_bytes,
    contract_digest,
    device_state_digest,
    stable_contract_id,
)
from nerve.representation_optimizer.lifecycle import (
    CandidateState,
    OptimizationSession,
)
from nerve.representation_optimizer.providers import (
    ProviderProblem,
    ProviderRegistry,
)
from nerve.representation_optimizer.providers.source_artifacts import (
    PackageSourceArtifactResolver,
)
from nerve.representation_optimizer.promotion.contracts import (
    ImplementationRegistry,
    PromotionContractError,
    append_implementation_registry_entries,
    create_runtime_implementation_predicate,
    implementation_id,
    validate_implementation_registry_entry,
)
from nerve.representation_optimizer.promotion.orchestrator import (
    PreparedPromotion,
    _derive_runtime_predicate,
    _validated_execution_envelope,
    prepare_candidate_promotion,
)
from nerve.representation_optimizer.promotion.publication import (
    publish_promoted_package,
)
from nerve.representation_optimizer.staging.contracts import (
    staged_artifact_digest,
)
from nerve.representation_optimizer.staging.orchestrator import (
    stage_candidate,
)
from nerve.representation_optimizer.validation.orchestrator import (
    prepare_candidate_for_benchmark,
    validate_benchmarked_candidate,
)
from nerve.representation_optimizer.validation.planning import (
    build_validation_plan,
)
from nerve.representation_optimizer.validation.proofs import (
    ProofVerifierRegistry,
)
from tests.test_candidate_benchmarking import (
    AdapterBehavior,
    FixtureExecutionAdapter,
)
from tests.test_candidate_staging import (
    CompletePhysicalOptimizer,
    CompleteRelowerer,
    CompleteSemanticConstructor,
    _session_with_candidate,
)
from tests.test_candidate_validation import (
    FixtureProofVerifier,
    FixtureValidationAdapter,
    ValidationBehavior,
)
from tests.test_package_integrity import minimal_package
from tests.test_representation_providers import (
    FixtureProvider,
    _descriptor_id,
    _descriptors,
)
from tests.test_representation_optimizer_contracts import (
    hardware_profile_contract,
)


@dataclass(frozen=True)
class QualifiedCandidate:
    package_dir: Path
    candidate_workspace: Path
    benchmark_workspace: Path
    validation_workspace: Path
    analysis_run_directories: tuple[Path, ...]
    candidate_plan: object
    construction: object
    benchmark: object
    validation: object
    profile: dict[str, object]


def test_phase_selective_execution_envelope_uses_full_runtime_range() -> None:
    assert _validated_execution_envelope(
        {
            "phases": ["decode", "prefill"],
            "alternative_phases": ["decode"],
            "source_retained_phases": ["prefill"],
            "activation_batch": {"minimum": 1, "maximum": 131_072},
            "context_activations": {"minimum": 0, "maximum": 131_072},
            "state_activations": {"minimum": 0, "maximum": 131_072},
        },
        [
            {
                "execution_phase": "decode",
                "activation_batch_width": 1,
                "context_size": 4_096,
                "state_size": 4_096,
            }
        ],
    ) == (
        ["decode", "prefill"],
        1,
        131_072,
        0,
        131_072,
        0,
        131_072,
    )


def test_phase_selective_execution_envelope_requires_every_changed_phase() -> None:
    with pytest.raises(
        ModelCompileError,
        match="every alternative execution phase",
    ):
        _validated_execution_envelope(
            {
                "phases": ["decode", "prefill"],
                "alternative_phases": ["decode"],
                "source_retained_phases": ["prefill"],
                "activation_batch": {"minimum": 1, "maximum": 131_072},
                "context_activations": {"minimum": 0, "maximum": 131_072},
                "state_activations": {"minimum": 0, "maximum": 131_072},
            },
            [],
        )


def _qualified_candidate(
    tmp_path: Path,
    *,
    benchmark_behavior: AdapterBehavior | None = None,
    validation_behavior: ValidationBehavior | None = None,
) -> QualifiedCandidate:
    package_dir = tmp_path / "package"
    manifest = minimal_package(package_dir)
    manifest_path = package_dir / "vulkan_resident_package.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    stage = read_json(package_dir / "optimization" / "stage.json")
    session = OptimizationSession.from_json(stage["session"])
    profile = hardware_profile_contract()
    candidate_plan, analysis_run_directory = _candidate_plan_and_analysis(
        package_dir,
        tmp_path,
        profile,
    )
    session = _session_with_candidate(session, candidate_plan)
    candidate_workspace = tmp_path / "candidate-workspace"
    construction = stage_candidate(
        package_dir=package_dir,
        source_artifacts=PackageSourceArtifactResolver(package_dir),
        workspace_root=candidate_workspace,
        plan=candidate_plan,
        session=session,
        semantic_constructor=CompleteSemanticConstructor([]),
        ordinary_relowerer=CompleteRelowerer([]),
        physical_optimizer=CompletePhysicalOptimizer([]),
    )
    exact_path = (
        package_dir / "lowered" / "execution_graph.circuits.json"
    )
    benchmark_plan = build_benchmark_plan(
        candidate_plan=candidate_plan,
        construction_record=construction.record,
        hardware_profiles=(profile,),
        reference_implementation_id="exact-reference",
        reference_contract_digest=candidate_plan.candidate.to_json()[
            "source_contract_digests"
        ][0],
        reference_artifact_refs=(
            {
                "path": exact_path.relative_to(package_dir).as_posix(),
                "digest": staged_artifact_digest(
                    exact_path.read_bytes()
                ),
            },
        ),
        matched_conditions={
            "devices": [
                {
                    "device_id": profile["hardware_identity"][
                        "stable_device_id"
                    ],
                    "hardware_profile_digest": contract_digest(profile),
                    "capability_class": profile["capability_class"],
                    "api": profile["provenance"]["api"],
                }
            ],
            "placement": {"fixture_scope": "vulkan:fixture"},
            "controls": {"scheduler": "normal"},
            "environment": {"power_profile": "matched"},
            "idle_device_state_digest": device_state_digest(
                {"fixture_state": "idle"}
            ),
            "exclusive_residency": True,
        },
    )
    validation_plan = build_validation_plan(
        candidate_plan=candidate_plan,
        construction_record=construction.record,
        benchmark_plan=benchmark_plan,
    )
    validation_workspace = tmp_path / "validation-workspace"
    validation_adapter = FixtureValidationAdapter(
        validation_behavior
    )
    prebenchmark = prepare_candidate_for_benchmark(
        package_dir=package_dir,
        candidate_workspace_root=candidate_workspace,
        validation_workspace_root=validation_workspace,
        candidate_plan=candidate_plan,
        construction_record=construction.record,
        validation_plan=validation_plan,
        session=construction.session,
        proof_verifiers=ProofVerifierRegistry.from_verifiers(
            (FixtureProofVerifier(),)
        ),
        adapter=validation_adapter,
    )
    assert prebenchmark.status == "passed"
    benchmark_workspace = tmp_path / "benchmark-workspace"
    benchmark = benchmark_candidate(
        plan=benchmark_plan,
        construction_record=construction.record,
        session=prebenchmark.session,
        adapter=FixtureExecutionAdapter(benchmark_behavior),
        workspace_root=benchmark_workspace,
    )
    validation = validate_benchmarked_candidate(
        plan=validation_plan,
        prebenchmark_record=prebenchmark.record,
        benchmark_record=benchmark.record,
        session=benchmark.session,
        adapter=validation_adapter,
        workspace_root=validation_workspace,
    )
    return QualifiedCandidate(
        package_dir=package_dir,
        candidate_workspace=candidate_workspace,
        benchmark_workspace=benchmark_workspace,
        validation_workspace=validation_workspace,
        analysis_run_directories=(analysis_run_directory,),
        candidate_plan=candidate_plan,
        construction=construction,
        benchmark=benchmark,
        validation=validation,
        profile=profile,
    )


def _prepare(
    qualified: QualifiedCandidate,
    *,
    analysis_run_directories: tuple[Path, ...] | None = None,
) -> PreparedPromotion:
    return prepare_candidate_promotion(
        package_dir=qualified.package_dir,
        candidate_workspace_root=qualified.candidate_workspace,
        benchmark_workspace_root=qualified.benchmark_workspace,
        validation_workspace_root=qualified.validation_workspace,
        analysis_run_directories=(
            qualified.analysis_run_directories
            if analysis_run_directories is None
            else analysis_run_directories
        ),
        candidate_plan=qualified.candidate_plan,
        construction_record=qualified.construction.record,
        benchmark_record=qualified.benchmark.record,
        validation_record=qualified.validation.record,
        hardware_profiles=(qualified.profile,),
        session=qualified.validation.session,
        reason=(
            "candidate won every matched regime and passed the complete "
            "behavioral validation funnel"
        ),
    )


def _candidate_plan_and_analysis(
    package_dir: Path,
    tmp_path: Path,
    profile: dict[str, object],
):
    catalog = read_json(package_dir / "optimization" / "scopes.json")
    scope = catalog["scopes"][0]
    source = next(
        source
        for source in catalog["source_contracts"]
        if source["scope_id"] == scope["scope_id"]
    )
    evidence, details = build_evidence(
        scope_id=scope["scope_id"],
        source_contract_digest=source["contract_digest"],
        analyzer_id="fixture.promotion",
        analyzer_version="1",
        claims=(
            {
                "kind": "fixture_structure",
                "status": "supported",
                "exact": True,
                "facts": {"fixture": True},
            },
        ),
        details={"fixture": "self-contained promotion provenance"},
    )
    run = build_analysis_run(
        package_id=catalog["package_id"],
        scope_id=scope["scope_id"],
        source_contract_digest=source["contract_digest"],
        budget={"fixture": True},
        evidence=(evidence,),
        details=(details,),
    )
    analysis_run_directory = (
        tmp_path / "analysis-workspace" / run.run_id
    )
    write_analysis_run(run, analysis_run_directory)
    problem = ProviderProblem.from_documents(
        package_id=catalog["package_id"],
        scopes=(scope,),
        source_contracts=(source,),
        evidence=(evidence,),
        hardware_profile=profile,
    )
    provider = FixtureProvider(
        "fixture.staging",
        _descriptor_id(),
    )
    report = ProviderRegistry.from_providers(
        descriptors=_descriptors(),
        providers=(provider,),
    ).run(problem)
    assert report.evaluations[0].status == "completed"
    return report.candidates[0], analysis_run_directory


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_promotion_publishes_complete_self_contained_package_atomically(
    tmp_path: Path,
) -> None:
    qualified = _qualified_candidate(tmp_path)
    prepared = _prepare(qualified)
    source_before = _tree_bytes(qualified.package_dir)
    destination = tmp_path / "optimized-package"

    published = publish_promoted_package(
        source_package_dir=qualified.package_dir,
        destination_package_dir=destination,
        promotions=(prepared,),
        session=prepared.session,
    )

    assert published == destination
    assert _tree_bytes(qualified.package_dir) == source_before
    assert (
        (
            qualified.package_dir
            / "lowered"
            / "execution_graph.circuits.json"
        ).stat().st_ino
        != (
            destination
            / "lowered"
            / "execution_graph.circuits.json"
        ).stat().st_ino
    )
    manifest = read_json(destination / "vulkan_resident_package.json")
    validate_compiled_package(destination, manifest)
    stage = read_json(destination / "optimization" / "stage.json")
    registry = ImplementationRegistry.from_json(
        read_json(destination / "optimization" / "implementations.json")
    )
    assert stage["status"] == "optimized"
    assert stage["exact_baseline"] == read_json(
        qualified.package_dir / "optimization" / "stage.json"
    )["exact_baseline"]
    assert stage["implementation_registry"]["implementation_count"] == 1
    assert len(registry.implementations) == 1
    entry = registry.implementations[0]
    assert entry["implementation_id"] == prepared.implementation_id
    root = destination / entry["artifact_bundle"]["root_ref"]
    assert (root / "candidate" / "integrity.json").is_file()
    assert (root / "promotion.json").is_file()
    assert (root / "construction_record.json").is_file()
    assert (
        root
        / "evidence"
        / "analysis"
        / prepared.analysis_runs[0].run.run_id
        / "analysis.json"
    ).is_file()
    assert (
        root
        / "evidence"
        / "hardware"
        / f"{qualified.profile['profile_id']}.json"
    ).is_file()
    assert (
        root
        / "evidence"
        / "prebenchmark"
        / prepared.prebenchmark_record.to_json()["prebenchmark_id"]
        / "integrity.json"
    ).is_file()
    assert not (tmp_path / ".nerve-package-staging").exists()
    lifecycle = next(
        candidate
        for candidate in stage["session"]["candidates"]
        if candidate["candidate_id"] == prepared.decision.candidate_id
    )
    assert lifecycle["state"] == CandidateState.PUBLISHED.value
    for event in lifecycle["history"]:
        for evidence_ref in event["evidence_refs"]:
            assert (destination / evidence_ref).is_file()
    integrity_files = manifest["artifact_integrity"]["files"]
    assert (
        f"{entry['artifact_bundle']['root_ref']}/promotion.json"
        in integrity_files
    )


def test_promoted_package_is_relocatable_and_has_no_workspace_dependency(
    tmp_path: Path,
) -> None:
    qualified = _qualified_candidate(tmp_path)
    prepared = _prepare(qualified)
    optimized = publish_promoted_package(
        source_package_dir=qualified.package_dir,
        destination_package_dir=tmp_path / "optimized",
        promotions=(prepared,),
        session=prepared.session,
    )

    shutil.rmtree(qualified.package_dir)
    shutil.rmtree(qualified.candidate_workspace)
    shutil.rmtree(qualified.benchmark_workspace)
    shutil.rmtree(qualified.validation_workspace)
    shutil.rmtree(
        qualified.analysis_run_directories[0].parent
    )
    relocated = tmp_path / "relocated" / "model"
    shutil.copytree(optimized, relocated)
    shutil.rmtree(optimized)

    manifest = read_json(relocated / "vulkan_resident_package.json")
    validate_compiled_package(relocated, manifest)


@pytest.mark.parametrize(
    "behavior",
    (
        AdapterBehavior(candidate_duration_ns=1_100_000),
        AdapterBehavior(candidate_duration_ns=1_000_000),
    ),
    ids=("slower", "equal"),
)
def test_candidate_that_is_not_faster_cannot_be_prepared_for_publication(
    tmp_path: Path,
    behavior: AdapterBehavior,
) -> None:
    qualified = _qualified_candidate(
        tmp_path,
        benchmark_behavior=behavior,
    )

    assert (
        qualified.benchmark.record.to_json()["decision"]
        != "materially_faster"
    )
    assert qualified.validation.status == CandidateState.REJECTED.value
    with pytest.raises(
        ModelCompileError,
        match="behaviorally validated before promotion",
    ):
        _prepare(qualified)


def test_inaccurate_candidate_cannot_be_prepared_for_publication(
    tmp_path: Path,
) -> None:
    qualified = _qualified_candidate(
        tmp_path,
        validation_behavior=ValidationBehavior(
            invalid_stage="full_local",
        ),
    )

    assert qualified.validation.status == CandidateState.REJECTED.value
    with pytest.raises(
        ModelCompileError,
        match="behaviorally validated before promotion",
    ):
        _prepare(qualified)


def test_publication_revalidates_evidence_and_leaves_no_partial_package(
    tmp_path: Path,
) -> None:
    qualified = _qualified_candidate(tmp_path)
    prepared = _prepare(qualified)
    record_path = prepared.benchmark_evidence_path / "record.json"
    record_path.write_bytes(record_path.read_bytes() + b"\n")
    destination = tmp_path / "must-not-exist"

    with pytest.raises(ModelCompileError, match="integrity"):
        publish_promoted_package(
            source_package_dir=qualified.package_dir,
            destination_package_dir=destination,
            promotions=(prepared,),
            session=prepared.session,
        )

    assert not destination.exists()
    assert not (tmp_path / ".nerve-package-staging").exists()
    manifest = read_json(
        qualified.package_dir / "vulkan_resident_package.json"
    )
    validate_compiled_package(qualified.package_dir, manifest)


def test_publication_cancellation_during_clone_removes_partial_package(
    tmp_path: Path,
) -> None:
    qualified = _qualified_candidate(tmp_path)
    prepared = _prepare(qualified)
    source_before = _tree_bytes(qualified.package_dir)
    destination = tmp_path / "must-not-exist"
    staging_root = tmp_path / ".nerve-package-staging"

    def cancel_after_first_cloned_file() -> bool:
        return staging_root.exists() and any(
            path.is_file() for path in staging_root.rglob("*")
        )

    with pytest.raises(ModelCompileCancelled, match="cancelled"):
        publish_promoted_package(
            source_package_dir=qualified.package_dir,
            destination_package_dir=destination,
            promotions=(prepared,),
            session=prepared.session,
            cancel_requested=cancel_after_first_cloned_file,
        )

    assert not destination.exists()
    assert not staging_root.exists()
    assert _tree_bytes(qualified.package_dir) == source_before


def test_missing_analysis_provenance_cannot_be_promoted(
    tmp_path: Path,
) -> None:
    qualified = _qualified_candidate(tmp_path)

    with pytest.raises(
        ModelCompileError,
        match="every analysis record",
    ):
        _prepare(qualified, analysis_run_directories=())


def test_promotion_predicate_is_derived_from_measured_regimes_and_target(
    tmp_path: Path,
) -> None:
    qualified = _qualified_candidate(tmp_path)
    prepared = _prepare(qualified)
    predicate = prepared.runtime_predicate.to_json()

    assert predicate["hardware"] == {
        "capability_class_counts": [
            {
                "capability_class": qualified.profile[
                    "capability_class"
                ],
                "count": 1,
            }
        ],
        "device_kinds": ["gpu"],
        "apis": ["vulkan"],
        "required_processes": [],
        "required_features": [],
    }
    assert predicate["execution"] == {
        "phases": ["decode", "prefill"],
        "alternative_phases": ["decode", "prefill"],
        "source_retained_phases": [],
        "activation_batch": {"minimum": 1, "maximum": 8},
        "context_activations": {
            "minimum": 4096,
            "maximum": 32768,
        },
        "state_activations": {
            "minimum": 4096,
            "maximum": 8192,
        },
        "speculative_draft_token_counts": [0],
    }
    assert predicate["placement"] == {
        "mode": "either",
        "minimum_device_count": 1,
        "maximum_device_count": 1,
        "required_interconnects": [],
    }


def test_promotion_rejects_unmountable_hardware_process_names(
    tmp_path: Path,
) -> None:
    qualified = _qualified_candidate(tmp_path)
    candidate = qualified.candidate_plan.candidate.to_json()
    candidate["target_predicate"]["required_processes"] = [
        "abstract_process_not_exposed_by_the_runtime"
    ]

    with pytest.raises(
        ModelCompileError,
        match="not mountable.*missing processes",
    ):
        _derive_runtime_predicate(
            benchmark_plan=qualified.benchmark.plan,
            validation_plan=qualified.validation.plan,
            hardware_profiles=(qualified.profile,),
            candidate=candidate,
        )


def test_registry_rejects_mount_contract_outside_implementation_bundle(
    tmp_path: Path,
) -> None:
    prepared = _prepare(_qualified_candidate(tmp_path))
    entry = deepcopy(prepared.registry_entry)
    entry["artifact_bundle"]["mount_plan_ref"] = (
        "optimization/unrelated/mount_plan.json"
    )

    with pytest.raises(
        PromotionContractError,
        match="must stay inside",
    ):
        validate_implementation_registry_entry(entry)


def test_promotion_rejects_hardware_profile_not_used_by_benchmark(
    tmp_path: Path,
) -> None:
    qualified = _qualified_candidate(tmp_path)
    mismatched_profile = deepcopy(qualified.profile)
    mismatched_profile["runtime_bindings"] = {
        "queue": "different measured binding"
    }

    with pytest.raises(
        ModelCompileError,
        match="do not match benchmark evidence",
    ):
        prepare_candidate_promotion(
            package_dir=qualified.package_dir,
            candidate_workspace_root=qualified.candidate_workspace,
            benchmark_workspace_root=qualified.benchmark_workspace,
            validation_workspace_root=qualified.validation_workspace,
            analysis_run_directories=(
                qualified.analysis_run_directories
            ),
            candidate_plan=qualified.candidate_plan,
            construction_record=qualified.construction.record,
            benchmark_record=qualified.benchmark.record,
            validation_record=qualified.validation.record,
            hardware_profiles=(mismatched_profile,),
            session=qualified.validation.session,
            reason="must not be accepted",
        )


def test_publication_revalidates_analysis_provenance(
    tmp_path: Path,
) -> None:
    qualified = _qualified_candidate(tmp_path)
    prepared = _prepare(qualified)
    details = next(
        (
            qualified.analysis_run_directories[0] / "details"
        ).iterdir()
    )
    details.write_text('{"mutated":true}\n')
    destination = tmp_path / "must-not-exist"

    with pytest.raises(ModelCompileError, match="details digest"):
        publish_promoted_package(
            source_package_dir=qualified.package_dir,
            destination_package_dir=destination,
            promotions=(prepared,),
            session=prepared.session,
        )

    assert not destination.exists()
    assert not (tmp_path / ".nerve-package-staging").exists()


def test_failure_before_atomic_rename_preserves_source_and_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualified = _qualified_candidate(tmp_path)
    prepared = _prepare(qualified)
    source_before = _tree_bytes(qualified.package_dir)
    destination = tmp_path / "must-not-exist"
    real_validate = publication.validate_compiled_package

    def fail_staged_validation(package_dir, manifest):
        if package_dir.resolve() != qualified.package_dir.resolve():
            raise ModelCompileError("injected staged-package failure")
        return real_validate(package_dir, manifest)

    monkeypatch.setattr(
        publication,
        "validate_compiled_package",
        fail_staged_validation,
    )

    with pytest.raises(
        ModelCompileError,
        match="injected staged-package failure",
    ):
        publish_promoted_package(
            source_package_dir=qualified.package_dir,
            destination_package_dir=destination,
            promotions=(prepared,),
            session=prepared.session,
        )

    assert not destination.exists()
    assert not (tmp_path / ".nerve-package-staging").exists()
    assert _tree_bytes(qualified.package_dir) == source_before


def test_failure_after_atomic_rename_removes_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualified = _qualified_candidate(tmp_path)
    prepared = _prepare(qualified)
    source_before = _tree_bytes(qualified.package_dir)
    destination = tmp_path / "must-not-exist"
    real_validate = publication.validate_compiled_package
    non_source_validations = 0

    def fail_post_rename_validation(package_dir, manifest):
        nonlocal non_source_validations
        if package_dir.resolve() == qualified.package_dir.resolve():
            return real_validate(package_dir, manifest)
        non_source_validations += 1
        if non_source_validations == 2:
            raise ModelCompileError("injected post-rename failure")
        return real_validate(package_dir, manifest)

    monkeypatch.setattr(
        publication,
        "validate_compiled_package",
        fail_post_rename_validation,
    )

    with pytest.raises(
        ModelCompileError,
        match="injected post-rename failure",
    ):
        publish_promoted_package(
            source_package_dir=qualified.package_dir,
            destination_package_dir=destination,
            promotions=(prepared,),
            session=prepared.session,
        )

    assert non_source_validations == 2
    assert not destination.exists()
    assert not (tmp_path / ".nerve-package-staging").exists()
    assert _tree_bytes(qualified.package_dir) == source_before


def test_registry_allows_distinct_verified_implementations_for_same_scope(
    tmp_path: Path,
) -> None:
    qualified = _qualified_candidate(tmp_path)
    prepared = _prepare(qualified)
    source_stage = read_json(
        qualified.package_dir / "optimization" / "stage.json"
    )
    registry = ImplementationRegistry.from_json(
        read_json(
            qualified.package_dir
            / source_stage["implementation_registry"]["artifact_ref"]
        )
    )
    first = deepcopy(prepared.registry_entry)
    second = deepcopy(first)
    second_candidate = stable_contract_id(
        "candidate",
        "same semantic scope, decode-specialized representation",
    )
    second_predicate = create_runtime_implementation_predicate(
        capability_classes=(
            count["capability_class"]
            for count in first["runtime_predicate"]["hardware"][
                "capability_class_counts"
            ]
            for _ in range(count["count"])
        ),
        device_kinds=first["runtime_predicate"]["hardware"][
            "device_kinds"
        ],
        apis=first["runtime_predicate"]["hardware"]["apis"],
        required_processes=first["runtime_predicate"]["hardware"][
            "required_processes"
        ],
        required_features=first["runtime_predicate"]["hardware"][
            "required_features"
        ],
        execution_phases=("decode",),
        alternative_execution_phases=("decode",),
        source_retained_execution_phases=(),
        activation_batch_minimum=1,
        activation_batch_maximum=1,
        context_activations_minimum=1,
        context_activations_maximum=16_384,
        state_activations_minimum=0,
        state_activations_maximum=16_384,
        speculative_draft_token_counts=(3,),
        placement_mode="local",
        minimum_device_count=1,
        maximum_device_count=1,
        required_interconnects=(),
    )
    second["candidate_id"] = second_candidate
    second["runtime_predicate"] = second_predicate.to_json()
    second["implementation_id"] = implementation_id(
        second_candidate,
        second_predicate,
    )
    second_root = (
        "optimization/implementations/"
        f"{second['implementation_id']}"
    )
    second["artifact_bundle"]["root_ref"] = second_root
    second["artifact_bundle"]["candidate_integrity_ref"] = (
        f"{second_root}/candidate/integrity.json"
    )
    second["artifact_bundle"]["mount_plan_ref"] = (
        f"{second_root}/candidate/contracts/mount_plan.json"
    )
    for name, value in tuple(second["evidence"].items()):
        if name == "analysis_run_refs":
            for reference in value:
                reference["artifact_ref"] = (
                    f"{second_root}/evidence/analysis/"
                    f"{reference['run_id']}"
                )
            continue
        if name == "hardware_profile_refs":
            for reference in value:
                reference["artifact_ref"] = (
                    f"{second_root}/evidence/hardware/"
                    f"{reference['profile_id']}.json"
                )
            continue
        suffix = value.split("/", 3)[-1]
        second["evidence"][name] = f"{second_root}/{suffix}"

    updated = append_implementation_registry_entries(
        registry,
        (first, second),
    )

    assert len(updated.implementations) == 2
    assert {
        tuple(entry["scope_ids"]) for entry in updated.implementations
    } == {tuple(first["scope_ids"])}
    assert {
        tuple(entry["runtime_predicate"]["execution"]["phases"])
        for entry in updated.implementations
    } == {("decode",), ("decode", "prefill")}


def test_publication_refuses_existing_destination(tmp_path: Path) -> None:
    qualified = _qualified_candidate(tmp_path)
    prepared = _prepare(qualified)
    destination = tmp_path / "existing"
    destination.mkdir()

    with pytest.raises(ModelCompileError, match="already exists"):
        publish_promoted_package(
            source_package_dir=qualified.package_dir,
            destination_package_dir=destination,
            promotions=(prepared,),
            session=prepared.session,
        )

    assert list(destination.iterdir()) == []
