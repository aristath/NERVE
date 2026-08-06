from __future__ import annotations

import struct
import time
from dataclasses import replace
from pathlib import Path

import pytest

import nerve.representation_optimizer.staging.orchestrator as staging_orchestrator
from nerve.compilation import ModelCompileError
from nerve.representation_optimizer.lifecycle import (
    CandidateState,
    OptimizationSession,
)
from nerve.representation_optimizer.providers import ProviderRegistry
from nerve.representation_optimizer.providers.source_artifacts import (
    PackageSourceArtifactResolver,
)
from nerve.representation_optimizer.staging.contracts import (
    CandidateBuildPlan,
    staged_artifact_digest,
)
from nerve.representation_optimizer.staging.artifact_validation import (
    ArtifactValidatorRegistry,
)
from nerve.representation_optimizer.staging.integrity import (
    validate_staged_candidate,
)
from nerve.representation_optimizer.staging.loading import load_staged_candidate
from nerve.representation_optimizer.staging.orchestrator import stage_candidate
from nerve.representation_optimizer.stage import initialize_optimizer_stage
from tests.test_representation_optimizer_stage import write_exact_baseline
from tests.test_representation_providers import (
    FixtureProvider,
    _descriptor_id,
    _descriptors,
    _problem,
)


class CompleteSemanticConstructor:
    def __init__(
        self,
        calls: list[str],
        *,
        source_path: str | None = None,
        stream_source: bool = False,
        stream_parameter: bool = False,
        source_regions: tuple[tuple[int, int], ...] | None = None,
        paired_stream: bool = False,
    ) -> None:
        self.calls = calls
        self.source_path = source_path
        self.stream_source = stream_source
        self.stream_parameter = stream_parameter
        self.source_regions = source_regions
        self.paired_stream = paired_stream

    def construct_semantic_artifacts(self, context) -> None:
        self.calls.append("semantic_construction")
        if self.source_path is not None:
            if self.source_regions is not None:
                assert context.read_source_artifact_regions(
                    self.source_path,
                    self.source_regions,
                    chunk_bytes=5,
                ) == (b"immutable", b"source", b"parameter")
            else:
                source = (
                    b"".join(
                        context.iter_source_artifact(
                            self.source_path,
                            chunk_bytes=5,
                        )
                    )
                    if self.stream_source
                    else context.read_source_artifact(self.source_path)
                )
                assert source == b"immutable source parameter"
        context.account_transient_bytes(4096)
        binary_paths = (
            "codebooks/table.bin",
            "corrections/residual.bin",
            "fields/samples.bin",
            "geometry/basis.bin",
            "indexes/search.bin",
            "programs/evaluator.bin",
        )
        if self.paired_stream:
            first, second = (f"fixture:{path}".encode() for path in binary_paths[:2])
            context.write_artifact_streams(
                binary_paths[:2],
                (
                    (first[:8], second[:8]),
                    (first[8:], second[8:]),
                ),
            )
            remaining_paths = binary_paths[2:]
        else:
            remaining_paths = binary_paths
        for path in remaining_paths:
            context.write_artifact(path, f"fixture:{path}".encode())
        if self.stream_parameter:
            context.write_artifact_stream(
                "parameters/sparse_weights.bin",
                (b"sparse-", b"para", b"meter"),
            )
        else:
            context.write_artifact(
                "parameters/sparse_weights.bin",
                b"sparse-parameter",
            )
        context.write_json_artifact(
            "state/compact_layout.json",
            {
                "schema": ("nerve.optimizer.vulkan_component_overlay.v2"),
                "source_component_id": "component",
                "component": {"fixture": True},
                "execution": {"fixture": True},
                "resident_derivations": [],
            },
        )
        context.account_transient_bytes(0)


class CompleteRelowerer:
    def __init__(self, calls: list[str], *, fail: bool = False) -> None:
        self.calls = calls
        self.fail = fail

    def run_ordinary_lowering(self, context) -> None:
        self.calls.append("ordinary_lowering")
        if self.fail:
            raise RuntimeError("ordinary lowering fixture failure")
        request = context.representation_graph
        context.write_artifact(
            "graphs/events.bin",
            b"graph:" + request["graph_id"].encode(),
        )
        context.write_artifact(
            "topology/events.bin",
            b"lowered:" + request["graph_id"].encode(),
        )


class CancellingStreamConstructor:
    def __init__(self, cancel_state: dict[str, bool]) -> None:
        self.cancel_state = cancel_state

    def construct_semantic_artifacts(self, context) -> None:
        def chunks():
            self.cancel_state["requested"] = True
            yield b"first chunk"
            yield b"must never be written"

        context.write_artifact_stream("codebooks/table.bin", chunks())


class CompletePhysicalOptimizer:
    def __init__(
        self,
        calls: list[str],
        *,
        omit_output: bool = False,
        malformed_spirv: bool = False,
    ) -> None:
        self.calls = calls
        self.omit_output = omit_output
        self.malformed_spirv = malformed_spirv

    def optimize_physical_artifacts(self, context) -> None:
        self.calls.append("physical_optimization")
        if not self.omit_output:
            context.write_artifact(
                "kernels/native_island.spv",
                (
                    b"not-spirv"
                    if self.malformed_spirv
                    else struct.pack("<5I", 0x07230203, 0x00010000, 0, 1, 0)
                ),
            )


def _package(tmp_path: Path) -> tuple[Path, OptimizationSession]:
    package_dir = tmp_path / "package"
    baseline, baseline_path = write_exact_baseline(package_dir)
    stage = initialize_optimizer_stage(
        package_id="fixture_package",
        package_dir=package_dir,
        lowered_index=baseline,
        lowered_index_path=baseline_path,
    )
    (package_dir / "vulkan_resident_package.json").write_text(
        "{\n"
        '  "package_id": "fixture_package",\n'
        '  "representation_optimization_path": "optimization/stage.json",\n'
        '  "artifact_integrity": {}\n'
        "}\n"
    )
    return package_dir, OptimizationSession.from_json(stage.document["session"])


def _plan():
    provider = FixtureProvider("fixture.staging", _descriptor_id())
    report = ProviderRegistry.from_providers(
        descriptors=_descriptors(),
        providers=[provider],
    ).run(_problem())
    assert report.evaluations[0].status == "completed"
    return report.candidates[0]


def _session_with_candidate(
    session: OptimizationSession,
    plan,
) -> OptimizationSession:
    return session.register_candidate(
        plan.candidate_id,
        tuple(plan.candidate.to_json()["source_contract_digests"]),
    )


def test_candidate_is_constructed_relowered_optimized_and_atomically_staged(
    tmp_path: Path,
) -> None:
    package_dir, session = _package(tmp_path)
    plan = _plan()
    session = _session_with_candidate(session, plan)
    workspace = tmp_path / "candidate-workspace"
    calls: list[str] = []
    baseline_before = (
        package_dir / "lowered" / "execution_graph.circuits.json"
    ).read_bytes()
    stage_before = (package_dir / "optimization" / "stage.json").read_bytes()

    outcome = stage_candidate(
        package_dir=package_dir,
        source_artifacts=PackageSourceArtifactResolver(package_dir),
        workspace_root=workspace,
        plan=plan,
        session=session,
        semantic_constructor=CompleteSemanticConstructor(calls),
        ordinary_relowerer=CompleteRelowerer(calls),
        physical_optimizer=CompletePhysicalOptimizer(calls),
    )

    assert outcome.status == "completed"
    assert calls == [
        "semantic_construction",
        "ordinary_lowering",
        "physical_optimization",
    ]
    assert outcome.staged_candidate_path == (workspace / "ready" / plan.candidate_id)
    assert outcome.staged_candidate_path.is_dir()
    assert not list((workspace / ".staging").iterdir())
    validate_staged_candidate(
        outcome.staged_candidate_path,
        expected_candidate_id=plan.candidate_id,
        expected_build_plan=plan.construction_requirements,
    )
    lifecycle = next(
        candidate
        for candidate in outcome.session.candidates
        if candidate.candidate_id == plan.candidate_id
    )
    assert lifecycle.state == CandidateState.STAGED
    record = outcome.record.to_json()
    assert record["status"] == "completed"
    assert [phase["name"] for phase in record["phases"]] == [
        "semantic_construction",
        "ordinary_lowering",
        "physical_optimization",
    ]
    assert all(phase["status"] == "completed" for phase in record["phases"])
    assert record["resource_measurements"]["construction_time_ns"] > 0
    assert record["resource_measurements"]["peak_temporary_bytes"] >= 4096
    assert record["resource_measurements"]["generated_artifact_bytes"] == sum(
        artifact["byte_count"] for artifact in record["artifacts"]
    )
    assert record["resource_measurements"]["final_permanent_bytes"] == 336
    assert record["integrity"]["file_count"] >= 16
    staged_bytes = sum(
        path.stat().st_size
        for path in outcome.staged_candidate_path.rglob("*")
        if path.is_file()
    )
    assert record["resource_measurements"]["peak_staging_bytes"] >= staged_bytes
    relowering = (
        outcome.staged_candidate_path / "contracts" / "relowering_request.json"
    ).read_text()
    assert "ordinary_lowering" in relowering
    assert "physical_optimization" in relowering
    assert (
        package_dir / "lowered" / "execution_graph.circuits.json"
    ).read_bytes() == baseline_before
    assert (package_dir / "optimization" / "stage.json").read_bytes() == stage_before
    loaded = load_staged_candidate(
        workspace,
        plan.candidate_id,
        package_dir=package_dir,
    )
    assert loaded.path == outcome.staged_candidate_path
    assert loaded.record == outcome.record


def test_declared_source_input_is_digest_checked_and_sealed(
    tmp_path: Path,
) -> None:
    package_dir, session = _package(tmp_path)
    source_path = package_dir / "weights" / "source.bin"
    source_path.parent.mkdir()
    source_path.write_bytes(b"immutable source parameter")
    plan = _plan()
    build_document = plan.construction_requirements.to_json()
    build_document["source_inputs"] = [
        {
            "path": "weights/source.bin",
            "digest": staged_artifact_digest(source_path.read_bytes()),
        }
    ]
    plan = replace(
        plan,
        construction_requirements=CandidateBuildPlan.from_json(build_document),
    )
    session = _session_with_candidate(session, plan)

    outcome = stage_candidate(
        package_dir=package_dir,
        source_artifacts=PackageSourceArtifactResolver(package_dir),
        workspace_root=tmp_path / "workspace",
        plan=plan,
        session=session,
        semantic_constructor=CompleteSemanticConstructor(
            [],
            source_path="weights/source.bin",
            stream_source=True,
            stream_parameter=True,
        ),
        ordinary_relowerer=CompleteRelowerer([]),
        physical_optimizer=CompletePhysicalOptimizer([]),
    )

    assert outcome.status == "completed"
    sealed_source = outcome.record.to_json()["source_seal"]["source_inputs"][
        "weights/source.bin"
    ]
    assert sealed_source["digest"] == staged_artifact_digest(
        b"immutable source parameter"
    )
    assert sealed_source["signature"]["byte_count"] == len(
        b"immutable source parameter"
    )
    assert source_path.read_bytes() == b"immutable source parameter"


def test_multiple_artifacts_can_be_emitted_from_one_source_pass(
    tmp_path: Path,
) -> None:
    package_dir, session = _package(tmp_path)
    plan = _plan()
    session = _session_with_candidate(session, plan)

    outcome = stage_candidate(
        package_dir=package_dir,
        source_artifacts=PackageSourceArtifactResolver(package_dir),
        workspace_root=tmp_path / "workspace",
        plan=plan,
        session=session,
        semantic_constructor=CompleteSemanticConstructor([], paired_stream=True),
        ordinary_relowerer=CompleteRelowerer([]),
        physical_optimizer=CompletePhysicalOptimizer([]),
    )

    assert outcome.status == "completed"
    assert outcome.staged_candidate_path is not None
    for path in ("codebooks/table.bin", "corrections/residual.bin"):
        assert (outcome.staged_candidate_path / path).read_bytes() == (
            f"fixture:{path}".encode()
        )


def test_declared_source_regions_are_streamed_and_whole_file_digest_checked(
    tmp_path: Path,
) -> None:
    package_dir, session = _package(tmp_path)
    source_path = package_dir / "weights" / "source.bin"
    source_path.parent.mkdir()
    source_path.write_bytes(b"immutable source parameter")
    plan = _plan()
    build_document = plan.construction_requirements.to_json()
    build_document["source_inputs"] = [
        {
            "path": "weights/source.bin",
            "digest": staged_artifact_digest(source_path.read_bytes()),
        }
    ]
    plan = replace(
        plan,
        construction_requirements=CandidateBuildPlan.from_json(build_document),
    )
    session = _session_with_candidate(session, plan)

    outcome = stage_candidate(
        package_dir=package_dir,
        source_artifacts=PackageSourceArtifactResolver(package_dir),
        workspace_root=tmp_path / "workspace",
        plan=plan,
        session=session,
        semantic_constructor=CompleteSemanticConstructor(
            [],
            source_path="weights/source.bin",
            source_regions=((0, 9), (10, 6), (17, 9)),
        ),
        ordinary_relowerer=CompleteRelowerer([]),
        physical_optimizer=CompletePhysicalOptimizer([]),
    )

    assert outcome.status == "completed"


def test_source_signature_drift_is_rejected_without_rehashing(
    tmp_path: Path,
) -> None:
    package_dir, session = _package(tmp_path)
    source_path = package_dir / "weights" / "source.bin"
    source_path.parent.mkdir()
    source_path.write_bytes(b"immutable source parameter")
    plan = _plan()
    build = plan.construction_requirements.to_json()
    build["source_inputs"] = [
        {
            "path": "weights/source.bin",
            "digest": staged_artifact_digest(source_path.read_bytes()),
        }
    ]
    plan = replace(
        plan,
        construction_requirements=CandidateBuildPlan.from_json(build),
    )
    session = _session_with_candidate(session, plan)
    complete = CompleteSemanticConstructor(
        [],
        source_path="weights/source.bin",
        stream_source=True,
    )

    class MutatingConstructor:
        def construct_semantic_artifacts(self, context) -> None:
            complete.construct_semantic_artifacts(context)
            source_path.write_bytes(b"changed source parameter!!")

    outcome = stage_candidate(
        package_dir=package_dir,
        source_artifacts=PackageSourceArtifactResolver(package_dir),
        workspace_root=tmp_path / "workspace",
        plan=plan,
        session=session,
        semantic_constructor=MutatingConstructor(),
        ordinary_relowerer=CompleteRelowerer([]),
        physical_optimizer=CompletePhysicalOptimizer([]),
    )

    assert outcome.status == "failed"
    assert "source package changed" in " ".join(outcome.record.to_json()["diagnostics"])


def test_failed_phase_is_isolated_and_leaves_no_candidate_artifacts(
    tmp_path: Path,
) -> None:
    package_dir, session = _package(tmp_path)
    plan = _plan()
    session = _session_with_candidate(session, plan)
    workspace = tmp_path / "workspace"
    calls: list[str] = []
    stage_before = (package_dir / "optimization" / "stage.json").read_bytes()

    outcome = stage_candidate(
        package_dir=package_dir,
        source_artifacts=PackageSourceArtifactResolver(package_dir),
        workspace_root=workspace,
        plan=plan,
        session=session,
        semantic_constructor=CompleteSemanticConstructor(calls),
        ordinary_relowerer=CompleteRelowerer(calls, fail=True),
        physical_optimizer=CompletePhysicalOptimizer(calls),
    )

    assert outcome.status == "failed"
    assert calls == ["semantic_construction", "ordinary_lowering"]
    assert outcome.staged_candidate_path is None
    assert not (workspace / "ready" / plan.candidate_id).exists()
    assert not list((workspace / ".staging").iterdir())
    record = outcome.record.to_json()
    assert record["artifacts"] == []
    assert record["integrity"] is None
    assert record["phases"][-1]["status"] == "failed"
    assert "ordinary lowering fixture failure" in record["diagnostics"][0]
    assert (package_dir / "optimization" / "stage.json").read_bytes() == stage_before
    lifecycle = next(
        candidate
        for candidate in outcome.session.candidates
        if candidate.candidate_id == plan.candidate_id
    )
    assert lifecycle.state == CandidateState.FAILED


def test_cancellation_removes_incomplete_workspace_and_records_no_artifacts(
    tmp_path: Path,
) -> None:
    package_dir, session = _package(tmp_path)
    plan = _plan()
    session = _session_with_candidate(session, plan)
    workspace = tmp_path / "workspace"

    outcome = stage_candidate(
        package_dir=package_dir,
        source_artifacts=PackageSourceArtifactResolver(package_dir),
        workspace_root=workspace,
        plan=plan,
        session=session,
        semantic_constructor=CompleteSemanticConstructor([]),
        ordinary_relowerer=CompleteRelowerer([]),
        physical_optimizer=CompletePhysicalOptimizer([]),
        cancel_requested=lambda: True,
    )

    assert outcome.status == "cancelled"
    assert outcome.staged_candidate_path is None
    assert not (workspace / "ready" / plan.candidate_id).exists()
    assert not list((workspace / ".staging").iterdir())
    record = outcome.record.to_json()
    assert record["artifacts"] == []
    assert record["phases"] == []
    assert record["diagnostics"] == ["candidate construction cancelled"]
    lifecycle = next(
        candidate
        for candidate in outcome.session.candidates
        if candidate.candidate_id == plan.candidate_id
    )
    assert lifecycle.state == CandidateState.CANCELLED


def test_pre_phase_resource_failure_is_recorded_and_cleaned(
    tmp_path: Path,
) -> None:
    package_dir, session = _package(tmp_path)
    plan = _plan()
    build = plan.construction_requirements.to_json()
    build["resource_limits"]["maximum_construction_time_ns"] = 1
    plan = replace(
        plan,
        construction_requirements=CandidateBuildPlan.from_json(build),
    )
    session = _session_with_candidate(session, plan)
    workspace = tmp_path / "workspace"

    outcome = stage_candidate(
        package_dir=package_dir,
        source_artifacts=PackageSourceArtifactResolver(package_dir),
        workspace_root=workspace,
        plan=plan,
        session=session,
        semantic_constructor=CompleteSemanticConstructor([]),
        ordinary_relowerer=CompleteRelowerer([]),
        physical_optimizer=CompletePhysicalOptimizer([]),
    )

    assert outcome.status == "failed"
    record = outcome.record.to_json()
    assert record["phases"] == []
    assert "construction time exceeded" in record["diagnostics"][0]
    assert record["artifacts"] == []
    assert record["integrity"] is None
    assert not (workspace / "ready" / plan.candidate_id).exists()
    assert not list((workspace / ".staging").iterdir())
    lifecycle = next(
        candidate
        for candidate in outcome.session.candidates
        if candidate.candidate_id == plan.candidate_id
    )
    assert lifecycle.state == CandidateState.FAILED


def test_cancellation_during_streamed_artifact_removes_partial_file(
    tmp_path: Path,
) -> None:
    package_dir, session = _package(tmp_path)
    plan = _plan()
    session = _session_with_candidate(session, plan)
    workspace = tmp_path / "workspace"
    cancel_state = {"requested": False}

    outcome = stage_candidate(
        package_dir=package_dir,
        source_artifacts=PackageSourceArtifactResolver(package_dir),
        workspace_root=workspace,
        plan=plan,
        session=session,
        semantic_constructor=CancellingStreamConstructor(cancel_state),
        ordinary_relowerer=CompleteRelowerer([]),
        physical_optimizer=CompletePhysicalOptimizer([]),
        cancel_requested=lambda: cancel_state["requested"],
    )

    assert outcome.status == "cancelled"
    assert outcome.record.to_json()["phases"][0]["status"] == "cancelled"
    assert outcome.record.to_json()["artifacts"] == []
    assert not (workspace / "ready" / plan.candidate_id).exists()
    assert not list((workspace / ".staging").iterdir())


def test_missing_declared_output_fails_before_atomic_stage_publication(
    tmp_path: Path,
) -> None:
    package_dir, session = _package(tmp_path)
    plan = _plan()
    session = _session_with_candidate(session, plan)
    workspace = tmp_path / "workspace"

    outcome = stage_candidate(
        package_dir=package_dir,
        source_artifacts=PackageSourceArtifactResolver(package_dir),
        workspace_root=workspace,
        plan=plan,
        session=session,
        semantic_constructor=CompleteSemanticConstructor([]),
        ordinary_relowerer=CompleteRelowerer([]),
        physical_optimizer=CompletePhysicalOptimizer([], omit_output=True),
    )

    assert outcome.status == "failed"
    assert "did not produce declared artifacts" in " ".join(
        outcome.record.to_json()["diagnostics"]
    )
    assert not (workspace / "ready" / plan.candidate_id).exists()


def test_atomic_ready_rename_failure_leaves_only_failure_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_dir, session = _package(tmp_path)
    plan = _plan()
    session = _session_with_candidate(session, plan)
    workspace = tmp_path / "workspace"
    original_replace = Path.replace

    def fail_ready_replace(path: Path, target: Path):
        if path.parent.name == ".staging" and target.parent.name == "ready":
            raise OSError("injected atomic publication failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_ready_replace)
    outcome = stage_candidate(
        package_dir=package_dir,
        source_artifacts=PackageSourceArtifactResolver(package_dir),
        workspace_root=workspace,
        plan=plan,
        session=session,
        semantic_constructor=CompleteSemanticConstructor([]),
        ordinary_relowerer=CompleteRelowerer([]),
        physical_optimizer=CompletePhysicalOptimizer([]),
    )

    assert outcome.status == "failed"
    assert "atomic publication failure" in " ".join(
        outcome.record.to_json()["diagnostics"]
    )
    assert not (workspace / "ready" / plan.candidate_id).exists()
    assert not list((workspace / ".staging").iterdir())
    records = list((workspace / "records").glob("*.json"))
    assert len(records) == 1
    assert not list((workspace / "records" / ".staging").iterdir())


def test_staged_corruption_is_detected_by_complete_integrity_manifest(
    tmp_path: Path,
) -> None:
    package_dir, session = _package(tmp_path)
    plan = _plan()
    session = _session_with_candidate(session, plan)
    outcome = stage_candidate(
        package_dir=package_dir,
        source_artifacts=PackageSourceArtifactResolver(package_dir),
        workspace_root=tmp_path / "workspace",
        plan=plan,
        session=session,
        semantic_constructor=CompleteSemanticConstructor([]),
        ordinary_relowerer=CompleteRelowerer([]),
        physical_optimizer=CompletePhysicalOptimizer([]),
    )
    assert outcome.staged_candidate_path is not None
    artifact = outcome.staged_candidate_path / "parameters" / "sparse_weights.bin"
    artifact.write_bytes(b"corrupted")

    with pytest.raises(ModelCompileError, match="integrity validation"):
        validate_staged_candidate(outcome.staged_candidate_path)


def test_loader_rejects_construction_record_tampering(
    tmp_path: Path,
) -> None:
    package_dir, session = _package(tmp_path)
    plan = _plan()
    session = _session_with_candidate(session, plan)
    workspace = tmp_path / "workspace"
    stage_candidate(
        package_dir=package_dir,
        source_artifacts=PackageSourceArtifactResolver(package_dir),
        workspace_root=workspace,
        plan=plan,
        session=session,
        semantic_constructor=CompleteSemanticConstructor([]),
        ordinary_relowerer=CompleteRelowerer([]),
        physical_optimizer=CompletePhysicalOptimizer([]),
    )
    record_path = next((workspace / "records").glob("*.json"))
    record_path.write_text('{"schema":"forged"}\n')

    with pytest.raises(ModelCompileError):
        load_staged_candidate(workspace, plan.candidate_id)


def test_complete_publication_is_recovered_without_reconstructing(
    tmp_path: Path,
) -> None:
    package_dir, session = _package(tmp_path)
    plan = _plan()
    session = _session_with_candidate(session, plan)
    workspace = tmp_path / "workspace"
    first_calls: list[str] = []
    first = stage_candidate(
        package_dir=package_dir,
        source_artifacts=PackageSourceArtifactResolver(package_dir),
        workspace_root=workspace,
        plan=plan,
        session=session,
        semantic_constructor=CompleteSemanticConstructor(first_calls),
        ordinary_relowerer=CompleteRelowerer(first_calls),
        physical_optimizer=CompletePhysicalOptimizer(first_calls),
    )
    recovery_calls: list[str] = []

    recovered = stage_candidate(
        package_dir=package_dir,
        source_artifacts=PackageSourceArtifactResolver(package_dir),
        workspace_root=workspace,
        plan=plan,
        session=session,
        semantic_constructor=CompleteSemanticConstructor(recovery_calls),
        ordinary_relowerer=CompleteRelowerer(recovery_calls),
        physical_optimizer=CompletePhysicalOptimizer(recovery_calls),
    )

    assert first_calls == [
        "semantic_construction",
        "ordinary_lowering",
        "physical_optimization",
    ]
    assert recovery_calls == []
    assert recovered.record == first.record
    assert recovered.staged_candidate_path == first.staged_candidate_path
    lifecycle = next(
        candidate
        for candidate in recovered.session.candidates
        if candidate.candidate_id == plan.candidate_id
    )
    assert lifecycle.state == CandidateState.STAGED
    assert "interrupted staging" in lifecycle.history[-1]["reason"]


def test_orphaned_atomic_publication_is_removed_before_clean_retry(
    tmp_path: Path,
) -> None:
    package_dir, session = _package(tmp_path)
    plan = _plan()
    session = _session_with_candidate(session, plan)
    workspace = tmp_path / "workspace"
    first = stage_candidate(
        package_dir=package_dir,
        source_artifacts=PackageSourceArtifactResolver(package_dir),
        workspace_root=workspace,
        plan=plan,
        session=session,
        semantic_constructor=CompleteSemanticConstructor([]),
        ordinary_relowerer=CompleteRelowerer([]),
        physical_optimizer=CompletePhysicalOptimizer([]),
    )
    first_record = next((workspace / "records").glob("*.json"))
    first_record.unlink()
    retry_calls: list[str] = []

    retried = stage_candidate(
        package_dir=package_dir,
        source_artifacts=PackageSourceArtifactResolver(package_dir),
        workspace_root=workspace,
        plan=plan,
        session=session,
        semantic_constructor=CompleteSemanticConstructor(retry_calls),
        ordinary_relowerer=CompleteRelowerer(retry_calls),
        physical_optimizer=CompletePhysicalOptimizer(retry_calls),
    )

    assert retry_calls == [
        "semantic_construction",
        "ordinary_lowering",
        "physical_optimization",
    ]
    assert retried.status == "completed"
    assert retried.record != first.record
    assert retried.staged_candidate_path == (workspace / "ready" / plan.candidate_id)
    assert len(list((workspace / "records").glob("*.json"))) == 1
    load_staged_candidate(workspace, plan.candidate_id)


def test_abandoned_private_workspace_is_removed_before_construction(
    tmp_path: Path,
) -> None:
    package_dir, session = _package(tmp_path)
    plan = _plan()
    session = _session_with_candidate(session, plan)
    workspace = tmp_path / "workspace"
    abandoned = workspace / ".staging" / f"{plan.candidate_id}.interrupted-process"
    abandoned.mkdir(parents=True)
    (abandoned / "partial.bin").write_bytes(b"partial")
    abandoned_record = (
        workspace
        / "records"
        / ".staging"
        / f"{plan.candidate_id}.interrupted-record.json"
    )
    abandoned_record.parent.mkdir(parents=True)
    abandoned_record.write_bytes(b"partial")

    outcome = stage_candidate(
        package_dir=package_dir,
        source_artifacts=PackageSourceArtifactResolver(package_dir),
        workspace_root=workspace,
        plan=plan,
        session=session,
        semantic_constructor=CompleteSemanticConstructor([]),
        ordinary_relowerer=CompleteRelowerer([]),
        physical_optimizer=CompletePhysicalOptimizer([]),
    )

    assert outcome.status == "completed"
    assert not abandoned.exists()
    assert not abandoned_record.exists()
    assert not list((workspace / ".staging").iterdir())


def test_construction_measurement_includes_source_sealing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_dir, session = _package(tmp_path)
    plan = _plan()
    session = _session_with_candidate(session, plan)
    original = staging_orchestrator.seal_source_package
    measured: dict[str, int] = {}

    def measured_seal(package, build_plan, source_artifacts):
        started = time.monotonic_ns()
        time.sleep(0.01)
        result = original(package, build_plan, source_artifacts)
        measured["duration_ns"] = time.monotonic_ns() - started
        return result

    monkeypatch.setattr(
        staging_orchestrator,
        "seal_source_package",
        measured_seal,
    )
    outcome = stage_candidate(
        package_dir=package_dir,
        source_artifacts=PackageSourceArtifactResolver(package_dir),
        workspace_root=tmp_path / "workspace",
        plan=plan,
        session=session,
        semantic_constructor=CompleteSemanticConstructor([]),
        ordinary_relowerer=CompleteRelowerer([]),
        physical_optimizer=CompletePhysicalOptimizer([]),
    )

    assert (
        outcome.record.to_json()["resource_measurements"]["construction_time_ns"]
        >= measured["duration_ns"]
    )


def test_integrity_manifest_bytes_are_subject_to_staging_limit(
    tmp_path: Path,
) -> None:
    package_dir, session = _package(tmp_path)
    plan = _plan()
    session = _session_with_candidate(session, plan)
    first = stage_candidate(
        package_dir=package_dir,
        source_artifacts=PackageSourceArtifactResolver(package_dir),
        workspace_root=tmp_path / "measure-workspace",
        plan=plan,
        session=session,
        semantic_constructor=CompleteSemanticConstructor([]),
        ordinary_relowerer=CompleteRelowerer([]),
        physical_optimizer=CompletePhysicalOptimizer([]),
    )
    assert first.staged_candidate_path is not None
    integrity_bytes = (first.staged_candidate_path / "integrity.json").stat().st_size
    measured_peak = first.record.to_json()["resource_measurements"][
        "peak_staging_bytes"
    ]
    build = plan.construction_requirements.to_json()
    build["resource_limits"]["maximum_staging_bytes"] = measured_peak - (
        integrity_bytes // 2
    )
    limited_plan = replace(
        plan,
        construction_requirements=CandidateBuildPlan.from_json(build),
    )
    limited_session = _session_with_candidate(
        _package(tmp_path / "limited")[1],
        limited_plan,
    )
    limited_package = tmp_path / "limited" / "package"

    outcome = stage_candidate(
        package_dir=limited_package,
        source_artifacts=PackageSourceArtifactResolver(limited_package),
        workspace_root=tmp_path / "limited-workspace",
        plan=limited_plan,
        session=limited_session,
        semantic_constructor=CompleteSemanticConstructor([]),
        ordinary_relowerer=CompleteRelowerer([]),
        physical_optimizer=CompletePhysicalOptimizer([]),
    )

    assert outcome.status == "failed"
    assert "staging bytes exceeded" in " ".join(outcome.record.to_json()["diagnostics"])
    assert outcome.staged_candidate_path is None
    assert not (tmp_path / "limited-workspace" / "ready" / plan.candidate_id).exists()


def test_loader_reports_missing_construction_record_as_contract_error(
    tmp_path: Path,
) -> None:
    package_dir, session = _package(tmp_path)
    plan = _plan()
    session = _session_with_candidate(session, plan)
    workspace = tmp_path / "workspace"
    stage_candidate(
        package_dir=package_dir,
        source_artifacts=PackageSourceArtifactResolver(package_dir),
        workspace_root=workspace,
        plan=plan,
        session=session,
        semantic_constructor=CompleteSemanticConstructor([]),
        ordinary_relowerer=CompleteRelowerer([]),
        physical_optimizer=CompletePhysicalOptimizer([]),
    )
    next((workspace / "records").glob("*.json")).unlink()

    with pytest.raises(
        ModelCompileError,
        match="construction record is unreadable",
    ):
        load_staged_candidate(workspace, plan.candidate_id)


def test_invalid_target_artifact_fails_kind_validation_before_atomic_stage(
    tmp_path: Path,
) -> None:
    package_dir, session = _package(tmp_path)
    plan = _plan()
    session = _session_with_candidate(session, plan)
    workspace = tmp_path / "workspace"

    outcome = stage_candidate(
        package_dir=package_dir,
        source_artifacts=PackageSourceArtifactResolver(package_dir),
        workspace_root=workspace,
        plan=plan,
        session=session,
        semantic_constructor=CompleteSemanticConstructor([]),
        ordinary_relowerer=CompleteRelowerer([]),
        physical_optimizer=CompletePhysicalOptimizer(
            [],
            malformed_spirv=True,
        ),
    )

    assert outcome.status == "failed"
    assert "SPIR-V" in " ".join(outcome.record.to_json()["diagnostics"])
    assert not (workspace / "ready" / plan.candidate_id).exists()


def test_provider_added_artifact_validator_is_used_without_core_dispatch(
    tmp_path: Path,
) -> None:
    package_dir, session = _package(tmp_path)
    plan = _plan()
    build = plan.construction_requirements.to_json()
    kernel_output = next(
        output
        for output in build["outputs"]
        if output["path"] == "kernels/native_island.spv"
    )
    kernel_output["validator_id"] = "fixture_spirv"
    kernel_output["validation_contract"] = {"marker": 0x07230203}
    plan = replace(
        plan,
        construction_requirements=CandidateBuildPlan.from_json(build),
    )
    session = _session_with_candidate(session, plan)
    registry = ArtifactValidatorRegistry.with_builtin_validators()
    calls: list[tuple[int, dict[str, int]]] = []

    def validate_fixture_spirv(path, contract):
        with path.open("rb") as stream:
            marker = struct.unpack("<I", stream.read(4))[0]
        calls.append((marker, contract))
        if marker != contract["marker"]:
            raise ModelCompileError("fixture marker mismatch")
        return {"marker": marker}

    registry.register("fixture_spirv", validate_fixture_spirv)
    outcome = stage_candidate(
        package_dir=package_dir,
        source_artifacts=PackageSourceArtifactResolver(package_dir),
        workspace_root=tmp_path / "workspace",
        plan=plan,
        session=session,
        semantic_constructor=CompleteSemanticConstructor([]),
        ordinary_relowerer=CompleteRelowerer([]),
        physical_optimizer=CompletePhysicalOptimizer([]),
        artifact_validators=registry,
    )

    assert outcome.status == "completed"
    assert calls == [(0x07230203, {"marker": 0x07230203})]
    kernel = next(
        artifact
        for artifact in outcome.record.to_json()["artifacts"]
        if artifact["kind"] == "spirv"
    )
    assert kernel["validation"] == {
        "validator_id": "fixture_spirv",
        "status": "passed",
        "facts": {"marker": 0x07230203},
    }


def test_source_digest_drift_is_rejected_before_workspace_creation(
    tmp_path: Path,
) -> None:
    package_dir, session = _package(tmp_path)
    source = package_dir / "weights" / "source.bin"
    source.parent.mkdir()
    source.write_bytes(b"current bytes")
    plan = _plan()
    build = plan.construction_requirements.to_json()
    build["source_inputs"] = [
        {
            "path": "weights/source.bin",
            "digest": staged_artifact_digest(b"older bytes"),
        }
    ]
    plan = replace(
        plan,
        construction_requirements=CandidateBuildPlan.from_json(build),
    )
    session = _session_with_candidate(session, plan)
    workspace = tmp_path / "workspace"

    with pytest.raises(ModelCompileError, match="source input digest mismatch"):
        stage_candidate(
            package_dir=package_dir,
            source_artifacts=PackageSourceArtifactResolver(package_dir),
            workspace_root=workspace,
            plan=plan,
            session=session,
            semantic_constructor=CompleteSemanticConstructor([]),
            ordinary_relowerer=CompleteRelowerer([]),
            physical_optimizer=CompletePhysicalOptimizer([]),
        )
    assert not workspace.exists()


def test_build_plan_rejects_path_escape_reserved_paths_and_unknown_validator_shape():
    plan = _plan().construction_requirements.to_json()
    plan["outputs"][0]["path"] = "../escape.bin"
    with pytest.raises(ModelCompileError, match="normalized relative path"):
        CandidateBuildPlan.from_json(plan)

    plan = _plan().construction_requirements.to_json()
    plan["outputs"][0]["path"] = "contracts/forged.json"
    with pytest.raises(ModelCompileError, match="reserved path"):
        CandidateBuildPlan.from_json(plan)


def test_build_plan_cannot_be_rebound_to_undeclared_candidate_artifact(
    tmp_path: Path,
) -> None:
    package_dir, session = _package(tmp_path)
    plan = _plan()
    build = plan.construction_requirements.to_json()
    kernel_output = next(
        output
        for output in build["outputs"]
        if output["path"] == "kernels/native_island.spv"
    )
    kernel_output["path"] = "kernels/other.spv"
    build["outputs"] = sorted(build["outputs"], key=lambda output: output["path"])
    plan = replace(
        plan,
        construction_requirements=CandidateBuildPlan.from_json(build),
    )
    session = _session_with_candidate(session, plan)

    with pytest.raises(ModelCompileError, match="artifact declarations"):
        stage_candidate(
            package_dir=package_dir,
            source_artifacts=PackageSourceArtifactResolver(package_dir),
            workspace_root=tmp_path / "workspace",
            plan=plan,
            session=session,
            semantic_constructor=CompleteSemanticConstructor([]),
            ordinary_relowerer=CompleteRelowerer([]),
            physical_optimizer=CompletePhysicalOptimizer([]),
        )


def test_workspace_cannot_be_inside_or_contain_immutable_package(
    tmp_path: Path,
) -> None:
    package_dir, session = _package(tmp_path)
    plan = _plan()
    session = _session_with_candidate(session, plan)
    services = {
        "semantic_constructor": CompleteSemanticConstructor([]),
        "ordinary_relowerer": CompleteRelowerer([]),
        "physical_optimizer": CompletePhysicalOptimizer([]),
    }

    with pytest.raises(ModelCompileError, match="outside"):
        stage_candidate(
            package_dir=package_dir,
            source_artifacts=PackageSourceArtifactResolver(package_dir),
            workspace_root=package_dir / "optimization" / "working",
            plan=plan,
            session=session,
            **services,
        )
    with pytest.raises(ModelCompileError, match="must not contain"):
        stage_candidate(
            package_dir=package_dir,
            source_artifacts=PackageSourceArtifactResolver(package_dir),
            workspace_root=tmp_path,
            plan=plan,
            session=session,
            **services,
        )
