from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from nerve.compilation import ModelCompileError
from nerve.representation_optimizer.benchmarking.orchestrator import (
    benchmark_candidate,
)
from nerve.representation_optimizer.benchmarking.planning import (
    build_benchmark_plan,
)
from nerve.representation_optimizer.contracts import (
    REPRESENTATION_CANDIDATE_SCHEMA,
    ContractDocument,
    contract_digest,
    device_state_digest,
    representation_candidate_id,
)
from nerve.representation_optimizer.lifecycle import CandidateState
from nerve.representation_optimizer.mounting import RuntimeMountPlan
from nerve.representation_optimizer.representation_ir import (
    RepresentationGraphDocument,
    representation_graph_id,
)
from nerve.representation_optimizer.providers.source_artifacts import (
    PackageSourceArtifactResolver,
)
from nerve.representation_optimizer.staging.contracts import (
    staged_artifact_digest,
)
from nerve.representation_optimizer.staging.orchestrator import (
    stage_candidate,
)
from nerve.representation_optimizer.validation.contracts import (
    PROOF_RESULT_SCHEMA,
    VALIDATION_REQUIREMENTS_SCHEMA,
    VALIDATION_RESIDENCY_EVENT_SCHEMA,
    VALIDATION_ROLE_RESULT_SCHEMA,
    ProofResult,
    ValidationPlan,
    ValidationRequirements,
    ValidationRoleResult,
    proof_result_id,
    validation_check_id,
    validation_plan_id,
    validation_requirements_id,
    validation_residency_event_id,
    validation_role_result_id,
)
from nerve.representation_optimizer.validation.orchestrator import (
    prepare_candidate_for_benchmark,
    validate_benchmarked_candidate,
)
from nerve.representation_optimizer.validation.planning import (
    build_validation_plan,
    create_behavioral_error_contract,
    create_validation_requirements,
)
from nerve.representation_optimizer.validation.proofs import (
    ProofVerifierRegistry,
)
from nerve.representation_optimizer.validation.runner import (
    execute_validation_stage,
)
from nerve.representation_optimizer.validation.storage import (
    _proof_artifact_readers,
    load_prebenchmark_evidence,
    load_validation_evidence,
)
from tests.test_candidate_benchmarking import (
    AdapterBehavior,
    FixtureExecutionAdapter,
)
from tests.test_candidate_staging import (
    CompletePhysicalOptimizer,
    CompleteRelowerer,
    CompleteSemanticConstructor,
    _package,
    _plan,
    _session_with_candidate,
)
from tests.test_representation_optimizer_contracts import (
    hardware_profile_contract,
)


class FixtureProofVerifier:
    verifier_id = "fixture.exact_reconstruction.v1"

    def __init__(
        self,
        status: str = "proven",
        *,
        emit_artifact: bool = False,
    ) -> None:
        self.status = status
        self.emit_artifact = emit_artifact
        self.requests = []
        self.artifact_payload = b"fixture algebraic proof certificate"

    def verify(self, request):
        self.requests.append(request)
        document = {
            "schema": PROOF_RESULT_SCHEMA,
            "proof_id": "",
            "plan_id": request.plan_id,
            "candidate_id": request.candidate_id,
            "obligation": request.obligation,
            "verifier_id": request.verifier_id,
            "source_contract_digests": list(
                request.source_contract_digests
            ),
            "construction_record_digest": (
                request.construction_record_digest
            ),
            "status": self.status,
            "facts": (
                {"canonical_reconstruction": True}
                if self.status == "proven"
                else {}
            ),
            "artifacts": (
                [
                    {
                        "path": "proofs/fixture-certificate.bin",
                        "digest": staged_artifact_digest(
                            self.artifact_payload
                        ),
                    }
                ]
                if self.emit_artifact
                else []
            ),
            "diagnostics": (
                []
                if self.status == "proven"
                else ["fixture proof does not establish equivalence"]
            ),
        }
        document["proof_id"] = proof_result_id(document)
        return ProofResult.from_json(document).to_json()

    def iter_proof_artifact(
        self,
        relative_path,
        *,
        chunk_bytes=8 * 1024 * 1024,
    ):
        if relative_path != "proofs/fixture-certificate.bin":
            raise KeyError(relative_path)
        for offset in range(
            0,
            len(self.artifact_payload),
            chunk_bytes,
        ):
            yield self.artifact_payload[offset : offset + chunk_bytes]


def test_identical_proof_artifact_can_support_multiple_obligations() -> None:
    verifier = FixtureProofVerifier(emit_artifact=True)
    registry = ProofVerifierRegistry.from_verifiers((verifier,))
    reference = {
        "path": "proofs/fixture-certificate.bin",
        "digest": staged_artifact_digest(verifier.artifact_payload),
    }

    readers = _proof_artifact_readers(
        [
            {
                "verifier_id": verifier.verifier_id,
                "artifacts": [reference],
            },
            {
                "verifier_id": verifier.verifier_id,
                "artifacts": [reference],
            },
        ],
        registry,
    )

    assert len(readers) == 1
    assert readers[0][0] == reference
    assert b"".join(readers[0][1](reference["path"])) == (
        verifier.artifact_payload
    )


def test_conflicting_proof_artifact_ownership_is_rejected() -> None:
    verifier = FixtureProofVerifier(emit_artifact=True)
    registry = ProofVerifierRegistry.from_verifiers((verifier,))

    with pytest.raises(
        ModelCompileError,
        match="conflicting artifact ownership",
    ):
        _proof_artifact_readers(
            [
                {
                    "verifier_id": verifier.verifier_id,
                    "artifacts": [
                        {
                            "path": "proofs/fixture-certificate.bin",
                            "digest": staged_artifact_digest(b"first"),
                        }
                    ],
                },
                {
                    "verifier_id": verifier.verifier_id,
                    "artifacts": [
                        {
                            "path": "proofs/fixture-certificate.bin",
                            "digest": staged_artifact_digest(b"second"),
                        }
                    ],
                },
            ],
            registry,
        )


@dataclass(frozen=True)
class ValidationBehavior:
    invalid_stage: str | None = None
    invalid_error: float = 0.5
    incomplete_steps: bool = False
    fail_execution_stage: str | None = None
    leak_residency: bool = False
    candidate_warmup_elapsed_ns: int = 100
    candidate_measured_elapsed_ns: int = 90
    candidate_measured_generated_tokens: int = 8
    candidate_bounded_wait_timeout_count: int = 0


class FixtureValidationAdapter:
    def __init__(
        self,
        behavior: ValidationBehavior | None = None,
    ) -> None:
        self.behavior = behavior or ValidationBehavior()
        self.fixture_artifacts = {
            "fixtures/decode-input.bin": b"fixture input",
            "fixtures/decode-state.bin": b"fixture state",
            "fixtures/prefill-input.bin": b"fixture prefill input",
            "fixtures/prefill-state.bin": b"fixture prefill state",
            "fixtures/model-limits.json": (
                b'{"max_context_tokens":131072,'
                b'"max_output_tokens":65536}'
            ),
        }
        self.trace_artifacts: dict[str, bytes] = {}
        self.mount_requests = []
        self.execution_requests = []
        self.closed_sessions: list[tuple[str, str, int]] = []
        self.fixture_candidate_ids: list[str] = []
        self.validation_stages: list[tuple[str, str]] = []

    @contextmanager
    def validation_stage(
        self,
        stage,
        *,
        execution_scope,
        cancel_requested=None,
    ):
        assert cancel_requested is None or not cancel_requested()
        self.validation_stages.append((stage, execution_scope))
        yield

    def iter_fixture_artifact(
        self,
        relative_path,
        *,
        candidate_id,
        chunk_bytes=8 * 1024 * 1024,
    ):
        self.fixture_candidate_ids.append(candidate_id)
        payload = self.fixture_artifacts[relative_path]
        for offset in range(0, len(payload), chunk_bytes):
            yield payload[offset : offset + chunk_bytes]

    def iter_trace_artifact(
        self,
        relative_path,
        *,
        chunk_bytes=8 * 1024 * 1024,
    ):
        payload = self.trace_artifacts[relative_path]
        for offset in range(0, len(payload), chunk_bytes):
            yield payload[offset : offset + chunk_bytes]

    def open_session(self, request):
        self.mount_requests.append(request)
        return FixtureValidationRoleSession(self, request)

    def compare_results(self, request, reference_result, candidate_result):
        invalid = self.behavior.invalid_stage == request["check"]["stage"]
        error = self.behavior.invalid_error if invalid else 0.0
        return {
            "metrics": [
                {
                    "name": name,
                    "reference_value": 1.0,
                    "candidate_value": 1.0 - error,
                    "error": error,
                    "unit": "normalized_error",
                }
                for name in request["check"]["metrics"]
            ],
            "diagnostics": [],
        }


class FixtureValidationRoleSession:
    def __init__(self, adapter, request) -> None:
        self.adapter = adapter
        self.request = request
        self.closed = False
        self.idle = request.matched_conditions[
            "idle_device_state_digest"
        ]
        self.mounted = device_state_digest(
            {
                "fixture_state": "mounted",
                "stage": request.stage,
                "role": request.role,
                "block_index": request.block_index,
            }
        )
        self._mount_event = self._residency_event(
            action="mount",
            before=self.idle,
            after=self.mounted,
            released=False,
        )

    @property
    def mount_event(self):
        return dict(self._mount_event)

    def execute(self, request):
        if self.closed:
            raise RuntimeError("validation fixture stage is closed")
        self.adapter.execution_requests.append(request)
        behavior = self.adapter.behavior
        if behavior.fail_execution_stage == request.check["stage"]:
            raise ModelCompileError("fixture validation execution failed")
        invalid = (
            request.role == "candidate"
            and behavior.invalid_stage == request.check["stage"]
        )
        exact_output = staged_artifact_digest(
            (
                f"output:{request.check['check_id']}:{request.seed}"
            ).encode()
        )
        output = (
            staged_artifact_digest(
                (
                    f"invalid-output:{request.check['check_id']}:"
                    f"{request.seed}"
                ).encode()
            )
            if invalid
            else exact_output
        )
        exact_state = staged_artifact_digest(
            (
                f"state:{request.check['check_id']}:{request.seed}"
            ).encode()
        )
        state = (
            staged_artifact_digest(
                (
                    f"invalid-state:{request.check['check_id']}:"
                    f"{request.seed}"
                ).encode()
            )
            if invalid
            else exact_state
        )
        horizon = request.check["horizon"]
        if horizon["completion_condition"] == "minimum_steps":
            minimum_steps = horizon["minimum_steps"]
            steps = (
                minimum_steps - 1
                if behavior.incomplete_steps
                else minimum_steps
            )
            horizon_completion = {
                "condition": "minimum_steps",
                "satisfied": not behavior.incomplete_steps,
                "observed_steps": steps,
                "minimum_steps": minimum_steps,
                "expected_turns": None,
                "completed_turns": None,
                "stop_reasons": [],
            }
        else:
            steps = 1
            horizon_completion = {
                "condition": (
                    "semantic_stop_or_allowance_per_turn"
                ),
                "satisfied": not behavior.incomplete_steps,
                "observed_steps": steps,
                "minimum_steps": None,
                "expected_turns": 1,
                "completed_turns": (
                    0 if behavior.incomplete_steps else 1
                ),
                "stop_reasons": (
                    [] if behavior.incomplete_steps else ["eos"]
                ),
            }
        payload = (
            f"{request.check['stage']}:{request.check['check_id']}:"
            f"{request.seed}:{request.role}"
        ).encode()
        path = (
            f"traces/{request.check['stage']}/"
            f"{request.check['check_id']}/{request.seed}/"
            f"{request.role}.bin"
        )
        self.adapter.trace_artifacts[path] = payload
        document = {
            "schema": VALIDATION_ROLE_RESULT_SCHEMA,
            "result_id": "",
            "plan_id": request.plan_id,
            "check_id": request.check["check_id"],
            "stage": request.check["stage"],
            "seed": request.seed,
            "role": request.role,
            "implementation_id": request.implementation[
                "implementation_id"
            ],
            "status": "completed",
            "output_digest": output,
            "state_digest": state,
            "steps": steps,
            "horizon_completion": horizon_completion,
            "traces": [
                {
                    "path": path,
                    "digest": staged_artifact_digest(payload),
                }
            ],
            "default_statistics": {
                "execution_path": "normal_fixture_runtime",
                "role": request.role,
                "host_execution_ns": (
                    200
                    if request.role == "reference"
                    else (
                        100
                        + behavior.candidate_measured_elapsed_ns
                    )
                ),
                "device_execution_ns": (
                    200
                    if request.role == "reference"
                    else (
                        100
                        + behavior.candidate_measured_elapsed_ns
                    )
                ),
                "transport_bytes": 0,
                "scheduler_steps": 2,
                "execution_counters": {
                    "execution_quantum_forced_yield_count": 0,
                    "resident_copy_waits": 0,
                    "resident_sequence_fence_waits": 0,
                },
                "turn_statistics": [
                    _fixture_turn_statistics(
                        turn_index=0,
                        elapsed_ns=(
                            100
                            if request.role == "reference"
                            else behavior.candidate_warmup_elapsed_ns
                        ),
                    ),
                    _fixture_turn_statistics(
                        turn_index=1,
                        elapsed_ns=(
                            100
                            if request.role == "reference"
                            else behavior.candidate_measured_elapsed_ns
                        ),
                        bounded_wait_timeout_count=(
                            0
                            if request.role == "reference"
                            else behavior.candidate_bounded_wait_timeout_count
                        ),
                        generated_tokens=(
                            8
                            if request.role == "reference"
                            else behavior.candidate_measured_generated_tokens
                        ),
                    ),
                ],
            },
            "diagnostics": [],
        }
        document["result_id"] = validation_role_result_id(document)
        return ValidationRoleResult.from_json(document).to_json()

    def close(self):
        if self.closed:
            raise RuntimeError("validation fixture stage closed twice")
        self.closed = True
        self.adapter.closed_sessions.append(
            (
                self.request.stage,
                self.request.role,
                self.request.block_index,
            )
        )
        after = (
            staged_artifact_digest(
                f"leaked:{self.request.stage}".encode()
            )
            if self.adapter.behavior.leak_residency
            else self.idle
        )
        return self._residency_event(
            action="unmount",
            before=self.mounted,
            after=after,
            released=True,
        )

    def _residency_event(self, *, action, before, after, released):
        document = {
            "schema": VALIDATION_RESIDENCY_EVENT_SCHEMA,
            "event_id": "",
            "plan_id": self.request.plan_id,
            "stage": self.request.stage,
            "check_id": self.request.check["check_id"],
            "seed": self.request.seed,
            "role": self.request.role,
            "implementation_id": self.request.implementation[
                "implementation_id"
            ],
            "block_index": self.request.block_index,
            "action": action,
            "duration_ns": 100,
            "device_state_before_digest": before,
            "device_state_after_digest": after,
            "released": released,
            "default_statistics": {
                "execution_path": "normal_fixture_runtime"
            },
        }
        document["event_id"] = validation_residency_event_id(document)
        return document


def _fixture_turn_statistics(
    *,
    turn_index: int,
    elapsed_ns: int,
    generated_tokens: int = 8,
    bounded_wait_timeout_count: int = 0,
) -> dict:
    return {
        "turn_index": turn_index,
        "generated_tokens": generated_tokens,
        "elapsed_ns": elapsed_ns,
        "scheduler_steps": 1,
        "execution_counters": {
            "execution_quantum_forced_yield_count": 0,
            "resident_copy_waits": 0,
            "resident_sequence_fence_waits": 0,
        },
        "speculative": {
            "proposed_draft_tokens": 0,
            "accepted_draft_tokens": 0,
        },
        "resident_feedback": {
            "bounded_wait_count": 0,
            "bounded_wait_timeout_count": bounded_wait_timeout_count,
        },
        "transport": {
            "published_packet_count": 0,
            "published_byte_count": 0,
            "received_packet_count": 0,
            "received_byte_count": 0,
            "direct_copy_count": 0,
            "direct_copy_byte_count": 0,
        },
    }


def _staged_fixture(tmp_path: Path, *, approximate: bool = False):
    package_dir, session = _package(tmp_path)
    candidate_plan = _plan()
    if approximate:
        candidate_plan = _approximate_plan(candidate_plan)
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
    profile = hardware_profile_contract()
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
                "path": "lowered/exact-reference.json",
                "digest": staged_artifact_digest(b"exact reference"),
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
    return (
        package_dir,
        candidate_workspace,
        candidate_plan,
        construction,
        benchmark_plan,
        validation_plan,
    )


def _approximate_plan(plan):
    document = plan.candidate.to_json()
    required_coverage = {
        coverage["kind"]
        for coverage in plan.validation_requirements.to_json()["coverage"]
        if coverage["applicability"] == "required"
    }
    error_contract = create_behavioral_error_contract(
        validity_predicates={"fixture_target": True},
        metric_limits={
            "exact_match": (
                0.01,
                "normalized_error",
                required_coverage,
            )
        },
        correction_mode="reject",
        correction_trigger_metrics=("exact_match",),
        correction_action="retain exact implementation",
    ).to_json()
    document["behavioral_contract"] = {
        "mode": "approximate",
        "proof_obligations": [],
        "error_contract": error_contract,
    }
    document["candidate_id"] = representation_candidate_id(document)
    candidate = ContractDocument.from_json(
        document,
        expected_schema=REPRESENTATION_CANDIDATE_SCHEMA,
    )
    graph_document = plan.representation_ir.to_json()
    graph_document["candidate_id"] = document["candidate_id"]
    graph_document["graph_id"] = representation_graph_id(graph_document)
    graph = RepresentationGraphDocument.from_json(graph_document)
    source_requirements = plan.validation_requirements.to_json()
    requirements = create_validation_requirements(
        candidate_id=document["candidate_id"],
        source_contract_digests=document["source_contract_digests"],
        proof_verifiers={},
        checks=source_requirements["checks"],
        not_applicable_reasons={
            coverage["kind"]: coverage["reason"]
            for coverage in source_requirements["coverage"]
            if coverage["applicability"] == "not_applicable"
        },
        counterexamples=source_requirements["counterexamples"],
    )
    mount_document = plan.mount_requirements.to_json()
    mount_document["candidate_id"] = document["candidate_id"]
    mount_requirements = RuntimeMountPlan.from_json(
        mount_document,
        candidate_id=document["candidate_id"],
        build_plan=plan.construction_requirements,
    )
    return replace(
        plan,
        candidate=candidate,
        representation_ir=graph,
        mount_requirements=mount_requirements,
        proof_or_error_contract=document["behavioral_contract"],
        validation_requirements=requirements,
    )


def _prevalidate(
    tmp_path: Path,
    fixture,
    *,
    adapter: FixtureValidationAdapter | None = None,
    verifier: FixtureProofVerifier | None = None,
):
    (
        package_dir,
        candidate_workspace,
        candidate_plan,
        construction,
        _,
        validation_plan,
    ) = fixture
    adapter = adapter or FixtureValidationAdapter()
    verifier = verifier or FixtureProofVerifier()
    outcome = prepare_candidate_for_benchmark(
        package_dir=package_dir,
        candidate_workspace_root=candidate_workspace,
        validation_workspace_root=tmp_path / "validation-workspace",
        candidate_plan=candidate_plan,
        construction_record=construction.record,
        validation_plan=validation_plan,
        session=construction.session,
        proof_verifiers=ProofVerifierRegistry.from_verifiers((verifier,)),
        adapter=adapter,
    )
    return outcome, adapter, verifier


def test_mixed_validation_stage_releases_component_scope_before_whole_model(
    tmp_path: Path,
) -> None:
    fixture = _staged_fixture(tmp_path)
    document = fixture[5].to_json()
    component_check = deepcopy(next(
        check
        for check in document["checks"]
        if check["regime"]["execution_scope"] == "component"
    ))
    component_check["stage"] = "full_local"
    component_check["check_id"] = validation_check_id(component_check)
    document["checks"].append(component_check)
    document["checks"].sort(key=lambda check: check["check_id"])
    requirements = {
        "schema": VALIDATION_REQUIREMENTS_SCHEMA,
        "requirements_id": "",
        "candidate_id": document["candidate_id"],
        "source_contract_digests": document["source_contract_digests"],
        "proofs": document["proofs"],
        "checks": document["checks"],
        "coverage": document["coverage"],
        "counterexamples": document["counterexamples"],
    }
    requirements["requirements_id"] = validation_requirements_id(
        requirements
    )
    document["requirements_digest"] = contract_digest(requirements)
    document["plan_id"] = validation_plan_id(document)
    plan = ValidationPlan.from_json(document)
    adapter = FixtureValidationAdapter()

    run = execute_validation_stage(
        plan,
        stage="full_local",
        adapter=adapter,
    )

    assert run.status == "completed"
    assert adapter.validation_stages == [
        ("full_local", "component"),
        ("full_local", "whole_model"),
    ]


def test_proven_exact_candidate_passes_complete_validation_funnel(
    tmp_path: Path,
) -> None:
    fixture = _staged_fixture(tmp_path)
    pre, adapter, verifier = _prevalidate(tmp_path, fixture)
    construction = fixture[3]
    benchmark_plan = fixture[4]

    assert pre.status == "passed"
    assert len(verifier.requests) == 1
    lifecycle = next(
        candidate
        for candidate in pre.session.candidates
        if candidate.candidate_id == fixture[2].candidate_id
    )
    assert lifecycle.state == CandidateState.PREBENCHMARK_VALIDATED
    benchmark = benchmark_candidate(
        plan=benchmark_plan,
        construction_record=construction.record,
        session=pre.session,
        adapter=FixtureExecutionAdapter(),
        workspace_root=tmp_path / "benchmark-workspace",
    )
    final = validate_benchmarked_candidate(
        plan=fixture[5],
        prebenchmark_record=pre.record,
        benchmark_record=benchmark.record,
        session=benchmark.session,
        adapter=adapter,
        workspace_root=tmp_path / "validation-workspace",
    )

    assert final.status == "passed"
    assert final.record.to_json()["status"] == "passed"
    assert [run.to_json()["stage"] for run in final.runs] == [
        "full_local",
        "whole_model",
    ]
    whole_model_run = final.runs[-1].to_json()
    assert all(
        observation[role]["steps"] == 1
        and observation[role]["horizon_completion"] == {
            "condition": "semantic_stop_or_allowance_per_turn",
            "satisfied": True,
            "observed_steps": 1,
            "minimum_steps": None,
            "expected_turns": 1,
            "completed_turns": 1,
            "stop_reasons": ["eos"],
        }
        for observation in whole_model_run["observations"]
        for role in ("reference", "candidate")
    )
    for run in (pre.sanity_run, *final.runs):
        assert run is not None
        run_document = run.to_json()
        assert len(run_document["residency_events"]) == (
            4 * len(run_document["observations"])
        )
        assert [
            (event["role"], event["action"])
            for event in run_document["residency_events"]
        ] == [
            pair
            for _observation in run_document["observations"]
            for role in ("reference", "candidate")
            for pair in ((role, "mount"), (role, "unmount"))
        ]
        assert all(
            event["device_state_before_digest"]
            == fixture[4].matched_conditions[
                "idle_device_state_digest"
            ]
            for event in run_document["residency_events"]
            if event["action"] == "mount"
        )
    assert {
        stage for stage, _role, _block in adapter.closed_sessions
    } == {"sanity", "full_local", "whole_model"}
    assert len(adapter.closed_sessions) == len(adapter.mount_requests)
    for completed_stage in ("sanity", "full_local", "whole_model"):
        stage_sessions = [
            (role, block)
            for stage, role, block in adapter.closed_sessions
            if stage == completed_stage
        ]
        blocks = [block for _role, block in stage_sessions]
        assert blocks == list(range(len(blocks)))
        roles = [role for role, _block in stage_sessions]
        assert roles == [
            role
            for _observation in range(len(roles) // 2)
            for role in ("reference", "candidate")
        ]
    assert set(adapter.fixture_candidate_ids) == {fixture[2].candidate_id}
    assert len(
        {
            request.seed
            for request in adapter.execution_requests
            if request.check["stage"] == "whole_model"
        }
    ) >= 2
    lifecycle = next(
        candidate
        for candidate in final.session.candidates
        if candidate.candidate_id == fixture[2].candidate_id
    )
    assert lifecycle.state == CandidateState.BEHAVIORALLY_VALIDATED
    assert load_prebenchmark_evidence(
        tmp_path / "validation-workspace",
        pre.record.to_json()["prebenchmark_id"],
    ) == (pre.plan, pre.record, pre.sanity_run)
    assert load_validation_evidence(
        tmp_path / "validation-workspace",
        final.record.to_json()["validation_id"],
    ) == (
        final.plan,
        pre.record,
        benchmark.record,
        final.runs,
        final.record,
    )


@pytest.mark.parametrize(
    ("behavior", "reason"),
    (
        (
            ValidationBehavior(candidate_measured_elapsed_ns=110),
            "not faster",
        ),
        (
            ValidationBehavior(
                candidate_measured_elapsed_ns=90,
                candidate_bounded_wait_timeout_count=1,
            ),
            "amplified",
        ),
    ),
)
def test_locally_faster_candidate_is_rejected_when_warmed_product_run_regresses(
    tmp_path: Path,
    behavior: ValidationBehavior,
    reason: str,
) -> None:
    fixture = _staged_fixture(tmp_path)
    adapter = FixtureValidationAdapter(behavior)
    pre, adapter, _ = _prevalidate(
        tmp_path,
        fixture,
        adapter=adapter,
    )
    benchmark = benchmark_candidate(
        plan=fixture[4],
        construction_record=fixture[3].record,
        session=pre.session,
        adapter=FixtureExecutionAdapter(),
        workspace_root=tmp_path / "benchmark-workspace",
    )

    final = validate_benchmarked_candidate(
        plan=fixture[5],
        prebenchmark_record=pre.record,
        benchmark_record=benchmark.record,
        session=benchmark.session,
        adapter=adapter,
        workspace_root=tmp_path / "validation-workspace",
    )

    assert final.status == CandidateState.REJECTED.value
    record = final.record.to_json()
    stage = next(
        stage
        for stage in record["stages"]
        if stage["name"] == "whole_model_product_performance"
    )
    assert stage["status"] == "failed"
    assert reason in stage["reason"]
    assert stage["metrics"]["warmup_turns_discarded"] >= 1
    assert record["status"] == "failed"
    lifecycle = next(
        candidate
        for candidate in final.session.candidates
        if candidate.candidate_id == fixture[2].candidate_id
    )
    assert lifecycle.state == CandidateState.REJECTED


def test_product_performance_discards_the_first_conversation_as_warmup(
    tmp_path: Path,
) -> None:
    fixture = _staged_fixture(tmp_path)
    adapter = FixtureValidationAdapter(
        ValidationBehavior(
            candidate_warmup_elapsed_ns=10_000,
            candidate_measured_elapsed_ns=90,
        )
    )
    pre, adapter, _ = _prevalidate(
        tmp_path,
        fixture,
        adapter=adapter,
    )
    benchmark = benchmark_candidate(
        plan=fixture[4],
        construction_record=fixture[3].record,
        session=pre.session,
        adapter=FixtureExecutionAdapter(),
        workspace_root=tmp_path / "benchmark-workspace",
    )

    final = validate_benchmarked_candidate(
        plan=fixture[5],
        prebenchmark_record=pre.record,
        benchmark_record=benchmark.record,
        session=benchmark.session,
        adapter=adapter,
        workspace_root=tmp_path / "validation-workspace",
    )

    assert final.status == "passed"
    stage = next(
        stage
        for stage in final.record.to_json()["stages"]
        if stage["name"] == "whole_model_product_performance"
    )
    assert stage["status"] == "passed"
    assert stage["metrics"]["reference_measured_elapsed_ns"] == 200
    assert stage["metrics"]["candidate_measured_elapsed_ns"] == 180
    assert stage["metrics"]["warmup_turns_discarded"] == 2


def test_product_performance_refuses_timings_for_different_generated_work(
    tmp_path: Path,
) -> None:
    fixture = _staged_fixture(tmp_path)
    adapter = FixtureValidationAdapter(
        ValidationBehavior(candidate_measured_generated_tokens=7)
    )
    pre, adapter, _ = _prevalidate(
        tmp_path,
        fixture,
        adapter=adapter,
    )
    benchmark = benchmark_candidate(
        plan=fixture[4],
        construction_record=fixture[3].record,
        session=pre.session,
        adapter=FixtureExecutionAdapter(),
        workspace_root=tmp_path / "benchmark-workspace",
    )

    final = validate_benchmarked_candidate(
        plan=fixture[5],
        prebenchmark_record=pre.record,
        benchmark_record=benchmark.record,
        session=benchmark.session,
        adapter=adapter,
        workspace_root=tmp_path / "validation-workspace",
    )

    assert final.status == CandidateState.FAILED.value
    stage = next(
        stage
        for stage in final.record.to_json()["stages"]
        if stage["name"] == "whole_model_product_performance"
    )
    assert stage["status"] == "failed"
    assert "different generated work" in stage["reason"]


def test_faster_but_behaviorally_invalid_approximation_is_rejected(
    tmp_path: Path,
) -> None:
    fixture = _staged_fixture(tmp_path, approximate=True)
    adapter = FixtureValidationAdapter(
        ValidationBehavior(invalid_stage="full_local")
    )
    pre, adapter, verifier = _prevalidate(
        tmp_path,
        fixture,
        adapter=adapter,
    )
    assert pre.status == "passed"
    assert verifier.requests == []
    benchmark = benchmark_candidate(
        plan=fixture[4],
        construction_record=fixture[3].record,
        session=pre.session,
        adapter=FixtureExecutionAdapter(),
        workspace_root=tmp_path / "benchmark-workspace",
    )
    assert benchmark.record.to_json()["decision"] == "materially_faster"

    final = validate_benchmarked_candidate(
        plan=fixture[5],
        prebenchmark_record=pre.record,
        benchmark_record=benchmark.record,
        session=benchmark.session,
        adapter=adapter,
        workspace_root=tmp_path / "validation-workspace",
    )

    assert final.status == CandidateState.REJECTED.value
    assert final.record.to_json()["status"] == "failed"
    stages = {
        stage["name"]: stage
        for stage in final.record.to_json()["stages"]
    }
    assert stages["matched_performance"]["status"] == "passed"
    assert stages["full_local_behavior"]["status"] == "failed"
    assert stages["whole_model_free_running"]["status"] == "not_run"
    assert stages["whole_model_product_performance"]["status"] == "not_run"
    assert final.record.to_json()["counterexamples"] == [
        {
            "path": "fixtures/prefill-input.bin",
            "digest": staged_artifact_digest(
                b"fixture prefill input"
            ),
        }
    ]
    assert "whole_model" not in {
        stage for stage, _role, _block in adapter.closed_sessions
    }


def test_exact_candidate_with_behavioral_divergence_is_rejected_before_timing(
    tmp_path: Path,
) -> None:
    fixture = _staged_fixture(tmp_path)
    pre, adapter, _ = _prevalidate(
        tmp_path,
        fixture,
        adapter=FixtureValidationAdapter(
            ValidationBehavior(invalid_stage="sanity")
        ),
    )

    assert pre.status == CandidateState.REJECTED.value
    assert [
        role for stage, role, _block in adapter.closed_sessions
        if stage == "sanity"
    ] == ["reference", "candidate"]
    assert pre.record.to_json()["stages"][2]["status"] == "failed"


def test_nonwinning_benchmark_skips_expensive_behavioral_validation(
    tmp_path: Path,
) -> None:
    fixture = _staged_fixture(tmp_path)
    pre, adapter, _ = _prevalidate(tmp_path, fixture)
    benchmark = benchmark_candidate(
        plan=fixture[4],
        construction_record=fixture[3].record,
        session=pre.session,
        adapter=FixtureExecutionAdapter(
            AdapterBehavior(candidate_duration_ns=1_100_000)
        ),
        workspace_root=tmp_path / "benchmark-workspace",
    )
    assert benchmark.record.to_json()["decision"] == (
        "not_materially_faster"
    )
    final = validate_benchmarked_candidate(
        plan=fixture[5],
        prebenchmark_record=pre.record,
        benchmark_record=benchmark.record,
        session=benchmark.session,
        adapter=adapter,
        workspace_root=tmp_path / "validation-workspace",
    )

    assert final.status == CandidateState.REJECTED.value
    assert final.runs == ()
    assert {
        stage for stage, _role, _block in adapter.closed_sessions
    } == {"sanity"}


def test_validation_requirements_cannot_waive_long_horizon_coverage(
    tmp_path: Path,
) -> None:
    requirements = _staged_fixture(tmp_path)[
        2
    ].validation_requirements.to_json()
    long_output = next(
        coverage
        for coverage in requirements["coverage"]
        if coverage["kind"] == "long_output"
    )
    long_output.update(
        {
            "applicability": "not_applicable",
            "check_ids": [],
            "reason": "conveniently skipped",
        }
    )
    requirements["requirements_id"] = validation_requirements_id(
        requirements
    )

    with pytest.raises(
        ModelCompileError,
        match="cannot waive whole-pipeline coverage",
    ):
        ValidationRequirements.from_json(requirements)


def test_validation_coverage_cannot_be_assigned_to_wrong_check_kind(
    tmp_path: Path,
) -> None:
    requirements = _staged_fixture(tmp_path)[
        2
    ].validation_requirements.to_json()
    check = next(
        check
        for check in requirements["checks"]
        if "free_running_long_horizon" in check["coverage"]
    )
    old_id = check["check_id"]
    check["kind"] = "component_comparison"
    check["check_id"] = validation_check_id(check)
    for coverage in requirements["coverage"]:
        coverage["check_ids"] = sorted(
            check["check_id"] if value == old_id else value
            for value in coverage["check_ids"]
        )
    requirements["checks"].sort(key=lambda item: item["check_id"])
    requirements["requirements_id"] = validation_requirements_id(
        requirements
    )

    with pytest.raises(ModelCompileError, match="incompatible"):
        ValidationRequirements.from_json(requirements)


def test_unproven_exact_candidate_never_reaches_timing(
    tmp_path: Path,
) -> None:
    fixture = _staged_fixture(tmp_path)
    adapter = FixtureValidationAdapter()
    pre, adapter, _ = _prevalidate(
        tmp_path,
        fixture,
        adapter=adapter,
        verifier=FixtureProofVerifier("inconclusive"),
    )

    assert pre.status == CandidateState.REJECTED.value
    assert pre.sanity_run is None
    assert adapter.mount_requests == []
    lifecycle = next(
        candidate
        for candidate in pre.session.candidates
        if candidate.candidate_id == fixture[2].candidate_id
    )
    assert lifecycle.state == CandidateState.REJECTED
    with pytest.raises(ModelCompileError, match="proof and prebenchmark"):
        benchmark_candidate(
            plan=fixture[4],
            construction_record=fixture[3].record,
            session=pre.session,
            adapter=FixtureExecutionAdapter(),
            workspace_root=tmp_path / "benchmark-workspace",
        )


def test_proof_certificates_are_published_with_prebenchmark_evidence(
    tmp_path: Path,
) -> None:
    fixture = _staged_fixture(tmp_path)
    verifier = FixtureProofVerifier(emit_artifact=True)
    pre, _, _ = _prevalidate(
        tmp_path,
        fixture,
        verifier=verifier,
    )

    assert pre.status == "passed"
    assert (
        pre.evidence_path / "proofs" / "fixture-certificate.bin"
    ).read_bytes() == verifier.artifact_payload


def test_tampered_staged_artifact_fails_static_gate_before_execution(
    tmp_path: Path,
) -> None:
    fixture = _staged_fixture(tmp_path)
    candidate_path = (
        fixture[1] / "ready" / fixture[2].candidate_id
    )
    artifact = candidate_path / "fields" / "samples.bin"
    artifact.write_bytes(b"tampered")
    pre, adapter, _ = _prevalidate(tmp_path, fixture)

    assert pre.status == CandidateState.FAILED.value
    assert pre.record.to_json()["stages"][0]["status"] == "failed"
    assert adapter.mount_requests == []


def test_validation_rejects_incomplete_declared_horizon(
    tmp_path: Path,
) -> None:
    fixture = _staged_fixture(tmp_path, approximate=True)
    adapter = FixtureValidationAdapter(
        ValidationBehavior(incomplete_steps=True)
    )
    pre, _, _ = _prevalidate(
        tmp_path,
        fixture,
        adapter=adapter,
    )

    assert pre.status == CandidateState.REJECTED.value
    assert pre.record.to_json()["stages"][2]["status"] == "failed"


def test_validation_refuses_residency_leak(
    tmp_path: Path,
) -> None:
    fixture = _staged_fixture(tmp_path)
    adapter = FixtureValidationAdapter(
        ValidationBehavior(leak_residency=True)
    )
    pre, _, _ = _prevalidate(
        tmp_path,
        fixture,
        adapter=adapter,
    )

    assert pre.status == CandidateState.FAILED.value
    assert adapter.closed_sessions == [("sanity", "reference", 0)]


def test_validation_evidence_detects_post_publication_tampering(
    tmp_path: Path,
) -> None:
    fixture = _staged_fixture(tmp_path)
    pre, _, _ = _prevalidate(tmp_path, fixture)
    (pre.evidence_path / "record.json").write_text("{}\n")

    with pytest.raises(ModelCompileError, match="failed integrity"):
        load_prebenchmark_evidence(
            tmp_path / "validation-workspace",
            pre.record.to_json()["prebenchmark_id"],
        )
