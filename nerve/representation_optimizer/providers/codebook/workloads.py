from __future__ import annotations

from nerve.compilation import Json
from nerve.representation_optimizer.benchmarking.planning import (
    create_benchmark_workload,
)
from nerve.representation_optimizer.providers.codebook.artifacts import (
    COMPONENT_FIXTURE_PATH,
    CONVERSATION_FIXTURE_PATH,
    MODEL_LIMITS_PATH,
    component_fixture,
    conversation_fixture,
    fixture_reference,
    model_limits_fixture,
)
from nerve.representation_optimizer.providers.codebook.discovery import (
    HeadNormCodebookOpportunity,
)
from nerve.representation_optimizer.providers.codebook.member_paths import (
    member_path,
)
from nerve.representation_optimizer.validation.contracts import (
    VALIDATION_COVERAGE_KINDS,
)
from nerve.representation_optimizer.validation.planning import (
    create_validation_check,
    create_validation_requirements,
)


def head_norm_benchmark_workloads(
    opportunity: HeadNormCodebookOpportunity,
    *,
    representation_name: str,
    artifact_scope_id: str | None = None,
    qualified_component_ids: tuple[str, ...] = (),
    execution_phases: tuple[str, ...] = ("decode", "prefill"),
) -> tuple[Json, ...]:
    _validate_execution_phases(execution_phases)
    fixture_path = (
        COMPONENT_FIXTURE_PATH
        if artifact_scope_id is None
        else member_path(artifact_scope_id, COMPONENT_FIXTURE_PATH)
    )
    fixture = fixture_reference(
        fixture_path,
        component_fixture(opportunity),
    )
    common = {
        "context_size": 0,
        "state_size": 0,
        "stream_count": 1,
        "mount_mode": "resident_reuse",
        "boundary_mode": "local",
        "input_artifact": fixture,
        "initial_state_artifact": None,
        "randomness_algorithm": "deterministic_fixture_counter",
        "seeds": (1,),
        "deterministic_replay_required": True,
        "permit_sampling_variance": False,
        "permit_numerical_nondeterminism": False,
        "permit_speculative_schedule_variance": False,
        "useful_work_unit": "fused_head_norm_rope_dispatches",
        "completion_condition": "all_dispatches_completed",
        "output_allowance": None,
        "output_allowance_basis": {"kind": "unlimited"},
        "sustained_window_count": 8,
    }
    workloads = (
        create_benchmark_workload(
            name=(
                f"exact {representation_name} fused head "
                "normalization decode"
            ),
            execution_phase="decode",
            activation_batch_width=1,
            controls={
                "execution": "ordinary",
                "phase": "decode",
                "component_id": opportunity.component_id,
                "physical_node_id": opportunity.physical_node_id,
                "qualified_component_ids": list(qualified_component_ids),
            },
            minimum_useful_work_units=8_192,
            **common,
        ).to_json(),
        create_benchmark_workload(
            name=(
                f"exact {representation_name} fused head "
                "normalization prefill"
            ),
            execution_phase="prefill",
            activation_batch_width=64,
            controls={
                "execution": "ordinary",
                "phase": "prefill",
                "component_id": opportunity.component_id,
                "physical_node_id": opportunity.physical_node_id,
                "qualified_component_ids": list(qualified_component_ids),
            },
            minimum_useful_work_units=4_096,
            **common,
        ).to_json(),
    )
    return tuple(
        workload
        for workload in workloads
        if workload["controls"]["phase"] in execution_phases
    )


def head_norm_validation_requirements(
    *,
    candidate: Json,
    opportunity: HeadNormCodebookOpportunity,
    max_context_activations: int,
    proof_verifier_id: str,
    representation_name: str,
    artifact_scope_id: str | None = None,
    execution_phases: tuple[str, ...] = ("decode", "prefill"),
    speculative_draft_tokens: int = 0,
) -> Json:
    _validate_execution_phases(execution_phases)
    def path(value: str) -> str:
        return (
            value
            if artifact_scope_id is None
            else member_path(artifact_scope_id, value)
        )

    component_ref = fixture_reference(
        path(COMPONENT_FIXTURE_PATH),
        component_fixture(opportunity),
    )
    conversation_ref = fixture_reference(
        path(CONVERSATION_FIXTURE_PATH),
        conversation_fixture(),
    )
    limits = model_limits_fixture(max_context_activations)
    limits_ref = fixture_reference(path(MODEL_LIMITS_PATH), limits)
    context_basis = {
        "kind": "declared_model_limit",
        "artifact": limits_ref,
        "json_pointer": "/max_context_tokens",
        "declared_limit": max_context_activations,
    }
    output_basis = {
        "kind": "declared_model_limit",
        "artifact": limits_ref,
        "json_pointer": "/max_output_tokens",
        "declared_limit": 65_536,
    }
    checks = (
        create_validation_check(
            name=(
                f"exact {representation_name} exhaustive decode "
                "component sanity"
            ),
            stage="sanity",
            kind="component_comparison",
            coverage=(
                "component_output_error",
                "distribution_divergence",
                "top_k_overlap",
                "rank_stability",
            ),
            execution_scope="component",
            activation_batch_width=1,
            context_size=0,
            context_size_basis={"kind": "not_applicable"},
            state_size=0,
            boundary_mode="local",
            input_artifact=component_ref,
            initial_state_artifact=None,
            controls={
                "phase": "decode",
                "component_id": opportunity.component_id,
                "physical_node_id": opportunity.physical_node_id,
                "edge_cases": [
                    "zeros",
                    "subnormals",
                    "finite_extrema",
                    "alternating_signs",
                    "deterministic_random",
                ],
            },
            seeds=(1,),
            step_unit="component_activations",
            completion_condition="minimum_steps",
            minimum_steps=4_096,
            output_allowance=None,
            output_allowance_basis={"kind": "unlimited"},
            metrics=("bf16_bit_exact",),
        ),
        create_validation_check(
            name=(
                f"exact {representation_name} exhaustive prefill "
                "component sanity"
            ),
            stage="sanity",
            kind="component_comparison",
            coverage=(
                "component_output_error",
                "distribution_divergence",
                "top_k_overlap",
                "rank_stability",
            ),
            execution_scope="component",
            activation_batch_width=64,
            context_size=0,
            context_size_basis={"kind": "not_applicable"},
            state_size=0,
            boundary_mode="local",
            input_artifact=component_ref,
            initial_state_artifact=None,
            controls={
                "phase": "prefill",
                "component_id": opportunity.component_id,
                "physical_node_id": opportunity.physical_node_id,
                "edge_cases": [
                    "zeros",
                    "subnormals",
                    "finite_extrema",
                    "alternating_signs",
                    "deterministic_random",
                ],
            },
            seeds=(1,),
            step_unit="component_activations",
            completion_condition="minimum_steps",
            minimum_steps=4_096,
            output_allowance=None,
            output_allowance_basis={"kind": "unlimited"},
            metrics=("bf16_bit_exact",),
        ),
        create_validation_check(
            name=(
                f"exact {representation_name} fixed-token model "
                "state replay"
            ),
            stage="full_local",
            kind="teacher_forced",
            coverage=(
                "teacher_forced_sequences",
                "multiple_fixed_seeds",
            ),
            execution_scope="whole_model",
            activation_batch_width=1,
            context_size=0,
            context_size_basis={"kind": "not_applicable"},
            state_size=0,
            boundary_mode="local",
            input_artifact=conversation_ref,
            initial_state_artifact=None,
            controls={
                "execution": "ordinary",
                "execution_mode": "teacher_forced",
                "enable_thinking": True,
            },
            seeds=(1, 2),
            step_unit="component_activations",
            completion_condition="all_fixture_turns",
            minimum_steps=None,
            output_allowance=None,
            output_allowance_basis={"kind": "unlimited"},
            metrics=("token_exact_match",),
        ),
        create_validation_check(
            name=f"exact {representation_name} lifecycle continuity",
            stage="full_local",
            kind="lifecycle_operation",
            coverage=(
                "interruption",
                "snapshot",
                "fork",
                "rollback",
                "resumption",
            ),
            execution_scope="whole_model",
            activation_batch_width=1,
            context_size=0,
            context_size_basis={"kind": "not_applicable"},
            state_size=0,
            boundary_mode="local",
            input_artifact=conversation_ref,
            initial_state_artifact=None,
            controls={
                "execution": "ordinary",
                "execution_mode": "lifecycle_teacher_forced",
                "enable_thinking": True,
            },
            seeds=(1,),
            step_unit="component_activations",
            completion_condition="all_fixture_turns",
            minimum_steps=None,
            output_allowance=None,
            output_allowance_basis={"kind": "unlimited"},
            metrics=("token_exact_match",),
        ),
        *(
            create_validation_check(
                name=(
                    f"exact {representation_name} graph edit {operation}"
                ),
                stage="full_local",
                kind="graph_edit",
                coverage=("graph_edits",),
                execution_scope="whole_model",
                activation_batch_width=1,
                context_size=0,
                context_size_basis={"kind": "not_applicable"},
                state_size=0,
                boundary_mode="local",
                input_artifact=conversation_ref,
                initial_state_artifact=None,
                controls={
                    "execution": "ordinary",
                    "execution_mode": "teacher_forced",
                    "graph_operation": operation,
                    "graph_target_component_id": (opportunity.component_id),
                    "enable_thinking": True,
                },
                seeds=(1,),
                step_unit="component_activations",
                completion_condition="all_fixture_turns",
                minimum_steps=None,
                output_allowance=None,
                output_allowance_basis={"kind": "unlimited"},
                metrics=(
                    "graph_contract_preserved",
                    "token_exact_match",
                ),
            )
            for operation in (
                "duplicate",
                "bypass",
                "rewire",
                "restore",
            )
        ),
        create_validation_check(
            name=f"exact {representation_name} alternative placement",
            stage="full_local",
            kind="placement",
            coverage=("alternative_placements",),
            execution_scope="whole_model",
            activation_batch_width=1,
            context_size=0,
            context_size_basis={"kind": "not_applicable"},
            state_size=0,
            boundary_mode="cross_device",
            input_artifact=conversation_ref,
            initial_state_artifact=None,
            controls={
                "execution": "ordinary",
                "execution_mode": "teacher_forced",
                "enable_thinking": True,
            },
            seeds=(1,),
            step_unit="component_activations",
            completion_condition="all_fixture_turns",
            minimum_steps=None,
            output_allowance=None,
            output_allowance_basis={"kind": "unlimited"},
            metrics=("token_exact_match",),
        ),
        create_validation_check(
            name=(
                f"exact {representation_name} adversarial decode "
                "component inputs"
            ),
            stage="full_local",
            kind="counterexample",
            coverage=("adversarial_counterexamples",),
            execution_scope="component",
            activation_batch_width=1,
            context_size=0,
            context_size_basis={"kind": "not_applicable"},
            state_size=0,
            boundary_mode="local",
            input_artifact=component_ref,
            initial_state_artifact=None,
            controls={
                "phase": "decode",
                "component_id": opportunity.component_id,
                "physical_node_id": opportunity.physical_node_id,
            },
            seeds=(0, 1, 0xFFFF_FFFF),
            step_unit="component_activations",
            completion_condition="minimum_steps",
            minimum_steps=65_536,
            output_allowance=None,
            output_allowance_basis={"kind": "unlimited"},
            metrics=("bf16_bit_exact",),
        ),
        create_validation_check(
            name=(
                f"exact {representation_name} adversarial prefill "
                "component inputs"
            ),
            stage="full_local",
            kind="counterexample",
            coverage=("adversarial_counterexamples",),
            execution_scope="component",
            activation_batch_width=64,
            context_size=0,
            context_size_basis={"kind": "not_applicable"},
            state_size=0,
            boundary_mode="local",
            input_artifact=component_ref,
            initial_state_artifact=None,
            controls={
                "phase": "prefill",
                "component_id": opportunity.component_id,
                "physical_node_id": opportunity.physical_node_id,
            },
            seeds=(0, 1, 0xFFFF_FFFF),
            step_unit="component_activations",
            completion_condition="minimum_steps",
            minimum_steps=65_536,
            output_allowance=None,
            output_allowance_basis={"kind": "unlimited"},
            metrics=("bf16_bit_exact",),
        ),
        create_validation_check(
            name=(
                f"exact {representation_name} reasoning conversation "
                "and long horizon"
            ),
            stage="whole_model",
            kind="reasoning_conversation",
            coverage=(
                "free_running_long_horizon",
                "reasoning_enabled_conversations",
                "long_context",
                "long_output",
            ),
            execution_scope="whole_model",
            activation_batch_width=1,
            context_size=max_context_activations,
            context_size_basis=context_basis,
            state_size=0,
            boundary_mode="local",
            input_artifact=conversation_ref,
            initial_state_artifact=None,
            controls={
                "execution": "ordinary",
                "execution_mode": "conversation",
                "enable_thinking": True,
                "max_output_tokens": 65_536,
                "speculative_draft_tokens": speculative_draft_tokens,
            },
            seeds=(1,),
            step_unit="component_activations",
            completion_condition="semantic_stop_or_allowance_per_turn",
            minimum_steps=None,
            output_allowance=65_536,
            output_allowance_basis=output_basis,
            metrics=(
                "token_exact_match",
                "semantic_consistency",
                "conversation_memory",
            ),
        ),
    )
    covered = {coverage for check in checks for coverage in check["coverage"]}
    not_applicable = {
        coverage: _not_applicable_reason(coverage)
        for coverage in VALIDATION_COVERAGE_KINDS
        if coverage not in covered
    }
    checks = tuple(
        check
        for check in checks
        if check["regime"]["execution_scope"] != "component"
        or check["controls"]["phase"] in execution_phases
    )
    return create_validation_requirements(
        candidate_id=candidate["candidate_id"],
        source_contract_digests=candidate["source_contract_digests"],
        proof_verifiers={
            obligation: proof_verifier_id
            for obligation in candidate["behavioral_contract"]["proof_obligations"]
        },
        checks=checks,
        not_applicable_reasons=not_applicable,
    ).to_json()


def bundled_head_norm_validation_requirements(
    *,
    candidate: Json,
    opportunities: tuple[HeadNormCodebookOpportunity, ...],
    max_context_activations: int,
    proof_verifier_id: str,
    representation_name: str,
    execution_phases: tuple[str, ...] = ("decode", "prefill"),
    speculative_draft_tokens: int = 0,
) -> Json:
    """Validate every member locally and the mounted set globally once."""

    if not opportunities:
        raise ValueError("bundled validation requires at least one opportunity")
    documents = [
        head_norm_validation_requirements(
            candidate={
                **candidate,
                "source_contract_digests": [opportunity.source_contract_digest],
            },
            opportunity=opportunity,
            max_context_activations=max_context_activations,
            proof_verifier_id=proof_verifier_id,
            representation_name=(
                f"{representation_name} member {opportunity.component_id}"
            ),
            artifact_scope_id=opportunity.scope_id,
            execution_phases=execution_phases,
            speculative_draft_tokens=speculative_draft_tokens,
        )
        for opportunity in opportunities
    ]
    checks = []
    for index, document in enumerate(documents):
        for check in document["checks"]:
            stage = check["stage"]
            execution_scope = check["regime"]["execution_scope"]
            if index == 0 or (
                stage != "sanity" and execution_scope == "component"
            ):
                checks.append(check)
    checks.sort(key=lambda item: item["check_id"])
    covered = {coverage for check in checks for coverage in check["coverage"]}
    not_applicable = {
        coverage: _not_applicable_reason(coverage)
        for coverage in VALIDATION_COVERAGE_KINDS
        if coverage not in covered
    }
    return create_validation_requirements(
        candidate_id=candidate["candidate_id"],
        source_contract_digests=candidate["source_contract_digests"],
        proof_verifiers={
            obligation: proof_verifier_id
            for obligation in candidate["behavioral_contract"]["proof_obligations"]
        },
        checks=checks,
        not_applicable_reasons=not_applicable,
    ).to_json()


def _validate_execution_phases(execution_phases: tuple[str, ...]) -> None:
    if (
        not execution_phases
        or execution_phases != tuple(sorted(set(execution_phases)))
        or any(phase not in {"decode", "prefill"} for phase in execution_phases)
    ):
        raise ValueError(
            "head-normalization execution phases must be a sorted, unique "
            "subset of decode and prefill"
        )


def _not_applicable_reason(coverage: str) -> str:
    reasons = {
        "state_transition_consistency": (
            "the representation changes immutable normalization parameters and "
            "does not read or write transient state"
        ),
        "route_recall": "the represented component performs no routing",
        "memory_recall": "the represented component owns no retrieval memory",
        "candidate_recall": "the represented component performs no candidate search",
        "confidence_calibration": (
            "the exact representation emits no probabilistic confidence"
        ),
        "correction_calibration": (
            "the exact representation has no correction or fallback path"
        ),
    }
    return reasons.get(
        coverage,
        f"the exact head-normalization representation has no {coverage} responsibility",
    )
