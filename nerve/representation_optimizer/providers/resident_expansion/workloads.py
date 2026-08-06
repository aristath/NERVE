from __future__ import annotations

from nerve.compilation import Json
from nerve.representation_optimizer.benchmarking.planning import (
    create_benchmark_workload,
)
from nerve.representation_optimizer.providers.codebook.artifacts import (
    fixture_reference,
)
from nerve.representation_optimizer.providers.resident_expansion.artifacts import (
    COMPONENT_FIXTURE_PATH,
    CONVERSATION_FIXTURE_PATH,
    MODEL_LIMITS_PATH,
    PRODUCT_CONVERSATION_FIXTURE_PATH,
    component_fixture,
    conversation_fixture,
    model_limits_fixture,
    product_conversation_fixture,
)
from nerve.representation_optimizer.providers.resident_expansion.contracts import (
    PROOF_VERIFIER_ID,
)
from nerve.representation_optimizer.providers.resident_expansion.discovery import (
    ResidentExpansionOpportunity,
)
from nerve.representation_optimizer.validation.contracts import (
    VALIDATION_COVERAGE_KINDS,
)
from nerve.representation_optimizer.validation.planning import (
    create_validation_check,
    create_validation_requirements,
)


def resident_expansion_benchmark_workloads(
    opportunity: ResidentExpansionOpportunity,
) -> tuple[Json, ...]:
    fixture = fixture_reference(
        COMPONENT_FIXTURE_PATH,
        _component_fixture(opportunity),
    )
    common = {
        "context_size": 0,
        "state_size": 0,
        "stream_count": 1,
        "boundary_mode": "local",
        "input_artifact": fixture,
        "initial_state_artifact": None,
        "randomness_algorithm": "deterministic_fixture_counter",
        "seeds": (1,),
        "deterministic_replay_required": True,
        "permit_sampling_variance": False,
        "permit_numerical_nondeterminism": False,
        "permit_speculative_schedule_variance": False,
        "useful_work_unit": "expert_projection_dispatches",
        "completion_condition": "all_dispatches_completed",
        "output_allowance": None,
        "output_allowance_basis": {"kind": "unlimited"},
        "sustained_window_count": 2,
    }
    workloads = []
    for node_id in opportunity.node_ids:
        for phase, width, units in (
            ("decode", 1, 2),
            ("prefill", 1, 2),
        ):
            workloads.append(
                create_benchmark_workload(
                    name=(f"exact resident expert {phase} {node_id}"),
                    execution_phase=phase,
                    activation_batch_width=width,
                    mount_mode="resident_reuse",
                    controls={
                        "execution": "ordinary",
                        "execution_scope": "node",
                        "phase": phase,
                        "component_id": opportunity.component_id,
                        "physical_node_id": node_id,
                    },
                    minimum_useful_work_units=units,
                    **common,
                ).to_json()
            )
    workloads.append(
        create_benchmark_workload(
            name="exact resident expert cold derivation lifecycle",
            execution_phase="decode",
            activation_batch_width=1,
            mount_mode="cold",
            controls={
                "execution": "ordinary",
                "execution_scope": "node",
                "phase": "decode",
                "component_id": opportunity.component_id,
                "physical_node_id": opportunity.node_ids[0],
            },
            minimum_useful_work_units=2,
            **common,
        ).to_json()
    )
    return tuple(workloads)


def resident_expansion_validation_requirements(
    *,
    candidate: Json,
    opportunity: ResidentExpansionOpportunity,
    speculative_draft_tokens: int,
) -> Json:
    component_ref = fixture_reference(
        COMPONENT_FIXTURE_PATH,
        _component_fixture(opportunity),
    )
    conversation_ref = fixture_reference(
        CONVERSATION_FIXTURE_PATH,
        conversation_fixture(),
    )
    product_ref = fixture_reference(
        PRODUCT_CONVERSATION_FIXTURE_PATH,
        product_conversation_fixture(),
    )
    limits = model_limits_fixture(opportunity.max_context_activations)
    limits_ref = fixture_reference(MODEL_LIMITS_PATH, limits)
    context_basis = {
        "kind": "declared_model_limit",
        "artifact": limits_ref,
        "json_pointer": "/max_context_tokens",
        "declared_limit": opportunity.max_context_activations,
    }
    output_basis = {
        "kind": "declared_model_limit",
        "artifact": limits_ref,
        "json_pointer": "/max_output_tokens",
        "declared_limit": 65_536,
    }
    checks = []
    for node_id in opportunity.node_ids:
        for phase in ("decode", "prefill"):
            checks.append(
                create_validation_check(
                    name=f"exact resident expert {phase} {node_id}",
                    stage="sanity",
                    kind="component_comparison",
                    product_performance=False,
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
                        "phase": phase,
                        "component_id": opportunity.component_id,
                        "physical_node_id": node_id,
                        "edge_cases": [
                            "zeros",
                            "finite_extrema",
                            "alternating_signs",
                            "first_experts",
                            "last_experts",
                            "deterministic_random",
                        ],
                    },
                    seeds=(1,),
                    step_unit="component_activations",
                    completion_condition="minimum_steps",
                    minimum_steps=2,
                    output_allowance=None,
                    output_allowance_basis={"kind": "unlimited"},
                    metrics=("bf16_bit_exact",),
                )
            )
    checks.extend(
        (
            _whole_model_check(
                name="exact resident expert fixed-token replay",
                stage="full_local",
                kind="teacher_forced",
                coverage=("teacher_forced_sequences", "multiple_fixed_seeds"),
                artifact=conversation_ref,
                controls={
                    "execution": "ordinary",
                    "execution_mode": "teacher_forced",
                    "enable_thinking": True,
                },
                seeds=(1, 2),
                metrics=("token_exact_match",),
            ),
            _whole_model_check(
                name="exact resident expert lifecycle continuity",
                stage="full_local",
                kind="lifecycle_operation",
                coverage=(
                    "interruption",
                    "snapshot",
                    "fork",
                    "rollback",
                    "resumption",
                ),
                artifact=conversation_ref,
                controls={
                    "execution": "ordinary",
                    "execution_mode": "lifecycle_teacher_forced",
                    "enable_thinking": True,
                },
                seeds=(1,),
                metrics=("token_exact_match",),
            ),
            _whole_model_check(
                name="exact resident expert graph duplication",
                stage="full_local",
                kind="graph_edit",
                coverage=("graph_edits",),
                artifact=conversation_ref,
                controls={
                    "execution": "ordinary",
                    "execution_mode": "teacher_forced",
                    "graph_operation": "duplicate",
                    "graph_target_component_id": opportunity.component_id,
                    "enable_thinking": True,
                },
                seeds=(1,),
                metrics=("graph_contract_preserved", "token_exact_match"),
            ),
            _whole_model_check(
                name="exact resident expert alternative placement",
                stage="full_local",
                kind="placement",
                coverage=("alternative_placements",),
                artifact=conversation_ref,
                controls={
                    "execution": "ordinary",
                    "execution_mode": "teacher_forced",
                    "enable_thinking": True,
                },
                seeds=(1,),
                metrics=("token_exact_match",),
                boundary_mode="cross_device",
            ),
            create_validation_check(
                name="exact resident expert reasoning and long stream",
                stage="whole_model",
                kind="reasoning_conversation",
                product_performance=False,
                coverage=(
                    "free_running_long_horizon",
                    "reasoning_enabled_conversations",
                    "long_context",
                    "long_output",
                ),
                execution_scope="whole_model",
                activation_batch_width=1,
                context_size=opportunity.max_context_activations,
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
                comparison={
                    "output_mode": "fixture_semantics",
                    "state_mode": "trajectory_local",
                },
                metrics=("semantic_consistency", "conversation_memory"),
            ),
            create_validation_check(
                name="exact resident expert product performance",
                stage="whole_model",
                kind="free_running",
                product_performance=True,
                coverage=("free_running_long_horizon",),
                execution_scope="whole_model",
                activation_batch_width=1,
                context_size=opportunity.max_context_activations,
                context_size_basis=context_basis,
                state_size=0,
                boundary_mode="local",
                input_artifact=product_ref,
                initial_state_artifact=None,
                controls={
                    "execution": "ordinary",
                    "execution_mode": "conversation",
                    "enable_thinking": True,
                    "max_output_tokens": 65_536,
                    "speculative_draft_tokens": speculative_draft_tokens,
                    "sampler": {"top_k": 1},
                },
                seeds=(1,),
                step_unit="component_activations",
                completion_condition="semantic_stop_or_allowance_per_turn",
                minimum_steps=None,
                output_allowance=65_536,
                output_allowance_basis=output_basis,
                metrics=("token_exact_match",),
            ),
        )
    )
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
            obligation: PROOF_VERIFIER_ID
            for obligation in candidate["behavioral_contract"]["proof_obligations"]
        },
        checks=tuple(checks),
        not_applicable_reasons=not_applicable,
    ).to_json()


def _component_fixture(opportunity: ResidentExpansionOpportunity) -> Json:
    return component_fixture(
        component_id=opportunity.component_id,
        node_ids=opportunity.node_ids,
        hidden_size=opportunity.hidden_size,
        intermediate_size=opportunity.intermediate_size,
        expert_count=opportunity.expert_count,
        experts_per_token=opportunity.experts_per_token,
    )


def _whole_model_check(
    *,
    name: str,
    stage: str,
    kind: str,
    coverage: tuple[str, ...],
    artifact: Json,
    controls: Json,
    seeds: tuple[int, ...],
    metrics: tuple[str, ...],
    boundary_mode: str = "local",
) -> Json:
    return create_validation_check(
        name=name,
        stage=stage,
        kind=kind,
        product_performance=False,
        coverage=coverage,
        execution_scope="whole_model",
        activation_batch_width=1,
        context_size=0,
        context_size_basis={"kind": "not_applicable"},
        state_size=0,
        boundary_mode=boundary_mode,
        input_artifact=artifact,
        initial_state_artifact=None,
        controls=controls,
        seeds=seeds,
        step_unit="component_activations",
        completion_condition="all_fixture_turns",
        minimum_steps=None,
        output_allowance=None,
        output_allowance_basis={"kind": "unlimited"},
        metrics=metrics,
    )


def _not_applicable_reason(coverage: str) -> str:
    reasons = {
        "state_transition_consistency": (
            "the alternative changes immutable parameter residency and owns no state"
        ),
        "route_recall": (
            "the alternative consumes the source router selection without changing it"
        ),
        "memory_recall": "the represented component owns no retrieval memory",
        "candidate_recall": "the represented component performs no candidate search",
        "confidence_calibration": "the exact alternative emits no confidence score",
        "correction_calibration": "the exact alternative has no correction path",
        "adversarial_counterexamples": (
            "exhaustive proof covers every finite source code and component checks "
            "cover boundary activations and selector extremes"
        ),
    }
    return reasons.get(
        coverage,
        f"the exact resident parameter alternative has no {coverage} responsibility",
    )
