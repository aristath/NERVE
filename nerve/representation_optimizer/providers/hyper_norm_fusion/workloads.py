from __future__ import annotations

from nerve.compilation import Json
from nerve.representation_optimizer.benchmarking.planning import (
    create_benchmark_workload,
)
from nerve.representation_optimizer.providers.codebook.artifacts import (
    fixture_reference,
)
from nerve.representation_optimizer.providers.hyper_norm_fusion.artifacts import (
    COMPONENT_FIXTURE_PATH,
    CONVERSATION_FIXTURE_PATH,
    MODEL_LIMITS_PATH,
    PRODUCT_CONVERSATION_FIXTURE_PATH,
    component_fixture,
    conversation_fixture,
    model_limits_fixture,
    product_conversation_fixture,
)
from nerve.representation_optimizer.providers.hyper_norm_fusion.contracts import (
    PROOF_VERIFIER_ID,
)
from nerve.representation_optimizer.providers.hyper_norm_fusion.discovery import (
    HyperNormFusionOpportunity,
)
from nerve.representation_optimizer.validation.contracts import (
    VALIDATION_COVERAGE_KINDS,
)
from nerve.representation_optimizer.validation.planning import (
    create_validation_check,
    create_validation_requirements,
)


def hyper_norm_benchmark_workloads(
    opportunity: HyperNormFusionOpportunity,
) -> tuple[Json, ...]:
    fixture = fixture_reference(
        COMPONENT_FIXTURE_PATH,
        _component_fixture(opportunity),
    )
    common = {
        "execution_phase": "component",
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
        "useful_work_unit": "complete_component_transactions",
        "minimum_useful_work_units": 2,
        "completion_condition": "all_dispatches_completed",
        "output_allowance": None,
        "output_allowance_basis": {"kind": "unlimited"},
        "sustained_window_count": 2,
    }
    return tuple(
        create_benchmark_workload(
            name=f"exact fused hyper/RMS {phase}",
            activation_batch_width=width,
            controls={
                "execution": "ordinary",
                "phase": phase,
                "component_id": opportunity.component_id,
                "physical_node_id": opportunity.terminal_node_id,
                "physical_execution_scope": "component",
            },
            **common,
        ).to_json()
        for phase, width in (("decode", 1), ("prefill", 4))
    )


def hyper_norm_validation_requirements(
    *,
    candidate: Json,
    opportunity: HyperNormFusionOpportunity,
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
    checks = [
        create_validation_check(
            name=f"exact fused hyper/RMS {phase}",
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
            activation_batch_width=width,
            context_size=0,
            context_size_basis={"kind": "not_applicable"},
            state_size=0,
            boundary_mode="local",
            input_artifact=component_ref,
            initial_state_artifact=None,
            controls={
                "phase": phase,
                "component_id": opportunity.component_id,
                "physical_node_id": opportunity.terminal_node_id,
                "physical_execution_scope": "component",
            },
            seeds=(1,),
            step_unit="component_activations",
            completion_condition="minimum_steps",
            minimum_steps=2,
            output_allowance=None,
            output_allowance_basis={"kind": "unlimited"},
            metrics=("bf16_bit_exact",),
        )
        for phase, width in (("decode", 1), ("prefill", 4))
    ]
    checks.extend(
        (
            _whole_model_check(
                name="fused hyper/RMS fixed-token replay",
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
                name="fused hyper/RMS lifecycle continuity",
                stage="full_local",
                kind="lifecycle_operation",
                coverage=("interruption", "snapshot", "fork", "rollback", "resumption"),
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
                name="fused hyper/RMS graph duplication",
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
                name="fused hyper/RMS alternative placement",
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
                name="fused hyper/RMS reasoning and long stream",
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
                context_size_basis={
                    "kind": "declared_model_limit",
                    "artifact": limits_ref,
                    "json_pointer": "/max_context_tokens",
                    "declared_limit": opportunity.max_context_activations,
                },
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
                output_allowance_basis={
                    "kind": "declared_model_limit",
                    "artifact": limits_ref,
                    "json_pointer": "/max_output_tokens",
                    "declared_limit": 65_536,
                },
                comparison={
                    "output_mode": "fixture_semantics",
                    "state_mode": "trajectory_local",
                },
                metrics=("conversation_memory", "semantic_consistency"),
            ),
            create_validation_check(
                name="fused hyper/RMS product conversation performance",
                stage="whole_model",
                kind="free_running",
                product_performance=True,
                coverage=("free_running_long_horizon", "memory_recall"),
                execution_scope="whole_model",
                activation_batch_width=1,
                context_size=opportunity.max_context_activations,
                context_size_basis={
                    "kind": "declared_model_limit",
                    "artifact": limits_ref,
                    "json_pointer": "/max_context_tokens",
                    "declared_limit": opportunity.max_context_activations,
                },
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
                output_allowance_basis={
                    "kind": "declared_model_limit",
                    "artifact": limits_ref,
                    "json_pointer": "/max_output_tokens",
                    "declared_limit": 65_536,
                },
                comparison={
                    "output_mode": "fixture_semantics",
                    "state_mode": "trajectory_local",
                },
                metrics=("conversation_memory", "semantic_consistency"),
            ),
        )
    )
    covered = {item for check in checks for item in check["coverage"]}
    return create_validation_requirements(
        candidate_id=candidate["candidate_id"],
        source_contract_digests=candidate["source_contract_digests"],
        proof_verifiers={
            obligation: PROOF_VERIFIER_ID
            for obligation in candidate["behavioral_contract"]["proof_obligations"]
        },
        checks=tuple(checks),
        not_applicable_reasons={
            coverage: (
                "the exact local execution fusion does not own this semantic responsibility"
            )
            for coverage in VALIDATION_COVERAGE_KINDS
            if coverage not in covered
        },
    ).to_json()


def _component_fixture(opportunity: HyperNormFusionOpportunity) -> Json:
    return component_fixture(
        component_id=opportunity.component_id,
        terminal_node_id=opportunity.terminal_node_id,
        hidden_size=opportunity.hidden_size,
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
