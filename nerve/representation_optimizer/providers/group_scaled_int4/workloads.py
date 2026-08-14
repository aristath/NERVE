from __future__ import annotations

from nerve.compilation import Json
from nerve.representation_optimizer.benchmarking.planning import (
    create_benchmark_workload,
)
from nerve.representation_optimizer.providers.codebook.artifacts import (
    fixture_reference,
)
from nerve.representation_optimizer.providers.group_scaled_int4.artifacts import (
    COMPONENT_FIXTURE_PATH,
    CONVERSATION_FIXTURE_PATH,
    MODEL_LIMITS_PATH,
    PRODUCT_CONVERSATION_FIXTURE_PATH,
    component_fixture,
    conversation_fixture,
    model_limits_fixture,
    product_conversation_fixture,
)
from nerve.representation_optimizer.providers.group_scaled_int4.discovery import (
    GroupScaledInt4Opportunity,
)
from nerve.representation_optimizer.validation.contracts import (
    VALIDATION_COVERAGE_KINDS,
)
from nerve.representation_optimizer.validation.planning import (
    create_behavioral_error_contract,
    create_validation_check,
    create_validation_requirements,
)


def group_scaled_int4_benchmark_workloads(
    opportunity: GroupScaledInt4Opportunity,
) -> tuple[Json, ...]:
    fixture = fixture_reference(
        COMPONENT_FIXTURE_PATH,
        _component_fixture(opportunity),
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
        "useful_work_unit": "linear_region_dispatches",
        "minimum_useful_work_units": 8,
        "completion_condition": "all_dispatches_completed",
        "output_allowance": None,
        "output_allowance_basis": {"kind": "unlimited"},
        "sustained_window_count": 2,
    }
    return (
        create_benchmark_workload(
            name="group-scaled INT4 linear decode",
            execution_phase="decode",
            activation_batch_width=1,
            controls={
                "execution": "ordinary",
                "phase": "decode",
                "component_id": opportunity.component_id,
                "physical_node_id": opportunity.node_id,
            },
            **common,
        ).to_json(),
        create_benchmark_workload(
            name="group-scaled INT4 linear prefill",
            execution_phase="prefill",
            activation_batch_width=4,
            controls={
                "execution": "ordinary",
                "phase": "prefill",
                "component_id": opportunity.component_id,
                "physical_node_id": opportunity.node_id,
            },
            **common,
        ).to_json(),
    )


def group_scaled_int4_error_contract(group_size: int) -> Json:
    return create_behavioral_error_contract(
        validity_predicates={
            "input_domain": "finite BF16 vectors",
            "parameter_encoding": (
                f"signed INT4 groups of {group_size} input columns with BF16 scales"
            ),
            "target_process": "Vulkan subgroup vector reduction over packed I32 words",
        },
        metric_limits={
            "conversation_memory": (
                0.0,
                "boolean_failure",
                ("memory_recall",),
            ),
            "normalized_rms_logit_error": (
                0.12,
                "relative_rms",
                ("component_output_error", "distribution_divergence"),
            ),
            "semantic_consistency": (
                0.0,
                "boolean_failure",
                (
                    "free_running_long_horizon",
                    "long_context",
                    "long_output",
                    "reasoning_enabled_conversations",
                ),
            ),
            "token_exact_match": (
                0.0,
                "boolean_failure",
                (
                    "alternative_placements",
                    "graph_edits",
                    "multiple_fixed_seeds",
                    "teacher_forced_sequences",
                ),
            ),
            "top_1_mismatch_rate": (
                0.0,
                "fraction",
                ("rank_stability", "route_recall"),
            ),
            "top_32_mismatch_rate": (
                0.25,
                "fraction",
                ("top_k_overlap",),
            ),
        },
        correction_mode="reject",
        correction_trigger_metrics=(
            "conversation_memory",
            "normalized_rms_logit_error",
            "semantic_consistency",
            "token_exact_match",
            "top_1_mismatch_rate",
            "top_32_mismatch_rate",
        ),
        correction_action="retain the source linear implementation",
    ).to_json()


def group_scaled_int4_validation_requirements(
    *,
    candidate: Json,
    opportunity: GroupScaledInt4Opportunity,
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
    product_conversation_ref = fixture_reference(
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
    teacher_forced_controls = {
        "execution": "ordinary",
        "execution_mode": "teacher_forced",
        "enable_thinking": True,
    }
    checks = (
        create_validation_check(
            name="group-scaled INT4 component numeric and route sanity",
            stage="sanity",
            kind="component_comparison",
            product_performance=False,
            coverage=(
                "component_output_error",
                "distribution_divergence",
                "rank_stability",
                "route_recall",
                "top_k_overlap",
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
                "physical_node_id": opportunity.node_id,
                "capture_output_values": True,
            },
            seeds=(1,),
            step_unit="component_activations",
            completion_condition="minimum_steps",
            minimum_steps=1,
            output_allowance=None,
            output_allowance_basis={"kind": "unlimited"},
            metrics=(
                "normalized_rms_logit_error",
                "top_1_mismatch_rate",
                "top_32_mismatch_rate",
            ),
        ),
        create_validation_check(
            name="group-scaled INT4 teacher-forced replay",
            stage="full_local",
            kind="teacher_forced",
            product_performance=False,
            coverage=("multiple_fixed_seeds", "teacher_forced_sequences"),
            execution_scope="whole_model",
            activation_batch_width=1,
            context_size=0,
            context_size_basis={"kind": "not_applicable"},
            state_size=0,
            boundary_mode="local",
            input_artifact=conversation_ref,
            initial_state_artifact=None,
            controls=teacher_forced_controls,
            seeds=(1, 2),
            step_unit="component_activations",
            completion_condition="all_fixture_turns",
            minimum_steps=None,
            output_allowance=None,
            output_allowance_basis={"kind": "unlimited"},
            metrics=("token_exact_match",),
        ),
        create_validation_check(
            name="group-scaled INT4 graph-edit compatibility",
            stage="full_local",
            kind="graph_edit",
            product_performance=False,
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
                **teacher_forced_controls,
                "graph_operation": "restore",
                "graph_target_component_id": opportunity.component_id,
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
            name="group-scaled INT4 placement compatibility",
            stage="full_local",
            kind="placement",
            product_performance=False,
            coverage=("alternative_placements",),
            execution_scope="whole_model",
            activation_batch_width=1,
            context_size=0,
            context_size_basis={"kind": "not_applicable"},
            state_size=0,
            boundary_mode="cross_device",
            input_artifact=conversation_ref,
            initial_state_artifact=None,
            controls=teacher_forced_controls,
            seeds=(1,),
            step_unit="component_activations",
            completion_condition="all_fixture_turns",
            minimum_steps=None,
            output_allowance=None,
            output_allowance_basis={"kind": "unlimited"},
            metrics=("token_exact_match",),
        ),
        create_validation_check(
            name="group-scaled INT4 reasoning and long-horizon conversation",
            stage="whole_model",
            kind="reasoning_conversation",
            product_performance=False,
            coverage=(
                "free_running_long_horizon",
                "long_context",
                "long_output",
                "reasoning_enabled_conversations",
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
            metrics=("conversation_memory", "semantic_consistency"),
        ),
        create_validation_check(
            name="group-scaled INT4 product conversation performance",
            stage="whole_model",
            kind="free_running",
            product_performance=True,
            coverage=("free_running_long_horizon", "memory_recall"),
            execution_scope="whole_model",
            activation_batch_width=1,
            context_size=opportunity.max_context_activations,
            context_size_basis=context_basis,
            state_size=0,
            boundary_mode="local",
            input_artifact=product_conversation_ref,
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
            comparison={
                "output_mode": "fixture_semantics",
                "state_mode": "trajectory_local",
            },
            metrics=("conversation_memory", "semantic_consistency"),
        ),
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
        proof_verifiers={},
        checks=checks,
        not_applicable_reasons=not_applicable,
    ).to_json()


def _component_fixture(opportunity: GroupScaledInt4Opportunity) -> Json:
    return component_fixture(
        component_id=opportunity.component_id,
        node_id=opportunity.node_id,
        input_features=opportunity.input_features,
        output_features=opportunity.output_features,
    )


def _not_applicable_reason(coverage: str) -> str:
    reasons = {
        "state_transition_consistency": "the stateless linear region owns no state",
        "candidate_recall": "the linear region performs no approximate search",
        "confidence_calibration": (
            "candidate acceptance uses measured error limits, not runtime confidence"
        ),
        "correction_calibration": (
            "correction is deterministic pre-promotion rejection"
        ),
        "interruption": "the stateless linear region adds no lifecycle state",
        "snapshot": "the stateless linear region adds no snapshot state",
        "fork": "the stateless linear region adds no forked state",
        "rollback": "the stateless linear region adds no rollback state",
        "resumption": "the stateless linear region adds no resumed state",
        "adversarial_counterexamples": (
            "no stored counterexample has been discovered for this candidate"
        ),
    }
    return reasons.get(
        coverage,
        f"the linear parameter region has no {coverage} responsibility",
    )


__all__ = [
    "group_scaled_int4_benchmark_workloads",
    "group_scaled_int4_error_contract",
    "group_scaled_int4_validation_requirements",
]
