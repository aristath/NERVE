from __future__ import annotations

from copy import deepcopy

import pytest

from nerve.compilation import ModelCompileError
from nerve.representation_optimizer.automation.residency_planner import (
    RUNTIME_RESIDENCY_PLAN_SCHEMA,
    RuntimeResidencyPlanningCase,
    _validate_runtime_residency_plan,
)


def _plan() -> dict[str, object]:
    return {
        "schema": RUNTIME_RESIDENCY_PLAN_SCHEMA,
        "package_id": "sha256:" + "1" * 64,
        "residency_policy": "demand_retained",
        "context_capacity_activations": 131_072,
        "speculative_draft_tokens": 2,
        "device_plans": [
            {
                "device_id": "vulkan-uuid:" + "2" * 32,
                "parameter_residency": {
                    "always_resident_bytes": 100,
                    "initial_dynamic_bytes": 0,
                    "current_resident_bytes": 100,
                    "maximum_addressable_bytes": 10_000,
                    "staging_headroom_bytes": 64,
                },
                "resource_store": {
                    "address_table_device_bytes": 10,
                    "parameter_slot_table_device_bytes": 6,
                    "metadata_device_bytes": 16,
                    "transfer_staging_slot_count": 2,
                    "transfer_staging_slot_byte_capacity": 24,
                    "transfer_staging_device_bytes": 48,
                    "maximum_load_wave_group_count": 1,
                    "maximum_load_wave_payload_bytes": 24,
                    "maximum_dynamic_allocation_padding_bytes": 0,
                },
                "working_set": {
                    "transient_state_bytes": 3,
                    "activation_headroom_bytes": 4,
                },
                "breakdown": {
                    "stream_state_bytes": 1,
                    "state_transaction_bytes": 1,
                    "activation_slot_bytes": 1,
                    "boundary_buffer_bytes": 1,
                    "edge_buffer_bytes": 0,
                    "stream_control_bytes": 0,
                    "output_transducer_workspace_bytes": 1,
                    "sampler_workspace_bytes": 0,
                    "feedback_workspace_bytes": 0,
                    "speculative_decoder_state_bytes": 1,
                    "causal_verification_snapshot_bytes": 0,
                    "speculative_decoder_activation_bytes": 1,
                    "speculative_decoder_workspace_bytes": 0,
                },
                "initial_device_resident_bytes": 171,
            }
        ],
        "total_initial_device_resident_bytes": 171,
        "total_current_resident_parameter_bytes": 100,
        "total_maximum_addressable_parameter_bytes": 10_000,
    }


def test_accepts_maximum_address_space_larger_than_mount_requirement() -> None:
    _validate_runtime_residency_plan(_plan())


def test_planning_case_preserves_exact_speculative_window() -> None:
    case = RuntimeResidencyPlanningCase(
        case_id="draft-two",
        default_device_id="gpu0",
        component_placement={"block_00": "gpu0"},
        context_capacity_activations=128,
        speculative_draft_tokens=2,
        residency_policy="demand_retained",
    )

    assert case.to_json()["speculative_draft_tokens"] == 2


def test_planning_case_rejects_boolean_speculative_window() -> None:
    case = RuntimeResidencyPlanningCase(
        case_id="bad-window",
        default_device_id="gpu0",
        component_placement={"block_00": "gpu0"},
        context_capacity_activations=128,
        speculative_draft_tokens=True,
        residency_policy="demand_retained",
    )

    with pytest.raises(ModelCompileError, match="nonnegative integer"):
        case.to_json()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda plan: plan["device_plans"][0]["parameter_residency"].update(
                initial_dynamic_bytes=1
            ),
            "parameter accounting",
        ),
        (
            lambda plan: plan["device_plans"][0]["parameter_residency"].update(
                maximum_addressable_bytes=99
            ),
            "parameter accounting",
        ),
        (
            lambda plan: plan["device_plans"][0]["working_set"].update(
                transient_state_bytes=2
            ),
            "working set",
        ),
        (
            lambda plan: plan["device_plans"][0]["resource_store"].update(
                metadata_device_bytes=15
            ),
            "resource-store accounting",
        ),
        (
            lambda plan: plan["device_plans"][0].update(
                initial_device_resident_bytes=10_000
            ),
            "initial total",
        ),
        (
            lambda plan: plan.update(total_maximum_addressable_parameter_bytes=9_999),
            "total disagrees",
        ),
    ),
)
def test_rejects_inconsistent_residency_accounting(mutation, message: str) -> None:
    plan = deepcopy(_plan())
    mutation(plan)

    with pytest.raises(ModelCompileError, match=message):
        _validate_runtime_residency_plan(plan)
