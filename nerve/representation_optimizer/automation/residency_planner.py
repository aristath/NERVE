from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from nerve.compilation import Json, ModelCompileError, check_compile_cancelled

RUNTIME_RESIDENCY_PLANNER_REQUEST_SCHEMA = (
    "nerve.runtime_residency_planner_request.v2"
)
RUNTIME_RESIDENCY_PLANNER_RESPONSE_SCHEMA = (
    "nerve.runtime_residency_planner_response.v2"
)
RUNTIME_RESIDENCY_PLAN_SCHEMA = "nerve.vulkan_runtime_residency_plan.v2"
RESIDENCY_POLICIES = frozenset(("demand_retained", "eager"))
PARAMETER_RESIDENCY_FIELDS = frozenset(
    (
        "always_resident_bytes",
        "initial_dynamic_bytes",
        "current_resident_bytes",
        "maximum_addressable_bytes",
        "staging_headroom_bytes",
    )
)
WORKING_SET_FIELDS = frozenset(
    ("transient_state_bytes", "activation_headroom_bytes")
)
TRANSIENT_BREAKDOWN_FIELDS = frozenset(
    (
        "stream_state_bytes",
        "state_transaction_bytes",
        "stream_control_bytes",
        "speculative_decoder_state_bytes",
    )
)
ACTIVATION_BREAKDOWN_FIELDS = frozenset(
    (
        "activation_slot_bytes",
        "boundary_buffer_bytes",
        "edge_buffer_bytes",
        "output_transducer_workspace_bytes",
        "sampler_workspace_bytes",
        "feedback_workspace_bytes",
        "speculative_decoder_activation_bytes",
        "speculative_decoder_workspace_bytes",
    )
)


@dataclass(frozen=True)
class RuntimeResidencyPlanningCase:
    case_id: str
    default_device_id: str
    component_placement: dict[str, str]
    context_capacity_activations: int
    mount_speculative_decoders: bool
    residency_policy: str

    def to_json(self) -> Json:
        if not self.case_id or not self.default_device_id:
            raise ModelCompileError(
                "runtime residency planning identities must be nonempty"
            )
        if self.context_capacity_activations <= 0:
            raise ModelCompileError(
                "runtime residency context capacity must be positive"
            )
        if self.residency_policy not in RESIDENCY_POLICIES:
            raise ModelCompileError(
                f"unsupported runtime residency policy "
                f"{self.residency_policy!r}"
            )
        return {
            "case_id": self.case_id,
            "default_device_id": self.default_device_id,
            "component_placement": dict(
                sorted(self.component_placement.items())
            ),
            "context_capacity_activations": (
                self.context_capacity_activations
            ),
            "mount_speculative_decoders": self.mount_speculative_decoders,
            "residency_policy": self.residency_policy,
        }


def plan_runtime_residency_cases(
    *,
    command: tuple[str, ...],
    package_manifest: Path,
    cases: Iterable[RuntimeResidencyPlanningCase],
    cancel_requested: Callable[[], bool] | None,
) -> dict[str, Json]:
    cases = tuple(cases)
    case_ids = [case.case_id for case in cases]
    if not cases or len(case_ids) != len(set(case_ids)):
        raise ModelCompileError(
            "runtime residency planning requires unique nonempty cases"
        )
    request = {
        "schema": RUNTIME_RESIDENCY_PLANNER_REQUEST_SCHEMA,
        "package_manifest": str(package_manifest.resolve()),
        "cases": [case.to_json() for case in cases],
    }
    check_compile_cancelled(cancel_requested)
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as error:
        raise ModelCompileError(
            f"could not start runtime residency planner {command[0]!r}: {error}"
        ) from error
    payload = json.dumps(
        request,
        sort_keys=True,
        separators=(",", ":"),
    )
    while True:
        try:
            stdout, stderr = process.communicate(payload, timeout=0.1)
            break
        except subprocess.TimeoutExpired:
            payload = None
            try:
                check_compile_cancelled(cancel_requested)
            except BaseException:
                process.kill()
                process.communicate()
                raise
    check_compile_cancelled(cancel_requested)
    if process.returncode != 0:
        diagnostic = stderr.strip() or stdout.strip()
        raise ModelCompileError(
            "runtime residency planner failed"
            + (f": {diagnostic}" if diagnostic else "")
        )
    try:
        response = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise ModelCompileError(
            "runtime residency planner returned invalid JSON"
        ) from error
    if (
        not isinstance(response, dict)
        or set(response) != {"schema", "plans"}
        or response.get("schema")
        != RUNTIME_RESIDENCY_PLANNER_RESPONSE_SCHEMA
        or not isinstance(response.get("plans"), list)
    ):
        raise ModelCompileError(
            "runtime residency planner returned an invalid response contract"
        )
    plans: dict[str, Json] = {}
    for result in response["plans"]:
        if (
            not isinstance(result, dict)
            or set(result) != {"case_id", "plan"}
            or not isinstance(result.get("case_id"), str)
            or not isinstance(result.get("plan"), dict)
        ):
            raise ModelCompileError(
                "runtime residency planner returned a malformed case"
            )
        case_id = result["case_id"]
        if case_id in plans:
            raise ModelCompileError(
                f"runtime residency planner repeated case {case_id!r}"
            )
        plan = result["plan"]
        _validate_runtime_residency_plan(plan)
        plans[case_id] = plan
    if set(plans) != set(case_ids):
        raise ModelCompileError(
            "runtime residency planner did not return exactly the requested cases"
        )
    return plans


def _validate_runtime_residency_plan(plan: Json) -> None:
    required = {
        "schema",
        "package_id",
        "residency_policy",
        "context_capacity_activations",
        "speculative_decoders_mounted",
        "device_plans",
        "total_initial_device_resident_bytes",
        "total_current_resident_parameter_bytes",
        "total_maximum_addressable_parameter_bytes",
    }
    if (
        set(plan) != required
        or plan.get("schema") != RUNTIME_RESIDENCY_PLAN_SCHEMA
        or not isinstance(plan.get("package_id"), str)
        or not plan["package_id"]
        or plan.get("residency_policy") not in RESIDENCY_POLICIES
        or not _positive_int(plan.get("context_capacity_activations"))
        or not isinstance(plan.get("speculative_decoders_mounted"), bool)
        or not isinstance(plan.get("device_plans"), list)
        or not plan["device_plans"]
        or not _nonnegative_int(
            plan.get("total_initial_device_resident_bytes")
        )
        or not _nonnegative_int(
            plan.get("total_current_resident_parameter_bytes")
        )
        or not _nonnegative_int(
            plan.get("total_maximum_addressable_parameter_bytes")
        )
    ):
        raise ModelCompileError(
            "runtime residency planner returned a malformed plan"
        )
    device_ids: set[str] = set()
    computed_initial_total = 0
    computed_current_parameter_total = 0
    computed_maximum_parameter_total = 0
    for device in plan["device_plans"]:
        if (
            not isinstance(device, dict)
            or set(device)
            != {
                "device_id",
                "parameter_residency",
                "working_set",
                "breakdown",
                "initial_device_resident_bytes",
            }
            or not isinstance(device.get("device_id"), str)
            or not device["device_id"]
            or device["device_id"] in device_ids
            or not isinstance(device.get("parameter_residency"), dict)
            or not isinstance(device.get("working_set"), dict)
            or not isinstance(device.get("breakdown"), dict)
            or not _nonnegative_int(
                device.get("initial_device_resident_bytes")
            )
        ):
            raise ModelCompileError(
                "runtime residency planner returned a malformed device plan"
            )
        device_ids.add(device["device_id"])
        parameters = device["parameter_residency"]
        working_set = device["working_set"]
        breakdown = device["breakdown"]
        if (
            set(parameters) != PARAMETER_RESIDENCY_FIELDS
            or any(
                not _nonnegative_int(value)
                for value in parameters.values()
            )
            or set(working_set) != WORKING_SET_FIELDS
            or any(
                not _nonnegative_int(value)
                for value in working_set.values()
            )
            or set(breakdown)
            != TRANSIENT_BREAKDOWN_FIELDS | ACTIVATION_BREAKDOWN_FIELDS
            or any(
                not isinstance(name, str) or not _nonnegative_int(value)
                for name, value in breakdown.items()
            )
        ):
            raise ModelCompileError(
                "runtime residency planner returned an invalid byte breakdown"
            )
        if (
            parameters["current_resident_bytes"]
            != parameters["always_resident_bytes"]
            + parameters["initial_dynamic_bytes"]
            or parameters["maximum_addressable_bytes"]
            < parameters["current_resident_bytes"]
            or (
                plan["residency_policy"] == "demand_retained"
                and parameters["initial_dynamic_bytes"] != 0
            )
            or (
                plan["residency_policy"] == "eager"
                and parameters["current_resident_bytes"]
                != parameters["maximum_addressable_bytes"]
            )
        ):
            raise ModelCompileError(
                "runtime residency parameter accounting is inconsistent"
            )
        if working_set["transient_state_bytes"] != sum(
            breakdown[name] for name in TRANSIENT_BREAKDOWN_FIELDS
        ) or working_set["activation_headroom_bytes"] != sum(
            breakdown[name] for name in ACTIVATION_BREAKDOWN_FIELDS
        ):
            raise ModelCompileError(
                "runtime residency working set disagrees with its breakdown"
            )
        device_initial = (
            parameters["current_resident_bytes"]
            + parameters["staging_headroom_bytes"]
            + working_set["transient_state_bytes"]
            + working_set["activation_headroom_bytes"]
        )
        if device_initial != device["initial_device_resident_bytes"]:
            raise ModelCompileError(
                "runtime residency initial total disagrees with its categories"
            )
        computed_initial_total += device_initial
        computed_current_parameter_total += parameters[
            "current_resident_bytes"
        ]
        computed_maximum_parameter_total += parameters[
            "maximum_addressable_bytes"
        ]
    if (
        computed_initial_total
        != plan["total_initial_device_resident_bytes"]
        or computed_current_parameter_total
        != plan["total_current_resident_parameter_bytes"]
        or computed_maximum_parameter_total
        != plan["total_maximum_addressable_parameter_bytes"]
    ):
        raise ModelCompileError(
            "runtime residency total disagrees with device plans"
        )


def _positive_int(value: object) -> bool:
    return _nonnegative_int(value) and value > 0


def _nonnegative_int(value: object) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    )
