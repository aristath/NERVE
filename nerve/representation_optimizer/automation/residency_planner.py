from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from nerve.compilation import Json, ModelCompileError, check_compile_cancelled

RUNTIME_RESIDENCY_PLANNER_REQUEST_SCHEMA = (
    "nerve.runtime_residency_planner_request.v1"
)
RUNTIME_RESIDENCY_PLANNER_RESPONSE_SCHEMA = (
    "nerve.runtime_residency_planner_response.v1"
)
RUNTIME_RESIDENCY_PLAN_SCHEMA = "nerve.vulkan_runtime_residency_plan.v1"


@dataclass(frozen=True)
class RuntimeResidencyPlanningCase:
    case_id: str
    default_device_id: str
    component_placement: dict[str, str]
    context_capacity_activations: int
    mount_speculative_decoders: bool

    def to_json(self) -> Json:
        if not self.case_id or not self.default_device_id:
            raise ModelCompileError(
                "runtime residency planning identities must be nonempty"
            )
        if self.context_capacity_activations <= 0:
            raise ModelCompileError(
                "runtime residency context capacity must be positive"
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
        "context_capacity_activations",
        "speculative_decoders_mounted",
        "device_plans",
        "total_device_resident_bytes",
    }
    if (
        set(plan) != required
        or plan.get("schema") != RUNTIME_RESIDENCY_PLAN_SCHEMA
        or not isinstance(plan.get("package_id"), str)
        or not plan["package_id"]
        or not _positive_int(plan.get("context_capacity_activations"))
        or not isinstance(plan.get("speculative_decoders_mounted"), bool)
        or not isinstance(plan.get("device_plans"), list)
        or not plan["device_plans"]
        or not _nonnegative_int(plan.get("total_device_resident_bytes"))
    ):
        raise ModelCompileError(
            "runtime residency planner returned a malformed plan"
        )
    device_ids: set[str] = set()
    computed_total = 0
    for device in plan["device_plans"]:
        if (
            not isinstance(device, dict)
            or set(device)
            != {"device_id", "breakdown", "total_device_resident_bytes"}
            or not isinstance(device.get("device_id"), str)
            or not device["device_id"]
            or device["device_id"] in device_ids
            or not isinstance(device.get("breakdown"), dict)
            or not _nonnegative_int(
                device.get("total_device_resident_bytes")
            )
        ):
            raise ModelCompileError(
                "runtime residency planner returned a malformed device plan"
            )
        device_ids.add(device["device_id"])
        breakdown = device["breakdown"]
        if not breakdown or any(
            not isinstance(name, str) or not _nonnegative_int(value)
            for name, value in breakdown.items()
        ):
            raise ModelCompileError(
                "runtime residency planner returned an invalid byte breakdown"
            )
        device_total = sum(breakdown.values())
        if device_total != device["total_device_resident_bytes"]:
            raise ModelCompileError(
                "runtime residency device total disagrees with its breakdown"
            )
        computed_total += device_total
    if computed_total != plan["total_device_resident_bytes"]:
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
