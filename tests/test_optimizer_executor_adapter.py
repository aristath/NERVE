from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.benchmarking.contracts import (
    BenchmarkObservation,
    BenchmarkResidencyEvent,
)
from nerve.representation_optimizer.benchmarking.executor_adapter import (
    ResidentComponentExecutionAdapter,
)
from nerve.representation_optimizer.benchmarking.executor_protocol import (
    EXECUTOR_RESPONSE_SCHEMA,
)
from nerve.representation_optimizer.benchmarking.planning import (
    create_benchmark_workload,
)
from nerve.representation_optimizer.benchmarking.protocols import (
    BenchmarkExecutionRequest,
    BenchmarkMountRequest,
)
from nerve.representation_optimizer.contracts import (
    contract_digest,
    stable_contract_id,
)
from nerve.representation_optimizer.staging.contracts import (
    staged_artifact_digest,
)


class FixtureExecutor:
    def __init__(self) -> None:
        self.commands: list[Json] = []
        self.closed = False
        self.aborted = False

    def request(self, document: Json) -> Json:
        self.commands.append(document)
        command = document["command"]
        if command == "mount":
            payload = {
                "package_id": "package-fixture",
                "candidate_id": document["candidate_id"],
                "component_id": document["component_id"],
                "physical_node_id": document["physical_node_id"],
                "logical_device_id": document["logical_device_id"],
                "physical_device_id": document["physical_device_id"],
                "device_name": "AMD fixture",
                "mount_duration_ns": 11,
                "resident_parameter_bytes": 4_096,
                "resident_transient_bytes": 512,
                "mounted_state_digest": _artifact_digest(b"mounted"),
            }
            status = "mounted"
        elif command == "execute":
            payload = {
                "component_id": "block_13",
                "node_id": "head_norm",
                "op": "parallel_head_norm_rope_2way_codebook_u8",
                "phase": "decode",
                "activation_batch_width": 1,
                "useful_units": document["useful_units"],
                "execution_ns": 400,
                "output_digest": _artifact_digest(b"output"),
                "state_digest": _artifact_digest(b"state"),
                "throughput_windows": [
                    {
                        "index": 0,
                        "start_unit": 0,
                        "end_unit": document["useful_units"],
                        "duration_ns": 400,
                    }
                ],
                "resident_parameter_bytes": 4_096,
                "resident_transient_bytes": 512,
                "physical_dispatch_count": 1,
                "queue_submission_count": 1,
                "synchronization_wait_count": 1,
                "synchronization_wait_ns": 500,
                "queue_wait_ns": 100,
            }
            status = "completed"
        elif command == "close":
            payload = {
                "released": True,
                "release_duration_ns": 7,
                "mounted_state_digest": _artifact_digest(b"mounted"),
            }
            status = "released"
        else:
            raise AssertionError(f"unexpected command {command!r}")
        return {
            "schema": EXECUTOR_RESPONSE_SCHEMA,
            "request_id": document["request_id"],
            "status": status,
            "payload": payload,
        }

    def close(self) -> None:
        self.closed = True

    def abort(self) -> None:
        self.aborted = True


def test_resident_component_adapter_uses_candidate_bound_ordinary_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, executor, loader_calls, environment = _adapter_fixture(
        tmp_path,
        monkeypatch,
    )
    mount_request, execution_request, candidate_id = _requests(tmp_path)

    fixture = b"candidate-specific input"
    candidate_root = tmp_path / "candidate"
    (candidate_root / "fixtures").mkdir(parents=True)
    (candidate_root / "fixtures" / "input.bin").write_bytes(fixture)
    assert (
        b"".join(
            adapter.iter_fixture_artifact(
                "fixtures/input.bin",
                candidate_id=candidate_id,
                chunk_bytes=3,
            )
        )
        == fixture
    )

    session = adapter.open_session(mount_request)
    mount = BenchmarkResidencyEvent.from_json(session.mount_event).to_json()
    observation = BenchmarkObservation.from_json(
        session.execute(execution_request)
    ).to_json()
    unmount = BenchmarkResidencyEvent.from_json(session.close()).to_json()

    assert loader_calls == [candidate_id, candidate_id]
    assert environment["VK_DRIVER_FILES"] == str(
        (tmp_path / "radeon_icd.json").resolve()
    )
    assert "VK_ICD_FILENAMES" not in environment
    assert executor.commands[0]["candidate_id"] == candidate_id
    assert executor.commands[0]["candidate_root"] == str(
        candidate_root.resolve()
    )
    assert executor.commands[0]["physical_device_id"] == "vulkan:amd-fixture"
    assert executor.commands[0]["maximum_quantum_wait_ns"] == 9_000_000
    assert executor.commands[1]["useful_units"] == 8
    assert observation["status"] == "completed"
    assert observation["work"]["useful_units"] == 8
    assert observation["device"]["busy_ns"] == 400
    assert observation["timing"]["queue_wait_ns"] == 100
    assert observation["synchronization"]["wait_ns"] == 500
    assert observation["transport"]["bytes"] > 0
    assert observation["default_statistics"]["physical_dispatch_count"] == 1
    assert set(observation["traces"]) == {
        "distribution",
        "tokens",
        "state",
        "random_draws",
        "schedule",
    }
    for artifact in observation["traces"].values():
        assert b"".join(adapter.iter_trace_artifact(artifact["path"]))
    assert mount["released"] is False
    assert unmount["released"] is True
    assert (
        mount["device_state_after_digest"]
        == unmount["device_state_before_digest"]
    )
    assert executor.closed is True
    assert executor.aborted is False


def test_resident_component_adapter_aborts_a_mismatched_mount(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, executor, _, _ = _adapter_fixture(tmp_path, monkeypatch)
    mount_request, _, _ = _requests(tmp_path)
    original_request = executor.request

    def mismatched(document: Json) -> Json:
        response = original_request(document)
        response["payload"]["physical_device_id"] = "vulkan:wrong-device"
        return response

    executor.request = mismatched  # type: ignore[method-assign]

    with pytest.raises(
        ModelCompileError,
        match="mounted different runtime conditions",
    ):
        adapter.open_session(mount_request)

    assert executor.aborted is True
    assert executor.closed is False


def test_resident_component_adapter_rejects_artifact_path_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _, _, _ = _adapter_fixture(tmp_path, monkeypatch)
    _, _, candidate_id = _requests(tmp_path)

    with pytest.raises(ModelCompileError, match="path is unsafe"):
        tuple(
            adapter.iter_fixture_artifact(
                "../outside",
                candidate_id=candidate_id,
            )
        )


def _adapter_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    ResidentComponentExecutionAdapter,
    FixtureExecutor,
    list[str],
    dict[str, str],
]:
    package = tmp_path / "package"
    package.mkdir()
    manifest = package / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    driver = tmp_path / "radeon_icd.json"
    driver.write_text("{}\n", encoding="utf-8")
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    executor = FixtureExecutor()
    loader_calls: list[str] = []
    captured_environment: dict[str, str] = {}

    def load_candidate(
        workspace_root: Path,
        candidate_id: str,
        package_dir: Path,
    ) -> object:
        assert workspace_root == (tmp_path / "workspace").resolve()
        assert package_dir == package.resolve()
        loader_calls.append(candidate_id)
        return SimpleNamespace(path=candidate_root.resolve())

    def executor_factory(
        command: tuple[str, ...],
        environment: dict[str, str],
    ) -> FixtureExecutor:
        assert command == ("fixture-executor",)
        captured_environment.update(environment)
        return executor

    monkeypatch.setenv("VK_ICD_FILENAMES", "/forbidden/nvidia.json")
    adapter = ResidentComponentExecutionAdapter(
        package_manifest=manifest,
        candidate_workspace=tmp_path / "workspace",
        trace_root=tmp_path / "traces",
        executor_command=("fixture-executor",),
        vulkan_driver_files=(driver,),
        executor_factory=executor_factory,
        staged_candidate_loader=load_candidate,  # type: ignore[arg-type]
    )
    return adapter, executor, loader_calls, captured_environment


def _requests(
    tmp_path: Path,
) -> tuple[BenchmarkMountRequest, BenchmarkExecutionRequest, str]:
    candidate_id = "candidate_" + "a" * 32
    input_digest = _artifact_digest(b"candidate-specific input")
    workload = create_benchmark_workload(
        name="fixture targeted component decode",
        execution_phase="component",
        activation_batch_width=1,
        context_size=0,
        state_size=0,
        stream_count=1,
        mount_mode="resident_reuse",
        boundary_mode="local",
        input_artifact={
            "path": "fixtures/input.bin",
            "digest": input_digest,
        },
        initial_state_artifact=None,
        controls={
            "execution": "ordinary",
            "phase": "decode",
            "component_id": "block_13",
            "physical_node_id": "head_norm",
        },
        randomness_algorithm="deterministic_fixture_counter",
        seeds=(17, 19),
        deterministic_replay_required=True,
        permit_sampling_variance=False,
        permit_numerical_nondeterminism=False,
        permit_speculative_schedule_variance=False,
        useful_work_unit="fused_head_norm_rope_dispatches",
        minimum_useful_work_units=8,
        completion_condition="all_dispatches_completed",
        output_allowance=None,
        output_allowance_basis={"kind": "unlimited"},
        sustained_window_count=1,
    ).to_json()
    matched_conditions = {
        "devices": [{"device_id": "vulkan:amd-fixture"}],
        "placement": {"block_13": "vulkan:amd-fixture"},
        "controls": {
            "scheduler": "normal",
            "maximum_quantum_wait_ns": 9_000_000,
        },
        "environment": {"power_profile": "matched"},
        "idle_device_state_digest": _artifact_digest(b"idle"),
        "exclusive_residency": True,
    }
    plan_id = stable_contract_id("benchmark_plan", str(tmp_path))
    implementation_id = f"staged-representation:{candidate_id}"
    mount = BenchmarkMountRequest(
        plan_id=plan_id,
        role="candidate",
        implementation={"implementation_id": implementation_id},
        workload=workload,
        matched_conditions=matched_conditions,
        matched_conditions_digest=contract_digest(matched_conditions),
        seed=17,
        block_index=0,
    )
    execution = BenchmarkExecutionRequest(
        plan_id=plan_id,
        role="candidate",
        implementation_id=implementation_id,
        workload=workload,
        matched_conditions=matched_conditions,
        matched_conditions_digest=contract_digest(matched_conditions),
        phase="measured",
        seed=17,
        pair_index=0,
        order_index=1,
    )
    return mount, execution, candidate_id


def _artifact_digest(payload: bytes) -> str:
    return staged_artifact_digest(payload)
