from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from nerve.compilation import ModelCompileCancelled, ModelCompileError
from nerve.compiler_target import (
    CompilerTarget,
    CompilerTargetDevice,
    compiler_device_probe_command,
    discover_compiler_target,
    synthetic_hardware_profile,
)


def device_payload(
    *,
    index: int,
    device_type: str,
    features: list[str],
) -> dict[str, object]:
    return {
        "physical_device_index": index,
        "physical_device_id": f"vulkan:test-{index}",
        "device_name": f"device {index}",
        "device_type": device_type,
        "pci_address": f"0000:{index:02x}:00.0",
        "vendor_id": 1,
        "device_id": index,
        "shader_features": features,
        "subgroup_operations": [],
        "subgroup_compute_supported": True,
        "subgroup_size": 32,
        "max_compute_work_group_invocations": 1024,
        "max_compute_work_group_size_x": 1024,
        "cooperative_float16_shapes": [[16, 16, 16]],
        "cooperative_bfloat16_shapes": [[16, 16, 16]],
        "cooperative_float8_e4m3_shapes": [[16, 16, 32]],
    }


def hardware_inventory(*devices: dict[str, object]) -> dict[str, object]:
    profiles = [
        synthetic_hardware_profile(CompilerTargetDevice.from_json(device))
        for device in devices
    ]
    profiles.sort(
        key=lambda profile: profile["hardware_identity"]["stable_device_id"]
    )
    return {
        "schema": "nerve.hardware_process_inventory.v1",
        "profiles": profiles,
    }


def test_compiler_target_preserves_dtype_supported_by_any_gpu() -> None:
    target = CompilerTarget.from_hardware_inventory_json(
        hardware_inventory(
            device_payload(
                index=0,
                device_type="discrete_gpu",
                features=["shader_float16"],
            ),
            device_payload(
                index=1,
                device_type="discrete_gpu",
                features=[
                    "shader_float8",
                    "shader_mixed_float_dot_product_float8_acc_float32",
                    "shader_bfloat16_type",
                ],
            ),
        )
    )

    assert target.supports_native_dtype("F8_E4M3")
    assert target.supports_native_dtype("BF16")
    assert target.supports_native_dtype("F16")
    assert target.supports_native_dtype("F32")
    assert not target.supports_native_dtype("Q8_0")
    assert target.devices[0].max_compute_work_group_invocations == 1024
    assert target.devices[0].max_compute_work_group_size_x == 1024
    assert target.devices[0].subgroup_size == 32
    assert target.to_json()["devices"][0]["subgroup_size"] == 32
    assert (
        target.to_json()["devices"][0]["max_compute_work_group_invocations"]
        == 1024
    )
    assert target.devices[0].cooperative_bfloat16_shapes == ((16, 16, 16),)
    assert target.devices[0].cooperative_float16_shapes == ((16, 16, 16),)
    assert target.devices[0].cooperative_float8_e4m3_shapes == ((16, 16, 32),)
    assert len(target.hardware_profiles) == 2
    assert CompilerTarget.from_json(target.to_json()) == target


def test_identical_gpu_capabilities_share_a_class_without_sharing_identity() -> None:
    peer = device_payload(
        index=1,
        device_type="discrete_gpu",
        features=["shader_float16"],
    )
    peer["device_id"] = 0
    first = hardware_inventory(
        device_payload(
            index=0,
            device_type="discrete_gpu",
            features=["shader_float16"],
        ),
        peer,
    )
    profiles = first["profiles"]

    assert profiles[0]["capability_class"] == profiles[1]["capability_class"]
    assert profiles[0]["profile_id"] != profiles[1]["profile_id"]


def test_compiler_target_ignores_cpu_vulkan_devices_and_requires_a_gpu() -> None:
    with pytest.raises(ModelCompileError, match="at least one Vulkan GPU"):
        CompilerTarget.from_hardware_inventory_json(
            hardware_inventory(
                device_payload(
                    index=0,
                    device_type="cpu",
                    features=[
                        "shader_float8",
                        "shader_mixed_float_dot_product_float8_acc_float32",
                    ],
                )
            )
        )


def test_compiler_target_rejects_malformed_device_entries() -> None:
    with pytest.raises(ModelCompileError, match="invalid hardware profile"):
        CompilerTarget.from_hardware_inventory_json(
            {
                "schema": "nerve.hardware_process_inventory.v1",
                "profiles": ["not a profile"],
            }
        )


def test_compiler_target_rejects_duplicate_vulkan_runtime_indices() -> None:
    peer = device_payload(
        index=1,
        device_type="discrete_gpu",
        features=["shader_float16"],
    )
    inventory = hardware_inventory(
        device_payload(
            index=0,
            device_type="discrete_gpu",
            features=["shader_float16"],
        ),
        peer,
    )
    inventory["profiles"][1]["runtime_bindings"]["vulkan_runtime_binding"][
        "physical_device_index"
    ] = 0

    with pytest.raises(ModelCompileError, match="duplicate Vulkan physical indices"):
        CompilerTarget.from_hardware_inventory_json(inventory)


def test_compiler_target_discovery_validates_runtime_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Completed:
        returncode = 0
        stderr = ""
        stdout = json.dumps(
            hardware_inventory(
                device_payload(
                    index=2,
                    device_type="discrete_gpu",
                    features=[
                        "shader_float8",
                        "shader_mixed_float_dot_product_float8_acc_float32",
                    ],
                )
            )
        )

    monkeypatch.setattr(
        "nerve.compiler_target.subprocess.run",
        lambda *args, **kwargs: Completed(),
    )

    target = discover_compiler_target(runtime_bin=Path("/tmp/nerve-runtime"))

    assert target.devices[0].physical_device_index == 2
    assert compiler_device_probe_command(
        runtime_bin=Path("/tmp/nerve-runtime")
    ) == ["/tmp/nerve-runtime", "--inspect-devices", "--json"]


def test_compiler_target_discovery_fails_closed_on_probe_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Completed:
        returncode = 1
        stderr = "device query failed"
        stdout = ""

    monkeypatch.setattr(
        "nerve.compiler_target.subprocess.run",
        lambda *args, **kwargs: Completed(),
    )

    with pytest.raises(ModelCompileError, match="device query failed"):
        discover_compiler_target(runtime_bin=Path("/tmp/nerve-runtime"))


def test_compiler_target_discovery_forwards_stable_allowlist_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    allowed = "vulkan-uuid:00000000070000000000000000000000"

    class Completed:
        returncode = 0
        stderr = ""
        stdout = json.dumps(
            hardware_inventory(
                device_payload(
                    index=2,
                    device_type="discrete_gpu",
                    features=["shader_float16"],
                )
            )
        )

    def run(command, **kwargs):
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        return Completed()

    monkeypatch.setattr("nerve.compiler_target.subprocess.run", run)

    discover_compiler_target(
        runtime_bin=Path("/tmp/nerve-runtime"),
        allowed_physical_device_ids=(allowed,),
        environment={"VK_DRIVER_FILES": "/tmp/radeon.json"},
    )

    assert captured["command"] == [
        "/tmp/nerve-runtime",
        "--inspect-devices",
        "--json",
        "--allow-physical-device",
        allowed,
    ]
    assert captured["environment"] == {
        "VK_DRIVER_FILES": "/tmp/radeon.json"
    }


def test_compiler_target_discovery_kills_probe_when_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"checks": 0, "killed": False}

    class Process:
        returncode = None

        def communicate(self, timeout=None):
            if timeout is not None and not state["killed"]:
                raise subprocess.TimeoutExpired(
                    cmd=["nerve-runtime"],
                    timeout=timeout,
                )
            return "", ""

        def kill(self):
            state["killed"] = True
            self.returncode = -9

    monkeypatch.setattr(
        "nerve.compiler_target.subprocess.Popen",
        lambda *args, **kwargs: Process(),
    )

    def cancel_after_process_start() -> bool:
        state["checks"] += 1
        return state["checks"] > 1

    with pytest.raises(ModelCompileCancelled, match="cancelled"):
        discover_compiler_target(
            runtime_bin=Path("/tmp/nerve-runtime"),
            cancel_requested=cancel_after_process_start,
        )

    assert state["killed"] is True
