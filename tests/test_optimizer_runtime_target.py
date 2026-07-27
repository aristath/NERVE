from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nerve.compilation import ModelCompileError
from nerve.compiler_target import (
    CompilerTarget,
    CompilerTargetDevice,
    synthetic_hardware_profile,
)
from nerve.representation_optimizer.automation.device_state import (
    DeviceIdlePolicy,
    LinuxAmdDeviceStateProbe,
    declared_idle_state_digest,
)
from nerve.representation_optimizer.automation.runtime_target import (
    balanced_component_placement,
    prepare_runtime_optimization_targets,
)
from nerve.representation_optimizer.contracts import ContractDocument


def test_linux_amd_probe_rejects_residency_and_attests_clean_release(
    tmp_path: Path,
) -> None:
    profile = _target(("0000:03:00.0",)).hardware_profiles[0].to_json()
    sysfs, proc = _device_filesystem(
        tmp_path,
        pci_address="0000:03:00.0",
        used_vram=64 * 1024 * 1024,
        busy_percent=0,
    )
    probe = LinuxAmdDeviceStateProbe(
        sysfs_drm_root=sysfs,
        proc_root=proc,
    )

    observation = probe.require_idle((profile,))
    expected = declared_idle_state_digest((profile,), probe.policy)
    assert observation[0]["pci_address"] == "0000:03:00.0"
    assert probe.idle_state_digest((profile,)) == expected

    target = SimpleNamespace(
        hardware_profiles=(profile,),
        matched_conditions={
            "environment": {
                "initial_idle_observations": list(observation),
            }
        },
    )
    device = next(
        (sysfs / "card0" / "device").resolve().glob("mem_info_vram_used")
    )
    device.write_text(f"{64 * 1024 * 1024 + 4_096}\n")
    with pytest.raises(ModelCompileError, match="initial VRAM"):
        probe.target_idle_state_digest(target)
    device.write_text(f"{64 * 1024 * 1024}\n")
    assert probe.target_idle_state_digest(target) == expected

    small_context = proc / "41"
    (small_context / "fdinfo").mkdir(parents=True)
    (small_context / "comm").write_text("vulkan-inspector\n")
    (small_context / "fdinfo" / "5").write_text(
        "drm-pdev:\t0000:03:00.0\n"
        "drm-memory-vram:\t12 KiB\n"
        "drm-memory-gtt:\t2048 KiB\n"
    )
    assert probe.require_idle((profile,))[0]["resident_processes"] == []
    (small_context / "fdinfo" / "5").write_text(
        "drm-pdev:\t0000:03:00.0\n"
        "drm-memory-vram:\t12 KiB\n"
        "drm-memory-gtt:\t2048 KiB\n"
        "drm-engine-compute:\t1 ns\n"
    )
    with pytest.raises(ModelCompileError, match="resident DRM consumers"):
        probe.require_idle((profile,))
    (small_context / "fdinfo" / "5").unlink()

    process = proc / "42"
    (process / "fdinfo").mkdir(parents=True)
    (process / "comm").write_text("resident-model\n")
    (process / "fdinfo" / "7").write_text(
        "drm-pdev:\t0000:03:00.0\n"
        "drm-memory-vram:\t4096 KiB\n"
    )
    with pytest.raises(ModelCompileError, match="resident DRM consumers"):
        probe.require_idle((profile,))


def test_linux_amd_probe_tolerates_inaccessible_unrelated_proc_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _target(("0000:03:00.0",)).hardware_profiles[0].to_json()
    sysfs, proc = _device_filesystem(
        tmp_path,
        pci_address="0000:03:00.0",
        used_vram=64 * 1024 * 1024,
        busy_percent=0,
    )
    inaccessible = proc / "1" / "fdinfo"
    inaccessible.mkdir(parents=True)
    original_iterdir = Path.iterdir

    def guarded_iterdir(path: Path):
        if path == inaccessible:
            raise PermissionError("fixture procfs boundary")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", guarded_iterdir)
    probe = LinuxAmdDeviceStateProbe(
        sysfs_drm_root=sysfs,
        proc_root=proc,
    )

    observation = probe.require_idle((profile,))
    assert observation[0]["resident_processes"] == []


def test_target_idle_attestation_waits_for_stable_counter_quiescence(
    tmp_path: Path,
) -> None:
    profile = _target(("0000:03:00.0",)).hardware_profiles[0].to_json()
    sysfs, proc = _device_filesystem(
        tmp_path,
        pci_address="0000:03:00.0",
        used_vram=64 * 1024 * 1024,
        busy_percent=0,
    )
    busy = next(
        (sysfs / "card0" / "device").resolve().glob("gpu_busy_percent")
    )
    clock = [0]
    sleeps = []

    def advance(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += round(seconds * 1_000_000_000)
        if len(sleeps) == 1:
            busy.write_text("0\n")

    policy = DeviceIdlePolicy(
        quiescence_poll_interval_ns=10,
        quiescence_required_observations=2,
        maximum_quiescence_wait_ns=100,
    )
    probe = LinuxAmdDeviceStateProbe(
        sysfs_drm_root=sysfs,
        proc_root=proc,
        policy=policy,
        monotonic_ns=lambda: clock[0],
        sleep=advance,
    )
    baseline = probe.require_idle((profile,))
    target = SimpleNamespace(
        hardware_profiles=(profile,),
        matched_conditions={
            "environment": {
                "initial_idle_observations": list(baseline),
            }
        },
    )
    busy.write_text("73\n")

    assert probe.target_idle_state_digest(target) == (
        declared_idle_state_digest((profile,), policy)
    )
    assert sleeps == [1e-08, 1e-08]


def test_target_idle_attestation_bounds_nonquiescent_counter_wait(
    tmp_path: Path,
) -> None:
    profile = _target(("0000:03:00.0",)).hardware_profiles[0].to_json()
    sysfs, proc = _device_filesystem(
        tmp_path,
        pci_address="0000:03:00.0",
        used_vram=64 * 1024 * 1024,
        busy_percent=0,
    )
    busy = next(
        (sysfs / "card0" / "device").resolve().glob("gpu_busy_percent")
    )
    clock = [0]

    def advance(seconds: float) -> None:
        clock[0] += round(seconds * 1_000_000_000)

    policy = DeviceIdlePolicy(
        quiescence_poll_interval_ns=10,
        quiescence_required_observations=2,
        maximum_quiescence_wait_ns=25,
    )
    probe = LinuxAmdDeviceStateProbe(
        sysfs_drm_root=sysfs,
        proc_root=proc,
        policy=policy,
        monotonic_ns=lambda: clock[0],
        sleep=advance,
    )
    baseline = probe.require_idle((profile,))
    target = SimpleNamespace(
        hardware_profiles=(profile,),
        matched_conditions={
            "environment": {
                "initial_idle_observations": list(baseline),
            }
        },
    )
    busy.write_text("51\n")

    with pytest.raises(ModelCompileError, match="stable idle baseline"):
        probe.target_idle_state_digest(target)
    assert clock[0] == policy.maximum_quiescence_wait_ns


def test_runtime_target_preparation_selects_minimum_idle_amd_group(
    tmp_path: Path,
) -> None:
    pci_addresses = (
        "0000:03:00.0",
        "0000:07:00.0",
        "0000:0a:00.0",
    )
    target = _target(pci_addresses, capacity_bytes=1_000)
    package = _package(
        tmp_path / "package",
        target,
        tensor_sizes=(600, 600),
    )
    sysfs = tmp_path / "sys" / "class" / "drm"
    proc = tmp_path / "proc"
    for index, pci_address in enumerate(pci_addresses):
        _device_filesystem(
            tmp_path,
            pci_address=pci_address,
            used_vram=1 if index != 0 else 300,
            busy_percent=0,
            card_index=index,
            roots=(sysfs, proc),
            total_vram=1_000,
        )
    probe = LinuxAmdDeviceStateProbe(
        sysfs_drm_root=sysfs,
        proc_root=proc,
    )
    driver = _driver(tmp_path)
    component = _executable(tmp_path / "component-executor")
    validation = _executable(tmp_path / "validation-executor")
    run_root = tmp_path / "run"

    prepared = prepare_runtime_optimization_targets(
        package_manifest=package,
        run_root=run_root,
        component_executor_bin=component,
        validation_executor_bin=validation,
        vulkan_driver_files=(driver,),
        idle_probe=probe,
        live_target=_target(pci_addresses[1:], capacity_bytes=1_000),
    )

    assert not run_root.exists()
    assert prepared.parameter_bytes == 1_200
    assert [
        item["device_id"] for item in prepared.selected_devices
    ] == [
        _device_id("0000:07:00.0"),
        _device_id("0000:0a:00.0"),
    ]
    assert prepared.excluded_devices[0]["device_id"] == _device_id(
        "0000:03:00.0"
    )
    assert len(prepared.targets) == 1
    optimization_target = prepared.targets[0]
    assert len(optimization_target.hardware_profiles) == 2
    placement = optimization_target.matched_conditions["placement"]
    assert set(placement) == {"component_0", "component_1"}
    assert set(placement.values()) == {
        _device_id("0000:07:00.0"),
        _device_id("0000:0a:00.0"),
    }


def test_explicit_busy_optimizer_device_is_never_substituted(
    tmp_path: Path,
) -> None:
    target = _target(("0000:03:00.0",))
    package = _package(tmp_path / "package", target, tensor_sizes=(100,))
    sysfs, proc = _device_filesystem(
        tmp_path,
        pci_address="0000:03:00.0",
        used_vram=100_000_000,
        busy_percent=8,
        total_vram=1_000_000_000,
    )
    probe = LinuxAmdDeviceStateProbe(
        sysfs_drm_root=sysfs,
        proc_root=proc,
    )

    with pytest.raises(ModelCompileError, match="not at an idle"):
        prepare_runtime_optimization_targets(
            package_manifest=package,
            run_root=tmp_path / "run",
            selected_device_ids=(_device_id("0000:03:00.0"),),
            vulkan_driver_files=(_driver(tmp_path),),
            idle_probe=probe,
            live_target=target,
            component_executor_bin=_executable(tmp_path / "component"),
            validation_executor_bin=_executable(tmp_path / "validation"),
        )


def test_balanced_placement_preserves_component_order_and_all_members(
    tmp_path: Path,
) -> None:
    target = _target(("0000:03:00.0", "0000:07:00.0"))
    package = _package(
        tmp_path / "package",
        target,
        tensor_sizes=(10, 100, 10, 100),
    )
    manifest = json.loads(package.read_bytes())
    first, second = (
        _device_id("0000:03:00.0"),
        _device_id("0000:07:00.0"),
    )

    placement = balanced_component_placement(
        package.parent,
        manifest,
        (first, second),
    )

    assigned = [placement[f"component_{index}"] for index in range(4)]
    assert assigned == [first, first, second, second]


def _target(
    pci_addresses: tuple[str, ...],
    *,
    capacity_bytes: int = 2_000_000_000,
) -> CompilerTarget:
    devices = tuple(
        CompilerTargetDevice(
            physical_device_index=index,
            physical_device_id=_device_id(pci_address),
            device_name="AMD test GPU",
            device_type="discrete_gpu",
            pci_address=pci_address,
            vendor_id=0x1002,
            device_id=0x7551,
            shader_features=frozenset(
                {
                    "shader_float16",
                    "shader_bfloat16_type",
                    "shader_int8",
                }
            ),
            subgroup_operations=frozenset({"basic", "arithmetic"}),
            subgroup_compute_supported=True,
            subgroup_size=64,
            max_compute_work_group_invocations=1024,
            max_compute_work_group_size_x=1024,
            cooperative_float16_shapes=((16, 16, 16),),
            cooperative_bfloat16_shapes=((16, 16, 16),),
            cooperative_float8_e4m3_shapes=(),
        )
        for index, pci_address in enumerate(pci_addresses)
    )
    profiles = []
    for device in devices:
        profile = synthetic_hardware_profile(device)
        for domain in profile["memory_domains"]:
            domain["capacity_bytes"] = capacity_bytes
        profile = _reidentify_profile(profile)
        profiles.append(ContractDocument.from_json(profile))
    return CompilerTarget(devices=devices, hardware_profiles=tuple(profiles))


def _reidentify_profile(profile: dict[str, object]) -> dict[str, object]:
    from nerve.representation_optimizer.contracts import stable_contract_id

    profile["capability_class"] = stable_contract_id(
        "hardware_capability",
        {
            "device_kind": profile["hardware_identity"]["device_kind"],
            "architecture": profile["hardware_identity"]["architecture"],
            "processes": profile["processes"],
            "memory_domains": profile["memory_domains"],
            "interconnects": profile["interconnects"],
            "api": profile["provenance"]["api"],
            "api_version": profile["provenance"]["api_version"],
            "capability_extensions": profile["capability_extensions"],
        },
    )
    profile["profile_id"] = stable_contract_id(
        "hardware_profile",
        [
            profile["hardware_identity"],
            profile["capability_class"],
            profile["provenance"],
            profile["identity_extensions"],
            profile["measurements"],
        ],
    )
    return profile


def _package(
    root: Path,
    target: CompilerTarget,
    *,
    tensor_sizes: tuple[int, ...],
) -> Path:
    root.mkdir()
    tensors = {
        f"tensor_{index}": {
            "dtype": "U8",
            "shape": [size],
            "data_offsets": [0, size],
            "parameter_count": size,
            "byte_count": size,
            "source_file": "weights/fixture.safetensors",
            "data_sha256": "0" * 64,
            "layout": "row_major",
        }
        for index, size in enumerate(tensor_sizes)
    }
    (root / "tensors.json").write_text(
        json.dumps(
            {
                "schema": "nerve.tensor_index.v1",
                "source": {},
                "tensors": tensors,
                "totals": {
                    "tensor_count": len(tensors),
                    "parameter_count": sum(tensor_sizes),
                    "byte_count": sum(tensor_sizes),
                },
            }
        )
    )
    components = [
        {
            "component_id": f"component_{index}",
            "params": {
                "refs": {
                    "weight": {"tensor": f"tensor_{index}"},
                }
            },
        }
        for index in range(len(tensor_sizes))
    ]
    manifest = {
        "schema": "nerve.vulkan_resident_model_package.v4",
        "package_id": "fixture-package",
        "tensor_index_path": "tensors.json",
        "compiler_target": target.to_json(),
        "circuit_graph": {"components": components},
    }
    path = root / "vulkan_resident_package.json"
    path.write_text(json.dumps(manifest))
    return path


def _device_filesystem(
    tmp_path: Path,
    *,
    pci_address: str,
    used_vram: int,
    busy_percent: int,
    total_vram: int = 34_000_000_000,
    card_index: int = 0,
    roots: tuple[Path, Path] | None = None,
) -> tuple[Path, Path]:
    sysfs, proc = roots or (
        tmp_path / "sys" / "class" / "drm",
        tmp_path / "proc",
    )
    sysfs.mkdir(parents=True, exist_ok=True)
    proc.mkdir(parents=True, exist_ok=True)
    device = tmp_path / "sys" / "devices" / pci_address
    device.mkdir(parents=True, exist_ok=True)
    (device / "vendor").write_text("0x1002\n")
    (device / "mem_info_vram_total").write_text(f"{total_vram}\n")
    (device / "mem_info_vram_used").write_text(f"{used_vram}\n")
    (device / "gpu_busy_percent").write_text(f"{busy_percent}\n")
    card = sysfs / f"card{card_index}"
    card.mkdir(exist_ok=True)
    (card / "device").symlink_to(device, target_is_directory=True)
    return sysfs, proc


def _driver(tmp_path: Path) -> Path:
    path = tmp_path / "radeon_icd.json"
    if not path.exists():
        path.write_text(
            json.dumps(
                {
                    "file_format_version": "1.0.0",
                    "ICD": {
                        "library_path": "libvulkan_radeon.so",
                        "api_version": "1.4.0",
                    },
                }
            )
        )
    return path


def _executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(path.stat().st_mode | 0o111)
    return path


def _device_id(pci_address: str) -> str:
    compact = pci_address.replace(":", "").replace(".", "")
    return f"vulkan-uuid:{compact:0<32}"
