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
    DeviceCapacityPolicy,
    LinuxAmdDeviceCapacityProbe,
    declared_capacity_reservation_digest,
)
from nerve.representation_optimizer.automation.runtime_target import (
    capacity_packed_component_placement,
    prepare_runtime_optimization_targets,
    runtime_executor_command,
    runtime_implementation_fingerprint,
)
from nerve.representation_optimizer.contracts import ContractDocument
from nerve.representation_optimizer.providers.resident_expansion.contracts import (
    PROOF_VERIFIER_ID as RESIDENT_EXPANSION_PROOF_VERIFIER_ID,
)


RUNTIME_IMPLEMENTATION_FINGERPRINT = runtime_implementation_fingerprint(
    Path(__file__).resolve().parents[1] / "runtime-rs"
)


def test_runtime_executor_command_asks_cargo_to_verify_source_freshness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repo"
    runtime = repository / "runtime-rs"
    manifest = runtime / "Cargo.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('[package]\nname = "fixture"\n')
    binary = runtime / "target" / "release" / "fixture-executor"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(binary.stat().st_mode | 0o111)
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> object:
        calls.append(command)
        assert kwargs["stdin"] is not None
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["check"] is False
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.delenv("FIXTURE_EXECUTOR_BIN", raising=False)
    monkeypatch.setattr(
        "nerve.representation_optimizer.automation.runtime_target.shutil.which",
        lambda executable: "/usr/bin/cargo" if executable == "cargo" else None,
    )
    monkeypatch.setattr(
        "nerve.representation_optimizer.automation.runtime_target.subprocess.run",
        run,
    )

    command = runtime_executor_command(
        "fixture-executor",
        explicit=None,
        features=("vulkan", "tokenizers"),
        repo_root=repository,
    )

    assert command == (str(binary.resolve()),)
    assert calls == [
        [
            "/usr/bin/cargo",
            "build",
            "--release",
            "--manifest-path",
            str(manifest),
            "--features",
            "vulkan,tokenizers",
            "--bin",
            "fixture-executor",
        ]
    ]


def test_linux_amd_probe_reserves_remaining_capacity_without_excluding_workloads(
    tmp_path: Path,
) -> None:
    profile = (
        _target(
            ("0000:03:00.0",),
            capacity_bytes=1_000_000_000,
        )
        .hardware_profiles[0]
        .to_json()
    )
    sysfs, proc = _device_filesystem(
        tmp_path,
        pci_address="0000:03:00.0",
        used_vram=100_000_000,
        busy_percent=83,
        total_vram=1_000_000_000,
    )
    process = proc / "42"
    (process / "fdinfo").mkdir(parents=True)
    (process / "comm").write_text("resident-model\n")
    (process / "fdinfo" / "7").write_text(
        "drm-pdev:\t0000:03:00.0\ndrm-memory-vram:\t96 MiB\n"
    )
    policy = DeviceCapacityPolicy(
        reservable_free_vram_fraction_ppm=1_000_000,
    )
    probe = LinuxAmdDeviceCapacityProbe(
        sysfs_drm_root=sysfs,
        proc_root=proc,
        policy=policy,
    )
    device_id = str(profile["hardware_identity"]["stable_device_id"])
    reservation = {device_id: 800_000_000}

    observation = probe.require_capacity((profile,), reservation)

    assert observation[0]["busy_percent"] == 83
    assert observation[0]["vram_free_bytes"] == 900_000_000
    assert observation[0]["reservable_vram_bytes"] == 900_000_000
    assert observation[0]["resident_processes"][0]["command"] == "resident-model"
    expected = declared_capacity_reservation_digest(
        (profile,),
        reservation,
        policy,
    )
    target = SimpleNamespace(
        hardware_profiles=(profile,),
        matched_conditions={
            "environment": {
                "capacity_observations": list(observation),
                "residency_admission": {
                    "reserved_device_capacity_bytes": reservation,
                },
            }
        },
    )
    state = probe.target_capacity_reservation_state(target)
    assert state.reservation_digest == expected
    assert state.observations == observation
    assert state.release_vram_tolerance_bytes == policy.release_vram_tolerance_bytes
    assert state.release_settle_timeout_ns == policy.release_settle_timeout_ns
    assert state.release_poll_interval_ns == policy.release_poll_interval_ns

    used = (sysfs / "card0" / "device").resolve() / "mem_info_vram_used"
    used.write_text("250000000\n")
    with pytest.raises(ModelCompileError, match="insufficient unreserved VRAM"):
        probe.target_capacity_reservation_state(target)
    used.write_text("100000000\n")
    assert (
        probe.target_capacity_reservation_state(target).reservation_digest == expected
    )
    (process / "fdinfo" / "7").unlink()
    with pytest.raises(ModelCompileError, match="no longer present"):
        probe.target_capacity_reservation_state(target)


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
    probe = LinuxAmdDeviceCapacityProbe(
        sysfs_drm_root=sysfs,
        proc_root=proc,
    )

    observation = probe.require_capacity((profile,))
    assert observation[0]["resident_processes"] == []


def test_capacity_probe_rejects_reservation_for_another_device(
    tmp_path: Path,
) -> None:
    profile = _target(("0000:03:00.0",)).hardware_profiles[0].to_json()
    sysfs, proc = _device_filesystem(
        tmp_path,
        pci_address="0000:03:00.0",
        used_vram=100,
        busy_percent=100,
        total_vram=1_000,
    )
    probe = LinuxAmdDeviceCapacityProbe(
        sysfs_drm_root=sysfs,
        proc_root=proc,
    )
    other_device = _device_id("0000:07:00.0")
    with pytest.raises(ModelCompileError, match="does not match the probed"):
        probe.require_capacity((profile,), {other_device: 100})


def test_capacity_probe_tolerates_counter_drift_but_not_real_capacity_loss(
    tmp_path: Path,
) -> None:
    pci_address = "0000:03:00.0"
    profile = (
        _target((pci_address,), capacity_bytes=1_000_000_000)
        .hardware_profiles[0]
        .to_json()
    )
    sysfs, proc = _device_filesystem(
        tmp_path,
        pci_address=pci_address,
        used_vram=100_000_000,
        busy_percent=0,
        total_vram=1_000_000_000,
    )
    policy = DeviceCapacityPolicy(
        reservable_free_vram_fraction_ppm=1_000_000,
        admission_vram_tolerance_bytes=16 * 1024 * 1024,
    )
    probe = LinuxAmdDeviceCapacityProbe(
        sysfs_drm_root=sysfs,
        proc_root=proc,
        policy=policy,
    )
    required = {_device_id(pci_address): 900_000_000}
    used = (sysfs / "card0" / "device").resolve() / "mem_info_vram_used"

    used.write_text(f"{100_000_000 + policy.admission_vram_tolerance_bytes}\n")
    probe.require_capacity((profile,), required)

    used.write_text(f"{100_000_001 + policy.admission_vram_tolerance_bytes}\n")
    with pytest.raises(ModelCompileError, match="capacity-observation tolerance"):
        probe.require_capacity((profile,), required)


def test_runtime_target_preparation_selects_minimum_capacity_amd_group(
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
    probe = LinuxAmdDeviceCapacityProbe(
        sysfs_drm_root=sysfs,
        proc_root=proc,
    )
    driver = _driver(tmp_path)
    component = _executable(tmp_path / "component-executor")
    validation = _executable(tmp_path / "validation-executor")
    planner = _residency_planner(tmp_path / "residency-planner")
    run_root = tmp_path / "run"

    prepared = prepare_runtime_optimization_targets(
        package_manifest=package,
        run_root=run_root,
        component_executor_bin=component,
        validation_executor_bin=validation,
        residency_planner_bin=planner,
        vulkan_driver_files=(driver,),
        capacity_probe=probe,
        live_target=target,
        speculative_draft_tokens=2,
    )

    assert not run_root.exists()
    assert prepared.parameter_bytes == 1_200
    assert len(prepared.residency_plans) == 1
    assert [item["device_id"] for item in prepared.selected_devices] == [
        _device_id("0000:07:00.0"),
        _device_id("0000:0a:00.0"),
    ]
    assert prepared.excluded_devices[0]["device_id"] == _device_id("0000:03:00.0")
    assert len(prepared.targets) == 1
    optimization_target = prepared.targets[0]
    assert RESIDENT_EXPANSION_PROOF_VERIFIER_ID in {
        verifier.verifier_id
        for verifier in optimization_target.proof_verifiers._verifiers
    }
    assert optimization_target.qualification_regime.speculative_draft_tokens == 2
    assert (
        optimization_target.matched_conditions["controls"]["speculative_draft_tokens"]
        == 2
    )
    assert len(optimization_target.hardware_profiles) == 2
    placement = optimization_target.matched_conditions["placement"]
    assert set(placement) == {"component_0", "component_1"}
    assert set(placement.values()) == {
        _device_id("0000:07:00.0"),
        _device_id("0000:0a:00.0"),
    }
    admission = optimization_target.matched_conditions["environment"][
        "residency_admission"
    ]
    assert admission["plan"] == prepared.residency_plans[0]
    assert set(admission["available_device_capacity_bytes"]) == {
        _device_id("0000:07:00.0"),
        _device_id("0000:0a:00.0"),
    }
    assert admission["reserved_device_capacity_bytes"] == {
        _device_id("0000:07:00.0"): 949,
        _device_id("0000:0a:00.0"): 949,
    }
    assert optimization_target.matched_conditions["residency_scope"] == (
        "capacity_partition"
    )


def test_runtime_target_counts_transient_working_set_before_selecting_topology(
    tmp_path: Path,
) -> None:
    pci_addresses = ("0000:03:00.0", "0000:07:00.0")
    target = _target(pci_addresses, capacity_bytes=1_000)
    package = _package(
        tmp_path / "package",
        target,
        tensor_sizes=(100, 100),
    )
    sysfs = tmp_path / "sys" / "class" / "drm"
    proc = tmp_path / "proc"
    for index, pci_address in enumerate(pci_addresses):
        _device_filesystem(
            tmp_path,
            pci_address=pci_address,
            used_vram=1,
            busy_percent=0,
            card_index=index,
            roots=(sysfs, proc),
            total_vram=1_000,
        )
    prepared = prepare_runtime_optimization_targets(
        package_manifest=package,
        run_root=tmp_path / "run",
        component_executor_bin=_executable(tmp_path / "component"),
        validation_executor_bin=_executable(tmp_path / "validation"),
        residency_planner_bin=_residency_planner(
            tmp_path / "residency",
            extra_bytes_per_device_by_count={1: 800, 2: 0},
        ),
        vulkan_driver_files=(_driver(tmp_path),),
        capacity_probe=LinuxAmdDeviceCapacityProbe(
            sysfs_drm_root=sysfs,
            proc_root=proc,
        ),
        live_target=target,
    )

    assert len(prepared.targets[0].hardware_profiles) == 2
    planned = prepared.residency_plans[0]["device_plans"]
    assert [device["initial_device_resident_bytes"] for device in planned] == [
        100,
        100,
    ]


def test_runtime_target_repacks_boundaries_after_exact_fixed_residency(
    tmp_path: Path,
) -> None:
    pci_addresses = ("0000:03:00.0", "0000:07:00.0")
    target = _target(pci_addresses, capacity_bytes=1_000)
    package = _package(
        tmp_path / "package",
        target,
        tensor_sizes=(300, 300, 300, 300),
    )
    sysfs = tmp_path / "sys" / "class" / "drm"
    proc = tmp_path / "proc"
    for index, pci_address in enumerate(pci_addresses):
        _device_filesystem(
            tmp_path,
            pci_address=pci_address,
            used_vram=1,
            busy_percent=0,
            card_index=index,
            roots=(sysfs, proc),
            total_vram=1_000,
        )

    prepared = prepare_runtime_optimization_targets(
        package_manifest=package,
        run_root=tmp_path / "run",
        component_executor_bin=_executable(tmp_path / "component"),
        validation_executor_bin=_executable(tmp_path / "validation"),
        residency_planner_bin=_residency_planner(
            tmp_path / "residency",
            extra_bytes_per_device_by_count={1: 200, 2: 200},
        ),
        vulkan_driver_files=(_driver(tmp_path),),
        capacity_probe=LinuxAmdDeviceCapacityProbe(
            sysfs_drm_root=sysfs,
            proc_root=proc,
        ),
        live_target=target,
    )

    placement = prepared.targets[0].matched_conditions["placement"]
    first, second = map(_device_id, pci_addresses)
    assert [placement[f"component_{index}"] for index in range(4)] == [
        first,
        first,
        second,
        second,
    ]
    assert [
        device["initial_device_resident_bytes"]
        for device in prepared.residency_plans[0]["device_plans"]
    ] == [800, 800]


def test_demand_retained_rejects_a_device_that_only_fits_the_cold_mount(
    tmp_path: Path,
) -> None:
    pci_address = "0000:07:00.0"
    target = _target((pci_address,), capacity_bytes=1_000)
    package = _package(
        tmp_path / "package",
        target,
        tensor_sizes=(100,),
    )
    sysfs, proc = _device_filesystem(
        tmp_path,
        pci_address=pci_address,
        used_vram=1,
        busy_percent=0,
        total_vram=1_000,
    )

    with pytest.raises(ModelCompileError, match="cannot safely host") as caught:
        prepare_runtime_optimization_targets(
            package_manifest=package,
            run_root=tmp_path / "run",
            component_executor_bin=_executable(tmp_path / "component"),
            validation_executor_bin=_executable(tmp_path / "validation"),
            residency_planner_bin=_residency_planner(
                tmp_path / "residency",
                maximum_dynamic_bytes_per_device_by_count={1: 10_000},
            ),
            vulkan_driver_files=(_driver(tmp_path),),
            capacity_probe=LinuxAmdDeviceCapacityProbe(
                sysfs_drm_root=sysfs,
                proc_root=proc,
            ),
            live_target=target,
        )

    assert "planned': 10100" in str(caught.value)
    assert "available_capacity': 949" in str(caught.value)


def test_runtime_target_records_post_context_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pci_address = "0000:03:00.0"
    target = _target((pci_address,))
    package = _package(
        tmp_path / "package",
        target,
        tensor_sizes=(100,),
    )
    initial_vram = 64 * 1024 * 1024
    sysfs, proc = _device_filesystem(
        tmp_path,
        pci_address=pci_address,
        used_vram=initial_vram,
        busy_percent=0,
    )
    used = next((sysfs / "card0" / "device").resolve().glob("mem_info_vram_used"))
    probe = LinuxAmdDeviceCapacityProbe(
        sysfs_drm_root=sysfs,
        proc_root=proc,
    )

    discovery_count = 0

    def discover_after_context_initialization(**kwargs) -> CompilerTarget:
        nonlocal discovery_count
        assert kwargs["initialize_device_contexts"] is True
        used.write_text(f"{initial_vram + 12_288}\n")
        discovery_count += 1
        return target

    monkeypatch.setattr(
        "nerve.representation_optimizer.automation.runtime_target."
        "discover_compiler_target",
        discover_after_context_initialization,
    )

    prepared = prepare_runtime_optimization_targets(
        package_manifest=package,
        run_root=tmp_path / "run",
        runtime_bin=_executable(tmp_path / "nerve-runtime"),
        selected_device_ids=(_device_id(pci_address),),
        component_executor_bin=_executable(tmp_path / "component"),
        validation_executor_bin=_executable(tmp_path / "validation"),
        residency_planner_bin=_residency_planner(tmp_path / "residency"),
        vulkan_driver_files=(_driver(tmp_path),),
        capacity_probe=probe,
    )

    expected_vram = initial_vram + 12_288
    assert discovery_count == 1
    assert prepared.selected_devices[0]["vram_used_bytes"] == expected_vram
    assert (
        prepared.targets[0].matched_conditions["environment"]["capacity_observations"][
            0
        ]["vram_used_bytes"]
        == expected_vram
    )


def test_runtime_target_rejects_stale_runtime_executor_before_device_work(
    tmp_path: Path,
) -> None:
    pci_address = "0000:03:00.0"
    target = _target((pci_address,))
    package = _package(
        tmp_path / "package",
        target,
        tensor_sizes=(100,),
    )
    sysfs, proc = _device_filesystem(
        tmp_path,
        pci_address=pci_address,
        used_vram=64 * 1024 * 1024,
        busy_percent=0,
    )
    probe = LinuxAmdDeviceCapacityProbe(
        sysfs_drm_root=sysfs,
        proc_root=proc,
    )
    stale = "nerve.runtime_implementation_sha256.v1:" + "b" * 64

    with pytest.raises(
        ModelCompileError,
        match="executables are stale relative to runtime source",
    ):
        prepare_runtime_optimization_targets(
            package_manifest=package,
            run_root=tmp_path / "run",
            component_executor_bin=_executable(
                tmp_path / "component",
                runtime_fingerprint=stale,
            ),
            validation_executor_bin=_executable(tmp_path / "validation"),
            residency_planner_bin=_residency_planner(tmp_path / "residency"),
            vulkan_driver_files=(_driver(tmp_path),),
            capacity_probe=probe,
            live_target=target,
        )

    assert not (tmp_path / "run").exists()


def test_explicit_busy_optimizer_device_uses_its_remaining_capacity(
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
    probe = LinuxAmdDeviceCapacityProbe(
        sysfs_drm_root=sysfs,
        proc_root=proc,
    )

    prepared = prepare_runtime_optimization_targets(
        package_manifest=package,
        run_root=tmp_path / "run",
        selected_device_ids=(_device_id("0000:03:00.0"),),
        vulkan_driver_files=(_driver(tmp_path),),
        capacity_probe=probe,
        live_target=target,
        component_executor_bin=_executable(tmp_path / "component"),
        validation_executor_bin=_executable(tmp_path / "validation"),
        residency_planner_bin=_residency_planner(tmp_path / "residency"),
    )

    assert prepared.selected_devices[0]["busy_percent"] == 8
    assert prepared.selected_devices[0]["vram_used_bytes"] == 100_000_000


def test_capacity_packed_placement_fills_first_device_before_spilling(
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

    placement = capacity_packed_component_placement(
        package.parent,
        manifest,
        {first: 1_000, second: 1_000},
    )

    assigned = [placement[f"component_{index}"] for index in range(4)]
    assert assigned == [first, first, first, second]


def test_capacity_packed_placement_spills_at_measured_remaining_capacity(
    tmp_path: Path,
) -> None:
    target = _target(("0000:03:00.0", "0000:07:00.0"))
    package = _package(
        tmp_path / "package",
        target,
        tensor_sizes=(100, 100, 100, 100),
    )
    manifest = json.loads(package.read_bytes())
    first = _device_id("0000:03:00.0")
    second = _device_id("0000:07:00.0")

    placement = capacity_packed_component_placement(
        package.parent,
        manifest,
        {first: 250, second: 1_000},
    )

    assert [placement[f"component_{index}"] for index in range(4)] == [
        first,
        first,
        second,
        second,
    ]


def test_capacity_packed_placement_excludes_processor_boundary_components(
    tmp_path: Path,
) -> None:
    target = _target(("0000:03:00.0", "0000:07:00.0"))
    package = _package(
        tmp_path / "package",
        target,
        tensor_sizes=(10, 100, 100, 10),
    )
    manifest = json.loads(package.read_bytes())
    manifest["circuit_graph"]["components"][0]["runtime_role"] = "input_transducer"
    manifest["circuit_graph"]["components"][3]["runtime_role"] = "sampler"
    first, second = (
        _device_id("0000:03:00.0"),
        _device_id("0000:07:00.0"),
    )

    placement = capacity_packed_component_placement(
        package.parent,
        manifest,
        {first: 1_000, second: 1_000},
    )

    assert placement == {
        "component_1": first,
        "component_2": second,
    }


def test_capacity_packed_placement_attaches_output_to_last_processor_device(
    tmp_path: Path,
) -> None:
    target = _target(("0000:03:00.0", "0000:07:00.0"))
    package = _package(
        tmp_path / "package",
        target,
        tensor_sizes=(10, 100, 100, 10),
    )
    manifest = json.loads(package.read_bytes())
    manifest["circuit_graph"]["components"][0]["runtime_role"] = "input_transducer"
    manifest["circuit_graph"]["components"][3]["runtime_role"] = "output_transducer"
    first, second = (
        _device_id("0000:03:00.0"),
        _device_id("0000:07:00.0"),
    )

    placement = capacity_packed_component_placement(
        package.parent,
        manifest,
        {first: 1_000, second: 1_000},
    )

    assert placement == {
        "component_1": first,
        "component_2": second,
        "component_3": second,
    }


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
            "runtime_role": "signal_processor",
            "params": {
                "refs": {
                    "weight": {"tensor": f"tensor_{index}"},
                }
            },
        }
        for index in range(len(tensor_sizes))
    ]
    manifest = {
        "schema": "nerve.vulkan_resident_model_package.v10",
        "package_id": "fixture-package",
        "tensor_index_path": "tensors.json",
        "compiler_target": target.to_json(),
        "max_context_activations": 128,
        "speculative_decoders": [],
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


def _executable(
    path: Path,
    *,
    runtime_fingerprint: str = RUNTIME_IMPLEMENTATION_FINGERPRINT,
) -> Path:
    path.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--runtime-implementation-fingerprint" ]; then\n'
        f"  printf '%s\\n' '{runtime_fingerprint}'\n"
        "  exit 0\n"
        "fi\n"
        "exit 0\n"
    )
    path.chmod(path.stat().st_mode | 0o111)
    return path


def _residency_planner(
    path: Path,
    *,
    extra_bytes_per_device_by_count: dict[int, int] | None = None,
    maximum_dynamic_bytes_per_device_by_count: (dict[int, int] | None) = None,
    runtime_fingerprint: str = RUNTIME_IMPLEMENTATION_FINGERPRINT,
) -> Path:
    extras = extra_bytes_per_device_by_count or {}
    maximum_dynamic = maximum_dynamic_bytes_per_device_by_count or {}
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        f"fingerprint = {runtime_fingerprint!r}\n"
        f"extras = {extras!r}\n"
        f"maximum_dynamic = {maximum_dynamic!r}\n"
        "if sys.argv[1:] == ['--runtime-implementation-fingerprint']:\n"
        "    print(fingerprint)\n"
        "    raise SystemExit(0)\n"
        "request = json.load(sys.stdin)\n"
        "manifest_path = pathlib.Path(request['package_manifest'])\n"
        "manifest = json.loads(manifest_path.read_text())\n"
        "index = json.loads((manifest_path.parent / manifest['tensor_index_path']).read_text())\n"
        "sizes = {name: value['byte_count'] for name, value in index['tensors'].items()}\n"
        "components = manifest['circuit_graph']['components']\n"
        "plans = []\n"
        "for case in request['cases']:\n"
        "    device_ids = sorted(set(case['component_placement'].values()))\n"
        "    activation = {device_id: extras.get(len(device_ids), 0) for device_id in device_ids}\n"
        "    tensors = {device_id: set() for device_id in device_ids}\n"
        "    for component in components:\n"
        "        if component.get('runtime_role') != 'signal_processor':\n"
        "            continue\n"
        "        device_id = case['component_placement'][component['component_id']]\n"
        "        for ref in component.get('params', {}).get('refs', {}).values():\n"
        "            tensors[device_id].add(ref['tensor'])\n"
        "    always = {device_id: sum(sizes[name] for name in tensors[device_id]) for device_id in device_ids}\n"
        "    dynamic = {device_id: maximum_dynamic.get(len(device_ids), 0) for device_id in device_ids}\n"
        "    current = {device_id: always[device_id] + (dynamic[device_id] if case['residency_policy'] == 'eager' else 0) for device_id in device_ids}\n"
        "    initial = {device_id: current[device_id] + activation[device_id] for device_id in device_ids}\n"
        "    device_plans = [\n"
        "        {'device_id': device_id,\n"
        "         'parameter_residency': {\n"
        "             'always_resident_bytes': always[device_id],\n"
        "             'initial_dynamic_bytes': current[device_id] - always[device_id],\n"
        "             'current_resident_bytes': current[device_id],\n"
        "             'maximum_addressable_bytes': always[device_id] + dynamic[device_id],\n"
        "             'staging_headroom_bytes': 0},\n"
        "         'resource_store': {\n"
        "             'address_table_device_bytes': 0,\n"
        "             'parameter_slot_table_device_bytes': 0,\n"
        "             'metadata_device_bytes': 0,\n"
        "             'transfer_staging_slot_count': 0,\n"
        "             'transfer_staging_slot_byte_capacity': 0,\n"
        "             'transfer_staging_device_bytes': 0,\n"
        "             'maximum_load_wave_group_count': 0,\n"
        "             'maximum_load_wave_payload_bytes': 0,\n"
        "             'maximum_dynamic_allocation_padding_bytes': 0},\n"
        "         'working_set': {'transient_state_bytes': 0, 'activation_headroom_bytes': activation[device_id]},\n"
        "         'breakdown': {\n"
        "             'stream_state_bytes': 0,\n"
        "             'state_transaction_bytes': 0,\n"
        "             'activation_slot_bytes': activation[device_id],\n"
        "             'boundary_buffer_bytes': 0,\n"
        "             'edge_buffer_bytes': 0,\n"
        "             'stream_control_bytes': 0,\n"
        "             'output_transducer_workspace_bytes': 0,\n"
        "             'sampler_workspace_bytes': 0,\n"
        "             'feedback_workspace_bytes': 0,\n"
        "             'speculative_decoder_state_bytes': 0,\n"
        "             'causal_verification_snapshot_bytes': 0,\n"
        "             'speculative_decoder_activation_bytes': 0,\n"
        "             'speculative_decoder_workspace_bytes': 0},\n"
        "         'initial_device_resident_bytes': initial[device_id]}\n"
        "        for device_id in device_ids\n"
        "    ]\n"
        "    plans.append({'case_id': case['case_id'], 'plan': {\n"
        "        'schema': 'nerve.vulkan_runtime_residency_plan.v3',\n"
        "        'package_id': manifest['package_id'],\n"
        "        'residency_policy': case['residency_policy'],\n"
        "        'context_capacity_activations': case['context_capacity_activations'],\n"
        "        'speculative_draft_tokens': case['speculative_draft_tokens'],\n"
        "        'device_plans': device_plans,\n"
        "        'total_initial_device_resident_bytes': sum(initial.values()),\n"
        "        'total_current_resident_parameter_bytes': sum(current.values()),\n"
        "        'total_maximum_addressable_parameter_bytes': sum(always[device_id] + dynamic[device_id] for device_id in device_ids),\n"
        "    }})\n"
        "json.dump({'schema': 'nerve.runtime_residency_planner_response.v3', 'plans': plans}, sys.stdout)\n"
    )
    path.chmod(path.stat().st_mode | 0o111)
    return path


def _device_id(pci_address: str) -> str:
    compact = pci_address.replace(":", "").replace(".", "")
    return f"vulkan-uuid:{compact:0<32}"
