from __future__ import annotations

from pathlib import Path

import pytest

from nerve.device_reservation_gate import (
    EXTERNAL_DEVICE_RESERVATION_SNAPSHOT_SCHEMA,
    DeviceReservationGateError,
    DrmDeviceReservation,
    DrmProcessReservation,
    ExternalDeviceReservationSnapshot,
    LinuxDrmExternalDeviceReservationProbe,
    verify_external_device_reservations_restored,
)


def _snapshot(
    *devices: DrmDeviceReservation,
) -> ExternalDeviceReservationSnapshot:
    return ExternalDeviceReservationSnapshot(
        schema=EXTERNAL_DEVICE_RESERVATION_SNAPSHOT_SCHEMA,
        devices=tuple(devices),
    )


def _process(
    pid: int = 42,
    start_time_ticks: int = 100,
    vram_bytes: int = 96 * 1024 * 1024,
    shared_bytes: int = 4 * 1024 * 1024,
) -> DrmProcessReservation:
    return DrmProcessReservation(
        pid=pid,
        start_time_ticks=start_time_ticks,
        command="existing-model",
        vram_bytes=vram_bytes,
        shared_bytes=shared_bytes,
    )


def _device(
    pci_address: str = "0000:03:00.0",
    *,
    used: int = 200 * 1024 * 1024,
    processes: tuple[DrmProcessReservation, ...] = (_process(),),
) -> DrmDeviceReservation:
    return DrmDeviceReservation(
        pci_address=pci_address,
        drm_card="card0",
        vram_total_bytes=32 * 1024**3,
        vram_used_bytes=used,
        busy_percent=0,
        resident_processes=processes,
    )


def _write_process_stat(path: Path, pid: int, start_time_ticks: int) -> None:
    fields = ["S", *(["0"] * 18), str(start_time_ticks), *(["0"] * 3)]
    path.write_text(f"{pid} (client process) " + " ".join(fields) + "\n")


def _drm_fixture(tmp_path: Path) -> tuple[Path, Path]:
    sysfs = tmp_path / "sys" / "class" / "drm"
    pci = tmp_path / "sys" / "devices" / "pci0000:00" / "0000:03:00.0"
    pci.mkdir(parents=True)
    (pci / "mem_info_vram_total").write_text(str(32 * 1024**3))
    (pci / "mem_info_vram_used").write_text(str(200 * 1024**2))
    (pci / "gpu_busy_percent").write_text("37")
    card = sysfs / "card0"
    card.mkdir(parents=True)
    (card / "device").symlink_to(pci, target_is_directory=True)

    proc = tmp_path / "proc"
    process = proc / "42"
    (process / "fdinfo").mkdir(parents=True)
    (process / "fd").mkdir()
    (process / "comm").write_text("existing-model\n")
    _write_process_stat(process / "stat", 42, 100)
    client = tmp_path / "drm-client"
    client.write_text("")
    for descriptor in ("7", "8"):
        (process / "fd" / descriptor).symlink_to(client)
        (process / "fdinfo" / descriptor).write_text(
            "drm-client-id:\t9\n"
            "drm-pdev:\t0000:03:00.0\n"
            "drm-memory-vram:\t96 MiB\n"
            "drm-memory-gtt:\t4 MiB\n"
        )
    return sysfs, proc


def test_linux_drm_reservation_probe_records_process_instance_and_deduplicates_fd(
    tmp_path: Path,
) -> None:
    sysfs, proc = _drm_fixture(tmp_path)
    probe = LinuxDrmExternalDeviceReservationProbe(
        sysfs_drm_root=sysfs,
        proc_root=proc,
    )

    snapshot = probe.capture()

    assert len(snapshot.devices) == 1
    device = snapshot.devices[0]
    assert device.pci_address == "0000:03:00.0"
    assert device.vram_used_bytes == 200 * 1024**2
    assert device.busy_percent == 37
    assert device.resident_processes == (_process(),)


def test_linux_drm_reservation_probe_sums_distinct_clients_in_one_process(
    tmp_path: Path,
) -> None:
    sysfs, proc = _drm_fixture(tmp_path)
    process = proc / "42"
    client = tmp_path / "second-drm-client"
    client.write_text("")
    (process / "fd" / "9").symlink_to(client)
    (process / "fdinfo" / "9").write_text(
        "drm-client-id:\t10\n"
        "drm-pdev:\t0000:03:00.0\n"
        "drm-memory-vram:\t8 MiB\n"
        "drm-memory-gtt:\t2 MiB\n"
    )
    probe = LinuxDrmExternalDeviceReservationProbe(
        sysfs_drm_root=sysfs,
        proc_root=proc,
    )

    snapshot = probe.capture()

    assert snapshot.devices[0].resident_processes == (
        _process(
            vram_bytes=104 * 1024**2,
            shared_bytes=6 * 1024**2,
        ),
    )


def test_linux_drm_reservation_probe_reads_generic_resident_memory_regions(
    tmp_path: Path,
) -> None:
    sysfs, proc = _drm_fixture(tmp_path)
    process = proc / "42"
    for descriptor in ("7", "8"):
        (process / "fdinfo" / descriptor).write_text(
            "drm-client-id:\t9\n"
            "drm-pdev:\t0000:03:00.0\n"
            "drm-resident-vram0:\t72 MiB\n"
            "drm-resident-vram1:\t8 MiB\n"
            "drm-resident-gtt:\t2 MiB\n"
            "drm-resident-system:\t3 MiB\n"
            "drm-resident-stolen:\t1 MiB\n"
        )
    probe = LinuxDrmExternalDeviceReservationProbe(
        sysfs_drm_root=sysfs,
        proc_root=proc,
    )

    snapshot = probe.capture()

    assert snapshot.devices[0].resident_processes == (
        _process(
            vram_bytes=80 * 1024**2,
            shared_bytes=6 * 1024**2,
        ),
    )


def test_linux_drm_reservation_probe_rejects_malformed_client_identity(
    tmp_path: Path,
) -> None:
    sysfs, proc = _drm_fixture(tmp_path)
    fdinfo = proc / "42" / "fdinfo" / "7"
    fdinfo.write_text(
        fdinfo.read_text().replace("drm-client-id:\t9", "drm-client-id:\tx")
    )
    probe = LinuxDrmExternalDeviceReservationProbe(
        sysfs_drm_root=sysfs,
        proc_root=proc,
    )

    with pytest.raises(DeviceReservationGateError, match="invalid DRM client identity"):
        probe.capture()


def test_external_reservation_accepts_new_clients_and_bounded_driver_noise() -> None:
    before = _snapshot(_device())
    after = _snapshot(
        _device(
            used=216 * 1024 * 1024,
            processes=(_process(), _process(pid=77, start_time_ticks=900)),
        )
    )

    report = verify_external_device_reservations_restored(
        ("0000:03:00.0",),
        before,
        after,
    )

    assert report.complete
    assert report.selected_device_count == 1
    assert report.restored_device_count == 1


def test_external_reservation_allows_growth_owned_by_a_concurrent_process() -> None:
    before = _snapshot(_device())
    new_process = _process(
        pid=77,
        start_time_ticks=900,
        vram_bytes=128 * 1024 * 1024,
        shared_bytes=0,
    )
    after = _snapshot(
        _device(
            used=328 * 1024 * 1024,
            processes=(_process(), new_process),
        )
    )

    report = verify_external_device_reservations_restored(
        ("0000:03:00.0",),
        before,
        after,
    )

    assert report.complete


def test_external_reservation_does_not_attribute_gtt_growth_to_vram() -> None:
    before = _snapshot(_device())
    new_process = _process(
        pid=77,
        start_time_ticks=900,
        vram_bytes=0,
        shared_bytes=128 * 1024 * 1024,
    )
    after = _snapshot(
        _device(
            used=217 * 1024 * 1024,
            processes=(_process(), new_process),
        )
    )

    report = verify_external_device_reservations_restored(
        ("0000:03:00.0",),
        before,
        after,
    )

    assert not report.complete
    assert any(
        "retained 17825792 unattributed aggregate VRAM bytes" in error
        for error in report.errors
    )


@pytest.mark.parametrize(
    ("after", "message"),
    (
        (_snapshot(), "absent from the post-workload"),
        (
            _snapshot(_device(processes=())),
            "lost pre-existing process",
        ),
        (
            _snapshot(_device(processes=(_process(start_time_ticks=101),))),
            "lost pre-existing process",
        ),
        (
            _snapshot(
                _device(
                    processes=(_process(vram_bytes=70 * 1024 * 1024, shared_bytes=0),)
                )
            ),
            "lost 31457280 DRM allocation bytes",
        ),
        (
            _snapshot(_device(used=217 * 1024 * 1024)),
            "retained 17825792 unattributed aggregate VRAM bytes",
        ),
    ),
)
def test_external_reservation_rejects_missing_reused_or_changed_allocations(
    after: ExternalDeviceReservationSnapshot,
    message: str,
) -> None:
    report = verify_external_device_reservations_restored(
        ("0000:03:00.0",),
        _snapshot(_device()),
        after,
    )

    assert not report.complete
    assert any(message in error for error in report.errors)


def test_external_reservation_rejects_vacuous_duplicate_and_invalid_selection() -> None:
    snapshot = _snapshot(_device())
    for selected, message in (
        ((), "no selected devices"),
        (("0000:03:00.0", "03:00.0"), "repeats a selected device"),
        (("not-pci",), "invalid PCI address"),
    ):
        report = verify_external_device_reservations_restored(
            selected,
            snapshot,
            snapshot,
        )
        assert not report.complete
        assert any(message in error for error in report.errors)


def test_external_reservation_rejects_malformed_or_duplicate_process_evidence() -> None:
    malformed = DrmProcessReservation(
        pid=0,
        start_time_ticks=-1,
        command="",
        vram_bytes=-1,
        shared_bytes=0,
    )
    device = DrmDeviceReservation(
        pci_address="0000:03:00.0",
        drm_card="",
        vram_total_bytes=100,
        vram_used_bytes=101,
        busy_percent=101,
        resident_processes=(malformed, malformed),
    )
    snapshot = _snapshot(device)

    report = verify_external_device_reservations_restored(
        ("0000:03:00.0",),
        snapshot,
        snapshot,
    )

    assert not report.complete
    assert any("invalid aggregate VRAM" in error for error in report.errors)
    assert any("invalid resident-process counters" in error for error in report.errors)
    assert any("repeats resident process" in error for error in report.errors)


def test_linux_drm_reservation_probe_rejects_partial_capacity_and_process_identity(
    tmp_path: Path,
) -> None:
    sysfs, proc = _drm_fixture(tmp_path)
    device = (sysfs / "card0" / "device").resolve()
    (device / "mem_info_vram_used").unlink()
    probe = LinuxDrmExternalDeviceReservationProbe(
        sysfs_drm_root=sysfs,
        proc_root=proc,
    )
    with pytest.raises(DeviceReservationGateError, match="incomplete VRAM"):
        probe.capture()

    (device / "mem_info_vram_used").write_text("1")
    (proc / "42" / "stat").write_text("malformed")
    with pytest.raises(DeviceReservationGateError, match="process identity"):
        probe.capture()

    (proc / "42" / "stat").unlink()
    assert probe.capture().devices[0].resident_processes == ()
