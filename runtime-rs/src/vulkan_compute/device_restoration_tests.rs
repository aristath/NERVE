fn device_restoration_snapshot(
    physical_device_id: &str,
) -> VulkanDeviceLocalMemoryRestorationSnapshot {
    VulkanDeviceLocalMemoryRestorationSnapshot {
        physical_device_id: physical_device_id.to_string(),
        device_name: "test device".to_string(),
        pci_address: Some("0000:01:00.0".to_string()),
        api_version: 1,
        driver_version: 2,
        memory_budget: VulkanDeviceLocalMemoryBudget {
            baseline_available_bytes: 1_000,
            reservable_bytes: 800,
            protected_headroom_bytes: 200,
            counter_tolerance_bytes: 16,
        },
        memory_accounting: VulkanDeviceLocalMemoryAccounting {
            baseline_available_bytes: 1_000,
            currently_available_bytes: 900,
            reservable_bytes: 800,
            tracked_allocation_bytes: 5,
            pending_reservation_bytes: 7,
            untracked_acquired_bytes: 9,
            remaining_bytes: 779,
            admissible_remaining_bytes: 795,
        },
        memory_pressure: VulkanDeviceLocalMemoryPressure {
            active: false,
            episode: 2,
            observed_available_bytes: 900,
            current_deficit_bytes: 0,
            peak_deficit_bytes: 0,
        },
    }
}

#[test]
fn device_restoration_accepts_only_driver_noise_within_tolerance() {
    let before = vec![
        device_restoration_snapshot("owner"),
        device_restoration_snapshot("worker"),
    ];
    let mut after = before.iter().cloned().rev().collect::<Vec<_>>();
    after[0].memory_accounting.currently_available_bytes -= 16;
    after[0].memory_accounting.untracked_acquired_bytes += 16;
    after[0].memory_accounting.remaining_bytes -= 16;
    after[0].memory_accounting.admissible_remaining_bytes -= 16;

    let report = verify_vulkan_device_local_memory_restoration(&before, &after);

    assert!(report.complete, "{:?}", report);
    assert_eq!(report.schema, VULKAN_DEVICE_LOCAL_MEMORY_RESTORATION_SCHEMA);
    assert_eq!(report.physical_device_count, 2);
    assert_eq!(report.restored_device_count, 2);
    assert_eq!(
        report
            .devices
            .iter()
            .map(|device| device.physical_device_id.as_str())
            .collect::<Vec<_>>(),
        vec!["owner", "worker"],
    );
}

#[test]
fn device_restoration_rejects_vacuous_duplicate_and_changed_target_sets() {
    let empty = verify_vulkan_device_local_memory_restoration(&[], &[]);
    assert!(!empty.complete);
    assert!(empty.errors.iter().any(|error| error.contains("no selected")));

    let duplicate = vec![
        device_restoration_snapshot("owner"),
        device_restoration_snapshot("owner"),
    ];
    let report = verify_vulkan_device_local_memory_restoration(&duplicate, &duplicate);
    assert!(!report.complete);
    assert!(report.errors.iter().any(|error| error.contains("repeats")));

    let before = vec![device_restoration_snapshot("owner")];
    let after = vec![device_restoration_snapshot("worker")];
    let report = verify_vulkan_device_local_memory_restoration(&before, &after);
    assert!(!report.complete);
    assert!(report.errors.iter().any(|error| error.contains("set changed")));
    assert_eq!(report.devices.len(), 2);
    assert!(report.devices.iter().all(|device| !device.restored));
}

#[test]
fn device_restoration_rejects_changed_device_identity_and_budget() {
    let before = vec![device_restoration_snapshot("owner")];
    let mut after = before.clone();
    after[0].driver_version += 1;
    after[0].memory_budget.reservable_bytes -= 1;

    let report = verify_vulkan_device_local_memory_restoration(&before, &after);

    assert!(!report.complete);
    let errors = &report.devices[0].errors;
    assert!(errors.iter().any(|error| error.contains("driver identity")));
    assert!(errors.iter().any(|error| error.contains("memory budget")));
}

#[test]
fn device_restoration_rejects_every_non_restored_accounting_class() {
    type Mutation = fn(&mut VulkanDeviceLocalMemoryRestorationSnapshot);
    let cases: [(&str, Mutation); 8] = [
        ("accounting baseline", |snapshot| {
            snapshot.memory_accounting.baseline_available_bytes += 1
        }),
        ("accounting reservable", |snapshot| {
            snapshot.memory_accounting.reservable_bytes += 1
        }),
        ("tracked allocation", |snapshot| {
            snapshot.memory_accounting.tracked_allocation_bytes += 1
        }),
        ("pending reservation", |snapshot| {
            snapshot.memory_accounting.pending_reservation_bytes += 1
        }),
        ("untracked acquired", |snapshot| {
            snapshot.memory_accounting.untracked_acquired_bytes += 17
        }),
        ("available device-local", |snapshot| {
            snapshot.memory_accounting.currently_available_bytes -= 17
        }),
        ("remaining reservable", |snapshot| {
            snapshot.memory_accounting.remaining_bytes -= 17
        }),
        ("admissible remaining", |snapshot| {
            snapshot.memory_accounting.admissible_remaining_bytes -= 17
        }),
    ];
    let before = vec![device_restoration_snapshot("owner")];
    for (expected, mutate) in cases {
        let mut after = before.clone();
        mutate(&mut after[0]);
        let report = verify_vulkan_device_local_memory_restoration(&before, &after);
        assert!(!report.complete, "{expected} was accepted");
        assert!(
            report.devices[0]
                .errors
                .iter()
                .any(|error| error.contains(expected)),
            "{expected}: {:?}",
            report.devices[0].errors,
        );
    }
}

#[test]
fn device_restoration_rejects_new_or_unresolved_pressure() {
    let before = vec![device_restoration_snapshot("owner")];
    let mut after = before.clone();
    after[0].memory_pressure.episode += 1;
    let report = verify_vulkan_device_local_memory_restoration(&before, &after);
    assert!(!report.complete);
    assert!(report.devices[0].errors[0].contains("memory-pressure"));

    let mut after = before.clone();
    after[0].memory_pressure.active = true;
    let report = verify_vulkan_device_local_memory_restoration(&before, &after);
    assert!(!report.complete);
    assert!(report.devices[0].errors[0].contains("memory-pressure"));
}

#[test]
fn device_restoration_report_serializes_complete_evidence() {
    let snapshots = vec![device_restoration_snapshot("owner")];
    let report = verify_vulkan_device_local_memory_restoration(&snapshots, &snapshots);

    let value = serde_json::to_value(report).unwrap();

    assert_eq!(
        value["schema"],
        VULKAN_DEVICE_LOCAL_MEMORY_RESTORATION_SCHEMA,
    );
    assert_eq!(value["complete"], true);
    assert_eq!(value["devices"][0]["before"]["memory_accounting"]["tracked_allocation_bytes"], 5);
    assert_eq!(value["devices"][0]["after"]["memory_pressure"]["episode"], 2);
}
