#[test]
fn device_local_memory_budget_preserves_headroom_from_the_opening_snapshot() {
    let budget = VulkanDeviceLocalMemoryBudget::capture(1_000_000);

    assert_eq!(budget.baseline_available_bytes, 1_000_000);
    assert_eq!(budget.reservable_bytes, 800_000);
    assert_eq!(budget.protected_headroom_bytes, 200_000);
    assert_eq!(budget.counter_tolerance_bytes, 50_000);
    assert_eq!(budget.remaining_bytes_at(1_000_000), 800_000);
    assert_eq!(budget.remaining_bytes_at(900_000), 700_000);
    assert_eq!(budget.remaining_bytes_at(1_100_000), 800_000);
    assert_eq!(budget.remaining_bytes_at(0), 0);
    assert_eq!(budget.protected_headroom_deficit_at(150_000), 0);
    assert_eq!(budget.protected_headroom_deficit_at(149_999), 1);
}

#[test]
fn device_local_memory_policy_is_the_authoritative_budget_partition() {
    let policy = vulkan_device_local_memory_policy();
    let budget = VulkanDeviceLocalMemoryBudget::capture(
        policy.capacity_parts_per_million,
    );

    assert_eq!(policy.schema, VULKAN_DEVICE_LOCAL_MEMORY_POLICY_SCHEMA);
    assert_eq!(
        policy.protected_headroom_fraction_ppm,
        budget.protected_headroom_bytes
    );
    assert_eq!(
        policy.reservable_free_vram_fraction_ppm,
        budget.reservable_bytes
    );
    assert_eq!(
        policy.protected_headroom_fraction_ppm
            + policy.reservable_free_vram_fraction_ppm,
        policy.capacity_parts_per_million
    );
}

#[test]
fn device_local_memory_budget_preserves_the_fraction_on_large_heaps() {
    let baseline = 64 * 1024 * 1024 * 1024;
    let budget = VulkanDeviceLocalMemoryBudget::capture(baseline);
    let protected = baseline
        * VULKAN_DEVICE_LOCAL_PROTECTED_HEADROOM_FRACTION_PPM
        / VULKAN_CAPACITY_PARTS_PER_MILLION;

    assert_eq!(budget.protected_headroom_bytes, protected);
    assert_eq!(budget.reservable_bytes, baseline - protected);
    assert!(protected > 4 * 1024 * 1024 * 1024);
}

#[test]
fn device_local_memory_budget_rejects_fixed_residency_before_dynamic_allocation() {
    let budget = VulkanDeviceLocalMemoryBudget::capture(1_000_000);
    let admission = budget.admit_pending_bytes_at(900_000, 650_000).unwrap();

    assert_eq!(admission.acquired_bytes, 100_000);
    assert_eq!(admission.pending_fixed_bytes, 650_000);
    assert_eq!(admission.allocatable_bytes, 50_000);
    let error = budget
        .admit_pending_bytes_at(900_000, 700_001)
        .unwrap_err();
    assert!(error.0.contains("stable device-local budget"));
    assert!(error.0.contains("700000 bytes remaining"));
}

#[test]
fn selected_device_memory_budget_never_exceeds_physical_device_local_memory() {
    let Some(raw_device_index) = std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX").ok() else {
        eprintln!("skipping device memory-budget test: explicit Vulkan device unset");
        return;
    };
    let device = VulkanComputeDevice::new_for_physical_device_index(
        raw_device_index
            .parse()
            .expect("NERVE_TEST_VULKAN_DEVICE_INDEX must be an integer"),
    )
    .expect("explicit AMD Vulkan device must open");
    let budget = device.device_local_memory_budget();

    assert!(
        budget.baseline_available_bytes <= device.device_local_memory_bytes(),
        "opening available bytes {} exceed physical device-local bytes {}",
        budget.baseline_available_bytes,
        device.device_local_memory_bytes(),
    );
    let protected_headroom = budget.baseline_available_bytes
        * VULKAN_DEVICE_LOCAL_PROTECTED_HEADROOM_FRACTION_PPM
        / VULKAN_CAPACITY_PARTS_PER_MILLION;
    assert_eq!(budget.protected_headroom_bytes, protected_headroom);
    assert_eq!(
        budget.reservable_bytes,
        budget.baseline_available_bytes - protected_headroom,
    );
}

#[test]
fn device_local_memory_reservations_are_bounded_and_released_by_lifetime() {
    let budget = VulkanDeviceLocalMemoryBudget::capture(1_000_000);
    let tracker = Arc::new(Mutex::new(VulkanDeviceLocalMemoryBudgetTracker::new(
        budget,
    )));
    let first = VulkanDeviceLocalMemoryReservation::acquire(&tracker, 1_000_000, 800_000)
        .unwrap();
    let accounting = tracker.lock().unwrap().accounting_at(200_000);
    assert_eq!(accounting.tracked_allocation_bytes, 800_000);
    assert_eq!(accounting.pending_reservation_bytes, 0);
    assert_eq!(accounting.untracked_acquired_bytes, 0);
    assert_eq!(accounting.remaining_bytes, 0);
    assert_eq!(accounting.admissible_remaining_bytes, 50_000);

    let error = VulkanDeviceLocalMemoryReservation::acquire(&tracker, 200_000, 50_001)
        .unwrap_err();
    assert!(error.0.contains("beyond the stable 800000-byte budget"));

    drop(first);
    let accounting = tracker.lock().unwrap().accounting_at(1_000_000);
    assert_eq!(accounting.tracked_allocation_bytes, 0);
    assert_eq!(accounting.remaining_bytes, 800_000);
    assert_eq!(accounting.admissible_remaining_bytes, 850_000);
}

#[test]
fn device_local_memory_reclaimer_registration_is_shared_and_lifetime_bounded() {
    #[derive(Debug)]
    struct CountingReclaimer(Arc<AtomicU64>);

    impl VulkanDeviceLocalMemoryReclaimer for CountingReclaimer {
        fn reclaim_device_local_memory(
            &self,
            _quiescence: VulkanDeviceLocalMemoryQuiescence,
            requested_bytes: usize,
        ) -> Result<usize, VulkanError> {
            self.0.fetch_add(1, Ordering::AcqRel);
            Ok(requested_bytes)
        }
    }

    let tracker = Arc::new(Mutex::new(VulkanDeviceLocalMemoryBudgetTracker::new(
        VulkanDeviceLocalMemoryBudget::capture(1_000_000),
    )));
    let calls = Arc::new(AtomicU64::new(0));
    let registration = VulkanDeviceLocalMemoryBudgetTracker::register_reclaimer(
        &tracker,
        Arc::new(CountingReclaimer(Arc::clone(&calls))),
    )
    .unwrap();
    let live = VulkanDeviceLocalMemoryBudgetTracker::live_reclaimers(&tracker).unwrap();
    assert_eq!(live.len(), 1);
    assert_eq!(
        live[0]
            .reclaim_device_local_memory(
                VulkanDeviceLocalMemoryQuiescence { _private: () },
                17,
            )
            .unwrap(),
        17,
    );
    assert_eq!(calls.load(Ordering::Acquire), 1);

    drop(live);
    drop(registration);
    assert!(
        VulkanDeviceLocalMemoryBudgetTracker::live_reclaimers(&tracker)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn device_local_memory_observation_never_invokes_reclamation_and_uses_hysteresis() {
    #[derive(Debug)]
    struct CountingReclaimer(Arc<AtomicU64>);

    impl VulkanDeviceLocalMemoryReclaimer for CountingReclaimer {
        fn reclaim_device_local_memory(
            &self,
            _quiescence: VulkanDeviceLocalMemoryQuiescence,
            requested_bytes: usize,
        ) -> Result<usize, VulkanError> {
            self.0.fetch_add(1, Ordering::AcqRel);
            Ok(requested_bytes)
        }
    }

    let tracker = Arc::new(Mutex::new(VulkanDeviceLocalMemoryBudgetTracker::new(
        VulkanDeviceLocalMemoryBudget::capture(1_000_000),
    )));
    let calls = Arc::new(AtomicU64::new(0));
    let _registration = VulkanDeviceLocalMemoryBudgetTracker::register_reclaimer(
        &tracker,
        Arc::new(CountingReclaimer(Arc::clone(&calls))),
    )
    .unwrap();

    VulkanDeviceLocalMemoryBudgetTracker::record_execution_observation(
        &tracker,
        0,
        149_999,
    )
    .unwrap();
    let pressured = VulkanDeviceLocalMemoryBudgetTracker::pressure(&tracker).unwrap();
    assert!(pressured.active);
    assert_eq!(pressured.episode, 1);
    assert_eq!(pressured.current_deficit_bytes, 1);
    assert_eq!(calls.load(Ordering::Acquire), 0);

    VulkanDeviceLocalMemoryBudgetTracker::record_execution_observation(
        &tracker,
        0,
        200_000,
    )
    .unwrap();
    assert!(
        VulkanDeviceLocalMemoryBudgetTracker::pressure(&tracker)
            .unwrap()
            .active,
        "returning barely above the entry threshold must not flap pressure state",
    );

    VulkanDeviceLocalMemoryBudgetTracker::record_execution_observation(
        &tracker,
        0,
        250_000,
    )
    .unwrap();
    let recovered = VulkanDeviceLocalMemoryBudgetTracker::pressure(&tracker).unwrap();
    assert!(!recovered.active);
    assert_eq!(recovered.episode, 1);
    assert_eq!(recovered.current_deficit_bytes, 0);
    assert_eq!(calls.load(Ordering::Acquire), 0);
}

#[test]
fn device_local_execution_pressure_reclaims_until_protected_headroom_is_restored() {
    #[derive(Debug)]
    struct RestoringReclaimer {
        available_bytes: Arc<AtomicU64>,
        calls: Arc<AtomicU64>,
    }

    impl VulkanDeviceLocalMemoryReclaimer for RestoringReclaimer {
        fn reclaim_device_local_memory(
            &self,
            _quiescence: VulkanDeviceLocalMemoryQuiescence,
            requested_bytes: usize,
        ) -> Result<usize, VulkanError> {
            self.calls.fetch_add(1, Ordering::AcqRel);
            self.available_bytes.fetch_add(
                u64::try_from(requested_bytes).unwrap_or(u64::MAX),
                Ordering::AcqRel,
            );
            Ok(requested_bytes)
        }
    }

    let budget = VulkanDeviceLocalMemoryBudget::capture(1_000_000);
    let tracker = VulkanDeviceLocalMemoryBudgetTracker::new(budget);
    let available_bytes = Arc::new(AtomicU64::new(100_000));
    let calls = Arc::new(AtomicU64::new(0));
    let accounting = restore_protected_device_local_headroom(
        budget,
        vec![Arc::new(RestoringReclaimer {
            available_bytes: Arc::clone(&available_bytes),
            calls: Arc::clone(&calls),
        })],
        VulkanDeviceLocalMemoryQuiescence { _private: () },
        Duration::ZERO,
        || {
            Ok(tracker.accounting_at(
                available_bytes.load(Ordering::Acquire),
            ))
        },
    )
    .unwrap();

    assert_eq!(calls.load(Ordering::Acquire), 1);
    assert_eq!(accounting.currently_available_bytes, 150_000);
    assert_eq!(budget.protected_headroom_deficit_at(150_000), 0);
}

#[test]
fn device_local_execution_pressure_fails_closed_when_release_does_not_settle() {
    #[derive(Debug)]
    struct UnsettledReclaimer;

    impl VulkanDeviceLocalMemoryReclaimer for UnsettledReclaimer {
        fn reclaim_device_local_memory(
            &self,
            _quiescence: VulkanDeviceLocalMemoryQuiescence,
            requested_bytes: usize,
        ) -> Result<usize, VulkanError> {
            Ok(requested_bytes)
        }
    }

    let budget = VulkanDeviceLocalMemoryBudget::capture(1_000_000);
    let tracker = VulkanDeviceLocalMemoryBudgetTracker::new(budget);
    let error = restore_protected_device_local_headroom(
        budget,
        vec![Arc::new(UnsettledReclaimer)],
        VulkanDeviceLocalMemoryQuiescence { _private: () },
        Duration::ZERO,
        || Ok(tracker.accounting_at(100_000)),
    )
    .unwrap_err();

    assert!(error.0.contains("execution refused"));
    assert!(error.0.contains("still lacks 50000 bytes"));
}

#[test]
fn device_local_execution_pressure_requires_an_evictable_store() {
    let budget = VulkanDeviceLocalMemoryBudget::capture(1_000_000);
    let tracker = VulkanDeviceLocalMemoryBudgetTracker::new(budget);
    let error = restore_protected_device_local_headroom(
        budget,
        Vec::new(),
        VulkanDeviceLocalMemoryQuiescence { _private: () },
        Duration::ZERO,
        || Ok(tracker.accounting_at(100_000)),
    )
    .unwrap_err();

    assert!(error.0.contains("execution refused"));
    assert!(error.0.contains("no evictable residency store"));
}

#[test]
fn device_local_execution_pressure_reuses_only_a_recent_physical_observation() {
    let tracker = Arc::new(Mutex::new(VulkanDeviceLocalMemoryBudgetTracker::new(
        VulkanDeviceLocalMemoryBudget::capture(1_000_000),
    )));
    let observations = Arc::new(AtomicU64::new(0));
    let observe = || {
        observations.fetch_add(1, Ordering::AcqRel);
        900_000
    };

    let first = VulkanDeviceLocalMemoryBudgetTracker::execution_accounting(
        &tracker,
        Duration::from_secs(1),
        observe,
    )
    .unwrap();
    let reused = VulkanDeviceLocalMemoryBudgetTracker::execution_accounting(
        &tracker,
        Duration::from_secs(1),
        observe,
    )
    .unwrap();
    let refreshed = VulkanDeviceLocalMemoryBudgetTracker::execution_accounting(
        &tracker,
        Duration::ZERO,
        observe,
    )
    .unwrap();

    assert_eq!(first.currently_available_bytes, 900_000);
    assert_eq!(reused, first);
    assert_eq!(refreshed.currently_available_bytes, 900_000);
    assert_eq!(observations.load(Ordering::Acquire), 2);
}

#[test]
fn device_local_execution_reads_the_control_plane_observation_without_querying() {
    let tracker = Arc::new(Mutex::new(VulkanDeviceLocalMemoryBudgetTracker::new(
        VulkanDeviceLocalMemoryBudget::capture(1_000_000),
    )));
    VulkanDeviceLocalMemoryBudgetTracker::record_execution_observation(
        &tracker,
        0,
        900_000,
    )
    .unwrap();

    let accounting = VulkanDeviceLocalMemoryBudgetTracker::recent_execution_accounting(
        &tracker,
        Duration::from_secs(1),
    )
    .unwrap()
    .expect("fresh control-plane observation must be available to execution");

    assert_eq!(accounting.currently_available_bytes, 900_000);
    assert!(
        VulkanDeviceLocalMemoryBudgetTracker::recent_execution_accounting(
            &tracker,
            Duration::ZERO,
        )
        .unwrap()
        .is_none(),
    );
}

#[test]
fn device_local_execution_observation_is_invalidated_by_tracked_allocations() {
    let tracker = Arc::new(Mutex::new(VulkanDeviceLocalMemoryBudgetTracker::new(
        VulkanDeviceLocalMemoryBudget::capture(1_000_000),
    )));
    let observations = Arc::new(AtomicU64::new(0));
    let observe = || {
        observations.fetch_add(1, Ordering::AcqRel);
        900_000
    };
    VulkanDeviceLocalMemoryBudgetTracker::execution_accounting(
        &tracker,
        Duration::from_secs(1),
        observe,
    )
    .unwrap();
    let reservation =
        VulkanDeviceLocalMemoryReservation::acquire(&tracker, 900_000, 100_000).unwrap();

    VulkanDeviceLocalMemoryBudgetTracker::execution_accounting(
        &tracker,
        Duration::from_secs(1),
        observe,
    )
    .unwrap();
    assert_eq!(observations.load(Ordering::Acquire), 2);

    drop(reservation);
    VulkanDeviceLocalMemoryBudgetTracker::execution_accounting(
        &tracker,
        Duration::from_secs(1),
        observe,
    )
    .unwrap();
    assert_eq!(observations.load(Ordering::Acquire), 3);
}

#[test]
fn device_local_execution_observation_retries_across_an_allocation_race() {
    let tracker = Arc::new(Mutex::new(VulkanDeviceLocalMemoryBudgetTracker::new(
        VulkanDeviceLocalMemoryBudget::capture(1_000_000),
    )));
    let mut observations = 0;
    let mut reservation = None;

    let accounting = VulkanDeviceLocalMemoryBudgetTracker::execution_accounting(
        &tracker,
        Duration::from_secs(1),
        || {
            observations += 1;
            if observations == 1 {
                reservation = Some(
                    VulkanDeviceLocalMemoryReservation::acquire(
                        &tracker,
                        900_000,
                        100_000,
                    )
                    .unwrap(),
                );
                // This is the value observed before the allocation completed.
                900_000
            } else {
                800_000
            }
        },
    )
    .unwrap();

    assert_eq!(observations, 2);
    assert_eq!(accounting.currently_available_bytes, 800_000);
    assert_eq!(accounting.tracked_allocation_bytes, 100_000);
    drop(reservation);
}

#[test]
fn device_local_capacity_permit_is_atomic_and_commits_only_actual_bytes() {
    let budget = VulkanDeviceLocalMemoryBudget::capture(1_000_000);
    let tracker = Arc::new(Mutex::new(VulkanDeviceLocalMemoryBudgetTracker::new(
        budget,
    )));
    let permit = VulkanDeviceLocalMemoryPermit::acquire(&tracker, 1_000_000, 400_000).unwrap();
    let pending = tracker.lock().unwrap().accounting_at(1_000_000);
    assert_eq!(pending.tracked_allocation_bytes, 0);
    assert_eq!(pending.pending_reservation_bytes, 400_000);
    assert_eq!(pending.admissible_remaining_bytes, 562_500);
    let error = VulkanDeviceLocalMemoryReservation::acquire(&tracker, 1_000_000, 562_501)
        .unwrap_err();
    assert!(error.to_string().contains("400000 pending"));

    let allocation = permit.commit(300_000).unwrap();
    let committed = tracker.lock().unwrap().accounting_at(700_000);
    assert_eq!(committed.tracked_allocation_bytes, 300_000);
    assert_eq!(committed.pending_reservation_bytes, 0);
    assert_eq!(committed.untracked_acquired_bytes, 0);

    drop(allocation);
    let released = tracker.lock().unwrap().accounting_at(1_000_000);
    assert_eq!(released.tracked_allocation_bytes, 0);
    assert_eq!(released.pending_reservation_bytes, 0);
}

#[test]
fn device_local_capacity_permit_releases_when_commit_cannot_fit() {
    let budget = VulkanDeviceLocalMemoryBudget::capture(1_000_000);
    let tracker = Arc::new(Mutex::new(VulkanDeviceLocalMemoryBudgetTracker::new(
        budget,
    )));
    let permit = VulkanDeviceLocalMemoryPermit::acquire(&tracker, 1_000_000, 400_000).unwrap();
    let error = permit
        .commit(400_001)
        .expect_err("an allocation cannot exceed its capacity permit");

    assert!(error.to_string().contains("permit holds 400000 bytes"));
    assert_eq!(
        tracker.lock().unwrap().pending_reservation_bytes,
        0,
        "the failed consuming commit must release its pending reservation",
    );
}

#[test]
fn device_local_capacity_permit_commit_does_not_readmit_async_counter_changes() {
    let budget = VulkanDeviceLocalMemoryBudget::capture(1_000_000);
    let tracker = Arc::new(Mutex::new(VulkanDeviceLocalMemoryBudgetTracker::new(
        budget,
    )));
    let permit = VulkanDeviceLocalMemoryPermit::acquire(&tracker, 1_000_000, 400_000).unwrap();

    // A live heap-budget snapshot could now report enough delayed or external
    // acquisition to reject a new 400,000-byte request. The existing permit is
    // already part of the accounting transaction and must remain committable.
    assert!(VulkanDeviceLocalMemoryPermit::acquire(&tracker, 400_000, 1).is_err());
    let allocation = permit.commit(400_000).unwrap();

    let committed = tracker.lock().unwrap().accounting_at(600_000);
    assert_eq!(committed.tracked_allocation_bytes, 400_000);
    assert_eq!(committed.pending_reservation_bytes, 0);
    assert_eq!(committed.untracked_acquired_bytes, 0);
    drop(allocation);
}

#[test]
fn device_local_memory_counter_tolerance_is_bounded_by_protected_headroom() {
    let budget = VulkanDeviceLocalMemoryBudget::capture(1_000_000);
    let tracker = Arc::new(Mutex::new(VulkanDeviceLocalMemoryBudgetTracker::new(
        budget,
    )));
    let allocation =
        VulkanDeviceLocalMemoryReservation::acquire(&tracker, 1_000_000, 850_000).unwrap();
    let error = VulkanDeviceLocalMemoryReservation::acquire(&tracker, 150_000, 1).unwrap_err();

    assert!(error.to_string().contains("bounded counter tolerance"));
    assert_eq!(budget.counter_tolerance_bytes, budget.protected_headroom_bytes / 4);
    drop(allocation);
}

#[test]
fn device_local_memory_tolerance_is_applied_before_capacity_saturation() {
    let budget = VulkanDeviceLocalMemoryBudget::capture(1_000_000);
    let tracker = Arc::new(Mutex::new(VulkanDeviceLocalMemoryBudgetTracker::new(
        budget,
    )));
    let allocation =
        VulkanDeviceLocalMemoryReservation::acquire(&tracker, 1_000_000, 840_000).unwrap();
    let accounting = tracker.lock().unwrap().accounting_at(155_000);

    assert_eq!(accounting.untracked_acquired_bytes, 5_000);
    assert_eq!(accounting.remaining_bytes, 0);
    assert_eq!(accounting.admissible_remaining_bytes, 5_000);
    drop(allocation);
}

#[test]
fn selected_device_buffer_lifetime_owns_and_releases_capacity() {
    let Some(raw_device_index) = std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX").ok() else {
        eprintln!("skipping device allocation-budget test: explicit Vulkan device unset");
        return;
    };
    let device = VulkanComputeDevice::new_for_physical_device_index(
        raw_device_index
            .parse()
            .expect("NERVE_TEST_VULKAN_DEVICE_INDEX must be an integer"),
    )
    .expect("explicit AMD Vulkan device must open");
    assert_eq!(
        device
            .device_local_memory_accounting()
            .unwrap()
            .tracked_allocation_bytes,
        0,
    );

    let buffer = device.create_addressable_resident_buffer(1 << 20).unwrap();
    let during = device.device_local_memory_accounting().unwrap();
    assert!(during.tracked_allocation_bytes >= buffer.byte_capacity() as u64);
    assert!(during.tracked_allocation_bytes <= during.reservable_bytes);

    drop(buffer);
    assert_eq!(
        device
            .device_local_memory_accounting()
            .unwrap()
            .tracked_allocation_bytes,
        0,
    );
}

#[test]
fn live_addressable_buffers_are_registered_for_device_fault_attribution() {
    let Some(raw_device_index) = std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX").ok() else {
        eprintln!("skipping device-fault attribution test: explicit Vulkan device unset");
        return;
    };
    let device = VulkanComputeDevice::new_for_physical_device_index(
        raw_device_index
            .parse()
            .expect("NERVE_TEST_VULKAN_DEVICE_INDEX must be an integer"),
    )
    .expect("explicit idle AMD Vulkan device must open");
    assert_eq!(
        device.supports_device_fault_reporting(),
        device.has_enabled_device_extension("VK_EXT_device_fault")
    );

    let buffer = device.create_addressable_resident_buffer(4096).unwrap();
    let address = buffer.device_address().unwrap();
    let resolved = device
        .device_address_registry
        .lock()
        .unwrap()
        .resolve(address + 2048)
        .unwrap();
    assert_eq!(resolved.byte_offset, 2048);
    assert_eq!(resolved.byte_capacity, 4096);
    assert!(resolved.label.contains("device-local addressable buffer"));

    drop(buffer);
    assert!(
        device
            .device_address_registry
            .lock()
            .unwrap()
            .resolve(address)
            .is_none()
    );
}
