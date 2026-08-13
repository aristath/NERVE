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
    #[derive(Clone, Debug)]
    struct CountingReclaimer(Arc<AtomicU64>);

    impl VulkanDeviceLocalMemoryReclamation for CountingReclaimer {
        fn reclaim_device_local_memory(
            &self,
            _quiescence: &VulkanDeviceLocalMemoryQuiescence<'_>,
            requested_bytes: usize,
        ) -> Result<usize, VulkanError> {
            self.0.fetch_add(1, Ordering::AcqRel);
            Ok(requested_bytes)
        }
    }

    impl VulkanDeviceLocalMemoryReclaimer for CountingReclaimer {
        fn begin_device_local_memory_reclamation(
            &self,
            _requested_bytes: usize,
        ) -> Result<Box<dyn VulkanDeviceLocalMemoryReclamation>, VulkanError> {
            Ok(Box::new(self.clone()))
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
    let reclamation = live[0]
        .begin_device_local_memory_reclamation(17)
        .unwrap();
    assert_eq!(with_test_device_local_memory_quiescence(|quiescence| {
        reclamation
            .reclaim_device_local_memory(quiescence, 17)
            .unwrap()
    }), 17);
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
    #[derive(Clone, Debug)]
    struct CountingReclaimer(Arc<AtomicU64>);

    impl VulkanDeviceLocalMemoryReclamation for CountingReclaimer {
        fn reclaim_device_local_memory(
            &self,
            _quiescence: &VulkanDeviceLocalMemoryQuiescence<'_>,
            requested_bytes: usize,
        ) -> Result<usize, VulkanError> {
            self.0.fetch_add(1, Ordering::AcqRel);
            Ok(requested_bytes)
        }
    }

    impl VulkanDeviceLocalMemoryReclaimer for CountingReclaimer {
        fn begin_device_local_memory_reclamation(
            &self,
            _requested_bytes: usize,
        ) -> Result<Box<dyn VulkanDeviceLocalMemoryReclamation>, VulkanError> {
            Ok(Box::new(self.clone()))
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

    impl VulkanDeviceLocalMemoryReclamation for RestoringReclaimer {
        fn reclaim_device_local_memory(
            &self,
            _quiescence: &VulkanDeviceLocalMemoryQuiescence<'_>,
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
    let accounting = with_test_device_local_memory_quiescence(|quiescence| {
        restore_protected_device_local_headroom(
            budget,
            vec![Box::new(RestoringReclaimer {
                available_bytes: Arc::clone(&available_bytes),
                calls: Arc::clone(&calls),
            })],
            quiescence,
            Duration::ZERO,
            || Ok(tracker.accounting_at(available_bytes.load(Ordering::Acquire))),
        )
        .unwrap()
    });

    assert_eq!(calls.load(Ordering::Acquire), 1);
    assert_eq!(accounting.currently_available_bytes, 150_000);
    assert_eq!(budget.protected_headroom_deficit_at(150_000), 0);
}

#[test]
fn device_local_execution_pressure_fails_closed_when_release_does_not_settle() {
    #[derive(Debug)]
    struct UnsettledReclaimer;

    impl VulkanDeviceLocalMemoryReclamation for UnsettledReclaimer {
        fn reclaim_device_local_memory(
            &self,
            _quiescence: &VulkanDeviceLocalMemoryQuiescence<'_>,
            requested_bytes: usize,
        ) -> Result<usize, VulkanError> {
            Ok(requested_bytes)
        }
    }

    let budget = VulkanDeviceLocalMemoryBudget::capture(1_000_000);
    let tracker = VulkanDeviceLocalMemoryBudgetTracker::new(budget);
    let error = with_test_device_local_memory_quiescence(|quiescence| {
        restore_protected_device_local_headroom(
            budget,
            vec![Box::new(UnsettledReclaimer)],
            quiescence,
            Duration::ZERO,
            || Ok(tracker.accounting_at(100_000)),
        )
        .unwrap_err()
    });

    assert!(error.0.contains("execution refused"));
    assert!(error.0.contains("still lacks 50000 bytes"));
}

#[test]
fn device_local_execution_pressure_requires_an_evictable_store() {
    let budget = VulkanDeviceLocalMemoryBudget::capture(1_000_000);
    let tracker = VulkanDeviceLocalMemoryBudgetTracker::new(budget);
    let error = with_test_device_local_memory_quiescence(|quiescence| {
        restore_protected_device_local_headroom(
            budget,
            Vec::new(),
            quiescence,
            Duration::ZERO,
            || Ok(tracker.accounting_at(100_000)),
        )
        .unwrap_err()
    });

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
fn device_local_capacity_permit_splits_without_double_counting() {
    let budget = VulkanDeviceLocalMemoryBudget::capture(1_000_000);
    let tracker = Arc::new(Mutex::new(VulkanDeviceLocalMemoryBudgetTracker::new(
        budget,
    )));
    let mut parent =
        VulkanDeviceLocalMemoryPermit::acquire(&tracker, 1_000_000, 400_000).unwrap();

    let first = parent.take(125_000).unwrap();
    let second = parent.take(75_000).unwrap();
    assert_eq!(parent.remaining_byte_count(), 200_000);
    assert_eq!(first.remaining_byte_count(), 125_000);
    assert_eq!(second.remaining_byte_count(), 75_000);
    assert_eq!(
        tracker.lock().unwrap().pending_reservation_bytes,
        400_000,
        "splitting an admitted permit must preserve one aggregate pending claim",
    );

    let first_allocation = first.commit(120_000).unwrap();
    let after_first = tracker.lock().unwrap().accounting_at(880_000);
    assert_eq!(after_first.tracked_allocation_bytes, 120_000);
    assert_eq!(after_first.pending_reservation_bytes, 275_000);

    drop(second);
    assert_eq!(
        tracker.lock().unwrap().pending_reservation_bytes,
        200_000,
    );
    drop(parent);
    let released = tracker.lock().unwrap().accounting_at(880_000);
    assert_eq!(released.tracked_allocation_bytes, 120_000);
    assert_eq!(released.pending_reservation_bytes, 0);
    drop(first_allocation);
}

#[test]
fn device_local_capacity_permit_rejects_invalid_split_without_mutation() {
    let budget = VulkanDeviceLocalMemoryBudget::capture(1_000_000);
    let tracker = Arc::new(Mutex::new(VulkanDeviceLocalMemoryBudgetTracker::new(
        budget,
    )));
    let mut permit =
        VulkanDeviceLocalMemoryPermit::acquire(&tracker, 1_000_000, 400_000).unwrap();

    assert!(permit.take(0).is_err());
    assert!(permit.take(400_001).is_err());
    assert_eq!(permit.remaining_byte_count(), 400_000);
    assert_eq!(
        tracker.lock().unwrap().pending_reservation_bytes,
        400_000,
    );
    drop(permit);
    assert_eq!(tracker.lock().unwrap().pending_reservation_bytes, 0);
}

#[test]
fn stream_memory_admission_scopes_exact_device_and_host_children() {
    let budget = VulkanDeviceLocalMemoryBudget::capture(1_000_000);
    let device_a = Arc::new(Mutex::new(VulkanDeviceLocalMemoryBudgetTracker::new(
        budget,
    )));
    let device_b = Arc::new(Mutex::new(VulkanDeviceLocalMemoryBudgetTracker::new(
        budget,
    )));
    let host = Arc::new(Mutex::new(VulkanHostMemoryBudgetTracker::default()));
    let device_a_key = Arc::as_ptr(&device_a) as usize;
    let device_b_key = Arc::as_ptr(&device_b) as usize;
    let host_key = Arc::as_ptr(&host) as usize;
    let admission = VulkanMemoryAdmission::from_test_permits(
        vec![
            (
                device_a_key,
                VulkanDeviceLocalMemoryPermit::acquire(&device_a, 1_000_000, 300_000).unwrap(),
            ),
            (
                device_b_key,
                VulkanDeviceLocalMemoryPermit::acquire(&device_b, 1_000_000, 200_000).unwrap(),
            ),
        ],
        Some((
            host_key,
            VulkanHostMemoryPermit::acquire(&host, 1_000_000, 100_000).unwrap(),
        )),
    );

    let device_a_allocation;
    let device_b_allocation;
    let host_allocation;
    {
        let _scope = admission.enter();
        device_a_allocation = take_scoped_device_local_memory_capacity(&device_a, 125_000)
            .expect("device A is in the active transaction")
            .unwrap()
            .commit(125_000)
            .unwrap();
        device_b_allocation = take_scoped_device_local_memory_capacity(&device_b, 75_000)
            .expect("device B is in the active transaction")
            .unwrap()
            .commit(75_000)
            .unwrap();
        host_allocation = take_scoped_host_memory_capacity(&host, 40_000)
            .expect("host memory is in the active transaction")
            .unwrap()
            .commit(40_000)
            .unwrap();
    }

    assert!(take_scoped_device_local_memory_capacity(&device_a, 1).is_none());
    assert!(take_scoped_host_memory_capacity(&host, 1).is_none());
    assert_eq!(admission.remaining_device_bytes(device_a_key), 175_000);
    assert_eq!(admission.remaining_device_bytes(device_b_key), 125_000);
    assert_eq!(admission.remaining_host_bytes(), 60_000);
    assert_eq!(device_a.lock().unwrap().tracked_allocation_bytes, 125_000);
    assert_eq!(device_b.lock().unwrap().tracked_allocation_bytes, 75_000);
    assert_eq!(host.lock().unwrap().tracked_allocation_bytes, 40_000);

    drop(admission);
    assert_eq!(device_a.lock().unwrap().pending_reservation_bytes, 0);
    assert_eq!(device_b.lock().unwrap().pending_reservation_bytes, 0);
    assert_eq!(host.lock().unwrap().pending_reservation_bytes, 0);
    drop(device_a_allocation);
    drop(device_b_allocation);
    drop(host_allocation);
    assert_eq!(device_a.lock().unwrap().tracked_allocation_bytes, 0);
    assert_eq!(device_b.lock().unwrap().tracked_allocation_bytes, 0);
    assert_eq!(host.lock().unwrap().tracked_allocation_bytes, 0);
}

#[test]
fn eager_stream_memory_admission_accepts_exactly_consumed_credit() {
    let budget = VulkanDeviceLocalMemoryBudget::capture(1_000_000);
    let device = Arc::new(Mutex::new(VulkanDeviceLocalMemoryBudgetTracker::new(
        budget,
    )));
    let host = Arc::new(Mutex::new(VulkanHostMemoryBudgetTracker::default()));
    let device_key = Arc::as_ptr(&device) as usize;
    let host_key = Arc::as_ptr(&host) as usize;
    let admission = VulkanMemoryAdmission::from_test_permits(
        vec![(
            device_key,
            VulkanDeviceLocalMemoryPermit::acquire(&device, 1_000_000, 100_000).unwrap(),
        )],
        Some((
            host_key,
            VulkanHostMemoryPermit::acquire(&host, 1_000_000, 50_000).unwrap(),
        )),
    );

    let device_allocation;
    let host_allocation;
    {
        let _scope = admission.enter();
        device_allocation = take_scoped_device_local_memory_capacity(&device, 100_000)
            .expect("device is in the eager admission")
            .unwrap()
            .commit(100_000)
            .unwrap();
        host_allocation = take_scoped_host_memory_capacity(&host, 50_000)
            .expect("host is in the eager admission")
            .unwrap()
            .commit(50_000)
            .unwrap();
    }

    admission
        .ensure_fully_consumed("eager fixture")
        .expect("every admitted byte was consumed");
    drop(device_allocation);
    drop(host_allocation);
    assert_eq!(admission.remaining_device_bytes(device_key), 0);
    assert_eq!(admission.remaining_host_bytes(), 0);
    assert_eq!(device.lock().unwrap().pending_reservation_bytes, 0);
    assert_eq!(device.lock().unwrap().tracked_allocation_bytes, 0);
    assert_eq!(host.lock().unwrap().pending_reservation_bytes, 0);
    assert_eq!(host.lock().unwrap().tracked_allocation_bytes, 0);
    drop(admission);
}

#[test]
fn eager_stream_memory_admission_rejects_unexplained_device_and_host_credit() {
    let budget = VulkanDeviceLocalMemoryBudget::capture(1_000_000);
    let device = Arc::new(Mutex::new(VulkanDeviceLocalMemoryBudgetTracker::new(
        budget,
    )));
    let host = Arc::new(Mutex::new(VulkanHostMemoryBudgetTracker::default()));
    let device_key = Arc::as_ptr(&device) as usize;
    let host_key = Arc::as_ptr(&host) as usize;
    let admission = VulkanMemoryAdmission::from_test_permits(
        vec![(
            device_key,
            VulkanDeviceLocalMemoryPermit::acquire(&device, 1_000_000, 100_000).unwrap(),
        )],
        Some((
            host_key,
            VulkanHostMemoryPermit::acquire(&host, 1_000_000, 50_000).unwrap(),
        )),
    );

    let device_allocation;
    let host_allocation;
    {
        let _scope = admission.enter();
        device_allocation = take_scoped_device_local_memory_capacity(&device, 90_000)
            .expect("device is in the eager admission")
            .unwrap()
            .commit(90_000)
            .unwrap();
        host_allocation = take_scoped_host_memory_capacity(&host, 40_000)
            .expect("host is in the eager admission")
            .unwrap()
            .commit(40_000)
            .unwrap();
    }

    let error = admission
        .ensure_fully_consumed("eager fixture")
        .expect_err("unconsumed credit must fail closed");
    assert!(error.to_string().contains("10000 device bytes"));
    assert!(error.to_string().contains("10000 host bytes"));
    assert!(error.to_string().contains("unexplained admission credit"));
    assert!(admission.ensure_fully_consumed("  ").is_err());

    drop(admission);
    drop(device_allocation);
    drop(host_allocation);
    assert_eq!(device.lock().unwrap().pending_reservation_bytes, 0);
    assert_eq!(device.lock().unwrap().tracked_allocation_bytes, 0);
    assert_eq!(host.lock().unwrap().pending_reservation_bytes, 0);
    assert_eq!(host.lock().unwrap().tracked_allocation_bytes, 0);
}

#[test]
fn reusable_stream_memory_admission_recycles_released_device_and_host_capacity() {
    let budget = VulkanDeviceLocalMemoryBudget::capture(1_000_000);
    let device = Arc::new(Mutex::new(VulkanDeviceLocalMemoryBudgetTracker::new(
        budget,
    )));
    let host = Arc::new(Mutex::new(VulkanHostMemoryBudgetTracker::default()));
    let device_key = Arc::as_ptr(&device) as usize;
    let host_key = Arc::as_ptr(&host) as usize;
    let admission = VulkanMemoryAdmission::from_test_partitioned_permits(
        vec![(
            device_key,
            VulkanMemoryAdmissionAllocationClass::PromptRunner,
            VulkanDeviceLocalMemoryPermit::acquire(&device, 1_000_000, 100_000).unwrap(),
        )],
        vec![(
            VulkanMemoryAdmissionAllocationClass::PromptRunner,
            host_key,
            VulkanHostMemoryPermit::acquire(&host, 1_000_000, 50_000).unwrap(),
        )],
    );

    let first_device_allocation;
    let first_host_allocation;
    {
        let _scope = admission.enter_prompt_runner();
        first_device_allocation = take_scoped_device_local_memory_capacity(&device, 100_000)
            .expect("device is in the reusable admission")
            .unwrap()
            .commit(100_000)
            .unwrap();
        first_host_allocation = take_scoped_host_memory_capacity(&host, 50_000)
            .expect("host is in the reusable admission")
            .unwrap()
            .commit(50_000)
            .unwrap();
    }
    admission.ensure_fully_consumed("first mount").unwrap();
    drop(first_device_allocation);
    drop(first_host_allocation);

    assert_eq!(admission.remaining_device_bytes(device_key), 100_000);
    assert_eq!(admission.remaining_host_bytes(), 50_000);
    assert_eq!(device.lock().unwrap().tracked_allocation_bytes, 0);
    assert_eq!(device.lock().unwrap().pending_reservation_bytes, 100_000);
    assert_eq!(host.lock().unwrap().tracked_allocation_bytes, 0);
    assert_eq!(host.lock().unwrap().pending_reservation_bytes, 50_000);

    let second_device_allocation;
    let second_host_allocation;
    {
        let _scope = admission.enter_prompt_runner();
        second_device_allocation = take_scoped_device_local_memory_capacity(&device, 100_000)
            .expect("recycled device capacity is reusable")
            .unwrap()
            .commit(100_000)
            .unwrap();
        second_host_allocation = take_scoped_host_memory_capacity(&host, 50_000)
            .expect("recycled host capacity is reusable")
            .unwrap()
            .commit(50_000)
            .unwrap();
    }
    admission.ensure_fully_consumed("second mount").unwrap();

    drop(admission);
    drop(second_device_allocation);
    drop(second_host_allocation);
    assert_eq!(device.lock().unwrap().pending_reservation_bytes, 0);
    assert_eq!(device.lock().unwrap().tracked_allocation_bytes, 0);
    assert_eq!(host.lock().unwrap().pending_reservation_bytes, 0);
    assert_eq!(host.lock().unwrap().tracked_allocation_bytes, 0);
}

#[test]
fn reusable_stream_memory_admission_rejects_partial_commit_without_losing_credit() {
    let budget = VulkanDeviceLocalMemoryBudget::capture(1_000_000);
    let device = Arc::new(Mutex::new(VulkanDeviceLocalMemoryBudgetTracker::new(
        budget,
    )));
    let host = Arc::new(Mutex::new(VulkanHostMemoryBudgetTracker::default()));
    let device_key = Arc::as_ptr(&device) as usize;
    let host_key = Arc::as_ptr(&host) as usize;
    let admission = VulkanMemoryAdmission::from_test_partitioned_permits(
        vec![(
            device_key,
            VulkanMemoryAdmissionAllocationClass::PromptRunner,
            VulkanDeviceLocalMemoryPermit::acquire(&device, 1_000_000, 100_000).unwrap(),
        )],
        vec![(
            VulkanMemoryAdmissionAllocationClass::PromptRunner,
            host_key,
            VulkanHostMemoryPermit::acquire(&host, 1_000_000, 50_000).unwrap(),
        )],
    );

    {
        let _scope = admission.enter_prompt_runner();
        let device_error = take_scoped_device_local_memory_capacity(&device, 100_000)
            .unwrap()
            .unwrap()
            .commit(99_999)
            .unwrap_err();
        let host_error = take_scoped_host_memory_capacity(&host, 50_000)
            .unwrap()
            .unwrap()
            .commit(49_999)
            .unwrap_err();
        assert!(device_error.to_string().contains("require exact physical consumption"));
        assert!(host_error.to_string().contains("require exact physical consumption"));
    }

    assert_eq!(admission.remaining_device_bytes(device_key), 100_000);
    assert_eq!(admission.remaining_host_bytes(), 50_000);
    assert_eq!(device.lock().unwrap().pending_reservation_bytes, 100_000);
    assert_eq!(device.lock().unwrap().tracked_allocation_bytes, 0);
    assert_eq!(host.lock().unwrap().pending_reservation_bytes, 50_000);
    assert_eq!(host.lock().unwrap().tracked_allocation_bytes, 0);
    drop(admission);
    assert_eq!(device.lock().unwrap().pending_reservation_bytes, 0);
    assert_eq!(host.lock().unwrap().pending_reservation_bytes, 0);
}

#[test]
fn classified_stream_memory_admission_never_borrows_between_runner_classes() {
    let budget = VulkanDeviceLocalMemoryBudget::capture(1_000_000);
    let device = Arc::new(Mutex::new(VulkanDeviceLocalMemoryBudgetTracker::new(
        budget,
    )));
    let host = Arc::new(Mutex::new(VulkanHostMemoryBudgetTracker::default()));
    let device_key = Arc::as_ptr(&device) as usize;
    let host_key = Arc::as_ptr(&host) as usize;
    let admission = VulkanMemoryAdmission::from_test_partitioned_permits(
        vec![
            (
                device_key,
                VulkanMemoryAdmissionAllocationClass::Permanent,
                VulkanDeviceLocalMemoryPermit::acquire(&device, 1_000_000, 100_000).unwrap(),
            ),
            (
                device_key,
                VulkanMemoryAdmissionAllocationClass::PromptRunner,
                VulkanDeviceLocalMemoryPermit::acquire(&device, 1_000_000, 200_000).unwrap(),
            ),
        ],
        vec![
            (
                VulkanMemoryAdmissionAllocationClass::Permanent,
                host_key,
                VulkanHostMemoryPermit::acquire(&host, 1_000_000, 50_000).unwrap(),
            ),
            (
                VulkanMemoryAdmissionAllocationClass::PromptRunner,
                host_key,
                VulkanHostMemoryPermit::acquire(&host, 1_000_000, 60_000).unwrap(),
            ),
        ],
    );

    let permanent_device_allocation;
    let permanent_host_allocation;
    {
        let _scope = admission.enter();
        permanent_device_allocation = take_scoped_device_local_memory_capacity(&device, 100_000)
            .unwrap()
            .unwrap()
            .commit(100_000)
            .unwrap();
        permanent_host_allocation = take_scoped_host_memory_capacity(&host, 50_000)
            .unwrap()
            .unwrap()
            .commit(50_000)
            .unwrap();
        assert!(
            take_scoped_device_local_memory_capacity(&device, 1)
                .unwrap()
                .is_err(),
            "permanent allocation must not borrow prompt-runner device credit",
        );
        assert!(
            take_scoped_host_memory_capacity(&host, 1)
                .unwrap()
                .is_err(),
            "permanent allocation must not borrow prompt-runner host credit",
        );
    }
    admission
        .ensure_class_fully_consumed(
            VulkanMemoryAdmissionAllocationClass::Permanent,
            "permanent fixture",
        )
        .unwrap();
    let prompt_error = admission
        .ensure_class_fully_consumed(
            VulkanMemoryAdmissionAllocationClass::PromptRunner,
            "prompt fixture",
        )
        .unwrap_err();
    assert!(prompt_error.to_string().contains("200000 device bytes"));
    assert!(prompt_error.to_string().contains("60000 host bytes"));
    assert_eq!(
        admission.remaining_device_bytes_for_class(
            device_key,
            VulkanMemoryAdmissionAllocationClass::Permanent,
        ),
        0,
    );
    assert_eq!(
        admission.remaining_device_bytes_for_class(
            device_key,
            VulkanMemoryAdmissionAllocationClass::PromptRunner,
        ),
        200_000,
    );

    let prompt_device_allocation;
    let prompt_host_allocation;
    {
        let _scope = admission.enter_prompt_runner();
        prompt_device_allocation = take_scoped_device_local_memory_capacity(&device, 200_000)
            .unwrap()
            .unwrap()
            .commit(200_000)
            .unwrap();
        prompt_host_allocation = take_scoped_host_memory_capacity(&host, 60_000)
            .unwrap()
            .unwrap()
            .commit(60_000)
            .unwrap();
    }
    admission.ensure_fully_consumed("complete fixture").unwrap();

    drop(prompt_device_allocation);
    drop(prompt_host_allocation);
    assert_eq!(
        admission.remaining_device_bytes_for_class(
            device_key,
            VulkanMemoryAdmissionAllocationClass::PromptRunner,
        ),
        200_000,
    );
    drop(admission);
    drop(permanent_device_allocation);
    drop(permanent_host_allocation);
    assert_eq!(device.lock().unwrap().pending_reservation_bytes, 0);
    assert_eq!(device.lock().unwrap().tracked_allocation_bytes, 0);
    assert_eq!(host.lock().unwrap().pending_reservation_bytes, 0);
    assert_eq!(host.lock().unwrap().tracked_allocation_bytes, 0);
}

#[test]
fn reusable_stream_memory_admission_isolates_every_lazy_runner_class() {
    let budget = VulkanDeviceLocalMemoryBudget::capture(1_000_000);
    let device = Arc::new(Mutex::new(VulkanDeviceLocalMemoryBudgetTracker::new(
        budget,
    )));
    let host = Arc::new(Mutex::new(VulkanHostMemoryBudgetTracker::default()));
    let device_key = Arc::as_ptr(&device) as usize;
    let host_key = Arc::as_ptr(&host) as usize;
    let classes: [(VulkanMemoryAdmissionAllocationClass, usize, usize); 3] = [
        (VulkanMemoryAdmissionAllocationClass::PromptRunner, 10_000, 11_000),
        (
            VulkanMemoryAdmissionAllocationClass::VerificationRunner,
            20_000,
            21_000,
        ),
        (
            VulkanMemoryAdmissionAllocationClass::CatchUpRunner,
            30_000,
            31_000,
        ),
    ];
    let admission = VulkanMemoryAdmission::from_test_partitioned_permits(
        classes
            .iter()
            .map(|(allocation_class, device_bytes, _)| {
                (
                    device_key,
                    *allocation_class,
                    VulkanDeviceLocalMemoryPermit::acquire(
                        &device,
                        1_000_000,
                        u64::try_from(*device_bytes).unwrap(),
                    )
                    .unwrap(),
                )
            })
            .collect(),
        classes
            .iter()
            .map(|(allocation_class, _, host_bytes)| {
                (
                    *allocation_class,
                    host_key,
                    VulkanHostMemoryPermit::acquire(&host, 1_000_000, *host_bytes).unwrap(),
                )
            })
            .collect(),
    );

    let mut allocations = Vec::new();
    for (allocation_class, device_bytes, host_bytes) in classes {
        let _scope = match allocation_class {
            VulkanMemoryAdmissionAllocationClass::PromptRunner => {
                admission.enter_prompt_runner()
            }
            VulkanMemoryAdmissionAllocationClass::VerificationRunner => {
                admission.enter_verification_runner()
            }
            VulkanMemoryAdmissionAllocationClass::CatchUpRunner => {
                admission.enter_catch_up_runner()
            }
            VulkanMemoryAdmissionAllocationClass::Permanent => unreachable!(),
        };
        let device_allocation = take_scoped_device_local_memory_capacity(&device, device_bytes)
            .unwrap()
            .unwrap()
            .commit(u64::try_from(device_bytes).unwrap())
            .unwrap();
        let host_allocation = take_scoped_host_memory_capacity(&host, host_bytes)
            .unwrap()
            .unwrap()
            .commit(host_bytes)
            .unwrap();
        assert!(
            take_scoped_device_local_memory_capacity(&device, 1)
                .unwrap()
                .is_err(),
            "{allocation_class:?} must not borrow another class's device credit",
        );
        assert!(
            take_scoped_host_memory_capacity(&host, 1)
                .unwrap()
                .is_err(),
            "{allocation_class:?} must not borrow another class's host credit",
        );
        admission
            .ensure_class_fully_consumed(allocation_class, "mounted lazy runner")
            .unwrap();
        allocations.push((device_allocation, host_allocation));
    }
    admission.ensure_fully_consumed("all lazy runners").unwrap();

    for (device_allocation, host_allocation) in allocations {
        drop(device_allocation);
        drop(host_allocation);
    }
    for (allocation_class, device_bytes, _) in classes {
        assert_eq!(
            admission.remaining_device_bytes_for_class(device_key, allocation_class),
            device_bytes,
        );
    }
    drop(admission);
    assert_eq!(device.lock().unwrap().pending_reservation_bytes, 0);
    assert_eq!(device.lock().unwrap().tracked_allocation_bytes, 0);
    assert_eq!(host.lock().unwrap().pending_reservation_bytes, 0);
    assert_eq!(host.lock().unwrap().tracked_allocation_bytes, 0);
}

#[test]
fn dropping_an_outer_stream_scope_preserves_the_active_inner_class() {
    let budget = VulkanDeviceLocalMemoryBudget::capture(1_000_000);
    let tracker = Arc::new(Mutex::new(VulkanDeviceLocalMemoryBudgetTracker::new(
        budget,
    )));
    let key = Arc::as_ptr(&tracker) as usize;
    let admission = VulkanMemoryAdmission::from_test_partitioned_permits(
        vec![
            (
                key,
                VulkanMemoryAdmissionAllocationClass::Permanent,
                VulkanDeviceLocalMemoryPermit::acquire(&tracker, 1_000_000, 10_000).unwrap(),
            ),
            (
                key,
                VulkanMemoryAdmissionAllocationClass::PromptRunner,
                VulkanDeviceLocalMemoryPermit::acquire(&tracker, 1_000_000, 20_000).unwrap(),
            ),
        ],
        Vec::new(),
    );

    let outer = admission.enter();
    let inner = admission.enter_prompt_runner();
    drop(outer);
    let prompt_allocation = take_scoped_device_local_memory_capacity(&tracker, 20_000)
        .expect("the inner prompt scope remains active")
        .unwrap()
        .commit(20_000)
        .unwrap();
    assert_eq!(
        admission.remaining_device_bytes_for_class(
            key,
            VulkanMemoryAdmissionAllocationClass::Permanent,
        ),
        10_000,
    );
    assert_eq!(
        admission.remaining_device_bytes_for_class(
            key,
            VulkanMemoryAdmissionAllocationClass::PromptRunner,
        ),
        0,
    );

    drop(inner);
    assert!(take_scoped_device_local_memory_capacity(&tracker, 1).is_none());
    drop(prompt_allocation);
    drop(admission);
    assert_eq!(tracker.lock().unwrap().pending_reservation_bytes, 0);
    assert_eq!(tracker.lock().unwrap().tracked_allocation_bytes, 0);
}

#[test]
fn host_memory_permits_contend_and_roll_back_without_leaking_capacity() {
    let tracker = Arc::new(Mutex::new(VulkanHostMemoryBudgetTracker::default()));
    let first = VulkanHostMemoryPermit::acquire(&tracker, 1_000_000, 600_000).unwrap();
    let error = VulkanHostMemoryPermit::acquire(&tracker, 1_000_000, 400_001)
        .expect_err("pending host transactions must contend atomically");
    assert!(error.to_string().contains("600000 pending"));

    let second = VulkanHostMemoryPermit::acquire(&tracker, 1_000_000, 400_000).unwrap();
    drop(first);
    assert_eq!(tracker.lock().unwrap().pending_reservation_bytes, 400_000);
    let committed = second.commit(350_000).unwrap();
    assert_eq!(tracker.lock().unwrap().pending_reservation_bytes, 0);
    assert_eq!(tracker.lock().unwrap().tracked_allocation_bytes, 350_000);
    drop(committed);
    assert_eq!(tracker.lock().unwrap().tracked_allocation_bytes, 0);
}

#[test]
fn host_memory_permit_accounts_committed_bytes_against_a_falling_live_snapshot() {
    let tracker = Arc::new(Mutex::new(VulkanHostMemoryBudgetTracker::default()));
    let first = VulkanHostMemoryPermit::acquire(&tracker, 1_000_000, 600_000)
        .unwrap()
        .commit(600_000)
        .unwrap();

    let second = VulkanHostMemoryPermit::acquire(&tracker, 400_000, 400_000).unwrap();
    assert_eq!(tracker.lock().unwrap().tracked_allocation_bytes, 600_000);
    assert_eq!(tracker.lock().unwrap().pending_reservation_bytes, 400_000);
    assert!(VulkanHostMemoryPermit::acquire(&tracker, 400_000, 1).is_err());

    drop(second);
    drop(first);
    assert_eq!(tracker.lock().unwrap().tracked_allocation_bytes, 0);
    assert_eq!(tracker.lock().unwrap().pending_reservation_bytes, 0);
}

#[test]
fn nested_stream_memory_admission_uses_the_innermost_transaction() {
    let budget = VulkanDeviceLocalMemoryBudget::capture(1_000_000);
    let tracker = Arc::new(Mutex::new(VulkanDeviceLocalMemoryBudgetTracker::new(
        budget,
    )));
    let key = Arc::as_ptr(&tracker) as usize;
    let outer = VulkanMemoryAdmission::from_test_permits(
        vec![(
            key,
            VulkanDeviceLocalMemoryPermit::acquire(&tracker, 1_000_000, 300_000).unwrap(),
        )],
        None,
    );
    let inner = VulkanMemoryAdmission::from_test_permits(
        vec![(
            key,
            VulkanDeviceLocalMemoryPermit::acquire(&tracker, 1_000_000, 200_000).unwrap(),
        )],
        None,
    );

    let _outer_scope = outer.enter();
    {
        let _inner_scope = inner.enter();
        drop(
            take_scoped_device_local_memory_capacity(&tracker, 50_000)
                .unwrap()
                .unwrap(),
        );
    }
    assert_eq!(inner.remaining_device_bytes(key), 150_000);
    assert_eq!(outer.remaining_device_bytes(key), 300_000);
    drop(
        take_scoped_device_local_memory_capacity(&tracker, 25_000)
            .unwrap()
            .unwrap(),
    );
    assert_eq!(outer.remaining_device_bytes(key), 275_000);
}

#[test]
fn nested_stream_memory_admission_never_borrows_an_outer_participant() {
    let budget = VulkanDeviceLocalMemoryBudget::capture(1_000_000);
    let outer_tracker = Arc::new(Mutex::new(VulkanDeviceLocalMemoryBudgetTracker::new(
        budget,
    )));
    let inner_tracker = Arc::new(Mutex::new(VulkanDeviceLocalMemoryBudgetTracker::new(
        budget,
    )));
    let host = Arc::new(Mutex::new(VulkanHostMemoryBudgetTracker::default()));
    let outer_key = Arc::as_ptr(&outer_tracker) as usize;
    let inner_key = Arc::as_ptr(&inner_tracker) as usize;
    let host_key = Arc::as_ptr(&host) as usize;
    let outer = VulkanMemoryAdmission::from_test_permits(
        vec![(
            outer_key,
            VulkanDeviceLocalMemoryPermit::acquire(&outer_tracker, 1_000_000, 300_000)
                .unwrap(),
        )],
        Some((
            host_key,
            VulkanHostMemoryPermit::acquire(&host, 1_000_000, 100_000).unwrap(),
        )),
    );
    let inner = VulkanMemoryAdmission::from_test_permits(
        vec![(
            inner_key,
            VulkanDeviceLocalMemoryPermit::acquire(&inner_tracker, 1_000_000, 200_000)
                .unwrap(),
        )],
        None,
    );

    let _outer_scope = outer.enter();
    let _inner_scope = inner.enter();
    let device_error = take_scoped_device_local_memory_capacity(&outer_tracker, 1)
        .unwrap()
        .unwrap_err();
    let host_error = take_scoped_host_memory_capacity(&host, 1)
        .unwrap()
        .unwrap_err();
    assert!(device_error.to_string().contains("active stream admission"));
    assert!(host_error.to_string().contains("no shared-host"));
    assert_eq!(outer.remaining_device_bytes(outer_key), 300_000);
    assert_eq!(outer.remaining_host_bytes(), 100_000);
}

#[test]
fn stream_memory_admission_rejects_unplanned_allocation_without_spending_credit() {
    let budget = VulkanDeviceLocalMemoryBudget::capture(1_000_000);
    let tracker = Arc::new(Mutex::new(VulkanDeviceLocalMemoryBudgetTracker::new(
        budget,
    )));
    let key = Arc::as_ptr(&tracker) as usize;
    let admission = VulkanMemoryAdmission::from_test_permits(
        vec![(
            key,
            VulkanDeviceLocalMemoryPermit::acquire(&tracker, 1_000_000, 100_000).unwrap(),
        )],
        None,
    );

    let _scope = admission.enter();
    let error = take_scoped_device_local_memory_capacity(&tracker, 100_001)
        .unwrap()
        .unwrap_err();
    assert!(error.to_string().contains("cannot provide a 100001-byte child"));
    assert_eq!(admission.remaining_device_bytes(key), 100_000);
    assert_eq!(tracker.lock().unwrap().pending_reservation_bytes, 100_000);
    assert_eq!(tracker.lock().unwrap().tracked_allocation_bytes, 0);
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
pub(crate) fn with_test_device_local_memory_quiescence<T>(
    operation: impl FnOnce(&VulkanDeviceLocalMemoryQuiescence<'_>) -> T,
) -> T {
    let memory_lifecycle = std::sync::RwLock::new(());
    let quiescence = VulkanDeviceLocalMemoryQuiescence {
        _memory_lifecycle: memory_lifecycle.write().unwrap(),
    };
    operation(&quiescence)
}
