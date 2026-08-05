use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc as SyncArc, Barrier};
use std::thread;

struct TestResidentPayload {
    bytes: usize,
    drops: SyncArc<AtomicUsize>,
}

impl DeviceResidentResourcePayload for TestResidentPayload {
    fn byte_count(&self) -> usize {
        self.bytes
    }
}

impl Drop for TestResidentPayload {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

fn residency_descriptor(
    group_digit: char,
    resource_digit: char,
    byte_count: usize,
) -> DeviceResourceGroupDescriptor {
    let group_id = format!("sha256:{}", group_digit.to_string().repeat(64));
    let resource_id =
        format!("sha256:{}", resource_digit.to_string().repeat(64));
    DeviceResourceGroupDescriptor::new(
        group_id,
        vec![resource_id.clone()],
        Vec::new(),
        vec![DeviceResourceDescriptor {
            id: resource_id,
            byte_count,
            compatibility: CompiledResourceCompatibility {
                device_api: "vulkan".to_string(),
                storage_class: "storage_buffer".to_string(),
                read_only: true,
                required_features: Vec::new(),
            },
        }],
    )
    .unwrap()
}

fn resident_test_group(
    descriptor: DeviceResourceGroupDescriptor,
    drops: SyncArc<AtomicUsize>,
) -> DeviceResidentResourceGroup<TestResidentPayload> {
    let resource = descriptor.resources[0].clone();
    let byte_count = resource.byte_count;
    DeviceResidentResourceGroup::new(
        descriptor,
        vec![
            DeviceResidentResource::new(
                resource,
                TestResidentPayload {
                    bytes: byte_count,
                    drops,
                },
            )
            .unwrap(),
        ],
    )
    .unwrap()
}

fn owner(value: &str) -> DeviceResourceResidencyOwnerId {
    DeviceResourceResidencyOwnerId::new(value).unwrap()
}

#[test]
fn resolved_descriptor_uses_derived_resident_size_not_packed_source_size() {
    let required_features = vec![
        "shader_float8".to_string(),
        "shader_int8".to_string(),
        "shader_mixed_float_dot_product_float8_acc_float32".to_string(),
    ];
    let resource_id = format!("sha256:{}", "1".repeat(64));
    let resolved = ResolvedCompiledResourceGroup::Atomic(
        ResolvedCompiledAtomicGroup {
            schema: RESOLVED_ATOMIC_GROUP_SCHEMA.to_string(),
            id: format!("sha256:{}", "2".repeat(64)),
            resource_ids: vec![resource_id.clone()],
            dependencies: Vec::new(),
            resources: vec![ResolvedCompiledResource {
                id: resource_id,
                ranges: vec![ResolvedCompiledResourceRange {
                    artifact_path: "weights.bin".to_string(),
                    byte_offset: 0,
                    byte_count: 8,
                    alignment_bytes: 8,
                    sha256: "0".repeat(64),
                }],
                compatibility: CompiledResourceCompatibility {
                    device_api: "vulkan".to_string(),
                    storage_class: "storage_buffer".to_string(),
                    read_only: true,
                    required_features: required_features.clone(),
                },
                resident_derivation: Some(CompiledResourceResidentDerivation {
                    schema: RESIDENT_DERIVATION_SCHEMA.to_string(),
                    kind: CompiledResourceResidentDerivationKind::Mxfp4E2m1ToFp8E4m3,
                    source_byte_count: 8,
                    resident_byte_count: 16,
                    required_features,
                }),
            }],
        },
    );

    let descriptor = DeviceResourceGroupDescriptor::from_resolved(&resolved).unwrap();

    assert_eq!(descriptor.byte_count, 16);
    assert_eq!(descriptor.resources[0].byte_count, 16);
}

#[test]
fn per_device_residency_single_flight_shares_one_atomic_publication() {
    const CALLER_COUNT: usize = 8;
    let manager = DeviceResourceResidencyManager::<TestResidentPayload>::new(
        "gpu0", 4096, 512,
    )
    .unwrap();
    let descriptor = residency_descriptor('1', '2', 256);
    let request_barrier = SyncArc::new(Barrier::new(CALLER_COUNT));
    let load_count = SyncArc::new(AtomicUsize::new(0));
    let drops = SyncArc::new(AtomicUsize::new(0));

    let leases = thread::scope(|scope| {
        let mut callers = Vec::new();
        for caller_index in 0..CALLER_COUNT {
            let manager = manager.clone();
            let descriptor = descriptor.clone();
            let request_barrier = SyncArc::clone(&request_barrier);
            let load_count = SyncArc::clone(&load_count);
            let drops = SyncArc::clone(&drops);
            callers.push(scope.spawn(move || {
                let request = manager
                    .request(
                        descriptor.clone(),
                        owner(&format!("graph-{caller_index}")),
                    )
                    .unwrap();
                request_barrier.wait();
                match request {
                    DeviceResourceResidencyRequest::LoadRequired(permit) => {
                        load_count.fetch_add(1, Ordering::Relaxed);
                        permit
                            .publish(resident_test_group(descriptor, drops))
                            .unwrap()
                    }
                    DeviceResourceResidencyRequest::Pending(waiter) => {
                        waiter.wait().unwrap()
                    }
                    DeviceResourceResidencyRequest::Resident(lease) => lease,
                }
            }));
        }
        callers
            .into_iter()
            .map(|caller| caller.join().unwrap())
            .collect::<Vec<_>>()
    });

    assert_eq!(load_count.load(Ordering::Relaxed), 1);
    assert!(
        leases
            .windows(2)
            .all(|pair| pair[0].shares_publication_with(&pair[1]))
    );
    let stats = manager.statistics().unwrap();
    assert_eq!(stats.resident_group_count, 1);
    assert_eq!(stats.dynamic_resident_bytes, 256);
    assert_eq!(stats.high_water_resident_group_count, 1);
    assert_eq!(stats.high_water_dynamic_resident_bytes, 256);
    assert_eq!(stats.single_flight_join_count, CALLER_COUNT as u64 - 1);
    let snapshot = manager.snapshot().unwrap();
    assert_eq!(snapshot.statistics, stats);
    assert_eq!(snapshot.directory.len(), 1);
    assert_eq!(
        snapshot.directory[0].state,
        ResourceResidencyState::Resident
    );
    assert_eq!(snapshot.directory[0].byte_count, 256);
    assert_eq!(manager.directory().unwrap()[0].owner_count, CALLER_COUNT);
    drop(leases);
    for caller_index in 0..CALLER_COUNT {
        manager
            .unload_owner(&owner(&format!("graph-{caller_index}")))
            .unwrap();
    }
    let unloaded = manager.statistics().unwrap();
    assert_eq!(unloaded.dynamic_resident_bytes, 0);
    assert_eq!(unloaded.high_water_dynamic_resident_bytes, 256);
    assert_eq!(unloaded.high_water_resident_group_count, 1);
    assert_eq!(drops.load(Ordering::Relaxed), 1);
}

#[test]
fn per_device_residency_shares_parameters_but_not_mutable_stream_state() {
    let manager = DeviceResourceResidencyManager::<TestResidentPayload>::new(
        "gpu0", 4096, 512,
    )
    .unwrap();
    let descriptor = residency_descriptor('3', '4', 128);
    let drops = SyncArc::new(AtomicUsize::new(0));
    let first = match manager
        .request(descriptor.clone(), owner("rewired-a"))
        .unwrap()
    {
        DeviceResourceResidencyRequest::LoadRequired(permit) => permit
            .publish(resident_test_group(
                descriptor.clone(),
                SyncArc::clone(&drops),
            ))
            .unwrap(),
        _ => panic!("first request did not own the load"),
    };
    let second = match manager
        .request(descriptor, owner("rewired-b"))
        .unwrap()
    {
        DeviceResourceResidencyRequest::Resident(lease) => lease,
        _ => panic!("second graph instance did not hit resident parameters"),
    };
    let mut first_stream_state = vec![1u32, 2, 3];
    let second_stream_state = vec![1u32, 2, 3];
    first_stream_state[0] = 99;

    assert!(first.shares_publication_with(&second));
    assert_ne!(first_stream_state, second_stream_state);
    assert_eq!(
        manager
            .unload_owner(&owner("rewired-a"))
            .err()
            .unwrap()
            .kind(),
        DeviceResourceResidencyErrorKind::InUse
    );
    drop(first);
    assert_eq!(manager.unload_owner(&owner("rewired-a")).unwrap().group_count, 0);
    assert_eq!(manager.statistics().unwrap().resident_group_count, 1);
    drop(second);
    assert_eq!(manager.unload_owner(&owner("rewired-b")).unwrap().group_count, 1);
    assert_eq!(drops.load(Ordering::Relaxed), 1);
}

#[test]
fn per_device_residency_rejects_capacity_before_atomic_loading_and_rolls_back() {
    let manager = DeviceResourceResidencyManager::<TestResidentPayload>::new(
        "gpu0", 100, 20,
    )
    .unwrap();
    let too_large = residency_descriptor('5', '6', 81);
    let error = manager.request(too_large, owner("model")).err().unwrap();
    assert_eq!(error.kind(), DeviceResourceResidencyErrorKind::Capacity);
    assert_eq!(
        manager.statistics().unwrap(),
        DeviceResourceResidencyStatistics {
            capacity_bytes: 100,
            always_resident_bytes: 20,
            ..Default::default()
        }
    );
    let mut tampered = residency_descriptor('6', '7', 80);
    tampered.byte_count = 1;
    assert_eq!(
        manager
            .request(tampered, owner("model"))
            .err()
            .unwrap()
            .kind(),
        DeviceResourceResidencyErrorKind::InvalidDescriptor
    );

    let descriptor = residency_descriptor('7', '8', 80);
    let permit = match manager
        .request(descriptor.clone(), owner("model"))
        .unwrap()
    {
        DeviceResourceResidencyRequest::LoadRequired(permit) => permit,
        _ => panic!("capacity-fitting request did not own the load"),
    };
    let wrong_descriptor = residency_descriptor('9', 'a', 80);
    let error = permit
        .publish(resident_test_group(
            wrong_descriptor,
            SyncArc::new(AtomicUsize::new(0)),
        ))
        .err()
        .unwrap();
    assert_eq!(
        error.kind(),
        DeviceResourceResidencyErrorKind::InvalidPublication
    );
    let stats = manager.statistics().unwrap();
    assert_eq!(stats.reserved_loading_bytes, 0);
    assert_eq!(stats.dynamic_resident_bytes, 0);
    assert_eq!(stats.failed_group_count, 1);
    manager.reset_failed_group(&descriptor.id).unwrap();
    assert!(manager.directory().unwrap().is_empty());
}

#[test]
fn demand_retained_package_can_exceed_capacity_until_its_observed_working_set_does() {
    let manager = DeviceResourceResidencyManager::<TestResidentPayload>::new(
        "gpu0", 100, 20,
    )
    .unwrap();
    let descriptors = [
        residency_descriptor('1', '2', 40),
        residency_descriptor('3', '4', 40),
        residency_descriptor('5', '6', 40),
    ];
    let declared_package_bytes = descriptors
        .iter()
        .map(|descriptor| descriptor.byte_count)
        .sum::<usize>();
    assert_eq!(declared_package_bytes, 120);
    assert!(declared_package_bytes > 100 - 20);

    let drops = SyncArc::new(AtomicUsize::new(0));
    let mut leases = Vec::new();
    for descriptor in &descriptors[..2] {
        let permit = match manager
            .request(descriptor.clone(), owner("model"))
            .unwrap()
        {
            DeviceResourceResidencyRequest::LoadRequired(permit) => permit,
            _ => panic!("first access to a fitting resource did not own its load"),
        };
        leases.push(
            permit
                .publish(resident_test_group(
                    descriptor.clone(),
                    SyncArc::clone(&drops),
                ))
                .unwrap(),
        );
    }
    let fitting = manager.statistics().unwrap();
    assert_eq!(fitting.dynamic_resident_bytes, 80);
    assert_eq!(fitting.resident_group_count, 2);
    assert_eq!(fitting.failed_group_count, 0);

    let error = manager
        .request(descriptors[2].clone(), owner("model"))
        .err()
        .expect("working-set growth beyond capacity must fail");
    assert_eq!(error.kind(), DeviceResourceResidencyErrorKind::Capacity);
    let failed_growth = manager.statistics().unwrap();
    assert_eq!(failed_growth.dynamic_resident_bytes, 80);
    assert_eq!(failed_growth.reserved_loading_bytes, 0);
    assert_eq!(failed_growth.resident_group_count, 2);
    assert_eq!(failed_growth.failed_group_count, 0);

    drop(leases);
    let released = manager.unload_owner(&owner("model")).unwrap();
    assert_eq!(released.group_count, 2);
    assert_eq!(released.byte_count, 80);
    assert_eq!(drops.load(Ordering::Relaxed), 2);
    assert!(manager.directory().unwrap().is_empty());
}

#[test]
fn demand_residency_evicts_the_least_recently_used_inactive_group_and_observes_reload() {
    let manager = DeviceResourceResidencyManager::<TestResidentPayload>::new(
        "gpu0", 100, 0,
    )
    .unwrap();
    let first = residency_descriptor('a', '1', 50);
    let second = residency_descriptor('b', '2', 50);
    let drops = SyncArc::new(AtomicUsize::new(0));

    for descriptor in [&first, &second] {
        let permit = match manager
            .request(descriptor.clone(), owner("conversation"))
            .unwrap()
        {
            DeviceResourceResidencyRequest::LoadRequired(permit) => permit,
            _ => panic!("first access did not require a load"),
        };
        drop(
            permit
                .publish(resident_test_group(
                    descriptor.clone(),
                    SyncArc::clone(&drops),
                ))
                .unwrap(),
        );
    }

    drop(
        match manager
            .request(first.clone(), owner("conversation"))
            .unwrap()
        {
            DeviceResourceResidencyRequest::Resident(lease) => lease,
            _ => panic!("resident group was not reused"),
        },
    );
    let candidates = manager.eviction_candidates(&BTreeSet::new()).unwrap();
    assert_eq!(
        candidates
            .iter()
            .map(|candidate| candidate.group_id.as_str())
            .collect::<Vec<_>>(),
        vec![second.id.as_str(), first.id.as_str()],
    );

    let eviction = manager
        .evict_inactive_groups(BTreeSet::from([second.id.clone()]))
        .unwrap();
    assert_eq!(eviction.release().group_count, 1);
    assert_eq!(eviction.release().byte_count, 50);
    assert_eq!(drops.load(Ordering::Relaxed), 0);
    assert_eq!(manager.statistics().unwrap().dynamic_resident_bytes, 50);
    drop(eviction);
    assert_eq!(drops.load(Ordering::Relaxed), 1);

    let permit = match manager
        .request(second.clone(), owner("conversation"))
        .unwrap()
    {
        DeviceResourceResidencyRequest::LoadRequired(permit) => permit,
        _ => panic!("evicted group was not loaded again"),
    };
    drop(
        permit
            .publish(resident_test_group(second, SyncArc::clone(&drops)))
            .unwrap(),
    );
    let stats = manager.statistics().unwrap();
    assert_eq!(stats.eviction_count, 1);
    assert_eq!(stats.evicted_group_count, 1);
    assert_eq!(stats.evicted_byte_count, 50);
    assert_eq!(stats.reload_count, 1);
}

#[test]
fn demand_residency_refuses_an_atomic_eviction_when_any_group_is_in_use() {
    let manager = DeviceResourceResidencyManager::<TestResidentPayload>::new(
        "gpu0", 100, 0,
    )
    .unwrap();
    let first = residency_descriptor('c', '3', 50);
    let second = residency_descriptor('d', '4', 50);
    let drops = SyncArc::new(AtomicUsize::new(0));
    let active = match manager
        .request(first.clone(), owner("conversation"))
        .unwrap()
    {
        DeviceResourceResidencyRequest::LoadRequired(permit) => permit
            .publish(resident_test_group(
                first.clone(),
                SyncArc::clone(&drops),
            ))
            .unwrap(),
        _ => panic!("first access did not require a load"),
    };
    let second_lease = match manager
        .request(second.clone(), owner("conversation"))
        .unwrap()
    {
        DeviceResourceResidencyRequest::LoadRequired(permit) => permit
            .publish(resident_test_group(
                second.clone(),
                SyncArc::clone(&drops),
            ))
            .unwrap(),
        _ => panic!("first access did not require a load"),
    };
    drop(second_lease);

    let error = manager
        .evict_inactive_groups(BTreeSet::from([
            first.id.clone(),
            second.id.clone(),
        ]))
        .err()
        .expect("an active execution lease must prevent eviction");
    assert_eq!(error.kind(), DeviceResourceResidencyErrorKind::InUse);
    assert_eq!(manager.statistics().unwrap().resident_group_count, 2);
    assert_eq!(manager.statistics().unwrap().eviction_count, 0);
    assert_eq!(drops.load(Ordering::Relaxed), 0);
    drop(active);
}

#[test]
fn per_device_residency_cancellation_and_failure_wake_waiters_cleanly() {
    let manager = DeviceResourceResidencyManager::<TestResidentPayload>::new(
        "gpu0", 4096, 512,
    )
    .unwrap();
    let descriptor = residency_descriptor('b', 'c', 128);
    let leader = match manager
        .request(descriptor.clone(), owner("leader"))
        .unwrap()
    {
        DeviceResourceResidencyRequest::LoadRequired(permit) => permit,
        _ => panic!("first request did not own the load"),
    };
    let waiter = match manager
        .request(descriptor.clone(), owner("follower"))
        .unwrap()
    {
        DeviceResourceResidencyRequest::Pending(waiter) => waiter,
        _ => panic!("second request did not join the load"),
    };
    leader.cancel().unwrap();
    let cancellation = match waiter.wait() {
        Ok(_) => panic!("cancelled follower received a resident group"),
        Err(error) => error,
    };
    assert_eq!(
        cancellation.kind(),
        DeviceResourceResidencyErrorKind::Cancelled
    );
    assert!(manager.directory().unwrap().is_empty());
    assert_eq!(manager.statistics().unwrap().reserved_loading_bytes, 0);

    let leader = match manager
        .request(descriptor.clone(), owner("leader"))
        .unwrap()
    {
        DeviceResourceResidencyRequest::LoadRequired(permit) => permit,
        _ => panic!("retry did not own the load"),
    };
    let failure = DeviceResourceResidencyError::load_failed("read failed");
    leader.fail(failure.clone()).unwrap();
    assert_eq!(
        manager
            .request(descriptor.clone(), owner("follower"))
            .err()
            .unwrap(),
        failure
    );
    manager.reset_failed_group(&descriptor.id).unwrap();
    let retry = manager.request(descriptor, owner("leader")).unwrap();
    assert!(matches!(
        retry,
        DeviceResourceResidencyRequest::LoadRequired(_)
    ));
    drop(retry);
    assert_eq!(manager.statistics().unwrap().reserved_loading_bytes, 0);
}

#[test]
fn per_device_residency_owner_unload_cancels_only_that_pending_request() {
    let manager = DeviceResourceResidencyManager::<TestResidentPayload>::new(
        "gpu0", 4096, 512,
    )
    .unwrap();
    let descriptor = residency_descriptor('a', 'b', 128);
    let drops = SyncArc::new(AtomicUsize::new(0));
    let leader = match manager
        .request(descriptor.clone(), owner("leader"))
        .unwrap()
    {
        DeviceResourceResidencyRequest::LoadRequired(permit) => permit,
        _ => panic!("first request did not own the load"),
    };
    let waiter = match manager
        .request(descriptor.clone(), owner("cancelled-follower"))
        .unwrap()
    {
        DeviceResourceResidencyRequest::Pending(waiter) => waiter,
        _ => panic!("second request did not join the load"),
    };
    assert_eq!(
        manager
            .unload_owner(&owner("cancelled-follower"))
            .unwrap()
            .cancelled_load_count,
        0
    );
    let leader_lease = leader
        .publish(resident_test_group(
            descriptor,
            SyncArc::clone(&drops),
        ))
        .unwrap();
    let error = match waiter.wait() {
        Ok(_) => panic!("unloaded follower received a residency lease"),
        Err(error) => error,
    };
    assert_eq!(error.kind(), DeviceResourceResidencyErrorKind::Cancelled);
    assert_eq!(manager.directory().unwrap()[0].owner_count, 1);
    drop(leader_lease);
    assert_eq!(manager.unload_owner(&owner("leader")).unwrap().group_count, 1);
    assert_eq!(drops.load(Ordering::Relaxed), 1);

    let descriptor = residency_descriptor('c', 'd', 128);
    let leader = match manager
        .request(descriptor.clone(), owner("cancelled-leader"))
        .unwrap()
    {
        DeviceResourceResidencyRequest::LoadRequired(permit) => permit,
        _ => panic!("second leader did not own the load"),
    };
    let waiter = match manager
        .request(descriptor.clone(), owner("surviving-follower"))
        .unwrap()
    {
        DeviceResourceResidencyRequest::Pending(waiter) => waiter,
        _ => panic!("surviving follower did not join the load"),
    };
    manager
        .unload_owner(&owner("cancelled-leader"))
        .unwrap();
    let error = leader
        .publish(resident_test_group(
            descriptor,
            SyncArc::clone(&drops),
        ))
        .err()
        .unwrap();
    assert_eq!(error.kind(), DeviceResourceResidencyErrorKind::Cancelled);
    let follower_lease = waiter.wait().unwrap();
    assert_eq!(manager.directory().unwrap()[0].owner_count, 1);
    drop(follower_lease);
    assert_eq!(
        manager
            .unload_owner(&owner("surviving-follower"))
            .unwrap()
            .group_count,
        1
    );
    assert_eq!(drops.load(Ordering::Relaxed), 2);
}

#[test]
fn per_device_residency_explicit_unload_refuses_live_leases_and_leaks_nothing() {
    let drops = SyncArc::new(AtomicUsize::new(0));
    let manager = DeviceResourceResidencyManager::<TestResidentPayload>::new(
        "gpu0", 4096, 512,
    )
    .unwrap();
    let descriptor = residency_descriptor('d', 'e', 256);
    let lease = match manager
        .request(descriptor.clone(), owner("slice"))
        .unwrap()
    {
        DeviceResourceResidencyRequest::LoadRequired(permit) => permit
            .publish(resident_test_group(
                descriptor,
                SyncArc::clone(&drops),
            ))
            .unwrap(),
        _ => panic!("first request did not own the load"),
    };

    assert_eq!(
        manager.unload_owner(&owner("slice")).err().unwrap().kind(),
        DeviceResourceResidencyErrorKind::InUse
    );
    assert_eq!(manager.statistics().unwrap().dynamic_resident_bytes, 256);
    drop(lease);
    let release = manager.unload_owner(&owner("slice")).unwrap();
    assert_eq!(release.group_count, 1);
    assert_eq!(release.byte_count, 256);
    assert_eq!(drops.load(Ordering::Relaxed), 1);
    assert!(manager.directory().unwrap().is_empty());
}

#[test]
fn per_device_residency_transforms_inactive_group_payload_ownership_atomically() {
    let manager = DeviceResourceResidencyManager::<TestResidentPayload>::new(
        "gpu0", 4096, 512,
    )
    .unwrap();
    let left_descriptor = residency_descriptor('1', '2', 128);
    let right_descriptor = residency_descriptor('3', '4', 128);
    let left_drops = SyncArc::new(AtomicUsize::new(0));
    let right_drops = SyncArc::new(AtomicUsize::new(0));
    for (descriptor, drops) in [
        (left_descriptor.clone(), SyncArc::clone(&left_drops)),
        (right_descriptor.clone(), SyncArc::clone(&right_drops)),
    ] {
        let request = manager.request(descriptor.clone(), owner("model")).unwrap();
        let DeviceResourceResidencyRequest::LoadRequired(permit) = request else {
            panic!("new group did not own its load")
        };
        drop(permit.publish(resident_test_group(descriptor, drops)).unwrap());
    }

    manager
        .transform_inactive_resident_groups(
            &left_descriptor.id,
            &right_descriptor.id,
            |left, right| {
                let rebuild = |logical: &DeviceResidentResourceGroup<TestResidentPayload>,
                               storage: &DeviceResidentResourceGroup<TestResidentPayload>| {
                    let descriptor = logical.descriptor().clone();
                    let resource = descriptor.resources[0].clone();
                    DeviceResidentResourceGroup::new(
                        descriptor,
                        vec![DeviceResidentResource::new(
                            resource,
                            TestResidentPayload {
                                bytes: storage.resources()[0].payload().bytes,
                                drops: SyncArc::clone(
                                    &storage.resources()[0].payload().drops,
                                ),
                            },
                        )?],
                    )
                };
                Ok((rebuild(left, right)?, rebuild(right, left)?))
            },
        )
        .unwrap();

    let left_lease = match manager
        .request(left_descriptor.clone(), owner("left-reader"))
        .unwrap()
    {
        DeviceResourceResidencyRequest::Resident(lease) => lease,
        _ => panic!("transformed left group stopped being resident"),
    };
    assert!(SyncArc::ptr_eq(
        &left_lease.group().resources()[0].payload().drops,
        &right_drops
    ));
    let error = manager
        .transform_inactive_resident_groups(
            &left_descriptor.id,
            &right_descriptor.id,
            |_, _| unreachable!("active leases must be rejected before transformation"),
        )
        .unwrap_err();
    assert_eq!(error.kind(), DeviceResourceResidencyErrorKind::InUse);
    drop(left_lease);

    let right_lease = match manager
        .request(right_descriptor, owner("right-reader"))
        .unwrap()
    {
        DeviceResourceResidencyRequest::Resident(lease) => lease,
        _ => panic!("transformed right group stopped being resident"),
    };
    assert!(SyncArc::ptr_eq(
        &right_lease.group().resources()[0].payload().drops,
        &left_drops
    ));
    drop(right_lease);
    assert_eq!(manager.statistics().unwrap().dynamic_resident_bytes, 256);
    assert_eq!(manager.unload_device().unwrap().byte_count, 256);
}

#[test]
fn per_device_residency_device_unload_releases_resident_and_loading_groups() {
    let drops = SyncArc::new(AtomicUsize::new(0));
    let manager = DeviceResourceResidencyManager::<TestResidentPayload>::new(
        "gpu0", 4096, 512,
    )
    .unwrap();
    for (group_digit, resource_digit) in [('1', '2'), ('3', '4')] {
        let descriptor =
            residency_descriptor(group_digit, resource_digit, 128);
        let lease = match manager
            .request(
                descriptor.clone(),
                owner(&format!("model-{group_digit}")),
            )
            .unwrap()
        {
            DeviceResourceResidencyRequest::LoadRequired(permit) => permit
                .publish(resident_test_group(
                    descriptor,
                    SyncArc::clone(&drops),
                ))
                .unwrap(),
            _ => panic!("first group request did not own the load"),
        };
        drop(lease);
    }
    let loading_descriptor = residency_descriptor('5', '6', 128);
    let loading = match manager
        .request(loading_descriptor.clone(), owner("loading-owner"))
        .unwrap()
    {
        DeviceResourceResidencyRequest::LoadRequired(permit) => permit,
        _ => panic!("loading group request did not own the load"),
    };
    let waiter = match manager
        .request(loading_descriptor, owner("loading-follower"))
        .unwrap()
    {
        DeviceResourceResidencyRequest::Pending(waiter) => waiter,
        _ => panic!("loading follower did not join the load"),
    };

    let release = manager.unload_device().unwrap();
    assert_eq!(release.group_count, 2);
    assert_eq!(release.byte_count, 256);
    assert_eq!(release.cancelled_load_count, 1);
    let error = match waiter.wait() {
        Ok(_) => panic!("device-unloaded waiter received a residency lease"),
        Err(error) => error,
    };
    assert_eq!(error.kind(), DeviceResourceResidencyErrorKind::Cancelled);
    drop(loading);
    assert!(manager.directory().unwrap().is_empty());
    let stats = manager.statistics().unwrap();
    assert_eq!(stats.dynamic_resident_bytes, 0);
    assert_eq!(stats.reserved_loading_bytes, 0);
    assert_eq!(drops.load(Ordering::Relaxed), 2);
}

#[test]
fn per_device_residency_directories_never_alias_physical_devices() {
    let first = DeviceResourceResidencyManager::<TestResidentPayload>::new(
        "gpu0", 4096, 512,
    )
    .unwrap();
    let second = DeviceResourceResidencyManager::<TestResidentPayload>::new(
        "gpu1", 4096, 512,
    )
    .unwrap();
    let descriptor = residency_descriptor('f', '0', 128);
    let first_request = first
        .request(descriptor.clone(), owner("model"))
        .unwrap();
    let second_request = second
        .request(descriptor, owner("model"))
        .unwrap();

    assert!(matches!(
        first_request,
        DeviceResourceResidencyRequest::LoadRequired(_)
    ));
    assert!(matches!(
        second_request,
        DeviceResourceResidencyRequest::LoadRequired(_)
    ));
    assert_eq!(
        first.directory().unwrap()[0].location,
        DeviceResourceResidencyLocation::Local {
            device_id: "gpu0".to_string()
        }
    );
    assert_eq!(
        second.directory().unwrap()[0].location,
        DeviceResourceResidencyLocation::Local {
            device_id: "gpu1".to_string()
        }
    );
    drop(first_request);
    drop(second_request);
    assert!(first.directory().unwrap().is_empty());
    assert!(second.directory().unwrap().is_empty());
}

#[test]
fn stable_resource_upload_capacity_failure_rolls_back_every_allocation_and_publication(
) {
    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping stable upload rollback test: {error}");
            return;
        }
    };
    let root =
        crate::test_support::TempDir::new("stable_upload_rollback");
    fs::write(root.path().join("weights.bin"), b"abcdefghABCDEFGH").unwrap();
    let compatibility = CompiledResourceCompatibility {
        device_api: "vulkan".to_string(),
        storage_class: "storage_buffer".to_string(),
        read_only: true,
        required_features: vec!["buffer_device_address".to_string()],
    };
    let resources = [
        (0usize, &b"abcdefgh"[..]),
        (8, &b"ABCDEFGH"[..]),
    ]
        .into_iter()
        .enumerate()
        .map(|(resource_index, (byte_offset, bytes))| {
            ResolvedCompiledResource {
                id: format!(
                    "sha256:{}",
                    char::from(b'1' + resource_index as u8)
                        .to_string()
                        .repeat(64)
                ),
                ranges: vec![ResolvedCompiledResourceRange {
                    artifact_path: "weights.bin".to_string(),
                    byte_offset,
                    byte_count: bytes.len(),
                    alignment_bytes: 8,
                    sha256: format!("{:x}", Sha256::digest(bytes)),
                }],
                compatibility: compatibility.clone(),
                resident_derivation: None,
            }
        })
        .collect::<Vec<_>>();
    let group = ResolvedCompiledPartitionGroup {
        schema: RESOLVED_PARTITION_GROUP_SCHEMA.to_string(),
        partition_template_id: format!("sha256:{}", "3".repeat(64)),
        partition_index: 0,
        id: format!("sha256:{}", "4".repeat(64)),
        resource_ids: resources
            .iter()
            .map(|resource| resource.id.clone())
            .collect(),
        dependencies: Vec::new(),
        resources,
    };
    let resolved = ResolvedCompiledResourceGroup::Partition(group);
    let descriptor =
        DeviceResourceGroupDescriptor::from_resolved(&resolved).unwrap();
    let backing_store = CompiledResourceBackingStore::new(
        root.path(),
        CompiledResourceBackingStoreLimits {
            worker_count: 1,
            queued_request_capacity: 1,
            maximum_ranges_per_group: 2,
            maximum_logical_bytes_per_group: 16,
            maximum_retained_payload_bytes: 16,
            maximum_coalesced_read_bytes: 16,
            maximum_coalescing_gap_bytes: 0,
        },
    )
    .unwrap();
    let loaded = backing_store.try_load(resolved).unwrap().wait().unwrap();
    let mut transfer =
        device.create_resident_transfer_stream(1, 64).unwrap();
    let arena = VulkanStableResourceArena::new(
        &device,
        VulkanStableResourceArenaConfig::new(8, 8).unwrap(),
        &[VulkanStableResourceGroupLayout::Explicit {
            resource_slots: vec![0, 1],
            resource_byte_counts: vec![8, 8],
        }],
    )
    .unwrap();
    let mut address_table =
        VulkanStableResourceAddressTable::new(&device, &mut transfer, 2)
            .unwrap();

    let error = match
        upload_loaded_compiled_resource_group_to_stable_address_space(
            &device,
            &mut transfer,
            &arena,
            &mut address_table,
            &descriptor,
            &loaded,
            &[0, 1],
            8,
        )
    {
        Ok(_) => panic!("capacity-constrained stable upload succeeded"),
        Err(error) => error,
    };

    assert!(
        error.0.contains(
            "sparse stable resources need "
        ) && error.0.contains(
            " additional physical bytes, but 0 of 8 bytes are already committed"
        ),
        "{error}"
    );
    assert_eq!(
        arena.stats().unwrap(),
        VulkanStableResourceArenaStats::default()
    );
    assert!(
        (0..address_table.slot_count())
            .all(|slot| address_table.record(slot).unwrap().resident == 0)
    );
    drop(loaded);
    assert_eq!(backing_store.retained_payload_bytes(), 0);
    arena.release_backing().unwrap();
}

#[test]
fn external_compiled_group_has_one_device_load_and_explicit_release() {
    let package_root = match std::env::var("NERVE_TEST_COMPILED_PACKAGE_ROOT") {
        Ok(path) => PathBuf::from(path),
        Err(std::env::VarError::NotPresent) => {
            eprintln!(
                "skipping external device residency load: package root is not set"
            );
            return;
        }
        Err(error) => panic!("could not read external package root: {error}"),
    };
    let device_index = std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX")
        .expect("NERVE_TEST_VULKAN_DEVICE_INDEX must select a compatible AMD GPU with sufficient safe remaining capacity")
        .parse::<usize>()
        .expect("NERVE_TEST_VULKAN_DEVICE_INDEX must be an integer");
    let manifest: VulkanResidentModelPackageManifest = serde_json::from_slice(
        &fs::read(package_root.join("vulkan_resident_package.json")).unwrap(),
    )
    .unwrap();
    let template = &manifest.resource_residency.partition_templates[0];
    let resolved = resolve_compiled_partition_group(
        &package_root,
        &manifest.resource_residency,
        &template.id,
        0,
    )
    .unwrap();
    let resolved = ResolvedCompiledResourceGroup::Partition(resolved);
    let descriptor =
        DeviceResourceGroupDescriptor::from_resolved(&resolved).unwrap();
    let store = CompiledResourceBackingStore::new(
        &package_root,
        CompiledResourceBackingStoreLimits {
            maximum_ranges_per_group: 256,
            maximum_logical_bytes_per_group: 128 * 1024 * 1024,
            maximum_coalesced_read_bytes: 32 * 1024 * 1024,
            ..Default::default()
        },
    )
    .unwrap();
    let loaded = store.try_load(resolved).unwrap().wait().unwrap();
    let device =
        VulkanComputeDevice::new_for_physical_device_index(device_index)
            .unwrap();
    let mut transfer = device
        .create_resident_transfer_stream(2, loaded.logical_byte_count)
        .unwrap();
    let manager =
        DeviceResourceResidencyManager::<VulkanResidentCompiledResource>::new(
            "gpu0",
            loaded.logical_byte_count,
            0,
        )
        .unwrap();
    let first = match manager
        .request(descriptor.clone(), owner("model-a"))
        .unwrap()
    {
        DeviceResourceResidencyRequest::LoadRequired(permit) => {
            let resident = upload_loaded_compiled_resource_group(
                &device,
                &mut transfer,
                &descriptor,
                &loaded,
            )
            .unwrap();
            permit.publish(resident).unwrap()
        }
        _ => panic!("first external request did not own the physical load"),
    };
    let second = match manager
        .request(descriptor, owner("model-b"))
        .unwrap()
    {
        DeviceResourceResidencyRequest::Resident(lease) => lease,
        _ => panic!("second external request did not reuse residency"),
    };

    assert!(first.shares_publication_with(&second));
    for (resident, expected) in
        first.group().resources().iter().zip(&loaded.resources)
    {
        let bytes = resident
            .payload()
            .buffer()
            .read_bytes(resident.payload().byte_count())
            .unwrap();
        for (placement, range) in
            resident.payload().ranges().iter().zip(&expected.ranges)
        {
            assert_eq!(
                &bytes[placement.byte_offset
                    ..placement.byte_offset + placement.byte_count],
                &*range.bytes
            );
        }
    }
    assert_eq!(manager.statistics().unwrap().successful_load_count, 1);
    assert_eq!(manager.statistics().unwrap().hit_count, 1);
    drop(first);
    drop(second);
    assert_eq!(
        manager.unload_owner(&owner("model-a")).unwrap().group_count,
        0
    );
    let release = manager.unload_owner(&owner("model-b")).unwrap();
    assert_eq!(release.group_count, 1);
    assert_eq!(release.byte_count, loaded.logical_byte_count);
    assert_eq!(manager.statistics().unwrap().dynamic_resident_bytes, 0);
    assert!(manager.directory().unwrap().is_empty());
}

#[test]
fn external_compiled_group_uses_stable_address_slots_and_explicit_retirement() {
    let package_root = match std::env::var("NERVE_TEST_COMPILED_PACKAGE_ROOT") {
        Ok(path) => PathBuf::from(path),
        Err(std::env::VarError::NotPresent) => {
            eprintln!(
                "skipping stable external device residency load: package root is not set"
            );
            return;
        }
        Err(error) => panic!("could not read external package root: {error}"),
    };
    let device_index = std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX")
        .expect("NERVE_TEST_VULKAN_DEVICE_INDEX must select a compatible AMD GPU with sufficient safe remaining capacity")
        .parse::<usize>()
        .expect("NERVE_TEST_VULKAN_DEVICE_INDEX must be an integer");
    let manifest: VulkanResidentModelPackageManifest = serde_json::from_slice(
        &fs::read(package_root.join("vulkan_resident_package.json")).unwrap(),
    )
    .unwrap();
    let template = &manifest.resource_residency.partition_templates[0];
    let resolved = ResolvedCompiledResourceGroup::Partition(
        resolve_compiled_partition_group(
            &package_root,
            &manifest.resource_residency,
            &template.id,
            0,
        )
        .unwrap(),
    );
    let descriptor =
        DeviceResourceGroupDescriptor::from_resolved(&resolved).unwrap();
    let store = CompiledResourceBackingStore::new(
        &package_root,
        CompiledResourceBackingStoreLimits {
            maximum_ranges_per_group: 256,
            maximum_logical_bytes_per_group: 128 * 1024 * 1024,
            maximum_coalesced_read_bytes: 32 * 1024 * 1024,
            ..Default::default()
        },
    )
    .unwrap();
    let loaded = store.try_load(resolved).unwrap().wait().unwrap();
    let device =
        VulkanComputeDevice::new_for_physical_device_index(device_index)
            .unwrap();
    let mut transfer = device
        .create_resident_transfer_stream(2, loaded.logical_byte_count)
        .unwrap();
    let alignment = 256usize;
    let capacity = loaded
        .logical_byte_count
        .checked_add(
            descriptor
                .resources
                .len()
                .checked_mul(alignment - 1)
                .unwrap(),
        )
        .unwrap();
    let arena = VulkanStableResourceArena::new(
        &device,
        VulkanStableResourceArenaConfig::new(
            capacity.max(128 * 1024 * 1024),
            alignment,
        )
        .unwrap(),
        &[VulkanStableResourceGroupLayout::Explicit {
            resource_slots: (0..descriptor.resources.len()).collect(),
            resource_byte_counts: descriptor
                .resources
                .iter()
                .map(|resource| resource.byte_count)
                .collect(),
        }],
    )
    .unwrap();
    let mut table = VulkanStableResourceAddressTable::new(
        &device,
        &mut transfer,
        descriptor.resources.len(),
    )
    .unwrap();
    let slots = (0..descriptor.resources.len()).collect::<Vec<_>>();

    let upload = upload_loaded_compiled_resource_group_to_stable_address_space(
        &device,
        &mut transfer,
        &arena,
        &mut table,
        &descriptor,
        &loaded,
        &slots,
        alignment,
    )
    .unwrap();

    assert_eq!(upload.resident_group().descriptor(), &descriptor);
    assert_eq!(upload.publications().len(), descriptor.resources.len());
    for (slot, resident) in upload
        .resident_group()
        .resources()
        .iter()
        .enumerate()
    {
        let address = resident.payload().stable_device_address().unwrap();
        assert_eq!(address % alignment as u64, 0);
        assert_eq!(table.record(slot).unwrap().device_address, address);
        assert_eq!(table.record(slot).unwrap().resident, 1);
        let buffer_address =
            resident.payload().buffer().device_address().unwrap();
        assert_eq!(
            resident.payload().ranges()[0].byte_offset as u64,
            address - buffer_address
        );
    }
    assert_eq!(
        arena.stats().unwrap().active_allocation_count,
        descriptor.resources.len()
    );

    upload.retire(&mut transfer, &mut table).unwrap();
    assert_eq!(arena.stats().unwrap().active_allocation_count, 0);
    arena.release_backing().unwrap();
    assert_eq!(arena.stats().unwrap(), VulkanStableResourceArenaStats::default());
    assert!(
        (0..table.slot_count())
            .all(|slot| table.record(slot).unwrap().resident == 0)
    );
}
