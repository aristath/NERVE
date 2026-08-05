fn compiled_store_process_file_descriptor_count() -> usize {
    fs::read_dir("/proc/self/fd").unwrap().count()
}

fn compiled_store_worker_thread_count() -> usize {
    fs::read_dir("/proc/self/task")
        .unwrap()
        .map(|entry| entry.unwrap().path().join("comm"))
        .filter_map(|path| fs::read_to_string(path).ok())
        .filter(|name| name.trim().starts_with("nerve-resource"))
        .count()
}

#[test]
fn compiled_resource_backing_workers_follow_wave_width_without_oversubscribing_cores() {
    assert_eq!(
        compiled_resource_backing_worker_count_for_parallelism(8, 32),
        8
    );
    assert_eq!(
        compiled_resource_backing_worker_count_for_parallelism(32, 8),
        8
    );
    assert_eq!(
        compiled_resource_backing_worker_count_for_parallelism(1, 32),
        1
    );
    assert_eq!(
        compiled_resource_backing_worker_count_for_parallelism(0, 0),
        1
    );
}

#[test]
fn device_capacity_settlement_waits_for_the_driver_acknowledgement() {
    let mut remaining = VecDeque::from([1usize, 7, 32]);
    let acknowledged = wait_for_compiled_resource_device_capacity(
        16,
        Duration::from_secs(1),
        || Ok(remaining.pop_front().unwrap()),
    )
    .unwrap();

    assert_eq!(acknowledged, 32);
    assert!(remaining.is_empty());
}

#[test]
fn device_capacity_settlement_never_invents_capacity_at_the_deadline() {
    let error = wait_for_compiled_resource_device_capacity(16, Duration::ZERO, || Ok(15))
        .expect_err("unacknowledged or externally consumed capacity must remain unavailable");

    assert!(error.to_string().contains("did not settle"));
    assert!(error.to_string().contains("needs 16 bytes"));
    assert!(error.to_string().contains("only 15 bytes"));
}

#[test]
fn rebalanced_device_capacity_waits_for_driver_acknowledgement_before_reallocation() {
    let mut observations = [63usize, 64].into_iter();
    let acknowledged = wait_for_rebalanced_compiled_resource_device_capacity(
        64,
        64,
        Duration::from_millis(10),
        || Ok(observations.next().unwrap_or(64)),
    )
    .unwrap();

    assert_eq!(acknowledged, 64);
    assert_eq!(observations.next(), None);
}

#[test]
fn rebalanced_device_capacity_never_waits_past_the_arena_hard_bound() {
    let mut polled = false;
    let error = wait_for_rebalanced_compiled_resource_device_capacity(
        65,
        64,
        Duration::from_millis(10),
        || {
            polled = true;
            Ok(usize::MAX)
        },
    )
    .unwrap_err();

    assert!(!polled);
    assert!(error.to_string().contains("arena bytes"));
}

#[test]
fn exact_tiered_resource_plan_never_exceeds_either_memory_budget() {
    let groups = BTreeMap::from([
        ("identity-0".to_string(), 40usize),
        ("identity-1".to_string(), 40usize),
        ("identity-2".to_string(), 40usize),
        ("identity-3".to_string(), 40usize),
    ]);
    let plan = VulkanCompiledResourceMemoryPlan::exact_tiered(&groups, 100, 80).unwrap();
    assert_eq!(plan.device_payload_bytes, 80);
    assert_eq!(plan.host_visible_payload_bytes, 80);
    assert!(plan.device_payload_bytes <= 100);
    assert!(plan.host_visible_payload_bytes <= 80);
    assert_eq!(
        plan.tier_for_group("identity-0").unwrap(),
        VulkanCompiledResourceMemoryTier::Device
    );
    assert_eq!(
        plan.tier_for_group("identity-3").unwrap(),
        VulkanCompiledResourceMemoryTier::HostVisible
    );
}

#[test]
fn exact_tiered_resource_plan_rejects_host_overcommit_instead_of_falling_back() {
    let groups = BTreeMap::from([
        ("identity-0".to_string(), 60usize),
        ("identity-1".to_string(), 60usize),
        ("identity-2".to_string(), 60usize),
    ]);
    let error = VulkanCompiledResourceMemoryPlan::exact_tiered(&groups, 100, 60).unwrap_err();
    assert!(error.to_string().contains("need 120 host-visible"));
}

#[test]
fn exact_tiered_resource_plan_rejects_unknown_groups() {
    let plan = VulkanCompiledResourceMemoryPlan::exact_tiered(
        &BTreeMap::from([("known".to_string(), 8usize)]),
        8,
        8,
    )
    .unwrap();
    assert!(plan.tier_for_group("unknown").is_err());
}

#[test]
fn dynamic_tiered_plan_fills_device_then_retains_overflow_in_host_memory() {
    let groups = BTreeMap::from([
        ("identity-0".to_string(), 40usize),
        ("identity-1".to_string(), 40usize),
        ("identity-2".to_string(), 40usize),
        ("identity-3".to_string(), 40usize),
    ]);
    let mut plan = VulkanCompiledResourceMemoryPlan::dynamic_tiered(&groups, 100, 80).unwrap();

    let admission = plan
        .admit_groups(&[
            ("identity-0".to_string(), 40),
            ("identity-1".to_string(), 40),
            ("identity-2".to_string(), 40),
        ])
        .unwrap();

    assert_eq!(
        admission.tiers,
        vec![
            VulkanCompiledResourceMemoryTier::Device,
            VulkanCompiledResourceMemoryTier::Device,
            VulkanCompiledResourceMemoryTier::HostVisible,
        ]
    );
    assert_eq!(plan.device_payload_bytes, 80);
    assert_eq!(plan.host_visible_payload_bytes, 40);
    assert_eq!(plan.group_tiers.len(), 3);
}

#[test]
fn dynamic_tiered_admission_is_atomic_when_both_tiers_are_full() {
    let groups = BTreeMap::from([
        ("resident".to_string(), 60usize),
        ("would-fit".to_string(), 40usize),
        ("overflow".to_string(), 60usize),
    ]);
    let mut plan = VulkanCompiledResourceMemoryPlan::dynamic_tiered(&groups, 100, 40).unwrap();
    plan.admit_groups(&[("resident".to_string(), 60)]).unwrap();
    let before = plan.clone();

    let error = plan
        .admit_groups(&[
            ("would-fit".to_string(), 40),
            ("overflow".to_string(), 60),
        ])
        .unwrap_err();

    assert!(error.to_string().contains("cannot admit group"));
    assert_eq!(plan, before);
}

#[test]
fn dynamic_tiered_release_reuses_the_released_tier_capacity() {
    let groups = BTreeMap::from([
        ("device".to_string(), 80usize),
        ("host-old".to_string(), 40usize),
        ("host-new".to_string(), 40usize),
    ]);
    let mut plan = VulkanCompiledResourceMemoryPlan::dynamic_tiered(&groups, 80, 40).unwrap();
    plan.admit_groups(&[
        ("device".to_string(), 80),
        ("host-old".to_string(), 40),
    ])
    .unwrap();

    plan.release_dynamic_groups(&BTreeSet::from(["host-old".to_string()]))
        .unwrap();
    let admission = plan
        .admit_groups(&[("host-new".to_string(), 40)])
        .unwrap();

    assert_eq!(
        admission.tiers,
        vec![VulkanCompiledResourceMemoryTier::HostVisible]
    );
    assert_eq!(plan.device_payload_bytes, 80);
    assert_eq!(plan.host_visible_payload_bytes, 40);
    assert!(!plan.group_tiers.contains_key("host-old"));
    assert_eq!(
        plan.tier_for_group("host-new").unwrap(),
        VulkanCompiledResourceMemoryTier::HostVisible
    );
}

#[test]
fn dynamic_tiered_admission_rollback_preserves_preexisting_assignments() {
    let groups = BTreeMap::from([
        ("preexisting".to_string(), 80usize),
        ("provisional".to_string(), 40usize),
    ]);
    let mut plan = VulkanCompiledResourceMemoryPlan::dynamic_tiered(&groups, 80, 40).unwrap();
    plan.admit_groups(&[("preexisting".to_string(), 80)])
        .unwrap();

    let admission = plan
        .admit_groups(&[
            ("preexisting".to_string(), 80),
            ("provisional".to_string(), 40),
        ])
        .unwrap();
    plan.rollback_admission(&admission).unwrap();

    assert_eq!(plan.device_payload_bytes, 80);
    assert_eq!(plan.host_visible_payload_bytes, 0);
    assert_eq!(
        plan.tier_for_group("preexisting").unwrap(),
        VulkanCompiledResourceMemoryTier::Device
    );
    assert!(plan.tier_for_group("provisional").is_err());
}

#[test]
fn compiled_resource_wave_device_demand_excludes_host_tier_groups() {
    let admission = VulkanCompiledResourceWaveAdmission {
        group_tiers: BTreeMap::from([
            (
                "device".to_string(),
                VulkanCompiledResourceMemoryTier::Device,
            ),
            (
                "host".to_string(),
                VulkanCompiledResourceMemoryTier::HostVisible,
            ),
        ]),
        new_group_ids: BTreeSet::from(["device".to_string(), "host".to_string()]),
        newly_assigned_group_ids: BTreeSet::from([
            "device".to_string(),
            "host".to_string(),
        ]),
        new_payload_bytes: 120,
    };

    assert_eq!(
        compiled_resource_device_tier_group_ids(
            ["device", "host", "already-resident"],
            &admission,
        )
        .unwrap(),
        BTreeSet::from(["device".to_string()])
    );
}

#[test]
fn dynamic_tiered_admission_uses_host_before_nonpreferred_device_capacity() {
    let groups = BTreeMap::from([
        ("preferred".to_string(), 40usize),
        ("spill-a".to_string(), 40usize),
        ("spill-b".to_string(), 40usize),
    ]);
    let mut plan = VulkanCompiledResourceMemoryPlan::dynamic_tiered(&groups, 120, 80).unwrap();

    let admission = plan
        .admit_groups_with_device_preferences(
            &[
                ("preferred".to_string(), 40),
                ("spill-a".to_string(), 40),
                ("spill-b".to_string(), 40),
            ],
            &BTreeSet::from(["preferred".to_string()]),
        )
        .unwrap();

    assert_eq!(
        admission.tiers,
        vec![
            VulkanCompiledResourceMemoryTier::Device,
            VulkanCompiledResourceMemoryTier::HostVisible,
            VulkanCompiledResourceMemoryTier::HostVisible,
        ]
    );
    assert_eq!(plan.device_payload_bytes, 40);
    assert_eq!(plan.host_visible_payload_bytes, 80);
}

#[test]
fn dynamic_tiered_admission_uses_device_when_preferred_host_is_full() {
    let groups = BTreeMap::from([
        ("host-resident".to_string(), 40usize),
        ("new".to_string(), 40usize),
    ]);
    let mut plan = VulkanCompiledResourceMemoryPlan::dynamic_tiered(&groups, 40, 40).unwrap();
    plan.admit_groups_with_device_preferences(
        &[("host-resident".to_string(), 40)],
        &BTreeSet::new(),
    )
    .unwrap();

    let admission = plan
        .admit_groups_with_device_preferences(
            &[("new".to_string(), 40)],
            &BTreeSet::new(),
        )
        .unwrap();

    assert_eq!(
        admission.tiers,
        vec![VulkanCompiledResourceMemoryTier::Device]
    );
    assert_eq!(plan.device_payload_bytes, 40);
    assert_eq!(plan.host_visible_payload_bytes, 40);
}

#[test]
fn physical_wave_budget_selects_only_the_device_prefix_that_really_fits() {
    let groups = vec![
        ("first".to_string(), 40usize),
        ("second".to_string(), 40usize),
        ("third".to_string(), 40usize),
    ];
    let mut estimates = Vec::new();

    let preferred = compiled_resource_preferred_device_group_ids(
        &groups,
        120,
        70,
        |group_ids| {
            estimates.push(group_ids.clone());
            Ok(group_ids.len() * 40)
        },
    )
    .unwrap();

    assert_eq!(preferred, BTreeSet::from(["first".to_string()]));
    assert_eq!(
        estimates,
        vec![
            BTreeSet::from(["first".to_string()]),
            BTreeSet::from(["first".to_string(), "second".to_string()]),
            BTreeSet::from(["first".to_string(), "third".to_string()]),
        ]
    );
}

#[test]
fn tiered_host_memory_budget_preserves_explicit_system_headroom() {
    let capacity = parse_vulkan_host_memory_capacity(
        "MemTotal:       67108864 kB\nMemAvailable:   41943040 kB\n",
    )
    .unwrap();
    assert_eq!(capacity.total_bytes, 64 * 1024 * 1024 * 1024);
    assert_eq!(capacity.available_bytes, 40 * 1024 * 1024 * 1024);
    assert_eq!(
        capacity.safe_tiered_payload_bytes(),
        39 * 1024 * 1024 * 1024
    );
}

#[test]
fn tiered_host_memory_budget_rejects_missing_or_inconsistent_kernel_data() {
    assert!(parse_vulkan_host_memory_capacity("MemTotal: 1024 kB\n").is_err());
    assert!(
        parse_vulkan_host_memory_capacity("MemTotal: 1024 kB\nMemAvailable: 2048 kB\n").is_err()
    );
    assert!(
        parse_vulkan_host_memory_capacity("MemTotal: 1024 bytes\nMemAvailable: 512 kB\n").is_err()
    );
}

#[test]
fn demand_tiered_host_budget_is_shared_fairly_across_physical_stores() {
    let mut remaining = 390usize;
    let first = reserve_fair_vulkan_host_visible_payload_bytes(
        &mut remaining,
        3,
        220,
        0,
    );
    let second = reserve_fair_vulkan_host_visible_payload_bytes(
        &mut remaining,
        2,
        220,
        0,
    );
    let third = reserve_fair_vulkan_host_visible_payload_bytes(
        &mut remaining,
        1,
        220,
        0,
    );

    assert_eq!((first, second, third), (130, 130, 130));
    assert_eq!(remaining, 0);
}

#[test]
fn demand_tiered_host_budget_preserves_overhead_and_redistributes_slack() {
    let mut remaining = 110usize;
    let first = reserve_fair_vulkan_host_visible_payload_bytes(
        &mut remaining,
        2,
        20,
        5,
    );
    let second = reserve_fair_vulkan_host_visible_payload_bytes(
        &mut remaining,
        1,
        100,
        5,
    );

    assert_eq!(first, 20);
    assert_eq!(second, 80);
    assert_eq!(remaining, 0);
}

#[test]
fn paged_eviction_reclaims_complete_unprotected_allocation_cohorts() {
    let cohort = |chunk_id| VulkanCompiledResourceAllocationCohort {
        tier: VulkanCompiledResourceMemoryTier::Device,
        chunk_id,
    };
    let candidates = vec![
        DeviceResourceResidencyEvictionCandidate {
            group_id: "old-protected-sibling".to_string(),
            byte_count: 40,
            last_access_epoch: 1,
        },
        DeviceResourceResidencyEvictionCandidate {
            group_id: "next-a".to_string(),
            byte_count: 40,
            last_access_epoch: 3,
        },
        DeviceResourceResidencyEvictionCandidate {
            group_id: "next-b".to_string(),
            byte_count: 40,
            last_access_epoch: 4,
        },
    ];
    let group_chunks = BTreeMap::from([
        (
            "old-protected-sibling".to_string(),
            BTreeSet::from([cohort(10)]),
        ),
        ("protected".to_string(), BTreeSet::from([cohort(10)])),
        ("next-a".to_string(), BTreeSet::from([cohort(20)])),
        ("next-b".to_string(), BTreeSet::from([cohort(20)])),
    ]);
    let chunk_groups = BTreeMap::from([
        (
            cohort(10),
            BTreeSet::from(["old-protected-sibling".to_string(), "protected".to_string()]),
        ),
        (
            cohort(20),
            BTreeSet::from(["next-a".to_string(), "next-b".to_string()]),
        ),
    ]);

    let selected = compiled_resource_lru_eviction_groups(
        &candidates,
        &group_chunks,
        &chunk_groups,
        &BTreeMap::new(),
        &BTreeSet::from(["protected".to_string()]),
        60,
        0,
    )
    .unwrap();

    assert_eq!(
        selected,
        BTreeSet::from(["next-a".to_string(), "next-b".to_string()])
    );
}

#[test]
fn physical_device_pressure_evicts_device_local_not_host_visible_cohorts() {
    let cohort = |tier, chunk_id| VulkanCompiledResourceAllocationCohort { tier, chunk_id };
    let host = cohort(VulkanCompiledResourceMemoryTier::HostVisible, 10);
    let device = cohort(VulkanCompiledResourceMemoryTier::Device, 20);
    let candidates = vec![
        DeviceResourceResidencyEvictionCandidate {
            group_id: "host-oldest".to_string(),
            byte_count: 80,
            last_access_epoch: 1,
        },
        DeviceResourceResidencyEvictionCandidate {
            group_id: "device-newer".to_string(),
            byte_count: 60,
            last_access_epoch: 2,
        },
    ];
    let group_chunks = BTreeMap::from([
        ("host-oldest".to_string(), BTreeSet::from([host])),
        ("device-newer".to_string(), BTreeSet::from([device])),
    ]);
    let chunk_groups = BTreeMap::from([
        (host, BTreeSet::from(["host-oldest".to_string()])),
        (device, BTreeSet::from(["device-newer".to_string()])),
    ]);
    let capacities = BTreeMap::from([(host, 96), (device, 64)]);

    let selected = compiled_resource_lru_eviction_groups(
        &candidates,
        &group_chunks,
        &chunk_groups,
        &capacities,
        &BTreeSet::new(),
        0,
        64,
    )
    .unwrap();

    assert_eq!(selected, BTreeSet::from(["device-newer".to_string()]));
}

#[test]
fn global_capacity_reclamation_returns_all_available_device_cohorts_when_one_store_is_insufficient()
{
    let cohort = VulkanCompiledResourceAllocationCohort {
        tier: VulkanCompiledResourceMemoryTier::Device,
        chunk_id: 20,
    };
    let candidates = vec![DeviceResourceResidencyEvictionCandidate {
        group_id: "available".to_string(),
        byte_count: 60,
        last_access_epoch: 2,
    }];
    let group_chunks =
        BTreeMap::from([("available".to_string(), BTreeSet::from([cohort]))]);
    let chunk_groups =
        BTreeMap::from([(cohort, BTreeSet::from(["available".to_string()]))]);
    let capacities = BTreeMap::from([(cohort, 64)]);

    let selection = compiled_resource_lru_eviction_selection(
        &candidates,
        &group_chunks,
        &chunk_groups,
        &capacities,
        &BTreeSet::new(),
        0,
        128,
    )
    .unwrap();

    assert_eq!(selection.group_ids, BTreeSet::from(["available".to_string()]));
    assert_eq!(selection.device_bytes, 64);
    assert_eq!(selection.payload_bytes, 60);
}

#[test]
fn eviction_closes_transitively_over_shared_allocation_chunks() {
    let cohort = |chunk_id| VulkanCompiledResourceAllocationCohort {
        tier: VulkanCompiledResourceMemoryTier::Device,
        chunk_id,
    };
    let first = cohort(10);
    let bridge = cohort(20);
    let last = cohort(30);
    let candidates = ["a", "b", "c"]
        .into_iter()
        .enumerate()
        .map(|(index, group_id)| DeviceResourceResidencyEvictionCandidate {
            group_id: group_id.to_string(),
            byte_count: 16,
            last_access_epoch: index as u64,
        })
        .collect::<Vec<_>>();
    let group_chunks = BTreeMap::from([
        ("a".to_string(), BTreeSet::from([first, bridge])),
        ("b".to_string(), BTreeSet::from([bridge, last])),
        ("c".to_string(), BTreeSet::from([last])),
    ]);
    let chunk_groups = BTreeMap::from([
        (first, BTreeSet::from(["a".to_string()])),
        (bridge, BTreeSet::from(["a".to_string(), "b".to_string()])),
        (last, BTreeSet::from(["b".to_string(), "c".to_string()])),
    ]);
    let capacities = BTreeMap::from([(first, 16), (bridge, 32), (last, 32)]);

    let selected = compiled_resource_lru_eviction_groups(
        &candidates,
        &group_chunks,
        &chunk_groups,
        &capacities,
        &BTreeSet::new(),
        1,
        1,
    )
    .unwrap();

    assert_eq!(
        selected,
        BTreeSet::from(["a".to_string(), "b".to_string(), "c".to_string()])
    );
}

#[test]
fn paged_eviction_reclaims_selector_overage_before_an_unrelated_working_set() {
    let candidates = vec![
        DeviceResourceResidencyEvictionCandidate {
            group_id: "under-budget-old".to_string(),
            byte_count: 40,
            last_access_epoch: 1,
        },
        DeviceResourceResidencyEvictionCandidate {
            group_id: "borrowed-newer".to_string(),
            byte_count: 140,
            last_access_epoch: 20,
        },
    ];
    let directory = vec![
        DeviceResourceResidencyDirectoryEntry {
            group_id: "under-budget-old".to_string(),
            state: ResourceResidencyState::Resident,
            location: DeviceResourceResidencyLocation::Local {
                device_id: "gpu0".to_string(),
            },
            byte_count: 40,
            owner_count: 1,
            active_lease_count: 0,
            last_access_epoch: 1,
        },
        DeviceResourceResidencyDirectoryEntry {
            group_id: "borrowed-newer".to_string(),
            state: ResourceResidencyState::Resident,
            location: DeviceResourceResidencyLocation::Local {
                device_id: "gpu0".to_string(),
            },
            byte_count: 140,
            owner_count: 1,
            active_lease_count: 0,
            last_access_epoch: 20,
        },
    ];
    let group_selector_ids = BTreeMap::from([
        ("under-budget-old".to_string(), "layer-a".to_string()),
        ("borrowed-newer".to_string(), "layer-b".to_string()),
    ]);
    let selector_payload_budgets =
        BTreeMap::from([("layer-a".to_string(), 100), ("layer-b".to_string(), 100)]);

    let ordered = compiled_resource_selector_fair_eviction_candidates(
        &candidates,
        &directory,
        &group_selector_ids,
        &selector_payload_budgets,
        "layer-a",
        40,
    )
    .unwrap();

    assert_eq!(ordered[0].group_id, "borrowed-newer");
    assert_eq!(ordered[1].group_id, "under-budget-old");
}

#[test]
fn compiled_resource_device_store_loads_reuses_and_retires_stable_resources() {
    let device = selected_test_vulkan_device().expect("selected Vulkan test device must open");
    let root = crate::test_support::TempDir::new("compiled_resource_device_store");
    let weight_bytes = b"abcdefghABCDEFGH";
    fs::write(root.path().join("weights.bin"), weight_bytes).unwrap();
    let mut digest_table = Vec::new();
    digest_table.extend_from_slice(&Sha256::digest(&weight_bytes[..8]));
    digest_table.extend_from_slice(&Sha256::digest(&weight_bytes[8..]));
    fs::write(root.path().join("digests.bin"), &digest_table).unwrap();

    let content_id = |byte: char| format!("sha256:{}", byte.to_string().repeat(64));
    let template_id = content_id('1');
    let member_seed = content_id('2');
    let selector_id = content_id('3');
    let mut contract = CompiledResourceResidencyContract {
        schema: COMPILED_RESOURCE_RESIDENCY_SCHEMA.to_string(),
        identity_algorithm: RESOURCE_IDENTITY_ALGORITHM.to_string(),
        state_machine_schema: RESOURCE_RESIDENCY_STATE_MACHINE_SCHEMA.to_string(),
        supported_policies: vec![
            ResourceResidencyPolicy::DemandRetained,
            ResourceResidencyPolicy::Eager,
        ],
        resources: Vec::new(),
        atomic_groups: Vec::new(),
        partition_templates: vec![CompiledPartitionTemplate {
            id: template_id.clone(),
            partition_count: 2,
            lifetime: CompiledResourceLifetime::Dynamic,
            group_identity_seed: content_id('4'),
            member_templates: vec![CompiledPartitionMemberTemplate {
                resource_identity_seed: member_seed.clone(),
                range_templates: vec![CompiledResourceRangeTemplate {
                    artifact_path: "weights.bin".to_string(),
                    base_byte_offset: 0,
                    stride_bytes: 8,
                    byte_count: 8,
                    alignment_bytes: 8,
                    integrity: CompiledResourceRangeIntegrityTemplate {
                        algorithm: "sha256_table".to_string(),
                        digest_table_path: "digests.bin".to_string(),
                        digest_table_byte_offset: 0,
                        digest_stride_bytes: 32,
                        table_sha256: Sha256::digest(&digest_table)
                            .iter()
                            .map(|byte| format!("{byte:02x}"))
                            .collect(),
                    },
                }],
                compatibility: CompiledResourceCompatibility {
                    device_api: "vulkan".to_string(),
                    storage_class: "storage_buffer".to_string(),
                    read_only: true,
                    required_features: vec!["buffer_device_address".to_string()],
                },
                resident_derivation: None,
            }],
            dependencies: Vec::new(),
        }],
        bindings: vec![CompiledResourceBinding {
            execution_scope: "target".to_string(),
            component_id: "component".to_string(),
            node_id: "selected_compute".to_string(),
            parameter_id: "bank".to_string(),
            mapping: CompiledResourceBindingMapping::PartitionTemplateMember {
                partition_template_id: template_id.clone(),
                resource_identity_seed: member_seed,
                selection_signal: "selected".to_string(),
                parameter_slot: 0,
            },
        }],
        selectors: vec![CompiledResourceSelector {
            id: selector_id.clone(),
            execution_scope: "target".to_string(),
            component_id: "component".to_string(),
            node_id: "choose".to_string(),
            domain_id: "resources".to_string(),
            resource_count: 2,
            selection_signal: "selected".to_string(),
            encoding: CompiledResourceSelectionEncoding {
                element_type: CompiledResourceSelectionElementType::U32,
                selection_count_per_activation: 2,
                index_shift: 0,
                index_mask: 0xffff,
            },
            mapping: CompiledResourceSelectorMapping::PartitionTemplate {
                partition_template_id: template_id,
            },
        }],
        checkpoints: Vec::new(),
    };
    let alias_selector_id = content_id('5');
    let mut alias_selector = contract.selectors[0].clone();
    alias_selector.id = alias_selector_id.clone();
    alias_selector.component_id = "component_repeat".to_string();
    contract.selectors.push(alias_selector);
    let unowned_selector_id = content_id('6');
    let mut unowned_selector = contract.selectors[0].clone();
    unowned_selector.id = unowned_selector_id.clone();
    unowned_selector.component_id = "other_component".to_string();
    contract.selectors.push(unowned_selector);
    let inspection = contract.inspection_report().unwrap();
    assert_eq!(inspection.supported_policies, ["demand-retained", "eager"]);
    assert_eq!(inspection.always_resident.unit_count, 0);
    assert_eq!(inspection.dynamically_addressable.unit_count, 2);
    assert_eq!(inspection.dynamically_addressable.resource_count, 2);
    assert_eq!(inspection.dynamically_addressable.maximum_payload_bytes, 16);
    assert_eq!(inspection.scopes.len(), 1);
    assert_eq!(inspection.scopes[0].component_count, 3);
    assert_eq!(inspection.scopes[0].selector_count, 3);
    assert_eq!(inspection.scopes[0].addressable_unit_count, 2);
    let contract = Arc::new(contract);
    let layout = Arc::new(VulkanCompiledResourceAddressLayout::from_contract(&contract).unwrap());
    let capacity_error = match VulkanCompiledResourceDeviceStore::new(
        &device,
        ResourceResidencyPolicy::DemandRetained,
        "amd-test",
        device.physical_device_id(),
        vec!["gpu0".to_string()],
        root.path(),
        Arc::clone(&contract),
        Arc::clone(&layout),
        BTreeSet::from([selector_id.clone(), alias_selector_id.clone()]),
        4096,
        4096,
        1024,
        1,
        128,
        64,
        layout.address_table_byte_count().unwrap(),
    ) {
        Ok(_) => panic!("physical allocation padding was not admitted"),
        Err(error) => error,
    };
    assert!(
        capacity_error
            .to_string()
            .contains("physical allocation bytes")
    );
    let over_capacity_store = VulkanCompiledResourceDeviceStore::new(
        &device,
        ResourceResidencyPolicy::DemandRetained,
        "amd-over-capacity-test",
        device.physical_device_id(),
        vec!["gpu0".to_string()],
        root.path(),
        Arc::clone(&contract),
        Arc::clone(&layout),
        BTreeSet::from([selector_id.clone()]),
        8,
        256 * 1024,
        8,
        1,
        128,
        64,
        layout.address_table_byte_count().unwrap(),
    )
    .unwrap();
    over_capacity_store.mark_mount_complete().unwrap();
    let over_capacity_owner = DeviceResourceResidencyOwnerId::new("over-capacity-graph").unwrap();
    over_capacity_store
        .load_selector_resource(&device, &selector_id, 0, over_capacity_owner.clone())
        .unwrap();
    let fitting_working_set = over_capacity_store.residency_report().unwrap();
    assert_eq!(fitting_working_set.maximum_payload_bytes, 8);
    assert_eq!(fitting_working_set.current_payload_bytes, 8);
    assert_eq!(fitting_working_set.resident_unit_count, 1);
    assert_eq!(fitting_working_set.addressable_unit_count, 2);
    let capacity_error = over_capacity_store
        .load_selector_resource(&device, &selector_id, 1, over_capacity_owner.clone())
        .expect_err("demand-retained residency must never evict a loaded component");
    assert!(capacity_error.to_string().contains("capacity is 8 bytes"));
    let retained = over_capacity_store.residency_report().unwrap();
    assert_eq!(retained.current_payload_bytes, 8);
    assert_eq!(retained.resident_unit_count, 1);
    assert_eq!(retained.failed_unit_count, 0);
    assert_eq!(retained.eviction_count, 0);
    assert_eq!(retained.evicted_unit_count, 0);
    assert_eq!(retained.evicted_payload_bytes, 0);
    assert_eq!(retained.released_device_bytes, 0);
    over_capacity_store
        .load_selector_resource(&device, &selector_id, 0, over_capacity_owner)
        .expect("the retained component must remain addressable after admission fails");
    let reused = over_capacity_store.residency_report().unwrap();
    assert_eq!(reused.eviction_count, 0);
    assert_eq!(reused.reload_count, 0);
    assert_eq!(
        over_capacity_store.unload().unwrap(),
        DeviceResourceResidencyRelease {
            group_count: 1,
            byte_count: 8,
            cancelled_load_count: 0,
        }
    );
    drop(over_capacity_store);

    let paged_store = Arc::new(VulkanCompiledResourceDeviceStore::new(
        &device,
        ResourceResidencyPolicy::DemandPaged,
        "amd-paged-test",
        device.physical_device_id(),
        vec!["gpu0".to_string()],
        root.path(),
        Arc::clone(&contract),
        Arc::clone(&layout),
        BTreeSet::from([selector_id.clone()]),
        8,
        256 * 1024,
        8,
        1,
        128,
        64,
        layout.address_table_byte_count().unwrap(),
    )
    .unwrap());
    paged_store
        .register_device_memory_reclaimer(&device)
        .unwrap();
    let paged_buffers = paged_store
        .dynamic_buffers_for_components(
            &device,
            "target",
            &BTreeSet::from(["component".to_string()]),
        )
        .unwrap();
    paged_store.mark_mount_complete().unwrap();
    let paged_owner = DeviceResourceResidencyOwnerId::new("paged-graph").unwrap();
    let address_words = || {
        paged_buffers
            .address_table()
            .read_bytes(layout.slot_count() * 32)
            .unwrap()
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
            .collect::<Vec<_>>()
    };
    let resource_slot = |resource_index| {
        layout.selectors[0]
            .mapping
            .resource_slots(resource_index)
            .unwrap()[0]
    };
    paged_store
        .load_selector_resource(&device, &selector_id, 0, paged_owner.clone())
        .unwrap();
    let first_words = address_words();
    assert_eq!(first_words[resource_slot(0) * 8 + 6], 1);
    assert_eq!(first_words[resource_slot(1) * 8 + 6], 0);

    paged_store
        .load_selector_resource(&device, &selector_id, 1, paged_owner.clone())
        .expect("bounded paging must evict an inactive cohort before admission");
    let second_words = address_words();
    assert_eq!(second_words[resource_slot(0) * 8 + 6], 0);
    assert_eq!(second_words[resource_slot(1) * 8 + 6], 1);

    paged_store
        .load_selector_resource(&device, &selector_id, 0, paged_owner)
        .expect("an evicted resource must reload into a valid stable slot");
    let reloaded_words = address_words();
    assert_eq!(reloaded_words[resource_slot(0) * 8 + 6], 1);
    assert_eq!(reloaded_words[resource_slot(1) * 8 + 6], 0);
    let paged = paged_store.residency_report().unwrap();
    assert_eq!(paged.current_payload_bytes, 8);
    assert_eq!(paged.resident_unit_count, 1);
    assert_eq!(paged.eviction_count, 2);
    assert_eq!(paged.evicted_unit_count, 2);
    assert_eq!(paged.evicted_payload_bytes, 16);
    assert_eq!(
        paged.released_device_bytes, 0,
        "ordinary expert replacement must reuse retained chunks instead of cycling Vulkan allocations",
    );
    assert_eq!(paged.reload_count, 1);
    let remaining = usize::try_from(
        device
            .device_local_memory_accounting()
            .unwrap()
            .admissible_remaining_bytes,
    )
    .unwrap();
    let competing_allocation = device.reserve_device_local_memory_capacity(remaining).unwrap();
    let fixed_runtime_buffer = device
        .create_resident_buffer(1)
        .expect("a fixed runtime allocation must reclaim inactive paged resources globally");
    let reclaimed = paged_store.residency_report().unwrap();
    assert_eq!(reclaimed.current_payload_bytes, 0);
    assert_eq!(reclaimed.resident_unit_count, 0);
    assert!(reclaimed.eviction_count > paged.eviction_count);
    assert!(
        reclaimed.released_device_bytes > paged.released_device_bytes,
        "explicit device-wide pressure must still be able to trim inactive arena backing",
    );
    drop(fixed_runtime_buffer);
    drop(competing_allocation);
    assert_eq!(
        paged_store.unload().unwrap(),
        DeviceResourceResidencyRelease {
            group_count: 0,
            byte_count: 0,
            cancelled_load_count: 0,
        }
    );
    drop(paged_store);

    let failed_tier_store = VulkanCompiledResourceDeviceStore::new_tiered(
        &device,
        ResourceResidencyPolicy::DemandPaged,
        "amd-failed-tier-admission-test",
        device.physical_device_id(),
        vec!["gpu0".to_string()],
        root.path(),
        Arc::clone(&contract),
        Arc::clone(&layout),
        BTreeSet::from([selector_id.clone()]),
        16,
        8,
        8,
        256 * 1024,
        8,
        1,
        128,
        64,
        layout.address_table_byte_count().unwrap(),
    )
    .unwrap();
    failed_tier_store.mark_mount_complete().unwrap();
    let failed_tier_owner =
        DeviceResourceResidencyOwnerId::new("failed-tier-admission-graph").unwrap();
    failed_tier_store
        .load_selector_resource(&device, &selector_id, 0, failed_tier_owner.clone())
        .unwrap();
    let mut corrupt_weight_bytes = weight_bytes.to_vec();
    corrupt_weight_bytes[8] ^= 0xff;
    fs::write(root.path().join("weights.bin"), &corrupt_weight_bytes).unwrap();
    let failed_tier_error = failed_tier_store
        .load_selector_resource(&device, &selector_id, 1, failed_tier_owner)
        .unwrap_err();
    assert!(failed_tier_error.to_string().contains("failed SHA-256"));
    fs::write(root.path().join("weights.bin"), weight_bytes).unwrap();
    let failed_tier_plan = failed_tier_store
        .memory_plan
        .as_ref()
        .unwrap()
        .lock()
        .unwrap();
    assert_eq!(failed_tier_plan.device_payload_bytes, 8);
    assert_eq!(failed_tier_plan.host_visible_payload_bytes, 0);
    assert_eq!(failed_tier_plan.group_tiers.len(), 1);
    drop(failed_tier_plan);
    assert_eq!(
        failed_tier_store.unload().unwrap(),
        DeviceResourceResidencyRelease {
            group_count: 1,
            byte_count: 8,
            cancelled_load_count: 0,
        }
    );
    drop(failed_tier_store);

    let demand_tiered_store = Arc::new(
        VulkanCompiledResourceDeviceStore::new_tiered(
            &device,
            ResourceResidencyPolicy::DemandPaged,
            "amd-demand-tiered-test",
            device.physical_device_id(),
            vec!["gpu0".to_string()],
            root.path(),
            Arc::clone(&contract),
            Arc::clone(&layout),
            BTreeSet::from([selector_id.clone()]),
            16,
            8,
            8,
            256 * 1024,
            8,
            1,
            128,
            64,
            layout.address_table_byte_count().unwrap(),
        )
        .unwrap(),
    );
    demand_tiered_store
        .register_device_memory_reclaimer(&device)
        .unwrap();
    demand_tiered_store.mark_mount_complete().unwrap();
    let demand_tiered_owner =
        DeviceResourceResidencyOwnerId::new("demand-tiered-graph").unwrap();
    demand_tiered_store
        .load_selector_resource(
            &device,
            &selector_id,
            0,
            demand_tiered_owner.clone(),
        )
        .unwrap();
    demand_tiered_store
        .load_selector_resource(&device, &selector_id, 1, demand_tiered_owner)
        .expect("host-tier capacity must retain a new group without evicting its device-tier peer");
    let demand_tiered_report = demand_tiered_store.residency_report().unwrap();
    assert_eq!(demand_tiered_report.current_payload_bytes, 16);
    assert_eq!(demand_tiered_report.resident_unit_count, 2);
    assert_eq!(demand_tiered_report.eviction_count, 0);
    let demand_tiered_plan = demand_tiered_store
        .memory_plan
        .as_ref()
        .unwrap()
        .lock()
        .unwrap();
    assert_eq!(demand_tiered_plan.device_payload_bytes, 8);
    assert_eq!(demand_tiered_plan.host_visible_payload_bytes, 8);
    drop(demand_tiered_plan);
    assert_eq!(
        demand_tiered_store.unload().unwrap(),
        DeviceResourceResidencyRelease {
            group_count: 2,
            byte_count: 16,
            cancelled_load_count: 0,
        }
    );
    drop(demand_tiered_store);

    let cohort_store = Arc::new(
        VulkanCompiledResourceDeviceStore::new_tiered(
            &device,
            ResourceResidencyPolicy::DemandPaged,
            "amd-complete-cohort-reclaimer-test",
            device.physical_device_id(),
            vec!["gpu0".to_string()],
            root.path(),
            Arc::clone(&contract),
            Arc::clone(&layout),
            BTreeSet::from([selector_id.clone()]),
            16,
            16,
            8,
            256 * 1024,
            8,
            1,
            128,
            64,
            layout.address_table_byte_count().unwrap(),
        )
        .unwrap(),
    );
    cohort_store
        .register_device_memory_reclaimer(&device)
        .unwrap();
    cohort_store.mark_mount_complete().unwrap();
    assert_eq!(
        cohort_store
            .load_all_allowed(
                &device,
                DeviceResourceResidencyOwnerId::new("complete-cohort-graph").unwrap(),
            )
            .unwrap(),
        2
    );
    let cohort_chunk_ids = cohort_store
        .address_state
        .lock()
        .unwrap()
        .group_chunks
        .values()
        .flatten()
        .copied()
        .collect::<BTreeSet<_>>();
    assert_eq!(
        cohort_chunk_ids.len(),
        1,
        "the fixture must prove complete shared-chunk reclamation"
    );
    let cohort_committed_bytes = cohort_store
        .device_arena
        .stats()
        .unwrap()
        .committed_byte_capacity;
    assert!(cohort_committed_bytes > 0);
    assert_eq!(
        cohort_store
            .reclaim_inactive_device_memory(1)
            .unwrap(),
        cohort_committed_bytes
    );
    assert_eq!(
        cohort_store
            .device_arena
            .stats()
            .unwrap()
            .committed_byte_capacity,
        0
    );
    assert_eq!(cohort_store.residency_report().unwrap().resident_unit_count, 0);
    assert_eq!(
        cohort_store.unload().unwrap(),
        DeviceResourceResidencyRelease {
            group_count: 0,
            byte_count: 0,
            cancelled_load_count: 0,
        }
    );
    drop(cohort_store);

    let tiered_store = VulkanCompiledResourceDeviceStore::new_tiered(
        &device,
        ResourceResidencyPolicy::DemandPaged,
        "amd-tiered-test",
        device.physical_device_id(),
        vec!["gpu0".to_string()],
        root.path(),
        Arc::clone(&contract),
        Arc::clone(&layout),
        BTreeSet::from([selector_id.clone()]),
        16,
        8,
        16,
        22,
        8,
        1,
        128,
        64,
        layout.address_table_byte_count().unwrap(),
    )
    .unwrap();
    let tiered_buffers = tiered_store
        .dynamic_buffers_for_components(
            &device,
            "target",
            &BTreeSet::from(["component".to_string()]),
        )
        .unwrap();
    assert_eq!(
        tiered_store
            .load_all_allowed(
                &device,
                DeviceResourceResidencyOwnerId::new("tiered-eager").unwrap(),
            )
            .unwrap(),
        2
    );
    tiered_store.mark_mount_complete().unwrap();
    assert_eq!(
        tiered_store.statistics().unwrap().dynamic_resident_bytes,
        16
    );
    let tiered_plan = tiered_store.memory_plan.as_ref().unwrap().lock().unwrap();
    assert_eq!(tiered_plan.device_payload_bytes, 8);
    assert_eq!(tiered_plan.host_visible_payload_bytes, 8);
    let tiered_groups = (0..2)
        .map(|resource_index| {
            tiered_store
                .resolve_selector_resource(&selector_id, resource_index)
                .unwrap()
                .id()
                .to_string()
        })
        .collect::<Vec<_>>();
    let device_resource_index = tiered_groups
        .iter()
        .position(|group_id| {
            tiered_plan.tier_for_group(group_id).unwrap()
                == VulkanCompiledResourceMemoryTier::Device
        })
        .unwrap();
    let host_resource_index = 1 - device_resource_index;
    drop(tiered_plan);
    assert_eq!(
        tiered_store
            .device_arena
            .stats()
            .unwrap()
            .allocated_byte_count,
        8
    );
    assert_eq!(
        tiered_store
            .host_visible_arena
            .as_ref()
            .unwrap()
            .stats()
            .unwrap()
            .allocated_byte_count,
        8
    );
    let tiered_address_words = tiered_buffers
        .address_table()
        .read_bytes(layout.slot_count() * 32)
        .unwrap()
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect::<Vec<_>>();
    for resource_index in 0..2 {
        let slot = layout.selectors[0]
            .mapping
            .resource_slots(resource_index)
            .unwrap()[0];
        assert_ne!(
            tiered_address_words[slot * 8] | tiered_address_words[slot * 8 + 1],
            0
        );
        assert_eq!(tiered_address_words[slot * 8 + 6], 1);
    }
    let malformed_telemetry = VulkanSelectionTelemetrySnapshot {
        domains: vec![VulkanSelectionTelemetryDomainSnapshot {
            execution_scope: "target".to_string(),
            component_id: "component".to_string(),
            node_id: "choose".to_string(),
            domain_id: "resources".to_string(),
            resource_count: 2,
            selection_counts: vec![1],
        }],
    };
    assert!(tiered_store
        .retier_from_selection_telemetry(&device, &malformed_telemetry)
        .is_err());
    let mut selection_counts = vec![1, 1];
    selection_counts[host_resource_index] = 100;
    let telemetry = VulkanSelectionTelemetrySnapshot {
        domains: vec![VulkanSelectionTelemetryDomainSnapshot {
            execution_scope: "target".to_string(),
            component_id: "component".to_string(),
            node_id: "choose".to_string(),
            domain_id: "resources".to_string(),
            resource_count: 2,
            selection_counts,
        }],
    };
    let (publications_before_failure, group_chunks_before_failure, chunk_groups_before_failure) = {
        let state = tiered_store.address_state.lock().unwrap();
        (
            state.publications.clone(),
            state.group_chunks.clone(),
            state.chunk_groups.clone(),
        )
    };
    let tiers_before_failure = tiered_store
        .memory_plan
        .as_ref()
        .unwrap()
        .lock()
        .unwrap()
        .group_tiers
        .clone();
    tiered_store.inject_retiering_failure_after_payload_exchange();
    let injected_failure = tiered_store
        .retier_from_selection_telemetry(&device, &telemetry)
        .unwrap_err();
    assert!(
        injected_failure
            .to_string()
            .contains("injected compiled resource retiering failure")
    );
    {
        let state = tiered_store.address_state.lock().unwrap();
        assert_eq!(state.publications, publications_before_failure);
        assert_eq!(state.group_chunks, group_chunks_before_failure);
        assert_eq!(state.chunk_groups, chunk_groups_before_failure);
    }
    assert_eq!(
        tiered_store
            .memory_plan
            .as_ref()
            .unwrap()
            .lock()
            .unwrap()
            .group_tiers,
        tiers_before_failure
    );
    for (resource_index, group_id) in tiered_groups.iter().enumerate() {
        let allocation = {
            let state = tiered_store.address_state.lock().unwrap();
            let publications = state.publications.get(group_id).unwrap();
            state
                .address_table
                .allocations_for_publications(publications)
                .unwrap()
                .remove(0)
        };
        let readback = device
            .read_resident_buffer_ranges(&[VulkanResidentBufferReadRange::new(
                allocation.buffer(),
                allocation.buffer_byte_offset(),
                allocation.byte_count(),
            )
            .unwrap()])
            .unwrap();
        assert_eq!(
            readback.range_bytes(0).unwrap(),
            &weight_bytes[resource_index * 8..(resource_index + 1) * 8],
            "failed retiering must restore each logical resource's exact payload"
        );
    }
    let retiering = tiered_store
        .retier_from_selection_telemetry(&device, &telemetry)
        .unwrap();
    assert_eq!(retiering.promoted_group_count, 1);
    assert_eq!(retiering.demoted_group_count, 1);
    assert_eq!(retiering.promoted_payload_bytes, 8);
    assert_eq!(retiering.copied_payload_bytes, 16);
    let tiered_plan = tiered_store.memory_plan.as_ref().unwrap().lock().unwrap();
    assert_eq!(
        tiered_plan
            .tier_for_group(&tiered_groups[host_resource_index])
            .unwrap(),
        VulkanCompiledResourceMemoryTier::Device
    );
    assert_eq!(
        tiered_plan
            .tier_for_group(&tiered_groups[device_resource_index])
            .unwrap(),
        VulkanCompiledResourceMemoryTier::HostVisible
    );
    drop(tiered_plan);
    for (resource_index, group_id) in tiered_groups.iter().enumerate() {
        let allocation = {
            let state = tiered_store.address_state.lock().unwrap();
            let publications = state.publications.get(group_id).unwrap();
            state
                .address_table
                .allocations_for_publications(publications)
                .unwrap()
                .remove(0)
        };
        let readback = device
            .read_resident_buffer_ranges(&[
                VulkanResidentBufferReadRange::new(
                    allocation.buffer(),
                    allocation.buffer_byte_offset(),
                    allocation.byte_count(),
                )
                .unwrap(),
            ])
            .unwrap();
        assert_eq!(
            readback.range_bytes(0).unwrap(),
            &weight_bytes[resource_index * 8..(resource_index + 1) * 8],
            "retiering must preserve each logical resource's exact payload"
        );
    }
    let mut shifted_selection_counts = vec![50, 50];
    shifted_selection_counts[host_resource_index] = 100;
    let shifted_telemetry = VulkanSelectionTelemetrySnapshot {
        domains: vec![VulkanSelectionTelemetryDomainSnapshot {
            execution_scope: "target".to_string(),
            component_id: "component".to_string(),
            node_id: "choose".to_string(),
            domain_id: "resources".to_string(),
            resource_count: 2,
            selection_counts: shifted_selection_counts,
        }],
    };
    assert_eq!(
        tiered_store
            .retier_from_selection_telemetry(&device, &shifted_telemetry)
            .unwrap()
            .promoted_group_count,
        0,
        "one interval favoring the old cold resource must not undo the cumulative working set"
    );
    let retiering_report = tiered_store.residency_report().unwrap();
    assert_eq!(retiering_report.retiering_event_count, 2);
    assert_eq!(retiering_report.retiering_promoted_group_count, 1);
    assert_eq!(retiering_report.retiering_promoted_payload_bytes, 8);
    assert_eq!(retiering_report.retiering_copied_payload_bytes, 16);
    assert_eq!(retiering_report.retiering_device_selection_count, 1);
    assert_eq!(retiering_report.retiering_host_visible_selection_count, 149);
    assert!(retiering_report.retiering_time_ns > 0);
    let committed_device_bytes = tiered_store
        .device_arena
        .stats()
        .unwrap()
        .committed_byte_capacity;
    assert!(committed_device_bytes > 0);
    assert_eq!(
        tiered_store.reclaim_inactive_device_memory(1).unwrap(),
        committed_device_bytes,
        "retiered logical ownership must release the physical device allocation"
    );
    assert_eq!(
        tiered_store
            .device_arena
            .stats()
            .unwrap()
            .committed_byte_capacity,
        0
    );
    assert_eq!(tiered_store.residency_report().unwrap().resident_unit_count, 1);
    let reload_slots = layout.selectors[0]
        .mapping
        .resource_slots(host_resource_index)
        .unwrap();
    let competing_device_allocation = tiered_store
        .device_arena
        .allocate_groups(&device, &[(&reload_slots, &[8])], 8)
        .unwrap();
    tiered_store
        .load_selector_resource(
            &device,
            &selector_id,
            host_resource_index,
            DeviceResourceResidencyOwnerId::new("retiered-reload").unwrap(),
        )
        .expect(
            "a retiered layout must support another physical generation when reloaded to the host tier",
        );
    drop(competing_device_allocation);
    let reloaded_plan = tiered_store.memory_plan.as_ref().unwrap().lock().unwrap();
    assert_eq!(reloaded_plan.device_payload_bytes, 0);
    assert_eq!(reloaded_plan.host_visible_payload_bytes, 16);
    drop(reloaded_plan);
    assert_eq!(
        tiered_store.unload().unwrap(),
        DeviceResourceResidencyRelease {
            group_count: 2,
            byte_count: 16,
            cancelled_load_count: 0,
        }
    );
    assert!(
        tiered_store
            .memory_plan
            .as_ref()
            .unwrap()
            .lock()
            .unwrap()
            .group_tiers
            .is_empty(),
        "demand-tier assignments must not outlive model teardown"
    );
    drop(tiered_buffers);
    drop(tiered_store);

    let store = VulkanCompiledResourceDeviceStore::new(
        &device,
        ResourceResidencyPolicy::DemandRetained,
        "amd-test",
        device.physical_device_id(),
        vec!["gpu0".to_string()],
        root.path(),
        Arc::clone(&contract),
        Arc::clone(&layout),
        BTreeSet::from([selector_id.clone(), alias_selector_id.clone()]),
        4096,
        256 * 1024,
        1024,
        1,
        128,
        64,
        layout.address_table_byte_count().unwrap(),
    )
    .unwrap();
    let buffers = store
        .dynamic_buffers_for_components(
            &device,
            "target",
            &BTreeSet::from(["component".to_string()]),
        )
        .unwrap();
    let owner = DeviceResourceResidencyOwnerId::new("graph").unwrap();
    store.mark_mount_complete().unwrap();
    let initial = store.residency_report().unwrap();
    assert_eq!(initial.initial_payload_bytes, 0);
    assert_eq!(initial.current_payload_bytes, 0);
    assert_eq!(initial.initial_resident_unit_count, 0);
    assert_eq!(initial.resident_unit_count, 0);
    assert_eq!(initial.addressable_unit_count, 2);
    assert_eq!(initial.always_resident_parameter_bytes, 128);
    assert_eq!(initial.runtime_working_set_device_bytes, 64);
    assert!(initial.initial_device_bytes >= 128 + 64 + initial.metadata_device_bytes);
    assert_eq!(initial.scopes.len(), 1);
    assert_eq!(initial.scopes[0].execution_scope, "target");
    assert_eq!(initial.scopes[0].component_count, 2);
    assert_eq!(initial.scopes[0].addressable_unit_count, 2);

    let unowned_error = store
        .load_selector_resource(&device, &unowned_selector_id, 0, owner.clone())
        .unwrap_err();
    assert!(unowned_error.to_string().contains("is unknown"));

    let mut corrupt_weight_bytes = weight_bytes.to_vec();
    corrupt_weight_bytes[8] ^= 0xff;
    fs::write(root.path().join("weights.bin"), &corrupt_weight_bytes).unwrap();
    let corrupt_error = store
        .load_selector_resource(&device, &selector_id, 1, owner.clone())
        .unwrap_err();
    assert!(
        corrupt_error.to_string().contains("failed SHA-256"),
        "unexpected corrupt-resource error: {corrupt_error}"
    );
    fs::write(root.path().join("weights.bin"), weight_bytes).unwrap();

    store
        .load_selector_resource(&device, &selector_id, 0, owner.clone())
        .unwrap();
    store
        .load_selector_resource(&device, &alias_selector_id, 0, owner)
        .unwrap();
    store.record_gpu_gate_misses(&selector_id, 2).unwrap();
    store.record_gpu_gate_misses(&alias_selector_id, 3).unwrap();

    let stats = store.statistics().unwrap();
    assert_eq!(stats.miss_count, 2);
    assert_eq!(stats.hit_count, 1);
    assert_eq!(stats.resident_group_count, 1);
    assert_eq!(stats.failed_group_count, 1);
    assert_eq!(stats.dynamic_resident_bytes, 8);
    assert_eq!(stats.high_water_resident_group_count, 1);
    assert_eq!(stats.high_water_dynamic_resident_bytes, 8);
    let report = store.residency_report().unwrap();
    assert_eq!(report.physical_device_id, device.physical_device_id());
    assert_eq!(report.logical_device_ids, ["gpu0"]);
    assert_eq!(report.current_payload_bytes, 8);
    assert_eq!(report.high_water_payload_bytes, 8);
    assert_eq!(report.resident_unit_count, 1);
    assert_eq!(report.high_water_resident_unit_count, 1);
    assert_eq!(report.residency_directory_hit_count, 1);
    assert_eq!(report.residency_load_required_count, 2);
    assert_eq!(report.gpu_selection_count, 0);
    assert_eq!(report.gpu_resident_hit_count, 0);
    assert_eq!(report.gpu_miss_count, 5);
    assert_eq!(report.successful_load_count, 1);
    assert_eq!(report.failed_load_count, 1);
    assert_eq!(report.failed_unit_count, 1);
    assert_eq!(report.physical_read_count, 1);
    assert_eq!(report.physical_bytes_read, 8);
    assert_eq!(report.uploaded_bytes, 8);
    assert_eq!(report.components.len(), 2);
    assert!(report.components.iter().all(
        |component| component.addressable_unit_count == 2 && component.resident_unit_count == 1
    ));
    assert_eq!(
        report
            .components
            .iter()
            .map(|component| component.gpu_miss_count)
            .collect::<Vec<_>>(),
        [2, 3]
    );
    let address_words = buffers
        .address_table()
        .read_bytes(layout.slot_count() * 32)
        .unwrap()
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect::<Vec<_>>();
    let selected_slot = layout.selectors[0].mapping.resource_slots(0).unwrap()[0];
    assert_eq!(address_words[selected_slot * 8 + 6], 1);
    assert_ne!(
        address_words[selected_slot * 8] | address_words[selected_slot * 8 + 1],
        0
    );

    store.inject_teardown_failure_before_address_clear();
    let injected = store.unload().unwrap_err();
    assert!(
        injected
            .to_string()
            .contains("injected compiled resource teardown failure")
    );
    let quiescing_error = store
        .load_selector_resource(
            &device,
            &selector_id,
            0,
            DeviceResourceResidencyOwnerId::new("quiescing-owner").unwrap(),
        )
        .unwrap_err();
    assert!(quiescing_error.to_string().contains("Quiescing"));
    assert_eq!(store.statistics().unwrap().dynamic_resident_bytes, 0);
    let retained_words = buffers
        .address_table()
        .read_bytes(layout.slot_count() * 32)
        .unwrap()
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(retained_words[selected_slot * 8 + 6], 1);

    let release = store.unload().unwrap();

    assert_eq!(release.group_count, 1);
    assert_eq!(release.byte_count, 8);
    assert_eq!(store.statistics().unwrap().dynamic_resident_bytes, 0);
    let unloaded = store.residency_report().unwrap();
    assert_eq!(unloaded.current_payload_bytes, 0);
    assert_eq!(unloaded.resident_unit_count, 0);
    assert_eq!(unloaded.failed_unit_count, 0);
    assert_eq!(unloaded.high_water_payload_bytes, 8);
    assert_eq!(unloaded.high_water_resident_unit_count, 1);
    let retired_words = buffers
        .address_table()
        .read_bytes(layout.slot_count() * 32)
        .unwrap()
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect::<Vec<_>>();
    for record in retired_words.chunks_exact(8) {
        assert_eq!(record[0] | record[1], 0);
        assert_eq!(record[2] | record[3], 0);
        assert_eq!(record[6], 0);
    }
    let repeated = store.unload().unwrap();
    assert_eq!(repeated, DeviceResourceResidencyRelease::default());
    let retired_error = store
        .load_selector_resource(
            &device,
            &selector_id,
            0,
            DeviceResourceResidencyOwnerId::new("retired-owner").unwrap(),
        )
        .unwrap_err();
    assert!(retired_error.to_string().contains("Unloaded"));
    drop(buffers);
    drop(store);

    let baseline_file_descriptors = compiled_store_process_file_descriptor_count();
    let baseline_workers = compiled_store_worker_thread_count();
    for cycle_index in 0..3 {
        let cycle_store = VulkanCompiledResourceDeviceStore::new(
            &device,
            ResourceResidencyPolicy::DemandRetained,
            format!("amd-test-cycle-{cycle_index}"),
            device.physical_device_id(),
            vec!["gpu0".to_string()],
            root.path(),
            Arc::clone(&contract),
            Arc::clone(&layout),
            BTreeSet::from([selector_id.clone(), alias_selector_id.clone()]),
            4096,
            256 * 1024,
            1024,
            1,
            128,
            64,
            layout.address_table_byte_count().unwrap(),
        )
        .unwrap();
        cycle_store.mark_mount_complete().unwrap();
        if cycle_index == 0 {
            cycle_store.inject_next_upload_as_device_lost();
            let device_loss = cycle_store
                .load_selector_resource(
                    &device,
                    &selector_id,
                    0,
                    DeviceResourceResidencyOwnerId::new("device-loss-owner").unwrap(),
                )
                .unwrap_err();
            assert!(device_loss.to_string().contains("ERROR_DEVICE_LOST"));
            let terminal = cycle_store
                .load_selector_resource(
                    &device,
                    &selector_id,
                    1,
                    DeviceResourceResidencyOwnerId::new("post-device-loss-owner").unwrap(),
                )
                .unwrap_err();
            assert!(terminal.to_string().contains("Failed"));
            assert!(terminal.to_string().contains("ERROR_DEVICE_LOST"));
            let cycle_release = cycle_store.unload().unwrap();
            assert_eq!(cycle_release, DeviceResourceResidencyRelease::default());
            drop(cycle_store);
            assert_eq!(compiled_store_worker_thread_count(), baseline_workers);
            assert_eq!(
                compiled_store_process_file_descriptor_count(),
                baseline_file_descriptors
            );
            continue;
        }
        if cycle_index == 1 {
            reset_vulkan_resident_execution_counters();
            assert_eq!(
                cycle_store
                    .load_selector_resources(
                        &device,
                        &selector_id,
                        &[0, 1],
                        DeviceResourceResidencyOwnerId::new("batched-cycle",).unwrap(),
                    )
                    .unwrap(),
                2
            );
            let counters = vulkan_resident_execution_counters();
            assert_eq!(counters.resident_copy_queue_submits, 2);
            assert_eq!(counters.resident_copy_waits, 2);
        } else {
            cycle_store
                .load_selector_resource(
                    &device,
                    &selector_id,
                    0,
                    DeviceResourceResidencyOwnerId::new(format!("cycle-{cycle_index}")).unwrap(),
                )
                .unwrap();
        }
        let expected_payload_bytes = if cycle_index == 1 { 16 } else { 8 };
        assert_eq!(
            cycle_store
                .residency_report()
                .unwrap()
                .current_payload_bytes,
            expected_payload_bytes
        );
        assert_eq!(cycle_store.backing_store.retained_payload_bytes(), 0);
        let cycle_release = cycle_store.unload().unwrap();
        assert_eq!(
            cycle_release.group_count,
            if cycle_index == 1 { 2 } else { 1 }
        );
        assert_eq!(cycle_release.byte_count, expected_payload_bytes);
        drop(cycle_store);
        assert_eq!(compiled_store_worker_thread_count(), baseline_workers);
        assert_eq!(
            compiled_store_process_file_descriptor_count(),
            baseline_file_descriptors
        );
    }
}

#[test]
fn optional_output_heads_follow_group_table_miss_load_hit_and_unload() {
    let device = selected_test_vulkan_device().expect("selected Vulkan test device must open");
    let root = crate::test_support::TempDir::new("optional_output_head_residency");
    let payloads = [
        &b"head0-w0"[..],
        &b"head0-b0"[..],
        &b"head1-w1"[..],
        &b"head1-b1"[..],
    ];
    let artifact_bytes = payloads.concat();
    fs::write(root.path().join("optional_heads.bin"), &artifact_bytes).unwrap();
    let content_id = |byte: char| format!("sha256:{}", byte.to_string().repeat(64));
    let compatibility = CompiledResourceCompatibility {
        device_api: "vulkan".to_string(),
        storage_class: "storage_buffer".to_string(),
        read_only: true,
        required_features: vec!["buffer_device_address".to_string()],
    };
    let resource_ids = ['1', '2', '3', '4']
        .into_iter()
        .map(content_id)
        .collect::<Vec<_>>();
    let resources = payloads
        .iter()
        .enumerate()
        .map(|(index, payload)| CompiledImmutableResource {
            id: resource_ids[index].clone(),
            lifetime: CompiledResourceLifetime::Dynamic,
            ranges: vec![CompiledResourceByteRange {
                artifact_path: "optional_heads.bin".to_string(),
                byte_offset: index * 8,
                byte_count: payload.len(),
                alignment_bytes: 8,
                integrity: CompiledResourceRangeIntegrity {
                    algorithm: "sha256".to_string(),
                    digest: Sha256::digest(payload)
                        .iter()
                        .map(|byte| format!("{byte:02x}"))
                        .collect(),
                },
            }],
            dependencies: Vec::new(),
            compatibility: compatibility.clone(),
            resident_derivation: None,
        })
        .collect::<Vec<_>>();
    let group_ids = vec![content_id('5'), content_id('6')];
    let selector_id = content_id('7');
    let contract = Arc::new(CompiledResourceResidencyContract {
        schema: COMPILED_RESOURCE_RESIDENCY_SCHEMA.to_string(),
        identity_algorithm: RESOURCE_IDENTITY_ALGORITHM.to_string(),
        state_machine_schema: RESOURCE_RESIDENCY_STATE_MACHINE_SCHEMA.to_string(),
        supported_policies: vec![
            ResourceResidencyPolicy::DemandRetained,
            ResourceResidencyPolicy::Eager,
        ],
        resources,
        atomic_groups: vec![
            CompiledAtomicResidencyGroup {
                id: group_ids[0].clone(),
                lifetime: CompiledResourceLifetime::Dynamic,
                resource_ids: resource_ids[0..2].to_vec(),
                dependencies: Vec::new(),
            },
            CompiledAtomicResidencyGroup {
                id: group_ids[1].clone(),
                lifetime: CompiledResourceLifetime::Dynamic,
                resource_ids: resource_ids[2..4].to_vec(),
                dependencies: Vec::new(),
            },
        ],
        partition_templates: Vec::new(),
        bindings: Vec::new(),
        selectors: vec![CompiledResourceSelector {
            id: selector_id.clone(),
            execution_scope: "target".to_string(),
            component_id: "optional_output_head".to_string(),
            node_id: "choose_head".to_string(),
            domain_id: "output_heads".to_string(),
            resource_count: 2,
            selection_signal: "selected_head".to_string(),
            encoding: CompiledResourceSelectionEncoding {
                element_type: CompiledResourceSelectionElementType::U32,
                selection_count_per_activation: 1,
                index_shift: 0,
                index_mask: 0xffff,
            },
            mapping: CompiledResourceSelectorMapping::GroupTable {
                atomic_group_ids: group_ids,
            },
        }],
        checkpoints: Vec::new(),
    });
    let inspection = contract.inspection_report().unwrap();
    assert_eq!(inspection.dynamically_addressable.unit_count, 2);
    assert_eq!(inspection.dynamically_addressable.resource_count, 4);
    assert_eq!(inspection.dynamically_addressable.maximum_payload_bytes, 32);
    let layout = Arc::new(VulkanCompiledResourceAddressLayout::from_contract(&contract).unwrap());
    let VulkanCompiledSelectorAddressMapping::GroupTable {
        resource_address_slots,
        resource_address_slot_offsets,
    } = &layout.selectors[0].mapping
    else {
        panic!("optional output heads did not compile to a group table");
    };
    assert_eq!(resource_address_slots, &[0, 1, 2, 3]);
    assert_eq!(resource_address_slot_offsets, &[0, 2, 4]);

    let store = VulkanCompiledResourceDeviceStore::new(
        &device,
        ResourceResidencyPolicy::DemandRetained,
        "amd-optional-head-test",
        device.physical_device_id(),
        vec!["gpu0".to_string()],
        root.path(),
        Arc::clone(&contract),
        Arc::clone(&layout),
        BTreeSet::from([selector_id.clone()]),
        32,
        256 * 1024,
        16,
        2,
        0,
        64,
        layout.address_table_byte_count().unwrap(),
    )
    .unwrap();
    let buffers = store
        .dynamic_buffers_for_components(
            &device,
            "target",
            &BTreeSet::from(["optional_output_head".to_string()]),
        )
        .unwrap();
    store.mark_mount_complete().unwrap();
    let initial = store.residency_report().unwrap();
    assert_eq!(initial.initial_payload_bytes, 0);
    assert_eq!(initial.resident_unit_count, 0);

    let selection = Arc::new(device.create_resident_buffer(size_of::<u32>()).unwrap());
    selection.write_bytes(&1u32.to_le_bytes()).unwrap();
    let continuation = Arc::new(device.create_conditional_resident_buffer(4).unwrap());
    continuation.write_bytes(&1u32.to_le_bytes()).unwrap();
    let gate = VulkanGpuResidencyGate::new(
        &device,
        &vulkan_gpu_residency_gate_spirv_words().unwrap(),
        selection,
        buffers.shared_address_table(),
        buffers.address_table_slot_count(),
        VulkanGpuResidencyMissQueue::new(&device, 1).unwrap(),
        continuation,
        VulkanGpuResidencyGateConfig {
            maximum_selection_count: 1,
            selection_count_per_lane: 1,
            selection_lane_stride_words: 1,
            selection_index_shift: 0,
            selection_index_mask: 0xffff,
            address_mapping: VulkanGpuResidencyAddressMapping::GroupTable {
                resource_address_slots: resource_address_slots.clone(),
                resource_address_slot_offsets: resource_address_slot_offsets.clone(),
            },
        },
    )
    .unwrap();
    let first_control = gate.push_constants(1, 17, true).unwrap();
    device
        .run_resident_kernel_dispatch(gate.dispatch(), &first_control)
        .unwrap();
    let missing = gate.missing_snapshot().unwrap();
    assert_eq!(
        missing.requests,
        [VulkanGpuResidencyMissingRequest {
            checkpoint_tag: 17,
            resource_index: 1,
        }]
    );
    assert_eq!(store.residency_report().unwrap().current_payload_bytes, 0);

    store.record_gpu_gate_misses(&selector_id, 1).unwrap();
    store
        .load_selector_resource(
            &device,
            &selector_id,
            1,
            DeviceResourceResidencyOwnerId::new("optional-head-graph").unwrap(),
        )
        .unwrap();
    gate.acknowledge_missing_through(missing.published_count)
        .unwrap();
    let second_control = gate.push_constants(1, 18, true).unwrap();
    device
        .run_resident_kernel_dispatch(gate.dispatch(), &second_control)
        .unwrap();
    assert!(gate.missing_snapshot().unwrap().requests.is_empty());

    let report = store.residency_report().unwrap();
    assert_eq!(report.current_payload_bytes, 16);
    assert_eq!(report.resident_unit_count, 1);
    assert_eq!(report.successful_load_count, 1);
    assert_eq!(report.physical_bytes_read, 16);
    assert_eq!(report.uploaded_bytes, 16);
    let records = buffers
        .address_table()
        .read_bytes(layout.address_table_byte_count().unwrap())
        .unwrap()
        .chunks_exact(32)
        .map(|record| u32::from_le_bytes(record[24..28].try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(records, [0, 0, 1, 1]);

    assert_eq!(
        store.unload().unwrap(),
        DeviceResourceResidencyRelease {
            group_count: 1,
            byte_count: 16,
            cancelled_load_count: 0,
        }
    );
    assert_eq!(store.residency_report().unwrap().current_payload_bytes, 0);
    drop(gate);
    drop(buffers);
    drop(store);
}
