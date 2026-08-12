fn distributed_cohort_store_plan(
    members: Vec<crate::vulkan_distributed::VulkanDistributedSelectedResourceResidencyCohortMemberPlan>,
) -> VulkanDistributedSelectedResourceStorePlan {
    use crate::vulkan_distributed::{
        VulkanDistributedSelectedResourceDevicePlan,
        VulkanDistributedSelectedResourceResidencyCohortPlan,
    };

    VulkanDistributedSelectedResourceStorePlan {
        devices: vec![
            VulkanDistributedSelectedResourceDevicePlan {
                device_id: "gpu0".to_string(),
                selectors: Vec::new(),
                unique_atomic_group_count: 0,
                maximum_atomic_group_bytes: 0,
                maximum_load_wave_bytes: 0,
                total_addressable_bytes: 0,
            },
            VulkanDistributedSelectedResourceDevicePlan {
                device_id: "gpu1".to_string(),
                selectors: Vec::new(),
                unique_atomic_group_count: 0,
                maximum_atomic_group_bytes: 0,
                maximum_load_wave_bytes: 0,
                total_addressable_bytes: 0,
            },
        ],
        tensor_sharded_residency_cohorts: vec![
            VulkanDistributedSelectedResourceResidencyCohortPlan {
                selector_id: "experts".to_string(),
                resource_index: 3,
                atomic_group_id: "expert-3".to_string(),
                members,
            },
        ],
        device_count: 2,
        selector_count: 1,
        selector_placement_count: 2,
        unique_atomic_group_count: 2,
        total_addressable_bytes: 128,
    }
}

#[test]
fn distributed_cohort_catalog_preserves_exact_fragment_membership() {
    use crate::vulkan_distributed::VulkanDistributedSelectedResourceResidencyCohortMemberPlan;

    let plan = distributed_cohort_store_plan(vec![
        VulkanDistributedSelectedResourceResidencyCohortMemberPlan {
            device_id: "gpu0".to_string(),
            logical_start: 0,
            logical_count: 32,
        },
        VulkanDistributedSelectedResourceResidencyCohortMemberPlan {
            device_id: "gpu1".to_string(),
            logical_start: 32,
            logical_count: 32,
        },
    ]);
    let coordinator = VulkanCompiledResourceDistributedCohortCoordinator::new(&plan).unwrap();
    let cohort = coordinator.cohort_for_selection("experts", 3).unwrap();
    assert_eq!(cohort.key.atomic_group_id, "expert-3");
    assert_eq!(cohort.members.len(), 2);
    assert_eq!(cohort.members[0].logical_device_id, "gpu0");
    assert_eq!(cohort.members[1].logical_device_id, "gpu1");
    assert!(coordinator.cohort_for_selection("experts", 2).is_none());
    assert_eq!(coordinator.group_keys.len(), 2);
    let _mutation = coordinator.begin_mutation().unwrap();
}

#[test]
fn distributed_cohort_catalog_rejects_gaps_and_duplicate_devices() {
    use crate::vulkan_distributed::VulkanDistributedSelectedResourceResidencyCohortMemberPlan;

    for members in [
        vec![
            VulkanDistributedSelectedResourceResidencyCohortMemberPlan {
                device_id: "gpu0".to_string(),
                logical_start: 0,
                logical_count: 31,
            },
            VulkanDistributedSelectedResourceResidencyCohortMemberPlan {
                device_id: "gpu1".to_string(),
                logical_start: 32,
                logical_count: 32,
            },
        ],
        vec![
            VulkanDistributedSelectedResourceResidencyCohortMemberPlan {
                device_id: "gpu0".to_string(),
                logical_start: 0,
                logical_count: 32,
            },
            VulkanDistributedSelectedResourceResidencyCohortMemberPlan {
                device_id: "gpu0".to_string(),
                logical_start: 32,
                logical_count: 32,
            },
        ],
    ] {
        assert!(
            VulkanCompiledResourceDistributedCohortCoordinator::new(
                &distributed_cohort_store_plan(members),
            )
            .is_err(),
        );
    }
}

#[test]
fn distributed_fault_plan_expands_one_fragment_miss_to_the_complete_cohort() {
    use crate::vulkan_distributed::VulkanDistributedSelectedResourceResidencyCohortMemberPlan;

    let coordinator = VulkanCompiledResourceDistributedCohortCoordinator::new(
        &distributed_cohort_store_plan(vec![
            VulkanDistributedSelectedResourceResidencyCohortMemberPlan {
                device_id: "gpu0".to_string(),
                logical_start: 0,
                logical_count: 32,
            },
            VulkanDistributedSelectedResourceResidencyCohortMemberPlan {
                device_id: "gpu1".to_string(),
                logical_start: 32,
                logical_count: 32,
            },
        ]),
    )
    .unwrap();
    let observations = vec![
        VulkanCompiledResourceDistributedFaultObservation {
            logical_device_id: "gpu0".to_string(),
            selector_id: "experts".to_string(),
            checkpoint_tag: 1,
            pending_resource_indices: vec![3],
        },
        VulkanCompiledResourceDistributedFaultObservation {
            logical_device_id: "gpu1".to_string(),
            selector_id: "experts".to_string(),
            checkpoint_tag: 1,
            pending_resource_indices: Vec::new(),
        },
    ];

    let plan = coordinator.plan_fault_resolution(&observations).unwrap();

    assert_eq!(
        plan.loads,
        vec![
            VulkanCompiledResourceDistributedFaultLoad {
                observation_index: 0,
                resource_indices: vec![3],
            },
            VulkanCompiledResourceDistributedFaultLoad {
                observation_index: 1,
                resource_indices: vec![3],
            },
        ],
    );
    assert_eq!(plan.commit_observation_indices, vec![0]);
}

#[test]
fn distributed_fault_plan_rejects_missing_or_ambiguous_cohort_gates() {
    use crate::vulkan_distributed::VulkanDistributedSelectedResourceResidencyCohortMemberPlan;

    let coordinator = VulkanCompiledResourceDistributedCohortCoordinator::new(
        &distributed_cohort_store_plan(vec![
            VulkanDistributedSelectedResourceResidencyCohortMemberPlan {
                device_id: "gpu0".to_string(),
                logical_start: 0,
                logical_count: 32,
            },
            VulkanDistributedSelectedResourceResidencyCohortMemberPlan {
                device_id: "gpu1".to_string(),
                logical_start: 32,
                logical_count: 32,
            },
        ]),
    )
    .unwrap();
    let fault = VulkanCompiledResourceDistributedFaultObservation {
        logical_device_id: "gpu0".to_string(),
        selector_id: "experts".to_string(),
        checkpoint_tag: 1,
        pending_resource_indices: vec![3],
    };
    assert!(
        coordinator
            .plan_fault_resolution(std::slice::from_ref(&fault))
            .unwrap_err()
            .to_string()
            .contains("no residency gate")
    );

    let duplicate = VulkanCompiledResourceDistributedFaultObservation {
        logical_device_id: "gpu1".to_string(),
        selector_id: "experts".to_string(),
        checkpoint_tag: 1,
        pending_resource_indices: Vec::new(),
    };
    assert!(
        coordinator
            .plan_fault_resolution(&[fault, duplicate.clone(), duplicate])
            .unwrap_err()
            .to_string()
            .contains("ambiguous residency gates")
    );
}

#[test]
fn distributed_fault_plan_deduplicates_cohort_loads_and_keeps_local_resources_local() {
    use crate::vulkan_distributed::VulkanDistributedSelectedResourceResidencyCohortMemberPlan;

    let coordinator = VulkanCompiledResourceDistributedCohortCoordinator::new(
        &distributed_cohort_store_plan(vec![
            VulkanDistributedSelectedResourceResidencyCohortMemberPlan {
                device_id: "gpu0".to_string(),
                logical_start: 0,
                logical_count: 32,
            },
            VulkanDistributedSelectedResourceResidencyCohortMemberPlan {
                device_id: "gpu1".to_string(),
                logical_start: 32,
                logical_count: 32,
            },
        ]),
    )
    .unwrap();
    let observations = vec![
        VulkanCompiledResourceDistributedFaultObservation {
            logical_device_id: "gpu0".to_string(),
            selector_id: "experts".to_string(),
            checkpoint_tag: 1,
            pending_resource_indices: vec![2, 3],
        },
        VulkanCompiledResourceDistributedFaultObservation {
            logical_device_id: "gpu1".to_string(),
            selector_id: "experts".to_string(),
            checkpoint_tag: 1,
            pending_resource_indices: vec![3],
        },
    ];

    let plan = coordinator.plan_fault_resolution(&observations).unwrap();

    assert_eq!(
        plan.loads,
        vec![
            VulkanCompiledResourceDistributedFaultLoad {
                observation_index: 0,
                resource_indices: vec![2, 3],
            },
            VulkanCompiledResourceDistributedFaultLoad {
                observation_index: 1,
                resource_indices: vec![3],
            },
        ],
    );
    assert_eq!(plan.commit_observation_indices, vec![0, 1]);
}
