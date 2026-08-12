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
    let mutation = coordinator.begin_mutation().unwrap();
    assert!(std::ptr::eq(mutation.coordinator(), &coordinator));
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
