#[test]
fn physical_execution_residency_replaces_owner_tensors_without_double_counting() {
    let base = physical_execution_residency_base_plan(1_000, 100);
    let parameters = VulkanDistributedParameterAllocationPlan {
        allocations: vec![
            VulkanDistributedParameterAllocation {
                device_id: "owner".to_string(),
                tensor: "weight".to_string(),
                byte_offset: 0,
                byte_count: 300,
                use_count: 1,
            },
            VulkanDistributedParameterAllocation {
                device_id: "helper".to_string(),
                tensor: "weight".to_string(),
                byte_offset: 300,
                byte_count: 300,
                use_count: 1,
            },
        ],
        allocation_count: 2,
        tensor_count: 1,
        total_byte_capacity: 600,
    };
    let exclusions = VulkanDistributedParameterExclusionPlan {
        devices: vec![VulkanDistributedDeviceParameterExclusions {
            device_id: "owner".to_string(),
            tensors: vec!["weight".to_string()],
            total_byte_capacity: 600,
        }],
        device_count: 1,
        unique_tensor_count: 1,
        excluded_full_allocation_count: 1,
        excluded_full_byte_capacity: 600,
    };
    let activations =
        physical_execution_activation_plan(VulkanSharedResidentBufferRoute::ExternalDeviceLocal);

    let plan = VulkanRuntimePhysicalExecutionResidencyPlan::plan(
        &base,
        &["owner".to_string(), "helper".to_string()],
        &parameters,
        &exclusions,
        &activations,
    )
    .unwrap();
    let owner = plan
        .device_plans
        .iter()
        .find(|device| device.device_id == "owner")
        .unwrap();
    let helper = plan
        .device_plans
        .iter()
        .find(|device| device.device_id == "helper")
        .unwrap();

    assert_eq!(
        owner
            .breakdown
            .owner_parameter_bytes_before_distributed_replacement,
        800
    );
    assert_eq!(owner.breakdown.excluded_owner_parameter_bytes, 600);
    assert_eq!(owner.mount_device_local_bytes, 500);
    assert_eq!(helper.mount_device_local_bytes, 300);
    assert_eq!(owner.stream_device_local_bytes, 132);
    assert_eq!(helper.stream_device_local_bytes, 32);
    assert_eq!(owner.breakdown.owner_stream_device_bytes, 36);
    assert!(owner.resident_stream_device_allocations.is_empty());
    assert_eq!(plan.total_mount_device_local_bytes, 800);
    assert_eq!(plan.total_stream_device_local_bytes, 164);
    assert_eq!(plan.total_stream_shared_host_bytes, 0);
}

#[test]
fn physical_execution_residency_charges_shared_host_once_without_device_local_aliases() {
    let base = physical_execution_residency_base_plan(1_000, 100);
    let plan = VulkanRuntimePhysicalExecutionResidencyPlan::plan(
        &base,
        &["owner".to_string(), "helper".to_string()],
        &VulkanDistributedParameterAllocationPlan {
            allocations: Vec::new(),
            allocation_count: 0,
            tensor_count: 0,
            total_byte_capacity: 0,
        },
        &VulkanDistributedParameterExclusionPlan {
            devices: Vec::new(),
            device_count: 0,
            unique_tensor_count: 0,
            excluded_full_allocation_count: 0,
            excluded_full_byte_capacity: 0,
        },
        &physical_execution_activation_plan(VulkanSharedResidentBufferRoute::SharedHost),
    )
    .unwrap();

    assert_eq!(plan.total_stream_device_local_bytes, 68);
    assert_eq!(plan.total_stream_shared_host_bytes, 96);
    let error = admit_vulkan_runtime_physical_execution_stream(
        &plan,
        &BTreeMap::from([
            ("owner".to_string(), "physical-a".to_string()),
            ("helper".to_string(), "physical-b".to_string()),
        ]),
        &BTreeMap::from([
            ("physical-a".to_string(), usize::MAX),
            ("physical-b".to_string(), usize::MAX),
        ]),
        95,
    )
    .unwrap_err();
    assert!(error.to_string().contains("needs 96 shared-host bytes"));
}

#[test]
fn physical_execution_residency_admits_exact_execution_transients_atomically() {
    let base = physical_execution_residency_base_plan(1_000, 100);
    let mut plan = VulkanRuntimePhysicalExecutionResidencyPlan::plan(
        &base,
        &["owner".to_string(), "helper".to_string()],
        &VulkanDistributedParameterAllocationPlan {
            allocations: Vec::new(),
            allocation_count: 0,
            tensor_count: 0,
            total_byte_capacity: 0,
        },
        &VulkanDistributedParameterExclusionPlan {
            devices: Vec::new(),
            device_count: 0,
            unique_tensor_count: 0,
            excluded_full_allocation_count: 0,
            excluded_full_byte_capacity: 0,
        },
        &physical_execution_activation_plan(VulkanSharedResidentBufferRoute::SharedHost),
    )
    .unwrap();
    let baseline = plan.clone();
    let shared_transient = VulkanRuntimeSharedHostTransientAllocation {
        mode: VulkanRuntimeSharedHostTransientAllocationMode::Always,
        owner_device_id: "owner".to_string(),
        participant_device_ids: vec!["helper".to_string(), "owner".to_string()],
        byte_capacity: 19,
        concern: "test allocation".to_string(),
    };
    let device_transients = vec![
        VulkanRuntimeDeviceLocalTransientAllocation {
            logical_device_id: "owner".to_string(),
            byte_capacity: 33,
            concern: "owner test allocation".to_string(),
        },
        VulkanRuntimeDeviceLocalTransientAllocation {
            logical_device_id: "helper".to_string(),
            byte_capacity: 17,
            concern: "helper test allocation".to_string(),
        },
    ];

    plan.add_execution_transient_reservation(
        &device_transients,
        std::slice::from_ref(&shared_transient),
    )
    .unwrap();

    let owner = plan
        .device_plans
        .iter()
        .find(|device| device.device_id == "owner")
        .unwrap();
    let helper = plan
        .device_plans
        .iter()
        .find(|device| device.device_id == "helper")
        .unwrap();
    assert_eq!(owner.breakdown.execution_transient_device_bytes_per_stream, 33);
    assert_eq!(helper.breakdown.execution_transient_device_bytes_per_stream, 17);
    assert_eq!(owner.execution_transient_device_allocations.len(), 1);
    assert_eq!(helper.execution_transient_device_allocations.len(), 1);
    assert_eq!(
        plan.total_stream_device_local_bytes,
        baseline.total_stream_device_local_bytes + 50
    );
    assert_eq!(plan.execution_transient_shared_host_bytes_per_stream, 19);
    assert_eq!(
        plan.execution_transient_shared_host_allocations,
        vec![shared_transient]
    );
    assert_eq!(
        plan.total_stream_shared_host_bytes,
        baseline.total_stream_shared_host_bytes + 19
    );
    assert_eq!(
        physical_execution_stream_working_set_bytes(
            &plan,
            &BTreeSet::from(["owner".to_string(), "helper".to_string()]),
        )
        .unwrap(),
        plan.total_stream_device_local_bytes
    );
    assert!(
        physical_execution_stream_working_set_bytes(
            &plan,
            &BTreeSet::from(["absent".to_string()]),
        )
        .unwrap_err()
        .to_string()
        .contains("no stream plan")
    );

    let accepted = plan.clone();
    let repeated = plan
        .add_execution_transient_reservation(&[], &[])
        .unwrap_err();
    assert!(repeated.to_string().contains("already attached"));
    assert_eq!(plan, accepted);

    let mut invalid_plan = baseline;
    let invalid_original = invalid_plan.clone();
    let error = invalid_plan
        .add_execution_transient_reservation(
            &[
                VulkanRuntimeDeviceLocalTransientAllocation {
                    logical_device_id: "owner".to_string(),
                    byte_capacity: 1,
                    concern: "valid prefix".to_string(),
                },
                VulkanRuntimeDeviceLocalTransientAllocation {
                    logical_device_id: "absent".to_string(),
                    byte_capacity: 1,
                    concern: "invalid suffix".to_string(),
                },
            ],
            &[],
        )
        .unwrap_err();
    assert!(error.to_string().contains("absent logical device"));
    assert_eq!(invalid_plan, invalid_original);
}

#[test]
fn physical_execution_residency_aggregates_aliases_before_admission() {
    let base = physical_execution_residency_base_plan(1_000, 100);
    let plan = VulkanRuntimePhysicalExecutionResidencyPlan::plan(
        &base,
        &["owner".to_string(), "helper".to_string()],
        &VulkanDistributedParameterAllocationPlan {
            allocations: vec![VulkanDistributedParameterAllocation {
                device_id: "helper".to_string(),
                tensor: "helper.weight".to_string(),
                byte_offset: 0,
                byte_count: 200,
                use_count: 1,
            }],
            allocation_count: 1,
            tensor_count: 1,
            total_byte_capacity: 200,
        },
        &VulkanDistributedParameterExclusionPlan {
            devices: Vec::new(),
            device_count: 0,
            unique_tensor_count: 0,
            excluded_full_allocation_count: 0,
            excluded_full_byte_capacity: 0,
        },
        &physical_execution_activation_plan(VulkanSharedResidentBufferRoute::SharedHost),
    )
    .unwrap();
    let physical = BTreeMap::from([
        ("owner".to_string(), "same-physical".to_string()),
        ("helper".to_string(), "same-physical".to_string()),
    ]);

    let admitted = admit_vulkan_runtime_physical_execution_mount(
        &plan,
        &physical,
        &BTreeMap::from([("same-physical".to_string(), 1_000)]),
    )
    .unwrap();
    assert_eq!(admitted["same-physical"], 1_000);
    let error = admit_vulkan_runtime_physical_execution_mount(
        &plan,
        &physical,
        &BTreeMap::from([("same-physical".to_string(), 999)]),
    )
    .unwrap_err();
    assert!(error.to_string().contains("needs 1000 mount device bytes"));
}

#[test]
fn physical_execution_residency_defers_eager_selected_payload_to_its_physical_store() {
    let mut base = physical_execution_residency_base_plan(1_250, 100);
    base.residency_policy = ResourceResidencyPolicy::Eager;
    let device = &mut base.device_plans[0];
    device.parameter_residency.initial_dynamic_bytes = 200;
    device.parameter_residency.current_resident_bytes = 1_000;
    device.parameter_residency.maximum_addressable_bytes = 1_000;
    device
        .resource_store
        .maximum_dynamic_allocation_padding_bytes = 50;
    base.total_current_resident_parameter_bytes = 1_000;
    base.total_maximum_addressable_parameter_bytes = 1_000;

    let plan = VulkanRuntimePhysicalExecutionResidencyPlan::plan(
        &base,
        &["owner".to_string(), "helper".to_string()],
        &VulkanDistributedParameterAllocationPlan {
            allocations: Vec::new(),
            allocation_count: 0,
            tensor_count: 0,
            total_byte_capacity: 0,
        },
        &VulkanDistributedParameterExclusionPlan {
            devices: Vec::new(),
            device_count: 0,
            unique_tensor_count: 0,
            excluded_full_allocation_count: 0,
            excluded_full_byte_capacity: 0,
        },
        &physical_execution_activation_plan(VulkanSharedResidentBufferRoute::SharedHost),
    )
    .unwrap();
    let owner = plan
        .device_plans
        .iter()
        .find(|device| device.device_id == "owner")
        .unwrap();

    assert_eq!(owner.mount_device_local_bytes, 800);
    assert_eq!(
        owner.breakdown.independently_admitted_resource_store_bytes,
        350
    );
    assert_eq!(owner.stream_device_local_bytes, 36);
}

#[test]
fn physical_execution_residency_rejects_unknown_helpers_and_owner_underflow() {
    let base = physical_execution_residency_base_plan(1_000, 100);
    let unknown = VulkanRuntimePhysicalExecutionResidencyPlan::plan(
        &base,
        &["owner".to_string(), "helper".to_string()],
        &VulkanDistributedParameterAllocationPlan {
            allocations: vec![VulkanDistributedParameterAllocation {
                device_id: "unbound".to_string(),
                tensor: "weight".to_string(),
                byte_offset: 0,
                byte_count: 1,
                use_count: 1,
            }],
            allocation_count: 1,
            tensor_count: 1,
            total_byte_capacity: 1,
        },
        &VulkanDistributedParameterExclusionPlan {
            devices: Vec::new(),
            device_count: 0,
            unique_tensor_count: 0,
            excluded_full_allocation_count: 0,
            excluded_full_byte_capacity: 0,
        },
        &physical_execution_activation_plan(VulkanSharedResidentBufferRoute::SharedHost),
    )
    .unwrap_err();
    assert!(
        unknown
            .to_string()
            .contains("outside the physical execution plan")
    );

    let underflow = VulkanRuntimePhysicalExecutionResidencyPlan::plan(
        &base,
        &["owner".to_string(), "helper".to_string()],
        &VulkanDistributedParameterAllocationPlan {
            allocations: Vec::new(),
            allocation_count: 0,
            tensor_count: 0,
            total_byte_capacity: 0,
        },
        &VulkanDistributedParameterExclusionPlan {
            devices: vec![VulkanDistributedDeviceParameterExclusions {
                device_id: "owner".to_string(),
                tensors: vec!["too-large".to_string()],
                total_byte_capacity: 901,
            }],
            device_count: 1,
            unique_tensor_count: 1,
            excluded_full_allocation_count: 1,
            excluded_full_byte_capacity: 901,
        },
        &physical_execution_activation_plan(VulkanSharedResidentBufferRoute::SharedHost),
    )
    .unwrap_err();
    assert!(underflow.to_string().contains("contains only 800 bytes"));
}

#[test]
fn physical_execution_residency_rejects_corrupt_allocation_summaries() {
    let base = physical_execution_residency_base_plan(1_000, 100);
    let mut parameters = VulkanDistributedParameterAllocationPlan {
        allocations: vec![VulkanDistributedParameterAllocation {
            device_id: "helper".to_string(),
            tensor: "weight".to_string(),
            byte_offset: 0,
            byte_count: 64,
            use_count: 1,
        }],
        allocation_count: 1,
        tensor_count: 1,
        total_byte_capacity: 63,
    };
    let exclusions = VulkanDistributedParameterExclusionPlan {
        devices: Vec::new(),
        device_count: 0,
        unique_tensor_count: 0,
        excluded_full_allocation_count: 0,
        excluded_full_byte_capacity: 0,
    };

    let parameter_error = VulkanRuntimePhysicalExecutionResidencyPlan::plan(
        &base,
        &["owner".to_string(), "helper".to_string()],
        &parameters,
        &exclusions,
        &physical_execution_activation_plan(VulkanSharedResidentBufferRoute::SharedHost),
    )
    .unwrap_err();
    assert!(parameter_error.to_string().contains("summary disagrees"));

    parameters.total_byte_capacity = 64;
    let mut activations =
        physical_execution_activation_plan(VulkanSharedResidentBufferRoute::SharedHost);
    activations.total_private_byte_capacity = 31;
    let activation_error = VulkanRuntimePhysicalExecutionResidencyPlan::plan(
        &base,
        &["owner".to_string(), "helper".to_string()],
        &parameters,
        &exclusions,
        &activations,
    )
    .unwrap_err();
    assert!(activation_error.to_string().contains("summary disagrees"));
}

#[test]
fn physical_execution_stream_control_moves_to_one_shared_host_allocation_across_devices() {
    let mut base = physical_execution_residency_base_plan(1_000, 100);
    base.device_plans[0].breakdown.stream_control_bytes =
        VULKAN_STREAM_CONTROL_BYTE_CAPACITY;
    add_helper_stream_control_device(&mut base);
    let mut plan = VulkanRuntimePhysicalExecutionResidencyPlan::plan(
        &base,
        &["owner".to_string(), "helper".to_string()],
        &empty_physical_execution_parameter_allocations(),
        &empty_physical_execution_parameter_exclusions(),
        &physical_execution_activation_plan(VulkanSharedResidentBufferRoute::SharedHost),
    )
    .unwrap();
    let baseline = plan.clone();

    plan.bind_stream_control_memory_domain(&BTreeMap::from([
        ("owner".to_string(), "physical-a".to_string()),
        ("helper".to_string(), "physical-b".to_string()),
    ]))
    .unwrap();

    assert_eq!(
        plan.total_stream_device_local_bytes,
        baseline.total_stream_device_local_bytes - 2 * VULKAN_STREAM_CONTROL_BYTE_CAPACITY
    );
    assert_eq!(
        plan.total_stream_shared_host_bytes,
        baseline.total_stream_shared_host_bytes + VULKAN_STREAM_CONTROL_BYTE_CAPACITY
    );
    assert_eq!(
        plan.shared_stream_control_host_bytes_per_stream,
        VULKAN_STREAM_CONTROL_BYTE_CAPACITY
    );
    assert_eq!(
        plan.device_plans
            .iter()
            .map(|device| {
                device
                    .breakdown
                    .owner_stream_control_device_bytes_per_stream
            })
            .sum::<usize>(),
        0
    );
    assert_eq!(
        plan.device_plans
            .iter()
            .map(|device| device.stream_shared_host_bytes)
            .sum::<usize>(),
        plan.total_stream_shared_host_bytes
    );
}

#[test]
fn aliased_logical_devices_retain_exactly_one_device_local_stream_control() {
    let mut base = physical_execution_residency_base_plan(1_000, 100);
    base.device_plans[0].breakdown.stream_control_bytes =
        VULKAN_STREAM_CONTROL_BYTE_CAPACITY;
    add_helper_stream_control_device(&mut base);

    let mut plan = VulkanRuntimePhysicalExecutionResidencyPlan::plan(
        &base,
        &["owner".to_string(), "helper".to_string()],
        &empty_physical_execution_parameter_allocations(),
        &empty_physical_execution_parameter_exclusions(),
        &physical_execution_activation_plan(VulkanSharedResidentBufferRoute::SharedHost),
    )
    .unwrap();
    let baseline = plan.clone();

    plan.bind_stream_control_memory_domain(&BTreeMap::from([
        ("owner".to_string(), "same-physical".to_string()),
        ("helper".to_string(), "same-physical".to_string()),
    ]))
    .unwrap();

    assert_eq!(
        plan.device_plans
            .iter()
            .map(|device| {
                device
                    .breakdown
                    .owner_stream_control_device_bytes_per_stream
            })
            .sum::<usize>(),
        VULKAN_STREAM_CONTROL_BYTE_CAPACITY
    );
    assert_eq!(
        plan.total_stream_device_local_bytes,
        baseline.total_stream_device_local_bytes - VULKAN_STREAM_CONTROL_BYTE_CAPACITY
    );
    assert_eq!(
        plan.total_stream_shared_host_bytes,
        baseline.total_stream_shared_host_bytes
    );
    assert_eq!(plan.shared_stream_control_host_bytes_per_stream, 0);
}

#[test]
fn stream_control_binding_rejects_incomplete_repeated_or_extra_maps_atomically() {
    let mut base = physical_execution_residency_base_plan(1_000, 100);
    base.device_plans[0].breakdown.stream_control_bytes =
        VULKAN_STREAM_CONTROL_BYTE_CAPACITY;
    let mut plan = VulkanRuntimePhysicalExecutionResidencyPlan::plan(
        &base,
        &["owner".to_string(), "helper".to_string()],
        &empty_physical_execution_parameter_allocations(),
        &empty_physical_execution_parameter_exclusions(),
        &physical_execution_activation_plan(VulkanSharedResidentBufferRoute::SharedHost),
    )
    .unwrap();
    let original = plan.clone();

    let incomplete = plan
        .bind_stream_control_memory_domain(&BTreeMap::from([(
            "owner".to_string(),
            "physical-a".to_string(),
        )]))
        .unwrap_err();
    assert!(incomplete.to_string().contains("has no physical device"));
    assert_eq!(plan, original);

    let extra = plan
        .bind_stream_control_memory_domain(&BTreeMap::from([
            ("owner".to_string(), "physical-a".to_string()),
            ("helper".to_string(), "physical-b".to_string()),
            ("absent".to_string(), "physical-c".to_string()),
        ]))
        .unwrap_err();
    assert!(extra.to_string().contains("contains extra logical devices"));
    assert_eq!(plan, original);

    let valid_binding = BTreeMap::from([
        ("owner".to_string(), "physical-a".to_string()),
        ("helper".to_string(), "physical-b".to_string()),
    ]);
    plan.bind_stream_control_memory_domain(&valid_binding)
        .unwrap();
    let once_bound = plan.clone();
    let repeated = plan
        .bind_stream_control_memory_domain(&valid_binding)
        .unwrap_err();
    assert!(repeated.to_string().contains("bound more than once"));
    assert_eq!(plan, once_bound);
}

#[test]
fn execution_transient_ledgers_reject_malformed_allocations_atomically() {
    let base = physical_execution_residency_base_plan(1_000, 100);
    let mut plan = VulkanRuntimePhysicalExecutionResidencyPlan::plan(
        &base,
        &["owner".to_string(), "helper".to_string()],
        &empty_physical_execution_parameter_allocations(),
        &empty_physical_execution_parameter_exclusions(),
        &physical_execution_activation_plan(VulkanSharedResidentBufferRoute::SharedHost),
    )
    .unwrap();
    let original = plan.clone();

    for malformed in [
        VulkanRuntimeDeviceLocalTransientAllocation {
            logical_device_id: "".to_string(),
            byte_capacity: 19,
            concern: "missing device".to_string(),
        },
        VulkanRuntimeDeviceLocalTransientAllocation {
            logical_device_id: "owner".to_string(),
            byte_capacity: 0,
            concern: "zero capacity".to_string(),
        },
        VulkanRuntimeDeviceLocalTransientAllocation {
            logical_device_id: "owner".to_string(),
            byte_capacity: 19,
            concern: "".to_string(),
        },
    ] {
        let error = plan
            .add_execution_transient_reservation(&[malformed], &[])
            .unwrap_err();
        assert!(error.to_string().contains("is malformed"));
        assert_eq!(plan, original);
    }

    for malformed in [
        VulkanRuntimeSharedHostTransientAllocation {
            mode: VulkanRuntimeSharedHostTransientAllocationMode::Always,
            owner_device_id: "owner".to_string(),
            participant_device_ids: vec!["helper".to_string()],
            byte_capacity: 19,
            concern: "missing owner".to_string(),
        },
        VulkanRuntimeSharedHostTransientAllocation {
            mode: VulkanRuntimeSharedHostTransientAllocationMode::Always,
            owner_device_id: "owner".to_string(),
            participant_device_ids: vec!["owner".to_string(), "owner".to_string()],
            byte_capacity: 19,
            concern: "duplicate participant".to_string(),
        },
        VulkanRuntimeSharedHostTransientAllocation {
            mode: VulkanRuntimeSharedHostTransientAllocationMode::Always,
            owner_device_id: "owner".to_string(),
            participant_device_ids: vec!["owner".to_string()],
            byte_capacity: 0,
            concern: "zero capacity".to_string(),
        },
        VulkanRuntimeSharedHostTransientAllocation {
            mode: VulkanRuntimeSharedHostTransientAllocationMode::Always,
            owner_device_id: "owner".to_string(),
            participant_device_ids: vec!["absent".to_string(), "owner".to_string()],
            byte_capacity: 19,
            concern: "unknown participant".to_string(),
        },
        VulkanRuntimeSharedHostTransientAllocation {
            mode: VulkanRuntimeSharedHostTransientAllocationMode::CrossDeviceTimelineStaging,
            owner_device_id: "owner".to_string(),
            participant_device_ids: vec!["owner".to_string()],
            byte_capacity: 19,
            concern: "incomplete boundary".to_string(),
        },
    ] {
        let error = plan
            .add_execution_transient_reservation(&[], &[malformed])
            .unwrap_err();
        assert!(error.to_string().contains("is malformed"));
        assert_eq!(plan, original);
    }
}

#[test]
fn execution_transient_host_requirements_are_queried_and_aligned_per_allocation() {
    let allocations = vec![
        VulkanRuntimeSharedHostTransientAllocation {
            mode: VulkanRuntimeSharedHostTransientAllocationMode::Always,
            owner_device_id: "owner".to_string(),
            participant_device_ids: vec!["helper".to_string(), "owner".to_string()],
            byte_capacity: 17,
            concern: "signal".to_string(),
        },
        VulkanRuntimeSharedHostTransientAllocation {
            mode: VulkanRuntimeSharedHostTransientAllocationMode::CrossDeviceTimelineStaging,
            owner_device_id: "helper".to_string(),
            participant_device_ids: vec!["helper".to_string(), "owner".to_string()],
            byte_capacity: 29,
            concern: "edge".to_string(),
        },
    ];
    let mut queried = Vec::new();

    let exact = execution_transient_shared_host_requirement_bytes_with(
        &allocations,
        |allocation| {
            queried.push(allocation.concern.clone());
            Ok(allocation.byte_capacity.next_multiple_of(64))
        },
    )
    .unwrap();

    assert_eq!(queried, vec!["signal", "edge"]);
    assert_eq!(exact, 128);
    assert_ne!(exact, (17usize + 29).next_multiple_of(64));

    let overflow = execution_transient_shared_host_requirement_bytes_with(
        &allocations,
        |allocation| {
            Ok(if allocation.concern == "signal" {
                usize::MAX
            } else {
                1
            })
        },
    )
    .unwrap_err();
    assert!(overflow.to_string().contains("requirements overflowed"));
}

#[test]
fn execution_transient_device_requirements_are_queried_and_aligned_per_allocation() {
    let allocations = vec![
        VulkanRuntimeDeviceLocalTransientAllocation {
            logical_device_id: "owner".to_string(),
            byte_capacity: 17,
            concern: "signal".to_string(),
        },
        VulkanRuntimeDeviceLocalTransientAllocation {
            logical_device_id: "owner".to_string(),
            byte_capacity: 29,
            concern: "control".to_string(),
        },
    ];
    let mut queried = Vec::new();

    let exact = execution_transient_device_requirement_bytes_with(&allocations, |allocation| {
        queried.push(allocation.concern.clone());
        Ok(allocation.byte_capacity.next_multiple_of(64))
    })
    .unwrap();

    assert_eq!(queried, vec!["signal", "control"]);
    assert_eq!(exact, 128);
    assert_ne!(exact, (17usize + 29).next_multiple_of(64));

    let overflow =
        execution_transient_device_requirement_bytes_with(&allocations, |allocation| {
            Ok(if allocation.concern == "signal" {
                usize::MAX
            } else {
                1
            })
        })
        .unwrap_err();
    assert!(overflow.to_string().contains("requirements overflowed"));
}

#[test]
fn resident_stream_device_requirements_are_queried_and_aligned_per_allocation() {
    let allocations = vec![
        VulkanRuntimeResidentStreamAllocation {
            kind: VulkanRuntimeResidentStreamAllocationKind::State {
                component_id: "component".to_string(),
                state_id: "state".to_string(),
            },
            byte_capacity: 17,
        },
        VulkanRuntimeResidentStreamAllocation {
            kind: VulkanRuntimeResidentStreamAllocationKind::SelectionTelemetry {
                component_id: "component".to_string(),
                node_id: "node".to_string(),
                domain_id: "domain".to_string(),
            },
            byte_capacity: 29,
        },
    ];
    let mut queried = Vec::new();

    let exact = resident_stream_device_requirement_bytes_with(&allocations, |allocation| {
        queried.push(allocation.kind.clone());
        Ok(allocation.byte_capacity.next_multiple_of(64))
    })
    .unwrap();

    assert_eq!(queried.len(), 2);
    assert_eq!(exact, 128);
    assert_ne!(exact, (17usize + 29).next_multiple_of(64));

    let overflow = resident_stream_device_requirement_bytes_with(&allocations, |allocation| {
        Ok(if allocation.byte_capacity == 17 {
            usize::MAX
        } else {
            1
        })
    })
    .unwrap_err();
    assert!(overflow.to_string().contains("requirements overflowed"));
}

#[test]
fn external_resident_device_requirements_are_queried_per_physical_allocation() {
    let allocations = vec![
        VulkanRuntimeExternalDeviceLocalResidentAllocation {
            owner_device_id: "owner".to_string(),
            participant_device_ids: vec!["helper".to_string(), "owner".to_string()],
            byte_capacity: 17,
            concern: "first edge".to_string(),
        },
        VulkanRuntimeExternalDeviceLocalResidentAllocation {
            owner_device_id: "helper".to_string(),
            participant_device_ids: vec!["helper".to_string(), "owner".to_string()],
            byte_capacity: 29,
            concern: "second edge".to_string(),
        },
    ];
    let mut queried = Vec::new();

    let exact = external_device_local_resident_requirement_bytes_with(
        &allocations,
        |allocation| {
            queried.push(allocation.concern.clone());
            Ok(allocation.byte_capacity.next_multiple_of(64))
        },
    )
    .unwrap();

    assert_eq!(queried, vec!["first edge", "second edge"]);
    assert_eq!(exact, 128);
    assert_ne!(exact, (17usize + 29).next_multiple_of(64));

    let overflow = external_device_local_resident_requirement_bytes_with(
        &allocations,
        |allocation| {
            Ok(if allocation.concern == "first edge" {
                usize::MAX
            } else {
                1
            })
        },
    )
    .unwrap_err();
    assert!(overflow.to_string().contains("requirements overflowed"));
}

#[test]
fn resident_shared_host_requirements_are_queried_per_physical_allocation() {
    let allocations = vec![
        VulkanRuntimeSharedHostResidentAllocation {
            owner_device_id: "owner".to_string(),
            participant_device_ids: vec!["helper".to_string(), "owner".to_string()],
            byte_capacity: 17,
            concern: "first staging edge".to_string(),
        },
        VulkanRuntimeSharedHostResidentAllocation {
            owner_device_id: "helper".to_string(),
            participant_device_ids: vec!["helper".to_string(), "owner".to_string()],
            byte_capacity: 29,
            concern: "second staging edge".to_string(),
        },
    ];
    let mut queried = Vec::new();

    let exact = resident_shared_host_requirement_bytes_with(&allocations, |allocation| {
        queried.push(allocation.concern.clone());
        Ok(allocation.byte_capacity.next_multiple_of(64))
    })
    .unwrap();

    assert_eq!(queried, vec!["first staging edge", "second staging edge"]);
    assert_eq!(exact, 128);
    assert_ne!(exact, (17usize + 29).next_multiple_of(64));

    let overflow = resident_shared_host_requirement_bytes_with(&allocations, |allocation| {
        Ok(if allocation.concern == "first staging edge" {
            usize::MAX
        } else {
            1
        })
    })
    .unwrap_err();
    assert!(overflow.to_string().contains("requirements overflowed"));
}

#[test]
fn physical_execution_residency_rejects_malformed_resident_ledgers_and_overrides() {
    let parameters = empty_physical_execution_parameter_allocations();
    let exclusions = empty_physical_execution_parameter_exclusions();
    let activations =
        physical_execution_activation_plan(VulkanSharedResidentBufferRoute::SharedHost);

    let mut repeated = physical_execution_residency_base_plan(1_000, 100);
    let repeated_allocation =
        repeated.device_plans[0].resident_stream_device_allocations[0].clone();
    repeated.device_plans[0]
        .resident_stream_device_allocations
        .push(repeated_allocation);
    repeated.device_plans[0].breakdown.activation_slot_bytes = 128;
    repeated.device_plans[0].working_set.activation_headroom_bytes = 128;
    repeated.device_plans[0].initial_device_resident_bytes = 1_064;
    repeated.total_initial_device_resident_bytes = 1_064;
    let repeated_error = VulkanRuntimePhysicalExecutionResidencyPlan::plan(
        &repeated,
        &["owner".to_string(), "helper".to_string()],
        &parameters,
        &exclusions,
        &activations,
    )
    .unwrap_err();
    assert!(repeated_error.to_string().contains("repeated resident stream allocation"));

    let mut mismatched = activations.clone();
    mismatched.allocations[0].byte_capacity = 65;
    mismatched.total_shared_byte_capacity = 97;
    let mismatch_error = VulkanRuntimePhysicalExecutionResidencyPlan::plan(
        &physical_execution_residency_base_plan(1_000, 100),
        &["owner".to_string(), "helper".to_string()],
        &parameters,
        &exclusions,
        &mismatched,
    )
    .unwrap_err();
    assert!(mismatch_error.to_string().contains("replaces a 64-byte resident allocation with 65"));

    let mut missing = physical_execution_residency_base_plan(1_000, 100);
    missing.device_plans[0]
        .resident_stream_device_allocations
        .clear();
    let missing_error = VulkanRuntimePhysicalExecutionResidencyPlan::plan(
        &missing,
        &["owner".to_string(), "helper".to_string()],
        &parameters,
        &exclusions,
        &activations,
    )
    .unwrap_err();
    assert!(missing_error.to_string().contains("activation slot allocation bytes"));
}

#[test]
fn physical_execution_residency_replaces_boundary_storage_by_exact_identity() {
    for (storage, kind) in [
        (
            VulkanDistributedActivationStorage::BoundaryInput,
            VulkanRuntimeResidentStreamAllocationKind::BoundaryInput {
                component_id: "component".to_string(),
                signal_id: "signal".to_string(),
            },
        ),
        (
            VulkanDistributedActivationStorage::BoundaryOutput,
            VulkanRuntimeResidentStreamAllocationKind::BoundaryOutput {
                component_id: "component".to_string(),
                signal_id: "signal".to_string(),
            },
        ),
    ] {
        let mut base = physical_execution_residency_base_plan(1_000, 100);
        base.device_plans[0].breakdown.activation_slot_bytes = 0;
        base.device_plans[0].breakdown.boundary_buffer_bytes = 64;
        base.device_plans[0].resident_stream_device_allocations =
            vec![VulkanRuntimeResidentStreamAllocation {
                kind,
                byte_capacity: 64,
            }];
        let mut activations =
            physical_execution_activation_plan(VulkanSharedResidentBufferRoute::SharedHost);
        activations.allocations[0].storage = storage;

        let plan = VulkanRuntimePhysicalExecutionResidencyPlan::plan(
            &base,
            &["owner".to_string(), "helper".to_string()],
            &empty_physical_execution_parameter_allocations(),
            &empty_physical_execution_parameter_exclusions(),
            &activations,
        )
        .unwrap();
        let owner = plan
            .device_plans
            .iter()
            .find(|device| device.device_id == "owner")
            .unwrap();
        assert_eq!(owner.breakdown.owner_stream_device_bytes, 36);
        assert!(owner.resident_stream_device_allocations.is_empty());

        activations.allocations[0].signal_ids = vec!["other".to_string()];
        let identity_error = VulkanRuntimePhysicalExecutionResidencyPlan::plan(
            &base,
            &["owner".to_string(), "helper".to_string()],
            &empty_physical_execution_parameter_allocations(),
            &empty_physical_execution_parameter_exclusions(),
            &activations,
        )
        .unwrap_err();
        assert!(identity_error.to_string().contains("replaces 0 resident allocations"));
    }
}

#[test]
fn graph_edge_binding_charges_external_device_local_storage_once() {
    let (base, edge_plans) = physical_execution_edge_base_plan();
    let mut plan = VulkanRuntimePhysicalExecutionResidencyPlan::plan(
        &base,
        &["owner".to_string(), "helper".to_string()],
        &empty_physical_execution_parameter_allocations(),
        &empty_physical_execution_parameter_exclusions(),
        &empty_physical_execution_activation_plan(),
    )
    .unwrap();
    let original_total = plan.total_stream_device_local_bytes;

    plan.bind_graph_edge_memory_domains(
        &edge_plans,
        &empty_physical_execution_activation_plan(),
        &BTreeMap::from([(
            5,
            physical_execution_edge_route(VulkanPlacedEdgeTransferRoute::ExternalDeviceLocal),
        )]),
        &BTreeMap::from([
            ("owner".to_string(), "physical-a".to_string()),
            ("helper".to_string(), "physical-b".to_string()),
        ]),
    )
    .unwrap();

    let owner = plan
        .device_plans
        .iter()
        .find(|device| device.device_id == "owner")
        .unwrap();
    let helper = plan
        .device_plans
        .iter()
        .find(|device| device.device_id == "helper")
        .unwrap();
    assert!(plan.graph_edge_memory_domains_bound);
    assert_eq!(owner.external_device_local_resident_allocations.len(), 1);
    assert_eq!(owner.breakdown.external_edge_device_bytes_per_stream, 8_192);
    assert_eq!(helper.stream_device_local_bytes, 0);
    assert_eq!(plan.total_stream_device_local_bytes, original_total - 8_192);
    assert_eq!(plan.total_stream_shared_host_bytes, 0);
}

#[test]
fn graph_edge_binding_charges_staging_per_physical_device_and_host_once() {
    let (base, edge_plans) = physical_execution_edge_base_plan();
    let mut plan = VulkanRuntimePhysicalExecutionResidencyPlan::plan(
        &base,
        &["owner".to_string(), "helper".to_string()],
        &empty_physical_execution_parameter_allocations(),
        &empty_physical_execution_parameter_exclusions(),
        &empty_physical_execution_activation_plan(),
    )
    .unwrap();
    let original_total = plan.total_stream_device_local_bytes;

    plan.bind_graph_edge_memory_domains(
        &edge_plans,
        &empty_physical_execution_activation_plan(),
        &BTreeMap::from([(
            5,
            physical_execution_edge_route(VulkanPlacedEdgeTransferRoute::DeviceLocalStaging),
        )]),
        &BTreeMap::from([
            ("owner".to_string(), "physical-a".to_string()),
            ("helper".to_string(), "physical-b".to_string()),
        ]),
    )
    .unwrap();

    assert_eq!(plan.total_stream_device_local_bytes, original_total);
    assert_eq!(plan.total_stream_shared_host_bytes, 8_192);
    assert_eq!(plan.resident_shared_host_allocations.len(), 1);
    assert_eq!(
        plan.device_plans
            .iter()
            .map(|device| device.breakdown.owner_edge_buffer_bytes_per_stream)
            .sum::<usize>(),
        2 * 8_192
    );
}

#[test]
fn graph_edge_binding_collapses_colocated_logical_endpoints() {
    let (base, edge_plans) = physical_execution_edge_base_plan();
    let mut plan = VulkanRuntimePhysicalExecutionResidencyPlan::plan(
        &base,
        &["owner".to_string(), "helper".to_string()],
        &empty_physical_execution_parameter_allocations(),
        &empty_physical_execution_parameter_exclusions(),
        &empty_physical_execution_activation_plan(),
    )
    .unwrap();
    let original_total = plan.total_stream_device_local_bytes;

    plan.bind_graph_edge_memory_domains(
        &edge_plans,
        &empty_physical_execution_activation_plan(),
        &BTreeMap::new(),
        &BTreeMap::from([
            ("owner".to_string(), "same-physical".to_string()),
            ("helper".to_string(), "same-physical".to_string()),
        ]),
    )
    .unwrap();

    assert_eq!(plan.total_stream_device_local_bytes, original_total - 8_192);
    assert_eq!(plan.total_stream_shared_host_bytes, 0);
    assert!(plan
        .device_plans
        .iter()
        .all(|device| device.external_device_local_resident_allocations.is_empty()));
}

#[test]
fn graph_edge_binding_rejects_missing_and_repeated_routes_atomically() {
    let (base, edge_plans) = physical_execution_edge_base_plan();
    let mut plan = VulkanRuntimePhysicalExecutionResidencyPlan::plan(
        &base,
        &["owner".to_string(), "helper".to_string()],
        &empty_physical_execution_parameter_allocations(),
        &empty_physical_execution_parameter_exclusions(),
        &empty_physical_execution_activation_plan(),
    )
    .unwrap();
    let original = plan.clone();
    let physical = BTreeMap::from([
        ("owner".to_string(), "physical-a".to_string()),
        ("helper".to_string(), "physical-b".to_string()),
    ]);

    let missing = plan
        .bind_graph_edge_memory_domains(
            &edge_plans,
            &empty_physical_execution_activation_plan(),
            &BTreeMap::new(),
            &physical,
        )
        .unwrap_err();
    assert!(missing.to_string().contains("without an exact mounted route"));
    assert_eq!(plan, original);

    plan.bind_graph_edge_memory_domains(
        &edge_plans,
        &empty_physical_execution_activation_plan(),
        &BTreeMap::from([(
            5,
            physical_execution_edge_route(VulkanPlacedEdgeTransferRoute::ExternalDeviceLocal),
        )]),
        &physical,
    )
    .unwrap();
    let once_bound = plan.clone();
    let repeated = plan
        .bind_graph_edge_memory_domains(
            &edge_plans,
            &empty_physical_execution_activation_plan(),
            &BTreeMap::from([(
                5,
                physical_execution_edge_route(
                    VulkanPlacedEdgeTransferRoute::ExternalDeviceLocal,
                ),
            )]),
            &physical,
        )
        .unwrap_err();
    assert!(repeated.to_string().contains("already bound"));
    assert_eq!(plan, once_bound);
}

#[test]
fn graph_edge_binding_includes_distributed_only_staging_participants() {
    let (mut base, edge_plans) = physical_execution_edge_base_plan();
    let mut shard = base.device_plans[1].clone();
    shard.device_id = "shard".to_string();
    shard.working_set = VulkanRuntimeWorkingSetBytes::default();
    shard.breakdown = VulkanRuntimeDeviceResidencyBreakdown::default();
    shard.resident_stream_device_allocations.clear();
    shard.initial_device_resident_bytes = 0;
    base.device_plans.push(shard);
    let activations = VulkanDistributedActivationBufferPlan {
        allocations: vec![VulkanDistributedActivationBufferAllocation {
            storage: VulkanDistributedActivationStorage::Edge {
                edge_index: 5,
                owner_device_id: "owner".to_string(),
            },
            owner_device_id: "owner".to_string(),
            component_id: "input_adapter".to_string(),
            slot: 5,
            byte_capacity: 8_192,
            signal_ids: vec!["shared_context".to_string()],
            device_ids: vec![
                "helper".to_string(),
                "owner".to_string(),
                "shard".to_string(),
            ],
            input_use_count: 1,
            output_use_count: 1,
        }],
        reduction_allocations: Vec::new(),
        private_intermediate_allocations: Vec::new(),
        allocation_count: 1,
        import_count: 2,
        reference_count: 2,
        total_shared_byte_capacity: 8_192,
        total_private_byte_capacity: 0,
        route: VulkanSharedResidentBufferRoute::SharedHost,
    };
    let mut plan = VulkanRuntimePhysicalExecutionResidencyPlan::plan(
        &base,
        &[
            "owner".to_string(),
            "helper".to_string(),
            "shard".to_string(),
        ],
        &empty_physical_execution_parameter_allocations(),
        &empty_physical_execution_parameter_exclusions(),
        &activations,
    )
    .unwrap();
    let original_total = plan.total_stream_device_local_bytes;

    plan.bind_graph_edge_memory_domains(
        &edge_plans,
        &activations,
        &BTreeMap::from([(
            5,
            physical_execution_edge_route(VulkanPlacedEdgeTransferRoute::DeviceLocalStaging),
        )]),
        &BTreeMap::from([
            ("owner".to_string(), "physical-a".to_string()),
            ("helper".to_string(), "physical-b".to_string()),
            ("shard".to_string(), "physical-c".to_string()),
        ]),
    )
    .unwrap();

    let shard = plan
        .device_plans
        .iter()
        .find(|device| device.device_id == "shard")
        .unwrap();
    assert_eq!(shard.stream_device_local_bytes, 8_192);
    assert!(shard.resident_stream_device_allocations.iter().any(|allocation| {
        matches!(
            &allocation.kind,
            VulkanRuntimeResidentStreamAllocationKind::EdgeStagingReplica { .. }
        )
    }));
    assert_eq!(plan.total_stream_device_local_bytes, original_total + 8_192);
    assert_eq!(plan.total_stream_shared_host_bytes, 8_192);
    assert_eq!(
        plan.resident_shared_host_allocations[0]
            .participant_device_ids
            .len(),
        3
    );
}

#[test]
fn graph_edge_binding_rejects_conflicting_distributed_route_atomically() {
    let (base, edge_plans) = physical_execution_edge_base_plan();
    let activations = VulkanDistributedActivationBufferPlan {
        allocations: vec![VulkanDistributedActivationBufferAllocation {
            storage: VulkanDistributedActivationStorage::Edge {
                edge_index: 5,
                owner_device_id: "owner".to_string(),
            },
            owner_device_id: "owner".to_string(),
            component_id: "input_adapter".to_string(),
            slot: 5,
            byte_capacity: 8_192,
            signal_ids: vec!["shared_context".to_string()],
            device_ids: vec!["helper".to_string(), "owner".to_string()],
            input_use_count: 1,
            output_use_count: 1,
        }],
        reduction_allocations: Vec::new(),
        private_intermediate_allocations: Vec::new(),
        allocation_count: 1,
        import_count: 1,
        reference_count: 1,
        total_shared_byte_capacity: 8_192,
        total_private_byte_capacity: 0,
        route: VulkanSharedResidentBufferRoute::SharedHost,
    };
    let mut plan = VulkanRuntimePhysicalExecutionResidencyPlan::plan(
        &base,
        &["owner".to_string(), "helper".to_string()],
        &empty_physical_execution_parameter_allocations(),
        &empty_physical_execution_parameter_exclusions(),
        &activations,
    )
    .unwrap();
    let original = plan.clone();

    let error = plan
        .bind_graph_edge_memory_domains(
            &edge_plans,
            &activations,
            &BTreeMap::from([(
                5,
                physical_execution_edge_route(
                    VulkanPlacedEdgeTransferRoute::ExternalDeviceLocal,
                ),
            )]),
            &BTreeMap::from([
                ("owner".to_string(), "physical-a".to_string()),
                ("helper".to_string(), "physical-b".to_string()),
            ]),
        )
        .unwrap_err();

    assert!(error
        .to_string()
        .contains("incompatible selected and distributed routes"));
    assert_eq!(plan, original);
}

fn empty_physical_execution_parameter_allocations(
) -> VulkanDistributedParameterAllocationPlan {
    VulkanDistributedParameterAllocationPlan {
        allocations: Vec::new(),
        allocation_count: 0,
        tensor_count: 0,
        total_byte_capacity: 0,
    }
}

fn empty_physical_execution_parameter_exclusions() -> VulkanDistributedParameterExclusionPlan {
    VulkanDistributedParameterExclusionPlan {
        devices: Vec::new(),
        device_count: 0,
        unique_tensor_count: 0,
        excluded_full_allocation_count: 0,
        excluded_full_byte_capacity: 0,
    }
}

fn empty_physical_execution_activation_plan() -> VulkanDistributedActivationBufferPlan {
    VulkanDistributedActivationBufferPlan {
        allocations: Vec::new(),
        reduction_allocations: Vec::new(),
        private_intermediate_allocations: Vec::new(),
        allocation_count: 0,
        import_count: 0,
        reference_count: 0,
        total_shared_byte_capacity: 0,
        total_private_byte_capacity: 0,
        route: VulkanSharedResidentBufferRoute::ExternalDeviceLocal,
    }
}

fn physical_execution_edge_route(
    route: VulkanPlacedEdgeTransferRoute,
) -> VulkanRuntimeMountedBoundaryRoute {
    VulkanRuntimeMountedBoundaryRoute {
        edge_index: 5,
        source_device_id: "owner".to_string(),
        destination_device_id: "helper".to_string(),
        frame_byte_count: 8_192,
        route,
    }
}

fn physical_execution_edge_base_plan() -> (VulkanRuntimeResidencyPlan, Vec<VulkanPlacedEdgeIoPlan>) {
    let mut outgoing = outgoing_fanout_endpoint(0, 5, "helper", "consumer");
    outgoing.local_device_id = "owner".to_string();
    outgoing.remote_device_id = "helper".to_string();
    outgoing.transport = EdgeTransport::CrossDevice {
        from_device_id: "owner".to_string(),
        to_device_id: "helper".to_string(),
    };
    let incoming = incoming_fanout_endpoint(&outgoing);
    let owner_edge_plan = VulkanPlacedEdgeIoPlan {
        backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
        device_id: "owner".to_string(),
        signal_element_bytes: Some(2),
        local_edges: Vec::new(),
        endpoints: vec![outgoing],
        local_edge_count: 0,
        incoming_endpoint_count: 0,
        outgoing_endpoint_count: 1,
        total_buffer_count: 1,
        total_endpoint_count: 1,
        total_byte_capacity: Some(8_192),
        unresolved_byte_edges: Vec::new(),
    };
    let helper_edge_plan = VulkanPlacedEdgeIoPlan {
        backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
        device_id: "helper".to_string(),
        signal_element_bytes: Some(2),
        local_edges: Vec::new(),
        endpoints: vec![incoming],
        local_edge_count: 0,
        incoming_endpoint_count: 1,
        outgoing_endpoint_count: 0,
        total_buffer_count: 1,
        total_endpoint_count: 1,
        total_byte_capacity: Some(8_192),
        unresolved_byte_edges: Vec::new(),
    };

    let mut base = physical_execution_residency_base_plan(1_000 + 8_192, 100);
    let owner = &mut base.device_plans[0];
    owner.working_set.activation_headroom_bytes += 8_192;
    owner.breakdown.edge_buffer_bytes = 8_192;
    owner
        .resident_stream_device_allocations
        .push(VulkanRuntimeResidentStreamAllocation {
            kind: VulkanRuntimeResidentStreamAllocationKind::EdgeProducedPort {
                component_id: "input_adapter".to_string(),
                port_id: "shared_context".to_string(),
                edge_indices: vec![5],
            },
            byte_capacity: 8_192,
        });
    let mut helper = owner.clone();
    helper.device_id = "helper".to_string();
    helper.parameter_residency = VulkanRuntimeParameterResidencyBytes::default();
    helper.resource_store = VulkanCompiledResourceStoreResidencyBytes::default();
    helper.working_set = VulkanRuntimeWorkingSetBytes {
        transient_state_bytes: 0,
        activation_headroom_bytes: 8_192,
    };
    helper.breakdown = VulkanRuntimeDeviceResidencyBreakdown {
        edge_buffer_bytes: 8_192,
        ..VulkanRuntimeDeviceResidencyBreakdown::default()
    };
    helper.resident_stream_device_allocations = vec![VulkanRuntimeResidentStreamAllocation {
        kind: VulkanRuntimeResidentStreamAllocationKind::EdgeIncoming { edge_index: 5 },
        byte_capacity: 8_192,
    }];
    helper.initial_device_resident_bytes = 8_192;
    base.device_plans.push(helper);
    base.total_initial_device_resident_bytes += 8_192;
    (
        base,
        vec![owner_edge_plan, helper_edge_plan],
    )
}

fn add_helper_stream_control_device(base: &mut VulkanRuntimeResidencyPlan) {
    let mut helper = base.device_plans[0].clone();
    helper.device_id = "helper".to_string();
    helper.parameter_residency = VulkanRuntimeParameterResidencyBytes {
        always_resident_bytes: 0,
        initial_dynamic_bytes: 0,
        current_resident_bytes: 0,
        maximum_addressable_bytes: 0,
        staging_headroom_bytes: 0,
    };
    helper.resource_store = VulkanCompiledResourceStoreResidencyBytes::default();
    helper.working_set = VulkanRuntimeWorkingSetBytes {
        transient_state_bytes: VULKAN_STREAM_CONTROL_BYTE_CAPACITY,
        activation_headroom_bytes: 0,
    };
    helper.breakdown = VulkanRuntimeDeviceResidencyBreakdown {
        stream_control_bytes: VULKAN_STREAM_CONTROL_BYTE_CAPACITY,
        ..VulkanRuntimeDeviceResidencyBreakdown::default()
    };
    helper.resident_stream_device_allocations.clear();
    helper.initial_device_resident_bytes = VULKAN_STREAM_CONTROL_BYTE_CAPACITY;
    base.device_plans.push(helper);
    base.total_initial_device_resident_bytes += VULKAN_STREAM_CONTROL_BYTE_CAPACITY;
}

fn physical_execution_residency_base_plan(
    initial_device_resident_bytes: usize,
    resource_store_bytes: usize,
) -> VulkanRuntimeResidencyPlan {
    VulkanRuntimeResidencyPlan {
        schema: VULKAN_RUNTIME_RESIDENCY_PLAN_SCHEMA.to_string(),
        package_id: "package".to_string(),
        residency_policy: ResourceResidencyPolicy::DemandRetained,
        context_capacity_activations: 1,
        speculative_draft_tokens: 0,
        device_plans: vec![VulkanRuntimeDeviceResidencyPlan {
            device_id: "owner".to_string(),
            parameter_residency: VulkanRuntimeParameterResidencyBytes {
                always_resident_bytes: 800,
                initial_dynamic_bytes: 0,
                current_resident_bytes: 800,
                maximum_addressable_bytes: 800,
                staging_headroom_bytes: 0,
            },
            resource_store: VulkanCompiledResourceStoreResidencyBytes {
                metadata_device_bytes: resource_store_bytes,
                ..VulkanCompiledResourceStoreResidencyBytes::default()
            },
            working_set: VulkanRuntimeWorkingSetBytes {
                transient_state_bytes: 36,
                activation_headroom_bytes: 64,
            },
            breakdown: VulkanRuntimeDeviceResidencyBreakdown {
                activation_slot_bytes: 64,
                ..VulkanRuntimeDeviceResidencyBreakdown::default()
            },
            resident_stream_device_allocations: vec![VulkanRuntimeResidentStreamAllocation {
                kind: VulkanRuntimeResidentStreamAllocationKind::ActivationSlot {
                    component_id: "component".to_string(),
                    slot: 0,
                },
                byte_capacity: 64,
            }],
            initial_device_resident_bytes,
        }],
        total_initial_device_resident_bytes: initial_device_resident_bytes,
        total_current_resident_parameter_bytes: 800,
        total_maximum_addressable_parameter_bytes: 800,
    }
}

fn physical_execution_activation_plan(
    route: VulkanSharedResidentBufferRoute,
) -> VulkanDistributedActivationBufferPlan {
    VulkanDistributedActivationBufferPlan {
        allocations: vec![VulkanDistributedActivationBufferAllocation {
            storage: VulkanDistributedActivationStorage::ActivationSlot,
            owner_device_id: "owner".to_string(),
            component_id: "component".to_string(),
            slot: 0,
            byte_capacity: 64,
            signal_ids: vec!["signal".to_string()],
            device_ids: vec!["owner".to_string(), "helper".to_string()],
            input_use_count: 1,
            output_use_count: 1,
        }],
        reduction_allocations: vec![VulkanDistributedReductionBufferAllocation {
            owner_device_id: "owner".to_string(),
            dispatch_index: 0,
            component_id: "component".to_string(),
            node_id: "node".to_string(),
            plane_byte_capacity: 16,
            byte_capacity: 32,
            device_ids: vec!["owner".to_string(), "helper".to_string()],
        }],
        private_intermediate_allocations: vec![
            VulkanDistributedPrivateIntermediateBufferAllocation {
                producer_dispatch_index: 0,
                consumer_dispatch_index: 1,
                component_id: "component".to_string(),
                signal_id: "private".to_string(),
                devices: vec![VulkanDistributedPrivateIntermediateDeviceAllocation {
                    device_id: "helper".to_string(),
                    byte_capacity: 32,
                }],
            },
        ],
        allocation_count: 3,
        import_count: 4,
        reference_count: 8,
        total_shared_byte_capacity: 96,
        total_private_byte_capacity: 32,
        route,
    }
}
