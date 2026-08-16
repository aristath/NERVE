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
    assert!(owner.private_activation_resident_allocations.is_empty());
    assert_eq!(
        helper.private_activation_resident_allocations,
        vec![VulkanRuntimePrivateActivationResidentAllocation {
            producer_dispatch_index: 0,
            consumer_dispatch_index: 1,
            component_id: "component".to_string(),
            signal_id: "private".to_string(),
            byte_capacity: 32,
        }],
    );
    validate_vulkan_runtime_private_activation_residency(&plan, &activations).unwrap();
    assert_eq!(plan.total_mount_device_local_bytes, 800);
    assert_eq!(plan.total_stream_device_local_bytes, 164);
    assert_eq!(plan.total_stream_shared_host_bytes, 0);
}

#[test]
fn private_activation_resident_requirements_are_queried_per_physical_allocation() {
    let allocations = vec![
        VulkanRuntimePrivateActivationResidentAllocation {
            producer_dispatch_index: 1,
            consumer_dispatch_index: 2,
            component_id: "layer_00".to_string(),
            signal_id: "first".to_string(),
            byte_capacity: 17,
        },
        VulkanRuntimePrivateActivationResidentAllocation {
            producer_dispatch_index: 3,
            consumer_dispatch_index: 4,
            component_id: "layer_00".to_string(),
            signal_id: "second".to_string(),
            byte_capacity: 29,
        },
    ];
    let mut queried = Vec::new();

    let exact = private_activation_resident_requirement_bytes_with(
        &allocations,
        |allocation| {
            queried.push(allocation.signal_id.clone());
            Ok(allocation.byte_capacity.next_multiple_of(64))
        },
    )
    .unwrap();

    assert_eq!(queried, vec!["first", "second"]);
    assert_eq!(exact, 128);
    assert_ne!(exact, (17usize + 29).next_multiple_of(64));
}

#[test]
fn private_activation_mount_validation_rejects_missing_and_altered_ledgers() {
    let base = physical_execution_residency_base_plan(1_000, 100);
    let activations =
        physical_execution_activation_plan(VulkanSharedResidentBufferRoute::SharedHost);
    let plan = VulkanRuntimePhysicalExecutionResidencyPlan::plan(
        &base,
        &["owner".to_string(), "helper".to_string()],
        &empty_physical_execution_parameter_allocations(),
        &empty_physical_execution_parameter_exclusions(),
        &activations,
    )
    .unwrap();

    validate_vulkan_runtime_private_activation_residency(&plan, &activations).unwrap();

    let mut missing = plan.clone();
    missing
        .device_plans
        .iter_mut()
        .find(|device| device.device_id == "helper")
        .unwrap()
        .private_activation_resident_allocations
        .clear();
    assert!(
        validate_vulkan_runtime_private_activation_residency(&missing, &activations)
            .unwrap_err()
            .to_string()
            .contains("disagrees with the mounted distributed plan")
    );

    let mut altered = plan;
    altered
        .device_plans
        .iter_mut()
        .find(|device| device.device_id == "helper")
        .unwrap()
        .private_activation_resident_allocations[0]
        .byte_capacity += 1;
    assert!(
        validate_vulkan_runtime_private_activation_residency(&altered, &activations)
            .unwrap_err()
            .to_string()
            .contains("disagrees with the mounted distributed plan")
    );
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
        allocation_class: VulkanRuntimeStreamAllocationClass::Permanent,
    };
    let device_transients = vec![
        VulkanRuntimeDeviceLocalTransientAllocation {
            logical_device_id: "owner".to_string(),
            participant_device_ids: vec!["owner".to_string()],
            byte_capacity: 33,
            concern: "owner test allocation".to_string(),
            usage: VulkanRuntimeDeviceLocalTransientAllocationUsage::Storage,
            allocation_class: VulkanRuntimeStreamAllocationClass::Permanent,
        },
        VulkanRuntimeDeviceLocalTransientAllocation {
            logical_device_id: "helper".to_string(),
            participant_device_ids: vec!["helper".to_string()],
            byte_capacity: 17,
            concern: "helper test allocation".to_string(),
            usage: VulkanRuntimeDeviceLocalTransientAllocationUsage::Storage,
            allocation_class: VulkanRuntimeStreamAllocationClass::PromptRunner,
        },
    ];
    let host_visible_transient = VulkanRuntimeHostVisibleTransientAllocation {
        logical_device_id: "helper".to_string(),
        byte_capacity: 13,
        concern: "helper host control".to_string(),
        allocation_class: VulkanRuntimeStreamAllocationClass::PromptRunner,
    };

    plan.add_execution_transient_reservation(
        &device_transients,
        std::slice::from_ref(&host_visible_transient),
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
    assert_eq!(
        helper
            .breakdown
            .execution_transient_host_visible_bytes_per_stream,
        13
    );
    assert_eq!(owner.execution_transient_device_allocations.len(), 1);
    assert_eq!(helper.execution_transient_device_allocations.len(), 1);
    assert_eq!(
        plan.total_stream_device_local_bytes,
        baseline.total_stream_device_local_bytes + 50
    );
    assert_eq!(plan.execution_transient_shared_host_bytes_per_stream, 19);
    assert_eq!(
        plan.execution_transient_host_visible_allocations,
        vec![host_visible_transient]
    );
    assert_eq!(
        plan.execution_transient_shared_host_allocations,
        vec![shared_transient]
    );
    assert_eq!(
        plan.total_stream_shared_host_bytes,
        baseline.total_stream_shared_host_bytes + 32
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
        .add_execution_transient_reservation(&[], &[], &[])
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
                    participant_device_ids: vec!["owner".to_string()],
                    byte_capacity: 1,
                    concern: "valid prefix".to_string(),
                    usage: VulkanRuntimeDeviceLocalTransientAllocationUsage::Storage,
                    allocation_class: VulkanRuntimeStreamAllocationClass::Permanent,
                },
                VulkanRuntimeDeviceLocalTransientAllocation {
                    logical_device_id: "absent".to_string(),
                    participant_device_ids: vec!["absent".to_string()],
                    byte_capacity: 1,
                    concern: "invalid suffix".to_string(),
                    usage: VulkanRuntimeDeviceLocalTransientAllocationUsage::Storage,
                    allocation_class: VulkanRuntimeStreamAllocationClass::Permanent,
                },
            ],
            &[],
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
    device
        .resource_store
        .retained_representation_cache_payload_bytes = 70;
    device
        .resource_store
        .retained_representation_cache_allocation_padding_bytes = 10;
    assert_eq!(
        device.resource_store.maximum_extra_device_bytes().unwrap(),
        230
    );
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
    let stream_control_allocations = plan
        .resident_shared_host_allocations
        .iter()
        .filter(|allocation| {
            allocation.kind == VulkanRuntimeSharedHostResidentAllocationKind::StreamControl
        })
        .collect::<Vec<_>>();
    assert_eq!(stream_control_allocations.len(), 1);
    assert_eq!(stream_control_allocations[0].owner_device_id, "owner");
    assert_eq!(
        stream_control_allocations[0].participant_device_ids,
        ["owner".to_string(), "helper".to_string()]
    );
    assert_eq!(
        stream_control_allocations[0].byte_capacity,
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
fn aliased_logical_devices_use_one_shared_host_stream_control() {
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
        0
    );
    assert_eq!(
        plan.total_stream_device_local_bytes,
        baseline.total_stream_device_local_bytes - 2 * VULKAN_STREAM_CONTROL_BYTE_CAPACITY
    );
    assert_eq!(
        plan.total_stream_shared_host_bytes,
        baseline.total_stream_shared_host_bytes + VULKAN_STREAM_CONTROL_BYTE_CAPACITY
    );
    let stream_control = plan
        .resident_shared_host_allocations
        .iter()
        .find(|allocation| {
            allocation.kind == VulkanRuntimeSharedHostResidentAllocationKind::StreamControl
        })
        .unwrap();
    assert_eq!(stream_control.participant_device_ids.len(), 1);
    assert_eq!(
        stream_control.byte_capacity,
        VULKAN_STREAM_CONTROL_BYTE_CAPACITY
    );
}

#[test]
fn stream_control_binding_omits_shard_only_helpers() {
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
    let baseline = plan.clone();
    let physical_devices = BTreeMap::from([
        ("owner".to_string(), "physical-a".to_string()),
        ("helper".to_string(), "physical-b".to_string()),
    ]);

    plan.bind_stream_control_memory_domain(&physical_devices)
        .unwrap();

    assert_eq!(
        plan.total_stream_device_local_bytes,
        baseline.total_stream_device_local_bytes - VULKAN_STREAM_CONTROL_BYTE_CAPACITY,
    );
    let stream_control = exact_stream_control_shared_host_allocation(
        &plan,
        &physical_devices,
    )
    .unwrap();
    assert_eq!(stream_control.owner_device_id, "owner");
    assert_eq!(
        stream_control.participant_device_ids,
        ["owner".to_string()],
    );
    assert_eq!(
        plan.device_plans
            .iter()
            .find(|device| device.device_id == "helper")
            .unwrap()
            .breakdown
            .owner_stream_control_device_bytes_per_stream,
        0,
    );
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
fn exact_stream_control_allocation_rejects_ledger_drift() {
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
    let binding = BTreeMap::from([
        ("owner".to_string(), "physical-a".to_string()),
        ("helper".to_string(), "physical-b".to_string()),
    ]);
    plan.bind_stream_control_memory_domain(&binding).unwrap();

    exact_stream_control_shared_host_allocation(&plan, &binding).unwrap();

    let stream_control_index = plan
        .resident_shared_host_allocations
        .iter()
        .position(|allocation| {
            allocation.kind == VulkanRuntimeSharedHostResidentAllocationKind::StreamControl
        })
        .unwrap();
    let mut missing = plan.clone();
    missing
        .resident_shared_host_allocations
        .remove(stream_control_index);
    assert!(
        exact_stream_control_shared_host_allocation(&missing, &binding)
            .unwrap_err()
            .to_string()
            .contains("expected one")
    );

    let mut wrong_owner = plan.clone();
    wrong_owner.resident_shared_host_allocations[stream_control_index].owner_device_id =
        "helper".to_string();
    assert!(
        exact_stream_control_shared_host_allocation(&wrong_owner, &binding)
            .unwrap_err()
            .to_string()
            .contains("exact physical participants")
    );

    let mut reordered = plan.clone();
    reordered.resident_shared_host_allocations[stream_control_index]
        .participant_device_ids
        .reverse();
    assert!(
        exact_stream_control_shared_host_allocation(&reordered, &binding)
            .unwrap_err()
            .to_string()
            .contains("exact physical participants")
    );

    let mut wrong_capacity = plan.clone();
    wrong_capacity.resident_shared_host_allocations[stream_control_index].byte_capacity += 1;
    assert!(
        exact_stream_control_shared_host_allocation(&wrong_capacity, &binding)
            .unwrap_err()
            .to_string()
            .contains("exact physical participants")
    );

    let mut unbound = plan;
    unbound.stream_control_memory_domain_bound = false;
    assert!(
        exact_stream_control_shared_host_allocation(&unbound, &binding)
            .unwrap_err()
            .to_string()
            .contains("not bound")
    );
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
            participant_device_ids: Vec::new(),
            byte_capacity: 19,
            concern: "missing device".to_string(),
            usage: VulkanRuntimeDeviceLocalTransientAllocationUsage::Storage,
            allocation_class: VulkanRuntimeStreamAllocationClass::Permanent,
        },
        VulkanRuntimeDeviceLocalTransientAllocation {
            logical_device_id: "owner".to_string(),
            participant_device_ids: vec!["owner".to_string()],
            byte_capacity: 0,
            concern: "zero capacity".to_string(),
            usage: VulkanRuntimeDeviceLocalTransientAllocationUsage::Storage,
            allocation_class: VulkanRuntimeStreamAllocationClass::Permanent,
        },
        VulkanRuntimeDeviceLocalTransientAllocation {
            logical_device_id: "owner".to_string(),
            participant_device_ids: vec!["owner".to_string()],
            byte_capacity: 19,
            concern: "".to_string(),
            usage: VulkanRuntimeDeviceLocalTransientAllocationUsage::Storage,
            allocation_class: VulkanRuntimeStreamAllocationClass::Permanent,
        },
        VulkanRuntimeDeviceLocalTransientAllocation {
            logical_device_id: "owner".to_string(),
            participant_device_ids: vec!["helper".to_string()],
            byte_capacity: 19,
            concern: "external allocation missing its owner".to_string(),
            usage: VulkanRuntimeDeviceLocalTransientAllocationUsage::ExternalSharedStorage,
            allocation_class: VulkanRuntimeStreamAllocationClass::Permanent,
        },
    ] {
        let error = plan
            .add_execution_transient_reservation(&[malformed], &[], &[])
            .unwrap_err();
        assert!(error.to_string().contains("is malformed"));
        assert_eq!(plan, original);
    }

    for malformed in [
        VulkanRuntimeHostVisibleTransientAllocation {
            logical_device_id: "".to_string(),
            byte_capacity: 19,
            concern: "missing device".to_string(),
            allocation_class: VulkanRuntimeStreamAllocationClass::Permanent,
        },
        VulkanRuntimeHostVisibleTransientAllocation {
            logical_device_id: "owner".to_string(),
            byte_capacity: 0,
            concern: "zero capacity".to_string(),
            allocation_class: VulkanRuntimeStreamAllocationClass::Permanent,
        },
        VulkanRuntimeHostVisibleTransientAllocation {
            logical_device_id: "absent".to_string(),
            byte_capacity: 19,
            concern: "unknown device".to_string(),
            allocation_class: VulkanRuntimeStreamAllocationClass::Permanent,
        },
    ] {
        let error = plan
            .add_execution_transient_reservation(&[], &[malformed], &[])
            .unwrap_err();
        assert!(
            error.to_string().contains("is malformed")
                || error.to_string().contains("absent logical device")
        );
        assert_eq!(plan, original);
    }

    for malformed in [
        VulkanRuntimeSharedHostTransientAllocation {
            mode: VulkanRuntimeSharedHostTransientAllocationMode::ConditionalPredicate,
            owner_device_id: "owner".to_string(),
            participant_device_ids: vec!["helper".to_string()],
            byte_capacity: 19,
            concern: "missing owner".to_string(),
            allocation_class: VulkanRuntimeStreamAllocationClass::Permanent,
        },
        VulkanRuntimeSharedHostTransientAllocation {
            mode: VulkanRuntimeSharedHostTransientAllocationMode::Always,
            owner_device_id: "owner".to_string(),
            participant_device_ids: vec!["owner".to_string(), "owner".to_string()],
            byte_capacity: 19,
            concern: "duplicate participant".to_string(),
            allocation_class: VulkanRuntimeStreamAllocationClass::Permanent,
        },
        VulkanRuntimeSharedHostTransientAllocation {
            mode: VulkanRuntimeSharedHostTransientAllocationMode::Always,
            owner_device_id: "owner".to_string(),
            participant_device_ids: vec!["owner".to_string()],
            byte_capacity: 0,
            concern: "zero capacity".to_string(),
            allocation_class: VulkanRuntimeStreamAllocationClass::Permanent,
        },
        VulkanRuntimeSharedHostTransientAllocation {
            mode: VulkanRuntimeSharedHostTransientAllocationMode::Always,
            owner_device_id: "owner".to_string(),
            participant_device_ids: vec!["absent".to_string(), "owner".to_string()],
            byte_capacity: 19,
            concern: "unknown participant".to_string(),
            allocation_class: VulkanRuntimeStreamAllocationClass::Permanent,
        },
        VulkanRuntimeSharedHostTransientAllocation {
            mode: VulkanRuntimeSharedHostTransientAllocationMode::CrossDeviceTimelineStaging,
            owner_device_id: "owner".to_string(),
            participant_device_ids: vec!["owner".to_string()],
            byte_capacity: 19,
            concern: "incomplete boundary".to_string(),
            allocation_class: VulkanRuntimeStreamAllocationClass::Permanent,
        },
    ] {
        let error = plan
            .add_execution_transient_reservation(&[], &[], &[malformed])
            .unwrap_err();
        assert!(error.to_string().contains("is malformed"));
        assert_eq!(plan, original);
    }
}

#[test]
fn execution_transient_host_requirements_are_queried_and_aligned_per_allocation() {
    let allocations = vec![
        VulkanRuntimeSharedHostTransientAllocation {
            mode: VulkanRuntimeSharedHostTransientAllocationMode::ConditionalPredicate,
            owner_device_id: "owner".to_string(),
            participant_device_ids: vec!["helper".to_string(), "owner".to_string()],
            byte_capacity: 17,
            concern: "signal".to_string(),
            allocation_class: VulkanRuntimeStreamAllocationClass::Permanent,
        },
        VulkanRuntimeSharedHostTransientAllocation {
            mode: VulkanRuntimeSharedHostTransientAllocationMode::CrossDeviceTimelineStaging,
            owner_device_id: "helper".to_string(),
            participant_device_ids: vec!["helper".to_string(), "owner".to_string()],
            byte_capacity: 29,
            concern: "edge".to_string(),
            allocation_class: VulkanRuntimeStreamAllocationClass::PromptRunner,
        },
    ];
    let mut queried = Vec::new();

    let exact = execution_transient_shared_host_requirement_bytes_with(
        &allocations,
        |allocation| {
            queried.push((allocation.concern.clone(), allocation.mode));
            Ok(allocation.byte_capacity.next_multiple_of(64))
        },
    )
    .unwrap();

    assert_eq!(
        queried,
        vec![
            (
                "signal".to_string(),
                VulkanRuntimeSharedHostTransientAllocationMode::ConditionalPredicate,
            ),
            (
                "edge".to_string(),
                VulkanRuntimeSharedHostTransientAllocationMode::CrossDeviceTimelineStaging,
            ),
        ],
    );
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
            participant_device_ids: vec!["owner".to_string()],
            byte_capacity: 17,
            concern: "signal".to_string(),
            usage: VulkanRuntimeDeviceLocalTransientAllocationUsage::Storage,
            allocation_class: VulkanRuntimeStreamAllocationClass::Permanent,
        },
        VulkanRuntimeDeviceLocalTransientAllocation {
            logical_device_id: "owner".to_string(),
            participant_device_ids: vec!["owner".to_string()],
            byte_capacity: 29,
            concern: "control".to_string(),
            usage: VulkanRuntimeDeviceLocalTransientAllocationUsage::ConditionalPredicate,
            allocation_class: VulkanRuntimeStreamAllocationClass::PromptRunner,
        },
        VulkanRuntimeDeviceLocalTransientAllocation {
            logical_device_id: "owner".to_string(),
            participant_device_ids: vec!["helper".to_string(), "owner".to_string()],
            byte_capacity: 33,
            concern: "external signal".to_string(),
            usage: VulkanRuntimeDeviceLocalTransientAllocationUsage::ExternalSharedStorage,
            allocation_class: VulkanRuntimeStreamAllocationClass::PromptRunner,
        },
    ];
    let mut queried = Vec::new();

    let exact = execution_transient_device_requirement_bytes_with(&allocations, |allocation| {
        queried.push((
            allocation.concern.clone(),
            allocation.usage,
            allocation.participant_device_ids.clone(),
        ));
        Ok(allocation.byte_capacity.next_multiple_of(64))
    })
    .unwrap();

    assert_eq!(
        queried,
        vec![
            (
                "signal".to_string(),
                VulkanRuntimeDeviceLocalTransientAllocationUsage::Storage,
                vec!["owner".to_string()],
            ),
            (
                "control".to_string(),
                VulkanRuntimeDeviceLocalTransientAllocationUsage::ConditionalPredicate,
                vec!["owner".to_string()],
            ),
            (
                "external signal".to_string(),
                VulkanRuntimeDeviceLocalTransientAllocationUsage::ExternalSharedStorage,
                vec!["helper".to_string(), "owner".to_string()],
            ),
        ],
    );
    assert_eq!(exact, 192);
    assert_ne!(exact, (17usize + 29 + 33).next_multiple_of(64));

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
fn stream_memory_admission_classifies_each_remountable_runner_independently() {
    let target_transaction = VulkanRuntimeResidentStreamAllocation {
        scope: VulkanRuntimeResidentStreamAllocationScope::Target,
        kind: VulkanRuntimeResidentStreamAllocationKind::StateTransaction {
            component_id: "layer_00".to_string(),
            state_id: "memory".to_string(),
        },
        byte_capacity: 64,
    };
    let target_snapshot = VulkanRuntimeResidentStreamAllocation {
        kind: VulkanRuntimeResidentStreamAllocationKind::CausalVerificationSnapshot {
            component_id: "layer_00".to_string(),
            state_id: "memory".to_string(),
        },
        ..target_transaction.clone()
    };
    let target_checkpoint = VulkanRuntimeResidentStreamAllocation {
        kind: VulkanRuntimeResidentStreamAllocationKind::TransactionCheckpoint { slot: 0 },
        ..target_transaction.clone()
    };
    let target_state = VulkanRuntimeResidentStreamAllocation {
        kind: VulkanRuntimeResidentStreamAllocationKind::State {
            component_id: "layer_00".to_string(),
            state_id: "memory".to_string(),
        },
        ..target_transaction.clone()
    };
    let draft_transaction = VulkanRuntimeResidentStreamAllocation {
        scope: VulkanRuntimeResidentStreamAllocationScope::SpeculativeDecoder {
            decoder_id: "draft".to_string(),
        },
        ..target_transaction.clone()
    };

    assert_eq!(
        resident_stream_allocation_class(&target_transaction),
        VulkanMemoryAdmissionAllocationClass::VerificationRunner,
    );
    assert_eq!(
        resident_stream_allocation_class(&target_snapshot),
        VulkanMemoryAdmissionAllocationClass::VerificationRunner,
    );
    assert_eq!(
        resident_stream_allocation_class(&target_checkpoint),
        VulkanMemoryAdmissionAllocationClass::TransactionCheckpoint,
    );
    assert_eq!(
        resident_stream_allocation_class(&target_state),
        VulkanMemoryAdmissionAllocationClass::Permanent,
    );
    assert_eq!(
        resident_stream_allocation_class(&draft_transaction),
        VulkanMemoryAdmissionAllocationClass::Permanent,
    );
    assert_eq!(
        stream_transient_allocation_class(VulkanRuntimeStreamAllocationClass::Permanent),
        VulkanMemoryAdmissionAllocationClass::Permanent,
    );
    assert_eq!(
        stream_transient_allocation_class(VulkanRuntimeStreamAllocationClass::PromptRunner),
        VulkanMemoryAdmissionAllocationClass::PromptRunner,
    );
    assert_eq!(
        stream_transient_allocation_class(VulkanRuntimeStreamAllocationClass::VerificationRunner),
        VulkanMemoryAdmissionAllocationClass::VerificationRunner,
    );
    assert_eq!(
        stream_transient_allocation_class(VulkanRuntimeStreamAllocationClass::CatchUpRunner),
        VulkanMemoryAdmissionAllocationClass::CatchUpRunner,
    );
}

#[test]
fn selected_resource_store_reserves_exact_physical_stream_bytes_before_cache_capacity() {
    let exact = BTreeMap::from([("gpu0".to_string(), 2_443_334_416usize)]);

    assert_eq!(
        physical_execution_store_pending_fixed_bytes(&exact, "gpu0", 4096).unwrap(),
        2_443_338_512,
    );
    assert!(
        physical_execution_store_pending_fixed_bytes(&exact, "gpu1", 4096)
            .unwrap_err()
            .to_string()
            .contains("no exact stream requirement"),
    );
    assert!(
        physical_execution_store_pending_fixed_bytes(
            &BTreeMap::from([("gpu0".to_string(), usize::MAX)]),
            "gpu0",
            1,
        )
        .unwrap_err()
        .to_string()
        .contains("overflowed"),
    );
}

#[test]
fn permanent_host_requirements_accumulate_across_memory_domains() {
    let sampler_history_physical_bytes = 2_101_248;
    let other_shared_host_physical_bytes = 2_097_024;
    let mut requirements = BTreeMap::from([(
        VulkanMemoryAdmissionAllocationClass::Permanent,
        sampler_history_physical_bytes,
    )]);

    add_classified_host_requirement(
        &mut requirements,
        VulkanMemoryAdmissionAllocationClass::Permanent,
        other_shared_host_physical_bytes,
        "base shared-host",
    )
    .unwrap();

    assert_eq!(
        requirements[&VulkanMemoryAdmissionAllocationClass::Permanent],
        sampler_history_physical_bytes + other_shared_host_physical_bytes,
        "a later permanent host class must add to the exact host-visible requirement instead of replacing it",
    );

    let before_overflow = requirements.clone();
    let error = add_classified_host_requirement(
        &mut requirements,
        VulkanMemoryAdmissionAllocationClass::Permanent,
        usize::MAX,
        "overflow fixture",
    )
    .unwrap_err();
    assert!(error.to_string().contains("host requirement overflowed"));
    assert_eq!(requirements, before_overflow);
}

#[test]
fn stream_memory_admission_uses_bound_shared_host_ownership() {
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
        ]),
    )
    .unwrap();

    assert_eq!(plan.total_stream_shared_host_bytes, 8_192);
    assert_eq!(plan.resident_shared_host_allocations.len(), 1);
    assert_eq!(
        physical_execution_unclassified_shared_host_logical_bytes(&plan).unwrap(),
        0,
        "a deferred graph edge belongs only to the bound route ledger",
    );
}

#[test]
fn stream_memory_admission_does_not_reserve_deferred_edges_twice() {
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
    let device_for = |_device_id: &str| -> Result<
        &VulkanComputeDevice,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        panic!("deferred graph edges must be resolved through the bound route ledger")
    };

    assert_eq!(
        distributed_shared_host_requirement_bytes(&activations, &device_for).unwrap(),
        0,
    );
}

#[test]
fn graph_edge_binding_preserves_execution_transient_shared_host_residency() {
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
    plan.add_execution_transient_reservation(
        &[],
        &[],
        &[VulkanRuntimeSharedHostTransientAllocation {
            mode: VulkanRuntimeSharedHostTransientAllocationMode::Always,
            owner_device_id: "owner".to_string(),
            participant_device_ids: vec!["helper".to_string(), "owner".to_string()],
            byte_capacity: 19,
            concern: "cross-device timeline".to_string(),
            allocation_class: VulkanRuntimeStreamAllocationClass::Permanent,
        }],
    )
    .unwrap();
    assert_eq!(plan.total_stream_shared_host_bytes, 19);

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
        ]),
    )
    .unwrap();

    assert_eq!(
        plan.total_stream_shared_host_bytes,
        8_192 + 19,
        "binding an exact graph-edge route must not erase ownerless execution staging",
    );
    assert_eq!(
        physical_execution_unclassified_shared_host_logical_bytes(&plan).unwrap(),
        0,
    );
}

#[test]
fn stream_memory_admission_rejects_inconsistent_shared_host_ledgers() {
    let base = physical_execution_residency_base_plan(1_000, 100);
    let mut plan = VulkanRuntimePhysicalExecutionResidencyPlan::plan(
        &base,
        &["owner".to_string()],
        &empty_physical_execution_parameter_allocations(),
        &empty_physical_execution_parameter_exclusions(),
        &empty_physical_execution_activation_plan(),
    )
    .unwrap();
    plan.execution_transient_shared_host_bytes_per_stream = 1;

    let error = physical_execution_unclassified_shared_host_logical_bytes(&plan).unwrap_err();
    assert!(error.to_string().contains("bound ledgers require 1"));
    assert!(error.to_string().contains("execution_transient=1"));
}

#[test]
fn resident_stream_device_requirements_are_queried_and_aligned_per_allocation() {
    let allocations = vec![
        VulkanRuntimeResidentStreamAllocation {
            scope: VulkanRuntimeResidentStreamAllocationScope::Target,
            kind: VulkanRuntimeResidentStreamAllocationKind::State {
                component_id: "component".to_string(),
                state_id: "state".to_string(),
            },
            byte_capacity: 17,
        },
        VulkanRuntimeResidentStreamAllocation {
            scope: VulkanRuntimeResidentStreamAllocationScope::Target,
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
            kind: VulkanRuntimeExternalDeviceLocalResidentAllocationKind::EdgeProducedPort {
                component_id: "first".to_string(),
                port_id: "output".to_string(),
                edge_indices: vec![1],
            },
            owner_device_id: "owner".to_string(),
            participant_device_ids: vec!["helper".to_string(), "owner".to_string()],
            byte_capacity: 17,
        },
        VulkanRuntimeExternalDeviceLocalResidentAllocation {
            kind: VulkanRuntimeExternalDeviceLocalResidentAllocationKind::EdgeProducedPort {
                component_id: "second".to_string(),
                port_id: "output".to_string(),
                edge_indices: vec![2],
            },
            owner_device_id: "helper".to_string(),
            participant_device_ids: vec!["helper".to_string(), "owner".to_string()],
            byte_capacity: 29,
        },
    ];
    let mut queried = Vec::new();

    let exact = external_device_local_resident_requirement_bytes_with(
        &allocations,
        |allocation| {
            queried.push(allocation.kind.clone());
            Ok(allocation.byte_capacity.next_multiple_of(64))
        },
    )
    .unwrap();

    assert_eq!(queried, allocations.iter().map(|allocation| allocation.kind.clone()).collect::<Vec<_>>());
    assert_eq!(exact, 128);
    assert_ne!(exact, (17usize + 29).next_multiple_of(64));

    let overflow = external_device_local_resident_requirement_bytes_with(
        &allocations,
        |allocation| {
            Ok(if allocation.byte_capacity == 17 {
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
            kind: VulkanRuntimeSharedHostResidentAllocationKind::EdgeStaging {
                component_id: "first".to_string(),
                port_id: "output".to_string(),
                edge_indices: vec![1],
            },
            owner_device_id: "owner".to_string(),
            participant_device_ids: vec!["helper".to_string(), "owner".to_string()],
            byte_capacity: 17,
        },
        VulkanRuntimeSharedHostResidentAllocation {
            kind: VulkanRuntimeSharedHostResidentAllocationKind::EdgeStaging {
                component_id: "second".to_string(),
                port_id: "output".to_string(),
                edge_indices: vec![2],
            },
            owner_device_id: "helper".to_string(),
            participant_device_ids: vec!["helper".to_string(), "owner".to_string()],
            byte_capacity: 29,
        },
    ];
    let mut queried = Vec::new();

    let exact = resident_shared_host_requirement_bytes_with(&allocations, |allocation| {
        queried.push(allocation.kind.clone());
        Ok(allocation.byte_capacity.next_multiple_of(64))
    })
    .unwrap();

    assert_eq!(queried, allocations.iter().map(|allocation| allocation.kind.clone()).collect::<Vec<_>>());
    assert_eq!(exact, 128);
    assert_ne!(exact, (17usize + 29).next_multiple_of(64));

    let overflow = resident_shared_host_requirement_bytes_with(&allocations, |allocation| {
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
fn distributed_shared_host_requirement_exactly_backs_runtime_activation_allocation() {
    let Some((owner, helper)) = selected_test_vulkan_device_pair() else {
        eprintln!("skipping exact shared-host admission test without two explicit Vulkan devices");
        return;
    };
    let byte_capacity = 32_768;
    let requirement = owner
        .shared_host_allocation_requirement_bytes(&[helper.as_ref()], byte_capacity)
        .unwrap();
    let plan = VulkanDistributedActivationBufferPlan {
        allocations: vec![VulkanDistributedActivationBufferAllocation {
            storage: VulkanDistributedActivationStorage::ActivationSlot,
            owner_device_id: "owner".to_string(),
            component_id: "component".to_string(),
            slot: 7,
            byte_capacity,
            signal_ids: vec!["selection".to_string()],
            device_ids: vec!["helper".to_string(), "owner".to_string()],
            input_use_count: 1,
            output_use_count: 1,
        }],
        reduction_allocations: Vec::new(),
        private_intermediate_allocations: Vec::new(),
        allocation_count: 1,
        import_count: 1,
        reference_count: 2,
        total_shared_byte_capacity: byte_capacity,
        total_private_byte_capacity: 0,
        route: VulkanSharedResidentBufferRoute::SharedHost,
    };
    let admission = VulkanMemoryAdmission::reserve(
        &[],
        Some((
            owner.as_ref(),
            vulkan_safe_host_available_bytes().unwrap(),
            requirement,
        )),
    )
    .unwrap();
    let _scope = admission.enter();

    let buffers = VulkanDistributedActivationBuffers::allocate(&plan, |device_id| match device_id {
        "owner" => Ok(owner.as_ref()),
        "helper" => Ok(helper.as_ref()),
        other => Err(format!("unexpected fixture device {other}")),
    })
    .unwrap();

    admission
        .ensure_fully_consumed("distributed activation fixture")
        .unwrap();
    assert_eq!(buffers.total_shared_byte_capacity, byte_capacity);
}

#[test]
fn feedback_control_mount_consumes_its_exact_physical_allocation_ledger() {
    let Some((owner, helper)) = selected_test_vulkan_device_pair() else {
        eprintln!("skipping exact feedback-control mount test without two explicit Vulkan devices");
        return;
    };
    let vocabulary_size = 129_280;
    let dispatch_capacity = 256;
    let byte_capacity =
        resident_feedback_control_byte_capacity(vocabulary_size, dispatch_capacity).unwrap();
    let requirement = owner
        .shared_host_allocation_requirement_bytes(&[helper.as_ref()], byte_capacity)
        .unwrap();
    let planned = VulkanRuntimeSharedHostResidentAllocation {
        kind: VulkanRuntimeSharedHostResidentAllocationKind::FeedbackControl {
            scope_id: "target".to_string(),
        },
        owner_device_id: "owner".to_string(),
        participant_device_ids: vec!["owner".to_string(), "helper".to_string()],
        byte_capacity,
    };
    let device_for = |device_id: &str| match device_id {
        "owner" => Ok::<_, String>(owner.as_ref()),
        "helper" => Ok(helper.as_ref()),
        other => Err(format!("unexpected fixture device {other}")),
    };
    let mut wrong_capacity = planned.clone();
    wrong_capacity.byte_capacity += 1;
    assert!(
        VulkanResidentFeedbackControlPlane::new(
            &["owner".to_string(), "helper".to_string()],
            "helper",
            vocabulary_size,
            dispatch_capacity,
            &wrong_capacity,
            &device_for,
        )
        .err()
        .unwrap()
        .to_string()
        .contains("disagrees with its physical allocation ledger")
    );
    let admission = VulkanMemoryAdmission::reserve(
        &[],
        Some((
            owner.as_ref(),
            vulkan_safe_host_available_bytes().unwrap(),
            requirement,
        )),
    )
    .unwrap();
    let _scope = admission.enter();

    let control = VulkanResidentFeedbackControlPlane::new(
        &["owner".to_string(), "helper".to_string()],
        "helper",
        vocabulary_size,
        dispatch_capacity,
        &planned,
        &device_for,
    )
    .unwrap();

    admission
        .ensure_fully_consumed("feedback-control fixture")
        .unwrap();
    assert!(control.buffers["owner"].shares_host_allocation_with(&control.buffers["helper"]));
    assert_eq!(control.host_buffer_device_id, "helper");
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
                scope: VulkanRuntimeResidentStreamAllocationScope::Target,
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
fn graph_edge_binding_excludes_distributed_only_participants_from_graph_staging() {
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
    assert_eq!(shard.stream_device_local_bytes, 0);
    assert!(!shard.resident_stream_device_allocations.iter().any(|allocation| {
        matches!(
            &allocation.kind,
            VulkanRuntimeResidentStreamAllocationKind::EdgeStagingReplica { .. }
        )
    }));
    assert_eq!(plan.total_stream_device_local_bytes, original_total - 8_192);
    assert_eq!(plan.total_stream_shared_host_bytes, 2 * 8_192);
    assert_eq!(plan.resident_shared_host_allocations.len(), 2);
    assert_eq!(
        plan.resident_shared_host_allocations
            .iter()
            .find(|allocation| matches!(
                allocation.kind,
                VulkanRuntimeSharedHostResidentAllocationKind::EdgeStaging { .. }
            ))
            .unwrap()
            .participant_device_ids
            .len(),
        2
    );
    assert!(plan.resident_shared_host_allocations.iter().any(|allocation| {
        matches!(
            allocation.kind,
            VulkanRuntimeSharedHostResidentAllocationKind::DistributedProducedPort { .. }
        ) && allocation.participant_device_ids
            == ["helper".to_string(), "owner".to_string(), "shard".to_string()]
    }));
}

#[test]
fn graph_edge_binding_keeps_external_distributed_and_graph_participants_in_one_domain() {
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
        route: VulkanSharedResidentBufferRoute::ExternalDeviceLocal,
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
            physical_execution_edge_route(VulkanPlacedEdgeTransferRoute::ExternalDeviceLocal),
        )]),
        &BTreeMap::from([
            ("owner".to_string(), "physical-a".to_string()),
            ("helper".to_string(), "physical-b".to_string()),
            ("shard".to_string(), "physical-c".to_string()),
        ]),
    )
    .unwrap();

    let owner = plan
        .device_plans
        .iter()
        .find(|device| device.device_id == "owner")
        .unwrap();
    let [allocation] = owner.external_device_local_resident_allocations.as_slice() else {
        panic!("external distributed produced port must use one physical allocation");
    };
    assert_eq!(
        allocation.participant_device_ids,
        ["helper".to_string(), "owner".to_string(), "shard".to_string()]
    );
    assert_eq!(allocation.byte_capacity, 8_192);
    assert_eq!(plan.total_stream_device_local_bytes, original_total - 8_192);
    assert_eq!(plan.total_stream_shared_host_bytes, 0);
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

#[test]
fn graph_edge_binding_never_consumes_same_named_speculative_allocations() {
    let (mut base, edge_plans) = physical_execution_edge_base_plan();
    let owner = base
        .device_plans
        .iter_mut()
        .find(|device| device.device_id == "owner")
        .unwrap();
    let mut speculative = owner
        .resident_stream_device_allocations
        .iter()
        .find(|allocation| {
            matches!(
                allocation.kind,
                VulkanRuntimeResidentStreamAllocationKind::EdgeProducedPort { .. }
            )
        })
        .unwrap()
        .clone();
    speculative.scope = VulkanRuntimeResidentStreamAllocationScope::SpeculativeDecoder {
        decoder_id: "draft".to_string(),
    };
    owner.breakdown.speculative_decoder_activation_bytes += speculative.byte_capacity;
    owner.working_set.activation_headroom_bytes += speculative.byte_capacity;
    owner.initial_device_resident_bytes += speculative.byte_capacity;
    base.total_initial_device_resident_bytes += speculative.byte_capacity;
    owner
        .resident_stream_device_allocations
        .push(speculative.clone());
    let mut plan = VulkanRuntimePhysicalExecutionResidencyPlan::plan(
        &base,
        &["owner".to_string(), "helper".to_string()],
        &empty_physical_execution_parameter_allocations(),
        &empty_physical_execution_parameter_exclusions(),
        &empty_physical_execution_activation_plan(),
    )
    .unwrap();

    plan.bind_graph_edge_memory_domains(
        &edge_plans,
        &empty_physical_execution_activation_plan(),
        &BTreeMap::from([(
            5,
            physical_execution_edge_route(VulkanPlacedEdgeTransferRoute::DeviceLocalStaging),
        )]),
        &BTreeMap::from([
            ("owner".to_string(), "physical-owner".to_string()),
            ("helper".to_string(), "physical-helper".to_string()),
        ]),
    )
    .unwrap();

    assert!(plan.device_plans.iter().any(|device| {
        device
            .resident_stream_device_allocations
            .iter()
            .any(|allocation| allocation == &speculative)
    }));
}

#[test]
fn feedback_control_binding_uses_one_shared_host_allocation_when_colocated() {
    let (mut base, _) = physical_execution_edge_base_plan();
    add_feedback_control_residency(&mut base, 12_345);
    let mut plan = VulkanRuntimePhysicalExecutionResidencyPlan::plan(
        &base,
        &["owner".to_string(), "helper".to_string()],
        &empty_physical_execution_parameter_allocations(),
        &empty_physical_execution_parameter_exclusions(),
        &empty_physical_execution_activation_plan(),
    )
    .unwrap();
    let original_total = plan.total_stream_device_local_bytes;

    plan.bind_feedback_control_memory_domain(&BTreeMap::from([
        ("owner".to_string(), "physical-a".to_string()),
        ("helper".to_string(), "physical-a".to_string()),
    ]))
    .unwrap();

    assert!(plan.feedback_control_memory_domain_bound);
    assert_eq!(
        plan.total_stream_device_local_bytes,
        original_total - 12_345
    );
    assert_eq!(plan.total_stream_shared_host_bytes, 12_345);
    let [allocation] = plan.resident_shared_host_allocations.as_slice() else {
        panic!("colocated feedback control must have one host allocation");
    };
    assert!(matches!(
        allocation.kind,
        VulkanRuntimeSharedHostResidentAllocationKind::FeedbackControl { .. }
    ));
    assert_eq!(allocation.owner_device_id, "owner");
    assert_eq!(allocation.participant_device_ids, ["owner".to_string()]);
}

#[test]
fn host_visible_sampler_buffers_move_to_typed_host_allocations() {
    let mut base = physical_execution_residency_base_plan(1_000, 100);
    let owner = &mut base.device_plans[0];
    for (buffer_id, byte_capacity) in [
        ("history_and_output", 144),
        ("scratch", 1_024),
        ("random_seed", 4),
        ("seen_token_batch", 256),
    ] {
        owner
            .resident_stream_device_allocations
            .push(VulkanRuntimeResidentStreamAllocation {
                scope: VulkanRuntimeResidentStreamAllocationScope::Target,
                kind: VulkanRuntimeResidentStreamAllocationKind::RuntimeBuffer {
                    class: VulkanRuntimeResidentBufferClass::SamplerWorkspace,
                    scope_id: "sampler".to_string(),
                    buffer_id: buffer_id.to_string(),
                },
                byte_capacity,
            });
        owner.breakdown.sampler_workspace_bytes += byte_capacity;
        owner.working_set.activation_headroom_bytes += byte_capacity;
        owner.initial_device_resident_bytes += byte_capacity;
        base.total_initial_device_resident_bytes += byte_capacity;
    }
    let mut plan = VulkanRuntimePhysicalExecutionResidencyPlan::plan(
        &base,
        &["owner".to_string()],
        &empty_physical_execution_parameter_allocations(),
        &empty_physical_execution_parameter_exclusions(),
        &empty_physical_execution_activation_plan(),
    )
    .unwrap();
    let original_device_bytes = plan.total_stream_device_local_bytes;

    plan.bind_host_visible_runtime_buffer_memory_domains()
        .unwrap();

    let host_bytes = 144 + 4 + 256;
    assert!(plan.host_visible_runtime_buffer_memory_domains_bound);
    assert_eq!(
        plan.total_stream_device_local_bytes,
        original_device_bytes - host_bytes
    );
    assert_eq!(plan.total_stream_shared_host_bytes, host_bytes);
    assert_eq!(plan.resident_shared_host_allocations.len(), 3);
    assert!(plan.resident_shared_host_allocations.iter().all(|allocation| {
        matches!(
            allocation.kind,
            VulkanRuntimeSharedHostResidentAllocationKind::HostVisibleRuntimeBuffer { .. }
        ) && allocation.owner_device_id == "owner"
            && allocation.participant_device_ids == ["owner".to_string()]
    }));
    let owner = &plan.device_plans[0];
    assert_eq!(
        owner
            .resident_stream_device_allocations
            .iter()
            .filter_map(|allocation| match &allocation.kind {
                VulkanRuntimeResidentStreamAllocationKind::RuntimeBuffer {
                    class: VulkanRuntimeResidentBufferClass::SamplerWorkspace,
                    buffer_id,
                    ..
                } => Some((buffer_id.as_str(), allocation.byte_capacity)),
                _ => None,
            })
            .collect::<Vec<_>>(),
        vec![("scratch", 1_024)]
    );
}

#[test]
fn speculative_decoder_host_visible_state_and_sampler_buffers_move_to_typed_host_allocations() {
    let mut base = physical_execution_residency_base_plan(1_000, 100);
    let owner = &mut base.device_plans[0];
    owner
        .resident_stream_device_allocations
        .push(VulkanRuntimeResidentStreamAllocation {
            scope: VulkanRuntimeResidentStreamAllocationScope::Target,
            kind: VulkanRuntimeResidentStreamAllocationKind::RuntimeBuffer {
                class: VulkanRuntimeResidentBufferClass::SamplerWorkspace,
                scope_id: "sampler".to_string(),
                buffer_id: "history_and_output".to_string(),
            },
            byte_capacity: 144,
        });
    owner.breakdown.sampler_workspace_bytes += 144;
    owner.working_set.activation_headroom_bytes += 144;
    owner.initial_device_resident_bytes += 144;
    base.total_initial_device_resident_bytes += 144;
    owner
        .resident_stream_device_allocations
        .push(VulkanRuntimeResidentStreamAllocation {
            scope: VulkanRuntimeResidentStreamAllocationScope::SpeculativeDecoder {
                decoder_id: "draft".to_string(),
            },
            kind: VulkanRuntimeResidentStreamAllocationKind::RuntimeBuffer {
                class: VulkanRuntimeResidentBufferClass::SpeculativeDecoderState,
                scope_id: "draft".to_string(),
                buffer_id: "stream_control".to_string(),
            },
            byte_capacity: VULKAN_STREAM_CONTROL_BYTE_CAPACITY,
        });
    owner.breakdown.speculative_decoder_state_bytes += VULKAN_STREAM_CONTROL_BYTE_CAPACITY;
    owner.working_set.activation_headroom_bytes += VULKAN_STREAM_CONTROL_BYTE_CAPACITY;
    owner.initial_device_resident_bytes += VULKAN_STREAM_CONTROL_BYTE_CAPACITY;
    base.total_initial_device_resident_bytes += VULKAN_STREAM_CONTROL_BYTE_CAPACITY;
    for (buffer_id, byte_capacity) in [
        ("history_and_output", 288),
        ("scratch", 2_048),
        ("random_seed", 4),
        ("seen_token_batch", 512),
    ] {
        owner
            .resident_stream_device_allocations
            .push(VulkanRuntimeResidentStreamAllocation {
                scope: VulkanRuntimeResidentStreamAllocationScope::SpeculativeDecoder {
                    decoder_id: "draft".to_string(),
                },
                kind: VulkanRuntimeResidentStreamAllocationKind::RuntimeBuffer {
                    class: VulkanRuntimeResidentBufferClass::SpeculativeDecoderWorkspace,
                    scope_id: "draft-sampler".to_string(),
                    buffer_id: buffer_id.to_string(),
                },
                byte_capacity,
            });
        owner.breakdown.speculative_decoder_workspace_bytes += byte_capacity;
        owner.working_set.activation_headroom_bytes += byte_capacity;
        owner.initial_device_resident_bytes += byte_capacity;
        base.total_initial_device_resident_bytes += byte_capacity;
    }
    let mut plan = VulkanRuntimePhysicalExecutionResidencyPlan::plan(
        &base,
        &["owner".to_string()],
        &empty_physical_execution_parameter_allocations(),
        &empty_physical_execution_parameter_exclusions(),
        &empty_physical_execution_activation_plan(),
    )
    .unwrap();
    let original_device_bytes = plan.total_stream_device_local_bytes;

    plan.bind_host_visible_runtime_buffer_memory_domains()
        .unwrap();

    assert_eq!(
        plan.total_stream_device_local_bytes,
        original_device_bytes - 144 - VULKAN_STREAM_CONTROL_BYTE_CAPACITY - 288 - 4 - 512
    );
    assert!(plan.resident_shared_host_allocations.iter().any(|allocation| {
        matches!(
            &allocation.kind,
            VulkanRuntimeSharedHostResidentAllocationKind::HostVisibleRuntimeBuffer {
                scope: VulkanRuntimeResidentStreamAllocationScope::SpeculativeDecoder {
                    decoder_id
                },
                class: VulkanRuntimeResidentBufferClass::SpeculativeDecoderState,
                scope_id,
                buffer_id,
            } if decoder_id == "draft" && scope_id == "draft" && buffer_id == "stream_control"
        ) && allocation.owner_device_id == "owner"
            && allocation.participant_device_ids == ["owner".to_string()]
            && allocation.byte_capacity == VULKAN_STREAM_CONTROL_BYTE_CAPACITY
    }));
    assert!(!plan.device_plans[0]
        .resident_stream_device_allocations
        .iter()
        .any(|allocation| matches!(
            &allocation.kind,
            VulkanRuntimeResidentStreamAllocationKind::RuntimeBuffer {
                class: VulkanRuntimeResidentBufferClass::SpeculativeDecoderState,
                buffer_id,
                ..
            } if buffer_id == "stream_control"
        )));
    for (buffer_id, byte_capacity) in [
        ("history_and_output", 288),
        ("random_seed", 4),
        ("seen_token_batch", 512),
    ] {
        assert!(plan.resident_shared_host_allocations.iter().any(|allocation| {
            matches!(
                &allocation.kind,
                VulkanRuntimeSharedHostResidentAllocationKind::HostVisibleRuntimeBuffer {
                    scope: VulkanRuntimeResidentStreamAllocationScope::SpeculativeDecoder {
                        decoder_id
                    },
                    class: VulkanRuntimeResidentBufferClass::SpeculativeDecoderWorkspace,
                    scope_id,
                    buffer_id: planned_buffer_id,
                } if decoder_id == "draft"
                    && scope_id == "draft-sampler"
                    && planned_buffer_id == buffer_id
            ) && allocation.owner_device_id == "owner"
                && allocation.participant_device_ids == ["owner".to_string()]
                && allocation.byte_capacity == byte_capacity
        }));
    }
    assert_eq!(
        plan.device_plans[0]
            .resident_stream_device_allocations
            .iter()
            .filter_map(|allocation| match &allocation.kind {
                VulkanRuntimeResidentStreamAllocationKind::RuntimeBuffer {
                    class: VulkanRuntimeResidentBufferClass::SpeculativeDecoderWorkspace,
                    buffer_id,
                    ..
                } => Some((buffer_id.as_str(), allocation.byte_capacity)),
                _ => None,
            })
            .collect::<Vec<_>>(),
        vec![("scratch", 2_048)]
    );
}

#[test]
fn feedback_control_resizing_updates_the_exact_owner_and_total_atomically() {
    let (mut base, _) = physical_execution_edge_base_plan();
    add_feedback_control_residency(&mut base, 12_345);
    let mut plan = VulkanRuntimePhysicalExecutionResidencyPlan::plan(
        &base,
        &["owner".to_string(), "helper".to_string()],
        &empty_physical_execution_parameter_allocations(),
        &empty_physical_execution_parameter_exclusions(),
        &empty_physical_execution_activation_plan(),
    )
    .unwrap();
    let previous_total = plan.total_stream_device_local_bytes;
    let previous_owner = plan
        .device_plans
        .iter()
        .find(|device| device.device_id == "owner")
        .unwrap()
        .stream_device_local_bytes;

    plan.resize_feedback_control_residency(4_321).unwrap();

    assert_eq!(plan.feedback_control_resident_byte_capacity().unwrap(), 4_321);
    assert_eq!(
        plan.total_stream_device_local_bytes,
        previous_total - 12_345 + 4_321
    );
    assert_eq!(
        plan.device_plans
            .iter()
            .find(|device| device.device_id == "owner")
            .unwrap()
            .stream_device_local_bytes,
        previous_owner - 12_345 + 4_321
    );

    let resized = plan.clone();
    let error = plan.resize_feedback_control_residency(0).unwrap_err();
    assert!(error.to_string().contains("positive capacity"));
    assert_eq!(plan, resized);
}

#[test]
fn feedback_control_binding_moves_exactly_one_allocation_to_shared_host() {
    let (mut base, _) = physical_execution_edge_base_plan();
    add_feedback_control_residency(&mut base, 12_345);
    let mut plan = VulkanRuntimePhysicalExecutionResidencyPlan::plan(
        &base,
        &["owner".to_string(), "helper".to_string()],
        &empty_physical_execution_parameter_allocations(),
        &empty_physical_execution_parameter_exclusions(),
        &empty_physical_execution_activation_plan(),
    )
    .unwrap();
    plan.add_execution_transient_reservation(
        &[],
        &[],
        &[VulkanRuntimeSharedHostTransientAllocation {
            mode: VulkanRuntimeSharedHostTransientAllocationMode::Always,
            owner_device_id: "owner".to_string(),
            participant_device_ids: vec!["helper".to_string(), "owner".to_string()],
            byte_capacity: 19,
            concern: "cross-device timeline".to_string(),
            allocation_class: VulkanRuntimeStreamAllocationClass::Permanent,
        }],
    )
    .unwrap();
    let original = plan.clone();
    let physical = BTreeMap::from([
        ("owner".to_string(), "physical-a".to_string()),
        ("helper".to_string(), "physical-b".to_string()),
    ]);

    plan.bind_feedback_control_memory_domain(&physical).unwrap();

    assert!(plan.feedback_control_memory_domain_bound);
    assert_eq!(
        plan.total_stream_device_local_bytes,
        original.total_stream_device_local_bytes - 12_345
    );
    assert_eq!(
        plan.total_stream_shared_host_bytes,
        original.total_stream_shared_host_bytes + 12_345,
        "feedback binding must preserve ownerless execution staging",
    );
    assert_eq!(plan.resident_shared_host_allocations.len(), 1);
    assert!(matches!(
        &plan.resident_shared_host_allocations[0].kind,
        VulkanRuntimeSharedHostResidentAllocationKind::FeedbackControl { scope_id }
            if scope_id == "package"
    ));
    assert_eq!(
        plan.resident_shared_host_allocations[0]
            .participant_device_ids
            .len(),
        2
    );
    assert!(plan.device_plans.iter().all(|device| {
        device
            .resident_stream_device_allocations
            .iter()
            .all(|allocation| {
                !matches!(
                    &allocation.kind,
                    VulkanRuntimeResidentStreamAllocationKind::RuntimeBuffer {
                        class: VulkanRuntimeResidentBufferClass::FeedbackWorkspace,
                        buffer_id,
                        ..
                    } if buffer_id == "control"
                )
            })
    }));

    let once_bound = plan.clone();
    let error = plan
        .bind_feedback_control_memory_domain(&physical)
        .unwrap_err();
    assert!(error.to_string().contains("already bound"));
    assert_eq!(plan, once_bound);
}

#[test]
fn feedback_control_binding_rejects_incomplete_topology_atomically() {
    let (mut base, _) = physical_execution_edge_base_plan();
    add_feedback_control_residency(&mut base, 12_345);
    let mut plan = VulkanRuntimePhysicalExecutionResidencyPlan::plan(
        &base,
        &["owner".to_string(), "helper".to_string()],
        &empty_physical_execution_parameter_allocations(),
        &empty_physical_execution_parameter_exclusions(),
        &empty_physical_execution_activation_plan(),
    )
    .unwrap();
    let original = plan.clone();

    let error = plan
        .bind_feedback_control_memory_domain(&BTreeMap::from([(
            "owner".to_string(),
            "physical-a".to_string(),
        )]))
        .unwrap_err();

    assert!(error.to_string().contains("incomplete"));
    assert_eq!(plan, original);
}

#[test]
fn physical_execution_residency_rejects_runtime_buffer_breakdown_mismatch() {
    let (mut base, _) = physical_execution_edge_base_plan();
    add_feedback_control_residency(&mut base, 12_345);
    base.device_plans
        .iter_mut()
        .find(|device| device.device_id == "owner")
        .unwrap()
        .breakdown
        .feedback_workspace_bytes += 1;

    let error = VulkanRuntimePhysicalExecutionResidencyPlan::plan(
        &base,
        &["owner".to_string(), "helper".to_string()],
        &empty_physical_execution_parameter_allocations(),
        &empty_physical_execution_parameter_exclusions(),
        &empty_physical_execution_activation_plan(),
    )
    .unwrap_err();

    assert!(error.to_string().contains("feedback workspace allocation bytes"));
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
            scope: VulkanRuntimeResidentStreamAllocationScope::Target,
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
        scope: VulkanRuntimeResidentStreamAllocationScope::Target,
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

fn add_feedback_control_residency(base: &mut VulkanRuntimeResidencyPlan, byte_capacity: usize) {
    let owner = base
        .device_plans
        .iter_mut()
        .find(|device| device.device_id == "owner")
        .unwrap();
    owner
        .resident_stream_device_allocations
        .push(VulkanRuntimeResidentStreamAllocation {
            scope: VulkanRuntimeResidentStreamAllocationScope::Target,
            kind: VulkanRuntimeResidentStreamAllocationKind::RuntimeBuffer {
                class: VulkanRuntimeResidentBufferClass::FeedbackWorkspace,
                scope_id: "package".to_string(),
                buffer_id: "control".to_string(),
            },
            byte_capacity,
        });
    owner.breakdown.feedback_workspace_bytes += byte_capacity;
    owner.working_set.activation_headroom_bytes += byte_capacity;
    owner.initial_device_resident_bytes += byte_capacity;
    base.total_initial_device_resident_bytes += byte_capacity;
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
                scope: VulkanRuntimeResidentStreamAllocationScope::Target,
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
