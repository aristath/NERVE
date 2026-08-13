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
    assert_eq!(owner.stream_device_local_bytes, 196);
    assert_eq!(helper.stream_device_local_bytes, 32);
    assert_eq!(plan.total_mount_device_local_bytes, 800);
    assert_eq!(plan.total_stream_device_local_bytes, 228);
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

    assert_eq!(plan.total_stream_device_local_bytes, 132);
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

    plan.add_execution_transient_reservation(
        &BTreeMap::from([("owner".to_string(), 33), ("helper".to_string(), 17)]),
        19,
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
        plan.total_stream_device_local_bytes,
        baseline.total_stream_device_local_bytes + 50
    );
    assert_eq!(plan.execution_transient_shared_host_bytes_per_stream, 19);
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
    let error = plan
        .add_execution_transient_reservation(
            &BTreeMap::from([("owner".to_string(), 1), ("absent".to_string(), 1)]),
            1,
        )
        .unwrap_err();
    assert!(error.to_string().contains("absent logical device"));
    assert_eq!(plan, accepted);
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
    assert_eq!(owner.stream_device_local_bytes, 100);
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
                transient_state_bytes: 40,
                activation_headroom_bytes: 60,
            },
            breakdown: VulkanRuntimeDeviceResidencyBreakdown::default(),
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
