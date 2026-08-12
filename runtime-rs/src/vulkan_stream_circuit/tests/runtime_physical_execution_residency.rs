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
