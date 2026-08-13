#[test]
fn deferred_graph_edge_allocation_does_not_touch_a_device_before_final_routing() {
    let allocation = VulkanDistributedActivationBufferAllocation {
        storage: VulkanDistributedActivationStorage::Edge {
            edge_index: 7,
            owner_device_id: "gpu0".to_string(),
        },
        owner_device_id: "gpu0".to_string(),
        component_id: "component".to_string(),
        slot: 0,
        byte_capacity: 32,
        signal_ids: vec!["hidden".to_string()],
        device_ids: vec!["gpu0".to_string(), "gpu1".to_string()],
        input_use_count: 1,
        output_use_count: 1,
    };
    let plan = VulkanDistributedActivationBufferPlan {
        allocations: vec![allocation.clone()],
        reduction_allocations: Vec::new(),
        private_intermediate_allocations: Vec::new(),
        allocation_count: 1,
        import_count: 2,
        reference_count: 2,
        total_shared_byte_capacity: 32,
        total_private_byte_capacity: 0,
        route: VulkanSharedResidentBufferRoute::ExternalDeviceLocal,
    };

    let mut buffers = VulkanDistributedActivationBuffers::allocate_deferring_graph_edges(
        &plan,
        |_device_id| -> Result<&VulkanComputeDevice, &'static str> {
            panic!("a deferred graph edge must not resolve or allocate its provisional route")
        },
    )
    .unwrap();

    assert_eq!(buffers.allocations.len(), 1);
    assert_eq!(buffers.allocations[0].planned, allocation);
    assert!(buffers.allocations[0].device_buffers.is_empty());
    assert_eq!(buffers.import_count, 0);
    assert_eq!(buffers.total_shared_byte_capacity, 32);
    let error = buffers
        .finalize_deferred_graph_edges()
        .expect_err("an unmaterialized edge must fail closed before execution");
    assert!(error.to_string().contains("finalized 0 buffers for 2"));
}

#[test]
fn final_graph_edge_devices_must_exactly_match_the_declared_participants() {
    let allocation = VulkanDistributedActivationBufferAllocation {
        storage: VulkanDistributedActivationStorage::Edge {
            edge_index: 7,
            owner_device_id: "gpu0".to_string(),
        },
        owner_device_id: "gpu0".to_string(),
        component_id: "component".to_string(),
        slot: 0,
        byte_capacity: 32,
        signal_ids: vec!["hidden".to_string()],
        device_ids: vec!["gpu0".to_string(), "gpu1".to_string()],
        input_use_count: 1,
        output_use_count: 1,
    };

    validate_final_distributed_activation_devices(&allocation, ["gpu0", "gpu1"]).unwrap();
    assert!(
        validate_final_distributed_activation_devices(&allocation, ["gpu0"]).is_err()
    );
    assert!(
        validate_final_distributed_activation_devices(&allocation, ["gpu0", "gpu2"])
            .is_err()
    );
}
