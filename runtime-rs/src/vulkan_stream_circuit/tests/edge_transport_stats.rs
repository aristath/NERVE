fn sampled_edge_transport_stats(
    publish_count: usize,
    sample_count: usize,
    sampled_duration_ns: u64,
    estimated_duration_ns: u64,
    maximum_duration_ns: u64,
) -> VulkanPlacedEdgeTransportEdgeStats {
    VulkanPlacedEdgeTransportEdgeStats {
        key: VulkanPlacedEdgePacketKey {
            edge_index: 7,
            from_device_id: "gpu0".to_string(),
            to_device_id: "gpu1".to_string(),
        },
        signal: "frame".to_string(),
        route: VulkanPlacedEdgeTransferRoute::DeviceLocalStaging,
        byte_capacity: 32_768,
        publish_count,
        receive_count: publish_count,
        transferred_byte_count: publish_count.saturating_mul(32_768),
        queue_signal_count: publish_count,
        queue_wait_count: publish_count,
        host_wait_count: 0,
        queue_overlap_eligible: true,
        overlap_submission_count: publish_count,
        device_duration_sample_count: sample_count,
        sampled_device_duration_ns: sampled_duration_ns,
        estimated_device_duration_ns: estimated_duration_ns,
        maximum_sampled_transfer_duration_ns: maximum_duration_ns,
    }
}

#[test]
fn edge_transport_accumulates_bounded_samples_and_estimates_independently() {
    let mut aggregate = sampled_edge_transport_stats(3, 2, 90, 270, 90);
    let next = sampled_edge_transport_stats(5, 2, 110, 550, 110);

    aggregate.accumulate(&next);

    assert_eq!(aggregate.publish_count, 8);
    assert_eq!(aggregate.device_duration_sample_count, 4);
    assert_eq!(aggregate.sampled_device_duration_ns, 200);
    assert_eq!(aggregate.estimated_device_duration_ns, 820);
    assert_eq!(aggregate.maximum_sampled_transfer_duration_ns, 110);
}

#[test]
fn edge_transport_tick_reset_clears_samples_without_changing_route_identity() {
    let mut edge = sampled_edge_transport_stats(3, 2, 90, 270, 90);

    edge.reset_tick_counts();

    assert_eq!(edge.key.edge_index, 7);
    assert_eq!(edge.route, VulkanPlacedEdgeTransferRoute::DeviceLocalStaging);
    assert_eq!(edge.byte_capacity, 32_768);
    assert_eq!(edge.publish_count, 0);
    assert_eq!(edge.device_duration_sample_count, 0);
    assert_eq!(edge.sampled_device_duration_ns, 0);
    assert_eq!(edge.estimated_device_duration_ns, 0);
    assert_eq!(edge.maximum_sampled_transfer_duration_ns, 0);
}
