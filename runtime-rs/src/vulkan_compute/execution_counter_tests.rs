#[test]
fn demand_sequence_device_timings_are_partitioned_and_reset() {
    reset_vulkan_resident_execution_counters();

    record_vulkan_demand_sequence_device_duration(false, 11);
    record_vulkan_demand_sequence_device_duration(false, 17);
    record_vulkan_demand_sequence_device_duration(true, 23);
    record_vulkan_resident_component_sequence_device_duration(29);
    record_vulkan_resident_component_sequence_device_duration(31);

    let counters = vulkan_resident_execution_counters();
    assert_eq!(counters.demand_initial_sequence_count, 2);
    assert_eq!(counters.demand_initial_device_duration_ns, 28);
    assert_eq!(counters.demand_initial_max_device_duration_ns, 17);
    assert_eq!(counters.demand_resume_sequence_count, 1);
    assert_eq!(counters.demand_resume_device_duration_ns, 23);
    assert_eq!(counters.demand_resume_max_device_duration_ns, 23);
    assert_eq!(counters.resident_component_sequence_count, 2);
    assert_eq!(counters.resident_component_device_duration_ns, 60);
    assert_eq!(counters.resident_component_max_device_duration_ns, 31);

    reset_vulkan_resident_execution_counters();
    let reset = vulkan_resident_execution_counters();
    assert_eq!(reset.demand_initial_sequence_count, 0);
    assert_eq!(reset.demand_initial_device_duration_ns, 0);
    assert_eq!(reset.demand_initial_max_device_duration_ns, 0);
    assert_eq!(reset.demand_resume_sequence_count, 0);
    assert_eq!(reset.demand_resume_device_duration_ns, 0);
    assert_eq!(reset.demand_resume_max_device_duration_ns, 0);
    assert_eq!(reset.resident_component_sequence_count, 0);
    assert_eq!(reset.resident_component_device_duration_ns, 0);
    assert_eq!(reset.resident_component_max_device_duration_ns, 0);
}

#[test]
fn distributed_submission_counters_preserve_phase_strategy_and_shards() {
    reset_vulkan_resident_execution_counters();

    record_vulkan_resident_distributed_execution_submission(
        VulkanResidentDistributedExecutionPhase::Decode,
        VulkanResidentDistributedExecutionKind::TensorParallel,
        3,
    );
    record_vulkan_resident_distributed_execution_submission(
        VulkanResidentDistributedExecutionPhase::Decode,
        VulkanResidentDistributedExecutionKind::WholeExpertParallel,
        2,
    );
    record_vulkan_resident_distributed_execution_submission(
        VulkanResidentDistributedExecutionPhase::Prefill,
        VulkanResidentDistributedExecutionKind::IntraExpertTensorParallel,
        4,
    );
    record_vulkan_resident_distributed_execution_submission(
        VulkanResidentDistributedExecutionPhase::Prefill,
        VulkanResidentDistributedExecutionKind::Hybrid,
        5,
    );

    let counters = vulkan_resident_execution_counters();
    assert_eq!(counters.distributed.decode.island_submissions, 2);
    assert_eq!(counters.distributed.decode.shard_submissions, 5);
    assert_eq!(
        counters
            .distributed
            .decode
            .tensor_parallel_island_submissions,
        1
    );
    assert_eq!(
        counters
            .distributed
            .decode
            .whole_expert_parallel_island_submissions,
        1
    );
    assert_eq!(counters.distributed.prefill.island_submissions, 2);
    assert_eq!(counters.distributed.prefill.shard_submissions, 9);
    assert_eq!(
        counters
            .distributed
            .prefill
            .intra_expert_tensor_parallel_island_submissions,
        1
    );
    assert_eq!(
        counters.distributed.prefill.hybrid_island_submissions,
        1
    );

    reset_vulkan_resident_execution_counters();
    assert_eq!(
        vulkan_resident_execution_counters().distributed,
        VulkanResidentDistributedExecutionCounters::default()
    );
}

#[test]
fn execution_counter_accumulation_sums_totals_and_preserves_maxima() {
    let mut total = VulkanResidentExecutionCounters::default();
    total.saturating_accumulate(VulkanResidentExecutionCounters {
        demand_initial_sequence_count: 2,
        demand_initial_device_duration_ns: 11,
        demand_initial_max_device_duration_ns: 7,
        resident_component_sequence_count: 3,
        resident_component_device_duration_ns: 13,
        resident_component_max_device_duration_ns: 8,
        distributed: VulkanResidentDistributedExecutionCounters {
            decode: VulkanResidentDistributedExecutionPhaseCounters {
                island_submissions: 1,
                shard_submissions: 2,
                tensor_parallel_island_submissions: 1,
                ..VulkanResidentDistributedExecutionPhaseCounters::default()
            },
            ..VulkanResidentDistributedExecutionCounters::default()
        },
        ..VulkanResidentExecutionCounters::default()
    });
    total.saturating_accumulate(VulkanResidentExecutionCounters {
        demand_initial_sequence_count: 5,
        demand_initial_device_duration_ns: 17,
        demand_initial_max_device_duration_ns: 6,
        resident_component_sequence_count: 7,
        resident_component_device_duration_ns: 19,
        resident_component_max_device_duration_ns: 12,
        distributed: VulkanResidentDistributedExecutionCounters {
            decode: VulkanResidentDistributedExecutionPhaseCounters {
                island_submissions: 2,
                shard_submissions: 5,
                hybrid_island_submissions: 2,
                ..VulkanResidentDistributedExecutionPhaseCounters::default()
            },
            ..VulkanResidentDistributedExecutionCounters::default()
        },
        ..VulkanResidentExecutionCounters::default()
    });

    assert_eq!(total.demand_initial_sequence_count, 7);
    assert_eq!(total.demand_initial_device_duration_ns, 28);
    assert_eq!(total.demand_initial_max_device_duration_ns, 7);
    assert_eq!(total.resident_component_sequence_count, 10);
    assert_eq!(total.resident_component_device_duration_ns, 32);
    assert_eq!(total.resident_component_max_device_duration_ns, 12);
    assert_eq!(total.distributed.decode.island_submissions, 3);
    assert_eq!(total.distributed.decode.shard_submissions, 7);
    assert_eq!(
        total.distributed.decode.tensor_parallel_island_submissions,
        1
    );
    assert_eq!(total.distributed.decode.hybrid_island_submissions, 2);
}
