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
