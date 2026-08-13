const GPU_RESIDENCY_GATE_SHADER: &str = "shaders/gpu_residency_gate.comp";
const GPU_RESIDENCY_GATE_DOWNSTREAM_SHADER: &str =
    "tests/fixtures/vulkan/gpu_residency_gate_downstream.comp";
const GPU_RESIDENCY_GUARDED_DOWNSTREAM_SHADER: &str =
    "tests/fixtures/vulkan/gpu_residency_guarded_downstream.comp";

#[test]
fn gpu_residency_gate_contract_rejects_unrepresentable_or_unbounded_work() {
    let valid = VulkanGpuResidencyGateConfig {
        maximum_selection_count: 2,
        selection_count_per_lane: 2,
        selection_lane_stride_words: 2,
        selection_index_shift: 0,
        selection_index_mask: 0xffff,
        address_mapping: VulkanGpuResidencyAddressMapping::GroupTable {
            resource_address_slots: vec![0, 1, 2, 3],
            resource_address_slot_offsets: vec![0, 2, 4],
        },
        owned_resource_indices: None,
    };
    assert!(valid.validate(8, 4, 4).is_ok());

    let mut invalid = valid.clone();
    invalid.maximum_selection_count = 0;
    assert!(invalid.validate(8, 4, 4).is_err());

    let mut padded_lanes = valid.clone();
    padded_lanes.maximum_selection_count = 4;
    padded_lanes.selection_count_per_lane = 2;
    padded_lanes.selection_lane_stride_words = 4;
    assert!(
        padded_lanes
            .validate(6 * size_of::<u32>(), 4, 4)
            .is_ok()
    );
    assert!(
        padded_lanes
            .validate(5 * size_of::<u32>(), 4, 4)
            .is_err()
    );

    let mut invalid = valid.clone();
    invalid.maximum_selection_count = 3;
    assert!(invalid.validate(12, 4, 4).is_err());

    let mut invalid = valid.clone();
    invalid.selection_lane_stride_words = 1;
    assert!(invalid.validate(8, 4, 4).is_err());

    let mut invalid = valid.clone();
    invalid.selection_index_mask = 0;
    assert!(invalid.validate(8, 4, 4).is_err());

    let mut owned_subset = valid.clone();
    owned_subset.owned_resource_indices = Some(BTreeSet::from([1]));
    assert!(owned_subset.validate(8, 4, 4).is_ok());
    owned_subset.owned_resource_indices = Some(BTreeSet::new());
    assert!(owned_subset.validate(8, 4, 4).is_err());
    owned_subset.owned_resource_indices = Some(BTreeSet::from([2]));
    assert!(owned_subset.validate(8, 4, 4).is_err());
    assert_eq!(
        vulkan_gpu_residency_ownership_words(
            65,
            Some(&BTreeSet::from([0, 31, 32, 64])),
        )
        .unwrap(),
        vec![0x8000_0001, 1, 1],
    );
    assert!(vulkan_gpu_residency_ownership_words(65, None)
        .unwrap()
        .is_empty());
    assert!(vulkan_gpu_residency_ownership_words(2, Some(&BTreeSet::new())).is_err());
    assert!(vulkan_gpu_residency_ownership_words(2, Some(&BTreeSet::from([2]))).is_err());

    let mut invalid = valid.clone();
    invalid.selection_index_mask = 0x1;
    invalid.address_mapping =
        VulkanGpuResidencyAddressMapping::GroupTable {
            resource_address_slots: vec![0, 1, 2],
            resource_address_slot_offsets: vec![0, 1, 2, 3],
        };
    assert!(invalid.validate(8, 4, 4).is_err());

    let mut invalid = valid.clone();
    if let VulkanGpuResidencyAddressMapping::GroupTable {
        resource_address_slots,
        ..
    } = &mut invalid.address_mapping
    {
        resource_address_slots[1] = 0;
    }
    assert!(invalid.validate(8, 4, 4).is_err());

    let mut invalid = valid.clone();
    if let VulkanGpuResidencyAddressMapping::GroupTable {
        resource_address_slots,
        ..
    } = &mut invalid.address_mapping
    {
        resource_address_slots[0] = 4;
    }
    assert!(invalid.validate(8, 4, 4).is_err());

    let partitioned = VulkanGpuResidencyGateConfig {
        address_mapping:
            VulkanGpuResidencyAddressMapping::Partitioned {
                member_slot_bases: vec![0, 4],
                resource_count: 4,
            },
        ..valid.clone()
    };
    assert!(partitioned.validate(8, 8, 4).is_ok());
    let overlapping = VulkanGpuResidencyGateConfig {
        address_mapping:
            VulkanGpuResidencyAddressMapping::Partitioned {
                member_slot_bases: vec![0, 3],
                resource_count: 4,
            },
        ..valid.clone()
    };
    assert!(overlapping.validate(8, 8, 4).is_err());

    assert!(valid.validate(8, 4, 1).is_err());
    assert!(valid.validate(4, 4, 4).is_err());
}

#[test]
fn gpu_residency_gate_residency_geometry_is_hardware_neutral_and_exact() {
    let mut config = VulkanGpuResidencyGateConfig {
        maximum_selection_count: 2,
        selection_count_per_lane: 2,
        selection_lane_stride_words: 2,
        selection_index_shift: 0,
        selection_index_mask: 0xffff,
        address_mapping: VulkanGpuResidencyAddressMapping::GroupTable {
            resource_address_slots: vec![0, 1, 2, 3],
            resource_address_slot_offsets: vec![0, 2, 4],
        },
        owned_resource_indices: None,
    };
    let unowned = config.private_device_bytes().unwrap();
    assert_eq!(unowned.configuration_bytes, 11 * size_of::<u32>());
    assert_eq!(unowned.resource_group_record_bytes, 4 * size_of::<u32>());
    assert_eq!(unowned.resource_address_slot_bytes, 4 * size_of::<u32>());
    assert_eq!(unowned.resolved_address_bytes, 43 * size_of::<u32>());
    assert_eq!(
        unowned.total_bytes,
        unowned.configuration_bytes
            + unowned.resource_group_record_bytes
            + unowned.resource_address_slot_bytes
            + unowned.resolved_address_bytes,
    );

    config.owned_resource_indices = Some(BTreeSet::from([1]));
    let owned = config.private_device_bytes().unwrap();
    assert_eq!(
        owned.configuration_bytes,
        unowned.configuration_bytes + size_of::<u32>(),
    );
    assert_eq!(owned.total_bytes, unowned.total_bytes + size_of::<u32>());

    let miss_queue = VulkanGpuResidencyMissQueue::device_bytes_for_capacity(4).unwrap();
    assert_eq!(miss_queue.capacity, 4);
    assert_eq!(miss_queue.byte_count, 12 * size_of::<u32>());
    assert!(VulkanGpuResidencyMissQueue::device_bytes_for_capacity(0).is_err());
}

#[test]
fn gpu_residency_gate_control_separates_local_and_transaction_restore() {
    let control =
        vulkan_gpu_residency_gate_push_constants(8, 6, 17, true, false, 23).unwrap();
    assert_eq!(u32::from_le_bytes(control[0..4].try_into().unwrap()), 6);
    assert_eq!(u32::from_le_bytes(control[4..8].try_into().unwrap()), 17);
    assert_eq!(u32::from_le_bytes(control[8..12].try_into().unwrap()), 1);
    assert_eq!(u32::from_le_bytes(control[12..16].try_into().unwrap()), 0);
    assert_eq!(u32::from_le_bytes(control[16..20].try_into().unwrap()), 23);
    assert!(vulkan_gpu_residency_gate_push_constants(8, 9, 17, true, true, 23).is_err());
}

#[test]
fn gpu_residency_gate_keeps_hits_on_device_and_publishes_only_real_misses() {
    let Some(device_index) = stable_resource_test_device_index() else {
        eprintln!(
            "skipping GPU residency gate test: explicit Vulkan device unset"
        );
        return;
    };
    let gate_shader =
        vulkan_gpu_residency_gate_spirv_words()
            .expect("embedded GPU residency gate shader must be valid");
    let downstream_shader = compile_gpu_residency_gate_shader(
        GPU_RESIDENCY_GATE_DOWNSTREAM_SHADER,
    )
    .expect("GPU residency gate test requires a GLSL compiler");

    let device =
        VulkanComputeDevice::new_for_physical_device_index(device_index)
            .unwrap();
    let mut transfer = device
        .create_resident_transfer_stream(2, 4096)
        .unwrap();
    let arena = VulkanStableResourceArena::new(
        &device,
        VulkanStableResourceArenaConfig::new(8192, 256).unwrap(),
        &[
            VulkanStableResourceGroupLayout::Explicit {
                resource_slots: vec![0],
                resource_byte_counts: vec![64],
            },
            VulkanStableResourceGroupLayout::Explicit {
                resource_slots: vec![1],
                resource_byte_counts: vec![64],
            },
        ],
    )
    .unwrap();
    let allocations = arena
        .allocate_groups(
            &device,
            &[(&[0], &[64]), (&[1], &[64])],
            256,
        )
        .unwrap();
    let first = Arc::clone(&allocations[0][0]);
    let second = Arc::clone(&allocations[1][0]);
    let first_bytes = (0u8..64).collect::<Vec<_>>();
    let second_bytes = (64u8..128).collect::<Vec<_>>();
    let upload = transfer
        .submit(&[
            VulkanResidentBufferWriteRange::new(
                first.buffer(),
                first.buffer_byte_offset(),
                &first_bytes,
            )
            .unwrap(),
            VulkanResidentBufferWriteRange::new(
                second.buffer(),
                second.buffer_byte_offset(),
                &second_bytes,
            )
            .unwrap(),
        ])
        .unwrap();
    transfer.wait(&upload).unwrap();
    let mut table =
        VulkanStableResourceAddressTable::new(&device, &mut transfer, 2)
            .unwrap();
    let first_publication = table
        .publish_group(&mut transfer, &[(0, Arc::clone(&first))])
        .unwrap();
    let second_publication = table
        .publish_group(&mut transfer, &[(1, Arc::clone(&second))])
        .unwrap();

    let selections = Arc::new(
        device
            .create_resident_buffer(512 * size_of::<u32>())
            .unwrap(),
    );
    selections
        .write_bytes(&u32_words_bytes(&[0, 1, 1, 0]))
        .unwrap();
    let continuation_predicate =
        Arc::new(device.create_conditional_resident_buffer(4).unwrap());
    continuation_predicate
        .write_bytes(&1u32.to_le_bytes())
        .unwrap();
    let missing_queue = VulkanGpuResidencyMissQueue::new(&device, 512).unwrap();
    let gate = VulkanGpuResidencyGate::new(
        &device,
        &gate_shader,
        Arc::clone(&selections),
        table.shared_buffer(),
        table.slot_count(),
        missing_queue,
        Arc::clone(&continuation_predicate),
        None,
        VulkanGpuResidencyGateConfig {
            maximum_selection_count: 512,
            selection_count_per_lane: 8,
            selection_lane_stride_words: 8,
            selection_index_shift: 0,
            selection_index_mask: 0xffff,
            address_mapping:
                VulkanGpuResidencyAddressMapping::Partitioned {
                    member_slot_bases: vec![0],
                    resource_count: 2,
                },
            owned_resource_indices: None,
        },
    )
    .unwrap();
    let output = device.create_resident_buffer(4).unwrap();
    output.write_bytes(&0u32.to_le_bytes()).unwrap();
    let downstream = device
        .create_resident_kernel_dispatch(
            &downstream_shader,
            &[VulkanResidentKernelBufferBinding::new(0, &output, 4)
                .with_access(VulkanResidentKernelBufferAccess::ReadWrite)],
            1,
            1,
            4,
        )
        .unwrap();
    let sequence = device.create_resident_kernel_sequence().unwrap();
    let increment = 1u32.to_le_bytes();

    run_gpu_residency_gate_sequence(
        &device,
        &sequence,
        &gate,
        &downstream,
        &increment,
        4,
        41,
    );
    assert_eq!(
        u32::from_le_bytes(output.read_bytes(4).unwrap().try_into().unwrap()),
        1
    );
    assert_eq!(gate.notification_epoch().unwrap(), 0);
    assert!(
        gate.missing_snapshot()
            .unwrap()
            .requests
            .is_empty()
    );
    assert_eq!(
        gate.selected_resource_indices(4).unwrap(),
        BTreeSet::from([0, 1])
    );
    let resolved =
        stable_resource_bytes_to_u32(&gate.resolved_addresses_buffer().read_bytes(
            gate.resolved_addresses_buffer().byte_capacity(),
        ).unwrap());
    assert_eq!(&resolved[..6], &[1, 2, 2, 0, 0, 41]);
    let resolved_addresses = resolved[8..8 + resolved[2] as usize * 8]
        .chunks_exact(8)
        .map(|record| {
            (
                record[0],
                u64::from(record[2]) | (u64::from(record[3]) << 32),
            )
        })
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        resolved_addresses.get(&0),
        Some(&first.device_address())
    );
    assert_eq!(
        resolved_addresses.get(&1),
        Some(&second.device_address())
    );

    table
        .clear_group(&mut transfer, &second_publication)
        .unwrap();
    selections.write_bytes(&1u32.to_le_bytes()).unwrap();
    run_gpu_residency_gate_sequence(
        &device,
        &sequence,
        &gate,
        &downstream,
        &increment,
        1,
        42,
    );
    assert_eq!(
        gate.selected_resource_indices(1).unwrap(),
        BTreeSet::from([1])
    );
    assert_eq!(
        u32::from_le_bytes(output.read_bytes(4).unwrap().try_into().unwrap()),
        1
    );
    let missing = gate.missing_snapshot().unwrap();
    assert_eq!(missing.notification_epoch, 1);
    assert!(!missing.overflowed);
    assert_eq!(
        missing.requests,
        vec![VulkanGpuResidencyMissingRequest {
            checkpoint_tag: 42,
            resource_index: 1,
        }]
    );
    gate.acknowledge_missing_through(missing.published_count)
        .unwrap();
    assert!(
        gate.missing_snapshot()
            .unwrap()
            .requests
            .is_empty()
    );

    let second_republication = table
        .publish_group(&mut transfer, &[(1, Arc::clone(&second))])
        .unwrap();
    run_gpu_residency_gate_sequence(
        &device,
        &sequence,
        &gate,
        &downstream,
        &increment,
        1,
        43,
    );
    assert_eq!(
        u32::from_le_bytes(output.read_bytes(4).unwrap().try_into().unwrap()),
        2
    );
    assert_eq!(gate.notification_epoch().unwrap(), 1);
    assert!(
        gate.missing_snapshot()
            .unwrap()
            .requests
            .is_empty()
    );

    let wide_selections = (0..512)
        .map(|index| u32::from(index % 2 != 0))
        .collect::<Vec<_>>();
    selections
        .write_bytes(&u32_words_bytes(&wide_selections))
        .unwrap();
    run_gpu_residency_gate_sequence(
        &device,
        &sequence,
        &gate,
        &downstream,
        &increment,
        512,
        44,
    );
    assert_eq!(
        u32::from_le_bytes(output.read_bytes(4).unwrap().try_into().unwrap()),
        3
    );
    let wide_resolved =
        stable_resource_bytes_to_u32(&gate.resolved_addresses_buffer().read_bytes(
            gate.resolved_addresses_buffer().byte_capacity(),
        ).unwrap());
    assert_eq!(&wide_resolved[..6], &[1, 2, 2, 0, 0, 44]);

    table
        .clear_group(&mut transfer, &second_republication)
        .unwrap();
    run_gpu_residency_gate_sequence(
        &device,
        &sequence,
        &gate,
        &downstream,
        &increment,
        512,
        45,
    );
    assert_eq!(
        u32::from_le_bytes(output.read_bytes(4).unwrap().try_into().unwrap()),
        3
    );
    let wide_missing = gate.missing_snapshot().unwrap();
    assert_eq!(wide_missing.notification_epoch, 2);
    assert_eq!(
        wide_missing.requests,
        vec![VulkanGpuResidencyMissingRequest {
            checkpoint_tag: 45,
            resource_index: 1,
        }],
        "a wide batch must publish each missing resource once"
    );
    gate.acknowledge_missing_through(wide_missing.published_count)
        .unwrap();

    drop(sequence);
    drop(downstream);
    drop(gate);
    table
        .clear_group(&mut transfer, &first_publication)
        .unwrap();
    drop(table);
    drop(second);
    drop(first);
    drop(allocations);
    assert_eq!(arena.stats().unwrap().active_allocation_count, 0);
    arena.release_backing().unwrap();
    assert_eq!(arena.stats().unwrap(), VulkanStableResourceArenaStats::default());
}

#[test]
fn gpu_residency_gate_chain_resumes_at_first_blocked_gate_without_replaying_prefix() {
    let Some(device_index) = stable_resource_test_device_index() else {
        eprintln!(
            "skipping GPU residency gate-chain test: explicit Vulkan device unset"
        );
        return;
    };
    let gate_shader =
        vulkan_gpu_residency_gate_spirv_words().expect("embedded gate shader must be valid");
    let downstream_shader = compile_gpu_residency_gate_shader(
        GPU_RESIDENCY_GATE_DOWNSTREAM_SHADER,
    )
    .expect("GPU residency gate-chain test requires a GLSL compiler");
    let device =
        VulkanComputeDevice::new_for_physical_device_index(device_index).unwrap();
    let mut transfer = device
        .create_resident_transfer_stream(2, 4096)
        .unwrap();
    let arena = VulkanStableResourceArena::new(
        &device,
        VulkanStableResourceArenaConfig::new(8192, 256).unwrap(),
        &[
            VulkanStableResourceGroupLayout::Explicit {
                resource_slots: vec![0],
                resource_byte_counts: vec![64],
            },
            VulkanStableResourceGroupLayout::Explicit {
                resource_slots: vec![1],
                resource_byte_counts: vec![64],
            },
        ],
    )
    .unwrap();
    let allocations = arena
        .allocate_groups(
            &device,
            &[(&[0], &[64]), (&[1], &[64])],
            256,
        )
        .unwrap();
    let first = Arc::clone(&allocations[0][0]);
    let second = Arc::clone(&allocations[1][0]);
    let upload = transfer
        .submit(&[
            VulkanResidentBufferWriteRange::new(
                first.buffer(),
                first.buffer_byte_offset(),
                &[1; 64],
            )
            .unwrap(),
            VulkanResidentBufferWriteRange::new(
                second.buffer(),
                second.buffer_byte_offset(),
                &[2; 64],
            )
            .unwrap(),
        ])
        .unwrap();
    transfer.wait(&upload).unwrap();
    let mut table =
        VulkanStableResourceAddressTable::new(&device, &mut transfer, 2).unwrap();
    let selection_first = Arc::new(device.create_resident_buffer(4).unwrap());
    let selection_second = Arc::new(device.create_resident_buffer(4).unwrap());
    selection_first.write_bytes(&0u32.to_le_bytes()).unwrap();
    selection_second.write_bytes(&0u32.to_le_bytes()).unwrap();
    let continuation_predicate =
        Arc::new(device.create_conditional_resident_buffer(4).unwrap());
    continuation_predicate
        .write_bytes(&1u32.to_le_bytes())
        .unwrap();
    let missing_queue = VulkanGpuResidencyMissQueue::new(&device, 2).unwrap();
    let first_gate = VulkanGpuResidencyGate::new(
        &device,
        &gate_shader,
        selection_first,
        table.shared_buffer(),
        table.slot_count(),
        missing_queue.clone(),
        Arc::clone(&continuation_predicate),
        None,
        VulkanGpuResidencyGateConfig {
            maximum_selection_count: 1,
            selection_count_per_lane: 1,
            selection_lane_stride_words: 1,
            selection_index_shift: 0,
            selection_index_mask: u32::MAX,
            address_mapping:
                VulkanGpuResidencyAddressMapping::GroupTable {
                    resource_address_slots: vec![0],
                    resource_address_slot_offsets: vec![0, 1],
                },
            owned_resource_indices: None,
        },
    )
    .unwrap();
    let second_gate = VulkanGpuResidencyGate::new(
        &device,
        &gate_shader,
        selection_second,
        table.shared_buffer(),
        table.slot_count(),
        missing_queue.clone(),
        Arc::clone(&continuation_predicate),
        None,
        VulkanGpuResidencyGateConfig {
            maximum_selection_count: 1,
            selection_count_per_lane: 1,
            selection_lane_stride_words: 1,
            selection_index_shift: 0,
            selection_index_mask: u32::MAX,
            address_mapping:
                VulkanGpuResidencyAddressMapping::GroupTable {
                    resource_address_slots: vec![1],
                    resource_address_slot_offsets: vec![0, 1],
                },
            owned_resource_indices: None,
        },
    )
    .unwrap();
    let first_output = device.create_resident_buffer(4).unwrap();
    let second_output = device.create_resident_buffer(4).unwrap();
    first_output.write_bytes(&0u32.to_le_bytes()).unwrap();
    second_output.write_bytes(&0u32.to_le_bytes()).unwrap();
    let first_compute = device
        .create_resident_kernel_dispatch(
            &downstream_shader,
            &[VulkanResidentKernelBufferBinding::new(0, &first_output, 4)
                .with_access(VulkanResidentKernelBufferAccess::ReadWrite)],
            1,
            1,
            4,
        )
        .unwrap();
    let second_compute = device
        .create_resident_kernel_dispatch(
            &downstream_shader,
            &[VulkanResidentKernelBufferBinding::new(0, &second_output, 4)
                .with_access(VulkanResidentKernelBufferAccess::ReadWrite)],
            1,
            1,
            4,
        )
        .unwrap();
    let full_sequence = device.create_resident_kernel_sequence().unwrap();
    let resume_sequence = device.create_resident_kernel_sequence().unwrap();
    let first_control = first_gate.push_constants(1, 11, true, true, 0).unwrap();
    let second_control = second_gate.push_constants(1, 22, true, true, 0).unwrap();
    let increment = 1u32.to_le_bytes();
    device
        .record_resident_kernel_sequence(
            &full_sequence,
            &[
                VulkanResidentKernelSequenceStep::new(first_gate.dispatch(), &first_control),
                VulkanResidentKernelSequenceStep::new_conditional(
                    &first_compute,
                    &increment,
                    &continuation_predicate,
                    0,
                    false,
                    1,
                )
                .unwrap(),
                VulkanResidentKernelSequenceStep::new_conditional(
                    second_gate.dispatch(),
                    &second_control,
                    &continuation_predicate,
                    0,
                    false,
                    2,
                )
                .unwrap(),
                VulkanResidentKernelSequenceStep::new_conditional(
                    &second_compute,
                    &increment,
                    &continuation_predicate,
                    0,
                    false,
                    3,
                )
                .unwrap(),
            ],
        )
        .unwrap();
    device
        .record_resident_kernel_sequence(
            &resume_sequence,
            &[
                VulkanResidentKernelSequenceStep::new(second_gate.dispatch(), &second_control),
                VulkanResidentKernelSequenceStep::new_conditional(
                    &second_compute,
                    &increment,
                    &continuation_predicate,
                    0,
                    false,
                    1,
                )
                .unwrap(),
            ],
        )
        .unwrap();

    device
        .run_recorded_resident_kernel_sequence(&full_sequence)
        .unwrap();
    assert_eq!(first_output.read_bytes(4).unwrap(), 0u32.to_le_bytes());
    assert_eq!(second_output.read_bytes(4).unwrap(), 0u32.to_le_bytes());
    let first_miss = first_gate.missing_snapshot().unwrap();
    assert_eq!(first_miss.requests[0].checkpoint_tag, 11);
    first_gate
        .acknowledge_missing_through(first_miss.published_count)
        .unwrap();
    let first_publication = table
        .publish_group(&mut transfer, &[(0, Arc::clone(&first))])
        .unwrap();

    device
        .run_recorded_resident_kernel_sequence(&full_sequence)
        .unwrap();
    assert_eq!(first_output.read_bytes(4).unwrap(), 1u32.to_le_bytes());
    assert_eq!(second_output.read_bytes(4).unwrap(), 0u32.to_le_bytes());
    let second_miss = second_gate.missing_snapshot().unwrap();
    assert_eq!(second_miss.requests[0].checkpoint_tag, 22);
    second_gate
        .acknowledge_missing_through(second_miss.published_count)
        .unwrap();
    let second_publication = table
        .publish_group(&mut transfer, &[(1, Arc::clone(&second))])
        .unwrap();

    device
        .run_recorded_resident_kernel_sequence(&resume_sequence)
        .unwrap();
    assert_eq!(
        first_output.read_bytes(4).unwrap(),
        1u32.to_le_bytes(),
        "resuming the second checkpoint must not replay preceding selected work"
    );
    assert_eq!(second_output.read_bytes(4).unwrap(), 1u32.to_le_bytes());
    assert!(first_gate.missing_snapshot().unwrap().requests.is_empty());
    assert!(second_gate.missing_snapshot().unwrap().requests.is_empty());

    drop(full_sequence);
    drop(resume_sequence);
    drop(first_compute);
    drop(second_compute);
    drop(first_gate);
    drop(second_gate);
    drop(continuation_predicate);
    table
        .clear_group(&mut transfer, &second_publication)
        .unwrap();
    table
        .clear_group(&mut transfer, &first_publication)
        .unwrap();
    drop(table);
    drop(second);
    drop(first);
    drop(allocations);
    assert_eq!(arena.stats().unwrap().active_allocation_count, 0);
    arena.release_backing().unwrap();
    assert_eq!(arena.stats().unwrap(), VulkanStableResourceArenaStats::default());
}

#[test]
fn gpu_residency_gate_warm_path_is_measured_against_eager_dispatch() {
    let started = std::time::Instant::now();
    let Some(device_index) = stable_resource_test_device_index() else {
        eprintln!(
            "skipping GPU residency gate microbenchmark: explicit Vulkan device unset"
        );
        return;
    };
    let gate_shader =
        compile_gpu_residency_gate_shader(GPU_RESIDENCY_GATE_SHADER)
            .expect("GPU residency gate microbenchmark requires a GLSL compiler");
    let table_shader = compile_gpu_residency_gate_shader(
        "tests/fixtures/vulkan/stable_resource_table_benchmark.comp",
    )
    .expect("GPU residency gate microbenchmark requires a GLSL compiler");

    const ELEMENT_COUNT: usize = 4 * 1024 * 1024;
    const BYTE_COUNT: usize = ELEMENT_COUNT * size_of::<u32>();
    let device =
        VulkanComputeDevice::new_for_physical_device_index(device_index)
            .unwrap();
    let mut transfer = device
        .create_resident_transfer_stream(2, BYTE_COUNT)
        .unwrap();
    let arena = VulkanStableResourceArena::new(
        &device,
        VulkanStableResourceArenaConfig::new(
            BYTE_COUNT + 256,
            256,
        )
        .unwrap(),
        &[VulkanStableResourceGroupLayout::Explicit {
            resource_slots: vec![0],
            resource_byte_counts: vec![BYTE_COUNT],
        }],
    )
    .unwrap();
    let source = Arc::clone(
        &arena
            .allocate_groups(&device, &[(&[0], &[BYTE_COUNT])], 256)
            .unwrap()[0][0],
    );
    let source_values = (0..ELEMENT_COUNT as u32)
        .map(|value| value.rotate_left(7) ^ 0xa5a5_5a5a)
        .collect::<Vec<_>>();
    let source_bytes = stable_resource_u32_bytes(&source_values);
    let upload = transfer
        .submit(&[
            VulkanResidentBufferWriteRange::new(
                source.buffer(),
                source.buffer_byte_offset(),
                &source_bytes,
            )
            .unwrap(),
        ])
        .unwrap();
    transfer.wait(&upload).unwrap();
    let mut table =
        VulkanStableResourceAddressTable::new(&device, &mut transfer, 1)
            .unwrap();
    let publication = table
        .publish_group(&mut transfer, &[(0, Arc::clone(&source))])
        .unwrap();
    let selection = Arc::new(device.create_resident_buffer(4).unwrap());
    selection.write_bytes(&0u32.to_le_bytes()).unwrap();
    let workgroup_count = u32::try_from(ELEMENT_COUNT / 256).unwrap();
    let missing_queue = VulkanGpuResidencyMissQueue::new(&device, 1).unwrap();
    let gate = VulkanGpuResidencyGate::new(
        &device,
        &gate_shader,
        Arc::clone(&selection),
        table.shared_buffer(),
        table.slot_count(),
        missing_queue,
        Arc::new(
            device
                .create_conditional_resident_buffer(4)
                .unwrap(),
        ),
        None,
        VulkanGpuResidencyGateConfig {
            maximum_selection_count: 1,
            selection_count_per_lane: 1,
            selection_lane_stride_words: 1,
            selection_index_shift: 0,
            selection_index_mask: u32::MAX,
            address_mapping:
                VulkanGpuResidencyAddressMapping::GroupTable {
                    resource_address_slots: vec![0],
                    resource_address_slot_offsets: vec![0, 1],
                },
            owned_resource_indices: None,
        },
    )
    .unwrap();
    let eager_output = device.create_resident_buffer(BYTE_COUNT).unwrap();
    let demand_output = device.create_resident_buffer(BYTE_COUNT).unwrap();
    eager_output.write_bytes(&vec![0; BYTE_COUNT]).unwrap();
    demand_output.write_bytes(&vec![0; BYTE_COUNT]).unwrap();
    let eager_dispatch = stable_resource_table_benchmark_dispatch(
        &device,
        &table,
        &eager_output,
        &table_shader,
        workgroup_count,
    );
    let demand_dispatch = stable_resource_table_benchmark_dispatch(
        &device,
        &table,
        &demand_output,
        &table_shader,
        workgroup_count,
    );
    let eager_sequence = device
        .create_timestamped_resident_kernel_sequence()
        .unwrap();
    let demand_sequence = device
        .create_timestamped_resident_kernel_sequence()
        .unwrap();
    let element_count = u32::try_from(ELEMENT_COUNT).unwrap().to_le_bytes();
    let gate_control = gate.push_constants(1, 7, true, true, 0).unwrap();
    device
        .record_resident_kernel_sequence(
            &eager_sequence,
            &[VulkanResidentKernelSequenceStep::new(
                &eager_dispatch,
                &element_count,
            )],
        )
        .unwrap();
    device
        .record_resident_kernel_sequence(
            &demand_sequence,
            &[
                VulkanResidentKernelSequenceStep::new(
                    gate.dispatch(),
                    &gate_control,
                ),
                VulkanResidentKernelSequenceStep::new_conditional(
                    &demand_dispatch,
                    &element_count,
                    gate.continuation_predicate(),
                    0,
                    false,
                    1,
                )
                .unwrap(),
            ],
        )
        .unwrap();

    let timeout = std::time::Duration::from_secs(5);
    device
        .run_timestamped_recorded_resident_kernel_sequence_for(
            &eager_sequence,
            timeout,
        )
        .unwrap();
    device
        .run_timestamped_recorded_resident_kernel_sequence_for(
            &demand_sequence,
            timeout,
        )
        .unwrap();
    let mut eager_ns = Vec::with_capacity(2);
    let mut demand_ns = Vec::with_capacity(2);
    for _ in 0..2 {
        eager_ns.push(
            device
                .run_timestamped_recorded_resident_kernel_sequence_for(
                    &eager_sequence,
                    timeout,
                )
                .unwrap(),
        );
        demand_ns.push(
            device
                .run_timestamped_recorded_resident_kernel_sequence_for(
                    &demand_sequence,
                    timeout,
                )
                .unwrap(),
        );
    }
    assert_eq!(
        eager_output.read_bytes(BYTE_COUNT).unwrap(),
        demand_output.read_bytes(BYTE_COUNT).unwrap()
    );
    assert_eq!(gate.notification_epoch().unwrap(), 0);
    assert!(
        gate.missing_snapshot()
            .unwrap()
            .requests
            .is_empty()
    );
    let eager_average_ns = eager_ns.iter().sum::<u64>() / 2;
    let demand_average_ns = demand_ns.iter().sum::<u64>() / 2;
    let ratio = demand_average_ns as f64 / eager_average_ns as f64;
    eprintln!(
        "gpu_residency_gate_microbenchmark eager_ns={eager_ns:?} demand_ns={demand_ns:?} eager_average_ns={eager_average_ns} demand_average_ns={demand_average_ns} demand_to_eager_ratio={ratio:.6} elapsed_ms={:.3}",
        started.elapsed().as_secs_f64() * 1000.0,
    );
    assert!(
        ratio <= 1.20,
        "warm GPU residency gate regressed representative execution by {:.2}%",
        (ratio - 1.0) * 100.0,
    );
    assert!(
        started.elapsed() < std::time::Duration::from_secs(60),
        "GPU residency gate microbenchmark exceeded one minute"
    );

    drop(eager_sequence);
    drop(demand_sequence);
    drop(eager_dispatch);
    drop(demand_dispatch);
    drop(gate);
    table
        .clear_group(&mut transfer, &publication)
        .unwrap();
    drop(table);
    drop(source);
    assert_eq!(arena.stats().unwrap().active_allocation_count, 0);
    arena.release_backing().unwrap();
    assert_eq!(arena.stats().unwrap(), VulkanStableResourceArenaStats::default());
}

#[test]
fn conditional_direct_compute_chain_is_measured_against_indirect_dispatch() {
    let started = std::time::Instant::now();
    let Some(device_index) = stable_resource_test_device_index() else {
        eprintln!(
            "skipping direct/indirect compute microbenchmark: explicit Vulkan device unset"
        );
        return;
    };
    let shader = compile_gpu_residency_gate_shader(
        GPU_RESIDENCY_GATE_DOWNSTREAM_SHADER,
    )
    .expect("direct/indirect compute microbenchmark requires a GLSL compiler");
    let guarded_shader = compile_gpu_residency_gate_shader(
        GPU_RESIDENCY_GUARDED_DOWNSTREAM_SHADER,
    )
    .expect("guarded-direct compute microbenchmark requires a GLSL compiler");
    let device =
        VulkanComputeDevice::new_for_physical_device_index(device_index).unwrap();
    const DISPATCH_COUNT: usize = 256;
    let direct_output = device.create_resident_buffer(4).unwrap();
    let guarded_output = device.create_resident_buffer(4).unwrap();
    let conditional_output = device.create_resident_buffer(4).unwrap();
    let indirect_output = device.create_resident_buffer(4).unwrap();
    let continuation = device.create_resident_buffer(4).unwrap();
    let predicate = device.create_conditional_resident_buffer(4).unwrap();
    direct_output.write_bytes(&0u32.to_le_bytes()).unwrap();
    guarded_output.write_bytes(&0u32.to_le_bytes()).unwrap();
    conditional_output.write_bytes(&0u32.to_le_bytes()).unwrap();
    indirect_output.write_bytes(&0u32.to_le_bytes()).unwrap();
    continuation.write_bytes(&1u32.to_le_bytes()).unwrap();
    predicate.write_bytes(&1u32.to_le_bytes()).unwrap();
    let direct_dispatch = device
        .create_resident_kernel_dispatch(
            &shader,
            &[VulkanResidentKernelBufferBinding::new(0, &direct_output, 4)
                .with_access(VulkanResidentKernelBufferAccess::ReadWrite)],
            1,
            1,
            4,
        )
        .unwrap();
    let conditional_dispatch = device
        .create_resident_kernel_dispatch(
            &shader,
            &[VulkanResidentKernelBufferBinding::new(
                0,
                &conditional_output,
                4,
            )
            .with_access(VulkanResidentKernelBufferAccess::ReadWrite)],
            1,
            1,
            4,
        )
        .unwrap();
    let guarded_dispatch = device
        .create_resident_kernel_dispatch(
            &guarded_shader,
            &[
                VulkanResidentKernelBufferBinding::new(0, &guarded_output, 4)
                    .with_access(VulkanResidentKernelBufferAccess::ReadWrite),
                VulkanResidentKernelBufferBinding::new(1, &continuation, 4),
            ],
            1,
            1,
            4,
        )
        .unwrap();
    let indirect_dispatch = device
        .create_resident_kernel_dispatch(
            &shader,
            &[VulkanResidentKernelBufferBinding::new(0, &indirect_output, 4)
                .with_access(VulkanResidentKernelBufferAccess::ReadWrite)],
            1,
            1,
            4,
        )
        .unwrap();
    let indirect_dimensions = Arc::new(
        device
            .create_resident_buffer(
                DISPATCH_COUNT * VULKAN_RESIDENT_INDIRECT_DISPATCH_BYTE_COUNT,
            )
            .unwrap(),
    );
    indirect_dimensions
        .write_bytes(
            &(0..DISPATCH_COUNT)
                .flat_map(|_| [1u32, 1, 1])
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let direct_sequence = device
        .create_timestamped_resident_kernel_sequence()
        .unwrap();
    let guarded_sequence = device
        .create_timestamped_resident_kernel_sequence()
        .unwrap();
    let conditional_sequence = device
        .create_timestamped_resident_kernel_sequence()
        .unwrap();
    let indirect_sequence = device
        .create_timestamped_resident_kernel_sequence()
        .unwrap();
    let increment = 1u32.to_le_bytes();
    let direct_steps = (0..DISPATCH_COUNT)
        .map(|_| VulkanResidentKernelSequenceStep::new(&direct_dispatch, &increment))
        .collect::<Vec<_>>();
    let guarded_steps = (0..DISPATCH_COUNT)
        .map(|_| {
            VulkanResidentKernelSequenceStep::new(
                &guarded_dispatch,
                &increment,
            )
        })
        .collect::<Vec<_>>();
    let conditional_steps = (0..DISPATCH_COUNT)
        .map(|_| {
            VulkanResidentKernelSequenceStep::new_conditional(
                &conditional_dispatch,
                &increment,
                &predicate,
                0,
                false,
                1,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let indirect_steps = (0..DISPATCH_COUNT)
        .map(|dispatch_index| {
            VulkanResidentKernelSequenceStep::new_indirect(
                &indirect_dispatch,
                &increment,
                &indirect_dimensions,
                dispatch_index * VULKAN_RESIDENT_INDIRECT_DISPATCH_BYTE_COUNT,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    device
        .record_resident_kernel_sequence(&direct_sequence, &direct_steps)
        .unwrap();
    device
        .record_resident_kernel_sequence(&guarded_sequence, &guarded_steps)
        .unwrap();
    device
        .record_resident_kernel_sequence(
            &conditional_sequence,
            &conditional_steps,
        )
        .unwrap();
    device
        .record_resident_kernel_sequence(&indirect_sequence, &indirect_steps)
        .unwrap();
    let timeout = std::time::Duration::from_secs(5);
    device
        .run_timestamped_recorded_resident_kernel_sequence_for(
            &direct_sequence,
            timeout,
        )
        .unwrap();
    device
        .run_timestamped_recorded_resident_kernel_sequence_for(
            &guarded_sequence,
            timeout,
        )
        .unwrap();
    device
        .run_timestamped_recorded_resident_kernel_sequence_for(
            &conditional_sequence,
            timeout,
        )
        .unwrap();
    device
        .run_timestamped_recorded_resident_kernel_sequence_for(
            &indirect_sequence,
            timeout,
        )
        .unwrap();
    let mut direct_ns = Vec::with_capacity(2);
    let mut guarded_ns = Vec::with_capacity(2);
    let mut conditional_ns = Vec::with_capacity(2);
    let mut indirect_ns = Vec::with_capacity(2);
    for _ in 0..2 {
        direct_ns.push(
            device
                .run_timestamped_recorded_resident_kernel_sequence_for(
                    &direct_sequence,
                    timeout,
                )
                .unwrap(),
        );
        guarded_ns.push(
            device
                .run_timestamped_recorded_resident_kernel_sequence_for(
                    &guarded_sequence,
                    timeout,
                )
                .unwrap(),
        );
        conditional_ns.push(
            device
                .run_timestamped_recorded_resident_kernel_sequence_for(
                    &conditional_sequence,
                    timeout,
                )
                .unwrap(),
        );
        indirect_ns.push(
            device
                .run_timestamped_recorded_resident_kernel_sequence_for(
                    &indirect_sequence,
                    timeout,
                )
                .unwrap(),
        );
    }
    assert_eq!(
        direct_output.read_bytes(4).unwrap(),
        guarded_output.read_bytes(4).unwrap()
    );
    assert_eq!(
        direct_output.read_bytes(4).unwrap(),
        conditional_output.read_bytes(4).unwrap()
    );
    assert_eq!(
        direct_output.read_bytes(4).unwrap(),
        indirect_output.read_bytes(4).unwrap()
    );
    let direct_average_ns = direct_ns.iter().sum::<u64>() / 2;
    let guarded_average_ns = guarded_ns.iter().sum::<u64>() / 2;
    let conditional_average_ns = conditional_ns.iter().sum::<u64>() / 2;
    let indirect_average_ns = indirect_ns.iter().sum::<u64>() / 2;
    let guarded_ratio = guarded_average_ns as f64 / direct_average_ns as f64;
    let conditional_ratio =
        conditional_average_ns as f64 / direct_average_ns as f64;
    let indirect_ratio = indirect_average_ns as f64 / direct_average_ns as f64;
    eprintln!(
        "recorded_compute_dispatch_microbenchmark direct_ns={direct_ns:?} guarded_ns={guarded_ns:?} conditional_ns={conditional_ns:?} indirect_ns={indirect_ns:?} direct_average_ns={direct_average_ns} guarded_average_ns={guarded_average_ns} conditional_average_ns={conditional_average_ns} indirect_average_ns={indirect_average_ns} guarded_to_direct_ratio={guarded_ratio:.6} conditional_to_direct_ratio={conditional_ratio:.6} indirect_to_direct_ratio={indirect_ratio:.6} elapsed_ms={:.3}",
        started.elapsed().as_secs_f64() * 1000.0,
    );
    assert!(
        conditional_average_ns < indirect_average_ns,
        "conditional direct dispatch must beat indirect dispatch: conditional={conditional_average_ns}ns indirect={indirect_average_ns}ns"
    );
    assert!(
        started.elapsed() < std::time::Duration::from_secs(60),
        "direct/indirect compute microbenchmark exceeded one minute"
    );
}

fn run_gpu_residency_gate_sequence(
    device: &VulkanComputeDevice,
    sequence: &VulkanResidentKernelSequence,
    gate: &VulkanGpuResidencyGate,
    downstream: &VulkanResidentKernelDispatch,
    downstream_push_constants: &[u8],
    selection_count: usize,
    checkpoint_tag: u32,
) {
    let gate_control = gate
        .push_constants(selection_count, checkpoint_tag, true, true, 0)
        .unwrap();
    device
        .record_resident_kernel_sequence(
            sequence,
            &[
                VulkanResidentKernelSequenceStep::new(
                    gate.dispatch(),
                    &gate_control,
                ),
                VulkanResidentKernelSequenceStep::new_conditional(
                    downstream,
                    downstream_push_constants,
                    gate.continuation_predicate(),
                    0,
                    false,
                    1,
                )
                .unwrap(),
            ],
        )
        .unwrap();
    device.run_recorded_resident_kernel_sequence(sequence).unwrap();
}

fn stable_resource_table_benchmark_dispatch(
    device: &VulkanComputeDevice,
    table: &VulkanStableResourceAddressTable,
    output: &VulkanResidentBuffer,
    shader: &[u32],
    workgroup_count: u32,
) -> VulkanResidentKernelDispatch {
    device
        .create_resident_kernel_dispatch(
            shader,
            &[
                VulkanResidentKernelBufferBinding::new(
                    0,
                    table.buffer(),
                    table.byte_capacity(),
                )
                .with_access(VulkanResidentKernelBufferAccess::Read),
                VulkanResidentKernelBufferBinding::new(
                    1,
                    output,
                    output.byte_capacity(),
                )
                .with_access(VulkanResidentKernelBufferAccess::Write),
            ],
            workgroup_count,
            256,
            4,
        )
        .unwrap()
}

fn compile_gpu_residency_gate_shader(relative: &str) -> Option<Vec<u32>> {
    let source_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(relative);
    compile_shader_words_from_source_path(&source_path)
}
