#[test]
fn stable_resource_address_contract_validates_alignment_and_layout() {
    assert!(VulkanStableResourceArenaConfig::new(8192, 8).is_ok());
    assert!(VulkanStableResourceArenaConfig::new(0, 8).is_err());
    assert!(VulkanStableResourceArenaConfig::new(8192, 4).is_err());
    assert!(VulkanStableResourceArenaConfig::new(8192, 24).is_err());
    assert_eq!(
        VulkanStableResourceArenaConfig::new(8192, 8)
            .unwrap()
            .host_visible()
            .memory_domain,
        VulkanStableResourceMemoryDomain::HostVisible
    );
    assert_eq!(
        std::mem::size_of::<VulkanStableResourceAddressRecord>(),
        VULKAN_STABLE_RESOURCE_ADDRESS_RECORD_BYTE_COUNT
    );
    assert_eq!(
        std::mem::align_of::<VulkanStableResourceAddressRecord>(),
        std::mem::align_of::<u64>()
    );
    let record = VulkanStableResourceAddressRecord {
        device_address: 0x0102_0304_0506_0708,
        byte_count: 0x1112_1314_1516_1718,
        generation: 0x2122_2324_2526_2728,
        resident: 0x3132_3334,
        representation: 0x4142_4344,
    };
    assert_eq!(
        record.bytes(),
        [
            0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x18, 0x17, 0x16, 0x15, 0x14, 0x13,
            0x12, 0x11, 0x28, 0x27, 0x26, 0x25, 0x24, 0x23, 0x22, 0x21, 0x34, 0x33, 0x32, 0x31,
            0x44, 0x43, 0x42, 0x41,
        ]
    );
}

#[test]
fn inactive_chunk_trimming_uses_the_smallest_sufficient_backing() {
    let mut candidates = vec![(1, 4096), (2, 1024), (3, 2048)];

    assert_eq!(
        inactive_stable_resource_chunks_to_trim(&mut candidates, 1500),
        vec![3],
    );
}

#[test]
fn inactive_chunk_trimming_combines_largest_backing_only_when_required() {
    let mut candidates = vec![(1, 4096), (2, 1024), (3, 2048)];

    assert_eq!(
        inactive_stable_resource_chunks_to_trim(&mut candidates, 5000),
        vec![1, 3],
    );
}

#[test]
fn stable_resource_allocations_are_attributed_to_compiled_slots() {
    let Some(device_index) = stable_resource_test_device_index() else {
        eprintln!("skipping stable resource attribution test: explicit Vulkan device unset");
        return;
    };
    let device = VulkanComputeDevice::new_for_physical_device_index(device_index).unwrap();
    let arena = VulkanStableResourceArena::new(
        &device,
        VulkanStableResourceArenaConfig::new(64 * 1024, 256).unwrap(),
        &[VulkanStableResourceGroupLayout::Explicit {
            resource_slots: vec![7, 17],
            resource_byte_counts: vec![1024, 2048],
        }],
    )
    .unwrap();
    let mut groups = arena
        .allocate_groups(&device, &[(&[7, 17], &[1024, 2048])], 256)
        .unwrap();
    let allocations = groups.pop().unwrap();

    for (expected_slot, allocation) in [7, 17].into_iter().zip(&allocations) {
        let resolved = device
            .device_address_registry
            .lock()
            .unwrap()
            .resolve(allocation.device_address() + allocation.byte_count() as u64 - 1)
            .unwrap();
        assert!(resolved.label.contains(&format!("resource slot={expected_slot}")));
        assert_eq!(resolved.byte_offset, allocation.byte_count() - 1);
        assert_eq!(resolved.byte_capacity, allocation.byte_count());
    }

    let first_address = allocations[0].device_address();
    drop(allocations);
    assert!(
        device
            .device_address_registry
            .lock()
            .unwrap()
            .resolve(first_address)
            .is_none()
    );
    arena.release_backing().unwrap();
}

#[test]
fn stable_resource_arena_preflights_exact_physical_chunk_capacity() {
    let Some(device_index) = stable_resource_test_device_index() else {
        eprintln!("skipping stable resource planning test: explicit Vulkan device unset");
        return;
    };
    let device = VulkanComputeDevice::new_for_physical_device_index(device_index).unwrap();
    let arena = VulkanStableResourceArena::new(
        &device,
        VulkanStableResourceArenaConfig::new(4096, 256).unwrap(),
        &[
            VulkanStableResourceGroupLayout::Explicit {
                resource_slots: vec![0, 1],
                resource_byte_counts: vec![13, 17],
            },
            VulkanStableResourceGroupLayout::Explicit {
                resource_slots: vec![2],
                resource_byte_counts: vec![300],
            },
        ],
    )
    .unwrap();
    let requests = [(&[0, 1][..], &[13, 17][..]), (&[2][..], &[300][..])];

    let planned_bytes = arena
        .additional_committed_byte_capacity_for_groups(&device, &requests, 256)
        .unwrap();
    assert_eq!(planned_bytes, 1024);
    let capacity_permit = device
        .reserve_device_local_memory_capacity(planned_bytes)
        .unwrap();
    assert_eq!(
        device
            .device_local_memory_accounting()
            .unwrap()
            .pending_reservation_bytes,
        planned_bytes as u64,
    );
    let first_groups = arena
        .allocate_groups_with_capacity_permit(
            &device,
            &requests,
            256,
            capacity_permit,
        )
        .unwrap();
    assert_eq!(
        device
            .device_local_memory_accounting()
            .unwrap()
            .pending_reservation_bytes,
        0,
    );
    let chunk_id = first_groups[0][0].chunk_id();
    assert_eq!(
        arena.committed_byte_capacity_for_chunk(chunk_id).unwrap(),
        1024,
    );
    let replacement_bytes = arena
        .additional_committed_byte_capacity_for_groups(&device, &requests, 256)
        .unwrap();
    assert_eq!(replacement_bytes, 1024);
    let replacement_permit = device
        .reserve_device_local_memory_capacity(replacement_bytes)
        .unwrap();
    let replacement_groups = arena
        .allocate_groups_with_capacity_permit(
            &device,
            &requests,
            256,
            replacement_permit,
        )
        .unwrap();
    assert_ne!(
        replacement_groups[0][0].allocation_id(),
        first_groups[0][0].allocation_id(),
        "a stable slot layout may have multiple physical generations"
    );
    assert_eq!(arena.stats().unwrap().active_allocation_count, 6);
    assert_eq!(arena.stats().unwrap().committed_byte_capacity, 2048);

    drop(first_groups);
    assert_eq!(arena.stats().unwrap().active_allocation_count, 3);
    assert_eq!(arena.stats().unwrap().committed_byte_capacity, 2048);
    assert_eq!(
        arena
            .additional_committed_byte_capacity_for_groups(&device, &requests, 256)
            .unwrap(),
        0,
        "an inactive chunk must satisfy the next physical generation without another Vulkan allocation",
    );
    let reused_groups = arena.allocate_groups(&device, &requests, 256).unwrap();
    assert_eq!(reused_groups[0][0].chunk_id(), chunk_id);
    assert_eq!(arena.stats().unwrap().active_allocation_count, 6);
    assert_eq!(arena.stats().unwrap().committed_byte_capacity, 2048);
    assert_eq!(arena.stats().unwrap().chunk_count, 2);
    drop(reused_groups);
    drop(replacement_groups);
    arena.release_backing().unwrap();
}

#[test]
fn stable_resource_arena_reuses_one_chunk_across_many_logical_evictions() {
    let Some(device_index) = stable_resource_test_device_index() else {
        eprintln!("skipping stable resource churn test: explicit Vulkan device unset");
        return;
    };
    let device = VulkanComputeDevice::new_for_physical_device_index(device_index).unwrap();
    let arena = VulkanStableResourceArena::new(
        &device,
        VulkanStableResourceArenaConfig::new(4096, 256).unwrap(),
        &[VulkanStableResourceGroupLayout::Explicit {
            resource_slots: vec![0, 1],
            resource_byte_counts: vec![257, 513],
        }],
    )
    .unwrap();
    let requests = [(&[0, 1][..], &[257, 513][..])];
    let (_, logical_chunk_byte_count) =
        stable_group_member_layout(&[257, 513], 256).unwrap();
    let physical_chunk_byte_count = device
        .addressable_resident_buffer_memory_requirement_bytes(logical_chunk_byte_count)
        .unwrap();
    let mut committed_byte_capacity = None;

    // This deliberately exceeds the 649 logical eviction cycles observed in
    // the crash-producing DeepSeek run. Logical generations must not become
    // Vulkan allocation/free generations.
    for generation in 0..1024 {
        assert_eq!(
            arena
                .additional_committed_byte_capacity_for_groups(&device, &requests, 256)
                .unwrap(),
            if generation == 0 {
                physical_chunk_byte_count
            } else {
                0
            },
        );
        let groups = arena.allocate_groups(&device, &requests, 256).unwrap();
        let stats = arena.stats().unwrap();
        assert_eq!(stats.chunk_count, 1);
        assert_eq!(stats.active_allocation_count, 2);
        assert_eq!(stats.allocated_byte_count, 770);
        match committed_byte_capacity {
            Some(capacity) => assert_eq!(stats.committed_byte_capacity, capacity),
            None => committed_byte_capacity = Some(stats.committed_byte_capacity),
        }
        drop(groups);
        let inactive = arena.stats().unwrap();
        assert_eq!(inactive.chunk_count, 1);
        assert_eq!(inactive.active_allocation_count, 0);
        assert_eq!(inactive.allocated_byte_count, 0);
        assert_eq!(
            inactive.committed_byte_capacity,
            committed_byte_capacity.unwrap()
        );
    }

    arena.release_backing().unwrap();
    assert_eq!(arena.stats().unwrap(), VulkanStableResourceArenaStats::default());
}

#[test]
fn stable_resource_arena_converges_for_alternating_chunk_shapes() {
    let Some(device_index) = stable_resource_test_device_index() else {
        eprintln!("skipping stable resource alternating-shape test: explicit Vulkan device unset");
        return;
    };
    let device = VulkanComputeDevice::new_for_physical_device_index(device_index).unwrap();
    let arena = VulkanStableResourceArena::new(
        &device,
        VulkanStableResourceArenaConfig::new(32 * 1024, 256).unwrap(),
        &[
            VulkanStableResourceGroupLayout::Explicit {
                resource_slots: vec![0],
                resource_byte_counts: vec![257],
            },
            VulkanStableResourceGroupLayout::Explicit {
                resource_slots: vec![1],
                resource_byte_counts: vec![4097],
            },
        ],
    )
    .unwrap();
    let small = [(&[0][..], &[257][..])];
    let large = [(&[1][..], &[4097][..])];
    let mut converged_commitment = None;

    for generation in 0..256 {
        let request = if generation % 2 == 0 { &small } else { &large };
        let additional = arena
            .additional_committed_byte_capacity_for_groups(&device, request, 256)
            .unwrap();
        if generation >= 2 {
            assert_eq!(
                additional, 0,
                "alternating logical shapes must reuse their established backing",
            );
        }
        let groups = arena.allocate_groups(&device, request, 256).unwrap();
        let active = arena.stats().unwrap();
        assert!(active.chunk_count <= 2);
        if generation >= 1 {
            match converged_commitment {
                Some(expected) => assert_eq!(active.committed_byte_capacity, expected),
                None => converged_commitment = Some(active.committed_byte_capacity),
            }
        }
        drop(groups);
    }

    assert_eq!(arena.stats().unwrap().chunk_count, 2);
    arena.release_backing().unwrap();
}

#[test]
fn stable_resource_commitment_uses_vulkan_requirements_not_logical_bytes() {
    let Some(device_index) = stable_resource_test_device_index() else {
        eprintln!("skipping physical commitment test: explicit Vulkan device unset");
        return;
    };
    let device = VulkanComputeDevice::new_for_physical_device_index(device_index).unwrap();
    let requirements = device
        .addressable_resident_buffer_memory_requirement_bytes(8)
        .unwrap();
    assert!(requirements >= 8);
    let arena = VulkanStableResourceArena::new(
        &device,
        VulkanStableResourceArenaConfig::new(requirements, 8).unwrap(),
        &[VulkanStableResourceGroupLayout::Explicit {
            resource_slots: vec![0],
            resource_byte_counts: vec![8],
        }],
    )
    .unwrap();
    let requests = [(&[0][..], &[8][..])];
    let planned = arena
        .additional_committed_byte_capacity_for_groups(&device, &requests, 8)
        .unwrap();
    assert_eq!(planned, requirements);
    let permit = device
        .reserve_device_local_memory_capacity(planned)
        .unwrap();
    let groups = arena
        .allocate_groups_with_capacity_permit(&device, &requests, 8, permit)
        .unwrap();

    assert_eq!(arena.stats().unwrap().committed_byte_capacity, requirements);
    drop(groups);
    arena.release_backing().unwrap();
}

#[test]
fn host_visible_stable_resource_is_directly_gpu_addressable() {
    let Some(device_index) = stable_resource_test_device_index() else {
        eprintln!("skipping host-visible stable resource test: explicit Vulkan device unset");
        return;
    };
    let Some(shader) = compile_stable_resource_shader(
        "host_visible_visibility",
        STABLE_RESOURCE_VISIBILITY_SHADER,
    ) else {
        eprintln!("skipping host-visible stable resource test: no GLSL compiler");
        return;
    };
    let device = VulkanComputeDevice::new_for_physical_device_index(device_index).unwrap();
    let arena = VulkanStableResourceArena::new(
        &device,
        VulkanStableResourceArenaConfig::new(64 * 1024, 256)
            .unwrap()
            .host_visible(),
        &[VulkanStableResourceGroupLayout::Explicit {
            resource_slots: vec![0],
            resource_byte_counts: vec![1024],
        }],
    )
    .unwrap();
    let allocation = Arc::clone(
        &arena
            .allocate_groups(&device, &[(&[0], &[1024])], 256)
            .unwrap()[0][0],
    );
    assert!(allocation.buffer().memory_access.is_directly_mappable());
    let values = (0..256u32)
        .map(|value| value.wrapping_mul(7).wrapping_add(23))
        .collect::<Vec<_>>();
    let value_bytes = stable_resource_u32_bytes(&values);
    allocation
        .buffer()
        .write_bytes_at(allocation.buffer_byte_offset(), &value_bytes)
        .unwrap();
    let mut transfer = device
        .create_resident_transfer_stream(2, 64 * 1024)
        .unwrap();
    let mut table = VulkanStableResourceAddressTable::new(&device, &mut transfer, 1).unwrap();
    let publications = table
        .publish_group(&mut transfer, &[(0, Arc::clone(&allocation))])
        .unwrap();
    let output = device.create_resident_buffer(16).unwrap();
    output.write_bytes(&[0; 16]).unwrap();
    let dispatch = device
        .create_resident_kernel_dispatch(
            &shader,
            &[
                VulkanResidentKernelBufferBinding::new(0, table.buffer(), table.byte_capacity())
                    .with_access(VulkanResidentKernelBufferAccess::Read),
                VulkanResidentKernelBufferBinding::new(1, &output, 16)
                    .with_access(VulkanResidentKernelBufferAccess::Write),
            ],
            1,
            1,
            4,
        )
        .unwrap();
    device
        .run_resident_kernel_dispatch(&dispatch, &0u32.to_le_bytes())
        .unwrap();
    assert_eq!(
        stable_resource_bytes_to_u32(&output.read_bytes(16).unwrap()),
        vec![23, values[255], 1, 1]
    );
    table.clear_group(&mut transfer, &publications).unwrap();
    drop(allocation);
    arena.release_backing().unwrap();
}

#[test]
fn stable_resource_groups_atomically_exchange_physical_backing() {
    let Some(device_index) = stable_resource_test_device_index() else {
        eprintln!("skipping stable resource exchange test: explicit Vulkan device unset");
        return;
    };
    let Some(shader) = compile_stable_resource_shader(
        "atomic_backing_exchange",
        STABLE_RESOURCE_VISIBILITY_SHADER,
    ) else {
        eprintln!("skipping stable resource exchange test: no GLSL compiler");
        return;
    };
    let device = VulkanComputeDevice::new_for_physical_device_index(device_index).unwrap();
    let layouts = [VulkanStableResourceGroupLayout::Explicit {
        resource_slots: vec![0],
        resource_byte_counts: vec![1024],
    }];
    let device_arena = VulkanStableResourceArena::new(
        &device,
        VulkanStableResourceArenaConfig::new(64 * 1024, 256).unwrap(),
        &layouts,
    )
    .unwrap();
    let host_arena = VulkanStableResourceArena::new(
        &device,
        VulkanStableResourceArenaConfig::new(64 * 1024, 256)
            .unwrap()
            .host_visible(),
        &layouts,
    )
    .unwrap();
    let device_allocation = Arc::clone(
        &device_arena
            .allocate_groups(&device, &[(&[0], &[1024])], 256)
            .unwrap()[0][0],
    );
    let host_allocation = Arc::clone(
        &host_arena
            .allocate_groups(&device, &[(&[0], &[1024])], 256)
            .unwrap()[0][0],
    );
    let device_values = (0..256u32)
        .map(|value| value.wrapping_mul(3).wrapping_add(11))
        .collect::<Vec<_>>();
    let host_values = (0..256u32)
        .map(|value| value.wrapping_mul(5).wrapping_add(17))
        .collect::<Vec<_>>();
    let device_bytes = stable_resource_u32_bytes(&device_values);
    let host_bytes = stable_resource_u32_bytes(&host_values);
    let mut transfer = device
        .create_resident_transfer_stream(2, 64 * 1024)
        .unwrap();
    let writes = [
        VulkanResidentBufferWriteRange::new(
            device_allocation.buffer(),
            device_allocation.buffer_byte_offset(),
            &device_bytes,
        )
        .unwrap(),
        VulkanResidentBufferWriteRange::new(
            host_allocation.buffer(),
            host_allocation.buffer_byte_offset(),
            &host_bytes,
        )
        .unwrap(),
    ];
    let ticket = transfer.submit(&writes).unwrap();
    transfer.wait(&ticket).unwrap();

    let mut table = VulkanStableResourceAddressTable::new(&device, &mut transfer, 2).unwrap();
    let device_publications = table
        .publish_tagged_group(
            &mut transfer,
            &[(0, Arc::clone(&device_allocation), 7)],
        )
        .unwrap();
    let host_publications = table
        .publish_tagged_group(&mut transfer, &[(1, Arc::clone(&host_allocation), 9)])
        .unwrap();
    assert_eq!(device_publications[0].representation(), 7);
    assert_eq!(host_publications[0].representation(), 9);
    assert_eq!(table.record(0).unwrap().representation, 7);
    assert_eq!(table.record(1).unwrap().representation, 9);
    let output = device.create_resident_buffer(16).unwrap();
    output.write_bytes(&[0; 16]).unwrap();
    let dispatch = device
        .create_resident_kernel_dispatch(
            &shader,
            &[
                VulkanResidentKernelBufferBinding::new(0, table.buffer(), table.byte_capacity())
                    .with_access(VulkanResidentKernelBufferAccess::Read),
                VulkanResidentKernelBufferBinding::new(1, &output, 16)
                    .with_access(VulkanResidentKernelBufferAccess::Write),
            ],
            1,
            1,
            4,
        )
        .unwrap();
    device
        .run_resident_kernel_dispatch(&dispatch, &0u32.to_le_bytes())
        .unwrap();
    assert_eq!(
        stable_resource_bytes_to_u32(&output.read_bytes(16).unwrap()),
        vec![11, device_values[255], 1, 1]
    );
    device
        .run_resident_kernel_dispatch(&dispatch, &1u32.to_le_bytes())
        .unwrap();
    assert_eq!(
        stable_resource_bytes_to_u32(&output.read_bytes(16).unwrap()),
        vec![17, host_values[255], 1, 1]
    );

    let resolved = table
        .allocations_for_publications(&device_publications)
        .unwrap();
    assert_eq!(resolved[0].allocation_id(), device_allocation.allocation_id());
    drop(resolved);
    let (exchanged_device_publications, exchanged_host_publications) = table
        .swap_groups(&mut transfer, &device_publications, &host_publications)
        .unwrap();
    assert_eq!(exchanged_device_publications[0].generation(), 2);
    assert_eq!(
        exchanged_device_publications[0].device_address(),
        host_allocation.device_address()
    );
    assert_eq!(exchanged_device_publications[0].representation(), 9);
    assert_eq!(table.record(0).unwrap().representation, 9);
    assert_eq!(exchanged_host_publications[0].generation(), 2);
    assert_eq!(
        exchanged_host_publications[0].device_address(),
        device_allocation.device_address()
    );
    assert_eq!(exchanged_host_publications[0].representation(), 7);
    assert_eq!(table.record(1).unwrap().representation, 7);
    assert!(table
        .swap_groups(
            &mut transfer,
            &device_publications,
            &exchanged_host_publications,
        )
        .is_err());
    assert!(table.clear_group(&mut transfer, &device_publications).is_err());

    device
        .run_resident_kernel_dispatch(&dispatch, &0u32.to_le_bytes())
        .unwrap();
    assert_eq!(
        stable_resource_bytes_to_u32(&output.read_bytes(16).unwrap()),
        vec![17, host_values[255], 1, 2]
    );
    device
        .run_resident_kernel_dispatch(&dispatch, &1u32.to_le_bytes())
        .unwrap();
    assert_eq!(
        stable_resource_bytes_to_u32(&output.read_bytes(16).unwrap()),
        vec![11, device_values[255], 1, 2]
    );

    table
        .clear_group(&mut transfer, &exchanged_device_publications)
        .unwrap();
    table
        .clear_group(&mut transfer, &exchanged_host_publications)
        .unwrap();
    drop(device_allocation);
    drop(host_allocation);
    device_arena.release_backing().unwrap();
    host_arena.release_backing().unwrap();
}

#[test]
fn stable_resource_group_atomically_replaces_variable_size_representation() {
    let Some(device_index) = stable_resource_test_device_index() else {
        eprintln!("skipping stable resource replacement test: explicit Vulkan device unset");
        return;
    };
    let device = VulkanComputeDevice::new_for_physical_device_index(device_index).unwrap();
    let source_arena = VulkanStableResourceArena::new(
        &device,
        VulkanStableResourceArenaConfig::new(64 * 1024, 256).unwrap(),
        &[VulkanStableResourceGroupLayout::Explicit {
            resource_slots: vec![0],
            resource_byte_counts: vec![1024],
        }],
    )
    .unwrap();
    let derived_arena = VulkanStableResourceArena::new(
        &device,
        VulkanStableResourceArenaConfig::new(64 * 1024, 256).unwrap(),
        &[VulkanStableResourceGroupLayout::Explicit {
            resource_slots: vec![0],
            resource_byte_counts: vec![2048],
        }],
    )
    .unwrap();
    let source = Arc::clone(
        &source_arena
            .allocate_groups(&device, &[(&[0], &[1024])], 256)
            .unwrap()[0][0],
    );
    let derived = Arc::clone(
        &derived_arena
            .allocate_groups(&device, &[(&[0], &[2048])], 256)
            .unwrap()[0][0],
    );
    let mut transfer = device
        .create_resident_transfer_stream(2, 64 * 1024)
        .unwrap();
    let mut table = VulkanStableResourceAddressTable::new(&device, &mut transfer, 1).unwrap();
    let source_publications = table
        .publish_tagged_group(&mut transfer, &[(0, Arc::clone(&source), 0)])
        .unwrap();

    let derived_publications = table
        .replace_group(
            &mut transfer,
            &source_publications,
            &[(0, Arc::clone(&derived), 1)],
        )
        .unwrap();
    assert_eq!(derived_publications[0].generation(), 2);
    assert_eq!(derived_publications[0].representation(), 1);
    assert_eq!(table.record(0).unwrap().byte_count, 2048);
    assert_eq!(table.record(0).unwrap().representation, 1);
    assert!(table.allocations_for_publications(&source_publications).is_err());

    let restored_publications = table
        .replace_group(
            &mut transfer,
            &derived_publications,
            &[(0, Arc::clone(&source), 0)],
        )
        .unwrap();
    assert_eq!(restored_publications[0].generation(), 3);
    assert_eq!(restored_publications[0].representation(), 0);
    assert_eq!(table.record(0).unwrap().byte_count, 1024);
    assert_eq!(table.record(0).unwrap().representation, 0);
    assert!(table.allocations_for_publications(&derived_publications).is_err());

    table
        .clear_group(&mut transfer, &restored_publications)
        .unwrap();
    drop(source);
    drop(derived);
    source_arena.release_backing().unwrap();
    derived_arena.release_backing().unwrap();
}

#[test]
fn dense_stable_resource_layout_packs_members_without_sparse_page_padding() {
    let (offsets, byte_capacity) = stable_group_member_layout(&[1024, 257, 2048], 256).unwrap();
    assert_eq!(offsets, vec![0, 1024, 1536]);
    assert_eq!(byte_capacity, 3584);
}

#[test]
fn partitioned_stable_resource_layout_resolves_member_order_per_request() {
    let layouts = VulkanStableResourceArenaLayouts {
        explicit: BTreeMap::new(),
        partitioned: vec![VulkanPartitionedStableResourcePlacement {
            member_slot_bases: vec![0, 10],
            resource_byte_offsets: vec![0, 256],
            resource_byte_counts: vec![8, 16],
            partition_count: 10,
            group_byte_capacity: 512,
        }],
        maximum_byte_capacity: 5120,
    };
    let placement = stable_resource_placement_for_slots(&layouts, &[17, 7], &[7, 17]).unwrap();
    assert_eq!(placement.resource_slots, vec![17, 7]);
    assert_eq!(placement.resource_byte_offsets, vec![256, 0]);
    assert_eq!(placement.resource_byte_counts, vec![16, 8]);
    assert_eq!(placement.group_byte_capacity, 512);
}

#[test]
fn dense_stable_resource_groups_allocate_on_demand_in_load_wave_chunks() {
    let Some(device_index) = stable_resource_test_device_index() else {
        eprintln!("skipping dense stable resource test: explicit Vulkan device unset");
        return;
    };
    let Some(shader) =
        compile_stable_resource_shader("dense_visibility", STABLE_RESOURCE_VISIBILITY_SHADER)
    else {
        eprintln!("skipping dense stable resource test: no GLSL compiler");
        return;
    };
    let device = VulkanComputeDevice::new_for_physical_device_index(device_index).unwrap();
    assert!(device.supports_buffer_device_address());
    let layouts = [
        VulkanStableResourceGroupLayout::Explicit {
            resource_slots: vec![0, 1],
            resource_byte_counts: vec![1024, 2048],
        },
        VulkanStableResourceGroupLayout::Explicit {
            resource_slots: vec![2],
            resource_byte_counts: vec![4096],
        },
    ];
    let arena = VulkanStableResourceArena::new(
        &device,
        VulkanStableResourceArenaConfig::new(512 * 1024, 256).unwrap(),
        &layouts,
    )
    .unwrap();
    assert_eq!(
        arena.stats().unwrap(),
        VulkanStableResourceArenaStats::default()
    );

    let later_group = arena
        .allocate_groups(&device, &[(&[2], &[4096])], 256)
        .unwrap()
        .pop()
        .unwrap();
    let earlier_group = arena
        .allocate_groups(&device, &[(&[0, 1], &[1024, 2048])], 256)
        .unwrap()
        .pop()
        .unwrap();
    assert_ne!(
        earlier_group[0].device_address(),
        later_group[0].device_address()
    );
    assert_eq!(earlier_group[0].device_address() % 256, 0);
    assert_eq!(earlier_group[1].device_address() % 256, 0);
    let stats = arena.stats().unwrap();
    assert_eq!(stats.allocated_byte_count, 7168);
    assert_eq!(stats.active_allocation_count, 3);
    assert_eq!(stats.chunk_count, 2);
    assert!(stats.committed_byte_capacity >= stats.allocated_byte_count);
    assert!(stats.committed_byte_capacity <= 512 * 1024);

    let first_values = (0..256u32)
        .map(|value| value.wrapping_mul(5).wrapping_add(13))
        .collect::<Vec<_>>();
    let first_bytes = stable_resource_u32_bytes(&first_values);
    let mut transfer = device
        .create_resident_transfer_stream(2, 64 * 1024)
        .unwrap();
    let ticket = transfer
        .submit(&[VulkanResidentBufferWriteRange::new(
            earlier_group[0].buffer(),
            earlier_group[0].buffer_byte_offset(),
            &first_bytes,
        )
        .unwrap()])
        .unwrap();
    transfer.wait(&ticket).unwrap();

    let mut table = VulkanStableResourceAddressTable::new(&device, &mut transfer, 3).unwrap();
    let publications = table
        .publish_group(&mut transfer, &[(0, Arc::clone(&earlier_group[0]))])
        .unwrap();
    let output = device.create_resident_buffer(16).unwrap();
    output.write_bytes(&[0; 16]).unwrap();
    let dispatch = device
        .create_resident_kernel_dispatch(
            &shader,
            &[
                VulkanResidentKernelBufferBinding::new(0, table.buffer(), table.byte_capacity())
                    .with_access(VulkanResidentKernelBufferAccess::Read),
                VulkanResidentKernelBufferBinding::new(1, &output, 16)
                    .with_access(VulkanResidentKernelBufferAccess::Write),
            ],
            1,
            1,
            4,
        )
        .unwrap();
    device
        .run_resident_kernel_dispatch(&dispatch, &0u32.to_le_bytes())
        .unwrap();
    assert_eq!(
        stable_resource_bytes_to_u32(&output.read_bytes(16).unwrap()),
        vec![13, first_values[255], 1, 1]
    );
    table.clear_group(&mut transfer, &publications).unwrap();
    drop(earlier_group);
    drop(later_group);
    assert_eq!(arena.stats().unwrap().active_allocation_count, 0);
    let retained_stats = arena.stats().unwrap();
    assert_eq!(retained_stats.chunk_count, 2);
    assert!(retained_stats.committed_byte_capacity >= 7168);
    assert_eq!(
        arena
            .additional_committed_byte_capacity_for_groups(
                &device,
                &[(&[0, 1], &[1024, 2048])],
                256,
            )
            .unwrap(),
        0,
    );
    let retry_group = arena
        .allocate_groups(&device, &[(&[0, 1], &[1024, 2048])], 256)
        .unwrap()
        .pop()
        .unwrap();
    let retry_stats = arena.stats().unwrap();
    assert_eq!(
        retry_stats.committed_byte_capacity,
        retained_stats.committed_byte_capacity
    );
    assert_eq!(retry_stats.chunk_count, 2);
    assert_eq!(retry_stats.allocated_byte_count, 3072);
    assert_eq!(retry_stats.active_allocation_count, 2);
    drop(retry_group);
    arena.release_backing().unwrap();
    assert_eq!(
        arena.stats().unwrap(),
        VulkanStableResourceArenaStats::default()
    );
}

#[test]
fn partition_layout_scales_without_reserving_one_million_gpu_ranges() {
    const PARTITION_COUNT: usize = 1_000_000;
    let Some(device_index) = stable_resource_test_device_index() else {
        eprintln!("skipping partition scale test: explicit Vulkan device unset");
        return;
    };
    let device = VulkanComputeDevice::new_for_physical_device_index(device_index).unwrap();
    let arena = VulkanStableResourceArena::new(
        &device,
        VulkanStableResourceArenaConfig::new(256 * 1024, 256).unwrap(),
        &[VulkanStableResourceGroupLayout::Partitioned {
            member_slot_bases: vec![0, PARTITION_COUNT],
            resource_byte_counts: vec![8, 16],
            partition_count: PARTITION_COUNT,
        }],
    )
    .unwrap();

    let first_slots = [PARTITION_COUNT, 0];
    let last_slots = [PARTITION_COUNT - 1, 2 * PARTITION_COUNT - 1];
    let first_byte_counts = [16, 8];
    let last_byte_counts = [8, 16];
    let groups = arena
        .allocate_groups(
            &device,
            &[
                (&first_slots, &first_byte_counts),
                (&last_slots, &last_byte_counts),
            ],
            256,
        )
        .unwrap();
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0].len(), 2);
    assert_eq!(groups[1].len(), 2);
    assert_eq!(groups[0][0].byte_count(), 16);
    assert_eq!(groups[0][1].byte_count(), 8);
    assert!(groups[1][0].buffer_byte_offset() > groups[0][1].buffer_byte_offset());
    let stats = arena.stats().unwrap();
    assert_eq!(stats.active_allocation_count, 4);
    assert_eq!(stats.allocated_byte_count, 48);
    assert_eq!(stats.chunk_count, 1);
    let maximum_backed = arena.maximum_backed_byte_capacity().unwrap();
    assert_eq!(maximum_backed % PARTITION_COUNT, 0);
    assert_eq!(maximum_backed, PARTITION_COUNT * 512);
    assert_eq!(stats.committed_byte_capacity, 1024);

    drop(groups);
    arena.release_backing().unwrap();
    assert_eq!(
        arena.stats().unwrap(),
        VulkanStableResourceArenaStats::default()
    );
}

#[test]
fn stable_resource_address_space_is_visible_stable_and_transactional() {
    let Some(device_index) = stable_resource_test_device_index() else {
        eprintln!("skipping stable resource address-space test: explicit Vulkan device unset");
        return;
    };
    let Some(shader) =
        compile_stable_resource_shader("visibility", STABLE_RESOURCE_VISIBILITY_SHADER)
    else {
        eprintln!("skipping stable resource address-space test: no GLSL compiler");
        return;
    };

    let device = VulkanComputeDevice::new_for_physical_device_index(device_index).unwrap();
    assert!(device.supports_buffer_device_address());
    let mut transfer = device
        .create_resident_transfer_stream(2, 64 * 1024)
        .unwrap();
    let arena = VulkanStableResourceArena::new(
        &device,
        VulkanStableResourceArenaConfig::new(32 * 1024, 8).unwrap(),
        &[
            VulkanStableResourceGroupLayout::Explicit {
                resource_slots: vec![1],
                resource_byte_counts: vec![1024],
            },
            VulkanStableResourceGroupLayout::Explicit {
                resource_slots: vec![3],
                resource_byte_counts: vec![2048],
            },
            VulkanStableResourceGroupLayout::Explicit {
                resource_slots: vec![2],
                resource_byte_counts: vec![4096],
            },
            VulkanStableResourceGroupLayout::Explicit {
                resource_slots: vec![0],
                resource_byte_counts: vec![128],
            },
        ],
    )
    .unwrap();
    assert_eq!(
        arena.stats().unwrap(),
        VulkanStableResourceArenaStats::default()
    );
    let mut undersized_transfer = device.create_resident_transfer_stream(1, 16).unwrap();
    assert!(VulkanStableResourceAddressTable::new(&device, &mut undersized_transfer, 1,).is_err());
    drop(undersized_transfer);
    let first = Arc::clone(
        &arena
            .allocate_groups(&device, &[(&[1], &[1024])], 8)
            .unwrap()[0][0],
    );
    let second = Arc::clone(
        &arena
            .allocate_groups(&device, &[(&[3], &[2048])], 8)
            .unwrap()[0][0],
    );
    assert_eq!(first.device_address() % 8, 0);
    assert_eq!(second.device_address() % 8, 0);
    assert_ne!(first.device_address(), second.device_address());
    let initial_stats = arena.stats().unwrap();
    assert_eq!(initial_stats.allocated_byte_count, 3072);
    assert_eq!(initial_stats.active_allocation_count, 2);
    assert_eq!(initial_stats.chunk_count, 2);
    assert!(initial_stats.committed_byte_capacity >= initial_stats.allocated_byte_count);

    let first_values = (0..256u32)
        .map(|value| value.wrapping_mul(3).wrapping_add(11))
        .collect::<Vec<_>>();
    let second_values = (0..512u32)
        .map(|value| value.wrapping_mul(7).wrapping_add(19))
        .collect::<Vec<_>>();
    let first_bytes = stable_resource_u32_bytes(&first_values);
    let second_bytes = stable_resource_u32_bytes(&second_values);
    let writes = [
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
    ];
    let ticket = transfer.submit(&writes).unwrap();
    transfer.wait(&ticket).unwrap();

    let mut table = VulkanStableResourceAddressTable::new(&device, &mut transfer, 4).unwrap();
    assert!(
        table
            .publish_group(&mut transfer, &[(4, Arc::clone(&first))])
            .is_err()
    );
    assert!(
        table
            .publish_group(
                &mut transfer,
                &[(1, Arc::clone(&first)), (1, Arc::clone(&second))],
            )
            .is_err()
    );
    assert!(
        table
            .records
            .iter()
            .all(|record| *record == VulkanStableResourceAddressRecord::default())
    );
    let publications = table
        .publish_group(
            &mut transfer,
            &[(1, Arc::clone(&first)), (3, Arc::clone(&second))],
        )
        .unwrap();
    assert_eq!(publications.len(), 2);
    assert_eq!(
        table.record(1).unwrap(),
        VulkanStableResourceAddressRecord {
            device_address: first.device_address(),
            byte_count: 1024,
            generation: 1,
            resident: 1,
            representation: 0,
        }
    );
    assert_eq!(table.record(3).unwrap().generation, 1);
    assert_eq!(table.record(0).unwrap().resident, 0);

    let first_address = first.device_address();
    let second_address = second.device_address();
    let third = Arc::clone(
        &arena
            .allocate_groups(&device, &[(&[2], &[4096])], 8)
            .unwrap()[0][0],
    );
    assert_eq!(third.device_address() % 1024, 0);
    assert_eq!(first.device_address(), first_address);
    assert_eq!(second.device_address(), second_address);
    let before_failure = arena.stats().unwrap();
    assert!(
        arena
            .allocate_groups(&device, &[(&[99], &[20 * 1024])], 8)
            .is_err()
    );
    assert_eq!(arena.stats().unwrap(), before_failure);

    let output = device.create_resident_buffer(16).unwrap();
    output.write_bytes(&[0; 16]).unwrap();
    let dispatch = device
        .create_resident_kernel_dispatch(
            &shader,
            &[
                VulkanResidentKernelBufferBinding::new(0, table.buffer(), table.byte_capacity())
                    .with_access(VulkanResidentKernelBufferAccess::Read),
                VulkanResidentKernelBufferBinding::new(1, &output, 16)
                    .with_access(VulkanResidentKernelBufferAccess::Write),
            ],
            1,
            1,
            4,
        )
        .unwrap();
    device
        .run_resident_kernel_dispatch(&dispatch, &1u32.to_le_bytes())
        .unwrap();
    assert_eq!(
        stable_resource_bytes_to_u32(&output.read_bytes(16).unwrap()),
        vec![11, first_values[255], 1, 1]
    );
    device
        .run_resident_kernel_dispatch(&dispatch, &3u32.to_le_bytes())
        .unwrap();
    assert_eq!(
        stable_resource_bytes_to_u32(&output.read_bytes(16).unwrap()),
        vec![19, second_values[511], 1, 1]
    );

    table.clear_group(&mut transfer, &publications).unwrap();
    assert_eq!(
        table.record(1).unwrap(),
        VulkanStableResourceAddressRecord {
            generation: 2,
            ..VulkanStableResourceAddressRecord::default()
        }
    );
    assert!(table.clear_group(&mut transfer, &publications).is_err());
    device
        .run_resident_kernel_dispatch(&dispatch, &1u32.to_le_bytes())
        .unwrap();
    assert_eq!(
        stable_resource_bytes_to_u32(&output.read_bytes(16).unwrap()),
        vec![0xdead_beef, 0, 0, 2]
    );

    let table_retained = Arc::clone(
        &arena
            .allocate_groups(&device, &[(&[0], &[128])], 8)
            .unwrap()[0][0],
    );
    let retained_publication = table
        .publish_group(&mut transfer, &[(0, Arc::clone(&table_retained))])
        .unwrap();
    let retained_stats = arena.stats().unwrap();
    drop(table_retained);
    assert_eq!(arena.stats().unwrap(), retained_stats);
    table
        .clear_group(&mut transfer, &retained_publication)
        .unwrap();
    assert_eq!(
        arena.stats().unwrap().active_allocation_count,
        retained_stats.active_allocation_count - 1
    );

    let republished = table
        .publish_group(&mut transfer, &[(1, Arc::clone(&first))])
        .unwrap();
    assert_eq!(republished[0].generation(), 3);
    assert_eq!(republished[0].device_address(), first_address);
    table.clear_group(&mut transfer, &republished).unwrap();

    drop(third);
    drop(second);
    drop(first);
    arena.release_backing().unwrap();
    assert_eq!(
        arena.stats().unwrap(),
        VulkanStableResourceArenaStats::default()
    );
}

#[test]
fn stable_resource_address_lookup_hot_path_is_measured_against_direct_binding() {
    let started = std::time::Instant::now();
    let Some(device_index) = stable_resource_test_device_index() else {
        eprintln!("skipping stable resource lookup microbenchmark: explicit Vulkan device unset");
        return;
    };
    let (Some(direct_shader), Some(table_shader)) = (
        compile_stable_resource_shader("direct_benchmark", STABLE_RESOURCE_DIRECT_BENCHMARK_SHADER),
        compile_stable_resource_shader("table_benchmark", STABLE_RESOURCE_TABLE_BENCHMARK_SHADER),
    ) else {
        eprintln!("skipping stable resource lookup microbenchmark: no GLSL compiler");
        return;
    };

    const ELEMENT_COUNT: usize = 256 * 1024;
    const BYTE_COUNT: usize = ELEMENT_COUNT * std::mem::size_of::<u32>();
    let device = VulkanComputeDevice::new_for_physical_device_index(device_index).unwrap();
    let mut transfer = device
        .create_resident_transfer_stream(2, BYTE_COUNT)
        .unwrap();
    let arena = VulkanStableResourceArena::new(
        &device,
        VulkanStableResourceArenaConfig::new(2 * BYTE_COUNT, 256).unwrap(),
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
    let ticket = transfer
        .submit(&[VulkanResidentBufferWriteRange::new(
            source.buffer(),
            source.buffer_byte_offset(),
            &source_bytes,
        )
        .unwrap()])
        .unwrap();
    transfer.wait(&ticket).unwrap();
    let mut table = VulkanStableResourceAddressTable::new(&device, &mut transfer, 1).unwrap();
    let publications = table
        .publish_group(&mut transfer, &[(0, Arc::clone(&source))])
        .unwrap();
    let direct_output = device.create_resident_buffer(BYTE_COUNT).unwrap();
    let table_output = device.create_resident_buffer(BYTE_COUNT).unwrap();
    direct_output.write_bytes(&vec![0; BYTE_COUNT]).unwrap();
    table_output.write_bytes(&vec![0; BYTE_COUNT]).unwrap();

    let workgroup_count = u32::try_from(ELEMENT_COUNT / 256).unwrap();
    let direct_dispatch = device
        .create_resident_kernel_dispatch(
            &direct_shader,
            &[
                VulkanResidentKernelBufferBinding::new(0, source.buffer(), BYTE_COUNT)
                    .with_byte_offset(source.buffer_byte_offset())
                    .with_access(VulkanResidentKernelBufferAccess::Read),
                VulkanResidentKernelBufferBinding::new(1, &direct_output, BYTE_COUNT)
                    .with_access(VulkanResidentKernelBufferAccess::Write),
            ],
            workgroup_count,
            256,
            4,
        )
        .unwrap();
    let table_dispatch = device
        .create_resident_kernel_dispatch(
            &table_shader,
            &[
                VulkanResidentKernelBufferBinding::new(0, table.buffer(), table.byte_capacity())
                    .with_access(VulkanResidentKernelBufferAccess::Read),
                VulkanResidentKernelBufferBinding::new(1, &table_output, BYTE_COUNT)
                    .with_access(VulkanResidentKernelBufferAccess::Write),
            ],
            workgroup_count,
            256,
            4,
        )
        .unwrap();
    let direct_sequence = device
        .create_timestamped_resident_kernel_sequence()
        .unwrap();
    let table_sequence = device
        .create_timestamped_resident_kernel_sequence()
        .unwrap();
    let element_count = u32::try_from(ELEMENT_COUNT).unwrap().to_le_bytes();
    device
        .record_resident_kernel_sequence(
            &direct_sequence,
            &[VulkanResidentKernelSequenceStep::new(
                &direct_dispatch,
                &element_count,
            )],
        )
        .unwrap();
    device
        .record_resident_kernel_sequence(
            &table_sequence,
            &[VulkanResidentKernelSequenceStep::new(
                &table_dispatch,
                &element_count,
            )],
        )
        .unwrap();

    let timeout = std::time::Duration::from_secs(5);
    device
        .run_timestamped_recorded_resident_kernel_sequence_for(&direct_sequence, timeout)
        .unwrap();
    device
        .run_timestamped_recorded_resident_kernel_sequence_for(&table_sequence, timeout)
        .unwrap();
    let mut direct_ns = Vec::with_capacity(2);
    let mut table_ns = Vec::with_capacity(2);
    for _ in 0..2 {
        direct_ns.push(
            device
                .run_timestamped_recorded_resident_kernel_sequence_for(&direct_sequence, timeout)
                .unwrap(),
        );
        table_ns.push(
            device
                .run_timestamped_recorded_resident_kernel_sequence_for(&table_sequence, timeout)
                .unwrap(),
        );
    }
    let direct_average_ns = direct_ns.iter().sum::<u64>() / 2;
    let table_average_ns = table_ns.iter().sum::<u64>() / 2;
    let ratio = table_average_ns as f64 / direct_average_ns as f64;
    eprintln!(
        "stable_resource_lookup_microbenchmark direct_ns={direct_ns:?} table_ns={table_ns:?} direct_average_ns={direct_average_ns} table_average_ns={table_average_ns} table_to_direct_ratio={ratio:.6} table_faster={}",
        table_average_ns < direct_average_ns
    );
    assert_eq!(
        direct_output.read_bytes(BYTE_COUNT).unwrap(),
        table_output.read_bytes(BYTE_COUNT).unwrap()
    );
    assert!(
        started.elapsed() < std::time::Duration::from_secs(60),
        "stable resource lookup microbenchmark exceeded one minute"
    );
    table.clear_group(&mut transfer, &publications).unwrap();
    drop(source);
    assert_eq!(arena.stats().unwrap().active_allocation_count, 0);
    arena.release_backing().unwrap();
    assert_eq!(
        arena.stats().unwrap(),
        VulkanStableResourceArenaStats::default()
    );
}

fn stable_resource_test_device_index() -> Option<usize> {
    std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
}

fn compile_stable_resource_shader(label: &str, source: &str) -> Option<Vec<u32>> {
    let source_path = std::env::temp_dir().join(format!(
        "nerve-stable-resource-{label}-{}.comp",
        std::process::id()
    ));
    std::fs::write(&source_path, source).ok()?;
    let words = compile_shader_words_from_source_path(&source_path);
    let _ = std::fs::remove_file(source_path);
    words
}

fn stable_resource_u32_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn stable_resource_bytes_to_u32(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(std::mem::size_of::<u32>())
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

const STABLE_RESOURCE_VISIBILITY_SHADER: &str =
    include_str!("../../tests/fixtures/vulkan/stable_resource_visibility.comp");
const STABLE_RESOURCE_DIRECT_BENCHMARK_SHADER: &str =
    include_str!("../../tests/fixtures/vulkan/stable_resource_direct_benchmark.comp");
const STABLE_RESOURCE_TABLE_BENCHMARK_SHADER: &str =
    include_str!("../../tests/fixtures/vulkan/stable_resource_table_benchmark.comp");
