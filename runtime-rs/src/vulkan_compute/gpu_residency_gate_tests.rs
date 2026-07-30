const GPU_RESIDENCY_GATE_SHADER: &str = "shaders/gpu_residency_gate.comp";
const GPU_RESIDENCY_GATE_DOWNSTREAM_SHADER: &str =
    "tests/fixtures/vulkan/gpu_residency_gate_downstream.comp";

#[test]
fn gpu_residency_gate_contract_rejects_unrepresentable_or_unbounded_work() {
    let valid = VulkanGpuResidencyGateConfig {
        maximum_selection_count: 2,
        selection_index_shift: 0,
        selection_index_mask: 0xffff,
        address_slots_by_resource_index: vec![vec![0, 1], vec![2, 3]],
        missing_request_capacity: 4,
        downstream_dispatches: vec![VulkanGpuResidencyIndirectDispatch {
            byte_offset: VULKAN_RESIDENT_INDIRECT_DISPATCH_BYTE_COUNT,
            dimensions: [2, 1, 1],
        }],
    };
    assert!(
        valid
            .validate(
                8,
                4,
                2 * VULKAN_RESIDENT_INDIRECT_DISPATCH_BYTE_COUNT,
            )
            .is_ok()
    );

    let mut invalid = valid.clone();
    invalid.maximum_selection_count = 0;
    assert!(
        invalid
            .validate(8, 4, 2 * VULKAN_RESIDENT_INDIRECT_DISPATCH_BYTE_COUNT)
            .is_err()
    );

    let mut invalid = valid.clone();
    invalid.selection_index_mask = 0;
    assert!(
        invalid
            .validate(8, 4, 2 * VULKAN_RESIDENT_INDIRECT_DISPATCH_BYTE_COUNT)
            .is_err()
    );

    let mut invalid = valid.clone();
    invalid.selection_index_mask = 0x1;
    invalid.address_slots_by_resource_index =
        vec![vec![0], vec![1], vec![2]];
    assert!(
        invalid
            .validate(8, 4, 2 * VULKAN_RESIDENT_INDIRECT_DISPATCH_BYTE_COUNT)
            .is_err()
    );

    let mut invalid = valid.clone();
    invalid.address_slots_by_resource_index[0].push(1);
    assert!(
        invalid
            .validate(8, 4, 2 * VULKAN_RESIDENT_INDIRECT_DISPATCH_BYTE_COUNT)
            .is_err()
    );

    let mut invalid = valid.clone();
    invalid.address_slots_by_resource_index[0].push(4);
    assert!(
        invalid
            .validate(8, 4, 2 * VULKAN_RESIDENT_INDIRECT_DISPATCH_BYTE_COUNT)
            .is_err()
    );

    let mut invalid = valid.clone();
    invalid.missing_request_capacity = 1;
    assert!(
        invalid
            .validate(8, 4, 2 * VULKAN_RESIDENT_INDIRECT_DISPATCH_BYTE_COUNT)
            .is_err()
    );

    let mut invalid = valid.clone();
    invalid.downstream_dispatches[0].dimensions[1] = 0;
    assert!(
        invalid
            .validate(8, 4, 2 * VULKAN_RESIDENT_INDIRECT_DISPATCH_BYTE_COUNT)
            .is_err()
    );

    assert!(
        valid
            .validate(4, 4, 2 * VULKAN_RESIDENT_INDIRECT_DISPATCH_BYTE_COUNT)
            .is_err()
    );
    assert!(
        valid
            .validate(8, 4, VULKAN_RESIDENT_INDIRECT_DISPATCH_BYTE_COUNT)
            .is_err()
    );
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
        compile_gpu_residency_gate_shader(GPU_RESIDENCY_GATE_SHADER)
            .expect("GPU residency gate test requires a GLSL compiler");
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
        VulkanStableResourceArenaConfig::new(4096, 8192, 256).unwrap(),
    )
    .unwrap();
    let first = Arc::new(arena.allocate(&device, 64, 256).unwrap());
    let second = Arc::new(arena.allocate(&device, 64, 256).unwrap());
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

    let selections = Arc::new(device.create_resident_buffer(8).unwrap());
    selections
        .write_bytes(&u32_words_bytes(&[0, 1]))
        .unwrap();
    let indirect_dispatches = Arc::new(
        device
            .create_resident_buffer(VULKAN_RESIDENT_INDIRECT_DISPATCH_BYTE_COUNT)
            .unwrap(),
    );
    indirect_dispatches
        .write_bytes(&vec![0; indirect_dispatches.byte_capacity()])
        .unwrap();
    let gate = VulkanGpuResidencyGate::new(
        &device,
        &gate_shader,
        Arc::clone(&selections),
        table.shared_buffer(),
        table.slot_count(),
        Arc::clone(&indirect_dispatches),
        VulkanGpuResidencyGateConfig {
            maximum_selection_count: 2,
            selection_index_shift: 0,
            selection_index_mask: 0xffff,
            address_slots_by_resource_index: vec![vec![0], vec![1]],
            missing_request_capacity: 4,
            downstream_dispatches: vec![VulkanGpuResidencyIndirectDispatch {
                byte_offset: 0,
                dimensions: [1, 1, 1],
            }],
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
        2,
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
    let resolved =
        stable_resource_bytes_to_u32(&gate.resolved_addresses_buffer().read_bytes(
            gate.resolved_addresses_buffer().byte_capacity(),
        ).unwrap());
    assert_eq!(&resolved[..6], &[1, 2, 2, 0, 0, 41]);
    assert_eq!(
        u64::from(resolved[10]) | (u64::from(resolved[11]) << 32),
        first.device_address()
    );
    assert_eq!(
        u64::from(resolved[18]) | (u64::from(resolved[19]) << 32),
        second.device_address()
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

    drop(sequence);
    drop(downstream);
    drop(gate);
    table
        .clear_group(&mut transfer, &second_republication)
        .unwrap();
    table
        .clear_group(&mut transfer, &first_publication)
        .unwrap();
    drop(table);
    drop(second);
    drop(first);
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
        VulkanStableResourceArenaConfig::new(4096, 8192, 256).unwrap(),
    )
    .unwrap();
    let first = Arc::new(arena.allocate(&device, 64, 256).unwrap());
    let second = Arc::new(arena.allocate(&device, 64, 256).unwrap());
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
    let indirect_dispatches = Arc::new(
        device
            .create_resident_buffer(3 * VULKAN_RESIDENT_INDIRECT_DISPATCH_BYTE_COUNT)
            .unwrap(),
    );
    indirect_dispatches
        .write_bytes(&vec![0; indirect_dispatches.byte_capacity()])
        .unwrap();
    let first_gate = VulkanGpuResidencyGate::new(
        &device,
        &gate_shader,
        selection_first,
        table.shared_buffer(),
        table.slot_count(),
        Arc::clone(&indirect_dispatches),
        VulkanGpuResidencyGateConfig {
            maximum_selection_count: 1,
            selection_index_shift: 0,
            selection_index_mask: u32::MAX,
            address_slots_by_resource_index: vec![vec![0]],
            missing_request_capacity: 1,
            downstream_dispatches: vec![
                VulkanGpuResidencyIndirectDispatch {
                    byte_offset: 0,
                    dimensions: [1, 1, 1],
                },
                VulkanGpuResidencyIndirectDispatch {
                    byte_offset: VULKAN_RESIDENT_INDIRECT_DISPATCH_BYTE_COUNT,
                    dimensions: [1, 1, 1],
                },
                VulkanGpuResidencyIndirectDispatch {
                    byte_offset: 2 * VULKAN_RESIDENT_INDIRECT_DISPATCH_BYTE_COUNT,
                    dimensions: [1, 1, 1],
                },
            ],
        },
    )
    .unwrap();
    let second_gate = VulkanGpuResidencyGate::new(
        &device,
        &gate_shader,
        selection_second,
        table.shared_buffer(),
        table.slot_count(),
        Arc::clone(&indirect_dispatches),
        VulkanGpuResidencyGateConfig {
            maximum_selection_count: 1,
            selection_index_shift: 0,
            selection_index_mask: u32::MAX,
            address_slots_by_resource_index: vec![vec![1]],
            missing_request_capacity: 1,
            downstream_dispatches: vec![VulkanGpuResidencyIndirectDispatch {
                byte_offset: 2 * VULKAN_RESIDENT_INDIRECT_DISPATCH_BYTE_COUNT,
                dimensions: [1, 1, 1],
            }],
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
    let first_control = first_gate.push_constants(1, 11).unwrap();
    let second_control = second_gate.push_constants(1, 22).unwrap();
    let increment = 1u32.to_le_bytes();
    device
        .record_resident_kernel_sequence(
            &full_sequence,
            &[
                VulkanResidentKernelSequenceStep::new(first_gate.dispatch(), &first_control),
                first_gate
                    .indirect_dispatch_step(0, &first_compute, &increment)
                    .unwrap(),
                VulkanResidentKernelSequenceStep::new_indirect(
                    second_gate.dispatch(),
                    &second_control,
                    &indirect_dispatches,
                    VULKAN_RESIDENT_INDIRECT_DISPATCH_BYTE_COUNT,
                )
                .unwrap(),
                second_gate
                    .indirect_dispatch_step(
                        2 * VULKAN_RESIDENT_INDIRECT_DISPATCH_BYTE_COUNT,
                        &second_compute,
                        &increment,
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
                second_gate
                    .indirect_dispatch_step(
                        2 * VULKAN_RESIDENT_INDIRECT_DISPATCH_BYTE_COUNT,
                        &second_compute,
                        &increment,
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
    assert!(second_gate.missing_snapshot().unwrap().requests.is_empty());
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
    drop(indirect_dispatches);
    table
        .clear_group(&mut transfer, &second_publication)
        .unwrap();
    table
        .clear_group(&mut transfer, &first_publication)
        .unwrap();
    drop(table);
    drop(second);
    drop(first);
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
            BYTE_COUNT + 256,
            256,
        )
        .unwrap(),
    )
    .unwrap();
    let source =
        Arc::new(arena.allocate(&device, BYTE_COUNT, 256).unwrap());
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
    let gate = VulkanGpuResidencyGate::new(
        &device,
        &gate_shader,
        Arc::clone(&selection),
        table.shared_buffer(),
        table.slot_count(),
        Arc::new(
            device
                .create_resident_buffer(VULKAN_RESIDENT_INDIRECT_DISPATCH_BYTE_COUNT)
                .unwrap(),
        ),
        VulkanGpuResidencyGateConfig {
            maximum_selection_count: 1,
            selection_index_shift: 0,
            selection_index_mask: u32::MAX,
            address_slots_by_resource_index: vec![vec![0]],
            missing_request_capacity: 1,
            downstream_dispatches: vec![VulkanGpuResidencyIndirectDispatch {
                byte_offset: 0,
                dimensions: [workgroup_count, 1, 1],
            }],
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
    let gate_control = gate.push_constants(1, 7).unwrap();
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
                gate.indirect_dispatch_step(
                    0,
                    &demand_dispatch,
                    &element_count,
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
    assert_eq!(arena.stats().unwrap(), VulkanStableResourceArenaStats::default());
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
        .push_constants(selection_count, checkpoint_tag)
        .unwrap();
    device
        .record_resident_kernel_sequence(
            sequence,
            &[
                VulkanResidentKernelSequenceStep::new(
                    gate.dispatch(),
                    &gate_control,
                ),
                gate.indirect_dispatch_step(
                    0,
                    downstream,
                    downstream_push_constants,
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
