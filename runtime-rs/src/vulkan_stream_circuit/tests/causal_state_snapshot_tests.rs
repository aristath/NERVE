fn static_state_test_buffers(
    device: &VulkanComputeDevice,
) -> VulkanStreamCircuitStreamBuffers {
    let plan = VulkanResidentStateBuffer {
        component_id: "component".to_string(),
        state_id: "recurrent".to_string(),
        state_type: "recurrent".to_string(),
        dtype: Some("U32".to_string()),
        layout: Some("flat".to_string()),
        static_elements: Some(4),
        elements_per_activation: None,
        max_dynamic_activations: None,
        static_bytes: Some(4 * std::mem::size_of::<u32>()),
        bytes_per_activation: None,
        clone_from: None,
    };
    let layout = VulkanTransientStateBufferLayout::for_state(&plan, 1).unwrap();
    let buffer = device.create_resident_buffer(layout.byte_capacity).unwrap();
    let mut initial = vec![0; layout.byte_capacity];
    let page_table = layout.initial_page_table_bytes().unwrap();
    initial[..page_table.len()].copy_from_slice(&page_table);
    buffer.write_bytes(&initial).unwrap();
    VulkanStreamCircuitStreamBuffers {
        dynamic_state_capacity_activations: 1,
        total_byte_capacity: layout.byte_capacity,
        state_buffers: vec![VulkanStreamStateBufferAllocation {
            component_id: plan.component_id,
            state_id: plan.state_id,
            state_type: plan.state_type,
            dtype: plan.dtype,
            byte_capacity: layout.byte_capacity,
            layout,
            static_byte_capacity: plan.static_bytes,
            bytes_per_activation: None,
            clone_from: None,
            buffer,
        }],
        selection_telemetry_buffers: Vec::new(),
        activation_slot_buffers: Vec::new(),
    }
}

#[test]
fn causal_state_snapshot_bank_commits_the_selected_prefix_without_replay() {
    let device = selected_test_vulkan_device().expect("selected Vulkan test device must open");
    let buffers = static_state_test_buffers(&device);
    let mut bank = VulkanCausalStateSnapshotBank::new(&device, 3, true).unwrap();
    let snapshot_bytes = (0u32..12)
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    bank.binding_buffer(&device, &buffers, 0)
        .unwrap()
        .write_bytes(&snapshot_bytes)
        .unwrap();
    bank.mount_commit_batches(&device, &buffers).unwrap();

    assert!(bank.commit_prefix(2).unwrap());
    let state = &buffers.state_buffers[0];
    let bytes = state.buffer.read_bytes(state.byte_capacity).unwrap();
    let committed = bytes
        [state.layout.static_data_offset..state.layout.static_data_offset + 16]
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(committed, [4, 5, 6, 7]);
}

#[test]
fn causal_state_snapshot_bank_fails_closed_when_capture_is_disabled() {
    let device = selected_test_vulkan_device().expect("selected Vulkan test device must open");
    let buffers = static_state_test_buffers(&device);
    let mut bank = VulkanCausalStateSnapshotBank::new(&device, 3, false).unwrap();
    bank.binding_buffer(&device, &buffers, 0).unwrap();
    bank.mount_commit_batches(&device, &buffers).unwrap();

    assert!(!bank.commit_prefix(2).unwrap());
}

#[test]
fn causal_state_snapshot_bank_initializes_and_digests_every_lane() {
    let device = selected_test_vulkan_device().expect("selected Vulkan test device must open");
    let buffers = static_state_test_buffers(&device);
    let state = &buffers.state_buffers[0];
    let source = [11u32, 22, 33, 44]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    state
        .buffer
        .write_bytes_at(state.layout.static_data_offset, &source)
        .unwrap();
    let mut bank = VulkanCausalStateSnapshotBank::new(&device, 3, true).unwrap();
    bank.binding_buffer(&device, &buffers, 0).unwrap();
    bank.mount_commit_batches(&device, &buffers).unwrap();
    bank.initialize_from_state_buffers(&buffers).unwrap();

    let snapshots = bank.entries[0]
        .snapshots
        .read_bytes(bank.entries[0].snapshots.byte_capacity())
        .unwrap();
    assert_eq!(snapshots, source.repeat(3));
    let mut first = Sha256::new();
    bank.update_digest(&buffers, &mut first).unwrap();
    let first = first.finalize().to_vec();
    bank.entries[0].snapshots.write_bytes(&vec![0; snapshots.len()]).unwrap();
    let mut second = Sha256::new();
    bank.update_digest(&buffers, &mut second).unwrap();
    assert_ne!(first, second.finalize().to_vec());
}

#[test]
fn temporal_causal_convolution_matches_repeated_scalar_dispatches() {
    const CHANNELS: usize = 10_240;
    const KERNEL_WIDTH: usize = 4;
    const WIDTH: usize = 4;
    const FRAME_BYTES: usize = CHANNELS * std::mem::size_of::<u16>();
    const STATE_BYTES: usize = CHANNELS * KERNEL_WIDTH * std::mem::size_of::<u16>();

    let Some(manifest_path) = std::env::var_os("NERVE_TEMPORAL_TEST_PACKAGE").map(PathBuf::from)
    else {
        eprintln!("skipping temporal convolution equivalence: NERVE_TEMPORAL_TEST_PACKAGE is unset");
        return;
    };
    let device = selected_test_vulkan_device().expect("selected Vulkan test device must open");
    let package_dir = manifest_path.parent().unwrap();
    let scalar_words = read_spirv_words(
        package_dir.join("shaders/causal_conv1d_silu_bf16_c10240_k4.spv"),
    )
    .unwrap();
    let temporal_words = read_spirv_words(
        package_dir.join(
            "shaders/causal_conv1d_silu_temporal_bf16_c10240_k4__pbc31.spv",
        ),
    )
    .unwrap();

    let pack_bf16 = |values: &[f32]| {
        values
            .chunks(2)
            .flat_map(|pair| {
                let encode = |value: f32| {
                    let bits = value.to_bits();
                    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
                };
                let low = u32::from(encode(pair[0]));
                let high = u32::from(pair.get(1).copied().map(encode).unwrap_or_default());
                (low | (high << 16)).to_le_bytes()
            })
            .collect::<Vec<_>>()
    };
    let input_values = (0..WIDTH * CHANNELS)
        .map(|index| {
            let position = index / CHANNELS;
            let channel = index % CHANNELS;
            ((position * 13 + channel % 19) as f32 - 9.0) * 0.015625
        })
        .collect::<Vec<_>>();
    let kernel_values = (0..CHANNELS * KERNEL_WIDTH)
        .map(|index| {
            let tap = index % KERNEL_WIDTH;
            let channel = index / KERNEL_WIDTH;
            ((tap * 5 + channel % 7) as f32 - 8.0) * 0.03125
        })
        .collect::<Vec<_>>();
    let state_values = (0..CHANNELS * KERNEL_WIDTH)
        .map(|index| {
            let tap = index % KERNEL_WIDTH;
            let channel = index / KERNEL_WIDTH;
            ((tap * 3 + channel % 11) as f32 - 7.0) * 0.015625
        })
        .collect::<Vec<_>>();
    let input_bytes = pack_bf16(&input_values);
    let kernel_bytes = pack_bf16(&kernel_values);
    let state_bytes = pack_bf16(&state_values);

    let new_state = || {
        let plan = VulkanResidentStateBuffer {
            component_id: "component".to_string(),
            state_id: "conv".to_string(),
            state_type: "rolling_channel_memory".to_string(),
            dtype: Some("BF16".to_string()),
            layout: Some("channel_time".to_string()),
            static_elements: Some(CHANNELS * KERNEL_WIDTH),
            elements_per_activation: None,
            max_dynamic_activations: None,
            static_bytes: Some(STATE_BYTES),
            bytes_per_activation: None,
            clone_from: None,
        };
        let layout = VulkanTransientStateBufferLayout::for_state(&plan, 1).unwrap();
        let buffer = device.create_resident_buffer(layout.byte_capacity).unwrap();
        let mut initial = vec![0; layout.byte_capacity];
        let page_table = layout.initial_page_table_bytes().unwrap();
        initial[..page_table.len()].copy_from_slice(&page_table);
        initial[layout.static_data_offset..layout.static_data_offset + STATE_BYTES]
            .copy_from_slice(&state_bytes);
        buffer.write_bytes(&initial).unwrap();
        (buffer, layout)
    };
    let (scalar_state, scalar_layout) = new_state();
    let (temporal_state, temporal_layout) = new_state();
    assert_eq!(scalar_layout, temporal_layout);

    let kernel = device.create_resident_buffer(kernel_bytes.len()).unwrap();
    kernel.write_bytes(&kernel_bytes).unwrap();
    let scalar_input = device.create_resident_buffer(FRAME_BYTES).unwrap();
    let scalar_output = device.create_resident_buffer(FRAME_BYTES).unwrap();
    let scalar_bindings = [
        VulkanResidentKernelBufferBinding::new(0, &scalar_input, FRAME_BYTES)
            .with_access(VulkanResidentKernelBufferAccess::Read),
        VulkanResidentKernelBufferBinding::new(1, &scalar_output, FRAME_BYTES)
            .with_access(VulkanResidentKernelBufferAccess::Write),
        VulkanResidentKernelBufferBinding::new(2, &kernel, kernel_bytes.len())
            .with_access(VulkanResidentKernelBufferAccess::Read),
        VulkanResidentKernelBufferBinding::new(3, &scalar_state, scalar_layout.byte_capacity)
            .with_access(VulkanResidentKernelBufferAccess::Read),
        VulkanResidentKernelBufferBinding::new(4, &scalar_state, scalar_layout.byte_capacity)
            .with_access(VulkanResidentKernelBufferAccess::Write),
    ];
    let scalar_dispatch = device
        .create_resident_kernel_dispatch(&scalar_words, &scalar_bindings, 1, 64, 0)
        .unwrap();
    let mut scalar_outputs = Vec::with_capacity(WIDTH * FRAME_BYTES);
    for position in 0..WIDTH {
        scalar_input
            .write_bytes(&input_bytes[position * FRAME_BYTES..(position + 1) * FRAME_BYTES])
            .unwrap();
        device
            .run_resident_kernel_dispatch(&scalar_dispatch, &[])
            .unwrap();
        scalar_outputs.extend(scalar_output.read_bytes(FRAME_BYTES).unwrap());
    }

    let temporal_input = device
        .create_resident_buffer(WIDTH * FRAME_BYTES)
        .unwrap();
    temporal_input.write_bytes(&input_bytes).unwrap();
    let temporal_output = device
        .create_resident_buffer(WIDTH * FRAME_BYTES)
        .unwrap();
    let snapshots = device
        .create_resident_buffer(WIDTH * STATE_BYTES)
        .unwrap();
    let control = device.create_host_visible_resident_buffer(8).unwrap();
    control
        .write_bytes(
            &[
                u32::try_from(WIDTH).unwrap().to_le_bytes(),
                0u32.to_le_bytes(),
            ]
            .concat(),
        )
        .unwrap();
    let temporal_bindings = [
        VulkanResidentKernelBufferBinding::new(0, &temporal_input, WIDTH * FRAME_BYTES)
            .with_access(VulkanResidentKernelBufferAccess::Read),
        VulkanResidentKernelBufferBinding::new(1, &temporal_output, WIDTH * FRAME_BYTES)
            .with_access(VulkanResidentKernelBufferAccess::Write),
        VulkanResidentKernelBufferBinding::new(2, &kernel, kernel_bytes.len())
            .with_access(VulkanResidentKernelBufferAccess::Read),
        VulkanResidentKernelBufferBinding::new(3, &temporal_state, temporal_layout.byte_capacity)
            .with_access(VulkanResidentKernelBufferAccess::Read),
        VulkanResidentKernelBufferBinding::new(4, &temporal_state, temporal_layout.byte_capacity)
            .with_access(VulkanResidentKernelBufferAccess::Write),
        VulkanResidentKernelBufferBinding::new(30, &snapshots, WIDTH * STATE_BYTES)
            .with_access(VulkanResidentKernelBufferAccess::Write),
        VulkanResidentKernelBufferBinding::new(31, &control, 8)
            .with_access(VulkanResidentKernelBufferAccess::Read),
    ];
    let temporal_dispatch = device
        .create_resident_kernel_dispatch(&temporal_words, &temporal_bindings, 80, 64, 0)
        .unwrap();
    device
        .run_resident_kernel_dispatch(&temporal_dispatch, &[])
        .unwrap();

    let temporal_outputs = temporal_output.read_bytes(WIDTH * FRAME_BYTES).unwrap();
    let first_output_mismatch = scalar_outputs
        .iter()
        .zip(&temporal_outputs)
        .position(|(scalar, temporal)| scalar != temporal);
    assert_eq!(
        first_output_mismatch, None,
        "temporal convolution output first differs at byte {first_output_mismatch:?}"
    );
    let scalar_state_all = scalar_state
        .read_bytes(scalar_layout.byte_capacity)
        .unwrap();
    let temporal_state_all = temporal_state
        .read_bytes(temporal_layout.byte_capacity)
        .unwrap();
    let scalar_state_bytes =
        &scalar_state_all[scalar_layout.static_data_offset..scalar_layout.static_data_offset + STATE_BYTES];
    let temporal_state_bytes =
        &temporal_state_all[temporal_layout.static_data_offset..temporal_layout.static_data_offset + STATE_BYTES];
    let first_state_mismatch = scalar_state_bytes
        .iter()
        .zip(temporal_state_bytes)
        .position(|(scalar, temporal)| scalar != temporal);
    assert_eq!(
        first_state_mismatch, None,
        "temporal convolution state first differs at byte {first_state_mismatch:?}"
    );
}

#[test]
fn batched_rms_norm_matches_repeated_scalar_dispatches() {
    const HIDDEN_SIZE: usize = 5_120;
    const BLOCK_COLUMNS: usize = 128;
    const WIDTH: usize = 4;
    const FRAME_BYTES: usize = HIDDEN_SIZE * std::mem::size_of::<u16>();
    const QUANTIZED_FRAME_BYTES: usize = HIDDEN_SIZE;
    const SCALE_FRAME_BYTES: usize =
        (HIDDEN_SIZE / BLOCK_COLUMNS) * std::mem::size_of::<f32>();

    let Some(manifest_path) = std::env::var_os("NERVE_TEMPORAL_TEST_PACKAGE").map(PathBuf::from)
    else {
        eprintln!("skipping batched RMS norm equivalence: NERVE_TEMPORAL_TEST_PACKAGE is unset");
        return;
    };
    let device = selected_test_vulkan_device().expect("selected Vulkan test device must open");
    let package_dir = manifest_path.parent().unwrap();
    let scalar_words = read_spirv_words(
        package_dir
            .join("shaders/rms_norm_quantize_fp8_e4m3_b128_h5120_eps1e-06_offset1.spv"),
    )
    .unwrap();
    let batch_words = read_spirv_words(
        package_dir.join(
            "shaders/rms_norm_quantize_batch4_fp8_e4m3_b128_h5120_eps1e-06_offset1__pbc31.spv",
        ),
    )
    .unwrap();
    let tensor_index =
        TensorIndex::from_package_json_file(package_dir.join("tensors.json")).unwrap();
    let embedding_storage = crate::tensor_storage::TensorStorage::from_index(
        &tensor_index,
        "model.language_model.embed_tokens.weight",
    )
    .unwrap();
    let mut embedding_file = std::fs::File::open(&embedding_storage.source_file).unwrap();
    let mut header_len_bytes = [0u8; 8];
    embedding_file.read_exact(&mut header_len_bytes).unwrap();
    let tensor_data_start = 8 + u64::from_le_bytes(header_len_bytes);
    let mut input_bytes = Vec::with_capacity(WIDTH * FRAME_BYTES);
    for token_id in [0usize, 1, 100, 1_000] {
        embedding_file
            .seek(SeekFrom::Start(
                tensor_data_start
                    + u64::try_from(embedding_storage.data_start + token_id * FRAME_BYTES)
                        .unwrap(),
            ))
            .unwrap();
        let mut frame = vec![0; FRAME_BYTES];
        embedding_file.read_exact(&mut frame).unwrap();
        input_bytes.extend(frame);
    }
    let weight_bytes = crate::tensor_storage::TensorStorage::from_index(
        &tensor_index,
        "model.language_model.layers.0.input_layernorm.weight",
    )
    .unwrap()
    .read_all()
    .unwrap();
    let weight = device.create_resident_buffer(weight_bytes.len()).unwrap();
    weight.write_bytes(&weight_bytes).unwrap();

    let scalar_input = device.create_resident_buffer(FRAME_BYTES).unwrap();
    let scalar_output = device.create_resident_buffer(FRAME_BYTES).unwrap();
    let scalar_quantized = device
        .create_resident_buffer(QUANTIZED_FRAME_BYTES)
        .unwrap();
    let scalar_scales = device.create_resident_buffer(SCALE_FRAME_BYTES).unwrap();
    let scalar_bindings = [
        VulkanResidentKernelBufferBinding::new(0, &scalar_input, FRAME_BYTES)
            .with_access(VulkanResidentKernelBufferAccess::Read),
        VulkanResidentKernelBufferBinding::new(1, &scalar_output, FRAME_BYTES)
            .with_access(VulkanResidentKernelBufferAccess::Write),
        VulkanResidentKernelBufferBinding::new(2, &scalar_quantized, QUANTIZED_FRAME_BYTES)
            .with_access(VulkanResidentKernelBufferAccess::Write),
        VulkanResidentKernelBufferBinding::new(3, &scalar_scales, SCALE_FRAME_BYTES)
            .with_access(VulkanResidentKernelBufferAccess::Write),
        VulkanResidentKernelBufferBinding::new(4, &weight, weight_bytes.len())
            .with_access(VulkanResidentKernelBufferAccess::Read),
    ];
    let scalar_dispatch = device
        .create_resident_kernel_dispatch(&scalar_words, &scalar_bindings, 1, 1_024, 0)
        .unwrap();
    let mut scalar_outputs = Vec::with_capacity(WIDTH * FRAME_BYTES);
    let mut scalar_quantized_outputs = Vec::with_capacity(WIDTH * QUANTIZED_FRAME_BYTES);
    let mut scalar_scale_outputs = Vec::with_capacity(WIDTH * SCALE_FRAME_BYTES);
    for position in 0..WIDTH {
        scalar_input
            .write_bytes(&input_bytes[position * FRAME_BYTES..(position + 1) * FRAME_BYTES])
            .unwrap();
        device
            .run_resident_kernel_dispatch(&scalar_dispatch, &[])
            .unwrap();
        scalar_outputs.extend(scalar_output.read_bytes(FRAME_BYTES).unwrap());
        scalar_quantized_outputs.extend(
            scalar_quantized
                .read_bytes(QUANTIZED_FRAME_BYTES)
                .unwrap(),
        );
        scalar_scale_outputs.extend(scalar_scales.read_bytes(SCALE_FRAME_BYTES).unwrap());
    }

    let batch_input = device
        .create_resident_buffer(WIDTH * FRAME_BYTES)
        .unwrap();
    batch_input.write_bytes(&input_bytes).unwrap();
    let batch_output = device
        .create_resident_buffer(WIDTH * FRAME_BYTES)
        .unwrap();
    let batch_quantized = device
        .create_resident_buffer(WIDTH * QUANTIZED_FRAME_BYTES)
        .unwrap();
    let batch_scales = device
        .create_resident_buffer(WIDTH * SCALE_FRAME_BYTES)
        .unwrap();
    let control = device.create_host_visible_resident_buffer(4).unwrap();
    control
        .write_bytes(&u32::try_from(WIDTH).unwrap().to_le_bytes())
        .unwrap();
    let batch_bindings = [
        VulkanResidentKernelBufferBinding::new(0, &batch_input, WIDTH * FRAME_BYTES)
            .with_access(VulkanResidentKernelBufferAccess::Read),
        VulkanResidentKernelBufferBinding::new(1, &batch_output, WIDTH * FRAME_BYTES)
            .with_access(VulkanResidentKernelBufferAccess::Write),
        VulkanResidentKernelBufferBinding::new(
            2,
            &batch_quantized,
            WIDTH * QUANTIZED_FRAME_BYTES,
        )
        .with_access(VulkanResidentKernelBufferAccess::Write),
        VulkanResidentKernelBufferBinding::new(3, &batch_scales, WIDTH * SCALE_FRAME_BYTES)
            .with_access(VulkanResidentKernelBufferAccess::Write),
        VulkanResidentKernelBufferBinding::new(4, &weight, weight_bytes.len())
            .with_access(VulkanResidentKernelBufferAccess::Read),
        VulkanResidentKernelBufferBinding::new(31, &control, 4)
            .with_access(VulkanResidentKernelBufferAccess::Read),
    ];
    let batch_dispatch = device
        .create_resident_kernel_dispatch_2d(
            &batch_words,
            &batch_bindings,
            1,
            1,
            1_024,
            0,
        )
        .unwrap();
    device
        .run_resident_kernel_dispatch(&batch_dispatch, &[])
        .unwrap();

    let assert_equal = |label: &str, scalar: &[u8], batch: Vec<u8>| {
        let first_mismatch = scalar
            .iter()
            .zip(&batch)
            .position(|(scalar, batch)| scalar != batch);
        assert_eq!(
            first_mismatch, None,
            "{label} first differs at byte {first_mismatch:?}"
        );
    };
    assert_equal(
        "normalized BF16 output",
        &scalar_outputs,
        batch_output.read_bytes(WIDTH * FRAME_BYTES).unwrap(),
    );
    assert_equal(
        "quantized FP8 output",
        &scalar_quantized_outputs,
        batch_quantized
            .read_bytes(WIDTH * QUANTIZED_FRAME_BYTES)
            .unwrap(),
    );
    assert_equal(
        "FP8 scales",
        &scalar_scale_outputs,
        batch_scales
            .read_bytes(WIDTH * SCALE_FRAME_BYTES)
            .unwrap(),
    );
}

#[test]
fn fused_pairpacked_rms_norm_matches_separate_exact_operations() {
    const HIDDEN_SIZE: usize = 5_120;
    const BLOCK_COLUMNS: usize = 32;
    const FRAME_BYTES: usize = HIDDEN_SIZE * std::mem::size_of::<u16>();
    const QUANTIZED_FRAME_BYTES: usize = HIDDEN_SIZE;
    const SCALE_FRAME_BYTES: usize =
        (HIDDEN_SIZE / BLOCK_COLUMNS) * std::mem::size_of::<f32>();
    const SUM_FRAME_BYTES: usize =
        (HIDDEN_SIZE / BLOCK_COLUMNS) * std::mem::size_of::<i32>();

    let (
        Some(manifest_path),
        Some(reference_manifest_path),
    ) = (
        std::env::var_os("NERVE_REPRESENTATION_CANDIDATE_PACKAGE")
            .map(PathBuf::from),
        std::env::var_os("NERVE_REPRESENTATION_REFERENCE_PACKAGE")
            .map(PathBuf::from),
    )
    else {
        eprintln!(
            "skipping fused pairpacked RMS norm equivalence: \
             representation candidate or reference package is unset"
        );
        return;
    };
    let device = selected_test_vulkan_device().expect("selected Vulkan test device must open");
    let package_dir = manifest_path.parent().unwrap();
    let reference_package_dir = reference_manifest_path.parent().unwrap();
    let norm_words = read_spirv_words(
        reference_package_dir
            .join("shaders/rms_norm_bf16_h5120_eps1e-06_offset1.spv"),
    )
    .unwrap();
    let quantize_words = read_spirv_words(
        reference_package_dir
            .join("shaders/quantize_int8_symmetric_pairpacked_b32_h5120.spv"),
    )
    .unwrap();
    let fused_words = read_spirv_words(
        package_dir.join(
            "shaders/rms_norm_quantize_int8_pairpacked_b32_h5120_eps1e-06_offset1.spv",
        ),
    )
    .unwrap();
    let tensor_index =
        TensorIndex::from_package_json_file(package_dir.join("tensors.json")).unwrap();
    let embedding_storage = crate::tensor_storage::TensorStorage::from_index(
        &tensor_index,
        "model.language_model.embed_tokens.weight",
    )
    .unwrap();
    let mut embedding_file = std::fs::File::open(&embedding_storage.source_file).unwrap();
    let mut header_len_bytes = [0u8; 8];
    embedding_file.read_exact(&mut header_len_bytes).unwrap();
    let tensor_data_start = 8 + u64::from_le_bytes(header_len_bytes);
    embedding_file
        .seek(SeekFrom::Start(
            tensor_data_start
                + u64::try_from(100usize * FRAME_BYTES + embedding_storage.data_start).unwrap(),
        ))
        .unwrap();
    let mut input_bytes = vec![0; FRAME_BYTES];
    embedding_file.read_exact(&mut input_bytes).unwrap();
    let weight_bytes = crate::tensor_storage::TensorStorage::from_index(
        &tensor_index,
        "model.language_model.layers.0.input_layernorm.weight",
    )
    .unwrap()
    .read_all()
    .unwrap();

    let input = device.create_resident_buffer(FRAME_BYTES).unwrap();
    input.write_bytes(&input_bytes).unwrap();
    let weight = device.create_resident_buffer(weight_bytes.len()).unwrap();
    weight.write_bytes(&weight_bytes).unwrap();

    let expected_output = device.create_resident_buffer(FRAME_BYTES).unwrap();
    let norm_bindings = [
        VulkanResidentKernelBufferBinding::new(0, &input, FRAME_BYTES)
            .with_access(VulkanResidentKernelBufferAccess::Read),
        VulkanResidentKernelBufferBinding::new(1, &expected_output, FRAME_BYTES)
            .with_access(VulkanResidentKernelBufferAccess::Write),
        VulkanResidentKernelBufferBinding::new(2, &weight, weight_bytes.len())
            .with_access(VulkanResidentKernelBufferAccess::Read),
    ];
    let norm_dispatch = device
        .create_resident_kernel_dispatch(&norm_words, &norm_bindings, 1, 64, 0)
        .unwrap();
    device
        .run_resident_kernel_dispatch(&norm_dispatch, &[])
        .unwrap();

    let expected_quantized = device
        .create_resident_buffer(QUANTIZED_FRAME_BYTES)
        .unwrap();
    let expected_scales = device.create_resident_buffer(SCALE_FRAME_BYTES).unwrap();
    let expected_sums = device.create_resident_buffer(SUM_FRAME_BYTES).unwrap();
    let quantize_bindings = [
        VulkanResidentKernelBufferBinding::new(0, &expected_output, FRAME_BYTES)
            .with_access(VulkanResidentKernelBufferAccess::Read),
        VulkanResidentKernelBufferBinding::new(
            1,
            &expected_quantized,
            QUANTIZED_FRAME_BYTES,
        )
        .with_access(VulkanResidentKernelBufferAccess::Write),
        VulkanResidentKernelBufferBinding::new(2, &expected_scales, SCALE_FRAME_BYTES)
            .with_access(VulkanResidentKernelBufferAccess::Write),
        VulkanResidentKernelBufferBinding::new(3, &expected_sums, SUM_FRAME_BYTES)
            .with_access(VulkanResidentKernelBufferAccess::Write),
    ];
    let quantize_dispatch = device
        .create_resident_kernel_dispatch(
            &quantize_words,
            &quantize_bindings,
            u32::try_from(HIDDEN_SIZE / BLOCK_COLUMNS).unwrap(),
            32,
            0,
        )
        .unwrap();
    device
        .run_resident_kernel_dispatch(&quantize_dispatch, &[])
        .unwrap();

    let actual_output = device.create_resident_buffer(FRAME_BYTES).unwrap();
    let actual_quantized = device
        .create_resident_buffer(QUANTIZED_FRAME_BYTES)
        .unwrap();
    let actual_scales = device.create_resident_buffer(SCALE_FRAME_BYTES).unwrap();
    let actual_sums = device.create_resident_buffer(SUM_FRAME_BYTES).unwrap();
    let fused_bindings = [
        VulkanResidentKernelBufferBinding::new(0, &input, FRAME_BYTES)
            .with_access(VulkanResidentKernelBufferAccess::Read),
        VulkanResidentKernelBufferBinding::new(1, &actual_output, FRAME_BYTES)
            .with_access(VulkanResidentKernelBufferAccess::Write),
        VulkanResidentKernelBufferBinding::new(2, &actual_quantized, QUANTIZED_FRAME_BYTES)
            .with_access(VulkanResidentKernelBufferAccess::Write),
        VulkanResidentKernelBufferBinding::new(3, &actual_scales, SCALE_FRAME_BYTES)
            .with_access(VulkanResidentKernelBufferAccess::Write),
        VulkanResidentKernelBufferBinding::new(4, &actual_sums, SUM_FRAME_BYTES)
            .with_access(VulkanResidentKernelBufferAccess::Write),
        VulkanResidentKernelBufferBinding::new(5, &weight, weight_bytes.len())
            .with_access(VulkanResidentKernelBufferAccess::Read),
    ];
    let fused_dispatch = device
        .create_resident_kernel_dispatch(&fused_words, &fused_bindings, 1, 1_024, 0)
        .unwrap();
    device
        .run_resident_kernel_dispatch(&fused_dispatch, &[])
        .unwrap();

    let assert_equal = |label: &str, expected: Vec<u8>, actual: Vec<u8>| {
        let first_mismatch = expected
            .iter()
            .zip(&actual)
            .position(|(expected, actual)| expected != actual);
        assert_eq!(
            first_mismatch, None,
            "{label} first differs at byte {first_mismatch:?}"
        );
    };
    assert_equal(
        "normalized BF16 output",
        expected_output.read_bytes(FRAME_BYTES).unwrap(),
        actual_output.read_bytes(FRAME_BYTES).unwrap(),
    );
    assert_equal(
        "pairpacked INT8 output",
        expected_quantized
            .read_bytes(QUANTIZED_FRAME_BYTES)
            .unwrap(),
        actual_quantized
            .read_bytes(QUANTIZED_FRAME_BYTES)
            .unwrap(),
    );
    assert_equal(
        "pairpacked scales",
        expected_scales.read_bytes(SCALE_FRAME_BYTES).unwrap(),
        actual_scales.read_bytes(SCALE_FRAME_BYTES).unwrap(),
    );
    assert_equal(
        "pairpacked block sums",
        expected_sums.read_bytes(SUM_FRAME_BYTES).unwrap(),
        actual_sums.read_bytes(SUM_FRAME_BYTES).unwrap(),
    );
}

#[test]
fn fused_batched_pairpacked_rms_norm_matches_separate_exact_operations() {
    const HIDDEN_SIZE: usize = 5_120;
    const BLOCK_COLUMNS: usize = 32;
    const WIDTH: usize = 12;
    const FRAME_BYTES: usize = HIDDEN_SIZE * std::mem::size_of::<u16>();
    const QUANTIZED_FRAME_BYTES: usize = HIDDEN_SIZE;
    const SCALE_FRAME_BYTES: usize =
        (HIDDEN_SIZE / BLOCK_COLUMNS) * std::mem::size_of::<f32>();
    const SUM_FRAME_BYTES: usize =
        (HIDDEN_SIZE / BLOCK_COLUMNS) * std::mem::size_of::<i32>();

    let (
        Some(manifest_path),
        Some(reference_manifest_path),
    ) = (
        std::env::var_os("NERVE_REPRESENTATION_CANDIDATE_PACKAGE")
            .map(PathBuf::from),
        std::env::var_os("NERVE_REPRESENTATION_REFERENCE_PACKAGE")
            .map(PathBuf::from),
    )
    else {
        eprintln!(
            "skipping fused batched pairpacked RMS norm equivalence: \
             representation candidate or reference package is unset"
        );
        return;
    };
    let device = selected_test_vulkan_device().expect("selected Vulkan test device must open");
    let package_dir = manifest_path.parent().unwrap();
    let reference_package_dir = reference_manifest_path.parent().unwrap();
    let norm_words = read_spirv_words(
        reference_package_dir
            .join("shaders/rms_norm_batch16_bf16_h5120_eps1e-06_offset1__pbc31.spv"),
    )
    .unwrap();
    let quantize_words = read_spirv_words(
        reference_package_dir.join(
            "shaders/quantize_batch16_int8_symmetric_pairpacked_b32_h5120__pbc31.spv",
        ),
    )
    .unwrap();
    let fused_words = read_spirv_words(
        package_dir.join(
            "shaders/rms_norm_quantize_batch16_int8_pairpacked_b32_h5120_eps1e-06_offset1__pbc31.spv",
        ),
    )
    .unwrap();
    let tensor_index =
        TensorIndex::from_package_json_file(package_dir.join("tensors.json")).unwrap();
    let embedding_storage = crate::tensor_storage::TensorStorage::from_index(
        &tensor_index,
        "model.language_model.embed_tokens.weight",
    )
    .unwrap();
    let mut embedding_file = std::fs::File::open(&embedding_storage.source_file).unwrap();
    let mut header_len_bytes = [0u8; 8];
    embedding_file.read_exact(&mut header_len_bytes).unwrap();
    let tensor_data_start = 8 + u64::from_le_bytes(header_len_bytes);
    let mut input_bytes = Vec::with_capacity(WIDTH * FRAME_BYTES);
    for token_id in 0..WIDTH {
        embedding_file
            .seek(SeekFrom::Start(
                tensor_data_start
                    + u64::try_from(embedding_storage.data_start + token_id * FRAME_BYTES)
                        .unwrap(),
            ))
            .unwrap();
        let mut frame = vec![0; FRAME_BYTES];
        embedding_file.read_exact(&mut frame).unwrap();
        input_bytes.extend(frame);
    }
    let weight_bytes = crate::tensor_storage::TensorStorage::from_index(
        &tensor_index,
        "model.language_model.layers.0.input_layernorm.weight",
    )
    .unwrap()
    .read_all()
    .unwrap();

    let input = device
        .create_resident_buffer(WIDTH * FRAME_BYTES)
        .unwrap();
    input.write_bytes(&input_bytes).unwrap();
    let weight = device.create_resident_buffer(weight_bytes.len()).unwrap();
    weight.write_bytes(&weight_bytes).unwrap();
    let control = device.create_host_visible_resident_buffer(4).unwrap();
    control
        .write_bytes(&u32::try_from(WIDTH).unwrap().to_le_bytes())
        .unwrap();

    let expected_output = device
        .create_resident_buffer(WIDTH * FRAME_BYTES)
        .unwrap();
    let norm_bindings = [
        VulkanResidentKernelBufferBinding::new(0, &input, WIDTH * FRAME_BYTES)
            .with_access(VulkanResidentKernelBufferAccess::Read),
        VulkanResidentKernelBufferBinding::new(1, &expected_output, WIDTH * FRAME_BYTES)
            .with_access(VulkanResidentKernelBufferAccess::Write),
        VulkanResidentKernelBufferBinding::new(2, &weight, weight_bytes.len())
            .with_access(VulkanResidentKernelBufferAccess::Read),
        VulkanResidentKernelBufferBinding::new(31, &control, 4)
            .with_access(VulkanResidentKernelBufferAccess::Read),
    ];
    let norm_dispatch = device
        .create_resident_kernel_dispatch_2d(&norm_words, &norm_bindings, 1, 1, 64, 0)
        .unwrap();
    device
        .run_resident_kernel_dispatch(&norm_dispatch, &[])
        .unwrap();

    let expected_quantized = device
        .create_resident_buffer(WIDTH * QUANTIZED_FRAME_BYTES)
        .unwrap();
    let expected_scales = device
        .create_resident_buffer(WIDTH * SCALE_FRAME_BYTES)
        .unwrap();
    let expected_sums = device
        .create_resident_buffer(WIDTH * SUM_FRAME_BYTES)
        .unwrap();
    let quantize_bindings = [
        VulkanResidentKernelBufferBinding::new(0, &expected_output, WIDTH * FRAME_BYTES)
            .with_access(VulkanResidentKernelBufferAccess::Read),
        VulkanResidentKernelBufferBinding::new(
            1,
            &expected_quantized,
            WIDTH * QUANTIZED_FRAME_BYTES,
        )
        .with_access(VulkanResidentKernelBufferAccess::Write),
        VulkanResidentKernelBufferBinding::new(
            2,
            &expected_scales,
            WIDTH * SCALE_FRAME_BYTES,
        )
        .with_access(VulkanResidentKernelBufferAccess::Write),
        VulkanResidentKernelBufferBinding::new(3, &expected_sums, WIDTH * SUM_FRAME_BYTES)
            .with_access(VulkanResidentKernelBufferAccess::Write),
        VulkanResidentKernelBufferBinding::new(31, &control, 4)
            .with_access(VulkanResidentKernelBufferAccess::Read),
    ];
    let quantize_dispatch = device
        .create_resident_kernel_dispatch_2d(
            &quantize_words,
            &quantize_bindings,
            u32::try_from(HIDDEN_SIZE / BLOCK_COLUMNS).unwrap(),
            1,
            32,
            0,
        )
        .unwrap();
    device
        .run_resident_kernel_dispatch(&quantize_dispatch, &[])
        .unwrap();

    let actual_output = device
        .create_resident_buffer(WIDTH * FRAME_BYTES)
        .unwrap();
    let actual_quantized = device
        .create_resident_buffer(WIDTH * QUANTIZED_FRAME_BYTES)
        .unwrap();
    let actual_scales = device
        .create_resident_buffer(WIDTH * SCALE_FRAME_BYTES)
        .unwrap();
    let actual_sums = device
        .create_resident_buffer(WIDTH * SUM_FRAME_BYTES)
        .unwrap();
    let fused_bindings = [
        VulkanResidentKernelBufferBinding::new(0, &input, WIDTH * FRAME_BYTES)
            .with_access(VulkanResidentKernelBufferAccess::Read),
        VulkanResidentKernelBufferBinding::new(1, &actual_output, WIDTH * FRAME_BYTES)
            .with_access(VulkanResidentKernelBufferAccess::Write),
        VulkanResidentKernelBufferBinding::new(
            2,
            &actual_quantized,
            WIDTH * QUANTIZED_FRAME_BYTES,
        )
        .with_access(VulkanResidentKernelBufferAccess::Write),
        VulkanResidentKernelBufferBinding::new(3, &actual_scales, WIDTH * SCALE_FRAME_BYTES)
            .with_access(VulkanResidentKernelBufferAccess::Write),
        VulkanResidentKernelBufferBinding::new(4, &actual_sums, WIDTH * SUM_FRAME_BYTES)
            .with_access(VulkanResidentKernelBufferAccess::Write),
        VulkanResidentKernelBufferBinding::new(5, &weight, weight_bytes.len())
            .with_access(VulkanResidentKernelBufferAccess::Read),
        VulkanResidentKernelBufferBinding::new(31, &control, 4)
            .with_access(VulkanResidentKernelBufferAccess::Read),
    ];
    let fused_dispatch = device
        .create_resident_kernel_dispatch_2d(&fused_words, &fused_bindings, 1, 1, 1_024, 0)
        .unwrap();
    device
        .run_resident_kernel_dispatch(&fused_dispatch, &[])
        .unwrap();

    let assert_equal = |label: &str, expected: Vec<u8>, actual: Vec<u8>| {
        let first_mismatch = expected
            .iter()
            .zip(&actual)
            .position(|(expected, actual)| expected != actual);
        assert_eq!(
            first_mismatch, None,
            "{label} first differs at byte {first_mismatch:?}"
        );
    };
    assert_equal(
        "batched normalized BF16 output",
        expected_output.read_bytes(WIDTH * FRAME_BYTES).unwrap(),
        actual_output.read_bytes(WIDTH * FRAME_BYTES).unwrap(),
    );
    assert_equal(
        "batched pairpacked INT8 output",
        expected_quantized
            .read_bytes(WIDTH * QUANTIZED_FRAME_BYTES)
            .unwrap(),
        actual_quantized
            .read_bytes(WIDTH * QUANTIZED_FRAME_BYTES)
            .unwrap(),
    );
    assert_equal(
        "batched pairpacked scales",
        expected_scales
            .read_bytes(WIDTH * SCALE_FRAME_BYTES)
            .unwrap(),
        actual_scales
            .read_bytes(WIDTH * SCALE_FRAME_BYTES)
            .unwrap(),
    );
    assert_equal(
        "batched pairpacked block sums",
        expected_sums
            .read_bytes(WIDTH * SUM_FRAME_BYTES)
            .unwrap(),
        actual_sums
            .read_bytes(WIDTH * SUM_FRAME_BYTES)
            .unwrap(),
    );
}

#[test]
fn fused_pairpacked_rms_norm_preserves_complete_component_output() {
    const HIDDEN_SIZE: usize = 5_120;
    const FRAME_BYTES: usize = HIDDEN_SIZE * std::mem::size_of::<u16>();

    let (
        Some(candidate_manifest_path),
        Some(reference_manifest_path),
    ) = (
        std::env::var_os("NERVE_REPRESENTATION_CANDIDATE_PACKAGE")
            .map(PathBuf::from),
        std::env::var_os("NERVE_REPRESENTATION_REFERENCE_PACKAGE")
            .map(PathBuf::from),
    )
    else {
        eprintln!(
            "skipping complete fused pairpacked RMS norm equivalence: \
             representation candidate or reference package is unset"
        );
        return;
    };
    let candidate_package_dir = candidate_manifest_path.parent().unwrap();
    let tensor_index = TensorIndex::from_package_json_file(
        candidate_package_dir.join("tensors.json"),
    )
    .unwrap();
    let embedding_storage = crate::tensor_storage::TensorStorage::from_index(
        &tensor_index,
        "model.language_model.embed_tokens.weight",
    )
    .unwrap();
    let mut embedding_file = std::fs::File::open(&embedding_storage.source_file).unwrap();
    let mut header_len_bytes = [0u8; 8];
    embedding_file.read_exact(&mut header_len_bytes).unwrap();
    let tensor_data_start = 8 + u64::from_le_bytes(header_len_bytes);
    embedding_file
        .seek(SeekFrom::Start(
            tensor_data_start
                + u64::try_from(embedding_storage.data_start + 5_834usize * FRAME_BYTES)
                    .unwrap(),
        ))
        .unwrap();
    let mut input_bytes = vec![0; FRAME_BYTES];
    embedding_file.read_exact(&mut input_bytes).unwrap();

    let device = selected_test_vulkan_device().expect("selected Vulkan test device must open");
    let run_layer = |manifest_path: &Path| {
        let package_dir = manifest_path.parent().unwrap();
        let manifest =
            VulkanResidentModelPackageManifest::from_json_file(manifest_path).unwrap();
        let chain = [("layer_00".to_string(), "layer_00".to_string())];
        let runtime_model = manifest
            .mount_runtime_graph_controls(
                Some("gpu0"),
                &BTreeMap::new(),
                &[],
                Some(&chain),
            )
            .unwrap();
        let slice =
            VulkanResidentModelPackageDeviceSlice::from_runtime_model_for_device(
                &device,
                package_dir,
                runtime_model,
                "gpu0",
                Some(16),
            )
            .unwrap();
        let mounted = slice.create_mounted_stream_circuit(&device).unwrap();
        mounted.buffers.zero_state_buffers().unwrap();
        mounted
            .buffers
            .apply_clone_state_policies()
            .unwrap();
        mounted
            .boundary_io
            .input_buffer("input_frame")
            .unwrap()
            .buffer
            .write_bytes(&input_bytes)
            .unwrap();
        let reusable_manifest =
            resident_package_reusable_kernel_manifest(&mounted.placed_plan);
        let bound = mounted
            .mounted_placed_bound_dispatch_plan(&reusable_manifest)
            .unwrap();
        let tick_plan = mounted.stream_tick_plan(&reusable_manifest).unwrap();
        let mut transport = VulkanInProcessPlacedEdgeTransport::new();
        let run = tick_plan
            .advance_with_resident_execution_graph_and_in_process_transport(
                &device,
                &mounted,
                &bound,
                slice.loaded_manifest(),
                &mut transport,
                0,
            )
            .unwrap();
        assert_eq!(
            run.tick_run.status,
            VulkanMountedPlacedStreamTickRunStatus::Completed
        );
        mounted
            .boundary_io
            .output_buffer("output_frame")
            .unwrap()
            .buffer
            .read_bytes(FRAME_BYTES)
            .unwrap()
    };

    let reference = run_layer(&reference_manifest_path);
    let candidate = run_layer(&candidate_manifest_path);
    let first_mismatch = reference
        .iter()
        .zip(&candidate)
        .position(|(reference, candidate)| reference != candidate);
    assert_eq!(
        first_mismatch, None,
        "complete layer_00 output first differs at byte {first_mismatch:?}"
    );
}
