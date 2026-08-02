#[test]
fn resident_greedy_sampler_selects_largest_logit() {
    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident sampler: {error}");
            return;
        }
    };
    let sampler_spirv_words =
        read_spirv_words(tiny_model_dir().join("shaders/greedy_sampler_f32_32.spv")).unwrap();
    let Some(sampler_kernels) = greedy_sampler_test_kernels(sampler_spirv_words) else {
        eprintln!("skipping sampler smoke: feedback control shader did not compile");
        return;
    };

    let logits_buffer = device
        .create_resident_buffer(FIXTURE_MODEL_LOGITS_BYTES)
        .unwrap();
    let mut logits = vec![0u8; FIXTURE_MODEL_LOGITS_BYTES];
    let token_7 = 7usize;
    let token_24 = 24usize;
    logits[(token_7 * 4)..((token_7 + 1) * 4)].copy_from_slice(&3.5f32.to_le_bytes());
    logits[(token_24 * 4)..((token_24 + 1) * 4)].copy_from_slice(&9.25f32.to_le_bytes());
    logits_buffer.write_bytes(&logits).unwrap();

    let stream_control_buffer = Arc::new(
        device
            .create_host_visible_resident_buffer(VULKAN_STREAM_CONTROL_BYTE_CAPACITY)
            .unwrap(),
    );
    stream_control_buffer
        .write_bytes(&stream_control_bytes(
            0,
            VulkanMountedPlacedStreamControl {
                stream_tick: 0x0000_0007_ffff_ffff,
                control_flags: 0,
                dynamic_state_capacity_activations: 8,
            },
        ))
        .unwrap();
    let runner = VulkanResidentSamplerRunner::from_logits_buffer(
        &device,
        stream_control_buffer.clone(),
        &logits_buffer,
        FIXTURE_MODEL_LOGITS_BYTES,
        &sampler_kernels,
        &fixture_model_greedy_sampler_spec(),
        VulkanResidentSamplerStreamConfig {
            history_capacity_activations: 8,
            random_seed: 0,
        },
    )
    .unwrap();
    assert_eq!(runner.sampler_id, FIXTURE_MODEL_GREEDY_SAMPLER_COMPONENT_ID);
    assert_eq!(runner.logits_byte_capacity, FIXTURE_MODEL_LOGITS_BYTES);
    assert_eq!(
        runner.output_byte_capacity,
        FIXTURE_MODEL_SAMPLER_OUTPUT_BYTES
    );
    assert_eq!(runner.dispatch_count, 1);
    assert_eq!(runner.descriptor_count, 5);
    assert_eq!(runner.workgroup_count_x, 1);
    assert_eq!(runner.push_constant_byte_count, 0);

    let run = runner.run(&device).unwrap();
    assert_eq!(run.sampler_id, FIXTURE_MODEL_GREEDY_SAMPLER_COMPONENT_ID);
    assert_eq!(run.token_id, token_24 as u32);
    assert_eq!(run.selected_logit_bits, 9.25f32.to_bits());
    assert_eq!(run.control_flags, 0);
    assert_eq!(run.descriptor_count, 5);
    assert_eq!(run.workgroup_count_x, 1);
    assert_eq!(run.push_constant_byte_count, 0);
    assert_eq!(
        runner.read_output_bytes().unwrap(),
        vec![
            0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x14, 0x41, 0, 0, 0, 0, 0, 0, 0, 0
        ]
    );
    let control = stream_control_buffer
        .read_bytes(VULKAN_STREAM_CONTROL_BYTE_CAPACITY)
        .unwrap();
    assert_eq!(u32::from_le_bytes(control[0..4].try_into().unwrap()), 24);
    assert_eq!(u32::from_le_bytes(control[4..8].try_into().unwrap()), 0);
    assert_eq!(u32::from_le_bytes(control[8..12].try_into().unwrap()), 8);
    assert_eq!(runner.completed_run_at(0x0000_0007_ffff_ffff).unwrap(), run);
}

#[test]
fn resident_temperature_top_k_top_p_sampler_matches_explicit_random_signal() {
    const VOCAB_SIZE: usize = 64;
    const LOGITS_BYTE_CAPACITY: usize = VOCAB_SIZE * std::mem::size_of::<f32>();
    const SEED: u32 = 0x5eed_1234;
    const TOP_TOKENS: [u32; 4] = [7, 8, 19, 51];

    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident sampled sampler: {error}");
            return;
        }
    };
    let partition_count = 4;
    let Some(sampler_kernels) = compile_temperature_top_k_top_p_sampler_test_kernels(
        VOCAB_SIZE,
        1.0,
        4,
        1.0,
        partition_count,
        16,
    ) else {
        eprintln!("skipping resident sampled sampler: no GLSL to SPIR-V compiler found");
        return;
    };

    let logits_buffer = device.create_resident_buffer(LOGITS_BYTE_CAPACITY).unwrap();
    let mut logits = vec![-100.0f32; VOCAB_SIZE];
    for token_id in TOP_TOKENS {
        logits[token_id as usize] = 2.0;
    }
    logits_buffer
        .write_bytes(
            &logits
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();

    let stream_control_buffer = Arc::new(
        device
            .create_host_visible_resident_buffer(VULKAN_STREAM_CONTROL_BYTE_CAPACITY)
            .unwrap(),
    );
    stream_control_buffer
        .write_bytes(&stream_control_bytes(
            0,
            VulkanMountedPlacedStreamControl {
                stream_tick: 0,
                control_flags: 0,
                dynamic_state_capacity_activations: 32,
            },
        ))
        .unwrap();
    let spec = VulkanResidentSamplerSpec {
        sampler_id: "temperature_top_k_top_p_sampler".to_string(),
        method: "temperature_top_k_top_p".to_string(),
        temperature: 1.0,
        top_k: 4,
        top_p: 1.0,
        min_p: 0.0,
        presence_penalty: 0.0,
        repetition_penalty: 1.0,
        top_k_capacity: 4,
        runtime_parameterized: false,
        logits_byte_capacity: LOGITS_BYTE_CAPACITY,
        output_byte_capacity: FIXTURE_MODEL_SAMPLER_OUTPUT_BYTES,
        scratch_byte_capacity: partition_count as usize * 4 * 8,
    };
    let mut invalid_spec = spec.clone();
    invalid_spec.scratch_byte_capacity -= 8;
    let invalid = VulkanResidentSamplerRunner::from_logits_buffer(
        &device,
        stream_control_buffer.clone(),
        &logits_buffer,
        LOGITS_BYTE_CAPACITY,
        &sampler_kernels,
        &invalid_spec,
        VulkanResidentSamplerStreamConfig {
            history_capacity_activations: 32,
            random_seed: SEED,
        },
    )
    .err()
    .expect("undersized sampler scratch must be rejected");
    assert!(
        invalid
            .to_string()
            .contains("invalid resident sampling spec")
    );
    let runner = VulkanResidentSamplerRunner::from_logits_buffer(
        &device,
        stream_control_buffer,
        &logits_buffer,
        LOGITS_BYTE_CAPACITY,
        &sampler_kernels,
        &spec,
        VulkanResidentSamplerStreamConfig {
            history_capacity_activations: 32,
            random_seed: SEED,
        },
    )
    .unwrap();

    let mut selected_tokens = Vec::new();
    for stream_tick in 0..16u32 {
        let run = runner.run(&device).unwrap();
        let random_bits = sampler_test_hash_u32(SEED ^ stream_tick ^ 0xa511_e9b3);
        let selected_index = (((random_bits >> 8) as u64 * 4) >> 24) as usize;
        let expected = TOP_TOKENS[selected_index];
        assert_eq!(run.token_id, expected);
        assert_eq!(run.selected_logit_bits, 2.0f32.to_bits());
        assert_eq!(run.control_flags, 1);
        assert_eq!(runner.dispatch_count, 2);
        assert_eq!(run.descriptor_count, 8);
        assert_eq!(run.workgroup_count_x, partition_count + 1);
        selected_tokens.push(run.token_id);
    }
    assert!(
        TOP_TOKENS
            .iter()
            .all(|token_id| selected_tokens.contains(token_id))
    );
}

#[test]
fn resident_runtime_nucleus_sampler_is_exact_across_equal_probability_ties() {
    const VOCAB_SIZE: usize = 64;
    const LOGITS_BYTE_CAPACITY: usize = VOCAB_SIZE * std::mem::size_of::<f32>();
    const SEED: u32 = 0x5eed_1234;
    const TOP_TOKENS: [u32; 4] = [7, 8, 19, 51];

    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident full-distribution sampler: {error}");
            return;
        }
    };
    let partition_count = 4;
    let Some(sampler_kernels) = compile_temperature_distribution_sampler_test_kernels(
        VOCAB_SIZE,
        1.0,
        partition_count,
        16,
    ) else {
        eprintln!("skipping resident full-distribution sampler: no GLSL to SPIR-V compiler found");
        return;
    };
    let logits_buffer = device.create_resident_buffer(LOGITS_BYTE_CAPACITY).unwrap();
    let mut logits = vec![-100.0f32; VOCAB_SIZE];
    for token_id in TOP_TOKENS {
        logits[token_id as usize] = 2.0;
    }
    logits_buffer
        .write_bytes(
            &logits
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let stream_control_buffer = Arc::new(
        device
            .create_host_visible_resident_buffer(VULKAN_STREAM_CONTROL_BYTE_CAPACITY)
            .unwrap(),
    );
    stream_control_buffer
        .write_bytes(&stream_control_bytes(
            0,
            VulkanMountedPlacedStreamControl {
                stream_tick: 0,
                control_flags: 0,
                dynamic_state_capacity_activations: 32,
            },
        ))
        .unwrap();
    let spec = VulkanResidentSamplerSpec {
        sampler_id: "runtime_temperature_distribution_sampler".to_string(),
        method: "temperature_top_p".to_string(),
        temperature: 1.0,
        top_k: 0,
        top_p: 0.5,
        min_p: 0.0,
        presence_penalty: 0.0,
        repetition_penalty: 1.0,
        top_k_capacity: 256,
        runtime_parameterized: true,
        logits_byte_capacity: LOGITS_BYTE_CAPACITY,
        output_byte_capacity: FIXTURE_MODEL_SAMPLER_OUTPUT_BYTES,
        scratch_byte_capacity: partition_count as usize * 8,
    };
    let runner = VulkanResidentSamplerRunner::from_logits_buffer(
        &device,
        stream_control_buffer,
        &logits_buffer,
        LOGITS_BYTE_CAPACITY,
        &sampler_kernels,
        &spec,
        VulkanResidentSamplerStreamConfig {
            history_capacity_activations: 32,
            random_seed: SEED,
        },
    )
    .unwrap();

    let selected = (0..16)
        .map(|_| runner.run(&device).unwrap().token_id)
        .collect::<Vec<_>>();
    assert!(selected.iter().all(|token_id| TOP_TOKENS[..2].contains(token_id)));
    assert!(selected.iter().all(|token_id| *token_id != 0));
}

#[test]
fn resident_full_distribution_sampler_seed_zero_does_not_collapse_to_token_zero() {
    const VOCAB_SIZE: usize = 64;
    const LOGITS_BYTE_CAPACITY: usize = VOCAB_SIZE * std::mem::size_of::<f32>();
    const EXPECTED_TOKEN: u32 = 17;

    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident full-distribution sampler: {error}");
            return;
        }
    };
    let partition_count = 4;
    let Some(sampler_kernels) = compile_temperature_distribution_sampler_test_kernels(
        VOCAB_SIZE,
        1.0,
        partition_count,
        16,
    ) else {
        eprintln!("skipping resident full-distribution sampler: no GLSL to SPIR-V compiler found");
        return;
    };
    let logits_buffer = device.create_resident_buffer(LOGITS_BYTE_CAPACITY).unwrap();
    let mut logits = vec![-100.0f32; VOCAB_SIZE];
    logits[0] = -12.0;
    logits[EXPECTED_TOKEN as usize] = 20.0;
    logits_buffer
        .write_bytes(
            &logits
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let stream_control_buffer = Arc::new(
        device
            .create_host_visible_resident_buffer(VULKAN_STREAM_CONTROL_BYTE_CAPACITY)
            .unwrap(),
    );
    stream_control_buffer
        .write_bytes(&stream_control_bytes(
            0,
            VulkanMountedPlacedStreamControl {
                stream_tick: 0,
                control_flags: 0,
                dynamic_state_capacity_activations: 32,
            },
        ))
        .unwrap();
    let spec = VulkanResidentSamplerSpec {
        sampler_id: "runtime_temperature_distribution_sampler".to_string(),
        method: "temperature_top_p".to_string(),
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        presence_penalty: 0.0,
        repetition_penalty: 1.0,
        top_k_capacity: 256,
        runtime_parameterized: true,
        logits_byte_capacity: LOGITS_BYTE_CAPACITY,
        output_byte_capacity: FIXTURE_MODEL_SAMPLER_OUTPUT_BYTES,
        scratch_byte_capacity: partition_count as usize * 8,
    };
    let runner = VulkanResidentSamplerRunner::from_logits_buffer(
        &device,
        stream_control_buffer,
        &logits_buffer,
        LOGITS_BYTE_CAPACITY,
        &sampler_kernels,
        &spec,
        VulkanResidentSamplerStreamConfig {
            history_capacity_activations: 32,
            random_seed: 0,
        },
    )
    .unwrap();

    let run = runner.run(&device).unwrap();
    assert_eq!(run.token_id, EXPECTED_TOKEN);
    assert_eq!(run.selected_logit_bits, 20.0f32.to_bits());
}

#[test]
fn resident_repetition_sampler_tracks_prompt_and_feedback_tokens_on_gpu() {
    const VOCAB_SIZE: usize = 64;
    const LOGITS_BYTE_CAPACITY: usize = VOCAB_SIZE * std::mem::size_of::<f32>();
    const PARTITION_COUNT: u32 = 4;
    const REPETITION_PENALTY: f32 = 1.1;

    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident repetition sampler: {error}");
            return;
        }
    };
    let Some(kernels) = compile_repetition_temperature_sampler_test_kernels(
        VOCAB_SIZE,
        REPETITION_PENALTY,
        1,
        PARTITION_COUNT,
        16,
    ) else {
        eprintln!("skipping resident repetition sampler: no GLSL to SPIR-V compiler found");
        return;
    };
    let logits_buffer = device.create_resident_buffer(LOGITS_BYTE_CAPACITY).unwrap();
    let write_logits = |values: &[f32]| {
        logits_buffer
            .write_bytes(
                &values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
    };
    let stream_control_buffer = Arc::new(
        device
            .create_host_visible_resident_buffer(VULKAN_STREAM_CONTROL_BYTE_CAPACITY)
            .unwrap(),
    );
    stream_control_buffer
        .write_bytes(&stream_control_bytes(
            0,
            VulkanMountedPlacedStreamControl {
                stream_tick: 0,
                control_flags: 0,
                dynamic_state_capacity_activations: 32,
            },
        ))
        .unwrap();
    let spec = VulkanResidentSamplerSpec {
        sampler_id: "repetition_sampler".to_string(),
        method: "temperature_top_k_top_p".to_string(),
        temperature: 1.0,
        top_k: 1,
        top_p: 1.0,
        min_p: 0.0,
        presence_penalty: 0.0,
        repetition_penalty: REPETITION_PENALTY,
        top_k_capacity: 1,
        runtime_parameterized: false,
        logits_byte_capacity: LOGITS_BYTE_CAPACITY,
        output_byte_capacity: FIXTURE_MODEL_SAMPLER_OUTPUT_BYTES,
        scratch_byte_capacity: PARTITION_COUNT as usize * 8,
    };
    let runner = VulkanResidentSamplerRunner::from_logits_buffer(
        &device,
        stream_control_buffer.clone(),
        &logits_buffer,
        LOGITS_BYTE_CAPACITY,
        &kernels,
        &spec,
        VulkanResidentSamplerStreamConfig {
            history_capacity_activations: 32,
            random_seed: 7,
        },
    )
    .unwrap();
    assert_eq!(runner.dispatch_count, 3);
    assert_eq!(runner.descriptor_count, 11);
    assert_eq!(runner.workgroup_count_x, 6);

    let mut logits = vec![-100.0; VOCAB_SIZE];
    logits[7] = 10.0;
    logits[8] = 9.6;
    write_logits(&logits);
    assert_eq!(runner.run(&device).unwrap().token_id, 7);

    runner.record_input_tokens(&device, &[7]).unwrap();
    assert_eq!(runner.run(&device).unwrap().token_id, 8);

    logits.fill(-100.0);
    logits[7] = -1.0;
    logits[8] = -1.05;
    write_logits(&logits);
    assert_eq!(runner.run(&device).unwrap().token_id, 8);

    logits.fill(-100.0);
    logits[9] = 5.0;
    logits[10] = 4.8;
    write_logits(&logits);
    stream_control_buffer
        .write_bytes(&9u32.to_le_bytes())
        .unwrap();
    device
        .run_resident_kernel_dispatch(&runner.input_tracking_dispatches()[0], &[])
        .unwrap();
    assert_eq!(runner.run(&device).unwrap().token_id, 10);
}

#[test]
fn resident_runtime_sampler_applies_presence_penalty_from_gpu_state() {
    const VOCAB_SIZE: usize = 64;
    const PARTITION_COUNT: u32 = 4;
    const TOP_K_CAPACITY: u32 = 8;
    const LOGITS_BYTE_CAPACITY: usize = VOCAB_SIZE * std::mem::size_of::<f32>();

    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping runtime presence sampler: {error}");
            return;
        }
    };
    let Some(kernels) = compile_runtime_temperature_sampler_test_kernels(
        VOCAB_SIZE,
        TOP_K_CAPACITY,
        PARTITION_COUNT,
        16,
    ) else {
        eprintln!("skipping runtime presence sampler: no GLSL to SPIR-V compiler found");
        return;
    };
    let logits_buffer = device.create_resident_buffer(LOGITS_BYTE_CAPACITY).unwrap();
    let write_logits = |values: &[f32]| {
        logits_buffer
            .write_bytes(
                &values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
    };
    let stream_control_buffer = Arc::new(
        device
            .create_host_visible_resident_buffer(VULKAN_STREAM_CONTROL_BYTE_CAPACITY)
            .unwrap(),
    );
    stream_control_buffer
        .write_bytes(&stream_control_bytes(
            0,
            VulkanMountedPlacedStreamControl {
                stream_tick: 0,
                control_flags: 0,
                dynamic_state_capacity_activations: 32,
            },
        ))
        .unwrap();
    let spec = VulkanResidentSamplerSpec {
        sampler_id: "runtime_sampler".to_string(),
        method: "temperature_top_k_top_p".to_string(),
        temperature: 1.0,
        top_k: 1,
        top_p: 1.0,
        min_p: 0.0,
        presence_penalty: 1.5,
        repetition_penalty: 1.0,
        top_k_capacity: TOP_K_CAPACITY,
        runtime_parameterized: true,
        logits_byte_capacity: LOGITS_BYTE_CAPACITY,
        output_byte_capacity: FIXTURE_MODEL_SAMPLER_OUTPUT_BYTES,
        scratch_byte_capacity: PARTITION_COUNT as usize
            * TOP_K_CAPACITY as usize
            * 2
            * std::mem::size_of::<u32>(),
    };
    let runner = VulkanResidentSamplerRunner::from_logits_buffer(
        &device,
        stream_control_buffer,
        &logits_buffer,
        LOGITS_BYTE_CAPACITY,
        &kernels,
        &spec,
        VulkanResidentSamplerStreamConfig {
            history_capacity_activations: 32,
            random_seed: 7,
        },
    )
    .unwrap();

    let mut logits = vec![-100.0; VOCAB_SIZE];
    logits[7] = 10.0;
    logits[8] = 9.0;
    write_logits(&logits);
    assert_eq!(runner.run(&device).unwrap().token_id, 7);

    runner.record_input_tokens(&device, &[7]).unwrap();
    assert_eq!(runner.run(&device).unwrap().token_id, 8);

    logits.fill(-100.0);
    logits[7] = -1.0;
    logits[9] = -1.25;
    write_logits(&logits);
    assert_eq!(runner.run(&device).unwrap().token_id, 9);

    runner.capture_token_state().unwrap();
    runner.record_input_tokens(&device, &[11]).unwrap();
    logits.fill(-100.0);
    logits[11] = 10.0;
    logits[12] = 9.0;
    write_logits(&logits);
    assert_eq!(runner.run(&device).unwrap().token_id, 12);
    runner.restore_token_state().unwrap();
    assert_eq!(runner.run(&device).unwrap().token_id, 11);
}

#[test]
fn speculative_sampler_views_isolate_hypothetical_presence_state() {
    const VOCAB_SIZE: usize = 64;
    const PARTITION_COUNT: u32 = 4;
    const TOP_K_CAPACITY: u32 = 8;
    const LOGITS_BYTE_CAPACITY: usize = VOCAB_SIZE * std::mem::size_of::<f32>();

    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping speculative presence sampler: {error}");
            return;
        }
    };
    let Some(kernels) = compile_runtime_temperature_sampler_test_kernels(
        VOCAB_SIZE,
        TOP_K_CAPACITY,
        PARTITION_COUNT,
        16,
    ) else {
        eprintln!("skipping speculative presence sampler: no GLSL compiler found");
        return;
    };
    let logits_buffer = device.create_resident_buffer(LOGITS_BYTE_CAPACITY).unwrap();
    let stream_control_buffer = Arc::new(
        device
            .create_host_visible_resident_buffer(VULKAN_STREAM_CONTROL_BYTE_CAPACITY)
            .unwrap(),
    );
    stream_control_buffer
        .write_bytes(&stream_control_bytes(
            0,
            VulkanMountedPlacedStreamControl {
                stream_tick: 0,
                control_flags: 0,
                dynamic_state_capacity_activations: 32,
            },
        ))
        .unwrap();
    let spec = VulkanResidentSamplerSpec {
        sampler_id: "runtime_sampler".to_string(),
        method: "temperature_top_k_top_p".to_string(),
        temperature: 1.0,
        top_k: 1,
        top_p: 1.0,
        min_p: 0.0,
        presence_penalty: 1.5,
        repetition_penalty: 1.0,
        top_k_capacity: TOP_K_CAPACITY,
        runtime_parameterized: true,
        logits_byte_capacity: LOGITS_BYTE_CAPACITY,
        output_byte_capacity: FIXTURE_MODEL_SAMPLER_OUTPUT_BYTES,
        scratch_byte_capacity: PARTITION_COUNT as usize
            * TOP_K_CAPACITY as usize
            * 2
            * std::mem::size_of::<u32>(),
    };
    let runner = VulkanResidentSamplerRunner::from_logits_buffer(
        &device,
        stream_control_buffer,
        &logits_buffer,
        LOGITS_BYTE_CAPACITY,
        &kernels,
        &spec,
        VulkanResidentSamplerStreamConfig {
            history_capacity_activations: 32,
            random_seed: 7,
        },
    )
    .unwrap();
    runner.record_input_tokens(&device, &[7]).unwrap();

    let batched_logits = device
        .create_resident_buffer(2 * LOGITS_BYTE_CAPACITY)
        .unwrap();
    let mut lane_zero = vec![-100.0f32; VOCAB_SIZE];
    lane_zero[8] = 10.0;
    lane_zero[11] = 9.0;
    let mut lane_one = vec![-100.0f32; VOCAB_SIZE];
    lane_one[10] = 10.0;
    lane_one[12] = 9.0;
    let batched_bytes = lane_zero
        .iter()
        .chain(&lane_one)
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    batched_logits.write_bytes(&batched_bytes).unwrap();

    let view_zero = runner
        .create_logits_view(&device, &batched_logits, 0, &kernels, &spec)
        .unwrap();
    view_zero.prepare_token_state(&device, &[8]).unwrap();
    view_zero.prepare_stream_tick(0, 32).unwrap();
    view_zero.record(&device).unwrap();
    device
        .run_recorded_resident_kernel_sequence(&view_zero.sequence)
        .unwrap();
    assert_eq!(runner.completed_run_at(0).unwrap().token_id, 11);

    let view_one = runner
        .create_logits_view(
            &device,
            &batched_logits,
            LOGITS_BYTE_CAPACITY,
            &kernels,
            &spec,
        )
        .unwrap();
    view_one.prepare_token_state(&device, &[9, 10]).unwrap();
    view_one.prepare_stream_tick(1, 32).unwrap();
    view_one.record(&device).unwrap();
    device
        .run_recorded_resident_kernel_sequence(&view_one.sequence)
        .unwrap();
    assert_eq!(runner.completed_run_at(1).unwrap().token_id, 12);

    logits_buffer
        .write_bytes(
            &lane_zero
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    assert_eq!(runner.run(&device).unwrap().token_id, 8);
}

#[test]
fn resident_temperature_top_64_sampler_matches_explicit_random_signal() {
    const VOCAB_SIZE: usize = 262_144;
    const TOP_K: u32 = 64;
    const PARTITION_COUNT: u32 = 128;
    const LOCAL_SIZE_X: u32 = 256;
    const SEED: u32 = 46;
    const HISTORY_CAPACITY: usize = 32;
    const LOGITS_BYTE_CAPACITY: usize = VOCAB_SIZE * std::mem::size_of::<f32>();

    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident top-64 sampler: {error}");
            return;
        }
    };
    let Some(sampler_kernels) = compile_temperature_top_k_top_p_sampler_test_kernels(
        VOCAB_SIZE,
        1.0,
        TOP_K,
        0.95,
        PARTITION_COUNT,
        LOCAL_SIZE_X,
    ) else {
        eprintln!("skipping resident top-64 sampler: no GLSL to SPIR-V compiler found");
        return;
    };

    let mut top_tokens = (0..TOP_K)
        .map(|index| (index * 4_093 + 7) % VOCAB_SIZE as u32)
        .collect::<Vec<_>>();
    top_tokens.sort_unstable();
    let mut logits = vec![-100.0f32; VOCAB_SIZE];
    for token_id in &top_tokens {
        logits[*token_id as usize] = 2.0;
    }
    let logits_buffer = device.create_resident_buffer(LOGITS_BYTE_CAPACITY).unwrap();
    logits_buffer
        .write_bytes(
            &logits
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let stream_control_buffer = Arc::new(
        device
            .create_host_visible_resident_buffer(VULKAN_STREAM_CONTROL_BYTE_CAPACITY)
            .unwrap(),
    );
    stream_control_buffer
        .write_bytes(&stream_control_bytes(
            0,
            VulkanMountedPlacedStreamControl {
                stream_tick: 0,
                control_flags: 0,
                dynamic_state_capacity_activations: HISTORY_CAPACITY as u32,
            },
        ))
        .unwrap();
    let spec = VulkanResidentSamplerSpec {
        sampler_id: "temperature_top_k_top_p_sampler".to_string(),
        method: "temperature_top_k_top_p".to_string(),
        temperature: 1.0,
        top_k: TOP_K,
        top_p: 0.95,
        min_p: 0.0,
        presence_penalty: 0.0,
        repetition_penalty: 1.0,
        top_k_capacity: TOP_K,
        runtime_parameterized: false,
        logits_byte_capacity: LOGITS_BYTE_CAPACITY,
        output_byte_capacity: FIXTURE_MODEL_SAMPLER_OUTPUT_BYTES,
        scratch_byte_capacity: PARTITION_COUNT as usize * TOP_K as usize * 8,
    };
    let runner = VulkanResidentSamplerRunner::from_logits_buffer(
        &device,
        stream_control_buffer,
        &logits_buffer,
        LOGITS_BYTE_CAPACITY,
        &sampler_kernels,
        &spec,
        VulkanResidentSamplerStreamConfig {
            history_capacity_activations: HISTORY_CAPACITY,
            random_seed: SEED,
        },
    )
    .unwrap();

    // With 64 equal top-k weights, top-p=0.95 retains the first 61.
    for stream_tick in 0..16u32 {
        let run = runner.run(&device).unwrap();
        let random_bits = sampler_test_hash_u32(SEED ^ stream_tick ^ 0xa511_e9b3);
        let selected_index = (((random_bits >> 8) as u64 * 61) >> 24) as usize;
        assert_eq!(run.token_id, top_tokens[selected_index]);
        assert_eq!(run.selected_logit_bits, 2.0f32.to_bits());
    }
}
