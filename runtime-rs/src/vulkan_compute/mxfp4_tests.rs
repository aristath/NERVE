#[cfg(test)]
mod mxfp4_tests {
    use super::*;

    const WIDTH: usize = 128;
    const WEIGHT_BYTES: usize = WIDTH * WIDTH / 2;
    const SCALE_BYTES: usize = WIDTH * WIDTH / 32;
    const FRAME_BYTES: usize = WIDTH * 2;

    #[test]
    fn native_mxfp4_expert_kernels_match_known_cpu_results() {
        let Some(raw_device_index) = std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX").ok() else {
            eprintln!("skipping native MXFP4 conformance: explicit Vulkan device index unset");
            return;
        };
        let device_index = raw_device_index
            .parse::<usize>()
            .expect("NERVE_TEST_VULKAN_DEVICE_INDEX must be an integer");
        let gate_shader = render_mxfp4_shader(
            "independent_sparse_moe_gate_up_mxfp4.comp.template",
            &[("{{TILE_ROWS}}", "32"), ("{{SWIGLU_LIMIT}}", "10.0")],
            false,
        );
        let prequant_gate_shader = render_mxfp4_shader(
            "independent_sparse_moe_gate_up_mxfp4.comp.template",
            &[("{{TILE_ROWS}}", "32"), ("{{SWIGLU_LIMIT}}", "10.0")],
            true,
        );
        let down_shader = render_mxfp4_shader(
            "independent_sparse_moe_down_mxfp4.comp.template",
            &[("{{TILE_ROWS}}", "64")],
            false,
        );
        let device = VulkanComputeDevice::new_for_physical_device_index(device_index)
            .expect("explicit idle AMD Vulkan device must open");
        assert!(device.supports_buffer_device_address());

        let hidden = device.create_resident_buffer(FRAME_BYTES).unwrap();
        hidden
            .write_bytes(&u16_bytes(&vec![f32_to_bf16_bits(1.0); WIDTH]))
            .unwrap();
        let routes = device.create_resident_buffer(4).unwrap();
        routes
            .write_bytes(&u32_bytes(&[u32::from(f32_to_bf16_bits(1.0)) << 16]))
            .unwrap();
        let intermediates = device.create_resident_buffer(FRAME_BYTES).unwrap();
        intermediates.write_bytes(&vec![0; FRAME_BYTES]).unwrap();

        let gate_weight = filled_addressable_buffer(&device, WEIGHT_BYTES, 0x22);
        let gate_scale = filled_addressable_buffer(&device, SCALE_BYTES, 120);
        let up_weight = filled_addressable_buffer(&device, WEIGHT_BYTES, 0x22);
        let up_scale = filled_addressable_buffer(&device, SCALE_BYTES, 120);
        let gate_addresses =
            address_table(&device, &[&gate_weight, &gate_scale, &up_weight, &up_scale]);
        let gate_slots = device.create_resident_buffer(16).unwrap();
        gate_slots.write_bytes(&u32_bytes(&[0, 1, 2, 3])).unwrap();
        let gate_dispatch = device
            .create_resident_kernel_dispatch(
                &gate_shader,
                &[
                    read_binding(0, &hidden, FRAME_BYTES),
                    read_binding(1, &routes, 4),
                    write_binding(2, &intermediates, FRAME_BYTES),
                    read_binding(3, &gate_addresses, 128),
                    read_binding(4, &gate_slots, 16),
                ],
                4,
                512,
                0,
            )
            .unwrap();
        device
            .run_resident_kernel_dispatch(&gate_dispatch, &[])
            .unwrap();

        let expected_activation = f32_to_bf16_bits(1.0 / (1.0 + (-1.0_f32).exp()));
        assert_eq!(
            intermediates.read_bytes(FRAME_BYTES).unwrap(),
            u16_bytes(&vec![expected_activation; WIDTH]),
            "native MXFP4 gate/up must preserve nibble order, E8M0 scales, matrix orientation, and SwiGLU"
        );

        let quantized_hidden = device.create_resident_buffer(WIDTH).unwrap();
        quantized_hidden.write_bytes(&vec![0x38; WIDTH]).unwrap();
        let hidden_scales = device.create_resident_buffer(4).unwrap();
        hidden_scales.write_bytes(&1.0_f32.to_le_bytes()).unwrap();
        intermediates.write_bytes(&vec![0; FRAME_BYTES]).unwrap();
        let prequant_gate_dispatch = device
            .create_resident_kernel_dispatch(
                &prequant_gate_shader,
                &[
                    read_binding(0, &quantized_hidden, WIDTH),
                    read_binding(1, &hidden_scales, 4),
                    read_binding(2, &routes, 4),
                    write_binding(3, &intermediates, FRAME_BYTES),
                    read_binding(4, &gate_addresses, 128),
                    read_binding(5, &gate_slots, 16),
                ],
                4,
                512,
                0,
            )
            .unwrap();
        device
            .run_resident_kernel_dispatch(&prequant_gate_dispatch, &[])
            .unwrap();
        assert_eq!(
            intermediates.read_bytes(FRAME_BYTES).unwrap(),
            u16_bytes(&vec![expected_activation; WIDTH]),
            "prequantized MXFP4 gate/up must consume the reusable FP8 physical activation without changing the result"
        );

        intermediates
            .write_bytes(&u16_bytes(&vec![f32_to_bf16_bits(1.0); WIDTH]))
            .unwrap();
        let outputs = device.create_resident_buffer(FRAME_BYTES).unwrap();
        outputs.write_bytes(&vec![0; FRAME_BYTES]).unwrap();
        let down_weight = filled_addressable_buffer(&device, WEIGHT_BYTES, 0x22);
        let down_scale = filled_addressable_buffer(&device, SCALE_BYTES, 120);
        let down_addresses = address_table(&device, &[&down_weight, &down_scale]);
        let down_slots = device.create_resident_buffer(8).unwrap();
        down_slots.write_bytes(&u32_bytes(&[0, 1])).unwrap();
        let down_dispatch = device
            .create_resident_kernel_dispatch(
                &down_shader,
                &[
                    read_binding(0, &intermediates, FRAME_BYTES),
                    read_binding(1, &routes, 4),
                    write_binding(2, &outputs, FRAME_BYTES),
                    read_binding(3, &down_addresses, 64),
                    read_binding(4, &down_slots, 8),
                ],
                2,
                512,
                0,
            )
            .unwrap();
        device
            .run_resident_kernel_dispatch(&down_dispatch, &[])
            .unwrap();
        assert_eq!(
            outputs.read_bytes(FRAME_BYTES).unwrap(),
            u16_bytes(&vec![f32_to_bf16_bits(1.0); WIDTH]),
            "native MXFP4 down projection must apply the route weight after the linear map"
        );

        let finite_e2m1_values = [
            0.0_f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0,
            -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        for (nibble, expected) in finite_e2m1_values.into_iter().enumerate() {
            let packed = u8::try_from(nibble | (nibble << 4)).unwrap();
            down_weight
                .write_bytes(&vec![packed; WEIGHT_BYTES])
                .unwrap();
            device
                .run_resident_kernel_dispatch(&down_dispatch, &[])
                .unwrap();
            assert_eq!(
                outputs.read_bytes(FRAME_BYTES).unwrap(),
                u16_bytes(&vec![f32_to_bf16_bits(expected); WIDTH]),
                "native MXFP4 down projection changed finite E2M1 code {nibble:#x}"
            );
        }
    }

    #[test]
    fn native_mxfp4_batch_expert_fails_closed_on_invalid_dynamic_metadata() {
        let Some(raw_device_index) = std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX").ok() else {
            eprintln!(
                "skipping native MXFP4 metadata guard: explicit Vulkan device index unset"
            );
            return;
        };
        let device_index = raw_device_index
            .parse::<usize>()
            .expect("NERVE_TEST_VULKAN_DEVICE_INDEX must be an integer");
        let gate_shader = render_mxfp4_shader(
            "independent_sparse_moe_gate_up_batch1_mxfp4.comp.template",
            &[("{{TILE_ROWS}}", "32"), ("{{SWIGLU_LIMIT}}", "10.0")],
            true,
        );
        let device = VulkanComputeDevice::new_for_physical_device_index(device_index)
            .expect("explicit idle AMD Vulkan device must open");

        let quantized_hidden = device.create_resident_buffer(WIDTH).unwrap();
        quantized_hidden.write_bytes(&vec![0x38; WIDTH]).unwrap();
        let hidden_scales = device.create_resident_buffer(4).unwrap();
        hidden_scales.write_bytes(&1.0_f32.to_le_bytes()).unwrap();
        let routes = device.create_resident_buffer(4).unwrap();
        routes
            .write_bytes(&u32_bytes(&[u32::from(f32_to_bf16_bits(1.0)) << 16]))
            .unwrap();
        let expert_frame_words = WIDTH / 2 + 1;
        let intermediates = device
            .create_resident_buffer(expert_frame_words * size_of::<u32>())
            .unwrap();

        let gate_weight = filled_addressable_buffer(&device, WEIGHT_BYTES, 0x22);
        let gate_scale = filled_addressable_buffer(&device, SCALE_BYTES, 120);
        let up_weight = filled_addressable_buffer(&device, WEIGHT_BYTES, 0x22);
        let up_scale = filled_addressable_buffer(&device, SCALE_BYTES, 120);
        let addresses = address_table(
            &device,
            &[&gate_weight, &gate_scale, &up_weight, &up_scale],
        );
        let slots = device.create_resident_buffer(16).unwrap();
        slots.write_bytes(&u32_bytes(&[0, 1, 2, 3])).unwrap();
        let batch_control = device.create_resident_buffer(28).unwrap();
        batch_control
            .write_bytes(&u32_bytes(&[1, 0, 0, 0, 0, 0, 0]))
            .unwrap();
        let dispatch = device
            .create_resident_kernel_dispatch_2d(
                &gate_shader,
                &[
                    read_binding(0, &quantized_hidden, WIDTH),
                    read_binding(1, &hidden_scales, 4),
                    read_binding(2, &routes, 4),
                    write_binding(
                        3,
                        &intermediates,
                        expert_frame_words * size_of::<u32>(),
                    ),
                    read_binding(4, &addresses, 128),
                    read_binding(5, &slots, 16),
                    read_binding(31, &batch_control, 28),
                ],
                4,
                1,
                512,
                0,
            )
            .unwrap();

        let valid_metadata = {
            let mut words = vec![0u32; expert_frame_words];
            words[WIDTH / 2] = 0;
            u32_bytes(&words)
        };
        intermediates.write_bytes(&valid_metadata).unwrap();
        device.run_resident_kernel_dispatch(&dispatch, &[]).unwrap();
        let expected_activation = f32_to_bf16_bits(1.0 / (1.0 + (-1.0_f32).exp()));
        assert_eq!(
            &intermediates.read_bytes(FRAME_BYTES).unwrap(),
            &u16_bytes(&vec![expected_activation; WIDTH]),
            "the guard must not alter a valid dynamic-resource execution"
        );

        let invalid_route_metadata = {
            let mut words = vec![0u32; expert_frame_words];
            words[WIDTH / 2] = u32::MAX;
            u32_bytes(&words)
        };
        intermediates.write_bytes(&invalid_route_metadata).unwrap();
        device.run_resident_kernel_dispatch(&dispatch, &[]).unwrap();
        assert_eq!(
            intermediates.read_bytes(FRAME_BYTES).unwrap(),
            vec![0; FRAME_BYTES],
            "out-of-range compact routes must not reach an address dereference"
        );

        intermediates.write_bytes(&valid_metadata).unwrap();
        addresses.write_bytes(&vec![0; 128]).unwrap();
        device.run_resident_kernel_dispatch(&dispatch, &[]).unwrap();
        assert_eq!(
            intermediates.read_bytes(FRAME_BYTES).unwrap(),
            vec![0; FRAME_BYTES],
            "unpublished address records must fail closed before buffer-reference access"
        );
    }

    #[test]
    fn native_mxfp4_batch_real_geometry_matches_and_finishes_under_one_minute() {
        let Some(raw_device_index) = std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX").ok() else {
            eprintln!(
                "skipping native MXFP4 real batch geometry: explicit Vulkan device index unset"
            );
            return;
        };
        let started = std::time::Instant::now();
        let device_index = raw_device_index
            .parse::<usize>()
            .expect("NERVE_TEST_VULKAN_DEVICE_INDEX must be an integer");
        let hidden_size = 4096usize;
        let intermediate_size = 2048usize;
        let num_experts = 256usize;
        let experts_per_token = 6usize;
        // Six lanes cover the full trained proposal width. An explicit smaller
        // width lets the same conformance test measure the adaptive selector's
        // actual execution shape without inventing a reduced tensor geometry.
        let batch_width = std::env::var("NERVE_TEST_MXFP4_BATCH_WIDTH")
            .ok()
            .map(|raw| {
                raw.parse::<usize>()
                    .expect("NERVE_TEST_MXFP4_BATCH_WIDTH must be an integer")
            })
            .unwrap_or(6);
        assert!(
            (1..=6).contains(&batch_width),
            "NERVE_TEST_MXFP4_BATCH_WIDTH must be in 1..=6"
        );
        let selected_experts = [2usize, 17, 63, 127, 191, 255];
        let gate_shader = render_mxfp4_shader_geometry(
            "independent_sparse_moe_gate_up_batch1_mxfp4.comp.template",
            hidden_size,
            intermediate_size,
            num_experts,
            experts_per_token,
            &[("{{TILE_ROWS}}", "32"), ("{{SWIGLU_LIMIT}}", "10.0")],
            true,
        );
        let down_shader = render_mxfp4_shader_geometry(
            "independent_sparse_moe_down_batch1_mxfp4.comp.template",
            hidden_size,
            intermediate_size,
            num_experts,
            experts_per_token,
            &[("{{TILE_ROWS}}", "64")],
            false,
        );
        let preexpanded_fp8_gate_shader = render_preexpanded_fp8_shader_geometry(
            "independent_sparse_moe_gate_up_batch1_mxfp4.comp.template",
            hidden_size,
            intermediate_size,
            num_experts,
            experts_per_token,
            &[("{{TILE_ROWS}}", "32"), ("{{SWIGLU_LIMIT}}", "10.0")],
            true,
        );
        let preexpanded_fp8_down_shader = render_preexpanded_fp8_shader_geometry(
            "independent_sparse_moe_down_batch1_mxfp4.comp.template",
            hidden_size,
            intermediate_size,
            num_experts,
            experts_per_token,
            &[("{{TILE_ROWS}}", "64")],
            false,
        );
        let device = VulkanComputeDevice::new_for_physical_device_index(device_index)
            .expect("explicit idle AMD Vulkan device must open");

        let hidden_fp8_words = hidden_size / 4;
        let hidden_blocks = hidden_size / 128;
        let quantized_hidden = device
            .create_resident_buffer(batch_width * hidden_size)
            .unwrap();
        quantized_hidden
            .write_bytes(&vec![0x38; batch_width * hidden_size])
            .unwrap();
        let hidden_scales = device
            .create_resident_buffer(batch_width * hidden_blocks * size_of::<f32>())
            .unwrap();
        hidden_scales
            .write_bytes(
                &(0..batch_width * hidden_blocks)
                    .flat_map(|_| 1.0_f32.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        assert_eq!(
            quantized_hidden.byte_capacity(),
            batch_width * hidden_fp8_words * size_of::<u32>()
        );

        let routes = device
            .create_resident_buffer(batch_width * experts_per_token * size_of::<u32>())
            .unwrap();
        let packed_routes = (0..batch_width)
            .flat_map(|batch| {
                (0..experts_per_token).map(move |route| {
                    selected_experts[(batch + route) % experts_per_token] as u32
                        | (u32::from(f32_to_bf16_bits(1.0)) << 16)
                })
            })
            .collect::<Vec<_>>();
        routes.write_bytes(&u32_bytes(&packed_routes)).unwrap();

        let intermediate_words = intermediate_size / 2;
        let expert_data_words = experts_per_token * intermediate_words;
        let expert_frame_words = expert_data_words + experts_per_token;
        let intermediates = device
            .create_resident_buffer(batch_width * expert_frame_words * size_of::<u32>())
            .unwrap();
        let mut intermediate_storage = vec![0u32; batch_width * expert_frame_words];
        for batch in 0..batch_width {
            for route in 0..experts_per_token {
                intermediate_storage
                    [batch * expert_frame_words + expert_data_words + route] =
                    (batch * experts_per_token + route) as u32;
            }
        }
        intermediates
            .write_bytes(&u32_bytes(&intermediate_storage))
            .unwrap();

        let gate_weight_bytes = hidden_size * intermediate_size / 2;
        let gate_scale_bytes = hidden_size * intermediate_size / 32;
        let gate_groups = (0..experts_per_token)
            .map(|_| {
                vec![
                    (gate_weight_bytes, 0x22),
                    (gate_scale_bytes, 120),
                    (gate_weight_bytes, 0x22),
                    (gate_scale_bytes, 120),
                ]
            })
            .collect::<Vec<_>>();
        let (_gate_arena, gate_addresses) =
            stable_resource_table(&device, &gate_groups, 256);
        let gate_slots = device
            .create_resident_buffer(num_experts * 4 * size_of::<u32>())
            .unwrap();
        let mut gate_slot_words = vec![u32::MAX; num_experts * 4];
        for (active_index, expert) in selected_experts.iter().copied().enumerate() {
            for parameter in 0..4 {
                gate_slot_words[expert * 4 + parameter] =
                    u32::try_from(active_index * 4 + parameter).unwrap();
            }
        }
        gate_slots.write_bytes(&u32_bytes(&gate_slot_words)).unwrap();

        let gate_dispatch_x = experts_per_token * intermediate_size.div_ceil(32);
        let gate_batch_control = device.create_resident_buffer(28).unwrap();
        gate_batch_control
            .write_bytes(&u32_bytes(&[
                batch_width as u32,
                0,
                0,
                0,
                gate_dispatch_x as u32,
                batch_width as u32,
                1,
            ]))
            .unwrap();
        let gate_dispatch = device
            .create_resident_kernel_dispatch_2d(
                &gate_shader,
                &[
                    read_binding(0, &quantized_hidden, batch_width * hidden_size),
                    read_binding(
                        1,
                        &hidden_scales,
                        batch_width * hidden_blocks * size_of::<f32>(),
                    ),
                    read_binding(
                        2,
                        &routes,
                        batch_width * experts_per_token * size_of::<u32>(),
                    ),
                    write_binding(
                        3,
                        &intermediates,
                        batch_width * expert_frame_words * size_of::<u32>(),
                    ),
                    read_binding(4, gate_addresses.buffer(), gate_addresses.byte_capacity()),
                    read_binding(5, &gate_slots, num_experts * 4 * size_of::<u32>()),
                    read_binding(31, &gate_batch_control, 28),
                ],
                gate_dispatch_x as u32,
                batch_width as u32,
                512,
                0,
            )
            .unwrap();
        device
            .run_resident_kernel_dispatch(&gate_dispatch, &[])
            .unwrap();
        let after_gate = intermediates.read_bytes(intermediates.byte_capacity()).unwrap();
        let after_gate_words = after_gate
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
            .collect::<Vec<_>>();
        for batch in 0..batch_width {
            for route in 0..experts_per_token {
                assert_eq!(
                    after_gate_words[batch * expert_frame_words + expert_data_words + route],
                    (batch * experts_per_token + route) as u32,
                    "batch compaction metadata must survive real-geometry gate/up"
                );
            }
        }

        let outputs = device
            .create_resident_buffer(
                batch_width * experts_per_token * hidden_size * size_of::<u16>(),
            )
            .unwrap();
        outputs
            .write_bytes(&vec![0; outputs.byte_capacity()])
            .unwrap();
        let down_weight_bytes = hidden_size * intermediate_size / 2;
        let down_scale_bytes = hidden_size * intermediate_size / 32;
        let down_groups = (0..experts_per_token)
            .map(|_| vec![(down_weight_bytes, 0x22), (down_scale_bytes, 120)])
            .collect::<Vec<_>>();
        let (_down_arena, down_addresses) =
            stable_resource_table(&device, &down_groups, 256);
        let down_slots = device
            .create_resident_buffer(num_experts * 2 * size_of::<u32>())
            .unwrap();
        let mut down_slot_words = vec![u32::MAX; num_experts * 2];
        for (active_index, expert) in selected_experts.iter().copied().enumerate() {
            for parameter in 0..2 {
                down_slot_words[expert * 2 + parameter] =
                    u32::try_from(active_index * 2 + parameter).unwrap();
            }
        }
        down_slots.write_bytes(&u32_bytes(&down_slot_words)).unwrap();
        let down_dispatch_x = experts_per_token * hidden_size.div_ceil(64);
        let down_batch_control = device.create_resident_buffer(28).unwrap();
        down_batch_control
            .write_bytes(&u32_bytes(&[
                batch_width as u32,
                0,
                0,
                0,
                down_dispatch_x as u32,
                batch_width as u32,
                1,
            ]))
            .unwrap();
        let down_dispatch = device
            .create_resident_kernel_dispatch_2d(
                &down_shader,
                &[
                    read_binding(
                        0,
                        &intermediates,
                        batch_width * expert_frame_words * size_of::<u32>(),
                    ),
                    read_binding(
                        1,
                        &routes,
                        batch_width * experts_per_token * size_of::<u32>(),
                    ),
                    write_binding(2, &outputs, outputs.byte_capacity()),
                    read_binding(3, down_addresses.buffer(), down_addresses.byte_capacity()),
                    read_binding(4, &down_slots, num_experts * 2 * size_of::<u32>()),
                    read_binding(31, &down_batch_control, 28),
                ],
                down_dispatch_x as u32,
                batch_width as u32,
                512,
                0,
            )
            .unwrap();
        device
            .run_resident_kernel_dispatch(&down_dispatch, &[])
            .unwrap();
        let native_output_bytes = outputs.read_bytes(outputs.byte_capacity()).unwrap();
        assert!(
            native_output_bytes.iter().any(|byte| *byte != 0),
            "real-geometry batched MXFP4 execution must produce routed expert output"
        );

        let gate_sequence = device
            .create_timestamped_resident_kernel_sequence()
            .unwrap();
        device
            .record_resident_kernel_sequence(
                &gate_sequence,
                &[VulkanResidentKernelSequenceStep::new(&gate_dispatch, &[])],
            )
            .unwrap();
        let down_sequence = device
            .create_timestamped_resident_kernel_sequence()
            .unwrap();
        device
            .record_resident_kernel_sequence(
                &down_sequence,
                &[VulkanResidentKernelSequenceStep::new(&down_dispatch, &[])],
            )
            .unwrap();
        let timeout = Duration::from_secs(10);
        device
            .run_timestamped_recorded_resident_kernel_sequence_for(&gate_sequence, timeout)
            .unwrap();
        device
            .run_timestamped_recorded_resident_kernel_sequence_for(&down_sequence, timeout)
            .unwrap();
        let gate_ns = device
            .run_timestamped_recorded_resident_kernel_sequence_for(&gate_sequence, timeout)
            .unwrap();
        let down_ns = device
            .run_timestamped_recorded_resident_kernel_sequence_for(&down_sequence, timeout)
            .unwrap();

        intermediates
            .write_bytes(&u32_bytes(&intermediate_storage))
            .unwrap();
        outputs
            .write_bytes(&vec![0; outputs.byte_capacity()])
            .unwrap();
        let preexpanded_gate_weight_bytes = hidden_size * intermediate_size;
        let preexpanded_gate_groups = (0..experts_per_token)
            .map(|_| {
                vec![
                    (preexpanded_gate_weight_bytes, 0x38),
                    (gate_scale_bytes, 120),
                    (preexpanded_gate_weight_bytes, 0x38),
                    (gate_scale_bytes, 120),
                ]
            })
            .collect::<Vec<_>>();
        let (_preexpanded_gate_arena, preexpanded_gate_addresses) =
            stable_resource_table(&device, &preexpanded_gate_groups, 256);
        let preexpanded_gate_dispatch = device
            .create_resident_kernel_dispatch_2d(
                &preexpanded_fp8_gate_shader,
                &[
                    read_binding(0, &quantized_hidden, batch_width * hidden_size),
                    read_binding(
                        1,
                        &hidden_scales,
                        batch_width * hidden_blocks * size_of::<f32>(),
                    ),
                    read_binding(
                        2,
                        &routes,
                        batch_width * experts_per_token * size_of::<u32>(),
                    ),
                    write_binding(
                        3,
                        &intermediates,
                        batch_width * expert_frame_words * size_of::<u32>(),
                    ),
                    read_binding(
                        4,
                        preexpanded_gate_addresses.buffer(),
                        preexpanded_gate_addresses.byte_capacity(),
                    ),
                    read_binding(5, &gate_slots, num_experts * 4 * size_of::<u32>()),
                    read_binding(31, &gate_batch_control, 28),
                ],
                gate_dispatch_x as u32,
                batch_width as u32,
                512,
                0,
            )
            .unwrap();
        let preexpanded_down_weight_bytes = hidden_size * intermediate_size;
        let preexpanded_down_groups = (0..experts_per_token)
            .map(|_| {
                vec![
                    (preexpanded_down_weight_bytes, 0x38),
                    (down_scale_bytes, 120),
                ]
            })
            .collect::<Vec<_>>();
        let (_preexpanded_down_arena, preexpanded_down_addresses) =
            stable_resource_table(&device, &preexpanded_down_groups, 256);
        let preexpanded_down_dispatch = device
            .create_resident_kernel_dispatch_2d(
                &preexpanded_fp8_down_shader,
                &[
                    read_binding(
                        0,
                        &intermediates,
                        batch_width * expert_frame_words * size_of::<u32>(),
                    ),
                    read_binding(
                        1,
                        &routes,
                        batch_width * experts_per_token * size_of::<u32>(),
                    ),
                    write_binding(2, &outputs, outputs.byte_capacity()),
                    read_binding(
                        3,
                        preexpanded_down_addresses.buffer(),
                        preexpanded_down_addresses.byte_capacity(),
                    ),
                    read_binding(4, &down_slots, num_experts * 2 * size_of::<u32>()),
                    read_binding(31, &down_batch_control, 28),
                ],
                down_dispatch_x as u32,
                batch_width as u32,
                512,
                0,
            )
            .unwrap();
        device
            .run_resident_kernel_dispatch(&preexpanded_gate_dispatch, &[])
            .unwrap();
        device
            .run_resident_kernel_dispatch(&preexpanded_down_dispatch, &[])
            .unwrap();
        assert_eq!(
            outputs.read_bytes(outputs.byte_capacity()).unwrap(),
            native_output_bytes,
            "pre-expanded FP8 sparse experts must preserve native MXFP4 BF16 boundaries"
        );
        let preexpanded_gate_sequence = device
            .create_timestamped_resident_kernel_sequence()
            .unwrap();
        device
            .record_resident_kernel_sequence(
                &preexpanded_gate_sequence,
                &[VulkanResidentKernelSequenceStep::new(
                    &preexpanded_gate_dispatch,
                    &[],
                )],
            )
            .unwrap();
        let preexpanded_down_sequence = device
            .create_timestamped_resident_kernel_sequence()
            .unwrap();
        device
            .record_resident_kernel_sequence(
                &preexpanded_down_sequence,
                &[VulkanResidentKernelSequenceStep::new(
                    &preexpanded_down_dispatch,
                    &[],
                )],
            )
            .unwrap();
        device
            .run_timestamped_recorded_resident_kernel_sequence_for(
                &preexpanded_gate_sequence,
                timeout,
            )
            .unwrap();
        device
            .run_timestamped_recorded_resident_kernel_sequence_for(
                &preexpanded_down_sequence,
                timeout,
            )
            .unwrap();
        let preexpanded_gate_ns = device
            .run_timestamped_recorded_resident_kernel_sequence_for(
                &preexpanded_gate_sequence,
                timeout,
            )
            .unwrap();
        let preexpanded_down_ns = device
            .run_timestamped_recorded_resident_kernel_sequence_for(
                &preexpanded_down_sequence,
                timeout,
            )
            .unwrap();
        eprintln!(
            "native_mxfp4_batch_real_geometry width={batch_width} routes={} gate_ms={:.6} down_ms={:.6} total_ms={:.6} preexpanded_fp8_gate_ms={:.6} preexpanded_fp8_down_ms={:.6} preexpanded_fp8_total_ms={:.6} ratio={:.6} elapsed_ms={:.3}",
            batch_width * experts_per_token,
            gate_ns as f64 / 1_000_000.0,
            down_ns as f64 / 1_000_000.0,
            (gate_ns + down_ns) as f64 / 1_000_000.0,
            preexpanded_gate_ns as f64 / 1_000_000.0,
            preexpanded_down_ns as f64 / 1_000_000.0,
            (preexpanded_gate_ns + preexpanded_down_ns) as f64 / 1_000_000.0,
            (preexpanded_gate_ns + preexpanded_down_ns) as f64 / (gate_ns + down_ns) as f64,
            started.elapsed().as_secs_f64() * 1_000.0,
        );
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "native MXFP4 real batch geometry exceeded one minute"
        );
    }

    #[test]
    fn native_mxfp4_real_geometry_microbenchmark_finishes_under_one_minute() {
        let Some(raw_device_index) = std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX").ok() else {
            eprintln!(
                "skipping native MXFP4 real-geometry microbenchmark: explicit Vulkan device index unset"
            );
            return;
        };
        let started = std::time::Instant::now();
        let device_index = raw_device_index
            .parse::<usize>()
            .expect("NERVE_TEST_VULKAN_DEVICE_INDEX must be an integer");
        let hidden_size = 4096usize;
        let intermediate_size = 2048usize;
        let expert_count = 6usize;
        let gate_shader = render_mxfp4_shader_geometry(
            "independent_sparse_moe_gate_up_mxfp4.comp.template",
            hidden_size,
            intermediate_size,
            expert_count,
            expert_count,
            &[("{{TILE_ROWS}}", "32"), ("{{SWIGLU_LIMIT}}", "10.0")],
            true,
        );
        let down_shader = render_mxfp4_shader_geometry(
            "independent_sparse_moe_down_mxfp4.comp.template",
            hidden_size,
            intermediate_size,
            expert_count,
            expert_count,
            &[("{{TILE_ROWS}}", "64")],
            false,
        );
        let device = VulkanComputeDevice::new_for_physical_device_index(device_index)
            .expect("explicit idle AMD Vulkan device must open");

        let quantized_hidden = device.create_resident_buffer(hidden_size).unwrap();
        quantized_hidden
            .write_bytes(&vec![0x38; hidden_size])
            .unwrap();
        let hidden_scales = device
            .create_resident_buffer(hidden_size / 128 * size_of::<f32>())
            .unwrap();
        hidden_scales
            .write_bytes(
                &(0..hidden_size / 128)
                    .flat_map(|_| 1.0_f32.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let routes = device
            .create_resident_buffer(expert_count * size_of::<u32>())
            .unwrap();
        routes
            .write_bytes(&u32_bytes(
                &(0..expert_count)
                    .map(|expert| expert as u32 | (u32::from(f32_to_bf16_bits(1.0)) << 16))
                    .collect::<Vec<_>>(),
            ))
            .unwrap();
        let expert_data_words = expert_count * intermediate_size / 2;
        let expert_frame_words = expert_data_words + expert_count;
        let intermediates = device
            .create_resident_buffer(expert_frame_words * size_of::<u32>())
            .unwrap();
        let mut intermediate_words = vec![0u32; expert_frame_words];
        for route in 0..expert_count {
            intermediate_words[expert_data_words + route] = route as u32;
        }
        intermediates
            .write_bytes(&u32_bytes(&intermediate_words))
            .unwrap();

        let gate_weight_bytes = hidden_size * intermediate_size / 2;
        let gate_scale_bytes = hidden_size * intermediate_size / 32;
        let mut gate_resources = Vec::with_capacity(expert_count * 4);
        for _ in 0..expert_count {
            gate_resources.push(filled_addressable_buffer(&device, gate_weight_bytes, 0x22));
            gate_resources.push(filled_addressable_buffer(&device, gate_scale_bytes, 120));
            gate_resources.push(filled_addressable_buffer(&device, gate_weight_bytes, 0x22));
            gate_resources.push(filled_addressable_buffer(&device, gate_scale_bytes, 120));
        }
        let gate_resource_refs = gate_resources.iter().collect::<Vec<_>>();
        let gate_addresses = address_table(&device, &gate_resource_refs);
        let gate_slots = device
            .create_resident_buffer(expert_count * 4 * size_of::<u32>())
            .unwrap();
        gate_slots
            .write_bytes(&u32_bytes(
                &(0..expert_count * 4)
                    .map(|slot| slot as u32)
                    .collect::<Vec<_>>(),
            ))
            .unwrap();
        let gate_dispatch = device
            .create_resident_kernel_dispatch(
                &gate_shader,
                &[
                    read_binding(0, &quantized_hidden, hidden_size),
                    read_binding(1, &hidden_scales, hidden_size / 128 * size_of::<f32>()),
                    read_binding(2, &routes, expert_count * size_of::<u32>()),
                    write_binding(3, &intermediates, expert_frame_words * size_of::<u32>()),
                    read_binding(4, &gate_addresses, expert_count * 4 * 32),
                    read_binding(5, &gate_slots, expert_count * 4 * size_of::<u32>()),
                ],
                u32::try_from(expert_count * intermediate_size / 32).unwrap(),
                512,
                0,
            )
            .unwrap();

        let outputs = device
            .create_resident_buffer(expert_count * hidden_size * size_of::<u16>())
            .unwrap();
        let down_weight_bytes = hidden_size * intermediate_size / 2;
        let down_scale_bytes = hidden_size * intermediate_size / 32;
        let mut down_resources = Vec::with_capacity(expert_count * 2);
        for _ in 0..expert_count {
            down_resources.push(filled_addressable_buffer(&device, down_weight_bytes, 0x22));
            down_resources.push(filled_addressable_buffer(&device, down_scale_bytes, 120));
        }
        let down_resource_refs = down_resources.iter().collect::<Vec<_>>();
        let down_addresses = address_table(&device, &down_resource_refs);
        let down_slots = device
            .create_resident_buffer(expert_count * 2 * size_of::<u32>())
            .unwrap();
        down_slots
            .write_bytes(&u32_bytes(
                &(0..expert_count * 2)
                    .map(|slot| slot as u32)
                    .collect::<Vec<_>>(),
            ))
            .unwrap();
        let down_dispatch = device
            .create_resident_kernel_dispatch(
                &down_shader,
                &[
                    read_binding(0, &intermediates, expert_frame_words * size_of::<u32>()),
                    read_binding(1, &routes, expert_count * size_of::<u32>()),
                    write_binding(2, &outputs, expert_count * hidden_size * size_of::<u16>()),
                    read_binding(3, &down_addresses, expert_count * 2 * 32),
                    read_binding(4, &down_slots, expert_count * 2 * size_of::<u32>()),
                ],
                u32::try_from(expert_count * hidden_size / 64).unwrap(),
                512,
                0,
            )
            .unwrap();

        let gate_sequence = device
            .create_timestamped_resident_kernel_sequence()
            .unwrap();
        device
            .record_resident_kernel_sequence(
                &gate_sequence,
                &[VulkanResidentKernelSequenceStep::new(&gate_dispatch, &[])],
            )
            .unwrap();
        let down_sequence = device
            .create_timestamped_resident_kernel_sequence()
            .unwrap();
        device
            .record_resident_kernel_sequence(
                &down_sequence,
                &[VulkanResidentKernelSequenceStep::new(&down_dispatch, &[])],
            )
            .unwrap();
        let timeout = Duration::from_secs(10);
        device
            .run_timestamped_recorded_resident_kernel_sequence_for(&gate_sequence, timeout)
            .unwrap();
        device
            .run_timestamped_recorded_resident_kernel_sequence_for(&down_sequence, timeout)
            .unwrap();
        let gate_ns = device
            .run_timestamped_recorded_resident_kernel_sequence_for(&gate_sequence, timeout)
            .unwrap();
        let down_ns = device
            .run_timestamped_recorded_resident_kernel_sequence_for(&down_sequence, timeout)
            .unwrap();

        let mut host_gate_resources = Vec::with_capacity(expert_count * 4);
        for _ in 0..expert_count {
            host_gate_resources.push(filled_addressable_buffer_in_domain(
                &device,
                gate_weight_bytes,
                0x22,
                true,
            ));
            host_gate_resources.push(filled_addressable_buffer_in_domain(
                &device,
                gate_scale_bytes,
                120,
                true,
            ));
            host_gate_resources.push(filled_addressable_buffer_in_domain(
                &device,
                gate_weight_bytes,
                0x22,
                true,
            ));
            host_gate_resources.push(filled_addressable_buffer_in_domain(
                &device,
                gate_scale_bytes,
                120,
                true,
            ));
        }
        let host_gate_resource_refs = host_gate_resources.iter().collect::<Vec<_>>();
        let host_gate_addresses = address_table(&device, &host_gate_resource_refs);
        let host_gate_dispatch = device
            .create_resident_kernel_dispatch(
                &gate_shader,
                &[
                    read_binding(0, &quantized_hidden, hidden_size),
                    read_binding(1, &hidden_scales, hidden_size / 128 * size_of::<f32>()),
                    read_binding(2, &routes, expert_count * size_of::<u32>()),
                    write_binding(3, &intermediates, expert_frame_words * size_of::<u32>()),
                    read_binding(4, &host_gate_addresses, expert_count * 4 * 32),
                    read_binding(5, &gate_slots, expert_count * 4 * size_of::<u32>()),
                ],
                u32::try_from(expert_count * intermediate_size / 32).unwrap(),
                512,
                0,
            )
            .unwrap();
        let mut host_down_resources = Vec::with_capacity(expert_count * 2);
        for _ in 0..expert_count {
            host_down_resources.push(filled_addressable_buffer_in_domain(
                &device,
                down_weight_bytes,
                0x22,
                true,
            ));
            host_down_resources.push(filled_addressable_buffer_in_domain(
                &device,
                down_scale_bytes,
                120,
                true,
            ));
        }
        let host_down_resource_refs = host_down_resources.iter().collect::<Vec<_>>();
        let host_down_addresses = address_table(&device, &host_down_resource_refs);
        let host_down_dispatch = device
            .create_resident_kernel_dispatch(
                &down_shader,
                &[
                    read_binding(0, &intermediates, expert_frame_words * size_of::<u32>()),
                    read_binding(1, &routes, expert_count * size_of::<u32>()),
                    write_binding(2, &outputs, expert_count * hidden_size * size_of::<u16>()),
                    read_binding(3, &host_down_addresses, expert_count * 2 * 32),
                    read_binding(4, &down_slots, expert_count * 2 * size_of::<u32>()),
                ],
                u32::try_from(expert_count * hidden_size / 64).unwrap(),
                512,
                0,
            )
            .unwrap();
        let host_gate_sequence = device
            .create_timestamped_resident_kernel_sequence()
            .unwrap();
        device
            .record_resident_kernel_sequence(
                &host_gate_sequence,
                &[VulkanResidentKernelSequenceStep::new(
                    &host_gate_dispatch,
                    &[],
                )],
            )
            .unwrap();
        let host_down_sequence = device
            .create_timestamped_resident_kernel_sequence()
            .unwrap();
        device
            .record_resident_kernel_sequence(
                &host_down_sequence,
                &[VulkanResidentKernelSequenceStep::new(
                    &host_down_dispatch,
                    &[],
                )],
            )
            .unwrap();
        device
            .run_timestamped_recorded_resident_kernel_sequence_for(&host_gate_sequence, timeout)
            .unwrap();
        device
            .run_timestamped_recorded_resident_kernel_sequence_for(&host_down_sequence, timeout)
            .unwrap();
        let host_gate_ns = device
            .run_timestamped_recorded_resident_kernel_sequence_for(&host_gate_sequence, timeout)
            .unwrap();
        let host_down_ns = device
            .run_timestamped_recorded_resident_kernel_sequence_for(&host_down_sequence, timeout)
            .unwrap();
        eprintln!(
            "native_mxfp4_real_geometry device_gate_ms={:.6} device_down_ms={:.6} device_total_ms={:.6} host_gate_ms={:.6} host_down_ms={:.6} host_total_ms={:.6} host_to_device_ratio={:.6} elapsed_ms={:.3}",
            gate_ns as f64 / 1_000_000.0,
            down_ns as f64 / 1_000_000.0,
            (gate_ns + down_ns) as f64 / 1_000_000.0,
            host_gate_ns as f64 / 1_000_000.0,
            host_down_ns as f64 / 1_000_000.0,
            (host_gate_ns + host_down_ns) as f64 / 1_000_000.0,
            (host_gate_ns + host_down_ns) as f64 / (gate_ns + down_ns) as f64,
            started.elapsed().as_secs_f64() * 1_000.0,
        );
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "native MXFP4 real-geometry microbenchmark exceeded one minute"
        );
    }

    #[test]
    fn selected_expert_host_to_device_copy_microbenchmark_finishes_under_one_minute() {
        let Some(raw_device_index) = std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX").ok() else {
            eprintln!(
                "skipping selected-expert copy microbenchmark: explicit Vulkan device index unset"
            );
            return;
        };
        let started = std::time::Instant::now();
        let device_index = raw_device_index
            .parse::<usize>()
            .expect("NERVE_TEST_VULKAN_DEVICE_INDEX must be an integer");
        let device = VulkanComputeDevice::new_for_physical_device_index(device_index)
            .expect("explicit idle AMD Vulkan device must open");
        let expert_group_bytes = 3 * (4_194_304 + 262_144);
        let selected_expert_bytes = 6 * expert_group_bytes;
        let source = device
            .create_host_visible_addressable_resident_buffer(selected_expert_bytes)
            .unwrap();
        source
            .write_bytes(&vec![0x5a; selected_expert_bytes])
            .unwrap();
        let destination = device
            .create_addressable_resident_buffer(selected_expert_bytes)
            .unwrap();
        let copy = device
            .create_timestamped_resident_buffer_copy(&source, &destination, selected_expert_bytes)
            .unwrap();
        copy.run_with_device_duration(selected_expert_bytes)
            .unwrap();
        let duration_ns = copy
            .run_with_device_duration(selected_expert_bytes)
            .unwrap();
        let gib_per_second = selected_expert_bytes as f64 / duration_ns as f64 * 1_000_000_000.0
            / 1024.0_f64.powi(3);
        eprintln!(
            "selected_expert_host_to_device_copy bytes={selected_expert_bytes} duration_ms={:.6} gib_per_second={gib_per_second:.6} elapsed_ms={:.3}",
            duration_ns as f64 / 1_000_000.0,
            started.elapsed().as_secs_f64() * 1_000.0,
        );
        assert!(
            duration_ns < 30_000_000,
            "one layer's selected experts regressed beyond the measured 30 ms copy ceiling"
        );
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "selected-expert copy microbenchmark exceeded one minute"
        );
    }

    fn render_mxfp4_shader(
        template_name: &str,
        stage_replacements: &[(&str, &str)],
        prequantized: bool,
    ) -> Vec<u32> {
        let shader_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders");
        let mut source = std::fs::read_to_string(shader_dir.join(template_name)).unwrap();
        for (pattern, value) in [
            ("{{HIDDEN_SIZE}}", "128"),
            ("{{INTERMEDIATE_SIZE}}", "128"),
            ("{{NUM_EXPERTS}}", "1"),
            ("{{EXPERTS_PER_TOKEN}}", "1"),
            (
                "{{PREQUANTIZED_INPUT}}",
                if prequantized { "1" } else { "0" },
            ),
        ]
        .into_iter()
        .chain(stage_replacements.iter().copied())
        {
            source = source.replace(pattern, value);
        }
        if template_name.contains("_batch1_") {
            source = source.replace(
                "layout(push_constant) uniform BatchControl",
                "layout(set = 0, binding = 31) readonly buffer BatchControl",
            );
        }
        let source_path = std::env::temp_dir().join(format!(
            "nerve-test-{}-{}.comp",
            template_name.replace(['/', '.'], "-"),
            std::process::id()
        ));
        std::fs::write(&source_path, source).unwrap();
        let words = compile_shader_words_from_source_path(&source_path)
            .unwrap_or_else(|| panic!("{template_name} must compile"));
        let _ = std::fs::remove_file(source_path);
        words
    }

    fn render_mxfp4_shader_geometry(
        template_name: &str,
        hidden_size: usize,
        intermediate_size: usize,
        num_experts: usize,
        experts_per_token: usize,
        stage_replacements: &[(&str, &str)],
        prequantized: bool,
    ) -> Vec<u32> {
        let source = render_mxfp4_shader_source(
            template_name,
            hidden_size,
            intermediate_size,
            num_experts,
            experts_per_token,
            stage_replacements,
            prequantized,
        );
        compile_rendered_shader(template_name, source)
    }

    fn render_preexpanded_fp8_shader_geometry(
        template_name: &str,
        hidden_size: usize,
        intermediate_size: usize,
        num_experts: usize,
        experts_per_token: usize,
        stage_replacements: &[(&str, &str)],
        prequantized: bool,
    ) -> Vec<u32> {
        let mut source = render_mxfp4_shader_source(
            template_name,
            hidden_size,
            intermediate_size,
            num_experts,
            experts_per_token,
            stage_replacements,
            prequantized,
        )
        .replace(
            "const uint WEIGHT_ROW_BYTES = HIDDEN_SIZE / 2u;",
            "const uint WEIGHT_ROW_BYTES = HIDDEN_SIZE;",
        )
        .replace(
            "const uint WEIGHT_ROW_BYTES = INTERMEDIATE_SIZE / 2u;",
            "const uint WEIGHT_ROW_BYTES = INTERMEDIATE_SIZE;",
        );
        let function_start = source
            .find("fe4m3vec4 read_mxfp4x4(")
            .expect("MXFP4 shader must contain its weight decoder");
        let function_end = source[function_start..]
            .find("\n}\n\nfloat read_e8m0_scale")
            .map(|offset| function_start + offset + 2)
            .expect("MXFP4 shader weight decoder must end before its scale reader");
        source.replace_range(
            function_start..function_end,
            "fe4m3vec4 read_mxfp4x4(DynamicU32Buffer weight, uint row, uint column) {\n    uint packed = weight.words[(row * WEIGHT_ROW_BYTES + column) >> 2u];\n    return uintBitsToFloate4m3EXT(u8vec4(\n        uint8_t(packed),\n        uint8_t(packed >> 8u),\n        uint8_t(packed >> 16u),\n        uint8_t(packed >> 24u)\n    ));\n}",
        );
        compile_rendered_shader(template_name, source)
    }

    fn render_mxfp4_shader_source(
        template_name: &str,
        hidden_size: usize,
        intermediate_size: usize,
        num_experts: usize,
        experts_per_token: usize,
        stage_replacements: &[(&str, &str)],
        prequantized: bool,
    ) -> String {
        let shader_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders");
        let mut source = std::fs::read_to_string(shader_dir.join(template_name)).unwrap();
        let hidden_size = hidden_size.to_string();
        let intermediate_size = intermediate_size.to_string();
        let num_experts = num_experts.to_string();
        let experts_per_token = experts_per_token.to_string();
        for (pattern, value) in [
            ("{{HIDDEN_SIZE}}", hidden_size.as_str()),
            ("{{INTERMEDIATE_SIZE}}", intermediate_size.as_str()),
            ("{{NUM_EXPERTS}}", num_experts.as_str()),
            ("{{EXPERTS_PER_TOKEN}}", experts_per_token.as_str()),
            (
                "{{PREQUANTIZED_INPUT}}",
                if prequantized { "1" } else { "0" },
            ),
        ]
        .into_iter()
        .chain(stage_replacements.iter().copied())
        {
            source = source.replace(pattern, value);
        }
        if template_name.contains("_batch1_") {
            source = source.replace(
                "layout(push_constant) uniform BatchControl",
                "layout(set = 0, binding = 31) readonly buffer BatchControl",
            );
        }
        source
    }

    fn compile_rendered_shader(template_name: &str, source: String) -> Vec<u32> {
        let source_path = std::env::temp_dir().join(format!(
            "nerve-test-{}-{}.comp",
            template_name.replace(['/', '.'], "-"),
            std::process::id()
        ));
        std::fs::write(&source_path, source).unwrap();
        let words = compile_shader_words_from_source_path(&source_path)
            .unwrap_or_else(|| panic!("{template_name} must compile"));
        let _ = std::fs::remove_file(source_path);
        words
    }

    fn filled_addressable_buffer(
        device: &VulkanComputeDevice,
        byte_count: usize,
        value: u8,
    ) -> VulkanResidentBuffer {
        filled_addressable_buffer_in_domain(device, byte_count, value, false)
    }

    fn filled_addressable_buffer_in_domain(
        device: &VulkanComputeDevice,
        byte_count: usize,
        value: u8,
        host_visible: bool,
    ) -> VulkanResidentBuffer {
        let buffer = if host_visible {
            device
                .create_host_visible_addressable_resident_buffer(byte_count)
                .unwrap()
        } else {
            device
                .create_addressable_resident_buffer(byte_count)
                .unwrap()
        };
        buffer.write_bytes(&vec![value; byte_count]).unwrap();
        buffer
    }

    fn address_table(
        device: &VulkanComputeDevice,
        resources: &[&VulkanResidentBuffer],
    ) -> VulkanResidentBuffer {
        let byte_count = resources.len() * 32;
        let table = device.create_resident_buffer(byte_count).unwrap();
        let mut words = vec![0u32; resources.len() * 8];
        for (slot, resource) in resources.iter().enumerate() {
            let address = resource.device_address().unwrap();
            let byte_count = resource.byte_capacity() as u64;
            words[slot * 8] = address as u32;
            words[slot * 8 + 1] = (address >> 32) as u32;
            words[slot * 8 + 2] = byte_count as u32;
            words[slot * 8 + 3] = (byte_count >> 32) as u32;
            words[slot * 8 + 4] = 1;
            words[slot * 8 + 6] = 1;
        }
        table.write_bytes(&u32_bytes(&words)).unwrap();
        table
    }

    fn stable_resource_table(
        device: &VulkanComputeDevice,
        groups: &[Vec<(usize, u8)>],
        alignment: usize,
    ) -> (VulkanStableResourceArena, VulkanStableResourceAddressTable) {
        let mut next_slot = 0usize;
        let layouts = groups
            .iter()
            .map(|group| {
                let resource_slots = (next_slot..next_slot + group.len()).collect::<Vec<_>>();
                next_slot += group.len();
                VulkanStableResourceGroupLayout::Explicit {
                    resource_slots,
                    resource_byte_counts: group.iter().map(|(bytes, _)| *bytes).collect(),
                }
            })
            .collect::<Vec<_>>();
        let payload_bytes = groups
            .iter()
            .flatten()
            .map(|(bytes, _)| *bytes)
            .sum::<usize>();
        let arena = VulkanStableResourceArena::new(
            device,
            VulkanStableResourceArenaConfig::new(
                payload_bytes + groups.len() * alignment,
                alignment,
            )
            .unwrap(),
            &layouts,
        )
        .unwrap();
        let resource_slots = layouts
            .iter()
            .map(|layout| match layout {
                VulkanStableResourceGroupLayout::Explicit { resource_slots, .. } => {
                    resource_slots.clone()
                }
                VulkanStableResourceGroupLayout::Partitioned { .. } => unreachable!(),
            })
            .collect::<Vec<_>>();
        let resource_byte_counts = groups
            .iter()
            .map(|group| group.iter().map(|(bytes, _)| *bytes).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let requests = resource_slots
            .iter()
            .zip(&resource_byte_counts)
            .map(|(slots, byte_counts)| (slots.as_slice(), byte_counts.as_slice()))
            .collect::<Vec<_>>();
        let allocations = arena.allocate_groups(device, &requests, alignment).unwrap();
        let staging_byte_capacity = groups
            .iter()
            .flatten()
            .map(|(bytes, _)| *bytes)
            .max()
            .unwrap()
            .max(64 * 1024);
        let mut transfer = device
            .create_resident_transfer_stream(2, staging_byte_capacity)
            .unwrap();
        for (group, allocation_group) in groups.iter().zip(&allocations) {
            for ((byte_count, fill), allocation) in group.iter().zip(allocation_group) {
                let bytes = vec![*fill; *byte_count];
                let write = VulkanResidentBufferWriteRange::new(
                    allocation.buffer(),
                        allocation.buffer_byte_offset(),
                    &bytes,
                )
                .unwrap();
                let ticket = transfer.submit(&[write]).unwrap();
                transfer.wait(&ticket).unwrap();
            }
        }
        let mut table = VulkanStableResourceAddressTable::new(
            device,
            &mut transfer,
            resource_slots.iter().map(Vec::len).sum(),
        )
        .unwrap();
        for (slots, allocation_group) in resource_slots.iter().zip(&allocations) {
            table
                .publish_group(
                    &mut transfer,
                    &slots
                        .iter()
                        .copied()
                        .zip(allocation_group.iter().cloned())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
        }
        (arena, table)
    }

    fn read_binding<'a>(
        binding: u32,
        buffer: &'a VulkanResidentBuffer,
        byte_count: usize,
    ) -> VulkanResidentKernelBufferBinding<'a> {
        VulkanResidentKernelBufferBinding::new(binding, buffer, byte_count)
            .with_access(VulkanResidentKernelBufferAccess::Read)
    }

    fn write_binding<'a>(
        binding: u32,
        buffer: &'a VulkanResidentBuffer,
        byte_count: usize,
    ) -> VulkanResidentKernelBufferBinding<'a> {
        VulkanResidentKernelBufferBinding::new(binding, buffer, byte_count)
            .with_access(VulkanResidentKernelBufferAccess::Write)
    }

    fn f32_to_bf16_bits(value: f32) -> u16 {
        let bits = value.to_bits();
        let lsb = (bits >> 16) & 1;
        ((bits + 0x7fff + lsb) >> 16) as u16
    }

    fn u16_bytes(values: &[u16]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn u32_bytes(values: &[u32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }
}
