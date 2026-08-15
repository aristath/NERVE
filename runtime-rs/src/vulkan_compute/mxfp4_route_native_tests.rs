#[cfg(test)]
mod mxfp4_route_native_tests {
    use super::*;

    const TEST_DEVICE_UUID_ENV: &str = "NERVE_TEST_VULKAN_DEVICE_UUID";

    #[test]
    fn route_native_mxfp4_experts_match_canonical_projection_bit_exactly() {
        let Some(device) = explicitly_selected_device() else {
            return;
        };
        let hidden_size = 128usize;
        let intermediate_size = 128usize;
        let num_experts = 4usize;
        let experts_per_token = 2usize;
        let intermediate_words = intermediate_size / 2;
        let expert_data_words = experts_per_token * intermediate_words;
        let expert_frame_words = expert_data_words + experts_per_token;

        let gate_shader = render_projection_shader(
            "independent_sparse_moe_gate_up_batch1_mxfp4.comp.template",
            hidden_size,
            intermediate_size,
            num_experts,
            experts_per_token,
            &["{{TILE_ROWS}}", "{{SWIGLU_LIMIT}}"],
            &["32", "10.0"],
            true,
        );
        let down_shader = render_projection_shader(
            "independent_sparse_moe_down_batch1_mxfp4.comp.template",
            hidden_size,
            intermediate_size,
            num_experts,
            experts_per_token,
            &["{{TILE_ROWS}}"],
            &["64"],
            false,
        );
        let route_compact_shader = render_route_shader(
            "moe_route_compact_batch1.comp.template",
            intermediate_size,
            experts_per_token,
            intermediate_size.div_ceil(32),
        );
        let route_count_shader = render_route_shader(
            "moe_route_count_batch1.comp.template",
            intermediate_size,
            experts_per_token,
            hidden_size.div_ceil(64),
        );

        let quantized_hidden = device.create_resident_buffer(hidden_size).unwrap();
        quantized_hidden.write_bytes(&vec![0x38; hidden_size]).unwrap();
        let hidden_scales = device.create_resident_buffer(4).unwrap();
        hidden_scales.write_bytes(&1.0f32.to_le_bytes()).unwrap();
        let routes = device
            .create_resident_buffer(experts_per_token * size_of::<u32>())
            .unwrap();
        routes
            .write_bytes(&u32_bytes(&[
                3 | (u32::from(f32_to_bf16_bits(0.75)) << 16),
                1 | (u32::from(f32_to_bf16_bits(0.25)) << 16),
            ]))
            .unwrap();
        let intermediates = device
            .create_resident_buffer(expert_frame_words * size_of::<u32>())
            .unwrap();
        let outputs = device
            .create_resident_buffer(experts_per_token * hidden_size * size_of::<u16>())
            .unwrap();

        let gate_weight_bytes = hidden_size * intermediate_size / 2;
        let gate_scale_bytes = hidden_size * intermediate_size / 32;
        let gate_resources = [
            filled_addressable_buffer(&device, gate_weight_bytes, 0x22),
            filled_addressable_buffer(&device, gate_scale_bytes, 120),
            filled_addressable_buffer(&device, gate_weight_bytes, 0x22),
            filled_addressable_buffer(&device, gate_scale_bytes, 120),
            filled_addressable_buffer(&device, gate_weight_bytes, 0x44),
            filled_addressable_buffer(&device, gate_scale_bytes, 120),
            filled_addressable_buffer(&device, gate_weight_bytes, 0x44),
            filled_addressable_buffer(&device, gate_scale_bytes, 120),
        ];
        let gate_addresses = address_table(&device, &gate_resources);
        let gate_slots = device
            .create_resident_buffer(num_experts * 4 * size_of::<u32>())
            .unwrap();
        let mut gate_slot_words = vec![u32::MAX; num_experts * 4];
        for (resource_base, expert) in [1usize, 3].into_iter().enumerate() {
            for parameter in 0..4 {
                gate_slot_words[expert * 4 + parameter] =
                    u32::try_from(resource_base * 4 + parameter).unwrap();
            }
        }
        gate_slots.write_bytes(&u32_bytes(&gate_slot_words)).unwrap();

        let down_weight_bytes = hidden_size * intermediate_size / 2;
        let down_scale_bytes = hidden_size * intermediate_size / 32;
        let down_resources = [
            filled_addressable_buffer(&device, down_weight_bytes, 0x22),
            filled_addressable_buffer(&device, down_scale_bytes, 120),
            filled_addressable_buffer(&device, down_weight_bytes, 0x44),
            filled_addressable_buffer(&device, down_scale_bytes, 120),
        ];
        let down_addresses = address_table(&device, &down_resources);
        let down_slots = device
            .create_resident_buffer(num_experts * 2 * size_of::<u32>())
            .unwrap();
        let mut down_slot_words = vec![u32::MAX; num_experts * 2];
        for (resource_base, expert) in [1usize, 3].into_iter().enumerate() {
            for parameter in 0..2 {
                down_slot_words[expert * 2 + parameter] =
                    u32::try_from(resource_base * 2 + parameter).unwrap();
            }
        }
        down_slots.write_bytes(&u32_bytes(&down_slot_words)).unwrap();

        let gate_control = device.create_resident_buffer(28).unwrap();
        let down_control = device.create_resident_buffer(28).unwrap();
        let gate_dispatch_x = experts_per_token * intermediate_size.div_ceil(32);
        let down_dispatch_x = experts_per_token * hidden_size.div_ceil(64);
        gate_control
            .write_bytes(&u32_bytes(&[1, 0, 0, 0, gate_dispatch_x as u32, 1, 1]))
            .unwrap();
        down_control
            .write_bytes(&u32_bytes(&[1, 0, 0, 0, down_dispatch_x as u32, 1, 1]))
            .unwrap();

        let gate_dispatch = device
            .create_resident_kernel_dispatch_2d(
                &gate_shader,
                &[
                    read_binding(0, &quantized_hidden),
                    read_binding(1, &hidden_scales),
                    read_binding(2, &routes),
                    write_binding(3, &intermediates),
                    read_binding(4, &gate_addresses),
                    read_binding(5, &gate_slots),
                    read_binding(31, &gate_control),
                ],
                gate_dispatch_x as u32,
                1,
                512,
                0,
            )
            .unwrap();
        let down_dispatch = device
            .create_resident_kernel_dispatch_2d(
                &down_shader,
                &[
                    read_binding(0, &intermediates),
                    read_binding(1, &routes),
                    write_binding(2, &outputs),
                    read_binding(3, &down_addresses),
                    read_binding(4, &down_slots),
                    read_binding(31, &down_control),
                ],
                down_dispatch_x as u32,
                1,
                512,
                0,
            )
            .unwrap();
        reset_intermediates(&intermediates, expert_frame_words, experts_per_token);
        outputs.write_bytes(&vec![0; outputs.byte_capacity()]).unwrap();
        device.run_resident_kernel_dispatch(&gate_dispatch, &[]).unwrap();
        device.run_resident_kernel_dispatch(&down_dispatch, &[]).unwrap();
        let canonical = outputs.read_bytes(outputs.byte_capacity()).unwrap();

        reset_intermediates(&intermediates, expert_frame_words, experts_per_token);
        outputs.write_bytes(&vec![0; outputs.byte_capacity()]).unwrap();
        gate_control
            .write_bytes(&u32_bytes(&[1, 0, num_experts as u32, 0, 0, 0, 0]))
            .unwrap();
        down_control
            .write_bytes(&u32_bytes(&[1, 0, num_experts as u32, 0, 0, 0, 0]))
            .unwrap();
        let compact_dispatch = device
            .create_resident_kernel_dispatch(
                &route_compact_shader,
                &[
                    read_binding(1, &routes),
                    write_binding(2, &intermediates),
                    read_write_binding(31, &gate_control),
                ],
                1,
                64,
                0,
            )
            .unwrap();
        let count_dispatch = device
            .create_resident_kernel_dispatch(
                &route_count_shader,
                &[
                    read_binding(1, &routes),
                    read_write_binding(31, &down_control),
                ],
                1,
                64,
                0,
            )
            .unwrap();
        let gate_sequence = device.create_resident_kernel_sequence().unwrap();
        device
            .run_resident_kernel_sequence(
                &gate_sequence,
                &[
                    VulkanResidentKernelSequenceStep::new(&compact_dispatch, &[]),
                    VulkanResidentKernelSequenceStep::new_indirect(
                        &gate_dispatch,
                        &[],
                        &gate_control,
                        16,
                    )
                    .unwrap(),
                ],
            )
            .unwrap();
        let down_sequence = device.create_resident_kernel_sequence().unwrap();
        device
            .run_resident_kernel_sequence(
                &down_sequence,
                &[
                    VulkanResidentKernelSequenceStep::new(&count_dispatch, &[]),
                    VulkanResidentKernelSequenceStep::new_indirect(
                        &down_dispatch,
                        &[],
                        &down_control,
                        16,
                    )
                    .unwrap(),
                ],
            )
            .unwrap();

        assert_eq!(
            u32_words(&gate_control),
            [1, 0, num_experts as u32, 2, gate_dispatch_x as u32, 1, 1],
        );
        assert_eq!(
            u32_words(&down_control),
            [1, 0, num_experts as u32, 2, down_dispatch_x as u32, 1, 1],
        );
        assert_eq!(
            outputs.read_bytes(outputs.byte_capacity()).unwrap(),
            canonical,
            "route-native helper plus indirect projection must preserve canonical expert output",
        );
    }

    #[test]
    fn one_participant_tensor_parallel_down_reduction_matches_canonical_bit_exactly() {
        let Some(device) = explicitly_selected_device() else {
            return;
        };
        let hidden_size = 128usize;
        let intermediate_size = 128usize;
        let num_experts = 4usize;
        let experts_per_token = 2usize;
        let intermediate_words = intermediate_size / 2;
        let expert_frame_words = experts_per_token * intermediate_words + experts_per_token;

        let canonical_shader = render_projection_shader(
            "independent_sparse_moe_down_batch1_mxfp4.comp.template",
            hidden_size,
            intermediate_size,
            num_experts,
            experts_per_token,
            &["{{TILE_ROWS}}"],
            &["64"],
            false,
        );
        let tensor_parallel_shader = render_projection_shader_with_tensor_parallel(
            "independent_sparse_moe_down_batch1_mxfp4.comp.template",
            hidden_size,
            intermediate_size,
            num_experts,
            experts_per_token,
            &["{{TILE_ROWS}}"],
            &["64"],
            false,
            true,
        );
        let reduction_shader = compile_shader(
            "distributed_sum_f32_scale_packed_bf16_to_bf16.comp",
            std::fs::read_to_string(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("shaders/distributed_sum_f32_scale_packed_bf16_to_bf16.comp"),
            )
            .unwrap(),
        );

        let intermediates = device
            .create_resident_buffer(expert_frame_words * size_of::<u32>())
            .unwrap();
        reset_intermediates(&intermediates, expert_frame_words, experts_per_token);
        for word in 0..experts_per_token * intermediate_words {
            let value = 0x3c38_4030u32.wrapping_add(word as u32);
            intermediates
                .write_bytes_at(word * size_of::<u32>(), &value.to_le_bytes())
                .unwrap();
        }
        let routes = device
            .create_resident_buffer(experts_per_token * size_of::<u32>())
            .unwrap();
        routes
            .write_bytes(&u32_bytes(&[
                3 | (u32::from(f32_to_bf16_bits(0.75)) << 16),
                1 | (u32::from(f32_to_bf16_bits(0.25)) << 16),
            ]))
            .unwrap();

        let down_weight_bytes = hidden_size * intermediate_size / 2;
        let down_scale_bytes = hidden_size * intermediate_size / 32;
        let resources = [
            filled_addressable_buffer(&device, down_weight_bytes, 0x22),
            filled_addressable_buffer(&device, down_scale_bytes, 120),
            filled_addressable_buffer(&device, down_weight_bytes, 0x44),
            filled_addressable_buffer(&device, down_scale_bytes, 120),
        ];
        let addresses = address_table(&device, &resources);
        let slots = device
            .create_resident_buffer(num_experts * 2 * size_of::<u32>())
            .unwrap();
        let mut slot_words = vec![u32::MAX; num_experts * 2];
        for (resource_base, expert) in [1usize, 3].into_iter().enumerate() {
            for parameter in 0..2 {
                slot_words[expert * 2 + parameter] =
                    u32::try_from(resource_base * 2 + parameter).unwrap();
            }
        }
        slots.write_bytes(&u32_bytes(&slot_words)).unwrap();

        let output_byte_count = experts_per_token * hidden_size * size_of::<u16>();
        let canonical_output = device.create_resident_buffer(output_byte_count).unwrap();
        let canonical_control = device.create_resident_buffer(28).unwrap();
        canonical_control
            .write_bytes(&u32_bytes(&[
                1,
                0,
                0,
                0,
                u32::try_from(experts_per_token * hidden_size.div_ceil(64)).unwrap(),
                1,
                1,
            ]))
            .unwrap();
        let canonical_dispatch = device
            .create_resident_kernel_dispatch_2d(
                &canonical_shader,
                &[
                    read_binding(0, &intermediates),
                    read_binding(1, &routes),
                    write_binding(2, &canonical_output),
                    read_binding(3, &addresses),
                    read_binding(4, &slots),
                    read_binding(31, &canonical_control),
                ],
                u32::try_from(experts_per_token * hidden_size.div_ceil(64)).unwrap(),
                1,
                512,
                0,
            )
            .unwrap();
        device
            .run_resident_kernel_dispatch(&canonical_dispatch, &[])
            .unwrap();
        let canonical = canonical_output
            .read_bytes(canonical_output.byte_capacity())
            .unwrap();

        let partials = device
            .create_resident_buffer(experts_per_token * hidden_size * size_of::<f32>())
            .unwrap();
        let tensor_parallel_dispatch = device
            .create_resident_kernel_dispatch_2d(
                &tensor_parallel_shader,
                &[
                    read_binding(0, &intermediates),
                    read_binding(1, &routes),
                    write_binding(2, &partials),
                    read_binding(3, &addresses),
                    read_binding(4, &slots),
                ],
                u32::try_from(hidden_size.div_ceil(64)).unwrap(),
                1,
                512,
                8,
            )
            .unwrap();
        device
            .run_resident_kernel_dispatch(
                &tensor_parallel_dispatch,
                &u32_bytes(&[0, u32::try_from(intermediate_size).unwrap()]),
            )
            .unwrap();

        let reduced_output = device.create_resident_buffer(output_byte_count).unwrap();
        let reduction_dispatch = device
            .create_resident_kernel_dispatch_2d(
                &reduction_shader,
                &[
                    read_binding(0, &partials),
                    read_binding(1, &routes),
                    write_binding(2, &reduced_output),
                ],
                u32::try_from((experts_per_token * hidden_size / 2).div_ceil(64)).unwrap(),
                1,
                64,
                20,
            )
            .unwrap();
        device
            .run_resident_kernel_dispatch(
                &reduction_dispatch,
                &u32_bytes(&[
                    u32::try_from(experts_per_token * hidden_size).unwrap(),
                    1,
                    1,
                    u32::try_from(hidden_size).unwrap(),
                    16,
                ]),
            )
            .unwrap();

        assert_eq!(
            reduced_output
                .read_bytes(reduced_output.byte_capacity())
                .unwrap(),
            canonical,
            "one-participant intra-expert TP must preserve canonical route scaling exactly",
        );
    }

    fn explicitly_selected_device() -> Option<VulkanComputeDevice> {
        let Ok(physical_device_id) = std::env::var(TEST_DEVICE_UUID_ENV) else {
            eprintln!("skipping route-native MXFP4 conformance: explicit Vulkan device UUID unset");
            return None;
        };
        let encoded = physical_device_id
            .strip_prefix("vulkan-uuid:")
            .expect("test device must use an exact vulkan-uuid reference");
        assert_eq!(encoded.len(), 32);
        let mut uuid = [0u8; 16];
        for (index, byte) in uuid.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&encoded[index * 2..index * 2 + 2], 16).unwrap();
        }
        Some(VulkanComputeDevice::new_for_device_uuid(uuid).unwrap())
    }

    #[allow(clippy::too_many_arguments)]
    fn render_projection_shader(
        template_name: &str,
        hidden_size: usize,
        intermediate_size: usize,
        num_experts: usize,
        experts_per_token: usize,
        patterns: &[&str],
        values: &[&str],
        prequantized: bool,
    ) -> Vec<u32> {
        render_projection_shader_with_tensor_parallel(
            template_name,
            hidden_size,
            intermediate_size,
            num_experts,
            experts_per_token,
            patterns,
            values,
            prequantized,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn render_projection_shader_with_tensor_parallel(
        template_name: &str,
        hidden_size: usize,
        intermediate_size: usize,
        num_experts: usize,
        experts_per_token: usize,
        patterns: &[&str],
        values: &[&str],
        prequantized: bool,
        tensor_parallel: bool,
    ) -> Vec<u32> {
        let mut source = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("shaders")
                .join(template_name),
        )
        .unwrap();
        for (pattern, value) in [
            ("{{HIDDEN_SIZE}}", hidden_size.to_string()),
            ("{{INTERMEDIATE_SIZE}}", intermediate_size.to_string()),
            ("{{NUM_EXPERTS}}", num_experts.to_string()),
            ("{{NATIVE_FP8_RESOURCE_START}}", num_experts.to_string()),
            ("{{EXPERTS_PER_TOKEN}}", experts_per_token.to_string()),
            ("{{PREQUANTIZED_INPUT}}", usize::from(prequantized).to_string()),
            ("{{PREEXPANDED_FP8}}", "0".to_string()),
            ("{{DYNAMIC_WEIGHT_REPRESENTATION}}", "0".to_string()),
            ("{{INPUT_BLOCK_MAJOR}}", "0".to_string()),
            (
                "{{TENSOR_PARALLEL}}",
                usize::from(tensor_parallel).to_string(),
            ),
        ] {
            source = source.replace(pattern, &value);
        }
        for (pattern, value) in patterns.iter().zip(values) {
            source = source.replace(pattern, value);
        }
        source = source.replace(
            "layout(push_constant) uniform BatchControl",
            "layout(set = 0, binding = 31) readonly buffer BatchControl",
        );
        compile_shader(template_name, source)
    }

    fn render_route_shader(
        template_name: &str,
        intermediate_size: usize,
        experts_per_token: usize,
        tiles_per_route: usize,
    ) -> Vec<u32> {
        let mut source = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("shaders")
                .join(template_name),
        )
        .unwrap();
        for (pattern, value) in [
            ("{{INTERMEDIATE_SIZE}}", intermediate_size),
            ("{{EXPERTS_PER_TOKEN}}", experts_per_token),
            ("{{TILES_PER_ROUTE}}", tiles_per_route),
        ] {
            source = source.replace(pattern, &value.to_string());
        }
        for pattern in [
            "{{SELECTED_RESOURCE_BINDINGS}}",
            "{{SELECTED_RESOURCE_HELPERS}}",
            "{{SELECTED_RESOURCE_REJECTION}}",
            "{{SELECTED_RESOURCE_MATCH}}",
        ] {
            source = source.replace(pattern, "");
        }
        compile_shader(template_name, source)
    }

    fn compile_shader(name: &str, source: String) -> Vec<u32> {
        let source_path = std::env::temp_dir().join(format!(
            "nerve-test-route-native-{}-{}.comp",
            name.replace(['/', '.'], "-"),
            std::process::id(),
        ));
        std::fs::write(&source_path, source).unwrap();
        let words = compile_shader_words_from_source_path(&source_path).unwrap();
        let _ = std::fs::remove_file(source_path);
        words
    }

    fn filled_addressable_buffer(
        device: &VulkanComputeDevice,
        byte_count: usize,
        value: u8,
    ) -> VulkanResidentBuffer {
        let buffer = device.create_addressable_resident_buffer(byte_count).unwrap();
        buffer.write_bytes(&vec![value; byte_count]).unwrap();
        buffer
    }

    fn address_table(
        device: &VulkanComputeDevice,
        resources: &[VulkanResidentBuffer],
    ) -> VulkanResidentBuffer {
        let table = device.create_resident_buffer(resources.len() * 32).unwrap();
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

    fn read_binding<'a>(
        binding: u32,
        buffer: &'a VulkanResidentBuffer,
    ) -> VulkanResidentKernelBufferBinding<'a> {
        VulkanResidentKernelBufferBinding::new(binding, buffer, buffer.byte_capacity())
            .with_access(VulkanResidentKernelBufferAccess::Read)
    }

    fn write_binding<'a>(
        binding: u32,
        buffer: &'a VulkanResidentBuffer,
    ) -> VulkanResidentKernelBufferBinding<'a> {
        VulkanResidentKernelBufferBinding::new(binding, buffer, buffer.byte_capacity())
            .with_access(VulkanResidentKernelBufferAccess::Write)
    }

    fn read_write_binding<'a>(
        binding: u32,
        buffer: &'a VulkanResidentBuffer,
    ) -> VulkanResidentKernelBufferBinding<'a> {
        VulkanResidentKernelBufferBinding::new(binding, buffer, buffer.byte_capacity())
            .with_access(VulkanResidentKernelBufferAccess::ReadWrite)
    }

    fn reset_intermediates(
        intermediates: &VulkanResidentBuffer,
        expert_frame_words: usize,
        experts_per_token: usize,
    ) {
        let mut words = vec![0u32; expert_frame_words];
        let expert_data_words = expert_frame_words - experts_per_token;
        for route in 0..experts_per_token {
            words[expert_data_words + route] = route as u32;
        }
        intermediates.write_bytes(&u32_bytes(&words)).unwrap();
    }

    fn u32_words(buffer: &VulkanResidentBuffer) -> Vec<u32> {
        buffer
            .read_bytes(buffer.byte_capacity())
            .unwrap()
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
            .collect()
    }

    fn f32_to_bf16_bits(value: f32) -> u16 {
        let bits = value.to_bits();
        let lsb = (bits >> 16) & 1;
        ((bits + 0x7fff + lsb) >> 16) as u16
    }

    fn u32_bytes(values: &[u32]) -> Vec<u8> {
        values.iter().flat_map(|value| value.to_le_bytes()).collect()
    }
}
