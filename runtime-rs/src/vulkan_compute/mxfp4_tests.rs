#[cfg(test)]
mod mxfp4_tests {
    use super::*;

    const WIDTH: usize = 128;
    const WEIGHT_BYTES: usize = WIDTH * WIDTH / 2;
    const SCALE_BYTES: usize = WIDTH * WIDTH / 32;
    const FRAME_BYTES: usize = WIDTH * 2;

    #[test]
    fn native_mxfp4_expert_kernels_match_known_cpu_results() {
        let Some(raw_device_index) = std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX").ok()
        else {
            eprintln!(
                "skipping native MXFP4 conformance: explicit Vulkan device index unset"
            );
            return;
        };
        let device_index = raw_device_index
            .parse::<usize>()
            .expect("NERVE_TEST_VULKAN_DEVICE_INDEX must be an integer");
        let gate_shader = render_mxfp4_shader(
            "independent_sparse_moe_gate_up_mxfp4.comp.template",
            &[("{{TILE_ROWS}}", "32"), ("{{SWIGLU_LIMIT}}", "10.0")],
        );
        let down_shader = render_mxfp4_shader(
            "independent_sparse_moe_down_mxfp4.comp.template",
            &[("{{TILE_ROWS}}", "64")],
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
        let gate_addresses = address_table(
            &device,
            &[&gate_weight, &gate_scale, &up_weight, &up_scale],
        );
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
    }

    fn render_mxfp4_shader(
        template_name: &str,
        stage_replacements: &[(&str, &str)],
    ) -> Vec<u32> {
        let shader_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders");
        let mut source = std::fs::read_to_string(shader_dir.join(template_name)).unwrap();
        for (pattern, value) in [
            ("{{HIDDEN_SIZE}}", "128"),
            ("{{INTERMEDIATE_SIZE}}", "128"),
            ("{{NUM_EXPERTS}}", "1"),
            ("{{EXPERTS_PER_TOKEN}}", "1"),
        ]
        .into_iter()
        .chain(stage_replacements.iter().copied())
        {
            source = source.replace(pattern, value);
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
        resources: &[&VulkanResidentBuffer],
    ) -> VulkanResidentBuffer {
        let byte_count = resources.len() * 32;
        let table = device.create_resident_buffer(byte_count).unwrap();
        let mut words = vec![0u32; resources.len() * 8];
        for (slot, resource) in resources.iter().enumerate() {
            let address = resource.device_address().unwrap();
            words[slot * 8] = address as u32;
            words[slot * 8 + 1] = (address >> 32) as u32;
        }
        table.write_bytes(&u32_bytes(&words)).unwrap();
        table
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
        values.iter().flat_map(|value| value.to_le_bytes()).collect()
    }

    fn u32_bytes(values: &[u32]) -> Vec<u8> {
        values.iter().flat_map(|value| value.to_le_bytes()).collect()
    }
}
