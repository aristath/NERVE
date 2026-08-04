#[cfg(test)]
mod cooperative_fp8_crossover_tests {
    use super::*;
    use std::mem::size_of;
    use std::time::{Duration, Instant};

    const BATCH_WIDTH: usize = 6;
    const BLOCK_SIZE: usize = 128;

    #[test]
    fn compact_cooperative_fp8_crosses_over_on_large_verifier_projection() {
        let Some((vector_ns, cooperative_ns)) = run_crossover(1024, 32768) else {
            return;
        };
        assert!(cooperative_ns < vector_ns);
    }

    #[test]
    fn compact_cooperative_fp8_does_not_replace_small_vector_projection() {
        let Some((vector_ns, cooperative_ns)) = run_crossover(4096, 2048) else {
            return;
        };
        assert!(vector_ns < cooperative_ns);
    }

    fn run_crossover(input_size: usize, output_size: usize) -> Option<(u64, u64)> {
        let Some(raw_device_index) = std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX").ok() else {
            eprintln!("skipping cooperative FP8 crossover: explicit Vulkan device index unset");
            return None;
        };
        let started = Instant::now();
        let device_index = raw_device_index
            .parse::<usize>()
            .expect("NERVE_TEST_VULKAN_DEVICE_INDEX must be an integer");
        let device = VulkanComputeDevice::new_for_physical_device_index(device_index)
            .expect("explicit AMD Vulkan device must open");
        assert!(device.supports_cooperative_float8_e4m3_shape(16, 16, 16));

        let vector_shader = render_shader(
            "linear_prequant_batch_fp8_e4m3.comp.template",
            input_size,
            output_size,
            &[
                ("{{BATCH_TILE_WIDTH}}", "8"),
                ("{{OUTPUT_TILE_ROWS}}", "16"),
                ("{{AUXILIARY_BUFFER}}", ""),
                ("{{OUTPUT_BINDING}}", "2"),
                ("{{WEIGHT_BINDING}}", "3"),
                ("{{WEIGHT_SCALE_BINDING}}", "4"),
                ("{{WEIGHT_SCALE_READ}}", e8m0_scale_read()),
                (
                    "{{FINALIZE_OUTPUT_FUNCTION}}",
                    "float finalize_output(uint batch_index, uint row, float value) { return value; }",
                ),
            ],
        );
        let cooperative_shader = render_shader(
            "linear_prequant_batch_cooperative_fp8_e4m3.comp.template",
            input_size,
            output_size,
            &[
                ("{{MATRIX_M}}", "16"),
                ("{{MATRIX_N}}", "16"),
                ("{{MATRIX_K}}", "16"),
                ("{{BATCH_TILE_MULTIPLIER}}", "1"),
                ("{{RESIDUAL_BINDING}}", ""),
                ("{{OUTPUT_BINDING}}", "2"),
                ("{{WEIGHT_BINDING}}", "3"),
                ("{{WEIGHT_SCALE_BINDING}}", "4"),
                ("{{WEIGHT_SCALE_READ}}", e8m0_scale_read()),
                (
                    "{{FINALIZE_FUNCTION}}",
                    "float finalize_result(uint batch_index, uint output_index, float value) { return value; }",
                ),
            ],
        );

        let input_blocks = input_size / BLOCK_SIZE;
        let output_blocks = output_size / BLOCK_SIZE;
        let inputs = filled_buffer(&device, BATCH_WIDTH * input_size, 0x30);
        let input_scales = device
            .create_resident_buffer(BATCH_WIDTH * input_blocks * size_of::<f32>())
            .unwrap();
        input_scales
            .write_bytes(
                &(0..BATCH_WIDTH * input_blocks)
                    .flat_map(|_| 1.0_f32.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let weight = filled_buffer(&device, input_size * output_size, 0x30);
        let weight_scales = filled_buffer(&device, output_blocks * input_blocks, 127);
        let vector_output = filled_buffer(&device, BATCH_WIDTH * output_size * size_of::<u16>(), 0);
        let cooperative_output =
            filled_buffer(&device, BATCH_WIDTH * output_size * size_of::<u16>(), 0);
        let vector_dispatch = dispatch(
            &device,
            &vector_shader,
            &inputs,
            &input_scales,
            &vector_output,
            &weight,
            &weight_scales,
            (output_size / 16) as u32,
            1024,
        );
        let cooperative_dispatch = dispatch(
            &device,
            &cooperative_shader,
            &inputs,
            &input_scales,
            &cooperative_output,
            &weight,
            &weight_scales,
            (output_size / 64) as u32,
            256,
        );
        let batch_control = (BATCH_WIDTH as u32).to_le_bytes();
        device
            .run_resident_kernel_dispatch(&vector_dispatch, &batch_control)
            .unwrap();
        device
            .run_resident_kernel_dispatch(&cooperative_dispatch, &batch_control)
            .unwrap();
        let vector_bytes = vector_output
            .read_bytes(vector_output.byte_capacity())
            .unwrap();
        assert_eq!(
            cooperative_output
                .read_bytes(cooperative_output.byte_capacity())
                .unwrap(),
            vector_bytes,
            "compact cooperative FP8 must match the vector BF16 boundary",
        );
        assert_eq!(
            vector_bytes,
            u16_bytes(&vec![
                f32_to_bf16_bits(input_size as f32 * 0.25);
                BATCH_WIDTH * output_size
            ]),
        );

        let vector_sequence = timestamped(&device, &vector_dispatch, &batch_control);
        let cooperative_sequence = timestamped(&device, &cooperative_dispatch, &batch_control);
        let timeout = Duration::from_secs(10);
        let vector_ns = device
            .run_timestamped_recorded_resident_kernel_sequence_for(&vector_sequence, timeout)
            .unwrap();
        let cooperative_ns = device
            .run_timestamped_recorded_resident_kernel_sequence_for(&cooperative_sequence, timeout)
            .unwrap();
        eprintln!(
            "cooperative_fp8_crossover width={BATCH_WIDTH} input={input_size} output={output_size} vector_ms={:.6} cooperative_ms={:.6} ratio={:.6} elapsed_ms={:.3}",
            vector_ns as f64 / 1_000_000.0,
            cooperative_ns as f64 / 1_000_000.0,
            cooperative_ns as f64 / vector_ns as f64,
            started.elapsed().as_secs_f64() * 1_000.0,
        );
        assert!(started.elapsed() < Duration::from_secs(60));
        Some((vector_ns, cooperative_ns))
    }

    fn e8m0_scale_read() -> &'static str {
        "uint packed = weight_scale_inv.words[index >> 2u];\n    uint e8m0 = (packed >> ((index & 3u) * 8u)) & 0xffu;\n    return uintBitsToFloat(e8m0 << 23u);"
    }

    fn render_shader(
        template_name: &str,
        input_size: usize,
        output_size: usize,
        replacements: &[(&str, &str)],
    ) -> Vec<u32> {
        let shader_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders");
        let mut source = std::fs::read_to_string(shader_dir.join(template_name)).unwrap();
        for (pattern, value) in replacements {
            source = source.replace(pattern, value);
        }
        source = source
            .replace("{{BLOCK_ROWS}}", "128")
            .replace("{{BLOCK_COLUMNS}}", "128")
            .replace("{{INPUT_SIZE}}", &input_size.to_string())
            .replace("{{OUTPUT_SIZE}}", &output_size.to_string());
        assert!(!source.contains("{{"));
        let path = std::env::temp_dir().join(format!(
            "nerve-test-{}-{}.comp",
            template_name.replace(['/', '.'], "-"),
            std::process::id(),
        ));
        std::fs::write(&path, source).unwrap();
        let words = compile_shader_words_from_source_path(&path)
            .unwrap_or_else(|| panic!("{template_name} must compile"));
        let _ = std::fs::remove_file(path);
        words
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch(
        device: &VulkanComputeDevice,
        shader: &[u32],
        inputs: &VulkanResidentBuffer,
        input_scales: &VulkanResidentBuffer,
        output: &VulkanResidentBuffer,
        weight: &VulkanResidentBuffer,
        weight_scales: &VulkanResidentBuffer,
        workgroups: u32,
        local_size: u32,
    ) -> VulkanResidentKernelDispatch {
        device
            .create_resident_kernel_dispatch(
                shader,
                &[
                    binding(0, inputs, VulkanResidentKernelBufferAccess::Read),
                    binding(1, input_scales, VulkanResidentKernelBufferAccess::Read),
                    binding(2, output, VulkanResidentKernelBufferAccess::Write),
                    binding(3, weight, VulkanResidentKernelBufferAccess::Read),
                    binding(4, weight_scales, VulkanResidentKernelBufferAccess::Read),
                ],
                workgroups,
                local_size,
                size_of::<u32>() as u32,
            )
            .unwrap()
    }

    fn binding(
        binding: u32,
        buffer: &VulkanResidentBuffer,
        access: VulkanResidentKernelBufferAccess,
    ) -> VulkanResidentKernelBufferBinding<'_> {
        VulkanResidentKernelBufferBinding::new(binding, buffer, buffer.byte_capacity())
            .with_access(access)
    }

    fn timestamped(
        device: &VulkanComputeDevice,
        dispatch: &VulkanResidentKernelDispatch,
        push_constants: &[u8],
    ) -> VulkanResidentKernelSequence {
        let sequence = device
            .create_timestamped_resident_kernel_sequence()
            .unwrap();
        device
            .record_resident_kernel_sequence(
                &sequence,
                &[VulkanResidentKernelSequenceStep::new(
                    dispatch,
                    push_constants,
                )],
            )
            .unwrap();
        sequence
    }

    fn filled_buffer(
        device: &VulkanComputeDevice,
        byte_count: usize,
        value: u8,
    ) -> VulkanResidentBuffer {
        let buffer = device.create_resident_buffer(byte_count).unwrap();
        buffer.write_bytes(&vec![value; byte_count]).unwrap();
        buffer
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
}
