#[cfg(test)]
mod hyper_connection_tests {
    use super::*;

    fn bf16_bits(value: f32) -> u16 {
        let bits = value.to_bits();
        let lsb = (bits >> 16) & 1;
        ((bits + 0x7fff + lsb) >> 16) as u16
    }

    fn bf16_value(bits: u16) -> f32 {
        f32::from_bits(u32::from(bits) << 16)
    }

    fn u16_bytes(values: &[u16]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn rendered_hyper_connection_pre(hidden_size: usize) -> String {
        std::fs::read_to_string(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("shaders")
                .join("hyper_connection_pre.comp.template"),
        )
        .unwrap()
        .replace(
            "{{SOURCE_BUFFERS}}",
            "layout(set = 0, binding = 0) readonly buffer InputStreams { uint words[]; } input_streams;",
        )
        .replace(
            "{{SOURCE_HYPER_WORD}}",
            "return input_streams.words[word_index];",
        )
        .replace("{{REDUCED_OUTPUT_BINDING}}", "1")
        .replace("{{POST_OUTPUT_BINDING}}", "2")
        .replace("{{COMBINATION_OUTPUT_BINDING}}", "3")
        .replace("{{FUNCTION_BINDING}}", "4")
        .replace("{{SCALE_BINDING}}", "5")
        .replace("{{BASE_BINDING}}", "6")
        .replace("{{MULTIPLICITY}}", "4")
        .replace("{{HIDDEN_SIZE}}", &hidden_size.to_string())
        .replace("{{SINKHORN_ITERATIONS}}", "20")
        .replace("{{NORMALIZATION_EPSILON}}", "1e-6")
        .replace("{{SINKHORN_EPSILON}}", "1e-6")
    }

    #[test]
    fn hyper_connection_post_consumes_source_to_output_matrix() {
        let device_index = std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX")
            .expect("NERVE_TEST_VULKAN_DEVICE_INDEX must select a compatible AMD GPU with sufficient safe remaining capacity")
            .parse::<usize>()
            .expect("NERVE_TEST_VULKAN_DEVICE_INDEX must be an integer");
        let template = std::fs::read_to_string(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("shaders")
                .join("hyper_connection_post.comp.template"),
        )
        .unwrap();
        let rendered = template
            .replace("{{MULTIPLICITY}}", "2")
            .replace("{{HIDDEN_SIZE}}", "4");
        let source_path = std::env::temp_dir().join(format!(
            "nerve-test-hyper-connection-post-{}.comp",
            std::process::id()
        ));
        std::fs::write(&source_path, rendered).unwrap();
        let spirv_words = compile_shader_words_from_source_path(&source_path)
            .expect("hyper-connection post shader must compile");
        let _ = std::fs::remove_file(source_path);

        let device = VulkanComputeDevice::new_for_physical_device_index(device_index).unwrap();
        let operator = device.create_resident_buffer(8).unwrap();
        operator
            .write_bytes(&u16_bytes(&[1.0, 2.0, 3.0, 4.0].map(bf16_bits)))
            .unwrap();
        let residual = device.create_resident_buffer(16).unwrap();
        residual
            .write_bytes(&u16_bytes(
                &[10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0]
                    .map(bf16_bits),
            ))
            .unwrap();
        let post = device.create_resident_buffer(8).unwrap();
        post.write_bytes(&f32_bytes(&[0.0, 0.0])).unwrap();
        let combination = device.create_resident_buffer(16).unwrap();
        combination
            .write_bytes(&f32_bytes(&[
                1.0, 2.0, // source 0 -> outputs 0, 1
                3.0, 4.0, // source 1 -> outputs 0, 1
            ]))
            .unwrap();
        let output = device.create_resident_buffer(16).unwrap();
        output.write_bytes(&[0; 16]).unwrap();

        let dispatch = device
            .create_resident_kernel_dispatch(
                &spirv_words,
                &[
                    VulkanResidentKernelBufferBinding::new(0, &operator, 8)
                        .with_access(VulkanResidentKernelBufferAccess::Read),
                    VulkanResidentKernelBufferBinding::new(1, &residual, 16)
                        .with_access(VulkanResidentKernelBufferAccess::Read),
                    VulkanResidentKernelBufferBinding::new(2, &post, 8)
                        .with_access(VulkanResidentKernelBufferAccess::Read),
                    VulkanResidentKernelBufferBinding::new(3, &combination, 16)
                        .with_access(VulkanResidentKernelBufferAccess::Read),
                    VulkanResidentKernelBufferBinding::new(4, &output, 16)
                        .with_access(VulkanResidentKernelBufferAccess::Write),
                ],
                1,
                64,
                0,
            )
            .unwrap();
        device.run_resident_kernel_dispatch(&dispatch, &[]).unwrap();

        let actual = output
            .read_bytes(16)
            .unwrap()
            .chunks_exact(2)
            .map(|bytes| bf16_value(u16::from_le_bytes([bytes[0], bytes[1]])))
            .collect::<Vec<_>>();
        let expected = [310.0, 620.0, 930.0, 1240.0, 420.0, 840.0, 1260.0, 1680.0];
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let expected = bf16_value(bf16_bits(expected));
            assert_eq!(
                *actual, expected,
                "hyper output element {index} used the wrong matrix orientation"
            );
        }
    }

    #[test]
    fn hyper_connection_pre_parallel_row_reduction_is_exact_and_fast_at_product_geometry() {
        let device_index = std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX")
            .expect("NERVE_TEST_VULKAN_DEVICE_INDEX must select a compatible AMD GPU with sufficient safe remaining capacity")
            .parse::<usize>()
            .expect("NERVE_TEST_VULKAN_DEVICE_INDEX must be an integer");
        const MULTIPLICITY: usize = 4;
        const HIDDEN_SIZE: usize = 4096;
        const MIX_COUNT: usize = (2 + MULTIPLICITY) * MULTIPLICITY;
        let source_path = std::env::temp_dir().join(format!(
            "nerve-test-hyper-connection-pre-{}.comp",
            std::process::id()
        ));
        std::fs::write(
            &source_path,
            rendered_hyper_connection_pre(HIDDEN_SIZE),
        )
        .unwrap();
        let spirv_words = compile_shader_words_from_source_path(&source_path)
            .expect("parallel hyper-connection pre shader must compile");
        let _ = std::fs::remove_file(source_path);

        let device = VulkanComputeDevice::new_for_physical_device_index(device_index).unwrap();
        let input_values = (0..MULTIPLICITY)
            .flat_map(|stream| {
                std::iter::repeat_n(bf16_bits((stream + 1) as f32), HIDDEN_SIZE)
            })
            .collect::<Vec<_>>();
        let input = device
            .create_resident_buffer(input_values.len() * size_of::<u16>())
            .unwrap();
        input.write_bytes(&u16_bytes(&input_values)).unwrap();
        let reduced = device
            .create_resident_buffer(HIDDEN_SIZE * size_of::<u16>())
            .unwrap();
        let post = device
            .create_resident_buffer(MULTIPLICITY * size_of::<f32>())
            .unwrap();
        let combination = device
            .create_resident_buffer(MULTIPLICITY * MULTIPLICITY * size_of::<f32>())
            .unwrap();
        let function = device
            .create_resident_buffer(
                MIX_COUNT * MULTIPLICITY * HIDDEN_SIZE * size_of::<f32>(),
            )
            .unwrap();
        function
            .write_bytes(&vec![
                0;
                MIX_COUNT * MULTIPLICITY * HIDDEN_SIZE * size_of::<f32>()
            ])
            .unwrap();
        let scale = device.create_resident_buffer(3 * size_of::<f32>()).unwrap();
        scale.write_bytes(&f32_bytes(&[1.0, 1.0, 1.0])).unwrap();
        let base = device
            .create_resident_buffer(MIX_COUNT * size_of::<f32>())
            .unwrap();
        base.write_bytes(&f32_bytes(&vec![0.0; MIX_COUNT])).unwrap();

        let dispatch = device
            .create_resident_kernel_dispatch(
                &spirv_words,
                &[
                    VulkanResidentKernelBufferBinding::new(
                        0,
                        &input,
                        input_values.len() * size_of::<u16>(),
                    )
                    .with_access(VulkanResidentKernelBufferAccess::Read),
                    VulkanResidentKernelBufferBinding::new(
                        1,
                        &reduced,
                        HIDDEN_SIZE * size_of::<u16>(),
                    )
                    .with_access(VulkanResidentKernelBufferAccess::Write),
                    VulkanResidentKernelBufferBinding::new(
                        2,
                        &post,
                        MULTIPLICITY * size_of::<f32>(),
                    )
                    .with_access(VulkanResidentKernelBufferAccess::Write),
                    VulkanResidentKernelBufferBinding::new(
                        3,
                        &combination,
                        MULTIPLICITY * MULTIPLICITY * size_of::<f32>(),
                    )
                    .with_access(VulkanResidentKernelBufferAccess::Write),
                    VulkanResidentKernelBufferBinding::new(
                        4,
                        &function,
                        MIX_COUNT * MULTIPLICITY * HIDDEN_SIZE * size_of::<f32>(),
                    )
                    .with_access(VulkanResidentKernelBufferAccess::Read),
                    VulkanResidentKernelBufferBinding::new(
                        5,
                        &scale,
                        3 * size_of::<f32>(),
                    )
                    .with_access(VulkanResidentKernelBufferAccess::Read),
                    VulkanResidentKernelBufferBinding::new(
                        6,
                        &base,
                        MIX_COUNT * size_of::<f32>(),
                    )
                    .with_access(VulkanResidentKernelBufferAccess::Read),
                ],
                1,
                1024,
                0,
            )
            .unwrap();
        let sequence = device
            .create_timestamped_resident_kernel_sequence()
            .unwrap();
        device
            .record_resident_kernel_sequence(
                &sequence,
                &[VulkanResidentKernelSequenceStep::new(&dispatch, &[])],
            )
            .unwrap();
        let timeout = std::time::Duration::from_secs(1);
        device
            .run_timestamped_recorded_resident_kernel_sequence_for(&sequence, timeout)
            .unwrap();
        let measured_ns = device
            .run_timestamped_recorded_resident_kernel_sequence_for(&sequence, timeout)
            .unwrap();
        eprintln!(
            "parallel hyper-connection pre product geometry: {:.3} ms",
            measured_ns as f64 / 1_000_000.0
        );

        let reduced_values = reduced
            .read_bytes(HIDDEN_SIZE * size_of::<u16>())
            .unwrap()
            .chunks_exact(2)
            .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
            .collect::<Vec<_>>();
        assert!(
            reduced_values.iter().all(|value| *value == bf16_bits(5.0)),
            "parallel row scheduling changed the BF16 hyper reduction"
        );
        let post_values = post
            .read_bytes(MULTIPLICITY * size_of::<f32>())
            .unwrap()
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(post_values, vec![1.0; MULTIPLICITY]);
        let combination_values = combination
            .read_bytes(MULTIPLICITY * MULTIPLICITY * size_of::<f32>())
            .unwrap()
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert!(combination_values.iter().all(|value| {
            value.is_finite() && (*value - 0.25).abs() < 2e-5
        }));
        assert!(
            measured_ns < 2_000_000,
            "parallel hyper-connection pre took {:.3} ms; the previous serial-row kernel was 7.176 ms",
            measured_ns as f64 / 1_000_000.0,
        );
    }
}
