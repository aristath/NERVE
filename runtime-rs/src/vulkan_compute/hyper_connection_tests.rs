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

    #[test]
    fn hyper_connection_post_consumes_source_to_output_matrix() {
        let device_index = std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX")
            .expect("NERVE_TEST_VULKAN_DEVICE_INDEX must select an idle AMD GPU")
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
}
