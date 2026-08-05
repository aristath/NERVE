#[cfg(test)]
mod radix_topk_tests {
    use super::*;

    const DISCRETE_AMD_DEVICE_UUID_ENV: &str = "NERVE_TEST_VULKAN_DEVICE_UUID";

    fn rendered_radix_topk(template: &str) -> String {
        std::fs::read_to_string(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("shaders")
                .join(template),
        )
        .unwrap()
        .replace("{{MAX_SCORES}}", "16")
        .replace("{{TOP_K}}", "8")
        .replace("{{SORT_CAPACITY}}", "8")
        .replace("{{COMPRESSION_RATIO}}", "1")
            .replace("{{INDEX_OFFSET}}", "128")
    }

    fn discrete_amd_test_device() -> VulkanComputeDevice {
        let encoded = std::env::var(DISCRETE_AMD_DEVICE_UUID_ENV).unwrap_or_else(|_| {
            panic!(
                "{DISCRETE_AMD_DEVICE_UUID_ENV} must select an explicitly approved discrete AMD GPU"
            )
        });
        let encoded = encoded
            .strip_prefix("vulkan-uuid:")
            .expect("test device must use an exact vulkan-uuid: reference");
        assert_eq!(
            encoded.len(),
            32,
            "test device UUID must contain exactly 32 hexadecimal digits"
        );
        let mut uuid = [0u8; 16];
        for (index, byte) in uuid.iter_mut().enumerate() {
            let offset = index * 2;
            *byte = u8::from_str_radix(&encoded[offset..offset + 2], 16)
                .expect("test device UUID must be hexadecimal");
        }
        let device = VulkanComputeDevice::new_for_device_uuid(uuid).unwrap();
        assert_eq!(
            device.physical_device_id(),
            format!("vulkan-uuid:{encoded}"),
            "Vulkan opened a device other than the explicitly approved discrete AMD GPU"
        );
        device
    }

    fn compile_rendered_shader(template: &str, label: &str) -> Vec<u32> {
        let source_path =
            std::env::temp_dir().join(format!("nerve-test-{label}-{}.comp", std::process::id()));
        std::fs::write(&source_path, rendered_radix_topk(template)).unwrap();
        let spirv_words = compile_shader_words_from_source_path(&source_path)
            .expect("radix top-k shader must compile");
        let _ = std::fs::remove_file(source_path);
        spirv_words
    }

    fn u32_bytes(values: &[u32]) -> Vec<u8> {
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
    fn radix_topk_is_score_ordered_and_deterministic_at_the_cutoff_tie() {
        let spirv_words = compile_rendered_shader("radix_topk_index.comp.template", "radix-topk");
        let device = discrete_amd_test_device();
        let scores = device.create_resident_buffer(16 * size_of::<f32>()).unwrap();
        scores
            .write_bytes(&f32_bytes(&[
                3.0, 5.0, 5.0, -1.0, 9.0, 0.0, 9.0, 4.0, 5.0, 2.0, 8.0, 8.0,
                7.0, 6.0, 1.0, 9.0,
            ]))
            .unwrap();
        let indices = device.create_resident_buffer(8 * size_of::<u32>()).unwrap();
        let stream_control = device.create_resident_buffer(64).unwrap();
        let mut control = vec![0u32; 16];
        control[1] = 15;
        stream_control.write_bytes(&u32_bytes(&control)).unwrap();
        let dispatch = device
            .create_resident_kernel_dispatch(
                &spirv_words,
                &[
                    VulkanResidentKernelBufferBinding::new(
                        0,
                        &scores,
                        16 * size_of::<f32>(),
                    )
                    .with_access(VulkanResidentKernelBufferAccess::Read),
                    VulkanResidentKernelBufferBinding::new(
                        1,
                        &indices,
                        8 * size_of::<u32>(),
                    )
                    .with_access(VulkanResidentKernelBufferAccess::Write),
                    VulkanResidentKernelBufferBinding::new(2, &stream_control, 64)
                        .with_access(VulkanResidentKernelBufferAccess::Read),
                ],
                1,
                1024,
                0,
            )
            .unwrap();
        let expected = [132, 134, 143, 138, 139, 140, 141, 129];
        for repetition in 0..4 {
            indices.write_bytes(&vec![0; 8 * size_of::<u32>()]).unwrap();
            device.run_resident_kernel_dispatch(&dispatch, &[]).unwrap();
            let actual = indices
                .read_bytes(8 * size_of::<u32>())
                .unwrap()
                .chunks_exact(size_of::<u32>())
                .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(
                actual, expected,
                "radix top-k changed score order or cutoff tie-breaking on repetition {repetition}"
            );
        }
    }

    #[test]
    fn temporal_radix_topk_orders_every_position_and_is_deterministic() {
        let spirv_words = compile_rendered_shader(
            "radix_topk_index_temporal.comp.template",
            "temporal-radix-topk",
        );
        let device = discrete_amd_test_device();
        let scores = device.create_resident_buffer(32 * size_of::<f32>()).unwrap();
        scores
            .write_bytes(&f32_bytes(&[
                3.0, 5.0, 5.0, -1.0, 9.0, 0.0, 9.0, 4.0, 5.0, 2.0, 8.0, 8.0,
                7.0, 6.0, 1.0, 9.0, 4.0, 6.0, 2.0, 7.0, 7.0, 3.0, 1.0, 8.0,
                8.0, 5.0, 0.0, 8.0, 9.0, -2.0, 4.0, 9.0,
            ]))
            .unwrap();
        let indices = device.create_resident_buffer(16 * size_of::<u32>()).unwrap();
        let dispatch = device
            .create_resident_kernel_dispatch(
                &spirv_words,
                &[
                    VulkanResidentKernelBufferBinding::new(
                        0,
                        &scores,
                        32 * size_of::<f32>(),
                    )
                    .with_access(VulkanResidentKernelBufferAccess::Read),
                    VulkanResidentKernelBufferBinding::new(
                        1,
                        &indices,
                        16 * size_of::<u32>(),
                    )
                    .with_access(VulkanResidentKernelBufferAccess::Write),
                ],
                1,
                1024,
                16,
            )
            .unwrap();
        let expected = [
            132, 134, 143, 138, 139, 140, 141, 129, 140, 143, 135, 136, 139, 131, 132, 129,
        ];
        let push_constants = [2u32, 15, 0, 128]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        for repetition in 0..4 {
            indices
                .write_bytes(&vec![0; 16 * size_of::<u32>()])
                .unwrap();
            device
                .run_resident_kernel_dispatch(&dispatch, &push_constants)
                .unwrap();
            let actual = indices
                .read_bytes(16 * size_of::<u32>())
                .unwrap()
                .chunks_exact(size_of::<u32>())
                .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(
                actual, expected,
                "temporal radix top-k changed score order or cutoff tie-breaking on repetition {repetition}"
            );
        }
    }
}
