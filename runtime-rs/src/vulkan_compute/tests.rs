#[cfg(test)]
pub(crate) fn compile_test_shader_words() -> Option<Vec<u32>> {
    use std::sync::atomic::{AtomicU64, Ordering};

    const SOURCE: &str = r#"#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Data {
    uint values[];
} data;

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index < data.values.length()) {
        data.values[index] = data.values[index] + 1;
    }
}
"#;

    static SOURCE_COUNTER: AtomicU64 = AtomicU64::new(0);
    let source_id = SOURCE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let source_path = std::env::temp_dir().join(format!(
        "nerve-test-increment-{}-{source_id}.comp",
        std::process::id()
    ));
    std::fs::write(&source_path, SOURCE).ok()?;
    let words = compile_shader_words_from_source_path(&source_path);
    let _ = std::fs::remove_file(source_path);
    words
}

#[cfg(test)]
pub(crate) fn compile_shader_words_from_source_path(shader: &Path) -> Option<Vec<u32>> {
    use std::process::Command;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COMPILE_COUNTER: AtomicU64 = AtomicU64::new(0);

    let compile_id = COMPILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let shader_file = shader
        .file_name()
        .and_then(|file_name| file_name.to_str())
        .unwrap_or("shader");
    let output = std::env::temp_dir().join(format!(
        "nerve-{}-{}-{}.spv",
        shader_file.replace(['/', '.'], "-"),
        std::process::id(),
        compile_id
    ));
    let compiled = if test_command_exists("glslangValidator") {
        Command::new("glslangValidator")
            .arg("-V")
            .arg("--target-env")
            .arg("vulkan1.4")
            .arg(shader)
            .arg("-o")
            .arg(&output)
            .status()
            .ok()?
            .success()
    } else if test_command_exists("glslc") {
        Command::new("glslc")
            .arg("--target-env=vulkan1.4")
            .arg(shader)
            .arg("-o")
            .arg(&output)
            .status()
            .ok()?
            .success()
    } else {
        return None;
    };
    if !compiled {
        return None;
    }
    let bytes = std::fs::read(&output).ok()?;
    let _ = std::fs::remove_file(&output);
    if bytes.len() % 4 != 0 {
        return None;
    }
    let words = bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    Some(words)
}

#[cfg(test)]
fn test_command_exists(command: &str) -> bool {
    use std::process::{Command, Stdio};

    Command::new(command)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

#[cfg(test)]
fn selected_test_vulkan_device() -> Result<VulkanComputeDevice, VulkanError> {
    let physical_device_id = std::env::var("NERVE_TEST_VULKAN_DEVICE_UUID").map_err(|error| {
        VulkanError(format!(
            "NERVE_TEST_VULKAN_DEVICE_UUID must select an approved discrete AMD UUID: {error}"
        ))
    })?;
    let encoded = physical_device_id.strip_prefix("vulkan-uuid:").ok_or_else(|| {
        VulkanError("Vulkan test UUID must use a vulkan-uuid: prefix".to_string())
    })?;
    if encoded.len() != vk::UUID_SIZE * 2 {
        return Err(VulkanError(format!(
            "Vulkan test UUID must contain {} hexadecimal digits",
            vk::UUID_SIZE * 2
        )));
    }
    let mut uuid = [0u8; vk::UUID_SIZE];
    for (index, byte) in uuid.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&encoded[index * 2..index * 2 + 2], 16).map_err(|error| {
            VulkanError(format!("Vulkan test UUID must be hexadecimal: {error}"))
        })?;
    }
    VulkanComputeDevice::new_for_device_uuid(uuid)
}

#[cfg(test)]
mod tests {
    use ash::vk::Handle as _;

    use super::*;

    #[test]
    fn resident_buffer_copy_visibility_covers_every_supported_producer_and_consumer() {
        let visibility = resident_buffer_copy_visibility();

        assert!(visibility.producer_stages.contains(vk::PipelineStageFlags::HOST));
        assert!(
            visibility
                .producer_stages
                .contains(vk::PipelineStageFlags::COMPUTE_SHADER)
        );
        assert!(
            visibility
                .producer_stages
                .contains(vk::PipelineStageFlags::TRANSFER)
        );
        assert!(visibility.producer_access.contains(vk::AccessFlags::HOST_WRITE));
        assert!(
            visibility
                .producer_access
                .contains(vk::AccessFlags::SHADER_WRITE)
        );
        assert!(
            visibility
                .producer_access
                .contains(vk::AccessFlags::TRANSFER_WRITE)
        );
        assert_eq!(visibility.copy_read_stage, vk::PipelineStageFlags::TRANSFER);
        assert_eq!(visibility.copy_read_access, vk::AccessFlags::TRANSFER_READ);
        assert_eq!(visibility.copy_write_stage, vk::PipelineStageFlags::TRANSFER);
        assert_eq!(visibility.copy_write_access, vk::AccessFlags::TRANSFER_WRITE);
        assert!(visibility.consumer_stages.contains(vk::PipelineStageFlags::HOST));
        assert!(
            visibility
                .consumer_stages
                .contains(vk::PipelineStageFlags::COMPUTE_SHADER)
        );
        assert!(
            visibility
                .consumer_stages
                .contains(vk::PipelineStageFlags::TRANSFER)
        );
        assert!(
            visibility
                .consumer_stages
                .contains(vk::PipelineStageFlags::DRAW_INDIRECT)
        );
        assert!(
            visibility
                .consumer_stages
                .contains(vk::PipelineStageFlags::CONDITIONAL_RENDERING_EXT)
        );
        assert!(visibility.consumer_access.contains(vk::AccessFlags::HOST_READ));
        assert!(
            visibility
                .consumer_access
                .contains(vk::AccessFlags::SHADER_READ)
        );
        assert!(
            visibility
                .consumer_access
                .contains(vk::AccessFlags::INDIRECT_COMMAND_READ)
        );
        assert!(
            visibility
                .consumer_access
                .contains(vk::AccessFlags::CONDITIONAL_RENDERING_READ_EXT)
        );
        assert!(
            visibility
                .consumer_access
                .contains(vk::AccessFlags::TRANSFER_READ)
        );
    }

    #[test]
    fn timeline_replay_rebases_each_logical_device_semaphore_independently() {
        let first = VulkanTimelineSemaphoreReplayIdentity {
            device_handle: 11,
            semaphore_handle: 101,
        };
        let second = VulkanTimelineSemaphoreReplayIdentity {
            device_handle: 22,
            semaphore_handle: 202,
        };
        let recorded = VulkanTimelineSemaphoreReplayState {
            next_values: BTreeMap::from([(first, 3), (second, 17)]),
        };
        let current = VulkanTimelineSemaphoreReplayState {
            next_values: BTreeMap::from([(first, 8), (second, 29)]),
        };
        let rebase = recorded.rebase_to(&current).unwrap();

        assert_eq!(
            rebase
                .value(vk::Device::from_raw(11), vk::Semaphore::from_raw(101), 4)
                .unwrap(),
            9
        );
        assert_eq!(
            rebase
                .value(vk::Device::from_raw(22), vk::Semaphore::from_raw(202), 19)
                .unwrap(),
            31
        );
    }

    #[test]
    fn timeline_replay_rejects_topology_changes_and_value_regression() {
        let identity = VulkanTimelineSemaphoreReplayIdentity {
            device_handle: 11,
            semaphore_handle: 101,
        };
        let recorded = VulkanTimelineSemaphoreReplayState {
            next_values: BTreeMap::from([(identity, 8)]),
        };
        assert!(
            recorded
                .rebase_to(&VulkanTimelineSemaphoreReplayState::default())
                .unwrap_err()
                .to_string()
                .contains("topology changed")
        );
        assert!(
            recorded
                .rebase_to(&VulkanTimelineSemaphoreReplayState {
                    next_values: BTreeMap::from([(identity, 7)]),
                })
                .unwrap_err()
                .to_string()
                .contains("regressed")
        );
    }

    fn queue_family(
        queue_flags: vk::QueueFlags,
        queue_count: u32,
    ) -> vk::QueueFamilyProperties {
        vk::QueueFamilyProperties {
            queue_flags,
            queue_count,
            ..Default::default()
        }
    }

    #[test]
    fn compute_queue_selection_prefers_a_non_graphics_family() {
        let queue_families = vec![
            queue_family(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE, 1),
            queue_family(vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER, 1),
        ];

        assert_eq!(
            preferred_compute_queue_family_indices(&queue_families),
            vec![1, 0]
        );
    }

    #[test]
    fn compute_queue_selection_falls_back_to_a_universal_family() {
        let queue_families = vec![
            queue_family(vk::QueueFlags::TRANSFER, 1),
            queue_family(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE, 1),
        ];

        assert_eq!(
            preferred_compute_queue_family_indices(&queue_families),
            vec![1]
        );
    }

    #[test]
    fn compute_queue_selection_ignores_families_without_queues() {
        let queue_families = vec![
            queue_family(vk::QueueFlags::COMPUTE, 0),
            queue_family(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE, 1),
        ];

        assert_eq!(
            preferred_compute_queue_family_indices(&queue_families),
            vec![1]
        );
    }

    #[test]
    fn indirect_dispatch_ranges_require_complete_aligned_commands() {
        assert!(validate_resident_indirect_dispatch_range(12, 0).is_ok());
        assert!(validate_resident_indirect_dispatch_range(28, 16).is_ok());

        assert_eq!(
            validate_resident_indirect_dispatch_range(16, 2).unwrap_err(),
            VulkanError("resident indirect dispatch offset 2 is not 4-byte aligned".to_string())
        );
        assert_eq!(
            validate_resident_indirect_dispatch_range(16, 8).unwrap_err(),
            VulkanError(
                "resident indirect dispatch range 8..20 exceeds buffer capacity 16".to_string()
            )
        );
        assert_eq!(
            validate_resident_indirect_dispatch_range(usize::MAX, usize::MAX - 3).unwrap_err(),
            VulkanError("resident indirect dispatch range overflowed".to_string())
        );
    }

    fn buffer_access(
        buffer: u64,
        access: VulkanResidentKernelBufferAccess,
    ) -> VulkanResidentKernelBufferAccessRecord {
        VulkanResidentKernelBufferAccessRecord {
            buffer: vk::Buffer::from_raw(buffer),
            access,
        }
    }

    #[test]
    fn semantic_timestamp_labels_expose_component_and_op_fields() {
        let label = "kernel=linear_00 component=block_00 node=attn_qkv op=parallel_linear_2way lane=3";

        assert_eq!(semantic_label_field(label, "component"), Some("block_00"));
        assert_eq!(
            semantic_label_field(label, "op"),
            Some("parallel_linear_2way")
        );
        assert_eq!(semantic_label_field(label, "node"), Some("attn_qkv"));
        assert_eq!(semantic_label_field(label, "missing"), None);
    }

    #[test]
    fn resident_kernel_dependencies_synchronize_only_conflicting_buffers() {
        let mut pending = vec![
            buffer_access(1, VulkanResidentKernelBufferAccess::Write),
            buffer_access(2, VulkanResidentKernelBufferAccess::Read),
        ];
        let current = [
            buffer_access(1, VulkanResidentKernelBufferAccess::Read),
            buffer_access(2, VulkanResidentKernelBufferAccess::Read),
        ];

        let dependencies = take_resident_kernel_buffer_dependencies(&mut pending, &current);

        assert_eq!(
            dependencies,
            vec![VulkanResidentKernelBufferDependency {
                buffer: vk::Buffer::from_raw(1),
            }]
        );
        assert_eq!(
            pending,
            vec![buffer_access(2, VulkanResidentKernelBufferAccess::Read)]
        );
    }

    #[test]
    fn resident_kernel_dependencies_preserve_read_after_read_without_a_barrier() {
        let access = buffer_access(1, VulkanResidentKernelBufferAccess::Read);
        let mut pending = vec![access];

        let dependencies = take_resident_kernel_buffer_dependencies(&mut pending, &[access]);

        assert!(dependencies.is_empty());
        assert_eq!(pending, vec![access]);
    }

    #[test]
    fn resident_kernel_access_merge_coalesces_each_buffer() {
        let mut pending = vec![buffer_access(1, VulkanResidentKernelBufferAccess::Read)];
        merge_resident_kernel_buffer_accesses(
            &mut pending,
            &[
                buffer_access(1, VulkanResidentKernelBufferAccess::Write),
                buffer_access(2, VulkanResidentKernelBufferAccess::Write),
            ],
        );

        assert_eq!(pending.len(), 2);
        assert_eq!(
            pending[0],
            buffer_access(1, VulkanResidentKernelBufferAccess::ReadWrite)
        );
        assert_eq!(
            pending[1],
            buffer_access(2, VulkanResidentKernelBufferAccess::Write)
        );
    }

    fn spirv_test_module(capabilities: &[u32], memory_model: u32) -> Vec<u32> {
        let mut words = vec![SPIRV_MAGIC, 0x0001_0600, 0, 1, 0];
        for capability in capabilities {
            words.extend([(2u32 << 16) | u32::from(SPIRV_OP_CAPABILITY), *capability]);
        }
        words.extend([
            (3u32 << 16) | u32::from(SPIRV_OP_MEMORY_MODEL),
            0,
            memory_model,
        ]);
        words
    }

    #[test]
    fn spirv_contract_extracts_every_feature_used_by_cooperative_bfloat16() {
        let words = spirv_test_module(&[1, 9, 22, 4433, 5116, 5118, 5345, 6022], 3);

        let requirements = vulkan_spirv_requirements(&words).unwrap();

        assert_eq!(
            requirements.shader_features,
            BTreeSet::from([
                VulkanShaderFeature::ShaderFloat16,
                VulkanShaderFeature::ShaderInt16,
                VulkanShaderFeature::StorageBuffer16BitAccess,
                VulkanShaderFeature::ShaderBfloat16Type,
                VulkanShaderFeature::ShaderBfloat16CooperativeMatrix,
                VulkanShaderFeature::VulkanMemoryModel,
                VulkanShaderFeature::CooperativeMatrix,
            ])
        );
    }

    #[test]
    fn spirv_contract_extracts_native_fp8_dot_product_feature() {
        let words = spirv_test_module(&[1, 4212, 6915], 1);

        let requirements = vulkan_spirv_requirements(&words).unwrap();

        assert_eq!(
            requirements.shader_features,
            BTreeSet::from([
                VulkanShaderFeature::ShaderFloat8,
                VulkanShaderFeature::ShaderMixedFloatDotProductFloat8AccFloat32,
            ])
        );
    }

    #[test]
    fn spirv_contract_extracts_every_feature_used_by_cooperative_float8() {
        let words = spirv_test_module(&[1, 39, 4212, 4213, 4448, 5345, 6022], 3);

        let requirements = vulkan_spirv_requirements(&words).unwrap();

        assert_eq!(
            requirements.shader_features,
            BTreeSet::from([
                VulkanShaderFeature::ShaderInt8,
                VulkanShaderFeature::StorageBuffer8BitAccess,
                VulkanShaderFeature::ShaderFloat8,
                VulkanShaderFeature::ShaderFloat8CooperativeMatrix,
                VulkanShaderFeature::VulkanMemoryModel,
                VulkanShaderFeature::CooperativeMatrix,
            ])
        );
    }

    #[test]
    fn spirv_contract_extracts_native_bfloat16_mixed_dot_product_feature() {
        let words = spirv_test_module(&[1, 5116, 6914], 1);

        let requirements = vulkan_spirv_requirements(&words).unwrap();

        assert_eq!(
            requirements.shader_features,
            BTreeSet::from([
                VulkanShaderFeature::ShaderBfloat16Type,
                VulkanShaderFeature::ShaderMixedFloatDotProductBfloat16Acc,
            ])
        );
    }

    #[test]
    fn spirv_contract_extracts_native_integer_dot_product_feature() {
        let words = spirv_test_module(&[1, 39, 6018, 6019], 1);

        let requirements = vulkan_spirv_requirements(&words).unwrap();

        assert_eq!(
            requirements.shader_features,
            BTreeSet::from([
                VulkanShaderFeature::ShaderInt8,
                VulkanShaderFeature::ShaderIntegerDotProduct,
            ])
        );
    }

    #[test]
    fn spirv_contract_extracts_buffer_device_address_feature() {
        let words = spirv_test_module(&[1, 11, 5347], 1);

        let requirements = vulkan_spirv_requirements(&words).unwrap();

        assert_eq!(
            requirements.shader_features,
            BTreeSet::from([
                VulkanShaderFeature::ShaderInt64,
                VulkanShaderFeature::BufferDeviceAddress,
            ])
        );
    }

    #[test]
    fn core_vulkan_versions_satisfy_promoted_device_extension_contracts() {
        assert_eq!(
            vulkan_core_device_extension_version(
                "VK_KHR_shader_integer_dot_product"
            ),
            Some(vk::API_VERSION_1_3)
        );
        assert!(
            vk::make_api_version(0, 1, 4, 0)
                >= vulkan_core_device_extension_version(
                    "VK_KHR_shader_integer_dot_product"
                )
                .unwrap()
        );
        assert_eq!(
            vulkan_core_device_extension_version("VK_EXT_shader_float8"),
            None
        );
    }

    #[test]
    fn spirv_contract_rejects_missing_device_features_before_gpu_submission() {
        let words = spirv_test_module(&[1, 5345], 3);

        let error = validate_spirv_device_contract(
            &words,
            &BTreeSet::new(),
            vk::ShaderStageFlags::COMPUTE,
            vk::SubgroupFeatureFlags::empty(),
        )
        .unwrap_err();

        assert_eq!(
            error,
            VulkanError(
                "shader artifact requires Vulkan features that were not enabled on the logical device: vulkan_memory_model"
                    .to_string()
            )
        );
    }

    #[test]
    fn spirv_contract_accepts_a_fully_provisioned_device_contract() {
        let words = spirv_test_module(&[1, 61, 63, 5345], 3);

        validate_spirv_device_contract(
            &words,
            &BTreeSet::from([VulkanShaderFeature::VulkanMemoryModel]),
            vk::ShaderStageFlags::COMPUTE,
            vk::SubgroupFeatureFlags::BASIC | vk::SubgroupFeatureFlags::ARITHMETIC,
        )
        .unwrap();
    }

    #[test]
    fn spirv_contract_rejects_unsupported_subgroup_operations() {
        let words = spirv_test_module(&[1, 61, 63], 1);

        let error = validate_spirv_device_contract(
            &words,
            &BTreeSet::new(),
            vk::ShaderStageFlags::COMPUTE,
            vk::SubgroupFeatureFlags::BASIC,
        )
        .unwrap_err();

        assert!(error.0.contains("arithmetic"));
    }

    #[test]
    fn package_capability_names_match_the_compiler_contract() {
        assert_eq!(
            serde_json::to_string(&VulkanShaderFeature::VulkanMemoryModel).unwrap(),
            "\"vulkan_memory_model\""
        );
        assert_eq!(
            serde_json::to_string(&VulkanShaderFeature::StorageBuffer16BitAccess).unwrap(),
            "\"storage_buffer16_bit_access\""
        );
        assert_eq!(
            serde_json::to_string(&VulkanSubgroupOperation::ShuffleRelative).unwrap(),
            "\"shuffle_relative\""
        );
    }

    #[test]
    fn spirv_contract_rejects_inconsistent_memory_model_declarations() {
        let vulkan_without_capability = spirv_test_module(&[1], 3);
        let capability_without_vulkan = spirv_test_module(&[1, 5345], 1);

        assert!(vulkan_spirv_requirements(&vulkan_without_capability).is_err());
        assert!(vulkan_spirv_requirements(&capability_without_vulkan).is_err());
    }

    #[test]
    fn spirv_contract_fails_closed_for_unmodeled_capabilities() {
        let words = spirv_test_module(&[1, 65_535], 1);

        assert_eq!(
            vulkan_spirv_requirements(&words).unwrap_err(),
            VulkanError(
                "shader artifact declares SPIR-V capability 65535, but the runtime has no device contract for it"
                    .to_string()
            )
        );
    }

    #[test]
    fn spirv_contract_rejects_truncated_instructions() {
        let mut words = spirv_test_module(&[1], 1);
        words.push((4u32 << 16) | 54);

        assert!(vulkan_spirv_requirements(&words).is_err());
    }

    #[test]
    fn timeline_replay_offsets_preserve_values_and_reject_overflow() {
        assert_eq!(offset_timeline_value(17, 64).unwrap(), 81);
        assert_eq!(offset_timeline_value(u64::MAX, 0).unwrap(), u64::MAX);
        assert!(offset_timeline_value(u64::MAX, 1).is_err());
    }

    #[test]
    fn cooperative_bfloat16_matrix_shader_preserves_matrix_orientation() {
        let (Some(shader_path), Some(device_index)) = (
            std::env::var_os("NERVE_TEST_COOPERATIVE_BFLOAT16_SHADER"),
            std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX")
                .ok()
                .and_then(|value| value.parse::<usize>().ok()),
        ) else {
            eprintln!("skipping cooperative BF16 matrix test: explicit shader/device unset");
            return;
        };
        let bytes = std::fs::read(shader_path).unwrap();
        let spirv_words = bytes
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
            .collect::<Vec<_>>();
        let device = VulkanComputeDevice::new_for_physical_device_index(device_index).unwrap();
        assert!(device.supports_cooperative_bfloat16_shape(16, 16, 16));
        assert_eq!(device.subgroup_size(), 64);
        assert!(device.supports_compute_local_size_x(256));

        let input_values = (0..256)
            .map(|index| f32_to_bf16_bits((index % 16) as f32 + 1.0))
            .collect::<Vec<_>>();
        let row_major_weight = (0..256)
            .map(|index| {
                let row = index / 16;
                let column = index % 16;
                f32_to_bf16_bits(if row == column { 2.0 } else { 0.0 })
            })
            .collect::<Vec<_>>();
        let input = device.create_resident_buffer(512).unwrap();
        let output = device.create_resident_buffer(512).unwrap();
        let weight = device.create_resident_buffer(512).unwrap();
        input.write_bytes(&u16_bytes(&input_values)).unwrap();
        output.write_bytes(&vec![0; 512]).unwrap();
        weight.write_bytes(&u16_bytes(&row_major_weight)).unwrap();
        let dispatch = device
            .create_resident_kernel_dispatch(
                &spirv_words,
                &[
                    VulkanResidentKernelBufferBinding::new(0, &input, 512),
                    VulkanResidentKernelBufferBinding::new(1, &output, 512),
                    VulkanResidentKernelBufferBinding::new(2, &weight, 512),
                ],
                1,
                256,
                4,
            )
            .unwrap();
        device
            .run_resident_kernel_dispatch(&dispatch, &16u32.to_le_bytes())
            .unwrap();

        let expected = input_values
            .iter()
            .map(|value| f32_to_bf16_bits(bf16_bits_to_f32(*value) * 2.0))
            .collect::<Vec<_>>();
        assert_eq!(output.read_bytes(512).unwrap(), u16_bytes(&expected));
    }

    fn f32_to_bf16_bits(value: f32) -> u16 {
        let bits = value.to_bits();
        let lsb = (bits >> 16) & 1;
        ((bits + 0x7fff + lsb) >> 16) as u16
    }

    fn bf16_bits_to_f32(value: u16) -> f32 {
        f32::from_bits(u32::from(value) << 16)
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

    fn bytes_to_u32(bytes: &[u8]) -> Vec<u32> {
        bytes
            .chunks_exact(std::mem::size_of::<u32>())
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    #[test]
    fn moe_topk_uses_all_lanes_without_changing_tie_or_weight_semantics() {
        let device_index = std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX")
            .expect("NERVE_TEST_VULKAN_DEVICE_INDEX must select a compatible AMD GPU with sufficient safe remaining capacity")
            .parse::<usize>()
            .expect("NERVE_TEST_VULKAN_DEVICE_INDEX must be an integer");
        let template = std::fs::read_to_string(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("shaders")
                .join("moe_topk_bf16.comp.template"),
        )
        .unwrap();
        let rendered = template
            .replace("{{NUM_EXPERTS}}", "128")
            .replace("{{EXPERTS_PER_TOKEN}}", "4");
        let source_path = std::env::temp_dir().join(format!(
            "nerve-test-moe-topk-{}.comp",
            std::process::id()
        ));
        std::fs::write(&source_path, rendered).unwrap();
        let spirv_words = compile_shader_words_from_source_path(&source_path)
            .expect("parallel MoE top-k shader must compile");
        let _ = std::fs::remove_file(source_path);

        let mut scores = vec![f32_to_bf16_bits(-10.0); 128];
        scores[3] = f32_to_bf16_bits(10.0);
        scores[64] = f32_to_bf16_bits(10.0);
        scores[65] = f32_to_bf16_bits(9.0);
        scores[7] = f32_to_bf16_bits(8.0);
        scores[9] = f32_to_bf16_bits(8.0);

        let device = VulkanComputeDevice::new_for_physical_device_index(device_index).unwrap();
        let router_logits = device.create_resident_buffer(256).unwrap();
        router_logits.write_bytes(&u16_bytes(&scores)).unwrap();
        let expert_routes = device.create_resident_buffer(16).unwrap();
        expert_routes.write_bytes(&[0; 16]).unwrap();
        let selection_telemetry = device.create_resident_buffer(128 * 4).unwrap();
        selection_telemetry.write_bytes(&[0; 128 * 4]).unwrap();
        let dispatch = device
            .create_resident_kernel_dispatch(
                &spirv_words,
                &[
                    VulkanResidentKernelBufferBinding::new(0, &router_logits, 256)
                        .with_access(VulkanResidentKernelBufferAccess::Read),
                    VulkanResidentKernelBufferBinding::new(1, &expert_routes, 16)
                        .with_access(VulkanResidentKernelBufferAccess::Write),
                    VulkanResidentKernelBufferBinding::new(
                        2,
                        &selection_telemetry,
                        128 * 4,
                    )
                    .with_access(VulkanResidentKernelBufferAccess::ReadWrite),
                ],
                1,
                64,
                0,
            )
            .unwrap();
        device
            .run_resident_kernel_dispatch(&dispatch, &[])
            .unwrap();

        let routes = bytes_to_u32(&expert_routes.read_bytes(16).unwrap());
        assert_eq!(
            routes.iter().map(|route| route & 0xffff).collect::<Vec<_>>(),
            vec![3, 7, 64, 65],
        );
        let weight_sum = routes
            .iter()
            .map(|route| bf16_bits_to_f32((route >> 16) as u16))
            .sum::<f32>();
        assert!(
            (weight_sum - 1.0).abs() < 0.01,
            "selected softmax weights sum to {weight_sum}"
        );
        assert_eq!(routes[0] >> 16, routes[2] >> 16);
        let counts = bytes_to_u32(&selection_telemetry.read_bytes(128 * 4).unwrap());
        assert_eq!(
            counts
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, count)| *count > 0)
                .collect::<Vec<_>>(),
            vec![(3, 1), (7, 1), (64, 1), (65, 1)],
        );
    }

    #[test]
    fn persistently_mapped_copy_moves_exact_bound_bytes() {
        let source = [1u8, 2, 3, 4, 5, 6];
        let mut destination = [0u8; 6];
        let copy = VulkanResidentMappedBufferCopy {
            source_address: source.as_ptr() as usize,
            destination_address: destination.as_mut_ptr() as usize,
            byte_len: source.len(),
        };

        copy.run(source.len()).unwrap();

        assert_eq!(destination, source);
        assert!(copy.run(source.len() - 1).is_err());
    }

    #[test]
    fn resident_byte_buffer_can_be_reused_for_raw_model_memory() {
        let device = match selected_test_vulkan_device() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping Vulkan smoke: {error}");
                return;
            }
        };
        let buffer = device.create_resident_buffer(16).unwrap();

        buffer.write_bytes(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(buffer.byte_capacity(), 16);
        assert_eq!(buffer.read_bytes(5).unwrap(), vec![1, 2, 3, 4, 5]);

        buffer.write_bytes(&[10, 20, 30]).unwrap();
        assert_eq!(buffer.read_bytes(3).unwrap(), vec![10, 20, 30]);
        assert!(buffer.read_bytes(17).is_err());
        assert!(buffer.write_bytes(&[0; 17]).is_err());
    }

    #[test]
    fn generic_resident_kernel_dispatch_runs_on_raw_byte_buffer() {
        let Some(spirv_words) = compile_test_shader_words() else {
            eprintln!("skipping Vulkan smoke: no GLSL to SPIR-V compiler found");
            return;
        };
        let device = match selected_test_vulkan_device() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping Vulkan smoke: {error}");
                return;
            }
        };
        let buffer = device.create_resident_buffer(12).unwrap();
        buffer.write_bytes(&u32_bytes(&[1, 2, 41])).unwrap();
        let binding = VulkanResidentKernelBufferBinding::new(0, &buffer, 12);

        let dispatch = device
            .create_resident_kernel_dispatch(&spirv_words, &[binding], 1, 64, 0)
            .unwrap();
        device.run_resident_kernel_dispatch(&dispatch, &[]).unwrap();

        assert_eq!(dispatch.descriptor_count(), 1);
        assert_eq!(dispatch.workgroup_count_x(), 1);
        assert_eq!(dispatch.push_constant_byte_count(), 0);
        assert_eq!(
            bytes_to_u32(&buffer.read_bytes(12).unwrap()),
            vec![2, 3, 42]
        );
    }

    #[test]
    fn sparse_moe_prequant_gate_matches_internal_quantization_byte_for_byte() {
        let Some(raw_device_index) = std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX").ok() else {
            eprintln!("skipping sparse MoE prequant equivalence: explicit Vulkan device index unset");
            return;
        };
        let device_index = raw_device_index
            .parse::<usize>()
            .expect("NERVE_TEST_VULKAN_DEVICE_INDEX must be an integer");
        let shader_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders");
        let render = |template_name: &str, replacements: &[(&str, &str)]| {
            let mut source =
                std::fs::read_to_string(shader_dir.join(template_name)).unwrap();
            for (pattern, value) in replacements {
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
        };
        let quantize_words = render(
            "quantize_fp8_e4m3.comp.template",
            &[("{{BLOCK_COLUMNS}}", "128"), ("{{ELEMENT_COUNT}}", "128")],
        );
        let gate_shape = [
            ("{{BLOCK_ROWS}}", "128"),
            ("{{BLOCK_COLUMNS}}", "128"),
            ("{{HIDDEN_SIZE}}", "128"),
            ("{{INTERMEDIATE_SIZE}}", "64"),
            ("{{NUM_EXPERTS}}", "1"),
            ("{{EXPERTS_PER_TOKEN}}", "1"),
        ];
        let mut legacy_shape = gate_shape.to_vec();
        legacy_shape.extend([
            ("{{LOCAL_SIZE_X}}", "512"),
            ("{{TILE_ROWS}}", "32"),
        ]);
        let legacy_words = render(
            "sparse_moe_gate_up_fp8_e4m3.comp.template",
            &legacy_shape,
        );
        let mut prequant_shape = gate_shape.to_vec();
        prequant_shape.extend([
            ("{{LOCAL_SIZE_X}}", "512"),
            ("{{TILE_ROWS}}", "32"),
        ]);
        let prequant_words = render(
            "sparse_moe_gate_up_prequant_fp8_e4m3.comp.template",
            &prequant_shape,
        );

        let device = VulkanComputeDevice::new_for_physical_device_index(device_index).unwrap();
        let hidden_values = (0..128)
            .map(|index| {
                let value = ((index as i32 % 19) - 9) as f32 * 0.125;
                f32_to_bf16_bits(value)
            })
            .collect::<Vec<_>>();
        let hidden = device.create_resident_buffer(256).unwrap();
        hidden.write_bytes(&u16_bytes(&hidden_values)).unwrap();
        let quantized = device.create_resident_buffer(128).unwrap();
        quantized.write_bytes(&[0; 128]).unwrap();
        let activation_scale = device.create_resident_buffer(4).unwrap();
        activation_scale.write_bytes(&[0; 4]).unwrap();
        let routes = device.create_resident_buffer(4).unwrap();
        routes.write_bytes(&u32_bytes(&[0])).unwrap();
        let weights = device.create_resident_buffer(16_384).unwrap();
        weights
            .write_bytes(&u32_bytes(&vec![0x3030_3030; 4_096]))
            .unwrap();
        let weight_scale = device.create_resident_buffer(4).unwrap();
        weight_scale
            .write_bytes(&u16_bytes(&[f32_to_bf16_bits(0.03125), 0]))
            .unwrap();
        let legacy_output = device.create_resident_buffer(128).unwrap();
        legacy_output.write_bytes(&[0; 128]).unwrap();
        let prequant_output = device.create_resident_buffer(128).unwrap();
        prequant_output.write_bytes(&[0; 128]).unwrap();

        let quantize = device
            .create_resident_kernel_dispatch(
                &quantize_words,
                &[
                    VulkanResidentKernelBufferBinding::new(0, &hidden, 256)
                        .with_access(VulkanResidentKernelBufferAccess::Read),
                    VulkanResidentKernelBufferBinding::new(1, &quantized, 128)
                        .with_access(VulkanResidentKernelBufferAccess::Write),
                    VulkanResidentKernelBufferBinding::new(2, &activation_scale, 4)
                        .with_access(VulkanResidentKernelBufferAccess::Write),
                ],
                1,
                32,
                0,
            )
            .unwrap();
        device
            .run_resident_kernel_dispatch(&quantize, &[])
            .unwrap();

        let legacy = device
            .create_resident_kernel_dispatch(
                &legacy_words,
                &[
                    VulkanResidentKernelBufferBinding::new(0, &hidden, 256)
                        .with_access(VulkanResidentKernelBufferAccess::Read),
                    VulkanResidentKernelBufferBinding::new(1, &routes, 4)
                        .with_access(VulkanResidentKernelBufferAccess::Read),
                    VulkanResidentKernelBufferBinding::new(2, &legacy_output, 128)
                        .with_access(VulkanResidentKernelBufferAccess::Write),
                    VulkanResidentKernelBufferBinding::new(3, &weights, 16_384)
                        .with_access(VulkanResidentKernelBufferAccess::Read),
                    VulkanResidentKernelBufferBinding::new(4, &weight_scale, 4)
                        .with_access(VulkanResidentKernelBufferAccess::Read),
                ],
                2,
                512,
                4,
            )
            .unwrap();
        device
            .run_resident_kernel_dispatch(&legacy, &0u32.to_le_bytes())
            .unwrap();

        let prequant = device
            .create_resident_kernel_dispatch(
                &prequant_words,
                &[
                    VulkanResidentKernelBufferBinding::new(0, &quantized, 128)
                        .with_access(VulkanResidentKernelBufferAccess::Read),
                    VulkanResidentKernelBufferBinding::new(1, &activation_scale, 4)
                        .with_access(VulkanResidentKernelBufferAccess::Read),
                    VulkanResidentKernelBufferBinding::new(2, &routes, 4)
                        .with_access(VulkanResidentKernelBufferAccess::Read),
                    VulkanResidentKernelBufferBinding::new(3, &prequant_output, 128)
                        .with_access(VulkanResidentKernelBufferAccess::Write),
                    VulkanResidentKernelBufferBinding::new(4, &weights, 16_384)
                        .with_access(VulkanResidentKernelBufferAccess::Read),
                    VulkanResidentKernelBufferBinding::new(5, &weight_scale, 4)
                        .with_access(VulkanResidentKernelBufferAccess::Read),
                ],
                2,
                512,
                4,
            )
            .unwrap();
        device
            .run_resident_kernel_dispatch(&prequant, &0u32.to_le_bytes())
            .unwrap();

        assert_eq!(
            prequant_output.read_bytes(128).unwrap(),
            legacy_output.read_bytes(128).unwrap()
        );
    }

    #[test]
    fn sparse_moe_route_compaction_groups_selected_routes_on_device() {
        let Some(raw_device_index) = std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX").ok() else {
            eprintln!("skipping sparse MoE route compaction: explicit Vulkan device index unset");
            return;
        };
        let device_index = raw_device_index
            .parse::<usize>()
            .expect("NERVE_TEST_VULKAN_DEVICE_INDEX must be an integer");
        let template = std::fs::read_to_string(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("shaders")
                .join("moe_route_compact_batch1.comp.template"),
        )
        .unwrap();
        let rendered = template
            .replace("{{INTERMEDIATE_SIZE}}", "4")
            .replace("{{EXPERTS_PER_TOKEN}}", "2")
            .replace("{{TILES_PER_ROUTE}}", "7");
        let source_path = std::env::temp_dir().join(format!(
            "nerve-test-sparse-moe-route-compact-{}.comp",
            std::process::id()
        ));
        std::fs::write(&source_path, rendered).unwrap();
        let spirv_words = compile_shader_words_from_source_path(&source_path)
            .expect("sparse MoE route compaction shader must compile");
        let _ = std::fs::remove_file(source_path);

        let device = VulkanComputeDevice::new_for_physical_device_index(device_index).unwrap();
        let expert_routes = device.create_resident_buffer(24).unwrap();
        expert_routes
            .write_bytes(&u32_bytes(&[5, 1, 3, 1, 4, 2]))
            .unwrap();
        let expert_intermediates = device.create_resident_buffer(72).unwrap();
        expert_intermediates.write_bytes(&[0; 72]).unwrap();
        let batch_control = device.create_resident_buffer(28).unwrap();
        batch_control
            .write_bytes(&u32_bytes(&[3, 0, 0, 0, 0, 0, 0]))
            .unwrap();

        let dispatch = device
            .create_resident_kernel_dispatch_2d(
                &spirv_words,
                &[
                    VulkanResidentKernelBufferBinding::new(1, &expert_routes, 24)
                        .with_access(VulkanResidentKernelBufferAccess::Read),
                    VulkanResidentKernelBufferBinding::new(2, &expert_intermediates, 72)
                        .with_access(VulkanResidentKernelBufferAccess::Write),
                    VulkanResidentKernelBufferBinding::new(31, &batch_control, 28)
                        .with_access(VulkanResidentKernelBufferAccess::ReadWrite),
                ],
                2,
                3,
                64,
                0,
            )
            .unwrap();
        device
            .run_resident_kernel_dispatch(&dispatch, &[])
            .unwrap();

        let words = bytes_to_u32(&expert_intermediates.read_bytes(72).unwrap());
        let compact_source_indices = [4usize, 5, 10, 11, 16, 17]
            .into_iter()
            .map(|index| words[index])
            .collect::<Vec<_>>();
        assert_eq!(compact_source_indices, vec![1, 3, 5, 2, 4, 0]);

        expert_intermediates.write_bytes(&[0; 72]).unwrap();
        batch_control
            .write_bytes(&u32_bytes(&[3, 2, 2, 0, 0, 0, 0]))
            .unwrap();
        device
            .run_resident_kernel_dispatch(&dispatch, &[])
            .unwrap();

        let words = bytes_to_u32(&expert_intermediates.read_bytes(72).unwrap());
        assert_eq!([words[4], words[5]], [2, 5]);
        assert_eq!(
            bytes_to_u32(&batch_control.read_bytes(28).unwrap()),
            vec![3, 2, 2, 2, 14, 1, 1],
        );
    }

    #[test]
    fn resident_kernel_sequence_records_and_replays_composed_dispatches() {
        let Some(spirv_words) = compile_test_shader_words() else {
            eprintln!("skipping Vulkan smoke: no GLSL to SPIR-V compiler found");
            return;
        };
        let device = match selected_test_vulkan_device() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping Vulkan smoke: {error}");
                return;
            }
        };
        let buffer = device.create_resident_buffer(12).unwrap();
        buffer.write_bytes(&u32_bytes(&[1, 2, 41])).unwrap();
        let binding = VulkanResidentKernelBufferBinding::new(0, &buffer, 12);
        let dispatch = device
            .create_resident_kernel_dispatch(&spirv_words, &[binding], 1, 64, 0)
            .unwrap();
        let sequence = device.create_resident_kernel_sequence().unwrap();
        assert!(!sequence.has_recorded_commands());
        assert!(
            device
                .run_recorded_resident_kernel_sequence(&sequence)
                .is_err()
        );

        device
            .run_resident_kernel_sequence(
                &sequence,
                &[
                    VulkanResidentKernelSequenceStep::new(&dispatch, &[]),
                    VulkanResidentKernelSequenceStep::new(&dispatch, &[]),
                ],
            )
            .unwrap();
        assert!(sequence.has_recorded_commands());

        assert_eq!(
            bytes_to_u32(&buffer.read_bytes(12).unwrap()),
            vec![3, 4, 43]
        );

        device
            .run_recorded_resident_kernel_sequence(&sequence)
            .unwrap();
        assert_eq!(
            bytes_to_u32(&buffer.read_bytes(12).unwrap()),
            vec![5, 6, 45]
        );
    }

    #[test]
    fn timestamped_resident_sequence_exposes_progress_without_changing_replay() {
        let spirv_words = compile_test_shader_words()
            .expect("timestamp progress test requires GLSL to SPIR-V tooling");
        let device = selected_test_vulkan_device().unwrap();
        let buffer = device.create_resident_buffer(12).unwrap();
        buffer.write_bytes(&u32_bytes(&[1, 2, 41])).unwrap();
        let dispatch = device
            .create_resident_kernel_dispatch_labeled(
                &spirv_words,
                &[VulkanResidentKernelBufferBinding::new(0, &buffer, 12)],
                1,
                64,
                0,
                Some("op=test_progress node=test_sequence".to_string()),
            )
            .unwrap();
        let sequence = device.create_timestamped_resident_kernel_sequence().unwrap();
        let steps = [VulkanResidentKernelSequenceStep::new(&dispatch, &[])];

        device.run_resident_kernel_sequence(&sequence, &steps).unwrap();
        assert!(device
            .read_recorded_resident_kernel_sequence_duration_ns(&sequence)
            .unwrap()
            > 0);
        device
            .submit_recorded_resident_kernel_sequence(&sequence)
            .unwrap();
        assert!(
            device
                .submit_recorded_resident_kernel_sequence(&sequence)
                .unwrap_err()
                .0
                .contains("pending completion")
        );
        device.wait_resident_kernel_sequence(&sequence).unwrap();
        assert_eq!(
            bytes_to_u32(&buffer.read_bytes(12).unwrap()),
            vec![3, 4, 43]
        );
        assert!(sequence.pending_wait_points.borrow().is_empty());
        assert!(sequence.completion.state.pending_value.get().is_none());
        assert_eq!(sequence.completion.state.last_reserved_value.get(), 2);
        assert_eq!(
            unsafe {
                device
                    .device
                    .get_semaphore_counter_value(sequence.completion.semaphore())
            }
            .unwrap(),
            2
        );
    }

    #[test]
    fn deferred_resident_queue_batch_reserves_completion_per_replay() {
        let spirv_words = compile_test_shader_words()
            .expect("deferred completion test requires GLSL to SPIR-V tooling");
        let device = selected_test_vulkan_device().unwrap();
        let buffer = device.create_resident_buffer(12).unwrap();
        buffer.write_bytes(&u32_bytes(&[1, 2, 41])).unwrap();
        let dispatch = device
            .create_resident_kernel_dispatch(
                &spirv_words,
                &[VulkanResidentKernelBufferBinding::new(0, &buffer, 12)],
                1,
                64,
                0,
            )
            .unwrap();
        let sequence = device.create_resident_kernel_sequence().unwrap();
        device
            .record_resident_kernel_sequence(
                &sequence,
                &[VulkanResidentKernelSequenceStep::new(&dispatch, &[])],
            )
            .unwrap();
        let batch = VulkanResidentQueueSubmissionBatch::new();
        batch
            .enqueue_recorded_sequence(&device, &sequence, &[], &[], true)
            .unwrap();
        let template = batch.mount().unwrap();
        assert!(sequence.completion.state.pending_value.get().is_none());

        for expected in [vec![2, 3, 42], vec![3, 4, 43]] {
            let previous_progress = device
                .compute_queue_submission
                .latest_progress_point()
                .map(|(_, value)| value)
                .unwrap_or_default();
            template.submit_with_timeline_value_offset(0).unwrap();
            let (progress_semaphore, progress_value) = device
                .compute_queue_submission
                .latest_progress_point()
                .expect("resident queue submission must publish queue progress");
            assert!(progress_value > previous_progress);
            assert!(sequence.completion.state.pending_value.get().is_some());
            device.wait_resident_kernel_sequence(&sequence).unwrap();
            assert!(
                unsafe {
                    device
                        .device
                        .get_semaphore_counter_value(progress_semaphore)
                }
                .unwrap()
                    >= progress_value
            );
            assert!(sequence.completion.state.pending_value.get().is_none());
            assert_eq!(bytes_to_u32(&buffer.read_bytes(12).unwrap()), expected);
        }
    }

    #[test]
    fn completed_queue_resource_can_replay_after_an_external_epoch_join() {
        let spirv_words = compile_test_shader_words()
            .expect("queue completion replay test requires GLSL to SPIR-V tooling");
        let device = selected_test_vulkan_device().unwrap();
        let buffer = device.create_resident_buffer(12).unwrap();
        buffer.write_bytes(&u32_bytes(&[1, 2, 41])).unwrap();
        let dispatch = device
            .create_resident_kernel_dispatch(
                &spirv_words,
                &[VulkanResidentKernelBufferBinding::new(0, &buffer, 12)],
                1,
                64,
                0,
            )
            .unwrap();
        let sequence = device.create_resident_kernel_sequence().unwrap();
        device
            .record_resident_kernel_sequence(
                &sequence,
                &[VulkanResidentKernelSequenceStep::new(&dispatch, &[])],
            )
            .unwrap();
        let batch = VulkanResidentQueueSubmissionBatch::new();
        batch
            .enqueue_recorded_sequence(&device, &sequence, &[], &[], true)
            .unwrap();
        let template = batch.mount().unwrap();

        template.submit_with_timeline_value_offset(0).unwrap();
        device.quiesce().unwrap();
        assert_eq!(sequence.completion.pending_value(), Some(1));
        template.submit_with_timeline_value_offset(0).unwrap();
        assert_eq!(sequence.completion.pending_value(), Some(2));
        device.wait_resident_kernel_sequence(&sequence).unwrap();

        assert_eq!(
            bytes_to_u32(&buffer.read_bytes(12).unwrap()),
            vec![3, 4, 43]
        );
    }

    #[test]
    fn resident_kernel_sequence_rerecords_changed_push_constants() {
        let Some(spirv_words) = compile_test_shader_words() else {
            eprintln!("skipping Vulkan sequence test: no GLSL to SPIR-V compiler found");
            return;
        };
        let device = match selected_test_vulkan_device() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping Vulkan sequence test: {error}");
                return;
            }
        };
        let buffer = device.create_resident_buffer(4).unwrap();
        let dispatch = device
            .create_resident_kernel_dispatch(
                &spirv_words,
                &[VulkanResidentKernelBufferBinding::new(0, &buffer, 4)],
                1,
                64,
                4,
            )
            .unwrap();
        let sequence = device.create_resident_kernel_sequence().unwrap();
        let first = 17u32.to_le_bytes();
        let second = 29u32.to_le_bytes();

        reset_vulkan_resident_execution_counters();
        device
            .record_resident_kernel_sequence(
                &sequence,
                &[VulkanResidentKernelSequenceStep::new(&dispatch, &first)],
            )
            .unwrap();
        device
            .record_resident_kernel_sequence(
                &sequence,
                &[VulkanResidentKernelSequenceStep::new(&dispatch, &second)],
            )
            .unwrap();
        device
            .record_resident_kernel_sequence(
                &sequence,
                &[VulkanResidentKernelSequenceStep::new(&dispatch, &second)],
            )
            .unwrap();

        let counters = vulkan_resident_execution_counters();
        assert_eq!(counters.resident_sequence_prepare_calls, 3);
        assert_eq!(counters.resident_sequence_recorded_command_buffers, 2);
        assert_eq!(counters.resident_sequence_reused_command_buffers, 1);
    }

    #[test]
    fn separate_resident_sequences_publish_compute_writes_to_the_next_sequence() {
        let Some(spirv_words) = compile_test_shader_words() else {
            eprintln!("skipping Vulkan sequence boundary test: no GLSL to SPIR-V compiler found");
            return;
        };
        let Some(device_index) = std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
        else {
            eprintln!("skipping Vulkan sequence boundary test: explicit Vulkan device index unset");
            return;
        };
        let device = VulkanComputeDevice::new_for_physical_device_index(device_index).unwrap();
        let buffer = device.create_resident_buffer(12).unwrap();
        buffer.write_bytes(&u32_bytes(&[1, 2, 41])).unwrap();
        let dispatch = device
            .create_resident_kernel_dispatch(
                &spirv_words,
                &[VulkanResidentKernelBufferBinding::new(0, &buffer, 12)],
                1,
                64,
                0,
            )
            .unwrap();
        let producer = device.create_resident_kernel_sequence().unwrap();
        let consumer = device.create_resident_kernel_sequence().unwrap();

        device
            .run_resident_kernel_sequence(
                &producer,
                &[VulkanResidentKernelSequenceStep::new(&dispatch, &[])],
            )
            .unwrap();
        device
            .run_resident_kernel_sequence(
                &consumer,
                &[VulkanResidentKernelSequenceStep::new(&dispatch, &[])],
            )
            .unwrap();

        assert_eq!(
            bytes_to_u32(&buffer.read_bytes(12).unwrap()),
            vec![3, 4, 43]
        );
    }

    #[test]
    fn cross_device_shared_resident_memory_reuses_persistent_semaphore_dependencies() {
        let Some(spirv_words) = compile_test_shader_words() else {
            eprintln!("skipping cross-device Vulkan test: no GLSL to SPIR-V compiler found");
            return;
        };
        let (Some(owner_index), Some(worker_index)) = (
            std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX")
                .ok()
                .and_then(|value| value.parse::<usize>().ok()),
            std::env::var("NERVE_TEST_VULKAN_SECONDARY_DEVICE_INDEX")
                .ok()
                .and_then(|value| value.parse::<usize>().ok()),
        ) else {
            eprintln!("skipping cross-device Vulkan test: explicit device pair unset");
            return;
        };
        assert_ne!(owner_index, worker_index);

        let owner = VulkanComputeDevice::new_for_physical_device_index(owner_index).unwrap();
        let worker = VulkanComputeDevice::new_for_physical_device_index(worker_index).unwrap();
        assert!(owner.supports_opaque_fd_timeline_semaphores());
        assert!(worker.supports_opaque_fd_timeline_semaphores());

        let shared = owner
            .create_shared_resident_buffers(&[&worker], 12)
            .unwrap();
        assert_eq!(shared.buffers.len(), 2);
        match shared.route {
            VulkanSharedResidentBufferRoute::ExternalDeviceLocal => {
                assert!(shared.external_device_local_error.is_none());
            }
            VulkanSharedResidentBufferRoute::SharedHost => {
                assert!(shared.external_device_local_error.is_some());
            }
        }
        let owner_buffer = &shared.buffers[0];
        let worker_buffer = &shared.buffers[1];
        assert!(owner_buffer.shares_storage_with(worker_buffer));
        owner_buffer.write_bytes(&u32_bytes(&[1, 2, 41])).unwrap();

        let owner_dispatch = owner
            .create_resident_kernel_dispatch(
                &spirv_words,
                &[VulkanResidentKernelBufferBinding::new(0, &owner_buffer, 12)],
                1,
                64,
                0,
            )
            .unwrap();
        let worker_dispatch = worker
            .create_resident_kernel_dispatch(
                &spirv_words,
                &[VulkanResidentKernelBufferBinding::new(
                    0,
                    &worker_buffer,
                    12,
                )],
                1,
                64,
                0,
            )
            .unwrap();
        let owner_first = owner.create_resident_kernel_sequence().unwrap();
        owner
            .record_resident_kernel_sequence(
                &owner_first,
                &[VulkanResidentKernelSequenceStep::new(&owner_dispatch, &[])],
            )
            .unwrap();
        let worker_sequence = worker.create_resident_kernel_sequence().unwrap();
        worker
            .record_resident_kernel_sequence(
                &worker_sequence,
                &[VulkanResidentKernelSequenceStep::new(&worker_dispatch, &[])],
            )
            .unwrap();
        let owner_last = owner.create_resident_kernel_sequence().unwrap();
        owner
            .record_resident_kernel_sequence(
                &owner_last,
                &[VulkanResidentKernelSequenceStep::new(&owner_dispatch, &[])],
            )
            .unwrap();

        let ready_source = owner
            .create_opaque_fd_exportable_timeline_semaphore(0)
            .unwrap();
        let ready_wait = worker.create_timeline_semaphore(0).unwrap();
        worker
            .import_timeline_semaphore_opaque_fd(
                &ready_wait,
                owner
                    .export_timeline_semaphore_opaque_fd(&ready_source)
                    .unwrap(),
            )
            .unwrap();
        let done_source = worker
            .create_opaque_fd_exportable_timeline_semaphore(0)
            .unwrap();
        let done_wait = owner.create_timeline_semaphore(0).unwrap();
        owner
            .import_timeline_semaphore_opaque_fd(
                &done_wait,
                worker
                    .export_timeline_semaphore_opaque_fd(&done_source)
                    .unwrap(),
            )
            .unwrap();

        for dependency_value in 1..=2 {
            owner
                .submit_recorded_resident_kernel_sequence_with_timeline_semaphores(
                    &owner_first,
                    &[],
                    &[VulkanTimelineSemaphorePoint::new(
                        &ready_source,
                        dependency_value,
                    )],
                )
                .unwrap();
            worker
                .submit_recorded_resident_kernel_sequence_with_timeline_semaphores(
                    &worker_sequence,
                    &[VulkanTimelineSemaphorePoint::new(
                        &ready_wait,
                        dependency_value,
                    )],
                    &[VulkanTimelineSemaphorePoint::new(
                        &done_source,
                        dependency_value,
                    )],
                )
                .unwrap();
            owner
                .submit_recorded_resident_kernel_sequence_with_timeline_semaphores(
                    &owner_last,
                    &[VulkanTimelineSemaphorePoint::new(
                        &done_wait,
                        dependency_value,
                    )],
                    &[],
                )
                .unwrap();
            owner.wait_resident_kernel_sequence(&owner_last).unwrap();
        }

        let owner_ready_batch = VulkanResidentQueueSubmissionBatch::new();
        owner_ready_batch
            .enqueue_recorded_sequence(
                &owner,
                &owner_first,
                &[],
                &[VulkanTimelineSemaphorePoint::new(&ready_source, 1)],
                false,
            )
            .unwrap();
        let owner_ready_template = owner_ready_batch.mount().unwrap();
        let owner_done_batch = VulkanResidentQueueSubmissionBatch::new();
        owner_done_batch
            .enqueue_recorded_sequence(
                &owner,
                &owner_last,
                &[VulkanTimelineSemaphorePoint::new(&done_wait, 1)],
                &[],
                true,
            )
            .unwrap();
        let owner_done_template = owner_done_batch.mount().unwrap();
        for dependency_value in 3..=4 {
            let timeline_value_offset = dependency_value - 1;
            owner_ready_template
                .submit_with_timeline_value_offset(timeline_value_offset)
                .unwrap();
            worker
                .submit_recorded_resident_kernel_sequence_with_timeline_semaphores(
                    &worker_sequence,
                    &[VulkanTimelineSemaphorePoint::new(
                        &ready_wait,
                        dependency_value,
                    )],
                    &[VulkanTimelineSemaphorePoint::new(
                        &done_source,
                        dependency_value,
                    )],
                )
                .unwrap();
            owner_done_template
                .submit_with_timeline_value_offset(timeline_value_offset)
                .unwrap();
            owner.wait_resident_kernel_sequence(&owner_last).unwrap();
        }

        assert_eq!(
            bytes_to_u32(&owner_buffer.read_bytes(12).unwrap()),
            vec![13, 14, 53]
        );
    }

    #[test]
    fn multi_device_resident_queue_batch_uses_one_host_submission_per_device() {
        fn exact_device_from_env(name: &str) -> VulkanComputeDevice {
            let encoded = std::env::var(name)
                .unwrap_or_else(|_| panic!("{name} must select an approved discrete AMD UUID"));
            let encoded = encoded
                .strip_prefix("vulkan-uuid:")
                .expect("Vulkan test UUID must use a vulkan-uuid: prefix");
            assert_eq!(encoded.len(), 32);
            let mut uuid = [0u8; 16];
            for (index, byte) in uuid.iter_mut().enumerate() {
                *byte = u8::from_str_radix(&encoded[index * 2..index * 2 + 2], 16)
                    .expect("Vulkan test UUID must be hexadecimal");
            }
            let device = VulkanComputeDevice::new_for_device_uuid(uuid).unwrap();
            assert_eq!(device.physical_device_id(), format!("vulkan-uuid:{encoded}"));
            device
        }

        let spirv_words =
            compile_test_shader_words().expect("Vulkan queue-batch test requires GLSL tooling");
        let owner = exact_device_from_env("NERVE_TEST_VULKAN_DEVICE_UUID");
        let worker = exact_device_from_env("NERVE_TEST_VULKAN_SECONDARY_DEVICE_UUID");
        assert_ne!(owner.physical_device_id(), worker.physical_device_id());
        let owner_buffer = owner.create_resident_buffer(12).unwrap();
        owner_buffer.write_bytes(&u32_bytes(&[1, 2, 41])).unwrap();
        let worker_buffer = worker.create_resident_buffer(12).unwrap();
        worker_buffer.write_bytes(&u32_bytes(&[1, 2, 41])).unwrap();
        let owner_dispatch = owner
            .create_resident_kernel_dispatch(
                &spirv_words,
                &[VulkanResidentKernelBufferBinding::new(
                    0,
                    &owner_buffer,
                    12,
                )],
                1,
                64,
                0,
            )
            .unwrap();
        let worker_dispatch = worker
            .create_resident_kernel_dispatch(
                &spirv_words,
                &[VulkanResidentKernelBufferBinding::new(
                    0,
                    &worker_buffer,
                    12,
                )],
                1,
                64,
                0,
            )
            .unwrap();
        let owner_first = owner.create_resident_kernel_sequence().unwrap();
        let owner_second = owner.create_resident_kernel_sequence().unwrap();
        let worker_sequence = worker.create_resident_kernel_sequence().unwrap();
        for sequence in [&owner_first, &owner_second] {
            owner
                .record_resident_kernel_sequence(
                    sequence,
                    &[VulkanResidentKernelSequenceStep::new(&owner_dispatch, &[])],
                )
                .unwrap();
        }
        worker
            .record_resident_kernel_sequence(
                &worker_sequence,
                &[VulkanResidentKernelSequenceStep::new(&worker_dispatch, &[])],
            )
            .unwrap();
        let ready_source = owner
            .create_opaque_fd_exportable_timeline_semaphore(0)
            .unwrap();
        let ready_wait = worker.create_timeline_semaphore(0).unwrap();
        worker
            .import_timeline_semaphore_opaque_fd(
                &ready_wait,
                owner
                    .export_timeline_semaphore_opaque_fd(&ready_source)
                    .unwrap(),
            )
            .unwrap();
        let batch = VulkanResidentQueueSubmissionBatch::new();
        batch
            .enqueue_recorded_sequence(&owner, &owner_first, &[], &[], false)
            .unwrap();
        batch
            .enqueue_recorded_sequence(
                &owner,
                &owner_second,
                &[],
                &[VulkanTimelineSemaphorePoint::new(&ready_source, 1)],
                false,
            )
            .unwrap();
        batch
            .enqueue_recorded_sequence(
                &worker,
                &worker_sequence,
                &[VulkanTimelineSemaphorePoint::new(&ready_wait, 1)],
                &[],
                true,
            )
            .unwrap();
        let template = batch.mount().unwrap();
        assert_eq!(template.submission_count(), 3);
        assert_eq!(template.host_queue_submit_count(), 2);
        reset_vulkan_resident_execution_counters();

        template.submit_with_timeline_value_offset(0).unwrap();
        worker
            .wait_resident_kernel_sequence(&worker_sequence)
            .unwrap();

        assert_eq!(
            bytes_to_u32(&owner_buffer.read_bytes(12).unwrap()),
            vec![3, 4, 43]
        );
        assert_eq!(
            bytes_to_u32(&worker_buffer.read_bytes(12).unwrap()),
            vec![2, 3, 42]
        );
        let counters = vulkan_resident_execution_counters();
        assert_eq!(counters.resident_queue_batch_submits, 2);
        assert_eq!(counters.resident_queue_batch_commands, 3);
        assert_eq!(counters.resident_sequence_queue_submits, 0);
        assert_eq!(counters.resident_sequence_completion_waits, 1);
    }

    #[test]
    fn cross_device_shared_predicate_suppresses_downstream_compute() {
        let spirv_words =
            compile_test_shader_words().expect("Vulkan predicate test requires a GLSL compiler");
        let owner_index = std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX")
            .expect("NERVE_TEST_VULKAN_DEVICE_INDEX must select a compatible AMD GPU with sufficient safe remaining capacity")
            .parse::<usize>()
            .expect("NERVE_TEST_VULKAN_DEVICE_INDEX must be an integer");
        let worker_index = std::env::var("NERVE_TEST_VULKAN_SECONDARY_DEVICE_INDEX")
            .expect("NERVE_TEST_VULKAN_SECONDARY_DEVICE_INDEX must select a compatible AMD GPU with sufficient safe remaining capacity")
            .parse::<usize>()
            .expect("NERVE_TEST_VULKAN_SECONDARY_DEVICE_INDEX must be an integer");
        assert_ne!(owner_index, worker_index);
        let owner = VulkanComputeDevice::new_for_physical_device_index(owner_index).unwrap();
        let worker = VulkanComputeDevice::new_for_physical_device_index(worker_index).unwrap();
        let shared = owner
            .create_shared_conditional_resident_buffers(&[&worker], 4)
            .unwrap();
        let owner_predicate = &shared.buffers[0];
        let worker_predicate = &shared.buffers[1];
        assert!(owner_predicate.shares_storage_with(worker_predicate));
        owner_predicate
            .write_bytes(&u32::MAX.to_le_bytes())
            .unwrap();

        let owner_dispatch = owner
            .create_resident_kernel_dispatch(
                &spirv_words,
                &[VulkanResidentKernelBufferBinding::new(
                    0,
                    owner_predicate,
                    4,
                )],
                1,
                64,
                0,
            )
            .unwrap();
        let worker_output = worker.create_resident_buffer(12).unwrap();
        worker_output
            .write_bytes(&u32_bytes(&[1, 2, 41]))
            .unwrap();
        let worker_dispatch = worker
            .create_resident_kernel_dispatch(
                &spirv_words,
                &[VulkanResidentKernelBufferBinding::new(
                    0,
                    &worker_output,
                    12,
                )],
                1,
                64,
                0,
            )
            .unwrap();
        let owner_sequence = owner.create_resident_kernel_sequence().unwrap();
        owner
            .record_resident_kernel_sequence(
                &owner_sequence,
                &[VulkanResidentKernelSequenceStep::new(
                    &owner_dispatch,
                    &[],
                )],
            )
            .unwrap();
        let worker_sequence = worker.create_resident_kernel_sequence().unwrap();
        worker
            .record_resident_kernel_sequence(
                &worker_sequence,
                &[VulkanResidentKernelSequenceStep::new_conditional(
                    &worker_dispatch,
                    &[],
                    worker_predicate,
                    0,
                    false,
                    1,
                )
                .unwrap()],
            )
            .unwrap();
        let owner_signal = owner
            .create_opaque_fd_exportable_timeline_semaphore(0)
            .unwrap();
        let worker_wait = worker.create_timeline_semaphore(0).unwrap();
        worker
            .import_timeline_semaphore_opaque_fd(
                &worker_wait,
                owner
                    .export_timeline_semaphore_opaque_fd(&owner_signal)
                    .unwrap(),
            )
            .unwrap();

        for timeline_value in 1..=2 {
            owner
                .submit_recorded_resident_kernel_sequence_with_timeline_semaphores(
                    &owner_sequence,
                    &[],
                    &[VulkanTimelineSemaphorePoint::new(
                        &owner_signal,
                        timeline_value,
                    )],
                )
                .unwrap();
            worker
                .submit_recorded_resident_kernel_sequence_with_timeline_semaphores(
                    &worker_sequence,
                    &[VulkanTimelineSemaphorePoint::new(
                        &worker_wait,
                        timeline_value,
                    )],
                    &[],
                )
                .unwrap();
            worker
                .wait_resident_kernel_sequence(&worker_sequence)
                .unwrap();
            if timeline_value == 1 {
                assert_eq!(
                    bytes_to_u32(&worker_output.read_bytes(12).unwrap()),
                    vec![1, 2, 41]
                );
            }
        }
        assert_eq!(
            bytes_to_u32(&owner_predicate.read_bytes(4).unwrap()),
            vec![1]
        );
        assert_eq!(
            bytes_to_u32(&worker_output.read_bytes(12).unwrap()),
            vec![2, 3, 42]
        );
    }

    #[test]
    fn shared_imported_predicate_restored_by_shader_enables_same_queue_continuation() {
        let spirv_words = compile_test_shader_words()
            .expect("Vulkan predicate restoration test requires a GLSL compiler");
        let owner_index = std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX")
            .expect("NERVE_TEST_VULKAN_DEVICE_INDEX must select the predicate owner")
            .parse::<usize>()
            .expect("NERVE_TEST_VULKAN_DEVICE_INDEX must be an integer");
        let worker_index = std::env::var("NERVE_TEST_VULKAN_SECONDARY_DEVICE_INDEX")
            .expect("NERVE_TEST_VULKAN_SECONDARY_DEVICE_INDEX must select the predicate consumer")
            .parse::<usize>()
            .expect("NERVE_TEST_VULKAN_SECONDARY_DEVICE_INDEX must be an integer");
        assert_ne!(owner_index, worker_index);
        let owner = VulkanComputeDevice::new_for_physical_device_index(owner_index).unwrap();
        let worker = VulkanComputeDevice::new_for_physical_device_index(worker_index).unwrap();
        let shared = owner
            .create_shared_conditional_resident_buffers(&[&worker], 4)
            .unwrap();
        let owner_predicate = &shared.buffers[0];
        let worker_predicate = &shared.buffers[1];
        owner_predicate.write_bytes(&0u32.to_le_bytes()).unwrap();

        let restore = worker
            .create_resident_kernel_dispatch(
                &spirv_words,
                &[VulkanResidentKernelBufferBinding::new(0, worker_predicate, 4)
                    .with_access(VulkanResidentKernelBufferAccess::ReadWrite)],
                1,
                64,
                0,
            )
            .unwrap();
        let output = worker.create_resident_buffer(4).unwrap();
        output.write_bytes(&0u32.to_le_bytes()).unwrap();
        let continuation = worker
            .create_resident_kernel_dispatch(
                &spirv_words,
                &[VulkanResidentKernelBufferBinding::new(0, &output, 4)
                    .with_access(VulkanResidentKernelBufferAccess::ReadWrite)],
                1,
                64,
                0,
            )
            .unwrap();
        let sequence = worker.create_resident_kernel_sequence().unwrap();
        worker
            .record_resident_kernel_sequence(
                &sequence,
                &[
                    VulkanResidentKernelSequenceStep::new(&restore, &[]),
                    VulkanResidentKernelSequenceStep::new_conditional(
                        &continuation,
                        &[],
                        worker_predicate,
                        0,
                        false,
                        1,
                    )
                    .unwrap(),
                ],
            )
            .unwrap();

        worker.run_recorded_resident_kernel_sequence(&sequence).unwrap();

        assert_eq!(owner_predicate.read_bytes(4).unwrap(), 1u32.to_le_bytes());
        assert_eq!(output.read_bytes(4).unwrap(), 1u32.to_le_bytes());
    }

    #[test]
    fn host_restores_gpu_written_shared_predicate_before_conditional_submission() {
        let spirv_words = compile_test_shader_words()
            .expect("Vulkan host predicate restoration test requires a GLSL compiler");
        let owner_index = std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX")
            .expect("NERVE_TEST_VULKAN_DEVICE_INDEX must select the predicate owner")
            .parse::<usize>()
            .expect("NERVE_TEST_VULKAN_DEVICE_INDEX must be an integer");
        let worker_index = std::env::var("NERVE_TEST_VULKAN_SECONDARY_DEVICE_INDEX")
            .expect("NERVE_TEST_VULKAN_SECONDARY_DEVICE_INDEX must select the predicate consumer")
            .parse::<usize>()
            .expect("NERVE_TEST_VULKAN_SECONDARY_DEVICE_INDEX must be an integer");
        assert_ne!(owner_index, worker_index);
        let owner = VulkanComputeDevice::new_for_physical_device_index(owner_index).unwrap();
        let worker = VulkanComputeDevice::new_for_physical_device_index(worker_index).unwrap();
        let shared = owner
            .create_shared_conditional_resident_buffers(&[&worker], 4)
            .unwrap();
        let owner_predicate = &shared.buffers[0];
        let worker_predicate = &shared.buffers[1];

        owner_predicate
            .write_bytes(&u32::MAX.to_le_bytes())
            .unwrap();
        let suppress = worker
            .create_resident_kernel_dispatch(
                &spirv_words,
                &[VulkanResidentKernelBufferBinding::new(0, worker_predicate, 4)
                    .with_access(VulkanResidentKernelBufferAccess::ReadWrite)],
                1,
                64,
                0,
            )
            .unwrap();
        let suppress_sequence = worker.create_resident_kernel_sequence().unwrap();
        worker
            .record_resident_kernel_sequence(
                &suppress_sequence,
                &[VulkanResidentKernelSequenceStep::new(&suppress, &[])],
            )
            .unwrap();
        worker
            .run_recorded_resident_kernel_sequence(&suppress_sequence)
            .unwrap();
        assert_eq!(owner_predicate.read_bytes(4).unwrap(), 0u32.to_le_bytes());

        owner_predicate.write_bytes(&1u32.to_le_bytes()).unwrap();
        let output = worker.create_resident_buffer(4).unwrap();
        output.write_bytes(&0u32.to_le_bytes()).unwrap();
        let continuation = worker
            .create_resident_kernel_dispatch(
                &spirv_words,
                &[VulkanResidentKernelBufferBinding::new(0, &output, 4)
                    .with_access(VulkanResidentKernelBufferAccess::ReadWrite)],
                1,
                64,
                0,
            )
            .unwrap();
        let continuation_sequence = worker.create_resident_kernel_sequence().unwrap();
        worker
            .record_resident_kernel_sequence(
                &continuation_sequence,
                &[VulkanResidentKernelSequenceStep::new_conditional(
                    &continuation,
                    &[],
                    worker_predicate,
                    0,
                    false,
                    1,
                )
                .unwrap()],
            )
            .unwrap();
        worker
            .run_recorded_resident_kernel_sequence(&continuation_sequence)
            .unwrap();

        assert_eq!(owner_predicate.read_bytes(4).unwrap(), 1u32.to_le_bytes());
        assert_eq!(output.read_bytes(4).unwrap(), 1u32.to_le_bytes());
    }

    #[test]
    fn direct_dispatch_after_suppressed_conditional_indirect_region_still_executes() {
        let spirv_words = compile_test_shader_words()
            .expect("Vulkan conditional-region test requires a GLSL compiler");
        let device = selected_test_vulkan_device().unwrap();
        let predicate = device.create_conditional_resident_buffer(4).unwrap();
        predicate.write_bytes(&0u32.to_le_bytes()).unwrap();
        let dimensions = device
            .create_resident_buffer(VULKAN_RESIDENT_INDIRECT_DISPATCH_BYTE_COUNT)
            .unwrap();
        dimensions
            .write_bytes(&u32_bytes(&[1, 1, 1]))
            .unwrap();
        let output = device.create_resident_buffer(4).unwrap();
        output.write_bytes(&0u32.to_le_bytes()).unwrap();
        let dispatch = device
            .create_resident_kernel_dispatch(
                &spirv_words,
                &[VulkanResidentKernelBufferBinding::new(0, &output, 4)
                    .with_access(VulkanResidentKernelBufferAccess::ReadWrite)],
                1,
                64,
                0,
            )
            .unwrap();
        let sequence = device.create_resident_kernel_sequence().unwrap();
        device
            .record_resident_kernel_sequence(
                &sequence,
                &[
                    VulkanResidentKernelSequenceStep::new_indirect(
                        &dispatch,
                        &[],
                        &dimensions,
                        0,
                    )
                    .unwrap()
                    .with_condition(&predicate, 0, false, 1)
                    .unwrap(),
                    VulkanResidentKernelSequenceStep::new(&dispatch, &[]),
                ],
            )
            .unwrap();

        device.run_recorded_resident_kernel_sequence(&sequence).unwrap();

        assert_eq!(output.read_bytes(4).unwrap(), 1u32.to_le_bytes());
    }

    #[test]
    fn resident_kernel_sequence_combines_input_and_intermediate_snapshot_copies() {
        let spirv_words =
            compile_test_shader_words().expect("Vulkan sequence test requires a GLSL compiler");
        let device = selected_test_vulkan_device().unwrap();
        let initial = device.create_resident_buffer(12).unwrap();
        initial.write_bytes(&u32_bytes(&[1, 2, 41])).unwrap();
        let state = device.create_resident_buffer(12).unwrap();
        state.write_bytes(&[0; 12]).unwrap();
        let snapshots = device.create_host_visible_resident_buffer(24).unwrap();
        let input_copy = device
            .create_resident_buffer_copy(&initial, &state, 12)
            .unwrap();
        let binding = VulkanResidentKernelBufferBinding::new(0, &state, 12);
        let dispatch = device
            .create_resident_kernel_dispatch(&spirv_words, &[binding], 1, 64, 0)
            .unwrap();
        let sequence = device.create_resident_kernel_sequence().unwrap();
        let steps = [
            VulkanResidentKernelSequenceStep::new(&dispatch, &[]),
            VulkanResidentKernelSequenceStep::new(&dispatch, &[]),
        ];
        let copies = [
            VulkanResidentKernelSequenceSnapshotCopy::new(0, &state, &snapshots, 0, 0, 12).unwrap(),
            VulkanResidentKernelSequenceSnapshotCopy::new(1, &state, &snapshots, 0, 12, 12)
                .unwrap(),
        ];

        device
            .run_resident_kernel_sequence_with_input_and_snapshot_copies(
                &sequence,
                &[VulkanResidentKernelSequenceInputCopy::new(&input_copy)],
                &steps,
                &copies,
            )
            .unwrap();

        assert_eq!(
            bytes_to_u32(&snapshots.read_bytes(24).unwrap()),
            vec![2, 3, 42, 3, 4, 43]
        );
    }

    #[test]
    fn conditional_sequence_requires_explicit_unconditional_snapshot_safety() {
        let spirv_words =
            compile_test_shader_words().expect("Vulkan sequence test requires a GLSL compiler");
        let device_index = std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX")
            .expect("NERVE_TEST_VULKAN_DEVICE_INDEX must select a compatible AMD GPU with sufficient safe remaining capacity")
            .parse::<usize>()
            .expect("NERVE_TEST_VULKAN_DEVICE_INDEX must be an integer");
        let device = VulkanComputeDevice::new_for_physical_device_index(device_index).unwrap();
        let state = device.create_resident_buffer(12).unwrap();
        state.write_bytes(&u32_bytes(&[1, 2, 41])).unwrap();
        let snapshot = device.create_host_visible_resident_buffer(12).unwrap();
        snapshot.write_bytes(&[0; 12]).unwrap();
        let predicate = device.create_conditional_resident_buffer(4).unwrap();
        predicate.write_bytes(&0u32.to_le_bytes()).unwrap();
        let dispatch = device
            .create_resident_kernel_dispatch(
                &spirv_words,
                &[VulkanResidentKernelBufferBinding::new(0, &state, 12)],
                1,
                64,
                0,
            )
            .unwrap();
        let step = VulkanResidentKernelSequenceStep::new_conditional(
            &dispatch,
            &[],
            &predicate,
            0,
            false,
            1,
        )
        .unwrap();
        let unsafe_copy =
            VulkanResidentKernelSequenceSnapshotCopy::new(0, &state, &snapshot, 0, 0, 12).unwrap();
        let rejected = device
            .record_resident_kernel_sequence_with_snapshot_copies(
                &device.create_resident_kernel_sequence().unwrap(),
                &[step],
                &[unsafe_copy],
            )
            .unwrap_err();
        assert!(rejected
            .to_string()
            .contains("without explicit checkpoint-resume safety"));

        let range =
            VulkanResidentBufferRangeCopy::new(&state, &snapshot, 0, 0, 12).unwrap();
        let safe_copy = VulkanResidentKernelSequenceSnapshotCopy::
            unconditional_from_range_after_conditional_step(0, range);
        let sequence = device.create_resident_kernel_sequence().unwrap();
        device
            .run_resident_kernel_sequence_with_snapshot_copies(
                &sequence,
                &[step],
                &[safe_copy],
            )
            .unwrap();
        assert_eq!(
            bytes_to_u32(&snapshot.read_bytes(12).unwrap()),
            vec![1, 2, 41]
        );

        predicate.write_bytes(&1u32.to_le_bytes()).unwrap();
        device
            .run_recorded_resident_kernel_sequence(&sequence)
            .unwrap();
        assert_eq!(
            bytes_to_u32(&snapshot.read_bytes(12).unwrap()),
            vec![2, 3, 42]
        );
    }

    #[test]
    fn generic_resident_kernel_dispatch_validates_push_constant_size() {
        let Some(spirv_words) = compile_test_shader_words() else {
            eprintln!("skipping Vulkan smoke: no GLSL to SPIR-V compiler found");
            return;
        };
        let device = match selected_test_vulkan_device() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping Vulkan smoke: {error}");
                return;
            }
        };
        let buffer = device.create_resident_buffer(4).unwrap();
        buffer.write_bytes(&u32_bytes(&[10])).unwrap();
        let binding = VulkanResidentKernelBufferBinding::new(0, &buffer, 4);
        let dispatch = device
            .create_resident_kernel_dispatch(&spirv_words, &[binding], 1, 64, 4)
            .unwrap();

        let error = device
            .run_resident_kernel_dispatch(&dispatch, &[])
            .unwrap_err();

        assert_eq!(
            error,
            VulkanError(
                "resident kernel sequence step 0 expects 4 push-constant bytes, got 0".to_string()
            )
        );
    }

    #[test]
    fn resident_byte_buffers_can_copy_on_device() {
        let device = match selected_test_vulkan_device() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping Vulkan smoke: {error}");
                return;
            }
        };
        let source = device.create_resident_buffer(8).unwrap();
        let destination = device.create_resident_buffer(8).unwrap();
        source.write_bytes(&[1, 2, 3, 4, 5, 6]).unwrap();
        destination.write_bytes(&[0, 0, 0, 0, 0, 0]).unwrap();

        device
            .copy_resident_buffer_bytes(&source, &destination, 6)
            .unwrap();

        assert_eq!(destination.read_bytes(6).unwrap(), vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn resident_buffer_batch_io_uses_one_transfer_per_direction() {
        let device = match selected_test_vulkan_device() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping Vulkan smoke: {error}");
                return;
            }
        };
        let first = device.create_resident_buffer(16).unwrap();
        let second = device.create_resident_buffer(16).unwrap();
        let first_bytes = [1, 2, 3, 4, 5, 6, 7, 8];
        let second_bytes = [11, 12, 13, 14, 15, 16, 17, 18];
        let writes = [
            VulkanResidentBufferWriteRange::new(&first, 4, &first_bytes).unwrap(),
            VulkanResidentBufferWriteRange::new(&second, 0, &second_bytes).unwrap(),
        ];
        reset_vulkan_resident_execution_counters();

        assert_eq!(
            device.write_resident_buffer_ranges(&writes).unwrap(),
            first_bytes.len() + second_bytes.len()
        );
        let reads = [
            VulkanResidentBufferReadRange::new(&second, 0, second_bytes.len()).unwrap(),
            VulkanResidentBufferReadRange::new(&first, 4, first_bytes.len()).unwrap(),
        ];
        let readback = device.read_resident_buffer_ranges(&reads).unwrap();

        assert_eq!(readback.range_count(), 2);
        assert_eq!(readback.range_bytes(0).unwrap(), second_bytes);
        assert_eq!(readback.range_bytes(1).unwrap(), first_bytes);
        assert!(readback.range_bytes(2).is_err());
        let counters = vulkan_resident_execution_counters();
        assert_eq!(counters.resident_copy_queue_submits, 2);
        assert_eq!(counters.resident_copy_waits, 2);
    }

    #[test]
    fn resident_buffer_readback_binding_reuses_one_packed_transfer() {
        let device = selected_test_vulkan_device().expect("selected Vulkan test device must open");
        let first = device.create_resident_buffer(16).unwrap();
        let second = device.create_resident_buffer(16).unwrap();
        assert!(device
            .create_resident_buffer_readback_binding(&[])
            .is_err());
        let reads = [
            VulkanResidentBufferReadRange::new(&second, 4, 8).unwrap(),
            VulkanResidentBufferReadRange::new(&first, 0, 4).unwrap(),
        ];
        let binding = device
            .create_resident_buffer_readback_binding(&reads)
            .unwrap();
        assert_eq!(binding.range_count(), 2);
        reset_vulkan_resident_execution_counters();

        first.write_bytes(&[1, 2, 3, 4]).unwrap();
        second
            .write_bytes(&[0, 0, 0, 0, 11, 12, 13, 14, 15, 16, 17, 18])
            .unwrap();
        let first_run = binding.run().unwrap();
        assert_eq!(first_run.range_bytes(0).unwrap(), &[11, 12, 13, 14, 15, 16, 17, 18]);
        assert_eq!(first_run.range_bytes(1).unwrap(), &[1, 2, 3, 4]);

        first.write_bytes(&[21, 22, 23, 24]).unwrap();
        second
            .write_bytes(&[0, 0, 0, 0, 31, 32, 33, 34, 35, 36, 37, 38])
            .unwrap();
        let second_run = binding.run().unwrap();
        assert_eq!(second_run.range_bytes(0).unwrap(), &[31, 32, 33, 34, 35, 36, 37, 38]);
        assert_eq!(second_run.range_bytes(1).unwrap(), &[21, 22, 23, 24]);

        let counters = vulkan_resident_execution_counters();
        assert_eq!(counters.resident_copy_queue_submits, 2);
        assert_eq!(counters.resident_copy_waits, 2);
    }

    #[test]
    fn retained_copy_batches_and_readback_mount_in_one_kernel_sequence() {
        let spirv_words =
            compile_test_shader_words().expect("Vulkan sequence test requires a GLSL compiler");
        let device = selected_test_vulkan_device().unwrap();
        let producer = device.create_resident_buffer(12).unwrap();
        let consumer = device.create_resident_buffer(12).unwrap();
        producer.write_bytes(&u32_bytes(&[1, 2, 41])).unwrap();
        consumer.write_bytes(&[0; 12]).unwrap();
        let transfer = device
            .create_resident_buffer_copy_batch(&[
                VulkanResidentBufferRangeCopy::new(&producer, &consumer, 0, 0, 12).unwrap(),
            ])
            .unwrap();
        let readback = device
            .create_resident_buffer_readback_binding(&[
                VulkanResidentBufferReadRange::new(&consumer, 0, 8).unwrap(),
                VulkanResidentBufferReadRange::new(&consumer, 8, 4).unwrap(),
            ])
            .unwrap();
        let producer_dispatch = device
            .create_resident_kernel_dispatch(
                &spirv_words,
                &[VulkanResidentKernelBufferBinding::new(0, &producer, 12)],
                1,
                64,
                0,
            )
            .unwrap();
        let consumer_dispatch = device
            .create_resident_kernel_dispatch(
                &spirv_words,
                &[VulkanResidentKernelBufferBinding::new(0, &consumer, 12)],
                1,
                64,
                0,
            )
            .unwrap();
        let steps = [
            VulkanResidentKernelSequenceStep::new(&producer_dispatch, &[]),
            VulkanResidentKernelSequenceStep::new(&consumer_dispatch, &[]),
        ];
        let mut copies = transfer
            .sequence_snapshot_copies_after_step(0)
            .unwrap();
        copies.extend(
            readback
                .sequence_snapshot_copies_after_step(1)
                .unwrap(),
        );
        let sequence = device.create_resident_kernel_sequence().unwrap();
        reset_vulkan_resident_execution_counters();

        device
            .run_resident_kernel_sequence_with_snapshot_copies(&sequence, &steps, &copies)
            .unwrap();
        let completed = readback.read_completed().unwrap();

        assert_eq!(completed.range_bytes(0).unwrap(), u32_bytes(&[3, 4]));
        assert_eq!(completed.range_bytes(1).unwrap(), 43u32.to_le_bytes());
        let counters = vulkan_resident_execution_counters();
        assert_eq!(counters.resident_sequence_queue_submits, 1);
        assert_eq!(counters.resident_sequence_completion_waits, 1);
        assert_eq!(counters.resident_copy_queue_submits, 0);
        assert_eq!(counters.resident_copy_waits, 0);
    }

    #[test]
    fn resident_transfer_stream_bounds_staging_and_completes_with_a_timeline() {
        let device = match selected_test_vulkan_device() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping Vulkan smoke: {error}");
                return;
            }
        };
        const ROUTER_BYTES: usize = 256;
        const SCALE_BYTES: usize = 128;
        const UP_BYTES: usize = 2 * 1024 * 1024;
        const DOWN_BYTES: usize = 1024 * 1024;
        const GROUP_BYTES: usize = ROUTER_BYTES + SCALE_BYTES + UP_BYTES + DOWN_BYTES;
        let router = device.create_resident_buffer(ROUTER_BYTES).unwrap();
        let scale = device.create_resident_buffer(SCALE_BYTES).unwrap();
        let up = device.create_resident_buffer(UP_BYTES).unwrap();
        let down = device.create_resident_buffer(DOWN_BYTES).unwrap();
        let first_router = vec![1u8; ROUTER_BYTES];
        let first_scale = vec![2u8; SCALE_BYTES];
        let first_up = vec![3u8; UP_BYTES];
        let first_down = vec![4u8; DOWN_BYTES];
        let second_router = vec![5u8; ROUTER_BYTES];
        let second_scale = vec![6u8; SCALE_BYTES];
        let second_up = vec![7u8; UP_BYTES];
        let second_down = vec![8u8; DOWN_BYTES];
        let first = [
            VulkanResidentBufferWriteRange::new(&router, 0, &first_router).unwrap(),
            VulkanResidentBufferWriteRange::new(&scale, 0, &first_scale).unwrap(),
            VulkanResidentBufferWriteRange::new(&up, 0, &first_up).unwrap(),
            VulkanResidentBufferWriteRange::new(&down, 0, &first_down).unwrap(),
        ];
        let second = [
            VulkanResidentBufferWriteRange::new(&router, 0, &second_router).unwrap(),
            VulkanResidentBufferWriteRange::new(&scale, 0, &second_scale).unwrap(),
            VulkanResidentBufferWriteRange::new(&up, 0, &second_up).unwrap(),
            VulkanResidentBufferWriteRange::new(&down, 0, &second_down).unwrap(),
        ];
        let mut undersized = device
            .create_resident_transfer_stream(1, GROUP_BYTES - 4)
            .unwrap();
        assert!(
            undersized
                .submit(&first)
                .unwrap_err()
                .to_string()
                .contains("bounded slot capacity")
        );
        drop(undersized);
        let mut stream = device
            .create_resident_transfer_stream(2, GROUP_BYTES)
            .unwrap();
        reset_vulkan_resident_execution_counters();

        let first_ticket = stream.submit(&first).unwrap();
        let second_ticket = stream.submit(&second).unwrap();
        let third_ticket = stream.submit(&first).unwrap();

        assert_eq!(first_ticket.uploaded_bytes(), GROUP_BYTES);
        assert_eq!(first_ticket.copy_count(), 4);
        assert_eq!(second_ticket.timeline_value(), 2);
        assert_eq!(third_ticket.timeline_value(), 3);
        assert!(stream.outstanding_transfer_count().unwrap() <= 2);
        assert_eq!(
            stream
                .completion_point(&third_ticket)
                .unwrap()
                .value,
            third_ticket.timeline_value()
        );
        stream.wait(&third_ticket).unwrap();
        assert!(stream.is_complete(&third_ticket).unwrap());
        assert_eq!(router.read_bytes(ROUTER_BYTES).unwrap(), first_router);
        assert_eq!(scale.read_bytes(SCALE_BYTES).unwrap(), first_scale);
        assert_eq!(up.read_bytes(UP_BYTES).unwrap(), first_up);
        assert_eq!(down.read_bytes(DOWN_BYTES).unwrap(), first_down);
        let counters = vulkan_resident_execution_counters();
        assert_eq!(counters.resident_copy_queue_submits, 3);
        assert!(counters.resident_copy_waits >= 2);
    }

    #[test]
    fn resident_byte_copy_binding_can_be_reused() {
        let device = match selected_test_vulkan_device() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping Vulkan smoke: {error}");
                return;
            }
        };
        let source = device.create_resident_buffer(8).unwrap();
        let destination = device.create_resident_buffer(8).unwrap();
        let binding = device
            .create_resident_buffer_copy(&source, &destination, 6)
            .unwrap();

        source.write_bytes(&[1, 2, 3, 4, 5, 6]).unwrap();
        device.run_resident_buffer_copy(&binding, 6).unwrap();
        assert_eq!(destination.read_bytes(6).unwrap(), vec![1, 2, 3, 4, 5, 6]);

        source.write_bytes(&[10, 20, 30, 40, 50, 60]).unwrap();
        device.run_resident_buffer_copy(&binding, 6).unwrap();
        assert_eq!(
            destination.read_bytes(6).unwrap(),
            vec![10, 20, 30, 40, 50, 60]
        );
        assert_eq!(binding.byte_len(), 6);
    }

    #[test]
    fn resident_copy_completion_observes_an_unfenced_shader_producer() {
        let spirv_words =
            compile_test_shader_words().expect("Vulkan copy test requires a GLSL compiler");
        let device = selected_test_vulkan_device().unwrap();
        let source = device.create_resident_buffer(12).unwrap();
        let destination = device.create_resident_buffer(12).unwrap();
        source.write_bytes(&u32_bytes(&[1, 2, 41])).unwrap();
        destination.write_bytes(&[0; 12]).unwrap();
        let dispatch = device
            .create_resident_kernel_dispatch(
                &spirv_words,
                &[VulkanResidentKernelBufferBinding::new(0, &source, 12)],
                1,
                64,
                0,
            )
            .unwrap();
        let sequence = device.create_resident_kernel_sequence().unwrap();
        device
            .record_resident_kernel_sequence(
                &sequence,
                &[VulkanResidentKernelSequenceStep::new(&dispatch, &[])],
            )
            .unwrap();
        let copy = device
            .create_resident_buffer_copy(&source, &destination, 12)
            .unwrap();
        let completion = device.create_timeline_semaphore(0).unwrap();

        device
            .submit_recorded_resident_kernel_sequence_unfenced_with_timeline_semaphores(
                &sequence,
                &[],
                &[],
            )
            .unwrap();
        device
            .submit_resident_buffer_copy_with_timeline_semaphores(
                &copy,
                &[],
                &[VulkanTimelineSemaphorePoint::new(&completion, 1)],
            )
            .unwrap();
        device.wait_timeline_semaphore_value(&completion, 1).unwrap();

        assert_eq!(
            bytes_to_u32(&destination.read_bytes(12).unwrap()),
            vec![2, 3, 42]
        );
    }

    #[test]
    fn resident_copy_replay_can_overlap_without_host_completion() {
        let device = selected_test_vulkan_device().unwrap();
        let source = device.create_resident_buffer(12).unwrap();
        let destination = device.create_resident_buffer(12).unwrap();
        source.write_bytes(&u32_bytes(&[7, 11, 13])).unwrap();
        destination.write_bytes(&[0; 12]).unwrap();
        let copy = device
            .create_resident_buffer_copy(&source, &destination, 12)
            .unwrap();
        let completion = device.create_timeline_semaphore(0).unwrap();

        for value in [1, 2] {
            device
                .submit_resident_buffer_copy_with_timeline_semaphores(
                    &copy,
                    &[],
                    &[VulkanTimelineSemaphorePoint::new(&completion, value)],
                )
                .unwrap();
        }
        device.wait_timeline_semaphore_value(&completion, 2).unwrap();

        assert_eq!(
            bytes_to_u32(&destination.read_bytes(12).unwrap()),
            vec![7, 11, 13]
        );
        assert!(copy.completion.pending_value().is_none());
    }
}
