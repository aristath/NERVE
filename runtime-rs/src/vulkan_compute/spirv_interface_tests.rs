#[cfg(test)]
mod spirv_interface_contract_tests {
    use super::*;

    fn descriptor_module(descriptors: &[(u32, u32, u32)]) -> Vec<u32> {
        let mut words = vec![SPIRV_MAGIC, 0x0001_0600, 0, 32, 0];
        for (target, set, binding) in descriptors {
            words.extend([
                (4u32 << 16) | u32::from(SPIRV_OP_DECORATE),
                *target,
                SPIRV_DECORATION_DESCRIPTOR_SET,
                *set,
            ]);
            words.extend([
                (4u32 << 16) | u32::from(SPIRV_OP_DECORATE),
                *target,
                SPIRV_DECORATION_BINDING,
                *binding,
            ]);
        }
        words
    }

    #[test]
    fn spirv_descriptor_interface_requires_every_declared_binding() {
        let words = descriptor_module(&[(9, 0, 31), (4, 0, 2), (7, 0, 5)]);

        validate_spirv_storage_descriptor_bindings(&words, &[2, 5, 31]).unwrap();

        assert_eq!(
            validate_spirv_storage_descriptor_bindings(&words, &[2, 31]).unwrap_err(),
            VulkanError(
                "shader descriptor interface is not covered by the generic storage pipeline: missing bindings [5]"
                    .to_string()
            )
        );
        validate_spirv_storage_descriptor_bindings(&words, &[2, 5, 17, 31]).unwrap();
    }

    #[test]
    fn spirv_descriptor_interface_rejects_unsupported_sets_and_incomplete_pairs() {
        let unsupported_set = descriptor_module(&[(4, 1, 2)]);
        assert!(
            validate_spirv_storage_descriptor_bindings(&unsupported_set, &[])
                .unwrap_err()
                .0
                .contains("other than set 0")
        );

        let mut incomplete = vec![SPIRV_MAGIC, 0x0001_0600, 0, 32, 0];
        incomplete.extend([
            (4u32 << 16) | u32::from(SPIRV_OP_DECORATE),
            4,
            SPIRV_DECORATION_BINDING,
            2,
        ]);
        assert!(
            vulkan_spirv_descriptor_interface(&incomplete)
                .unwrap_err()
                .0
                .contains("incomplete descriptor set/binding")
        );
    }

    #[test]
    fn spirv_descriptor_interface_rejects_malformed_instructions() {
        let mut malformed = descriptor_module(&[(4, 0, 2)]);
        malformed.push((4u32 << 16) | u32::from(SPIRV_OP_DECORATE));

        assert!(vulkan_spirv_descriptor_interface(&malformed).is_err());
    }
}
