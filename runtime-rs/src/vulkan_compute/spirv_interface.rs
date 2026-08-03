const SPIRV_OP_DECORATE: u16 = 71;
const SPIRV_DECORATION_BINDING: u32 = 33;
const SPIRV_DECORATION_DESCRIPTOR_SET: u32 = 34;

#[derive(Default)]
struct VulkanSpirvDescriptorDecorations {
    descriptor_set: Option<u32>,
    binding: Option<u32>,
}

fn vulkan_spirv_descriptor_interface(
    spirv_words: &[u32],
) -> Result<BTreeMap<u32, BTreeSet<u32>>, VulkanError> {
    if spirv_words.len() < 5 || spirv_words[0] != SPIRV_MAGIC {
        return Err(VulkanError(
            "shader artifact is not a valid little-endian SPIR-V module".to_string(),
        ));
    }

    let mut decorations = BTreeMap::<u32, VulkanSpirvDescriptorDecorations>::new();
    let mut cursor = 5usize;
    while cursor < spirv_words.len() {
        let instruction = spirv_words[cursor];
        let word_count = (instruction >> 16) as usize;
        let opcode = (instruction & 0xffff) as u16;
        if word_count == 0 || cursor + word_count > spirv_words.len() {
            return Err(VulkanError(format!(
                "shader artifact has a malformed SPIR-V instruction at word {cursor}"
            )));
        }
        if opcode == SPIRV_OP_DECORATE {
            if word_count < 3 {
                return Err(VulkanError(format!(
                    "shader artifact has a malformed OpDecorate at word {cursor}"
                )));
            }
            let target = spirv_words[cursor + 1];
            let decoration = spirv_words[cursor + 2];
            if matches!(
                decoration,
                SPIRV_DECORATION_BINDING | SPIRV_DECORATION_DESCRIPTOR_SET
            ) {
                if word_count != 4 {
                    return Err(VulkanError(format!(
                        "shader artifact has a malformed descriptor OpDecorate at word {cursor}"
                    )));
                }
                let value = spirv_words[cursor + 3];
                let target = decorations.entry(target).or_default();
                let field = if decoration == SPIRV_DECORATION_BINDING {
                    &mut target.binding
                } else {
                    &mut target.descriptor_set
                };
                if field.replace(value).is_some() {
                    return Err(VulkanError(format!(
                        "shader artifact decorates SPIR-V id {} with descriptor {} more than once",
                        spirv_words[cursor + 1],
                        if decoration == SPIRV_DECORATION_BINDING {
                            "binding"
                        } else {
                            "set"
                        }
                    )));
                }
            }
        }
        cursor += word_count;
    }

    let mut interface = BTreeMap::<u32, BTreeSet<u32>>::new();
    for (target, decoration) in decorations {
        let (Some(descriptor_set), Some(binding)) =
            (decoration.descriptor_set, decoration.binding)
        else {
            return Err(VulkanError(format!(
                "shader artifact gives SPIR-V id {target} an incomplete descriptor set/binding contract"
            )));
        };
        if !interface.entry(descriptor_set).or_default().insert(binding) {
            return Err(VulkanError(format!(
                "shader artifact declares descriptor set {descriptor_set} binding {binding} more than once"
            )));
        }
    }
    Ok(interface)
}

fn validate_spirv_storage_descriptor_bindings(
    spirv_words: &[u32],
    provided_bindings: &[u32],
) -> Result<(), VulkanError> {
    let mut interface = vulkan_spirv_descriptor_interface(spirv_words)?;
    let declared_bindings = interface.remove(&0).unwrap_or_default();
    if !interface.is_empty() {
        return Err(VulkanError(format!(
            "generic storage pipeline cannot bind shader descriptor sets other than set 0: {:?}",
            interface.keys().collect::<Vec<_>>()
        )));
    }
    let provided_bindings = provided_bindings.iter().copied().collect::<BTreeSet<_>>();
    let missing = declared_bindings
        .difference(&provided_bindings)
        .copied()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(VulkanError(format!(
            "shader descriptor interface is not covered by the generic storage pipeline: missing bindings {missing:?}"
        )));
    }
    Ok(())
}
