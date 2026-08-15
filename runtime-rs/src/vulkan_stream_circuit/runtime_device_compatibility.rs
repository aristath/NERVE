struct VulkanRuntimeDeviceCapabilities {
    shader_features: BTreeSet<String>,
    subgroup_operations: BTreeSet<String>,
    extensions: BTreeSet<String>,
    subgroup_compute_supported: bool,
    subgroup_size: Option<u32>,
    max_workgroup_invocations: Option<u32>,
    max_workgroup_size_x: Option<u32>,
    cooperative_bfloat16_shapes: BTreeSet<[u32; 3]>,
    cooperative_float8_e4m3_shapes: BTreeSet<[u32; 3]>,
}

impl VulkanRuntimeDeviceCapabilities {
    fn from_profile(profile: &crate::HardwareProcessProfile) -> Self {
        let compiler = profile
            .capability_extensions
            .get("vulkan_compiler_capabilities")
            .unwrap_or(&Value::Null);
        let vulkan_device = profile
            .capability_extensions
            .get("vulkan_device")
            .unwrap_or(&Value::Null);
        Self {
            shader_features: vulkan_runtime_json_string_set(compiler.get("shader_features")),
            subgroup_operations: vulkan_runtime_json_string_set(
                compiler.get("subgroup_operations"),
            ),
            extensions: vulkan_runtime_json_string_set(vulkan_device.get("extensions")),
            subgroup_compute_supported: compiler
                .get("subgroup_compute_supported")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            subgroup_size: compiler
                .get("subgroup_size")
                .and_then(Value::as_u64)
                .and_then(|value| u32::try_from(value).ok()),
            max_workgroup_invocations: compiler
                .get("max_compute_work_group_invocations")
                .and_then(Value::as_u64)
                .and_then(|value| u32::try_from(value).ok()),
            max_workgroup_size_x: compiler
                .get("max_compute_work_group_size_x")
                .and_then(Value::as_u64)
                .and_then(|value| u32::try_from(value).ok()),
            cooperative_bfloat16_shapes: vulkan_runtime_json_shape_set(
                compiler.get("cooperative_bfloat16_shapes"),
            ),
            cooperative_float8_e4m3_shapes: vulkan_runtime_json_shape_set(
                compiler.get("cooperative_float8_e4m3_shapes"),
            ),
        }
    }

    fn validate_local_size(&self, local_size_x: u32) -> Result<(), String> {
        if self
            .max_workgroup_invocations
            .is_some_and(|maximum| local_size_x > maximum)
            || self
                .max_workgroup_size_x
                .is_some_and(|maximum| local_size_x > maximum)
        {
            return Err(format!(
                "local size {local_size_x} exceeds the device compute-workgroup limit"
            ));
        }
        Ok(())
    }

    fn validate_shader(
        &self,
        package_root: &Path,
        shader_path: &str,
        local_size_x: u32,
    ) -> Result<(), String> {
        self.validate_local_size(local_size_x)?;
        let words = crate::read_spirv_words(package_root.join(shader_path))
            .map_err(|error| format!("could not inspect shader {shader_path:?}: {error}"))?;
        let requirements = crate::vulkan_spirv_requirements(&words)
            .map_err(|error| format!("could not inspect shader {shader_path:?}: {error}"))?;
        let missing_features = requirements
            .shader_features
            .iter()
            .map(|feature| feature.label())
            .filter(|feature| !self.shader_features.contains(*feature))
            .collect::<Vec<_>>();
        if !missing_features.is_empty() {
            return Err(format!(
                "shader {shader_path:?} requires unsupported Vulkan features {}",
                missing_features.join(", ")
            ));
        }
        let missing_subgroups = requirements
            .subgroup_operations
            .iter()
            .map(|operation| operation.label())
            .filter(|operation| !self.subgroup_operations.contains(*operation))
            .collect::<Vec<_>>();
        if !requirements.subgroup_operations.is_empty() && !self.subgroup_compute_supported {
            return Err(format!(
                "shader {shader_path:?} requires compute-stage subgroup support"
            ));
        }
        if !missing_subgroups.is_empty() {
            return Err(format!(
                "shader {shader_path:?} requires unsupported subgroup operations {}",
                missing_subgroups.join(", ")
            ));
        }
        Ok(())
    }

    fn validate_batch_requirements(
        &self,
        requirements: &VulkanResidentVulkanDeviceRequirements,
    ) -> Result<(), String> {
        let missing_extensions = requirements
            .vulkan_device_extensions
            .iter()
            .filter(|extension| !self.extensions.contains(extension.as_str()))
            .map(String::as_str)
            .collect::<Vec<_>>();
        if !missing_extensions.is_empty() {
            return Err(format!(
                "requires unavailable Vulkan extensions {}",
                missing_extensions.join(", ")
            ));
        }
        let missing_features = requirements
            .vulkan_features
            .iter()
            .map(|feature| feature.label())
            .filter(|feature| !self.shader_features.contains(*feature))
            .collect::<Vec<_>>();
        if !missing_features.is_empty() {
            return Err(format!(
                "requires unsupported Vulkan features {}",
                missing_features.join(", ")
            ));
        }
        let missing_subgroups = requirements
            .subgroup_operations
            .iter()
            .map(|operation| operation.label())
            .filter(|operation| !self.subgroup_operations.contains(*operation))
            .collect::<Vec<_>>();
        if !missing_subgroups.is_empty() {
            return Err(format!(
                "requires unsupported subgroup operations {}",
                missing_subgroups.join(", ")
            ));
        }
        if !requirements.subgroup_operations.is_empty() && !self.subgroup_compute_supported {
            return Err("requires compute-stage subgroup support".to_string());
        }
        if requirements
            .subgroup_size
            .is_some_and(|required| self.subgroup_size != Some(required))
        {
            return Err(format!(
                "requires subgroup size {}, device reports {:?}",
                requirements.subgroup_size.unwrap_or_default(),
                self.subgroup_size
            ));
        }
        if requirements
            .cooperative_bfloat16_shape
            .is_some_and(|shape| !self.cooperative_bfloat16_shapes.contains(&shape))
        {
            return Err(format!(
                "requires unsupported cooperative BF16 shape {:?}",
                requirements.cooperative_bfloat16_shape.unwrap_or_default()
            ));
        }
        if requirements
            .cooperative_float8_e4m3_shape
            .is_some_and(|shape| !self.cooperative_float8_e4m3_shapes.contains(&shape))
        {
            return Err(format!(
                "requires unsupported cooperative FP8 E4M3 shape {:?}",
                requirements
                    .cooperative_float8_e4m3_shape
                    .unwrap_or_default()
            ));
        }
        Ok(())
    }
}

fn validate_vulkan_runtime_component_execution_hardware_compatibility(
    package_root: &Path,
    execution: &VulkanResidentComponentExecutionSpec,
    capabilities: &VulkanRuntimeDeviceCapabilities,
) -> Result<(), String> {
    for kernel in &execution.kernels {
        capabilities.validate_shader(package_root, &kernel.shader_path, kernel.local_size_x)?;
        if !kernel.batch_implementations.is_empty()
            && !kernel.batch_implementations.iter().any(|implementation| {
                capabilities
                    .validate_batch_requirements(&implementation.device_requirements)
                    .and_then(|_| {
                        implementation.stages.iter().try_for_each(|stage| {
                            capabilities.validate_shader(
                                package_root,
                                &stage.shader_path,
                                stage.local_size_x,
                            )
                        })
                    })
                    .is_ok()
            })
        {
            return Err(format!(
                "kernel {:?} has no compatible prefill implementation",
                kernel.node_id
            ));
        }
    }
    Ok(())
}

fn validate_vulkan_runtime_output_transducer_hardware_compatibility(
    package_root: &Path,
    output: &VulkanResidentOutputTransducerPackageSpec,
    capabilities: &VulkanRuntimeDeviceCapabilities,
) -> Result<(), String> {
    for (path, local_size_x) in [
        (&output.embedding_norm_shader_path, output.spec.norm_local_size_x),
        (
            &output.embedding_norm_batch_shader_path,
            output.spec.norm_local_size_x,
        ),
        (&output.projection_shader_path, output.spec.projection_local_size_x),
        (
            &output.projection_batch_shader_path,
            output.spec.projection_local_size_x,
        ),
    ] {
        capabilities.validate_shader(package_root, path, local_size_x)?;
    }
    Ok(())
}

pub fn validate_vulkan_package_source_component_hardware_compatibility(
    package_root: &Path,
    manifest: &VulkanResidentModelPackageManifest,
    source_component_id: &str,
    profile: &crate::HardwareProcessProfile,
) -> Result<(), String> {
    let source = manifest
        .circuit_graph
        .components
        .iter()
        .find(|source| source.component_id == source_component_id)
        .ok_or_else(|| format!("unknown source component {source_component_id:?}"))?;
    let capabilities = VulkanRuntimeDeviceCapabilities::from_profile(profile);
    match source.runtime_role {
        CircuitRuntimeRole::SignalProcessor => {
            let execution = manifest
                .component_executions
                .iter()
                .find(|execution| execution.component_id == source_component_id)
                .ok_or_else(|| {
                    format!("source component {source_component_id:?} has no execution contract")
                })?;
            validate_vulkan_runtime_component_execution_hardware_compatibility(
                package_root,
                execution,
                &capabilities,
            )?;
        }
        CircuitRuntimeRole::InputTransducer => {
            let spec = &manifest.input_transducer;
            capabilities.validate_shader(
                package_root,
                &spec.shader_path,
                spec.spec.local_size_x,
            )?;
            capabilities.validate_shader(
                package_root,
                &spec.batch_shader_path,
                spec.spec.local_size_x,
            )?;
        }
        CircuitRuntimeRole::OutputTransducer => {
            validate_vulkan_runtime_output_transducer_hardware_compatibility(
                package_root,
                &manifest.output_transducer,
                &capabilities,
            )?;
        }
        CircuitRuntimeRole::Sampler => {
            for kernel in &manifest.sampler.kernels {
                capabilities.validate_shader(
                    package_root,
                    &kernel.shader_path,
                    kernel.local_size_x,
                )?;
            }
        }
        CircuitRuntimeRole::DraftProcessor => {
            let execution = manifest
                .speculative_decoders
                .iter()
                .flat_map(|decoder| &decoder.component_executions)
                .find(|execution| execution.component_id == source_component_id)
                .ok_or_else(|| {
                    format!("draft component {source_component_id:?} has no execution contract")
                })?;
            validate_vulkan_runtime_component_execution_hardware_compatibility(
                package_root,
                execution,
                &capabilities,
            )?;
        }
        CircuitRuntimeRole::DraftOutputTransducer => {
            let output = manifest
                .speculative_decoders
                .iter()
                .filter_map(|decoder| decoder.dedicated_output_transducer())
                .find(|output| output.component_id == source_component_id);
            if let Some(output) = output {
                capabilities.validate_shader(
                    package_root,
                    &output.norm_shader_path,
                    output.norm_local_size_x,
                )?;
                capabilities.validate_shader(
                    package_root,
                    &output.projection_shader_path,
                    output.projection_local_size_x,
                )?;
            } else {
                let execution = manifest
                    .speculative_decoders
                    .iter()
                    .flat_map(|decoder| &decoder.component_executions)
                    .find(|execution| execution.component_id == source_component_id)
                    .ok_or_else(|| {
                        format!("draft output {source_component_id:?} has no execution contract")
                    })?;
                validate_vulkan_runtime_component_execution_hardware_compatibility(
                    package_root,
                    execution,
                    &capabilities,
                )?;
            }
        }
        CircuitRuntimeRole::DraftInputAdapter => {}
    }
    Ok(())
}

/// Computes exact per-instance baseline incompatibility for the concrete
/// mounted placement. Alternative representation selection may cover these
/// instances; compatible neighbors remain free to retain their exact compiled
/// implementation.
pub fn vulkan_runtime_exact_baseline_incompatible_instance_ids(
    runtime_model: &VulkanResidentRuntimeModel,
    package_root: impl AsRef<Path>,
    profiles_by_logical_device: &BTreeMap<String, crate::HardwareProcessProfile>,
) -> Result<BTreeSet<String>, VulkanResidentTokenModelPackageError> {
    let package_root = package_root.as_ref();
    let source_roles = runtime_model
        .package
        .circuit_graph
        .components
        .iter()
        .map(|component| (component.component_id.as_str(), component.runtime_role))
        .collect::<BTreeMap<_, _>>();
    let mut incompatible = BTreeSet::new();
    for instance in runtime_model.runtime_graph.instances.iter().filter(|instance| {
        source_roles
            .get(instance.source_component_id.as_str())
            .is_some_and(|role| role.is_runtime_implementation_target())
    }) {
        let role = source_roles[instance.source_component_id.as_str()];
        let mut logical_device_ids = vec![
            runtime_model
                .placement
                .device_for_component(&instance.instance_id)
                .to_string(),
        ];
        if let Some(shards) = runtime_model
            .placement
            .component_shard_devices
            .get(&instance.instance_id)
        {
            logical_device_ids.extend(shards.iter().cloned());
        }
        logical_device_ids.sort();
        logical_device_ids.dedup();
        let mut compatible = true;
        for logical_device_id in logical_device_ids {
            let profile = profiles_by_logical_device.get(&logical_device_id).ok_or_else(|| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "runtime component {:?} uses logical device {:?} without a hardware profile",
                    instance.instance_id, logical_device_id,
                ))
            })?;
            let capabilities = VulkanRuntimeDeviceCapabilities::from_profile(profile);
            let validation = match role {
                CircuitRuntimeRole::SignalProcessor => {
                    let execution = runtime_model
                        .component_executions
                        .iter()
                        .find(|execution| execution.component_id == instance.instance_id)
                        .ok_or_else(|| {
                            VulkanResidentTokenModelPackageError::new(format!(
                                "runtime component {:?} has no execution contract",
                                instance.instance_id,
                            ))
                        })?;
                    validate_vulkan_runtime_component_execution_hardware_compatibility(
                        package_root,
                        execution,
                        &capabilities,
                    )
                }
                CircuitRuntimeRole::OutputTransducer => {
                    validate_vulkan_runtime_output_transducer_hardware_compatibility(
                        package_root,
                        &runtime_model.package.output_transducer,
                        &capabilities,
                    )
                }
                _ => unreachable!("only runtime implementation targets are visited"),
            };
            if validation.is_err() {
                compatible = false;
                break;
            }
        }
        if !compatible {
            incompatible.insert(instance.instance_id.clone());
        }
    }
    Ok(incompatible)
}

fn vulkan_runtime_json_string_set(value: Option<&Value>) -> BTreeSet<String> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(ToOwned::to_owned)
        .collect()
}

fn vulkan_runtime_json_shape_set(value: Option<&Value>) -> BTreeSet<[u32; 3]> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|shape| {
            let values = shape.as_array()?;
            if values.len() != 3 {
                return None;
            }
            Some([
                u32::try_from(values.first()?.as_u64()?).ok()?,
                u32::try_from(values.get(1)?.as_u64()?).ok()?,
                u32::try_from(values.get(2)?.as_u64()?).ok()?,
            ])
        })
        .collect()
}

#[cfg(test)]
mod runtime_device_compatibility_tests {
    use super::*;

    #[test]
    fn capability_shape_parser_accepts_only_exact_unsigned_triplets() {
        let value = serde_json::json!([
            [16, 16, 32],
            [8, 8],
            [1, 2, 3, 4],
            [-1, 2, 3],
            ["16", 16, 16]
        ]);
        assert_eq!(
            vulkan_runtime_json_shape_set(Some(&value)),
            BTreeSet::from([[16, 16, 32]])
        );
    }

    fn capability_view() -> VulkanRuntimeDeviceCapabilities {
        VulkanRuntimeDeviceCapabilities {
            shader_features: BTreeSet::from([
                crate::VulkanShaderFeature::ShaderFloat16.label().to_string(),
            ]),
            subgroup_operations: BTreeSet::from([
                crate::VulkanSubgroupOperation::Basic.label().to_string(),
            ]),
            extensions: BTreeSet::from(["VK_EXT_fixture".to_string()]),
            subgroup_compute_supported: true,
            subgroup_size: Some(32),
            max_workgroup_invocations: Some(256),
            max_workgroup_size_x: Some(128),
            cooperative_bfloat16_shapes: BTreeSet::from([[16, 16, 32]]),
            cooperative_float8_e4m3_shapes: BTreeSet::from([[16, 16, 64]]),
        }
    }

    #[test]
    fn device_capability_view_enforces_workgroup_limits_at_the_boundary() {
        let capabilities = capability_view();
        assert!(capabilities.validate_local_size(128).is_ok());
        assert!(
            capabilities
                .validate_local_size(129)
                .unwrap_err()
                .contains("workgroup limit")
        );
    }

    #[test]
    fn batch_requirements_require_every_declared_device_capability() {
        let capabilities = capability_view();
        let requirements = VulkanResidentVulkanDeviceRequirements {
            vulkan_device_extensions: vec!["VK_EXT_fixture".to_string()],
            vulkan_features: vec![crate::VulkanShaderFeature::ShaderFloat16],
            subgroup_operations: vec![crate::VulkanSubgroupOperation::Basic],
            cooperative_bfloat16_shape: Some([16, 16, 32]),
            cooperative_float8_e4m3_shape: Some([16, 16, 64]),
            subgroup_size: Some(32),
        };
        assert!(
            capabilities
                .validate_batch_requirements(&requirements)
                .is_ok()
        );

        let mut missing_extension = requirements.clone();
        missing_extension.vulkan_device_extensions = vec!["VK_EXT_missing".to_string()];
        assert!(
            capabilities
                .validate_batch_requirements(&missing_extension)
                .unwrap_err()
                .contains("VK_EXT_missing")
        );
        let mut wrong_subgroup_size = requirements.clone();
        wrong_subgroup_size.subgroup_size = Some(64);
        assert!(
            capabilities
                .validate_batch_requirements(&wrong_subgroup_size)
                .unwrap_err()
                .contains("subgroup size 64")
        );
        let mut unsupported_shape = requirements;
        unsupported_shape.cooperative_float8_e4m3_shape = Some([8, 8, 16]);
        assert!(
            capabilities
                .validate_batch_requirements(&unsupported_shape)
                .unwrap_err()
                .contains("FP8")
        );
    }
}
