struct DeviceCapabilityView {
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

impl DeviceCapabilityView {
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
            shader_features: json_string_set(compiler.get("shader_features")),
            subgroup_operations: json_string_set(compiler.get("subgroup_operations")),
            extensions: json_string_set(vulkan_device.get("extensions")),
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
            cooperative_bfloat16_shapes: json_shape_set(
                compiler.get("cooperative_bfloat16_shapes"),
            ),
            cooperative_float8_e4m3_shapes: json_shape_set(
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
        requirements: &crate::VulkanResidentVulkanDeviceRequirements,
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

impl RuntimeModelEditor {
    pub fn validate_instance_device_compatibility(
        &self,
        instance_id: &str,
        device_id: &str,
    ) -> Result<(), RuntimeEditorError> {
        let instance = self
            .draft
            .instances
            .iter()
            .find(|instance| instance.instance_id == instance_id)
            .ok_or_else(|| {
                RuntimeEditorError(format!(
                    "runtime graph has no node instance {instance_id:?}"
                ))
            })?;
        let device = self
            .available_devices
            .iter()
            .find(|device| device.device_id == device_id)
            .ok_or_else(|| RuntimeEditorError(format!("runtime device {device_id:?} is unknown")))?;
        if !device.available
            || device.can_host_runtime_components_on_physical_device == Some(false)
        {
            return Err(RuntimeEditorError(format!(
                "runtime device {device_id:?} is unavailable or cannot host runtime components"
            )));
        }
        let Some(profile) = &device.hardware_profile else {
            return Ok(());
        };
        let capabilities = DeviceCapabilityView::from_profile(profile);
        self.validate_source_shaders(
            &instance.source_component_id,
            &capabilities,
        )
        .map_err(|error| {
            RuntimeEditorError(format!(
                "runtime device {device_id:?} cannot host instance {instance_id:?}: {error}"
            ))
        })
    }

    fn validate_source_shaders(
        &self,
        source_component_id: &str,
        capabilities: &DeviceCapabilityView,
    ) -> Result<(), String> {
        let source = self
            .source_components
            .iter()
            .find(|source| source.source_id == source_component_id)
            .ok_or_else(|| format!("unknown source component {source_component_id:?}"))?;
        match source.runtime_role {
            CircuitRuntimeRole::SignalProcessor => {
                let execution = self
                    .manifest
                    .component_executions
                    .iter()
                    .find(|execution| execution.component_id == source_component_id)
                    .ok_or_else(|| {
                        format!("source component {source_component_id:?} has no execution contract")
                    })?;
                self.validate_component_execution(execution, capabilities)?;
            }
            CircuitRuntimeRole::InputTransducer => {
                let spec = &self.manifest.input_transducer;
                capabilities.validate_shader(
                    &self.package_root,
                    &spec.shader_path,
                    spec.spec.local_size_x,
                )?;
                capabilities.validate_shader(
                    &self.package_root,
                    &spec.batch_shader_path,
                    spec.spec.local_size_x,
                )?;
            }
            CircuitRuntimeRole::OutputTransducer => {
                let spec = &self.manifest.output_transducer;
                for (path, local_size_x) in [
                    (&spec.embedding_norm_shader_path, spec.spec.norm_local_size_x),
                    (&spec.embedding_norm_batch_shader_path, spec.spec.norm_local_size_x),
                    (&spec.projection_shader_path, spec.spec.projection_local_size_x),
                    (&spec.projection_batch_shader_path, spec.spec.projection_local_size_x),
                ] {
                    capabilities.validate_shader(&self.package_root, path, local_size_x)?;
                }
            }
            CircuitRuntimeRole::Sampler => {
                for kernel in &self.manifest.sampler.kernels {
                    capabilities.validate_shader(
                        &self.package_root,
                        &kernel.shader_path,
                        kernel.local_size_x,
                    )?;
                }
            }
            CircuitRuntimeRole::DraftProcessor => {
                let execution = self
                    .manifest
                    .speculative_decoders
                    .iter()
                    .flat_map(|decoder| &decoder.component_executions)
                    .find(|execution| execution.component_id == source_component_id)
                    .ok_or_else(|| {
                        format!("draft component {source_component_id:?} has no execution contract")
                    })?;
                self.validate_component_execution(execution, capabilities)?;
            }
            CircuitRuntimeRole::DraftOutputTransducer => {
                let output = self
                    .manifest
                    .speculative_decoders
                    .iter()
                    .map(|decoder| &decoder.output_transducer)
                    .find(|output| output.component_id == source_component_id)
                    .ok_or_else(|| {
                        format!("draft output {source_component_id:?} has no execution contract")
                    })?;
                capabilities.validate_shader(
                    &self.package_root,
                    &output.norm_shader_path,
                    output.norm_local_size_x,
                )?;
                capabilities.validate_shader(
                    &self.package_root,
                    &output.projection_shader_path,
                    output.projection_local_size_x,
                )?;
            }
            CircuitRuntimeRole::DraftInputAdapter => {}
        }
        Ok(())
    }

    fn validate_component_execution(
        &self,
        execution: &crate::VulkanResidentComponentExecutionSpec,
        capabilities: &DeviceCapabilityView,
    ) -> Result<(), String> {
        for kernel in &execution.kernels {
            capabilities.validate_shader(
                &self.package_root,
                &kernel.shader_path,
                kernel.local_size_x,
            )?;
            if !kernel.batch_implementations.is_empty()
                && !kernel.batch_implementations.iter().any(|implementation| {
                    capabilities
                        .validate_batch_requirements(&implementation.device_requirements)
                        .and_then(|_| {
                            implementation.stages.iter().try_for_each(|stage| {
                                capabilities.validate_shader(
                                    &self.package_root,
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
}

fn json_string_set(value: Option<&Value>) -> BTreeSet<String> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(ToOwned::to_owned)
        .collect()
}

fn json_shape_set(value: Option<&Value>) -> BTreeSet<[u32; 3]> {
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
mod device_compatibility_tests {
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
        assert_eq!(json_shape_set(Some(&value)), BTreeSet::from([[16, 16, 32]]));
    }

    fn capability_view() -> DeviceCapabilityView {
        DeviceCapabilityView {
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
        let requirements = crate::VulkanResidentVulkanDeviceRequirements {
            vulkan_device_extensions: vec!["VK_EXT_fixture".to_string()],
            vulkan_features: vec![crate::VulkanShaderFeature::ShaderFloat16],
            subgroup_operations: vec![crate::VulkanSubgroupOperation::Basic],
            cooperative_bfloat16_shape: Some([16, 16, 32]),
            cooperative_float8_e4m3_shape: Some([16, 16, 64]),
            subgroup_size: Some(32),
        };
        assert!(capabilities.validate_batch_requirements(&requirements).is_ok());

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
