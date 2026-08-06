impl RuntimeModelEditor {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, RuntimeEditorError> {
        Self::load_with_device_provider(path, |default_device_id| {
            discover_runtime_devices(default_device_id, None)
        })
    }

    pub fn load_with_available_devices(
        path: impl AsRef<Path>,
        available_devices: Vec<RuntimeAvailableDevice>,
    ) -> Result<Self, RuntimeEditorError> {
        Self::load_with_device_provider(path, |_| available_devices)
    }

    fn load_with_device_provider(
        path: impl AsRef<Path>,
        devices: impl FnOnce(&str) -> Vec<RuntimeAvailableDevice>,
    ) -> Result<Self, RuntimeEditorError> {
        let manifest_path = match classify_runtime_model_path(path)? {
            RuntimeModelPathKind::CompiledPackage { manifest } => manifest,
            RuntimeModelPathKind::SafetensorsSource { .. } => {
                return Err(RuntimeEditorError(
                    "Safetensors sources must be compiled before loading the runtime editor"
                        .to_string(),
                ));
            }
        };
        let package_root = manifest_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let manifest = VulkanResidentModelPackageManifest::from_json_file(&manifest_path)?;
        let implementation_catalog = manifest
            .implementation_catalog(&package_root)
            .map_err(|error| RuntimeEditorError(error.to_string()))?;
        let source_graph = manifest
            .resolved_source_graph(package_root.clone())
            .map_err(|error| RuntimeEditorError(error.to_string()))?;
        let draft = manifest
            .runtime_graph_from_controls(None, &BTreeMap::new(), &[], None)
            .map_err(|error| RuntimeEditorError(error.to_string()))?;
        let source_components = source_components(
            &manifest,
            &implementation_catalog,
        );
        let source_by_layer = source_components
            .iter()
            .filter_map(|component| {
                component
                    .layer_index
                    .map(|layer_index| (layer_index, component.source_id.clone()))
            })
            .fold(
                BTreeMap::<usize, Vec<String>>::new(),
                |mut by_layer, entry| {
                    by_layer.entry(entry.0).or_default().push(entry.1);
                    by_layer
                },
            );
        let source_ids = source_components
            .iter()
            .map(|component| component.source_id.clone())
            .collect();
        let available_devices = devices(RUNTIME_DEFAULT_LOGICAL_DEVICE_ID);
        Ok(Self {
            package_manifest_path: manifest_path,
            package_root,
            manifest,
            implementation_catalog,
            source_graph,
            source_components,
            source_by_layer,
            source_ids,
            available_devices,
            draft,
        })
    }

    pub fn package_manifest_path(&self) -> &Path {
        &self.package_manifest_path
    }

    pub fn package_root(&self) -> &Path {
        &self.package_root
    }

    pub fn package_id(&self) -> &str {
        &self.manifest.package_id
    }

    pub fn max_context_activations(&self) -> usize {
        self.manifest.max_context_activations
    }

    pub fn supported_resource_residency_policies(
        &self,
    ) -> &[ResourceResidencyPolicy] {
        &self.manifest.resource_residency.supported_policies
    }

    pub fn supports_runtime_resource_residency_policy(
        &self,
        policy: ResourceResidencyPolicy,
    ) -> bool {
        self.manifest
            .resource_residency
            .supported_policies
            .contains(&policy.required_compiled_loading_policy())
    }

    pub fn available_runtime_resource_residency_policies(
        &self,
    ) -> Vec<ResourceResidencyPolicy> {
        [
            ResourceResidencyPolicy::Eager,
            ResourceResidencyPolicy::DemandRetained,
            ResourceResidencyPolicy::DemandPaged,
        ]
        .into_iter()
        .filter(|policy| self.supports_runtime_resource_residency_policy(*policy))
        .collect()
    }

    pub fn source_components(&self) -> &[RuntimeEditorSourceComponent] {
        &self.source_components
    }

    pub fn implementation_catalog(
        &self,
    ) -> &crate::RuntimeImplementationCatalog {
        &self.implementation_catalog
    }

    pub fn runtime_implementation_selection(
        &self,
    ) -> Result<
        crate::RuntimeImplementationSelectionReport,
        RuntimeEditorError,
    > {
        let runtime_model = self
            .manifest
            .clone()
            .mount_runtime_graph(&self.draft)
            .map_err(|error| {
                RuntimeEditorError(error.to_string())
            })?;
        let mut profiles = BTreeMap::new();
        for logical_device_id in runtime_model.placement_device_ids() {
            let device = self
                .available_devices
                .iter()
                .find(|device| {
                    device.device_id == logical_device_id
                        || device.runtime_device_id.as_deref()
                            == Some(logical_device_id.as_str())
                })
                .ok_or_else(|| {
                    RuntimeEditorError(format!(
                        "runtime device {logical_device_id:?} is unavailable"
                    ))
                })?;
            let profile = device.hardware_profile.clone().ok_or_else(
                || {
                    RuntimeEditorError(format!(
                        "runtime device {logical_device_id:?} has no hardware-process profile"
                    ))
                },
            )?;
            profiles.insert(logical_device_id, profile);
        }
        let request =
            crate::RuntimeSelectionRequest::from_vulkan_runtime_model(
                &runtime_model,
                &profiles,
                crate::RuntimeExecutionEnvelope {
                    phases: vec![
                        "decode".to_string(),
                        "prefill".to_string(),
                    ],
                    activation_batch:
                        crate::RuntimeInclusiveRange {
                            minimum: 1,
                            maximum: self
                                .manifest
                                .max_context_activations,
                        },
                    context_activations:
                        crate::RuntimeInclusiveRange {
                            minimum: 0,
                            maximum: self
                                .manifest
                                .max_context_activations,
                        },
                    state_activations:
                        crate::RuntimeInclusiveRange {
                            minimum: 0,
                            maximum: self
                                .manifest
                                .max_context_activations,
                        },
                    speculative_draft_tokens: 0,
                    residency_policy: "eager".to_string(),
                },
                true,
            )
            .map_err(|error| {
                RuntimeEditorError(error.to_string())
            })?;
        self.implementation_catalog
            .select(&request)
            .map_err(|error| RuntimeEditorError(error.to_string()))
    }

    pub fn available_devices(&self) -> &[RuntimeAvailableDevice] {
        &self.available_devices
    }

    pub fn replace_available_devices(
        &mut self,
        available_devices: Vec<RuntimeAvailableDevice>,
    ) {
        self.available_devices = available_devices;
    }

    pub fn draft(&self) -> &StreamCircuitRuntimeGraph {
        &self.draft
    }

    pub fn layer_sequence(&self) -> Vec<usize> {
        let layer_by_source = self
            .source_components
            .iter()
            .filter_map(|component| {
                component
                    .layer_index
                    .map(|layer_index| (component.source_id.as_str(), layer_index))
            })
            .collect::<BTreeMap<_, _>>();
        self.draft
            .instances
            .iter()
            .filter_map(|instance| {
                layer_by_source
                    .get(instance.source_component_id.as_str())
                    .copied()
            })
            .collect()
    }

    pub fn source_sequence(&self) -> Vec<String> {
        self.draft
            .instances
            .iter()
            .filter(|instance| instance.enabled)
            .map(|instance| instance.source_component_id.clone())
            .collect()
    }

    pub fn instances(&self) -> Vec<RuntimeEditorInstance> {
        let layer_by_source = self
            .source_components
            .iter()
            .map(|component| (component.source_id.as_str(), component.layer_index))
            .collect::<BTreeMap<_, _>>();
        let mut occurrences = BTreeMap::<&str, usize>::new();
        self.draft
            .instances
            .iter()
            .filter_map(|instance| {
                let layer_index = *layer_by_source.get(instance.source_component_id.as_str())?;
                let occurrence = occurrences
                    .entry(instance.source_component_id.as_str())
                    .and_modify(|value| *value += 1)
                    .or_insert(1);
                Some(RuntimeEditorInstance {
                    instance_id: instance.instance_id.clone(),
                    source_id: instance.source_component_id.clone(),
                    layer_index,
                    occurrence: *occurrence,
                    device_id: instance.device_id.clone(),
                    enabled: instance.enabled,
                    control_values: instance.control_values.clone(),
                    state_policy: instance.state_policy.clone(),
                })
            })
            .collect()
    }

    pub fn layer_instances(&self) -> Vec<RuntimeEditorInstance> {
        self.instances()
            .into_iter()
            .filter(|instance| instance.layer_index.is_some())
            .collect()
    }

    pub fn replace_layer_sequence(
        &mut self,
        layer_sequence: &[usize],
    ) -> Result<(), RuntimeEditorError> {
        if layer_sequence.is_empty() {
            return Err(RuntimeEditorError(
                "layer sequence must contain at least one layer".to_string(),
            ));
        }
        let source_sequence = layer_sequence
            .iter()
            .map(|layer_index| {
                let sources = self.source_by_layer.get(layer_index).ok_or_else(|| {
                    RuntimeEditorError(format!(
                        "unknown layer {layer_index}; available layers: {}",
                        available_layer_range(&self.source_by_layer)
                    ))
                })?;
                if sources.len() != 1 {
                    return Err(RuntimeEditorError(format!(
                        "layer {layer_index} has {} source components; edit the source sequence by id",
                        sources.len()
                    )));
                }
                Ok(sources[0].clone())
            })
            .collect::<Result<Vec<_>, RuntimeEditorError>>()?;
        self.replace_signal_processor_sequence(&source_sequence)
    }

    pub fn replace_signal_processor_sequence(
        &mut self,
        source_sequence: &[String],
    ) -> Result<(), RuntimeEditorError> {
        let processor_instances = self.instances_for_source_sequence(source_sequence)?;
        let chain = processor_instances
            .iter()
            .map(|instance| {
                (
                    instance.instance_id.clone(),
                    instance.source_component_id.clone(),
                )
            })
            .collect::<Vec<_>>();
        self.draft = self
            .draft
            .clone()
            .with_signal_processor_chain(&self.source_graph, &chain)?;
        Ok(())
    }

    pub fn duplicate_layer_instance_after(
        &mut self,
        instance_id: &str,
    ) -> Result<String, RuntimeEditorError> {
        let instance = self
            .layer_instances()
            .into_iter()
            .find(|instance| instance.instance_id == instance_id)
            .ok_or_else(|| {
                RuntimeEditorError(format!(
                    "runtime graph has no editable layer instance {instance_id:?}"
                ))
            })?;
        let used_instance_ids = self
            .draft
            .instances
            .iter()
            .map(|instance| instance.instance_id.clone())
            .collect::<BTreeSet<_>>();
        let occurrence = self
            .draft
            .instances
            .iter()
            .filter(|candidate| {
                candidate.source_component_id == instance.source_id
            })
            .count()
            + 1;
        let duplicate_id = allocate_instance_id(
            &instance.source_id,
            occurrence,
            &used_instance_ids,
        );
        let candidate = self.draft.clone().duplicate_after_instance(
            &self.source_graph,
            instance_id,
            duplicate_id.clone(),
        )?;
        candidate.validate_against_graph(&self.source_graph)?;
        self.draft = candidate;
        Ok(duplicate_id)
    }

    pub fn remove_layer_instance(
        &mut self,
        instance_id: &str,
    ) -> Result<(), RuntimeEditorError> {
        let remaining = self
            .layer_instances()
            .into_iter()
            .filter(|instance| instance.instance_id != instance_id)
            .map(|instance| (instance.instance_id, instance.source_id))
            .collect::<Vec<_>>();
        if remaining.len() == self.layer_instances().len() {
            return Err(RuntimeEditorError(format!(
                "runtime graph has no editable layer instance {instance_id:?}"
            )));
        }
        self.replace_signal_processor_instances(&remaining)
    }

    pub fn reorder_layer_instances(
        &mut self,
        ordered_instance_ids: &[String],
    ) -> Result<(), RuntimeEditorError> {
        let current = self.layer_instances();
        let current_ids = current
            .iter()
            .map(|instance| instance.instance_id.as_str())
            .collect::<BTreeSet<_>>();
        let ordered_ids = ordered_instance_ids
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if ordered_instance_ids.len() != current.len()
            || ordered_ids.len() != ordered_instance_ids.len()
            || ordered_ids != current_ids
        {
            return Err(RuntimeEditorError(
                "layer reorder must contain every existing instance exactly once".to_string(),
            ));
        }
        let source_by_instance = current
            .into_iter()
            .map(|instance| (instance.instance_id, instance.source_id))
            .collect::<BTreeMap<_, _>>();
        let ordered = ordered_instance_ids
            .iter()
            .map(|instance_id| {
                (
                    instance_id.clone(),
                    source_by_instance[instance_id].clone(),
                )
            })
            .collect::<Vec<_>>();
        self.replace_signal_processor_instances(&ordered)
    }

    fn replace_signal_processor_instances(
        &mut self,
        instances: &[(String, String)],
    ) -> Result<(), RuntimeEditorError> {
        if instances.is_empty() {
            return Err(RuntimeEditorError(
                "layer sequence must contain at least one layer".to_string(),
            ));
        }
        self.draft = self
            .draft
            .clone()
            .with_signal_processor_chain(&self.source_graph, instances)?;
        Ok(())
    }

    fn instances_for_source_sequence(
        &self,
        source_sequence: &[String],
    ) -> Result<Vec<StreamCircuitNodeInstance>, RuntimeEditorError> {
        let mut previous_by_source =
            BTreeMap::<String, VecDeque<StreamCircuitNodeInstance>>::new();
        for instance in &self.draft.instances {
            previous_by_source
                .entry(instance.source_component_id.clone())
                .or_default()
                .push_back(instance.clone());
        }
        let mut occurrence_by_source = BTreeMap::<String, usize>::new();
        let mut used_instance_ids = BTreeSet::new();
        let mut instances = Vec::with_capacity(source_sequence.len());
        for source_id in source_sequence {
            if !self.source_ids.contains(source_id) {
                return Err(RuntimeEditorError(format!(
                    "unknown source component {source_id:?}"
                )));
            }
            let occurrence = occurrence_by_source
                .entry(source_id.clone())
                .and_modify(|value| *value += 1)
                .or_insert(1);
            let previous = previous_by_source
                .get_mut(source_id)
                .and_then(VecDeque::pop_front);
            let instance = if let Some(previous) = previous {
                used_instance_ids.insert(previous.instance_id.clone());
                previous
            } else {
                let instance_id = allocate_instance_id(source_id, *occurrence, &used_instance_ids);
                used_instance_ids.insert(instance_id.clone());
                StreamCircuitNodeInstance {
                    instance_id,
                    source_component_id: source_id.clone(),
                    device_id: self.draft.default_device_id.clone(),
                    device_assignment:
                        StreamCircuitNodeDeviceAssignment::Automatic,
                    enabled: true,
                    control_values: BTreeMap::new(),
                    state_policy: StreamCircuitNodeInstanceStatePolicy::Fresh,
                }
            };
            instances.push(instance);
        }
        Ok(instances)
    }

    pub fn set_instance_device(
        &mut self,
        instance_id: &str,
        device_id: &str,
    ) -> Result<(), RuntimeEditorError> {
        self.validate_instance_device_compatibility(instance_id, device_id)?;
        let candidate = self
            .draft
            .clone()
            .with_instance_device(instance_id, device_id)?;
        candidate.validate_against_graph(&self.source_graph)?;
        self.draft = candidate;
        Ok(())
    }

    pub fn set_instance_enabled(
        &mut self,
        instance_id: &str,
        enabled: bool,
    ) -> Result<(), RuntimeEditorError> {
        let candidate = self
            .draft
            .clone()
            .with_instance_enabled(instance_id, enabled)?;
        candidate.validate_against_graph(&self.source_graph)?;
        self.draft = candidate;
        Ok(())
    }

    pub fn set_instance_control_value(
        &mut self,
        instance_id: &str,
        control_id: &str,
        value: Value,
    ) -> Result<(), RuntimeEditorError> {
        let source = self.source_component_for_instance(instance_id).ok_or_else(|| {
            RuntimeEditorError(format!(
                "runtime graph has no node instance {instance_id:?}"
            ))
        })?;
        let schema = source
            .control_schemas
            .iter()
            .find(|schema| schema.id == control_id)
            .ok_or_else(|| {
                RuntimeEditorError(format!(
                    "source component {} declares no control {control_id:?}",
                    source.source_id
                ))
            })?;
        validate_runtime_editor_control_value(schema, &value)?;
        let instance = self
            .draft
            .instances
            .iter_mut()
            .find(|instance| instance.instance_id == instance_id)
            .ok_or_else(|| {
                RuntimeEditorError(format!(
                    "runtime graph has no node instance {instance_id:?}"
                ))
            })?;
        instance
            .control_values
            .insert(control_id.to_string(), value);
        Ok(())
    }

    pub fn effective_instance_control_value(
        &self,
        instance_id: &str,
        control_id: &str,
    ) -> Option<Value> {
        let instance = self
            .draft
            .instances
            .iter()
            .find(|instance| instance.instance_id == instance_id)?;
        if let Some(value) = instance.control_values.get(control_id) {
            return Some(value.clone());
        }
        self.source_component_for_instance(instance_id)?
            .control_schemas
            .iter()
            .find(|schema| schema.id == control_id)
            .and_then(|schema| {
                schema
                    .current_value
                    .clone()
                    .or_else(|| schema.default_value.clone())
            })
    }

    pub fn set_instance_state_policy(
        &mut self,
        instance_id: &str,
        state_policy: StreamCircuitNodeInstanceStatePolicy,
    ) -> Result<(), RuntimeEditorError> {
        let mut candidate = self.draft.clone();
        let instance = candidate
            .instances
            .iter_mut()
            .find(|instance| instance.instance_id == instance_id)
            .ok_or_else(|| {
                RuntimeEditorError(format!(
                    "runtime graph has no node instance {instance_id:?}"
                ))
            })?;
        instance.state_policy = state_policy;
        candidate.validate_against_graph(&self.source_graph)?;
        self.draft = candidate;
        Ok(())
    }

    pub fn state_policy_target_ids(
        &self,
        instance_id: &str,
    ) -> Result<Vec<String>, RuntimeEditorError> {
        let source = self.source_component_for_instance(instance_id).ok_or_else(|| {
            RuntimeEditorError(format!(
                "runtime graph has no node instance {instance_id:?}"
            ))
        })?;
        if source.state_ports.is_empty() {
            return Ok(Vec::new());
        }
        Ok(self
            .draft
            .instances
            .iter()
            .filter(|candidate| candidate.instance_id != instance_id && candidate.enabled)
            .filter(|candidate| {
                self.source_components
                    .iter()
                    .find(|component| component.source_id == candidate.source_component_id)
                    .is_some_and(|component| component.state_ports == source.state_ports)
            })
            .map(|candidate| candidate.instance_id.clone())
            .collect())
    }

    pub fn validation(&self) -> RuntimeEditorValidation {
        let mut errors = Vec::new();
        for instance in &self.draft.instances {
            if let Err(error) = self.validate_instance_device_compatibility(
                &instance.instance_id,
                &instance.device_id,
            ) {
                errors.push(error.to_string());
            }
            if let Some(source) = self
                .source_components
                .iter()
                .find(|source| source.source_id == instance.source_component_id)
            {
                for (control_id, value) in &instance.control_values {
                    match source
                        .control_schemas
                        .iter()
                        .find(|schema| schema.id == *control_id)
                    {
                        Some(schema) => {
                            if let Err(error) = validate_runtime_editor_control_value(schema, value)
                            {
                                errors.push(format!(
                                    "instance {} control {}: {}",
                                    instance.instance_id, control_id, error
                                ));
                            }
                        }
                        None => errors.push(format!(
                            "instance {} has undeclared control {}",
                            instance.instance_id, control_id
                        )),
                    }
                }
            }
        }
        if let Err(error) = self.draft.validate_against_graph(&self.source_graph) {
            errors.push(error.to_string());
        }
        let placement = if errors.is_empty() {
            self.source_graph
                .instantiate_runtime_graph(&self.draft)
                .and_then(|graph| graph.placement_plan(&self.draft.placement_spec()))
                .map_err(|error| errors.push(error.to_string()))
                .ok()
        } else {
            None
        };
        RuntimeEditorValidation {
            valid: errors.is_empty(),
            errors,
            warnings: Vec::new(),
            placement,
        }
    }

    pub fn source_component_for_instance(
        &self,
        instance_id: &str,
    ) -> Option<&RuntimeEditorSourceComponent> {
        let source_id = self
            .draft
            .instances
            .iter()
            .find(|instance| instance.instance_id == instance_id)?
            .source_component_id
            .as_str();
        self.source_components
            .iter()
            .find(|component| component.source_id == source_id)
    }
}

#[cfg(test)]
pub(crate) fn load_runtime_model_editor_without_hardware(
    path: impl AsRef<Path>,
) -> Result<RuntimeModelEditor, RuntimeEditorError> {
    let path = path.as_ref();
    let manifest_path = match classify_runtime_model_path(path)? {
        RuntimeModelPathKind::CompiledPackage { manifest } => manifest,
        RuntimeModelPathKind::SafetensorsSource { .. } => {
            return Err(RuntimeEditorError(
                "test editor requires a compiled package".to_string(),
            ));
        }
    };
    let device_id = RUNTIME_DEFAULT_LOGICAL_DEVICE_ID.to_string();
    RuntimeModelEditor::load_with_available_devices(
        manifest_path,
        vec![RuntimeAvailableDevice {
            device_id: device_id.clone(),
            backend: "test".to_string(),
            available: true,
            hardware_profile: None,
            runtime_device_id: Some(device_id),
            physical_device_id: Some("test:0".to_string()),
            physical_device_index: Some(0),
            device_name: Some("Deterministic test device".to_string()),
            device_type: Some("test".to_string()),
            vendor_id: None,
            raw_device_id: None,
            api_version: None,
            driver_version: None,
            compute_queue_family_indices: Some(vec![0]),
            memory_heaps: Some(Vec::new()),
            selected_by_default: Some(true),
            selected_by_runtime: Some(true),
            runtime_binding: Some("test_only".to_string()),
            can_host_runtime_components_on_physical_device: Some(true),
            notes: vec!["hardware discovery disabled for this test".to_string()],
            error: None,
        }],
    )
}

#[cfg(test)]
mod model_editor_invariant_tests {
    use super::*;

    fn editor() -> RuntimeModelEditor {
        load_runtime_model_editor_without_hardware(crate::test_support::tiny_model_dir()).unwrap()
    }

    #[test]
    fn layer_identity_operations_reject_incomplete_or_unknown_requests_transactionally() {
        let mut editor = editor();
        let original = editor.draft().clone();
        assert!(editor.remove_layer_instance("missing").is_err());
        assert_eq!(editor.draft(), &original);
        assert!(editor.remove_layer_instance("layer_00").is_err());
        assert_eq!(editor.draft(), &original);

        let duplicate = editor.duplicate_layer_instance_after("layer_00").unwrap();
        let duplicated = editor.draft().clone();
        for invalid in [
            vec!["layer_00".to_string()],
            vec!["layer_00".to_string(), "layer_00".to_string()],
            vec!["layer_00".to_string(), "missing".to_string()],
        ] {
            assert!(editor.reorder_layer_instances(&invalid).is_err());
            assert_eq!(editor.draft(), &duplicated);
        }
        assert_eq!(duplicate, "layer_00@2");
    }

    #[test]
    fn duplicate_reorder_and_remove_preserve_exact_instance_identity() {
        let mut editor = editor();
        let second = editor.duplicate_layer_instance_after("layer_00").unwrap();
        let third = editor.duplicate_layer_instance_after(&second).unwrap();
        editor
            .set_instance_enabled(&second, false)
            .expect("another enabled layer keeps the graph valid");
        editor
            .reorder_layer_instances(&[third.clone(), "layer_00".to_string(), second.clone()])
            .unwrap();
        assert_eq!(
            editor
                .layer_instances()
                .iter()
                .map(|instance| instance.instance_id.as_str())
                .collect::<Vec<_>>(),
            [third.as_str(), "layer_00", second.as_str()]
        );
        assert!(!editor.layer_instances()[2].enabled);

        editor.remove_layer_instance("layer_00").unwrap();
        assert_eq!(
            editor
                .layer_instances()
                .iter()
                .map(|instance| instance.instance_id.as_str())
                .collect::<Vec<_>>(),
            [third.as_str(), second.as_str()]
        );
    }

    #[test]
    fn replacing_device_inventory_never_rewrites_the_graph_draft() {
        let mut editor = editor();
        let draft = editor.draft().clone();
        let mut unavailable = editor.available_devices()[0].clone();
        unavailable.available = false;
        unavailable.error = Some("disconnected".to_string());
        editor.replace_available_devices(vec![unavailable]);
        assert_eq!(editor.draft(), &draft);
        assert!(!editor.validation().valid);
    }
}
