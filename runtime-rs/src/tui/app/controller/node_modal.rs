impl App {
    fn open_selected_node(&mut self) {
        let Some(editor) = &self.editor else {
            return;
        };
        let Some(instance_id) = &self.selected_instance_id else {
            return;
        };
        let instances = editor.instances();
        let Some(instance) = instances
            .iter()
            .find(|instance| &instance.instance_id == instance_id)
            .cloned()
        else {
            return;
        };
        let Some(source) = editor.source_component_for_instance(instance_id).cloned() else {
            return;
        };
        let devices = editor
            .available_devices()
            .iter()
            .map(|device| {
                let name = device.device_name.as_deref().unwrap_or("unnamed device");
                let memory = device
                    .memory_heaps
                    .as_ref()
                    .and_then(|heaps| {
                        heaps
                            .iter()
                            .filter(|heap| heap.device_local)
                            .map(|heap| heap.size_bytes)
                            .max()
                    })
                    .map(|bytes| format!(" · {:.1} GiB", bytes as f64 / 1_073_741_824.0))
                    .unwrap_or_default();
                let compatibility = editor
                    .validate_instance_device_compatibility(instance_id, &device.device_id);
                let status = if !device.available {
                    "UNAVAILABLE · "
                } else if compatibility.is_err() {
                    "INCOMPATIBLE · "
                } else {
                    ""
                };
                let selectable = compatibility.is_ok();
                (
                    device.device_id.clone(),
                    format!("{status}{} · {name}{memory}", device.device_id),
                    selectable,
                    compatibility.err().map(|error| error.to_string()),
                )
            })
            .chain(
                (!editor
                    .available_devices()
                    .iter()
                    .any(|device| device.device_id == instance.device_id))
                .then(|| {
                    (
                        instance.device_id.clone(),
                        format!("UNAVAILABLE · {}", instance.device_id),
                        false,
                        Some(format!(
                            "runtime device {:?} is no longer present",
                            instance.device_id
                        )),
                    )
                }),
            )
            .collect::<Vec<_>>();
        let device_ids = devices
            .iter()
            .map(|(id, _, _, _)| id.clone())
            .collect::<Vec<_>>();
        let device_labels = devices
            .iter()
            .map(|(_, label, _, _)| label.clone())
            .collect::<Vec<_>>();
        let device_selectable = devices
            .iter()
            .map(|(_, _, selectable, _)| *selectable)
            .collect::<Vec<_>>();
        let device_diagnostics = devices
            .iter()
            .filter_map(|(id, _, _, error)| {
                error.as_ref().map(|error| format!("{id}: {error}"))
            })
            .collect::<Vec<_>>();
        let device_index = device_ids
            .iter()
            .position(|device| device == &instance.device_id)
            .unwrap_or(0);
        let policy_targets = editor
            .state_policy_target_ids(instance_id)
            .unwrap_or_default();
        let (policy, target) = match &instance.state_policy {
            StreamCircuitNodeInstanceStatePolicy::Fresh => (NodePolicyKind::Independent, None),
            StreamCircuitNodeInstanceStatePolicy::CloneFrom { instance_id } => {
                (NodePolicyKind::Clone, Some(instance_id))
            }
            StreamCircuitNodeInstanceStatePolicy::ShareWith { instance_id } => {
                (NodePolicyKind::Share, Some(instance_id))
            }
        };
        let policy_target_index = target
            .and_then(|target| {
                policy_targets
                    .iter()
                    .position(|candidate| candidate == target)
            })
            .unwrap_or(0);
        let properties = source
            .control_schemas
            .iter()
            .cloned()
            .map(|schema| {
                let value = editor
                    .effective_instance_control_value(instance_id, &schema.id)
                    .unwrap_or(Value::Null);
                NodePropertyDraft::new(schema, value)
            })
            .collect();
        let (selected_implementation_id, implementation_selection_error) =
            match editor.runtime_implementation_selection() {
                Ok(selection) => (
                    selection
                        .selected
                        .iter()
                        .find(|selected| {
                            selected
                                .instance_ids
                                .contains(instance_id)
                        })
                        .map(|selected| {
                            selected.implementation_id.clone()
                        }),
                    None,
                ),
                Err(error) => (None, Some(error.to_string())),
            };
        self.overlay = Some(Overlay::Node(NodeModalState {
            instance_id: instance.instance_id,
            source,
            occurrence: instance.occurrence,
            device_ids,
            device_labels,
            device_selectable,
            device_diagnostics,
            device_index,
            original_device_id: instance.device_id,
            selected_implementation_id,
            implementation_selection_error,
            enabled: instance.enabled,
            policy,
            policy_targets,
            policy_target_index,
            properties,
            anatomy_expanded: false,
            anatomy_scroll: 0,
            focus_row: 0,
            error: None,
        }));
    }

    fn apply_node_modal(&mut self) {
        let Some(Overlay::Node(modal)) = &self.overlay else {
            return;
        };
        let modal = modal.clone();
        let Some(editor) = &self.editor else {
            return;
        };
        let Some(device_id) = modal.device_ids.get(modal.device_index) else {
            if let Some(Overlay::Node(modal)) = &mut self.overlay {
                modal.error = Some("No compatible runtime device is available".to_string());
            }
            return;
        };
        if !modal
            .device_selectable
            .get(modal.device_index)
            .copied()
            .unwrap_or(false)
        {
            if let Some(Overlay::Node(modal)) = &mut self.overlay {
                modal.error = Some(format!(
                    "Runtime device {device_id:?} is unavailable or incompatible"
                ));
            }
            return;
        }
        if modal.policy != NodePolicyKind::Independent && modal.policy_targets.is_empty() {
            if let Some(Overlay::Node(modal)) = &mut self.overlay {
                modal.error = Some("This state policy needs another node instance".to_string());
            }
            return;
        }
        if let Some(property) = modal
            .properties
            .iter()
            .find(|property| property.editable() && property.error.is_some())
        {
            if let Some(Overlay::Node(modal)) = &mut self.overlay {
                modal.error = Some(format!(
                    "{}: {}",
                    property.schema.name,
                    property.error.as_deref().unwrap_or("invalid value")
                ));
            }
            return;
        }
        let mut candidate = editor.clone();
        // Remove the instance's old state dependency first so a coordinated
        // enabled/device/policy edit is validated as one modal transaction.
        let mut result = candidate
            .set_instance_state_policy(
                &modal.instance_id,
                StreamCircuitNodeInstanceStatePolicy::Fresh,
            )
            .and_then(|_| candidate.set_instance_enabled(&modal.instance_id, modal.enabled))
            .and_then(|_| {
                if device_id == &modal.original_device_id {
                    Ok(())
                } else {
                    candidate.set_instance_device(&modal.instance_id, device_id)
                }
            })
            .and_then(|_| {
                candidate.set_instance_state_policy(&modal.instance_id, modal.state_policy())
            });
        if result.is_ok() {
            for property in modal
                .properties
                .iter()
                .filter(|property| property.editable() && property.changed())
            {
                if let Err(error) = candidate.set_instance_control_value(
                    &modal.instance_id,
                    &property.schema.id,
                    property.value.clone(),
                ) {
                    result = Err(error);
                    break;
                }
            }
        }
        if result.is_ok() {
            let validation = candidate.validation();
            if !validation.valid {
                result = Err(RuntimeEditorError(validation.errors.join("; ")));
            }
        }
        match result {
            Ok(()) => {
                let lifecycle = modal
                    .properties
                    .iter()
                    .filter(|property| property.editable() && property.changed())
                    .flat_map(|property| {
                        [
                            property
                                .schema
                                .requires_state_reset
                                .then_some("state reset"),
                            property.schema.requires_remount.then_some("remount"),
                            property.schema.requires_recompile.then_some("recompile"),
                        ]
                    })
                    .flatten()
                    .collect::<BTreeSet<_>>();
                self.editor = Some(candidate);
                self.overlay = None;
                self.status = if lifecycle.is_empty() {
                    format!("Updated {} · draft not mounted", modal.instance_id)
                } else {
                    format!(
                        "Updated {} · requires {} · draft not mounted",
                        modal.instance_id,
                        lifecycle.into_iter().collect::<Vec<_>>().join(", ")
                    )
                };
            }
            Err(error) => {
                if let Some(Overlay::Node(modal)) = &mut self.overlay {
                    modal.error = Some(error.to_string());
                }
            }
        }
    }

}
