fn placement_device_ids(components: &[ComponentPlacement]) -> Vec<String> {
    let mut device_ids = components
        .iter()
        .map(|component| component.device_id.clone())
        .collect::<Vec<_>>();
    device_ids.sort();
    device_ids.dedup();
    device_ids
}

fn runtime_model_placement(
    manifest_dir: &Path,
    runtime_model: &VulkanResidentRuntimeModel,
) -> Result<nerve_runtime::StreamCircuitPlacementPlan, Box<dyn Error>> {
    let graph = runtime_model.resolved_graph(manifest_dir.to_path_buf())?;
    graph
        .placement_plan(&runtime_model.placement)
        .map_err(|error| Box::new(error) as Box<dyn Error>)
}

fn tokenizer_dir_from_package(package_manifest: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let manifest = VulkanResidentModelPackageManifest::from_json_file(package_manifest)?;
    let manifest_dir = package_manifest
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let tokenizer_dir = resolve_package_path(&manifest_dir, &manifest.tokenizer.path);
    if !tokenizer_dir.join("tokenizer.json").is_file() {
        return Err(Box::new(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "compiled package declares tokenizer at {}, but tokenizer.json is missing",
                tokenizer_dir.display()
            ),
        )));
    }
    Ok(tokenizer_dir)
}

fn resolve_package_path(manifest_dir: &Path, raw_path: &str) -> PathBuf {
    let path = Path::new(raw_path);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        manifest_dir.join(path)
    }
}

fn runtime_model(
    args: &Args,
    package_manifest: &Path,
) -> Result<VulkanResidentRuntimeModel, Box<dyn Error>> {
    let manifest = VulkanResidentModelPackageManifest::from_json_file(package_manifest)?;
    let mut model = manifest.mount_runtime_graph_controls(
        args.default_device_id.as_deref(),
        &args.node_devices,
        &args.duplicate_after,
        args.source_chain.as_deref(),
    )?;
    for (component_id, device_ids) in &args.component_shard_devices {
        model = model.with_component_shard_devices(component_id, device_ids.clone())?;
    }
    Ok(model)
}

fn runtime_uses_explicit_placement(args: &Args) -> bool {
    args.default_device_id.is_some()
        || !args.node_devices.is_empty()
        || !args.component_shard_devices.is_empty()
        || !args.device_bindings.is_empty()
        || args.vulkan_device_index.is_some()
}

fn runtime_auto_placement_device_is_eligible(device: &VulkanComputeDeviceInfo) -> bool {
    device.device_type != "integrated_gpu"
}

fn rank_runtime_auto_placement_candidates(
    mut measured: Vec<(bool, usize, VulkanRuntimePlacementCandidate)>,
) -> Vec<VulkanRuntimePlacementCandidate> {
    measured.sort_by_key(|(selected_by_default, index, candidate)| {
        (
            std::cmp::Reverse(candidate.safe_capacity_bytes),
            !*selected_by_default,
            *index,
        )
    });
    measured
        .into_iter()
        .map(|(_, _, candidate)| candidate)
        .collect()
}

fn runtime_capacity_packed_model(
    args: &Args,
    manifest_dir: &Path,
    runtime_model: VulkanResidentRuntimeModel,
    context_capacity_activations: usize,
) -> Result<VulkanResidentRuntimeModel, Box<dyn Error>> {
    if runtime_uses_explicit_placement(args) {
        return Ok(runtime_model);
    }
    let catalog = runtime_vulkan_device_catalog(args)?;
    let available_devices = catalog.available_compute_devices();
    let profiles = catalog.available_hardware_profiles()?;
    let default_physical_index = available_devices
        .iter()
        .find(|device| device.selected_by_default)
        .or_else(|| available_devices.first())
        .map(|device| device.physical_device_index);
    let mut capability_groups = BTreeMap::<String, Vec<&VulkanComputeDeviceInfo>>::new();
    for device in available_devices {
        // Integrated display devices are not automatic inference targets. They
        // commonly own scanout/compositor allocations and probing them would
        // itself create a runtime context. A user can still target one through
        // explicit placement controls when that trade-off is intentional.
        if !runtime_auto_placement_device_is_eligible(device) {
            continue;
        }
        let profile = profiles
            .iter()
            .find(|profile| {
                profile.hardware_identity.stable_device_id == device.physical_device_id
            })
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "physical device {:?} has no hardware-process profile",
                        device.physical_device_id,
                    ),
                )
            })?;
        capability_groups
            .entry(profile.capability_class.clone())
            .or_default()
            .push(device);
    }
    if capability_groups.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "automatic placement found no non-integrated Vulkan compute devices",
        )
        .into());
    }
    let mut groups = capability_groups.into_values().collect::<Vec<_>>();
    for group in &mut groups {
        group.sort_by_key(|device| {
            (
                Some(device.physical_device_index) != default_physical_index,
                device.physical_device_index,
            )
        });
    }
    groups.sort_by_key(|group| {
        (
            !group.iter().any(|device| {
                Some(device.physical_device_index) == default_physical_index
            }),
            group
                .first()
                .map(|device| device.physical_device_index)
                .unwrap_or(usize::MAX),
        )
    });

    let mut failures = Vec::new();
    for group in groups {
        let mut measured_candidates = Vec::with_capacity(group.len());
        let mut opened_devices = Vec::with_capacity(group.len());
        for device_info in group {
            let device = catalog.open_physical_device_index(device_info.physical_device_index)?;
            let safe_capacity_bytes = usize::try_from(
                device.device_local_memory_budget().reservable_bytes,
            )
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Vulkan reservable device memory exceeds usize",
                )
            })?;
            measured_candidates.push((
                device_info.selected_by_default,
                device_info.physical_device_index,
                VulkanRuntimePlacementCandidate {
                    device_id: device_info.physical_device_id.clone(),
                    safe_capacity_bytes,
                },
            ));
            opened_devices.push(device);
        }
        // Capacity is measured before ordering so a partially reserved default
        // GPU cannot force an unnecessary spill when another compatible GPU
        // can host the model alone. Equal capacities retain stable topology and
        // catalog-default preference.
        let candidates = rank_runtime_auto_placement_candidates(measured_candidates);
        let first_candidate = candidates
            .first()
            .expect("nonempty capability group produced candidates");
        let first_profile = profiles
            .iter()
            .find(|profile| {
                profile.hardware_identity.stable_device_id == first_candidate.device_id
            })
            .expect("candidate hardware profile was validated above")
            .clone();
        let colocated = runtime_model
            .clone()
            .coalesce_placement_to_device(&first_candidate.device_id);
        let selected_model = match colocated.select_and_apply_runtime_implementations(
            manifest_dir,
            &BTreeMap::from([(first_candidate.device_id.clone(), first_profile)]),
            RuntimeExecutionEnvelope {
                phases: vec!["decode".to_string(), "prefill".to_string()],
                activation_batch: RuntimeInclusiveRange {
                    minimum: 1,
                    maximum: context_capacity_activations.max(1),
                },
                context_activations: RuntimeInclusiveRange {
                    minimum: 0,
                    maximum: context_capacity_activations,
                },
                state_activations: RuntimeInclusiveRange {
                    minimum: 0,
                    maximum: context_capacity_activations,
                },
                speculative_draft_tokens: args.speculative_draft_tokens,
            },
        ) {
            Ok((selected, _)) => selected,
            Err(error) => {
                failures.push(format!(
                    "{} compatible device(s) cannot select runtime representations: {error}",
                    candidates.len(),
                ));
                drop(opened_devices);
                continue;
            }
        };
        let tensor_index = selected_model.load_runtime_tensor_index(manifest_dir)?;
        match capacity_pack_vulkan_runtime_model(
            manifest_dir,
            &selected_model,
            &tensor_index,
            &candidates,
            context_capacity_activations,
            args.speculative_draft_tokens,
            args.resource_residency_policy,
        ) {
            Ok(selected) => {
                let retained_bytes = selected
                    .residency_plan
                    .device_plans
                    .iter()
                    .map(|plan| {
                        vulkan_runtime_maximum_device_resident_bytes(plan)
                            .map(|bytes| format!("{}={bytes}", plan.device_id))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                eprintln!(
                    "nerve runtime auto-placement: strategy=capacity_packed_minimum_devices, devices={:?}, maximum_retained_bytes={:?}",
                    selected.selected_device_ids,
                    retained_bytes,
                );
                drop(opened_devices);
                return Ok(selected.runtime_model);
            }
            Err(error) => failures.push(format!(
                "{} compatible device(s): {error}",
                candidates.len(),
            )),
        }
        drop(opened_devices);
    }
    Err(io::Error::new(
        io::ErrorKind::OutOfMemory,
        format!(
            "no compatible capacity-packed device group can retain the runtime model: {}",
            failures.join("; "),
        ),
    )
    .into())
}

struct RuntimeBoundVulkanDevices {
    devices: BTreeMap<String, Rc<VulkanComputeDevice>>,
    hardware_profiles: BTreeMap<String, HardwareProcessProfile>,
    physical_device_indices: BTreeMap<String, usize>,
    physical_device_ids: BTreeMap<String, String>,
    available_devices: Vec<VulkanComputeDeviceInfo>,
}

fn runtime_vulkan_device_catalog(
    args: &Args,
) -> Result<VulkanComputeDeviceCatalog, nerve_runtime::VulkanError> {
    if args.allowed_physical_device_ids.is_empty() {
        VulkanComputeDeviceCatalog::discover()
    } else {
        VulkanComputeDeviceCatalog::discover_allowed_physical_device_ids(
            &args.allowed_physical_device_ids,
        )
    }
}

fn runtime_physical_device_bindings_in(
    args: &Args,
    logical_device_ids: &[String],
    available_devices: &[VulkanComputeDeviceInfo],
) -> Result<BTreeMap<String, usize>, io::Error> {
    let default_physical_device_index = if let Some(index) = args.vulkan_device_index {
        index
    } else {
        available_devices
            .iter()
            .find(|device| device.selected_by_default)
            .or_else(|| available_devices.first())
            .map(|device| device.physical_device_index)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    "no Vulkan compute-capable physical devices are available",
                )
            })?
    };
    let mut logical_device_ids = logical_device_ids.to_vec();
    logical_device_ids.sort();
    logical_device_ids.dedup();
    logical_device_ids
        .into_iter()
        .map(|logical_device_id| {
            let physical_device_index = runtime_mount_physical_device_index(
                args,
                &logical_device_id,
                default_physical_device_index,
                available_devices,
            )?;
            Ok((logical_device_id, physical_device_index))
        })
        .collect()
}

fn runtime_bound_vulkan_devices(
    args: &Args,
    logical_device_ids: &[String],
) -> Result<RuntimeBoundVulkanDevices, Box<dyn Error>> {
    validate_explicit_logical_device_bindings(args, logical_device_ids)?;
    let device_catalog = runtime_vulkan_device_catalog(args)?;
    let available_devices = device_catalog.available_compute_devices();
    let available_profiles = device_catalog.available_hardware_profiles()?;
    let requested_bindings =
        runtime_physical_device_bindings_in(args, logical_device_ids, available_devices)?;
    let mut devices = BTreeMap::new();
    let mut physical_devices: BTreeMap<usize, Rc<VulkanComputeDevice>> = BTreeMap::new();
    let mut physical_device_indices = BTreeMap::new();
    let mut physical_device_ids = BTreeMap::new();
    let mut hardware_profiles = BTreeMap::new();

    for (logical_device_id, physical_device_index) in requested_bindings {
        let available_device = available_devices
            .iter()
            .find(|device| device.physical_device_index == physical_device_index)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!(
                        "Vulkan physical device index {physical_device_index} is not available"
                    ),
                )
            })?;
        let device = if let Some(device) = physical_devices.get(&physical_device_index) {
            Rc::clone(device)
        } else {
            let device = Rc::new(device_catalog.open_physical_device_index(physical_device_index)?);
            physical_devices.insert(physical_device_index, Rc::clone(&device));
            device
        };
        devices.insert(logical_device_id.clone(), device);
        physical_device_indices.insert(logical_device_id.clone(), physical_device_index);
        physical_device_ids.insert(
            logical_device_id.clone(),
            available_device.physical_device_id.clone(),
        );
        let hardware_profile = available_profiles
            .iter()
            .find(|profile| {
                profile.hardware_identity.stable_device_id
                    == available_device.physical_device_id
            })
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "physical device {:?} has no hardware-process profile",
                        available_device.physical_device_id
                    ),
                )
            })?;
        hardware_profiles.insert(
            logical_device_id.clone(),
            hardware_profile.clone(),
        );
    }

    Ok(RuntimeBoundVulkanDevices {
        devices,
        hardware_profiles,
        physical_device_indices,
        physical_device_ids,
        available_devices: available_devices.to_vec(),
    })
}

fn validate_explicit_logical_device_bindings(
    args: &Args,
    logical_device_ids: &[String],
) -> Result<(), io::Error> {
    let declared = logical_device_ids
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let unknown = args
        .device_bindings
        .keys()
        .filter(|device_id| !declared.contains(device_id.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    if !unknown.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "runtime device bindings reference logical devices absent from \
                 the effective graph: {unknown:?}; declared logical devices: \
                 {logical_device_ids:?}"
            ),
        ));
    }
    Ok(())
}

fn bound_devices_report(bound_devices: &RuntimeBoundVulkanDevices) -> Vec<RuntimeBoundDevice> {
    bound_devices
        .devices
        .iter()
        .map(|(logical_device_id, device)| {
            let physical_device_index = bound_devices
                .physical_device_indices
                .get(logical_device_id)
                .copied();
            RuntimeBoundDevice {
                device_id: logical_device_id.clone(),
                target: bound_devices
                    .physical_device_ids
                    .get(logical_device_id)
                    .cloned(),
                physical_device_index,
                device_name: device.device_name().to_string(),
            }
        })
        .collect::<Vec<_>>()
}

fn runtime_edge_routes_report(
    args: &Args,
    edges: &[ComponentEdgePlacement],
    available_devices: &[VulkanComputeDeviceInfo],
) -> RuntimeEdgeRoutes {
    RuntimeEdgeRoutes::from_edges(edges, |device_id| {
        runtime_target_for_logical_device(args, device_id, available_devices)
    })
}

fn bound_edge_routes_report(
    bound_devices: &RuntimeBoundVulkanDevices,
    edges: &[ComponentEdgePlacement],
) -> RuntimeEdgeRoutes {
    RuntimeEdgeRoutes::from_edges(edges, |device_id| {
        let physical_device_index = bound_devices
            .physical_device_indices
            .get(device_id)
            .copied();
        RuntimeEdgeRouteTarget {
            target: bound_devices.physical_device_ids.get(device_id).cloned(),
            physical_device_index,
            binding_source: "mounted".to_string(),
        }
    })
}

fn runtime_mount_physical_device_index(
    args: &Args,
    logical_device_id: &str,
    default_physical_device_index: usize,
    available_devices: &[VulkanComputeDeviceInfo],
) -> Result<usize, io::Error> {
    if let Some(target) = args.device_bindings.get(logical_device_id) {
        return resolve_runtime_vulkan_physical_device_ref_in(target, available_devices)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "logical device {logical_device_id:?} is bound to unsupported target {target:?}; local mounted execution supports vulkan:N or cpuN targets"
                    ),
                )
            });
    }
    match resolve_runtime_vulkan_physical_device_ref_in(logical_device_id, available_devices)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?
    {
        Some(index) => Ok(index),
        None if logical_device_id.contains(':') => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "logical device id {logical_device_id:?} looks like an unsupported direct runtime target; local mounted execution supports vulkan:N or cpuN targets"
            ),
        )),
        None => Ok(default_physical_device_index),
    }
}

fn runtime_target_for_logical_device(
    args: &Args,
    logical_device_id: &str,
    available_devices: &[VulkanComputeDeviceInfo],
) -> RuntimeEdgeRouteTarget {
    if let Some(target) = args.device_bindings.get(logical_device_id) {
        let physical_device_index = resolve_runtime_vulkan_physical_device_ref_in(
            target,
            available_devices,
        )
            .ok()
            .flatten();
        return RuntimeEdgeRouteTarget {
            target: Some(target.clone()),
            physical_device_index,
            binding_source: "explicit".to_string(),
        };
    }
    match resolve_runtime_vulkan_physical_device_ref_in(logical_device_id, available_devices) {
        Ok(Some(index)) => RuntimeEdgeRouteTarget {
            target: Some(logical_device_id.to_string()),
            physical_device_index: Some(index),
            binding_source: "device_id".to_string(),
        },
        Ok(None) | Err(_) if logical_device_id.contains(':') => RuntimeEdgeRouteTarget {
            target: Some(logical_device_id.to_string()),
            physical_device_index: None,
            binding_source: "device_id".to_string(),
        },
        Ok(None) | Err(_) => {
            let default_physical_device_index =
                runtime_report_default_vulkan_physical_device_index(args, available_devices);
            let target = default_physical_device_index.map(|index| format!("vulkan:{index}"));
            RuntimeEdgeRouteTarget {
                physical_device_index: default_physical_device_index,
                target,
                binding_source: if args.vulkan_device_index.is_some() {
                    "process_default".to_string()
                } else {
                    "runtime_default".to_string()
                },
            }
        }
    }
}

fn runtime_report_default_vulkan_physical_device_index(
    args: &Args,
    available_devices: &[VulkanComputeDeviceInfo],
) -> Option<usize> {
    args.vulkan_device_index
        .or_else(|| {
            args.default_device_id.as_deref().and_then(|device_id| {
                resolve_runtime_vulkan_physical_device_ref_in(device_id, available_devices)
                    .ok()
                    .flatten()
            })
        })
        .or_else(|| {
            available_devices
                .iter()
                .find(|device| device.selected_by_default)
                .or_else(|| available_devices.first())
                .map(|device| device.physical_device_index)
        })
}
