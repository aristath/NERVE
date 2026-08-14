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
    // Shard-only chat controls are component-local physical overrides on top
    // of automatic placement. Their stable owner does not exist until that
    // placement converges, so validating them against the manifest's logical
    // default here would reject a valid physical pool prematurely.
    if !args.chat || runtime_uses_explicit_placement(args) {
        for (component_id, device_ids) in &args.component_shard_devices {
            model = model.with_component_shard_devices(component_id, device_ids.clone())?;
        }
    }
    Ok(model)
}

fn runtime_uses_explicit_placement(args: &Args) -> bool {
    args.default_device_id.is_some()
        || !args.node_devices.is_empty()
        || !args.device_bindings.is_empty()
        || args.vulkan_device_index.is_some()
}

fn runtime_model_without_explicit_component_shards(
    mut runtime_model: VulkanResidentRuntimeModel,
) -> VulkanResidentRuntimeModel {
    runtime_model.placement.component_shard_devices.clear();
    runtime_model
}

fn rank_runtime_auto_placement_candidates_across_capability_classes(
    mut measured: Vec<(u128, bool, usize, String, VulkanRuntimePlacementCandidate)>,
    primary_capability_class: Option<&str>,
) -> Vec<VulkanRuntimePlacementCandidate> {
    measured.sort_by_key(
        |(execution_cost, selected_by_default, index, capability_class, candidate)| {
            (
                *execution_cost,
                primary_capability_class.is_some_and(|primary| capability_class != primary),
                std::cmp::Reverse(candidate.safe_capacity_bytes),
                !*selected_by_default,
                *index,
            )
        },
    );
    measured
        .into_iter()
        .map(|(_, _, _, _, candidate)| candidate)
        .collect()
}

#[derive(Clone, Debug, PartialEq)]
struct RuntimeAutoPlacementContext {
    candidates: Vec<VulkanRuntimePlacementCandidate>,
    costs: VulkanRuntimePlacementCostModel,
    calibration_catalog: VulkanPlacementCalibrationCatalog,
    exact_runtime_model: VulkanResidentRuntimeModel,
}

struct RuntimeCapacityPackedModel {
    runtime_model: VulkanResidentRuntimeModel,
    auto_placement: Option<RuntimeAutoPlacementContext>,
}

fn runtime_capacity_packed_model(
    args: &Args,
    manifest_dir: &Path,
    runtime_model: VulkanResidentRuntimeModel,
    context_capacity_activations: usize,
) -> Result<RuntimeCapacityPackedModel, Box<dyn Error>> {
    if runtime_uses_explicit_placement(args) {
        return Ok(RuntimeCapacityPackedModel {
            runtime_model,
            auto_placement: None,
        });
    }
    // A component-local physical shard request constrains only that execution
    // island. It must not turn the rest of the graph into a caller-placed
    // model or influence the canonical baseline used by capacity packing and
    // measured hybrid selection. The request is overlaid after those global
    // decisions converge and is admitted by the ordinary exact mount planner.
    let runtime_model = runtime_model_without_explicit_component_shards(runtime_model);
    let speculative_draft_tokens = effective_speculative_draft_tokens(args, &runtime_model)?;
    let catalog = runtime_vulkan_device_catalog(args)?;
    let available_devices = catalog.available_compute_devices();
    let profiles = catalog.available_hardware_profiles()?;
    let default_physical_index = available_devices
        .iter()
        .find(|device| device.selected_by_default)
        .or_else(|| available_devices.first())
        .map(|device| device.physical_device_index);
    let primary_capability_class = default_physical_index.and_then(|index| {
        available_devices
            .iter()
            .find(|device| device.physical_device_index == index)
            .and_then(|device| {
                profiles.iter().find(|profile| {
                    profile.hardware_identity.stable_device_id == device.physical_device_id
                })
            })
            .map(|profile| profile.capability_class.clone())
    });
    let mut eligible_devices = Vec::new();
    let mut profiles_by_physical_device = BTreeMap::new();
    for device in available_devices {
        let profile = profiles
            .iter()
            .find(|profile| profile.hardware_identity.stable_device_id == device.physical_device_id)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "physical device {:?} has no hardware-process profile",
                        device.physical_device_id,
                    ),
                )
            })?;
        eligible_devices.push((device, profile));
        profiles_by_physical_device.insert(device.physical_device_id.clone(), profile.clone());
    }
    if eligible_devices.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "automatic placement found no Vulkan compute devices",
        )
        .into());
    }
    let package_manifest = args.package_manifest.as_ref().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "automatic placement requires a compiled package manifest",
        )
    })?;
    let mut editor_devices = runtime_devices_from_compute_devices(
        RUNTIME_DEFAULT_LOGICAL_DEVICE_ID,
        None,
        available_devices,
    );
    for editor_device in &mut editor_devices {
        let Some(physical_device_id) = editor_device.physical_device_id.clone() else {
            continue;
        };
        editor_device.device_id = physical_device_id.clone();
        editor_device.runtime_device_id = Some(physical_device_id.clone());
        editor_device.hardware_profile = profiles
            .iter()
            .find(|profile| profile.hardware_identity.stable_device_id == physical_device_id)
            .cloned();
    }
    let compatibility_editor =
        RuntimeModelEditor::load_with_available_devices(package_manifest, editor_devices)?;
    let mut incompatibilities = Vec::new();
    let runtime_role_by_instance = runtime_model
        .circuit_graph
        .components
        .iter()
        .map(|component| (component.component_id.as_str(), component.runtime_role))
        .collect::<BTreeMap<_, _>>();
    let mut partially_compatible_devices = Vec::with_capacity(eligible_devices.len());
    for (device, profile) in eligible_devices {
        let mut compatible_signal_components = BTreeSet::new();
        let mut incompatible_signal_components = Vec::new();
        let mut incompatible_default_components = Vec::new();
        for instance in &runtime_model.runtime_graph.instances {
            let runtime_role = runtime_role_by_instance
                .get(instance.instance_id.as_str())
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "runtime instance {:?} has no mounted circuit component",
                            instance.instance_id,
                        ),
                    )
                })?;
            match compatibility_editor.validate_source_component_device_compatibility(
                &instance.source_component_id,
                &device.physical_device_id,
            ) {
                Ok(()) if runtime_role.is_signal_processor() => {
                    compatible_signal_components.insert(instance.instance_id.clone());
                }
                Ok(()) => {}
                Err(error) if runtime_role.is_signal_processor() => {
                    incompatible_signal_components.push(format!(
                        "{} (source {}): {error}",
                        instance.instance_id, instance.source_component_id,
                    ));
                }
                Err(error) => {
                    incompatible_default_components.push(format!(
                        "{} (source {}): {error}",
                        instance.instance_id, instance.source_component_id,
                    ));
                }
            }
        }
        if compatible_signal_components.is_empty() {
            incompatibilities.push(format!(
                "{} ({}) cannot execute any signal processor: {}",
                device.physical_device_id,
                device.device_name,
                incompatible_signal_components.join("; "),
            ));
            continue;
        }
        if !incompatible_signal_components.is_empty() {
            eprintln!(
                "nerve runtime auto-placement: device={} remains eligible for {} signal processors; incompatible components={}",
                device.physical_device_id,
                compatible_signal_components.len(),
                incompatible_signal_components.join("; "),
            );
        }
        let can_host_default_graph = incompatible_default_components.is_empty();
        if !can_host_default_graph {
            eprintln!(
                "nerve runtime auto-placement: device={} remains eligible only as an interior signal-processor target; incompatible default components={}",
                device.physical_device_id,
                incompatible_default_components.join("; "),
            );
        }
        partially_compatible_devices.push((
            device,
            profile,
            compatible_signal_components,
            can_host_default_graph,
        ));
    }
    let eligible_devices = partially_compatible_devices;
    if eligible_devices.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!(
                "automatic placement found no device compatible with any complete signal-processor placement: {}",
                incompatibilities.join("; "),
            ),
        )
        .into());
    }
    for incompatibility in incompatibilities {
        eprintln!("nerve runtime auto-placement: excluded {incompatibility}");
    }
    let calibration_started = Instant::now();
    let mut calibration_suite = VulkanRuntimePlacementCalibrationSuite::prepare(
        manifest_dir,
        &runtime_model,
        context_capacity_activations,
    )?;
    eprintln!(
        "nerve runtime placement calibration: execution_signatures={}",
        calibration_suite.targets().len(),
    );
    let total_signal_component_count = calibration_suite
        .targets()
        .iter()
        .map(|target| target.component_ids.len())
        .sum::<usize>();
    let mut calibration_evidence = BTreeMap::<String, (String, String)>::new();
    let mut placement_costs = VulkanRuntimePlacementCostModel::default();
    let mut exact_calibration_catalog =
        load_vulkan_package_placement_calibration_catalog(manifest_dir)?.unwrap_or_default();
    let mut runtime_transfer_calibration_catalog = VulkanPlacementCalibrationCatalog::default();
    let mut measured_candidates = Vec::with_capacity(eligible_devices.len());
    let mut opened_devices = BTreeMap::new();
    for (
        device_info,
        profile,
        compatible_signal_components,
        can_host_default_graph,
    ) in eligible_devices
    {
        if calibration_started.elapsed() >= VULKAN_RUNTIME_PLACEMENT_CALIBRATION_MAXIMUM_DURATION {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "runtime package-specific placement calibration exceeded its one-minute bound",
            )
            .into());
        }
        let device =
            Rc::new(catalog.open_physical_device_index(device_info.physical_device_index)?);
        let budget = device.device_local_memory_budget();
        let available_before = device.available_device_local_memory_bytes();
        let safe_capacity_bytes = usize::try_from(budget.reservable_bytes).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "Vulkan reservable device memory exceeds usize",
            )
        })?;
        eprintln!(
            "nerve runtime placement calibration: device={}, name={:?}, total_bytes={}, available_before_bytes={}, reservable_bytes={}, protected_headroom_bytes={}",
            device_info.physical_device_id,
            device_info.device_name,
            device.device_local_memory_bytes(),
            available_before,
            budget.reservable_bytes,
            budget.protected_headroom_bytes,
        );
        let calibrations = calibrate_vulkan_runtime_placement_candidate_components(
            Rc::clone(&device),
            manifest_dir,
            &profile.capability_class,
            &mut calibration_suite,
            &compatible_signal_components,
        )?;
        if can_host_default_graph {
            placement_costs
                .record_default_graph_compatibility(&device_info.physical_device_id)?;
        }
        let available_after = device.available_device_local_memory_bytes();
        for calibration in calibrations {
            if calibration.output_digest.is_empty() || calibration.state_digest.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "runtime placement calibration omitted deterministic output evidence",
                )
                .into());
            }
            let evidence = (
                calibration.output_digest.clone(),
                calibration.state_digest.clone(),
            );
            if calibration_evidence
                .get(&calibration.target.signature_id)
                .is_some_and(|expected| expected != &evidence)
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "runtime placement calibration produced different output or state across compatible devices for execution signature {}",
                        calibration.target.signature_id,
                    ),
                )
                .into());
            }
            calibration_evidence
                .entry(calibration.target.signature_id.clone())
                .or_insert(evidence);
            placement_costs.record_calibration(
                &calibration.physical_device_id,
                &calibration.target,
                calibration.measured_ns_per_activation,
            )?;
            eprintln!(
                "nerve runtime placement calibration: device={}, signature={}, representative={}.{}, occurrences={}, implementation={}, shared_prepare_ns={}, slice_plan_prepare_ns={}, slice_materialize_ns={}, session_mount_ns={}, warmup_ns={}, measured_ns={}, measured_ns_per_activation={}, dispatches={}, available_after_bytes={}",
                calibration.physical_device_id,
                calibration.target.signature_id,
                calibration.target.component_id,
                calibration.target.terminal_node_id,
                calibration.target.component_ids.len(),
                calibration.target.implementation,
                calibration.shared_prepare_ns,
                calibration.slice_plan_prepare_ns,
                calibration.slice_materialize_ns,
                calibration.session_mount_ns,
                calibration.warmup_execution_ns,
                calibration.measured_execution_ns,
                calibration.measured_ns_per_activation,
                calibration.physical_dispatch_count,
                available_after,
            );
        }
        let aggregate_execution_ns = placement_costs.normalized_device_execution_ns(
            &device_info.physical_device_id,
            total_signal_component_count,
        )?;
        measured_candidates.push((
            aggregate_execution_ns,
            device_info.selected_by_default,
            device_info.physical_device_index,
            profile.capability_class.clone(),
            VulkanRuntimePlacementCandidate {
                device_id: device_info.physical_device_id.clone(),
                safe_capacity_bytes,
            },
        ));
        opened_devices.insert(device_info.physical_device_id.clone(), device);
    }
    if calibration_started.elapsed() > VULKAN_RUNTIME_PLACEMENT_CALIBRATION_MAXIMUM_DURATION {
        return Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "runtime package-specific placement calibration exceeded its one-minute bound",
        )
        .into());
    }
    // A smaller model stays on the primary capability class, but every other
    // compatible discrete class remains available as a contiguous spill
    // target instead of being placed into an isolated, mutually exclusive
    // group.
    let candidates = rank_runtime_auto_placement_candidates_across_capability_classes(
        measured_candidates,
        primary_capability_class.as_deref(),
    );
    let transfer_byte_counts = vulkan_runtime_placement_transfer_byte_counts(&runtime_model)?;
    if candidates.len() > 1 && !transfer_byte_counts.is_empty() {
        for source in &candidates {
            for target in &candidates {
                if source.device_id == target.device_id {
                    continue;
                }
                if calibration_started.elapsed()
                    >= VULKAN_RUNTIME_PLACEMENT_CALIBRATION_MAXIMUM_DURATION
                {
                    return Err(io::Error::new(
                        io::ErrorKind::TimedOut,
                        "runtime package-specific placement calibration exceeded its one-minute bound",
                    )
                    .into());
                }
                let reports = calibrate_vulkan_runtime_placement_transfers(
                    &source.device_id,
                    opened_devices
                        .get(&source.device_id)
                        .expect("every placement candidate remains open"),
                    &target.device_id,
                    opened_devices
                        .get(&target.device_id)
                        .expect("every placement candidate remains open"),
                    &transfer_byte_counts,
                )?;
                for report in reports {
                    record_vulkan_runtime_transfer_calibration_report(
                        &mut runtime_transfer_calibration_catalog,
                        &report,
                    )?;
                    placement_costs.record_boundary_transfer_cost(
                        &report.source_device_id,
                        &report.target_device_id,
                        report.byte_count,
                        report.measured_ns,
                    )?;
                    eprintln!(
                        "nerve runtime placement calibration: transfer={}=>{}, bytes={}, route={:?}, warmup_ns={}, measured_ns={}, fixture_digest={}, output_digest={}",
                        report.source_device_id,
                        report.target_device_id,
                        report.byte_count,
                        report.route,
                        report.warmup_ns,
                        report.measured_ns,
                        report.fixture_digest,
                        report.output_digest,
                    );
                }
            }
        }
    }
    exact_calibration_catalog.merge(&runtime_transfer_calibration_catalog)?;
    if calibration_started.elapsed() > VULKAN_RUNTIME_PLACEMENT_CALIBRATION_MAXIMUM_DURATION {
        return Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "runtime package-specific placement calibration exceeded its one-minute bound",
        )
        .into());
    }
    let selected = capacity_pack_and_select_vulkan_runtime_model(
        manifest_dir,
        &runtime_model,
        &candidates,
        Some(&placement_costs),
        &profiles_by_physical_device,
        context_capacity_activations,
        speculative_draft_tokens,
        args.resource_residency_policy,
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
            speculative_draft_tokens,
            residency_policy: args
                .resource_residency_policy
                .as_runtime_name()
                .replace('-', "_"),
        },
    )
    .map_err(|error| {
        io::Error::new(
            io::ErrorKind::OutOfMemory,
            format!(
                "no heterogeneous capacity-packed placement can retain the runtime model: {error}",
            ),
        )
    })?;
    let admitted_bytes = selected
        .residency_plan
        .device_plans
        .iter()
        .map(|plan| {
            vulkan_runtime_device_capacity_admission_bytes(plan, args.resource_residency_policy)
                .map(|bytes| format!("{}={bytes}", plan.device_id))
        })
        .collect::<Result<Vec<_>, _>>()?;
    eprintln!(
        "nerve runtime auto-placement: strategy=capacity_packed_heterogeneous_fixed_point, policy={}, devices={:?}, capacity_admission_bytes={:?}",
        args.resource_residency_policy.as_runtime_name(),
        selected.selected_device_ids,
        admitted_bytes,
    );
    drop(opened_devices);
    Ok(RuntimeCapacityPackedModel {
        runtime_model: selected.runtime_model,
        auto_placement: Some(RuntimeAutoPlacementContext {
            candidates,
            costs: placement_costs,
            calibration_catalog: exact_calibration_catalog,
            exact_runtime_model: selected.exact_runtime_model,
        }),
    })
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
    validate_explicit_distributed_physical_bindings(args, &requested_bindings)?;
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
                profile.hardware_identity.stable_device_id == available_device.physical_device_id
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
        hardware_profiles.insert(logical_device_id.clone(), hardware_profile.clone());
    }

    Ok(RuntimeBoundVulkanDevices {
        devices,
        hardware_profiles,
        physical_device_indices,
        physical_device_ids,
        available_devices: available_devices.to_vec(),
    })
}

fn validate_explicit_distributed_physical_bindings(
    args: &Args,
    physical_device_index_by_logical_device: &BTreeMap<String, usize>,
) -> Result<(), io::Error> {
    for component_id in args.component_physical_strategies.keys() {
        let logical_devices = args
            .component_shard_devices
            .get(component_id)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "explicit distributed component {component_id:?} has no shard pool"
                    ),
                )
            })?;
        let physical_devices = logical_devices
            .iter()
            .map(|logical_device_id| {
                physical_device_index_by_logical_device
                    .get(logical_device_id)
                    .copied()
                    .ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!(
                                "explicit distributed component {component_id:?} has no physical binding for logical participant {logical_device_id:?}"
                            ),
                        )
                    })
            })
            .collect::<Result<BTreeSet<_>, _>>()?;
        if physical_devices.len() != logical_devices.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "explicit distributed component {component_id:?} requires one distinct physical device per logical participant; logical participants {logical_devices:?} resolve to physical device indices {physical_devices:?}"
                ),
            ));
        }
    }
    Ok(())
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
        let physical_device_index =
            resolve_runtime_vulkan_physical_device_ref_in(target, available_devices)
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
