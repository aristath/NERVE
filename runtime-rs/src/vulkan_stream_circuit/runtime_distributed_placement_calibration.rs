#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanRuntimeDistributedPlacementCalibrationReport {
    pub physical_device_ids: Vec<String>,
    pub target: VulkanRuntimePlacementCalibrationTarget,
    pub phase: String,
    pub activation_batch_width: usize,
    pub sampled_workload: bool,
    pub measured_execution_ns: u64,
    pub measured_ns_per_activation: u64,
    pub measured_windows: Vec<VulkanTargetedComponentThroughputWindow>,
    pub physical_dispatch_count: usize,
    pub shard_count: usize,
    pub output_digest: String,
    pub state_digest: String,
    pub resident_parameter_bytes_by_device: BTreeMap<String, usize>,
    pub resident_transient_bytes_by_device: BTreeMap<String, usize>,
    pub activation_routes: Vec<String>,
    pub dispatch_work: Vec<VulkanRuntimeDistributedPlacementDispatchWork>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanRuntimeDistributedPlacementDispatchWork {
    pub component_id: String,
    pub node_id: String,
    pub sampled_rows: usize,
    pub full_rows: usize,
}

struct VulkanRuntimeDistributedPlacementSession {
    physical_device_ids: Vec<String>,
    logical_device_ids: Vec<String>,
    logical_devices: BTreeMap<String, Rc<VulkanComputeDevice>>,
    owner_device_id: String,
    target: VulkanRuntimePlacementCalibrationTarget,
    phase: VulkanTargetedComponentExecutionPhase,
    placed_slice: VulkanResidentInProcessPlacedStreamProcessorDevice,
    terminal_dispatch: VulkanMountedPlacedBoundDispatch,
    prefill_runner: Option<VulkanResidentPlacedComponentBatchRunner>,
    distributed_runners: Option<VulkanDistributedDispatchRunners>,
    _distributed_activation_buffers: VulkanDistributedActivationBuffers,
    edge_synchronizations: VulkanPlacedEdgeTimelineSynchronizations,
    _distributed_parameter_buffers: VulkanDistributedParameterBuffers,
    _parameter_pool: VulkanResidentBufferPool,
    resident_parameter_bytes_by_device: BTreeMap<String, usize>,
    resident_transient_bytes_by_device: BTreeMap<String, usize>,
    activation_routes: Vec<String>,
    dispatch_work: Vec<VulkanRuntimeDistributedPlacementDispatchWork>,
    sampled_workload: bool,
}

pub fn calibrate_vulkan_runtime_distributed_placement_candidate_with_policy(
    devices: Vec<(String, Rc<VulkanComputeDevice>)>,
    manifest_dir: impl AsRef<Path>,
    runtime_model: &VulkanResidentRuntimeModel,
    target: &VulkanRuntimePlacementCalibrationTarget,
    policy: VulkanRuntimePlacementCalibrationPolicy,
) -> Result<Option<VulkanRuntimeDistributedPlacementCalibrationReport>, VulkanResidentTokenModelPackageError>
{
    calibrate_vulkan_runtime_distributed_placement_phase_candidate_with_policy(
        devices,
        manifest_dir.as_ref(),
        runtime_model,
        target,
        VulkanTargetedComponentExecutionPhase::Decode,
        policy,
    )
}

pub fn calibrate_vulkan_runtime_distributed_prefill_placement_candidate_with_policy(
    devices: Vec<(String, Rc<VulkanComputeDevice>)>,
    manifest_dir: impl AsRef<Path>,
    runtime_model: &VulkanResidentRuntimeModel,
    target: &VulkanRuntimePlacementCalibrationTarget,
    activation_batch_width: usize,
    policy: VulkanRuntimePlacementCalibrationPolicy,
) -> Result<Option<VulkanRuntimeDistributedPlacementCalibrationReport>, VulkanResidentTokenModelPackageError>
{
    if activation_batch_width == 0 {
        return distributed_calibration_error(
            "distributed prefill calibration requires a positive activation batch width",
        );
    }
    calibrate_vulkan_runtime_distributed_placement_phase_candidate_with_policy(
        devices,
        manifest_dir.as_ref(),
        runtime_model,
        target,
        VulkanTargetedComponentExecutionPhase::Prefill {
            activation_batch_width,
        },
        policy,
    )
}

fn calibrate_vulkan_runtime_distributed_placement_phase_candidate_with_policy(
    devices: Vec<(String, Rc<VulkanComputeDevice>)>,
    manifest_dir: &Path,
    runtime_model: &VulkanResidentRuntimeModel,
    target: &VulkanRuntimePlacementCalibrationTarget,
    phase: VulkanTargetedComponentExecutionPhase,
    policy: VulkanRuntimePlacementCalibrationPolicy,
) -> Result<Option<VulkanRuntimeDistributedPlacementCalibrationReport>, VulkanResidentTokenModelPackageError>
{
    if devices.is_empty() {
        return distributed_calibration_error(
            "runtime shard calibration requires at least one device",
        );
    }
    if policy.warmup_units == 0
        || policy.measured_units == 0
        || policy.maximum_duration.is_zero()
        || policy.maximum_resident_parameter_bytes == 0
    {
        return distributed_calibration_error(
            "distributed runtime placement calibration policy has invalid zero bounds",
        );
    }
    let mut physical_ids = BTreeSet::new();
    if let Some((physical_id, _)) = devices
        .iter()
        .find(|(physical_id, _)| physical_id.is_empty() || !physical_ids.insert(physical_id))
    {
        return distributed_calibration_error(format!(
            "distributed runtime placement calibration repeats or omits physical device ID {physical_id:?}",
        ));
    }

    let started = Instant::now();
    let Some(mut session) = VulkanRuntimeDistributedPlacementSession::prepare(
        devices,
        manifest_dir,
        runtime_model,
        target,
        phase,
        policy.maximum_resident_parameter_bytes,
    )?
    else {
        return Ok(None);
    };
    let execution_result = (|| {
        let activation_batch_width = phase.activation_batch_width();
        let warmup_useful_units = policy
            .warmup_units
            .checked_mul(activation_batch_width)
            .ok_or_else(|| distributed_calibration_error_value(
                "distributed calibration warmup work overflowed",
            ))?;
        let measured_useful_units = policy
            .measured_units
            .checked_mul(activation_batch_width)
            .ok_or_else(|| distributed_calibration_error_value(
                "distributed calibration measured work overflowed",
            ))?;
        let warmup = session.execute(
            warmup_useful_units,
            0,
            remaining_calibration_duration(started, policy.maximum_duration)?,
        )?;
        let measured = session.execute(
            measured_useful_units,
            0,
            remaining_calibration_duration(started, policy.maximum_duration)?,
        )?;
        if warmup_useful_units == measured_useful_units
            && (warmup.output_digest != measured.output_digest
                || warmup.state_digest != measured.state_digest)
        {
            return distributed_calibration_error(
                "distributed runtime placement calibration changed deterministic component output or state",
            );
        }
        let measured_ns_per_activation = measured
            .execution_ns
            .saturating_add((measured_useful_units / 2) as u64)
            / measured_useful_units as u64;
        let remap_device_bytes = |bytes: &BTreeMap<String, usize>| {
            session
                .logical_device_ids
                .iter()
                .zip(&session.physical_device_ids)
                .map(|(logical_id, physical_id)| {
                    (physical_id.clone(), bytes.get(logical_id).copied().unwrap_or(0))
                })
                .collect::<BTreeMap<_, _>>()
        };
        Ok(VulkanRuntimeDistributedPlacementCalibrationReport {
            physical_device_ids: session.physical_device_ids.clone(),
            target: session.target.clone(),
            phase: measured.phase,
            activation_batch_width: measured.activation_batch_width,
            sampled_workload: session.sampled_workload,
            measured_execution_ns: measured.execution_ns,
            measured_ns_per_activation,
            measured_windows: measured.windows,
            physical_dispatch_count: measured.physical_dispatch_count,
            shard_count: measured.shard_count,
            output_digest: measured.output_digest,
            state_digest: measured.state_digest,
            resident_parameter_bytes_by_device: remap_device_bytes(
                &session.resident_parameter_bytes_by_device,
            ),
            resident_transient_bytes_by_device: remap_device_bytes(
                &session.resident_transient_bytes_by_device,
            ),
            activation_routes: session.activation_routes.clone(),
            dispatch_work: session.dispatch_work.clone(),
        })
    })();
    let cleanup_result = session.cleanup();
    match (execution_result, cleanup_result) {
        (Ok(report), Ok(())) => Ok(Some(report)),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(cleanup_error)) => Err(cleanup_error),
        (Err(error), Err(cleanup_error)) => Err(distributed_calibration_error_value(format!(
            "{error}; cleanup also failed: {cleanup_error}",
        ))),
    }
}

struct VulkanRuntimeDistributedPlacementExecution {
    phase: String,
    activation_batch_width: usize,
    execution_ns: u64,
    windows: Vec<VulkanTargetedComponentThroughputWindow>,
    physical_dispatch_count: usize,
    shard_count: usize,
    output_digest: String,
    state_digest: String,
}

impl VulkanRuntimeDistributedPlacementSession {
    fn prepare(
        devices: Vec<(String, Rc<VulkanComputeDevice>)>,
        manifest_dir: &Path,
        runtime_model: &VulkanResidentRuntimeModel,
        target: &VulkanRuntimePlacementCalibrationTarget,
        phase: VulkanTargetedComponentExecutionPhase,
        maximum_resident_parameter_bytes: usize,
    ) -> Result<Option<Self>, VulkanResidentTokenModelPackageError> {
        let logical_device_ids = (0..devices.len())
            .map(|index| format!("calibration:shard:{index}"))
            .collect::<Vec<_>>();
        let planning_device_ids = if logical_device_ids.len() == 1 {
            vec![
                logical_device_ids[0].clone(),
                "calibration:planning_shard:1".to_string(),
            ]
        } else {
            logical_device_ids.clone()
        };
        let owner_device_id = logical_device_ids[0].clone();
        let logical_devices = logical_device_ids
            .iter()
            .cloned()
            .zip(devices.iter().map(|(_, device)| Rc::clone(device)))
            .collect::<BTreeMap<_, _>>();
        let owner_device = logical_devices
            .get(&owner_device_id)
            .expect("distributed calibration owner was inserted");
        let mut placed_model = vulkan_runtime_model_with_component_placement(
            runtime_model,
            "calibration:unmounted",
            &BTreeMap::from([(target.component_id.clone(), owner_device_id.clone())]),
        )
        .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        placed_model = placed_model
            .with_component_shard_devices(&target.component_id, planning_device_ids.clone())?;
        let capacity = placed_model
            .package
            .max_context_activations
            .max(1)
            .min(VULKAN_RUNTIME_PLACEMENT_CALIBRATION_MAXIMUM_STATE_ACTIVATIONS);
        let tensor_index = Arc::new(placed_model.load_runtime_tensor_index(manifest_dir)?);
        let contract = Arc::new(
            instantiate_runtime_resource_contract(&placed_model)
                .map_err(|error| distributed_calibration_error_value(error.to_string()))?,
        );
        let residency_plan = plan_vulkan_runtime_residency_with_contract(
            manifest_dir,
            &placed_model,
            &tensor_index,
            capacity,
            0,
            ResourceResidencyPolicy::DemandRetained,
            &contract,
        )
        .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let targeted_plan = VulkanResidentTargetedModelPackageDeviceSlicePlan::prepare(
            owner_device,
            manifest_dir,
            &placed_model,
            &target.component_id,
            &owner_device_id,
            capacity,
            Arc::clone(&tensor_index),
            Arc::clone(&contract),
            residency_plan,
        )?;
        let loaded_manifest = resident_package_loaded_kernel_manifest_for_slice_plans(
            std::slice::from_ref(&targeted_plan.slice_plan),
        )?;
        let artifact_manifest = VulkanReusableKernelArtifactManifest::new(
            loaded_manifest
                .artifacts
                .iter()
                .map(|artifact| artifact.artifact.clone())
                .collect(),
        );
        let graph = placed_model.executable_circuit_graph()?;
        let (_, placement_plan, _) = plan_resident_package_placed_stream_circuit_with_tensor_index(
            &owner_device_id,
            &placed_model.placement,
            &graph,
            manifest_dir,
            &tensor_index,
            placed_model.package.activation_element_bytes,
        )?;
        let alignment = logical_devices
            .values()
            .map(|device| device.min_storage_buffer_offset_alignment())
            .max()
            .unwrap_or(1);
        let full_distributed_execution_plan = VulkanDistributedExecutionPlan::from_prepared_plans(
            &[(
                owner_device_id.as_str(),
                &targeted_plan.slice_plan.prepared_plan,
            )],
            &tensor_index,
            &artifact_manifest,
            &BTreeMap::from([(target.component_id.clone(), planning_device_ids)]),
            &placement_plan.edges,
            alignment,
        )
        .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        if full_distributed_execution_plan.dispatches.is_empty() {
            return Ok(None);
        }
        if logical_device_ids.len() > 1
            && full_distributed_execution_plan
                .dispatches
                .iter()
                .any(|dispatch| dispatch.shards.len() < logical_device_ids.len())
        {
            return Ok(None);
        }
        let exclusion_plan =
            VulkanDistributedParameterExclusionPlan::from_execution_and_prepared_plans(
                &full_distributed_execution_plan,
                &[(
                    owner_device_id.as_str(),
                    &targeted_plan.slice_plan.prepared_plan,
                )],
                &tensor_index,
            )
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let owner_exclusions = exclusion_plan.tensors_for_device(&owner_device_id);
        let owner_parameter_plan =
            VulkanPermanentParameterBufferPlan::from_placed_resident_plan_excluding_tensors(
                &targeted_plan.slice_plan.placed_plan.placed_resident_plan,
                &owner_exclusions,
            )
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let owner_static_bytes = owner_parameter_plan.total_byte_capacity.ok_or_else(|| {
            distributed_calibration_error_value(
                "distributed calibration owner has unresolved permanent parameter bytes",
            )
        })?;
        let Some(distributed_parameter_budget) =
            maximum_resident_parameter_bytes.checked_sub(owner_static_bytes)
        else {
            return Ok(None);
        };
        let Some(distributed_execution_plan) = full_distributed_execution_plan
            .sampled_for_parameter_budget(
                &tensor_index,
                &logical_device_ids,
                distributed_parameter_budget,
            )
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?
        else {
            return Ok(None);
        };
        let dispatch_work = full_distributed_execution_plan
            .dispatches
            .iter()
            .zip(&distributed_execution_plan.dispatches)
            .map(|(full, sampled)| {
                if full.component_id != sampled.component_id
                    || full.node_id != sampled.node_id
                    || full.output_rows == 0
                {
                    return distributed_calibration_error(
                        "sampled distributed dispatch order or geometry differs from its full plan",
                    );
                }
                let sampled_rows = sampled.shards.iter().try_fold(
                    0usize,
                    |total, shard| {
                        total.checked_add(shard.row_count).ok_or_else(|| {
                            distributed_calibration_error_value(
                                "sampled distributed work rows overflowed",
                            )
                        })
                    },
                )?;
                if sampled_rows == 0 || sampled_rows > full.output_rows {
                    return distributed_calibration_error(
                        "sampled distributed work rows fall outside the full dispatch geometry",
                    );
                }
                Ok(VulkanRuntimeDistributedPlacementDispatchWork {
                    component_id: full.component_id.clone(),
                    node_id: full.node_id.clone(),
                    sampled_rows,
                    full_rows: full.output_rows,
                })
            })
            .collect::<Result<Vec<_>, VulkanResidentTokenModelPackageError>>()?;
        let sampled_workload = distributed_execution_plan.dispatches.iter().any(|dispatch| {
            dispatch
                .shards
                .iter()
                .map(|shard| shard.row_count)
                .sum::<usize>()
                < dispatch.output_rows
        });
        let used_devices = distributed_execution_plan
            .dispatches
            .iter()
            .flat_map(|dispatch| dispatch.shards.iter().map(|shard| shard.device_id.as_str()))
            .collect::<BTreeSet<_>>();
        if logical_device_ids
            .iter()
            .any(|device_id| !used_devices.contains(device_id.as_str()))
        {
            return Ok(None);
        }
        let activation_plan =
            VulkanDistributedActivationBufferPlan::from_execution_plan(&distributed_execution_plan)
                .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let parameter_plan =
            VulkanDistributedParameterAllocationPlan::from_sampled_execution_plan(
                &distributed_execution_plan,
                &tensor_index,
            )
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let mut resident_parameter_bytes_by_device = logical_device_ids
            .iter()
            .map(|device_id| (device_id.clone(), 0usize))
            .collect::<BTreeMap<_, _>>();
        for allocation in &parameter_plan.allocations {
            let total = resident_parameter_bytes_by_device
                .get_mut(&allocation.device_id)
                .ok_or_else(|| {
                    distributed_calibration_error_value(format!(
                        "distributed parameter allocation references unknown device {:?}",
                        allocation.device_id,
                    ))
                })?;
            *total = total.checked_add(allocation.byte_count).ok_or_else(|| {
                distributed_calibration_error_value(
                    "distributed parameter byte accounting overflowed",
                )
            })?;
        }
        *resident_parameter_bytes_by_device
            .get_mut(&owner_device_id)
            .expect("distributed calibration owner was inserted") =
            resident_parameter_bytes_by_device[&owner_device_id]
                .checked_add(owner_static_bytes)
                .ok_or_else(|| {
                    distributed_calibration_error_value(
                        "distributed owner parameter byte accounting overflowed",
                    )
                })?;
        let total_resident_parameter_bytes = resident_parameter_bytes_by_device
            .values()
            .try_fold(0usize, |total, bytes| total.checked_add(*bytes))
            .ok_or_else(|| distributed_calibration_error_value(
                "distributed calibration total parameter bytes overflowed",
            ))?;
        if total_resident_parameter_bytes > maximum_resident_parameter_bytes {
            return Ok(None);
        }

        let parameter_pool = VulkanResidentBufferPool::default();
        for (device_id, device) in &logical_devices {
            parameter_pool
                .register_device(device_id, Rc::clone(device))
                .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        }
        let distributed_parameter_buffers =
            VulkanDistributedParameterBuffers::allocate_and_load_from_pool(
                &parameter_plan,
                &tensor_index,
                &parameter_pool,
            )
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let targeted = targeted_plan.materialize_excluding_tensors(
            owner_device,
            manifest_dir,
            &parameter_pool,
            &owner_exclusions,
        )?;
        let distributed_activation_buffers =
            VulkanDistributedActivationBuffers::allocate(&activation_plan, |device_id| {
                logical_devices
                    .get(device_id)
                    .map(Rc::as_ref)
                    .ok_or_else(|| format!("missing distributed calibration device {device_id:?}"))
            })
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let distributed_runners = (phase == VulkanTargetedComponentExecutionPhase::Decode)
            .then(|| {
                VulkanDistributedDispatchRunners::create(
                    &distributed_execution_plan,
                    &distributed_parameter_buffers,
                    &distributed_activation_buffers,
                    &loaded_manifest,
                    |device_id| {
                        logical_devices
                            .get(device_id)
                            .map(Rc::as_ref)
                            .ok_or_else(|| {
                                format!(
                                    "missing distributed calibration device {device_id:?}"
                                )
                            })
                    },
                )
                .map_err(|error| distributed_calibration_error_value(error.to_string()))
            })
            .transpose()?;
        let activation_overrides =
            distributed_activation_buffers.activation_overrides_for_owner_device(&owner_device_id);
        let boundary_overrides =
            distributed_activation_buffers.boundary_overrides_for_owner_device(&owner_device_id);
        let (local_edge_overrides, endpoint_overrides) = distributed_calibration_edge_overrides(
            &targeted.slice,
            &distributed_activation_buffers,
            &owner_device_id,
        );
        let mounted = targeted
            .slice
            .create_mounted_stream_circuit_with_all_buffer_overrides(
                owner_device,
                &activation_overrides,
                &local_edge_overrides,
                &endpoint_overrides,
                &boundary_overrides,
                None,
            )?;
        mounted
            .buffers
            .zero_state_buffers()
            .and_then(|_| mounted.buffers.apply_clone_state_policies())
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let reusable_manifest = resident_package_reusable_kernel_manifest(&mounted.placed_plan);
        let physical_execution_islands = distributed_execution_plan
            .execution_islands
            .iter()
            .map(VulkanPhysicalExecutionIslandPlan::dispatch_indices)
            .collect::<Vec<_>>();
        let replaced_parameter_dispatches = physical_execution_islands
            .iter()
            .flatten()
            .copied()
            .collect::<BTreeSet<_>>();
        let mounted_bound = mounted
            .mounted_placed_bound_dispatch_plan_with_replaced_parameter_dispatches(
                &reusable_manifest,
                &replaced_parameter_dispatches,
            )
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let terminal_dispatch = mounted_bound
            .dispatch(&target.component_id, &target.terminal_node_id)
            .cloned()
            .ok_or_else(|| {
                distributed_calibration_error_value(format!(
                    "distributed calibration has no terminal dispatch {}.{}",
                    target.component_id, target.terminal_node_id,
                ))
            })?;
        let tick_plan = distributed_calibration_dispatch_tick_plan(&mounted_bound);
        let execution_plan =
            VulkanMountedPlacedResidentStreamTickExecutionPlan::
                from_tick_plan_with_physical_execution_islands_and_demand(
                    owner_device,
                    &mounted,
                    &mounted_bound,
                    targeted.slice.loaded_manifest(),
                    tick_plan,
                    &physical_execution_islands,
                    targeted.demand_context.as_ref().map(|_| {
                        targeted.slice.physical_residency_schedule()
                    }),
                    targeted.demand_context.as_ref(),
                    None,
                )
                .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let activation_routes = distributed_activation_buffers
            .allocations
            .iter()
            .map(|allocation| format!("{:?}", allocation.route).to_lowercase())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let owner_transient = mounted
            .buffers
            .total_byte_capacity
            .saturating_add(mounted.boundary_io.total_byte_capacity)
            .saturating_add(mounted.edge_io.total_byte_capacity)
            .saturating_add(mounted.stream_control_buffer.byte_capacity());
        let mut resident_transient_bytes_by_device = activation_plan
            .allocations
            .iter()
            .flat_map(|allocation| {
                allocation
                    .device_ids
                    .iter()
                    .map(move |device_id| (device_id.clone(), allocation.byte_capacity))
            })
            .fold(
                BTreeMap::<String, usize>::new(),
                |mut totals, (device_id, bytes)| {
                    let total = totals.entry(device_id).or_default();
                    *total = total.saturating_add(bytes);
                    totals
                },
            );
        *resident_transient_bytes_by_device
            .entry(owner_device_id.clone())
            .or_default() = resident_transient_bytes_by_device
            .get(&owner_device_id)
            .copied()
            .unwrap_or(0)
            .saturating_add(owner_transient);

        let package_slice = Arc::new(targeted.slice);
        let placed_slice = VulkanResidentInProcessPlacedStreamProcessorDevice {
            device_id: owner_device_id.clone(),
            hosted_component_count: package_slice.hosted_component_count,
            incoming_edge_count: package_slice.incoming_edge_count,
            outgoing_edge_count: package_slice.outgoing_edge_count,
            dispatch_count: mounted_bound.dispatches.len(),
            package_slice,
            mounted,
            mounted_bound,
            resident_execution_plan: execution_plan,
            demand_residency_context: targeted.demand_context,
        };
        if let VulkanTargetedComponentExecutionPhase::Prefill {
            activation_batch_width,
        } = phase
        {
            let all_distributed_dispatches_supported = distributed_execution_plan
                .dispatches
                .iter()
                .all(|dispatch| {
                    selected_distributed_component_batch_artifact(
                        &logical_devices,
                        &placed_slice.package_slice,
                        dispatch,
                        VulkanComponentBatchExecutionMode::CausalSequence,
                        activation_batch_width,
                    )
                    .is_some()
                });
            if !all_distributed_dispatches_supported {
                return Ok(None);
            }
        }
        let prefill_runner = match phase {
            VulkanTargetedComponentExecutionPhase::Decode => None,
            VulkanTargetedComponentExecutionPhase::Prefill {
                activation_batch_width,
            } => {
                let runtime_execution_identity = canonical_runtime_execution_identity(
                    &placed_model,
                    capacity,
                    false,
                    ResourceResidencyPolicy::DemandRetained,
                )?;
                let quantum_calibrators = BTreeMap::from([(
                    owner_device_id.clone(),
                    Rc::new(RefCell::new(RuntimeExecutionQuantumCalibrator::default())),
                )]);
                Some(
                    VulkanResidentPlacedComponentBatchRunner::new(
                        &logical_devices,
                        std::slice::from_ref(&placed_slice),
                        &runtime_execution_identity,
                        &quantum_calibrators,
                        activation_batch_width,
                        VulkanComponentBatchExecutionMode::CausalSequence,
                        &BTreeMap::new(),
                        true,
                        &distributed_execution_plan,
                        &distributed_parameter_buffers,
                    )
                    .map_err(|error| {
                        distributed_calibration_error_value(error.to_string())
                    })?,
                )
            }
        };

        Ok(Some(Self {
            physical_device_ids: devices
                .into_iter()
                .map(|(physical_id, _)| physical_id)
                .collect(),
            logical_device_ids,
            logical_devices,
            owner_device_id,
            target: target.clone(),
            phase,
            placed_slice,
            terminal_dispatch,
            prefill_runner,
            distributed_runners,
            _distributed_activation_buffers: distributed_activation_buffers,
            edge_synchronizations: VulkanPlacedEdgeTimelineSynchronizations::default(),
            _distributed_parameter_buffers: distributed_parameter_buffers,
            _parameter_pool: parameter_pool,
            resident_parameter_bytes_by_device,
            resident_transient_bytes_by_device,
            activation_routes,
            dispatch_work,
            sampled_workload,
        }))
    }

    fn execute(
        &mut self,
        useful_units: usize,
        seed: u32,
        maximum_duration: Duration,
    ) -> Result<VulkanRuntimeDistributedPlacementExecution, VulkanResidentTokenModelPackageError>
    {
        if useful_units == 0 || maximum_duration.is_zero() {
            return distributed_calibration_error(
                "distributed calibration execution requires positive work and duration bounds",
            );
        }
        match self.phase {
            VulkanTargetedComponentExecutionPhase::Decode => {
                self.execute_decode(useful_units, seed, maximum_duration)
            }
            VulkanTargetedComponentExecutionPhase::Prefill {
                activation_batch_width,
            } => self.execute_prefill(
                useful_units,
                activation_batch_width,
                seed,
                maximum_duration,
            ),
        }
    }

    fn execute_decode(
        &mut self,
        useful_units: usize,
        seed: u32,
        maximum_duration: Duration,
    ) -> Result<VulkanRuntimeDistributedPlacementExecution, VulkanResidentTokenModelPackageError>
    {
        distributed_calibration_write_fixture(
            &self.placed_slice.mounted,
            &self.placed_slice.mounted_bound,
            seed,
        )?;
        let started = Instant::now();
        let schedule = VulkanMountedPlacedResidentInProcessSchedule::from_tick_plans(&[
            &self.placed_slice.resident_execution_plan.tick_plan,
        ])
        .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let distributed_runners = self.distributed_runners.as_ref().ok_or_else(|| {
            distributed_calibration_error_value(
                "distributed decode calibration has no dispatch runners",
            )
        })?;
        let mut transport = VulkanInProcessPlacedEdgeTransport::new();
        let mut windows = Vec::with_capacity(useful_units);
        for index in 0..useful_units {
            if started.elapsed() >= maximum_duration {
                return distributed_calibration_error(
                    "distributed calibration exceeded its configured duration",
                );
            }
            let unit_started = Instant::now();
            let mut slices = [
                VulkanMountedPlacedResidentInProcessStreamTickSlice::new_with_dispatch_extensions(
                    self.logical_devices[&self.owner_device_id].as_ref(),
                    &self.placed_slice.mounted,
                    &self.placed_slice.resident_execution_plan,
                    VulkanMountedPlacedResidentStreamTickDispatchExtensions::default(),
                    0,
                ),
            ];
            run_mounted_placed_resident_stream_tick_slices_in_process_with_schedule_and_distributed(
                &mut slices,
                &mut transport,
                &schedule,
                Some(distributed_runners),
                Some(&self.edge_synchronizations),
                VulkanPlacedSubmissionContext {
                    participant_devices: Some(&self.logical_devices),
                    ..VulkanPlacedSubmissionContext::SYNCHRONOUS
                },
            )
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
            windows.push(VulkanTargetedComponentThroughputWindow {
                index,
                start_unit: index,
                end_unit: index + 1,
                duration_ns: elapsed_nanoseconds(unit_started).max(1),
            });
        }
        let execution_ns = windows
            .iter()
            .fold(0u64, |total, window| total.saturating_add(window.duration_ns));
        let output_digest = distributed_calibration_output_digest(
            &self.placed_slice.mounted,
            &self.terminal_dispatch,
        )?;
        let state_digest = distributed_calibration_state_digest(&self.placed_slice.mounted)?;
        let dispatches_per_unit = self
            .placed_slice
            .resident_execution_plan
            .dispatch_count
            .saturating_add(distributed_runners.shard_count);
        Ok(VulkanRuntimeDistributedPlacementExecution {
            phase: "decode".to_string(),
            activation_batch_width: 1,
            execution_ns: execution_ns.max(1),
            windows,
            physical_dispatch_count: useful_units.saturating_mul(dispatches_per_unit),
            shard_count: distributed_runners.shard_count,
            output_digest,
            state_digest,
        })
    }

    fn execute_prefill(
        &mut self,
        useful_units: usize,
        activation_batch_width: usize,
        seed: u32,
        maximum_duration: Duration,
    ) -> Result<VulkanRuntimeDistributedPlacementExecution, VulkanResidentTokenModelPackageError>
    {
        if !useful_units.is_multiple_of(activation_batch_width) {
            return distributed_calibration_error(format!(
                "distributed prefill work {useful_units} is not divisible by width {activation_batch_width}",
            ));
        }
        let runner = self.prefill_runner.as_ref().ok_or_else(|| {
            distributed_calibration_error_value(
                "distributed prefill calibration has no causal batch runner",
            )
        })?;
        distributed_calibration_write_prefill_fixture(
            runner,
            &self.placed_slice,
            activation_batch_width,
            seed,
        )?;
        let start_stream_tick = u64::try_from(
            self.placed_slice
                .package_slice
                .dynamic_state_capacity_activations
                .checked_sub(activation_batch_width)
                .ok_or_else(|| distributed_calibration_error_value(format!(
                    "distributed prefill width {activation_batch_width} exceeds dynamic-state capacity {}",
                    self.placed_slice.package_slice.dynamic_state_capacity_activations,
                )))?,
        )
        .map_err(|_| distributed_calibration_error_value(
            "distributed prefill start stream tick exceeds u64",
        ))?;
        let token_ids = vec![0u32; activation_batch_width];
        let repetitions = useful_units / activation_batch_width;
        let started = Instant::now();
        let mut windows = Vec::with_capacity(repetitions);
        for index in 0..repetitions {
            if started.elapsed() >= maximum_duration {
                return distributed_calibration_error(
                    "distributed prefill calibration exceeded its configured duration",
                );
            }
            let unit_started = Instant::now();
            runner
                .run_causal_sequence(
                    &self.logical_devices,
                    0,
                    &self.owner_device_id,
                    &self.placed_slice.mounted,
                    &token_ids,
                    start_stream_tick,
                    u32::try_from(
                        self.placed_slice
                            .package_slice
                            .dynamic_state_capacity_activations,
                    )
                    .map_err(|_| distributed_calibration_error_value(
                        "distributed prefill state capacity exceeds u32",
                    ))?,
                    VulkanComponentBatchCompletionMode::Blocking,
                )
                .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
            windows.push(VulkanTargetedComponentThroughputWindow {
                index,
                start_unit: index.saturating_mul(activation_batch_width),
                end_unit: (index + 1).saturating_mul(activation_batch_width),
                duration_ns: elapsed_nanoseconds(unit_started).max(1),
            });
        }
        runner
            .commit_causal_state_prefix(activation_batch_width)
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let execution_ns = windows
            .iter()
            .fold(0u64, |total, window| total.saturating_add(window.duration_ns));
        let output_digest = distributed_calibration_prefill_output_digest(
            runner,
            &self.placed_slice,
            &self.terminal_dispatch,
            activation_batch_width,
        )?;
        let state_digest = distributed_calibration_state_digest(&self.placed_slice.mounted)?;
        let local_dispatch_count = runner
            .slices
            .iter()
            .map(|slice| slice.steps.len())
            .sum::<usize>();
        let distributed_dispatch_count = runner
            .distributed_dispatches
            .dispatches
            .iter()
            .flat_map(|dispatch| &dispatch.shards)
            .map(|shard| shard.dispatches.len())
            .sum::<usize>();
        let shard_count = runner
            .distributed_dispatches
            .dispatches
            .iter()
            .map(|dispatch| dispatch.shards.len())
            .sum::<usize>();
        Ok(VulkanRuntimeDistributedPlacementExecution {
            phase: "prefill".to_string(),
            activation_batch_width,
            execution_ns: execution_ns.max(1),
            windows,
            physical_dispatch_count: repetitions.saturating_mul(
                local_dispatch_count.saturating_add(distributed_dispatch_count),
            ),
            shard_count,
            output_digest,
            state_digest,
        })
    }

    fn cleanup(self) -> Result<(), VulkanResidentTokenModelPackageError> {
        let Self {
            logical_devices,
            logical_device_ids: _,
            owner_device_id: _,
            physical_device_ids: _,
            target: _,
            phase: _,
            placed_slice,
            terminal_dispatch,
            prefill_runner,
            distributed_runners,
            _distributed_activation_buffers: distributed_activation_buffers,
            edge_synchronizations,
            _distributed_parameter_buffers: distributed_parameter_buffers,
            _parameter_pool: parameter_pool,
            resident_parameter_bytes_by_device: _,
            resident_transient_bytes_by_device: _,
            activation_routes: _,
            dispatch_work: _,
            sampled_workload: _,
        } = self;
        let mut cleanup_errors = Vec::new();
        for device in logical_devices.values() {
            if let Err(error) = device.quiesce() {
                cleanup_errors.push(error.to_string());
            }
        }
        drop(prefill_runner);
        drop(distributed_runners);
        drop(edge_synchronizations);
        drop(terminal_dispatch);
        drop(placed_slice);
        drop(distributed_activation_buffers);
        drop(distributed_parameter_buffers);
        for device_id in logical_devices.keys() {
            if let Err(error) = parameter_pool.release_device(device_id) {
                cleanup_errors.push(error.to_string());
            }
        }
        let stats = parameter_pool.stats();
        if parameter_pool.registered_device_count() != 0
            || stats.resident_allocation_count != 0
            || stats.resident_buffer_count != 0
            || stats.resident_bytes != 0
        {
            cleanup_errors.push(format!(
                "distributed calibration parameter pool retained {} devices, {} allocations, {} buffers, and {} bytes",
                parameter_pool.registered_device_count(),
                stats.resident_allocation_count,
                stats.resident_buffer_count,
                stats.resident_bytes,
            ));
        }
        for device in logical_devices.values() {
            if let Err(error) = device.quiesce() {
                cleanup_errors.push(error.to_string());
            }
            match device.device_local_memory_accounting() {
                Ok(accounting)
                    if accounting.tracked_allocation_bytes == 0
                        && accounting.pending_reservation_bytes == 0 => {}
                Ok(accounting) => cleanup_errors.push(format!(
                    "device {:?} retained {} tracked and {} pending bytes",
                    device.physical_device_id(),
                    accounting.tracked_allocation_bytes,
                    accounting.pending_reservation_bytes,
                )),
                Err(error) => cleanup_errors.push(error.to_string()),
            }
        }
        if cleanup_errors.is_empty() {
            Ok(())
        } else {
            distributed_calibration_error(format!(
                "distributed runtime placement calibration cleanup failed: {}",
                cleanup_errors.join("; "),
            ))
        }
    }
}

fn distributed_calibration_dispatch_tick_plan(
    mounted_bound: &VulkanMountedPlacedBoundDispatchPlan,
) -> VulkanMountedPlacedStreamTickPlan {
    let base = VulkanMountedPlacedStreamTickPlan::from_mounted_bound_plan(mounted_bound);
    let stages = base
        .stages
        .into_iter()
        .filter_map(|stage| match stage {
            VulkanMountedPlacedStreamTickStage::Dispatch { dispatch, .. } => Some(dispatch),
            _ => None,
        })
        .enumerate()
        .map(
            |(stage_index, dispatch)| VulkanMountedPlacedStreamTickStage::Dispatch {
                stage_index,
                dispatch,
            },
        )
        .collect::<Vec<_>>();
    VulkanMountedPlacedStreamTickPlan {
        backend_id: base.backend_id,
        device_id: base.device_id,
        stage_count: stages.len(),
        dispatch_stage_count: stages.len(),
        stages,
        receive_stage_count: 0,
        publish_stage_count: 0,
        local_edge_read_count: base.local_edge_read_count,
        local_edge_write_count: base.local_edge_write_count,
        incoming_edge_read_count: base.incoming_edge_read_count,
        outgoing_edge_write_count: base.outgoing_edge_write_count,
        model_input_read_count: base.model_input_read_count,
        model_output_write_count: base.model_output_write_count,
        can_execute: true,
    }
}

fn distributed_calibration_edge_overrides(
    slice: &VulkanResidentModelPackageDeviceSlice,
    activations: &VulkanDistributedActivationBuffers,
    owner_device_id: &str,
) -> (
    Vec<VulkanPlacedLocalEdgeBufferOverride>,
    Vec<VulkanPlacedEdgeEndpointBufferOverride>,
) {
    let edge_plan =
        VulkanPlacedEdgeIoPlan::from_placed_resident_plan(&slice.placed_plan.placed_resident_plan)
            .expect("validated targeted slice has a valid edge plan");
    let mut local = Vec::new();
    let mut endpoints = Vec::new();
    for allocation in &activations.allocations {
        let VulkanDistributedActivationStorage::Edge { edge_index, .. } =
            allocation.planned.storage
        else {
            continue;
        };
        let Some(buffer) = allocation.device_buffers.get(owner_device_id).cloned() else {
            continue;
        };
        if edge_plan
            .local_edges
            .iter()
            .any(|edge| edge.edge_index == edge_index)
        {
            local.push(VulkanPlacedLocalEdgeBufferOverride {
                edge_index,
                buffer: Arc::clone(&buffer),
            });
        }
        for endpoint in edge_plan
            .endpoints
            .iter()
            .filter(|endpoint| endpoint.edge_index == edge_index)
        {
            endpoints.push(VulkanPlacedEdgeEndpointBufferOverride {
                direction: endpoint.direction,
                edge_index,
                buffer: Arc::clone(&buffer),
            });
        }
    }
    (local, endpoints)
}

fn distributed_calibration_write_fixture(
    mounted: &VulkanMountedPlacedStreamCircuit,
    plan: &VulkanMountedPlacedBoundDispatchPlan,
    seed: u32,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    mounted
        .buffers
        .zero_state_buffers()
        .and_then(|_| mounted.buffers.apply_clone_state_policies())
        .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
    let produced_signals = plan
        .dispatches
        .iter()
        .flat_map(|dispatch| dispatch.descriptors.iter())
        .filter(|descriptor| descriptor.usage == VulkanKernelDescriptorUsage::OutputSignal)
        .map(|descriptor| descriptor.name.as_str())
        .collect::<BTreeSet<_>>();
    let mut initialized_inputs = BTreeSet::new();
    let mut cleared_outputs = BTreeSet::new();
    for dispatch in &plan.dispatches {
        for descriptor in &dispatch.descriptors {
            let binding = mounted
                .resident_kernel_buffer_binding(dispatch, descriptor)
                .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
            match descriptor.usage {
                VulkanKernelDescriptorUsage::InputSignal
                    if !produced_signals.contains(descriptor.name.as_str())
                        && targeted_signal_accepts_fixture_mutation(descriptor)
                        && initialized_inputs.insert(descriptor.name.clone()) =>
                {
                    binding
                        .buffer
                        .write_bytes_at(
                            binding.byte_offset,
                            &targeted_fixture_bytes(binding.byte_len, seed, descriptor.binding),
                        )
                        .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
                }
                VulkanKernelDescriptorUsage::OutputSignal
                    if targeted_signal_accepts_fixture_mutation(descriptor)
                        && cleared_outputs.insert(descriptor.name.clone()) =>
                {
                    binding
                        .buffer
                        .write_bytes_at(binding.byte_offset, &vec![0; binding.byte_len])
                        .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
                }
                _ => {}
            }
        }
    }
    Ok(())
}

fn distributed_calibration_write_prefill_fixture(
    runner: &VulkanResidentPlacedComponentBatchRunner,
    placed_slice: &VulkanResidentInProcessPlacedStreamProcessorDevice,
    activation_batch_width: usize,
    seed: u32,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    placed_slice
        .mounted
        .buffers
        .zero_state_buffers()
        .and_then(|_| placed_slice.mounted.buffers.apply_clone_state_policies())
        .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
    let batch_slice = runner
        .slice(0)
        .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
    for signal in &batch_slice.signal_buffers {
        signal
            .buffer
            .write_bytes(&vec![0; signal.buffer.byte_capacity()])
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
    }
    let produced_signals = placed_slice
        .mounted_bound
        .dispatches
        .iter()
        .flat_map(|dispatch| dispatch.descriptors.iter())
        .filter(|descriptor| descriptor.usage == VulkanKernelDescriptorUsage::OutputSignal)
        .filter_map(|descriptor| {
            component_batch_signal_target_with_mounted(&placed_slice.mounted, descriptor)
                .transpose()
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| distributed_calibration_error_value(error.to_string()))?
        .into_iter()
        .map(|(key, _)| key)
        .collect::<BTreeSet<_>>();
    let mut initialized_inputs = BTreeSet::new();
    for dispatch in &placed_slice.mounted_bound.dispatches {
        for descriptor in dispatch
            .descriptors
            .iter()
            .filter(|descriptor| descriptor.usage == VulkanKernelDescriptorUsage::InputSignal)
        {
            let Some((key, frame_byte_capacity)) =
                component_batch_signal_target_with_mounted(&placed_slice.mounted, descriptor)
                    .map_err(|error| distributed_calibration_error_value(error.to_string()))?
            else {
                if targeted_signal_accepts_fixture_mutation(descriptor) {
                    return distributed_calibration_error(format!(
                        "distributed prefill input {} has no signal buffer",
                        descriptor.name,
                    ));
                }
                continue;
            };
            if produced_signals.contains(&key) || !initialized_inputs.insert(key.clone()) {
                continue;
            }
            let signal = batch_slice
                .signal_buffer(&key)
                .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
            let byte_count = frame_byte_capacity
                .checked_mul(activation_batch_width)
                .ok_or_else(|| distributed_calibration_error_value(
                    "distributed prefill fixture size overflowed",
                ))?;
            signal
                .buffer
                .write_bytes(&targeted_fixture_bytes(
                    byte_count,
                    seed.wrapping_add(
                        u32::try_from(dispatch.dispatch_index).unwrap_or(u32::MAX),
                    ),
                    descriptor.binding,
                ))
                .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        }
    }
    Ok(())
}

fn distributed_calibration_prefill_output_digest(
    runner: &VulkanResidentPlacedComponentBatchRunner,
    placed_slice: &VulkanResidentInProcessPlacedStreamProcessorDevice,
    terminal: &VulkanMountedPlacedBoundDispatch,
    activation_batch_width: usize,
) -> Result<String, VulkanResidentTokenModelPackageError> {
    let batch_slice = runner
        .slice(0)
        .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
    let mut digest = Sha256::new();
    for descriptor in terminal
        .descriptors
        .iter()
        .filter(|descriptor| descriptor.usage == VulkanKernelDescriptorUsage::OutputSignal)
    {
        let Some((key, frame_byte_capacity)) =
            component_batch_signal_target_with_mounted(&placed_slice.mounted, descriptor)
                .map_err(|error| distributed_calibration_error_value(error.to_string()))?
        else {
            if targeted_signal_accepts_fixture_mutation(descriptor) {
                return distributed_calibration_error(format!(
                    "distributed prefill output {} has no signal buffer",
                    descriptor.name,
                ));
            }
            continue;
        };
        let byte_count = frame_byte_capacity
            .checked_mul(activation_batch_width)
            .ok_or_else(|| distributed_calibration_error_value(
                "distributed prefill output size overflowed",
            ))?;
        digest.update(descriptor.binding.to_le_bytes());
        digest.update(descriptor.name.as_bytes());
        digest.update(
            batch_slice
                .signal_buffer(&key)
                .map_err(|error| distributed_calibration_error_value(error.to_string()))?
                .buffer
                .read_bytes(byte_count)
                .map_err(|error| distributed_calibration_error_value(error.to_string()))?,
        );
    }
    Ok(targeted_finalized_artifact_digest(
        digest.finalize().as_slice(),
    ))
}

fn distributed_calibration_output_digest(
    mounted: &VulkanMountedPlacedStreamCircuit,
    terminal: &VulkanMountedPlacedBoundDispatch,
) -> Result<String, VulkanResidentTokenModelPackageError> {
    let mut digest = Sha256::new();
    for descriptor in terminal
        .descriptors
        .iter()
        .filter(|descriptor| descriptor.usage == VulkanKernelDescriptorUsage::OutputSignal)
    {
        let binding = mounted
            .resident_kernel_buffer_binding(terminal, descriptor)
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        digest.update(descriptor.binding.to_le_bytes());
        digest.update(descriptor.name.as_bytes());
        digest.update(
            binding
                .buffer
                .read_bytes_at(
                    binding.byte_offset,
                    targeted_signal_byte_count(descriptor, binding.byte_len),
                )
                .map_err(|error| distributed_calibration_error_value(error.to_string()))?,
        );
    }
    Ok(targeted_finalized_artifact_digest(
        digest.finalize().as_slice(),
    ))
}

fn distributed_calibration_state_digest(
    mounted: &VulkanMountedPlacedStreamCircuit,
) -> Result<String, VulkanResidentTokenModelPackageError> {
    let mut digest = Sha256::new();
    for state in &mounted.buffers.state_buffers {
        digest.update(state.component_id.as_bytes());
        digest.update(state.state_id.as_bytes());
        digest.update(
            state
                .buffer
                .read_bytes(state.buffer.byte_capacity())
                .map_err(|error| distributed_calibration_error_value(error.to_string()))?,
        );
    }
    Ok(targeted_finalized_artifact_digest(
        digest.finalize().as_slice(),
    ))
}

fn remaining_calibration_duration(
    started: Instant,
    maximum_duration: Duration,
) -> Result<Duration, VulkanResidentTokenModelPackageError> {
    maximum_duration
        .checked_sub(started.elapsed())
        .ok_or_else(|| {
            distributed_calibration_error_value(
                "distributed runtime placement calibration exceeded its configured duration",
            )
        })
}

fn distributed_calibration_error<T>(
    message: impl Into<String>,
) -> Result<T, VulkanResidentTokenModelPackageError> {
    Err(distributed_calibration_error_value(message))
}

fn distributed_calibration_error_value(
    message: impl Into<String>,
) -> VulkanResidentTokenModelPackageError {
    VulkanResidentTokenModelPackageError::new(message.into())
}
