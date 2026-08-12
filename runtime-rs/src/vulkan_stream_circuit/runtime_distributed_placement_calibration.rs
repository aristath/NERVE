#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanRuntimeDistributedPlacementCalibrationReport {
    pub physical_device_ids: Vec<String>,
    pub target: VulkanRuntimePlacementCalibrationTarget,
    pub phase: String,
    pub activation_batch_width: usize,
    pub sampled_workload: bool,
    pub sample_fraction_millionths: usize,
    pub measured_execution_ns: u64,
    pub measured_ns_per_activation: u64,
    pub measured_windows: Vec<VulkanTargetedComponentThroughputWindow>,
    pub physical_dispatch_count: usize,
    pub shard_count: usize,
    pub output_digest: String,
    pub output_artifact: Option<VulkanPlacementOutputArtifact>,
    pub state_digest: String,
    pub resident_parameter_bytes_by_device: BTreeMap<String, usize>,
    pub resident_transient_bytes_by_device: BTreeMap<String, usize>,
    pub resident_host_transient_bytes: usize,
    pub activation_routes: Vec<String>,
    pub dispatch_work: Vec<VulkanRuntimeDistributedPlacementDispatchWork>,
    pub execution_case: VulkanPlacementExecutionCaseIdentity,
    pub warmup_call_count: usize,
    pub measured_call_count: usize,
    pub useful_activation_count: usize,
}

impl VulkanRuntimeDistributedPlacementCalibrationReport {
    pub fn canonical_reference(
        &self,
    ) -> Result<VulkanPlacementCanonicalReference, VulkanPlacementCalibrationCatalogError> {
        if self.execution_case.strategy != VulkanPlacementExecutionStrategy::SingleDevice {
            return Err(VulkanPlacementCalibrationCatalogError(
                "only a single-device execution of the exact sampled contract may establish its canonical placement reference"
                    .to_string(),
            ));
        }
        Ok(VulkanPlacementCanonicalReference {
            behavior: self.execution_case.behavior.clone(),
            output_digest: self.output_digest.clone(),
            output_artifact: self.output_artifact.clone(),
            state_digest: self.state_digest.clone(),
        })
    }

    pub fn calibration_observation(
        &self,
        output_equivalence: VulkanPlacementOutputEquivalenceEvidence,
    ) -> VulkanPlacementCalibrationObservation {
        VulkanPlacementCalibrationObservation {
            execution_case: self.execution_case.clone(),
            warmup_call_count: self.warmup_call_count,
            measured_call_count: self.measured_call_count,
            complete_transaction: true,
            duration_ns: self.measured_execution_ns,
            useful_activation_count: self.useful_activation_count,
            output_digest: self.output_digest.clone(),
            output_artifact: self.output_artifact.clone(),
            output_equivalence,
            state_digest: self.state_digest.clone(),
            resident_bytes_by_physical_device: self.resident_parameter_bytes_by_device.clone(),
            transient_peak_bytes_by_physical_device: self
                .resident_transient_bytes_by_device
                .clone(),
            host_resident_bytes: 0,
            host_transient_peak_bytes: self.resident_host_transient_bytes,
        }
    }
}

pub fn record_vulkan_runtime_distributed_calibration_report(
    catalog: &mut VulkanPlacementCalibrationCatalog,
    report: &VulkanRuntimeDistributedPlacementCalibrationReport,
) -> Result<(), VulkanPlacementCalibrationCatalogError> {
    if report.execution_case.strategy == VulkanPlacementExecutionStrategy::SingleDevice
        && catalog
            .canonical_reference(&report.execution_case.behavior)
            .is_none()
    {
        catalog.record_reference(report.canonical_reference()?)?;
    }
    let reference = catalog
        .canonical_reference(&report.execution_case.behavior)
        .ok_or_else(|| {
            VulkanPlacementCalibrationCatalogError(
                "distributed placement candidate has no measured single-device reference"
                    .to_string(),
            )
        })?;
    let output_equivalence = validate_vulkan_placement_output_equivalence(
        &report.execution_case.behavior.equivalence,
        &reference.output_digest,
        reference.output_artifact.as_ref(),
        &report.output_digest,
        report.output_artifact.as_ref(),
    )?;
    catalog.record_observation(report.calibration_observation(output_equivalence))
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
    distributed_resource_stores: BTreeMap<String, Arc<VulkanCompiledResourceDeviceStore>>,
    _distributed_dynamic_resource_buffers: BTreeMap<String, Arc<VulkanDynamicResourceBuffers>>,
    _distributed_transaction_predicates: BTreeMap<String, Arc<VulkanResidentBuffer>>,
    _parameter_pool: VulkanResidentBufferPool,
    resident_parameter_bytes_by_device: BTreeMap<String, usize>,
    resident_transient_bytes_by_device: BTreeMap<String, usize>,
    resident_host_transient_bytes: usize,
    activation_routes: Vec<String>,
    dispatch_work: Vec<VulkanRuntimeDistributedPlacementDispatchWork>,
    sampled_workload: bool,
    sample_fraction_millionths: usize,
    execution_case: VulkanPlacementExecutionCaseIdentity,
}

pub fn calibrate_vulkan_runtime_distributed_placement_candidate_with_policy(
    devices: Vec<(String, Rc<VulkanComputeDevice>)>,
    manifest_dir: impl AsRef<Path>,
    runtime_model: &VulkanResidentRuntimeModel,
    target: &VulkanRuntimePlacementCalibrationTarget,
    policy: VulkanRuntimePlacementCalibrationPolicy,
) -> Result<
    Option<VulkanRuntimeDistributedPlacementCalibrationReport>,
    VulkanResidentTokenModelPackageError,
> {
    calibrate_vulkan_runtime_distributed_placement_phase_candidate_with_policy(
        devices,
        manifest_dir.as_ref(),
        runtime_model,
        target,
        VulkanTargetedComponentExecutionPhase::Decode,
        None,
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
) -> Result<
    Option<VulkanRuntimeDistributedPlacementCalibrationReport>,
    VulkanResidentTokenModelPackageError,
> {
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
        None,
        policy,
    )
}

fn calibrate_vulkan_runtime_distributed_placement_phase_candidate_with_policy(
    devices: Vec<(String, Rc<VulkanComputeDevice>)>,
    manifest_dir: &Path,
    runtime_model: &VulkanResidentRuntimeModel,
    target: &VulkanRuntimePlacementCalibrationTarget,
    phase: VulkanTargetedComponentExecutionPhase,
    required_sample_fraction_millionths: Option<usize>,
    policy: VulkanRuntimePlacementCalibrationPolicy,
) -> Result<
    Option<VulkanRuntimeDistributedPlacementCalibrationReport>,
    VulkanResidentTokenModelPackageError,
> {
    if devices.is_empty() {
        return distributed_calibration_error(
            "runtime shard calibration requires at least one device",
        );
    }
    if policy.warmup_units == 0
        || policy.measured_units == 0
        || policy.maximum_duration.is_zero()
        || policy.maximum_total_resident_parameter_bytes == 0
        || policy
            .maximum_resident_parameter_bytes_by_physical_device
            .iter()
            .any(|(physical_id, capacity)| physical_id.is_empty() || *capacity == 0)
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

    let logical_parameter_capacities = devices
        .iter()
        .enumerate()
        .map(|(index, (physical_id, _))| {
            policy
                .parameter_capacity_for_physical_device(physical_id)
                .map(|capacity| (format!("calibration:shard:{index}"), capacity))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    let aggregate_physical_capacity = logical_parameter_capacities
        .values()
        .try_fold(0usize, |total, capacity| total.checked_add(*capacity))
        .unwrap_or(usize::MAX);
    let maximum_total_resident_parameter_bytes = policy
        .maximum_total_resident_parameter_bytes
        .min(aggregate_physical_capacity);
    let started = Instant::now();
    let Some(mut session) = VulkanRuntimeDistributedPlacementSession::prepare(
        devices,
        manifest_dir,
        runtime_model,
        target,
        phase,
        required_sample_fraction_millionths,
        maximum_total_resident_parameter_bytes,
        &logical_parameter_capacities,
    )?
    else {
        return Ok(None);
    };
    let execution_result = (|| {
        let activation_batch_width = phase.activation_batch_width();
        let warmup_useful_units = policy
            .warmup_units
            .checked_mul(activation_batch_width)
            .ok_or_else(|| {
                distributed_calibration_error_value(
                    "distributed calibration warmup work overflowed",
                )
            })?;
        let measured_useful_units = policy
            .measured_units
            .checked_mul(activation_batch_width)
            .ok_or_else(|| {
                distributed_calibration_error_value(
                    "distributed calibration measured work overflowed",
                )
            })?;
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
                    (
                        physical_id.clone(),
                        bytes.get(logical_id).copied().unwrap_or(0),
                    )
                })
                .collect::<BTreeMap<_, _>>()
        };
        let current_resident_parameter_bytes =
            session.current_resident_parameter_bytes_by_device()?;
        Ok(VulkanRuntimeDistributedPlacementCalibrationReport {
            physical_device_ids: session.physical_device_ids.clone(),
            target: session.target.clone(),
            phase: measured.phase,
            activation_batch_width: measured.activation_batch_width,
            sampled_workload: session.sampled_workload,
            sample_fraction_millionths: session.sample_fraction_millionths,
            measured_execution_ns: measured.execution_ns,
            measured_ns_per_activation,
            measured_windows: measured.windows,
            physical_dispatch_count: measured.physical_dispatch_count,
            shard_count: measured.shard_count,
            output_digest: measured.output_digest,
            output_artifact: measured.output_artifact,
            state_digest: measured.state_digest,
            resident_parameter_bytes_by_device: remap_device_bytes(
                &current_resident_parameter_bytes,
            ),
            resident_transient_bytes_by_device: remap_device_bytes(
                &session.resident_transient_bytes_by_device,
            ),
            resident_host_transient_bytes: session.resident_host_transient_bytes,
            activation_routes: session.activation_routes.clone(),
            dispatch_work: session.dispatch_work.clone(),
            execution_case: session.execution_case.clone(),
            warmup_call_count: policy.warmup_units,
            measured_call_count: policy.measured_units,
            useful_activation_count: measured_useful_units,
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

fn distributed_calibration_execution_case(
    devices: &[(String, Rc<VulkanComputeDevice>)],
    logical_device_ids: &[String],
    execution_plan: &VulkanDistributedExecutionPlan,
    loaded_manifest: &VulkanLoadedKernelArtifactCatalog,
    compiled_execution_signature: String,
    artifact_digest: String,
    execution_graph_digest: String,
    phase: VulkanTargetedComponentExecutionPhase,
    dispatch_work: &[VulkanRuntimeDistributedPlacementDispatchWork],
) -> Result<VulkanPlacementExecutionCaseIdentity, VulkanResidentTokenModelPackageError> {
    if devices.len() != logical_device_ids.len()
        || execution_plan.dispatches.len() != dispatch_work.len()
        || execution_plan.execution_islands.is_empty()
    {
        return distributed_calibration_error(
            "distributed calibration cannot identify an incomplete physical execution case",
        );
    }
    let physical_id_by_logical = logical_device_ids
        .iter()
        .cloned()
        .zip(devices.iter().map(|(physical_id, _)| physical_id.clone()))
        .collect::<BTreeMap<_, _>>();
    let participant_ordinal_by_logical = logical_device_ids
        .iter()
        .enumerate()
        .map(|(ordinal, logical_id)| (logical_id.as_str(), ordinal))
        .collect::<BTreeMap<_, _>>();
    let physical_id = |logical_id: &str| {
        physical_id_by_logical
            .get(logical_id)
            .cloned()
            .ok_or_else(|| {
                distributed_calibration_error_value(format!(
                    "distributed calibration physical case references unknown logical device {logical_id:?}",
                ))
            })
    };

    let mut devices = devices
        .iter()
        .map(
            |(physical_device_id, device)| VulkanPlacementDeviceExecutionIdentity {
                physical_device_id: physical_device_id.clone(),
                api_version: device.api_version(),
                driver_version: device.driver_version(),
            },
        )
        .collect::<Vec<_>>();
    devices.sort();

    let mut contract_digests = BTreeMap::<String, String>::new();
    for island in &execution_plan.execution_islands {
        for (contract_id, implementation_digest) in island
            .contract_ids
            .iter()
            .zip(&island.implementation_digests)
        {
            if let Some(existing) =
                contract_digests.insert(contract_id.clone(), implementation_digest.clone())
                && existing != *implementation_digest
            {
                return distributed_calibration_error(format!(
                    "distributed calibration contract {contract_id:?} has conflicting implementation digests",
                ));
            }
        }
    }
    if contract_digests.is_empty() {
        return distributed_calibration_error(
            "distributed calibration physical case has no implementation contract",
        );
    }
    let contract_ids = contract_digests.keys().cloned().collect::<Vec<_>>();
    let implementation_digests = contract_digests.values().cloned().collect::<Vec<_>>();

    let mut operations = Vec::with_capacity(execution_plan.dispatches.len());
    let mut shards = Vec::new();
    for (dispatch_ordinal, (dispatch, work)) in execution_plan
        .dispatches
        .iter()
        .zip(dispatch_work)
        .enumerate()
    {
        let artifact = loaded_manifest
            .physical_artifact(&dispatch.physical_artifact_id)
            .ok_or_else(|| {
                distributed_calibration_error_value(format!(
                    "distributed calibration case is missing loaded artifact {:?}",
                    dispatch.physical_artifact_id,
                ))
            })?;
        let workgroup_count_x = dispatch.shards.iter().try_fold(0u32, |total, shard| {
            total.checked_add(shard.workgroup_count_x).ok_or_else(|| {
                distributed_calibration_error_value(
                    "distributed calibration workgroup geometry overflowed",
                )
            })
        })?;
        operations.push(VulkanPlacementOperationGeometry::Dispatch {
            geometry: VulkanPlacementDispatchGeometry {
                contract_id: dispatch.physical_execution_contract_id.clone(),
                logical_extent: work.full_rows,
                sampled_extent: work.sampled_rows,
                input_width: dispatch.input_width,
                workgroup_count_x,
                local_size_x: artifact.artifact.local_size_x,
            },
        });
        if let Some(reduction) = distributed_calibration_reduction_geometry(
            &dispatch.physical_execution_contract_id,
            dispatch.reduction.as_ref(),
            dispatch.shards.len(),
        )? {
            operations.push(reduction);
        }
        for shard in &dispatch.shards {
            let participant_ordinal = participant_ordinal_by_logical
                .get(shard.device_id.as_str())
                .copied()
                .ok_or_else(|| {
                    distributed_calibration_error_value(format!(
                        "distributed calibration shard references unknown logical device {:?}",
                        shard.device_id,
                    ))
                })?;
            let parameter_bytes =
                shard
                    .parameters
                    .iter()
                    .try_fold(0usize, |total, parameter| {
                        total.checked_add(parameter.byte_count).ok_or_else(|| {
                            distributed_calibration_error_value(
                                "distributed calibration shard parameter bytes overflowed",
                            )
                        })
                    })?;
            let selected_resource_indices_by_partition =
                distributed_calibration_normalized_selected_resource_indices(
                    dispatch
                        .selected_resource_partitions
                        .iter()
                        .map(|partition| partition.selector_id.as_str()),
                    &shard.selected_resource_indices,
                    &shard.device_id,
                )?;
            let selected_resource_fragments_by_partition =
                distributed_calibration_normalized_selected_resource_fragments(
                    dispatch
                        .selected_resource_partitions
                        .iter()
                        .map(|partition| partition.selector_id.as_str()),
                    &shard.selected_resource_fragments,
                    &shard.device_id,
                )?;
            shards.push(VulkanPlacementShardIdentity {
                dispatch_ordinal,
                participant_ordinal,
                physical_device_id: physical_id(&shard.device_id)?,
                distribution: distributed_calibration_distribution_name(dispatch.distribution)
                    .to_string(),
                logical_start: shard.row_start,
                logical_count: shard.row_count,
                selected_resource_indices_by_partition,
                selected_resource_fragments_by_partition,
                parameter_bytes,
            });
        }
    }
    shards.sort();

    let first_island = execution_plan
        .execution_islands
        .first()
        .expect("checked nonempty above");
    let last_island = execution_plan
        .execution_islands
        .last()
        .expect("checked nonempty above");
    let input_physical_device_id = physical_id(&first_island.entry_device_id)?;
    let output_physical_device_id = physical_id(&last_island.exit_device_id)?;
    let owner_physical_device_id = physical_id(&first_island.owner_device_id)?;
    if execution_plan
        .execution_islands
        .iter()
        .any(|island| island.owner_device_id != first_island.owner_device_id)
    {
        return distributed_calibration_error(
            "one distributed calibration case cannot span multiple island owners",
        );
    }
    let mut transports = execution_plan
        .execution_islands
        .iter()
        .flat_map(|island| &island.transport_routes)
        .map(|route| {
            Ok(VulkanPlacementTransportIdentity {
                source_physical_device_id: physical_id(&route.source_device_id)?,
                destination_physical_device_id: physical_id(&route.destination_device_id)?,
                byte_capacity: route.byte_capacity,
                route: match route.kind {
                    VulkanPhysicalExecutionTransportKind::ExternalDeviceLocal => {
                        "external_device_local"
                    }
                    VulkanPhysicalExecutionTransportKind::SharedHost => "shared_host",
                }
                .to_string(),
            })
        })
        .collect::<Result<Vec<_>, VulkanResidentTokenModelPackageError>>()?;
    transports.sort();
    transports.dedup();

    let shape = VulkanPlacementShapeClass {
        activation_batch_width: phase.activation_batch_width(),
        input_byte_capacity: first_island.leader().input_byte_capacity,
        output_byte_capacity: last_island.tail().output_byte_capacity,
        operations,
    };
    let execution_phase = match phase {
        VulkanTargetedComponentExecutionPhase::Decode => {
            nerve_execution_contracts::ExecutionPhase::Decode
        }
        VulkanTargetedComponentExecutionPhase::Prefill { .. } => {
            nerve_execution_contracts::ExecutionPhase::Prefill
        }
    };
    let input_fixture_digest =
        distributed_calibration_fixture_identity(execution_phase, &shape, 0)?;
    let equivalence = distributed_calibration_equivalence(execution_plan)?;
    Ok(VulkanPlacementExecutionCaseIdentity {
        behavior: VulkanPlacementBehaviorIdentity {
            compiled_execution_signature,
            contract_ids,
            implementation_digests,
            artifact_digest,
            execution_graph_digest,
            runtime_implementation_fingerprint: crate::RUNTIME_IMPLEMENTATION_FINGERPRINT
                .to_string(),
            phase: execution_phase,
            shape,
            input_fixture_digest,
            equivalence,
        },
        strategy: vulkan_distributed_placement_strategy(
            devices.len(),
            execution_plan
                .dispatches
                .iter()
                .map(|dispatch| dispatch.execution_strategy),
        )
        .map_err(|error| distributed_calibration_error_value(error.to_string()))?,
        devices,
        shards,
        input_physical_device_id,
        output_physical_device_id,
        owner_physical_device_id,
        transports,
    })
}

fn distributed_calibration_normalized_selected_resource_indices<'a>(
    selector_ids: impl IntoIterator<Item = &'a str>,
    selected_resource_indices: &BTreeMap<String, Vec<usize>>,
    device_id: &str,
) -> Result<BTreeMap<usize, Vec<usize>>, VulkanResidentTokenModelPackageError> {
    selector_ids
        .into_iter()
        .enumerate()
        .map(|(partition_ordinal, selector_id)| {
            selected_resource_indices
                .get(selector_id)
                .cloned()
                .map(|indices| (partition_ordinal, indices))
                .ok_or_else(|| {
                    distributed_calibration_error_value(format!(
                        "distributed calibration shard on {device_id:?} has no ownership for partition ordinal {partition_ordinal}",
                    ))
                })
        })
        .collect()
}

fn distributed_calibration_normalized_selected_resource_fragments<'a>(
    selector_ids: impl Iterator<Item = &'a str>,
    selected_resource_fragments: &BTreeMap<
        String,
        Vec<VulkanDistributedSelectedResourceFragmentPlan>,
    >,
    device_id: &str,
) -> Result<
    BTreeMap<usize, Vec<VulkanPlacementSelectedResourceFragmentIdentity>>,
    VulkanResidentTokenModelPackageError,
> {
    let selector_ordinals = selector_ids.enumerate().collect::<BTreeMap<_, _>>();
    let known_selectors = selector_ordinals
        .values()
        .copied()
        .collect::<BTreeSet<_>>();
    if selected_resource_fragments
        .keys()
        .any(|selector_id| !known_selectors.contains(selector_id.as_str()))
    {
        return distributed_calibration_error(format!(
            "distributed calibration shard on {device_id:?} references an unknown selected-resource fragment selector",
        ));
    }
    selector_ordinals
        .into_iter()
        .filter_map(|(partition_ordinal, selector_id)| {
            let fragments = selected_resource_fragments.get(selector_id)?;
            Some(Ok((
                partition_ordinal,
                fragments
                    .iter()
                    .map(|fragment| VulkanPlacementSelectedResourceFragmentIdentity {
                        resource_index: fragment.resource_index,
                        atomic_group_id: fragment.atomic_group_id.clone(),
                        logical_start: fragment.logical_start,
                        logical_count: fragment.logical_count,
                        parameters: fragment
                            .parameters
                            .iter()
                            .map(|parameter| {
                                VulkanPlacementSelectedResourceParameterFragmentIdentity {
                                    parameter_slot: parameter.parameter_slot,
                                    resource_id: parameter.resource_id.clone(),
                                    resource_byte_count: parameter.resource_byte_count,
                                    byte_offset: parameter.byte_offset,
                                    byte_count: parameter.byte_count,
                                }
                            })
                            .collect(),
                    })
                    .collect(),
            )))
        })
        .collect()
}

fn distributed_calibration_equivalence(
    execution_plan: &VulkanDistributedExecutionPlan,
) -> Result<VulkanPlacementEquivalenceIdentity, VulkanResidentTokenModelPackageError> {
    let dispatches = execution_plan
        .execution_islands
        .iter()
        .flat_map(|island| &island.dispatches)
        .collect::<Vec<_>>();
    distributed_calibration_equivalence_from_dispatches(&dispatches)
}

fn distributed_calibration_equivalence_from_dispatches(
    dispatches: &[&VulkanDistributedDispatchPlan],
) -> Result<VulkanPlacementEquivalenceIdentity, VulkanResidentTokenModelPackageError> {
    let contracts = dispatches
        .iter()
        .map(|dispatch| {
            (
                dispatch.equivalence.clone(),
                dispatch
                    .reduction
                    .as_ref()
                    .map(|reduction| reduction.finalization.clone()),
            )
        })
        .collect::<Vec<_>>();
    distributed_calibration_equivalence_from_contracts(&contracts)
}

fn distributed_calibration_equivalence_from_contracts(
    contracts: &[(
        crate::VulkanDistributedEquivalencePlan,
        Option<VulkanDistributedReductionFinalizationPlan>,
    )],
) -> Result<VulkanPlacementEquivalenceIdentity, VulkanResidentTokenModelPackageError> {
    let Some((tail, tail_finalization)) = contracts.last() else {
        return distributed_calibration_error(
            "distributed calibration equivalence requires an executable dispatch",
        );
    };
    if contracts[..contracts.len() - 1]
        .iter()
        .any(|(equivalence, _)| equivalence.output != VulkanDistributedEquivalenceKind::BitExact)
    {
        return distributed_calibration_error(
            "distributed calibration cannot compose a tolerant intermediate without a compiler-declared region equivalence",
        );
    }
    if contracts
        .iter()
        .any(|(equivalence, _)| equivalence.state != VulkanDistributedEquivalenceKind::BitExact)
    {
        return distributed_calibration_error(
            "distributed calibration cannot validate tolerant state without a typed compiled state layout",
        );
    }
    let output = match tail.output {
        VulkanDistributedEquivalenceKind::BitExact => VulkanPlacementEquivalenceKind::BitExact,
        VulkanDistributedEquivalenceKind::AbsoluteRelativeTolerance => {
            VulkanPlacementEquivalenceKind::AbsoluteRelativeTolerance
        }
    };
    let output_scalar_format = match output {
        VulkanPlacementEquivalenceKind::BitExact => None,
        VulkanPlacementEquivalenceKind::AbsoluteRelativeTolerance => match tail_finalization {
            Some(VulkanDistributedReductionFinalizationPlan::StoreF32) => {
                Some(VulkanPlacementScalarFormat::F32)
            }
            Some(
                VulkanDistributedReductionFinalizationPlan::StoreF32ToBf16
                | VulkanDistributedReductionFinalizationPlan::AddBf16ResidualToBf16 { .. }
            ) => {
                Some(VulkanPlacementScalarFormat::Bf16)
            }
            None => {
                return distributed_calibration_error(
                    "tolerant distributed output has no typed reduction finalization",
                );
            }
        },
    };
    let equivalence = VulkanPlacementEquivalenceIdentity {
        output,
        state: VulkanPlacementEquivalenceKind::BitExact,
        absolute_tolerance_bits: tail.absolute_tolerance_bits,
        relative_tolerance_bits: tail.relative_tolerance_bits,
        output_scalar_format,
    };
    equivalence
        .validate()
        .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
    Ok(equivalence)
}

fn distributed_calibration_reduction_geometry(
    contract_id: &str,
    reduction: Option<&VulkanDistributedReductionPlan>,
    participant_count: usize,
) -> Result<Option<VulkanPlacementOperationGeometry>, VulkanResidentTokenModelPackageError> {
    let Some(reduction) = reduction else {
        return Ok(None);
    };
    if contract_id.is_empty() || reduction.element_count == 0 || participant_count < 2 {
        return distributed_calibration_error(
            "distributed calibration reduction geometry is incomplete",
        );
    }
    Ok(Some(VulkanPlacementOperationGeometry::Reduction {
        contract_id: contract_id.to_string(),
        element_count: reduction.element_count,
        element_byte_count: size_of::<f32>(),
        participant_count,
    }))
}

fn distributed_calibration_distribution_name(
    distribution: VulkanDistributedDispatchDistribution,
) -> &'static str {
    match distribution {
        VulkanDistributedDispatchDistribution::OutputRows => "output_rows",
        VulkanDistributedDispatchDistribution::InputColumns => "input_columns",
        VulkanDistributedDispatchDistribution::ExpertRange => "expert_range",
    }
}

fn distributed_calibration_activation_backing_bytes<'a>(
    device_ids: &[String],
    route: VulkanSharedResidentBufferRoute,
    allocations: impl IntoIterator<Item = (&'a str, usize)>,
) -> Result<(BTreeMap<String, usize>, usize), VulkanResidentTokenModelPackageError> {
    let mut device_bytes = device_ids
        .iter()
        .map(|device_id| (device_id.clone(), 0usize))
        .collect::<BTreeMap<_, _>>();
    let mut host_bytes = 0usize;
    for (owner_device_id, byte_capacity) in allocations {
        if byte_capacity == 0 || !device_bytes.contains_key(owner_device_id) {
            return distributed_calibration_error(format!(
                "distributed activation allocation has unknown owner {owner_device_id:?} or zero bytes",
            ));
        }
        match route {
            VulkanSharedResidentBufferRoute::ExternalDeviceLocal => {
                let total = device_bytes
                    .get_mut(owner_device_id)
                    .expect("owner presence checked above");
                *total = total.checked_add(byte_capacity).ok_or_else(|| {
                    distributed_calibration_error_value(
                        "distributed device-local activation byte accounting overflowed",
                    )
                })?;
            }
            VulkanSharedResidentBufferRoute::SharedHost => {
                host_bytes = host_bytes.checked_add(byte_capacity).ok_or_else(|| {
                    distributed_calibration_error_value(
                        "distributed shared-host activation byte accounting overflowed",
                    )
                })?;
            }
        }
    }
    Ok((device_bytes, host_bytes))
}

fn distributed_calibration_fixture_identity(
    phase: nerve_execution_contracts::ExecutionPhase,
    shape: &VulkanPlacementShapeClass,
    seed: u32,
) -> Result<String, VulkanResidentTokenModelPackageError> {
    let payload = serde_json::to_vec(&(
        "nerve.distributed_calibration_fixture.v1",
        phase,
        shape,
        seed,
    ))
    .map_err(|error| {
        distributed_calibration_error_value(format!(
            "failed to encode distributed calibration fixture identity: {error}",
        ))
    })?;
    Ok(format!("sha256:{:x}", Sha256::digest(payload)))
}

fn distributed_calibration_artifact_digest(
    loaded_manifest: &VulkanLoadedKernelArtifactCatalog,
    execution_plan: &VulkanDistributedExecutionPlan,
) -> Result<String, VulkanResidentTokenModelPackageError> {
    let mut digest = Sha256::new();
    digest.update(b"nerve.distributed_calibration_artifacts.v2\0");
    if execution_plan.dispatches.is_empty() {
        return distributed_calibration_error("distributed calibration artifact identity is empty");
    }
    for dispatch in &execution_plan.dispatches {
        let artifact = loaded_manifest
            .physical_artifact(&dispatch.physical_artifact_id)
            .ok_or_else(|| {
                distributed_calibration_error_value(format!(
                    "distributed calibration artifact identity is missing physical artifact {:?}",
                    dispatch.physical_artifact_id,
                ))
            })?;
        distributed_calibration_update_artifact_digest(&mut digest, artifact)?;
    }
    Ok(format!("sha256:{:x}", digest.finalize()))
}

fn distributed_calibration_update_artifact_digest(
    digest: &mut Sha256,
    artifact: &VulkanLoadedPhysicalKernelArtifact,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    // IDs, operation labels, and source paths are not executable identity.
    // Hash only the interface and SPIR-V that the selected physical dispatch
    // will execute, in dispatch order.
    let interface = serde_json::to_vec(&(
        artifact.artifact.entry_point.as_str(),
        artifact.artifact.local_size_x,
        artifact.artifact.workgroup_count_x,
        &artifact.artifact.descriptor_signature,
        &artifact.artifact.push_constants,
        artifact.artifact.stream_control_binding,
    ))
    .map_err(|error| {
        distributed_calibration_error_value(format!(
            "failed to encode distributed calibration artifact interface: {error}",
        ))
    })?;
    digest.update((interface.len() as u64).to_le_bytes());
    digest.update(interface);
    digest.update((artifact.words.len() as u64).to_le_bytes());
    for word in &artifact.words {
        digest.update(word.to_le_bytes());
    }
    Ok(())
}

fn distributed_calibration_execution_graph_digest(
    target: &VulkanRuntimePlacementCalibrationTarget,
    tick_plan: &VulkanMountedPlacedStreamTickPlan,
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"nerve.distributed_calibration_execution_graph.v2\0");
    // The target signature already identifies the complete compiler-emitted
    // physical transaction while deliberately excluding semantic labels. Keep
    // that property here: equivalent repeated components must share placement
    // evidence, while a different stage/IO topology must not.
    digest.update(target.signature_id.as_bytes());
    digest.update([0]);
    digest.update(tick_plan.stage_count.to_le_bytes());
    digest.update(tick_plan.receive_stage_count.to_le_bytes());
    digest.update(tick_plan.dispatch_stage_count.to_le_bytes());
    digest.update(tick_plan.publish_stage_count.to_le_bytes());
    for stage in &tick_plan.stages {
        match stage {
            VulkanMountedPlacedStreamTickStage::ReceiveEdge { byte_capacity, .. } => {
                digest.update([0]);
                digest.update(byte_capacity.to_le_bytes());
            }
            VulkanMountedPlacedStreamTickStage::Dispatch { dispatch, .. } => {
                digest.update([1]);
                digest.update(dispatch.descriptor_count.to_le_bytes());
                digest.update(dispatch.resident_descriptor_count.to_le_bytes());
                digest.update(dispatch.reads.len().to_le_bytes());
                for read in &dispatch.reads {
                    distributed_calibration_update_io_topology_digest(&mut digest, read);
                }
                digest.update(dispatch.writes.len().to_le_bytes());
                for write in &dispatch.writes {
                    distributed_calibration_update_io_topology_digest(&mut digest, write);
                }
            }
            VulkanMountedPlacedStreamTickStage::PublishEdge { byte_capacity, .. } => {
                digest.update([2]);
                digest.update(byte_capacity.to_le_bytes());
            }
        }
    }
    format!("sha256:{:x}", digest.finalize())
}

fn distributed_calibration_update_io_topology_digest(
    digest: &mut Sha256,
    io: &VulkanMountedPlacedStreamTickIo,
) {
    match io {
        VulkanMountedPlacedStreamTickIo::ActivationSlot { slot, .. } => {
            digest.update([0]);
            digest.update(slot.to_le_bytes());
        }
        VulkanMountedPlacedStreamTickIo::ModelSignal { .. } => digest.update([1]),
        VulkanMountedPlacedStreamTickIo::LocalEdgeBuffer { byte_capacity, .. } => {
            digest.update([2]);
            digest.update(byte_capacity.to_le_bytes());
        }
        VulkanMountedPlacedStreamTickIo::IncomingEdgeBuffer { byte_capacity, .. } => {
            digest.update([3]);
            digest.update(byte_capacity.to_le_bytes());
        }
        VulkanMountedPlacedStreamTickIo::OutgoingEdgeBuffer { byte_capacity, .. } => {
            digest.update([4]);
            digest.update(byte_capacity.to_le_bytes());
        }
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
    output_artifact: Option<VulkanPlacementOutputArtifact>,
    state_digest: String,
}

fn distributed_contract_phase_and_shape(
    phase: VulkanTargetedComponentExecutionPhase,
) -> (
    nerve_execution_contracts::ExecutionPhase,
    nerve_execution_contracts::ExecutionShape,
) {
    match phase {
        VulkanTargetedComponentExecutionPhase::Decode => (
            nerve_execution_contracts::ExecutionPhase::Decode,
            nerve_execution_contracts::ExecutionShape::SingleLane,
        ),
        VulkanTargetedComponentExecutionPhase::Prefill { .. } => (
            nerve_execution_contracts::ExecutionPhase::Prefill,
            nerve_execution_contracts::ExecutionShape::MultiLane,
        ),
    }
}

impl VulkanRuntimeDistributedPlacementSession {
    fn prepare(
        devices: Vec<(String, Rc<VulkanComputeDevice>)>,
        manifest_dir: &Path,
        runtime_model: &VulkanResidentRuntimeModel,
        target: &VulkanRuntimePlacementCalibrationTarget,
        phase: VulkanTargetedComponentExecutionPhase,
        required_sample_fraction_millionths: Option<usize>,
        maximum_total_resident_parameter_bytes: usize,
        maximum_resident_parameter_bytes_by_logical_device: &BTreeMap<String, usize>,
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
        let artifact_manifest = VulkanPhysicalKernelArtifactManifest::new(
            loaded_manifest
                .physical_artifacts
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
        let (contract_phase, execution_shape) = distributed_contract_phase_and_shape(phase);
        let full_distributed_execution_plan =
            VulkanDistributedExecutionPlan::from_prepared_plans_for_phase_with_resource_contract(
                &[(
                    owner_device_id.as_str(),
                    &targeted_plan.slice_plan.prepared_plan,
                )],
                &tensor_index,
                &artifact_manifest,
                &BTreeMap::from([(target.component_id.clone(), planning_device_ids)]),
                &placement_plan.edges,
                alignment,
                contract_phase,
                execution_shape,
                &placed_model.execution_scope,
                &contract,
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
            maximum_total_resident_parameter_bytes.checked_sub(owner_static_bytes)
        else {
            return Ok(None);
        };
        let (distributed_execution_plan, sample_fraction_millionths) =
            if let Some(fraction_millionths) = required_sample_fraction_millionths {
                (
                    full_distributed_execution_plan
                        .sampled_for_fraction_millionths(&logical_device_ids, fraction_millionths)
                        .map_err(|error| distributed_calibration_error_value(error.to_string()))?,
                    fraction_millionths,
                )
            } else {
                let Some(sampled) = full_distributed_execution_plan
                    .sampled_for_parameter_budget_with_fraction(
                        &tensor_index,
                        &logical_device_ids,
                        distributed_parameter_budget,
                    )
                    .map_err(|error| distributed_calibration_error_value(error.to_string()))?
                else {
                    return Ok(None);
                };
                sampled
            };
        let has_distributed_selected_resources = distributed_execution_plan
            .dispatches
            .iter()
            .any(|dispatch| !dispatch.selected_resource_partitions.is_empty());
        if has_distributed_selected_resources {
            if phase != VulkanTargetedComponentExecutionPhase::Decode {
                // Distributed component-batch execution binds dynamic tables,
                // but it does not yet publish and resume exact per-lane misses.
                // Treat that physical candidate as unavailable rather than
                // measuring zero-address expert skips as valid prefill.
                return Ok(None);
            }
            let target_selector_ids = targeted_demand_selector_ids(
                &contract.selectors,
                &placed_model.execution_scope,
                &target.component_id,
            );
            let distributed_selector_ids = distributed_execution_plan
                .dispatches
                .iter()
                .flat_map(|dispatch| &dispatch.selected_resource_partitions)
                .map(|partition| partition.selector_id.clone())
                .collect::<BTreeSet<_>>();
            if target_selector_ids != distributed_selector_ids {
                return Ok(None);
            }
        }
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
                let sampled_rows = sampled.shards.iter().try_fold(0usize, |total, shard| {
                    total.checked_add(shard.row_count).ok_or_else(|| {
                        distributed_calibration_error_value(
                            "sampled distributed work rows overflowed",
                        )
                    })
                })?;
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
        let sampled_workload = distributed_execution_plan
            .dispatches
            .iter()
            .any(|dispatch| {
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
        let parameter_plan = VulkanDistributedParameterAllocationPlan::from_sampled_execution_plan(
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
            .ok_or_else(|| {
                distributed_calibration_error_value(
                    "distributed calibration total parameter bytes overflowed",
                )
            })?;
        if total_resident_parameter_bytes > maximum_total_resident_parameter_bytes
            || resident_parameter_bytes_by_device
                .iter()
                .any(|(device_id, bytes)| {
                    maximum_resident_parameter_bytes_by_logical_device
                        .get(device_id)
                        .is_none_or(|capacity| bytes > capacity)
                })
        {
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
        let mut targeted = if has_distributed_selected_resources {
            targeted_plan.materialize_static_excluding_tensors(
                owner_device,
                &parameter_pool,
                &owner_exclusions,
            )?
        } else {
            targeted_plan.materialize_excluding_tensors(
                owner_device,
                manifest_dir,
                &parameter_pool,
                &owner_exclusions,
            )?
        };
        let remaining_dynamic_parameter_budget = maximum_total_resident_parameter_bytes
            .checked_sub(total_resident_parameter_bytes)
            .expect("total static parameter bytes were bounded above");
        let remaining_dynamic_parameter_bytes_by_device =
            maximum_resident_parameter_bytes_by_logical_device
                .iter()
                .map(|(device_id, capacity)| {
                    let static_bytes = resident_parameter_bytes_by_device
                        .get(device_id)
                        .copied()
                        .unwrap_or(0);
                    capacity
                        .checked_sub(static_bytes)
                        .map(|remaining| (device_id.clone(), remaining))
                        .ok_or_else(|| {
                            distributed_calibration_error_value(format!(
                                "distributed calibration static parameters exceed capacity on {device_id:?}",
                            ))
                        })
                })
                .collect::<Result<BTreeMap<_, _>, _>>()?;
        let Some(selected_resource_mount) = mount_distributed_calibration_selected_resources(
            manifest_dir,
            &placed_model.execution_scope,
            &contract,
            &distributed_execution_plan,
            &logical_devices,
            remaining_dynamic_parameter_budget,
            &remaining_dynamic_parameter_bytes_by_device,
        )?
        else {
            return Ok(None);
        };
        if has_distributed_selected_resources {
            targeted.slice.dynamic_resource_buffers = Some(
                selected_resource_mount
                    .dynamic_buffers
                    .get(&owner_device_id)
                    .cloned()
                    .ok_or_else(|| {
                        distributed_calibration_error_value(
                            "distributed selected-resource calibration omitted owner buffers",
                        )
                    })?,
            );
        }
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
                    &selected_resource_mount.dynamic_buffers,
                    &selected_resource_mount.stores,
                    Some(&selected_resource_mount.transaction_predicates),
                    "target",
                    &distributed_activation_buffers,
                    &loaded_manifest,
                    |device_id| {
                        logical_devices
                            .get(device_id)
                            .map(Rc::as_ref)
                            .ok_or_else(|| {
                                format!("missing distributed calibration device {device_id:?}")
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
        let artifact_digest =
            distributed_calibration_artifact_digest(&loaded_manifest, &distributed_execution_plan)?;
        let execution_graph_digest =
            distributed_calibration_execution_graph_digest(target, &tick_plan);
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
        // A distributed activation has one physical backing allocation. It is
        // charged to the owner for external device-local memory or to host
        // memory for shared-host transport; imported peer views are aliases.
        let (mut resident_transient_bytes_by_device, mut resident_host_transient_bytes) =
            distributed_calibration_activation_backing_bytes(
                &logical_device_ids,
                activation_plan.route,
                activation_plan.allocations.iter().map(|allocation| {
                    (
                        allocation.owner_device_id.as_str(),
                        allocation.byte_capacity,
                    )
                }),
            )?;
        let owner_total = resident_transient_bytes_by_device
            .get_mut(&owner_device_id)
            .expect("distributed calibration owner was inserted");
        *owner_total = owner_total.checked_add(owner_transient).ok_or_else(|| {
            distributed_calibration_error_value(
                "distributed owner transient byte accounting overflowed",
            )
        })?;
        for (device_id, bytes) in &selected_resource_mount.resident_transient_bytes_by_device {
            let total = resident_transient_bytes_by_device
                .get_mut(device_id)
                .ok_or_else(|| {
                    distributed_calibration_error_value(format!(
                        "selected-resource transient allocation references unknown device {device_id:?}",
                    ))
                })?;
            *total = total.checked_add(*bytes).ok_or_else(|| {
                distributed_calibration_error_value(
                    "selected-resource device transient byte accounting overflowed",
                )
            })?;
        }
        resident_host_transient_bytes = resident_host_transient_bytes
            .checked_add(selected_resource_mount.resident_host_transient_bytes)
            .ok_or_else(|| {
                distributed_calibration_error_value(
                    "selected-resource host transient byte accounting overflowed",
                )
            })?;

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
                    &VulkanRuntimePhysicalExecutionPlan::uniform(&placed_model),
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
                        &selected_resource_mount.dynamic_buffers,
                    )
                    .map_err(|error| distributed_calibration_error_value(error.to_string()))?,
                )
            }
        };
        if let Some(prefill_runner) = &prefill_runner {
            for (device_id, bytes) in prefill_runner
                .resident_transient_bytes_by_device()
                .map_err(|error| distributed_calibration_error_value(error.to_string()))?
            {
                let total = resident_transient_bytes_by_device
                    .get_mut(&device_id)
                    .ok_or_else(|| {
                        distributed_calibration_error_value(format!(
                            "distributed prefill transient allocation references unknown device {device_id:?}",
                        ))
                    })?;
                *total = total.checked_add(bytes).ok_or_else(|| {
                    distributed_calibration_error_value(
                        "distributed prefill transient byte accounting overflowed",
                    )
                })?;
            }
        }

        let execution_case = distributed_calibration_execution_case(
            &devices,
            &logical_device_ids,
            &distributed_execution_plan,
            &loaded_manifest,
            target.signature_id.clone(),
            artifact_digest,
            execution_graph_digest,
            phase,
            &dispatch_work,
        )?;
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
            distributed_resource_stores: selected_resource_mount.stores,
            _distributed_dynamic_resource_buffers: selected_resource_mount.dynamic_buffers,
            _distributed_transaction_predicates: selected_resource_mount.transaction_predicates,
            _parameter_pool: parameter_pool,
            resident_parameter_bytes_by_device,
            resident_transient_bytes_by_device,
            resident_host_transient_bytes,
            activation_routes,
            dispatch_work,
            sampled_workload,
            sample_fraction_millionths,
            execution_case,
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
            } => self.execute_prefill(useful_units, activation_batch_width, seed, maximum_duration),
        }
    }

    fn current_resident_parameter_bytes_by_device(
        &self,
    ) -> Result<BTreeMap<String, usize>, VulkanResidentTokenModelPackageError> {
        let mut current = self.resident_parameter_bytes_by_device.clone();
        for (device_id, store) in &self.distributed_resource_stores {
            let store_report = store
                .residency_report()
                .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
            let resident_parameter_bytes = store_report
                .current_device_bytes
                .checked_sub(store_report.metadata_device_bytes)
                .ok_or_else(|| {
                    distributed_calibration_error_value(
                        "distributed resource store device-byte accounting underflowed",
                    )
                })?;
            let total = current.get_mut(device_id).ok_or_else(|| {
                distributed_calibration_error_value(format!(
                    "distributed resource store references unknown calibration device {device_id:?}",
                ))
            })?;
            *total = total.checked_add(resident_parameter_bytes).ok_or_else(|| {
                distributed_calibration_error_value(
                    "distributed calibration current parameter bytes overflowed",
                )
            })?;
        }
        Ok(current)
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
        let schedule = VulkanMountedPlacedResidentInProcessSchedule::from_tick_plans(&[&self
            .placed_slice
            .resident_execution_plan
            .tick_plan])
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
        let execution_ns = windows.iter().fold(0u64, |total, window| {
            total.saturating_add(window.duration_ns)
        });
        let captured_output = distributed_calibration_output_artifact(
            &self.placed_slice.mounted,
            &self.terminal_dispatch,
            self.execution_case
                .behavior
                .equivalence
                .output_scalar_format
                .unwrap_or(VulkanPlacementScalarFormat::Bf16),
        )?;
        let output_digest = vulkan_placement_output_artifact_digest(&captured_output)
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let output_artifact = (self.execution_case.behavior.equivalence.output
            == VulkanPlacementEquivalenceKind::AbsoluteRelativeTolerance)
            .then_some(captured_output);
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
            output_artifact,
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
                    .map_err(|_| {
                        distributed_calibration_error_value(
                            "distributed prefill state capacity exceeds u32",
                        )
                    })?,
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
        let execution_ns = windows.iter().fold(0u64, |total, window| {
            total.saturating_add(window.duration_ns)
        });
        let captured_output = distributed_calibration_prefill_output_artifact(
            runner,
            &self.placed_slice,
            &self.terminal_dispatch,
            activation_batch_width,
            self.execution_case
                .behavior
                .equivalence
                .output_scalar_format
                .unwrap_or(VulkanPlacementScalarFormat::Bf16),
        )?;
        let output_digest = vulkan_placement_output_artifact_digest(&captured_output)
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let output_artifact = (self.execution_case.behavior.equivalence.output
            == VulkanPlacementEquivalenceKind::AbsoluteRelativeTolerance)
            .then_some(captured_output);
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
            physical_dispatch_count: repetitions
                .saturating_mul(local_dispatch_count.saturating_add(distributed_dispatch_count)),
            shard_count,
            output_digest,
            output_artifact,
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
            distributed_resource_stores,
            _distributed_dynamic_resource_buffers: distributed_dynamic_resource_buffers,
            _distributed_transaction_predicates: distributed_transaction_predicates,
            _parameter_pool: parameter_pool,
            resident_parameter_bytes_by_device: _,
            resident_transient_bytes_by_device: _,
            resident_host_transient_bytes: _,
            activation_routes: _,
            dispatch_work: _,
            sampled_workload: _,
            sample_fraction_millionths: _,
            execution_case: _,
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
        drop(distributed_dynamic_resource_buffers);
        drop(distributed_transaction_predicates);
        let mut unloaded_store_pointers = BTreeSet::new();
        for store in distributed_resource_stores.values() {
            if unloaded_store_pointers.insert(Arc::as_ptr(store) as usize)
                && let Err(error) = store.unload()
            {
                cleanup_errors.push(error.to_string());
            }
        }
        drop(distributed_resource_stores);
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
                .ok_or_else(|| {
                    distributed_calibration_error_value(
                        "distributed prefill fixture size overflowed",
                    )
                })?;
            signal
                .buffer
                .write_bytes(&targeted_fixture_bytes(
                    byte_count,
                    seed.wrapping_add(u32::try_from(dispatch.dispatch_index).unwrap_or(u32::MAX)),
                    descriptor.binding,
                ))
                .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        }
    }
    Ok(())
}

fn distributed_calibration_prefill_output_artifact(
    runner: &VulkanResidentPlacedComponentBatchRunner,
    placed_slice: &VulkanResidentInProcessPlacedStreamProcessorDevice,
    terminal: &VulkanMountedPlacedBoundDispatch,
    activation_batch_width: usize,
    scalar_format: VulkanPlacementScalarFormat,
) -> Result<VulkanPlacementOutputArtifact, VulkanResidentTokenModelPackageError> {
    let batch_slice = runner
        .slice(0)
        .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
    let mut segments = Vec::new();
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
            .ok_or_else(|| {
                distributed_calibration_error_value("distributed prefill output size overflowed")
            })?;
        segments.push(VulkanPlacementOutputSegment {
            binding: descriptor.binding,
            name: descriptor.name.clone(),
            bytes: batch_slice
                .signal_buffer(&key)
                .map_err(|error| distributed_calibration_error_value(error.to_string()))?
                .buffer
                .read_bytes(byte_count)
                .map_err(|error| distributed_calibration_error_value(error.to_string()))?,
        });
    }
    segments.sort_by(|left, right| {
        (left.binding, left.name.as_str()).cmp(&(right.binding, right.name.as_str()))
    });
    Ok(VulkanPlacementOutputArtifact {
        scalar_format,
        segments,
    })
}

fn distributed_calibration_output_artifact(
    mounted: &VulkanMountedPlacedStreamCircuit,
    terminal: &VulkanMountedPlacedBoundDispatch,
    scalar_format: VulkanPlacementScalarFormat,
) -> Result<VulkanPlacementOutputArtifact, VulkanResidentTokenModelPackageError> {
    let mut segments = Vec::new();
    for descriptor in terminal
        .descriptors
        .iter()
        .filter(|descriptor| descriptor.usage == VulkanKernelDescriptorUsage::OutputSignal)
    {
        let binding = mounted
            .resident_kernel_buffer_binding(terminal, descriptor)
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        segments.push(VulkanPlacementOutputSegment {
            binding: descriptor.binding,
            name: descriptor.name.clone(),
            bytes: binding
                .buffer
                .read_bytes_at(
                    binding.byte_offset,
                    targeted_signal_byte_count(descriptor, binding.byte_len),
                )
                .map_err(|error| distributed_calibration_error_value(error.to_string()))?,
        });
    }
    segments.sort_by(|left, right| {
        (left.binding, left.name.as_str()).cmp(&(right.binding, right.name.as_str()))
    });
    Ok(VulkanPlacementOutputArtifact {
        scalar_format,
        segments,
    })
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

#[cfg(test)]
mod runtime_distributed_placement_calibration_strategy_tests {
    use super::*;

    fn digest_target(component_id: &str, node_id: &str) -> VulkanRuntimePlacementCalibrationTarget {
        VulkanRuntimePlacementCalibrationTarget {
            signature_id: "same-physical-signature".to_string(),
            component_id: component_id.to_string(),
            component_ids: vec![component_id.to_string()],
            terminal_node_id: node_id.to_string(),
            implementation: "physical-implementation".to_string(),
            planned_resident_parameter_bytes: 0,
        }
    }

    fn digest_plan(
        component_id: &str,
        node_id: &str,
        edge_bytes: usize,
    ) -> VulkanMountedPlacedStreamTickPlan {
        VulkanMountedPlacedStreamTickPlan {
            backend_id: "vulkan".to_string(),
            device_id: "semantic-device-label".to_string(),
            stages: vec![VulkanMountedPlacedStreamTickStage::Dispatch {
                stage_index: 17,
                dispatch: VulkanMountedPlacedStreamTickDispatch {
                    dispatch_index: 91,
                    kernel_id: format!("kernel-for-{component_id}"),
                    component_id: component_id.to_string(),
                    node_id: node_id.to_string(),
                    op: "semantic-op-label".to_string(),
                    descriptor_count: 3,
                    resident_descriptor_count: 1,
                    reads: vec![
                        VulkanMountedPlacedStreamTickIo::ModelSignal {
                            signal_id: format!("input-for-{component_id}"),
                        },
                        VulkanMountedPlacedStreamTickIo::LocalEdgeBuffer {
                            edge_index: 41,
                            buffer_index: 7,
                            byte_capacity: edge_bytes,
                        },
                    ],
                    writes: vec![VulkanMountedPlacedStreamTickIo::ModelSignal {
                        signal_id: format!("output-for-{component_id}"),
                    }],
                },
            }],
            stage_count: 1,
            receive_stage_count: 0,
            dispatch_stage_count: 1,
            publish_stage_count: 0,
            local_edge_read_count: 1,
            local_edge_write_count: 0,
            incoming_edge_read_count: 0,
            outgoing_edge_write_count: 0,
            model_input_read_count: 1,
            model_output_write_count: 1,
            can_execute: true,
        }
    }

    #[test]
    fn execution_graph_identity_reuses_equivalent_components_without_hiding_topology() {
        let first = distributed_calibration_execution_graph_digest(
            &digest_target("block.0", "down.0"),
            &digest_plan("block.0", "down.0", 8_192),
        );
        let relabeled = distributed_calibration_execution_graph_digest(
            &digest_target("block.19", "down.19"),
            &digest_plan("block.19", "down.19", 8_192),
        );
        let different_topology = distributed_calibration_execution_graph_digest(
            &digest_target("block.19", "down.19"),
            &digest_plan("block.19", "down.19", 16_384),
        );

        assert_eq!(first, relabeled);
        assert_ne!(first, different_topology);
    }

    #[test]
    fn selected_resource_identity_uses_partition_ordinals_not_runtime_selector_ids() {
        let first = distributed_calibration_normalized_selected_resource_indices(
            ["layer.0.router", "layer.0.shared"],
            &BTreeMap::from([
                ("layer.0.router".to_string(), vec![0, 2, 4]),
                ("layer.0.shared".to_string(), vec![0]),
            ]),
            "gpu0",
        )
        .unwrap();
        let relabeled = distributed_calibration_normalized_selected_resource_indices(
            ["layer.19.router", "layer.19.shared"],
            &BTreeMap::from([
                ("layer.19.router".to_string(), vec![0, 2, 4]),
                ("layer.19.shared".to_string(), vec![0]),
            ]),
            "gpu0",
        )
        .unwrap();

        assert_eq!(first, relabeled);
        assert_eq!(first[&0], [0, 2, 4]);
        assert!(
            distributed_calibration_normalized_selected_resource_indices(
                ["layer.19.router", "layer.19.shared"],
                &BTreeMap::from([("layer.19.router".to_string(), vec![0, 2, 4])]),
                "gpu0",
            )
            .unwrap_err()
            .to_string()
            .contains("partition ordinal 1")
        );
    }

    fn artifact_digest(artifact: &VulkanLoadedPhysicalKernelArtifact) -> String {
        let mut digest = Sha256::new();
        distributed_calibration_update_artifact_digest(&mut digest, artifact).unwrap();
        format!("sha256:{:x}", digest.finalize())
    }

    fn loaded_artifact(
        artifact_id: &str,
        op: &str,
        path: &str,
        words: Vec<u32>,
    ) -> VulkanLoadedPhysicalKernelArtifact {
        VulkanLoadedPhysicalKernelArtifact {
            artifact: VulkanPhysicalKernelArtifact {
                artifact_id: artifact_id.to_string(),
                op: op.to_string(),
                path: path.to_string(),
                entry_point: "main".to_string(),
                local_size_x: 256,
                workgroup_count_x: 8,
                descriptor_signature: Vec::new(),
                push_constants: Vec::new(),
                stream_control_binding: None,
            },
            resolved_path: PathBuf::from(path),
            words,
        }
    }

    #[test]
    fn artifact_identity_uses_executable_bytes_and_interface_not_semantic_labels() {
        let first = loaded_artifact("block.0.down", "down", "block.0/down.spv", vec![1, 2, 3]);
        let relabeled = loaded_artifact(
            "block.19.projection",
            "semantic-projection-label",
            "another/path.spv",
            vec![1, 2, 3],
        );
        let different_binary = loaded_artifact(
            "block.19.projection",
            "semantic-projection-label",
            "another/path.spv",
            vec![1, 2, 4],
        );

        assert_eq!(artifact_digest(&first), artifact_digest(&relabeled));
        assert_ne!(artifact_digest(&first), artifact_digest(&different_binary));
    }

    #[test]
    fn calibration_selects_contracts_by_phase_and_lane_shape() {
        assert_eq!(
            distributed_contract_phase_and_shape(VulkanTargetedComponentExecutionPhase::Decode,),
            (
                nerve_execution_contracts::ExecutionPhase::Decode,
                nerve_execution_contracts::ExecutionShape::SingleLane,
            ),
        );
        assert_eq!(
            distributed_contract_phase_and_shape(VulkanTargetedComponentExecutionPhase::Prefill {
                activation_batch_width: 64,
            },),
            (
                nerve_execution_contracts::ExecutionPhase::Prefill,
                nerve_execution_contracts::ExecutionShape::MultiLane,
            ),
        );
    }

    #[test]
    fn classifies_physical_strategy_from_partition_contracts() {
        assert_eq!(
            vulkan_distributed_placement_strategy(
                1,
                [nerve_execution_contracts::ExecutionStrategy::ExpertParallel],
            )
            .unwrap(),
            VulkanPlacementExecutionStrategy::SingleDevice,
        );
        assert_eq!(
            vulkan_distributed_placement_strategy(
                2,
                [nerve_execution_contracts::ExecutionStrategy::TensorParallel],
            )
            .unwrap(),
            VulkanPlacementExecutionStrategy::TensorParallel,
        );
        assert_eq!(
            vulkan_distributed_placement_strategy(
                2,
                [nerve_execution_contracts::ExecutionStrategy::TensorParallelExpert],
            )
            .unwrap(),
            VulkanPlacementExecutionStrategy::IntraExpertTensorParallel,
        );
        assert_eq!(
            vulkan_distributed_placement_strategy(
                3,
                [nerve_execution_contracts::ExecutionStrategy::ExpertParallel],
            )
            .unwrap(),
            VulkanPlacementExecutionStrategy::WholeExpertParallel,
        );
        assert_eq!(
            vulkan_distributed_placement_strategy(
                4,
                [
                    nerve_execution_contracts::ExecutionStrategy::ExpertParallel,
                    nerve_execution_contracts::ExecutionStrategy::TensorParallel,
                ],
            )
            .unwrap(),
            VulkanPlacementExecutionStrategy::Hybrid,
        );
    }

    #[test]
    fn rejects_an_empty_physical_strategy() {
        let no_devices = vulkan_distributed_placement_strategy(
            0,
            [nerve_execution_contracts::ExecutionStrategy::TensorParallel],
        )
        .unwrap_err();
        assert!(no_devices.to_string().contains("physical device"));

        let no_dispatches =
            vulkan_distributed_placement_strategy(1, std::iter::empty()).unwrap_err();
        assert!(no_dispatches.to_string().contains("partitioned dispatch"));
    }

    #[test]
    fn records_complete_f32_reduction_geometry_in_the_behavior_shape() {
        let reduction = VulkanDistributedReductionPlan {
            operation: nerve_execution_contracts::ReductionOperation::SumF32,
            element_count: 4096,
            partial_byte_capacity: 4096 * size_of::<f32>(),
            finalization: VulkanDistributedReductionFinalizationPlan::StoreF32,
        };
        assert_eq!(
            distributed_calibration_reduction_geometry("contract", Some(&reduction), 3).unwrap(),
            Some(VulkanPlacementOperationGeometry::Reduction {
                contract_id: "contract".to_string(),
                element_count: 4096,
                element_byte_count: size_of::<f32>(),
                participant_count: 3,
            })
        );
        assert!(
            distributed_calibration_reduction_geometry("contract", Some(&reduction), 1).is_err()
        );
        assert_eq!(
            distributed_calibration_reduction_geometry("contract", None, 1).unwrap(),
            None
        );
    }

    #[test]
    fn preserves_only_a_compiler_declared_tolerant_terminal_output() {
        let exact = crate::VulkanDistributedEquivalencePlan {
            output: VulkanDistributedEquivalenceKind::BitExact,
            state: VulkanDistributedEquivalenceKind::BitExact,
            absolute_tolerance_bits: None,
            relative_tolerance_bits: None,
        };
        let tolerant = crate::VulkanDistributedEquivalencePlan {
            output: VulkanDistributedEquivalenceKind::AbsoluteRelativeTolerance,
            state: VulkanDistributedEquivalenceKind::BitExact,
            absolute_tolerance_bits: Some(0.01f64.to_bits()),
            relative_tolerance_bits: Some(0.02f64.to_bits()),
        };
        let accepted = distributed_calibration_equivalence_from_contracts(&[
            (exact.clone(), None),
            (
                tolerant.clone(),
                Some(
                    VulkanDistributedReductionFinalizationPlan::AddBf16ResidualToBf16 {
                        residual_input_index: 1,
                    },
                ),
            ),
        ])
        .unwrap();
        assert_eq!(
            accepted.output,
            VulkanPlacementEquivalenceKind::AbsoluteRelativeTolerance,
        );
        assert_eq!(
            accepted.output_scalar_format,
            Some(VulkanPlacementScalarFormat::Bf16),
        );
        assert_eq!(accepted.absolute_tolerance(), Some(0.01));
        assert_eq!(accepted.relative_tolerance(), Some(0.02));

        assert!(
            distributed_calibration_equivalence_from_contracts(&[
                (tolerant.clone(), None),
                (exact, None),
            ])
            .unwrap_err()
            .to_string()
            .contains("tolerant intermediate")
        );
        assert!(
            distributed_calibration_equivalence_from_contracts(&[(tolerant, None)])
                .unwrap_err()
                .to_string()
                .contains("typed reduction")
        );
    }

    #[test]
    fn accounts_shared_activation_backing_on_its_actual_memory_tier() {
        let devices = vec!["gpu-a".to_string(), "gpu-b".to_string()];
        let allocations = [("gpu-a", 64usize), ("gpu-b", 32usize)];
        let (shared_host_devices, shared_host_bytes) =
            distributed_calibration_activation_backing_bytes(
                &devices,
                VulkanSharedResidentBufferRoute::SharedHost,
                allocations,
            )
            .unwrap();
        assert_eq!(shared_host_devices["gpu-a"], 0);
        assert_eq!(shared_host_devices["gpu-b"], 0);
        assert_eq!(shared_host_bytes, 96);

        let (device_local, host_bytes) = distributed_calibration_activation_backing_bytes(
            &devices,
            VulkanSharedResidentBufferRoute::ExternalDeviceLocal,
            allocations,
        )
        .unwrap();
        assert_eq!(device_local["gpu-a"], 64);
        assert_eq!(device_local["gpu-b"], 32);
        assert_eq!(host_bytes, 0);
    }

    #[test]
    fn rejects_invalid_activation_backing_accounting() {
        let devices = vec!["gpu-a".to_string()];
        assert!(
            distributed_calibration_activation_backing_bytes(
                &devices,
                VulkanSharedResidentBufferRoute::SharedHost,
                [("gpu-b", 1)],
            )
            .is_err(),
        );
        assert!(
            distributed_calibration_activation_backing_bytes(
                &devices,
                VulkanSharedResidentBufferRoute::ExternalDeviceLocal,
                [("gpu-a", 0)],
            )
            .is_err(),
        );
    }
}
