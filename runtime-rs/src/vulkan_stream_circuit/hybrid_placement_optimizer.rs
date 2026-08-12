#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanHybridRegionCandidate {
    pub candidate_id: String,
    pub component_start: usize,
    pub component_end: usize,
    pub execution_case: VulkanPlacementExecutionCaseIdentity,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanHybridBoundaryCandidate {
    pub boundary_index: usize,
    pub byte_count: usize,
    pub execution_case: VulkanPlacementExecutionCaseIdentity,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanHybridScheduledStep {
    Boundary {
        boundary_index: usize,
        execution_case: VulkanPlacementExecutionCaseIdentity,
    },
    Region {
        candidate_id: String,
        component_start: usize,
        component_end: usize,
        execution_case: VulkanPlacementExecutionCaseIdentity,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanHybridPlacementPlan {
    pub steps: Vec<VulkanHybridScheduledStep>,
    pub predicted_duration_ns_per_activation: u128,
    pub resident_bytes_by_device: BTreeMap<VulkanPlacementDeviceExecutionIdentity, usize>,
    pub host_resident_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanHybridPlacementError(pub String);

impl Display for VulkanHybridPlacementError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for VulkanHybridPlacementError {}

#[derive(Clone)]
struct VulkanHybridResolvedRegionCandidate<'a> {
    request: &'a VulkanHybridRegionCandidate,
    observation: &'a VulkanPlacementCalibrationObservation,
}

#[derive(Clone)]
struct VulkanHybridResolvedBoundaryCandidate<'a> {
    request: &'a VulkanHybridBoundaryCandidate,
    observation: &'a VulkanPlacementCalibrationObservation,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanHybridPlacementState {
    cursor: usize,
    output_physical_device_id: Option<String>,
    steps: Vec<VulkanHybridScheduledStep>,
    predicted_duration_ns_per_activation: u128,
    resident_bytes_by_device: BTreeMap<VulkanPlacementDeviceExecutionIdentity, usize>,
    host_resident_bytes: usize,
}

/// Selects exact measured physical islands for a canonical ordered graph.
///
/// Every request is merely a legal candidate identity. Missing catalog
/// evidence, a stale device identity, an unmeasured cross-device boundary, or
/// insufficient reservation-aware capacity makes that path unavailable. The
/// dynamic program retains partial plans with distinct output placement and
/// non-dominated resource vectors, so a locally slower island can still win by
/// avoiding a later transfer or capacity dead end.
pub fn plan_vulkan_hybrid_ordered_graph(
    catalog: &VulkanPlacementCalibrationCatalog,
    component_count: usize,
    region_candidates: &[VulkanHybridRegionCandidate],
    boundary_candidates: &[VulkanHybridBoundaryCandidate],
    capacity: &VulkanPlacementCapacityEnvelope,
) -> Result<VulkanHybridPlacementPlan, VulkanHybridPlacementError> {
    try_plan_vulkan_hybrid_ordered_graph(
        catalog,
        component_count,
        region_candidates,
        boundary_candidates,
        capacity,
    )?
    .ok_or_else(|| {
        VulkanHybridPlacementError(
            "no exact measured hybrid placement covers the graph within current capacity"
                .to_string(),
        )
    })
}

/// Returns `None` when all structurally valid, exactly measured candidates are
/// unavailable for the current device identities, boundaries, or capacity.
/// Invalid evidence remains an error. This distinction lets a product runtime
/// preserve its valid single-device/serialized placement instead of treating a
/// stale optional optimization catalog as a fatal model-load failure.
pub fn try_plan_vulkan_hybrid_ordered_graph(
    catalog: &VulkanPlacementCalibrationCatalog,
    component_count: usize,
    region_candidates: &[VulkanHybridRegionCandidate],
    boundary_candidates: &[VulkanHybridBoundaryCandidate],
    capacity: &VulkanPlacementCapacityEnvelope,
) -> Result<Option<VulkanHybridPlacementPlan>, VulkanHybridPlacementError> {
    if component_count == 0 {
        return Err(VulkanHybridPlacementError(
            "hybrid placement requires a nonempty ordered graph".to_string(),
        ));
    }
    validate_hybrid_capacity_envelope(capacity)?;
    if region_candidates.is_empty() {
        return Ok(None);
    }

    let mut candidate_ids = BTreeSet::new();
    let mut expected_phase = None;
    let mut behavior_by_range = BTreeMap::new();
    let mut regions_by_start = BTreeMap::<usize, Vec<VulkanHybridResolvedRegionCandidate>>::new();
    for candidate in region_candidates {
        if candidate.candidate_id.is_empty()
            || !candidate_ids.insert(candidate.candidate_id.as_str())
            || candidate.component_start >= candidate.component_end
            || candidate.component_end > component_count
        {
            return Err(VulkanHybridPlacementError(
                "hybrid region candidates require unique IDs and valid nonempty graph ranges"
                    .to_string(),
            ));
        }
        let Some(observation) = catalog.exact_observation(&candidate.execution_case) else {
            continue;
        };
        if !matches!(
            observation.execution_case.strategy,
            VulkanPlacementExecutionStrategy::SingleDevice
                | VulkanPlacementExecutionStrategy::Serialized
                | VulkanPlacementExecutionStrategy::TensorParallel
                | VulkanPlacementExecutionStrategy::WholeExpertParallel
                | VulkanPlacementExecutionStrategy::IntraExpertTensorParallel
                | VulkanPlacementExecutionStrategy::Hybrid
        ) {
            return Err(VulkanHybridPlacementError(format!(
                "hybrid region candidate {:?} has non-region strategy {:?}",
                candidate.candidate_id, observation.execution_case.strategy,
            )));
        }
        let range = (candidate.component_start, candidate.component_end);
        if behavior_by_range
            .get(&range)
            .is_some_and(|behavior| behavior != &observation.execution_case.behavior)
        {
            return Err(VulkanHybridPlacementError(format!(
                "hybrid candidates for graph range {:?} do not share one exact behavior identity",
                range,
            )));
        }
        behavior_by_range
            .entry(range)
            .or_insert_with(|| observation.execution_case.behavior.clone());
        validate_hybrid_phase(
            &mut expected_phase,
            observation.execution_case.behavior.phase,
        )?;
        regions_by_start
            .entry(candidate.component_start)
            .or_default()
            .push(VulkanHybridResolvedRegionCandidate {
                request: candidate,
                observation,
            });
    }
    for candidates in regions_by_start.values_mut() {
        candidates
            .sort_by(|left, right| left.request.candidate_id.cmp(&right.request.candidate_id));
    }

    let mut boundaries_by_index =
        BTreeMap::<usize, Vec<VulkanHybridResolvedBoundaryCandidate>>::new();
    for candidate in boundary_candidates {
        if candidate.boundary_index >= component_count.saturating_sub(1)
            || candidate.byte_count == 0
        {
            return Err(VulkanHybridPlacementError(format!(
                "hybrid boundary index {} falls outside a {component_count}-component graph",
                candidate.boundary_index,
            )));
        }
        let Some(observation) = catalog.exact_observation(&candidate.execution_case) else {
            continue;
        };
        if observation.execution_case.strategy != VulkanPlacementExecutionStrategy::DirectedBoundary
            || !is_exact_directed_boundary_observation(observation, candidate.byte_count)
        {
            return Err(VulkanHybridPlacementError(format!(
                "hybrid boundary {} does not reference one complete directed-boundary transaction",
                candidate.boundary_index,
            )));
        }
        validate_hybrid_phase(
            &mut expected_phase,
            observation.execution_case.behavior.phase,
        )?;
        boundaries_by_index
            .entry(candidate.boundary_index)
            .or_default()
            .push(VulkanHybridResolvedBoundaryCandidate {
                request: candidate,
                observation,
            });
    }

    let mut states_by_cursor = vec![Vec::<VulkanHybridPlacementState>::new(); component_count + 1];
    states_by_cursor[0].push(VulkanHybridPlacementState {
        cursor: 0,
        output_physical_device_id: None,
        steps: Vec::new(),
        predicted_duration_ns_per_activation: 0,
        resident_bytes_by_device: BTreeMap::new(),
        host_resident_bytes: 0,
    });

    for cursor in 0..component_count {
        let states = std::mem::take(&mut states_by_cursor[cursor]);
        let Some(candidates) = regions_by_start.get(&cursor) else {
            continue;
        };
        for state in states {
            for candidate in candidates {
                let boundary_options = matching_hybrid_boundary_options(
                    &state,
                    candidate.observation,
                    cursor,
                    &boundaries_by_index,
                );
                for boundary in boundary_options {
                    let Some(mut next) = apply_hybrid_boundary(&state, boundary, capacity)? else {
                        continue;
                    };
                    let Some((resident_bytes_by_device, host_resident_bytes)) =
                        reserve_hybrid_observation_resources(
                            &next,
                            candidate.observation,
                            capacity,
                        )?
                    else {
                        continue;
                    };
                    next.predicted_duration_ns_per_activation = next
                        .predicted_duration_ns_per_activation
                        .checked_add(normalized_hybrid_duration_ns(candidate.observation)?)
                        .ok_or_else(|| {
                            VulkanHybridPlacementError(
                                "hybrid placement predicted duration overflowed".to_string(),
                            )
                        })?;
                    next.resident_bytes_by_device = resident_bytes_by_device;
                    next.host_resident_bytes = host_resident_bytes;
                    next.cursor = candidate.request.component_end;
                    next.output_physical_device_id = Some(
                        candidate
                            .observation
                            .execution_case
                            .output_physical_device_id
                            .clone(),
                    );
                    next.steps.push(VulkanHybridScheduledStep::Region {
                        candidate_id: candidate.request.candidate_id.clone(),
                        component_start: candidate.request.component_start,
                        component_end: candidate.request.component_end,
                        execution_case: candidate.observation.execution_case.clone(),
                    });
                    insert_hybrid_pareto_state(
                        &mut states_by_cursor[candidate.request.component_end],
                        next,
                    );
                }
            }
        }
    }

    let Some(best) = states_by_cursor
        .pop()
        .expect("a state bucket exists for the graph terminus")
        .into_iter()
        .min_by(|left, right| {
            hybrid_state_ordering_key(left).cmp(&hybrid_state_ordering_key(right))
        })
    else {
        return Ok(None);
    };
    Ok(Some(VulkanHybridPlacementPlan {
        steps: best.steps,
        predicted_duration_ns_per_activation: best.predicted_duration_ns_per_activation,
        resident_bytes_by_device: best.resident_bytes_by_device,
        host_resident_bytes: best.host_resident_bytes,
    }))
}

fn validate_hybrid_capacity_envelope(
    capacity: &VulkanPlacementCapacityEnvelope,
) -> Result<(), VulkanHybridPlacementError> {
    let mut physical_ids = BTreeSet::new();
    if capacity.available_bytes_by_device.is_empty()
        || capacity
            .available_bytes_by_device
            .iter()
            .any(|(device, bytes)| {
                device.physical_device_id.is_empty()
                    || !physical_ids.insert(device.physical_device_id.as_str())
                    || *bytes == 0
            })
    {
        return Err(VulkanHybridPlacementError(
            "hybrid capacity requires unique exact device identities and positive available bytes"
                .to_string(),
        ));
    }
    Ok(())
}

fn validate_hybrid_phase(
    expected: &mut Option<nerve_execution_contracts::ExecutionPhase>,
    actual: nerve_execution_contracts::ExecutionPhase,
) -> Result<(), VulkanHybridPlacementError> {
    if expected.is_some_and(|phase| phase != actual) {
        return Err(VulkanHybridPlacementError(
            "one hybrid solve cannot mix decode and prefill evidence".to_string(),
        ));
    }
    *expected = Some(actual);
    Ok(())
}

fn is_exact_directed_boundary_observation(
    observation: &VulkanPlacementCalibrationObservation,
    expected_byte_count: usize,
) -> bool {
    matches!(
        observation.execution_case.operations.as_slice(),
        [VulkanPlacementOperationGeometry::DirectedTransfer { byte_count, .. }]
            if *byte_count == expected_byte_count
    )
}

fn matching_hybrid_boundary_options<'a>(
    state: &VulkanHybridPlacementState,
    next: &VulkanPlacementCalibrationObservation,
    cursor: usize,
    boundaries_by_index: &'a BTreeMap<usize, Vec<VulkanHybridResolvedBoundaryCandidate<'a>>>,
) -> Vec<Option<&'a VulkanHybridResolvedBoundaryCandidate<'a>>> {
    let Some(source) = state.output_physical_device_id.as_deref() else {
        return vec![None];
    };
    let destination = next.execution_case.input_physical_device_id.as_str();
    if source == destination {
        return vec![None];
    }
    boundaries_by_index
        .get(&(cursor - 1))
        .into_iter()
        .flatten()
        .filter(|boundary| {
            boundary.observation.execution_case.input_physical_device_id == source
                && boundary
                    .observation
                    .execution_case
                    .output_physical_device_id
                    == destination
        })
        .map(Some)
        .collect()
}

fn apply_hybrid_boundary(
    state: &VulkanHybridPlacementState,
    boundary: Option<&VulkanHybridResolvedBoundaryCandidate>,
    capacity: &VulkanPlacementCapacityEnvelope,
) -> Result<Option<VulkanHybridPlacementState>, VulkanHybridPlacementError> {
    let Some(boundary) = boundary else {
        return Ok(Some(state.clone()));
    };
    let Some((resident_bytes_by_device, host_resident_bytes)) =
        reserve_hybrid_observation_resources(state, boundary.observation, capacity)?
    else {
        return Ok(None);
    };
    let mut next = state.clone();
    next.predicted_duration_ns_per_activation = next
        .predicted_duration_ns_per_activation
        .checked_add(normalized_hybrid_duration_ns(boundary.observation)?)
        .ok_or_else(|| {
            VulkanHybridPlacementError("hybrid boundary predicted duration overflowed".to_string())
        })?;
    next.resident_bytes_by_device = resident_bytes_by_device;
    next.host_resident_bytes = host_resident_bytes;
    next.steps.push(VulkanHybridScheduledStep::Boundary {
        boundary_index: boundary.request.boundary_index,
        execution_case: boundary.observation.execution_case.clone(),
    });
    Ok(Some(next))
}

fn reserve_hybrid_observation_resources(
    state: &VulkanHybridPlacementState,
    observation: &VulkanPlacementCalibrationObservation,
    capacity: &VulkanPlacementCapacityEnvelope,
) -> Result<
    Option<(
        BTreeMap<VulkanPlacementDeviceExecutionIdentity, usize>,
        usize,
    )>,
    VulkanHybridPlacementError,
> {
    let mut resident_bytes_by_device = state.resident_bytes_by_device.clone();
    for device in &observation.execution_case.devices {
        let Some(available) = capacity.available_bytes_by_device.get(device).copied() else {
            return Ok(None);
        };
        let resident = observation
            .resident_bytes_by_physical_device
            .get(&device.physical_device_id)
            .copied()
            .ok_or_else(|| {
                VulkanHybridPlacementError(
                    "validated placement observation lost device resident-byte evidence"
                        .to_string(),
                )
            })?;
        let transient = observation
            .transient_peak_bytes_by_physical_device
            .get(&device.physical_device_id)
            .copied()
            .ok_or_else(|| {
                VulkanHybridPlacementError(
                    "validated placement observation lost device transient-byte evidence"
                        .to_string(),
                )
            })?;
        let retained = resident_bytes_by_device
            .get(device)
            .copied()
            .unwrap_or(0)
            .checked_add(resident)
            .ok_or_else(|| {
                VulkanHybridPlacementError(
                    "hybrid device resident-byte accounting overflowed".to_string(),
                )
            })?;
        if retained
            .checked_add(transient)
            .is_none_or(|required| required > available)
        {
            return Ok(None);
        }
        resident_bytes_by_device.insert(device.clone(), retained);
    }
    let host_resident_bytes = state
        .host_resident_bytes
        .checked_add(observation.host_resident_bytes)
        .ok_or_else(|| {
            VulkanHybridPlacementError(
                "hybrid host resident-byte accounting overflowed".to_string(),
            )
        })?;
    if host_resident_bytes
        .checked_add(observation.host_transient_peak_bytes)
        .is_none_or(|required| required > capacity.host_available_bytes)
    {
        return Ok(None);
    }
    Ok(Some((resident_bytes_by_device, host_resident_bytes)))
}

fn normalized_hybrid_duration_ns(
    observation: &VulkanPlacementCalibrationObservation,
) -> Result<u128, VulkanHybridPlacementError> {
    let useful = u128::try_from(observation.useful_activation_count).map_err(|_| {
        VulkanHybridPlacementError(
            "placement useful activation count does not fit u128".to_string(),
        )
    })?;
    if useful == 0 || observation.duration_ns == 0 {
        return Err(VulkanHybridPlacementError(
            "placement timing requires positive duration and useful work".to_string(),
        ));
    }
    Ok(u128::from(observation.duration_ns).div_ceil(useful))
}

fn insert_hybrid_pareto_state(
    states: &mut Vec<VulkanHybridPlacementState>,
    proposed: VulkanHybridPlacementState,
) {
    if states
        .iter()
        .any(|current| hybrid_state_dominates(current, &proposed))
    {
        return;
    }
    states.retain(|current| !hybrid_state_dominates(&proposed, current));
    states.push(proposed);
}

fn hybrid_state_dominates(
    left: &VulkanHybridPlacementState,
    right: &VulkanHybridPlacementState,
) -> bool {
    if left.cursor != right.cursor
        || left.output_physical_device_id != right.output_physical_device_id
        || left.predicted_duration_ns_per_activation > right.predicted_duration_ns_per_activation
        || left.host_resident_bytes > right.host_resident_bytes
    {
        return false;
    }
    let device_ids = left
        .resident_bytes_by_device
        .keys()
        .chain(right.resident_bytes_by_device.keys())
        .collect::<BTreeSet<_>>();
    device_ids.into_iter().all(|device| {
        left.resident_bytes_by_device
            .get(device)
            .copied()
            .unwrap_or(0)
            <= right
                .resident_bytes_by_device
                .get(device)
                .copied()
                .unwrap_or(0)
    })
}

fn hybrid_state_ordering_key(
    state: &VulkanHybridPlacementState,
) -> (u128, usize, usize, Vec<String>) {
    let resident_bytes = state
        .resident_bytes_by_device
        .values()
        .copied()
        .fold(0usize, usize::saturating_add);
    let step_ids = state
        .steps
        .iter()
        .map(|step| match step {
            VulkanHybridScheduledStep::Boundary { boundary_index, .. } => {
                format!("boundary:{boundary_index}")
            }
            VulkanHybridScheduledStep::Region { candidate_id, .. } => candidate_id.clone(),
        })
        .collect();
    (
        state.predicted_duration_ns_per_activation,
        resident_bytes,
        state.host_resident_bytes,
        step_ids,
    )
}

#[cfg(test)]
mod hybrid_placement_optimizer_tests {
    use super::*;

    fn digest(byte: char) -> String {
        format!("sha256:{}", byte.to_string().repeat(64))
    }

    fn device(
        physical_device_id: &str,
        driver_version: u32,
    ) -> VulkanPlacementDeviceExecutionIdentity {
        VulkanPlacementDeviceExecutionIdentity {
            physical_device_id: physical_device_id.to_string(),
            api_version: 1,
            driver_version,
        }
    }

    fn region_behavior(contract: &str) -> VulkanPlacementBehaviorIdentity {
        VulkanPlacementBehaviorIdentity {
            compiled_execution_signature: format!("signature:{contract}"),
            runtime_implementation_fingerprint: "runtime".to_string(),
            phase: nerve_execution_contracts::ExecutionPhase::Decode,
            shape: VulkanPlacementShapeClass {
                activation_batch_width: 1,
                input_byte_capacity: 16,
                output_byte_capacity: 16,
            },
            input_fixture_digest: digest('d'),
        }
    }

    fn boundary_behavior(contract: &str, _byte_count: usize) -> VulkanPlacementBehaviorIdentity {
        region_behavior(contract)
    }

    #[allow(clippy::too_many_arguments)]
    fn region_observation(
        behavior: VulkanPlacementBehaviorIdentity,
        strategy: VulkanPlacementExecutionStrategy,
        devices: &[(&str, usize)],
        input: &str,
        output: &str,
        owner: &str,
        duration_ns: u64,
        useful_activation_count: usize,
    ) -> VulkanPlacementCalibrationObservation {
        let contract_id = behavior.compiled_execution_signature.clone();
        let device_identities = devices
            .iter()
            .map(|(id, _)| device(id, 2))
            .collect::<Vec<_>>();
        let shards = devices
            .iter()
            .enumerate()
            .map(|(index, (id, bytes))| VulkanPlacementShardIdentity {
                dispatch_ordinal: 0,
                participant_ordinal: index,
                physical_device_id: (*id).to_string(),
                distribution: "output_rows".to_string(),
                logical_start: index,
                logical_count: 1,
                selected_resource_indices_by_partition: BTreeMap::new(),
                selected_resource_fragments_by_partition: BTreeMap::new(),
                parameter_bytes: (*bytes).max(1),
            })
            .collect::<Vec<_>>();
        VulkanPlacementCalibrationObservation {
            execution_case: VulkanPlacementExecutionCaseIdentity {
                behavior,
                contract_ids: vec![contract_id.clone()],
                implementation_digests: vec![digest('a')],
                artifact_digest: digest('b'),
                execution_graph_digest: digest('c'),
                operations: vec![VulkanPlacementOperationGeometry::Dispatch {
                    geometry: VulkanPlacementDispatchGeometry {
                        contract_id,
                        logical_extent: 8,
                        sampled_extent: 8,
                        input_width: 8,
                        workgroup_count_x: 1,
                        local_size_x: 64,
                    },
                }],
                equivalence: VulkanPlacementEquivalenceIdentity::bit_exact(),
                strategy,
                devices: device_identities,
                shards,
                input_physical_device_id: input.to_string(),
                output_physical_device_id: output.to_string(),
                owner_physical_device_id: owner.to_string(),
                transports: Vec::new(),
            },
            warmup_call_count: 1,
            measured_call_count: 1,
            complete_transaction: true,
            duration_ns,
            useful_activation_count,
            output_digest: "output".to_string(),
            output_artifact: None,
            output_equivalence: VulkanPlacementOutputEquivalenceEvidence::BitExact,
            state_digest: "state".to_string(),
            resident_bytes_by_physical_device: devices
                .iter()
                .map(|(id, bytes)| ((*id).to_string(), *bytes))
                .collect(),
            transient_peak_bytes_by_physical_device: devices
                .iter()
                .map(|(id, _)| ((*id).to_string(), 0))
                .collect(),
            host_resident_bytes: 0,
            host_transient_peak_bytes: 0,
        }
    }

    fn boundary_observation(
        source: &str,
        destination: &str,
        duration_ns: u64,
    ) -> VulkanPlacementCalibrationObservation {
        let behavior = boundary_behavior("boundary", 16);
        let mut devices = vec![device(source, 2), device(destination, 2)];
        devices.sort();
        VulkanPlacementCalibrationObservation {
            execution_case: VulkanPlacementExecutionCaseIdentity {
                behavior,
                contract_ids: vec!["boundary".to_string()],
                implementation_digests: vec![digest('a')],
                artifact_digest: digest('b'),
                execution_graph_digest: digest('c'),
                operations: vec![VulkanPlacementOperationGeometry::DirectedTransfer {
                    contract_id: "boundary".to_string(),
                    byte_count: 16,
                }],
                equivalence: VulkanPlacementEquivalenceIdentity::bit_exact(),
                strategy: VulkanPlacementExecutionStrategy::DirectedBoundary,
                devices,
                shards: Vec::new(),
                input_physical_device_id: source.to_string(),
                output_physical_device_id: destination.to_string(),
                owner_physical_device_id: source.to_string(),
                transports: vec![VulkanPlacementTransportIdentity {
                    source_physical_device_id: source.to_string(),
                    destination_physical_device_id: destination.to_string(),
                    byte_capacity: 16,
                    route: "shared_host".to_string(),
                }],
            },
            warmup_call_count: 1,
            measured_call_count: 1,
            complete_transaction: true,
            duration_ns,
            useful_activation_count: 1,
            output_digest: "output".to_string(),
            output_artifact: None,
            output_equivalence: VulkanPlacementOutputEquivalenceEvidence::BitExact,
            state_digest: "state".to_string(),
            resident_bytes_by_physical_device: BTreeMap::from([
                (source.to_string(), 0),
                (destination.to_string(), 0),
            ]),
            transient_peak_bytes_by_physical_device: BTreeMap::from([
                (source.to_string(), 0),
                (destination.to_string(), 0),
            ]),
            host_resident_bytes: 0,
            host_transient_peak_bytes: 0,
        }
    }

    fn record(
        catalog: &mut VulkanPlacementCalibrationCatalog,
        observation: VulkanPlacementCalibrationObservation,
    ) -> VulkanPlacementExecutionCaseIdentity {
        if catalog
            .canonical_reference(&observation.execution_case.behavior)
            .is_none()
        {
            catalog
                .record_reference(VulkanPlacementCanonicalReference {
                    behavior: observation.execution_case.behavior.clone(),
                    output_digest: "output".to_string(),
                    output_artifact: None,
                    state_digest: "state".to_string(),
                })
                .unwrap();
        }
        let execution_case = observation.execution_case.clone();
        catalog.record_observation(observation).unwrap();
        execution_case
    }

    fn capacity(driver_version: u32, gpu0: usize, gpu1: usize) -> VulkanPlacementCapacityEnvelope {
        VulkanPlacementCapacityEnvelope {
            available_bytes_by_device: BTreeMap::from([
                (device("gpu0", driver_version), gpu0),
                (device("gpu1", driver_version), gpu1),
            ]),
            host_available_bytes: 1_000,
        }
    }

    fn selected_region_ids(plan: &VulkanHybridPlacementPlan) -> Vec<&str> {
        plan.steps
            .iter()
            .filter_map(|step| match step {
                VulkanHybridScheduledStep::Region { candidate_id, .. } => {
                    Some(candidate_id.as_str())
                }
                VulkanHybridScheduledStep::Boundary { .. } => None,
            })
            .collect()
    }

    #[test]
    fn globally_slower_local_candidate_wins_when_it_removes_a_boundary() {
        let mut catalog = VulkanPlacementCalibrationCatalog::default();
        let first_behavior = region_behavior("first");
        let second_behavior = region_behavior("second");
        let fast_remote = record(
            &mut catalog,
            region_observation(
                first_behavior.clone(),
                VulkanPlacementExecutionStrategy::SingleDevice,
                &[("gpu0", 10)],
                "gpu0",
                "gpu0",
                "gpu0",
                8,
                1,
            ),
        );
        let slower_local = record(
            &mut catalog,
            region_observation(
                first_behavior,
                VulkanPlacementExecutionStrategy::SingleDevice,
                &[("gpu1", 10)],
                "gpu1",
                "gpu1",
                "gpu1",
                10,
                1,
            ),
        );
        let second = record(
            &mut catalog,
            region_observation(
                second_behavior,
                VulkanPlacementExecutionStrategy::SingleDevice,
                &[("gpu1", 10)],
                "gpu1",
                "gpu1",
                "gpu1",
                10,
                1,
            ),
        );
        let boundary = record(&mut catalog, boundary_observation("gpu0", "gpu1", 100));

        let plan = plan_vulkan_hybrid_ordered_graph(
            &catalog,
            2,
            &[
                VulkanHybridRegionCandidate {
                    candidate_id: "fast-remote".to_string(),
                    component_start: 0,
                    component_end: 1,
                    execution_case: fast_remote,
                },
                VulkanHybridRegionCandidate {
                    candidate_id: "slower-local".to_string(),
                    component_start: 0,
                    component_end: 1,
                    execution_case: slower_local,
                },
                VulkanHybridRegionCandidate {
                    candidate_id: "second".to_string(),
                    component_start: 1,
                    component_end: 2,
                    execution_case: second,
                },
            ],
            &[VulkanHybridBoundaryCandidate {
                boundary_index: 0,
                byte_count: 16,
                execution_case: boundary,
            }],
            &capacity(2, 100, 100),
        )
        .unwrap();

        assert_eq!(selected_region_ids(&plan), ["slower-local", "second"]);
        assert_eq!(plan.predicted_duration_ns_per_activation, 20);
        assert_eq!(plan.steps.len(), 2);
    }

    #[test]
    fn pareto_frontier_keeps_a_slower_low_residency_path_needed_later() {
        let mut catalog = VulkanPlacementCalibrationCatalog::default();
        let first_behavior = region_behavior("first");
        let second_behavior = region_behavior("second");
        let fast_large = record(
            &mut catalog,
            region_observation(
                first_behavior.clone(),
                VulkanPlacementExecutionStrategy::SingleDevice,
                &[("gpu0", 80)],
                "gpu0",
                "gpu0",
                "gpu0",
                5,
                1,
            ),
        );
        let slower_small = record(
            &mut catalog,
            region_observation(
                first_behavior,
                VulkanPlacementExecutionStrategy::TensorParallel,
                &[("gpu0", 10), ("gpu1", 10)],
                "gpu0",
                "gpu0",
                "gpu0",
                8,
                1,
            ),
        );
        let second = record(
            &mut catalog,
            region_observation(
                second_behavior,
                VulkanPlacementExecutionStrategy::SingleDevice,
                &[("gpu0", 40)],
                "gpu0",
                "gpu0",
                "gpu0",
                5,
                1,
            ),
        );

        let plan = plan_vulkan_hybrid_ordered_graph(
            &catalog,
            2,
            &[
                VulkanHybridRegionCandidate {
                    candidate_id: "fast-large".to_string(),
                    component_start: 0,
                    component_end: 1,
                    execution_case: fast_large,
                },
                VulkanHybridRegionCandidate {
                    candidate_id: "slower-small".to_string(),
                    component_start: 0,
                    component_end: 1,
                    execution_case: slower_small,
                },
                VulkanHybridRegionCandidate {
                    candidate_id: "second".to_string(),
                    component_start: 1,
                    component_end: 2,
                    execution_case: second,
                },
            ],
            &[],
            &capacity(2, 100, 100),
        )
        .unwrap();

        assert_eq!(selected_region_ids(&plan), ["slower-small", "second"]);
        assert_eq!(plan.predicted_duration_ns_per_activation, 13);
    }

    #[test]
    fn compares_complete_measurements_per_useful_activation() {
        let mut catalog = VulkanPlacementCalibrationCatalog::default();
        let behavior = region_behavior("region");
        let one_activation = record(
            &mut catalog,
            region_observation(
                behavior.clone(),
                VulkanPlacementExecutionStrategy::SingleDevice,
                &[("gpu0", 10)],
                "gpu0",
                "gpu0",
                "gpu0",
                10,
                1,
            ),
        );
        let two_activations = record(
            &mut catalog,
            region_observation(
                behavior,
                VulkanPlacementExecutionStrategy::SingleDevice,
                &[("gpu1", 10)],
                "gpu1",
                "gpu1",
                "gpu1",
                18,
                2,
            ),
        );

        let plan = plan_vulkan_hybrid_ordered_graph(
            &catalog,
            1,
            &[
                VulkanHybridRegionCandidate {
                    candidate_id: "one".to_string(),
                    component_start: 0,
                    component_end: 1,
                    execution_case: one_activation,
                },
                VulkanHybridRegionCandidate {
                    candidate_id: "two".to_string(),
                    component_start: 0,
                    component_end: 1,
                    execution_case: two_activations,
                },
            ],
            &[],
            &capacity(2, 100, 100),
        )
        .unwrap();

        assert_eq!(selected_region_ids(&plan), ["two"]);
        assert_eq!(plan.predicted_duration_ns_per_activation, 9);
    }

    #[test]
    fn missing_boundary_or_stale_device_evidence_is_unavailable_not_free() {
        let mut catalog = VulkanPlacementCalibrationCatalog::default();
        let first = record(
            &mut catalog,
            region_observation(
                region_behavior("first"),
                VulkanPlacementExecutionStrategy::SingleDevice,
                &[("gpu0", 10)],
                "gpu0",
                "gpu0",
                "gpu0",
                5,
                1,
            ),
        );
        let second = record(
            &mut catalog,
            region_observation(
                region_behavior("second"),
                VulkanPlacementExecutionStrategy::SingleDevice,
                &[("gpu1", 10)],
                "gpu1",
                "gpu1",
                "gpu1",
                5,
                1,
            ),
        );
        let candidates = [
            VulkanHybridRegionCandidate {
                candidate_id: "first".to_string(),
                component_start: 0,
                component_end: 1,
                execution_case: first,
            },
            VulkanHybridRegionCandidate {
                candidate_id: "second".to_string(),
                component_start: 1,
                component_end: 2,
                execution_case: second,
            },
        ];

        assert!(
            try_plan_vulkan_hybrid_ordered_graph(
                &catalog,
                2,
                &candidates,
                &[],
                &capacity(2, 100, 100),
            )
            .unwrap()
            .is_none()
        );

        assert!(
            plan_vulkan_hybrid_ordered_graph(
                &catalog,
                2,
                &candidates,
                &[],
                &capacity(2, 100, 100),
            )
            .unwrap_err()
            .0
                .contains("no exact measured")
        );

        assert!(
            try_plan_vulkan_hybrid_ordered_graph(
                &catalog,
                2,
                &candidates,
                &[],
                &capacity(3, 100, 100),
            )
            .unwrap()
            .is_none()
        );

        let boundary = record(&mut catalog, boundary_observation("gpu0", "gpu1", 7));
        assert!(
            plan_vulkan_hybrid_ordered_graph(
                &catalog,
                2,
                &candidates,
                &[VulkanHybridBoundaryCandidate {
                    boundary_index: 0,
                    byte_count: 32,
                    execution_case: boundary,
                }],
                &capacity(2, 100, 100),
            )
            .unwrap_err()
            .0
            .contains("directed-boundary transaction")
        );
        assert!(
            plan_vulkan_hybrid_ordered_graph(
                &catalog,
                2,
                &candidates,
                &[],
                &capacity(3, 100, 100),
            )
            .unwrap_err()
            .0
            .contains("no exact measured")
        );
    }
}
