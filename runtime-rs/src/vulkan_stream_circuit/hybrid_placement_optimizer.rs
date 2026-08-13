#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanHybridRegionCandidate {
    pub candidate_id: String,
    pub component_start: usize,
    pub component_end: usize,
    /// Explicit source-semantic equivalence class. Different compiled
    /// signatures may compete for one graph range only when the compiler's
    /// validated representation contract assigns them this same identity.
    pub semantic_contract_id: String,
    pub execution_case: VulkanPlacementExecutionCaseIdentity,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanHybridBoundaryCandidate {
    pub boundary_index: usize,
    pub byte_count: usize,
    pub execution_case: VulkanPlacementExecutionCaseIdentity,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VulkanHybridCandidateResourceCatalog {
    region_resources_by_candidate_id: BTreeMap<String, VulkanHybridCandidateResources>,
    boundary_resources_by_case:
        BTreeMap<(usize, VulkanPlacementExecutionCaseIdentity), VulkanHybridCandidateResources>,
}

impl VulkanHybridCandidateResourceCatalog {
    pub fn from_calibration(
        catalog: &VulkanPlacementCalibrationCatalog,
        region_candidates: &[VulkanHybridRegionCandidate],
        boundary_candidates: &[VulkanHybridBoundaryCandidate],
    ) -> Result<Self, VulkanHybridPlacementError> {
        let mut resources = Self::default();
        for candidate in region_candidates {
            let Some(observation) = catalog.exact_observation(&candidate.execution_case) else {
                continue;
            };
            resources.region_resources_by_candidate_id.insert(
                candidate.candidate_id.clone(),
                hybrid_calibration_candidate_resources(
                    &format!("region:{}", candidate.candidate_id),
                    observation,
                )?,
            );
        }
        for candidate in boundary_candidates {
            let Some(observation) = catalog.exact_observation(&candidate.execution_case) else {
                continue;
            };
            let key = (candidate.boundary_index, candidate.execution_case.clone());
            if resources
                .boundary_resources_by_case
                .insert(
                    key,
                    hybrid_calibration_candidate_resources(
                        &format!(
                            "boundary:{}:{}",
                            candidate.boundary_index,
                            candidate.execution_case.execution_graph_digest,
                        ),
                        observation,
                    )?,
                )
                .is_some()
            {
                return Err(VulkanHybridPlacementError(
                    "hybrid resource catalog repeats an exact boundary candidate".to_string(),
                ));
            }
        }
        Ok(resources)
    }

    pub fn replace_region_claims(
        &mut self,
        candidate_id: &str,
        claims: Vec<VulkanHybridResourceClaim>,
    ) -> Result<(), VulkanHybridPlacementError> {
        let resources = self
            .region_resources_by_candidate_id
            .get_mut(candidate_id)
            .ok_or_else(|| {
                VulkanHybridPlacementError(format!(
                    "hybrid resource catalog has no measured region candidate {candidate_id:?}",
                ))
            })?;
        *resources = VulkanHybridCandidateResources::new(claims);
        Ok(())
    }

    pub fn replace_region_resource_class_claims(
        &mut self,
        candidate_id: &str,
        class: VulkanHybridResourceClass,
        mut claims: Vec<VulkanHybridResourceClaim>,
    ) -> Result<(), VulkanHybridPlacementError> {
        if claims.iter().any(|claim| claim.class != class) {
            return Err(VulkanHybridPlacementError(format!(
                "hybrid replacement for {class:?} contains another resource class",
            )));
        }
        let resources = self
            .region_resources_by_candidate_id
            .get_mut(candidate_id)
            .ok_or_else(|| {
                VulkanHybridPlacementError(format!(
                    "hybrid resource catalog has no measured region candidate {candidate_id:?}",
                ))
            })?;
        resources.claims.retain(|claim| claim.class != class);
        resources.claims.append(&mut claims);
        Ok(())
    }
}

fn hybrid_calibration_candidate_resources(
    namespace: &str,
    observation: &VulkanPlacementCalibrationObservation,
) -> Result<VulkanHybridCandidateResources, VulkanHybridPlacementError> {
    let mut claims = Vec::new();
    for device in &observation.execution_case.devices {
        let physical_device_id = &device.physical_device_id;
        let resident = observation
            .resident_bytes_by_physical_device
            .get(physical_device_id)
            .copied()
            .ok_or_else(|| {
                VulkanHybridPlacementError(
                    "validated placement observation lost device resident-byte evidence"
                        .to_string(),
                )
            })?;
        if resident > 0 {
            claims.push(VulkanHybridResourceClaim::exclusive_device(
                format!("{namespace}:device:{physical_device_id}:resident"),
                device.clone(),
                VulkanHybridResourceClass::Permanent,
                resident,
            ));
        }
        let transient = observation
            .transient_peak_bytes_by_physical_device
            .get(physical_device_id)
            .copied()
            .ok_or_else(|| {
                VulkanHybridPlacementError(
                    "validated placement observation lost device transient-byte evidence"
                        .to_string(),
                )
            })?;
        if transient > 0 {
            claims.push(VulkanHybridResourceClaim::exclusive_device(
                format!("{namespace}:device:{physical_device_id}:transient"),
                device.clone(),
                VulkanHybridResourceClass::ExecutionTransient,
                transient,
            ));
        }
    }
    if observation.host_resident_bytes > 0 {
        claims.push(VulkanHybridResourceClaim::exclusive_host(
            format!("{namespace}:host:resident"),
            VulkanHybridResourceClass::Permanent,
            observation.host_resident_bytes,
        ));
    }
    if observation.host_transient_peak_bytes > 0 {
        claims.push(VulkanHybridResourceClaim::exclusive_host(
            format!("{namespace}:host:transient"),
            VulkanHybridResourceClass::ExecutionTransient,
            observation.host_transient_peak_bytes,
        ));
    }
    Ok(VulkanHybridCandidateResources::new(claims))
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
    pub resource_reservations: VulkanHybridResourceReservations,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanHybridPlacementRoute {
    pub steps: Vec<VulkanHybridScheduledStep>,
    pub predicted_duration_ns_per_activation: u128,
    pub calibration_resource_reservations: VulkanHybridResourceReservations,
    pub authoritative_resource_reservations: VulkanHybridResourceReservations,
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
    resources: &'a VulkanHybridCandidateResources,
}

#[derive(Clone)]
struct VulkanHybridResolvedBoundaryCandidate<'a> {
    request: &'a VulkanHybridBoundaryCandidate,
    observation: &'a VulkanPlacementCalibrationObservation,
    resources: &'a VulkanHybridCandidateResources,
}

struct VulkanHybridResolvedCandidateGraph<'a> {
    regions_by_start: BTreeMap<usize, Vec<VulkanHybridResolvedRegionCandidate<'a>>>,
    boundaries_by_index: BTreeMap<usize, Vec<VulkanHybridResolvedBoundaryCandidate<'a>>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanHybridPlacementState {
    cursor: usize,
    output_physical_device_id: Option<String>,
    steps: Vec<VulkanHybridScheduledStep>,
    predicted_duration_ns_per_activation: u128,
    resource_reservations: VulkanHybridResourceReservations,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanHybridRouteSearchState {
    cursor: usize,
    output_physical_device_id: Option<String>,
    steps: Vec<VulkanHybridScheduledStep>,
    predicted_duration_ns_per_activation: u128,
    calibration_resource_reservations: VulkanHybridResourceReservations,
    authoritative_resource_reservations: VulkanHybridResourceReservations,
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

pub fn plan_vulkan_hybrid_ordered_graph_with_resources(
    catalog: &VulkanPlacementCalibrationCatalog,
    component_count: usize,
    region_candidates: &[VulkanHybridRegionCandidate],
    boundary_candidates: &[VulkanHybridBoundaryCandidate],
    resources: &VulkanHybridCandidateResourceCatalog,
    capacity: &VulkanPlacementCapacityEnvelope,
) -> Result<VulkanHybridPlacementPlan, VulkanHybridPlacementError> {
    try_plan_vulkan_hybrid_ordered_graph_with_resources(
        catalog,
        component_count,
        region_candidates,
        boundary_candidates,
        resources,
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
    let resources = VulkanHybridCandidateResourceCatalog::from_calibration(
        catalog,
        region_candidates,
        boundary_candidates,
    )?;
    try_plan_vulkan_hybrid_ordered_graph_with_resources(
        catalog,
        component_count,
        region_candidates,
        boundary_candidates,
        &resources,
        capacity,
    )
}

pub fn try_plan_vulkan_hybrid_ordered_graph_with_resources(
    catalog: &VulkanPlacementCalibrationCatalog,
    component_count: usize,
    region_candidates: &[VulkanHybridRegionCandidate],
    boundary_candidates: &[VulkanHybridBoundaryCandidate],
    resources: &VulkanHybridCandidateResourceCatalog,
    capacity: &VulkanPlacementCapacityEnvelope,
) -> Result<Option<VulkanHybridPlacementPlan>, VulkanHybridPlacementError> {
    Ok(plan_vulkan_hybrid_ordered_graph_candidates_with_resources(
        catalog,
        component_count,
        region_candidates,
        boundary_candidates,
        resources,
        capacity,
    )?
    .into_iter()
    .next())
}

pub fn plan_vulkan_hybrid_ordered_graph_candidates_with_resources(
    catalog: &VulkanPlacementCalibrationCatalog,
    component_count: usize,
    region_candidates: &[VulkanHybridRegionCandidate],
    boundary_candidates: &[VulkanHybridBoundaryCandidate],
    resources: &VulkanHybridCandidateResourceCatalog,
    capacity: &VulkanPlacementCapacityEnvelope,
) -> Result<Vec<VulkanHybridPlacementPlan>, VulkanHybridPlacementError> {
    if component_count == 0 {
        return Err(VulkanHybridPlacementError(
            "hybrid placement requires a nonempty ordered graph".to_string(),
        ));
    }
    validate_hybrid_capacity_envelope(capacity)?;
    if region_candidates.is_empty() {
        return Ok(Vec::new());
    }

    let resolved = resolve_vulkan_hybrid_candidate_graph(
        catalog,
        component_count,
        region_candidates,
        boundary_candidates,
        resources,
    )?;

    let mut states_by_cursor = vec![Vec::<VulkanHybridPlacementState>::new(); component_count + 1];
    states_by_cursor[0].push(VulkanHybridPlacementState {
        cursor: 0,
        output_physical_device_id: None,
        steps: Vec::new(),
        predicted_duration_ns_per_activation: 0,
        resource_reservations: VulkanHybridResourceReservations::default(),
    });

    for cursor in 0..component_count {
        let states = std::mem::take(&mut states_by_cursor[cursor]);
        let Some(candidates) = resolved.regions_by_start.get(&cursor) else {
            continue;
        };
        for state in states {
            for candidate in candidates {
                let boundary_options = matching_hybrid_boundary_options(
                    state.output_physical_device_id.as_deref(),
                    candidate.observation,
                    cursor,
                    &resolved.boundaries_by_index,
                );
                for boundary in boundary_options {
                    let Some(mut next) = apply_hybrid_boundary(&state, boundary, capacity)? else {
                        continue;
                    };
                    let Some(resource_reservations) = next
                        .resource_reservations
                        .reserve(candidate.resources, capacity)
                        .map_err(|error| VulkanHybridPlacementError(error.to_string()))?
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
                    next.resource_reservations = resource_reservations;
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

    let mut complete = states_by_cursor
        .pop()
        .expect("a state bucket exists for the graph terminus")
        .into_iter()
        .collect::<Vec<_>>();
    complete.sort_by(|left, right| {
        hybrid_state_ordering_key(left).cmp(&hybrid_state_ordering_key(right))
    });
    Ok(complete
        .into_iter()
        .map(|state| VulkanHybridPlacementPlan {
            steps: state.steps,
            predicted_duration_ns_per_activation: state.predicted_duration_ns_per_activation,
            resource_reservations: state.resource_reservations,
        })
        .collect())
}

/// Visits complete measured routes in nondecreasing predicted-duration order
/// until the caller accepts one. Resource observations are validated but never
/// used to discard a partial route: the caller can therefore apply an exact
/// full-mount verifier without making route discovery depend on sampled
/// calibration residency. The search is lazy and has no arbitrary candidate
/// cap; an admissible suffix-duration bound keeps unrelated slower prefixes out
/// of the frontier until they can matter.
pub fn visit_vulkan_hybrid_ordered_graph_routes_by_duration<T, F>(
    catalog: &VulkanPlacementCalibrationCatalog,
    component_count: usize,
    region_candidates: &[VulkanHybridRegionCandidate],
    boundary_candidates: &[VulkanHybridBoundaryCandidate],
    resources: &VulkanHybridCandidateResourceCatalog,
    authoritative_resource_classes: &BTreeSet<VulkanHybridResourceClass>,
    eligible_capacity: &VulkanPlacementCapacityEnvelope,
    mut visitor: F,
) -> Result<Option<T>, VulkanHybridPlacementError>
where
    F: FnMut(&VulkanHybridPlacementRoute) -> Result<Option<T>, VulkanHybridPlacementError>,
{
    if component_count == 0 {
        return Err(VulkanHybridPlacementError(
            "hybrid placement requires a nonempty ordered graph".to_string(),
        ));
    }
    validate_hybrid_capacity_envelope(eligible_capacity)?;
    if region_candidates.is_empty() {
        return Ok(None);
    }
    let resolved = resolve_vulkan_hybrid_candidate_graph(
        catalog,
        component_count,
        region_candidates,
        boundary_candidates,
        resources,
    )?;
    let minimum_suffix_duration =
        minimum_hybrid_route_suffix_durations(component_count, &resolved, eligible_capacity)?;
    let Some(initial_bound) = minimum_suffix_duration[0] else {
        return Ok(None);
    };
    let unbounded_capacity = VulkanPlacementCapacityEnvelope {
        available_bytes_by_device: eligible_capacity
            .available_bytes_by_device
            .keys()
            .cloned()
            .map(|device| (device, usize::MAX))
            .collect(),
        host_available_bytes: usize::MAX,
    };
    let mut frontier = BTreeMap::<(u128, u128, u64), VulkanHybridRouteSearchState>::new();
    frontier.insert(
        (initial_bound, 0, 0),
        VulkanHybridRouteSearchState {
            cursor: 0,
            output_physical_device_id: None,
            steps: Vec::new(),
            predicted_duration_ns_per_activation: 0,
            calibration_resource_reservations: VulkanHybridResourceReservations::default(),
            authoritative_resource_reservations: VulkanHybridResourceReservations::default(),
        },
    );
    let mut insertion_ordinal = 0u64;

    while let Some((_, state)) = frontier.pop_first() {
        if state.cursor == component_count {
            let route = VulkanHybridPlacementRoute {
                steps: state.steps,
                predicted_duration_ns_per_activation: state.predicted_duration_ns_per_activation,
                calibration_resource_reservations: state.calibration_resource_reservations,
                authoritative_resource_reservations: state.authoritative_resource_reservations,
            };
            if let Some(result) = visitor(&route)? {
                return Ok(Some(result));
            }
            continue;
        }
        let Some(candidates) = resolved.regions_by_start.get(&state.cursor) else {
            continue;
        };
        for candidate in candidates {
            if !hybrid_execution_case_uses_only_eligible_devices(
                &candidate.observation.execution_case,
                eligible_capacity,
            ) {
                continue;
            }
            let boundary_options = matching_hybrid_boundary_options(
                state.output_physical_device_id.as_deref(),
                candidate.observation,
                state.cursor,
                &resolved.boundaries_by_index,
            );
            for boundary in boundary_options {
                if boundary.is_some_and(|boundary| {
                    !hybrid_execution_case_uses_only_eligible_devices(
                        &boundary.observation.execution_case,
                        eligible_capacity,
                    )
                }) {
                    continue;
                }
                let mut next = state.clone();
                if let Some(boundary) = boundary {
                    let Some(authoritative_resource_reservations) = next
                        .authoritative_resource_reservations
                        .reserve_classes(
                            boundary.resources,
                            authoritative_resource_classes,
                            eligible_capacity,
                        )
                        .map_err(|error| VulkanHybridPlacementError(error.to_string()))?
                    else {
                        continue;
                    };
                    next.authoritative_resource_reservations = authoritative_resource_reservations;
                    next.calibration_resource_reservations = next
                        .calibration_resource_reservations
                        .reserve(boundary.resources, &unbounded_capacity)
                        .map_err(|error| VulkanHybridPlacementError(error.to_string()))?
                        .ok_or_else(|| {
                            VulkanHybridPlacementError(
                                "hybrid boundary calibration resource claims exceed an unbounded route envelope"
                                    .to_string(),
                            )
                        })?;
                    next.predicted_duration_ns_per_activation = next
                        .predicted_duration_ns_per_activation
                        .checked_add(normalized_hybrid_duration_ns(boundary.observation)?)
                        .ok_or_else(|| {
                            VulkanHybridPlacementError(
                                "hybrid boundary predicted duration overflowed".to_string(),
                            )
                        })?;
                    next.steps.push(VulkanHybridScheduledStep::Boundary {
                        boundary_index: boundary.request.boundary_index,
                        execution_case: boundary.observation.execution_case.clone(),
                    });
                }
                let Some(authoritative_resource_reservations) = next
                    .authoritative_resource_reservations
                    .reserve_classes(
                        candidate.resources,
                        authoritative_resource_classes,
                        eligible_capacity,
                    )
                    .map_err(|error| VulkanHybridPlacementError(error.to_string()))?
                else {
                    continue;
                };
                next.authoritative_resource_reservations = authoritative_resource_reservations;
                next.calibration_resource_reservations = next
                    .calibration_resource_reservations
                    .reserve(candidate.resources, &unbounded_capacity)
                    .map_err(|error| VulkanHybridPlacementError(error.to_string()))?
                    .ok_or_else(|| {
                        VulkanHybridPlacementError(
                            "hybrid region calibration resource claims exceed an unbounded route envelope"
                                .to_string(),
                        )
                    })?;
                next.predicted_duration_ns_per_activation = next
                    .predicted_duration_ns_per_activation
                    .checked_add(normalized_hybrid_duration_ns(candidate.observation)?)
                    .ok_or_else(|| {
                        VulkanHybridPlacementError(
                            "hybrid placement predicted duration overflowed".to_string(),
                        )
                    })?;
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
                let Some(suffix) = minimum_suffix_duration[next.cursor] else {
                    continue;
                };
                let estimated_duration = next
                    .predicted_duration_ns_per_activation
                    .checked_add(suffix)
                    .ok_or_else(|| {
                        VulkanHybridPlacementError(
                            "hybrid route duration bound overflowed".to_string(),
                        )
                    })?;
                insertion_ordinal = insertion_ordinal.checked_add(1).ok_or_else(|| {
                    VulkanHybridPlacementError(
                        "hybrid route search insertion ordinal overflowed".to_string(),
                    )
                })?;
                frontier.insert(
                    (
                        estimated_duration,
                        next.predicted_duration_ns_per_activation,
                        insertion_ordinal,
                    ),
                    next,
                );
            }
        }
    }
    Ok(None)
}

fn minimum_hybrid_route_suffix_durations(
    component_count: usize,
    resolved: &VulkanHybridResolvedCandidateGraph<'_>,
    eligible_capacity: &VulkanPlacementCapacityEnvelope,
) -> Result<Vec<Option<u128>>, VulkanHybridPlacementError> {
    let mut suffix = vec![None; component_count + 1];
    suffix[component_count] = Some(0);
    for cursor in (0..component_count).rev() {
        let Some(candidates) = resolved.regions_by_start.get(&cursor) else {
            continue;
        };
        for candidate in candidates {
            if !hybrid_execution_case_uses_only_eligible_devices(
                &candidate.observation.execution_case,
                eligible_capacity,
            ) {
                continue;
            }
            let Some(remaining) = suffix[candidate.request.component_end] else {
                continue;
            };
            let duration = normalized_hybrid_duration_ns(candidate.observation)?
                .checked_add(remaining)
                .ok_or_else(|| {
                    VulkanHybridPlacementError(
                        "hybrid route suffix duration overflowed".to_string(),
                    )
                })?;
            suffix[cursor] = Some(suffix[cursor].map_or(duration, |current| current.min(duration)));
        }
    }
    Ok(suffix)
}

fn hybrid_execution_case_uses_only_eligible_devices(
    execution_case: &VulkanPlacementExecutionCaseIdentity,
    eligible_capacity: &VulkanPlacementCapacityEnvelope,
) -> bool {
    execution_case.devices.iter().all(|device| {
        eligible_capacity
            .available_bytes_by_device
            .contains_key(device)
    })
}

fn resolve_vulkan_hybrid_candidate_graph<'a>(
    catalog: &'a VulkanPlacementCalibrationCatalog,
    component_count: usize,
    region_candidates: &'a [VulkanHybridRegionCandidate],
    boundary_candidates: &'a [VulkanHybridBoundaryCandidate],
    resources: &'a VulkanHybridCandidateResourceCatalog,
) -> Result<VulkanHybridResolvedCandidateGraph<'a>, VulkanHybridPlacementError> {
    let mut candidate_ids = BTreeSet::new();
    let mut expected_phase = None;
    let mut semantic_cohort_by_range = BTreeMap::new();
    let mut regions_by_start = BTreeMap::<usize, Vec<VulkanHybridResolvedRegionCandidate>>::new();
    for candidate in region_candidates {
        if candidate.candidate_id.is_empty()
            || candidate.semantic_contract_id.is_empty()
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
                | VulkanPlacementExecutionStrategy::SerializedRegion
                | VulkanPlacementExecutionStrategy::TensorParallel
                | VulkanPlacementExecutionStrategy::WholeExpertParallel
                | VulkanPlacementExecutionStrategy::IntraExpertTensorParallel
                | VulkanPlacementExecutionStrategy::Hybrid
                | VulkanPlacementExecutionStrategy::HybridRegion
        ) {
            return Err(VulkanHybridPlacementError(format!(
                "hybrid region candidate {:?} has non-region strategy {:?}",
                candidate.candidate_id, observation.execution_case.strategy,
            )));
        }
        let range = (candidate.component_start, candidate.component_end);
        let semantic_cohort = (
            candidate.semantic_contract_id.clone(),
            observation.execution_case.behavior.phase,
            observation.execution_case.behavior.shape.clone(),
            observation
                .execution_case
                .behavior
                .input_fixture_digest
                .clone(),
        );
        if semantic_cohort_by_range
            .get(&range)
            .is_some_and(|existing| existing != &semantic_cohort)
        {
            return Err(VulkanHybridPlacementError(format!(
                "hybrid candidates for graph range {:?} do not share one explicit semantic contract and exact execution cohort",
                range,
            )));
        }
        semantic_cohort_by_range
            .entry(range)
            .or_insert(semantic_cohort);
        validate_hybrid_phase(
            &mut expected_phase,
            observation.execution_case.behavior.phase,
        )?;
        let candidate_resources = resources
            .region_resources_by_candidate_id
            .get(&candidate.candidate_id)
            .ok_or_else(|| {
                VulkanHybridPlacementError(format!(
                    "hybrid resource catalog omitted measured region candidate {:?}",
                    candidate.candidate_id,
                ))
            })?;
        regions_by_start
            .entry(candidate.component_start)
            .or_default()
            .push(VulkanHybridResolvedRegionCandidate {
                request: candidate,
                observation,
                resources: candidate_resources,
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
        let candidate_resources = resources
            .boundary_resources_by_case
            .get(&(candidate.boundary_index, candidate.execution_case.clone()))
            .ok_or_else(|| {
                VulkanHybridPlacementError(format!(
                    "hybrid resource catalog omitted measured boundary candidate {}",
                    candidate.boundary_index,
                ))
            })?;
        boundaries_by_index
            .entry(candidate.boundary_index)
            .or_default()
            .push(VulkanHybridResolvedBoundaryCandidate {
                request: candidate,
                observation,
                resources: candidate_resources,
            });
    }
    for candidates in boundaries_by_index.values_mut() {
        candidates.sort_by(|left, right| {
            left.observation
                .execution_case
                .cmp(&right.observation.execution_case)
        });
    }
    Ok(VulkanHybridResolvedCandidateGraph {
        regions_by_start,
        boundaries_by_index,
    })
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
    output_physical_device_id: Option<&str>,
    next: &VulkanPlacementCalibrationObservation,
    cursor: usize,
    boundaries_by_index: &'a BTreeMap<usize, Vec<VulkanHybridResolvedBoundaryCandidate<'a>>>,
) -> Vec<Option<&'a VulkanHybridResolvedBoundaryCandidate<'a>>> {
    let Some(source) = output_physical_device_id else {
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
    let Some(resource_reservations) = state
        .resource_reservations
        .reserve(boundary.resources, capacity)
        .map_err(|error| VulkanHybridPlacementError(error.to_string()))?
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
    next.resource_reservations = resource_reservations;
    next.steps.push(VulkanHybridScheduledStep::Boundary {
        boundary_index: boundary.request.boundary_index,
        execution_case: boundary.observation.execution_case.clone(),
    });
    Ok(Some(next))
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
    {
        return false;
    }
    left.resource_reservations
        .claims_are_subset_of(&right.resource_reservations)
}

fn hybrid_state_ordering_key(
    state: &VulkanHybridPlacementState,
) -> (u128, usize, usize, usize, usize, Vec<String>) {
    let (resident_bytes, transient_peak_bytes, host_resident_bytes, host_transient_peak_bytes) =
        state.resource_reservations.ordering_totals();
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
        transient_peak_bytes,
        host_resident_bytes,
        host_transient_peak_bytes,
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
        let shards = if strategy == VulkanPlacementExecutionStrategy::SingleDevice {
            Vec::new()
        } else {
            devices
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
                .collect::<Vec<_>>()
        };
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
        selected_step_region_ids(&plan.steps)
    }

    fn selected_step_region_ids(steps: &[VulkanHybridScheduledStep]) -> Vec<&str> {
        steps
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

        let candidates = [
            VulkanHybridRegionCandidate {
                candidate_id: "fast-remote".to_string(),
                component_start: 0,
                component_end: 1,
                semantic_contract_id: "first".to_string(),
                execution_case: fast_remote,
            },
            VulkanHybridRegionCandidate {
                candidate_id: "slower-local".to_string(),
                component_start: 0,
                component_end: 1,
                semantic_contract_id: "first".to_string(),
                execution_case: slower_local,
            },
            VulkanHybridRegionCandidate {
                candidate_id: "second".to_string(),
                component_start: 1,
                component_end: 2,
                semantic_contract_id: "second".to_string(),
                execution_case: second,
            },
        ];
        let boundaries = [VulkanHybridBoundaryCandidate {
            boundary_index: 0,
            byte_count: 16,
            execution_case: boundary,
        }];
        let placement_capacity = capacity(2, 100, 100);
        let plan = plan_vulkan_hybrid_ordered_graph(
            &catalog,
            2,
            &candidates,
            &boundaries,
            &placement_capacity,
        )
        .unwrap();

        assert_eq!(selected_region_ids(&plan), ["slower-local", "second"]);
        assert_eq!(plan.predicted_duration_ns_per_activation, 20);
        assert_eq!(plan.steps.len(), 2);

        let resources = VulkanHybridCandidateResourceCatalog::from_calibration(
            &catalog,
            &candidates,
            &boundaries,
        )
        .unwrap();
        let ordered = visit_vulkan_hybrid_ordered_graph_routes_by_duration(
            &catalog,
            2,
            &candidates,
            &boundaries,
            &resources,
            &BTreeSet::new(),
            &placement_capacity,
            |route| Ok(Some(route.clone())),
        )
        .unwrap()
        .expect("the duration-ordered visitor must produce a complete route");
        assert_eq!(
            selected_step_region_ids(&ordered.steps),
            ["slower-local", "second"]
        );
        assert_eq!(ordered.predicted_duration_ns_per_activation, 20);
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
                    semantic_contract_id: "first".to_string(),
                    execution_case: fast_large,
                },
                VulkanHybridRegionCandidate {
                    candidate_id: "slower-small".to_string(),
                    component_start: 0,
                    component_end: 1,
                    semantic_contract_id: "first".to_string(),
                    execution_case: slower_small,
                },
                VulkanHybridRegionCandidate {
                    candidate_id: "second".to_string(),
                    component_start: 1,
                    component_end: 2,
                    semantic_contract_id: "second".to_string(),
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
                    semantic_contract_id: "region".to_string(),
                    execution_case: one_activation,
                },
                VulkanHybridRegionCandidate {
                    candidate_id: "two".to_string(),
                    component_start: 0,
                    component_end: 1,
                    semantic_contract_id: "region".to_string(),
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
    fn verified_representations_with_distinct_signatures_compete_by_source_semantics() {
        let mut catalog = VulkanPlacementCalibrationCatalog::default();
        let native = record(
            &mut catalog,
            region_observation(
                region_behavior("native-fp8"),
                VulkanPlacementExecutionStrategy::SingleDevice,
                &[("gpu0", 20)],
                "gpu0",
                "gpu0",
                "gpu0",
                20,
                1,
            ),
        );
        let int4_tp = record(
            &mut catalog,
            region_observation(
                region_behavior("int4-tp"),
                VulkanPlacementExecutionStrategy::TensorParallel,
                &[("gpu0", 8), ("gpu1", 8)],
                "gpu0",
                "gpu0",
                "gpu0",
                9,
                1,
            ),
        );

        let plan = plan_vulkan_hybrid_ordered_graph(
            &catalog,
            1,
            &[
                VulkanHybridRegionCandidate {
                    candidate_id: "native".to_string(),
                    component_start: 0,
                    component_end: 1,
                    semantic_contract_id: "source-contract".to_string(),
                    execution_case: native,
                },
                VulkanHybridRegionCandidate {
                    candidate_id: "int4-tp".to_string(),
                    component_start: 0,
                    component_end: 1,
                    semantic_contract_id: "source-contract".to_string(),
                    execution_case: int4_tp,
                },
            ],
            &[],
            &capacity(2, 100, 100),
        )
        .unwrap();

        assert_eq!(selected_region_ids(&plan), ["int4-tp"]);
        assert_eq!(plan.predicted_duration_ns_per_activation, 9);
    }

    #[test]
    fn representations_cannot_claim_equivalence_across_source_contracts() {
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
                10,
                1,
            ),
        );
        let second = record(
            &mut catalog,
            region_observation(
                region_behavior("second"),
                VulkanPlacementExecutionStrategy::SingleDevice,
                &[("gpu0", 10)],
                "gpu0",
                "gpu0",
                "gpu0",
                9,
                1,
            ),
        );

        let error = plan_vulkan_hybrid_ordered_graph(
            &catalog,
            1,
            &[
                VulkanHybridRegionCandidate {
                    candidate_id: "first".to_string(),
                    component_start: 0,
                    component_end: 1,
                    semantic_contract_id: "source-a".to_string(),
                    execution_case: first,
                },
                VulkanHybridRegionCandidate {
                    candidate_id: "second".to_string(),
                    component_start: 0,
                    component_end: 1,
                    semantic_contract_id: "source-b".to_string(),
                    execution_case: second,
                },
            ],
            &[],
            &capacity(1, 100, 100),
        )
        .unwrap_err();

        assert!(error.0.contains("explicit semantic contract"));
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
                semantic_contract_id: "first".to_string(),
                execution_case: first,
            },
            VulkanHybridRegionCandidate {
                candidate_id: "second".to_string(),
                component_start: 1,
                component_end: 2,
                semantic_contract_id: "second".to_string(),
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

    #[test]
    fn final_residency_preserves_every_candidate_transient_peak() {
        let mut catalog = VulkanPlacementCalibrationCatalog::default();
        let mut first_observation = region_observation(
            region_behavior("first"),
            VulkanPlacementExecutionStrategy::SingleDevice,
            &[("gpu0", 10)],
            "gpu0",
            "gpu0",
            "gpu0",
            5,
            1,
        );
        first_observation
            .transient_peak_bytes_by_physical_device
            .insert("gpu0".to_string(), 50);
        let first = record(&mut catalog, first_observation);
        let second = record(
            &mut catalog,
            region_observation(
                region_behavior("second"),
                VulkanPlacementExecutionStrategy::SingleDevice,
                &[("gpu0", 50)],
                "gpu0",
                "gpu0",
                "gpu0",
                5,
                1,
            ),
        );

        let plan = try_plan_vulkan_hybrid_ordered_graph(
            &catalog,
            2,
            &[
                VulkanHybridRegionCandidate {
                    candidate_id: "first".to_string(),
                    component_start: 0,
                    component_end: 1,
                    semantic_contract_id: "first".to_string(),
                    execution_case: first,
                },
                VulkanHybridRegionCandidate {
                    candidate_id: "second".to_string(),
                    component_start: 1,
                    component_end: 2,
                    semantic_contract_id: "second".to_string(),
                    execution_case: second,
                },
            ],
            &[],
            &capacity(2, 60, 100),
        )
        .unwrap();

        assert!(
            plan.is_none(),
            "all permanently mounted regions must leave headroom for the largest execution transient"
        );
    }

    #[test]
    fn pareto_frontier_keeps_a_slower_path_with_less_transient_pressure() {
        let mut catalog = VulkanPlacementCalibrationCatalog::default();
        let mut fast_observation = region_observation(
            region_behavior("first"),
            VulkanPlacementExecutionStrategy::SingleDevice,
            &[("gpu0", 10)],
            "gpu0",
            "gpu0",
            "gpu0",
            5,
            1,
        );
        fast_observation
            .transient_peak_bytes_by_physical_device
            .insert("gpu0".to_string(), 80);
        let fast = record(&mut catalog, fast_observation);
        let mut slower_observation = region_observation(
            region_behavior("first"),
            VulkanPlacementExecutionStrategy::SingleDevice,
            &[("gpu0", 20)],
            "gpu0",
            "gpu0",
            "gpu0",
            8,
            1,
        );
        slower_observation.execution_case.implementation_digests = vec![digest('e')];
        slower_observation.execution_case.artifact_digest = digest('f');
        let slower = record(&mut catalog, slower_observation);
        let second = record(
            &mut catalog,
            region_observation(
                region_behavior("second"),
                VulkanPlacementExecutionStrategy::SingleDevice,
                &[("gpu0", 70)],
                "gpu0",
                "gpu0",
                "gpu0",
                5,
                1,
            ),
        );

        let candidates = [
            VulkanHybridRegionCandidate {
                candidate_id: "fast-high-transient".to_string(),
                component_start: 0,
                component_end: 1,
                semantic_contract_id: "first".to_string(),
                execution_case: fast,
            },
            VulkanHybridRegionCandidate {
                candidate_id: "slow-low-transient".to_string(),
                component_start: 0,
                component_end: 1,
                semantic_contract_id: "first".to_string(),
                execution_case: slower,
            },
            VulkanHybridRegionCandidate {
                candidate_id: "second".to_string(),
                component_start: 1,
                component_end: 2,
                semantic_contract_id: "second".to_string(),
                execution_case: second,
            },
        ];
        let resources =
            VulkanHybridCandidateResourceCatalog::from_calibration(&catalog, &candidates, &[])
                .unwrap();
        let alternatives = plan_vulkan_hybrid_ordered_graph_candidates_with_resources(
            &catalog,
            2,
            &candidates,
            &[],
            &resources,
            &capacity(2, 200, 100),
        )
        .unwrap();
        assert_eq!(alternatives.len(), 2);
        assert_eq!(
            selected_region_ids(&alternatives[0]),
            ["fast-high-transient", "second"]
        );
        assert_eq!(
            selected_region_ids(&alternatives[1]),
            ["slow-low-transient", "second"]
        );

        let plan = plan_vulkan_hybrid_ordered_graph_with_resources(
            &catalog,
            2,
            &candidates,
            &[],
            &resources,
            &capacity(2, 100, 100),
        )
        .unwrap();

        assert_eq!(selected_region_ids(&plan), ["slow-low-transient", "second"]);
        assert_eq!(plan.predicted_duration_ns_per_activation, 13);
        assert_eq!(
            plan.resource_reservations.device_bytes[&device("gpu0", 2)].permanent_bytes,
            90
        );
        assert_eq!(
            plan.resource_reservations.device_bytes[&device("gpu0", 2)]
                .execution_transient_peak_bytes,
            0
        );
    }

    #[test]
    fn duration_ordered_route_search_reaches_a_sampled_resource_dominated_alternative() {
        let mut catalog = VulkanPlacementCalibrationCatalog::default();
        let fast = record(
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
        let mut slow_observation = region_observation(
            region_behavior("first"),
            VulkanPlacementExecutionStrategy::SingleDevice,
            &[("gpu0", 20)],
            "gpu0",
            "gpu0",
            "gpu0",
            8,
            1,
        );
        slow_observation.execution_case.implementation_digests = vec![digest('e')];
        slow_observation.execution_case.artifact_digest = digest('f');
        let slow = record(&mut catalog, slow_observation);
        let second = record(
            &mut catalog,
            region_observation(
                region_behavior("second"),
                VulkanPlacementExecutionStrategy::SingleDevice,
                &[("gpu0", 10)],
                "gpu0",
                "gpu0",
                "gpu0",
                5,
                1,
            ),
        );
        let candidates = [
            VulkanHybridRegionCandidate {
                candidate_id: "fast".to_string(),
                component_start: 0,
                component_end: 1,
                semantic_contract_id: "first".to_string(),
                execution_case: fast,
            },
            VulkanHybridRegionCandidate {
                candidate_id: "slow".to_string(),
                component_start: 0,
                component_end: 1,
                semantic_contract_id: "first".to_string(),
                execution_case: slow,
            },
            VulkanHybridRegionCandidate {
                candidate_id: "second".to_string(),
                component_start: 1,
                component_end: 2,
                semantic_contract_id: "second".to_string(),
                execution_case: second,
            },
        ];
        let resources =
            VulkanHybridCandidateResourceCatalog::from_calibration(&catalog, &candidates, &[])
                .unwrap();
        let capacity = capacity(2, 100, 100);
        let pareto = plan_vulkan_hybrid_ordered_graph_candidates_with_resources(
            &catalog,
            2,
            &candidates,
            &[],
            &resources,
            &capacity,
        )
        .unwrap();
        assert_eq!(pareto.len(), 1);
        assert_eq!(selected_region_ids(&pareto[0]), ["fast", "second"]);

        let mut visited = Vec::new();
        let selected = visit_vulkan_hybrid_ordered_graph_routes_by_duration(
            &catalog,
            2,
            &candidates,
            &[],
            &resources,
            &BTreeSet::new(),
            &capacity,
            |route| {
                let ids = selected_step_region_ids(&route.steps)
                    .into_iter()
                    .map(str::to_string)
                    .collect::<Vec<_>>();
                visited.push(ids.clone());
                Ok((ids[0] == "slow").then(|| route.clone()))
            },
        )
        .unwrap()
        .expect("the exact terminal verifier accepts the second route");

        assert_eq!(visited, [["fast", "second"], ["slow", "second"]]);
        assert_eq!(
            selected_step_region_ids(&selected.steps),
            ["slow", "second"]
        );
        assert_eq!(selected.predicted_duration_ns_per_activation, 13);
    }

    #[test]
    fn duration_ordered_route_search_never_visits_an_ineligible_device() {
        let mut catalog = VulkanPlacementCalibrationCatalog::default();
        let unavailable = record(
            &mut catalog,
            region_observation(
                region_behavior("only"),
                VulkanPlacementExecutionStrategy::SingleDevice,
                &[("gpu2", 10)],
                "gpu2",
                "gpu2",
                "gpu2",
                1,
                1,
            ),
        );
        let available = record(
            &mut catalog,
            region_observation(
                region_behavior("only"),
                VulkanPlacementExecutionStrategy::SingleDevice,
                &[("gpu0", 10)],
                "gpu0",
                "gpu0",
                "gpu0",
                20,
                1,
            ),
        );
        let candidates = [
            VulkanHybridRegionCandidate {
                candidate_id: "unavailable".to_string(),
                component_start: 0,
                component_end: 1,
                semantic_contract_id: "only".to_string(),
                execution_case: unavailable,
            },
            VulkanHybridRegionCandidate {
                candidate_id: "available".to_string(),
                component_start: 0,
                component_end: 1,
                semantic_contract_id: "only".to_string(),
                execution_case: available,
            },
        ];
        let resources =
            VulkanHybridCandidateResourceCatalog::from_calibration(&catalog, &candidates, &[])
                .unwrap();
        let mut visited = Vec::new();

        let selected = visit_vulkan_hybrid_ordered_graph_routes_by_duration(
            &catalog,
            1,
            &candidates,
            &[],
            &resources,
            &BTreeSet::new(),
            &capacity(2, 100, 100),
            |route| {
                visited.push(selected_step_region_ids(&route.steps)[0].to_string());
                Ok(Some(route.clone()))
            },
        )
        .unwrap()
        .expect("the eligible slower route remains available");

        assert_eq!(visited, ["available"]);
        assert_eq!(selected_step_region_ids(&selected.steps), ["available"]);
    }

    #[test]
    fn exact_resource_claims_deduplicate_shared_cache_and_reject_an_oversized_load_wave() {
        let mut catalog = VulkanPlacementCalibrationCatalog::default();
        let first = record(
            &mut catalog,
            region_observation(
                region_behavior("first"),
                VulkanPlacementExecutionStrategy::SingleDevice,
                &[("gpu0", 0)],
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
                &[("gpu0", 0)],
                "gpu0",
                "gpu0",
                "gpu0",
                5,
                1,
            ),
        );
        let candidates = [
            VulkanHybridRegionCandidate {
                candidate_id: "first".to_string(),
                component_start: 0,
                component_end: 1,
                semantic_contract_id: "first".to_string(),
                execution_case: first,
            },
            VulkanHybridRegionCandidate {
                candidate_id: "second".to_string(),
                component_start: 1,
                component_end: 2,
                semantic_contract_id: "second".to_string(),
                execution_case: second,
            },
        ];
        let gpu0 = device("gpu0", 2);
        let shared_cache = VulkanHybridResourceClaim::device(
            "store:gpu0:cache",
            gpu0.clone(),
            VulkanHybridResourceClass::CacheQuota,
            60,
        );
        let shared_wave = VulkanHybridResourceClaim::device(
            "store:gpu0:wave",
            gpu0.clone(),
            VulkanHybridResourceClass::AtomicLoadWave,
            60,
        );
        let mut resources =
            VulkanHybridCandidateResourceCatalog::from_calibration(&catalog, &candidates, &[])
                .unwrap();
        resources
            .replace_region_claims(
                "first",
                vec![
                    VulkanHybridResourceClaim::device(
                        "state:first",
                        gpu0.clone(),
                        VulkanHybridResourceClass::MutableState,
                        10,
                    ),
                    shared_cache.clone(),
                    shared_wave.clone(),
                ],
            )
            .unwrap();
        resources
            .replace_region_claims(
                "second",
                vec![
                    VulkanHybridResourceClaim::device(
                        "state:second",
                        gpu0.clone(),
                        VulkanHybridResourceClass::MutableState,
                        10,
                    ),
                    shared_cache,
                    shared_wave,
                ],
            )
            .unwrap();

        let plan = plan_vulkan_hybrid_ordered_graph_with_resources(
            &catalog,
            2,
            &candidates,
            &[],
            &resources,
            &capacity(2, 80, 100),
        )
        .unwrap();
        let reservation = &plan.resource_reservations.device_bytes[&gpu0];
        assert_eq!(reservation.mutable_state_bytes, 20);
        assert_eq!(reservation.cache_quota_bytes, 60);
        assert_eq!(reservation.atomic_load_wave_bytes, 60);
        assert_eq!(reservation.required_capacity_bytes().unwrap(), 80);

        resources
            .replace_region_claims(
                "second",
                vec![
                    VulkanHybridResourceClaim::device(
                        "state:second",
                        gpu0.clone(),
                        VulkanHybridResourceClass::MutableState,
                        10,
                    ),
                    VulkanHybridResourceClaim::device(
                        "store:gpu0:cache",
                        gpu0.clone(),
                        VulkanHybridResourceClass::CacheQuota,
                        60,
                    ),
                    VulkanHybridResourceClaim::device(
                        "store:gpu0:wave:oversized",
                        gpu0,
                        VulkanHybridResourceClass::AtomicLoadWave,
                        61,
                    ),
                ],
            )
            .unwrap();
        assert!(
            try_plan_vulkan_hybrid_ordered_graph_with_resources(
                &catalog,
                2,
                &candidates,
                &[],
                &resources,
                &capacity(2, 80, 100),
            )
            .unwrap()
            .is_none(),
            "an atomic load wave must fit within the admitted cache quota"
        );
    }

    #[test]
    fn exact_resource_class_replacement_preserves_other_classes_and_rejects_mixed_input() {
        let mut catalog = VulkanPlacementCalibrationCatalog::default();
        let mut observation = region_observation(
            region_behavior("component"),
            VulkanPlacementExecutionStrategy::SingleDevice,
            &[("gpu0", 10)],
            "gpu0",
            "gpu0",
            "gpu0",
            5,
            1,
        );
        observation
            .transient_peak_bytes_by_physical_device
            .insert("gpu0".to_string(), 7);
        let execution_case = record(&mut catalog, observation);
        let candidates = [VulkanHybridRegionCandidate {
            candidate_id: "component".to_string(),
            component_start: 0,
            component_end: 1,
            semantic_contract_id: "component".to_string(),
            execution_case,
        }];
        let mut resources =
            VulkanHybridCandidateResourceCatalog::from_calibration(&catalog, &candidates, &[])
                .unwrap();
        let gpu0 = device("gpu0", 2);
        resources
            .replace_region_resource_class_claims(
                "component",
                VulkanHybridResourceClass::Permanent,
                vec![VulkanHybridResourceClaim::device(
                    "exact:parameter",
                    gpu0.clone(),
                    VulkanHybridResourceClass::Permanent,
                    42,
                )],
            )
            .unwrap();

        let claims = &resources.region_resources_by_candidate_id["component"].claims;
        assert!(claims.iter().any(|claim| {
            claim.class == VulkanHybridResourceClass::Permanent && claim.byte_count == 42
        }));
        assert!(claims.iter().any(|claim| {
            claim.class == VulkanHybridResourceClass::ExecutionTransient && claim.byte_count == 7
        }));
        assert!(!claims.iter().any(|claim| {
            claim.class == VulkanHybridResourceClass::Permanent && claim.byte_count == 10
        }));

        let before = claims.clone();
        let error = resources
            .replace_region_resource_class_claims(
                "component",
                VulkanHybridResourceClass::Permanent,
                vec![VulkanHybridResourceClaim::device(
                    "wrong:class",
                    gpu0,
                    VulkanHybridResourceClass::MutableState,
                    1,
                )],
            )
            .unwrap_err();
        assert!(error.0.contains("contains another resource class"));
        assert_eq!(
            resources.region_resources_by_candidate_id["component"].claims,
            before,
            "rejected replacement must leave the catalog unchanged",
        );
    }
}
