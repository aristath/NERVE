#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanRuntimeHybridOrderedPlacement {
    pub component_ids: Vec<String>,
    pub execution_phase: nerve_execution_contracts::ExecutionPhase,
    pub activation_batch_width: usize,
    pub plan: VulkanHybridPlacementPlan,
    pub region_executions_by_case: BTreeMap<
        VulkanPlacementExecutionCaseIdentity,
        VulkanPlacementRegionExecutionCalibration,
    >,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanRuntimeHybridPhaseSetPlacement {
    pub decode: VulkanRuntimeHybridOrderedPlacement,
    pub prefill: Option<VulkanRuntimeHybridOrderedPlacement>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VulkanRuntimeHybridPhysicalExecutionResolution {
    pub runtime_model: VulkanResidentRuntimeModel,
    pub physical_execution_plan: VulkanRuntimePhysicalExecutionPlan,
    pub decode_predicted_duration_ns_per_activation: u128,
    pub prefill_activation_batch_width: Option<usize>,
    pub prefill_predicted_duration_ns_per_activation: Option<u128>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VulkanRuntimeHybridLoweredPhasePlacement {
    /// Stable logical model with only the selected backbone/coordinator owner
    /// of each component applied. Internal shard pools remain phase-specific.
    pub runtime_model: VulkanResidentRuntimeModel,
    pub execution_phase: nerve_execution_contracts::ExecutionPhase,
    pub activation_batch_width: usize,
    pub component_device_pools: BTreeMap<String, Vec<String>>,
    pub execution_cases_by_component: BTreeMap<String, VulkanPlacementExecutionCaseIdentity>,
    pub boundary_executions: BTreeMap<usize, VulkanRuntimePhysicalBoundaryExecution>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanRuntimeHybridPlacementError(pub String);

struct VulkanRuntimeHybridCandidateGraph {
    component_ids: Vec<String>,
    execution_phase: nerve_execution_contracts::ExecutionPhase,
    activation_batch_width: usize,
    region_candidates: Vec<VulkanHybridRegionCandidate>,
    boundary_candidates: Vec<VulkanHybridBoundaryCandidate>,
    region_executions_by_case: BTreeMap<
        VulkanPlacementExecutionCaseIdentity,
        VulkanPlacementRegionExecutionCalibration,
    >,
}

impl Display for VulkanRuntimeHybridPlacementError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for VulkanRuntimeHybridPlacementError {}

/// Maps exact calibration evidence back onto the mounted model's canonical
/// signal-processor chain and solves the whole ordered graph. A catalog with
/// multiple behavior/shape cohorts for one compiler signature is deliberately
/// ambiguous: the caller must first select the exact workload cohort instead
/// of allowing the runtime to compare unlike sampled work.
pub fn plan_vulkan_runtime_hybrid_ordered_graph(
    runtime_model: &VulkanResidentRuntimeModel,
    catalog: &VulkanPlacementCalibrationCatalog,
    capacity: &VulkanPlacementCapacityEnvelope,
    phase: VulkanTargetedComponentExecutionPhase,
) -> Result<VulkanRuntimeHybridOrderedPlacement, VulkanRuntimeHybridPlacementError> {
    try_plan_vulkan_runtime_hybrid_ordered_graph_with_owners(
        runtime_model,
        catalog,
        capacity,
        phase,
        None,
    )?
    .ok_or_else(|| {
        VulkanRuntimeHybridPlacementError(
            "no exact measured runtime hybrid placement is available for the current devices and capacity"
                .to_string(),
        )
    })
}

/// Chooses one persistent decode-owned backbone and then optimizes prefill
/// against those same coordinators. Helpers and shard counts remain free to
/// differ by phase, but phase switching never remounts or duplicates the
/// logical component chain.
pub fn plan_vulkan_runtime_hybrid_phase_set(
    runtime_model: &VulkanResidentRuntimeModel,
    catalog: &VulkanPlacementCalibrationCatalog,
    capacity: &VulkanPlacementCapacityEnvelope,
    prefill_activation_batch_width: Option<usize>,
) -> Result<VulkanRuntimeHybridPhaseSetPlacement, VulkanRuntimeHybridPlacementError> {
    let phase_set = try_plan_vulkan_runtime_hybrid_phase_set(
        runtime_model,
        catalog,
        capacity,
        prefill_activation_batch_width,
    )?
    .ok_or_else(|| {
        VulkanRuntimeHybridPlacementError(
            "no exact measured runtime hybrid decode placement is available for the current devices and capacity"
                .to_string(),
        )
    })?;
    if prefill_activation_batch_width.is_some() && phase_set.prefill.is_none() {
        return runtime_hybrid_error(
            "no exact measured runtime hybrid prefill placement preserves the decode-owned backbone",
        );
    }
    Ok(phase_set)
}

/// Resolves exact measured physical execution for the ordinary runtime path.
///
/// Missing exact decode evidence or a capacity envelope that cannot admit a
/// complete measured graph returns `None`; the caller may retain its validated
/// scalar/serialized placement. Invalid or ambiguous evidence remains an
/// error. Prefill is selected only from a complete common measured width no
/// larger than the mounted context, while decode retains stable ownership.
pub fn resolve_vulkan_runtime_hybrid_physical_execution(
    runtime_model: &VulkanResidentRuntimeModel,
    catalog: &VulkanPlacementCalibrationCatalog,
    capacity: &VulkanPlacementCapacityEnvelope,
    context_capacity_activations: usize,
    logical_device_id_by_physical_device: &BTreeMap<String, String>,
) -> Result<Option<VulkanRuntimeHybridPhysicalExecutionResolution>, VulkanRuntimeHybridPlacementError>
{
    let prefill_activation_batch_width = vulkan_runtime_hybrid_calibrated_prefill_widths(
        runtime_model,
        catalog,
    )?
    .into_iter()
    .filter(|width| *width <= context_capacity_activations)
    .max();
    let Some(placement) = try_plan_vulkan_runtime_hybrid_phase_set(
        runtime_model,
        catalog,
        capacity,
        prefill_activation_batch_width,
    )?
    else {
        return Ok(None);
    };
    let decode_predicted_duration_ns_per_activation =
        placement.decode.plan.predicted_duration_ns_per_activation;
    let prefill_predicted_duration_ns_per_activation = placement
        .prefill
        .as_ref()
        .map(|prefill| prefill.plan.predicted_duration_ns_per_activation);
    let selected_prefill_activation_batch_width = placement
        .prefill
        .as_ref()
        .map(|prefill| prefill.activation_batch_width);
    let (runtime_model, physical_execution_plan) = lower_vulkan_runtime_hybrid_phase_set(
        runtime_model,
        &placement,
        logical_device_id_by_physical_device,
    )?;
    Ok(Some(VulkanRuntimeHybridPhysicalExecutionResolution {
        runtime_model,
        physical_execution_plan,
        decode_predicted_duration_ns_per_activation,
        prefill_activation_batch_width: selected_prefill_activation_batch_width,
        prefill_predicted_duration_ns_per_activation,
    }))
}

/// Attempts to select exact phase-local physical execution without making the
/// optimization catalog a model-load dependency. Invalid or ambiguous evidence
/// is still rejected. `None` means that valid evidence exists but no complete
/// decode graph fits the current devices/capacity; an unavailable prefill
/// cohort leaves decode selected and prefill on the stable coordinators.
pub fn try_plan_vulkan_runtime_hybrid_phase_set(
    runtime_model: &VulkanResidentRuntimeModel,
    catalog: &VulkanPlacementCalibrationCatalog,
    capacity: &VulkanPlacementCapacityEnvelope,
    prefill_activation_batch_width: Option<usize>,
) -> Result<Option<VulkanRuntimeHybridPhaseSetPlacement>, VulkanRuntimeHybridPlacementError> {
    if prefill_activation_batch_width.is_some_and(|width| width < 2) {
        return runtime_hybrid_error(
            "hybrid phase-set prefill requires a multi-lane activation batch width",
        );
    }
    let Some(decode) = try_plan_vulkan_runtime_hybrid_ordered_graph_with_owners(
        runtime_model,
        catalog,
        capacity,
        VulkanTargetedComponentExecutionPhase::Decode,
        None,
    )?
    else {
        return Ok(None);
    };
    let decode_owner_by_component = runtime_hybrid_physical_owners(&decode)?;
    let prefill = prefill_activation_batch_width
        .map(|activation_batch_width| {
            try_plan_vulkan_runtime_hybrid_ordered_graph_with_owners(
                runtime_model,
                catalog,
                capacity,
                VulkanTargetedComponentExecutionPhase::Prefill {
                    activation_batch_width,
                },
                Some(&decode_owner_by_component),
            )
        })
        .transpose()?
        .flatten();
    Ok(Some(VulkanRuntimeHybridPhaseSetPlacement {
        decode,
        prefill,
    }))
}

fn try_plan_vulkan_runtime_hybrid_ordered_graph_with_owners(
    runtime_model: &VulkanResidentRuntimeModel,
    catalog: &VulkanPlacementCalibrationCatalog,
    capacity: &VulkanPlacementCapacityEnvelope,
    phase: VulkanTargetedComponentExecutionPhase,
    required_owner_by_component: Option<&BTreeMap<String, String>>,
) -> Result<Option<VulkanRuntimeHybridOrderedPlacement>, VulkanRuntimeHybridPlacementError> {
    let candidates = runtime_hybrid_candidate_graph(
        runtime_model,
        catalog,
        phase,
        required_owner_by_component,
    )?;
    let plan = try_plan_vulkan_hybrid_ordered_graph(
        catalog,
        candidates.component_ids.len(),
        &candidates.region_candidates,
        &candidates.boundary_candidates,
        capacity,
    )
    .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    Ok(plan.map(|plan| {
        let selected_region_executions = plan
            .steps
            .iter()
            .filter_map(|step| match step {
                VulkanHybridScheduledStep::Region {
                    execution_case, ..
                } => candidates
                    .region_executions_by_case
                    .get(execution_case)
                    .map(|calibration| (execution_case.clone(), calibration.clone())),
                VulkanHybridScheduledStep::Boundary { .. } => None,
            })
            .collect();
        VulkanRuntimeHybridOrderedPlacement {
            component_ids: candidates.component_ids,
            execution_phase: candidates.execution_phase,
            activation_batch_width: candidates.activation_batch_width,
            plan,
            region_executions_by_case: selected_region_executions,
        }
    }))
}

fn runtime_hybrid_candidate_graph(
    runtime_model: &VulkanResidentRuntimeModel,
    catalog: &VulkanPlacementCalibrationCatalog,
    phase: VulkanTargetedComponentExecutionPhase,
    required_owner_by_component: Option<&BTreeMap<String, String>>,
) -> Result<VulkanRuntimeHybridCandidateGraph, VulkanRuntimeHybridPlacementError> {
    let execution_phase = runtime_hybrid_execution_phase(phase)?;
    let component_ids = runtime_model
        .circuit_graph
        .components
        .iter()
        .filter(|component| component.runtime_role.is_signal_processor())
        .map(|component| component.component_id.clone())
        .collect::<Vec<_>>();
    if component_ids.is_empty() {
        return runtime_hybrid_error("hybrid placement found no signal-processor components");
    }
    if let Some(required_owners) = required_owner_by_component
        && required_owners.keys().collect::<BTreeSet<_>>()
            != component_ids.iter().collect::<BTreeSet<_>>()
    {
        return runtime_hybrid_error(
            "hybrid owner constraints must cover every ordered component exactly once",
        );
    }

    // Reuse the graph's exact boundary validator. Besides computing physical
    // payloads it rejects non-nearest-neighbor wiring rather than silently
    // flattening a general DAG into a different ordered graph.
    let boundaries = vulkan_runtime_placement_boundaries(runtime_model)
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    if boundaries.len() != component_ids.len().saturating_sub(1) {
        return runtime_hybrid_error(
            "hybrid placement boundary count does not cover the ordered component graph",
        );
    }

    let component_targets = component_ids
        .iter()
        .map(|component_id| {
            vulkan_runtime_placement_calibration_target_for_component(
                runtime_model,
                component_id,
                phase,
            )
            .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut region_candidates = Vec::new();
    for (component_index, (component_id, target)) in
        component_ids.iter().zip(&component_targets).enumerate()
    {
        let behaviors = catalog
            .candidate_behaviors_for_compiled_execution(
                &target.signature_id,
                crate::RUNTIME_IMPLEMENTATION_FINGERPRINT,
                execution_phase,
            )
            .into_iter()
            .filter(|behavior| {
                behavior.shape.activation_batch_width == phase.activation_batch_width()
            })
            .collect::<Vec<_>>();
        if behaviors.len() > 1 {
            return runtime_hybrid_error(format!(
                "compiled component {component_id:?} has {} exact calibration behavior cohorts; expected one",
                behaviors.len(),
            ));
        }
        let Some(behavior) = behaviors.first() else {
            continue;
        };
        for (candidate_index, observation) in catalog
            .candidates_for_behavior(behavior)
            .into_iter()
            .filter(|observation| {
                matches!(
                    observation.execution_case.strategy,
                    VulkanPlacementExecutionStrategy::SingleDevice
                        | VulkanPlacementExecutionStrategy::TensorParallel
                        | VulkanPlacementExecutionStrategy::WholeExpertParallel
                        | VulkanPlacementExecutionStrategy::IntraExpertTensorParallel
                        | VulkanPlacementExecutionStrategy::Hybrid
                )
            })
            .filter(|observation| {
                required_owner_by_component.is_none_or(|required_owners| {
                    observation.execution_case.owner_physical_device_id
                        == required_owners[component_id]
                })
            })
            .enumerate()
        {
            region_candidates.push(VulkanHybridRegionCandidate {
                candidate_id: format!("component:{component_index}:case:{candidate_index}"),
                component_start: component_index,
                component_end: component_index + 1,
                execution_case: observation.execution_case.clone(),
            });
        }
    }

    let activation_batch_width = phase.activation_batch_width();
    let boundary_byte_counts = boundaries
        .iter()
        .map(|boundary| {
            let [transfer] = boundary.transfers.as_slice() else {
                return Ok(None);
            };
            if !transfer.source_in_prefix {
                return Ok(None);
            }
            transfer
                .byte_count
                .checked_mul(activation_batch_width)
                .map(Some)
                .ok_or_else(|| {
                    VulkanRuntimeHybridPlacementError(
                        "hybrid boundary activation byte count overflowed".to_string(),
                    )
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut region_executions_by_case = BTreeMap::new();
    for (region_index, calibration) in catalog.region_executions().iter().enumerate() {
        let outer = &calibration.execution_case;
        if outer.behavior.runtime_implementation_fingerprint
            != crate::RUNTIME_IMPLEMENTATION_FINGERPRINT
            || outer.behavior.phase != execution_phase
            || outer.behavior.shape.activation_batch_width != activation_batch_width
        {
            continue;
        }
        let region_len = calibration.component_cases.len();
        if region_len > component_ids.len() {
            continue;
        }
        for component_start in 0..=component_ids.len() - region_len {
            let component_end = component_start + region_len;
            let expected_signatures = component_targets[component_start..component_end]
                .iter()
                .map(|target| target.signature_id.as_str())
                .collect::<Vec<_>>();
            if calibration
                .component_cases
                .iter()
                .map(|case| case.behavior.compiled_execution_signature.as_str())
                .ne(expected_signatures)
            {
                continue;
            }
            let Some(expected_boundary_bytes) = boundary_byte_counts
                [component_start..component_end.saturating_sub(1)]
                .iter()
                .copied()
                .collect::<Option<Vec<_>>>()
            else {
                continue;
            };
            if calibration.boundary_byte_counts != expected_boundary_bytes
                || required_owner_by_component.is_some_and(|required_owners| {
                    calibration.component_cases.iter().enumerate().any(
                        |(offset, case)| {
                            case.owner_physical_device_id
                                != required_owners[&component_ids[component_start + offset]]
                        },
                    )
                })
            {
                continue;
            }
            region_candidates.push(VulkanHybridRegionCandidate {
                candidate_id: format!(
                    "region:{component_start}:{component_end}:case:{region_index}"
                ),
                component_start,
                component_end,
                execution_case: outer.clone(),
            });
            region_executions_by_case.insert(outer.clone(), calibration.clone());
        }
    }

    let mut boundary_candidates = Vec::new();
    for (boundary_index, boundary) in boundaries.iter().enumerate() {
        // Multiple or reverse-direction edge transfers require one measured
        // bundled boundary transaction. Independent transfer observations are
        // not summed here because that would invent synchronization/overlap
        // cost. Same-device paths remain legal without a boundary candidate.
        let [transfer] = boundary.transfers.as_slice() else {
            continue;
        };
        if !transfer.source_in_prefix {
            continue;
        }
        let byte_count = transfer
            .byte_count
            .checked_mul(activation_batch_width)
            .ok_or_else(|| {
                VulkanRuntimeHybridPlacementError(
                    "hybrid boundary activation byte count overflowed".to_string(),
                )
            })?;
        for observation in catalog.directed_boundary_candidates(
            crate::RUNTIME_IMPLEMENTATION_FINGERPRINT,
            execution_phase,
            activation_batch_width,
            byte_count,
        ).into_iter().filter(|observation| {
            runtime_hybrid_boundary_execution_case_is_compatible(
                execution_phase,
                Some(activation_batch_width),
                byte_count,
                &observation.execution_case,
            )
        }) {
            boundary_candidates.push(VulkanHybridBoundaryCandidate {
                boundary_index,
                byte_count,
                execution_case: observation.execution_case.clone(),
            });
        }
    }

    Ok(VulkanRuntimeHybridCandidateGraph {
        component_ids,
        execution_phase,
        activation_batch_width,
        region_candidates,
        boundary_candidates,
        region_executions_by_case,
    })
}

pub fn vulkan_runtime_hybrid_phase_is_calibrated(
    runtime_model: &VulkanResidentRuntimeModel,
    catalog: &VulkanPlacementCalibrationCatalog,
    phase: VulkanTargetedComponentExecutionPhase,
) -> Result<bool, VulkanRuntimeHybridPlacementError> {
    let candidates = runtime_hybrid_candidate_graph(runtime_model, catalog, phase, None)?;
    Ok(runtime_hybrid_candidate_graph_has_complete_route(
        &candidates,
    ))
}

fn runtime_hybrid_candidate_graph_has_complete_route(
    candidates: &VulkanRuntimeHybridCandidateGraph,
) -> bool {
    let component_count = candidates.component_ids.len();
    let mut outputs_by_cursor = vec![BTreeSet::<String>::new(); component_count + 1];
    for cursor in 0..component_count {
        for region in candidates
            .region_candidates
            .iter()
            .filter(|region| region.component_start == cursor)
        {
            let input = region.execution_case.input_physical_device_id.as_str();
            let input_is_reachable = if cursor == 0 {
                true
            } else {
                outputs_by_cursor[cursor].iter().any(|output| {
                    output == input
                        || candidates.boundary_candidates.iter().any(|boundary| {
                            boundary.boundary_index == cursor - 1
                                && boundary.execution_case.input_physical_device_id == *output
                                && boundary.execution_case.output_physical_device_id == input
                        })
                })
            };
            if input_is_reachable {
                outputs_by_cursor[region.component_end]
                    .insert(region.execution_case.output_physical_device_id.clone());
            }
        }
    }
    !outputs_by_cursor[component_count].is_empty()
}

pub fn vulkan_runtime_hybrid_calibrated_prefill_widths(
    runtime_model: &VulkanResidentRuntimeModel,
    catalog: &VulkanPlacementCalibrationCatalog,
) -> Result<Vec<usize>, VulkanRuntimeHybridPlacementError> {
    let mut candidate_widths = BTreeSet::new();
    for component in runtime_model
        .circuit_graph
        .components
        .iter()
        .filter(|component| component.runtime_role.is_signal_processor())
    {
        let execution = runtime_model
            .component_executions
            .iter()
            .find(|execution| execution.component_id == component.component_id)
            .ok_or_else(|| {
                VulkanRuntimeHybridPlacementError(format!(
                    "hybrid prefill calibration found no execution for component {:?}",
                    component.component_id,
                ))
            })?;
        if !execution
            .kernels
            .iter()
            .any(|kernel| kernel.execution_domain.supports_prefill())
        {
            return Ok(Vec::new());
        }
        let target = vulkan_runtime_placement_calibration_target_for_component(
            runtime_model,
            &component.component_id,
            VulkanTargetedComponentExecutionPhase::Prefill {
                activation_batch_width: 2,
            },
        )
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
        for behavior in catalog.candidate_behaviors_for_compiled_execution(
            &target.signature_id,
            crate::RUNTIME_IMPLEMENTATION_FINGERPRINT,
            nerve_execution_contracts::ExecutionPhase::Prefill,
        ) {
            if behavior.shape.activation_batch_width >= 2 {
                candidate_widths.insert(behavior.shape.activation_batch_width);
            }
        }
    }
    candidate_widths.extend(
        catalog
            .region_executions()
            .iter()
            .map(|region| &region.execution_case.behavior)
            .filter(|behavior| {
                behavior.runtime_implementation_fingerprint
                    == crate::RUNTIME_IMPLEMENTATION_FINGERPRINT
                    && behavior.phase == nerve_execution_contracts::ExecutionPhase::Prefill
                    && behavior.shape.activation_batch_width >= 2
            })
            .map(|behavior| behavior.shape.activation_batch_width),
    );
    let mut complete_widths = Vec::new();
    for activation_batch_width in candidate_widths {
        if vulkan_runtime_hybrid_phase_is_calibrated(
            runtime_model,
            catalog,
            VulkanTargetedComponentExecutionPhase::Prefill {
                activation_batch_width,
            },
        )? {
            complete_widths.push(activation_batch_width);
        }
    }
    Ok(complete_widths)
}

fn runtime_hybrid_physical_owners(
    placement: &VulkanRuntimeHybridOrderedPlacement,
) -> Result<BTreeMap<String, String>, VulkanRuntimeHybridPlacementError> {
    let mut owners = BTreeMap::new();
    for step in &placement.plan.steps {
        let VulkanHybridScheduledStep::Region {
            component_start,
            component_end,
            execution_case,
            ..
        } = step
        else {
            continue;
        };
        for (offset, component_case) in runtime_hybrid_step_component_cases(
            placement,
            *component_start,
            *component_end,
            execution_case,
        )?
        .into_iter()
        .enumerate()
        {
            let component_id = placement
                .component_ids
                .get(*component_start + offset)
                .ok_or_else(|| {
                    VulkanRuntimeHybridPlacementError(
                        "runtime hybrid phase set contains an out-of-range component".to_string(),
                    )
                })?;
            if owners
                .insert(
                    component_id.clone(),
                    component_case.owner_physical_device_id.clone(),
                )
                .is_some()
            {
                return runtime_hybrid_error(
                    "runtime hybrid phase set contains duplicate component ownership",
                );
            }
        }
    }
    if owners.len() != placement.component_ids.len() {
        return runtime_hybrid_error(
            "runtime hybrid phase set does not assign every component owner",
        );
    }
    Ok(owners)
}

fn runtime_hybrid_step_component_cases<'a>(
    placement: &'a VulkanRuntimeHybridOrderedPlacement,
    component_start: usize,
    component_end: usize,
    execution_case: &'a VulkanPlacementExecutionCaseIdentity,
) -> Result<Vec<&'a VulkanPlacementExecutionCaseIdentity>, VulkanRuntimeHybridPlacementError> {
    if component_start >= component_end || component_end > placement.component_ids.len() {
        return runtime_hybrid_error("runtime hybrid region has an invalid component range");
    }
    if component_end == component_start + 1 {
        if matches!(
            execution_case.strategy,
            VulkanPlacementExecutionStrategy::SerializedRegion
                | VulkanPlacementExecutionStrategy::HybridRegion
        ) {
            return runtime_hybrid_error(
                "runtime hybrid region strategy cannot replay as one component",
            );
        }
        return Ok(vec![execution_case]);
    }
    let calibration = placement
        .region_executions_by_case
        .get(execution_case)
        .ok_or_else(|| {
            VulkanRuntimeHybridPlacementError(
                "runtime hybrid multi-component region has no exact replay calibration"
                    .to_string(),
            )
        })?;
    if calibration.execution_case != *execution_case
        || calibration.component_cases.len() != component_end - component_start
    {
        return runtime_hybrid_error(
            "runtime hybrid region replay does not match its scheduled component range",
        );
    }
    Ok(calibration.component_cases.iter().collect())
}

pub fn lower_vulkan_runtime_hybrid_phase_set(
    runtime_model: &VulkanResidentRuntimeModel,
    placement: &VulkanRuntimeHybridPhaseSetPlacement,
    logical_device_id_by_physical_device: &BTreeMap<String, String>,
) -> Result<
    (
        VulkanResidentRuntimeModel,
        VulkanRuntimePhysicalExecutionPlan,
    ),
    VulkanRuntimeHybridPlacementError,
> {
    let decode = lower_vulkan_runtime_hybrid_phase_placement(
        runtime_model,
        &placement.decode,
        logical_device_id_by_physical_device,
    )?;
    let prefill = placement
        .prefill
        .as_ref()
        .map(|prefill| {
            lower_vulkan_runtime_hybrid_phase_placement(
                runtime_model,
                prefill,
                logical_device_id_by_physical_device,
            )
        })
        .transpose()?;
    if prefill
        .as_ref()
        .is_some_and(|prefill| prefill.runtime_model != decode.runtime_model)
    {
        return runtime_hybrid_error(
            "hybrid phase set changed stable component ownership between decode and prefill",
        );
    }
    let mut physical_execution_plan = VulkanRuntimePhysicalExecutionPlan {
        component_device_pools: VulkanDistributedPhaseComponentDevicePools {
            decode: decode.component_device_pools,
            decode_batch: BTreeMap::new(),
            prefill: BTreeMap::new(),
        },
        decode_execution_cases_by_component: decode.execution_cases_by_component,
        decode_boundary_executions: decode.boundary_executions,
        ..VulkanRuntimePhysicalExecutionPlan::default()
    };
    if let Some(prefill) = prefill {
        physical_execution_plan.component_device_pools.prefill = prefill.component_device_pools;
        physical_execution_plan.prefill_execution_cases_by_component =
            prefill.execution_cases_by_component;
        physical_execution_plan.prefill_boundary_executions = prefill.boundary_executions;
    }
    physical_execution_plan.validate(&decode.runtime_model)?;
    Ok((decode.runtime_model, physical_execution_plan))
}

/// Lowers a solved phase into stable component owners plus phase-local exact
/// physical cases. It intentionally does not write internal shard pools into
/// `runtime_model.placement`: that field is shared by decode and prefill, and
/// doing so would silently force one phase's winning split onto every other
/// execution shape.
pub fn lower_vulkan_runtime_hybrid_phase_placement(
    runtime_model: &VulkanResidentRuntimeModel,
    placement: &VulkanRuntimeHybridOrderedPlacement,
    logical_device_id_by_physical_device: &BTreeMap<String, String>,
) -> Result<VulkanRuntimeHybridLoweredPhasePlacement, VulkanRuntimeHybridPlacementError> {
    validate_runtime_hybrid_device_bindings(logical_device_id_by_physical_device)?;
    let current_component_ids = runtime_model
        .circuit_graph
        .components
        .iter()
        .filter(|component| component.runtime_role.is_signal_processor())
        .map(|component| component.component_id.clone())
        .collect::<Vec<_>>();
    if current_component_ids != placement.component_ids {
        return runtime_hybrid_error(
            "hybrid placement component order does not match the mounted runtime graph",
        );
    }

    let mut owner_by_component = BTreeMap::new();
    let mut component_device_pools = BTreeMap::new();
    let mut execution_cases_by_component = BTreeMap::new();
    let mut next_component = 0usize;
    for step in &placement.plan.steps {
        let VulkanHybridScheduledStep::Region {
            component_start,
            component_end,
            execution_case,
            ..
        } = step
        else {
            continue;
        };
        if *component_start != next_component {
            return runtime_hybrid_error(
                "runtime hybrid replay does not cover ordered components contiguously",
            );
        }
        let component_cases = runtime_hybrid_step_component_cases(
            placement,
            *component_start,
            *component_end,
            execution_case,
        )?;
        for (offset, component_case) in component_cases.into_iter().enumerate() {
            let component_index = *component_start + offset;
            let component_id = &placement.component_ids[component_index];
            validate_runtime_hybrid_case_for_component(
                runtime_model,
                component_id,
                placement.execution_phase,
                placement.activation_batch_width,
                component_case,
            )?;
            let physical_devices = runtime_hybrid_case_device_pool(component_case)?;
            let devices = physical_devices
                .iter()
                .map(|physical_device_id| {
                    logical_device_id_by_physical_device
                        .get(physical_device_id)
                        .cloned()
                        .ok_or_else(|| {
                            VulkanRuntimeHybridPlacementError(format!(
                                "hybrid physical case references unbound physical device {physical_device_id:?}",
                            ))
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let owner_device_id = logical_device_id_by_physical_device
                .get(&component_case.owner_physical_device_id)
                .expect("the owner is one of the resolved physical participants")
                .clone();
            owner_by_component.insert(component_id.clone(), owner_device_id);
            if devices.len() > 1 {
                component_device_pools.insert(component_id.clone(), devices);
            }
            execution_cases_by_component.insert(component_id.clone(), component_case.clone());
        }
        next_component = *component_end;
    }
    if next_component != placement.component_ids.len() {
        return runtime_hybrid_error(
            "runtime hybrid replay does not cover every ordered component exactly once",
        );
    }
    let default_device_id = owner_by_component
        .get(&placement.component_ids[0])
        .expect("the first component was covered above")
        .clone();
    let runtime_model = vulkan_runtime_model_with_component_placement(
        runtime_model,
        &default_device_id,
        &owner_by_component,
    )
    .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    let graph_boundaries = vulkan_runtime_placement_boundaries(&runtime_model)
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    let mut boundary_executions = BTreeMap::new();
    let mut scheduled_boundaries = Vec::<(
        usize,
        &VulkanPlacementExecutionCaseIdentity,
    )>::new();
    for step in &placement.plan.steps {
        match step {
            VulkanHybridScheduledStep::Boundary {
                boundary_index,
                execution_case,
            } => scheduled_boundaries.push((*boundary_index, execution_case)),
            VulkanHybridScheduledStep::Region {
                component_start,
                component_end,
                execution_case,
                ..
            } if *component_end > *component_start + 1 => {
                let calibration = placement
                    .region_executions_by_case
                    .get(execution_case)
                    .ok_or_else(|| {
                        VulkanRuntimeHybridPlacementError(
                            "runtime hybrid region has no internal boundary replay".to_string(),
                        )
                    })?;
                scheduled_boundaries.extend(calibration.boundary_cases.iter().map(|boundary| {
                    (
                        *component_start + boundary.boundary_ordinal,
                        &boundary.execution_case,
                    )
                }));
            }
            VulkanHybridScheduledStep::Region { .. } => {}
        }
    }
    scheduled_boundaries.sort_by_key(|(boundary_index, _)| *boundary_index);
    if scheduled_boundaries
        .windows(2)
        .any(|pair| pair[0].0 == pair[1].0)
    {
        return runtime_hybrid_error(
            "runtime hybrid placement repeats one physical boundary execution",
        );
    }
    for (boundary_index, execution_case) in scheduled_boundaries {
        let graph_boundary = graph_boundaries.get(boundary_index).ok_or_else(|| {
            VulkanRuntimeHybridPlacementError(format!(
                "hybrid physical boundary {boundary_index} is outside the mounted graph",
            ))
        })?;
        let [transfer] = graph_boundary.transfers.as_slice() else {
            return runtime_hybrid_error(format!(
                "hybrid physical boundary {boundary_index} does not address one transfer",
            ));
        };
        if !transfer.source_in_prefix {
            return runtime_hybrid_error(format!(
                "hybrid physical boundary {boundary_index} has reverse direction",
            ));
        }
        let source_component_id = placement
            .component_ids
            .get(boundary_index)
            .cloned()
            .ok_or_else(|| {
                VulkanRuntimeHybridPlacementError(format!(
                    "hybrid physical boundary {boundary_index} has no source component",
                ))
            })?;
        let destination_component_id = placement
            .component_ids
            .get(boundary_index + 1)
            .cloned()
            .ok_or_else(|| {
                VulkanRuntimeHybridPlacementError(format!(
                    "hybrid physical boundary {boundary_index} has no destination component",
                ))
            })?;
        let matching_edges = runtime_model
            .circuit_graph
            .edges
            .iter()
            .enumerate()
            .filter(|(_, edge)| {
                edge.source.component_id == source_component_id
                    && edge.destination.component_id == destination_component_id
            })
            .collect::<Vec<_>>();
        let [(edge_index, graph_edge)] = matching_edges.as_slice() else {
            return runtime_hybrid_error(format!(
                "hybrid physical boundary {boundary_index} does not identify exactly one mounted graph edge",
            ));
        };
        let source_device_id = logical_device_id_by_physical_device
            .get(&execution_case.input_physical_device_id)
            .cloned()
            .ok_or_else(|| {
                VulkanRuntimeHybridPlacementError(format!(
                    "hybrid boundary references unbound source physical device {:?}",
                    execution_case.input_physical_device_id,
                ))
            })?;
        let destination_device_id = logical_device_id_by_physical_device
            .get(&execution_case.output_physical_device_id)
            .cloned()
            .ok_or_else(|| {
                VulkanRuntimeHybridPlacementError(format!(
                    "hybrid boundary references unbound destination physical device {:?}",
                    execution_case.output_physical_device_id,
                ))
            })?;
        let boundary = VulkanRuntimePhysicalBoundaryExecution {
            boundary_index,
            edge_index: *edge_index,
            source_component_id,
            source_port_id: graph_edge.source.port_id.clone(),
            destination_component_id,
            destination_port_id: graph_edge.destination.port_id.clone(),
            source_device_id,
            destination_device_id,
            frame_byte_count: transfer.byte_count,
            execution_case: execution_case.clone(),
        };
        validate_runtime_hybrid_boundary_case(
            placement.execution_phase,
            (placement.execution_phase == nerve_execution_contracts::ExecutionPhase::Decode)
                .then_some(1),
            &boundary,
        )?;
        if boundary_executions
            .insert(boundary_index, boundary)
            .is_some()
        {
            return runtime_hybrid_error(format!(
                "hybrid physical boundary {boundary_index} is repeated",
            ));
        }
    }
    Ok(VulkanRuntimeHybridLoweredPhasePlacement {
        runtime_model,
        execution_phase: placement.execution_phase,
        activation_batch_width: placement.activation_batch_width,
        component_device_pools,
        execution_cases_by_component,
        boundary_executions,
    })
}

fn validate_runtime_hybrid_device_bindings(
    logical_device_id_by_physical_device: &BTreeMap<String, String>,
) -> Result<(), VulkanRuntimeHybridPlacementError> {
    if logical_device_id_by_physical_device.is_empty()
        || logical_device_id_by_physical_device
            .iter()
            .any(|(physical, logical)| physical.is_empty() || logical.is_empty())
        || logical_device_id_by_physical_device
            .values()
            .collect::<BTreeSet<_>>()
            .len()
            != logical_device_id_by_physical_device.len()
    {
        return runtime_hybrid_error(
            "hybrid replay requires a nonempty one-to-one physical-to-logical device binding",
        );
    }
    Ok(())
}

fn validate_runtime_hybrid_case_for_component(
    runtime_model: &VulkanResidentRuntimeModel,
    component_id: &str,
    execution_phase: nerve_execution_contracts::ExecutionPhase,
    activation_batch_width: usize,
    execution_case: &VulkanPlacementExecutionCaseIdentity,
) -> Result<(), VulkanRuntimeHybridPlacementError> {
    let target_phase = match execution_phase {
        nerve_execution_contracts::ExecutionPhase::Decode if activation_batch_width == 1 => {
            VulkanTargetedComponentExecutionPhase::Decode
        }
        nerve_execution_contracts::ExecutionPhase::Prefill => {
            VulkanTargetedComponentExecutionPhase::Prefill {
                activation_batch_width,
            }
        }
        nerve_execution_contracts::ExecutionPhase::Decode => {
            return runtime_hybrid_error(
                "multi-lane decode requires its own compiled calibration target",
            );
        }
    };
    let target = vulkan_runtime_placement_calibration_target_for_component(
        runtime_model,
        component_id,
        target_phase,
    )
    .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    if execution_case.behavior.compiled_execution_signature != target.signature_id
        || execution_case.behavior.runtime_implementation_fingerprint
            != crate::RUNTIME_IMPLEMENTATION_FINGERPRINT
        || execution_case.behavior.phase != execution_phase
        || execution_case.behavior.shape.activation_batch_width != activation_batch_width
    {
        return runtime_hybrid_error(format!(
            "hybrid physical case does not match compiled component {component_id:?} and its exact phase geometry",
        ));
    }
    Ok(())
}

fn runtime_hybrid_case_device_pool(
    execution_case: &VulkanPlacementExecutionCaseIdentity,
) -> Result<Vec<String>, VulkanRuntimeHybridPlacementError> {
    if execution_case.strategy == VulkanPlacementExecutionStrategy::SingleDevice {
        if execution_case.devices.len() != 1 {
            return runtime_hybrid_error(
                "single-device hybrid case does not contain exactly one physical target",
            );
        }
        return Ok(vec![execution_case.owner_physical_device_id.clone()]);
    }
    if execution_case.strategy == VulkanPlacementExecutionStrategy::Serialized {
        return runtime_hybrid_error(
            "a serialized multi-component case cannot be replayed as one component island",
        );
    }
    let participant_ids = execution_case
        .devices
        .iter()
        .map(|device| device.physical_device_id.as_str())
        .collect::<BTreeSet<_>>();
    let shard_ids = execution_case
        .shards
        .iter()
        .map(|shard| shard.physical_device_id.as_str())
        .collect::<BTreeSet<_>>();
    if execution_case.shards.is_empty()
        || participant_ids != shard_ids
        || !participant_ids.contains(execution_case.owner_physical_device_id.as_str())
    {
        return runtime_hybrid_error(
            "distributed hybrid case does not contain exact shard coverage for every participant",
        );
    }
    let mut participant_by_ordinal = BTreeMap::<usize, &str>::new();
    let mut ordinal_by_participant = BTreeMap::<&str, usize>::new();
    for shard in &execution_case.shards {
        let physical_device_id = shard.physical_device_id.as_str();
        if participant_by_ordinal
            .insert(shard.participant_ordinal, physical_device_id)
            .is_some_and(|existing| existing != physical_device_id)
            || ordinal_by_participant
                .insert(physical_device_id, shard.participant_ordinal)
                .is_some_and(|existing| existing != shard.participant_ordinal)
        {
            return runtime_hybrid_error(
                "distributed hybrid case contains conflicting calibrated participant order",
            );
        }
    }
    if participant_by_ordinal.len() != participant_ids.len()
        || ordinal_by_participant.len() != participant_ids.len()
        || !participant_by_ordinal
            .keys()
            .copied()
            .eq(0..participant_ids.len())
        || participant_by_ordinal.get(&0).copied()
            != Some(execution_case.owner_physical_device_id.as_str())
    {
        return runtime_hybrid_error(
            "distributed hybrid case does not contain a complete calibrated participant order rooted at its owner",
        );
    }
    Ok(participant_by_ordinal
        .into_values()
        .map(str::to_string)
        .collect())
}

fn runtime_hybrid_execution_phase(
    phase: VulkanTargetedComponentExecutionPhase,
) -> Result<nerve_execution_contracts::ExecutionPhase, VulkanRuntimeHybridPlacementError> {
    match phase {
        VulkanTargetedComponentExecutionPhase::Decode => {
            Ok(nerve_execution_contracts::ExecutionPhase::Decode)
        }
        VulkanTargetedComponentExecutionPhase::Prefill {
            activation_batch_width: 0,
        } => runtime_hybrid_error("hybrid prefill placement requires a positive batch width"),
        VulkanTargetedComponentExecutionPhase::Prefill { .. } => {
            Ok(nerve_execution_contracts::ExecutionPhase::Prefill)
        }
    }
}

fn runtime_hybrid_error<T>(
    message: impl Into<String>,
) -> Result<T, VulkanRuntimeHybridPlacementError> {
    Err(VulkanRuntimeHybridPlacementError(message.into()))
}
