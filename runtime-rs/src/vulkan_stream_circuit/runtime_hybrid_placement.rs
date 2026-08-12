#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanRuntimeHybridOrderedPlacement {
    pub component_ids: Vec<String>,
    pub execution_phase: nerve_execution_contracts::ExecutionPhase,
    pub activation_batch_width: usize,
    pub plan: VulkanHybridPlacementPlan,
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
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanRuntimeHybridPlacementError(pub String);

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

    let mut region_candidates = Vec::new();
    for (component_index, component_id) in component_ids.iter().enumerate() {
        let target = vulkan_runtime_placement_calibration_target_for_component(
            runtime_model,
            component_id,
            phase,
        )
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
        let behaviors = catalog
            .candidate_behaviors_for_compiled_execution(&target.signature_id, execution_phase);
        let [behavior] = behaviors.as_slice() else {
            return runtime_hybrid_error(format!(
                "compiled component {component_id:?} has {} exact calibration behavior cohorts; expected one",
                behaviors.len(),
            ));
        };
        for (candidate_index, observation) in catalog
            .candidates_for_behavior(behavior)
            .into_iter()
            .filter(|observation| {
                matches!(
                    observation.execution_case.strategy,
                    VulkanPlacementExecutionStrategy::SingleDevice
                        | VulkanPlacementExecutionStrategy::Serialized
                        | VulkanPlacementExecutionStrategy::TensorParallel
                        | VulkanPlacementExecutionStrategy::WholeExpertParallel
                        | VulkanPlacementExecutionStrategy::IntraExpertTensorParallel
                        | VulkanPlacementExecutionStrategy::Hybrid
                )
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
            execution_phase,
            activation_batch_width,
            byte_count,
        ) {
            boundary_candidates.push(VulkanHybridBoundaryCandidate {
                boundary_index,
                byte_count,
                execution_case: observation.execution_case.clone(),
            });
        }
    }

    let plan = plan_vulkan_hybrid_ordered_graph(
        catalog,
        component_ids.len(),
        &region_candidates,
        &boundary_candidates,
        capacity,
    )
    .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    Ok(VulkanRuntimeHybridOrderedPlacement {
        component_ids,
        execution_phase,
        activation_batch_width,
        plan,
    })
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
        if *component_start != next_component || *component_end != component_start + 1 {
            return runtime_hybrid_error(
                "runtime hybrid replay currently requires one exact physical case per ordered component",
            );
        }
        let component_id = &placement.component_ids[*component_start];
        validate_runtime_hybrid_case_for_component(
            runtime_model,
            component_id,
            placement.execution_phase,
            placement.activation_batch_width,
            execution_case,
        )?;
        let physical_devices = runtime_hybrid_case_device_pool(execution_case)?;
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
            .get(&execution_case.owner_physical_device_id)
            .expect("the owner is one of the resolved physical participants")
            .clone();
        owner_by_component.insert(component_id.clone(), owner_device_id);
        if devices.len() > 1 {
            component_device_pools.insert(component_id.clone(), devices);
        }
        execution_cases_by_component.insert(component_id.clone(), execution_case.clone());
        next_component += 1;
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
    Ok(VulkanRuntimeHybridLoweredPhasePlacement {
        runtime_model,
        execution_phase: placement.execution_phase,
        activation_batch_width: placement.activation_batch_width,
        component_device_pools,
        execution_cases_by_component,
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
