#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanRuntimeHybridOrderedPlacement {
    pub component_ids: Vec<String>,
    pub plan: VulkanHybridPlacementPlan,
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
        plan,
    })
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
