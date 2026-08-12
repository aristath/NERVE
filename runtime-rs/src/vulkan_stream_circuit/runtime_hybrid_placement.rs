#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanRuntimeHybridOrderedPlacement {
    pub component_ids: Vec<String>,
    pub execution_phase: nerve_execution_contracts::ExecutionPhase,
    pub activation_batch_width: usize,
    pub plan: VulkanHybridPlacementPlan,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanRuntimeHybridPhaseSetPlacement {
    pub decode: VulkanRuntimeHybridOrderedPlacement,
    pub prefill: Option<VulkanRuntimeHybridOrderedPlacement>,
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

    let mut region_candidates = Vec::new();
    for (component_index, component_id) in component_ids.iter().enumerate() {
        let target = vulkan_runtime_placement_calibration_target_for_component(
            runtime_model,
            component_id,
            phase,
        )
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
        let behaviors = catalog
            .candidate_behaviors_for_compiled_execution(&target.signature_id, execution_phase)
            .into_iter()
            .filter(|behavior| {
                behavior.shape.activation_batch_width == phase.activation_batch_width()
            })
            .collect::<Vec<_>>();
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
                // Serialized evidence describes a multi-component region.
                // This runtime adapter currently lowers one exact case per
                // component, so admitting it here would let the optimizer
                // select a case that exact replay must reject.
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

    let plan = try_plan_vulkan_hybrid_ordered_graph(
        catalog,
        component_ids.len(),
        &region_candidates,
        &boundary_candidates,
        capacity,
    )
    .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    Ok(plan.map(|plan| VulkanRuntimeHybridOrderedPlacement {
        component_ids,
        execution_phase,
        activation_batch_width,
        plan,
    }))
}

pub fn vulkan_runtime_hybrid_phase_is_calibrated(
    runtime_model: &VulkanResidentRuntimeModel,
    catalog: &VulkanPlacementCalibrationCatalog,
    phase: VulkanTargetedComponentExecutionPhase,
) -> Result<bool, VulkanRuntimeHybridPlacementError> {
    let execution_phase = runtime_hybrid_execution_phase(phase)?;
    for component in runtime_model
        .circuit_graph
        .components
        .iter()
        .filter(|component| component.runtime_role.is_signal_processor())
    {
        let target = vulkan_runtime_placement_calibration_target_for_component(
            runtime_model,
            &component.component_id,
            phase,
        )
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
        let behavior_count = catalog
            .candidate_behaviors_for_compiled_execution(&target.signature_id, execution_phase)
            .into_iter()
            .filter(|behavior| {
                behavior.shape.activation_batch_width == phase.activation_batch_width()
            })
            .count();
        match behavior_count {
            0 => return Ok(false),
            1 => {}
            count => {
                return runtime_hybrid_error(format!(
                    "compiled component {:?} has {count} exact calibration behavior cohorts for activation width {}",
                    component.component_id,
                    phase.activation_batch_width(),
                ));
            }
        }
    }
    Ok(true)
}

pub fn vulkan_runtime_hybrid_calibrated_prefill_widths(
    runtime_model: &VulkanResidentRuntimeModel,
    catalog: &VulkanPlacementCalibrationCatalog,
) -> Result<Vec<usize>, VulkanRuntimeHybridPlacementError> {
    let mut common_widths = None::<BTreeSet<usize>>;
    for component in runtime_model
        .circuit_graph
        .components
        .iter()
        .filter(|component| component.runtime_role.is_signal_processor())
    {
        let target = vulkan_runtime_placement_calibration_target_for_component(
            runtime_model,
            &component.component_id,
            VulkanTargetedComponentExecutionPhase::Prefill {
                activation_batch_width: 2,
            },
        )
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
        let mut behavior_count_by_width = BTreeMap::<usize, usize>::new();
        for behavior in catalog.candidate_behaviors_for_compiled_execution(
            &target.signature_id,
            nerve_execution_contracts::ExecutionPhase::Prefill,
        ) {
            *behavior_count_by_width
                .entry(behavior.shape.activation_batch_width)
                .or_default() += 1;
        }
        if let Some((width, count)) = behavior_count_by_width
            .iter()
            .find(|(_, count)| **count != 1)
        {
            return runtime_hybrid_error(format!(
                "compiled component {:?} has {count} exact prefill behavior cohorts for activation width {width}",
                component.component_id,
            ));
        }
        let widths = behavior_count_by_width
            .into_keys()
            .filter(|width| *width >= 2)
            .collect::<BTreeSet<_>>();
        common_widths = Some(match common_widths {
            None => widths,
            Some(common) => common.intersection(&widths).copied().collect(),
        });
    }
    Ok(common_widths.unwrap_or_default().into_iter().collect())
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
        if *component_end != component_start + 1 {
            return runtime_hybrid_error(
                "runtime hybrid phase set requires one physical case per component",
            );
        }
        let component_id = placement
            .component_ids
            .get(*component_start)
            .ok_or_else(|| {
                VulkanRuntimeHybridPlacementError(
                    "runtime hybrid phase set contains an out-of-range component".to_string(),
                )
            })?;
        if owners
            .insert(
                component_id.clone(),
                execution_case.owner_physical_device_id.clone(),
            )
            .is_some()
        {
            return runtime_hybrid_error(
                "runtime hybrid phase set contains duplicate component ownership",
            );
        }
    }
    if owners.len() != placement.component_ids.len() {
        return runtime_hybrid_error(
            "runtime hybrid phase set does not assign every component owner",
        );
    }
    Ok(owners)
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
    let graph_boundaries = vulkan_runtime_placement_boundaries(&runtime_model)
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    let mut boundary_executions = BTreeMap::new();
    for step in &placement.plan.steps {
        let VulkanHybridScheduledStep::Boundary {
            boundary_index,
            execution_case,
        } = step
        else {
            continue;
        };
        let graph_boundary = graph_boundaries.get(*boundary_index).ok_or_else(|| {
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
            .get(*boundary_index)
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
            boundary_index: *boundary_index,
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
            .insert(*boundary_index, boundary)
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
