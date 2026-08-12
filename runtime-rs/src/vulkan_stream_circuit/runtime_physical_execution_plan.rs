#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanRuntimePhysicalBoundaryExecution {
    pub boundary_index: usize,
    pub edge_index: usize,
    pub source_component_id: String,
    pub source_port_id: String,
    pub destination_component_id: String,
    pub destination_port_id: String,
    pub source_device_id: String,
    pub destination_device_id: String,
    pub frame_byte_count: usize,
    pub execution_case: VulkanPlacementExecutionCaseIdentity,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanRuntimeMountedBoundaryRoute {
    edge_index: usize,
    source_device_id: String,
    destination_device_id: String,
    frame_byte_count: usize,
    route: VulkanPlacedEdgeTransferRoute,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VulkanRuntimePhysicalExecutionPlan {
    pub component_device_pools: VulkanDistributedPhaseComponentDevicePools,
    pub decode_execution_cases_by_component:
        BTreeMap<String, VulkanPlacementExecutionCaseIdentity>,
    pub decode_batch_execution_cases_by_component:
        BTreeMap<String, VulkanPlacementExecutionCaseIdentity>,
    pub prefill_execution_cases_by_component:
        BTreeMap<String, VulkanPlacementExecutionCaseIdentity>,
    pub decode_boundary_executions:
        BTreeMap<usize, VulkanRuntimePhysicalBoundaryExecution>,
    pub decode_batch_boundary_executions:
        BTreeMap<usize, VulkanRuntimePhysicalBoundaryExecution>,
    pub prefill_boundary_executions:
        BTreeMap<usize, VulkanRuntimePhysicalBoundaryExecution>,
}

impl VulkanRuntimePhysicalExecutionPlan {
    pub fn uniform(runtime_model: &VulkanResidentRuntimeModel) -> Self {
        let signal_processor_placement = runtime_model
            .circuit_graph
            .signal_processor_placement(&runtime_model.placement);
        Self {
            component_device_pools: VulkanDistributedPhaseComponentDevicePools::uniform(
                &signal_processor_placement.component_shard_devices,
            ),
            ..Self::default()
        }
    }

    pub fn validate(
        &self,
        runtime_model: &VulkanResidentRuntimeModel,
    ) -> Result<(), VulkanRuntimeHybridPlacementError> {
        let component_ids = runtime_model
            .circuit_graph
            .components
            .iter()
            .filter(|component| component.runtime_role.is_signal_processor())
            .map(|component| component.component_id.as_str())
            .collect::<BTreeSet<_>>();
        if component_ids.is_empty() {
            return runtime_hybrid_error(
                "physical execution plan requires at least one signal processor",
            );
        }

        self.validate_phase_pools(
            runtime_model,
            "decode",
            &self.component_device_pools.decode,
            &component_ids,
        )?;
        self.validate_phase_pools(
            runtime_model,
            "decode_batch",
            &self.component_device_pools.decode_batch,
            &component_ids,
        )?;
        self.validate_phase_pools(
            runtime_model,
            "prefill",
            &self.component_device_pools.prefill,
            &component_ids,
        )?;
        self.validate_exact_phase_cases(
            runtime_model,
            "decode",
            nerve_execution_contracts::ExecutionPhase::Decode,
            Some(1),
            &component_ids,
            &self.component_device_pools.decode,
            &self.decode_execution_cases_by_component,
        )?;
        self.validate_exact_phase_cases(
            runtime_model,
            "decode_batch",
            nerve_execution_contracts::ExecutionPhase::Decode,
            None,
            &component_ids,
            &self.component_device_pools.decode_batch,
            &self.decode_batch_execution_cases_by_component,
        )?;
        self.validate_exact_phase_cases(
            runtime_model,
            "prefill",
            nerve_execution_contracts::ExecutionPhase::Prefill,
            None,
            &component_ids,
            &self.component_device_pools.prefill,
            &self.prefill_execution_cases_by_component,
        )?;
        self.validate_exact_boundary_executions(
            runtime_model,
            "decode",
            nerve_execution_contracts::ExecutionPhase::Decode,
            Some(1),
            &self.decode_execution_cases_by_component,
            &self.decode_boundary_executions,
        )?;
        self.validate_exact_boundary_executions(
            runtime_model,
            "decode_batch",
            nerve_execution_contracts::ExecutionPhase::Decode,
            None,
            &self.decode_batch_execution_cases_by_component,
            &self.decode_batch_boundary_executions,
        )?;
        self.validate_exact_boundary_executions(
            runtime_model,
            "prefill",
            nerve_execution_contracts::ExecutionPhase::Prefill,
            None,
            &self.prefill_execution_cases_by_component,
            &self.prefill_boundary_executions,
        )?;
        self.validate_stable_boundary_routes()?;
        Ok(())
    }

    pub fn device_ids(&self, runtime_model: &VulkanResidentRuntimeModel) -> Vec<String> {
        runtime_model
            .circuit_graph
            .signal_processor_device_ids(&runtime_model.placement)
            .into_iter()
            .chain(
                [
                    &self.component_device_pools.decode,
                    &self.component_device_pools.decode_batch,
                    &self.component_device_pools.prefill,
                ]
                .into_iter()
                .flat_map(|pools| pools.values().flatten().cloned()),
            )
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    fn mounted_boundary_routes(
        &self,
    ) -> Result<
        BTreeMap<usize, VulkanRuntimeMountedBoundaryRoute>,
        VulkanRuntimeHybridPlacementError,
    > {
        let mut mounted = BTreeMap::new();
        for boundaries in [
            &self.decode_boundary_executions,
            &self.decode_batch_boundary_executions,
            &self.prefill_boundary_executions,
        ] {
            for boundary in boundaries.values() {
                let selected = VulkanRuntimeMountedBoundaryRoute {
                    edge_index: boundary.edge_index,
                    source_device_id: boundary.source_device_id.clone(),
                    destination_device_id: boundary.destination_device_id.clone(),
                    frame_byte_count: boundary.frame_byte_count,
                    route: runtime_mounted_boundary_route(
                        &boundary.execution_case.transports[0].route,
                    )?,
                };
                if mounted
                    .insert(boundary.edge_index, selected.clone())
                    .is_some_and(|existing| existing != selected)
                {
                    return runtime_hybrid_error(format!(
                        "phase-local physical plans require incompatible mounted routes for edge {}",
                        boundary.edge_index,
                    ));
                }
            }
        }
        Ok(mounted)
    }

    fn validate_bound_boundary_device_identities(
        &self,
        device_identity_by_logical_device: &BTreeMap<
            String,
            VulkanPlacementDeviceExecutionIdentity,
        >,
    ) -> Result<(), VulkanRuntimeHybridPlacementError> {
        for boundaries in [
            &self.decode_boundary_executions,
            &self.decode_batch_boundary_executions,
            &self.prefill_boundary_executions,
        ] {
            for boundary in boundaries.values() {
                let source = device_identity_by_logical_device
                    .get(&boundary.source_device_id)
                    .ok_or_else(|| {
                        VulkanRuntimeHybridPlacementError(format!(
                            "exact physical boundary references unbound logical source {:?}",
                            boundary.source_device_id,
                        ))
                    })?;
                let destination = device_identity_by_logical_device
                    .get(&boundary.destination_device_id)
                    .ok_or_else(|| {
                        VulkanRuntimeHybridPlacementError(format!(
                            "exact physical boundary references unbound logical destination {:?}",
                            boundary.destination_device_id,
                        ))
                    })?;
                let selected = boundary
                    .execution_case
                    .devices
                    .iter()
                    .collect::<BTreeSet<_>>();
                let bound = [source, destination].into_iter().collect::<BTreeSet<_>>();
                if selected != bound
                    || boundary.execution_case.input_physical_device_id
                        != source.physical_device_id
                    || boundary.execution_case.output_physical_device_id
                        != destination.physical_device_id
                {
                    return runtime_hybrid_error(format!(
                        "exact physical boundary {} was calibrated for different bound devices or drivers",
                        boundary.boundary_index,
                    ));
                }
            }
        }
        Ok(())
    }

    fn validate_phase_pools(
        &self,
        runtime_model: &VulkanResidentRuntimeModel,
        phase: &str,
        pools: &BTreeMap<String, Vec<String>>,
        component_ids: &BTreeSet<&str>,
    ) -> Result<(), VulkanRuntimeHybridPlacementError> {
        for (component_id, device_ids) in pools {
            if !component_ids.contains(component_id.as_str()) {
                return runtime_hybrid_error(format!(
                    "physical {phase} plan references non-signal-processor component {component_id:?}",
                ));
            }
            if device_ids.len() < 2
                || device_ids.iter().any(String::is_empty)
                || device_ids.iter().collect::<BTreeSet<_>>().len() != device_ids.len()
            {
                return runtime_hybrid_error(format!(
                    "physical {phase} shard pool for component {component_id:?} requires at least two distinct nonempty devices",
                ));
            }
            let owner = runtime_model.placement.device_for_component(component_id);
            if device_ids.first().map(String::as_str) != Some(owner) {
                return runtime_hybrid_error(format!(
                    "physical {phase} shard pool for component {component_id:?} must begin with stable owner {owner:?}",
                ));
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn validate_exact_phase_cases(
        &self,
        runtime_model: &VulkanResidentRuntimeModel,
        phase_name: &str,
        execution_phase: nerve_execution_contracts::ExecutionPhase,
        exact_batch_width: Option<usize>,
        component_ids: &BTreeSet<&str>,
        pools: &BTreeMap<String, Vec<String>>,
        cases: &BTreeMap<String, VulkanPlacementExecutionCaseIdentity>,
    ) -> Result<(), VulkanRuntimeHybridPlacementError> {
        if cases.is_empty() {
            return Ok(());
        }
        if cases.keys().map(String::as_str).collect::<BTreeSet<_>>() != *component_ids {
            return runtime_hybrid_error(format!(
                "exact physical {phase_name} plan must cover every signal processor exactly once",
            ));
        }
        for (component_id, case) in cases {
            let batch_width = case.behavior.shape.activation_batch_width;
            if case.behavior.phase != execution_phase
                || exact_batch_width.is_some_and(|expected| batch_width != expected)
                || exact_batch_width.is_none() && batch_width < 2
            {
                return runtime_hybrid_error(format!(
                    "exact physical {phase_name} case for component {component_id:?} has incompatible phase geometry",
                ));
            }
            let distributed = case.strategy != VulkanPlacementExecutionStrategy::SingleDevice;
            if case.strategy == VulkanPlacementExecutionStrategy::Serialized
                || distributed != pools.contains_key(component_id)
            {
                return runtime_hybrid_error(format!(
                    "exact physical {phase_name} case for component {component_id:?} disagrees with its phase-local shard pool",
                ));
            }
            if let Some(pool) = pools.get(component_id)
                && pool.len() != case.devices.len()
            {
                return runtime_hybrid_error(format!(
                    "exact physical {phase_name} case for component {component_id:?} has {} participants but its logical pool has {}",
                    case.devices.len(),
                    pool.len(),
                ));
            }
            validate_runtime_hybrid_case_for_component(
                runtime_model,
                component_id,
                execution_phase,
                batch_width,
                case,
            )?;
        }
        Ok(())
    }

    fn validate_exact_boundary_executions(
        &self,
        runtime_model: &VulkanResidentRuntimeModel,
        phase_name: &str,
        execution_phase: nerve_execution_contracts::ExecutionPhase,
        exact_batch_width: Option<usize>,
        phase_cases: &BTreeMap<String, VulkanPlacementExecutionCaseIdentity>,
        boundaries: &BTreeMap<usize, VulkanRuntimePhysicalBoundaryExecution>,
    ) -> Result<(), VulkanRuntimeHybridPlacementError> {
        let graph_boundaries = vulkan_runtime_placement_boundaries(runtime_model)
            .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
        let signal_processor_ids = runtime_model
            .circuit_graph
            .components
            .iter()
            .filter(|component| component.runtime_role.is_signal_processor())
            .map(|component| component.component_id.as_str())
            .collect::<Vec<_>>();
        let required_boundaries = if phase_cases.is_empty() {
            BTreeSet::new()
        } else {
            (0..graph_boundaries.len())
                .filter(|boundary_index| {
                    runtime_model
                        .placement
                        .device_for_component(signal_processor_ids[*boundary_index])
                        != runtime_model
                            .placement
                            .device_for_component(signal_processor_ids[*boundary_index + 1])
                })
                .collect::<BTreeSet<_>>()
        };
        if boundaries.keys().copied().collect::<BTreeSet<_>>() != required_boundaries {
            return runtime_hybrid_error(format!(
                "exact physical {phase_name} boundary plan must cover every and only cross-device component boundary",
            ));
        }
        for (boundary_index, boundary) in boundaries {
            if *boundary_index != boundary.boundary_index {
                return runtime_hybrid_error(format!(
                    "exact physical {phase_name} boundary key {boundary_index} disagrees with its payload index {}",
                    boundary.boundary_index,
                ));
            }
            let Some(graph_boundary) = graph_boundaries.get(*boundary_index) else {
                return runtime_hybrid_error(format!(
                    "exact physical {phase_name} boundary {boundary_index} is outside the mounted graph",
                ));
            };
            let [graph_transfer] = graph_boundary.transfers.as_slice() else {
                return runtime_hybrid_error(format!(
                    "exact physical {phase_name} boundary {boundary_index} does not address one transfer",
                ));
            };
            if !graph_transfer.source_in_prefix
                || graph_transfer.byte_count != boundary.frame_byte_count
            {
                return runtime_hybrid_error(format!(
                    "exact physical {phase_name} boundary {boundary_index} has stale direction or frame geometry",
                ));
            }
            let graph_edge = runtime_model
                .circuit_graph
                .edges
                .get(boundary.edge_index)
                .ok_or_else(|| {
                    VulkanRuntimeHybridPlacementError(format!(
                        "exact physical {phase_name} boundary {boundary_index} references missing graph edge {}",
                        boundary.edge_index,
                    ))
                })?;
            if graph_edge.source.component_id != boundary.source_component_id
                || graph_edge.source.port_id != boundary.source_port_id
                || graph_edge.destination.component_id != boundary.destination_component_id
                || graph_edge.destination.port_id != boundary.destination_port_id
            {
                return runtime_hybrid_error(format!(
                    "exact physical {phase_name} boundary {boundary_index} does not match mounted graph edge {}",
                    boundary.edge_index,
                ));
            }
            let source_owner = runtime_model
                .placement
                .device_for_component(&boundary.source_component_id);
            let destination_owner = runtime_model
                .placement
                .device_for_component(&boundary.destination_component_id);
            if source_owner != boundary.source_device_id
                || destination_owner != boundary.destination_device_id
                || source_owner == destination_owner
            {
                return runtime_hybrid_error(format!(
                    "exact physical {phase_name} boundary {boundary_index} disagrees with mounted component ownership",
                ));
            }
            validate_runtime_hybrid_boundary_case(
                execution_phase,
                exact_batch_width,
                boundary,
            )?;
        }
        Ok(())
    }

    fn validate_stable_boundary_routes(&self) -> Result<(), VulkanRuntimeHybridPlacementError> {
        self.mounted_boundary_routes().map(|_| ())
    }
}

fn runtime_mounted_boundary_route(
    route: &str,
) -> Result<VulkanPlacedEdgeTransferRoute, VulkanRuntimeHybridPlacementError> {
    match route {
        "external_device_local" => Ok(VulkanPlacedEdgeTransferRoute::ExternalDeviceLocal),
        "device_local_staging" => Ok(VulkanPlacedEdgeTransferRoute::DeviceLocalStaging),
        _ => runtime_hybrid_error(format!(
            "exact physical boundary route {route:?} has no resident runtime implementation",
        )),
    }
}

fn validate_runtime_hybrid_boundary_case(
    execution_phase: nerve_execution_contracts::ExecutionPhase,
    exact_batch_width: Option<usize>,
    boundary: &VulkanRuntimePhysicalBoundaryExecution,
) -> Result<(), VulkanRuntimeHybridPlacementError> {
    let case = &boundary.execution_case;
    let batch_width = case.behavior.shape.activation_batch_width;
    if case.strategy != VulkanPlacementExecutionStrategy::DirectedBoundary
        || case.behavior.phase != execution_phase
        || exact_batch_width.is_some_and(|expected| batch_width != expected)
        || exact_batch_width.is_none() && batch_width < 2
    {
        return runtime_hybrid_error(
            "exact physical boundary has incompatible strategy or phase geometry",
        );
    }
    let byte_count = boundary
        .frame_byte_count
        .checked_mul(batch_width)
        .ok_or_else(|| {
            VulkanRuntimeHybridPlacementError(
                "exact physical boundary byte geometry overflowed".to_string(),
            )
        })?;
    if !runtime_hybrid_boundary_execution_case_is_compatible(
        execution_phase,
        exact_batch_width,
        byte_count,
        case,
    ) {
        return runtime_hybrid_error(
            "exact physical boundary has incompatible endpoints, bytes, or mounted route",
        );
    }
    Ok(())
}

fn runtime_hybrid_boundary_execution_case_is_compatible(
    execution_phase: nerve_execution_contracts::ExecutionPhase,
    exact_batch_width: Option<usize>,
    byte_count: usize,
    case: &VulkanPlacementExecutionCaseIdentity,
) -> bool {
    let batch_width = case.behavior.shape.activation_batch_width;
    let [transport] = case.transports.as_slice() else {
        return false;
    };
    matches!(
        case.operations.as_slice(),
        [VulkanPlacementOperationGeometry::DirectedTransfer {
            byte_count: operation_bytes,
            ..
        }] if *operation_bytes == byte_count
    ) && case.strategy == VulkanPlacementExecutionStrategy::DirectedBoundary
        && case.behavior.phase == execution_phase
        && exact_batch_width.is_none_or(|expected| batch_width == expected)
        && (exact_batch_width.is_some() || batch_width >= 2)
        && transport.source_physical_device_id == case.input_physical_device_id
        && transport.destination_physical_device_id == case.output_physical_device_id
        && case.owner_physical_device_id == case.input_physical_device_id
        && transport.byte_capacity == byte_count
        && case.behavior.shape.input_byte_capacity == byte_count
        && case.behavior.shape.output_byte_capacity == byte_count
        && matches!(
            transport.route.as_str(),
            "external_device_local" | "device_local_staging"
        )
}

#[cfg(test)]
mod runtime_physical_execution_plan_tests {
    use super::*;

    #[test]
    fn uniform_physical_plan_preserves_manual_shards_for_every_phase() {
        let canonical = tests::tiny_fixture_model_runtime_model_with_placement(
            StreamCircuitPlacementSpec::new("gpu0"),
        );
        let component_id = canonical
            .circuit_graph
            .components
            .iter()
            .find(|component| component.runtime_role.is_signal_processor())
            .unwrap()
            .component_id
            .clone();
        let model = canonical
            .with_component_shard_devices(
                &component_id,
                vec!["gpu0".to_string(), "gpu1".to_string()],
            )
            .unwrap();
        let plan = VulkanRuntimePhysicalExecutionPlan::uniform(&model);

        plan.validate(&model).unwrap();
        assert_eq!(
            plan.component_device_pools.decode,
            plan.component_device_pools.prefill
        );
        assert_eq!(plan.device_ids(&model), ["gpu0", "gpu1"]);
    }

    #[test]
    fn physical_plan_rejects_pool_that_is_not_rooted_at_stable_owner() {
        let model = tests::tiny_fixture_model_runtime_model_with_placement(
            StreamCircuitPlacementSpec::new("gpu0"),
        );
        let component_id = model
            .circuit_graph
            .components
            .iter()
            .find(|component| component.runtime_role.is_signal_processor())
            .unwrap()
            .component_id
            .clone();
        let mut plan = VulkanRuntimePhysicalExecutionPlan::uniform(&model);
        plan.component_device_pools.decode.insert(
            component_id,
            vec!["gpu1".to_string(), "gpu0".to_string()],
        );

        assert!(
            plan.validate(&model)
                .unwrap_err()
                .0
                .contains("must begin with stable owner")
        );
    }
}
