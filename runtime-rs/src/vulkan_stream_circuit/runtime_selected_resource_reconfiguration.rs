#[derive(Clone)]
struct VulkanRuntimeSelectedResourceReconfigurationContext {
    catalog: Arc<VulkanPlacementCalibrationCatalog>,
    requirements: Vec<VulkanRuntimeSelectedResourceExecutionRequirementPlan>,
    capacities: Vec<VulkanPlacementSelectedResourceDeviceCapacity>,
    cache_arbiter: Arc<VulkanSelectedResourceCacheArbiter>,
}

struct VulkanRuntimeSelectedResourceAdaptationState {
    context: Arc<VulkanRuntimeSelectedResourceReconfigurationContext>,
    execution_plans: VulkanDistributedExecutionPlanSet,
    telemetry_baseline: Option<VulkanSelectionTelemetrySnapshot>,
    cache_telemetry_baseline: Option<VulkanSelectionTelemetrySnapshot>,
    generation: u64,
}

fn initial_vulkan_runtime_selected_resource_adaptation_state(
    context: Arc<VulkanRuntimeSelectedResourceReconfigurationContext>,
    package_execution_plans: &VulkanDistributedExecutionPlanSet,
    source: Option<&VulkanRuntimeSelectedResourceAdaptationState>,
) -> VulkanRuntimeSelectedResourceAdaptationState {
    let (execution_plans, generation) = initial_vulkan_runtime_selected_resource_adaptation_seed(
        package_execution_plans,
        source.map(|source| (&source.execution_plans, source.generation)),
    );
    VulkanRuntimeSelectedResourceAdaptationState {
        context,
        execution_plans,
        // Telemetry buffers are stream-local. A cloned stream inherits the
        // physical ownership that makes its state executable, but establishes
        // its own observation window before proposing another move.
        telemetry_baseline: None,
        cache_telemetry_baseline: None,
        generation,
    }
}

fn initial_vulkan_runtime_selected_resource_adaptation_seed(
    package_execution_plans: &VulkanDistributedExecutionPlanSet,
    source: Option<(&VulkanDistributedExecutionPlanSet, u64)>,
) -> (VulkanDistributedExecutionPlanSet, u64) {
    source
        .map(|(execution_plans, generation)| (execution_plans.clone(), generation))
        .unwrap_or_else(|| (package_execution_plans.clone(), 0))
}

fn initialize_vulkan_stream_selected_resource_execution_ownership(
    buffers_by_device: &BTreeMap<String, Arc<VulkanDynamicResourceBuffers>>,
    execution_plan: &VulkanDistributedExecutionPlan,
) -> Result<(), VulkanError> {
    let placements = selected_resource_placements_from_execution_plan(execution_plan)
        .map_err(|error| VulkanError(error.to_string()))?;
    let mut replacements = Vec::new();
    for placement in placements {
        let resource_count = placement.assignments.len();
        let ownership = placement
            .execution_ownership_by_device(resource_count)
            .map_err(|error| VulkanError(error.to_string()))?;
        for (device_id, owned_resource_indices) in ownership {
            let buffers = buffers_by_device.get(&device_id).ok_or_else(|| {
                VulkanError(format!(
                    "selected-resource stream ownership for selector {:?} has no buffers on {device_id:?}",
                    placement.selector_id,
                ))
            })?;
            if buffers
                .selector_execution_ownership(&placement.selector_id)?
                .is_none()
            {
                return Err(VulkanError(format!(
                    "selected-resource stream buffers on {device_id:?} omit selector {:?}",
                    placement.selector_id,
                )));
            }
            replacements.push((
                device_id,
                placement.selector_id.clone(),
                owned_resource_indices,
            ));
        }
    }
    for (device_id, selector_id, owned_resource_indices) in replacements {
        buffers_by_device[&device_id]
            .replace_selector_execution_ownership_at_quiescent_boundary(
                &selector_id,
                &owned_resource_indices,
            )?;
    }
    Ok(())
}

fn apply_vulkan_selected_resource_reconfigurations_transactionally<E, F>(
    reconfigurations: &[VulkanSelectedResourceReconfigurationPlan],
    mut apply: F,
) -> Result<usize, (E, Option<E>)>
where
    F: FnMut(
        &str,
        &VulkanSelectedResourcePlacementPlan,
    ) -> Result<VulkanSelectedResourcePlacementPlan, E>,
{
    let mut applied = Vec::with_capacity(reconfigurations.len());
    for reconfiguration in reconfigurations {
        match apply(
            &reconfiguration.selector_id,
            &reconfiguration.proposed,
        ) {
            Ok(previous) => applied.push((reconfiguration.selector_id.as_str(), previous)),
            Err(error) => {
                let mut rollback_error = None;
                for (selector_id, previous) in applied.iter().rev() {
                    if let Err(error) = apply(selector_id, previous)
                        && rollback_error.is_none()
                    {
                        rollback_error = Some(error);
                    }
                }
                return Err((error, rollback_error));
            }
        }
    }
    Ok(applied.len())
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanSelectedResourceReplicaPreloadBatch {
    selector_id: String,
    destination_device_id: String,
    resource_indices: Vec<usize>,
}

fn selected_resource_replica_preload_batches(
    reconfigurations: &[VulkanSelectedResourceReconfigurationPlan],
) -> Result<Vec<VulkanSelectedResourceReplicaPreloadBatch>, VulkanResidentTokenModelPackageError> {
    let mut resources = BTreeMap::<(String, String), BTreeSet<usize>>::new();
    let mut moved_resources = BTreeSet::new();
    for reconfiguration in reconfigurations {
        if reconfiguration.selector_id.trim().is_empty()
            || reconfiguration.moves.is_empty()
            || reconfiguration
                .moves
                .iter()
                .any(|movement| movement.source_device_id == movement.destination_device_id)
        {
            return Err(VulkanResidentTokenModelPackageError::new(
                "selected-resource replica preload has an empty selector, no moves, or a local move",
            ));
        }
        for movement in &reconfiguration.moves {
            if !moved_resources.insert((
                reconfiguration.selector_id.as_str(),
                movement.resource_index,
            )) {
                return Err(VulkanResidentTokenModelPackageError::new(format!(
                    "selected-resource replica preload repeats selector {:?} resource {}",
                    reconfiguration.selector_id, movement.resource_index,
                )));
            }
            resources
                .entry((
                    reconfiguration.selector_id.clone(),
                    movement.destination_device_id.clone(),
                ))
                .or_default()
                .insert(movement.resource_index);
        }
    }
    Ok(resources
        .into_iter()
        .map(
            |((selector_id, destination_device_id), resource_indices)| {
                VulkanSelectedResourceReplicaPreloadBatch {
                    selector_id,
                    destination_device_id,
                    resource_indices: resource_indices.into_iter().collect(),
                }
            },
        )
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn build_vulkan_runtime_selected_resource_reconfiguration_context(
    runtime_model: &VulkanResidentRuntimeModel,
    resource_contract: &CompiledResourceResidencyContract,
    loaded_manifest: &VulkanLoadedKernelArtifactCatalog,
    execution_plans: &VulkanDistributedExecutionPlanSet,
    execution_ownership_plan: &VulkanDistributedSelectedResourceStorePlan,
    stores: &BTreeMap<String, Arc<VulkanCompiledResourceDeviceStore>>,
    device_identities: &BTreeMap<String, VulkanPlacementDeviceExecutionIdentity>,
    residency_policy: ResourceResidencyPolicy,
    catalog: Option<&VulkanPlacementCalibrationCatalog>,
) -> Result<Option<Arc<VulkanRuntimeSelectedResourceReconfigurationContext>>, VulkanResidentTokenModelPackageError>
{
    let Some(catalog) = catalog.filter(|_| residency_policy.is_demand_loaded()) else {
        return Ok(None);
    };
    let requirements = vulkan_runtime_selected_resource_execution_requirements(
        runtime_model,
        resource_contract,
        loaded_manifest,
        &execution_plans.decode,
        VulkanTargetedComponentExecutionPhase::Decode,
    )?;
    if requirements.is_empty() {
        return Ok(None);
    }
    let mut physical_store_ids = BTreeSet::new();
    let mut capacities = Vec::with_capacity(execution_ownership_plan.devices.len());
    for execution_device in &execution_ownership_plan.devices {
        if execution_device
            .selectors
            .iter()
            .all(|selector| selector.owned_resource_indices.is_empty())
        {
            continue;
        }
        let store = stores.get(&execution_device.device_id).ok_or_else(|| {
            VulkanResidentTokenModelPackageError::new(format!(
                "selected-resource reconfiguration has no store for {:?}",
                execution_device.device_id,
            ))
        })?;
        if !physical_store_ids.insert(store.device_id()) {
            // Two logical participants sharing one physical store cannot be
            // independently rebalanced. Preserve the mounted ownership rather
            // than double-counting one cache quota.
            return Ok(None);
        }
        let identity = device_identities
            .get(&execution_device.device_id)
            .cloned()
            .ok_or_else(|| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "selected-resource reconfiguration has no execution identity for {:?}",
                    execution_device.device_id,
                ))
            })?;
        capacities.push(VulkanPlacementSelectedResourceDeviceCapacity {
            device_id: execution_device.device_id.clone(),
            identity,
            resident_payload_capacity_bytes: store
                .dynamic_device_payload_capacity_bytes(),
        });
    }
    if capacities.len() < 2 {
        return Ok(None);
    }
    let adaptive_selector_ids = requirements
        .iter()
        .map(|requirement| requirement.selector_id.clone())
        .collect::<BTreeSet<_>>();
    let cache_arbiter =
        VulkanSelectedResourceCacheArbiter::new(stores, &adaptive_selector_ids)?;
    Ok(Some(Arc::new(
        VulkanRuntimeSelectedResourceReconfigurationContext {
            catalog: Arc::new(catalog.clone()),
            requirements,
            capacities,
            cache_arbiter,
        },
    )))
}

impl VulkanResidentInProcessPlacedStreamProcessor {
    fn active_prefill_distributed_execution_plan(
        &self,
    ) -> &VulkanDistributedExecutionPlan {
        self.selected_resource_adaptation
            .as_ref()
            .map(|state| &state.execution_plans.prefill)
            .unwrap_or_else(|| self.model.prefill_distributed_execution_plan())
    }

    fn active_decode_batch_distributed_execution_plan(
        &self,
    ) -> &VulkanDistributedExecutionPlan {
        self.selected_resource_adaptation
            .as_ref()
            .map(|state| &state.execution_plans.decode_batch)
            .unwrap_or_else(|| self.model.decode_batch_distributed_execution_plan())
    }

    fn selected_resource_adaptation_generation(&self) -> u64 {
        self.selected_resource_adaptation
            .as_ref()
            .map(|state| state.generation)
            .unwrap_or(0)
    }

    fn adapt_selected_resource_ownership_at_prompt_boundary(
        &mut self,
        telemetry: &VulkanSelectionTelemetrySnapshot,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
    ) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError> {
        let Some(mut state) = self.selected_resource_adaptation.take() else {
            return Ok(0);
        };
        let result = self.adapt_selected_resource_ownership_with_state(
            telemetry,
            devices,
            &mut state,
        );
        self.selected_resource_adaptation = Some(state);
        result
    }

    fn adapt_selected_resource_ownership_with_state(
        &mut self,
        telemetry: &VulkanSelectionTelemetrySnapshot,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        state: &mut VulkanRuntimeSelectedResourceAdaptationState,
    ) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError> {
        let window = state
            .telemetry_baseline
            .as_ref()
            .map(|baseline| telemetry.delta_since(baseline))
            .transpose()?
            .unwrap_or_else(|| telemetry.clone());
        let cache_window = state
            .cache_telemetry_baseline
            .as_ref()
            .map(|baseline| telemetry.delta_since(baseline))
            .transpose()?
            .unwrap_or_else(|| telemetry.clone());
        let (cache_arbiter, cache_stream_id) = {
            let cache_registration = self
                .selected_resource_cache_registration
                .as_ref()
                .ok_or_else(|| {
                    selection_telemetry_error(
                        "selected-resource adaptation has no cache registration".to_string(),
                    )
                })?;
            (
                Arc::clone(&cache_registration.arbiter),
                cache_registration.stream_id(),
            )
        };
        let _adaptation_transaction = cache_arbiter
            .adaptation_transaction
            .lock()
            .map_err(|_| {
                selection_telemetry_error(
                    "selected-resource adaptation transaction was poisoned".to_string(),
                )
            })?;
        let peers = cache_arbiter
            .peer_snapshot(cache_stream_id)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?;
        let current_stream_id = cache_stream_id.to_string();
        let reconfigurations =
            try_plan_vulkan_runtime_warm_selected_resource_reconfigurations(
                &state.execution_plans.decode,
                &state.context.requirements,
                &state.context.catalog,
                &state.context.capacities,
                &window,
                &current_stream_id,
                &peers,
                self.model.resource_residency_policy,
                nerve_execution_contracts::ExecutionPhase::Decode,
            )
            .map_err(|error| {
                selection_telemetry_error(format!(
                    "warm selected-resource planning failed: {error}"
                ))
            })?;
        let previous_execution_plans = state.execution_plans.clone();
        let next_generation = (!reconfigurations.is_empty())
            .then(|| {
                state.generation.checked_add(1).ok_or_else(|| {
                    selection_telemetry_error(
                        "selected-resource adaptation generation overflowed".to_string(),
                    )
                })
            })
            .transpose()?;
        let applied_count = if reconfigurations.is_empty() {
            0
        } else {
            self.preload_selected_resource_replicas(
                &reconfigurations,
                devices,
                &current_stream_id,
            )?;
            apply_vulkan_selected_resource_reconfigurations_transactionally(
                &reconfigurations,
                |selector_id, proposed| {
                    self.apply_selected_resource_placement_at_quiescent_boundary(
                        state,
                        selector_id,
                        proposed,
                    )
                },
            )
            .map_err(|(error, rollback_error)| match rollback_error {
                Some(rollback_error) => selection_telemetry_error(format!(
                    "warm selected-resource transaction failed: {error}; rollback also failed: {rollback_error}",
                )),
                None => error,
            })?
        };
        let cache_demand = state
            .context
            .cache_arbiter
            .stream_demand(&state.execution_plans.decode, &cache_window)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Package);
        let cache_update = cache_demand.and_then(|cache_demand| {
            cache_arbiter
                .replace_stream_demand_at_generation(
                    cache_stream_id,
                    peers.generation,
                    cache_demand,
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)
        });
        if let Err(error) = cache_update {
            let rollback_error = if applied_count == 0 {
                None
            } else {
                self.rollback_selected_resource_placements_at_quiescent_boundary(
                    state,
                    &previous_execution_plans,
                    &reconfigurations,
                )
            };
            return Err(match rollback_error {
                Some(rollback_error) => selection_telemetry_error(format!(
                    "selected-resource cache adaptation failed: {error}; ownership rollback also failed: {rollback_error}",
                )),
                None => error,
            });
        }
        state.cache_telemetry_baseline = Some(telemetry.clone());
        if applied_count > 0 {
            state.generation = next_generation
                .expect("a nonempty accepted reconfiguration reserves a generation");
            state.telemetry_baseline = Some(telemetry.clone());
        }
        // Batch runners retain dispatch-local gate masks and parameter-slot
        // descriptors. Dropping them here is safe because prompt completion is
        // quiescent; the next prefill remounts only stream-local execution
        // resources from the updated plan, never the package backbone.
        if applied_count > 0 {
            self.temporal_block_executions.borrow_mut().clear();
        }
        Ok(applied_count)
    }

    fn preload_selected_resource_replicas(
        &self,
        reconfigurations: &[VulkanSelectedResourceReconfigurationPlan],
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        stream_id: &str,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        for batch in selected_resource_replica_preload_batches(reconfigurations)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?
        {
            let store = self
                .model
                .compiled_resource_device_stores
                .get(&batch.destination_device_id)
                .ok_or_else(|| {
                    selection_telemetry_error(format!(
                        "selected-resource replica destination {:?} has no compiled resource store",
                        batch.destination_device_id,
                    ))
                })?;
            let device = devices.get(&batch.destination_device_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                    device_id: batch.destination_device_id.clone(),
                }
            })?;
            let owner = DeviceResourceResidencyOwnerId::new(format!(
                "replica:{}:{stream_id}:{}:{}",
                self.model.package_id,
                batch.selector_id,
                batch.destination_device_id,
            ))
            .map_err(|error| selection_telemetry_error(error.to_string()))?;
            store
                .load_selector_resources(
                    device,
                    &batch.selector_id,
                    &batch.resource_indices,
                    owner,
                )
                .map_err(|error| {
                    selection_telemetry_error(format!(
                        "selected-resource replica preload failed for selector {:?} resources {:?} on {:?}: {error}",
                        batch.selector_id,
                        batch.resource_indices,
                        batch.destination_device_id,
                    ))
                })?;
        }
        Ok(())
    }

    fn rollback_selected_resource_placements_at_quiescent_boundary(
        &mut self,
        state: &mut VulkanRuntimeSelectedResourceAdaptationState,
        previous_execution_plans: &VulkanDistributedExecutionPlanSet,
        reconfigurations: &[VulkanSelectedResourceReconfigurationPlan],
    ) -> Option<VulkanResidentInProcessPlacedRuntimeError> {
        let previous = match selected_resource_placements_from_execution_plan(
            &previous_execution_plans.decode,
        ) {
            Ok(previous) => previous
                .into_iter()
                .map(|placement| (placement.selector_id.clone(), placement))
                .collect::<BTreeMap<_, _>>(),
            Err(error) => return Some(selection_telemetry_error(error.to_string())),
        };
        let mut first_error = None;
        for reconfiguration in reconfigurations.iter().rev() {
            let result = previous
                .get(&reconfiguration.selector_id)
                .ok_or_else(|| {
                    selection_telemetry_error(format!(
                        "selected-resource rollback omits selector {:?}",
                        reconfiguration.selector_id,
                    ))
                })
                .and_then(|placement| {
                    self.apply_selected_resource_placement_at_quiescent_boundary(
                        state,
                        &reconfiguration.selector_id,
                        placement,
                    )
                    .map(|_| ())
                });
            if let Err(error) = result
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        }
        first_error
    }

    fn apply_selected_resource_placement_at_quiescent_boundary(
        &mut self,
        state: &mut VulkanRuntimeSelectedResourceAdaptationState,
        selector_id: &str,
        proposed: &VulkanSelectedResourcePlacementPlan,
    ) -> Result<VulkanSelectedResourcePlacementPlan, VulkanResidentInProcessPlacedRuntimeError> {
        if proposed.selector_id != selector_id {
            return Err(selection_telemetry_error(format!(
                "warm selected-resource proposal belongs to selector {:?}, expected {selector_id:?}",
                proposed.selector_id,
            )));
        }
        let current = selected_resource_placements_from_execution_plan(
            &state.execution_plans.decode,
        )
        .map_err(|error| {
            selection_telemetry_error(format!(
                "mounted selected-resource ownership is invalid: {error}"
            ))
        })?
        .into_iter()
        .find(|placement| placement.selector_id == selector_id)
        .ok_or_else(|| {
            selection_telemetry_error(format!(
                "mounted execution omits selector {:?}",
                selector_id,
            ))
        })?;
        let resource_count = current.assignments.len();
        let current_ownership = current
            .execution_ownership_by_device(resource_count)
            .map_err(|error| selection_telemetry_error(error.to_string()))?;
        let replacement_ownership = proposed
            .execution_ownership_by_device(resource_count)
            .map_err(|error| selection_telemetry_error(error.to_string()))?;
        validate_selected_resource_execution_ownership_replacement(
            &current_ownership,
            &replacement_ownership,
            resource_count,
        )
        .map_err(|error| selection_telemetry_error(error.to_string()))?;

        let mut next_plans = state.execution_plans.clone();
        next_plans
            .apply_selected_resource_placements(std::slice::from_ref(proposed))
            .map_err(|error| {
                selection_telemetry_error(format!(
                    "warm selected-resource plan replay failed: {error}"
                ))
            })?;

        let participant_ids = current_ownership.keys().cloned().collect::<Vec<_>>();
        for device_id in &participant_ids {
            let buffers = self
                .distributed_dynamic_resource_buffers
                .get(device_id)
                .ok_or_else(|| {
                    selection_telemetry_error(format!(
                        "warm selected-resource ownership has no stream-local buffers on {device_id:?}"
                    ))
                })?;
            let mounted = buffers
                .selector_execution_ownership(selector_id)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?
                .ok_or_else(|| {
                    selection_telemetry_error(format!(
                        "stream-local buffers on {device_id:?} omit selector {:?}",
                        selector_id,
                    ))
                })?;
            if mounted != current_ownership[device_id] {
                return Err(selection_telemetry_error(format!(
                    "stream-local ownership for selector {:?} on {device_id:?} is stale",
                    selector_id,
                )));
            }
        }

        let mut updated_buffers = Vec::new();
        for device_id in &participant_ids {
            let buffers = &self.distributed_dynamic_resource_buffers[device_id];
            if let Err(error) = buffers
                .replace_selector_execution_ownership_at_quiescent_boundary(
                    selector_id,
                    &replacement_ownership[device_id],
                )
            {
                let mut rollback_error = None;
                for rollback_device_id in updated_buffers.iter().rev() {
                    if let Err(error) = self.distributed_dynamic_resource_buffers
                        [*rollback_device_id]
                        .replace_selector_execution_ownership_at_quiescent_boundary(
                            selector_id,
                            &current_ownership[*rollback_device_id],
                        )
                        && rollback_error.is_none()
                    {
                        rollback_error = Some(error);
                    }
                }
                return Err(selection_telemetry_error(match rollback_error {
                    Some(rollback_error) => format!(
                        "failed to update selector {:?} parameter slots: {error}; rollback also failed: {rollback_error}",
                        selector_id,
                    ),
                    None => format!(
                        "failed to update selector {:?} parameter slots: {error}",
                        selector_id,
                    ),
                }));
            }
            updated_buffers.push(device_id);
        }
        if let Err(error) = self
            .distributed_dispatch_runners
            .replace_selected_resource_execution_ownership_at_quiescent_boundary(
                selector_id,
                &current_ownership,
                &replacement_ownership,
            )
        {
            let mut rollback_error = None;
            for rollback_device_id in updated_buffers.iter().rev() {
                if let Err(error) = self.distributed_dynamic_resource_buffers
                    [*rollback_device_id]
                    .replace_selector_execution_ownership_at_quiescent_boundary(
                        selector_id,
                        &current_ownership[*rollback_device_id],
                    )
                    && rollback_error.is_none()
                {
                    rollback_error = Some(error);
                }
            }
            return Err(selection_telemetry_error(match rollback_error {
                Some(rollback_error) => format!(
                    "failed to update selector {:?} residency gates: {error}; parameter-slot rollback also failed: {rollback_error}",
                    selector_id,
                ),
                None => format!(
                    "failed to update selector {:?} residency gates: {error}",
                    selector_id,
                ),
            }));
        }
        state.execution_plans = next_plans;
        Ok(current)
    }
}

#[cfg(test)]
mod runtime_selected_resource_reconfiguration_tests {
    use super::*;
    use crate::vulkan_distributed::{
        VulkanSelectedResourceAssignment, VulkanSelectedResourcePlacementMove,
    };

    fn empty_execution_plan(device_id: &str) -> VulkanDistributedExecutionPlan {
        VulkanDistributedExecutionPlan {
            device_ids: vec![device_id.to_string()],
            storage_buffer_offset_alignment: 1,
            dispatches: Vec::new(),
            execution_islands: Vec::new(),
            shared_activation_route: VulkanSharedResidentBufferRoute::SharedHost,
            shared_input_byte_capacity: 1,
            shared_output_byte_capacity: 1,
            distributed_parameter_byte_count: 0,
        }
    }

    fn execution_plan_set(device_id: &str) -> VulkanDistributedExecutionPlanSet {
        let plan = empty_execution_plan(device_id);
        VulkanDistributedExecutionPlanSet {
            decode: plan.clone(),
            decode_batch: plan.clone(),
            prefill: plan,
        }
    }

    #[test]
    fn cloned_stream_inherits_adapted_ownership_generation_not_package_baseline() {
        let package = execution_plan_set("package-owner");
        let source = execution_plan_set("adapted-owner");

        let (fresh, fresh_generation) =
            initial_vulkan_runtime_selected_resource_adaptation_seed(&package, None);
        assert_eq!(fresh, package);
        assert_eq!(fresh_generation, 0);

        let (inherited, inherited_generation) =
            initial_vulkan_runtime_selected_resource_adaptation_seed(
                &package,
                Some((&source, 7)),
            );
        assert_eq!(inherited, source);
        assert_eq!(inherited_generation, 7);
        assert_ne!(inherited, package);
    }

    fn placement(selector_id: &str, device_id: &str) -> VulkanSelectedResourcePlacementPlan {
        VulkanSelectedResourcePlacementPlan {
            selector_id: selector_id.to_string(),
            assignments: vec![VulkanSelectedResourceAssignment {
                resource_index: 0,
                device_id: device_id.to_string(),
            }],
            device_loads: Vec::new(),
            maximum_first_moment_ns: 0,
            maximum_second_moment_ns2: 0,
        }
    }

    fn reconfiguration(
        selector_id: &str,
        device_id: &str,
    ) -> VulkanSelectedResourceReconfigurationPlan {
        VulkanSelectedResourceReconfigurationPlan {
            selector_id: selector_id.to_string(),
            observed_activation_count: 1,
            current_duration_ns_per_activation: 2,
            proposed_duration_ns_per_activation: 1,
            migration_critical_path_ns: 1,
            break_even_activation_count: 1,
            moves: Vec::new(),
            proposed: placement(selector_id, device_id),
        }
    }

    #[test]
    fn warm_reconfiguration_rolls_back_every_earlier_selector_after_failure() {
        let reconfigurations = [
            reconfiguration("first", "next-first"),
            reconfiguration("second", "fail"),
        ];
        let mut mounted = BTreeMap::from([
            ("first".to_string(), placement("first", "old-first")),
            ("second".to_string(), placement("second", "old-second")),
        ]);
        let result = apply_vulkan_selected_resource_reconfigurations_transactionally(
            &reconfigurations,
            |selector_id, proposed| {
                if proposed.assignments[0].device_id == "fail" {
                    return Err("injected apply failure");
                }
                let current = mounted
                    .insert(selector_id.to_string(), proposed.clone())
                    .expect("fixture selector must exist");
                Ok(current)
            },
        );

        assert_eq!(result, Err(("injected apply failure", None)));
        assert_eq!(mounted["first"].assignments[0].device_id, "old-first");
        assert_eq!(mounted["second"].assignments[0].device_id, "old-second");
    }

    #[test]
    fn warm_reconfiguration_commits_every_selector_as_one_generation() {
        let reconfigurations = [
            reconfiguration("first", "next-first"),
            reconfiguration("second", "next-second"),
        ];
        let mut mounted = BTreeMap::from([
            ("first".to_string(), placement("first", "old-first")),
            ("second".to_string(), placement("second", "old-second")),
        ]);
        let applied = apply_vulkan_selected_resource_reconfigurations_transactionally(
            &reconfigurations,
            |selector_id, proposed| {
                Ok::<_, &'static str>(
                    mounted
                        .insert(selector_id.to_string(), proposed.clone())
                        .expect("fixture selector must exist"),
                )
            },
        )
        .unwrap();

        assert_eq!(applied, 2);
        assert_eq!(mounted["first"].assignments[0].device_id, "next-first");
        assert_eq!(mounted["second"].assignments[0].device_id, "next-second");
    }

    #[test]
    fn warm_reconfiguration_attempts_all_rollbacks_after_one_rollback_fails() {
        let reconfigurations = [
            reconfiguration("first", "next-first"),
            reconfiguration("second", "next-second"),
            reconfiguration("third", "fail-apply"),
        ];
        let mut mounted = BTreeMap::from([
            ("first".to_string(), placement("first", "old-first")),
            ("second".to_string(), placement("second", "old-second")),
            ("third".to_string(), placement("third", "old-third")),
        ]);
        let result = apply_vulkan_selected_resource_reconfigurations_transactionally(
            &reconfigurations,
            |selector_id, proposed| {
                let destination = proposed.assignments[0].device_id.as_str();
                if destination == "fail-apply" {
                    return Err("injected apply failure");
                }
                if selector_id == "second" && destination == "old-second" {
                    return Err("injected rollback failure");
                }
                let current = mounted
                    .insert(selector_id.to_string(), proposed.clone())
                    .expect("fixture selector must exist");
                Ok(current)
            },
        );

        assert_eq!(
            result,
            Err(("injected apply failure", Some("injected rollback failure")))
        );
        assert_eq!(mounted["first"].assignments[0].device_id, "old-first");
        assert_eq!(mounted["second"].assignments[0].device_id, "next-second");
        assert_eq!(mounted["third"].assignments[0].device_id, "old-third");
    }

    fn preload_reconfiguration(
        selector_id: &str,
        resource_index: usize,
        destination_device_id: &str,
    ) -> VulkanSelectedResourceReconfigurationPlan {
        VulkanSelectedResourceReconfigurationPlan {
            selector_id: selector_id.to_string(),
            observed_activation_count: 10,
            current_duration_ns_per_activation: 2,
            proposed_duration_ns_per_activation: 1,
            migration_critical_path_ns: 1,
            break_even_activation_count: 1,
            moves: vec![VulkanSelectedResourcePlacementMove {
                resource_index,
                source_device_id: "gpu0".to_string(),
                destination_device_id: destination_device_id.to_string(),
                payload_bytes: 10,
                destination_load_duration_ns: 1,
            }],
            proposed: placement(selector_id, destination_device_id),
        }
    }

    #[test]
    fn replica_preload_batches_group_unique_resources_by_selector_and_destination() {
        let mut first = preload_reconfiguration("experts", 2, "gpu1");
        first.moves.push(VulkanSelectedResourcePlacementMove {
            resource_index: 0,
            source_device_id: "gpu0".to_string(),
            destination_device_id: "gpu1".to_string(),
            payload_bytes: 10,
            destination_load_duration_ns: 1,
        });
        let batches = selected_resource_replica_preload_batches(&[
            first,
            preload_reconfiguration("other", 1, "gpu2"),
        ])
        .unwrap();

        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].selector_id, "experts");
        assert_eq!(batches[0].destination_device_id, "gpu1");
        assert_eq!(batches[0].resource_indices, vec![0, 2]);
        assert_eq!(batches[1].selector_id, "other");
        assert_eq!(batches[1].destination_device_id, "gpu2");
        assert_eq!(batches[1].resource_indices, vec![1]);
    }

    #[test]
    fn replica_preload_rejects_duplicate_or_local_moves_before_loading() {
        let repeated = [
            preload_reconfiguration("experts", 0, "gpu1"),
            preload_reconfiguration("experts", 0, "gpu2"),
        ];
        assert!(
            selected_resource_replica_preload_batches(&repeated)
                .unwrap_err()
                .to_string()
                .contains("repeats")
        );

        let mut local = preload_reconfiguration("experts", 0, "gpu1");
        local.moves[0].source_device_id = "gpu1".to_string();
        assert!(
            selected_resource_replica_preload_batches(&[local])
                .unwrap_err()
                .to_string()
                .contains("local move")
        );
    }
}
