struct VulkanRuntimeHybridExactCandidateResourcePlanner<'a> {
    package_root: &'a Path,
    logical_device_id_by_physical_device: &'a BTreeMap<String, String>,
    planning_devices: &'a [VulkanRuntimePhysicalPlanningDevice],
    context_capacity_activations: usize,
    speculative_draft_tokens: usize,
}

impl VulkanRuntimeHybridExactCandidateResourcePlanner<'_> {
    fn validate_bindings(&self) -> Result<(), VulkanRuntimeHybridPlacementError> {
        for device in self.planning_devices {
            let physical_device_id = &device.identity.physical_device_id;
            if self
                .logical_device_id_by_physical_device
                .get(physical_device_id)
                != Some(&device.logical_device_id)
            {
                return runtime_hybrid_error(format!(
                    "exact hybrid planning device {physical_device_id:?} is not bound to logical device {:?}",
                    device.logical_device_id,
                ));
            }
        }
        Ok(())
    }

    fn execution_case_is_eligible(
        &self,
        execution_case: &VulkanPlacementExecutionCaseIdentity,
    ) -> bool {
        let eligible_identities = self
            .planning_devices
            .iter()
            .map(|device| &device.identity)
            .collect::<BTreeSet<_>>();
        execution_case
            .devices
            .iter()
            .all(|device| eligible_identities.contains(device))
    }

    fn shared_resource_requirements(
        &self,
        runtime_model: &VulkanResidentRuntimeModel,
        phase: VulkanTargetedComponentExecutionPhase,
        component_ids: &[String],
        execution_case: &VulkanPlacementExecutionCaseIdentity,
        region_execution: Option<&VulkanPlacementRegionExecutionCalibration>,
        tensor_index: &TensorIndex,
        resource_contract: &CompiledResourceResidencyContract,
    ) -> Result<Vec<VulkanHybridSharedRangeRequirement>, VulkanRuntimeHybridPlacementError> {
        let component_cases = if let Some(region) = region_execution {
            if region.component_cases.len() != component_ids.len()
                || region.execution_case != *execution_case
            {
                return runtime_hybrid_error(
                    "exact hybrid candidate region does not match its component range",
                );
            }
            region.component_cases.as_slice()
        } else if component_ids.len() == 1 {
            std::slice::from_ref(execution_case)
        } else {
            return runtime_hybrid_error(
                "multi-component exact hybrid candidate has no region execution",
            );
        };

        let identity_by_logical_device = self
            .planning_devices
            .iter()
            .map(|device| (device.logical_device_id.clone(), device.identity.clone()))
            .collect::<BTreeMap<_, _>>();
        if identity_by_logical_device.len() != self.planning_devices.len() {
            return runtime_hybrid_error(
                "exact hybrid candidate resources repeat a logical planning device",
            );
        }
        let storage_buffer_offset_alignment = self
            .planning_devices
            .iter()
            .map(|device| device.storage_buffer_offset_alignment)
            .max()
            .unwrap_or(1);

        let mut requirements = Vec::new();
        for (component_id, component_case) in component_ids.iter().zip(component_cases) {
            requirements.extend(exact_vulkan_runtime_hybrid_component_resource_requirements(
                self.package_root,
                runtime_model,
                component_id,
                component_case,
                phase,
                self.logical_device_id_by_physical_device,
                &identity_by_logical_device,
                storage_buffer_offset_alignment,
                self.context_capacity_activations,
                self.speculative_draft_tokens,
                tensor_index,
                resource_contract,
            )?);
        }
        Ok(requirements)
    }
}

#[allow(clippy::too_many_arguments)]
fn exact_vulkan_runtime_hybrid_component_state_requirements(
    package_root: &Path,
    runtime_model: &VulkanResidentRuntimeModel,
    placed_model: &VulkanResidentRuntimeModel,
    component_id: &str,
    execution_case: &VulkanPlacementExecutionCaseIdentity,
    owner_logical_device_id: &str,
    placed_plan: &VulkanPlacedStreamCircuitPlan,
    context_capacity_activations: usize,
    speculative_draft_tokens: usize,
    tensor_index: &TensorIndex,
) -> Result<Vec<VulkanHybridSharedRangeRequirement>, VulkanRuntimeHybridPlacementError> {
    let stream = plan_component_stream_circuit_residency(
        placed_plan,
        component_id,
        context_capacity_activations,
        speculative_draft_tokens > 0,
        speculative_draft_tokens,
    )
    .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    let mut breakdown = VulkanRuntimeDeviceResidencyBreakdown {
        stream_state_bytes: stream.state_bytes,
        state_transaction_bytes: stream.transaction_bytes,
        activation_slot_bytes: stream.activation_bytes,
        causal_verification_snapshot_bytes: stream.causal_verification_snapshot_bytes,
        ..VulkanRuntimeDeviceResidencyBreakdown::default()
    };
    let (_, output_component_id) = placed_model
        .circuit_graph
        .signal_processor_endpoint_component_ids()
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    if component_id == output_component_id {
        breakdown.output_transducer_workspace_bytes = checked_residency_add(
            placed_model
                .package
                .output_transducer
                .spec
                .normalized_frame_byte_capacity,
            placed_model.package.output_transducer.spec.logits_byte_capacity,
            "output transducer workspace",
        )
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
        breakdown.sampler_workspace_bytes = sampler_workspace_bytes(
            &placed_model.package.sampler.spec,
            context_capacity_activations,
            false,
        )
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
        breakdown.feedback_workspace_bytes = main_feedback_workspace_bytes(
            &placed_model,
            context_capacity_activations,
            speculative_draft_tokens > 0,
        )
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
        if speculative_draft_tokens > 0 {
            let mut by_device =
                BTreeMap::from([(owner_logical_device_id.to_string(), breakdown)]);
            for decoder in &placed_model.package.speculative_decoders {
                plan_speculative_decoder_residency(
                    &mut by_device,
                    package_root,
                    &placed_model,
                    tensor_index,
                    decoder,
                    owner_logical_device_id,
                    context_capacity_activations,
                )
                .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
            }
            breakdown = by_device
                .remove(owner_logical_device_id)
                .expect("the output working-set device remains present");
        }
    }
    let state_bytes = sum_transient_state_breakdown(&breakdown)
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    let activation_bytes = sum_activation_headroom_breakdown(&breakdown)
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    let owner_identity = execution_case
        .devices
        .iter()
        .find(|device| device.physical_device_id == execution_case.owner_physical_device_id)
        .ok_or_else(|| {
            VulkanRuntimeHybridPlacementError(
                "exact hybrid state candidate has no owner execution identity".to_string(),
            )
        })?;
    let identity_prefix = format!(
        "runtime-state:{}:{component_id}:context:{context_capacity_activations}:draft:{speculative_draft_tokens}",
        runtime_model.execution_scope,
    );
    let mut requirements = Vec::new();
    if state_bytes > 0 {
        requirements.push(VulkanHybridSharedRangeRequirement {
            resource_identity: format!("{identity_prefix}:state"),
            target: VulkanHybridResourceTarget::Device(owner_identity.clone()),
            class: VulkanHybridResourceClass::MutableState,
            byte_offset: 0,
            byte_count: state_bytes,
        });
    }
    if activation_bytes > 0 {
        requirements.push(VulkanHybridSharedRangeRequirement {
            resource_identity: format!("{identity_prefix}:activation"),
            target: VulkanHybridResourceTarget::Device(owner_identity.clone()),
            class: VulkanHybridResourceClass::MutableState,
            byte_offset: 0,
            byte_count: activation_bytes,
        });
    }
    // One stream-control allocation is shared by every component on a target.
    // Give it a component-independent identity so graph-wide canonicalization
    // deduplicates it without merging any component-owned state.
    requirements.push(VulkanHybridSharedRangeRequirement {
        resource_identity: format!(
            "runtime-state:{}:stream-control:context:{context_capacity_activations}:draft:{speculative_draft_tokens}",
            runtime_model.execution_scope,
        ),
        target: VulkanHybridResourceTarget::Device(owner_identity.clone()),
        class: VulkanHybridResourceClass::MutableState,
        byte_offset: 0,
        byte_count: VULKAN_STREAM_CONTROL_BYTE_CAPACITY,
    });
    requirements.extend(exact_vulkan_runtime_hybrid_component_edge_requirements(
        runtime_model,
        placed_plan,
        component_id,
        owner_identity,
    )?);
    Ok(requirements)
}

fn exact_vulkan_runtime_hybrid_component_edge_requirements(
    runtime_model: &VulkanResidentRuntimeModel,
    placed_plan: &VulkanPlacedStreamCircuitPlan,
    component_id: &str,
    owner_identity: &VulkanPlacementDeviceExecutionIdentity,
) -> Result<Vec<VulkanHybridSharedRangeRequirement>, VulkanRuntimeHybridPlacementError> {
    let mut requirements = Vec::new();
    let edge_plan = VulkanPlacedEdgeIoPlan::from_placed_resident_plan(
        &placed_plan.placed_resident_plan,
    )
    .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    for (edge_index, byte_capacity) in edge_plan
        .local_edges
        .iter()
        .filter(|edge| {
            edge.source_component_id == component_id
                || edge.destination_component_id == component_id
        })
        .map(|edge| (edge.edge_index, edge.byte_capacity))
        .chain(
            edge_plan
                .endpoints
                .iter()
                .filter(|endpoint| endpoint.local_component_id == component_id)
                .map(|endpoint| (endpoint.edge_index, endpoint.byte_capacity)),
        )
    {
        let byte_count = byte_capacity.ok_or_else(|| {
            VulkanRuntimeHybridPlacementError(format!(
                "exact hybrid edge {edge_index} has no byte capacity",
            ))
        })?;
        requirements.push(VulkanHybridSharedRangeRequirement {
            resource_identity: format!(
                "runtime-state:{}:edge:{edge_index}",
                runtime_model.execution_scope,
            ),
            target: VulkanHybridResourceTarget::Device(owner_identity.clone()),
            class: VulkanHybridResourceClass::MutableState,
            byte_offset: 0,
            byte_count,
        });
    }

    let boundary_plan = VulkanModelBoundaryBufferPlan::from_placed_plan(placed_plan)
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    let mut input_identity_by_alias = BTreeMap::new();
    for input in boundary_plan
        .inputs
        .iter()
        .filter(|input| input.component_id == component_id)
    {
        let identity = format!(
            "runtime-state:{}:boundary:{}:input:{}",
            runtime_model.execution_scope, input.component_id, input.port_id,
        );
        input_identity_by_alias.insert(
            (
                input.component_id.clone(),
                input.signal_id.clone(),
                input.shape.clone(),
            ),
            identity.clone(),
        );
        requirements.push(exact_vulkan_runtime_hybrid_boundary_requirement(
            identity,
            input.byte_capacity,
            owner_identity,
        )?);
    }
    for output in boundary_plan
        .outputs
        .iter()
        .filter(|output| output.component_id == component_id)
    {
        let alias_key = (
            output.component_id.clone(),
            output.signal_id.clone(),
            output.shape.clone(),
        );
        let identity = output
            .source_signal_id
            .as_ref()
            .and_then(|_| input_identity_by_alias.get(&alias_key))
            .cloned()
            .unwrap_or_else(|| {
                format!(
                    "runtime-state:{}:boundary:{}:output:{}",
                    runtime_model.execution_scope, output.component_id, output.port_id,
                )
            });
        requirements.push(exact_vulkan_runtime_hybrid_boundary_requirement(
            identity,
            output.byte_capacity,
            owner_identity,
        )?);
    }
    Ok(requirements)
}

fn exact_vulkan_runtime_hybrid_boundary_requirement(
    resource_identity: String,
    byte_capacity: Option<usize>,
    owner_identity: &VulkanPlacementDeviceExecutionIdentity,
) -> Result<VulkanHybridSharedRangeRequirement, VulkanRuntimeHybridPlacementError> {
    let byte_count = byte_capacity.ok_or_else(|| {
        VulkanRuntimeHybridPlacementError(format!(
            "exact hybrid boundary {resource_identity:?} has no byte capacity",
        ))
    })?;
    Ok(VulkanHybridSharedRangeRequirement {
        resource_identity,
        target: VulkanHybridResourceTarget::Device(owner_identity.clone()),
        class: VulkanHybridResourceClass::MutableState,
        byte_offset: 0,
        byte_count,
    })
}

#[allow(clippy::too_many_arguments)]
fn exact_vulkan_runtime_hybrid_component_resource_requirements(
    package_root: &Path,
    runtime_model: &VulkanResidentRuntimeModel,
    component_id: &str,
    execution_case: &VulkanPlacementExecutionCaseIdentity,
    phase: VulkanTargetedComponentExecutionPhase,
    logical_device_id_by_physical_device: &BTreeMap<String, String>,
    identity_by_logical_device: &BTreeMap<String, VulkanPlacementDeviceExecutionIdentity>,
    storage_buffer_offset_alignment: usize,
    context_capacity_activations: usize,
    speculative_draft_tokens: usize,
    tensor_index: &TensorIndex,
    resource_contract: &CompiledResourceResidencyContract,
) -> Result<Vec<VulkanHybridSharedRangeRequirement>, VulkanRuntimeHybridPlacementError> {
    let owner_logical_device_id = logical_device_id_by_physical_device
        .get(&execution_case.owner_physical_device_id)
        .ok_or_else(|| {
            VulkanRuntimeHybridPlacementError(format!(
                "exact hybrid candidate owner {:?} has no logical binding",
                execution_case.owner_physical_device_id,
            ))
        })?;
    let participant_logical_device_ids = runtime_hybrid_candidate_participant_logical_device_ids(
        execution_case,
        logical_device_id_by_physical_device,
    )?;
    let mut placed_model = vulkan_runtime_model_with_component_placement(
        runtime_model,
        "hybrid:unmounted",
        &BTreeMap::from([(component_id.to_string(), owner_logical_device_id.clone())]),
    )
    .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    if execution_case.strategy != VulkanPlacementExecutionStrategy::SingleDevice {
        placed_model = placed_model
            .with_component_shard_devices(component_id, participant_logical_device_ids.clone())
            .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    }
    let slice_plan = VulkanResidentModelPackageDeviceSlicePlan::prepare_for_physical_planning(
        package_root,
        &placed_model,
        resource_contract,
        tensor_index,
        owner_logical_device_id,
        context_capacity_activations,
    )
    .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    let loaded_manifest =
        resident_package_loaded_kernel_manifest_for_slice_plans(std::slice::from_ref(&slice_plan))
            .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    let artifact_manifest = VulkanPhysicalKernelArtifactManifest::new(
        loaded_manifest
            .physical_artifacts
            .iter()
            .map(|artifact| artifact.artifact.clone())
            .collect(),
    );
    let graph = placed_model
        .executable_circuit_graph()
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    let (_, placement_plan, placed_plan) =
        plan_resident_package_placed_stream_circuit_with_tensor_index(
        owner_logical_device_id,
        &placed_model.placement,
        &graph,
        package_root,
        tensor_index,
        placed_model.package.activation_element_bytes,
    )
    .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    let (execution_phase, execution_shape) = distributed_contract_phase_and_shape(phase);
    let component_device_pools =
        if execution_case.strategy == VulkanPlacementExecutionStrategy::SingleDevice {
            BTreeMap::new()
        } else {
            BTreeMap::from([(component_id.to_string(), participant_logical_device_ids)])
        };
    let mut execution_plan =
        VulkanDistributedExecutionPlan::from_prepared_plans_for_phase_with_resource_contract(
            &[(owner_logical_device_id.as_str(), &slice_plan.prepared_plan)],
            tensor_index,
            &artifact_manifest,
            &component_device_pools,
            &placement_plan.edges,
            storage_buffer_offset_alignment,
            execution_phase,
            execution_shape,
            &placed_model.execution_scope,
            resource_contract,
        )
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    replay_exact_execution_cases_to_phase(
        &mut execution_plan,
        &BTreeMap::from([(component_id.to_string(), execution_case.clone())]),
        execution_phase,
        identity_by_logical_device,
        &loaded_manifest,
    )
    .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    let distributed_activation_requirements =
        exact_vulkan_runtime_hybrid_distributed_activation_requirements(
            runtime_model,
            &execution_plan,
            identity_by_logical_device,
        )?;
    let execution_plans = VulkanDistributedExecutionPlanSet {
        decode: execution_plan.clone(),
        decode_batch: execution_plan.clone(),
        prefill: execution_plan,
    };
    let mut requirements = vulkan_hybrid_dispatch_parameter_requirements_by_component(
        &[(owner_logical_device_id.as_str(), &slice_plan.prepared_plan)],
        &execution_plans,
        tensor_index,
        identity_by_logical_device,
        |prepared, parameter_id, actual_tensor| {
            exact_vulkan_hybrid_parameter_resource_identity_for_tensor(
                runtime_model,
                resource_contract,
                tensor_index,
                prepared,
                parameter_id,
                actual_tensor,
            )
        },
    )
    .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    append_vulkan_hybrid_graph_parameter_requirements(
        &placed_model,
        tensor_index,
        resource_contract,
        identity_by_logical_device,
        Some(&BTreeSet::from([component_id.to_string()])),
        &requirements.prepared_parameter_tensors,
        &mut requirements.requirements_by_component,
    )
    .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    let mut component_requirements = requirements
        .requirements_by_component
        .get(component_id)
        .cloned()
        .unwrap_or_default();
    component_requirements.extend(exact_vulkan_runtime_hybrid_component_state_requirements(
        package_root,
        runtime_model,
        &placed_model,
        component_id,
        execution_case,
        owner_logical_device_id,
        &placed_plan,
        context_capacity_activations,
        speculative_draft_tokens,
        tensor_index,
    )?);
    component_requirements.extend(distributed_activation_requirements);
    Ok(component_requirements)
}

fn exact_vulkan_runtime_hybrid_distributed_activation_requirements(
    runtime_model: &VulkanResidentRuntimeModel,
    execution_plan: &VulkanDistributedExecutionPlan,
    identity_by_logical_device: &BTreeMap<String, VulkanPlacementDeviceExecutionIdentity>,
) -> Result<Vec<VulkanHybridSharedRangeRequirement>, VulkanRuntimeHybridPlacementError> {
    let plan = VulkanDistributedActivationBufferPlan::from_execution_plan(execution_plan)
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    exact_vulkan_runtime_hybrid_distributed_activation_requirements_from_plan(
        runtime_model,
        &plan,
        identity_by_logical_device,
    )
}

fn exact_vulkan_runtime_hybrid_distributed_activation_requirements_from_plan(
    runtime_model: &VulkanResidentRuntimeModel,
    plan: &VulkanDistributedActivationBufferPlan,
    identity_by_logical_device: &BTreeMap<String, VulkanPlacementDeviceExecutionIdentity>,
) -> Result<Vec<VulkanHybridSharedRangeRequirement>, VulkanRuntimeHybridPlacementError> {
    let mut requirements = Vec::new();
    for allocation in &plan.allocations {
        requirements.push(exact_vulkan_runtime_hybrid_distributed_shared_requirement(
            runtime_model,
            plan.route,
            &allocation.owner_device_id,
            format!(
                "activation:{:?}:{}:slot:{}",
                allocation.storage, allocation.component_id, allocation.slot,
            ),
            allocation.byte_capacity,
            identity_by_logical_device,
        )?);
    }
    for allocation in &plan.reduction_allocations {
        requirements.push(exact_vulkan_runtime_hybrid_distributed_shared_requirement(
            runtime_model,
            plan.route,
            &allocation.owner_device_id,
            format!(
                "reduction:{}:{}:{}",
                allocation.component_id, allocation.node_id, allocation.dispatch_index,
            ),
            allocation.byte_capacity,
            identity_by_logical_device,
        )?);
    }
    for allocation in &plan.private_intermediate_allocations {
        for device in &allocation.devices {
            let physical = identity_by_logical_device
                .get(&device.device_id)
                .ok_or_else(|| {
                    VulkanRuntimeHybridPlacementError(format!(
                        "exact hybrid private activation has no physical identity for {:?}",
                        device.device_id,
                    ))
                })?;
            requirements.push(VulkanHybridSharedRangeRequirement {
                resource_identity: format!(
                    "runtime-state:{}:distributed:private:{}:{}:{}->{}",
                    runtime_model.execution_scope,
                    allocation.component_id,
                    allocation.signal_id,
                    allocation.producer_dispatch_index,
                    allocation.consumer_dispatch_index,
                ),
                target: VulkanHybridResourceTarget::Device(physical.clone()),
                class: VulkanHybridResourceClass::MutableState,
                byte_offset: 0,
                byte_count: device.byte_capacity,
            });
        }
    }
    Ok(requirements)
}

fn exact_vulkan_runtime_hybrid_distributed_shared_requirement(
    runtime_model: &VulkanResidentRuntimeModel,
    route: VulkanSharedResidentBufferRoute,
    owner_logical_device_id: &str,
    allocation_identity: String,
    byte_count: usize,
    identity_by_logical_device: &BTreeMap<String, VulkanPlacementDeviceExecutionIdentity>,
) -> Result<VulkanHybridSharedRangeRequirement, VulkanRuntimeHybridPlacementError> {
    let target = match route {
        VulkanSharedResidentBufferRoute::ExternalDeviceLocal => {
            VulkanHybridResourceTarget::Device(
                identity_by_logical_device
                    .get(owner_logical_device_id)
                    .cloned()
                    .ok_or_else(|| {
                        VulkanRuntimeHybridPlacementError(format!(
                            "exact hybrid shared activation has no physical identity for {owner_logical_device_id:?}",
                        ))
                    })?,
            )
        }
        VulkanSharedResidentBufferRoute::SharedHost => VulkanHybridResourceTarget::Host,
    };
    Ok(VulkanHybridSharedRangeRequirement {
        resource_identity: format!(
            "runtime-state:{}:distributed:{allocation_identity}",
            runtime_model.execution_scope,
        ),
        target,
        class: VulkanHybridResourceClass::MutableState,
        byte_offset: 0,
        byte_count,
    })
}

fn runtime_hybrid_candidate_participant_logical_device_ids(
    execution_case: &VulkanPlacementExecutionCaseIdentity,
    logical_device_id_by_physical_device: &BTreeMap<String, String>,
) -> Result<Vec<String>, VulkanRuntimeHybridPlacementError> {
    if execution_case.strategy == VulkanPlacementExecutionStrategy::SingleDevice {
        return logical_device_id_by_physical_device
            .get(&execution_case.owner_physical_device_id)
            .cloned()
            .map(|device| vec![device])
            .ok_or_else(|| {
                VulkanRuntimeHybridPlacementError(
                    "single-device hybrid candidate has no logical owner binding".to_string(),
                )
            });
    }
    let mut physical_by_ordinal = BTreeMap::new();
    for shard in &execution_case.shards {
        if let Some(existing) =
            physical_by_ordinal.insert(shard.participant_ordinal, shard.physical_device_id.as_str())
            && existing != shard.physical_device_id
        {
            return runtime_hybrid_error(
                "exact hybrid candidate participant ordinal changes physical device",
            );
        }
    }
    if physical_by_ordinal.is_empty()
        || physical_by_ordinal
            .keys()
            .copied()
            .ne(0..physical_by_ordinal.len())
    {
        return runtime_hybrid_error(
            "distributed hybrid candidate has an incomplete participant ordering",
        );
    }
    physical_by_ordinal
        .into_values()
        .map(|physical| {
            logical_device_id_by_physical_device
                .get(physical)
                .cloned()
                .ok_or_else(|| {
                    VulkanRuntimeHybridPlacementError(format!(
                        "exact hybrid candidate participant {physical:?} has no logical binding",
                    ))
                })
        })
        .collect()
}
