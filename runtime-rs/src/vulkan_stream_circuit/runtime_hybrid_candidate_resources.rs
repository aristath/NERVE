struct VulkanRuntimeHybridExactCandidateResourcePlanner<'a> {
    package_root: &'a Path,
    logical_device_id_by_physical_device: &'a BTreeMap<String, String>,
    planning_devices: &'a [VulkanRuntimePhysicalPlanningDevice],
    context_capacity_activations: usize,
    speculative_draft_tokens: usize,
    residency_policy: ResourceResidencyPolicy,
}

#[derive(Default)]
struct VulkanRuntimeHybridExactCandidateResourceRequirements {
    shared_ranges: Vec<VulkanHybridSharedRangeRequirement>,
    direct_claims: Vec<VulkanHybridResourceClaim>,
}

impl VulkanRuntimeHybridExactCandidateResourceRequirements {
    fn extend(&mut self, other: Self) {
        self.shared_ranges.extend(other.shared_ranges);
        self.direct_claims.extend(other.direct_claims);
    }
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

    fn resource_requirements(
        &self,
        runtime_model: &VulkanResidentRuntimeModel,
        planning_graph: &VulkanResidentPackageCircuitGraph,
        planning_basis: &VulkanResidentPackagePlanningBasis,
        phase: VulkanTargetedComponentExecutionPhase,
        component_ids: &[String],
        execution_case: &VulkanPlacementExecutionCaseIdentity,
        region_execution: Option<&VulkanPlacementRegionExecutionCalibration>,
        tensor_index: &TensorIndex,
        resource_contract: &CompiledResourceResidencyContract,
        resource_layout: &VulkanCompiledResourceAddressLayout,
    ) -> Result<
        VulkanRuntimeHybridExactCandidateResourceRequirements,
        VulkanRuntimeHybridPlacementError,
    > {
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
        let contract_alignment =
            compiled_resource_contract_minimum_upload_alignment(resource_contract)
                .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
        let upload_alignment_by_logical_device = self
            .planning_devices
            .iter()
            .map(|device| {
                (
                    device.logical_device_id.clone(),
                    device
                        .storage_buffer_offset_alignment
                        .max(contract_alignment)
                        .max(std::mem::align_of::<u64>()),
                )
            })
            .collect::<BTreeMap<_, _>>();

        let mut requirements = VulkanRuntimeHybridExactCandidateResourceRequirements::default();
        for (component_id, component_case) in component_ids.iter().zip(component_cases) {
            requirements.extend(exact_vulkan_runtime_hybrid_component_resource_requirements(
                self.package_root,
                runtime_model,
                planning_graph,
                planning_basis,
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
                resource_layout,
                &upload_alignment_by_logical_device,
                self.residency_policy,
            )?);
        }
        requirements.shared_ranges.extend(
            exact_vulkan_runtime_hybrid_internal_boundary_requirements(
                runtime_model,
                component_ids,
                component_cases,
                region_execution,
                phase,
            )?,
        );
        if let VulkanTargetedComponentExecutionPhase::Prefill {
            activation_batch_width,
        } = phase
        {
            let transient = self.prefill_transient_plan(
                runtime_model,
                planning_graph,
                planning_basis,
                component_ids,
                component_cases,
                activation_batch_width,
                storage_buffer_offset_alignment,
                tensor_index,
                resource_contract,
                resource_layout,
                &identity_by_logical_device,
            )?;
            let host_bytes = transient.host_bytes();
            for (logical_device_id, byte_count) in transient.device_bytes_by_logical_device {
                if byte_count == 0 {
                    continue;
                }
                let identity = identity_by_logical_device.get(&logical_device_id).ok_or_else(|| {
                    VulkanRuntimeHybridPlacementError(format!(
                        "exact hybrid prefill transient references unknown logical device {logical_device_id:?}",
                    ))
                })?;
                requirements.direct_claims.push(VulkanHybridResourceClaim::exclusive_device(
                    format!(
                        "runtime-execution:{}:prefill-width:{activation_batch_width}:device:{logical_device_id}",
                        runtime_model.execution_scope,
                    ),
                    identity.clone(),
                    VulkanHybridResourceClass::ExecutionTransient,
                    byte_count,
                ));
            }
            if host_bytes > 0 {
                requirements.direct_claims.push(VulkanHybridResourceClaim::exclusive_host(
                    format!(
                        "runtime-execution:{}:prefill-width:{activation_batch_width}:host-staging",
                        runtime_model.execution_scope,
                    ),
                    VulkanHybridResourceClass::ExecutionTransient,
                    host_bytes,
                ));
            }
        }
        Ok(requirements)
    }

    fn route_execution_transient_claims(
        &self,
        runtime_model: &VulkanResidentRuntimeModel,
        phase: VulkanTargetedComponentExecutionPhase,
        placement: &VulkanRuntimeHybridOrderedPlacement,
    ) -> Result<Vec<VulkanHybridResourceClaim>, VulkanRuntimeHybridPlacementError> {
        let VulkanTargetedComponentExecutionPhase::Prefill {
            activation_batch_width,
        } = phase
        else {
            return Ok(Vec::new());
        };
        if placement.component_ids
            != runtime_model
                .circuit_graph
                .components
                .iter()
                .filter(|component| component.runtime_role.is_signal_processor())
                .map(|component| component.component_id.clone())
                .collect::<Vec<_>>()
            || placement.execution_phase != nerve_execution_contracts::ExecutionPhase::Prefill
            || placement.activation_batch_width != activation_batch_width
        {
            return runtime_hybrid_error(
                "exact route transient planning received a different graph or phase",
            );
        }
        let mut component_cases = Vec::with_capacity(placement.component_ids.len());
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
            if *component_start != component_cases.len() {
                return runtime_hybrid_error(
                    "exact route transient planning found a noncontiguous region",
                );
            }
            component_cases.extend(
                runtime_hybrid_step_component_cases(
                    placement,
                    *component_start,
                    *component_end,
                    execution_case,
                )?
                .into_iter()
                .cloned(),
            );
        }
        if component_cases.len() != placement.component_ids.len() {
            return runtime_hybrid_error(
                "exact route transient planning does not cover every component",
            );
        }
        let tensor_index = runtime_model
            .load_runtime_tensor_index(self.package_root)
            .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
        let resource_contract = instantiate_runtime_resource_contract(runtime_model)
            .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
        let resource_layout = VulkanCompiledResourceAddressLayout::from_contract(
            &resource_contract,
        )
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
        let planning_graph = runtime_model
            .executable_circuit_graph()
            .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
        let planning_basis = prepare_resident_package_planning_basis(
            &planning_graph,
            self.package_root,
            &tensor_index,
        )
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
        let identity_by_logical_device = self
            .planning_devices
            .iter()
            .map(|device| (device.logical_device_id.clone(), device.identity.clone()))
            .collect::<BTreeMap<_, _>>();
        let transient = self.prefill_transient_plan(
            runtime_model,
            &planning_graph,
            &planning_basis,
            &placement.component_ids,
            &component_cases,
            activation_batch_width,
            self.planning_devices
                .iter()
                .map(|device| device.storage_buffer_offset_alignment)
                .max()
                .unwrap_or(1),
            &tensor_index,
            &resource_contract,
            &resource_layout,
            &identity_by_logical_device,
        )?;
        let host_bytes = transient.host_bytes();
        let mut claims = Vec::new();
        for (logical_device_id, byte_count) in transient.device_bytes_by_logical_device {
            if byte_count == 0 {
                continue;
            }
            let identity = identity_by_logical_device
                .get(&logical_device_id)
                .ok_or_else(|| {
                    VulkanRuntimeHybridPlacementError(format!(
                        "exact route transient references unknown logical device {logical_device_id:?}",
                    ))
                })?;
            claims.push(VulkanHybridResourceClaim::exclusive_device(
                format!(
                    "runtime-route:{}:prefill-width:{activation_batch_width}:device:{logical_device_id}",
                    runtime_model.execution_scope,
                ),
                identity.clone(),
                VulkanHybridResourceClass::ExecutionTransient,
                byte_count,
            ));
        }
        if host_bytes > 0 {
            claims.push(VulkanHybridResourceClaim::exclusive_host(
                format!(
                    "runtime-route:{}:prefill-width:{activation_batch_width}:host",
                    runtime_model.execution_scope,
                ),
                VulkanHybridResourceClass::ExecutionTransient,
                host_bytes,
            ));
        }
        Ok(claims)
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_transient_plan(
        &self,
        runtime_model: &VulkanResidentRuntimeModel,
        planning_graph: &VulkanResidentPackageCircuitGraph,
        planning_basis: &VulkanResidentPackagePlanningBasis,
        component_ids: &[String],
        component_cases: &[VulkanPlacementExecutionCaseIdentity],
        activation_batch_width: usize,
        storage_buffer_offset_alignment: usize,
        tensor_index: &TensorIndex,
        resource_contract: &CompiledResourceResidencyContract,
        resource_layout: &VulkanCompiledResourceAddressLayout,
        identity_by_logical_device: &BTreeMap<
            String,
            VulkanPlacementDeviceExecutionIdentity,
        >,
    ) -> Result<VulkanRuntimeHybridExecutionTransientPlan, VulkanRuntimeHybridPlacementError> {
        let mut placement = BTreeMap::new();
        let mut component_device_pools = BTreeMap::new();
        let mut exact_cases = BTreeMap::new();
        let mut owner_logical_device_ids = BTreeSet::new();
        for (component_id, execution_case) in component_ids.iter().zip(component_cases) {
            let owner = self
                .logical_device_id_by_physical_device
                .get(&execution_case.owner_physical_device_id)
                .ok_or_else(|| {
                    VulkanRuntimeHybridPlacementError(format!(
                        "exact hybrid prefill owner {:?} has no logical binding",
                        execution_case.owner_physical_device_id,
                    ))
                })?
                .clone();
            placement.insert(component_id.clone(), owner.clone());
            owner_logical_device_ids.insert(owner);
            if execution_case.strategy != VulkanPlacementExecutionStrategy::SingleDevice {
                let participants = runtime_hybrid_candidate_participant_logical_device_ids(
                    execution_case,
                    self.logical_device_id_by_physical_device,
                )?;
                component_device_pools.insert(component_id.clone(), participants);
            }
            exact_cases.insert(component_id.clone(), execution_case.clone());
        }
        let mut placed_model = vulkan_runtime_model_with_component_placement(
            runtime_model,
            "hybrid:unmounted",
            &placement,
        )
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
        for (component_id, device_ids) in &component_device_pools {
            placed_model = placed_model
                .with_component_shard_devices(component_id, device_ids.clone())
                .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
        }
        let mut slice_plans = Vec::new();
        for logical_device_id in &owner_logical_device_ids {
            slice_plans.push(
                VulkanResidentModelPackageDeviceSlicePlan::prepare_for_physical_planning_from_basis(
                    self.package_root,
                    &placed_model,
                    resource_contract,
                    tensor_index,
                    logical_device_id,
                    self.context_capacity_activations,
                    planning_graph,
                    planning_basis,
                )
                .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?,
            );
        }
        let loaded_manifest = resident_package_loaded_kernel_manifest_for_slice_plans(&slice_plans)
            .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
        let artifact_manifest = VulkanPhysicalKernelArtifactManifest::new(
            loaded_manifest
                .physical_artifacts
                .iter()
                .map(|artifact| artifact.artifact.clone())
                .collect(),
        );
        let (placement_plan, _) = plan_resident_package_from_planning_basis(
            "hybrid:unmounted",
            &placed_model.placement,
            planning_graph,
            tensor_index,
            placed_model.package.activation_element_bytes,
            planning_basis,
        )
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
        let prepared_plans = slice_plans
            .iter()
            .map(|slice| (slice.device_id.as_str(), &slice.prepared_plan))
            .collect::<Vec<_>>();
        let mut execution_plan =
            VulkanDistributedExecutionPlan::from_prepared_plans_for_phase_with_resource_contract(
                &prepared_plans,
                tensor_index,
                &artifact_manifest,
                &component_device_pools,
                &placement_plan.edges,
                storage_buffer_offset_alignment,
                nerve_execution_contracts::ExecutionPhase::Prefill,
                nerve_execution_contracts::ExecutionShape::MultiLane,
                &placed_model.execution_scope,
                resource_contract,
            )
            .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
        replay_exact_execution_cases_to_phase(
            &mut execution_plan,
            &exact_cases,
            nerve_execution_contracts::ExecutionPhase::Prefill,
            identity_by_logical_device,
            &loaded_manifest,
        )
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
        let component_ids = component_ids.iter().cloned().collect::<BTreeSet<_>>();
        exact_vulkan_runtime_hybrid_prefill_runners_transient_plan(
            &placed_model,
            &component_ids,
            &slice_plans,
            &execution_plan,
            activation_batch_width,
            resource_contract,
            resource_layout,
            self.residency_policy,
            self.speculative_draft_tokens,
        )
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
            let mut allocations_by_device =
                BTreeMap::from([(owner_logical_device_id.to_string(), Vec::new())]);
            for decoder in &placed_model.package.speculative_decoders {
                plan_speculative_decoder_residency(
                    &mut by_device,
                    &mut allocations_by_device,
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
    let signal_processor_ids = runtime_model
        .circuit_graph
        .components
        .iter()
        .filter(|component| component.runtime_role.is_signal_processor())
        .map(|component| component.component_id.as_str())
        .collect::<BTreeSet<_>>();
    let edge_plan = VulkanPlacedEdgeIoPlan::from_placed_resident_plan(
        &placed_plan.placed_resident_plan,
    )
    .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    for (source_component_id, source_port_id, edge_index, byte_capacity) in edge_plan
        .local_edges
        .iter()
        .filter_map(|edge| {
            let candidate_owns_backing = if signal_processor_ids
                .contains(edge.source_component_id.as_str())
            {
                edge.source_component_id == component_id
            } else {
                edge.destination_component_id == component_id
            };
            candidate_owns_backing.then_some((
                edge.source_component_id.as_str(),
                edge.source_port_id.as_str(),
                edge.edge_index,
                edge.byte_capacity,
            ))
        })
        .chain(
            edge_plan
                .endpoints
                .iter()
                .filter_map(|endpoint| {
                    let (source_component_id, source_port_id, destination_component_id) =
                        match endpoint.direction {
                            VulkanPlacedEdgeDirection::Outgoing => (
                                endpoint.local_component_id.as_str(),
                                endpoint.local_port_id.as_str(),
                                endpoint.remote_component_id.as_str(),
                            ),
                            VulkanPlacedEdgeDirection::Incoming => (
                                endpoint.remote_component_id.as_str(),
                                endpoint.remote_port_id.as_str(),
                                endpoint.local_component_id.as_str(),
                            ),
                        };
                    let candidate_owns_backing = if signal_processor_ids
                        .contains(source_component_id)
                    {
                        source_component_id == component_id
                    } else {
                        destination_component_id == component_id
                    };
                    candidate_owns_backing.then_some((
                        source_component_id,
                        source_port_id,
                        endpoint.edge_index,
                        endpoint.byte_capacity,
                    ))
                }),
        )
    {
        let byte_count = byte_capacity.ok_or_else(|| {
            VulkanRuntimeHybridPlacementError(format!(
                "exact hybrid edge {edge_index} has no byte capacity",
            ))
        })?;
        requirements.push(VulkanHybridSharedRangeRequirement {
            resource_identity: format!(
                "runtime-state:{}:produced-port:{source_component_id}:{source_port_id}",
                runtime_model.execution_scope
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

fn exact_vulkan_runtime_hybrid_internal_boundary_requirements(
    runtime_model: &VulkanResidentRuntimeModel,
    component_ids: &[String],
    component_cases: &[VulkanPlacementExecutionCaseIdentity],
    region_execution: Option<&VulkanPlacementRegionExecutionCalibration>,
    phase: VulkanTargetedComponentExecutionPhase,
) -> Result<Vec<VulkanHybridSharedRangeRequirement>, VulkanRuntimeHybridPlacementError> {
    if component_ids.len() < 2 {
        return Ok(Vec::new());
    }
    let region = region_execution.ok_or_else(|| {
        VulkanRuntimeHybridPlacementError(
            "multi-component exact hybrid candidate has no internal boundary replay".to_string(),
        )
    })?;
    let ordered_components = runtime_model
        .circuit_graph
        .components
        .iter()
        .filter(|component| component.runtime_role.is_signal_processor())
        .map(|component| component.component_id.as_str())
        .collect::<Vec<_>>();
    let component_start = ordered_components
        .iter()
        .position(|candidate| *candidate == component_ids[0])
        .ok_or_else(|| {
            VulkanRuntimeHybridPlacementError(
                "exact hybrid region begins at an unknown signal processor".to_string(),
            )
        })?;
    if ordered_components
        .get(component_start..component_start + component_ids.len())
        .is_none_or(|ordered| {
            ordered
                .iter()
                .copied()
                .ne(component_ids.iter().map(String::as_str))
        })
    {
        return runtime_hybrid_error(
            "exact hybrid region components are not one contiguous graph range",
        );
    }
    let boundaries = vulkan_runtime_placement_boundaries(runtime_model)
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    let boundary_cases = region
        .boundary_cases
        .iter()
        .map(|boundary| (boundary.boundary_ordinal, &boundary.execution_case))
        .collect::<BTreeMap<_, _>>();
    let width = phase.activation_batch_width();
    let mut requirements = Vec::new();
    for local_boundary in 0..component_ids.len() - 1 {
        let source = &component_cases[local_boundary];
        let destination = &component_cases[local_boundary + 1];
        let crosses_devices =
            source.output_physical_device_id != destination.input_physical_device_id;
        let boundary_case = boundary_cases.get(&local_boundary).copied();
        if !crosses_devices {
            if boundary_case.is_some() {
                return runtime_hybrid_error(
                    "same-device exact hybrid region unexpectedly retains a boundary route",
                );
            }
            continue;
        }
        let boundary_case = boundary_case.ok_or_else(|| {
            VulkanRuntimeHybridPlacementError(format!(
                "exact hybrid region omits cross-device boundary {local_boundary}",
            ))
        })?;
        let global_boundary = component_start + local_boundary;
        let [transfer] = boundaries[global_boundary].transfers.as_slice() else {
            return runtime_hybrid_error(
                "exact hybrid internal boundary does not contain exactly one transfer",
            );
        };
        let base_byte_count = transfer.byte_count;
        let expected_case_bytes = base_byte_count.checked_mul(width).ok_or_else(|| {
            VulkanRuntimeHybridPlacementError(
                "exact hybrid internal boundary byte geometry overflowed".to_string(),
            )
        })?;
        if !runtime_hybrid_boundary_execution_case_is_compatible(
            runtime_hybrid_execution_phase(phase)?,
            Some(width),
            expected_case_bytes,
            boundary_case,
        ) {
            return runtime_hybrid_error(
                "exact hybrid internal boundary replay has incompatible geometry",
            );
        }
        append_exact_vulkan_runtime_hybrid_staged_boundary_requirements(
            runtime_model,
            global_boundary,
            base_byte_count,
            boundary_case,
            &mut requirements,
        )?;
    }
    Ok(requirements)
}

fn append_exact_vulkan_runtime_hybrid_staged_boundary_requirements(
    runtime_model: &VulkanResidentRuntimeModel,
    boundary_index: usize,
    base_byte_count: usize,
    execution_case: &VulkanPlacementExecutionCaseIdentity,
    requirements: &mut Vec<VulkanHybridSharedRangeRequirement>,
) -> Result<(), VulkanRuntimeHybridPlacementError> {
    let [transport] = execution_case.transports.as_slice() else {
        return runtime_hybrid_error(
            "exact hybrid boundary must contain exactly one transport route",
        );
    };
    match runtime_mounted_boundary_route(&transport.route)? {
        VulkanPlacedEdgeTransferRoute::ExternalDeviceLocal => Ok(()),
        VulkanPlacedEdgeTransferRoute::DeviceLocalStaging => {
            let destination = execution_case
                .devices
                .iter()
                .find(|device| {
                    device.physical_device_id == execution_case.output_physical_device_id
                })
                .ok_or_else(|| {
                    VulkanRuntimeHybridPlacementError(
                        "exact hybrid staged boundary has no destination identity".to_string(),
                    )
                })?;
            let resource_identity = format!(
                "runtime-state:{}:boundary:{boundary_index}:staging",
                runtime_model.execution_scope,
            );
            requirements.push(VulkanHybridSharedRangeRequirement {
                resource_identity: format!("{resource_identity}:destination"),
                target: VulkanHybridResourceTarget::Device(destination.clone()),
                class: VulkanHybridResourceClass::MutableState,
                byte_offset: 0,
                byte_count: base_byte_count,
            });
            requirements.push(VulkanHybridSharedRangeRequirement {
                resource_identity: format!("{resource_identity}:host"),
                target: VulkanHybridResourceTarget::Host,
                class: VulkanHybridResourceClass::MutableState,
                byte_offset: 0,
                byte_count: base_byte_count,
            });
            Ok(())
        }
        route => runtime_hybrid_error(format!(
            "exact hybrid boundary selected unsupported mounted route {route:?}",
        )),
    }
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
    planning_graph: &VulkanResidentPackageCircuitGraph,
    planning_basis: &VulkanResidentPackagePlanningBasis,
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
    resource_layout: &VulkanCompiledResourceAddressLayout,
    upload_alignment_by_logical_device: &BTreeMap<String, usize>,
    residency_policy: ResourceResidencyPolicy,
) -> Result<
    VulkanRuntimeHybridExactCandidateResourceRequirements,
    VulkanRuntimeHybridPlacementError,
> {
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
    let slice_plan = VulkanResidentModelPackageDeviceSlicePlan::prepare_for_physical_planning_from_basis(
        package_root,
        &placed_model,
        resource_contract,
        tensor_index,
        owner_logical_device_id,
        context_capacity_activations,
        planning_graph,
        planning_basis,
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
    let (placement_plan, placed_plan) = plan_resident_package_from_planning_basis(
        owner_logical_device_id,
        &placed_model.placement,
        planning_graph,
        tensor_index,
        placed_model.package.activation_element_bytes,
        planning_basis,
    )
    .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    let (execution_phase, execution_shape) = distributed_contract_phase_and_shape(phase);
    let component_device_pools =
        if execution_case.strategy == VulkanPlacementExecutionStrategy::SingleDevice {
            BTreeMap::new()
        } else {
            BTreeMap::from([(
                component_id.to_string(),
                participant_logical_device_ids.clone(),
            )])
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
    let selected_resource_requirements =
        exact_vulkan_runtime_hybrid_selected_resource_requirements(
            &placed_model,
            component_id,
            &execution_plan,
            &participant_logical_device_ids,
            speculative_draft_tokens > 0,
            resource_contract,
            resource_layout,
            identity_by_logical_device,
            upload_alignment_by_logical_device,
            residency_policy,
        )?;
    let distributed_activation_requirements =
        exact_vulkan_runtime_hybrid_distributed_activation_requirements(
            runtime_model,
            &execution_plan,
            identity_by_logical_device,
        )?;
    let execution_plans = VulkanDistributedExecutionPlanSet {
        decode: execution_plan.clone(),
        decode_batch: execution_plan.clone(),
        prefill: execution_plan.clone(),
    };
    let fixed_resource_identities =
        VulkanHybridFixedResourceIdentityIndex::new(resource_contract);
    let mut requirements = vulkan_hybrid_dispatch_parameter_requirements_by_component(
        &[(owner_logical_device_id.as_str(), &slice_plan.prepared_plan)],
        &execution_plans,
        tensor_index,
        identity_by_logical_device,
        |prepared, parameter_id, actual_tensor| {
            exact_vulkan_hybrid_parameter_resource_identity_for_tensor(
                runtime_model,
                &fixed_resource_identities,
                tensor_index,
                prepared,
                parameter_id,
                actual_tensor,
            )
        },
    )
    .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    let graph_parameter_component_ids =
        exact_vulkan_runtime_hybrid_graph_parameter_anchor_ids(runtime_model, component_id)?;
    append_vulkan_hybrid_graph_parameter_requirements(
        &placed_model,
        tensor_index,
        resource_contract,
        identity_by_logical_device,
        Some(&graph_parameter_component_ids),
        &requirements.prepared_parameter_tensors,
        &mut requirements.requirements_by_component,
    )
    .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    let mut component_requirements = graph_parameter_component_ids
        .iter()
        .flat_map(|anchor_id| {
            requirements
                .requirements_by_component
                .get(anchor_id)
                .into_iter()
                .flatten()
                .cloned()
        })
        .collect::<Vec<_>>();
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
    component_requirements.extend(selected_resource_requirements.shared_ranges);
    if phase == VulkanTargetedComponentExecutionPhase::Decode {
        let gate_bytes = exact_vulkan_runtime_hybrid_gate_device_bytes(
            &BTreeSet::from([component_id.to_string()]),
            &BTreeMap::from([(
                component_id.to_string(),
                owner_logical_device_id.to_string(),
            )]),
            &execution_plan,
            1,
            resource_contract,
            resource_layout,
            residency_policy,
        )?;
        for (logical_device_id, byte_count) in gate_bytes {
            if byte_count == 0 {
                continue;
            }
            let identity = identity_by_logical_device.get(&logical_device_id).ok_or_else(|| {
                VulkanRuntimeHybridPlacementError(format!(
                    "exact hybrid decode gate references unknown logical device {logical_device_id:?}",
                ))
            })?;
            component_requirements.push(VulkanHybridSharedRangeRequirement {
                resource_identity: format!(
                    "runtime-gates:{}:{component_id}:decode:{logical_device_id}",
                    runtime_model.execution_scope,
                ),
                target: VulkanHybridResourceTarget::Device(identity.clone()),
                class: VulkanHybridResourceClass::MutableState,
                byte_offset: 0,
                byte_count,
            });
        }
    }
    Ok(VulkanRuntimeHybridExactCandidateResourceRequirements {
        shared_ranges: component_requirements,
        direct_claims: selected_resource_requirements.direct_claims,
    })
}

/// Assigns graph-owned parameters to the signal component that determines
/// their physical endpoint. Input-transducer resources follow the first signal
/// owner; output, sampler, and draft resources follow the last. Keeping these
/// fixed bytes in candidate admission prevents the terminal mount verifier from
/// rediscovering that an otherwise complete route overfilled its endpoint GPU.
fn exact_vulkan_runtime_hybrid_graph_parameter_anchor_ids(
    runtime_model: &VulkanResidentRuntimeModel,
    component_id: &str,
) -> Result<BTreeSet<String>, VulkanRuntimeHybridPlacementError> {
    let signal_component_ids = runtime_model
        .circuit_graph
        .components
        .iter()
        .filter(|component| component.runtime_role.is_signal_processor())
        .map(|component| component.component_id.as_str())
        .collect::<Vec<_>>();
    let (Some(first), Some(last)) = (
        signal_component_ids.first().copied(),
        signal_component_ids.last().copied(),
    ) else {
        return runtime_hybrid_error(
            "exact hybrid graph parameter planning found no signal processor",
        );
    };
    if !signal_component_ids.contains(&component_id) {
        return runtime_hybrid_error(format!(
            "exact hybrid graph parameter planning references unknown signal processor {component_id:?}",
        ));
    }
    Ok(runtime_model
        .circuit_graph
        .components
        .iter()
        .filter(|component| match component.runtime_role {
            CircuitRuntimeRole::SignalProcessor => component.component_id == component_id,
            CircuitRuntimeRole::InputTransducer => component_id == first,
            CircuitRuntimeRole::OutputTransducer
            | CircuitRuntimeRole::Sampler
            | CircuitRuntimeRole::DraftProcessor
            | CircuitRuntimeRole::DraftInputAdapter
            | CircuitRuntimeRole::DraftOutputTransducer => component_id == last,
        })
        .map(|component| component.component_id.clone())
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn exact_vulkan_runtime_hybrid_selected_resource_requirements(
    runtime_model: &VulkanResidentRuntimeModel,
    component_id: &str,
    execution_plan: &VulkanDistributedExecutionPlan,
    participant_logical_device_ids: &[String],
    mount_speculative_decoders: bool,
    resource_contract: &CompiledResourceResidencyContract,
    resource_layout: &VulkanCompiledResourceAddressLayout,
    identity_by_logical_device: &BTreeMap<String, VulkanPlacementDeviceExecutionIdentity>,
    upload_alignment_by_logical_device: &BTreeMap<String, usize>,
    residency_policy: ResourceResidencyPolicy,
) -> Result<
    VulkanRuntimeHybridExactCandidateResourceRequirements,
    VulkanRuntimeHybridPlacementError,
> {
    let mut component_plan = execution_plan.clone();
    component_plan
        .dispatches
        .retain(|dispatch| dispatch.component_id == component_id);
    let distributed_store_plan = VulkanDistributedSelectedResourceStorePlan::from_execution_plan(
        &component_plan,
    )
    .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    let (input_component_id, output_component_id) = runtime_model
        .circuit_graph
        .signal_processor_endpoint_component_ids()
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    let input_device_id = runtime_model
        .placement
        .device_for_component(&input_component_id);
    let output_device_id = runtime_model
        .placement
        .device_for_component(&output_component_id);
    let mut requirements = VulkanRuntimeHybridExactCandidateResourceRequirements::default();
    for logical_device_id in participant_logical_device_ids {
        let identity = identity_by_logical_device
            .get(logical_device_id)
            .ok_or_else(|| {
                VulkanRuntimeHybridPlacementError(format!(
                    "exact hybrid selected-resource store has no physical identity for {:?}",
                    logical_device_id,
                ))
            })?;
        let upload_alignment = upload_alignment_by_logical_device
            .get(logical_device_id)
            .copied()
            .ok_or_else(|| {
                VulkanRuntimeHybridPlacementError(format!(
                    "exact hybrid selected-resource store has no upload alignment for {:?}",
                    logical_device_id,
                ))
            })?;
        let logical_device_ids = BTreeSet::from([logical_device_id.clone()]);
        let Some(ownership) = compiled_resource_selector_ownership_for_device_set(
            runtime_model,
            resource_contract,
            input_device_id,
            output_device_id,
            &logical_device_ids,
            mount_speculative_decoders,
            &distributed_store_plan,
        )
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?
        else {
            continue;
        };
        let parameters =
            plan_compiled_parameter_residency_for_device_set_with_selector_ownership(
                runtime_model,
                resource_contract,
                input_device_id,
                output_device_id,
                &logical_device_ids,
                mount_speculative_decoders,
                residency_policy,
                &ownership,
            )
            .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
        if parameters.staging_headroom_bytes == 0 {
            return runtime_hybrid_error(format!(
                "exact hybrid selected-resource store on {logical_device_id:?} has no atomic load wave",
            ));
        }
        let store = plan_compiled_resource_store_residency_for_ownership(
            resource_contract,
            resource_layout,
            &ownership,
            parameters.staging_headroom_bytes,
            upload_alignment,
        )
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
        let payload_bytes_by_slot = resource_layout
            .source_payload_bytes_by_address_slot_for_ownership(resource_contract, &ownership)
            .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
        let dynamic_addressable_bytes = parameters
            .maximum_addressable_bytes
            .checked_sub(parameters.always_resident_bytes)
            .ok_or_else(|| {
                VulkanRuntimeHybridPlacementError(
                    "exact hybrid dynamic addressability underflowed".to_string(),
                )
            })?;
        let payload_bytes = payload_bytes_by_slot.values().try_fold(
            0usize,
            |total, bytes| {
                total.checked_add(*bytes).ok_or_else(|| {
                    VulkanRuntimeHybridPlacementError(
                        "exact hybrid selected-resource payload bytes overflowed".to_string(),
                    )
                })
            },
        )?;
        if payload_bytes != dynamic_addressable_bytes {
            return runtime_hybrid_error(format!(
                "exact hybrid selected-resource slots contain {payload_bytes} bytes but residency on {logical_device_id:?} addresses {dynamic_addressable_bytes}",
            ));
        }
        append_exact_vulkan_runtime_hybrid_store_fixed_requirements(
            runtime_model,
            component_id,
            identity,
            resource_layout,
            &ownership,
            &store,
            upload_alignment,
            &mut requirements.shared_ranges,
        )?;
        if !execution_plan.dispatches.is_empty() && residency_policy.is_demand_loaded() {
            append_exact_vulkan_runtime_hybrid_shared_bytes(
                &mut requirements.shared_ranges,
                format!(
                    "runtime-cache:{}:transaction-predicate",
                    runtime_model.execution_scope,
                ),
                identity,
                VulkanHybridResourceClass::MutableState,
                size_of::<u32>(),
            );
        }
        append_exact_vulkan_runtime_hybrid_cache_requirements(
            runtime_model,
            component_id,
            identity,
            &payload_bytes_by_slot,
            store.maximum_load_wave_payload_bytes,
            store
                .retained_representation_cache_device_bytes()
                .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?,
            store.retained_representation_cache_identity.as_deref(),
            residency_policy,
            &mut requirements,
        )?;
    }
    Ok(requirements)
}

#[allow(clippy::too_many_arguments)]
fn append_exact_vulkan_runtime_hybrid_store_fixed_requirements(
    runtime_model: &VulkanResidentRuntimeModel,
    component_id: &str,
    identity: &VulkanPlacementDeviceExecutionIdentity,
    resource_layout: &VulkanCompiledResourceAddressLayout,
    ownership: &VulkanCompiledResourceSelectorOwnership,
    store: &VulkanCompiledResourceStoreResidencyBytes,
    upload_alignment: usize,
    requirements: &mut Vec<VulkanHybridSharedRangeRequirement>,
) -> Result<(), VulkanRuntimeHybridPlacementError> {
    let prefix = format!("runtime-cache:{}", runtime_model.execution_scope);
    append_exact_vulkan_runtime_hybrid_shared_bytes(
        requirements,
        format!("{prefix}:address-table"),
        identity,
        VulkanHybridResourceClass::MutableState,
        store.address_table_device_bytes,
    );
    append_exact_vulkan_runtime_hybrid_shared_bytes(
        requirements,
        format!("{prefix}:parameter-slots:{component_id}"),
        identity,
        VulkanHybridResourceClass::MutableState,
        store.parameter_slot_table_device_bytes,
    );
    for slot in 0..store.transfer_staging_slot_count {
        append_exact_vulkan_runtime_hybrid_shared_bytes(
            requirements,
            format!("{prefix}:transfer-staging:slot:{slot}"),
            identity,
            VulkanHybridResourceClass::MutableState,
            store.transfer_staging_slot_byte_capacity,
        );
    }
    let padding_per_slot = upload_alignment.saturating_sub(1);
    if padding_per_slot > 0 {
        for slot in resource_layout
            .addressable_slots_for_ownership(ownership)
            .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?
        {
            append_exact_vulkan_runtime_hybrid_shared_bytes(
                requirements,
                format!("{prefix}:allocation-padding:slot:{slot}"),
                identity,
                VulkanHybridResourceClass::MutableState,
                padding_per_slot,
            );
        }
    }
    let planned_fixed = store
        .maximum_source_extra_device_bytes()
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    let emitted_fixed = store
        .address_table_device_bytes
        .checked_add(store.parameter_slot_table_device_bytes)
        .and_then(|bytes| bytes.checked_add(store.transfer_staging_device_bytes))
        .and_then(|bytes| bytes.checked_add(store.maximum_dynamic_allocation_padding_bytes))
        .ok_or_else(|| {
            VulkanRuntimeHybridPlacementError(
                "exact hybrid selected-resource emitted fixed bytes overflowed".to_string(),
            )
        })?;
    if emitted_fixed != planned_fixed {
        return runtime_hybrid_error(
            "exact hybrid selected-resource fixed claims disagree with store residency",
        );
    }
    Ok(())
}

fn append_exact_vulkan_runtime_hybrid_cache_requirements(
    runtime_model: &VulkanResidentRuntimeModel,
    component_id: &str,
    identity: &VulkanPlacementDeviceExecutionIdentity,
    payload_bytes_by_slot: &BTreeMap<usize, usize>,
    maximum_load_wave_bytes: usize,
    retained_representation_cache_bytes: usize,
    retained_representation_cache_identity: Option<&str>,
    residency_policy: ResourceResidencyPolicy,
    requirements: &mut VulkanRuntimeHybridExactCandidateResourceRequirements,
) -> Result<(), VulkanRuntimeHybridPlacementError> {
    if residency_policy == ResourceResidencyPolicy::DemandPaged {
        append_exact_vulkan_runtime_hybrid_shared_bytes(
            &mut requirements.shared_ranges,
            format!("runtime-cache:{}:payload-arena", runtime_model.execution_scope),
            identity,
            VulkanHybridResourceClass::CacheQuota,
            maximum_load_wave_bytes,
        );
    } else {
        for (slot, bytes) in payload_bytes_by_slot {
            append_exact_vulkan_runtime_hybrid_shared_bytes(
                &mut requirements.shared_ranges,
                format!(
                    "runtime-cache:{}:retained-payload-slot:{slot}",
                    runtime_model.execution_scope,
                ),
                identity,
                VulkanHybridResourceClass::CacheQuota,
                *bytes,
            );
        }
    }
    if retained_representation_cache_bytes > 0 {
        let retained_representation_cache_identity = retained_representation_cache_identity
            .ok_or_else(|| {
                VulkanRuntimeHybridPlacementError(
                    "retained representation cache has bytes but no exact identity".to_string(),
                )
            })?;
        append_exact_vulkan_runtime_hybrid_shared_bytes(
            &mut requirements.shared_ranges,
            format!(
                "runtime-cache:{}:retained-representations:{retained_representation_cache_identity}",
                runtime_model.execution_scope,
            ),
            identity,
            VulkanHybridResourceClass::CacheQuota,
            retained_representation_cache_bytes,
        );
    } else if retained_representation_cache_identity.is_some() {
        return runtime_hybrid_error(
            "retained representation cache has an identity but no physical bytes",
        );
    }
    requirements
        .direct_claims
        .push(VulkanHybridResourceClaim::device(
            format!(
                "exact-load-wave:{}:{component_id}:{}:{}:{}:{}",
                runtime_model.execution_scope,
                identity.physical_device_id,
                identity.api_version,
                identity.driver_version,
                maximum_load_wave_bytes,
            ),
            identity.clone(),
            VulkanHybridResourceClass::AtomicLoadWave,
            maximum_load_wave_bytes,
        ));
    Ok(())
}

fn append_exact_vulkan_runtime_hybrid_shared_bytes(
    requirements: &mut Vec<VulkanHybridSharedRangeRequirement>,
    resource_identity: String,
    identity: &VulkanPlacementDeviceExecutionIdentity,
    class: VulkanHybridResourceClass,
    byte_count: usize,
) {
    if byte_count > 0 {
        requirements.push(VulkanHybridSharedRangeRequirement {
            resource_identity,
            target: VulkanHybridResourceTarget::Device(identity.clone()),
            class,
            byte_offset: 0,
            byte_count,
        });
    }
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
