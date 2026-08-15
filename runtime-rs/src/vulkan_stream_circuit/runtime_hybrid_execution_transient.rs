#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct VulkanRuntimeHybridExecutionTransientPlan {
    device_bytes_by_logical_device: BTreeMap<String, usize>,
    device_allocations: Vec<VulkanRuntimeDeviceLocalTransientAllocation>,
    host_visible_allocations: Vec<VulkanRuntimeHostVisibleTransientAllocation>,
    shared_host_allocations: Vec<VulkanRuntimeSharedHostTransientAllocation>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VulkanRuntimeStreamAllocationClass {
    Permanent,
    PromptRunner,
    VerificationRunner,
    CatchUpRunner,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VulkanRuntimeDeviceLocalTransientAllocation {
    pub logical_device_id: String,
    pub participant_device_ids: Vec<String>,
    pub byte_capacity: usize,
    pub concern: String,
    pub usage: VulkanRuntimeDeviceLocalTransientAllocationUsage,
    pub allocation_class: VulkanRuntimeStreamAllocationClass,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VulkanRuntimeDeviceLocalTransientAllocationUsage {
    #[default]
    Storage,
    ConditionalPredicate,
    ExternalSharedStorage,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VulkanRuntimeHostVisibleTransientAllocation {
    pub logical_device_id: String,
    pub byte_capacity: usize,
    pub concern: String,
    pub allocation_class: VulkanRuntimeStreamAllocationClass,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VulkanRuntimeSharedHostTransientAllocation {
    pub mode: VulkanRuntimeSharedHostTransientAllocationMode,
    pub owner_device_id: String,
    pub participant_device_ids: Vec<String>,
    pub byte_capacity: usize,
    pub concern: String,
    pub allocation_class: VulkanRuntimeStreamAllocationClass,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VulkanRuntimeSharedHostTransientAllocationMode {
    Always,
    CrossDeviceTimelineStaging,
    ConditionalPredicate,
}

impl VulkanRuntimeHybridExecutionTransientPlan {
    fn into_allocation_class(mut self, allocation_class: VulkanRuntimeStreamAllocationClass) -> Self {
        for allocation in &mut self.device_allocations {
            allocation.allocation_class = allocation_class;
        }
        for allocation in &mut self.host_visible_allocations {
            allocation.allocation_class = allocation_class;
        }
        for allocation in &mut self.shared_host_allocations {
            allocation.allocation_class = allocation_class;
        }
        self
    }

    fn extend(
        &mut self,
        other: Self,
    ) -> Result<(), VulkanRuntimeHybridPlacementError> {
        let mut next = self.clone();
        for allocation in other.device_allocations {
            next.add_device_allocation_with_usage_and_class(
                &allocation.logical_device_id,
                allocation.participant_device_ids,
                allocation.byte_capacity,
                &allocation.concern,
                allocation.usage,
                allocation.allocation_class,
            )?;
        }
        for allocation in other.host_visible_allocations {
            next.add_host_visible_allocation_with_class(
                &allocation.logical_device_id,
                allocation.byte_capacity,
                &allocation.concern,
                allocation.allocation_class,
            )?;
        }
        for allocation in other.shared_host_allocations {
            next.add_shared_host_allocation_with_class(
                allocation.mode,
                &allocation.owner_device_id,
                allocation.participant_device_ids,
                allocation.byte_capacity,
                &allocation.concern,
                allocation.allocation_class,
            )?;
        }
        *self = next;
        Ok(())
    }

    fn add_device_allocation(
        &mut self,
        logical_device_id: &str,
        byte_count: usize,
        concern: &str,
    ) -> Result<(), VulkanRuntimeHybridPlacementError> {
        self.add_device_allocation_with_usage_and_class(
            logical_device_id,
            [logical_device_id.to_string()],
            byte_count,
            concern,
            VulkanRuntimeDeviceLocalTransientAllocationUsage::Storage,
            VulkanRuntimeStreamAllocationClass::Permanent,
        )
    }

    fn add_conditional_device_allocation(
        &mut self,
        logical_device_id: &str,
        byte_count: usize,
        concern: &str,
    ) -> Result<(), VulkanRuntimeHybridPlacementError> {
        self.add_device_allocation_with_usage_and_class(
            logical_device_id,
            [logical_device_id.to_string()],
            byte_count,
            concern,
            VulkanRuntimeDeviceLocalTransientAllocationUsage::ConditionalPredicate,
            VulkanRuntimeStreamAllocationClass::Permanent,
        )
    }

    fn add_external_shared_device_allocation(
        &mut self,
        logical_device_id: &str,
        participant_device_ids: impl IntoIterator<Item = String>,
        byte_count: usize,
        concern: &str,
    ) -> Result<(), VulkanRuntimeHybridPlacementError> {
        self.add_device_allocation_with_usage_and_class(
            logical_device_id,
            participant_device_ids,
            byte_count,
            concern,
            VulkanRuntimeDeviceLocalTransientAllocationUsage::ExternalSharedStorage,
            VulkanRuntimeStreamAllocationClass::Permanent,
        )
    }

    fn add_device_allocation_with_usage_and_class(
        &mut self,
        logical_device_id: &str,
        participant_device_ids: impl IntoIterator<Item = String>,
        byte_count: usize,
        concern: &str,
        usage: VulkanRuntimeDeviceLocalTransientAllocationUsage,
        allocation_class: VulkanRuntimeStreamAllocationClass,
    ) -> Result<(), VulkanRuntimeHybridPlacementError> {
        let participant_device_ids = participant_device_ids
            .into_iter()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let participant_identity_is_valid = !participant_device_ids.is_empty()
            && participant_device_ids
                .iter()
                .all(|device_id| !device_id.trim().is_empty())
            && participant_device_ids
                .iter()
                .any(|device_id| device_id == logical_device_id)
            && match usage {
                VulkanRuntimeDeviceLocalTransientAllocationUsage::ExternalSharedStorage => {
                    true
                }
                VulkanRuntimeDeviceLocalTransientAllocationUsage::Storage
                | VulkanRuntimeDeviceLocalTransientAllocationUsage::ConditionalPredicate => {
                    participant_device_ids.len() == 1
                        && participant_device_ids[0] == logical_device_id
                }
            };
        if logical_device_id.trim().is_empty()
            || byte_count == 0
            || concern.trim().is_empty()
            || !participant_identity_is_valid
        {
            return Err(VulkanRuntimeHybridPlacementError(
                "exact hybrid device transient allocation requires a device, positive capacity, concern, and a canonical participant identity matching its usage"
                    .to_string(),
            ));
        }
        let total = self
            .device_bytes_by_logical_device
            .entry(logical_device_id.to_string())
            .or_default();
        *total = total.checked_add(byte_count).ok_or_else(|| {
            VulkanRuntimeHybridPlacementError(format!(
                "exact hybrid {concern} transient bytes overflowed on {logical_device_id:?}",
            ))
        })?;
        self.device_allocations
            .push(VulkanRuntimeDeviceLocalTransientAllocation {
                logical_device_id: logical_device_id.to_string(),
                participant_device_ids,
                byte_capacity: byte_count,
                concern: concern.to_string(),
                usage,
                allocation_class,
            });
        Ok(())
    }

    fn host_bytes(&self) -> usize {
        self.shared_host_allocations
            .iter()
            .map(|allocation| allocation.byte_capacity)
            .sum()
    }

    fn add_host_visible_allocation(
        &mut self,
        logical_device_id: &str,
        byte_count: usize,
        concern: &str,
    ) -> Result<(), VulkanRuntimeHybridPlacementError> {
        self.add_host_visible_allocation_with_class(
            logical_device_id,
            byte_count,
            concern,
            VulkanRuntimeStreamAllocationClass::Permanent,
        )
    }

    fn add_host_visible_allocation_with_class(
        &mut self,
        logical_device_id: &str,
        byte_count: usize,
        concern: &str,
        allocation_class: VulkanRuntimeStreamAllocationClass,
    ) -> Result<(), VulkanRuntimeHybridPlacementError> {
        if logical_device_id.trim().is_empty() || byte_count == 0 || concern.trim().is_empty() {
            return Err(VulkanRuntimeHybridPlacementError(
                "exact hybrid host-visible transient allocation requires a device, positive capacity, and concern"
                    .to_string(),
            ));
        }
        self.host_visible_allocations.push(
            VulkanRuntimeHostVisibleTransientAllocation {
                logical_device_id: logical_device_id.to_string(),
                byte_capacity: byte_count,
                concern: concern.to_string(),
                allocation_class,
            },
        );
        Ok(())
    }

    fn add_shared_host_allocation(
        &mut self,
        mode: VulkanRuntimeSharedHostTransientAllocationMode,
        owner_device_id: &str,
        participant_device_ids: impl IntoIterator<Item = String>,
        byte_count: usize,
        concern: &str,
    ) -> Result<(), VulkanRuntimeHybridPlacementError> {
        self.add_shared_host_allocation_with_class(
            mode,
            owner_device_id,
            participant_device_ids,
            byte_count,
            concern,
            VulkanRuntimeStreamAllocationClass::Permanent,
        )
    }

    fn add_shared_host_allocation_with_class(
        &mut self,
        mode: VulkanRuntimeSharedHostTransientAllocationMode,
        owner_device_id: &str,
        participant_device_ids: impl IntoIterator<Item = String>,
        byte_count: usize,
        concern: &str,
        allocation_class: VulkanRuntimeStreamAllocationClass,
    ) -> Result<(), VulkanRuntimeHybridPlacementError> {
        let participant_device_ids = participant_device_ids
            .into_iter()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        if owner_device_id.trim().is_empty()
            || byte_count == 0
            || !participant_device_ids
                .iter()
                .any(|device_id| device_id == owner_device_id)
        {
            return Err(VulkanRuntimeHybridPlacementError(format!(
                "exact hybrid {concern} shared-host allocation requires a nonempty owner, positive capacity, and an owner participant",
            )));
        }
        self.host_bytes().checked_add(byte_count).ok_or_else(|| {
            VulkanRuntimeHybridPlacementError(format!(
                "exact hybrid {concern} host transient bytes overflowed",
            ))
        })?;
        self.shared_host_allocations
            .push(VulkanRuntimeSharedHostTransientAllocation {
                mode,
                owner_device_id: owner_device_id.to_string(),
                participant_device_ids,
                byte_capacity: byte_count,
                concern: concern.to_string(),
                allocation_class,
            });
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn exact_vulkan_runtime_dynamic_resource_stream_fork_transient_plan(
    runtime_model: &VulkanResidentRuntimeModel,
    resource_contract: &CompiledResourceResidencyContract,
    resource_layout: &VulkanCompiledResourceAddressLayout,
    logical_device_ids: &[String],
    input_device_id: &str,
    output_device_id: &str,
    mount_speculative_decoders: bool,
    execution_ownership_plan: &VulkanDistributedSelectedResourceStorePlan,
) -> Result<VulkanRuntimeHybridExecutionTransientPlan, VulkanResidentTokenModelPackageError> {
    let mut transient = VulkanRuntimeHybridExecutionTransientPlan::default();
    for logical_device_id in logical_device_ids {
        let logical_device_set = BTreeSet::from([logical_device_id.clone()]);
        let Some(execution_ownership) =
            compiled_resource_selector_ownership_for_device_set(
                runtime_model,
                resource_contract,
                input_device_id,
                output_device_id,
                &logical_device_set,
                mount_speculative_decoders,
                execution_ownership_plan,
            )
            .map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to plan stream-owned dynamic resources on {logical_device_id:?}: {error}",
                ))
            })?
        else {
            continue;
        };
        let component_ids = compiled_resource_component_ids_for_selector_ownership(
            resource_contract,
            &execution_ownership,
        )
        .map_err(|error| {
            VulkanResidentTokenModelPackageError::new(format!(
                "failed to resolve stream-owned dynamic-resource components on {logical_device_id:?}: {error}",
            ))
        })?;
        for table in resource_layout.parameter_slot_tables.iter().filter(|table| {
            table.execution_scope == runtime_model.execution_scope
                && component_ids.contains(&table.key.component_id)
        }) {
            let byte_capacity = table
                .slot_count()
                .and_then(|slot_count| slot_count.checked_mul(size_of::<u32>()))
                .filter(|byte_capacity| *byte_capacity > 0)
                .ok_or_else(|| {
                    VulkanResidentTokenModelPackageError::new(format!(
                        "stream-owned dynamic-resource table {}.{} has an invalid byte capacity",
                        table.key.component_id, table.key.node_id,
                    ))
                })?;
            transient
                .add_device_allocation(
                    logical_device_id,
                    byte_capacity,
                    &format!(
                        "stream dynamic parameter slots {}.{}:{}",
                        table.key.component_id,
                        table.key.node_id,
                        table.key.selection_signal,
                    ),
                )
                .map_err(|error| {
                    VulkanResidentTokenModelPackageError::new(error.to_string())
                })?;
        }
    }
    Ok(transient)
}

#[allow(clippy::too_many_arguments)]
fn exact_vulkan_runtime_hybrid_prefill_runners_transient_plan(
    runtime_model: &VulkanResidentRuntimeModel,
    component_ids: &BTreeSet<String>,
    slice_plans: &[VulkanResidentModelPackageDeviceSlicePlan],
    execution_plan: &VulkanDistributedExecutionPlan,
    normal_active_width: usize,
    resource_contract: &CompiledResourceResidencyContract,
    resource_layout: &VulkanCompiledResourceAddressLayout,
    residency_policy: ResourceResidencyPolicy,
    speculative_draft_tokens: usize,
) -> Result<VulkanRuntimeHybridExecutionTransientPlan, VulkanRuntimeHybridPlacementError> {
    let retain_speculative_source_taps = speculative_draft_tokens > 0;
    let normal_lane_capacity = causal_component_block_lane_capacity(normal_active_width)
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    let mut transient = exact_vulkan_runtime_hybrid_prefill_transient_plan(
        runtime_model,
        component_ids,
        slice_plans,
        execution_plan,
        normal_active_width,
        normal_lane_capacity,
        resource_contract,
        resource_layout,
        residency_policy,
        retain_speculative_source_taps,
        false,
    )?
    .into_allocation_class(VulkanRuntimeStreamAllocationClass::PromptRunner);
    if retain_speculative_source_taps && !runtime_model.package.speculative_decoders.is_empty() {
        let verification_width = causal_component_block_lane_capacity(
            speculative_draft_tokens.checked_add(1).ok_or_else(|| {
                VulkanRuntimeHybridPlacementError(
                    "exact hybrid speculative verification width overflowed".to_string(),
                )
            })?,
        )
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
        transient.extend(
            exact_vulkan_runtime_hybrid_prefill_transient_plan(
                runtime_model,
                component_ids,
                slice_plans,
                execution_plan,
                verification_width,
                verification_width,
                resource_contract,
                resource_layout,
                residency_policy,
                true,
                true,
            )?
            .into_allocation_class(VulkanRuntimeStreamAllocationClass::VerificationRunner),
        )?;
    }
    Ok(transient)
}

fn exact_vulkan_runtime_mounted_prefill_transient_plan(
    runtime_model: &VulkanResidentRuntimeModel,
    slice_plans: &[VulkanResidentModelPackageDeviceSlicePlan],
    execution_plan: &VulkanDistributedExecutionPlan,
    normal_lane_capacity: usize,
    resource_contract: &CompiledResourceResidencyContract,
    residency_policy: ResourceResidencyPolicy,
    speculative_draft_tokens: usize,
) -> Result<VulkanRuntimeHybridExecutionTransientPlan, VulkanResidentTokenModelPackageError> {
    let component_ids = runtime_model
        .circuit_graph
        .components
        .iter()
        .filter(|component| component.runtime_role.is_signal_processor())
        .map(|component| component.component_id.clone())
        .collect::<BTreeSet<_>>();
    let resource_layout = VulkanCompiledResourceAddressLayout::from_contract(resource_contract)
        .map_err(|error| {
            VulkanResidentTokenModelPackageError::new(format!(
                "failed to plan mounted execution-transient resource layout: {error}",
            ))
        })?;
    exact_vulkan_runtime_hybrid_prefill_runners_transient_plan(
        runtime_model,
        &component_ids,
        slice_plans,
        execution_plan,
        normal_lane_capacity,
        resource_contract,
        &resource_layout,
        residency_policy,
        speculative_draft_tokens,
    )
    .map_err(|error| {
        VulkanResidentTokenModelPackageError::new(format!(
            "failed to plan mounted prefill execution transients: {error}",
        ))
    })
}

fn exact_vulkan_runtime_decode_scalar_queue_groups(
    slice_plans: &[VulkanResidentModelPackageDeviceSlicePlan],
    execution_plan: &VulkanDistributedExecutionPlan,
) -> Result<BTreeMap<String, String>, VulkanRuntimeHybridPlacementError> {
    let mut segments_by_selector = BTreeMap::<String, BTreeSet<(String, usize)>>::new();
    for slice in slice_plans {
        let physical_execution_islands = execution_plan
            .execution_islands
            .iter()
            .filter(|island| island.owner_device_id == slice.device_id)
            .map(|island| island.dispatch_indices())
            .collect::<Vec<_>>();
        let distributed_owned_checkpoint_ids =
            distributed_owned_physical_residency_checkpoint_ids(
                &slice.physical_residency_schedule,
                &physical_execution_islands,
            )
            .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
        let distributed_dispatch_indices = physical_execution_islands
            .iter()
            .flatten()
            .copied()
            .collect::<BTreeSet<_>>();
        let resident = &slice.placed_plan.placed_resident_plan;
        let mut segment_open = false;
        let mut segment_index = 0usize;
        let mut segment_by_dispatch = BTreeMap::new();
        for dispatch in &slice.prepared_plan.dispatches {
            let receives_remote_input = dispatch.descriptors.iter().any(|descriptor| {
                let VulkanDescriptorResourceAddress::BoundaryInput { signal_id } =
                    &descriptor.resource
                else {
                    return false;
                };
                matches!(
                    classify_boundary_input(&dispatch.component_id, signal_id, resident),
                    VulkanPlacedBoundDescriptorTarget::IncomingEdge { .. }
                )
            });
            if receives_remote_input
                || distributed_dispatch_indices.contains(&dispatch.dispatch_index)
            {
                segment_open = false;
            }
            if distributed_dispatch_indices.contains(&dispatch.dispatch_index) {
                continue;
            }
            if !segment_open {
                segment_index = segment_index.checked_add(1).ok_or_else(|| {
                    VulkanRuntimeHybridPlacementError(
                        "decode demand-segment count overflowed".to_string(),
                    )
                })?;
                segment_open = true;
            }
            segment_by_dispatch.insert(dispatch.dispatch_index, segment_index);
            let publishes_remote_output = dispatch.descriptors.iter().any(|descriptor| {
                let VulkanDescriptorResourceAddress::BoundaryOutput { signal_id } =
                    &descriptor.resource
                else {
                    return false;
                };
                matches!(
                    classify_boundary_output(&dispatch.component_id, signal_id, resident),
                    VulkanPlacedBoundDescriptorTarget::ProducedPort {
                        outgoing_edges,
                        ..
                    } if !outgoing_edges.is_empty()
                )
            });
            if publishes_remote_output {
                segment_open = false;
            }
        }
        for checkpoint in slice
            .physical_residency_schedule
            .checkpoints
            .iter()
            .filter(|checkpoint| !distributed_owned_checkpoint_ids.contains(&checkpoint.id))
        {
            let selection_segment = segment_by_dispatch
                .get(&checkpoint.selection_dispatch_index)
                .copied()
                .ok_or_else(|| {
                    VulkanRuntimeHybridPlacementError(format!(
                        "scalar residency checkpoint {:?} selection dispatch {} has no local decode segment on {:?}",
                        checkpoint.id, checkpoint.selection_dispatch_index, slice.device_id,
                    ))
                })?;
            let checkpoint_dispatches = checkpoint
                .selected_computation_dispatch_indices
                .iter()
                .copied()
                .chain(checkpoint.selected_result_continuation_dispatch_index);
            for dispatch_index in checkpoint_dispatches {
                if segment_by_dispatch.get(&dispatch_index).copied() != Some(selection_segment) {
                    return runtime_hybrid_error(format!(
                        "scalar residency checkpoint {:?} crosses a local decode segment boundary on {:?}",
                        checkpoint.id, slice.device_id,
                    ));
                }
            }
            for selector_id in &checkpoint.selector_ids {
                segments_by_selector
                    .entry(selector_id.clone())
                    .or_default()
                    .insert((slice.device_id.clone(), selection_segment));
            }
        }
    }
    exact_vulkan_runtime_decode_scalar_queue_groups_from_segments(
        &segments_by_selector,
    )
}

fn exact_vulkan_runtime_decode_scalar_queue_groups_from_segments(
    segments_by_selector: &BTreeMap<String, BTreeSet<(String, usize)>>,
) -> Result<BTreeMap<String, String>, VulkanRuntimeHybridPlacementError> {
    segments_by_selector
        .iter()
        .map(|(selector_id, segments)| {
            let [(device_id, segment_index)] = segments.iter().collect::<Vec<_>>().as_slice()
            else {
                return runtime_hybrid_error(format!(
                    "scalar demand selector {selector_id:?} maps to {} local decode segments",
                    segments.len(),
                ));
            };
            Ok((
                selector_id.clone(),
                format!("{device_id}:decode-segment:{segment_index}"),
            ))
        })
        .collect()
}

fn exact_vulkan_runtime_mounted_decode_transient_plan(
    runtime_model: &VulkanResidentRuntimeModel,
    slice_plans: &[VulkanResidentModelPackageDeviceSlicePlan],
    execution_plan: &VulkanDistributedExecutionPlan,
    resource_contract: &CompiledResourceResidencyContract,
    residency_policy: ResourceResidencyPolicy,
    devices: &[VulkanRuntimeSelectedResourceMountDevice],
    feedback_lane_capacity: usize,
) -> Result<VulkanRuntimeHybridExecutionTransientPlan, VulkanResidentTokenModelPackageError> {
    let component_ids = runtime_model
        .circuit_graph
        .components
        .iter()
        .filter(|component| component.runtime_role.is_signal_processor())
        .map(|component| component.component_id.clone())
        .collect::<BTreeSet<_>>();
    let component_owner_logical_device_ids = slice_plans
        .iter()
        .flat_map(|slice| {
            slice
                .placed_plan
                .placed_resident_plan
                .hosted_component_ids
                .iter()
                .filter(|component_id| component_ids.contains(*component_id))
                .map(|component_id| (component_id.clone(), slice.device_id.clone()))
        })
        .collect::<BTreeMap<_, _>>();
    let resource_layout = VulkanCompiledResourceAddressLayout::from_contract(resource_contract)
        .map_err(|error| {
            VulkanResidentTokenModelPackageError::new(format!(
                "failed to plan mounted decode resource layout: {error}",
            ))
        })?;
    let logical_device_ids = devices
        .iter()
        .map(|device| device.logical_device_id.clone())
        .collect::<Vec<_>>();
    let physical_device_by_logical_device = devices
        .iter()
        .map(|device| {
            (
                device.logical_device_id.clone(),
                device.physical_device_id.clone(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let scalar_queue_groups = exact_vulkan_runtime_decode_scalar_queue_groups(
        slice_plans,
        execution_plan,
    )
    .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;
    exact_vulkan_runtime_hybrid_gate_device_plan(
        &component_ids,
        &component_owner_logical_device_ids,
        execution_plan,
        1,
        resource_contract,
        &resource_layout,
        residency_policy,
        feedback_lane_capacity,
        Some(&scalar_queue_groups),
        Some((
            logical_device_ids.as_slice(),
            &physical_device_by_logical_device,
        )),
    )
    .map_err(|error| {
        VulkanResidentTokenModelPackageError::new(format!(
            "failed to plan mounted decode execution transients: {error}",
        ))
    })
}

fn exact_vulkan_runtime_speculative_catch_up_transient_plan(
    runtime_model: &VulkanResidentRuntimeModel,
    decoder_slice_plans: &BTreeMap<String, VulkanResidentModelPackageDeviceSlicePlan>,
    normal_prefill_lane_capacity: usize,
    speculative_draft_tokens: usize,
    residency_policy: ResourceResidencyPolicy,
) -> Result<VulkanRuntimeHybridExecutionTransientPlan, VulkanResidentTokenModelPackageError> {
    let mut transient = VulkanRuntimeHybridExecutionTransientPlan::default();
    if speculative_draft_tokens == 0 || residency_policy.is_demand_loaded() {
        return Ok(transient);
    }
    let lane_capacity = speculative_catch_up_execution_lane_capacity(
        normal_prefill_lane_capacity,
        speculative_draft_tokens,
    )
    .map_err(|error| {
        VulkanResidentTokenModelPackageError::new(format!(
            "failed to plan speculative catch-up width: {error}",
        ))
    })?;
    let (batch_control_binding, batch_control_byte_count, batch_control_payload) = runtime_model
        .package
        .input_transducer
        .batch_control
        .storage_buffer();
    if batch_control_payload != VulkanResidentComponentBatchControlPayload::Width
        || batch_control_byte_count != VULKAN_COMPONENT_BATCH_WIDTH_CONTROL_BYTE_CAPACITY
    {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "speculative catch-up input batch control binding {batch_control_binding} carries {batch_control_payload:?} in {batch_control_byte_count} bytes",
        )));
    }
    for decoder in runtime_model.package.speculative_decoders.iter().filter(|decoder| {
        decoder
            .execution_contract
            .uses_dedicated_autoregressive_io()
    }) {
        let slice = decoder_slice_plans.get(&decoder.id).ok_or_else(|| {
            VulkanResidentTokenModelPackageError::new(format!(
                "speculative decoder {:?} has no prepared catch-up slice",
                decoder.id,
            ))
        })?;
        let allocation_plan = VulkanComponentBatchResidentAllocationPlan::for_single_device(
            &slice.placed_plan,
            &slice.prepared_plan,
            &slice.batch_kernels,
            lane_capacity,
            VulkanComponentBatchExecutionMode::CausalSequence,
            &VulkanComponentBatchExecutionScope::all(),
            &BTreeSet::new(),
            false,
            None,
        )
        .map_err(|error| {
            VulkanResidentTokenModelPackageError::new(format!(
                "failed to plan speculative decoder {:?} catch-up runner: {error}",
                decoder.id,
            ))
        })?;
        for allocation in allocation_plan.allocations {
            let concern = format!(
                "speculative catch-up {} component batch {:?}",
                decoder.id, allocation.kind,
            );
            let result = if allocation.host_visible {
                transient.add_host_visible_allocation(
                    &slice.device_id,
                    allocation.byte_capacity,
                    &concern,
                )
            } else {
                transient.add_device_allocation(
                    &slice.device_id,
                    allocation.byte_capacity,
                    &concern,
                )
            };
            result
                .map_err(|error| {
                    VulkanResidentTokenModelPackageError::new(error.to_string())
                })?;
        }
        for (byte_capacity, concern) in [
            (
                lane_capacity
                    .checked_mul(size_of::<u32>())
                    .ok_or_else(|| {
                        VulkanResidentTokenModelPackageError::new(
                            "speculative catch-up embedding token capacity overflowed",
                        )
                    })?,
                "embedding token IDs",
            ),
            (batch_control_byte_count as usize, "embedding batch control"),
        ] {
            transient
                .add_host_visible_allocation(
                    &slice.device_id,
                    byte_capacity,
                    &format!("speculative catch-up {} {concern}", decoder.id),
                )
                .map_err(|error| {
                    VulkanResidentTokenModelPackageError::new(error.to_string())
                })?;
        }
    }
    Ok(transient.into_allocation_class(VulkanRuntimeStreamAllocationClass::CatchUpRunner))
}

fn exact_vulkan_runtime_speculative_source_tap_device_id<'a>(
    runtime_model: &'a VulkanResidentRuntimeModel,
    tap: &StreamCircuitGraphSourceTap,
) -> Result<&'a str, VulkanResidentTokenModelPackageError> {
    match tap.instance_selection {
        StreamCircuitGraphSourceTapInstanceSelection::LastInExecutionOrder => runtime_model
            .circuit_graph
            .components
            .iter()
            .enumerate()
            .filter_map(|(execution_index, component)| {
                runtime_model
                    .runtime_graph
                    .instances
                    .iter()
                    .find(|instance| instance.instance_id == component.component_id)
                    .filter(|instance| instance.source_component_id == tap.component_id)
                    .map(|instance| (execution_index, instance.device_id.as_str()))
            })
            .max_by_key(|(execution_index, _)| *execution_index)
            .map(|(_, device_id)| device_id)
            .ok_or_else(|| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "speculative source tap references absent target component {:?}",
                    tap.component_id,
                ))
            }),
    }
}

#[allow(clippy::too_many_arguments)]
fn exact_vulkan_runtime_add_speculative_source_tap_staging(
    transient: &mut VulkanRuntimeHybridExecutionTransientPlan,
    runtime_model: &VulkanResidentRuntimeModel,
    decoder: &VulkanResidentSpeculativeDecoderPackageSpec,
    slice: &VulkanResidentModelPackageDeviceSlicePlan,
    physical_device_by_logical_device: &BTreeMap<&str, &str>,
    selected_destination_signal_ids: Option<&BTreeSet<String>>,
    lane_capacity: usize,
    concern: &str,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    if lane_capacity == 0 {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "speculative decoder {:?} {concern} source-tap lane capacity is zero",
            decoder.id,
        )));
    }
    let boundary_plan = VulkanModelBoundaryBufferPlan::from_placed_plan(&slice.placed_plan)
        .map_err(|error| {
            VulkanResidentTokenModelPackageError::new(format!(
                "failed to plan speculative decoder {:?} {concern} boundary: {error}",
                decoder.id,
            ))
        })?;
    let destination_physical_device = physical_device_by_logical_device
        .get(slice.device_id.as_str())
        .copied()
        .ok_or_else(|| {
            VulkanResidentTokenModelPackageError::new(format!(
                "speculative decoder {:?} device {:?} has no physical identity",
                decoder.id, slice.device_id,
            ))
        })?;
    for port in decoder
        .circuit_graph
        .boundary
        .external_inputs
        .iter()
        .filter(|port| port.source_tap.is_some())
        .filter(|port| {
            selected_destination_signal_ids
                .is_none_or(|selected| selected.contains(&port.endpoint.port_id))
        })
    {
        let tap = port.source_tap.as_ref().expect("filtered source tap");
        let source_device_id =
            exact_vulkan_runtime_speculative_source_tap_device_id(runtime_model, tap)?;
        let source_physical_device = physical_device_by_logical_device
            .get(source_device_id)
            .copied()
            .ok_or_else(|| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "speculative decoder {:?} source-tap device {source_device_id:?} has no physical identity",
                    decoder.id,
                ))
            })?;
        if source_physical_device == destination_physical_device {
            continue;
        }
        let frame_byte_capacity = boundary_plan
            .inputs
            .iter()
            .find(|input| input.signal_id == port.id)
            .and_then(|input| input.byte_capacity)
            .ok_or_else(|| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "speculative decoder {:?} source-tap input {:?} has no fixed boundary capacity among {:?}",
                    decoder.id,
                    port.id,
                    boundary_plan
                        .inputs
                        .iter()
                        .map(|input| (&input.signal_id, input.byte_capacity))
                        .collect::<Vec<_>>(),
                ))
            })?;
        let byte_capacity = frame_byte_capacity.checked_mul(lane_capacity).ok_or_else(|| {
            VulkanResidentTokenModelPackageError::new(format!(
                "speculative decoder {:?} {concern} source-tap capacity overflowed",
                decoder.id,
            ))
        })?;
        for (logical_device_id, staging_side) in [
            (source_device_id, "source staging"),
            (slice.device_id.as_str(), "destination staging"),
        ] {
            transient
                .add_host_visible_allocation(
                    logical_device_id,
                    byte_capacity,
                    &format!(
                        "speculative decoder {} {concern} source tap {} {staging_side}",
                        decoder.id, port.id,
                    ),
                )
                .map_err(|error| {
                    VulkanResidentTokenModelPackageError::new(error.to_string())
                })?;
        }
    }
    Ok(())
}

fn exact_vulkan_runtime_add_component_batch_allocations(
    transient: &mut VulkanRuntimeHybridExecutionTransientPlan,
    logical_device_id: &str,
    decoder_id: &str,
    concern: &str,
    allocation_plan: VulkanComponentBatchResidentAllocationPlan,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    for allocation in allocation_plan.allocations {
        let concern = format!(
            "speculative decoder {decoder_id} {concern} {:?}",
            allocation.kind,
        );
        let result = if allocation.host_visible {
            transient.add_host_visible_allocation(
                logical_device_id,
                allocation.byte_capacity,
                &concern,
            )
        } else if allocation.kind
            == VulkanComponentBatchResidentAllocationKind::DemandPipelinePredicate
        {
            transient.add_conditional_device_allocation(
                logical_device_id,
                allocation.byte_capacity,
                &concern,
            )
        } else {
            transient.add_device_allocation(
                logical_device_id,
                allocation.byte_capacity,
                &concern,
            )
        };
        result.map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;
    }
    Ok(())
}

fn exact_vulkan_runtime_parallel_speculative_processor_transient_plan(
    runtime_model: &VulkanResidentRuntimeModel,
    decoder_slice_plans: &BTreeMap<String, VulkanResidentModelPackageDeviceSlicePlan>,
    devices: &[VulkanRuntimeSelectedResourceMountDevice],
    speculative_draft_tokens: usize,
    residency_policy: ResourceResidencyPolicy,
) -> Result<VulkanRuntimeHybridExecutionTransientPlan, VulkanResidentTokenModelPackageError> {
    let mut transient = VulkanRuntimeHybridExecutionTransientPlan::default();
    if speculative_draft_tokens == 0 {
        return Ok(transient);
    }
    let physical_device_by_logical_device = devices
        .iter()
        .map(|device| {
            (
                device.logical_device_id.as_str(),
                device.physical_device_id.as_str(),
            )
        })
        .collect::<BTreeMap<_, _>>();

    for decoder in runtime_model.package.speculative_decoders.iter().filter(|decoder| {
        matches!(
            decoder.execution_contract,
            VulkanResidentSpeculativeExecutionContract::ParallelBlock { .. }
        )
    }) {
        let VulkanResidentSpeculativeExecutionContract::ParallelBlock { block_width, .. } =
            decoder.execution_contract
        else {
            unreachable!("filtered parallel speculative decoder")
        };
        let slice = decoder_slice_plans.get(&decoder.id).ok_or_else(|| {
            VulkanResidentTokenModelPackageError::new(format!(
                "parallel speculative decoder {:?} has no prepared slice",
                decoder.id,
            ))
        })?;
        let demand_contract = if residency_policy.is_demand_loaded() {
            Some(
                instantiate_runtime_resource_contract(&speculative_decoder_runtime_model(
                    runtime_model,
                    decoder,
                    &slice.device_id,
                ))
                .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?,
            )
        } else {
            None
        };
        let demand_layout = demand_contract
            .as_ref()
            .map(VulkanCompiledResourceAddressLayout::from_contract)
            .transpose()
            .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;
        let demand_context = demand_contract.as_ref().zip(demand_layout.as_ref()).map(
            |(contract, layout)| VulkanComponentBatchDemandResidencyPlanContext {
                schedule: &slice.physical_residency_schedule,
                contract,
                layout,
            },
        );
        let scopes = parallel_speculative_execution_scopes(decoder)?;
        let proposal_scope = VulkanComponentBatchExecutionScope::nodes(
            scopes.proposal_node_ids_by_component.clone(),
        )
        .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;
        let proposal = VulkanComponentBatchResidentAllocationPlan::for_single_device(
            &slice.placed_plan,
            &slice.prepared_plan,
            &slice.batch_kernels,
            block_width,
            VulkanComponentBatchExecutionMode::ParallelBlock,
            &proposal_scope,
            &BTreeSet::new(),
            false,
            demand_context,
        )
        .map_err(|error| {
            VulkanResidentTokenModelPackageError::new(format!(
                "failed to plan parallel speculative decoder {:?} proposal runner: {error}",
                decoder.id,
            ))
        })?;
        exact_vulkan_runtime_add_component_batch_allocations(
            &mut transient,
            &slice.device_id,
            &decoder.id,
            "proposal",
            proposal,
        )?;

        let committed_context_scope = VulkanComponentBatchExecutionScope::nodes(
            scopes.state_node_ids_by_component,
        )
        .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;
        let committed_context = VulkanComponentBatchResidentAllocationPlan::for_single_device(
            &slice.placed_plan,
            &slice.prepared_plan,
            &slice.batch_kernels,
            1,
            VulkanComponentBatchExecutionMode::ParallelBlock,
            &committed_context_scope,
            &BTreeSet::new(),
            false,
            demand_context,
        )
        .map_err(|error| {
            VulkanResidentTokenModelPackageError::new(format!(
                "failed to plan parallel speculative decoder {:?} committed-context runner: {error}",
                decoder.id,
            ))
        })?;
        exact_vulkan_runtime_add_component_batch_allocations(
            &mut transient,
            &slice.device_id,
            &decoder.id,
            "committed context",
            committed_context,
        )?;

        let readback_byte_capacity = block_width
            .checked_mul(size_of::<u32>() + size_of::<f32>())
            .ok_or_else(|| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "parallel speculative decoder {:?} readback capacity overflowed",
                    decoder.id,
                ))
            })?;
        transient
            .add_host_visible_allocation(
                &slice.device_id,
                readback_byte_capacity,
                &format!("speculative decoder {} output readback", decoder.id),
            )
            .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;

        exact_vulkan_runtime_add_speculative_source_tap_staging(
            &mut transient,
            runtime_model,
            decoder,
            slice,
            &physical_device_by_logical_device,
            None,
            1,
            "permanent",
        )?;
    }
    Ok(transient)
}

fn exact_vulkan_runtime_parallel_speculative_state_ingestion_transient_plan(
    runtime_model: &VulkanResidentRuntimeModel,
    decoder_slice_plans: &BTreeMap<String, VulkanResidentModelPackageDeviceSlicePlan>,
    devices: &[VulkanRuntimeSelectedResourceMountDevice],
    normal_prefill_active_width: usize,
    speculative_draft_tokens: usize,
    residency_policy: ResourceResidencyPolicy,
) -> Result<VulkanRuntimeHybridExecutionTransientPlan, VulkanResidentTokenModelPackageError> {
    let mut transient = VulkanRuntimeHybridExecutionTransientPlan::default();
    if speculative_draft_tokens == 0 {
        return Ok(transient);
    }
    let normal_lane_capacity = causal_component_block_lane_capacity(normal_prefill_active_width)
        .map_err(|error| {
            VulkanResidentTokenModelPackageError::new(format!(
                "failed to plan parallel speculative normal state ingestion: {error}",
            ))
        })?;
    let verification_lane_capacity = speculative_catch_up_lane_capacity(speculative_draft_tokens)
        .map_err(|error| {
            VulkanResidentTokenModelPackageError::new(format!(
                "failed to plan parallel speculative verification state ingestion: {error}",
            ))
        })?;
    let physical_device_by_logical_device = devices
        .iter()
        .map(|device| {
            (
                device.logical_device_id.as_str(),
                device.physical_device_id.as_str(),
            )
        })
        .collect::<BTreeMap<_, _>>();

    for decoder in runtime_model.package.speculative_decoders.iter().filter(|decoder| {
        matches!(
            decoder.execution_contract,
            VulkanResidentSpeculativeExecutionContract::ParallelBlock { .. }
        )
    }) {
        let slice = decoder_slice_plans.get(&decoder.id).ok_or_else(|| {
            VulkanResidentTokenModelPackageError::new(format!(
                "parallel speculative decoder {:?} has no prepared state-ingestion slice",
                decoder.id,
            ))
        })?;
        let demand_contract = if residency_policy.is_demand_loaded() {
            Some(
                instantiate_runtime_resource_contract(&speculative_decoder_runtime_model(
                    runtime_model,
                    decoder,
                    &slice.device_id,
                ))
                .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?,
            )
        } else {
            None
        };
        let demand_layout = demand_contract
            .as_ref()
            .map(VulkanCompiledResourceAddressLayout::from_contract)
            .transpose()
            .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;
        let demand_context = demand_contract.as_ref().zip(demand_layout.as_ref()).map(
            |(contract, layout)| VulkanComponentBatchDemandResidencyPlanContext {
                schedule: &slice.physical_residency_schedule,
                contract,
                layout,
            },
        );
        let scopes = parallel_speculative_execution_scopes(decoder)?;
        let state_input_signal_ids = scopes.state_input_signal_ids;
        let execution_scope = VulkanComponentBatchExecutionScope::nodes(
            scopes.state_ingestion_node_ids_by_component,
        )
        .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;
        for (lane_class, allocation_class, lane_capacity) in [
            (
                "normal prefill",
                VulkanRuntimeStreamAllocationClass::PromptRunner,
                normal_lane_capacity,
            ),
            (
                "causal verification",
                VulkanRuntimeStreamAllocationClass::VerificationRunner,
                verification_lane_capacity,
            ),
        ] {
            let mut lane_transient = VulkanRuntimeHybridExecutionTransientPlan::default();
            let allocation_plan = VulkanComponentBatchResidentAllocationPlan::for_single_device(
                &slice.placed_plan,
                &slice.prepared_plan,
                &slice.batch_kernels,
                lane_capacity,
                VulkanComponentBatchExecutionMode::CausalSequence,
                &execution_scope,
                &BTreeSet::new(),
                false,
                demand_context,
            )
            .map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to plan parallel speculative decoder {:?} {lane_class} state-ingestion runner: {error}",
                    decoder.id,
                ))
            })?;
            exact_vulkan_runtime_add_component_batch_allocations(
                &mut lane_transient,
                &slice.device_id,
                &decoder.id,
                &format!("{lane_class} state ingestion"),
                allocation_plan,
            )?;
            exact_vulkan_runtime_add_speculative_source_tap_staging(
                &mut lane_transient,
                runtime_model,
                decoder,
                slice,
                &physical_device_by_logical_device,
                Some(&state_input_signal_ids),
                lane_capacity,
                &format!("{lane_class} state ingestion"),
            )?;
            transient
                .extend(lane_transient.into_allocation_class(allocation_class))
                .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;
        }
    }
    Ok(transient)
}

fn vulkan_runtime_normal_prefill_lane_capacity_candidates(
    slice_plans: &[VulkanResidentModelPackageDeviceSlicePlan],
    exact_lane_capacity: Option<usize>,
) -> Result<Vec<usize>, VulkanResidentTokenModelPackageError> {
    if slice_plans.is_empty() {
        return Err(VulkanResidentTokenModelPackageError::new(
            "normal prefill lane planning requires at least one owner slice",
        ));
    }
    if let Some(exact_lane_capacity) = exact_lane_capacity {
        if exact_lane_capacity == 0 || exact_lane_capacity > VULKAN_BACKEND_LOOP_MAX_WINDOW {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "exact normal prefill lane capacity {exact_lane_capacity} exceeds resident window {VULKAN_BACKEND_LOOP_MAX_WINDOW}",
            )));
        }
        return Ok(vec![exact_lane_capacity]);
    }

    let mut maximum = VULKAN_BACKEND_LOOP_MAX_WINDOW;
    for slice in slice_plans {
        for artifact in slice.batch_kernels.iter().filter(|artifact| {
            artifact.batch_mode == VulkanResidentComponentKernelBatchMode::CausalScan
        }) {
            maximum = maximum.min(artifact.lane_tile_width);
        }
        let mut scalar_dispatches_per_lane_by_component = BTreeMap::<&str, usize>::new();
        for dispatch in &slice.prepared_plan.dispatches {
            if !slice.batch_kernels.iter().any(|artifact| {
                artifact.component_id == dispatch.component_id
                    && artifact.node_id == dispatch.node_id
            }) {
                *scalar_dispatches_per_lane_by_component
                    .entry(&dispatch.component_id)
                    .or_default() += 1;
            }
        }
        let scalar_dispatches_per_lane = scalar_dispatches_per_lane_by_component
            .values()
            .copied()
            .max()
            .unwrap_or_default();
        if scalar_dispatches_per_lane > 0 {
            const RECORDED_DISPATCH_BUDGET_PER_SUBMISSION: usize = 65_536;
            maximum = maximum.min(
                RECORDED_DISPATCH_BUDGET_PER_SUBMISSION
                    .checked_div(scalar_dispatches_per_lane)
                    .unwrap_or(1)
                    .max(1),
            );
        }
    }
    let mut lane_capacity = maximum.max(1).min(VULKAN_BACKEND_LOOP_MAX_WINDOW);
    if !lane_capacity.is_power_of_two() {
        lane_capacity = lane_capacity
            .checked_next_power_of_two()
            .and_then(|next| next.checked_div(2))
            .unwrap_or(1);
    }
    let mut candidates = Vec::new();
    loop {
        candidates.push(lane_capacity);
        if lane_capacity == 1 {
            break;
        }
        lane_capacity /= 2;
    }
    Ok(candidates)
}

fn exact_vulkan_runtime_hybrid_prefill_transient_plan(
    runtime_model: &VulkanResidentRuntimeModel,
    component_ids: &BTreeSet<String>,
    slice_plans: &[VulkanResidentModelPackageDeviceSlicePlan],
    execution_plan: &VulkanDistributedExecutionPlan,
    activation_batch_width: usize,
    allocation_lane_capacity: usize,
    resource_contract: &CompiledResourceResidencyContract,
    resource_layout: &VulkanCompiledResourceAddressLayout,
    residency_policy: ResourceResidencyPolicy,
    retain_speculative_source_taps: bool,
    causal_snapshot_storage_preclaimed: bool,
) -> Result<VulkanRuntimeHybridExecutionTransientPlan, VulkanRuntimeHybridPlacementError> {
    if activation_batch_width == 0
        || allocation_lane_capacity < activation_batch_width
        || component_ids.is_empty()
        || slice_plans.is_empty()
    {
        return runtime_hybrid_error(
            "exact hybrid prefill transient planning requires components, slices, and an allocation capacity covering the active width",
        );
    }
    let mut plan = VulkanRuntimeHybridExecutionTransientPlan::default();
    let input_component_id = runtime_model
        .circuit_graph
        .signal_processor_endpoint_component_ids()
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?
        .0;
    let input_device_id = runtime_model
        .placement
        .device_for_component(&input_component_id);
    plan.add_host_visible_allocation(
        input_device_id,
        allocation_lane_capacity
            .checked_mul(size_of::<u32>())
            .ok_or_else(|| {
                VulkanRuntimeHybridPlacementError(
                    "exact hybrid input token-id capacity overflowed".to_string(),
                )
            })?,
        "input embedding token IDs",
    )?;
    plan.add_host_visible_allocation(
        input_device_id,
        VULKAN_COMPONENT_BATCH_WIDTH_CONTROL_BYTE_CAPACITY as usize,
        "input embedding batch control",
    )?;
    let distributed_dispatch_indices = execution_plan
        .dispatches
        .iter()
        .map(|dispatch| dispatch.dispatch_index)
        .collect::<BTreeSet<_>>();
    let private_activations = distributed_component_batch_private_activation_specs(execution_plan)
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    let retained_signal_keys_by_device = if retain_speculative_source_taps {
        exact_vulkan_speculative_source_tap_signal_keys_by_device(
            runtime_model,
            component_ids,
            slice_plans,
        )?
    } else {
        BTreeMap::new()
    };
    let no_retained_signal_keys = BTreeSet::new();
    for slice in slice_plans {
        let selected_dispatches = slice
            .prepared_plan
            .dispatches
            .iter()
            .filter(|dispatch| component_ids.contains(&dispatch.component_id))
            .collect::<Vec<_>>();
        if selected_dispatches.is_empty() {
            return runtime_hybrid_error(format!(
                "exact hybrid prefill owner slice {:?} has no selected component dispatches",
                slice.device_id,
            ));
        }
        let (signal_buffer_indices, signal_buffers) =
            component_batch_signal_buffer_plan_from_prepared_dispatches_retaining(
            &slice.placed_plan,
            selected_dispatches.iter().copied(),
            retained_signal_keys_by_device
                .get(&slice.device_id)
                .unwrap_or(&no_retained_signal_keys),
        )
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
        let mut shared_device_ids_by_buffer = BTreeMap::<usize, BTreeSet<String>>::new();
        for dispatch in execution_plan
            .dispatches
            .iter()
            .filter(|dispatch| dispatch.owner_device_id == slice.device_id)
        {
            for activation in
                distributed_component_batch_shared_activations(dispatch, &private_activations)
            {
                let key = distributed_component_batch_signal_key(
                    activation,
                    &signal_buffer_indices,
                )
                .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
                let buffer_index = signal_buffer_indices.get(&key).copied().ok_or_else(|| {
                    VulkanRuntimeHybridPlacementError(format!(
                        "exact hybrid distributed prefill has no signal buffer for {key:?}",
                    ))
                })?;
                shared_device_ids_by_buffer
                    .entry(buffer_index)
                    .or_default()
                    .extend(dispatch.shards.iter().map(|shard| shard.device_id.clone()));
            }
        }
        for (buffer_index, buffer) in signal_buffers.into_iter().enumerate() {
            let byte_count = buffer
                .frame_byte_capacity
                .checked_mul(allocation_lane_capacity)
                .ok_or_else(|| {
                    VulkanRuntimeHybridPlacementError(
                        "exact hybrid component-batch signal capacity overflowed".to_string(),
                    )
                })?;
            if let Some(shared_device_ids) = shared_device_ids_by_buffer.get(&buffer_index) {
                add_exact_vulkan_runtime_shared_device_allocation(
                    &mut plan,
                    execution_plan.shared_activation_route,
                    &slice.device_id,
                    shared_device_ids.iter().cloned(),
                    byte_count,
                    "shared component-batch signal",
                )?;
            } else {
                plan.add_device_allocation(
                    &slice.device_id,
                    byte_count,
                    "component-batch signal",
                )?;
            }
        }
        for _ in 0..allocation_lane_capacity {
            plan.add_host_visible_allocation(
                &slice.device_id,
                VULKAN_STREAM_CONTROL_BYTE_CAPACITY,
                "component-batch lane stream-control",
            )?;
        }
        plan.add_host_visible_allocation(
            &slice.device_id,
            size_of::<u32>()
                .checked_mul(allocation_lane_capacity)
                .ok_or_else(|| {
                    VulkanRuntimeHybridPlacementError(
                        "exact hybrid component-batch token-id capacity overflowed".to_string(),
                    )
                })?,
            "component-batch token IDs",
        )?;
        for payload in exact_vulkan_component_batch_fixed_control_payloads() {
            plan.add_host_visible_allocation(
                &slice.device_id,
                payload.byte_count() as usize,
                "component-batch control",
            )?;
        }
        let snapshot_allocations = if causal_snapshot_storage_preclaimed {
            vec![size_of::<u32>()]
        } else {
            exact_vulkan_component_batch_snapshot_allocation_bytes(
                runtime_model,
                &selected_dispatches,
                &distributed_dispatch_indices,
                activation_batch_width,
                allocation_lane_capacity,
            )?
        };
        for byte_capacity in snapshot_allocations {
            plan.add_device_allocation(
                &slice.device_id,
                byte_capacity,
                "component-batch causal snapshot",
            )?;
        }
    }

    for spec in private_activations.values() {
        for (logical_device_id, frame_byte_capacity) in &spec.frame_byte_capacities {
            plan.add_device_allocation(
                logical_device_id,
                frame_byte_capacity
                    .checked_mul(allocation_lane_capacity)
                    .ok_or_else(|| {
                        VulkanRuntimeHybridPlacementError(
                            "exact hybrid private batch activation capacity overflowed"
                                .to_string(),
                        )
                    })?,
                "private batch activation",
            )?;
        }
    }
    let activations = VulkanDistributedActivationBufferPlan::from_execution_plan(execution_plan)
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    for reduction in &activations.reduction_allocations {
        let byte_count = reduction
            .byte_capacity
            .checked_mul(allocation_lane_capacity)
            .ok_or_else(|| {
                VulkanRuntimeHybridPlacementError(
                    "exact hybrid batch reduction capacity overflowed".to_string(),
                )
            })?;
        add_exact_vulkan_runtime_shared_device_allocation(
            &mut plan,
            execution_plan.shared_activation_route,
            &reduction.owner_device_id,
            reduction.device_ids.iter().cloned(),
            byte_count,
            "shared batch reduction",
        )?;
    }
    for dispatch in &execution_plan.dispatches {
        if distributed_component_batch_kernel_path(dispatch)
            != VulkanDistributedComponentBatchKernelPath::CompiledBatchArtifact
        {
            // The physical InputColumns and tensor-parallel OutputRows paths
            // dispatch their compiler-declared artifact directly across Y
            // lanes. They allocate no per-shard batch-control buffers.
            continue;
        }
        let kernel = exact_vulkan_runtime_component_kernel(
            runtime_model,
            &dispatch.component_id,
            &dispatch.node_id,
        )?;
        let implementation = vulkan_runtime_placement_prefill_implementation(
            kernel,
            activation_batch_width,
        )
        .map_err(|_| {
            VulkanRuntimeHybridPlacementError(format!(
                "exact hybrid distributed prefill {}.{} cannot execute width {activation_batch_width}",
                dispatch.component_id, dispatch.node_id,
            ))
        })?
        .ok_or_else(|| {
            VulkanRuntimeHybridPlacementError(format!(
                "exact hybrid distributed prefill {}.{} has no batch implementation",
                dispatch.component_id, dispatch.node_id,
            ))
        })?;
        let control_payloads = implementation
            .stages
            .iter()
            .map(|stage| stage.control.storage_buffer().2)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        for shard in &dispatch.shards {
            for payload in &control_payloads {
                plan.add_host_visible_allocation(
                    &shard.device_id,
                    payload.byte_count() as usize,
                    "distributed batch control",
                )?;
            }
        }
    }
    let component_owner_logical_device_ids = slice_plans
        .iter()
        .flat_map(|slice| {
            slice
                .placed_plan
                .placed_resident_plan
                .hosted_component_ids
                .iter()
                .filter(|component_id| component_ids.contains(*component_id))
                .map(|component_id| (component_id.clone(), slice.device_id.clone()))
        })
        .collect::<BTreeMap<_, _>>();
    let scalar_queue_groups =
        exact_vulkan_runtime_decode_scalar_queue_groups(slice_plans, execution_plan)?;
    let mut staged_edges = BTreeSet::new();
    for slice in slice_plans {
        let edge_plan = VulkanPlacedEdgeIoPlan::from_placed_resident_plan(
            &slice.placed_plan.placed_resident_plan,
        )
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
        for endpoint in edge_plan
            .endpoints
            .iter()
            .filter(|endpoint| endpoint.direction == VulkanPlacedEdgeDirection::Outgoing)
        {
            if !component_ids.contains(&endpoint.local_component_id)
                || !component_ids.contains(&endpoint.remote_component_id)
                || !staged_edges.insert(endpoint.edge_index)
            {
                continue;
            }
            plan.add_shared_host_allocation(
                VulkanRuntimeSharedHostTransientAllocationMode::CrossDeviceTimelineStaging,
                &slice.device_id,
                [slice.device_id.clone(), endpoint.remote_device_id.clone()],
                exact_vulkan_component_batch_edge_frame_bytes(
                    &endpoint.connection,
                    endpoint.byte_capacity,
                )?
                .checked_mul(allocation_lane_capacity)
                .ok_or_else(|| {
                    VulkanRuntimeHybridPlacementError(
                        "exact hybrid batch edge staging capacity overflowed".to_string(),
                    )
                })?,
                "component-batch edge staging",
            )?;
        }
    }
    let gates = exact_vulkan_runtime_hybrid_gate_device_plan(
        component_ids,
        &component_owner_logical_device_ids,
        execution_plan,
        allocation_lane_capacity,
        resource_contract,
        resource_layout,
        residency_policy,
        1,
        Some(&scalar_queue_groups),
        None,
    )?;
    plan.extend(gates)?;
    if retain_speculative_source_taps && !runtime_model.package.speculative_decoders.is_empty() {
        let output_component_id = runtime_model
            .circuit_graph
            .signal_processor_endpoint_component_ids()
            .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?
            .1;
        exact_vulkan_runtime_add_batched_output_projection_allocations(
            &mut plan,
            runtime_model.placement.device_for_component(&output_component_id),
            allocation_lane_capacity,
            &runtime_model.package.output_transducer.spec,
            &runtime_model.package.sampler.spec,
        )?;
    }
    Ok(plan)
}

fn exact_vulkan_runtime_add_batched_output_projection_allocations(
    plan: &mut VulkanRuntimeHybridExecutionTransientPlan,
    logical_device_id: &str,
    lane_capacity: usize,
    output_spec: &VulkanResidentOutputTransducerSpec,
    sampler_spec: &VulkanResidentSamplerSpec,
) -> Result<(), VulkanRuntimeHybridPlacementError> {
    let output = VulkanResidentBatchedOutputProjectionAllocationPlan::from_spec(
        output_spec,
        lane_capacity,
    )
    .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    plan.add_device_allocation(
        logical_device_id,
        output.normalized_frames_device_bytes,
        "batched output projection normalized frames",
    )?;
    plan.add_device_allocation(
        logical_device_id,
        output.logits_device_bytes,
        "batched output projection logits",
    )?;

    let sampler = VulkanResidentSamplerLogitsViewAllocationPlan::from_spec(sampler_spec)
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    for _ in 0..lane_capacity {
        if let Some(byte_capacity) = sampler.scratch_device_bytes {
            plan.add_device_allocation(
                logical_device_id,
                byte_capacity,
                "batched sampler scratch",
            )?;
        }
        plan.add_host_visible_allocation(
            logical_device_id,
            sampler.stream_control_host_bytes,
            "batched sampler stream control",
        )?;
        if let Some(byte_capacity) = sampler.seen_token_device_bytes {
            plan.add_device_allocation(
                logical_device_id,
                byte_capacity,
                "batched sampler seen-token state",
            )?;
        }
        if let Some(byte_capacity) = sampler.seen_token_batch_host_bytes {
            plan.add_host_visible_allocation(
                logical_device_id,
                byte_capacity,
                "batched sampler token batch",
            )?;
        }
        plan.add_device_allocation(
            logical_device_id,
            sampler.feedback_control_device_bytes,
            "batched sampler feedback control",
        )?;
    }
    Ok(())
}

fn add_exact_vulkan_runtime_shared_device_allocation(
    plan: &mut VulkanRuntimeHybridExecutionTransientPlan,
    route: VulkanSharedResidentBufferRoute,
    owner_device_id: &str,
    participant_device_ids: impl IntoIterator<Item = String>,
    byte_count: usize,
    concern: &str,
) -> Result<(), VulkanRuntimeHybridPlacementError> {
    let participant_device_ids = participant_device_ids.into_iter().collect::<Vec<_>>();
    match route {
        VulkanSharedResidentBufferRoute::ExternalDeviceLocal => plan
            .add_external_shared_device_allocation(
                owner_device_id,
                participant_device_ids,
                byte_count,
                concern,
            ),
        VulkanSharedResidentBufferRoute::SharedHost => plan.add_shared_host_allocation(
            VulkanRuntimeSharedHostTransientAllocationMode::Always,
            owner_device_id,
            participant_device_ids,
            byte_count,
            concern,
        ),
    }
}

fn exact_vulkan_runtime_hybrid_gate_device_bytes(
    component_ids: &BTreeSet<String>,
    component_owner_logical_device_ids: &BTreeMap<String, String>,
    execution_plan: &VulkanDistributedExecutionPlan,
    lane_count: usize,
    resource_contract: &CompiledResourceResidencyContract,
    resource_layout: &VulkanCompiledResourceAddressLayout,
    residency_policy: ResourceResidencyPolicy,
) -> Result<BTreeMap<String, usize>, VulkanRuntimeHybridPlacementError> {
    exact_vulkan_runtime_hybrid_gate_device_plan(
        component_ids,
        component_owner_logical_device_ids,
        execution_plan,
        lane_count,
        resource_contract,
        resource_layout,
        residency_policy,
        1,
        None,
        None,
    )
    .map(|plan| plan.device_bytes_by_logical_device)
}

fn exact_vulkan_runtime_scalar_selector_is_planned(
    component_has_distributed_dispatch: bool,
    scalar_queue_group_by_selector: Option<&BTreeMap<String, String>>,
    selector_id: &str,
) -> bool {
    if let Some(groups) = scalar_queue_group_by_selector {
        return groups.contains_key(selector_id);
    }
    !component_has_distributed_dispatch
}

fn exact_vulkan_runtime_hybrid_gate_device_plan(
    component_ids: &BTreeSet<String>,
    component_owner_logical_device_ids: &BTreeMap<String, String>,
    execution_plan: &VulkanDistributedExecutionPlan,
    lane_count: usize,
    resource_contract: &CompiledResourceResidencyContract,
    resource_layout: &VulkanCompiledResourceAddressLayout,
    residency_policy: ResourceResidencyPolicy,
    scalar_gate_replica_count: usize,
    scalar_queue_group_by_selector: Option<&BTreeMap<String, String>>,
    decode_predicate_placement: Option<(&[String], &BTreeMap<String, String>)>,
) -> Result<VulkanRuntimeHybridExecutionTransientPlan, VulkanRuntimeHybridPlacementError> {
    if lane_count == 0 || scalar_gate_replica_count == 0 {
        return runtime_hybrid_error(
            "exact hybrid residency gate lane and scalar replica counts must be positive",
        );
    }
    if !residency_policy.is_demand_loaded() {
        return Ok(VulkanRuntimeHybridExecutionTransientPlan::default());
    }
    let mut plan = VulkanRuntimeHybridExecutionTransientPlan::default();
    let mut has_residency_gate = false;
    let mut scalar_missing_queues = BTreeMap::<String, (String, usize)>::new();
    let mut scalar_predicate_groups = BTreeSet::<(String, String)>::new();
    let distributed_components = execution_plan
        .dispatches
        .iter()
        .map(|dispatch| dispatch.component_id.clone())
        .collect::<BTreeSet<_>>();
    for component_id in component_ids {
        let component_has_distributed_dispatch =
            distributed_components.contains(component_id);
        let owner = component_owner_logical_device_ids.get(component_id).ok_or_else(|| {
            VulkanRuntimeHybridPlacementError(format!(
                "exact hybrid scalar residency gate has no owner for {component_id:?}",
            ))
        })?;
        let selector_ids = resource_contract
            .checkpoints
            .iter()
            .filter(|checkpoint| checkpoint.component_id == *component_id)
            .flat_map(|checkpoint| checkpoint.selector_ids.iter().cloned())
            .filter(|selector_id| {
                exact_vulkan_runtime_scalar_selector_is_planned(
                    component_has_distributed_dispatch,
                    scalar_queue_group_by_selector,
                    selector_id,
                )
            })
            .collect::<BTreeSet<_>>();
        if selector_ids.is_empty() {
            continue;
        }
        has_residency_gate = true;
        for selector_id in selector_ids {
            let selector = exact_vulkan_runtime_hybrid_selector(
                resource_contract,
                component_id,
                &selector_id,
            )?;
            let selection_count = selector
                .encoding
                .selection_count_per_activation
                .checked_mul(lane_count)
                .ok_or_else(|| {
                    VulkanRuntimeHybridPlacementError(
                        "exact hybrid scalar gate selection capacity overflowed".to_string(),
                    )
                })?;
            let config = VulkanGpuResidencyGateConfig {
                maximum_selection_count: selection_count,
                selection_count_per_lane: selector.encoding.selection_count_per_activation,
                selection_lane_stride_words: selector.encoding.selection_count_per_activation,
                selection_index_shift: selector.encoding.index_shift,
                selection_index_mask: selector.encoding.index_mask,
                address_mapping: exact_vulkan_runtime_hybrid_gate_address_mapping(
                    resource_layout,
                    &selector_id,
                )?,
                owned_resource_indices: None,
            };
            let private = config
                .private_device_bytes()
                .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
            for _ in 0..scalar_gate_replica_count {
                for byte_capacity in [
                    private.configuration_bytes,
                    private.resource_group_record_bytes,
                    private.resource_address_slot_bytes,
                    private.resolved_address_bytes,
                ] {
                    plan.add_device_allocation(owner, byte_capacity, "scalar residency gate")?;
                }
            }
            let queue_group = match scalar_queue_group_by_selector {
                Some(groups) => groups.get(&selector_id).cloned().ok_or_else(|| {
                    VulkanRuntimeHybridPlacementError(format!(
                        "exact hybrid scalar residency queue has no decode segment for selector {selector_id:?}",
                    ))
                })?,
                None => component_id.clone(),
            };
            match scalar_missing_queues.get_mut(&queue_group) {
                Some((group_owner, group_capacity)) => {
                    if group_owner != owner {
                        return runtime_hybrid_error(format!(
                            "scalar residency queue group {queue_group:?} spans owners {group_owner:?} and {owner:?}",
                        ));
                    }
                    *group_capacity = (*group_capacity).max(selection_count);
                }
                None => {
                    scalar_missing_queues.insert(
                        queue_group.clone(),
                        (owner.clone(), selection_count),
                    );
                }
            }
            scalar_predicate_groups.insert((owner.clone(), queue_group));
        }
    }

    if decode_predicate_placement.is_none() {
        for (owner, _) in &scalar_predicate_groups {
            plan.add_conditional_device_allocation(
                owner,
                size_of::<u32>(),
                "scalar residency predicate",
            )?;
        }
    }

    for (owner, missing_capacity) in scalar_missing_queues.values() {
        let missing_queue_byte_capacity =
            VulkanGpuResidencyMissQueue::device_bytes_for_capacity(*missing_capacity)
                .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?
                .byte_count;
        for _ in 0..scalar_gate_replica_count {
            plan.add_host_visible_allocation(
                owner,
                missing_queue_byte_capacity,
                "scalar residency miss queue",
            )?;
        }
    }

    let store_plan = VulkanDistributedSelectedResourceStorePlan::from_execution_plan(execution_plan)
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    for island in &execution_plan.execution_islands {
        let leader = island.leader();
        if !island
            .dispatches
            .iter()
            .any(|dispatch| !dispatch.selected_resource_partitions.is_empty())
        {
            continue;
        }
        has_residency_gate = true;
        for shard in &leader.shards {
            let device_plan = store_plan.device(&shard.device_id).ok_or_else(|| {
                VulkanRuntimeHybridPlacementError(format!(
                    "exact hybrid distributed gate has no store ownership on {:?}",
                    shard.device_id,
                ))
            })?;
            let mut gate_count = 0usize;
            for partition in &leader.selected_resource_partitions {
                let selector = exact_vulkan_runtime_hybrid_selector(
                    resource_contract,
                    &leader.component_id,
                    &partition.selector_id,
                )?;
                let ownership = device_plan
                    .selectors
                    .iter()
                    .find(|ownership| ownership.selector_id == partition.selector_id)
                    .ok_or_else(|| {
                        VulkanRuntimeHybridPlacementError(format!(
                            "exact hybrid distributed gate selector {:?} has no ownership on {:?}",
                            partition.selector_id, shard.device_id,
                        ))
                    })?;
                let selection_count = selector
                    .encoding
                    .selection_count_per_activation
                    .checked_mul(lane_count)
                    .ok_or_else(|| {
                        VulkanRuntimeHybridPlacementError(
                            "exact hybrid distributed gate selection capacity overflowed"
                                .to_string(),
                        )
                    })?;
                let config = VulkanGpuResidencyGateConfig {
                    maximum_selection_count: selection_count,
                    selection_count_per_lane: selector.encoding.selection_count_per_activation,
                    selection_lane_stride_words: selector.encoding.selection_count_per_activation,
                    selection_index_shift: selector.encoding.index_shift,
                    selection_index_mask: selector.encoding.index_mask,
                    address_mapping: exact_vulkan_runtime_hybrid_gate_address_mapping(
                        resource_layout,
                        &partition.selector_id,
                    )?,
                    owned_resource_indices: Some(
                        ownership.owned_resource_indices.iter().copied().collect(),
                    ),
                };
                let private = config
                    .private_device_bytes()
                    .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
                let queue = VulkanGpuResidencyMissQueue::device_bytes_for_capacity(selection_count)
                    .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?
                    .byte_count;
                for byte_capacity in [
                    private.configuration_bytes,
                    private.resource_group_record_bytes,
                    private.resource_address_slot_bytes,
                    private.resolved_address_bytes,
                ] {
                    plan.add_device_allocation(
                        &shard.device_id,
                        byte_capacity,
                        "distributed residency gate",
                    )?;
                }
                plan.add_host_visible_allocation(
                    &shard.device_id,
                    queue,
                    "distributed residency miss queue",
                )?;
                gate_count += 1;
            }
            if gate_count > 0 {
                // The runtime creates one predicate on each shard and shares
                // that predicate across every selected-resource partition on
                // the shard.
                match decode_predicate_placement {
                    Some((_, physical_device_by_logical_device)) => {
                        add_exact_vulkan_runtime_decode_shard_predicate(
                            &mut plan,
                            &island.owner_device_id,
                            &shard.device_id,
                            physical_device_by_logical_device,
                        )?;
                    }
                    None => plan.add_conditional_device_allocation(
                        &shard.device_id,
                        VULKAN_DEMAND_FEEDBACK_PREDICATE_BYTE_CAPACITY,
                        "distributed shard residency predicate",
                    )?,
                }
            }
        }
    }
    if has_residency_gate {
        if let Some((logical_device_ids, physical_device_by_logical_device)) =
            decode_predicate_placement
        {
            add_exact_vulkan_runtime_decode_pipeline_predicate(
                &mut plan,
                logical_device_ids,
                physical_device_by_logical_device,
            )?;
        }
    }
    Ok(plan)
}

fn add_exact_vulkan_runtime_decode_pipeline_predicate(
    plan: &mut VulkanRuntimeHybridExecutionTransientPlan,
    logical_device_ids: &[String],
    physical_device_by_logical_device: &BTreeMap<String, String>,
) -> Result<(), VulkanRuntimeHybridPlacementError> {
    let Some(owner_device_id) = logical_device_ids.first() else {
        return runtime_hybrid_error(
            "exact decode demand-feedback predicate has no logical device",
        );
    };
    if logical_device_ids.iter().any(|device_id| {
        !physical_device_by_logical_device.contains_key(device_id)
    }) {
        return runtime_hybrid_error(
            "exact decode demand-feedback predicate has an unbound logical device",
        );
    }
    if logical_device_ids.len() == 1 {
        plan.add_conditional_device_allocation(
            owner_device_id,
            VULKAN_DEMAND_FEEDBACK_PREDICATE_BYTE_CAPACITY,
            "decode demand-feedback pipeline predicate",
        )
    } else {
        plan.add_shared_host_allocation(
            VulkanRuntimeSharedHostTransientAllocationMode::ConditionalPredicate,
            owner_device_id,
            logical_device_ids.iter().cloned(),
            VULKAN_DEMAND_FEEDBACK_PREDICATE_BYTE_CAPACITY,
            "decode demand-feedback pipeline predicate",
        )
    }
}

fn add_exact_vulkan_runtime_decode_shard_predicate(
    plan: &mut VulkanRuntimeHybridExecutionTransientPlan,
    owner_device_id: &str,
    shard_device_id: &str,
    physical_device_by_logical_device: &BTreeMap<String, String>,
) -> Result<(), VulkanRuntimeHybridPlacementError> {
    let owner_physical_device_id = physical_device_by_logical_device
        .get(owner_device_id)
        .ok_or_else(|| {
            VulkanRuntimeHybridPlacementError(format!(
                "exact decode shard predicate owner {owner_device_id:?} is unbound",
            ))
        })?;
    let shard_physical_device_id = physical_device_by_logical_device
        .get(shard_device_id)
        .ok_or_else(|| {
            VulkanRuntimeHybridPlacementError(format!(
                "exact decode shard predicate participant {shard_device_id:?} is unbound",
            ))
        })?;
    if owner_physical_device_id == shard_physical_device_id {
        plan.add_conditional_device_allocation(
            owner_device_id,
            VULKAN_DEMAND_FEEDBACK_PREDICATE_BYTE_CAPACITY,
            "decode local shard residency predicate",
        )
    } else {
        plan.add_shared_host_allocation(
            VulkanRuntimeSharedHostTransientAllocationMode::ConditionalPredicate,
            owner_device_id,
            [owner_device_id.to_string(), shard_device_id.to_string()],
            VULKAN_DEMAND_FEEDBACK_PREDICATE_BYTE_CAPACITY,
            "decode cross-device shard residency predicate",
        )
    }
}

fn exact_vulkan_runtime_hybrid_selector<'a>(
    resource_contract: &'a CompiledResourceResidencyContract,
    component_id: &str,
    selector_id: &str,
) -> Result<&'a CompiledResourceSelector, VulkanRuntimeHybridPlacementError> {
    resource_contract
        .selectors
        .iter()
        .find(|selector| {
            selector.id == selector_id && selector.component_id == component_id
        })
        .ok_or_else(|| {
            VulkanRuntimeHybridPlacementError(format!(
                "exact hybrid residency gate has no selector {selector_id:?} for {component_id:?}",
            ))
        })
}

fn exact_vulkan_runtime_hybrid_gate_address_mapping(
    resource_layout: &VulkanCompiledResourceAddressLayout,
    selector_id: &str,
) -> Result<VulkanGpuResidencyAddressMapping, VulkanRuntimeHybridPlacementError> {
    let layout = resource_layout
        .selectors
        .iter()
        .find(|layout| layout.selector_id == selector_id)
        .ok_or_else(|| {
            VulkanRuntimeHybridPlacementError(format!(
                "exact hybrid residency gate selector {selector_id:?} has no address layout",
            ))
        })?;
    Ok(match &layout.mapping {
        VulkanCompiledSelectorAddressMapping::GroupTable {
            resource_address_slots,
            resource_address_slot_offsets,
        } => VulkanGpuResidencyAddressMapping::GroupTable {
            resource_address_slots: resource_address_slots.clone(),
            resource_address_slot_offsets: resource_address_slot_offsets.clone(),
        },
        VulkanCompiledSelectorAddressMapping::PartitionTemplate {
            member_slot_bases,
            resource_count,
            ..
        } => VulkanGpuResidencyAddressMapping::Partitioned {
            member_slot_bases: member_slot_bases.clone(),
            resource_count: *resource_count,
        },
    })
}

fn exact_vulkan_component_batch_fixed_control_payloads(
) -> [VulkanResidentComponentBatchControlPayload; 6] {
    [
        VulkanResidentComponentBatchControlPayload::Width,
        VulkanResidentComponentBatchControlPayload::WidthStateSnapshots,
        VulkanResidentComponentBatchControlPayload::WidthExpertStart,
        VulkanResidentComponentBatchControlPayload::WidthExpertRangeIndirect,
        VulkanResidentComponentBatchControlPayload::Temporal,
        VulkanResidentComponentBatchControlPayload::TemporalStateSnapshots,
    ]
}

#[cfg(test)]
fn exact_vulkan_component_batch_fixed_control_bytes(
) -> Result<usize, VulkanRuntimeHybridPlacementError> {
    exact_vulkan_component_batch_fixed_control_payloads()
        .into_iter()
        .try_fold(0usize, |total, payload| {
            total
                .checked_add(payload.byte_count() as usize)
                .ok_or_else(|| {
                    VulkanRuntimeHybridPlacementError(
                        "exact hybrid component-batch control capacity overflowed".to_string(),
                    )
                })
        })
}

fn exact_vulkan_runtime_component_kernel<'a>(
    runtime_model: &'a VulkanResidentRuntimeModel,
    component_id: &str,
    node_id: &str,
) -> Result<&'a VulkanResidentComponentKernelSpec, VulkanRuntimeHybridPlacementError> {
    runtime_model
        .component_executions
        .iter()
        .find(|execution| execution.component_id == component_id)
        .and_then(|execution| execution.kernels.iter().find(|kernel| kernel.node_id == node_id))
        .ok_or_else(|| {
            VulkanRuntimeHybridPlacementError(format!(
                "exact hybrid prefill has no kernel declaration for {component_id:?}.{node_id:?}",
            ))
        })
}

fn exact_vulkan_component_batch_snapshot_allocation_bytes(
    runtime_model: &VulkanResidentRuntimeModel,
    dispatches: &[&VulkanPreparedDispatch],
    distributed_dispatch_indices: &BTreeSet<usize>,
    activation_batch_width: usize,
    allocation_lane_capacity: usize,
) -> Result<Vec<usize>, VulkanRuntimeHybridPlacementError> {
    let mut snapshot_reader_exists = false;
    let mut requested_states = BTreeMap::<(String, String), usize>::new();
    for dispatch in dispatches {
        if distributed_dispatch_indices.contains(&dispatch.dispatch_index) {
            continue;
        }
        let kernel = exact_vulkan_runtime_component_kernel(
            runtime_model,
            &dispatch.component_id,
            &dispatch.node_id,
        )?;
        let Some(implementation) = vulkan_runtime_placement_prefill_implementation(
            kernel,
            activation_batch_width,
        )
        .map_err(|_| {
            VulkanRuntimeHybridPlacementError(format!(
                "exact hybrid local prefill {}.{} cannot execute width {activation_batch_width}",
                dispatch.component_id, dispatch.node_id,
            ))
        })?
        else {
            continue;
        };
        for stage in &implementation.stages {
            if let Some(source_binding) = stage.state_snapshot_source_binding {
                snapshot_reader_exists = true;
                let descriptor = dispatch
                    .descriptors
                    .iter()
                    .find(|descriptor| {
                        u32::try_from(descriptor.binding).ok() == Some(source_binding)
                    })
                    .ok_or_else(|| {
                        VulkanRuntimeHybridPlacementError(format!(
                            "exact hybrid snapshot reader {}.{} references absent binding {source_binding}",
                            dispatch.component_id, dispatch.node_id,
                        ))
                    })?;
                exact_vulkan_component_batch_insert_snapshot_state(
                    descriptor,
                    &mut requested_states,
                )?;
            } else if stage.state_snapshot_binding.is_some() {
                let writers = exact_vulkan_component_batch_snapshot_writer_states(dispatch)?;
                if writers.len() != 1 {
                    return runtime_hybrid_error(format!(
                        "exact hybrid snapshot writer {}.{} has {} static state targets",
                        dispatch.component_id,
                        dispatch.node_id,
                        writers.len(),
                    ));
                }
                let (state, static_bytes) = writers
                    .into_iter()
                    .next()
                    .expect("one snapshot writer was checked above");
                exact_vulkan_component_batch_insert_snapshot_state_capacity(
                    state,
                    static_bytes,
                    &mut requested_states,
                )?;
            }
        }
    }
    let mut allocations = vec![size_of::<u32>()];
    if snapshot_reader_exists {
        for static_bytes in requested_states.values() {
            allocations.push(
                static_bytes
                .checked_mul(allocation_lane_capacity)
                .ok_or_else(|| {
                    VulkanRuntimeHybridPlacementError(
                        "exact hybrid snapshot lane capacity overflowed".to_string(),
                    )
                })?,
            );
        }
    }
    Ok(allocations)
}

fn exact_vulkan_component_batch_snapshot_writer_states(
    dispatch: &VulkanPreparedDispatch,
) -> Result<BTreeMap<(String, String), usize>, VulkanRuntimeHybridPlacementError> {
    let mut writers = BTreeMap::new();
    for descriptor in &dispatch.descriptors {
        if !matches!(
            descriptor.usage,
            VulkanKernelDescriptorUsage::StateWrite | VulkanKernelDescriptorUsage::StateView
        ) || !matches!(
            descriptor.resource,
            VulkanDescriptorResourceAddress::StateBuffer {
                static_bytes: Some(bytes),
                ..
            } | VulkanDescriptorResourceAddress::StateView {
                static_bytes: Some(bytes),
                ..
            } if bytes > 0
        ) {
            continue;
        }
        exact_vulkan_component_batch_insert_snapshot_state(descriptor, &mut writers)?;
    }
    Ok(writers)
}

fn exact_vulkan_component_batch_insert_snapshot_state(
    descriptor: &VulkanResolvedDescriptorBinding,
    states: &mut BTreeMap<(String, String), usize>,
) -> Result<(), VulkanRuntimeHybridPlacementError> {
    let (component_id, state_id, static_bytes) = match &descriptor.resource {
        VulkanDescriptorResourceAddress::StateBuffer {
            component_id,
            state_id,
            static_bytes: Some(static_bytes),
            ..
        }
        | VulkanDescriptorResourceAddress::StateView {
            component_id,
            state_id,
            static_bytes: Some(static_bytes),
            ..
        } if *static_bytes > 0 => (component_id, state_id, *static_bytes),
        _ => {
            return runtime_hybrid_error(format!(
                "exact hybrid snapshot binding {} is not nonempty static state",
                descriptor.binding,
            ));
        }
    };
    exact_vulkan_component_batch_insert_snapshot_state_capacity(
        (component_id.clone(), state_id.clone()),
        static_bytes,
        states,
    )
}

fn exact_vulkan_component_batch_insert_snapshot_state_capacity(
    key: (String, String),
    static_bytes: usize,
    states: &mut BTreeMap<(String, String), usize>,
) -> Result<(), VulkanRuntimeHybridPlacementError> {
    if states
        .insert(key.clone(), static_bytes)
        .is_some_and(|existing| existing != static_bytes)
    {
        return runtime_hybrid_error(format!(
            "exact hybrid snapshot state {:?}.{:?} has conflicting capacities",
            key.0, key.1,
        ));
    }
    Ok(())
}

fn exact_vulkan_speculative_source_tap_signal_keys_by_device(
    runtime_model: &VulkanResidentRuntimeModel,
    component_ids: &BTreeSet<String>,
    slice_plans: &[VulkanResidentModelPackageDeviceSlicePlan],
) -> Result<
    BTreeMap<String, BTreeSet<VulkanComponentBatchSignalKey>>,
    VulkanRuntimeHybridPlacementError,
> {
    let runtime_instance_by_id = runtime_model
        .runtime_graph
        .instances
        .iter()
        .map(|instance| (instance.instance_id.as_str(), instance))
        .collect::<BTreeMap<_, _>>();
    let mut retained = BTreeMap::<String, BTreeSet<VulkanComponentBatchSignalKey>>::new();
    for tap in runtime_model
        .package
        .speculative_decoders
        .iter()
        .flat_map(|decoder| &decoder.circuit_graph.boundary.external_inputs)
        .filter_map(|port| port.source_tap.as_ref())
    {
        let instance_id = match &tap.instance_selection {
            StreamCircuitGraphSourceTapInstanceSelection::LastInExecutionOrder => runtime_model
                .circuit_graph
                .components
                .iter()
                .filter(|component| component_ids.contains(&component.component_id))
                .filter_map(|component| {
                    runtime_instance_by_id
                        .get(component.component_id.as_str())
                        .filter(|instance| instance.source_component_id == tap.component_id)
                        .map(|_| component.component_id.as_str())
                })
                .next_back(),
        };
        let Some(instance_id) = instance_id else {
            let complete_component_set = runtime_model
                .circuit_graph
                .components
                .iter()
                .filter(|component| component.runtime_role.is_signal_processor())
                .all(|component| component_ids.contains(&component.component_id));
            if complete_component_set {
                return runtime_hybrid_error(format!(
                    "exact hybrid speculative source tap references absent component {:?}",
                    tap.component_id,
                ));
            }
            continue;
        };
        let slice = slice_plans
            .iter()
            .find(|slice| slice.placed_plan.binding_plan.circuit(instance_id).is_some())
            .ok_or_else(|| {
                VulkanRuntimeHybridPlacementError(format!(
                    "exact hybrid speculative source tap instance {instance_id:?} has no owner slice",
                ))
            })?;
        let circuit = slice
            .placed_plan
            .binding_plan
            .circuit(instance_id)
            .expect("the source-tap owner slice was selected by this circuit");
        let port = circuit
            .output_ports
            .iter()
            .find(|port| port.id == tap.port_id)
            .ok_or_else(|| {
                VulkanRuntimeHybridPlacementError(format!(
                    "exact hybrid speculative source tap {instance_id:?} has no output port {:?}",
                    tap.port_id,
                ))
            })?;
        let source_signal = port.source.as_deref().unwrap_or(port.id.as_str());
        let dispatch = slice
            .prepared_plan
            .dispatches
            .iter()
            .filter(|dispatch| dispatch.component_id == instance_id)
            .find(|dispatch| {
                dispatch.descriptors.iter().any(|descriptor| {
                    descriptor.usage == VulkanKernelDescriptorUsage::OutputSignal
                        && descriptor.name == source_signal
                })
            })
            .ok_or_else(|| {
                VulkanRuntimeHybridPlacementError(format!(
                    "exact hybrid speculative source tap {instance_id:?}.{:?} has no output dispatch",
                    tap.port_id,
                ))
            })?;
        let descriptor = dispatch
            .descriptors
            .iter()
            .find(|descriptor| {
                descriptor.usage == VulkanKernelDescriptorUsage::OutputSignal
                    && descriptor.name == source_signal
            })
            .expect("the source-tap dispatch was selected by this descriptor");
        let edge_plan = VulkanPlacedEdgeIoPlan::from_placed_resident_plan(
            &slice.placed_plan.placed_resident_plan,
        )
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
        let boundary_plan = VulkanModelBoundaryBufferPlan::from_placed_plan(&slice.placed_plan)
            .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
        let (key, _) = component_batch_prepared_signal_target(
            &slice.placed_plan,
            &boundary_plan,
            &edge_plan,
            dispatch,
            descriptor,
        )
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?
        .ok_or_else(|| {
            VulkanRuntimeHybridPlacementError(format!(
                "exact hybrid speculative source tap {instance_id:?}.{:?} is not batch-addressable",
                tap.port_id,
            ))
        })?;
        retained.entry(slice.device_id.clone()).or_default().insert(key);
    }
    Ok(retained)
}

fn exact_vulkan_component_batch_edge_frame_bytes(
    connection: &StreamCircuitConnection,
    byte_capacity: Option<usize>,
) -> Result<usize, VulkanRuntimeHybridPlacementError> {
    let byte_capacity = byte_capacity.ok_or_else(|| {
        VulkanRuntimeHybridPlacementError(
            "exact hybrid component-batch edge has unknown capacity".to_string(),
        )
    })?;
    component_batch_edge_frame_byte_capacity(connection, byte_capacity)
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))
}
