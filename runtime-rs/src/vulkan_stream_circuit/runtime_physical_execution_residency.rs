pub const VULKAN_RUNTIME_PHYSICAL_EXECUTION_RESIDENCY_PLAN_SCHEMA: &str =
    "nerve.vulkan_runtime_physical_execution_residency_plan.v7";

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum VulkanRuntimeExternalDeviceLocalResidentAllocationKind {
    EdgeProducedPort {
        component_id: String,
        port_id: String,
        edge_indices: Vec<usize>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VulkanRuntimeExternalDeviceLocalResidentAllocation {
    pub kind: VulkanRuntimeExternalDeviceLocalResidentAllocationKind,
    pub owner_device_id: String,
    pub participant_device_ids: Vec<String>,
    pub byte_capacity: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum VulkanRuntimeSharedHostResidentAllocationKind {
    EdgeStaging {
        component_id: String,
        port_id: String,
        edge_indices: Vec<usize>,
    },
    FeedbackControl {
        scope_id: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VulkanRuntimeSharedHostResidentAllocation {
    pub kind: VulkanRuntimeSharedHostResidentAllocationKind,
    pub owner_device_id: String,
    pub participant_device_ids: Vec<String>,
    pub byte_capacity: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct VulkanRuntimePhysicalExecutionResidencyBreakdown {
    pub owner_parameter_bytes_before_distributed_replacement: usize,
    pub excluded_owner_parameter_bytes: usize,
    pub independently_admitted_resource_store_bytes: usize,
    pub owner_stream_device_bytes: usize,
    pub owner_stream_control_device_bytes_per_stream: usize,
    pub owner_edge_buffer_bytes_per_stream: usize,
    pub distributed_parameter_bytes: usize,
    pub distributed_shared_activation_device_bytes_per_stream: usize,
    pub distributed_private_activation_device_bytes_per_stream: usize,
    pub distributed_shared_host_bytes_per_stream: usize,
    pub external_edge_device_bytes_per_stream: usize,
    pub staged_edge_shared_host_bytes_per_stream: usize,
    pub feedback_control_shared_host_bytes_per_stream: usize,
    pub execution_transient_device_bytes_per_stream: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VulkanRuntimePhysicalExecutionDeviceResidencyPlan {
    pub device_id: String,
    pub breakdown: VulkanRuntimePhysicalExecutionResidencyBreakdown,
    pub mount_device_local_bytes: usize,
    pub stream_device_local_bytes: usize,
    pub stream_shared_host_bytes: usize,
    pub resident_stream_device_allocations: Vec<VulkanRuntimeResidentStreamAllocation>,
    pub external_device_local_resident_allocations:
        Vec<VulkanRuntimeExternalDeviceLocalResidentAllocation>,
    pub execution_transient_device_allocations:
        Vec<VulkanRuntimeDeviceLocalTransientAllocation>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VulkanRuntimePhysicalExecutionResidencyPlan {
    pub schema: String,
    pub package_id: String,
    pub device_plans: Vec<VulkanRuntimePhysicalExecutionDeviceResidencyPlan>,
    pub total_mount_device_local_bytes: usize,
    pub total_stream_device_local_bytes: usize,
    pub total_stream_shared_host_bytes: usize,
    pub execution_transient_shared_host_bytes_per_stream: usize,
    pub execution_transient_shared_host_allocations:
        Vec<VulkanRuntimeSharedHostTransientAllocation>,
    pub resident_shared_host_allocations: Vec<VulkanRuntimeSharedHostResidentAllocation>,
    pub shared_stream_control_host_bytes_per_stream: usize,
    pub graph_edge_memory_domains_bound: bool,
    pub feedback_control_memory_domain_bound: bool,
}

impl VulkanRuntimePhysicalExecutionResidencyPlan {
    pub fn plan(
        base: &VulkanRuntimeResidencyPlan,
        logical_device_ids: &[String],
        parameter_allocations: &VulkanDistributedParameterAllocationPlan,
        parameter_exclusions: &VulkanDistributedParameterExclusionPlan,
        activations: &VulkanDistributedActivationBufferPlan,
    ) -> Result<Self, VulkanRuntimeResidencyPlanError> {
        let allowed = logical_device_ids
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if allowed.is_empty()
            || allowed.len() != logical_device_ids.len()
            || allowed.iter().any(|device_id| device_id.trim().is_empty())
        {
            return Err(VulkanRuntimeResidencyPlanError(
                "physical execution residency requires unique nonempty logical devices".to_string(),
            ));
        }
        validate_physical_execution_residency_inputs(
            parameter_allocations,
            parameter_exclusions,
            activations,
        )?;
        let mut breakdowns = logical_device_ids
            .iter()
            .cloned()
            .map(|device_id| {
                (
                    device_id,
                    VulkanRuntimePhysicalExecutionResidencyBreakdown::default(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let mut resident_stream_device_allocations = logical_device_ids
            .iter()
            .cloned()
            .map(|device_id| (device_id, Vec::new()))
            .collect::<BTreeMap<_, _>>();

        let mut base_devices = BTreeSet::new();
        for device in &base.device_plans {
            ensure_physical_residency_device(&allowed, &device.device_id, "base residency")?;
            if !base_devices.insert(device.device_id.as_str()) {
                return Err(VulkanRuntimeResidencyPlanError(format!(
                    "base residency repeats logical device {:?}",
                    device.device_id
                )));
            }
            let store_bytes = match base.residency_policy {
                ResourceResidencyPolicy::Eager => {
                    device.resource_store.maximum_extra_device_bytes()?
                }
                ResourceResidencyPolicy::DemandPaged | ResourceResidencyPolicy::DemandRetained => {
                    device.resource_store.fixed_device_bytes()?
                }
            };
            let owner_stream_device_bytes = checked_residency_add(
                device.working_set.transient_state_bytes,
                device.working_set.activation_headroom_bytes,
                "owner per-stream residency",
            )?;
            let expected_current_parameter_bytes = checked_residency_add(
                device.parameter_residency.always_resident_bytes,
                device.parameter_residency.initial_dynamic_bytes,
                "current parameter residency",
            )?;
            if expected_current_parameter_bytes != device.parameter_residency.current_resident_bytes
            {
                return Err(VulkanRuntimeResidencyPlanError(format!(
                    "base residency on {:?} declares {} current parameter bytes but its static and initial dynamic parameters require {expected_current_parameter_bytes}",
                    device.device_id, device.parameter_residency.current_resident_bytes
                )));
            }
            let independently_admitted_resource_store_bytes = checked_residency_add(
                store_bytes,
                device.parameter_residency.initial_dynamic_bytes,
                "independently admitted compiled-resource residency",
            )?;
            let expected_initial_bytes = [
                device.parameter_residency.always_resident_bytes,
                independently_admitted_resource_store_bytes,
                owner_stream_device_bytes,
            ]
            .into_iter()
            .try_fold(0usize, |total, bytes| {
                checked_residency_add(total, bytes, "physical execution base residency")
            })?;
            if expected_initial_bytes != device.initial_device_resident_bytes {
                return Err(VulkanRuntimeResidencyPlanError(format!(
                    "base residency on {:?} declares {} initial bytes but its parameters, independent store, and stream working set require {expected_initial_bytes}",
                    device.device_id, device.initial_device_resident_bytes
                )));
            }
            let breakdown = breakdowns
                .get_mut(&device.device_id)
                .expect("base device was validated against the physical execution set");
            breakdown.owner_parameter_bytes_before_distributed_replacement =
                device.parameter_residency.always_resident_bytes;
            breakdown.independently_admitted_resource_store_bytes =
                independently_admitted_resource_store_bytes;
            breakdown.owner_stream_device_bytes = owner_stream_device_bytes;
            breakdown.owner_stream_control_device_bytes_per_stream =
                device.breakdown.stream_control_bytes;
            breakdown.owner_edge_buffer_bytes_per_stream = device.breakdown.edge_buffer_bytes;
            validate_resident_stream_allocation_ledger(device)?;
            *resident_stream_device_allocations
                .get_mut(&device.device_id)
                .expect("base device was validated against the physical execution set") =
                device.resident_stream_device_allocations.clone();
        }

        for allocation in &activations.allocations {
            let replaces_resident = match &allocation.storage {
                VulkanDistributedActivationStorage::ActivationSlot => true,
                VulkanDistributedActivationStorage::BoundaryInput
                | VulkanDistributedActivationStorage::BoundaryOutput => {
                    if allocation.signal_ids.len() != 1 {
                        return Err(VulkanRuntimeResidencyPlanError(format!(
                            "distributed {:?} allocation {:?}.slot_{} requires exactly one signal identity",
                            allocation.storage, allocation.component_id, allocation.slot,
                        )));
                    }
                    true
                }
                VulkanDistributedActivationStorage::Edge { .. } => false,
            };
            let allocations = resident_stream_device_allocations
                .get_mut(&allocation.owner_device_id)
                .ok_or_else(|| {
                    VulkanRuntimeResidencyPlanError(format!(
                        "distributed activation replaces resident storage on absent logical device {:?}",
                        allocation.owner_device_id,
                    ))
                })?;
            let mut removed_bytes = 0usize;
            let mut removed_count = 0usize;
            let mut retained = Vec::with_capacity(allocations.len());
            for resident in std::mem::take(allocations) {
                if distributed_activation_replaces_resident_allocation(allocation, &resident)? {
                    if allocation.byte_capacity != resident.byte_capacity {
                        return Err(VulkanRuntimeResidencyPlanError(format!(
                            "distributed activation {:?} replaces a {}-byte resident allocation with {} bytes",
                            allocation.storage, resident.byte_capacity, allocation.byte_capacity,
                        )));
                    }
                    removed_count = removed_count.checked_add(1).ok_or_else(|| {
                        VulkanRuntimeResidencyPlanError(
                            "distributed activation replacement count overflowed".to_string(),
                        )
                    })?;
                    removed_bytes = checked_residency_add(
                        removed_bytes,
                        resident.byte_capacity,
                        "replaced resident stream allocation",
                    )?;
                } else {
                    retained.push(resident);
                }
            }
            *allocations = retained;
            if replaces_resident && removed_count != 1 {
                return Err(VulkanRuntimeResidencyPlanError(format!(
                    "distributed activation {:?} for {:?}.slot_{} replaces {removed_count} resident allocations, expected exactly one",
                    allocation.storage, allocation.component_id, allocation.slot,
                )));
            }
            if removed_bytes > 0 {
                let breakdown = breakdowns
                    .get_mut(&allocation.owner_device_id)
                    .expect("distributed activation owner was validated above");
                breakdown.owner_stream_device_bytes = breakdown
                    .owner_stream_device_bytes
                    .checked_sub(removed_bytes)
                    .ok_or_else(|| {
                        VulkanRuntimeResidencyPlanError(format!(
                            "distributed activation replacement exceeds owner stream residency on {:?}",
                            allocation.owner_device_id,
                        ))
                    })?;
            }
        }

        let mut exclusion_devices = BTreeSet::new();
        for device in &parameter_exclusions.devices {
            ensure_physical_residency_device(
                &allowed,
                &device.device_id,
                "distributed parameter exclusion",
            )?;
            if !base_devices.contains(device.device_id.as_str()) {
                return Err(VulkanRuntimeResidencyPlanError(format!(
                    "distributed parameter exclusion targets helper device {:?} without owner residency",
                    device.device_id
                )));
            }
            if !exclusion_devices.insert(device.device_id.as_str()) {
                return Err(VulkanRuntimeResidencyPlanError(format!(
                    "distributed parameter exclusions repeat logical device {:?}",
                    device.device_id
                )));
            }
            let breakdown = breakdowns
                .get_mut(&device.device_id)
                .expect("exclusion device was validated above");
            if device.total_byte_capacity
                > breakdown.owner_parameter_bytes_before_distributed_replacement
            {
                return Err(VulkanRuntimeResidencyPlanError(format!(
                    "distributed replacement excludes {} bytes from {:?}, but its owner parameter residency contains only {} bytes",
                    device.total_byte_capacity,
                    device.device_id,
                    breakdown.owner_parameter_bytes_before_distributed_replacement
                )));
            }
            breakdown.excluded_owner_parameter_bytes = device.total_byte_capacity;
        }

        for allocation in &parameter_allocations.allocations {
            ensure_physical_residency_device(
                &allowed,
                &allocation.device_id,
                "distributed parameter allocation",
            )?;
            let breakdown = breakdowns
                .get_mut(&allocation.device_id)
                .expect("parameter device was validated above");
            breakdown.distributed_parameter_bytes = checked_residency_add(
                breakdown.distributed_parameter_bytes,
                allocation.byte_count,
                "distributed parameter residency",
            )?;
        }

        for allocation in &activations.allocations {
            // Graph-edge storage is deliberately deferred until the complete
            // produced-port fan-out and selected physical boundary route are
            // known. Charging the generic distributed activation here would
            // duplicate the buffer materialized by create_placed_device_links.
            if matches!(allocation.storage, VulkanDistributedActivationStorage::Edge { .. }) {
                continue;
            }
            ensure_physical_residency_activation_devices(
                &allowed,
                &allocation.owner_device_id,
                &allocation.device_ids,
                "distributed activation",
            )?;
            add_physical_residency_shared_activation(
                &mut breakdowns,
                &allocation.owner_device_id,
                allocation.byte_capacity,
                activations.route,
            )?;
        }
        for allocation in &activations.reduction_allocations {
            ensure_physical_residency_activation_devices(
                &allowed,
                &allocation.owner_device_id,
                &allocation.device_ids,
                "distributed reduction",
            )?;
            add_physical_residency_shared_activation(
                &mut breakdowns,
                &allocation.owner_device_id,
                allocation.byte_capacity,
                activations.route,
            )?;
        }
        for allocation in &activations.private_intermediate_allocations {
            if allocation.devices.is_empty() {
                return Err(VulkanRuntimeResidencyPlanError(
                    "distributed private activation has no devices".to_string(),
                ));
            }
            let mut private_devices = BTreeSet::new();
            for device in &allocation.devices {
                ensure_physical_residency_device(
                    &allowed,
                    &device.device_id,
                    "distributed private activation",
                )?;
                if !private_devices.insert(device.device_id.as_str()) || device.byte_capacity == 0 {
                    return Err(VulkanRuntimeResidencyPlanError(
                        "distributed private activation has duplicate devices or empty storage"
                            .to_string(),
                    ));
                }
                let breakdown = breakdowns
                    .get_mut(&device.device_id)
                    .expect("private activation device was validated above");
                breakdown.distributed_private_activation_device_bytes_per_stream =
                    checked_residency_add(
                        breakdown.distributed_private_activation_device_bytes_per_stream,
                        device.byte_capacity,
                        "distributed private activation residency",
                    )?;
            }
        }

        let mut total_mount_device_local_bytes = 0usize;
        let mut total_stream_device_local_bytes = 0usize;
        let mut total_stream_shared_host_bytes = 0usize;
        let mut device_plans = Vec::with_capacity(breakdowns.len());
        for (device_id, breakdown) in breakdowns {
            let retained_owner_bytes = breakdown
                .owner_parameter_bytes_before_distributed_replacement
                .checked_sub(breakdown.excluded_owner_parameter_bytes)
                .ok_or_else(|| {
                    VulkanRuntimeResidencyPlanError(
                        "distributed owner replacement accounting underflowed".to_string(),
                    )
                })?;
            let mount_device_local_bytes = checked_residency_add(
                retained_owner_bytes,
                breakdown.distributed_parameter_bytes,
                "physical execution mount residency",
            )?;
            let stream_device_local_bytes = checked_residency_add(
                breakdown.owner_stream_device_bytes,
                checked_residency_add(
                    breakdown.distributed_shared_activation_device_bytes_per_stream,
                    breakdown.distributed_private_activation_device_bytes_per_stream,
                    "distributed physical execution per-stream residency",
                )?,
                "physical execution per-stream residency",
            )?;
            let stream_shared_host_bytes = breakdown.distributed_shared_host_bytes_per_stream;
            total_mount_device_local_bytes = checked_residency_add(
                total_mount_device_local_bytes,
                mount_device_local_bytes,
                "physical execution total mount residency",
            )?;
            total_stream_device_local_bytes = checked_residency_add(
                total_stream_device_local_bytes,
                stream_device_local_bytes,
                "physical execution total per-stream residency",
            )?;
            total_stream_shared_host_bytes = checked_residency_add(
                total_stream_shared_host_bytes,
                stream_shared_host_bytes,
                "physical execution total shared-host residency",
            )?;
            let resident_stream_device_allocations = resident_stream_device_allocations
                .remove(&device_id)
                .expect("physical execution device allocation ledger was initialized");
            device_plans.push(VulkanRuntimePhysicalExecutionDeviceResidencyPlan {
                device_id,
                breakdown,
                mount_device_local_bytes,
                stream_device_local_bytes,
                stream_shared_host_bytes,
                resident_stream_device_allocations,
                external_device_local_resident_allocations: Vec::new(),
                execution_transient_device_allocations: Vec::new(),
            });
        }
        Ok(Self {
            schema: VULKAN_RUNTIME_PHYSICAL_EXECUTION_RESIDENCY_PLAN_SCHEMA.to_string(),
            package_id: base.package_id.clone(),
            device_plans,
            total_mount_device_local_bytes,
            total_stream_device_local_bytes,
            total_stream_shared_host_bytes,
            execution_transient_shared_host_bytes_per_stream: 0,
            execution_transient_shared_host_allocations: Vec::new(),
            resident_shared_host_allocations: Vec::new(),
            shared_stream_control_host_bytes_per_stream: 0,
            graph_edge_memory_domains_bound: false,
            feedback_control_memory_domain_bound: false,
        })
    }

    fn resize_feedback_control_residency(
        &mut self,
        exact_byte_capacity: usize,
    ) -> Result<(), VulkanRuntimeResidencyPlanError> {
        if exact_byte_capacity == 0 || self.feedback_control_memory_domain_bound {
            return Err(VulkanRuntimeResidencyPlanError(
                "feedback-control resizing requires a positive capacity before memory-domain binding"
                    .to_string(),
            ));
        }
        let matches = self
            .device_plans
            .iter()
            .enumerate()
            .flat_map(|(device_index, device)| {
                device
                    .resident_stream_device_allocations
                    .iter()
                    .enumerate()
                    .filter_map(move |(allocation_index, allocation)| {
                        matches!(
                            &allocation.kind,
                            VulkanRuntimeResidentStreamAllocationKind::RuntimeBuffer {
                                class: VulkanRuntimeResidentBufferClass::FeedbackWorkspace,
                                buffer_id,
                                ..
                            } if buffer_id == "control"
                        )
                        .then_some((device_index, allocation_index))
                    })
            })
            .collect::<Vec<_>>();
        let [(device_index, allocation_index)] = matches.as_slice() else {
            return Err(VulkanRuntimeResidencyPlanError(format!(
                "feedback-control resizing found {} control allocations, expected one",
                matches.len(),
            )));
        };
        let previous = self.device_plans[*device_index].resident_stream_device_allocations
            [*allocation_index]
            .byte_capacity;
        if previous == exact_byte_capacity {
            return Ok(());
        }
        let next_owner_stream_bytes = self.device_plans[*device_index]
            .breakdown
            .owner_stream_device_bytes
            .checked_sub(previous)
            .and_then(|bytes| bytes.checked_add(exact_byte_capacity))
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(
                    "feedback-control owner residency resizing overflowed".to_string(),
                )
            })?;
        let next_stream_bytes = self.device_plans[*device_index]
            .stream_device_local_bytes
            .checked_sub(previous)
            .and_then(|bytes| bytes.checked_add(exact_byte_capacity))
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(
                    "feedback-control stream residency resizing overflowed".to_string(),
                )
            })?;
        let next_total_stream_bytes = self
            .total_stream_device_local_bytes
            .checked_sub(previous)
            .and_then(|bytes| bytes.checked_add(exact_byte_capacity))
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(
                    "feedback-control total residency resizing overflowed".to_string(),
                )
            })?;
        let device = &mut self.device_plans[*device_index];
        device.resident_stream_device_allocations[*allocation_index].byte_capacity =
            exact_byte_capacity;
        device.breakdown.owner_stream_device_bytes = next_owner_stream_bytes;
        device.stream_device_local_bytes = next_stream_bytes;
        self.total_stream_device_local_bytes = next_total_stream_bytes;
        Ok(())
    }

    fn feedback_control_resident_byte_capacity(
        &self,
    ) -> Result<usize, VulkanRuntimeResidencyPlanError> {
        let device_capacities = self.device_plans.iter().flat_map(|device| {
            device
                .resident_stream_device_allocations
                .iter()
                .filter_map(|allocation| {
                    matches!(
                        &allocation.kind,
                        VulkanRuntimeResidentStreamAllocationKind::RuntimeBuffer {
                            class: VulkanRuntimeResidentBufferClass::FeedbackWorkspace,
                            buffer_id,
                            ..
                        } if buffer_id == "control"
                    )
                    .then_some(allocation.byte_capacity)
                })
        });
        let host_capacities = self.resident_shared_host_allocations.iter().filter_map(
            |allocation| {
                matches!(
                    allocation.kind,
                    VulkanRuntimeSharedHostResidentAllocationKind::FeedbackControl { .. }
                )
                .then_some(allocation.byte_capacity)
            },
        );
        let capacities = device_capacities
            .chain(host_capacities)
            .collect::<Vec<_>>();
        let [capacity] = capacities.as_slice() else {
            return Err(VulkanRuntimeResidencyPlanError(format!(
                "physical residency contains {} feedback-control allocations, expected one",
                capacities.len(),
            )));
        };
        Ok(*capacity)
    }

    fn bind_graph_edge_memory_domains(
        &mut self,
        edge_plans: &[VulkanPlacedEdgeIoPlan],
        activations: &VulkanDistributedActivationBufferPlan,
        selected_boundary_routes: &BTreeMap<usize, VulkanRuntimeMountedBoundaryRoute>,
        physical_device_by_logical_device: &BTreeMap<String, String>,
    ) -> Result<(), VulkanRuntimeResidencyPlanError> {
        if self.graph_edge_memory_domains_bound {
            return Err(VulkanRuntimeResidencyPlanError(
                "graph-edge memory domains were already bound".to_string(),
            ));
        }
        let mut next = self.clone();
        let planned_devices = next
            .device_plans
            .iter()
            .map(|device| device.device_id.as_str())
            .collect::<BTreeSet<_>>();
        if physical_device_by_logical_device.len() != planned_devices.len()
            || planned_devices.iter().any(|device_id| {
                !physical_device_by_logical_device.contains_key(*device_id)
            })
        {
            return Err(VulkanRuntimeResidencyPlanError(
                "graph-edge memory-domain binding is incomplete or contains extra logical devices"
                    .to_string(),
            ));
        }
        let groups = group_placed_edge_pairs_by_produced_port(
            pair_placed_edge_endpoints(edge_plans).map_err(residency_display_error)?,
        )
        .map_err(residency_display_error)?;
        let mut removals = BTreeSet::<(String, VulkanRuntimeResidentStreamAllocationKind)>::new();
        let mut additions = BTreeMap::<String, Vec<VulkanRuntimeResidentStreamAllocation>>::new();
        let mut external_additions = BTreeMap::<
            String,
            Vec<VulkanRuntimeExternalDeviceLocalResidentAllocation>,
        >::new();
        let mut shared_host_additions = Vec::<VulkanRuntimeSharedHostResidentAllocation>::new();

        for group in groups {
            let matching_local_edges = edge_plans
                .iter()
                .find(|plan| plan.device_id == group.source_device_id)
                .into_iter()
                .flat_map(|plan| &plan.local_edges)
                .filter(|edge| {
                    edge.source_component_id == group.source_component_id
                        && edge.source_port_id == group.source_port_id
                })
                .map(|edge| edge.edge_index)
                .collect::<BTreeSet<_>>();
            let produced_edge_indices = matching_local_edges
                .iter()
                .copied()
                .chain(
                    group
                        .edges
                        .iter()
                        .map(|(outgoing, _)| outgoing.edge_index),
                )
                .collect::<BTreeSet<_>>();
            let produced_edge_indices_vec = produced_edge_indices.iter().copied().collect::<Vec<_>>();
            let source_plan = next
                .device_plans
                .iter()
                .find(|device| device.device_id == group.source_device_id)
                .ok_or_else(|| {
                    VulkanRuntimeResidencyPlanError(format!(
                        "graph-edge produced port {}.{} references absent source device {:?}",
                        group.source_component_id, group.source_port_id, group.source_device_id,
                    ))
                })?;
            let produced_matches = source_plan
                .resident_stream_device_allocations
                .iter()
                .filter(|allocation| {
                    matches!(
                        &allocation.kind,
                        VulkanRuntimeResidentStreamAllocationKind::EdgeProducedPort {
                            component_id,
                            port_id,
                            edge_indices,
                        } if component_id == &group.source_component_id
                            && port_id == &group.source_port_id
                            && edge_indices == &produced_edge_indices_vec
                    )
                })
                .collect::<Vec<_>>();
            let [produced_allocation] = produced_matches.as_slice() else {
                return Err(VulkanRuntimeResidencyPlanError(format!(
                    "graph-edge produced port {}.{} resolves {} resident allocations, expected one",
                    group.source_component_id,
                    group.source_port_id,
                    produced_matches.len(),
                )));
            };
            if produced_allocation.byte_capacity != group.byte_capacity {
                return Err(VulkanRuntimeResidencyPlanError(format!(
                    "graph-edge produced port {}.{} has {} resident bytes but its route requires {}",
                    group.source_component_id,
                    group.source_port_id,
                    produced_allocation.byte_capacity,
                    group.byte_capacity,
                )));
            }

            let mut participant_device_ids = BTreeSet::from([group.source_device_id.clone()]);
            participant_device_ids.extend(
                group
                    .edges
                    .iter()
                    .map(|(_, incoming)| incoming.local_device_id.clone()),
            );
            let mut has_distributed_edge = false;
            for allocation in &activations.allocations {
                if matches!(
                    allocation.storage,
                    VulkanDistributedActivationStorage::Edge { edge_index, .. }
                        if produced_edge_indices.contains(&edge_index)
                ) {
                    if allocation.owner_device_id != group.source_device_id
                        || allocation.byte_capacity != group.byte_capacity
                    {
                        return Err(VulkanRuntimeResidencyPlanError(format!(
                            "distributed graph-edge allocation disagrees with produced port {}.{}",
                            group.source_component_id, group.source_port_id,
                        )));
                    }
                    has_distributed_edge = true;
                    participant_device_ids.extend(allocation.device_ids.iter().cloned());
                }
            }
            let mut logical_devices_by_physical =
                BTreeMap::<String, BTreeSet<String>>::new();
            for device_id in &participant_device_ids {
                let physical_device_id = physical_device_by_logical_device
                    .get(device_id)
                    .ok_or_else(|| {
                        VulkanRuntimeResidencyPlanError(format!(
                            "graph-edge produced port {}.{} has no physical binding for {device_id:?}",
                            group.source_component_id, group.source_port_id,
                        ))
                    })?;
                logical_devices_by_physical
                    .entry(physical_device_id.clone())
                    .or_default()
                    .insert(device_id.clone());
            }
            let source_physical_device_id = physical_device_by_logical_device
                .get(&group.source_device_id)
                .expect("source graph-edge device was validated above");
            let mut representative_device_ids = logical_devices_by_physical
                .iter()
                .map(|(physical_device_id, logical_device_ids)| {
                    if physical_device_id == source_physical_device_id {
                        group.source_device_id.clone()
                    } else {
                        logical_device_ids
                            .first()
                            .expect("physical graph-edge participant set is nonempty")
                            .clone()
                    }
                })
                .collect::<Vec<_>>();
            representative_device_ids.sort();
            let selected_route = required_vulkan_boundary_route_for_edge_group(
                &group,
                selected_boundary_routes,
            )
            .map_err(residency_display_error)?;
            let distributed_route = has_distributed_edge.then_some(match activations.route {
                VulkanSharedResidentBufferRoute::ExternalDeviceLocal => {
                    VulkanPlacedEdgeTransferRoute::ExternalDeviceLocal
                }
                VulkanSharedResidentBufferRoute::SharedHost => {
                    VulkanPlacedEdgeTransferRoute::DeviceLocalStaging
                }
            });
            let route = resolve_vulkan_produced_port_resident_route(
                &group,
                selected_route,
                distributed_route,
                logical_devices_by_physical.len(),
            )
            .map_err(residency_display_error)?;

            let mut incoming_by_physical = BTreeMap::<
                String,
                Vec<(String, VulkanRuntimeResidentStreamAllocationKind)>,
            >::new();
            for (_, incoming) in &group.edges {
                let destination_plan = next
                    .device_plans
                    .iter()
                    .find(|device| device.device_id == incoming.local_device_id)
                    .ok_or_else(|| {
                        VulkanRuntimeResidencyPlanError(format!(
                            "graph-edge {} references absent destination {:?}",
                            incoming.edge_index, incoming.local_device_id,
                        ))
                    })?;
                let kind = VulkanRuntimeResidentStreamAllocationKind::EdgeIncoming {
                    edge_index: incoming.edge_index,
                };
                let matches = destination_plan
                    .resident_stream_device_allocations
                    .iter()
                    .filter(|allocation| allocation.kind == kind)
                    .collect::<Vec<_>>();
                let [incoming_allocation] = matches.as_slice() else {
                    return Err(VulkanRuntimeResidencyPlanError(format!(
                        "graph-edge {} resolves {} incoming resident allocations, expected one",
                        incoming.edge_index,
                        matches.len(),
                    )));
                };
                if incoming_allocation.byte_capacity != group.byte_capacity {
                    return Err(VulkanRuntimeResidencyPlanError(format!(
                        "graph-edge {} incoming residency has {} bytes but its produced port requires {}",
                        incoming.edge_index,
                        incoming_allocation.byte_capacity,
                        group.byte_capacity,
                    )));
                }
                let physical_device_id = physical_device_by_logical_device
                    .get(&incoming.local_device_id)
                    .expect("incoming graph-edge device was validated above");
                incoming_by_physical
                    .entry(physical_device_id.clone())
                    .or_default()
                    .push((incoming.local_device_id.clone(), kind));
            }

            let produced_kind = produced_allocation.kind.clone();
            match route {
                None => {
                    for entries in incoming_by_physical.values() {
                        removals.extend(entries.iter().cloned());
                    }
                }
                Some(VulkanPlacedEdgeTransferRoute::ExternalDeviceLocal) => {
                    removals.insert((group.source_device_id.clone(), produced_kind));
                    for entries in incoming_by_physical.values() {
                        removals.extend(entries.iter().cloned());
                    }
                    external_additions
                        .entry(group.source_device_id.clone())
                        .or_default()
                        .push(VulkanRuntimeExternalDeviceLocalResidentAllocation {
                            kind: VulkanRuntimeExternalDeviceLocalResidentAllocationKind::EdgeProducedPort {
                                component_id: group.source_component_id.clone(),
                                port_id: group.source_port_id.clone(),
                                edge_indices: produced_edge_indices_vec.clone(),
                            },
                            owner_device_id: group.source_device_id.clone(),
                            participant_device_ids: representative_device_ids,
                            byte_capacity: group.byte_capacity,
                        });
                }
                Some(VulkanPlacedEdgeTransferRoute::DeviceLocalStaging) => {
                    for (physical_device_id, logical_device_ids) in
                        &logical_devices_by_physical
                    {
                        if physical_device_id == source_physical_device_id {
                            if let Some(entries) = incoming_by_physical.get(physical_device_id) {
                                removals.extend(entries.iter().cloned());
                            }
                            continue;
                        }
                        let representative = logical_device_ids
                            .first()
                            .expect("physical graph-edge participant set is nonempty");
                        let mut retained = false;
                        if let Some(entries) = incoming_by_physical.get(physical_device_id) {
                            for (device_id, kind) in entries {
                                if !retained && device_id == representative {
                                    retained = true;
                                } else {
                                    removals.insert((device_id.clone(), kind.clone()));
                                }
                            }
                        }
                        if !retained {
                            additions.entry(representative.clone()).or_default().push(
                                VulkanRuntimeResidentStreamAllocation {
                                    kind: VulkanRuntimeResidentStreamAllocationKind::EdgeStagingReplica {
                                        component_id: group.source_component_id.clone(),
                                        port_id: group.source_port_id.clone(),
                                        edge_indices: produced_edge_indices_vec.clone(),
                                    },
                                    byte_capacity: group.byte_capacity,
                                },
                            );
                        }
                    }
                    shared_host_additions.push(VulkanRuntimeSharedHostResidentAllocation {
                        kind: VulkanRuntimeSharedHostResidentAllocationKind::EdgeStaging {
                            component_id: group.source_component_id.clone(),
                            port_id: group.source_port_id.clone(),
                            edge_indices: produced_edge_indices_vec.clone(),
                        },
                        owner_device_id: group.source_device_id.clone(),
                        participant_device_ids: representative_device_ids,
                        byte_capacity: group.byte_capacity,
                    });
                }
                Some(route) => {
                    return Err(VulkanRuntimeResidencyPlanError(format!(
                        "graph-edge produced port {}.{} selects unsupported resident route {route:?}",
                        group.source_component_id, group.source_port_id,
                    )));
                }
            }
        }

        for device in &mut next.device_plans {
            let mut removed_bytes = 0usize;
            let mut retained = Vec::with_capacity(device.resident_stream_device_allocations.len());
            for allocation in std::mem::take(&mut device.resident_stream_device_allocations) {
                if removals.remove(&(device.device_id.clone(), allocation.kind.clone())) {
                    removed_bytes = checked_residency_add(
                        removed_bytes,
                        allocation.byte_capacity,
                        "removed logical graph-edge residency",
                    )?;
                } else {
                    retained.push(allocation);
                }
            }
            let new_allocations = additions.remove(&device.device_id).unwrap_or_default();
            let added_bytes = new_allocations.iter().try_fold(0usize, |total, allocation| {
                checked_residency_add(
                    total,
                    allocation.byte_capacity,
                    "added staged graph-edge residency",
                )
            })?;
            retained.extend(new_allocations);
            device.resident_stream_device_allocations = retained;
            device.breakdown.owner_stream_device_bytes = device
                .breakdown
                .owner_stream_device_bytes
                .checked_sub(removed_bytes)
                .and_then(|bytes| bytes.checked_add(added_bytes))
                .ok_or_else(|| {
                    VulkanRuntimeResidencyPlanError(format!(
                        "graph-edge device residency adjustment overflowed on {:?}",
                        device.device_id,
                    ))
                })?;
            device.stream_device_local_bytes = device
                .stream_device_local_bytes
                .checked_sub(removed_bytes)
                .and_then(|bytes| bytes.checked_add(added_bytes))
                .ok_or_else(|| {
                    VulkanRuntimeResidencyPlanError(format!(
                        "graph-edge stream residency adjustment overflowed on {:?}",
                        device.device_id,
                    ))
                })?;
            device.breakdown.owner_edge_buffer_bytes_per_stream = device
                .breakdown
                .owner_edge_buffer_bytes_per_stream
                .checked_sub(removed_bytes)
                .and_then(|bytes| bytes.checked_add(added_bytes))
                .ok_or_else(|| {
                    VulkanRuntimeResidencyPlanError(format!(
                        "graph-edge breakdown adjustment overflowed on {:?}",
                        device.device_id,
                    ))
                })?;

            let external_allocations = external_additions
                .remove(&device.device_id)
                .unwrap_or_default();
            let external_bytes = external_allocations.iter().try_fold(
                0usize,
                |total, allocation| {
                    checked_residency_add(
                        total,
                        allocation.byte_capacity,
                        "external graph-edge residency",
                    )
                },
            )?;
            device.breakdown.external_edge_device_bytes_per_stream = external_bytes;
            device.breakdown.owner_stream_device_bytes = checked_residency_add(
                device.breakdown.owner_stream_device_bytes,
                external_bytes,
                "external graph-edge owner residency",
            )?;
            device.breakdown.owner_edge_buffer_bytes_per_stream = checked_residency_add(
                device.breakdown.owner_edge_buffer_bytes_per_stream,
                external_bytes,
                "external graph-edge breakdown",
            )?;
            device.stream_device_local_bytes = checked_residency_add(
                device.stream_device_local_bytes,
                external_bytes,
                "external graph-edge stream residency",
            )?;
            device.external_device_local_resident_allocations = external_allocations;
        }
        if !removals.is_empty() || !additions.is_empty() || !external_additions.is_empty() {
            return Err(VulkanRuntimeResidencyPlanError(
                "graph-edge residency binding left unmatched device allocations".to_string(),
            ));
        }
        for allocation in &shared_host_additions {
            let owner = next
                .device_plans
                .iter_mut()
                .find(|device| device.device_id == allocation.owner_device_id)
                .expect("graph-edge shared-host owner was validated above");
            owner.breakdown.staged_edge_shared_host_bytes_per_stream = checked_residency_add(
                owner.breakdown.staged_edge_shared_host_bytes_per_stream,
                allocation.byte_capacity,
                "staged graph-edge shared-host breakdown",
            )?;
            owner.stream_shared_host_bytes = checked_residency_add(
                owner.stream_shared_host_bytes,
                allocation.byte_capacity,
                "staged graph-edge shared-host residency",
            )?;
        }
        next.resident_shared_host_allocations = shared_host_additions;
        next.total_stream_device_local_bytes = next.device_plans.iter().try_fold(
            0usize,
            |total, device| {
                checked_residency_add(
                    total,
                    device.stream_device_local_bytes,
                    "bound graph-edge total device residency",
                )
            },
        )?;
        next.total_stream_shared_host_bytes = next.device_plans.iter().try_fold(
            0usize,
            |total, device| {
                checked_residency_add(
                    total,
                    device.stream_shared_host_bytes,
                    "bound graph-edge total shared-host residency",
                )
            },
        )?;
        next.graph_edge_memory_domains_bound = true;
        *self = next;
        Ok(())
    }

    fn bind_feedback_control_memory_domain(
        &mut self,
        physical_device_by_logical_device: &BTreeMap<String, String>,
    ) -> Result<(), VulkanRuntimeResidencyPlanError> {
        if self.feedback_control_memory_domain_bound {
            return Err(VulkanRuntimeResidencyPlanError(
                "feedback-control memory domain was already bound".to_string(),
            ));
        }
        let mut next = self.clone();
        let planned_devices = next
            .device_plans
            .iter()
            .map(|device| device.device_id.as_str())
            .collect::<BTreeSet<_>>();
        if physical_device_by_logical_device.len() != planned_devices.len()
            || planned_devices.iter().any(|device_id| {
                !physical_device_by_logical_device.contains_key(*device_id)
            })
        {
            return Err(VulkanRuntimeResidencyPlanError(
                "feedback-control memory-domain binding is incomplete or contains extra logical devices"
                    .to_string(),
            ));
        }
        let matches = next
            .device_plans
            .iter()
            .enumerate()
            .flat_map(|(device_index, device)| {
                device
                    .resident_stream_device_allocations
                    .iter()
                    .enumerate()
                    .filter_map(move |(allocation_index, allocation)| {
                        matches!(
                            &allocation.kind,
                            VulkanRuntimeResidentStreamAllocationKind::RuntimeBuffer {
                                class: VulkanRuntimeResidentBufferClass::FeedbackWorkspace,
                                buffer_id,
                                ..
                            } if buffer_id == "control"
                        )
                        .then_some((device_index, allocation_index))
                    })
            })
            .collect::<Vec<_>>();
        let [(owner_index, allocation_index)] = matches.as_slice() else {
            return Err(VulkanRuntimeResidencyPlanError(format!(
                "feedback-control memory-domain binding found {} control allocations, expected one",
                matches.len(),
            )));
        };
        let owner_device_id = next.device_plans[*owner_index].device_id.clone();
        let owner_physical_device_id = physical_device_by_logical_device
            .get(&owner_device_id)
            .expect("feedback control owner was validated above");
        let mut logical_devices_by_physical = BTreeMap::<String, BTreeSet<String>>::new();
        for device_id in planned_devices {
            logical_devices_by_physical
                .entry(
                    physical_device_by_logical_device
                        .get(device_id)
                        .expect("feedback control participant was validated above")
                        .clone(),
                )
                .or_default()
                .insert(device_id.to_string());
        }
        if logical_devices_by_physical.len() == 1 {
            next.feedback_control_memory_domain_bound = true;
            *self = next;
            return Ok(());
        }

        let mut participant_device_ids = logical_devices_by_physical
            .iter()
            .map(|(physical_device_id, logical_device_ids)| {
                if physical_device_id == owner_physical_device_id {
                    owner_device_id.clone()
                } else {
                    logical_device_ids
                        .first()
                        .expect("feedback control physical participant is nonempty")
                        .clone()
                }
            })
            .collect::<Vec<_>>();
        participant_device_ids.sort();
        let allocation = next.device_plans[*owner_index]
            .resident_stream_device_allocations
            .remove(*allocation_index);
        let scope_id = match &allocation.kind {
            VulkanRuntimeResidentStreamAllocationKind::RuntimeBuffer {
                class: VulkanRuntimeResidentBufferClass::FeedbackWorkspace,
                scope_id,
                buffer_id,
            } if buffer_id == "control" => scope_id.clone(),
            _ => unreachable!("feedback control allocation identity was selected above"),
        };
        let owner = &mut next.device_plans[*owner_index];
        owner.breakdown.owner_stream_device_bytes = owner
            .breakdown
            .owner_stream_device_bytes
            .checked_sub(allocation.byte_capacity)
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(
                    "feedback-control device residency underflowed".to_string(),
                )
            })?;
        owner.stream_device_local_bytes = owner
            .stream_device_local_bytes
            .checked_sub(allocation.byte_capacity)
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(
                    "feedback-control stream device residency underflowed".to_string(),
                )
            })?;
        owner.breakdown.feedback_control_shared_host_bytes_per_stream = allocation.byte_capacity;
        owner.stream_shared_host_bytes = checked_residency_add(
            owner.stream_shared_host_bytes,
            allocation.byte_capacity,
            "feedback-control shared-host residency",
        )?;
        next.resident_shared_host_allocations
            .push(VulkanRuntimeSharedHostResidentAllocation {
                kind: VulkanRuntimeSharedHostResidentAllocationKind::FeedbackControl { scope_id },
                owner_device_id,
                participant_device_ids,
                byte_capacity: allocation.byte_capacity,
            });
        next.total_stream_device_local_bytes = next.device_plans.iter().try_fold(
            0usize,
            |total, device| {
                checked_residency_add(
                    total,
                    device.stream_device_local_bytes,
                    "feedback-control total device residency",
                )
            },
        )?;
        next.total_stream_shared_host_bytes = next.device_plans.iter().try_fold(
            0usize,
            |total, device| {
                checked_residency_add(
                    total,
                    device.stream_shared_host_bytes,
                    "feedback-control total shared-host residency",
                )
            },
        )?;
        next.feedback_control_memory_domain_bound = true;
        *self = next;
        Ok(())
    }

    fn bind_stream_control_memory_domain(
        &mut self,
        physical_device_by_logical_device: &BTreeMap<String, String>,
    ) -> Result<(), VulkanRuntimeResidencyPlanError> {
        let mut next = self.clone();
        let mut indices_by_physical_device = BTreeMap::<String, Vec<usize>>::new();
        for (index, device) in next.device_plans.iter().enumerate() {
            let physical_device_id = physical_device_by_logical_device
                .get(&device.device_id)
                .ok_or_else(|| {
                    VulkanRuntimeResidencyPlanError(format!(
                        "stream-control binding has no physical device for {:?}",
                        device.device_id,
                    ))
                })?;
            indices_by_physical_device
                .entry(physical_device_id.clone())
                .or_default()
                .push(index);
        }
        if indices_by_physical_device.is_empty()
            || physical_device_by_logical_device.len() != next.device_plans.len()
        {
            return Err(VulkanRuntimeResidencyPlanError(
                "stream-control binding is incomplete or contains extra logical devices"
                    .to_string(),
            ));
        }

        let spans_physical_devices = indices_by_physical_device.len() > 1;
        let mut removed_device_bytes = 0usize;
        for indices in indices_by_physical_device.values() {
            let retained_index = (!spans_physical_devices)
                .then(|| {
                    indices.iter().copied().find(|index| {
                        next.device_plans[*index]
                            .breakdown
                            .owner_stream_control_device_bytes_per_stream
                            > 0
                    })
                })
                .flatten();
            for index in indices {
                if Some(*index) == retained_index {
                    continue;
                }
                let device = &mut next.device_plans[*index];
                let control_bytes = device
                    .breakdown
                    .owner_stream_control_device_bytes_per_stream;
                device.breakdown.owner_stream_control_device_bytes_per_stream = 0;
                device.breakdown.owner_stream_device_bytes = device
                    .breakdown
                    .owner_stream_device_bytes
                    .checked_sub(control_bytes)
                    .ok_or_else(|| {
                        VulkanRuntimeResidencyPlanError(format!(
                            "stream-control bytes exceed owner working set on {:?}",
                            device.device_id,
                        ))
                    })?;
                device.stream_device_local_bytes = device
                    .stream_device_local_bytes
                    .checked_sub(control_bytes)
                    .ok_or_else(|| {
                        VulkanRuntimeResidencyPlanError(format!(
                            "stream-control bytes exceed stream residency on {:?}",
                            device.device_id,
                        ))
                    })?;
                removed_device_bytes = checked_residency_add(
                    removed_device_bytes,
                    control_bytes,
                    "physically bound stream-control residency",
                )?;
            }
        }
        next.total_stream_device_local_bytes = next
            .total_stream_device_local_bytes
            .checked_sub(removed_device_bytes)
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(
                    "stream-control total device residency underflowed".to_string(),
                )
            })?;

        if spans_physical_devices {
            if next.shared_stream_control_host_bytes_per_stream != 0 {
                return Err(VulkanRuntimeResidencyPlanError(
                    "stream-control memory domain was bound more than once".to_string(),
                ));
            }
            let representative = next.device_plans.first_mut().ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(
                    "stream-control binding has no representative device".to_string(),
                )
            })?;
            representative.stream_shared_host_bytes = checked_residency_add(
                representative.stream_shared_host_bytes,
                VULKAN_STREAM_CONTROL_BYTE_CAPACITY,
                "shared stream-control device residency",
            )?;
            next.total_stream_shared_host_bytes = checked_residency_add(
                next.total_stream_shared_host_bytes,
                VULKAN_STREAM_CONTROL_BYTE_CAPACITY,
                "shared stream-control total residency",
            )?;
            next.shared_stream_control_host_bytes_per_stream =
                VULKAN_STREAM_CONTROL_BYTE_CAPACITY;
        }
        *self = next;
        Ok(())
    }

    fn add_execution_transient_reservation(
        &mut self,
        device_allocations: &[VulkanRuntimeDeviceLocalTransientAllocation],
        shared_host_allocations: &[VulkanRuntimeSharedHostTransientAllocation],
    ) -> Result<(), VulkanRuntimeResidencyPlanError> {
        // Construct the augmented plan off to the side. A stale logical
        // binding or arithmetic failure must not partially mutate the
        // authoritative admission plan.
        let mut next = self.clone();
        if next.execution_transient_shared_host_bytes_per_stream != 0
            || !next
                .execution_transient_shared_host_allocations
                .is_empty()
            || next.device_plans.iter().any(|device| {
                device
                    .breakdown
                    .execution_transient_device_bytes_per_stream
                    != 0
                    || !device.execution_transient_device_allocations.is_empty()
            })
        {
            return Err(VulkanRuntimeResidencyPlanError(
                "execution transient reservation was already attached".to_string(),
            ));
        }
        for allocation in device_allocations {
            if allocation.logical_device_id.trim().is_empty()
                || allocation.concern.trim().is_empty()
                || allocation.byte_capacity == 0
            {
                return Err(VulkanRuntimeResidencyPlanError(format!(
                    "execution transient device allocation {:?} is malformed",
                    allocation.concern,
                )));
            }
            let device = next
                .device_plans
                .iter_mut()
                .find(|device| device.device_id == allocation.logical_device_id)
                .ok_or_else(|| {
                    VulkanRuntimeResidencyPlanError(format!(
                        "execution transient reservation references absent logical device {:?}",
                        allocation.logical_device_id,
                    ))
                })?;
            device.breakdown.execution_transient_device_bytes_per_stream =
                checked_residency_add(
                    device
                        .breakdown
                        .execution_transient_device_bytes_per_stream,
                    allocation.byte_capacity,
                    "execution transient device residency",
                )?;
            device.stream_device_local_bytes = checked_residency_add(
                device.stream_device_local_bytes,
                allocation.byte_capacity,
                "execution transient stream residency",
            )?;
            next.total_stream_device_local_bytes = checked_residency_add(
                next.total_stream_device_local_bytes,
                allocation.byte_capacity,
                "execution transient total stream residency",
            )?;
            device
                .execution_transient_device_allocations
                .push(allocation.clone());
        }
        let admitted_logical_devices = next
            .device_plans
            .iter()
            .map(|device| device.device_id.as_str())
            .collect::<BTreeSet<_>>();
        let shared_host_bytes = shared_host_allocations.iter().try_fold(
            0usize,
            |total, allocation| {
                if allocation.owner_device_id.trim().is_empty()
                    || allocation.concern.trim().is_empty()
                    || allocation.byte_capacity == 0
                    || !allocation
                        .participant_device_ids
                        .iter()
                        .any(|device_id| device_id == &allocation.owner_device_id)
                    || allocation
                        .participant_device_ids
                        .iter()
                        .any(|device_id| device_id.trim().is_empty())
                    || allocation
                        .participant_device_ids
                        .windows(2)
                        .any(|pair| pair[0] >= pair[1])
                    || allocation
                        .participant_device_ids
                        .iter()
                        .any(|device_id| !admitted_logical_devices.contains(device_id.as_str()))
                    || (allocation.mode
                        == VulkanRuntimeSharedHostTransientAllocationMode::CrossDeviceTimelineStaging
                        && allocation.participant_device_ids.len() != 2)
                {
                    return Err(VulkanRuntimeResidencyPlanError(format!(
                        "execution transient shared-host allocation {:?} is malformed",
                        allocation.concern,
                    )));
                }
                checked_residency_add(
                    total,
                    allocation.byte_capacity,
                    "execution transient shared-host allocation ledger",
                )
            },
        )?;
        next.execution_transient_shared_host_bytes_per_stream = checked_residency_add(
            next.execution_transient_shared_host_bytes_per_stream,
            shared_host_bytes,
            "execution transient shared-host residency",
        )?;
        next.total_stream_shared_host_bytes = checked_residency_add(
            next.total_stream_shared_host_bytes,
            shared_host_bytes,
            "execution transient total shared-host residency",
        )?;
        next.execution_transient_shared_host_allocations = shared_host_allocations.to_vec();
        *self = next;
        Ok(())
    }
}

fn validate_resident_stream_allocation_ledger(
    device: &VulkanRuntimeDeviceResidencyPlan,
) -> Result<(), VulkanRuntimeResidencyPlanError> {
    let mut identities = BTreeSet::new();
    let mut state_bytes = 0usize;
    let mut state_transaction_bytes = 0usize;
    let mut causal_verification_snapshot_bytes = 0usize;
    let mut selection_telemetry_bytes = 0usize;
    let mut activation_slot_bytes = 0usize;
    let mut boundary_buffer_bytes = 0usize;
    let mut edge_buffer_bytes = 0usize;
    let mut output_transducer_workspace_bytes = 0usize;
    let mut sampler_workspace_bytes = 0usize;
    let mut feedback_workspace_bytes = 0usize;
    for allocation in &device.resident_stream_device_allocations {
        if allocation.byte_capacity == 0 || !identities.insert(&allocation.kind) {
            return Err(VulkanRuntimeResidencyPlanError(format!(
                "base residency on {:?} has an empty or repeated resident stream allocation {:?}",
                device.device_id, allocation.kind,
            )));
        }
        let (bytes, label) = match &allocation.kind {
            VulkanRuntimeResidentStreamAllocationKind::State { .. } => {
                (&mut state_bytes, "resident stream state")
            }
            VulkanRuntimeResidentStreamAllocationKind::StateTransaction { .. } => {
                (&mut state_transaction_bytes, "resident state transaction")
            }
            VulkanRuntimeResidentStreamAllocationKind::CausalVerificationSnapshot { .. } => (
                &mut causal_verification_snapshot_bytes,
                "resident causal verification snapshot",
            ),
            VulkanRuntimeResidentStreamAllocationKind::SelectionTelemetry { .. } => {
                (&mut selection_telemetry_bytes, "resident selection telemetry")
            }
            VulkanRuntimeResidentStreamAllocationKind::ActivationSlot { .. } => {
                (&mut activation_slot_bytes, "resident activation slot")
            }
            VulkanRuntimeResidentStreamAllocationKind::BoundaryInput { .. }
            | VulkanRuntimeResidentStreamAllocationKind::BoundaryOutput { .. } => {
                (&mut boundary_buffer_bytes, "resident boundary buffer")
            }
            VulkanRuntimeResidentStreamAllocationKind::EdgeProducedPort { .. }
            | VulkanRuntimeResidentStreamAllocationKind::EdgeIncoming { .. }
            | VulkanRuntimeResidentStreamAllocationKind::EdgeStagingReplica { .. } => {
                (&mut edge_buffer_bytes, "resident edge buffer")
            }
            VulkanRuntimeResidentStreamAllocationKind::RuntimeBuffer { class, .. } => match class {
                VulkanRuntimeResidentBufferClass::OutputTransducerWorkspace => (
                    &mut output_transducer_workspace_bytes,
                    "resident output transducer workspace",
                ),
                VulkanRuntimeResidentBufferClass::SamplerWorkspace => {
                    (&mut sampler_workspace_bytes, "resident sampler workspace")
                }
                VulkanRuntimeResidentBufferClass::FeedbackWorkspace => {
                    (&mut feedback_workspace_bytes, "resident feedback workspace")
                }
            },
        };
        *bytes = checked_residency_add(*bytes, allocation.byte_capacity, label)?;
    }
    let declared = &device.breakdown;
    let actual = [
        ("stream state", state_bytes, declared.stream_state_bytes),
        (
            "state transaction",
            state_transaction_bytes,
            declared.state_transaction_bytes,
        ),
        (
            "causal verification snapshot",
            causal_verification_snapshot_bytes,
            declared.causal_verification_snapshot_bytes,
        ),
        (
            "selection telemetry",
            selection_telemetry_bytes,
            declared.selection_telemetry_bytes,
        ),
        (
            "activation slot",
            activation_slot_bytes,
            declared.activation_slot_bytes,
        ),
        (
            "boundary buffer",
            boundary_buffer_bytes,
            declared.boundary_buffer_bytes,
        ),
        ("edge buffer", edge_buffer_bytes, declared.edge_buffer_bytes),
        (
            "output transducer workspace",
            output_transducer_workspace_bytes,
            declared.output_transducer_workspace_bytes,
        ),
        (
            "sampler workspace",
            sampler_workspace_bytes,
            declared.sampler_workspace_bytes,
        ),
        (
            "feedback workspace",
            feedback_workspace_bytes,
            declared.feedback_workspace_bytes,
        ),
    ];
    if let Some((label, allocation_bytes, breakdown_bytes)) = actual
        .into_iter()
        .find(|(_, allocation_bytes, breakdown_bytes)| allocation_bytes != breakdown_bytes)
    {
        return Err(VulkanRuntimeResidencyPlanError(format!(
            "base residency on {:?} has {allocation_bytes} {label} allocation bytes but its breakdown declares {breakdown_bytes}",
            device.device_id,
        )));
    }
    Ok(())
}

fn distributed_activation_replaces_resident_allocation(
    allocation: &VulkanDistributedActivationBufferAllocation,
    resident: &VulkanRuntimeResidentStreamAllocation,
) -> Result<bool, VulkanRuntimeResidencyPlanError> {
    let replaced = match (&allocation.storage, &resident.kind) {
        (
            VulkanDistributedActivationStorage::ActivationSlot,
            VulkanRuntimeResidentStreamAllocationKind::ActivationSlot { component_id, slot },
        ) => allocation.component_id == *component_id && allocation.slot == *slot,
        (
            VulkanDistributedActivationStorage::BoundaryInput,
            VulkanRuntimeResidentStreamAllocationKind::BoundaryInput {
                component_id,
                signal_id,
            },
        ) => {
            let [distributed_signal_id] = allocation.signal_ids.as_slice() else {
                return Err(VulkanRuntimeResidencyPlanError(format!(
                    "distributed boundary-input allocation {:?}.slot_{} requires exactly one signal identity",
                    allocation.component_id, allocation.slot,
                )));
            };
            allocation.component_id == *component_id && distributed_signal_id == signal_id
        }
        (
            VulkanDistributedActivationStorage::BoundaryOutput,
            VulkanRuntimeResidentStreamAllocationKind::BoundaryOutput {
                component_id,
                signal_id,
            },
        ) => {
            let [distributed_signal_id] = allocation.signal_ids.as_slice() else {
                return Err(VulkanRuntimeResidencyPlanError(format!(
                    "distributed boundary-output allocation {:?}.slot_{} requires exactly one signal identity",
                    allocation.component_id, allocation.slot,
                )));
            };
            allocation.component_id == *component_id && distributed_signal_id == signal_id
        }
        (VulkanDistributedActivationStorage::Edge { .. }, _) => false,
        _ => false,
    };
    Ok(replaced)
}

fn validate_physical_execution_residency_inputs(
    parameter_allocations: &VulkanDistributedParameterAllocationPlan,
    parameter_exclusions: &VulkanDistributedParameterExclusionPlan,
    activations: &VulkanDistributedActivationBufferPlan,
) -> Result<(), VulkanRuntimeResidencyPlanError> {
    let parameter_bytes =
        parameter_allocations
            .allocations
            .iter()
            .try_fold(0usize, |total, allocation| {
                if allocation.byte_count == 0 {
                    return Err(VulkanRuntimeResidencyPlanError(
                        "distributed parameter residency contains an empty allocation".to_string(),
                    ));
                }
                checked_residency_add(
                    total,
                    allocation.byte_count,
                    "distributed parameter residency input",
                )
            })?;
    let parameter_tensor_count = parameter_allocations
        .allocations
        .iter()
        .map(|allocation| allocation.tensor.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    if parameter_allocations.allocation_count != parameter_allocations.allocations.len()
        || parameter_allocations.tensor_count != parameter_tensor_count
        || parameter_allocations.total_byte_capacity != parameter_bytes
    {
        return Err(VulkanRuntimeResidencyPlanError(
            "distributed parameter residency summary disagrees with its allocations".to_string(),
        ));
    }

    let exclusion_bytes =
        parameter_exclusions
            .devices
            .iter()
            .try_fold(0usize, |total, device| {
                checked_residency_add(
                    total,
                    device.total_byte_capacity,
                    "distributed exclusion residency input",
                )
            })?;
    let exclusion_allocation_count =
        parameter_exclusions
            .devices
            .iter()
            .try_fold(0usize, |total, device| {
                total.checked_add(device.tensors.len()).ok_or_else(|| {
                    VulkanRuntimeResidencyPlanError(
                        "distributed exclusion allocation count overflowed".to_string(),
                    )
                })
            })?;
    let exclusion_tensor_count = parameter_exclusions
        .devices
        .iter()
        .flat_map(|device| &device.tensors)
        .map(String::as_str)
        .collect::<BTreeSet<_>>()
        .len();
    if parameter_exclusions.device_count != parameter_exclusions.devices.len()
        || parameter_exclusions.unique_tensor_count != exclusion_tensor_count
        || parameter_exclusions.excluded_full_allocation_count != exclusion_allocation_count
        || parameter_exclusions.excluded_full_byte_capacity != exclusion_bytes
    {
        return Err(VulkanRuntimeResidencyPlanError(
            "distributed parameter exclusion summary disagrees with its devices".to_string(),
        ));
    }

    let shared_activation_bytes = activations
        .allocations
        .iter()
        .map(|allocation| allocation.byte_capacity)
        .chain(
            activations
                .reduction_allocations
                .iter()
                .map(|allocation| allocation.byte_capacity),
        )
        .try_fold(0usize, |total, bytes| {
            checked_residency_add(total, bytes, "distributed shared activation input")
        })?;
    let private_activation_bytes = activations
        .private_intermediate_allocations
        .iter()
        .flat_map(|allocation| &allocation.devices)
        .try_fold(0usize, |total, device| {
            checked_residency_add(
                total,
                device.byte_capacity,
                "distributed private activation input",
            )
        })?;
    let private_allocation_count = activations
        .private_intermediate_allocations
        .iter()
        .try_fold(0usize, |total, allocation| {
            total.checked_add(allocation.devices.len()).ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(
                    "distributed private activation count overflowed".to_string(),
                )
            })
        })?;
    let activation_allocation_count = activations
        .allocations
        .len()
        .checked_add(activations.reduction_allocations.len())
        .and_then(|count| count.checked_add(private_allocation_count))
        .ok_or_else(|| {
            VulkanRuntimeResidencyPlanError(
                "distributed activation allocation count overflowed".to_string(),
            )
        })?;
    if activations.allocation_count != activation_allocation_count
        || activations.total_shared_byte_capacity != shared_activation_bytes
        || activations.total_private_byte_capacity != private_activation_bytes
    {
        return Err(VulkanRuntimeResidencyPlanError(
            "distributed activation residency summary disagrees with its allocations".to_string(),
        ));
    }
    Ok(())
}

pub fn admit_vulkan_runtime_physical_execution_mount(
    plan: &VulkanRuntimePhysicalExecutionResidencyPlan,
    physical_device_by_logical_device: &BTreeMap<String, String>,
    safe_capacity_by_physical_device: &BTreeMap<String, usize>,
) -> Result<BTreeMap<String, usize>, VulkanRuntimeResidencyPlanError> {
    admit_vulkan_runtime_physical_execution_device_bytes(
        plan,
        physical_device_by_logical_device,
        safe_capacity_by_physical_device,
        |device| device.mount_device_local_bytes,
        "mount",
    )
}

pub fn admit_vulkan_runtime_physical_execution_stream(
    plan: &VulkanRuntimePhysicalExecutionResidencyPlan,
    physical_device_by_logical_device: &BTreeMap<String, String>,
    safe_capacity_by_physical_device: &BTreeMap<String, usize>,
    safe_host_bytes: usize,
) -> Result<BTreeMap<String, usize>, VulkanRuntimeResidencyPlanError> {
    if plan.total_stream_shared_host_bytes > safe_host_bytes {
        return Err(VulkanRuntimeResidencyPlanError(format!(
            "physical execution stream needs {} shared-host bytes but only {safe_host_bytes} safe host bytes are available",
            plan.total_stream_shared_host_bytes
        )));
    }
    admit_vulkan_runtime_physical_execution_device_bytes(
        plan,
        physical_device_by_logical_device,
        safe_capacity_by_physical_device,
        |device| device.stream_device_local_bytes,
        "stream",
    )
}

fn admit_vulkan_runtime_physical_execution_device_bytes<F>(
    plan: &VulkanRuntimePhysicalExecutionResidencyPlan,
    physical_device_by_logical_device: &BTreeMap<String, String>,
    safe_capacity_by_physical_device: &BTreeMap<String, usize>,
    bytes_for: F,
    phase: &str,
) -> Result<BTreeMap<String, usize>, VulkanRuntimeResidencyPlanError>
where
    F: Fn(&VulkanRuntimePhysicalExecutionDeviceResidencyPlan) -> usize,
{
    let mut required = BTreeMap::<String, usize>::new();
    for device in &plan.device_plans {
        let physical_device_id = physical_device_by_logical_device
            .get(&device.device_id)
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(format!(
                    "physical execution device {:?} has no physical-device binding",
                    device.device_id
                ))
            })?;
        let total = required.entry(physical_device_id.clone()).or_default();
        *total = checked_residency_add(*total, bytes_for(device), "physical execution admission")?;
    }
    for (physical_device_id, required_bytes) in &required {
        let safe_capacity = safe_capacity_by_physical_device
            .get(physical_device_id)
            .copied()
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(format!(
                    "physical device {physical_device_id:?} has no live capacity budget"
                ))
            })?;
        if *required_bytes > safe_capacity {
            return Err(VulkanRuntimeResidencyPlanError(format!(
                "physical device {physical_device_id:?} needs {required_bytes} {phase} device bytes but its live safe capacity is {safe_capacity}"
            )));
        }
    }
    Ok(required)
}

fn ensure_physical_residency_device(
    allowed: &BTreeSet<&str>,
    device_id: &str,
    label: &str,
) -> Result<(), VulkanRuntimeResidencyPlanError> {
    if device_id.trim().is_empty() || !allowed.contains(device_id) {
        return Err(VulkanRuntimeResidencyPlanError(format!(
            "{label} references logical device {device_id:?} outside the physical execution plan"
        )));
    }
    Ok(())
}

fn ensure_physical_residency_activation_devices(
    allowed: &BTreeSet<&str>,
    owner_device_id: &str,
    device_ids: &[String],
    label: &str,
) -> Result<(), VulkanRuntimeResidencyPlanError> {
    ensure_physical_residency_device(allowed, owner_device_id, label)?;
    let unique = device_ids
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if unique.len() != device_ids.len()
        || !unique.contains(owner_device_id)
        || unique
            .iter()
            .any(|device_id| ensure_physical_residency_device(allowed, device_id, label).is_err())
    {
        return Err(VulkanRuntimeResidencyPlanError(format!(
            "{label} has invalid physical execution participants"
        )));
    }
    Ok(())
}

fn add_physical_residency_shared_activation(
    breakdowns: &mut BTreeMap<String, VulkanRuntimePhysicalExecutionResidencyBreakdown>,
    owner_device_id: &str,
    byte_capacity: usize,
    route: VulkanSharedResidentBufferRoute,
) -> Result<(), VulkanRuntimeResidencyPlanError> {
    if byte_capacity == 0 {
        return Err(VulkanRuntimeResidencyPlanError(
            "distributed shared activation has empty storage".to_string(),
        ));
    }
    let owner = breakdowns
        .get_mut(owner_device_id)
        .expect("shared activation owner was validated above");
    match route {
        VulkanSharedResidentBufferRoute::ExternalDeviceLocal => {
            owner.distributed_shared_activation_device_bytes_per_stream = checked_residency_add(
                owner.distributed_shared_activation_device_bytes_per_stream,
                byte_capacity,
                "distributed shared device-local activation residency",
            )?;
        }
        VulkanSharedResidentBufferRoute::SharedHost => {
            owner.distributed_shared_host_bytes_per_stream = checked_residency_add(
                owner.distributed_shared_host_bytes_per_stream,
                byte_capacity,
                "distributed shared-host activation residency",
            )?;
        }
    }
    Ok(())
}
