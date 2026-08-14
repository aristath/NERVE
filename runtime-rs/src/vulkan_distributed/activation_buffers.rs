#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedActivationBufferPlan {
    pub allocations: Vec<VulkanDistributedActivationBufferAllocation>,
    pub reduction_allocations: Vec<VulkanDistributedReductionBufferAllocation>,
    pub private_intermediate_allocations: Vec<VulkanDistributedPrivateIntermediateBufferAllocation>,
    pub allocation_count: usize,
    pub import_count: usize,
    pub reference_count: usize,
    pub total_shared_byte_capacity: usize,
    pub total_private_byte_capacity: usize,
    pub route: VulkanSharedResidentBufferRoute,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedPrivateIntermediateDeviceAllocation {
    pub device_id: String,
    pub byte_capacity: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedPrivateIntermediateBufferAllocation {
    pub producer_dispatch_index: usize,
    pub consumer_dispatch_index: usize,
    pub component_id: String,
    pub signal_id: String,
    pub devices: Vec<VulkanDistributedPrivateIntermediateDeviceAllocation>,
}

impl VulkanDistributedActivationBufferPlan {
    pub fn has_same_shared_activation_interface(&self, other: &Self) -> bool {
        self.route == other.route && self.allocations == other.allocations
    }

    pub fn from_execution_plan_set(
        plans: &VulkanDistributedExecutionPlanSet,
    ) -> Result<Self, VulkanDistributedPlanError> {
        let alternatives = plans
            .all()
            .into_iter()
            .map(Self::from_execution_plan)
            .collect::<Result<Vec<_>, _>>()?;
        Self::merged_for_alternatives(&alternatives)
    }

    fn merged_for_alternatives(
        plans: &[VulkanDistributedActivationBufferPlan],
    ) -> Result<Self, VulkanDistributedPlanError> {
        let Some(first) = plans.first() else {
            return Err(VulkanDistributedPlanError(
                "distributed activation alternatives must not be empty".to_string(),
            ));
        };
        if plans.iter().any(|candidate| candidate.route != first.route) {
            return Err(VulkanDistributedPlanError(
                "distributed activation alternatives require different transport routes"
                    .to_string(),
            ));
        }
        let mut allocations = BTreeMap::<
            VulkanDistributedActivationBufferAllocationKey,
            VulkanDistributedActivationBufferAllocation,
        >::new();
        for plan in plans {
            for candidate in &plan.allocations {
                let key = distributed_activation_buffer_allocation_key(candidate)?;
                if candidate.byte_capacity == 0
                    || candidate.signal_ids.is_empty()
                    || candidate.device_ids.is_empty()
                    || candidate
                        .signal_ids
                        .windows(2)
                        .any(|pair| pair[0] >= pair[1])
                    || candidate
                        .device_ids
                        .windows(2)
                        .any(|pair| pair[0] >= pair[1])
                    || !candidate.device_ids.contains(&candidate.owner_device_id)
                {
                    return Err(VulkanDistributedPlanError(
                        "distributed alternative has an invalid shared activation allocation"
                            .to_string(),
                    ));
                }
                let Some(merged) = allocations.get_mut(&key) else {
                    allocations.insert(key, candidate.clone());
                    continue;
                };
                if merged.storage != candidate.storage
                    || merged.owner_device_id != candidate.owner_device_id
                    || merged.component_id != candidate.component_id
                    || merged.slot != candidate.slot
                    || merged.byte_capacity != candidate.byte_capacity
                {
                    return Err(VulkanDistributedPlanError(
                        "distributed alternatives disagree on one shared activation identity"
                            .to_string(),
                    ));
                }
                merged
                    .signal_ids
                    .extend(candidate.signal_ids.iter().cloned());
                merged.signal_ids.sort();
                merged.signal_ids.dedup();
                merged
                    .device_ids
                    .extend(candidate.device_ids.iter().cloned());
                merged.device_ids.sort();
                merged.device_ids.dedup();
                merged.input_use_count = merged.input_use_count.max(candidate.input_use_count);
                merged.output_use_count = merged.output_use_count.max(candidate.output_use_count);
            }
        }

        let reduction_key = |allocation: &VulkanDistributedReductionBufferAllocation| {
            (
                allocation.owner_device_id.clone(),
                allocation.dispatch_index,
                allocation.component_id.clone(),
                allocation.node_id.clone(),
            )
        };
        let mut reduction_allocations = BTreeMap::<
            (String, usize, String, String),
            VulkanDistributedReductionBufferAllocation,
        >::new();
        for plan in plans {
            for allocation in &plan.reduction_allocations {
                let expected_bytes = allocation
                    .plane_byte_capacity
                    .checked_mul(allocation.device_ids.len())
                    .ok_or_else(|| {
                        VulkanDistributedPlanError(
                            "distributed alternative reduction capacity overflowed".to_string(),
                        )
                    })?;
                if allocation.plane_byte_capacity == 0
                    || allocation.device_ids.is_empty()
                    || allocation.byte_capacity != expected_bytes
                    || allocation.device_ids.iter().collect::<BTreeSet<_>>().len()
                        != allocation.device_ids.len()
                    || !allocation.device_ids.contains(&allocation.owner_device_id)
                {
                    return Err(VulkanDistributedPlanError(
                        "distributed alternative has an invalid reduction allocation".to_string(),
                    ));
                }
                let key = reduction_key(allocation);
                if let Some(merged) = reduction_allocations.get_mut(&key) {
                    merged.plane_byte_capacity = merged
                        .plane_byte_capacity
                        .max(allocation.plane_byte_capacity);
                    merged
                        .device_ids
                        .extend(allocation.device_ids.iter().cloned());
                    merged.device_ids.sort();
                    merged.device_ids.dedup();
                    merged.byte_capacity = merged
                        .plane_byte_capacity
                        .checked_mul(merged.device_ids.len())
                        .ok_or_else(|| {
                            VulkanDistributedPlanError(
                                "merged distributed reduction capacity overflowed".to_string(),
                            )
                        })?;
                } else {
                    reduction_allocations.insert(key, allocation.clone());
                }
            }
        }

        let private_key = |allocation: &VulkanDistributedPrivateIntermediateBufferAllocation| {
            (
                allocation.producer_dispatch_index,
                allocation.consumer_dispatch_index,
                allocation.component_id.clone(),
                allocation.signal_id.clone(),
            )
        };
        let mut private_capacities =
            BTreeMap::<(usize, usize, String, String), BTreeMap<String, usize>>::new();
        for plan in plans {
            for allocation in &plan.private_intermediate_allocations {
                let key = private_key(allocation);
                if allocation.devices.is_empty() {
                    return Err(VulkanDistributedPlanError(
                        "distributed alternative has an empty private intermediate".to_string(),
                    ));
                }
                let candidate_devices = allocation
                    .devices
                    .iter()
                    .map(|device| (device.device_id.clone(), device.byte_capacity))
                    .collect::<BTreeMap<_, _>>();
                if candidate_devices.len() != allocation.devices.len()
                    || candidate_devices.iter().any(|(device_id, byte_capacity)| {
                        device_id.is_empty() || *byte_capacity == 0
                    })
                {
                    return Err(VulkanDistributedPlanError(
                        "distributed alternative has an invalid private intermediate allocation"
                            .to_string(),
                    ));
                }
                let merged_devices = private_capacities.entry(key).or_default();
                for (device_id, byte_capacity) in candidate_devices {
                    merged_devices
                        .entry(device_id)
                        .and_modify(|merged| *merged = (*merged).max(byte_capacity))
                        .or_insert(byte_capacity);
                }
            }
        }
        let private_intermediate_allocations = private_capacities
            .into_iter()
            .map(
                |(
                    (producer_dispatch_index, consumer_dispatch_index, component_id, signal_id),
                    devices,
                )| VulkanDistributedPrivateIntermediateBufferAllocation {
                    producer_dispatch_index,
                    consumer_dispatch_index,
                    component_id,
                    signal_id,
                    devices: devices
                        .into_iter()
                        .map(|(device_id, byte_capacity)| {
                            VulkanDistributedPrivateIntermediateDeviceAllocation {
                                device_id,
                                byte_capacity,
                            }
                        })
                        .collect(),
                },
            )
            .collect::<Vec<_>>();
        let allocations = allocations.into_values().collect::<Vec<_>>();
        let reduction_allocations = reduction_allocations.into_values().collect::<Vec<_>>();
        let import_count = allocations
            .iter()
            .map(|allocation| allocation.device_ids.len())
            .chain(
                reduction_allocations
                    .iter()
                    .map(|allocation| allocation.device_ids.len()),
            )
            .try_fold(0usize, |total, count| total.checked_add(count))
            .ok_or_else(|| {
                VulkanDistributedPlanError(
                    "merged distributed activation import count overflowed".to_string(),
                )
            })?;
        let total_shared_byte_capacity = allocations
            .iter()
            .map(|allocation| allocation.byte_capacity)
            .chain(
                reduction_allocations
                    .iter()
                    .map(|allocation| allocation.byte_capacity),
            )
            .try_fold(0usize, |total, count| total.checked_add(count))
            .ok_or_else(|| {
                VulkanDistributedPlanError(
                    "merged distributed shared activation capacity overflowed".to_string(),
                )
            })?;
        let total_private_byte_capacity = private_intermediate_allocations
            .iter()
            .flat_map(|allocation| &allocation.devices)
            .try_fold(0usize, |total, device| {
                total.checked_add(device.byte_capacity)
            })
            .ok_or_else(|| {
                VulkanDistributedPlanError(
                    "merged distributed private activation capacity overflowed".to_string(),
                )
            })?;
        let private_allocation_count = private_intermediate_allocations
            .iter()
            .try_fold(0usize, |total, allocation| {
                total.checked_add(allocation.devices.len())
            })
            .ok_or_else(|| {
                VulkanDistributedPlanError(
                    "merged distributed activation allocation count overflowed".to_string(),
                )
            })?;
        let allocation_count = allocations
            .len()
            .checked_add(reduction_allocations.len())
            .and_then(|count| count.checked_add(private_allocation_count))
            .ok_or_else(|| {
                VulkanDistributedPlanError(
                    "merged distributed activation allocation count overflowed".to_string(),
                )
            })?;
        let reference_count = allocations
            .iter()
            .try_fold(0usize, |total, allocation| {
                total
                    .checked_add(allocation.input_use_count)
                    .and_then(|count| count.checked_add(allocation.output_use_count))
            })
            .and_then(|count| {
                reduction_allocations
                    .iter()
                    .try_fold(count, |total, allocation| {
                        total
                            .checked_add(allocation.device_ids.len())
                            .and_then(|count| count.checked_add(1))
                    })
            })
            .and_then(|count| {
                private_intermediate_allocations
                    .iter()
                    .try_fold(count, |total, allocation| {
                        allocation
                            .devices
                            .len()
                            .checked_mul(2)
                            .and_then(|references| total.checked_add(references))
                    })
            })
            .ok_or_else(|| {
                VulkanDistributedPlanError(
                    "merged distributed activation reference count overflowed".to_string(),
                )
            })?;

        Ok(Self {
            allocations,
            reduction_allocations,
            private_intermediate_allocations,
            allocation_count,
            import_count,
            reference_count,
            total_shared_byte_capacity,
            total_private_byte_capacity,
            route: first.route,
        })
    }

    pub fn from_execution_plan(
        execution_plan: &VulkanDistributedExecutionPlan,
    ) -> Result<Self, VulkanDistributedPlanError> {
        let device_ids = execution_plan
            .device_ids
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let mut allocations = BTreeMap::<
            VulkanDistributedActivationBufferAllocationKey,
            VulkanDistributedActivationBufferAllocation,
        >::new();
        let mut reduction_allocations = Vec::new();
        let mut reduction_keys = BTreeSet::new();
        let mut private_intermediate_allocations = Vec::new();
        let mut private_producer_dispatches = BTreeSet::new();
        let mut private_consumer_dispatches = BTreeSet::new();
        for island in &execution_plan.execution_islands {
            for pair in island.dispatches.windows(2) {
                let producer = &pair[0];
                let consumer = &pair[1];
                if !local_shard_handoff(producer, consumer) {
                    continue;
                }
                if !private_producer_dispatches.insert(producer.dispatch_index)
                    || !private_consumer_dispatches.insert(consumer.dispatch_index)
                {
                    return Err(VulkanDistributedPlanError(
                        "distributed private intermediate dispatch is not one-to-one".to_string(),
                    ));
                }
                let devices = producer
                    .shards
                    .iter()
                    .zip(&consumer.shards)
                    .map(|(producer_shard, consumer_shard)| {
                        if producer_shard.device_id != consumer_shard.device_id
                            || producer_shard.output_byte_count
                                != consumer_shard.input_range.byte_count
                        {
                            return Err(VulkanDistributedPlanError(format!(
                                "private intermediate {} -> {} has incompatible shard storage on {:?}",
                                producer.node_id,
                                consumer.node_id,
                                producer_shard.device_id,
                            )));
                        }
                        Ok(VulkanDistributedPrivateIntermediateDeviceAllocation {
                            device_id: producer_shard.device_id.clone(),
                            byte_capacity: producer_shard.output_byte_count,
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if devices
                    .iter()
                    .map(|device| device.device_id.as_str())
                    .collect::<BTreeSet<_>>()
                    .len()
                    != devices.len()
                {
                    return Err(VulkanDistributedPlanError(format!(
                        "private intermediate {} -> {} repeats a participant device",
                        producer.node_id, consumer.node_id,
                    )));
                }
                private_intermediate_allocations.push(
                    VulkanDistributedPrivateIntermediateBufferAllocation {
                        producer_dispatch_index: producer.dispatch_index,
                        consumer_dispatch_index: consumer.dispatch_index,
                        component_id: producer.component_id.clone(),
                        signal_id: producer.output_activation.signal_id.clone(),
                        devices,
                    },
                );
            }
        }

        for dispatch in &execution_plan.dispatches {
            let participant_device_ids = dispatch
                .shards
                .iter()
                .map(|shard| shard.device_id.as_str())
                .collect::<BTreeSet<_>>();
            if participant_device_ids.is_empty() {
                return Err(VulkanDistributedPlanError(format!(
                    "distributed dispatch {}.{} has no device shards",
                    dispatch.component_id, dispatch.node_id
                )));
            }
            if !participant_device_ids.contains(dispatch.owner_device_id.as_str()) {
                return Err(VulkanDistributedPlanError(format!(
                    "distributed dispatch {}.{} does not include its owner {:?}",
                    dispatch.component_id, dispatch.node_id, dispatch.owner_device_id
                )));
            }
            if let Some(device_id) = participant_device_ids
                .iter()
                .find(|device_id| !device_ids.contains(**device_id))
            {
                return Err(VulkanDistributedPlanError(format!(
                    "distributed dispatch {}.{} uses device {device_id:?} outside the execution pool",
                    dispatch.component_id, dispatch.node_id
                )));
            }

            if !private_consumer_dispatches.contains(&dispatch.dispatch_index) {
                accumulate_activation_allocation(
                    &mut allocations,
                    &dispatch.owner_device_id,
                    &dispatch.input_activation,
                    &participant_device_ids,
                    VulkanDistributedActivationAccess::Input,
                )?;
            }
            for activation in &dispatch.auxiliary_input_activations {
                accumulate_activation_allocation(
                    &mut allocations,
                    &dispatch.owner_device_id,
                    activation,
                    &participant_device_ids,
                    VulkanDistributedActivationAccess::Input,
                )?;
            }
            for activation in &dispatch.selected_resource_activations {
                accumulate_activation_allocation(
                    &mut allocations,
                    &dispatch.owner_device_id,
                    activation,
                    &participant_device_ids,
                    VulkanDistributedActivationAccess::Input,
                )?;
            }
            let output_participant_device_ids = if dispatch.reduction.is_some() {
                BTreeSet::from([dispatch.owner_device_id.as_str()])
            } else {
                participant_device_ids.clone()
            };
            if !private_producer_dispatches.contains(&dispatch.dispatch_index) {
                accumulate_activation_allocation(
                    &mut allocations,
                    &dispatch.owner_device_id,
                    &dispatch.output_activation,
                    &output_participant_device_ids,
                    VulkanDistributedActivationAccess::Output,
                )?;
            }
            if let Some(reduction) = &dispatch.reduction {
                if !reduction_keys
                    .insert((dispatch.owner_device_id.as_str(), dispatch.dispatch_index))
                {
                    return Err(VulkanDistributedPlanError(format!(
                        "distributed reduction repeats dispatch {} owned by {:?}",
                        dispatch.dispatch_index, dispatch.owner_device_id
                    )));
                }
                let device_ids = dispatch
                    .shards
                    .iter()
                    .map(|shard| shard.device_id.clone())
                    .collect::<Vec<_>>();
                if device_ids.iter().collect::<BTreeSet<_>>().len() != device_ids.len() {
                    return Err(VulkanDistributedPlanError(format!(
                        "distributed reduction {}.{} repeats a participant device",
                        dispatch.component_id, dispatch.node_id
                    )));
                }
                let byte_capacity = reduction
                    .partial_byte_capacity
                    .checked_mul(device_ids.len())
                    .ok_or_else(|| {
                        VulkanDistributedPlanError(format!(
                            "distributed reduction {}.{} byte capacity overflowed",
                            dispatch.component_id, dispatch.node_id
                        ))
                    })?;
                reduction_allocations.push(VulkanDistributedReductionBufferAllocation {
                    owner_device_id: dispatch.owner_device_id.clone(),
                    dispatch_index: dispatch.dispatch_index,
                    component_id: dispatch.component_id.clone(),
                    node_id: dispatch.node_id.clone(),
                    plane_byte_capacity: reduction.partial_byte_capacity,
                    byte_capacity,
                    device_ids,
                });
            }
        }

        let activation_import_count =
            allocations.values().try_fold(0usize, |total, allocation| {
                total
                    .checked_add(allocation.device_ids.len())
                    .ok_or_else(|| {
                        VulkanDistributedPlanError(
                            "distributed activation import count overflowed".to_string(),
                        )
                    })
            })?;
        let reduction_import_count =
            reduction_allocations
                .iter()
                .try_fold(0usize, |total, allocation| {
                    total
                        .checked_add(allocation.device_ids.len())
                        .ok_or_else(|| {
                            VulkanDistributedPlanError(
                                "distributed reduction import count overflowed".to_string(),
                            )
                        })
                })?;
        let import_count = activation_import_count
            .checked_add(reduction_import_count)
            .ok_or_else(|| {
                VulkanDistributedPlanError("distributed buffer import count overflowed".to_string())
            })?;
        let activation_reference_count =
            allocations.values().try_fold(0usize, |total, allocation| {
                total
                    .checked_add(allocation.input_use_count)
                    .and_then(|count| count.checked_add(allocation.output_use_count))
                    .ok_or_else(|| {
                        VulkanDistributedPlanError(
                            "distributed activation reference count overflowed".to_string(),
                        )
                    })
            })?;
        let reduction_reference_count =
            reduction_allocations
                .iter()
                .try_fold(0usize, |total, allocation| {
                    total
                        .checked_add(allocation.device_ids.len())
                        .and_then(|count| count.checked_add(1))
                        .ok_or_else(|| {
                            VulkanDistributedPlanError(
                                "distributed reduction reference count overflowed".to_string(),
                            )
                        })
                })?;
        let reference_count = activation_reference_count
            .checked_add(reduction_reference_count)
            .and_then(|count| {
                private_intermediate_allocations
                    .iter()
                    .try_fold(count, |total, allocation| {
                        allocation
                            .devices
                            .len()
                            .checked_mul(2)
                            .and_then(|references| total.checked_add(references))
                    })
            })
            .ok_or_else(|| {
                VulkanDistributedPlanError(
                    "distributed buffer reference count overflowed".to_string(),
                )
            })?;
        let activation_byte_capacity =
            allocations.values().try_fold(0usize, |total, allocation| {
                total.checked_add(allocation.byte_capacity).ok_or_else(|| {
                    VulkanDistributedPlanError(
                        "distributed activation byte capacity overflowed".to_string(),
                    )
                })
            })?;
        let reduction_byte_capacity =
            reduction_allocations
                .iter()
                .try_fold(0usize, |total, allocation| {
                    total.checked_add(allocation.byte_capacity).ok_or_else(|| {
                        VulkanDistributedPlanError(
                            "distributed reduction byte capacity overflowed".to_string(),
                        )
                    })
                })?;
        let total_shared_byte_capacity = activation_byte_capacity
            .checked_add(reduction_byte_capacity)
            .ok_or_else(|| {
                VulkanDistributedPlanError(
                    "distributed buffer byte capacity overflowed".to_string(),
                )
            })?;
        let total_private_byte_capacity = private_intermediate_allocations
            .iter()
            .flat_map(|allocation| &allocation.devices)
            .try_fold(0usize, |total, device| {
                total.checked_add(device.byte_capacity).ok_or_else(|| {
                    VulkanDistributedPlanError(
                        "distributed private intermediate byte capacity overflowed".to_string(),
                    )
                })
            })?;
        let allocations = allocations.into_values().collect::<Vec<_>>();

        let allocation_count = allocations
            .len()
            .checked_add(reduction_allocations.len())
            .and_then(|count| {
                private_intermediate_allocations
                    .iter()
                    .try_fold(count, |total, allocation| {
                        total.checked_add(allocation.devices.len())
                    })
            })
            .ok_or_else(|| {
                VulkanDistributedPlanError(
                    "distributed buffer allocation count overflowed".to_string(),
                )
            })?;
        Ok(Self {
            allocation_count,
            allocations,
            reduction_allocations,
            private_intermediate_allocations,
            import_count,
            reference_count,
            total_shared_byte_capacity,
            total_private_byte_capacity,
            route: execution_plan.shared_activation_route,
        })
    }

    pub fn allocation(
        &self,
        owner_device_id: &str,
        component_id: &str,
        slot: usize,
    ) -> Option<&VulkanDistributedActivationBufferAllocation> {
        self.allocations.iter().find(|allocation| {
            allocation.storage == VulkanDistributedActivationStorage::ActivationSlot
                && allocation.owner_device_id == owner_device_id
                && allocation.component_id == component_id
                && allocation.slot == slot
        })
    }

    pub fn edge_allocation(
        &self,
        edge_index: usize,
    ) -> Option<&VulkanDistributedActivationBufferAllocation> {
        self.allocations.iter().find(|allocation| {
            matches!(
                allocation.storage,
                VulkanDistributedActivationStorage::Edge {
                    edge_index: candidate,
                    ..
                } if candidate == edge_index
            )
        })
    }

    pub fn reduction_allocation(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
    ) -> Option<&VulkanDistributedReductionBufferAllocation> {
        self.reduction_allocations.iter().find(|allocation| {
            allocation.owner_device_id == owner_device_id
                && allocation.dispatch_index == dispatch_index
        })
    }

    pub fn private_intermediate_allocation(
        &self,
        producer_dispatch_index: usize,
        consumer_dispatch_index: usize,
    ) -> Option<&VulkanDistributedPrivateIntermediateBufferAllocation> {
        self.private_intermediate_allocations
            .iter()
            .find(|allocation| {
                allocation.producer_dispatch_index == producer_dispatch_index
                    && allocation.consumer_dispatch_index == consumer_dispatch_index
            })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedActivationBufferAllocation {
    pub storage: VulkanDistributedActivationStorage,
    pub owner_device_id: String,
    pub component_id: String,
    pub slot: usize,
    pub byte_capacity: usize,
    pub signal_ids: Vec<String>,
    pub device_ids: Vec<String>,
    pub input_use_count: usize,
    pub output_use_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedReductionBufferAllocation {
    pub owner_device_id: String,
    pub dispatch_index: usize,
    pub component_id: String,
    pub node_id: String,
    pub plane_byte_capacity: usize,
    pub byte_capacity: usize,
    pub device_ids: Vec<String>,
}

pub struct VulkanDistributedActivationBuffers {
    pub plan: VulkanDistributedActivationBufferPlan,
    pub lane_capacity: usize,
    pub allocations: Vec<VulkanDistributedActivationBuffer>,
    pub reduction_allocations: Vec<VulkanDistributedReductionBuffer>,
    pub private_intermediate_allocations: Vec<VulkanDistributedPrivateIntermediateBuffer>,
    pub allocation_count: usize,
    pub import_count: usize,
    pub total_shared_byte_capacity: usize,
    pub total_private_byte_capacity: usize,
}

impl VulkanDistributedActivationBuffers {
    pub fn allocate<'a, F, E>(
        plan: &VulkanDistributedActivationBufferPlan,
        device_for: F,
    ) -> Result<Self, VulkanDistributedActivationBufferError>
    where
        F: FnMut(&str) -> Result<&'a VulkanComputeDevice, E>,
        E: Display,
    {
        Self::allocate_for_lanes(plan, 1, device_for)
    }

    /// Allocates every distributed activation except graph edges whose final
    /// transport is selected by the mounted boundary plan. Those edges are
    /// installed once by `create_placed_device_links`; allocating the generic
    /// route first would retain two physical copies at the mount peak.
    pub(crate) fn allocate_deferring_graph_edges<'a, F, E>(
        plan: &VulkanDistributedActivationBufferPlan,
        device_for: F,
    ) -> Result<Self, VulkanDistributedActivationBufferError>
    where
        F: FnMut(&str) -> Result<&'a VulkanComputeDevice, E>,
        E: Display,
    {
        Self::allocate_for_lanes_with_deferred_graph_edges(plan, 1, true, device_for)
    }

    pub fn allocate_for_lanes<'a, F, E>(
        plan: &VulkanDistributedActivationBufferPlan,
        lane_capacity: usize,
        device_for: F,
    ) -> Result<Self, VulkanDistributedActivationBufferError>
    where
        F: FnMut(&str) -> Result<&'a VulkanComputeDevice, E>,
        E: Display,
    {
        Self::allocate_for_lanes_with_deferred_graph_edges(
            plan,
            lane_capacity,
            false,
            device_for,
        )
    }

    fn allocate_for_lanes_with_deferred_graph_edges<'a, F, E>(
        plan: &VulkanDistributedActivationBufferPlan,
        lane_capacity: usize,
        defer_graph_edges: bool,
        mut device_for: F,
    ) -> Result<Self, VulkanDistributedActivationBufferError>
    where
        F: FnMut(&str) -> Result<&'a VulkanComputeDevice, E>,
        E: Display,
    {
        if lane_capacity == 0 {
            return Err(VulkanDistributedActivationBufferError(
                "distributed activation lane capacity must not be zero".to_string(),
            ));
        }
        let mut allocations = Vec::with_capacity(plan.allocations.len());
        let mut import_count = 0usize;
        let mut total_shared_byte_capacity = 0usize;
        for planned in &plan.allocations {
            let byte_capacity = planned
                .byte_capacity
                .checked_mul(lane_capacity)
                .ok_or_else(|| {
                    VulkanDistributedActivationBufferError(format!(
                        "distributed activation {}.slot_{} lane capacity overflowed",
                        planned.component_id, planned.slot
                    ))
                })?;
            let shared = if defer_graph_edges
                && matches!(planned.storage, VulkanDistributedActivationStorage::Edge { .. })
            {
                VulkanDistributedSharedBufferAllocation {
                    route: plan.route,
                    external_device_local_error: None,
                    device_buffers: BTreeMap::new(),
                }
            } else {
                allocate_distributed_shared_buffer(
                    &planned.owner_device_id,
                    &planned.device_ids,
                    byte_capacity,
                    plan.route,
                    &format!("activation {}.slot_{}", planned.component_id, planned.slot),
                    &mut device_for,
                )?
            };
            import_count = import_count
                .checked_add(shared.device_buffers.len())
                .ok_or_else(|| {
                    VulkanDistributedActivationBufferError(
                        "distributed activation import count overflowed".to_string(),
                    )
                })?;
            total_shared_byte_capacity = total_shared_byte_capacity
                .checked_add(byte_capacity)
                .ok_or_else(|| {
                    VulkanDistributedActivationBufferError(
                        "distributed activation byte capacity overflowed".to_string(),
                    )
                })?;
            allocations.push(VulkanDistributedActivationBuffer {
                planned: planned.clone(),
                route: shared.route,
                external_device_local_error: shared.external_device_local_error,
                device_buffers: shared.device_buffers,
            });
        }
        let mut reduction_allocations = Vec::with_capacity(plan.reduction_allocations.len());
        for planned in &plan.reduction_allocations {
            let byte_capacity = planned
                .byte_capacity
                .checked_mul(lane_capacity)
                .ok_or_else(|| {
                    VulkanDistributedActivationBufferError(format!(
                        "distributed reduction {}.{} lane capacity overflowed",
                        planned.component_id, planned.node_id
                    ))
                })?;
            let shared = allocate_distributed_shared_buffer(
                &planned.owner_device_id,
                &planned.device_ids,
                byte_capacity,
                plan.route,
                &format!("reduction {}.{}", planned.component_id, planned.node_id),
                &mut device_for,
            )?;
            import_count = import_count
                .checked_add(shared.device_buffers.len())
                .ok_or_else(|| {
                    VulkanDistributedActivationBufferError(
                        "distributed reduction import count overflowed".to_string(),
                    )
                })?;
            total_shared_byte_capacity = total_shared_byte_capacity
                .checked_add(byte_capacity)
                .ok_or_else(|| {
                    VulkanDistributedActivationBufferError(
                        "distributed reduction byte capacity overflowed".to_string(),
                    )
                })?;
            reduction_allocations.push(VulkanDistributedReductionBuffer {
                planned: planned.clone(),
                route: shared.route,
                external_device_local_error: shared.external_device_local_error,
                device_buffers: shared.device_buffers,
            });
        }
        let mut private_intermediate_allocations =
            Vec::with_capacity(plan.private_intermediate_allocations.len());
        let mut total_private_byte_capacity = 0usize;
        for planned in &plan.private_intermediate_allocations {
            let mut device_buffers = BTreeMap::new();
            for device_allocation in &planned.devices {
                let byte_capacity = device_allocation
                    .byte_capacity
                    .checked_mul(lane_capacity)
                    .ok_or_else(|| {
                        VulkanDistributedActivationBufferError(format!(
                            "distributed private intermediate {} lane capacity overflowed",
                            planned.signal_id
                        ))
                    })?;
                let device = device_for(&device_allocation.device_id).map_err(|error| {
                    VulkanDistributedActivationBufferError(format!(
                        "failed to resolve private intermediate {} device {:?}: {error}",
                        planned.signal_id, device_allocation.device_id
                    ))
                })?;
                let buffer = Arc::new(device.create_resident_buffer(byte_capacity).map_err(
                    |error| {
                        VulkanDistributedActivationBufferError(format!(
                            "failed to allocate {byte_capacity} private intermediate bytes for {} on {:?}: {error}",
                            planned.signal_id, device_allocation.device_id
                        ))
                    },
                )?);
                if device_buffers
                    .insert(device_allocation.device_id.clone(), buffer)
                    .is_some()
                {
                    return Err(VulkanDistributedActivationBufferError(format!(
                        "private intermediate {} repeats device {:?}",
                        planned.signal_id, device_allocation.device_id
                    )));
                }
                total_private_byte_capacity = total_private_byte_capacity
                    .checked_add(byte_capacity)
                    .ok_or_else(|| {
                        VulkanDistributedActivationBufferError(
                            "distributed private intermediate allocation overflowed".to_string(),
                        )
                    })?;
            }
            private_intermediate_allocations.push(VulkanDistributedPrivateIntermediateBuffer {
                planned: planned.clone(),
                device_buffers,
            });
        }
        let expected_private_byte_capacity = plan
            .total_private_byte_capacity
            .checked_mul(lane_capacity)
            .ok_or_else(|| {
                VulkanDistributedActivationBufferError(
                    "distributed private intermediate plan capacity overflowed".to_string(),
                )
            })?;
        if total_private_byte_capacity != expected_private_byte_capacity {
            return Err(VulkanDistributedActivationBufferError(format!(
                "distributed private intermediate allocation has {total_private_byte_capacity} bytes, expected {expected_private_byte_capacity}"
            )));
        }

        Ok(Self {
            plan: plan.clone(),
            lane_capacity,
            allocation_count: plan.allocation_count,
            allocations,
            reduction_allocations,
            private_intermediate_allocations,
            import_count,
            total_shared_byte_capacity,
            total_private_byte_capacity,
        })
    }

    pub(crate) fn finalize_deferred_graph_edges(
        &mut self,
    ) -> Result<(), VulkanDistributedActivationBufferError> {
        for allocation in &self.allocations {
            validate_final_distributed_activation_devices(
                &allocation.planned,
                allocation.device_buffers.keys().map(String::as_str),
            )?;
        }
        self.import_count = self
            .allocations
            .iter()
            .map(|allocation| allocation.device_buffers.len())
            .chain(
                self.reduction_allocations
                    .iter()
                    .map(|allocation| allocation.device_buffers.len()),
            )
            .try_fold(0usize, |total, count| total.checked_add(count))
            .ok_or_else(|| {
                VulkanDistributedActivationBufferError(
                    "final distributed activation import count overflowed".to_string(),
                )
            })?;
        Ok(())
    }

    pub fn activation_buffer(
        &self,
        dispatch_owner_device_id: &str,
        activation: &VulkanDistributedActivationSlot,
        device_id: &str,
    ) -> Option<&Arc<VulkanResidentBuffer>> {
        self.allocations
            .iter()
            .find(|allocation| {
                distributed_activation_allocation_matches(
                    dispatch_owner_device_id,
                    activation,
                    &allocation.planned,
                )
            })
            .and_then(|allocation| allocation.device_buffers.get(device_id))
    }

    pub fn activation_overrides_for_owner_device(
        &self,
        owner_device_id: &str,
    ) -> Vec<VulkanActivationSlotBufferOverride> {
        self.allocations
            .iter()
            .filter(|allocation| {
                allocation.planned.storage == VulkanDistributedActivationStorage::ActivationSlot
                    && allocation.planned.owner_device_id == owner_device_id
            })
            .filter_map(|allocation| {
                allocation
                    .device_buffers
                    .get(owner_device_id)
                    .map(|buffer| VulkanActivationSlotBufferOverride {
                        component_id: allocation.planned.component_id.clone(),
                        slot: allocation.planned.slot,
                        buffer: Arc::clone(buffer),
                    })
            })
            .collect()
    }

    pub fn boundary_overrides_for_owner_device(
        &self,
        owner_device_id: &str,
    ) -> Vec<VulkanModelBoundaryBufferOverride> {
        self.allocations
            .iter()
            .filter(|allocation| allocation.planned.owner_device_id == owner_device_id)
            .filter_map(|allocation| {
                let direction = match allocation.planned.storage {
                    VulkanDistributedActivationStorage::BoundaryInput => {
                        VulkanModelBoundaryDirection::Input
                    }
                    VulkanDistributedActivationStorage::BoundaryOutput => {
                        VulkanModelBoundaryDirection::Output
                    }
                    _ => return None,
                };
                allocation
                    .device_buffers
                    .get(owner_device_id)
                    .and_then(|buffer| {
                        Some(VulkanModelBoundaryBufferOverride {
                            direction,
                            component_id: allocation.planned.component_id.clone(),
                            signal_id: allocation.planned.signal_ids.first()?.clone(),
                            buffer: Arc::clone(buffer),
                        })
                    })
            })
            .collect()
    }

    pub fn edge_buffer(
        &self,
        edge_index: usize,
        device_id: &str,
    ) -> Option<&Arc<VulkanResidentBuffer>> {
        self.edge_allocation(edge_index)
            .and_then(|allocation| allocation.device_buffers.get(device_id))
    }

    pub fn reduction_partial_buffer(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
        device_id: &str,
    ) -> Option<&Arc<VulkanResidentBuffer>> {
        self.reduction_allocations
            .iter()
            .find(|allocation| {
                allocation.planned.owner_device_id == owner_device_id
                    && allocation.planned.dispatch_index == dispatch_index
            })
            .and_then(|allocation| allocation.device_buffers.get(device_id))
    }

    pub fn private_intermediate_buffer(
        &self,
        producer_dispatch_index: usize,
        consumer_dispatch_index: usize,
        device_id: &str,
    ) -> Option<&Arc<VulkanResidentBuffer>> {
        self.private_intermediate_allocations
            .iter()
            .find(|allocation| {
                allocation.planned.producer_dispatch_index == producer_dispatch_index
                    && allocation.planned.consumer_dispatch_index == consumer_dispatch_index
            })
            .and_then(|allocation| allocation.device_buffers.get(device_id))
    }

    pub(crate) fn edge_allocation(
        &self,
        edge_index: usize,
    ) -> Option<&VulkanDistributedActivationBuffer> {
        self.allocations.iter().find(|allocation| {
            matches!(
                allocation.planned.storage,
                VulkanDistributedActivationStorage::Edge {
                    edge_index: candidate,
                    ..
                } if candidate == edge_index
            )
        })
    }
}

fn validate_final_distributed_activation_devices<'a>(
    planned: &VulkanDistributedActivationBufferAllocation,
    actual_device_ids: impl IntoIterator<Item = &'a str>,
) -> Result<(), VulkanDistributedActivationBufferError> {
    let actual = actual_device_ids.into_iter().collect::<BTreeSet<_>>();
    let expected = planned
        .device_ids
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if actual != expected || expected.len() != planned.device_ids.len() {
        return Err(VulkanDistributedActivationBufferError(format!(
            "distributed activation {}.slot_{} finalized {} buffers for {} declared devices",
            planned.component_id,
            planned.slot,
            actual.len(),
            planned.device_ids.len(),
        )));
    }
    Ok(())
}

pub struct VulkanDistributedActivationBuffer {
    pub planned: VulkanDistributedActivationBufferAllocation,
    pub route: VulkanSharedResidentBufferRoute,
    pub external_device_local_error: Option<String>,
    pub device_buffers: BTreeMap<String, Arc<VulkanResidentBuffer>>,
}

pub struct VulkanDistributedReductionBuffer {
    pub planned: VulkanDistributedReductionBufferAllocation,
    pub route: VulkanSharedResidentBufferRoute,
    pub external_device_local_error: Option<String>,
    pub device_buffers: BTreeMap<String, Arc<VulkanResidentBuffer>>,
}

pub struct VulkanDistributedPrivateIntermediateBuffer {
    pub planned: VulkanDistributedPrivateIntermediateBufferAllocation,
    pub device_buffers: BTreeMap<String, Arc<VulkanResidentBuffer>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedActivationBufferError(pub String);

impl Display for VulkanDistributedActivationBufferError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for VulkanDistributedActivationBufferError {}

pub(crate) struct VulkanDistributedSharedBufferAllocation {
    pub(crate) route: VulkanSharedResidentBufferRoute,
    pub(crate) external_device_local_error: Option<String>,
    pub(crate) device_buffers: BTreeMap<String, Arc<VulkanResidentBuffer>>,
}

pub(crate) fn allocate_distributed_shared_buffer<'a, F, E>(
    owner_device_id: &str,
    device_ids: &[String],
    byte_capacity: usize,
    route: VulkanSharedResidentBufferRoute,
    label: &str,
    device_for: &mut F,
) -> Result<VulkanDistributedSharedBufferAllocation, VulkanDistributedActivationBufferError>
where
    F: FnMut(&str) -> Result<&'a VulkanComputeDevice, E>,
    E: Display,
{
    let owner = device_for(owner_device_id).map_err(|error| {
        VulkanDistributedActivationBufferError(format!(
            "failed to resolve distributed {label} owner {owner_device_id:?}: {error}"
        ))
    })?;
    let peers = device_ids
        .iter()
        .filter(|device_id| device_id.as_str() != owner_device_id)
        .map(|device_id| {
            device_for(device_id).map_err(|error| {
                VulkanDistributedActivationBufferError(format!(
                    "failed to resolve distributed {label} participant {device_id:?}: {error}"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let shared = owner
        .create_shared_resident_buffers_for_route(&peers, byte_capacity, route)
        .map_err(|error| {
            VulkanDistributedActivationBufferError(format!(
                "failed to allocate {byte_capacity} shared bytes for distributed {label}: {error}"
            ))
        })?;
    let mut buffers = shared.buffers.into_iter();
    let mut device_buffers = BTreeMap::from([(
        owner_device_id.to_string(),
        buffers
            .next()
            .expect("shared allocation always contains its owner"),
    )]);
    for (device_id, buffer) in device_ids
        .iter()
        .filter(|device_id| device_id.as_str() != owner_device_id)
        .zip(buffers)
    {
        if device_buffers.insert(device_id.clone(), buffer).is_some() {
            return Err(VulkanDistributedActivationBufferError(format!(
                "distributed {label} repeats device {device_id:?}"
            )));
        }
    }
    if device_buffers.len() != device_ids.len() {
        return Err(VulkanDistributedActivationBufferError(format!(
            "distributed {label} resolved {} buffers for {} devices",
            device_buffers.len(),
            device_ids.len()
        )));
    }
    Ok(VulkanDistributedSharedBufferAllocation {
        route: shared.route,
        external_device_local_error: shared.external_device_local_error,
        device_buffers,
    })
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum VulkanDistributedActivationBufferAllocationKey {
    ActivationSlot {
        owner_device_id: String,
        component_id: String,
        slot: usize,
    },
    BoundaryInput {
        owner_device_id: String,
        component_id: String,
        signal_id: String,
    },
    BoundaryOutput {
        owner_device_id: String,
        component_id: String,
        signal_id: String,
    },
    Edge {
        edge_index: usize,
        owner_device_id: String,
    },
}

fn distributed_activation_buffer_allocation_key(
    allocation: &VulkanDistributedActivationBufferAllocation,
) -> Result<VulkanDistributedActivationBufferAllocationKey, VulkanDistributedPlanError> {
    match &allocation.storage {
        VulkanDistributedActivationStorage::ActivationSlot => Ok(
            VulkanDistributedActivationBufferAllocationKey::ActivationSlot {
                owner_device_id: allocation.owner_device_id.clone(),
                component_id: allocation.component_id.clone(),
                slot: allocation.slot,
            },
        ),
        VulkanDistributedActivationStorage::BoundaryInput => {
            let [signal_id] = allocation.signal_ids.as_slice() else {
                return Err(VulkanDistributedPlanError(
                    "distributed boundary-input allocation requires exactly one signal identity"
                        .to_string(),
                ));
            };
            Ok(
                VulkanDistributedActivationBufferAllocationKey::BoundaryInput {
                    owner_device_id: allocation.owner_device_id.clone(),
                    component_id: allocation.component_id.clone(),
                    signal_id: signal_id.clone(),
                },
            )
        }
        VulkanDistributedActivationStorage::BoundaryOutput => {
            let [signal_id] = allocation.signal_ids.as_slice() else {
                return Err(VulkanDistributedPlanError(
                    "distributed boundary-output allocation requires exactly one signal identity"
                        .to_string(),
                ));
            };
            Ok(
                VulkanDistributedActivationBufferAllocationKey::BoundaryOutput {
                    owner_device_id: allocation.owner_device_id.clone(),
                    component_id: allocation.component_id.clone(),
                    signal_id: signal_id.clone(),
                },
            )
        }
        VulkanDistributedActivationStorage::Edge {
            edge_index,
            owner_device_id,
        } => {
            if owner_device_id != &allocation.owner_device_id {
                return Err(VulkanDistributedPlanError(
                    "distributed edge allocation disagrees with its embedded owner".to_string(),
                ));
            }
            Ok(VulkanDistributedActivationBufferAllocationKey::Edge {
                edge_index: *edge_index,
                owner_device_id: owner_device_id.clone(),
            })
        }
    }
}

fn distributed_activation_allocation_key(
    dispatch_owner_device_id: &str,
    activation: &VulkanDistributedActivationSlot,
) -> VulkanDistributedActivationBufferAllocationKey {
    match &activation.storage {
        VulkanDistributedActivationStorage::ActivationSlot => {
            VulkanDistributedActivationBufferAllocationKey::ActivationSlot {
                owner_device_id: dispatch_owner_device_id.to_string(),
                component_id: activation.component_id.clone(),
                slot: activation.slot,
            }
        }
        VulkanDistributedActivationStorage::BoundaryInput => {
            VulkanDistributedActivationBufferAllocationKey::BoundaryInput {
                owner_device_id: dispatch_owner_device_id.to_string(),
                component_id: activation.component_id.clone(),
                signal_id: activation.signal_id.clone(),
            }
        }
        VulkanDistributedActivationStorage::BoundaryOutput => {
            VulkanDistributedActivationBufferAllocationKey::BoundaryOutput {
                owner_device_id: dispatch_owner_device_id.to_string(),
                component_id: activation.component_id.clone(),
                signal_id: activation.signal_id.clone(),
            }
        }
        VulkanDistributedActivationStorage::Edge {
            edge_index,
            owner_device_id,
        } => VulkanDistributedActivationBufferAllocationKey::Edge {
            edge_index: *edge_index,
            owner_device_id: owner_device_id.clone(),
        },
    }
}

fn distributed_activation_allocation_matches(
    dispatch_owner_device_id: &str,
    activation: &VulkanDistributedActivationSlot,
    allocation: &VulkanDistributedActivationBufferAllocation,
) -> bool {
    if allocation.storage != activation.storage {
        return false;
    }
    match &activation.storage {
        VulkanDistributedActivationStorage::ActivationSlot => {
            allocation.owner_device_id == dispatch_owner_device_id
                && allocation.component_id == activation.component_id
                && allocation.slot == activation.slot
        }
        VulkanDistributedActivationStorage::BoundaryInput
        | VulkanDistributedActivationStorage::BoundaryOutput => {
            allocation.owner_device_id == dispatch_owner_device_id
                && allocation.component_id == activation.component_id
                && allocation.signal_ids.contains(&activation.signal_id)
        }
        VulkanDistributedActivationStorage::Edge {
            edge_index,
            owner_device_id,
        } => {
            allocation.owner_device_id == *owner_device_id
                && matches!(
                    allocation.storage,
                    VulkanDistributedActivationStorage::Edge {
                        edge_index: candidate,
                        ..
                    } if candidate == *edge_index
                )
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VulkanDistributedActivationAccess {
    Input,
    Output,
}
