#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedSelectedResourceStorePlan {
    pub devices: Vec<VulkanDistributedSelectedResourceDevicePlan>,
    /// Logical selected resources whose immutable parameter group is split
    /// across more than one execution device. Every member is one physical
    /// fragment of the same all-or-nothing residency cohort.
    pub tensor_sharded_residency_cohorts:
        Vec<VulkanDistributedSelectedResourceResidencyCohortPlan>,
    pub device_count: usize,
    pub selector_count: usize,
    pub selector_placement_count: usize,
    pub unique_atomic_group_count: usize,
    pub total_addressable_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedSelectedResourceResidencyCohortPlan {
    pub selector_id: String,
    pub resource_index: usize,
    pub atomic_group_id: String,
    pub members: Vec<VulkanDistributedSelectedResourceResidencyCohortMemberPlan>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedSelectedResourceResidencyCohortMemberPlan {
    pub device_id: String,
    pub logical_start: usize,
    pub logical_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedSelectedResourceDevicePlan {
    pub device_id: String,
    pub selectors: Vec<VulkanDistributedSelectedResourceOwnership>,
    pub unique_atomic_group_count: usize,
    pub maximum_atomic_group_bytes: usize,
    pub maximum_load_wave_bytes: usize,
    pub total_addressable_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedSelectedResourceOwnership {
    pub execution_scope: String,
    pub selector_id: String,
    pub component_id: String,
    pub node_id: String,
    pub domain_id: String,
    pub selection_signal: String,
    pub resource_count: usize,
    pub selection_count_per_activation: usize,
    pub owned_resource_indices: Vec<usize>,
    pub atomic_group_ids: Vec<String>,
    pub atomic_group_byte_counts: Vec<usize>,
    pub fragmented_resources: Vec<VulkanDistributedSelectedResourceFragmentOwnership>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedSelectedResourceFragmentOwnership {
    pub resource_index: usize,
    pub atomic_group_id: String,
    pub logical_start: usize,
    pub logical_count: usize,
    pub resources: Vec<VulkanDistributedSelectedResourceSourceRange>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct VulkanDistributedSelectedResourceSourceRange {
    pub resource_id: String,
    pub source_byte_count: usize,
    pub byte_offset: usize,
    pub byte_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanDistributedSelectedResourceSelectorIdentity {
    execution_scope: String,
    component_id: String,
    node_id: String,
    domain_id: String,
    selection_signal: String,
    resource_count: usize,
    selection_count_per_activation: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanDistributedSelectedResourceFragmentAccumulator {
    atomic_group_id: String,
    logical_start: usize,
    logical_count: usize,
    resources: BTreeMap<String, VulkanDistributedSelectedResourceSourceRange>,
}

impl VulkanDistributedSelectedResourceStorePlan {
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

    /// Expands whole-resource selectors into a stable addressability envelope
    /// for their existing execution participants.
    ///
    /// This does not change execution ownership. It gives every participant
    /// that already owns part of a whole-resource selector enough immutable
    /// addressing metadata to accept a later ownership move without remounting
    /// the model. Tensor-fragment selectors remain exact: changing their
    /// projections would change the compiled physical implementation rather
    /// than merely moving an independently executable resource.
    pub fn with_whole_resource_addressability_envelope(
        &self,
    ) -> Result<Self, VulkanDistributedPlanError> {
        let mut whole_resources = BTreeMap::<
            String,
            BTreeMap<usize, (String, usize)>,
        >::new();
        let mut whole_participants = BTreeMap::<String, BTreeSet<String>>::new();
        let mut fragmented_selectors = BTreeSet::new();
        for device in &self.devices {
            for selector in &device.selectors {
                let whole = !selector.owned_resource_indices.is_empty();
                let fragmented = !selector.fragmented_resources.is_empty();
                if whole == fragmented
                    || selector.owned_resource_indices.len() != selector.atomic_group_ids.len()
                    || selector.owned_resource_indices.len()
                        != selector.atomic_group_byte_counts.len()
                {
                    return Err(VulkanDistributedPlanError(format!(
                        "selected-resource store plan has invalid addressability for selector {:?} on {:?}",
                        selector.selector_id, device.device_id,
                    )));
                }
                if fragmented {
                    fragmented_selectors.insert(selector.selector_id.clone());
                    continue;
                }
                whole_participants
                    .entry(selector.selector_id.clone())
                    .or_default()
                    .insert(device.device_id.clone());
                let resources = whole_resources
                    .entry(selector.selector_id.clone())
                    .or_default();
                for ((resource_index, atomic_group_id), byte_count) in selector
                    .owned_resource_indices
                    .iter()
                    .copied()
                    .zip(selector.atomic_group_ids.iter().cloned())
                    .zip(selector.atomic_group_byte_counts.iter().copied())
                {
                    let identity = (atomic_group_id, byte_count);
                    if let Some(previous) = resources.insert(resource_index, identity.clone())
                        && previous != identity
                    {
                        return Err(VulkanDistributedPlanError(format!(
                            "selected-resource selector {:?} resource {resource_index} changes atomic identity between participants",
                            selector.selector_id,
                        )));
                    }
                }
            }
        }
        if fragmented_selectors
            .iter()
            .any(|selector_id| whole_resources.contains_key(selector_id))
        {
            return Err(VulkanDistributedPlanError(
                "selected-resource addressability cannot mix whole and tensor-fragment selectors"
                    .to_string(),
            ));
        }

        let mut device_selectors = self
            .devices
            .iter()
            .map(|device| (device.device_id.clone(), device.selectors.clone()))
            .collect::<BTreeMap<_, _>>();
        for (selector_id, participants) in whole_participants {
            let resources = whole_resources
                .get(&selector_id)
                .expect("whole selector participants have canonical resources");
            for device_id in participants {
                let selectors = device_selectors.get_mut(&device_id).ok_or_else(|| {
                    VulkanDistributedPlanError(format!(
                        "selected-resource addressability participant {device_id:?} is absent"
                    ))
                })?;
                let selector = selectors
                    .iter_mut()
                    .find(|selector| selector.selector_id == selector_id)
                    .ok_or_else(|| {
                        VulkanDistributedPlanError(format!(
                            "selected-resource addressability participant {device_id:?} omits selector {selector_id:?}"
                        ))
                    })?;
                if resources.len() != selector.resource_count
                    || resources.keys().copied().ne(0..selector.resource_count)
                {
                    return Err(VulkanDistributedPlanError(format!(
                        "selected-resource selector {selector_id:?} does not cover resources 0..{} across its participants",
                        selector.resource_count,
                    )));
                }
                selector.owned_resource_indices = resources.keys().copied().collect();
                selector.atomic_group_ids = resources
                    .values()
                    .map(|(group_id, _)| group_id.clone())
                    .collect();
                selector.atomic_group_byte_counts = resources
                    .values()
                    .map(|(_, byte_count)| *byte_count)
                    .collect();
            }
        }
        selected_resource_store_plan_from_device_selectors(
            device_selectors,
            self.selector_count,
        )
    }

    fn merged_for_alternatives(
        plans: &[VulkanDistributedSelectedResourceStorePlan],
    ) -> Result<Self, VulkanDistributedPlanError> {
        let mut selector_identities =
            BTreeMap::<String, VulkanDistributedSelectedResourceSelectorIdentity>::new();
        let mut canonical_resources = BTreeMap::<(String, usize), (String, usize)>::new();
        let mut resources_by_device_selector =
            BTreeMap::<(String, String), BTreeMap<usize, (String, usize)>>::new();
        let mut fragments_by_device_selector = BTreeMap::<
            (String, String),
            BTreeMap<usize, VulkanDistributedSelectedResourceFragmentOwnership>,
        >::new();
        for plan in plans {
            for device in &plan.devices {
                for selector in &device.selectors {
                    let whole = !selector.owned_resource_indices.is_empty();
                    let fragmented = !selector.fragmented_resources.is_empty();
                    if whole == fragmented
                        || selector.owned_resource_indices.len() != selector.atomic_group_ids.len()
                        || selector.owned_resource_indices.len()
                            != selector.atomic_group_byte_counts.len()
                        || selector
                            .owned_resource_indices
                            .windows(2)
                            .any(|pair| pair[0] >= pair[1])
                        || selector
                            .owned_resource_indices
                            .iter()
                            .any(|index| *index >= selector.resource_count)
                    {
                        return Err(VulkanDistributedPlanError(
                            "selected-resource alternative contains invalid ownership".to_string(),
                        ));
                    }
                    if fragmented
                        && (selector
                            .fragmented_resources
                            .windows(2)
                            .any(|pair| pair[0].resource_index >= pair[1].resource_index)
                            || selector.fragmented_resources.iter().any(|fragment| {
                                fragment.resource_index >= selector.resource_count
                                    || fragment.logical_count == 0
                                    || fragment.resources.is_empty()
                                    || fragment
                                        .resources
                                        .windows(2)
                                        .any(|pair| pair[0].resource_id >= pair[1].resource_id)
                            }))
                    {
                        return Err(VulkanDistributedPlanError(
                            "selected-resource alternative contains invalid fragment ownership"
                                .to_string(),
                        ));
                    }
                    let identity = VulkanDistributedSelectedResourceSelectorIdentity {
                        execution_scope: selector.execution_scope.clone(),
                        component_id: selector.component_id.clone(),
                        node_id: selector.node_id.clone(),
                        domain_id: selector.domain_id.clone(),
                        selection_signal: selector.selection_signal.clone(),
                        resource_count: selector.resource_count,
                        selection_count_per_activation: selector.selection_count_per_activation,
                    };
                    if let Some(existing) =
                        selector_identities.insert(selector.selector_id.clone(), identity.clone())
                        && existing != identity
                    {
                        return Err(VulkanDistributedPlanError(format!(
                            "selected-resource selector {:?} changes identity between execution alternatives",
                            selector.selector_id,
                        )));
                    }
                    if whole {
                        let device_resources = resources_by_device_selector
                            .entry((device.device_id.clone(), selector.selector_id.clone()))
                            .or_default();
                        for ((resource_index, group_id), byte_count) in selector
                            .owned_resource_indices
                            .iter()
                            .copied()
                            .zip(selector.atomic_group_ids.iter().cloned())
                            .zip(selector.atomic_group_byte_counts.iter().copied())
                        {
                            let resource = (group_id, byte_count);
                            if let Some(existing) = canonical_resources.insert(
                                (selector.selector_id.clone(), resource_index),
                                resource.clone(),
                            ) && existing != resource
                            {
                                return Err(VulkanDistributedPlanError(format!(
                                    "selected-resource selector {:?} resource {resource_index} changes atomic identity between execution alternatives",
                                    selector.selector_id,
                                )));
                            }
                            if let Some(existing) =
                                device_resources.insert(resource_index, resource.clone())
                                && existing != resource
                            {
                                return Err(VulkanDistributedPlanError(format!(
                                    "selected-resource selector {:?} resource {resource_index} conflicts on device {:?}",
                                    selector.selector_id, device.device_id,
                                )));
                            }
                        }
                    }
                    if fragmented {
                        let device_fragments = fragments_by_device_selector
                            .entry((device.device_id.clone(), selector.selector_id.clone()))
                            .or_default();
                        for fragment in &selector.fragmented_resources {
                            if let Some(existing) =
                                device_fragments.insert(fragment.resource_index, fragment.clone())
                                && existing != *fragment
                            {
                                return Err(VulkanDistributedPlanError(format!(
                                    "selected-resource selector {:?} resource {} changes fragment ownership on device {:?} between execution alternatives",
                                    selector.selector_id,
                                    fragment.resource_index,
                                    device.device_id,
                                )));
                            }
                        }
                    }
                }
            }
        }
        let mut device_selectors =
            BTreeMap::<String, Vec<VulkanDistributedSelectedResourceOwnership>>::new();
        for ((device_id, selector_id), resources) in resources_by_device_selector {
            let identity = selector_identities
                .get(&selector_id)
                .expect("alternative resource ownership has a selector identity");
            let mut owned_resource_indices = Vec::with_capacity(resources.len());
            let mut atomic_group_ids = Vec::with_capacity(resources.len());
            let mut atomic_group_byte_counts = Vec::with_capacity(resources.len());
            for (resource_index, (group_id, byte_count)) in resources {
                owned_resource_indices.push(resource_index);
                atomic_group_ids.push(group_id);
                atomic_group_byte_counts.push(byte_count);
            }
            device_selectors.entry(device_id).or_default().push(
                VulkanDistributedSelectedResourceOwnership {
                    execution_scope: identity.execution_scope.clone(),
                    selector_id,
                    component_id: identity.component_id.clone(),
                    node_id: identity.node_id.clone(),
                    domain_id: identity.domain_id.clone(),
                    selection_signal: identity.selection_signal.clone(),
                    resource_count: identity.resource_count,
                    selection_count_per_activation: identity.selection_count_per_activation,
                    owned_resource_indices,
                    atomic_group_ids,
                    atomic_group_byte_counts,
                    fragmented_resources: Vec::new(),
                },
            );
        }
        for ((device_id, selector_id), fragments) in fragments_by_device_selector {
            let identity = selector_identities
                .get(&selector_id)
                .expect("alternative fragment ownership has a selector identity");
            if device_selectors.get(&device_id).is_some_and(|selectors| {
                selectors
                    .iter()
                    .any(|selector| selector.selector_id == selector_id)
            }) {
                return Err(VulkanDistributedPlanError(format!(
                    "selected-resource selector {selector_id:?} mixes whole-resource and fragment alternatives on device {device_id:?}",
                )));
            }
            device_selectors.entry(device_id).or_default().push(
                VulkanDistributedSelectedResourceOwnership {
                    execution_scope: identity.execution_scope.clone(),
                    selector_id,
                    component_id: identity.component_id.clone(),
                    node_id: identity.node_id.clone(),
                    domain_id: identity.domain_id.clone(),
                    selection_signal: identity.selection_signal.clone(),
                    resource_count: identity.resource_count,
                    selection_count_per_activation: identity.selection_count_per_activation,
                    owned_resource_indices: Vec::new(),
                    atomic_group_ids: Vec::new(),
                    atomic_group_byte_counts: Vec::new(),
                    fragmented_resources: fragments.into_values().collect(),
                },
            );
        }
        selected_resource_store_plan_from_device_selectors(
            device_selectors,
            selector_identities.len(),
        )
    }

    pub fn from_execution_plan(
        plan: &VulkanDistributedExecutionPlan,
    ) -> Result<Self, VulkanDistributedPlanError> {
        Self::from_execution_plan_for_resources(plan, None)
    }

    pub(crate) fn from_execution_plan_for_selected_resources(
        plan: &VulkanDistributedExecutionPlan,
        selected_resources: &BTreeMap<String, BTreeSet<usize>>,
    ) -> Result<Self, VulkanDistributedPlanError> {
        if selected_resources.is_empty()
            || selected_resources.values().any(BTreeSet::is_empty)
        {
            return Err(VulkanDistributedPlanError(
                "selected-resource store projection requires a nonempty exact resource set"
                    .to_string(),
            ));
        }
        Self::from_execution_plan_for_resources(plan, Some(selected_resources))
    }

    fn from_execution_plan_for_resources(
        plan: &VulkanDistributedExecutionPlan,
        selected_resources: Option<&BTreeMap<String, BTreeSet<usize>>>,
    ) -> Result<Self, VulkanDistributedPlanError> {
        let execution_devices = plan
            .device_ids
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let mut identities =
            BTreeMap::<String, VulkanDistributedSelectedResourceSelectorIdentity>::new();
        let mut cohort_identities =
            BTreeMap::<(String, usize), (String, usize, Vec<String>)>::new();
        let mut group_devices = BTreeMap::<(String, String), String>::new();
        let mut required_resources_by_selector = BTreeMap::<String, BTreeSet<usize>>::new();
        let mut resources_by_device_selector = BTreeMap::<(String, String), BTreeSet<usize>>::new();
        let mut fragments_by_device_selector_resource = BTreeMap::<
            (String, String, usize),
            VulkanDistributedSelectedResourceFragmentAccumulator,
        >::new();

        for dispatch in &plan.dispatches {
            for partition in &dispatch.selected_resource_partitions {
                validate_selected_resource_partition(dispatch, partition)?;
                let required_resources = match selected_resources {
                    Some(resources) => resources
                        .get(&partition.selector_id)
                        .cloned()
                        .ok_or_else(|| {
                            VulkanDistributedPlanError(format!(
                                "selected resource selector {:?} has no exact required resource set",
                                partition.selector_id,
                            ))
                        })?,
                    None => (0..partition.resource_count).collect(),
                };
                if required_resources.is_empty()
                    || required_resources
                        .iter()
                        .any(|resource_index| *resource_index >= partition.resource_count)
                {
                    return Err(VulkanDistributedPlanError(format!(
                        "selected resource selector {:?} has an invalid required resource set",
                        partition.selector_id,
                    )));
                }
                if let Some(existing) = required_resources_by_selector
                    .insert(partition.selector_id.clone(), required_resources.clone())
                    && existing != required_resources
                {
                    return Err(VulkanDistributedPlanError(format!(
                        "selected resource selector {:?} changes its required resource projection between distributed dispatches",
                        partition.selector_id,
                    )));
                }
                let identity = VulkanDistributedSelectedResourceSelectorIdentity {
                    execution_scope: partition.execution_scope.clone(),
                    component_id: dispatch.component_id.clone(),
                    node_id: partition.node_id.clone(),
                    domain_id: partition.domain_id.clone(),
                    selection_signal: partition.selection_signal.clone(),
                    resource_count: partition.resource_count,
                    selection_count_per_activation: partition.selection_count_per_activation,
                };
                if let Some(existing) =
                    identities.insert(partition.selector_id.clone(), identity.clone())
                    && existing != identity
                {
                    return Err(VulkanDistributedPlanError(format!(
                        "selected resource selector {:?} changes identity between distributed dispatches",
                        partition.selector_id
                    )));
                }
                for resource_index in 0..partition.resource_count {
                    let cohort_identity = (
                        partition.atomic_group_ids[resource_index].clone(),
                        partition.atomic_group_byte_counts[resource_index],
                        partition.atomic_group_resource_ids[resource_index].clone(),
                    );
                    if let Some(existing) = cohort_identities.insert(
                        (partition.selector_id.clone(), resource_index),
                        cohort_identity.clone(),
                    ) && existing != cohort_identity
                    {
                        return Err(VulkanDistributedPlanError(format!(
                            "selected resource selector {:?} resource {resource_index} changes atomic cohort identity between distributed dispatches",
                            partition.selector_id,
                        )));
                    }
                }

                if !partition.parameter_partitions.is_empty() {
                    collect_fragmented_selected_resource_partition(
                        dispatch,
                        partition,
                        &execution_devices,
                        &required_resources,
                        &mut fragments_by_device_selector_resource,
                    )?;
                    continue;
                }

                let mut coverage = vec![0u8; partition.resource_count];
                for shard in &dispatch.shards {
                    if !execution_devices.contains(shard.device_id.as_str()) {
                        return Err(VulkanDistributedPlanError(format!(
                            "selected resource shard for {}.{} uses device {:?} outside the execution pool",
                            dispatch.component_id, dispatch.node_id, shard.device_id
                        )));
                    }
                    let resource_indices = shard
                        .selected_resource_indices
                        .get(&partition.selector_id)
                        .ok_or_else(|| {
                            VulkanDistributedPlanError(format!(
                                "selected resource shard on {:?} has no exact ownership for selector {:?}",
                                shard.device_id, partition.selector_id,
                            ))
                        })?;
                    if resource_indices.is_empty()
                        || resource_indices.windows(2).any(|pair| pair[0] >= pair[1])
                        || resource_indices
                            .iter()
                            .any(|index| *index >= partition.resource_count)
                    {
                        return Err(VulkanDistributedPlanError(format!(
                            "selected resource shard on {:?} has invalid ownership for selector {:?}",
                            shard.device_id, partition.selector_id,
                        )));
                    }
                    let owned = resources_by_device_selector
                        .entry((shard.device_id.clone(), partition.selector_id.clone()))
                        .or_default();
                    for resource_index in resource_indices {
                        coverage[*resource_index] =
                            coverage[*resource_index].checked_add(1).ok_or_else(|| {
                                VulkanDistributedPlanError(
                                    "selected resource coverage count overflowed".to_string(),
                                )
                            })?;
                        let group_id = partition.atomic_group_ids[*resource_index].clone();
                        let group_key = (partition.selector_id.clone(), group_id.clone());
                        if let Some(existing_device) =
                            group_devices.insert(group_key, shard.device_id.clone())
                            && existing_device != shard.device_id
                        {
                            return Err(VulkanDistributedPlanError(format!(
                                "selected atomic group {group_id:?} moves from device {existing_device:?} to {:?} between distributed dispatches",
                                shard.device_id
                            )));
                        }
                        owned.insert(*resource_index);
                    }
                }
                if coverage.iter().enumerate().any(|(resource_index, count)| {
                    *count
                        != if required_resources.contains(&resource_index) {
                            1
                        } else {
                            0
                        }
                }) {
                    return Err(VulkanDistributedPlanError(format!(
                        "selected resource selector {:?} is not partitioned exactly once across its shards for every required resource",
                        partition.selector_id,
                    )));
                }
            }
        }
        if let Some(selected_resources) = selected_resources
            && selected_resources.keys().collect::<BTreeSet<_>>()
                != required_resources_by_selector.keys().collect::<BTreeSet<_>>()
        {
            return Err(VulkanDistributedPlanError(
                "selected-resource store projection does not exactly cover the plan selectors"
                    .to_string(),
            ));
        }

        let mut device_selectors =
            BTreeMap::<String, Vec<VulkanDistributedSelectedResourceOwnership>>::new();
        for ((device_id, selector_id), indices) in resources_by_device_selector {
            let identity = identities.get(&selector_id).expect(
                "selected resource ownership was created from a validated selector identity",
            );
            let owned_resource_indices = indices.into_iter().collect::<Vec<_>>();
            let atomic_group_ids = owned_resource_indices
                .iter()
                .map(|index| {
                    cohort_identities
                        .get(&(selector_id.clone(), *index))
                        .expect("whole-resource ownership has an atomic cohort identity")
                        .0
                        .clone()
                })
                .collect::<Vec<_>>();
            let atomic_group_byte_counts = owned_resource_indices
                .iter()
                .map(|index| {
                    cohort_identities
                        .get(&(selector_id.clone(), *index))
                        .expect("whole-resource ownership has an atomic cohort identity")
                        .1
                })
                .collect::<Vec<_>>();
            device_selectors.entry(device_id).or_default().push(
                VulkanDistributedSelectedResourceOwnership {
                    execution_scope: identity.execution_scope.clone(),
                    selector_id,
                    component_id: identity.component_id.clone(),
                    node_id: identity.node_id.clone(),
                    domain_id: identity.domain_id.clone(),
                    selection_signal: identity.selection_signal.clone(),
                    resource_count: identity.resource_count,
                    selection_count_per_activation: identity.selection_count_per_activation,
                    owned_resource_indices,
                    atomic_group_ids,
                    atomic_group_byte_counts,
                    fragmented_resources: Vec::new(),
                },
            );
        }

        let mut fragments_by_device_selector = BTreeMap::<
            (String, String),
            Vec<VulkanDistributedSelectedResourceFragmentOwnership>,
        >::new();
        for ((device_id, selector_id, resource_index), fragment) in
            fragments_by_device_selector_resource
        {
            let expected_resource_ids = cohort_identities
                .get(&(selector_id.clone(), resource_index))
                .expect("selected resource fragment has an atomic cohort identity")
                .2
                .iter()
                .map(String::as_str)
                .collect::<BTreeSet<_>>();
            let actual_resource_ids = fragment
                .resources
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>();
            if actual_resource_ids != expected_resource_ids {
                return Err(VulkanDistributedPlanError(format!(
                    "fragmented selector {selector_id:?} resource {resource_index} does not project every member of atomic group {:?}",
                    fragment.atomic_group_id,
                )));
            }
            fragments_by_device_selector
                .entry((device_id, selector_id))
                .or_default()
                .push(VulkanDistributedSelectedResourceFragmentOwnership {
                    resource_index,
                    atomic_group_id: fragment.atomic_group_id,
                    logical_start: fragment.logical_start,
                    logical_count: fragment.logical_count,
                    resources: fragment.resources.into_values().collect(),
                });
        }
        for ((device_id, selector_id), mut fragmented_resources) in
            fragments_by_device_selector
        {
            let identity = identities.get(&selector_id).expect(
                "selected resource fragments were created from a validated selector identity",
            );
            let required_resources = required_resources_by_selector
                .get(&selector_id)
                .expect("selected resource fragments have a required resource projection");
            fragmented_resources.sort_by_key(|fragment| fragment.resource_index);
            if fragmented_resources
                .iter()
                .map(|fragment| fragment.resource_index)
                .ne(required_resources.iter().copied())
            {
                return Err(VulkanDistributedPlanError(format!(
                    "fragmented selector {selector_id:?} on device {device_id:?} does not cover every required logical resource",
                )));
            }
            if device_selectors.get(&device_id).is_some_and(|selectors| {
                selectors
                    .iter()
                    .any(|selector| selector.selector_id == selector_id)
            }) {
                return Err(VulkanDistributedPlanError(format!(
                    "selected resource selector {selector_id:?} mixes whole-resource and fragmented ownership on device {device_id:?}",
                )));
            }
            device_selectors.entry(device_id).or_default().push(
                VulkanDistributedSelectedResourceOwnership {
                    execution_scope: identity.execution_scope.clone(),
                    selector_id,
                    component_id: identity.component_id.clone(),
                    node_id: identity.node_id.clone(),
                    domain_id: identity.domain_id.clone(),
                    selection_signal: identity.selection_signal.clone(),
                    resource_count: identity.resource_count,
                    selection_count_per_activation: identity.selection_count_per_activation,
                    owned_resource_indices: Vec::new(),
                    atomic_group_ids: Vec::new(),
                    atomic_group_byte_counts: Vec::new(),
                    fragmented_resources,
                },
            );
        }

        selected_resource_store_plan_from_device_selectors(device_selectors, identities.len())
    }

    pub fn device(&self, device_id: &str) -> Option<&VulkanDistributedSelectedResourceDevicePlan> {
        self.devices
            .iter()
            .find(|device| device.device_id == device_id)
    }
}

fn collect_fragmented_selected_resource_partition(
    dispatch: &VulkanDistributedDispatchPlan,
    partition: &VulkanDistributedSelectedResourcePartitionPlan,
    execution_devices: &BTreeSet<&str>,
    required_resources: &BTreeSet<usize>,
    accumulated: &mut BTreeMap<
        (String, String, usize),
        VulkanDistributedSelectedResourceFragmentAccumulator,
    >,
) -> Result<(), VulkanDistributedPlanError> {
    let partitioned_slots = partition
        .parameter_partitions
        .iter()
        .map(|parameter| parameter.parameter_slot)
        .collect::<BTreeSet<_>>();
    let mut logical_ranges = vec![Vec::<(usize, usize)>::new(); partition.resource_count];
    let mut parameter_ranges = vec![
        vec![Vec::<(usize, usize)>::new(); partition.parameters_per_resource];
        partition.resource_count
    ];

    for shard in &dispatch.shards {
        if !execution_devices.contains(shard.device_id.as_str()) {
            return Err(VulkanDistributedPlanError(format!(
                "selected resource shard for {}.{} uses device {:?} outside the execution pool",
                dispatch.component_id, dispatch.node_id, shard.device_id,
            )));
        }
        if shard
            .selected_resource_indices
            .contains_key(&partition.selector_id)
        {
            return Err(VulkanDistributedPlanError(format!(
                "fragmented selector {:?} also declares whole-resource ownership on {:?}",
                partition.selector_id, shard.device_id,
            )));
        }
        let fragments = shard
            .selected_resource_fragments
            .get(&partition.selector_id)
            .ok_or_else(|| {
                VulkanDistributedPlanError(format!(
                    "selected resource shard on {:?} has no exact fragments for selector {:?}",
                    shard.device_id, partition.selector_id,
                ))
            })?;
        if fragments.len() != required_resources.len()
            || fragments
                .iter()
                .map(|fragment| fragment.resource_index)
                .ne(required_resources.iter().copied())
        {
            return Err(VulkanDistributedPlanError(format!(
                "selected resource shard on {:?} has incomplete fragments for selector {:?}",
                shard.device_id, partition.selector_id,
            )));
        }
        for fragment in fragments {
            if fragment.atomic_group_id
                != partition.atomic_group_ids[fragment.resource_index]
                || fragment.logical_start != shard.row_start
                || fragment.logical_count != shard.row_count
                || fragment.logical_count == 0
                || fragment.parameters.len() != partition.parameters_per_resource
                || fragment
                    .parameters
                    .iter()
                    .map(|parameter| parameter.parameter_slot)
                    .ne(0..partition.parameters_per_resource)
            {
                return Err(VulkanDistributedPlanError(format!(
                    "selected resource shard on {:?} has invalid fragment geometry for selector {:?} resource {}",
                    shard.device_id, partition.selector_id, fragment.resource_index,
                )));
            }
            logical_ranges[fragment.resource_index]
                .push((fragment.logical_start, fragment.logical_count));
            let key = (
                shard.device_id.clone(),
                partition.selector_id.clone(),
                fragment.resource_index,
            );
            let entry = accumulated.entry(key).or_insert_with(|| {
                VulkanDistributedSelectedResourceFragmentAccumulator {
                    atomic_group_id: fragment.atomic_group_id.clone(),
                    logical_start: fragment.logical_start,
                    logical_count: fragment.logical_count,
                    resources: BTreeMap::new(),
                }
            });
            if entry.atomic_group_id != fragment.atomic_group_id
                || entry.logical_start != fragment.logical_start
                || entry.logical_count != fragment.logical_count
            {
                return Err(VulkanDistributedPlanError(format!(
                    "fragmented selector {:?} resource {} changes physical range between connected dispatches on {:?}",
                    partition.selector_id, fragment.resource_index, shard.device_id,
                )));
            }
            let mut fragment_resource_ids = BTreeSet::new();
            for parameter in &fragment.parameters {
                let parameter_slot = parameter.parameter_slot;
                if parameter.resource_id
                    != partition.parameter_resource_ids[fragment.resource_index][parameter_slot]
                    || parameter.resource_byte_count
                        != partition.parameter_resource_byte_counts[fragment.resource_index]
                            [parameter_slot]
                    || parameter.byte_count == 0
                    || parameter
                        .byte_offset
                        .checked_add(parameter.byte_count)
                        .is_none_or(|end| end > parameter.resource_byte_count)
                    || !fragment_resource_ids.insert(parameter.resource_id.as_str())
                {
                    return Err(VulkanDistributedPlanError(format!(
                        "fragmented selector {:?} resource {} has an invalid parameter slot {parameter_slot}",
                        partition.selector_id, fragment.resource_index,
                    )));
                }
                parameter_ranges[fragment.resource_index][parameter_slot]
                    .push((parameter.byte_offset, parameter.byte_count));
                let range = VulkanDistributedSelectedResourceSourceRange {
                    resource_id: parameter.resource_id.clone(),
                    source_byte_count: parameter.resource_byte_count,
                    byte_offset: parameter.byte_offset,
                    byte_count: parameter.byte_count,
                };
                if let Some(previous) =
                    entry.resources.insert(parameter.resource_id.clone(), range.clone())
                    && previous != range
                {
                    return Err(VulkanDistributedPlanError(format!(
                        "fragmented selector {:?} resource {} changes source range for {:?} between connected dispatches",
                        partition.selector_id, fragment.resource_index, parameter.resource_id,
                    )));
                }
            }
        }
    }

    for resource_index in required_resources {
        let resource_index = *resource_index;
        logical_ranges[resource_index].sort_unstable();
        validate_contiguous_selected_resource_ranges(
            &logical_ranges[resource_index],
            dispatch.output_rows,
            &format!(
                "selector {:?} resource {resource_index} logical",
                partition.selector_id
            ),
        )?;
        for parameter_slot in 0..partition.parameters_per_resource {
            let ranges = &mut parameter_ranges[resource_index][parameter_slot];
            ranges.sort_unstable();
            let expected_bytes =
                partition.parameter_resource_byte_counts[resource_index][parameter_slot];
            if partitioned_slots.contains(&parameter_slot) {
                validate_contiguous_selected_resource_ranges(
                    ranges,
                    expected_bytes,
                    &format!(
                        "selector {:?} resource {resource_index} parameter {parameter_slot}",
                        partition.selector_id
                    ),
                )?;
            } else if ranges.len() != dispatch.shards.len()
                || ranges
                    .iter()
                    .any(|range| *range != (0, expected_bytes))
            {
                return Err(VulkanDistributedPlanError(format!(
                    "selector {:?} resource {resource_index} parameter {parameter_slot} is neither exactly partitioned nor fully replicated",
                    partition.selector_id,
                )));
            }
        }
    }
    Ok(())
}

fn validate_contiguous_selected_resource_ranges(
    ranges: &[(usize, usize)],
    expected_extent: usize,
    label: &str,
) -> Result<(), VulkanDistributedPlanError> {
    let mut frontier = 0usize;
    for (start, count) in ranges {
        if *start != frontier || *count == 0 {
            return Err(VulkanDistributedPlanError(format!(
                "fragmented selected-resource {label} ranges are not contiguous",
            )));
        }
        frontier = frontier.checked_add(*count).ok_or_else(|| {
            VulkanDistributedPlanError(
                "fragmented selected-resource range coverage overflowed".to_string(),
            )
        })?;
    }
    if frontier != expected_extent {
        return Err(VulkanDistributedPlanError(format!(
            "fragmented selected-resource {label} ranges cover {frontier}, expected {expected_extent}",
        )));
    }
    Ok(())
}

fn selected_resource_store_plan_from_device_selectors(
    device_selectors: BTreeMap<String, Vec<VulkanDistributedSelectedResourceOwnership>>,
    selector_count: usize,
) -> Result<VulkanDistributedSelectedResourceStorePlan, VulkanDistributedPlanError> {
    let tensor_sharded_residency_cohorts =
        tensor_sharded_selected_resource_residency_cohorts(&device_selectors)?;
    let mut devices = Vec::with_capacity(device_selectors.len());
    let mut global_groups = BTreeMap::<String, usize>::new();
    let mut selector_placement_count = 0usize;
    let mut total_addressable_bytes = 0usize;
    for (device_id, mut selectors) in device_selectors {
        selectors.sort_by(|left, right| left.selector_id.cmp(&right.selector_id));
        selector_placement_count = selector_placement_count
            .checked_add(selectors.len())
            .ok_or_else(|| {
                VulkanDistributedPlanError(
                    "selected resource selector count overflowed".to_string(),
                )
            })?;
        let mut device_groups = BTreeMap::<String, usize>::new();
        let mut maximum_load_wave_bytes = 0usize;
        for selector in &selectors {
            let mut selector_group_bytes = selector
                .atomic_group_ids
                .iter()
                .cloned()
                .zip(selector.atomic_group_byte_counts.iter().copied())
                .collect::<Vec<_>>();
            selector_group_bytes.extend(
                selector
                    .fragmented_resources
                    .iter()
                    .map(|fragment| {
                        let bytes = fragment.resources.iter().try_fold(
                            0usize,
                            |total, resource| total.checked_add(resource.byte_count),
                        ).ok_or_else(|| {
                            VulkanDistributedPlanError(
                                "selected resource fragment byte count overflowed".to_string(),
                            )
                        })?;
                        Ok((
                            format!(
                                "{}@{}+{}",
                                fragment.atomic_group_id,
                                fragment.logical_start,
                                fragment.logical_count,
                            ),
                            bytes,
                        ))
                    })
                    .collect::<Result<Vec<_>, VulkanDistributedPlanError>>()?,
            );
            selector_group_bytes.sort_by_key(|(_, bytes)| std::cmp::Reverse(*bytes));
            let selection_count = selector
                .selection_count_per_activation
                .min(selector_group_bytes.len());
            let load_wave_bytes = selector_group_bytes
                .iter()
                .take(selection_count)
                .try_fold(0usize, |total, (_, bytes)| total.checked_add(*bytes))
                .ok_or_else(|| {
                    VulkanDistributedPlanError(
                        "selected resource load-wave capacity overflowed".to_string(),
                    )
                })?;
            maximum_load_wave_bytes = maximum_load_wave_bytes.max(load_wave_bytes);
            for (group_id, bytes) in selector_group_bytes {
                insert_group_bytes(&mut device_groups, &group_id, bytes)?;
                insert_group_bytes(&mut global_groups, &group_id, bytes)?;
            }
        }
        let device_addressable_bytes = device_groups
            .values()
            .try_fold(0usize, |total, bytes| total.checked_add(*bytes))
            .ok_or_else(|| {
                VulkanDistributedPlanError(
                    "selected resource device capacity overflowed".to_string(),
                )
            })?;
        total_addressable_bytes = total_addressable_bytes
            .checked_add(device_addressable_bytes)
            .ok_or_else(|| {
                VulkanDistributedPlanError(
                    "selected resource total capacity overflowed".to_string(),
                )
            })?;
        devices.push(VulkanDistributedSelectedResourceDevicePlan {
            device_id,
            unique_atomic_group_count: device_groups.len(),
            maximum_atomic_group_bytes: device_groups.values().copied().max().unwrap_or(0),
            maximum_load_wave_bytes,
            total_addressable_bytes: device_addressable_bytes,
            selectors,
        });
    }
    Ok(VulkanDistributedSelectedResourceStorePlan {
        tensor_sharded_residency_cohorts,
        device_count: devices.len(),
        selector_count,
        selector_placement_count,
        unique_atomic_group_count: global_groups.len(),
        total_addressable_bytes,
        devices,
    })
}

fn tensor_sharded_selected_resource_residency_cohorts(
    device_selectors: &BTreeMap<String, Vec<VulkanDistributedSelectedResourceOwnership>>,
) -> Result<Vec<VulkanDistributedSelectedResourceResidencyCohortPlan>, VulkanDistributedPlanError>
{
    let mut members = BTreeMap::<
        (String, usize, String),
        BTreeMap<String, VulkanDistributedSelectedResourceResidencyCohortMemberPlan>,
    >::new();
    for (device_id, selectors) in device_selectors {
        for selector in selectors {
            for fragment in &selector.fragmented_resources {
                let key = (
                    selector.selector_id.clone(),
                    fragment.resource_index,
                    fragment.atomic_group_id.clone(),
                );
                let member = VulkanDistributedSelectedResourceResidencyCohortMemberPlan {
                    device_id: device_id.clone(),
                    logical_start: fragment.logical_start,
                    logical_count: fragment.logical_count,
                };
                if let Some(previous) = members
                    .entry(key)
                    .or_default()
                    .insert(device_id.clone(), member.clone())
                    && previous != member
                {
                    return Err(VulkanDistributedPlanError(format!(
                        "tensor-sharded selected resource {:?} index {} changes its fragment on device {device_id:?}",
                        selector.selector_id, fragment.resource_index,
                    )));
                }
            }
        }
    }

    members
        .into_iter()
        .map(
            |((selector_id, resource_index, atomic_group_id), by_device)| {
                let mut cohort_members = by_device.into_values().collect::<Vec<_>>();
                cohort_members.sort_by(|left, right| {
                    (left.logical_start, left.device_id.as_str())
                        .cmp(&(right.logical_start, right.device_id.as_str()))
                });
                let mut frontier = 0usize;
                for member in &cohort_members {
                    if member.logical_count == 0 || member.logical_start != frontier {
                        return Err(VulkanDistributedPlanError(format!(
                            "tensor-sharded selected resource {selector_id:?} index {resource_index} does not form one contiguous residency cohort",
                        )));
                    }
                    frontier = frontier.checked_add(member.logical_count).ok_or_else(|| {
                        VulkanDistributedPlanError(
                            "tensor-sharded residency cohort logical extent overflowed".to_string(),
                        )
                    })?;
                }
                if cohort_members.is_empty() {
                    return Err(VulkanDistributedPlanError(
                        "tensor-sharded residency cohort has no physical members".to_string(),
                    ));
                }
                Ok(VulkanDistributedSelectedResourceResidencyCohortPlan {
                    selector_id,
                    resource_index,
                    atomic_group_id,
                    members: cohort_members,
                })
            },
        )
        .collect()
}

fn validate_selected_resource_partition(
    dispatch: &VulkanDistributedDispatchPlan,
    partition: &VulkanDistributedSelectedResourcePartitionPlan,
) -> Result<(), VulkanDistributedPlanError> {
    if partition.execution_scope.trim().is_empty()
        || partition.selector_id.trim().is_empty()
        || partition.selection_signal.trim().is_empty()
        || partition.resource_count == 0
        || partition.parameters_per_resource == 0
        || partition.parameter_partitions.windows(2).any(|pair| {
            pair[0].parameter_slot >= pair[1].parameter_slot
        })
        || partition.parameter_partitions.iter().any(|parameter| {
            parameter.parameter_slot >= partition.parameters_per_resource
                || parameter.alignment_elements == 0
                || parameter.logical_elements_per_index == 0
        })
        || partition.selection_count_per_activation == 0
        || partition.selection_count_per_activation > partition.resource_count
        || partition.resource_operation_class_ids.len() != partition.resource_count
        || partition
            .resource_operation_class_ids
            .iter()
            .any(|class_id| !valid_selected_resource_execution_class_id(class_id))
        || partition.atomic_group_ids.len() != partition.resource_count
        || partition.atomic_group_byte_counts.len() != partition.resource_count
        || partition.atomic_group_resource_ids.len() != partition.resource_count
        || partition.parameter_resource_ids.len() != partition.resource_count
        || partition.parameter_resource_byte_counts.len() != partition.resource_count
        || partition
            .atomic_group_ids
            .iter()
            .any(|id| id.trim().is_empty())
        || partition
            .atomic_group_byte_counts
            .iter()
            .any(|bytes| *bytes == 0)
        || (0..partition.resource_count).any(|resource_index| {
            partition.atomic_group_resource_ids[resource_index].is_empty()
                || partition.parameter_resource_ids[resource_index].len()
                    != partition.parameters_per_resource
                || partition.parameter_resource_byte_counts[resource_index].len()
                    != partition.parameters_per_resource
                || partition.parameter_resource_ids[resource_index]
                    .iter()
                    .any(|resource_id| resource_id.trim().is_empty())
                || partition.parameter_resource_byte_counts[resource_index]
                    .iter()
                    .any(|bytes| *bytes == 0)
                || partition.parameter_resource_ids[resource_index]
                    .iter()
                    .any(|resource_id| {
                        !partition.atomic_group_resource_ids[resource_index]
                            .contains(resource_id)
                    })
        })
    {
        return Err(VulkanDistributedPlanError(format!(
            "distributed dispatch {}.{} has an invalid selected resource partition",
            dispatch.component_id, dispatch.node_id
        )));
    }
    Ok(())
}

fn insert_group_bytes(
    groups: &mut BTreeMap<String, usize>,
    group_id: &str,
    bytes: usize,
) -> Result<(), VulkanDistributedPlanError> {
    if let Some(previous) = groups.insert(group_id.to_string(), bytes)
        && previous != bytes
    {
        return Err(VulkanDistributedPlanError(format!(
            "selected atomic group {group_id:?} has conflicting byte counts {previous} and {bytes}",
        )));
    }
    Ok(())
}
