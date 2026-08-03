#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPhysicalResidencySchedule {
    pub execution_scope: String,
    pub checkpoints: Vec<VulkanPhysicalResidencyCheckpoint>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPhysicalResidencyCheckpoint {
    pub id: String,
    pub execution_scope: String,
    pub component_id: String,
    pub selector_ids: Vec<String>,
    pub selection_dispatch_index: usize,
    pub selected_computation_dispatch_indices: Vec<usize>,
    pub selected_result_continuation_dispatch_index: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct VulkanSelectedResourceIndex {
    pub selector_id: String,
    pub resource_index: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VulkanPhysicalResidencyResponsibility {
    Selection,
    Availability,
    SelectedComputation,
    SelectedResultContinuation,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPhysicalResidencyTraceEntry {
    pub checkpoint_id: String,
    pub responsibility: VulkanPhysicalResidencyResponsibility,
    pub dispatch_indices: Vec<usize>,
    pub selected_group_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanPhysicalResidencyActivationStatus {
    Paused {
        checkpoint_id: String,
        missing_group_ids: Vec<String>,
        resume_at: VulkanPhysicalResidencyResponsibility,
    },
    Completed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPhysicalResidencyActivation {
    checkpoint: VulkanPhysicalResidencyCheckpoint,
    selected_group_ids: Vec<String>,
    next_responsibility: VulkanPhysicalResidencyResponsibility,
    blocked_missing_group_ids: Vec<String>,
    completed: bool,
    trace: Vec<VulkanPhysicalResidencyTraceEntry>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanPhysicalResidencyDispatch {
    dispatch_index: usize,
    component_id: String,
    node_index: usize,
    node_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPhysicalResidencyCheckpointError(String);

impl VulkanPhysicalResidencyCheckpointError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl Display for VulkanPhysicalResidencyCheckpointError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for VulkanPhysicalResidencyCheckpointError {}

impl VulkanPhysicalResidencySchedule {
    pub fn from_prepared_dispatch_plan(
        contract: &CompiledResourceResidencyContract,
        execution_scope: impl Into<String>,
        prepared: &VulkanPreparedDispatchPlan,
    ) -> Result<Self, VulkanPhysicalResidencyCheckpointError> {
        let dispatches = prepared
            .dispatches
            .iter()
            .map(|dispatch| VulkanPhysicalResidencyDispatch {
                dispatch_index: dispatch.dispatch_index,
                component_id: dispatch.component_id.clone(),
                node_index: dispatch.node_index,
                node_id: dispatch.node_id.clone(),
            })
            .collect::<Vec<_>>();
        Self::from_dispatches(contract, execution_scope.into(), &dispatches)
    }

    fn from_dispatches(
        contract: &CompiledResourceResidencyContract,
        execution_scope: String,
        dispatches: &[VulkanPhysicalResidencyDispatch],
    ) -> Result<Self, VulkanPhysicalResidencyCheckpointError> {
        if execution_scope.trim().is_empty() {
            return Err(VulkanPhysicalResidencyCheckpointError::new(
                "physical residency execution scope is empty",
            ));
        }
        let selector_by_id = contract
            .selectors
            .iter()
            .map(|selector| (selector.id.as_str(), selector))
            .collect::<BTreeMap<_, _>>();
        let mut checkpoints = Vec::new();
        for compiled in contract
            .checkpoints
            .iter()
            .filter(|checkpoint| checkpoint.execution_scope == execution_scope)
        {
            let component_dispatches = dispatches
                .iter()
                .filter(|dispatch| dispatch.component_id == compiled.component_id)
                .collect::<Vec<_>>();
            if component_dispatches.is_empty() {
                continue;
            }
            if component_dispatches
                .windows(2)
                .any(|pair| pair[0].node_index >= pair[1].node_index)
            {
                return Err(VulkanPhysicalResidencyCheckpointError::new(format!(
                    "physical dispatch order for component {:?} is not strictly increasing",
                    compiled.component_id
                )));
            }
            let selection_position = component_dispatches
                .iter()
                .position(|dispatch| dispatch.node_id == compiled.after_node_id)
                .ok_or_else(|| {
                    VulkanPhysicalResidencyCheckpointError::new(format!(
                        "checkpoint {:?} selector node {:?} is absent from its placed dispatch slice",
                        compiled.id, compiled.after_node_id
                    ))
                })?;
            let selected_selectors = compiled
                .selector_ids
                .iter()
                .map(|selector_id| {
                    let selector = selector_by_id.get(selector_id.as_str()).ok_or_else(|| {
                        VulkanPhysicalResidencyCheckpointError::new(format!(
                            "checkpoint {:?} references unknown selector {selector_id:?}",
                            compiled.id
                        ))
                    })?;
                    if selector.execution_scope != compiled.execution_scope
                        || selector.component_id != compiled.component_id
                        || selector.node_id != compiled.after_node_id
                    {
                        return Err(VulkanPhysicalResidencyCheckpointError::new(format!(
                            "checkpoint {:?} selector {selector_id:?} crosses its physical boundary",
                            compiled.id
                        )));
                    }
                    Ok(*selector)
                })
                .collect::<Result<Vec<_>, VulkanPhysicalResidencyCheckpointError>>()?;
            let selected_node_ids = contract
                .bindings
                .iter()
                .filter(|binding| {
                    binding.execution_scope == compiled.execution_scope
                        && binding.component_id == compiled.component_id
                        && selected_selectors
                            .iter()
                            .any(|selector| binding_is_selected_by(&binding.mapping, selector))
                })
                .map(|binding| binding.node_id.as_str())
                .collect::<BTreeSet<_>>();
            if selected_node_ids.is_empty() {
                return Err(VulkanPhysicalResidencyCheckpointError::new(format!(
                    "checkpoint {:?} has no physically dependent dispatch",
                    compiled.id
                )));
            }
            let selected_positions = component_dispatches
                .iter()
                .enumerate()
                .filter_map(|(position, dispatch)| {
                    selected_node_ids
                        .contains(dispatch.node_id.as_str())
                        .then_some(position)
                })
                .collect::<Vec<_>>();
            if selected_positions.len() != selected_node_ids.len() {
                return Err(VulkanPhysicalResidencyCheckpointError::new(format!(
                    "checkpoint {:?} selected parameter bindings do not map one-to-one to placed dispatches",
                    compiled.id
                )));
            }
            let first_selected = *selected_positions
                .first()
                .expect("selected positions are non-empty");
            let last_selected = *selected_positions
                .last()
                .expect("selected positions are non-empty");
            if first_selected <= selection_position
                || component_dispatches[first_selected].node_id != compiled.resume_node_id
            {
                return Err(VulkanPhysicalResidencyCheckpointError::new(format!(
                    "checkpoint {:?} does not resume at its first dependent dispatch",
                    compiled.id
                )));
            }
            let selected_computation_dispatch_indices = component_dispatches
                [first_selected..=last_selected]
                .iter()
                .map(|dispatch| dispatch.dispatch_index)
                .collect::<Vec<_>>();
            let selected_result_continuation_dispatch_index = component_dispatches
                .get(last_selected + 1)
                .map(|dispatch| dispatch.dispatch_index);
            checkpoints.push(VulkanPhysicalResidencyCheckpoint {
                id: compiled.id.clone(),
                execution_scope: compiled.execution_scope.clone(),
                component_id: compiled.component_id.clone(),
                selector_ids: compiled.selector_ids.clone(),
                selection_dispatch_index: component_dispatches[selection_position].dispatch_index,
                selected_computation_dispatch_indices,
                selected_result_continuation_dispatch_index,
            });
        }
        checkpoints.sort_by_key(|checkpoint| checkpoint.selection_dispatch_index);
        if checkpoints
            .windows(2)
            .any(|pair| pair[0].selection_dispatch_index >= pair[1].selection_dispatch_index)
        {
            return Err(VulkanPhysicalResidencyCheckpointError::new(
                "physical residency checkpoints overlap or repeat a selector dispatch",
            ));
        }
        Ok(Self {
            execution_scope,
            checkpoints,
        })
    }

    pub fn checkpoint(&self, checkpoint_id: &str) -> Option<&VulkanPhysicalResidencyCheckpoint> {
        self.checkpoints
            .iter()
            .find(|checkpoint| checkpoint.id == checkpoint_id)
    }
}

fn validate_physical_residency_schedule_coverage<'a>(
    contract: &CompiledResourceResidencyContract,
    execution_scope: &str,
    schedules: impl IntoIterator<Item = &'a VulkanPhysicalResidencySchedule>,
) -> Result<(), VulkanPhysicalResidencyCheckpointError> {
    let expected = contract
        .checkpoints
        .iter()
        .filter(|checkpoint| checkpoint.execution_scope == execution_scope)
        .map(|checkpoint| checkpoint.id.as_str())
        .collect::<BTreeSet<_>>();
    let mut actual = BTreeSet::new();
    for schedule in schedules {
        if schedule.execution_scope != execution_scope {
            return Err(VulkanPhysicalResidencyCheckpointError::new(format!(
                "physical residency schedule scope {:?} does not match {execution_scope:?}",
                schedule.execution_scope
            )));
        }
        for checkpoint in &schedule.checkpoints {
            if !actual.insert(checkpoint.id.as_str()) {
                return Err(VulkanPhysicalResidencyCheckpointError::new(format!(
                    "physical residency checkpoint {:?} is owned by more than one device slice",
                    checkpoint.id
                )));
            }
        }
    }
    if actual != expected {
        let missing = expected.difference(&actual).copied().collect::<Vec<_>>();
        let unexpected = actual.difference(&expected).copied().collect::<Vec<_>>();
        return Err(VulkanPhysicalResidencyCheckpointError::new(format!(
            "placed physical residency checkpoint coverage is incomplete; missing={missing:?}, unexpected={unexpected:?}"
        )));
    }
    Ok(())
}

fn binding_is_selected_by(
    binding: &CompiledResourceBindingMapping,
    selector: &CompiledResourceSelector,
) -> bool {
    match (binding, &selector.mapping) {
        (
            CompiledResourceBindingMapping::SelectedAtomicGroup {
                atomic_group_id,
                selection_signal,
                ..
            },
            CompiledResourceSelectorMapping::GroupTable { atomic_group_ids },
        ) => {
            atomic_group_ids.contains(atomic_group_id)
                && selection_signal == &selector.selection_signal
        }
        (
            CompiledResourceBindingMapping::PartitionTemplateMember {
                partition_template_id: binding_template,
                selection_signal,
                ..
            },
            CompiledResourceSelectorMapping::PartitionTemplate {
                partition_template_id: selector_template,
            },
        ) => {
            binding_template == selector_template
                && selection_signal == &selector.selection_signal
        }
        _ => false,
    }
}

impl VulkanPhysicalResidencyCheckpoint {
    pub fn resolve_selected_group_ids(
        &self,
        contract: &CompiledResourceResidencyContract,
        selections: &[VulkanSelectedResourceIndex],
    ) -> Result<Vec<String>, VulkanPhysicalResidencyCheckpointError> {
        let mut indices_by_selector = BTreeMap::<&str, BTreeSet<usize>>::new();
        for selection in selections {
            if !self
                .selector_ids
                .iter()
                .any(|selector_id| selector_id == &selection.selector_id)
            {
                return Err(VulkanPhysicalResidencyCheckpointError::new(format!(
                    "selection for selector {:?} does not belong to checkpoint {:?}",
                    selection.selector_id, self.id
                )));
            }
            if !indices_by_selector
                .entry(selection.selector_id.as_str())
                .or_default()
                .insert(selection.resource_index)
            {
                return Err(VulkanPhysicalResidencyCheckpointError::new(format!(
                    "checkpoint {:?} repeats selector {:?} resource index {}",
                    self.id, selection.selector_id, selection.resource_index
                )));
            }
        }
        let mut group_ids = BTreeSet::new();
        for selector_id in &self.selector_ids {
            let indices = indices_by_selector
                .get(selector_id.as_str())
                .filter(|indices| !indices.is_empty())
                .ok_or_else(|| {
                    VulkanPhysicalResidencyCheckpointError::new(format!(
                        "checkpoint {:?} has no selected index for selector {selector_id:?}",
                        self.id
                    ))
                })?;
            let selector = contract
                .selectors
                .iter()
                .find(|selector| selector.id == *selector_id)
                .ok_or_else(|| {
                    VulkanPhysicalResidencyCheckpointError::new(format!(
                        "checkpoint {:?} references unknown selector {selector_id:?}",
                        self.id
                    ))
                })?;
            for index in indices {
                if *index >= selector.resource_count {
                    return Err(VulkanPhysicalResidencyCheckpointError::new(format!(
                        "selector {selector_id:?} index {index} exceeds resource count {}",
                        selector.resource_count
                    )));
                }
                let group_id = match &selector.mapping {
                    CompiledResourceSelectorMapping::GroupTable { atomic_group_ids } => {
                        atomic_group_ids[*index].clone()
                    }
                    CompiledResourceSelectorMapping::PartitionTemplate {
                        partition_template_id,
                    } => {
                        let template = contract
                            .partition_templates
                            .iter()
                            .find(|template| template.id == *partition_template_id)
                            .ok_or_else(|| {
                                VulkanPhysicalResidencyCheckpointError::new(format!(
                                    "selector {selector_id:?} references unknown partition template {partition_template_id:?}"
                                ))
                            })?;
                        derived_partition_resource_id(
                            &template.group_identity_seed,
                            *index,
                        )
                        .map_err(|error| {
                            VulkanPhysicalResidencyCheckpointError::new(format!(
                                "failed to derive selected group identity: {error}"
                            ))
                        })?
                    }
                };
                group_ids.insert(group_id);
            }
        }
        Ok(group_ids.into_iter().collect())
    }

    pub fn begin_activation(
        &self,
        selected_group_ids: Vec<String>,
    ) -> Result<VulkanPhysicalResidencyActivation, VulkanPhysicalResidencyCheckpointError> {
        if selected_group_ids.is_empty()
            || selected_group_ids
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(VulkanPhysicalResidencyCheckpointError::new(
                "selected residency groups must be non-empty, unique, and sorted",
            ));
        }
        Ok(VulkanPhysicalResidencyActivation {
            checkpoint: self.clone(),
            selected_group_ids,
            next_responsibility: VulkanPhysicalResidencyResponsibility::Selection,
            blocked_missing_group_ids: Vec::new(),
            completed: false,
            trace: Vec::new(),
        })
    }
}

impl VulkanPhysicalResidencyActivation {
    pub fn advance(
        &mut self,
        resident_group_ids: &BTreeSet<String>,
    ) -> Result<VulkanPhysicalResidencyActivationStatus, VulkanPhysicalResidencyCheckpointError> {
        if self.completed {
            return Ok(VulkanPhysicalResidencyActivationStatus::Completed);
        }
        if !self.blocked_missing_group_ids.is_empty() {
            return Err(VulkanPhysicalResidencyCheckpointError::new(
                "paused residency activation must resume through atomic publication",
            ));
        }
        loop {
            match self.next_responsibility {
                VulkanPhysicalResidencyResponsibility::Selection => {
                    self.push_trace(
                        VulkanPhysicalResidencyResponsibility::Selection,
                        vec![self.checkpoint.selection_dispatch_index],
                    );
                    self.next_responsibility =
                        VulkanPhysicalResidencyResponsibility::Availability;
                }
                VulkanPhysicalResidencyResponsibility::Availability => {
                    self.push_trace(
                        VulkanPhysicalResidencyResponsibility::Availability,
                        Vec::new(),
                    );
                    let missing_group_ids = self
                        .selected_group_ids
                        .iter()
                        .filter(|group_id| !resident_group_ids.contains(*group_id))
                        .cloned()
                        .collect::<Vec<_>>();
                    self.next_responsibility =
                        VulkanPhysicalResidencyResponsibility::SelectedComputation;
                    if !missing_group_ids.is_empty() {
                        self.blocked_missing_group_ids = missing_group_ids.clone();
                        return Ok(VulkanPhysicalResidencyActivationStatus::Paused {
                            checkpoint_id: self.checkpoint.id.clone(),
                            missing_group_ids,
                            resume_at:
                                VulkanPhysicalResidencyResponsibility::SelectedComputation,
                        });
                    }
                }
                VulkanPhysicalResidencyResponsibility::SelectedComputation => {
                    self.push_trace(
                        VulkanPhysicalResidencyResponsibility::SelectedComputation,
                        self.checkpoint
                            .selected_computation_dispatch_indices
                            .clone(),
                    );
                    if self
                        .checkpoint
                        .selected_result_continuation_dispatch_index
                        .is_some()
                    {
                        self.next_responsibility =
                            VulkanPhysicalResidencyResponsibility::SelectedResultContinuation;
                    } else {
                        self.completed = true;
                        return Ok(VulkanPhysicalResidencyActivationStatus::Completed);
                    }
                }
                VulkanPhysicalResidencyResponsibility::SelectedResultContinuation => {
                    self.push_trace(
                        VulkanPhysicalResidencyResponsibility::SelectedResultContinuation,
                        vec![
                            self.checkpoint
                                .selected_result_continuation_dispatch_index
                                .expect("continuation responsibility requires a dispatch"),
                        ],
                    );
                    self.completed = true;
                    return Ok(VulkanPhysicalResidencyActivationStatus::Completed);
                }
            }
        }
    }

    pub fn resume_after_atomic_publication(
        &mut self,
        resident_group_ids: &BTreeSet<String>,
    ) -> Result<VulkanPhysicalResidencyActivationStatus, VulkanPhysicalResidencyCheckpointError> {
        if self.blocked_missing_group_ids.is_empty()
            || self.next_responsibility
                != VulkanPhysicalResidencyResponsibility::SelectedComputation
        {
            return Err(VulkanPhysicalResidencyCheckpointError::new(
                "residency activation is not paused at selected computation",
            ));
        }
        let unpublished = self
            .blocked_missing_group_ids
            .iter()
            .filter(|group_id| !resident_group_ids.contains(*group_id))
            .cloned()
            .collect::<Vec<_>>();
        if !unpublished.is_empty() {
            return Err(VulkanPhysicalResidencyCheckpointError::new(format!(
                "cannot resume checkpoint {:?}; atomic groups remain unpublished: {}",
                self.checkpoint.id,
                unpublished.join(", ")
            )));
        }
        self.blocked_missing_group_ids.clear();
        self.advance(resident_group_ids)
    }

    pub fn trace(&self) -> &[VulkanPhysicalResidencyTraceEntry] {
        &self.trace
    }

    fn push_trace(
        &mut self,
        responsibility: VulkanPhysicalResidencyResponsibility,
        dispatch_indices: Vec<usize>,
    ) {
        self.trace.push(VulkanPhysicalResidencyTraceEntry {
            checkpoint_id: self.checkpoint.id.clone(),
            responsibility,
            dispatch_indices,
            selected_group_ids: self.selected_group_ids.clone(),
        });
    }
}
