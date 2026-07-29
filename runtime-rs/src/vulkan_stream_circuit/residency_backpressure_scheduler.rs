#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidencyBackpressureLimits {
    pub maximum_owned_activations: usize,
    pub maximum_groups_per_activation: usize,
    pub maximum_outstanding_loads: usize,
}

impl VulkanResidencyBackpressureLimits {
    pub fn validate(&self) -> Result<(), VulkanResidencyBackpressureError> {
        if self.maximum_owned_activations == 0
            || self.maximum_groups_per_activation == 0
            || self.maximum_outstanding_loads == 0
        {
            return Err(VulkanResidencyBackpressureError::configuration(
                "residency scheduler limits must be non-zero",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct VulkanResidencyCheckpointActivationKey {
    pub activation_id: u64,
    pub stream_id: String,
    pub device_id: String,
    pub checkpoint_ordinal: usize,
}

impl VulkanResidencyCheckpointActivationKey {
    pub fn new(
        activation_id: u64,
        stream_id: impl Into<String>,
        device_id: impl Into<String>,
        checkpoint_ordinal: usize,
    ) -> Result<Self, VulkanResidencyBackpressureError> {
        let key = Self {
            activation_id,
            stream_id: stream_id.into(),
            device_id: device_id.into(),
            checkpoint_ordinal,
        };
        if key.stream_id.trim().is_empty() || key.device_id.trim().is_empty() {
            return Err(VulkanResidencyBackpressureError::configuration(
                "residency checkpoint activation has an empty stream or device identity",
            ));
        }
        Ok(key)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VulkanResidencyBackpressureErrorKind {
    Cancelled,
    Configuration,
    Load,
    QueueFull,
    SchedulerState,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidencyBackpressureError {
    kind: VulkanResidencyBackpressureErrorKind,
    message: String,
}

impl VulkanResidencyBackpressureError {
    fn new(
        kind: VulkanResidencyBackpressureErrorKind,
        message: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    fn configuration(message: impl Into<String>) -> Self {
        Self::new(VulkanResidencyBackpressureErrorKind::Configuration, message)
    }

    fn queue_full(message: impl Into<String>) -> Self {
        Self::new(VulkanResidencyBackpressureErrorKind::QueueFull, message)
    }

    fn scheduler_state(message: impl Into<String>) -> Self {
        Self::new(VulkanResidencyBackpressureErrorKind::SchedulerState, message)
    }

    pub fn kind(&self) -> VulkanResidencyBackpressureErrorKind {
        self.kind
    }
}

impl Display for VulkanResidencyBackpressureError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for VulkanResidencyBackpressureError {}

impl From<DeviceResourceResidencyError> for VulkanResidencyBackpressureError {
    fn from(error: DeviceResourceResidencyError) -> Self {
        let kind = match error.kind() {
            DeviceResourceResidencyErrorKind::Backpressure => {
                VulkanResidencyBackpressureErrorKind::QueueFull
            }
            DeviceResourceResidencyErrorKind::Cancelled => {
                VulkanResidencyBackpressureErrorKind::Cancelled
            }
            _ => VulkanResidencyBackpressureErrorKind::Load,
        };
        Self::new(kind, error.to_string())
    }
}

impl From<VulkanPhysicalResidencyCheckpointError> for VulkanResidencyBackpressureError {
    fn from(error: VulkanPhysicalResidencyCheckpointError) -> Self {
        Self::new(
            VulkanResidencyBackpressureErrorKind::Configuration,
            error.to_string(),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanResidencyCheckpointAdmission {
    Ready {
        activation_id: u64,
    },
    Blocked {
        activation_id: u64,
        missing_group_ids: Vec<String>,
        new_load_count: usize,
        joined_load_count: usize,
    },
}

pub struct VulkanResidencyReadyActivation<C, P: DeviceResidentResourcePayload> {
    key: VulkanResidencyCheckpointActivationKey,
    continuation: C,
    checkpoint: VulkanPhysicalResidencyActivation,
    leases: Vec<DeviceResourceResidencyLease<P>>,
}

impl<C, P: DeviceResidentResourcePayload> VulkanResidencyReadyActivation<C, P> {
    pub fn key(&self) -> &VulkanResidencyCheckpointActivationKey {
        &self.key
    }

    pub fn continuation(&self) -> &C {
        &self.continuation
    }

    pub fn checkpoint_trace(&self) -> &[VulkanPhysicalResidencyTraceEntry] {
        self.checkpoint.trace()
    }

    pub fn resident_group_ids(&self) -> Vec<&str> {
        self.leases
            .iter()
            .map(|lease| lease.group().descriptor().id.as_str())
            .collect()
    }

    pub fn into_continuation(self) -> C {
        self.continuation
    }
}

pub struct VulkanResidencyFailedActivation<C> {
    key: VulkanResidencyCheckpointActivationKey,
    continuation: C,
    checkpoint: VulkanPhysicalResidencyActivation,
    error: VulkanResidencyBackpressureError,
}

impl<C> VulkanResidencyFailedActivation<C> {
    pub fn key(&self) -> &VulkanResidencyCheckpointActivationKey {
        &self.key
    }

    pub fn continuation(&self) -> &C {
        &self.continuation
    }

    pub fn checkpoint_trace(&self) -> &[VulkanPhysicalResidencyTraceEntry] {
        self.checkpoint.trace()
    }

    pub fn error(&self) -> &VulkanResidencyBackpressureError {
        &self.error
    }

    pub fn into_continuation(self) -> C {
        self.continuation
    }
}

pub struct VulkanResidencyCancelledActivation<C> {
    pub key: VulkanResidencyCheckpointActivationKey,
    pub continuation: C,
}

struct VulkanResidencyLoadCompletion<P: DeviceResidentResourcePayload> {
    load_id: u64,
    device_id: String,
    group_id: String,
    result:
        Result<DeviceResourceResidencyLease<P>, DeviceResourceResidencyError>,
}

pub struct VulkanResidencyLoadCommand<P: DeviceResidentResourcePayload> {
    load_id: u64,
    device_id: String,
    group_id: String,
    permit: Option<DeviceResourceLoadPermit<P>>,
    completion_sender:
        std::sync::mpsc::Sender<VulkanResidencyLoadCompletion<P>>,
}

impl<P: DeviceResidentResourcePayload> VulkanResidencyLoadCommand<P> {
    pub fn load_id(&self) -> u64 {
        self.load_id
    }

    pub fn device_id(&self) -> &str {
        &self.device_id
    }

    pub fn group_id(&self) -> &str {
        &self.group_id
    }

    pub fn descriptor(&self) -> &DeviceResourceGroupDescriptor {
        self.permit
            .as_ref()
            .expect("unfinished residency load command owns its permit")
            .descriptor()
    }

    pub fn publish(
        mut self,
        group: DeviceResidentResourceGroup<P>,
    ) -> Result<(), VulkanResidencyBackpressureError> {
        let result = self
            .permit
            .take()
            .expect("residency load command can finish only once")
            .publish(group);
        let public_result = result
            .as_ref()
            .map(|_| ())
            .map_err(|error| {
                VulkanResidencyBackpressureError::from(error.clone())
            });
        self.send_completion(result)?;
        public_result
    }

    pub fn fail(
        mut self,
        error: DeviceResourceResidencyError,
    ) -> Result<(), VulkanResidencyBackpressureError> {
        let result = self
            .permit
            .take()
            .expect("residency load command can finish only once")
            .fail(error.clone())
            .and(Err(error));
        self.send_completion(result)
    }

    pub fn cancel(mut self) -> Result<(), VulkanResidencyBackpressureError> {
        let result = self
            .permit
            .take()
            .expect("residency load command can finish only once")
            .cancel()
            .and(Err(DeviceResourceResidencyError::new(
                DeviceResourceResidencyErrorKind::Cancelled,
                "residency load command was cancelled",
            )));
        self.send_completion(result)
    }

    fn send_completion(
        &self,
        result: Result<DeviceResourceResidencyLease<P>, DeviceResourceResidencyError>,
    ) -> Result<(), VulkanResidencyBackpressureError> {
        self.completion_sender
            .send(VulkanResidencyLoadCompletion {
                load_id: self.load_id,
                device_id: self.device_id.clone(),
                group_id: self.group_id.clone(),
                result,
            })
            .map_err(|_| {
                VulkanResidencyBackpressureError::scheduler_state(
                    "residency scheduler was dropped before a load completed",
                )
            })
    }
}

impl<P: DeviceResidentResourcePayload> Drop for VulkanResidencyLoadCommand<P> {
    fn drop(&mut self) {
        let Some(permit) = self.permit.take() else {
            return;
        };
        let result = permit.cancel().and(Err(DeviceResourceResidencyError::new(
            DeviceResourceResidencyErrorKind::Cancelled,
            "residency load command was dropped before completion",
        )));
        let _ = self.completion_sender.send(VulkanResidencyLoadCompletion {
            load_id: self.load_id,
            device_id: self.device_id.clone(),
            group_id: self.group_id.clone(),
            result,
        });
    }
}

enum VulkanBlockedResidencyGroup<P: DeviceResidentResourcePayload> {
    Loading { load_id: u64 },
    Waiting(DeviceResourceResidencyWaiter<P>),
}

struct VulkanBlockedResidencyActivation<C, P: DeviceResidentResourcePayload> {
    admission_sequence: u64,
    key: VulkanResidencyCheckpointActivationKey,
    continuation: C,
    checkpoint: VulkanPhysicalResidencyActivation,
    pending_groups: BTreeMap<String, VulkanBlockedResidencyGroup<P>>,
    leases: BTreeMap<String, DeviceResourceResidencyLease<P>>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct VulkanResidencyDeviceGroupKey {
    device_id: String,
    group_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanResidencyInflightLoad {
    device_group: VulkanResidencyDeviceGroupKey,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VulkanResidencyBackpressureSnapshot {
    pub blocked_activation_count: usize,
    pub ready_activation_count: usize,
    pub failed_activation_count: usize,
    pub queued_load_count: usize,
    pub outstanding_load_count: usize,
}

pub struct VulkanResidencyBackpressureScheduler<
    C,
    P: DeviceResidentResourcePayload,
> {
    limits: VulkanResidencyBackpressureLimits,
    next_admission_sequence: u64,
    next_load_id: u64,
    blocked: BTreeMap<u64, VulkanBlockedResidencyActivation<C, P>>,
    blocked_by_group: BTreeMap<VulkanResidencyDeviceGroupKey, BTreeSet<u64>>,
    ready: VecDeque<VulkanResidencyReadyActivation<C, P>>,
    failed: VecDeque<VulkanResidencyFailedActivation<C>>,
    owned_streams: BTreeMap<String, u64>,
    queued_loads: VecDeque<VulkanResidencyLoadCommand<P>>,
    inflight_loads: BTreeMap<u64, VulkanResidencyInflightLoad>,
    completion_sender:
        std::sync::mpsc::Sender<VulkanResidencyLoadCompletion<P>>,
    completion_receiver:
        std::sync::mpsc::Receiver<VulkanResidencyLoadCompletion<P>>,
}

impl<C, P: DeviceResidentResourcePayload>
    VulkanResidencyBackpressureScheduler<C, P>
{
    pub fn new(
        limits: VulkanResidencyBackpressureLimits,
    ) -> Result<Self, VulkanResidencyBackpressureError> {
        limits.validate()?;
        let (completion_sender, completion_receiver) =
            std::sync::mpsc::channel();
        Ok(Self {
            limits,
            next_admission_sequence: 0,
            next_load_id: 0,
            blocked: BTreeMap::new(),
            blocked_by_group: BTreeMap::new(),
            ready: VecDeque::new(),
            failed: VecDeque::new(),
            owned_streams: BTreeMap::new(),
            queued_loads: VecDeque::new(),
            inflight_loads: BTreeMap::new(),
            completion_sender,
            completion_receiver,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn admit_checkpoint(
        &mut self,
        manager: &DeviceResourceResidencyManager<P>,
        owner: DeviceResourceResidencyOwnerId,
        key: VulkanResidencyCheckpointActivationKey,
        checkpoint: &VulkanPhysicalResidencyCheckpoint,
        descriptors: Vec<DeviceResourceGroupDescriptor>,
        continuation: C,
    ) -> Result<VulkanResidencyCheckpointAdmission, VulkanResidencyBackpressureError> {
        if manager.device_id() != key.device_id {
            return Err(VulkanResidencyBackpressureError::configuration(format!(
                "activation device {:?} does not match residency manager {:?}",
                key.device_id,
                manager.device_id()
            )));
        }
        if descriptors.is_empty()
            || descriptors.len() > self.limits.maximum_groups_per_activation
        {
            return Err(VulkanResidencyBackpressureError::configuration(format!(
                "checkpoint activation has {} groups; bounded maximum is {}",
                descriptors.len(),
                self.limits.maximum_groups_per_activation
            )));
        }
        if descriptors
            .windows(2)
            .any(|pair| pair[0].id >= pair[1].id)
        {
            return Err(VulkanResidencyBackpressureError::configuration(
                "checkpoint group descriptors must be unique and strictly sorted",
            ));
        }
        if self.owned_activation_count() >= self.limits.maximum_owned_activations
        {
            return Err(VulkanResidencyBackpressureError::queue_full(
                "residency scheduler activation capacity is full",
            ));
        }
        if self.blocked.contains_key(&key.activation_id)
            || self
                .ready
                .iter()
                .any(|ready| ready.key.activation_id == key.activation_id)
            || self
                .failed
                .iter()
                .any(|failed| failed.key.activation_id == key.activation_id)
        {
            return Err(VulkanResidencyBackpressureError::configuration(format!(
                "activation {} is already owned by the residency scheduler",
                key.activation_id
            )));
        }
        if let Some(existing) = self.owned_streams.get(&key.stream_id) {
            return Err(VulkanResidencyBackpressureError::configuration(format!(
                "stream {:?} already has residency activation {} in flight",
                key.stream_id, existing
            )));
        }
        self.next_admission_sequence
            .checked_add(1)
            .ok_or_else(|| {
                VulkanResidencyBackpressureError::scheduler_state(
                    "residency admission sequence exhausted",
                )
            })?;
        self.next_load_id
            .checked_add(u64::try_from(descriptors.len()).map_err(|_| {
                VulkanResidencyBackpressureError::configuration(
                    "checkpoint group count does not fit u64",
                )
            })?)
            .ok_or_else(|| {
                VulkanResidencyBackpressureError::scheduler_state(
                    "residency load identity exhausted",
                )
            })?;

        let selected_group_ids = descriptors
            .iter()
            .map(|descriptor| descriptor.id.clone())
            .collect::<Vec<_>>();
        let mut physical = checkpoint.begin_activation(selected_group_ids)?;
        let available_load_slots = self
            .limits
            .maximum_outstanding_loads
            .saturating_sub(self.inflight_loads.len());
        let requests = manager.request_batch_with_new_load_limit(
            descriptors.clone(),
            owner,
            available_load_slots,
        )?;
        let mut resident_ids = BTreeSet::new();
        let mut leases = BTreeMap::new();
        let mut pending_groups = BTreeMap::new();
        let mut new_load_count = 0usize;
        let mut joined_load_count = 0usize;

        for (descriptor, request) in descriptors.into_iter().zip(requests) {
            match request {
                DeviceResourceResidencyRequest::Resident(lease) => {
                    resident_ids.insert(descriptor.id.clone());
                    leases.insert(descriptor.id, lease);
                }
                DeviceResourceResidencyRequest::LoadRequired(permit) => {
                    self.next_load_id += 1;
                    let load_id = self.next_load_id;
                    let device_group = VulkanResidencyDeviceGroupKey {
                        device_id: key.device_id.clone(),
                        group_id: descriptor.id.clone(),
                    };
                    self.inflight_loads.insert(
                        load_id,
                        VulkanResidencyInflightLoad {
                            device_group: device_group.clone(),
                        },
                    );
                    self.queued_loads.push_back(VulkanResidencyLoadCommand {
                        load_id,
                        device_id: key.device_id.clone(),
                        group_id: descriptor.id.clone(),
                        permit: Some(permit),
                        completion_sender: self.completion_sender.clone(),
                    });
                    pending_groups.insert(
                        descriptor.id,
                        VulkanBlockedResidencyGroup::Loading { load_id },
                    );
                    new_load_count += 1;
                }
                DeviceResourceResidencyRequest::Pending(waiter) => {
                    pending_groups.insert(
                        descriptor.id,
                        VulkanBlockedResidencyGroup::Waiting(waiter),
                    );
                    joined_load_count += 1;
                }
            }
        }

        let status = physical.advance(&resident_ids)?;
        self.next_admission_sequence += 1;
        let admission_sequence = self.next_admission_sequence;
        self.owned_streams
            .insert(key.stream_id.clone(), key.activation_id);
        match status {
            VulkanPhysicalResidencyActivationStatus::Completed => {
                if !pending_groups.is_empty() {
                    return Err(VulkanResidencyBackpressureError::scheduler_state(
                        "physical checkpoint completed while residency requests remain pending",
                    ));
                }
                self.ready.push_back(VulkanResidencyReadyActivation {
                    key: key.clone(),
                    continuation,
                    checkpoint: physical,
                    leases: leases.into_values().collect(),
                });
                Ok(VulkanResidencyCheckpointAdmission::Ready {
                    activation_id: key.activation_id,
                })
            }
            VulkanPhysicalResidencyActivationStatus::Paused {
                missing_group_ids, ..
            } => {
                let pending_ids = pending_groups.keys().cloned().collect::<Vec<_>>();
                if missing_group_ids != pending_ids {
                    return Err(VulkanResidencyBackpressureError::scheduler_state(
                        "physical checkpoint miss set diverged from residency requests",
                    ));
                }
                for group_id in &missing_group_ids {
                    self.blocked_by_group
                        .entry(VulkanResidencyDeviceGroupKey {
                            device_id: key.device_id.clone(),
                            group_id: group_id.clone(),
                        })
                        .or_default()
                        .insert(key.activation_id);
                }
                self.blocked.insert(
                    key.activation_id,
                    VulkanBlockedResidencyActivation {
                        admission_sequence,
                        key: key.clone(),
                        continuation,
                        checkpoint: physical,
                        pending_groups,
                        leases,
                    },
                );
                Ok(VulkanResidencyCheckpointAdmission::Blocked {
                    activation_id: key.activation_id,
                    missing_group_ids,
                    new_load_count,
                    joined_load_count,
                })
            }
        }
    }

    pub fn pop_load_command(&mut self) -> Option<VulkanResidencyLoadCommand<P>> {
        self.queued_loads.pop_front()
    }

    pub fn poll_load_completions(
        &mut self,
    ) -> Result<usize, VulkanResidencyBackpressureError> {
        let mut completed_count = 0usize;
        loop {
            match self.completion_receiver.try_recv() {
                Ok(completion) => {
                    self.handle_load_completion(completion)?;
                    completed_count = completed_count.saturating_add(1);
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    return Err(VulkanResidencyBackpressureError::scheduler_state(
                        "residency load completion channel disconnected",
                    ));
                }
            }
        }
        completed_count = completed_count
            .saturating_add(self.poll_externally_completed_groups()?);
        Ok(completed_count)
    }

    pub fn pop_ready(&mut self) -> Option<VulkanResidencyReadyActivation<C, P>> {
        let ready = self.ready.pop_front()?;
        self.owned_streams.remove(&ready.key.stream_id);
        Some(ready)
    }

    pub fn pop_ready_for_device(
        &mut self,
        device_id: &str,
    ) -> Option<VulkanResidencyReadyActivation<C, P>> {
        let index = self
            .ready
            .iter()
            .position(|ready| ready.key.device_id == device_id)?;
        let ready = self.ready.remove(index)?;
        self.owned_streams.remove(&ready.key.stream_id);
        Some(ready)
    }

    pub fn pop_failed(&mut self) -> Option<VulkanResidencyFailedActivation<C>> {
        let failed = self.failed.pop_front()?;
        self.owned_streams.remove(&failed.key.stream_id);
        Some(failed)
    }

    pub fn cancel_activation(
        &mut self,
        activation_id: u64,
    ) -> Option<VulkanResidencyCancelledActivation<C>> {
        if let Some(blocked) = self.remove_blocked_activation(activation_id) {
            self.owned_streams.remove(&blocked.key.stream_id);
            return Some(VulkanResidencyCancelledActivation {
                key: blocked.key,
                continuation: blocked.continuation,
            });
        }
        if let Some(index) = self
            .ready
            .iter()
            .position(|ready| ready.key.activation_id == activation_id)
        {
            let ready = self.ready.remove(index)?;
            self.owned_streams.remove(&ready.key.stream_id);
            return Some(VulkanResidencyCancelledActivation {
                key: ready.key,
                continuation: ready.continuation,
            });
        }
        if let Some(index) = self
            .failed
            .iter()
            .position(|failed| failed.key.activation_id == activation_id)
        {
            let failed = self.failed.remove(index)?;
            self.owned_streams.remove(&failed.key.stream_id);
            return Some(VulkanResidencyCancelledActivation {
                key: failed.key,
                continuation: failed.continuation,
            });
        }
        None
    }

    pub fn cancel_stream(
        &mut self,
        stream_id: &str,
    ) -> Option<VulkanResidencyCancelledActivation<C>> {
        let activation_id = self.owned_streams.get(stream_id).copied()?;
        self.cancel_activation(activation_id)
    }

    pub fn snapshot(&self) -> VulkanResidencyBackpressureSnapshot {
        VulkanResidencyBackpressureSnapshot {
            blocked_activation_count: self.blocked.len(),
            ready_activation_count: self.ready.len(),
            failed_activation_count: self.failed.len(),
            queued_load_count: self.queued_loads.len(),
            outstanding_load_count: self.inflight_loads.len(),
        }
    }

    fn owned_activation_count(&self) -> usize {
        self.blocked.len() + self.ready.len() + self.failed.len()
    }

    fn handle_load_completion(
        &mut self,
        completion: VulkanResidencyLoadCompletion<P>,
    ) -> Result<(), VulkanResidencyBackpressureError> {
        let inflight = self
            .inflight_loads
            .remove(&completion.load_id)
            .ok_or_else(|| {
                VulkanResidencyBackpressureError::scheduler_state(format!(
                    "unknown or duplicate residency load completion {}",
                    completion.load_id
                ))
            })?;
        let completed_group = VulkanResidencyDeviceGroupKey {
            device_id: completion.device_id,
            group_id: completion.group_id,
        };
        if inflight.device_group != completed_group {
            return Err(VulkanResidencyBackpressureError::scheduler_state(
                "residency load completion identity does not match its command",
            ));
        }
        let affected = self
            .blocked_by_group
            .remove(&completed_group)
            .unwrap_or_default();
        match completion.result {
            Ok(leader_lease) => {
                self.finish_successful_group_load(
                    completion.load_id,
                    completed_group,
                    affected,
                    leader_lease,
                )?;
            }
            Err(error) => {
                self.fail_affected_activations(
                    affected,
                    VulkanResidencyBackpressureError::from(error),
                );
            }
        }
        Ok(())
    }

    fn finish_successful_group_load(
        &mut self,
        load_id: u64,
        completed_group: VulkanResidencyDeviceGroupKey,
        affected: BTreeSet<u64>,
        leader_lease: DeviceResourceResidencyLease<P>,
    ) -> Result<(), VulkanResidencyBackpressureError> {
        let mut leader_lease = Some(leader_lease);
        let mut resume_candidates = Vec::new();
        for activation_id in affected {
            let Some(blocked) = self.blocked.get_mut(&activation_id) else {
                continue;
            };
            let request = blocked
                .pending_groups
                .remove(&completed_group.group_id)
                .ok_or_else(|| {
                    VulkanResidencyBackpressureError::scheduler_state(format!(
                        "blocked activation {activation_id} has no request for completed group {:?}",
                        completed_group.group_id
                    ))
                })?;
            let lease = match request {
                VulkanBlockedResidencyGroup::Loading {
                    load_id: expected,
                } if expected == load_id => leader_lease.take().ok_or_else(|| {
                    VulkanResidencyBackpressureError::scheduler_state(
                        "more than one activation claimed the single-flight leader lease",
                    )
                })?,
                VulkanBlockedResidencyGroup::Loading { .. } => {
                    return Err(VulkanResidencyBackpressureError::scheduler_state(
                        "blocked activation references the wrong residency load",
                    ));
                }
                VulkanBlockedResidencyGroup::Waiting(mut waiter) => {
                    match waiter.try_wait() {
                        Ok(Some(lease)) => lease,
                        Ok(None) => {
                            return Err(VulkanResidencyBackpressureError::scheduler_state(
                                "single-flight completion arrived before its joined waiter was published",
                            ));
                        }
                        Err(error) => {
                            self.fail_blocked_activation(
                                activation_id,
                                VulkanResidencyBackpressureError::from(error),
                            );
                            continue;
                        }
                    }
                }
            };
            blocked
                .leases
                .insert(completed_group.group_id.clone(), lease);
            if blocked.pending_groups.is_empty() {
                resume_candidates.push((
                    blocked.admission_sequence,
                    activation_id,
                ));
            }
        }
        drop(leader_lease);
        resume_candidates.sort_unstable();
        for (_, activation_id) in resume_candidates {
            self.resume_blocked_activation(activation_id)?;
        }
        Ok(())
    }

    fn poll_externally_completed_groups(
        &mut self,
    ) -> Result<usize, VulkanResidencyBackpressureError> {
        let locally_owned = self
            .inflight_loads
            .values()
            .map(|load| load.device_group.clone())
            .collect::<BTreeSet<_>>();
        let candidates = self
            .blocked_by_group
            .keys()
            .filter(|group| !locally_owned.contains(*group))
            .cloned()
            .collect::<Vec<_>>();
        let mut completed_count = 0usize;
        for group in candidates {
            if self.poll_externally_completed_group(&group)? {
                completed_count = completed_count.saturating_add(1);
            }
        }
        Ok(completed_count)
    }

    fn poll_externally_completed_group(
        &mut self,
        group: &VulkanResidencyDeviceGroupKey,
    ) -> Result<bool, VulkanResidencyBackpressureError> {
        let affected = self
            .blocked_by_group
            .get(group)
            .cloned()
            .unwrap_or_default();
        let Some(first_activation_id) = affected.iter().next().copied() else {
            self.blocked_by_group.remove(group);
            return Ok(false);
        };
        let first_result = {
            let blocked = self
                .blocked
                .get_mut(&first_activation_id)
                .ok_or_else(|| {
                    VulkanResidencyBackpressureError::scheduler_state(
                        "external residency waiter references an absent activation",
                    )
                })?;
            match blocked.pending_groups.get_mut(&group.group_id) {
                Some(VulkanBlockedResidencyGroup::Waiting(waiter)) => {
                    waiter.try_wait()
                }
                Some(VulkanBlockedResidencyGroup::Loading { .. }) => {
                    return Err(VulkanResidencyBackpressureError::scheduler_state(
                        "external residency group unexpectedly has a local load leader",
                    ));
                }
                None => {
                    return Err(VulkanResidencyBackpressureError::scheduler_state(
                        "external residency waiter is absent from its activation",
                    ));
                }
            }
        };
        let mut first_lease = match first_result {
            Ok(Some(lease)) => Some(lease),
            Ok(None) => return Ok(false),
            Err(error) => {
                self.blocked_by_group.remove(group);
                self.fail_affected_activations(
                    affected,
                    VulkanResidencyBackpressureError::from(error),
                );
                return Ok(true);
            }
        };

        self.blocked_by_group.remove(group);
        let mut resume_candidates = Vec::new();
        for activation_id in affected {
            let Some(blocked) = self.blocked.get_mut(&activation_id) else {
                continue;
            };
            let request = blocked
                .pending_groups
                .remove(&group.group_id)
                .ok_or_else(|| {
                    VulkanResidencyBackpressureError::scheduler_state(
                        "external residency publication lost a dependent request",
                    )
                })?;
            let lease = if activation_id == first_activation_id {
                first_lease.take().expect("first external waiter acquired a lease")
            } else {
                let VulkanBlockedResidencyGroup::Waiting(mut waiter) = request else {
                    return Err(VulkanResidencyBackpressureError::scheduler_state(
                        "external residency publication encountered a local load leader",
                    ));
                };
                match waiter.try_wait() {
                    Ok(Some(lease)) => lease,
                    Ok(None) => {
                        return Err(VulkanResidencyBackpressureError::scheduler_state(
                            "atomic publication did not wake every external waiter",
                        ));
                    }
                    Err(error) => {
                        self.fail_blocked_activation(
                            activation_id,
                            VulkanResidencyBackpressureError::from(error),
                        );
                        continue;
                    }
                }
            };
            blocked.leases.insert(group.group_id.clone(), lease);
            if blocked.pending_groups.is_empty() {
                resume_candidates.push((blocked.admission_sequence, activation_id));
            }
        }
        resume_candidates.sort_unstable();
        for (_, activation_id) in resume_candidates {
            self.resume_blocked_activation(activation_id)?;
        }
        Ok(true)
    }

    fn resume_blocked_activation(
        &mut self,
        activation_id: u64,
    ) -> Result<(), VulkanResidencyBackpressureError> {
        let mut blocked = self
            .blocked
            .remove(&activation_id)
            .ok_or_else(|| {
                VulkanResidencyBackpressureError::scheduler_state(format!(
                    "cannot resume unknown blocked activation {activation_id}"
                ))
            })?;
        if !blocked.pending_groups.is_empty() {
            return Err(VulkanResidencyBackpressureError::scheduler_state(
                "cannot resume an activation with unpublished residency groups",
            ));
        }
        let resident_group_ids = blocked.leases.keys().cloned().collect();
        let status = blocked
            .checkpoint
            .resume_after_atomic_publication(&resident_group_ids)?;
        if status != VulkanPhysicalResidencyActivationStatus::Completed {
            return Err(VulkanResidencyBackpressureError::scheduler_state(
                "physical residency checkpoint did not complete after publication",
            ));
        }
        self.ready.push_back(VulkanResidencyReadyActivation {
            key: blocked.key,
            continuation: blocked.continuation,
            checkpoint: blocked.checkpoint,
            leases: blocked.leases.into_values().collect(),
        });
        Ok(())
    }

    fn fail_blocked_activation(
        &mut self,
        activation_id: u64,
        error: VulkanResidencyBackpressureError,
    ) {
        let Some(blocked) = self.remove_blocked_activation(activation_id) else {
            return;
        };
        self.failed.push_back(VulkanResidencyFailedActivation {
            key: blocked.key,
            continuation: blocked.continuation,
            checkpoint: blocked.checkpoint,
            error,
        });
    }

    fn fail_affected_activations(
        &mut self,
        affected: BTreeSet<u64>,
        error: VulkanResidencyBackpressureError,
    ) {
        let mut ordered = affected
            .into_iter()
            .filter_map(|activation_id| {
                self.blocked
                    .get(&activation_id)
                    .map(|blocked| (blocked.admission_sequence, activation_id))
            })
            .collect::<Vec<_>>();
        ordered.sort_unstable();
        for (_, activation_id) in ordered {
            self.fail_blocked_activation(activation_id, error.clone());
        }
    }

    fn remove_blocked_activation(
        &mut self,
        activation_id: u64,
    ) -> Option<VulkanBlockedResidencyActivation<C, P>> {
        let blocked = self.blocked.remove(&activation_id)?;
        for group_id in blocked.pending_groups.keys() {
            let key = VulkanResidencyDeviceGroupKey {
                device_id: blocked.key.device_id.clone(),
                group_id: group_id.clone(),
            };
            let remove_key = if let Some(waiters) = self.blocked_by_group.get_mut(&key) {
                waiters.remove(&activation_id);
                waiters.is_empty()
            } else {
                false
            };
            if remove_key {
                self.blocked_by_group.remove(&key);
            }
        }
        Some(blocked)
    }
}
