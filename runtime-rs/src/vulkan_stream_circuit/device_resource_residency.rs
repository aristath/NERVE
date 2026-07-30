use std::sync::{Condvar as StdCondvar, Mutex as StdMutex, Weak};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceResourceDescriptor {
    pub id: String,
    pub byte_count: usize,
    pub compatibility: CompiledResourceCompatibility,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceResourceGroupDescriptor {
    pub id: String,
    pub resource_ids: Vec<String>,
    pub dependencies: Vec<String>,
    pub resources: Vec<DeviceResourceDescriptor>,
    pub byte_count: usize,
}

impl DeviceResourceGroupDescriptor {
    pub fn from_resolved(
        group: &ResolvedCompiledResourceGroup,
    ) -> Result<Self, DeviceResourceResidencyError> {
        let resources = group
            .resources()
            .iter()
            .map(|resource| {
                let byte_count =
                    resource.ranges.iter().try_fold(0usize, |total, range| {
                        total.checked_add(range.byte_count).ok_or_else(|| {
                            DeviceResourceResidencyError::invalid_descriptor(
                                "compiled resource byte count overflowed",
                            )
                        })
                    })?;
                Ok(DeviceResourceDescriptor {
                    id: resource.id.clone(),
                    byte_count,
                    compatibility: resource.compatibility.clone(),
                })
            })
            .collect::<Result<Vec<_>, DeviceResourceResidencyError>>()?;
        Self::new(
            group.id().to_string(),
            group.resource_ids().to_vec(),
            group.dependencies().to_vec(),
            resources,
        )
    }

    pub fn new(
        id: String,
        resource_ids: Vec<String>,
        dependencies: Vec<String>,
        resources: Vec<DeviceResourceDescriptor>,
    ) -> Result<Self, DeviceResourceResidencyError> {
        package::validate_content_id("device residency group id", &id)
            .map_err(|error| {
                DeviceResourceResidencyError::invalid_descriptor(
                    error.to_string(),
                )
            })?;
        if resources.is_empty() {
            return Err(DeviceResourceResidencyError::invalid_descriptor(
                "device residency group has no resources",
            ));
        }
        let actual_resource_ids = resources
            .iter()
            .map(|resource| resource.id.clone())
            .collect::<Vec<_>>();
        if resource_ids != actual_resource_ids
            || resource_ids
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(DeviceResourceResidencyError::invalid_descriptor(
                "device residency group resources must match sorted resource ids",
            ));
        }
        if dependencies
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            return Err(DeviceResourceResidencyError::invalid_descriptor(
                "device residency group dependencies are not strictly sorted",
            ));
        }
        for dependency in &dependencies {
            package::validate_content_id(
                "device residency dependency id",
                dependency,
            )
            .map_err(|error| {
                DeviceResourceResidencyError::invalid_descriptor(
                    error.to_string(),
                )
            })?;
        }
        let byte_count = resources.iter().try_fold(0usize, |total, resource| {
            package::validate_content_id(
                "device residency resource id",
                &resource.id,
            )
            .map_err(|error| {
                DeviceResourceResidencyError::invalid_descriptor(
                    error.to_string(),
                )
            })?;
            if resource.byte_count == 0
                || resource.compatibility.device_api.trim().is_empty()
                || resource.compatibility.storage_class.trim().is_empty()
                || !resource.compatibility.read_only
            {
                return Err(DeviceResourceResidencyError::invalid_descriptor(
                    "device residency resource descriptor is invalid",
                ));
            }
            total.checked_add(resource.byte_count).ok_or_else(|| {
                DeviceResourceResidencyError::invalid_descriptor(
                    "device residency group byte count overflowed",
                )
            })
        })?;
        Ok(Self {
            id,
            resource_ids,
            dependencies,
            resources,
            byte_count,
        })
    }

    fn validate_integrity(
        &self,
    ) -> Result<(), DeviceResourceResidencyError> {
        let canonical = Self::new(
            self.id.clone(),
            self.resource_ids.clone(),
            self.dependencies.clone(),
            self.resources.clone(),
        )?;
        if canonical != *self {
            return Err(DeviceResourceResidencyError::invalid_descriptor(
                "device residency group derived fields were modified",
            ));
        }
        Ok(())
    }
}

pub trait DeviceResidentResourcePayload: Send + Sync + 'static {
    fn byte_count(&self) -> usize;
}

pub struct DeviceResidentResource<P: DeviceResidentResourcePayload> {
    descriptor: DeviceResourceDescriptor,
    payload: Arc<P>,
}

impl<P: DeviceResidentResourcePayload> DeviceResidentResource<P> {
    pub fn new(
        descriptor: DeviceResourceDescriptor,
        payload: P,
    ) -> Result<Self, DeviceResourceResidencyError> {
        if payload.byte_count() != descriptor.byte_count {
            return Err(DeviceResourceResidencyError::invalid_publication(
                "resident payload size does not match its immutable resource descriptor",
            ));
        }
        Ok(Self {
            descriptor,
            payload: Arc::new(payload),
        })
    }

    pub fn descriptor(&self) -> &DeviceResourceDescriptor {
        &self.descriptor
    }

    pub fn payload(&self) -> &P {
        &self.payload
    }
}

pub struct DeviceResidentResourceGroup<P: DeviceResidentResourcePayload> {
    descriptor: DeviceResourceGroupDescriptor,
    resources: Vec<DeviceResidentResource<P>>,
}

impl<P: DeviceResidentResourcePayload> DeviceResidentResourceGroup<P> {
    pub fn new(
        descriptor: DeviceResourceGroupDescriptor,
        resources: Vec<DeviceResidentResource<P>>,
    ) -> Result<Self, DeviceResourceResidencyError> {
        descriptor.validate_integrity()?;
        if resources.len() != descriptor.resources.len() {
            return Err(DeviceResourceResidencyError::invalid_publication(
                "resident resource count does not match the atomic group",
            ));
        }
        for (resident, expected) in resources.iter().zip(&descriptor.resources)
        {
            if resident.descriptor != *expected
                || resident.payload.byte_count() != expected.byte_count
            {
                return Err(DeviceResourceResidencyError::invalid_publication(
                    "resident resource identity, compatibility, or size does not match the atomic group",
                ));
            }
        }
        Ok(Self {
            descriptor,
            resources,
        })
    }

    pub fn descriptor(&self) -> &DeviceResourceGroupDescriptor {
        &self.descriptor
    }

    pub fn resources(&self) -> &[DeviceResidentResource<P>] {
        &self.resources
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DeviceResourceResidencyOwnerId(String);

impl DeviceResourceResidencyOwnerId {
    pub fn new(value: impl Into<String>) -> Result<Self, DeviceResourceResidencyError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(DeviceResourceResidencyError::invalid_descriptor(
                "device residency owner id is empty",
            ));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceResourceResidencyErrorKind {
    Backpressure,
    Cancelled,
    Capacity,
    Failed,
    IdentityConflict,
    InUse,
    InvalidDescriptor,
    InvalidPublication,
    StaleOperation,
    Stopped,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceResourceResidencyError {
    kind: DeviceResourceResidencyErrorKind,
    message: String,
}

impl DeviceResourceResidencyError {
    fn new(
        kind: DeviceResourceResidencyErrorKind,
        message: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    fn invalid_descriptor(message: impl Into<String>) -> Self {
        Self::new(
            DeviceResourceResidencyErrorKind::InvalidDescriptor,
            message,
        )
    }

    fn invalid_publication(message: impl Into<String>) -> Self {
        Self::new(
            DeviceResourceResidencyErrorKind::InvalidPublication,
            message,
        )
    }

    pub fn load_failed(message: impl Into<String>) -> Self {
        Self::new(DeviceResourceResidencyErrorKind::Failed, message)
    }

    pub fn kind(&self) -> DeviceResourceResidencyErrorKind {
        self.kind
    }
}

impl Display for DeviceResourceResidencyError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for DeviceResourceResidencyError {}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DeviceResourceResidencyLocation {
    Local { device_id: String },
    Remote { device_id: String },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct DeviceResourceResidencyDirectoryEntry {
    pub group_id: String,
    pub state: ResourceResidencyState,
    pub location: DeviceResourceResidencyLocation,
    pub byte_count: usize,
    pub owner_count: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct DeviceResourceResidencyStatistics {
    pub capacity_bytes: usize,
    pub always_resident_bytes: usize,
    pub reserved_loading_bytes: usize,
    pub dynamic_resident_bytes: usize,
    pub high_water_dynamic_resident_bytes: usize,
    pub loading_group_count: usize,
    pub resident_group_count: usize,
    pub high_water_resident_group_count: usize,
    pub failed_group_count: usize,
    pub hit_count: u64,
    pub miss_count: u64,
    pub single_flight_join_count: u64,
    pub successful_load_count: u64,
    pub failed_load_count: u64,
    pub cancelled_load_count: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct DeviceResourceResidencySnapshot {
    pub statistics: DeviceResourceResidencyStatistics,
    pub directory: Vec<DeviceResourceResidencyDirectoryEntry>,
}

struct DeviceResourceLoadOutcome {
    value: StdMutex<Option<Result<(), DeviceResourceResidencyError>>>,
    changed: StdCondvar,
}

impl DeviceResourceLoadOutcome {
    fn new() -> Self {
        Self {
            value: StdMutex::new(None),
            changed: StdCondvar::new(),
        }
    }

    fn finish(
        &self,
        value: Result<(), DeviceResourceResidencyError>,
    ) {
        if let Ok(mut outcome) = self.value.lock()
            && outcome.is_none()
        {
            *outcome = Some(value);
            self.changed.notify_all();
        }
    }

    fn wait(
        &self,
    ) -> Result<(), DeviceResourceResidencyError> {
        let mut outcome = self.value.lock().map_err(|_| {
            DeviceResourceResidencyError::new(
                DeviceResourceResidencyErrorKind::Stopped,
                "device residency load outcome was poisoned",
            )
        })?;
        loop {
            if let Some(result) = outcome.as_ref() {
                return result.clone();
            }
            outcome = self.changed.wait(outcome).map_err(|_| {
                DeviceResourceResidencyError::new(
                    DeviceResourceResidencyErrorKind::Stopped,
                    "device residency load outcome was poisoned",
                )
            })?;
        }
    }

    fn try_result(
        &self,
    ) -> Result<Option<Result<(), DeviceResourceResidencyError>>, DeviceResourceResidencyError>
    {
        self.value.lock().map(|outcome| outcome.clone()).map_err(|_| {
            DeviceResourceResidencyError::new(
                DeviceResourceResidencyErrorKind::Stopped,
                "device residency load outcome was poisoned",
            )
        })
    }
}

enum DeviceResourceResidencyEntry<P: DeviceResidentResourcePayload> {
    Loading {
        descriptor: DeviceResourceGroupDescriptor,
        operation_id: u64,
        owners: BTreeSet<DeviceResourceResidencyOwnerId>,
        outcome: Arc<DeviceResourceLoadOutcome>,
    },
    Resident {
        descriptor: DeviceResourceGroupDescriptor,
        owners: BTreeSet<DeviceResourceResidencyOwnerId>,
        active_leases: BTreeMap<DeviceResourceResidencyOwnerId, usize>,
        group: Arc<DeviceResidentResourceGroup<P>>,
    },
    Failed {
        descriptor: DeviceResourceGroupDescriptor,
        error: DeviceResourceResidencyError,
    },
}

struct DeviceResourceResidencyState<P: DeviceResidentResourcePayload> {
    next_operation_id: u64,
    entries: BTreeMap<String, DeviceResourceResidencyEntry<P>>,
    reserved_loading_bytes: usize,
    dynamic_resident_bytes: usize,
    high_water_dynamic_resident_bytes: usize,
    high_water_resident_group_count: usize,
    hit_count: u64,
    miss_count: u64,
    single_flight_join_count: u64,
    successful_load_count: u64,
    failed_load_count: u64,
    cancelled_load_count: u64,
}

impl<P: DeviceResidentResourcePayload> Default
    for DeviceResourceResidencyState<P>
{
    fn default() -> Self {
        Self {
            next_operation_id: 0,
            entries: BTreeMap::new(),
            reserved_loading_bytes: 0,
            dynamic_resident_bytes: 0,
            high_water_dynamic_resident_bytes: 0,
            high_water_resident_group_count: 0,
            hit_count: 0,
            miss_count: 0,
            single_flight_join_count: 0,
            successful_load_count: 0,
            failed_load_count: 0,
            cancelled_load_count: 0,
        }
    }
}

struct DeviceResourceResidencyManagerInner<P: DeviceResidentResourcePayload> {
    device_id: String,
    capacity_bytes: usize,
    always_resident_bytes: usize,
    state: StdMutex<DeviceResourceResidencyState<P>>,
}

pub struct DeviceResourceResidencyManager<P: DeviceResidentResourcePayload> {
    inner: Arc<DeviceResourceResidencyManagerInner<P>>,
}

impl<P: DeviceResidentResourcePayload> Clone
    for DeviceResourceResidencyManager<P>
{
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<P: DeviceResidentResourcePayload> DeviceResourceResidencyManager<P> {
    pub fn new(
        device_id: impl Into<String>,
        capacity_bytes: usize,
        always_resident_bytes: usize,
    ) -> Result<Self, DeviceResourceResidencyError> {
        let device_id = device_id.into();
        if device_id.trim().is_empty()
            || capacity_bytes == 0
            || always_resident_bytes > capacity_bytes
        {
            return Err(DeviceResourceResidencyError::invalid_descriptor(
                "per-device residency capacity or device identity is invalid",
            ));
        }
        Ok(Self {
            inner: Arc::new(DeviceResourceResidencyManagerInner {
                device_id,
                capacity_bytes,
                always_resident_bytes,
                state: StdMutex::new(DeviceResourceResidencyState::default()),
            }),
        })
    }

    pub fn device_id(&self) -> &str {
        &self.inner.device_id
    }

    pub fn request(
        &self,
        descriptor: DeviceResourceGroupDescriptor,
        owner: DeviceResourceResidencyOwnerId,
    ) -> Result<DeviceResourceResidencyRequest<P>, DeviceResourceResidencyError>
    {
        let mut requests = self.request_batch([descriptor], owner)?;
        Ok(requests
            .pop()
            .expect("a one-element residency batch returns one request"))
    }

    pub fn request_batch<I>(
        &self,
        descriptors: I,
        owner: DeviceResourceResidencyOwnerId,
    ) -> Result<Vec<DeviceResourceResidencyRequest<P>>, DeviceResourceResidencyError>
    where
        I: IntoIterator<Item = DeviceResourceGroupDescriptor>,
    {
        self.request_batch_with_new_load_limit(descriptors, owner, usize::MAX)
    }

    pub fn request_batch_with_new_load_limit<I>(
        &self,
        descriptors: I,
        owner: DeviceResourceResidencyOwnerId,
        maximum_new_loads: usize,
    ) -> Result<Vec<DeviceResourceResidencyRequest<P>>, DeviceResourceResidencyError>
    where
        I: IntoIterator<Item = DeviceResourceGroupDescriptor>,
    {
        let descriptors = descriptors.into_iter().collect::<Vec<_>>();
        if descriptors.is_empty() {
            return Err(DeviceResourceResidencyError::invalid_descriptor(
                "device residency request batch is empty",
            ));
        }
        for descriptor in &descriptors {
            descriptor.validate_integrity()?;
        }
        if descriptors
            .windows(2)
            .any(|pair| pair[0].id >= pair[1].id)
        {
            return Err(DeviceResourceResidencyError::invalid_descriptor(
                "device residency request batch groups must be unique and strictly sorted",
            ));
        }

        let mut state = self.inner.state.lock().map_err(|_| {
            DeviceResourceResidencyError::new(
                DeviceResourceResidencyErrorKind::Stopped,
                "per-device residency manager was poisoned",
            )
        })?;

        let mut new_group_bytes = 0usize;
        let mut new_group_count = 0usize;
        for descriptor in &descriptors {
            let Some(entry) = state.entries.get(&descriptor.id) else {
                new_group_bytes = new_group_bytes
                    .checked_add(descriptor.byte_count)
                    .ok_or_else(|| {
                        DeviceResourceResidencyError::new(
                            DeviceResourceResidencyErrorKind::Capacity,
                            "per-device residency request batch byte count overflowed",
                        )
                    })?;
                new_group_count = new_group_count.checked_add(1).ok_or_else(|| {
                    DeviceResourceResidencyError::new(
                        DeviceResourceResidencyErrorKind::Stopped,
                        "per-device residency operation count overflowed",
                    )
                })?;
                continue;
            };
            let actual_descriptor = match entry {
                DeviceResourceResidencyEntry::Loading {
                    descriptor, ..
                }
                | DeviceResourceResidencyEntry::Resident {
                    descriptor, ..
                }
                | DeviceResourceResidencyEntry::Failed {
                    descriptor, ..
                } => descriptor,
            };
            if *actual_descriptor != *descriptor {
                return Err(DeviceResourceResidencyError::new(
                    DeviceResourceResidencyErrorKind::IdentityConflict,
                    "compiled group identity was reused with different physical resources",
                ));
            }
            if let DeviceResourceResidencyEntry::Failed { error, .. } = entry {
                return Err(error.clone());
            }
        }
        if new_group_count > maximum_new_loads {
            return Err(DeviceResourceResidencyError::new(
                DeviceResourceResidencyErrorKind::Backpressure,
                format!(
                    "device {:?} residency batch needs {new_group_count} new loads but the scheduler has capacity for {maximum_new_loads}",
                    self.inner.device_id
                ),
            ));
        }

        let used_bytes = self
            .inner
            .always_resident_bytes
            .checked_add(state.dynamic_resident_bytes)
            .and_then(|used| used.checked_add(state.reserved_loading_bytes))
            .ok_or_else(|| {
                DeviceResourceResidencyError::new(
                    DeviceResourceResidencyErrorKind::Capacity,
                    "per-device residency accounting overflowed",
                )
            })?;
        let required_bytes = used_bytes.checked_add(new_group_bytes).ok_or_else(|| {
                DeviceResourceResidencyError::new(
                    DeviceResourceResidencyErrorKind::Capacity,
                    "per-device residency request byte count overflowed",
                )
            })?;
        if required_bytes > self.inner.capacity_bytes {
            return Err(DeviceResourceResidencyError::new(
                DeviceResourceResidencyErrorKind::Capacity,
                format!(
                    "device {:?} needs {} bytes for residency batch, but capacity is {} bytes",
                    self.inner.device_id,
                    required_bytes,
                    self.inner.capacity_bytes
                ),
            ));
        }
        state
            .next_operation_id
            .checked_add(u64::try_from(new_group_count).map_err(|_| {
                DeviceResourceResidencyError::new(
                    DeviceResourceResidencyErrorKind::Stopped,
                    "per-device residency operation count does not fit u64",
                )
            })?)
            .ok_or_else(|| {
                DeviceResourceResidencyError::new(
                    DeviceResourceResidencyErrorKind::Stopped,
                    "per-device residency operation identity exhausted",
                )
            })?;

        let mut requests = Vec::with_capacity(descriptors.len());
        for descriptor in descriptors {
            if let Some(entry) = state.entries.get_mut(&descriptor.id) {
                match entry {
                    DeviceResourceResidencyEntry::Loading {
                        owners, outcome, ..
                    } => {
                        owners.insert(owner.clone());
                        let outcome = Arc::clone(outcome);
                        state.single_flight_join_count =
                            state.single_flight_join_count.saturating_add(1);
                        requests.push(DeviceResourceResidencyRequest::Pending(
                            DeviceResourceResidencyWaiter {
                                manager: Arc::downgrade(&self.inner),
                                group_id: descriptor.id,
                                owner: owner.clone(),
                                outcome,
                            },
                        ));
                    }
                    DeviceResourceResidencyEntry::Resident {
                        owners,
                        active_leases,
                        group,
                        ..
                    } => {
                        owners.insert(owner.clone());
                        *active_leases.entry(owner.clone()).or_default() += 1;
                        let group = Arc::clone(group);
                        state.hit_count = state.hit_count.saturating_add(1);
                        requests.push(DeviceResourceResidencyRequest::Resident(
                            DeviceResourceResidencyLease {
                                manager: Arc::downgrade(&self.inner),
                                group_id: descriptor.id,
                                owner: owner.clone(),
                                group,
                            },
                        ));
                    }
                    DeviceResourceResidencyEntry::Failed { .. } => {
                        unreachable!("failed entries were rejected before batch mutation")
                    }
                }
                continue;
            }

            debug_assert!(
                ResourceResidencyState::Absent
                    .can_transition_to(ResourceResidencyState::Requested, false)
            );
            debug_assert!(
                ResourceResidencyState::Requested
                    .can_transition_to(ResourceResidencyState::Loading, false)
            );
            state.next_operation_id += 1;
            let operation_id = state.next_operation_id;
            state.reserved_loading_bytes += descriptor.byte_count;
            state.miss_count = state.miss_count.saturating_add(1);
            let outcome = Arc::new(DeviceResourceLoadOutcome::new());
            let mut owners = BTreeSet::new();
            owners.insert(owner.clone());
            state.entries.insert(
                descriptor.id.clone(),
                DeviceResourceResidencyEntry::Loading {
                    descriptor: descriptor.clone(),
                    operation_id,
                    owners,
                    outcome: Arc::clone(&outcome),
                },
            );
            requests.push(DeviceResourceResidencyRequest::LoadRequired(
                DeviceResourceLoadPermit {
                    manager: Arc::downgrade(&self.inner),
                    descriptor,
                    operation_id,
                    owner: owner.clone(),
                    outcome,
                    finished: false,
                },
            ));
        }
        Ok(requests)
    }

    pub fn reset_failed_group(
        &self,
        group_id: &str,
    ) -> Result<(), DeviceResourceResidencyError> {
        let mut state = self.inner.state.lock().map_err(|_| {
            DeviceResourceResidencyError::new(
                DeviceResourceResidencyErrorKind::Stopped,
                "per-device residency manager was poisoned",
            )
        })?;
        match state.entries.get(group_id) {
            Some(DeviceResourceResidencyEntry::Failed { .. }) => {
                debug_assert!(
                    ResourceResidencyState::Failed.can_transition_to(
                        ResourceResidencyState::Absent,
                        true
                    )
                );
                state.entries.remove(group_id);
                Ok(())
            }
            Some(_) => Err(DeviceResourceResidencyError::new(
                DeviceResourceResidencyErrorKind::InUse,
                "only a failed residency group can be explicitly reset",
            )),
            None => Ok(()),
        }
    }

    pub fn unload_owner(
        &self,
        owner: &DeviceResourceResidencyOwnerId,
    ) -> Result<DeviceResourceResidencyRelease, DeviceResourceResidencyError>
    {
        unload_owner_from_manager(&self.inner, owner)
    }

    pub fn unload_device(
        &self,
    ) -> Result<DeviceResourceResidencyRelease, DeviceResourceResidencyError>
    {
        unload_entire_manager(&self.inner)
    }

    pub fn statistics(
        &self,
    ) -> Result<DeviceResourceResidencyStatistics, DeviceResourceResidencyError>
    {
        self.snapshot().map(|snapshot| snapshot.statistics)
    }

    pub fn directory(
        &self,
    ) -> Result<Vec<DeviceResourceResidencyDirectoryEntry>, DeviceResourceResidencyError>
    {
        self.snapshot().map(|snapshot| snapshot.directory)
    }

    pub fn snapshot(
        &self,
    ) -> Result<DeviceResourceResidencySnapshot, DeviceResourceResidencyError>
    {
        let state = self.inner.state.lock().map_err(|_| {
            DeviceResourceResidencyError::new(
                DeviceResourceResidencyErrorKind::Stopped,
                "per-device residency manager was poisoned",
            )
        })?;
        Ok(DeviceResourceResidencySnapshot {
            statistics: statistics_for_state(&self.inner, &state),
            directory: directory_for_state(&self.inner, &state),
        })
    }
}

pub enum DeviceResourceResidencyRequest<P: DeviceResidentResourcePayload> {
    Resident(DeviceResourceResidencyLease<P>),
    LoadRequired(DeviceResourceLoadPermit<P>),
    Pending(DeviceResourceResidencyWaiter<P>),
}

pub struct DeviceResourceResidencyLease<P: DeviceResidentResourcePayload> {
    manager: Weak<DeviceResourceResidencyManagerInner<P>>,
    group_id: String,
    owner: DeviceResourceResidencyOwnerId,
    group: Arc<DeviceResidentResourceGroup<P>>,
}

impl<P: DeviceResidentResourcePayload> DeviceResourceResidencyLease<P> {
    pub fn owner(&self) -> &DeviceResourceResidencyOwnerId {
        &self.owner
    }

    pub fn group(&self) -> &DeviceResidentResourceGroup<P> {
        &self.group
    }

    pub fn shares_publication_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.group, &other.group)
    }
}

impl<P: DeviceResidentResourcePayload> Drop
    for DeviceResourceResidencyLease<P>
{
    fn drop(&mut self) {
        if let Some(manager) = self.manager.upgrade() {
            release_active_lease(&manager, &self.group_id, &self.owner);
        }
    }
}

pub struct DeviceResourceResidencyWaiter<P: DeviceResidentResourcePayload> {
    manager: Weak<DeviceResourceResidencyManagerInner<P>>,
    group_id: String,
    owner: DeviceResourceResidencyOwnerId,
    outcome: Arc<DeviceResourceLoadOutcome>,
}

impl<P: DeviceResidentResourcePayload> DeviceResourceResidencyWaiter<P> {
    pub fn wait(
        self,
    ) -> Result<DeviceResourceResidencyLease<P>, DeviceResourceResidencyError>
    {
        self.outcome.wait()?;
        let manager = self.manager.upgrade().ok_or_else(|| {
            DeviceResourceResidencyError::new(
                DeviceResourceResidencyErrorKind::Cancelled,
                "per-device residency manager was dropped before a waiter resumed",
            )
        })?;
        acquire_published_lease(&manager, &self.group_id, &self.owner)
    }

    pub fn try_wait(
        &mut self,
    ) -> Result<Option<DeviceResourceResidencyLease<P>>, DeviceResourceResidencyError> {
        let Some(outcome) = self.outcome.try_result()? else {
            return Ok(None);
        };
        outcome?;
        let manager = self.manager.upgrade().ok_or_else(|| {
            DeviceResourceResidencyError::new(
                DeviceResourceResidencyErrorKind::Cancelled,
                "per-device residency manager was dropped before a waiter resumed",
            )
        })?;
        acquire_published_lease(&manager, &self.group_id, &self.owner).map(Some)
    }
}

pub struct DeviceResourceLoadPermit<P: DeviceResidentResourcePayload> {
    manager: Weak<DeviceResourceResidencyManagerInner<P>>,
    descriptor: DeviceResourceGroupDescriptor,
    operation_id: u64,
    owner: DeviceResourceResidencyOwnerId,
    outcome: Arc<DeviceResourceLoadOutcome>,
    finished: bool,
}

impl<P: DeviceResidentResourcePayload> DeviceResourceLoadPermit<P> {
    pub fn descriptor(&self) -> &DeviceResourceGroupDescriptor {
        &self.descriptor
    }

    pub fn publish(
        mut self,
        group: DeviceResidentResourceGroup<P>,
    ) -> Result<DeviceResourceResidencyLease<P>, DeviceResourceResidencyError>
    {
        let manager = self.manager.upgrade().ok_or_else(|| {
            DeviceResourceResidencyError::new(
                DeviceResourceResidencyErrorKind::Stopped,
                "per-device residency manager was dropped during a load",
            )
        })?;
        let result = publish_loaded_group(
            &manager,
            self.operation_id,
            &self.descriptor,
            group,
        );
        self.finished = true;
        match result {
            Ok(()) => acquire_published_lease(
                &manager,
                &self.descriptor.id,
                &self.owner,
            ),
            Err(error) => Err(error),
        }
    }

    pub fn fail(
        mut self,
        error: DeviceResourceResidencyError,
    ) -> Result<(), DeviceResourceResidencyError> {
        let manager = self.manager.upgrade().ok_or_else(|| {
            DeviceResourceResidencyError::new(
                DeviceResourceResidencyErrorKind::Stopped,
                "per-device residency manager was dropped during a load",
            )
        })?;
        fail_loaded_group(
            &manager,
            self.operation_id,
            &self.descriptor,
            error,
        )?;
        self.finished = true;
        Ok(())
    }

    pub fn cancel(
        mut self,
    ) -> Result<(), DeviceResourceResidencyError> {
        let manager = self.manager.upgrade().ok_or_else(|| {
            DeviceResourceResidencyError::new(
                DeviceResourceResidencyErrorKind::Stopped,
                "per-device residency manager was dropped during a load",
            )
        })?;
        cancel_loaded_group(
            &manager,
            self.operation_id,
            &self.descriptor.id,
        )?;
        self.finished = true;
        Ok(())
    }
}

impl<P: DeviceResidentResourcePayload> Drop
    for DeviceResourceLoadPermit<P>
{
    fn drop(&mut self) {
        if !self.finished {
            if let Some(manager) = self.manager.upgrade() {
                let _ = cancel_loaded_group(
                    &manager,
                    self.operation_id,
                    &self.descriptor.id,
                );
            } else {
                self.outcome.finish(Err(
                    DeviceResourceResidencyError::new(
                        DeviceResourceResidencyErrorKind::Cancelled,
                        "per-device residency load was cancelled",
                    ),
                ));
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceResourceResidencyRelease {
    pub group_count: usize,
    pub byte_count: usize,
    pub cancelled_load_count: usize,
}

fn acquire_published_lease<P: DeviceResidentResourcePayload>(
    manager: &Arc<DeviceResourceResidencyManagerInner<P>>,
    group_id: &str,
    owner: &DeviceResourceResidencyOwnerId,
) -> Result<DeviceResourceResidencyLease<P>, DeviceResourceResidencyError> {
    let mut state = manager.state.lock().map_err(|_| {
        DeviceResourceResidencyError::new(
            DeviceResourceResidencyErrorKind::Stopped,
            "per-device residency manager was poisoned",
        )
    })?;
    let entry = state.entries.get_mut(group_id).ok_or_else(|| {
        DeviceResourceResidencyError::new(
            DeviceResourceResidencyErrorKind::Cancelled,
            "residency owner was unloaded before the request resumed",
        )
    })?;
    let (active_leases, group) = match entry {
        DeviceResourceResidencyEntry::Resident {
            owners,
            active_leases,
            group,
            ..
        } if owners.contains(owner) => (active_leases, group),
        _ => {
            return Err(DeviceResourceResidencyError::new(
                DeviceResourceResidencyErrorKind::Cancelled,
                "residency owner was unloaded before the request resumed",
            ));
        }
    };
    *active_leases.entry(owner.clone()).or_default() += 1;
    Ok(DeviceResourceResidencyLease {
        manager: Arc::downgrade(manager),
        group_id: group_id.to_string(),
        owner: owner.clone(),
        group: Arc::clone(group),
    })
}

fn release_active_lease<P: DeviceResidentResourcePayload>(
    manager: &Arc<DeviceResourceResidencyManagerInner<P>>,
    group_id: &str,
    owner: &DeviceResourceResidencyOwnerId,
) {
    let Ok(mut state) = manager.state.lock() else {
        return;
    };
    let Some(DeviceResourceResidencyEntry::Resident {
        active_leases, ..
    }) = state.entries.get_mut(group_id)
    else {
        return;
    };
    if let Some(count) = active_leases.get_mut(owner) {
        *count = count.saturating_sub(1);
        if *count == 0 {
            active_leases.remove(owner);
        }
    }
}

fn publish_loaded_group<P: DeviceResidentResourcePayload>(
    manager: &Arc<DeviceResourceResidencyManagerInner<P>>,
    operation_id: u64,
    descriptor: &DeviceResourceGroupDescriptor,
    group: DeviceResidentResourceGroup<P>,
) -> Result<(), DeviceResourceResidencyError> {
    if group.descriptor != *descriptor {
        let error = DeviceResourceResidencyError::invalid_publication(
            "published group descriptor does not match the reserved atomic load",
        );
        fail_loaded_group(manager, operation_id, descriptor, error.clone())?;
        return Err(error);
    }
    let group = Arc::new(group);
    let mut state = manager.state.lock().map_err(|_| {
        DeviceResourceResidencyError::new(
            DeviceResourceResidencyErrorKind::Stopped,
            "per-device residency manager was poisoned",
        )
    })?;
    let published_dynamic_bytes = state
        .dynamic_resident_bytes
        .checked_add(descriptor.byte_count)
        .ok_or_else(|| {
            DeviceResourceResidencyError::new(
                DeviceResourceResidencyErrorKind::Capacity,
                "per-device resident byte accounting overflowed",
            )
        })?;
    let entry = state.entries.remove(&descriptor.id).ok_or_else(|| {
        DeviceResourceResidencyError::new(
            DeviceResourceResidencyErrorKind::StaleOperation,
            "residency load no longer owns its atomic group",
        )
    })?;
    let (owners, outcome) = match entry {
        DeviceResourceResidencyEntry::Loading {
            descriptor: actual,
            operation_id: actual_operation,
            owners,
            outcome,
        } if actual == *descriptor && actual_operation == operation_id => {
            (owners, outcome)
        }
        other => {
            state.entries.insert(descriptor.id.clone(), other);
            return Err(DeviceResourceResidencyError::new(
                DeviceResourceResidencyErrorKind::StaleOperation,
                "residency load operation identity is stale",
            ));
        }
    };
    debug_assert!(
        ResourceResidencyState::Loading
            .can_transition_to(ResourceResidencyState::Resident, false)
    );
    state.reserved_loading_bytes = state
        .reserved_loading_bytes
        .saturating_sub(descriptor.byte_count);
    state.dynamic_resident_bytes = published_dynamic_bytes;
    state.high_water_dynamic_resident_bytes = state
        .high_water_dynamic_resident_bytes
        .max(published_dynamic_bytes);
    state.successful_load_count =
        state.successful_load_count.saturating_add(1);
    state.entries.insert(
        descriptor.id.clone(),
        DeviceResourceResidencyEntry::Resident {
            descriptor: descriptor.clone(),
            owners,
            active_leases: BTreeMap::new(),
            group: Arc::clone(&group),
        },
    );
    state.high_water_resident_group_count = state
        .high_water_resident_group_count
        .max(
            state
                .entries
                .values()
                .filter(|entry| {
                    matches!(
                        entry,
                        DeviceResourceResidencyEntry::Resident { .. }
                    )
                })
                .count(),
        );
    drop(state);
    outcome.finish(Ok(()));
    Ok(())
}

fn fail_loaded_group<P: DeviceResidentResourcePayload>(
    manager: &Arc<DeviceResourceResidencyManagerInner<P>>,
    operation_id: u64,
    descriptor: &DeviceResourceGroupDescriptor,
    error: DeviceResourceResidencyError,
) -> Result<(), DeviceResourceResidencyError> {
    let mut state = manager.state.lock().map_err(|_| {
        DeviceResourceResidencyError::new(
            DeviceResourceResidencyErrorKind::Stopped,
            "per-device residency manager was poisoned",
        )
    })?;
    let entry = state.entries.remove(&descriptor.id).ok_or_else(|| {
        DeviceResourceResidencyError::new(
            DeviceResourceResidencyErrorKind::StaleOperation,
            "residency load no longer owns its atomic group",
        )
    })?;
    let outcome = match entry {
        DeviceResourceResidencyEntry::Loading {
            descriptor: actual,
            operation_id: actual_operation,
            outcome,
            ..
        } if actual == *descriptor && actual_operation == operation_id => outcome,
        other => {
            state.entries.insert(descriptor.id.clone(), other);
            return Err(DeviceResourceResidencyError::new(
                DeviceResourceResidencyErrorKind::StaleOperation,
                "residency load operation identity is stale",
            ));
        }
    };
    debug_assert!(
        ResourceResidencyState::Loading
            .can_transition_to(ResourceResidencyState::Failed, false)
    );
    state.reserved_loading_bytes = state
        .reserved_loading_bytes
        .saturating_sub(descriptor.byte_count);
    state.failed_load_count = state.failed_load_count.saturating_add(1);
    state.entries.insert(
        descriptor.id.clone(),
        DeviceResourceResidencyEntry::Failed {
            descriptor: descriptor.clone(),
            error: error.clone(),
        },
    );
    drop(state);
    outcome.finish(Err(error));
    Ok(())
}

fn cancel_loaded_group<P: DeviceResidentResourcePayload>(
    manager: &Arc<DeviceResourceResidencyManagerInner<P>>,
    operation_id: u64,
    group_id: &str,
) -> Result<(), DeviceResourceResidencyError> {
    let mut state = manager.state.lock().map_err(|_| {
        DeviceResourceResidencyError::new(
            DeviceResourceResidencyErrorKind::Stopped,
            "per-device residency manager was poisoned",
        )
    })?;
    let entry = state.entries.remove(group_id).ok_or_else(|| {
        DeviceResourceResidencyError::new(
            DeviceResourceResidencyErrorKind::StaleOperation,
            "residency load no longer owns its atomic group",
        )
    })?;
    let (descriptor, outcome) = match entry {
        DeviceResourceResidencyEntry::Loading {
            descriptor,
            operation_id: actual_operation,
            outcome,
            ..
        } if actual_operation == operation_id => (descriptor, outcome),
        other => {
            state.entries.insert(group_id.to_string(), other);
            return Err(DeviceResourceResidencyError::new(
                DeviceResourceResidencyErrorKind::StaleOperation,
                "residency load operation identity is stale",
            ));
        }
    };
    debug_assert!(
        ResourceResidencyState::Loading
            .can_transition_to(ResourceResidencyState::Absent, false)
    );
    state.reserved_loading_bytes = state
        .reserved_loading_bytes
        .saturating_sub(descriptor.byte_count);
    state.cancelled_load_count =
        state.cancelled_load_count.saturating_add(1);
    drop(state);
    outcome.finish(Err(DeviceResourceResidencyError::new(
        DeviceResourceResidencyErrorKind::Cancelled,
        "per-device residency load was cancelled",
    )));
    Ok(())
}

fn unload_owner_from_manager<P: DeviceResidentResourcePayload>(
    manager: &Arc<DeviceResourceResidencyManagerInner<P>>,
    owner: &DeviceResourceResidencyOwnerId,
) -> Result<DeviceResourceResidencyRelease, DeviceResourceResidencyError> {
    let mut state = manager.state.lock().map_err(|_| {
        DeviceResourceResidencyError::new(
            DeviceResourceResidencyErrorKind::Stopped,
            "per-device residency manager was poisoned",
        )
    })?;
    for entry in state.entries.values() {
        if let DeviceResourceResidencyEntry::Resident {
            active_leases, ..
        } = entry
            && active_leases.get(owner).copied().unwrap_or_default() != 0
        {
            return Err(DeviceResourceResidencyError::new(
                DeviceResourceResidencyErrorKind::InUse,
                "cannot unload a residency owner while its execution lease is active",
            ));
        }
    }
    let mut release = DeviceResourceResidencyRelease::default();
    let group_ids = state.entries.keys().cloned().collect::<Vec<_>>();
    let mut cancelled = Vec::new();
    for group_id in group_ids {
        let action = match state.entries.get_mut(&group_id) {
            Some(DeviceResourceResidencyEntry::Loading {
                owners,
                descriptor,
                outcome,
                ..
            }) => {
                let removed = owners.remove(owner);
                (removed && owners.is_empty()).then(|| {
                    (
                        "loading",
                        descriptor.byte_count,
                        Some(Arc::clone(outcome)),
                    )
                })
            }
            Some(DeviceResourceResidencyEntry::Resident {
                owners,
                descriptor,
                ..
            }) => {
                let removed = owners.remove(owner);
                (removed && owners.is_empty())
                    .then_some(("resident", descriptor.byte_count, None))
            }
            _ => None,
        };
        if let Some((kind, byte_count, outcome)) = action {
            state.entries.remove(&group_id);
            match kind {
                "loading" => {
                    state.reserved_loading_bytes =
                        state.reserved_loading_bytes.saturating_sub(byte_count);
                    state.cancelled_load_count =
                        state.cancelled_load_count.saturating_add(1);
                    release.cancelled_load_count += 1;
                    cancelled.extend(outcome);
                }
                "resident" => {
                    state.dynamic_resident_bytes =
                        state.dynamic_resident_bytes.saturating_sub(byte_count);
                    release.group_count += 1;
                    release.byte_count += byte_count;
                }
                _ => unreachable!(),
            }
        }
    }
    drop(state);
    for outcome in cancelled {
        outcome.finish(Err(DeviceResourceResidencyError::new(
            DeviceResourceResidencyErrorKind::Cancelled,
            "per-device residency load was cancelled by explicit owner unload",
        )));
    }
    Ok(release)
}

fn unload_entire_manager<P: DeviceResidentResourcePayload>(
    manager: &Arc<DeviceResourceResidencyManagerInner<P>>,
) -> Result<DeviceResourceResidencyRelease, DeviceResourceResidencyError> {
    let mut state = manager.state.lock().map_err(|_| {
        DeviceResourceResidencyError::new(
            DeviceResourceResidencyErrorKind::Stopped,
            "per-device residency manager was poisoned",
        )
    })?;
    for entry in state.entries.values() {
        if let DeviceResourceResidencyEntry::Resident {
            active_leases, ..
        } = entry
            && !active_leases.is_empty()
        {
            return Err(DeviceResourceResidencyError::new(
                DeviceResourceResidencyErrorKind::InUse,
                "cannot unload a device while an execution lease is active",
            ));
        }
    }
    let mut release = DeviceResourceResidencyRelease::default();
    let mut cancelled = Vec::new();
    for (_, entry) in std::mem::take(&mut state.entries) {
        match entry {
            DeviceResourceResidencyEntry::Loading {
                descriptor,
                outcome,
                ..
            } => {
                release.cancelled_load_count += 1;
                state.reserved_loading_bytes = state
                    .reserved_loading_bytes
                    .saturating_sub(descriptor.byte_count);
                state.cancelled_load_count =
                    state.cancelled_load_count.saturating_add(1);
                cancelled.push(outcome);
            }
            DeviceResourceResidencyEntry::Resident {
                descriptor, ..
            } => {
                release.group_count += 1;
                release.byte_count += descriptor.byte_count;
                state.dynamic_resident_bytes = state
                    .dynamic_resident_bytes
                    .saturating_sub(descriptor.byte_count);
            }
            DeviceResourceResidencyEntry::Failed { .. } => {}
        }
    }
    drop(state);
    for outcome in cancelled {
        outcome.finish(Err(DeviceResourceResidencyError::new(
            DeviceResourceResidencyErrorKind::Cancelled,
            "per-device residency load was cancelled by explicit device unload",
        )));
    }
    Ok(release)
}

fn statistics_for_state<P: DeviceResidentResourcePayload>(
    manager: &DeviceResourceResidencyManagerInner<P>,
    state: &DeviceResourceResidencyState<P>,
) -> DeviceResourceResidencyStatistics {
    let mut loading_group_count = 0;
    let mut resident_group_count = 0;
    let mut failed_group_count = 0;
    for entry in state.entries.values() {
        match entry {
            DeviceResourceResidencyEntry::Loading { .. } => {
                loading_group_count += 1;
            }
            DeviceResourceResidencyEntry::Resident { .. } => {
                resident_group_count += 1;
            }
            DeviceResourceResidencyEntry::Failed { .. } => {
                failed_group_count += 1;
            }
        }
    }
    DeviceResourceResidencyStatistics {
        capacity_bytes: manager.capacity_bytes,
        always_resident_bytes: manager.always_resident_bytes,
        reserved_loading_bytes: state.reserved_loading_bytes,
        dynamic_resident_bytes: state.dynamic_resident_bytes,
        high_water_dynamic_resident_bytes:
            state.high_water_dynamic_resident_bytes,
        loading_group_count,
        resident_group_count,
        high_water_resident_group_count:
            state.high_water_resident_group_count,
        failed_group_count,
        hit_count: state.hit_count,
        miss_count: state.miss_count,
        single_flight_join_count: state.single_flight_join_count,
        successful_load_count: state.successful_load_count,
        failed_load_count: state.failed_load_count,
        cancelled_load_count: state.cancelled_load_count,
    }
}

fn directory_for_state<P: DeviceResidentResourcePayload>(
    manager: &DeviceResourceResidencyManagerInner<P>,
    state: &DeviceResourceResidencyState<P>,
) -> Vec<DeviceResourceResidencyDirectoryEntry> {
    state
        .entries
        .iter()
        .map(|(group_id, entry)| {
            let (residency_state, descriptor, owner_count) = match entry {
                DeviceResourceResidencyEntry::Loading {
                    descriptor, owners, ..
                } => (
                    ResourceResidencyState::Loading,
                    descriptor,
                    owners.len(),
                ),
                DeviceResourceResidencyEntry::Resident {
                    descriptor, owners, ..
                } => (
                    ResourceResidencyState::Resident,
                    descriptor,
                    owners.len(),
                ),
                DeviceResourceResidencyEntry::Failed {
                    descriptor, ..
                } => (
                    ResourceResidencyState::Failed,
                    descriptor,
                    0,
                ),
            };
            DeviceResourceResidencyDirectoryEntry {
                group_id: group_id.clone(),
                state: residency_state,
                location: DeviceResourceResidencyLocation::Local {
                    device_id: manager.device_id.clone(),
                },
                byte_count: descriptor.byte_count,
                owner_count,
            }
        })
        .collect()
}
