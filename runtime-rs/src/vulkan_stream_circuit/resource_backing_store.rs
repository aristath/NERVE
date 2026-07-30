use std::os::unix::fs::FileExt as _;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Condvar, Mutex, mpsc};
use std::thread;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompiledResourceBackingStoreLimits {
    pub worker_count: usize,
    pub queued_request_capacity: usize,
    pub maximum_ranges_per_group: usize,
    pub maximum_logical_bytes_per_group: usize,
    pub maximum_retained_payload_bytes: usize,
    pub maximum_coalesced_read_bytes: usize,
    pub maximum_coalescing_gap_bytes: usize,
}

impl Default for CompiledResourceBackingStoreLimits {
    fn default() -> Self {
        Self {
            worker_count: 2,
            queued_request_capacity: 32,
            maximum_ranges_per_group: 64,
            maximum_logical_bytes_per_group: 64 * 1024 * 1024,
            maximum_retained_payload_bytes: 128 * 1024 * 1024,
            maximum_coalesced_read_bytes: 16 * 1024 * 1024,
            maximum_coalescing_gap_bytes: 64 * 1024,
        }
    }
}

impl CompiledResourceBackingStoreLimits {
    fn validate(&self) -> Result<(), CompiledResourceBackingStoreError> {
        if self.worker_count == 0
            || self.queued_request_capacity == 0
            || self.maximum_ranges_per_group == 0
            || self.maximum_logical_bytes_per_group == 0
            || self.maximum_retained_payload_bytes == 0
            || self.maximum_coalesced_read_bytes == 0
        {
            return Err(CompiledResourceBackingStoreError::configuration(
                "backing-store limits must all be non-zero except the coalescing gap",
            ));
        }
        if self.maximum_coalesced_read_bytes > self.maximum_logical_bytes_per_group {
            return Err(CompiledResourceBackingStoreError::configuration(
                "maximum coalesced read bytes cannot exceed maximum logical group bytes",
            ));
        }
        if self.maximum_retained_payload_bytes < self.maximum_logical_bytes_per_group {
            return Err(CompiledResourceBackingStoreError::configuration(
                "retained payload budget cannot be smaller than one maximum-size group",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompiledResourceBackingStoreErrorKind {
    Cancelled,
    Configuration,
    Integrity,
    Io,
    QueueFull,
    Stopped,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompiledResourceBackingStoreError {
    kind: CompiledResourceBackingStoreErrorKind,
    message: String,
}

impl CompiledResourceBackingStoreError {
    fn new(kind: CompiledResourceBackingStoreErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    fn cancelled() -> Self {
        Self::new(
            CompiledResourceBackingStoreErrorKind::Cancelled,
            "compiled resource load was cancelled",
        )
    }

    fn configuration(message: impl Into<String>) -> Self {
        Self::new(
            CompiledResourceBackingStoreErrorKind::Configuration,
            message,
        )
    }

    pub fn kind(&self) -> CompiledResourceBackingStoreErrorKind {
        self.kind
    }
}

impl Display for CompiledResourceBackingStoreError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for CompiledResourceBackingStoreError {}

#[derive(Clone, Debug)]
pub struct CompiledResourceLoadCancellation {
    cancelled: Arc<AtomicBool>,
}

impl CompiledResourceLoadCancellation {
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, AtomicOrdering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(AtomicOrdering::Acquire)
    }
}

impl Default for CompiledResourceLoadCancellation {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub struct LoadedCompiledResourceRange {
    pub descriptor: ResolvedCompiledResourceRange,
    pub bytes: Arc<[u8]>,
}

#[derive(Clone, Debug)]
pub struct LoadedCompiledResource {
    pub id: String,
    pub ranges: Vec<LoadedCompiledResourceRange>,
    pub compatibility: CompiledResourceCompatibility,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LoadedCompiledResourceGroupOrigin {
    Atomic {
        atomic_group_id: String,
    },
    Partition {
        partition_template_id: String,
        partition_index: usize,
    },
}

#[derive(Clone, Debug)]
pub struct LoadedCompiledResourceGroup {
    pub id: String,
    pub origin: LoadedCompiledResourceGroupOrigin,
    pub resource_ids: Vec<String>,
    pub dependencies: Vec<String>,
    pub resources: Vec<LoadedCompiledResource>,
    pub logical_range_count: usize,
    pub physical_read_count: usize,
    pub logical_byte_count: usize,
    pub physical_byte_count: usize,
    pub elapsed: Duration,
    _host_memory_reservation: Arc<CompiledResourceHostMemoryReservation>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct CompiledResourceBackingStoreStatistics {
    pub submitted_requests: u64,
    pub active_requests: u64,
    pub completed_requests: u64,
    pub cancelled_requests: u64,
    pub failed_requests: u64,
    pub logical_ranges: u64,
    pub physical_reads: u64,
    pub logical_bytes: u64,
    pub physical_bytes: u64,
    pub read_time_ns: u64,
}

#[derive(Default)]
struct CompiledResourceBackingStoreAtomicStatistics {
    submitted_requests: AtomicU64,
    active_requests: AtomicU64,
    completed_requests: AtomicU64,
    cancelled_requests: AtomicU64,
    failed_requests: AtomicU64,
    logical_ranges: AtomicU64,
    physical_reads: AtomicU64,
    logical_bytes: AtomicU64,
    physical_bytes: AtomicU64,
    read_time_ns: AtomicU64,
}

impl CompiledResourceBackingStoreAtomicStatistics {
    fn snapshot(&self) -> CompiledResourceBackingStoreStatistics {
        CompiledResourceBackingStoreStatistics {
            submitted_requests: self.submitted_requests.load(AtomicOrdering::Relaxed),
            active_requests: self.active_requests.load(AtomicOrdering::Acquire),
            completed_requests: self.completed_requests.load(AtomicOrdering::Relaxed),
            cancelled_requests: self.cancelled_requests.load(AtomicOrdering::Relaxed),
            failed_requests: self.failed_requests.load(AtomicOrdering::Relaxed),
            logical_ranges: self.logical_ranges.load(AtomicOrdering::Relaxed),
            physical_reads: self.physical_reads.load(AtomicOrdering::Relaxed),
            logical_bytes: self.logical_bytes.load(AtomicOrdering::Relaxed),
            physical_bytes: self.physical_bytes.load(AtomicOrdering::Relaxed),
            read_time_ns: self.read_time_ns.load(AtomicOrdering::Relaxed),
        }
    }
}

struct CompiledResourceBackingStoreWork {
    group: ResolvedCompiledResourceGroup,
    cancellation: CompiledResourceLoadCancellation,
    response: mpsc::Sender<
        Result<LoadedCompiledResourceGroup, CompiledResourceBackingStoreError>,
    >,
}

#[derive(Debug)]
struct CompiledResourceHostMemoryBudget {
    maximum_bytes: usize,
    retained_bytes: Mutex<usize>,
    changed: Condvar,
    stopped: AtomicBool,
}

#[derive(Debug)]
struct CompiledResourceHostMemoryReservation {
    budget: Arc<CompiledResourceHostMemoryBudget>,
    byte_count: usize,
}

impl CompiledResourceHostMemoryBudget {
    fn new(maximum_bytes: usize) -> Self {
        Self {
            maximum_bytes,
            retained_bytes: Mutex::new(0),
            changed: Condvar::new(),
            stopped: AtomicBool::new(false),
        }
    }

    fn reserve(
        self: &Arc<Self>,
        byte_count: usize,
        cancellation: &CompiledResourceLoadCancellation,
    ) -> Result<Arc<CompiledResourceHostMemoryReservation>, CompiledResourceBackingStoreError> {
        if byte_count > self.maximum_bytes {
            return Err(CompiledResourceBackingStoreError::configuration(format!(
                "compiled resource load needs {byte_count} retained host bytes but the bounded budget is {}",
                self.maximum_bytes
            )));
        }
        let mut retained = self.retained_bytes.lock().map_err(|_| {
            CompiledResourceBackingStoreError::new(
                CompiledResourceBackingStoreErrorKind::Stopped,
                "compiled resource host-memory budget was poisoned",
            )
        })?;
        loop {
            if cancellation.is_cancelled() {
                return Err(CompiledResourceBackingStoreError::cancelled());
            }
            if self.stopped.load(AtomicOrdering::Acquire) {
                return Err(CompiledResourceBackingStoreError::new(
                    CompiledResourceBackingStoreErrorKind::Stopped,
                    "compiled resource backing store is stopping",
                ));
            }
            if retained
                .checked_add(byte_count)
                .is_some_and(|total| total <= self.maximum_bytes)
            {
                *retained += byte_count;
                return Ok(Arc::new(CompiledResourceHostMemoryReservation {
                    budget: Arc::clone(self),
                    byte_count,
                }));
            }
            let (following, _) = self
                .changed
                .wait_timeout(retained, Duration::from_millis(10))
                .map_err(|_| {
                    CompiledResourceBackingStoreError::new(
                        CompiledResourceBackingStoreErrorKind::Stopped,
                        "compiled resource host-memory budget was poisoned",
                    )
                })?;
            retained = following;
        }
    }

    fn stop(&self) {
        self.stopped.store(true, AtomicOrdering::Release);
        self.changed.notify_all();
    }
}

impl Drop for CompiledResourceHostMemoryReservation {
    fn drop(&mut self) {
        if let Ok(mut retained) = self.budget.retained_bytes.lock() {
            *retained = retained.saturating_sub(self.byte_count);
            self.budget.changed.notify_one();
        }
    }
}

pub struct CompiledResourceLoadTicket {
    cancellation: CompiledResourceLoadCancellation,
    response: mpsc::Receiver<
        Result<LoadedCompiledResourceGroup, CompiledResourceBackingStoreError>,
    >,
}

impl CompiledResourceLoadTicket {
    pub fn cancellation(&self) -> CompiledResourceLoadCancellation {
        self.cancellation.clone()
    }

    pub fn cancel(&self) {
        self.cancellation.cancel();
    }

    pub fn wait(
        self,
    ) -> Result<LoadedCompiledResourceGroup, CompiledResourceBackingStoreError> {
        self.response.recv().map_err(|_| {
            CompiledResourceBackingStoreError::new(
                CompiledResourceBackingStoreErrorKind::Stopped,
                "compiled resource backing-store worker stopped without a response",
            )
        })?
    }
}

impl Drop for CompiledResourceLoadTicket {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

pub struct CompiledResourceBackingStore {
    request_sender: Option<mpsc::SyncSender<CompiledResourceBackingStoreWork>>,
    workers: Vec<thread::JoinHandle<()>>,
    statistics: Arc<CompiledResourceBackingStoreAtomicStatistics>,
    host_memory_budget: Arc<CompiledResourceHostMemoryBudget>,
}

impl CompiledResourceBackingStore {
    pub fn new(
        package_root: impl Into<PathBuf>,
        limits: CompiledResourceBackingStoreLimits,
    ) -> Result<Self, CompiledResourceBackingStoreError> {
        limits.validate()?;
        let package_root = package_root.into();
        let (request_sender, request_receiver) =
            mpsc::sync_channel::<CompiledResourceBackingStoreWork>(
                limits.queued_request_capacity,
            );
        let request_receiver = Arc::new(Mutex::new(request_receiver));
        let statistics = Arc::new(CompiledResourceBackingStoreAtomicStatistics::default());
        let host_memory_budget = Arc::new(CompiledResourceHostMemoryBudget::new(
            limits.maximum_retained_payload_bytes,
        ));
        let mut workers = Vec::with_capacity(limits.worker_count);
        for worker_index in 0..limits.worker_count {
            let receiver = Arc::clone(&request_receiver);
            let worker_root = package_root.clone();
            let worker_limits = limits.clone();
            let worker_statistics = Arc::clone(&statistics);
            let worker_host_memory_budget = Arc::clone(&host_memory_budget);
            let worker = thread::Builder::new()
                .name(format!("nerve-resource-store-{worker_index}"))
                .spawn(move || {
                    loop {
                        let work = {
                            let receiver = match receiver.lock() {
                                Ok(receiver) => receiver,
                                Err(_) => break,
                            };
                            match receiver.recv() {
                                Ok(work) => work,
                                Err(_) => break,
                            }
                        };
                        worker_statistics
                            .active_requests
                            .fetch_add(1, AtomicOrdering::Release);
                        let result = read_compiled_resource_group_with_limits(
                            &worker_root,
                            &work.group,
                            &work.cancellation,
                            &worker_limits,
                            &worker_host_memory_budget,
                        );
                        worker_statistics
                            .active_requests
                            .fetch_sub(1, AtomicOrdering::Release);
                        match &result {
                            Ok(loaded) => {
                                worker_statistics
                                    .completed_requests
                                    .fetch_add(1, AtomicOrdering::Relaxed);
                                worker_statistics.logical_ranges.fetch_add(
                                    loaded.logical_range_count as u64,
                                    AtomicOrdering::Relaxed,
                                );
                                worker_statistics.physical_reads.fetch_add(
                                    loaded.physical_read_count as u64,
                                    AtomicOrdering::Relaxed,
                                );
                                worker_statistics.logical_bytes.fetch_add(
                                    loaded.logical_byte_count as u64,
                                    AtomicOrdering::Relaxed,
                                );
                                worker_statistics.physical_bytes.fetch_add(
                                    loaded.physical_byte_count as u64,
                                    AtomicOrdering::Relaxed,
                                );
                                worker_statistics.read_time_ns.fetch_add(
                                    u64::try_from(loaded.elapsed.as_nanos())
                                        .unwrap_or(u64::MAX),
                                    AtomicOrdering::Relaxed,
                                );
                            }
                            Err(error)
                                if error.kind
                                    == CompiledResourceBackingStoreErrorKind::Cancelled =>
                            {
                                worker_statistics
                                    .cancelled_requests
                                    .fetch_add(1, AtomicOrdering::Relaxed);
                            }
                            Err(_) => {
                                worker_statistics
                                    .failed_requests
                                    .fetch_add(1, AtomicOrdering::Relaxed);
                            }
                        }
                        let _ = work.response.send(result);
                    }
                })
                .map_err(|error| {
                    CompiledResourceBackingStoreError::new(
                        CompiledResourceBackingStoreErrorKind::Io,
                        format!("failed to start compiled resource worker: {error}"),
                    )
                })?;
            workers.push(worker);
        }
        Ok(Self {
            request_sender: Some(request_sender),
            workers,
            statistics,
            host_memory_budget,
        })
    }

    pub fn try_load<G>(
        &self,
        group: G,
    ) -> Result<CompiledResourceLoadTicket, CompiledResourceBackingStoreError>
    where
        G: Into<ResolvedCompiledResourceGroup>,
    {
        self.try_load_with_cancellation(group, CompiledResourceLoadCancellation::new())
    }

    pub fn try_load_with_cancellation<G>(
        &self,
        group: G,
        cancellation: CompiledResourceLoadCancellation,
    ) -> Result<CompiledResourceLoadTicket, CompiledResourceBackingStoreError>
    where
        G: Into<ResolvedCompiledResourceGroup>,
    {
        let (response_sender, response_receiver) = mpsc::channel();
        let request = CompiledResourceBackingStoreWork {
            group: group.into(),
            cancellation: cancellation.clone(),
            response: response_sender,
        };
        let sender = self.request_sender.as_ref().ok_or_else(|| {
            CompiledResourceBackingStoreError::new(
                CompiledResourceBackingStoreErrorKind::Stopped,
                "compiled resource backing store is stopped",
            )
        })?;
        sender.try_send(request).map_err(|error| match error {
            mpsc::TrySendError::Full(_) => CompiledResourceBackingStoreError::new(
                CompiledResourceBackingStoreErrorKind::QueueFull,
                "compiled resource backing-store request queue is full",
            ),
            mpsc::TrySendError::Disconnected(_) => CompiledResourceBackingStoreError::new(
                CompiledResourceBackingStoreErrorKind::Stopped,
                "compiled resource backing store is stopped",
            ),
        })?;
        self.statistics
            .submitted_requests
            .fetch_add(1, AtomicOrdering::Relaxed);
        Ok(CompiledResourceLoadTicket {
            cancellation,
            response: response_receiver,
        })
    }

    pub fn statistics(&self) -> CompiledResourceBackingStoreStatistics {
        self.statistics.snapshot()
    }

    pub fn retained_payload_bytes(&self) -> usize {
        self.host_memory_budget
            .retained_bytes
            .lock()
            .map(|retained| *retained)
            .unwrap_or(0)
    }
}

impl Drop for CompiledResourceBackingStore {
    fn drop(&mut self) {
        self.host_memory_budget.stop();
        self.request_sender.take();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct PhysicalResourceRangeKey {
    artifact_path: String,
    byte_offset: usize,
    byte_count: usize,
    sha256: String,
}

#[derive(Clone, Debug)]
struct CoalescedPhysicalRead {
    artifact_path: String,
    byte_offset: usize,
    byte_count: usize,
    range_keys: Vec<PhysicalResourceRangeKey>,
}

fn read_compiled_resource_group_with_limits(
    package_root: &Path,
    group: &ResolvedCompiledResourceGroup,
    cancellation: &CompiledResourceLoadCancellation,
    limits: &CompiledResourceBackingStoreLimits,
    host_memory_budget: &Arc<CompiledResourceHostMemoryBudget>,
) -> Result<LoadedCompiledResourceGroup, CompiledResourceBackingStoreError> {
    let started = Instant::now();
    validate_resolved_group_for_loading(group)?;
    if cancellation.is_cancelled() {
        return Err(CompiledResourceBackingStoreError::cancelled());
    }
    let logical_ranges = group
        .resources()
        .iter()
        .flat_map(|resource| resource.ranges.iter())
        .collect::<Vec<_>>();
    if logical_ranges.len() > limits.maximum_ranges_per_group {
        return Err(CompiledResourceBackingStoreError::configuration(format!(
            "compiled group has {} ranges but the bounded maximum is {}",
            logical_ranges.len(),
            limits.maximum_ranges_per_group
        )));
    }
    let logical_byte_count = logical_ranges.iter().try_fold(0usize, |total, range| {
        total.checked_add(range.byte_count).ok_or_else(|| {
            CompiledResourceBackingStoreError::configuration(
                "compiled group logical byte count overflowed",
            )
        })
    })?;
    if logical_byte_count > limits.maximum_logical_bytes_per_group {
        return Err(CompiledResourceBackingStoreError::configuration(format!(
            "compiled group has {logical_byte_count} logical bytes but the bounded maximum is {}",
            limits.maximum_logical_bytes_per_group
        )));
    }
    let host_memory_reservation =
        host_memory_budget.reserve(logical_byte_count, cancellation)?;

    let unique_ranges = logical_ranges
        .iter()
        .map(|range| PhysicalResourceRangeKey {
            artifact_path: range.artifact_path.clone(),
            byte_offset: range.byte_offset,
            byte_count: range.byte_count,
            sha256: range.sha256.clone(),
        })
        .collect::<BTreeSet<_>>();
    let reads = coalesce_physical_resource_reads(&unique_ranges, limits)?;
    let mut loaded_ranges = BTreeMap::<PhysicalResourceRangeKey, Arc<[u8]>>::new();
    let mut physical_byte_count = 0usize;
    for read in &reads {
        if cancellation.is_cancelled() {
            return Err(CompiledResourceBackingStoreError::cancelled());
        }
        package::validate_resident_package_relative_path(
            "resolved resource artifact",
            &read.artifact_path,
        )
        .map_err(|error| {
            CompiledResourceBackingStoreError::configuration(error.to_string())
        })?;
        let path = package_root.join(&read.artifact_path);
        let source = fs::File::open(&path).map_err(|error| {
            CompiledResourceBackingStoreError::new(
                CompiledResourceBackingStoreErrorKind::Io,
                format!(
                    "failed to open compiled resource artifact {}: {error}",
                    path.display()
                ),
            )
        })?;
        let mut bytes = vec![0u8; read.byte_count];
        source
            .read_exact_at(
                &mut bytes,
                u64::try_from(read.byte_offset).map_err(|_| {
                    CompiledResourceBackingStoreError::configuration(
                        "compiled resource offset does not fit u64",
                    )
                })?,
            )
            .map_err(|error| {
                CompiledResourceBackingStoreError::new(
                    CompiledResourceBackingStoreErrorKind::Io,
                    format!(
                        "failed to read compiled resource range {}:{}+{}: {error}",
                        read.artifact_path, read.byte_offset, read.byte_count
                    ),
                )
            })?;
        physical_byte_count = physical_byte_count
            .checked_add(bytes.len())
            .ok_or_else(|| {
                CompiledResourceBackingStoreError::configuration(
                    "compiled physical byte count overflowed",
                )
            })?;
        for key in &read.range_keys {
            let relative_start = key.byte_offset - read.byte_offset;
            let relative_end = relative_start + key.byte_count;
            let payload = &bytes[relative_start..relative_end];
            if format!("{:x}", Sha256::digest(payload)) != key.sha256 {
                return Err(CompiledResourceBackingStoreError::new(
                    CompiledResourceBackingStoreErrorKind::Integrity,
                    format!(
                        "compiled resource range {}:{}+{} failed SHA-256",
                        key.artifact_path, key.byte_offset, key.byte_count
                    ),
                ));
            }
            loaded_ranges.insert(key.clone(), Arc::from(payload));
        }
    }
    if cancellation.is_cancelled() {
        return Err(CompiledResourceBackingStoreError::cancelled());
    }

    let resources = group
        .resources()
        .iter()
        .map(|resource| {
            let ranges = resource
                .ranges
                .iter()
                .map(|range| {
                    let key = PhysicalResourceRangeKey {
                        artifact_path: range.artifact_path.clone(),
                        byte_offset: range.byte_offset,
                        byte_count: range.byte_count,
                        sha256: range.sha256.clone(),
                    };
                    Ok(LoadedCompiledResourceRange {
                        descriptor: range.clone(),
                        bytes: Arc::clone(loaded_ranges.get(&key).ok_or_else(|| {
                            CompiledResourceBackingStoreError::configuration(
                                "verified compiled range was not retained",
                            )
                        })?),
                    })
                })
                .collect::<Result<Vec<_>, CompiledResourceBackingStoreError>>()?;
            Ok(LoadedCompiledResource {
                id: resource.id.clone(),
                ranges,
                compatibility: resource.compatibility.clone(),
            })
        })
        .collect::<Result<Vec<_>, CompiledResourceBackingStoreError>>()?;
    Ok(LoadedCompiledResourceGroup {
        id: group.id().to_string(),
        origin: match group {
            ResolvedCompiledResourceGroup::Atomic(group) => {
                LoadedCompiledResourceGroupOrigin::Atomic {
                    atomic_group_id: group.id.clone(),
                }
            }
            ResolvedCompiledResourceGroup::Partition(group) => {
                LoadedCompiledResourceGroupOrigin::Partition {
                    partition_template_id: group.partition_template_id.clone(),
                    partition_index: group.partition_index,
                }
            }
        },
        resource_ids: group.resource_ids().to_vec(),
        dependencies: group.dependencies().to_vec(),
        resources,
        logical_range_count: logical_ranges.len(),
        physical_read_count: reads.len(),
        logical_byte_count,
        physical_byte_count,
        elapsed: started.elapsed(),
        _host_memory_reservation: host_memory_reservation,
    })
}

fn validate_resolved_group_for_loading(
    group: &ResolvedCompiledResourceGroup,
) -> Result<(), CompiledResourceBackingStoreError> {
    let valid_schema = match group {
        ResolvedCompiledResourceGroup::Atomic(group) => {
            group.schema == RESOLVED_ATOMIC_GROUP_SCHEMA
        }
        ResolvedCompiledResourceGroup::Partition(group) => {
            group.schema == RESOLVED_PARTITION_GROUP_SCHEMA
        }
    };
    if !valid_schema {
        return Err(CompiledResourceBackingStoreError::configuration(
            "resolved compiled resource group has an unsupported schema",
        ));
    }
    let actual_resource_ids = group
        .resources()
        .iter()
        .map(|resource| resource.id.clone())
        .collect::<Vec<_>>();
    if group.resource_ids() != actual_resource_ids
        || group
            .resource_ids()
            .iter()
            .collect::<BTreeSet<_>>()
            .len()
            != group.resource_ids().len()
    {
        return Err(CompiledResourceBackingStoreError::configuration(
            "resolved partition group resource identities are inconsistent",
        ));
    }
    for range in group
        .resources()
        .iter()
        .flat_map(|resource| resource.ranges.iter())
    {
        if range.byte_count == 0
            || !range.alignment_bytes.is_power_of_two()
            || !range.byte_offset.is_multiple_of(range.alignment_bytes)
            || !package::is_lower_hex_sha256(&range.sha256)
        {
            return Err(CompiledResourceBackingStoreError::configuration(
                "resolved compiled resource range is invalid",
            ));
        }
    }
    Ok(())
}

fn coalesce_physical_resource_reads(
    unique_ranges: &BTreeSet<PhysicalResourceRangeKey>,
    limits: &CompiledResourceBackingStoreLimits,
) -> Result<Vec<CoalescedPhysicalRead>, CompiledResourceBackingStoreError> {
    let mut reads = Vec::<CoalescedPhysicalRead>::new();
    for key in unique_ranges {
        let range_end = key.byte_offset.checked_add(key.byte_count).ok_or_else(|| {
            CompiledResourceBackingStoreError::configuration(
                "compiled resource interval overflowed",
            )
        })?;
        let can_extend = reads.last().is_some_and(|read| {
            if read.artifact_path != key.artifact_path {
                return false;
            }
            let read_end = read.byte_offset + read.byte_count;
            let gap = key.byte_offset.saturating_sub(read_end);
            let merged_end = read_end.max(range_end);
            key.byte_offset <= read_end.saturating_add(limits.maximum_coalescing_gap_bytes)
                && gap <= limits.maximum_coalescing_gap_bytes
                && merged_end - read.byte_offset <= limits.maximum_coalesced_read_bytes
        });
        if can_extend {
            let read = reads.last_mut().expect("checked final coalesced read");
            read.byte_count = (read.byte_offset + read.byte_count).max(range_end) - read.byte_offset;
            read.range_keys.push(key.clone());
        } else {
            if key.byte_count > limits.maximum_coalesced_read_bytes {
                return Err(CompiledResourceBackingStoreError::configuration(format!(
                    "compiled resource range has {} bytes but the bounded read maximum is {}",
                    key.byte_count, limits.maximum_coalesced_read_bytes
                )));
            }
            reads.push(CoalescedPhysicalRead {
                artifact_path: key.artifact_path.clone(),
                byte_offset: key.byte_offset,
                byte_count: key.byte_count,
                range_keys: vec![key.clone()],
            });
        }
    }
    Ok(reads)
}

#[cfg(test)]
mod resource_backing_store_tests {
    use super::*;
    use std::os::fd::AsRawFd;
    use std::sync::Barrier;

    fn digest(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn compatibility() -> CompiledResourceCompatibility {
        CompiledResourceCompatibility {
            device_api: "vulkan".to_string(),
            storage_class: "storage_buffer".to_string(),
            read_only: true,
            required_features: Vec::new(),
        }
    }

    fn group_with_ranges(
        ranges: Vec<(&str, usize, &[u8])>,
    ) -> (crate::test_support::TempDir, ResolvedCompiledPartitionGroup) {
        let root = crate::test_support::TempDir::new("resource_backing_store");
        let mut artifacts = BTreeMap::<String, Vec<u8>>::new();
        let resolved_ranges = ranges
            .iter()
            .map(|(path, offset, bytes)| {
                let artifact = artifacts.entry((*path).to_string()).or_default();
                artifact.resize(artifact.len().max(offset + bytes.len()), 0xA5);
                artifact[*offset..*offset + bytes.len()].copy_from_slice(bytes);
                ResolvedCompiledResourceRange {
                    artifact_path: (*path).to_string(),
                    byte_offset: *offset,
                    byte_count: bytes.len(),
                    alignment_bytes: 4,
                    sha256: digest(bytes),
                }
            })
            .collect::<Vec<_>>();
        for (path, bytes) in artifacts {
            let destination = root.path().join(path);
            fs::create_dir_all(destination.parent().unwrap()).unwrap();
            fs::write(destination, bytes).unwrap();
        }
        let resource_id = format!("sha256:{}", "1".repeat(64));
        (
            root,
            ResolvedCompiledPartitionGroup {
                schema: RESOLVED_PARTITION_GROUP_SCHEMA.to_string(),
                partition_template_id: format!("sha256:{}", "2".repeat(64)),
                partition_index: 0,
                id: format!("sha256:{}", "3".repeat(64)),
                resource_ids: vec![resource_id.clone()],
                dependencies: Vec::new(),
                resources: vec![ResolvedCompiledResource {
                    id: resource_id,
                    ranges: resolved_ranges,
                    compatibility: compatibility(),
                }],
            },
        )
    }

    fn process_file_descriptor_count() -> usize {
        fs::read_dir("/proc/self/fd").unwrap().count()
    }

    fn process_named_thread_count(prefix: &str) -> usize {
        fs::read_dir("/proc/self/task")
            .unwrap()
            .map(|entry| entry.unwrap().path().join("comm"))
            .filter_map(|path| fs::read_to_string(path).ok())
            .filter(|name| name.trim().starts_with(prefix))
            .count()
    }

    #[test]
    fn isolated_and_adjacent_ranges_are_verified_and_coalesced() {
        let (root, group) = group_with_ranges(vec![
            ("weights/a.bin", 0, b"abcdefgh"),
            ("weights/a.bin", 8, b"ijklmnop"),
            ("weights/b.bin", 0, b"qrstuvwx"),
        ]);
        let store = CompiledResourceBackingStore::new(
            root.path(),
            CompiledResourceBackingStoreLimits {
                worker_count: 1,
                queued_request_capacity: 1,
                maximum_ranges_per_group: 8,
                maximum_logical_bytes_per_group: 1024,
                maximum_retained_payload_bytes: 1024,
                maximum_coalesced_read_bytes: 1024,
                maximum_coalescing_gap_bytes: 0,
            },
        )
        .unwrap();
        let loaded = store.try_load(group).unwrap().wait().unwrap();
        assert_eq!(loaded.logical_range_count, 3);
        assert_eq!(loaded.physical_read_count, 2);
        assert_eq!(loaded.logical_byte_count, 24);
        assert_eq!(loaded.physical_byte_count, 24);
        assert_eq!(&*loaded.resources[0].ranges[1].bytes, b"ijklmnop");
        let statistics = store.statistics();
        assert_eq!(statistics.physical_reads, 2);
        assert_eq!(statistics.physical_bytes, 24);
        assert_eq!(
            statistics.read_time_ns,
            u64::try_from(loaded.elapsed.as_nanos())
                .unwrap_or(u64::MAX)
        );
    }

    #[test]
    fn duplicate_physical_ranges_share_one_verified_payload() {
        let (root, mut group) =
            group_with_ranges(vec![("weights/a.bin", 0, b"abcdefgh")]);
        let duplicate_id = format!("sha256:{}", "5".repeat(64));
        let duplicate = ResolvedCompiledResource {
            id: duplicate_id.clone(),
            ranges: group.resources[0].ranges.clone(),
            compatibility: compatibility(),
        };
        group.resource_ids.push(duplicate_id);
        group.resources.push(duplicate);
        let store = CompiledResourceBackingStore::new(
            root.path(),
            CompiledResourceBackingStoreLimits {
                worker_count: 1,
                queued_request_capacity: 1,
                maximum_ranges_per_group: 8,
                maximum_logical_bytes_per_group: 1024,
                maximum_retained_payload_bytes: 1024,
                maximum_coalesced_read_bytes: 1024,
                maximum_coalescing_gap_bytes: 0,
            },
        )
        .unwrap();
        let loaded = store.try_load(group).unwrap().wait().unwrap();
        assert_eq!(loaded.logical_range_count, 2);
        assert_eq!(loaded.physical_read_count, 1);
        assert_eq!(loaded.physical_byte_count, 8);
        assert!(Arc::ptr_eq(
            &loaded.resources[0].ranges[0].bytes,
            &loaded.resources[1].ranges[0].bytes,
        ));
    }

    #[test]
    fn concrete_atomic_groups_use_the_same_verified_backing_store() {
        let root = crate::test_support::TempDir::new("atomic_resource_backing_store");
        fs::create_dir_all(root.path().join("weights")).unwrap();
        fs::write(root.path().join("weights/a.bin"), b"abcdefgh").unwrap();
        let resource_id = format!("sha256:{}", "1".repeat(64));
        let group_id = format!("sha256:{}", "2".repeat(64));
        let contract = CompiledResourceResidencyContract {
            schema: COMPILED_RESOURCE_RESIDENCY_SCHEMA.to_string(),
            identity_algorithm: RESOURCE_IDENTITY_ALGORITHM.to_string(),
            state_machine_schema: RESOURCE_RESIDENCY_STATE_MACHINE_SCHEMA.to_string(),
            supported_policies: vec![
                ResourceResidencyPolicy::DemandRetained,
                ResourceResidencyPolicy::Eager,
            ],
            resources: vec![CompiledImmutableResource {
                id: resource_id.clone(),
                lifetime: CompiledResourceLifetime::Dynamic,
                ranges: vec![CompiledResourceByteRange {
                    artifact_path: "weights/a.bin".to_string(),
                    byte_offset: 0,
                    byte_count: 8,
                    alignment_bytes: 4,
                    integrity: CompiledResourceRangeIntegrity {
                        algorithm: "sha256".to_string(),
                        digest: digest(b"abcdefgh"),
                    },
                }],
                dependencies: Vec::new(),
                compatibility: compatibility(),
            }],
            atomic_groups: vec![CompiledAtomicResidencyGroup {
                id: group_id.clone(),
                lifetime: CompiledResourceLifetime::Dynamic,
                resource_ids: vec![resource_id],
                dependencies: Vec::new(),
            }],
            partition_templates: Vec::new(),
            bindings: Vec::new(),
            selectors: Vec::new(),
            checkpoints: Vec::new(),
        };
        let resolved = resolve_compiled_atomic_group(&contract, &group_id).unwrap();
        let store = CompiledResourceBackingStore::new(
            root.path(),
            CompiledResourceBackingStoreLimits {
                worker_count: 1,
                queued_request_capacity: 1,
                maximum_ranges_per_group: 8,
                maximum_logical_bytes_per_group: 1024,
                maximum_retained_payload_bytes: 1024,
                maximum_coalesced_read_bytes: 1024,
                maximum_coalescing_gap_bytes: 0,
            },
        )
        .unwrap();

        let loaded = store.try_load(resolved).unwrap().wait().unwrap();

        assert_eq!(
            loaded.origin,
            LoadedCompiledResourceGroupOrigin::Atomic {
                atomic_group_id: group_id,
            }
        );
        assert_eq!(&*loaded.resources[0].ranges[0].bytes, b"abcdefgh");
    }

    #[test]
    fn cancellation_and_queue_bounds_release_workers() {
        let (root, group) =
            group_with_ranges(vec![("weights/a.bin", 0, b"abcdefgh")]);
        let store = CompiledResourceBackingStore::new(
            root.path(),
            CompiledResourceBackingStoreLimits {
                worker_count: 1,
                queued_request_capacity: 1,
                maximum_ranges_per_group: 8,
                maximum_logical_bytes_per_group: 1024,
                maximum_retained_payload_bytes: 1024,
                maximum_coalesced_read_bytes: 1024,
                maximum_coalescing_gap_bytes: 0,
            },
        )
        .unwrap();
        let cancellation = CompiledResourceLoadCancellation::new();
        cancellation.cancel();
        let ticket = store
            .try_load_with_cancellation(group.clone(), cancellation)
            .unwrap();
        assert_eq!(
            ticket.wait().unwrap_err().kind(),
            CompiledResourceBackingStoreErrorKind::Cancelled
        );
        let loaded = store.try_load(group).unwrap().wait().unwrap();
        assert_eq!(&*loaded.resources[0].ranges[0].bytes, b"abcdefgh");
        assert_eq!(store.statistics().cancelled_requests, 1);
        assert_eq!(store.retained_payload_bytes(), 8);
        drop(loaded);
        assert_eq!(store.retained_payload_bytes(), 0);
    }

    #[test]
    fn retained_payload_and_request_queue_limits_apply_backpressure() {
        let (root, group) =
            group_with_ranges(vec![("weights/a.bin", 0, b"abcdefgh")]);
        let store = CompiledResourceBackingStore::new(
            root.path(),
            CompiledResourceBackingStoreLimits {
                worker_count: 1,
                queued_request_capacity: 1,
                maximum_ranges_per_group: 1,
                maximum_logical_bytes_per_group: 8,
                maximum_retained_payload_bytes: 8,
                maximum_coalesced_read_bytes: 8,
                maximum_coalescing_gap_bytes: 0,
            },
        )
        .unwrap();
        let first = store.try_load(group.clone()).unwrap().wait().unwrap();
        assert_eq!(store.retained_payload_bytes(), 8);
        let second = store.try_load(group.clone()).unwrap();
        let deadline = Instant::now() + Duration::from_secs(1);
        while store.statistics().active_requests != 1 && Instant::now() < deadline {
            thread::yield_now();
        }
        assert_eq!(store.statistics().active_requests, 1);
        let third = store.try_load(group.clone()).unwrap();
        assert_eq!(
            store.try_load(group).err().unwrap().kind(),
            CompiledResourceBackingStoreErrorKind::QueueFull
        );
        second.cancel();
        assert_eq!(
            second.wait().unwrap_err().kind(),
            CompiledResourceBackingStoreErrorKind::Cancelled
        );
        third.cancel();
        assert_eq!(
            third.wait().unwrap_err().kind(),
            CompiledResourceBackingStoreErrorKind::Cancelled
        );
        drop(first);
        assert_eq!(store.retained_payload_bytes(), 0);
    }

    #[test]
    fn corrupt_and_failed_requests_do_not_poison_following_work() {
        let (root, good_group) =
            group_with_ranges(vec![("weights/a.bin", 0, b"abcdefgh")]);
        let mut corrupt_group = good_group.clone();
        corrupt_group.resources[0].ranges[0].sha256 = digest(b"xxxxxxxx");
        let mut missing_group = good_group.clone();
        missing_group.resources[0].ranges[0].artifact_path =
            "weights/missing.bin".to_string();
        let short_group = good_group.clone();
        let store = CompiledResourceBackingStore::new(
            root.path(),
            CompiledResourceBackingStoreLimits {
                worker_count: 1,
                queued_request_capacity: 2,
                maximum_ranges_per_group: 8,
                maximum_logical_bytes_per_group: 1024,
                maximum_retained_payload_bytes: 1024,
                maximum_coalesced_read_bytes: 1024,
                maximum_coalescing_gap_bytes: 0,
            },
        )
        .unwrap();
        assert_eq!(
            store
                .try_load(corrupt_group)
                .unwrap()
                .wait()
                .unwrap_err()
                .kind(),
            CompiledResourceBackingStoreErrorKind::Integrity
        );
        assert_eq!(
            store
                .try_load(missing_group)
                .unwrap()
                .wait()
                .unwrap_err()
                .kind(),
            CompiledResourceBackingStoreErrorKind::Io
        );
        fs::write(root.path().join("weights/a.bin"), b"abcd").unwrap();
        assert_eq!(
            store
                .try_load(short_group)
                .unwrap()
                .wait()
                .unwrap_err()
                .kind(),
            CompiledResourceBackingStoreErrorKind::Io
        );
        fs::write(root.path().join("weights/a.bin"), b"abcdefgh").unwrap();
        let loaded = store.try_load(good_group).unwrap().wait().unwrap();
        assert_eq!(&*loaded.resources[0].ranges[0].bytes, b"abcdefgh");
        assert_eq!(store.statistics().failed_requests, 3);
    }

    #[test]
    fn repeated_backing_store_lifecycles_release_host_memory_workers_and_files() {
        const CYCLE_COUNT: usize = 4;
        const WORKER_COUNT: usize = 2;
        let baseline_file_descriptors = process_file_descriptor_count();
        let baseline_workers = process_named_thread_count("nerve-resource");

        for _ in 0..CYCLE_COUNT {
            let (root, group) =
                group_with_ranges(vec![("weights/a.bin", 0, b"abcdefgh")]);
            let store = CompiledResourceBackingStore::new(
                root.path(),
                CompiledResourceBackingStoreLimits {
                    worker_count: WORKER_COUNT,
                    queued_request_capacity: 2,
                    maximum_ranges_per_group: 8,
                    maximum_logical_bytes_per_group: 1024,
                    maximum_retained_payload_bytes: 1024,
                    maximum_coalesced_read_bytes: 1024,
                    maximum_coalescing_gap_bytes: 0,
                },
            )
            .unwrap();
            assert_eq!(
                process_named_thread_count("nerve-resource"),
                baseline_workers + WORKER_COUNT
            );
            let loaded = store.try_load(group).unwrap().wait().unwrap();
            assert_eq!(store.retained_payload_bytes(), 8);
            drop(loaded);
            assert_eq!(store.retained_payload_bytes(), 0);
            drop(store);
            assert_eq!(
                process_named_thread_count("nerve-resource"),
                baseline_workers
            );
            assert_eq!(
                process_file_descriptor_count(),
                baseline_file_descriptors
            );
        }
    }

    #[test]
    fn workers_execute_requests_asynchronously_with_a_bounded_queue() {
        let (root, group) =
            group_with_ranges(vec![("weights/a.bin", 0, b"abcdefgh")]);
        let store = Arc::new(
            CompiledResourceBackingStore::new(
                root.path(),
                CompiledResourceBackingStoreLimits {
                    worker_count: 2,
                    queued_request_capacity: 2,
                    maximum_ranges_per_group: 8,
                    maximum_logical_bytes_per_group: 1024,
                    maximum_retained_payload_bytes: 1024,
                    maximum_coalesced_read_bytes: 1024,
                    maximum_coalescing_gap_bytes: 0,
                },
            )
            .unwrap(),
        );
        let barrier = Arc::new(Barrier::new(2));
        let caller = {
            let store = Arc::clone(&store);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                let ticket = store.try_load(group).unwrap();
                barrier.wait();
                ticket.wait().unwrap()
            })
        };
        barrier.wait();
        let loaded = caller.join().unwrap();
        assert_eq!(&*loaded.resources[0].ranges[0].bytes, b"abcdefgh");
        assert_eq!(store.statistics().completed_requests, 1);
    }

    #[test]
    fn external_compiled_group_reads_verifies_and_uploads_as_one_cold_load() {
        let package_root = match std::env::var("NERVE_TEST_COMPILED_PACKAGE_ROOT") {
            Ok(path) => PathBuf::from(path),
            Err(std::env::VarError::NotPresent) => {
                eprintln!("skipping external compiled resource load: package root is not set");
                return;
            }
            Err(error) => panic!("could not read external package root: {error}"),
        };
        let device_index = std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX")
            .expect("NERVE_TEST_VULKAN_DEVICE_INDEX must select an idle AMD GPU")
            .parse::<usize>()
            .expect("NERVE_TEST_VULKAN_DEVICE_INDEX must be an integer");
        let partition_template_index = std::env::var(
            "NERVE_TEST_RESIDENCY_PARTITION_TEMPLATE_INDEX",
        )
        .ok()
        .map(|value| value.parse::<usize>().unwrap())
        .unwrap_or(0);
        let partition_index = std::env::var("NERVE_TEST_RESIDENCY_PARTITION_INDEX")
            .ok()
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(0);
        let manifest: VulkanResidentModelPackageManifest = serde_json::from_slice(
            &fs::read(package_root.join("vulkan_resident_package.json")).unwrap(),
        )
        .unwrap();
        let template = manifest
            .resource_residency
            .partition_templates
            .get(partition_template_index)
            .expect("external package has no requested partition template");
        let resolved = resolve_compiled_partition_group(
            &package_root,
            &manifest.resource_residency,
            &template.id,
            partition_index,
        )
        .unwrap();
        for range in resolved
            .resources
            .iter()
            .flat_map(|resource| resource.ranges.iter())
        {
            let source = fs::File::open(package_root.join(&range.artifact_path)).unwrap();
            let result = unsafe {
                libc::posix_fadvise(
                    source.as_raw_fd(),
                    range.byte_offset as libc::off_t,
                    range.byte_count as libc::off_t,
                    libc::POSIX_FADV_DONTNEED,
                )
            };
            assert_eq!(result, 0, "could not evict the external test range");
        }
        let limits = CompiledResourceBackingStoreLimits {
            maximum_ranges_per_group: 256,
            maximum_logical_bytes_per_group: 128 * 1024 * 1024,
            maximum_coalesced_read_bytes: 32 * 1024 * 1024,
            ..Default::default()
        };
        let store = CompiledResourceBackingStore::new(&package_root, limits).unwrap();
        let loaded = store.try_load(resolved).unwrap().wait().unwrap();
        let device = VulkanComputeDevice::new_for_physical_device_index(device_index).unwrap();
        let loaded_ranges = loaded
            .resources
            .iter()
            .flat_map(|resource| resource.ranges.iter())
            .collect::<Vec<_>>();
        let destinations = loaded_ranges
            .iter()
            .map(|range| device.create_resident_buffer(range.bytes.len()).unwrap())
            .collect::<Vec<_>>();
        let writes = loaded_ranges
            .iter()
            .zip(&destinations)
            .map(|(range, destination)| {
                VulkanResidentBufferWriteRange::new(destination, 0, &range.bytes).unwrap()
            })
            .collect::<Vec<_>>();
        let mut transfer = device
            .create_resident_transfer_stream(2, loaded.logical_byte_count)
            .unwrap();
        let upload_started = Instant::now();
        let ticket = transfer.submit(&writes).unwrap();
        transfer.wait(&ticket).unwrap();
        let upload_elapsed = upload_started.elapsed();
        for (range, destination) in loaded_ranges.iter().zip(&destinations) {
            assert_eq!(
                destination.read_bytes(range.bytes.len()).unwrap(),
                &*range.bytes
            );
        }
        let read_gib_per_second = loaded.physical_byte_count as f64
            / loaded.elapsed.as_secs_f64()
            / (1024.0 * 1024.0 * 1024.0);
        let upload_gib_per_second = ticket.uploaded_bytes() as f64
            / upload_elapsed.as_secs_f64()
            / (1024.0 * 1024.0 * 1024.0);
        eprintln!(
            "cold_resource_load group={} logical_ranges={} physical_reads={} bytes={} read_ms={:.3} read_gib_s={:.3} upload_ms={:.3} upload_gib_s={:.3} distinct_transfer_queue={}",
            loaded.id,
            loaded.logical_range_count,
            loaded.physical_read_count,
            loaded.logical_byte_count,
            loaded.elapsed.as_secs_f64() * 1000.0,
            read_gib_per_second,
            upload_elapsed.as_secs_f64() * 1000.0,
            upload_gib_per_second,
            device.has_distinct_transfer_queue(),
        );
    }
}
