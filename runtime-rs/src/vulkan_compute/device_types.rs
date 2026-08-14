pub struct VulkanComputeDevice {
    context: Arc<VulkanInstanceContext>,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    logical_device_lifetime: Arc<VulkanLogicalDeviceLifetime>,
    queue_family_index: u32,
    transfer_queue_is_distinct: bool,
    compute_queue_submission: VulkanQueueSubmissionGate,
    transfer_queue_submission: VulkanQueueSubmissionGate,
    compute_queue_progress_semaphore: vk::Semaphore,
    transfer_queue_progress_semaphore: Option<vk::Semaphore>,
    physical_queue_quiescer: Option<Arc<VulkanPhysicalQueueQuiescer>>,
    activity_lease: RefCell<Option<VulkanDeviceActivityLease>>,
    device_health: VulkanDeviceHealth,
    buffer_device_address_supported: bool,
    api_version: u32,
    physical_device_id: String,
    device_name: String,
    pci_address: Option<String>,
    enabled_device_extensions: BTreeSet<String>,
    enabled_shader_features: BTreeSet<VulkanShaderFeature>,
    shared_host_memory_alignment: Option<usize>,
    shared_device_memory_supported: bool,
    opaque_fd_timeline_semaphore_supported: bool,
    cooperative_bfloat16_shapes: BTreeSet<(u32, u32, u32)>,
    cooperative_float8_e4m3_shapes: BTreeSet<(u32, u32, u32)>,
    cooperative_sint8_shapes: BTreeSet<(u32, u32, u32)>,
    subgroup_size: u32,
    subgroup_supported_stages: vk::ShaderStageFlags,
    subgroup_supported_operations: vk::SubgroupFeatureFlags,
    max_compute_work_group_invocations: u32,
    max_compute_work_group_size_x: u32,
    max_compute_work_group_count_x: u32,
    min_storage_buffer_offset_alignment: usize,
    device_local_memory_bytes: u64,
    memory_budget_supported: bool,
    device_local_memory_budget: VulkanDeviceLocalMemoryBudget,
    device_local_memory_budget_tracker: Arc<Mutex<VulkanDeviceLocalMemoryBudgetTracker>>,
    timestamp_period_ns: f32,
    conditional_rendering: Option<ash::ext::conditional_rendering::Device>,
    device_fault: Option<ash::ext::device_fault::Device>,
    device_address_registry: Arc<Mutex<VulkanDeviceAddressRegistry>>,
    generic_storage_pipelines: RefCell<HashMap<VulkanGenericPipelineKey, VulkanStoragePipeline>>,
    immediate_kernel_sequence: RefCell<Option<VulkanResidentKernelSequence>>,
}

#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    Serialize,
    Deserialize,
)]
pub struct VulkanResidentExecutionCounters {
    pub resident_sequence_prepare_calls: u64,
    pub resident_sequence_recorded_command_buffers: u64,
    pub resident_sequence_reused_command_buffers: u64,
    pub resident_sequence_queue_submits: u64,
    pub resident_sequence_completion_waits: u64,
    pub resident_queue_batch_submits: u64,
    pub resident_queue_batch_commands: u64,
    pub resident_copy_queue_submits: u64,
    pub resident_copy_waits: u64,
    pub demand_initial_sequence_count: u64,
    pub demand_initial_device_duration_ns: u64,
    pub demand_initial_max_device_duration_ns: u64,
    pub demand_resume_sequence_count: u64,
    pub demand_resume_device_duration_ns: u64,
    pub demand_resume_max_device_duration_ns: u64,
    pub resident_component_sequence_count: u64,
    pub resident_component_device_duration_ns: u64,
    pub resident_component_max_device_duration_ns: u64,
    pub execution_quantum_count: u64,
    pub execution_quantum_region_count: u64,
    pub execution_quantum_forced_yield_count: u64,
    pub execution_quantum_estimated_work_units: u64,
    pub execution_quantum_estimated_memory_bytes: u64,
    pub execution_quantum_dispatch_count: u64,
    pub execution_quantum_predicted_duration_ns: u64,
    pub execution_quantum_host_submit_wait_duration_ns: u64,
    pub execution_quantum_max_region_count: u64,
    pub execution_quantum_max_host_submit_wait_duration_ns: u64,
    pub distributed: VulkanResidentDistributedExecutionCounters,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanResidentDistributedExecutionCounters {
    pub decode: VulkanResidentDistributedExecutionPhaseCounters,
    pub prefill: VulkanResidentDistributedExecutionPhaseCounters,
}

impl VulkanResidentExecutionCounters {
    pub fn saturating_accumulate(&mut self, value: Self) {
        self.resident_sequence_prepare_calls = self
            .resident_sequence_prepare_calls
            .saturating_add(value.resident_sequence_prepare_calls);
        self.resident_sequence_recorded_command_buffers = self
            .resident_sequence_recorded_command_buffers
            .saturating_add(value.resident_sequence_recorded_command_buffers);
        self.resident_sequence_reused_command_buffers = self
            .resident_sequence_reused_command_buffers
            .saturating_add(value.resident_sequence_reused_command_buffers);
        self.resident_sequence_queue_submits = self
            .resident_sequence_queue_submits
            .saturating_add(value.resident_sequence_queue_submits);
        self.resident_sequence_completion_waits = self
            .resident_sequence_completion_waits
            .saturating_add(value.resident_sequence_completion_waits);
        self.resident_queue_batch_submits = self
            .resident_queue_batch_submits
            .saturating_add(value.resident_queue_batch_submits);
        self.resident_queue_batch_commands = self
            .resident_queue_batch_commands
            .saturating_add(value.resident_queue_batch_commands);
        self.resident_copy_queue_submits = self
            .resident_copy_queue_submits
            .saturating_add(value.resident_copy_queue_submits);
        self.resident_copy_waits = self
            .resident_copy_waits
            .saturating_add(value.resident_copy_waits);
        self.demand_initial_sequence_count = self
            .demand_initial_sequence_count
            .saturating_add(value.demand_initial_sequence_count);
        self.demand_initial_device_duration_ns = self
            .demand_initial_device_duration_ns
            .saturating_add(value.demand_initial_device_duration_ns);
        self.demand_initial_max_device_duration_ns = self
            .demand_initial_max_device_duration_ns
            .max(value.demand_initial_max_device_duration_ns);
        self.demand_resume_sequence_count = self
            .demand_resume_sequence_count
            .saturating_add(value.demand_resume_sequence_count);
        self.demand_resume_device_duration_ns = self
            .demand_resume_device_duration_ns
            .saturating_add(value.demand_resume_device_duration_ns);
        self.demand_resume_max_device_duration_ns = self
            .demand_resume_max_device_duration_ns
            .max(value.demand_resume_max_device_duration_ns);
        self.resident_component_sequence_count = self
            .resident_component_sequence_count
            .saturating_add(value.resident_component_sequence_count);
        self.resident_component_device_duration_ns = self
            .resident_component_device_duration_ns
            .saturating_add(value.resident_component_device_duration_ns);
        self.resident_component_max_device_duration_ns = self
            .resident_component_max_device_duration_ns
            .max(value.resident_component_max_device_duration_ns);
        self.execution_quantum_count = self
            .execution_quantum_count
            .saturating_add(value.execution_quantum_count);
        self.execution_quantum_region_count = self
            .execution_quantum_region_count
            .saturating_add(value.execution_quantum_region_count);
        self.execution_quantum_forced_yield_count = self
            .execution_quantum_forced_yield_count
            .saturating_add(value.execution_quantum_forced_yield_count);
        self.execution_quantum_estimated_work_units = self
            .execution_quantum_estimated_work_units
            .saturating_add(value.execution_quantum_estimated_work_units);
        self.execution_quantum_estimated_memory_bytes = self
            .execution_quantum_estimated_memory_bytes
            .saturating_add(value.execution_quantum_estimated_memory_bytes);
        self.execution_quantum_dispatch_count = self
            .execution_quantum_dispatch_count
            .saturating_add(value.execution_quantum_dispatch_count);
        self.execution_quantum_predicted_duration_ns = self
            .execution_quantum_predicted_duration_ns
            .saturating_add(value.execution_quantum_predicted_duration_ns);
        self.execution_quantum_host_submit_wait_duration_ns = self
            .execution_quantum_host_submit_wait_duration_ns
            .saturating_add(value.execution_quantum_host_submit_wait_duration_ns);
        self.execution_quantum_max_region_count = self
            .execution_quantum_max_region_count
            .max(value.execution_quantum_max_region_count);
        self.execution_quantum_max_host_submit_wait_duration_ns = self
            .execution_quantum_max_host_submit_wait_duration_ns
            .max(value.execution_quantum_max_host_submit_wait_duration_ns);
        self.distributed.saturating_accumulate(value.distributed);
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanResidentDistributedExecutionPhaseCounters {
    pub island_submissions: u64,
    pub shard_submissions: u64,
    pub tensor_parallel_island_submissions: u64,
    pub whole_expert_parallel_island_submissions: u64,
    pub intra_expert_tensor_parallel_island_submissions: u64,
    pub hybrid_island_submissions: u64,
}

impl VulkanResidentDistributedExecutionPhaseCounters {
    fn saturating_accumulate(&mut self, value: Self) {
        self.island_submissions = self
            .island_submissions
            .saturating_add(value.island_submissions);
        self.shard_submissions = self
            .shard_submissions
            .saturating_add(value.shard_submissions);
        self.tensor_parallel_island_submissions = self
            .tensor_parallel_island_submissions
            .saturating_add(value.tensor_parallel_island_submissions);
        self.whole_expert_parallel_island_submissions = self
            .whole_expert_parallel_island_submissions
            .saturating_add(value.whole_expert_parallel_island_submissions);
        self.intra_expert_tensor_parallel_island_submissions = self
            .intra_expert_tensor_parallel_island_submissions
            .saturating_add(value.intra_expert_tensor_parallel_island_submissions);
        self.hybrid_island_submissions = self
            .hybrid_island_submissions
            .saturating_add(value.hybrid_island_submissions);
    }
}

impl VulkanResidentDistributedExecutionCounters {
    fn saturating_accumulate(&mut self, value: Self) {
        self.decode.saturating_accumulate(value.decode);
        self.prefill.saturating_accumulate(value.prefill);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VulkanResidentDistributedExecutionPhase {
    Decode,
    Prefill,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VulkanResidentDistributedExecutionKind {
    TensorParallel,
    WholeExpertParallel,
    IntraExpertTensorParallel,
    Hybrid,
}

struct VulkanResidentDistributedExecutionCounterBank {
    island_submissions: AtomicU64,
    shard_submissions: AtomicU64,
    tensor_parallel_island_submissions: AtomicU64,
    whole_expert_parallel_island_submissions: AtomicU64,
    intra_expert_tensor_parallel_island_submissions: AtomicU64,
    hybrid_island_submissions: AtomicU64,
}

impl VulkanResidentDistributedExecutionCounterBank {
    const fn new() -> Self {
        Self {
            island_submissions: AtomicU64::new(0),
            shard_submissions: AtomicU64::new(0),
            tensor_parallel_island_submissions: AtomicU64::new(0),
            whole_expert_parallel_island_submissions: AtomicU64::new(0),
            intra_expert_tensor_parallel_island_submissions: AtomicU64::new(0),
            hybrid_island_submissions: AtomicU64::new(0),
        }
    }

    fn reset(&self) {
        self.island_submissions.store(0, Ordering::Relaxed);
        self.shard_submissions.store(0, Ordering::Relaxed);
        self.tensor_parallel_island_submissions
            .store(0, Ordering::Relaxed);
        self.whole_expert_parallel_island_submissions
            .store(0, Ordering::Relaxed);
        self.intra_expert_tensor_parallel_island_submissions
            .store(0, Ordering::Relaxed);
        self.hybrid_island_submissions.store(0, Ordering::Relaxed);
    }

    fn snapshot(&self) -> VulkanResidentDistributedExecutionPhaseCounters {
        VulkanResidentDistributedExecutionPhaseCounters {
            island_submissions: self.island_submissions.load(Ordering::Relaxed),
            shard_submissions: self.shard_submissions.load(Ordering::Relaxed),
            tensor_parallel_island_submissions: self
                .tensor_parallel_island_submissions
                .load(Ordering::Relaxed),
            whole_expert_parallel_island_submissions: self
                .whole_expert_parallel_island_submissions
                .load(Ordering::Relaxed),
            intra_expert_tensor_parallel_island_submissions: self
                .intra_expert_tensor_parallel_island_submissions
                .load(Ordering::Relaxed),
            hybrid_island_submissions: self
                .hybrid_island_submissions
                .load(Ordering::Relaxed),
        }
    }

    fn record(&self, kind: VulkanResidentDistributedExecutionKind, shard_count: usize) {
        self.island_submissions.fetch_add(1, Ordering::Relaxed);
        self.shard_submissions.fetch_add(
            u64::try_from(shard_count).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        let strategy_counter = match kind {
            VulkanResidentDistributedExecutionKind::TensorParallel => {
                &self.tensor_parallel_island_submissions
            }
            VulkanResidentDistributedExecutionKind::WholeExpertParallel => {
                &self.whole_expert_parallel_island_submissions
            }
            VulkanResidentDistributedExecutionKind::IntraExpertTensorParallel => {
                &self.intra_expert_tensor_parallel_island_submissions
            }
            VulkanResidentDistributedExecutionKind::Hybrid => &self.hybrid_island_submissions,
        };
        strategy_counter.fetch_add(1, Ordering::Relaxed);
    }
}

static DISTRIBUTED_DECODE_EXECUTION_COUNTERS: VulkanResidentDistributedExecutionCounterBank =
    VulkanResidentDistributedExecutionCounterBank::new();
static DISTRIBUTED_PREFILL_EXECUTION_COUNTERS: VulkanResidentDistributedExecutionCounterBank =
    VulkanResidentDistributedExecutionCounterBank::new();

static RESIDENT_SEQUENCE_PREPARE_CALLS: AtomicU64 = AtomicU64::new(0);
static RESIDENT_SEQUENCE_RECORDED_COMMAND_BUFFERS: AtomicU64 = AtomicU64::new(0);
static RESIDENT_SEQUENCE_REUSED_COMMAND_BUFFERS: AtomicU64 = AtomicU64::new(0);
static RESIDENT_SEQUENCE_QUEUE_SUBMITS: AtomicU64 = AtomicU64::new(0);
static RESIDENT_SEQUENCE_COMPLETION_WAITS: AtomicU64 = AtomicU64::new(0);
static RESIDENT_QUEUE_BATCH_SUBMITS: AtomicU64 = AtomicU64::new(0);
static RESIDENT_QUEUE_BATCH_COMMANDS: AtomicU64 = AtomicU64::new(0);
static RESIDENT_COPY_QUEUE_SUBMITS: AtomicU64 = AtomicU64::new(0);
static RESIDENT_COPY_WAITS: AtomicU64 = AtomicU64::new(0);
static DEMAND_INITIAL_SEQUENCE_COUNT: AtomicU64 = AtomicU64::new(0);
static DEMAND_INITIAL_DEVICE_DURATION_NS: AtomicU64 = AtomicU64::new(0);
static DEMAND_INITIAL_MAX_DEVICE_DURATION_NS: AtomicU64 = AtomicU64::new(0);
static DEMAND_RESUME_SEQUENCE_COUNT: AtomicU64 = AtomicU64::new(0);
static DEMAND_RESUME_DEVICE_DURATION_NS: AtomicU64 = AtomicU64::new(0);
static DEMAND_RESUME_MAX_DEVICE_DURATION_NS: AtomicU64 = AtomicU64::new(0);
static RESIDENT_COMPONENT_SEQUENCE_COUNT: AtomicU64 = AtomicU64::new(0);
static RESIDENT_COMPONENT_DEVICE_DURATION_NS: AtomicU64 = AtomicU64::new(0);
static RESIDENT_COMPONENT_MAX_DEVICE_DURATION_NS: AtomicU64 = AtomicU64::new(0);
static EXECUTION_QUANTUM_COUNT: AtomicU64 = AtomicU64::new(0);
static EXECUTION_QUANTUM_REGION_COUNT: AtomicU64 = AtomicU64::new(0);
static EXECUTION_QUANTUM_FORCED_YIELD_COUNT: AtomicU64 = AtomicU64::new(0);
static EXECUTION_QUANTUM_ESTIMATED_WORK_UNITS: AtomicU64 = AtomicU64::new(0);
static EXECUTION_QUANTUM_ESTIMATED_MEMORY_BYTES: AtomicU64 = AtomicU64::new(0);
static EXECUTION_QUANTUM_DISPATCH_COUNT: AtomicU64 = AtomicU64::new(0);
static EXECUTION_QUANTUM_PREDICTED_DURATION_NS: AtomicU64 = AtomicU64::new(0);
static EXECUTION_QUANTUM_HOST_SUBMIT_WAIT_DURATION_NS: AtomicU64 = AtomicU64::new(0);
static EXECUTION_QUANTUM_MAX_REGION_COUNT: AtomicU64 = AtomicU64::new(0);
static EXECUTION_QUANTUM_MAX_HOST_SUBMIT_WAIT_DURATION_NS: AtomicU64 = AtomicU64::new(0);

pub fn reset_vulkan_resident_execution_counters() {
    RESIDENT_SEQUENCE_PREPARE_CALLS.store(0, Ordering::Relaxed);
    RESIDENT_SEQUENCE_RECORDED_COMMAND_BUFFERS.store(0, Ordering::Relaxed);
    RESIDENT_SEQUENCE_REUSED_COMMAND_BUFFERS.store(0, Ordering::Relaxed);
    RESIDENT_SEQUENCE_QUEUE_SUBMITS.store(0, Ordering::Relaxed);
    RESIDENT_SEQUENCE_COMPLETION_WAITS.store(0, Ordering::Relaxed);
    RESIDENT_QUEUE_BATCH_SUBMITS.store(0, Ordering::Relaxed);
    RESIDENT_QUEUE_BATCH_COMMANDS.store(0, Ordering::Relaxed);
    RESIDENT_COPY_QUEUE_SUBMITS.store(0, Ordering::Relaxed);
    RESIDENT_COPY_WAITS.store(0, Ordering::Relaxed);
    DEMAND_INITIAL_SEQUENCE_COUNT.store(0, Ordering::Relaxed);
    DEMAND_INITIAL_DEVICE_DURATION_NS.store(0, Ordering::Relaxed);
    DEMAND_INITIAL_MAX_DEVICE_DURATION_NS.store(0, Ordering::Relaxed);
    DEMAND_RESUME_SEQUENCE_COUNT.store(0, Ordering::Relaxed);
    DEMAND_RESUME_DEVICE_DURATION_NS.store(0, Ordering::Relaxed);
    DEMAND_RESUME_MAX_DEVICE_DURATION_NS.store(0, Ordering::Relaxed);
    RESIDENT_COMPONENT_SEQUENCE_COUNT.store(0, Ordering::Relaxed);
    RESIDENT_COMPONENT_DEVICE_DURATION_NS.store(0, Ordering::Relaxed);
    RESIDENT_COMPONENT_MAX_DEVICE_DURATION_NS.store(0, Ordering::Relaxed);
    EXECUTION_QUANTUM_COUNT.store(0, Ordering::Relaxed);
    EXECUTION_QUANTUM_REGION_COUNT.store(0, Ordering::Relaxed);
    EXECUTION_QUANTUM_FORCED_YIELD_COUNT.store(0, Ordering::Relaxed);
    EXECUTION_QUANTUM_ESTIMATED_WORK_UNITS.store(0, Ordering::Relaxed);
    EXECUTION_QUANTUM_ESTIMATED_MEMORY_BYTES.store(0, Ordering::Relaxed);
    EXECUTION_QUANTUM_DISPATCH_COUNT.store(0, Ordering::Relaxed);
    EXECUTION_QUANTUM_PREDICTED_DURATION_NS.store(0, Ordering::Relaxed);
    EXECUTION_QUANTUM_HOST_SUBMIT_WAIT_DURATION_NS.store(0, Ordering::Relaxed);
    EXECUTION_QUANTUM_MAX_REGION_COUNT.store(0, Ordering::Relaxed);
    EXECUTION_QUANTUM_MAX_HOST_SUBMIT_WAIT_DURATION_NS.store(0, Ordering::Relaxed);
    DISTRIBUTED_DECODE_EXECUTION_COUNTERS.reset();
    DISTRIBUTED_PREFILL_EXECUTION_COUNTERS.reset();
}

pub fn vulkan_resident_execution_counters() -> VulkanResidentExecutionCounters {
    VulkanResidentExecutionCounters {
        resident_sequence_prepare_calls: RESIDENT_SEQUENCE_PREPARE_CALLS.load(Ordering::Relaxed),
        resident_sequence_recorded_command_buffers: RESIDENT_SEQUENCE_RECORDED_COMMAND_BUFFERS
            .load(Ordering::Relaxed),
        resident_sequence_reused_command_buffers: RESIDENT_SEQUENCE_REUSED_COMMAND_BUFFERS
            .load(Ordering::Relaxed),
        resident_sequence_queue_submits: RESIDENT_SEQUENCE_QUEUE_SUBMITS.load(Ordering::Relaxed),
        resident_sequence_completion_waits: RESIDENT_SEQUENCE_COMPLETION_WAITS
            .load(Ordering::Relaxed),
        resident_queue_batch_submits: RESIDENT_QUEUE_BATCH_SUBMITS.load(Ordering::Relaxed),
        resident_queue_batch_commands: RESIDENT_QUEUE_BATCH_COMMANDS.load(Ordering::Relaxed),
        resident_copy_queue_submits: RESIDENT_COPY_QUEUE_SUBMITS.load(Ordering::Relaxed),
        resident_copy_waits: RESIDENT_COPY_WAITS.load(Ordering::Relaxed),
        demand_initial_sequence_count: DEMAND_INITIAL_SEQUENCE_COUNT.load(Ordering::Relaxed),
        demand_initial_device_duration_ns: DEMAND_INITIAL_DEVICE_DURATION_NS
            .load(Ordering::Relaxed),
        demand_initial_max_device_duration_ns: DEMAND_INITIAL_MAX_DEVICE_DURATION_NS
            .load(Ordering::Relaxed),
        demand_resume_sequence_count: DEMAND_RESUME_SEQUENCE_COUNT.load(Ordering::Relaxed),
        demand_resume_device_duration_ns: DEMAND_RESUME_DEVICE_DURATION_NS
            .load(Ordering::Relaxed),
        demand_resume_max_device_duration_ns: DEMAND_RESUME_MAX_DEVICE_DURATION_NS
            .load(Ordering::Relaxed),
        resident_component_sequence_count: RESIDENT_COMPONENT_SEQUENCE_COUNT
            .load(Ordering::Relaxed),
        resident_component_device_duration_ns: RESIDENT_COMPONENT_DEVICE_DURATION_NS
            .load(Ordering::Relaxed),
        resident_component_max_device_duration_ns: RESIDENT_COMPONENT_MAX_DEVICE_DURATION_NS
            .load(Ordering::Relaxed),
        execution_quantum_count: EXECUTION_QUANTUM_COUNT.load(Ordering::Relaxed),
        execution_quantum_region_count: EXECUTION_QUANTUM_REGION_COUNT.load(Ordering::Relaxed),
        execution_quantum_forced_yield_count: EXECUTION_QUANTUM_FORCED_YIELD_COUNT
            .load(Ordering::Relaxed),
        execution_quantum_estimated_work_units: EXECUTION_QUANTUM_ESTIMATED_WORK_UNITS
            .load(Ordering::Relaxed),
        execution_quantum_estimated_memory_bytes: EXECUTION_QUANTUM_ESTIMATED_MEMORY_BYTES
            .load(Ordering::Relaxed),
        execution_quantum_dispatch_count: EXECUTION_QUANTUM_DISPATCH_COUNT.load(Ordering::Relaxed),
        execution_quantum_predicted_duration_ns: EXECUTION_QUANTUM_PREDICTED_DURATION_NS
            .load(Ordering::Relaxed),
        execution_quantum_host_submit_wait_duration_ns:
            EXECUTION_QUANTUM_HOST_SUBMIT_WAIT_DURATION_NS.load(Ordering::Relaxed),
        execution_quantum_max_region_count: EXECUTION_QUANTUM_MAX_REGION_COUNT
            .load(Ordering::Relaxed),
        execution_quantum_max_host_submit_wait_duration_ns:
            EXECUTION_QUANTUM_MAX_HOST_SUBMIT_WAIT_DURATION_NS.load(Ordering::Relaxed),
        distributed: VulkanResidentDistributedExecutionCounters {
            decode: DISTRIBUTED_DECODE_EXECUTION_COUNTERS.snapshot(),
            prefill: DISTRIBUTED_PREFILL_EXECUTION_COUNTERS.snapshot(),
        },
    }
}

pub(crate) fn record_vulkan_resident_distributed_execution_submission(
    phase: VulkanResidentDistributedExecutionPhase,
    kind: VulkanResidentDistributedExecutionKind,
    shard_count: usize,
) {
    match phase {
        VulkanResidentDistributedExecutionPhase::Decode => {
            DISTRIBUTED_DECODE_EXECUTION_COUNTERS.record(kind, shard_count)
        }
        VulkanResidentDistributedExecutionPhase::Prefill => {
            DISTRIBUTED_PREFILL_EXECUTION_COUNTERS.record(kind, shard_count)
        }
    }
}

pub(crate) fn record_vulkan_resident_component_sequence_device_duration(duration_ns: u64) {
    RESIDENT_COMPONENT_SEQUENCE_COUNT.fetch_add(1, Ordering::Relaxed);
    RESIDENT_COMPONENT_DEVICE_DURATION_NS.fetch_add(duration_ns, Ordering::Relaxed);
    RESIDENT_COMPONENT_MAX_DEVICE_DURATION_NS.fetch_max(duration_ns, Ordering::Relaxed);
}

pub(crate) fn record_vulkan_demand_sequence_device_duration(
    resumed: bool,
    duration_ns: u64,
) {
    let (count, total, maximum) = if resumed {
        (
            &DEMAND_RESUME_SEQUENCE_COUNT,
            &DEMAND_RESUME_DEVICE_DURATION_NS,
            &DEMAND_RESUME_MAX_DEVICE_DURATION_NS,
        )
    } else {
        (
            &DEMAND_INITIAL_SEQUENCE_COUNT,
            &DEMAND_INITIAL_DEVICE_DURATION_NS,
            &DEMAND_INITIAL_MAX_DEVICE_DURATION_NS,
        )
    };
    count.fetch_add(1, Ordering::Relaxed);
    total.fetch_add(duration_ns, Ordering::Relaxed);
    maximum.fetch_max(duration_ns, Ordering::Relaxed);
}

pub(crate) fn record_vulkan_execution_quantum_measurement(
    measurement: &VulkanResidentExecutionQuantumMeasurement,
) {
    EXECUTION_QUANTUM_COUNT.fetch_add(1, Ordering::Relaxed);
    EXECUTION_QUANTUM_REGION_COUNT.fetch_add(
        u64::try_from(measurement.region_count).unwrap_or(u64::MAX),
        Ordering::Relaxed,
    );
    EXECUTION_QUANTUM_FORCED_YIELD_COUNT.fetch_add(
        u64::from(measurement.forced_yield_after),
        Ordering::Relaxed,
    );
    EXECUTION_QUANTUM_ESTIMATED_WORK_UNITS
        .fetch_add(measurement.cost.work_units, Ordering::Relaxed);
    EXECUTION_QUANTUM_ESTIMATED_MEMORY_BYTES
        .fetch_add(measurement.cost.memory_bytes, Ordering::Relaxed);
    EXECUTION_QUANTUM_DISPATCH_COUNT
        .fetch_add(measurement.cost.dispatches, Ordering::Relaxed);
    EXECUTION_QUANTUM_PREDICTED_DURATION_NS.fetch_add(
        measurement.cost.predicted_duration_ns,
        Ordering::Relaxed,
    );
    EXECUTION_QUANTUM_HOST_SUBMIT_WAIT_DURATION_NS
        .fetch_add(measurement.duration_ns, Ordering::Relaxed);
    EXECUTION_QUANTUM_MAX_REGION_COUNT.fetch_max(
        u64::try_from(measurement.region_count).unwrap_or(u64::MAX),
        Ordering::Relaxed,
    );
    EXECUTION_QUANTUM_MAX_HOST_SUBMIT_WAIT_DURATION_NS
        .fetch_max(measurement.duration_ns, Ordering::Relaxed);
}

struct VulkanInstanceContext {
    _entry: Entry,
    instance: ash::Instance,
    device_local_memory_budget_trackers:
        Mutex<BTreeMap<String, std::sync::Weak<Mutex<VulkanDeviceLocalMemoryBudgetTracker>>>>,
    host_memory_budget_tracker: Arc<Mutex<VulkanHostMemoryBudgetTracker>>,
}

/// Owns the Vulkan logical device independently from any one runtime wrapper.
///
/// Recorded queue templates and their timeline semaphores are deliberately
/// reusable after the temporary object that opened the device has gone away.
/// Retaining this object makes that lifetime contract physical: the logical
/// device is destroyed only after its final retained child resource.
struct VulkanLogicalDeviceLifetime {
    device: ash::Device,
    _instance_context: Arc<VulkanInstanceContext>,
}

impl Drop for VulkanLogicalDeviceLifetime {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
        }
    }
}

impl Drop for VulkanInstanceContext {
    fn drop(&mut self) {
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}

pub struct VulkanComputeDeviceCatalog {
    context: Arc<VulkanInstanceContext>,
    physical_devices: Vec<vk::PhysicalDevice>,
    available_devices: Vec<VulkanComputeDeviceInfo>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanComputeDeviceInfo {
    pub physical_device_index: usize,
    pub physical_device_id: String,
    pub device_uuid: [u8; vk::UUID_SIZE],
    pub device_name: String,
    pub pci_address: Option<String>,
    pub device_type: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub api_version: u32,
    pub driver_version: u32,
    pub compute_queue_family_indices: Vec<u32>,
    pub memory_heaps: Vec<VulkanMemoryHeapInfo>,
    pub selected_by_default: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VulkanComputeTargetCapabilities {
    pub physical_device_index: usize,
    pub physical_device_id: String,
    pub device_name: String,
    pub device_type: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub shader_features: BTreeSet<VulkanShaderFeature>,
    pub subgroup_operations: BTreeSet<VulkanSubgroupOperation>,
    pub subgroup_compute_supported: bool,
    pub subgroup_size: u32,
    pub max_compute_work_group_invocations: u32,
    pub max_compute_work_group_size_x: u32,
    pub cooperative_float16_shapes: BTreeSet<(u32, u32, u32)>,
    pub cooperative_bfloat16_shapes: BTreeSet<(u32, u32, u32)>,
    pub cooperative_float8_e4m3_shapes: BTreeSet<(u32, u32, u32)>,
    pub cooperative_sint8_shapes: BTreeSet<(u32, u32, u32)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanMemoryHeapInfo {
    pub heap_index: u32,
    pub size_bytes: u64,
    pub device_local: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VulkanDeviceLocalMemorySnapshot {
    pub physical_device_id: String,
    pub device_name: String,
    pub pci_address: Option<String>,
    pub heap_index: u32,
    pub physical_heap_bytes: u64,
    pub memory_budget_supported: bool,
    pub budget_bytes: Option<u64>,
    pub usage_bytes: Option<u64>,
    pub available_bytes: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct VulkanGenericPipelineKey {
    spirv_words: Vec<u32>,
    descriptor_bindings: Vec<u32>,
    push_constant_byte_count: u32,
    local_size_x: u32,
}

struct VulkanStoragePipeline {
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    shader_module: vk::ShaderModule,
    pipeline: vk::Pipeline,
}

pub struct VulkanResidentBuffer {
    device: ash::Device,
    buffer: vk::Buffer,
    memory: Option<vk::DeviceMemory>,
    memory_access: VulkanResidentMemoryAccess,
    byte_capacity: vk::DeviceSize,
    device_address: Option<vk::DeviceAddress>,
    device_address_registry: Option<Arc<Mutex<VulkanDeviceAddressRegistry>>>,
    persistent_mapping: Option<usize>,
    persistent_mapping_requires_unmap: bool,
    _shared_host_allocation: Option<Arc<VulkanSharedHostAllocation>>,
    _shared_device_memory_identity: Option<Arc<VulkanSharedDeviceMemoryIdentity>>,
    _device_local_memory_reservation: Option<Arc<VulkanDeviceLocalMemoryReservation>>,
}

/// Page-aligned host memory imported into multiple Vulkan devices. GPUs access
/// the same bytes directly; the host does not relay activation data.
pub struct VulkanSharedHostAllocation {
    address: usize,
    layout: Layout,
    byte_capacity: usize,
    _host_memory_reservation: Option<Arc<VulkanHostMemoryReservation>>,
}

/// Identity shared by device-local buffers that import the same external
/// memory payload. Each Vulkan logical device owns its buffer and memory
/// handles; this identity only proves that their bytes alias.
pub struct VulkanSharedDeviceMemoryIdentity;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VulkanSharedResidentBufferRoute {
    ExternalDeviceLocal,
    SharedHost,
}

pub struct VulkanSharedResidentBufferSet {
    pub route: VulkanSharedResidentBufferRoute,
    /// The owner buffer is first, followed by one buffer for each peer in the
    /// same order supplied to `create_shared_resident_buffers`.
    pub buffers: Vec<Arc<VulkanResidentBuffer>>,
    pub external_device_local_error: Option<String>,
}

pub struct VulkanTimelineSemaphore {
    device: ash::Device,
    device_handle: vk::Device,
    semaphore: vk::Semaphore,
    opaque_fd_exportable: bool,
    permanent_opaque_fd_imported: Cell<bool>,
    _logical_device_lifetime: Arc<VulkanLogicalDeviceLifetime>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct VulkanTimelineSemaphoreReplayIdentity {
    device_handle: u64,
    semaphore_handle: u64,
}

/// The logical next value of every timeline semaphore participating in a
/// replayable queue topology. Imported handles are deliberately separate
/// entries: Vulkan may assign a different handle to the same external
/// semaphore on each logical device.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VulkanTimelineSemaphoreReplayState {
    next_values: BTreeMap<VulkanTimelineSemaphoreReplayIdentity, u64>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VulkanTimelineSemaphoreValueRebase {
    offsets: BTreeMap<VulkanTimelineSemaphoreReplayIdentity, u64>,
}

impl VulkanTimelineSemaphoreReplayState {
    pub fn capture(
        &mut self,
        semaphore: &VulkanTimelineSemaphore,
        next_value: u64,
    ) -> Result<(), VulkanError> {
        let identity = semaphore.replay_identity();
        if let Some(previous) = self.next_values.insert(identity, next_value)
            && previous != next_value
        {
            return Err(VulkanError(format!(
                "timeline semaphore replay state records conflicting next values {previous} and {next_value}"
            )));
        }
        Ok(())
    }

    pub fn rebase_to(
        &self,
        current: &Self,
    ) -> Result<VulkanTimelineSemaphoreValueRebase, VulkanError> {
        if self.next_values.keys().ne(current.next_values.keys()) {
            return Err(VulkanError(
                "timeline semaphore replay topology changed between recording and replay"
                    .to_string(),
            ));
        }
        let offsets = self
            .next_values
            .iter()
            .map(|(identity, recorded)| {
                let current = current
                    .next_values
                    .get(identity)
                    .expect("validated replay topology has the same semaphore keys");
                let offset = current.checked_sub(*recorded).ok_or_else(|| {
                    VulkanError(format!(
                        "timeline semaphore replay next value regressed from {recorded} to {current}"
                    ))
                })?;
                Ok((*identity, offset))
            })
            .collect::<Result<_, _>>()?;
        Ok(VulkanTimelineSemaphoreValueRebase { offsets })
    }
}

impl VulkanTimelineSemaphore {
    fn replay_identity(&self) -> VulkanTimelineSemaphoreReplayIdentity {
        VulkanTimelineSemaphoreReplayIdentity {
            device_handle: self.device_handle.as_raw(),
            semaphore_handle: self.semaphore.as_raw(),
        }
    }
}

impl VulkanTimelineSemaphoreValueRebase {
    fn value(
        &self,
        device_handle: vk::Device,
        semaphore: vk::Semaphore,
        recorded_value: u64,
    ) -> Result<u64, VulkanError> {
        let identity = VulkanTimelineSemaphoreReplayIdentity {
            device_handle: device_handle.as_raw(),
            semaphore_handle: semaphore.as_raw(),
        };
        let offset = self.offsets.get(&identity).ok_or_else(|| {
            VulkanError(format!(
                "queue template references timeline semaphore {} on device {} outside its replay state",
                identity.semaphore_handle, identity.device_handle
            ))
        })?;
        offset_timeline_value(recorded_value, *offset)
    }
}

#[derive(Clone, Copy)]
pub struct VulkanTimelineSemaphorePoint<'a> {
    semaphore: &'a VulkanTimelineSemaphore,
    value: u64,
}

impl<'a> VulkanTimelineSemaphorePoint<'a> {
    pub fn new(semaphore: &'a VulkanTimelineSemaphore, value: u64) -> Self {
        Self { semaphore, value }
    }
}

/// Collects already-recorded resident command buffers by logical device.
/// Timeline waits and signals remain attached to their original command. A
/// bounded batch is partitioned into execution quanta before submission so a
/// complete graph cannot accidentally become one watchdog-visible GPU job.
pub struct VulkanResidentQueueSubmissionBatch<'a> {
    groups: RefCell<Vec<VulkanResidentQueueSubmissionGroup<'a>>>,
    distributed_execution_observations:
        RefCell<Vec<VulkanResidentDistributedExecutionObservation>>,
    quantum_budget: Option<RuntimeExecutionQuantumBudget>,
}

/// A mounted queue-submission topology. Command buffers, queue ordering, and
/// semaphore edges stay fixed; replay only advances timeline values. The
/// template owns the lightweight queue handles it needs, so its lifetime is
/// independent from the temporary references used while recording it.
pub struct VulkanResidentQueueSubmissionTemplate {
    groups: Vec<VulkanResidentQueueSubmissionTemplateGroup>,
    submission_count: usize,
    distributed_execution_observations: Vec<VulkanResidentDistributedExecutionObservation>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct VulkanResidentDistributedExecutionObservation {
    phase: VulkanResidentDistributedExecutionPhase,
    kind: VulkanResidentDistributedExecutionKind,
    shard_count: usize,
}

impl VulkanResidentDistributedExecutionObservation {
    fn new(
        phase: VulkanResidentDistributedExecutionPhase,
        kind: VulkanResidentDistributedExecutionKind,
        shard_count: usize,
    ) -> Result<Self, VulkanError> {
        if shard_count == 0 {
            return Err(VulkanError(
                "distributed execution observation has no submitted shards".to_string(),
            ));
        }
        Ok(Self {
            phase,
            kind,
            shard_count,
        })
    }

    fn record(self) {
        record_vulkan_resident_distributed_execution_submission(
            self.phase,
            self.kind,
            self.shard_count,
        );
    }
}

struct VulkanResidentQueueSubmissionGroup<'a> {
    device: &'a VulkanComputeDevice,
    submissions: Vec<VulkanPreparedResidentQueueSubmission>,
    quantum_ranges: Vec<std::ops::Range<usize>>,
    quanta: Vec<Option<RuntimeExecutionQuantum>>,
}

struct VulkanResidentQueueSubmissionTemplateGroup {
    submitter: VulkanResidentQueueSubmitter,
    submissions: Vec<VulkanPreparedResidentQueueSubmission>,
    quantum_ranges: Vec<std::ops::Range<usize>>,
    quanta: Vec<Option<RuntimeExecutionQuantum>>,
}

#[derive(Clone)]
struct VulkanResidentQueueSubmitter {
    device: ash::Device,
    device_handle: vk::Device,
    queue_submission: VulkanQueueSubmissionGate,
    device_health: VulkanDeviceHealth,
    device_fault: Option<ash::ext::device_fault::Device>,
    device_address_registry: Arc<Mutex<VulkanDeviceAddressRegistry>>,
    completion: Rc<VulkanMonotonicQueueCompletion>,
}

#[derive(Clone, Copy)]
enum VulkanTimelineValueTransform<'a> {
    UniformOffset(u64),
    PerSemaphore(&'a VulkanTimelineSemaphoreValueRebase),
}

impl VulkanTimelineValueTransform<'_> {
    fn value(
        self,
        device_handle: vk::Device,
        semaphore: vk::Semaphore,
        recorded_value: u64,
    ) -> Result<u64, VulkanError> {
        match self {
            Self::UniformOffset(offset) => offset_timeline_value(recorded_value, offset),
            Self::PerSemaphore(rebase) => {
                rebase.value(device_handle, semaphore, recorded_value)
            }
        }
    }
}

struct VulkanPreparedResidentQueueSubmission {
    command_buffer: Option<vk::CommandBuffer>,
    wait_points: Vec<(vk::Semaphore, u64)>,
    signal_points: Vec<(vk::Semaphore, u64)>,
    completion: Option<Rc<VulkanMonotonicQueueCompletion>>,
    execution_region: Option<RuntimeExecutionRegion>,
}

struct VulkanSubmittedResidentQueueBatch {
    batch_completion_value: Option<u64>,
    resource_completions: Vec<(Rc<VulkanMonotonicQueueCompletion>, u64)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentExecutionQuantumMeasurement {
    pub cost: crate::execution_schedule::RuntimeExecutionCost,
    pub region_count: usize,
    pub component_ids: Vec<String>,
    pub kernel_families: Vec<String>,
    pub duration_ns: u64,
    pub forced_yield_after: bool,
}

impl Default for VulkanResidentQueueSubmissionBatch<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> VulkanResidentQueueSubmissionBatch<'a> {
    pub fn new() -> Self {
        Self {
            groups: RefCell::new(Vec::new()),
            distributed_execution_observations: RefCell::new(Vec::new()),
            quantum_budget: None,
        }
    }

    pub fn new_bounded(quantum_budget: RuntimeExecutionQuantumBudget) -> Self {
        Self {
            groups: RefCell::new(Vec::new()),
            distributed_execution_observations: RefCell::new(Vec::new()),
            quantum_budget: Some(quantum_budget),
        }
    }

    pub(crate) fn defer_distributed_execution_observation(
        &self,
        phase: VulkanResidentDistributedExecutionPhase,
        kind: VulkanResidentDistributedExecutionKind,
        shard_count: usize,
    ) -> Result<(), VulkanError> {
        self.distributed_execution_observations
            .borrow_mut()
            .push(VulkanResidentDistributedExecutionObservation::new(
                phase,
                kind,
                shard_count,
            )?);
        Ok(())
    }

    pub fn enqueue_recorded_sequence(
        &self,
        device: &'a VulkanComputeDevice,
        sequence: &VulkanResidentKernelSequence,
        wait_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_completion: bool,
    ) -> Result<(), VulkanError> {
        self.enqueue_recorded_sequence_with_execution_region(
            device,
            sequence,
            wait_points,
            signal_points,
            signal_completion,
            None,
        )
    }

    pub fn enqueue_timeline_semaphore_bridge(
        &self,
        device: &'a VulkanComputeDevice,
        wait_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_points: &[VulkanTimelineSemaphorePoint<'_>],
    ) -> Result<(), VulkanError> {
        if wait_points.is_empty() && signal_points.is_empty() {
            return Err(VulkanError(
                "timeline semaphore bridge has no wait or signal points".to_string(),
            ));
        }
        for point in wait_points.iter().chain(signal_points) {
            device.validate_local_timeline_semaphore(point.semaphore)?;
        }
        let submission = VulkanPreparedResidentQueueSubmission {
            command_buffer: None,
            wait_points: wait_points
                .iter()
                .map(|point| (point.semaphore.semaphore, point.value))
                .collect(),
            signal_points: signal_points
                .iter()
                .map(|point| (point.semaphore.semaphore, point.value))
                .collect(),
            completion: None,
            execution_region: None,
        };
        let mut groups = self.groups.borrow_mut();
        if let Some(group) = groups
            .iter_mut()
            .find(|group| group.device.shares_logical_device_with(device))
        {
            group.submissions.push(submission);
        } else {
            groups.push(VulkanResidentQueueSubmissionGroup {
                device,
                submissions: vec![submission],
                quantum_ranges: Vec::new(),
                quanta: Vec::new(),
            });
        }
        Ok(())
    }

    pub fn enqueue_recorded_sequence_with_execution_region(
        &self,
        device: &'a VulkanComputeDevice,
        sequence: &VulkanResidentKernelSequence,
        wait_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_completion: bool,
        execution_region: Option<RuntimeExecutionRegion>,
    ) -> Result<(), VulkanError> {
        if !sequence.has_recorded_commands() {
            return Err(VulkanError(
                "resident kernel sequence has no recorded commands".to_string(),
            ));
        }
        if sequence.device.handle() != device.device.handle() {
            return Err(VulkanError(
                "resident queue submission sequence belongs to another logical device".to_string(),
            ));
        }
        for point in wait_points.iter().chain(signal_points) {
            device.validate_local_timeline_semaphore(point.semaphore)?;
        }
        self.enqueue_command_buffer(
            device,
            sequence.command_buffer,
            signal_completion.then(|| Rc::clone(&sequence.completion)),
            wait_points,
            signal_points,
            execution_region,
        )
    }

    pub fn enqueue_resident_buffer_copy(
        &self,
        device: &'a VulkanComputeDevice,
        binding: &VulkanResidentBufferCopy,
        wait_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_points: &[VulkanTimelineSemaphorePoint<'_>],
    ) -> Result<(), VulkanError> {
        if binding.device.handle() != device.device.handle() {
            return Err(VulkanError(
                "resident queue submission copy belongs to another logical device".to_string(),
            ));
        }
        for point in wait_points.iter().chain(signal_points) {
            device.validate_local_timeline_semaphore(point.semaphore)?;
        }
        self.enqueue_command_buffer(
            device,
            binding.command_buffer,
            None,
            wait_points,
            signal_points,
            None,
        )
    }

    pub fn enqueue_resident_buffer_copy_batch(
        &self,
        device: &'a VulkanComputeDevice,
        binding: &VulkanResidentBufferCopyBatch,
        wait_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_completion: bool,
    ) -> Result<(), VulkanError> {
        if binding.device.handle() != device.device.handle() {
            return Err(VulkanError(
                "resident queue submission copy batch belongs to another logical device"
                    .to_string(),
            ));
        }
        for point in wait_points.iter().chain(signal_points) {
            device.validate_local_timeline_semaphore(point.semaphore)?;
        }
        self.enqueue_command_buffer(
            device,
            binding.command_buffer,
            signal_completion.then(|| Rc::clone(&binding.completion)),
            wait_points,
            signal_points,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_command_buffer(
        &self,
        device: &'a VulkanComputeDevice,
        command_buffer: vk::CommandBuffer,
        completion: Option<Rc<VulkanMonotonicQueueCompletion>>,
        wait_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_points: &[VulkanTimelineSemaphorePoint<'_>],
        execution_region: Option<RuntimeExecutionRegion>,
    ) -> Result<(), VulkanError> {
        let submission = VulkanPreparedResidentQueueSubmission {
            command_buffer: Some(command_buffer),
            wait_points: wait_points
                .iter()
                .map(|point| (point.semaphore.semaphore, point.value))
                .collect(),
            signal_points: signal_points
                .iter()
                .map(|point| (point.semaphore.semaphore, point.value))
                .collect(),
            completion,
            execution_region,
        };
        let mut groups = self.groups.borrow_mut();
        if let Some(group) = groups
            .iter_mut()
            .find(|group| group.device.shares_logical_device_with(device))
        {
            group.submissions.push(submission);
        } else {
            groups.push(VulkanResidentQueueSubmissionGroup {
                device,
                submissions: vec![submission],
                quantum_ranges: Vec::new(),
                quanta: Vec::new(),
            });
        }
        Ok(())
    }

    pub fn pending_submission_count(&self) -> usize {
        self.groups
            .borrow()
            .iter()
            .map(|group| group.submissions.len())
            .sum()
    }

    pub fn mount(self) -> Result<VulkanResidentQueueSubmissionTemplate, VulkanError> {
        self.mount_with_calibrator(None)
    }

    pub fn mount_calibrated(
        self,
        calibrator: &RuntimeExecutionQuantumCalibrator,
        shape_class_id: &str,
    ) -> Result<VulkanResidentQueueSubmissionTemplate, VulkanError> {
        self.mount_with_calibrator(Some((calibrator, shape_class_id)))
    }

    fn mount_with_calibrator(
        self,
        calibrator: Option<(&RuntimeExecutionQuantumCalibrator, &str)>,
    ) -> Result<VulkanResidentQueueSubmissionTemplate, VulkanError> {
        let quantum_budget = self.quantum_budget;
        let distributed_execution_observations =
            self.distributed_execution_observations.into_inner();
        let mut groups = self.groups.into_inner();
        if quantum_budget.is_some() || calibrator.is_some() {
            for group in &mut groups {
                let mut regions = group
                    .submissions
                    .iter()
                    .enumerate()
                    .map(|(submission_index, submission)| {
                        submission.execution_region.clone().ok_or_else(|| {
                            VulkanError(format!(
                                "bounded resident queue submission {submission_index} has no execution-region contract"
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let quantum_budget = if let Some((calibrator, shape_class_id)) = calibrator {
                    calibrator.prepare_regions(shape_class_id, &mut regions)
                } else {
                    quantum_budget
                        .expect("bounded submission has a quantum budget")
                };
                let schedule = RuntimeExecutionSchedule::linear(&regions, quantum_budget)
                    .map_err(|error| VulkanError(error.to_string()))?;
                group.quantum_ranges = schedule
                    .quanta
                    .iter()
                    .map(|quantum| quantum.region_range.clone())
                    .collect();
                group.quanta = schedule.quanta.into_iter().map(Some).collect();
            }
        } else {
            for group in &mut groups {
                if !group.submissions.is_empty() {
                    group.quantum_ranges.push(0..group.submissions.len());
                    group.quanta.push(None);
                }
            }
        }
        let submission_count = groups.iter().try_fold(0usize, |total, group| {
            total.checked_add(group.submissions.len()).ok_or_else(|| {
                VulkanError("resident queue submission count overflowed".to_string())
            })
        })?;
        if submission_count == 0 && !distributed_execution_observations.is_empty() {
            return Err(VulkanError(
                "distributed execution observations have no queue submissions".to_string(),
            ));
        }
        let groups = groups
            .into_iter()
            .map(|group| {
                let completion = Rc::new(VulkanMonotonicQueueCompletion::new(
                    group.device.create_timeline_semaphore(0)?,
                    group.device.device_health.clone(),
                ));
                Ok(VulkanResidentQueueSubmissionTemplateGroup {
                    submitter: VulkanResidentQueueSubmitter {
                        device: group.device.device.clone(),
                        device_handle: group.device.device.handle(),
                        queue_submission: group.device.compute_queue_submission.clone(),
                        device_health: group.device.device_health.clone(),
                        device_fault: group.device.device_fault.clone(),
                        device_address_registry: Arc::clone(
                            &group.device.device_address_registry,
                        ),
                        completion,
                    },
                    submissions: group.submissions,
                    quantum_ranges: group.quantum_ranges,
                    quanta: group.quanta,
                })
            })
            .collect::<Result<Vec<_>, VulkanError>>()?;
        Ok(VulkanResidentQueueSubmissionTemplate {
            groups,
            submission_count,
            distributed_execution_observations,
        })
    }
}

impl VulkanResidentQueueSubmissionTemplate {
    fn record_distributed_execution_observations(&self) {
        for observation in &self.distributed_execution_observations {
            observation.record();
        }
    }

    pub fn submission_count(&self) -> usize {
        self.submission_count
    }

    /// Returns the number of host queue-submit calls made by one replay.
    ///
    /// A template may contain many ordered Vulkan submit records so their
    /// timeline dependencies remain explicit, but all records in one bounded
    /// execution quantum are passed to Vulkan in one `queue_submit2` call.
    /// This count therefore scales with physical devices and watchdog quanta,
    /// not with command buffers, graph nodes, or stream ticks.
    pub fn host_queue_submit_count(&self) -> usize {
        self.groups
            .iter()
            .map(|group| group.quantum_ranges.len())
            .sum()
    }

    pub fn submit_with_timeline_value_offset(
        &self,
        timeline_value_offset: u64,
    ) -> Result<usize, VulkanError> {
        for group in &self.groups {
            for submission in &group.submissions {
                for (_, value) in submission
                    .wait_points
                    .iter()
                    .chain(&submission.signal_points)
                {
                    offset_timeline_value(*value, timeline_value_offset)?;
                }
            }
        }
        for group in &self.groups {
            for quantum_range in &group.quantum_ranges {
                group.submitter.submit_prepared_resident_queue_batch(
                    &group.submissions[quantum_range.clone()],
                    VulkanTimelineValueTransform::UniformOffset(timeline_value_offset),
                    false,
                )?;
            }
        }
        self.record_distributed_execution_observations();
        Ok(self.submission_count)
    }

    pub fn submit_with_timeline_value_rebase(
        &self,
        rebase: &VulkanTimelineSemaphoreValueRebase,
    ) -> Result<usize, VulkanError> {
        for group in &self.groups {
            for submission in &group.submissions {
                for (semaphore, value) in submission
                    .wait_points
                    .iter()
                    .chain(&submission.signal_points)
                {
                    rebase.value(group.submitter.device_handle, *semaphore, *value)?;
                }
            }
        }
        for group in &self.groups {
            for quantum_range in &group.quantum_ranges {
                group.submitter.submit_prepared_resident_queue_batch(
                    &group.submissions[quantum_range.clone()],
                    VulkanTimelineValueTransform::PerSemaphore(rebase),
                    false,
                )?;
            }
        }
        self.record_distributed_execution_observations();
        Ok(self.submission_count)
    }

    pub fn submit_calibrated_quanta_and_wait(
        &self,
        timeline_value_offset: u64,
    ) -> Result<Vec<VulkanResidentExecutionQuantumMeasurement>, VulkanError> {
        if self.groups.len() != 1 {
            return Err(VulkanError(
                "calibrated execution quanta require one logical device per mounted template"
                    .to_string(),
            ));
        }
        for group in &self.groups {
            for submission in &group.submissions {
                for (_, value) in submission
                    .wait_points
                    .iter()
                    .chain(&submission.signal_points)
                {
                    offset_timeline_value(*value, timeline_value_offset)?;
                }
            }
        }
        let total_quantum_count = self
            .groups
            .iter()
            .map(|group| group.quantum_ranges.len())
            .sum::<usize>();
        let mut measurements = Vec::with_capacity(total_quantum_count);
        for group in &self.groups {
            for (quantum_index, (quantum_range, quantum)) in group
                .quantum_ranges
                .iter()
                .zip(&group.quanta)
                .enumerate()
            {
                let quantum = quantum.as_ref().ok_or_else(|| {
                    VulkanError(
                        "execution quantum measurement requires calibrated schedule metadata"
                            .to_string(),
                    )
                })?;
                let submissions = &group.submissions[quantum_range.clone()];
                let started = Instant::now();
                let submitted = group.submitter.submit_prepared_resident_queue_batch(
                    submissions,
                    VulkanTimelineValueTransform::UniformOffset(timeline_value_offset),
                    true,
                )?;
                let completion_value = submitted.batch_completion_value.ok_or_else(|| {
                    VulkanError("execution quantum did not reserve completion".to_string())
                })?;
                group
                    .submitter
                    .wait_for_batch_completion(completion_value)?;
                for (completion, value) in submitted.resource_completions {
                    completion.complete(value)?;
                }
                measurements.push(VulkanResidentExecutionQuantumMeasurement {
                    cost: quantum.cost,
                    region_count: quantum.region_count(),
                    component_ids: quantum.component_ids.clone(),
                    kernel_families: quantum.kernel_families.clone(),
                    duration_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                    forced_yield_after: quantum_index + 1 < total_quantum_count,
                });
            }
        }
        self.record_distributed_execution_observations();
        Ok(measurements)
    }
}

fn offset_timeline_value(value: u64, offset: u64) -> Result<u64, VulkanError> {
    value.checked_add(offset).ok_or_else(|| {
        VulkanError(format!(
            "timeline semaphore value {value} overflows with replay offset {offset}"
        ))
    })
}

#[derive(Clone)]
struct VulkanResidentMemoryAccess {
    queue_submission: VulkanQueueSubmissionGate,
    queue_family_index: u32,
    device_health: VulkanDeviceHealth,
    property_flags: vk::MemoryPropertyFlags,
    staging_memory_type_index: Option<u32>,
}

impl VulkanResidentMemoryAccess {
    fn is_directly_mappable(&self) -> bool {
        self.property_flags.contains(
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
    }
}

pub struct VulkanResidentKernelBufferBinding<'a> {
    pub binding: u32,
    pub buffer: &'a VulkanResidentBuffer,
    pub byte_offset: usize,
    pub byte_len: usize,
    pub access: VulkanResidentKernelBufferAccess,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VulkanResidentKernelBufferAccess {
    Read,
    Write,
    ReadWrite,
}

impl VulkanResidentKernelBufferAccess {
    fn reads(self) -> bool {
        matches!(self, Self::Read | Self::ReadWrite)
    }

    fn writes(self) -> bool {
        matches!(self, Self::Write | Self::ReadWrite)
    }

    fn conflicts_with(self, next: Self) -> bool {
        self.writes() || next.writes()
    }

    fn merge(self, other: Self) -> Self {
        match (
            self.reads() || other.reads(),
            self.writes() || other.writes(),
        ) {
            (true, true) => Self::ReadWrite,
            (true, false) => Self::Read,
            (false, true) => Self::Write,
            (false, false) => unreachable!("a resident buffer access must read or write"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct VulkanResidentKernelBufferAccessRecord {
    // A descriptor may expose a byte range, but the compiled shader contract does not yet
    // prove that every physical access stays inside that logical range. Keep synchronization
    // at the Vulkan-buffer boundary until the compiler can certify exact access footprints.
    buffer: vk::Buffer,
    access: VulkanResidentKernelBufferAccess,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct VulkanResidentKernelBufferDependency {
    buffer: vk::Buffer,
}

fn take_resident_kernel_buffer_dependencies(
    pending: &mut Vec<VulkanResidentKernelBufferAccessRecord>,
    current: &[VulkanResidentKernelBufferAccessRecord],
) -> Vec<VulkanResidentKernelBufferDependency> {
    let dependencies = current
        .iter()
        .filter(|current_access| {
            pending.iter().any(|pending_access| {
                pending_access.buffer == current_access.buffer
                    && pending_access.access.conflicts_with(current_access.access)
            })
        })
        .map(|current_access| VulkanResidentKernelBufferDependency {
            buffer: current_access.buffer,
        })
        .collect::<Vec<_>>();
    pending.retain(|pending_access| {
        !current.iter().any(|current_access| {
            pending_access.buffer == current_access.buffer
                && pending_access.access.conflicts_with(current_access.access)
        })
    });
    dependencies
}

fn merge_resident_kernel_buffer_accesses(
    pending: &mut Vec<VulkanResidentKernelBufferAccessRecord>,
    current: &[VulkanResidentKernelBufferAccessRecord],
) {
    for current_access in current {
        if let Some(pending_access) = pending
            .iter_mut()
            .find(|pending_access| pending_access.buffer == current_access.buffer)
        {
            pending_access.access = pending_access.access.merge(current_access.access);
        } else {
            pending.push(*current_access);
        }
    }
}

impl<'a> VulkanResidentKernelBufferBinding<'a> {
    pub fn new(binding: u32, buffer: &'a VulkanResidentBuffer, byte_len: usize) -> Self {
        Self {
            binding,
            buffer,
            byte_offset: 0,
            byte_len,
            access: VulkanResidentKernelBufferAccess::ReadWrite,
        }
    }

    pub fn with_byte_offset(mut self, byte_offset: usize) -> Self {
        self.byte_offset = byte_offset;
        self
    }

    pub fn with_access(mut self, access: VulkanResidentKernelBufferAccess) -> Self {
        self.access = access;
        self
    }
}
