pub struct VulkanResidentKernelDispatch {
    device: ash::Device,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    pipeline_key: VulkanGenericPipelineKey,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_count: usize,
    workgroup_count_x: u32,
    workgroup_count_y: u32,
    base_workgroup_z: u32,
    push_constant_byte_count: u32,
    buffer_accesses: Vec<VulkanResidentKernelBufferAccessRecord>,
    estimated_memory_bytes: u64,
    semantic_label: Option<String>,
}

/// Owns the Vulkan recording/submission resources for a composed sequence of
/// resident kernel dispatches. Kernel bindings remain independently reusable;
/// this object defines an execution boundary, not a model or component boundary.
pub struct VulkanResidentKernelSequence {
    device: ash::Device,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    completion_fence: vk::Fence,
    timestamp_period_ns: f32,
    timestamp_query_pool: Option<vk::QueryPool>,
    profiling_timestamp_query_pool: Option<(vk::QueryPool, u32)>,
    recorded_input_copies: RefCell<Option<Vec<VulkanResidentKernelRecordedInputCopy>>>,
    recorded_steps: RefCell<Option<Vec<VulkanResidentKernelRecordedStep>>>,
    recorded_snapshot_copies: RefCell<Option<Vec<VulkanResidentKernelRecordedSnapshotCopy>>>,
}

#[derive(Clone, PartialEq, Eq)]
struct VulkanResidentKernelRecordedInputCopy {
    source: vk::Buffer,
    destination: vk::Buffer,
    source_offset: vk::DeviceSize,
    destination_offset: vk::DeviceSize,
    byte_len: vk::DeviceSize,
}

#[derive(Clone, PartialEq, Eq)]
struct VulkanResidentKernelRecordedStep {
    pipeline: vk::Pipeline,
    descriptor_set: vk::DescriptorSet,
    workgroup_count_x: u32,
    workgroup_count_y: u32,
    base_workgroup_z: u32,
    indirect_dispatch: Option<VulkanResidentKernelRecordedIndirectDispatch>,
    condition: Option<VulkanResidentKernelRecordedCondition>,
    push_constants: Vec<u8>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct VulkanResidentKernelRecordedIndirectDispatch {
    buffer: vk::Buffer,
    offset: vk::DeviceSize,
}

#[derive(Clone, PartialEq, Eq)]
struct VulkanResidentKernelRecordedSnapshotCopy {
    after_step_index: usize,
    source: vk::Buffer,
    destination: vk::Buffer,
    source_offset: vk::DeviceSize,
    destination_offset: vk::DeviceSize,
    byte_len: vk::DeviceSize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct VulkanResidentKernelRecordedCondition {
    buffer: vk::Buffer,
    offset: vk::DeviceSize,
    inverted: bool,
    region_id: u32,
}

#[derive(Clone, Copy)]
pub struct VulkanResidentKernelSequenceStep<'a> {
    dispatch: &'a VulkanResidentKernelDispatch,
    push_constants: &'a [u8],
    direct_workgroup_count: Option<[u32; 2]>,
    indirect_dispatch: Option<VulkanResidentKernelSequenceIndirectDispatch<'a>>,
    condition: Option<VulkanResidentKernelSequenceCondition<'a>>,
}

#[derive(Clone, Copy)]
struct VulkanResidentKernelSequenceIndirectDispatch<'a> {
    buffer: &'a VulkanResidentBuffer,
    offset: vk::DeviceSize,
}

#[derive(Clone, Copy)]
struct VulkanResidentKernelSequenceCondition<'a> {
    buffer: &'a VulkanResidentBuffer,
    offset: vk::DeviceSize,
    inverted: bool,
    region_id: u32,
}

impl VulkanResidentKernelSequenceCondition<'_> {
    fn recorded(self) -> VulkanResidentKernelRecordedCondition {
        VulkanResidentKernelRecordedCondition {
            buffer: self.buffer.buffer,
            offset: self.offset,
            inverted: self.inverted,
            region_id: self.region_id,
        }
    }
}

impl<'a> VulkanResidentKernelSequenceStep<'a> {
    pub fn new(dispatch: &'a VulkanResidentKernelDispatch, push_constants: &'a [u8]) -> Self {
        Self {
            dispatch,
            push_constants,
            direct_workgroup_count: None,
            indirect_dispatch: None,
            condition: None,
        }
    }

    pub fn new_direct_with_workgroup_count(
        dispatch: &'a VulkanResidentKernelDispatch,
        push_constants: &'a [u8],
        workgroup_count_x: u32,
        workgroup_count_y: u32,
    ) -> Result<Self, VulkanError> {
        if workgroup_count_x == 0 || workgroup_count_y == 0 {
            return Err(VulkanError(
                "direct resident dispatch workgroup counts must be positive".to_string(),
            ));
        }
        Ok(Self {
            dispatch,
            push_constants,
            direct_workgroup_count: Some([workgroup_count_x, workgroup_count_y]),
            indirect_dispatch: None,
            condition: None,
        })
    }

    pub fn new_indirect(
        dispatch: &'a VulkanResidentKernelDispatch,
        push_constants: &'a [u8],
        buffer: &'a VulkanResidentBuffer,
        byte_offset: usize,
    ) -> Result<Self, VulkanError> {
        validate_resident_indirect_dispatch_range(buffer.byte_capacity(), byte_offset)?;
        if dispatch.base_workgroup_z != 0 {
            return Err(VulkanError(format!(
                "indirect resident dispatch cannot encode nonzero base workgroup z {}",
                dispatch.base_workgroup_z
            )));
        }
        Ok(Self {
            dispatch,
            push_constants,
            direct_workgroup_count: None,
            indirect_dispatch: Some(VulkanResidentKernelSequenceIndirectDispatch {
                buffer,
                offset: byte_offset as vk::DeviceSize,
            }),
            condition: None,
        })
    }

    pub fn new_conditional(
        dispatch: &'a VulkanResidentKernelDispatch,
        push_constants: &'a [u8],
        predicate: &'a VulkanResidentBuffer,
        byte_offset: usize,
        inverted: bool,
        region_id: u32,
    ) -> Result<Self, VulkanError> {
        Self::new(dispatch, push_constants).with_condition(
            predicate,
            byte_offset,
            inverted,
            region_id,
        )
    }

    pub fn with_condition(
        mut self,
        predicate: &'a VulkanResidentBuffer,
        byte_offset: usize,
        inverted: bool,
        region_id: u32,
    ) -> Result<Self, VulkanError> {
        let range_end = byte_offset
            .checked_add(std::mem::size_of::<u32>())
            .ok_or_else(|| {
            VulkanError("conditional resident dispatch range overflowed".to_string())
        })?;
        if !byte_offset.is_multiple_of(std::mem::size_of::<u32>())
            || range_end > predicate.byte_capacity()
        {
            return Err(VulkanError(format!(
                "conditional resident dispatch predicate range {byte_offset}..{range_end} is invalid for buffer capacity {}",
                predicate.byte_capacity()
            )));
        }
        self.condition = Some(VulkanResidentKernelSequenceCondition {
            buffer: predicate,
            offset: byte_offset as vk::DeviceSize,
            inverted,
            region_id,
        });
        Ok(self)
    }

    fn workgroup_count_x(self) -> u32 {
        self.direct_workgroup_count
            .map(|count| count[0])
            .unwrap_or(self.dispatch.workgroup_count_x)
    }

    fn workgroup_count_y(self) -> u32 {
        self.direct_workgroup_count
            .map(|count| count[1])
            .unwrap_or(self.dispatch.workgroup_count_y)
    }
}

pub const VULKAN_RESIDENT_INDIRECT_DISPATCH_BYTE_COUNT: usize =
    3 * std::mem::size_of::<u32>();

fn validate_resident_indirect_dispatch_range(
    buffer_byte_capacity: usize,
    byte_offset: usize,
) -> Result<(), VulkanError> {
    if !byte_offset.is_multiple_of(std::mem::size_of::<u32>()) {
        return Err(VulkanError(format!(
            "resident indirect dispatch offset {byte_offset} is not 4-byte aligned"
        )));
    }
    let range_end = byte_offset
        .checked_add(VULKAN_RESIDENT_INDIRECT_DISPATCH_BYTE_COUNT)
        .ok_or_else(|| VulkanError("resident indirect dispatch range overflowed".to_string()))?;
    if range_end > buffer_byte_capacity {
        return Err(VulkanError(format!(
            "resident indirect dispatch range {byte_offset}..{range_end} exceeds buffer capacity {buffer_byte_capacity}"
        )));
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub struct VulkanResidentKernelSequenceInputCopy<'a> {
    copy: VulkanResidentKernelSequenceInputCopySource<'a>,
}

#[derive(Clone, Copy)]
enum VulkanResidentKernelSequenceInputCopySource<'a> {
    Binding(&'a VulkanResidentBufferCopy),
    Range(VulkanResidentBufferRangeCopy<'a>),
}

impl<'a> VulkanResidentKernelSequenceInputCopy<'a> {
    pub fn new(copy: &'a VulkanResidentBufferCopy) -> Self {
        Self {
            copy: VulkanResidentKernelSequenceInputCopySource::Binding(copy),
        }
    }

    pub fn from_range(copy: VulkanResidentBufferRangeCopy<'a>) -> Self {
        Self {
            copy: VulkanResidentKernelSequenceInputCopySource::Range(copy),
        }
    }

    fn source(self) -> vk::Buffer {
        match self.copy {
            VulkanResidentKernelSequenceInputCopySource::Binding(copy) => copy.source,
            VulkanResidentKernelSequenceInputCopySource::Range(copy) => copy.source.buffer,
        }
    }

    fn destination(self) -> vk::Buffer {
        match self.copy {
            VulkanResidentKernelSequenceInputCopySource::Binding(copy) => copy.destination,
            VulkanResidentKernelSequenceInputCopySource::Range(copy) => copy.destination.buffer,
        }
    }

    fn source_offset(self) -> vk::DeviceSize {
        match self.copy {
            VulkanResidentKernelSequenceInputCopySource::Binding(_) => 0,
            VulkanResidentKernelSequenceInputCopySource::Range(copy) => copy.source_offset,
        }
    }

    fn destination_offset(self) -> vk::DeviceSize {
        match self.copy {
            VulkanResidentKernelSequenceInputCopySource::Binding(_) => 0,
            VulkanResidentKernelSequenceInputCopySource::Range(copy) => copy.destination_offset,
        }
    }

    fn byte_len(self) -> vk::DeviceSize {
        match self.copy {
            VulkanResidentKernelSequenceInputCopySource::Binding(copy) => copy.byte_len,
            VulkanResidentKernelSequenceInputCopySource::Range(copy) => copy.byte_len,
        }
    }

    fn recorded(self) -> VulkanResidentKernelRecordedInputCopy {
        VulkanResidentKernelRecordedInputCopy {
            source: self.source(),
            destination: self.destination(),
            source_offset: self.source_offset(),
            destination_offset: self.destination_offset(),
            byte_len: self.byte_len(),
        }
    }
}

#[derive(Clone, Copy)]
pub struct VulkanResidentKernelSequenceSnapshotCopy<'a> {
    pub after_step_index: usize,
    source: &'a VulkanResidentBuffer,
    destination: &'a VulkanResidentBuffer,
    source_offset: vk::DeviceSize,
    destination_offset: vk::DeviceSize,
    byte_len: vk::DeviceSize,
    allow_after_conditional_step: bool,
}

impl<'a> VulkanResidentKernelSequenceSnapshotCopy<'a> {
    pub fn new(
        after_step_index: usize,
        source: &'a VulkanResidentBuffer,
        destination: &'a VulkanResidentBuffer,
        source_offset: usize,
        destination_offset: usize,
        byte_len: usize,
    ) -> Result<Self, VulkanError> {
        if byte_len == 0 {
            return Err(VulkanError(
                "resident kernel sequence snapshot length must not be zero".to_string(),
            ));
        }
        let source_end = source_offset
            .checked_add(byte_len)
            .ok_or_else(|| VulkanError("resident snapshot source range overflowed".to_string()))?;
        let destination_end = destination_offset.checked_add(byte_len).ok_or_else(|| {
            VulkanError("resident snapshot destination range overflowed".to_string())
        })?;
        if source_end > source.byte_capacity() {
            return Err(VulkanError(format!(
                "resident snapshot source capacity {} cannot copy {} bytes at offset {}",
                source.byte_capacity(),
                byte_len,
                source_offset
            )));
        }
        if destination_end > destination.byte_capacity() {
            return Err(VulkanError(format!(
                "resident snapshot destination capacity {} cannot copy {} bytes at offset {}",
                destination.byte_capacity(),
                byte_len,
                destination_offset
            )));
        }
        Ok(Self {
            after_step_index,
            source,
            destination,
            source_offset: source_offset as vk::DeviceSize,
            destination_offset: destination_offset as vk::DeviceSize,
            byte_len: byte_len as vk::DeviceSize,
            allow_after_conditional_step: false,
        })
    }

    /// Vulkan transfer copies are not affected by conditional rendering.
    /// Callers may opt into an unconditional snapshot after a conditional
    /// dispatch only when a later checkpoint resume is guaranteed to
    /// overwrite any stale copy before it can be consumed.
    pub fn unconditional_from_range_after_conditional_step(
        after_step_index: usize,
        copy: VulkanResidentBufferRangeCopy<'a>,
    ) -> Self {
        Self {
            after_step_index,
            source: copy.source,
            destination: copy.destination,
            source_offset: copy.source_offset,
            destination_offset: copy.destination_offset,
            byte_len: copy.byte_len,
            allow_after_conditional_step: true,
        }
    }

    fn recorded(self) -> VulkanResidentKernelRecordedSnapshotCopy {
        VulkanResidentKernelRecordedSnapshotCopy {
            after_step_index: self.after_step_index,
            source: self.source.buffer,
            destination: self.destination.buffer,
            source_offset: self.source_offset,
            destination_offset: self.destination_offset,
            byte_len: self.byte_len,
        }
    }
}

fn print_resident_kernel_timestamp_summary(
    steps: &[VulkanResidentKernelSequenceStep<'_>],
    timestamps: &[u64],
    timestamp_period_ns: f32,
    host_elapsed_ns: u128,
) {
    if timestamps.len() != steps.len() + 1 {
        eprintln!(
            "nerve Vulkan timings unavailable: expected {} timestamps, received {}",
            steps.len() + 1,
            timestamps.len()
        );
        return;
    }

    let mut shape_groups = HashMap::<(u32, u32, usize, u32), (usize, f64)>::new();
    let mut semantic_groups = HashMap::<String, (usize, f64)>::new();
    let mut component_groups = HashMap::<String, (usize, f64)>::new();
    let mut op_groups = HashMap::<String, (usize, f64)>::new();
    let mut intervals = Vec::with_capacity(steps.len());
    let mut semantic_intervals = Vec::with_capacity(steps.len());
    for (step_index, step) in steps.iter().enumerate() {
        let elapsed_ticks = timestamps[step_index + 1].saturating_sub(timestamps[step_index]);
        let elapsed_ns = elapsed_ticks as f64 * f64::from(timestamp_period_ns);
        let key = (
            step.workgroup_count_x(),
            step.workgroup_count_y(),
            step.dispatch.descriptor_count,
            step.dispatch.push_constant_byte_count,
        );
        let group = shape_groups.entry(key).or_insert((0, 0.0));
        group.0 += 1;
        group.1 += elapsed_ns;
        intervals.push((step_index, key, elapsed_ns));
        if let Some(label) = step.dispatch.semantic_label() {
            let group = semantic_groups.entry(label.to_string()).or_insert((0, 0.0));
            group.0 += 1;
            group.1 += elapsed_ns;
            if let Some(component_id) = semantic_label_field(label, "component") {
                let group = component_groups.entry(component_id.to_string()).or_insert((0, 0.0));
                group.0 += 1;
                group.1 += elapsed_ns;
            }
            if let Some(op) = semantic_label_field(label, "op") {
                let group = op_groups.entry(op.to_string()).or_insert((0, 0.0));
                group.0 += 1;
                group.1 += elapsed_ns;
            }
            semantic_intervals.push((step_index, label.to_string(), elapsed_ns));
        }
    }

    let total_ns = timestamps
        .last()
        .copied()
        .unwrap_or_default()
        .saturating_sub(timestamps[0]) as f64
        * f64::from(timestamp_period_ns);
    eprintln!(
        "nerve Vulkan timings: steps={}, gpu_total_ms={:.3}, host_submit_wait_ms={:.3}, host_minus_gpu_ms={:.3}",
        steps.len(),
        total_ns / 1_000_000.0,
        host_elapsed_ns as f64 / 1_000_000.0,
        (host_elapsed_ns as f64 - total_ns).max(0.0) / 1_000_000.0,
    );

    print_resident_kernel_named_timestamp_groups("grouped component intervals", component_groups);
    print_resident_kernel_named_timestamp_groups("grouped op intervals", op_groups);

    if !semantic_groups.is_empty() {
        let mut groups = semantic_groups.into_iter().collect::<Vec<_>>();
        groups.sort_by(|left, right| {
            right
                .1
                .1
                .partial_cmp(&left.1.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        eprintln!("  grouped semantic intervals:");
        for (label, (count, elapsed_ns)) in groups {
            eprintln!(
                "    {label} count={count:<3} total_us={:.3} avg_us={:.3}",
                elapsed_ns / 1_000.0,
                elapsed_ns / count as f64 / 1_000.0,
            );
        }

        semantic_intervals.sort_by(|left, right| {
            right
                .2
                .partial_cmp(&left.2)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        eprintln!("  slowest semantic intervals:");
        for (step_index, label, elapsed_ns) in semantic_intervals.into_iter().take(12) {
            eprintln!(
                "    step={step_index:<3} {label} elapsed_us={:.3}",
                elapsed_ns / 1_000.0,
            );
        }
    }

    let mut groups = shape_groups.into_iter().collect::<Vec<_>>();
    groups.sort_by(|left, right| {
        right
            .1
            .1
            .partial_cmp(&left.1.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    eprintln!("  grouped step intervals (dispatch plus preceding dependency):");
    for ((workgroups_x, workgroups_y, descriptors, push_bytes), (count, elapsed_ns)) in groups {
        eprintln!(
            "    workgroups={workgroups_x}x{workgroups_y:<5} descriptors={descriptors:<2} push_bytes={push_bytes:<3} count={count:<3} total_us={:.3} avg_us={:.3}",
            elapsed_ns / 1_000.0,
            elapsed_ns / count as f64 / 1_000.0,
        );
    }

    intervals.sort_by(|left, right| {
        right
            .2
            .partial_cmp(&left.2)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    eprintln!("  slowest step intervals:");
    for (step_index, (workgroups_x, workgroups_y, descriptors, push_bytes), elapsed_ns) in
        intervals.into_iter().take(12)
    {
        eprintln!(
            "    step={step_index:<3} workgroups={workgroups_x}x{workgroups_y:<5} descriptors={descriptors:<2} push_bytes={push_bytes:<3} elapsed_us={:.3}",
            elapsed_ns / 1_000.0,
        );
    }
}

pub(crate) fn semantic_label_field<'a>(label: &'a str, field: &str) -> Option<&'a str> {
    let prefix = format!("{field}=");
    label
        .split_ascii_whitespace()
        .find_map(|part| part.strip_prefix(&prefix))
        .filter(|value| !value.is_empty())
}

fn print_resident_kernel_named_timestamp_groups(
    heading: &str,
    groups: HashMap<String, (usize, f64)>,
) {
    if groups.is_empty() {
        return;
    }
    let mut groups = groups.into_iter().collect::<Vec<_>>();
    groups.sort_by(|left, right| {
        right
            .1
            .1
            .partial_cmp(&left.1.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    eprintln!("  {heading}:");
    for (label, (count, elapsed_ns)) in groups {
        eprintln!(
            "    {label} count={count:<3} total_us={:.3} avg_us={:.3}",
            elapsed_ns / 1_000.0,
            elapsed_ns / count as f64 / 1_000.0,
        );
    }
}
