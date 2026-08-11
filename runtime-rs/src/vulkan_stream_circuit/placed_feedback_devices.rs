pub struct VulkanResidentInProcessPlacedStreamProcessorDevice {
    pub device_id: String,
    pub hosted_component_count: usize,
    pub incoming_edge_count: usize,
    pub outgoing_edge_count: usize,
    pub dispatch_count: usize,
    package_slice: Arc<VulkanResidentModelPackageDeviceSlice>,
    mounted: VulkanMountedPlacedStreamCircuit,
    mounted_bound: VulkanMountedPlacedBoundDispatchPlan,
    resident_execution_plan: VulkanMountedPlacedResidentStreamTickExecutionPlan,
    demand_residency_context: Option<VulkanDemandResidencyExecutionContext>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct VulkanResidentInProcessPlacedFeedbackLoopEligibility {
    device_slice_count: usize,
    every_slice_has_terminal_segment: bool,
    distributed_dispatches_are_bridged: bool,
    demand_dispatches_are_pipeline_guarded: bool,
    demand_checkpoint_resume_is_unambiguous: bool,
    every_edge_is_resident_replayable: bool,
    feedback_stream_control_is_resident_replayable: bool,
    speculative_state_is_resident_replayable: bool,
    has_dynamic_push_constants: bool,
    window_width: usize,
    sampler_history_capacity: usize,
}

impl VulkanResidentInProcessPlacedFeedbackLoopEligibility {
    fn disabled_reason(self) -> Option<&'static str> {
        if self.device_slice_count == 0 {
            Some("no_device_slices")
        } else if !self.every_slice_has_terminal_segment {
            Some("missing_terminal_segment")
        } else if !self.distributed_dispatches_are_bridged {
            Some("unbridged_distributed_dispatch")
        } else if !self.demand_dispatches_are_pipeline_guarded {
            Some("unguarded_demand_distributed_dispatch")
        } else if !self.demand_checkpoint_resume_is_unambiguous {
            Some("ambiguous_demand_checkpoint_resume")
        } else if !self.every_edge_is_resident_replayable {
            Some("host_staged_edge")
        } else if !self.feedback_stream_control_is_resident_replayable {
            Some("host_staged_feedback_stream_control")
        } else if !self.speculative_state_is_resident_replayable {
            Some("unreplayable_speculative_state_sync")
        } else {
            None
        }
    }

    fn window_width(self) -> Option<usize> {
        if self.disabled_reason().is_some() {
            return None;
        }
        let width = self.window_width.min(self.sampler_history_capacity.max(1));
        (width >= 2).then_some(width)
    }
}

struct VulkanResidentInProcessPlacedFeedbackLoop {
    feedback_synchronization: Option<Box<VulkanResidentPlacedFeedbackTimelineSynchronization>>,
    output_synchronization: Box<VulkanResidentPlacedOutputTimelineSynchronization>,
    control: VulkanResidentFeedbackControlPlane,
    window_policy: VulkanResidentFeedbackWindowPolicy,
    replayable: bool,
    scheduler_turn_count_per_tick: usize,
    completed_stage_count_per_tick: usize,
    demand_residency: Option<VulkanResidentDemandFeedbackState>,
}

struct VulkanResidentSpeculativeTargetFrameHistory {
    frames: VulkanResidentBuffer,
    frame_byte_capacity: usize,
    lane_copies: Vec<VulkanResidentBufferCopyBatch>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct VulkanResidentSpeculativeFeedbackHistoryRequirements {
    parallel_state: bool,
    normalized_frames: bool,
}

fn resident_speculative_feedback_history_requirements(
    decoder_is_parallel: impl IntoIterator<Item = bool>,
) -> VulkanResidentSpeculativeFeedbackHistoryRequirements {
    decoder_is_parallel.into_iter().fold(
        VulkanResidentSpeculativeFeedbackHistoryRequirements::default(),
        |mut requirements, is_parallel| {
            requirements.parallel_state |= is_parallel;
            requirements.normalized_frames |= !is_parallel;
            requirements
        },
    )
}

impl VulkanResidentSpeculativeTargetFrameHistory {
    fn new_if_needed(
        model: &VulkanResidentInProcessPlacedModelPackage,
        output_device: &VulkanComputeDevice,
        output_transducer: &VulkanResidentOutputTransducerRunner,
        sampler: &VulkanResidentSamplerRunner,
    ) -> Result<Option<Self>, VulkanError> {
        let requirements = resident_speculative_feedback_history_requirements(
            model.speculative_decoders.iter().map(|decoder| {
                matches!(
                    decoder.execution,
                    VulkanResidentSpeculativeDecoderModelExecution::ParallelBlock { .. }
                )
            }),
        );
        if !requirements.normalized_frames {
            return Ok(None);
        }
        let lane_capacity =
            VULKAN_BACKEND_LOOP_MAX_WINDOW.min(sampler.history_capacity_activations.max(1));
        let frame_byte_capacity = model
            .output_transducer_spec
            .normalized_frame_byte_capacity;
        let history_byte_capacity = frame_byte_capacity
            .checked_mul(lane_capacity)
            .ok_or_else(|| {
                VulkanError(
                    "speculative target-frame history capacity overflowed".to_string(),
                )
            })?;
        let frames = output_device.create_resident_buffer(history_byte_capacity)?;
        let lane_copies = (0..lane_capacity)
            .map(|lane| {
                let destination_offset =
                    lane.checked_mul(frame_byte_capacity).ok_or_else(|| {
                        VulkanError(
                            "speculative target-frame offset overflowed".to_string(),
                        )
                    })?;
                let copy = VulkanResidentBufferRangeCopy::new(
                    output_transducer.normalized_frame_buffer(),
                    &frames,
                    0,
                    destination_offset,
                    frame_byte_capacity,
                )?;
                output_device.create_resident_buffer_copy_batch(&[copy])
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Some(Self {
            frames,
            frame_byte_capacity,
            lane_copies,
        }))
    }
}

const VULKAN_RESIDENT_FEEDBACK_TARGET_CONTROL_LATENCY_NS: u64 = 250_000_000;

/// Learns how many already-recorded ticks may be submitted before returning to
/// a host control boundary. The width is an execution/responsiveness decision,
/// never a limit on how many tokens an input event may generate.
#[derive(Debug)]
struct VulkanResidentFeedbackWindowPolicy {
    maximum_tick_count: usize,
    next_tick_count: Cell<usize>,
    estimated_tick_time_ns: Cell<Option<u64>>,
}

impl VulkanResidentFeedbackWindowPolicy {
    fn new(maximum_tick_count: usize) -> Self {
        debug_assert!(maximum_tick_count >= 2);
        Self {
            maximum_tick_count,
            next_tick_count: Cell::new(2),
            estimated_tick_time_ns: Cell::new(None),
        }
    }

    fn next_tick_count(&self) -> usize {
        self.next_tick_count.get()
    }

    fn observe_completed_window(
        &self,
        planned_tick_count: usize,
        executed_tick_count: usize,
        elapsed_time_ns: u64,
        stopped: bool,
    ) {
        if stopped
            || planned_tick_count != executed_tick_count
            || executed_tick_count == 0
            || elapsed_time_ns == 0
        {
            return;
        }
        // Interrupted and predicated windows are deliberately excluded: their
        // elapsed time does not describe the cost of the submitted shape.
        let observed_tick_time_ns =
            elapsed_time_ns.div_ceil(u64::try_from(executed_tick_count).unwrap_or(u64::MAX));
        let estimated_tick_time_ns = self
            .estimated_tick_time_ns
            .get()
            .map(|previous| {
                previous
                    .saturating_mul(3)
                    .saturating_add(observed_tick_time_ns)
                    .div_ceil(4)
            })
            .unwrap_or(observed_tick_time_ns)
            .max(1);
        self.estimated_tick_time_ns
            .set(Some(estimated_tick_time_ns));
        let responsive_tick_count =
            VULKAN_RESIDENT_FEEDBACK_TARGET_CONTROL_LATENCY_NS / estimated_tick_time_ns;
        self.next_tick_count.set(
            usize::try_from(responsive_tick_count)
                .unwrap_or(usize::MAX)
                .clamp(2, self.maximum_tick_count),
        );
    }
}

struct VulkanResidentPlacedFeedbackMount<'a> {
    input_transducer: &'a VulkanResidentInputEmbeddingTransducerRunner,
    output_transducer: &'a VulkanResidentOutputTransducerRunner,
    sampler: &'a VulkanResidentSamplerRunner,
    control: VulkanResidentFeedbackControlPlane,
    demand_pipeline_predicates: Option<BTreeMap<String, Arc<VulkanResidentBuffer>>>,
    speculative_state_is_resident_replayable: bool,
}

struct VulkanResidentPlacedFeedbackTimelineSynchronization {
    output_signal: VulkanTimelineSemaphore,
    destination_waits: BTreeMap<String, VulkanTimelineSemaphore>,
    next_value: Cell<u64>,
    pending_value: Cell<Option<u64>>,
    device_local_staging: Option<VulkanResidentPlacedFeedbackDeviceLocalStaging>,
}

struct VulkanResidentPlacedFeedbackDeviceLocalStaging {
    output_copy: Box<VulkanResidentBufferCopy>,
    destination_copies: BTreeMap<String, Box<VulkanResidentBufferCopy>>,
    _output_staging: Arc<VulkanResidentBuffer>,
    _destination_staging: BTreeMap<String, Arc<VulkanResidentBuffer>>,
}

#[derive(Clone, Copy)]
struct VulkanPlacedFeedbackTimelineTurn<'a> {
    synchronization: &'a VulkanResidentPlacedFeedbackTimelineSynchronization,
    output_device_id: &'a str,
    destination_value: Option<u64>,
    output_value: u64,
}

impl<'a> VulkanPlacedFeedbackTimelineTurn<'a> {
    fn destination_wait(
        self,
        device_id: &str,
    ) -> Option<VulkanTimelineSemaphorePoint<'a>> {
        let value = self.destination_value?;
        self.synchronization
            .destination_waits
            .get(device_id)
            .map(|wait| VulkanTimelineSemaphorePoint::new(wait, value))
    }

    fn destination_copy(self, device_id: &str) -> Option<&'a VulkanResidentBufferCopy> {
        self.destination_value?;
        self.synchronization
            .device_local_staging
            .as_ref()?
            .destination_copies
            .get(device_id)
            .map(Box::as_ref)
    }

    fn output_signal(self) -> VulkanTimelineSemaphorePoint<'a> {
        VulkanTimelineSemaphorePoint::new(
            &self.synchronization.output_signal,
            self.output_value,
        )
    }

    fn output_copy(self) -> Option<&'a VulkanResidentBufferCopy> {
        self.synchronization
            .device_local_staging
            .as_ref()
            .map(|staging| staging.output_copy.as_ref())
    }
}

struct VulkanResidentPlacedOutputTimelineSynchronization {
    signal: VulkanTimelineSemaphore,
    next_value: Cell<u64>,
}

#[derive(Clone, Copy)]
struct VulkanPlacedOutputTimelineTurn<'a> {
    output_device_id: &'a str,
    signal: VulkanTimelineSemaphorePoint<'a>,
    value: u64,
}

struct VulkanResidentPlacedFeedbackSubmissionReplay {
    template: VulkanResidentQueueSubmissionTemplate,
    tick_count: usize,
    recorded_timeline_state: VulkanTimelineSemaphoreReplayState,
    transport_stats: VulkanPlacedEdgeTransportStats,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct VulkanResidentPlacedFeedbackTemplateKey {
    runtime_execution_identity: String,
    tick_count: usize,
}

type VulkanResidentPlacedFeedbackTemplateCatalog =
    BTreeMap<VulkanResidentPlacedFeedbackTemplateKey, VulkanResidentPlacedFeedbackSubmissionReplay>;

impl VulkanResidentPlacedFeedbackTimelineSynchronization {
    fn new<'a>(
        device_stream_controls: &[(&str, &'a VulkanComputeDevice, &'a VulkanResidentBuffer)],
        output_device_id: &str,
    ) -> Result<Option<Self>, VulkanError> {
        let (output_device, output_stream_control) = device_stream_controls
            .iter()
            .find(|(device_id, _, _)| *device_id == output_device_id)
            .map(|(_, device, control)| (*device, *control))
            .ok_or_else(|| {
                VulkanError(format!(
                    "resident feedback output device {output_device_id:?} has no stream control"
                ))
            })?;
        let destinations = device_stream_controls
            .iter()
            .filter(|(_, device, _)| !device.shares_logical_device_with(output_device))
            .copied()
            .collect::<Vec<_>>();
        if destinations.is_empty() {
            return Ok(None);
        }
        if !output_device.supports_opaque_fd_timeline_semaphores()
            || destinations
                .iter()
                .any(|(_, device, _)| !device.supports_opaque_fd_timeline_semaphores())
        {
            return Err(VulkanError(
                "cross-device resident feedback requires persistent opaque-file timeline semaphores"
                    .to_string(),
            ));
        }
        let output_signal = output_device.create_opaque_fd_exportable_timeline_semaphore(0)?;
        let mut destination_waits = BTreeMap::new();
        for (device_id, device, _) in &destinations {
            let wait = device.create_timeline_semaphore(0)?;
            device.import_timeline_semaphore_opaque_fd(
                &wait,
                output_device.export_timeline_semaphore_opaque_fd(&output_signal)?,
            )?;
            destination_waits.insert((*device_id).to_string(), wait);
        }
        let controls_are_directly_shared = destinations.iter().all(|(_, _, control)| {
            control.shares_device_memory_with(output_stream_control)
                || control.shares_host_allocation_with(output_stream_control)
        });
        let device_local_staging = if controls_are_directly_shared {
            None
        } else {
            let peers = destinations
                .iter()
                .map(|(_, device, _)| *device)
                .collect::<Vec<_>>();
            let allocation = output_device
                .create_shared_host_allocation(&peers, VULKAN_STREAM_CONTROL_BYTE_CAPACITY)?;
            let output_staging = Arc::new(
                output_device.import_shared_host_buffer(Arc::clone(&allocation))?,
            );
            let output_copy = Box::new(output_device.create_resident_buffer_copy(
                output_stream_control,
                &output_staging,
                VULKAN_STREAM_CONTROL_BYTE_CAPACITY,
            )?);
            let mut destination_copies = BTreeMap::new();
            let mut destination_staging = BTreeMap::new();
            for (device_id, device, stream_control) in &destinations {
                let staging = Arc::new(
                    device.import_shared_host_buffer(Arc::clone(&allocation))?,
                );
                let copy = Box::new(device.create_resident_buffer_copy(
                    &staging,
                    stream_control,
                    VULKAN_STREAM_CONTROL_BYTE_CAPACITY,
                )?);
                destination_copies.insert((*device_id).to_string(), copy);
                destination_staging.insert((*device_id).to_string(), staging);
            }
            Some(VulkanResidentPlacedFeedbackDeviceLocalStaging {
                output_copy,
                destination_copies,
                _output_staging: output_staging,
                _destination_staging: destination_staging,
            })
        };
        Ok(Some(Self {
            output_signal,
            destination_waits,
            next_value: Cell::new(1),
            pending_value: Cell::new(None),
            device_local_staging,
        }))
    }

    fn prepare_turn<'a>(
        &'a self,
        output_device_id: &'a str,
    ) -> Result<VulkanPlacedFeedbackTimelineTurn<'a>, VulkanError> {
        let value = self.next_value.get();
        self.next_value.set(value.checked_add(1).ok_or_else(|| {
            VulkanError("resident feedback timeline semaphore exhausted its values".to_string())
        })?);
        let destination_value = self.pending_value.replace(Some(value));
        Ok(VulkanPlacedFeedbackTimelineTurn {
            synchronization: self,
            output_device_id,
            destination_value,
            output_value: value,
        })
    }

    fn prepare_resumed_turn<'a>(
        &'a self,
        output_device_id: &'a str,
    ) -> Result<VulkanPlacedFeedbackTimelineTurn<'a>, VulkanError> {
        let value = self.next_value.get();
        self.next_value.set(value.checked_add(1).ok_or_else(|| {
            VulkanError("resident feedback timeline semaphore exhausted its values".to_string())
        })?);
        // The failed attempt already consumed this lane's input dependency.
        // Its own output and every later logical turn were suppressed by the
        // shared demand predicate, so replace that abandoned carry with the
        // continuation's new output without waiting on it as an input.
        self.pending_value.set(Some(value));
        Ok(VulkanPlacedFeedbackTimelineTurn {
            synchronization: self,
            output_device_id,
            destination_value: None,
            output_value: value,
        })
    }

    fn advance_replayed_turns(&self, count: usize) -> Result<(), VulkanError> {
        let count = u64::try_from(count)
            .map_err(|_| VulkanError("resident feedback replay width exceeds u64".to_string()))?;
        if count == 0 {
            return Err(VulkanError(
                "resident feedback replay width must not be zero".to_string(),
            ));
        }
        let first_value = self.next_value.get();
        let expected_pending = first_value.checked_sub(1).ok_or_else(|| {
            VulkanError("resident feedback replay has no preceding timeline value".to_string())
        })?;
        if self.pending_value.get() != Some(expected_pending) {
            return Err(VulkanError(format!(
                "resident feedback replay expected pending timeline value {expected_pending}, found {:?}",
                self.pending_value.get()
            )));
        }
        let next_value = first_value.checked_add(count).ok_or_else(|| {
            VulkanError("resident feedback replay exhausted timeline values".to_string())
        })?;
        self.next_value.set(next_value);
        self.pending_value.set(Some(next_value - 1));
        Ok(())
    }

    fn discard_aborted_turns(&self) {
        // Timeline semaphore values are monotonic and remain valid after an
        // aborted demand attempt. Only the logical carry into the next window
        // must be discarded: its payload belongs to rolled-back execution.
        self.pending_value.set(None);
    }

    fn capture_replay_timeline_state(
        &self,
        state: &mut VulkanTimelineSemaphoreReplayState,
    ) -> Result<(), VulkanError> {
        let next_value = self.next_value.get();
        state.capture(&self.output_signal, next_value)?;
        for wait in self.destination_waits.values() {
            state.capture(wait, next_value)?;
        }
        Ok(())
    }
}

impl VulkanResidentPlacedOutputTimelineSynchronization {
    fn new(output_device: &VulkanComputeDevice) -> Result<Self, VulkanError> {
        Ok(Self {
            signal: output_device.create_timeline_semaphore(0)?,
            next_value: Cell::new(1),
        })
    }

    fn prepare_turn<'a>(
        &'a self,
        output_device_id: &'a str,
    ) -> Result<VulkanPlacedOutputTimelineTurn<'a>, VulkanError> {
        let value = self.next_value.get();
        self.next_value.set(value.checked_add(1).ok_or_else(|| {
            VulkanError("resident output timeline semaphore exhausted its values".to_string())
        })?);
        Ok(VulkanPlacedOutputTimelineTurn {
            output_device_id,
            signal: VulkanTimelineSemaphorePoint::new(&self.signal, value),
            value,
        })
    }

    fn wait_for_turn(
        &self,
        output_device: &VulkanComputeDevice,
        value: u64,
    ) -> Result<(), VulkanError> {
        output_device.wait_timeline_semaphore_value(&self.signal, value)
    }

    fn wait_for_turn_for(
        &self,
        output_device: &VulkanComputeDevice,
        value: u64,
        timeout_ns: u64,
    ) -> Result<bool, VulkanError> {
        output_device.wait_timeline_semaphore_value_for(&self.signal, value, timeout_ns)
    }

    fn turn_is_complete(
        &self,
        output_device: &VulkanComputeDevice,
        value: u64,
    ) -> Result<bool, VulkanError> {
        Ok(output_device.timeline_semaphore_value(&self.signal)? >= value)
    }

    fn turn_point(&self, value: u64) -> VulkanTimelineSemaphorePoint<'_> {
        VulkanTimelineSemaphorePoint::new(&self.signal, value)
    }

    fn reserve_replayed_turns(&self, count: usize) -> Result<Vec<u64>, VulkanError> {
        let count = u64::try_from(count)
            .map_err(|_| VulkanError("resident output replay width exceeds u64".to_string()))?;
        if count == 0 {
            return Err(VulkanError(
                "resident output replay width must not be zero".to_string(),
            ));
        }
        let first_value = self.next_value.get();
        let next_value = first_value.checked_add(count).ok_or_else(|| {
            VulkanError("resident output replay exhausted timeline values".to_string())
        })?;
        self.next_value.set(next_value);
        Ok((first_value..next_value).collect())
    }

    fn capture_replay_timeline_state(
        &self,
        state: &mut VulkanTimelineSemaphoreReplayState,
    ) -> Result<(), VulkanError> {
        state.capture(&self.signal, self.next_value.get())
    }
}

impl VulkanResidentPlacedFeedbackSubmissionReplay {
    fn new(
        template: VulkanResidentQueueSubmissionTemplate,
        tick_count: usize,
        recorded_timeline_state: VulkanTimelineSemaphoreReplayState,
        transport_stats: VulkanPlacedEdgeTransportStats,
    ) -> Result<Self, VulkanError> {
        if tick_count == 0 {
            return Err(VulkanError(
                "resident feedback replay width must not be zero".to_string(),
            ));
        }
        Ok(Self {
            template,
            tick_count,
            recorded_timeline_state,
            transport_stats,
        })
    }

    fn validate_tick_count(&self, tick_count: usize) -> Result<(), VulkanError> {
        if tick_count != self.tick_count {
            return Err(VulkanError(format!(
                "resident feedback replay was mounted for {} ticks, received {tick_count}",
                self.tick_count
            )));
        }
        Ok(())
    }

    fn submit_next(
        &self,
        tick_count: usize,
        current_timeline_state: &VulkanTimelineSemaphoreReplayState,
    ) -> Result<usize, VulkanError> {
        self.validate_tick_count(tick_count)?;
        let rebase = self
            .recorded_timeline_state
            .rebase_to(current_timeline_state)?;
        self.template.submit_with_timeline_value_rebase(&rebase)
    }
}

fn apply_placed_clone_state_policies(
    devices: &[VulkanResidentInProcessPlacedStreamProcessorDevice],
    initialized: &BTreeSet<(String, String)>,
) -> Result<usize, VulkanError> {
    let mut state_index = BTreeMap::<(String, String), (usize, usize)>::new();
    let mut states = Vec::new();
    for (device_index, device) in devices.iter().enumerate() {
        for (state_index_on_device, state) in
            device.mounted.buffers.state_buffers.iter().enumerate()
        {
            let key = (state.component_id.clone(), state.state_id.clone());
            if state_index
                .insert(key.clone(), (device_index, state_index_on_device))
                .is_some()
            {
                return Err(VulkanError(format!(
                    "duplicate placed state buffer {}.{}",
                    key.0, key.1
                )));
            }
            states.push((key, state.clone_from.clone()));
        }
    }
    let copies = ordered_clone_state_copies(states, initialized)?;
    let mut total_copied = 0usize;
    for (target_id, source_id) in copies {
        let (target_device_index, target_state_index) = state_index
            .get(&target_id)
            .copied()
            .expect("clone target was indexed from resident states");
        let (source_device_index, source_state_index) = state_index
            .get(&source_id)
            .copied()
            .expect("planned clone source must exist");
        let target =
            &devices[target_device_index].mounted.buffers.state_buffers[target_state_index];
        let source =
            &devices[source_device_index].mounted.buffers.state_buffers[source_state_index];
        validate_state_buffer_copy(target, source)?;
        let bytes = source.buffer.read_bytes(source.byte_capacity)?;
        target.buffer.write_bytes(&bytes)?;
        total_copied = total_copied
            .checked_add(bytes.len())
            .ok_or_else(|| VulkanError("placed clone state byte count overflowed".to_string()))?;
    }
    Ok(total_copied)
}

fn inherit_matching_placed_stream_state(
    target_devices: &[VulkanResidentInProcessPlacedStreamProcessorDevice],
    source_devices: &[VulkanResidentInProcessPlacedStreamProcessorDevice],
) -> Result<(usize, BTreeSet<(String, String)>), VulkanError> {
    let source_by_id = source_devices
        .iter()
        .flat_map(|device| device.mounted.buffers.state_buffers.iter())
        .map(|state| ((state.component_id.as_str(), state.state_id.as_str()), state))
        .collect::<BTreeMap<_, _>>();
    let mut copied = BTreeSet::new();
    let mut total_copied = 0usize;
    for target in target_devices
        .iter()
        .flat_map(|device| device.mounted.buffers.state_buffers.iter())
    {
        let key = (target.component_id.as_str(), target.state_id.as_str());
        let Some(source) = source_by_id.get(&key) else {
            continue;
        };
        validate_state_buffer_copy(target, source)?;
        let bytes = source.buffer.read_bytes(source.byte_capacity)?;
        target.buffer.write_bytes(&bytes)?;
        total_copied = total_copied.checked_add(bytes.len()).ok_or_else(|| {
            VulkanError("inherited placed state byte count overflowed".to_string())
        })?;
        copied.insert((target.component_id.clone(), target.state_id.clone()));
    }
    Ok((total_copied, copied))
}

impl VulkanResidentInProcessPlacedFeedbackLoop {
    fn new_if_supported<'a, F, E>(
        model: &VulkanResidentInProcessPlacedModelPackage,
        device_slices: &[VulkanResidentInProcessPlacedStreamProcessorDevice],
        activation_schedule: &VulkanMountedPlacedResidentInProcessSchedule,
        every_edge_is_resident_replayable: bool,
        feedback_stream_control_is_resident_replayable: bool,
        mount: VulkanResidentPlacedFeedbackMount<'_>,
        device_for: &F,
    ) -> Result<Option<Self>, VulkanError>
    where
        F: Fn(&str) -> Result<&'a VulkanComputeDevice, E>,
        E: Display,
    {
        let VulkanResidentPlacedFeedbackMount {
            input_transducer,
            output_transducer,
            sampler,
            control,
            demand_pipeline_predicates,
            speculative_state_is_resident_replayable,
        } = mount;
        let has_dynamic_push_constants = input_transducer
            .resident_dispatch
            .push_constant_byte_count()
            != 0
            || output_transducer
                .embedding_norm_dispatch
                .push_constant_byte_count()
                != 0
            || output_transducer
                .tied_projection_dispatch
                .push_constant_byte_count()
                != 0
            || sampler
                .resident_dispatches()
                .iter()
                .any(|dispatch| dispatch.push_constant_byte_count() != 0)
            || device_slices
                .iter()
                .flat_map(|slice| &slice.resident_execution_plan.dispatch_segments)
                .flat_map(|segment| &segment.dispatches)
                .any(|dispatch| {
                    dispatch.resident_dispatch.push_constant_byte_count() != 0
                        && dispatch.push_constants.as_slice()
                            != [VulkanKernelScalarBinding {
                                name: "expert_start".to_string(),
                                scalar_type: "u32".to_string(),
                                source: VulkanKernelScalarSource::PushConstant,
                            }]
                });
        let has_demand_checkpoints = model.resource_residency_policy.is_demand_loaded()
            && (device_slices.iter().any(|slice| {
                !slice
                    .package_slice
                    .physical_residency_schedule()
                    .checkpoints
                    .is_empty()
            }) || model
                .distributed_execution_plans
                .decode
                .dispatches
                .iter()
                .any(|dispatch| !dispatch.selected_resource_partitions.is_empty()));
        let window_width = VULKAN_BACKEND_LOOP_MAX_WINDOW
            .min(sampler.history_capacity_activations.max(1));
        let demand_checkpoint_resume_is_unambiguous = if has_demand_checkpoints {
            let tick_plans = device_slices
                .iter()
                .map(|slice| slice.resident_execution_plan.tick_plan.as_ref())
                .collect::<Vec<_>>();
            VulkanDemandFeedbackStageTopology::from_tick_plans(&tick_plans)?.is_total_ordered()
        } else {
            true
        };
        let demand_pipeline_is_guarded = !has_demand_checkpoints
            || demand_pipeline_predicates.is_some();
        let eligibility = VulkanResidentInProcessPlacedFeedbackLoopEligibility {
            device_slice_count: device_slices.len(),
            every_slice_has_terminal_segment: device_slices
                .iter()
                .all(|slice| !slice.resident_execution_plan.dispatch_segments.is_empty()),
            distributed_dispatches_are_bridged: device_slices.iter().all(|slice| {
                slice
                    .resident_execution_plan
                    .distributed_dispatch_dependencies
                    .values()
                    .all(|dependency| {
                        dependency.has_owner_producer && dependency.has_owner_continuation
                    })
            }),
            demand_dispatches_are_pipeline_guarded: demand_pipeline_is_guarded,
            demand_checkpoint_resume_is_unambiguous,
            every_edge_is_resident_replayable,
            feedback_stream_control_is_resident_replayable,
            speculative_state_is_resident_replayable,
            has_dynamic_push_constants,
            window_width,
            sampler_history_capacity: sampler.history_capacity_activations,
        };
        let Some(window_width) = eligibility.window_width() else {
            return Ok(None);
        };
        let output_device = device_for(&model.output_device_id).map_err(|error| {
            VulkanError(format!("feedback output device resolution failed: {error}"))
        })?;
        let device_stream_controls = device_slices
            .iter()
            .map(|slice| {
                device_for(&slice.device_id)
                    .map(|device| {
                        (
                            slice.device_id.as_str(),
                            device,
                            slice.mounted.stream_control_buffer.as_ref(),
                        )
                    })
                    .map_err(|error| {
                        VulkanError(format!(
                            "feedback device {:?} resolution failed: {error}",
                            slice.device_id
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let feedback_synchronization = VulkanResidentPlacedFeedbackTimelineSynchronization::new(
            &device_stream_controls,
            &model.output_device_id,
        )?
        .map(Box::new);
        let output_synchronization = Box::new(
            VulkanResidentPlacedOutputTimelineSynchronization::new(output_device)?,
        );
        let completed_stage_count_per_tick =
            device_slices.iter().try_fold(0usize, |total, slice| {
                total
                    .checked_add(slice.resident_execution_plan.tick_plan.stage_count)
                    .ok_or_else(|| {
                        VulkanError("placed feedback stage count overflowed".to_string())
                    })
            })?;
        let demand_residency = match (
            has_demand_checkpoints,
            demand_pipeline_predicates,
        ) {
            (true, Some(predicates)) => Some(VulkanResidentDemandFeedbackState::new(
                predicates,
                model,
                device_slices,
                &model.output_device_id,
            )?),
            (true, None) => {
                return Err(VulkanError(
                    "demand-loaded resident feedback has no shared pipeline predicate"
                        .to_string(),
                ));
            }
            (false, Some(_)) => {
                return Err(VulkanError(
                    "resident feedback without physical demand checkpoints unexpectedly received demand predicates"
                        .to_string(),
                ));
            }
            (false, None) => None,
        };
        Ok(Some(Self {
            feedback_synchronization,
            output_synchronization,
            control,
            window_policy: VulkanResidentFeedbackWindowPolicy::new(window_width),
            replayable: demand_residency.is_none() && !has_dynamic_push_constants,
            scheduler_turn_count_per_tick: activation_schedule.turns.len(),
            completed_stage_count_per_tick,
            demand_residency,
        }))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VulkanResidentPlacedTokenTickTail {
    None,
    Hidden,
    Logits,
    Sample,
}

impl VulkanResidentPlacedTokenTickTail {
    fn sequence_variant(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Hidden => 1,
            Self::Logits => 2,
            Self::Sample => 3,
        }
    }

    fn produces_logits(self) -> bool {
        matches!(self, Self::Logits | Self::Sample)
    }
}

fn placed_token_input(
    token_id: u32,
    input_device_id: &str,
    output_device_id: &str,
    input_is_feedback: bool,
) -> VulkanResidentPlacedTokenInput {
    if !input_is_feedback {
        VulkanResidentPlacedTokenInput::HostSupplied(token_id)
    } else if input_device_id == output_device_id {
        VulkanResidentPlacedTokenInput::ResidentFeedback(token_id)
    } else {
        VulkanResidentPlacedTokenInput::EdgeFeedback(token_id)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VulkanResidentPlacedTokenInput {
    HostSupplied(u32),
    ResidentFeedback(u32),
    EdgeFeedback(u32),
}

impl VulkanResidentPlacedTokenInput {
    fn token_id(self) -> u32 {
        match self {
            Self::HostSupplied(token_id)
            | Self::ResidentFeedback(token_id)
            | Self::EdgeFeedback(token_id) => token_id,
        }
    }
}

fn pair_placed_edge_endpoints(
    plans: &[VulkanPlacedEdgeIoPlan],
) -> Result<Vec<(VulkanPlacedEdgeEndpoint, VulkanPlacedEdgeEndpoint)>, VulkanError> {
    let mut incoming_by_key = BTreeMap::new();
    for plan in plans {
        for endpoint in plan
            .endpoints
            .iter()
            .filter(|endpoint| endpoint.direction == VulkanPlacedEdgeDirection::Incoming)
        {
            let key = VulkanPlacedEdgePacketKey::from_incoming_endpoint(endpoint);
            if incoming_by_key
                .insert(key.clone(), endpoint.clone())
                .is_some()
            {
                return Err(VulkanError(format!(
                    "placed execution_graph repeats incoming edge endpoint {key:?}"
                )));
            }
        }
    }

    let mut pairs = Vec::with_capacity(incoming_by_key.len());
    let mut outgoing_keys = BTreeSet::new();
    for plan in plans {
        for outgoing in plan
            .endpoints
            .iter()
            .filter(|endpoint| endpoint.direction == VulkanPlacedEdgeDirection::Outgoing)
        {
            let key = VulkanPlacedEdgePacketKey::from_outgoing_endpoint(outgoing);
            if !outgoing_keys.insert(key.clone()) {
                return Err(VulkanError(format!(
                    "placed execution_graph repeats outgoing edge endpoint {key:?}"
                )));
            }
            let incoming = incoming_by_key.remove(&key).ok_or_else(|| {
                VulkanError(format!(
                    "placed execution_graph has no incoming endpoint for edge {key:?}"
                ))
            })?;
            let outgoing_byte_capacity = outgoing.byte_capacity.ok_or_else(|| {
                VulkanError(format!("outgoing edge {key:?} has unknown byte capacity"))
            })?;
            let incoming_byte_capacity = incoming.byte_capacity.ok_or_else(|| {
                VulkanError(format!("incoming edge {key:?} has unknown byte capacity"))
            })?;
            if outgoing_byte_capacity != incoming_byte_capacity {
                return Err(VulkanError(format!(
                    "placed edge {key:?} has outgoing capacity {outgoing_byte_capacity} and incoming capacity {incoming_byte_capacity}"
                )));
            }
            pairs.push((outgoing.clone(), incoming));
        }
    }
    if let Some(key) = incoming_by_key.keys().next() {
        return Err(VulkanError(format!(
            "placed execution_graph has no outgoing endpoint for edge {key:?}"
        )));
    }
    Ok(pairs)
}

#[derive(Clone, Debug)]
struct VulkanPlacedProducedPortEdgeGroup {
    source_device_id: String,
    source_component_id: String,
    source_port_id: String,
    byte_capacity: usize,
    edges: Vec<(VulkanPlacedEdgeEndpoint, VulkanPlacedEdgeEndpoint)>,
}

fn group_placed_edge_pairs_by_produced_port(
    edge_pairs: Vec<(VulkanPlacedEdgeEndpoint, VulkanPlacedEdgeEndpoint)>,
) -> Result<Vec<VulkanPlacedProducedPortEdgeGroup>, VulkanError> {
    let mut groups = BTreeMap::<
        (String, String, String),
        VulkanPlacedProducedPortEdgeGroup,
    >::new();
    for (outgoing, incoming) in edge_pairs {
        let byte_capacity = outgoing
            .byte_capacity
            .expect("paired outgoing edge capacity was validated");
        let key = (
            outgoing.local_device_id.clone(),
            outgoing.local_component_id.clone(),
            outgoing.local_port_id.clone(),
        );
        let group = groups
            .entry(key)
            .or_insert_with(|| VulkanPlacedProducedPortEdgeGroup {
                source_device_id: outgoing.local_device_id.clone(),
                source_component_id: outgoing.local_component_id.clone(),
                source_port_id: outgoing.local_port_id.clone(),
                byte_capacity,
                edges: Vec::new(),
            });
        if group.byte_capacity != byte_capacity {
            return Err(VulkanError(format!(
                "produced port {}.{} on {:?} has incompatible outgoing capacities {} and {byte_capacity}",
                group.source_component_id,
                group.source_port_id,
                group.source_device_id,
                group.byte_capacity,
            )));
        }
        group.edges.push((outgoing, incoming));
    }
    Ok(groups.into_values().collect())
}

struct VulkanPlacedDeviceLinks {
    local_edge_overrides: BTreeMap<String, Vec<VulkanPlacedLocalEdgeBufferOverride>>,
    endpoint_overrides: BTreeMap<String, Vec<VulkanPlacedEdgeEndpointBufferOverride>>,
    synchronizations: VulkanPlacedEdgeTimelineSynchronizations,
    stream_control_buffers: BTreeMap<String, Arc<VulkanResidentBuffer>>,
    every_edge_is_resident_replayable: bool,
    feedback_stream_control_is_resident_replayable: bool,
}

#[derive(Default)]
struct VulkanPlacedEdgeTimelineSynchronizations {
    edges: BTreeMap<VulkanPlacedEdgePacketKey, VulkanPlacedEdgeTimelineSynchronization>,
    sample_cursor: Cell<usize>,
}

struct VulkanPlacedEdgeTimelineSynchronization {
    source_signal: VulkanTimelineSemaphore,
    destination_wait: VulkanTimelineSemaphore,
    next_value: Cell<u64>,
    pending_value: Cell<Option<u64>>,
    transfer_route: VulkanPlacedEdgeTransferRoute,
    device_local_staging: Option<VulkanPlacedEdgeDeviceLocalStaging>,
}

struct VulkanPlacedEdgeDeviceLocalStaging {
    source_copy: Box<VulkanResidentBufferCopy>,
    destination_copy: Box<VulkanResidentBufferCopy>,
    sample_source_copy: Box<VulkanResidentBufferCopy>,
    sample_destination_copy: Box<VulkanResidentBufferCopy>,
    last_sampled_transfer_duration_ns: Cell<Option<u64>>,
    _source_staging: Arc<VulkanResidentBuffer>,
    _destination_staging: Arc<VulkanResidentBuffer>,
}

impl VulkanPlacedEdgeTimelineSynchronizations {
    fn sample_completed_device_local_staging_transfers(
        &self,
        stats: &mut VulkanPlacedEdgeTransportStats,
    ) -> Result<(), VulkanError> {
        let candidates = stats
            .edges
            .iter()
            .filter(|edge| {
                edge.route == VulkanPlacedEdgeTransferRoute::DeviceLocalStaging
                    && edge.publish_count != 0
            })
            .map(|edge| edge.key.clone())
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return Ok(());
        }
        let sample_index = self.sample_cursor.get() % candidates.len();
        self.sample_cursor
            .set(self.sample_cursor.get().wrapping_add(1));
        let sampled_key = &candidates[sample_index];
        let sampled_synchronization = self.edges.get(sampled_key).ok_or_else(|| {
            VulkanError(format!(
                "completed staged edge {sampled_key:?} has no mounted synchronization"
            ))
        })?;
        let sampled_staging = sampled_synchronization
            .device_local_staging
            .as_ref()
            .ok_or_else(|| {
                VulkanError(format!(
                    "completed staged edge {sampled_key:?} has no device-local copies"
                ))
            })?;
        let (source_duration_ns, destination_duration_ns) = {
            // One bounded diagnostic pair runs after the production event has
            // completed. Production transfers retain their original command
            // buffers and never wait for or read timestamp queries.
            let _transfer =
                runtime_critical_path_span(RuntimeCriticalPathPhase::CrossDeviceTransfer);
            let source_duration_ns = sampled_staging
                .sample_source_copy
                .run_with_device_duration(sampled_staging.sample_source_copy.byte_len())?;
            let destination_duration_ns = sampled_staging
                .sample_destination_copy
                .run_with_device_duration(sampled_staging.sample_destination_copy.byte_len())?;
            (source_duration_ns, destination_duration_ns)
        };
        let sampled_transfer_duration_ns =
            source_duration_ns.saturating_add(destination_duration_ns);
        sampled_staging
            .last_sampled_transfer_duration_ns
            .set(Some(sampled_transfer_duration_ns));
        record_runtime_critical_path_device_duration(
            RuntimeCriticalPathPhase::CrossDeviceTransfer,
            source_duration_ns,
        );
        record_runtime_critical_path_device_duration(
            RuntimeCriticalPathPhase::CrossDeviceTransfer,
            destination_duration_ns,
        );

        for edge in &mut stats.edges {
            if edge.route != VulkanPlacedEdgeTransferRoute::DeviceLocalStaging
                || edge.publish_count == 0
            {
                continue;
            }
            let synchronization = self.edges.get(&edge.key).ok_or_else(|| {
                VulkanError(format!(
                    "completed staged edge {:?} has no mounted synchronization",
                    edge.key
                ))
            })?;
            let staging = synchronization.device_local_staging.as_ref().ok_or_else(|| {
                VulkanError(format!(
                    "completed staged edge {:?} has no device-local copies",
                    edge.key
                ))
            })?;
            let Some(estimate_per_transfer_ns) =
                staging.last_sampled_transfer_duration_ns.get()
            else {
                continue;
            };
            if edge.key == *sampled_key {
                edge.device_duration_sample_count =
                    edge.device_duration_sample_count.saturating_add(2);
                edge.sampled_device_duration_ns = edge
                    .sampled_device_duration_ns
                    .saturating_add(sampled_transfer_duration_ns);
                edge.maximum_sampled_transfer_duration_ns = edge
                    .maximum_sampled_transfer_duration_ns
                    .max(sampled_transfer_duration_ns);
            }
            edge.estimated_device_duration_ns = edge
                .estimated_device_duration_ns
                .saturating_add(
                    estimate_per_transfer_ns.saturating_mul(
                        u64::try_from(edge.publish_count).unwrap_or(u64::MAX),
                    ),
                );
        }
        Ok(())
    }

    fn transfer_route(
        &self,
        key: &VulkanPlacedEdgePacketKey,
    ) -> Option<VulkanPlacedEdgeTransferRoute> {
        self.edges
            .get(key)
            .map(|synchronization| synchronization.transfer_route)
    }

    fn edge_uses_device_local_staging(&self, key: &VulkanPlacedEdgePacketKey) -> bool {
        self.edges.get(key).is_some_and(|synchronization| {
            synchronization.transfer_route == VulkanPlacedEdgeTransferRoute::DeviceLocalStaging
        })
    }

    fn advance_replayed_dependencies(&self, count: usize) -> Result<(), VulkanError> {
        let count = u64::try_from(count)
            .map_err(|_| VulkanError("placed edge replay width exceeds u64".to_string()))?;
        for (key, synchronization) in &self.edges {
            if synchronization.pending_value.get().is_some() {
                return Err(VulkanError(format!(
                    "cross-device edge {key:?} cannot replay with an unconsumed timeline dependency"
                )));
            }
            synchronization
                .next_value
                .get()
                .checked_add(count)
                .ok_or_else(|| {
                    VulkanError(format!(
                        "cross-device edge {key:?} exhausts its timeline values during replay"
                    ))
                })?;
        }
        for synchronization in self.edges.values() {
            synchronization.next_value.set(
                synchronization
                    .next_value
                    .get()
                    .checked_add(count)
                    .expect("placed edge replay advance was validated"),
            );
        }
        Ok(())
    }

    fn capture_replay_timeline_state(
        &self,
        state: &mut VulkanTimelineSemaphoreReplayState,
    ) -> Result<(), VulkanError> {
        for synchronization in self.edges.values() {
            let next_value = synchronization.next_value.get();
            state.capture(&synchronization.source_signal, next_value)?;
            state.capture(&synchronization.destination_wait, next_value)?;
        }
        Ok(())
    }

    fn prepare_source_signal<'a>(
        &'a self,
        endpoint: &VulkanPlacedEdgeEndpoint,
    ) -> Result<Option<VulkanTimelineSemaphorePoint<'a>>, VulkanError> {
        let key = VulkanPlacedEdgePacketKey::from_outgoing_endpoint(endpoint);
        let Some(synchronization) = self.edges.get(&key) else {
            return Ok(None);
        };
        if synchronization.pending_value.get().is_some() {
            return Err(VulkanError(format!(
                "cross-device edge {key:?} already has an unconsumed timeline dependency"
            )));
        }
        let value = synchronization.next_value.get();
        let next = value.checked_add(1).ok_or_else(|| {
            VulkanError(format!(
                "cross-device edge {key:?} exhausted its timeline semaphore values"
            ))
        })?;
        synchronization.next_value.set(next);
        synchronization.pending_value.set(Some(value));
        Ok(Some(VulkanTimelineSemaphorePoint::new(
            &synchronization.source_signal,
            value,
        )))
    }

    fn take_destination_wait<'a>(
        &'a self,
        endpoint: &VulkanPlacedEdgeEndpoint,
    ) -> Result<Option<VulkanTimelineSemaphorePoint<'a>>, VulkanError> {
        let key = VulkanPlacedEdgePacketKey::from_incoming_endpoint(endpoint);
        let Some(synchronization) = self.edges.get(&key) else {
            return Ok(None);
        };
        let value = synchronization.pending_value.take().ok_or_else(|| {
            VulkanError(format!(
                "cross-device edge {key:?} has no queued timeline dependency"
            ))
        })?;
        Ok(Some(VulkanTimelineSemaphorePoint::new(
            &synchronization.destination_wait,
            value,
        )))
    }

    fn enqueue_source_staging_transfer<'a>(
        &'a self,
        endpoint: &VulkanPlacedEdgeEndpoint,
        source_device: &'a VulkanComputeDevice,
        wait_points: &[VulkanTimelineSemaphorePoint<'_>],
        submission_batch: Option<&VulkanResidentQueueSubmissionBatch<'a>>,
    ) -> Result<bool, VulkanError> {
        let key = VulkanPlacedEdgePacketKey::from_outgoing_endpoint(endpoint);
        let Some(synchronization) = self.edges.get(&key) else {
            return Ok(false);
        };
        let Some(staging) = &synchronization.device_local_staging else {
            return Ok(false);
        };
        let signal = self
            .prepare_source_signal(endpoint)?
            .expect("staged edge synchronization has a source signal");
        if let Some(batch) = submission_batch {
            batch.enqueue_resident_buffer_copy(
                source_device,
                &staging.source_copy,
                wait_points,
                &[signal],
            )?;
        } else {
            source_device.submit_resident_buffer_copy_with_timeline_semaphores(
                &staging.source_copy,
                wait_points,
                &[signal],
            )?;
        }
        Ok(true)
    }

    fn enqueue_destination_staging_transfer<'a>(
        &'a self,
        endpoint: &VulkanPlacedEdgeEndpoint,
        destination_device: &'a VulkanComputeDevice,
        submission_batch: Option<&VulkanResidentQueueSubmissionBatch<'a>>,
    ) -> Result<bool, VulkanError> {
        let key = VulkanPlacedEdgePacketKey::from_incoming_endpoint(endpoint);
        let Some(synchronization) = self.edges.get(&key) else {
            return Ok(false);
        };
        let Some(staging) = &synchronization.device_local_staging else {
            return Ok(false);
        };
        let wait = self
            .take_destination_wait(endpoint)?
            .expect("staged edge synchronization has a destination wait");
        if let Some(batch) = submission_batch {
            batch.enqueue_resident_buffer_copy(
                destination_device,
                &staging.destination_copy,
                &[wait],
                &[],
            )?;
        } else {
            destination_device.submit_resident_buffer_copy_with_timeline_semaphores(
                &staging.destination_copy,
                &[wait],
                &[],
            )?;
        }
        Ok(true)
    }

    fn has_pending_dependencies(&self) -> bool {
        self.edges
            .values()
            .any(|synchronization| synchronization.pending_value.get().is_some())
    }

}

fn create_placed_device_links<'a, F>(
    device_slices: &[Arc<VulkanResidentModelPackageDeviceSlice>],
    distributed_activation_buffers: &mut VulkanDistributedActivationBuffers,
    device_for: &F,
) -> Result<VulkanPlacedDeviceLinks, VulkanResidentInProcessPlacedRuntimeError>
where
    F: Fn(&str) -> Result<&'a VulkanComputeDevice, VulkanResidentInProcessPlacedRuntimeError>,
{
    let plans = device_slices
        .iter()
        .map(|slice| {
            VulkanPlacedEdgeIoPlan::from_placed_resident_plan(
                &slice.placed_plan.placed_resident_plan,
            )
            .map_err(|error| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                    "failed to plan shared edge endpoints for {:?}: {error}",
                    slice.device_id
                )))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let edge_groups = group_placed_edge_pairs_by_produced_port(
        pair_placed_edge_endpoints(&plans)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
    )
    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;

    let mut local_edge_overrides =
        BTreeMap::<String, Vec<VulkanPlacedLocalEdgeBufferOverride>>::new();
    let mut endpoint_overrides =
        BTreeMap::<String, Vec<VulkanPlacedEdgeEndpointBufferOverride>>::new();
    let mut synchronizations = BTreeMap::new();
    let mut every_edge_is_resident_replayable = true;
    for group in edge_groups {
        let source_device = device_for(&group.source_device_id)?;
        let matching_local_edges = plans
            .iter()
            .find(|plan| plan.device_id == group.source_device_id)
            .into_iter()
            .flat_map(|plan| &plan.local_edges)
            .filter(|edge| {
                edge.source_component_id == group.source_component_id
                    && edge.source_port_id == group.source_port_id
            })
            .cloned()
            .collect::<Vec<_>>();
        let produced_edge_indices = matching_local_edges
            .iter()
            .map(|edge| edge.edge_index)
            .chain(group.edges.iter().map(|(outgoing, _)| outgoing.edge_index))
            .collect::<BTreeSet<_>>();

        let mut participant_device_ids = group
            .edges
            .iter()
            .flat_map(|(outgoing, incoming)| {
                [
                    outgoing.local_device_id.clone(),
                    incoming.local_device_id.clone(),
                ]
            })
            .collect::<BTreeSet<_>>();
        for allocation in &distributed_activation_buffers.allocations {
            if matches!(
                allocation.planned.storage,
                VulkanDistributedActivationStorage::Edge { edge_index, .. }
                    if produced_edge_indices.contains(&edge_index)
            ) {
                if allocation.planned.byte_capacity != group.byte_capacity {
                    return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                        VulkanError(format!(
                            "distributed produced port {}.{} has capacities {} and {}",
                            group.source_component_id,
                            group.source_port_id,
                            group.byte_capacity,
                            allocation.planned.byte_capacity,
                        )),
                    ));
                }
                participant_device_ids.extend(allocation.device_buffers.keys().cloned());
            }
        }

        let mut unique_devices = Vec::<(&VulkanComputeDevice, Vec<String>)>::new();
        for device_id in &participant_device_ids {
            let device = device_for(device_id)?;
            if let Some((_, logical_ids)) = unique_devices
                .iter_mut()
                .find(|(candidate, _)| candidate.shares_logical_device_with(device))
            {
                logical_ids.push(device_id.clone());
            } else {
                unique_devices.push((device, vec![device_id.clone()]));
            }
        }
        let owner_index = unique_devices
            .iter()
            .position(|(device, _)| device.shares_logical_device_with(source_device))
            .expect("produced-port participants include the source device");
        unique_devices.swap(0, owner_index);
        let peer_devices = unique_devices
            .iter()
            .skip(1)
            .map(|(device, _)| *device)
            .collect::<Vec<_>>();
        let supports_cross_queue_timeline = peer_devices.iter().all(|destination| {
            source_device.supports_opaque_fd_timeline_semaphores()
                && destination.supports_opaque_fd_timeline_semaphores()
        });

        let (physical_buffers, shared_route, staging_buffers, group_is_resident_replayable) =
            if peer_devices.is_empty() {
                (
                    vec![Arc::new(
                        source_device
                            .create_resident_buffer(group.byte_capacity)
                            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
                    )],
                    None,
                    None,
                    true,
                )
            } else if supports_cross_queue_timeline
                && let Ok(shared) = source_device
                    .create_shared_resident_buffers(&peer_devices, group.byte_capacity)
            {
                match shared.route {
                    VulkanSharedResidentBufferRoute::ExternalDeviceLocal => (
                        shared.buffers,
                        Some(VulkanSharedResidentBufferRoute::ExternalDeviceLocal),
                        None,
                        true,
                    ),
                    VulkanSharedResidentBufferRoute::SharedHost => {
                        let device_local = unique_devices
                            .iter()
                            .map(|(device, _)| {
                                device
                                    .create_resident_buffer(group.byte_capacity)
                                    .map(Arc::new)
                                    .map_err(
                                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop,
                                    )
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        (device_local, None, Some(shared.buffers), true)
                    }
                }
            } else {
                let staging_allocation = source_device
                    .create_shared_host_allocation(&peer_devices, group.byte_capacity)
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                let staging = unique_devices
                    .iter()
                    .map(|(device, _)| {
                        device
                            .import_shared_host_buffer(Arc::clone(&staging_allocation))
                            .map(Arc::new)
                            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let device_local = unique_devices
                    .iter()
                    .map(|(device, _)| {
                        device
                            .create_resident_buffer(group.byte_capacity)
                            .map(Arc::new)
                            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                (
                    device_local,
                    None,
                    Some(staging),
                    supports_cross_queue_timeline,
                )
            };
        every_edge_is_resident_replayable &= group_is_resident_replayable;
        let mut group_buffers = BTreeMap::<String, Arc<VulkanResidentBuffer>>::new();
        for ((_, logical_ids), buffer) in unique_devices.iter().zip(&physical_buffers) {
            for device_id in logical_ids {
                group_buffers.insert(device_id.clone(), Arc::clone(buffer));
            }
        }
        let source_buffer = group_buffers
            .get(&group.source_device_id)
            .cloned()
            .expect("produced-port source buffer was allocated");

        for allocation in &mut distributed_activation_buffers.allocations {
            if matches!(
                allocation.planned.storage,
                VulkanDistributedActivationStorage::Edge { edge_index, .. }
                    if produced_edge_indices.contains(&edge_index)
            ) {
                for (device_id, buffer) in &mut allocation.device_buffers {
                    *buffer = group_buffers.get(device_id).cloned().ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            format!(
                                "produced port {}.{} has no shared allocation on {device_id:?}",
                                group.source_component_id, group.source_port_id,
                            ),
                        ))
                    })?;
                }
                if let Some(route) = shared_route {
                    allocation.route = route;
                    allocation.external_device_local_error = None;
                }
            }
        }

        for edge in matching_local_edges {
            local_edge_overrides
                .entry(group.source_device_id.clone())
                .or_default()
                .push(VulkanPlacedLocalEdgeBufferOverride {
                    edge_index: edge.edge_index,
                    buffer: Arc::clone(&source_buffer),
                });
        }

        for (outgoing, incoming) in group.edges {
            let destination_device = device_for(&incoming.local_device_id)?;
            let devices_share_queue = source_device.shares_logical_device_with(destination_device);
            let incoming_buffer = group_buffers
                .get(&incoming.local_device_id)
                .cloned()
                .expect("produced-port destination buffer was allocated");
            if !devices_share_queue && (shared_route.is_some() || staging_buffers.is_some()) {
                let (transfer_route, device_local_staging) = if let Some(route) = shared_route {
                    let transfer_route = match route {
                        VulkanSharedResidentBufferRoute::ExternalDeviceLocal => {
                            VulkanPlacedEdgeTransferRoute::ExternalDeviceLocal
                        }
                        VulkanSharedResidentBufferRoute::SharedHost => unreachable!(
                            "host-shared produced ports use explicit device-local staging"
                        ),
                    };
                    (transfer_route, None)
                } else {
                    let source_staging = staging_buffers
                        .as_ref()
                        .and_then(|buffers| buffers.first())
                        .cloned()
                        .ok_or_else(|| {
                            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                "staged produced port has no source staging buffer".to_string(),
                            ))
                        })?;
                    let destination_index = unique_devices
                        .iter()
                        .position(|(device, _)| {
                            device.shares_logical_device_with(destination_device)
                        })
                        .expect("produced-port participants include the destination device");
                    let destination_staging = staging_buffers
                        .as_ref()
                        .and_then(|buffers| buffers.get(destination_index))
                        .cloned()
                        .ok_or_else(|| {
                            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                "staged produced port has no destination staging buffer"
                                    .to_string(),
                            ))
                        })?;
                    let source_copy = Box::new(
                        source_device
                            .create_resident_buffer_copy(
                                &source_buffer,
                                &source_staging,
                                group.byte_capacity,
                            )
                            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
                    );
                    let destination_copy = Box::new(
                        destination_device
                            .create_resident_buffer_copy(
                                &destination_staging,
                                &incoming_buffer,
                                group.byte_capacity,
                            )
                            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
                    );
                    let sample_source_copy = Box::new(
                        source_device
                            .create_timestamped_resident_buffer_copy(
                                &source_buffer,
                                &source_staging,
                                group.byte_capacity,
                            )
                            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
                    );
                    let sample_destination_copy = Box::new(
                        destination_device
                            .create_timestamped_resident_buffer_copy(
                                &destination_staging,
                                &incoming_buffer,
                                group.byte_capacity,
                            )
                            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
                    );
                    (
                        VulkanPlacedEdgeTransferRoute::DeviceLocalStaging,
                        Some(VulkanPlacedEdgeDeviceLocalStaging {
                            source_copy,
                            destination_copy,
                            sample_source_copy,
                            sample_destination_copy,
                            last_sampled_transfer_duration_ns: Cell::new(None),
                            _source_staging: source_staging,
                            _destination_staging: destination_staging,
                        }),
                    )
                };
                let source_signal = source_device
                    .create_opaque_fd_exportable_timeline_semaphore(0)
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                let destination_wait = destination_device
                    .create_timeline_semaphore(0)
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                destination_device
                    .import_timeline_semaphore_opaque_fd(
                        &destination_wait,
                        source_device
                            .export_timeline_semaphore_opaque_fd(&source_signal)
                            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
                    )
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                let key = VulkanPlacedEdgePacketKey::from_outgoing_endpoint(&outgoing);
                if synchronizations
                    .insert(
                        key.clone(),
                        VulkanPlacedEdgeTimelineSynchronization {
                            source_signal,
                            destination_wait,
                            next_value: Cell::new(1),
                            pending_value: Cell::new(None),
                            transfer_route,
                            device_local_staging,
                        },
                    )
                    .is_some()
                {
                    return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                        VulkanError(format!(
                            "cross-device edge synchronization repeats {key:?}"
                        )),
                    ));
                }
            }
            endpoint_overrides
                .entry(outgoing.local_device_id.clone())
                .or_default()
                .push(VulkanPlacedEdgeEndpointBufferOverride {
                    direction: VulkanPlacedEdgeDirection::Outgoing,
                    edge_index: outgoing.edge_index,
                    buffer: Arc::clone(&source_buffer),
                });
            endpoint_overrides
                .entry(incoming.local_device_id.clone())
                .or_default()
                .push(VulkanPlacedEdgeEndpointBufferOverride {
                    direction: VulkanPlacedEdgeDirection::Incoming,
                    edge_index: incoming.edge_index,
                    buffer: incoming_buffer,
                });
        }
    }
    let mut unique_devices = Vec::<(&VulkanComputeDevice, Vec<String>)>::new();
    for slice in device_slices {
        let device = device_for(&slice.device_id)?;
        if let Some((_, device_ids)) = unique_devices
            .iter_mut()
            .find(|(candidate, _)| candidate.shares_logical_device_with(device))
        {
            device_ids.push(slice.device_id.clone());
        } else {
            unique_devices.push((device, vec![slice.device_id.clone()]));
        }
    }
    let mut stream_control_buffers = BTreeMap::new();
    let feedback_stream_control_is_resident_replayable = true;
    if let Some((owner_device, _)) = unique_devices.first() {
        let buffers = if unique_devices.len() == 1 {
            let mut buffer = owner_device
                .create_host_visible_resident_buffer(VULKAN_STREAM_CONTROL_BYTE_CAPACITY)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            buffer
                .persistently_map()
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            vec![Arc::new(buffer)]
        } else {
            let peers = unique_devices
                .iter()
                .skip(1)
                .map(|(device, _)| *device)
                .collect::<Vec<_>>();
            // Token/tick metadata is a tiny coherence-critical control plane,
            // not a bulk activation. Keep one host-coherent allocation bound
            // into every physical device so scalar execution, resident
            // feedback, rollback, and replay all observe exactly the same
            // bytes. A device-local DMA-BUF would need explicit external queue
            // ownership transfers for cross-device read-after-write; the few
            // control words cannot justify that weaker and more complex path.
            let allocation = owner_device
                .create_shared_host_allocation(&peers, VULKAN_STREAM_CONTROL_BYTE_CAPACITY)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            unique_devices
                .iter()
                .map(|(device, _)| {
                    device
                        .import_shared_host_buffer(Arc::clone(&allocation))
                        .map(Arc::new)
                        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
                })
                .collect::<Result<Vec<_>, _>>()?
        };
        for buffer in &buffers {
            buffer
                .write_bytes(&[0; VULKAN_STREAM_CONTROL_BYTE_CAPACITY])
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        }
        for ((_, device_ids), buffer) in unique_devices.iter().zip(buffers) {
            for device_id in device_ids {
                stream_control_buffers.insert(device_id.clone(), buffer.clone());
            }
        }
    }
    Ok(VulkanPlacedDeviceLinks {
        local_edge_overrides,
        endpoint_overrides,
        synchronizations: VulkanPlacedEdgeTimelineSynchronizations {
            edges: synchronizations,
            sample_cursor: Cell::new(0),
        },
        stream_control_buffers,
        every_edge_is_resident_replayable,
        feedback_stream_control_is_resident_replayable,
    })
}
