struct VulkanResidentBatchedOutputNormRunner {
    batch_capacity: usize,
    normalized_frames_buffer: VulkanResidentBuffer,
    norm_dispatch: VulkanResidentKernelDispatch,
    sequence_catalog: RefCell<BTreeMap<usize, VulkanResidentKernelSequence>>,
}

impl VulkanResidentBatchedOutputNormRunner {
    #[allow(clippy::too_many_arguments)]
    fn new(
        device: &VulkanComputeDevice,
        batch_capacity: usize,
        norm_batch_lane_tile_width: u32,
        raw_frames_buffer: &VulkanResidentBuffer,
        norm_weight: &VulkanPermanentParameterBufferAllocation,
        norm_spirv_words: &[u32],
        output_spec: &VulkanResidentOutputTransducerSpec,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        if batch_capacity == 0 {
            return Err(VulkanResidentInProcessPlacedRuntimeError::ZeroTickBudget);
        }
        let norm_tile_width = usize::try_from(norm_batch_lane_tile_width).map_err(|_| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "batched output norm lane tile width exceeds usize".to_string(),
            ))
        })?;
        if norm_tile_width == 0 {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError("batched output norm lane tile width is zero".to_string()),
            ));
        }
        validate_output_embedding_norm_weight(norm_weight, output_spec)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::OutputTransducer)?;
        let normalized_frames_byte_capacity = output_spec
            .normalized_frame_byte_capacity
            .checked_mul(batch_capacity)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "batched normalized frame capacity overflowed".to_string(),
                ))
            })?;
        if raw_frames_buffer.byte_capacity() < normalized_frames_byte_capacity {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "batched raw output buffer has {} bytes, requires {normalized_frames_byte_capacity}",
                    raw_frames_buffer.byte_capacity()
                )),
            ));
        }
        let normalized_frames_buffer = device
            .create_resident_buffer(normalized_frames_byte_capacity)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let norm_workgroup_count_y = batch_capacity
            .checked_add(norm_tile_width - 1)
            .map(|width| width / norm_tile_width)
            .and_then(|count| u32::try_from(count).ok())
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "batched output norm workgroup count overflowed".to_string(),
                ))
            })?;
        let norm_bindings = [
            VulkanResidentKernelBufferBinding::new(
                0,
                raw_frames_buffer,
                normalized_frames_byte_capacity,
            )
            .with_access(VulkanResidentKernelBufferAccess::Read),
            VulkanResidentKernelBufferBinding::new(
                1,
                &normalized_frames_buffer,
                normalized_frames_byte_capacity,
            )
            .with_access(VulkanResidentKernelBufferAccess::Write),
            norm_weight
                .kernel_binding(2)
                .with_access(VulkanResidentKernelBufferAccess::Read),
        ];
        let norm_dispatch = device
            .create_resident_kernel_dispatch_2d(
                norm_spirv_words,
                &norm_bindings,
                1,
                norm_workgroup_count_y,
                output_spec.norm_local_size_x,
                std::mem::size_of::<u32>() as u32,
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        Ok(Self {
            batch_capacity,
            normalized_frames_buffer,
            norm_dispatch,
            sequence_catalog: RefCell::new(BTreeMap::new()),
        })
    }

    fn run(
        &self,
        device: &VulkanComputeDevice,
        batch_width: usize,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        if batch_width == 0 || batch_width > self.batch_capacity {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "batched output norm capacity {} cannot process {} frames",
                    self.batch_capacity, batch_width
                )),
            ));
        }
        let control = u32::try_from(batch_width)
            .map_err(|_| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "batched output norm width exceeds u32".to_string(),
                ))
            })?
            .to_le_bytes();
        if !self.sequence_catalog.borrow().contains_key(&batch_width) {
            let sequence = device
                .create_resident_kernel_sequence()
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            self.sequence_catalog
                .borrow_mut()
                .insert(batch_width, sequence);
        }
        let catalog = self.sequence_catalog.borrow();
        let sequence = catalog
            .get(&batch_width)
            .expect("batched output norm sequence was inserted");
        if sequence.has_recorded_commands() {
            device
                .run_recorded_resident_kernel_sequence(sequence)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
        } else {
            device
                .run_resident_kernel_sequence(
                    sequence,
                    &[VulkanResidentKernelSequenceStep::new(
                        &self.norm_dispatch,
                        &control,
                    )],
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
        }
    }
}

struct VulkanResidentBatchedOutputProjectionRunner {
    projection: VulkanResidentBatchedOutputProjectionKernelRunner,
    sampler_submission_catalog:
        RefCell<BTreeMap<usize, VulkanResidentQueueSubmissionTemplate>>,
    projection_sampler_submission_catalog:
        RefCell<BTreeMap<usize, VulkanResidentQueueSubmissionTemplate>>,
    sampler_views: Vec<VulkanResidentSamplerLogitsView>,
}

struct VulkanResidentBatchedOutputProjectionKernelRunner {
    batch_capacity: usize,
    norm: VulkanResidentBatchedOutputNormRunner,
    batched_logits_buffer: VulkanResidentBuffer,
    projection_dispatch: VulkanResidentKernelDispatch,
    projection_sequence_catalog: RefCell<BTreeMap<usize, VulkanResidentKernelSequence>>,
}

impl VulkanResidentBatchedOutputProjectionKernelRunner {
    #[allow(clippy::too_many_arguments)]
    fn new(
        device: &VulkanComputeDevice,
        batch_capacity: usize,
        norm_batch_lane_tile_width: u32,
        batch_lane_tile_width: u32,
        raw_frames_buffer: &VulkanResidentBuffer,
        norm_weight: &VulkanPermanentParameterBufferAllocation,
        projection_weight: &VulkanPermanentParameterBufferAllocation,
        projection_scale: Option<&VulkanPermanentParameterBufferAllocation>,
        norm_spirv_words: &[u32],
        projection_spirv_words: &[u32],
        output_spec: &VulkanResidentOutputTransducerSpec,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        if batch_capacity == 0 {
            return Err(VulkanResidentInProcessPlacedRuntimeError::ZeroTickBudget);
        }
        let tile_width = usize::try_from(batch_lane_tile_width).map_err(|_| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "batched output projection lane tile width exceeds usize".to_string(),
            ))
        })?;
        if tile_width == 0 {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError("batched output projection lane tile width is zero".to_string()),
            ));
        }
        validate_output_projection_weight(projection_weight, output_spec)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::OutputTransducer)?;
        validate_output_projection_scale(projection_scale, output_spec)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::OutputTransducer)?;
        let norm = VulkanResidentBatchedOutputNormRunner::new(
            device,
            batch_capacity,
            norm_batch_lane_tile_width,
            raw_frames_buffer,
            norm_weight,
            norm_spirv_words,
            output_spec,
        )?;
        let normalized_frames_byte_capacity = norm.normalized_frames_buffer.byte_capacity();
        let batched_logits_byte_capacity = output_spec
            .logits_byte_capacity
            .checked_mul(batch_capacity)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "batched logits capacity overflowed".to_string(),
                ))
            })?;
        let batched_logits_buffer = device
            .create_resident_buffer(batched_logits_byte_capacity)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let workgroup_count_y = batch_capacity
            .checked_add(tile_width - 1)
            .map(|width| width / tile_width)
            .and_then(|count| u32::try_from(count).ok())
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "batched output projection workgroup count overflowed".to_string(),
                ))
            })?;
        let mut bindings = vec![
            VulkanResidentKernelBufferBinding::new(
                0,
                &norm.normalized_frames_buffer,
                normalized_frames_byte_capacity,
            )
            .with_access(VulkanResidentKernelBufferAccess::Read),
            projection_weight
                .kernel_binding(1)
                .with_access(VulkanResidentKernelBufferAccess::Read),
            VulkanResidentKernelBufferBinding::new(
                2,
                &batched_logits_buffer,
                batched_logits_byte_capacity,
            )
            .with_access(VulkanResidentKernelBufferAccess::Write),
        ];
        if let Some(scale) = projection_scale {
            bindings.push(
                scale
                    .kernel_binding(3)
                    .with_access(VulkanResidentKernelBufferAccess::Read),
            );
        }
        let projection_dispatch = device
            .create_resident_kernel_dispatch_2d(
                projection_spirv_words,
                &bindings,
                output_spec.projection_workgroup_count_x,
                workgroup_count_y,
                output_spec.projection_local_size_x,
                std::mem::size_of::<u32>() as u32,
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        Ok(Self {
            batch_capacity,
            norm,
            batched_logits_buffer,
            projection_dispatch,
            projection_sequence_catalog: RefCell::new(BTreeMap::new()),
        })
    }

    fn projection_sequence(
        &self,
        device: &VulkanComputeDevice,
        batch_width: usize,
    ) -> Result<
        std::cell::Ref<'_, VulkanResidentKernelSequence>,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        if batch_width == 0 || batch_width > self.batch_capacity {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "batched output projection capacity {} cannot process {} frames",
                    self.batch_capacity, batch_width
                )),
            ));
        }
        let batch_width = u32::try_from(batch_width).map_err(|_| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "batched output projection width exceeds u32".to_string(),
            ))
        })?;
        let batch_width_usize = batch_width as usize;
        if !self
            .projection_sequence_catalog
            .borrow()
            .contains_key(&batch_width_usize)
        {
            let sequence = if std::env::var_os("NERVE_VK_PERF_LOGGER").is_some() {
                device.create_profiled_resident_kernel_sequence(2)
            } else {
                device.create_resident_kernel_sequence()
            }
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            self.projection_sequence_catalog
                .borrow_mut()
                .insert(batch_width_usize, sequence);
        }
        let catalog = self.projection_sequence_catalog.borrow();
        let sequence = catalog
            .get(&batch_width_usize)
            .expect("batched projection sequence was inserted");
        if !sequence.has_recorded_commands() {
            device
                .record_resident_kernel_sequence(
                    sequence,
                    &[
                            VulkanResidentKernelSequenceStep::new(
                            &self.norm.norm_dispatch,
                            &batch_width.to_le_bytes(),
                        ),
                        VulkanResidentKernelSequenceStep::new(
                            &self.projection_dispatch,
                            &batch_width.to_le_bytes(),
                        ),
                    ],
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        }
        Ok(std::cell::Ref::map(catalog, |catalog| {
            catalog
                .get(&batch_width_usize)
                .expect("batched projection sequence was inserted")
        }))
    }

    fn project(
        &self,
        device: &VulkanComputeDevice,
        batch_width: usize,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let _projection = runtime_critical_path_span(RuntimeCriticalPathPhase::OutputProjection);
        let sequence = self.projection_sequence(device, batch_width)?;
        device
            .run_recorded_resident_kernel_sequence(&sequence)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let duration_ns = device
            .read_recorded_resident_kernel_sequence_duration_ns(&sequence)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        record_runtime_critical_path_device_duration(
            RuntimeCriticalPathPhase::OutputProjection,
            duration_ns,
        );
        Ok(())
    }
}

impl VulkanResidentBatchedOutputProjectionRunner {
    #[allow(clippy::too_many_arguments)]
    fn new_for_sampler_lanes(
        device: &VulkanComputeDevice,
        norm_batch_lane_tile_width: u32,
        batch_lane_tile_width: u32,
        raw_frames_buffer: &VulkanResidentBuffer,
        norm_weight: &VulkanPermanentParameterBufferAllocation,
        projection_weight: &VulkanPermanentParameterBufferAllocation,
        projection_scale: Option<&VulkanPermanentParameterBufferAllocation>,
        norm_spirv_words: &[u32],
        projection_spirv_words: &[u32],
        output_spec: &VulkanResidentOutputTransducerSpec,
        sampler_lanes: &[&VulkanResidentSamplerRunner],
        sampler_kernels: &[VulkanResidentSamplerKernelArtifact],
        sampler_spec: &VulkanResidentSamplerSpec,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        let projection = VulkanResidentBatchedOutputProjectionKernelRunner::new(
            device,
            sampler_lanes.len(),
            norm_batch_lane_tile_width,
            batch_lane_tile_width,
            raw_frames_buffer,
            norm_weight,
            projection_weight,
            projection_scale,
            norm_spirv_words,
            projection_spirv_words,
            output_spec,
        )?;
        let mut sampler_views = Vec::with_capacity(sampler_lanes.len());
        for (batch_index, sampler) in
            sampler_lanes.iter().copied().enumerate()
        {
            let logits_byte_offset = output_spec
                .logits_byte_capacity
                .checked_mul(batch_index)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                        VulkanError(
                            "batched sampler logits offset overflowed"
                                .to_string(),
                        ),
                    )
                })?;
            sampler_views.push(
                sampler
                    .create_logits_view(
                        device,
                        &projection.batched_logits_buffer,
                        logits_byte_offset,
                        sampler_kernels,
                        sampler_spec,
                    )
                    .map_err(
                        VulkanResidentInProcessPlacedRuntimeError::Sampler,
                    )?,
            );
        }
        Ok(Self {
            projection,
            sampler_submission_catalog: RefCell::new(BTreeMap::new()),
            projection_sampler_submission_catalog: RefCell::new(
                BTreeMap::new(),
            ),
            sampler_views,
        })
    }

    fn projection_sequence(
        &self,
        device: &VulkanComputeDevice,
        batch_width: usize,
    ) -> Result<
        std::cell::Ref<'_, VulkanResidentKernelSequence>,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        self.projection.projection_sequence(device, batch_width)
    }

    fn project(
        &self,
        device: &VulkanComputeDevice,
        batch_width: usize,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        self.projection.project(device, batch_width)
    }

    fn sample_independent_streams(
        &self,
        device: &VulkanComputeDevice,
        input_token_ids: &[u32],
        stream_ticks: &[u64],
        dynamic_state_capacities: &[u32],
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let token_prefixes = vec![&[][..]; input_token_ids.len()];
        self.sample_lanes(
            device,
            &token_prefixes,
            stream_ticks,
            dynamic_state_capacities,
        )
    }

    fn sample_lanes(
        &self,
        device: &VulkanComputeDevice,
        token_prefixes: &[&[u32]],
        stream_ticks: &[u64],
        dynamic_state_capacities: &[u32],
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let _sampling = runtime_critical_path_span(RuntimeCriticalPathPhase::Sampling);
        let batch_width = self.prepare_sampler_lanes(
            device,
            token_prefixes,
            stream_ticks,
            dynamic_state_capacities,
        )?;
        if !self
            .sampler_submission_catalog
            .borrow()
            .contains_key(&batch_width)
        {
            let submission_batch = VulkanResidentQueueSubmissionBatch::new();
            for (batch_index, view) in
                self.sampler_views.iter().take(batch_width).enumerate()
            {
                submission_batch
                    .enqueue_recorded_sequence(
                        device,
                        &view.sequence,
                        &[],
                        &[],
                        batch_index + 1 == batch_width,
                    )
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            }
            let template = submission_batch
                .mount()
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            self.sampler_submission_catalog
                .borrow_mut()
                .insert(batch_width, template);
        }
        self.sampler_submission_catalog
            .borrow()
            .get(&batch_width)
            .expect("batched sampler submission template was inserted")
            .submit_with_timeline_value_offset(0)
            .map(|_| ())
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        device
            .wait_resident_kernel_sequence(&self.sampler_views[batch_width - 1].sequence)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        for view in self.sampler_views.iter().take(batch_width) {
            let duration_ns = device
                .read_recorded_resident_kernel_sequence_duration_ns(&view.sequence)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            record_runtime_critical_path_device_duration(
                RuntimeCriticalPathPhase::Sampling,
                duration_ns,
            );
        }
        Ok(())
    }

    fn project_and_sample_lanes(
        &self,
        device: &VulkanComputeDevice,
        token_prefixes: &[&[u32]],
        stream_ticks: &[u64],
        dynamic_state_capacities: &[u32],
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let _output = runtime_critical_path_span(RuntimeCriticalPathPhase::OutputProjection);
        let profile = std::env::var_os("NERVE_VK_PERF_LOGGER").is_some();
        let started = profile.then(std::time::Instant::now);
        let batch_width = self.prepare_sampler_lanes(
            device,
            token_prefixes,
            stream_ticks,
            dynamic_state_capacities,
        )?;
        let prepared = started.map(|started| started.elapsed());
        if !self
            .projection_sampler_submission_catalog
            .borrow()
            .contains_key(&batch_width)
        {
            let projection_sequence = self.projection_sequence(device, batch_width)?;
            let submission_batch = VulkanResidentQueueSubmissionBatch::new();
            submission_batch
                .enqueue_recorded_sequence(
                    device,
                    &projection_sequence,
                    &[],
                    &[],
                    false,
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            for (batch_index, view) in
                self.sampler_views.iter().take(batch_width).enumerate()
            {
                submission_batch
                    .enqueue_recorded_sequence(
                        device,
                        &view.sequence,
                        &[],
                        &[],
                        batch_index + 1 == batch_width,
                    )
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            }
            let template = submission_batch
                .mount()
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            self.projection_sampler_submission_catalog
                .borrow_mut()
                .insert(batch_width, template);
        }
        self.projection_sampler_submission_catalog
            .borrow()
            .get(&batch_width)
            .expect("batched projection and sampler submission template was inserted")
            .submit_with_timeline_value_offset(0)
            .map(|_| ())
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        device
            .wait_resident_kernel_sequence(&self.sampler_views[batch_width - 1].sequence)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let projection_sequence = self.projection.projection_sequence(device, batch_width)?;
        let projection_duration_ns = device
            .read_recorded_resident_kernel_sequence_duration_ns(&projection_sequence)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        record_runtime_critical_path_device_duration(
            RuntimeCriticalPathPhase::OutputProjection,
            projection_duration_ns,
        );
        for view in self.sampler_views.iter().take(batch_width) {
            let sampler_duration_ns = device
                .read_recorded_resident_kernel_sequence_duration_ns(&view.sequence)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            record_runtime_critical_path_device_duration(
                RuntimeCriticalPathPhase::Sampling,
                sampler_duration_ns,
            );
        }
        if profile {
            let catalog =
                self.projection.projection_sequence_catalog.borrow();
            let sequence = catalog
                .get(&batch_width)
                .expect("batched projection sequence was inserted");
            let durations = device
                .read_recorded_resident_kernel_step_durations_ns(sequence)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            if durations.len() != 2 {
                return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                    VulkanError(format!(
                        "batched output projection produced {} profile steps; expected 2",
                        durations.len()
                    )),
                ));
            }
            eprintln!(
                "nerve Vulkan output projection: width={} norm_us={:.3} projection_us={:.3}",
                batch_width,
                durations[0] as f64 / 1_000.0,
                durations[1] as f64 / 1_000.0,
            );
            for (lane, view) in self.sampler_views.iter().take(batch_width).enumerate() {
                let sampler_durations = device
                    .read_recorded_resident_kernel_step_durations_ns(&view.sequence)
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                let stages = sampler_durations
                    .iter()
                    .map(|duration| format!("{:.3}", *duration as f64 / 1_000.0))
                    .collect::<Vec<_>>()
                    .join(",");
                eprintln!(
                    "nerve Vulkan sampler: width={} lane={} stages_us=[{}] total_us={:.3}",
                    batch_width,
                    lane,
                    stages,
                    sampler_durations.iter().sum::<u64>() as f64 / 1_000.0,
                );
            }
            eprintln!(
                "nerve Vulkan output wall: width={} prepare_us={:.3} total_us={:.3}",
                batch_width,
                prepared.expect("profile timer exists").as_secs_f64() * 1_000_000.0,
                started.expect("profile timer exists").elapsed().as_secs_f64() * 1_000_000.0,
            );
        }
        Ok(())
    }

    fn prepare_sampler_lanes(
        &self,
        device: &VulkanComputeDevice,
        token_prefixes: &[&[u32]],
        stream_ticks: &[u64],
        dynamic_state_capacities: &[u32],
    ) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError> {
        let batch_width = token_prefixes.len();
        if batch_width == 0 || batch_width > self.sampler_views.len() {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "batched output projection has {} sampler lanes, cannot sample {batch_width}",
                    self.sampler_views.len()
                )),
            ));
        }
        if stream_ticks.len() != batch_width
            || dynamic_state_capacities.len() != batch_width
        {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "batched sampler has {batch_width} token lanes, {} stream ticks, and {} state capacities",
                    stream_ticks.len(),
                    dynamic_state_capacities.len()
                )),
            ));
        }
        for (batch_index, view) in self.sampler_views.iter().take(batch_width).enumerate() {
            view.prepare_token_state(device, token_prefixes[batch_index])
                .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)?;
            view.prepare_stream_tick(
                stream_ticks[batch_index],
                dynamic_state_capacities[batch_index],
            )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            if !view.sequence.has_recorded_commands() {
                view.record(device)
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            }
        }
        Ok(batch_width)
    }
}
