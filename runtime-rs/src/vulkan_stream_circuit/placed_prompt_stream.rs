struct VulkanResidentInProcessPlacedPendingStreamFeedbackWindow {
    window: VulkanResidentInProcessPlacedPendingFeedbackWindow,
    started_at: Instant,
    adaptive_feedback_loads_before: Option<u64>,
}

fn demand_feedback_retry_deferred_after_scalar(
    was_deferred: bool,
    input_is_feedback: bool,
    should_emit_public_output: bool,
    input_closes_loop_after_processing: bool,
    residency_changed: bool,
) -> bool {
    was_deferred
        && (!input_is_feedback
            || !should_emit_public_output
            || input_closes_loop_after_processing
            || residency_changed)
}

pub struct VulkanResidentInProcessPlacedPromptStream {
    package: Arc<VulkanResidentInProcessPlacedModelPackage>,
    processor: VulkanResidentInProcessPlacedStreamProcessor,
    devices: BTreeMap<String, Rc<VulkanComputeDevice>>,
    session: VulkanResidentInProcessPlacedPromptSession,
    transient_state_pages: VulkanResidentTransientStatePageTable,
    transaction_page_cow_depth: usize,
    active_input_event: Option<VulkanResidentInProcessPlacedActivePromptEvent>,
    pending_input_events: VecDeque<VulkanResidentTokenInputEvent>,
    speculative_draft_tokens: usize,
    speculative_confidence_threshold: f32,
    feedback_execution_selector: Option<VulkanAdaptiveFeedbackExecutionSelector>,
    demand_feedback_retry_deferred: bool,
    resident_feedback_template_catalog: VulkanResidentPlacedFeedbackTemplateCatalog,
    pending_scheduler_activation:
        Option<VulkanResidentInProcessPlacedPendingSchedulerActivation>,
}

impl VulkanResidentInProcessPlacedPromptStream {
    pub fn new(
        package: Arc<VulkanResidentInProcessPlacedModelPackage>,
        devices: BTreeMap<String, Rc<VulkanComputeDevice>>,
        random_seed: u32,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        Self::from_package_devices_and_session(package, devices, random_seed, 0)
    }

    pub fn with_speculative_draft_tokens(
        mut self,
        speculative_draft_tokens: usize,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        if speculative_draft_tokens > 0 && self.processor.speculative_decoder_count() == 0 {
            return Err(placed_scheduler_divergence(
                "speculative draft tokens require a mounted speculative decoder",
            ));
        }
        self.speculative_draft_tokens = speculative_draft_tokens;
        self.feedback_execution_selector = (speculative_draft_tokens > 0).then(|| {
            VulkanAdaptiveFeedbackExecutionSelector::new(
                self.processor
                    .effective_speculative_draft_token_count(speculative_draft_tokens),
                self.processor.resident_feedback_next_window_tick_count() >= 2,
            )
        });
        Ok(self)
    }

    pub fn with_speculative_confidence_threshold(
        mut self,
        confidence_threshold: f32,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        if !confidence_threshold.is_finite() || !(0.0..=1.0).contains(&confidence_threshold) {
            return Err(placed_scheduler_divergence(
                "speculative confidence threshold must be finite and in [0, 1]",
            ));
        }
        self.speculative_confidence_threshold = confidence_threshold;
        Ok(self)
    }

    pub fn from_runtime_model_for_bound_devices(
        devices: BTreeMap<String, Rc<VulkanComputeDevice>>,
        manifest_dir: impl AsRef<Path>,
        runtime_model: VulkanResidentRuntimeModel,
        dynamic_state_capacity_activations: Option<usize>,
        random_seed: u32,
        speculative_draft_tokens: usize,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        Self::from_runtime_model_for_bound_devices_with_sampler_config(
            devices,
            manifest_dir,
            runtime_model,
            dynamic_state_capacity_activations,
            random_seed,
            speculative_draft_tokens,
            VulkanResidentSamplerRuntimeConfig::default(),
        )
    }

    pub fn from_runtime_model_for_bound_devices_with_sampler_config(
        devices: BTreeMap<String, Rc<VulkanComputeDevice>>,
        manifest_dir: impl AsRef<Path>,
        runtime_model: VulkanResidentRuntimeModel,
        dynamic_state_capacity_activations: Option<usize>,
        random_seed: u32,
        speculative_draft_tokens: usize,
        sampler_config: VulkanResidentSamplerRuntimeConfig,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        Self::from_runtime_model_for_bound_devices_with_sampler_config_and_residency_policy(
            devices,
            manifest_dir,
            runtime_model,
            dynamic_state_capacity_activations,
            random_seed,
            speculative_draft_tokens,
            sampler_config,
            ResourceResidencyPolicy::Eager,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_runtime_model_for_bound_devices_with_sampler_config_and_residency_policy(
        devices: BTreeMap<String, Rc<VulkanComputeDevice>>,
        manifest_dir: impl AsRef<Path>,
        mut runtime_model: VulkanResidentRuntimeModel,
        dynamic_state_capacity_activations: Option<usize>,
        random_seed: u32,
        speculative_draft_tokens: usize,
        sampler_config: VulkanResidentSamplerRuntimeConfig,
        resource_residency_policy: ResourceResidencyPolicy,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        runtime_model.package.sampler.spec = sampler_config
            .apply_to(&runtime_model.package.sampler.spec)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)?;
        let package = Arc::new(
            VulkanResidentInProcessPlacedModelPackage::from_runtime_model_for_bound_devices_with_residency_policy(
                &devices,
                manifest_dir,
                runtime_model,
                dynamic_state_capacity_activations,
                speculative_draft_tokens,
                resource_residency_policy,
            )?,
        );
        Self::new(package, devices, random_seed)?
            .with_speculative_draft_tokens(speculative_draft_tokens)
    }

    pub fn from_package_devices_and_session(
        package: Arc<VulkanResidentInProcessPlacedModelPackage>,
        devices: BTreeMap<String, Rc<VulkanComputeDevice>>,
        random_seed: u32,
        start_stream_tick: u64,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        for device_id in &package.device_ids {
            if !devices.contains_key(device_id) {
                return Err(
                    VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                        device_id: device_id.clone(),
                    },
                );
            }
        }
        let processor = package.create_stream_processor_for_bound_devices(&devices, random_seed)?;
        let session = processor.prompt_session_from_stream_tick(start_stream_tick);
        Ok(Self {
            package,
            processor,
            devices,
            session,
            transient_state_pages: VulkanResidentTransientStatePageTable::default(),
            transaction_page_cow_depth: 0,
            active_input_event: None,
            pending_input_events: VecDeque::new(),
            speculative_draft_tokens: 0,
            speculative_confidence_threshold: 0.0,
            feedback_execution_selector: None,
            demand_feedback_retry_deferred: false,
            resident_feedback_template_catalog: BTreeMap::new(),
            pending_scheduler_activation: None,
        })
    }

    pub fn package(&self) -> &VulkanResidentInProcessPlacedModelPackage {
        &self.package
    }

    pub fn session(&self) -> &VulkanResidentInProcessPlacedPromptSession {
        &self.session
    }

    pub fn devices(&self) -> &BTreeMap<String, Rc<VulkanComputeDevice>> {
        &self.devices
    }

    pub fn resident_feedback_cancellation_handle(
        &self,
    ) -> Option<VulkanResidentFeedbackCancellationHandle> {
        self.processor
            .resident_feedback_loop
            .as_ref()
            .map(|feedback_loop| feedback_loop.control.cancellation_handle())
    }

    pub fn remount_model_preserving_state(
        &mut self,
        package: Arc<VulkanResidentInProcessPlacedModelPackage>,
        random_seed: u32,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let processor = package.create_stream_processor_inheriting_state_for_bound_devices(
            &self.devices,
            random_seed,
            &self.processor,
        )?;
        self.session.transport = VulkanInProcessPlacedEdgeTransport::new();
        self.package = package;
        self.processor = processor;
        self.feedback_execution_selector = (self.speculative_draft_tokens > 0).then(|| {
            VulkanAdaptiveFeedbackExecutionSelector::new(
                self.processor
                    .effective_speculative_draft_token_count(self.speculative_draft_tokens),
                self.processor.resident_feedback_next_window_tick_count() >= 2,
            )
        });
        self.resident_feedback_template_catalog.clear();
        self.pending_scheduler_activation = None;
        Ok(())
    }

    pub fn fork_preserving_state(
        &self,
        random_seed: u32,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        if !self.is_idle() || self.pending_scheduler_activation.is_some() {
            return Err(placed_scheduler_divergence(
                "cannot fork a placed prompt stream while work is pending",
            ));
        }
        let processor = self
            .package
            .create_stream_processor_inheriting_state_for_bound_devices(
                &self.devices,
                random_seed,
                &self.processor,
            )?;
        Ok(Self {
            package: Arc::clone(&self.package),
            processor,
            devices: self.devices.clone(),
            session: VulkanResidentInProcessPlacedPromptSession {
                next_stream_tick: self.session.next_stream_tick,
                completed_prompt_event_count: self.session.completed_prompt_event_count,
                generated_token_count: self.session.generated_token_count,
                output_token_count: self.session.output_token_count,
                transport: VulkanInProcessPlacedEdgeTransport::new(),
            },
            transient_state_pages: self.transient_state_pages.clone(),
            transaction_page_cow_depth: 0,
            active_input_event: None,
            pending_input_events: VecDeque::new(),
            speculative_draft_tokens: self.speculative_draft_tokens,
            speculative_confidence_threshold: self.speculative_confidence_threshold,
            feedback_execution_selector: self.feedback_execution_selector.clone(),
            demand_feedback_retry_deferred: self.demand_feedback_retry_deferred,
            resident_feedback_template_catalog: BTreeMap::new(),
            pending_scheduler_activation: None,
        })
    }

    pub fn reset_transient_state(
        &mut self,
    ) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError> {
        if !self.is_idle() || self.pending_scheduler_activation.is_some() {
            return Err(placed_scheduler_divergence(
                "cannot reset placed prompt stream state while work is pending",
            ));
        }
        let zeroed = self.processor.reset_transient_state_buffers()?;
        self.transient_state_pages.clear();
        self.session.next_stream_tick = 0;
        Ok(zeroed)
    }

    fn restore_initial_transaction_state(
        &mut self,
    ) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError> {
        if !self.is_idle() || self.pending_scheduler_activation.is_some() {
            return Err(placed_scheduler_divergence(
                "cannot restore initial transaction state while placed prompt work is pending",
            ));
        }
        let initialized = self
            .processor
            .restore_initial_transaction_state(&self.devices)?;
        self.transient_state_pages.clear();
        self.session.next_stream_tick = 0;
        Ok(initialized)
    }

    fn can_checkpoint_transaction_with_page_cow(
        &self,
        state: &TransientStateTableSnapshot,
    ) -> Result<bool, VulkanResidentInProcessPlacedRuntimeError> {
        for entry in state
            .entries
            .iter()
            .filter(|entry| entry.shape.retention == TransientStateRetention::Append)
        {
            let resident_state = self
                .processor
                .mounted_state_buffer(&entry.key)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::Package(
                        VulkanResidentTokenModelPackageError::new(format!(
                            "cannot checkpoint non-resident transaction state {}.{}",
                            entry.key.node_instance_id, entry.key.state_id,
                        )),
                    )
                })?;
            if !entry.block_ids.is_empty()
                && !self
                    .transient_state_pages
                    .has_free_physical_page_for(&entry.key, resident_state)
            {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn restore_transaction_page_checkpoint(
        &mut self,
        checkpoint: VulkanResidentTransientStatePageTable,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        checkpoint
            .sync_page_tables(&self.processor)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?;
        self.transient_state_pages = checkpoint;
        Ok(())
    }

    fn release_transaction_page_checkpoint(
        &mut self,
        state: &TransientStateTableSnapshot,
    ) {
        self.transient_state_pages.retain_blocks(state);
    }

    fn begin_transaction_page_cow(
        &mut self,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        self.transaction_page_cow_depth = self
            .transaction_page_cow_depth
            .checked_add(1)
            .ok_or_else(|| placed_scheduler_divergence("transaction page-COW depth overflowed"))?;
        Ok(())
    }

    fn end_transaction_page_cow(&mut self) {
        self.transaction_page_cow_depth = self.transaction_page_cow_depth.saturating_sub(1);
    }

    pub fn reset_for_new_session(
        &mut self,
        random_seed: u32,
    ) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError> {
        if !self.is_idle() || self.pending_scheduler_activation.is_some() {
            return Err(placed_scheduler_divergence(
                "cannot reset placed prompt stream for a new session while work is pending",
            ));
        }
        let zeroed = self
            .processor
            .reset_for_new_session(&self.devices, random_seed)?;
        self.processor.reset_resident_feedback_session_state()?;
        self.transient_state_pages.clear();
        self.session.next_stream_tick = 0;
        self.session.completed_prompt_event_count = 0;
        self.session.generated_token_count = 0;
        self.session.output_token_count = 0;
        self.session.transport.reset_tick_state();
        self.active_input_event = None;
        self.pending_input_events.clear();
        self.pending_scheduler_activation = None;
        self.demand_feedback_retry_deferred = false;
        self.resident_feedback_template_catalog.clear();
        Ok(zeroed)
    }

    pub fn set_random_seed(
        &self,
        random_seed: u32,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        self.processor.set_random_seed(random_seed)
    }

    pub fn next_stream_tick(&self) -> u64 {
        self.session.next_stream_tick
    }

    pub fn completed_prompt_event_count(&self) -> usize {
        self.session.completed_prompt_event_count
    }

    pub fn pending_input_event_count(&self) -> usize {
        self.pending_input_events.len() + usize::from(self.active_input_event.is_some())
    }

    pub fn is_idle(&self) -> bool {
        self.active_input_event.is_none() && self.pending_input_events.is_empty()
    }

    fn adaptive_feedback_uses_resident_loop(&self) -> bool {
        if self.demand_feedback_retry_deferred {
            return false;
        }
        self.feedback_execution_selector
            .as_ref()
            .filter(|_| self.speculative_confidence_threshold == 0.0)
            .map(|selector| {
                selector.next_candidate() == VulkanFeedbackExecutionCandidate::Resident
            })
            .unwrap_or(self.speculative_draft_tokens == 0)
    }

    pub fn resident_state_digest(
        &self,
    ) -> Result<String, VulkanResidentInProcessPlacedRuntimeError> {
        use sha2::{Digest, Sha256};

        if !self.is_idle() || self.pending_scheduler_activation.is_some() {
            return Err(placed_scheduler_divergence(
                "cannot snapshot placed prompt stream state while work is pending",
            ));
        }
        let mut digest = Sha256::new();
        digest.update(
            self.processor
                .resident_state_snapshot_digest(
                    &self.devices,
                    &self.transient_state_pages,
                )?,
        );
        let pages = self.transient_state_pages.snapshot_bytes();
        digest.update((pages.len() as u64).to_le_bytes());
        digest.update(pages);
        for value in [
            self.session.next_stream_tick as u64,
            self.session.completed_prompt_event_count as u64,
            self.session.generated_token_count as u64,
            self.session.output_token_count as u64,
        ] {
            digest.update(value.to_le_bytes());
        }
        Ok(format!(
            "nerve.optimizer.artifact_sha256.v1:{:x}",
            digest.finalize()
        ))
    }

    fn quiesce_and_discard_transaction_work(
        &mut self,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let pending_scheduler_activation =
            self.pending_scheduler_activation.take();
        let quiescence = pending_scheduler_activation
            .as_ref()
            .map(|pending| {
                self.wait_resident_feedback_window_for(&pending.window, u64::MAX)
                    .and_then(|completed| {
                        completed.then_some(()).ok_or_else(|| {
                            placed_scheduler_divergence(
                                "submitted feedback work did not quiesce during transaction restoration",
                            )
                        })
                    })
            })
            .unwrap_or(Ok(()));
        self.active_input_event = None;
        self.pending_input_events.clear();
        self.session.transport.reset_tick_state();
        quiescence
    }

    pub fn enqueue_input_event(
        &mut self,
        event: VulkanResidentTokenInputEvent,
    ) -> VulkanResidentInProcessPlacedQueuedInputEvent {
        self.pending_input_events.push_back(event.clone());
        VulkanResidentInProcessPlacedQueuedInputEvent {
            input_event: event,
            pending_input_event_count: self.pending_input_event_count(),
            next_stream_tick: self.next_stream_tick(),
        }
    }

    fn activate_next_input_event(
        &mut self,
    ) -> Result<bool, VulkanResidentInProcessPlacedRuntimeError> {
        if self.active_input_event.is_some() {
            return Ok(true);
        }
        let Some(input_event) = self.pending_input_events.pop_front() else {
            return Ok(false);
        };
        self.active_input_event = Some(VulkanResidentInProcessPlacedActivePromptEvent::new(
            input_event,
            self.session.next_stream_tick,
        )?);
        Ok(true)
    }

    fn run_temporal_external_input_block_with_output<F>(
        &mut self,
        on_output_event: &mut F,
    ) -> Result<
        (usize, Option<VulkanResidentInProcessPlacedSubmittedInputRun>),
        VulkanResidentInProcessPlacedRuntimeError,
    >
    where
        F: FnMut(VulkanResidentTokenOutputEvent),
    {
        self.run_temporal_external_input_block_limited_with_output(usize::MAX, on_output_event)
    }

    fn run_temporal_external_input_block_limited_with_output<F>(
        &mut self,
        max_external_inputs: usize,
        on_output_event: &mut F,
    ) -> Result<
        (usize, Option<VulkanResidentInProcessPlacedSubmittedInputRun>),
        VulkanResidentInProcessPlacedRuntimeError,
    >
    where
        F: FnMut(VulkanResidentTokenOutputEvent),
    {
        if !self.activate_next_input_event()? {
            return Ok((0, None));
        }
        if max_external_inputs < 2 {
            return Ok((0, None));
        }
        let active = self
            .active_input_event
            .as_ref()
            .expect("temporal block requires an active input event");
        let external_input_count = active
            .input_event
            .token_ids
            .len()
            .saturating_sub(active.next_external_input_index);
        if external_input_count < 2 || active.pending_feedback.is_some() {
            return Ok((0, None));
        }
        let block_width = self
            .processor
            .temporal_block_width(&self.devices, external_input_count)?;
        if block_width < 2 {
            return Ok((0, None));
        }
        if block_width > max_external_inputs {
            return Ok((0, None));
        }
        let block_start_index = active.next_external_input_index;
        let block_end_index = block_start_index + block_width;
        let input_token_ids =
            active.input_event.token_ids[block_start_index..block_end_index].to_vec();
        let sample_last = block_end_index == active.input_event.token_ids.len()
            && active.remaining_public_outputs > 0;
        let start_stream_tick = self.session.next_stream_tick;
        let block_run = self.processor.run_temporal_prompt_block(
            &self.devices,
            &input_token_ids,
            start_stream_tick,
            sample_last,
        )?;

        for (block_index, input_token_id) in input_token_ids.iter().enumerate() {
            let stream_tick = self.session.next_stream_tick;
            let activation = self
                .active_input_event
                .as_ref()
                .and_then(VulkanResidentInProcessPlacedActivePromptEvent::next_activation)
                .ok_or(VulkanResidentInProcessPlacedRuntimeError::MissingPrivateFeedback)?;
            if activation.input_is_feedback || activation.input_token_id != *input_token_id {
                return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                    VulkanError(
                        "temporal block diverged from the external input queue".to_string(),
                    ),
                ));
            }
            let transport_stats = if block_index + 1 == block_width {
                &block_run.transport_stats
            } else {
                &VulkanPlacedEdgeTransportStats::default()
            };
            let sampled_token = (block_index + 1 == block_width)
                .then_some(block_run.sampled_token)
                .flatten();
            let output_event = self
                .active_input_event
                .as_mut()
                .expect("temporal block requires an active input event")
                .complete_activation(
                    &activation,
                    stream_tick,
                    block_run.scheduler_turn_count_per_tick,
                    block_run.completed_stage_count_per_tick,
                    transport_stats,
                    sampled_token,
                )?;
            self.session.next_stream_tick = stream_tick
                .checked_add(1)
                .ok_or(VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
            if let Some(output_event) = output_event {
                on_output_event(output_event);
            }
        }

        let completed_input_run = self
            .active_input_event
            .as_ref()
            .is_some_and(VulkanResidentInProcessPlacedActivePromptEvent::is_complete)
            .then(|| self.complete_active_input_event())
            .transpose()?;
        Ok((block_width, completed_input_run))
    }

    pub fn run_next_activation(
        &mut self,
    ) -> Result<
        Option<VulkanResidentInProcessPlacedPromptStreamActivationRun>,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        if !self.activate_next_input_event()? {
            return Ok(None);
        }

        let activation = self
            .active_input_event
            .as_ref()
            .and_then(VulkanResidentInProcessPlacedActivePromptEvent::next_activation)
            .ok_or(VulkanResidentInProcessPlacedRuntimeError::MissingPrivateFeedback)?;
        let input_event_id = self
            .active_input_event
            .as_ref()
            .expect("active prompt event was initialized")
            .input_event
            .id
            .clone();
        let stream_tick = self.session.next_stream_tick;
        let adaptive_scalar_is_observed = self
            .feedback_execution_selector
            .as_ref()
            .is_some_and(|selector| {
                self.speculative_confidence_threshold == 0.0
                    && !selector.is_calibrated()
                    && selector.next_candidate() == VulkanFeedbackExecutionCandidate::Scalar
                    && activation.input_is_feedback
                    && activation.should_emit_public_output
                    && !activation.input_closes_loop_after_processing
            });
        let demand_retry_is_observed = self.demand_feedback_retry_deferred
            && activation.input_is_feedback
            && activation.should_emit_public_output
            && !activation.input_closes_loop_after_processing;
        let adaptive_scalar_loads_before =
            (adaptive_scalar_is_observed || demand_retry_is_observed)
                .then(|| {
                self.package
                    .compiled_resource_load_required_count()
                    .map_err(|error| {
                        VulkanResidentInProcessPlacedRuntimeError::Package(
                            VulkanResidentTokenModelPackageError::new(format!(
                                "failed to read scalar-feedback calibration residency: {error}"
                            )),
                        )
                    })
                })
            .transpose()?;
        let adaptive_scalar_started_at = adaptive_scalar_loads_before.map(|_| Instant::now());
        let tail = if activation.should_emit_public_output {
            VulkanResidentPlacedTokenTickTail::Sample
        } else if self.processor.speculative_decoder_count() > 0 {
            VulkanResidentPlacedTokenTickTail::Hidden
        } else {
            VulkanResidentPlacedTokenTickTail::None
        };
        self.processor.prepare_token_input(placed_token_input(
            activation.input_token_id,
            &self.processor.model.input_device_id,
            &self.processor.model.output_device_id,
            activation.input_is_feedback,
        ))?;
        let placed_run = self
            .processor
            .execute_prepared_token_id_stream_tick_on_bound_devices_in_process_with_transport(
                &self.devices,
                &mut self.session.transport,
                stream_tick,
                tail,
            )?;
        let sampled_token = if activation.should_emit_public_output {
            Some(
                VulkanResidentSampledToken::from(&self.processor
                    .sampler
                    .completed_run()
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)?),
            )
        } else {
            None
        };
        self.processor
            .synchronize_speculative_decoders_after_target_tick(
                &self.devices,
                activation.input_token_id,
                stream_tick,
            )?;
        let output_event = self
            .active_input_event
            .as_mut()
            .expect("active prompt event was initialized")
            .complete_activation(
                &activation,
                stream_tick,
                placed_run.scheduler_turn_count,
                placed_run.completed_stage_delta,
                &placed_run.transport_stats,
                sampled_token,
            )?;
        self.session.next_stream_tick = stream_tick
            .checked_add(1)
            .ok_or(VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;

        let completed_input_run = if self
            .active_input_event
            .as_ref()
            .is_some_and(VulkanResidentInProcessPlacedActivePromptEvent::is_complete)
        {
            Some(self.complete_active_input_event()?)
        } else {
            None
        };

        let adaptive_scalar_had_no_loads = adaptive_scalar_loads_before
            .map(|before| {
                self.package
                    .compiled_resource_load_required_count()
                    .map(|after| after == before)
                    .map_err(|error| {
                        VulkanResidentInProcessPlacedRuntimeError::Package(
                            VulkanResidentTokenModelPackageError::new(format!(
                                "failed to verify scalar-feedback calibration residency: {error}"
                            )),
                        )
                    })
            })
            .transpose()?
            .unwrap_or(false);
        self.demand_feedback_retry_deferred = demand_feedback_retry_deferred_after_scalar(
            self.demand_feedback_retry_deferred,
            activation.input_is_feedback,
            activation.should_emit_public_output,
            activation.input_closes_loop_after_processing,
            !adaptive_scalar_had_no_loads,
        );
        if output_event.is_some()
            && completed_input_run.is_none()
            && adaptive_scalar_is_observed
            && adaptive_scalar_had_no_loads
        {
            self.feedback_execution_selector
                .as_mut()
                .expect("scalar-feedback calibration requires an execution selector")
                .record_scalar_tick(
                    u64::try_from(
                        adaptive_scalar_started_at
                            .expect("scalar calibration start accompanies residency counter")
                            .elapsed()
                            .as_nanos(),
                    )
                    .unwrap_or(u64::MAX),
                );
        }

        Ok(Some(
            VulkanResidentInProcessPlacedPromptStreamActivationRun {
                input_event_id,
                stream_tick,
                input_token_id: activation.input_token_id,
                input_is_feedback: activation.input_is_feedback,
                output_event,
                completed_input_run,
            },
        ))
    }

    pub fn interrupt(
        &mut self,
        reason: impl Into<String>,
    ) -> Result<
        VulkanResidentInProcessPlacedPromptStreamControlRun,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        let reason = reason.into();
        let control_event = if let Some(active_input_event) = &mut self.active_input_event {
            active_input_event.interrupt(reason)
        } else {
            VulkanResidentStreamControlEvent {
                event_type: VulkanResidentStreamControlEventType::Interrupt,
                reason,
                cleared_private_feedback_ids: Vec::new(),
                closing_private_feedback_id: None,
                state_preserved: true,
            }
        };
        let completed_input_run = self
            .active_input_event
            .as_ref()
            .is_some_and(VulkanResidentInProcessPlacedActivePromptEvent::is_complete)
            .then(|| self.complete_active_input_event())
            .transpose()?;
        Ok(VulkanResidentInProcessPlacedPromptStreamControlRun {
            control_event,
            completed_input_run,
        })
    }

    pub fn stop_after_current(
        &mut self,
        reason: impl Into<String>,
    ) -> VulkanResidentInProcessPlacedPromptStreamControlRun {
        let reason = reason.into();
        let control_event = if let Some(active_input_event) = &mut self.active_input_event {
            active_input_event.stop_after_current(reason)
        } else {
            VulkanResidentStreamControlEvent {
                event_type: VulkanResidentStreamControlEventType::StopAfterCurrent,
                reason,
                cleared_private_feedback_ids: Vec::new(),
                closing_private_feedback_id: None,
                state_preserved: true,
            }
        };
        VulkanResidentInProcessPlacedPromptStreamControlRun {
            control_event,
            completed_input_run: None,
        }
    }

    pub fn run_next_queued_input_event(
        &mut self,
    ) -> Result<
        Option<VulkanResidentInProcessPlacedSubmittedInputRun>,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        self.run_next_queued_input_event_with_output(|_| {})
    }

    pub fn run_next_queued_input_event_with_output<F>(
        &mut self,
        mut on_output_event: F,
    ) -> Result<
        Option<VulkanResidentInProcessPlacedSubmittedInputRun>,
        VulkanResidentInProcessPlacedRuntimeError,
    >
    where
        F: FnMut(VulkanResidentTokenOutputEvent),
    {
        if self.is_idle() {
            return Ok(None);
        }
        loop {
            let (processed_external_inputs, completed_input_run) =
                self.run_temporal_external_input_block_with_output(&mut on_output_event)?;
            if let Some(completed_input_run) = completed_input_run {
                return Ok(Some(completed_input_run));
            }
            if processed_external_inputs > 0 {
                continue;
            }
            if self.run_speculative_feedback_window_limited_with_output(
                usize::MAX,
                &mut on_output_event,
            )? {
                if self
                    .active_input_event
                    .as_ref()
                    .is_some_and(VulkanResidentInProcessPlacedActivePromptEvent::is_complete)
                {
                    return Ok(Some(self.complete_active_input_event()?));
                }
                continue;
            }
            let resident_tick_limit = self
                .processor
                .resident_feedback_next_window_tick_count()
                .max(1);
            if self.adaptive_feedback_uses_resident_loop()
                && self.run_resident_feedback_window_limited_with_output(
                    resident_tick_limit,
                    &mut on_output_event,
                )?
            {
                if self
                    .active_input_event
                    .as_ref()
                    .is_some_and(VulkanResidentInProcessPlacedActivePromptEvent::is_complete)
                {
                    return Ok(Some(self.complete_active_input_event()?));
                }
                continue;
            }
            let activation = self
                .run_next_activation()?
                .ok_or(VulkanResidentInProcessPlacedRuntimeError::MissingPrivateFeedback)?;
            if let Some(output_event) = activation.output_event {
                on_output_event(output_event);
            }
            if let Some(completed_input_run) = activation.completed_input_run {
                return Ok(Some(completed_input_run));
            }
        }
    }

    fn run_speculative_feedback_window_limited_with_output<F>(
        &mut self,
        max_public_outputs: usize,
        on_output_event: &mut F,
    ) -> Result<bool, VulkanResidentInProcessPlacedRuntimeError>
    where
        F: FnMut(VulkanResidentTokenOutputEvent),
    {
        if self.speculative_draft_tokens == 0 || self.processor.speculative_decoder_count() == 0 {
            return Ok(false);
        }
        let Some(active) = self.active_input_event.as_ref() else {
            return Ok(false);
        };
        let Some(activation) = active.next_activation() else {
            return Ok(false);
        };
        if !activation.input_is_feedback
            || activation.input_closes_loop_after_processing
            || !activation.should_emit_public_output
            || active.remaining_public_outputs < 2
            || max_public_outputs < 2
        {
            return Ok(false);
        }
        let selected_candidate = if self.speculative_confidence_threshold == 0.0 {
            self.feedback_execution_selector
                .as_ref()
                .map(VulkanAdaptiveFeedbackExecutionSelector::next_candidate)
                .unwrap_or(VulkanFeedbackExecutionCandidate::Speculative {
                    draft_width: self.speculative_draft_tokens,
                })
        } else {
            VulkanFeedbackExecutionCandidate::Speculative {
                draft_width: self.speculative_draft_tokens,
            }
        };
        let selected_window_width = match selected_candidate {
            VulkanFeedbackExecutionCandidate::Scalar
            | VulkanFeedbackExecutionCandidate::Resident => return Ok(false),
            VulkanFeedbackExecutionCandidate::Speculative { draft_width } => draft_width,
        };
        let draft_token_count = selected_window_width
            .min(active.remaining_public_outputs - 1)
            .min(max_public_outputs - 1);
        let stop_token_ids = active
            .input_event
            .stop_token_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let start_stream_tick = self.session.next_stream_tick;
        let adaptive_feedback_loads_before = self
            .feedback_execution_selector
            .as_ref()
            .filter(|selector| {
                self.speculative_confidence_threshold == 0.0 && !selector.is_calibrated()
            })
            .map(|_| {
                self.package
                    .compiled_resource_load_required_count()
                    .map_err(|error| {
                        VulkanResidentInProcessPlacedRuntimeError::Package(
                            VulkanResidentTokenModelPackageError::new(format!(
                                "failed to read speculative calibration residency: {error}"
                            )),
                        )
                    })
            })
            .transpose()?;
        let cycle = self.processor.run_speculative_cycle_on_bound_devices(
            &self.devices,
            activation.input_token_id,
            start_stream_tick,
            draft_token_count,
            self.speculative_confidence_threshold,
            &stop_token_ids,
        )?;
        let stopped = cycle
            .verification
            .emitted_tokens
            .last()
            .is_some_and(|token| stop_token_ids.contains(&token.token_id));
        if self.speculative_confidence_threshold == 0.0 && !stopped {
            let adaptive_feedback_residency_changed = adaptive_feedback_loads_before
                .map(|before| {
                    self.package
                        .compiled_resource_load_required_count()
                        .map(|after| after != before)
                        .map_err(|error| {
                            VulkanResidentInProcessPlacedRuntimeError::Package(
                                VulkanResidentTokenModelPackageError::new(format!(
                                    "failed to verify speculative calibration residency: {error}"
                                )),
                            )
                        })
                })
                .transpose()?
                .unwrap_or(false);
            self.feedback_execution_selector
                .as_mut()
                .expect("configured speculative decoding has an execution selector")
                .record_speculative_cycle(
                    selected_window_width,
                    &cycle,
                    adaptive_feedback_residency_changed,
                );
        }
        self.active_input_event
            .as_mut()
            .expect("speculative feedback cycle requires an active input event")
            .speculative_decode
            .record_cycle(&cycle);
        for sampled_token in cycle.verification.emitted_tokens {
            let stream_tick = self.session.next_stream_tick;
            let output_event = {
                let active = self
                    .active_input_event
                    .as_mut()
                    .expect("speculative feedback cycle requires an active input event");
                let activation = active
                    .next_activation()
                    .ok_or(VulkanResidentInProcessPlacedRuntimeError::MissingPrivateFeedback)?;
                active.complete_activation(
                    &activation,
                    stream_tick,
                    0,
                    0,
                    &VulkanPlacedEdgeTransportStats::default(),
                    Some(sampled_token),
                )?
            };
            self.session.next_stream_tick = stream_tick
                .checked_add(1)
                .ok_or(VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
            if let Some(output_event) = output_event {
                on_output_event(output_event);
            }
        }
        Ok(true)
    }

    pub fn submit_input_event(
        &mut self,
        event: VulkanResidentTokenInputEvent,
    ) -> Result<
        VulkanResidentInProcessPlacedSubmittedInputRun,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        self.ensure_idle_for_immediate_input_event()?;
        self.enqueue_input_event(event);
        self.run_next_queued_input_event()?
            .ok_or(VulkanResidentInProcessPlacedRuntimeError::EmptyPromptEvent)
    }

    pub fn run_queued_input_events_until_idle(
        &mut self,
    ) -> Result<VulkanResidentInProcessPlacedInputQueueRun, VulkanResidentInProcessPlacedRuntimeError>
    {
        let start_stream_tick = self.next_stream_tick();
        let mut submitted_runs = Vec::new();
        while let Some(submitted_run) = self.run_next_queued_input_event()? {
            submitted_runs.push(submitted_run);
        }
        let next_stream_tick = self.next_stream_tick();
        let output_events = submitted_runs
            .iter()
            .flat_map(|submitted_run| submitted_run.output_events.iter().cloned())
            .collect::<Vec<_>>();
        let generated_token_ids = output_events
            .iter()
            .map(|event| event.token_id)
            .collect::<Vec<_>>();
        let tick_count = submitted_runs
            .iter()
            .map(|submitted_run| submitted_run.session_run.tick_count)
            .sum::<usize>();

        Ok(VulkanResidentInProcessPlacedInputQueueRun {
            start_stream_tick,
            next_stream_tick,
            submitted_runs,
            output_events,
            generated_token_ids,
            tick_count,
            pending_input_event_count: self.pending_input_event_count(),
        })
    }

    pub fn submit_input_events_until_idle<I>(
        &mut self,
        events: I,
    ) -> Result<VulkanResidentInProcessPlacedInputQueueRun, VulkanResidentInProcessPlacedRuntimeError>
    where
        I: IntoIterator<Item = VulkanResidentTokenInputEvent>,
    {
        for event in events {
            self.enqueue_input_event(event);
        }
        self.run_queued_input_events_until_idle()
    }

    fn ensure_idle_for_immediate_input_event(
        &self,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        if self.is_idle() {
            Ok(())
        } else {
            Err(VulkanResidentInProcessPlacedRuntimeError::PromptStreamBusy)
        }
    }

    fn submit_resident_feedback_window_limited(
        &mut self,
        max_feedback_ticks: usize,
    ) -> Result<
        Option<VulkanResidentInProcessPlacedPendingStreamFeedbackWindow>,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        let window_tick_count = self.processor.resident_feedback_next_window_tick_count();
        let tick_count = self
            .active_input_event
            .as_ref()
            .map(|event| event.resident_feedback_window_tick_count(window_tick_count))
            .unwrap_or(0)
            .min(max_feedback_ticks);
        if tick_count < 2 {
            return Ok(None);
        }

        let tick_delta = u64::try_from(tick_count)
            .map_err(|_| VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
        self.session
            .next_stream_tick
            .checked_add(tick_delta)
            .ok_or(VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
        let feedback_depth_delta = u32::try_from(tick_count)
            .map_err(|_| VulkanResidentInProcessPlacedRuntimeError::FeedbackDepthOverflow)?;
        self.active_input_event
            .as_ref()
            .and_then(VulkanResidentInProcessPlacedActivePromptEvent::next_activation)
            .ok_or(VulkanResidentInProcessPlacedRuntimeError::MissingPrivateFeedback)?
            .input_feedback_depth
            .checked_add(feedback_depth_delta)
            .ok_or(VulkanResidentInProcessPlacedRuntimeError::FeedbackDepthOverflow)?;

        let stop_token_ids = self
            .active_input_event
            .as_ref()
            .expect("resident feedback window requires an active input event")
            .input_event
            .stop_token_ids
            .clone();
        let input_token_id = self
            .active_input_event
            .as_ref()
            .and_then(VulkanResidentInProcessPlacedActivePromptEvent::next_activation)
            .ok_or(VulkanResidentInProcessPlacedRuntimeError::MissingPrivateFeedback)?
            .input_token_id;
        let replay_slot = self
            .processor
            .resident_feedback_loop
            .as_ref()
            .is_some_and(|feedback_loop| feedback_loop.replayable)
            .then_some(&mut self.resident_feedback_template_catalog);
        let adaptive_feedback_loads_before = self
            .feedback_execution_selector
            .as_ref()
            .filter(|selector| {
                self.speculative_confidence_threshold == 0.0
                    && !selector.is_calibrated()
                    && selector.next_candidate() == VulkanFeedbackExecutionCandidate::Resident
            })
            .map(|_| {
                self.package
                    .compiled_resource_load_required_count()
                    .map_err(|error| {
                        VulkanResidentInProcessPlacedRuntimeError::Package(
                            VulkanResidentTokenModelPackageError::new(format!(
                                "failed to read resident-feedback calibration residency: {error}"
                            )),
                        )
                    })
            })
            .transpose()?;
        let started_at = Instant::now();
        let submission = self.processor.submit_resident_feedback_window(
            &self.devices,
            self.session.next_stream_tick,
            tick_count,
            input_token_id,
            &stop_token_ids,
            replay_slot,
        )?;
        let VulkanResidentInProcessPlacedFeedbackWindowSubmission::Submitted(window) = submission
        else {
            self.demand_feedback_retry_deferred = true;
            return Ok(None);
        };
        Ok(Some(
            VulkanResidentInProcessPlacedPendingStreamFeedbackWindow {
                window,
                started_at,
                adaptive_feedback_loads_before,
            },
        ))
    }

    fn resident_feedback_window_is_complete(
        &self,
        pending: &VulkanResidentInProcessPlacedPendingStreamFeedbackWindow,
    ) -> Result<bool, VulkanResidentInProcessPlacedRuntimeError> {
        self.processor
            .resident_feedback_window_is_complete(&self.devices, &pending.window)
    }

    fn wait_resident_feedback_window_for(
        &self,
        pending: &VulkanResidentInProcessPlacedPendingStreamFeedbackWindow,
        timeout_ns: u64,
    ) -> Result<bool, VulkanResidentInProcessPlacedRuntimeError> {
        self.processor.wait_resident_feedback_window_for(
            &self.devices,
            &pending.window,
            timeout_ns,
        )
    }

    fn complete_submitted_resident_feedback_window<F>(
        &mut self,
        pending: VulkanResidentInProcessPlacedPendingStreamFeedbackWindow,
        on_output_event: &mut F,
    ) -> Result<
        VulkanResidentFeedbackControlCompletion,
        VulkanResidentInProcessPlacedRuntimeError,
    >
    where
        F: FnMut(VulkanResidentTokenOutputEvent),
    {
        let planned_tick_count = pending.window.tick_count;
        let start_stream_tick = pending.window.start_stream_tick;
        let adaptive_feedback_loads_before = pending.adaptive_feedback_loads_before;
        let processor = &self.processor;
        let devices = &self.devices;
        let active_input_event = &mut self.active_input_event;
        let session = &mut self.session;
        let mut executed_input_token_ids = Vec::with_capacity(planned_tick_count);
        let completion = processor.complete_resident_feedback_window(
            pending.window,
            | _tick_index,
              sampled_token,
              scheduler_turn_count,
              completed_stage_count,
              closes_after_device_cancel,
              transport_stats | {
                let stream_tick = session.next_stream_tick;
                let output_event = {
                    let active_input_event = active_input_event
                        .as_mut()
                        .expect("resident feedback window requires an active input event");
                    let activation = active_input_event.next_activation().ok_or(
                        VulkanResidentInProcessPlacedRuntimeError::MissingPrivateFeedback,
                    )?;
                    executed_input_token_ids.push(activation.input_token_id);
                    let output_event = active_input_event.complete_activation(
                        &activation,
                        stream_tick,
                        scheduler_turn_count,
                        completed_stage_count,
                        transport_stats,
                        activation
                            .should_emit_public_output
                            .then_some(sampled_token),
                    )?;
                    if closes_after_device_cancel {
                        active_input_event.stop_after_current("cancelled");
                    }
                    output_event
                };
                session.next_stream_tick = stream_tick
                    .checked_add(1)
                    .ok_or(VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
                if let Some(output_event) = output_event {
                    on_output_event(output_event);
                }
                Ok(())
            },
        )?;
        active_input_event
            .as_mut()
            .expect("resident feedback window requires an active input event")
            .resident_feedback
            .record_window(
                planned_tick_count,
                completion.executed_tick_count,
                completion.sampled_tick_count,
                completion.template_replayed,
            );
        for _ in completion.sampled_tick_count..completion.executed_tick_count {
            let stream_tick = session.next_stream_tick;
            let active = active_input_event
                .as_mut()
                .expect("resident feedback drain requires an active input event");
            let activation = active
                .next_activation()
                .ok_or(VulkanResidentInProcessPlacedRuntimeError::MissingPrivateFeedback)?;
            if !activation.input_closes_loop_after_processing
                || activation.should_emit_public_output
            {
                return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                    VulkanError(
                        "resident feedback control executed a drain tick without a closing private feedback input"
                            .to_string(),
                    ),
                ));
            }
            executed_input_token_ids.push(activation.input_token_id);
            active.complete_activation(
                &activation,
                stream_tick,
                processor
                    .resident_feedback_loop
                    .as_ref()
                    .expect("resident feedback loop is mounted")
                    .scheduler_turn_count_per_tick,
                processor
                    .resident_feedback_loop
                    .as_ref()
                    .expect("resident feedback loop is mounted")
                    .completed_stage_count_per_tick,
                &VulkanPlacedEdgeTransportStats::default(),
                None,
            )?;
            session.next_stream_tick = stream_tick
                .checked_add(1)
                .ok_or(VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
        }
        processor.synchronize_speculative_decoders_after_target_window(
            devices,
            &executed_input_token_ids,
            start_stream_tick,
            planned_tick_count,
        )?;
        let elapsed_time_ns =
            u64::try_from(pending.started_at.elapsed().as_nanos()).unwrap_or(u64::MAX);
        processor
            .resident_feedback_loop
            .as_ref()
            .expect("resident feedback loop is mounted")
            .window_policy
            .observe_completed_window(
                planned_tick_count,
                completion.executed_tick_count,
                elapsed_time_ns,
                completion.stop_reason != VULKAN_FEEDBACK_STOP_REASON_NONE,
            );
        if completion.stop_reason == VULKAN_FEEDBACK_STOP_REASON_NONE
            && planned_tick_count == completion.executed_tick_count
            && let Some(loads_before) = adaptive_feedback_loads_before
        {
            let adaptive_feedback_residency_changed = self
                .package
                .compiled_resource_load_required_count()
                .map(|loads_after| loads_after != loads_before)
                .map_err(|error| {
                    VulkanResidentInProcessPlacedRuntimeError::Package(
                        VulkanResidentTokenModelPackageError::new(format!(
                            "failed to verify resident-feedback calibration residency: {error}"
                        )),
                    )
                })?;
            self.feedback_execution_selector
                .as_mut()
                .expect("resident-feedback calibration requires an execution selector")
                .record_resident_window(
                    completion.sampled_tick_count,
                    elapsed_time_ns,
                    adaptive_feedback_residency_changed,
                );
        }
        Ok(completion)
    }

    fn run_resident_feedback_window_limited_with_output<F>(
        &mut self,
        max_feedback_ticks: usize,
        on_output_event: &mut F,
    ) -> Result<bool, VulkanResidentInProcessPlacedRuntimeError>
    where
        F: FnMut(VulkanResidentTokenOutputEvent),
    {
        let mut remaining_feedback_ticks = max_feedback_ticks;
        let mut ran_window = false;
        loop {
            let Some(pending) =
                self.submit_resident_feedback_window_limited(remaining_feedback_ticks)?
            else {
                break;
            };
            let tick_count = pending.window.tick_count;
            self.wait_resident_feedback_window_for(&pending, u64::MAX)?;
            let completion =
                self.complete_submitted_resident_feedback_window(pending, on_output_event)?;
            ran_window = true;
            remaining_feedback_ticks =
                remaining_feedback_ticks.saturating_sub(completion.executed_tick_count);
            if remaining_feedback_ticks == 0 {
                break;
            }
            if completion.stop_reason != VULKAN_FEEDBACK_STOP_REASON_NONE
                || completion.executed_tick_count < tick_count
            {
                break;
            }
        }
        Ok(ran_window)
    }

    fn complete_active_input_event(
        &mut self,
    ) -> Result<
        VulkanResidentInProcessPlacedSubmittedInputRun,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        let active_input_event = self
            .active_input_event
            .take()
            .expect("completed prompt event was active");
        debug_assert!(active_input_event.is_complete());
        let input_event = active_input_event.input_event.clone();
        let output_events = active_input_event.output_events.clone();
        let generated_token_ids = active_input_event.generated_token_ids.clone();
        let start_stream_tick = active_input_event.start_stream_tick;
        let event_run = active_input_event.into_event_run(
            self.package.input_device_id.clone(),
            self.package.output_device_id.clone(),
        );
        let session_run = self
            .session
            .complete_prompt_event(start_stream_tick, event_run)?;
        self.retier_compiled_resources_at_prompt_boundary()?;
        Ok(VulkanResidentInProcessPlacedSubmittedInputRun {
            input_event,
            pending_input_event_count: self.pending_input_event_count(),
            session_run,
            output_events,
            generated_token_ids,
        })
    }

    fn retier_compiled_resources_at_prompt_boundary(
        &self,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let stores = self.package.adaptive_retiering_stores();
        if stores.is_empty() {
            return Ok(());
        }
        let telemetry = self.processor.selection_telemetry_snapshot(&self.devices)?;
        for store in stores {
            let logical_device_id = store.logical_device_ids().first().ok_or_else(|| {
                selection_telemetry_error(format!(
                    "adaptive resource store {:?} has no logical execution device",
                    store.device_id()
                ))
            })?;
            let device = self.devices.get(logical_device_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                    device_id: logical_device_id.clone(),
                }
            })?;
            store
                .retier_from_selection_telemetry(device, &telemetry)
                .map_err(|error| {
                    selection_telemetry_error(format!(
                        "adaptive compiled-resource retiering failed for {:?}: {error}",
                        store.device_id()
                    ))
                })?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentInProcessPlacedQueuedInputEvent {
    pub input_event: VulkanResidentTokenInputEvent,
    pub pending_input_event_count: usize,
    pub next_stream_tick: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentInProcessPlacedSubmittedInputRun {
    pub input_event: VulkanResidentTokenInputEvent,
    pub pending_input_event_count: usize,
    pub session_run: VulkanResidentInProcessPlacedPromptSessionRun,
    pub output_events: Vec<VulkanResidentTokenOutputEvent>,
    pub generated_token_ids: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentInProcessPlacedPromptStreamActivationRun {
    pub input_event_id: String,
    pub stream_tick: u64,
    pub input_token_id: u32,
    pub input_is_feedback: bool,
    pub output_event: Option<VulkanResidentTokenOutputEvent>,
    pub completed_input_run: Option<VulkanResidentInProcessPlacedSubmittedInputRun>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentInProcessPlacedPromptStreamControlRun {
    pub control_event: VulkanResidentStreamControlEvent,
    pub completed_input_run: Option<VulkanResidentInProcessPlacedSubmittedInputRun>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentInProcessPlacedInputQueueRun {
    pub start_stream_tick: u64,
    pub next_stream_tick: u64,
    pub submitted_runs: Vec<VulkanResidentInProcessPlacedSubmittedInputRun>,
    pub output_events: Vec<VulkanResidentTokenOutputEvent>,
    pub generated_token_ids: Vec<u32>,
    pub tick_count: usize,
    pub pending_input_event_count: usize,
}

fn placed_scheduler_divergence(message: impl Into<String>) -> VulkanResidentInProcessPlacedRuntimeError {
    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
        "placed stream diverged from runtime scheduler: {}",
        message.into()
    )))
}
