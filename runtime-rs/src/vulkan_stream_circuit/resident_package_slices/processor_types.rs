pub struct VulkanResidentInProcessPlacedStreamProcessor {
    stream_memory_admission: Arc<VulkanMemoryAdmission>,
    _parallel_speculative_feedback_memory_admission: Option<Arc<VulkanMemoryAdmission>>,
    distributed_dispatch_runners: VulkanDistributedDispatchRunners,
    distributed_dynamic_resource_buffers:
        BTreeMap<String, Arc<VulkanDynamicResourceBuffers>>,
    selected_resource_adaptation:
        Option<VulkanRuntimeSelectedResourceAdaptationState>,
    selected_resource_cache_registration:
        Option<VulkanSelectedResourceCacheRegistration>,
    _distributed_activation_buffers: VulkanDistributedActivationBuffers,
    edge_synchronizations: VulkanPlacedEdgeTimelineSynchronizations,
    model: Arc<VulkanResidentInProcessPlacedModelPackage>,
    input_transducer: VulkanResidentInputEmbeddingTransducerRunner,
    output_transducer: VulkanResidentOutputTransducerRunner,
    sampler: VulkanResidentSamplerRunner,
    output_synchronization: VulkanResidentPlacedOutputTimelineSynchronization,
    resident_feedback_loop: Option<VulkanResidentInProcessPlacedFeedbackLoop>,
    speculative_target_frame_history: Option<VulkanResidentSpeculativeTargetFrameHistory>,
    parallel_speculative_feedback_state:
        Option<VulkanResidentParallelSpeculativeFeedbackState>,
    activation_schedule: VulkanMountedPlacedResidentInProcessSchedule,
    device_slices: Vec<VulkanResidentInProcessPlacedStreamProcessorDevice>,
    execution_quantum_calibrators:
        BTreeMap<String, Rc<RefCell<RuntimeExecutionQuantumCalibrator>>>,
    speculative_decoders: Vec<VulkanResidentSpeculativeDecoderProcessor>,
    verification_state_transactions: RefCell<Option<Vec<VulkanResidentStateTransactionBank>>>,
    // One runner per state-snapshot mode. The runner is mounted at the
    // pipeline's canonical lane capacity and executes narrower active widths
    // through its runtime control buffers. Keying this cache by every observed
    // prompt width would retain a second copy of all signal/state buffers for
    // each remainder width encountered over a long conversation.
    temporal_block_executions:
        RefCell<BTreeMap<bool, VulkanResidentPlacedTemporalBlockRunner>>,
}

impl Drop for VulkanResidentInProcessPlacedStreamProcessor {
    fn drop(&mut self) {
        for decoder in &self.speculative_decoders {
            decoder.discard_catch_up_batch();
        }
    }
}

impl VulkanResidentInProcessPlacedStreamProcessor {
    fn enter_transaction_checkpoint_memory(&self) -> VulkanMemoryAdmissionScope {
        self.stream_memory_admission.enter_transaction_checkpoint()
    }

    fn resident_state_snapshot_digest(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        transient_pages: &VulkanResidentTransientStatePageTable,
    ) -> Result<[u8; 32], VulkanResidentInProcessPlacedRuntimeError> {
        use sha2::{Digest, Sha256};

        let mut digest = Sha256::new();
        for slice in &self.device_slices {
            update_digest_frame(&mut digest, slice.device_id.as_bytes());
            let device = devices.get(&slice.device_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                    device_id: slice.device_id.clone(),
                }
            })?;
            let state_snapshots = slice
                .mounted
                .buffers
                .state_buffers
                .iter()
                .map(|state| {
                    let key = TransientStateKey::new(
                        state.component_id.clone(),
                        state.state_id.clone(),
                    );
                    let page_indices =
                        transient_pages.resident_page_indices(&key);
                    let read_byte_count = page_indices
                        .iter()
                        .copied()
                        .try_fold(
                            state.layout.dynamic_data_offset,
                            |maximum, page_index| {
                                state
                                    .layout
                                    .dynamic_physical_page_offset(page_index)
                                    .and_then(|offset| {
                                        offset
                                            .checked_add(
                                                state
                                                    .layout
                                                    .dynamic_page_byte_capacity,
                                            )
                                            .ok_or_else(|| {
                                                VulkanError(
                                                    "resident state digest range overflowed"
                                                        .to_string(),
                                                )
                                            })
                                    })
                                    .map(|end| maximum.max(end))
                            },
                        )?;
                    Ok((page_indices, read_byte_count))
                })
                .collect::<Result<Vec<_>, VulkanError>>()
                .map_err(
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop,
                )?;
            let read_ranges = slice
                .mounted
                .buffers
                .state_buffers
                .iter()
                .zip(&state_snapshots)
                .map(|(state, (_, read_byte_count))| {
                    VulkanResidentBufferReadRange::new(
                        &state.buffer,
                        0,
                        *read_byte_count,
                    )
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop,
                )?;
            let readback = device
                .read_resident_buffer_ranges(&read_ranges)
                .map_err(
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop,
                )?;
            for (state_index, (state, (page_indices, _))) in slice
                .mounted
                .buffers
                .state_buffers
                .iter()
                .zip(&state_snapshots)
                .enumerate()
            {
                update_digest_frame(&mut digest, state.component_id.as_bytes());
                update_digest_frame(&mut digest, state.state_id.as_bytes());
                update_digest_frame(&mut digest, state.state_type.as_bytes());
                update_digest_frame(
                    &mut digest,
                    &state.byte_capacity.to_le_bytes(),
                );
                let snapshot = readback
                    .range_bytes(state_index)
                    .map_err(
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop,
                    )?;
                update_digest_frame(
                    &mut digest,
                    &snapshot[..state.layout.dynamic_data_offset],
                );
                for page_index in page_indices {
                    update_digest_frame(
                        &mut digest,
                        &page_index.to_le_bytes(),
                    );
                    let offset = state
                        .layout
                        .dynamic_physical_page_offset(*page_index)
                        .map_err(
                            VulkanResidentInProcessPlacedRuntimeError::BackendLoop,
                        )?;
                    update_digest_frame(
                        &mut digest,
                        &snapshot[offset
                            ..offset
                                + state.layout.dynamic_page_byte_capacity],
                    );
                }
            }
        }
        Ok(digest.finalize().into())
    }

    fn mounted_state_buffer_with_device_id(
        &self,
        key: &TransientStateKey,
    ) -> Option<(&str, &VulkanStreamStateBufferAllocation)> {
        self.device_slices.iter().find_map(|slice| {
            slice
                .mounted
                .buffers
                .state_buffer(&key.node_instance_id, &key.state_id)
                .map(|state| (slice.device_id.as_str(), state))
        })
    }

    fn mounted_state_buffer(
        &self,
        key: &TransientStateKey,
    ) -> Option<&VulkanStreamStateBufferAllocation> {
        self.mounted_state_buffer_with_device_id(key)
            .map(|(_, state)| state)
    }

    fn reset_transient_state_buffers(
        &self,
    ) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError> {
        let target_bytes = self
            .device_slices
            .iter()
            .try_fold(0usize, |total, slice| {
                let state_bytes = slice
                    .mounted
                    .buffers
                    .zero_state_buffers()
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                let telemetry_bytes = slice
                    .mounted
                    .buffers
                    .zero_selection_telemetry_buffers()
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                total
                    .checked_add(state_bytes)
                    .and_then(|total| total.checked_add(telemetry_bytes))
                    .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "reset transient state byte count overflowed".to_string(),
                    ))
                    })
            })?;
        self.speculative_decoders
            .iter()
            .try_fold(target_bytes, |total, decoder| {
                let reset_bytes = decoder.reset_transient_state_buffers()?;
                total.checked_add(reset_bytes).ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "reset transient state byte count overflowed".to_string(),
                    ))
                })
            })
    }

    fn initialize_transient_state_buffers(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
    ) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError> {
        let target_bytes = self.device_slices
            .iter()
            .try_fold(0usize, |total, slice| {
                let device = devices.get(&slice.device_id).ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                        device_id: slice.device_id.clone(),
                    }
                })?;
                let bytes = slice
                    .mounted
                    .buffers
                    .initialize_state_buffers(device)
                    .map_err(
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop,
                    )?;
                total.checked_add(bytes).ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                        VulkanError(
                            "state initialization byte count overflowed"
                                .to_string(),
                        ),
                    )
                })
            })?;
        self.speculative_decoders
            .iter()
            .try_fold(target_bytes, |total, decoder| {
                let device = devices.get(&decoder.device_id).ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                        device_id: decoder.device_id.clone(),
                    }
                })?;
                let bytes = decoder.initialize_transient_state_buffers(device)?;
                total.checked_add(bytes).ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "speculative decoder state initialization byte count overflowed"
                            .to_string(),
                    ))
                })
            })
    }

    fn reset_for_new_session(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        random_seed: u32,
    ) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError> {
        let initialized = self.initialize_transient_state_buffers(devices)?;
        self.sampler
            .reset_session_state(random_seed)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)?;
        let initialized = self.speculative_decoders.iter().try_fold(
            initialized,
            |total, decoder| {
                let bytes = decoder.reset_session_state(random_seed)?;
                total.checked_add(bytes).ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "new-session reset byte count overflowed".to_string(),
                    ))
                })
            },
        )?;
        self.verification_state_transactions.borrow_mut().take();
        Ok(initialized)
    }

    fn restore_initial_transaction_state(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
    ) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError> {
        let initialized = self.initialize_transient_state_buffers(devices)?;
        self.sampler
            .reset_token_state()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)?;
        let initialized = self.speculative_decoders.iter().try_fold(
            initialized,
            |total, decoder| {
                let bytes = decoder.restore_initial_session_state()?;
                total.checked_add(bytes).ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "initial-session restore byte count overflowed".to_string(),
                    ))
                })
            },
        )?;
        self.verification_state_transactions.borrow_mut().take();
        Ok(initialized)
    }

    fn set_random_seed(
        &self,
        random_seed: u32,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        self.sampler
            .set_random_seed(random_seed)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)?;
        for decoder in &self.speculative_decoders {
            decoder.set_random_seed(random_seed)?;
        }
        Ok(())
    }
}

fn update_digest_frame(digest: &mut sha2::Sha256, bytes: &[u8]) {
    use sha2::Digest;

    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
}

fn create_placed_state_transactions<'a, F>(
    devices: &[VulkanResidentInProcessPlacedStreamProcessorDevice],
    transaction_width: usize,
    device_for: &F,
) -> Result<Vec<VulkanResidentStateTransactionBank>, VulkanResidentInProcessPlacedRuntimeError>
where
    F: Fn(&str) -> Result<&'a VulkanComputeDevice, VulkanResidentInProcessPlacedRuntimeError>,
{
    devices
        .iter()
        .map(|slice| {
            VulkanResidentStateTransactionBank::new_transactional(
                device_for(&slice.device_id)?,
                &slice.mounted.buffers,
                transaction_width,
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
        })
        .collect()
}

struct VulkanResidentAutoregressiveSpeculativeDecoderProcessor {
    id: String,
    device_id: String,
    device_slice: VulkanResidentInProcessPlacedStreamProcessorDevice,
    input_transducer: VulkanResidentInputEmbeddingTransducerRunner,
    input_embedding_batch_spirv_words: Vec<u32>,
    input_embedding_batch_control: VulkanResidentComponentBatchControlSpec,
    input_embedding_spec: VulkanResidentInputEmbeddingTransducerSpec,
    input_embedding_weight: VulkanPermanentParameterBufferAllocation,
    output_transducer: VulkanResidentOutputTransducerRunner,
    sampler: VulkanResidentSamplerRunner,
    draft_sequence: VulkanResidentKernelSequence,
    state_sequence: VulkanResidentKernelSequence,
    catch_up_sequence: VulkanResidentKernelSequence,
    hidden_input_signal_id: String,
    pending_hidden_input_copies: [VulkanResidentBufferCopy; 2],
    update_pending_hidden_copies: [VulkanResidentBufferCopy; 2],
    pending_target_hiddens: [VulkanResidentBuffer; 2],
    active_pending_target_hidden: Cell<usize>,
    catch_up_lane_capacity: usize,
    catch_up_batch: RefCell<Option<VulkanResidentSpeculativeCatchUpBatch>>,
    catch_up_controls: VulkanResidentBuffer,
    catch_up_controls_initial_copy: VulkanResidentBufferCopy,
    state_transaction: VulkanResidentStateTransactionBank,
}

struct VulkanResidentParallelBlockSpeculativeDecoderProcessor {
    device_slice: VulkanResidentInProcessPlacedStreamProcessorDevice,
    input_phase: VulkanMountedPlacedResidentExecutionGraphRunner,
    processor_phase: VulkanResidentPlacedComponentBatchRunner,
    state_processor_phase: VulkanResidentPlacedComponentBatchRunner,
    output_phase: VulkanMountedPlacedResidentExecutionGraphRunner,
    source_taps: Vec<VulkanSpeculativeSourceTapTransfer>,
    batch_source_taps: Vec<VulkanParallelSpeculativeSourceTapBatchBinding>,
    state_ingestion_node_ids_by_component: BTreeMap<String, BTreeSet<String>>,
    ingress_copies: VulkanResidentBufferCopyBatch,
    state_ingress_copies: VulkanResidentBufferCopyBatch,
    egress_copies: VulkanResidentBufferCopyBatch,
    output_readback: VulkanResidentBufferReadbackBinding,
    anchor_input_signal_id: String,
    minimum_draft_token_count: usize,
    block_width: usize,
    source_context_tick_offset: i64,
    state_transaction: VulkanResidentStateTransactionBank,
}

struct VulkanResidentSpeculativeDecoderProcessor {
    id: String,
    device_id: String,
    execution: VulkanResidentSpeculativeDecoderExecutionProcessor,
}

enum VulkanResidentSpeculativeDecoderExecutionProcessor {
    Autoregressive(VulkanResidentAutoregressiveSpeculativeDecoderProcessor),
    ParallelBlock(VulkanResidentParallelBlockSpeculativeDecoderProcessor),
}
