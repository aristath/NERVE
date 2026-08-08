struct VulkanResidentPlacedTemporalBlockRunner {
    execution_graph: VulkanResidentPlacedComponentBatchRunner,
    input_embedding: VulkanResidentBatchedInputEmbeddingRunner,
    output_frame_copies: Vec<VulkanResidentBufferCopyBatch>,
    speculative_source_tap_frame_copies: Vec<Vec<VulkanResidentBufferCopyBatch>>,
    parallel_speculative_state_ingestions:
        Vec<VulkanResidentParallelSpeculativeStateIngestion>,
    speculative_target_output: Option<VulkanResidentBatchedOutputProjectionRunner>,
    pipeline: Vec<usize>,
}

impl VulkanResidentPlacedTemporalBlockRunner {
    fn publish_speculative_source_tap_frame(
        &self,
        frame_index: usize,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        if self.speculative_source_tap_frame_copies.is_empty() {
            return Ok(());
        }
        let copies = self
            .speculative_source_tap_frame_copies
            .get(frame_index)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    format!(
                        "speculative source-tap frame {frame_index} exceeds retained batch capacity {}",
                        self.speculative_source_tap_frame_copies.len()
                    ),
                ))
            })?;
        for copy in copies {
            copy.run()
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        }
        Ok(())
    }
}

struct VulkanResidentTemporalBlockRun {
    sampled_token: Option<VulkanResidentSampledToken>,
    scheduler_turn_count_per_tick: usize,
    completed_stage_count_per_tick: usize,
    transport_stats: VulkanPlacedEdgeTransportStats,
}

fn deferred_pipeline_signals_completion(
    pipeline_position: usize,
    pipeline_len: usize,
) -> Result<bool, VulkanError> {
    if pipeline_len == 0 {
        return Err(VulkanError(
            "deferred component pipeline must not be empty".to_string(),
        ));
    }
    if pipeline_position >= pipeline_len {
        return Err(VulkanError(format!(
            "deferred component pipeline position {pipeline_position} exceeds length {pipeline_len}"
        )));
    }
    Ok(pipeline_position + 1 == pipeline_len)
}

enum VulkanComponentBatchEdgeTransferBinding {
    Resident(Box<VulkanResidentBufferCopy>),
    HostStaging {
        source: Arc<VulkanResidentBuffer>,
        destination: Arc<VulkanResidentBuffer>,
        byte_len: usize,
    },
    DeviceLocalStaging {
        source_device: Rc<VulkanComputeDevice>,
        destination_device: Rc<VulkanComputeDevice>,
        source_copy: Box<VulkanResidentBufferCopy>,
        destination_copy: Box<VulkanResidentBufferCopy>,
        source_signal: VulkanTimelineSemaphore,
        destination_wait: VulkanTimelineSemaphore,
        next_value: Cell<u64>,
        _source_staging: Arc<VulkanResidentBuffer>,
        _destination_staging: Arc<VulkanResidentBuffer>,
    },
}

struct VulkanComponentBatchEdgeTransfer {
    source_device_index: usize,
    destination_device_index: usize,
    edge_index: usize,
    binding: VulkanComponentBatchEdgeTransferBinding,
}

impl VulkanComponentBatchEdgeTransfer {
    fn supports_deferred_completion(&self) -> bool {
        matches!(
            self.binding,
            VulkanComponentBatchEdgeTransferBinding::DeviceLocalStaging { .. }
        )
    }

    fn run(
        &self,
    ) -> Result<VulkanPlacedEdgeTransferRoute, VulkanResidentInProcessPlacedRuntimeError> {
        match &self.binding {
            VulkanComponentBatchEdgeTransferBinding::Resident(copy) => copy
                .run(copy.byte_len())
                .map(|()| VulkanPlacedEdgeTransferRoute::DeviceLocalCopy)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop),
            VulkanComponentBatchEdgeTransferBinding::HostStaging {
                source,
                destination,
                byte_len,
            } => {
                let bytes = source
                    .read_bytes(*byte_len)
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                destination
                    .write_bytes(&bytes)
                    .map(|()| VulkanPlacedEdgeTransferRoute::HostStaging)
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
            }
            VulkanComponentBatchEdgeTransferBinding::DeviceLocalStaging {
                source_device,
                destination_device,
                source_copy,
                destination_copy,
                source_signal,
                destination_wait,
                next_value,
                ..
            } => {
                let value = next_value.get();
                next_value.set(value.checked_add(1).ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "component batch edge timeline exhausted its values".to_string(),
                    ))
                })?);
                source_device
                    .submit_resident_buffer_copy_with_timeline_semaphores(
                        source_copy,
                        &[],
                        &[VulkanTimelineSemaphorePoint::new(source_signal, value)],
                    )
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                destination_device
                    .submit_resident_buffer_copy_with_timeline_semaphores(
                        destination_copy,
                        &[VulkanTimelineSemaphorePoint::new(destination_wait, value)],
                        &[],
                    )
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                // The destination circuit is submitted to the same queue immediately
                // after this copy. Queue order plus the circuit's transfer-to-compute
                // input barrier provides the dependency. The retained copy is
                // recorded for simultaneous replay, so the next stream turn
                // does not need a host completion round-trip to reuse it.
                Ok(VulkanPlacedEdgeTransferRoute::DeviceLocalStaging)
            }
        }
    }

    fn enqueue_deferred<'a>(
        &'a self,
        submission_batch: &VulkanResidentQueueSubmissionBatch<'a>,
    ) -> Result<VulkanPlacedEdgeTransferRoute, VulkanResidentInProcessPlacedRuntimeError> {
        let VulkanComponentBatchEdgeTransferBinding::DeviceLocalStaging {
            source_device,
            destination_device,
            source_copy,
            destination_copy,
            source_signal,
            destination_wait,
            next_value,
            ..
        } = &self.binding
        else {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(
                    "deferred component-batch edge requires device-local staging".to_string(),
                ),
            ));
        };
        let value = next_value.get();
        next_value.set(value.checked_add(1).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "component batch edge timeline exhausted its values".to_string(),
            ))
        })?);
        submission_batch
            .enqueue_resident_buffer_copy(
                source_device,
                source_copy,
                &[],
                &[VulkanTimelineSemaphorePoint::new(source_signal, value)],
            )
            .and_then(|()| {
                submission_batch.enqueue_resident_buffer_copy(
                    destination_device,
                    destination_copy,
                    &[VulkanTimelineSemaphorePoint::new(destination_wait, value)],
                    &[],
                )
            })
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        Ok(VulkanPlacedEdgeTransferRoute::DeviceLocalStaging)
    }
}
