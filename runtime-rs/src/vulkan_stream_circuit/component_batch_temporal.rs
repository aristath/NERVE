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
                // input barrier provides the dependency; its completion fence also
                // proves this copy completed before the binding can be reused.
                Ok(VulkanPlacedEdgeTransferRoute::DeviceLocalStaging)
            }
        }
    }
}
