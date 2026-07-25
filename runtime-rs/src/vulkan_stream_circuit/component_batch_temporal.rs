struct VulkanResidentPlacedTemporalBlockRunner {
    execution_graph: VulkanResidentPlacedComponentBatchRunner,
    input_embedding: VulkanResidentBatchedInputEmbeddingRunner,
    input_frame_copies: Vec<VulkanResidentBufferCopyBatch>,
    output_frame_copies: Vec<VulkanResidentBufferCopyBatch>,
    pipeline: Vec<usize>,
}

struct VulkanResidentTemporalBlockRun {
    sampled_token_id: Option<u32>,
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
                destination_device
                    .wait_resident_buffer_copy(destination_copy)
                    .map(|()| VulkanPlacedEdgeTransferRoute::DeviceLocalStaging)
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
            }
        }
    }
}
