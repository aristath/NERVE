pub struct VulkanResidentInProcessPlacedStreamProcessor {
    distributed_dispatch_runners: VulkanDistributedDispatchRunners,
    _distributed_activation_buffers: VulkanDistributedActivationBuffers,
    edge_synchronizations: VulkanPlacedEdgeTimelineSynchronizations,
    model: Arc<VulkanResidentInProcessPlacedModelPackage>,
    input_transducer: VulkanResidentInputEmbeddingTransducerRunner,
    output_transducer: VulkanResidentOutputTransducerRunner,
    sampler: VulkanResidentSamplerRunner,
    output_synchronization: VulkanResidentPlacedOutputTimelineSynchronization,
    resident_feedback_loop: Option<VulkanResidentInProcessPlacedFeedbackLoop>,
    activation_schedule: VulkanMountedPlacedResidentInProcessSchedule,
    device_slices: Vec<VulkanResidentInProcessPlacedStreamProcessorDevice>,
    execution_quantum_calibrators:
        BTreeMap<String, Rc<RefCell<RuntimeExecutionQuantumCalibrator>>>,
    speculative_decoders: Vec<VulkanResidentSpeculativeDecoderProcessor>,
    verification_state_transactions: RefCell<Option<Vec<VulkanResidentStateTransactionBank>>>,
    temporal_block_executions:
        RefCell<BTreeMap<(usize, bool), VulkanResidentPlacedTemporalBlockRunner>>,
}

impl VulkanResidentInProcessPlacedStreamProcessor {
    fn resident_state_snapshot_digest(
        &self,
    ) -> Result<[u8; 32], VulkanResidentInProcessPlacedRuntimeError> {
        use sha2::{Digest, Sha256};

        let mut digest = Sha256::new();
        for slice in &self.device_slices {
            update_digest_frame(&mut digest, slice.device_id.as_bytes());
            for state in &slice.mounted.buffers.state_buffers {
                update_digest_frame(&mut digest, state.component_id.as_bytes());
                update_digest_frame(&mut digest, state.state_id.as_bytes());
                update_digest_frame(&mut digest, state.state_type.as_bytes());
                update_digest_frame(
                    &mut digest,
                    &state.byte_capacity.to_le_bytes(),
                );
                let bytes = state
                    .buffer
                    .read_bytes(state.byte_capacity)
                    .map_err(
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop,
                    )?;
                update_digest_frame(&mut digest, &bytes);
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
        self.device_slices
            .iter()
            .try_fold(0usize, |total, slice| {
                let bytes = slice
                    .mounted
                    .buffers
                    .zero_state_buffers()
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                total.checked_add(bytes).ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "reset transient state byte count overflowed".to_string(),
                    ))
                })
            })
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

struct VulkanResidentSpeculativeDecoderProcessor {
    id: String,
    device_id: String,
    mounted: VulkanMountedPlacedStreamCircuit,
    execution_plan: VulkanMountedPlacedResidentStreamTickExecutionPlan,
    input_transducer: VulkanResidentInputEmbeddingTransducerRunner,
    output_transducer: VulkanResidentOutputTransducerRunner,
    sampler: VulkanResidentSamplerRunner,
    draft_sequence: VulkanResidentKernelSequence,
    state_sequence: VulkanResidentKernelSequence,
    catch_up_sequence: VulkanResidentKernelSequence,
    hidden_input_signal_id: String,
    pending_hidden_input_copy: VulkanResidentBufferCopy,
    update_pending_hidden_copy: VulkanResidentBufferCopy,
    pending_target_hidden: VulkanResidentBuffer,
    catch_up_controls: VulkanResidentBuffer,
    catch_up_controls_initial_copy: VulkanResidentBufferCopy,
    state_transaction: VulkanResidentStateTransactionBank,
}
