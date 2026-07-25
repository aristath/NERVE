struct VulkanResidentInProcessPlacedPromptSessionCheckpoint {
    next_stream_tick: u64,
    completed_prompt_event_count: usize,
    generated_token_count: usize,
    output_token_count: usize,
}

struct VulkanResidentInProcessPlacedPromptEngineStreamTransaction {
    stream_id: String,
    scheduler_state: RuntimeStreamStateCheckpoint,
    resident_state: VulkanResidentPlacedPrefixStateEntry,
    history: VulkanResidentInProcessPlacedPromptEngineStreamHistory,
    session: VulkanResidentInProcessPlacedPromptSessionCheckpoint,
}

impl VulkanResidentInProcessPlacedPromptEngine {
    fn checkpoint_stream_transaction(
        &mut self,
        stream_id: &str,
    ) -> Result<
        VulkanResidentInProcessPlacedPromptEngineStreamTransaction,
        VulkanResidentInProcessPlacedPromptEngineError,
    > {
        if self.active_transaction_stream_ids.contains(stream_id) {
            return Err(VulkanResidentInProcessPlacedPromptEngineError::Stream(
                placed_scheduler_divergence(format!(
                    "stream {stream_id:?} already has an active transaction"
                )),
            ));
        }
        let stream = self.streams.get(stream_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedPromptEngineError::UnknownStream {
                stream_id: stream_id.to_string(),
            }
        })?;
        if !stream.is_idle() || stream.pending_scheduler_activation.is_some() {
            return Err(VulkanResidentInProcessPlacedPromptEngineError::Stream(
                placed_scheduler_divergence(format!(
                    "cannot checkpoint non-idle placed stream {stream_id:?}"
                )),
            ));
        }
        let history = self
            .stream_histories
            .get(stream_id)
            .cloned()
            .ok_or_else(|| VulkanResidentInProcessPlacedPromptEngineError::UnknownStream {
                stream_id: stream_id.to_string(),
            })?;
        if !history.pending_feedback_token_ids.is_empty() {
            return Err(VulkanResidentInProcessPlacedPromptEngineError::Stream(
                placed_scheduler_divergence(format!(
                    "cannot checkpoint stream {stream_id:?} with uncommitted feedback"
                )),
            ));
        }
        if history.committed_state_token_ids.is_empty() {
            return Err(VulkanResidentInProcessPlacedPromptEngineError::Stream(
                placed_scheduler_divergence(format!(
                    "cannot checkpoint stream {stream_id:?} before its first state token"
                )),
            ));
        }
        let state = self
            .runtime_scheduler
            .stream_transient_state_snapshot(stream_id)?;
        let key = RuntimePrefixStateCacheKey::from_token_prefix(
            stream.package().stream_execution_class_id(),
            stream.package().runtime_execution_identity.clone(),
            &history.committed_state_token_ids,
            &prefix_cache_runtime_modifier_bytes(stream)?,
            state.entries.iter().map(|entry| entry.key.clone()),
        )
        .map_err(RuntimeStreamSchedulerError::from)?;
        let resident_state =
            self.resident_prefix_state_cache
                .prepare_capture(key, stream, &state)?;
        let session = VulkanResidentInProcessPlacedPromptSessionCheckpoint {
            next_stream_tick: stream.session.next_stream_tick,
            completed_prompt_event_count: stream.session.completed_prompt_event_count,
            generated_token_count: stream.session.generated_token_count,
            output_token_count: stream.session.output_token_count,
        };
        let scheduler_state = self
            .runtime_scheduler
            .checkpoint_stream_state(stream_id)?;
        self.active_transaction_stream_ids
            .insert(stream_id.to_string());
        Ok(
            VulkanResidentInProcessPlacedPromptEngineStreamTransaction {
                stream_id: stream_id.to_string(),
                scheduler_state,
                resident_state,
                history,
                session,
            },
        )
    }

    fn restore_stream_transaction(
        &mut self,
        transaction: VulkanResidentInProcessPlacedPromptEngineStreamTransaction,
    ) -> Result<(), VulkanResidentInProcessPlacedPromptEngineError> {
        let stream_id = transaction.stream_id;
        if !self.active_transaction_stream_ids.contains(&stream_id) {
            return Err(VulkanResidentInProcessPlacedPromptEngineError::Stream(
                placed_scheduler_divergence(format!(
                    "stream {stream_id:?} has no active transaction"
                )),
            ));
        }
        self.runtime_scheduler
            .restore_stream_state_checkpoint(transaction.scheduler_state)?;
        let stream = self.streams.get_mut(&stream_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedPromptEngineError::UnknownStream {
                stream_id: stream_id.clone(),
            }
        })?;
        VulkanResidentPlacedPrefixStateCache::restore_entry(
            &transaction.resident_state,
            stream,
        )?;
        stream.session.next_stream_tick = transaction.session.next_stream_tick;
        stream.session.completed_prompt_event_count =
            transaction.session.completed_prompt_event_count;
        stream.session.generated_token_count = transaction.session.generated_token_count;
        stream.session.output_token_count = transaction.session.output_token_count;
        self.stream_histories
            .insert(stream_id.clone(), transaction.history);
        self.active_transaction_stream_ids.remove(&stream_id);
        Ok(())
    }

    pub fn submit_input_event_transactionally_until_idle_with_output<F>(
        &mut self,
        stream_id: &str,
        event: VulkanResidentTokenInputEvent,
        on_output_event: F,
    ) -> Result<
        VulkanResidentInProcessPlacedPromptEngineSubmittedInputRun,
        VulkanResidentInProcessPlacedPromptEngineError,
    >
    where
        F: FnMut(VulkanResidentTokenRuntimeSchedulerOutputEvent),
    {
        let transaction = self.checkpoint_stream_transaction(stream_id)?;
        let run =
            self.submit_input_event_until_idle_with_output(stream_id, event, on_output_event);
        let restore = self.restore_stream_transaction(transaction);
        match (run, restore) {
            (Ok(run), Ok(())) => Ok(run),
            (Err(run_error), Ok(())) => Err(run_error),
            (Ok(_), Err(restore_error)) => Err(restore_error),
            (Err(run_error), Err(restore_error)) => Err(
                VulkanResidentInProcessPlacedPromptEngineError::Stream(
                    placed_scheduler_divergence(format!(
                        "transactional stream run failed ({run_error}) and state restoration also failed ({restore_error})"
                    )),
                ),
            ),
        }
    }
}
