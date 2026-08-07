struct VulkanResidentInProcessPlacedPromptSessionCheckpoint {
    next_stream_tick: u64,
    completed_prompt_event_count: usize,
    generated_token_count: usize,
    output_token_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VulkanResidentOutputControl {
    Continue,
    Abort,
}

pub(crate) struct VulkanResidentInProcessPlacedPromptEngineStreamTransaction {
    stream_id: String,
    depth: usize,
    scheduler_state: RuntimeStreamStateCheckpoint,
    resident_state: Option<VulkanResidentPlacedPrefixStateEntry>,
    resident_page_checkpoint: VulkanResidentTransientStatePageTable,
    page_cow: bool,
    history: VulkanResidentInProcessPlacedPromptEngineStreamHistory,
    session: VulkanResidentInProcessPlacedPromptSessionCheckpoint,
}

impl VulkanResidentInProcessPlacedPromptEngine {
    pub(crate) fn begin_stream_transaction(
        &mut self,
        stream_id: &str,
    ) -> Result<
        VulkanResidentInProcessPlacedPromptEngineStreamTransaction,
        VulkanResidentInProcessPlacedPromptEngineError,
    > {
        let _state_commit = runtime_critical_path_span(RuntimeCriticalPathPhase::StateCommit);
        let depth = self
            .active_transaction_depths
            .get(stream_id)
            .copied()
            .unwrap_or_default()
            .checked_add(1)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedPromptEngineError::Stream(
                    placed_scheduler_divergence(format!(
                        "stream {stream_id:?} transaction depth overflowed",
                    )),
                )
            })?;
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
        let state = self
            .runtime_scheduler
            .stream_transient_state_snapshot(stream_id)?;
        let resident_page_checkpoint = stream.transient_state_pages.clone();
        let page_cow = !history.committed_state_token_ids.is_empty()
            && stream.can_checkpoint_transaction_with_page_cow(&state)?;
        let resident_state = if history.committed_state_token_ids.is_empty() {
            None
        } else {
            let key = RuntimePrefixStateCacheKey::from_token_prefix(
                stream.package().stream_execution_class_id(),
                stream.package().runtime_execution_identity.clone(),
                &history.committed_state_token_ids,
                &prefix_cache_runtime_modifier_bytes(stream)?,
                state.entries.iter().map(|entry| entry.key.clone()),
            )
            .map_err(RuntimeStreamSchedulerError::from)?;
            Some(if page_cow {
                self.resident_prefix_state_cache
                    .prepare_transaction_capture(key, stream, &state)?
            } else {
                self.resident_prefix_state_cache
                    .prepare_capture(key, stream, &state)?
            })
        };
        let session = VulkanResidentInProcessPlacedPromptSessionCheckpoint {
            next_stream_tick: stream.session.next_stream_tick,
            completed_prompt_event_count: stream.session.completed_prompt_event_count,
            generated_token_count: stream.session.generated_token_count,
            output_token_count: stream.session.output_token_count,
        };
        let scheduler_state = self
            .runtime_scheduler
            .checkpoint_stream_state(stream_id)?;
        if page_cow {
            self.streams
                .get_mut(stream_id)
                .expect("transaction stream was validated")
                .begin_transaction_page_cow()?;
        }
        self.active_transaction_depths
            .insert(stream_id.to_string(), depth);
        Ok(
            VulkanResidentInProcessPlacedPromptEngineStreamTransaction {
                stream_id: stream_id.to_string(),
                depth,
                scheduler_state,
                resident_state,
                resident_page_checkpoint,
                page_cow,
                history,
                session,
            },
        )
    }

    fn validate_current_stream_transaction(
        &self,
        transaction: &VulkanResidentInProcessPlacedPromptEngineStreamTransaction,
    ) -> Result<(), VulkanResidentInProcessPlacedPromptEngineError> {
        let current_depth = self
            .active_transaction_depths
            .get(&transaction.stream_id)
            .copied();
        if current_depth != Some(transaction.depth) {
            return Err(VulkanResidentInProcessPlacedPromptEngineError::Stream(
                placed_scheduler_divergence(format!(
                    "stream {:?} transaction depth is {:?}, expected {}",
                    transaction.stream_id, current_depth, transaction.depth,
                )),
            ));
        }
        Ok(())
    }

    fn close_stream_transaction_depth(
        &mut self,
        stream_id: &str,
        depth: usize,
    ) {
        if depth == 1 {
            self.active_transaction_depths.remove(stream_id);
        } else {
            self.active_transaction_depths
                .insert(stream_id.to_string(), depth - 1);
        }
    }

    pub(crate) fn restore_stream_transaction(
        &mut self,
        transaction: VulkanResidentInProcessPlacedPromptEngineStreamTransaction,
    ) -> Result<(), VulkanResidentInProcessPlacedPromptEngineError> {
        let _state_commit = runtime_critical_path_span(RuntimeCriticalPathPhase::StateCommit);
        self.validate_current_stream_transaction(&transaction)?;
        let stream_id = transaction.stream_id.clone();
        let depth = transaction.depth;
        let restore = (|| {
            self.streams
                .get_mut(&stream_id)
                .ok_or_else(|| VulkanResidentInProcessPlacedPromptEngineError::UnknownStream {
                    stream_id: stream_id.clone(),
                })?
                .quiesce_and_discard_transaction_work()?;
            self.runtime_scheduler
                .restore_stream_state_checkpoint(transaction.scheduler_state)?;
            let stream = self.streams.get_mut(&stream_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedPromptEngineError::UnknownStream {
                    stream_id: stream_id.clone(),
                }
            })?;
            if let Some(mut resident_state) = transaction.resident_state {
                if transaction.page_cow {
                    stream.restore_transaction_page_checkpoint(
                        transaction.resident_page_checkpoint,
                    )?;
                    VulkanResidentPlacedPrefixStateCache::restore_transaction_entry(
                        &mut resident_state,
                        stream,
                    )?;
                } else {
                    VulkanResidentPlacedPrefixStateCache::restore_entry(
                        &resident_state,
                        stream,
                    )?;
                }
            } else {
                stream.restore_initial_transaction_state()?;
            }
            stream.session.next_stream_tick = transaction.session.next_stream_tick;
            stream.session.completed_prompt_event_count =
                transaction.session.completed_prompt_event_count;
            stream.session.generated_token_count = transaction.session.generated_token_count;
            stream.session.output_token_count = transaction.session.output_token_count;
            self.stream_histories
                .insert(stream_id.clone(), transaction.history);
            Ok(())
        })();
        if transaction.page_cow
            && let Some(stream) = self.streams.get_mut(&stream_id)
        {
            stream.end_transaction_page_cow();
        }
        self.close_stream_transaction_depth(&stream_id, depth);
        restore
    }

    pub(crate) fn commit_stream_transaction(
        &mut self,
        transaction: VulkanResidentInProcessPlacedPromptEngineStreamTransaction,
    ) -> Result<(), VulkanResidentInProcessPlacedPromptEngineError> {
        let _state_commit = runtime_critical_path_span(RuntimeCriticalPathPhase::StateCommit);
        self.validate_current_stream_transaction(&transaction)?;
        let stream_id = transaction.stream_id.clone();
        let depth = transaction.depth;
        let commit = self
            .runtime_scheduler
            .discard_stream_state_checkpoint(transaction.scheduler_state)
            .map_err(VulkanResidentInProcessPlacedPromptEngineError::from);
        if transaction.page_cow
            && let Some(stream) = self.streams.get_mut(&stream_id)
        {
            stream.end_transaction_page_cow();
        }
        let commit = commit.and_then(|_| {
            if depth == 1 {
                let state = self
                    .runtime_scheduler
                    .stream_transient_state_snapshot(&stream_id)?;
                self.streams
                    .get_mut(&stream_id)
                    .ok_or_else(|| {
                        VulkanResidentInProcessPlacedPromptEngineError::UnknownStream {
                            stream_id: stream_id.clone(),
                        }
                    })?
                    .release_transaction_page_checkpoint(&state);
            }
            Ok(())
        });
        self.close_stream_transaction_depth(&stream_id, depth);
        commit
    }

    pub fn submit_input_event_transactionally_until_idle_with_output<F>(
        &mut self,
        stream_id: &str,
        event: VulkanResidentTokenInputEvent,
        mut on_output_event: F,
    ) -> Result<
        VulkanResidentInProcessPlacedPromptEngineSubmittedInputRun,
        VulkanResidentInProcessPlacedPromptEngineError,
    >
    where
        F: FnMut(VulkanResidentTokenRuntimeSchedulerOutputEvent) -> VulkanResidentOutputControl,
    {
        let transaction = self.begin_stream_transaction(stream_id)?;
        let abort_requested = std::cell::Cell::new(false);
        let run = self.submit_input_event_until_idle_abortable_with_output(
            stream_id,
            event,
            &abort_requested,
            |event| {
                if !abort_requested.get()
                    && on_output_event(event) == VulkanResidentOutputControl::Abort
                {
                    abort_requested.set(true);
                }
            },
        );
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
