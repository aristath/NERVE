#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanPlacedPromptEngineShutdownReport {
    pub stream_count: usize,
    pub package_count: usize,
    pub scheduler_in_flight_activation_count: usize,
    pub physical_device_count: usize,
    pub acknowledged_device_count: usize,
    pub released_unit_count: usize,
    pub released_payload_bytes: usize,
    pub cancelled_load_count: usize,
    pub resource_teardowns:
        Vec<VulkanCompiledResourceTeardownReport>,
    pub complete: bool,
    pub errors: Vec<String>,
}

impl VulkanPlacedPromptEngineShutdownReport {
    fn record_resource_teardown(
        &mut self,
        teardown: &VulkanCompiledResourceTeardownReport,
    ) -> Result<(), String> {
        let physical_device_count = self
            .physical_device_count
            .checked_add(teardown.physical_device_count)
            .ok_or_else(|| {
                "engine shutdown physical-device count overflowed".to_string()
            })?;
        let acknowledged_device_count = self
            .acknowledged_device_count
            .checked_add(teardown.acknowledged_device_count)
            .ok_or_else(|| {
                "engine shutdown acknowledged-device count overflowed"
                    .to_string()
            })?;
        let released_unit_count = self
            .released_unit_count
            .checked_add(teardown.released_unit_count)
            .ok_or_else(|| {
                "engine shutdown released-unit count overflowed".to_string()
            })?;
        let released_payload_bytes = self
            .released_payload_bytes
            .checked_add(teardown.released_payload_bytes)
            .ok_or_else(|| {
                "engine shutdown released payload bytes overflowed".to_string()
            })?;
        let cancelled_load_count = self
            .cancelled_load_count
            .checked_add(teardown.cancelled_load_count)
            .ok_or_else(|| {
                "engine shutdown cancelled-load count overflowed".to_string()
            })?;
        self.physical_device_count = physical_device_count;
        self.acknowledged_device_count = acknowledged_device_count;
        self.released_unit_count = released_unit_count;
        self.released_payload_bytes = released_payload_bytes;
        self.cancelled_load_count = cancelled_load_count;
        Ok(())
    }
}

impl VulkanResidentInProcessPlacedPromptEngine {
    pub fn shutdown(
        mut self,
    ) -> VulkanPlacedPromptEngineShutdownReport {
        self.shutdown_in_place()
    }

    fn shutdown_in_place(
        &mut self,
    ) -> VulkanPlacedPromptEngineShutdownReport {
        if self.shutdown_attempted {
            return VulkanPlacedPromptEngineShutdownReport {
                complete: true,
                ..Default::default()
            };
        }
        self.shutdown_attempted = true;
        let mut report = VulkanPlacedPromptEngineShutdownReport {
            stream_count: self.streams.len(),
            complete: true,
            ..Default::default()
        };
        let stream_ids = self.streams.keys().cloned().collect::<Vec<_>>();
        let mut package_pointers = BTreeSet::new();
        let mut packages =
            Vec::<Arc<VulkanResidentInProcessPlacedModelPackage>>::new();
        for stream in self.streams.values() {
            if package_pointers
                .insert(Arc::as_ptr(&stream.package) as usize)
            {
                packages.push(Arc::clone(&stream.package));
            }
        }
        packages.sort_by(|left, right| {
            left.package_id
                .cmp(&right.package_id)
                .then_with(|| {
                    left.execution_scope.cmp(&right.execution_scope)
                })
                .then_with(|| {
                    left.runtime_execution_identity
                        .cmp(&right.runtime_execution_identity)
                })
        });
        report.package_count = packages.len();

        for stream_id in &stream_ids {
            if let Some(stream) = self.streams.get_mut(stream_id)
                && let Err(error) =
                    stream.quiesce_and_discard_transaction_work()
            {
                report.errors.push(format!(
                    "stream {stream_id:?} failed to quiesce: {error}"
                ));
            }
            if let Err(error) = self
                .runtime_scheduler
                .interrupt_stream(
                    stream_id,
                    "explicit engine shutdown",
                )
            {
                report.errors.push(format!(
                    "stream {stream_id:?} scheduler interruption failed: {error}"
                ));
            }
        }
        self.active_transaction_stream_ids.clear();
        self.multi_stream_batch_runners.clear();

        let scheduler = self.runtime_scheduler.snapshot();
        report.scheduler_in_flight_activation_count =
            scheduler.in_flight_activation_count;
        if scheduler.in_flight_activation_count != 0 {
            report.errors.push(format!(
                "scheduler retained {} in-flight activations after shutdown quiescence",
                scheduler.in_flight_activation_count,
            ));
        }

        for package in &packages {
            let teardown = package.teardown_compiled_resources();
            if !teardown.complete {
                report.errors.push(format!(
                    "package {:?} failed compiled-resource teardown on {} of {} physical devices",
                    package.package_id,
                    teardown
                        .physical_device_count
                        .saturating_sub(
                            teardown.acknowledged_device_count,
                        ),
                    teardown.physical_device_count,
                ));
            }
            if let Err(error) = report.record_resource_teardown(&teardown) {
                report.errors.push(error);
            }
            report.resource_teardowns.push(teardown);
        }

        self.streams.clear();
        self.stream_histories.clear();
        self.latest_prefix_checkpoint_by_stream.clear();
        self.resident_prefix_state_cache =
            VulkanResidentPlacedPrefixStateCache::default();
        self.multi_stream_batch_runners.clear();
        drop(packages);

        report.complete = report.errors.is_empty()
            && report.scheduler_in_flight_activation_count == 0
            && report
                .resource_teardowns
                .iter()
                .all(|teardown| teardown.complete);
        report
    }
}
