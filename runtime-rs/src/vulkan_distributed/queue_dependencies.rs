/// Reusable owner/helper queue dependency for shared resident memory.
///
/// The owner publishes input writes through `ready_source`; the helper waits
/// through its permanently imported `ready_wait`. The helper publishes output
/// writes through `done_source`; the owner consumes them through `done_wait`.
/// Keeping both directions as device-side timeline edges is what makes shared
/// memory a coherent execution route rather than merely aliased storage.
pub(crate) struct VulkanDistributedQueueSynchronization {
    pub(crate) device_id: String,
    ready_source: VulkanTimelineSemaphore,
    ready_wait: VulkanTimelineSemaphore,
    done_source: VulkanTimelineSemaphore,
    done_wait: VulkanTimelineSemaphore,
}

impl VulkanDistributedQueueSynchronization {
    pub(crate) fn new(
        owner_device: &VulkanComputeDevice,
        helper_device: &VulkanComputeDevice,
        owner_device_id: &str,
        helper_device_id: &str,
        label: &str,
    ) -> Result<Self, VulkanError> {
        if !owner_device.supports_opaque_fd_timeline_semaphores()
            || !helper_device.supports_opaque_fd_timeline_semaphores()
        {
            return Err(VulkanError(format!(
                "{label} requires persistent opaque-file timeline semaphores on owner {owner_device_id:?} and helper {helper_device_id:?}"
            )));
        }
        let ready_source = owner_device.create_opaque_fd_exportable_timeline_semaphore(0)?;
        let ready_wait = helper_device.create_timeline_semaphore(0)?;
        helper_device.import_timeline_semaphore_opaque_fd(
            &ready_wait,
            owner_device.export_timeline_semaphore_opaque_fd(&ready_source)?,
        )?;
        let done_source = helper_device.create_opaque_fd_exportable_timeline_semaphore(0)?;
        let done_wait = owner_device.create_timeline_semaphore(0)?;
        owner_device.import_timeline_semaphore_opaque_fd(
            &done_wait,
            helper_device.export_timeline_semaphore_opaque_fd(&done_source)?,
        )?;
        Ok(Self {
            device_id: helper_device_id.to_string(),
            ready_source,
            ready_wait,
            done_source,
            done_wait,
        })
    }

    pub(crate) fn owner_ready(&self, value: u64) -> VulkanTimelineSemaphorePoint<'_> {
        VulkanTimelineSemaphorePoint::new(&self.ready_source, value)
    }

    pub(crate) fn helper_ready(&self, value: u64) -> VulkanTimelineSemaphorePoint<'_> {
        VulkanTimelineSemaphorePoint::new(&self.ready_wait, value)
    }

    pub(crate) fn helper_done(&self, value: u64) -> VulkanTimelineSemaphorePoint<'_> {
        VulkanTimelineSemaphorePoint::new(&self.done_source, value)
    }

    pub(crate) fn owner_done(&self, value: u64) -> VulkanTimelineSemaphorePoint<'_> {
        VulkanTimelineSemaphorePoint::new(&self.done_wait, value)
    }
}

pub(crate) struct VulkanDistributedDependencyClock {
    next_value: Cell<u64>,
}

impl VulkanDistributedDependencyClock {
    pub(crate) fn new() -> Self {
        Self {
            next_value: Cell::new(1),
        }
    }

    pub(crate) fn reserve(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
    ) -> Result<u64, VulkanDistributedDispatchRunnerError> {
        let value = self.next_value.get();
        let next = value.checked_add(1).ok_or_else(|| {
            VulkanDistributedDispatchRunnerError(format!(
                "distributed dispatch {dispatch_index} owned by {owner_device_id:?} exhausted its timeline semaphore values"
            ))
        })?;
        self.next_value.set(next);
        Ok(value)
    }

    fn validate_advance(
        &self,
        count: u64,
        owner_device_id: &str,
        dispatch_index: usize,
    ) -> Result<(), VulkanDistributedDispatchRunnerError> {
        self.next_value.get().checked_add(count).ok_or_else(|| {
            VulkanDistributedDispatchRunnerError(format!(
                "distributed dispatch {dispatch_index} owned by {owner_device_id:?} exhausts its timeline semaphore values during replay"
            ))
        })?;
        Ok(())
    }

    fn advance(&self, count: u64) {
        self.next_value.set(
            self.next_value
                .get()
                .checked_add(count)
                .expect("distributed replay dependency advance was validated"),
        );
    }
}
