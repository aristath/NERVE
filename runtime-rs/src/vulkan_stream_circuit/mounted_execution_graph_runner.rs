pub struct VulkanMountedPlacedResidentExecutionGraphRunner {
    pub device_id: String,
    pub components: Vec<VulkanMountedPlacedResidentComponentRunner>,
    pub total_descriptor_count: usize,
    pub total_push_constant_byte_count: u32,
    sequence: VulkanResidentKernelSequence,
}

impl VulkanMountedPlacedResidentExecutionGraphRunner {
    fn from_mounted_bound_plan<I, S>(
        device: &VulkanComputeDevice,
        mounted: &VulkanMountedPlacedStreamCircuit,
        mounted_bound_plan: &VulkanMountedPlacedBoundDispatchPlan,
        component_ids: I,
        loaded_manifest: &VulkanLoadedReusableKernelArtifactManifest,
    ) -> Result<Self, VulkanMountedPlacedResidentKernelDispatchError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut components = Vec::new();
        let mut total_descriptor_count = 0usize;
        let mut total_push_constant_byte_count = 0u32;

        for component_id in component_ids {
            let component_id = component_id.as_ref();
            let runner = VulkanMountedPlacedResidentComponentRunner::from_mounted_bound_plan(
                device,
                mounted,
                mounted_bound_plan,
                component_id,
                loaded_manifest,
            )?;
            total_descriptor_count = total_descriptor_count
                .checked_add(runner.total_descriptor_count)
                .ok_or_else(|| {
                    VulkanMountedPlacedResidentKernelDispatchError::ExecutionGraphRunnerDescriptorCountOverflow {
                        device_id: mounted_bound_plan.device_id.clone(),
                    }
                })?;
            total_push_constant_byte_count = total_push_constant_byte_count
                .checked_add(runner.total_push_constant_byte_count)
                .ok_or_else(|| {
                    VulkanMountedPlacedResidentKernelDispatchError::ExecutionGraphRunnerPushConstantByteCountOverflow {
                        device_id: mounted_bound_plan.device_id.clone(),
                    }
                })?;
            components.push(runner);
        }

        if components.is_empty() {
            return Err(
                VulkanMountedPlacedResidentKernelDispatchError::MissingExecutionGraphComponents {
                    device_id: mounted_bound_plan.device_id.clone(),
                },
            );
        }

        Ok(Self {
            device_id: mounted_bound_plan.device_id.clone(),
            components,
            total_descriptor_count,
            total_push_constant_byte_count,
            sequence: device
                .create_resident_kernel_sequence()
                .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?,
        })
    }

    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    pub fn dispatch_count(&self) -> usize {
        self.components
            .iter()
            .map(VulkanMountedPlacedResidentComponentRunner::dispatch_count)
            .sum()
    }

    pub fn component_ids(&self) -> Vec<&str> {
        self.components
            .iter()
            .map(|component| component.component_id.as_str())
            .collect()
    }

    pub fn run_zeroed_push_constants(
        &self,
        device: &VulkanComputeDevice,
    ) -> Result<
        VulkanMountedPlacedResidentExecutionGraphRun,
        VulkanMountedPlacedResidentKernelDispatchError,
    > {
        self.run_with_push_constant_bytes(device, |dispatch| {
            Ok(vec![
                0u8;
                dispatch.resident_dispatch.push_constant_byte_count()
                    as usize
            ])
        })
    }

    pub fn run_with_stream_control(
        &self,
        device: &VulkanComputeDevice,
        control: VulkanMountedPlacedStreamControl,
    ) -> Result<
        VulkanMountedPlacedResidentExecutionGraphRun,
        VulkanMountedPlacedResidentKernelDispatchError,
    > {
        self.components
            .first()
            .expect("resident execution graph components are non-empty")
            .stream_control_buffer
            .write_bytes_at(
                VULKAN_STREAM_CONTROL_METADATA_OFFSET,
                &stream_control_metadata_bytes(control),
            )
            .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
        self.run_with_push_constant_bytes(device, |dispatch| {
            stream_control_push_constant_bytes(&dispatch.push_constants, control)
        })
    }

    fn run_with_push_constant_bytes<F>(
        &self,
        device: &VulkanComputeDevice,
        mut push_constant_bytes_for: F,
    ) -> Result<
        VulkanMountedPlacedResidentExecutionGraphRun,
        VulkanMountedPlacedResidentKernelDispatchError,
    >
    where
        F: FnMut(
            &VulkanMountedPlacedResidentComponentDispatch,
        ) -> Result<Vec<u8>, VulkanMountedPlacedResidentKernelDispatchError>,
    {
        let dispatches = self
            .components
            .iter()
            .flat_map(|component| component.dispatches.iter())
            .collect::<Vec<_>>();
        let push_constants = dispatches
            .iter()
            .map(|dispatch| push_constant_bytes_for(dispatch))
            .collect::<Result<Vec<_>, _>>()?;
        let steps = dispatches
            .iter()
            .zip(&push_constants)
            .map(|(dispatch, push_constants)| {
                VulkanResidentKernelSequenceStep::new(
                    &dispatch.resident_dispatch,
                    push_constants,
                )
            })
            .collect::<Vec<_>>();
        device
            .run_resident_kernel_sequence(&self.sequence, &steps)
            .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
        Ok(self.completed_sequence_run())
    }

    fn completed_sequence_run(&self) -> VulkanMountedPlacedResidentExecutionGraphRun {
        VulkanMountedPlacedResidentExecutionGraphRun {
            device_id: self.device_id.clone(),
            component_runs: self
                .components
                .iter()
                .map(VulkanMountedPlacedResidentComponentRunner::completed_sequence_run)
                .collect(),
        }
    }
}
