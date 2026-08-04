pub struct VulkanMountedStreamCircuit {
    pub resident_plan: VulkanStreamCircuitResidentPlan,
    pub binding_plan: VulkanStreamCircuitBindingPlan,
    pub kernel_interface_plan: VulkanKernelInterfacePlan,
    pub dispatch_plan: VulkanKernelDispatchPlan,
    pub reusable_kernel_plan: VulkanReusableKernelPlan,
    pub buffers: VulkanStreamCircuitStreamBuffers,
}

impl VulkanMountedStreamCircuit {
    pub fn from_plans(
        device: &VulkanComputeDevice,
        execution_plan: &StreamCircuitExecutionPlan,
        resource_plan: &StreamCircuitResourcePlan,
        resident_plan: VulkanStreamCircuitResidentPlan,
        dynamic_state_capacity_activations: usize,
    ) -> Result<Self, VulkanStreamCircuitMountError> {
        let binding_plan = VulkanStreamCircuitBindingPlan::from_plans(
            execution_plan,
            resource_plan,
            &resident_plan,
        )?;
        let kernel_interface_plan = VulkanKernelInterfacePlan::from_binding_plan(&binding_plan);
        let dispatch_plan =
            VulkanKernelDispatchPlan::from_kernel_interfaces(&kernel_interface_plan);
        let reusable_kernel_plan = VulkanReusableKernelPlan::from_dispatch_plan(&dispatch_plan);
        let buffers =
            resident_plan.allocate_stream_buffers(device, dynamic_state_capacity_activations)?;
        Ok(Self {
            resident_plan,
            binding_plan,
            kernel_interface_plan,
            dispatch_plan,
            reusable_kernel_plan,
            buffers,
        })
    }

    pub fn can_execute(&self) -> bool {
        false
    }

    pub fn reusable_kernel_coverage_report<I, S>(
        &self,
        available_family_ids: I,
    ) -> VulkanReusableKernelCoverageReport
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.reusable_kernel_plan
            .coverage_report(available_family_ids)
    }

    pub fn link_reusable_kernels(
        &self,
        manifest: &VulkanReusableKernelArtifactManifest,
    ) -> VulkanLinkedReusableKernelPlan {
        self.reusable_kernel_plan.link_artifacts(manifest)
    }

    pub fn descriptor_resource_plan(
        &self,
    ) -> Result<VulkanDescriptorResourcePlan, VulkanDescriptorResourcePlanError> {
        VulkanDescriptorResourcePlan::from_plans(
            &self.dispatch_plan,
            &self.resident_plan,
            self.buffers.dynamic_state_capacity_activations,
        )
    }

    pub fn prepared_dispatch_plan(
        &self,
        manifest: &VulkanReusableKernelArtifactManifest,
    ) -> Result<VulkanPreparedDispatchPlan, VulkanPreparedDispatchPlanError> {
        let descriptor_plan = self
            .descriptor_resource_plan()
            .map_err(VulkanPreparedDispatchPlanError::DescriptorResource)?;
        VulkanPreparedDispatchPlan::from_plans(
            &self.dispatch_plan,
            &self.reusable_kernel_plan,
            &descriptor_plan,
            manifest,
        )
    }

    pub fn bound_dispatch_plan(
        &self,
        manifest: &VulkanReusableKernelArtifactManifest,
    ) -> Result<VulkanBoundDispatchPlan, VulkanBoundDispatchPlanError> {
        let prepared_plan = self
            .prepared_dispatch_plan(manifest)
            .map_err(VulkanBoundDispatchPlanError::PreparedDispatch)?;
        VulkanBoundDispatchPlan::from_prepared_plan(&prepared_plan, &self.buffers)
    }
}

#[derive(Debug)]
pub enum VulkanStreamCircuitMountError {
    Binding(VulkanBindingPlanError),
    BoundaryIo(VulkanModelBoundaryBufferPlanError),
    EdgeIo(VulkanPlacedEdgeIoPlanError),
    PermanentParameters(VulkanPermanentParameterBufferPlanError),
    Vulkan(VulkanError),
}

impl Display for VulkanStreamCircuitMountError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Binding(error) => Display::fmt(error, f),
            Self::BoundaryIo(error) => Display::fmt(error, f),
            Self::EdgeIo(error) => Display::fmt(error, f),
            Self::PermanentParameters(error) => Display::fmt(error, f),
            Self::Vulkan(error) => Display::fmt(error, f),
        }
    }
}

impl Error for VulkanStreamCircuitMountError {}

impl From<VulkanBindingPlanError> for VulkanStreamCircuitMountError {
    fn from(error: VulkanBindingPlanError) -> Self {
        Self::Binding(error)
    }
}

impl From<VulkanModelBoundaryBufferPlanError> for VulkanStreamCircuitMountError {
    fn from(error: VulkanModelBoundaryBufferPlanError) -> Self {
        Self::BoundaryIo(error)
    }
}

impl From<VulkanPlacedEdgeIoPlanError> for VulkanStreamCircuitMountError {
    fn from(error: VulkanPlacedEdgeIoPlanError) -> Self {
        Self::EdgeIo(error)
    }
}

impl From<VulkanPermanentParameterBufferPlanError> for VulkanStreamCircuitMountError {
    fn from(error: VulkanPermanentParameterBufferPlanError) -> Self {
        Self::PermanentParameters(error)
    }
}

impl From<VulkanError> for VulkanStreamCircuitMountError {
    fn from(error: VulkanError) -> Self {
        Self::Vulkan(error)
    }
}

pub struct VulkanMountedPlacedStreamCircuit {
    pub placed_plan: VulkanPlacedStreamCircuitPlan,
    pub parameter_buffers: Arc<VulkanPermanentParameterBuffers>,
    pub buffers: VulkanStreamCircuitStreamBuffers,
    pub boundary_io: VulkanModelBoundaryBuffers,
    pub edge_io: VulkanPlacedEdgeIoBuffers,
    pub stream_control_buffer: Arc<VulkanResidentBuffer>,
    pub dynamic_resource_buffers: Option<Arc<VulkanDynamicResourceBuffers>>,
}

fn boundary_input_passthrough_edge_overrides(
    placed_plan: &VulkanPlacedStreamCircuitPlan,
    boundary_io: &VulkanModelBoundaryBuffers,
) -> (
    Vec<VulkanPlacedLocalEdgeBufferOverride>,
    Vec<VulkanPlacedEdgeEndpointBufferOverride>,
) {
    let mut local = Vec::new();
    let mut outgoing = Vec::new();
    for circuit in &placed_plan.binding_plan.circuits {
        for output in &circuit.output_ports {
            let Some(source_signal_id) = output.source.as_deref() else {
                continue;
            };
            let Some(input) = boundary_io.input_buffers.iter().find(|input| {
                input.boundary.component_id == circuit.component_id
                    && input.boundary.signal_id == source_signal_id
                    && input.boundary.shape == output.shape
            }) else {
                continue;
            };
            for edge in &placed_plan.placed_resident_plan.local_edges {
                if edge.source_component_id == circuit.component_id
                    && edge.source_port_id == output.id
                {
                    local.push(VulkanPlacedLocalEdgeBufferOverride {
                        edge_index: edge.edge_index,
                        buffer: Arc::clone(&input.buffer),
                    });
                }
            }
            for edge in &placed_plan.placed_resident_plan.outgoing_edges {
                if edge.source_component_id == circuit.component_id
                    && edge.source_port_id == output.id
                {
                    outgoing.push(VulkanPlacedEdgeEndpointBufferOverride {
                        direction: VulkanPlacedEdgeDirection::Outgoing,
                        edge_index: edge.edge_index,
                        buffer: Arc::clone(&input.buffer),
                    });
                }
            }
        }
    }
    (local, outgoing)
}

impl VulkanMountedPlacedStreamCircuit {
    pub fn from_placed_plan(
        device: &VulkanComputeDevice,
        placed_plan: VulkanPlacedStreamCircuitPlan,
        dynamic_state_capacity_activations: usize,
    ) -> Result<Self, VulkanStreamCircuitMountError> {
        let parameter_buffer_plan = VulkanPermanentParameterBufferPlan::from_placed_resident_plan(
            &placed_plan.placed_resident_plan,
        )?;
        let parameter_buffers = Arc::new(parameter_buffer_plan.allocate_buffers(device)?);
        Self::from_placed_plan_with_parameter_buffers(
            device,
            placed_plan,
            dynamic_state_capacity_activations,
            parameter_buffers,
        )
    }

    pub fn from_placed_plan_with_parameter_buffers(
        device: &VulkanComputeDevice,
        placed_plan: VulkanPlacedStreamCircuitPlan,
        dynamic_state_capacity_activations: usize,
        parameter_buffers: Arc<VulkanPermanentParameterBuffers>,
    ) -> Result<Self, VulkanStreamCircuitMountError> {
        Self::from_placed_plan_with_parameter_buffers_and_activation_overrides(
            device,
            placed_plan,
            dynamic_state_capacity_activations,
            parameter_buffers,
            &[],
        )
    }

    pub fn from_placed_plan_with_parameter_buffers_and_activation_overrides(
        device: &VulkanComputeDevice,
        placed_plan: VulkanPlacedStreamCircuitPlan,
        dynamic_state_capacity_activations: usize,
        parameter_buffers: Arc<VulkanPermanentParameterBuffers>,
        activation_overrides: &[VulkanActivationSlotBufferOverride],
    ) -> Result<Self, VulkanStreamCircuitMountError> {
        Self::from_placed_plan_with_parameter_buffers_and_buffer_overrides(
            device,
            placed_plan,
            dynamic_state_capacity_activations,
            parameter_buffers,
            activation_overrides,
            &[],
            None,
        )
    }

    pub fn from_placed_plan_with_parameter_buffers_and_buffer_overrides(
        device: &VulkanComputeDevice,
        placed_plan: VulkanPlacedStreamCircuitPlan,
        dynamic_state_capacity_activations: usize,
        parameter_buffers: Arc<VulkanPermanentParameterBuffers>,
        activation_overrides: &[VulkanActivationSlotBufferOverride],
        edge_endpoint_overrides: &[VulkanPlacedEdgeEndpointBufferOverride],
        stream_control_override: Option<Arc<VulkanResidentBuffer>>,
    ) -> Result<Self, VulkanStreamCircuitMountError> {
        Self::from_placed_plan_with_parameter_buffers_and_all_buffer_overrides(
            device,
            placed_plan,
            dynamic_state_capacity_activations,
            parameter_buffers,
            activation_overrides,
            &[],
            edge_endpoint_overrides,
            stream_control_override,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_placed_plan_with_parameter_buffers_and_all_buffer_overrides(
        device: &VulkanComputeDevice,
        placed_plan: VulkanPlacedStreamCircuitPlan,
        dynamic_state_capacity_activations: usize,
        parameter_buffers: Arc<VulkanPermanentParameterBuffers>,
        activation_overrides: &[VulkanActivationSlotBufferOverride],
        local_edge_overrides: &[VulkanPlacedLocalEdgeBufferOverride],
        edge_endpoint_overrides: &[VulkanPlacedEdgeEndpointBufferOverride],
        stream_control_override: Option<Arc<VulkanResidentBuffer>>,
        dynamic_resource_buffers: Option<
            Arc<VulkanDynamicResourceBuffers>,
        >,
    ) -> Result<Self, VulkanStreamCircuitMountError> {
        let buffers = placed_plan
            .placed_resident_plan
            .resident_plan
            .allocate_stream_buffers_with_activation_overrides(
                device,
                dynamic_state_capacity_activations,
                activation_overrides,
            )?;
        let boundary_io_plan = VulkanModelBoundaryBufferPlan::from_placed_plan(&placed_plan)?;
        let boundary_io = boundary_io_plan.allocate_buffers(device)?;
        let edge_io_plan =
            VulkanPlacedEdgeIoPlan::from_placed_resident_plan(&placed_plan.placed_resident_plan)?;
        let (passthrough_local_overrides, passthrough_endpoint_overrides) =
            boundary_input_passthrough_edge_overrides(&placed_plan, &boundary_io);
        let passthrough_local_edge_ids = passthrough_local_overrides
            .iter()
            .map(|override_| override_.edge_index)
            .collect::<BTreeSet<_>>();
        let passthrough_endpoint_keys = passthrough_endpoint_overrides
            .iter()
            .map(|override_| (override_.direction, override_.edge_index))
            .collect::<BTreeSet<_>>();
        let effective_local_overrides = local_edge_overrides
            .iter()
            .filter(|override_| !passthrough_local_edge_ids.contains(&override_.edge_index))
            .map(|override_| VulkanPlacedLocalEdgeBufferOverride {
                edge_index: override_.edge_index,
                buffer: Arc::clone(&override_.buffer),
            })
            .chain(passthrough_local_overrides)
            .collect::<Vec<_>>();
        let effective_endpoint_overrides = edge_endpoint_overrides
            .iter()
            .filter(|override_| {
                !passthrough_endpoint_keys.contains(&(override_.direction, override_.edge_index))
            })
            .map(|override_| VulkanPlacedEdgeEndpointBufferOverride {
                direction: override_.direction,
                edge_index: override_.edge_index,
                buffer: Arc::clone(&override_.buffer),
            })
            .chain(passthrough_endpoint_overrides)
            .collect::<Vec<_>>();
        let edge_io = edge_io_plan.allocate_buffers_with_overrides(
            device,
            &effective_local_overrides,
            &effective_endpoint_overrides,
        )?;
        let stream_control_buffer = if let Some(stream_control_buffer) = stream_control_override {
            if !device.owns_resident_buffer(&stream_control_buffer) {
                return Err(VulkanStreamCircuitMountError::Vulkan(VulkanError(
                    "stream-control buffer override belongs to a different Vulkan logical device"
                        .to_string(),
                )));
            }
            if stream_control_buffer.byte_capacity() < VULKAN_STREAM_CONTROL_BYTE_CAPACITY {
                return Err(VulkanStreamCircuitMountError::Vulkan(VulkanError(format!(
                    "stream-control buffer override has {} bytes, needs {VULKAN_STREAM_CONTROL_BYTE_CAPACITY}",
                    stream_control_buffer.byte_capacity()
                ))));
            }
            if !stream_control_buffer.is_persistently_mapped() {
                return Err(VulkanStreamCircuitMountError::Vulkan(VulkanError(
                    "stream-control buffer override is not persistently host mapped".to_string(),
                )));
            }
            stream_control_buffer
        } else {
            let mut stream_control_buffer =
                device.create_host_visible_resident_buffer(VULKAN_STREAM_CONTROL_BYTE_CAPACITY)?;
            stream_control_buffer.persistently_map()?;
            stream_control_buffer.write_bytes(&[0; VULKAN_STREAM_CONTROL_BYTE_CAPACITY])?;
            Arc::new(stream_control_buffer)
        };
        Ok(Self {
            placed_plan,
            parameter_buffers,
            buffers,
            boundary_io,
            edge_io,
            stream_control_buffer,
            dynamic_resource_buffers,
        })
    }

    pub fn can_execute(&self) -> bool {
        false
    }

    pub fn device_id(&self) -> &str {
        &self.placed_plan.device_id
    }

    pub fn descriptor_resource_plan(
        &self,
    ) -> Result<VulkanDescriptorResourcePlan, VulkanDescriptorResourcePlanError> {
        VulkanDescriptorResourcePlan::from_plans(
            &self.placed_plan.dispatch_plan,
            &self.placed_plan.placed_resident_plan.resident_plan,
            self.buffers.dynamic_state_capacity_activations,
        )
    }

    pub fn prepared_dispatch_plan(
        &self,
        manifest: &VulkanReusableKernelArtifactManifest,
    ) -> Result<VulkanPreparedDispatchPlan, VulkanPreparedDispatchPlanError> {
        let descriptor_plan = self
            .descriptor_resource_plan()
            .map_err(VulkanPreparedDispatchPlanError::DescriptorResource)?;
        VulkanPreparedDispatchPlan::from_plans(
            &self.placed_plan.dispatch_plan,
            &self.placed_plan.reusable_kernel_plan,
            &descriptor_plan,
            manifest,
        )
    }

    pub fn bound_dispatch_plan(
        &self,
        manifest: &VulkanReusableKernelArtifactManifest,
    ) -> Result<VulkanBoundDispatchPlan, VulkanBoundDispatchPlanError> {
        let prepared_plan = self
            .prepared_dispatch_plan(manifest)
            .map_err(VulkanBoundDispatchPlanError::PreparedDispatch)?;
        VulkanBoundDispatchPlan::from_prepared_plan(&prepared_plan, &self.buffers)
    }

    pub fn placed_bound_dispatch_plan(
        &self,
        manifest: &VulkanReusableKernelArtifactManifest,
    ) -> Result<VulkanPlacedBoundDispatchPlan, VulkanBoundDispatchPlanError> {
        let bound_plan = self.bound_dispatch_plan(manifest)?;
        Ok(VulkanPlacedBoundDispatchPlan::from_bound_plan(
            &bound_plan,
            &self.placed_plan.placed_resident_plan,
        ))
    }

    pub fn mounted_placed_bound_dispatch_plan(
        &self,
        manifest: &VulkanReusableKernelArtifactManifest,
    ) -> Result<VulkanMountedPlacedBoundDispatchPlan, VulkanBoundDispatchPlanError> {
        let placed_bound_plan = self.placed_bound_dispatch_plan(manifest)?;
        VulkanMountedPlacedBoundDispatchPlan::from_placed_bound_plan(
            &placed_bound_plan,
            &self.edge_io,
        )
    }

    pub fn stream_tick_plan(
        &self,
        manifest: &VulkanReusableKernelArtifactManifest,
    ) -> Result<VulkanMountedPlacedStreamTickPlan, VulkanBoundDispatchPlanError> {
        let mounted_bound_plan = self.mounted_placed_bound_dispatch_plan(manifest)?;
        Ok(VulkanMountedPlacedStreamTickPlan::from_mounted_bound_plan(
            &mounted_bound_plan,
        ))
    }

    pub fn advance_stream_tick(
        &self,
        manifest: &VulkanReusableKernelArtifactManifest,
        stream_tick: u64,
    ) -> Result<VulkanMountedPlacedStreamTickRun, VulkanMountedPlacedStreamTickError> {
        let tick_plan = self.stream_tick_plan(manifest)?;
        Ok(tick_plan.advance(stream_tick))
    }

    pub fn resident_kernel_dispatch_readiness_plan(
        &self,
        manifest: &VulkanReusableKernelArtifactManifest,
        loaded_manifest: &VulkanLoadedReusableKernelArtifactManifest,
    ) -> Result<VulkanMountedPlacedResidentKernelDispatchReadinessPlan, VulkanBoundDispatchPlanError>
    {
        let mounted_bound_plan = self.mounted_placed_bound_dispatch_plan(manifest)?;
        Ok(
            VulkanMountedPlacedResidentKernelDispatchReadinessPlan::from_mounted_bound_plan(
                self,
                &mounted_bound_plan,
                loaded_manifest,
            ),
        )
    }

    pub fn resident_kernel_dispatch_readiness_for_bound_dispatch(
        &self,
        dispatch: &VulkanMountedPlacedBoundDispatch,
        loaded_manifest: &VulkanLoadedReusableKernelArtifactManifest,
    ) -> VulkanMountedPlacedResidentKernelDispatchStatus {
        let Some(artifact) = loaded_manifest.artifact(&dispatch.reusable_family_id) else {
            return VulkanMountedPlacedResidentKernelDispatchStatus::Blocked {
                error: VulkanMountedPlacedResidentKernelDispatchError::MissingLoadedArtifact {
                    dispatch_index: dispatch.dispatch_index,
                    family_id: dispatch.reusable_family_id.clone(),
                },
            };
        };

        let descriptor_count =
            match self.resident_kernel_buffer_bindings_for_bound_dispatch(dispatch) {
                Ok(bindings) => bindings.len(),
                Err(error) => {
                    return VulkanMountedPlacedResidentKernelDispatchStatus::Blocked { error };
                }
            };
        let push_constant_byte_count = match push_constant_byte_count(&dispatch.push_constants) {
            Ok(bytes) => bytes,
            Err(error) => {
                return VulkanMountedPlacedResidentKernelDispatchStatus::Blocked { error };
            }
        };

        VulkanMountedPlacedResidentKernelDispatchStatus::Instantiable {
            descriptor_count,
            workgroup_count_x: artifact.artifact.workgroup_count_x,
            local_size_x: artifact.artifact.local_size_x,
            push_constant_byte_count,
        }
    }

    pub fn resident_kernel_buffer_bindings_for_bound_dispatch<'a>(
        &'a self,
        dispatch: &VulkanMountedPlacedBoundDispatch,
    ) -> Result<
        Vec<VulkanResidentKernelBufferBinding<'a>>,
        VulkanMountedPlacedResidentKernelDispatchError,
    > {
        let mut bindings = dispatch
            .descriptors
            .iter()
            .map(|descriptor| self.resident_kernel_buffer_binding(dispatch, descriptor))
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(binding) = dispatch.stream_control_binding {
            if usize::try_from(binding).ok() != Some(dispatch.descriptors.len()) {
                return Err(
                    VulkanMountedPlacedResidentKernelDispatchError::StreamControlBindingMismatch {
                        dispatch_index: dispatch.dispatch_index,
                        compiled_binding: binding,
                        runtime_descriptor_count: dispatch.descriptors.len(),
                    },
                );
            }
            bindings.push(
                VulkanResidentKernelBufferBinding::new(
                    binding,
                    &self.stream_control_buffer,
                    VULKAN_STREAM_CONTROL_BYTE_CAPACITY,
                )
                .with_access(VulkanResidentKernelBufferAccess::Read),
            );
        }
        Ok(bindings)
    }

    pub fn create_resident_kernel_dispatch_for_bound_dispatch(
        &self,
        device: &VulkanComputeDevice,
        dispatch: &VulkanMountedPlacedBoundDispatch,
        loaded_manifest: &VulkanLoadedReusableKernelArtifactManifest,
    ) -> Result<VulkanResidentKernelDispatch, VulkanMountedPlacedResidentKernelDispatchError> {
        let artifact = loaded_manifest
            .artifact(&dispatch.reusable_family_id)
            .ok_or_else(|| {
                VulkanMountedPlacedResidentKernelDispatchError::MissingLoadedArtifact {
                    dispatch_index: dispatch.dispatch_index,
                    family_id: dispatch.reusable_family_id.clone(),
                }
            })?;
        let buffer_bindings = self.resident_kernel_buffer_bindings_for_bound_dispatch(dispatch)?;
        device
            .create_resident_kernel_dispatch_labeled(
                &artifact.words,
                &buffer_bindings,
                artifact.artifact.workgroup_count_x,
                artifact.artifact.local_size_x,
                push_constant_byte_count(&dispatch.push_constants)?,
                Some(vulkan_dispatch_semantic_label(dispatch, None)),
            )
            .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)
    }

    pub fn create_resident_component_runner(
        &self,
        device: &VulkanComputeDevice,
        mounted_bound_plan: &VulkanMountedPlacedBoundDispatchPlan,
        component_id: &str,
        loaded_manifest: &VulkanLoadedReusableKernelArtifactManifest,
    ) -> Result<
        VulkanMountedPlacedResidentComponentRunner,
        VulkanMountedPlacedResidentKernelDispatchError,
    > {
        VulkanMountedPlacedResidentComponentRunner::from_mounted_bound_plan(
            device,
            self,
            mounted_bound_plan,
            component_id,
            loaded_manifest,
        )
    }

    pub fn create_resident_execution_graph_runner<I, S>(
        &self,
        device: &VulkanComputeDevice,
        mounted_bound_plan: &VulkanMountedPlacedBoundDispatchPlan,
        component_ids: I,
        loaded_manifest: &VulkanLoadedReusableKernelArtifactManifest,
    ) -> Result<
        VulkanMountedPlacedResidentExecutionGraphRunner,
        VulkanMountedPlacedResidentKernelDispatchError,
    >
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        VulkanMountedPlacedResidentExecutionGraphRunner::from_mounted_bound_plan(
            device,
            self,
            mounted_bound_plan,
            component_ids,
            loaded_manifest,
        )
    }

    fn resident_kernel_buffer_binding<'a>(
        &'a self,
        dispatch: &VulkanMountedPlacedBoundDispatch,
        descriptor: &VulkanMountedPlacedBoundDescriptor,
    ) -> Result<VulkanResidentKernelBufferBinding<'a>, VulkanMountedPlacedResidentKernelDispatchError>
    {
        let binding = u32::try_from(descriptor.binding).map_err(|_| {
            VulkanMountedPlacedResidentKernelDispatchError::DescriptorBindingOverflow {
                dispatch_index: dispatch.dispatch_index,
                binding: descriptor.binding,
            }
        })?;
        let (buffer, byte_offset, byte_len) = match &descriptor.target {
            VulkanMountedPlacedBoundDescriptorTarget::Resident { target } => {
                self.resident_kernel_buffer_for_resident_target(dispatch, descriptor, target)?
            }
            VulkanMountedPlacedBoundDescriptorTarget::ModelInput { signal_id } => {
                let allocation = self.boundary_io.input_buffer(signal_id).ok_or_else(|| {
                    VulkanMountedPlacedResidentKernelDispatchError::MissingModelBoundaryBuffer {
                        dispatch_index: dispatch.dispatch_index,
                        binding: descriptor.binding,
                        direction: VulkanModelBoundaryDirection::Input,
                        signal_id: signal_id.clone(),
                    }
                })?;
                (allocation.buffer.as_ref(), 0, allocation.byte_capacity)
            }
            VulkanMountedPlacedBoundDescriptorTarget::ModelOutput { signal_id } => {
                let allocation = self.boundary_io.output_buffer(signal_id).ok_or_else(|| {
                    VulkanMountedPlacedResidentKernelDispatchError::MissingModelBoundaryBuffer {
                        dispatch_index: dispatch.dispatch_index,
                        binding: descriptor.binding,
                        direction: VulkanModelBoundaryDirection::Output,
                        signal_id: signal_id.clone(),
                    }
                })?;
                (allocation.buffer.as_ref(), 0, allocation.byte_capacity)
            }
            VulkanMountedPlacedBoundDescriptorTarget::LocalEdgeInputBuffer { edge } => {
                let allocation = self
                    .edge_io
                    .local_buffers
                    .get(edge.buffer_index)
                    .ok_or_else(|| {
                        VulkanMountedPlacedResidentKernelDispatchError::MissingMountedBuffer {
                            dispatch_index: dispatch.dispatch_index,
                            binding: descriptor.binding,
                            buffer_kind: "local_edge".to_string(),
                            buffer_index: edge.buffer_index,
                        }
                    })?;
                (allocation.buffer.as_ref(), 0, edge.byte_capacity)
            }
            VulkanMountedPlacedBoundDescriptorTarget::IncomingEdgeBuffer { endpoint } => {
                let allocation = self
                    .edge_io
                    .incoming_buffers
                    .get(endpoint.buffer_index)
                    .ok_or_else(|| {
                        VulkanMountedPlacedResidentKernelDispatchError::MissingMountedBuffer {
                            dispatch_index: dispatch.dispatch_index,
                            binding: descriptor.binding,
                            buffer_kind: "incoming_edge".to_string(),
                            buffer_index: endpoint.buffer_index,
                        }
                    })?;
                (allocation.buffer.as_ref(), 0, endpoint.byte_capacity)
            }
            VulkanMountedPlacedBoundDescriptorTarget::ProducedPortBuffer { port } => {
                let allocation = port.buffer(&self.edge_io).ok_or_else(|| {
                    VulkanMountedPlacedResidentKernelDispatchError::MissingMountedBuffer {
                        dispatch_index: dispatch.dispatch_index,
                        binding: descriptor.binding,
                        buffer_kind: "produced_port".to_string(),
                        buffer_index: 0,
                    }
                })?;
                (allocation.as_ref(), 0, port.byte_capacity)
            }
        };

        let access = match descriptor.usage {
            VulkanKernelDescriptorUsage::InputSignal
            | VulkanKernelDescriptorUsage::Parameter
            | VulkanKernelDescriptorUsage::DynamicResourceAddressTable
            | VulkanKernelDescriptorUsage::DynamicResourceParameterSlots
            | VulkanKernelDescriptorUsage::StateRead => VulkanResidentKernelBufferAccess::Read,
            VulkanKernelDescriptorUsage::OutputSignal | VulkanKernelDescriptorUsage::StateWrite => {
                VulkanResidentKernelBufferAccess::Write
            }
            VulkanKernelDescriptorUsage::StateView
            | VulkanKernelDescriptorUsage::SelectionTelemetry => {
                VulkanResidentKernelBufferAccess::ReadWrite
            }
        };

        Ok(VulkanResidentKernelBufferBinding::new(binding, buffer, byte_len)
            .with_byte_offset(byte_offset)
            .with_access(access))
    }

    fn resident_kernel_buffer_for_resident_target<'a>(
        &'a self,
        dispatch: &VulkanMountedPlacedBoundDispatch,
        descriptor: &VulkanMountedPlacedBoundDescriptor,
        target: &VulkanBoundDescriptorTarget,
    ) -> Result<
        (&'a VulkanResidentBuffer, usize, usize),
        VulkanMountedPlacedResidentKernelDispatchError,
    >
    {
        match target {
            VulkanBoundDescriptorTarget::RuntimeControl {
                runtime_source,
                byte_capacity,
            } => {
                if runtime_source != "input_token_id" || *byte_capacity != size_of::<u32>() {
                    return Err(
                        VulkanMountedPlacedResidentKernelDispatchError::UnsupportedRuntimeControl {
                            dispatch_index: dispatch.dispatch_index,
                            binding: descriptor.binding,
                            runtime_source: runtime_source.clone(),
                            byte_capacity: *byte_capacity,
                        },
                    );
                }
                Ok((&self.stream_control_buffer, 0, *byte_capacity))
            }
            VulkanBoundDescriptorTarget::PermanentParameter {
                param_id,
                tensor,
                byte_count,
            } => {
                let allocation = self.parameter_buffers.parameter_buffer(tensor).ok_or_else(
                    || {
                        VulkanMountedPlacedResidentKernelDispatchError::MissingPermanentParameterBuffer {
                            dispatch_index: dispatch.dispatch_index,
                            binding: descriptor.binding,
                            param_id: param_id.clone(),
                            tensor: tensor.clone(),
                            byte_count: *byte_count,
                        }
                    },
                )?;
                Ok((
                    allocation.buffer.as_ref(),
                    allocation.byte_offset,
                    allocation.byte_capacity,
                ))
            }
            VulkanBoundDescriptorTarget::DynamicResourceAddressTable {
                ..
            } => {
                let resources = self.dynamic_resource_buffers.as_ref().ok_or(
                    VulkanMountedPlacedResidentKernelDispatchError::
                        MissingDynamicResourceBuffers {
                            dispatch_index: dispatch.dispatch_index,
                            binding: descriptor.binding,
                        },
                )?;
                let buffer = resources.address_table();
                Ok((buffer, 0, buffer.byte_capacity()))
            }
            VulkanBoundDescriptorTarget::DynamicResourceParameterSlots {
                component_id,
                node_id,
                selection_signal,
                ..
            } => {
                let resources = self.dynamic_resource_buffers.as_ref().ok_or(
                    VulkanMountedPlacedResidentKernelDispatchError::
                        MissingDynamicResourceBuffers {
                            dispatch_index: dispatch.dispatch_index,
                            binding: descriptor.binding,
                        },
                )?;
                let buffer = resources
                    .parameter_slots(
                        component_id,
                        node_id,
                        selection_signal,
                    )
                    .ok_or_else(|| {
                        VulkanMountedPlacedResidentKernelDispatchError::
                            MissingDynamicResourceParameterSlots {
                                dispatch_index: dispatch.dispatch_index,
                                binding: descriptor.binding,
                                component_id: component_id.clone(),
                                node_id: node_id.clone(),
                                selection_signal: selection_signal.clone(),
                            }
                    })?;
                Ok((buffer, 0, buffer.byte_capacity()))
            }
            VulkanBoundDescriptorTarget::BoundaryInput { signal_id }
            | VulkanBoundDescriptorTarget::BoundaryOutput { signal_id } => Err(
                VulkanMountedPlacedResidentKernelDispatchError::ModelBoundaryBufferUnavailable {
                    dispatch_index: dispatch.dispatch_index,
                    binding: descriptor.binding,
                    signal_id: signal_id.clone(),
                },
            ),
            VulkanBoundDescriptorTarget::ActivationSlot {
                buffer_index,
                byte_capacity,
                ..
            } => {
                let allocation = self
                    .buffers
                    .activation_slot_buffers
                    .get(*buffer_index)
                    .ok_or_else(|| {
                        VulkanMountedPlacedResidentKernelDispatchError::MissingMountedBuffer {
                            dispatch_index: dispatch.dispatch_index,
                            binding: descriptor.binding,
                            buffer_kind: "activation_slot".to_string(),
                            buffer_index: *buffer_index,
                        }
                    })?;
                Ok((&allocation.buffer, 0, *byte_capacity))
            }
            VulkanBoundDescriptorTarget::StreamStateBuffer {
                buffer_index,
                byte_capacity,
                ..
            }
            | VulkanBoundDescriptorTarget::StreamStateView {
                buffer_index,
                byte_capacity,
                ..
            } => {
                let allocation =
                    self.buffers
                        .state_buffers
                        .get(*buffer_index)
                        .ok_or_else(|| {
                            VulkanMountedPlacedResidentKernelDispatchError::MissingMountedBuffer {
                                dispatch_index: dispatch.dispatch_index,
                                binding: descriptor.binding,
                                buffer_kind: "stream_state".to_string(),
                                buffer_index: *buffer_index,
                            }
                        })?;
                Ok((&allocation.buffer, 0, *byte_capacity))
            }
            VulkanBoundDescriptorTarget::SelectionTelemetry {
                buffer_index,
                byte_capacity,
                ..
            } => {
                let allocation = self
                    .buffers
                    .selection_telemetry_buffers
                    .get(*buffer_index)
                    .ok_or_else(|| {
                        VulkanMountedPlacedResidentKernelDispatchError::MissingMountedBuffer {
                            dispatch_index: dispatch.dispatch_index,
                            binding: descriptor.binding,
                            buffer_kind: "selection_telemetry".to_string(),
                            buffer_index: *buffer_index,
                        }
                    })?;
                Ok((&allocation.buffer, 0, *byte_capacity))
            }
        }
    }
}
