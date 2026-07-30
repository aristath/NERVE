impl VulkanResidentModelPackageDeviceSlice {
    pub fn from_manifest_file_for_device(
        device: &VulkanComputeDevice,
        manifest_path: impl AsRef<Path>,
        device_id: impl AsRef<str>,
        dynamic_state_capacity_activations: Option<usize>,
    ) -> Result<Self, VulkanResidentTokenModelPackageError> {
        let manifest_path = manifest_path.as_ref();
        let manifest =
            VulkanResidentModelPackageManifest::from_json_file(manifest_path).map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to load resident model package manifest {:?}: {error}",
                    manifest_path
                ))
            })?;
        let manifest_dir = manifest_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let runtime_model =
            manifest.mount_runtime_graph_controls(None, &BTreeMap::new(), &[], None)?;
        Self::from_runtime_model_for_device(
            device,
            &manifest_dir,
            runtime_model,
            device_id,
            dynamic_state_capacity_activations,
        )
    }

    pub fn from_runtime_model_for_device(
        device: &VulkanComputeDevice,
        manifest_dir: impl AsRef<Path>,
        runtime_model: VulkanResidentRuntimeModel,
        device_id: impl AsRef<str>,
        dynamic_state_capacity_activations: Option<usize>,
    ) -> Result<Self, VulkanResidentTokenModelPackageError> {
        Self::from_runtime_model_for_device_with_optional_parameter_pool(
            device,
            manifest_dir,
            runtime_model,
            device_id,
            dynamic_state_capacity_activations,
            None,
        )
    }

    pub fn from_runtime_model_for_device_with_parameter_pool(
        device: &VulkanComputeDevice,
        manifest_dir: impl AsRef<Path>,
        runtime_model: VulkanResidentRuntimeModel,
        device_id: impl AsRef<str>,
        dynamic_state_capacity_activations: Option<usize>,
        parameter_pool: &VulkanResidentBufferPool,
    ) -> Result<Self, VulkanResidentTokenModelPackageError> {
        Self::from_runtime_model_for_device_with_optional_parameter_pool(
            device,
            manifest_dir,
            runtime_model,
            device_id,
            dynamic_state_capacity_activations,
            Some(parameter_pool),
        )
    }

    fn from_runtime_model_for_device_with_optional_parameter_pool(
        device: &VulkanComputeDevice,
        manifest_dir: impl AsRef<Path>,
        runtime_model: VulkanResidentRuntimeModel,
        device_id: impl AsRef<str>,
        dynamic_state_capacity_activations: Option<usize>,
        parameter_pool: Option<&VulkanResidentBufferPool>,
    ) -> Result<Self, VulkanResidentTokenModelPackageError> {
        let manifest_dir = manifest_dir.as_ref();
        let device_id = device_id.as_ref();
        let capacity = dynamic_state_capacity_activations
            .unwrap_or(runtime_model.package.max_context_activations);
        let tensor_index = runtime_model.load_runtime_tensor_index(manifest_dir)?;
        let plan = VulkanResidentModelPackageDeviceSlicePlan::prepare(
            device,
            manifest_dir,
            &runtime_model,
            &tensor_index,
            device_id,
            capacity,
        )?;
        plan.materialize(
            device,
            &tensor_index,
            &BTreeSet::new(),
            parameter_pool,
        )
    }

    pub fn create_mounted_stream_circuit(
        &self,
        device: &VulkanComputeDevice,
    ) -> Result<VulkanMountedPlacedStreamCircuit, VulkanResidentTokenModelPackageError> {
        self.create_mounted_stream_circuit_with_activation_overrides(device, &[])
    }

    pub fn create_mounted_stream_circuit_with_activation_overrides(
        &self,
        device: &VulkanComputeDevice,
        activation_overrides: &[VulkanActivationSlotBufferOverride],
    ) -> Result<VulkanMountedPlacedStreamCircuit, VulkanResidentTokenModelPackageError> {
        self.create_mounted_stream_circuit_with_buffer_overrides(
            device,
            activation_overrides,
            &[],
            None,
        )
    }

    pub fn create_mounted_stream_circuit_with_buffer_overrides(
        &self,
        device: &VulkanComputeDevice,
        activation_overrides: &[VulkanActivationSlotBufferOverride],
        edge_endpoint_overrides: &[VulkanPlacedEdgeEndpointBufferOverride],
        stream_control_override: Option<Arc<VulkanResidentBuffer>>,
    ) -> Result<VulkanMountedPlacedStreamCircuit, VulkanResidentTokenModelPackageError> {
        self.create_mounted_stream_circuit_with_all_buffer_overrides(
            device,
            activation_overrides,
            &[],
            edge_endpoint_overrides,
            stream_control_override,
        )
    }

    pub fn create_mounted_stream_circuit_with_all_buffer_overrides(
        &self,
        device: &VulkanComputeDevice,
        activation_overrides: &[VulkanActivationSlotBufferOverride],
        local_edge_overrides: &[VulkanPlacedLocalEdgeBufferOverride],
        edge_endpoint_overrides: &[VulkanPlacedEdgeEndpointBufferOverride],
        stream_control_override: Option<Arc<VulkanResidentBuffer>>,
    ) -> Result<VulkanMountedPlacedStreamCircuit, VulkanResidentTokenModelPackageError> {
        VulkanMountedPlacedStreamCircuit::
            from_placed_plan_with_parameter_buffers_and_all_buffer_overrides(
            device,
            self.placed_plan.clone(),
            self.dynamic_state_capacity_activations,
            self.parameter_buffers.clone(),
            activation_overrides,
            local_edge_overrides,
            edge_endpoint_overrides,
            stream_control_override,
            self.dynamic_resource_buffers.clone(),
        )
        .map_err(|error| {
            VulkanResidentTokenModelPackageError::new(format!(
                "failed to mount Vulkan stream circuit for device {:?}: {error}",
                self.device_id
            ))
        })
    }

    pub fn loaded_manifest(&self) -> &VulkanLoadedReusableKernelArtifactManifest {
        &self.loaded_manifest
    }

    pub fn prepared_plan(&self) -> &VulkanPreparedDispatchPlan {
        &self.prepared_plan
    }

    pub fn physical_residency_schedule(&self) -> &VulkanPhysicalResidencySchedule {
        &self.physical_residency_schedule
    }
}
