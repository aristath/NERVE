impl VulkanResidentInProcessPlacedModelPackage {
    pub fn from_manifest_file(
        device: &VulkanComputeDevice,
        manifest_path: impl AsRef<Path>,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        Self::from_manifest_file_with_capacity(device, manifest_path, None)
    }

    pub fn from_manifest_file_with_capacity(
        device: &VulkanComputeDevice,
        manifest_path: impl AsRef<Path>,
        dynamic_state_capacity_activations: Option<usize>,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        let manifest_path = manifest_path.as_ref();
        let manifest =
            VulkanResidentModelPackageManifest::from_json_file(manifest_path).map_err(|error| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(format!(
                        "failed to load resident placed model package manifest {:?}: {error}",
                        manifest_path
                    )),
                )
            })?;
        let manifest_dir = manifest_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let runtime_model = manifest
            .mount_runtime_graph_controls(None, &BTreeMap::new(), &[], None)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?;
        Self::from_runtime_model_for_devices(
            device,
            &manifest_dir,
            runtime_model,
            dynamic_state_capacity_activations,
        )
    }

    pub fn from_runtime_model_for_devices(
        device: &VulkanComputeDevice,
        manifest_dir: impl AsRef<Path>,
        runtime_model: VulkanResidentRuntimeModel,
        dynamic_state_capacity_activations: Option<usize>,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        Self::from_runtime_model_for_device_resolver(
            manifest_dir,
            runtime_model,
            None,
            None,
            dynamic_state_capacity_activations,
            1,
            ResourceResidencyPolicy::Eager,
            None,
            None,
            |_| Ok(device),
        )
    }

    pub fn from_runtime_model_for_bound_devices(
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        manifest_dir: impl AsRef<Path>,
        runtime_model: VulkanResidentRuntimeModel,
        dynamic_state_capacity_activations: Option<usize>,
        speculative_draft_tokens: usize,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        Self::from_runtime_model_for_device_resolver(
            manifest_dir,
            runtime_model,
            None,
            None,
            dynamic_state_capacity_activations,
            speculative_draft_tokens,
            ResourceResidencyPolicy::Eager,
            None,
            None,
            |device_id| {
                devices
                    .get(device_id)
                    .map(|device| device.as_ref())
                    .ok_or_else(
                        || VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                            device_id: device_id.to_string(),
                        },
                    )
            },
        )
    }

    pub fn from_runtime_model_for_bound_devices_with_residency_policy(
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        manifest_dir: impl AsRef<Path>,
        runtime_model: VulkanResidentRuntimeModel,
        dynamic_state_capacity_activations: Option<usize>,
        speculative_draft_tokens: usize,
        resource_residency_policy: ResourceResidencyPolicy,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        Self::from_runtime_model_for_device_resolver(
            manifest_dir,
            runtime_model,
            None,
            None,
            dynamic_state_capacity_activations,
            speculative_draft_tokens,
            resource_residency_policy,
            None,
            None,
            |device_id| {
                devices
                    .get(device_id)
                    .map(|device| device.as_ref())
                    .ok_or_else(
                        || VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                            device_id: device_id.to_string(),
                        },
                    )
            },
        )
    }

    pub fn from_runtime_model_for_bound_devices_with_parameter_pool(
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        manifest_dir: impl AsRef<Path>,
        runtime_model: VulkanResidentRuntimeModel,
        dynamic_state_capacity_activations: Option<usize>,
        speculative_draft_tokens: usize,
        resource_residency_policy: ResourceResidencyPolicy,
        parameter_pool: &VulkanResidentBufferPool,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        for (device_id, device) in devices {
            parameter_pool
                .register_device(device_id, device.clone())
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        }
        Self::from_runtime_model_for_device_resolver(
            manifest_dir,
            runtime_model,
            None,
            None,
            dynamic_state_capacity_activations,
            speculative_draft_tokens,
            resource_residency_policy,
            Some(parameter_pool),
            None,
            |device_id| {
                devices
                    .get(device_id)
                    .map(|device| device.as_ref())
                    .ok_or_else(
                        || VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                            device_id: device_id.to_string(),
                        },
                    )
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_runtime_model_for_bound_devices_with_parameter_pool_and_retained_stores(
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        manifest_dir: impl AsRef<Path>,
        runtime_model: VulkanResidentRuntimeModel,
        dynamic_state_capacity_activations: Option<usize>,
        speculative_draft_tokens: usize,
        resource_residency_policy: ResourceResidencyPolicy,
        parameter_pool: &VulkanResidentBufferPool,
        retained_stores: &VulkanRetainedCompiledResourceStores,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        for (device_id, device) in devices {
            parameter_pool
                .register_device(device_id, device.clone())
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        }
        Self::from_runtime_model_for_device_resolver(
            manifest_dir,
            runtime_model,
            None,
            None,
            dynamic_state_capacity_activations,
            speculative_draft_tokens,
            resource_residency_policy,
            Some(parameter_pool),
            Some(retained_stores),
            |device_id| {
                devices
                    .get(device_id)
                    .map(|device| device.as_ref())
                    .ok_or_else(
                        || VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                            device_id: device_id.to_string(),
                        },
                    )
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_runtime_model_for_bound_devices_with_physical_execution_plan(
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        manifest_dir: impl AsRef<Path>,
        runtime_model: VulkanResidentRuntimeModel,
        physical_execution_plan: VulkanRuntimePhysicalExecutionPlan,
        placement_calibration_catalog: Option<&VulkanPlacementCalibrationCatalog>,
        dynamic_state_capacity_activations: Option<usize>,
        speculative_draft_tokens: usize,
        resource_residency_policy: ResourceResidencyPolicy,
        parameter_pool: &VulkanResidentBufferPool,
        retained_stores: Option<&VulkanRetainedCompiledResourceStores>,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        for (device_id, device) in devices {
            parameter_pool
                .register_device(device_id, device.clone())
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        }
        Self::from_runtime_model_for_device_resolver(
            manifest_dir,
            runtime_model,
            Some(physical_execution_plan),
            placement_calibration_catalog,
            dynamic_state_capacity_activations,
            speculative_draft_tokens,
            resource_residency_policy,
            Some(parameter_pool),
            retained_stores,
            |device_id| {
                devices
                    .get(device_id)
                    .map(|device| device.as_ref())
                    .ok_or_else(
                        || VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                            device_id: device_id.to_string(),
                        },
                    )
            },
        )
    }
}
