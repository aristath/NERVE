impl RuntimeModelEditor {
    pub fn validate_all_instance_device_compatibility(
        &self,
        device_id: &str,
    ) -> Result<(), RuntimeEditorError> {
        for instance in self.draft.instances.iter().filter(|instance| instance.enabled) {
            self.validate_instance_device_compatibility(&instance.instance_id, device_id)?;
        }
        Ok(())
    }

    pub fn validate_instance_device_compatibility(
        &self,
        instance_id: &str,
        device_id: &str,
    ) -> Result<(), RuntimeEditorError> {
        let instance = self
            .draft
            .instances
            .iter()
            .find(|instance| instance.instance_id == instance_id)
            .ok_or_else(|| {
                RuntimeEditorError(format!(
                    "runtime graph has no node instance {instance_id:?}"
                ))
            })?;
        self.validate_source_component_device_compatibility(
            &instance.source_component_id,
            device_id,
        )
        .map_err(|error| {
            RuntimeEditorError(format!(
                "runtime device {device_id:?} cannot host instance {instance_id:?}: {error}"
            ))
        })
    }

    pub fn validate_source_component_device_compatibility(
        &self,
        source_component_id: &str,
        device_id: &str,
    ) -> Result<(), RuntimeEditorError> {
        if !self
            .source_components
            .iter()
            .any(|source| source.source_id == source_component_id)
        {
            return Err(RuntimeEditorError(format!(
                "runtime graph has no source component {source_component_id:?}"
            )));
        }
        let device = self
            .available_devices
            .iter()
            .find(|device| device.device_id == device_id)
            .ok_or_else(|| RuntimeEditorError(format!("runtime device {device_id:?} is unknown")))?;
        if !device.available
            || device.can_host_runtime_components_on_physical_device == Some(false)
        {
            return Err(RuntimeEditorError(format!(
                "runtime device {device_id:?} is unavailable or cannot host runtime components"
            )));
        }
        let Some(profile) = &device.hardware_profile else {
            return Ok(());
        };
        crate::validate_vulkan_package_source_component_hardware_compatibility(
            &self.package_root,
            &self.manifest,
            source_component_id,
            profile,
        )
        .map_err(RuntimeEditorError)
    }
}
