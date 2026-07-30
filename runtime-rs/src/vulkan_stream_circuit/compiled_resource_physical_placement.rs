#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VulkanCompiledResourcePlacementAction {
    /// The resource is loaded into the physical device that executes the
    /// selecting component. A different physical device therefore represents
    /// an explicit additional resident copy, never implicit driver paging.
    LocalToExecutionDevice,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VulkanCompiledResourcePhysicalPlacement {
    pub store_id: String,
    pub physical_device_id: String,
    pub action: VulkanCompiledResourcePlacementAction,
    pub logical_device_ids: Vec<String>,
    pub executing_component_ids: Vec<String>,
    pub selector_ids: Vec<String>,
    pub selector_placements:
        Vec<VulkanCompiledResourceSelectorPhysicalPlacement>,
    pub maximum_dynamic_payload_bytes: usize,
    pub maximum_atomic_group_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VulkanCompiledResourceSelectorPhysicalPlacement {
    pub selector_id: String,
    pub cross_device_choice:
        Option<VulkanCompiledResourceCrossDeviceAccessChoice>,
    pub previously_resident_physical_device_ids: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VulkanCompiledResourceCrossDeviceAccessChoice {
    RemoteExecution,
    ActivationTransport,
    PeerTransfer,
    SecondResidentCopy,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanCompiledResourceCrossDeviceAccessRequest {
    pub selector_id: String,
    pub execution_physical_device_id: String,
    pub resident_physical_device_ids: Vec<String>,
}

pub fn require_explicit_compiled_resource_cross_device_choice(
    request: &VulkanCompiledResourceCrossDeviceAccessRequest,
    choice: Option<VulkanCompiledResourceCrossDeviceAccessChoice>,
) -> Result<VulkanCompiledResourceCrossDeviceAccessChoice, VulkanRuntimeResidencyPlanError> {
    if request.selector_id.trim().is_empty()
        || request.execution_physical_device_id.trim().is_empty()
        || request.resident_physical_device_ids.is_empty()
        || request
            .resident_physical_device_ids
            .iter()
            .any(|device_id| device_id.trim().is_empty())
    {
        return Err(VulkanRuntimeResidencyPlanError(
            "compiled resource cross-device request is invalid".to_string(),
        ));
    }
    choice.ok_or_else(|| {
        VulkanRuntimeResidencyPlanError(format!(
            "compiled selector {:?} is resident on {:?} but execution is placed on {:?}; choose remote execution, activation transport, peer transfer, or a second resident copy explicitly",
            request.selector_id,
            request.resident_physical_device_ids,
            request.execution_physical_device_id
        ))
    })
}

fn group_compiled_resource_logical_devices_by_physical(
    logical_devices: &[(String, &VulkanComputeDevice)],
) -> Result<Vec<Vec<String>>, VulkanRuntimeResidencyPlanError> {
    let mut groups = Vec::<Vec<String>>::new();
    for (logical_device_id, device) in logical_devices {
        if let Some(group) = groups.iter_mut().find(|group| {
            let representative_id = &group[0];
            let representative = logical_devices
                .iter()
                .find(|(candidate_id, _)| {
                    candidate_id == representative_id
                })
                .expect("physical group representative must remain indexed")
                .1;
            representative.shares_physical_device_with(device)
        }) {
            let representative = logical_devices
                .iter()
                .find(|(candidate_id, _)| candidate_id == &group[0])
                .expect("physical group representative must remain indexed")
                .1;
            if !representative.shares_logical_device_with(device) {
                return Err(VulkanRuntimeResidencyPlanError(format!(
                    "logical devices {:?} and {logical_device_id:?} target physical device {:?} through different Vulkan logical devices; bind every alias of one physical GPU to the same opened device",
                    group,
                    device.physical_device_id()
                )));
            }
            group.push(logical_device_id.clone());
        } else {
            groups.push(vec![logical_device_id.clone()]);
        }
    }
    Ok(groups)
}
