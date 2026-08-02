#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanDeviceAddressRange {
    owner_id: u64,
    start: vk::DeviceAddress,
    byte_capacity: usize,
    label: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanResolvedDeviceAddress {
    label: String,
    byte_offset: usize,
    byte_capacity: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDeviceFaultAddressReport {
    pub address_type: i32,
    pub reported_address: u64,
    pub address_precision: u64,
    pub allocation: Option<String>,
    pub allocation_byte_offset: Option<usize>,
    pub allocation_byte_capacity: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDeviceFaultReport {
    pub description: String,
    pub addresses: Vec<VulkanDeviceFaultAddressReport>,
    pub vendor_descriptions: Vec<String>,
}

#[derive(Default)]
struct VulkanDeviceAddressRegistry {
    ranges: BTreeMap<vk::DeviceAddress, VulkanDeviceAddressRange>,
}

impl VulkanDeviceAddressRegistry {
    fn register(
        &mut self,
        owner_id: u64,
        start: vk::DeviceAddress,
        byte_capacity: usize,
        label: impl Into<String>,
    ) -> Result<(), VulkanError> {
        if start == 0 || byte_capacity == 0 {
            return Err(VulkanError(
                "device-address registry range must be non-empty".to_string(),
            ));
        }
        let end = start
            .checked_add(u64::try_from(byte_capacity).map_err(|_| {
                VulkanError("device-address range exceeds u64".to_string())
            })?)
            .ok_or_else(|| VulkanError("device-address range overflowed".to_string()))?;
        if let Some(previous) = self.ranges.range(..=start).next_back().map(|(_, range)| range) {
            let previous_end = previous
                .start
                .checked_add(u64::try_from(previous.byte_capacity).unwrap_or(u64::MAX))
                .unwrap_or(u64::MAX);
            if previous_end > start {
                return Err(VulkanError(format!(
                    "device-address range 0x{start:x}..0x{end:x} overlaps {:?}",
                    previous.label
                )));
            }
        }
        if let Some(next) = self.ranges.range(start..).next().map(|(_, range)| range)
            && next.start < end
        {
            return Err(VulkanError(format!(
                "device-address range 0x{start:x}..0x{end:x} overlaps {:?}",
                next.label
            )));
        }
        self.ranges.insert(
            start,
            VulkanDeviceAddressRange {
                owner_id,
                start,
                byte_capacity,
                label: label.into(),
            },
        );
        Ok(())
    }

    fn unregister(
        &mut self,
        owner_id: u64,
        start: vk::DeviceAddress,
    ) -> Result<(), VulkanError> {
        let range = self.ranges.get(&start).ok_or_else(|| {
            VulkanError(format!(
                "device-address registry has no allocation at 0x{start:x}"
            ))
        })?;
        if range.owner_id != owner_id {
            return Err(VulkanError(format!(
                "device-address allocation at 0x{start:x} belongs to another owner"
            )));
        }
        self.ranges.remove(&start);
        Ok(())
    }

    fn resolve(
        &self,
        address: vk::DeviceAddress,
    ) -> Option<VulkanResolvedDeviceAddress> {
        let range = self.ranges.range(..=address).next_back()?.1;
        let byte_offset = usize::try_from(address.checked_sub(range.start)?).ok()?;
        (byte_offset < range.byte_capacity).then(|| VulkanResolvedDeviceAddress {
            label: range.label.clone(),
            byte_offset,
            byte_capacity: range.byte_capacity,
        })
    }
}

impl VulkanComputeDevice {
    pub fn supports_device_fault_reporting(&self) -> bool {
        self.device_fault.is_some()
    }

    pub fn device_fault_report(&self) -> Result<Option<VulkanDeviceFaultReport>, VulkanError> {
        query_vulkan_device_fault(
            self.device_fault.as_ref(),
            &self.device_address_registry,
        )
    }

    fn vulkan_operation_error(&self, context: &str, error: vk::Result) -> VulkanError {
        if error != vk::Result::ERROR_DEVICE_LOST {
            return VulkanError(format!("{context}: {error:?}"));
        }
        match self.device_fault_report() {
            Ok(Some(report)) => VulkanError(format!(
                "{context}: {error:?}; device_fault={report:?}"
            )),
            Ok(None) => VulkanError(format!(
                "{context}: {error:?}; VK_EXT_device_fault is unavailable"
            )),
            Err(fault_error) => VulkanError(format!(
                "{context}: {error:?}; device-fault query failed: {fault_error}"
            )),
        }
    }

    fn track_addressable_buffer(
        &self,
        mut buffer: VulkanResidentBuffer,
        kind: &str,
    ) -> Result<VulkanResidentBuffer, VulkanError> {
        let address = buffer.device_address()?;
        let owner_id = buffer.buffer.as_raw();
        let label = format!("{kind} handle=0x{owner_id:x}");
        self.device_address_registry
            .lock()
            .map_err(|_| VulkanError("device-address registry was poisoned".to_string()))?
            .register(owner_id, address, buffer.byte_capacity(), label)?;
        buffer.device_address_registry = Some(Arc::clone(&self.device_address_registry));
        Ok(buffer)
    }
}

fn query_vulkan_device_fault(
    extension: Option<&ash::ext::device_fault::Device>,
    registry: &Arc<Mutex<VulkanDeviceAddressRegistry>>,
) -> Result<Option<VulkanDeviceFaultReport>, VulkanError> {
    let Some(extension) = extension else {
        return Ok(None);
    };
    let mut counts = vk::DeviceFaultCountsEXT::default();
    let first = unsafe {
        (extension.fp().get_device_fault_info_ext)(
            extension.device(),
            &mut counts,
            std::ptr::null_mut(),
        )
    };
    if first != vk::Result::SUCCESS {
        return Err(VulkanError(format!(
            "failed to query Vulkan device-fault counts: {first:?}"
        )));
    }
    let address_capacity = counts.address_info_count as usize;
    let vendor_capacity = counts.vendor_info_count as usize;
    let mut addresses = vec![vk::DeviceFaultAddressInfoEXT::default(); address_capacity];
    let mut vendor_infos = vec![vk::DeviceFaultVendorInfoEXT::default(); vendor_capacity];
    counts.vendor_binary_size = 0;
    let mut info = vk::DeviceFaultInfoEXT::default();
    info.p_address_infos = addresses.as_mut_ptr();
    info.p_vendor_infos = vendor_infos.as_mut_ptr();
    let second = unsafe {
        (extension.fp().get_device_fault_info_ext)(
            extension.device(),
            &mut counts,
            &mut info,
        )
    };
    if second != vk::Result::SUCCESS && second != vk::Result::INCOMPLETE {
        return Err(VulkanError(format!(
            "failed to query Vulkan device-fault details: {second:?}"
        )));
    }
    addresses.truncate((counts.address_info_count as usize).min(address_capacity));
    vendor_infos.truncate((counts.vendor_info_count as usize).min(vendor_capacity));
    let registry = registry
        .lock()
        .map_err(|_| VulkanError("device-address registry was poisoned".to_string()))?;
    let addresses = addresses
        .into_iter()
        .map(|address| {
            let resolved = registry.resolve(address.reported_address);
            VulkanDeviceFaultAddressReport {
                address_type: address.address_type.as_raw(),
                reported_address: address.reported_address,
                address_precision: address.address_precision,
                allocation: resolved.as_ref().map(|range| range.label.clone()),
                allocation_byte_offset: resolved.as_ref().map(|range| range.byte_offset),
                allocation_byte_capacity: resolved.map(|range| range.byte_capacity),
            }
        })
        .collect();
    let description = info
        .description_as_c_str()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_else(|_| "invalid device-fault description".to_string());
    let vendor_descriptions = vendor_infos
        .iter()
        .map(|vendor| {
            vendor
                .description_as_c_str()
                .map(|value| value.to_string_lossy().into_owned())
                .unwrap_or_else(|_| "invalid vendor-fault description".to_string())
        })
        .collect();
    Ok(Some(VulkanDeviceFaultReport {
        description,
        addresses,
        vendor_descriptions,
    }))
}

impl VulkanResidentQueueSubmitter {
    fn vulkan_operation_error(&self, context: &str, error: vk::Result) -> VulkanError {
        if error != vk::Result::ERROR_DEVICE_LOST {
            return VulkanError(format!("{context}: {error:?}"));
        }
        match query_vulkan_device_fault(
            self.device_fault.as_ref(),
            &self.device_address_registry,
        ) {
            Ok(Some(report)) => VulkanError(format!(
                "{context}: {error:?}; device_fault={report:?}"
            )),
            Ok(None) => VulkanError(format!(
                "{context}: {error:?}; VK_EXT_device_fault is unavailable"
            )),
            Err(fault_error) => VulkanError(format!(
                "{context}: {error:?}; device-fault query failed: {fault_error}"
            )),
        }
    }
}
