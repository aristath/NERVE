const VULKAN_DEVICE_ADDRESS_RETIREMENT_HISTORY_CAPACITY: usize = 65_536;

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
struct VulkanNearestDeviceAddress {
    canonical_address: vk::DeviceAddress,
    label: String,
    signed_byte_offset: i128,
    byte_capacity: usize,
    gap_bytes: u128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDeviceFaultAddressReport {
    pub address_type: i32,
    pub reported_address: u64,
    pub address_precision: u64,
    pub resolved_address: Option<u64>,
    pub allocation: Option<String>,
    pub allocation_byte_offset: Option<usize>,
    pub allocation_byte_capacity: Option<usize>,
    pub nearest_allocation: Option<String>,
    pub nearest_allocation_signed_byte_offset: Option<i128>,
    pub nearest_allocation_byte_capacity: Option<usize>,
    pub nearest_allocation_gap_bytes: Option<u128>,
    pub retired_allocation: Option<String>,
    pub retired_allocation_byte_offset: Option<usize>,
    pub retired_allocation_byte_capacity: Option<usize>,
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
    annotations: BTreeMap<vk::DeviceAddress, VulkanDeviceAddressRange>,
    retired_ranges: std::collections::VecDeque<VulkanDeviceAddressRange>,
    retired_annotations: std::collections::VecDeque<VulkanDeviceAddressRange>,
}

impl VulkanDeviceAddressRegistry {
    fn register(
        &mut self,
        owner_id: u64,
        start: vk::DeviceAddress,
        byte_capacity: usize,
        label: impl Into<String>,
    ) -> Result<(), VulkanError> {
        register_device_address_range(
            &mut self.ranges,
            owner_id,
            start,
            byte_capacity,
            label,
        )
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
        let range = self
            .ranges
            .remove(&start)
            .expect("validated device-address allocation remained registered");
        record_retired_device_address_range(&mut self.retired_ranges, range);
        Ok(())
    }

    fn register_annotation(
        &mut self,
        owner_id: u64,
        start: vk::DeviceAddress,
        byte_capacity: usize,
        label: impl Into<String>,
    ) -> Result<(), VulkanError> {
        let end = checked_device_address_range_end(start, byte_capacity)?;
        let parent = resolve_device_address_range(&self.ranges, start).ok_or_else(|| {
            VulkanError(format!(
                "device-address annotation 0x{start:x}..0x{end:x} is not contained in a registered allocation"
            ))
        })?;
        let parent_end = checked_device_address_range_end(parent.start, parent.byte_capacity)?;
        if end > parent_end {
            return Err(VulkanError(format!(
                "device-address annotation 0x{start:x}..0x{end:x} is not contained in registered allocation {:?}",
                parent.label
            )));
        }
        register_device_address_range(
            &mut self.annotations,
            owner_id,
            start,
            byte_capacity,
            label,
        )
    }

    fn unregister_annotation(
        &mut self,
        owner_id: u64,
        start: vk::DeviceAddress,
    ) -> Result<(), VulkanError> {
        let range = self.annotations.get(&start).ok_or_else(|| {
            VulkanError(format!(
                "device-address registry has no annotation at 0x{start:x}"
            ))
        })?;
        if range.owner_id != owner_id {
            return Err(VulkanError(format!(
                "device-address annotation at 0x{start:x} belongs to another owner"
            )));
        }
        let range = self
            .annotations
            .remove(&start)
            .expect("validated device-address annotation remained registered");
        record_retired_device_address_range(&mut self.retired_annotations, range);
        Ok(())
    }

    fn resolve(
        &self,
        address: vk::DeviceAddress,
    ) -> Option<VulkanResolvedDeviceAddress> {
        let range = resolve_device_address_range(&self.annotations, address)
            .or_else(|| resolve_device_address_range(&self.ranges, address))?;
        let byte_offset = usize::try_from(address.checked_sub(range.start)?).ok()?;
        Some(VulkanResolvedDeviceAddress {
            label: range.label.clone(),
            byte_offset,
            byte_capacity: range.byte_capacity,
        })
    }

    fn resolve_reported_fault_address(
        &self,
        reported_address: vk::DeviceAddress,
    ) -> Option<(vk::DeviceAddress, VulkanResolvedDeviceAddress)> {
        if let Some(resolved) = self.resolve(reported_address) {
            return Some((reported_address, resolved));
        }

        let mut match_found = None;
        for address_bit_count in 32..64 {
            let low_mask = (1u64 << address_bit_count) - 1;
            let high_mask = !low_mask;
            if reported_address & high_mask != high_mask {
                continue;
            }
            let canonical_address = reported_address & low_mask;
            let Some(resolved) = self.resolve(canonical_address) else {
                continue;
            };
            match &match_found {
                Some((existing_address, existing))
                    if *existing_address != canonical_address || *existing != resolved =>
                {
                    return None;
                }
                Some(_) => {}
                None => match_found = Some((canonical_address, resolved)),
            }
        }
        match_found
    }

    fn resolve_retired_reported_fault_address(
        &self,
        reported_address: vk::DeviceAddress,
    ) -> Option<(vk::DeviceAddress, VulkanResolvedDeviceAddress)> {
        let mut match_found = None;
        for canonical_address in reported_device_address_candidates(reported_address) {
            let range = self
                .retired_annotations
                .iter()
                .rev()
                .find(|range| device_address_range_contains(range, canonical_address))
                .or_else(|| {
                    self.retired_ranges
                        .iter()
                        .rev()
                        .find(|range| device_address_range_contains(range, canonical_address))
                });
            let Some(range) = range else {
                continue;
            };
            let resolved = VulkanResolvedDeviceAddress {
                label: range.label.clone(),
                byte_offset: usize::try_from(canonical_address - range.start).ok()?,
                byte_capacity: range.byte_capacity,
            };
            match &match_found {
                Some((existing_address, existing))
                    if *existing_address != canonical_address || *existing != resolved =>
                {
                    return None;
                }
                Some(_) => {}
                None => match_found = Some((canonical_address, resolved)),
            }
        }
        match_found
    }

    fn nearest_reported_fault_address(
        &self,
        reported_address: vk::DeviceAddress,
    ) -> Option<VulkanNearestDeviceAddress> {
        let mut best = None;
        for canonical_address in reported_device_address_candidates(reported_address) {
            for range in self.annotations.values().chain(self.ranges.values()) {
                let signed_byte_offset = i128::from(canonical_address)
                    - i128::from(range.start);
                let byte_capacity = i128::try_from(range.byte_capacity).ok()?;
                let gap_bytes = if signed_byte_offset < 0 {
                    signed_byte_offset.unsigned_abs()
                } else if signed_byte_offset >= byte_capacity {
                    u128::try_from(signed_byte_offset - byte_capacity + 1).ok()?
                } else {
                    0
                };
                let candidate = VulkanNearestDeviceAddress {
                    canonical_address,
                    label: range.label.clone(),
                    signed_byte_offset,
                    byte_capacity: range.byte_capacity,
                    gap_bytes,
                };
                if best.as_ref().is_none_or(|current: &VulkanNearestDeviceAddress| {
                    (candidate.gap_bytes, candidate.byte_capacity)
                        < (current.gap_bytes, current.byte_capacity)
                }) {
                    best = Some(candidate);
                }
            }
        }
        best
    }
}

fn reported_device_address_candidates(
    reported_address: vk::DeviceAddress,
) -> BTreeSet<vk::DeviceAddress> {
    let mut candidates = BTreeSet::from([reported_address]);
    for address_bit_count in 32..64 {
        let low_mask = (1u64 << address_bit_count) - 1;
        let high_mask = !low_mask;
        if reported_address & high_mask == high_mask {
            candidates.insert(reported_address & low_mask);
        }
    }
    candidates
}

fn record_retired_device_address_range(
    history: &mut std::collections::VecDeque<VulkanDeviceAddressRange>,
    range: VulkanDeviceAddressRange,
) {
    if history.len() == VULKAN_DEVICE_ADDRESS_RETIREMENT_HISTORY_CAPACITY {
        history.pop_front();
    }
    history.push_back(range);
}

fn checked_device_address_range_end(
    start: vk::DeviceAddress,
    byte_capacity: usize,
) -> Result<vk::DeviceAddress, VulkanError> {
    if start == 0 || byte_capacity == 0 {
        return Err(VulkanError(
            "device-address registry range must be non-empty".to_string(),
        ));
    }
    start
        .checked_add(
            u64::try_from(byte_capacity)
                .map_err(|_| VulkanError("device-address range exceeds u64".to_string()))?,
        )
        .ok_or_else(|| VulkanError("device-address range overflowed".to_string()))
}

fn resolve_device_address_range(
    ranges: &BTreeMap<vk::DeviceAddress, VulkanDeviceAddressRange>,
    address: vk::DeviceAddress,
) -> Option<&VulkanDeviceAddressRange> {
    let range = ranges.range(..=address).next_back()?.1;
    let byte_offset = usize::try_from(address.checked_sub(range.start)?).ok()?;
    (byte_offset < range.byte_capacity).then_some(range)
}

fn device_address_range_contains(
    range: &VulkanDeviceAddressRange,
    address: vk::DeviceAddress,
) -> bool {
    address
        .checked_sub(range.start)
        .and_then(|offset| usize::try_from(offset).ok())
        .is_some_and(|offset| offset < range.byte_capacity)
}

fn register_device_address_range(
    ranges: &mut BTreeMap<vk::DeviceAddress, VulkanDeviceAddressRange>,
    owner_id: u64,
    start: vk::DeviceAddress,
    byte_capacity: usize,
    label: impl Into<String>,
) -> Result<(), VulkanError> {
    let end = checked_device_address_range_end(start, byte_capacity)?;
    if let Some(previous) = ranges.range(..=start).next_back().map(|(_, range)| range) {
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
    if let Some(next) = ranges.range(start..).next().map(|(_, range)| range)
        && next.start < end
    {
        return Err(VulkanError(format!(
            "device-address range 0x{start:x}..0x{end:x} overlaps {:?}",
            next.label
        )));
    }
    ranges.insert(
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
        vulkan_operation_error_with_device_fault(
            context,
            error,
            self.device_fault.as_ref(),
            &self.device_address_registry,
        )
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

fn vulkan_operation_error_with_device_fault(
    context: &str,
    error: vk::Result,
    extension: Option<&ash::ext::device_fault::Device>,
    registry: &Arc<Mutex<VulkanDeviceAddressRegistry>>,
) -> VulkanError {
        if error != vk::Result::ERROR_DEVICE_LOST {
            return VulkanError(format!("{context}: {error:?}"));
        }
        match query_vulkan_device_fault(extension, registry) {
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
            let resolved = registry.resolve_reported_fault_address(address.reported_address);
            let retired = resolved.is_none().then(|| {
                registry.resolve_retired_reported_fault_address(address.reported_address)
            }).flatten();
            let nearest = resolved.is_none().then(|| {
                registry.nearest_reported_fault_address(address.reported_address)
            }).flatten();
            VulkanDeviceFaultAddressReport {
                address_type: address.address_type.as_raw(),
                reported_address: address.reported_address,
                address_precision: address.address_precision,
                resolved_address: resolved.as_ref().map(|(address, _)| *address),
                allocation: resolved.as_ref().map(|(_, range)| range.label.clone()),
                allocation_byte_offset: resolved
                    .as_ref()
                    .map(|(_, range)| range.byte_offset),
                allocation_byte_capacity: resolved.map(|(_, range)| range.byte_capacity),
                nearest_allocation: nearest.as_ref().map(|range| range.label.clone()),
                nearest_allocation_signed_byte_offset: nearest
                    .as_ref()
                    .map(|range| range.signed_byte_offset),
                nearest_allocation_byte_capacity: nearest
                    .as_ref()
                    .map(|range| range.byte_capacity),
                nearest_allocation_gap_bytes: nearest.map(|range| range.gap_bytes),
                retired_allocation: retired.as_ref().map(|(_, range)| range.label.clone()),
                retired_allocation_byte_offset: retired
                    .as_ref()
                    .map(|(_, range)| range.byte_offset),
                retired_allocation_byte_capacity: retired
                    .map(|(_, range)| range.byte_capacity),
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
        vulkan_operation_error_with_device_fault(
            context,
            error,
            self.device_fault.as_ref(),
            &self.device_address_registry,
        )
    }
}
