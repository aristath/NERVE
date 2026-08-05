#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapacityPackedPlacementComponent {
    pub component_id: String,
    pub resident_weight_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapacityPackedPlacementDevice {
    pub device_id: String,
    pub capacity_bytes: usize,
}

/// Assigns an ordered component chain to ordered devices while minimizing
/// physical boundaries. Every device except the last is filled with the
/// longest contiguous prefix that fits its measured capacity. Callers choose
/// the device order according to execution and transfer cost; identifiers are
/// deliberately never sorted here.
pub fn capacity_packed_component_placement(
    components: &[CapacityPackedPlacementComponent],
    devices: &[CapacityPackedPlacementDevice],
) -> Result<BTreeMap<String, String>, CircuitPlacementError> {
    if components.is_empty() {
        return Err(CircuitPlacementError(
            "capacity-packed placement requires components".to_string(),
        ));
    }
    if devices.is_empty() {
        return Err(CircuitPlacementError(
            "capacity-packed placement requires devices".to_string(),
        ));
    }
    if devices.len() > components.len() {
        return Err(CircuitPlacementError(format!(
            "capacity-packed placement cannot assign {} devices to only {} components",
            devices.len(),
            components.len(),
        )));
    }
    let mut component_ids = BTreeSet::new();
    for component in components {
        if component.component_id.is_empty()
            || !component_ids.insert(component.component_id.as_str())
        {
            return Err(CircuitPlacementError(
                "capacity-packed placement component ids must be nonempty and unique"
                    .to_string(),
            ));
        }
    }
    let mut device_ids = BTreeSet::new();
    for device in devices {
        if device.device_id.is_empty()
            || device.capacity_bytes == 0
            || !device_ids.insert(device.device_id.as_str())
        {
            return Err(CircuitPlacementError(
                "capacity-packed placement devices must have unique nonempty ids and positive capacities"
                    .to_string(),
            ));
        }
    }

    let mut placement = BTreeMap::new();
    let mut cursor = 0usize;
    for (device_index, device) in devices.iter().enumerate() {
        let remaining_devices = devices.len() - device_index;
        if device_index + 1 == devices.len() {
            let required = components[cursor..].iter().try_fold(
                0usize,
                |total, component| {
                    total.checked_add(component.resident_weight_bytes).ok_or_else(|| {
                        CircuitPlacementError(
                            "capacity-packed component weights overflow usize".to_string(),
                        )
                    })
                },
            )?;
            if required > device.capacity_bytes {
                return Err(CircuitPlacementError(format!(
                    "capacity-packed final segment requires {required} bytes on device {:?}, which has {} bytes",
                    device.device_id, device.capacity_bytes,
                )));
            }
            for component in &components[cursor..] {
                placement.insert(component.component_id.clone(), device.device_id.clone());
            }
            break;
        }

        let mut assigned_bytes = 0usize;
        let mut assigned_component_count = 0usize;
        while cursor < components.len() {
            let components_after = components.len() - cursor - 1;
            if components_after < remaining_devices - 1 {
                break;
            }
            let component = &components[cursor];
            let next_bytes = assigned_bytes
                .checked_add(component.resident_weight_bytes)
                .ok_or_else(|| {
                    CircuitPlacementError(
                        "capacity-packed component weights overflow usize".to_string(),
                    )
                })?;
            if next_bytes > device.capacity_bytes {
                break;
            }
            placement.insert(component.component_id.clone(), device.device_id.clone());
            assigned_bytes = next_bytes;
            assigned_component_count += 1;
            cursor += 1;
        }
        if assigned_component_count == 0 {
            let component = &components[cursor];
            return Err(CircuitPlacementError(format!(
                "component {:?} requires {} bytes but device {:?} has only {} bytes",
                component.component_id,
                component.resident_weight_bytes,
                device.device_id,
                device.capacity_bytes,
            )));
        }
    }
    if placement.len() != components.len() {
        return Err(CircuitPlacementError(
            "capacity-packed placement omitted components".to_string(),
        ));
    }
    Ok(placement)
}
