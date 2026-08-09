#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VulkanRuntimeSelectedResourcePressure {
    pub group_id: String,
    pub payload_bytes: usize,
    pub selection_count: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VulkanRuntimeComponentWorkingSetPressure {
    pub execution_scope: String,
    pub component_id: String,
    pub addressable_unit_count: usize,
    pub resident_unit_count: usize,
    pub selected_unit_count: usize,
    pub addressable_payload_bytes: usize,
    pub resident_payload_bytes: usize,
    pub selected_payload_bytes: usize,
    pub selection_count: u64,
    pub gpu_miss_count: u64,
    pub selected_resources: Vec<VulkanRuntimeSelectedResourcePressure>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VulkanRuntimeDeviceWorkingSetPressure {
    pub store_id: String,
    pub physical_device_id: String,
    pub logical_device_ids: Vec<String>,
    pub current_device_bytes: usize,
    pub maximum_device_bytes: usize,
    pub current_payload_bytes: usize,
    pub maximum_payload_bytes: usize,
    pub eviction_count: u64,
    pub reload_count: u64,
    pub blocking_time_ns: u64,
    pub components: Vec<VulkanRuntimeComponentWorkingSetPressure>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VulkanRuntimeWorkingSetPressureSnapshot {
    pub stores: Vec<VulkanRuntimeDeviceWorkingSetPressure>,
}

impl VulkanRuntimeWorkingSetPressureSnapshot {
    pub fn delta_since(
        &self,
        previous: &Self,
    ) -> Result<Self, VulkanCompiledResourceDeviceStoreError> {
        if self.stores.len() != previous.stores.len() {
            return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                "working-set store count changed from {} to {}",
                previous.stores.len(),
                self.stores.len(),
            )));
        }
        let stores = self
            .stores
            .iter()
            .zip(&previous.stores)
            .map(|(current, previous)| working_set_store_delta(current, previous))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { stores })
    }
}

fn working_set_store_delta(
    current: &VulkanRuntimeDeviceWorkingSetPressure,
    previous: &VulkanRuntimeDeviceWorkingSetPressure,
) -> Result<VulkanRuntimeDeviceWorkingSetPressure, VulkanCompiledResourceDeviceStoreError> {
    if current.store_id != previous.store_id
        || current.physical_device_id != previous.physical_device_id
        || current.logical_device_ids != previous.logical_device_ids
        || current.components.len() != previous.components.len()
    {
        return Err(VulkanCompiledResourceDeviceStoreError::new(
            "working-set store identity or component coverage changed",
        ));
    }
    let mut delta = current.clone();
    delta.eviction_count = checked_working_set_counter_delta(
        current.eviction_count,
        previous.eviction_count,
        "eviction",
    )?;
    delta.reload_count = checked_working_set_counter_delta(
        current.reload_count,
        previous.reload_count,
        "reload",
    )?;
    delta.blocking_time_ns = checked_working_set_counter_delta(
        current.blocking_time_ns,
        previous.blocking_time_ns,
        "blocking time",
    )?;
    delta.components = current
        .components
        .iter()
        .zip(&previous.components)
        .map(|(current, previous)| working_set_component_delta(current, previous))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(delta)
}

fn working_set_component_delta(
    current: &VulkanRuntimeComponentWorkingSetPressure,
    previous: &VulkanRuntimeComponentWorkingSetPressure,
) -> Result<VulkanRuntimeComponentWorkingSetPressure, VulkanCompiledResourceDeviceStoreError> {
    if current.execution_scope != previous.execution_scope
        || current.component_id != previous.component_id
        || current.addressable_unit_count != previous.addressable_unit_count
        || current.addressable_payload_bytes != previous.addressable_payload_bytes
        || current.selected_resources.len() != previous.selected_resources.len()
    {
        return Err(VulkanCompiledResourceDeviceStoreError::new(
            "working-set component identity or addressable resources changed",
        ));
    }
    let mut delta = current.clone();
    delta.selection_count = checked_working_set_counter_delta(
        current.selection_count,
        previous.selection_count,
        "selection",
    )?;
    delta.gpu_miss_count = checked_working_set_counter_delta(
        current.gpu_miss_count,
        previous.gpu_miss_count,
        "GPU miss",
    )?;
    delta.selected_unit_count = 0;
    delta.selected_payload_bytes = 0;
    delta.selected_resources = current
        .selected_resources
        .iter()
        .zip(&previous.selected_resources)
        .map(|(current, previous)| {
            if current.group_id != previous.group_id
                || current.payload_bytes != previous.payload_bytes
            {
                return Err(VulkanCompiledResourceDeviceStoreError::new(
                    "working-set selected-resource identity changed",
                ));
            }
            Ok(VulkanRuntimeSelectedResourcePressure {
                group_id: current.group_id.clone(),
                payload_bytes: current.payload_bytes,
                selection_count: checked_working_set_counter_delta(
                    current.selection_count,
                    previous.selection_count,
                    "resource selection",
                )?,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    for resource in &delta.selected_resources {
        if resource.selection_count == 0 {
            continue;
        }
        delta.selected_unit_count += 1;
        delta.selected_payload_bytes = delta
            .selected_payload_bytes
            .checked_add(resource.payload_bytes)
            .ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "working-set interval selected payload bytes overflowed",
                )
            })?;
    }
    Ok(delta)
}

fn checked_working_set_counter_delta(
    current: u64,
    previous: u64,
    label: &str,
) -> Result<u64, VulkanCompiledResourceDeviceStoreError> {
    current.checked_sub(previous).ok_or_else(|| {
        VulkanCompiledResourceDeviceStoreError::new(format!(
            "working-set {label} counter regressed from {previous} to {current}",
        ))
    })
}

impl VulkanCompiledResourceDeviceStore {
    fn working_set_pressure(
        &self,
        telemetry: &VulkanSelectionTelemetrySnapshot,
    ) -> Result<VulkanRuntimeDeviceWorkingSetPressure, VulkanCompiledResourceDeviceStoreError> {
        let selection_counts = self.selection_counts_by_group(telemetry)?;
        let residency = self.residency_report()?;
        let resident_group_ids = self
            .manager
            .snapshot()
            .map_err(compiled_device_store_residency_error)?
            .directory
            .into_iter()
            .filter(|entry| entry.state == ResourceResidencyState::Resident)
            .map(|entry| entry.group_id)
            .collect::<BTreeSet<_>>();
        let misses = self.instrumentation.gpu_misses_by_component()?;
        let mut components = self
            .coverage_index
            .iter()
            .map(|coverage| {
                component_working_set_pressure_from_groups(
                    coverage,
                    &self.group_payload_bytes,
                    &resident_group_ids,
                    &selection_counts,
                    misses
                        .get(&(
                            coverage.execution_scope.clone(),
                            coverage.component_id.clone(),
                        ))
                        .copied()
                        .unwrap_or_default(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        components.sort_by(|left, right| {
            (left.execution_scope.as_str(), left.component_id.as_str())
                .cmp(&(right.execution_scope.as_str(), right.component_id.as_str()))
        });
        Ok(VulkanRuntimeDeviceWorkingSetPressure {
            store_id: residency.store_id,
            physical_device_id: residency.physical_device_id,
            logical_device_ids: residency.logical_device_ids,
            current_device_bytes: residency.current_device_bytes,
            maximum_device_bytes: residency.maximum_device_bytes,
            current_payload_bytes: residency.current_payload_bytes,
            maximum_payload_bytes: residency.maximum_payload_bytes,
            eviction_count: residency.eviction_count,
            reload_count: residency.reload_count,
            blocking_time_ns: residency.blocking_time_ns,
            components,
        })
    }
}

fn component_working_set_pressure_from_groups(
    coverage: &VulkanCompiledResourceComponentCoverageIndex,
    group_payload_bytes: &BTreeMap<String, usize>,
    resident_group_ids: &BTreeSet<String>,
    selection_counts: &BTreeMap<String, u64>,
    gpu_miss_count: u64,
) -> Result<VulkanRuntimeComponentWorkingSetPressure, VulkanCompiledResourceDeviceStoreError> {
    let mut pressure = VulkanRuntimeComponentWorkingSetPressure {
        execution_scope: coverage.execution_scope.clone(),
        component_id: coverage.component_id.clone(),
        addressable_unit_count: coverage.group_ids.len(),
        gpu_miss_count,
        ..Default::default()
    };
    for group_id in &coverage.group_ids {
        let payload_bytes = group_payload_bytes.get(group_id).copied().ok_or_else(|| {
            VulkanCompiledResourceDeviceStoreError::new(format!(
                "compiled working-set component {}.{} references group {group_id:?} without a payload size",
                coverage.execution_scope, coverage.component_id,
            ))
        })?;
        pressure.addressable_payload_bytes = pressure
            .addressable_payload_bytes
            .checked_add(payload_bytes)
            .ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled working-set addressable payload bytes overflowed",
                )
            })?;
        if resident_group_ids.contains(group_id) {
            pressure.resident_unit_count += 1;
            pressure.resident_payload_bytes = pressure
                .resident_payload_bytes
                .checked_add(payload_bytes)
                .ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled working-set resident payload bytes overflowed",
                    )
                })?;
        }
        let selection_count = selection_counts.get(group_id).copied().unwrap_or_default();
        pressure
            .selected_resources
            .push(VulkanRuntimeSelectedResourcePressure {
                group_id: group_id.clone(),
                payload_bytes,
                selection_count,
            });
        if selection_count > 0 {
            pressure.selected_unit_count += 1;
            pressure.selected_payload_bytes = pressure
                .selected_payload_bytes
                .checked_add(payload_bytes)
                .ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled working-set selected payload bytes overflowed",
                    )
                })?;
            pressure.selection_count = pressure
                .selection_count
                .checked_add(selection_count)
                .ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled working-set selection count overflowed",
                    )
                })?;
        }
    }
    pressure
        .selected_resources
        .sort_by(|left, right| left.group_id.cmp(&right.group_id));
    Ok(pressure)
}

impl VulkanResidentInProcessPlacedModelPackage {
    pub fn working_set_pressure_snapshot(
        &self,
        telemetry: &VulkanSelectionTelemetrySnapshot,
    ) -> Result<VulkanRuntimeWorkingSetPressureSnapshot, VulkanCompiledResourceDeviceStoreError>
    {
        let mut unique_stores = BTreeMap::new();
        for store in self.compiled_resource_device_stores.values() {
            unique_stores
                .entry(store.device_id().to_string())
                .or_insert_with(|| Arc::clone(store));
        }
        let mut stores = unique_stores
            .into_values()
            .map(|store| store.working_set_pressure(telemetry))
            .collect::<Result<Vec<_>, _>>()?;
        stores.sort_by(|left, right| {
            left.physical_device_id
                .cmp(&right.physical_device_id)
                .then_with(|| left.store_id.cmp(&right.store_id))
        });
        Ok(VulkanRuntimeWorkingSetPressureSnapshot { stores })
    }
}

#[cfg(test)]
mod runtime_working_set_pressure_tests {
    use super::*;

    fn fixture_coverage() -> VulkanCompiledResourceComponentCoverageIndex {
        VulkanCompiledResourceComponentCoverageIndex {
            execution_scope: "target".to_string(),
            component_id: "block_7".to_string(),
            group_ids: ["expert_a", "expert_b", "expert_c"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        }
    }

    #[test]
    fn component_working_set_pressure_counts_unique_bytes_not_repeated_hits() {
        let pressure = component_working_set_pressure_from_groups(
            &fixture_coverage(),
            &BTreeMap::from([
                ("expert_a".to_string(), 8),
                ("expert_b".to_string(), 16),
                ("expert_c".to_string(), 32),
            ]),
            &["expert_a".to_string(), "expert_c".to_string()]
                .into_iter()
                .collect(),
            &BTreeMap::from([
                ("expert_a".to_string(), 100),
                ("expert_b".to_string(), 2),
                ("expert_c".to_string(), 0),
            ]),
            3,
        )
        .unwrap();

        assert_eq!(pressure.addressable_unit_count, 3);
        assert_eq!(pressure.addressable_payload_bytes, 56);
        assert_eq!(pressure.resident_unit_count, 2);
        assert_eq!(pressure.resident_payload_bytes, 40);
        assert_eq!(pressure.selected_unit_count, 2);
        assert_eq!(pressure.selected_payload_bytes, 24);
        assert_eq!(pressure.selection_count, 102);
        assert_eq!(pressure.gpu_miss_count, 3);
    }

    #[test]
    fn component_working_set_pressure_rejects_missing_payload_evidence() {
        let error = component_working_set_pressure_from_groups(
            &fixture_coverage(),
            &BTreeMap::from([
                ("expert_a".to_string(), 8),
                ("expert_b".to_string(), 16),
            ]),
            &BTreeSet::new(),
            &BTreeMap::new(),
            0,
        )
        .unwrap_err();

        assert!(error.to_string().contains("expert_c"));
        assert!(error.to_string().contains("without a payload size"));
    }

    #[test]
    fn component_working_set_pressure_rejects_byte_overflow() {
        let error = component_working_set_pressure_from_groups(
            &fixture_coverage(),
            &BTreeMap::from([
                ("expert_a".to_string(), usize::MAX),
                ("expert_b".to_string(), 1),
                ("expert_c".to_string(), 1),
            ]),
            &BTreeSet::new(),
            &BTreeMap::new(),
            0,
        )
        .unwrap_err();

        assert!(error.to_string().contains("addressable payload bytes overflowed"));
    }

    #[test]
    fn working_set_delta_keeps_repeated_hot_resources_in_the_interval() {
        let previous_component = component_working_set_pressure_from_groups(
            &fixture_coverage(),
            &BTreeMap::from([
                ("expert_a".to_string(), 8),
                ("expert_b".to_string(), 16),
                ("expert_c".to_string(), 32),
            ]),
            &["expert_a".to_string()].into_iter().collect(),
            &BTreeMap::from([
                ("expert_a".to_string(), 10),
                ("expert_b".to_string(), 0),
                ("expert_c".to_string(), 0),
            ]),
            1,
        )
        .unwrap();
        let current_component = component_working_set_pressure_from_groups(
            &fixture_coverage(),
            &BTreeMap::from([
                ("expert_a".to_string(), 8),
                ("expert_b".to_string(), 16),
                ("expert_c".to_string(), 32),
            ]),
            &["expert_a".to_string(), "expert_b".to_string()]
                .into_iter()
                .collect(),
            &BTreeMap::from([
                ("expert_a".to_string(), 12),
                ("expert_b".to_string(), 1),
                ("expert_c".to_string(), 0),
            ]),
            2,
        )
        .unwrap();
        let fixture_store = |component, evictions, reloads, blocking| {
            VulkanRuntimeDeviceWorkingSetPressure {
                store_id: "store".to_string(),
                physical_device_id: "device".to_string(),
                logical_device_ids: vec!["gpu0".to_string()],
                current_device_bytes: 80,
                maximum_device_bytes: 100,
                current_payload_bytes: 60,
                maximum_payload_bytes: 70,
                eviction_count: evictions,
                reload_count: reloads,
                blocking_time_ns: blocking,
                components: vec![component],
            }
        };
        let previous = VulkanRuntimeWorkingSetPressureSnapshot {
            stores: vec![fixture_store(previous_component, 1, 2, 100)],
        };
        let current = VulkanRuntimeWorkingSetPressureSnapshot {
            stores: vec![fixture_store(current_component, 3, 5, 160)],
        };

        let delta = current.delta_since(&previous).unwrap();
        let store = &delta.stores[0];
        let component = &store.components[0];
        assert_eq!(store.eviction_count, 2);
        assert_eq!(store.reload_count, 3);
        assert_eq!(store.blocking_time_ns, 60);
        assert_eq!(component.selection_count, 3);
        assert_eq!(component.gpu_miss_count, 1);
        assert_eq!(component.selected_unit_count, 2);
        assert_eq!(component.selected_payload_bytes, 24);
        assert_eq!(component.selected_resources[0].selection_count, 2);
        assert_eq!(component.selected_resources[1].selection_count, 1);
    }

    #[test]
    fn working_set_delta_rejects_regressed_counters() {
        let store = |eviction_count| VulkanRuntimeDeviceWorkingSetPressure {
            store_id: "store".to_string(),
            physical_device_id: "device".to_string(),
            logical_device_ids: vec!["gpu0".to_string()],
            eviction_count,
            ..Default::default()
        };
        let previous = VulkanRuntimeWorkingSetPressureSnapshot {
            stores: vec![store(2)],
        };
        let current = VulkanRuntimeWorkingSetPressureSnapshot {
            stores: vec![store(1)],
        };

        let error = current.delta_since(&previous).unwrap_err();

        assert!(error.to_string().contains("eviction counter regressed"));
    }
}
