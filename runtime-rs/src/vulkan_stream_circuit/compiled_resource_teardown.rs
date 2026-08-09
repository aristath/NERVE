#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanCompiledResourceDeviceTeardownReport {
    pub store_id: String,
    pub physical_device_id: String,
    pub logical_device_ids: Vec<String>,
    pub released_unit_count: usize,
    pub released_payload_bytes: usize,
    pub cancelled_load_count: usize,
    pub remaining_unit_count: usize,
    pub remaining_payload_bytes: usize,
    pub acknowledged: bool,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanCompiledResourceTeardownReport {
    pub package_id: String,
    pub execution_scope: String,
    pub physical_device_count: usize,
    pub retained_device_count: usize,
    pub released_unit_count: usize,
    pub released_payload_bytes: usize,
    pub cancelled_load_count: usize,
    pub acknowledged_device_count: usize,
    pub complete: bool,
    pub devices: Vec<VulkanCompiledResourceDeviceTeardownReport>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct VulkanRetainedCompiledResourceStoreCapacities {
    store_payload_bytes: usize,
    device_payload_bytes: usize,
    host_visible_payload_bytes: usize,
    available_dynamic_device_bytes: usize,
}

#[derive(Clone, Default)]
pub struct VulkanRetainedCompiledResourceStores {
    stores_by_logical_device: BTreeMap<String, Arc<VulkanCompiledResourceDeviceStore>>,
}

impl std::fmt::Debug for VulkanRetainedCompiledResourceStores {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VulkanRetainedCompiledResourceStores")
            .field("logical_device_ids", &self.stores_by_logical_device.keys())
            .field("physical_store_count", &self.physical_store_count())
            .finish()
    }
}

impl VulkanRetainedCompiledResourceStores {
    pub fn physical_store_count(&self) -> usize {
        self.stores_by_logical_device
            .values()
            .map(|store| store.device_id())
            .collect::<BTreeSet<_>>()
            .len()
    }

    fn store_for_logical_devices(
        &self,
        logical_device_ids: &[String],
    ) -> Result<Option<Arc<VulkanCompiledResourceDeviceStore>>, String> {
        let candidates = logical_device_ids
            .iter()
            .filter_map(|device_id| self.stores_by_logical_device.get(device_id))
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return Ok(None);
        }
        if candidates.len() != logical_device_ids.len()
            || candidates
                .iter()
                .any(|candidate| !Arc::ptr_eq(candidates[0], candidate))
        {
            return Err(format!(
                "retained compiled-resource aliases for logical devices {logical_device_ids:?} are incomplete or disagree",
            ));
        }
        Ok(Some(Arc::clone(candidates[0])))
    }

    fn shared_host_cache(
        &self,
    ) -> Result<Option<Arc<VulkanCompiledResourceSharedHostCache>>, String> {
        let mut cache = None;
        let mut seen = BTreeSet::new();
        for store in self.stores_by_logical_device.values() {
            if !seen.insert(store.device_id()) {
                continue;
            }
            let Some(candidate) = store.shared_host_cache_for_remount() else {
                continue;
            };
            if cache
                .as_ref()
                .is_some_and(|current| !Arc::ptr_eq(current, &candidate))
            {
                return Err(
                    "retained compiled-resource stores disagree on their shared host cache"
                        .to_string(),
                );
            }
            cache = Some(candidate);
        }
        Ok(cache)
    }
}

#[derive(Clone, Debug)]
pub struct VulkanSessionRemountRelease {
    pub teardown: VulkanCompiledResourceTeardownReport,
    pub retained_stores: VulkanRetainedCompiledResourceStores,
}

impl VulkanCompiledResourceTeardownReport {
    fn record_acknowledged_device(
        &mut self,
        device: &VulkanCompiledResourceDeviceTeardownReport,
    ) -> Result<(), String> {
        let acknowledged_device_count = self
            .acknowledged_device_count
            .checked_add(1)
            .ok_or_else(|| {
                "compiled resource teardown device count overflowed"
                    .to_string()
            })?;
        let released_unit_count = self
            .released_unit_count
            .checked_add(device.released_unit_count)
            .ok_or_else(|| {
                "compiled resource teardown unit count overflowed".to_string()
            })?;
        let released_payload_bytes = self
            .released_payload_bytes
            .checked_add(device.released_payload_bytes)
            .ok_or_else(|| {
                "compiled resource teardown payload bytes overflowed"
                    .to_string()
            })?;
        let cancelled_load_count = self
            .cancelled_load_count
            .checked_add(device.cancelled_load_count)
            .ok_or_else(|| {
                "compiled resource teardown cancellation count overflowed"
                    .to_string()
            })?;
        self.acknowledged_device_count = acknowledged_device_count;
        self.released_unit_count = released_unit_count;
        self.released_payload_bytes = released_payload_bytes;
        self.cancelled_load_count = cancelled_load_count;
        Ok(())
    }
}

impl VulkanResidentInProcessPlacedModelPackage {
    fn teardown_compiled_resources(
        &self,
    ) -> VulkanCompiledResourceTeardownReport {
        self.teardown_compiled_resources_except(&BTreeSet::new())
    }

    fn compiled_resource_stores_for_session_remount(
        &self,
        retained_logical_device_ids: &BTreeSet<String>,
    ) -> (BTreeSet<String>, VulkanRetainedCompiledResourceStores) {
        let retained_placements = self
            .compiled_resource_physical_placements
            .iter()
            .filter(|placement| {
                placement
                    .logical_device_ids
                    .iter()
                    .all(|device_id| retained_logical_device_ids.contains(device_id))
            })
            .collect::<Vec<_>>();
        let retained_store_ids = retained_placements
            .iter()
            .map(|placement| placement.store_id.clone())
            .collect::<BTreeSet<_>>();
        let mut stores_by_logical_device = BTreeMap::new();
        for placement in retained_placements {
            let Some(first_device_id) = placement.logical_device_ids.first() else {
                continue;
            };
            let Some(store) = self.compiled_resource_device_stores.get(first_device_id) else {
                continue;
            };
            for logical_device_id in &placement.logical_device_ids {
                stores_by_logical_device.insert(logical_device_id.clone(), Arc::clone(store));
            }
        }
        (
            retained_store_ids,
            VulkanRetainedCompiledResourceStores {
                stores_by_logical_device,
            },
        )
    }

    fn teardown_compiled_resources_except(
        &self,
        retained_store_ids: &BTreeSet<String>,
    ) -> VulkanCompiledResourceTeardownReport {
        let mut report = VulkanCompiledResourceTeardownReport {
            package_id: self.package_id.clone(),
            execution_scope: self.execution_scope.clone(),
            physical_device_count: self
                .compiled_resource_physical_placements
                .iter()
                .filter(|placement| !retained_store_ids.contains(&placement.store_id))
                .count(),
            retained_device_count: retained_store_ids.len(),
            complete: true,
            ..Default::default()
        };
        let mut placements =
            self.compiled_resource_physical_placements
                .iter()
                .filter(|placement| !retained_store_ids.contains(&placement.store_id))
                .collect::<Vec<_>>();
        placements.sort_by(|left, right| {
            left.physical_device_id
                .cmp(&right.physical_device_id)
                .then_with(|| left.store_id.cmp(&right.store_id))
        });
        for placement in placements {
            let mut device_report =
                VulkanCompiledResourceDeviceTeardownReport {
                    store_id: placement.store_id.clone(),
                    physical_device_id:
                        placement.physical_device_id.clone(),
                    logical_device_ids:
                        placement.logical_device_ids.clone(),
                    released_unit_count: 0,
                    released_payload_bytes: 0,
                    cancelled_load_count: 0,
                    remaining_unit_count: usize::MAX,
                    remaining_payload_bytes: usize::MAX,
                    acknowledged: false,
                    error: None,
                };
            let result =
                compiled_resource_store_for_physical_placement(
                    self, placement,
                )
                .and_then(|store| {
                    let released =
                        store.unload().map_err(|error| error.to_string())?;
                    let state = store
                        .residency_report()
                        .map_err(|error| error.to_string())?;
                    device_report.released_unit_count =
                        released.group_count;
                    device_report.released_payload_bytes =
                        released.byte_count;
                    device_report.cancelled_load_count =
                        released.cancelled_load_count;
                    device_report.remaining_unit_count = state
                        .resident_unit_count
                        .checked_add(state.loading_unit_count)
                        .and_then(|count| {
                            count.checked_add(state.failed_unit_count)
                        })
                        .ok_or_else(|| {
                            "compiled resource teardown remaining-unit count overflowed"
                                .to_string()
                        })?;
                    device_report.remaining_payload_bytes =
                        state.current_payload_bytes;
                    if device_report.remaining_unit_count != 0
                        || device_report.remaining_payload_bytes != 0
                    {
                        return Err(format!(
                            "compiled resource store {:?} retained {} units and {} payload bytes after teardown",
                            placement.store_id,
                            device_report.remaining_unit_count,
                            device_report.remaining_payload_bytes,
                        ));
                    }
                    Ok(())
                });
            match result {
                Ok(()) => {
                    device_report.acknowledged = true;
                    if let Err(error) =
                        report.record_acknowledged_device(&device_report)
                    {
                        report.complete = false;
                        device_report.acknowledged = false;
                        device_report.error = Some(error);
                    }
                }
                Err(error) => {
                    report.complete = false;
                    device_report.error = Some(error);
                }
            }
            report.devices.push(device_report);
        }
        report.complete = report.complete
            && report.acknowledged_device_count
                == report.physical_device_count;
        report
    }
}

fn compiled_resource_store_for_physical_placement<'a>(
    package: &'a VulkanResidentInProcessPlacedModelPackage,
    placement: &VulkanCompiledResourcePhysicalPlacement,
) -> Result<&'a VulkanCompiledResourceDeviceStore, String> {
    let first_logical_device_id =
        placement.logical_device_ids.first().ok_or_else(|| {
            "compiled resource physical placement has no logical device"
                .to_string()
        })?;
    let store = package
        .compiled_resource_device_stores
        .get(first_logical_device_id)
        .ok_or_else(|| {
            format!(
                "compiled resource physical placement {:?} has no store for logical device {first_logical_device_id:?}",
                placement.store_id,
            )
        })?;
    if store.device_id() != placement.store_id
        || store.physical_device_id() != placement.physical_device_id
        || store.logical_device_ids() != placement.logical_device_ids
    {
        return Err(format!(
            "compiled resource store identity {:?}/{:?}/{:?} disagrees with placement {:?}/{:?}/{:?}",
            store.device_id(),
            store.physical_device_id(),
            store.logical_device_ids(),
            placement.store_id,
            placement.physical_device_id,
            placement.logical_device_ids,
        ));
    }
    for logical_device_id in &placement.logical_device_ids {
        let alias = package
            .compiled_resource_device_stores
            .get(logical_device_id)
            .ok_or_else(|| {
                format!(
                    "compiled resource placement {:?} has no store for logical device {logical_device_id:?}",
                    placement.store_id,
                )
            })?;
        if !Arc::ptr_eq(store, alias) {
            return Err(format!(
                "compiled resource placement {:?} aliases logical device {logical_device_id:?} to another physical store",
                placement.store_id,
            ));
        }
    }
    Ok(store)
}
