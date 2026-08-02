pub const VULKAN_COMPILED_RESOURCE_RESIDENCY_REPORT_SCHEMA: &str =
    "nerve.vulkan_compiled_resource_residency_report.v1";

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanCompiledResourceComponentCoverageReport {
    pub execution_scope: String,
    pub component_id: String,
    pub addressable_unit_count: usize,
    pub resident_unit_count: usize,
    pub gpu_selection_count: u64,
    pub gpu_resident_hit_count: u64,
    pub gpu_miss_count: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanCompiledResourceScopeCoverageReport {
    pub execution_scope: String,
    pub component_count: usize,
    pub addressable_unit_count: usize,
    pub resident_unit_count: usize,
    pub gpu_selection_count: u64,
    pub gpu_resident_hit_count: u64,
    pub gpu_miss_count: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanCompiledResourceStoreReport {
    pub store_id: String,
    pub physical_device_id: String,
    pub logical_device_ids: Vec<String>,
    pub initial_device_bytes: usize,
    pub current_device_bytes: usize,
    pub maximum_device_bytes: usize,
    pub high_water_device_bytes: usize,
    pub always_resident_parameter_bytes: usize,
    pub runtime_working_set_device_bytes: usize,
    pub metadata_device_bytes: usize,
    pub transfer_staging_host_bytes: usize,
    pub initial_payload_bytes: usize,
    pub current_payload_bytes: usize,
    pub maximum_payload_bytes: usize,
    pub high_water_payload_bytes: usize,
    pub addressable_unit_count: usize,
    pub initial_resident_unit_count: usize,
    pub resident_unit_count: usize,
    pub high_water_resident_unit_count: usize,
    pub loading_unit_count: usize,
    pub failed_unit_count: usize,
    pub gpu_selection_count: u64,
    pub gpu_resident_hit_count: u64,
    pub gpu_miss_count: u64,
    pub residency_directory_hit_count: u64,
    pub residency_load_required_count: u64,
    pub deduplicated_load_count: u64,
    pub successful_load_count: u64,
    pub failed_load_count: u64,
    pub cancelled_load_count: u64,
    pub eviction_count: u64,
    pub evicted_unit_count: u64,
    pub evicted_payload_bytes: u64,
    pub released_device_bytes: u64,
    pub reload_count: u64,
    pub logical_read_count: u64,
    pub physical_read_count: u64,
    pub logical_bytes_read: u64,
    pub physical_bytes_read: u64,
    pub uploaded_bytes: u64,
    pub read_time_ns: u64,
    pub upload_time_ns: u64,
    pub blocking_time_ns: u64,
    pub scopes: Vec<VulkanCompiledResourceScopeCoverageReport>,
    pub components: Vec<VulkanCompiledResourceComponentCoverageReport>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanCompiledResourceResidencyTotalsReport {
    pub physical_store_count: usize,
    pub initial_device_bytes: usize,
    pub current_device_bytes: usize,
    pub maximum_device_bytes: usize,
    pub high_water_device_bytes: usize,
    pub always_resident_parameter_bytes: usize,
    pub runtime_working_set_device_bytes: usize,
    pub metadata_device_bytes: usize,
    pub transfer_staging_host_bytes: usize,
    pub initial_payload_bytes: usize,
    pub current_payload_bytes: usize,
    pub maximum_payload_bytes: usize,
    pub high_water_payload_bytes: usize,
    pub addressable_unit_count: usize,
    pub initial_resident_unit_count: usize,
    pub resident_unit_count: usize,
    pub high_water_resident_unit_count: usize,
    pub gpu_selection_count: u64,
    pub gpu_resident_hit_count: u64,
    pub gpu_miss_count: u64,
    pub residency_directory_hit_count: u64,
    pub residency_load_required_count: u64,
    pub deduplicated_load_count: u64,
    pub successful_load_count: u64,
    pub failed_load_count: u64,
    pub cancelled_load_count: u64,
    pub eviction_count: u64,
    pub evicted_unit_count: u64,
    pub evicted_payload_bytes: u64,
    pub released_device_bytes: u64,
    pub reload_count: u64,
    pub physical_read_count: u64,
    pub physical_bytes_read: u64,
    pub uploaded_bytes: u64,
    pub read_time_ns: u64,
    pub upload_time_ns: u64,
    pub blocking_time_ns: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanCompiledResourceResidencyReport {
    pub schema: String,
    pub policy: ResourceResidencyPolicy,
    pub totals: VulkanCompiledResourceResidencyTotalsReport,
    pub target: VulkanCompiledResourceScopeCoverageReport,
    pub mtp: Vec<VulkanCompiledResourceScopeCoverageReport>,
    pub stores: Vec<VulkanCompiledResourceStoreReport>,
}

#[derive(Default)]
struct VulkanCompiledResourceStoreInstrumentation {
    initial_payload_bytes: std::sync::atomic::AtomicU64,
    initial_resident_unit_count: std::sync::atomic::AtomicU64,
    initial_committed_device_bytes: std::sync::atomic::AtomicU64,
    high_water_committed_device_bytes: std::sync::atomic::AtomicU64,
    uploaded_bytes: std::sync::atomic::AtomicU64,
    upload_time_ns: std::sync::atomic::AtomicU64,
    blocking_time_ns: std::sync::atomic::AtomicU64,
    eviction_count: std::sync::atomic::AtomicU64,
    evicted_unit_count: std::sync::atomic::AtomicU64,
    evicted_payload_bytes: std::sync::atomic::AtomicU64,
    released_device_bytes: std::sync::atomic::AtomicU64,
    gpu_misses_by_component:
        std::sync::Mutex<BTreeMap<(String, String), u64>>,
}

impl VulkanCompiledResourceStoreInstrumentation {
    fn mark_mount_complete(
        &self,
        payload_bytes: usize,
        resident_unit_count: usize,
        committed_device_bytes: usize,
    ) {
        use std::sync::atomic::Ordering;

        self.initial_payload_bytes.store(
            u64::try_from(payload_bytes).unwrap_or(u64::MAX),
            Ordering::Release,
        );
        self.initial_resident_unit_count.store(
            u64::try_from(resident_unit_count).unwrap_or(u64::MAX),
            Ordering::Release,
        );
        self.initial_committed_device_bytes.store(
            u64::try_from(committed_device_bytes).unwrap_or(u64::MAX),
            Ordering::Release,
        );
        self.high_water_committed_device_bytes.fetch_max(
            u64::try_from(committed_device_bytes).unwrap_or(u64::MAX),
            Ordering::AcqRel,
        );
    }

    fn record_upload(
        &self,
        uploaded_bytes: usize,
        upload_time_ns: u64,
        committed_device_bytes: usize,
    ) {
        use std::sync::atomic::Ordering;

        self.uploaded_bytes.fetch_add(
            u64::try_from(uploaded_bytes).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        self.upload_time_ns
            .fetch_add(upload_time_ns, Ordering::Relaxed);
        self.high_water_committed_device_bytes.fetch_max(
            u64::try_from(committed_device_bytes).unwrap_or(u64::MAX),
            Ordering::AcqRel,
        );
    }

    fn record_blocking_time(&self, blocking_time_ns: u64) {
        self.blocking_time_ns.fetch_add(
            blocking_time_ns,
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    fn record_eviction(
        &self,
        group_count: usize,
        payload_bytes: usize,
        released_device_bytes: usize,
    ) {
        use std::sync::atomic::Ordering;

        self.eviction_count.fetch_add(1, Ordering::Relaxed);
        self.evicted_unit_count.fetch_add(
            u64::try_from(group_count).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        self.evicted_payload_bytes.fetch_add(
            u64::try_from(payload_bytes).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        self.released_device_bytes.fetch_add(
            u64::try_from(released_device_bytes).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
    }

    fn record_gpu_gate_misses(
        &self,
        execution_scope: &str,
        component_id: &str,
        miss_count: usize,
    ) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
        if miss_count == 0 {
            return Ok(());
        }
        let mut misses = self.gpu_misses_by_component.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource GPU-miss instrumentation was poisoned",
            )
        })?;
        let count = misses
            .entry((
                execution_scope.to_string(),
                component_id.to_string(),
            ))
            .or_default();
        *count = count
            .checked_add(u64::try_from(miss_count).unwrap_or(u64::MAX))
            .ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource GPU-miss count overflowed",
                )
            })?;
        Ok(())
    }

    fn gpu_misses_by_component(
        &self,
    ) -> Result<
        BTreeMap<(String, String), u64>,
        VulkanCompiledResourceDeviceStoreError,
    > {
        self.gpu_misses_by_component
            .lock()
            .map(|misses| misses.clone())
            .map_err(|_| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource GPU-miss instrumentation was poisoned",
                )
            })
    }
}

struct VulkanCompiledResourceBlockingTimer<'a> {
    instrumentation: &'a VulkanCompiledResourceStoreInstrumentation,
    started: Instant,
}

impl<'a> VulkanCompiledResourceBlockingTimer<'a> {
    fn new(
        instrumentation: &'a VulkanCompiledResourceStoreInstrumentation,
    ) -> Self {
        Self {
            instrumentation,
            started: Instant::now(),
        }
    }
}

impl Drop for VulkanCompiledResourceBlockingTimer<'_> {
    fn drop(&mut self) {
        self.instrumentation.record_blocking_time(
            u64::try_from(self.started.elapsed().as_nanos())
                .unwrap_or(u64::MAX),
        );
    }
}

#[derive(Clone, Debug)]
struct VulkanCompiledResourceComponentCoverageIndex {
    execution_scope: String,
    component_id: String,
    group_ids: BTreeSet<String>,
}

impl VulkanResidentInProcessPlacedModelPackage {
    pub fn compiled_resource_residency_report(
        &self,
        selection_coverage: &RuntimeSelectionCoverageReport,
    ) -> Result<
        VulkanCompiledResourceResidencyReport,
        VulkanCompiledResourceDeviceStoreError,
    > {
        let mut stores = Vec::with_capacity(
            self.compiled_resource_physical_placements.len(),
        );
        for placement in &self.compiled_resource_physical_placements {
            let logical_device_id = placement
                .logical_device_ids
                .first()
                .ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled resource physical placement has no logical device",
                    )
                })?;
            let store = self
                .compiled_resource_device_stores
                .get(logical_device_id)
                .ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(format!(
                        "compiled resource physical placement {:?} has no device store",
                        placement.store_id
                    ))
                })?;
            let report = store.residency_report()?;
            if report.store_id != placement.store_id
                || report.physical_device_id
                    != placement.physical_device_id
                || report.logical_device_ids
                    != placement.logical_device_ids
            {
                return Err(
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled resource store report disagrees with physical placement",
                    ),
                );
            }
            stores.push(report);
        }
        stores.sort_by(|left, right| {
            left.physical_device_id
                .cmp(&right.physical_device_id)
                .then_with(|| left.store_id.cmp(&right.store_id))
        });

        apply_selection_coverage(&mut stores, selection_coverage)?;

        let mut totals =
            VulkanCompiledResourceResidencyTotalsReport::default();
        let mut scopes =
            BTreeMap::<String, VulkanCompiledResourceScopeCoverageReport>::new();
        for store in &stores {
            totals.add_store(store)?;
            for scope in &store.scopes {
                let aggregate = scopes
                    .entry(scope.execution_scope.clone())
                    .or_insert_with(|| {
                        VulkanCompiledResourceScopeCoverageReport {
                            execution_scope: scope.execution_scope.clone(),
                            ..Default::default()
                        }
                    });
                checked_report_add(
                    &mut aggregate.component_count,
                    scope.component_count,
                    "scope component count",
                )?;
                checked_report_add(
                    &mut aggregate.addressable_unit_count,
                    scope.addressable_unit_count,
                    "scope addressable unit count",
                )?;
                checked_report_add(
                    &mut aggregate.resident_unit_count,
                    scope.resident_unit_count,
                    "scope resident unit count",
                )?;
                checked_report_add_u64(
                    &mut aggregate.gpu_selection_count,
                    scope.gpu_selection_count,
                    "scope GPU-selection count",
                )?;
                checked_report_add_u64(
                    &mut aggregate.gpu_resident_hit_count,
                    scope.gpu_resident_hit_count,
                    "scope GPU-hit count",
                )?;
                checked_report_add_u64(
                    &mut aggregate.gpu_miss_count,
                    scope.gpu_miss_count,
                    "scope GPU-miss count",
                )?;
            }
        }
        let target = scopes.remove("target").unwrap_or_else(|| {
            VulkanCompiledResourceScopeCoverageReport {
                execution_scope: "target".to_string(),
                ..Default::default()
            }
        });
        if let Some(scope) = scopes.keys().find(|scope| {
            !scope.starts_with("draft:")
        }) {
            return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                "compiled resource report has unsupported execution scope {scope:?}"
            )));
        }
        let mtp = scopes.into_values().collect();
        Ok(VulkanCompiledResourceResidencyReport {
            schema:
                VULKAN_COMPILED_RESOURCE_RESIDENCY_REPORT_SCHEMA.to_string(),
            policy: self.resource_residency_policy,
            totals,
            target,
            mtp,
            stores,
        })
    }
}

fn apply_selection_coverage(
    stores: &mut [VulkanCompiledResourceStoreReport],
    selection_coverage: &RuntimeSelectionCoverageReport,
) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
    let mut selections_by_component =
        BTreeMap::<(String, String), u64>::new();
    for domain in &selection_coverage.domains {
        let key = (
            domain.execution_scope.clone(),
            domain.component_id.clone(),
        );
        let count = selections_by_component.entry(key).or_default();
        *count = count
            .checked_add(domain.selection_count)
            .ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource GPU-selection count overflowed",
                )
            })?;
    }
    for store in stores {
        for component in &mut store.components {
            component.gpu_selection_count = selections_by_component
                .remove(&(
                    component.execution_scope.clone(),
                    component.component_id.clone(),
                ))
                .unwrap_or_default();
            component.gpu_resident_hit_count = component
                .gpu_selection_count
                .checked_sub(component.gpu_miss_count)
                .ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(format!(
                        "compiled resource component {}.{} reports {} GPU misses for only {} selections",
                        component.execution_scope,
                        component.component_id,
                        component.gpu_miss_count,
                        component.gpu_selection_count,
                    ))
                })?;
        }
        for scope in &mut store.scopes {
            scope.gpu_selection_count = store
                .components
                .iter()
                .filter(|component| {
                    component.execution_scope == scope.execution_scope
                })
                .map(|component| component.gpu_selection_count)
                .try_fold(0u64, checked_gpu_selection_add)?;
            scope.gpu_resident_hit_count = store
                .components
                .iter()
                .filter(|component| {
                    component.execution_scope == scope.execution_scope
                })
                .map(|component| component.gpu_resident_hit_count)
                .try_fold(0u64, checked_gpu_hit_add)?;
            scope.gpu_miss_count = store
                .components
                .iter()
                .filter(|component| {
                    component.execution_scope == scope.execution_scope
                })
                .map(|component| component.gpu_miss_count)
                .try_fold(0u64, |total, count| {
                    total.checked_add(count).ok_or_else(|| {
                        VulkanCompiledResourceDeviceStoreError::new(
                            "compiled resource scope GPU-miss count overflowed",
                        )
                    })
                })?;
        }
        store.gpu_selection_count = store
            .components
            .iter()
            .map(|component| component.gpu_selection_count)
            .try_fold(0u64, checked_gpu_selection_add)?;
        store.gpu_resident_hit_count = store
            .components
            .iter()
            .map(|component| component.gpu_resident_hit_count)
            .try_fold(0u64, checked_gpu_hit_add)?;
        store.gpu_miss_count = store
            .components
            .iter()
            .map(|component| component.gpu_miss_count)
            .try_fold(0u64, |total, count| {
                total.checked_add(count).ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled resource store GPU-miss count overflowed",
                    )
                })
            })?;
    }
    if !selections_by_component.is_empty() {
        return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
            "selection telemetry references components outside the compiled dynamic-resource stores: {:?}",
            selections_by_component.keys().collect::<Vec<_>>()
        )));
    }
    Ok(())
}

fn checked_gpu_selection_add(
    total: u64,
    count: u64,
) -> Result<u64, VulkanCompiledResourceDeviceStoreError> {
    total.checked_add(count).ok_or_else(|| {
        VulkanCompiledResourceDeviceStoreError::new(
            "compiled resource GPU-selection count overflowed",
        )
    })
}

fn checked_gpu_hit_add(
    total: u64,
    count: u64,
) -> Result<u64, VulkanCompiledResourceDeviceStoreError> {
    total.checked_add(count).ok_or_else(|| {
        VulkanCompiledResourceDeviceStoreError::new(
            "compiled resource GPU-hit count overflowed",
        )
    })
}

impl VulkanCompiledResourceResidencyTotalsReport {
    fn add_store(
        &mut self,
        store: &VulkanCompiledResourceStoreReport,
    ) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
        checked_report_add(
            &mut self.physical_store_count,
            1,
            "physical store count",
        )?;
        macro_rules! add_usize {
            ($field:ident) => {
                checked_report_add(
                    &mut self.$field,
                    store.$field,
                    stringify!($field),
                )?
            };
        }
        add_usize!(initial_device_bytes);
        add_usize!(current_device_bytes);
        add_usize!(maximum_device_bytes);
        add_usize!(high_water_device_bytes);
        add_usize!(always_resident_parameter_bytes);
        add_usize!(runtime_working_set_device_bytes);
        add_usize!(metadata_device_bytes);
        add_usize!(transfer_staging_host_bytes);
        add_usize!(initial_payload_bytes);
        add_usize!(current_payload_bytes);
        add_usize!(maximum_payload_bytes);
        add_usize!(high_water_payload_bytes);
        add_usize!(addressable_unit_count);
        add_usize!(initial_resident_unit_count);
        add_usize!(resident_unit_count);
        add_usize!(high_water_resident_unit_count);

        macro_rules! add_u64 {
            ($field:ident) => {
                self.$field = self
                    .$field
                    .checked_add(store.$field)
                    .ok_or_else(|| {
                        VulkanCompiledResourceDeviceStoreError::new(format!(
                            "compiled resource report {} overflowed",
                            stringify!($field)
                        ))
                    })?
            };
        }
        add_u64!(gpu_selection_count);
        add_u64!(gpu_resident_hit_count);
        add_u64!(gpu_miss_count);
        add_u64!(residency_directory_hit_count);
        add_u64!(residency_load_required_count);
        add_u64!(deduplicated_load_count);
        add_u64!(successful_load_count);
        add_u64!(failed_load_count);
        add_u64!(cancelled_load_count);
        add_u64!(eviction_count);
        add_u64!(evicted_unit_count);
        add_u64!(evicted_payload_bytes);
        add_u64!(released_device_bytes);
        add_u64!(reload_count);
        add_u64!(physical_read_count);
        add_u64!(physical_bytes_read);
        add_u64!(uploaded_bytes);
        add_u64!(read_time_ns);
        add_u64!(upload_time_ns);
        add_u64!(blocking_time_ns);
        Ok(())
    }
}

fn checked_report_add(
    total: &mut usize,
    value: usize,
    label: &str,
) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
    *total = total.checked_add(value).ok_or_else(|| {
        VulkanCompiledResourceDeviceStoreError::new(format!(
            "compiled resource report {label} overflowed"
        ))
    })?;
    Ok(())
}

fn checked_report_add_u64(
    total: &mut u64,
    value: u64,
    label: &str,
) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
    *total = total.checked_add(value).ok_or_else(|| {
        VulkanCompiledResourceDeviceStoreError::new(format!(
            "compiled resource report {label} overflowed"
        ))
    })?;
    Ok(())
}

#[cfg(test)]
mod compiled_resource_residency_report_tests {
    use super::*;

    #[test]
    fn selection_coverage_reconciles_gpu_hits_and_misses_by_scope() {
        let mut stores = vec![VulkanCompiledResourceStoreReport {
            scopes: vec![VulkanCompiledResourceScopeCoverageReport {
                execution_scope: "target".to_string(),
                component_count: 1,
                ..Default::default()
            }],
            components: vec![
                VulkanCompiledResourceComponentCoverageReport {
                    execution_scope: "target".to_string(),
                    component_id: "layer_0".to_string(),
                    gpu_miss_count: 3,
                    ..Default::default()
                },
            ],
            ..Default::default()
        }];
        let selection_coverage = RuntimeSelectionCoverageReport {
            domain_count: 2,
            selection_count: 12,
            domains: vec![
                RuntimeSelectionDomainCoverageReport {
                    execution_scope: "target".to_string(),
                    component_id: "layer_0".to_string(),
                    node_id: "router".to_string(),
                    domain_id: "experts".to_string(),
                    resource_count: 4,
                    selected_resource_count: 2,
                    selection_count: 5,
                    selected_resources: Vec::new(),
                },
                RuntimeSelectionDomainCoverageReport {
                    execution_scope: "target".to_string(),
                    component_id: "layer_0".to_string(),
                    node_id: "shared_router".to_string(),
                    domain_id: "shared_experts".to_string(),
                    resource_count: 2,
                    selected_resource_count: 2,
                    selection_count: 7,
                    selected_resources: Vec::new(),
                },
            ],
            ..Default::default()
        };

        apply_selection_coverage(&mut stores, &selection_coverage).unwrap();

        assert_eq!(stores[0].gpu_selection_count, 12);
        assert_eq!(stores[0].gpu_resident_hit_count, 9);
        assert_eq!(stores[0].gpu_miss_count, 3);
        assert_eq!(stores[0].scopes[0].gpu_selection_count, 12);
        assert_eq!(stores[0].scopes[0].gpu_resident_hit_count, 9);
        assert_eq!(stores[0].scopes[0].gpu_miss_count, 3);
        assert_eq!(stores[0].components[0].gpu_selection_count, 12);
        assert_eq!(stores[0].components[0].gpu_resident_hit_count, 9);
    }

    #[test]
    fn selection_coverage_rejects_more_gpu_misses_than_selections() {
        let mut stores = vec![VulkanCompiledResourceStoreReport {
            components: vec![
                VulkanCompiledResourceComponentCoverageReport {
                    execution_scope: "target".to_string(),
                    component_id: "layer_0".to_string(),
                    gpu_miss_count: 2,
                    ..Default::default()
                },
            ],
            ..Default::default()
        }];
        let selection_coverage = RuntimeSelectionCoverageReport {
            domains: vec![RuntimeSelectionDomainCoverageReport {
                execution_scope: "target".to_string(),
                component_id: "layer_0".to_string(),
                node_id: "router".to_string(),
                domain_id: "experts".to_string(),
                resource_count: 2,
                selected_resource_count: 1,
                selection_count: 1,
                selected_resources: Vec::new(),
            }],
            ..Default::default()
        };

        let error =
            apply_selection_coverage(&mut stores, &selection_coverage)
                .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("2 GPU misses for only 1 selections")
        );
    }

    #[test]
    fn compiled_resource_residency_totals_reconcile_physical_stores() {
        let store = VulkanCompiledResourceStoreReport {
            initial_device_bytes: 100,
            current_device_bytes: 120,
            maximum_device_bytes: 200,
            high_water_device_bytes: 140,
            always_resident_parameter_bytes: 40,
            runtime_working_set_device_bytes: 20,
            metadata_device_bytes: 10,
            transfer_staging_host_bytes: 8,
            initial_payload_bytes: 30,
            current_payload_bytes: 50,
            maximum_payload_bytes: 130,
            high_water_payload_bytes: 70,
            addressable_unit_count: 13,
            initial_resident_unit_count: 3,
            resident_unit_count: 5,
            high_water_resident_unit_count: 7,
            gpu_selection_count: 11,
            gpu_resident_hit_count: 9,
            gpu_miss_count: 2,
            residency_directory_hit_count: 1,
            residency_load_required_count: 2,
            deduplicated_load_count: 1,
            successful_load_count: 2,
            eviction_count: 3,
            evicted_unit_count: 4,
            evicted_payload_bytes: 40,
            released_device_bytes: 48,
            reload_count: 1,
            physical_read_count: 2,
            physical_bytes_read: 50,
            uploaded_bytes: 50,
            read_time_ns: 5,
            upload_time_ns: 6,
            blocking_time_ns: 7,
            ..Default::default()
        };
        let mut totals =
            VulkanCompiledResourceResidencyTotalsReport::default();
        totals.add_store(&store).unwrap();
        totals.add_store(&store).unwrap();

        assert_eq!(totals.physical_store_count, 2);
        assert_eq!(totals.initial_device_bytes, 200);
        assert_eq!(totals.current_device_bytes, 240);
        assert_eq!(totals.always_resident_parameter_bytes, 80);
        assert_eq!(totals.runtime_working_set_device_bytes, 40);
        assert_eq!(totals.metadata_device_bytes, 20);
        assert_eq!(totals.transfer_staging_host_bytes, 16);
        assert_eq!(totals.resident_unit_count, 10);
        assert_eq!(totals.gpu_selection_count, 22);
        assert_eq!(totals.gpu_resident_hit_count, 18);
        assert_eq!(totals.gpu_miss_count, 4);
        assert_eq!(totals.physical_bytes_read, 100);
        assert_eq!(totals.eviction_count, 6);
        assert_eq!(totals.evicted_unit_count, 8);
        assert_eq!(totals.evicted_payload_bytes, 80);
        assert_eq!(totals.released_device_bytes, 96);
        assert_eq!(totals.reload_count, 2);
        assert_eq!(totals.blocking_time_ns, 14);
    }

    #[test]
    fn compiled_resource_residency_totals_reject_overflow() {
        let mut totals =
            VulkanCompiledResourceResidencyTotalsReport {
                current_device_bytes: usize::MAX,
                ..Default::default()
            };
        let store = VulkanCompiledResourceStoreReport {
            current_device_bytes: 1,
            ..Default::default()
        };

        let error = totals.add_store(&store).unwrap_err();

        assert!(error.to_string().contains("current_device_bytes"));
    }
}
