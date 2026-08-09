#[derive(Clone, Debug, PartialEq)]
pub struct VulkanRuntimeWorkingSetRebalance {
    pub placement: VulkanRuntimeAutoPlacement,
    pub moved_component_ids: Vec<String>,
    pub retained_logical_device_ids: BTreeSet<String>,
    pub current_predicted_ns_per_activation: u128,
    pub proposed_predicted_ns_per_activation: u128,
    pub observed_blocking_ns: u64,
    pub estimated_remount_ns: u128,
    pub estimated_net_benefit_ns: u128,
}

#[allow(clippy::too_many_arguments)]
pub fn rebalance_demand_paged_vulkan_runtime_model_from_working_set(
    manifest_dir: impl AsRef<Path>,
    runtime_model: &VulkanResidentRuntimeModel,
    candidates: &[VulkanRuntimePlacementCandidate],
    placement_costs: &VulkanRuntimePlacementCostModel,
    cumulative_pressure: &VulkanRuntimeWorkingSetPressureSnapshot,
    interval_pressure: &VulkanRuntimeWorkingSetPressureSnapshot,
    observed_activation_count: u64,
    context_capacity_activations: usize,
    speculative_draft_tokens: usize,
) -> Result<Option<VulkanRuntimeWorkingSetRebalance>, VulkanRuntimeResidencyPlanError> {
    if observed_activation_count == 0 {
        return Ok(None);
    }
    validate_working_set_pressure_identity(cumulative_pressure, interval_pressure)?;
    if !interval_pressure
        .stores
        .iter()
        .any(|store| store.eviction_count > 0 || store.reload_count > 0)
    {
        return Ok(None);
    }
    let manifest_dir = manifest_dir.as_ref();
    let tensor_index = runtime_model
        .load_runtime_tensor_index(manifest_dir)
        .map_err(|error| VulkanRuntimeResidencyPlanError(error.to_string()))?;
    let components = capacity_packed_runtime_components(
        runtime_model,
        &tensor_index,
        speculative_draft_tokens > 0,
    )?;
    let mut balance = runtime_paged_placement_balance(
        runtime_model,
        &tensor_index,
        &components,
        speculative_draft_tokens > 0,
    )?;
    let observed_by_component = observed_component_pressure(cumulative_pressure)?;
    for (index, component) in components.iter().enumerate() {
        let Some(observed) = observed_by_component.get(component.component_id.as_str()) else {
            continue;
        };
        let static_bytes = balance.component_weights[index];
        let addressable_bytes = observed.addressable_payload_bytes as u128;
        let fixed_bytes = static_bytes.checked_sub(addressable_bytes).ok_or_else(|| {
            VulkanRuntimeResidencyPlanError(format!(
                "observed dynamic payload for component {:?} exceeds its compiled placement weight",
                component.component_id,
            ))
        })?;
        balance.component_weights[index] = fixed_bytes
            .checked_add(observed.selected_payload_bytes as u128)
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(
                    "observed component working-set weight overflowed".to_string(),
                )
            })?
            .max(1);
    }
    let boundaries = vulkan_runtime_placement_boundaries(runtime_model)?;
    let current_predicted_ns_per_activation = predicted_runtime_placement_ns(
        runtime_model,
        &components,
        &boundaries,
        placement_costs,
    )?;
    let proposed = capacity_pack_demand_paged_vulkan_runtime_model_on_devices(
        manifest_dir,
        runtime_model,
        &tensor_index,
        &components,
        candidates,
        Some(placement_costs),
        &balance,
        context_capacity_activations,
        speculative_draft_tokens,
    )?;
    let moved_component_ids = moved_runtime_component_ids(runtime_model, &proposed.runtime_model);
    if moved_component_ids.is_empty()
        || !rebalance_relieves_observed_churn(
            runtime_model,
            &proposed.runtime_model,
            interval_pressure,
        )?
    {
        return Ok(None);
    }
    let proposed_boundaries = vulkan_runtime_placement_boundaries(&proposed.runtime_model)?;
    let proposed_predicted_ns_per_activation = predicted_runtime_placement_ns(
        &proposed.runtime_model,
        &components,
        &proposed_boundaries,
        placement_costs,
    )?;
    let observed_blocking_ns = interval_pressure.stores.iter().try_fold(
        0u64,
        |total, store| {
            total.checked_add(store.blocking_time_ns).ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(
                    "observed working-set blocking time overflowed".to_string(),
                )
            })
        },
    )?;
    let retained_logical_device_ids = unchanged_runtime_placement_device_ids(
        runtime_model,
        &proposed.runtime_model,
    );
    let estimated_remount_ns = estimate_working_set_remount_ns(
        &retained_logical_device_ids,
        cumulative_pressure,
        interval_pressure,
    )?;
    let Some(estimated_net_benefit_ns) = working_set_rebalance_net_benefit_ns(
        observed_blocking_ns,
        estimated_remount_ns,
        current_predicted_ns_per_activation,
        proposed_predicted_ns_per_activation,
        observed_activation_count,
    )?
    else {
        return Ok(None);
    };
    Ok(Some(VulkanRuntimeWorkingSetRebalance {
        placement: proposed,
        moved_component_ids,
        retained_logical_device_ids,
        current_predicted_ns_per_activation,
        proposed_predicted_ns_per_activation,
        observed_blocking_ns,
        estimated_remount_ns,
        estimated_net_benefit_ns,
    }))
}

fn working_set_rebalance_net_benefit_ns(
    observed_blocking_ns: u64,
    estimated_remount_ns: u128,
    current_predicted_ns_per_activation: u128,
    proposed_predicted_ns_per_activation: u128,
    observed_activation_count: u64,
) -> Result<Option<u128>, VulkanRuntimeResidencyPlanError> {
    let added_execution_ns = proposed_predicted_ns_per_activation
        .saturating_sub(current_predicted_ns_per_activation)
        .checked_mul(u128::from(observed_activation_count))
        .ok_or_else(|| {
            VulkanRuntimeResidencyPlanError(
                "working-set rebalance execution cost overflowed".to_string(),
            )
        })?;
    let total_cost = estimated_remount_ns
        .checked_add(added_execution_ns)
        .ok_or_else(|| {
            VulkanRuntimeResidencyPlanError(
                "working-set rebalance total cost overflowed".to_string(),
            )
        })?;
    Ok(u128::from(observed_blocking_ns)
        .checked_sub(total_cost)
        .filter(|benefit| *benefit > 0))
}

fn validate_working_set_pressure_identity(
    cumulative: &VulkanRuntimeWorkingSetPressureSnapshot,
    interval: &VulkanRuntimeWorkingSetPressureSnapshot,
) -> Result<(), VulkanRuntimeResidencyPlanError> {
    if cumulative.stores.len() != interval.stores.len()
        || cumulative
            .stores
            .iter()
            .zip(&interval.stores)
            .any(|(cumulative, interval)| {
                cumulative.store_id != interval.store_id
                    || cumulative.physical_device_id != interval.physical_device_id
                    || cumulative.logical_device_ids != interval.logical_device_ids
                    || cumulative.components.len() != interval.components.len()
                    || cumulative
                        .components
                        .iter()
                        .zip(&interval.components)
                        .any(|(cumulative, interval)| {
                            cumulative.execution_scope != interval.execution_scope
                                || cumulative.component_id != interval.component_id
                        })
            })
    {
        return Err(VulkanRuntimeResidencyPlanError(
            "cumulative and interval working-set pressure have different identities".to_string(),
        ));
    }
    Ok(())
}

fn observed_component_pressure(
    pressure: &VulkanRuntimeWorkingSetPressureSnapshot,
) -> Result<
    BTreeMap<&str, &VulkanRuntimeComponentWorkingSetPressure>,
    VulkanRuntimeResidencyPlanError,
> {
    let mut components = BTreeMap::new();
    for component in pressure
        .stores
        .iter()
        .flat_map(|store| &store.components)
        .filter(|component| component.execution_scope == "target")
    {
        if components
            .insert(component.component_id.as_str(), component)
            .is_some()
        {
            return Err(VulkanRuntimeResidencyPlanError(format!(
                "observed working-set component {:?} appears in more than one physical store",
                component.component_id,
            )));
        }
    }
    Ok(components)
}

fn predicted_runtime_placement_ns(
    runtime_model: &VulkanResidentRuntimeModel,
    components: &[CapacityPackedPlacementComponent],
    boundaries: &[VulkanRuntimePlacementBoundary],
    costs: &VulkanRuntimePlacementCostModel,
) -> Result<u128, VulkanRuntimeResidencyPlanError> {
    if boundaries.len() != components.len().saturating_sub(1) {
        return Err(VulkanRuntimeResidencyPlanError(
            "predicted placement boundary count differs from its component chain".to_string(),
        ));
    }
    let devices = components
        .iter()
        .map(|component| {
            runtime_model
                .runtime_graph
                .instances
                .iter()
                .find(|instance| instance.instance_id == component.component_id)
                .map(|instance| instance.device_id.as_str())
                .ok_or_else(|| {
                    VulkanRuntimeResidencyPlanError(format!(
                        "runtime placement omits component {:?}",
                        component.component_id,
                    ))
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut total = 0u128;
    for (index, component) in components.iter().enumerate() {
        total = total
            .checked_add(u128::from(
                costs.component_execution_ns(devices[index], &component.component_id)?,
            ))
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(
                    "predicted placement execution time overflowed".to_string(),
                )
            })?;
        if index > 0 && devices[index - 1] != devices[index] {
            total = total
                .checked_add(u128::from(runtime_placement_boundary_cost_ns(
                    &boundaries[index - 1],
                    devices[index - 1],
                    devices[index],
                    costs,
                )?))
                .ok_or_else(|| {
                    VulkanRuntimeResidencyPlanError(
                        "predicted placement transfer time overflowed".to_string(),
                    )
                })?;
        }
    }
    Ok(total)
}

fn moved_runtime_component_ids(
    current: &VulkanResidentRuntimeModel,
    proposed: &VulkanResidentRuntimeModel,
) -> Vec<String> {
    let proposed_by_id = proposed
        .runtime_graph
        .instances
        .iter()
        .map(|instance| (instance.instance_id.as_str(), instance.device_id.as_str()))
        .collect::<BTreeMap<_, _>>();
    current
        .runtime_graph
        .instances
        .iter()
        .filter(|instance| {
            proposed_by_id
                .get(instance.instance_id.as_str())
                .is_some_and(|device_id| **device_id != instance.device_id)
        })
        .map(|instance| instance.instance_id.clone())
        .collect()
}

fn unchanged_runtime_placement_device_ids(
    current: &VulkanResidentRuntimeModel,
    proposed: &VulkanResidentRuntimeModel,
) -> BTreeSet<String> {
    fn instances_by_device(
        model: &VulkanResidentRuntimeModel,
    ) -> BTreeMap<&str, BTreeSet<&str>> {
        let mut instances = BTreeMap::<&str, BTreeSet<&str>>::new();
        for instance in &model.runtime_graph.instances {
            instances
                .entry(instance.device_id.as_str())
                .or_default()
                .insert(instance.instance_id.as_str());
        }
        instances
    }

    let current = instances_by_device(current);
    let proposed = instances_by_device(proposed);
    current
        .into_iter()
        .filter_map(|(device_id, current_instances)| {
            (proposed.get(device_id) == Some(&current_instances)).then(|| device_id.to_string())
        })
        .collect()
}

fn rebalance_relieves_observed_churn(
    current: &VulkanResidentRuntimeModel,
    proposed: &VulkanResidentRuntimeModel,
    interval: &VulkanRuntimeWorkingSetPressureSnapshot,
) -> Result<bool, VulkanRuntimeResidencyPlanError> {
    let current_devices = current
        .runtime_graph
        .instances
        .iter()
        .map(|instance| (instance.instance_id.as_str(), instance.device_id.as_str()))
        .collect::<BTreeMap<_, _>>();
    let proposed_devices = proposed
        .runtime_graph
        .instances
        .iter()
        .map(|instance| (instance.instance_id.as_str(), instance.device_id.as_str()))
        .collect::<BTreeMap<_, _>>();
    for store in &interval.stores {
        if store.eviction_count == 0 && store.reload_count == 0 {
            continue;
        }
        let device_id = store.logical_device_ids.first().ok_or_else(|| {
            VulkanRuntimeResidencyPlanError(
                "working-set store has no logical execution device".to_string(),
            )
        })?;
        if store.components.iter().any(|component| {
            current_devices.get(component.component_id.as_str()) == Some(&device_id.as_str())
                && proposed_devices.get(component.component_id.as_str()) != Some(&device_id.as_str())
                && component.selected_payload_bytes > 0
        }) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn estimate_working_set_remount_ns(
    retained_logical_device_ids: &BTreeSet<String>,
    cumulative: &VulkanRuntimeWorkingSetPressureSnapshot,
    interval: &VulkanRuntimeWorkingSetPressureSnapshot,
) -> Result<u128, VulkanRuntimeResidencyPlanError> {
    let observed_blocking_ns = interval.stores.iter().try_fold(0u128, |total, store| {
        total
            .checked_add(u128::from(store.blocking_time_ns))
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(
                    "working-set remount blocking time overflowed".to_string(),
                )
            })
    })?;
    let mut observed_payload_bytes = 0u128;
    let mut lost_hot_payload_bytes = 0u128;
    for (cumulative_store, interval_store) in cumulative.stores.iter().zip(&interval.stores) {
        let observed_store_bytes = interval_store
            .components
            .iter()
            .try_fold(0u128, |total, component| {
                total
                    .checked_add(component.selected_payload_bytes as u128)
                    .ok_or_else(|| {
                        VulkanRuntimeResidencyPlanError(
                            "working-set interval payload bytes overflowed".to_string(),
                        )
                    })
            })?;
        observed_payload_bytes = observed_payload_bytes
            .checked_add(observed_store_bytes)
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(
                    "working-set remount observed payload overflowed".to_string(),
                )
            })?;
        let retained = cumulative_store
            .logical_device_ids
            .iter()
            .all(|device_id| retained_logical_device_ids.contains(device_id));
        if !retained {
            let store_hot_bytes = cumulative_store.components.iter().try_fold(
                0u128,
                |total, component| {
                total
                    .checked_add(component.selected_payload_bytes as u128)
                    .ok_or_else(|| {
                        VulkanRuntimeResidencyPlanError(
                            "working-set lost hot payload bytes overflowed".to_string(),
                        )
                    })
                },
            )?;
            lost_hot_payload_bytes = lost_hot_payload_bytes
                .checked_add(store_hot_bytes)
                .ok_or_else(|| {
                    VulkanRuntimeResidencyPlanError(
                        "working-set lost hot payload bytes overflowed".to_string(),
                    )
                })?;
        }
    }
    if observed_blocking_ns > 0 && observed_payload_bytes == 0 {
        return Err(VulkanRuntimeResidencyPlanError(
            "working-set blocking time has no selected payload evidence".to_string(),
        ));
    }
    if observed_payload_bytes == 0 || lost_hot_payload_bytes == 0 {
        return Ok(0);
    }
    observed_blocking_ns
        .checked_mul(lost_hot_payload_bytes)
        .map(|cost| cost / observed_payload_bytes)
        .ok_or_else(|| {
            VulkanRuntimeResidencyPlanError(
                "working-set remount estimate overflowed".to_string(),
            )
        })
}

#[cfg(test)]
mod runtime_working_set_rebalance_tests {
    use super::*;

    fn fixture_store(
        store_id: &str,
        physical_device_id: &str,
        component_id: &str,
    ) -> VulkanRuntimeDeviceWorkingSetPressure {
        VulkanRuntimeDeviceWorkingSetPressure {
            store_id: store_id.to_string(),
            physical_device_id: physical_device_id.to_string(),
            logical_device_ids: vec![physical_device_id.to_string()],
            components: vec![VulkanRuntimeComponentWorkingSetPressure {
                execution_scope: "target".to_string(),
                component_id: component_id.to_string(),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn working_set_rebalance_requires_strict_positive_net_benefit() {
        assert_eq!(
            working_set_rebalance_net_benefit_ns(100, 40, 10, 12, 20).unwrap(),
            Some(20),
        );
        assert_eq!(
            working_set_rebalance_net_benefit_ns(80, 40, 10, 12, 20).unwrap(),
            None,
        );
        assert_eq!(
            working_set_rebalance_net_benefit_ns(79, 40, 10, 12, 20).unwrap(),
            None,
        );
    }

    #[test]
    fn working_set_rebalance_rejects_pressure_from_another_mount() {
        let cumulative = VulkanRuntimeWorkingSetPressureSnapshot {
            stores: vec![fixture_store("store-a", "device-a", "block_0")],
        };
        let interval = VulkanRuntimeWorkingSetPressureSnapshot {
            stores: vec![fixture_store("store-b", "device-a", "block_0")],
        };

        let error = validate_working_set_pressure_identity(&cumulative, &interval).unwrap_err();

        assert!(error.to_string().contains("different identities"));
    }

    #[test]
    fn observed_component_pressure_rejects_cross_store_aliasing() {
        let pressure = VulkanRuntimeWorkingSetPressureSnapshot {
            stores: vec![
                fixture_store("store-a", "device-a", "block_0"),
                fixture_store("store-b", "device-b", "block_0"),
            ],
        };

        let error = observed_component_pressure(&pressure).unwrap_err();

        assert!(error.to_string().contains("more than one physical store"));
    }

    #[test]
    fn working_set_remount_prices_every_hot_component_on_changed_stores() {
        let mut cumulative_a = fixture_store("store-a", "device-a", "block_0");
        cumulative_a.components[0].selected_payload_bytes = 100;
        let mut cumulative_b = fixture_store("store-b", "device-b", "block_1");
        cumulative_b.components[0].selected_payload_bytes = 200;
        let mut cumulative_c = fixture_store("store-c", "device-c", "block_2");
        cumulative_c.components[0].selected_payload_bytes = 300;
        let mut interval_a = fixture_store("store-a", "device-a", "block_0");
        interval_a.components[0].selected_payload_bytes = 10;
        interval_a.blocking_time_ns = 100;
        let mut interval_b = fixture_store("store-b", "device-b", "block_1");
        interval_b.components[0].selected_payload_bytes = 10;
        interval_b.blocking_time_ns = 200;
        let mut interval_c = fixture_store("store-c", "device-c", "block_2");
        interval_c.components[0].selected_payload_bytes = 10;
        interval_c.blocking_time_ns = 300;
        let retained = ["device-a".to_string(), "device-c".to_string()]
            .into_iter()
            .collect();

        let estimate = estimate_working_set_remount_ns(
            &retained,
            &VulkanRuntimeWorkingSetPressureSnapshot {
                stores: vec![cumulative_a, cumulative_b, cumulative_c],
            },
            &VulkanRuntimeWorkingSetPressureSnapshot {
                stores: vec![interval_a, interval_b, interval_c],
            },
        )
        .unwrap();

        assert_eq!(estimate, 4_000);
    }

    #[test]
    fn working_set_remount_does_not_retain_a_partially_unchanged_physical_store() {
        let mut cumulative = fixture_store("store-ab", "physical-ab", "block_0");
        cumulative.logical_device_ids = vec!["device-a".to_string(), "device-b".to_string()];
        cumulative.components[0].selected_payload_bytes = 120;
        let mut interval = cumulative.clone();
        interval.components[0].selected_payload_bytes = 12;
        interval.blocking_time_ns = 60;

        let estimate = estimate_working_set_remount_ns(
            &["device-a".to_string()].into_iter().collect(),
            &VulkanRuntimeWorkingSetPressureSnapshot {
                stores: vec![cumulative],
            },
            &VulkanRuntimeWorkingSetPressureSnapshot {
                stores: vec![interval],
            },
        )
        .unwrap();

        assert_eq!(estimate, 600);
    }
}
