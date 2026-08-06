use super::{
    LoadedRuntimeImplementation, RuntimeImplementationCatalog,
    RuntimeImplementationSelectionReport, RuntimeRejectedImplementation,
    RuntimeSelectedImplementation, RuntimeSelectionDevice, RuntimeSelectionInstance,
    RuntimeSelectionRequest,
};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::io;

#[derive(Clone)]
struct EligibleApplication<'a> {
    loaded: &'a LoadedRuntimeImplementation,
    instance_ids: Vec<String>,
    selected: RuntimeSelectedImplementation,
}

impl RuntimeImplementationCatalog {
    pub fn select(
        &self,
        request: &RuntimeSelectionRequest,
    ) -> io::Result<RuntimeImplementationSelectionReport> {
        validate_request(request)?;
        let devices = request
            .devices
            .iter()
            .map(|device| (device.logical_device_id.as_str(), device))
            .collect::<BTreeMap<_, _>>();
        let mut eligible = Vec::new();
        let mut rejected = Vec::new();

        for loaded in &self.implementations {
            let applications = independent_region_applications(
                &loaded.mount_plan.regions,
                &request.instances,
                &request.edges,
            );
            if applications.is_empty() {
                rejected.push(RuntimeRejectedImplementation {
                    implementation_id: loaded.implementation.implementation_id.clone(),
                    instance_ids: Vec::new(),
                    reasons: vec![
                        "runtime topology has no complete matching semantic region".to_string(),
                    ],
                });
                continue;
            }
            for instance_ids in applications {
                let physical_devices =
                    application_devices(&instance_ids, &request.instances, &devices)?;
                let reasons = loaded
                    .implementation
                    .runtime_predicate
                    .mismatch_reasons(&request.execution, &physical_devices);
                if !reasons.is_empty() {
                    rejected.push(RuntimeRejectedImplementation {
                        implementation_id: loaded.implementation.implementation_id.clone(),
                        instance_ids,
                        reasons,
                    });
                    continue;
                }
                let metrics = select_metrics(loaded, request)?;
                eligible.push(EligibleApplication {
                    loaded,
                    instance_ids: instance_ids.clone(),
                    selected: RuntimeSelectedImplementation {
                        implementation_id: loaded.implementation.implementation_id.clone(),
                        candidate_id: loaded.implementation.candidate_id.clone(),
                        instance_ids,
                        scope_ids: loaded.implementation.scope_ids.clone(),
                        mount_adapter_id: loaded.mount_plan.adapter_id.clone(),
                        predicate: loaded.implementation.runtime_predicate.clone(),
                        representation: loaded.implementation.representation.clone(),
                        provenance: loaded.implementation.provenance.clone(),
                        benchmark_id: loaded.implementation.comparison.benchmark_id.clone(),
                        validation_id: loaded.implementation.comparison.validation_id.clone(),
                        validation_status: loaded
                            .implementation
                            .comparison
                            .validation_status
                            .clone(),
                        speedup_ppm: metrics.speedup_ppm,
                        estimated_saved_ns: metrics.estimated_saved_ns,
                        conversion_ns: metrics.conversion_ns,
                        conversion_bytes: metrics.conversion_bytes,
                        boundary_count: metrics.boundary_count,
                        decision_reason: loaded.implementation.decision_reason.clone(),
                    },
                });
            }
        }
        eligible.sort_by(application_order);
        let selected_indices = optimal_nonoverlapping_applications(&eligible);
        let selected_index_set = selected_indices.iter().copied().collect::<BTreeSet<_>>();
        let mut selected = selected_indices
            .into_iter()
            .map(|index| eligible[index].selected.clone())
            .collect::<Vec<_>>();
        selected.sort_by(|left, right| {
            (
                left.instance_ids.as_slice(),
                left.implementation_id.as_str(),
            )
                .cmp(&(
                    right.instance_ids.as_slice(),
                    right.implementation_id.as_str(),
                ))
        });

        for (index, application) in eligible.iter().enumerate() {
            if !selected_index_set.contains(&index) {
                rejected.push(RuntimeRejectedImplementation {
                    implementation_id: application
                        .loaded
                        .implementation
                        .implementation_id
                        .clone(),
                    instance_ids: application.instance_ids.clone(),
                    reasons: vec![
                        "a higher-value compatible implementation set covers an overlapping runtime region"
                            .to_string(),
                    ],
                });
            }
        }
        rejected.sort_by(|left, right| {
            (
                left.implementation_id.as_str(),
                left.instance_ids.as_slice(),
            )
                .cmp(&(
                    right.implementation_id.as_str(),
                    right.instance_ids.as_slice(),
                ))
        });

        let covered_instances = selected
            .iter()
            .flat_map(|selection| selection.instance_ids.iter().cloned())
            .collect::<BTreeSet<_>>();
        let exact_instance_ids = request
            .instances
            .iter()
            .map(|instance| instance.instance_id.clone())
            .filter(|instance_id| !covered_instances.contains(instance_id))
            .collect::<Vec<_>>();
        if !request.exact_baseline_compatible && !exact_instance_ids.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!(
                    "no compatible implementation covers runtime instances {:?}; exact baseline is incompatible",
                    exact_instance_ids
                ),
            ));
        }
        let total_estimated_saved_ns = checked_metric_sum(
            selected
                .iter()
                .map(|selection| selection.estimated_saved_ns),
            "estimated saved time",
        )?;
        let total_conversion_ns = checked_metric_sum(
            selected.iter().map(|selection| selection.conversion_ns),
            "conversion time",
        )?;
        let total_conversion_bytes = checked_metric_sum(
            selected.iter().map(|selection| selection.conversion_bytes),
            "conversion bytes",
        )?;
        let total_boundary_count = checked_metric_sum(
            selected.iter().map(|selection| selection.boundary_count),
            "representation boundary count",
        )?;
        Ok(RuntimeImplementationSelectionReport {
            package_id: self.package_id.clone(),
            execution: request.execution.clone(),
            total_estimated_saved_ns,
            total_conversion_ns,
            total_conversion_bytes,
            total_boundary_count,
            selected,
            exact_instance_ids,
            rejected,
        })
    }
}

/// Enumerates independently selectable applications of every declared mount
/// region. A region is the indivisible semantic replacement boundary; separate
/// regions in one verified artifact bundle are alternatives that may be chosen
/// independently according to each application's placement and measured cost.
pub(crate) fn independent_region_applications(
    regions: &[super::RuntimeMountRegion],
    instances: &[RuntimeSelectionInstance],
    edges: &[super::RuntimeSelectionEdge],
) -> Vec<Vec<String>> {
    let mut applications = regions
        .iter()
        .flat_map(|region| {
            let required_sources = region
                .replacements
                .iter()
                .map(|replacement| replacement.source_component_id().to_string())
                .collect::<Vec<_>>();
            matching_applications(&required_sources, instances, edges)
        })
        .collect::<Vec<_>>();
    applications.sort();
    applications.dedup();
    applications
}

fn checked_metric_sum(mut values: impl Iterator<Item = u64>, label: &str) -> io::Result<u64> {
    values.try_fold(0u64, |total, value| {
        total.checked_add(value).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("runtime selection {label} exceeds the report range"),
            )
        })
    })
}

fn validate_request(request: &RuntimeSelectionRequest) -> io::Result<()> {
    if request.execution.phases.is_empty()
        || request.execution.activation_batch.minimum == 0
        || !strictly_sorted_unique(
            &request
                .execution
                .phases
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
        )
        || [
            request.execution.activation_batch,
            request.execution.context_activations,
            request.execution.state_activations,
        ]
        .iter()
        .any(|range| range.minimum > range.maximum)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "runtime selection execution envelope is invalid",
        ));
    }
    let logical_devices = request
        .devices
        .iter()
        .map(|device| device.logical_device_id.as_str())
        .collect::<Vec<_>>();
    if !strictly_sorted_unique(&logical_devices) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "runtime selection devices must be sorted and unique",
        ));
    }
    for device in &request.devices {
        device
            .profile
            .validate()
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
        if device.physical_device_id != device.profile.hardware_identity.stable_device_id {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "runtime selection device binding and hardware profile disagree",
            ));
        }
    }
    let instance_ids = request
        .instances
        .iter()
        .map(|instance| instance.instance_id.as_str())
        .collect::<Vec<_>>();
    if !strictly_sorted_unique(&instance_ids) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "runtime selection instances must be sorted and unique",
        ));
    }
    let known_devices = logical_devices.into_iter().collect::<BTreeSet<_>>();
    for instance in &request.instances {
        if instance.source_component_id.is_empty()
            || instance.logical_device_ids.is_empty()
            || !strictly_sorted_unique(
                &instance
                    .logical_device_ids
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
            )
            || instance
                .logical_device_ids
                .iter()
                .any(|device| !known_devices.contains(device.as_str()))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "runtime selection instance {:?} has invalid device placement",
                    instance.instance_id
                ),
            ));
        }
    }
    Ok(())
}

fn matching_applications(
    required_sources: &[String],
    instances: &[RuntimeSelectionInstance],
    edges: &[super::RuntimeSelectionEdge],
) -> Vec<Vec<String>> {
    let by_source = required_sources
        .iter()
        .map(|source| {
            (
                source.as_str(),
                instances
                    .iter()
                    .filter(|instance| instance.source_component_id == *source)
                    .map(|instance| instance.instance_id.clone())
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    if by_source.iter().any(|(_, matches)| matches.is_empty()) {
        return Vec::new();
    }
    if by_source.len() == 1 {
        return by_source[0]
            .1
            .iter()
            .map(|instance_id| vec![instance_id.clone()])
            .collect();
    }
    let adjacency = edges.iter().fold(
        BTreeMap::<&str, BTreeSet<&str>>::new(),
        |mut adjacency, edge| {
            adjacency
                .entry(edge.source_instance_id.as_str())
                .or_default()
                .insert(edge.destination_instance_id.as_str());
            adjacency
                .entry(edge.destination_instance_id.as_str())
                .or_default()
                .insert(edge.source_instance_id.as_str());
            adjacency
        },
    );
    let mut combinations = Vec::new();
    enumerate_source_combinations(
        &by_source,
        0,
        &mut Vec::new(),
        &adjacency,
        &mut combinations,
    );
    combinations.sort();
    combinations.dedup();
    combinations
}

fn enumerate_source_combinations(
    sources: &[(&str, Vec<String>)],
    index: usize,
    selected: &mut Vec<String>,
    adjacency: &BTreeMap<&str, BTreeSet<&str>>,
    output: &mut Vec<Vec<String>>,
) {
    if index == sources.len() {
        if connected(selected, adjacency) {
            let mut result = selected.clone();
            result.sort();
            output.push(result);
        }
        return;
    }
    for instance_id in &sources[index].1 {
        if selected.contains(instance_id) {
            continue;
        }
        selected.push(instance_id.clone());
        enumerate_source_combinations(sources, index + 1, selected, adjacency, output);
        selected.pop();
    }
}

fn connected(instance_ids: &[String], adjacency: &BTreeMap<&str, BTreeSet<&str>>) -> bool {
    let allowed = instance_ids
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let mut pending = vec![instance_ids[0].as_str()];
    let mut visited = BTreeSet::new();
    while let Some(instance_id) = pending.pop() {
        if !visited.insert(instance_id) {
            continue;
        }
        pending.extend(
            adjacency
                .get(instance_id)
                .into_iter()
                .flat_map(|neighbors| neighbors.iter().copied())
                .filter(|neighbor| allowed.contains(neighbor)),
        );
    }
    visited == allowed
}

fn application_devices<'a>(
    instance_ids: &[String],
    instances: &[RuntimeSelectionInstance],
    devices: &BTreeMap<&str, &'a RuntimeSelectionDevice>,
) -> io::Result<Vec<&'a RuntimeSelectionDevice>> {
    let instance_by_id = instances
        .iter()
        .map(|instance| (instance.instance_id.as_str(), instance))
        .collect::<BTreeMap<_, _>>();
    let mut result = BTreeMap::new();
    for instance_id in instance_ids {
        let instance = instance_by_id.get(instance_id.as_str()).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "runtime selection application references an unknown instance",
            )
        })?;
        for logical_device_id in &instance.logical_device_ids {
            let device = devices.get(logical_device_id.as_str()).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "runtime selection placement references an unknown device",
                )
            })?;
            result.insert(device.physical_device_id.as_str(), *device);
        }
    }
    Ok(result.into_values().collect())
}

fn select_metrics(
    loaded: &LoadedRuntimeImplementation,
    request: &RuntimeSelectionRequest,
) -> io::Result<AggregatedMetrics> {
    let mut selected = Vec::new();
    for phase in request.execution.phases.iter().filter(|phase| {
        loaded
            .implementation
            .runtime_predicate
            .execution
            .alternative_phases
            .contains(phase)
    }) {
        let metrics = loaded
            .workload_metrics
            .iter()
            .filter(|metrics| metrics.phase == *phase)
            .min_by_key(|metrics| {
                (
                    metrics.speedup_ppm,
                    metrics
                        .reference_latency_ns
                        .saturating_sub(metrics.candidate_latency_ns),
                    std::cmp::Reverse(metrics.conversion_ns),
                    metrics.workload_id.as_str(),
                )
            })
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "implementation {:?} has no benchmark for execution phase {phase:?}",
                        loaded.implementation.implementation_id
                    ),
                )
            })?;
        selected.push(metrics);
    }
    let estimated_saved_ns = checked_metric_sum(
        selected.iter().map(|metrics| {
            metrics
                .reference_latency_ns
                .saturating_sub(metrics.candidate_latency_ns)
        }),
        "conservative measured savings",
    )?;
    let conversion_ns = checked_metric_sum(
        selected.iter().map(|metrics| metrics.conversion_ns),
        "conversion time",
    )?;
    let conversion_bytes = checked_metric_sum(
        selected.iter().map(|metrics| metrics.conversion_bytes),
        "conversion bytes",
    )?;
    let boundary_count = checked_metric_sum(
        selected.iter().map(|metrics| metrics.boundary_count),
        "representation boundary count",
    )?;
    Ok(AggregatedMetrics {
        speedup_ppm: selected
            .iter()
            .map(|metrics| metrics.speedup_ppm)
            .min()
            .unwrap_or_default(),
        estimated_saved_ns,
        conversion_ns,
        conversion_bytes,
        boundary_count,
    })
}

struct AggregatedMetrics {
    speedup_ppm: i64,
    estimated_saved_ns: u64,
    conversion_ns: u64,
    conversion_bytes: u64,
    boundary_count: u64,
}

fn application_order(left: &EligibleApplication<'_>, right: &EligibleApplication<'_>) -> Ordering {
    (
        left.instance_ids.as_slice(),
        left.loaded.implementation.implementation_id.as_str(),
    )
        .cmp(&(
            right.instance_ids.as_slice(),
            right.loaded.implementation.implementation_id.as_str(),
        ))
}

fn optimal_nonoverlapping_applications(applications: &[EligibleApplication<'_>]) -> Vec<usize> {
    let suffix_savings = (0..applications.len())
        .rev()
        .scan(0u64, |total, index| {
            *total = total.saturating_add(applications[index].selected.estimated_saved_ns);
            Some(*total)
        })
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>();
    let mut best = SelectionSet::default();
    search_selection_sets(
        applications,
        &suffix_savings,
        0,
        &mut BTreeSet::new(),
        &mut SelectionSet::default(),
        &mut best,
    );
    best.indices
}

#[derive(Clone, Default)]
struct SelectionSet {
    indices: Vec<usize>,
    savings: u64,
    conversions: u64,
}

fn search_selection_sets(
    applications: &[EligibleApplication<'_>],
    suffix_savings: &[u64],
    index: usize,
    occupied_instances: &mut BTreeSet<String>,
    current: &mut SelectionSet,
    best: &mut SelectionSet,
) {
    if index == applications.len() {
        if selection_set_is_better(current, best, applications) {
            *best = current.clone();
        }
        return;
    }
    if current.savings.saturating_add(suffix_savings[index]) < best.savings {
        return;
    }
    let application = &applications[index];
    if application
        .instance_ids
        .iter()
        .all(|instance| !occupied_instances.contains(instance))
    {
        occupied_instances.extend(application.instance_ids.iter().cloned());
        current.indices.push(index);
        current.savings = current
            .savings
            .saturating_add(application.selected.estimated_saved_ns);
        current.conversions = current
            .conversions
            .saturating_add(application.selected.conversion_ns);
        search_selection_sets(
            applications,
            suffix_savings,
            index + 1,
            occupied_instances,
            current,
            best,
        );
        current.indices.pop();
        current.savings = current
            .savings
            .saturating_sub(application.selected.estimated_saved_ns);
        current.conversions = current
            .conversions
            .saturating_sub(application.selected.conversion_ns);
        for instance in &application.instance_ids {
            occupied_instances.remove(instance);
        }
    }
    search_selection_sets(
        applications,
        suffix_savings,
        index + 1,
        occupied_instances,
        current,
        best,
    );
}

fn selection_set_is_better(
    candidate: &SelectionSet,
    current: &SelectionSet,
    applications: &[EligibleApplication<'_>],
) -> bool {
    if candidate.savings != current.savings {
        return candidate.savings > current.savings;
    }
    if candidate.conversions != current.conversions {
        return candidate.conversions < current.conversions;
    }
    let candidate_ids = candidate
        .indices
        .iter()
        .map(|index| {
            applications[*index]
                .loaded
                .implementation
                .implementation_id
                .as_str()
        })
        .collect::<Vec<_>>();
    let current_ids = current
        .indices
        .iter()
        .map(|index| {
            applications[*index]
                .loaded
                .implementation
                .implementation_id
                .as_str()
        })
        .collect::<Vec<_>>();
    candidate_ids < current_ids
}

fn strictly_sorted_unique(values: &[&str]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}
