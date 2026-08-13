use super::{
    LoadedRuntimeImplementation, RuntimeImplementationCatalog,
    RuntimeImplementationSelectionReport, RuntimeImplementationWorkloadMetrics,
    RuntimeRejectedImplementation, RuntimeSelectedImplementation, RuntimeSelectionDevice,
    RuntimeSelectionInstance, RuntimeSelectionRequest,
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
        let (eligible, mut rejected) = eligible_applications(self, request)?;
        let coverable_instances = eligible
            .iter()
            .flat_map(|application| application.instance_ids.iter().cloned())
            .collect::<BTreeSet<_>>();
        let individually_uncoverable = request
            .exact_baseline_incompatible_instance_ids
            .difference(&coverable_instances)
            .collect::<Vec<_>>();
        if !individually_uncoverable.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!(
                    "no compatible implementation can cover runtime instances {individually_uncoverable:?}; their exact baselines are incompatible",
                ),
            ));
        }
        let selected_indices = optimal_nonoverlapping_applications(
            &eligible,
            &request.exact_baseline_incompatible_instance_ids,
        );
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
        let uncovered_incompatible = exact_instance_ids
            .iter()
            .filter(|instance_id| {
                request
                    .exact_baseline_incompatible_instance_ids
                    .contains(instance_id.as_str())
            })
            .collect::<Vec<_>>();
        if !uncovered_incompatible.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!(
                    "no compatible implementation covers runtime instances {:?}; their exact baselines are incompatible",
                    uncovered_incompatible
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
        let total_resource_load_count = checked_metric_sum(
            selected
                .iter()
                .map(|selection| selection.resource_load_count),
            "resource load count",
        )?;
        let total_resource_reload_count = checked_metric_sum(
            selected
                .iter()
                .map(|selection| selection.resource_reload_count),
            "resource reload count",
        )?;
        let total_resource_physical_read_bytes = checked_metric_sum(
            selected
                .iter()
                .map(|selection| selection.resource_physical_read_bytes),
            "resource physical read bytes",
        )?;
        let total_resource_resident_bytes_produced = checked_metric_sum(
            selected
                .iter()
                .map(|selection| selection.resource_resident_bytes_produced),
            "resource resident bytes produced",
        )?;
        let total_resource_uploaded_bytes = checked_metric_sum(
            selected
                .iter()
                .map(|selection| selection.resource_uploaded_bytes),
            "resource uploaded bytes",
        )?;
        let total_resource_read_ns = checked_metric_sum(
            selected.iter().map(|selection| selection.resource_read_ns),
            "resource read time",
        )?;
        let total_resource_derivation_ns = checked_metric_sum(
            selected
                .iter()
                .map(|selection| selection.resource_derivation_ns),
            "resource derivation time",
        )?;
        let total_resource_upload_ns = checked_metric_sum(
            selected
                .iter()
                .map(|selection| selection.resource_upload_ns),
            "resource upload time",
        )?;
        let total_resource_blocking_ns = checked_metric_sum(
            selected
                .iter()
                .map(|selection| selection.resource_blocking_ns),
            "resource blocking time",
        )?;
        Ok(RuntimeImplementationSelectionReport {
            package_id: self.package_id.clone(),
            execution: request.execution.clone(),
            total_estimated_saved_ns,
            total_conversion_ns,
            total_conversion_bytes,
            total_boundary_count,
            total_resource_load_count,
            total_resource_reload_count,
            total_resource_physical_read_bytes,
            total_resource_resident_bytes_produced,
            total_resource_uploaded_bytes,
            total_resource_read_ns,
            total_resource_derivation_ns,
            total_resource_upload_ns,
            total_resource_blocking_ns,
            selected,
            exact_instance_ids,
            rejected,
        })
    }

    /// Returns one selection report for every independently applicable,
    /// compiler-validated replacement region. Unlike `select`, this does not
    /// choose a winning non-overlapping set. It is intended for exhaustive
    /// artifact calibration: every representation that could participate in a
    /// legal runtime selection can be mounted and measured in isolation while
    /// all unrelated instances retain the exact baseline.
    pub fn independent_application_selections(
        &self,
        request: &RuntimeSelectionRequest,
    ) -> io::Result<Vec<RuntimeImplementationSelectionReport>> {
        let (eligible, rejected) = eligible_applications(self, request)?;
        eligible
            .into_iter()
            .map(|application| {
                selection_report_for_independent_application(
                    self,
                    request,
                    application.selected,
                    rejected.clone(),
                )
            })
            .collect()
    }

    /// Returns the exact set of selections calibration must mount: every
    /// independently applicable region plus the globally selected compatible
    /// set. Equivalent selections are canonicalized so a one-region winner is
    /// not measured twice.
    pub fn calibration_selections(
        &self,
        request: &RuntimeSelectionRequest,
    ) -> io::Result<Vec<RuntimeImplementationSelectionReport>> {
        let mut reports = self.independent_application_selections(request)?;
        let selected = self.select(request)?;
        if !selected.selected.is_empty() {
            reports.push(selected);
        }
        reports.sort_by(|left, right| selection_identity(left).cmp(&selection_identity(right)));
        reports.dedup_by(|left, right| selection_identity(left) == selection_identity(right));
        Ok(reports)
    }

    /// Reconstructs one canonical, fully validated report from applications
    /// selected by a measured physical planner. The physical planner may rank
    /// representations differently from their isolated compiler benchmark,
    /// but it may not invent, alter, overlap, or leave required applications
    /// uncovered.
    pub fn selection_report_for_applications(
        &self,
        request: &RuntimeSelectionRequest,
        mut selected: Vec<RuntimeSelectedImplementation>,
    ) -> io::Result<RuntimeImplementationSelectionReport> {
        let (eligible, mut rejected) = eligible_applications(self, request)?;
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
        if selected.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "runtime physical selection contains a duplicate implementation application",
            ));
        }

        let mut covered_instances = BTreeSet::new();
        let mut chosen_applications: Vec<&EligibleApplication<'_>> = Vec::new();
        for chosen in &selected {
            let application = eligible
                .iter()
                .find(|application| application.selected == *chosen)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!(
                            "runtime physical selection contains an ineligible or altered application {:?} for instances {:?}",
                            chosen.implementation_id, chosen.instance_ids,
                        ),
                    )
                })?;
            if chosen_applications
                .iter()
                .any(|existing| !applications_are_composable(existing, application))
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "runtime physical selection contains conflicting implementation applications",
                ));
            }
            covered_instances.extend(chosen.instance_ids.iter().cloned());
            chosen_applications.push(application);
        }

        let exact_instance_ids = request
            .instances
            .iter()
            .map(|instance| instance.instance_id.clone())
            .filter(|instance_id| !covered_instances.contains(instance_id))
            .collect::<Vec<_>>();
        let uncovered_incompatible = exact_instance_ids
            .iter()
            .filter(|instance_id| {
                request
                    .exact_baseline_incompatible_instance_ids
                    .contains(instance_id.as_str())
            })
            .collect::<Vec<_>>();
        if !uncovered_incompatible.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!(
                    "runtime physical selection leaves incompatible exact runtime instances {uncovered_incompatible:?}",
                ),
            ));
        }

        for application in eligible.iter().filter(|application| {
            !selected
                .iter()
                .any(|chosen| chosen == &application.selected)
        }) {
            rejected.push(RuntimeRejectedImplementation {
                implementation_id: application.loaded.implementation.implementation_id.clone(),
                instance_ids: application.instance_ids.clone(),
                reasons: vec![
                    "a measured physical execution plan selected another compatible representation"
                        .to_string(),
                ],
            });
        }
        rejected.sort_by(|left, right| {
            (
                left.implementation_id.as_str(),
                left.instance_ids.as_slice(),
                left.reasons.as_slice(),
            )
                .cmp(&(
                    right.implementation_id.as_str(),
                    right.instance_ids.as_slice(),
                    right.reasons.as_slice(),
                ))
        });
        rejected.dedup();

        Ok(RuntimeImplementationSelectionReport {
            package_id: self.package_id.clone(),
            execution: request.execution.clone(),
            total_estimated_saved_ns: checked_metric_sum(
                selected
                    .iter()
                    .map(|application| application.estimated_saved_ns),
                "estimated saved time",
            )?,
            total_conversion_ns: checked_metric_sum(
                selected.iter().map(|application| application.conversion_ns),
                "conversion time",
            )?,
            total_conversion_bytes: checked_metric_sum(
                selected
                    .iter()
                    .map(|application| application.conversion_bytes),
                "conversion bytes",
            )?,
            total_boundary_count: checked_metric_sum(
                selected
                    .iter()
                    .map(|application| application.boundary_count),
                "representation boundary count",
            )?,
            total_resource_load_count: checked_metric_sum(
                selected
                    .iter()
                    .map(|application| application.resource_load_count),
                "resource load count",
            )?,
            total_resource_reload_count: checked_metric_sum(
                selected
                    .iter()
                    .map(|application| application.resource_reload_count),
                "resource reload count",
            )?,
            total_resource_physical_read_bytes: checked_metric_sum(
                selected
                    .iter()
                    .map(|application| application.resource_physical_read_bytes),
                "resource physical read bytes",
            )?,
            total_resource_resident_bytes_produced: checked_metric_sum(
                selected
                    .iter()
                    .map(|application| application.resource_resident_bytes_produced),
                "resource resident bytes produced",
            )?,
            total_resource_uploaded_bytes: checked_metric_sum(
                selected
                    .iter()
                    .map(|application| application.resource_uploaded_bytes),
                "resource uploaded bytes",
            )?,
            total_resource_read_ns: checked_metric_sum(
                selected
                    .iter()
                    .map(|application| application.resource_read_ns),
                "resource read time",
            )?,
            total_resource_derivation_ns: checked_metric_sum(
                selected
                    .iter()
                    .map(|application| application.resource_derivation_ns),
                "resource derivation time",
            )?,
            total_resource_upload_ns: checked_metric_sum(
                selected
                    .iter()
                    .map(|application| application.resource_upload_ns),
                "resource upload time",
            )?,
            total_resource_blocking_ns: checked_metric_sum(
                selected
                    .iter()
                    .map(|application| application.resource_blocking_ns),
                "resource blocking time",
            )?,
            selected,
            exact_instance_ids,
            rejected,
        })
    }
}

fn selection_identity(selection: &RuntimeImplementationSelectionReport) -> Vec<(&str, &[String])> {
    selection
        .selected
        .iter()
        .map(|selected| {
            (
                selected.implementation_id.as_str(),
                selected.instance_ids.as_slice(),
            )
        })
        .collect()
}

fn eligible_applications<'a>(
    catalog: &'a RuntimeImplementationCatalog,
    request: &RuntimeSelectionRequest,
) -> io::Result<(
    Vec<EligibleApplication<'a>>,
    Vec<RuntimeRejectedImplementation>,
)> {
    validate_request(request)?;
    let devices = request
        .devices
        .iter()
        .map(|device| (device.logical_device_id.as_str(), device))
        .collect::<BTreeMap<_, _>>();
    let mut eligible = Vec::new();
    let mut rejected = Vec::new();
    for loaded in &catalog.implementations {
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
                    source_contract_digests: loaded.implementation.source_contract_digests.clone(),
                    mount_adapter_id: loaded.mount_plan.adapter_id.clone(),
                    predicate: loaded.implementation.runtime_predicate.clone(),
                    representation: loaded.implementation.representation.clone(),
                    provenance: loaded.implementation.provenance.clone(),
                    benchmark_id: loaded.implementation.comparison.benchmark_id.clone(),
                    validation_id: loaded.implementation.comparison.validation_id.clone(),
                    validation_status: loaded.implementation.comparison.validation_status.clone(),
                    speedup_ppm: metrics.speedup_ppm,
                    estimated_saved_ns: metrics.estimated_saved_ns,
                    conversion_ns: metrics.conversion_ns,
                    conversion_bytes: metrics.conversion_bytes,
                    boundary_count: metrics.boundary_count,
                    resource_load_count: metrics.resource_load_count,
                    resource_reload_count: metrics.resource_reload_count,
                    resource_physical_read_bytes: metrics.resource_physical_read_bytes,
                    resource_resident_bytes_produced: metrics.resource_resident_bytes_produced,
                    resource_uploaded_bytes: metrics.resource_uploaded_bytes,
                    resource_read_ns: metrics.resource_read_ns,
                    resource_derivation_ns: metrics.resource_derivation_ns,
                    resource_upload_ns: metrics.resource_upload_ns,
                    resource_blocking_ns: metrics.resource_blocking_ns,
                    decision_reason: loaded.implementation.decision_reason.clone(),
                },
            });
        }
    }
    eligible.sort_by(application_order);
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
    Ok((eligible, rejected))
}

fn selection_report_for_independent_application(
    catalog: &RuntimeImplementationCatalog,
    request: &RuntimeSelectionRequest,
    selected: RuntimeSelectedImplementation,
    rejected: Vec<RuntimeRejectedImplementation>,
) -> io::Result<RuntimeImplementationSelectionReport> {
    let covered_instances = selected.instance_ids.iter().collect::<BTreeSet<_>>();
    let exact_instance_ids = request
        .instances
        .iter()
        .map(|instance| instance.instance_id.clone())
        .filter(|instance_id| !covered_instances.contains(instance_id))
        .collect::<Vec<_>>();
    let uncovered_incompatible = exact_instance_ids
        .iter()
        .filter(|instance_id| {
            request
                .exact_baseline_incompatible_instance_ids
                .contains(instance_id.as_str())
        })
        .collect::<Vec<_>>();
    if !uncovered_incompatible.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!(
                "an independent implementation application leaves incompatible exact runtime instances {uncovered_incompatible:?}",
            ),
        ));
    }
    Ok(RuntimeImplementationSelectionReport {
        package_id: catalog.package_id.clone(),
        execution: request.execution.clone(),
        total_estimated_saved_ns: selected.estimated_saved_ns,
        total_conversion_ns: selected.conversion_ns,
        total_conversion_bytes: selected.conversion_bytes,
        total_boundary_count: selected.boundary_count,
        total_resource_load_count: selected.resource_load_count,
        total_resource_reload_count: selected.resource_reload_count,
        total_resource_physical_read_bytes: selected.resource_physical_read_bytes,
        total_resource_resident_bytes_produced: selected.resource_resident_bytes_produced,
        total_resource_uploaded_bytes: selected.resource_uploaded_bytes,
        total_resource_read_ns: selected.resource_read_ns,
        total_resource_derivation_ns: selected.resource_derivation_ns,
        total_resource_upload_ns: selected.resource_upload_ns,
        total_resource_blocking_ns: selected.resource_blocking_ns,
        selected: vec![selected],
        exact_instance_ids,
        rejected,
    })
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
    if request
        .exact_baseline_incompatible_instance_ids
        .iter()
        .any(|instance_id| !instance_ids.contains(&instance_id.as_str()))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "runtime selection exact-baseline incompatibility references an unknown instance",
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
    let lifecycle_metrics = loaded
        .workload_metrics
        .iter()
        .filter(|metrics| {
            request.execution.phases.contains(&metrics.phase)
                && loaded
                    .implementation
                    .runtime_predicate
                    .execution
                    .alternative_phases
                    .contains(&metrics.phase)
        })
        .max_by_key(|metrics| {
            (
                metrics.resource_reload_count,
                metrics.resource_blocking_ns,
                metrics.resource_physical_read_bytes,
                metrics.resource_load_count,
                metrics.resource_resident_bytes_produced,
                metrics.resource_uploaded_bytes,
                metrics.resource_read_ns,
                metrics.resource_derivation_ns,
                metrics.resource_upload_ns,
                metrics.workload_id.as_str(),
            )
        });
    let lifecycle = |field: fn(&RuntimeImplementationWorkloadMetrics) -> u64| {
        lifecycle_metrics.map(field).unwrap_or_default()
    };
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
        resource_load_count: lifecycle(|metrics| metrics.resource_load_count),
        resource_reload_count: lifecycle(|metrics| metrics.resource_reload_count),
        resource_physical_read_bytes: lifecycle(|metrics| metrics.resource_physical_read_bytes),
        resource_resident_bytes_produced: lifecycle(|metrics| {
            metrics.resource_resident_bytes_produced
        }),
        resource_uploaded_bytes: lifecycle(|metrics| metrics.resource_uploaded_bytes),
        resource_read_ns: lifecycle(|metrics| metrics.resource_read_ns),
        resource_derivation_ns: lifecycle(|metrics| metrics.resource_derivation_ns),
        resource_upload_ns: lifecycle(|metrics| metrics.resource_upload_ns),
        resource_blocking_ns: lifecycle(|metrics| metrics.resource_blocking_ns),
    })
}

struct AggregatedMetrics {
    speedup_ppm: i64,
    estimated_saved_ns: u64,
    conversion_ns: u64,
    conversion_bytes: u64,
    boundary_count: u64,
    resource_load_count: u64,
    resource_reload_count: u64,
    resource_physical_read_bytes: u64,
    resource_resident_bytes_produced: u64,
    resource_uploaded_bytes: u64,
    resource_read_ns: u64,
    resource_derivation_ns: u64,
    resource_upload_ns: u64,
    resource_blocking_ns: u64,
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

fn application_is_region_only(application: &EligibleApplication<'_>) -> bool {
    application
        .loaded
        .mount_plan
        .regions
        .iter()
        .flat_map(|region| &region.replacements)
        .all(super::RuntimeReplacement::is_component_region)
}

fn applications_are_composable(
    left: &EligibleApplication<'_>,
    right: &EligibleApplication<'_>,
) -> bool {
    let shared_instance = left
        .instance_ids
        .iter()
        .any(|instance| right.instance_ids.contains(instance));
    if !shared_instance {
        return true;
    }
    let overlapping_scope = left
        .loaded
        .implementation
        .scope_ids
        .iter()
        .any(|scope| right.loaded.implementation.scope_ids.contains(scope));
    !overlapping_scope && (application_is_region_only(left) || application_is_region_only(right))
}

fn optimal_nonoverlapping_applications(
    applications: &[EligibleApplication<'_>],
    required_covered_instances: &BTreeSet<String>,
) -> Vec<usize> {
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
        &mut BTreeMap::new(),
        &mut SelectionSet::default(),
        &mut best,
        required_covered_instances,
    );
    best.indices
}

#[derive(Clone, Default)]
struct SelectionSet {
    indices: Vec<usize>,
    savings: u64,
    resource_reloads: u64,
    resource_blocking_ns: u64,
    resource_physical_read_bytes: u64,
    resource_loads: u64,
    conversions: u64,
    feasible: bool,
}

fn search_selection_sets(
    applications: &[EligibleApplication<'_>],
    suffix_savings: &[u64],
    index: usize,
    occupied_instances: &mut BTreeMap<String, usize>,
    current: &mut SelectionSet,
    best: &mut SelectionSet,
    required_covered_instances: &BTreeSet<String>,
) {
    if index == applications.len() {
        if required_covered_instances
            .iter()
            .all(|instance| occupied_instances.contains_key(instance))
            && selection_set_is_better(current, best, applications)
        {
            *best = current.clone();
            best.feasible = true;
        }
        return;
    }
    if best.feasible && current.savings.saturating_add(suffix_savings[index]) < best.savings {
        return;
    }
    let application = &applications[index];
    if current
        .indices
        .iter()
        .all(|selected| applications_are_composable(&applications[*selected], application))
    {
        for instance in &application.instance_ids {
            *occupied_instances.entry(instance.clone()).or_default() += 1;
        }
        current.indices.push(index);
        current.savings = current
            .savings
            .saturating_add(application.selected.estimated_saved_ns);
        current.resource_reloads = current
            .resource_reloads
            .saturating_add(application.selected.resource_reload_count);
        current.resource_blocking_ns = current
            .resource_blocking_ns
            .saturating_add(application.selected.resource_blocking_ns);
        current.resource_physical_read_bytes = current
            .resource_physical_read_bytes
            .saturating_add(application.selected.resource_physical_read_bytes);
        current.resource_loads = current
            .resource_loads
            .saturating_add(application.selected.resource_load_count);
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
            required_covered_instances,
        );
        current.indices.pop();
        current.savings = current
            .savings
            .saturating_sub(application.selected.estimated_saved_ns);
        current.resource_reloads = current
            .resource_reloads
            .saturating_sub(application.selected.resource_reload_count);
        current.resource_blocking_ns = current
            .resource_blocking_ns
            .saturating_sub(application.selected.resource_blocking_ns);
        current.resource_physical_read_bytes = current
            .resource_physical_read_bytes
            .saturating_sub(application.selected.resource_physical_read_bytes);
        current.resource_loads = current
            .resource_loads
            .saturating_sub(application.selected.resource_load_count);
        current.conversions = current
            .conversions
            .saturating_sub(application.selected.conversion_ns);
        for instance in &application.instance_ids {
            let count = occupied_instances
                .get_mut(instance)
                .expect("selected application must occupy every instance");
            *count -= 1;
            if *count == 0 {
                occupied_instances.remove(instance);
            }
        }
    }
    search_selection_sets(
        applications,
        suffix_savings,
        index + 1,
        occupied_instances,
        current,
        best,
        required_covered_instances,
    );
}

fn selection_set_is_better(
    candidate: &SelectionSet,
    current: &SelectionSet,
    applications: &[EligibleApplication<'_>],
) -> bool {
    if !current.feasible {
        return true;
    }
    if candidate.savings != current.savings {
        return candidate.savings > current.savings;
    }
    if candidate.resource_reloads != current.resource_reloads {
        return candidate.resource_reloads < current.resource_reloads;
    }
    if candidate.resource_blocking_ns != current.resource_blocking_ns {
        return candidate.resource_blocking_ns < current.resource_blocking_ns;
    }
    if candidate.resource_physical_read_bytes != current.resource_physical_read_bytes {
        return candidate.resource_physical_read_bytes < current.resource_physical_read_bytes;
    }
    if candidate.resource_loads != current.resource_loads {
        return candidate.resource_loads < current.resource_loads;
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
