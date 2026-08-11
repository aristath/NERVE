use std::collections::{BTreeMap, BTreeSet};

use crate::model::{
    BenchmarkPlan, BenchmarkRun, ComparisonSet, Implementation, Measurement, PLAN_SCHEMA,
    RUN_SCHEMA, RunPolicy, Selection, Target, WorkloadSpec, now_unix_ms,
};
use crate::vulkan_features::{
    EXTERNAL_MEMORY_DMA_BUF_FEATURE, EXTERNAL_MEMORY_HOST_FEATURE,
    EXTERNAL_TIMELINE_SEMAPHORE_FEATURE,
};

const SMALL_PAYLOAD_COMPARISON_GROUP: &str = "small_payload_placement_comparison";
pub(crate) const MULTI_TARGET_TENSOR_PARALLEL_STRATEGY: &str = "multi_target_tensor_parallel";

pub fn plan_benchmarks(
    discovered_targets: Vec<Target>,
    selection: Selection,
    mut policy: RunPolicy,
) -> BenchmarkPlan {
    policy.max_group_size = policy
        .max_group_size
        .max(1)
        .min(selection.selected_target_ids.len().max(1));
    let format_count = policy.benchmark_formats.len();
    let workload_count = policy.benchmark_workloads.len();
    let payload_bytes = policy.payload_bytes;
    let requested_axis_count = format_count * workload_count;
    let executable_vulkan_axis_count = policy
        .benchmark_formats
        .iter()
        .flat_map(|format| {
            policy
                .benchmark_workloads
                .iter()
                .filter(move |workload| benchmark_axis_supported(format, workload))
        })
        .count();
    let tensor_parallel_axis_count = policy
        .benchmark_formats
        .iter()
        .flat_map(|format| {
            policy.benchmark_workloads.iter().filter(move |workload| {
                benchmark_axis_supported(format, workload)
                    && benchmark_supports_tensor_parallel(workload)
            })
        })
        .count();
    let component_chain_axis_count = policy
        .benchmark_formats
        .iter()
        .flat_map(|format| {
            policy.benchmark_workloads.iter().filter(move |workload| {
                benchmark_axis_supported(format, workload)
                    && benchmark_supports_component_chain(workload)
            })
        })
        .count();
    let selected = selection
        .selected_target_ids
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let selected_vulkan_targets = discovered_targets
        .iter()
        .filter(|target| selected.contains(&target.stable_target_id) && target.backend == "vulkan")
        .collect::<Vec<_>>();
    let selected_vulkan_count = selected_vulkan_targets.len();
    let eligible_pair_count = tensor_parallel_combination_count(&selected_vulkan_targets, 2);
    let estimated_single_measurement_count =
        selected_vulkan_count * (executable_vulkan_axis_count + component_chain_axis_count);
    let estimated_pair_measurement_count = if policy.pair_measurements && policy.max_group_size >= 2
    {
        let ordered_routes = selected_vulkan_count * selected_vulkan_count.saturating_sub(1);
        let eligible_ordered_pairs = eligible_pair_count * 2;
        ordered_routes * component_chain_axis_count
            + eligible_ordered_pairs * tensor_parallel_axis_count * 2
    } else {
        0
    };
    let estimated_group_measurement_count = expected_group_measurement_count(
        &selected_vulkan_targets,
        &policy,
        tensor_parallel_axis_count,
        component_chain_axis_count,
    );
    let estimated_comparison_set_count = 0;
    let estimated_measurement_count = estimated_single_measurement_count
        + estimated_pair_measurement_count
        + estimated_group_measurement_count;
    let mut diagnostics = selection.diagnostics.clone();
    diagnostics.push("dry_plan_only_no_benchmark_measurements_were_executed".to_string());
    diagnostics.push(format!(
        "requested_axes={requested_axis_count} formats={format_count} workloads={workload_count}"
    ));
    BenchmarkPlan {
        schema: PLAN_SCHEMA.to_string(),
        created_at_unix_ms: now_unix_ms(),
        policy,
        discovered_target_count: discovered_targets.len(),
        selected_target_ids: selection.selected_target_ids,
        skipped_targets: selection.skipped_targets,
        requested_format_count: format_count,
        requested_workload_count: workload_count,
        estimated_single_measurement_count,
        estimated_pair_measurement_count,
        estimated_group_measurement_count,
        estimated_comparison_set_count,
        estimated_measurement_count,
        max_payload_bytes_per_measurement: payload_bytes,
        diagnostics,
    }
}

pub fn run_benchmarks(
    discovered_targets: Vec<Target>,
    selection: Selection,
    mut policy: RunPolicy,
) -> BenchmarkRun {
    policy.max_group_size = policy
        .max_group_size
        .max(1)
        .min(selection.selected_target_ids.len().max(1));
    let started_at_unix_ms = now_unix_ms();
    let selected = selection
        .selected_target_ids
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let selected_targets = discovered_targets
        .iter()
        .filter(|target| selected.contains(&target.stable_target_id))
        .collect::<Vec<_>>();

    let mut measurements = Vec::new();
    let vulkan_targets = selected_targets
        .iter()
        .copied()
        .filter(|target| target.backend == "vulkan")
        .collect::<Vec<_>>();
    let workload_specs = build_workload_specs(
        policy.payload_bytes,
        &policy.benchmark_formats,
        &policy.benchmark_workloads,
        policy.max_group_size,
    );
    let comparison_sets = Vec::new();
    let vulkan = policy.execute.then(|| {
        crate::vulkan_exec::run_vulkan_benchmarks(
            &vulkan_targets,
            policy.payload_bytes,
            policy.samples,
            &policy.benchmark_formats,
            &policy.benchmark_workloads,
            policy
                .pair_measurements
                .then_some(policy.max_group_size)
                .unwrap_or(1),
        )
    });
    if let Some(vulkan) = vulkan {
        measurements.extend(vulkan.measurements);
        let pair_measurements = vulkan.pair_measurements;
        let group_measurements = vulkan.group_measurements;
        return finish_benchmark_run(
            started_at_unix_ms,
            discovered_targets,
            selection,
            policy,
            workload_specs,
            comparison_sets,
            measurements,
            pair_measurements,
            group_measurements,
        );
    }

    finish_benchmark_run(
        started_at_unix_ms,
        discovered_targets,
        selection,
        policy,
        workload_specs,
        comparison_sets,
        measurements,
        Vec::new(),
        Vec::new(),
    )
}

pub fn validate_execution_coverage(run: &BenchmarkRun) -> Result<(), String> {
    if !run.policy.execute {
        return Ok(());
    }
    let selected = run
        .selected_target_ids
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let vulkan_targets = run
        .discovered_targets
        .iter()
        .filter(|target| target.backend == "vulkan" && selected.contains(&target.stable_target_id))
        .collect::<Vec<_>>();
    let vulkan_ids = vulkan_targets
        .iter()
        .map(|target| target.stable_target_id.as_str())
        .collect::<BTreeSet<_>>();
    let selected_non_vulkan = run
        .discovered_targets
        .iter()
        .filter(|target| selected.contains(&target.stable_target_id) && target.backend != "vulkan")
        .map(|target| target.stable_target_id.as_str())
        .collect::<Vec<_>>();
    let executable_axes = run
        .policy
        .benchmark_formats
        .iter()
        .flat_map(|format| {
            run.policy
                .benchmark_workloads
                .iter()
                .filter(move |workload| benchmark_axis_supported(format, workload))
        })
        .count();
    let tensor_parallel_axes =
        benchmark_axis_count(&run.policy, benchmark_supports_tensor_parallel);
    let component_chain_axes =
        benchmark_axis_count(&run.policy, benchmark_supports_component_chain);
    let expected_single = vulkan_ids.len() * (executable_axes + component_chain_axes);
    let actual_single = run
        .measurements
        .iter()
        .filter(|measurement| vulkan_ids.contains(measurement.target_id.as_str()))
        .count();
    let ordered_pairs = vulkan_ids.len() * vulkan_ids.len().saturating_sub(1);
    let eligible_pairs = tensor_parallel_combination_count(&vulkan_targets, 2);
    let expected_pairs = if run.policy.pair_measurements && run.policy.max_group_size >= 2 {
        ordered_pairs * component_chain_axes + eligible_pairs * 2 * tensor_parallel_axes * 2
    } else {
        0
    };
    let expected_groups = expected_group_measurement_count(
        &vulkan_targets,
        &run.policy,
        tensor_parallel_axes,
        component_chain_axes,
    );
    let expected_comparison_sets = 0;
    let mut errors = Vec::new();
    if vulkan_ids.is_empty() {
        errors.push("no executable Vulkan target was selected".to_string());
    }
    for target_id in selected_non_vulkan {
        errors.push(format!(
            "selected target {target_id} has no Vulkan execution backend"
        ));
    }
    for (label, actual, expected) in [
        ("single", actual_single, expected_single),
        ("pair", run.pair_measurements.len(), expected_pairs),
        ("group", run.group_measurements.len(), expected_groups),
        (
            "comparison-set",
            run.comparison_sets.len(),
            expected_comparison_sets,
        ),
    ] {
        if actual != expected {
            errors.push(format!(
                "{label} coverage has {actual} rows but requires {expected}"
            ));
        }
    }
    for (identity, status, reason, sample_count) in run
        .measurements
        .iter()
        .filter(|measurement| vulkan_ids.contains(measurement.target_id.as_str()))
        .map(|measurement| {
            (
                format!(
                    "single:{}:{}",
                    measurement.target_id, measurement.workload_id
                ),
                measurement.status.as_str(),
                measurement.reason.as_deref(),
                measurement.samples.len(),
            )
        })
        .chain(run.pair_measurements.iter().map(|measurement| {
            (
                format!(
                    "pair:{}->{}:{}",
                    measurement.source_target_id,
                    measurement.destination_target_id,
                    measurement.workload_id
                ),
                measurement.status.as_str(),
                measurement.reason.as_deref(),
                measurement.samples.len(),
            )
        }))
        .chain(run.group_measurements.iter().map(|measurement| {
            (
                format!(
                    "group:{}:{}",
                    measurement.target_ids.join("->"),
                    measurement.workload_id
                ),
                measurement.status.as_str(),
                measurement.reason.as_deref(),
                measurement.samples.len(),
            )
        }))
    {
        if status == "completed" && sample_count != run.policy.samples {
            errors.push(format!(
                "{identity} has {sample_count} samples but requires {}",
                run.policy.samples
            ));
        } else if status != "completed" && reason.is_none_or(str::is_empty) {
            errors.push(format!("{identity} is {status} without a reason"));
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        let omitted = errors.len().saturating_sub(8);
        errors.truncate(8);
        let mut message = format!("benchmark coverage is incomplete: {}", errors.join("; "));
        if omitted > 0 {
            message.push_str(&format!("; and {omitted} more failures"));
        }
        Err(message)
    }
}

fn finish_benchmark_run(
    started_at_unix_ms: u128,
    discovered_targets: Vec<Target>,
    selection: Selection,
    policy: RunPolicy,
    workload_specs: Vec<WorkloadSpec>,
    comparison_sets: Vec<ComparisonSet>,
    measurements: Vec<Measurement>,
    pair_measurements: Vec<crate::model::PairMeasurement>,
    group_measurements: Vec<crate::model::GroupMeasurement>,
) -> BenchmarkRun {
    let mut diagnostics = selection.diagnostics.clone();
    diagnostics.push("Only real Vulkan execution rows can become placement evidence.".to_string());
    diagnostics.push(format!(
        "Each synthetic workload is capped to {} payload bytes.",
        policy.payload_bytes
    ));

    BenchmarkRun {
        schema: RUN_SCHEMA.to_string(),
        started_at_unix_ms,
        finished_at_unix_ms: now_unix_ms(),
        implementation: Implementation::current(),
        policy,
        discovered_targets,
        selected_target_ids: selection.selected_target_ids,
        skipped_targets: selection.skipped_targets,
        workload_specs,
        comparison_sets,
        measurements,
        pair_measurements,
        group_measurements,
        diagnostics,
    }
}

pub(crate) fn target_index_combinations(count: usize, choose: usize) -> Vec<Vec<usize>> {
    fn extend(
        count: usize,
        start: usize,
        remaining: usize,
        current: &mut Vec<usize>,
        result: &mut Vec<Vec<usize>>,
    ) {
        if remaining == 0 {
            result.push(current.clone());
            return;
        }
        for index in start..=count - remaining {
            current.push(index);
            extend(count, index + 1, remaining - 1, current, result);
            current.pop();
        }
    }

    if choose == 0 {
        return vec![Vec::new()];
    }
    if choose > count {
        return Vec::new();
    }
    let mut result = Vec::new();
    extend(
        count,
        0,
        choose,
        &mut Vec::with_capacity(choose),
        &mut result,
    );
    result
}

/// Expands device groups one participant at a time. Pair evidence is always
/// the first multi-target stage; every larger group has a measured predecessor
/// instead of appearing through an unrelated fixed-width sweep.
pub(crate) fn staged_target_index_groups(
    count: usize,
    max_group_size: usize,
) -> BTreeMap<usize, Vec<Vec<usize>>> {
    let max_group_size = max_group_size.min(count);
    if max_group_size < 2 {
        return BTreeMap::new();
    }
    let mut stages = BTreeMap::from([(2, target_index_combinations(count, 2))]);
    for group_size in 3..=max_group_size {
        let mut expanded = BTreeSet::new();
        for predecessor in &stages[&(group_size - 1)] {
            let start = predecessor.last().copied().unwrap_or(0) + 1;
            for next in start..count {
                let mut candidate = predecessor.clone();
                candidate.push(next);
                expanded.insert(candidate);
            }
        }
        if expanded.is_empty() {
            break;
        }
        stages.insert(group_size, expanded.into_iter().collect());
    }
    stages
}

pub(crate) fn staged_tensor_parallel_target_groups(
    targets: &[&Target],
    max_group_size: usize,
) -> BTreeMap<usize, Vec<Vec<usize>>> {
    let mut viable = BTreeMap::<usize, Vec<Vec<usize>>>::new();
    for (group_size, candidates) in staged_target_index_groups(targets.len(), max_group_size) {
        let candidates = candidates
            .into_iter()
            .filter(|indices| {
                let group = indices
                    .iter()
                    .map(|index| targets[*index])
                    .collect::<Vec<_>>();
                targets_support_tensor_parallel(&group)
            })
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            break;
        }
        viable.insert(group_size, candidates);
    }
    viable
}

pub(crate) fn targets_support_tensor_parallel(targets: &[&Target]) -> bool {
    !targets.is_empty()
        && targets
            .iter()
            .all(|target| target_has_vulkan_feature(target, EXTERNAL_TIMELINE_SEMAPHORE_FEATURE))
        && (targets
            .iter()
            .all(|target| target_has_vulkan_feature(target, EXTERNAL_MEMORY_DMA_BUF_FEATURE))
            || targets
                .iter()
                .all(|target| target_has_vulkan_feature(target, EXTERNAL_MEMORY_HOST_FEATURE)))
}

fn target_has_vulkan_feature(target: &Target, feature: &str) -> bool {
    target
        .vulkan
        .as_ref()
        .is_some_and(|vulkan| vulkan.feature_flags.iter().any(|item| item == feature))
}

fn tensor_parallel_combination_count(targets: &[&Target], choose: usize) -> usize {
    target_index_combinations(targets.len(), choose)
        .into_iter()
        .filter(|indices| {
            let group = indices
                .iter()
                .map(|index| targets[*index])
                .collect::<Vec<_>>();
            targets_support_tensor_parallel(&group)
        })
        .count()
}

fn expected_group_measurement_count(
    targets: &[&Target],
    policy: &RunPolicy,
    tensor_parallel_axis_count: usize,
    component_chain_axis_count: usize,
) -> usize {
    if !policy.pair_measurements || policy.max_group_size < 3 {
        return 0;
    }
    let max_group_size = policy.max_group_size.min(targets.len());
    staged_tensor_parallel_target_groups(targets, max_group_size)
        .into_iter()
        .filter(|(group_size, _)| *group_size >= 3)
        .map(|(group_size, groups)| {
            let compatible_groups = groups.len();
            compatible_groups
                * (tensor_parallel_axis_count * group_size * 2 + component_chain_axis_count)
        })
        .sum()
}

fn benchmark_workload_family(workload: &str) -> &str {
    for family in [
        "dense_projection",
        "moe_expert",
        "kv_cache",
        "router_reduction",
    ] {
        if workload == family || workload.starts_with(&format!("{family}_")) {
            return family;
        }
    }
    workload
}

fn benchmark_supports_tensor_parallel(workload: &str) -> bool {
    matches!(
        benchmark_workload_family(workload),
        "dense_projection" | "moe_expert"
    )
}

fn benchmark_supports_component_chain(workload: &str) -> bool {
    benchmark_workload_family(workload) == "dense_projection"
}

fn benchmark_axis_count(policy: &RunPolicy, predicate: fn(&str) -> bool) -> usize {
    policy
        .benchmark_formats
        .iter()
        .flat_map(|format| {
            policy.benchmark_workloads.iter().filter(move |workload| {
                benchmark_axis_supported(format, workload) && predicate(workload)
            })
        })
        .count()
}

fn benchmark_axis_supported(format: &str, workload: &str) -> bool {
    benchmark_workload_family(workload) != "kv_cache" || format == "bf16"
}

pub(crate) fn single_target_status_measurements(
    target_id: &str,
    payload_bytes: usize,
    formats: &[String],
    workloads: &[String],
    status: &str,
    reason: &str,
) -> Vec<Measurement> {
    formats
        .iter()
        .flat_map(|format| {
            workloads.iter().map(|workload| {
                single_target_status_measurement(
                    target_id,
                    payload_bytes,
                    workload,
                    format,
                    status,
                    reason,
                )
            })
        })
        .collect()
}

pub(crate) fn single_target_status_measurement(
    target_id: &str,
    payload_bytes: usize,
    workload: &str,
    format: &str,
    status: &str,
    reason: &str,
) -> Measurement {
    Measurement {
        workload_id: format_workload_id("single_target_small_payload", workload, format),
        comparison_group: SMALL_PAYLOAD_COMPARISON_GROUP.to_string(),
        workload_class: workload.to_string(),
        placement_strategy: "single_target_serial".to_string(),
        target_id: target_id.to_string(),
        pattern: "single_target_compute".to_string(),
        operation_family: workload.to_string(),
        regime: "small_payload".to_string(),
        format: format.to_string(),
        status: status.to_string(),
        reason: Some(reason.to_string()),
        payload_bytes,
        working_set_bytes: payload_bytes,
        activation_bytes: activation_bytes_for_payload(payload_bytes),
        output_bytes: output_bytes_for_payload(payload_bytes),
        samples: Vec::new(),
        summary: None,
    }
}

pub(crate) fn component_chain_regime(participant_count: usize) -> String {
    assert!(
        participant_count >= 2,
        "component chain participant count must be at least two"
    );
    format!("component_chain_{participant_count}")
}

pub(crate) fn tensor_parallel_group_workload_id(
    participant_count: usize,
    workload_class: &str,
    format: &str,
) -> String {
    format_workload_id(
        &tensor_parallel_group_pattern(participant_count),
        workload_class,
        format,
    )
}

pub(crate) fn tensor_parallel_group_pattern(participant_count: usize) -> String {
    format!("synthetic_tensor_parallel_group_{participant_count}_small_payload")
}

fn build_workload_specs(
    payload_bytes: usize,
    formats: &[String],
    workloads: &[String],
    max_group_size: usize,
) -> Vec<WorkloadSpec> {
    let activation_bytes = activation_bytes_for_payload(payload_bytes);
    let output_bytes = output_bytes_for_payload(payload_bytes);
    let mut specs = Vec::new();
    for format in formats {
        for workload in workloads {
            if !benchmark_axis_supported(format, workload) {
                continue;
            }
            specs.push(WorkloadSpec {
                workload_id: format_workload_id("single_target_small_payload", workload, format),
                comparison_group: SMALL_PAYLOAD_COMPARISON_GROUP.to_string(),
                workload_class: workload.clone(),
                placement_strategy: "single_target_serial".to_string(),
                pattern: "single_target_compute".to_string(),
                format: format.clone(),
                participant_count: 1,
                payload_bytes,
                parameter_bytes_per_participant: payload_bytes,
                activation_bytes,
                output_bytes,
                description: format!("Run the comparison projection on one target using {format}."),
            });
            if benchmark_supports_component_chain(workload) {
                specs.extend([
                    WorkloadSpec {
                        workload_id: format!("single_target_2_component_chain:{workload}:{format}"),
                        comparison_group: SMALL_PAYLOAD_COMPARISON_GROUP.to_string(),
                        workload_class: workload.clone(),
                        placement_strategy: "single_target_serial".to_string(),
                        pattern: "direct_two_component_chain".to_string(),
                        format: format.clone(),
                        participant_count: 1,
                        payload_bytes,
                        parameter_bytes_per_participant: payload_bytes,
                        activation_bytes,
                        output_bytes,
                        description: format!("Run the two-stage serial baseline on one target using {format}."),
                    },
                    WorkloadSpec {
                        workload_id: format_workload_id("synthetic_layer_split_small_payload", workload, format),
                        comparison_group: SMALL_PAYLOAD_COMPARISON_GROUP.to_string(),
                        workload_class: workload.clone(),
                        placement_strategy: "two_target_serial".to_string(),
                        pattern: "synthetic_layer_split_small_payload".to_string(),
                        format: format.clone(),
                        participant_count: 2,
                        payload_bytes,
                        parameter_bytes_per_participant: payload_bytes / 2,
                        activation_bytes,
                        output_bytes,
                        description: format!("Run the equivalent two-stage serial path across two targets using {format}."),
                    },
                ]);
            }
            if benchmark_supports_tensor_parallel(workload) {
                specs.extend([
                    WorkloadSpec {
                        workload_id: format_workload_id("synthetic_tensor_parallel_small_payload", workload, format),
                        comparison_group: SMALL_PAYLOAD_COMPARISON_GROUP.to_string(),
                        workload_class: workload.clone(),
                        placement_strategy: "two_target_tensor_parallel".to_string(),
                        pattern: "synthetic_tensor_parallel_small_payload".to_string(),
                        format: format.clone(),
                        participant_count: 2,
                        payload_bytes,
                        parameter_bytes_per_participant: payload_bytes / 2,
                        activation_bytes,
                        output_bytes,
                        description: format!("Run the comparison projection tensor-parallel across two targets using {format}."),
                    },
                    WorkloadSpec {
                        workload_id: format_workload_id("synthetic_tensor_parallel_forced_split_2", workload, format),
                        comparison_group: "forced_split_tp_vs_serialized".to_string(),
                        workload_class: workload.clone(),
                        placement_strategy: "two_target_tensor_parallel".to_string(),
                        pattern: "synthetic_tensor_parallel_forced_split_2".to_string(),
                        format: format.clone(),
                        participant_count: 2,
                        payload_bytes,
                        parameter_bytes_per_participant: payload_bytes / 2,
                        activation_bytes,
                        output_bytes,
                        description: format!("Compare an equivalent two-stage TP path against serialization using {format}."),
                    },
                ]);
            }
            if max_group_size >= 3 && benchmark_supports_tensor_parallel(workload) {
                for participant_count in 3..=max_group_size {
                    let pattern = tensor_parallel_group_pattern(participant_count);
                    specs.extend([
                        WorkloadSpec {
                            workload_id: format_workload_id(&pattern, workload, format),
                            pattern,
                            comparison_group: SMALL_PAYLOAD_COMPARISON_GROUP.to_string(),
                            workload_class: workload.clone(),
                            placement_strategy: MULTI_TARGET_TENSOR_PARALLEL_STRATEGY.to_string(),
                            format: format.clone(),
                            participant_count,
                            payload_bytes,
                            parameter_bytes_per_participant: payload_bytes / participant_count,
                            activation_bytes,
                            output_bytes,
                            description: format!("Run the comparison projection tensor-parallel across {participant_count} targets using {format}."),
                        },
                        WorkloadSpec {
                            workload_id: format_workload_id(&format!("synthetic_serialized_forced_split_{participant_count}"), workload, format),
                            pattern: format!("synthetic_serialized_forced_split_{participant_count}"),
                            comparison_group: "forced_split_tp_vs_serialized".to_string(),
                            workload_class: workload.clone(),
                            placement_strategy: "multi_target_serial".to_string(),
                            format: format.clone(),
                            participant_count,
                            payload_bytes,
                            parameter_bytes_per_participant: payload_bytes / participant_count,
                            activation_bytes,
                            output_bytes,
                            description: format!("Run the selected serialized {participant_count}-target forced split using {format}."),
                        },
                        WorkloadSpec {
                            workload_id: format_workload_id(&format!("synthetic_tensor_parallel_forced_split_{participant_count}"), workload, format),
                            pattern: format!("synthetic_tensor_parallel_forced_split_{participant_count}"),
                            comparison_group: "forced_split_tp_vs_serialized".to_string(),
                            workload_class: workload.clone(),
                            placement_strategy: MULTI_TARGET_TENSOR_PARALLEL_STRATEGY.to_string(),
                            format: format.clone(),
                            participant_count,
                            payload_bytes,
                            parameter_bytes_per_participant: payload_bytes / participant_count,
                            activation_bytes,
                            output_bytes,
                            description: format!("Run the equivalent {participant_count}-stage TP forced split using {format}."),
                        },
                    ]);
                }
            }
        }
    }
    specs
}

pub(crate) fn format_workload_id(base: &str, workload_class: &str, format: &str) -> String {
    format!("{base}:{workload_class}:{format}")
}

pub(crate) fn activation_bytes_for_payload(payload_bytes: usize) -> usize {
    payload_bytes.min(256 * 1024).max(4)
}

pub(crate) fn output_bytes_for_payload(payload_bytes: usize) -> usize {
    (payload_bytes / 16).clamp(4, 512 * 1024)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::VulkanDeviceInfo;

    fn target(id: &str) -> Target {
        Target {
            stable_target_id: id.to_string(),
            backend: "vulkan".to_string(),
            kind: "discrete_gpu".to_string(),
            name: id.to_string(),
            vendor_id: None,
            vendor_name: None,
            device_id: None,
            pci_address: None,
            physical_location: None,
            numa_node: None,
            boot_vga: None,
            pci_link: None,
            vulkan: Some(VulkanDeviceInfo {
                physical_device_index: 0,
                device_name: id.to_string(),
                device_type: "discrete_gpu".to_string(),
                api_version: "1.3".to_string(),
                driver_version: 1,
                vendor_id: "0x1002".to_string(),
                device_id: "0x0001".to_string(),
                memory_heaps: Vec::new(),
                queue_families: Vec::new(),
                extension_names: Vec::new(),
                feature_flags: vec![
                    EXTERNAL_TIMELINE_SEMAPHORE_FEATURE.to_string(),
                    EXTERNAL_MEMORY_HOST_FEATURE.to_string(),
                ],
            }),
            capabilities: Vec::new(),
            format_capabilities: Vec::new(),
            diagnostics: Vec::new(),
        }
    }

    fn policy() -> RunPolicy {
        RunPolicy {
            payload_bytes: 5 * 1024 * 1024,
            samples: 1,
            benchmark_formats: vec!["f16".to_string(), "fp8_e4m3".to_string()],
            benchmark_workloads: vec!["dense_projection_decode".to_string()],
            include_targets: Vec::new(),
            exclude_targets: Vec::new(),
            exclude_pci: Vec::new(),
            exclude_kinds: Vec::new(),
            pair_measurements: true,
            max_group_size: 3,
            execute: false,
        }
    }

    #[test]
    fn compact_plan_counts_only_comparison_candidates() {
        let targets = vec![target("gpu:a"), target("gpu:b"), target("gpu:c")];
        let selection = Selection {
            selected_target_ids: targets
                .iter()
                .map(|target| target.stable_target_id.clone())
                .collect(),
            skipped_targets: Vec::new(),
            diagnostics: Vec::new(),
        };
        let plan = plan_benchmarks(targets, selection, policy());
        assert_eq!(plan.estimated_single_measurement_count, 12);
        assert_eq!(plan.estimated_pair_measurement_count, 36);
        assert_eq!(plan.estimated_group_measurement_count, 14);
        assert_eq!(plan.estimated_comparison_set_count, 0);
        assert_eq!(plan.estimated_measurement_count, 62);
    }

    #[test]
    fn workload_specs_have_no_exhaustive_phase_or_chain_axes() {
        let policy = policy();
        let specs = build_workload_specs(
            policy.payload_bytes,
            &policy.benchmark_formats,
            &policy.benchmark_workloads,
            policy.max_group_size,
        );
        assert_eq!(specs.len(), policy.benchmark_formats.len() * 8);
        assert_eq!(
            specs
                .iter()
                .map(|spec| spec.workload_id.as_str())
                .collect::<BTreeSet<_>>()
                .len(),
            specs.len()
        );
        assert!(specs.iter().all(|spec| {
            !spec.pattern.contains("component_chain_3")
                && !spec.pattern.contains("layer_split_group")
                && !spec.pattern.contains("resource_load")
        }));
    }

    #[test]
    fn combinations_cover_each_tp_group_once() {
        assert_eq!(target_index_combinations(5, 1).len(), 5);
        assert_eq!(target_index_combinations(5, 2).len(), 10);
        assert_eq!(target_index_combinations(5, 3).len(), 10);
        assert_eq!(target_index_combinations(5, 4).len(), 5);
    }

    #[test]
    fn staged_expansion_has_no_fixed_device_ceiling() {
        let stages = staged_target_index_groups(7, 7);
        assert_eq!(stages[&2].len(), 21);
        assert_eq!(stages[&3].len(), 35);
        assert_eq!(stages[&4].len(), 35);
        assert_eq!(stages[&5].len(), 21);
        assert_eq!(stages[&6].len(), 7);
        assert_eq!(stages[&7], [vec![0, 1, 2, 3, 4, 5, 6]]);
    }

    #[test]
    fn staged_tp_expansion_only_grows_transport_compatible_predecessors() {
        let mut targets = vec![
            target("gpu:a"),
            target("gpu:b"),
            target("gpu:c"),
            target("gpu:incompatible"),
        ];
        targets[3].vulkan.as_mut().unwrap().feature_flags.clear();
        let borrowed = targets.iter().collect::<Vec<_>>();

        let stages = staged_tensor_parallel_target_groups(&borrowed, 4);
        assert_eq!(stages[&2], [vec![0, 1], vec![0, 2], vec![1, 2]]);
        assert_eq!(stages[&3], [vec![0, 1, 2]]);
        assert!(!stages.contains_key(&4));
    }
}
