use std::collections::BTreeSet;
use std::hint::black_box;
use std::time::Instant;

use crate::model::{
    BenchmarkRun, ComparisonCandidate, ComparisonSet, GroupMeasurement, Implementation,
    Measurement, PairMeasurement, RUN_SCHEMA, RunPolicy, Sample, Selection, Summary, Target,
    WorkloadSpec, now_unix_ms,
};

const SMALL_PAYLOAD_COMPARISON_GROUP: &str = "small_payload_placement_comparison";

pub fn run_benchmarks(
    discovered_targets: Vec<Target>,
    selection: Selection,
    policy: RunPolicy,
) -> BenchmarkRun {
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
    for target in &selected_targets {
        if target.kind == "cpu" {
            measurements.extend(run_cpu_measurements(
                &target.stable_target_id,
                policy.payload_bytes,
                policy.samples,
                &policy.benchmark_formats,
                &policy.benchmark_workloads,
            ));
        } else {
            measurements.extend(unmeasured_single_target(
                &target.stable_target_id,
                policy.payload_bytes,
                &policy.benchmark_formats,
                &policy.benchmark_workloads,
                "gpu_backend_not_implemented",
            ));
        }
    }

    let pair_measurements = if policy.pair_measurements && policy.max_group_size >= 2 {
        build_pair_placeholders(
            &selected_targets,
            policy.payload_bytes,
            &policy.benchmark_formats,
            &policy.benchmark_workloads,
        )
    } else {
        Vec::new()
    };
    let group_measurements = if policy.pair_measurements && policy.max_group_size >= 3 {
        build_group_placeholders(
            &selected_targets,
            policy.payload_bytes,
            &policy.benchmark_formats,
            &policy.benchmark_workloads,
            policy.max_group_size,
        )
    } else {
        Vec::new()
    };
    let workload_specs = build_workload_specs(
        policy.payload_bytes,
        &policy.benchmark_formats,
        &policy.benchmark_workloads,
        policy.max_group_size,
    );
    let comparison_sets = build_comparison_sets(
        &selected_targets,
        policy.pair_measurements,
        &policy.benchmark_formats,
        &policy.benchmark_workloads,
        policy.max_group_size,
    );

    let mut diagnostics = selection.diagnostics.clone();
    diagnostics.push(
        "GPU discovery is passive; GPU benchmark workloads are unmeasured until the Vulkan backend is implemented."
            .to_string(),
    );
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

fn run_cpu_measurements(
    target_id: &str,
    payload_bytes: usize,
    samples: usize,
    formats: &[String],
    workloads: &[String],
) -> Vec<Measurement> {
    let mut measurements = vec![
        run_cpu_copy(target_id, payload_bytes, samples),
        run_cpu_f32_dot(target_id, payload_bytes, samples),
    ];
    measurements.extend(run_cpu_requested_single_target_measurements(
        target_id,
        payload_bytes,
        samples,
        formats,
        workloads,
    ));
    measurements.extend(run_cpu_compound_reference(
        target_id,
        payload_bytes,
        samples,
    ));
    measurements
}

fn run_cpu_requested_single_target_measurements(
    target_id: &str,
    payload_bytes: usize,
    samples: usize,
    formats: &[String],
    workloads: &[String],
) -> Vec<Measurement> {
    let mut measurements = Vec::new();
    for format in formats {
        for workload in workloads {
            if format == "f32" {
                measurements.push(run_cpu_f32_requested_workload(
                    target_id,
                    payload_bytes,
                    samples,
                    workload,
                ));
            } else {
                measurements.push(unsupported_cpu_requested_workload(
                    target_id,
                    payload_bytes,
                    workload,
                    format,
                ));
            }
        }
    }
    measurements
}

fn run_cpu_f32_requested_workload(
    target_id: &str,
    payload_bytes: usize,
    samples: usize,
    workload_class: &str,
) -> Measurement {
    let elements = (payload_bytes / (2 * std::mem::size_of::<f32>())).max(1);
    let left = (0..elements)
        .map(|index| ((index % 1024) as f32) * 0.001)
        .collect::<Vec<_>>();
    let right = (0..elements)
        .map(|index| (((index * 17) % 1024) as f32) * 0.001)
        .collect::<Vec<_>>();

    black_box(cpu_f32_workload(workload_class, &left, &right));

    let mut measured_samples = Vec::with_capacity(samples);
    for sample_index in 0..samples {
        let started = Instant::now();
        let value = cpu_f32_workload(workload_class, &left, &right);
        black_box(value);
        let duration = started.elapsed();
        measured_samples.push(Sample {
            sample_index,
            duration_ns: duration.as_nanos(),
            iterations: 1,
            bytes_read: (elements * 2 * std::mem::size_of::<f32>()) as u64,
            bytes_written: std::mem::size_of::<f32>() as u64,
            operations: cpu_f32_workload_operations(workload_class, elements),
        });
    }

    Measurement {
        workload_id: format_workload_id("single_target_small_payload", workload_class, "f32"),
        comparison_group: SMALL_PAYLOAD_COMPARISON_GROUP.to_string(),
        workload_class: workload_class.to_string(),
        placement_strategy: "single_target_serial".to_string(),
        target_id: target_id.to_string(),
        pattern: "single_target_compute".to_string(),
        operation_family: workload_class.to_string(),
        regime: "small_payload".to_string(),
        format: "f32".to_string(),
        status: "completed".to_string(),
        reason: None,
        payload_bytes,
        working_set_bytes: elements * 2 * std::mem::size_of::<f32>(),
        summary: summarize(&measured_samples),
        samples: measured_samples,
    }
}

fn unsupported_cpu_requested_workload(
    target_id: &str,
    payload_bytes: usize,
    workload_class: &str,
    format: &str,
) -> Measurement {
    Measurement {
        workload_id: format_workload_id("single_target_small_payload", workload_class, format),
        comparison_group: SMALL_PAYLOAD_COMPARISON_GROUP.to_string(),
        workload_class: workload_class.to_string(),
        placement_strategy: "single_target_serial".to_string(),
        target_id: target_id.to_string(),
        pattern: "single_target_compute".to_string(),
        operation_family: workload_class.to_string(),
        regime: "small_payload".to_string(),
        format: format.to_string(),
        status: "unsupported".to_string(),
        reason: Some("cpu_format_backend_not_implemented".to_string()),
        payload_bytes,
        working_set_bytes: payload_bytes,
        samples: Vec::new(),
        summary: None,
    }
}

fn cpu_f32_workload(workload_class: &str, left: &[f32], right: &[f32]) -> f32 {
    match workload_class {
        "dense_projection" => left
            .iter()
            .zip(right.iter())
            .fold(0.0_f32, |sum, (left, right)| {
                (sum + left.mul_add(*right, 0.125)).mul_add(0.999_999, 0.000_001)
            }),
        "moe_expert" => left
            .iter()
            .zip(right.iter())
            .fold(0.0_f32, |sum, (left, right)| {
                let gate = if *left > *right { *left } else { *right };
                sum + gate * (left + right) * 0.5
            }),
        "router_reduction" => dot_product(left, right),
        _ => dot_product(left, right),
    }
}

fn cpu_f32_workload_operations(workload_class: &str, elements: usize) -> u64 {
    let operations_per_element = match workload_class {
        "dense_projection" => 5,
        "moe_expert" => 6,
        "router_reduction" => 2,
        _ => 2,
    };
    elements as u64 * operations_per_element
}

fn run_cpu_copy(target_id: &str, payload_bytes: usize, samples: usize) -> Measurement {
    let mut source = vec![0_u8; payload_bytes];
    let mut destination = vec![0_u8; payload_bytes];
    for (index, value) in source.iter_mut().enumerate() {
        *value = (index & 0xff) as u8;
    }

    destination.copy_from_slice(&source);
    black_box(&destination);

    let mut measured_samples = Vec::with_capacity(samples);
    for sample_index in 0..samples {
        let started = Instant::now();
        destination.copy_from_slice(&source);
        black_box(&destination);
        let duration = started.elapsed();
        measured_samples.push(Sample {
            sample_index,
            duration_ns: duration.as_nanos(),
            iterations: 1,
            bytes_read: payload_bytes as u64,
            bytes_written: payload_bytes as u64,
            operations: 0,
        });
    }

    Measurement {
        workload_id: "single_cpu_u8_copy".to_string(),
        comparison_group: "single_target_primitives".to_string(),
        workload_class: "memory_copy".to_string(),
        placement_strategy: "single_target_serial".to_string(),
        target_id: target_id.to_string(),
        pattern: "single_target_copy".to_string(),
        operation_family: "memory_copy".to_string(),
        regime: "small_payload".to_string(),
        format: "u8".to_string(),
        status: "completed".to_string(),
        reason: None,
        payload_bytes,
        working_set_bytes: payload_bytes * 2,
        summary: summarize(&measured_samples),
        samples: measured_samples,
    }
}

fn run_cpu_f32_dot(target_id: &str, payload_bytes: usize, samples: usize) -> Measurement {
    let elements = (payload_bytes / (2 * std::mem::size_of::<f32>())).max(1);
    let left = (0..elements)
        .map(|index| ((index % 1024) as f32) * 0.001)
        .collect::<Vec<_>>();
    let right = (0..elements)
        .map(|index| (((index * 17) % 1024) as f32) * 0.001)
        .collect::<Vec<_>>();

    black_box(dot_product(&left, &right));

    let mut measured_samples = Vec::with_capacity(samples);
    for sample_index in 0..samples {
        let started = Instant::now();
        let value = dot_product(&left, &right);
        black_box(value);
        let duration = started.elapsed();
        measured_samples.push(Sample {
            sample_index,
            duration_ns: duration.as_nanos(),
            iterations: 1,
            bytes_read: (elements * 2 * std::mem::size_of::<f32>()) as u64,
            bytes_written: std::mem::size_of::<f32>() as u64,
            operations: (elements as u64) * 2,
        });
    }

    Measurement {
        workload_id: "single_cpu_f32_dot".to_string(),
        comparison_group: "single_target_primitives".to_string(),
        workload_class: "router_reduction".to_string(),
        placement_strategy: "single_target_serial".to_string(),
        target_id: target_id.to_string(),
        pattern: "single_target_reduction".to_string(),
        operation_family: "dot_product_reduction".to_string(),
        regime: "small_payload".to_string(),
        format: "f32".to_string(),
        status: "completed".to_string(),
        reason: None,
        payload_bytes,
        working_set_bytes: elements * 2 * std::mem::size_of::<f32>(),
        summary: summarize(&measured_samples),
        samples: measured_samples,
    }
}

fn dot_product(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .fold(0.0_f32, |sum, (left, right)| sum + left * right)
}

fn run_cpu_compound_reference(
    target_id: &str,
    payload_bytes: usize,
    samples: usize,
) -> Vec<Measurement> {
    vec![
        run_cpu_compound_pattern(
            target_id,
            payload_bytes,
            samples,
            CpuCompoundPattern::Serialized,
        ),
        run_cpu_compound_pattern(
            target_id,
            payload_bytes,
            samples,
            CpuCompoundPattern::LayerSplit,
        ),
        run_cpu_compound_pattern(
            target_id,
            payload_bytes,
            samples,
            CpuCompoundPattern::TensorSplit,
        ),
    ]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CpuCompoundPattern {
    Serialized,
    LayerSplit,
    TensorSplit,
}

impl CpuCompoundPattern {
    fn workload_id(self) -> &'static str {
        match self {
            Self::Serialized => "cpu_reference_serialized_small_payload",
            Self::LayerSplit => "cpu_reference_layer_split_small_payload",
            Self::TensorSplit => "cpu_reference_tensor_split_small_payload",
        }
    }

    fn pattern(self) -> &'static str {
        match self {
            Self::Serialized => "serialized_small_payload",
            Self::LayerSplit => "synthetic_layer_split_small_payload",
            Self::TensorSplit => "synthetic_tensor_split_small_payload",
        }
    }

    fn placement_strategy(self) -> &'static str {
        match self {
            Self::Serialized => "single_target_serial",
            Self::LayerSplit => "two_stage_serial_reference",
            Self::TensorSplit => "two_shard_parallel_reference",
        }
    }
}

fn run_cpu_compound_pattern(
    target_id: &str,
    payload_bytes: usize,
    samples: usize,
    pattern: CpuCompoundPattern,
) -> Measurement {
    let source = patterned_payload(payload_bytes);
    let activation_bytes = activation_bytes_for_payload(payload_bytes);
    let output_bytes = output_bytes_for_payload(payload_bytes);
    let mut scratch = vec![0_u8; payload_bytes];
    let mut activation = vec![0_u8; activation_bytes];
    let mut output = vec![0_u8; output_bytes];

    execute_cpu_compound_pattern(pattern, &source, &mut scratch, &mut activation, &mut output);
    black_box(checksum(&output));

    let mut measured_samples = Vec::with_capacity(samples);
    for sample_index in 0..samples {
        let started = Instant::now();
        execute_cpu_compound_pattern(pattern, &source, &mut scratch, &mut activation, &mut output);
        black_box(checksum(&output));
        let duration = started.elapsed();
        measured_samples.push(Sample {
            sample_index,
            duration_ns: duration.as_nanos(),
            iterations: 1,
            bytes_read: compound_bytes_read(payload_bytes, activation_bytes, pattern),
            bytes_written: compound_bytes_written(payload_bytes, activation_bytes, output_bytes),
            operations: compound_operations(payload_bytes, output_bytes, pattern),
        });
    }

    Measurement {
        workload_id: pattern.workload_id().to_string(),
        comparison_group: SMALL_PAYLOAD_COMPARISON_GROUP.to_string(),
        workload_class: "cpu_reference_compound".to_string(),
        placement_strategy: pattern.placement_strategy().to_string(),
        target_id: target_id.to_string(),
        pattern: pattern.pattern().to_string(),
        operation_family: "cpu_reference_compound".to_string(),
        regime: "small_payload".to_string(),
        format: "u8_synthetic".to_string(),
        status: "completed".to_string(),
        reason: None,
        payload_bytes,
        working_set_bytes: payload_bytes + scratch.len() + activation.len() + output.len(),
        summary: summarize(&measured_samples),
        samples: measured_samples,
    }
}

fn execute_cpu_compound_pattern(
    pattern: CpuCompoundPattern,
    source: &[u8],
    scratch: &mut [u8],
    activation: &mut [u8],
    output: &mut [u8],
) {
    match pattern {
        CpuCompoundPattern::Serialized => {
            transform_bytes(source, scratch, 17);
            fill_activation_from_payload(scratch, activation);
            fill_output_from_payload(scratch, activation, output);
        }
        CpuCompoundPattern::LayerSplit => {
            let split = source.len() / 2;
            transform_bytes(&source[..split], &mut scratch[..split], 29);
            fill_activation_from_payload(&scratch[..split], activation);
            transform_bytes(&source[split..], &mut scratch[split..], 43);
            fill_output_from_payload(&scratch[split..], activation, output);
        }
        CpuCompoundPattern::TensorSplit => {
            let split = source.len() / 2;
            transform_bytes(&source[..split], &mut scratch[..split], 61);
            transform_bytes(&source[split..], &mut scratch[split..], 79);
            fill_activation_from_payload(scratch, activation);
            fill_output_from_tensor_shards(
                &scratch[..split],
                &scratch[split..],
                activation,
                output,
            );
        }
    }
}

fn patterned_payload(payload_bytes: usize) -> Vec<u8> {
    (0..payload_bytes)
        .map(|index| ((index.wrapping_mul(37).wrapping_add(11)) & 0xff) as u8)
        .collect()
}

fn transform_bytes(source: &[u8], destination: &mut [u8], salt: u8) {
    debug_assert_eq!(source.len(), destination.len());
    for (index, (source, destination)) in source.iter().zip(destination.iter_mut()).enumerate() {
        let index_byte = (index & 0xff) as u8;
        *destination = source
            .wrapping_mul(3)
            .wrapping_add(salt)
            .rotate_left((index_byte & 7) as u32);
    }
}

fn fill_activation_from_payload(payload: &[u8], activation: &mut [u8]) {
    if payload.is_empty() {
        activation.fill(0);
        return;
    }
    for (index, activation_byte) in activation.iter_mut().enumerate() {
        let first = payload[index % payload.len()];
        let second = payload[(index.wrapping_mul(17).wrapping_add(5)) % payload.len()];
        *activation_byte = first ^ second.rotate_left((index & 7) as u32);
    }
}

fn fill_output_from_payload(payload: &[u8], activation: &[u8], output: &mut [u8]) {
    if payload.is_empty() || activation.is_empty() {
        output.fill(0);
        return;
    }
    for (index, output_byte) in output.iter_mut().enumerate() {
        let value = payload[(index.wrapping_mul(13)) % payload.len()]
            .wrapping_add(activation[index % activation.len()]);
        *output_byte = value.rotate_right((index & 7) as u32);
    }
}

fn fill_output_from_tensor_shards(left: &[u8], right: &[u8], activation: &[u8], output: &mut [u8]) {
    if left.is_empty() || right.is_empty() || activation.is_empty() {
        output.fill(0);
        return;
    }
    for (index, output_byte) in output.iter_mut().enumerate() {
        let left_value = left[(index.wrapping_mul(7)) % left.len()];
        let right_value = right[(index.wrapping_mul(11)) % right.len()];
        *output_byte = left_value
            .wrapping_add(right_value)
            .wrapping_add(activation[index % activation.len()])
            .rotate_left((index & 7) as u32);
    }
}

fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0_u64, |sum, byte| {
        sum.rotate_left(5).wrapping_add(u64::from(*byte))
    })
}

fn compound_bytes_read(
    payload_bytes: usize,
    activation_bytes: usize,
    pattern: CpuCompoundPattern,
) -> u64 {
    let output_bytes = output_bytes_for_payload(payload_bytes);
    let payload_reads = match pattern {
        CpuCompoundPattern::Serialized => payload_bytes * 3,
        CpuCompoundPattern::LayerSplit => payload_bytes * 2,
        CpuCompoundPattern::TensorSplit => payload_bytes * 3,
    };
    (payload_reads + activation_bytes + output_bytes) as u64
}

fn compound_bytes_written(
    payload_bytes: usize,
    activation_bytes: usize,
    output_bytes: usize,
) -> u64 {
    (payload_bytes + activation_bytes + output_bytes) as u64
}

fn compound_operations(
    payload_bytes: usize,
    output_bytes: usize,
    pattern: CpuCompoundPattern,
) -> u64 {
    let transform_operations = payload_bytes as u64 * 4;
    let output_operations = output_bytes as u64
        * match pattern {
            CpuCompoundPattern::Serialized | CpuCompoundPattern::LayerSplit => 3,
            CpuCompoundPattern::TensorSplit => 5,
        };
    transform_operations + output_operations
}

fn unmeasured_single_target(
    target_id: &str,
    payload_bytes: usize,
    formats: &[String],
    workloads: &[String],
    reason: &str,
) -> Vec<Measurement> {
    formats
        .iter()
        .flat_map(|format| {
            workloads.iter().map(|workload| Measurement {
                workload_id: format_workload_id("single_target_small_payload", workload, format),
                comparison_group: SMALL_PAYLOAD_COMPARISON_GROUP.to_string(),
                workload_class: workload.clone(),
                placement_strategy: "single_target_serial".to_string(),
                target_id: target_id.to_string(),
                pattern: "single_target_compute".to_string(),
                operation_family: workload.clone(),
                regime: "small_payload".to_string(),
                format: format.clone(),
                status: "unmeasured".to_string(),
                reason: Some(reason.to_string()),
                payload_bytes,
                working_set_bytes: payload_bytes,
                samples: Vec::new(),
                summary: None,
            })
        })
        .collect()
}

fn build_pair_placeholders(
    targets: &[&Target],
    payload_bytes: usize,
    formats: &[String],
    workloads: &[String],
) -> Vec<PairMeasurement> {
    let mut measurements = Vec::new();
    for format in formats {
        for workload in workloads {
            for source in targets {
                for destination in targets {
                    if source.stable_target_id == destination.stable_target_id {
                        continue;
                    }
                    measurements.push(unmeasured_pair(
                        &source.stable_target_id,
                        &destination.stable_target_id,
                        "ordered_activation_transfer",
                        "activation_transfer_only",
                        "activation_transfer",
                        workload,
                        format,
                        "requires_device_backend",
                        payload_bytes,
                    ));
                }
            }
            for left_index in 0..targets.len() {
                for right_index in (left_index + 1)..targets.len() {
                    let left = &targets[left_index].stable_target_id;
                    let right = &targets[right_index].stable_target_id;
                    measurements.push(unmeasured_pair(
                        left,
                        right,
                        "synthetic_layer_split_small_payload",
                        "two_target_serial",
                        "layer_split",
                        workload,
                        format,
                        "requires_device_backend",
                        payload_bytes,
                    ));
                    measurements.push(unmeasured_pair(
                        right,
                        left,
                        "synthetic_layer_split_small_payload",
                        "two_target_serial",
                        "layer_split",
                        workload,
                        format,
                        "requires_device_backend",
                        payload_bytes,
                    ));
                    measurements.push(unmeasured_pair(
                        left,
                        right,
                        "synthetic_tensor_split_small_payload",
                        "two_target_parallel",
                        "tensor_split",
                        workload,
                        format,
                        "requires_device_backend",
                        payload_bytes,
                    ));
                }
            }
        }
    }
    measurements
}

fn build_comparison_sets(
    targets: &[&Target],
    pair_measurements: bool,
    formats: &[String],
    workloads: &[String],
    max_group_size: usize,
) -> Vec<ComparisonSet> {
    if !pair_measurements || max_group_size < 2 {
        return Vec::new();
    }
    let mut comparisons = Vec::new();
    for format in formats {
        for workload in workloads {
            for left_index in 0..targets.len() {
                for right_index in (left_index + 1)..targets.len() {
                    let left = &targets[left_index].stable_target_id;
                    let right = &targets[right_index].stable_target_id;
                    let comparison_id = format!(
                        "{SMALL_PAYLOAD_COMPARISON_GROUP}:{workload}:{format}:{left}|{right}"
                    );
                    comparisons.push(ComparisonSet {
                        comparison_id: comparison_id.clone(),
                        comparison_group: SMALL_PAYLOAD_COMPARISON_GROUP.to_string(),
                        workload_class: workload.clone(),
                        regime: "small_payload".to_string(),
                        format: format.clone(),
                        target_ids: vec![left.clone(), right.clone()],
                        candidates: vec![
                            comparison_candidate(
                                &comparison_id,
                                "single_left",
                                "single_target_serial",
                                "single",
                                &format_workload_id(
                                    "single_target_small_payload",
                                    workload,
                                    format,
                                ),
                                vec![left.clone()],
                                "Run the whole payload on the first target only.",
                            ),
                            comparison_candidate(
                                &comparison_id,
                                "single_right",
                                "single_target_serial",
                                "single",
                                &format_workload_id(
                                    "single_target_small_payload",
                                    workload,
                                    format,
                                ),
                                vec![right.clone()],
                                "Run the whole payload on the second target only.",
                            ),
                            comparison_candidate(
                                &comparison_id,
                                "serial_left_to_right",
                                "two_target_serial",
                                "pair",
                                &format_workload_id(
                                    "synthetic_layer_split_small_payload",
                                    workload,
                                    format,
                                ),
                                vec![left.clone(), right.clone()],
                                "Run the first stage on the first target, then transfer activation to the second target.",
                            ),
                            comparison_candidate(
                                &comparison_id,
                                "serial_right_to_left",
                                "two_target_serial",
                                "pair",
                                &format_workload_id(
                                    "synthetic_layer_split_small_payload",
                                    workload,
                                    format,
                                ),
                                vec![right.clone(), left.clone()],
                                "Run the first stage on the second target, then transfer activation to the first target.",
                            ),
                            comparison_candidate(
                                &comparison_id,
                                "parallel_pair",
                                "two_target_parallel",
                                "pair",
                                &format_workload_id(
                                    "synthetic_tensor_split_small_payload",
                                    workload,
                                    format,
                                ),
                                vec![left.clone(), right.clone()],
                                "Split the same logical payload across both targets in parallel.",
                            ),
                        ],
                    });
                }
            }
        }
    }
    comparisons
}

fn comparison_candidate(
    comparison_id: &str,
    candidate_suffix: &str,
    placement_strategy: &str,
    measurement_kind: &str,
    workload_id: &str,
    target_ids: Vec<String>,
    notes: &str,
) -> ComparisonCandidate {
    ComparisonCandidate {
        candidate_id: format!("{comparison_id}:{candidate_suffix}"),
        placement_strategy: placement_strategy.to_string(),
        measurement_kind: measurement_kind.to_string(),
        workload_id: workload_id.to_string(),
        target_ids,
        notes: notes.to_string(),
    }
}

fn build_group_placeholders(
    targets: &[&Target],
    payload_bytes: usize,
    formats: &[String],
    workloads: &[String],
    max_group_size: usize,
) -> Vec<GroupMeasurement> {
    let mut measurements = Vec::new();
    let max_group_size = max_group_size.min(3);
    if max_group_size < 3 || targets.len() < 3 {
        return measurements;
    }
    for format in formats {
        for workload in workloads {
            for first in 0..targets.len() {
                for second in (first + 1)..targets.len() {
                    for third in (second + 1)..targets.len() {
                        let target_ids = vec![
                            targets[first].stable_target_id.clone(),
                            targets[second].stable_target_id.clone(),
                            targets[third].stable_target_id.clone(),
                        ];
                        measurements.push(unmeasured_group(
                            target_ids.clone(),
                            "synthetic_layer_split_group_small_payload",
                            "three_target_serial",
                            "layer_split",
                            workload,
                            format,
                            "requires_device_backend",
                            payload_bytes,
                        ));
                        measurements.push(unmeasured_group(
                            target_ids,
                            "synthetic_tensor_split_group_small_payload",
                            "three_target_parallel",
                            "tensor_split",
                            workload,
                            format,
                            "requires_device_backend",
                            payload_bytes,
                        ));
                    }
                }
            }
        }
    }
    measurements
}

fn unmeasured_pair(
    source: &str,
    destination: &str,
    pattern: &str,
    placement_strategy: &str,
    operation_family: &str,
    workload_class: &str,
    format: &str,
    reason: &str,
    payload_bytes: usize,
) -> PairMeasurement {
    let half_payload = payload_bytes / 2;
    let activation_bytes = activation_bytes_for_payload(payload_bytes);
    let output_bytes = output_bytes_for_payload(payload_bytes);
    PairMeasurement {
        workload_id: format_workload_id(pattern, workload_class, format),
        comparison_group: SMALL_PAYLOAD_COMPARISON_GROUP.to_string(),
        workload_class: workload_class.to_string(),
        placement_strategy: placement_strategy.to_string(),
        source_target_id: source.to_string(),
        destination_target_id: destination.to_string(),
        pattern: pattern.to_string(),
        operation_family: operation_family.to_string(),
        regime: "small_payload".to_string(),
        format: format.to_string(),
        status: "unmeasured".to_string(),
        reason: Some(reason.to_string()),
        payload_bytes,
        source_payload_bytes: half_payload,
        destination_payload_bytes: payload_bytes - half_payload,
        activation_bytes,
        output_bytes,
        samples: Vec::new(),
        summary: None,
    }
}

fn unmeasured_group(
    target_ids: Vec<String>,
    pattern: &str,
    placement_strategy: &str,
    operation_family: &str,
    workload_class: &str,
    format: &str,
    reason: &str,
    payload_bytes: usize,
) -> GroupMeasurement {
    let payload_bytes_per_participant = split_bytes(payload_bytes, target_ids.len());
    GroupMeasurement {
        workload_id: format_workload_id(pattern, workload_class, format),
        comparison_group: SMALL_PAYLOAD_COMPARISON_GROUP.to_string(),
        workload_class: workload_class.to_string(),
        placement_strategy: placement_strategy.to_string(),
        target_ids,
        pattern: pattern.to_string(),
        operation_family: operation_family.to_string(),
        regime: "small_payload".to_string(),
        format: format.to_string(),
        status: "unmeasured".to_string(),
        reason: Some(reason.to_string()),
        participant_count: payload_bytes_per_participant.len(),
        payload_bytes,
        payload_bytes_per_participant,
        activation_bytes: activation_bytes_for_payload(payload_bytes),
        output_bytes: output_bytes_for_payload(payload_bytes),
        samples: Vec::new(),
        summary: None,
    }
}

fn build_workload_specs(
    payload_bytes: usize,
    formats: &[String],
    workloads: &[String],
    max_group_size: usize,
) -> Vec<WorkloadSpec> {
    let half_payload = payload_bytes / 2;
    let activation_bytes = activation_bytes_for_payload(payload_bytes);
    let output_bytes = output_bytes_for_payload(payload_bytes);
    let mut specs = vec![
        WorkloadSpec {
            workload_id: "cpu_reference_serialized_small_payload".to_string(),
            comparison_group: SMALL_PAYLOAD_COMPARISON_GROUP.to_string(),
            workload_class: "cpu_reference_compound".to_string(),
            placement_strategy: "single_target_serial".to_string(),
            pattern: "serialized_small_payload".to_string(),
            format: "u8_synthetic".to_string(),
            participant_count: 1,
            payload_bytes,
            parameter_bytes_per_participant: payload_bytes,
            activation_bytes,
            output_bytes,
            description: "Run the full small logical payload through the CPU reference serialized dataflow.".to_string(),
        },
        WorkloadSpec {
            workload_id: "cpu_reference_layer_split_small_payload".to_string(),
            comparison_group: SMALL_PAYLOAD_COMPARISON_GROUP.to_string(),
            workload_class: "cpu_reference_compound".to_string(),
            placement_strategy: "two_stage_serial_reference".to_string(),
            pattern: "synthetic_layer_split_small_payload".to_string(),
            format: "u8_synthetic".to_string(),
            participant_count: 1,
            payload_bytes,
            parameter_bytes_per_participant: half_payload,
            activation_bytes,
            output_bytes,
            description: "Run the same small logical payload through the CPU reference layer-split dataflow.".to_string(),
        },
        WorkloadSpec {
            workload_id: "cpu_reference_tensor_split_small_payload".to_string(),
            comparison_group: SMALL_PAYLOAD_COMPARISON_GROUP.to_string(),
            workload_class: "cpu_reference_compound".to_string(),
            placement_strategy: "two_shard_parallel_reference".to_string(),
            pattern: "synthetic_tensor_split_small_payload".to_string(),
            format: "u8_synthetic".to_string(),
            participant_count: 1,
            payload_bytes,
            parameter_bytes_per_participant: half_payload,
            activation_bytes,
            output_bytes,
            description: "Run the same small logical payload through the CPU reference tensor-split dataflow.".to_string(),
        },
    ];
    for format in formats {
        for workload in workloads {
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
                description: format!(
                    "Run the full {workload} small logical payload on one target using {format}."
                ),
            });
            specs.push(WorkloadSpec {
                workload_id: format_workload_id("ordered_activation_transfer", workload, format),
                comparison_group: SMALL_PAYLOAD_COMPARISON_GROUP.to_string(),
                workload_class: workload.clone(),
                placement_strategy: "activation_transfer_only".to_string(),
                pattern: "ordered_activation_transfer".to_string(),
                format: format.clone(),
                participant_count: 2,
                payload_bytes,
                parameter_bytes_per_participant: 0,
                activation_bytes,
                output_bytes: 0,
                description: format!("Move one activation-sized {format} payload for {workload} from source to destination."),
            });
            specs.push(WorkloadSpec {
                workload_id: format_workload_id("synthetic_layer_split_small_payload", workload, format),
                comparison_group: SMALL_PAYLOAD_COMPARISON_GROUP.to_string(),
                workload_class: workload.clone(),
                placement_strategy: "two_target_serial".to_string(),
                pattern: "synthetic_layer_split_small_payload".to_string(),
                format: format.clone(),
                participant_count: 2,
                payload_bytes,
                parameter_bytes_per_participant: half_payload,
                activation_bytes,
                output_bytes,
                description: format!("Run half the {workload} logical payload using {format} on the first target, transfer activation, then run the other half on the second target."),
            });
            specs.push(WorkloadSpec {
                workload_id: format_workload_id("synthetic_tensor_split_small_payload", workload, format),
                comparison_group: SMALL_PAYLOAD_COMPARISON_GROUP.to_string(),
                workload_class: workload.clone(),
                placement_strategy: "two_target_parallel".to_string(),
                pattern: "synthetic_tensor_split_small_payload".to_string(),
                format: format.clone(),
                participant_count: 2,
                payload_bytes,
                parameter_bytes_per_participant: half_payload,
                activation_bytes,
                output_bytes,
                description: format!("Split the same {workload} logical {format} payload across two targets, broadcast activation, compute shards, then collect output."),
            });
            if max_group_size >= 3 {
                let third_payload = payload_bytes / 3;
                specs.push(WorkloadSpec {
                    workload_id: format_workload_id("synthetic_layer_split_group_small_payload", workload, format),
                    comparison_group: SMALL_PAYLOAD_COMPARISON_GROUP.to_string(),
                    workload_class: workload.clone(),
                    placement_strategy: "three_target_serial".to_string(),
                    pattern: "synthetic_layer_split_group_small_payload".to_string(),
                    format: format.clone(),
                    participant_count: 3,
                    payload_bytes,
                    parameter_bytes_per_participant: third_payload,
                    activation_bytes,
                    output_bytes,
                    description: format!("Run thirds of the {workload} logical {format} payload across three ordered targets with activation movement between stages."),
                });
                specs.push(WorkloadSpec {
                    workload_id: format_workload_id("synthetic_tensor_split_group_small_payload", workload, format),
                    comparison_group: SMALL_PAYLOAD_COMPARISON_GROUP.to_string(),
                    workload_class: workload.clone(),
                    placement_strategy: "three_target_parallel".to_string(),
                    pattern: "synthetic_tensor_split_group_small_payload".to_string(),
                    format: format.clone(),
                    participant_count: 3,
                    payload_bytes,
                    parameter_bytes_per_participant: third_payload,
                    activation_bytes,
                    output_bytes,
                    description: format!("Split the same {workload} logical {format} payload across three targets, broadcast activation, compute shards, then collect output."),
                });
            }
        }
    }
    specs
}

fn format_workload_id(base: &str, workload_class: &str, format: &str) -> String {
    format!("{base}:{workload_class}:{format}")
}

fn split_bytes(total: usize, parts: usize) -> Vec<usize> {
    let base = total / parts;
    let remainder = total % parts;
    (0..parts)
        .map(|index| base + usize::from(index < remainder))
        .collect()
}

fn activation_bytes_for_payload(payload_bytes: usize) -> usize {
    payload_bytes.clamp(4 * 1024, 256 * 1024)
}

fn output_bytes_for_payload(payload_bytes: usize) -> usize {
    (payload_bytes / 16).clamp(4 * 1024, 512 * 1024)
}

fn summarize(samples: &[Sample]) -> Option<Summary> {
    if samples.is_empty() {
        return None;
    }
    let mut durations = samples
        .iter()
        .map(|sample| sample.duration_ns)
        .collect::<Vec<_>>();
    durations.sort_unstable();
    let median = durations[durations.len() / 2];
    let min = durations[0];
    let total_bytes = samples
        .iter()
        .map(|sample| sample.bytes_read + sample.bytes_written)
        .sum::<u64>() as f64;
    let total_operations = samples.iter().map(|sample| sample.operations).sum::<u64>() as f64;
    let total_seconds = samples
        .iter()
        .map(|sample| sample.duration_ns as f64 / 1_000_000_000.0)
        .sum::<f64>();
    Some(Summary {
        min_duration_ns: min,
        median_duration_ns: median,
        bytes_per_second: total_bytes / total_seconds.max(f64::EPSILON),
        operations_per_second: total_operations / total_seconds.max(f64::EPSILON),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn formats() -> Vec<String> {
        vec!["f32".to_string()]
    }

    fn workloads() -> Vec<String> {
        vec!["dense_projection".to_string()]
    }

    fn target(id: &str, kind: &str) -> Target {
        Target {
            stable_target_id: id.to_string(),
            backend: "test".to_string(),
            kind: kind.to_string(),
            name: id.to_string(),
            vendor_id: None,
            vendor_name: None,
            device_id: None,
            pci_address: None,
            physical_location: None,
            numa_node: None,
            boot_vga: None,
            pci_link: None,
            capabilities: Vec::new(),
            format_capabilities: Vec::new(),
            diagnostics: Vec::new(),
        }
    }

    #[test]
    fn summarizes_samples() {
        let samples = [
            Sample {
                sample_index: 0,
                duration_ns: 30,
                iterations: 1,
                bytes_read: 10,
                bytes_written: 10,
                operations: 2,
            },
            Sample {
                sample_index: 1,
                duration_ns: 10,
                iterations: 1,
                bytes_read: 10,
                bytes_written: 10,
                operations: 2,
            },
            Sample {
                sample_index: 2,
                duration_ns: 20,
                iterations: 1,
                bytes_read: 10,
                bytes_written: 10,
                operations: 2,
            },
        ];
        let summary = summarize(&samples).unwrap();
        assert_eq!(summary.min_duration_ns, 10);
        assert_eq!(summary.median_duration_ns, 20);
    }

    #[test]
    fn workload_specs_describe_small_split_patterns() {
        let formats = formats();
        let workloads = workloads();
        let specs = build_workload_specs(5 * 1024 * 1024, &formats, &workloads, 2);
        let ids = specs
            .iter()
            .map(|spec| spec.workload_id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            ids,
            [
                "cpu_reference_serialized_small_payload",
                "cpu_reference_layer_split_small_payload",
                "cpu_reference_tensor_split_small_payload",
                "single_target_small_payload:dense_projection:f32",
                "ordered_activation_transfer:dense_projection:f32",
                "synthetic_layer_split_small_payload:dense_projection:f32",
                "synthetic_tensor_split_small_payload:dense_projection:f32",
            ]
        );
        let layer = specs
            .iter()
            .find(|spec| {
                spec.workload_id == "synthetic_layer_split_small_payload:dense_projection:f32"
            })
            .unwrap();
        assert_eq!(layer.participant_count, 2);
        assert_eq!(layer.comparison_group, SMALL_PAYLOAD_COMPARISON_GROUP);
        assert_eq!(layer.placement_strategy, "two_target_serial");
        assert_eq!(layer.parameter_bytes_per_participant, 2_621_440);
        assert!(layer.activation_bytes <= 256 * 1024);

        let strategies = specs
            .iter()
            .map(|spec| spec.placement_strategy.as_str())
            .collect::<BTreeSet<_>>();
        assert!(strategies.contains("single_target_serial"));
        assert!(strategies.contains("two_target_serial"));
        assert!(strategies.contains("two_target_parallel"));
    }

    #[test]
    fn cpu_compound_reference_runs_all_small_patterns() {
        let measurements = run_cpu_compound_reference("cpu:host", 64 * 1024, 1);
        let ids = measurements
            .iter()
            .map(|measurement| measurement.workload_id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            ids,
            [
                "cpu_reference_serialized_small_payload",
                "cpu_reference_layer_split_small_payload",
                "cpu_reference_tensor_split_small_payload",
            ]
        );
        assert!(
            measurements
                .iter()
                .all(|measurement| measurement.status == "completed")
        );
        assert!(
            measurements
                .iter()
                .all(|measurement| measurement.summary.is_some())
        );
    }

    #[test]
    fn group_specs_describe_triplet_patterns() {
        let formats = formats();
        let workloads = workloads();
        let specs = build_workload_specs(5 * 1024 * 1024, &formats, &workloads, 3);
        let ids = specs
            .iter()
            .map(|spec| spec.workload_id.as_str())
            .collect::<Vec<_>>();
        assert!(ids.contains(&"synthetic_layer_split_group_small_payload:dense_projection:f32"));
        assert!(ids.contains(&"synthetic_tensor_split_group_small_payload:dense_projection:f32"));
        let split = split_bytes(10, 3);
        assert_eq!(split, [4, 3, 3]);
    }

    #[test]
    fn run_emits_triplet_placeholders_without_backend_access() {
        let targets = vec![
            target("gpu:a", "discrete_gpu"),
            target("gpu:b", "discrete_gpu"),
            target("gpu:c", "discrete_gpu"),
        ];
        let selection = Selection {
            selected_target_ids: targets
                .iter()
                .map(|target| target.stable_target_id.clone())
                .collect(),
            skipped_targets: Vec::new(),
            diagnostics: Vec::new(),
        };
        let policy = RunPolicy {
            payload_bytes: 5 * 1024 * 1024,
            samples: 1,
            benchmark_formats: formats(),
            benchmark_workloads: workloads(),
            include_targets: Vec::new(),
            exclude_targets: Vec::new(),
            exclude_pci: Vec::new(),
            exclude_kinds: Vec::new(),
            pair_measurements: true,
            max_group_size: 3,
        };
        let run = run_benchmarks(targets, selection, policy);
        assert_eq!(run.group_measurements.len(), 2);
        assert!(
            run.group_measurements
                .iter()
                .any(|measurement| measurement.operation_family == "tensor_split")
        );
        assert_eq!(
            run.group_measurements[0]
                .payload_bytes_per_participant
                .len(),
            3
        );
    }

    #[test]
    fn pair_comparison_sets_include_single_serial_and_parallel_candidates() {
        let targets = vec![
            target("gpu:a", "discrete_gpu"),
            target("gpu:b", "discrete_gpu"),
        ];
        let selection = Selection {
            selected_target_ids: targets
                .iter()
                .map(|target| target.stable_target_id.clone())
                .collect(),
            skipped_targets: Vec::new(),
            diagnostics: Vec::new(),
        };
        let policy = RunPolicy {
            payload_bytes: 5 * 1024 * 1024,
            samples: 1,
            benchmark_formats: formats(),
            benchmark_workloads: workloads(),
            include_targets: Vec::new(),
            exclude_targets: Vec::new(),
            exclude_pci: Vec::new(),
            exclude_kinds: Vec::new(),
            pair_measurements: true,
            max_group_size: 2,
        };
        let run = run_benchmarks(targets, selection, policy);
        assert_eq!(run.comparison_sets.len(), 1);
        let strategies = run.comparison_sets[0]
            .candidates
            .iter()
            .map(|candidate| candidate.placement_strategy.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            strategies,
            [
                "single_target_serial",
                "single_target_serial",
                "two_target_serial",
                "two_target_serial",
                "two_target_parallel",
            ]
        );
        let serial_pair_count = run
            .pair_measurements
            .iter()
            .filter(|measurement| measurement.placement_strategy == "two_target_serial")
            .count();
        assert_eq!(serial_pair_count, 2);
    }

    #[test]
    fn cpu_requested_single_target_measurements_match_comparison_candidates() {
        let targets = vec![target("cpu:host", "cpu"), target("gpu:a", "discrete_gpu")];
        let selection = Selection {
            selected_target_ids: targets
                .iter()
                .map(|target| target.stable_target_id.clone())
                .collect(),
            skipped_targets: Vec::new(),
            diagnostics: Vec::new(),
        };
        let policy = RunPolicy {
            payload_bytes: 64 * 1024,
            samples: 1,
            benchmark_formats: vec!["f32".to_string(), "fp8".to_string()],
            benchmark_workloads: workloads(),
            include_targets: Vec::new(),
            exclude_targets: Vec::new(),
            exclude_pci: Vec::new(),
            exclude_kinds: Vec::new(),
            pair_measurements: true,
            max_group_size: 2,
        };
        let run = run_benchmarks(targets, selection, policy);
        assert!(run.measurements.iter().any(|measurement| {
            measurement.target_id == "cpu:host"
                && measurement.workload_id == "single_target_small_payload:dense_projection:f32"
                && measurement.status == "completed"
        }));
        assert!(run.measurements.iter().any(|measurement| {
            measurement.target_id == "cpu:host"
                && measurement.workload_id == "single_target_small_payload:dense_projection:fp8"
                && measurement.status == "unsupported"
        }));

        let summary = run.summary();
        assert!(summary.candidate_statuses.iter().any(|candidate| {
            candidate.workload_class == "dense_projection"
                && candidate.format == "f32"
                && candidate.placement_strategy == "single_target_serial"
                && candidate.status == "completed"
        }));
    }
}
