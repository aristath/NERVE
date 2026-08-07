use std::collections::BTreeSet;
use std::hint::black_box;
use std::time::Instant;

use crate::model::{
    BenchmarkRun, Implementation, Measurement, PairMeasurement, RUN_SCHEMA, RunPolicy, Sample,
    Selection, Summary, Target, WorkloadSpec, now_unix_ms,
};

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
            ));
        } else {
            measurements.push(unmeasured_single_target(
                &target.stable_target_id,
                policy.payload_bytes,
                "gpu_backend_not_implemented",
            ));
        }
    }

    let pair_measurements = if policy.pair_measurements {
        build_pair_placeholders(&selected_targets, policy.payload_bytes)
    } else {
        Vec::new()
    };
    let workload_specs = build_workload_specs(policy.payload_bytes);

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
        measurements,
        pair_measurements,
        diagnostics,
    }
}

fn run_cpu_measurements(target_id: &str, payload_bytes: usize, samples: usize) -> Vec<Measurement> {
    vec![
        run_cpu_copy(target_id, payload_bytes, samples),
        run_cpu_f32_dot(target_id, payload_bytes, samples),
    ]
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

fn unmeasured_single_target(target_id: &str, payload_bytes: usize, reason: &str) -> Measurement {
    Measurement {
        workload_id: "single_target_gpu_small_payload".to_string(),
        target_id: target_id.to_string(),
        pattern: "single_target_gpu_compute".to_string(),
        operation_family: "backend_compute".to_string(),
        regime: "small_payload".to_string(),
        format: "backend_selected".to_string(),
        status: "unmeasured".to_string(),
        reason: Some(reason.to_string()),
        payload_bytes,
        working_set_bytes: payload_bytes,
        samples: Vec::new(),
        summary: None,
    }
}

fn build_pair_placeholders(targets: &[&Target], payload_bytes: usize) -> Vec<PairMeasurement> {
    let mut measurements = Vec::new();
    for source in targets {
        for destination in targets {
            if source.stable_target_id == destination.stable_target_id {
                continue;
            }
            measurements.push(unmeasured_pair(
                &source.stable_target_id,
                &destination.stable_target_id,
                "ordered_activation_transfer",
                "activation_transfer",
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
                "layer_split",
                "requires_device_backend",
                payload_bytes,
            ));
            measurements.push(unmeasured_pair(
                left,
                right,
                "synthetic_tensor_split_small_payload",
                "tensor_split",
                "requires_device_backend",
                payload_bytes,
            ));
        }
    }
    measurements
}

fn unmeasured_pair(
    source: &str,
    destination: &str,
    pattern: &str,
    operation_family: &str,
    reason: &str,
    payload_bytes: usize,
) -> PairMeasurement {
    let half_payload = payload_bytes / 2;
    let activation_bytes = activation_bytes_for_payload(payload_bytes);
    let output_bytes = output_bytes_for_payload(payload_bytes);
    PairMeasurement {
        workload_id: pattern.to_string(),
        source_target_id: source.to_string(),
        destination_target_id: destination.to_string(),
        pattern: pattern.to_string(),
        operation_family: operation_family.to_string(),
        regime: "small_payload".to_string(),
        format: "backend_selected".to_string(),
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

fn build_workload_specs(payload_bytes: usize) -> Vec<WorkloadSpec> {
    let half_payload = payload_bytes / 2;
    let activation_bytes = activation_bytes_for_payload(payload_bytes);
    let output_bytes = output_bytes_for_payload(payload_bytes);
    vec![
        WorkloadSpec {
            workload_id: "single_target_gpu_small_payload".to_string(),
            pattern: "single_target_gpu_compute".to_string(),
            format: "backend_selected".to_string(),
            participant_count: 1,
            payload_bytes,
            parameter_bytes_per_participant: payload_bytes,
            activation_bytes,
            output_bytes,
            description: "Run the full small logical payload on one target.".to_string(),
        },
        WorkloadSpec {
            workload_id: "ordered_activation_transfer".to_string(),
            pattern: "ordered_activation_transfer".to_string(),
            format: "backend_selected".to_string(),
            participant_count: 2,
            payload_bytes,
            parameter_bytes_per_participant: 0,
            activation_bytes,
            output_bytes: 0,
            description: "Move one activation-sized payload from source to destination.".to_string(),
        },
        WorkloadSpec {
            workload_id: "synthetic_layer_split_small_payload".to_string(),
            pattern: "synthetic_layer_split_small_payload".to_string(),
            format: "backend_selected".to_string(),
            participant_count: 2,
            payload_bytes,
            parameter_bytes_per_participant: half_payload,
            activation_bytes,
            output_bytes,
            description: "Run half the logical payload on the first target, transfer the activation, then run the other half on the second target.".to_string(),
        },
        WorkloadSpec {
            workload_id: "synthetic_tensor_split_small_payload".to_string(),
            pattern: "synthetic_tensor_split_small_payload".to_string(),
            format: "backend_selected".to_string(),
            participant_count: 2,
            payload_bytes,
            parameter_bytes_per_participant: half_payload,
            activation_bytes,
            output_bytes,
            description: "Split the same logical payload across two targets, broadcast the activation, compute both shards, then collect the output.".to_string(),
        },
    ]
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
        let specs = build_workload_specs(5 * 1024 * 1024);
        let ids = specs
            .iter()
            .map(|spec| spec.workload_id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            ids,
            [
                "single_target_gpu_small_payload",
                "ordered_activation_transfer",
                "synthetic_layer_split_small_payload",
                "synthetic_tensor_split_small_payload",
            ]
        );
        let layer = specs
            .iter()
            .find(|spec| spec.workload_id == "synthetic_layer_split_small_payload")
            .unwrap();
        assert_eq!(layer.participant_count, 2);
        assert_eq!(layer.parameter_bytes_per_participant, 2_621_440);
        assert!(layer.activation_bytes <= 256 * 1024);
    }
}
