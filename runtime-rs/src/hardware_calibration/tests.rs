use super::schema::*;
use super::*;
use crate::hardware_profile::stable_hardware_id;
use serde_json::Value;
use std::collections::BTreeMap;
#[cfg(feature = "vulkan")]
use std::process::Command;

fn workload(operation: &str, regime: &[(&str, &str)]) -> HardwareCalibrationWorkload {
    let mut workload = HardwareCalibrationWorkload {
        workload_id: String::new(),
        process_names: vec!["synthetic_cpu_process".to_string()],
        executor: CalibrationExecutor::Cpu,
        operation: operation.to_string(),
        regime: regime
            .iter()
            .map(|(name, value)| ((*name).to_string(), (*value).to_string()))
            .collect(),
        work: CalibrationUsefulWork {
            items_per_iteration: 64,
            operations_per_iteration: 512,
            bytes_read_per_iteration: 512,
            bytes_written_per_iteration: 512,
        },
        artifacts: Vec::new(),
        validation: CalibrationValidationContract {
            mode: "digest".to_string(),
            expected_digest: None,
            maximum_error_ppm: 0,
        },
    };
    workload.workload_id = stable_hardware_id(
        "calibration_workload",
        &[
            serde_json::to_value(&workload.process_names).unwrap(),
            serde_json::to_value(workload.executor).unwrap(),
            Value::String(workload.operation.clone()),
            serde_json::to_value(&workload.regime).unwrap(),
            serde_json::to_value(&workload.work).unwrap(),
            serde_json::to_value(&workload.artifacts).unwrap(),
            serde_json::to_value(&workload.validation).unwrap(),
        ],
    )
    .unwrap();
    workload
}

#[cfg(feature = "vulkan")]
#[test]
fn explicit_vulkan_calibration_runs_a_real_resident_shader_sequentially() {
    let Some(device_index) = std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
    else {
        eprintln!("skipping Vulkan calibration: explicit idle AMD device is not selected");
        return;
    };
    let mut shader = workload(
        "shader_scalar",
        &[("dependency", "independent_chains"), ("format", "f32")],
    );
    shader.executor = CalibrationExecutor::VulkanCompute;
    shader.work = CalibrationUsefulWork {
        items_per_iteration: 65_536,
        operations_per_iteration: 8 * 65_536,
        bytes_read_per_iteration: 4 * 4 * 65_536,
        bytes_written_per_iteration: 4 * 65_536,
    };
    shader.artifacts = vec![CalibrationArtifactDeclaration {
        name: "shader_scalar_f32".to_string(),
        kind: "spirv_compute".to_string(),
        digest: None,
    }];
    shader.workload_id = stable_hardware_id(
        "calibration_workload",
        &[
            serde_json::to_value(&shader.process_names).unwrap(),
            serde_json::to_value(shader.executor).unwrap(),
            Value::String(shader.operation.clone()),
            serde_json::to_value(&shader.regime).unwrap(),
            serde_json::to_value(&shader.work).unwrap(),
            serde_json::to_value(&shader.artifacts).unwrap(),
            serde_json::to_value(&shader.validation).unwrap(),
        ],
    )
    .unwrap();
    let mut calibration_plan = plan(vec![shader]);
    calibration_plan.policy.minimum_sample_duration_ns = 1_000_000;
    calibration_plan.policy.sustained_window_duration_ms = 1;
    calibration_plan.policy.sustained_window_count = 1;
    calibration_plan.plan_id = stable_hardware_id(
        "calibration_plan",
        &[
            Value::String(calibration_plan.hardware_profile_id.clone()),
            Value::String(calibration_plan.capability_class.clone()),
            serde_json::to_value(&calibration_plan.implementation).unwrap(),
            serde_json::to_value(&calibration_plan.policy).unwrap(),
            serde_json::to_value(&calibration_plan.workloads).unwrap(),
            serde_json::to_value(&calibration_plan.excluded_processes).unwrap(),
        ],
    )
    .unwrap();
    let temporary = std::env::temp_dir().join(format!(
        "nerve-vulkan-calibration-test-{}",
        std::process::id()
    ));
    let result = run_calibration_plan(
        &calibration_plan,
        &CalibrationRunnerOptions {
            artifact_directory: temporary.clone(),
            vulkan_physical_device_index: Some(device_index),
            ..CalibrationRunnerOptions::default()
        },
    )
    .unwrap();
    assert_eq!(result.status, CalibrationRunStatus::Completed);
    assert_eq!(result.workloads.len(), 1);
    assert_eq!(result.workloads[0].artifacts.len(), 1);
    assert!(
        result.workloads[0]
            .samples
            .iter()
            .all(|sample| sample.valid)
    );
    std::fs::remove_dir_all(temporary).unwrap();
}

#[cfg(feature = "vulkan")]
#[test]
fn explicit_vulkan_transfer_calibration_moves_real_bytes_sequentially() {
    let Some(device_index) = std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
    else {
        eprintln!("skipping Vulkan transfer calibration: explicit idle AMD device is unset");
        return;
    };
    let mut workloads = Vec::new();
    for direction in ["host_to_device", "device_to_host", "device_to_device"] {
        let mut transfer = workload(
            "buffer_copy",
            &[("bytes", "1048576"), ("direction", direction)],
        );
        transfer.executor = CalibrationExecutor::VulkanTransfer;
        transfer.work = CalibrationUsefulWork {
            items_per_iteration: 262_144,
            operations_per_iteration: 262_144,
            bytes_read_per_iteration: 1_048_576,
            bytes_written_per_iteration: 1_048_576,
        };
        transfer.workload_id = stable_hardware_id(
            "calibration_workload",
            &[
                serde_json::to_value(&transfer.process_names).unwrap(),
                serde_json::to_value(transfer.executor).unwrap(),
                Value::String(transfer.operation.clone()),
                serde_json::to_value(&transfer.regime).unwrap(),
                serde_json::to_value(&transfer.work).unwrap(),
                serde_json::to_value(&transfer.artifacts).unwrap(),
                serde_json::to_value(&transfer.validation).unwrap(),
            ],
        )
        .unwrap();
        workloads.push(transfer);
    }
    let calibration_plan = plan(workloads);
    let result = run_calibration_plan(
        &calibration_plan,
        &CalibrationRunnerOptions {
            vulkan_physical_device_index: Some(device_index),
            ..CalibrationRunnerOptions::default()
        },
    )
    .unwrap();
    assert_eq!(result.status, CalibrationRunStatus::Completed);
    assert_eq!(result.workloads.len(), 3);
    assert!(result.workloads.iter().all(|workload| {
        workload.validation.status == CalibrationValidationStatus::Passed
            && workload.samples.iter().all(|sample| sample.valid)
    }));
}

#[cfg(feature = "vulkan")]
#[test]
fn explicit_vulkan_texture_calibration_samples_a_real_image_sequentially() {
    let Some(device_index) = std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
    else {
        eprintln!("skipping Vulkan texture calibration: explicit idle AMD device is unset");
        return;
    };
    let mut texture = workload("texture_sampling", &[("filter", "linear")]);
    texture.executor = CalibrationExecutor::VulkanGraphics;
    texture.work = CalibrationUsefulWork {
        items_per_iteration: 65_536,
        operations_per_iteration: 65_536 * 8,
        bytes_read_per_iteration: 65_536 * 8,
        bytes_written_per_iteration: 65_536 * 4,
    };
    texture.artifacts = vec![CalibrationArtifactDeclaration {
        name: "texture_sampling_linear".to_string(),
        kind: "spirv_compute".to_string(),
        digest: None,
    }];
    texture.workload_id = stable_hardware_id(
        "calibration_workload",
        &[
            serde_json::to_value(&texture.process_names).unwrap(),
            serde_json::to_value(texture.executor).unwrap(),
            Value::String(texture.operation.clone()),
            serde_json::to_value(&texture.regime).unwrap(),
            serde_json::to_value(&texture.work).unwrap(),
            serde_json::to_value(&texture.artifacts).unwrap(),
            serde_json::to_value(&texture.validation).unwrap(),
        ],
    )
    .unwrap();
    let calibration_plan = plan(vec![texture]);
    let temporary = std::env::temp_dir().join(format!(
        "nerve-vulkan-texture-calibration-test-{}",
        std::process::id()
    ));
    let result = run_calibration_plan(
        &calibration_plan,
        &CalibrationRunnerOptions {
            artifact_directory: temporary.clone(),
            vulkan_physical_device_index: Some(device_index),
            ..CalibrationRunnerOptions::default()
        },
    )
    .unwrap();
    assert_eq!(result.status, CalibrationRunStatus::Completed);
    assert_eq!(result.workloads.len(), 1);
    assert_eq!(
        result.workloads[0].validation.status,
        CalibrationValidationStatus::Passed
    );
    assert!(
        result.workloads[0]
            .validation
            .observed_digest
            .as_deref()
            .is_some_and(|digest| digest.starts_with("nerve.calibration_output_sha256.v1:"))
    );
    assert!(result.workloads[0].samples.iter().all(|sample| sample.valid));
    std::fs::remove_dir_all(temporary).unwrap();
}

#[cfg(feature = "vulkan")]
#[test]
fn explicit_vulkan_compute_calibration_executes_every_native_family_sequentially() {
    let Some(device_index) = std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
    else {
        eprintln!("skipping native Vulkan calibration: explicit idle AMD device is unset");
        return;
    };
    let cases = [
        ("shader_scalar", "format", "f32"),
        ("shader_scalar", "format", "f64"),
        ("shader_scalar", "format", "f16"),
        ("shader_scalar", "format", "i8"),
        ("shader_scalar", "format", "u8"),
        ("shader_scalar", "format", "i16"),
        ("shader_scalar", "format", "u16"),
        ("shader_scalar", "format", "i32"),
        ("shader_scalar", "format", "u32"),
        ("shader_scalar", "format", "i64"),
        ("shader_scalar", "format", "u64"),
        ("packed_dot_product", "format", "i8"),
        ("packed_dot_product", "format", "bf16"),
        ("packed_dot_product", "format", "f16"),
        ("packed_dot_product", "format", "f8_e4m3"),
        ("cooperative_matrix_multiply", "format", "f16"),
        ("cooperative_matrix_multiply", "format", "bf16"),
        ("cooperative_matrix_multiply", "format", "f8_e4m3"),
        ("subgroup_reduce", "operation", "reduce"),
        ("subgroup_scan", "operation", "scan"),
        ("subgroup_shuffle", "operation", "shuffle"),
        ("subgroup_ballot", "operation", "ballot"),
        ("sequential_copy", "working_set_bytes", "262144"),
        ("strided_read", "working_set_bytes", "262144"),
        ("gather_scatter", "working_set_bytes", "262144"),
        ("packed_decode", "working_set_bytes", "262144"),
        ("register_pressure_sweep", "working_set_bytes", "262144"),
        ("shared_memory_tiled_copy", "working_set_bytes", "262144"),
        ("atomic_add", "contention", "independent"),
        ("atomic_add", "contention", "workgroup"),
        ("atomic_add", "contention", "global"),
        ("command_queues", "dispatch_count", "1"),
        ("indirect_work_generation", "dispatch_count", "16"),
        ("resident_command_replay", "dispatch_count", "1"),
        ("synchronization_round_trip", "primitive", "fence"),
    ];
    let workloads = cases
        .into_iter()
        .enumerate()
        .map(|(index, (operation, regime_name, regime_value))| {
            let mut candidate = workload(operation, &[(regime_name, regime_value)]);
            candidate.executor = CalibrationExecutor::VulkanCompute;
            let items = if operation == "cooperative_matrix_multiply" {
                64
            } else if matches!(
                operation,
                "command_queues" | "resident_command_replay" | "synchronization_round_trip"
            ) {
                1
            } else {
                65_536
            };
            candidate.work = CalibrationUsefulWork {
                items_per_iteration: items,
                operations_per_iteration: items.saturating_mul(64),
                bytes_read_per_iteration: items.saturating_mul(16),
                bytes_written_per_iteration: items.saturating_mul(4),
            };
            candidate.artifacts = vec![CalibrationArtifactDeclaration {
                name: format!("native_family_{index:03}"),
                kind: "spirv_compute".to_string(),
                digest: None,
            }];
            candidate.workload_id = stable_hardware_id(
                "calibration_workload",
                &[
                    serde_json::to_value(&candidate.process_names).unwrap(),
                    serde_json::to_value(candidate.executor).unwrap(),
                    Value::String(candidate.operation.clone()),
                    serde_json::to_value(&candidate.regime).unwrap(),
                    serde_json::to_value(&candidate.work).unwrap(),
                    serde_json::to_value(&candidate.artifacts).unwrap(),
                    serde_json::to_value(&candidate.validation).unwrap(),
                ],
            )
            .unwrap();
            candidate
        })
        .collect();
    let calibration_plan = plan(workloads);
    let temporary = std::env::temp_dir().join(format!(
        "nerve-vulkan-native-family-test-{}",
        std::process::id()
    ));
    let result = run_calibration_plan(
        &calibration_plan,
        &CalibrationRunnerOptions {
            artifact_directory: temporary.clone(),
            vulkan_physical_device_index: Some(device_index),
            ..CalibrationRunnerOptions::default()
        },
    )
    .unwrap();
    assert_eq!(result.status, CalibrationRunStatus::Completed);
    assert_eq!(result.workloads.len(), cases.len());
    assert!(result.workloads.iter().all(|workload| {
        workload.validation.status == CalibrationValidationStatus::Passed
            && workload.samples.iter().all(|sample| sample.valid)
    }));
    std::fs::remove_dir_all(temporary).unwrap();
}

#[cfg(feature = "vulkan")]
#[test]
fn every_generated_vulkan_compute_calibration_shader_compiles_for_vulkan_1_4() {
    let cases = [
        ("shader_scalar", "format", "f32"),
        ("shader_scalar", "format", "f64"),
        ("shader_scalar", "format", "f16"),
        ("shader_scalar", "format", "i8"),
        ("shader_scalar", "format", "u8"),
        ("shader_scalar", "format", "i16"),
        ("shader_scalar", "format", "u16"),
        ("shader_scalar", "format", "i32"),
        ("shader_scalar", "format", "u32"),
        ("shader_scalar", "format", "i64"),
        ("shader_scalar", "format", "u64"),
        ("packed_dot_product", "format", "i8"),
        ("packed_dot_product", "format", "bf16"),
        ("packed_dot_product", "format", "f16"),
        ("packed_dot_product", "format", "f8_e4m3"),
        ("cooperative_matrix_multiply", "format", "f16"),
        ("cooperative_matrix_multiply", "format", "bf16"),
        ("cooperative_matrix_multiply", "format", "f8_e4m3"),
        ("subgroup_reduce", "operation", "reduce"),
        ("subgroup_scan", "operation", "scan"),
        ("subgroup_shuffle", "operation", "shuffle"),
        ("subgroup_ballot", "operation", "ballot"),
        ("sequential_copy", "working_set_bytes", "4096"),
        ("strided_read", "working_set_bytes", "4096"),
        ("gather_scatter", "working_set_bytes", "4096"),
        ("packed_decode", "working_set_bytes", "4096"),
        ("register_pressure_sweep", "working_set_bytes", "4096"),
        ("shared_memory_tiled_copy", "working_set_bytes", "4096"),
        ("atomic_add", "contention", "independent"),
        ("atomic_add", "contention", "workgroup"),
        ("atomic_add", "contention", "global"),
        ("command_queues", "dispatch_count", "1"),
        ("device_generated_commands", "dispatch_count", "1"),
        ("indirect_work_generation", "dispatch_count", "1"),
        ("resident_command_replay", "dispatch_count", "1"),
        ("synchronization_round_trip", "primitive", "fence"),
    ];
    let temporary = std::env::temp_dir().join(format!(
        "nerve-vulkan-calibration-shader-contract-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&temporary).unwrap();
    for (index, (operation, regime_name, regime_value)) in cases.into_iter().enumerate() {
        let candidate = workload(operation, &[(regime_name, regime_value)]);
        let source = super::vulkan_compute_shaders::compute_shader_source(&candidate).unwrap();
        let source_path = temporary.join(format!("{index:03}_{operation}.comp"));
        let spirv_path = temporary.join(format!("{index:03}_{operation}.spv"));
        std::fs::write(&source_path, source).unwrap();
        let output = Command::new("glslangValidator")
            .arg("-V")
            .arg("--target-env")
            .arg("vulkan1.4")
            .arg("-o")
            .arg(&spirv_path)
            .arg(&source_path)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "calibration shader {operation}/{regime_value} did not compile:\n{}",
            format!(
                "{}{}",
                String::from_utf8_lossy(&output.stderr),
                String::from_utf8_lossy(&output.stdout)
            )
        );
        assert!(std::fs::metadata(&spirv_path).unwrap().len() > 20);
    }
    std::fs::remove_dir_all(temporary).unwrap();
}

fn plan(mut workloads: Vec<HardwareCalibrationWorkload>) -> HardwareCalibrationPlan {
    workloads.sort_by(|left, right| left.workload_id.cmp(&right.workload_id));
    let mut plan = HardwareCalibrationPlan {
        schema: HARDWARE_CALIBRATION_PLAN_SCHEMA.to_string(),
        plan_id: String::new(),
        hardware_profile_id: format!("hardware_profile_{}", "1".repeat(32)),
        capability_class: format!("hardware_capability_{}", "2".repeat(32)),
        implementation: CalibrationImplementation {
            name: "nerve-hardware-calibrator".to_string(),
            version: "1".to_string(),
            fingerprint: format!("nerve.hardware_calibrator_sha256.v1:{}", "3".repeat(64)),
        },
        policy: HardwareCalibrationPolicy {
            warmup_iterations: 1,
            steady_iterations: 5,
            minimum_sample_duration_ns: 1,
            sustained_window_duration_ms: 1,
            sustained_window_count: 1,
            confidence_level_ppm: 950_000,
            maximum_relative_ci_width_ppm: 500_000,
        },
        workloads,
        excluded_processes: Vec::new(),
    };
    plan.plan_id = stable_hardware_id(
        "calibration_plan",
        &[
            Value::String(plan.hardware_profile_id.clone()),
            Value::String(plan.capability_class.clone()),
            serde_json::to_value(&plan.implementation).unwrap(),
            serde_json::to_value(&plan.policy).unwrap(),
            serde_json::to_value(&plan.workloads).unwrap(),
            serde_json::to_value(&plan.excluded_processes).unwrap(),
        ],
    )
    .unwrap();
    plan
}

#[test]
fn calibration_plan_rejects_mutation_and_unknown_fields() {
    let valid = plan(vec![workload("scalar_integer", &[])]);
    valid.validate().unwrap();

    let mut mutated = valid.clone();
    mutated.workloads[0].work.operations_per_iteration += 1;
    assert!(mutated.validate().is_err());

    let mut document = serde_json::to_value(&valid).unwrap();
    document
        .as_object_mut()
        .unwrap()
        .insert("unknown".to_string(), Value::Bool(true));
    assert!(serde_json::from_value::<HardwareCalibrationPlan>(document).is_err());
}

#[test]
fn cpu_calibration_executes_every_declared_cpu_operation_sequentially() {
    let mut operations = vec![
        ("scalar_integer", vec![]),
        ("scalar_floating_point", vec![]),
        ("out_of_order_control_flow", vec![]),
        (
            "branch_dispatch",
            vec![("predictability", "data_dependent")],
        ),
        ("blocked_matrix_multiply", vec![("format", "f32")]),
        ("bit_population_mix", vec![]),
        ("sequential_read", vec![("working_set_bytes", "4096")]),
        ("sequential_copy", vec![("working_set_bytes", "4096")]),
        ("strided_read", vec![("working_set_bytes", "4096")]),
        ("pointer_chase", vec![("working_set_bytes", "4096")]),
        ("gather_scatter", vec![("working_set_bytes", "4096")]),
        (
            "generated_code_dispatch",
            vec![("instruction_footprint", "small")],
        ),
        (
            "generated_code_dispatch",
            vec![("instruction_footprint", "large")],
        ),
        ("atomic_fetch_add", vec![("contention", "shared")]),
        ("atomic_fetch_add", vec![("contention", "independent")]),
        ("numa_local_copy", vec![("working_set_bytes", "4096")]),
    ];
    operations.extend(
        [
            "bf16", "f32", "f64", "i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64",
        ]
        .into_iter()
        .map(|format| {
            (
                "vector_fused_arithmetic",
                vec![("format", format), ("vector_width_bits", "512")],
            )
        }),
    );
    let plan = plan(
        operations
            .into_iter()
            .map(|(operation, regime)| workload(operation, &regime))
            .collect(),
    );
    let run = run_calibration_plan(&plan, &CalibrationRunnerOptions::default()).unwrap();

    assert_eq!(run.status, CalibrationRunStatus::Completed);
    assert_eq!(run.workloads.len(), plan.workloads.len());
    assert!(run.workloads.iter().all(|result| {
        result.status == CalibrationRunStatus::Completed
            && result.validation.status == CalibrationValidationStatus::Passed
            && result.samples.len()
                == plan.policy.warmup_iterations
                    + plan.policy.steady_iterations
                    + plan.policy.sustained_window_count
    }));
    run.validate().unwrap();
}

#[test]
fn cancelled_calibration_does_not_claim_completion() {
    let plan = plan(vec![workload("scalar_integer", &[])]);
    let options = CalibrationRunnerOptions::default();
    options
        .cancelled
        .store(true, std::sync::atomic::Ordering::Relaxed);

    let run = run_calibration_plan(&plan, &options).unwrap();
    assert_eq!(run.status, CalibrationRunStatus::Cancelled);
    assert!(run.workloads.is_empty());
}

#[test]
fn workload_result_counters_round_trip_without_unknown_values() {
    let result = HardwareCalibrationWorkloadResult {
        workload_id: format!("calibration_workload_{}", "4".repeat(32)),
        status: CalibrationRunStatus::Completed,
        construction_duration_ns: 1,
        artifacts: Vec::new(),
        samples: vec![HardwareCalibrationSample {
            sample_index: 0,
            phase: CalibrationSamplePhase::Steady,
            duration_ns: 1,
            device_duration_ns: None,
            iterations: 1,
            window_index: None,
            thermal_millidegrees_celsius: None,
            valid: true,
        }],
        validation: CalibrationValidationResult {
            status: CalibrationValidationStatus::Passed,
            observed_digest: Some(format!(
                "nerve.calibration_output_sha256.v1:{}",
                "5".repeat(64)
            )),
            maximum_error_ppm: 0,
        },
        counters: BTreeMap::from([("logical_iterations".to_string(), 1)]),
        diagnostics: Vec::new(),
    };
    let encoded = serde_json::to_vec(&result).unwrap();
    let decoded = serde_json::from_slice::<HardwareCalibrationWorkloadResult>(&encoded).unwrap();
    assert_eq!(decoded, result);
}
