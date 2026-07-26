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
    refresh_workload_id(&mut workload);
    workload
}

fn refresh_workload_id(workload: &mut HardwareCalibrationWorkload) {
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
            .all(|sample| sample.valid && sample.device_duration_ns.is_some_and(|value| value > 0))
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
fn explicit_vulkan_fixed_graphics_calibration_uses_real_pipeline_stages_sequentially() {
    let Some(device_index) = std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
    else {
        eprintln!("skipping Vulkan graphics calibration: explicit idle AMD device is unset");
        return;
    };
    let workloads = [
        "rasterization",
        "fixed_function_interpolation",
        "depth_stencil",
        "blending",
    ]
    .into_iter()
    .map(|operation| {
        let mut graphics = workload(
            operation,
            &[
                ("render_target", "512x512"),
                ("format", "rgba16f"),
                ("overdraw", "4"),
            ],
        );
        graphics.executor = CalibrationExecutor::VulkanGraphics;
        graphics.work = CalibrationUsefulWork {
            items_per_iteration: 1_048_576,
            operations_per_iteration: 1_048_576,
            bytes_read_per_iteration: 4_194_304,
            bytes_written_per_iteration: 2_097_152,
        };
        graphics.artifacts = vec![
            CalibrationArtifactDeclaration {
                name: format!("{operation}_vertex"),
                kind: "spirv_vertex".to_string(),
                digest: None,
            },
            CalibrationArtifactDeclaration {
                name: format!("{operation}_fragment"),
                kind: "spirv_fragment".to_string(),
                digest: None,
            },
        ];
        graphics
            .artifacts
            .sort_by(|left, right| left.name.cmp(&right.name));
        graphics.workload_id = stable_hardware_id(
            "calibration_workload",
            &[
                serde_json::to_value(&graphics.process_names).unwrap(),
                serde_json::to_value(graphics.executor).unwrap(),
                Value::String(graphics.operation.clone()),
                serde_json::to_value(&graphics.regime).unwrap(),
                serde_json::to_value(&graphics.work).unwrap(),
                serde_json::to_value(&graphics.artifacts).unwrap(),
                serde_json::to_value(&graphics.validation).unwrap(),
            ],
        )
        .unwrap();
        graphics
    })
    .collect::<Vec<_>>();
    let calibration_plan = plan(workloads);
    let temporary = std::env::temp_dir().join(format!(
        "nerve-vulkan-fixed-graphics-test-{}",
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
    assert_eq!(result.workloads.len(), 4);
    assert!(result.workloads.iter().all(|workload| {
        workload.artifacts.len() == 2
            && workload.validation.status == CalibrationValidationStatus::Passed
            && workload
                .samples
                .iter()
                .all(|sample| sample.device_duration_ns.is_some_and(|value| value > 0))
    }));
    std::fs::remove_dir_all(temporary).unwrap();
}

#[cfg(feature = "vulkan")]
#[test]
fn explicit_vulkan_ray_calibration_builds_and_queries_real_acceleration_structures_sequentially() {
    let Some(device_index) = std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
    else {
        eprintln!("skipping Vulkan ray calibration: explicit idle AMD device is unset");
        return;
    };
    let workloads = [
        ("build_acceleration_structure", false),
        ("ray_query_traversal", true),
    ]
    .into_iter()
    .map(|(operation, query)| {
        let mut ray = workload(operation, &[("primitives", "4096"), ("rays", "65536")]);
        ray.executor = CalibrationExecutor::VulkanRay;
        ray.work = CalibrationUsefulWork {
            items_per_iteration: if query { 65_536 } else { 4_096 },
            operations_per_iteration: if query { 65_536 } else { 4_096 },
            bytes_read_per_iteration: 98_304,
            bytes_written_per_iteration: if query { 262_144 } else { 0 },
        };
        ray.artifacts = vec![CalibrationArtifactDeclaration {
            name: "ray_scene".to_string(),
            kind: "procedural_ray_scene".to_string(),
            digest: None,
        }];
        if query {
            ray.artifacts.push(CalibrationArtifactDeclaration {
                name: "ray_query_shader".to_string(),
                kind: "spirv_compute".to_string(),
                digest: None,
            });
        }
        ray.artifacts
            .sort_by(|left, right| left.name.cmp(&right.name));
        ray.workload_id = stable_hardware_id(
            "calibration_workload",
            &[
                serde_json::to_value(&ray.process_names).unwrap(),
                serde_json::to_value(ray.executor).unwrap(),
                Value::String(ray.operation.clone()),
                serde_json::to_value(&ray.regime).unwrap(),
                serde_json::to_value(&ray.work).unwrap(),
                serde_json::to_value(&ray.artifacts).unwrap(),
                serde_json::to_value(&ray.validation).unwrap(),
            ],
        )
        .unwrap();
        ray
    })
    .collect::<Vec<_>>();
    let calibration_plan = plan(workloads);
    let temporary = std::env::temp_dir().join(format!(
        "nerve-vulkan-ray-calibration-test-{}",
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
    assert_eq!(result.workloads.len(), 2);
    assert!(result.workloads.iter().all(|workload| {
        workload.validation.status == CalibrationValidationStatus::Passed
            && workload
                .samples
                .iter()
                .all(|sample| sample.device_duration_ns.is_some_and(|value| value > 0))
    }));
    std::fs::remove_dir_all(temporary).unwrap();
}

#[cfg(feature = "vulkan")]
#[test]
fn explicit_vulkan_video_calibration_encodes_and_decodes_real_av1_sequentially() {
    let Some(device_index) = std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
    else {
        eprintln!("skipping Vulkan video calibration: explicit idle AMD device is unset");
        return;
    };
    let workloads = ["video_encode", "video_decode"]
        .into_iter()
        .map(|operation| {
            let mut video = workload(
                operation,
                &[
                    ("codec", "av1"),
                    ("resolution", "640x360"),
                    ("frames", "10"),
                    ("timeout_ms", "30000"),
                ],
            );
            video.executor = CalibrationExecutor::VulkanVideo;
            video.work = CalibrationUsefulWork {
                items_per_iteration: 10,
                operations_per_iteration: 10,
                bytes_read_per_iteration: 3_456_000,
                bytes_written_per_iteration: 3_456_000,
            };
            video.artifacts = vec![
                CalibrationArtifactDeclaration {
                    name: "video_backend_manifest".to_string(),
                    kind: "external_backend_manifest".to_string(),
                    digest: None,
                },
                CalibrationArtifactDeclaration {
                    name: "video_bitstream".to_string(),
                    kind: "video_fixture_av1".to_string(),
                    digest: None,
                },
            ];
            video
                .artifacts
                .sort_by(|left, right| left.name.cmp(&right.name));
            video.workload_id = stable_hardware_id(
                "calibration_workload",
                &[
                    serde_json::to_value(&video.process_names).unwrap(),
                    serde_json::to_value(video.executor).unwrap(),
                    Value::String(video.operation.clone()),
                    serde_json::to_value(&video.regime).unwrap(),
                    serde_json::to_value(&video.work).unwrap(),
                    serde_json::to_value(&video.artifacts).unwrap(),
                    serde_json::to_value(&video.validation).unwrap(),
                ],
            )
            .unwrap();
            video
        })
        .collect::<Vec<_>>();
    let calibration_plan = plan(workloads);
    let temporary = std::env::temp_dir().join(format!(
        "nerve-vulkan-video-calibration-test-{}",
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
    assert_eq!(result.workloads.len(), 2);
    assert!(result.workloads.iter().all(|workload| {
        workload.artifacts.len() == 2
            && workload.validation.status == CalibrationValidationStatus::Passed
            && workload
                .samples
                .iter()
                .all(|sample| sample.valid && sample.iterations > 0)
    }));
    std::fs::remove_dir_all(temporary).unwrap();
}

#[cfg(feature = "vulkan")]
#[test]
fn explicit_vulkan_device_generated_commands_execute_real_generated_dispatches_sequentially() {
    let Some(device_index) = std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
    else {
        eprintln!("skipping Vulkan DGC calibration: explicit idle AMD device is unset");
        return;
    };
    let mut dgc = workload(
        "device_generated_commands",
        &[("dispatch_count", "16"), ("command_reuse", "resident")],
    );
    dgc.executor = CalibrationExecutor::VulkanDgc;
    dgc.work = CalibrationUsefulWork {
        items_per_iteration: 16,
        operations_per_iteration: 1024,
        bytes_read_per_iteration: 192,
        bytes_written_per_iteration: 4096,
    };
    dgc.artifacts = vec![CalibrationArtifactDeclaration {
        name: "device_generated_commands_shader".to_string(),
        kind: "spirv_compute".to_string(),
        digest: None,
    }];
    dgc.workload_id = stable_hardware_id(
        "calibration_workload",
        &[
            serde_json::to_value(&dgc.process_names).unwrap(),
            serde_json::to_value(dgc.executor).unwrap(),
            Value::String(dgc.operation.clone()),
            serde_json::to_value(&dgc.regime).unwrap(),
            serde_json::to_value(&dgc.work).unwrap(),
            serde_json::to_value(&dgc.artifacts).unwrap(),
            serde_json::to_value(&dgc.validation).unwrap(),
        ],
    )
    .unwrap();
    let calibration_plan = plan(vec![dgc]);
    let temporary = std::env::temp_dir().join(format!(
        "nerve-vulkan-dgc-calibration-test-{}",
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
            .samples
            .iter()
            .all(|sample| sample.device_duration_ns.is_some_and(|value| value > 0))
    );
    std::fs::remove_dir_all(temporary).unwrap();
}

#[cfg(feature = "vulkan")]
#[test]
fn explicit_vulkan_synchronization_calibration_uses_real_primitives_sequentially() {
    let Some(device_index) = std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
    else {
        eprintln!("skipping Vulkan synchronization calibration: explicit idle AMD device is unset");
        return;
    };
    let mut workloads = ["pipeline_barrier", "fence", "timeline_semaphore"]
        .into_iter()
        .map(|primitive| {
            let mut synchronization = workload(
                "synchronization_round_trip",
                &[("primitive", primitive), ("round_trips", "16")],
            );
            synchronization.executor = CalibrationExecutor::VulkanSynchronization;
            synchronization.work = CalibrationUsefulWork {
                items_per_iteration: 16,
                operations_per_iteration: 16,
                bytes_read_per_iteration: 64,
                bytes_written_per_iteration: 64,
            };
            synchronization.artifacts.clear();
            refresh_workload_id(&mut synchronization);
            synchronization
        })
        .collect::<Vec<_>>();
    let mut queue_contention = workload(
        "queue_contention",
        &[("queue_count", "1"), ("streams", "2")],
    );
    queue_contention.executor = CalibrationExecutor::VulkanSynchronization;
    queue_contention.work = CalibrationUsefulWork {
        items_per_iteration: 2,
        operations_per_iteration: 2,
        bytes_read_per_iteration: 8_388_608,
        bytes_written_per_iteration: 8_388_608,
    };
    queue_contention.artifacts.clear();
    refresh_workload_id(&mut queue_contention);
    workloads.push(queue_contention);
    let result = run_calibration_plan(
        &plan(workloads),
        &CalibrationRunnerOptions {
            vulkan_physical_device_index: Some(device_index),
            ..CalibrationRunnerOptions::default()
        },
    )
    .unwrap();
    assert_eq!(result.status, CalibrationRunStatus::Completed);
    assert_eq!(result.workloads.len(), 4);
    assert!(result.workloads.iter().all(|workload| {
        workload.validation.status == CalibrationValidationStatus::Passed
            && workload.samples.iter().all(|sample| {
                sample
                    .device_duration_ns
                    .is_some_and(|duration| duration > 0)
            })
    }));
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
        ("sparse_compaction", "density_ppm", "125000"),
        ("bitfield_mix", "format", "u32"),
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
    ];
    let workloads = cases
        .into_iter()
        .enumerate()
        .map(|(index, (operation, regime_name, regime_value))| {
            let mut candidate = workload(operation, &[(regime_name, regime_value)]);
            candidate.executor = CalibrationExecutor::VulkanCompute;
            let items = if operation == "cooperative_matrix_multiply" {
                64
            } else if matches!(operation, "command_queues" | "resident_command_replay") {
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
        ("sparse_compaction", "density_ppm", "125000"),
        ("bitfield_mix", "format", "u32"),
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
        ("indirect_work_generation", "dispatch_count", "1"),
        ("resident_command_replay", "dispatch_count", "1"),
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

#[cfg(feature = "vulkan")]
#[test]
fn ray_query_calibration_shader_compiles_for_vulkan_1_4() {
    let temporary = std::env::temp_dir().join(format!(
        "nerve-ray-query-shader-contract-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&temporary).unwrap();
    let source_path = temporary.join("ray_query.comp");
    let spirv_path = temporary.join("ray_query.spv");
    std::fs::write(&source_path, super::vulkan_specialized::ray_query_shader()).unwrap();
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
        "ray-query shader did not compile:\n{}{}",
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(std::fs::metadata(&spirv_path).unwrap().len() > 20);
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
            minimum_warmup_samples: 2,
            maximum_warmup_samples: 4,
            warmup_stability_window_samples: 1,
            minimum_warmup_duration_ns: 1,
            maximum_warmup_relative_shift_ppm: 20_000,
            minimum_steady_samples: 5,
            maximum_steady_samples: 25,
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
            "binary_tree_lookup",
            vec![("entries", "64"), ("queries", "64")],
        ),
        ("hash_lookup", vec![("entries", "64"), ("queries", "64")]),
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
        let warmup_count = result
            .samples
            .iter()
            .filter(|sample| sample.phase == CalibrationSamplePhase::Warmup)
            .count();
        let steady_count = result
            .samples
            .iter()
            .filter(|sample| sample.phase == CalibrationSamplePhase::Steady)
            .count();
        let sustained_count = result
            .samples
            .iter()
            .filter(|sample| sample.phase == CalibrationSamplePhase::Sustained)
            .count();
        result.status == CalibrationRunStatus::Completed
            && result.validation.status == CalibrationValidationStatus::Passed
            && (plan.policy.minimum_warmup_samples..=plan.policy.maximum_warmup_samples)
                .contains(&warmup_count)
            && (plan.policy.minimum_steady_samples..=plan.policy.maximum_steady_samples)
                .contains(&steady_count)
            && sustained_count == plan.policy.sustained_window_count
    }));
    run.validate().unwrap();
}

#[test]
fn cpu_lookup_construction_persists_real_index_artifacts_sequentially() {
    let workloads = [
        ("binary_tree_lookup", "eytzinger_index"),
        ("hash_lookup", "open_address_hash_index"),
    ]
    .into_iter()
    .map(|(operation, kind)| {
        let mut candidate = workload(operation, &[("entries", "4096"), ("queries", "4096")]);
        candidate.work = CalibrationUsefulWork {
            items_per_iteration: 4_096,
            operations_per_iteration: 81_920,
            bytes_read_per_iteration: 32_768,
            bytes_written_per_iteration: 8,
        };
        candidate.artifacts = vec![CalibrationArtifactDeclaration {
            name: operation.to_string(),
            kind: kind.to_string(),
            digest: None,
        }];
        refresh_workload_id(&mut candidate);
        candidate
    })
    .collect::<Vec<_>>();
    let temporary = std::env::temp_dir().join(format!(
        "nerve-cpu-index-calibration-test-{}",
        std::process::id()
    ));
    let run = run_calibration_plan(
        &plan(workloads),
        &CalibrationRunnerOptions {
            artifact_directory: temporary.clone(),
            ..CalibrationRunnerOptions::default()
        },
    )
    .unwrap();
    assert_eq!(run.status, CalibrationRunStatus::Completed);
    assert!(run.workloads.iter().all(|result| {
        result.validation.status == CalibrationValidationStatus::Passed
            && result.artifacts.len() == 1
            && result.artifacts[0].byte_length > 16_384
            && temporary.join(&result.artifacts[0].relative_path).is_file()
    }));
    std::fs::remove_dir_all(temporary).unwrap();
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
fn failed_workload_validation_fails_the_complete_run() {
    let mut candidate = workload("scalar_integer", &[]);
    candidate.validation.expected_digest = Some(format!(
        "nerve.calibration_output_sha256.v1:{}",
        "0".repeat(64)
    ));
    refresh_workload_id(&mut candidate);
    let run =
        run_calibration_plan(&plan(vec![candidate]), &CalibrationRunnerOptions::default()).unwrap();
    assert_eq!(run.status, CalibrationRunStatus::Failed);
    assert_eq!(run.workloads[0].status, CalibrationRunStatus::Failed);
    assert!(!run.diagnostics.is_empty());
    run.validate().unwrap();
}

#[test]
fn default_calibration_artifact_directories_are_collision_free() {
    let first = CalibrationRunnerOptions::default();
    let second = CalibrationRunnerOptions::default();
    assert_ne!(first.artifact_directory, second.artifact_directory);
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
