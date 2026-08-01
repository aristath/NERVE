use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use super::*;
use crate::stream_circuit::{
    ResolvedLoweredExecutionGraph, StreamCircuitPlacementSpec, StreamCircuitRuntimeGraph,
};
use crate::stream_plan::{StreamCircuitExecutionPlan, StreamCircuitResourcePlan};
use crate::test_support::{
    tiny_model_dir, tiny_model_lowered_graph_path, tiny_model_package_manifest_path,
    tiny_model_tensor_index_path,
};

const FIXTURE_MODEL_GREEDY_SAMPLER_COMPONENT_ID: &str = "greedy_sampler";
const FIXTURE_MODEL_EMBED_TOKENS_TENSOR: &str = "model.embed_tokens.weight";
const FIXTURE_MODEL_INPUT_FRAME_SIGNAL: &str = "input_frame";
const FIXTURE_MODEL_HIDDEN_SIZE: usize = 16;

#[test]
fn package_loader_rejects_unsupported_package_schema_before_package_setup() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let manifest_path = std::env::temp_dir().join(format!(
        "nerve-stale-compiler-contract-{}-{unique}.json",
        std::process::id()
    ));

    std::fs::write(
        &manifest_path,
        serde_json::to_vec(&serde_json::json!({
            "schema": "nerve.vulkan_resident_model_package.v2"
        }))
        .unwrap(),
    )
    .unwrap();
    let schema_error = VulkanResidentModelPackageManifest::from_json_file(&manifest_path)
        .unwrap_err()
        .to_string();
    assert!(schema_error.contains("recompile the model"));

    std::fs::remove_file(manifest_path).unwrap();
}

#[test]
fn loaded_artifact_manifest_preserves_compiled_launch_geometry() {
    let loaded = VulkanLoadedReusableKernelArtifactManifest {
        schema: VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA.to_string(),
        backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
        artifacts: vec![VulkanLoadedReusableKernelArtifact {
            artifact: VulkanReusableKernelArtifact {
                family_id: "sparse-moe-gate-up".to_string(),
                op: "sparse_moe_gate_up".to_string(),
                path: "kernels/sparse-moe-gate-up.spv".to_string(),
                entry_point: DEFAULT_SPIRV_ENTRY_POINT.to_string(),
                local_size_x: 64,
                workgroup_count_x: 2_048,
                descriptor_signature: Vec::new(),
                push_constants: Vec::new(),
                uses_stream_tick: false,
            },
            resolved_path: PathBuf::from("kernels/sparse-moe-gate-up.spv"),
            words: vec![0x0723_0203],
        }],
        total_word_count: 1,
    };

    let physical = loaded.artifact_manifest();

    assert_eq!(physical.artifacts.len(), 1);
    assert_eq!(physical.artifacts[0].workgroup_count_x, 2_048);
    assert_eq!(physical.artifacts[0].local_size_x, 64);
}

#[test]
fn component_batch_signal_liveness_reuses_only_compatible_dead_buffers() {
    let key = |signal_id: &str| VulkanComponentBatchSignalKey::Activation {
        component_id: "component".to_string(),
        signal_id: signal_id.to_string(),
    };
    let lifetimes = vec![
        VulkanComponentBatchSignalLifetime {
            key: key("first"),
            frame_byte_capacity: 4_096,
            host_visible: false,
            first_dispatch: 0,
            last_dispatch: 2,
        },
        VulkanComponentBatchSignalLifetime {
            key: key("overlapping"),
            frame_byte_capacity: 4_096,
            host_visible: false,
            first_dispatch: 2,
            last_dispatch: 3,
        },
        VulkanComponentBatchSignalLifetime {
            key: key("reusable"),
            frame_byte_capacity: 4_096,
            host_visible: false,
            first_dispatch: 3,
            last_dispatch: 4,
        },
        VulkanComponentBatchSignalLifetime {
            key: key("different_size"),
            frame_byte_capacity: 8_192,
            host_visible: false,
            first_dispatch: 5,
            last_dispatch: 6,
        },
        VulkanComponentBatchSignalLifetime {
            key: VulkanComponentBatchSignalKey::IncomingEdge(7),
            frame_byte_capacity: 4_096,
            host_visible: true,
            first_dispatch: 5,
            last_dispatch: 6,
        },
    ];

    let (indices, buffers) = allocate_component_batch_signal_lifetimes(lifetimes);

    assert_eq!(buffers.len(), 4);
    assert_ne!(indices[&key("first")], indices[&key("overlapping")]);
    assert_eq!(indices[&key("first")], indices[&key("reusable")]);
    assert_ne!(indices[&key("first")], indices[&key("different_size")]);
    assert_ne!(
        indices[&key("first")],
        indices[&VulkanComponentBatchSignalKey::IncomingEdge(7)]
    );
}

#[test]
fn component_batch_execution_uses_standalone_component_submissions() {
    let span = |component_id: &str,
                dispatch_index: usize,
                step_start: usize,
                step_end: usize,
                distributed: bool| VulkanComponentBatchDispatchSpan {
        component_id: component_id.to_string(),
        dispatch_index,
        step_start,
        step_end,
        distributed,
    };
    let spans = vec![
        span("layer_00", 0, 0, 2, false),
        span("layer_00", 1, 2, 5, false),
        span("layer_01", 2, 5, 7, false),
        span("layer_01", 3, 7, 7, true),
        span("layer_01", 4, 7, 10, false),
        span("layer_02", 5, 10, 12, false),
    ];

    assert_eq!(
        component_batch_execution_units(&spans).unwrap(),
        vec![
            VulkanComponentBatchExecutionUnit::LocalComponent {
                component_id: "layer_00".to_string(),
                step_start: 0,
                step_end: 5,
            },
            VulkanComponentBatchExecutionUnit::LocalComponent {
                component_id: "layer_01".to_string(),
                step_start: 5,
                step_end: 7,
            },
            VulkanComponentBatchExecutionUnit::DistributedDispatch { dispatch_index: 3 },
            VulkanComponentBatchExecutionUnit::LocalComponent {
                component_id: "layer_01".to_string(),
                step_start: 7,
                step_end: 10,
            },
            VulkanComponentBatchExecutionUnit::LocalComponent {
                component_id: "layer_02".to_string(),
                step_start: 10,
                step_end: 12,
            },
        ]
    );
}

#[test]
fn component_batch_execution_does_not_create_empty_local_submissions() {
    let spans = vec![
        VulkanComponentBatchDispatchSpan {
            component_id: "layer_00".to_string(),
            dispatch_index: 0,
            step_start: 0,
            step_end: 0,
            distributed: true,
        },
        VulkanComponentBatchDispatchSpan {
            component_id: "layer_01".to_string(),
            dispatch_index: 1,
            step_start: 0,
            step_end: 0,
            distributed: true,
        },
    ];

    assert_eq!(
        component_batch_execution_units(&spans).unwrap(),
        vec![
            VulkanComponentBatchExecutionUnit::DistributedDispatch { dispatch_index: 0 },
            VulkanComponentBatchExecutionUnit::DistributedDispatch { dispatch_index: 1 },
        ]
    );
}

#[test]
fn component_batch_execution_submits_only_distributed_group_leaders() {
    let spans = (0..3)
        .map(|dispatch_index| VulkanComponentBatchDispatchSpan {
            component_id: "layer_00".to_string(),
            dispatch_index,
            step_start: 0,
            step_end: 0,
            distributed: true,
        })
        .collect::<Vec<_>>();

    assert_eq!(
        component_batch_execution_units_for_distributed_groups(&spans, &BTreeSet::from([0, 2]),)
            .unwrap(),
        vec![
            VulkanComponentBatchExecutionUnit::DistributedDispatch { dispatch_index: 0 },
            VulkanComponentBatchExecutionUnit::DistributedDispatch { dispatch_index: 2 },
        ]
    );
}

#[test]
fn component_batch_demand_execution_keeps_contiguous_local_units_in_one_range() {
    let local = |component_id: &str, step_start: usize, step_end: usize| {
        VulkanComponentBatchExecutionUnit::LocalComponent {
            component_id: component_id.to_string(),
            step_start,
            step_end,
        }
    };
    let units = vec![
        local("embedding", 0, 2),
        local("layer_00", 2, 7),
        local("layer_01", 7, 12),
        VulkanComponentBatchExecutionUnit::DistributedDispatch { dispatch_index: 9 },
        local("layer_02", 12, 17),
        local("layer_03", 17, 22),
        VulkanComponentBatchExecutionUnit::DistributedDispatch { dispatch_index: 21 },
    ];

    assert_eq!(
        component_batch_local_execution_unit_ranges(&units),
        vec![(0, 3), (4, 6)]
    );
}

#[test]
fn demand_batch_replay_rejects_dynamic_push_constants_in_its_remaining_commands() {
    let commands = vec![
        VulkanDemandResidencyBatchCommand::Step(0),
        VulkanDemandResidencyBatchCommand::Gate(0),
        VulkanDemandResidencyBatchCommand::Step(1),
        VulkanDemandResidencyBatchCommand::Gate(1),
        VulkanDemandResidencyBatchCommand::Step(2),
    ];

    assert!(!demand_batch_commands_are_replay_stable(
        &commands,
        0,
        |step_index| step_index != 1,
    ));
    assert!(!demand_batch_commands_are_replay_stable(
        &commands,
        1,
        |step_index| step_index != 1,
    ));
    assert!(demand_batch_commands_are_replay_stable(
        &commands,
        3,
        |step_index| step_index != 1,
    ));
    assert!(!demand_batch_commands_are_replay_stable(
        &commands,
        commands.len() + 1,
        |_| true,
    ));
}

#[test]
fn component_batch_execution_rejects_noncontiguous_dispatch_steps() {
    let spans = vec![
        VulkanComponentBatchDispatchSpan {
            component_id: "layer_00".to_string(),
            dispatch_index: 0,
            step_start: 0,
            step_end: 2,
            distributed: false,
        },
        VulkanComponentBatchDispatchSpan {
            component_id: "layer_00".to_string(),
            dispatch_index: 1,
            step_start: 3,
            step_end: 4,
            distributed: false,
        },
    ];

    let error = component_batch_execution_units(&spans).unwrap_err();
    assert!(error.to_string().contains("starts at step 3, expected 2"));
}

#[test]
fn component_batch_execution_marks_only_mutating_state_descriptors_as_commits() {
    assert!(!component_batch_descriptors_commit_state(
        [
            VulkanKernelDescriptorUsage::InputSignal,
            VulkanKernelDescriptorUsage::StateRead,
            VulkanKernelDescriptorUsage::Parameter,
        ]
        .iter()
    ));
    assert!(component_batch_descriptors_commit_state(
        [VulkanKernelDescriptorUsage::StateWrite].iter()
    ));
    assert!(component_batch_descriptors_commit_state(
        [VulkanKernelDescriptorUsage::StateView].iter()
    ));
}

const FIXTURE_MODEL_FRAME_BYTES: usize = FIXTURE_MODEL_HIDDEN_SIZE * 2;
const FIXTURE_MODEL_LOGITS_BYTES: usize = 32 * 4;
const FIXTURE_MODEL_SAMPLER_OUTPUT_BYTES: usize = 16;
const FIXTURE_MODEL_EMBED_TOKENS_BYTES: usize = 32 * FIXTURE_MODEL_FRAME_BYTES;

#[test]
fn speculative_verification_commits_through_the_first_mismatch() {
    let result = verify_speculative_token_prefix(&[11, 12, 13], &[11, 99, 88, 77]).unwrap();

    assert_eq!(result.accepted_draft_count, 1);
    assert_eq!(result.committed_target_tick_count, 2);
    assert_eq!(result.emitted_token_ids, [11, 99]);
}

#[test]
fn speculative_verification_emits_the_bonus_token_when_all_drafts_match() {
    let result = verify_speculative_token_prefix(&[11, 12], &[11, 12, 13]).unwrap();

    assert_eq!(result.accepted_draft_count, 2);
    assert_eq!(result.committed_target_tick_count, 3);
    assert_eq!(result.emitted_token_ids, [11, 12, 13]);
}

#[test]
fn speculative_verification_rejects_incomplete_target_results() {
    let error = verify_speculative_token_prefix(&[11, 12], &[11, 12]).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("2 draft tokens but 2 target predictions; expected 3")
    );
}

#[test]
fn speculative_verification_stops_at_the_first_emitted_stop_token() {
    let mut result = verify_speculative_token_prefix(&[11, 12], &[11, 12, 99]).unwrap();

    truncate_speculative_verification_at_stop(&mut result, &BTreeSet::from([11]));

    assert_eq!(result.accepted_draft_count, 1);
    assert_eq!(result.committed_target_tick_count, 1);
    assert_eq!(result.emitted_token_ids, [11]);
}

#[test]
fn speculative_decode_stats_report_rollbacks_and_total_cost() {
    let mut stats = VulkanSpeculativeDecodeStats::default();
    stats.record_cycle(&VulkanSpeculativeCycleRun {
        decoder_id: "draft".to_string(),
        initial_token_id: 1,
        start_stream_tick: 0,
        draft_token_ids: vec![2, 3],
        target_token_ids: vec![2, 9, 4],
        verification: VulkanSpeculativeVerificationResult {
            accepted_draft_count: 1,
            committed_target_tick_count: 2,
            emitted_token_ids: vec![2, 9],
        },
        draft_time_ns: 20,
        target_verification_time_ns: 60,
        draft_catch_up_time_ns: 10,
        total_time_ns: 100,
    });

    assert_eq!(stats.cycle_count, 1);
    assert_eq!(stats.rollback_cycle_count, 1);
    assert_eq!(stats.proposed_draft_token_count, 2);
    assert_eq!(stats.accepted_draft_token_count, 1);
    assert_eq!(stats.emitted_token_count, 2);
    assert_eq!(stats.total_time_ns, 100);
}

#[test]
fn component_batches_select_only_mode_compatible_kernels() {
    let artifact = |lane_tile_width,
                    independent_candidate_compatible,
                    causal_sequence_compatible| VulkanResidentComponentBatchKernelArtifact {
            component_id: "processor".to_string(),
            node_id: "project".to_string(),
            execution_domain: VulkanResidentComponentKernelExecutionDomain::DecodeAndPrefill,
            batch_mode: VulkanResidentComponentKernelBatchMode::WeightShared,
            lane_tile_width,
            independent_candidate_compatible,
            causal_sequence_compatible,
            parallel_block_compatible: false,
            device_requirements: VulkanResidentVulkanDeviceRequirements::default(),
            stages: Vec::new(),
        };
    let artifacts = vec![
        artifact(64, false, true),
        artifact(2, true, true),
        artifact(4, true, true),
        artifact(8, true, true),
        artifact(16, true, true),
    ];

    let independent_streams = select_component_batch_kernel_artifact(
        &artifacts,
        "processor",
        "project",
        VulkanComponentBatchExecutionMode::IndependentStreams,
        6,
    )
    .unwrap();
    assert_eq!(independent_streams.lane_tile_width, 8);
    assert!(independent_streams.independent_candidate_compatible);

    let causal = select_component_batch_kernel_artifact(
        &artifacts,
        "processor",
        "project",
        VulkanComponentBatchExecutionMode::CausalSequence,
        6,
    )
    .unwrap();
    assert_eq!(causal.lane_tile_width, 8);
    assert!(causal.causal_sequence_compatible);

    let full_prefill = select_component_batch_kernel_artifact(
        &artifacts,
        "processor",
        "project",
        VulkanComponentBatchExecutionMode::CausalSequence,
        64,
    )
    .unwrap();
    assert_eq!(full_prefill.lane_tile_width, 64);

    let heterogeneous = select_component_batch_kernel_artifact_where(
        &artifacts,
        "processor",
        "project",
        VulkanComponentBatchExecutionMode::CausalSequence,
        6,
        |artifact| artifact.lane_tile_width != 64,
    )
    .unwrap();
    assert_eq!(heterogeneous.lane_tile_width, 8);

    let parallel_only = VulkanResidentComponentBatchKernelArtifact {
        component_id: "processor".to_string(),
        node_id: "attend".to_string(),
        execution_domain: VulkanResidentComponentKernelExecutionDomain::DecodeAndPrefill,
        batch_mode: VulkanResidentComponentKernelBatchMode::WeightShared,
        lane_tile_width: 64,
        independent_candidate_compatible: false,
        causal_sequence_compatible: false,
        parallel_block_compatible: true,
        device_requirements: VulkanResidentVulkanDeviceRequirements::default(),
        stages: Vec::new(),
    };
    assert!(
        select_component_batch_kernel_artifact(
            &[parallel_only],
            "processor",
            "attend",
            VulkanComponentBatchExecutionMode::ParallelBlock,
            7,
        )
        .is_some()
    );
    assert!(select_component_batch_kernel_artifact(
        &artifacts,
        "processor",
        "project",
        VulkanComponentBatchExecutionMode::ParallelBlock,
        7,
    )
    .is_none());
    let mut explicit_parallel = artifact(8, true, true);
    explicit_parallel.parallel_block_compatible = true;
    assert_eq!(
        select_component_batch_kernel_artifact(
            &[explicit_parallel],
            "processor",
            "project",
            VulkanComponentBatchExecutionMode::ParallelBlock,
            7,
        )
        .unwrap()
        .lane_tile_width,
        8
    );
}

#[test]
fn causal_component_batches_use_bounded_power_of_two_capacity_classes() {
    assert_eq!(causal_component_block_lane_capacity(1).unwrap(), 1);
    assert_eq!(causal_component_block_lane_capacity(2).unwrap(), 2);
    assert_eq!(causal_component_block_lane_capacity(3).unwrap(), 4);
    assert_eq!(causal_component_block_lane_capacity(4).unwrap(), 4);
    assert_eq!(causal_component_block_lane_capacity(5).unwrap(), 8);
    assert_eq!(
        causal_component_block_lane_capacity(VULKAN_BACKEND_LOOP_MAX_WINDOW).unwrap(),
        VULKAN_BACKEND_LOOP_MAX_WINDOW,
    );
    assert!(causal_component_block_lane_capacity(0).is_err());
    assert!(
        causal_component_block_lane_capacity(VULKAN_BACKEND_LOOP_MAX_WINDOW + 1).is_err()
    );
}

#[test]
fn component_batches_select_only_artifacts_for_the_requested_execution_domain() {
    let artifact = |execution_domain, lane_tile_width| VulkanResidentComponentBatchKernelArtifact {
        component_id: "processor".to_string(),
        node_id: "project".to_string(),
        execution_domain,
        batch_mode: VulkanResidentComponentKernelBatchMode::WeightShared,
        lane_tile_width,
        independent_candidate_compatible: true,
        causal_sequence_compatible: true,
        parallel_block_compatible: false,
        device_requirements: VulkanResidentVulkanDeviceRequirements::default(),
        stages: Vec::new(),
    };
    let artifacts = vec![
        artifact(VulkanResidentComponentKernelExecutionDomain::Prefill, 4),
        artifact(VulkanResidentComponentKernelExecutionDomain::Decode, 8),
        artifact(
            VulkanResidentComponentKernelExecutionDomain::DecodeAndPrefill,
            16,
        ),
    ];

    let decode = select_component_batch_kernel_artifact(
        &artifacts,
        "processor",
        "project",
        VulkanComponentBatchExecutionMode::IndependentStreams,
        4,
    )
    .unwrap();
    assert_eq!(
        decode.execution_domain,
        VulkanResidentComponentKernelExecutionDomain::Decode
    );
    assert_eq!(decode.lane_tile_width, 8);

    let prefill = select_component_batch_kernel_artifact(
        &artifacts,
        "processor",
        "project",
        VulkanComponentBatchExecutionMode::CausalSequence,
        4,
    )
    .unwrap();
    assert_eq!(
        prefill.execution_domain,
        VulkanResidentComponentKernelExecutionDomain::Prefill
    );
    assert_eq!(prefill.lane_tile_width, 4);
}

#[test]
fn component_batches_use_causal_compatibility_for_temporal_prefill_kernels() {
    let artifacts = vec![VulkanResidentComponentBatchKernelArtifact {
        component_id: "processor".to_string(),
        node_id: "attention".to_string(),
        execution_domain: VulkanResidentComponentKernelExecutionDomain::Prefill,
        batch_mode: VulkanResidentComponentKernelBatchMode::CausalScan,
        lane_tile_width: 64,
        independent_candidate_compatible: false,
        causal_sequence_compatible: true,
        parallel_block_compatible: false,
        device_requirements: VulkanResidentVulkanDeviceRequirements::default(),
        stages: Vec::new(),
    }];

    assert!(
        select_component_batch_kernel_artifact(
            &artifacts,
            "processor",
            "attention",
            VulkanComponentBatchExecutionMode::IndependentStreams,
            4,
        )
        .is_none()
    );
    let causal = select_component_batch_kernel_artifact(
        &artifacts,
        "processor",
        "attention",
        VulkanComponentBatchExecutionMode::CausalSequence,
        4,
    )
    .unwrap();
    assert!(causal.causal_sequence_compatible);
    assert!(!causal.independent_candidate_compatible);
}

#[test]
fn component_batch_execution_contract_requires_matching_shader_mode() {
    let execution = |batch_mode, batch_shader_path: Option<String>| {
        let batch_implementations = batch_shader_path
            .into_iter()
            .map(|shader_path| VulkanResidentComponentBatchImplementationSpec {
                execution_domain: VulkanResidentComponentKernelExecutionDomain::DecodeAndPrefill,
                lane_tile_width: 16,
                independent_candidate_compatible: true,
                causal_sequence_compatible: true,
                parallel_block_compatible: false,
                device_requirements: VulkanResidentVulkanDeviceRequirements::default(),
                stages: vec![VulkanResidentComponentBatchStageSpec {
                    shader_path,
                    local_size_x: 64,
                    workgroup_count_x: 1,
                    descriptor_bindings: Vec::new(),
                    state_snapshot_binding: None,
                    control: VulkanResidentComponentBatchControlSpec::StorageBuffer {
                        byte_count: VULKAN_COMPONENT_BATCH_WIDTH_CONTROL_BYTE_CAPACITY,
                        binding: 31,
                        payload: VulkanResidentComponentBatchControlPayload::Width,
                        access: VulkanResidentComponentBatchControlAccess::Read,
                    },
                    indirect_dispatch_byte_offset: None,
                    dispatch_y_from_batch_width: false,
                }],
            })
            .collect();
        vec![VulkanResidentComponentExecutionSpec {
            component_id: "processor".to_string(),
            operator_type: "fixture".to_string(),
            implementation: "exact_reference".to_string(),
            kernels: vec![VulkanResidentComponentKernelSpec {
                execution_index: 0,
                node_id: "project".to_string(),
                op: "linear".to_string(),
                source_node_ids: vec!["project".to_string()],
                semantic_module_ids: vec!["layer.token_mixer.output_projection".to_string()],
                execution_domain: VulkanResidentComponentKernelExecutionDomain::Decode,
                shader_path: "shaders/project.spv".to_string(),
                local_size_x: 64,
                workgroup_count_x: 1,
                batch_mode,
                batch_implementations,
            }],
        }]
    };

    validate_component_executions(
        "fixture",
        &execution(VulkanResidentComponentKernelBatchMode::SerialLanes, None),
    )
    .unwrap();
    validate_component_executions(
        "fixture",
        &execution(
            VulkanResidentComponentKernelBatchMode::WeightShared,
            Some("shaders/project_batch.spv".to_string()),
        ),
    )
    .unwrap();
    validate_component_executions(
        "fixture",
        &execution(
            VulkanResidentComponentKernelBatchMode::CausalScan,
            Some("shaders/project_scan.spv".to_string()),
        ),
    )
    .unwrap();

    let serial_error = validate_component_executions(
        "fixture",
        &execution(
            VulkanResidentComponentKernelBatchMode::SerialLanes,
            Some("shaders/project_batch.spv".to_string()),
        ),
    )
    .unwrap_err();
    assert!(serial_error.to_string().contains("invalid SerialLanes"));

    let batch_error = validate_component_executions(
        "fixture",
        &execution(VulkanResidentComponentKernelBatchMode::WeightShared, None),
    )
    .unwrap_err();
    assert!(batch_error.to_string().contains("invalid WeightShared"));

    let mut invalid_control = execution(
        VulkanResidentComponentKernelBatchMode::WeightShared,
        Some("shaders/project_batch.spv".to_string()),
    );
    invalid_control[0].kernels[0].batch_implementations[0].stages[0].control =
        VulkanResidentComponentBatchControlSpec::StorageBuffer {
            byte_count: VULKAN_COMPONENT_BATCH_WIDTH_CONTROL_BYTE_CAPACITY,
            binding: 3,
            payload: VulkanResidentComponentBatchControlPayload::Temporal,
            access: VulkanResidentComponentBatchControlAccess::Read,
        };
    let control_error = validate_component_executions("fixture", &invalid_control).unwrap_err();
    assert!(control_error.to_string().contains("invalid WeightShared"));

    let mut invalid_descriptor_mapping = execution(
        VulkanResidentComponentKernelBatchMode::WeightShared,
        Some("shaders/project_batch.spv".to_string()),
    );
    invalid_descriptor_mapping[0].kernels[0].batch_implementations[0].stages[0]
        .descriptor_bindings = vec![
        VulkanResidentComponentBatchDescriptorBindingSpec {
            binding: 1,
            source_binding: 0,
        },
        VulkanResidentComponentBatchDescriptorBindingSpec {
            binding: 1,
            source_binding: 2,
        },
    ];
    let descriptor_error =
        validate_component_executions("fixture", &invalid_descriptor_mapping).unwrap_err();
    assert!(descriptor_error.to_string().contains("invalid WeightShared"));
}

#[test]
fn component_batch_control_preserves_temporal_position_and_capacity() {
    let bytes = component_batch_control_bytes(64, 0x1122_3344_5566_7788, 65_536);

    assert_eq!(&bytes[0..4], &64u32.to_le_bytes());
    assert_eq!(&bytes[4..12], &0x1122_3344_5566_7788u64.to_le_bytes());
    assert_eq!(&bytes[12..16], &65_536u32.to_le_bytes());
}

#[test]
fn component_batch_control_uses_typed_persistent_buffers_for_every_payload() {
    let width_only = VulkanResidentComponentBatchStageArtifact {
        shader_path: "shaders/linear_batch2_fp8_e4m3_b128x128_5120x5120.spv".to_string(),
        spirv_words: Vec::new(),
        local_size_x: 64,
        workgroup_count_x: 1,
        descriptor_bindings: Vec::new(),
        state_snapshot_binding: None,
        control: VulkanResidentComponentBatchControlSpec::StorageBuffer {
            byte_count: VULKAN_COMPONENT_BATCH_WIDTH_CONTROL_BYTE_CAPACITY,
            binding: 31,
            payload: VulkanResidentComponentBatchControlPayload::Width,
            access: VulkanResidentComponentBatchControlAccess::Read,
        },
        indirect_dispatch_byte_offset: None,
        dispatch_y_from_batch_width: false,
    };
    let temporal = VulkanResidentComponentBatchStageArtifact {
        shader_path: "shaders/append_kv_temporal_commit_bf16_kv8_d128_w0.spv".to_string(),
        spirv_words: Vec::new(),
        local_size_x: 64,
        workgroup_count_x: 1,
        descriptor_bindings: Vec::new(),
        state_snapshot_binding: None,
        control: VulkanResidentComponentBatchControlSpec::StorageBuffer {
            byte_count: VULKAN_COMPONENT_BATCH_CONTROL_BYTE_CAPACITY,
            binding: 7,
            payload: VulkanResidentComponentBatchControlPayload::Temporal,
            access: VulkanResidentComponentBatchControlAccess::Read,
        },
        indirect_dispatch_byte_offset: None,
        dispatch_y_from_batch_width: false,
    };
    let sparse = VulkanResidentComponentBatchStageArtifact {
        shader_path: "shaders/sparse_moe_gate_up_batch1_bf16.spv".to_string(),
        spirv_words: Vec::new(),
        local_size_x: 64,
        workgroup_count_x: 1,
        descriptor_bindings: Vec::new(),
        state_snapshot_binding: None,
        control: VulkanResidentComponentBatchControlSpec::StorageBuffer {
            byte_count: 2 * VULKAN_COMPONENT_BATCH_WIDTH_CONTROL_BYTE_CAPACITY,
            binding: 31,
            payload: VulkanResidentComponentBatchControlPayload::WidthExpertStart,
            access: VulkanResidentComponentBatchControlAccess::Read,
        },
        indirect_dispatch_byte_offset: None,
        dispatch_y_from_batch_width: false,
    };

    assert_eq!(
        width_only.control.storage_buffer(),
        (
            31,
            VULKAN_COMPONENT_BATCH_WIDTH_CONTROL_BYTE_CAPACITY,
            VulkanResidentComponentBatchControlPayload::Width,
        )
    );
    assert_eq!(
        temporal.control.storage_buffer(),
        (
            7,
            VULKAN_COMPONENT_BATCH_CONTROL_BYTE_CAPACITY,
            VulkanResidentComponentBatchControlPayload::Temporal,
        )
    );
    assert_eq!(
        sparse.control.storage_buffer(),
        (
            31,
            2 * VULKAN_COMPONENT_BATCH_WIDTH_CONTROL_BYTE_CAPACITY,
            VulkanResidentComponentBatchControlPayload::WidthExpertStart,
        )
    );
    let expert_start = [VulkanKernelScalarBinding {
        name: "expert_start".to_string(),
        scalar_type: "u32".to_string(),
        source: VulkanKernelScalarSource::PushConstant,
    }];
    assert!(component_batch_stages_replace_push_constants(
        std::slice::from_ref(&sparse),
        &expert_start,
    ));
    assert!(!component_batch_stages_replace_push_constants(
        std::slice::from_ref(&width_only),
        &expert_start,
    ));
    assert!(!component_batch_stages_replace_push_constants(
        std::slice::from_ref(&sparse),
        &[VulkanKernelScalarBinding {
            name: "model_specific_scalar".to_string(),
            scalar_type: "u32".to_string(),
            source: VulkanKernelScalarSource::PushConstant,
        }],
    ));

    let control = component_batch_control_bytes(64, 0x1122_3344_5566_7788, 65_536);
    assert_eq!(
        component_batch_control_payload_bytes(
            VulkanResidentComponentBatchControlPayload::Width,
            &control,
            false,
        ),
        64u32.to_le_bytes(),
    );
    assert_eq!(
        component_batch_control_payload_bytes(
            VulkanResidentComponentBatchControlPayload::WidthStateSnapshots,
            &control,
            true,
        ),
        [64u32.to_le_bytes(), 1u32.to_le_bytes()].concat(),
    );
    assert_eq!(
        component_batch_control_payload_bytes(
            VulkanResidentComponentBatchControlPayload::WidthExpertStart,
            &control,
            false,
        ),
        [64u32.to_le_bytes(), 0u32.to_le_bytes()].concat(),
    );
    assert_eq!(
        distributed_component_batch_control_payload_bytes(
            VulkanResidentComponentBatchControlPayload::WidthExpertStart,
            &control,
            128,
            64,
        ),
        [64u32.to_le_bytes(), 128u32.to_le_bytes()].concat(),
    );
    assert_eq!(
        distributed_component_batch_control_payload_bytes(
            VulkanResidentComponentBatchControlPayload::WidthExpertRangeIndirect,
            &control,
            128,
            64,
        ),
        [
            64u32.to_le_bytes(),
            128u32.to_le_bytes(),
            64u32.to_le_bytes(),
            0u32.to_le_bytes(),
            0u32.to_le_bytes(),
            0u32.to_le_bytes(),
            0u32.to_le_bytes(),
        ]
        .concat(),
    );
    assert_eq!(
        component_batch_control_payload_bytes(
            VulkanResidentComponentBatchControlPayload::Temporal,
            &control,
            false,
        ),
        control,
    );
}

#[test]
fn distributed_batch_group_retains_every_members_control_buffer_set() {
    let shard = |expert_start| VulkanDistributedComponentBatchShardRunner {
        device_id: "gpu0".to_string(),
        dispatches: Vec::new(),
        expert_start,
        expert_count: 128,
        batch_control_buffer_sets: vec![BTreeMap::new()],
        sequence_catalog: RefCell::new(BTreeMap::new()),
    };
    let mut grouped = shard(128);

    grouped.append_group_member(shard(128)).unwrap();

    assert_eq!(grouped.batch_control_buffer_sets.len(), 2);
    let error = grouped.append_group_member(shard(0)).unwrap_err();
    assert!(error.to_string().contains("changes expert start"));
}

#[test]
fn distributed_batch_keeps_group_internal_activations_private_to_each_shard() {
    use crate::vulkan_distributed::{
        VulkanDistributedActivationRange, VulkanDistributedDispatchPlan,
        VulkanDistributedDispatchShard,
    };

    let activation = |binding, signal_id: &str, slot| VulkanDistributedActivationSlot {
        binding,
        component_id: "layer".to_string(),
        signal_id: signal_id.to_string(),
        slot,
        byte_capacity: 8_224,
        signal_byte_capacity: 8_224,
        storage: VulkanDistributedActivationStorage::ActivationSlot,
    };
    let shards = || {
        vec![
            VulkanDistributedDispatchShard {
                device_id: "gpu0".to_string(),
                row_start: 0,
                row_count: 128,
                workgroup_count_x: 8,
                base_workgroup_z: 0,
                input_range: VulkanDistributedActivationRange {
                    byte_offset: 0,
                    byte_count: 8_224,
                },
                auxiliary_input_ranges: Vec::new(),
                output_byte_offset: 0,
                output_byte_count: 8_224,
                parameters: Vec::new(),
            },
            VulkanDistributedDispatchShard {
                device_id: "gpu1".to_string(),
                row_start: 128,
                row_count: 128,
                workgroup_count_x: 8,
                base_workgroup_z: 128,
                input_range: VulkanDistributedActivationRange {
                    byte_offset: 0,
                    byte_count: 8_224,
                },
                auxiliary_input_ranges: Vec::new(),
                output_byte_offset: 0,
                output_byte_count: 8_224,
                parameters: Vec::new(),
            },
        ]
    };
    let dispatch = |dispatch_index,
                    node_id: &str,
                    input_activation,
                    output_activation| VulkanDistributedDispatchPlan {
        owner_device_id: "gpu0".to_string(),
        dispatch_index,
        component_id: "layer".to_string(),
        node_id: node_id.to_string(),
        reusable_family_id: node_id.to_string(),
        input_byte_capacity: 8_224,
        output_byte_capacity: 8_224,
        output_rows: 256,
        input_width: 2_048,
        row_alignment: 1,
        input_activation,
        auxiliary_input_activations: Vec::new(),
        output_activation,
        distribution: VulkanDistributedDispatchDistribution::ExpertRange,
        distributed_parameter_byte_count: 0,
        shards: shards(),
    };
    let gate = dispatch(
        9,
        "sparse_moe_gate_up",
        activation(0, "normalized", 0),
        activation(3, "expert_intermediates", 1),
    );
    let down = dispatch(
        10,
        "sparse_moe_down",
        activation(0, "expert_intermediates", 1),
        activation(2, "expert_outputs", 2),
    );
    let group = VulkanDistributedDispatchGroup {
        owner_device_id: "gpu0".to_string(),
        dispatches: vec![gate.clone(), down.clone()],
    };
    let plan = VulkanDistributedExecutionPlan {
        device_ids: vec!["gpu0".to_string(), "gpu1".to_string()],
        storage_buffer_offset_alignment: 256,
        dispatches: vec![gate, down],
        dispatch_groups: vec![group],
        shared_input_byte_capacity: 8_224,
        shared_output_byte_capacity: 8_224,
        distributed_parameter_byte_count: 0,
    };

    let specs = distributed_component_batch_private_activation_specs(&plan);

    assert_eq!(specs.len(), 1);
    let (key, spec) = specs.first_key_value().unwrap();
    assert_eq!(key.owner_device_id, "gpu0");
    assert_eq!(key.component_id, "layer");
    assert_eq!(key.signal_id, "expert_intermediates");
    assert_eq!(key.slot, 1);
    assert_eq!(spec.signal_byte_capacity, 8_224);
    assert_eq!(
        spec.device_ids,
        BTreeSet::from(["gpu0".to_string(), "gpu1".to_string()])
    );
}

#[test]
fn distributed_batch_output_binding_repeats_the_full_lane_stride() {
    let (offset, byte_capacity) =
        distributed_batch_shard_output_binding_range(8_192, 4, 2_048, 2_048).unwrap();

    assert_eq!(offset, 2_048);
    assert_eq!(byte_capacity, 26_624);
    assert_eq!(offset + byte_capacity, 28_672);
    assert!(offset + byte_capacity <= 4 * 8_192);
}

#[test]
fn distributed_batch_output_binding_rejects_a_shard_past_the_frame() {
    let error = distributed_batch_shard_output_binding_range(8_192, 4, 7_168, 2_048).unwrap_err();

    assert!(error.to_string().contains("exceeds frame capacity 8192"));
}

#[test]
fn distributed_batch_workgroups_preserve_the_compiled_row_granularity() {
    assert_eq!(
        distributed_batch_rows_per_workgroup(32_768, 512, "layer", "ffn").unwrap(),
        64
    );

    let error = distributed_batch_rows_per_workgroup(32_769, 512, "layer", "ffn").unwrap_err();
    assert!(
        error
            .to_string()
            .contains("cannot partition 32769 rows across 512 workgroups")
    );
}
