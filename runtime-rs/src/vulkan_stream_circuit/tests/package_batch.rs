use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use super::*;
use crate::stream_circuit::{
    ResolvedLoweredExecutionGraph, StreamCircuitPlacementSpec, StreamCircuitRuntimeGraph,
};
use crate::stream_plan::{StreamCircuitExecutionPlan, StreamCircuitResourcePlan};
use crate::test_support::{
    tiny_model_dir, tiny_model_package_manifest_path, tiny_model_tensor_index_path,
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
                stream_control_binding: None,
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
fn component_batch_fanout_owns_one_lifetime_through_its_last_consumer() {
    let fanout = produced_port_signal_key("input_adapter", "shared_context");
    let scratch = VulkanComponentBatchSignalKey::Activation {
        component_id: "draft_00".to_string(),
        signal_id: "scratch".to_string(),
    };
    let lifetimes = vec![
        VulkanComponentBatchSignalLifetime {
            key: fanout.clone(),
            frame_byte_capacity: 8_192,
            host_visible: false,
            first_dispatch: 0,
            last_dispatch: 6,
        },
        VulkanComponentBatchSignalLifetime {
            key: scratch.clone(),
            frame_byte_capacity: 8_192,
            host_visible: false,
            first_dispatch: 1,
            last_dispatch: 5,
        },
    ];

    let (indices, buffers) = allocate_component_batch_signal_lifetimes(lifetimes);

    assert_eq!(buffers.len(), 2);
    assert_ne!(indices[&fanout], indices[&scratch]);
}

#[test]
fn component_batch_sibling_edges_resolve_to_one_produced_port() {
    let descriptor = |edge_index: usize, destination_component_id: &str| {
        VulkanMountedPlacedBoundDescriptor {
            binding: 0,
            usage: VulkanKernelDescriptorUsage::OutputSignal,
            name: "shared_context".to_string(),
            target: VulkanMountedPlacedBoundDescriptorTarget::ProducedPortBuffer {
                port: VulkanPlacedProducedPortBufferBinding {
                    local_edges: vec![VulkanPlacedLocalEdgeBufferBinding {
                    buffer_index: edge_index,
                    edge: VulkanPlacedLocalEdge {
                        buffer_index: edge_index,
                        edge_id: format!("edge_{edge_index}_local"),
                        edge_index,
                        connection: StreamCircuitConnection::Forward,
                        signal: "shared_context".to_string(),
                        shape: vec![4_096],
                        element_count: 4_096,
                        byte_capacity: Some(8_192),
                        device_id: "gpu0".to_string(),
                        source_component_id: "input_adapter".to_string(),
                        source_port_id: "shared_context".to_string(),
                        source_component_port: Some("shared_context".to_string()),
                        destination_component_id: destination_component_id.to_string(),
                        destination_port_id: "shared_context".to_string(),
                        destination_component_port: Some("shared_context".to_string()),
                        transport: EdgeTransport::LocalBuffer {
                            device_id: "gpu0".to_string(),
                        },
                    },
                    byte_capacity: 8_192,
                    }],
                    outgoing_endpoints: Vec::new(),
                    byte_capacity: 8_192,
                },
            },
        }
    };

    let (first_key, first_capacity) =
        component_batch_signal_target(&descriptor(4, "draft_00"))
            .unwrap()
            .unwrap();
    let (second_key, second_capacity) =
        component_batch_signal_target(&descriptor(5, "draft_01"))
            .unwrap()
            .unwrap();

    assert_eq!(first_key, second_key);
    assert_eq!(
        first_key,
        produced_port_signal_key("input_adapter", "shared_context")
    );
    assert_eq!(first_capacity, 8_192);
    assert_eq!(second_capacity, 8_192);
}

#[test]
fn component_batch_fanout_rejects_incompatible_sibling_edge_frames() {
    let fanout = produced_port_signal_key("producer", "output");
    let mut lifetimes = BTreeMap::new();
    merge_component_batch_signal_lifetime(
        &mut lifetimes,
        fanout.clone(),
        8_192,
        false,
        0,
        1,
    )
    .unwrap();

    let error = merge_component_batch_signal_lifetime(
        &mut lifetimes,
        fanout,
        4_096,
        false,
        0,
        2,
    )
    .unwrap_err();

    assert!(error
        .to_string()
        .contains("incompatible physical requirements"));
}

#[test]
fn speculative_source_tap_signal_liveness_survives_the_complete_batch() {
    let key = |signal_id: &str| VulkanComponentBatchSignalKey::Activation {
        component_id: "component".to_string(),
        signal_id: signal_id.to_string(),
    };
    let retained = key("retained_target_hidden");
    let later = key("later_scratch");
    let mut lifetimes = vec![
        VulkanComponentBatchSignalLifetime {
            key: retained.clone(),
            frame_byte_capacity: 4_096,
            host_visible: false,
            first_dispatch: 0,
            last_dispatch: 1,
        },
        VulkanComponentBatchSignalLifetime {
            key: later.clone(),
            frame_byte_capacity: 4_096,
            host_visible: false,
            first_dispatch: 2,
            last_dispatch: 3,
        },
    ];

    retain_component_batch_signal_lifetimes(
        &mut lifetimes,
        &BTreeSet::from([retained.clone()]),
        4,
    )
    .unwrap();
    let (indices, buffers) = allocate_component_batch_signal_lifetimes(lifetimes);

    assert_eq!(buffers.len(), 2);
    assert_ne!(indices[&retained], indices[&later]);
}

#[test]
fn speculative_source_tap_retention_rejects_an_unproduced_signal() {
    let retained = VulkanComponentBatchSignalKey::Activation {
        component_id: "component".to_string(),
        signal_id: "target_hidden".to_string(),
    };
    let missing = VulkanComponentBatchSignalKey::Activation {
        component_id: "missing".to_string(),
        signal_id: "target_hidden".to_string(),
    };
    let mut lifetimes = vec![VulkanComponentBatchSignalLifetime {
        key: retained,
        frame_byte_capacity: 4_096,
        host_visible: false,
        first_dispatch: 0,
        last_dispatch: 1,
    }];

    let error = retain_component_batch_signal_lifetimes(
        &mut lifetimes,
        &BTreeSet::from([missing]),
        2,
    )
    .unwrap_err();

    assert!(error
        .to_string()
        .contains("has no physical lifetime"));
}

#[test]
fn parallel_block_edge_capacity_is_partitioned_into_exact_lane_frames() {
    assert_eq!(
        component_batch_edge_frame_byte_capacity(
            &StreamCircuitConnection::ParallelBlockScatter { width: 7 },
            7 * 4096,
        )
        .unwrap(),
        4096
    );
    assert_eq!(
        component_batch_edge_frame_byte_capacity(
            &StreamCircuitConnection::ParallelBlockGather { width: 7 },
            7 * 4096,
        )
        .unwrap(),
        4096
    );
    assert!(
        component_batch_edge_frame_byte_capacity(
            &StreamCircuitConnection::ParallelBlockScatter { width: 7 },
            4096,
        )
        .is_err()
    );
}

#[test]
fn produced_parallel_block_port_uses_one_lane_as_its_batch_frame() {
    let port = VulkanPlacedProducedPortBufferBinding {
        local_edges: vec![VulkanPlacedLocalEdgeBufferBinding {
            buffer_index: 0,
            edge: VulkanPlacedLocalEdge {
                buffer_index: 0,
                edge_id: "edge_3_local".to_string(),
                edge_index: 3,
                connection: StreamCircuitConnection::ParallelBlockGather { width: 5 },
                signal: "stream_frame_block".to_string(),
                shape: vec![5, 4, 4096],
                element_count: 5 * 4 * 4096,
                byte_capacity: Some(5 * 4 * 4096 * 2),
                device_id: "gpu0".to_string(),
                source_component_id: "draft_layer_02".to_string(),
                source_port_id: "output_frame".to_string(),
                source_component_port: Some("output_frame".to_string()),
                destination_component_id: "draft_output".to_string(),
                destination_port_id: "input_frames".to_string(),
                destination_component_port: Some("input_frames".to_string()),
                transport: EdgeTransport::LocalBuffer {
                    device_id: "gpu0".to_string(),
                },
            },
            byte_capacity: 5 * 4 * 4096 * 2,
        }],
        outgoing_endpoints: Vec::new(),
        byte_capacity: 5 * 4 * 4096 * 2,
    };

    assert_eq!(
        component_batch_produced_port_frame_byte_capacity(&port).unwrap(),
        4 * 4096 * 2,
    );
}

#[test]
fn speculative_source_tap_selects_last_runtime_instance_of_source_component() {
    let instances = vec![
        VulkanRuntimeComponentInstance {
            instance_id: "layer_05_first".to_string(),
            source_component_id: "layer_05".to_string(),
            device_id: "amd0".to_string(),
            execution_index: 5,
        },
        VulkanRuntimeComponentInstance {
            instance_id: "layer_05_last".to_string(),
            source_component_id: "layer_05".to_string(),
            device_id: "amd1".to_string(),
            execution_index: 9,
        },
    ];
    let tap = StreamCircuitGraphSourceTap {
        component_id: "layer_05".to_string(),
        port_id: "output_frame".to_string(),
        instance_selection: StreamCircuitGraphSourceTapInstanceSelection::LastInExecutionOrder,
    };

    assert_eq!(
        resolve_speculative_source_tap_instance(&instances, &tap)
            .unwrap()
            .instance_id,
        "layer_05_last"
    );
    let missing = StreamCircuitGraphSourceTap {
        component_id: "missing".to_string(),
        ..tap
    };
    assert!(resolve_speculative_source_tap_instance(&instances, &missing).is_err());
}

#[test]
fn speculative_planning_mounts_every_executable_draft_phase() {
    for role in [
        CircuitRuntimeRole::DraftInputAdapter,
        CircuitRuntimeRole::DraftProcessor,
        CircuitRuntimeRole::DraftOutputTransducer,
    ] {
        assert_eq!(
            speculative_decoder_planning_role(role, true),
            CircuitRuntimeRole::SignalProcessor
        );
        assert_eq!(speculative_decoder_planning_role(role, false), role);
    }
    assert_eq!(
        speculative_decoder_planning_role(CircuitRuntimeRole::OutputTransducer, true),
        CircuitRuntimeRole::OutputTransducer
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
fn local_demand_batch_guards_only_commands_after_its_direct_gate() {
    let commands = vec![
        VulkanDemandResidencyBatchCommand::Step(0),
        VulkanDemandResidencyBatchCommand::Gate(0),
        VulkanDemandResidencyBatchCommand::Step(1),
        VulkanDemandResidencyBatchCommand::Gate(1),
        VulkanDemandResidencyBatchCommand::Step(2),
    ];

    assert_eq!(
        demand_batch_conditional_regions(&commands, 0, 1, false, false).unwrap(),
        vec![None, None, Some(1), Some(1), Some(2)]
    );
}

#[test]
fn shared_demand_batch_guards_its_entire_initial_submission() {
    let commands = vec![
        VulkanDemandResidencyBatchCommand::Step(0),
        VulkanDemandResidencyBatchCommand::Gate(0),
        VulkanDemandResidencyBatchCommand::Step(1),
        VulkanDemandResidencyBatchCommand::Gate(1),
        VulkanDemandResidencyBatchCommand::Step(2),
    ];

    assert_eq!(
        demand_batch_conditional_regions(&commands, 0, 1, true, false).unwrap(),
        vec![Some(1), Some(1), Some(2), Some(2), Some(3)]
    );
}

#[test]
fn shared_demand_batch_resume_executes_its_gate_before_guarding_the_suffix() {
    let commands = vec![
        VulkanDemandResidencyBatchCommand::Step(0),
        VulkanDemandResidencyBatchCommand::Gate(0),
        VulkanDemandResidencyBatchCommand::Step(1),
        VulkanDemandResidencyBatchCommand::Gate(1),
        VulkanDemandResidencyBatchCommand::Step(2),
    ];

    assert_eq!(
        demand_batch_conditional_regions(&commands, 1, 1, true, true).unwrap(),
        vec![None, Some(1), Some(1), Some(2)]
    );
}

#[test]
fn demand_batch_conditional_layout_rejects_an_invalid_direct_gate() {
    let commands = vec![
        VulkanDemandResidencyBatchCommand::Step(0),
        VulkanDemandResidencyBatchCommand::Gate(0),
        VulkanDemandResidencyBatchCommand::Step(1),
    ];

    assert!(demand_batch_conditional_regions(&commands, 0, 0, true, false).is_err());
    assert!(demand_batch_conditional_regions(&commands, 2, 1, true, true).is_err());
    assert!(demand_batch_conditional_regions(&commands, 0, commands.len(), true, false).is_err());
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
    let targets = [11, 99, 88, 77].map(sampled_token);
    let result = verify_speculative_token_prefix(&[11, 12, 13], &targets).unwrap();

    assert_eq!(result.accepted_draft_count, 1);
    assert_eq!(result.committed_target_tick_count, 2);
    assert_eq!(result.emitted_tokens, targets[..2]);
}

#[test]
fn speculative_verification_emits_the_bonus_token_when_all_drafts_match() {
    let targets = [11, 12, 13].map(sampled_token);
    let result = verify_speculative_token_prefix(&[11, 12], &targets).unwrap();

    assert_eq!(result.accepted_draft_count, 2);
    assert_eq!(result.committed_target_tick_count, 3);
    assert_eq!(result.emitted_tokens, targets);
}

#[test]
fn speculative_verification_rejects_incomplete_target_results() {
    let error = verify_speculative_token_prefix(&[11, 12], &[11, 12].map(sampled_token)).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("2 draft tokens but 2 target predictions; expected 3")
    );
}

#[test]
fn speculative_verification_stops_at_the_first_emitted_stop_token() {
    let mut result =
        verify_speculative_token_prefix(&[11, 12], &[11, 12, 99].map(sampled_token)).unwrap();

    truncate_speculative_verification_at_stop(&mut result, &BTreeSet::from([11]));

    assert_eq!(result.accepted_draft_count, 1);
    assert_eq!(result.committed_target_tick_count, 1);
    assert_eq!(result.emitted_tokens, [sampled_token(11)]);
}

#[test]
fn speculative_confidence_keeps_only_the_contiguous_confident_prefix() {
    let logits = [3.0, 0.0, -0.25, 8.0];

    assert_eq!(speculative_confident_prefix_len(&logits, 0.0).unwrap(), 4);
    assert_eq!(speculative_confident_prefix_len(&logits, 0.5).unwrap(), 2);
    assert_eq!(speculative_confident_prefix_len(&logits, 1.0).unwrap(), 0);
}

#[test]
fn speculative_confidence_rejects_invalid_thresholds_and_logits() {
    for threshold in [-0.01, 1.01, f32::NAN, f32::INFINITY] {
        assert!(speculative_confident_prefix_len(&[0.0], threshold).is_err());
    }
    assert!(speculative_confident_prefix_len(&[f32::NAN], 0.0).is_err());
    assert!(speculative_confident_prefix_len(&[f32::INFINITY], 0.5).is_err());
}

#[test]
fn speculative_decode_stats_report_rollbacks_and_total_cost() {
    let mut stats = VulkanSpeculativeDecodeStats::default();
    stats.record_cycle(&VulkanSpeculativeCycleRun {
        decoder_id: "draft".to_string(),
        initial_token_id: 1,
        start_stream_tick: 0,
        draft_token_ids: vec![2, 3],
        target_tokens: vec![2, 9, 4].into_iter().map(sampled_token).collect(),
        verification: VulkanSpeculativeVerificationResult {
            accepted_draft_count: 1,
            committed_target_tick_count: 2,
            emitted_tokens: vec![sampled_token(2), sampled_token(9)],
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
    assert_eq!(
        stats.windows.get(&2),
        Some(&VulkanSpeculativeWindowStats {
            draft_width: 2,
            cycle_count: 1,
            emitted_token_count: 2,
            total_time_ns: 100,
        })
    );
    assert_eq!(stats.cycle_traces.len(), 1);
    assert_eq!(stats.cycle_traces[0].draft_token_ids, vec![2, 3]);
    assert_eq!(stats.cycle_traces[0].target_token_ids, vec![2, 9, 4]);
}

fn adaptive_window_cycle(
    draft_width: usize,
    emitted_token_count: usize,
    total_time_ns: u64,
) -> VulkanSpeculativeCycleRun {
    let draft_token_ids = (0..draft_width)
        .map(|index| u32::try_from(index + 2).unwrap())
        .collect::<Vec<_>>();
    let target_tokens = (0..=draft_width)
        .map(|index| sampled_token(u32::try_from(index + 2).unwrap()))
        .collect::<Vec<_>>();
    VulkanSpeculativeCycleRun {
        decoder_id: "draft".to_string(),
        initial_token_id: 1,
        start_stream_tick: 0,
        draft_token_ids,
        target_tokens: target_tokens.clone(),
        verification: VulkanSpeculativeVerificationResult {
            accepted_draft_count: emitted_token_count.saturating_sub(1),
            committed_target_tick_count: emitted_token_count,
            emitted_tokens: target_tokens[..emitted_token_count].to_vec(),
        },
        draft_time_ns: total_time_ns / 10,
        target_verification_time_ns: total_time_ns.saturating_mul(8) / 10,
        draft_catch_up_time_ns: total_time_ns / 100,
        total_time_ns,
    }
}

#[test]
fn adaptive_speculative_windows_probe_execution_shape_boundaries() {
    assert_eq!(
        adaptive_speculative_window_candidates(5),
        vec![5, 1, 2, 3, 4]
    );
    assert_eq!(
        adaptive_speculative_window_candidates(15),
        vec![15, 1, 2, 3, 4, 7, 8]
    );
    assert_eq!(adaptive_speculative_window_candidates(1), vec![1]);
}

#[test]
fn adaptive_speculative_window_selects_measured_accepted_throughput() {
    let mut selector = VulkanAdaptiveSpeculativeWindowSelector::new(5);
    let scores = BTreeMap::from([
        (1, (2, 100)),
        (2, (2, 160)),
        (3, (3, 260)),
        (4, (3, 330)),
        (5, (3, 400)),
    ]);

    for expected_width in [5, 1, 2, 3, 4] {
        for _ in 0..3 {
            assert_eq!(selector.next_window_width(), expected_width);
            let (emitted, elapsed) = scores[&expected_width];
            selector.record_cycle(
                expected_width,
                &adaptive_window_cycle(expected_width, emitted, elapsed),
            );
        }
    }

    assert!(selector.is_calibrated());
    assert_eq!(selector.next_window_width(), 1);
}

#[test]
fn adaptive_speculative_window_excludes_first_shape_warmup() {
    let mut selector = VulkanAdaptiveSpeculativeWindowSelector::new(2);

    for (width, emitted, elapsed) in [
        (2, 3, 10),
        (2, 2, 200),
        (2, 2, 200),
        (1, 1, 10_000),
        (1, 2, 100),
        (1, 2, 100),
    ] {
        assert_eq!(selector.next_window_width(), width);
        selector.record_cycle(width, &adaptive_window_cycle(width, emitted, elapsed));
    }

    assert!(selector.is_calibrated());
    assert_eq!(selector.next_window_width(), 1);
}

#[test]
fn adaptive_speculative_window_ignores_truncated_or_failed_measurements() {
    let mut selector = VulkanAdaptiveSpeculativeWindowSelector::new(2);

    selector.record_cycle(2, &adaptive_window_cycle(1, 2, 100));
    assert_eq!(selector.next_window_width(), 2);
    selector.record_cycle(2, &adaptive_window_cycle(2, 2, 0));
    assert_eq!(selector.next_window_width(), 2);
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
            selection_priority: 0,
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
        selection_priority: 0,
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
fn component_batches_honor_package_owned_selection_priority() {
    let artifact =
        |lane_tile_width, selection_priority| VulkanResidentComponentBatchKernelArtifact {
            component_id: "processor".to_string(),
            node_id: "project".to_string(),
            execution_domain: VulkanResidentComponentKernelExecutionDomain::DecodeAndPrefill,
            batch_mode: VulkanResidentComponentKernelBatchMode::WeightShared,
            lane_tile_width,
            selection_priority,
            independent_candidate_compatible: true,
            causal_sequence_compatible: true,
            parallel_block_compatible: true,
            device_requirements: VulkanResidentVulkanDeviceRequirements::default(),
            stages: Vec::new(),
        };
    let artifacts = vec![artifact(8, 0), artifact(16, 1)];

    let selected = select_component_batch_kernel_artifact(
        &artifacts,
        "processor",
        "project",
        VulkanComponentBatchExecutionMode::CausalSequence,
        6,
    )
    .unwrap();

    assert_eq!(selected.lane_tile_width, 16);
    assert_eq!(selected.selection_priority, 1);
}

#[test]
fn component_batches_never_trade_lane_coverage_for_selection_priority() {
    let artifact =
        |lane_tile_width, selection_priority| VulkanResidentComponentBatchKernelArtifact {
            component_id: "processor".to_string(),
            node_id: "project".to_string(),
            execution_domain: VulkanResidentComponentKernelExecutionDomain::DecodeAndPrefill,
            batch_mode: VulkanResidentComponentKernelBatchMode::WeightShared,
            lane_tile_width,
            selection_priority,
            independent_candidate_compatible: true,
            causal_sequence_compatible: true,
            parallel_block_compatible: true,
            device_requirements: VulkanResidentVulkanDeviceRequirements::default(),
            stages: Vec::new(),
        };
    let artifacts = vec![artifact(16, 1), artifact(64, 0)];

    let selected = select_component_batch_kernel_artifact(
        &artifacts,
        "processor",
        "project",
        VulkanComponentBatchExecutionMode::CausalSequence,
        48,
    )
    .unwrap();

    assert_eq!(selected.lane_tile_width, 64);
    assert_eq!(selected.selection_priority, 0);
}

#[test]
fn component_batches_prefer_the_widest_tile_when_every_candidate_is_undersized() {
    let artifact =
        |lane_tile_width, selection_priority| VulkanResidentComponentBatchKernelArtifact {
            component_id: "processor".to_string(),
            node_id: "project".to_string(),
            execution_domain: VulkanResidentComponentKernelExecutionDomain::DecodeAndPrefill,
            batch_mode: VulkanResidentComponentKernelBatchMode::WeightShared,
            lane_tile_width,
            selection_priority,
            independent_candidate_compatible: true,
            causal_sequence_compatible: true,
            parallel_block_compatible: true,
            device_requirements: VulkanResidentVulkanDeviceRequirements::default(),
            stages: Vec::new(),
        };
    let artifacts = vec![artifact(16, 1), artifact(32, 0)];

    let selected = select_component_batch_kernel_artifact(
        &artifacts,
        "processor",
        "project",
        VulkanComponentBatchExecutionMode::CausalSequence,
        48,
    )
    .unwrap();

    assert_eq!(selected.lane_tile_width, 32);
    assert_eq!(selected.selection_priority, 0);
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
        selection_priority: 0,
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
        selection_priority: 0,
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
fn only_selected_snapshot_reader_artifacts_require_temporal_state_capture() {
    let artifact = |source_binding: Option<u32>| VulkanResidentComponentBatchKernelArtifact {
        component_id: "processor".to_string(),
        node_id: "attention".to_string(),
        execution_domain: VulkanResidentComponentKernelExecutionDomain::Prefill,
        batch_mode: VulkanResidentComponentKernelBatchMode::CausalScan,
        lane_tile_width: 8,
        selection_priority: 0,
        independent_candidate_compatible: false,
        causal_sequence_compatible: true,
        parallel_block_compatible: false,
        device_requirements: VulkanResidentVulkanDeviceRequirements::default(),
        stages: vec![VulkanResidentComponentBatchStageArtifact {
            shader_path: "shaders/attention.comp".to_string(),
            spirv_words: Vec::new(),
            local_size_x: 64,
            workgroup_count_x: 1,
            descriptor_bindings: Vec::new(),
            state_snapshot_binding: source_binding.map(|_| 30),
            state_snapshot_source_binding: source_binding,
            control: VulkanResidentComponentBatchControlSpec::StorageBuffer {
                byte_count: if source_binding.is_some() { 20 } else { 16 },
                binding: 8,
                payload: if source_binding.is_some() {
                    VulkanResidentComponentBatchControlPayload::TemporalStateSnapshots
                } else {
                    VulkanResidentComponentBatchControlPayload::Temporal
                },
                access: VulkanResidentComponentBatchControlAccess::Read,
            },
            indirect_dispatch_byte_offset: None,
            dispatch_y_from_batch_width: false,
        }],
    };

    assert!(component_batch_artifact_reads_state_snapshots(&artifact(Some(1))));
    assert!(!component_batch_artifact_reads_state_snapshots(&artifact(None)));
}

#[test]
fn component_batch_execution_contract_requires_matching_shader_mode() {
    let execution = |batch_mode, batch_shader_path: Option<String>| {
        let batch_implementations = batch_shader_path
            .into_iter()
            .map(|shader_path| VulkanResidentComponentBatchImplementationSpec {
                execution_domain: VulkanResidentComponentKernelExecutionDomain::DecodeAndPrefill,
                lane_tile_width: 16,
                selection_priority: 0,
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
                    state_snapshot_source_binding: None,
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
                stream_control_binding: None,
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

    let mut invalid_indirect = execution(
        VulkanResidentComponentKernelBatchMode::WeightShared,
        Some("shaders/project_batch.spv".to_string()),
    );
    let indirect_consumer = &mut invalid_indirect[0].kernels[0].batch_implementations[0].stages[0];
    indirect_consumer.control = VulkanResidentComponentBatchControlSpec::StorageBuffer {
        byte_count: VulkanResidentComponentBatchControlPayload::WidthExpertRangeIndirect
            .byte_count(),
        binding: 31,
        payload: VulkanResidentComponentBatchControlPayload::WidthExpertRangeIndirect,
        access: VulkanResidentComponentBatchControlAccess::Read,
    };
    indirect_consumer.indirect_dispatch_byte_offset = Some(16);
    let indirect_error =
        validate_component_executions("fixture", &invalid_indirect).unwrap_err();
    assert!(indirect_error.to_string().contains("invalid WeightShared"));

    let mut valid_indirect = invalid_indirect;
    let mut producer = valid_indirect[0].kernels[0].batch_implementations[0].stages[0].clone();
    producer.shader_path = "shaders/dispatch_producer.spv".to_string();
    producer.indirect_dispatch_byte_offset = None;
    producer.control = VulkanResidentComponentBatchControlSpec::StorageBuffer {
        byte_count: VulkanResidentComponentBatchControlPayload::WidthExpertRangeIndirect
            .byte_count(),
        binding: 31,
        payload: VulkanResidentComponentBatchControlPayload::WidthExpertRangeIndirect,
        access: VulkanResidentComponentBatchControlAccess::ReadWrite,
    };
    valid_indirect[0].kernels[0].batch_implementations[0]
        .stages
        .insert(0, producer);
    validate_component_executions("fixture", &valid_indirect).unwrap();
}

#[test]
fn component_batch_control_preserves_temporal_position_and_capacity() {
    let bytes = component_batch_control_bytes(64, 0x1122_3344_5566_7788, 65_536);

    assert_eq!(&bytes[0..4], &64u32.to_le_bytes());
    assert_eq!(&bytes[4..12], &0x1122_3344_5566_7788u64.to_le_bytes());
    assert_eq!(&bytes[12..16], &65_536u32.to_le_bytes());
}

#[test]
fn component_batch_dispatch_control_preserves_compiled_indirect_execution() {
    let mut stage = VulkanResidentComponentBatchStageArtifact {
        shader_path: "shaders/routed_expert.spv".to_string(),
        spirv_words: Vec::new(),
        local_size_x: 64,
        workgroup_count_x: 384,
        descriptor_bindings: Vec::new(),
        state_snapshot_binding: None,
        state_snapshot_source_binding: None,
        control: VulkanResidentComponentBatchControlSpec::StorageBuffer {
            byte_count: VulkanResidentComponentBatchControlPayload::WidthExpertRangeIndirect
                .byte_count(),
            binding: 31,
            payload: VulkanResidentComponentBatchControlPayload::WidthExpertRangeIndirect,
            access: VulkanResidentComponentBatchControlAccess::ReadWrite,
        },
        indirect_dispatch_byte_offset: Some(16),
        dispatch_y_from_batch_width: false,
    };

    assert_eq!(
        component_batch_dispatch_control(&stage).unwrap(),
        VulkanComponentBatchDispatchControl::Indirect {
            payload: VulkanResidentComponentBatchControlPayload::WidthExpertRangeIndirect,
            byte_offset: 16,
        }
    );

    stage.indirect_dispatch_byte_offset = None;
    assert_eq!(
        component_batch_dispatch_control(&stage).unwrap(),
        VulkanComponentBatchDispatchControl::Fixed,
    );

    stage.dispatch_y_from_batch_width = true;
    assert_eq!(
        component_batch_dispatch_control(&stage).unwrap(),
        VulkanComponentBatchDispatchControl::BatchWidthY,
    );

    stage.indirect_dispatch_byte_offset = Some(16);
    assert!(component_batch_dispatch_control(&stage).is_err());
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
        state_snapshot_source_binding: None,
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
        state_snapshot_source_binding: None,
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
        state_snapshot_source_binding: None,
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
    assert_eq!(
        component_batch_control_payload_bytes(
            VulkanResidentComponentBatchControlPayload::TemporalStateSnapshots,
            &control,
            true,
        ),
        [control.as_slice(), 1u32.to_le_bytes().as_slice()].concat(),
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
