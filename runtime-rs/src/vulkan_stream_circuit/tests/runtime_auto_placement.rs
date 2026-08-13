fn maximum_resident_bytes(plan: &VulkanRuntimeResidencyPlan, device_id: &str) -> usize {
    vulkan_runtime_maximum_device_resident_bytes(
        plan.device_plans
            .iter()
            .find(|device| device.device_id == device_id)
            .unwrap(),
    )
    .unwrap()
}

fn auto_placement_hardware_profile(
    stable_device_id: &str,
    architecture: &str,
) -> crate::HardwareProcessProfile {
    crate::HardwareProcessProfile::create(crate::HardwareProcessProfileDefinition {
        hardware_identity: crate::HardwareIdentity {
            device_kind: crate::HardwareDeviceKind::Gpu,
            stable_device_id: stable_device_id.to_string(),
            name: format!("{architecture} fixture"),
            vendor_id: "fixture".to_string(),
            device_id: architecture.to_string(),
            architecture: architecture.to_string(),
            physical_location: stable_device_id.to_string(),
        },
        processes: vec![crate::HardwareProcessCapability::new(
            "compute",
            crate::HardwareProcessCategory::Arithmetic,
            crate::HardwareProcessAvailability::Available,
            crate::HardwareProcessProgrammability::Direct,
            "vulkan",
        )],
        memory_domains: vec![crate::HardwareMemoryDomain {
            name: "device_local".to_string(),
            kind: "device_local".to_string(),
            capacity_bytes: 64 * 1024 * 1024 * 1024,
            host_visible: false,
            device_local: true,
            coherent: false,
            cached: false,
            minimum_alignment_bytes: 4,
            properties: BTreeMap::new(),
        }],
        interconnects: Vec::new(),
        provenance: crate::HardwareProfileProvenance {
            api: "vulkan".to_string(),
            api_version: "1.4".to_string(),
            driver: "fixture".to_string(),
            driver_version: "1".to_string(),
            compiler: "fixture".to_string(),
            operating_system: "linux".to_string(),
            discovery_backend: "fixture".to_string(),
        },
        capability_extensions: BTreeMap::from([
            (
                "vulkan_compiler_capabilities".to_string(),
                serde_json::json!({
                    "shader_features": [
                        "buffer_device_address",
                        "cooperative_matrix",
                        "shader_bfloat16_cooperative_matrix",
                        "shader_bfloat16_dot_product",
                        "shader_bfloat16_type",
                        "shader_float16",
                        "shader_int8",
                        "shader_int16",
                        "shader_integer_dot_product",
                        "storage_buffer8_bit_access",
                        "storage_buffer16_bit_access",
                        "uniform_and_storage_buffer8_bit_access",
                        "uniform_and_storage_buffer16_bit_access",
                        "vulkan_memory_model",
                        "vulkan_memory_model_device_scope"
                    ],
                    "subgroup_operations": [
                        "arithmetic", "ballot", "basic", "shuffle", "shuffle_relative", "vote"
                    ],
                    "subgroup_compute_supported": true,
                    "subgroup_size": 64,
                    "max_compute_work_group_invocations": 1024,
                    "max_compute_work_group_size_x": 1024,
                    "cooperative_bfloat16_shapes": [[16, 16, 16]],
                    "cooperative_float8_e4m3_shapes": []
                }),
            ),
            (
                "vulkan_device".to_string(),
                serde_json::json!({
                    "extensions": [
                        "VK_KHR_cooperative_matrix",
                        "VK_KHR_shader_bfloat16"
                    ]
                }),
            ),
        ]),
        identity_extensions: BTreeMap::new(),
        runtime_bindings: BTreeMap::new(),
    })
    .unwrap()
}

#[test]
fn runtime_component_weights_include_target_endpoint_parameters() {
    let runtime_model = fixture_model_runtime_model();
    let tensor_index = runtime_model
        .load_runtime_tensor_index(tiny_model_dir())
        .unwrap();
    let components =
        capacity_packed_runtime_components(&runtime_model, &tensor_index, false).unwrap();
    let expected = runtime_model
        .circuit_graph
        .components
        .iter()
        .flat_map(|component| component.params.refs.values())
        .filter_map(|parameter| parameter.tensor.as_deref())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|tensor| tensor_index.tensors[tensor].byte_count.unwrap())
        .sum::<usize>();

    assert_eq!(components.len(), 1);
    assert_eq!(components[0].resident_weight_bytes, expected);
}

#[test]
fn runtime_component_weights_charge_mounted_draft_parameters_to_the_output_tail() {
    let mut runtime_model = fixture_model_runtime_model();
    let mut tensor_index = runtime_model
        .load_runtime_tensor_index(tiny_model_dir())
        .unwrap();
    let without_draft =
        capacity_packed_runtime_components(&runtime_model, &tensor_index, false).unwrap();
    let mut draft_graph = runtime_model.circuit_graph.clone();
    let draft_component = draft_graph
        .components
        .iter_mut()
        .find(|component| component.runtime_role.is_signal_processor())
        .unwrap();
    let draft_parameter = draft_component
        .params
        .refs
        .values_mut()
        .find(|parameter| parameter.tensor.is_some())
        .unwrap();
    let source_tensor = draft_parameter.tensor.clone().unwrap();
    let unique_draft_tensor = "draft.unique.weight".to_string();
    draft_parameter.tensor = Some(unique_draft_tensor.clone());
    let draft_tensor_bytes = tensor_index.tensors[&source_tensor].byte_count.unwrap();
    tensor_index.tensors.insert(
        unique_draft_tensor,
        tensor_index.tensors[&source_tensor].clone(),
    );
    runtime_model
        .package
        .speculative_decoders
        .push(VulkanResidentSpeculativeDecoderPackageSpec {
            id: "draft_fixture".to_string(),
            decoder_type: "fixture".to_string(),
            source_prefix: "draft".to_string(),
            execution_contract:
                VulkanResidentSpeculativeExecutionContract::AutoregressiveFeedback {
                    processor_schedule: "one_token_per_tick".to_string(),
                    output_schedule: "dedicated_token_transducer".to_string(),
                },
            proposal_contract: None,
            circuit_graph: draft_graph,
            input_adapter: None,
            output_transducer: None,
            component_executions: Vec::new(),
            state_contract: serde_json::json!({}),
            verification_contract: serde_json::json!({}),
        });

    let with_draft =
        capacity_packed_runtime_components(&runtime_model, &tensor_index, true).unwrap();
    let without_draft_balance =
        runtime_paged_placement_balance(&runtime_model, &tensor_index, &with_draft, false).unwrap();
    let with_draft_balance =
        runtime_paged_placement_balance(&runtime_model, &tensor_index, &with_draft, true).unwrap();

    assert_eq!(with_draft.len(), 1);
    assert_eq!(
        with_draft[0].resident_weight_bytes,
        without_draft[0].resident_weight_bytes + draft_tensor_bytes,
    );
    assert_eq!(
        with_draft_balance.component_weights,
        without_draft_balance.component_weights,
    );
    assert_eq!(
        with_draft_balance.output_auxiliary_weight_bytes,
        without_draft_balance.output_auxiliary_weight_bytes + draft_tensor_bytes as u128,
    );
}

#[test]
fn runtime_auto_placement_uses_one_device_when_the_complete_retained_set_fits() {
    let runtime_model = fixture_model_runtime_model_with_colocated_three_layer_series();
    let tensor_index = runtime_model
        .load_runtime_tensor_index(tiny_model_dir())
        .unwrap();
    let baseline = plan_vulkan_runtime_residency(
        tiny_model_dir(),
        &runtime_model,
        &tensor_index,
        8,
        0,
        ResourceResidencyPolicy::DemandRetained,
    )
    .unwrap();
    let required = vulkan_runtime_maximum_device_resident_bytes(&baseline.device_plans[0]).unwrap();

    let placed = capacity_pack_vulkan_runtime_model(
        tiny_model_dir(),
        &runtime_model,
        &tensor_index,
        &[
            VulkanRuntimePlacementCandidate {
                device_id: "preferred".to_string(),
                safe_capacity_bytes: required,
            },
            VulkanRuntimePlacementCandidate {
                device_id: "unneeded".to_string(),
                safe_capacity_bytes: required,
            },
        ],
        8,
        0,
        ResourceResidencyPolicy::DemandRetained,
    )
    .unwrap();

    assert_eq!(placed.selected_device_ids, ["preferred"]);
    assert_eq!(placed.runtime_model.placement_device_ids(), ["preferred"]);
}

#[test]
fn runtime_placement_calibration_groups_identical_complete_decode_transactions() {
    let mut runtime_model = fixture_model_runtime_model();
    let original = runtime_model.component_executions[0].clone();
    let mut identical = original.clone();
    identical.component_id = "layer_identical".to_string();
    runtime_model.component_executions.push(identical);
    let mut distinct = original.clone();
    distinct.component_id = "layer_distinct".to_string();
    distinct.kernels[0].workgroup_count_x = original.kernels[0].workgroup_count_x + 1;
    runtime_model.component_executions.push(distinct);
    let mut component = runtime_model
        .circuit_graph
        .components
        .iter()
        .find(|component| component.runtime_role.is_signal_processor())
        .unwrap()
        .clone();
    component.component_id = "layer_identical".to_string();
    runtime_model
        .circuit_graph
        .components
        .push(component.clone());
    component.component_id = "layer_distinct".to_string();
    runtime_model.circuit_graph.components.push(component);

    let targets = vulkan_runtime_placement_calibration_targets(&runtime_model).unwrap();

    assert_eq!(targets.len(), 2);
    assert_eq!(targets[0].component_id, original.component_id);
    assert_eq!(
        targets[0].component_ids,
        [original.component_id.clone(), "layer_identical".to_string()],
    );
    assert_eq!(targets[1].component_id, "layer_distinct");
    assert_ne!(targets[0].signature_id, targets[1].signature_id);
    assert_eq!(
        targets[1].terminal_node_id,
        original.kernels.last().unwrap().node_id,
    );
}

#[test]
fn runtime_placement_calibration_signature_includes_physical_contract_identity() {
    let runtime_model = fixture_model_runtime_model();
    let component_id = runtime_model.component_executions[0].component_id.clone();
    let original = vulkan_runtime_placement_calibration_target_for_component(
        &runtime_model,
        &component_id,
        VulkanTargetedComponentExecutionPhase::Decode,
    )
    .unwrap();

    let mut changed = runtime_model.clone();
    changed.component_executions[0].kernels[0].physical_execution_contracts[0]
        .implementation_digest = format!("sha256:{}", "f".repeat(64));
    let changed = vulkan_runtime_placement_calibration_target_for_component(
        &changed,
        &component_id,
        VulkanTargetedComponentExecutionPhase::Decode,
    )
    .unwrap();

    assert_ne!(original.signature_id, changed.signature_id);
}

#[test]
fn runtime_placement_calibration_signature_canonicalizes_contract_order() {
    let runtime_model = fixture_model_runtime_model();
    let component_id = runtime_model.component_executions[0].component_id.clone();
    let original = vulkan_runtime_placement_calibration_target_for_component(
        &runtime_model,
        &component_id,
        VulkanTargetedComponentExecutionPhase::Decode,
    )
    .unwrap();

    let mut reordered = runtime_model.clone();
    reordered.component_executions[0].kernels[0]
        .physical_execution_contracts
        .reverse();
    let reordered = vulkan_runtime_placement_calibration_target_for_component(
        &reordered,
        &component_id,
        VulkanTargetedComponentExecutionPhase::Decode,
    )
    .unwrap();

    assert_eq!(original.signature_id, reordered.signature_id);
}

#[test]
fn runtime_placement_calibration_resolves_the_requested_component_phase_exactly() {
    let runtime_model = fixture_model_runtime_model();
    let execution = runtime_model.component_executions.first().unwrap();
    let component_id = execution.component_id.clone();
    let decode_terminal = execution.kernels.last().unwrap().node_id.clone();

    let decode = vulkan_runtime_placement_calibration_target_for_component(
        &runtime_model,
        &component_id,
        VulkanTargetedComponentExecutionPhase::Decode,
    )
    .unwrap();
    let prefill = vulkan_runtime_placement_calibration_target_for_component(
        &runtime_model,
        &component_id,
        VulkanTargetedComponentExecutionPhase::Prefill {
            activation_batch_width: 64,
        },
    )
    .unwrap();

    assert_eq!(decode.component_id, component_id);
    assert_eq!(decode.component_ids, [component_id.clone()]);
    assert_eq!(decode.terminal_node_id, decode_terminal);
    assert_eq!(prefill.component_id, component_id);
    assert_eq!(prefill.terminal_node_id, decode_terminal);
    assert_ne!(decode.signature_id, prefill.signature_id);
}

#[test]
fn incomplete_prefill_transaction_is_not_published_as_calibration_evidence() {
    let mut runtime_model = fixture_model_runtime_model();
    let causal_scan = runtime_model.component_executions[0]
        .kernels
        .iter_mut()
        .find(|kernel| kernel.batch_mode == VulkanResidentComponentKernelBatchMode::CausalScan)
        .unwrap();
    causal_scan
        .batch_implementations
        .iter_mut()
        .for_each(|implementation| implementation.lane_tile_width = 1);
    let targets = vulkan_runtime_placement_calibration_targets_for_phase(
        &runtime_model,
        VulkanTargetedComponentExecutionPhase::Prefill {
            activation_batch_width: 64,
        },
    )
    .unwrap();

    assert!(targets.is_empty());
}

#[test]
fn complete_compiler_declared_causal_prefill_is_published() {
    let runtime_model = fixture_model_runtime_model();
    let targets = vulkan_runtime_placement_calibration_targets_for_phase(
        &runtime_model,
        VulkanTargetedComponentExecutionPhase::Prefill {
            activation_batch_width: 64,
        },
    )
    .unwrap();

    assert!(!targets.is_empty());
    let expected_occurrences = runtime_model
        .circuit_graph
        .components
        .iter()
        .filter(|component| component.runtime_role.is_signal_processor())
        .count();
    assert_eq!(
        targets
            .iter()
            .map(|target| target.component_ids.len())
            .sum::<usize>(),
        expected_occurrences,
    );
    assert_eq!(
        targets[0].terminal_node_id,
        runtime_model.component_executions[0]
            .kernels
            .last()
            .unwrap()
            .node_id,
    );
}

#[test]
fn scalar_lane_prefill_fallback_is_published_as_a_complete_transaction() {
    let mut runtime_model = fixture_model_runtime_model();
    for kernel in &mut runtime_model.component_executions[0].kernels {
        kernel.batch_mode = VulkanResidentComponentKernelBatchMode::SerialLanes;
        kernel.batch_implementations.clear();
    }

    let targets = vulkan_runtime_placement_calibration_targets_for_phase(
        &runtime_model,
        VulkanTargetedComponentExecutionPhase::Prefill {
            activation_batch_width: 4,
        },
    )
    .unwrap();

    assert!(!targets.is_empty());
}

#[test]
fn prefill_calibration_identity_tracks_the_selected_batch_width() {
    let runtime_model = fixture_model_runtime_model();
    let component_id = runtime_model.component_executions[0].component_id.clone();
    let narrow = vulkan_runtime_placement_calibration_target_for_component(
        &runtime_model,
        &component_id,
        VulkanTargetedComponentExecutionPhase::Prefill {
            activation_batch_width: 4,
        },
    )
    .unwrap();
    let wide = vulkan_runtime_placement_calibration_target_for_component(
        &runtime_model,
        &component_id,
        VulkanTargetedComponentExecutionPhase::Prefill {
            activation_batch_width: 64,
        },
    )
    .unwrap();

    assert_ne!(narrow.signature_id, wide.signature_id);
}

#[test]
fn prefill_wider_than_a_causal_scan_tile_is_not_published() {
    let runtime_model = fixture_model_runtime_model();
    let targets = vulkan_runtime_placement_calibration_targets_for_phase(
        &runtime_model,
        VulkanTargetedComponentExecutionPhase::Prefill {
            activation_batch_width: 65,
        },
    )
    .unwrap();

    assert!(targets.is_empty());
}

#[test]
fn runtime_placement_calibration_rejects_zero_width_prefill_target() {
    let runtime_model = fixture_model_runtime_model();
    let component_id = runtime_model.component_executions[0].component_id.clone();
    let error = vulkan_runtime_placement_calibration_target_for_component(
        &runtime_model,
        &component_id,
        VulkanTargetedComponentExecutionPhase::Prefill {
            activation_batch_width: 0,
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("positive prefill batch width"));
}

fn fixture_placement_costs(costs: &[(&str, &str, u64)]) -> VulkanRuntimePlacementCostModel {
    VulkanRuntimePlacementCostModel {
        component_execution: costs
            .iter()
            .map(|(device_id, component_id, cost)| {
                (
                    ((*device_id).to_string(), (*component_id).to_string()),
                    (format!("signature-{component_id}"), *cost),
                )
            })
            .collect(),
        boundary_transfer_ns: BTreeMap::new(),
        default_graph_compatible_devices: costs
            .iter()
            .map(|(device_id, _, _)| (*device_id).to_string())
            .collect(),
    }
}

fn fixture_empty_placement_boundaries(
    component_count: usize,
) -> Vec<VulkanRuntimePlacementBoundary> {
    (1..component_count)
        .map(|_| VulkanRuntimePlacementBoundary {
            transfers: Vec::new(),
        })
        .collect()
}

#[test]
fn cost_aware_contiguous_placement_jointly_selects_device_order_and_boundary() {
    let components = ["a", "b", "c", "d"].map(|component_id| CapacityPackedPlacementComponent {
        component_id: component_id.to_string(),
        resident_weight_bytes: 1,
    });
    let candidates = [
        VulkanRuntimePlacementCandidate {
            device_id: "tail-fast".to_string(),
            safe_capacity_bytes: 3,
        },
        VulkanRuntimePlacementCandidate {
            device_id: "head-fast".to_string(),
            safe_capacity_bytes: 3,
        },
    ];
    let costs = fixture_placement_costs(&[
        ("head-fast", "a", 1),
        ("head-fast", "b", 100),
        ("head-fast", "c", 100),
        ("head-fast", "d", 100),
        ("tail-fast", "a", 100),
        ("tail-fast", "b", 1),
        ("tail-fast", "c", 1),
        ("tail-fast", "d", 1),
    ]);

    let boundaries = fixture_empty_placement_boundaries(components.len());
    let placed = cost_aware_contiguous_component_placement(
        &components,
        &candidates,
        &costs,
        &boundaries,
        None,
    )
    .unwrap();

    assert_eq!(placed.ordered_device_ids, ["head-fast", "tail-fast"]);
    assert_eq!(placed.placement["a"], "head-fast");
    assert_eq!(placed.placement["b"], "tail-fast");
    assert_eq!(placed.placement["c"], "tail-fast");
    assert_eq!(placed.placement["d"], "tail-fast");
    assert_eq!(placed.predicted_execution_ns, 4);
}

#[test]
fn cost_aware_placement_uses_a_sparse_component_compatibility_matrix() {
    let components = ["a", "b", "c", "d"].map(|component_id| CapacityPackedPlacementComponent {
        component_id: component_id.to_string(),
        resident_weight_bytes: 1,
    });
    let candidates = [
        VulkanRuntimePlacementCandidate {
            device_id: "head".to_string(),
            safe_capacity_bytes: 4,
        },
        VulkanRuntimePlacementCandidate {
            device_id: "tail".to_string(),
            safe_capacity_bytes: 4,
        },
    ];
    let mut costs = fixture_placement_costs(&[
        ("head", "a", 1),
        ("head", "b", 1),
        ("tail", "c", 1),
        ("tail", "d", 1),
    ]);
    costs.default_graph_compatible_devices = BTreeSet::from(["head".to_string()]);

    let placed = cost_aware_contiguous_component_placement(
        &components,
        &candidates,
        &costs,
        &fixture_empty_placement_boundaries(components.len()),
        None,
    )
    .unwrap();

    assert_eq!(placed.ordered_device_ids, ["head", "tail"]);
    assert_eq!(placed.placement["a"], "head");
    assert_eq!(placed.placement["b"], "head");
    assert_eq!(placed.placement["c"], "tail");
    assert_eq!(placed.placement["d"], "tail");
    assert_eq!(placed.predicted_execution_ns, 4);
}

#[test]
fn cost_aware_placement_rejects_noncontiguous_partial_device_coverage() {
    let components = ["a", "b", "c", "d"].map(|component_id| CapacityPackedPlacementComponent {
        component_id: component_id.to_string(),
        resident_weight_bytes: 1,
    });
    let candidates = [
        VulkanRuntimePlacementCandidate {
            device_id: "alternating-a".to_string(),
            safe_capacity_bytes: 4,
        },
        VulkanRuntimePlacementCandidate {
            device_id: "alternating-b".to_string(),
            safe_capacity_bytes: 4,
        },
    ];
    let costs = fixture_placement_costs(&[
        ("alternating-a", "a", 1),
        ("alternating-a", "c", 1),
        ("alternating-b", "b", 1),
        ("alternating-b", "d", 1),
    ]);

    let error = cost_aware_contiguous_component_placement(
        &components,
        &candidates,
        &costs,
        &fixture_empty_placement_boundaries(components.len()),
        None,
    )
    .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("no cost-aware contiguous placement")
    );
}

#[test]
fn normalized_device_cost_compares_partial_targets_by_mean_component_cost() {
    let costs = fixture_placement_costs(&[
        ("partial-fast", "a", 10),
        ("partial-fast", "b", 20),
        ("complete-slow", "a", 20),
        ("complete-slow", "b", 20),
        ("complete-slow", "c", 20),
        ("complete-slow", "d", 20),
    ]);

    assert_eq!(
        costs
            .normalized_device_execution_ns("partial-fast", 4)
            .unwrap(),
        60,
    );
    assert_eq!(
        costs
            .normalized_device_execution_ns("complete-slow", 4)
            .unwrap(),
        80,
    );
}

#[test]
fn measured_candidate_subsets_include_nonprefix_combinations_once() {
    let candidates = ["a", "b", "c"].map(|device_id| VulkanRuntimePlacementCandidate {
        device_id: device_id.to_string(),
        safe_capacity_bytes: 1,
    });

    let subsets = runtime_placement_candidate_subsets(&candidates, 2)
        .unwrap()
        .into_iter()
        .map(|subset| {
            subset
                .into_iter()
                .map(|candidate| candidate.device_id)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    assert_eq!(
        subsets,
        [
            vec!["a".to_string(), "b".to_string()],
            vec!["a".to_string(), "c".to_string()],
            vec!["b".to_string(), "c".to_string()],
        ],
    );
    assert!(runtime_placement_candidate_subsets(&candidates, 0).is_err());
    assert!(runtime_placement_candidate_subsets(&candidates, 4).is_err());
}

#[test]
fn runtime_placement_discovers_exact_graph_boundary_payloads() {
    let runtime_model = fixture_model_runtime_model_with_colocated_three_layer_series();
    let boundaries = vulkan_runtime_placement_boundaries(&runtime_model).unwrap();
    let byte_counts = vulkan_runtime_placement_transfer_byte_counts(&runtime_model).unwrap();

    assert_eq!(boundaries.len(), 2);
    assert!(
        boundaries
            .iter()
            .all(|boundary| !boundary.transfers.is_empty())
    );
    assert_eq!(
        byte_counts,
        boundaries
            .iter()
            .flat_map(|boundary| &boundary.transfers)
            .map(|transfer| transfer.byte_count)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>(),
    );
}

#[test]
fn automatic_cost_placement_rejects_a_non_chain_graph_without_flattening_its_wiring() {
    let mut runtime_model = fixture_model_runtime_model_with_colocated_three_layer_series();
    let processor_ids = runtime_model
        .circuit_graph
        .components
        .iter()
        .filter(|component| component.runtime_role.is_signal_processor())
        .map(|component| component.component_id.clone())
        .collect::<Vec<_>>();
    let mut nonlocal = runtime_model
        .circuit_graph
        .edges
        .iter()
        .find(|edge| {
            edge.source.component_id == processor_ids[0]
                && edge.destination.component_id == processor_ids[1]
        })
        .unwrap()
        .clone();
    nonlocal.id = "nonlocal-fixture-edge".to_string();
    nonlocal.destination = runtime_model
        .circuit_graph
        .edges
        .iter()
        .find(|edge| edge.destination.component_id == processor_ids[2])
        .unwrap()
        .destination
        .clone();
    runtime_model.circuit_graph.edges.push(nonlocal);

    let error = vulkan_runtime_placement_boundaries(&runtime_model).unwrap_err();

    assert!(error.to_string().contains("nearest-neighbor"));
    assert!(error.to_string().contains("use explicit wiring"));
}

#[test]
fn cost_aware_placement_uses_payload_specific_directional_transfer_costs() {
    let components = ["a", "b", "c"].map(|component_id| CapacityPackedPlacementComponent {
        component_id: component_id.to_string(),
        resident_weight_bytes: 1,
    });
    let candidates = [
        VulkanRuntimePlacementCandidate {
            device_id: "a-device".to_string(),
            safe_capacity_bytes: 2,
        },
        VulkanRuntimePlacementCandidate {
            device_id: "b-device".to_string(),
            safe_capacity_bytes: 2,
        },
    ];
    let mut costs = fixture_placement_costs(&[
        ("a-device", "a", 1),
        ("a-device", "b", 1),
        ("a-device", "c", 1),
        ("b-device", "a", 1),
        ("b-device", "b", 1),
        ("b-device", "c", 1),
    ]);
    for (source, target, bytes, nanoseconds) in [
        ("a-device", "b-device", 16, 1),
        ("b-device", "a-device", 16, 100),
        ("a-device", "b-device", 32, 100),
        ("b-device", "a-device", 32, 50),
    ] {
        costs
            .record_boundary_transfer_cost(source, target, bytes, nanoseconds)
            .unwrap();
    }
    let boundaries = [
        VulkanRuntimePlacementBoundary {
            transfers: vec![VulkanRuntimePlacementBoundaryTransfer {
                source_in_prefix: true,
                byte_count: 16,
            }],
        },
        VulkanRuntimePlacementBoundary {
            transfers: vec![VulkanRuntimePlacementBoundaryTransfer {
                source_in_prefix: false,
                byte_count: 32,
            }],
        },
    ];

    let placed = cost_aware_contiguous_component_placement(
        &components,
        &candidates,
        &costs,
        &boundaries,
        None,
    )
    .unwrap();

    assert_eq!(placed.ordered_device_ids, ["a-device", "b-device"]);
    assert_eq!(placed.placement["a"], "a-device");
    assert_eq!(placed.placement["b"], "b-device");
    assert_eq!(placed.placement["c"], "b-device");
    assert_eq!(placed.predicted_execution_ns, 4);
}

#[test]
fn cost_aware_placement_rejects_an_unmeasured_boundary_instead_of_assuming_zero() {
    let costs = VulkanRuntimePlacementCostModel::default();
    let boundary = VulkanRuntimePlacementBoundary {
        transfers: vec![VulkanRuntimePlacementBoundaryTransfer {
            source_in_prefix: true,
            byte_count: 32,
        }],
    };

    let error =
        runtime_placement_boundary_cost_ns(&boundary, "source", "target", &costs).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("no measured 32-byte boundary cost")
    );
}

#[test]
fn cost_aware_paged_placement_prevents_fast_cache_from_claiming_every_component() {
    let components = ["a", "b", "c", "d"].map(|component_id| CapacityPackedPlacementComponent {
        component_id: component_id.to_string(),
        resident_weight_bytes: 10,
    });
    let candidates = [
        VulkanRuntimePlacementCandidate {
            device_id: "fast".to_string(),
            safe_capacity_bytes: 100,
        },
        VulkanRuntimePlacementCandidate {
            device_id: "slow".to_string(),
            safe_capacity_bytes: 100,
        },
    ];
    let costs = fixture_placement_costs(&[
        ("fast", "a", 1),
        ("fast", "b", 1),
        ("fast", "c", 1),
        ("fast", "d", 1),
        ("slow", "a", 100),
        ("slow", "b", 100),
        ("slow", "c", 100),
        ("slow", "d", 100),
    ]);

    let balance = VulkanRuntimePagedPlacementBalance {
        component_weights: components
            .iter()
            .map(|component| component.resident_weight_bytes as u128)
            .collect(),
        input_auxiliary_weight_bytes: 0,
        output_auxiliary_weight_bytes: 0,
    };
    let placed = cost_aware_contiguous_component_placement(
        &components,
        &candidates,
        &costs,
        &fixture_empty_placement_boundaries(components.len()),
        Some(&balance),
    )
    .unwrap();

    let fast_count = placed
        .placement
        .values()
        .filter(|device_id| device_id.as_str() == "fast")
        .count();
    assert_eq!(fast_count, 3);
    assert!(
        placed
            .placement
            .values()
            .any(|device_id| device_id == "slow")
    );
}

#[test]
fn cost_aware_paged_placement_cannot_strand_material_retained_capacity() {
    let components = (0..10)
        .map(|index| CapacityPackedPlacementComponent {
            component_id: format!("component-{index}"),
            resident_weight_bytes: 10,
        })
        .collect::<Vec<_>>();
    let candidates = [
        VulkanRuntimePlacementCandidate {
            device_id: "fast".to_string(),
            safe_capacity_bytes: 100,
        },
        VulkanRuntimePlacementCandidate {
            device_id: "slow".to_string(),
            safe_capacity_bytes: 100,
        },
    ];
    let mut entries = Vec::new();
    for component in &components {
        entries.push(("fast", component.component_id.as_str(), 1));
        entries.push(("slow", component.component_id.as_str(), 100));
    }
    let costs = fixture_placement_costs(&entries);

    let balance = VulkanRuntimePagedPlacementBalance {
        component_weights: components
            .iter()
            .map(|component| component.resident_weight_bytes as u128)
            .collect(),
        input_auxiliary_weight_bytes: 0,
        output_auxiliary_weight_bytes: 0,
    };
    let placed = cost_aware_contiguous_component_placement(
        &components,
        &candidates,
        &costs,
        &fixture_empty_placement_boundaries(components.len()),
        Some(&balance),
    )
    .unwrap();

    let fast_count = placed
        .placement
        .values()
        .filter(|device_id| device_id.as_str() == "fast")
        .count();
    assert!((4..=6).contains(&fast_count));
}

#[test]
fn demand_paged_subset_expands_while_another_device_can_reduce_paging() {
    const LAYER_BYTES: u128 = 843_498_240;
    const DEVICE_BYTES: usize = 27 * 1024 * 1024 * 1024;
    let balance = VulkanRuntimePagedPlacementBalance {
        component_weights: vec![LAYER_BYTES; 40],
        input_auxiliary_weight_bytes: 0,
        output_auxiliary_weight_bytes: 0,
    };
    let candidates = [
        VulkanRuntimePlacementCandidate {
            device_id: "first".to_string(),
            safe_capacity_bytes: DEVICE_BYTES,
        },
        VulkanRuntimePlacementCandidate {
            device_id: "second".to_string(),
            safe_capacity_bytes: DEVICE_BYTES,
        },
    ];

    assert!(demand_paged_subset_has_addressable_shortfall(&balance, &candidates[..1],).unwrap());
    assert!(!demand_paged_subset_has_addressable_shortfall(&balance, &candidates,).unwrap());
}

#[test]
fn demand_paged_subset_accounts_for_endpoint_auxiliary_residency() {
    let balance = VulkanRuntimePagedPlacementBalance {
        component_weights: vec![40, 40],
        input_auxiliary_weight_bytes: 15,
        output_auxiliary_weight_bytes: 15,
    };
    let candidates = [
        VulkanRuntimePlacementCandidate {
            device_id: "first".to_string(),
            safe_capacity_bytes: 100,
        },
        VulkanRuntimePlacementCandidate {
            device_id: "second".to_string(),
            safe_capacity_bytes: 100,
        },
    ];

    assert!(demand_paged_subset_has_addressable_shortfall(&balance, &candidates[..1],).unwrap());
    assert!(!demand_paged_subset_has_addressable_shortfall(&balance, &candidates,).unwrap());
}

#[test]
fn demand_paged_subset_reports_shortfall_even_when_no_larger_set_exists() {
    let balance = VulkanRuntimePagedPlacementBalance {
        component_weights: vec![100, 100],
        input_auxiliary_weight_bytes: 0,
        output_auxiliary_weight_bytes: 0,
    };
    let candidate = VulkanRuntimePlacementCandidate {
        device_id: "only".to_string(),
        safe_capacity_bytes: 100,
    };

    assert!(demand_paged_subset_has_addressable_shortfall(&balance, &[candidate],).unwrap());
}

#[test]
fn cost_aware_paged_placement_reserves_auxiliary_graphs_on_the_endpoint() {
    let components = (0..6)
        .map(|index| CapacityPackedPlacementComponent {
            component_id: format!("component-{index}"),
            resident_weight_bytes: 10,
        })
        .collect::<Vec<_>>();
    let candidates = [
        VulkanRuntimePlacementCandidate {
            device_id: "small-fast".to_string(),
            safe_capacity_bytes: 30,
        },
        VulkanRuntimePlacementCandidate {
            device_id: "large-slow".to_string(),
            safe_capacity_bytes: 100,
        },
    ];
    let mut entries = Vec::new();
    for component in &components {
        entries.push(("small-fast", component.component_id.as_str(), 1));
        entries.push(("large-slow", component.component_id.as_str(), 100));
    }
    let costs = fixture_placement_costs(&entries);
    let balance = VulkanRuntimePagedPlacementBalance {
        component_weights: vec![10; components.len()],
        input_auxiliary_weight_bytes: 0,
        output_auxiliary_weight_bytes: 40,
    };

    let placed = cost_aware_contiguous_component_placement(
        &components,
        &candidates,
        &costs,
        &fixture_empty_placement_boundaries(components.len()),
        Some(&balance),
    )
    .unwrap();

    assert_eq!(placed.ordered_device_ids, ["small-fast", "large-slow"]);
    assert_eq!(placed.placement["component-5"], "large-slow");
    let small_count = placed
        .placement
        .values()
        .filter(|device_id| device_id.as_str() == "small-fast")
        .count();
    assert!((1..=3).contains(&small_count));
}

#[test]
fn observed_working_set_rebalance_moves_only_a_profitable_contiguous_boundary() {
    let runtime_model = fixture_model_runtime_model_with_colocated_three_layer_series();
    let component_ids = runtime_model
        .circuit_graph
        .components
        .iter()
        .filter(|component| component.runtime_role.is_signal_processor())
        .map(|component| component.component_id.clone())
        .collect::<Vec<_>>();
    let current = vulkan_runtime_model_with_component_placement(
        &runtime_model,
        "first",
        &BTreeMap::from([
            (component_ids[0].clone(), "first".to_string()),
            (component_ids[1].clone(), "first".to_string()),
            (component_ids[2].clone(), "third".to_string()),
        ]),
    )
    .unwrap();
    let candidates = [
        VulkanRuntimePlacementCandidate {
            device_id: "first".to_string(),
            safe_capacity_bytes: 32 * 1024 * 1024 * 1024,
        },
        VulkanRuntimePlacementCandidate {
            device_id: "second".to_string(),
            safe_capacity_bytes: 32 * 1024 * 1024 * 1024,
        },
        VulkanRuntimePlacementCandidate {
            device_id: "third".to_string(),
            safe_capacity_bytes: 32 * 1024 * 1024 * 1024,
        },
    ];
    let targets = vulkan_runtime_placement_calibration_targets(&current).unwrap();
    let mut costs = VulkanRuntimePlacementCostModel::default();
    for candidate in &candidates {
        costs
            .record_default_graph_compatibility(&candidate.device_id)
            .unwrap();
        for target in &targets {
            costs
                .record_calibration(&candidate.device_id, target, 10)
                .unwrap();
        }
    }
    for byte_count in vulkan_runtime_placement_transfer_byte_counts(&current).unwrap() {
        for source in ["first", "second", "third"] {
            for destination in ["first", "second", "third"] {
                if source != destination {
                    costs
                        .record_boundary_transfer_cost(source, destination, byte_count, 1)
                        .unwrap();
                }
            }
        }
    }
    let pressure_component = |component_id: &str, selected_payload_bytes: usize| {
        VulkanRuntimeComponentWorkingSetPressure {
            execution_scope: "target".to_string(),
            component_id: component_id.to_string(),
            selected_unit_count: usize::from(selected_payload_bytes > 0),
            selected_payload_bytes,
            selection_count: selected_payload_bytes as u64,
            ..Default::default()
        }
    };
    let cumulative = VulkanRuntimeWorkingSetPressureSnapshot {
        stores: vec![
            VulkanRuntimeDeviceWorkingSetPressure {
                store_id: "first-store".to_string(),
                physical_device_id: "first".to_string(),
                logical_device_ids: vec!["first".to_string()],
                components: vec![
                    pressure_component(&component_ids[0], 1),
                    pressure_component(&component_ids[1], 100),
                ],
                ..Default::default()
            },
            VulkanRuntimeDeviceWorkingSetPressure {
                store_id: "third-store".to_string(),
                physical_device_id: "third".to_string(),
                logical_device_ids: vec!["third".to_string()],
                components: vec![pressure_component(&component_ids[2], 1)],
                ..Default::default()
            },
        ],
    };
    let mut interval = cumulative.clone();
    interval.stores[0].eviction_count = 1;
    interval.stores[0].reload_count = 1;
    interval.stores[0].blocking_time_ns = 1_000_000;

    let rebalanced = rebalance_demand_paged_vulkan_runtime_model_from_working_set(
        tiny_model_dir(),
        &current,
        &candidates,
        &costs,
        &cumulative,
        &interval,
        100,
        8,
        0,
    )
    .unwrap()
    .expect("observed churn should repay one boundary move");

    assert_eq!(rebalanced.moved_component_ids, [component_ids[1].clone()]);
    assert_eq!(
        rebalanced
            .placement
            .runtime_model
            .runtime_graph
            .instances
            .iter()
            .find(|instance| instance.instance_id == component_ids[0])
            .unwrap()
            .device_id,
        "first",
    );
    assert_eq!(
        rebalanced
            .placement
            .runtime_model
            .runtime_graph
            .instances
            .iter()
            .filter(|instance| component_ids.contains(&instance.instance_id))
            .map(|instance| instance.device_id.as_str())
            .collect::<Vec<_>>(),
        ["first", "second", "third"],
    );
    assert_eq!(
        rebalanced.retained_logical_device_ids,
        ["third".to_string()].into_iter().collect(),
    );
    assert!(rebalanced.estimated_net_benefit_ns > 0);
}

#[test]
fn placement_cost_model_rejects_a_changed_compiled_execution_signature() {
    let mut runtime_model = fixture_model_runtime_model();
    let targets = vulkan_runtime_placement_calibration_targets(&runtime_model).unwrap();
    let candidate = VulkanRuntimePlacementCandidate {
        device_id: "physical-a".to_string(),
        safe_capacity_bytes: 1,
    };
    let mut costs = VulkanRuntimePlacementCostModel::default();
    costs
        .record_default_graph_compatibility(&candidate.device_id)
        .unwrap();
    for target in &targets {
        costs
            .record_calibration(&candidate.device_id, target, 1)
            .unwrap();
    }
    costs
        .validate_runtime_model(&runtime_model, std::slice::from_ref(&candidate))
        .unwrap();

    runtime_model.component_executions[0].kernels[0].workgroup_count_x += 1;
    let error = costs
        .validate_runtime_model(&runtime_model, &[candidate])
        .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("different compiled execution signature")
    );
}

#[test]
fn placement_cost_model_accepts_exact_sparse_device_coverage() {
    let runtime_model = fixture_model_runtime_model_with_colocated_three_layer_series();
    let targets = vulkan_runtime_placement_calibration_targets(&runtime_model).unwrap();
    let candidates = [
        VulkanRuntimePlacementCandidate {
            device_id: "first".to_string(),
            safe_capacity_bytes: 1,
        },
        VulkanRuntimePlacementCandidate {
            device_id: "second".to_string(),
            safe_capacity_bytes: 1,
        },
    ];
    let all_component_ids = targets
        .iter()
        .flat_map(|target| target.component_ids.iter().cloned())
        .collect::<Vec<_>>();
    assert!(all_component_ids.len() >= 2);
    let midpoint = all_component_ids.len() / 2;
    let mut costs = VulkanRuntimePlacementCostModel::default();
    costs.record_default_graph_compatibility("first").unwrap();
    for (device_id, selected_ids) in [
        ("first", &all_component_ids[..midpoint]),
        ("second", &all_component_ids[midpoint..]),
    ] {
        for component_id in selected_ids {
            let target = targets
                .iter()
                .find(|target| target.component_ids.contains(component_id))
                .unwrap();
            let mut selected_target = target.clone();
            selected_target.component_ids = vec![component_id.clone()];
            costs
                .record_calibration(device_id, &selected_target, 1)
                .unwrap();
        }
    }
    for byte_count in vulkan_runtime_placement_transfer_byte_counts(&runtime_model).unwrap() {
        costs
            .record_boundary_transfer_cost("first", "second", byte_count, 1)
            .unwrap();
        costs
            .record_boundary_transfer_cost("second", "first", byte_count, 1)
            .unwrap();
    }

    costs
        .validate_runtime_model(&runtime_model, &candidates)
        .unwrap();
}

#[test]
fn placement_cost_model_rejects_a_component_uncovered_by_every_device() {
    let runtime_model = fixture_model_runtime_model_with_colocated_three_layer_series();
    let targets = vulkan_runtime_placement_calibration_targets(&runtime_model).unwrap();
    let candidate = VulkanRuntimePlacementCandidate {
        device_id: "partial".to_string(),
        safe_capacity_bytes: 1,
    };
    let all_component_ids = targets
        .iter()
        .flat_map(|target| target.component_ids.iter().cloned())
        .collect::<Vec<_>>();
    assert!(all_component_ids.len() >= 2);
    let mut costs = VulkanRuntimePlacementCostModel::default();
    costs
        .record_default_graph_compatibility(&candidate.device_id)
        .unwrap();
    for component_id in &all_component_ids[..all_component_ids.len() - 1] {
        let target = targets
            .iter()
            .find(|target| target.component_ids.contains(component_id))
            .unwrap();
        let mut selected_target = target.clone();
        selected_target.component_ids = vec![component_id.clone()];
        costs
            .record_calibration(&candidate.device_id, &selected_target, 1)
            .unwrap();
    }

    let error = costs
        .validate_runtime_model(&runtime_model, &[candidate])
        .unwrap_err();

    assert!(error.to_string().contains("cannot execute components"));
    assert!(
        error
            .to_string()
            .contains(all_component_ids.last().unwrap())
    );
}

#[test]
fn placement_cost_model_requires_a_default_graph_host() {
    let runtime_model = fixture_model_runtime_model();
    let candidate = VulkanRuntimePlacementCandidate {
        device_id: "interior-only".to_string(),
        safe_capacity_bytes: 1,
    };
    let mut costs = VulkanRuntimePlacementCostModel::default();
    for target in vulkan_runtime_placement_calibration_targets(&runtime_model).unwrap() {
        costs
            .record_calibration(&candidate.device_id, &target, 1)
            .unwrap();
    }

    let error = costs
        .validate_runtime_model(&runtime_model, &[candidate])
        .unwrap_err();

    assert!(error.to_string().contains("no candidate compatible"));
    assert!(error.to_string().contains("input/output graph"));
}

#[test]
fn explicit_component_placement_is_the_shared_calibration_and_runtime_transform() {
    let runtime_model = fixture_model_runtime_model_with_colocated_three_layer_series();
    let component_ids = runtime_model
        .runtime_graph
        .instances
        .iter()
        .map(|instance| instance.instance_id.clone())
        .collect::<Vec<_>>();
    let placed = vulkan_runtime_model_with_component_placement(
        &runtime_model,
        "physical-owner",
        &component_ids
            .iter()
            .cloned()
            .map(|component_id| (component_id, "physical-owner".to_string()))
            .collect(),
    )
    .unwrap();

    assert!(
        placed
            .runtime_graph
            .instances
            .iter()
            .all(|instance| instance.device_id == "physical-owner")
    );
    assert_eq!(placed.placement_device_ids(), ["physical-owner"]);
}

#[test]
fn runtime_auto_placement_spills_once_instead_of_balancing_across_extra_devices() {
    let runtime_model = fixture_model_runtime_model_with_colocated_three_layer_series();
    let tensor_index = runtime_model
        .load_runtime_tensor_index(tiny_model_dir())
        .unwrap();
    let baseline = plan_vulkan_runtime_residency(
        tiny_model_dir(),
        &runtime_model,
        &tensor_index,
        8,
        0,
        ResourceResidencyPolicy::DemandRetained,
    )
    .unwrap();
    let one_device_required = maximum_resident_bytes(&baseline, "gpu0");

    let placed = capacity_pack_vulkan_runtime_model(
        tiny_model_dir(),
        &runtime_model,
        &tensor_index,
        &[
            VulkanRuntimePlacementCandidate {
                device_id: "first".to_string(),
                safe_capacity_bytes: one_device_required - 1,
            },
            VulkanRuntimePlacementCandidate {
                device_id: "second".to_string(),
                safe_capacity_bytes: one_device_required,
            },
            VulkanRuntimePlacementCandidate {
                device_id: "third".to_string(),
                safe_capacity_bytes: one_device_required,
            },
        ],
        8,
        0,
        ResourceResidencyPolicy::DemandRetained,
    )
    .unwrap();

    assert_eq!(placed.selected_device_ids, ["first", "second"]);
    assert_eq!(
        placed.runtime_model.placement_device_ids(),
        ["first", "second"]
    );
    let signal_devices = placed
        .runtime_model
        .circuit_graph
        .components
        .iter()
        .filter(|component| component.runtime_role.is_signal_processor())
        .map(|component| {
            placed
                .runtime_model
                .placement
                .device_for_component(&component.component_id)
        })
        .collect::<Vec<_>>();
    assert_eq!(signal_devices, ["first", "first", "second"]);
}

#[test]
fn measured_auto_placement_can_select_a_nonprefix_single_device() {
    let runtime_model = fixture_model_runtime_model_with_colocated_three_layer_series();
    let tensor_index = runtime_model
        .load_runtime_tensor_index(tiny_model_dir())
        .unwrap();
    let baseline = plan_vulkan_runtime_residency(
        tiny_model_dir(),
        &runtime_model,
        &tensor_index,
        8,
        0,
        ResourceResidencyPolicy::DemandRetained,
    )
    .unwrap();
    let required = maximum_resident_bytes(&baseline, "gpu0");
    let candidates = [
        VulkanRuntimePlacementCandidate {
            device_id: "ranked-first-partial".to_string(),
            safe_capacity_bytes: required,
        },
        VulkanRuntimePlacementCandidate {
            device_id: "ranked-second-partial".to_string(),
            safe_capacity_bytes: required,
        },
        VulkanRuntimePlacementCandidate {
            device_id: "complete".to_string(),
            safe_capacity_bytes: required,
        },
    ];
    let targets = vulkan_runtime_placement_calibration_targets(&runtime_model).unwrap();
    let component_ids = targets
        .iter()
        .flat_map(|target| target.component_ids.iter().cloned())
        .collect::<Vec<_>>();
    assert!(component_ids.len() >= 3);
    let mut costs = VulkanRuntimePlacementCostModel::default();
    for device_id in ["ranked-first-partial", "complete"] {
        costs.record_default_graph_compatibility(device_id).unwrap();
    }
    for (device_id, selected_component_ids) in [
        ("ranked-first-partial", component_ids[..1].to_vec()),
        (
            "ranked-second-partial",
            component_ids[component_ids.len() - 1..].to_vec(),
        ),
        ("complete", component_ids.clone()),
    ] {
        for component_id in selected_component_ids {
            let target = targets
                .iter()
                .find(|target| target.component_ids.contains(&component_id))
                .unwrap();
            let mut selected_target = target.clone();
            selected_target.component_ids = vec![component_id];
            costs
                .record_calibration(device_id, &selected_target, 1)
                .unwrap();
        }
    }
    for byte_count in vulkan_runtime_placement_transfer_byte_counts(&runtime_model).unwrap() {
        for source in &candidates {
            for destination in &candidates {
                if source.device_id != destination.device_id {
                    costs
                        .record_boundary_transfer_cost(
                            &source.device_id,
                            &destination.device_id,
                            byte_count,
                            1,
                        )
                        .unwrap();
                }
            }
        }
    }

    let placed = capacity_pack_vulkan_runtime_model_with_costs(
        tiny_model_dir(),
        &runtime_model,
        &tensor_index,
        &candidates,
        Some(&costs),
        8,
        0,
        ResourceResidencyPolicy::DemandRetained,
    )
    .unwrap();

    assert_eq!(placed.selected_device_ids, ["complete"]);
    assert!(
        placed
            .runtime_model
            .placement_device_ids()
            .iter()
            .all(|device_id| device_id == "complete")
    );
}

#[test]
fn representation_selection_converges_across_heterogeneous_placement() {
    let runtime_model = fixture_model_runtime_model_with_colocated_three_layer_series();
    let tensor_index = runtime_model
        .load_runtime_tensor_index(tiny_model_dir())
        .unwrap();
    let baseline = plan_vulkan_runtime_residency(
        tiny_model_dir(),
        &runtime_model,
        &tensor_index,
        8,
        0,
        ResourceResidencyPolicy::DemandRetained,
    )
    .unwrap();
    let one_device_required = maximum_resident_bytes(&baseline, "gpu0");
    let primary = auto_placement_hardware_profile("primary", "primary-architecture");
    let spill = auto_placement_hardware_profile("spill", "spill-architecture");
    assert_ne!(primary.capability_class, spill.capability_class);

    let placed = capacity_pack_and_select_vulkan_runtime_model(
        tiny_model_dir(),
        &runtime_model,
        &[
            VulkanRuntimePlacementCandidate {
                device_id: "primary".to_string(),
                safe_capacity_bytes: one_device_required - 1,
            },
            VulkanRuntimePlacementCandidate {
                device_id: "spill".to_string(),
                safe_capacity_bytes: one_device_required,
            },
        ],
        None,
        &BTreeMap::from([
            ("primary".to_string(), primary),
            ("spill".to_string(), spill),
        ]),
        8,
        0,
        ResourceResidencyPolicy::DemandRetained,
        crate::RuntimeExecutionEnvelope {
            phases: vec!["decode".to_string(), "prefill".to_string()],
            activation_batch: crate::RuntimeInclusiveRange {
                minimum: 1,
                maximum: 8,
            },
            context_activations: crate::RuntimeInclusiveRange {
                minimum: 0,
                maximum: 8,
            },
            state_activations: crate::RuntimeInclusiveRange {
                minimum: 0,
                maximum: 8,
            },
            speculative_draft_tokens: 0,
            residency_policy: "demand_retained".to_string(),
        },
    )
    .unwrap();

    assert_eq!(placed.selected_device_ids, ["primary", "spill"]);
    assert_eq!(
        placed.runtime_model.placement_device_ids(),
        ["primary", "spill"]
    );
    assert_eq!(
        placed.exact_runtime_model.placement_device_ids(),
        ["primary", "spill"]
    );
    assert!(
        placed
            .exact_runtime_model
            .implementation_selection
            .is_none()
    );
    let selection = placed.runtime_model.implementation_selection.unwrap();
    assert!(selection.selected.is_empty());
    assert!(!selection.exact_instance_ids.is_empty());
}

#[test]
fn heterogeneous_selection_rejects_a_selected_device_without_a_profile() {
    let runtime_model = fixture_model_runtime_model();
    let tensor_index = runtime_model
        .load_runtime_tensor_index(tiny_model_dir())
        .unwrap();
    let baseline = plan_vulkan_runtime_residency(
        tiny_model_dir(),
        &runtime_model,
        &tensor_index,
        8,
        0,
        ResourceResidencyPolicy::DemandRetained,
    )
    .unwrap();
    let required = vulkan_runtime_maximum_device_resident_bytes(&baseline.device_plans[0]).unwrap();

    let error = capacity_pack_and_select_vulkan_runtime_model(
        tiny_model_dir(),
        &runtime_model,
        &[VulkanRuntimePlacementCandidate {
            device_id: "unprofiled".to_string(),
            safe_capacity_bytes: required,
        }],
        None,
        &BTreeMap::new(),
        8,
        0,
        ResourceResidencyPolicy::DemandRetained,
        crate::RuntimeExecutionEnvelope {
            phases: vec!["decode".to_string()],
            activation_batch: crate::RuntimeInclusiveRange {
                minimum: 1,
                maximum: 1,
            },
            context_activations: crate::RuntimeInclusiveRange {
                minimum: 0,
                maximum: 8,
            },
            state_activations: crate::RuntimeInclusiveRange {
                minimum: 0,
                maximum: 8,
            },
            speculative_draft_tokens: 0,
            residency_policy: "demand_retained".to_string(),
        },
    )
    .unwrap_err();

    assert_eq!(
        error.to_string(),
        "runtime placement device \"unprofiled\" has no hardware profile",
    );
}

#[test]
fn runtime_auto_placement_admits_eventual_lazy_retention_not_only_initial_mount() {
    let runtime_model = fixture_model_runtime_model_with_dynamic_partition(1_000, 64);
    let tensor_index = runtime_model
        .load_runtime_tensor_index(tiny_model_dir())
        .unwrap();
    let demand = plan_vulkan_runtime_residency(
        tiny_model_dir(),
        &runtime_model,
        &tensor_index,
        8,
        0,
        ResourceResidencyPolicy::DemandRetained,
    )
    .unwrap();
    let device = &demand.device_plans[0];
    let maximum = vulkan_runtime_maximum_device_resident_bytes(device).unwrap();
    assert!(maximum > device.initial_device_resident_bytes);

    let error = capacity_pack_vulkan_runtime_model(
        tiny_model_dir(),
        &runtime_model,
        &tensor_index,
        &[VulkanRuntimePlacementCandidate {
            device_id: "too_small_after_warmup".to_string(),
            safe_capacity_bytes: device.initial_device_resident_bytes,
        }],
        8,
        0,
        ResourceResidencyPolicy::DemandRetained,
    )
    .unwrap_err();

    assert!(
        error.0.contains("capacity") || error.0.contains("retain"),
        "unexpected auto-placement error: {error}",
    );
}

#[test]
fn runtime_auto_placement_admits_a_bounded_paged_cache_not_the_virtual_resource_space() {
    let runtime_model = fixture_model_runtime_model_with_dynamic_partition(1_000, 64);
    let tensor_index = runtime_model
        .load_runtime_tensor_index(tiny_model_dir())
        .unwrap();
    let paged = plan_vulkan_runtime_residency(
        tiny_model_dir(),
        &runtime_model,
        &tensor_index,
        8,
        0,
        ResourceResidencyPolicy::DemandPaged,
    )
    .unwrap();
    let device = &paged.device_plans[0];
    let paged_admission = vulkan_runtime_device_capacity_admission_bytes(
        device,
        ResourceResidencyPolicy::DemandPaged,
    )
    .unwrap();
    let retained_admission = vulkan_runtime_device_capacity_admission_bytes(
        device,
        ResourceResidencyPolicy::DemandRetained,
    )
    .unwrap();

    assert_eq!(
        paged_admission,
        device.initial_device_resident_bytes
            + device.resource_store.maximum_load_wave_payload_bytes
            + device
                .resource_store
                .maximum_dynamic_allocation_padding_bytes
    );
    assert!(paged_admission < retained_admission);

    let placed = capacity_pack_vulkan_runtime_model(
        tiny_model_dir(),
        &runtime_model,
        &tensor_index,
        &[VulkanRuntimePlacementCandidate {
            device_id: "paged".to_string(),
            safe_capacity_bytes: paged_admission,
        }],
        8,
        0,
        ResourceResidencyPolicy::DemandPaged,
    )
    .unwrap();
    assert_eq!(placed.selected_device_ids, ["paged"]);

    let error = capacity_pack_vulkan_runtime_model(
        tiny_model_dir(),
        &runtime_model,
        &tensor_index,
        &[VulkanRuntimePlacementCandidate {
            device_id: "too_small_for_one_wave".to_string(),
            safe_capacity_bytes: paged_admission - 1,
        }],
        8,
        0,
        ResourceResidencyPolicy::DemandPaged,
    )
    .unwrap_err();
    assert!(error.to_string().contains("capacity"));
}

#[test]
fn demand_paged_virtual_overcommit_uses_every_cache_in_contiguous_proportion() {
    let placement = proportional_paged_component_placement(
        &[
            CapacityPackedPlacementComponent {
                component_id: "a".to_string(),
                resident_weight_bytes: 30,
            },
            CapacityPackedPlacementComponent {
                component_id: "b".to_string(),
                resident_weight_bytes: 30,
            },
            CapacityPackedPlacementComponent {
                component_id: "c".to_string(),
                resident_weight_bytes: 30,
            },
            CapacityPackedPlacementComponent {
                component_id: "d".to_string(),
                resident_weight_bytes: 30,
            },
            CapacityPackedPlacementComponent {
                component_id: "e".to_string(),
                resident_weight_bytes: 30,
            },
        ],
        &[
            VulkanRuntimePlacementCandidate {
                device_id: "large-a".to_string(),
                safe_capacity_bytes: 40,
            },
            VulkanRuntimePlacementCandidate {
                device_id: "large-b".to_string(),
                safe_capacity_bytes: 40,
            },
            VulkanRuntimePlacementCandidate {
                device_id: "reserved-tail".to_string(),
                safe_capacity_bytes: 20,
            },
        ],
        None,
    )
    .unwrap();

    assert_eq!(placement["a"], "large-a");
    assert_eq!(placement["b"], "large-a");
    assert_eq!(placement["c"], "large-b");
    assert_eq!(placement["d"], "large-b");
    assert_eq!(placement["e"], "reserved-tail");
}

#[test]
fn proportional_paged_placement_reserves_auxiliary_graphs_on_the_endpoint() {
    let components = (0..6)
        .map(|index| CapacityPackedPlacementComponent {
            component_id: format!("component-{index}"),
            resident_weight_bytes: 10,
        })
        .collect::<Vec<_>>();
    let placement = proportional_paged_component_placement(
        &components,
        &[
            VulkanRuntimePlacementCandidate {
                device_id: "input".to_string(),
                safe_capacity_bytes: 50,
            },
            VulkanRuntimePlacementCandidate {
                device_id: "output".to_string(),
                safe_capacity_bytes: 50,
            },
        ],
        Some(&VulkanRuntimePagedPlacementBalance {
            component_weights: vec![10; components.len()],
            input_auxiliary_weight_bytes: 0,
            output_auxiliary_weight_bytes: 40,
        }),
    )
    .unwrap();

    assert_eq!(placement["component-0"], "input");
    assert_eq!(placement["component-4"], "input");
    assert_eq!(placement["component-5"], "output");
}

#[test]
fn runtime_auto_placement_rewires_without_discarding_selected_artifacts() {
    let mut selected = fixture_model_runtime_model();
    selected
        .tensor_index_fragments
        .push(VulkanRuntimeTensorIndexFragment {
            index_path: PathBuf::from("selected/tensors.json"),
            candidate_root: PathBuf::from("selected"),
        });
    let execution_before = selected.component_executions.clone();

    let rewired = vulkan_runtime_model_with_component_placement(
        &selected,
        "physical-a",
        &BTreeMap::from([("layer_00".to_string(), "physical-a".to_string())]),
    )
    .unwrap();

    assert_eq!(
        rewired.tensor_index_fragments,
        selected.tensor_index_fragments
    );
    assert_eq!(rewired.component_executions, execution_before);
    assert_eq!(rewired.placement_device_ids(), ["physical-a"]);
}

#[test]
fn owned_runtime_placement_matches_the_borrowed_transformation() {
    let runtime_model = fixture_model_runtime_model();
    let placement = BTreeMap::from([
        ("layer_00".to_string(), "physical-b".to_string()),
        ("output_transducer".to_string(), "physical-c".to_string()),
    ]);
    let borrowed =
        vulkan_runtime_model_with_component_placement(&runtime_model, "physical-a", &placement)
            .unwrap();
    let owned = vulkan_runtime_model_with_component_placement_owned(
        runtime_model,
        "physical-a",
        &placement,
    )
    .unwrap();

    assert_eq!(owned, borrowed);
}
