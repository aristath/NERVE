fn maximum_resident_bytes(plan: &VulkanRuntimeResidencyPlan, device_id: &str) -> usize {
    vulkan_runtime_maximum_device_resident_bytes(
        plan.device_plans
            .iter()
            .find(|device| device.device_id == device_id)
            .unwrap(),
    )
    .unwrap()
}

#[test]
fn runtime_component_weights_include_target_endpoint_parameters() {
    let runtime_model = fixture_model_runtime_model();
    let tensor_index = runtime_model.load_runtime_tensor_index(tiny_model_dir()).unwrap();
    let components = capacity_packed_runtime_components(&runtime_model, &tensor_index, false)
        .unwrap();
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
    let mut tensor_index = runtime_model.load_runtime_tensor_index(tiny_model_dir()).unwrap();
    let without_draft = capacity_packed_runtime_components(&runtime_model, &tensor_index, false)
        .unwrap();
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
    runtime_model.package.speculative_decoders.push(
        VulkanResidentSpeculativeDecoderPackageSpec {
            id: "draft_fixture".to_string(),
            decoder_type: "fixture".to_string(),
            source_prefix: "draft".to_string(),
            execution_contract: VulkanResidentSpeculativeExecutionContract::AutoregressiveFeedback {
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
        },
    );

    let with_draft = capacity_packed_runtime_components(&runtime_model, &tensor_index, true)
        .unwrap();

    assert_eq!(with_draft.len(), 1);
    assert_eq!(
        with_draft[0].resident_weight_bytes,
        without_draft[0].resident_weight_bytes + draft_tensor_bytes,
    );
}

#[test]
fn runtime_auto_placement_uses_one_device_when_the_complete_retained_set_fits() {
    let runtime_model = fixture_model_runtime_model_with_colocated_three_layer_series();
    let tensor_index = runtime_model.load_runtime_tensor_index(tiny_model_dir()).unwrap();
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
fn runtime_auto_placement_spills_once_instead_of_balancing_across_extra_devices() {
    let runtime_model = fixture_model_runtime_model_with_colocated_three_layer_series();
    let tensor_index = runtime_model.load_runtime_tensor_index(tiny_model_dir()).unwrap();
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
    assert_eq!(placed.runtime_model.placement_device_ids(), ["first", "second"]);
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
fn runtime_auto_placement_admits_eventual_lazy_retention_not_only_initial_mount() {
    let runtime_model = fixture_model_runtime_model_with_dynamic_partition(1_000, 64);
    let tensor_index = runtime_model.load_runtime_tensor_index(tiny_model_dir()).unwrap();
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
    let tensor_index = runtime_model.load_runtime_tensor_index(tiny_model_dir()).unwrap();
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
    )
    .unwrap();

    assert_eq!(placement["a"], "large-a");
    assert_eq!(placement["b"], "large-a");
    assert_eq!(placement["c"], "large-b");
    assert_eq!(placement["d"], "large-b");
    assert_eq!(placement["e"], "reserved-tail");
}

#[test]
fn runtime_auto_placement_rewires_without_discarding_selected_artifacts() {
    let mut selected = fixture_model_runtime_model();
    selected.tensor_index_fragments.push(VulkanRuntimeTensorIndexFragment {
        index_path: PathBuf::from("selected/tensors.json"),
        candidate_root: PathBuf::from("selected"),
    });
    let execution_before = selected.component_executions.clone();

    let rewired = runtime_model_with_capacity_packed_placement(
        &selected,
        "physical-a",
        &BTreeMap::from([("layer_00".to_string(), "physical-a".to_string())]),
    )
    .unwrap();

    assert_eq!(rewired.tensor_index_fragments, selected.tensor_index_fragments);
    assert_eq!(rewired.component_executions, execution_before);
    assert_eq!(rewired.placement_device_ids(), ["physical-a"]);
}
