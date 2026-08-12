fn physical_checkpoint_fixture() -> (
    CompiledResourceResidencyContract,
    Vec<VulkanPhysicalResidencyDispatch>,
) {
    let selector = CompiledResourceSelector {
        id: "selector".to_string(),
        execution_scope: "target".to_string(),
        component_id: "component".to_string(),
        node_id: "choose".to_string(),
        domain_id: "units".to_string(),
        resource_count: 3,
        selection_signal: "selected".to_string(),
        encoding: CompiledResourceSelectionEncoding {
            element_type: CompiledResourceSelectionElementType::U32,
            selection_count_per_activation: 1,
            index_shift: 0,
            index_mask: 0xffff,
        },
        mapping: CompiledResourceSelectorMapping::GroupTable {
            atomic_group_ids: vec![
                "group_0".to_string(),
                "group_1".to_string(),
                "group_2".to_string(),
            ],
        },
    };
    let checkpoint = CompiledResidencyCheckpoint {
        id: "checkpoint".to_string(),
        execution_scope: "target".to_string(),
        component_id: "component".to_string(),
        after_node_id: "choose".to_string(),
        resume_node_id: "compute_a".to_string(),
        selector_ids: vec![selector.id.clone()],
    };
    let bindings = ["compute_a", "compute_b"]
        .into_iter()
        .map(|node_id| CompiledResourceBinding {
            execution_scope: "target".to_string(),
            component_id: "component".to_string(),
            node_id: node_id.to_string(),
            parameter_id: format!("{node_id}_parameter"),
            mapping: CompiledResourceBindingMapping::SelectedAtomicGroup {
                atomic_group_id: "group_0".to_string(),
                resource_id: format!("{node_id}_resource"),
                selection_signal: "selected".to_string(),
                selector_index: 0,
                parameter_slot: 0,
            },
        })
        .collect();
    let contract = CompiledResourceResidencyContract {
        schema: COMPILED_RESOURCE_RESIDENCY_SCHEMA.to_string(),
        identity_algorithm: RESOURCE_IDENTITY_ALGORITHM.to_string(),
        state_machine_schema: RESOURCE_RESIDENCY_STATE_MACHINE_SCHEMA.to_string(),
        supported_policies: vec![
            ResourceResidencyPolicy::DemandRetained,
            ResourceResidencyPolicy::Eager,
        ],
        resources: Vec::new(),
        atomic_groups: Vec::new(),
        partition_templates: Vec::new(),
        bindings,
        selectors: vec![selector],
        checkpoints: vec![checkpoint],
    };
    let dispatches = [
        (10, 0, "choose"),
        (11, 1, "compute_a"),
        (12, 2, "between_selected_work"),
        (13, 3, "compute_b"),
        (14, 4, "combine_selected_results"),
        (15, 5, "ordinary_continuation"),
    ]
    .into_iter()
    .map(
        |(dispatch_index, node_index, node_id)| VulkanPhysicalResidencyDispatch {
            dispatch_index,
            component_id: "component".to_string(),
            node_index,
            node_id: node_id.to_string(),
        },
    )
    .collect();
    (contract, dispatches)
}

#[test]
fn physical_residency_schedule_derives_generic_selected_execution_boundaries() {
    let (contract, dispatches) = physical_checkpoint_fixture();

    let schedule =
        VulkanPhysicalResidencySchedule::from_dispatches(
            &contract,
            "target".to_string(),
            &dispatches,
        )
        .unwrap();

    assert_eq!(schedule.checkpoints.len(), 1);
    let checkpoint = &schedule.checkpoints[0];
    assert_eq!(checkpoint.selection_dispatch_index, 10);
    assert_eq!(
        checkpoint.selected_computation_dispatch_indices,
        [11, 12, 13]
    );
    assert_eq!(
        checkpoint.selected_result_continuation_dispatch_index,
        Some(14)
    );
    assert!(!checkpoint
        .selected_computation_dispatch_indices
        .contains(&15));
}

#[test]
fn exact_preselection_can_gate_before_non_selected_router_weighting() {
    let (contract, _) = physical_checkpoint_fixture();
    let dispatches = [
        (10, 0, "choose"),
        (11, 1, "router_projection"),
        (12, 2, "weight_preselected_routes"),
        (13, 3, "compute_a"),
        (14, 4, "between_selected_work"),
        (15, 5, "compute_b"),
        (16, 6, "combine_selected_results"),
    ]
    .into_iter()
    .map(
        |(dispatch_index, node_index, node_id)| VulkanPhysicalResidencyDispatch {
            dispatch_index,
            component_id: "component".to_string(),
            node_index,
            node_id: node_id.to_string(),
        },
    )
    .collect::<Vec<_>>();

    let schedule = VulkanPhysicalResidencySchedule::from_dispatches(
        &contract,
        "target".to_string(),
        &dispatches,
    )
    .unwrap();
    let checkpoint = &schedule.checkpoints[0];

    assert_eq!(checkpoint.selection_dispatch_index, 10);
    assert_eq!(
        checkpoint.selected_computation_dispatch_indices,
        [13, 14, 15]
    );
    assert_eq!(
        checkpoint.selected_result_continuation_dispatch_index,
        Some(16)
    );
}

#[test]
fn physical_residency_schedule_materializes_gates_only_for_demand_policies() {
    let (contract, dispatches) = physical_checkpoint_fixture();
    let schedule = VulkanPhysicalResidencySchedule::from_dispatches(
        &contract,
        "target".to_string(),
        &dispatches,
    )
    .unwrap();

    assert_eq!(schedule.demand_gate_count(ResourceResidencyPolicy::Eager), 0);
    assert!(!schedule.requires_demand_execution(ResourceResidencyPolicy::Eager));
    for policy in [
        ResourceResidencyPolicy::DemandRetained,
        ResourceResidencyPolicy::DemandPaged,
    ] {
        assert_eq!(schedule.demand_gate_count(policy), 1);
        assert!(schedule.requires_demand_execution(policy));
    }
}

#[test]
fn physical_residency_checkpoint_resolves_topk_indices_to_atomic_groups() {
    let (contract, dispatches) = physical_checkpoint_fixture();
    let schedule =
        VulkanPhysicalResidencySchedule::from_dispatches(
            &contract,
            "target".to_string(),
            &dispatches,
        )
        .unwrap();
    let checkpoint = &schedule.checkpoints[0];

    let selected = checkpoint
        .resolve_selected_group_ids(
            &contract,
            &[
                VulkanSelectedResourceIndex {
                    selector_id: "selector".to_string(),
                    resource_index: 2,
                },
                VulkanSelectedResourceIndex {
                    selector_id: "selector".to_string(),
                    resource_index: 0,
                },
            ],
        )
        .unwrap();

    assert_eq!(selected, ["group_0", "group_2"]);
}

#[test]
fn demand_checkpoint_resumes_selected_work_without_replaying_selection() {
    let (contract, dispatches) = physical_checkpoint_fixture();
    let schedule =
        VulkanPhysicalResidencySchedule::from_dispatches(
            &contract,
            "target".to_string(),
            &dispatches,
        )
        .unwrap();
    let checkpoint = &schedule.checkpoints[0];
    let selected = vec!["group_0".to_string(), "group_2".to_string()];
    let fully_resident = selected.iter().cloned().collect::<BTreeSet<_>>();

    let mut eager = checkpoint.begin_activation(selected.clone()).unwrap();
    assert_eq!(
        eager.advance(&fully_resident).unwrap(),
        VulkanPhysicalResidencyActivationStatus::Completed
    );

    let mut demand = checkpoint.begin_activation(selected).unwrap();
    assert_eq!(
        demand.advance(&BTreeSet::new()).unwrap(),
        VulkanPhysicalResidencyActivationStatus::Paused {
            checkpoint_id: "checkpoint".to_string(),
            missing_group_ids: vec!["group_0".to_string(), "group_2".to_string()],
            resume_at: VulkanPhysicalResidencyResponsibility::SelectedComputation,
        }
    );
    assert_eq!(
        demand
            .trace()
            .iter()
            .map(|entry| entry.responsibility)
            .collect::<Vec<_>>(),
        [
            VulkanPhysicalResidencyResponsibility::Selection,
            VulkanPhysicalResidencyResponsibility::Availability,
        ]
    );
    let partial = BTreeSet::from(["group_0".to_string()]);
    assert!(demand.resume_after_atomic_publication(&partial).is_err());
    assert_eq!(demand.trace().len(), 2);

    assert_eq!(
        demand
            .resume_after_atomic_publication(&fully_resident)
            .unwrap(),
        VulkanPhysicalResidencyActivationStatus::Completed
    );
    assert_eq!(demand.trace(), eager.trace());
    assert_eq!(
        demand
            .trace()
            .iter()
            .filter(|entry| {
                entry.responsibility == VulkanPhysicalResidencyResponsibility::Selection
            })
            .count(),
        1
    );
    assert_eq!(
        demand.advance(&fully_resident).unwrap(),
        VulkanPhysicalResidencyActivationStatus::Completed
    );
    assert_eq!(demand.trace(), eager.trace());
}

#[test]
fn physical_checkpoint_supports_non_expert_terminal_selected_resources() {
    let (mut contract, mut dispatches) = physical_checkpoint_fixture();
    contract.bindings.truncate(1);
    contract.checkpoints[0].resume_node_id = "compute_a".to_string();
    dispatches.truncate(2);

    let schedule =
        VulkanPhysicalResidencySchedule::from_dispatches(
            &contract,
            "target".to_string(),
            &dispatches,
        )
        .unwrap();
    let checkpoint = &schedule.checkpoints[0];
    assert_eq!(checkpoint.selected_computation_dispatch_indices, [11]);
    assert_eq!(
        checkpoint.selected_result_continuation_dispatch_index,
        None
    );

    let resident = BTreeSet::from(["group_1".to_string()]);
    let mut activation = checkpoint
        .begin_activation(vec!["group_1".to_string()])
        .unwrap();
    assert_eq!(
        activation.advance(&resident).unwrap(),
        VulkanPhysicalResidencyActivationStatus::Completed
    );
    assert_eq!(
        activation
            .trace()
            .iter()
            .map(|entry| entry.responsibility)
            .collect::<Vec<_>>(),
        [
            VulkanPhysicalResidencyResponsibility::Selection,
            VulkanPhysicalResidencyResponsibility::Availability,
            VulkanPhysicalResidencyResponsibility::SelectedComputation,
        ]
    );
}

#[test]
fn physical_residency_coverage_rejects_missing_or_duplicate_device_ownership() {
    let (contract, dispatches) = physical_checkpoint_fixture();
    let schedule =
        VulkanPhysicalResidencySchedule::from_dispatches(
            &contract,
            "target".to_string(),
            &dispatches,
        )
        .unwrap();
    let empty = VulkanPhysicalResidencySchedule {
        execution_scope: "target".to_string(),
        checkpoints: Vec::new(),
    };

    validate_physical_residency_schedule_coverage(
        &contract,
        "target",
        [&schedule],
    )
    .unwrap();
    assert!(
        validate_physical_residency_schedule_coverage(
            &contract,
            "target",
            [&empty],
        )
        .unwrap_err()
        .to_string()
        .contains("incomplete")
    );
    assert!(
        validate_physical_residency_schedule_coverage(
            &contract,
            "target",
            [&schedule, &schedule],
        )
        .unwrap_err()
        .to_string()
        .contains("more than one device slice")
    );
}
