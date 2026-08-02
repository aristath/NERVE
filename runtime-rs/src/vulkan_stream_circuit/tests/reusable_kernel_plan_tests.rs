#[test]
fn reusable_kernel_plan_keeps_compile_time_specializations_distinct() {
    let command =
        |dispatch_index: usize, component_id: &str, specialization: &str| VulkanKernelDispatchCommand {
            dispatch_index,
            circuit_index: dispatch_index,
            kernel_id: format!("{component_id}.per_layer_embedding"),
            component_id: component_id.to_string(),
            circuit_id: format!("{component_id}_circuit"),
            node_index: 0,
            node_id: "per_layer_embedding".to_string(),
            op: "per_layer_embedding".to_string(),
            specialization: specialization.to_string(),
            descriptor_bindings: Vec::new(),
            push_constants: Vec::new(),
            stream_control_binding: Some(0),
        };
    let dispatch_plan = VulkanKernelDispatchPlan {
        backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
        commands: vec![
            command(0, "layer_00", r#"{"layer_index":0}"#),
            command(1, "layer_01", r#"{"layer_index":1}"#),
        ],
    };

    let reusable_plan = VulkanReusableKernelPlan::from_dispatch_plan(&dispatch_plan);

    assert_eq!(reusable_plan.total_family_count(), 2);
    assert_eq!(reusable_plan.reusable_family_count(), 0);
    assert!(
        reusable_plan
            .families
            .iter()
            .all(|family| family.command_refs.len() == 1)
    );

    let reversed_dispatch_plan = VulkanKernelDispatchPlan {
        backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
        commands: dispatch_plan.commands.iter().rev().cloned().collect(),
    };
    let reversed_reusable_plan =
        VulkanReusableKernelPlan::from_dispatch_plan(&reversed_dispatch_plan);
    assert_eq!(
        reusable_plan
            .families
            .iter()
            .map(|family| family.family_id.as_str())
            .collect::<BTreeSet<_>>(),
        reversed_reusable_plan
            .families
            .iter()
            .map(|family| family.family_id.as_str())
            .collect::<BTreeSet<_>>()
    );
}

#[test]
fn parses_explicit_shared_state_sources() {
    assert_eq!(
        shared_state_source("shared_from:layer_22.kv_memory").unwrap(),
        Some(("layer_22".to_string(), "kv_memory".to_string()))
    );
    assert_eq!(shared_state_source("private").unwrap(), None);
    assert!(shared_state_source("shared_from:kv_memory").is_err());
}

#[test]
fn reusable_kernel_plan_preserves_fixture_model_compiled_kernel_contracts() {
    let graph = fixture_model_execution_graph();
    let tensor_index = TensorIndex::from_json_file(fixture_model_tensor_index_path()).unwrap();
    let execution_plan =
        StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index).unwrap();
    let resource_plan =
        StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
    let resident_plan = VulkanStreamCircuitResidentPlan::from_resource_plan(
        &resource_plan,
        Some(&tensor_index),
        Some(2),
    )
    .unwrap();
    let binding_plan =
        VulkanStreamCircuitBindingPlan::from_plans(&execution_plan, &resource_plan, &resident_plan)
            .unwrap();
    let dispatch_plan = VulkanKernelDispatchPlan::from_binding_plan(&binding_plan);

    let reusable_plan = VulkanReusableKernelPlan::from_dispatch_plan(&dispatch_plan);

    assert_eq!(reusable_plan.backend_id, VULKAN_STREAM_CIRCUIT_BACKEND_ID);
    assert_eq!(reusable_plan.total_command_count, 10);
    assert_eq!(reusable_plan.total_family_count(), reusable_plan.total_command_count);
    assert_eq!(reusable_plan.reusable_family_count(), 0);
    assert_eq!(
        reusable_plan
            .families
            .iter()
            .map(|family| family.command_refs.len())
            .sum::<usize>(),
        reusable_plan.total_command_count
    );

    let q_projection = reusable_family_with_kernel(
        &reusable_plan,
        "layer_00.q_projection__k_projection__v_projection",
    );
    assert_eq!(q_projection.op, "parallel_linear_3way");
    assert_eq!(q_projection.stream_control_binding, None);
    assert!(q_projection.command_refs.iter().any(|command| {
        command.kernel_id == "layer_00.q_projection__k_projection__v_projection"
            && command.dispatch_index == 1
    }));
    assert_eq!(
        q_projection.descriptor_signature[4],
        VulkanKernelDescriptorSlotSignature {
            binding: 4,
            usage: VulkanKernelDescriptorUsage::Parameter,
            resource_class: VulkanKernelDescriptorResourceClass::ParameterBuffer,
            byte_capacity: Some(512),
            shape: Some(vec![16, 16]),
        }
    );

    let rope_command_count = reusable_plan
        .families_for_op("rotary_position_embedding")
        .iter()
        .map(|family| family.command_refs.len())
        .sum::<usize>();
    assert_eq!(rope_command_count, 2);
    assert!(
        reusable_plan
            .families_for_op("rotary_position_embedding")
            .iter()
            .all(|family| family.stream_control_binding.is_some() && family.push_constants.is_empty())
    );

    let append = reusable_family_with_kernel(
        &reusable_plan,
        "layer_00.kv_memory_append__attention_read",
    );
    assert_eq!(append.op, "append_scaled_dot_product_attention");
    assert_eq!(append.stream_control_binding, Some(7));
    assert_eq!(
        append
            .descriptor_signature
            .iter()
            .filter(|slot| {
                matches!(
                    slot.usage,
                    VulkanKernelDescriptorUsage::StateRead
                        | VulkanKernelDescriptorUsage::StateWrite
                        | VulkanKernelDescriptorUsage::StateView
                )
            })
            .count(),
        2
    );
    assert_eq!(
        append
            .descriptor_signature
            .iter()
            .filter(|slot| {
                slot.resource_class == VulkanKernelDescriptorResourceClass::StateBuffer
            })
            .count(),
        2
    );
}

#[test]
fn reusable_kernel_coverage_reports_missing_gpu_component_circuits() {
    let reusable_plan = fixture_model_reusable_kernel_plan();
    let selected = reusable_family_with_kernel(&reusable_plan, "layer_00.q_projection__k_projection__v_projection");
    let selected_family_id = selected.family_id.as_str();
    let required_family_count = reusable_plan.total_family_count();
    let required_command_count = reusable_plan.total_command_count;

    let empty = reusable_plan.coverage_report(std::iter::empty::<&str>());
    assert!(!empty.all_available());
    assert_eq!(empty.required_family_count, required_family_count);
    assert_eq!(empty.available_family_count, 0);
    assert_eq!(empty.missing_family_count, required_family_count);
    assert_eq!(empty.required_command_count, required_command_count);
    assert_eq!(empty.covered_command_count, 0);
    assert_eq!(empty.missing_command_count, required_command_count);
    assert!(
        empty
            .missing_families()
            .iter()
            .any(|family| family.family_id == selected_family_id
                && family.command_count == selected.command_refs.len())
    );

    let partial = reusable_plan.coverage_report([selected_family_id]);
    assert!(!partial.all_available());
    assert_eq!(partial.available_family_count, 1);
    assert_eq!(partial.missing_family_count, required_family_count - 1);
    assert_eq!(partial.covered_command_count, selected.command_refs.len());
    assert_eq!(
        partial.missing_command_count,
        required_command_count - selected.command_refs.len()
    );

    let full = reusable_plan.coverage_report(
        reusable_plan
            .families
            .iter()
            .map(|family| family.family_id.as_str()),
    );
    assert!(full.all_available());
    assert_eq!(full.available_family_count, required_family_count);
    assert_eq!(full.missing_family_count, 0);
    assert_eq!(full.covered_command_count, required_command_count);
    assert_eq!(full.missing_command_count, 0);
}

#[test]
fn reusable_kernel_artifact_manifest_links_fixture_model_kernel_families() {
    let reusable_plan = fixture_model_reusable_kernel_plan();
    let selected = reusable_family_with_kernel(&reusable_plan, "layer_00.q_projection__k_projection__v_projection");
    let selected_family_id = selected.family_id.as_str();
    let selected_artifact_path = artifact_path_for_family(selected);
    let family_count = reusable_plan.total_family_count();
    let command_count = reusable_plan.total_command_count;
    let manifest = VulkanReusableKernelArtifactManifest::new(
        reusable_plan
            .families
            .iter()
            .map(|family| {
                VulkanReusableKernelArtifact::from_family(
                    family,
                    format!("kernels/{}.spv", family.family_id),
                )
            })
            .collect(),
    );

    let link_plan = reusable_plan.link_artifacts(&manifest);

    assert_eq!(
        manifest.schema,
        VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA
    );
    assert_eq!(manifest.backend_id, VULKAN_STREAM_CIRCUIT_BACKEND_ID);
    assert_eq!(manifest.artifacts.len(), family_count);
    assert!(link_plan.is_fully_linked());
    assert_eq!(link_plan.required_family_count, family_count);
    assert_eq!(link_plan.linked_family_count, family_count);
    assert_eq!(link_plan.missing_family_count, 0);
    assert_eq!(link_plan.incompatible_family_count, 0);
    assert_eq!(link_plan.required_command_count, command_count);
    assert_eq!(link_plan.linked_command_count, command_count);
    assert!(link_plan.issues.is_empty());

    let linked = link_plan.family(selected_family_id).unwrap();
    assert_eq!(linked.status, VulkanReusableKernelLinkStatus::Linked);
    assert_eq!(linked.command_count, selected.command_refs.len());
    assert_eq!(
        linked.artifact_path.as_deref(),
        Some(selected_artifact_path.as_str())
    );

    let manifest_path = std::env::temp_dir().join(format!(
        "nerve-reusable-kernel-manifest-{}.json",
        std::process::id()
    ));
    manifest.write_json_file(&manifest_path).unwrap();
    let read = VulkanReusableKernelArtifactManifest::from_json_file(&manifest_path).unwrap();
    std::fs::remove_file(&manifest_path).unwrap();
    assert_eq!(read, manifest);
    assert_eq!(read.family_ids().len(), family_count);

    let artifact_root = std::env::temp_dir().join(format!(
        "nerve-reusable-kernel-artifacts-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(artifact_root.join("kernels")).unwrap();
    for (index, artifact) in manifest.artifacts.iter().enumerate() {
        crate::vulkan::write_spirv_words(
            artifact_root.join(&artifact.path),
            &[0x0723_0203, index as u32],
        )
        .unwrap();
    }

    let loaded = manifest.load_artifacts(&artifact_root).unwrap();
    assert_eq!(loaded.artifacts.len(), family_count);
    assert_eq!(loaded.family_ids().len(), family_count);
    assert_eq!(loaded.total_word_count, family_count * 2);
    let loaded_selected = loaded.artifact(selected_family_id).unwrap();
    assert_eq!(loaded_selected.artifact.family_id, selected_family_id);
    assert_eq!(
        loaded_selected.resolved_path,
        artifact_root.join(&selected_artifact_path)
    );
    assert_eq!(loaded_selected.words[0], 0x0723_0203);
    std::fs::remove_dir_all(&artifact_root).unwrap();
}

#[test]
fn reusable_kernel_link_plan_reports_partial_and_incompatible_artifacts() {
    let reusable_plan = fixture_model_reusable_kernel_plan();
    let selected = reusable_family_with_kernel(&reusable_plan, "layer_00.q_projection__k_projection__v_projection");
    let selected_family_id = selected.family_id.as_str();
    let selected_command_count = selected.command_refs.len();
    let family_count = reusable_plan.total_family_count();
    let command_count = reusable_plan.total_command_count;

    let partial_manifest = VulkanReusableKernelArtifactManifest::empty().with_artifact(
        VulkanReusableKernelArtifact::from_family(selected, artifact_path_for_family(selected)),
    );
    let partial_link = reusable_plan.link_artifacts(&partial_manifest);

    assert!(!partial_link.is_fully_linked());
    assert_eq!(partial_link.linked_family_count, 1);
    assert_eq!(partial_link.missing_family_count, family_count - 1);
    assert_eq!(partial_link.incompatible_family_count, 0);
    assert_eq!(partial_link.linked_command_count, selected_command_count);
    assert_eq!(
        partial_link.missing_command_count,
        command_count - selected_command_count
    );
    assert_eq!(
        partial_link.family(selected_family_id).unwrap().status,
        VulkanReusableKernelLinkStatus::Linked
    );

    let mut bad_selected = VulkanReusableKernelArtifact::from_family(selected, "")
        .with_entry_point("not_main")
        .with_local_size_x(0);
    bad_selected.op = "multiply".to_string();
    bad_selected.descriptor_signature.pop();
    let incompatible_manifest =
        VulkanReusableKernelArtifactManifest::empty().with_artifact(bad_selected);
    let incompatible_link = reusable_plan.link_artifacts(&incompatible_manifest);

    assert!(!incompatible_link.is_fully_linked());
    assert_eq!(incompatible_link.linked_family_count, 0);
    assert_eq!(incompatible_link.missing_family_count, family_count - 1);
    assert_eq!(incompatible_link.incompatible_family_count, 1);
    assert_eq!(
        incompatible_link.incompatible_command_count,
        selected_command_count
    );
    assert_eq!(
        incompatible_link.missing_command_count,
        command_count - selected_command_count
    );
    let selected_link = incompatible_link.family(selected_family_id).unwrap();
    assert_eq!(
        selected_link.status,
        VulkanReusableKernelLinkStatus::Incompatible
    );
    assert!(selected_link.issues.iter().any(|issue| matches!(
        issue.problem,
        VulkanReusableKernelLinkProblem::OpMismatch { .. }
    )));
    assert!(selected_link.issues.iter().any(|issue| matches!(
        issue.problem,
        VulkanReusableKernelLinkProblem::DescriptorSignatureMismatch
    )));
    assert!(selected_link.issues.iter().any(|issue| matches!(
        issue.problem,
        VulkanReusableKernelLinkProblem::EmptySpirvPath
    )));
    assert!(selected_link.issues.iter().any(|issue| matches!(
        issue.problem,
        VulkanReusableKernelLinkProblem::UnsupportedEntryPoint { .. }
    )));
    assert!(selected_link.issues.iter().any(|issue| matches!(
        issue.problem,
        VulkanReusableKernelLinkProblem::InvalidLocalSizeX { .. }
    )));
    assert_eq!(incompatible_link.incompatible_families().len(), 1);
}

fn fixture_model_reusable_kernel_plan() -> VulkanReusableKernelPlan {
    let graph = fixture_model_execution_graph();
    let tensor_index = TensorIndex::from_json_file(fixture_model_tensor_index_path()).unwrap();
    let execution_plan =
        StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index).unwrap();
    let resource_plan =
        StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
    let resident_plan = VulkanStreamCircuitResidentPlan::from_resource_plan(
        &resource_plan,
        Some(&tensor_index),
        Some(2),
    )
    .unwrap();
    let binding_plan =
        VulkanStreamCircuitBindingPlan::from_plans(&execution_plan, &resource_plan, &resident_plan)
            .unwrap();
    let dispatch_plan = VulkanKernelDispatchPlan::from_binding_plan(&binding_plan);
    VulkanReusableKernelPlan::from_dispatch_plan(&dispatch_plan)
}
