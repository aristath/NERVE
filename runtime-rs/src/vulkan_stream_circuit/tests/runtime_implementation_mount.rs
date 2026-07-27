fn copy_runtime_implementation_fixture_tree(
    source: &Path,
    destination: &Path,
) {
    std::fs::create_dir_all(destination).unwrap();
    for entry in std::fs::read_dir(source).unwrap() {
        let entry = entry.unwrap();
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_runtime_implementation_fixture_tree(
                &source_path,
                &destination_path,
            );
        } else {
            std::fs::copy(source_path, destination_path).unwrap();
        }
    }
}

fn runtime_implementation_test_predicate(
) -> crate::RuntimeImplementationPredicate {
    crate::RuntimeImplementationPredicate {
        schema:
            crate::RUNTIME_IMPLEMENTATION_PREDICATE_SCHEMA.to_string(),
        predicate_id: "runtime_predicate_fixture".to_string(),
        hardware: crate::RuntimeHardwarePredicate {
            capability_class_counts: vec![
                crate::RuntimeCapabilityClassCount {
                    capability_class:
                        "hardware_capability_fixture".to_string(),
                    count: 1,
                },
            ],
            device_kinds: vec!["gpu".to_string()],
            apis: vec!["vulkan".to_string()],
            required_processes: Vec::new(),
            required_features: Vec::new(),
        },
        execution: crate::RuntimeExecutionPredicate {
            phases: vec!["decode".to_string()],
            activation_batch: crate::RuntimeInclusiveRange {
                minimum: 1,
                maximum: 1,
            },
            context_activations: crate::RuntimeInclusiveRange {
                minimum: 0,
                maximum: 65_536,
            },
            state_activations: crate::RuntimeInclusiveRange {
                minimum: 0,
                maximum: 65_536,
            },
        },
        placement: crate::RuntimePlacementPredicate {
            mode: "local".to_string(),
            minimum_device_count: 1,
            maximum_device_count: 1,
            required_interconnects: Vec::new(),
        },
    }
}

#[test]
fn selected_runtime_component_overlay_replaces_physical_execution() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "nerve-runtime-implementation-mount-{}-{unique}",
        std::process::id()
    ));
    let package_root = root.join("package");
    copy_runtime_implementation_fixture_tree(
        &tiny_model_dir(),
        &package_root,
    );
    let manifest =
        VulkanResidentModelPackageManifest::from_json_file(
            package_root.join("vulkan_resident_package.json"),
        )
        .unwrap();
    let runtime_model = manifest
        .mount_runtime_graph_controls(
            Some("gpu0"),
            &BTreeMap::new(),
            &[],
            None,
        )
        .unwrap();
    let source_component = runtime_model
        .package
        .circuit_graph
        .components
        .iter()
        .find(|component| component.component_id == "layer_00")
        .unwrap()
        .clone();
    let source_execution = runtime_model
        .component_executions
        .iter()
        .find(|execution| execution.component_id == "layer_00")
        .unwrap()
        .clone();
    let candidate_root = package_root.join(
        "optimization/implementations/fixture/candidate",
    );
    let mut component = source_component.clone();
    component.implementation =
        "verified_alternative_representation".to_string();
    component.circuit.implementation =
        "verified_alternative_representation".to_string();
    let mut execution = source_execution.clone();
    execution.implementation =
        "verified_alternative_representation".to_string();
    let changed_shader = "kernels/candidate_only.spv";
    let candidate_shader = candidate_root.join(changed_shader);
    std::fs::create_dir_all(candidate_shader.parent().unwrap()).unwrap();
    std::fs::write(&candidate_shader, b"candidate shader").unwrap();
    execution.kernels[0].shader_path = changed_shader.to_string();
    let overlay_ref = "overlays/layer_00.json";
    let overlay_path = candidate_root.join(overlay_ref);
    std::fs::create_dir_all(overlay_path.parent().unwrap()).unwrap();
    std::fs::write(
        &overlay_path,
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema": crate::VULKAN_COMPONENT_OVERLAY_SCHEMA,
            "source_component_id": "layer_00",
            "component": component,
            "execution": execution,
        }))
        .unwrap(),
    )
    .unwrap();

    let predicate = runtime_implementation_test_predicate();
    let implementation = crate::RuntimeImplementation {
        implementation_id: "implementation_fixture".to_string(),
        candidate_id: "candidate_fixture".to_string(),
        scope_ids: vec!["scope_fixture".to_string()],
        source_contract_digests: vec![
            "source_contract_fixture".to_string(),
        ],
        representation: serde_json::json!({
            "kind": "fixture_alternative"
        }),
        behavioral_contract: serde_json::json!({
            "mode": "exact"
        }),
        runtime_predicate: predicate.clone(),
        artifact_bundle:
            crate::RuntimeImplementationArtifactBundle {
                root_ref: "optimization/implementations/fixture"
                    .to_string(),
                candidate_integrity_ref:
                    "optimization/implementations/fixture/candidate/integrity.json"
                        .to_string(),
                mount_plan_ref:
                    "optimization/implementations/fixture/candidate/contracts/mount_plan.json"
                        .to_string(),
                candidate_integrity_digest: "fixture".to_string(),
                artifact_count: 1,
            },
        evidence: crate::RuntimeImplementationEvidence {
            promotion_decision_ref: "promotion.json".to_string(),
            candidate_contract_ref: "candidate.json".to_string(),
            construction_record_ref: "construction.json".to_string(),
            prebenchmark_record_ref: "prebenchmark.json".to_string(),
            benchmark_record_ref: "benchmark.json".to_string(),
            validation_record_ref: "validation.json".to_string(),
            analysis_run_refs: Vec::new(),
            hardware_profile_refs: Vec::new(),
        },
        provenance: serde_json::json!({
            "provider": "fixture"
        }),
        comparison: crate::RuntimeImplementationComparison {
            exact_implementation_id: "exact".to_string(),
            exact_contract_digest: "exact_digest".to_string(),
            benchmark_id: "benchmark_fixture".to_string(),
            benchmark_decision: "materially_faster".to_string(),
            workloads: Vec::new(),
            validation_id: "validation_fixture".to_string(),
            validation_status: "passed".to_string(),
            behavioral_contract: serde_json::json!({
                "mode": "exact"
            }),
        },
        decision_reason: "verified fixture alternative".to_string(),
    };
    let loaded = crate::LoadedRuntimeImplementation {
        implementation: implementation.clone(),
        source_component_ids: vec!["layer_00".to_string()],
        workload_metrics: Vec::new(),
        candidate_root: candidate_root.clone(),
        mount_plan: crate::RuntimeMountPlan {
            schema: crate::RUNTIME_MOUNT_PLAN_SCHEMA.to_string(),
            candidate_id: implementation.candidate_id.clone(),
            adapter_id:
                crate::VULKAN_STREAM_CIRCUIT_OVERLAY_ADAPTER
                    .to_string(),
            component_replacements: vec![
                crate::RuntimeComponentReplacement {
                    source_component_id: "layer_00".to_string(),
                    overlay_ref: overlay_ref.to_string(),
                },
            ],
            tensor_index_refs: Vec::new(),
        },
    };
    let catalog = crate::RuntimeImplementationCatalog {
        package_id: runtime_model.package.package_id.clone(),
        package_root: package_root.clone(),
        stage_status: "optimized".to_string(),
        exact_baseline: crate::RuntimeExactImplementation {
            artifact_ref:
                "lowered/execution_graph.circuits.json".to_string(),
            contract_digest: "exact".to_string(),
            mutable: false,
        },
        scopes: BTreeMap::new(),
        implementations: vec![loaded],
    };
    let selected = crate::RuntimeSelectedImplementation {
        implementation_id: implementation.implementation_id.clone(),
        candidate_id: implementation.candidate_id.clone(),
        instance_ids: vec!["layer_00".to_string()],
        scope_ids: implementation.scope_ids.clone(),
        mount_adapter_id:
            crate::VULKAN_STREAM_CIRCUIT_OVERLAY_ADAPTER.to_string(),
        predicate,
        representation: implementation.representation.clone(),
        provenance: implementation.provenance.clone(),
        benchmark_id: "benchmark_fixture".to_string(),
        validation_id: "validation_fixture".to_string(),
        validation_status: "passed".to_string(),
        speedup_ppm: 100_000,
        estimated_saved_ns: 100,
        conversion_ns: 0,
        conversion_bytes: 0,
        boundary_count: 0,
        decision_reason: "verified fixture alternative".to_string(),
    };
    let report = crate::RuntimeImplementationSelectionReport {
        package_id: runtime_model.package.package_id.clone(),
        execution: crate::RuntimeExecutionEnvelope {
            phases: vec!["decode".to_string()],
            activation_batch: crate::RuntimeInclusiveRange {
                minimum: 1,
                maximum: 1,
            },
            context_activations: crate::RuntimeInclusiveRange {
                minimum: 0,
                maximum: 65_536,
            },
            state_activations: crate::RuntimeInclusiveRange {
                minimum: 0,
                maximum: 65_536,
            },
        },
        selected: vec![selected],
        exact_instance_ids: Vec::new(),
        rejected: Vec::new(),
        total_estimated_saved_ns: 100,
        total_conversion_ns: 0,
        total_conversion_bytes: 0,
        total_boundary_count: 0,
    };

    let mounted = runtime_model
        .apply_runtime_implementation_catalog_selection(
            &package_root,
            &catalog,
            report,
        )
        .unwrap();

    let mounted_component = mounted
        .circuit_graph
        .components
        .iter()
        .find(|component| component.component_id == "layer_00")
        .unwrap();
    assert_eq!(
        mounted_component.implementation,
        "verified_alternative_representation"
    );
    let mounted_execution = mounted
        .component_executions
        .iter()
        .find(|execution| execution.component_id == "layer_00")
        .unwrap();
    assert!(
        Path::new(&mounted_execution.kernels[0].shader_path)
            .starts_with(&candidate_root)
    );
    assert!(
        mounted_execution.kernels[1..]
            .iter()
            .all(|kernel| Path::new(&kernel.shader_path)
                .starts_with(&package_root))
    );
    assert!(
        mounted_execution
            .kernels
            .iter()
            .flat_map(|kernel| &kernel.batch_implementations)
            .flat_map(|implementation| &implementation.stages)
            .all(|stage| Path::new(&stage.shader_path)
                .starts_with(&package_root))
    );
    assert!(mounted.implementation_selection.is_some());
    mounted.load_runtime_tensor_index(&package_root).unwrap();

    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn runtime_overlay_shader_resolution_rejects_path_spoofing_and_symlinks() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "nerve-runtime-overlay-paths-{}-{unique}",
        std::process::id()
    ));
    let package_root = root.join("package");
    let candidate_root = root.join("candidate");
    std::fs::create_dir_all(&package_root).unwrap();
    std::fs::create_dir_all(&candidate_root).unwrap();
    std::fs::write(package_root.join("source.spv"), b"source").unwrap();
    std::fs::write(candidate_root.join("candidate.spv"), b"candidate")
        .unwrap();

    let inherited = rebase_overlay_shader_path(
        "source.spv",
        Some("source.spv"),
        &package_root,
        &candidate_root,
        "shader",
    )
    .unwrap();
    assert!(Path::new(&inherited).starts_with(&package_root));
    let changed = rebase_overlay_shader_path(
        "candidate.spv",
        Some("source.spv"),
        &package_root,
        &candidate_root,
        "shader",
    )
    .unwrap();
    assert!(Path::new(&changed).starts_with(&candidate_root));
    assert!(
        rebase_overlay_shader_path(
            "../source.spv",
            Some("source.spv"),
            &package_root,
            &candidate_root,
            "shader",
        )
        .is_err()
    );

    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(
            package_root.join("source.spv"),
            candidate_root.join("linked.spv"),
        )
        .unwrap();
        assert!(
            rebase_overlay_shader_path(
                "linked.spv",
                Some("source.spv"),
                &package_root,
                &candidate_root,
                "shader",
            )
            .is_err()
        );
    }

    std::fs::remove_dir_all(root).unwrap();
}
