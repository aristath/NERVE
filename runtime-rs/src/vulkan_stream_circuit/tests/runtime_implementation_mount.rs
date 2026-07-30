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

fn fixture_sha256(path: &Path) -> String {
    let bytes = std::fs::read(path).unwrap();
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(unix)]
fn fixture_source_file_signature(path: &Path) -> serde_json::Value {
    use std::os::unix::fs::MetadataExt;

    let metadata = path.metadata().unwrap();
    let timestamp_ns = |seconds: i64, nanoseconds: i64| {
        u64::try_from(seconds).unwrap() * 1_000_000_000
            + u64::try_from(nanoseconds).unwrap()
    };
    serde_json::json!({
        "device": metadata.dev(),
        "inode": metadata.ino(),
        "byte_count": metadata.len(),
        "modified_ns": timestamp_ns(metadata.mtime(), metadata.mtime_nsec()),
        "changed_ns": timestamp_ns(metadata.ctime(), metadata.ctime_nsec()),
    })
}

fn staged_candidate_integrity_files(
    root: &Path,
    directory: &Path,
    output: &mut Vec<serde_json::Value>,
) {
    let mut entries = std::fs::read_dir(directory)
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        if entry.file_type().unwrap().is_dir() {
            staged_candidate_integrity_files(root, &path, output);
        } else if path.file_name().unwrap() != "integrity.json" {
            output.push(serde_json::json!({
                "path": path.strip_prefix(root).unwrap().to_string_lossy(),
                "byte_count": path.metadata().unwrap().len(),
                "sha256": fixture_sha256(&path),
            }));
        }
    }
}

fn seal_staged_runtime_candidate(candidate_root: &Path, candidate_id: &str) {
    let mut files = Vec::new();
    staged_candidate_integrity_files(
        candidate_root,
        candidate_root,
        &mut files,
    );
    std::fs::write(
        candidate_root.join("integrity.json"),
        serde_json::to_vec(&serde_json::json!({
            "schema": crate::STAGED_CANDIDATE_INTEGRITY_SCHEMA,
            "candidate_id": candidate_id,
            "construction_id": "construction_fixture",
            "files": files,
        }))
        .unwrap(),
    )
    .unwrap();
}

fn staged_runtime_candidate_fixture() -> (
    PathBuf,
    PathBuf,
    PathBuf,
    VulkanResidentRuntimeModel,
) {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "nerve-staged-runtime-candidate-{}-{unique}",
        std::process::id()
    ));
    let package_root = root.join("package");
    let candidate_root = root.join("candidate");
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
    let mut component = runtime_model
        .package
        .circuit_graph
        .components
        .iter()
        .find(|component| component.component_id == "layer_00")
        .unwrap()
        .clone();
    let mut execution = runtime_model
        .component_executions
        .iter()
        .find(|execution| execution.component_id == "layer_00")
        .unwrap()
        .clone();
    component.implementation = "staged_alternative".to_string();
    component.circuit.implementation = "staged_alternative".to_string();
    execution.implementation = "staged_alternative".to_string();
    execution.kernels[0].shader_path =
        "kernels/staged_candidate.spv".to_string();
    std::fs::create_dir_all(candidate_root.join("kernels")).unwrap();
    std::fs::write(
        candidate_root.join("kernels/staged_candidate.spv"),
        b"staged candidate shader",
    )
    .unwrap();
    std::fs::create_dir_all(candidate_root.join("overlays")).unwrap();
    std::fs::write(
        candidate_root.join("overlays/layer_00.json"),
        serde_json::to_vec(&serde_json::json!({
            "schema": crate::VULKAN_COMPONENT_OVERLAY_SCHEMA,
            "source_component_id": "layer_00",
            "component": component,
            "execution": execution,
        }))
        .unwrap(),
    )
    .unwrap();

    let candidate_id = "candidate_0123456789abcdef0123456789abcdef";
    std::fs::create_dir_all(candidate_root.join("contracts")).unwrap();
    std::fs::write(
        candidate_root.join("contracts/candidate.json"),
        serde_json::to_vec(&serde_json::json!({
            "schema": "nerve.optimizer.representation_candidate.v1",
            "candidate_id": candidate_id,
        }))
        .unwrap(),
    )
    .unwrap();
    let source_path = package_root.join("vulkan_resident_package.json");
    std::fs::write(
        candidate_root.join("contracts/build_plan.json"),
        serde_json::to_vec(&serde_json::json!({
            "schema": "nerve.optimizer.candidate_build_plan.v1",
            "source_inputs": [{
                "path": "vulkan_resident_package.json",
                "digest": format!(
                    "{}:{}",
                    crate::STAGED_ARTIFACT_DIGEST_SCHEMA,
                    fixture_sha256(&source_path),
                ),
            }],
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        candidate_root.join("contracts/source_seal.json"),
        serde_json::to_vec(&serde_json::json!({
            "schema": "nerve.optimizer.source_package_seal.v2",
            "package_id": "fixture_package",
            "manifest_digest": "fixture_manifest_digest",
            "optimizer_stage_digest": "fixture_stage_digest",
            "exact_baseline_digest": "fixture_exact_digest",
            "scope_catalog_digest": "fixture_scope_digest",
            "package_integrity_contract_digest": "fixture_integrity_digest",
            "source_inputs": {
                "vulkan_resident_package.json": {
                    "digest": format!(
                        "{}:{}",
                        crate::STAGED_ARTIFACT_DIGEST_SCHEMA,
                        fixture_sha256(&source_path),
                    ),
                    "signature": fixture_source_file_signature(&source_path),
                },
            },
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        candidate_root.join("contracts/mount_plan.json"),
        serde_json::to_vec(&serde_json::json!({
            "schema": crate::RUNTIME_MOUNT_PLAN_SCHEMA,
            "candidate_id": candidate_id,
            "adapter_id": crate::VULKAN_STREAM_CIRCUIT_OVERLAY_ADAPTER,
            "regions": [{
                "replacements": [{
                    "kind": "component",
                    "source_component_id": "layer_00",
                    "overlay_ref": "overlays/layer_00.json",
                }],
            }],
            "tensor_index_refs": [],
        }))
        .unwrap(),
    )
    .unwrap();
    seal_staged_runtime_candidate(&candidate_root, candidate_id);
    (root, package_root, candidate_root, runtime_model)
}

fn runtime_implementation_test_predicate(
) -> crate::RuntimeImplementationPredicate {
    crate::RuntimeImplementationPredicate {
        schema:
            crate::RUNTIME_IMPLEMENTATION_PREDICATE_SCHEMA.to_string(),
        predicate_id: "runtime_predicate_fixture".to_string(),
        hardware: crate::RuntimeHardwarePredicate {
            capability_classes: vec![
                "hardware_capability_fixture".to_string(),
            ],
            device_kinds: vec!["gpu".to_string()],
            apis: vec!["vulkan".to_string()],
            required_processes: Vec::new(),
            required_features: Vec::new(),
        },
        execution: crate::RuntimeExecutionPredicate {
            phases: vec!["decode".to_string()],
            alternative_phases: vec!["decode".to_string()],
            source_retained_phases: Vec::new(),
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
            speculative_draft_token_counts: vec![0],
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
            regions: vec![crate::RuntimeMountRegion {
                replacements: vec![
                    crate::RuntimeReplacement::Component {
                        source_component_id: "layer_00".to_string(),
                        overlay_ref: overlay_ref.to_string(),
                    },
                ],
            }],
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
            speculative_draft_tokens: 0,
        },
        selected: vec![selected],
        exact_instance_ids: vec!["output_transducer".to_string()],
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
fn output_transducer_overlay_mounts_target_representation_as_one_unit() {
    let manifest =
        VulkanResidentModelPackageManifest::from_json_file(
            tiny_model_dir().join("vulkan_resident_package.json"),
        )
        .unwrap();
    let mut runtime_model = manifest
        .mount_runtime_graph_controls(
            Some("gpu0"),
            &BTreeMap::new(),
            &[],
            None,
        )
        .unwrap();
    let source = runtime_model
        .package
        .circuit_graph
        .components
        .iter()
        .find(|component| {
            component.runtime_role == CircuitRuntimeRole::OutputTransducer
        })
        .unwrap()
        .clone();
    let runtime_instance_id = source.component_id.clone();
    let mut component = source.clone();
    component.implementation = "optimized_output_representation".to_string();
    component.circuit.implementation =
        "optimized_output_representation".to_string();
    let mut output_transducer =
        runtime_model.package.output_transducer.clone();
    output_transducer.spec.projection_parameter_tensor =
        "optimizer.output_projection".to_string();
    output_transducer.spec.projection_parameter_dtype =
        "F8_E4M3".to_string();
    output_transducer.spec.projection_parameter_byte_capacity /= 2;
    output_transducer.spec.projection_scale_parameter_tensor =
        Some("optimizer.output_projection.scale".to_string());
    output_transducer.spec.projection_scale_parameter_dtype =
        Some("BF16".to_string());
    output_transducer.spec.projection_scale_parameter_shape =
        Some(vec![1]);
    output_transducer.spec.projection_scale_parameter_byte_capacity =
        Some(2);
    output_transducer.projection_shader_path =
        "kernels/output_projection.spv".to_string();
    output_transducer.projection_batch_shader_path =
        "kernels/output_projection_batch.spv".to_string();
    let mut overlay = VulkanRuntimeOutputTransducerOverlay {
        schema: crate::VULKAN_OUTPUT_TRANSDUCER_OVERLAY_SCHEMA.to_string(),
        source_component_id: source.component_id.clone(),
        component,
        output_transducer,
        speculative_output_transducers: Vec::new(),
    };
    let effective_edges = runtime_model
        .runtime_graph
        .effective_edges()
        .unwrap();
    let island_ids =
        BTreeSet::from([runtime_instance_id.as_str()]);
    validate_runtime_output_transducer_overlay(
        &runtime_model,
        &overlay,
        &source,
        &runtime_instance_id,
        &island_ids,
        &effective_edges,
    )
    .unwrap();

    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let candidate_root = std::env::temp_dir().join(format!(
        "nerve-output-overlay-{}-{unique}",
        std::process::id()
    ));
    std::fs::create_dir_all(candidate_root.join("kernels")).unwrap();
    for shader in [
        "kernels/output_projection.spv",
        "kernels/output_projection_batch.spv",
    ] {
        std::fs::write(candidate_root.join(shader), b"candidate shader")
            .unwrap();
    }
    rebase_output_transducer_overlay_shader_paths(
        &mut overlay,
        &runtime_model.package,
        &tiny_model_dir(),
        &candidate_root,
    )
    .unwrap();
    mount_runtime_output_transducer_overlay(
        &mut runtime_model,
        &runtime_instance_id,
        overlay,
    )
    .unwrap();

    assert_eq!(
        runtime_model
            .package
            .output_transducer
            .spec
            .projection_parameter_dtype,
        "F8_E4M3"
    );
    assert!(
        Path::new(
            &runtime_model.package.output_transducer.projection_shader_path
        )
        .starts_with(&candidate_root)
    );
    assert_eq!(
        runtime_model
            .circuit_graph
            .components
            .iter()
            .find(|component| component.component_id == runtime_instance_id)
            .unwrap()
            .implementation,
        "optimized_output_representation"
    );
    std::fs::remove_dir_all(candidate_root).unwrap();
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

#[test]
fn sealed_staged_candidate_uses_the_normal_runtime_mount_path() {
    let (root, package_root, candidate_root, runtime_model) =
        staged_runtime_candidate_fixture();
    let candidate = crate::RuntimeStagedCandidate::load(
        &package_root,
        &candidate_root,
    )
    .unwrap();
    let mounted = runtime_model
        .apply_staged_runtime_candidate(&package_root, &candidate)
        .unwrap();

    let component = mounted
        .circuit_graph
        .components
        .iter()
        .find(|component| component.component_id == "layer_00")
        .unwrap();
    assert_eq!(component.implementation, "staged_alternative");
    let execution = mounted
        .component_executions
        .iter()
        .find(|execution| execution.component_id == "layer_00")
        .unwrap();
    assert!(
        Path::new(&execution.kernels[0].shader_path)
            .starts_with(&candidate_root)
    );
    assert!(mounted.implementation_selection.is_none());
    mounted.load_runtime_tensor_index(&package_root).unwrap();

    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn sealed_staged_candidate_mounts_every_duplicated_source_instance() {
    let (root, package_root, candidate_root, _) =
        staged_runtime_candidate_fixture();
    let manifest =
        VulkanResidentModelPackageManifest::from_json_file(
            package_root.join("vulkan_resident_package.json"),
        )
        .unwrap();
    let source = manifest.resolved_source_graph(&package_root).unwrap();
    let runtime_graph = manifest
        .runtime_graph_from_controls(
            Some("gpu0"),
            &BTreeMap::new(),
            &[],
            None,
        )
        .unwrap()
        .duplicate_after_instance(
            &source,
            "layer_00",
            "layer_00__duplicate",
        )
        .unwrap();
    let runtime_model =
        manifest.mount_runtime_graph(&runtime_graph).unwrap();
    let candidate = crate::RuntimeStagedCandidate::load(
        &package_root,
        &candidate_root,
    )
    .unwrap();
    let mounted = runtime_model
        .apply_staged_runtime_candidate(&package_root, &candidate)
        .unwrap();

    let staged_components = mounted
        .circuit_graph
        .components
        .iter()
        .filter(|component| {
            component.implementation == "staged_alternative"
        })
        .map(|component| component.component_id.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        staged_components,
        BTreeSet::from(["layer_00", "layer_00__duplicate"])
    );
    let staged_executions = mounted
        .component_executions
        .iter()
        .filter(|execution| {
            execution.implementation == "staged_alternative"
        })
        .map(|execution| execution.component_id.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        staged_executions,
        BTreeSet::from(["layer_00", "layer_00__duplicate"])
    );

    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn staged_candidate_loader_rejects_artifact_and_source_drift() {
    let (root, package_root, candidate_root, runtime_model) =
        staged_runtime_candidate_fixture();
    let loaded = crate::RuntimeStagedCandidate::load(
        &package_root,
        &candidate_root,
    )
    .unwrap();
    let overlay_path = candidate_root.join("overlays/layer_00.json");
    let overlay = std::fs::read(&overlay_path).unwrap();
    std::fs::write(&overlay_path, b"tampered").unwrap();
    assert!(
        runtime_model
            .apply_staged_runtime_candidate(&package_root, &loaded)
            .is_err()
    );
    assert!(
        crate::RuntimeStagedCandidate::load(
            &package_root,
            &candidate_root,
        )
        .is_err()
    );
    std::fs::write(&overlay_path, overlay).unwrap();

    let source_path = package_root.join("vulkan_resident_package.json");
    let source = std::fs::read(&source_path).unwrap();
    std::fs::write(&source_path, b"tampered").unwrap();
    assert!(
        crate::RuntimeStagedCandidate::load(
            &package_root,
            &candidate_root,
        )
        .is_err()
    );
    std::fs::write(source_path, source).unwrap();

    std::fs::write(
        candidate_root.join("contracts/build_plan.json"),
        serde_json::to_vec(&serde_json::json!({
            "schema": "nerve.optimizer.candidate_build_plan.v1",
            "source_inputs": [],
        }))
        .unwrap(),
    )
    .unwrap();
    seal_staged_runtime_candidate(
        &candidate_root,
        "candidate_0123456789abcdef0123456789abcdef",
    );
    assert!(
        crate::RuntimeStagedCandidate::load(
            &package_root,
            &candidate_root,
        )
        .is_err()
    );

    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn runtime_candidate_application_requires_one_connected_island() {
    let edge = |id: &str, source: &str, destination: &str| {
        crate::stream_circuit::StreamCircuitGraphEdge {
            id: id.to_string(),
            source: crate::stream_circuit::StreamCircuitEdgeEndpoint {
                component_id: source.to_string(),
                port_id: "output".to_string(),
            },
            destination: crate::stream_circuit::StreamCircuitEdgeEndpoint {
                component_id: destination.to_string(),
                port_id: "input".to_string(),
            },
            connection: crate::stream_circuit::StreamCircuitConnection::Forward,
        }
    };
    let edges = vec![
        edge("a_to_b", "a", "b"),
        edge("b_to_c", "b", "c"),
        edge("x_to_y", "x", "y"),
    ];
    assert!(runtime_candidate_island_connected(
        &BTreeSet::from(["a", "b", "c"]),
        &edges,
    ));
    assert!(!runtime_candidate_island_connected(
        &BTreeSet::from(["a", "x"]),
        &edges,
    ));
    assert!(runtime_candidate_island_connected(
        &BTreeSet::from(["a"]),
        &edges,
    ));
    assert!(!runtime_candidate_island_connected(
        &BTreeSet::new(),
        &edges,
    ));
}
