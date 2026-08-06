use super::*;
use crate::{
    HardwareDeviceKind, HardwareIdentity, HardwareInterconnect, HardwareMemoryDomain,
    HardwareProcessAvailability, HardwareProcessCapability, HardwareProcessCategory,
    HardwareProcessProfile, HardwareProcessProfileDefinition, HardwareProcessProgrammability,
    HardwareProfileProvenance,
};
use serde_json::json;
use std::collections::BTreeMap;
use std::path::PathBuf;

fn profile(
    device_kind: HardwareDeviceKind,
    stable_device_id: &str,
    architecture: &str,
    api: &str,
) -> HardwareProcessProfile {
    HardwareProcessProfile::create(HardwareProcessProfileDefinition {
        hardware_identity: HardwareIdentity {
            device_kind,
            stable_device_id: stable_device_id.to_string(),
            name: format!("{architecture} device"),
            vendor_id: "0x1002".to_string(),
            device_id: "0x0001".to_string(),
            architecture: architecture.to_string(),
            physical_location: stable_device_id.to_string(),
        },
        processes: vec![HardwareProcessCapability {
            name: "vector_compute".to_string(),
            category: HardwareProcessCategory::Arithmetic,
            availability: HardwareProcessAvailability::Available,
            programmability: HardwareProcessProgrammability::Direct,
            api: api.to_string(),
            operations: vec!["fused_multiply_add".to_string()],
            numeric_formats: vec!["fp8".to_string()],
            required_extensions: Vec::new(),
            required_features: vec!["native_fp8".to_string()],
            limits: BTreeMap::new(),
            properties: BTreeMap::new(),
        }],
        memory_domains: vec![HardwareMemoryDomain {
            name: "local_memory".to_string(),
            kind: "device_local".to_string(),
            capacity_bytes: 32 * 1024 * 1024 * 1024,
            host_visible: device_kind == HardwareDeviceKind::Cpu,
            device_local: true,
            coherent: device_kind == HardwareDeviceKind::Cpu,
            cached: true,
            minimum_alignment_bytes: 64,
            properties: BTreeMap::new(),
        }],
        interconnects: vec![HardwareInterconnect {
            name: "pcie".to_string(),
            kind: "pcie".to_string(),
            availability: HardwareProcessAvailability::Available,
            api: api.to_string(),
            operations: vec!["device_transfer".to_string()],
            properties: BTreeMap::new(),
        }],
        provenance: HardwareProfileProvenance {
            api: api.to_string(),
            api_version: "1.0".to_string(),
            driver: "fixture".to_string(),
            driver_version: "1".to_string(),
            compiler: "fixture".to_string(),
            operating_system: "linux".to_string(),
            discovery_backend: "fixture".to_string(),
        },
        capability_extensions: BTreeMap::new(),
        identity_extensions: BTreeMap::new(),
        runtime_bindings: BTreeMap::new(),
    })
    .unwrap()
}

fn predicate(profiles: &[&HardwareProcessProfile], mode: &str) -> RuntimeImplementationPredicate {
    RuntimeImplementationPredicate {
        schema: RUNTIME_IMPLEMENTATION_PREDICATE_SCHEMA.to_string(),
        predicate_id: "runtime_predicate_fixture".to_string(),
        hardware: RuntimeHardwarePredicate {
            measured_profile_ids: profiles
                .iter()
                .map(|profile| profile.profile_id.clone())
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect(),
            capability_classes: profiles
                .iter()
                .map(|profile| profile.capability_class.clone())
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect(),
            device_kinds: profiles
                .iter()
                .map(|profile| profile.hardware_identity.device_kind.as_str().to_string())
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect(),
            apis: profiles
                .iter()
                .map(|profile| profile.provenance.api.clone())
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect(),
            required_processes: vec!["vector_compute".to_string()],
            required_features: vec!["native_fp8".to_string()],
        },
        execution: RuntimeExecutionPredicate {
            phases: vec!["decode".to_string()],
            alternative_phases: vec!["decode".to_string()],
            source_retained_phases: Vec::new(),
            activation_batch: RuntimeInclusiveRange {
                minimum: 1,
                maximum: 8,
            },
            context_activations: RuntimeInclusiveRange {
                minimum: 0,
                maximum: 65_536,
            },
            state_activations: RuntimeInclusiveRange {
                minimum: 0,
                maximum: 65_536,
            },
            speculative_draft_token_counts: vec![3],
            residency_policies: vec!["demand_paged".to_string()],
        },
        placement: RuntimePlacementPredicate {
            mode: mode.to_string(),
            minimum_device_count: profiles.len(),
            maximum_device_count: profiles.len(),
            required_interconnects: if profiles.len() > 1 {
                vec!["pcie".to_string()]
            } else {
                Vec::new()
            },
        },
    }
}

#[test]
fn selector_rejects_an_unmeasured_physical_profile_with_the_same_capability_class() {
    let measured = profile(
        HardwareDeviceKind::Gpu,
        "gpu-measured",
        "gfx-fixture",
        "vulkan",
    );
    let unmeasured = profile(
        HardwareDeviceKind::Gpu,
        "gpu-unmeasured",
        "gfx-fixture",
        "vulkan",
    );
    assert_eq!(measured.capability_class, unmeasured.capability_class);
    assert_ne!(measured.profile_id, unmeasured.profile_id);
    let catalog = RuntimeImplementationCatalog {
        package_id: "package".to_string(),
        package_root: PathBuf::from("."),
        stage_status: "optimized".to_string(),
        scopes: BTreeMap::new(),
        exact_baseline: RuntimeExactImplementation {
            artifact_ref: "exact.json".to_string(),
            contract_digest: "exact".to_string(),
            mutable: false,
        },
        implementations: vec![loaded_implementation(
            "implementation_measured_elsewhere",
            &["layer"],
            &["scope_layer"],
            predicate(&[&measured], "local"),
            1_000,
            800,
            0,
        )],
    };
    let request = request(
        vec![selection_device("gpu0", unmeasured)],
        &[("layer0", "layer", &["gpu0"])],
        &[],
    );

    let report = catalog.select(&request).unwrap();

    assert!(report.selected.is_empty());
    assert_eq!(report.exact_instance_ids, vec!["layer0"]);
    assert!(
        report.rejected[0]
            .reasons
            .iter()
            .any(|reason| reason.contains("were not measured"))
    );
}

fn loaded_implementation(
    implementation_id: &str,
    source_components: &[&str],
    scope_ids: &[&str],
    predicate: RuntimeImplementationPredicate,
    reference_latency_ns: u64,
    candidate_latency_ns: u64,
    conversion_ns: u64,
) -> LoadedRuntimeImplementation {
    let compared = RuntimeComparedWorkload {
        workload_id: "workload_decode".to_string(),
        decision: "materially_faster".to_string(),
        paired: RuntimePairedComparison {
            speedup_ppm: i64::try_from(
                (reference_latency_ns.saturating_sub(candidate_latency_ns)) * 1_000_000
                    / reference_latency_ns,
            )
            .unwrap(),
            candidate_is_faster: candidate_latency_ns < reference_latency_ns,
        },
    };
    LoadedRuntimeImplementation {
        implementation: RuntimeImplementation {
            implementation_id: implementation_id.to_string(),
            candidate_id: format!("candidate_{implementation_id}"),
            scope_ids: scope_ids.iter().map(|value| (*value).to_string()).collect(),
            source_contract_digests: scope_ids
                .iter()
                .map(|value| format!("digest_{value}"))
                .collect(),
            representation: json!({
                "kind": "fixture_native_island"
            }),
            behavioral_contract: json!({"mode": "exact"}),
            runtime_predicate: predicate,
            artifact_bundle: RuntimeImplementationArtifactBundle {
                root_ref: format!("optimization/{implementation_id}"),
                candidate_integrity_ref: format!("optimization/{implementation_id}/integrity.json"),
                mount_plan_ref: format!(
                    "optimization/{implementation_id}/candidate/contracts/mount_plan.json"
                ),
                candidate_integrity_digest: "digest".to_string(),
                artifact_count: 1,
            },
            evidence: RuntimeImplementationEvidence {
                promotion_decision_ref: "promotion.json".to_string(),
                candidate_contract_ref: "candidate.json".to_string(),
                construction_record_ref: "construction.json".to_string(),
                prebenchmark_record_ref: "prebenchmark.json".to_string(),
                benchmark_record_ref: "benchmark.json".to_string(),
                validation_record_ref: "validation.json".to_string(),
                analysis_run_refs: Vec::new(),
                hardware_profile_refs: Vec::new(),
            },
            provenance: json!({"provider": "fixture"}),
            comparison: RuntimeImplementationComparison {
                exact_implementation_id: "exact".to_string(),
                exact_contract_digest: "exact_digest".to_string(),
                benchmark_id: "benchmark".to_string(),
                benchmark_decision: "materially_faster".to_string(),
                workloads: vec![compared.clone()],
                validation_id: "validation".to_string(),
                validation_status: "passed".to_string(),
                behavioral_contract: json!({"mode": "exact"}),
            },
            decision_reason: "measured exact win".to_string(),
        },
        source_component_ids: source_components
            .iter()
            .map(|value| (*value).to_string())
            .collect(),
        workload_metrics: vec![RuntimeImplementationWorkloadMetrics {
            workload_id: compared.workload_id,
            phase: "decode".to_string(),
            activation_batch_width: 1,
            context_activations: 4096,
            state_activations: 4096,
            reference_latency_ns,
            candidate_latency_ns,
            conversion_ns,
            conversion_bytes: conversion_ns,
            boundary_count: u64::from(conversion_ns > 0),
            speedup_ppm: compared.paired.speedup_ppm,
        }],
        candidate_root: PathBuf::from(format!("/fixture/{implementation_id}/candidate")),
        mount_plan: RuntimeMountPlan {
            schema: RUNTIME_MOUNT_PLAN_SCHEMA.to_string(),
            candidate_id: format!("candidate_{implementation_id}"),
            adapter_id: VULKAN_STREAM_CIRCUIT_OVERLAY_ADAPTER.to_string(),
            regions: vec![RuntimeMountRegion {
                replacements: source_components
                    .iter()
                    .map(|source_component_id| RuntimeReplacement::Component {
                        source_component_id: (*source_component_id).to_string(),
                        overlay_ref: format!("overlays/{source_component_id}.json"),
                    })
                    .collect(),
            }],
            tensor_index_refs: Vec::new(),
        },
    }
}

fn selection_device(logical: &str, profile: HardwareProcessProfile) -> RuntimeSelectionDevice {
    RuntimeSelectionDevice {
        logical_device_id: logical.to_string(),
        physical_device_id: profile.hardware_identity.stable_device_id.clone(),
        profile,
    }
}

fn request(
    mut devices: Vec<RuntimeSelectionDevice>,
    placements: &[(&str, &str, &[&str])],
    edges: &[(&str, &str)],
) -> RuntimeSelectionRequest {
    devices.sort_by(|left, right| left.logical_device_id.cmp(&right.logical_device_id));
    let mut instances = placements
        .iter()
        .map(
            |(instance_id, source_component_id, device_ids)| RuntimeSelectionInstance {
                instance_id: (*instance_id).to_string(),
                source_component_id: (*source_component_id).to_string(),
                logical_device_ids: device_ids
                    .iter()
                    .map(|value| (*value).to_string())
                    .collect(),
            },
        )
        .collect::<Vec<_>>();
    instances.sort_by(|left, right| left.instance_id.cmp(&right.instance_id));
    RuntimeSelectionRequest {
        execution: RuntimeExecutionEnvelope {
            phases: vec!["decode".to_string()],
            activation_batch: RuntimeInclusiveRange {
                minimum: 1,
                maximum: 1,
            },
            context_activations: RuntimeInclusiveRange {
                minimum: 4096,
                maximum: 4096,
            },
            state_activations: RuntimeInclusiveRange {
                minimum: 4096,
                maximum: 4096,
            },
            speculative_draft_tokens: 3,
            residency_policy: "demand_paged".to_string(),
        },
        devices,
        instances,
        edges: edges
            .iter()
            .map(
                |(source_instance_id, destination_instance_id)| RuntimeSelectionEdge {
                    source_instance_id: (*source_instance_id).to_string(),
                    destination_instance_id: (*destination_instance_id).to_string(),
                },
            )
            .collect(),
        exact_baseline_compatible: true,
    }
}

#[test]
fn paired_comparison_uses_only_the_binary_microbenchmark_decision() {
    let comparison: RuntimePairedComparison = serde_json::from_value(json!({
        "speedup_ppm": 1,
        "candidate_is_faster": true,
    }))
    .unwrap();
    assert_eq!(comparison.speedup_ppm, 1);
    assert!(comparison.candidate_is_faster);

    let statistical_record = serde_json::from_value::<RuntimePairedComparison>(json!({
        "geometric_speedup_ppm": 1,
        "confidence_interval_low_ppm": 1,
        "confidence_interval_high_ppm": 1,
        "relative_ci_width_ppm": 1,
        "order_bias_ppm": 0,
    }))
    .unwrap_err();
    assert!(statistical_record.to_string().contains("unknown field"));
}

#[test]
fn selector_prefers_measured_island_savings_including_conversion_costs() {
    let gpu = profile(HardwareDeviceKind::Gpu, "gpu-a", "gfx-fixture", "vulkan");
    let target = predicate(&[&gpu], "local");
    let catalog = RuntimeImplementationCatalog {
        package_id: "package".to_string(),
        package_root: PathBuf::from("."),
        stage_status: "optimized".to_string(),
        exact_baseline: RuntimeExactImplementation {
            artifact_ref: "exact.json".to_string(),
            contract_digest: "exact".to_string(),
            mutable: false,
        },
        scopes: BTreeMap::new(),
        implementations: vec![
            loaded_implementation(
                "implementation_narrow_a",
                &["a"],
                &["scope_a"],
                target.clone(),
                1_000,
                900,
                20,
            ),
            loaded_implementation(
                "implementation_narrow_b",
                &["b"],
                &["scope_b"],
                target.clone(),
                1_000,
                900,
                20,
            ),
            loaded_implementation(
                "implementation_wide",
                &["a", "b"],
                &["scope_a", "scope_b"],
                target,
                2_000,
                1_650,
                10,
            ),
        ],
    };
    let request = request(
        vec![selection_device("gpu0", gpu)],
        &[("a0", "a", &["gpu0"]), ("b0", "b", &["gpu0"])],
        &[("a0", "b0")],
    );

    let report = catalog.select(&request).unwrap();

    assert_eq!(report.selected.len(), 1);
    assert_eq!(report.selected[0].implementation_id, "implementation_wide");
    assert_eq!(report.exact_instance_ids, Vec::<String>::new());
    assert_eq!(report.total_estimated_saved_ns, 350);
    assert_eq!(report.total_conversion_ns, 10);
    assert!(report.rejected.iter().any(|rejection| {
        rejection
            .reasons
            .iter()
            .any(|reason| reason.contains("higher-value"))
    }));
}

#[test]
fn selector_uses_the_weakest_measured_anchor_across_a_sustained_regime() {
    let gpu = profile(HardwareDeviceKind::Gpu, "gpu-a", "gfx-fixture", "vulkan");
    let mut implementation = loaded_implementation(
        "implementation_sustained",
        &["layer"],
        &["scope_layer"],
        predicate(&[&gpu], "local"),
        1_000,
        800,
        4,
    );
    implementation
        .workload_metrics
        .push(RuntimeImplementationWorkloadMetrics {
            workload_id: "workload_long_context".to_string(),
            phase: "decode".to_string(),
            activation_batch_width: 1,
            context_activations: 65_536,
            state_activations: 65_536,
            reference_latency_ns: 2_000,
            candidate_latency_ns: 1_975,
            conversion_ns: 9,
            conversion_bytes: 9,
            boundary_count: 1,
            speedup_ppm: 12_500,
        });
    let catalog = RuntimeImplementationCatalog {
        package_id: "package".to_string(),
        package_root: PathBuf::from("."),
        stage_status: "optimized".to_string(),
        exact_baseline: RuntimeExactImplementation {
            artifact_ref: "exact.json".to_string(),
            contract_digest: "exact".to_string(),
            mutable: false,
        },
        scopes: BTreeMap::new(),
        implementations: vec![implementation],
    };
    let request = request(
        vec![selection_device("gpu0", gpu)],
        &[("layer0", "layer", &["gpu0"])],
        &[],
    );

    let report = catalog.select(&request).unwrap();

    assert_eq!(report.selected.len(), 1);
    assert_eq!(report.selected[0].estimated_saved_ns, 25);
    assert_eq!(report.selected[0].speedup_ppm, 12_500);
    assert_eq!(report.selected[0].conversion_ns, 9);
}

#[test]
fn selector_prices_only_alternative_phases_in_a_phase_selective_implementation() {
    let gpu = profile(HardwareDeviceKind::Gpu, "gpu-a", "gfx-fixture", "vulkan");
    let mut target = predicate(&[&gpu], "local");
    target.execution.phases = vec!["decode".to_string(), "prefill".to_string()];
    target.execution.alternative_phases = vec!["decode".to_string()];
    target.execution.source_retained_phases = vec!["prefill".to_string()];
    let catalog = RuntimeImplementationCatalog {
        package_id: "package".to_string(),
        package_root: PathBuf::from("."),
        stage_status: "optimized".to_string(),
        exact_baseline: RuntimeExactImplementation {
            artifact_ref: "exact.json".to_string(),
            contract_digest: "exact".to_string(),
            mutable: false,
        },
        scopes: BTreeMap::new(),
        implementations: vec![loaded_implementation(
            "implementation_decode_only",
            &["layer"],
            &["scope_layer"],
            target,
            1_000,
            800,
            0,
        )],
    };
    let mut mixed_request = request(
        vec![selection_device("gpu0", gpu.clone())],
        &[("layer0", "layer", &["gpu0"])],
        &[],
    );
    mixed_request.execution.phases = vec!["decode".to_string(), "prefill".to_string()];

    let mixed = catalog.select(&mixed_request).unwrap();
    assert_eq!(mixed.selected.len(), 1);
    assert_eq!(mixed.selected[0].speedup_ppm, 200_000);

    mixed_request.execution.phases = vec!["prefill".to_string()];
    let source_only = catalog.select(&mixed_request).unwrap();
    assert!(source_only.selected.is_empty());
    assert!(
        source_only.rejected[0]
            .reasons
            .iter()
            .any(|reason| reason.contains("does not execute an alternative phase"))
    );
}

#[test]
fn predicates_distinguish_cpu_single_gpu_multi_gpu_and_mixed_targets() {
    let cpu = profile(HardwareDeviceKind::Cpu, "cpu-a", "zen-fixture", "native");
    let gpu_a = profile(HardwareDeviceKind::Gpu, "gpu-a", "gfx-fixture", "vulkan");
    let gpu_b = profile(HardwareDeviceKind::Gpu, "gpu-b", "gfx-fixture", "vulkan");
    let cpu_device = selection_device("cpu0", cpu.clone());
    let gpu_a_device = selection_device("gpu0", gpu_a.clone());
    let gpu_b_device = selection_device("gpu1", gpu_b.clone());
    let execution = RuntimeExecutionEnvelope {
        phases: vec!["decode".to_string()],
        activation_batch: RuntimeInclusiveRange {
            minimum: 1,
            maximum: 1,
        },
        context_activations: RuntimeInclusiveRange {
            minimum: 4096,
            maximum: 4096,
        },
        state_activations: RuntimeInclusiveRange {
            minimum: 4096,
            maximum: 4096,
        },
        speculative_draft_tokens: 3,
        residency_policy: "demand_paged".to_string(),
    };

    assert!(
        predicate(&[&cpu], "local")
            .mismatch_reasons(&execution, &[&cpu_device])
            .is_empty()
    );
    assert!(
        predicate(&[&gpu_a], "local")
            .mismatch_reasons(&execution, &[&gpu_a_device])
            .is_empty()
    );
    assert!(
        predicate(&[&gpu_a, &gpu_b], "distributed")
            .mismatch_reasons(&execution, &[&gpu_a_device, &gpu_b_device],)
            .is_empty()
    );
    assert!(
        predicate(&[&cpu, &gpu_a], "distributed")
            .mismatch_reasons(&execution, &[&cpu_device, &gpu_a_device],)
            .is_empty()
    );
    assert!(
        predicate(&[&cpu, &gpu_a], "distributed")
            .mismatch_reasons(&execution, &[&gpu_a_device, &gpu_b_device],)
            .iter()
            .any(|reason| reason.contains("were not measured"))
    );
    let mut unsupported_gpu = gpu_b.clone();
    unsupported_gpu.capability_class = "hardware_capability_unsupported".to_string();
    let unsupported_gpu_device = selection_device("gpu1", unsupported_gpu);
    let mut flexible_gpu_predicate = predicate(&[&gpu_a], "local");
    flexible_gpu_predicate.placement.mode = "either".to_string();
    flexible_gpu_predicate.placement.maximum_device_count = 2;
    assert!(
        flexible_gpu_predicate
            .mismatch_reasons(&execution, &[&gpu_a_device, &unsupported_gpu_device],)
            .iter()
            .any(|reason| reason.contains("capability classes"))
    );
    let mut different_speculative_mode = execution.clone();
    different_speculative_mode.speculative_draft_tokens = 0;
    assert!(
        predicate(&[&gpu_a], "local")
            .mismatch_reasons(&different_speculative_mode, &[&gpu_a_device],)
            .iter()
            .any(|reason| reason.contains("speculative draft-token count"))
    );
    let mut different_residency = execution.clone();
    different_residency.residency_policy = "eager".to_string();
    assert!(
        predicate(&[&gpu_a], "local")
            .mismatch_reasons(&different_residency, &[&gpu_a_device],)
            .iter()
            .any(|reason| reason.contains("residency policy"))
    );
}

#[test]
fn selector_refuses_uncovered_regions_when_exact_execution_is_incompatible() {
    let cpu = profile(HardwareDeviceKind::Cpu, "cpu-a", "zen-fixture", "native");
    let catalog = RuntimeImplementationCatalog {
        package_id: "package".to_string(),
        package_root: PathBuf::from("."),
        stage_status: "optimized".to_string(),
        exact_baseline: RuntimeExactImplementation {
            artifact_ref: "exact.json".to_string(),
            contract_digest: "exact".to_string(),
            mutable: false,
        },
        scopes: BTreeMap::new(),
        implementations: Vec::new(),
    };
    let mut request = request(
        vec![selection_device("cpu0", cpu)],
        &[("layer0", "layer", &["cpu0"])],
        &[],
    );
    request.exact_baseline_compatible = false;

    let error = catalog.select(&request).unwrap_err();

    assert_eq!(error.kind(), std::io::ErrorKind::Unsupported);
    assert!(error.to_string().contains("layer0"));
}

#[test]
fn one_implementation_exposes_each_duplicate_source_instance_as_an_independent_application() {
    let gpu_a = profile(HardwareDeviceKind::Gpu, "gpu-a", "gfx-fixture", "vulkan");
    let gpu_b = profile(HardwareDeviceKind::Gpu, "gpu-b", "gfx-fixture", "vulkan");
    let mut local_on_either_measured_gpu = predicate(&[&gpu_a, &gpu_b], "distributed");
    local_on_either_measured_gpu.placement = RuntimePlacementPredicate {
        mode: "local".to_string(),
        minimum_device_count: 1,
        maximum_device_count: 1,
        required_interconnects: Vec::new(),
    };
    let catalog = RuntimeImplementationCatalog {
        package_id: "package".to_string(),
        package_root: PathBuf::from("."),
        stage_status: "optimized".to_string(),
        exact_baseline: RuntimeExactImplementation {
            artifact_ref: "exact.json".to_string(),
            contract_digest: "exact".to_string(),
            mutable: false,
        },
        scopes: BTreeMap::new(),
        implementations: vec![loaded_implementation(
            "implementation_layer",
            &["layer"],
            &["scope_layer"],
            local_on_either_measured_gpu,
            1_000,
            800,
            0,
        )],
    };
    let request = request(
        vec![
            selection_device("gpu0", gpu_a),
            selection_device("gpu1", gpu_b),
        ],
        &[
            ("layer_original", "layer", &["gpu0"]),
            ("layer_duplicate", "layer", &["gpu1"]),
        ],
        &[("layer_original", "layer_duplicate")],
    );

    let report = catalog.select(&request).unwrap();

    assert_eq!(report.selected.len(), 2);
    assert_eq!(
        report
            .selected
            .iter()
            .map(|selection| selection.instance_ids.as_slice())
            .collect::<Vec<_>>(),
        vec![
            ["layer_duplicate".to_string()].as_slice(),
            ["layer_original".to_string()].as_slice(),
        ]
    );
}

#[test]
fn one_implementation_exposes_disconnected_semantic_regions_independently() {
    let gpu = profile(HardwareDeviceKind::Gpu, "gpu", "gfx-fixture", "vulkan");
    let mut implementation = loaded_implementation(
        "implementation_regions",
        &["layer_a", "layer_b"],
        &["scope_a", "scope_b"],
        predicate(&[&gpu], "local"),
        1_000,
        800,
        0,
    );
    implementation.mount_plan.regions = vec![
        RuntimeMountRegion {
            replacements: vec![RuntimeReplacement::Component {
                source_component_id: "layer_a".to_string(),
                overlay_ref: "overlays/layer_a.json".to_string(),
            }],
        },
        RuntimeMountRegion {
            replacements: vec![RuntimeReplacement::Component {
                source_component_id: "layer_b".to_string(),
                overlay_ref: "overlays/layer_b.json".to_string(),
            }],
        },
    ];
    let catalog = RuntimeImplementationCatalog {
        package_id: "package".to_string(),
        package_root: PathBuf::from("."),
        stage_status: "optimized".to_string(),
        exact_baseline: RuntimeExactImplementation {
            artifact_ref: "exact.json".to_string(),
            contract_digest: "exact".to_string(),
            mutable: false,
        },
        scopes: BTreeMap::new(),
        implementations: vec![implementation],
    };
    let request = request(
        vec![selection_device("gpu0", gpu)],
        &[
            ("instance_a", "layer_a", &["gpu0"]),
            ("exact_middle", "middle", &["gpu0"]),
            ("instance_b", "layer_b", &["gpu0"]),
        ],
        &[
            ("instance_a", "exact_middle"),
            ("exact_middle", "instance_b"),
        ],
    );

    let report = catalog.select(&request).unwrap();

    assert_eq!(report.selected.len(), 2);
    assert_eq!(
        report
            .selected
            .iter()
            .map(|selection| selection.instance_ids.as_slice())
            .collect::<Vec<_>>(),
        vec![
            ["instance_a".to_string()].as_slice(),
            ["instance_b".to_string()].as_slice(),
        ]
    );
    assert_eq!(report.exact_instance_ids, ["exact_middle".to_string()]);
}

#[test]
fn catalog_loader_accepts_the_self_contained_compiled_fixture() {
    let package = crate::test_support::tiny_model_dir();
    let catalog = RuntimeImplementationCatalog::load(
        &package,
        "optimization/stage.json",
        "model_d119caf1_vulkan_resident",
    )
    .unwrap();

    assert_eq!(catalog.stage_status, "exact_baseline_retained");
    assert_eq!(catalog.scopes.len(), 52);
    assert!(catalog.implementations.is_empty());
    assert!(!catalog.exact_baseline.mutable);
}
