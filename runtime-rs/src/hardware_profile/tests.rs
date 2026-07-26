use super::*;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;

fn synthetic_cpu_facts() -> CpuHardwareFacts {
    CpuHardwareFacts {
        architecture: "x86_64".to_string(),
        vendor_id: "AuthenticAMD".to_string(),
        family: "26".to_string(),
        model: "68".to_string(),
        stepping: "0".to_string(),
        model_name: "Synthetic CPU".to_string(),
        flags: BTreeSet::from([
            "avx".to_string(),
            "avx2".to_string(),
            "avx512f".to_string(),
            "avx512_bf16".to_string(),
            "bmi1".to_string(),
            "bmi2".to_string(),
            "fma".to_string(),
            "popcnt".to_string(),
        ]),
        logical_processor_count: 32,
        physical_core_count: 16,
        socket_count: 1,
        total_memory_bytes: 128 * 1024 * 1024 * 1024,
        cache_domains: vec![HardwareMemoryDomain {
            name: "cpu_l3_unified_cache".to_string(),
            kind: "unified_cache".to_string(),
            capacity_bytes: 64 * 1024 * 1024,
            host_visible: true,
            device_local: true,
            coherent: true,
            cached: true,
            minimum_alignment_bytes: 64,
            properties: BTreeMap::new(),
        }],
        numa_node_count: 1,
        machine_identity: "cpu_00000000000000000000000000000000".to_string(),
    }
}

#[test]
fn synthetic_cpu_profile_covers_distinct_hardware_processes() {
    let profile = build_cpu_hardware_profile(synthetic_cpu_facts()).unwrap();
    let by_name = profile
        .processes
        .iter()
        .map(|process| (process.name.as_str(), process))
        .collect::<BTreeMap<_, _>>();
    for required in [
        "atomics",
        "bit_manipulation",
        "branch_execution",
        "cache_hierarchy",
        "dma_engines",
        "hardware_prefetch",
        "host_memory_copy",
        "instruction_cache",
        "main_memory",
        "matrix_extension",
        "memory_bandwidth",
        "micro_op_cache",
        "numa_memory_policy",
        "out_of_order_control_flow",
        "scalar_floating_point",
        "scalar_integer",
        "simd_vector",
    ] {
        assert!(
            by_name.contains_key(required),
            "missing CPU process {required}"
        );
    }

    assert_eq!(
        profile.hardware_identity.device_kind,
        HardwareDeviceKind::Cpu
    );
    assert_eq!(
        by_name["simd_vector"].limits["maximum_vector_width_bits"],
        512
    );
    assert!(
        by_name["simd_vector"]
            .numeric_formats
            .contains(&"bf16".to_string())
    );
    assert_eq!(
        by_name["matrix_extension"].availability,
        HardwareProcessAvailability::Unavailable
    );
    assert_eq!(
        by_name["micro_op_cache"].availability,
        HardwareProcessAvailability::Opaque
    );
    assert_eq!(profile.measurements, Vec::<HardwareMeasurement>::new());
    assert!(profile.identity_extensions.is_empty());
    assert!(profile.runtime_bindings.is_empty());
    profile.validate().unwrap();
}

#[test]
fn capability_class_ignores_physical_identity_but_profile_id_does_not() {
    let first = build_cpu_hardware_profile(synthetic_cpu_facts()).unwrap();
    let mut second_facts = synthetic_cpu_facts();
    second_facts.machine_identity = "cpu_11111111111111111111111111111111".to_string();
    let second = build_cpu_hardware_profile(second_facts).unwrap();

    assert_eq!(first.capability_class, second.capability_class);
    assert_ne!(first.profile_id, second.profile_id);
}

#[test]
fn calibration_measurements_change_profile_identity_not_capability_class() {
    let uncalibrated = build_cpu_hardware_profile(synthetic_cpu_facts()).unwrap();
    let calibrated = uncalibrated
        .clone()
        .with_measurements(vec![HardwareMeasurement {
            name: "simd_vector_f32_throughput".to_string(),
            unit: "operations_per_second".to_string(),
            regime: BTreeMap::from([("vector_width".to_string(), "512".to_string())]),
            samples: vec![1_000, 1_010, 995],
        }])
        .unwrap();

    assert_eq!(uncalibrated.capability_class, calibrated.capability_class);
    assert_ne!(uncalibrated.profile_id, calibrated.profile_id);
    calibrated.validate().unwrap();
}

#[test]
fn hardware_inventory_fails_closed_on_duplicate_or_corrupt_profiles() {
    let profile = build_cpu_hardware_profile(synthetic_cpu_facts()).unwrap();
    assert!(HardwareProcessInventory::new(vec![profile.clone(), profile]).is_err());

    let mut corrupt = build_cpu_hardware_profile(synthetic_cpu_facts()).unwrap();
    corrupt.processes[0].name.clear();
    assert!(corrupt.validate().is_err());
}

#[test]
fn hardware_profile_deserialization_fails_closed() {
    let profile = build_cpu_hardware_profile(synthetic_cpu_facts()).unwrap();
    let mut document = serde_json::to_value(&profile).unwrap();
    document.as_object_mut().unwrap().insert(
        "unknown_capability_fact".to_string(),
        serde_json::json!(true),
    );
    assert!(serde_json::from_value::<HardwareProcessProfile>(document).is_err());

    let mut document = serde_json::to_value(&profile).unwrap();
    document
        .as_object_mut()
        .unwrap()
        .remove("capability_extensions");
    assert!(serde_json::from_value::<HardwareProcessProfile>(document).is_err());
}

#[test]
fn live_cpu_discovery_returns_a_stable_valid_profile() {
    let first = discover_cpu_hardware_profile().unwrap();
    let second = discover_cpu_hardware_profile().unwrap();

    assert_eq!(first, second);
    assert_eq!(first.hardware_identity.device_kind, HardwareDeviceKind::Cpu);
    assert!(first.hardware_identity.stable_device_id.starts_with("cpu_"));
    assert!(!first.memory_domains.is_empty());
    first.validate().unwrap();
}

#[test]
fn cpu_cache_discovery_preserves_distinct_shared_cache_domains() {
    let root = std::env::temp_dir().join(format!("nerve-cpu-cache-profile-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    for (cpu, shared) in [("cpu0", "0-1"), ("cpu1", "0-1"), ("cpu2", "2-3")] {
        let cache = root.join(cpu).join("cache").join("index3");
        fs::create_dir_all(&cache).unwrap();
        for (name, value) in [
            ("level", "3"),
            ("type", "Unified"),
            ("size", "32M"),
            ("coherency_line_size", "64"),
            ("shared_cpu_list", shared),
        ] {
            fs::write(cache.join(name), value).unwrap();
        }
    }

    let domains = discover_cpu_cache_domains(&root).unwrap();
    fs::remove_dir_all(&root).unwrap();

    assert_eq!(domains.len(), 2);
    assert_eq!(
        domains
            .iter()
            .map(|domain| domain.capacity_bytes)
            .sum::<u64>(),
        64 * 1024 * 1024
    );
    assert_ne!(domains[0].name, domains[1].name);
}
