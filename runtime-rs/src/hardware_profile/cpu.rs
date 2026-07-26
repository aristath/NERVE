use super::{
    HardwareDeviceKind, HardwareIdentity, HardwareInterconnect, HardwareMemoryDomain,
    HardwareProcessAvailability, HardwareProcessCapability, HardwareProcessCategory,
    HardwareProcessProfile, HardwareProcessProfileDefinition, HardwareProcessProgrammability,
    HardwareProfileProvenance, stable_hardware_id,
};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuHardwareFacts {
    pub architecture: String,
    pub vendor_id: String,
    pub family: String,
    pub model: String,
    pub stepping: String,
    pub model_name: String,
    pub flags: BTreeSet<String>,
    pub logical_processor_count: u64,
    pub physical_core_count: u64,
    pub socket_count: u64,
    pub total_memory_bytes: u64,
    pub cache_domains: Vec<HardwareMemoryDomain>,
    pub numa_node_count: u64,
    pub machine_identity: String,
}

pub fn discover_cpu_hardware_profile() -> Result<HardwareProcessProfile, String> {
    build_cpu_hardware_profile(discover_cpu_hardware_facts()?)
}

pub fn discover_cpu_hardware_facts() -> Result<CpuHardwareFacts, String> {
    let cpuinfo = fs::read_to_string("/proc/cpuinfo")
        .map_err(|error| format!("could not read /proc/cpuinfo: {error}"))?;
    let meminfo = fs::read_to_string("/proc/meminfo")
        .map_err(|error| format!("could not read /proc/meminfo: {error}"))?;
    let architecture = std::env::consts::ARCH.to_string();
    let fields = first_cpuinfo_record(&cpuinfo);
    let vendor_id = cpuinfo_field(&fields, &["vendor_id", "CPU implementer"])
        .unwrap_or("unknown")
        .to_string();
    let family = cpuinfo_field(&fields, &["cpu family", "CPU architecture"])
        .unwrap_or("unknown")
        .to_string();
    let model = cpuinfo_field(&fields, &["model", "CPU part"])
        .unwrap_or("unknown")
        .to_string();
    let stepping = cpuinfo_field(&fields, &["stepping", "CPU revision"])
        .unwrap_or("unknown")
        .to_string();
    let model_name = cpuinfo_field(&fields, &["model name", "Processor"])
        .unwrap_or("unknown CPU")
        .to_string();
    let flags = cpuinfo_field(&fields, &["flags", "Features"])
        .unwrap_or("")
        .split_whitespace()
        .map(str::to_string)
        .collect::<BTreeSet<_>>();
    let logical_processor_count = cpuinfo
        .lines()
        .filter(|line| line.starts_with("processor"))
        .count()
        .max(1) as u64;
    let topology = cpu_topology(&cpuinfo);
    let physical_core_count = topology.core_ids.len().max(1) as u64;
    let socket_count = topology.socket_ids.len().max(1) as u64;
    let total_memory_bytes = parse_mem_total_bytes(&meminfo)?;
    let cache_domains = discover_cpu_cache_domains(Path::new("/sys/devices/system/cpu"))?;
    let numa_node_count =
        discover_numbered_entries(Path::new("/sys/devices/system/node"), "node").max(1) as u64;
    let machine_identity =
        stable_machine_identity(&architecture, &vendor_id, &family, &model, &stepping)?;
    Ok(CpuHardwareFacts {
        architecture,
        vendor_id,
        family,
        model,
        stepping,
        model_name,
        flags,
        logical_processor_count,
        physical_core_count,
        socket_count,
        total_memory_bytes,
        cache_domains,
        numa_node_count,
        machine_identity,
    })
}

pub fn build_cpu_hardware_profile(
    facts: CpuHardwareFacts,
) -> Result<HardwareProcessProfile, String> {
    if facts.logical_processor_count == 0
        || facts.physical_core_count == 0
        || facts.socket_count == 0
        || facts.total_memory_bytes == 0
    {
        return Err("CPU hardware facts contain zero-sized topology or memory".to_string());
    }
    let identity = HardwareIdentity {
        device_kind: HardwareDeviceKind::Cpu,
        stable_device_id: facts.machine_identity.clone(),
        name: facts.model_name.clone(),
        vendor_id: facts.vendor_id.clone(),
        device_id: format!("{}:{}:{}", facts.family, facts.model, facts.stepping),
        architecture: facts.architecture.clone(),
        physical_location: "system_cpu".to_string(),
    };
    let mut processes = vec![
        cpu_scalar_process(&facts),
        cpu_float_process(&facts),
        cpu_control_flow_process(&facts),
        cpu_branch_process(&facts),
        cpu_simd_process(&facts),
        cpu_matrix_process(&facts),
        cpu_bit_process(&facts),
        cpu_cache_process(&facts),
        cpu_memory_process(&facts),
        cpu_memory_bandwidth_process(&facts),
        cpu_prefetch_process(&facts),
        cpu_instruction_cache_process(&facts),
        cpu_micro_op_cache_process(&facts),
        cpu_atomic_process(&facts),
        cpu_copy_process(&facts),
        cpu_numa_process(&facts),
        cpu_dma_process(&facts),
    ];
    processes.sort_by(|left, right| left.name.cmp(&right.name));
    let mut memory_domains = facts.cache_domains.clone();
    memory_domains.push(HardwareMemoryDomain {
        name: "host_main_memory".to_string(),
        kind: "system_ram".to_string(),
        capacity_bytes: facts.total_memory_bytes,
        host_visible: true,
        device_local: true,
        coherent: true,
        cached: true,
        minimum_alignment_bytes: 64,
        properties: BTreeMap::from([
            (
                "numa_node_count".to_string(),
                facts.numa_node_count.to_string(),
            ),
            (
                "address_space".to_string(),
                "native_virtual_memory".to_string(),
            ),
        ]),
    });
    let interconnects = vec![HardwareInterconnect {
        name: "cpu_numa_fabric".to_string(),
        kind: "numa".to_string(),
        availability: if facts.numa_node_count > 1 {
            HardwareProcessAvailability::Available
        } else {
            HardwareProcessAvailability::Unavailable
        },
        api: "linux_native".to_string(),
        operations: if facts.numa_node_count > 1 {
            vec![
                "cpu_affinity".to_string(),
                "memory_policy".to_string(),
                "remote_memory_access".to_string(),
            ]
        } else {
            Vec::new()
        },
        properties: BTreeMap::from([("node_count".to_string(), facts.numa_node_count.to_string())]),
    }];
    let provenance = HardwareProfileProvenance {
        api: "native".to_string(),
        api_version: std::env::consts::ARCH.to_string(),
        driver: "linux_kernel".to_string(),
        driver_version: kernel_release(),
        compiler: env!("NERVE_HARDWARE_DISCOVERY_FINGERPRINT").to_string(),
        operating_system: std::env::consts::OS.to_string(),
        discovery_backend: "procfs_sysfs_and_runtime_detection".to_string(),
    };
    HardwareProcessProfile::create(HardwareProcessProfileDefinition {
        hardware_identity: identity,
        processes,
        memory_domains,
        interconnects,
        provenance,
        capability_extensions: BTreeMap::from([(
            "native_cpu".to_string(),
            json!({
                "logical_processor_count": facts.logical_processor_count,
                "physical_core_count": facts.physical_core_count,
                "socket_count": facts.socket_count,
                "numa_node_count": facts.numa_node_count,
                "flags": facts.flags,
            }),
        )]),
        identity_extensions: BTreeMap::new(),
        runtime_bindings: BTreeMap::new(),
    })
}

fn cpu_scalar_process(facts: &CpuHardwareFacts) -> HardwareProcessCapability {
    let mut process = available_cpu_process(
        "scalar_integer",
        HardwareProcessCategory::Arithmetic,
        &["add", "compare", "divide", "multiply", "shift"],
    );
    process.numeric_formats = vec![
        "i16".to_string(),
        "i32".to_string(),
        "i64".to_string(),
        "i8".to_string(),
        "u16".to_string(),
        "u32".to_string(),
        "u64".to_string(),
        "u8".to_string(),
    ];
    cpu_topology_limits(&mut process, facts);
    process
}

fn cpu_float_process(facts: &CpuHardwareFacts) -> HardwareProcessCapability {
    let mut process = available_cpu_process(
        "scalar_floating_point",
        HardwareProcessCategory::Arithmetic,
        &["add", "compare", "divide", "fused_multiply_add", "multiply"],
    );
    process.numeric_formats = vec!["f32".to_string(), "f64".to_string()];
    cpu_topology_limits(&mut process, facts);
    process
}

fn cpu_control_flow_process(facts: &CpuHardwareFacts) -> HardwareProcessCapability {
    let mut process = available_cpu_process(
        "out_of_order_control_flow",
        HardwareProcessCategory::ControlFlow,
        &["dependency_scheduling", "instruction_level_parallelism"],
    );
    process.programmability = HardwareProcessProgrammability::Indirect;
    process.properties.insert(
        "exposure".to_string(),
        "microarchitecture_managed".to_string(),
    );
    cpu_topology_limits(&mut process, facts);
    process
}

fn cpu_branch_process(facts: &CpuHardwareFacts) -> HardwareProcessCapability {
    let mut process = available_cpu_process(
        "branch_execution",
        HardwareProcessCategory::ControlFlow,
        &[
            "branch",
            "branch_prediction",
            "indirect_branch",
            "predication",
        ],
    );
    process.programmability = HardwareProcessProgrammability::Indirect;
    cpu_topology_limits(&mut process, facts);
    process
}

fn cpu_simd_process(facts: &CpuHardwareFacts) -> HardwareProcessCapability {
    let width = cpu_simd_width_bits(&facts.flags);
    let availability = if width > 0 {
        HardwareProcessAvailability::Available
    } else {
        HardwareProcessAvailability::Unavailable
    };
    let mut process = HardwareProcessCapability::new(
        "simd_vector",
        HardwareProcessCategory::Arithmetic,
        availability,
        if width > 0 {
            HardwareProcessProgrammability::Direct
        } else {
            HardwareProcessProgrammability::None
        },
        "native",
    );
    if width > 0 {
        process.operations = vec![
            "arithmetic".to_string(),
            "compare".to_string(),
            "dot_product".to_string(),
            "fused_multiply_add".to_string(),
            "gather".to_string(),
            "mask".to_string(),
            "permutation".to_string(),
            "scatter".to_string(),
        ];
        process.numeric_formats = cpu_simd_formats(&facts.flags);
        process
            .limits
            .insert("maximum_vector_width_bits".to_string(), width);
    }
    process.required_features = cpu_simd_feature_flags(&facts.flags);
    cpu_topology_limits(&mut process, facts);
    process
}

fn cpu_matrix_process(facts: &CpuHardwareFacts) -> HardwareProcessCapability {
    let features = ["amx_tile", "amx_int8", "amx_bf16"]
        .into_iter()
        .filter(|feature| facts.flags.contains(*feature))
        .map(str::to_string)
        .collect::<Vec<_>>();
    let mut process = HardwareProcessCapability::new(
        "matrix_extension",
        HardwareProcessCategory::Arithmetic,
        if features.is_empty() {
            HardwareProcessAvailability::Unavailable
        } else {
            HardwareProcessAvailability::Available
        },
        if features.is_empty() {
            HardwareProcessProgrammability::None
        } else {
            HardwareProcessProgrammability::Direct
        },
        "native",
    );
    process.required_features = features;
    if facts.flags.contains("amx_int8") {
        process
            .numeric_formats
            .extend(["i8".to_string(), "u8".to_string()]);
    }
    if facts.flags.contains("amx_bf16") {
        process.numeric_formats.push("bf16".to_string());
    }
    if !process.numeric_formats.is_empty() {
        process.operations = vec![
            "matrix_multiply_accumulate".to_string(),
            "tile_load".to_string(),
            "tile_store".to_string(),
        ];
    }
    process
}

fn cpu_bit_process(facts: &CpuHardwareFacts) -> HardwareProcessCapability {
    let supported = ["bmi1", "bmi2", "lzcnt", "popcnt"]
        .into_iter()
        .filter(|feature| facts.flags.contains(*feature))
        .map(str::to_string)
        .collect::<Vec<_>>();
    let mut process = HardwareProcessCapability::new(
        "bit_manipulation",
        HardwareProcessCategory::Arithmetic,
        if supported.is_empty() {
            HardwareProcessAvailability::Unavailable
        } else {
            HardwareProcessAvailability::Available
        },
        if supported.is_empty() {
            HardwareProcessProgrammability::None
        } else {
            HardwareProcessProgrammability::Direct
        },
        "native",
    );
    process.required_features = supported;
    process.operations = process
        .required_features
        .iter()
        .map(|feature| match feature.as_str() {
            "bmi1" | "bmi2" => "bit_extract_deposit",
            "lzcnt" => "leading_zero_count",
            "popcnt" => "population_count",
            _ => feature,
        })
        .map(str::to_string)
        .collect();
    process
}

fn cpu_cache_process(facts: &CpuHardwareFacts) -> HardwareProcessCapability {
    let mut process = available_cpu_process(
        "cache_hierarchy",
        HardwareProcessCategory::Memory,
        &["cache_line_reuse", "prefetch_target", "temporal_locality"],
    );
    process.programmability = HardwareProcessProgrammability::Indirect;
    process.limits.insert(
        "discovered_cache_domain_count".to_string(),
        facts.cache_domains.len() as u64,
    );
    process
}

fn cpu_memory_process(facts: &CpuHardwareFacts) -> HardwareProcessCapability {
    let mut process = available_cpu_process(
        "main_memory",
        HardwareProcessCategory::Memory,
        &["load", "prefetch", "store", "virtual_memory"],
    );
    process
        .limits
        .insert("capacity_bytes".to_string(), facts.total_memory_bytes);
    process
}

fn cpu_memory_bandwidth_process(facts: &CpuHardwareFacts) -> HardwareProcessCapability {
    let mut process = available_cpu_process(
        "memory_bandwidth",
        HardwareProcessCategory::Memory,
        &["streaming_copy", "streaming_read", "streaming_write"],
    );
    process.programmability = HardwareProcessProgrammability::Indirect;
    process.properties.insert(
        "realized_bandwidth".to_string(),
        "requires_hardware_calibration".to_string(),
    );
    process
        .limits
        .insert("numa_node_count".to_string(), facts.numa_node_count);
    process
}

fn cpu_prefetch_process(_facts: &CpuHardwareFacts) -> HardwareProcessCapability {
    let mut process = available_cpu_process(
        "hardware_prefetch",
        HardwareProcessCategory::Memory,
        &["automatic_prefetch", "software_prefetch_hint"],
    );
    process.programmability = HardwareProcessProgrammability::Indirect;
    process.properties.insert(
        "policy".to_string(),
        "microarchitecture_managed".to_string(),
    );
    process
}

fn cpu_instruction_cache_process(_facts: &CpuHardwareFacts) -> HardwareProcessCapability {
    let mut process = available_cpu_process(
        "instruction_cache",
        HardwareProcessCategory::Memory,
        &["generated_code_residency", "instruction_fetch"],
    );
    process.programmability = HardwareProcessProgrammability::Indirect;
    process
}

fn cpu_micro_op_cache_process(_facts: &CpuHardwareFacts) -> HardwareProcessCapability {
    let mut process = HardwareProcessCapability::new(
        "micro_op_cache",
        HardwareProcessCategory::Memory,
        HardwareProcessAvailability::Opaque,
        HardwareProcessProgrammability::None,
        "native",
    );
    process.properties.insert(
        "reason".to_string(),
        "not exposed by the portable native execution API".to_string(),
    );
    process
}

fn cpu_atomic_process(facts: &CpuHardwareFacts) -> HardwareProcessCapability {
    let mut process = available_cpu_process(
        "atomics",
        HardwareProcessCategory::Synchronization,
        &[
            "compare_exchange",
            "fetch_add",
            "load",
            "memory_fence",
            "store",
        ],
    );
    process.numeric_formats = vec!["u16".to_string(), "u32".to_string(), "u64".to_string()];
    cpu_topology_limits(&mut process, facts);
    process
}

fn cpu_copy_process(_facts: &CpuHardwareFacts) -> HardwareProcessCapability {
    available_cpu_process(
        "host_memory_copy",
        HardwareProcessCategory::Transfer,
        &["copy", "gather", "scatter", "streaming_store"],
    )
}

fn cpu_numa_process(facts: &CpuHardwareFacts) -> HardwareProcessCapability {
    let available = facts.numa_node_count > 1;
    let mut process = HardwareProcessCapability::new(
        "numa_memory_policy",
        HardwareProcessCategory::Memory,
        if available {
            HardwareProcessAvailability::Available
        } else {
            HardwareProcessAvailability::Unavailable
        },
        if available {
            HardwareProcessProgrammability::Direct
        } else {
            HardwareProcessProgrammability::None
        },
        "linux_native",
    );
    if available {
        process.operations = vec![
            "bind_memory".to_string(),
            "bind_thread".to_string(),
            "remote_memory_access".to_string(),
        ];
    }
    process
        .limits
        .insert("numa_node_count".to_string(), facts.numa_node_count);
    process
}

fn cpu_dma_process(_facts: &CpuHardwareFacts) -> HardwareProcessCapability {
    let mut process = HardwareProcessCapability::new(
        "dma_engines",
        HardwareProcessCategory::Transfer,
        HardwareProcessAvailability::Opaque,
        HardwareProcessProgrammability::None,
        "native",
    );
    process.properties.insert(
        "reason".to_string(),
        "no general-purpose native DMA contract is exposed".to_string(),
    );
    process
}

fn available_cpu_process(
    name: &str,
    category: HardwareProcessCategory,
    operations: &[&str],
) -> HardwareProcessCapability {
    let mut process = HardwareProcessCapability::new(
        name,
        category,
        HardwareProcessAvailability::Available,
        HardwareProcessProgrammability::Direct,
        "native",
    );
    process.operations = operations
        .iter()
        .map(|value| (*value).to_string())
        .collect();
    process
}

fn cpu_topology_limits(process: &mut HardwareProcessCapability, facts: &CpuHardwareFacts) {
    process.limits.insert(
        "logical_processor_count".to_string(),
        facts.logical_processor_count,
    );
    process
        .limits
        .insert("physical_core_count".to_string(), facts.physical_core_count);
    process
        .limits
        .insert("socket_count".to_string(), facts.socket_count);
}

fn cpu_simd_width_bits(flags: &BTreeSet<String>) -> u64 {
    if flags.contains("avx512f") {
        512
    } else if flags.contains("avx") || flags.contains("avx2") {
        256
    } else if flags.contains("sse2") || flags.contains("asimd") {
        128
    } else {
        0
    }
}

fn cpu_simd_formats(flags: &BTreeSet<String>) -> Vec<String> {
    let mut formats = vec![
        "f32".to_string(),
        "f64".to_string(),
        "i16".to_string(),
        "i32".to_string(),
        "i64".to_string(),
        "i8".to_string(),
        "u16".to_string(),
        "u32".to_string(),
        "u64".to_string(),
        "u8".to_string(),
    ];
    if flags.contains("avx512_bf16") || flags.contains("bf16") {
        formats.push("bf16".to_string());
    }
    if flags.contains("avx512_fp16") || flags.contains("fphp") {
        formats.push("f16".to_string());
    }
    formats.sort();
    formats
}

fn cpu_simd_feature_flags(flags: &BTreeSet<String>) -> Vec<String> {
    [
        "asimd",
        "avx",
        "avx2",
        "avx512_bf16",
        "avx512_fp16",
        "avx512_vnni",
        "avx512f",
        "fma",
        "sse2",
    ]
    .into_iter()
    .filter(|feature| flags.contains(*feature))
    .map(str::to_string)
    .collect()
}

#[derive(Default)]
struct CpuTopology {
    socket_ids: BTreeSet<String>,
    core_ids: BTreeSet<(String, String)>,
}

fn cpu_topology(cpuinfo: &str) -> CpuTopology {
    let mut topology = CpuTopology::default();
    for record in cpuinfo
        .split("\n\n")
        .filter(|record| !record.trim().is_empty())
    {
        let fields = first_cpuinfo_record(record);
        let socket = fields
            .get("physical id")
            .cloned()
            .unwrap_or_else(|| "0".to_string());
        let core = fields
            .get("core id")
            .cloned()
            .or_else(|| fields.get("processor").cloned())
            .unwrap_or_else(|| topology.core_ids.len().to_string());
        topology.socket_ids.insert(socket.clone());
        topology.core_ids.insert((socket, core));
    }
    topology
}

fn first_cpuinfo_record(cpuinfo: &str) -> BTreeMap<String, String> {
    cpuinfo
        .lines()
        .take_while(|line| !line.trim().is_empty())
        .filter_map(|line| line.split_once(':'))
        .map(|(key, value)| (key.trim().to_string(), value.trim().to_string()))
        .collect()
}

fn cpuinfo_field<'a>(fields: &'a BTreeMap<String, String>, candidates: &[&str]) -> Option<&'a str> {
    candidates
        .iter()
        .find_map(|candidate| fields.get(*candidate).map(String::as_str))
}

fn parse_mem_total_bytes(meminfo: &str) -> Result<u64, String> {
    let line = meminfo
        .lines()
        .find(|line| line.starts_with("MemTotal:"))
        .ok_or_else(|| "/proc/meminfo contains no MemTotal".to_string())?;
    let kibibytes = line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| "MemTotal has no value".to_string())?
        .parse::<u64>()
        .map_err(|error| format!("MemTotal is invalid: {error}"))?;
    kibibytes
        .checked_mul(1024)
        .ok_or_else(|| "MemTotal exceeds u64".to_string())
}

pub(crate) fn discover_cpu_cache_domains(root: &Path) -> Result<Vec<HardwareMemoryDomain>, String> {
    if !root.is_dir() {
        return Err(format!(
            "CPU cache topology is unavailable at {}",
            root.display()
        ));
    }
    let cpu_paths = fs::read_dir(root)
        .map_err(|error| format!("could not read CPU topology: {error}"))?
        .filter_map(Result::ok)
        .filter(|entry| numbered_name(&entry.file_name().to_string_lossy(), "cpu"))
        .map(|entry| entry.path().join("cache"))
        .filter(|path| path.is_dir())
        .collect::<Vec<_>>();
    let mut unique_domains = BTreeMap::new();
    for cache_root in cpu_paths {
        for entry in fs::read_dir(&cache_root)
            .map_err(|error| {
                format!(
                    "could not read CPU cache topology {}: {error}",
                    cache_root.display()
                )
            })?
            .filter_map(Result::ok)
            .filter(|entry| entry.file_name().to_string_lossy().starts_with("index"))
        {
            let domain = cache_domain(entry.path())?;
            unique_domains.entry(domain.name.clone()).or_insert(domain);
        }
    }
    if unique_domains.is_empty() {
        return Err(format!(
            "CPU cache topology contains no cache domains under {}",
            root.display()
        ));
    }
    Ok(unique_domains.into_values().collect())
}

fn cache_domain(path: PathBuf) -> Result<HardwareMemoryDomain, String> {
    let level = read_trimmed(path.join("level"))?;
    let kind = read_trimmed(path.join("type"))?.to_lowercase();
    let size_bytes = parse_cache_size(&read_trimmed(path.join("size"))?)?;
    let line_bytes = read_trimmed(path.join("coherency_line_size"))?
        .parse::<u64>()
        .map_err(|error| format!("invalid cache line size: {error}"))?;
    let shared = read_trimmed(path.join("shared_cpu_list"))?;
    let shared_name = shared
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    Ok(HardwareMemoryDomain {
        name: format!("cpu_l{level}_{kind}_cache_cpus_{shared_name}"),
        kind: format!("{kind}_cache"),
        capacity_bytes: size_bytes,
        host_visible: true,
        device_local: true,
        coherent: true,
        cached: true,
        minimum_alignment_bytes: line_bytes.max(1).next_power_of_two(),
        properties: BTreeMap::from([
            ("level".to_string(), level),
            ("shared_cpu_list".to_string(), shared),
        ]),
    })
}

fn parse_cache_size(raw: &str) -> Result<u64, String> {
    let (digits, multiplier) = if let Some(value) = raw.strip_suffix('K') {
        (value, 1024)
    } else if let Some(value) = raw.strip_suffix('M') {
        (value, 1024 * 1024)
    } else {
        (raw, 1)
    };
    digits
        .parse::<u64>()
        .map_err(|error| format!("invalid cache size {raw:?}: {error}"))?
        .checked_mul(multiplier)
        .ok_or_else(|| format!("cache size {raw:?} exceeds u64"))
}

fn stable_machine_identity(
    architecture: &str,
    vendor_id: &str,
    family: &str,
    model: &str,
    stepping: &str,
) -> Result<String, String> {
    let platform_identity = read_optional_trimmed("/sys/class/dmi/id/product_uuid")
        .or_else(|| read_optional_trimmed("/etc/machine-id"))
        .unwrap_or_else(|| "unavailable".to_string());
    stable_hardware_id(
        "cpu",
        &[json!([
            platform_identity,
            architecture,
            vendor_id,
            family,
            model,
            stepping,
        ])],
    )
}

fn discover_numbered_entries(root: &Path, prefix: &str) -> usize {
    fs::read_dir(root)
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .filter(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            numbered_name(&name, prefix)
        })
        .count()
}

fn numbered_name(name: &str, prefix: &str) -> bool {
    name.strip_prefix(prefix).is_some_and(|suffix| {
        !suffix.is_empty() && suffix.chars().all(|character| character.is_ascii_digit())
    })
}

fn kernel_release() -> String {
    read_optional_trimmed("/proc/sys/kernel/osrelease").unwrap_or_else(|| "unknown".to_string())
}

fn read_trimmed(path: PathBuf) -> Result<String, String> {
    fs::read_to_string(&path)
        .map(|value| value.trim().to_string())
        .map_err(|error| format!("could not read {}: {error}", path.display()))
}

fn read_optional_trimmed(path: impl AsRef<Path>) -> Option<String> {
    fs::read_to_string(path)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}
