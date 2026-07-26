use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub const HARDWARE_PROCESS_INVENTORY_SCHEMA: &str = "nerve.hardware_process_inventory.v1";
pub const HARDWARE_PROCESS_PROFILE_SCHEMA: &str = "nerve.optimizer.hardware_process_profile.v1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareProcessInventory {
    pub schema: String,
    pub profiles: Vec<HardwareProcessProfile>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareProcessProfile {
    pub schema: String,
    pub profile_id: String,
    pub hardware_identity: HardwareIdentity,
    pub capability_class: String,
    pub processes: Vec<HardwareProcessCapability>,
    pub memory_domains: Vec<HardwareMemoryDomain>,
    pub interconnects: Vec<HardwareInterconnect>,
    pub measurements: Vec<HardwareMeasurement>,
    pub provenance: HardwareProfileProvenance,
    pub capability_extensions: BTreeMap<String, Value>,
    pub identity_extensions: BTreeMap<String, Value>,
    pub runtime_bindings: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HardwareProcessProfileDefinition {
    pub hardware_identity: HardwareIdentity,
    pub processes: Vec<HardwareProcessCapability>,
    pub memory_domains: Vec<HardwareMemoryDomain>,
    pub interconnects: Vec<HardwareInterconnect>,
    pub provenance: HardwareProfileProvenance,
    pub capability_extensions: BTreeMap<String, Value>,
    pub identity_extensions: BTreeMap<String, Value>,
    pub runtime_bindings: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareIdentity {
    pub device_kind: HardwareDeviceKind,
    pub stable_device_id: String,
    pub name: String,
    pub vendor_id: String,
    pub device_id: String,
    pub architecture: String,
    pub physical_location: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HardwareDeviceKind {
    Cpu,
    Gpu,
}

impl HardwareDeviceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareProcessCapability {
    pub name: String,
    pub category: HardwareProcessCategory,
    pub availability: HardwareProcessAvailability,
    pub programmability: HardwareProcessProgrammability,
    pub api: String,
    pub operations: Vec<String>,
    pub numeric_formats: Vec<String>,
    pub required_extensions: Vec<String>,
    pub required_features: Vec<String>,
    pub limits: BTreeMap<String, u64>,
    pub properties: BTreeMap<String, String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HardwareProcessCategory {
    Arithmetic,
    ControlFlow,
    Memory,
    Transfer,
    Synchronization,
    Scheduling,
    Sampling,
    Graphics,
    RayTraversal,
    Media,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HardwareProcessAvailability {
    Available,
    Unavailable,
    Opaque,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HardwareProcessProgrammability {
    Direct,
    Indirect,
    None,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareMemoryDomain {
    pub name: String,
    pub kind: String,
    pub capacity_bytes: u64,
    pub host_visible: bool,
    pub device_local: bool,
    pub coherent: bool,
    pub cached: bool,
    pub minimum_alignment_bytes: u64,
    pub properties: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareInterconnect {
    pub name: String,
    pub kind: String,
    pub availability: HardwareProcessAvailability,
    pub api: String,
    pub operations: Vec<String>,
    pub properties: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareMeasurement {
    pub name: String,
    pub unit: String,
    pub regime: BTreeMap<String, String>,
    pub samples: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareProfileProvenance {
    pub api: String,
    pub api_version: String,
    pub driver: String,
    pub driver_version: String,
    pub compiler: String,
    pub operating_system: String,
    pub discovery_backend: String,
}

impl HardwareProcessInventory {
    pub fn new(mut profiles: Vec<HardwareProcessProfile>) -> Result<Self, String> {
        profiles.sort_by(|left, right| {
            left.hardware_identity
                .stable_device_id
                .cmp(&right.hardware_identity.stable_device_id)
        });
        let inventory = Self {
            schema: HARDWARE_PROCESS_INVENTORY_SCHEMA.to_string(),
            profiles,
        };
        inventory.validate()?;
        Ok(inventory)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != HARDWARE_PROCESS_INVENTORY_SCHEMA {
            return Err(format!(
                "unsupported hardware-process inventory schema {:?}",
                self.schema
            ));
        }
        if self.profiles.is_empty() {
            return Err("hardware-process inventory contains no profiles".to_string());
        }
        let identities = self
            .profiles
            .iter()
            .map(|profile| profile.hardware_identity.stable_device_id.as_str())
            .collect::<Vec<_>>();
        if !strictly_sorted_unique(&identities) {
            return Err(
                "hardware-process inventory profiles must have unique sorted identities"
                    .to_string(),
            );
        }
        for profile in &self.profiles {
            profile.validate()?;
        }
        Ok(())
    }
}

impl HardwareProcessProfile {
    pub fn create(mut definition: HardwareProcessProfileDefinition) -> Result<Self, String> {
        let processes = &mut definition.processes;
        processes.sort_by(|left, right| left.name.cmp(&right.name));
        for process in processes.iter_mut() {
            normalize_strings(&mut process.operations);
            normalize_strings(&mut process.numeric_formats);
            normalize_strings(&mut process.required_extensions);
            normalize_strings(&mut process.required_features);
        }
        definition
            .memory_domains
            .sort_by(|left, right| left.name.cmp(&right.name));
        definition
            .interconnects
            .sort_by(|left, right| left.name.cmp(&right.name));
        for interconnect in &mut definition.interconnects {
            normalize_strings(&mut interconnect.operations);
        }
        let mut profile = Self {
            schema: HARDWARE_PROCESS_PROFILE_SCHEMA.to_string(),
            profile_id: String::new(),
            hardware_identity: definition.hardware_identity,
            capability_class: String::new(),
            processes: definition.processes,
            memory_domains: definition.memory_domains,
            interconnects: definition.interconnects,
            measurements: Vec::new(),
            provenance: definition.provenance,
            capability_extensions: definition.capability_extensions,
            identity_extensions: definition.identity_extensions,
            runtime_bindings: definition.runtime_bindings,
        };
        profile.refresh_derived_identities()?;
        profile.validate()?;
        Ok(profile)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != HARDWARE_PROCESS_PROFILE_SCHEMA {
            return Err(format!(
                "unsupported hardware-process profile schema {:?}",
                self.schema
            ));
        }
        self.hardware_identity.validate()?;
        self.provenance.validate()?;
        if self.processes.is_empty() {
            return Err(format!(
                "hardware profile {:?} contains no processes",
                self.profile_id
            ));
        }
        let process_names = self
            .processes
            .iter()
            .map(|process| process.name.as_str())
            .collect::<Vec<_>>();
        if !strictly_sorted_unique(&process_names) {
            return Err("hardware processes must have unique sorted names".to_string());
        }
        for process in &self.processes {
            process.validate()?;
        }
        if self.memory_domains.is_empty() {
            return Err("hardware profile contains no memory domains".to_string());
        }
        let memory_names = self
            .memory_domains
            .iter()
            .map(|domain| domain.name.as_str())
            .collect::<Vec<_>>();
        if !strictly_sorted_unique(&memory_names) {
            return Err("hardware memory domains must have unique sorted names".to_string());
        }
        for domain in &self.memory_domains {
            domain.validate()?;
        }
        let interconnect_names = self
            .interconnects
            .iter()
            .map(|interconnect| interconnect.name.as_str())
            .collect::<Vec<_>>();
        if !strictly_sorted_unique(&interconnect_names) {
            return Err("hardware interconnects must have unique sorted names".to_string());
        }
        for interconnect in &self.interconnects {
            interconnect.validate()?;
        }
        for measurement in &self.measurements {
            measurement.validate()?;
        }
        let measurement_names = self
            .measurements
            .iter()
            .map(|measurement| measurement.name.as_str())
            .collect::<Vec<_>>();
        if !strictly_sorted_unique(&measurement_names) {
            return Err("hardware measurements must have unique sorted names".to_string());
        }
        let (capability_class, profile_id) = self.derived_hardware_identities()?;
        if self.capability_class != capability_class || self.profile_id != profile_id {
            return Err(
                "hardware profile identity does not match its canonical capabilities".to_string(),
            );
        }
        Ok(())
    }

    pub fn with_measurements(
        mut self,
        mut measurements: Vec<HardwareMeasurement>,
    ) -> Result<Self, String> {
        measurements.sort_by(|left, right| left.name.cmp(&right.name));
        self.measurements = measurements;
        self.refresh_derived_identities()?;
        self.validate()?;
        Ok(self)
    }

    fn refresh_derived_identities(&mut self) -> Result<(), String> {
        let (capability_class, profile_id) = self.derived_hardware_identities()?;
        self.capability_class = capability_class;
        self.profile_id = profile_id;
        Ok(())
    }

    fn derived_hardware_identities(&self) -> Result<(String, String), String> {
        let capability_body = serde_json::json!({
            "device_kind": self.hardware_identity.device_kind,
            "architecture": self.hardware_identity.architecture,
            "processes": self.processes,
            "memory_domains": self.memory_domains,
            "interconnects": self.interconnects,
            "api": self.provenance.api,
            "api_version": self.provenance.api_version,
            "capability_extensions": self.capability_extensions,
        });
        let capability_class = stable_hardware_id("hardware_capability", &[capability_body])?;
        let profile_identity = serde_json::json!([
            self.hardware_identity,
            capability_class,
            self.provenance,
            self.identity_extensions,
            self.measurements,
        ]);
        let profile_id = stable_hardware_id("hardware_profile", &[profile_identity])?;
        Ok((capability_class, profile_id))
    }
}

impl HardwareIdentity {
    fn validate(&self) -> Result<(), String> {
        for (field, value) in [
            ("stable_device_id", self.stable_device_id.as_str()),
            ("name", self.name.as_str()),
            ("vendor_id", self.vendor_id.as_str()),
            ("device_id", self.device_id.as_str()),
            ("architecture", self.architecture.as_str()),
            ("physical_location", self.physical_location.as_str()),
        ] {
            if value.is_empty() {
                return Err(format!("hardware identity {field} must not be empty"));
            }
        }
        Ok(())
    }
}

impl HardwareProcessCapability {
    pub fn new(
        name: impl Into<String>,
        category: HardwareProcessCategory,
        availability: HardwareProcessAvailability,
        programmability: HardwareProcessProgrammability,
        api: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            category,
            availability,
            programmability,
            api: api.into(),
            operations: Vec::new(),
            numeric_formats: Vec::new(),
            required_extensions: Vec::new(),
            required_features: Vec::new(),
            limits: BTreeMap::new(),
            properties: BTreeMap::new(),
        }
    }

    fn validate(&self) -> Result<(), String> {
        if self.name.is_empty() || self.api.is_empty() {
            return Err("hardware process name and API must not be empty".to_string());
        }
        if self.availability == HardwareProcessAvailability::Available
            && self.programmability == HardwareProcessProgrammability::None
        {
            return Err(format!(
                "available hardware process {:?} must be programmable",
                self.name
            ));
        }
        if self.availability == HardwareProcessAvailability::Unavailable
            && self.programmability != HardwareProcessProgrammability::None
        {
            return Err(format!(
                "unavailable hardware process {:?} cannot be programmable",
                self.name
            ));
        }
        for (field, values) in [
            ("operations", &self.operations),
            ("numeric_formats", &self.numeric_formats),
            ("required_extensions", &self.required_extensions),
            ("required_features", &self.required_features),
        ] {
            if !sorted_unique(values) {
                return Err(format!(
                    "hardware process {:?} {field} must be unique and sorted",
                    self.name
                ));
            }
        }
        Ok(())
    }
}

impl HardwareMemoryDomain {
    fn validate(&self) -> Result<(), String> {
        if self.name.is_empty()
            || self.kind.is_empty()
            || self.capacity_bytes == 0
            || self.minimum_alignment_bytes == 0
            || !self.minimum_alignment_bytes.is_power_of_two()
        {
            return Err(format!("hardware memory domain {:?} is invalid", self.name));
        }
        Ok(())
    }
}

impl HardwareInterconnect {
    fn validate(&self) -> Result<(), String> {
        if self.name.is_empty() || self.kind.is_empty() || self.api.is_empty() {
            return Err("hardware interconnect identity is invalid".to_string());
        }
        if !sorted_unique(&self.operations) {
            return Err(format!(
                "hardware interconnect {:?} has duplicate or unsorted values",
                self.name
            ));
        }
        Ok(())
    }
}

impl HardwareMeasurement {
    fn validate(&self) -> Result<(), String> {
        if self.name.is_empty() || self.unit.is_empty() || self.samples.is_empty() {
            return Err("hardware measurement is incomplete".to_string());
        }
        Ok(())
    }
}

impl HardwareProfileProvenance {
    fn validate(&self) -> Result<(), String> {
        for (field, value) in [
            ("api", self.api.as_str()),
            ("api_version", self.api_version.as_str()),
            ("driver", self.driver.as_str()),
            ("driver_version", self.driver_version.as_str()),
            ("compiler", self.compiler.as_str()),
            ("operating_system", self.operating_system.as_str()),
            ("discovery_backend", self.discovery_backend.as_str()),
        ] {
            if value.is_empty() {
                return Err(format!(
                    "hardware profile provenance {field} must not be empty"
                ));
            }
        }
        Ok(())
    }
}

pub fn stable_hardware_id(prefix: &str, identity_parts: &[Value]) -> Result<String, String> {
    if prefix.is_empty() {
        return Err("stable hardware id prefix must not be empty".to_string());
    }
    let bytes = serde_json::to_vec(identity_parts)
        .map_err(|error| format!("could not serialize hardware identity: {error}"))?;
    let digest = Sha256::digest(bytes);
    Ok(format!("{prefix}_{digest:x}")[..prefix.len() + 33].to_string())
}

fn normalize_strings(values: &mut Vec<String>) {
    values.sort();
    values.dedup();
}

fn sorted_unique(values: &[String]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

fn strictly_sorted_unique(values: &[&str]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}
