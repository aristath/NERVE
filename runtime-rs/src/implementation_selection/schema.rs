use crate::HardwareProcessProfile;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

pub const IMPLEMENTATION_REGISTRY_SCHEMA: &str = "nerve.optimizer.implementation_registry.v1";
pub const OPTIMIZER_STAGE_SCHEMA: &str = "nerve.optimizer.stage.v3";
pub const OPTIMIZATION_SCOPE_CATALOG_SCHEMA: &str = "nerve.optimizer.optimization_scope_catalog.v1";
pub const RUNTIME_IMPLEMENTATION_PREDICATE_SCHEMA: &str =
    "nerve.optimizer.runtime_implementation_predicate.v6";
pub const PROMOTION_DECISION_SCHEMA: &str = "nerve.optimizer.promotion_decision.v2";
pub const BENCHMARK_RECORD_SCHEMA: &str = "nerve.optimizer.benchmark_record.v3";
pub const VALIDATION_RECORD_SCHEMA: &str = "nerve.optimizer.validation_record.v2";
pub const RUNTIME_MOUNT_PLAN_SCHEMA: &str = "nerve.optimizer.runtime_mount_plan.v3";
pub const VULKAN_COMPONENT_OVERLAY_SCHEMA: &str = "nerve.optimizer.vulkan_component_overlay.v2";
pub const VULKAN_COMPONENT_REGION_OVERLAY_SCHEMA: &str =
    "nerve.optimizer.vulkan_component_region_overlay.v2";
pub const VULKAN_OUTPUT_TRANSDUCER_OVERLAY_SCHEMA: &str =
    "nerve.optimizer.vulkan_output_transducer_overlay.v1";
pub const VULKAN_STREAM_CIRCUIT_OVERLAY_ADAPTER: &str = "vulkan_stream_circuit_overlay.v2";
pub const STAGED_CANDIDATE_INTEGRITY_SCHEMA: &str = "nerve.optimizer.staged_candidate_integrity.v1";
pub const STAGED_CANDIDATE_INTEGRITY_FILE: &str = "integrity.json";
pub const STAGED_ARTIFACT_DIGEST_SCHEMA: &str = "nerve.optimizer.artifact_sha256.v1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeImplementationPredicate {
    pub schema: String,
    pub predicate_id: String,
    pub hardware: RuntimeHardwarePredicate,
    pub execution: RuntimeExecutionPredicate,
    pub placement: RuntimePlacementPredicate,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeHardwarePredicate {
    pub measured_profile_ids: Vec<String>,
    pub capability_classes: Vec<String>,
    pub device_kinds: Vec<String>,
    pub apis: Vec<String>,
    pub required_processes: Vec<String>,
    pub required_features: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeExecutionPredicate {
    pub phases: Vec<String>,
    pub alternative_phases: Vec<String>,
    pub source_retained_phases: Vec<String>,
    pub activation_batch: RuntimeInclusiveRange,
    pub context_activations: RuntimeInclusiveRange,
    pub state_activations: RuntimeInclusiveRange,
    pub speculative_draft_token_counts: Vec<usize>,
    pub residency_policies: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeInclusiveRange {
    pub minimum: usize,
    pub maximum: usize,
}

impl RuntimeInclusiveRange {
    pub fn contains(&self, value: usize) -> bool {
        self.minimum <= value && value <= self.maximum
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimePlacementPredicate {
    pub mode: String,
    pub minimum_device_count: usize,
    pub maximum_device_count: usize,
    pub required_interconnects: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeImplementationRegistry {
    pub schema: String,
    pub registry_id: String,
    pub package_id: String,
    pub exact_baseline: RuntimeExactImplementation,
    pub implementations: Vec<RuntimeImplementation>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeExactImplementation {
    pub artifact_ref: String,
    pub contract_digest: String,
    pub mutable: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeImplementation {
    pub implementation_id: String,
    pub candidate_id: String,
    pub scope_ids: Vec<String>,
    pub source_contract_digests: Vec<String>,
    pub representation: Value,
    pub behavioral_contract: Value,
    pub runtime_predicate: RuntimeImplementationPredicate,
    pub artifact_bundle: RuntimeImplementationArtifactBundle,
    pub evidence: RuntimeImplementationEvidence,
    pub provenance: Value,
    pub comparison: RuntimeImplementationComparison,
    pub decision_reason: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeImplementationArtifactBundle {
    pub root_ref: String,
    pub candidate_integrity_ref: String,
    pub mount_plan_ref: String,
    pub candidate_integrity_digest: String,
    pub artifact_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeImplementationEvidence {
    pub promotion_decision_ref: String,
    pub candidate_contract_ref: String,
    pub construction_record_ref: String,
    pub prebenchmark_record_ref: String,
    pub benchmark_record_ref: String,
    pub validation_record_ref: String,
    pub analysis_run_refs: Vec<RuntimeEvidenceReference>,
    pub hardware_profile_refs: Vec<RuntimeHardwareProfileReference>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeEvidenceReference {
    pub run_id: String,
    pub artifact_ref: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeHardwareProfileReference {
    pub profile_id: String,
    pub artifact_ref: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeImplementationComparison {
    pub exact_implementation_id: String,
    pub exact_contract_digest: String,
    pub benchmark_id: String,
    pub benchmark_decision: String,
    pub workloads: Vec<RuntimeComparedWorkload>,
    pub validation_id: String,
    pub validation_status: String,
    pub behavioral_contract: Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeComparedWorkload {
    pub workload_id: String,
    pub decision: String,
    pub paired: RuntimePairedComparison,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimePairedComparison {
    pub speedup_ppm: i64,
    pub candidate_is_faster: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeOptimizationScope {
    pub scope_id: String,
    pub source_contract_digest: String,
    pub component_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeImplementationWorkloadMetrics {
    pub workload_id: String,
    pub phase: String,
    pub activation_batch_width: usize,
    pub context_activations: usize,
    pub state_activations: usize,
    pub reference_latency_ns: u64,
    pub candidate_latency_ns: u64,
    pub conversion_ns: u64,
    pub conversion_bytes: u64,
    pub boundary_count: u64,
    pub resource_load_count: u64,
    pub resource_reload_count: u64,
    pub resource_physical_read_bytes: u64,
    pub resource_resident_bytes_produced: u64,
    pub resource_uploaded_bytes: u64,
    pub resource_read_ns: u64,
    pub resource_derivation_ns: u64,
    pub resource_upload_ns: u64,
    pub resource_blocking_ns: u64,
    pub speedup_ppm: i64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeMountPlan {
    pub schema: String,
    pub candidate_id: String,
    pub adapter_id: String,
    pub regions: Vec<RuntimeMountRegion>,
    pub tensor_index_refs: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeMountRegion {
    pub replacements: Vec<RuntimeReplacement>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeStagedCandidate {
    pub candidate_id: String,
    pub candidate_root: PathBuf,
    pub source_component_ids: Vec<String>,
    pub mount_plan: RuntimeMountPlan,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RuntimeReplacement {
    Component {
        source_component_id: String,
        overlay_ref: String,
    },
    ComponentRegion {
        source_component_id: String,
        overlay_ref: String,
    },
    OutputTransducer {
        source_component_id: String,
        overlay_ref: String,
    },
}

impl RuntimeReplacement {
    pub fn source_component_id(&self) -> &str {
        match self {
            Self::Component {
                source_component_id,
                ..
            }
            | Self::ComponentRegion {
                source_component_id,
                ..
            }
            | Self::OutputTransducer {
                source_component_id,
                ..
            } => source_component_id,
        }
    }

    pub fn overlay_ref(&self) -> &str {
        match self {
            Self::Component { overlay_ref, .. }
            | Self::ComponentRegion { overlay_ref, .. }
            | Self::OutputTransducer { overlay_ref, .. } => overlay_ref,
        }
    }

    pub fn is_component_region(&self) -> bool {
        matches!(self, Self::ComponentRegion { .. })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LoadedRuntimeImplementation {
    pub implementation: RuntimeImplementation,
    pub source_component_ids: Vec<String>,
    pub workload_metrics: Vec<RuntimeImplementationWorkloadMetrics>,
    pub candidate_root: PathBuf,
    pub mount_plan: RuntimeMountPlan,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RuntimeImplementationCatalog {
    pub package_id: String,
    pub package_root: PathBuf,
    pub stage_status: String,
    pub exact_baseline: RuntimeExactImplementation,
    pub scopes: BTreeMap<String, RuntimeOptimizationScope>,
    pub implementations: Vec<LoadedRuntimeImplementation>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimeImplementationCatalogReport {
    pub package_id: String,
    pub stage_status: String,
    pub exact_baseline: RuntimeExactImplementation,
    pub implementations: Vec<RuntimeImplementation>,
}

impl RuntimeImplementationCatalog {
    pub fn report(&self) -> RuntimeImplementationCatalogReport {
        RuntimeImplementationCatalogReport {
            package_id: self.package_id.clone(),
            stage_status: self.stage_status.clone(),
            exact_baseline: self.exact_baseline.clone(),
            implementations: self
                .implementations
                .iter()
                .map(|loaded| loaded.implementation.clone())
                .collect(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeExecutionEnvelope {
    pub phases: Vec<String>,
    pub activation_batch: RuntimeInclusiveRange,
    pub context_activations: RuntimeInclusiveRange,
    pub state_activations: RuntimeInclusiveRange,
    pub speculative_draft_tokens: usize,
    pub residency_policy: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeSelectionDevice {
    pub logical_device_id: String,
    pub physical_device_id: String,
    pub profile: HardwareProcessProfile,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeSelectionInstance {
    pub instance_id: String,
    pub source_component_id: String,
    pub logical_device_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeSelectionEdge {
    pub source_instance_id: String,
    pub destination_instance_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeSelectionRequest {
    pub execution: RuntimeExecutionEnvelope,
    pub devices: Vec<RuntimeSelectionDevice>,
    pub instances: Vec<RuntimeSelectionInstance>,
    pub edges: Vec<RuntimeSelectionEdge>,
    /// Runtime instances whose exact compiled implementation cannot execute on
    /// their current physical placement. A selected alternative must cover
    /// every member; unrelated compatible instances may retain the exact
    /// baseline.
    pub exact_baseline_incompatible_instance_ids: BTreeSet<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeSelectedImplementation {
    pub implementation_id: String,
    pub candidate_id: String,
    pub instance_ids: Vec<String>,
    pub scope_ids: Vec<String>,
    /// Immutable semantic contracts of the exact source regions this
    /// implementation was validated to replace. Physical planners use these
    /// identities to compare different compiled representations without
    /// confusing artifact signatures with model behavior.
    pub source_contract_digests: Vec<String>,
    pub mount_adapter_id: String,
    pub predicate: RuntimeImplementationPredicate,
    pub representation: Value,
    pub provenance: Value,
    pub benchmark_id: String,
    pub validation_id: String,
    pub validation_status: String,
    pub speedup_ppm: i64,
    pub estimated_saved_ns: u64,
    pub conversion_ns: u64,
    pub conversion_bytes: u64,
    pub boundary_count: u64,
    pub resource_load_count: u64,
    pub resource_reload_count: u64,
    pub resource_physical_read_bytes: u64,
    pub resource_resident_bytes_produced: u64,
    pub resource_uploaded_bytes: u64,
    pub resource_read_ns: u64,
    pub resource_derivation_ns: u64,
    pub resource_upload_ns: u64,
    pub resource_blocking_ns: u64,
    pub decision_reason: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeRejectedImplementation {
    pub implementation_id: String,
    pub instance_ids: Vec<String>,
    pub reasons: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeImplementationSelectionReport {
    pub package_id: String,
    pub execution: RuntimeExecutionEnvelope,
    pub selected: Vec<RuntimeSelectedImplementation>,
    pub exact_instance_ids: Vec<String>,
    pub rejected: Vec<RuntimeRejectedImplementation>,
    pub total_estimated_saved_ns: u64,
    pub total_conversion_ns: u64,
    pub total_conversion_bytes: u64,
    pub total_boundary_count: u64,
    pub total_resource_load_count: u64,
    pub total_resource_reload_count: u64,
    pub total_resource_physical_read_bytes: u64,
    pub total_resource_resident_bytes_produced: u64,
    pub total_resource_uploaded_bytes: u64,
    pub total_resource_read_ns: u64,
    pub total_resource_derivation_ns: u64,
    pub total_resource_upload_ns: u64,
    pub total_resource_blocking_ns: u64,
}
