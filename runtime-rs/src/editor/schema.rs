#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimeEditorSemanticModule {
    pub id: String,
    pub role: String,
    pub responsibility: String,
    pub parent_id: Option<String>,
    pub child_ids: Vec<String>,
    pub source_node_ids: Vec<String>,
    pub parameter_ref_ids: Vec<String>,
    pub owned_state_port_ids: Vec<String>,
    pub input_signals: Vec<String>,
    pub output_signals: Vec<String>,
    pub optimized_node_ids: Vec<String>,
    pub kernel_node_ids: Vec<String>,
    pub measured_cost: Option<Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimeEditorImplementationOption {
    pub implementation_id: String,
    pub candidate_id: String,
    pub scope_ids: Vec<String>,
    pub runtime_predicate: crate::RuntimeImplementationPredicate,
    pub representation: Value,
    pub provenance: Value,
    pub benchmark_id: String,
    pub validation_id: String,
    pub validation_status: String,
    pub decision_reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimeEditorSourceComponent {
    pub source_id: String,
    pub layer_index: Option<usize>,
    pub operator_type: String,
    pub runtime_role: CircuitRuntimeRole,
    pub implementation: String,
    pub behavioral_role: String,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub state_ports: Vec<Value>,
    pub controls: Vec<Value>,
    pub control_schemas: Vec<RuntimeEditorControlSchema>,
    pub parameter_ref_count: usize,
    pub node_count: usize,
    pub kernel_count: usize,
    pub semantic_modules: Vec<RuntimeEditorSemanticModule>,
    pub semantic_module_root_id: Option<String>,
    pub implementation_options: Vec<RuntimeEditorImplementationOption>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimeEditorRepresentationGraph {
    pub schema: String,
    pub graph_id: String,
    pub candidate_id: String,
    pub scope_ids: Vec<String>,
    pub source_contract_digests: BTreeMap<String, String>,
    pub logical_contracts: Vec<crate::RepresentationLogicalContract>,
    pub physical_representations: Vec<crate::PhysicalRepresentation>,
    pub signals: Vec<crate::RepresentationSignal>,
    pub resources: Vec<crate::RepresentationResource>,
    pub nodes: Vec<crate::RepresentationNode>,
    pub connections: Vec<crate::RepresentationConnection>,
    pub public_ports: Vec<crate::RepresentationPublicPort>,
    pub islands: Vec<crate::RepresentationIsland>,
    pub absorbed_transforms: Vec<crate::RepresentationAbsorbedTransform>,
    pub physical_kernels: Vec<crate::RepresentationPhysicalKernel>,
    pub confidence: crate::RepresentationConfidence,
    pub unresolved: Vec<crate::RepresentationUnresolved>,
    pub correction_requests: Vec<crate::RepresentationCorrectionRequest>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimeEditorControlChoice {
    pub value: Value,
    pub label: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum RuntimeEditorControlKind {
    Boolean,
    Integer,
    Number,
    Text,
    Enumeration {
        choices: Vec<RuntimeEditorControlChoice>,
    },
    ReadOnly,
    Unsupported {
        declared_type: String,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimeEditorControlSchema {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub kind: RuntimeEditorControlKind,
    pub current_value: Option<Value>,
    pub default_value: Option<Value>,
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
    pub step: Option<f64>,
    pub units: Option<String>,
    pub editable_at_runtime: bool,
    pub requires_state_reset: bool,
    pub requires_remount: bool,
    pub requires_recompile: bool,
    pub scope: String,
    pub raw: Value,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeEditorInstance {
    pub instance_id: String,
    pub source_id: String,
    pub layer_index: Option<usize>,
    pub occurrence: usize,
    pub device_id: String,
    pub enabled: bool,
    pub control_values: BTreeMap<String, Value>,
    pub state_policy: StreamCircuitNodeInstanceStatePolicy,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimeEditorValidation {
    pub valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub placement: Option<StreamCircuitPlacementPlan>,
}

#[derive(Clone, Debug)]
pub struct RuntimeModelEditor {
    package_manifest_path: PathBuf,
    package_root: PathBuf,
    manifest: VulkanResidentModelPackageManifest,
    implementation_catalog: crate::RuntimeImplementationCatalog,
    source_graph: ResolvedLoweredExecutionGraph,
    source_components: Vec<RuntimeEditorSourceComponent>,
    source_by_layer: BTreeMap<usize, Vec<String>>,
    source_ids: BTreeSet<String>,
    available_devices: Vec<RuntimeAvailableDevice>,
    draft: StreamCircuitRuntimeGraph,
}
