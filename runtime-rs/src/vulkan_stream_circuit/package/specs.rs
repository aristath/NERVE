#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VulkanResidentModelPackageManifest {
    pub schema: String,
    pub package_id: String,
    pub compiler_fingerprint: String,
    pub circuit_graph: VulkanResidentPackageCircuitGraph,
    pub tensor_index_path: String,
    pub behavioral_validation_path: String,
    pub representation_optimization_path: String,
    pub config_path: String,
    pub tokenizer: VulkanResidentTokenizerPackageSpec,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activation_element_bytes: Option<usize>,
    pub max_context_activations: usize,
    pub required_vulkan_device_extensions: Vec<String>,
    pub required_vulkan_features: Vec<VulkanShaderFeature>,
    pub required_vulkan_subgroup_operations: Vec<VulkanSubgroupOperation>,
    pub input_transducer: VulkanResidentInputEmbeddingTransducerPackageSpec,
    pub output_transducer: VulkanResidentOutputTransducerPackageSpec,
    pub sampler: VulkanResidentSamplerPackageSpec,
    pub component_executions: Vec<VulkanResidentComponentExecutionSpec>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub speculative_decoders: Vec<VulkanResidentSpeculativeDecoderPackageSpec>,
    pub artifact_integrity: VulkanResidentPackageArtifactIntegrity,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VulkanResidentRuntimeModel {
    pub package: VulkanResidentModelPackageManifest,
    pub runtime_graph: StreamCircuitRuntimeGraph,
    pub placement: StreamCircuitPlacementSpec,
    pub circuit_graph: VulkanResidentPackageCircuitGraph,
    pub component_executions: Vec<VulkanResidentComponentExecutionSpec>,
    pub tensor_index_fragments: Vec<VulkanRuntimeTensorIndexFragment>,
    pub implementation_selection:
        Option<crate::RuntimeImplementationSelectionReport>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanRuntimeTensorIndexFragment {
    pub index_path: PathBuf,
    pub candidate_root: PathBuf,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanResidentPackageArtifactIntegrity {
    pub schema: String,
    pub algorithm: String,
    pub files: BTreeMap<String, VulkanResidentPackageArtifactDigest>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanResidentPackageArtifactDigest {
    pub byte_count: usize,
    pub sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VulkanResidentPackageComponentCircuit {
    pub component_id: String,
    pub operator_type: String,
    pub runtime_role: crate::stream_circuit::CircuitRuntimeRole,
    pub implementation: String,
    pub behavioral_role: String,
    pub circuit: StreamCircuit,
    pub params: CircuitParamsArtifact,
    pub state: CircuitStateArtifact,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanResidentTokenizerPackageSpec {
    pub path: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub files: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanResidentInputEmbeddingTransducerPackageSpec {
    pub spec: VulkanResidentInputEmbeddingTransducerSpec,
    pub shader_path: String,
    pub batch_shader_path: String,
    pub batch_control: VulkanResidentComponentBatchControlSpec,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanResidentOutputTransducerPackageSpec {
    pub spec: VulkanResidentOutputTransducerSpec,
    pub embedding_norm_shader_path: String,
    pub embedding_norm_batch_shader_path: String,
    pub embedding_norm_batch_lane_tile_width: u32,
    pub projection_shader_path: String,
    pub projection_batch_shader_path: String,
    pub projection_batch_lane_tile_width: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VulkanResidentSamplerPackageSpec {
    pub spec: VulkanResidentSamplerSpec,
    pub kernels: Vec<VulkanResidentSamplerKernelPackageSpec>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanResidentSamplerKernelPackageSpec {
    pub role: String,
    pub shader_path: String,
    pub local_size_x: u32,
    pub workgroup_count_x: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanResidentComponentExecutionSpec {
    pub component_id: String,
    pub operator_type: String,
    pub implementation: String,
    pub kernels: Vec<VulkanResidentComponentKernelSpec>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanResidentComponentKernelSpec {
    pub execution_index: usize,
    pub node_id: String,
    pub op: String,
    pub source_node_ids: Vec<String>,
    pub semantic_module_ids: Vec<String>,
    pub execution_domain: VulkanResidentComponentKernelExecutionDomain,
    pub shader_path: String,
    pub local_size_x: u32,
    pub workgroup_count_x: u32,
    pub batch_mode: VulkanResidentComponentKernelBatchMode,
    pub batch_implementations: Vec<VulkanResidentComponentBatchImplementationSpec>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanResidentComponentBatchImplementationSpec {
    pub execution_domain: VulkanResidentComponentKernelExecutionDomain,
    pub lane_tile_width: u32,
    pub independent_candidate_compatible: bool,
    pub causal_sequence_compatible: bool,
    pub device_requirements: VulkanResidentVulkanDeviceRequirements,
    pub stages: Vec<VulkanResidentComponentBatchStageSpec>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanResidentComponentBatchStageSpec {
    pub shader_path: String,
    pub local_size_x: u32,
    pub workgroup_count_x: u32,
    #[serde(default)]
    pub descriptor_bindings: Vec<VulkanResidentComponentBatchDescriptorBindingSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_snapshot_binding: Option<u32>,
    pub control: VulkanResidentComponentBatchControlSpec,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub indirect_dispatch_byte_offset: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanResidentComponentBatchDescriptorBindingSpec {
    pub binding: u32,
    pub source_binding: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum VulkanResidentComponentBatchControlSpec {
    StorageBuffer {
        byte_count: u32,
        binding: u32,
        payload: VulkanResidentComponentBatchControlPayload,
        #[serde(default)]
        access: VulkanResidentComponentBatchControlAccess,
    },
}

impl VulkanResidentComponentBatchControlSpec {
    pub(crate) fn storage_buffer(
        self,
    ) -> (u32, u32, VulkanResidentComponentBatchControlPayload) {
        match self {
            Self::StorageBuffer {
                byte_count,
                binding,
                payload,
                ..
            } => (binding, byte_count, payload),
        }
    }

    pub(crate) fn access(self) -> VulkanResidentComponentBatchControlAccess {
        match self {
            Self::StorageBuffer { access, .. } => access,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VulkanResidentComponentBatchControlAccess {
    #[default]
    Read,
    ReadWrite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VulkanResidentComponentBatchControlPayload {
    Width,
    WidthStateSnapshots,
    WidthExpertStart,
    WidthExpertRangeIndirect,
    Temporal,
}

impl VulkanResidentComponentBatchControlPayload {
    pub(crate) fn byte_count(self) -> u32 {
        match self {
            Self::Width => VULKAN_COMPONENT_BATCH_WIDTH_CONTROL_BYTE_CAPACITY,
            Self::WidthStateSnapshots => {
                2 * VULKAN_COMPONENT_BATCH_WIDTH_CONTROL_BYTE_CAPACITY
            }
            Self::WidthExpertStart => 2 * VULKAN_COMPONENT_BATCH_WIDTH_CONTROL_BYTE_CAPACITY,
            Self::WidthExpertRangeIndirect => {
                7 * VULKAN_COMPONENT_BATCH_WIDTH_CONTROL_BYTE_CAPACITY
            }
            Self::Temporal => VULKAN_COMPONENT_BATCH_CONTROL_BYTE_CAPACITY,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanResidentVulkanDeviceRequirements {
    pub vulkan_device_extensions: Vec<String>,
    pub vulkan_features: Vec<VulkanShaderFeature>,
    pub subgroup_operations: Vec<VulkanSubgroupOperation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cooperative_bfloat16_shape: Option<[u32; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cooperative_float8_e4m3_shape: Option<[u32; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subgroup_size: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VulkanResidentComponentKernelBatchMode {
    SerialLanes,
    WeightShared,
    CausalScan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VulkanResidentComponentKernelExecutionDomain {
    Decode,
    Prefill,
    DecodeAndPrefill,
}

impl VulkanResidentComponentKernelExecutionDomain {
    pub(super) fn supports_decode(self) -> bool {
        matches!(
            self,
            VulkanResidentComponentKernelExecutionDomain::Decode
                | VulkanResidentComponentKernelExecutionDomain::DecodeAndPrefill
        )
    }

    pub(super) fn supports_prefill(self) -> bool {
        matches!(
            self,
            VulkanResidentComponentKernelExecutionDomain::Prefill
                | VulkanResidentComponentKernelExecutionDomain::DecodeAndPrefill
        )
    }

    pub(super) fn supports_batch_mode(self, mode: VulkanComponentBatchExecutionMode) -> bool {
        match mode {
            VulkanComponentBatchExecutionMode::IndependentStreams => self.supports_decode(),
            VulkanComponentBatchExecutionMode::CausalSequence => self.supports_prefill(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VulkanResidentSpeculativeDecoderPackageSpec {
    pub id: String,
    #[serde(rename = "type")]
    pub decoder_type: String,
    pub source_prefix: String,
    pub circuit_graph: VulkanResidentPackageCircuitGraph,
    pub input_adapter: VulkanResidentDraftInputAdapterPackageSpec,
    pub output_transducer: VulkanResidentDraftOutputTransducerPackageSpec,
    pub component_executions: Vec<VulkanResidentComponentExecutionSpec>,
    pub state_contract: Value,
    pub verification_contract: Value,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanResidentDraftInputAdapterPackageSpec {
    pub component_id: String,
    pub token_embedding_signal_id: String,
    pub target_hidden_signal_id: String,
    pub output_signal_id: String,
    pub input_frame_byte_capacity: usize,
    pub target_hidden_byte_capacity: usize,
    pub output_frame_byte_capacity: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanResidentDraftOutputTransducerPackageSpec {
    pub component_id: String,
    pub input_signal_id: String,
    pub hidden_signal_id: String,
    pub logits_signal_id: String,
    pub norm_parameter_tensor: String,
    pub norm_parameter_dtype: String,
    pub norm_parameter_shape: Vec<usize>,
    pub norm_parameter_byte_capacity: usize,
    pub projection_parameter_tensor: String,
    pub projection_parameter_dtype: String,
    pub projection_parameter_shape: Vec<usize>,
    pub projection_parameter_byte_capacity: usize,
    pub projection_scale_parameter_tensor: Option<String>,
    pub projection_scale_parameter_dtype: Option<String>,
    pub projection_scale_parameter_shape: Option<Vec<usize>>,
    pub projection_scale_parameter_byte_capacity: Option<usize>,
    pub input_frame_byte_capacity: usize,
    pub output_hidden_byte_capacity: usize,
    pub logits_byte_capacity: usize,
    pub vocabulary_size: usize,
    pub hidden_size: usize,
    pub projection_workgroup_count_x: u32,
    pub norm_local_size_x: u32,
    pub projection_local_size_x: u32,
    pub norm_shader_path: String,
    pub projection_shader_path: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanResidentComponentKernelShaderRef {
    pub component_id: String,
    pub node_id: String,
    pub shader_path: String,
    pub local_size_x: u32,
    pub workgroup_count_x: u32,
}
