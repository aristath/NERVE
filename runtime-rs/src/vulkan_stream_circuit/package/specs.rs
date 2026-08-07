#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VulkanResidentModelPackageManifest {
    pub schema: String,
    pub package_id: String,
    pub resource_residency: CompiledResourceResidencyContract,
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

impl VulkanResidentModelPackageManifest {
    /// Returns the package-owned speculative width, when one is declared.
    /// Multiple attached decoders share one stream-level verification width,
    /// so independently declared recommendations must agree.
    pub fn recommended_speculative_draft_tokens(&self) -> Result<Option<usize>, String> {
        let recommendations = self
            .speculative_decoders
            .iter()
            .filter_map(VulkanResidentSpeculativeDecoderPackageSpec::recommended_draft_tokens)
            .collect::<BTreeSet<_>>();
        match recommendations.len() {
            0 => Ok(None),
            1 => Ok(recommendations.first().copied()),
            _ => Err(format!(
                "compiled speculative decoders disagree on their package-owned default widths: {:?}",
                recommendations,
            )),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct VulkanResidentRuntimeModel {
    pub execution_scope: String,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_codec: Option<String>,
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
    pub stream_control_binding: Option<u32>,
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
    pub selection_priority: u32,
    pub independent_candidate_compatible: bool,
    pub causal_sequence_compatible: bool,
    #[serde(default)]
    pub parallel_block_compatible: bool,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_snapshot_source_binding: Option<u32>,
    pub control: VulkanResidentComponentBatchControlSpec,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub indirect_dispatch_byte_offset: Option<u32>,
    #[serde(default)]
    pub dispatch_y_from_batch_width: bool,
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
    TemporalStateSnapshots,
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
            Self::TemporalStateSnapshots => {
                VULKAN_COMPONENT_BATCH_CONTROL_BYTE_CAPACITY
                    + VULKAN_COMPONENT_BATCH_WIDTH_CONTROL_BYTE_CAPACITY
            }
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
            VulkanComponentBatchExecutionMode::ParallelBlock => self.supports_decode(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VulkanResidentSpeculativeDecoderPackageSpec {
    pub id: String,
    #[serde(rename = "type")]
    pub decoder_type: String,
    pub source_prefix: String,
    pub execution_contract: VulkanResidentSpeculativeExecutionContract,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proposal_contract: Option<Value>,
    pub circuit_graph: VulkanResidentPackageCircuitGraph,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_adapter: Option<VulkanResidentDraftInputAdapterPackageSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_transducer: Option<VulkanResidentDraftOutputTransducerPackageSpec>,
    pub component_executions: Vec<VulkanResidentComponentExecutionSpec>,
    pub state_contract: Value,
    pub verification_contract: Value,
}

impl VulkanResidentSpeculativeDecoderPackageSpec {
    pub fn minimum_draft_tokens(&self) -> Option<usize> {
        self.proposal_contract
            .as_ref()
            .and_then(Value::as_object)
            .and_then(|proposal| proposal.get("minimum_draft_tokens"))
            .and_then(Value::as_u64)
            .and_then(|tokens| usize::try_from(tokens).ok())
            .filter(|tokens| *tokens > 0)
    }

    pub fn recommended_draft_tokens(&self) -> Option<usize> {
        self.proposal_contract
            .as_ref()
            .and_then(Value::as_object)
            .and_then(|proposal| proposal.get("default_draft_tokens"))
            .and_then(Value::as_u64)
            .and_then(|tokens| usize::try_from(tokens).ok())
            .filter(|tokens| *tokens > 0)
    }

    pub fn dedicated_input_adapter(&self) -> Option<&VulkanResidentDraftInputAdapterPackageSpec> {
        self.input_adapter.as_ref()
    }

    pub fn dedicated_output_transducer(
        &self,
    ) -> Option<&VulkanResidentDraftOutputTransducerPackageSpec> {
        self.output_transducer.as_ref()
    }

    pub fn validate_execution_io(&self) -> Result<(), String> {
        let has_dedicated_io = self.input_adapter.is_some() && self.output_transducer.is_some();
        let has_partial_io = self.input_adapter.is_some() != self.output_transducer.is_some();
        if has_partial_io
            || self.execution_contract.uses_dedicated_autoregressive_io() != has_dedicated_io
        {
            return Err(format!(
                "speculative decoder {:?} dedicated I/O does not match its execution contract",
                self.id
            ));
        }
        if let VulkanResidentSpeculativeExecutionContract::ParallelBlock { block_width, .. } =
            &self.execution_contract
        {
            let proposal = self
                .proposal_contract
                .as_ref()
                .and_then(Value::as_object)
                .ok_or_else(|| {
                    format!(
                        "parallel speculative decoder {:?} has no proposal contract",
                        self.id
                    )
                })?;
            let block_width = u64::try_from(*block_width).ok();
            let configured = proposal
                .get("configured_block_size")
                .and_then(Value::as_u64);
            let minimum = proposal
                .get("minimum_draft_tokens")
                .and_then(Value::as_u64);
            let recommended = proposal
                .get("default_draft_tokens")
                .and_then(Value::as_u64);
            if proposal
                .get("execution_block_size")
                .and_then(Value::as_u64)
                != block_width
                || configured
                    .zip(block_width)
                    .is_none_or(|(configured, capacity)| {
                        configured == 0 || configured > capacity
                    })
                || minimum
                    .zip(recommended)
                    .zip(block_width)
                    .is_none_or(|((minimum, recommended), capacity)| {
                        minimum == 0 || minimum > recommended || recommended > capacity
                    })
                || proposal.get("confidence_prefix").and_then(Value::as_str)
                    != Some("first_sigmoid_below_runtime_threshold")
            {
                return Err(format!(
                    "parallel speculative decoder {:?} proposal contract disagrees with execution",
                    self.id
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum VulkanResidentSpeculativeExecutionContract {
    AutoregressiveFeedback {
        processor_schedule: String,
        output_schedule: String,
    },
    ParallelBlock {
        block_width: usize,
        source_context_tick_offset: i64,
        processor_schedule: String,
        output_schedule: String,
    },
}

impl VulkanResidentSpeculativeExecutionContract {
    pub fn validate(&self, decoder_id: &str) -> Result<(), String> {
        let valid = match self {
            Self::AutoregressiveFeedback {
                processor_schedule,
                output_schedule,
            } => {
                processor_schedule == "one_token_per_tick"
                    && output_schedule == "dedicated_token_transducer"
            }
            Self::ParallelBlock {
                block_width,
                source_context_tick_offset: _,
                processor_schedule,
                output_schedule,
            } => {
                *block_width > 0
                    && processor_schedule == "parallel_lanes"
                    && output_schedule == "compiled_component_graph"
            }
        };
        valid.then_some(()).ok_or_else(|| {
            format!("speculative decoder {decoder_id:?} has an invalid execution contract")
        })
    }

    pub fn block_width(&self) -> Option<usize> {
        match self {
            Self::AutoregressiveFeedback { .. } => None,
            Self::ParallelBlock { block_width, .. } => Some(*block_width),
        }
    }

    pub fn uses_dedicated_autoregressive_io(&self) -> bool {
        matches!(self, Self::AutoregressiveFeedback { .. })
    }
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
