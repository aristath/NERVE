use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;

use nerve_runtime::{
    CircuitPort, ComponentEdgePlacement, ComponentPlacement, HardwareProcessInventory,
    HardwareProcessProfile, RUNTIME_DEFAULT_LOGICAL_DEVICE_ID, RUNTIME_TOPOLOGY_SCHEMA,
    ResourceResidencyPolicy, RuntimeAssistantStreamProtocolAction, RuntimeAvailableDevice,
    RuntimeBoundDevice, RuntimeChatGeneratedOutputControl, RuntimeChatSession,
    RuntimeCompiledExecutionGraphSummary, RuntimeComponentPortSummary, RuntimeCriticalPathPhase,
    RuntimeCriticalPathReport, RuntimeDeviceBindings, RuntimeDeviceSliceReport,
    RuntimeDeviceTickPlanReport, RuntimeEdgeRouteTarget, RuntimeEdgeRoutes,
    RuntimeEffectiveExecutionGraphTopology, RuntimeExecutionEnvelope,
    RuntimeFeedbackExecutionReport, RuntimeGraphControls, RuntimeGraphDuplicateAfterControl,
    RuntimeGraphInspectionReport, RuntimeGraphPlacementReport, RuntimeGraphSourceChainEntry,
    RuntimeImplementationSelectionReport, RuntimeInclusiveRange, RuntimeLocalEdgeBufferReport,
    RuntimeModelEditor, RuntimePackageInspectionReport, RuntimePlacedComponentTimingSummaryReport,
    RuntimePlacedPromptRunReport, RuntimePlacedTransportEdgeReport, RuntimePlacedTransportReport,
    RuntimePlacementReport, RuntimePreparedChatTurn, RuntimePromptTimingReport,
    RuntimeRecoverableChatTurnError, RuntimeRemoteEdgeBufferReport, RuntimeSelectionCoverageReport,
    RuntimeSourceComponent, RuntimeSparseMoeWorkReport, RuntimeTokenizerOptionsReport,
    RuntimeTopologyReport, VULKAN_RUNTIME_PLACEMENT_CALIBRATION_MAXIMUM_DURATION,
    VulkanCompiledResourceResidencyReport, VulkanComputeDevice, VulkanComputeDeviceCatalog,
    VulkanComputeDeviceInfo, VulkanPlacedEdgeTransferRoute, VulkanPlacedEdgeTransportStats,
    VulkanPlacedPromptEngineShutdownReport, VulkanPlacementCalibrationCatalog,
    VulkanPlacementCapacityEnvelope, VulkanPlacementDeviceExecutionIdentity,
    VulkanResidentBufferPool, VulkanResidentChatTransactionPhase, VulkanResidentExecutionCounters,
    VulkanResidentFeedbackExecutionStats, VulkanResidentHfTokenizerTextCodec,
    VulkanResidentInProcessPlacedModelPackage, VulkanResidentInProcessPlacedPromptEngine,
    VulkanResidentInProcessPlacedPromptStream, VulkanResidentModelPackageDeviceSlice,
    VulkanResidentModelPackageManifest, VulkanResidentPlacedPrefixStateCacheStats,
    VulkanResidentRuntimeModel, VulkanResidentSamplerRuntimeConfig, VulkanResidentTokenInputEvent,
    VulkanResidentTokenTextCodec, VulkanRetainedCompiledResourceStores,
    VulkanReusableKernelArtifactManifest, VulkanRuntimePhysicalExecutionPlan,
    VulkanRuntimePlacementCalibrationSuite, VulkanRuntimePlacementCandidate,
    VulkanRuntimePlacementCostModel, VulkanSpeculativeCycleTrace, VulkanSpeculativeWindowStats,
    VulkanTargetedComponentExecutionPhase, calibrate_vulkan_runtime_placement_candidate,
    calibrate_vulkan_runtime_placement_transfers, capacity_pack_and_select_vulkan_runtime_model,
    chat_stop_token_ids_from_manifest, chat_transcript_codec, discover_cpu_hardware_profile,
    execute_vulkan_resident_chat_transaction, load_vulkan_package_placement_calibration_catalog,
    rebalance_demand_paged_vulkan_runtime_model_from_working_set,
    record_vulkan_runtime_transfer_calibration_report, reset_runtime_critical_path_counters,
    resolve_vulkan_runtime_hybrid_physical_execution,
    reset_vulkan_resident_execution_counters, runtime_critical_path_report,
    runtime_critical_path_span, runtime_devices_from_compute_devices,
    vulkan_resident_execution_counters,
    vulkan_runtime_device_capacity_admission_bytes,
    vulkan_runtime_hybrid_phase_is_calibrated,
    vulkan_runtime_placement_transfer_byte_counts, vulkan_safe_host_available_bytes,
};

#[derive(Clone, Debug, PartialEq)]
struct Args {
    package_manifest: Option<PathBuf>,
    prompt: Option<String>,
    chat: bool,
    inspect_runtime: bool,
    inspect_package: bool,
    inspect_graph: bool,
    inspect_placement: bool,
    inspect_device_slice: Option<String>,
    inspect_devices: bool,
    initialize_device_contexts: bool,
    default_device_id: Option<String>,
    node_devices: BTreeMap<String, String>,
    component_shard_devices: BTreeMap<String, Vec<String>>,
    device_bindings: BTreeMap<String, String>,
    allowed_physical_device_ids: BTreeSet<String>,
    duplicate_after: Vec<(String, String)>,
    source_chain: Option<Vec<(String, String)>>,
    chat_template_variables: BTreeMap<String, serde_json::Value>,
    max_new_tokens: usize,
    speculative_draft_tokens: Option<usize>,
    speculative_confidence_threshold: f32,
    resource_residency_policy: ResourceResidencyPolicy,
    context_size: Option<usize>,
    vulkan_device_index: Option<usize>,
    random_seed: u32,
    temperature: Option<f32>,
    top_k: Option<u32>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    presence_penalty: Option<f32>,
    repetition_penalty: Option<f32>,
    add_special_tokens: bool,
    skip_special_tokens: bool,
    generated_only: bool,
    json: bool,
}

struct PromptRunContext<'a> {
    args: &'a Args,
    package_manifest: &'a Path,
    manifest_dir: &'a Path,
    tokenizer_dir: &'a Path,
    prompt: &'a str,
    prompt_ids: &'a [u32],
    scheduled_token_activations: usize,
    capacity: usize,
    codec: &'a VulkanResidentHfTokenizerTextCodec,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            package_manifest: None,
            prompt: None,
            chat: false,
            inspect_runtime: false,
            inspect_package: false,
            inspect_graph: false,
            inspect_placement: false,
            inspect_device_slice: None,
            inspect_devices: false,
            initialize_device_contexts: false,
            default_device_id: None,
            node_devices: BTreeMap::new(),
            component_shard_devices: BTreeMap::new(),
            device_bindings: BTreeMap::new(),
            allowed_physical_device_ids: BTreeSet::new(),
            duplicate_after: Vec::new(),
            source_chain: None,
            chat_template_variables: BTreeMap::new(),
            max_new_tokens: 65_536,
            speculative_draft_tokens: None,
            speculative_confidence_threshold: 0.0,
            resource_residency_policy: ResourceResidencyPolicy::Eager,
            context_size: None,
            vulkan_device_index: None,
            random_seed: 0,
            temperature: None,
            top_k: None,
            top_p: None,
            min_p: None,
            presence_penalty: None,
            repetition_penalty: None,
            add_special_tokens: true,
            skip_special_tokens: true,
            generated_only: false,
            json: false,
        }
    }
}

fn effective_speculative_draft_tokens(
    args: &Args,
    runtime_model: &VulkanResidentRuntimeModel,
) -> Result<usize, io::Error> {
    resolve_speculative_draft_tokens(args.speculative_draft_tokens, || {
        runtime_model.package.recommended_speculative_draft_tokens()
    })
}

fn resolve_speculative_draft_tokens(
    explicit: Option<usize>,
    package_recommendation: impl FnOnce() -> Result<Option<usize>, String>,
) -> Result<usize, io::Error> {
    if let Some(explicit) = explicit {
        return Ok(explicit);
    }
    package_recommendation()
        .map(|recommended| recommended.unwrap_or(0))
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

fn sampler_runtime_config(args: &Args) -> VulkanResidentSamplerRuntimeConfig {
    VulkanResidentSamplerRuntimeConfig {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        min_p: args.min_p,
        presence_penalty: args.presence_penalty,
        repetition_penalty: args.repetition_penalty,
    }
}
