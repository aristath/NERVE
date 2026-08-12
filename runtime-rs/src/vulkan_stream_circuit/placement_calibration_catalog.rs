pub const VULKAN_PLACEMENT_CALIBRATION_CATALOG_SCHEMA: &str =
    "nerve.vulkan_placement_calibration_catalog.v7";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VulkanPlacementExecutionStrategy {
    SingleDevice,
    Serialized,
    TensorParallel,
    WholeExpertParallel,
    IntraExpertTensorParallel,
    Hybrid,
    DirectedBoundary,
    Reduction,
    LazyLoadWave,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct VulkanPlacementDeviceExecutionIdentity {
    pub physical_device_id: String,
    pub api_version: u32,
    pub driver_version: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct VulkanPlacementDispatchGeometry {
    pub contract_id: String,
    pub logical_extent: usize,
    pub sampled_extent: usize,
    pub input_width: usize,
    pub workgroup_count_x: u32,
    pub local_size_x: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum VulkanPlacementOperationGeometry {
    Dispatch {
        geometry: VulkanPlacementDispatchGeometry,
    },
    DirectedTransfer {
        contract_id: String,
        byte_count: usize,
    },
    Reduction {
        contract_id: String,
        element_count: usize,
        element_byte_count: usize,
        participant_count: usize,
    },
    LazyLoadWave {
        contract_id: String,
        resource_count: usize,
        byte_count: usize,
    },
}

impl VulkanPlacementOperationGeometry {
    fn contract_id(&self) -> &str {
        match self {
            Self::Dispatch { geometry } => &geometry.contract_id,
            Self::DirectedTransfer { contract_id, .. }
            | Self::Reduction { contract_id, .. }
            | Self::LazyLoadWave { contract_id, .. } => contract_id,
        }
    }

    fn is_valid(&self) -> bool {
        match self {
            Self::Dispatch { geometry } => {
                !geometry.contract_id.is_empty()
                    && geometry.logical_extent > 0
                    && geometry.sampled_extent > 0
                    && geometry.sampled_extent <= geometry.logical_extent
                    && geometry.input_width > 0
                    && geometry.workgroup_count_x > 0
                    && geometry.local_size_x > 0
            }
            Self::DirectedTransfer {
                contract_id,
                byte_count,
            } => !contract_id.is_empty() && *byte_count > 0,
            Self::Reduction {
                contract_id,
                element_count,
                element_byte_count,
                participant_count,
            } => {
                !contract_id.is_empty()
                    && *element_count > 0
                    && *element_byte_count > 0
                    && *participant_count >= 2
            }
            Self::LazyLoadWave {
                contract_id,
                resource_count,
                byte_count,
            } => !contract_id.is_empty() && *resource_count > 0 && *byte_count > 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct VulkanPlacementShapeClass {
    pub activation_batch_width: usize,
    pub input_byte_capacity: usize,
    pub output_byte_capacity: usize,
    pub operations: Vec<VulkanPlacementOperationGeometry>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct VulkanPlacementShardIdentity {
    pub dispatch_ordinal: usize,
    /// Stable position of this shard's physical participant in the calibrated
    /// execution pool. Logical ranges cannot recover this for every split
    /// strategy (input-column shards can share the same output-row range).
    pub participant_ordinal: usize,
    pub physical_device_id: String,
    pub distribution: String,
    pub logical_start: usize,
    pub logical_count: usize,
    /// Exact ownership by partition ordinal within this dispatch. Runtime
    /// selector IDs contain component-instance identity and therefore cannot
    /// be used to replay one equivalent compiled transaction at another
    /// component instance.
    pub selected_resource_indices_by_partition: BTreeMap<usize, Vec<usize>>,
    pub selected_resource_fragments_by_partition:
        BTreeMap<usize, Vec<VulkanPlacementSelectedResourceFragmentIdentity>>,
    pub parameter_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct VulkanPlacementSelectedResourceFragmentIdentity {
    pub resource_index: usize,
    pub atomic_group_id: String,
    pub logical_start: usize,
    pub logical_count: usize,
    pub parameters: Vec<VulkanPlacementSelectedResourceParameterFragmentIdentity>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct VulkanPlacementSelectedResourceParameterFragmentIdentity {
    pub parameter_slot: usize,
    pub resource_id: String,
    pub resource_byte_count: usize,
    pub byte_offset: usize,
    pub byte_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct VulkanPlacementTransportIdentity {
    pub source_physical_device_id: String,
    pub destination_physical_device_id: String,
    pub byte_capacity: usize,
    pub route: String,
}

/// Exact semantic and executable identity shared by canonically equivalent
/// placement candidates. Physical placement is intentionally absent so one
/// validated reference can authorize comparison across legal placements.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct VulkanPlacementBehaviorIdentity {
    /// Stable producer-emitted identity of the executable transaction. This
    /// is the model-independent join key that binds measurements back to the
    /// same compiled contract. Producers for repeated graph components must
    /// exclude semantic component names so equivalent instances can share
    /// evidence.
    pub compiled_execution_signature: String,
    pub contract_ids: Vec<String>,
    pub implementation_digests: Vec<String>,
    pub artifact_digest: String,
    pub execution_graph_digest: String,
    pub runtime_implementation_fingerprint: String,
    pub phase: nerve_execution_contracts::ExecutionPhase,
    pub shape: VulkanPlacementShapeClass,
    pub input_fixture_digest: String,
    pub equivalence: VulkanPlacementEquivalenceIdentity,
}

/// Exact measured execution case. Changing a driver, shard, endpoint, owner,
/// or transport route creates a different case rather than reusing stale cost.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct VulkanPlacementExecutionCaseIdentity {
    pub behavior: VulkanPlacementBehaviorIdentity,
    pub strategy: VulkanPlacementExecutionStrategy,
    pub devices: Vec<VulkanPlacementDeviceExecutionIdentity>,
    pub shards: Vec<VulkanPlacementShardIdentity>,
    pub input_physical_device_id: String,
    pub output_physical_device_id: String,
    pub owner_physical_device_id: String,
    pub transports: Vec<VulkanPlacementTransportIdentity>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanPlacementCanonicalReference {
    pub behavior: VulkanPlacementBehaviorIdentity,
    pub output_digest: String,
    pub output_artifact: Option<VulkanPlacementOutputArtifact>,
    pub state_digest: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanPlacementCalibrationObservation {
    pub execution_case: VulkanPlacementExecutionCaseIdentity,
    pub warmup_call_count: usize,
    pub measured_call_count: usize,
    pub complete_transaction: bool,
    pub duration_ns: u64,
    pub useful_activation_count: usize,
    pub output_digest: String,
    pub output_artifact: Option<VulkanPlacementOutputArtifact>,
    pub output_equivalence: VulkanPlacementOutputEquivalenceEvidence,
    pub state_digest: String,
    pub resident_bytes_by_physical_device: BTreeMap<String, usize>,
    pub transient_peak_bytes_by_physical_device: BTreeMap<String, usize>,
    pub host_resident_bytes: usize,
    pub host_transient_peak_bytes: usize,
}

/// Capacity available to a runtime placement decision after preserving every
/// pre-existing reservation. Calibration evidence is usable only when the
/// complete measured transaction fits this exact envelope on the same Vulkan
/// API and driver identity; absent or changed devices are unavailable, not
/// zero-cost spill targets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPlacementCapacityEnvelope {
    pub available_bytes_by_device: BTreeMap<VulkanPlacementDeviceExecutionIdentity, usize>,
    pub host_available_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPlacementCalibrationCatalogError(pub String);

impl Display for VulkanPlacementCalibrationCatalogError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for VulkanPlacementCalibrationCatalogError {}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanPlacementCalibrationCatalog {
    pub schema: String,
    references: Vec<VulkanPlacementCanonicalReference>,
    observations: Vec<VulkanPlacementCalibrationObservation>,
}

impl Default for VulkanPlacementCalibrationCatalog {
    fn default() -> Self {
        Self {
            schema: VULKAN_PLACEMENT_CALIBRATION_CATALOG_SCHEMA.to_string(),
            references: Vec::new(),
            observations: Vec::new(),
        }
    }
}

impl VulkanPlacementCalibrationCatalog {
    pub fn reference_count(&self) -> usize {
        self.references.len()
    }

    pub fn observation_count(&self) -> usize {
        self.observations.len()
    }

    /// Transactionally unions independently produced exact catalogs. An exact
    /// execution case measured more than once is retained conservatively at
    /// its slowest complete observed duration. All non-timing evidence must be
    /// identical; changing resources, outputs, state, or call shape is a
    /// conflict rather than a remeasurement.
    pub fn merge(&mut self, other: &Self) -> Result<(), VulkanPlacementCalibrationCatalogError> {
        self.validate()?;
        other.validate()?;
        let mut merged = self.clone();
        for reference in &other.references {
            merged.record_reference(reference.clone())?;
        }
        for observation in &other.observations {
            match merged.observations.binary_search_by(|existing| {
                existing.execution_case.cmp(&observation.execution_case)
            }) {
                Ok(index) => {
                    let existing = &merged.observations[index];
                    if !compatible_remeasurement(existing, observation) {
                        return Err(VulkanPlacementCalibrationCatalogError(
                            "placement catalogs contain conflicting evidence for one exact execution case"
                                .to_string(),
                        ));
                    }
                    if observation.duration_ns > existing.duration_ns {
                        merged.observations[index] = observation.clone();
                    }
                }
                Err(_) => merged.record_observation(observation.clone())?,
            }
        }
        merged.validate()?;
        *self = merged;
        Ok(())
    }

    pub fn from_json_slice(payload: &[u8]) -> Result<Self, VulkanPlacementCalibrationCatalogError> {
        let decoded: Self = serde_json::from_slice(payload).map_err(|error| {
            VulkanPlacementCalibrationCatalogError(format!(
                "failed to decode placement calibration catalog: {error}",
            ))
        })?;
        decoded.validate()?;
        Ok(decoded)
    }

    pub fn to_json_bytes(&self) -> Result<Vec<u8>, VulkanPlacementCalibrationCatalogError> {
        self.validate()?;
        serde_json::to_vec_pretty(self).map_err(|error| {
            VulkanPlacementCalibrationCatalogError(format!(
                "failed to encode placement calibration catalog: {error}",
            ))
        })
    }

    pub fn validate(&self) -> Result<(), VulkanPlacementCalibrationCatalogError> {
        if self.schema != VULKAN_PLACEMENT_CALIBRATION_CATALOG_SCHEMA {
            return Err(VulkanPlacementCalibrationCatalogError(format!(
                "unsupported placement calibration catalog schema {:?}",
                self.schema,
            )));
        }
        let mut rebuilt = Self::default();
        for reference in &self.references {
            rebuilt.record_reference(reference.clone())?;
        }
        for observation in &self.observations {
            rebuilt.record_observation(observation.clone())?;
        }
        if &rebuilt != self {
            return Err(VulkanPlacementCalibrationCatalogError(
                "placement calibration catalog records are not canonical".to_string(),
            ));
        }
        Ok(())
    }

    pub fn record_reference(
        &mut self,
        reference: VulkanPlacementCanonicalReference,
    ) -> Result<(), VulkanPlacementCalibrationCatalogError> {
        validate_behavior_identity(&reference.behavior)?;
        if reference.output_digest.is_empty() || reference.state_digest.is_empty() {
            return Err(VulkanPlacementCalibrationCatalogError(
                "placement reference requires output and state digests".to_string(),
            ));
        }
        validate_vulkan_placement_output_equivalence(
            &reference.behavior.equivalence,
            &reference.output_digest,
            reference.output_artifact.as_ref(),
            &reference.output_digest,
            reference.output_artifact.as_ref(),
        )?;
        match self
            .references
            .binary_search_by(|existing| existing.behavior.cmp(&reference.behavior))
        {
            Ok(index) if self.references[index] != reference => {
                return Err(VulkanPlacementCalibrationCatalogError(
                    "placement behavior has conflicting canonical output evidence".to_string(),
                ));
            }
            Ok(_) => return Ok(()),
            Err(index) => self.references.insert(index, reference),
        }
        Ok(())
    }

    pub fn record_observation(
        &mut self,
        observation: VulkanPlacementCalibrationObservation,
    ) -> Result<(), VulkanPlacementCalibrationCatalogError> {
        validate_observation(&observation)?;
        let reference = self
            .references
            .binary_search_by(|reference| {
                reference.behavior.cmp(&observation.execution_case.behavior)
            })
            .ok()
            .map(|index| &self.references[index])
            .ok_or_else(|| {
                VulkanPlacementCalibrationCatalogError(
                    "placement candidate has no exact canonical behavior reference".to_string(),
                )
            })?;
        if observation.state_digest != reference.state_digest {
            return Err(VulkanPlacementCalibrationCatalogError(
                "placement candidate state differs from its canonical reference".to_string(),
            ));
        }
        let output_equivalence = validate_vulkan_placement_output_equivalence(
            &reference.behavior.equivalence,
            &reference.output_digest,
            reference.output_artifact.as_ref(),
            &observation.output_digest,
            observation.output_artifact.as_ref(),
        )?;
        if observation.output_equivalence != output_equivalence {
            return Err(VulkanPlacementCalibrationCatalogError(
                "placement candidate output-equivalence evidence is not reproducible".to_string(),
            ));
        }
        match self
            .observations
            .binary_search_by(|existing| existing.execution_case.cmp(&observation.execution_case))
        {
            Ok(_) => {
                return Err(VulkanPlacementCalibrationCatalogError(
                    "placement calibration repeats an exact execution case".to_string(),
                ));
            }
            Err(index) => self.observations.insert(index, observation),
        }
        Ok(())
    }

    pub fn exact_observation(
        &self,
        execution_case: &VulkanPlacementExecutionCaseIdentity,
    ) -> Option<&VulkanPlacementCalibrationObservation> {
        self.observations
            .binary_search_by(|observation| observation.execution_case.cmp(execution_case))
            .ok()
            .map(|index| &self.observations[index])
    }

    pub fn canonical_reference(
        &self,
        behavior: &VulkanPlacementBehaviorIdentity,
    ) -> Option<&VulkanPlacementCanonicalReference> {
        self.references
            .binary_search_by(|reference| reference.behavior.cmp(behavior))
            .ok()
            .map(|index| &self.references[index])
    }

    pub fn observations_for_behavior(
        &self,
        behavior: &VulkanPlacementBehaviorIdentity,
    ) -> Vec<&VulkanPlacementCalibrationObservation> {
        self.observations
            .iter()
            .filter(|observation| &observation.execution_case.behavior == behavior)
            .collect()
    }

    /// Returns every exact, output-valid candidate for this behavior. Local
    /// duration and memory are insufficient to prune alternatives with
    /// different shards, queues, strategies, owners, or transport routes: the
    /// globally faster choice depends on neighboring islands and legal
    /// overlap. Dominance therefore belongs to the graph scheduler, which has
    /// that complete context.
    pub fn candidates_for_behavior(
        &self,
        behavior: &VulkanPlacementBehaviorIdentity,
    ) -> Vec<&VulkanPlacementCalibrationObservation> {
        self.observations_for_behavior(behavior)
    }

    pub fn candidate_behaviors_for_compiled_execution(
        &self,
        compiled_execution_signature: &str,
        phase: nerve_execution_contracts::ExecutionPhase,
    ) -> Vec<&VulkanPlacementBehaviorIdentity> {
        let mut behaviors = self
            .observations
            .iter()
            .filter(|observation| {
                matches!(
                    observation.execution_case.strategy,
                    VulkanPlacementExecutionStrategy::SingleDevice
                        | VulkanPlacementExecutionStrategy::Serialized
                        | VulkanPlacementExecutionStrategy::TensorParallel
                        | VulkanPlacementExecutionStrategy::WholeExpertParallel
                        | VulkanPlacementExecutionStrategy::IntraExpertTensorParallel
                        | VulkanPlacementExecutionStrategy::Hybrid
                ) && observation.execution_case.behavior.compiled_execution_signature
                    == compiled_execution_signature
                    && observation.execution_case.behavior.phase == phase
            })
            .map(|observation| &observation.execution_case.behavior)
            .collect::<Vec<_>>();
        behaviors.sort();
        behaviors.dedup();
        behaviors
    }

    pub fn directed_boundary_candidates(
        &self,
        phase: nerve_execution_contracts::ExecutionPhase,
        activation_batch_width: usize,
        byte_count: usize,
    ) -> Vec<&VulkanPlacementCalibrationObservation> {
        self.observations
            .iter()
            .filter(|observation| {
                observation.execution_case.strategy
                    == VulkanPlacementExecutionStrategy::DirectedBoundary
                    && observation.execution_case.behavior.phase == phase
                    && observation
                        .execution_case
                        .behavior
                        .shape
                        .activation_batch_width
                        == activation_batch_width
                    && matches!(
                        observation.execution_case.behavior.shape.operations.as_slice(),
                        [VulkanPlacementOperationGeometry::DirectedTransfer {
                            byte_count: measured,
                            ..
                        }] if *measured == byte_count
                    )
            })
            .collect()
    }

    /// Returns every exact candidate that fits the current reservation-aware
    /// capacity envelope. Resident bytes and transient peak bytes coexist
    /// during the measured transaction and are therefore added, with overflow
    /// making the candidate unavailable.
    pub fn candidates_for_capacity(
        &self,
        behavior: &VulkanPlacementBehaviorIdentity,
        capacity: &VulkanPlacementCapacityEnvelope,
    ) -> Vec<&VulkanPlacementCalibrationObservation> {
        self.candidates_for_behavior(behavior)
            .into_iter()
            .filter(|candidate| observation_fits_capacity(candidate, capacity))
            .collect()
    }
}

fn observation_fits_capacity(
    observation: &VulkanPlacementCalibrationObservation,
    capacity: &VulkanPlacementCapacityEnvelope,
) -> bool {
    let host_required = observation
        .host_resident_bytes
        .checked_add(observation.host_transient_peak_bytes);
    if !host_required.is_some_and(|required| required <= capacity.host_available_bytes) {
        return false;
    }
    observation.execution_case.devices.iter().all(|device| {
        let physical_id = &device.physical_device_id;
        let required = observation
            .resident_bytes_by_physical_device
            .get(physical_id)
            .copied()
            .and_then(|resident| {
                observation
                    .transient_peak_bytes_by_physical_device
                    .get(physical_id)
                    .copied()
                    .and_then(|transient| resident.checked_add(transient))
            });
        required.is_some_and(|required| {
            capacity
                .available_bytes_by_device
                .get(device)
                .is_some_and(|available| required <= *available)
        })
    })
}

fn compatible_remeasurement(
    left: &VulkanPlacementCalibrationObservation,
    right: &VulkanPlacementCalibrationObservation,
) -> bool {
    left.execution_case == right.execution_case
        && left.warmup_call_count == right.warmup_call_count
        && left.measured_call_count == right.measured_call_count
        && left.complete_transaction == right.complete_transaction
        && left.useful_activation_count == right.useful_activation_count
        && left.output_digest == right.output_digest
        && left.output_artifact == right.output_artifact
        && left.output_equivalence == right.output_equivalence
        && left.state_digest == right.state_digest
        && left.resident_bytes_by_physical_device == right.resident_bytes_by_physical_device
        && left.transient_peak_bytes_by_physical_device
            == right.transient_peak_bytes_by_physical_device
        && left.host_resident_bytes == right.host_resident_bytes
        && left.host_transient_peak_bytes == right.host_transient_peak_bytes
}

fn validate_behavior_identity(
    behavior: &VulkanPlacementBehaviorIdentity,
) -> Result<(), VulkanPlacementCalibrationCatalogError> {
    if behavior.contract_ids.is_empty()
        || behavior.compiled_execution_signature.is_empty()
        || behavior.contract_ids.iter().any(String::is_empty)
        || behavior.implementation_digests.len() != behavior.contract_ids.len()
        || behavior.implementation_digests.iter().any(String::is_empty)
        || behavior.artifact_digest.is_empty()
        || behavior.execution_graph_digest.is_empty()
        || behavior.runtime_implementation_fingerprint.is_empty()
        || behavior.shape.activation_batch_width == 0
        || behavior.shape.input_byte_capacity == 0
        || behavior.shape.output_byte_capacity == 0
        || behavior.shape.operations.is_empty()
        || !valid_sha256_digest(&behavior.input_fixture_digest)
    {
        return Err(VulkanPlacementCalibrationCatalogError(
            "placement behavior identity is incomplete".to_string(),
        ));
    }
    behavior.equivalence.validate()?;
    if !is_strictly_sorted(&behavior.contract_ids)
        || !valid_sha256_digest(&behavior.artifact_digest)
        || !valid_sha256_digest(&behavior.execution_graph_digest)
        || behavior
            .implementation_digests
            .iter()
            .any(|digest| !valid_sha256_digest(digest))
        || behavior.shape.operations.iter().any(|operation| {
            !behavior
                .contract_ids
                .iter()
                .any(|id| id == operation.contract_id())
        })
    {
        return Err(VulkanPlacementCalibrationCatalogError(
            "placement behavior identity is not canonical or digest-complete".to_string(),
        ));
    }
    if behavior
        .shape
        .operations
        .iter()
        .any(|operation| !operation.is_valid())
    {
        return Err(VulkanPlacementCalibrationCatalogError(
            "placement shape contains an invalid dispatch geometry".to_string(),
        ));
    }
    Ok(())
}

fn validate_observation(
    observation: &VulkanPlacementCalibrationObservation,
) -> Result<(), VulkanPlacementCalibrationCatalogError> {
    validate_behavior_identity(&observation.execution_case.behavior)?;
    let case = &observation.execution_case;
    if case.devices.is_empty()
        || case
            .devices
            .iter()
            .any(|device| device.physical_device_id.is_empty())
        || case
            .devices
            .iter()
            .map(|device| device.physical_device_id.as_str())
            .collect::<BTreeSet<_>>()
            .len()
            != case.devices.len()
        || !is_strictly_sorted(&case.devices)
        || !is_strictly_sorted(&case.shards)
        || !is_strictly_sorted(&case.transports)
        || case.input_physical_device_id.is_empty()
        || case.output_physical_device_id.is_empty()
        || case.owner_physical_device_id.is_empty()
        || !case
            .devices
            .iter()
            .any(|device| device.physical_device_id == case.owner_physical_device_id)
        || observation.warmup_call_count == 0
        || !(1..=2).contains(&observation.measured_call_count)
        || !observation.complete_transaction
        || observation.duration_ns == 0
        || observation.useful_activation_count == 0
        || observation.output_digest.is_empty()
        || observation.state_digest.is_empty()
    {
        return Err(VulkanPlacementCalibrationCatalogError(
            "placement observation is incomplete or not a bounded complete transaction".to_string(),
        ));
    }
    let participant_count_valid = match case.strategy {
        VulkanPlacementExecutionStrategy::SingleDevice => case.devices.len() == 1,
        VulkanPlacementExecutionStrategy::Serialized
        | VulkanPlacementExecutionStrategy::TensorParallel
        | VulkanPlacementExecutionStrategy::WholeExpertParallel
        | VulkanPlacementExecutionStrategy::IntraExpertTensorParallel
        | VulkanPlacementExecutionStrategy::Hybrid
        | VulkanPlacementExecutionStrategy::DirectedBoundary
        | VulkanPlacementExecutionStrategy::Reduction => case.devices.len() >= 2,
        VulkanPlacementExecutionStrategy::LazyLoadWave => true,
    };
    let devices = case
        .devices
        .iter()
        .map(|device| device.physical_device_id.as_str())
        .collect::<BTreeSet<_>>();
    let resident_devices = observation
        .resident_bytes_by_physical_device
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let transient_devices = observation
        .transient_peak_bytes_by_physical_device
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if !participant_count_valid
        || !devices.contains(case.input_physical_device_id.as_str())
        || !devices.contains(case.output_physical_device_id.as_str())
        || resident_devices != devices
        || transient_devices != devices
        || case.shards.iter().any(|shard| {
            !devices.contains(shard.physical_device_id.as_str())
                || shard.participant_ordinal >= case.devices.len()
                || shard.logical_count == 0
                || (shard.parameter_bytes == 0
                    && shard.selected_resource_indices_by_partition.is_empty()
                    && shard.selected_resource_fragments_by_partition.is_empty())
                || shard.distribution.is_empty()
                || !shard_selected_resource_partition_ordinals(shard)
                || shard
                    .selected_resource_indices_by_partition
                    .iter()
                    .any(|(_, indices)| {
                        indices.is_empty()
                            || indices.windows(2).any(|pair| pair[0] >= pair[1])
                    })
                || shard
                    .selected_resource_fragments_by_partition
                    .values()
                    .any(|fragments| !valid_selected_resource_fragments(fragments))
        })
        || case.transports.iter().any(|route| {
            !devices.contains(route.source_physical_device_id.as_str())
                || !devices.contains(route.destination_physical_device_id.as_str())
                || route.source_physical_device_id == route.destination_physical_device_id
                || route.byte_capacity == 0
                || route.route.is_empty()
        })
    {
        return Err(VulkanPlacementCalibrationCatalogError(
            "placement observation references an invalid physical route or shard".to_string(),
        ));
    }
    Ok(())
}

fn shard_selected_resource_partition_ordinals(shard: &VulkanPlacementShardIdentity) -> bool {
    let mut ordinals = shard
        .selected_resource_indices_by_partition
        .keys()
        .chain(shard.selected_resource_fragments_by_partition.keys())
        .copied()
        .collect::<Vec<_>>();
    ordinals.sort_unstable();
    ordinals.dedup();
    ordinals.into_iter().eq(0..shard
        .selected_resource_indices_by_partition
        .len()
        .checked_add(shard.selected_resource_fragments_by_partition.len())
        .unwrap_or(usize::MAX))
        && shard
            .selected_resource_indices_by_partition
            .keys()
            .all(|ordinal| !shard.selected_resource_fragments_by_partition.contains_key(ordinal))
}

fn valid_selected_resource_fragments(
    fragments: &[VulkanPlacementSelectedResourceFragmentIdentity],
) -> bool {
    !fragments.is_empty()
        && fragments
            .windows(2)
            .all(|pair| pair[0].resource_index < pair[1].resource_index)
        && fragments.iter().all(|fragment| {
            !fragment.atomic_group_id.is_empty()
                && fragment.logical_count > 0
                && !fragment.parameters.is_empty()
                && fragment
                    .parameters
                    .windows(2)
                    .all(|pair| pair[0].parameter_slot < pair[1].parameter_slot)
                && fragment.parameters.iter().all(|parameter| {
                    !parameter.resource_id.is_empty()
                        && parameter.resource_byte_count > 0
                        && parameter.byte_count > 0
                        && parameter
                            .byte_offset
                            .checked_add(parameter.byte_count)
                            .is_some_and(|end| end <= parameter.resource_byte_count)
                })
        })
}

fn is_strictly_sorted<T: Ord>(items: &[T]) -> bool {
    items.windows(2).all(|pair| pair[0] < pair[1])
}

fn valid_sha256_digest(value: &str) -> bool {
    value
        .strip_prefix("sha256:")
        .is_some_and(|hex| hex.len() == 64 && hex.bytes().all(|byte| byte.is_ascii_hexdigit()))
}

#[cfg(test)]
mod placement_calibration_catalog_tests {
    use super::*;

    fn behavior() -> VulkanPlacementBehaviorIdentity {
        VulkanPlacementBehaviorIdentity {
            compiled_execution_signature: format!("sha256:{}", "f".repeat(64)),
            contract_ids: vec!["contract".to_string()],
            implementation_digests: vec![format!("sha256:{}", "a".repeat(64))],
            artifact_digest: format!("sha256:{}", "b".repeat(64)),
            execution_graph_digest: format!("sha256:{}", "d".repeat(64)),
            runtime_implementation_fingerprint: "runtime".to_string(),
            phase: nerve_execution_contracts::ExecutionPhase::Decode,
            shape: VulkanPlacementShapeClass {
                activation_batch_width: 1,
                input_byte_capacity: 16,
                output_byte_capacity: 16,
                operations: vec![VulkanPlacementOperationGeometry::Dispatch {
                    geometry: VulkanPlacementDispatchGeometry {
                        contract_id: "contract".to_string(),
                        logical_extent: 8,
                        sampled_extent: 8,
                        input_width: 8,
                        workgroup_count_x: 1,
                        local_size_x: 64,
                    },
                }],
            },
            input_fixture_digest: format!("sha256:{}", "c".repeat(64)),
            equivalence: VulkanPlacementEquivalenceIdentity::bit_exact(),
        }
    }

    fn observation(
        behavior: VulkanPlacementBehaviorIdentity,
        owner: &str,
        output: &str,
        duration_ns: u64,
        resident_bytes: usize,
    ) -> VulkanPlacementCalibrationObservation {
        let devices = ["gpu0", "gpu1"]
            .into_iter()
            .map(
                |physical_device_id| VulkanPlacementDeviceExecutionIdentity {
                    physical_device_id: physical_device_id.to_string(),
                    api_version: 1,
                    driver_version: 2,
                },
            )
            .collect::<Vec<_>>();
        VulkanPlacementCalibrationObservation {
            execution_case: VulkanPlacementExecutionCaseIdentity {
                behavior,
                strategy: VulkanPlacementExecutionStrategy::TensorParallel,
                devices,
                shards: vec![VulkanPlacementShardIdentity {
                    dispatch_ordinal: 0,
                    participant_ordinal: 0,
                    physical_device_id: owner.to_string(),
                    distribution: "output_rows".to_string(),
                    logical_start: 0,
                    logical_count: 8,
                    selected_resource_indices_by_partition: BTreeMap::new(),
                    selected_resource_fragments_by_partition: BTreeMap::new(),
                    parameter_bytes: 16,
                }],
                input_physical_device_id: "gpu0".to_string(),
                output_physical_device_id: output.to_string(),
                owner_physical_device_id: owner.to_string(),
                transports: vec![VulkanPlacementTransportIdentity {
                    source_physical_device_id: "gpu0".to_string(),
                    destination_physical_device_id: "gpu1".to_string(),
                    byte_capacity: 16,
                    route: "shared_host".to_string(),
                }],
            },
            warmup_call_count: 1,
            measured_call_count: 1,
            complete_transaction: true,
            duration_ns,
            useful_activation_count: 1,
            output_digest: "output".to_string(),
            output_artifact: None,
            output_equivalence: VulkanPlacementOutputEquivalenceEvidence::BitExact,
            state_digest: "state".to_string(),
            resident_bytes_by_physical_device: BTreeMap::from([
                ("gpu0".to_string(), resident_bytes),
                ("gpu1".to_string(), resident_bytes),
            ]),
            transient_peak_bytes_by_physical_device: BTreeMap::from([
                ("gpu0".to_string(), 8),
                ("gpu1".to_string(), 8),
            ]),
            host_resident_bytes: 0,
            host_transient_peak_bytes: 0,
        }
    }

    fn catalog_with_reference() -> VulkanPlacementCalibrationCatalog {
        let mut catalog = VulkanPlacementCalibrationCatalog::default();
        catalog
            .record_reference(VulkanPlacementCanonicalReference {
                behavior: behavior(),
                output_digest: "output".to_string(),
                output_artifact: None,
                state_digest: "state".to_string(),
            })
            .unwrap();
        catalog
    }

    fn device_identity(physical_device_id: &str) -> VulkanPlacementDeviceExecutionIdentity {
        VulkanPlacementDeviceExecutionIdentity {
            physical_device_id: physical_device_id.to_string(),
            api_version: 1,
            driver_version: 2,
        }
    }

    #[test]
    fn selected_resource_shard_identity_is_exact_and_allows_dynamic_parameters() {
        let mut selected = observation(behavior(), "gpu0", "gpu0", 100, 16);
        selected.execution_case.shards[0].distribution = "expert_range".to_string();
        selected.execution_case.shards[0].parameter_bytes = 0;
        selected.execution_case.shards[0].selected_resource_indices_by_partition =
            BTreeMap::from([(0, vec![0, 2, 4, 6])]);
        let mut catalog = catalog_with_reference();
        catalog.record_observation(selected.clone()).unwrap();

        let mut different_ownership = selected.clone();
        *different_ownership.execution_case.shards[0]
            .selected_resource_indices_by_partition
            .get_mut(&0)
            .unwrap() = vec![1, 3, 5, 7];
        assert_ne!(selected.execution_case, different_ownership.execution_case);

        let mut malformed = selected.clone();
        *malformed.execution_case.shards[0]
            .selected_resource_indices_by_partition
            .get_mut(&0)
            .unwrap() = vec![2, 1];
        assert!(catalog.record_observation(malformed).is_err());

        let mut invalid_participant = selected.clone();
        invalid_participant.execution_case.shards[0].participant_ordinal = 2;
        assert!(catalog.record_observation(invalid_participant).is_err());

        let mut skipped_partition = selected;
        let indices = skipped_partition.execution_case.shards[0]
            .selected_resource_indices_by_partition
            .remove(&0)
            .unwrap();
        skipped_partition.execution_case.shards[0]
            .selected_resource_indices_by_partition
            .insert(1, indices);
        assert!(catalog.record_observation(skipped_partition).is_err());
    }

    #[test]
    fn selected_resource_fragment_identity_is_exact_and_bounded() {
        let mut selected = observation(behavior(), "gpu0", "gpu0", 100, 16);
        selected.execution_case.strategy =
            VulkanPlacementExecutionStrategy::IntraExpertTensorParallel;
        selected.execution_case.shards[0].distribution = "output_rows".to_string();
        selected.execution_case.shards[0].parameter_bytes = 0;
        selected.execution_case.shards[0].selected_resource_fragments_by_partition =
            BTreeMap::from([(
                0,
                vec![VulkanPlacementSelectedResourceFragmentIdentity {
                    resource_index: 0,
                    atomic_group_id: "expert-0".to_string(),
                    logical_start: 0,
                    logical_count: 4,
                    parameters: vec![
                        VulkanPlacementSelectedResourceParameterFragmentIdentity {
                            parameter_slot: 0,
                            resource_id: "weight-0".to_string(),
                            resource_byte_count: 16,
                            byte_offset: 0,
                            byte_count: 8,
                        },
                    ],
                }],
            )]);
        let mut catalog = catalog_with_reference();
        catalog.record_observation(selected.clone()).unwrap();

        let mut escaped = selected.clone();
        escaped.execution_case.shards[0]
            .selected_resource_fragments_by_partition
            .get_mut(&0)
            .unwrap()[0]
            .parameters[0]
            .byte_count = 17;
        assert!(catalog.record_observation(escaped).is_err());

        let mut mixed = selected;
        mixed.execution_case.shards[0]
            .selected_resource_indices_by_partition
            .insert(0, vec![0]);
        assert!(catalog.record_observation(mixed).is_err());
    }

    #[test]
    fn catalog_merge_preserves_distinct_exact_candidates() {
        let mut left = catalog_with_reference();
        left.record_observation(observation(behavior(), "gpu0", "gpu0", 100, 16))
            .unwrap();
        let mut right = catalog_with_reference();
        right
            .record_observation(observation(behavior(), "gpu1", "gpu1", 120, 16))
            .unwrap();

        left.merge(&right).unwrap();

        assert_eq!(left.reference_count(), 1);
        assert_eq!(left.observation_count(), 2);
        assert_eq!(left.candidates_for_behavior(&behavior()).len(), 2);
    }

    #[test]
    fn catalog_merge_is_commutative_and_conservative_for_remeasurement() {
        let mut faster = catalog_with_reference();
        faster
            .record_observation(observation(behavior(), "gpu0", "gpu0", 80, 16))
            .unwrap();
        let execution_case = faster.observations[0].execution_case.clone();
        let mut slower = catalog_with_reference();
        slower
            .record_observation(observation(behavior(), "gpu0", "gpu0", 110, 16))
            .unwrap();

        let mut forward = faster.clone();
        forward.merge(&slower).unwrap();
        let mut reverse = slower;
        reverse.merge(&faster).unwrap();

        assert_eq!(forward, reverse);
        assert_eq!(
            forward
                .exact_observation(&execution_case)
                .unwrap()
                .duration_ns,
            110
        );
    }

    #[test]
    fn catalog_merge_rejects_conflicts_without_modifying_the_receiver() {
        let mut accepted = catalog_with_reference();
        accepted
            .record_observation(observation(behavior(), "gpu0", "gpu0", 80, 16))
            .unwrap();
        let original = accepted.clone();
        let mut conflicting = catalog_with_reference();
        conflicting
            .record_observation(observation(behavior(), "gpu0", "gpu0", 80, 32))
            .unwrap();

        let error = accepted.merge(&conflicting).unwrap_err();

        assert!(error.to_string().contains("conflicting evidence"));
        assert_eq!(accepted, original);
    }

    #[test]
    fn observation_requires_an_exact_canonical_reference() {
        let mut catalog = VulkanPlacementCalibrationCatalog::default();
        let error = catalog
            .record_observation(observation(behavior(), "gpu0", "gpu0", 10, 16))
            .unwrap_err();
        assert!(error.0.contains("no exact canonical"));
    }

    #[test]
    fn observation_rejects_output_drift_and_non_micro_measurements() {
        let mut catalog = catalog_with_reference();
        let mut drifted = observation(behavior(), "gpu0", "gpu0", 10, 16);
        drifted.output_digest = "different".to_string();
        assert!(
            catalog
                .record_observation(drifted)
                .unwrap_err()
                .0
                .contains("differs")
        );

        let mut excessive = observation(behavior(), "gpu0", "gpu0", 10, 16);
        excessive.measured_call_count = 3;
        assert!(
            catalog
                .record_observation(excessive)
                .unwrap_err()
                .0
                .contains("bounded complete transaction")
        );
    }

    #[test]
    fn catalog_recomputes_compiler_declared_numeric_equivalence() {
        let artifact = |value: f32| VulkanPlacementOutputArtifact {
            scalar_format: VulkanPlacementScalarFormat::Bf16,
            segments: vec![VulkanPlacementOutputSegment {
                binding: 1,
                name: "hidden".to_string(),
                bytes: (((value.to_bits() >> 16) as u16).to_le_bytes()).to_vec(),
            }],
        };
        let mut behavior = behavior();
        behavior.equivalence = VulkanPlacementEquivalenceIdentity {
            output: VulkanPlacementEquivalenceKind::AbsoluteRelativeTolerance,
            state: VulkanPlacementEquivalenceKind::BitExact,
            absolute_tolerance_bits: Some(0.01f64.to_bits()),
            relative_tolerance_bits: Some(0.01f64.to_bits()),
            output_scalar_format: Some(VulkanPlacementScalarFormat::Bf16),
        };
        let reference_artifact = artifact(1.0);
        let candidate_artifact = artifact(1.0078125);
        let reference_digest =
            vulkan_placement_output_artifact_digest(&reference_artifact).unwrap();
        let candidate_digest =
            vulkan_placement_output_artifact_digest(&candidate_artifact).unwrap();
        let evidence = validate_vulkan_placement_output_equivalence(
            &behavior.equivalence,
            &reference_digest,
            Some(&reference_artifact),
            &candidate_digest,
            Some(&candidate_artifact),
        )
        .unwrap();
        let mut catalog = VulkanPlacementCalibrationCatalog::default();
        catalog
            .record_reference(VulkanPlacementCanonicalReference {
                behavior: behavior.clone(),
                output_digest: reference_digest,
                output_artifact: Some(reference_artifact),
                state_digest: "state".to_string(),
            })
            .unwrap();
        let mut accepted = observation(behavior.clone(), "gpu0", "gpu0", 10, 16);
        accepted.output_digest = candidate_digest;
        accepted.output_artifact = Some(candidate_artifact);
        accepted.output_equivalence = evidence;
        catalog.record_observation(accepted).unwrap();
        VulkanPlacementCalibrationCatalog::from_json_slice(&catalog.to_json_bytes().unwrap())
            .unwrap();

        let mut rejected = observation(behavior, "gpu1", "gpu1", 10, 16);
        let rejected_artifact = artifact(1.03125);
        rejected.output_digest =
            vulkan_placement_output_artifact_digest(&rejected_artifact).unwrap();
        rejected.output_artifact = Some(rejected_artifact);
        rejected.output_equivalence =
            VulkanPlacementOutputEquivalenceEvidence::AbsoluteRelativeTolerance {
                compared_element_count: 1,
                maximum_absolute_error_bits: 0,
                maximum_relative_error_bits: 0,
            };
        assert!(catalog.record_observation(rejected).is_err());
    }

    #[test]
    fn exact_case_identity_invalidates_driver_or_transport_changes() {
        let mut catalog = catalog_with_reference();
        let accepted = observation(behavior(), "gpu0", "gpu0", 10, 16);
        let accepted_case = accepted.execution_case.clone();
        catalog.record_observation(accepted).unwrap();

        let mut stale_driver = accepted_case.clone();
        stale_driver.devices[0].driver_version += 1;
        assert!(catalog.exact_observation(&stale_driver).is_none());

        let mut another_route = accepted_case;
        another_route.transports[0].route = "external_device_local".to_string();
        assert!(catalog.exact_observation(&another_route).is_none());
    }

    #[test]
    fn compiled_execution_signature_separates_same_shape_transactions() {
        let first_behavior = behavior();
        let mut second_behavior = first_behavior.clone();
        second_behavior.compiled_execution_signature = format!("sha256:{}", "e".repeat(64));

        let mut catalog = VulkanPlacementCalibrationCatalog::default();
        for behavior in [first_behavior.clone(), second_behavior.clone()] {
            catalog
                .record_reference(VulkanPlacementCanonicalReference {
                    behavior: behavior.clone(),
                    output_digest: "output".to_string(),
                    output_artifact: None,
                    state_digest: "state".to_string(),
                })
                .unwrap();
            catalog
                .record_observation(observation(behavior, "gpu0", "gpu1", 10, 16))
                .unwrap();
        }

        assert_ne!(first_behavior, second_behavior);
        assert_eq!(catalog.candidates_for_behavior(&first_behavior).len(), 1);
        assert_eq!(catalog.candidates_for_behavior(&second_behavior).len(), 1);
    }

    #[test]
    fn candidate_set_keeps_a_slower_candidate_with_a_different_exit() {
        let mut catalog = catalog_with_reference();
        catalog
            .record_observation(observation(behavior(), "gpu0", "gpu0", 10, 16))
            .unwrap();
        catalog
            .record_observation(observation(behavior(), "gpu1", "gpu1", 12, 16))
            .unwrap();

        assert_eq!(catalog.candidates_for_behavior(&behavior()).len(), 2);
    }

    #[test]
    fn candidate_set_preserves_a_slower_candidate_with_a_different_owner() {
        let mut catalog = catalog_with_reference();
        catalog
            .record_observation(observation(behavior(), "gpu0", "gpu0", 10, 16))
            .unwrap();
        catalog
            .record_observation(observation(behavior(), "gpu1", "gpu0", 12, 32))
            .unwrap();

        assert_eq!(catalog.candidates_for_behavior(&behavior()).len(), 2);
    }

    #[test]
    fn candidate_set_does_not_locally_prune_a_different_execution_strategy() {
        let mut catalog = catalog_with_reference();
        catalog
            .record_observation(observation(behavior(), "gpu0", "gpu0", 10, 16))
            .unwrap();
        let mut slower = observation(behavior(), "gpu0", "gpu0", 12, 32);
        slower.execution_case.strategy = VulkanPlacementExecutionStrategy::Hybrid;
        catalog.record_observation(slower).unwrap();

        assert_eq!(catalog.candidates_for_behavior(&behavior()).len(), 2);
    }

    #[test]
    fn candidate_set_preserves_distinct_bounded_measurement_shapes() {
        let behavior = behavior();
        let mut catalog = catalog_with_reference();
        let mut two_calls = observation(behavior.clone(), "gpu0", "gpu0", 18, 16);
        two_calls.execution_case.strategy = VulkanPlacementExecutionStrategy::Hybrid;
        two_calls.measured_call_count = 2;
        two_calls.useful_activation_count = 2;
        let one_call = observation(behavior.clone(), "gpu0", "gpu0", 10, 16);
        catalog.record_observation(two_calls).unwrap();
        catalog.record_observation(one_call).unwrap();

        let candidates = catalog.candidates_for_behavior(&behavior);
        assert_eq!(candidates.len(), 2);
        assert!(candidates.iter().any(|candidate| {
            candidate.duration_ns == 18 && candidate.useful_activation_count == 2
        }));
    }

    #[test]
    fn candidate_set_preserves_host_vs_device_resource_tradeoffs() {
        let behavior = behavior();
        let mut catalog = catalog_with_reference();
        let mut faster_shared_host = observation(behavior.clone(), "gpu0", "gpu0", 8, 16);
        faster_shared_host.host_transient_peak_bytes = 4096;
        let mut slower_device_local = observation(behavior.clone(), "gpu0", "gpu0", 10, 16);
        slower_device_local.execution_case.transports[0].route =
            "external_device_local".to_string();
        slower_device_local.transient_peak_bytes_by_physical_device =
            BTreeMap::from([("gpu0".to_string(), 4096), ("gpu1".to_string(), 8)]);
        catalog.record_observation(faster_shared_host).unwrap();
        catalog.record_observation(slower_device_local).unwrap();

        assert_eq!(catalog.candidates_for_behavior(&behavior).len(), 2);
    }

    #[test]
    fn capacity_filter_requires_the_complete_transaction_on_every_participant() {
        let behavior = behavior();
        let mut catalog = catalog_with_reference();
        let mut candidate = observation(behavior.clone(), "gpu0", "gpu0", 10, 80);
        candidate.transient_peak_bytes_by_physical_device =
            BTreeMap::from([("gpu0".to_string(), 20), ("gpu1".to_string(), 20)]);
        candidate.host_resident_bytes = 30;
        candidate.host_transient_peak_bytes = 10;
        catalog.record_observation(candidate).unwrap();

        let exact_fit = VulkanPlacementCapacityEnvelope {
            available_bytes_by_device: BTreeMap::from([
                (device_identity("gpu0"), 100),
                (device_identity("gpu1"), 100),
            ]),
            host_available_bytes: 40,
        };
        assert_eq!(
            catalog.candidates_for_capacity(&behavior, &exact_fit).len(),
            1,
        );

        let mut missing_participant = exact_fit.clone();
        missing_participant
            .available_bytes_by_device
            .remove(&device_identity("gpu1"));
        assert!(
            catalog
                .candidates_for_capacity(&behavior, &missing_participant)
                .is_empty()
        );

        let mut transient_does_not_fit = exact_fit.clone();
        transient_does_not_fit
            .available_bytes_by_device
            .insert(device_identity("gpu0"), 99);
        assert!(
            catalog
                .candidates_for_capacity(&behavior, &transient_does_not_fit)
                .is_empty()
        );

        let mut stale_driver = exact_fit.clone();
        stale_driver
            .available_bytes_by_device
            .remove(&device_identity("gpu1"));
        stale_driver.available_bytes_by_device.insert(
            VulkanPlacementDeviceExecutionIdentity {
                physical_device_id: "gpu1".to_string(),
                api_version: 1,
                driver_version: 3,
            },
            100,
        );
        assert!(
            catalog
                .candidates_for_capacity(&behavior, &stale_driver)
                .is_empty()
        );

        let mut host_does_not_fit = exact_fit;
        host_does_not_fit.host_available_bytes = 39;
        assert!(
            catalog
                .candidates_for_capacity(&behavior, &host_does_not_fit)
                .is_empty()
        );
    }

    #[test]
    fn capacity_filter_rejects_overflow_instead_of_wrapping_to_a_false_fit() {
        let behavior = behavior();
        let mut catalog = catalog_with_reference();
        let mut candidate = observation(behavior.clone(), "gpu0", "gpu0", 10, usize::MAX);
        candidate.transient_peak_bytes_by_physical_device =
            BTreeMap::from([("gpu0".to_string(), 1), ("gpu1".to_string(), 1)]);
        catalog.record_observation(candidate).unwrap();
        let capacity = VulkanPlacementCapacityEnvelope {
            available_bytes_by_device: BTreeMap::from([
                (device_identity("gpu0"), usize::MAX),
                (device_identity("gpu1"), usize::MAX),
            ]),
            host_available_bytes: usize::MAX,
        };

        assert!(
            catalog
                .candidates_for_capacity(&behavior, &capacity)
                .is_empty()
        );
    }

    #[test]
    fn duplicate_or_incomplete_resource_evidence_is_rejected_without_mutation() {
        let mut catalog = catalog_with_reference();
        let accepted = observation(behavior(), "gpu0", "gpu0", 10, 16);
        let execution_case = accepted.execution_case.clone();
        catalog.record_observation(accepted).unwrap();
        let replacement = observation(behavior(), "gpu0", "gpu0", 99, 16);
        assert!(catalog.record_observation(replacement).is_err());
        assert_eq!(
            catalog
                .exact_observation(&execution_case)
                .unwrap()
                .duration_ns,
            10,
        );

        let mut incomplete = observation(behavior(), "gpu1", "gpu1", 12, 16);
        incomplete.resident_bytes_by_physical_device.remove("gpu1");
        assert!(catalog.record_observation(incomplete).is_err());
    }

    #[test]
    fn catalog_deserialization_rejects_a_stale_schema() {
        let mut catalog = catalog_with_reference();
        catalog.schema = "nerve.vulkan_placement_calibration_catalog.v1".to_string();
        let payload = serde_json::to_vec(&catalog).unwrap();
        assert!(
            VulkanPlacementCalibrationCatalog::from_json_slice(&payload)
                .unwrap_err()
                .0
                .contains("unsupported")
        );
    }

    #[test]
    fn catalog_round_trip_preserves_structured_exact_identities() {
        let mut catalog = catalog_with_reference();
        let accepted = observation(behavior(), "gpu0", "gpu0", 10, 16);
        let execution_case = accepted.execution_case.clone();
        catalog.record_observation(accepted).unwrap();

        let payload = catalog.to_json_bytes().unwrap();
        let decoded = VulkanPlacementCalibrationCatalog::from_json_slice(&payload).unwrap();
        assert_eq!(decoded, catalog);
        assert_eq!(
            decoded
                .exact_observation(&execution_case)
                .unwrap()
                .duration_ns,
            10,
        );
    }

    #[test]
    fn typed_non_dispatch_transactions_validate_without_fake_shader_geometry() {
        let mut catalog = VulkanPlacementCalibrationCatalog::default();
        for (suffix, operation) in [
            (
                "transfer",
                VulkanPlacementOperationGeometry::DirectedTransfer {
                    contract_id: "contract".to_string(),
                    byte_count: 4096,
                },
            ),
            (
                "reduction",
                VulkanPlacementOperationGeometry::Reduction {
                    contract_id: "contract".to_string(),
                    element_count: 1024,
                    element_byte_count: 4,
                    participant_count: 3,
                },
            ),
            (
                "load",
                VulkanPlacementOperationGeometry::LazyLoadWave {
                    contract_id: "contract".to_string(),
                    resource_count: 6,
                    byte_count: 8192,
                },
            ),
        ] {
            let mut behavior = behavior();
            behavior.execution_graph_digest = format!(
                "sha256:{}",
                match suffix {
                    "transfer" => "1",
                    "reduction" => "2",
                    _ => "3",
                }
                .repeat(64),
            );
            behavior.shape.operations = vec![operation];
            catalog
                .record_reference(VulkanPlacementCanonicalReference {
                    behavior,
                    output_digest: format!("output-{suffix}"),
                    output_artifact: None,
                    state_digest: format!("state-{suffix}"),
                })
                .unwrap();
        }
        catalog.validate().unwrap();
    }

    #[test]
    fn typed_transactions_reject_invalid_reduction_and_unknown_contract() {
        let mut invalid_reduction = behavior();
        invalid_reduction.shape.operations = vec![VulkanPlacementOperationGeometry::Reduction {
            contract_id: "contract".to_string(),
            element_count: 128,
            element_byte_count: 4,
            participant_count: 1,
        }];
        assert!(validate_behavior_identity(&invalid_reduction).is_err());

        let mut unknown_contract = behavior();
        unknown_contract.shape.operations =
            vec![VulkanPlacementOperationGeometry::DirectedTransfer {
                contract_id: "other-contract".to_string(),
                byte_count: 16,
            }];
        assert!(validate_behavior_identity(&unknown_contract).is_err());
    }
}
