pub const VULKAN_PLACEMENT_CALIBRATION_CATALOG_SCHEMA: &str =
    "nerve.vulkan_placement_calibration_catalog.v2";

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
    pub physical_device_id: String,
    pub distribution: String,
    pub logical_start: usize,
    pub logical_count: usize,
    pub parameter_bytes: usize,
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
    pub contract_ids: Vec<String>,
    pub implementation_digests: Vec<String>,
    pub artifact_digest: String,
    pub execution_graph_digest: String,
    pub runtime_implementation_fingerprint: String,
    pub phase: nerve_execution_contracts::ExecutionPhase,
    pub shape: VulkanPlacementShapeClass,
    pub input_fixture_digest: String,
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
    pub state_digest: String,
    pub resident_bytes_by_physical_device: BTreeMap<String, usize>,
    pub transient_peak_bytes_by_physical_device: BTreeMap<String, usize>,
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
                reference
                    .behavior
                    .cmp(&observation.execution_case.behavior)
            })
            .ok()
            .map(|index| &self.references[index])
            .ok_or_else(|| {
                VulkanPlacementCalibrationCatalogError(
                    "placement candidate has no exact canonical behavior reference".to_string(),
                )
            })?;
        if observation.output_digest != reference.output_digest
            || observation.state_digest != reference.state_digest
        {
            return Err(VulkanPlacementCalibrationCatalogError(
                "placement candidate output or state differs from its canonical reference"
                    .to_string(),
            ));
        }
        match self.observations.binary_search_by(|existing| {
            existing
                .execution_case
                .cmp(&observation.execution_case)
        }) {
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

    pub fn observations_for_behavior(
        &self,
        behavior: &VulkanPlacementBehaviorIdentity,
    ) -> Vec<&VulkanPlacementCalibrationObservation> {
        self.observations
            .iter()
            .filter(|observation| &observation.execution_case.behavior == behavior)
            .collect()
    }

    /// Returns candidates which are not dominated for the same physical graph
    /// interface. Entry and exit placement remain in the partition key, so a
    /// locally slower owner that removes a neighboring boundary is preserved.
    pub fn pareto_candidates(
        &self,
        behavior: &VulkanPlacementBehaviorIdentity,
    ) -> Vec<&VulkanPlacementCalibrationObservation> {
        let candidates = self.observations_for_behavior(behavior);
        candidates
            .iter()
            .copied()
            .filter(|candidate| {
                !candidates.iter().copied().any(|other| {
                    other.execution_case.input_physical_device_id
                        == candidate.execution_case.input_physical_device_id
                        && other.execution_case.output_physical_device_id
                            == candidate.execution_case.output_physical_device_id
                        && other.execution_case.devices == candidate.execution_case.devices
                        && observation_dominates(other, candidate)
                })
            })
            .collect()
    }
}

fn validate_behavior_identity(
    behavior: &VulkanPlacementBehaviorIdentity,
) -> Result<(), VulkanPlacementCalibrationCatalogError> {
    if behavior.contract_ids.is_empty()
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
    if !is_strictly_sorted(&behavior.contract_ids)
        || !valid_sha256_digest(&behavior.artifact_digest)
        || !valid_sha256_digest(&behavior.execution_graph_digest)
        || behavior
            .implementation_digests
            .iter()
            .any(|digest| !valid_sha256_digest(digest))
        || behavior
            .shape
            .operations
            .iter()
            .any(|operation| !behavior.contract_ids.iter().any(|id| id == operation.contract_id()))
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
        || case.devices.iter().any(|device| device.physical_device_id.is_empty())
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
            "placement observation is incomplete or not a bounded complete transaction"
                .to_string(),
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
                || shard.logical_count == 0
                || shard.parameter_bytes == 0
                || shard.distribution.is_empty()
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

fn is_strictly_sorted<T: Ord>(items: &[T]) -> bool {
    items.windows(2).all(|pair| pair[0] < pair[1])
}

fn valid_sha256_digest(value: &str) -> bool {
    value
        .strip_prefix("sha256:")
        .is_some_and(|hex| hex.len() == 64 && hex.bytes().all(|byte| byte.is_ascii_hexdigit()))
}

fn observation_dominates(
    left: &VulkanPlacementCalibrationObservation,
    right: &VulkanPlacementCalibrationObservation,
) -> bool {
    let resident_no_worse = resource_vector_no_worse(
        &left.resident_bytes_by_physical_device,
        &right.resident_bytes_by_physical_device,
    );
    let transient_no_worse = resource_vector_no_worse(
        &left.transient_peak_bytes_by_physical_device,
        &right.transient_peak_bytes_by_physical_device,
    );
    let strict = left.duration_ns < right.duration_ns
        || left.resident_bytes_by_physical_device != right.resident_bytes_by_physical_device
        || left.transient_peak_bytes_by_physical_device
            != right.transient_peak_bytes_by_physical_device;
    left.duration_ns <= right.duration_ns && resident_no_worse && transient_no_worse && strict
}

fn resource_vector_no_worse(
    left: &BTreeMap<String, usize>,
    right: &BTreeMap<String, usize>,
) -> bool {
    left.keys()
        .chain(right.keys())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .all(|device_id| {
            left.get(device_id).copied().unwrap_or(0)
                <= right.get(device_id).copied().unwrap_or(0)
        })
}

#[cfg(test)]
mod placement_calibration_catalog_tests {
    use super::*;

    fn behavior() -> VulkanPlacementBehaviorIdentity {
        VulkanPlacementBehaviorIdentity {
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
            .map(|physical_device_id| VulkanPlacementDeviceExecutionIdentity {
                physical_device_id: physical_device_id.to_string(),
                api_version: 1,
                driver_version: 2,
            })
            .collect::<Vec<_>>();
        VulkanPlacementCalibrationObservation {
            execution_case: VulkanPlacementExecutionCaseIdentity {
                behavior,
                strategy: VulkanPlacementExecutionStrategy::TensorParallel,
                devices,
                shards: vec![VulkanPlacementShardIdentity {
                    dispatch_ordinal: 0,
                    physical_device_id: owner.to_string(),
                    distribution: "output_rows".to_string(),
                    logical_start: 0,
                    logical_count: 8,
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
            state_digest: "state".to_string(),
            resident_bytes_by_physical_device: BTreeMap::from([
                ("gpu0".to_string(), resident_bytes),
                ("gpu1".to_string(), resident_bytes),
            ]),
            transient_peak_bytes_by_physical_device: BTreeMap::from([
                ("gpu0".to_string(), 8),
                ("gpu1".to_string(), 8),
            ]),
        }
    }

    fn catalog_with_reference() -> VulkanPlacementCalibrationCatalog {
        let mut catalog = VulkanPlacementCalibrationCatalog::default();
        catalog
            .record_reference(VulkanPlacementCanonicalReference {
                behavior: behavior(),
                output_digest: "output".to_string(),
                state_digest: "state".to_string(),
            })
            .unwrap();
        catalog
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
    fn pareto_frontier_keeps_a_slower_candidate_with_a_different_exit() {
        let mut catalog = catalog_with_reference();
        catalog
            .record_observation(observation(behavior(), "gpu0", "gpu0", 10, 16))
            .unwrap();
        catalog
            .record_observation(observation(behavior(), "gpu1", "gpu1", 12, 16))
            .unwrap();

        let frontier = catalog.pareto_candidates(&behavior());
        assert_eq!(frontier.len(), 2);
    }

    #[test]
    fn pareto_frontier_removes_only_same_interface_resource_dominance() {
        let mut catalog = catalog_with_reference();
        catalog
            .record_observation(observation(behavior(), "gpu0", "gpu0", 10, 16))
            .unwrap();
        catalog
            .record_observation(observation(behavior(), "gpu1", "gpu0", 12, 32))
            .unwrap();

        let frontier = catalog.pareto_candidates(&behavior());
        assert_eq!(frontier.len(), 1);
        assert_eq!(frontier[0].duration_ns, 10);
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
            catalog.exact_observation(&execution_case).unwrap().duration_ns,
            10,
        );

        let mut incomplete = observation(behavior(), "gpu1", "gpu1", 12, 16);
        incomplete
            .resident_bytes_by_physical_device
            .remove("gpu1");
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
        unknown_contract.shape.operations = vec![
            VulkanPlacementOperationGeometry::DirectedTransfer {
                contract_id: "other-contract".to_string(),
                byte_count: 16,
            },
        ];
        assert!(validate_behavior_identity(&unknown_contract).is_err());
    }
}
