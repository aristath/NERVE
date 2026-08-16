#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum VulkanHybridResourceClass {
    Permanent,
    MutableState,
    CacheQuota,
    AtomicLoadWave,
    ExecutionTransient,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum VulkanHybridResourceTarget {
    Device(VulkanPlacementDeviceExecutionIdentity),
    Host,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct VulkanHybridResourceClaim {
    pub claim_id: String,
    pub target: VulkanHybridResourceTarget,
    pub class: VulkanHybridResourceClass,
    pub byte_count: usize,
    pub shared: bool,
}

impl VulkanHybridResourceClaim {
    pub fn device(
        claim_id: impl Into<String>,
        device: VulkanPlacementDeviceExecutionIdentity,
        class: VulkanHybridResourceClass,
        byte_count: usize,
    ) -> Self {
        Self {
            claim_id: claim_id.into(),
            target: VulkanHybridResourceTarget::Device(device),
            class,
            byte_count,
            shared: true,
        }
    }

    pub fn host(
        claim_id: impl Into<String>,
        class: VulkanHybridResourceClass,
        byte_count: usize,
    ) -> Self {
        Self {
            claim_id: claim_id.into(),
            target: VulkanHybridResourceTarget::Host,
            class,
            byte_count,
            shared: true,
        }
    }

    pub fn exclusive_device(
        claim_id: impl Into<String>,
        device: VulkanPlacementDeviceExecutionIdentity,
        class: VulkanHybridResourceClass,
        byte_count: usize,
    ) -> Self {
        let mut claim = Self::device(claim_id, device, class, byte_count);
        claim.shared = false;
        claim
    }

    pub fn exclusive_host(
        claim_id: impl Into<String>,
        class: VulkanHybridResourceClass,
        byte_count: usize,
    ) -> Self {
        let mut claim = Self::host(claim_id, class, byte_count);
        claim.shared = false;
        claim
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VulkanHybridCandidateResources {
    pub claims: Vec<VulkanHybridResourceClaim>,
}

impl VulkanHybridCandidateResources {
    pub fn new(claims: Vec<VulkanHybridResourceClaim>) -> Self {
        Self { claims }
    }
}

/// One exact byte range of an immutable resource used by a physical candidate.
///
/// Candidate islands frequently refer to different views of the same tensor:
/// a local implementation may retain the complete allocation while a
/// distributed implementation retains one or more fragments. Treating those
/// views as unrelated byte counts double-charges overlaps and makes capacity
/// pruning depend on the route used to discover them. Canonicalization splits
/// every shared physical resource at the union of all candidate boundaries so
/// every candidate claims the same indivisible blocks.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct VulkanHybridSharedRangeRequirement {
    pub resource_identity: String,
    pub target: VulkanHybridResourceTarget,
    pub class: VulkanHybridResourceClass,
    pub byte_offset: usize,
    pub byte_count: usize,
}

impl VulkanHybridSharedRangeRequirement {
    pub fn device_parameter(
        resource_identity: impl Into<String>,
        device: VulkanPlacementDeviceExecutionIdentity,
        byte_offset: usize,
        byte_count: usize,
    ) -> Self {
        Self {
            resource_identity: resource_identity.into(),
            target: VulkanHybridResourceTarget::Device(device),
            class: VulkanHybridResourceClass::Permanent,
            byte_offset,
            byte_count,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct VulkanHybridSharedRangeKey {
    resource_identity: String,
    target: VulkanHybridResourceTarget,
    class: VulkanHybridResourceClass,
}

/// Converts exact candidate byte ranges into shared, globally canonical
/// resource claims. The returned catalog is deterministic and additive across
/// candidates: overlapping ranges deduplicate, disjoint ranges accumulate, and
/// the same resource on distinct physical targets remains distinct.
pub fn canonical_vulkan_hybrid_shared_range_resources(
    requirements_by_candidate: &BTreeMap<String, Vec<VulkanHybridSharedRangeRequirement>>,
) -> Result<BTreeMap<String, VulkanHybridCandidateResources>, VulkanHybridResourceError> {
    let mut boundaries_by_resource = BTreeMap::<VulkanHybridSharedRangeKey, BTreeSet<usize>>::new();
    let mut requirements_by_candidate_and_resource =
        BTreeMap::<String, BTreeMap<VulkanHybridSharedRangeKey, Vec<(usize, usize)>>>::new();
    for (candidate_id, requirements) in requirements_by_candidate {
        if candidate_id.trim().is_empty() {
            return Err(VulkanHybridResourceError(
                "hybrid shared-range resources require nonempty candidate identities".to_string(),
            ));
        }
        for requirement in requirements {
            validate_hybrid_shared_range_requirement(requirement)?;
            let end = requirement
                .byte_offset
                .checked_add(requirement.byte_count)
                .expect("shared-range validation proved a bounded interval");
            let key = VulkanHybridSharedRangeKey {
                resource_identity: requirement.resource_identity.clone(),
                target: requirement.target.clone(),
                class: requirement.class,
            };
            let boundaries = boundaries_by_resource.entry(key.clone()).or_default();
            boundaries.insert(requirement.byte_offset);
            boundaries.insert(end);
            requirements_by_candidate_and_resource
                .entry(candidate_id.clone())
                .or_default()
                .entry(key)
                .or_default()
                .push((requirement.byte_offset, end));
        }
        requirements_by_candidate_and_resource
            .entry(candidate_id.clone())
            .or_default();
    }

    let mut result = BTreeMap::new();
    for (candidate_id, requirements_by_resource) in requirements_by_candidate_and_resource {
        let mut claims = Vec::new();
        for (key, ranges) in requirements_by_resource {
            let boundaries = boundaries_by_resource
                .get(&key)
                .expect("every candidate range populated global boundaries")
                .iter()
                .copied()
                .collect::<Vec<_>>();
            for pair in boundaries.windows(2) {
                let [start, end] = pair else {
                    unreachable!("a boundary window always contains two entries");
                };
                if start == end
                    || !ranges
                        .iter()
                        .any(|(range_start, range_end)| range_start <= start && end <= range_end)
                {
                    continue;
                }
                let byte_count = end.checked_sub(*start).ok_or_else(|| {
                    VulkanHybridResourceError(
                        "hybrid shared-range canonical interval underflowed".to_string(),
                    )
                })?;
                claims.push(VulkanHybridResourceClaim {
                    claim_id: canonical_hybrid_shared_range_claim_id(&key, *start, *end),
                    target: key.target.clone(),
                    class: key.class,
                    byte_count,
                    shared: true,
                });
            }
        }
        result.insert(candidate_id, VulkanHybridCandidateResources::new(claims));
    }
    Ok(result)
}

fn validate_hybrid_shared_range_requirement(
    requirement: &VulkanHybridSharedRangeRequirement,
) -> Result<(), VulkanHybridResourceError> {
    if requirement.resource_identity.trim().is_empty()
        || requirement.byte_count == 0
        || requirement
            .byte_offset
            .checked_add(requirement.byte_count)
            .is_none()
    {
        return Err(VulkanHybridResourceError(
            "hybrid shared-range requirements need a nonempty resource and a positive bounded interval"
                .to_string(),
        ));
    }
    if matches!(
        &requirement.target,
        VulkanHybridResourceTarget::Device(device)
            if device.physical_device_id.trim().is_empty()
    ) {
        return Err(VulkanHybridResourceError(
            "hybrid shared-range requirement has an empty physical device identity".to_string(),
        ));
    }
    Ok(())
}

fn canonical_hybrid_shared_range_claim_id(
    key: &VulkanHybridSharedRangeKey,
    byte_start: usize,
    byte_end: usize,
) -> String {
    let mut hasher = Sha256::new();
    update_hybrid_claim_digest_field(&mut hasher, b"nerve.hybrid.shared_range.v1");
    match &key.target {
        VulkanHybridResourceTarget::Device(device) => {
            update_hybrid_claim_digest_field(&mut hasher, b"device");
            update_hybrid_claim_digest_field(&mut hasher, device.physical_device_id.as_bytes());
            update_hybrid_claim_digest_field(&mut hasher, &device.api_version.to_le_bytes());
            update_hybrid_claim_digest_field(&mut hasher, &device.driver_version.to_le_bytes());
        }
        VulkanHybridResourceTarget::Host => {
            update_hybrid_claim_digest_field(&mut hasher, b"host");
        }
    }
    update_hybrid_claim_digest_field(
        &mut hasher,
        match key.class {
            VulkanHybridResourceClass::Permanent => b"permanent",
            VulkanHybridResourceClass::MutableState => b"mutable_state",
            VulkanHybridResourceClass::CacheQuota => b"cache_quota",
            VulkanHybridResourceClass::AtomicLoadWave => b"atomic_load_wave",
            VulkanHybridResourceClass::ExecutionTransient => b"execution_transient",
        },
    );
    update_hybrid_claim_digest_field(&mut hasher, key.resource_identity.as_bytes());
    update_hybrid_claim_digest_field(&mut hasher, &(byte_start as u128).to_le_bytes());
    update_hybrid_claim_digest_field(&mut hasher, &(byte_end as u128).to_le_bytes());
    format!("shared-range:sha256:{:x}", hasher.finalize())
}

fn update_hybrid_claim_digest_field(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u128).to_le_bytes());
    hasher.update(value);
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VulkanHybridResourceBytes {
    pub permanent_bytes: usize,
    pub mutable_state_bytes: usize,
    pub cache_quota_bytes: usize,
    pub atomic_load_wave_bytes: usize,
    pub execution_transient_peak_bytes: usize,
}

impl VulkanHybridResourceBytes {
    pub fn retained_bytes(&self) -> Result<usize, VulkanHybridResourceError> {
        self.permanent_bytes
            .checked_add(self.mutable_state_bytes)
            .and_then(|bytes| bytes.checked_add(self.cache_quota_bytes))
            .ok_or_else(|| {
                VulkanHybridResourceError(
                    "hybrid retained resource capacity overflowed".to_string(),
                )
            })
    }

    pub fn required_capacity_bytes(&self) -> Result<usize, VulkanHybridResourceError> {
        if self.atomic_load_wave_bytes > self.cache_quota_bytes {
            return Err(VulkanHybridResourceError(format!(
                "hybrid atomic load wave needs {} bytes but its admitted cache quota contains only {} bytes",
                self.atomic_load_wave_bytes, self.cache_quota_bytes,
            )));
        }
        self.retained_bytes()?
            .checked_add(self.execution_transient_peak_bytes)
            .ok_or_else(|| {
                VulkanHybridResourceError(
                    "hybrid required resource capacity overflowed".to_string(),
                )
            })
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VulkanHybridResourceReservations {
    pub device_bytes: BTreeMap<VulkanPlacementDeviceExecutionIdentity, VulkanHybridResourceBytes>,
    pub host_bytes: VulkanHybridResourceBytes,
    shared_claims_by_id: BTreeMap<String, VulkanHybridResourceClaim>,
    exclusive_claim_ids: BTreeSet<String>,
}

impl VulkanHybridResourceReservations {
    fn reserve(
        &self,
        resources: &VulkanHybridCandidateResources,
        capacity: &VulkanPlacementCapacityEnvelope,
    ) -> Result<Option<Self>, VulkanHybridResourceError> {
        let mut next = self.clone();
        for claim in &resources.claims {
            validate_hybrid_resource_claim(claim)?;
            if claim.shared {
                if let Some(existing) = next.shared_claims_by_id.get(&claim.claim_id) {
                    if existing != claim {
                        return Err(VulkanHybridResourceError(format!(
                            "hybrid resource claim {:?} has conflicting definitions",
                            claim.claim_id,
                        )));
                    }
                    continue;
                }
                next.shared_claims_by_id
                    .insert(claim.claim_id.clone(), claim.clone());
            } else if !next.exclusive_claim_ids.insert(claim.claim_id.clone()) {
                return Err(VulkanHybridResourceError(format!(
                    "hybrid exclusive resource claim {:?} was reserved twice",
                    claim.claim_id,
                )));
            }
            let bytes = match &claim.target {
                VulkanHybridResourceTarget::Device(device) => {
                    if !capacity.available_bytes_by_device.contains_key(device) {
                        return Ok(None);
                    }
                    next.device_bytes.entry(device.clone()).or_default()
                }
                VulkanHybridResourceTarget::Host => &mut next.host_bytes,
            };
            add_hybrid_resource_claim(bytes, claim)?;
        }
        for (device, bytes) in &next.device_bytes {
            let Some(available) = capacity.available_bytes_by_device.get(device).copied() else {
                return Ok(None);
            };
            let Ok(required) = bytes.required_capacity_bytes() else {
                return Ok(None);
            };
            if required > available {
                return Ok(None);
            }
        }
        let Ok(host_required) = next.host_bytes.required_capacity_bytes() else {
            return Ok(None);
        };
        if host_required > capacity.host_available_bytes {
            return Ok(None);
        }
        Ok(Some(next))
    }

    fn claims_are_subset_of(&self, other: &Self) -> bool {
        self.shared_claims_by_id
            .iter()
            .all(|(claim_id, claim)| other.shared_claims_by_id.get(claim_id) == Some(claim))
            && self.resources_no_greater_than(other)
    }

    fn resources_no_greater_than(&self, other: &Self) -> bool {
        let devices = self
            .device_bytes
            .keys()
            .chain(other.device_bytes.keys())
            .collect::<BTreeSet<_>>();
        devices.into_iter().all(|device| {
            self.device_bytes
                .get(device)
                .cloned()
                .unwrap_or_default()
                .no_greater_than(&other.device_bytes.get(device).cloned().unwrap_or_default())
        }) && self.host_bytes.no_greater_than(&other.host_bytes)
    }

    fn ordering_totals(&self) -> (usize, usize, usize, usize) {
        let (device_retained, device_transient) =
            self.device_bytes
                .values()
                .fold((0usize, 0usize), |(retained, transient), bytes| {
                    (
                        retained.saturating_add(bytes.retained_bytes().unwrap_or(usize::MAX)),
                        transient.saturating_add(bytes.execution_transient_peak_bytes),
                    )
                });
        (
            device_retained,
            device_transient,
            self.host_bytes.retained_bytes().unwrap_or(usize::MAX),
            self.host_bytes.execution_transient_peak_bytes,
        )
    }
}

impl VulkanHybridResourceBytes {
    fn no_greater_than(&self, other: &Self) -> bool {
        self.permanent_bytes <= other.permanent_bytes
            && self.mutable_state_bytes <= other.mutable_state_bytes
            && self.cache_quota_bytes <= other.cache_quota_bytes
            && self.atomic_load_wave_bytes <= other.atomic_load_wave_bytes
            && self.execution_transient_peak_bytes <= other.execution_transient_peak_bytes
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanHybridResourceError(pub String);

impl Display for VulkanHybridResourceError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for VulkanHybridResourceError {}

fn validate_hybrid_resource_claim(
    claim: &VulkanHybridResourceClaim,
) -> Result<(), VulkanHybridResourceError> {
    if claim.claim_id.trim().is_empty() || claim.byte_count == 0 {
        return Err(VulkanHybridResourceError(
            "hybrid resource claims require a nonempty identity and positive byte count"
                .to_string(),
        ));
    }
    if matches!(
        &claim.target,
        VulkanHybridResourceTarget::Device(device) if device.physical_device_id.trim().is_empty()
    ) {
        return Err(VulkanHybridResourceError(
            "hybrid device resource claim has an empty physical identity".to_string(),
        ));
    }
    Ok(())
}

fn add_hybrid_resource_claim(
    bytes: &mut VulkanHybridResourceBytes,
    claim: &VulkanHybridResourceClaim,
) -> Result<(), VulkanHybridResourceError> {
    let destination = match claim.class {
        VulkanHybridResourceClass::Permanent => &mut bytes.permanent_bytes,
        VulkanHybridResourceClass::MutableState => &mut bytes.mutable_state_bytes,
        VulkanHybridResourceClass::CacheQuota => &mut bytes.cache_quota_bytes,
        VulkanHybridResourceClass::AtomicLoadWave => {
            bytes.atomic_load_wave_bytes = bytes.atomic_load_wave_bytes.max(claim.byte_count);
            return Ok(());
        }
        VulkanHybridResourceClass::ExecutionTransient => {
            bytes.execution_transient_peak_bytes =
                bytes.execution_transient_peak_bytes.max(claim.byte_count);
            return Ok(());
        }
    };
    *destination = destination.checked_add(claim.byte_count).ok_or_else(|| {
        VulkanHybridResourceError("hybrid retained resource accounting overflowed".to_string())
    })?;
    Ok(())
}

#[cfg(test)]
mod hybrid_placement_resource_tests {
    use super::*;

    fn device(id: &str) -> VulkanPlacementDeviceExecutionIdentity {
        VulkanPlacementDeviceExecutionIdentity {
            physical_device_id: id.to_string(),
            api_version: 1,
            driver_version: 2,
        }
    }

    fn capacity() -> VulkanPlacementCapacityEnvelope {
        VulkanPlacementCapacityEnvelope {
            available_bytes_by_device: BTreeMap::from([(device("gpu0"), 100)]),
            host_available_bytes: 100,
        }
    }

    #[test]
    fn shared_claim_identity_requires_one_immutable_definition() {
        let initial = VulkanHybridResourceReservations::default()
            .reserve(
                &VulkanHybridCandidateResources::new(vec![VulkanHybridResourceClaim::device(
                    "shared",
                    device("gpu0"),
                    VulkanHybridResourceClass::Permanent,
                    10,
                )]),
                &capacity(),
            )
            .unwrap()
            .unwrap();
        let error = initial
            .reserve(
                &VulkanHybridCandidateResources::new(vec![VulkanHybridResourceClaim::device(
                    "shared",
                    device("gpu0"),
                    VulkanHybridResourceClass::Permanent,
                    11,
                )]),
                &capacity(),
            )
            .unwrap_err();

        assert!(error.0.contains("conflicting definitions"));
    }

    #[test]
    fn host_state_and_transient_share_the_final_capacity_envelope() {
        let resources = VulkanHybridCandidateResources::new(vec![
            VulkanHybridResourceClaim::host("state", VulkanHybridResourceClass::MutableState, 70),
            VulkanHybridResourceClaim::host(
                "scratch",
                VulkanHybridResourceClass::ExecutionTransient,
                31,
            ),
        ]);

        assert!(
            VulkanHybridResourceReservations::default()
                .reserve(&resources, &capacity())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn shared_ranges_canonicalize_full_tensors_and_overlapping_fragments() {
        let requirements = BTreeMap::from([
            (
                "full".to_string(),
                vec![VulkanHybridSharedRangeRequirement::device_parameter(
                    "tensor.weight",
                    device("gpu0"),
                    0,
                    100,
                )],
            ),
            (
                "fragment".to_string(),
                vec![VulkanHybridSharedRangeRequirement::device_parameter(
                    "tensor.weight",
                    device("gpu0"),
                    20,
                    60,
                )],
            ),
        ]);
        let resources = canonical_vulkan_hybrid_shared_range_resources(&requirements).unwrap();
        assert_eq!(
            resources["full"]
                .claims
                .iter()
                .map(|claim| claim.byte_count)
                .collect::<Vec<_>>(),
            [20, 60, 20]
        );
        assert_eq!(resources["fragment"].claims.len(), 1);

        let after_full = VulkanHybridResourceReservations::default()
            .reserve(&resources["full"], &capacity())
            .unwrap()
            .unwrap();
        let after_overlap = after_full
            .reserve(&resources["fragment"], &capacity())
            .unwrap()
            .unwrap();
        assert_eq!(
            after_overlap.device_bytes[&device("gpu0")].permanent_bytes,
            100
        );
    }

    #[test]
    fn shared_ranges_union_overlaps_within_one_candidate() {
        let requirements = BTreeMap::from([(
            "candidate".to_string(),
            vec![
                VulkanHybridSharedRangeRequirement::device_parameter(
                    "tensor.weight",
                    device("gpu0"),
                    0,
                    60,
                ),
                VulkanHybridSharedRangeRequirement::device_parameter(
                    "tensor.weight",
                    device("gpu0"),
                    40,
                    60,
                ),
            ],
        )]);
        let resources = canonical_vulkan_hybrid_shared_range_resources(&requirements).unwrap();
        let reservation = VulkanHybridResourceReservations::default()
            .reserve(&resources["candidate"], &capacity())
            .unwrap()
            .unwrap();

        assert_eq!(
            reservation.device_bytes[&device("gpu0")].permanent_bytes,
            100
        );
    }

    #[test]
    fn shared_ranges_keep_identical_tensor_ranges_on_distinct_devices_separate() {
        let requirements = BTreeMap::from([
            (
                "gpu0".to_string(),
                vec![VulkanHybridSharedRangeRequirement::device_parameter(
                    "tensor.weight",
                    device("gpu0"),
                    0,
                    40,
                )],
            ),
            (
                "gpu1".to_string(),
                vec![VulkanHybridSharedRangeRequirement::device_parameter(
                    "tensor.weight",
                    device("gpu1"),
                    0,
                    40,
                )],
            ),
        ]);
        let resources = canonical_vulkan_hybrid_shared_range_resources(&requirements).unwrap();

        assert_ne!(
            resources["gpu0"].claims[0].claim_id,
            resources["gpu1"].claims[0].claim_id
        );
    }

    #[test]
    fn shared_ranges_reject_empty_and_overflowing_intervals() {
        for requirement in [
            VulkanHybridSharedRangeRequirement::device_parameter(
                "tensor.weight",
                device("gpu0"),
                0,
                0,
            ),
            VulkanHybridSharedRangeRequirement::device_parameter(
                "tensor.weight",
                device("gpu0"),
                usize::MAX,
                2,
            ),
        ] {
            let error = canonical_vulkan_hybrid_shared_range_resources(&BTreeMap::from([(
                "candidate".to_string(),
                vec![requirement],
            )]))
            .unwrap_err();
            assert!(error.0.contains("positive bounded interval"));
        }
    }
}
