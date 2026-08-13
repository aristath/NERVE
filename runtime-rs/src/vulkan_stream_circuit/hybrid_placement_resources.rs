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
}
