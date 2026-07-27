#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanSpeculativeVerificationResult {
    pub accepted_draft_count: usize,
    pub committed_target_tick_count: usize,
    pub emitted_token_ids: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanSpeculativeCycleRun {
    pub decoder_id: String,
    pub initial_token_id: u32,
    pub start_stream_tick: u64,
    pub draft_token_ids: Vec<u32>,
    pub target_token_ids: Vec<u32>,
    pub verification: VulkanSpeculativeVerificationResult,
    pub draft_time_ns: u64,
    pub target_verification_time_ns: u64,
    pub draft_catch_up_time_ns: u64,
    pub total_time_ns: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VulkanSpeculativeDecodeStats {
    pub cycle_count: usize,
    pub rollback_cycle_count: usize,
    pub proposed_draft_token_count: usize,
    pub accepted_draft_token_count: usize,
    pub emitted_token_count: usize,
    pub draft_time_ns: u64,
    pub target_verification_time_ns: u64,
    pub draft_catch_up_time_ns: u64,
    pub total_time_ns: u64,
}

impl VulkanSpeculativeDecodeStats {
    fn record_cycle(&mut self, cycle: &VulkanSpeculativeCycleRun) {
        self.cycle_count = self.cycle_count.saturating_add(1);
        if cycle.verification.accepted_draft_count < cycle.draft_token_ids.len() {
            self.rollback_cycle_count = self.rollback_cycle_count.saturating_add(1);
        }
        self.proposed_draft_token_count = self
            .proposed_draft_token_count
            .saturating_add(cycle.draft_token_ids.len());
        self.accepted_draft_token_count = self
            .accepted_draft_token_count
            .saturating_add(cycle.verification.accepted_draft_count);
        self.emitted_token_count = self
            .emitted_token_count
            .saturating_add(cycle.verification.emitted_token_ids.len());
        self.draft_time_ns = self.draft_time_ns.saturating_add(cycle.draft_time_ns);
        self.target_verification_time_ns = self
            .target_verification_time_ns
            .saturating_add(cycle.target_verification_time_ns);
        self.draft_catch_up_time_ns = self
            .draft_catch_up_time_ns
            .saturating_add(cycle.draft_catch_up_time_ns);
        self.total_time_ns = self.total_time_ns.saturating_add(cycle.total_time_ns);
    }
}

const VULKAN_SPECULATIVE_POLICY_OBSERVATION_WINDOW: usize = 32;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct VulkanExecutionArmObservations {
    observations: VecDeque<(u64, usize)>,
    observed_time_ns: u64,
    observed_useful_token_count: usize,
}

impl VulkanExecutionArmObservations {
    fn estimated_useful_token_time_ns(&self) -> Option<u64> {
        (self.observed_useful_token_count > 0 && self.observed_time_ns > 0).then(|| {
            self.observed_time_ns
                .div_ceil(
                    u64::try_from(self.observed_useful_token_count)
                        .unwrap_or(u64::MAX),
                )
                .max(1)
        })
    }

    fn observe(&mut self, elapsed_time_ns: u64, useful_token_count: usize) {
        if useful_token_count == 0 || elapsed_time_ns == 0 {
            return;
        }
        self.observations
            .push_back((elapsed_time_ns, useful_token_count));
        self.observed_time_ns = self.observed_time_ns.saturating_add(elapsed_time_ns);
        self.observed_useful_token_count = self
            .observed_useful_token_count
            .saturating_add(useful_token_count);
        while self.observations.len() > VULKAN_SPECULATIVE_POLICY_OBSERVATION_WINDOW {
            let (elapsed_time_ns, emitted_token_count) = self
                .observations
                .pop_front()
                .expect("speculative policy observation window is non-empty");
            self.observed_time_ns =
                self.observed_time_ns.saturating_sub(elapsed_time_ns);
            self.observed_useful_token_count = self
                .observed_useful_token_count
                .saturating_sub(emitted_token_count);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VulkanSpeculativeExecutionArm {
    Speculative,
    Resident,
}

impl VulkanSpeculativeExecutionArm {
    fn alternate(self) -> Self {
        match self {
            Self::Speculative => Self::Resident,
            Self::Resident => Self::Speculative,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct VulkanSpeculativeExecutionPolicy {
    // Both arms use elapsed wall time per emitted token. Comparing the
    // speculative cycle against a narrower GPU-only resident estimate can
    // permanently select the slower host-visible path.
    speculative: VulkanExecutionArmObservations,
    resident: VulkanExecutionArmObservations,
    last_observed_arm: Option<VulkanSpeculativeExecutionArm>,
    consecutive_observation_count: usize,
}

impl VulkanSpeculativeExecutionPolicy {
    fn begin_input_event(&mut self) {
        *self = Self::default();
    }

    fn preferred_arm(&self) -> VulkanSpeculativeExecutionArm {
        let preferred = match (
            self.speculative.estimated_useful_token_time_ns(),
            self.resident.estimated_useful_token_time_ns(),
        ) {
            (None, _) => VulkanSpeculativeExecutionArm::Speculative,
            (Some(_), None) => VulkanSpeculativeExecutionArm::Resident,
            (Some(speculative), Some(resident)) if speculative < resident => {
                VulkanSpeculativeExecutionArm::Speculative
            }
            (Some(_), Some(_)) => VulkanSpeculativeExecutionArm::Resident,
        };
        if self.last_observed_arm == Some(preferred)
            && self.consecutive_observation_count
                >= VULKAN_SPECULATIVE_POLICY_OBSERVATION_WINDOW
        {
            // Refresh the inactive estimate after the active arm has replaced
            // one complete rolling evidence window. This prevents permanent
            // lockout when acceptance or runtime conditions change.
            preferred.alternate()
        } else {
            preferred
        }
    }

    fn should_use_speculative(&self) -> bool {
        self.preferred_arm() == VulkanSpeculativeExecutionArm::Speculative
    }

    fn observe_speculative_cycle(&mut self, cycle: &VulkanSpeculativeCycleRun) {
        self.observe(
            VulkanSpeculativeExecutionArm::Speculative,
            cycle.total_time_ns,
            cycle.verification.emitted_token_ids.len(),
        );
    }

    fn observe_resident_window(&mut self, elapsed_time_ns: u64, useful_token_count: usize) {
        self.observe(
            VulkanSpeculativeExecutionArm::Resident,
            elapsed_time_ns,
            useful_token_count,
        );
    }

    fn observe(
        &mut self,
        arm: VulkanSpeculativeExecutionArm,
        elapsed_time_ns: u64,
        useful_token_count: usize,
    ) {
        if useful_token_count == 0 || elapsed_time_ns == 0 {
            return;
        }
        match arm {
            VulkanSpeculativeExecutionArm::Speculative => self
                .speculative
                .observe(elapsed_time_ns, useful_token_count),
            VulkanSpeculativeExecutionArm::Resident => {
                self.resident.observe(elapsed_time_ns, useful_token_count)
            }
        }
        if self.last_observed_arm == Some(arm) {
            self.consecutive_observation_count =
                self.consecutive_observation_count.saturating_add(1);
        } else {
            self.last_observed_arm = Some(arm);
            self.consecutive_observation_count = 1;
        }
    }
}

pub fn verify_speculative_token_prefix(
    draft_token_ids: &[u32],
    target_token_ids: &[u32],
) -> Result<VulkanSpeculativeVerificationResult, VulkanError> {
    let expected_target_count = draft_token_ids
        .len()
        .checked_add(1)
        .ok_or_else(|| VulkanError("speculative verification width overflowed".to_string()))?;
    if target_token_ids.len() != expected_target_count {
        return Err(VulkanError(format!(
            "speculative verification has {} draft tokens but {} target predictions; expected {}",
            draft_token_ids.len(),
            target_token_ids.len(),
            expected_target_count
        )));
    }

    let accepted_draft_count = draft_token_ids
        .iter()
        .zip(target_token_ids)
        .take_while(|(draft, target)| draft == target)
        .count();
    let committed_target_tick_count = accepted_draft_count
        .checked_add(1)
        .ok_or_else(|| VulkanError("speculative commit width overflowed".to_string()))?;
    let mut emitted_token_ids = draft_token_ids[..accepted_draft_count].to_vec();
    emitted_token_ids.push(target_token_ids[accepted_draft_count]);

    Ok(VulkanSpeculativeVerificationResult {
        accepted_draft_count,
        committed_target_tick_count,
        emitted_token_ids,
    })
}

fn truncate_speculative_verification_at_stop(
    verification: &mut VulkanSpeculativeVerificationResult,
    stop_token_ids: &BTreeSet<u32>,
) {
    let Some(stop_index) = verification
        .emitted_token_ids
        .iter()
        .position(|token_id| stop_token_ids.contains(token_id))
    else {
        return;
    };
    verification.accepted_draft_count = verification
        .accepted_draft_count
        .min(stop_index.saturating_add(1));
    verification.committed_target_tick_count = stop_index.saturating_add(1);
    verification.emitted_token_ids.truncate(stop_index + 1);
}
