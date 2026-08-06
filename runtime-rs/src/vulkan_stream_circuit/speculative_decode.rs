#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanSpeculativeVerificationResult {
    pub accepted_draft_count: usize,
    pub committed_target_tick_count: usize,
    pub emitted_tokens: Vec<VulkanResidentSampledToken>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanSpeculativeCycleRun {
    pub decoder_id: String,
    pub initial_token_id: u32,
    pub start_stream_tick: u64,
    pub draft_token_ids: Vec<u32>,
    pub target_tokens: Vec<VulkanResidentSampledToken>,
    pub verification: VulkanSpeculativeVerificationResult,
    pub draft_time_ns: u64,
    pub target_verification_time_ns: u64,
    pub draft_catch_up_time_ns: u64,
    pub total_time_ns: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanSpeculativeCycleTrace {
    pub start_stream_tick: u64,
    pub initial_token_id: u32,
    pub draft_token_ids: Vec<u32>,
    pub target_token_ids: Vec<u32>,
    pub accepted_draft_count: usize,
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
    pub windows: BTreeMap<usize, VulkanSpeculativeWindowStats>,
    pub cycle_traces: Vec<VulkanSpeculativeCycleTrace>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanSpeculativeWindowStats {
    pub draft_width: usize,
    pub cycle_count: usize,
    pub emitted_token_count: usize,
    pub total_time_ns: u64,
}

const SPECULATIVE_CYCLE_TRACE_LIMIT: usize = 8;
const ADAPTIVE_SPECULATIVE_WINDOW_WARMUP_CYCLES: usize = 1;
const ADAPTIVE_SPECULATIVE_WINDOW_MEASURED_CYCLES: usize = 2;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct VulkanFeedbackExecutionObservation {
    valid_cycle_count: usize,
    measurement_started: bool,
    measured_cycle_count: usize,
    emitted_token_count: usize,
    total_time_ns: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum VulkanFeedbackExecutionCandidate {
    Scalar,
    Resident,
    Speculative { draft_width: usize },
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanAdaptiveFeedbackExecutionSelector {
    candidates: Vec<VulkanFeedbackExecutionCandidate>,
    candidate_index: usize,
    selected_candidate: VulkanFeedbackExecutionCandidate,
    observations: BTreeMap<VulkanFeedbackExecutionCandidate, VulkanFeedbackExecutionObservation>,
}

fn adaptive_feedback_execution_candidates(
    maximum_width: usize,
    resident_feedback_available: bool,
) -> Vec<VulkanFeedbackExecutionCandidate> {
    assert!(maximum_width > 0, "speculative window width must be nonzero");
    let mut widths = BTreeSet::from([1, maximum_width]);
    let mut class_start = 2usize;
    while class_start <= maximum_width {
        let class_end = class_start
            .checked_mul(2)
            .and_then(|next| next.checked_sub(1))
            .unwrap_or(usize::MAX)
            .min(maximum_width);
        widths.insert(class_start);
        widths.insert(class_end);
        let Some(next) = class_start.checked_mul(2) else {
            break;
        };
        class_start = next;
    }
    let mut candidates =
        Vec::with_capacity(widths.len() + 1 + usize::from(resident_feedback_available));
    candidates.push(VulkanFeedbackExecutionCandidate::Scalar);
    if resident_feedback_available {
        candidates.push(VulkanFeedbackExecutionCandidate::Resident);
    }
    candidates.push(VulkanFeedbackExecutionCandidate::Speculative {
        draft_width: maximum_width,
    });
    candidates.extend(
        widths
            .into_iter()
            .filter(|width| *width != maximum_width)
            .map(|draft_width| VulkanFeedbackExecutionCandidate::Speculative { draft_width }),
    );
    candidates
}

impl VulkanAdaptiveFeedbackExecutionSelector {
    fn new(maximum_width: usize, resident_feedback_available: bool) -> Self {
        let candidates =
            adaptive_feedback_execution_candidates(maximum_width, resident_feedback_available);
        let candidate_index = usize::from(candidates.len() == 1);
        let selected_candidate = candidates[0];
        Self {
            candidates,
            candidate_index,
            selected_candidate,
            observations: BTreeMap::new(),
        }
    }

    fn is_calibrated(&self) -> bool {
        self.candidate_index >= self.candidates.len()
    }

    fn next_candidate(&self) -> VulkanFeedbackExecutionCandidate {
        self.candidates
            .get(self.candidate_index)
            .copied()
            .unwrap_or(self.selected_candidate)
    }

    fn record_speculative_cycle(
        &mut self,
        requested_width: usize,
        cycle: &VulkanSpeculativeCycleRun,
        residency_changed: bool,
    ) {
        let candidate = VulkanFeedbackExecutionCandidate::Speculative {
            draft_width: requested_width,
        };
        if self.is_calibrated()
            || candidate != self.next_candidate()
            || cycle.draft_token_ids.len() != requested_width
            || cycle.total_time_ns == 0
            || cycle.verification.emitted_tokens.is_empty()
        {
            return;
        }
        self.record_observation(
            candidate,
            cycle.verification.emitted_tokens.len(),
            cycle.total_time_ns,
            residency_changed,
        );
    }

    fn record_resident_window(
        &mut self,
        emitted_token_count: usize,
        total_time_ns: u64,
        residency_changed: bool,
    ) {
        let candidate = VulkanFeedbackExecutionCandidate::Resident;
        if self.is_calibrated()
            || candidate != self.next_candidate()
            || total_time_ns == 0
            || emitted_token_count == 0
        {
            return;
        }
        self.record_observation(
            candidate,
            emitted_token_count,
            total_time_ns,
            residency_changed,
        );
    }

    fn record_scalar_tick(&mut self, total_time_ns: u64) {
        let candidate = VulkanFeedbackExecutionCandidate::Scalar;
        if self.is_calibrated()
            || candidate != self.next_candidate()
            || total_time_ns == 0
        {
            return;
        }
        self.record_observation(candidate, 1, total_time_ns, false);
    }

    fn record_observation(
        &mut self,
        candidate: VulkanFeedbackExecutionCandidate,
        emitted_token_count: usize,
        total_time_ns: u64,
        residency_changed: bool,
    ) {
        let observation = self.observations.entry(candidate).or_default();
        observation.valid_cycle_count = observation.valid_cycle_count.saturating_add(1);
        if !observation.measurement_started {
            if !residency_changed {
                observation.measurement_started = true;
                return;
            }
            if observation.valid_cycle_count <= ADAPTIVE_SPECULATIVE_WINDOW_WARMUP_CYCLES {
                return;
            }
        }
        observation.measured_cycle_count = observation.measured_cycle_count.saturating_add(1);
        observation.emitted_token_count = observation
            .emitted_token_count
            .saturating_add(emitted_token_count);
        observation.total_time_ns = observation
            .total_time_ns
            .saturating_add(total_time_ns);
        if observation.measured_cycle_count < ADAPTIVE_SPECULATIVE_WINDOW_MEASURED_CYCLES {
            return;
        }
        self.candidate_index = self.candidate_index.saturating_add(1);
        if self.is_calibrated() {
            self.selected_candidate = self.measured_winner();
        }
    }

    fn measured_winner(&self) -> VulkanFeedbackExecutionCandidate {
        let mut winner = self.selected_candidate;
        for candidate in &self.candidates {
            let Some(candidate_observation) = self.observations.get(candidate) else {
                continue;
            };
            if candidate_observation.measured_cycle_count == 0
                || candidate_observation.total_time_ns == 0
            {
                continue;
            }
            let Some(winner_observation) = self.observations.get(&winner) else {
                winner = *candidate;
                continue;
            };
            let candidate_score = (candidate_observation.emitted_token_count as u128)
                .saturating_mul(winner_observation.total_time_ns as u128);
            let winner_score = (winner_observation.emitted_token_count as u128)
                .saturating_mul(candidate_observation.total_time_ns as u128);
            if candidate_score > winner_score {
                winner = *candidate;
            }
        }
        winner
    }
}

fn speculative_confident_prefix_len(
    confidence_logits: &[f32],
    confidence_threshold: f32,
) -> Result<usize, VulkanError> {
    if !confidence_threshold.is_finite() || !(0.0..=1.0).contains(&confidence_threshold) {
        return Err(VulkanError(format!(
            "speculative confidence threshold {confidence_threshold} is outside [0, 1]"
        )));
    }
    if confidence_logits.iter().any(|logit| !logit.is_finite()) {
        return Err(VulkanError(
            "speculative decoder produced a non-finite confidence logit".to_string(),
        ));
    }
    if confidence_threshold == 0.0 {
        return Ok(confidence_logits.len());
    }
    if confidence_threshold == 1.0 {
        return Ok(0);
    }

    let threshold_logit = (confidence_threshold / (1.0 - confidence_threshold)).ln();
    Ok(confidence_logits
        .iter()
        .take_while(|logit| **logit >= threshold_logit)
        .count())
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
            .saturating_add(cycle.verification.emitted_tokens.len());
        self.draft_time_ns = self.draft_time_ns.saturating_add(cycle.draft_time_ns);
        self.target_verification_time_ns = self
            .target_verification_time_ns
            .saturating_add(cycle.target_verification_time_ns);
        self.draft_catch_up_time_ns = self
            .draft_catch_up_time_ns
            .saturating_add(cycle.draft_catch_up_time_ns);
        self.total_time_ns = self.total_time_ns.saturating_add(cycle.total_time_ns);
        let window = self.windows.entry(cycle.draft_token_ids.len()).or_insert_with(|| {
            VulkanSpeculativeWindowStats {
                draft_width: cycle.draft_token_ids.len(),
                ..VulkanSpeculativeWindowStats::default()
            }
        });
        window.cycle_count = window.cycle_count.saturating_add(1);
        window.emitted_token_count = window
            .emitted_token_count
            .saturating_add(cycle.verification.emitted_tokens.len());
        window.total_time_ns = window.total_time_ns.saturating_add(cycle.total_time_ns);
        if self.cycle_traces.len() < SPECULATIVE_CYCLE_TRACE_LIMIT {
            self.cycle_traces.push(VulkanSpeculativeCycleTrace {
                start_stream_tick: cycle.start_stream_tick,
                initial_token_id: cycle.initial_token_id,
                draft_token_ids: cycle.draft_token_ids.clone(),
                target_token_ids: cycle
                    .target_tokens
                    .iter()
                    .map(|token| token.token_id)
                    .collect(),
                accepted_draft_count: cycle.verification.accepted_draft_count,
            });
        }
    }
}

pub fn verify_speculative_token_prefix(
    draft_token_ids: &[u32],
    target_tokens: &[VulkanResidentSampledToken],
) -> Result<VulkanSpeculativeVerificationResult, VulkanError> {
    let expected_target_count = draft_token_ids
        .len()
        .checked_add(1)
        .ok_or_else(|| VulkanError("speculative verification width overflowed".to_string()))?;
    if target_tokens.len() != expected_target_count {
        return Err(VulkanError(format!(
            "speculative verification has {} draft tokens but {} target predictions; expected {}",
            draft_token_ids.len(),
            target_tokens.len(),
            expected_target_count
        )));
    }

    let accepted_draft_count = draft_token_ids
        .iter()
        .zip(target_tokens)
        .take_while(|(draft, target)| **draft == target.token_id)
        .count();
    let committed_target_tick_count = accepted_draft_count
        .checked_add(1)
        .ok_or_else(|| VulkanError("speculative commit width overflowed".to_string()))?;
    let emitted_tokens = target_tokens[..committed_target_tick_count].to_vec();

    Ok(VulkanSpeculativeVerificationResult {
        accepted_draft_count,
        committed_target_tick_count,
        emitted_tokens,
    })
}

fn truncate_speculative_verification_at_stop(
    verification: &mut VulkanSpeculativeVerificationResult,
    stop_token_ids: &BTreeSet<u32>,
) {
    let Some(stop_index) = verification
        .emitted_tokens
        .iter()
        .position(|token| stop_token_ids.contains(&token.token_id))
    else {
        return;
    };
    verification.accepted_draft_count = verification
        .accepted_draft_count
        .min(stop_index.saturating_add(1));
    verification.committed_target_tick_count = stop_index.saturating_add(1);
    verification.emitted_tokens.truncate(stop_index + 1);
}
