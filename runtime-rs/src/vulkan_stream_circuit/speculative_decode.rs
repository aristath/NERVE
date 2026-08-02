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
