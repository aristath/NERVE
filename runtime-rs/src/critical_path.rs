use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum RuntimeCriticalPathPhase {
    Protocol = 0,
    SchedulerControl,
    InputPreparation,
    CommandPreparation,
    QueueSubmission,
    HostSynchronization,
    Routing,
    ResidencyGate,
    ExpertCompute,
    DenseProjection,
    GroupedProjection,
    Normalization,
    Quantization,
    StateMemory,
    IndexTransform,
    AttentionScore,
    AttentionSelection,
    AttentionRead,
    PositionalEncoding,
    HyperConnection,
    PointwiseActivation,
    CrossDeviceTransfer,
    OutputProjection,
    Sampling,
    SpeculativeDraft,
    SpeculativeVerification,
    StateCommit,
    Telemetry,
    MixedDeviceCompute,
}

impl RuntimeCriticalPathPhase {
    pub const ALL: [Self; 29] = [
        Self::Protocol,
        Self::SchedulerControl,
        Self::InputPreparation,
        Self::CommandPreparation,
        Self::QueueSubmission,
        Self::HostSynchronization,
        Self::Routing,
        Self::ResidencyGate,
        Self::ExpertCompute,
        Self::DenseProjection,
        Self::GroupedProjection,
        Self::Normalization,
        Self::Quantization,
        Self::StateMemory,
        Self::IndexTransform,
        Self::AttentionScore,
        Self::AttentionSelection,
        Self::AttentionRead,
        Self::PositionalEncoding,
        Self::HyperConnection,
        Self::PointwiseActivation,
        Self::CrossDeviceTransfer,
        Self::OutputProjection,
        Self::Sampling,
        Self::SpeculativeDraft,
        Self::SpeculativeVerification,
        Self::StateCommit,
        Self::Telemetry,
        Self::MixedDeviceCompute,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Protocol => "protocol",
            Self::SchedulerControl => "scheduler_control",
            Self::InputPreparation => "input_preparation",
            Self::CommandPreparation => "command_preparation",
            Self::QueueSubmission => "queue_submission",
            Self::HostSynchronization => "host_synchronization",
            Self::Routing => "routing",
            Self::ResidencyGate => "residency_gate",
            Self::ExpertCompute => "expert_compute",
            Self::DenseProjection => "dense_projection",
            Self::GroupedProjection => "grouped_projection",
            Self::Normalization => "normalization",
            Self::Quantization => "quantization",
            Self::StateMemory => "state_memory",
            Self::IndexTransform => "index_transform",
            Self::AttentionScore => "attention_score",
            Self::AttentionSelection => "attention_selection",
            Self::AttentionRead => "attention_read",
            Self::PositionalEncoding => "positional_encoding",
            Self::HyperConnection => "hyper_connection",
            Self::PointwiseActivation => "pointwise_activation",
            Self::CrossDeviceTransfer => "cross_device_transfer",
            Self::OutputProjection => "output_projection",
            Self::Sampling => "sampling",
            Self::SpeculativeDraft => "speculative_draft",
            Self::SpeculativeVerification => "speculative_verification",
            Self::StateCommit => "state_commit",
            Self::Telemetry => "telemetry",
            Self::MixedDeviceCompute => "mixed_device_compute",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeCriticalPathPhaseReport {
    pub phase: String,
    pub host_invocation_count: u64,
    pub host_inclusive_duration_ns: u64,
    pub host_exclusive_duration_ns: u64,
    pub host_max_inclusive_duration_ns: u64,
    pub device_timestamp_count: u64,
    pub device_duration_ns: u64,
    pub device_max_duration_ns: u64,
    pub host_exclusive_per_generated_token_ns: Option<u64>,
    pub device_per_generated_token_ns: Option<u64>,
    pub host_exclusive_per_execution_window_ns: Option<u64>,
    pub device_per_execution_window_ns: Option<u64>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeCriticalPathReport {
    pub wall_duration_ns: u64,
    pub host_exclusive_work_duration_ns: u64,
    pub host_attributed_critical_path_duration_ns: u64,
    pub host_unattributed_duration_ns: u64,
    pub host_parallel_overlap_duration_ns: u64,
    pub host_coverage_basis_points: u16,
    pub device_timestamp_duration_ns: u64,
    pub generated_token_count: usize,
    pub execution_window_count: usize,
    pub phases: Vec<RuntimeCriticalPathPhaseReport>,
}

impl RuntimeCriticalPathReport {
    pub fn with_normalization(
        mut self,
        generated_token_count: usize,
        execution_window_count: usize,
    ) -> Self {
        self.generated_token_count = generated_token_count;
        self.execution_window_count = execution_window_count;
        for phase in &mut self.phases {
            phase.host_exclusive_per_generated_token_ns =
                per_unit(phase.host_exclusive_duration_ns, generated_token_count);
            phase.device_per_generated_token_ns =
                per_unit(phase.device_duration_ns, generated_token_count);
            phase.host_exclusive_per_execution_window_ns =
                per_unit(phase.host_exclusive_duration_ns, execution_window_count);
            phase.device_per_execution_window_ns =
                per_unit(phase.device_duration_ns, execution_window_count);
        }
        self
    }
}

struct PhaseCounters {
    host_invocation_count: AtomicU64,
    host_inclusive_duration_ns: AtomicU64,
    host_exclusive_duration_ns: AtomicU64,
    host_max_inclusive_duration_ns: AtomicU64,
    device_timestamp_count: AtomicU64,
    device_duration_ns: AtomicU64,
    device_max_duration_ns: AtomicU64,
}

impl PhaseCounters {
    const fn new() -> Self {
        Self {
            host_invocation_count: AtomicU64::new(0),
            host_inclusive_duration_ns: AtomicU64::new(0),
            host_exclusive_duration_ns: AtomicU64::new(0),
            host_max_inclusive_duration_ns: AtomicU64::new(0),
            device_timestamp_count: AtomicU64::new(0),
            device_duration_ns: AtomicU64::new(0),
            device_max_duration_ns: AtomicU64::new(0),
        }
    }

    fn reset(&self) {
        self.host_invocation_count.store(0, Ordering::Relaxed);
        self.host_inclusive_duration_ns.store(0, Ordering::Relaxed);
        self.host_exclusive_duration_ns.store(0, Ordering::Relaxed);
        self.host_max_inclusive_duration_ns
            .store(0, Ordering::Relaxed);
        self.device_timestamp_count.store(0, Ordering::Relaxed);
        self.device_duration_ns.store(0, Ordering::Relaxed);
        self.device_max_duration_ns.store(0, Ordering::Relaxed);
    }
}

static CRITICAL_PATH_EPOCH: AtomicU64 = AtomicU64::new(1);
static NEXT_SPAN_ID: AtomicU64 = AtomicU64::new(1);
static PHASE_COUNTERS: [PhaseCounters; RuntimeCriticalPathPhase::ALL.len()] =
    [const { PhaseCounters::new() }; RuntimeCriticalPathPhase::ALL.len()];

struct ActiveSpan {
    id: u64,
    epoch: u64,
    started: Instant,
    child_duration_ns: u64,
}

struct ActiveDevicePhaseOverride {
    id: u64,
    epoch: u64,
    phase: RuntimeCriticalPathPhase,
}

thread_local! {
    static ACTIVE_SPANS: RefCell<Vec<ActiveSpan>> = const { RefCell::new(Vec::new()) };
    static ACTIVE_DEVICE_PHASE_OVERRIDES: RefCell<Vec<ActiveDevicePhaseOverride>> = const { RefCell::new(Vec::new()) };
}

#[must_use = "the critical-path span must be held for the duration of the measured operation"]
pub struct RuntimeCriticalPathSpan {
    id: u64,
    epoch: u64,
    phase: RuntimeCriticalPathPhase,
    _not_send: PhantomData<Rc<()>>,
}

#[must_use = "the device phase scope must be held for the duration of the measured operation"]
pub struct RuntimeCriticalPathDevicePhaseScope {
    id: u64,
    epoch: u64,
    _not_send: PhantomData<Rc<()>>,
}

impl Drop for RuntimeCriticalPathDevicePhaseScope {
    fn drop(&mut self) {
        ACTIVE_DEVICE_PHASE_OVERRIDES.with(|overrides| {
            let mut overrides = overrides.borrow_mut();
            let Some(index) = overrides.iter().rposition(|scope| scope.id == self.id) else {
                return;
            };
            let scope = overrides.remove(index);
            debug_assert_eq!(scope.epoch, self.epoch);
        });
    }
}

impl Drop for RuntimeCriticalPathSpan {
    fn drop(&mut self) {
        let elapsed_ns = ACTIVE_SPANS.with(|active_spans| {
            let mut active_spans = active_spans.borrow_mut();
            let Some(index) = active_spans.iter().rposition(|span| span.id == self.id) else {
                return None;
            };
            let span = active_spans.remove(index);
            if span.epoch != self.epoch || self.epoch != CRITICAL_PATH_EPOCH.load(Ordering::Relaxed)
            {
                return None;
            }
            let inclusive_ns = elapsed_nanos_u64(span.started);
            let exclusive_ns = inclusive_ns.saturating_sub(span.child_duration_ns);
            if let Some(parent) = active_spans
                .iter_mut()
                .rev()
                .find(|parent| parent.epoch == self.epoch)
            {
                parent.child_duration_ns = parent.child_duration_ns.saturating_add(inclusive_ns);
            }
            Some((inclusive_ns, exclusive_ns))
        });
        let Some((inclusive_ns, exclusive_ns)) = elapsed_ns else {
            return;
        };
        let counters = &PHASE_COUNTERS[self.phase as usize];
        counters
            .host_invocation_count
            .fetch_add(1, Ordering::Relaxed);
        counters
            .host_inclusive_duration_ns
            .fetch_add(inclusive_ns, Ordering::Relaxed);
        counters
            .host_exclusive_duration_ns
            .fetch_add(exclusive_ns, Ordering::Relaxed);
        counters
            .host_max_inclusive_duration_ns
            .fetch_max(inclusive_ns, Ordering::Relaxed);
    }
}

pub fn runtime_critical_path_span(phase: RuntimeCriticalPathPhase) -> RuntimeCriticalPathSpan {
    let epoch = CRITICAL_PATH_EPOCH.load(Ordering::Relaxed);
    let id = NEXT_SPAN_ID.fetch_add(1, Ordering::Relaxed);
    ACTIVE_SPANS.with(|active_spans| {
        let mut active_spans = active_spans.borrow_mut();
        active_spans.retain(|span| span.epoch == epoch);
        active_spans.push(ActiveSpan {
            id,
            epoch,
            started: Instant::now(),
            child_duration_ns: 0,
        });
    });
    RuntimeCriticalPathSpan {
        id,
        epoch,
        phase,
        _not_send: PhantomData,
    }
}

pub fn runtime_critical_path_device_phase_scope(
    phase: RuntimeCriticalPathPhase,
) -> RuntimeCriticalPathDevicePhaseScope {
    let epoch = CRITICAL_PATH_EPOCH.load(Ordering::Relaxed);
    let id = NEXT_SPAN_ID.fetch_add(1, Ordering::Relaxed);
    ACTIVE_DEVICE_PHASE_OVERRIDES.with(|overrides| {
        let mut overrides = overrides.borrow_mut();
        overrides.retain(|scope| scope.epoch == epoch);
        overrides.push(ActiveDevicePhaseOverride { id, epoch, phase });
    });
    RuntimeCriticalPathDevicePhaseScope {
        id,
        epoch,
        _not_send: PhantomData,
    }
}

pub fn reset_runtime_critical_path_counters() {
    CRITICAL_PATH_EPOCH.fetch_add(1, Ordering::Relaxed);
    for counters in &PHASE_COUNTERS {
        counters.reset();
    }
    ACTIVE_SPANS.with(|active_spans| active_spans.borrow_mut().clear());
    ACTIVE_DEVICE_PHASE_OVERRIDES.with(|overrides| overrides.borrow_mut().clear());
}

pub fn record_runtime_critical_path_device_duration(
    phase: RuntimeCriticalPathPhase,
    duration_ns: u64,
) {
    let epoch = CRITICAL_PATH_EPOCH.load(Ordering::Relaxed);
    let phase = ACTIVE_DEVICE_PHASE_OVERRIDES.with(|overrides| {
        overrides
            .borrow()
            .iter()
            .rev()
            .find(|scope| scope.epoch == epoch)
            .map(|scope| scope.phase)
            .unwrap_or(phase)
    });
    let counters = &PHASE_COUNTERS[phase as usize];
    counters
        .device_timestamp_count
        .fetch_add(1, Ordering::Relaxed);
    counters
        .device_duration_ns
        .fetch_add(duration_ns, Ordering::Relaxed);
    counters
        .device_max_duration_ns
        .fetch_max(duration_ns, Ordering::Relaxed);
}

pub fn runtime_critical_path_report(wall_duration_ns: u64) -> RuntimeCriticalPathReport {
    let phases = RuntimeCriticalPathPhase::ALL
        .iter()
        .copied()
        .map(|phase| {
            let counters = &PHASE_COUNTERS[phase as usize];
            RuntimeCriticalPathPhaseReport {
                phase: phase.as_str().to_string(),
                host_invocation_count: counters.host_invocation_count.load(Ordering::Relaxed),
                host_inclusive_duration_ns: counters
                    .host_inclusive_duration_ns
                    .load(Ordering::Relaxed),
                host_exclusive_duration_ns: counters
                    .host_exclusive_duration_ns
                    .load(Ordering::Relaxed),
                host_max_inclusive_duration_ns: counters
                    .host_max_inclusive_duration_ns
                    .load(Ordering::Relaxed),
                device_timestamp_count: counters.device_timestamp_count.load(Ordering::Relaxed),
                device_duration_ns: counters.device_duration_ns.load(Ordering::Relaxed),
                device_max_duration_ns: counters.device_max_duration_ns.load(Ordering::Relaxed),
                host_exclusive_per_generated_token_ns: None,
                device_per_generated_token_ns: None,
                host_exclusive_per_execution_window_ns: None,
                device_per_execution_window_ns: None,
            }
        })
        .collect::<Vec<_>>();
    let host_exclusive_work_duration_ns = phases.iter().fold(0u64, |total, phase| {
        total.saturating_add(phase.host_exclusive_duration_ns)
    });
    let host_attributed_critical_path_duration_ns =
        host_exclusive_work_duration_ns.min(wall_duration_ns);
    let host_unattributed_duration_ns =
        wall_duration_ns.saturating_sub(host_exclusive_work_duration_ns);
    let host_parallel_overlap_duration_ns =
        host_exclusive_work_duration_ns.saturating_sub(wall_duration_ns);
    let host_coverage_basis_points = if wall_duration_ns == 0 {
        0
    } else {
        u16::try_from(
            host_attributed_critical_path_duration_ns.saturating_mul(10_000) / wall_duration_ns,
        )
        .unwrap_or(10_000)
    };
    let device_timestamp_duration_ns = phases.iter().fold(0u64, |total, phase| {
        total.saturating_add(phase.device_duration_ns)
    });
    RuntimeCriticalPathReport {
        wall_duration_ns,
        host_exclusive_work_duration_ns,
        host_attributed_critical_path_duration_ns,
        host_unattributed_duration_ns,
        host_parallel_overlap_duration_ns,
        host_coverage_basis_points,
        device_timestamp_duration_ns,
        generated_token_count: 0,
        execution_window_count: 0,
        phases,
    }
}

fn per_unit(duration_ns: u64, count: usize) -> Option<u64> {
    let count = u64::try_from(count).ok()?;
    (count > 0).then(|| duration_ns / count)
}

fn elapsed_nanos_u64(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos())
        .unwrap_or(u64::MAX)
        .max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn phase<'a>(
        report: &'a RuntimeCriticalPathReport,
        phase: RuntimeCriticalPathPhase,
    ) -> &'a RuntimeCriticalPathPhaseReport {
        report
            .phases
            .iter()
            .find(|candidate| candidate.phase == phase.as_str())
            .expect("phase report")
    }

    #[test]
    fn nested_spans_do_not_double_count_host_work() {
        reset_runtime_critical_path_counters();
        {
            let _outer = runtime_critical_path_span(RuntimeCriticalPathPhase::Protocol);
            std::thread::sleep(Duration::from_millis(2));
            {
                let _inner = runtime_critical_path_span(RuntimeCriticalPathPhase::QueueSubmission);
                std::thread::sleep(Duration::from_millis(2));
            }
        }
        let report = runtime_critical_path_report(20_000_000);
        let outer = phase(&report, RuntimeCriticalPathPhase::Protocol);
        let inner = phase(&report, RuntimeCriticalPathPhase::QueueSubmission);
        assert!(outer.host_inclusive_duration_ns >= inner.host_inclusive_duration_ns);
        assert_eq!(
            report.host_exclusive_work_duration_ns,
            outer
                .host_exclusive_duration_ns
                .saturating_add(inner.host_exclusive_duration_ns)
        );
        assert!(
            report.host_exclusive_work_duration_ns
                <= outer.host_inclusive_duration_ns.saturating_add(1_000_000)
        );
    }

    #[test]
    fn reset_discards_a_span_that_crosses_the_measurement_boundary() {
        reset_runtime_critical_path_counters();
        let stale = runtime_critical_path_span(RuntimeCriticalPathPhase::Protocol);
        reset_runtime_critical_path_counters();
        drop(stale);
        let report = runtime_critical_path_report(1_000);
        assert_eq!(
            phase(&report, RuntimeCriticalPathPhase::Protocol).host_invocation_count,
            0
        );
        assert_eq!(report.host_unattributed_duration_ns, 1_000);
    }

    #[test]
    fn device_timestamps_are_not_counted_as_host_critical_path_time() {
        reset_runtime_critical_path_counters();
        record_runtime_critical_path_device_duration(
            RuntimeCriticalPathPhase::ExpertCompute,
            90_000,
        );
        let report = runtime_critical_path_report(10_000);
        assert_eq!(report.host_exclusive_work_duration_ns, 0);
        assert_eq!(report.host_unattributed_duration_ns, 10_000);
        assert_eq!(report.device_timestamp_duration_ns, 90_000);
        assert_eq!(
            phase(&report, RuntimeCriticalPathPhase::ExpertCompute).device_duration_ns,
            90_000
        );
    }

    #[test]
    fn device_phase_scopes_override_structural_attribution_and_restore_when_nested() {
        reset_runtime_critical_path_counters();
        record_runtime_critical_path_device_duration(RuntimeCriticalPathPhase::ExpertCompute, 10);
        {
            let _commit =
                runtime_critical_path_device_phase_scope(RuntimeCriticalPathPhase::StateCommit);
            record_runtime_critical_path_device_duration(
                RuntimeCriticalPathPhase::ExpertCompute,
                20,
            );
            {
                let _verification = runtime_critical_path_device_phase_scope(
                    RuntimeCriticalPathPhase::SpeculativeVerification,
                );
                record_runtime_critical_path_device_duration(
                    RuntimeCriticalPathPhase::ExpertCompute,
                    30,
                );
            }
            record_runtime_critical_path_device_duration(
                RuntimeCriticalPathPhase::AttentionRead,
                40,
            );
        }
        record_runtime_critical_path_device_duration(RuntimeCriticalPathPhase::ExpertCompute, 50);

        let report = runtime_critical_path_report(1);
        assert_eq!(
            phase(&report, RuntimeCriticalPathPhase::ExpertCompute).device_duration_ns,
            60
        );
        assert_eq!(
            phase(&report, RuntimeCriticalPathPhase::StateCommit).device_duration_ns,
            60
        );
        assert_eq!(
            phase(&report, RuntimeCriticalPathPhase::SpeculativeVerification).device_duration_ns,
            30
        );
        assert_eq!(
            phase(&report, RuntimeCriticalPathPhase::AttentionRead).device_duration_ns,
            0
        );
    }

    #[test]
    fn reset_discards_device_phase_scope_from_the_previous_measurement() {
        reset_runtime_critical_path_counters();
        let stale = runtime_critical_path_device_phase_scope(RuntimeCriticalPathPhase::StateCommit);
        reset_runtime_critical_path_counters();
        record_runtime_critical_path_device_duration(RuntimeCriticalPathPhase::ExpertCompute, 70);
        drop(stale);

        let report = runtime_critical_path_report(1);
        assert_eq!(
            phase(&report, RuntimeCriticalPathPhase::ExpertCompute).device_duration_ns,
            70
        );
        assert_eq!(
            phase(&report, RuntimeCriticalPathPhase::StateCommit).device_duration_ns,
            0
        );
    }

    #[test]
    fn concurrent_host_work_is_reported_as_overlap_instead_of_negative_unattributed_time() {
        reset_runtime_critical_path_counters();
        {
            let _span = runtime_critical_path_span(RuntimeCriticalPathPhase::Protocol);
            std::thread::sleep(Duration::from_millis(1));
        }
        let report = runtime_critical_path_report(1);
        assert_eq!(report.host_unattributed_duration_ns, 0);
        assert!(report.host_parallel_overlap_duration_ns > 0);
        assert_eq!(report.host_coverage_basis_points, 10_000);
    }

    #[test]
    fn normalization_reports_per_token_and_per_window_without_dividing_by_zero() {
        reset_runtime_critical_path_counters();
        record_runtime_critical_path_device_duration(RuntimeCriticalPathPhase::Sampling, 120);
        let unnormalized = runtime_critical_path_report(1_000).with_normalization(0, 0);
        let empty = phase(&unnormalized, RuntimeCriticalPathPhase::Sampling);
        assert_eq!(empty.device_per_generated_token_ns, None);
        assert_eq!(empty.device_per_execution_window_ns, None);

        let normalized = runtime_critical_path_report(1_000).with_normalization(4, 3);
        let sampling = phase(&normalized, RuntimeCriticalPathPhase::Sampling);
        assert_eq!(sampling.device_per_generated_token_ns, Some(30));
        assert_eq!(sampling.device_per_execution_window_ns, Some(40));
    }
}
