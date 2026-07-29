# Scheduler Backpressure and Checkpoint Resume Acceptance

Milestone 11 is complete.

## Execution architecture

NERVE now has one generic, bounded residency-backpressure coordinator between
the GPU miss boundary, per-device residency directory, load pipeline, physical
checkpoint cursor, and stream executor.

An admitted checkpoint carries:

- the existing runtime activation identity;
- stream, physical-device, and checkpoint identities;
- the physical checkpoint cursor;
- an opaque continuation owned by the real executor; and
- execution leases for every selected atomic group.

The continuation is moved into the coordinator and returned unchanged. The
coordinator does not rebuild an activation, reserve replacement transient
state, advance recurrent or attention state, consume random state, or replay
selection. The original runtime activation therefore remains in flight while
its physical work is paused. After the final required group is atomically
published, the existing checkpoint cursor resumes at selected computation and
then its continuation dispatch.

Ready activations, failed activations, load commands, outstanding loads, groups
per activation, and scheduler-owned activations are bounded. One activation per
stream can be owned at a time, preserving stream order. The ready queue is FIFO:
warm work can pass blocked cold work, while a resumed cold activation is placed
behind only the finite work already admitted. Consumers may also take the first
ready activation for a particular device, so a miss on one device does not
block another device's queue.

## Atomic admission and single flight

The per-device residency manager now admits a sorted group batch while holding
one state lock. Before changing any directory entry it validates every
descriptor, identity, failed state, aggregate byte requirement, operation-id
range, and caller-provided new-load allowance. A capacity or backpressure
failure therefore leaves no partial batch loading.

For every genuinely absent group the coordinator creates exactly one movable
load command. Later activations join the manager's existing outcome rather than
launching another read or upload. Completion is event-driven:

- publishing a command atomically installs the complete group and sends one
  completion event;
- failing or dropping a command sends one deterministic failure event;
- joined waiters are tested non-blockingly after publication; and
- a load started by another compatible graph through the same per-device
  manager is also observed non-blockingly.

Cancellation removes the activation and returns its unchanged continuation for
the stream scheduler to roll back or close. It does not cancel a group still
needed by another activation. Under demand-retained policy, a resource selected
by real execution may finish loading and remain owned by the graph even if that
individual activation is cancelled.

The implementation contains no model name, family, layer, expert, or MoE
branch.

## Adversarial review

The review found and fixed four defects before acceptance:

1. The original single-group residency request API could not reserve several
   selected groups atomically. A later capacity failure could otherwise leave
   an earlier group loading. Batched admission now preflights the complete set.
2. Conservatively reserving one load slot for every selected group would reject
   a fully resident hit whenever the cold-load queue was full. The manager now
   applies the scheduler allowance only to groups that are actually absent.
3. A compatible load initiated outside this coordinator could wake the
   per-device manager but had no scheduler completion event. Joined waiters now
   have a non-blocking publication probe, so compatible graph instances share
   correctly.
4. Ordered maps could wake activations by numeric activation id rather than
   admission order. Ready and failure transitions now sort by the explicit
   monotonic admission sequence. Reverse-id tests prove the distinction.

## Sequential proof

These exact tests passed individually with `CARGO_BUILD_JOBS=1` and
`-- --exact --test-threads=1`:

- `residency_backpressure_interleaves_hits_and_shared_misses_without_replay`
- `residency_backpressure_cancellation_preserves_state_and_does_not_cancel_shared_work`
- `residency_backpressure_load_failure_wakes_every_dependent_activation_in_order`
- `residency_pause_keeps_the_runtime_activation_and_transient_reservation_in_flight`
- `residency_backpressure_observes_a_shared_manager_load_started_elsewhere`
- `residency_backpressure_resumes_only_after_every_group_is_atomically_published`
- `residency_batch_capacity_failure_is_atomic_and_scheduler_queues_are_bounded`
- `residency_load_backpressure_does_not_reject_a_resident_hit`
- `per_device_residency_single_flight_shares_one_atomic_publication`
- `per_device_residency_rejects_capacity_before_atomic_loading_and_rolls_back`

The tests deterministically cover resident hits, one- and multi-group misses,
out-of-order group completion, shared misses, external single flights,
cancellation, load failure, bounded queues, warm-work progress, admission-order
wakeups, exact checkpoint traces, and preservation of a real runtime
activation's transient-state reservation across its pause.

`cargo fmt --check`, `cargo check --all-targets --features "vulkan tokenizers"`,
and `git diff --check` passed. Production files remain below the repository's
concern-size thresholds: the coordinator is 1,051 lines and the per-device
manager is 1,446 lines.

## Canonical conversation regression gate

Milestone 11 adds the generic scheduling machinery; milestone 12 connects
Qwen's compiled routed-resource contract to it. The current Qwen package still
executes its eager parameter path, so this run is a whole-engine regression and
quality gate rather than a claim that demand loading is already active.

The unchanged canonical two-AMD, thinking-enabled Qwen3.6-35B-A3B FP8
conversation used a 131,072-token context, 65,536 maximum new tokens, two MTP
draft tokens, seed 0, the discarded `hi` warmup, and all five measured turns in
one continuously resident process.

- setup: 35,076.778 ms
- mean decode: **41.7388 tok/s**
- mean prefill: **63.8690 tok/s**
- warmup decode: 45.732 tok/s
- measured decode: 46.534, 45.092, 35.528, 39.590, 41.950 tok/s
- milestone 10 mean decode: 41.1722 tok/s
- decode delta: +1.38%
- throughput floor: passed (30 tok/s)
- quality gate: passed, including thinking and cross-turn Greece recall
- report:
  `/tmp/nerve-m11-backpressure-mtp2-v1/report.json`
- transcript:
  `/tmp/nerve-m11-backpressure-mtp2-v1/conversation-seed-0.log`
- report SHA-256:
  `aed8c20229e0c4dd3309ccdc2b2f467539af60b747d4ff70b57ce5f54ece00bf`
- transcript SHA-256:
  `6559d9fafcd6ac8af816690a2884d6f2af65857810d537778ef25632f7d19c2a`

Before and after execution, both used AMD GPUs were at the exact
59,973,632-byte / 0%-busy idle baseline. NVIDIA was neither enumerated nor
used.

The next remaining milestone is to compile Qwen's routed expert bundles as
generic atomic groups and connect its real selector/gate/miss path to this
coordinator without introducing any Qwen-specific runtime behavior.
