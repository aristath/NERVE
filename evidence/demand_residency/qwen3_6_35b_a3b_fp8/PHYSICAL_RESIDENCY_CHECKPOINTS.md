# Physical Residency Checkpoint Acceptance

Milestone 9 is complete.

## Physical schedule

NERVE now lowers the compiled residency contract into a physical schedule for
every placed target device slice. This does not fragment or modify the editable
semantic graph. The schedule is derived from:

- compiled selectors and checkpoint boundaries;
- generic parameter-to-resource bindings;
- the prepared physical dispatch order; and
- runtime placement.

Each checkpoint records:

- the selector dispatch;
- the complete selected-computation dispatch range, including any inseparable
  physical work between the first and last selected parameter consumer;
- the immediate selected-result continuation when one exists; and
- the selector identities needed to map selected indices to atomic groups.

For Qwen3.6-35B-A3B, the selected computation is the gate/up and down expert
work and the immediate selected-result continuation is `moe_reduce`. The
runtime does not identify Qwen, MoE, experts, layers, or these operation names
when constructing the schedule. A structurally different terminal selectable
projection is covered by the same implementation and legitimately has no
selected-result continuation.

The placed package validates aggregate ownership after all device slices are
planned. Every target checkpoint must occur on exactly one owner slice.
Missing or duplicate checkpoint ownership fails package construction rather
than silently disabling demand residency.

## Selection and pause/resume contract

One selector can supply multiple selected indices, so top-k selection resolves
to a sorted, deduplicated set of complete atomic group identities. Both
concrete group tables and partition-template identities use the same ABI.
Out-of-range, duplicate, missing-selector, and cross-checkpoint selections fail
deterministically.

The physical activation cursor executes four responsibilities:

1. selection;
2. atomic-group availability;
3. selected computation; and
4. selected-result continuation, such as MoE reduction, when present.

On a miss, the cursor records selection and availability, suppresses every
dependent dispatch, and pauses with an explicit
`SelectedComputation` resume responsibility. Resume is accepted only after all
missing atomic groups are published. Per-resource or partial-group visibility
cannot satisfy this contract. A successful resume executes selected
computation directly; selection and availability are not replayed.

Matched eager and miss-then-resume traces contain identical selected group
identities, dispatch ranges, and physical responsibilities. The demand trace
contains exactly one selection responsibility even after resume.

## Adversarial review and sequential tests

The review found one initial gap: an individual device schedule must skip
components owned elsewhere, but that made it possible for every slice to skip
a checkpoint without an aggregate failure. Package-level exact-ownership
validation was added before acceptance.

The following exact tests passed individually with `CARGO_BUILD_JOBS=1` and
`-- --exact --test-threads=1`:

- `physical_residency_schedule_derives_generic_selected_execution_boundaries`
- `physical_residency_checkpoint_resolves_topk_indices_to_atomic_groups`
- `demand_checkpoint_resumes_selected_work_without_replaying_selection`
- `physical_checkpoint_supports_non_expert_terminal_selected_resources`
- `physical_residency_coverage_rejects_missing_or_duplicate_device_ownership`

They prove range derivation, top-k identity resolution, suppression before
publication, rejection of partial publication, direct checkpoint resume,
trace equivalence, non-MoE generality, and exact multi-device ownership.

`cargo fmt --check`, `cargo check --all-targets --features "vulkan tokenizers"`,
and `git diff --check` passed. The new implementation and test files are 561
and 288 lines respectively, below the repository review threshold.

## Matched conversation benchmark

The final canonical two-AMD, thinking-enabled Qwen3.6-35B-A3B FP8 conversation
used a 131,072-token context, 65,536 maximum new tokens, two MTP draft tokens,
seed 0, the discarded `hi` warmup, and all five measured turns in one
continuously resident process.

Results:

- mean decode: **42.1928 tok/s**
- mean prefill: **67.0376 tok/s**
- warmup decode: 43.584 tok/s
- measured decode rates: 46.697, 43.225, 38.876, 41.093, 41.073 tok/s
- milestone 8 mean decode: 41.4644 tok/s
- decode delta: +1.76%
- throughput floor: passed (30 tok/s)
- quality gate: passed, including thinking, Qwen identity, Athens, a relevant
  qualified Corinth answer, the knowledge-cutoff response, and cross-turn
  Greece recall
- report:
  `/tmp/nerve-m9-physical-checkpoint-mtp2-v2/report.json`
- transcript:
  `/tmp/nerve-m9-physical-checkpoint-mtp2-v2/conversation-seed-0.log`
- report SHA-256:
  `2e081d43ed7c3fc392aebb261b934aa054b1db197ee740bed7e085f0dbe7e179`
- transcript SHA-256:
  `fa404001f29425f965a610cfd7f354a3ae75dc5b0ac3506ad5a1facf03643f1a`

Immediately before and after the final benchmark, both AMD GPUs were at the
exact idle baseline of 59,973,632 bytes VRAM used and 0% busy. NVIDIA was
neither enumerated nor used.

No new performance bottleneck was introduced. The next required work is the
existing milestone 10: make the warm resident availability check and address
resolution execute entirely on the GPU, with host notification only on a real
miss.
