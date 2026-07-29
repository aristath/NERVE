# GPU-Resident Hit Path Acceptance

Milestone 10 is complete.

## Execution architecture

NERVE now has one generic Vulkan residency gate between a selector and its
physically dependent dispatches. The gate consumes the selector's existing
device-resident output directly. It does not copy route decisions to the host.

Its immutable GPU configuration describes:

- the selection word's index bit field;
- one selectable-resource ordinal for each selector result;
- the complete stable address-table slot set for each atomic resource;
- bounded resolved-address and missing-request capacities; and
- the original workgroup dimensions of every dependent dispatch.

The configuration is data, not shader specialization. One shader therefore
serves different selectors, resource counts, atomic-group widths, address-slot
layouts, and downstream dispatch shapes without model-family branches.

For every unique selected resource, the gate checks all member records in the
stable GPU address table. A hit requires a resident bit, nonzero stable address,
nonzero byte count, and nonzero publication generation for every member. Only
then does it publish the complete resolved-address records and restore the
dependent `VkDispatchIndirectCommand` values. A partial group or invalid
selection zeros all dependent dispatch dimensions.

The gate's descriptor set owns strong references to its selector,
configuration, address-table, resolved-address, miss-queue, and indirect
buffers. Downstream work consumes the GPU-written indirect table directly.
There is no resident-hit host decision, queue drain, device-wide wait, or
selection readback in this path.

## Miss notification

Misses append compact `(checkpoint_tag, resource_index)` records to a bounded
single-consumer ring. The published and consumed counters are wrapping `u32`
tickets; pending iteration is explicitly wrapping-safe. A notification epoch
advances once only when a gate execution observed at least one real miss.
Warm hits do not modify that epoch and append nothing.

The host-visible queue is the scheduler boundary, not part of the warm
decision. Its snapshot and acknowledgement methods perform no Vulkan wait.
The scheduler must call them after its existing completion synchronization;
milestone 11 connects this queue to blocked activations, single-flight loads,
fair scheduling, and checkpoint resume.

## Adversarial review

The review found and fixed four defects before acceptance:

1. The initial Vulkan test resolved shader paths from the wrong root and could
   pass by skipping. With an explicit test device, the final tests require a
   working shader compiler and real execution.
2. The general sequence recorder issued indirect dispatches but omitted the
   indirect buffer from dependency construction unless the downstream shader
   also bound it. GPU-generated commands could be read without a
   shader-write-to-indirect-read barrier. Every indirect command buffer is now
   an explicit sequence access, independent of shader descriptors.
3. The first benchmark arena omitted absolute-address alignment slack. The
   corrected fixture reserves the allocator's real worst-case requirement.
4. Initial host queue iteration did not survive a `u32` ticket wrap. It now
   derives every ticket with wrapping addition from the bounded pending count.

The production gate, shader, and sequence fix contain no Qwen, MoE, expert,
layer, or model-family branch.

## Sequential proof

These exact tests passed individually with `CARGO_BUILD_JOBS=1` and
`-- --exact --test-threads=1`:

- `gpu_residency_gate_contract_rejects_unrepresentable_or_unbounded_work`
- `gpu_residency_gate_keeps_hits_on_device_and_publishes_only_real_misses`
- `gpu_residency_gate_warm_path_is_measured_against_eager_dispatch`

The functional Vulkan test proves:

- two selected resources resolve to their exact stable device addresses;
- a fully resident selection executes dependent work indirectly;
- the warm path leaves the notification epoch and miss queue unchanged;
- clearing one publication suppresses the dependent dispatch;
- that miss publishes exactly one compact request;
- acknowledgement consumes the bounded request; and
- republishing the resource restores execution without another notification.

The test also exposed and therefore covers the general indirect-command
visibility regression. `cargo fmt --check`,
`cargo check --all-targets --features "vulkan tokenizers"`, and
`git diff --check` passed. The new production and test files are 496 and 520
lines, and the touched sequence recorder remains 1,041 lines.

## Matched warm-path microbenchmark

The release-mode benchmark compares identical 16 MiB stable-address work:

- eager: the prepared dispatch executes directly;
- fully warm demand-retained: the GPU gate resolves the already-resident
  address and enables the same dispatch indirectly.

Each path used one discarded warmup and two measured executions. Shader
compilation, allocation, upload, correctness comparison, measurement, and
teardown completed in 1.006 seconds.

- eager durations: 35,200 ns and 34,560 ns
- eager mean: 34,880 ns
- warm demand durations: 38,560 ns and 38,760 ns
- warm demand mean: 38,660 ns
- absolute gate cost: 3,780 ns
- demand/eager ratio: 1.108372

At 41 routed checkpoints, the measured gate cost totals about 0.155 ms, below
0.7% of the current approximately 23.7 ms/token end-to-end budget. A separate
dispatch remains materially preferable to a host round trip and preserves a
single reusable gate. A future optimizer may fuse the gate into a compatible
selector when measured hardware evidence proves that representation better.

## Canonical conversation benchmark

The final canonical two-AMD, thinking-enabled Qwen3.6-35B-A3B FP8 conversation
used a 131,072-token context, 65,536 maximum new tokens, two MTP draft tokens,
seed 0, the discarded `hi` warmup, and all five measured turns in one
continuously resident process.

- mean decode: **41.1722 tok/s**
- mean prefill: **89.0018 tok/s**
- warmup decode: 43.015 tok/s
- measured decode: 46.184, 42.589, 36.423, 40.540, 40.125 tok/s
- milestone 9 mean decode: 42.1928 tok/s
- decode delta: -2.42%
- throughput floor: passed (30 tok/s)
- quality gate: passed, including thinking and cross-turn recall
- report:
  `/tmp/nerve-m10-gpu-residency-gate-mtp2-v1/report.json`
- transcript:
  `/tmp/nerve-m10-gpu-residency-gate-mtp2-v1/conversation-seed-0.log`
- report SHA-256:
  `c451141c321787290ee7fd28990511d3c76902e59d775a0db0eb414ede480436`
- transcript SHA-256:
  `efdb2715345959ba594b904d0aa6c10a4b57ed7a3a7f7952f3391c93c33e3f9e`

The conversational variation did not expose a material whole-engine
regression, and the result remains 37.24% above the required throughput floor.
Before and after every Vulkan execution, the used AMD GPUs were at the exact
59,973,632-byte / 0%-busy idle baseline. NVIDIA was neither enumerated nor
used.

The next remaining work is milestone 11: make miss records schedulable
backpressure, deduplicate their loads, and resume blocked activations at the
physical checkpoint without replay.
