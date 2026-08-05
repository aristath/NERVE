# TODO

## Goal

Compile and run `/mnt/models/models/deepseek-v4/flash-0731/safetensors` as a
daily-driver NERVE model. Preserve source behavior, keep sparse routed experts
independently demand-resident, expose runtime per-component representation
selection, and optimize real multi-turn agentic inference past 30 tok/s and
toward 50 tok/s without regressing supported Qwen models.

## Work queue

1. Complete capability-driven per-component representation selection. Preserve
   a native source representation when it is optimal on the selected device;
   otherwise measure only structurally valid INT4 2:4, dense INT4, FP8 2:4,
   dense FP8/INT8, and FP16 candidates. Promote a candidate only after hardware
   measurement and behavioral-equivalence checks. Keep representation and
   placement as runtime choices, never model-family switches. SINT8 matrix
   calibration alone is insufficient: the previous inline MXFP4-to-SINT8
   reconstruction lost to native MXFP4 because activation quantization and
   packed-weight reconstruction were not amortized. Hardware capability is
   candidate discovery, not promotion: the rejected schema-v4 experiment
   selected expanded FP8 merely because every target exposed the required
   features. It doubled sparse-expert residency, reduced the first DeepSeek
   turn to 0.719 tok/s, and caused 207.3 GB of uploads for 143 decode tokens.
   The baseline compiler now keeps native MXFP4; attach any derived resident
   representation only through measured optimizer evidence that includes the
   whole routed working set, footprint, reload cost, and device headroom. The
   package/runtime contract can now materialize one explicitly selected MXFP4
   to FP8 derivation without changing source artifacts. Complete the remaining
   alternative-set contract, device-local measurement evidence, and runtime
   choice so native and derived representations can coexist and be selected per
   component on the component's actual target device.

2. Extend the generic quality gate with a representative tool-call round trip
   and a long-stream continuity case. Use package-owned chat behavior, official
   thinking and sampling defaults, the 65,536-token output allowance, agentic
   context, complete-conversation warmup, coherent final answers, turn recall,
   and exact teardown. Exercise both successful canonical commits and malformed
   output rollback without model-name branches.

3. Raise DeepSeek's accepted decode rate past 30 tok/s and continue until no
   material bottleneck remains. The fresh native-MXFP4 package and the current
   demand-paged runtime now complete the authoritative thinking-enabled product
   gate at 128K context and a 65,536-token output allowance. The gate discards
   one complete conversation, measures a second conversation without unloading
   the model, preserves Greece/Athens turn recall, and releases every acquired
   allocation. Its current truth baseline is only **2.7258 decode tok/s** and
   **5.1398 prefill tok/s** across 3,055 measured generated tokens. After the
   truth-set `hi`, the five measured turns reread 971.79 GB, evicted 969.25 GB,
   performed 72,423 reloads, and blocked 486.21 seconds on residency. No derived
   representation was selected. Fix the following in measured order:

   - Replace isolated per-device-store host-visible budgets with a shared,
     borrowable physical residency cache. Preserve contiguous component
     ownership while allowing unused host/device capacity to satisfy a hot
     store, enforce one global hard bound, and keep an evicted expert in a
     reusable host tier instead of rereading it from disk. Make cache admission,
     promotion, demotion, and eviction route-aware and evidence-driven.
   - Stop speculative verification from multiplying sparse-expert traffic. The
     current five-token target window accepts roughly 20-34% and executes far
     more routed branches than it commits. Measure scalar and each useful block
     width on the same resident model, then select speculation only when useful
     committed tokens per complete cycle win. Compile draft generation, target
     projection, comparison, state selection, commit, and draft catch-up into
     one device-resident transaction with no host wait until a real residency
     miss or completed emitted block.
   - Optimize the independently material MXFP4 expert path. Retain the source's
     packed 4-bit payload, amortize or fuse unpacking and activation conversion,
     and benchmark exact native MXFP4, dense/structured INT4, FP8, and INT8
     alternatives on each actual target device. Include resident footprint and
     reload traffic in promotion evidence, not kernel time alone.
   - Add device timestamps around routing, residency, expert projection,
     speculative target verification, queue submission, and host synchronization
     so remaining kernel and orchestration costs are separable after churn is
     removed. Reduce the current tens of thousands of copy submissions and
     waits without weakening bounded execution quanta or driver-timeout safety.
   - Restore repeated-process determinism before accepting throughput changes.
     Audit radix/top-k tie order, sampler RNG consumption, accepted-prefix
     commit, rollback, reset state, and expert selection. Require equality for
     committed tokens, routed experts, accepted prefixes, and state digests.

4. Complete heterogeneous cost-based auto-placement. The runtime now admits
   demand-paged models against fixed residency plus one full selector wave,
   chooses the smallest capacity-safe device prefix when complete residency
   fits, and proportionally assigns a larger virtual model across every
   compatible cache when it does not. It respects partial VRAM reservations,
   keeps contiguous component segments, excludes integrated display GPUs, and
   leaves explicit wiring untouched. Complete candidate ranking and spill
   across capability classes using measured execution and transfer costs rather
   than model or vendor names. Select each component representation for its
   target device, reject incompatible boundaries before allocation, and permit
   a model that exhausts the AMD group to continue contiguously onto a
   compatible discrete Intel GPU or CPU without recompilation.

   Make temporal and speculative batch execution device-segment aware. A valid
   component path may revisit one physical device after traversing others, but
   the current one-slice-per-device batch runner cannot distinguish the early
   and late segments and fails only after scalar decoding begins. Represent and
   execute ordered component segments, add a regression graph such as
   `gpu0 -> gpu1 -> gpu0`, and preserve batched verification without silently
   disabling speculation. Until that is complete, product performance gates
   must use a non-revisiting contiguous device pipeline.

5. Before every runtime-performance commit, run Qwen3.6-35B-A3B and Qwen3.5-9B
   quality/performance gates sequentially on equivalent healthy AMD placement.
   Restrict discovery to the AMD Vulkan ICD and bind PCI-derived UUIDs so an
   added Intel or NVIDIA adapter cannot silently change the comparison. The
   current thinking-enabled gates each discard one complete in-process warmup
   conversation, reset model state without unloading, pass correct Greece
   recall, and release their exact recorded reservations. On the current
   pressure-safe runtime, Qwen3.6-35B-A3B reaches 96.71 decode tok/s and 119.79
   prefill tok/s with a contiguous two-AMD split. Qwen3.5-9B reaches 53.65
   decode tok/s and 133.15 prefill tok/s on one AMD using the official
   temperature 1.0, top-k 20, top-p 0.95, min-p 0, presence-penalty 1.5,
   repetition-penalty 1.0 thinking profile. Keep repetition,
   structured-protocol, conversation, and teardown checks active so throughput
   alone cannot pass.

6. Perform a final adversarial review against `CONCEPT.md`: compiled artifacts
   remain self-contained and model-specific, while compiler discovery, runtime
   operators, graph wiring, placement, representation, residency, and stream
   transactions remain capability-driven and reusable by unseen models. Finish
   with an empty TODO, a clean worktree, and every milestone committed and
   pushed.
