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

   Sampling provenance is part of this contract. DeepSeek's compiled package
   owns its source `generation_config.json` defaults (temperature 1.0, top-p
   1.0, no top-k or penalties); never reuse Qwen's top-k/presence-penalty
   profile for it. A rejected gate using the Qwen overrides entered a repeated
   cutoff/identity reasoning loop, while the package-owned profile completed
   coherently with correct Greece/Athens recall.

3. Raise DeepSeek's accepted decode rate past 30 tok/s and continue until no
   material bottleneck remains. The fresh native-MXFP4 package and the current
   demand-paged runtime now complete the authoritative thinking-enabled product
   gate at 128K context and a 65,536-token output allowance. The gate discards
   one complete conversation, measures a second conversation without unloading
   the model, preserves Greece/Athens turn recall, and releases every acquired
   allocation. Its current official-profile truth baseline is **8.0818 decode
   tok/s** and **8.4564 prefill tok/s** before the compact-MXFP4 remap. The
   exact integer remap keeps the source's packed 4-bit residency and replaces
   per-value floating reconstruction with direct E2M1-to-E4M3 bit mapping. It
   reduced the real-geometry scalar expert pair from 1.03344 ms to 0.72736 ms
   (29.6%) while exhaustive coverage proved all 16 nibble codes unchanged. The
   six-token native batch moved only from 2.45352 ms to 2.41128 ms. In the full
   model, identical truth token counts completed at **7.9532 decode tok/s** and
   **8.5312 prefill tok/s**, statistically flat against the earlier run. This
   proves the optimized compact math is being masked by orchestration overhead,
   not that unpacking remains the primary end-to-end limit. The shared,
   borrowable physical host cache replaced
   isolated per-store budgets and enforces one measured global hard bound. It
   reduced the measured truth conversation from 108.68 GB and 7,866 source
   reloads to 4.13 GB and 6 reloads, with one 80.2 MB eviction and 6.59 seconds
   of residency blocking. Cache reservations are transactional, reclaim full
   allocation cohorts, and use a global-before-store lock order so simultaneous
   stores cannot deadlock. Residency is no longer the dominant steady-state
   bottleneck. No derived representation was selected. Fix the following in
   measured order:
   - Stop speculative verification from multiplying sparse-expert traffic. The
     current five-token target window accepts roughly 20-34% and executes far
     more routed branches than it commits. Measure scalar and each useful block
     width on the same resident model, then select speculation only when useful
     committed tokens per complete cycle win. The runtime now calibrates scalar,
     resident-loop (when available), and useful speculative widths using complete
     cycle elapsed time; mode-specific residency loads are charged instead of
     discarded. On this sparse package it selects scalar and improves decode by
     20.7% over the previous 6.6966 baseline, but remains far below the 30 tok/s
     floor. Before treating modes as interchangeable, certify committed-token,
     routed-expert, sampler, and state-digest equivalence from identical
     checkpoints; scalar and batched verification currently follow different
     deterministic token paths. Demand-resident temporal target execution now
     mounts input embedding, ordered device segments, and device-local edge
     copies into one queue batch with completion attached only to the terminal
     segment. On an identical warmup turn this reduced individual sequence
     submits from 9,340 to 9,184 and copy submits from 7,719 to 7,447 while
     preserving exact token, selection, and state digests. The complete truth
     conversation remained statistically flat at **8.0350 decode tok/s** and
     **8.2372 prefill tok/s**. An analogous all-device scalar batch was rejected:
     sparse expert misses forced suffix retries, producing **7.8518 decode
     tok/s** and **8.1042 prefill tok/s**. Do not revive that host-retry design.
     The remaining transaction must keep demand resolution on device, then fuse
     draft generation, target projection, comparison, state selection, commit,
     and draft catch-up without a host wait until a true external residency miss
     or completed emitted block.
     DeepSeek-V4-Flash-0731 ships its DSpark module inside the target checkpoint;
     this is the intended fast path, not an optional external draft model. The
     compiler discovers its structure without a model-name branch as a
     three-stage `parallel_backbone_markov` decoder: target taps from layers
     40-42, a semi-autoregressive backbone, a rank-256 Markov head, and
     confidence-prefix verification. The source `dspark_block_size = 5` is now
     retained as training/checkpoint provenance while the compiled execution
     capacity follows the official seven-token runtime recommendation. Fresh
     b7 input, backbone, sequential Markov, confidence, and output shaders are
     compiled; widths 1-7 remain legal runtime views of that capacity. Make the
     package-owned seven-token recommendation activate automatically when an
     attached decoder is present, while preserving an explicit runtime disable.
     Finish reference-equivalence evidence for proposal, confidence,
     verification, commit, and rollback behavior.

     The first five-device demand-paged run proves the b7 decoder executes, but
     not yet profitably: calibration measured 10 DSpark cycles, proposed 34
     draft tokens, accepted 14 (41.18%), and selected scalar afterward. Width 4
     was best at 3.734 useful tok/s; width 7 reached only 1.514 useful tok/s.
     Fuse the complete DSpark proposal, target verification, prefix comparison,
     state selection, canonical commit, and draft catch-up transaction on the
     GPU so seven-token execution no longer expands into thousands of host
     submissions and waits. Re-run one complete warmup conversation followed
     by the untouched truth conversation; DSpark is complete only when it is
     behaviorally correct and wins the selector on measured useful committed
     tokens per second.
     A fresh complete-conversation warmup followed by the canonical truth
     conversation measured **8.1878 decode tok/s** and **8.4208 prefill tok/s**,
     statistically flat against the accepted 8.0350/8.2372 run. The measured
     1,485-token truth turn exposed the actual scalar path: demand-loaded token
     execution deliberately bypasses deferred component batching and uses the
     synchronous stream-tick executor, while demand-loaded packages are
     categorically excluded from the resident feedback loop. That turn issued
     27,000 resident-sequence submissions, waited on 28,919 sequence fences,
     issued 8,260 copy submissions, and waited on 8,474 copies. Ordinary
     component batches accounted for only 220.477 ms of device time and
     404.528 ms of calibrated-quantum wall time within 191,078.849 ms of decode.
     Build a demand-aware resident feedback transaction: record multiple causal
     ticks with GPU residency gates, stop the graph at the first real miss,
     materialize only the missing resource cohort, resume from that checkpoint,
     and otherwise wait only at the emitted-window boundary. It must retain
     bounded watchdog quanta and exact stop-token, sampler, state, and routed-
     expert semantics. Do not disguise the synchronous scalar path behind a
     larger host loop or revive the rejected suffix-retry batch.
   - Optimize the independently material MXFP4 expert path. Retain the source's
     packed 4-bit payload, amortize or fuse unpacking and activation conversion,
     and benchmark exact native MXFP4, dense/structured INT4, FP8, and INT8
     alternatives on each actual target device. Include resident footprint and
     reload traffic in promotion evidence, not kernel time alone. Direct integer
     E2M1-to-E4M3 remapping is now exact and materially improves the scalar
     kernel without expanding expert residency; retain it. The remaining work is
     device-local alternative measurement and whole-working-set promotion, not
     another inline scalar reconstruction variant.
   - Add device timestamps around routing, residency, expert projection,
     speculative target verification, queue submission, and host synchronization
     so remaining kernel and orchestration costs are separable after churn is
     removed. Reduce the current tens of thousands of copy submissions and
     waits without weakening bounded execution quanta or driver-timeout safety.
     The scalar real-geometry microbenchmark now measures 0.72736 ms of device
     work but 5.58972 ms through individually submitted host calls, a 7.69x
     host/device gap. A representative 1,597-token truth turn recorded 28,167
     sequence submits, 30,163 fence waits, 8,944 copy submits, and 9,313 copy
     waits. Collapse cached-hit execution into ordered per-device component
     segments with timeline synchronization only at real segment boundaries;
     host waits must remain limited to residency misses, bounded watchdog
     quanta, and completed emitted blocks.
   - Complete repeated-process determinism before accepting throughput changes.
     Scalar and temporal radix/top-k now have deterministic cutoff-tie ordering,
     score-descending output, compiler bounds, and numeric Vulkan coverage.
     Audit the remaining sampler RNG consumption, accepted-prefix commit,
     rollback, reset state, and expert-selection state. Require equality for
     committed tokens, routed experts, accepted prefixes, and state digests,
     and measure whether the current 512-entry bitonic ordering costs more than
     a deterministic radix-prefix implementation would.

4. Complete heterogeneous cost-based auto-placement. The runtime now admits
   demand-paged models against fixed residency plus one full selector wave,
   chooses the smallest capacity-safe device prefix when complete residency
   fits, and proportionally assigns a larger virtual model across every
   compatible cache when it does not. It respects partial VRAM reservations,
   keeps contiguous component segments, excludes integrated display GPUs, and
   leaves explicit wiring untouched. Placement and representation selection now
   run as one fixed-point solve: exact artifacts establish a capacity-safe
   graph, alternatives are selected against each component's actual target
   profile, and exact residency is replanned until both decisions converge.
   Whole-graph SPIR-V compatibility is checked before opening candidate
   devices. A fresh DeepSeek mount selected the smallest contiguous five-AMD
   prefix, excluded the discrete Intel target before allocation because the
   current package's FP8 shaders require unsupported `shader_float8`, and
   released every byte it acquired. Finish compiler emission and measurement
   of compatible BF16/INT8 alternatives, and rank cross-class spill using
   measured execution and transfer costs rather than model or vendor names, so
   a model that exhausts the AMD group can continue contiguously onto a
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
   recall, and release their exact recorded reservations. Before the compact
   MXFP4 performance commit, Qwen3.6-35B-A3B passed at 93.821 decode tok/s and
   126.699 prefill tok/s with a contiguous two-AMD split. Qwen3.5-9B passed at
   54.1862 decode tok/s and 135.1144 prefill tok/s on one AMD using the official
   temperature 1.0, top-k 20, top-p 0.95, min-p 0, presence-penalty 1.5,
   repetition-penalty 1.0 thinking profile. Keep repetition,
   structured-protocol, conversation, and teardown checks active so throughput
   alone cannot pass. The demand-aware resident-feedback runtime now passes the
   complete Qwen3.6-35B-A3B gate with its attached speculative decoder enabled
   on one discrete AMD GPU at **62.7030 decode tok/s** and **50.4392 prefill
   tok/s**. It also passes Qwen3.5-9B on one discrete AMD GPU at **48.4574
   decode tok/s** and **131.4108 prefill tok/s**. Both runs used 128K context,
   the 65,536-token output allowance, official thinking/sampling behavior, one
   discarded complete in-process conversation, a measured five-turn truth
   conversation, correct Greece recall, and exact teardown. After moving lazy
   temporal runners and deferred demand-chain materialization ahead of shared
   execution epochs, the same gates pass at **61.9432 decode tok/s** and
   **48.6530 prefill tok/s** for Qwen3.6-35B-A3B, and **48.7122 decode tok/s**
   and **137.1206 prefill tok/s** for Qwen3.5-9B. This regression exercised a
   near-capacity expert cache through both warmup and truth conversations; no
   allocator failure recurred and both GPUs returned exactly to their recorded
   pre-workload reservations. Retain these gates for every subsequent
   runtime-performance commit.

6. Perform a final adversarial review against `CONCEPT.md`: compiled artifacts
   remain self-contained and model-specific, while compiler discovery, runtime
   operators, graph wiring, placement, representation, residency, and stream
   transactions remain capability-driven and reusable by unseen models. Finish
   with an empty TODO, a clean worktree, and every milestone committed and
   pushed.
