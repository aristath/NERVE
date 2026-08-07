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
   to FP8 derivation without changing source artifacts. A generic optimizer
   provider now discovers complete component-local sparse expert pairs from
   semantic roles, tensor metadata, selector-addressable residency, and target
   capabilities; constructs their exact resident FP8 shaders and overlay; and
   independently proves exhaustive numeric equivalence plus complete source
   weight/execution-path coverage. The real DeepSeek `layer_00` boundary passes
   with 768 independently derived resources and four scalar/batch paths, but
   capability still creates only a candidate and never promotes it. Complete
   the remaining alternative-set contract, device-local measurement evidence,
   and runtime choice so native and derived representations can coexist and be
   selected per component on the component's actual target device.

   The fresh v7 run now qualifies that exact resident-FP8 alternative through
   a complete matched hardware benchmark in less than the one-minute contract.
   One pooled executor retains only sealed CPU-side runtime-model templates;
   independent Vulkan sessions and dynamic resource stores remain trial-local.
   All 20 observations and 24 residency events completed. Two scalar workloads
   favored resident FP8, but three gate/up, down, or batch workloads crossed the
   permitted regression floor, so the optimizer correctly retained native
   compact MXFP4. This closes resident FP8 as an unmeasured possibility; add
   INT4/INT8 and structured-sparse alternatives only as independently measured
   candidates, never as capability-selected defaults.

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
   allocation. The optimizer's product gate must discard a complete first
   conversation, measure every turn of the complete second conversation, and
   run a third complete conversation when the second still performs residency
   loads. Finish the executor/protocol integration and prove malformed or
   incomplete warmup evidence fails closed; never compare a cold reference
   with a warm candidate. The 2026-08-06 workstation crash was an amdgpu TTM
   LRU corruption during global BO eviction after NERVE filled every discrete
   AMD heap to the old capped 4 GiB watermark. Finish uncapped proportional
   headroom and proactive live-budget reclamation, then prove demand-paged
   execution stays below the physical pressure watermark while preserving
   unrelated allocations before any further full-model benchmark.

   Its current official-profile truth baseline is **8.0818 decode tok/s** and
   **8.4564 prefill tok/s** before the compact-MXFP4 remap. The
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
     cycle elapsed time. One-time candidate materialization is now excluded as a
     residency warmup, while a mode that keeps loading on consecutive cycles has
     its second and third load cycles charged and cannot stall or game
     calibration. On this sparse package it selects scalar and improves decode by
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
     compiled; widths 1-7 remain legal runtime views of that capacity. The
     package-owned seven-token recommendation now activates when the runtime
     option is omitted, while an explicit `--speculative-draft-tokens 0`
     disables it. Conflicting attached-decoder recommendations fail closed.
     A normal 128K DeepSeek chat mount reported `speculative_draft_tokens=7`
     without an override and released every acquired allocation. Finish
     reference-equivalence evidence for proposal, confidence, verification,
     commit, and rollback behavior.

     The first five-device demand-paged run proves the b7 decoder executes, but
     not yet profitably: calibration measured 10 DSpark cycles, proposed 34
     draft tokens, accepted 14 (41.18%), and selected scalar afterward. Width 4
     was best at 3.734 useful tok/s; width 7 reached only 1.514 useful tok/s.
     After eliminating the obvious host submission/fence bottleneck below,
     prioritize the attached MTP/DSpark/DFlash-class decoder as the primary
     throughput multiplier. Fuse proposal, target verification, prefix
     comparison, state selection, canonical commit, and draft catch-up into one
     GPU transaction so seven-token execution no longer expands into thousands
     of host submissions and waits. Discover attached draft structures and
     their execution contracts from package semantics and capabilities, never
     from model names. Re-run two complete discarded conversations when the
     second still loads resources, followed by the untouched truth conversation;
     the attached draft path is complete only when it is behaviorally correct
     and wins the selector on measured useful committed tokens per second.
     The demand-aware resident feedback checkpoint transaction is now implemented.
     A faulting traversal follows only the causal suffix from each GPU checkpoint,
     loads the missing cohort without replaying completed lanes, and performs one
     clean commit traversal from the captured model/sampler baseline after every
     demanded resource is resident. It does not revive the rejected complete-graph
     replay after every miss. Vulkan producer/copy/consumer visibility is explicit,
     and protocol-terminated generation branches now finalize and retain their real
     event report before transactional state restoration. Exact tests cover causal
     frontiers, indirect offsets, cross-device predicates, completion visibility,
     protocol termination, and canonical rollback; real Qwen3.6-35B and Qwen3.5-9B
     gates remain above their accepted floors.

     Routed demand continuation now consumes the immutable GPU miss record rather
     than rereading shared selector working memory. One causal checkpoint may
     legitimately expose several disjoint expert sets as earlier dependencies
     become resident; the runtime tracks checkpoint-resource pairs against the
     compiled selector-domain bound and rejects only a resource that faults twice.
     A forced resident cold run that previously failed immediately completed a
     coherent thinking response, loaded 5,002 distinct experts with zero eviction,
     and released every allocation. Its 1.196 decode tok/s included 71.70 seconds
     of source reads and 51.81 seconds of blocking transfer, so it is correctness
     evidence rather than a warm performance result.

     The 2026-08-07 authoritative DeepSeek run discarded two complete conversations
     because the second still loaded resources, then measured the complete third
     conversation in the same mounted process. It produced coherent thinking-enabled
     answers with correct Greece/Athens recall at **8.2296 decode tok/s** and
     **8.4030 prefill tok/s**. The measured set had exactly zero residency misses,
     loads, reads, uploads, reloads, or evictions, proving steady-state execution—not
     NVMe paging—is now the limiting path. The first discarded set loaded 132.54 GB;
     the second loaded another 6.36 GB; the third reused all 1,149,648 selected expert
     accesses from residency. Teardown acknowledged all five physical devices and
     preserved the pre-existing PCI `0000:03:00.0` allocation; no discrete-GPU
     timeout, reset, or page fault occurred.

     The representative 2,224-token steady-state turn remains the next dominant
     bottleneck. The latest run spent 302,202.763 ms in decode; the prior instrumented
     run spent 292,640.171 ms while measured execution quanta
     accounted for only 528.660 ms and resident component device work for 263.870 ms.
     NERVE issued 36,834 resident-sequence submissions, waited on 39,465 sequence
     fences, issued 31,837 copy submissions, and waited on 10,852 copies. Each of the
     four cross-device edges also executed 2,242 separate 32-KiB transfers. Replace
     lane-by-lane host submission with a persistent resident feedback window per
     physical device: encode causal lane iteration, component segments, and edge
     copies in GPU-resident indirect command/control state; submit one bounded command
     stream per device/window; preserve watchdog quanta; and return to the host only
     for a real residency miss, cancellation, or completed emitted block. Prove exact
     stop-token, sampler, routed-expert, state-digest, protocol-boundary, and teardown
     equivalence before promotion.

     Do not mistake command-buffer recording for that persistent stream. A rejected
     demand-template experiment replayed every completed feedback window and reused
     8,234 of 8,452 resident sequence command buffers, yet a 124-token DeepSeek turn
     fell to **1.205 decode tok/s**. It still issued 7,976 queue batches, 6,238 copy
     submissions, and 8,828 copy waits. Reusing the same fragmented submission graph
     is therefore not the solution. The next candidate must change the submission
     topology itself: one bounded device-owned transaction must carry routing,
     demand predicates, expert execution, cross-device edges, sampling, and the
     attached DSpark proposal/verification/commit loop. Host-visible checkpoints
     remain only for a true residency fault, cancellation, watchdog progress, or a
     completed emitted block.
   - Isolate product Vulkan execution behind supervised per-device worker
     processes. Normal inference fence, timeline, transfer, and quiescence waits
     now poll at the existing 250 ms execution-quantum target, quarantine a
     device after four quanta with no observable progress, retain the first
     failure, and never revive it through the DRM activity lease. This bounds
     the coordinator's wait, but an in-process driver fault can still leave
     Vulkan objects in flight. The coordinator must be able to terminate only
     the poisoned worker context without unwinding live GPU objects, keep other
     component workers and the UI responsive, and report the exact device and
     last completed checkpoint. Validation/optimizer executors already provide
     process boundaries; generalize that mechanism instead of inventing a
     second model-specific recovery path.
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
   pre-workload reservations. The package-default milestone then re-ran both
   gates with explicit widths: Qwen3.6-35B-A3B passed at **62.6418 decode
   tok/s** and **50.4346 prefill tok/s**, while Qwen3.5-9B passed with explicit
   speculative disable at **48.7560 decode tok/s** and **140.2734 prefill
   tok/s**. The residency-aware execution-selector milestone was then rebuilt
   after the workstation reboot and re-ran both complete gates on individually
   allowlisted, unaffected AMD devices. Qwen3.6-35B-A3B passed at **62.1522
   decode tok/s** and **58.1906 prefill tok/s**; Qwen3.5-9B passed at **48.8166
   decode tok/s** and **142.8648 prefill tok/s**. Both retained correct Greece
   recall, emitted full thinking traces, returned exactly to their recorded
   59,985,920-byte VRAM baselines, and produced no new kernel GPU fault. Retain
   these gates for every subsequent runtime-performance commit.
   After routed checkpoint convergence was fixed, the same complete gates passed
   again: Qwen3.6-35B-A3B measured **57.8022 decode tok/s** and **51.9794 prefill
   tok/s** on one discrete AMD GPU with its attached decoder enabled; Qwen3.5-9B
   measured **48.8246 decode tok/s** and **140.0010 prefill tok/s** with explicit
   speculative disable and its official thinking/sampling profile. Both retained
   correct Greece recall and returned exactly to their recorded pre-run VRAM
   reservations.

6. Perform a final adversarial review against `CONCEPT.md`: compiled artifacts
   remain self-contained and model-specific, while compiler discovery, runtime
   operators, graph wiring, placement, representation, residency, and stream
   transactions remain capability-driven and reusable by unseen models. Finish
   with an empty TODO, a clean worktree, and every milestone committed and
   pushed.
