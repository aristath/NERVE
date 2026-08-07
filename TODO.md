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
     compiled. The checkpoint-trained width of five is the minimum legal query
     geometry; the execution graph may extend that complete block to the
     official seven-token runtime width, but widths one through four are not
     valid cheaper views of the same learned circuit. The
     package-owned seven-token recommendation now activates when the runtime
     option is omitted, while an explicit `--speculative-draft-tokens 0`
     disables it. Conflicting attached-decoder recommendations fail closed.
     A normal 128K DeepSeek chat mount reported `speculative_draft_tokens=7`
     without an override and released every acquired allocation. Finish
     reference-equivalence evidence for proposal, confidence, verification,
     commit, and rollback behavior.

     The first five-device demand-paged run proved the b7 decoder executes, but
     its width-one-through-four measurements used invalid partial query
     geometry and are superseded. The corrected runtime rejects explicit
     widths below five and calibrates only complete width-five and width-seven
     blocks before comparing them with scalar and resident feedback. The fresh
     corrected calibration ran three cycles per legal geometry: six cycles
     proposed 36 tokens and accepted 12 (33.33%). Width five delivered 2.853
     useful committed tok/s and width seven delivered 0.967, so the selector
     correctly retained scalar execution instead of slowing normal chat.
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

     The corrected 2026-08-07 authoritative DeepSeek run discarded three complete
     conversations because residency converged through 10,004 loads, then 414,
     then 8. It measured the complete fourth conversation in the same mounted
     process at **8.4228 decode tok/s** and **8.3570 prefill tok/s**, with coherent
     thinking-enabled answers and correct Greece/Athens recall. The truth set had
     exactly zero residency misses, loads, reads, uploads, reloads, evictions, or
     blocking time, proving steady-state execution—not NVMe paging—is the limiting
     path. The discarded sets uploaded 133.75 GB, 5.53 GB, and 106.95 MB before the
     final set reused its complete working set. Teardown released every NERVE
     allocation on all five physical devices, preserved the pre-existing PCI
     `0000:03:00.0` allocation, and produced no discrete-GPU timeout, reset, or
     page fault.

     The representative 2,224-token steady-state turn defines the orchestration
     baseline. It spent 292,793.624 ms in decode, or 131.652 ms per token, while
     the currently instrumented execution quanta accounted for only 517.488 ms.
     NERVE prepared 44,727 resident sequences, recorded 23,679 command buffers,
     issued 36,834 sequence submissions and 31,726 copy submissions, waited on
     39,465 sequence fences and 10,704 copies, and made 2,631 additional queue
     batch submissions. That is approximately 32 queue submissions and 22.6
     waits per generated token. Each of the four cross-device edges also made
     2,242 separately synchronized 32-KiB transfers. The zero-miss truth set
     proves this is the steady-state execution path, not paging.

     Do not mistake command-buffer caching for the required solution. A rejected
     demand-template experiment reused 8,234 of 8,452 command buffers but fell
     to **1.205 decode tok/s** because it preserved the fragmented submission
     topology. The implementation must change who advances the stream and where
     synchronization occurs, not merely replay the same host-driven calls.

   - **3.a Account for the complete critical path.** Add low-overhead host spans
     and Vulkan timestamps around scheduler/control work, command preparation and
     recording, queue submission, fence/timeline waits, routing, residency gates,
     expert gate/up/down projection, attention/index selection, cross-device
     copies, output projection, sampling, speculative verification, and state
     commit. Keep aggregate counters and timings enabled during normal chats;
     do not introduce a special profiling execution path. Attribute at least 95%
     of wall-clock critical-path time without double-counting overlapped GPU work,
     report per-token and per-window totals, and prove the instrumentation itself
     causes no material throughput or quality regression. The current 517.488 ms
     quantum figure is explicitly incomplete and must not be interpreted as the
     model's complete GPU time.

   - **3.b Compile a persistent per-device stream transaction.** Turn each ordered
     physical-device component segment into a stable, bounded command topology.
     Put stream ticks, token IDs, dispatch dimensions, router results, expert
     addresses, sampler state, stop/cancel flags, causal frontiers, and commit
     records in GPU-resident control buffers consumed through indirect dispatch
     and predicates. Submit one bounded stream window per device and watchdog
     quantum rather than one call per component or token. A normal completed
     window must return to the host only for emitted output, cancellation,
     watchdog progress, or teardown. Command buffers may be re-recorded when the
     compiled topology changes, but normal token values, context growth, expert
     choices, and address-table updates must not force re-recording. Prove that
     zero-miss queue submissions scale with physical devices and windows, not
     with tokens, layers, selected experts, or graph nodes.

   - **3.c Make a resident hit a completely device-owned operation.** Keep the
     selector-to-resource address table and validity/version metadata resident.
     A residency gate whose selected addresses are valid must continue directly
     into expert execution without a host fence, notification read, or round trip
     through a host execution epoch. Only a real miss may publish an immutable fault record and
     stop at the exact causal checkpoint. The host then loads or derives the
     missing cohort, updates the address table, acknowledges the fault, and resumes
     only the uncommitted suffix. Cover all-hit windows, one and multiple disjoint
     misses, eviction between windows, address-version changes, repeated-fault
     rejection, cancellation during a fault, rollback, and exact teardown. The
     all-hit and miss/resume paths must produce identical committed tokens,
     routed experts, sampler state, and state digests from the same checkpoint.

   - **3.d Make cross-device cables part of the compiled transaction.** Represent
     every physical-device visit as an ordered segment, including graphs that
     revisit a device such as `gpu0 -> gpu1 -> gpu0`. Use persistent ping-pong or
     ring activation buffers and timeline semaphores; enqueue source copies,
     destination staging when peer access is unavailable, and the consumer segment
     as one dependency chain. A 32-KiB edge must not cause a host wait. Select
     direct peer transfer or staged transfer from measured device capabilities,
     keep contiguous layer/component placement by default, and retain arbitrary
     user wiring. First optimize single-stream latency; only then pipeline different
     streams across otherwise idle device segments to improve aggregate throughput.
     Never substitute tensor parallelism for this layer/component pipeline on this
     workstation.

   - **3.e Fuse and specialize the sparse-MoE hot path after 3.b-3.d.** Keep router
     top-k, address validation, the selected expert gate/up/down work, and weighted
     reduction inside the same transaction. Benchmark grouped/fused expert dispatch
     against separate dispatches using the real six-expert geometry. Retain packed
     MXFP4/INT4 through consumption, amortize conversion, and test native MXFP4,
     dense or exactly valid structured INT4, FP8, and INT8 implementations on the
     assigned device. Structured sparsity is eligible only when the source already
     satisfies the required structure or a behaviorally validated transformation
     preserves quality; hardware capability alone is not permission to prune or
     reinterpret weights. Promotion evidence must include end-to-end throughput,
     footprint, reload traffic, and headroom, not kernel time alone. Retain the
     exact E2M1-to-E4M3 remap, but do not spend another milestone on an isolated
     unpacking variant before orchestration is removed.

   - **3.f Make temporal prefill a real multi-token device transaction.** Execute
     prompt blocks through the same resident gates, ordered device segments,
     transfers, attention state updates, and terminal completion without a host
     loop per token. Select block width from context geometry, available transient
     state, residency headroom, and measured hardware behavior. Verify every token's
     causal state against scalar prefill, and measure time-to-first-token separately
     from decode. The current **8.3570 prefill tok/s** truth result is not acceptable
     for agentic workloads with large prompts.

   - **3.g Rebuild attached DSpark execution on the persistent transaction.** After
     scalar execution no longer returns to the host inside a resident window, fuse
     proposal, target verification, confidence/prefix comparison, accepted-state
     selection, canonical commit or rollback, and draft catch-up into one
     device-owned transaction. Preserve the trained five-token minimum geometry
     and the package's legal seven-token recommendation. Compare scalar, resident,
     width-five, and width-seven modes using useful committed tokens per complete cycle;
     include all target work, transfer, rollback, and catch-up time. The current
     33.33% acceptance with 2.853 and 0.967 useful tok/s is a rejected baseline,
     not a reason to force speculation. Promote DSpark only after exact proposal,
     accepted-prefix, sampler, routed-expert, and committed-state equivalence and
     only when it wins normal chat automatically.

   - **3.h Optimize long-context attention and deterministic selection after the
     orchestration milestones.** Use the new timers to isolate local attention,
     compressed-memory indexing, score generation, and the 512-entry top-k. Evaluate
     fused score-and-select, deterministic radix-prefix, blockwise/hierarchical
     selection, and avoidance of full-score materialization. Preserve exact
     cutoff-tie ordering and score-descending output. Benchmark at progressively larger
     real contexts because the steady turn already declines from 8.527 to 7.118
     tok/s between its first and fourth sustained-decode windows; optimize the
     growing component rather than hiding it with a short context.

   - **3.i Preserve bounded failure handling without putting it on the hot path.**
     Isolate Vulkan execution behind supervised per-device worker processes, but
     send complete stream transactions across that boundary rather than component
     calls. Retain the 250 ms watchdog quantum, quarantine a device after four
     quanta without observable progress, preserve the first failure, and never
     revive it through the DRM activity lease. The coordinator must be able to
     terminate only the poisoned worker, keep the other device workers and UI
     responsive, and report the device and last completed causal checkpoint.
     Generalize the existing optimizer/validation worker mechanism; do not create
     a model-specific recovery path or add per-token IPC.

   - **3.j Gate and promote every milestone with product behavior.** Before each
     performance commit, run tests sequentially and then run the complete DeepSeek,
     Qwen3.6-35B-A3B, and Qwen3.5-9B quality/performance gates on equivalent allowed
     AMD placement. DeepSeek truth begins only after a complete conversation has
     zero residency loads; discard another complete conversation when necessary.
     Use 128K context, the 65,536-token output allowance, package-owned thinking and
     sampling behavior, the established short five-turn conversation, turn recall,
     structured/tool protocol checks, malformed-output rollback, long-stream
     continuity, and exact teardown. Record per-token critical-path time, queue
     submissions, waits, command recordings, edge transfers, useful speculative
     tokens, and prefill/decode rates. Reject any change that wins a microbenchmark
     but does not improve the full resident conversation, changes committed behavior,
     regresses Qwen materially, violates placement rules, or fails to restore every
     pre-workload VRAM reservation. Reach **30 decode tok/s**, continue toward 50,
     and keep iterating until the attributed critical path contains no material
     avoidable host round trip, GPU bubble, representation conversion, or unfused
     memory pass.

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
   After enforcing trained parallel-block draft geometry, both gates passed again
   from `main`: Qwen3.6-35B-A3B measured **53.6328 decode tok/s** and **60.4842
   prefill tok/s** with its attached decoder enabled, while Qwen3.5-9B measured
   **48.8470 decode tok/s** and **141.6512 prefill tok/s** with explicit speculative
   disable. Both used the 128K context, 65,536-token output allowance, official
   thinking/sampling behavior, one complete discarded conversation, correct
   Greece/Athens recall, exact teardown, and no new kernel GPU fault.

6. Perform a final adversarial review against `CONCEPT.md`: compiled artifacts
   remain self-contained and model-specific, while compiler discovery, runtime
   operators, graph wiring, placement, representation, residency, and stream
   transactions remain capability-driven and reusable by unseen models. Finish
   with an empty TODO, a clean worktree, and every milestone committed and
   pushed.
