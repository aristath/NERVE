# TODO

## Goal

Compile and run `/mnt/models/models/deepseek-v4/flash-0731/safetensors` as a
daily-driver NERVE model. Preserve source behavior, keep sparse routed experts
independently demand-resident, expose runtime per-component representation
selection, and optimize real multi-turn agentic inference past 30 tok/s and
toward 50 tok/s without regressing supported Qwen models.

All implementation must remain capability- and graph-driven. Model-specific
facts belong in the self-contained compiled package, never in runtime model-name
branches. Normal chat is the benchmark path: 128K context, a 65,536-token output
allowance, package-owned thinking and sampling behavior, complete-conversation
warmup, turn recall, and exact teardown. Tests and model gates run sequentially.

## Current evidence

- The latest accepted zero-load DeepSeek truth conversation is 8.4228 decode
  tok/s and 8.3570 prefill tok/s. It followed three discarded complete
  conversations whose residency loads converged from 10,004 to 414 to 8; the
  measured fourth conversation had zero reads, uploads, reloads, evictions, or
  residency blocking. NVMe paging is therefore not the steady-state limiter.
- Complete critical-path instrumentation is enabled in normal chat and covers
  99.99% of host wall time. The refined 2,224-token DeepSeek truth turn incurred
  only 4.788 seconds of residency blocking while attributing 70.598 seconds to
  attention read, 49.812 seconds to hyper-connections, 45.682 seconds to expert
  gate/up, 29.396 seconds to attention score, 26.420 seconds to dense projection,
  25.478 seconds to expert down, 8.505 seconds to state memory, and 6.972 seconds
  to grouped projection. The host is predominantly waiting for device work; the
  earlier claim that compact math was simply masked by host orchestration is no
  longer supported.
- Semantic attribution separates genuine sparse-expert operations from grouped
  attention projections and distinguishes expert gate/up, down and reduction,
  dense/grouped projections, normalization, quantization, state memory, index
  transforms, attention score, selection and read, positional encoding,
  hyper-connections, and pointwise activation. Cross-device device time is
  sampled by a bounded rotating probe at completed prompt boundaries; production
  transfers remain timestamp-free and have zero host waits. A real staged
  DeepSeek event sampled about 0.1 ms for one source/destination boundary pair,
  so physical activation movement is not a material steady-state limiter.
- The refined instrumentation passed exact sequential tests plus complete
  thinking-enabled regression gates. Qwen3.6-35B-A3B measured 60.6544 decode and
  45.8668 prefill tok/s; Qwen3.5-9B measured 48.6830 decode and 139.3640 prefill
  tok/s. Both produced correct turn recall and restored the exact pre-workload
  VRAM reservations.
- The current real six-expert MXFP4 microbenchmark measures 1.00392 ms for
  gate/up and 0.43088 ms for down. A host-visible-weight path is 3.912x slower.
  Gate/up is the first sparse-expert kernel target. Two alternative indexed
  attention schedules were rejected: subgroup-per-token changed product output,
  while an exact reduction-order-preserving tiled version was slower at both
  tiny and production geometry (1.79814 vs 1.57616 ms).
- A third indexed-attention candidate cached each packed BF16 key/value word in
  workgroup memory and reused it for score and value accumulation. At the exact
  q64/kv1/d512/r128/k8192 geometry it was byte-exact on the local fixture and
  reduced best device time from 19.96104 to 8.73592 ms, but the authoritative
  complete gate rejected it. With the accepted contiguous five-GPU placement,
  three complete warmups, and a zero-load measured conversation, decode fell
  from 8.4228 to 8.3290 tok/s and generated behavior changed. The shader,
  compiler selection, and tests were removed. Shared-memory reuse is therefore
  not promotion evidence unless every reachable geometry and the full product
  path remain exact and faster.
- Three further exact local-kernel candidates were rejected after the refined
  trace. Paired packed-BF16 state reads were 1.1% slower, and parallel tile
  exponentials were 0.5% slower; both preserved every BF16 output bit. A
  subgroup-tree hyper-connection reduction was 36% faster in isolation (0.16316
  vs 0.25456 ms) and byte-exact, but failed the complete product gate at 7.800
  mean decode tok/s versus the accepted 8.4228 baseline. It was removed from
  source and the compiled package was restored. Local shader wins are not
  promotion evidence when the complete routed stream does not improve.
- Four compact-expert representation candidates were also rejected on the same
  discrete AMD GPU and real six-expert geometry. Resolving and caching dynamic
  addresses once per workgroup preserved output but made gate/up 54% slower
  (0.67512 vs 0.43760 ms). Raising the gate tile from 32 to 128 rows was slower
  (0.86488 vs 0.43760 ms). A byte-exact once-per-route FP8 intermediate plus a
  prequantized down dispatch was slower than the fused down path (0.28520 vs
  0.24268 ms). Finally, llama.cpp-style MXFP4-to-signed-INT8 decode and packed
  INT8 dot products preserved a representable conformance input but improved
  gate/up by only 2.9% (0.42648 vs 0.43876 ms), which cannot justify changing
  the activation representation and model numerics. All four candidates were
  removed; no experimental runtime path remains.
- A generic mounted execution-graph phase now records all of its component
  dispatches as one ordered resident sequence. Exact Vulkan tests prove byte-for-
  byte output and state equivalence while reducing a colocated graph from one
  host submission per dispatch to one submission for the phase. In the identical
  cold DeepSeek `hi` probe this reduced direct sequence submissions from 4,276 to
  3,284 and total host queue-submit calls from 5,430 to 4,438 without changing
  generated tokens or residency decisions. It is one transaction segment, not
  completion of the persistent stream transaction.
- Device-to-host output ranges can now be mounted as a reusable packed readback
  transaction with retained command resources and persistently mapped staging.
  The attached parallel decoder uses one two-range transaction for token and
  confidence outputs instead of rebuilding two independent read paths. Its exact
  test replays changed source data twice with one transfer per replay. The
  deterministic cold DeepSeek probe improved from 2.027 to 2.077 decode tok/s and
  reduced host synchronization from 17.763 to 17.310 seconds. Qwen3.6-35B-A3B
  passed at 59.4312 decode tok/s and Qwen3.5-9B at 48.6694, with correct recall
  and exact VRAM restoration.
- A follow-up decoder experiment proved that fewer host submissions are not, by
  themselves, a resident transaction. It grouped input, ingress copy,
  demand-resident processing, egress copy, conditional output, and packed
  readback as five `SubmitInfo2` records in one `queue_submit2` call. On the
  zero-load DeepSeek truth conversation it cut sequence queue submissions from
  60,634 to 21,655 and fence waits from 64,965 to 25,986, but regressed decode
  from 8.4228 to 7.8368 tok/s. Removing a redundant host rewrite of the demand
  predicate recovered only 8.0160 tok/s. Both variants preserved exact tokens,
  zero-load residency, recall, teardown, and VRAM reservations, so the measured
  regression is in the device execution topology rather than paging or model
  behavior. The experiment was removed in full. Do not retry multi-command
  queue batching as if it were command fusion.
- A stricter follow-up recorded input, ingress, demand-resident processing,
  egress, guarded output, and packed readback into one primary Vulkan command
  buffer. The fourth complete conversation, after three discarded warmups, was
  fully resident and behaviorally exact but measured 8.0284 mean decode tok/s
  versus the accepted 8.4228 baseline: a 4.68% regression. Its result is nearly
  identical to the rejected multi-command queue batch despite eliminating the
  intermediate submissions. Serializing independent transfer and compute work
  into one queue therefore destroys useful device overlap. The integration was
  removed; the generic retained-copy and owned-invocation primitives remain as
  building blocks for an asynchronous hardware execution graph.
- Splitting each device's existing per-token command graph between independent
  compute and transfer queues was also rejected. An initial bridge-submission
  topology was behaviorally exact but measured 7.9682 mean decode tok/s. A
  direct timeline-handoff version removed the empty compute bridges and reached
  only 8.1466 tok/s, still 3.28% below the accepted 8.4228 baseline after three
  complete warmups and with zero measured residency loads. Both variants
  restored every GPU reservation exactly and were removed in full. Two queue
  submissions per device per token cost more than the roughly 0.1-ms staged
  activation edge can recover; independent hardware queues only become useful
  when a bounded resident window exposes genuine inter-token or inter-stream
  overlap instead of repartitioning the current serialized token schedule.
- Demand-paged residency correctness, immutable miss records, causal suffix
  resume, complete-conversation convergence, shared bounded host caching, and
  exact teardown are implemented. Resident FP8 coexistence is also implemented
  and measured, but the complete matched trial rejected it because several real
  geometries regressed and its footprint was worse than native compact MXFP4.

## Work queue

1. Optimize the hottest refined DeepSeek device kernels without changing model
   semantics.

   - Indexed sparse-attention read is the largest measured device phase. The
     exact local scheduling/load candidates tried so far are exhausted or slower;
     pursue a materially different compiled attention transaction rather than
     another barrier-for-barrier shader rewrite. Eliminate redundant
     state-address translation, duplicate key/value reads,
     serial score reduction, barriers, and full-score materialization where an
     exact fused or tiled schedule is faster. Preserve score accumulation order,
     online-softmax semantics, sink handling, compressed-index ordering, and
     BF16 output bits.
   - Native compact-MXFP4 vector alternatives are now locally exhausted: address
     caching, larger persistent tiles, once-per-route intermediate quantization,
     and packed INT8 dot products all lost or were immaterial. Do not retry those
     shapes. A future expert candidate must be a materially different compiled
     transaction, such as direct compact decode into a hardware matrix tile that
     amortizes its otherwise wasted columns across real attached proposal lanes,
     keeps SwiGLU on chip, preserves verified behavior, and does not permanently
     expand every sparse expert.
   - Every candidate microbenchmark must answer the binary faster/not-faster
     question in under one minute with only enough repetitions to avoid a cold
     anomaly. Reject a candidate immediately when it is slower or changes the
     deterministic product path.

2. Compile a persistent per-device stream transaction that preserves measured
   asynchronous device overlap.

   - Turn each ordered physical-device component segment into a stable bounded
     hardware execution topology. Preserve independent compute and transfer
     streams and synchronize them with device-side timeline dependencies and
     exact range hazards. Do not force independent engines through one serial
     primary command buffer, and do not disguise existing serialization as
     several `SubmitInfo2` records in one host call.
   - Put ticks, token IDs, dispatch dimensions, router results, expert addresses,
     sampler state, stop/cancel flags, causal frontiers, and commit records in
     GPU-resident control buffers consumed through predicates and indirect
     dispatch. Normal token values, context growth, expert choices, and
     address-table updates must not require command re-recording or a host return.
   - Submit one bounded stream window and watchdog quantum as a small fixed set of
     asynchronous device streams. Prove zero-miss host submissions and waits scale
     with devices, windows, and the fixed stream topology—not tokens, layers,
     selected experts, or graph nodes. Command-buffer caching alone is not
     completion; a previous template experiment reused almost every command buffer
     but fell to 1.205 tok/s because it retained the fragmented submission graph.
     Do not retry the rejected per-token compute/transfer split: even direct
     timeline handoffs regressed the complete gate to 8.1466 tok/s. The next
     topology must span a bounded resident window and overlap independent token
     streams or proposal lanes before paying for additional queue submissions.
   - The mounted input and output graph phases and packed host readback are now
     reusable transaction segments. Compose their dispatches, ingress/egress
     copies, demand gates, processor dispatches, and terminal readback into the
     bounded asynchronous topology. Preserve the accepted path until that topology
     wins the complete product gate; neither rejected decoder transaction may
     remain as a fallback, hidden mode, or future target shape.

3. Make a resident sparse-expert hit completely device-owned.

   - Keep selector-to-resource addresses plus validity/version metadata resident.
     An all-hit gate must continue directly into expert execution without a host
     fence, notification read, or execution-epoch round trip.
   - Only a real miss may publish an immutable fault record and stop at the exact
     causal checkpoint. Resume only the uncommitted suffix after the host updates
     the address table and acknowledges the fault.
   - Cover all-hit windows, disjoint misses, eviction, version changes, repeated
     faults, cancellation, rollback, and teardown. All-hit and miss/resume must
     produce identical committed tokens, routed experts, sampler state, and state
     digests from the same checkpoint.

4. Make cross-device transfers part of the compiled transaction.

   - Preserve arbitrary ordered visits, including `gpu0 -> gpu1 -> gpu0`, with
     persistent activation rings and timeline dependencies. A 32-KiB edge must not
     cause a host wait.
   - Select direct peer or staged transfer from measured capabilities and costs.
     Keep contiguous layer/component placement by default and never use tensor
     parallelism on this workstation.
   - First optimize single-stream latency. Pipeline independent streams across
     otherwise idle segments only after that path is correct and measured.

5. Make temporal prefill a real multi-token device transaction. Execute prompt
   blocks through the same resident gates, ordered segments, transfers, attention
   updates, and terminal completion without a host loop per token. Choose block
   width from context geometry, transient-state capacity, residency headroom, and
   measured hardware behavior. Verify every causal state against scalar prefill
   and report time-to-first-token separately from decode.

6. Rebuild attached DSpark execution on the persistent transaction. The compiler
   already discovers the package-owned `parallel_backbone_markov` decoder and its
   trained five-token minimum and legal seven-token execution width without model
   names. Fuse proposal, target verification, confidence-prefix comparison,
   accepted-state selection, commit/rollback, and draft catch-up into one
   device-owned transaction. Promote it automatically only when exact proposal,
   accepted-prefix, routed-expert, sampler, and state equivalence pass and useful
   committed tokens per complete cycle beat scalar chat. The current 33.33%
   acceptance and 2.853/0.967 useful tok/s width-five/width-seven results are a
   rejected baseline, not a mode to force.

7. Optimize deterministic attention scoring and selection after the attention-read
   path. Evaluate fused score/select, deterministic radix-prefix, blockwise or
   hierarchical selection, and avoiding full-score materialization. Preserve
   exact tie ordering and score-descending output. Measure progressively larger
   real contexts rather than hiding growth behind a short benchmark.

8. Preserve bounded failure handling without putting it on the hot path. Isolate
   Vulkan execution behind supervised per-device workers carrying complete stream
   transactions, not component calls. Retain bounded watchdog progress, quarantine
   only the poisoned worker, preserve the first failure and last causal checkpoint,
   and keep all other workers and the UI responsive without per-token IPC.

9. Complete capability-driven per-component representation selection. Preserve
    the native source representation whenever it wins on the assigned target.
    Add dense or exactly valid structured INT4, FP8, INT8, and FP16 candidates only
    through device-local measurement and behavioral-equivalence evidence. Promotion
    must account for the whole routed working set, resident footprint, reload
    traffic, representation boundaries, and live headroom; capability advertises a
    candidate but never selects it by itself.

10. Extend the generic product gate with a representative tool-call round trip,
    malformed-output rollback, and long-stream continuity. Use package-owned chat
    behavior and sampling defaults, official thinking behavior, agentic context,
    complete-conversation warmup, coherent final answers, turn recall, and exact
    teardown without model-name branches.

11. Complete heterogeneous cost-based placement. Keep the existing smallest
    capacity-safe contiguous device prefix, partial-reservation handling, integrated
    GPU exclusion, explicit-wiring preservation, and representation/placement
    fixed-point solve. Add compiler-emitted compatible BF16/INT8 alternatives and
    measured cross-class execution/transfer ranking so a graph may spill to a
    compatible discrete Intel GPU or CPU without recompilation when AMD capacity is
    exhausted.

12. Gate every runtime-performance milestone before commit. Run exact sequential
    tests, then full DeepSeek, Qwen3.6-35B-A3B, and Qwen3.5-9B conversations on
    equivalent allowlisted discrete AMD placement. DeepSeek truth starts only after
    a complete zero-load conversation. Reject microbenchmark-only wins, behavioral
    changes, Qwen regressions, placement violations, GPU faults, or failure to
    restore every pre-workload VRAM reservation. Reach 30 decode tok/s, continue
    toward 50, and stop only when the attributed path has no material avoidable host
    round trip, GPU bubble, conversion, or unfused memory pass.

13. Perform a final adversarial review against `CONCEPT.md`. Compiled artifacts
    must remain self-contained and model-specific; compiler discovery, runtime
    operators, graph wiring, placement, representation, residency, and stream
    transactions must remain reusable by unseen models. Finish with an empty TODO,
    a clean owned worktree, and every milestone committed and pushed.
