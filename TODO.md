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

- The accepted zero-load DeepSeek truth conversation remains 8.4228 decode
  tok/s and 8.3570 prefill tok/s. It followed three discarded complete
  conversations whose residency loads converged from 10,004 to 414 to 8; the
  measured fourth conversation had zero reads, uploads, reloads, evictions, or
  residency blocking. NVMe paging is therefore not the steady-state limiter.
- A fresh schema-v10 compilation with explicit fixed-source MXFP4 dispatch
  contracts passed a complete four-conversation product gate, preserved behavior
  and turn recall, and restored every GPU reservation, but its zero-load truth
  conversation reached only 7.4952 decode and 7.8016 prefill tok/s. The same
  10,004 -> 414 -> 8 -> 0 load convergence proves the 11.01% decode regression
  is not paging. Device time regressed broadly rather than in one host phase:
  the 2,224-token truth turn measured 74.625 seconds of attention read, 52.927
  seconds of hyper-connections, 50.215 seconds of expert gate/up, 32.073 seconds
  of attention score, 28.071 seconds of expert down, and 27.590 seconds of dense
  projection. Keep 8.4228/8.3570 as the acceptance baseline until an equivalent
  complete gate beats it.
- Equal layer counts are not an execution-cost-equivalent placement on the live
  workstation even across five nominally identical R9700s. The exact sequential
  six-expert compact-MXFP4 microcase measured 0.586, 1.194, 0.415, 0.418, and
  1.375 ms on PCI 03, 07, 0a, 0d, and 21 respectively. A width-six native
  compact batch took 2.480 ms on 0d versus 5.040 ms on 21, while preexpanded FP8
  was essentially tied at 1.392 versus 1.362 ms. Links were all PCIe 5.0 x16 and
  reservations were restored exactly. Placement and representation therefore
  need a live, implementation-specific cost signal; device name and capability
  class are insufficient.
- Complete host critical-path instrumentation covers 99.99% of wall time. The
  prior fully timestamped 2,224-token DeepSeek truth turn incurred only 4.788
  seconds of residency blocking while attributing 70.598 seconds to attention
  read, 49.812 seconds to hyper-connections, 45.682 seconds to expert gate/up,
  29.396 seconds to attention score, 26.420 seconds to dense projection, 25.478
  seconds to expert down, 8.505 seconds to state memory, and 6.972 seconds to
  grouped projection. That diagnosis established that the host predominantly
  waits for device work; keeping expensive per-dispatch timestamps on every
  token was no longer justified.
- Normal chat now records detailed device phases for the first execution window
  and then one bounded sample every 128 windows. Profiled and unprofiled command
  sequences and submission templates have distinct identities, deferred paths
  never emit timestamps that their completion path cannot consume, and reports
  no longer mislabel a sampled duration as a full-stream per-token cost. On the
  same fixed topology and current package, DeepSeek improved from 7.8596 to
  8.0284 decode tok/s and from 7.8136 to 7.9794 prefill tok/s. The zero-load
  truth conversation preserved behavior and recall and restored all five exact
  reservations. Qwen3.6-35B-A3B improved from a matched 70.8880 to 71.4034
  decode tok/s, while Qwen3.5-9B reached 48.8568; both had zero measured paging
  and exact teardown. The remaining DeepSeek gap is device execution, not
  always-on telemetry.
- Semantic attribution separates genuine sparse-expert operations from grouped
  attention projections and distinguishes expert gate/up, down and reduction,
  dense/grouped projections, normalization, quantization, state memory, index
  transforms, attention score, selection and read, positional encoding,
  hyper-connections, and pointwise activation. Cross-device device time is
  sampled by a bounded rotating probe at completed prompt boundaries; production
  transfers remain timestamp-free and have zero host waits. Automatic placement
  now derives every cut payload and direction from the compiled graph, measures
  the production-compatible physical route for each ordered device pair after
  one discarded replay, and includes the exact payload-specific cost in the
  contiguous-partition objective. Missing measurements fail placement instead
  of becoming a zero-cost edge, and non-chain wiring remains an explicit graph
  concern rather than being flattened. The real Qwen boundary is 4,096 bytes;
  its staged route measured 23.16 microseconds from PCI 0a to 0d and 22.12
  microseconds in the reverse direction, confirming that physical activation
  movement is not a material steady-state limiter.
- The refined instrumentation passed exact sequential tests plus complete
  thinking-enabled regression gates. Qwen3.6-35B-A3B measured 60.6544 decode and
  45.8668 prefill tok/s. A fresh Qwen3.5-9B package now discovers its producer's
  unambiguous documented sampling policy when its machine-readable generation
  file omits one; the package-owned warmed gate measured 48.8240 decode and
  142.5696 prefill tok/s with correct turn recall, zero measured residency
  activity, and exact pre-workload VRAM restoration. Conflicting documented
  profiles are not guessed, explicit machine metadata always wins, and no model
  name participates in discovery.
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
- Splitting indexed attention into an exact score transaction followed by a
  value transaction was also rejected after two complete product gates. The
  reduction-order-preserving score kernel was byte-exact and faster in isolation:
  0.35104 versus 0.47208 ms at a 4K live context, 1.29800 versus 2.79764 ms at
  128K, and 7.02364 versus 20.05572 ms at the compiled maximum. Nevertheless,
  static maximum dispatch produced only 7.9582 mean decode tok/s, and a generic
  context-capacity-bound dispatch reduced the authoritative zero-load truth run
  further to 7.7540 decode and 7.9702 prefill tok/s. The latter had zero reads,
  uploads, reloads, evictions, or residency blocking, retained coherent answers
  and turn recall, and restored every GPU reservation. Both implementations were
  removed. A full F32 score plane, an added dispatch and buffer transaction, and
  duplicated score/value traversal cost more in the complete stream than the
  locally faster score kernel saves. The next attention design must fuse score,
  selection, and value consumption into one compiled transaction and improve the
  whole attributed attention phase; do not materialize all scores between
  independently scheduled kernels.
- Caching one paged-state base address per token in the existing fused indexed-
  attention workgroup was byte-exact and 10.9% faster in the exact
  q64/kv1/d512/r128/k8192 microbenchmark (2.67260 versus 3.00004 ms), but it too
  failed the complete product gate. After three full warmup conversations, the
  measured conversation was fully resident, preserved coherent answers and turn
  recall, and restored every GPU reservation, yet reached only 7.9096 decode and
  7.9520 prefill tok/s versus the accepted 8.4228/8.3570 baseline. The 6.09%
  decode regression proves that removing redundant address translation inside
  this barrier topology does not improve the complete stream. The shader,
  renderer selection, exact fixture, and compiled package were removed. Do not
  retry state-base caching without also replacing the serial full-width reduction
  and barrier schedule as one materially different fused transaction.
- A fused 16-head by 16-token cooperative-matrix indexed-attention transaction
  eliminated the scalar 512-wide score reduction and reused each latent state
  tile across score and value multiplication. At q64/kv1/d512/r128/k8192 it
  reduced the representative 128K device microcase from 3.07856 to 1.44916 ms,
  but rounding online-softmax weights to BF16 changed the output (0.00048828
  maximum and 0.00001689 RMS error). The real product path confirmed the
  numerical change was not acceptable promotion evidence: a 1,615-token
  zero-new-load warmup turn sustained only 6.424 decode tok/s versus the
  accepted 8.4228 baseline. The gate was stopped immediately, all five exact
  pre-run VRAM reservations were restored, and the shader, selection logic,
  tests, and compiled package were removed. A future matrix-tiled attention
  transaction must retain F32 online-softmax weights and exact BF16 outputs;
  raw cooperative-matrix throughput does not compensate for an altered stream.
- Three further exact local-kernel candidates were rejected after the refined
  trace. Paired packed-BF16 state reads were 1.1% slower, and parallel tile
  exponentials were 0.5% slower; both preserved every BF16 output bit. A
  subgroup-tree hyper-connection reduction was 36% faster in isolation (0.16316
  vs 0.25456 ms) and byte-exact, but failed the complete product gate at 7.800
  mean decode tok/s versus the accepted 8.4228 baseline. It was removed from
  source and the compiled package was restored. Local shader wins are not
  promotion evidence when the complete routed stream does not improve.
- A grouped-query-head indexed-attention schedule inspired by the row blocking
  in llama.cpp's Vulkan flash-attention path was also rejected locally before a
  product gate. It reused each latent KV read across independent query heads
  while preserving every head's existing reduction, online-softmax order, and
  every BF16 output bit at q64/kv1/d512/r128/k8192. Four heads per workgroup took
  30.46644 ms versus 19.16268 ms for the scalar-head kernel; even two heads took
  23.33104 ms versus 19.77996 ms. Serial 512-wide reductions and their register/
  shared-state pressure cost more than the duplicate read traffic. The shader,
  renderer, and exact fixture were removed. Future head-row blocking must split
  dimensions across smaller subgroups or use matrix tiles; do not repeat several
  full-width head reductions inside one workgroup.
- Folding the 512 dimensions onto a 256-thread workgroup was likewise
  byte-exact but slower at the same real geometry. On the target's native
  64-lane subgroups, each lane handled two dimensions while each original
  subgroup reduction and the eight-partial serial sum retained their exact
  order. It still took 24.44188 ms versus 19.75720 ms, a 23.71% regression.
  The candidate and exact fixture were removed. Maximum dimension parallelism
  is useful here; reducing resident waves does not pay for serial folded loads
  and reductions.
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
- Replaying the stable initial demand-gated feedback topology was also tested
  and removed. The implementation correctly keyed fresh-input and carried-input
  shapes separately, survived real miss rollback/resume, replayed three warmed
  calibration windows, preserved exact long-form behavior and recall, and
  restored every GPU reservation. It still measured only 8.1736 mean decode
  tok/s versus the accepted 8.4228 baseline. Normal adaptive execution selected
  scalar decode after calibration, so the cached transaction was absent from
  steady-state truth turns and could not improve them. Do not add more caching
  around an execution candidate that does not win; the resident transaction
  itself must remove device work or expose useful overlap before replay matters.
- Demand-paged residency correctness, immutable miss records, causal suffix
  resume, complete-conversation convergence, shared bounded host caching, and
  exact teardown are implemented. Resident FP8 coexistence is also implemented
  and measured, but the complete matched trial rejected it because several real
  geometries regressed and its footprint was worse than native compact MXFP4.
- Package-derived placement calibration now measures six real execution
  signatures once, reuses prepared tensor/slice plans across all targets, and
  finishes the five-device sweep in under one minute. A capacity-constrained
  contiguous dynamic program jointly selects device order and boundaries from
  those component costs. Demand-paged placement enforces both lower and upper
  retained-byte quotas around the capacity-proportional share; the rejected
  upper-only variant caused pathological paging, while the bounded variant
  completed the cold DeepSeek conversation without evictions.
- Demand-paged prefix selection now distinguishes physical executability from
  the best compatible residency plan. It keeps adding caller-ranked compatible
  devices while their aggregate safe capacity can eliminate addressable-set
  paging, then applies the paged working-set quota before live latency ranking.
  The real Qwen3.6-35B-A3B auto-placement moved from a calibration-sensitive
  31/9-layer split with 312 loads and 39.6642 decode tok/s to a stable 19/21
  split with zero measured loads, reloads, or evictions and 69.2016 decode
  tok/s. Qwen3.5-9B still selected one GPU and measured 48.8308 decode tok/s.
  The unchanged fixed DeepSeek control completed three discarded conversations
  followed by a zero-load, zero-reload truth conversation at 7.9334 decode and
  7.8914 prefill tok/s with correct recall. Every selected GPU returned to its
  exact recorded pre-workload reservation.
- The first endpoint-aware, timeline-only DeepSeek product probe exposed and
  corrected two completion-lifecycle defects rather than hiding them: retained
  stream-edge copies now permit timeline-ordered simultaneous replay without an
  internal host completion, and completed deferred resources retire an observed
  monotonic value before reuse after a downstream epoch join. Exact sequential
  Vulkan tests cover both cases. The runtime Vulkan execution module now contains
  no binary-fence create/reset/wait path and no unbounded device-idle teardown;
  every selected GPU returned to its exact pre-probe reservation.
- With those fixes, four fresh thinking-enabled `hi` conversations completed in
  one mounted DeepSeek process. Decode progressed from 2.47 cold, to 5.14, 7.01,
  and 9.31 tok/s. The fourth sustained windows were 10.02, 9.69, 9.06, and 8.80
  tok/s. The process retained 6,159 expert units / 82.34 GB of payload, but the
  first ten-layer store reached 26.73/27.18 GB while other stores retained 7--13
  GB of headroom, causing 12 evictions and 11 reloads. Static component cost and
  addressable-byte balance therefore remain insufficient: observed per-component
  sparse working-set pressure must participate in boundary selection at safe
  stream/conversation epochs.
- The first authoritative gate of that placement exposed a deterministic host
  completion failure at the recall turn. Device timestamps proved all 484
  recorded dispatches (59,904,077 work units and 22.63 GB of estimated traffic)
  had reached both TOP_OF_PIPE and BOTTOM_OF_PIPE in about 5.29 ms, while the
  reusable binary fence remained unsignaled. No ring timeout, reset, allocation
  leak, or reservation loss occurred. Direct sequence replay is therefore being
  migrated to monotonically increasing timeline completion; reusable binary
  fence reset is not a valid persistent-stream completion protocol.
- Payload-aware transfer pricing passed exact sequential graph, direction,
  missing-measurement, and non-chain-wiring tests plus real thinking-enabled
  gates. Qwen3.6-35B-A3B measured 69.7402 decode tok/s, Qwen3.5-9B measured
  48.7356, and the unchanged explicit DeepSeek control measured 7.8666 decode
  and 7.9348 prefill tok/s after three discarded complete conversations. All
  truth sets had zero loads, misses, reloads, evictions, and residency blocking,
  preserved behavior and recall, and restored every exact pre-run reservation.
- Runtime placement can now learn exact per-component and per-resource hot-set
  pressure from selector activity, distinguish selected/resident/addressable
  bytes without double-counting aliased resources, and evaluate a contiguous
  boundary change only at a completed conversation checkpoint. The objective
  prices calibrated execution, exact graph-derived directional transfers, new
  admission, and the complete hot set lost from every changed physical store.
  Unchanged physical stores retain the same validated store object and residency
  across remount; partial aliases and incompatible mounts are rejected rather
  than silently reused. Exact sequential tests cover positive and negative net
  benefit, partial aliasing, compatibility rejection, retained residency, final
  teardown, and shared-package refusal. Thinking-enabled product gates remained
  healthy: Qwen3.6-35B-A3B measured 69.6690 decode tok/s, Qwen3.5-9B measured
  48.8408, and the explicit fixed-topology DeepSeek control measured 7.8162
  decode and 7.8300 prefill tok/s after three discarded complete conversations.
  The DeepSeek truth set recorded 1,149,648 resident hits and zero loads, misses,
  evictions, reloads, or transfer stalls; behavior, turn recall, and exact NERVE
  allocation release were preserved. Explicit placement never invokes the
  adaptive remounter.

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
     BF16 output bits. Do not retry full-width multi-head workgroups: h2 and h4
     were exact but 17.95% and 58.99% slower at the real geometry. A new blocked
     design must map dimensions, heads, and KV tiles cooperatively enough to keep
     occupancy, as reference flash-attention implementations do. Do not retry
     dimension folding either: local-size 256 preserved the native subgroup
     arithmetic but was 23.71% slower than local-size 512.
     Do not quantize online-softmax weights to BF16 merely to feed a cooperative
     value matrix: that fused kernel won its microcase by 52.9% but changed BF16
     outputs and the real stream fell to 6.424 decode tok/s. Any matrix-tiled
     replacement must preserve the F32 weight path and exact output contract.
     Do not split score production from value consumption through a materialized
     F32 score plane: its score microkernel won by 25.6% at 4K and 53.6% at 128K,
     yet both static and context-bounded complete gates regressed. A replacement
     must eliminate that intermediate and be evaluated by combined attention
     score/read device time before a product gate.
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
     Caching the current demand-gated resident window is likewise exhausted: it
     replayed correctly but remained slower than scalar execution and the
     complete gate fell to 8.1736 tok/s. Change the executed device topology,
     not merely how the losing topology is recorded.
     A terminal-wait-only one-tick demand path was also rejected and removed in
     full. It preserved coherent output and exact teardown, but a repeated turn
     reached only 4.403 decode tok/s with 826 incremental loads; even subtracting
     all measured residency blocking leaves an estimated upper bound near 5.6
     tok/s. It still regrouped the scalar graph into the rejected per-device
     batch topology. Do not revive it as a mode or fallback.
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

    - Rank physical instances inside the same capability class with a fast live
      calibration of the package's actual implementation families. The probe must
      use one fixed warmup and enough useful work to leave transient idle clocks,
      finish in under one minute for the whole candidate set, preserve pre-existing
      reservations, and report representation-specific costs rather than a single
      synthetic FLOP score. Feed those costs into the representation/placement
      fixed point so native MXFP4 work can avoid a device that is only competitive
      for FP8, without changing graph order or introducing tensor parallelism.
    - Optimize the contiguous partition for predicted serial stream latency subject
      to exact per-device capacity and communication cost. Do not equate component
      count, advertised capability, or free bytes with execution cost. Retain the
      smallest device prefix only among placements that do not create avoidable
      paging or a materially slower serial bottleneck.
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
