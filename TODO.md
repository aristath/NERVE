# TODO

## Goal

Compile and run `/mnt/models/models/deepseek-v4/flash-0731/safetensors` as a
daily-driver NERVE model while preserving source behavior and the architecture
defined in `CONCEPT.md`.

The performance target is **17-20 useful decode tokens per second** for the
fully warm, demand-resident DeepSeek conversation workload:

- 17 tok/s is the primary completion target;
- 20 tok/s is the optimization target, not a mandatory artificial stopping
  point; and
- results below 17 tok/s indicate that material work remains unless hardware
  measurements prove a lower physical ceiling.

The target applies to real multi-turn agentic inference, not a synthetic token
loop. The product gate uses 128K context, a 65,536-token output allowance,
package-owned thinking and sampling behavior, complete-conversation warmup,
turn recall, and exact teardown. Useful DSpark-accepted tokens count; rejected
proposals do not.

NERVE must remain model-independent. DeepSeek is the first demanding proof of
the architecture, not a runtime special case. Model-specific structure and
facts belong in the self-contained compiled package. Runtime execution,
placement, transport, residency, representation selection, and planning must
be driven by typed contracts and graph structure rather than model names.

## Architectural direction

NERVE will use **hybrid physical execution islands** beneath its stable logical
execution graph.

A layer remains a standalone logical component with typed ports, state, and
editable wiring. Its selected physical realization may use:

- one device;
- serialized components or regions on different devices;
- tensor parallelism inside selected operations or connected regions;
- whole-expert parallelism;
- tensor parallelism within selected experts;
- replicated hot experts or other resources;
- independently demand-resident cold resources; or
- a measured combination of these strategies.

TP is neither globally enabled nor globally forbidden. The runtime selects it
only where a complete, output-valid measurement shows that it wins for the
actual execution contract, geometry, device group, owner, split, and transport.

The compiler describes legal implementation and partition contracts. It does
not assign physical devices. The runtime resolves those contracts against the
currently selected targets, their reservations and safe remaining capacity,
the local placement calibration, and the mounted graph.

## Current baseline and constraints

- The latest accepted fully warm DeepSeek product gate recorded 8.7166 decode
  tok/s and 8.7574 prefill tok/s with byte-identical responses, zero truth
  loads, and exact reservation restoration. The detailed phase attribution
  below comes from the preceding 8.6600 decode tok/s zero-load truth run.
- The queue-liveness fix now survives the former DeepSeek quarantine point and
  all three Qwen controls, but the latest five-R9700 warmup exposed an
  independent placement failure: its protected cache quotas plateaued near
  122 GiB, then a long reasoning turn thrashed below 1 tok/s while the GPUs
  remained mostly idle. Placement must prove that the expected warm expert
  working set fits the selected cache quotas, or include additional compatible
  targets such as the discrete Intel GPU. A paged mount being admissible is not
  evidence that its steady-state working set is viable.
- That sampled hot decode path spends approximately 20.138 ms in
  hyper-connections, 19.150 ms in attention reads, 17.904 ms in expert gate/up,
  15.123 ms in state commit, 11.840 ms in attention score, 10.081 ms in expert
  down, and 9.805 ms in dense projection. The 17 tok/s target requires about
  58.8 ms per useful token; 20 tok/s requires 50 ms.
- NVMe paging is not the warm steady-state limiter. Lazy expert loading remains
  required for capacity and cold execution, but warm decode optimization must
  focus on the device execution graph and its critical path.
- Equal layer counts and nominally identical GPUs are not equal-cost placement.
  Execution cost depends on the selected implementation, format, geometry,
  device, transport, current reservation, and surrounding execution island.
- `nerve-gpu-bench/placement-benchmark.json` invalidates the old assumption
  that serialization is always preferable on this workstation. Across its 910
  equal-work forced-split comparisons, TP wins 835. On the five R9700s, FP8,
  INT4, and MXFP4 results show that the winning strategy changes with format,
  target group, and participant count.
- The same benchmark also proves that TP is not universally faster. Some pairs
  and formats favor serialization, and the fastest absolute group is often
  smaller than the largest available group.
- Cross-vendor device-local external memory may allocate successfully while
  producing invalid shader results. Transport routes must be output-validated;
  shared-host transport remains a valid measured fallback.
- The August 11 failure reached AMD TTM LRU corruption during a long,
  near-capacity run. Do not resume live inference without explicit
  authorization. The first authorized validation must record every selected
  target's pre-run reservation, stop on the first kernel or driver anomaly, and
  prove exact release of NERVE-owned capacity without disturbing pre-existing
  allocations.
- The normal production mount now derives exact selected-resource execution
  classes from the lowered decode plan, consumes only identity-matching suite
  calibration, and rebuilds activation, parameter, ownership, and physical
  residency plans as one coherent fixed point. Missing, ambiguous,
  phase-incompatible, or capacity-infeasible evidence preserves the compiled
  baseline. Before allocating or entering chat, exact distributed replay now
  reconstructs the selected component transaction from the concretely loaded
  SPIR-V, graph topology, operation and reduction geometry, equivalence,
  activation shape, devices and drivers, endpoints, owner, shards, selected
  resources, and transport routes. Normal chat reports both the mounted
  physical-execution summary and transaction-local distributed submissions by
  phase and physical strategy. When TP proof is requested, the conversation
  gate now requires every completed warmup and measured turn to submit at
  least one actual TP, intra-expert TP, or hybrid island in both decode and
  package-supported prefill. Mounted-but-unused islands, whole-expert-only
  execution, missing counters, inconsistent island classification, and
  impossible shard counts fail closed. This path is covered hardware-neutrally;
  its first real-model proof remains blocked by the live inference quarantine
  above.
- Representation and physical-island selection now terminate in the same
  production mount transaction. Auto-placement preserves the untouched exact
  model at its converged logical placement, the hybrid solver compares exact
  and independently compiler-validated signal representations using their
  source-semantic identities and exact measured physical cases, and the
  implementation catalog reconstructs and validates one non-overlapping,
  complete selection report before any overlay is mounted. Endpoint
  implementations join that same report. Prefill is planned only against the
  representation selected for decode, so phase switching cannot silently
  remount another format. Normal chat consumes this joint result directly and
  binds every eligible auto-placement target before resolving physical
  islands. Hardware-neutral tests cover the complete selection, overlay mount,
  physical-plan validation, duplicate/altered/overlapping application
  rejection, and incompatible-baseline coverage. Real-model TP execution is
  still awaiting the explicitly authorized live gate.
- The fresh self-contained DeepSeek package is 157 GiB, contains 238 compiled
  shaders, passes its exact behavioral proof, and exposes 6,818 single-device,
  258 whole-expert, and 172 intra-expert tensor-parallel contracts. The 172 TP
  contracts cover routed MXFP4 gate/up and down execution in all 43 transformer
  layers, with distinct single-lane decode and multi-lane decode/prefill
  implementations. The 256 routed experts remain a homogeneous independently
  addressable bank while the always-executed native-FP8 shared expert remains a
  separate dense branch. The normal hardware-neutral graph inspector accepts
  the complete package after compiler/runtime selector-identity parity was
  restored. It currently retains the local compiled baseline because no
  identity-matching measured placement calibration selects a TP candidate.
  Therefore the package proves that TP is compiled and mountable, not that an
  inference token has executed through TP; selection, equivalence, performance,
  and teardown remain part of the explicitly authorized live gate.
- Calibration now separates a component-instance-independent compiled
  transaction signature from the exact component contracts used at replay.
  Repeated layers with identical implementation digests, artifacts, operation
  geometry, graph topology, representation dispatch, batching, and phase are
  measured once. Before execution, the selected observation is rebound to the
  current layer's exact contract IDs and selected-resource identities; changed
  implementations, SPIR-V, geometry, graph topology, equivalence, shard shape,
  or expert-fragment layout still fail closed. Production hybrid selection
  expands one observation per physical target to every matching component
  instance before exact replay. The placement-catalog schema is v13 so older
  evidence cannot cross this identity boundary. The bounded
  `calibrate-package` path now includes every exact selected-resource execution
  class and its paired singleton load evidence on every requested participant;
  it publishes nothing when a component candidate is unavailable or any
  selected-resource class is missing. A first real-model TP proof therefore no
  longer requires the broad suite merely to construct a production-consumable
  sparse-component catalog.
- The refreshed workload-free DeepSeek preflight at 128K context,
  package-default seven-token speculation, and demand-paged residency preserves
  all 720 component occurrences across the eight currently detected targets,
  while reducing the measurement surface from 720 instance cases to 96
  structural cases. It exposes 48 decode and 48 width-four causal-prefill
  cases, 64 whole-expert and 64 intra-expert TP candidates across both phases,
  32 representative distributed contracts, and 112 directed boundary cases.
  The former 4,416-case standalone load-wave pass was removed because no
  production placement decision consumed it. Exact singleton load-wave
  evidence remains inseparably paired with every selected-resource execution
  class, while maximum atomic load admission is derived from the compiled
  residency contract. The 43 transformer occurrences per target and phase
  collapse into four exact structural cohorts; input and output adapters retain
  their scalar-lane causal fallback. This dry plan opens no compute device and
  submits no GPU work. It is authoritative preflight evidence, not live TP
  execution or performance evidence.
- The calibration suite now produces canonical serialized and predicted
  mixed-hybrid region evidence from bounded, complete mounted transactions.
  It measures compute, synchronization, transfers, collectives, output, state,
  participant-only memory, and exact teardown together; it never synthesizes
  an outer result by summing component timings. Fully reserved selected targets
  remain in reservation-restoration checks while receiving no placement work.
- All tests and model gates run sequentially. Every Rust test command uses
  `-- --test-threads=1`; Vulkan tests are selected and executed individually.

The following phase budget is a diagnostic guide, not a collection of isolated
microbenchmark targets. Overlap changes the critical path, so the complete
transaction remains authoritative:

| Effective critical-path phase | Current | 17 tok/s guide | 20 tok/s guide |
| --- | ---: | ---: | ---: |
| Hyper-connections | 20.1 ms | 10 ms | 8 ms |
| Attention read and score | 31.0 ms | 18 ms | 15 ms |
| Expert gate/up and down | 28.0 ms | 13 ms | 11 ms |
| State commit | 15.1 ms | 8 ms | 7 ms |
| Dense projection | 9.8 ms | 5 ms | 4 ms |
| Remaining work | about 11.4 ms | about 5 ms | about 5 ms |

## Completion discipline

For every numbered item below:

1. Define its behavioral, residency, placement, and performance acceptance
   criteria before implementation.
2. Add canonical unit and integration coverage, including unhappy paths and
   teardown.
3. Run a fast exact microbenchmark using one warmup and only enough calls to
   answer whether the candidate wins. A microbenchmark taking more than one
   minute is itself a failed design.
4. Run the complete DeepSeek product gate after the candidate passes locally.
5. Run the Qwen3.6-35B-A3B, Qwen3.6-27B, and Qwen3.5-9B regression gates before
   committing a runtime-performance milestone. Maintain a broader
   hardware-neutral compile and contract-smoke matrix for every supported
   architecture so shared runtime work cannot silently become DeepSeek-shaped.
6. Reject and remove candidates that change behavior, regress the complete
   stream, violate placement or residency, destabilize a GPU, or fail to
   restore exact pre-workload reservations.
7. Remove the TODO item only after an adversarial review confirms that every
   acceptance criterion is satisfied.

## Work queue

### 4. Prove dense FFN tensor parallelism on mounted execution

- Hardware-neutral acceptance is complete for the compiler/runtime boundary:
  BF16 and FP8 gate/up output-row shards connect to input-column down shards by
  a participant-private BF16 handoff for immediate decode and package-supported
  prefill; FP8 scale partitions preserve their logical row alignment; F32
  partial collection is accounted at its true byte width; fully distributed
  physical resources exclude redundant canonical owner tensors while a mixed
  canonical use prevents exclusion; a natively supported format without a
  distributed implementation remains local; and successful immediate and
  causal-batch submissions feed the normal per-turn decode/prefill strategy
  counters used by the conversation gate. The remaining acceptance below is
  the explicitly authorized mounted real-model proof.
- On an explicitly selected real transformer component, compare immediate
  decode and package-supported prefill output plus persistent state against the
  canonical single-device execution. Exercise the compiler-declared BF16 and
  FP8 gate/up output-row to local input-column down-projection island; formats
  without a correct distributed contract must remain unavailable rather than
  being relabelled as TP.
- Prove from live allocations that each participant retains only its assigned
  permanent parameter ranges, the owner has no redundant canonical full
  tensors, private intermediates and F32 partials match their declared transient
  accounting, cancellation quiesces every participant, and teardown restores
  each target's exact pre-run reservation.

### 5. Implement lazy whole-expert parallelism

- Cold miss recovery now preserves expert-owner parallelism in both immediate
  decode and multi-lane batch execution. It builds a validated replay schedule
  from the exact faulting shards, prepares every participant before submission,
  submits only affected owners before waiting, uses a fresh timeline value for
  affected helper completion in immediate execution, and runs the coordinator
  exactly once after those helpers. Unaffected resident owners and committed
  graph progress are not replayed. The remaining bullets cover the complete
  mounted transaction, atomic cross-device sharded residency, measured
  intra-expert selection, and online ownership/cache adaptation.
- The structure-driven compiler now separates heterogeneous sparse-expert
  execution cohorts before physical implementation discovery: the routed
  MXFP4 bank receives distinct immediate and multi-lane intra-expert TP
  artifacts for its complete output-row gate/up to local input-column down
  island, while the always-executed native-FP8 shared expert remains an exact
  dense branch and combines with the routed reduction exactly once. This is
  driven by tensor roles and representation contracts, not a model identity.
  Multi-lane execution uses the physical artifacts directly, keeps each
  participant's route-major intermediate at its compact local shard stride,
  writes participant-major F32 partials, and retains the existing whole-expert
  path for non-TP contracts. Shader compilation, contract phase/shape
  selection, fragmented-resource planning, uneven local buffer sizing, zero
  always-selected routed resources, and invalid participant geometry are
  covered hardware-neutrally. The fresh complete package audit is now accepted;
  live equivalence and the measured selection decision remain part of the
  authorized mounted proof below.
- Hardware-neutral execution acceptance now validates the complete sparse
  chain at normal package startup: exactly one structural router feeds each
  routed gate/up, the matching routed gate/up and down form one distributed
  island, the always-selected shared expert is a separate execution cohort,
  and exactly one coordinator-local reduction/add sequence combines their
  outputs. The scheduler keeps router stages before the island and the
  reduction after it, while distributed submission launches every owner shard
  before waiting or running its coordinator. The remaining acceptance is the
  explicitly authorized mounted proof below.
- Prove on the mounted real-model transaction that the router executes once on
  the layer coordinator and routing metadata stays on the device.
- Prove that the six selected routed experts dispatch concurrently to their
  owners. Each expert's gate/up, activation, weighting, and down projection
  must remain on the same device so its intermediate never crosses a device
  boundary.
- Prove that the shared expert executes concurrently when dependencies allow,
  then routed and shared expert results reduce exactly once on the selected
  coordinator.
- Prove on mounted decode and multi-lane prefill that each expert remains
  independently demand-resident: an unavailable expert publishes an immutable
  fault at the exact causal checkpoint, only affected shards resume, and
  resident experts plus already committed graph progress are not replayed.
- Hardware-neutral atomic residency acceptance is complete. The mounted plan
  now records the exact physical fragments of each tensor-sharded expert;
  immediate decode and multi-lane prefill resolve all fragment faults through
  one shared coordinator; no gate is acknowledged until every required load
  succeeds; failure rolls back only transaction-owned fragments; and pressure
  eviction closes transitively over every cohort sharing a physical group,
  atomically retires all physical directory members, then clears each store's
  publications and tier accounting. Co-located logical shards remain ordinary
  local residency rather than ceremonial distributed cohorts. The remaining
  acceptance is the explicitly authorized mounted proof that load failure,
  eviction, cancellation, and teardown preserve this invariant in live Vulkan
  execution and restore exact pre-run reservations.
- Prove and separately measure the compiler-declared intra-expert TP candidate,
  including its dynamic output-row gate/up to local input-column down-projection
  batch path. Do not shard every expert merely because the mechanism exists.
- Use marginal expert frequency and joint co-selection telemetry to place and
  replicate hot experts. Optimize concurrent per-device expert makespan, not
  the sum of six independent expected costs.
- Feed warm session selection and co-selection telemetry back into expert
  ownership, replication, and cache-quota planning. The initial production
  mount now consumes exact calibration but uses a uniform prior; warm
  adaptation must not replace the stable layer coordinators or dense/attention
  execution islands, remount the model, or disable hybrid physical execution.
  The hardware-neutral reconfiguration planner is complete: it validates exact
  selector, telemetry, execution-class, device, phase, and capacity identities;
  rescoring never trusts stale plan summaries; joint telemetry influences the
  proposed makespan; and every move reports its destination load cost plus the
  exact cold break-even activation count. Residency gates now derive arithmetic
  ownership from the concrete dispatch shard rather than the physical store's
  broader addressability. Demand-loaded mounts now also build an explicit
  whole-expert addressability envelope across the current participants while
  retaining exact per-shard execution ownership in their parameter-slot tables;
  tensor-fragment projections remain fixed physical contracts. The mounted
  package exposes the two plans separately, and refuses an execution projection
  that exceeds or changes its store mapping. The stream-local
  quiescent-boundary executor is also complete: it changes every accepted
  selector in one rollback-safe transaction, updates both arithmetic parameter
  tables and residency-gate ownership without remounting the backbone, keeps
  addressable replicas from executing duplicate arithmetic, and lets cloned
  streams inherit adapted ownership without leaking changes into sibling
  streams. Package-wide cache-quota arbitration is now complete as well: each
  stream publishes a recent telemetry window, the package aggregates weighted
  demand and the union of hot resources across every live stream, unrelated
  selector shares remain fixed, and complete per-store eviction policies swap
  transactionally. A failed quota publication restores both the previous
  shared policy and any ownership change from the same prompt boundary. The
  remaining work is measured replication and the explicitly authorized mounted
  proof.
- Allow a compiler-declared predictable router dependency to trigger safe
  prefetch or preselection without a DeepSeek-specific runtime branch.

### 6. Build the hybrid placement and scheduling optimizer

- Enumerate legal single-device, serialized, expert-parallel, TP, replicated,
  demand-resident, and hybrid execution-island candidates from the mounted
  graph and compiled contracts.
- Treat every detected compatible target as eligible unless the caller
  explicitly excludes it. Current allocations are reservations: inspect each
  target immediately before planning and use only its measured safe remaining
  capacity.
- Per-component physical compatibility is complete for the serialized
  backbone candidate graph. Runtime discovery now distinguishes source
  components from mounted instances, calibrates only exact execution
  signatures supported by each target, permits a signal-only target to serve
  as an interior device without pretending it can host the input/output graph,
  and requires the union of selected targets to cover every component.
  Placement treats the measured execution matrix as sparse, rejects illegal
  component/device assignments, compares every target subset instead of only
  ranked prefixes, and retains a feasible demand-paged fallback when no larger
  legal subset can remove its addressable-capacity shortfall. Partial targets
  are ranked by normalized measured component cost so a device is not made to
  look artificially fast merely because it supports fewer operations. The
  same compatibility boundary now feeds representation selection per mounted
  runtime instance: an alternative needs to cover only the exact instances
  whose baselines cannot execute on their assigned targets, while compatible
  neighbors retain their exact implementation. The editor and production
  runtime use one shared shader/capability validator and require a profile for
  every assigned owner or shard. Representation candidates and local measured
  TP/serialized/hybrid physical cases now share one semantic candidate graph
  and one canonical mount transaction. Whole-expert, replicated-resource, and
  general DAG candidate families remain to be incorporated into that same
  solve without reintroducing whole-model device assumptions.
- The ordered-graph solver now retains mounted bytes and the maximum execution
  transient as separate per-device and host dimensions. Capacity is checked
  against the final union of mounted regions plus every earlier transient, and
  Pareto pruning cannot discard a slower route merely because its lower
  transient pressure was previously invisible. Replace calibration-session
  aggregates with exact permanent, KV/state, cache-quota, and atomic load-wave
  claims derived for the requested context, speculation, residency policy, and
  lowered physical plan. The solver contract and production candidate graph
  now carry those typed dimensions and immutable claim identities; shared
  claims deduplicate only when their definitions match, cache waves must fit
  the admitted quota, and different claim sets remain distinct on the Pareto
  frontier. The representation-aware production resolver no longer treats
  bounded component-calibration memory as final admission or as a route-pruning
  dimension. It lazily visits complete decode and owner-compatible prefill
  routes in nondecreasing predicted-duration order with no arbitrary candidate
  cap, lowers each route, and runs the ordinary package's exact slice,
  distributed parameter/exclusion, activation, selected-resource ownership,
  cache-quota, atomic load-wave, and physical-residency planners at the
  requested context, speculation, residency policy, and live reservation
  envelope without opening Vulkan or allocating VRAM. An infeasible fast route
  is skipped in favor of the next complete route even when bounded calibration
  data made that route look resource-dominated; exact sequential tests cover
  both representation/decode and prefill route discovery. Normal mount repeats
  the same centralized derivations and remains the final race-safe admission
  gate. Exact immutable parameter claims now move ahead of the ordered solve:
  every eligible local, TP, and measured-region candidate is replayed
  workload-free against its concrete compiled descriptors and physical shards
  before route search. Historical calibration candidates for currently
  ineligible devices are excluded before replay, while inconsistent
  selected-device bindings fail closed. Raw local full allocations and
  arbitrary overlapping distributed fragments are canonicalized together
  across the complete candidate graph, including representation alternatives,
  into shared physical byte-range blocks with exact target identity and
  deterministic claim IDs. This lets exact parameter infeasibility prune
  partial routes without double-charging aliases or full/fragment overlap.
  Sampled calibration transients are replaced by workload-free physical
  geometry. Candidate-local execution-transient claims remain advisory because
  graph-wide activation liveness aliases buffers across component boundaries;
  treating those lower bounds as additive during partial search would reject
  valid routes. Every complete prefill route is instead replayed into its exact
  owner slices and distributed execution plan before acceptance. That replay
  accounts for activation liveness and retained speculative source taps, shared
  and private distributed activations, F32 reduction planes, host edge staging,
  stream/token/control buffers, demand-residency gates, and the simultaneously
  cached normal-prompt and causal-verification runners. The complete-route
  transient reservation is authoritative and a one-byte-short route is
  rejected. Input-column and physical output-row TP use their compiler-declared
  multi-lane artifacts directly and therefore do not inherit nonexistent batch
  control allocations. Workload-free physical mounting independently
  reconstructs the same fixed claims from local prepared descriptors, exact
  replayed TP fragments, non-dispatch transducer parameters, every mounted
  phase, and compiled resource identities. Compiler-generated physical layouts
  retain their own immutable storage-range identity, so they cannot be
  incorrectly deduplicated against their canonical source tensor. Exact
  mutable-state claims now precede route search as well. They are derived from
  component-scoped physical stream-state layouts at the requested context and
  speculation width, including transactional and causal-verification storage,
  activation slots, true model boundaries, same- and cross-device graph-edge
  allocations, shared stream control, output/sampler/feedback workspaces, and
  speculative-decoder state. Replayed distributed candidates add their exact
  shared activation, reduction, and participant-private intermediate backing;
  shared-host routes charge host capacity while device-local routes charge the
  concrete owner. Stable allocation identities deduplicate only the same edge,
  boundary alias, stream-control allocation, or distributed buffer. A
  candidate cannot inherit unrelated components' state merely because the
  workload-free planning copy co-locates the rest of the graph on its owner.
  Exact selected-resource cache and load-wave claims now precede route search
  too. Dynamic compiler-selected tensors are excluded from immutable parameter
  claims and charged exactly once through the store that owns them. The planner
  combines local and distributed selector ownership per physical participant,
  reconstructs projected TP fragments, exact address slots, address/parameter
  tables, double-buffered transfer staging, and allocation padding. Retained and
  eager policies reserve the union of unique source-payload slots; demand-paged
  routes reserve the true largest legal selector wave. That wave is calculated
  from the largest actually owned groups rather than the conservative product
  of one maximum group and the route count, and must fit inside its admitted
  quota. Adaptive derived representations share that physical quota while the
  mandatory source bytes remain available for construction and restoration.
  Terminal shared-host cache capacity now excludes the stream's own shared-host
  reservation instead of double-spending it. Permanent, mutable-state,
  cache-quota, and atomic-load-wave capacity are authoritative during partial
  route search. Execution transients become authoritative only at complete-route
  admission; their candidate-local projections remain advisory.
  Exact sequential tests compare a complete local route with the ordinary
  workload-free residency plan, prove component scoping and same-device edge
  deduplication, exercise device-local/shared-host distributed backing, and
  reject a candidate one byte below its exact retained need. Terminal physical
  mounting and the real package loader now reconstruct the selected route's
  complete prompt and causal-verification runners from the final replayed
  execution plans. Selected-resource placement and transient sizing iterate to
  a fixed point, so moving resource gates cannot leave stale participant
  memory. The resulting device and shared-host bytes reduce cache capacity,
  augment the authoritative mount and per-stream residency plan, and are
  checked again against live capacity. The mounted normal-prompt runner is
  capped at the same lane width that was calibrated and admitted; it can no
  longer grow opportunistically after placement. Physical store headroom uses
  the augmented per-stream plan rather than the older local-only working-set
  estimate. Exact terminal tests prove the selected prefill width contributes
  nonzero residency and that one byte below the complete mount plus stream
  requirement is rejected.

  Local and otherwise unmeasured fallback no longer escapes this contract. The
  mount derives legal power-of-two widths from the compiled causal-batch
  artifacts and recorded-command budget, tries them widest-first against the
  same selected-resource/transient fixed point, and records the winning width
  in both the workload-free mount and real package. A one-byte-short widest
  plan downshifts instead of failing or mounting an unaccounted runner; capacity
  below the scalar contract fails closed. Multi-stream admission is now one
  atomic, deterministically ordered physical-device and process-wide shared-host
  reservation transaction. Each device-local or shared-host Vulkan allocation
  consumes a split child permit from the active stream transaction at its
  queried memory-requirement size; committed reservations follow buffer
  lifetime, unused prompt and
  verification credit remains attached to the stream, nested constructions
  cannot borrow an outer stream's participants, and any undeclared allocation
  fails closed. Shared-host reservations use the falling live MemAvailable
  snapshot without double-charging already tracked allocations. Exact
  sequential tests cover split accounting, competing transactions, nested
  isolation, partial commit, rollback, and teardown.

  Prefill and speculative-verification shared-host transients now retain an
  allocation-level ledger from runner planning through physical residency and
  terminal admission. Every signal, reduction, and staged boundary records its
  logical owner, complete participant set, capacity, and concern; malformed or
  out-of-plan participants fail atomically. Terminal admission queries each
  concrete Vulkan allocation separately, so memory-type alignment is no longer
  lost by summing logical capacities first. Device-local prompt and
  verification transients now use the same contract: individual signal,
  private activation, reduction, per-lane stream control, token/control,
  causal snapshot, selected-resource gate, miss-queue, and predicate buffers
  survive into physical residency, and terminal admission queries each Vulkan
  requirement separately. The logical totals are derived and cross-checked
  against those ledgers before reservation. Resident stream state, state
  transactions, causal-verification snapshots, selection telemetry, activation
  slots, and nonaliased model boundaries now retain exact allocation identities
  too. Distributed activation and boundary storage replace their matching local
  allocation exactly once, rather than leaving both charged, and terminal
  admission queries every retained Vulkan allocation separately. Empty,
  repeated, missing, capacity-mismatched, and identity-mismatched ledgers fail
  closed. This also closes an omitted-residency defect for physically allocated
  selection telemetry. Resident graph edges now preserve the mount's actual
  allocation topology as well: local and outgoing fan-out from one produced
  port shares one ledger entry, incoming storage remains distinct, and a
  boundary-input passthrough adds no second allocation. Those identities are
  now rebound to the exact selected physical route before terminal admission.
  An external-device-local produced port owns one exportable allocation on its
  physical source; a staged port owns one device-local allocation per physical
  participant plus one shared-host allocation; and colocated logical endpoints
  collapse to the source allocation. Deferred distributed-edge participants are
  included without charging a provisional activation allocation, conflicting or
  missing routes fail atomically, and planning and mounting use the same pure
  physical-route resolver. Terminal admission queries every external and staged
  allocation separately at its exact Vulkan requirement rather than rounding a
  logical aggregate. Output-transducer and main-sampler buffers now retain
  stable allocation identities too, including history/output, random scratch
  and seed, token state/snapshot/batch, and runtime parameters. This closes a
  `temperature_top_p` omission in the old sampler aggregate. Feedback control
  and speculative target-frame history also have distinct identities; the
  control allocation stays device-local for one physical target and moves to
  one shared-host allocation for a multi-target stream. Parallel-only
  speculative decoders no longer reserve an autoregressive target-frame
  history they never allocate. Feedback-control capacity now comes from the
  exact final decode transaction: local dispatches exclude distributed
  replacements, local and per-shard residency gates are counted from their
  actual store ownership and policy, distributed island dispatches are counted
  per participant, and the input, output, and selected sampler dispatches are
  explicit. The materialized loader independently reconstructs the registered
  dispatch count and fails if its allocation differs from the workload-free
  plan. Every currently planned speculative-decoder scalar state, transaction,
  telemetry, activation, boundary, edge, stream-control, output, sampler, and
  auxiliary buffer now retains both its decoder scope and allocation identity.
  This exposed and fixed two undercounts: draft selection telemetry was absent,
  and autoregressive execution reserves two alternating pending target-hidden
  buffers rather than one. Autoregressive catch-up now retains one canonical
  runner at the configured target-window lane class instead of one heavy bank
  per observed width and source. Its copy-command binding uses the stable Vulkan
  device/buffer identity, is rebuilt when the source or frame geometry changes,
  is explicitly invalidated before a temporal source runner is replaced, and is
  dropped before the stream's source buffers during teardown. Its device-filtered
  component-batch allocation plan plus the batched embedding token/control
  buffers are now attached to the terminal physical mount before admission, and
  lazy construction enters that pre-reserved stream transaction. Demand-resident
  autoregressive catch-up stays on the serial path and therefore reserves no
  unused batch bank. Exact hybrid prefill planning also separates the active
  calibrated width from the power-of-two runner allocation capacity, so an
  odd-width selection cannot under-reserve its buffers. Parallel speculative
  proposal and committed-context runners now share one structural scope
  derivation with their materialized processors, and their per-allocation
  component-batch ledgers, packed output readback, and cross-physical-device
  source-tap staging are part of terminal admission. Co-located logical devices
  correctly allocate no ceremonial staging. The two cached temporal
  state-ingestion lane classes—normal prefill and causal verification—now also
  contribute their exact node-scoped component-batch allocations, per-runner
  demand predicates, and cross-physical-device batched source staging. Their
  active widths and rounded capacities match the target runners they accompany.
  Resident-feedback state ingestion is now conditionally admitted only after
  the mounted topology has passed the authoritative replayability gate. One
  atomic per-stream reservation covers every node-scoped causal runner
  allocation and each device-local source history at the mounted feedback
  window capacity; the allocation plan is built before allocation, the buffers
  consume only that scoped reservation, and materialization independently
  reconstructs and compares per-device totals. The eager feedback transaction
  now also fails closed unless every exact physical device/host permit byte is
  consumed, so a requirement/alignment mismatch cannot survive as unexplained
  credit. An ineligible feedback loop or a stream with no parallel decoder
  reserves nothing. Main-stream physical admission is now partitioned into
  permanent, prompt-runner, verification-runner, and catch-up-runner classes.
  The permanent class is reconciled after the base mount; prompt,
  causal-verification, verification-state, and serial catch-up allocations are
  reconciled when their complete canonical runner class is first materialized.
  No class may borrow another class's device or host credit. Deferred classes
  use reusable admission leases: releasing, replacing, or rolling back a cached
  runner atomically converts its tracked device/host bytes back into the owning
  stream's same-class pending credit, while permanent allocations remain
  one-shot. Failed construction, unexplained credit, and non-exact reusable
  commits tear down transaction-owned caches and return their complete credit.
  Unique scope identities preserve the innermost allocation class even under
  out-of-order teardown. Exact sequential tests cover all four class mappings,
  class isolation, nested and out-of-order scopes, partial commits, recycling,
  physical ledgers, prompt/verification state-ingestion separation, permanent
  speculative runners, and catch-up classification. The remaining completion
  is the explicitly authorized live proof of physical requirement consumption,
  output/state equivalence, and teardown on local, staged, speculative, and TP
  model transactions.
  The stream-control allocation now follows its actual physical memory domain:
  logical slices aliased to one physical device retain one device-local charge,
  while a multi-device stream replaces every imported-device charge with one
  shared-host allocation. Admission and mounting select the same deterministic
  physical owner, and malformed, duplicate, or incomplete bindings fail
  atomically. Provisional distributed graph-edge buffers no longer contribute a
  hidden mount peak: their generic route is left
  unmaterialized, boundary mounting installs the selected final physical route
  once, and finalization rejects missing, extra, or substituted participant
  buffers before dispatch construction.
  On the authorized live gate, prove that admitted device/host totals are
  consumed exactly—neither exhausted early nor left as unexplained credit—on
  local, staged, direct device-local, speculative, and TP stream shapes. Exact
  sequential hardware-neutral tests also cover source
  projections, uneven selector waves, paged versus retained/eager quotas,
  selected-resource exclusion from permanent graph fallback, insufficient
  parameter and cache capacity, shared-host cache/stream isolation, unchanged
  non-parameter classes, malformed bindings, ineligible historical candidates,
  representation/decode and prefill discovery, joint mount, local full-route
  totals, and four-participant TP fragment totals.
- Optimize **scheduled critical-path time**, not a simple sum of operation
  durations. Model compute and transfer queues, dependency edges, collectives,
  independent expert branches, resource contention, and legal overlap.
- Use a hierarchical solve:
  1. place the persistent backbone and layer coordinators;
  2. assign expert banks, replicas, and lazy cache quotas;
  3. choose local physical implementations and shard groups; and
  4. construct the resource-constrained execution schedule.
- Use Pareto/resource-constrained dynamic programming for the canonical ordered
  graph while retaining a general DAG/island interface for custom NERVE wiring.
  Never discard a partial plan solely because it is locally slower when it has
  different remaining capacity, output placement, residency, or future routing
  options.
- Optimize warm decode, prefill/TTFT, and cold/miss behavior as separate measured
  objectives. Do not hide severe miss latency inside one average or remount the
  model merely to switch phases.
- Replan when reservations, selected targets, graph wiring, or implementation
  contracts change. Expert hotness may adapt online without rebuilding the
  stable backbone plan per token.
- Reject a demand-paged candidate whose measured warm working-set estimate
  exceeds its aggregate per-device cache quotas when another compatible target
  can make that shortfall avoidable. Capacity admission and steady-state
  working-set viability are separate proofs.

### 7. Make hybrid execution device-owned

- Make a resident sparse-expert hit continue directly into expert execution
  without a host predicate read, fence, or terminal wait. Only a real miss may
  stop the bounded transaction and publish a fault record.
- Make cross-device edges, TP fan-out/collection, expert dispatch/reduction,
  and arbitrary ordered visits such as `gpu0 -> gpu1 -> gpu0` part of the
  compiled transaction through persistent activation rings and timeline
  dependencies.
- Preserve independent compute and transfer engines. Do not serialize them into
  one primary command buffer or disguise the same serial topology as multiple
  submissions in one host call.
- Put ticks, token IDs, dispatch dimensions, router results, expert addresses,
  state, sampler control, stop/cancel flags, causal frontiers, and commit records
  in device-resident control buffers consumed through predicates and indirect
  dispatch.
- Prove that the fully resident hot path scales host submissions and waits with
  bounded windows and physical execution streams—not tokens, layers, selected
  experts, shards, or graph nodes.
- Preserve bounded watchdog and failure handling outside the hot path. A failed
  worker is quarantined at the last causal checkpoint without corrupting other
  devices, streams, or the UI.

### 8. Reduce the remaining non-expert critical path

- Add measured execution-island candidates for the four-way hyper-connection
  structure. Evaluate branch parallelism, local fusion, and selective TP while
  preserving Sinkhorn, reduction, and residual ordering.
- Treat attention as its own partition family. Define legal query/head,
  indexer, latent-state, KV/state, output-projection, and reduction contracts
  rather than reusing the FFN partition rules.
- Improve attention read and score as one complete transaction. Do not
  materialize a full F32 score plane or split score and value into independently
  scheduled kernels when the complete stream loses.
- Reduce state-commit cost by publishing only authoritative causal changes
  through the persistent transaction. Do not reintroduce full-state baseline
  copies, clean replay, or hot-path completion polling.
- Re-evaluate dense projections inside the surrounding island so local outputs
  can feed their next consumer without unnecessary publication or conversion.

### 9. Make temporal prefill a true multi-token transaction

- Hardware-neutral calibration planning now treats compiler-declared causal
  batch artifacts and scalar-per-lane adapters as one complete prefill
  transaction. It keeps widths that select different artifacts in distinct
  identities, rejects incomplete causal-scan widths, discovers exact hybrid
  cohorts from the catalog rather than a decode-only surrogate signature, and
  produces complete DeepSeek width-four component, distributed, boundary, and
  selected-resource execution cases. Each selected-resource case retains its
  exact singleton load evidence. Mounted equivalence and performance remain to
  be proven.
- Execute prompt blocks through the same resident gates, execution islands,
  transfers, attention updates, and terminal completion without a host loop per
  token.
- Choose block width from exact context geometry, transient-state capacity,
  residency headroom, and measured hardware behavior.
- Verify every causal state transition against scalar prefill and report TTFT,
  prompt throughput, and decode separately.

### 10. Rebuild DSpark on the hybrid persistent engine

- Use the package-discovered `parallel_backbone_markov` contract and its legal
  proposal widths; do not identify DSpark through a model-name branch.
- Fuse proposal, target verification, confidence-prefix comparison,
  accepted-state selection, commit/rollback, and draft catch-up into a
  device-owned multi-lane transaction.
- Let DSpark exploit otherwise idle execution islands and GPUs without slowing
  the scalar target critical path.
- Promote it automatically only when proposal state, accepted prefix, routed
  experts, sampler state, final stream state, and generated behavior pass the
  canonical gate and useful committed tokens per complete cycle improve.
- Report proposal throughput, acceptance distribution, verification cost, and
  useful tok/s. Do not count generated-but-rejected draft tokens as throughput.

### 11. Complete capability-driven representation selection

- Preserve the native source representation whenever it is supported and wins
  on the assigned target; prove this and incompatible-baseline replacement on
  real mounted Qwen and DeepSeek components once live validation is authorized.
- Add alternative structured INT4, FP8, INT8, FP16, BF16, or other formats only
  through compiler-emitted legal contracts and behavioral-equivalence evidence.
- Select representation and placement together. Account for whole-island
  execution, conversion boundaries, resident footprint, lazy reload traffic,
  transient peak, and current headroom—not an isolated kernel or advertised
  TOPS figure. The current joint candidate graph uses measured complete-region
  execution and capacity vectors; conversion/lazy-reload costs and
  representation-dependent steady-state cache quotas must still become direct
  optimizer resources rather than remaining only compiler-selection metrics.
- Reuse the mechanism for unseen compatible models; no DeepSeek, Qwen, vendor,
  or device-name branches belong in runtime selection.

### 12. Complete the product and regression gate

- Extend the generic product gate with a representative tool-call round trip,
  malformed-output rollback, cancellation, repeated residency faults, eviction,
  long-stream continuity, and turn recall.
- Test single-target, serialized multi-target, TP, expert-parallel, and selected
  hybrid plans with equivalent model work. Multi-target measurements include
  computation, synchronization, transfers, and collectives.
- Warm DeepSeek with complete conversations until one full conversation has zero
  residency loads. Discard all warmup timings and measure the following complete
  conversation without unloading or remounting the model.
- Require coherent answers, package-owned thinking and sampling, identical
  accepted behavior and state digests where exactness is required, and exact
  post-run restoration of every selected target's pre-workload reservation.
- Require at least 17 useful decode tok/s, continue optimizing toward 20 tok/s,
  and investigate any regression before accepting a milestone.
- Run equivalent fully warmed Qwen3.6-35B-A3B, Qwen3.6-27B, and Qwen3.5-9B
  conversations before each runtime-performance milestone is committed.
- Before final completion, compile and run the generic product smoke gate for
  every downloaded architecture NERVE claims to support, except a model the
  caller has explicitly excluded. Unsupported structures must fail through a
  typed missing contract, never a model-name branch or silent fallback.

### 13. Perform the final adversarial review

- Verify the implementation against `CONCEPT.md` and the goal at the top of this
  file.
- Confirm that the compiled package remains self-contained and model-specific,
  while compiler discovery, runtime operators, physical execution islands,
  graph wiring, placement, transport, representation, residency, and stream
  transactions remain reusable by unseen models.
- Confirm that no global serialization, global TP, fixed device count, vendor
  assumption, model-name branch, format-only ranking, or synthetic benchmark
  shortcut controls production placement.
- Confirm that the 17-20 useful tok/s result is from the normal chat path with
  the required warmup, context, output allowance, thinking behavior, recall,
  safety checks, and exact teardown.
- Remove completed work from this file. Completion means the work queue is empty,
  tests and product gates pass sequentially, the owned worktree is clean, and all
  accepted milestones are committed and pushed.

## Rejected shortcuts and non-regression guardrails

- A faster isolated shader is not promotion evidence. The complete routed stream
  must remain behaviorally correct and faster.
- Do not reconstruct TP cost from advertised compute or independent bandwidth
  numbers. Measure the complete executable contract.
- Do not replace the old global-serialization assumption with global TP.
- Do not make every expert tensor-parallel. Prefer whole-expert concurrency when
  it wins, then selectively shard only the experts or shared paths that benefit.
- Do not treat successful allocation or external-memory import as proof of valid
  cross-device execution.
- Do not serialize independent compute and transfer work into one queue merely
  to reduce submission counts.
- Do not add host-visible expert weights, full-state snapshots, clean token
  replay, full-score materialization, or new hot-path polling surfaces.
- Do not force DSpark when accepted useful tokens per cycle lose to scalar chat.
- Do not optimize for an arbitrary token cap, short context, disabled thinking,
  synthetic text generation, cold-run timing, or benchmark-only success.
- Do not keep rejected experimental modes or compatibility fallbacks in this
  unreleased codebase.
