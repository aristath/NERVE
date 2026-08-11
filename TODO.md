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

### 1. Define typed physical execution contracts

- Introduce a model-independent contract identifying:
  - operation or connected-region family;
  - compiled artifact and implementation digest;
  - decode or prefill phase;
  - physical storage, compute, and accumulation formats;
  - exact geometry or representative compiler-emitted shape class;
  - parameter partition dimension and aligned shard boundaries;
  - replicated, sharded, or routed inputs;
  - concatenated, reduced, routed, or locally retained outputs;
  - legal local-intermediate flow;
  - persistent, transient, KV/state, and lazy-resource requirements; and
  - canonical numerical and state-equivalence requirements.
- Make the compiler emit legal contracts and alternate implementations without
  selecting devices or embedding workstation policy in the package.
- Fail closed when the selected artifact does not declare a valid partition
  contract. Do not infer distribution from operation names, descriptor counts,
  tensor names, or model-level dtype.
- Use the same typed contract definitions in the compiler, runtime, benchmark,
  and tests.

### 2. Add a resolved physical execution-island plan

- Keep the logical runtime graph and its standalone layer/node boundaries
  intact. A physical island implements one node or a compatible connected
  region without changing observable ports, state, duplication, bypass, or
  rewiring semantics.
- Replace the insufficient owner-plus-`component_shard_devices` description
  with a resolved plan that records, per node instance or region:
  - implementation contract;
  - entry and exit targets;
  - participant devices and roles;
  - owner/coordinator;
  - shard ranges or expert ownership;
  - transport and synchronization routes;
  - phase-specific execution schedule; and
  - exact residency and transient-memory requirements.
- TP participants are physical implementation roles, not new logical graph
  nodes. Expert resources remain internal independently selectable groups, not
  hundreds of peer layer components in the public graph.
- Permit a decode and prefill schedule to differ while sharing parameter
  residency when doing so does not remount tensors or invalidate stream state.

### 3. Make placement calibration safe for runtime consumption

- Keep the existing placement artifact as evidence only until its schema
  preserves exact execution-case identity. Do not consume its current
  format-only ranking directly in automatic placement.
- Make benchmark cases execute the same artifacts and partition contracts that
  inference can select. Synthetic data may surround the real contract, but the
  benchmark must not contain an approximate second implementation of TP.
- Preserve every valid non-dominated candidate rather than only the locally
  fastest owner or transport. A slightly slower local owner may remove the next
  graph boundary and produce the fastest complete plan.
- Key observations by contract, phase, shape class, device group, split,
  input target, output target, owner, transport, and relevant driver/artifact
  identity.
- Measure representative compiler-emitted geometries rather than one arbitrary
  payload or an expensive size sweep. One warmup and one or two timed calls are
  sufficient to decide whether a candidate is faster.
- Measure complete single-device, serialized, TP, expert-parallel, hybrid,
  directed-boundary, reduction, and lazy-load-wave transactions. Include all
  computation, synchronization, transport, and collection needed before the
  output is usable.
- Do not hardcode a four-device architectural limit. Use staged candidate
  expansion: measure singles and pairs, expand promising groups, and directly
  validate every final group the planner may select.
- Record only canonically output-valid observations. Missing or stale
  observations make that candidate unavailable rather than free or assumed.

### 4. Complete the dense FFN tensor-parallel substrate

- Implement and validate the complete gate/up-to-down island:
  1. Publish or distribute the normalized hidden input.
  2. Compute aligned gate/up output-row shards.
  3. Keep every activated intermediate shard local.
  4. Consume the matching down-projection input-column shard locally.
  5. Produce full-width F32 partial outputs.
  6. Reduce once on the selected coordinator.
  7. Convert and add the residual exactly once.
- Ensure every participant stores only its assigned permanent tensor ranges;
  the owner must not retain a redundant full tensor.
- Support decode and prefill through artifact-declared BF16, FP8, MXFP4, INT4,
  and other valid representations. A format without a correct distributed
  contract remains single-device or serialized.
- Validate immediate component output, persistent state, parameter residency,
  transient peak, transport, cancellation, and teardown against the canonical
  single-device execution.

### 5. Implement lazy whole-expert parallelism

- Execute the router once on the layer coordinator and keep routing metadata on
  the device.
- Dispatch the six selected routed experts concurrently to their owners. Run
  each expert's gate/up, activation, weighting, and down projection on the same
  device so its intermediate never crosses a device boundary.
- Execute the shared expert concurrently when dependencies allow, then reduce
  routed and shared expert results exactly once on the selected coordinator.
- Keep each expert independently demand-resident. An unavailable expert may
  publish an immutable fault at the exact causal checkpoint; resident experts
  and already committed graph progress must not be replayed.
- Use atomic residency groups for a tensor-sharded expert. It is runnable only
  when all required fragments are resident, and eviction must never leave a
  partially resident unusable expert.
- Add optional intra-expert TP as a separately measured candidate. Do not shard
  every expert merely because the mechanism exists.
- Use marginal expert frequency and joint co-selection telemetry to place and
  replicate hot experts. Optimize concurrent per-device expert makespan, not
  the sum of six independent expected costs.
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
- Attach exact permanent, transient-peak, KV/state, cache-quota, and atomic
  load-wave byte vectors to every candidate.
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
  on the assigned target.
- Add alternative structured INT4, FP8, INT8, FP16, BF16, or other formats only
  through compiler-emitted legal contracts and behavioral-equivalence evidence.
- Select representation and placement together. Account for whole-island
  execution, conversion boundaries, resident footprint, lazy reload traffic,
  transient peak, and current headroom—not an isolated kernel or advertised
  TOPS figure.
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
