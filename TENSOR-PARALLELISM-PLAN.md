# Tensor Parallelism Plan

## Purpose

This document records the current codebase findings and a proposed path for
making tensor-parallel execution a measured, local placement option rather than
a global assumption.

The goal is not to replace serialized or contiguous component placement. The
goal is to let the runtime decide, from evidence, when a component or dispatch
should run as:

- serialized component/layer placement;
- internally sharded execution across a small GPU set;
- expert-range placement;
- replicated hot resources;
- demand-loaded cold resources; or
- a hybrid of the above.

## Current Finding

The codebase already contains more distributed execution infrastructure than a
simple serialized runtime.

Runtime placement supports ordinary component ownership plus optional internal
component shard pools. The CLI exposes the participants through
`--shard-component`. A bounded manual proof may pair that pool with
`--physical-strategy` to select one compiler-declared `tensor_parallel`,
`expert_parallel`, or `tensor_parallel_expert` family. The runtime resolves a
unique maximal complete contract set independently for immediate decode,
multi-lane decode, and prefill; missing or ambiguous phase coverage fails
before model allocation. `--inspect-graph` reports the resolved contract IDs
without entering inference. These manual contracts remain explicitly
unmeasured and cannot masquerade as automatic placement calibration.

Relevant code:

- `runtime-rs/src/bin/nerve_runtime/args_parsing.rs`
- `runtime-rs/src/stream_circuit/runtime_graph/placement_spec.rs`
- `runtime-rs/src/vulkan_stream_circuit/runtime_physical_execution_plan.rs`

The Vulkan loader already lowers those shard pools into:

- distributed execution plans;
- distributed activation buffers;
- distributed parameter shard allocation;
- distributed parameter exclusion from the normal full-tensor slice; and
- shard-aware dispatch runners.

Relevant code:

- `runtime-rs/src/vulkan_stream_circuit/placed_model_package_loader.rs`
- `runtime-rs/src/vulkan_distributed.rs`
- `runtime-rs/src/vulkan_distributed/planning.rs`
- `runtime-rs/src/vulkan_distributed/dispatch_runners.rs`
- `runtime-rs/src/vulkan_distributed/parameter_exclusions.rs`

This means hybrid tensor/expert parallelism is not a new runtime category. It
is already partially present as explicit component-local distributed execution.
However, automatic benchmark-backed selection is not the only missing part.
The existing implementation proves a narrow output-row path, but it does not
yet provide complete transformer-block TP, distributed lazy experts, general
format coverage, or a validated transport choice for every device group.

## Supported Distributed Shapes Today

The distributed planner is deliberately narrow. It does not try to shard every
possible operator.

The planner currently recognizes these dispatch families:

- `parallel_linear_silu_multiply`
- `linear_residual`
- `sparse_moe_gate_up`
- `sparse_moe_down`

The implemented dense path covers BF16 output-row splitting for
`parallel_linear_silu_multiply` and block-scaled FP8 output-row splitting for
that operation and `linear_residual`. Each shard receives complete inputs,
owns a contiguous range of weight rows, and writes a disjoint output range.
This is real tensor parallelism, but independently splitting consecutive
projections causes every participant to access the globally assembled
intermediate instead of keeping its intermediate shard local.

The sparse path divides whole experts by expert range. That is expert
parallelism, not tensor parallelism within an expert. More importantly, the
current sparse distributed planner derives shards from permanent parameter
descriptors. Lazy selected experts use dynamic resource address and parameter
slot tables instead, so the existing sparse planning tests do not prove that
the runtime's actual lazy expert path executes distributed.

The existing two-device prompt-stream test proves that at least one eligible
dispatch can run distributed and preserve final token behavior for the tiny
fixture. It does not prove a complete FFN transaction, every intermediate,
every physical format, or lazy MoE residency.

That is closer to "local tensor parallelism inside selected components" than
to global model-wide tensor parallelism.

The current DeepSeek package now also passes a workload-free manual preflight
for both whole-expert and intra-expert TP on a selected transformer component.
For intra-expert TP, it resolves the gate/up plus down pair for single-lane
decode and the corresponding pair for both multi-lane decode and prefill. This
proves that a real package can select the exact executable contract family; it
does not prove a real-model token, numerical equivalence, performance, or live
teardown.

Selected-resource replication is now represented separately from component
TP. At a quiescent prompt boundary, the package can jointly score exact
selection and co-selection telemetry from multiple live streams, hold every
peer stream's arithmetic ownership fixed, and reassign only the current
stream. If two streams execute one immutable resource on different targets,
that resource has two physical copies but still exactly one arithmetic owner
per stream. The solver counts each physical copy once against the live
selector quota, uses compiler-bound measured execution and load costs, and
accepts a move only when the joint per-device makespan improves enough to
repay the destination load within the observed window. A package-level
generation guard serializes this preload/ownership/cache-policy transaction
without serializing token execution. This is hardware-neutral runtime support;
live Vulkan replication behavior and performance remain unproven while the
recorded inference quarantine is active.

## Runtime TP Completion Criteria

Before automatic placement consumes benchmark results, manual TP must satisfy
one concrete end-to-end contract:

- an explicitly selected real transformer component runs on two or more GPUs;
- decode and package-supported prefill both use the distributed path;
- every completed normal chat turn reports successful phase- and
  strategy-specific TP island submissions, rather than treating a mounted plan
  as execution evidence;
- immediate component outputs and state are canonical-equivalent to the
  single-device execution;
- each GPU stores only the permanent tensor ranges assigned to it;
- local intermediate shards remain local unless communication is required by
  the mathematical operation;
- lazy experts remain lazy and are not pinned by mounting a distributed plan;
- every shared-memory route used by the execution is known to produce correct
  values; and
- teardown releases the capacity acquired by TP without disturbing prior
  allocations.

Generated token equality is useful end-to-end evidence, but it is not a
substitute for comparing the output and state at the distributed component
boundary.

## Runtime TP Implementation

### Artifact-Specific Partition Contracts

Partitioning must be described by each concrete compiled kernel artifact, not
inferred from an operation name, descriptor count, or model-level dtype. The
artifact needs to state:

- which parameter dimension can be partitioned;
- whether each input is replicated or partitioned;
- whether output ranges concatenate directly or represent partial sums;
- tensor, packing, scale-block, and workgroup alignment requirements;
- the accumulation type for partial results;
- whether local intermediates can flow directly into the next distributed
  dispatch; and
- whether the implementation is valid for decode, prefill, or both.

This keeps BF16, F16, F32, FP8, Q8, INT4, MXFP4, and future physical formats in
their actual kernel implementations. A format is distributable only when its
selected artifact declares a valid partition contract. Unsupported artifacts
remain serialized and fail closed when explicitly requested for TP.

### Distribution Forms

The runtime needs three distinct execution forms:

- replicated input with partitioned output rows, where shards write disjoint
  output ranges and no numerical reduction is required;
- partitioned input columns with full-size partial outputs, where a reduction
  is required; and
- whole-expert ownership, where selected routes execute on the GPU that owns
  the corresponding lazy expert.

Tensor parallelism within one expert is a composition of the first two forms.
It must not be conflated with whole-expert parallelism.

### Complete FFN TP Sequence

The first complete TP unit should be the FFN sequence already represented by
`parallel_linear_silu_multiply` followed by the FFN `linear_residual`:

1. The owner publishes the normalized hidden input to every participant.
2. Each GPU computes a disjoint gate/up output-channel range.
3. The activated intermediate range remains in a local shard buffer on that
   GPU.
4. Each GPU consumes its local intermediate and the matching input-column
   range of the down-projection weight.
5. Each GPU writes one full-width F32 partial hidden vector into its own range
   of a shared collection buffer.
6. The owner reduces the partial vectors, converts once to the required output
   representation, adds the residual exactly once, and publishes the completed
   component output.
7. Ordinary serialized graph execution resumes on the owner.

The existing fused `linear_residual` kernel cannot be executed independently
on every input-column shard because that would add the residual once per GPU.
The distributed down path therefore needs a partial-projection kernel and one
owner-side reduction-and-residual kernel. F32 partial accumulation avoids
rounding every shard to BF16 before reduction.

Decode and prefill need equivalent sequences. Batch artifacts may have
different physical partition requirements and must declare their own
contracts.

### Distributed Activation Storage

The activation planner must distinguish:

- shared replicated inputs;
- local sharded intermediates that are never imported by other GPUs;
- shared disjoint outputs used for direct concatenation; and
- shared partial-output collections consumed by a reduction.

The current tendency to place distributed activations in one globally shared
allocation is correct for some output-row operations but prevents the efficient
FFN sequence above. Consecutive compatible dispatches should be recorded as one
distributed sequence so local intermediate shards never return to the owner
between gate/up and down.

### Dense And Lazy Parameter Residency

Permanent dense tensor shards may be resident with their placed component, but
the owner must not retain an excluded full tensor in addition to those shards.

Lazy experts must continue through the compiled-resource stores and dynamic
address tables. Whole-expert parallelism assigns each expert to an execution
GPU, binds only locally resident experts there, and lets unbound route slots do
no work. Gate/up and down for one selected expert should execute on the same
GPU before the owner performs the existing route reduction.

If one expert is tensor-sharded, all of its required fragments form one atomic
residency group across the participating GPUs. The dispatch is runnable only
when every required fragment is resident, and eviction must not leave a
partially resident but unusable expert. Distributed expert fragments must not
be allocated and loaded eagerly at package mount.

### Transport Correctness

Shared-buffer allocation must accept an explicit transport choice. Importing
device-local external memory successfully is not proof that cross-device shader
reads and writes produce correct values. Device-local and shared-host routes
must therefore be selectable independently and must fail closed when the
requested route is unavailable.

Route validation belongs in the canonical execution tests and the standalone
benchmark. The runtime should consume the resulting route choice later rather
than implementing a second performance benchmark internally. Until placement
integration supplies that choice, manual execution and tests must select the
route explicitly instead of silently trusting device-local sharing.

### Correctness Coverage

Every artifact that declares a partition contract needs canonical comparisons
covering:

- immediate component output and persistent state;
- decode and prefill;
- every supported participant count up to the configured runtime limit;
- every supported physical format and fallback implementation;
- direct output concatenation and partial-output reduction;
- device-local and shared-host transport where available;
- permanent dense shards, whole-expert placement, and TP within one expert;
  and
- parameter and transient residency before, during, and after execution.

Attention requires a separate head and KV-cache partition contract. Vocabulary
or logits sharding requires a distributed selection or collection contract.
They should reuse the same artifact contracts, activation storage, residency,
transport, and synchronization substrate, but arbitrary-model TP is not
complete until every tensor that may require splitting has a correct execution
form.

## Current Auto-Placement Behavior

Automatic placement is still capacity-first and contiguous.

The auto-placer:

- discovers compatible Vulkan devices;
- excludes integrated GPUs from automatic placement;
- opens devices to measure safe/reservable VRAM;
- ranks candidate devices mainly by capability class and capacity;
- tries the smallest prefix of ranked devices that can satisfy residency; and
- assigns graph-order contiguous component segments.

Relevant code:

- `runtime-rs/src/bin/nerve_runtime/runtime_devices.rs`
- `runtime-rs/src/vulkan_stream_circuit/runtime_auto_placement.rs`
- `runtime-rs/src/stream_circuit/capacity_packed_placement.rs`

It does not currently choose `component_shard_devices` automatically. It also
does not consume measured inter-GPU link costs or per-operation execution costs
when deciding whether sharding is worthwhile.

## Expert Residency Matters More Than Layer Residency

For MoE models, a layer is the wrong primary planning unit. The runtime can
load independently selectable resource groups, and sparse MoE only executes
selected expert routes.

Existing infrastructure already treats resources as independently addressable
groups with:

- byte counts;
- dependencies;
- compatibility descriptors;
- loading state;
- hit/miss counts;
- eviction and reload counts; and
- high-water residency metrics.

Relevant code:

- `runtime-rs/src/vulkan_stream_circuit/device_resource_residency.rs`
- `runtime-rs/src/vulkan_stream_circuit/compiled_resource_residency_plan.rs`
- `runtime-rs/src/vulkan_stream_circuit/compiled_resource_memory_plan.rs`

Sparse MoE execution already distinguishes declared expert slots from selected
routes per activation.

Relevant code:

- `runtime-rs/src/vulkan_stream_circuit/sparse_moe_execution.rs`

Runtime selection telemetry can report which resource IDs were selected.

Relevant code:

- `runtime-rs/src/vulkan_stream_circuit/selection_telemetry.rs`

That suggests the future planner should reason about resident expert sets,
route probability, load cost, and activation movement instead of treating all
experts in a layer as one indivisible block.

## Benchmark Role

`nerve-gpu-bench` is a standalone Rust and Vulkan package whose only job is to
measure the relative cost of placement choices on the current host. It is not a
general hardware profiler and it must not load a model, manifest, package, or
downloaded tensor.

The benchmark must answer concrete placement questions:

- Is one target faster than another for the same runtime operation and format?
- Is a TP group faster than every one of its members running alone?
- When work must span a target set, is TP faster than serialized placement?
- Which serial direction, TP owner, shard split, and valid transport is fastest?
- What does it cost to cross from the output target of one placement to the
  input target of the next?
- For lazy resources, is execution on a target still worthwhile after expected
  materialization cost is included?

One fixed, small logical payload is sufficient for each case. The durations are
relative placement weights, not claims about absolute model latency. There is
no size sweep, normalized score, global GPU ranking, or arbitrary equivalence
threshold.

## Shared Execution Contracts

The benchmark must measure the same execution forms that inference can select.
Each model-independent execution contract must identify:

- operation family;
- decode or prefill phase;
- physical storage format;
- selected compute implementation and accumulation format;
- parameter partition method;
- input distribution method;
- output concatenation or reduction method; and
- alignment requirements for valid shard boundaries.

The runtime and benchmark must consume partition descriptors and shader
artifacts generated from one source. The standalone benchmark may allocate
deterministic synthetic parameters and activations around those artifacts, but
it must not contain an approximate reimplementation of TP. It remains runnable
without the inference runtime, model files, or model-specific metadata.

Initial contracts should cover only execution forms the runtime actually
supports:

- complete FFN TP islands;
- whole-expert execution;
- TP within one expert;
- router and expert-output reduction;
- attention head and KV-cache partitioning once implemented; and
- lazy compiled-resource materialization.

Storage format and compute path are distinct. An MXFP4 tensor dequantized for a
BF16 kernel is a different contract from a native low-precision path. This lets
a target without native support compete through its real fallback
implementation instead of being rejected from capability information alone.

## Current Benchmark Gap

The existing package already provides useful mechanics:

- standalone Rust and Vulkan execution with deterministic synthetic data;
- a small fixed logical parameter budget;
- single-target, directed serial, and multi-target TP runs;
- selected target groups up to a configured participant limit;
- F32, F16, BF16, FP8, FP4, MXFP4, INT8, INT4, and Q8 paths; and
- output validation before a result is retained.

Its compact placement conversion is not yet safe for inference placement. It
groups observations by format and broad regime, then deduplicates them by
strategy and target identity. Workload family, phase, kernel contract, and
shape are discarded. A format ranking can therefore retain whichever unrelated
case happened to be fastest, leaving the runtime unable to determine what was
actually measured.

The JSON must preserve an exact execution-case identity before automatic
placement consumes it.

## Benchmark Target Policy

The package must discover every target exposed by the Vulkan backend, including
CPU Vulkan implementations, integrated GPUs, discrete GPUs, and devices from
every vendor. It must not encode workstation-specific or device-class bans.

The user can explicitly include or exclude targets for a run:

```text
nerve-gpu-bench --include-target <stable-target-id>
nerve-gpu-bench --exclude-target <stable-target-id>
nerve-gpu-bench --exclude-pci 0000:8a:00.0
nerve-gpu-bench --exclude-kind integrated_gpu
```

Before each workload, the runner must inspect the selected targets' current
allocation, usable capacity, and activity. Existing allocations are
reservations. A benchmark may share a target only when its small buffers fit in
the measured remaining capacity, and teardown must release only the capacity
the benchmark acquired.

Target groups are generated from the selected viable targets, not from a
hard-coded machine inventory. Groups range from one target through the
configured runtime participant limit. The intended initial limit is four.

## Benchmark Cases

### Component Placement

A component case compares one complete executable contract:

- `single` runs the entire contract on one target; and
- `tp` runs the same contract on a target group, including input distribution,
  local shard computation, synchronization, output concatenation or reduction,
  and publication of the completed output.

The input begins on an explicit input target and the completed output ends on
an explicit output target. For the initial TP contracts both are normally the
owner, but the JSON must not assume that they are always identical.

### Forced-Split Placement

A forced-split case represents work that cannot fit on one target. It uses a
small fixed sequence of identical contract units and compares:

- `single`, with the complete sequence on each individual target where valid;
- `serial`, with contiguous units assigned across an ordered target path; and
- `tp`, with every unit sharded across the same target group.

All strategies use the same total parameter budget, operation count, input,
and output. Every directed pair is measured in both orders. Larger target sets
cover every serial order and TP owner supported by the configured participant
limit. This directly answers TP versus serialization without deriving one from
unrelated primitive timings.

### Directed Boundaries

The benchmark separately measures activation movement from every selected
source to every selected destination using the same sharing and synchronization
routes available to inference. Direction matters.

These boundary measurements compose hybrid model plans. If one component ends
on target A and the next begins on target B, the planner adds the measured
`A -> B` boundary cost. Distribution and collection internal to a TP candidate
are already included in that candidate and must not be counted again.

### Lazy Resource Loads

Small synthetic load-wave cases measure:

- host backing storage to one target;
- host backing storage to every member of a shard group; and
- concurrent materialization of an atomic sharded resource group.

The runtime combines these measurements with the real resource byte count,
current residency, and observed expert selection frequency. The benchmark does
not need a model or a simulated expert catalog.

### Shard Splits And Routes

Balanced TP is measured whenever it is supported. Unequal splits are included
only after the runtime can execute them and the partition contract can express
their aligned shard boundaries. Each distinct measured split remains a
separate candidate.

The runner may try every transport route, but the placement JSON retains only
completed, output-valid candidates. The chosen route remains attached to the
candidate because transport is part of its execution contract.

## Measurement Contract

One candidate measurement begins after reusable pipelines, descriptors, and
fixed buffers have been prepared. It ends only when the completed output is
usable on the declared output target. It includes command submission, all GPU
work, cross-device synchronization, transfers, collectives, and owner-side
continuation required by that placement strategy.

Multi-device GPU clocks cannot be assumed to share a time domain, so the
authoritative comparison is one monotonic host duration around the complete
transaction. Warmup and repeated samples remain deliberately small, and each
candidate has a hard ten-second execution timeout. Setup and shader compilation
are excluded unless the measured case is explicitly a lazy materialization
case.

Every candidate uses deterministic data and is compared with the contract's
canonical reference result at the component boundary. Single-target candidates
are validated too; one GPU is not treated as truth for the others. The
comparison must cover both output and persistent state using the numerical
requirements of the selected format. Successful allocation, finite output, or
repeated output is not enough. A route that returns an incorrect value is
invalid even if it is fast.

The artifact contains only completed, output-valid observations. Missing
observations are unavailable to placement. Discovery errors and failed attempts
stay in command output, while every successful independent case remains usable
in the artifact. The artifact never claims that an omitted combination was
supported or measured.

## Placement JSON

The JSON is compact and human-readable. It contains raw durations grouped by an
exact case identity:

```json
{
  "schema": "nerve.placement_bench",
  "payload_bytes": 5242880,
  "cases": {
    "execution:ffn_island:bf16_f32:decode": [
      {
        "mode": "single",
        "targets": ["vulkan:pci:0000:03:00.0"],
        "input_target": "vulkan:pci:0000:03:00.0",
        "output_target": "vulkan:pci:0000:03:00.0",
        "duration_ns": 310000
      },
      {
        "mode": "tp",
        "targets": [
          "vulkan:pci:0000:03:00.0",
          "vulkan:pci:0000:07:00.0"
        ],
        "input_target": "vulkan:pci:0000:03:00.0",
        "output_target": "vulkan:pci:0000:03:00.0",
        "owner": "vulkan:pci:0000:03:00.0",
        "split": [1, 1],
        "transport": "shared_host",
        "duration_ns": 240000
      }
    ]
  }
}
```

Case identities are serialized from typed Rust contract values; planner code
must not recover meaning by ad hoc string parsing. Serial target arrays preserve
execution order. Non-serial target groups use stable physical target order.

The artifact does not contain a separate target inventory, capability table,
missing-result list, raw samples, normalized scores, uncertainty report, or
performance-equivalence threshold. The cases and observations present are the
measured usable set.

## Placement Cost

The planner must use the measured complete transaction cost instead of
reconstructing TP from independent bandwidth and compute metrics:

```text
plan_cost =
  sum(measured candidate execution costs)
  + sum(measured directed boundary costs)
  + sum(expected lazy-resource load costs)
```

The candidate execution duration already contains shard compute, fan-out,
collection or reduction, synchronization, transport, and layout conversion.
The model contributes exact parameter and transient byte counts, component
repetition, resource dependencies, context requirements, and expected phase
mix. Benchmark durations remain raw; they are not converted into a synthetic
global score.

For MoE resources:

```text
expected load contribution =
  selection probability
  * miss probability
  * measured load-wave cost
```

Selection and miss probabilities come from runtime behavior, not from the
standalone benchmark. Permanent and lazy tensor byte counts likewise come from
the loaded model.

## Placement Pipeline

After runtime TP is complete, automatic placement follows this pipeline:

1. Read the benchmark artifact and map its stable physical target IDs to the
   targets currently exposed by Vulkan.
2. Apply explicit caller exclusions, then inspect every remaining target's
   current allocation, safe remaining capacity, and activity.
3. Map each compiled component and lazy resource group to the exact benchmark
   contract implemented by its selected artifacts.
4. Enumerate only executable candidates: single-target ownership, serialized
   component boundaries, whole-expert ownership, TP shard groups, replicated
   hot resources, and demand-loaded resources.
5. Attach the raw measured execution, boundary, and load-wave costs. A missing
   measurement makes that candidate unavailable; capability claims do not fill
   the gap.
6. Attach each candidate's exact per-target permanent, transient, KV-state, and
   atomic lazy-resource byte requirements from the model's residency plans.
7. Build a candidate graph in model execution order. An edge joins compatible
   output and input targets and adds the measured directed boundary cost when
   they differ.
8. Find the lowest-cost complete path that satisfies every target's current
   capacity. This is a constrained candidate search with exact byte vectors,
   not a greedy global device ranking. Dominated states may be discarded, but
   a faster partial path cannot replace another path when their remaining
   capacities differ.
9. Admit the selected fixed and lazy residency plans using the runtime's exact
   allocation checks. If current capacity changed during planning, refresh the
   reservation snapshot and replan rather than silently changing the selected
   strategy.
10. Materialize the plan as component owners, `component_shard_devices`,
    transport choices, expert placement preferences, and atomic lazy-resource
    groups.
11. After execution or unload, verify that NERVE released the capacity it
    acquired and that pre-existing allocations remain present.

There is no global fastest-GPU list. The winner is selected per execution case,
phase, target set, and current capacity. A model can therefore use one TP island,
then serialized components on individual targets, then another TP island.

When prefill and decode prefer different placements, the runtime may use
separate plans only if switching does not require remounting tensors or breaking
residency guarantees. Otherwise the caller's expected phase mix determines the
combined path cost.

## Implementation Order

1. Complete and canonically validate manual FFN TP for decode and prefill.
2. Complete whole-expert placement, TP within lazy experts, and atomic sharded
   residency without eager expert loading.
3. Define typed model-independent execution contracts and make runtime and
   benchmark artifacts come from one source.
4. Replace the benchmark's synthetic approximation with complete contract
   transactions while retaining its small deterministic buffers.
5. Replace format-only JSON collapsing with exact case-preserving candidates,
   directed boundaries, and lazy load-wave cases.
6. Add a runtime JSON reader that rejects unknown contracts and unavailable
   target mappings.
7. Extend `VulkanRuntimePlacementCostModel` from single-device component costs
   to complete single, serial, expert, and TP placement candidates.
8. Extend automatic placement from contiguous serial segments to the
   capacity-constrained candidate graph described above.
9. Compare the chosen plan against direct equal-work single, serialized, and TP
   benchmark cases before enabling benchmark-backed automatic placement.

## Placement Rules

- All detected compatible Vulkan targets are eligible unless explicitly
  excluded for the current run.
- No device or link is assumed fast or slow from vendor, target class, PCIe
  generation, advertised format support, or prior workstation behavior.
- No TP candidate is used unless the exact contract, group, split, owner, and
  transport completed correctly in the benchmark.
- Single-target, serialized, and TP candidates must perform equivalent work.
- The lower measured complete cost wins; there is no arbitrary speedup margin
  or performance-equivalence band.
- Current safe capacity and exact model residency decide feasibility; benchmark
  rankings never override capacity.
- Demand-paged load-wave admission does not prove warm working-set viability.
  Candidate subset selection separately compares each exact per-device maximum
  addressable residency—including store overhead, state, and activations—with
  that target's safe capacity. It keeps considering additional compatible
  targets while a complete warm fit is avoidable, and accepts a smaller paged
  cache only when no legal larger subset removes the shortfall. Runtime
  telemetry may subsequently rebalance a materially smaller observed hot set.
- Lazy experts remain independently selectable resources rather than being
  collapsed back into whole-layer placement.
- The generated JSON files remain local calibration artifacts and are not
  committed to the repository.
