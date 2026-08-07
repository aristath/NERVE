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
component shard pools. The CLI exposes this through `--shard-component`, which
populates `component_shard_devices`.

Relevant code:

- `runtime-rs/src/bin/nerve_runtime/args_parsing.rs`
- `runtime-rs/src/stream_circuit/runtime_graph/placement_spec.rs`

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
The missing part is automatic, benchmark-backed selection.

## Supported Distributed Shapes Today

The distributed planner is deliberately narrow. It does not try to shard every
possible operator.

Currently supported dispatch families include:

- `parallel_linear_silu_multiply`
- `linear_residual`
- `sparse_moe_gate_up`
- `sparse_moe_down`

Dense and residual projections are currently sharded by output rows. Sparse MoE
dispatches are sharded by expert range. This is a useful starting point because
these operations have clear parameter partitions and understandable activation
movement.

That is closer to "local tensor parallelism inside selected components" than
to global model-wide tensor parallelism.

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

## Calibration Status

The project already has a hardware calibration pipeline.

It can generate per-device workloads for:

- GPU compute;
- GPU memory behavior;
- GPU transfer;
- GPU synchronization;
- scheduling and command queue behavior; and
- cooperative matrix shapes where supported.

Relevant code:

- `nerve/hardware_calibration/planning.py`
- `nerve/hardware_calibration/orchestrator.py`
- `nerve/hardware_calibration/statistics.py`
- `runtime-rs/src/hardware_calibration/schema.rs`
- `runtime-rs/src/hardware_profile/schema.rs`

The current profile schema can store process capabilities, memory domains,
interconnect declarations, and measurements. What appears to be missing is an
explicit pairwise GPU-link calibration matrix.

Capability discovery is only a filter. It can say that a device exposes a
format, feature, queue, memory route, or synchronization primitive, but it
cannot say whether that path is worth using. The planner must not infer that a
GPU is fast from advertised support, PCIe generation, device age, or marketing
class.

Every relevant device/format/operator/shape/phase combination needs an
empirical entry:

```text
(device, format, operator_family, shape, phase) -> measured cost
```

Unsupported formats and operations still matter. A GPU without native FP4 may
be the fastest F32 or BF16 device, the best router/logits/sampler device, the
best reduction owner, or the best fallback target for dequantized work. A GPU
with native low-precision support may still lose on decode-sized workloads if
dispatch, memory, synchronization, or transfer costs dominate.

The benchmark matrix therefore needs to include native and fallback paths, for
example:

- FP4 native, where available;
- FP4 unpack/dequantize into BF16 or F32;
- INT4 unpack with BF16 or F32 accumulation;
- FP8 block-scaled projection paths;
- BF16 dense projections;
- F16 dense projections;
- F32 scalar, vector, reduction, router, logits, and sampler paths;
- sparse MoE gate/up/down paths;
- KV append/read paths;
- activation copies; and
- resource load-wave transfers.

Each measurement must be tied to a workload regime, not just an operator name.
Important regimes include:

- batch-1 decode;
- small prefill;
- large prefill;
- long-context attention/state access;
- hidden width;
- intermediate width;
- expert count;
- experts per token;
- route tensor size;
- expert intermediate size;
- partial output size; and
- load-wave size.

Benchmark results should be classified explicitly as:

- supported and reliable;
- supported but slow;
- supported but unreliable;
- unsupported;
- unmeasured; or
- invalid for the requested regime.

That distinction matters because absence of evidence must not silently become
permission to place work on a device.

## Benchmark Package Target Policy

The standalone benchmark package must not encode workstation-specific bans as
global rules. Hardware that was risky or unsuitable on one machine can be valid
on another machine, and the benchmarker is supposed to discover that from the
current host.

The package should discover every target class it can see, including:

- CPU targets;
- integrated GPUs;
- discrete GPUs;
- AMD devices;
- NVIDIA devices;
- Intel devices; and
- any other supported accelerator exposed by the backend.

The user should then choose the target set explicitly. The CLI should support
both inclusion and exclusion controls, for example:

```text
nerve-gpu-bench --include-target <stable-target-id> --include-target <stable-target-id>
nerve-gpu-bench --exclude-target <stable-target-id>
nerve-gpu-bench --exclude-pci 0000:8a:00.0
nerve-gpu-bench --exclude-kind integrated_gpu
```

Discovery should report all targets even when a local policy excludes them
from a particular run. Exclusion is a run policy decision, not a device-class
truth.

The JSON output should preserve:

- every discovered target;
- the selected benchmark target set;
- explicit user exclusions;
- automatic safety exclusions, if any;
- the reason each target was skipped; and
- enough stable identity to let the inference engine map measurements back to
  current runtime devices.

Local deployment policy can still choose to avoid a display iGPU, a problematic
driver stack, a busy production GPU, or any other target. That policy belongs
in the run configuration and result provenance, not as a hard-coded package
assumption.

The existing transfer calibration is per selected device and covers
host-to-device, device-to-host, and device-local buffer copy. For tensor
parallelism decisions we also need pairwise measurements such as:

- GPU A to GPU B activation-sized transfer;
- GPU B to GPU A activation-sized transfer;
- imported/shared memory route availability;
- external semaphore round-trip cost;
- queue dependency cost;
- host staging fallback cost; and
- contention behavior when multiple shard helpers are active.

## Cost Model

A future planner should compare serialized and distributed execution using a
cost model with at least these terms:

```text
serialized_cost(component, owner_gpu)

distributed_cost(component, gpu_set) =
  max(shard_compute_costs)
  + input_distribution_cost
  + output_collection_cost
  + synchronization_cost
  + layout_conversion_cost
  + extra_residency_cost
```

For MoE, the cost model needs expected route behavior:

```text
expected_expert_cost(expert, gpu) =
  route_probability(expert)
  * execution_cost(expert, gpu)
  + miss_probability(expert, gpu)
  * load_cost(expert, gpu)
  + eviction_pressure_cost(expert, gpu)
```

The important inequality is:

```text
saved_compute_time > transfer_cost + synchronization_cost + residency_cost
```

This should be evaluated per operation family, per execution phase, and per GPU
set. Decode and prefill may choose different answers.

## Planner Shape

The planner should produce candidate placements at three levels.

### 1. Device And Link Facts

Inputs:

- physical device IDs;
- current safe capacity;
- per-device calibrated compute/memory profiles;
- pairwise link bandwidth and latency;
- synchronization capability and cost;
- shared/imported memory route support; and
- runtime compatibility.

Output:

- compatible device sets;
- fast-pair or fast-island candidates;
- slow-link exclusions;
- per-route transfer costs for activation-sized payloads.

### 2. Component And Dispatch Candidates

Inputs:

- runtime graph;
- prepared dispatches;
- tensor index;
- shardable op families;
- activation shapes;
- parameter byte ranges;
- resource residency contract; and
- selection telemetry when available.

Output:

- serialized candidate;
- output-row sharded candidate;
- expert-range sharded candidate;
- replicated-resource candidate;
- demand-loaded-resource candidate.

### 3. Whole Runtime Plan

Inputs:

- residency policy;
- context size;
- speculative draft tokens;
- expected phase mix;
- route telemetry;
- device/link facts;
- candidate costs.

Output:

- component ownership placement;
- optional `component_shard_devices`;
- resource placement preferences;
- hot expert pinning/replication recommendations;
- cold expert lazy-load policy;
- expected cost report;
- fallback reason if sharding is rejected.

## Staged Implementation Plan

### Stage 1: Reporting Only

Keep automatic placement unchanged. Add an inspection/report mode that says:

- which components have distributable dispatches;
- which dispatches are output-row shardable;
- which dispatches are expert-range shardable;
- how many bytes would move per activation;
- how much parameter data would be sharded;
- which GPU sets are structurally valid; and
- which information is missing for a cost decision.

This stage should not change runtime behavior.

### Stage 2: Pairwise Calibration

Extend calibration to produce pairwise GPU-link measurements. The output should
be published as stable evidence, not ad hoc logs.

The measurements should cover sizes that correspond to actual runtime payloads:

- hidden activation frames;
- route tensors;
- expert intermediate tensors;
- partial output fragments;
- KV-like state payloads where applicable; and
- full load-wave sized parameter transfers.

### Stage 3: Manual Candidate Scoring

Given an explicit `--shard-component` placement, produce an estimated score
before running and an observed score after running.

Compare:

- predicted transfer bytes;
- actual transport stats;
- predicted shard count;
- actual distributed dispatch count;
- predicted residency pressure;
- actual hit/miss/reload stats;
- predicted token latency;
- observed token latency.

This lets the cost model be calibrated before it is trusted for automatic
placement.

### Stage 4: Conservative Automatic Sharding

Allow automatic placement to add `component_shard_devices` only when all of the
following are true:

- the component has a supported distributed dispatch;
- the GPU set has measured link and sync costs;
- every selected GPU has sufficient safe capacity;
- predicted distributed cost beats serialized cost by a configured margin;
- the residency policy can still be admitted exactly;
- the plan does not involve a target excluded by the active run policy; and
- the generated report explains why sharding was selected.

### Stage 5: Expert-Aware Residency Planning

Use selection telemetry to estimate expert hotness.

Then consider:

- pinning hot experts on fast GPUs;
- replicating very hot experts when replication beats transfer or lazy-load cost;
- placing medium experts by capacity and link cost;
- leaving rare experts demand-loaded;
- preserving route dependencies and atomic group boundaries; and
- validating that demand-paged or demand-retained semantics remain explicit.

## Policy Implications

The current repository rule says not to use tensor parallelism on this
workstation. Based on this review, that rule should be treated as a safety
policy until deliberately revised, not as a proven performance fact.

The codebase already supports explicit local sharding in some cases. A future
policy could be more precise:

- no unmeasured TP;
- no global TP by default;
- no slow-link participants without measured justification;
- no hard-coded global bans for CPU, integrated GPU, discrete GPU, AMD, NVIDIA,
  Intel, or any other target class;
- target exclusions are explicit run policy, not permanent device-class facts;
- local component sharding is allowed only from calibrated evidence; and
- benchmarks must compare equivalent execution settings and placement
  strategies.

## Open Questions

- Should pairwise link measurements live inside each `HardwareProcessProfile`
  as endpoint-named regimes, or should there be an inventory-level topology
  profile?
- Should the auto-placer remain a capacity planner plus a separate sharding
  recommender, or should it become one combined cost optimizer?
- Should expert hotness be learned per model globally, per stream, or per
  workload class?
- How should the planner account for prefill versus decode when one placement
  wins prefill and another wins decode?
- When is expert replication worth the extra residency compared with remote
  execution, activation transport, peer transfer, or a second resident copy?
- What minimum predicted speedup margin is required before automatic sharding
  is allowed?

## Near-Term Recommendation

Do not start by changing execution behavior.

Start by adding a planner/reporting surface that makes the existing distributed
capabilities visible. Then add pairwise calibration. Once predictions and
observations line up, allow automatic local sharding for the narrow supported
dispatch families first.

That path preserves the current serialized placement as the reliable baseline
while giving tensor/expert parallelism a fair, evidence-driven way to prove
itself.
