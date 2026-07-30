# TODO — Demand-Resident Compiled Resources

## Goal

Build a generic, access-triggered residency system for compiled NERVE models.
Mounting a model must not require copying every immutable parameter into device
memory. The runtime should keep the model's execution spine resident, load a
selectable resource only when execution actually requests it, and retain that
resource until the model or its device slice is explicitly unloaded.

Qwen3.6-35B-A3B is the first proof workload because its routed experts provide a
clear, high-value residency boundary. It is not the architecture of the feature.
The same compiler and runtime machinery must support any independently
selectable immutable resource, including expert bundles, tensor partitions,
optional prediction heads, vocabulary partitions, adapters, multimodal
components, and future compiled representations.

This first policy is **demand-retained residency**:

- an unaccessed dynamic resource consumes no device-local parameter memory;
- the first real access loads and atomically publishes the resource;
- subsequent accesses reuse the resident resource without host intervention;
- resident resources are never evicted implicitly; and
- a request fails deterministically when its accumulated working set cannot fit.

This makes models larger than available VRAM usable when the workload's accessed
working set fits. It does not pretend that an unbounded working set can fit
without eviction. A future bounded-residency policy may add measured eviction,
but it must remain a separate policy rather than silently changing
demand-retained semantics.

The compiled package must remain self-contained and relocatable. The compiler
describes addressable resources and their access relationships; runtime policy,
device placement, loading, sharing, and lifetime remain runtime concerns.

## Architectural invariants

- The mechanism is generic. No runtime or package-contract branch may identify
  Qwen, MoE model families, or model names.
- A semantic graph component and a physical residency unit are different
  concepts. One component may use several residency units, and one atomic
  residency group may contain several tensors.
- Residency units are derived from compiled access semantics. They are not
  created merely because a tensor, layer, or source file exists.
- Resources required for every execution remain in an explicitly declared
  always-resident spine. Dynamically selected resources are independently
  addressable.
- Strict demand-retained mode loads only resources selected by actual execution.
  Speculative prefetch may be explored later as a separately named policy.
- The resident-hit path remains entirely on the GPU. A hit must not require a
  CPU round trip, queue drain, or device-wide synchronization.
- A miss is schedulable backpressure. The affected activation pauses at a
  physical execution checkpoint, missing resources are loaded, and execution
  resumes there without replaying the token or the whole layer.
- Atomic resource groups become visible only after every member is present,
  verified, uploaded, and synchronized.
- Immutable resources are content-addressable, shareable by compatible graph
  instances and streams, and loaded through a single-flight operation per
  physical device.
- Model parameters, mutable stream state, and transient activation storage keep
  separate ownership and lifetime rules.
- Placement remains a runtime decision. The same package can use one device or
  several devices without recompilation or compiler-authored placement files.
- There is no hidden driver paging, silent CPU fallback, automatic eviction, or
  best-effort partial execution.
- Capacity failures, corrupt resources, cancelled loads, and device failures
  produce deterministic errors and leave residency state internally coherent.
- Model and device teardown is explicit, serialized per physical device, and
  proven to return each used AMD GPU to its pre-workload idle baseline.
- Residency and loading statistics are reported by normal execution. They do
  not require a profiling-only runtime mode.
- All GPU validation is sequential, uses only verified-idle AMD devices, and
  follows `AGENTS.md`.

## Work plan

### 15. Harden failure, teardown, and recovery behavior

- Make partial reads, verification failures, upload failures, capacity failures,
  device loss, and cancellation roll back without publishing partial groups.
- Quiesce dependent work before releasing residency.
- Release model residency one physical device at a time, destroy contexts only
  after their allocations are gone, and explicitly acknowledge completion.
- Verify used AMD devices return to the exact pre-workload idle baseline.
- Demonstrate that a failed load does not poison unrelated resident resources or
  streams.

Completion requires adversarial fault-injection tests and repeated
mount/use/unmount cycles without GPU, host-memory, file-descriptor, or scheduler
leaks.

### 16. Qualify real usability and measure the result

- Run Qwen3.6-35B-A3B in eager and demand-retained modes under matched placement,
  context, thinking, sampling, and MTP settings.
- Use one warmup conversation followed by meaningful multi-turn prompts; allow
  up to 65,536 generated tokens rather than imposing artificial tiny limits.
- Compare startup latency, first-use stalls, prefill, decode, warm-turn
  throughput, initial VRAM, high-water VRAM, working-set growth, output quality,
  and final teardown.
- Confirm that unselected experts never become resident, a repeated expert
  incurs no second transfer, and residency growth matches recorded selections.
- Coalesce all misses reported by one checkpoint into bounded backing-store
  reads, device uploads, stable-address publications, queue submissions, and
  waits. A cold turn must not perform one transfer lifecycle per tensor member
  or selected expert.
- Keep the fully warm decode path on the GPU under demand-retained policy.
  Resident feedback windows must continue across hits and interrupt at the
  exact physical checkpoint only when a real miss requires host loading; a warm
  scalar token must not drain a queue or wait on the host.
- Test a workload whose package maximum exceeds available VRAM but whose
  observed working set fits, then test deterministic failure when a working set
  truly exceeds capacity.

Completion requires materially lower initial and observed VRAM than eager mode,
identical behavior under deterministic settings, usable real conversations, and
no material fully warm throughput regression.

### 17. Perform the final genericity and architecture review

- Audit the compiler, package, runtime, Vulkan backend, scheduler, CLI, and TUI
  for model-family assumptions, duplicate abstractions, dead code, fallbacks,
  hidden paging, and misplaced placement policy.
- Exercise a second non-Qwen residency pattern end to end, using a real compiled
  model when available or a structurally faithful synthetic package otherwise.
- Stress metadata and runtime bookkeeping at substantially larger expert/unit
  counts to prove the design scales without file or metadata explosion.
- Update `CONCEPT.md`, `README.txt`, and `EXPERIMENTS.md` to describe the proven
  architecture, limitations, measurements, and future bounded-residency policy.
- Remove every item from this file only after it has been implemented, reviewed,
  tested, benchmarked where relevant, committed atomically, and pushed.

Completion requires a clean final review with no unresolved work. At that point
this work plan must be empty.

## Milestone discipline

Work through one numbered item at a time. For every item:

1. implement the complete architectural slice;
2. review it adversarially against the goal and invariants;
3. run only the sequential tests needed to prove it;
4. benchmark its impact against the latest valid eager and demand-retained
   baselines when it touches execution;
5. update this file with any newly discovered work;
6. remove the completed item only when it is complete in production paths, not
   merely represented by a stub or passing test; and
7. create an atomic commit and push the verified milestone.

Microbenchmarks must remain bounded to under one minute and answer a concrete
binary performance question with a discarded warmup and a small number of
matched measurements. End-to-end qualification is separate and must use real
conversation settings. Tests and GPU work are never parallelized.

## Overall completion criteria

The goal is complete only when:

- a self-contained compiled model can mount without allocating its full dynamic
  parameter set;
- only execution-selected dynamic resources are read, uploaded, and published;
- resident resources remain reusable until explicit unload;
- the warm resident path stays on the GPU and has no material regression from
  eager execution;
- misses pause and resume at physical checkpoints without replaying completed
  inference work;
- immutable resources are shared safely across streams and compatible graph
  instances;
- runtime placement works on one device and across multiple devices without
  recompilation;
- memory accounting and capacity failures are exact and deterministic;
- package integrity, runtime failures, cancellation, and teardown are robust;
- Qwen3.6-35B-A3B produces correct real conversations with materially reduced
  initial and working-set VRAM;
- at least one structurally different resource-selection pattern proves the
  architecture is generic;
- documentation describes what was actually proven rather than intended; and
- this TODO file contains no remaining work items.
