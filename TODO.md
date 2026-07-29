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

### 5. Separate initial, current, and maximum residency planning

- Replace the assumption that every permanent parameter buffer is allocated and
  loaded at mount time.
- Plan and report always-resident bytes, initial dynamic bytes, current resident
  bytes, maximum addressable bytes, staging headroom, transient state, and
  activation headroom separately.
- Admit a model in demand-retained mode from its initial requirement, while
  retaining deterministic checks for each later growth operation.
- Ensure a large maximum address space does not itself require a same-sized
  device allocation.

Completion requires admission tests showing that a package can mount when its
maximum parameter size exceeds free VRAM but its initial resident set fits.

### 6. Implement the asynchronous backing-store path

- Add bounded range reads using the best measured host mechanism for the target
  system, with reusable pinned staging storage.
- Coalesce nearby or concurrent missing ranges while preserving atomic-group
  boundaries and integrity checks.
- Upload with transfer queues and timeline synchronization where supported;
  never use `deviceWaitIdle` as a loading primitive.
- Bound host memory, staging memory, outstanding I/O, and outstanding transfers.
- Make cancellation and I/O failure release staging resources predictably.

Completion requires sequential read/upload tests for isolated, adjacent,
duplicate, cancelled, corrupt, and failed requests, with measured cold-load
latency and bandwidth.

### 7. Implement the per-device residency manager

- Own the generic residency state machine, capacity accounting, resource
  identity, load coalescing, atomic publication, reference ownership, and
  explicit unload.
- Guarantee a single physical load for concurrent requests for the same
  immutable resource on the same device.
- Share compatible resident resources across streams and duplicated or rewired
  graph instances while keeping their mutable state independent.
- Return deterministic capacity errors before beginning a load that cannot be
  completed atomically.
- Maintain a per-device directory that can later distinguish local residency
  from a resource resident on another execution device.

Completion requires concurrency, sharing, capacity, rollback, cancellation, and
explicit-unload tests with no leaked allocations.

### 8. Add stable GPU addressing for independently resident resources

- Build device-local immutable arenas or another measured stable-address scheme
  using current Vulkan capabilities, including buffer device addresses where
  appropriate.
- Publish resource addresses and resident bits through GPU-visible tables with
  narrowly scoped synchronization.
- Keep already published addresses stable for their full resident lifetime.
- Avoid allocating virtual or physical storage proportional to maximum model
  size unless measurements prove it is superior and the device supports it
  safely.
- Benchmark monolithic direct addressing against the new indirection so the
  package can carry materially different verified implementations when useful.

Completion requires address-stability, alignment, visibility, and capacity
tests, plus a sub-minute hot-path microbenchmark showing whether the chosen
lookup mechanism adds material overhead.

### 9. Split selectable execution at a physical residency checkpoint

- Lower routed execution into distinct physical stages: selection, availability
  check/request emission, selected computation, and reduction.
- Keep this a physical schedule concern; do not fragment or weaken the editable
  semantic component graph.
- Ensure a missing resource suppresses only work that depends on it and cannot
  expose partially initialized parameters.
- Resume the paused activation at the selected-computation stage after the
  required atomic groups become resident.
- Generalize the checkpoint ABI so future selectors can request non-expert
  resources.

Completion requires eager and demand-retained traces to select the same
resources, execute the same physical responsibilities, and resume without
whole-token or whole-layer replay.

### 10. Keep the resident-hit path on the GPU

- Check resident bits and resolve addresses in GPU work.
- Dispatch resident selected work directly or indirectly without copying route
  decisions to the host.
- Append compact missing-resource requests only on actual misses.
- Notify the scheduler only when a miss queue is non-empty.
- Prove that a fully warm turn has no per-token host residency decision or
  device-wide synchronization.

Completion requires trace-based proof of the fast path and matched eager versus
fully warm demand-retained microbenchmarks. A material warm-path regression must
be fixed before continuing.

### 11. Add scheduler backpressure and checkpoint resume

- Represent an activation blocked on one or more residency groups without
  blocking unrelated ready streams or devices.
- Deduplicate and batch missing groups, launch their load once, and wake all
  dependent activations after atomic publication.
- Preserve stream ordering, recurrent state, transient attention state, random
  state, and cancellation semantics across a pause.
- Resume from the residency checkpoint rather than restarting inference work.
- Expose bounded queues and fair scheduling so cold streams cannot starve warm
  work.

Completion requires deterministic interleaved-stream tests covering hits,
misses, shared misses, cancellation, load failure, and resume ordering.

### 12. Prove the Qwen routed-expert implementation without Qwen special cases

- Compile each main routed expert as one generic atomic group containing every
  tensor and scale required by its gate/up/down execution.
- Request exactly the groups selected by top-k routing.
- Keep routers, shared experts, recurrent/attention machinery, norms, state, and
  execution control in the always-resident spine.
- Load MTP resources only when MTP execution is enabled; route MTP experts
  through the same generic mechanism.
- Confirm that no runtime type, module, command-line option, or package schema
  mentions Qwen to implement the behavior.

Completion requires exact eager-versus-demand agreement for routes, generated
tokens, and persistent state across real multi-turn conversations with thinking
enabled.

### 13. Integrate runtime placement and multi-device ownership

- Make residency ownership explicit per physical device selected by the runtime
  graph.
- Support the same compiled model entirely on one device or split across
  multiple devices without recompilation.
- Initially load a requested resource on the device executing its component.
- Define the generic boundary for future choices between remote execution,
  activation transport, peer transfer, and a second resident copy; do not
  silently duplicate resources now.
- Preserve sharing and correct ownership when components are moved, duplicated,
  bypassed, or rewired.
- Make placement and device-slice inspection derive internal shard-worker
  ownership correctly. Inspection must neither reject a logical shard device
  merely because no whole component is assigned to it nor mount every component
  when asked to inspect one internal shard.

Completion requires sequential one-AMD and multi-AMD tests of placement changes,
duplicated components, graph rewiring, internal shard inspection, residency
reuse, and clean teardown.

### 14. Expose policy, state, and normal-operation metrics

- Add runtime and TUI selection for `eager` and `demand-retained` residency.
- Show why a resource is always resident or dynamically addressable before
  execution.
- Report initial, current, maximum, and high-water device bytes; resident and
  addressable unit counts; hits, misses, deduplicated loads, bytes read and
  uploaded, read/upload/blocking time, failed loads, and per-component coverage.
- Report MTP residency separately when it exists.
- Include the statistics in ordinary chat and benchmark summaries; do not add a
  profiling-only execution path.
- Keep normal human-readable output bounded. Full per-resource counter arrays
  belong in explicit machine-readable artifacts, not an unbounded chat dump.

Completion requires consistent CLI/TUI behavior and counters reconciled against
the runtime residency directory and device allocations.

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
