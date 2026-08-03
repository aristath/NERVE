# NERVE

**Neural Execution & Rewiring Virtual Engine**

NERVE is an experimental inference engine for running language models as long-lived, editable execution graphs instead of one-shot request/response jobs.

The design is documented in [`CONCEPT.md`](CONCEPT.md). The short version is:

- A model becomes a compiled, self-contained package.
- The package exposes reusable components, parameters, ports, kernels, transducers, state declarations, and a canonical topology.
- Runtime creates a concrete execution graph from that package.
- A stream is the primary runtime object.
- User prompts are events injected into an existing stream.
- Transient state belongs to the stream and is part of the running circuit.
- Placement and wiring are runtime decisions, not compiler decisions.

The original intuition was a guitar pedalboard with feedback, but the project now uses standard graph/compiler/runtime language: components, node instances, ports, edges, runtime graphs, placement, transports, streams, and transient state.

## Current status

NERVE is not a production inference engine yet. It is a real implementation of the architecture, but still under active construction.

### Currently implemented

- Safetensors source-model discovery and compilation.
- Self-contained compiled model directories under `compiled_models/`.
- A package manifest named `vulkan_resident_package.json`.
- Transpiled model graph artifacts under `transpiled/`.
- Lowered stream-circuit artifacts under `lowered/`.
- Packaged tensors, tokenizer files, SPIR-V shaders, runtime config, and artifact integrity metadata inside the compiled model directory.
- Rust runtime graph types with node instances, edges, placement, duplicate nodes, explicit chains, and state policies.
- Runtime device discovery and logical-to-physical device binding.
- Vulkan/SPIR-V resident package mounting.
- Interactive chat and one-shot prompt execution through `nerve-runtime`.
- A TUI entrypoint through `nerve-tui`.
- A stream scheduler with persistent streams, queued input events, chunked prefill, decode feedback windows, activation batching, timing counters, and normal chat performance reporting.
- Backend-neutral transient state arenas and per-stream state tables.
- Ref-counted transient state blocks with reset, fork, share, snapshot, and prefix-cache primitives.
- Route-native MoE compiler structures and shader families for selected expert routes.
- MTP/speculative decoding package structures and runtime flow.
- Shape-aware dispatch vocabulary for prefill versus decode.
- Versioned semantic-module trees and source provenance preserved through
  lowering, optimization, packaging, graph editing, and runtime inspection.
- CPU and Vulkan hardware-process discovery, empirical calibration, stable
  capability classes, and physical-device identity.
- An end-to-end behavioral representation optimizer that enumerates semantic
  scopes, analyzes source algebra and structure, asks registered providers for
  alternatives, constructs candidates in isolation, benchmarks matched
  reference/candidate pairs, validates behavior, and publishes only qualified
  implementations.
- Self-contained optimized packages that retain the immutable exact baseline,
  promoted implementation artifacts, proofs, benchmarks, validations, hardware
  evidence, runtime predicates, and package-wide integrity metadata.
- Runtime implementation selection after graph editing and placement, with
  exact fallback for uncovered compatible scopes and normal chat reporting of
  selected alternatives and representation-boundary cost.
- Generic compiled-resource contracts that separate always-resident execution
  structure from independently selectable immutable resource groups.
- A demand-retained residency policy: selected resources are loaded from the
  self-contained package on first access, atomically published to a stable
  sparse Vulkan address space, reused entirely on the device, and released by
  explicit package teardown.
- Compact affine residency metadata for regular partitioned resources and a
  group-table representation for irregular resources, so metadata and address
  bookkeeping scale with structure rather than expanded tensor counts.
- Physical-device resource stores with exact capacity accounting,
  content-addressed sharing, single-flight loads, deterministic failures, and
  placement-independent one- or multi-device execution.
- Default chat reporting for resource selection, hits, misses, I/O, uploads,
  memory watermarks, per-device stores, and acknowledged teardown.

### Important unfinished work

- The Vulkan backend still has transitional fixed/circular resident state buffers in places. The scheduler has page/block-managed transient-state semantics, but resident Vulkan bindings still need to become truly page-backed before automatic prefix reuse can be wired into normal prompt admission.
- Multi-stream activation batches are scheduled, but some placed Vulkan batch paths still execute activations sequentially internally.
- FP8, INT4, MoE, MTP, prefill, and decode paths all need more real-model benchmarking and kernel work.
- Demand-retained residency deliberately has no implicit eviction. If the
  accessed working set exceeds exact device capacity, execution fails
  deterministically. A future bounded-residency policy must be explicit,
  measured, and separately selectable; it must not weaken demand-retained
  semantics or silently page through host memory.
- The representation-optimizer machinery is complete, but the built-in
  provider library covers only the alternative representations implemented so
  far. Additional representation families remain product and performance work,
  not missing optimizer infrastructure.
- The TUI exists as a runtime surface, but the full graph-editing product experience is still being built.

## Repository map

### [`CONCEPT.md`](CONCEPT.md)

The architectural source of truth. Start here to understand the model: streams, compiled packages, components, runtime graphs, placement, transient state, feedback, and Vulkan/SPIR-V as the baseline backend.

### [`TODO.md`](TODO.md)

The current engineering goal, architectural invariants, completion criteria,
and any work that remains for that goal.

### [`nerve/`](nerve/)

Python compiler and CLI package.

| Path | Purpose |
| --- | --- |
| `nerve/cli.py` | User-facing command dispatcher. |
| `nerve/model_compiler.py` | High-level source discovery, staged compilation, and atomic publish into a self-contained compiled model directory. |
| `nerve/model_package.py` | End-to-end package build: transpile, lower, copy tokenizer/tensors, compile shaders, build manifest, validate package. |
| `nerve/model_transpiler*.py` | Source checkpoint discovery and conversion into model/circuit graph facts. |
| `nerve/circuit_*.py` | Stream-circuit IR, lowering system, lowering operators, and optimization. |
| `nerve/model_package_manifest.py` | Builds `vulkan_resident_package.json`. |
| `nerve/model_package_tensors.py` | Tensor packaging. |
| `nerve/model_package_shaders.py` | Shader packaging. |
| `nerve/model_package_shader_selection.py` | Shader selection. |
| `nerve/model_package_shader_templates.py` | Shader template rendering. |
| `nerve/model_package_sparse_projection_shaders.py` | Sparse BF16, FP8, and INT4 projection shader rendering. |
| `nerve/model_package_shader_compiler.py` | SPIR-V artifact creation. |
| `nerve/resource_residency.py` | Compiler-side addressable-resource discovery and compact residency contracts. |
| `nerve/conversation_gate.py` | Canonical resident multi-turn correctness and performance gate. |
| `nerve/representation_optimizer/` | Semantic scope enumeration, structural analysis, hardware targets, providers, isolated construction, matched benchmarking, behavioral validation, promotion, evidence storage, and package publication. |
| `nerve/representation_optimizer/ARCHITECTURE.md` | Complete schemas and invariants for the behavioral representation optimizer. |

CLI examples:

```bash
python -m nerve --discover-model MODEL_DIR
python -m nerve --compile-model MODEL_DIR
python -m nerve --run COMPILED_MODEL_DIR_OR_MANIFEST
```

### [`runtime-rs/`](runtime-rs/)

Rust runtime crate.

| Path | Purpose |
| --- | --- |
| `runtime-rs/src/bin/nerve_runtime.rs` | CLI runtime binary entrypoint. |
| `runtime-rs/src/bin/nerve_runtime/` | Prompt/chat execution, package inspection, runtime graph controls, placement flags, sampler options, device binding, chat templates, and reporting. |
| `runtime-rs/src/bin/nerve_tui.rs` | TUI binary entrypoint. |
| `runtime-rs/src/tui/` | Terminal UI application modules. |
| `runtime-rs/src/stream_circuit/` | Core graph and artifact model: components, ports, state ports, lowered graphs, runtime graph topology, runtime node instances, placement, routes, reports, and validation. |
| `runtime-rs/src/editor/` | Runtime graph editor schema and editor state used by UI-facing code. |
| `runtime-rs/src/stream_runtime.rs` | Backend-neutral stream scheduler. |
| `runtime-rs/src/stream_runtime_tests.rs` | Stream scheduler tests. |
| `runtime-rs/src/stream_state.rs` | Backend-neutral transient state arena and per-stream state tables. |
| `runtime-rs/src/stream_prefix_cache.rs` | Backend-neutral prefix/state reuse primitives: prefix keys, retained cache entries, longest-compatible-prefix lookup, block-aligned insertion, restore, ref counts, and eviction. |
| `runtime-rs/src/vulkan_compute/` | Vulkan device discovery, feature/capability handling, resident buffers, pipeline creation, dispatch, sequence submission, and buffer copies. |
| `runtime-rs/src/vulkan_stream_circuit/` | Vulkan resident package loading, placement, device slices, resident plan buffers, dispatch binding, prompt streams, placed prompt engine, token runtime, sampler, speculative decode, batching, distributed execution, and reusable kernel/sequence machinery. |
| `runtime-rs/src/vulkan_stream_circuit/compiled_resource_device_store.rs` | Per-physical-device resource lifetime, sharing, capacity, and single-flight load coordination. |
| `runtime-rs/src/vulkan_stream_circuit/device_resource_residency.rs` | Stable sparse Vulkan resource arena, compact address mapping, atomic publication, and unload. |
| `runtime-rs/src/vulkan_stream_circuit/resource_backing_store.rs` | Verified reads from self-contained compiled-resource payloads. |
| `runtime-rs/src/implementation_selection/` | Target-predicate validation, compatible-scope enumeration, measured-cost selection, and exact/alternative implementation planning. |
| `runtime-rs/shaders/` | GLSL compute shader templates and generated/compiled shader inputs for BF16, FP8, INT4, attention/state, recurrent/conv, sampler, MoE, and related runtime operations. |

### [`tests/`](tests/)

Python compiler/package tests.

## Behavioral representation optimization

The optimizer is the concrete implementation of
[`CONCEPT.md`](CONCEPT.md)'s behavioral-compilation rule: the exact compiled
model specifies behavior, but does not dictate the permanent physical
representation. The executable flow is:

```text
immutable exact package
    -> semantic and coupled scopes
    -> algebraic and structural evidence
    -> registered representation providers
    -> isolated candidate construction and ordinary re-lowering
    -> matched binary microbenchmark
    -> proof, local, and whole-model validation
    -> target-guarded self-contained package publication
    -> runtime selection after graph editing and placement
```

The exact lowered graph remains present throughout. A rejected, failed, slower,
or invalid candidate cannot mutate it or become runtime-visible.

### Hardware-process profiles

A `hardware_process_profile.v1` describes what a CPU or GPU can actually expose
to an implementation: scalar and vector arithmetic, matrix instructions,
packed dot products, subgroups, caches, local memory, texture and fixed-function
paths, indirect execution, copy engines, synchronization, interconnects, and
other discoverable processes. Unsupported and API-opaque facilities are
recorded explicitly. Calibration adds measured performance without changing the
underlying capability class.

Capability and physical identity are separate. Equivalent devices can share a
capability class while retaining distinct stable device IDs and runtime
bindings. The optimizer qualifies a candidate against exact capability-class
multiplicities and an execution regime; it does not publish the benchmark
machine's placement as a model requirement.

### Provider contract

A representation provider is a complete, model-independent strategy conforming
to `RepresentationProvider` in
`nerve/representation_optimizer/providers/protocol.py`. It must:

1. match semantic responsibility and structural evidence separately;
2. interpret cited evidence and synthesize deterministic candidate contracts;
3. emit backend-neutral `representation_graph.v1` IR;
4. lower that IR for a hardware-process profile;
5. estimate feasibility, storage, construction, and steady-state work;
6. declare construction and runtime-mount requirements;
7. provide exact proof obligations or an explicit approximation error contract;
8. declare matched benchmark workloads; and
9. declare complete validation coverage.

Providers are registered by provider identity plus a data-defined representation
descriptor. They are evaluated in deterministic order, may decline normally,
and fail independently. The registry has no model-name dispatch table.

### Candidate lifecycle and benchmark rule

Each candidate has an immutable evidence-linked lifecycle:

```text
synthesized
    -> staged
    -> statically_validated
    -> prebenchmark_validated
    -> benchmarked
    -> behaviorally_validated
    -> promotable
    -> published
```

Construction occurs in a private, source-sealed workspace. The provider's
semantic constructor, ordinary re-lowerer, and physical optimizer can write only
declared outputs. Every source input and produced byte is digest checked before
an atomic ready rename.

A microbenchmark answers only “is the candidate faster?” For each declared
role and workload, the engine makes one discarded warmup call and one matched
measured call through the normal execution adapter. Reference and candidate
must perform identical useful work. Setup, mounting, execution, conversion,
transport, synchronization, teardown, and device-state restoration remain
separate evidence. A microbenchmark that exceeds one minute fails instead of
collecting more samples.

### Validation, promotion, and runtime selection

The rejection funnel is static integrity, exact or bounded-error proofs, cheap
numerical/state sanity, matched performance, full local behavior, and
whole-model free-running behavior. Exact candidates reproduce every declared
output and state observation exactly. Approximate candidates must satisfy a
versioned error contract over their complete claimed validity regime.
Long-context, reasoning, state, graph, placement, and conversation checks cannot
be replaced by a tiny convenience token limit.

Promotion creates a new package; it never edits the exact source package in
place. The published package contains the alternative implementation, mount
plan, hardware profiles, runtime predicate, all cited evidence, and a rebuilt
package-integrity contract. It is independently loadable and relocatable.

At runtime, NERVE first resolves duplication, bypass, rewiring, sharding,
placement, and physical device bindings. It then matches promoted predicates
against the resulting graph, hardware processes, device-count multiplicity,
interconnects, execution phase, activation/context/state envelope, and
qualified speculative-draft-token count. A measured-cost planner selects the
best compatible non-overlapping alternatives, including conversion costs.
Uncovered compatible regions retain the immutable exact implementation.

### Running the optimizer

The optimizer consumes a compiled package, selected physical devices, and the
same speculative regime intended for product execution:

```bash
python -m nerve \
  --optimize-model COMPILED_MODEL \
  --optimized-model-dir COMPILED_MODEL_OPTIMIZED \
  --optimizer-run-dir OPTIMIZER_EVIDENCE \
  --allow-physical-device vulkan-uuid:FIRST_AMD_UUID \
  --allow-physical-device vulkan-uuid:SECOND_AMD_UUID \
  --vulkan-driver-manifest /path/to/radeon_icd.json \
  --speculative-draft-tokens 2
```

The output model is the deployable artifact. The run directory contains
rejected and failed experiment evidence and is not a runtime dependency.
Device allowlisting, live reservable-capacity verification, sequential
execution, and clean release remain mandatory. Existing AMD workloads are
recorded reservations, not device-level exclusions: NERVE may use the measured
unreserved VRAM while preserving those workloads.

Automatic optimizer placement measures each compatible AMD device's current
VRAM usage, reserves a conservative share of what remains, selects the smallest
admissible device set, and partitions the ordered component stream
proportionally to those per-device capacities. Components stay contiguous to
minimize cross-device edges, while devices are ordered by physical PCI topology.
The lease lock serializes NERVE optimizers only; it never claims exclusive
ownership of a physical GPU or evicts another process.

### Adding a representation provider

To add a representation without changing model-specific runtime code:

1. add or load a validated `representation_descriptor.v1` document;
2. implement every method in `RepresentationProvider`;
3. implement a `CandidateToolchainResolver` that supplies the semantic
   constructor, ordinary re-lowerer, physical optimizer, and artifact
   validators declared by the candidate;
4. register the provider, descriptor, and resolver in the product's provider
   assembly;
5. add contract, rejection, exactness/error, construction-integrity,
   benchmark, validation, promotion, relocation, and runtime-selection tests;
   and
6. qualify it on the intended hardware and execution regime.

Model structure is discovered through semantic scopes and evidence. A provider
must not branch on a model family name. Full schema detail and extension
invariants are in
[`nerve/representation_optimizer/ARCHITECTURE.md`](nerve/representation_optimizer/ARCHITECTURE.md).

## Validation

Tests in this repository must run sequentially. The Python suite disables xdist,
and Rust test invocations must always include `-- --test-threads=1`. Vulkan tests
must be selected individually after binding an explicitly verified AMD device
with sufficient unreserved capacity; an explicit device that cannot be opened
is a test failure, never a passing skip.

Compiler/package tests:

```bash
.venv/bin/python -m pytest -p no:xdist
```

Backend-neutral Rust tests:

```bash
cargo test --manifest-path runtime-rs/Cargo.toml --lib -- --test-threads=1
```

[`scripts/run_conversation_gate.py`](scripts/run_conversation_gate.py) drives the
normal `nerve-runtime --chat` interface. It keeps each model resident, sends
`hi` as the discarded warmup, sends the five canonical measured turns, retains
the 65,536-token output allowance, parses the runtime's default statistics, and
recognizes both closing-tag and decoded-channel reasoning protocols. It fails on
malformed thinking output, repeated output, turn contamination, incorrect
cross-turn recall, or a missed throughput floor. Each invocation runs exactly
one fixed sampler seed. Repeat the command for other seeds only after verifying
that NERVE released the capacity it acquired and the recorded pre-existing
allocations remain present:

Demand-resident sparse packages require a stronger steady-state measurement.
Pass `--warmup-conversation-sets 1` to run the complete canonical six-input
conversation twice in one uninterrupted model process. The first conversation
is validated but discarded; only the second conversation is used for the
throughput gate. The report includes cumulative residency counter snapshots and
their per-conversation deltas, so a second set that still loads or reloads
resources is visible rather than mislabeled as fully warm.

```bash
.venv/bin/python scripts/run_conversation_gate.py \
  --seeds 0 \
  --minimum-decode-tps 20 \
  --require-thinking \
  --warmup-conversation-sets 1 \
  --transcript-dir /tmp/nerve-conversation-gate \
  --report /tmp/nerve-conversation-gate/report.json \
  -- \
  runtime-rs/target/release/nerve-runtime \
  --package COMPILED_MODEL/vulkan_resident_package.json \
  --chat \
  --context-size 131072 \
  --speculative-draft-tokens 3 \
  --bind-device gpu0=vulkan-uuid:AMD_DEVICE_UUID \
  --bind-device gpu1=vulkan-uuid:SECOND_AMD_DEVICE_UUID
```

The command and compiled package schema, compiler target, compiled shader
variant count, per-turn statistics, responses, and transcript hashes are
captured in the report. Physical bindings are also printed in the normal chat
readiness line, so a benchmark transcript identifies the devices it actually
mounted.

## Demand-retained compiled resources

`--residency-policy demand-retained` changes parameter residency, not model
semantics. The compiled package declares resources that may be selected
independently and an always-resident execution spine. The runtime resolves
placement against the physical devices selected for that run. Mounting reserves
a stable sparse virtual address range but commits no device-local payload pages
for unselected dynamic resources.

Execution then follows one contract:

```text
GPU selection -> resident gate
    hit  -> continue on the GPU
    miss -> pause at the physical checkpoint
         -> verified backing-store read
         -> upload every member of the atomic group
         -> publish residency
         -> resume at the checkpoint
```

No token or completed component is replayed. A resident hit has no host round
trip. The resource remains resident until explicit unload, and teardown
acknowledges every physical-device store before the package is released.
Regular partitioned resources use affine address metadata; genuinely irregular
groups use compact tables. The mechanism is semantic and model-family neutral:
the same path has been exercised with routed expert partitions and with an
irregular, optional output-head package whose resource order differs from its
compiled member order.

The qualification run on 2026-07-29 used the compiled FP8
Qwen3.6-35B-A3B package across two AMD Vulkan devices, 131,072-token context,
65,536-token output allowance, thinking enabled, seed 0, and two MTP draft
tokens. One discarded warmup and five retained-state conversation turns
produced meaningful responses and correctly recalled Greece from an earlier
turn. The five measured turns averaged:

| Metric | Result |
| --- | ---: |
| Generated throughput | 42.420 tok/s |
| Decode throughput | 49.152 tok/s |
| Prefill throughput | 70.283 tok/s |
| Initial device bytes | 6.701 GB |
| Final/high-water device bytes | 36.184 GB |
| Final selected payload | 29.448 GB |
| Addressable dynamic payload | 33.022 GB |
| Final resident resources | 9,360 / 10,496 |
| Load failures | 0 |

Both stores acknowledged teardown, released all 9,360 resources and 29.448 GB
of payload, and both GPUs returned to their recorded pre-run capacity
reservations with no NERVE allocation remaining. This
proves demand-triggered loading and retained reuse; it does not claim that every
prompt will touch a bounded subset. Workloads whose cumulative working set
exceeds capacity need a future explicit bounded-residency policy.

## Concept to implementation

The table below maps the language in [`CONCEPT.md`](CONCEPT.md) to the implementation that currently carries it.

| Concept term | Implementation |
| --- | --- |
| Compiled model | `compiled_models/<slug>/`, built by `nerve/model_compiler.py` |
| Package manifest | `vulkan_resident_package.json`, built by `nerve/model_package_manifest.py` |
| Component | `StreamCircuit` in `runtime-rs/src/stream_circuit/graph.rs` |
| Port | `CircuitPort` and `StatePort` in `stream_circuit/graph.rs` |
| Node instance | `StreamCircuitNodeInstance` in `stream_circuit/runtime_graph/instances.rs` |
| Runtime graph | `StreamCircuitRuntimeGraph` in `stream_circuit/runtime_graph/graph.rs` |
| Placement | `stream_circuit/placement.rs` and runtime CLI placement flags |
| Edge / transport | `stream_circuit/runtime_routes.rs` and `vulkan_stream_circuit/edge_*.rs` |
| Stream | `RuntimeStreamScheduler` stream state plus placed prompt streams |
| Transient state | `stream_state.rs` and `stream_prefix_cache.rs` |
| Permanent circuits | Package tensors, resident buffers, SPIR-V shaders, component executions |
| Demand-resident resource | Package-declared immutable group plus stable sparse Vulkan placement |
| Resource selection | Compiled semantic selection domain and GPU resident gate |
| Physical resource store | Per-device shared, capacity-accounted compiled-resource store |
| Input transducer | Package manifest `input_transducer` and Vulkan resident package loader |
| Output transducer | Package manifest `output_transducer` and token output pipeline |
| Sampler | Package sampler spec/kernels and runtime sampler config |
| Device-owned loop | Placed prompt stream / feedback window execution in `vulkan_stream_circuit` |
| Runtime graph editor | `runtime-rs/src/editor/` and `runtime-rs/src/tui/` |
