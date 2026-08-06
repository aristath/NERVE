# Behavioral representation optimizer architecture

## Compiler position

The behavioral representation optimizer consumes the immutable exact semantic
execution graph after source discovery, transpilation, semantic lowering, and
required exact dtype lowering. It runs before target-specific physical
implementations and the final package manifest are published.

The compiled package records this boundary in `optimization/stage.json`. The
stage artifact contains:

- the package-relative exact-baseline artifact;
- its canonical contract digest;
- an optimization session rooted in that digest;
- the versioned contract schemas understood by the compiler; and
- an explicit status that distinguishes an untouched exact baseline from a
  package containing promoted representations.

The exact baseline remains available even when additional implementations are
later promoted. Representation optimization adds verified implementations; it
does not erase the semantic specification.

## Contract identities

Contracts use canonical, finite JSON. Object-key order and insignificant JSON
formatting cannot change an identity. Semantic optimization-scope identifiers
are derived from:

- package identity;
- scope kind;
- ordered component identities;
- ordered semantic-module identities; and
- ordered source-node identities.

The source behavioral contract has its own content digest. Candidates,
evidence, construction records, benchmarks, validations, promotions, and
re-lowering requests reference these identities rather than model-family names
or filesystem accidents.

Schema versions are independent. A reader fails closed on an unknown schema,
unknown field, invalid stable identifier, malformed digest, duplicate identity,
non-finite number, or internally inconsistent contract.

## Versioned contracts

| Contract | Responsibility |
| --- | --- |
| `optimization_scope.v1` | Semantic region and its exact boundary |
| `optimization_scope_catalog.v1` | Deduplicated scopes, linked source contracts, and rejected-scope diagnostics |
| `source_behavior_contract.v1` | Observable source behavior and exact implementation |
| `algebraic_evidence.v1` | Analyzer identity, structural claims, and evidence artifacts |
| `hardware_process_profile.v1` | Target identity, processes, measurements, and provenance |
| `representation_descriptor.v1` | Open representation-family vocabulary, evidence requirements, physical forms, costs, and correction paths |
| `representation_candidate.v1` | Proposed representation, target predicate, and error contract |
| `representation_graph.v1` | Logical contracts, physical forms, resources, transducers, islands, corrections, and provenance |
| `candidate_build_plan.v1` | Ordered construction phases, sealed source inputs, declared outputs, validators, and resource limits |
| `candidate_construction.v1` | Isolated construction result, artifacts, resources, and diagnostics |
| `source_package_seal.v2` | Immutable baseline, stage, package-integrity, and source-input evidence |
| `staged_candidate_integrity.v1` | Complete byte coverage for one atomically ready candidate |
| `benchmark_workload.v1` | Immutable input, state, randomness, useful work, and validity regime |
| `benchmark_plan.v1` | Counterbalanced matched reference/candidate experiment |
| `benchmark_observation.v1` | One normal-runtime timing, work, resource, and trace observation |
| `benchmark_residency_event.v1` | Mount/unmount cost and device-state evidence |
| `benchmark_run.v1` | Ordered raw observations and residency lifecycle |
| `benchmark_record.v2` | Binary matched-speed result and measured evidence |
| `benchmark_evidence_integrity.v1` | Complete byte coverage of benchmark evidence |
| `behavioral_error_contract.v1` | Approximation validity predicates, metric limits, and correction policy |
| `validation_requirements.v1` | Proof verifiers, behavioral checks, applicability map, and counterexamples |
| `validation_plan.v1` | Candidate-bound implementations, matched conditions, and ordered checks |
| `proof_result.v1` | Request-bound exact or bounded-error proof evidence |
| `validation_observation.v1` | Paired output, state, metric, trace, seed, and horizon evidence |
| `validation_residency_event.v1` | Validation mount/unmount device-state evidence |
| `validation_run.v1` | Ordered sanity, local, or whole-model observations |
| `prebenchmark_record.v1` | Static integrity, proof, and cheap-sanity gate result |
| `validation_record.v2` | Complete funnel, benchmark link, full runs, warmed product-performance gate, and counterexamples |
| `validation_evidence_integrity.v1` | Complete byte coverage of validation evidence |
| `runtime_implementation_predicate.v5` | Exact measured hardware-profile identities plus capability multiplicities, explicit alternative/source-retained phase ownership, execution-regime and placement guards, and qualified speculative-decoding modes for one verified implementation |
| `runtime_mount_plan.v3` | Runtime-adapter identity, connected replacement regions, and candidate-local component-overlay and tensor-index artifacts |
| `vulkan_component_overlay.v2` | One component's candidate circuit, execution kernels, and sorted parameter-level resident derivations |
| `promotion_decision.v2` | Candidate, proof, benchmark, validation, target, artifact, and provenance decision |
| `implementation_registry.v1` | Exact baseline plus all published physical implementations |
| `relowering_request.v1` | Representation-aware request to repeat ordinary lowering passes |

Every contract round-trips through the same canonical serializer. The canonical
SHA-256 contract digest is for compiler identity and evidence linkage; package
artifact integrity independently protects the bytes published on disk.

## Representation descriptors

A representation descriptor is not a candidate and is not a hardcoded list of
implementations. It is a registerable, model-independent declaration of a
family of possible expressions. Descriptors state:

- the semantic responsibilities and composition scopes they may express;
- the exact or sampled evidence required before considering them;
- their signal, parameter, and state forms;
- supported topology and time models;
- compatible CPU and GPU processes;
- construction phases and artifact lifetimes;
- accepted and produced boundary formats plus measured conversion costs;
- whether adjacent scopes may retain the native representation;
- proof obligations, validity predicates, and approximation error contracts;
  and
- exact fallback or correction paths.

Built-in descriptor documents live in
`nerve/representation_optimizer/descriptors/`. The registry loads every JSON
document in that directory through the same strict contract validator and also
accepts external descriptor documents. Neither responsibility names nor
representation kinds are closed enums, so a new expression does not require a
model-family conditional or a central catalog edit. Namespace, name, and
version collisions fail closed, while each descriptor's content-derived
identifier prevents silent semantic drift.

The initial data-defined vocabulary spans structured transforms with
exceptions, lookup/codebook circuits, indexed search, sampled fields, generated
programs, packed symbolic logic, sparse event graphs, bounded multiscale state,
hierarchical output construction, reconstructed parameter streams,
coarse-to-fine evaluation, verified correction, and heterogeneous
representation islands. These are composable examples from `EXPERIMENTS.md`,
not an exhaustive list.

Descriptor identity and contents are recorded in every analysis and candidate
contract. Published implementations are accepted only when those content
digests, their generated artifacts, and the package's exact source seal all
match. The package schema is the compiler/runtime ABI boundary; tool source
changes that leave that ABI intact do not invalidate compiled models.

## Semantic optimization scopes

The scope enumerator walks the lowered semantic module trees together with the
inter-component dependency graph. It emits architecture-neutral regions for
individual source operators, unambiguously owned leaf modules, generic coupled
sibling modules, token mixers, feature transforms, complete layers, state
writer/reader systems, corresponding modules across layers, adjacent
producer/consumer representation boundaries, and complete input, output,
sampling, or feedback transducers.

Scopes refer to qualified component/module/node identities. A cross-component
representation island contains only the executable producer and consumer at
that connection; it does not accidentally absorb both complete components.
Every internal cross-component dependency retains its edge identity, endpoint,
connection kind, temporal-delay metadata, and covered consumers. The linked
source contract also references the lowered execution-graph artifact, so a
feedback or transport boundary cannot be presented as exact while silently
discarding its wiring semantics.
Scopes with the same executable semantic region are merged once and retain all
applicable classifications. Ambiguous module ownership or dependency
boundaries are rejected into deterministic diagnostics rather than becoming
optimization candidates.

Every accepted scope records exact external data inputs, outputs, parameter
references, state reads/writes, controls, and explicit or implicit randomness.
It is paired one-to-one with an immutable source-behavior contract referencing
the exact lowered artifacts. The self-contained catalog lives at
`optimization/scopes.json`; its content-derived identity, link digests,
classification counts, and stage reference all fail closed on drift.

## Algebraic and structural evidence

`nerve.representation_optimizer.analysis` examines an enumerated scope without
changing its exact source implementation. The engine decodes effective BF16,
block-scaled FP8, native NERVE Q8, and group-scaled packed INT4 parameter values
through a package tensor repository. Every exhaustive or deterministic-grid
observation carries its domain, storage format, effective-value status, and
sample indices; a sampled observation cannot become an exact proof.

Independent analyzers cover elementwise structure, matrix and tensor
factorizations, coupled parameters, semantic graph topology, procedural
generators, and optional reachable activations. Coupled analysis canonicalizes
permutation-equivalent coordinates before comparing tensors and searches for
shared subspaces, generators, repeated experts, and cross-component motifs.
Numerical tolerances remain distinct from exact equality. Activation traces
must declare their prompts, positions, seeds, or other reachable domain and are
always refinement evidence rather than exhaustive behavioral proof.

An analysis run emits one content-identified `algebraic_evidence.v1` record and
one integrity-checked details artifact per analyzer. The run index, evidence,
and details are deterministic, written atomically to a fresh output directory,
and fail validation after content drift. This stage records hypotheses and
proof evidence only: it neither synthesizes a replacement nor mutates the
compiled package.

## Representation providers

A representation provider is a complete, independently registered strategy for
turning evidence into one alternative physical implementation. The provider
boundary requires separate semantic and structural matchers, provider-specific
evidence interpretation, candidate synthesis, backend-neutral representation
IR, target lowering, static feasibility and cost estimates, construction and
mount requirements, a proof or approximation contract, benchmark workloads,
and validation requirements. A provider can decline a scope normally at either
matcher or after evidence analysis.

`ProviderProblem` validates and freezes the selected scopes, their one-to-one
source contracts, algebraic evidence, and hardware profile. Each provider sees
copy-out documents bound to its registered data-defined representation
descriptor. A structural match and accepted evidence analysis must cite
evidence inside that problem; the resulting candidate permanently records the
same evidence references and descriptor identity. Hardware availability by
itself cannot pass the structural matcher.

`ProviderRegistry` has no model-family dispatch table. Providers register by
identity and descriptor and are evaluated in deterministic identity order. A
provider failure is reported without preventing independent providers from
running. Candidate identities derive from canonical candidate content, and the
registry removes semantically equivalent candidates across providers while
retaining an audit record of the kept and discarded identities. Candidate
plans retain the provider's representation IR, target lowering, estimates,
requirements, benchmark workloads, and validation obligations for subsequent
staging.

## Physical representation graph

Provider IR is one shared `representation_graph.v1` contract rather than an
opaque provider-owned document. It preserves the public logical signal, shape,
and dtype separately from every physical encoding. Signals, parameters,
transient state, and topology can therefore use dense tensors, sparse records,
fields, codebooks, programs, graphs, or future provider-defined forms without
changing the semantic boundary seen by graph editing.

The graph makes conversions ordinary `transducer` nodes. Their physical input
and output representations and estimated or measured cost are explicit; an
incompatible connection cannot imply a hidden conversion. A representation
island binds adjacent semantic scopes, physical nodes, native connections, and
boundary ports so intermediate source-format activations need not be
materialized. Proven basis or encoding transforms can instead be absorbed into
adjacent parameter artifacts.

Every signal, physical resource, node, transform, and kernel retains source
scope, source-node, transformation, and evidence provenance. Confidence is
explicitly exact, verified approximation, or unresolved. Approximate graphs
must expose correction requests and their error contracts, while unresolved
graphs retain the subjects and evidence that prevent promotion.

The generic planner validates the graph, rejects executable cycles and
incompatible physical edges, derives a deterministic execution order, accounts
for transducer and kernel costs, and identifies native cross-scope connections
versus source materializations. Rust consumes the same versioned schema and
provides a lossless editor inspection view; execution and package publication
remain later lifecycle decisions.

## Candidate lifecycle

A candidate moves only through the following evidence-carrying sequence:

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

At each active state it may instead become `rejected`, `cancelled`, or `failed`
where allowed. Forward progress, rejection, and failure require an evidence
reference. Cancellation may record only a reason because it is not an
evaluation result.

The lifecycle is immutable: a transition returns a new session value. Its
history is contiguous and replay-validated when loaded. Skipped phases,
retroactive edits, duplicated candidates, missing evidence, and transitions
from terminal states are rejected.

`statically_validated` means the staged contract, build plan, construction
record, source-package seal, artifact validators, and integrity manifest all
agree. `prebenchmark_validated` additionally means every declared proof
obligation and cheap numerical/state sanity check passed. Static integrity
alone cannot make a candidate eligible for timing.

## Failure isolation

An optimization session contains the immutable exact-baseline digest and
independent candidate lifecycles. Failing or rejecting one candidate changes
only that candidate's new session value. It cannot:

- mutate the exact semantic graph;
- mutate another candidate;
- publish partial artifacts;
- mark another candidate as evaluated; or
- turn a failed experiment into a runtime implementation.

Filesystem staging and atomic publication build on this contract during
candidate construction and target-guarded package publication.

## Isolated candidate construction

Candidate construction never writes into the compiled source package. A build
plan declares every source input by path and SHA-256 digest, every output by
kind and lifetime, the phase that owns it, its resident byte cost, and the
validator contract that must accept it. The staging engine exposes only
digest-checked source reads and declared-output writes to construction code.
Both boundaries support bounded chunk streams; multi-gigabyte parameters,
indexes, and codebooks are never forced through one Python byte array merely to
construct, hash, validate, seal, or load them. Paths cannot escape the private
workspace, use staging-engine contract paths, cross symbolic links, or
overwrite an earlier artifact.

Every candidate runs three ordered phases: semantic construction, ordinary
re-lowering, and physical optimization. A representation change therefore
cannot bypass the normal lowering/optimization boundary. Providers may add
artifact validators to the model-neutral registry; built-in validators cover
JSON contracts, non-empty structured binary payloads, and SPIR-V headers.
Unregistered validators, missing outputs, undeclared files, corrupt JSON or
SPIR-V, resource-limit violations, source drift, and phase failures all fail
closed.

The engine measures phase and total construction time—including source
sealing—peak temporary memory, peak staging storage including engine-owned
integrity evidence, generated bytes, and final resident bytes. It writes the
candidate, representation graph, target lowering, build plan, and re-lowering
request beside the declared artifacts, then covers every byte with one
integrity manifest. Integrity bytes remain subject to the declared staging
limit. Only after re-validation and a second source-package seal check does one
atomic rename make the candidate ready.

Cancellation and failure delete the private tree, retain only an auditable
construction record, and transition only that candidate's immutable lifecycle.
Per-candidate filesystem locks serialize construction without serializing
unrelated CPU analysis. Private directories carry their candidate identity, so
an interrupted attempt is removed on retry. If interruption happens after the
atomic ready rename, a complete ready tree plus its durable record is recovered
without reconstruction; a ready tree whose record was not yet committed is
removed before a clean retry. Corrupt or ambiguously linked publications fail
closed instead of being silently replaced.
The ready-candidate loader rechecks integrity, kind-validation evidence,
construction-record links, all contract digests, and optionally the live source
seal before any later benchmark or execution phase can consume the candidate.

## Matched candidate benchmarking

The benchmark engine mounts the exact source implementation and staged
candidate through the same normal execution-adapter contract. It accepts only
hardware profiles that exactly describe the declared devices and an explicit
semantic-scope-to-device placement. Workloads bind immutable input and initial
state bytes by digest, declare control and randomness contracts, and describe
actual execution regimes: prefill or decode, activation width, context and
state size, stream count, cold or resident mounting, and local or cross-device
boundaries. A bounded output window must cite a digest-bound source-model
metadata artifact and JSON pointer, or a candidate validity predicate; an
arbitrary convenience cap cannot form a valid workload.

Before mounting anything, the runner streams and verifies every workload
fixture. It then makes exactly one discarded warmup call and one measured call
for the reference and candidate under each representative workload. Candidate
and reference must complete identical useful work. The microbenchmark has a
hard one-minute wall-clock ceiling; crossing it is a failed experiment, not a
reason to collect more samples.
The normal runtime supplies its default statistics plus separate counters for
useful, speculative, cancelled, discarded, and corrective work; setup,
execution, teardown, queueing, synchronization, transport, conversion,
residency, memory, and device-utilization costs remain distinct.

Every observation retains content-addressed distribution, token, state,
random-draw, and schedule traces. Repetitions are classified as identical,
permitted sampling variance, permitted numerical nondeterminism, speculative
scheduling variance, or a correctness defect. The raw trace bytes are streamed
again into the published evidence tree and covered by its integrity manifest,
so a digest label without its auditable evidence is insufficient.

Inputs, initial states, and limit evidence are preserved beside those traces.
The summary answers one binary question from that measured pair: whether the
candidate is faster. Sustained-window throughput within each call remains a
guard against degradation, while complete correctness and behavioral
qualification happen only after the fast screen. Publication is an atomic
rename of the deterministic plan, ordered raw run, summary, traces, and exact
integrity manifest. Failed mount validation closes the already-open session,
and every successful mount must prove that unmount restored the matched
capacity reservation.

Role unmount and executor shutdown are separate protocols. Unmount resets a
role and may retain immutable residency for the next matched call. Shutdown
first quiesces every submitted queue, drops the reusable role, then releases
pooled allocations and destroys Vulkan contexts one physical device at a time.
The executor acknowledges the ordered release only after no registered device
or pooled buffer remains; the host re-attests the declared reservable capacity
only after that acknowledgement. Other processes and their allocations are not
part of the NERVE lease and are never unloaded to manufacture an idle device.
Normal completion never treats stdin EOF, process exit,
destructor order, or an expired experiment deadline as accelerator teardown.
Mount and execution commands are bounded cancellation quanta: cancellation is
checked before submission, while a submitted command is allowed to return to
the protocol boundary before ordered release. A deadline therefore rejects the
experiment without turning live accelerator state into an asynchronous process
kill.

Changing a mounted validation role is also an accelerator residency
transaction. The executor quiesces every participating device before dropping
an incompatible role. If the next role moves parameters between devices, idle
pooled allocations are evicted one physical device at a time before any
replacement allocation; same-placement reference/candidate transitions may
retain shared immutable buffers. This ordering prevents a placement check from
temporarily materializing two complete model copies and forcing driver-managed
VRAM eviction.

## Proof and behavioral-validation funnel

Validation is an ordered rejection funnel, not one similarity score:

```text
static contracts and artifacts
    -> exact or bounded-error proof obligations
    -> cheap numerical and state sanity
    -> matched performance gate
    -> full local behavior
    -> whole-model free-running behavior
```

Static integrity, proof, and cheap sanity run before benchmarking. Full local
validation runs only after a statistically material matched speed win, and
whole-model free-running validation runs only after full local behavior passes.
Only the complete funnel may produce `behaviorally_validated`.

Every provider returns the shared `validation_requirements.v1` contract. It
classifies every validation dimension as required with concrete checks or not
applicable with a reason. Output comparison, teacher-forced behavior,
long-horizon free-running behavior, multiple fixed seeds, source-declared
context and output limits, graph editing, and alternative placement cannot be
waived. Stateful, routing, memory, correction, reasoning, lifecycle, and
counterexample checks become mandatory whenever their semantic responsibility
applies. Coverage is constrained to compatible check kinds and stages, so a
component comparison cannot masquerade as a whole-model conversation, graph,
placement, or state-lifecycle test.

Exact candidates require named proof obligations and must reproduce paired
output, transient state, and declared metrics exactly. Approximate candidates
may prove bounded claims, but every observed error metric must belong to an
explicit `behavioral_error_contract.v1`. That contract binds numeric maximum
errors to behavioral dimensions, a validity regime, and a correction or
rejection policy. Generated-text equality is neither the sole acceptance
criterion nor a substitute for distribution, rank, route, memory, calibration,
state, and long-horizon evidence.

Checks execute reference and candidate pairs through a backend-neutral
normal-runtime adapter. They bind immutable input, initial-state,
counterexample, context-limit, and output-limit evidence by digest. Long
context and output allowances must resolve through source-model JSON metadata;
a convenient small cap is not valid evidence. Full and whole-model checks use
at least two fixed seeds and an explicit minimum executed horizon.

Each validation-stage mount records device state before and after residency.
Teardown must restore the declared capacity-reservation digest even on
rejection or execution failure. Raw traces, fixtures, plans, runs, records, and integrity manifests
are streamed into atomically published evidence trees. Unproven exact
candidates never reach timing; faster approximations that exceed any declared
error limit are rejected before promotion.

## Target-guarded promotion and package publication

Promotion is a new compiled package, never an in-place mutation of the exact
source package. A candidate is eligible only in `behaviorally_validated` state
and only when its complete matched benchmark record—and every workload regime
represented by the resulting guard—reports `materially_faster`. The linked
validation record must report `passed`. Preparation reloads the candidate,
source seal, raw prebenchmark evidence, raw matched benchmark, full validation,
cited analysis runs, and measured hardware profiles from their integrity
checked publications before it creates a promotion decision.

The runtime predicate is derived from evidence rather than supplied as an
unverified label. It records exact measured hardware-profile identities,
allowed capability classes, device kinds, APIs, required processes and
features; prefill, decode, component, mixed, or state-transition phases;
measured activation-batch, context, and state ranges; and local, distributed,
or either placement with an exact device-count range and any required
interconnects. Physical profile documents remain in the published bundle. An
implementation may run only on a physical profile represented in its measured
evidence; matching a capability class alone does not prove equal realized
performance.

Each registry entry retains:

- the exact semantic scopes and source-contract digests it implements;
- the representation and behavioral contract;
- the target predicate;
- the complete candidate artifact integrity contract;
- the provider, descriptor, analysis, representation-graph, target-lowering,
  and re-lowering provenance;
- the exact implementation and per-regime paired comparison it beat;
- full prebenchmark, benchmark, and validation evidence; and
- the explicit promotion reason.

Several candidate identities may implement the same semantic scopes. Their
predicates distinguish the regimes in which each measured representation is
eligible. The immutable exact baseline remains in both `stage.json` and
`implementations.json`; promotion only adds physical choices.

Publication clones the self-contained source package into a sibling private
tree. Linux reflinks avoid rewriting immutable model bytes when the filesystem
supports them; the fallback is an independent streaming copy, never a hard
link. Candidate artifacts and all cited evidence are copied into
`optimization/implementations/<implementation_id>/`. The implementation
registry, package-local lifecycle references, optimizer stage, and whole
package artifact-integrity manifest are rebuilt and fully reloaded before one
atomic rename exposes the destination. A second post-rename validation runs
before success is returned. Any failure removes the private tree or the newly
renamed destination and leaves the source package byte-for-byte unchanged.

Optimized packages contain only published candidate lifecycles. Every
lifecycle evidence reference resolves to a file inside that package; rejected
and failed experiment workspaces do not leak into the deployable model.
Relocation validation therefore needs neither the source package nor analysis,
construction, benchmark, or validation workspaces.

## Runtime implementation selection and mounting

Placement remains a runtime decision. After graph edits, duplication, bypass,
rewiring, sharding, and logical-to-physical device binding are resolved, NERVE
builds a selection request from the effective component instances, edges,
actual hardware-process profiles, and the complete execution envelope. The
guard compares exact capability-class multiplicities rather than a set of
device labels, so one GPU, two equivalent GPUs, CPU plus GPU, and two unrelated
devices are distinct targets.

Selection enumerates every connected occurrence of each promoted semantic
scope, including duplicated source components. It rejects applications whose
hardware, required processes and features, placement, interconnects, phase,
activation width, context, or transient-state envelope falls outside the
published predicate. A branch-and-bound search then chooses the compatible
non-overlapping set with the greatest measured savings after conversion costs.
Uncovered instances retain the immutable exact implementation only when that
implementation is compatible; otherwise execution fails closed.

Each selected package entry names a `runtime_mount_plan.v3` explicitly. The
current Vulkan adapter replaces the physical component circuit and execution
specification while preserving semantic identity and all externally visible
ports. A multi-component island may retain a native representation across
internal boundaries. Shader and tensor-index references are canonicalized,
confined to the candidate bundle, integrity checked as package artifacts, and
mounted atomically. Tensor fragments may add candidate-owned parameters but
may never shadow an exact tensor. The fully mounted graph, execution specs,
generation contract, and merged tensor index are revalidated before model
residency begins.

A `vulkan_component_overlay.v2` may also attach an explicitly benchmarked
resident derivation to selector-addressed parameter resources. Source tensor
bytes remain immutable: derivation occurs only when the resource becomes
resident. Runtime mounting validates that the request covers every weight
consumed by the selected physical kernel, rejects shared-resource or
duplicate-instance representation conflicts, and recomputes every dependent
resource, atomic-group, partition-template, selector, and checkpoint identity
before residency planning. Lazy expert loading is therefore preserved while
native packed weights and measured resident alternatives remain runtime
choices rather than compiler-wide model conversions.

Normal package and placement inspection expose every verified option and the
selection report. The TUI shows the option, provider provenance, predicate,
benchmark, validation, and the implementation selected for the current draft
placement. Normal chat statistics report the chosen instances and measured
representation-boundary time, bytes, and count. They also divide observed
inter-token time into ordered context/state windows, making sustained decode
changes visible without a profiling-only execution path.

## Hardware-process profiles

`nerve-runtime --inspect-devices --json` emits one
`hardware_process_profile.v1` for the native CPU and one for every
compute-capable Vulkan GPU exposed by the selected ICD. The inventory describes
processes the optimizer may target rather than reducing hardware to a list of
matrix data types.

CPU discovery covers scalar, branch, out-of-order, SIMD, matrix-extension,
bit-manipulation, cache, prefetch, atomics, memory-copy, NUMA, and DMA exposure.
Vulkan discovery covers shader arithmetic, packed dot products, cooperative
matrices and their exact shapes, subgroups, registers, occupancy, workgroup
memory, caches, texture sampling and format conversion, graphics fixed
functions, ray traversal, collectives, indirect and device-generated work,
execution graphs, command replay, copy queues, external memory and
synchronization, and media queues. A facility is marked unavailable or opaque
when the selected API does not expose a programmable contract or a required
resource limit.

CPU and GPU memory-bandwidth processes are marked available but deliberately
carry no invented throughput number; realized bandwidth is supplied by the
empirical calibration stage.

Capability identity and physical identity are deliberately separate:

- `capability_extensions` participates in the capability-class digest and
  contains only compiler-relevant processes, formats, limits, and API facts;
- `identity_extensions` participates in the physical-profile digest and
  contains stable device-specific facts not shared by the capability class;
- `runtime_bindings` carries ephemeral API bindings such as a Vulkan physical
  index but does not participate in capability or profile identity; and
- calibration measurements change the physical-profile identity without
  changing the underlying capability class.

Profile provenance embeds a SHA-256 implementation fingerprint over the Rust
discovery schemas, CPU/Vulkan probes, API bindings, dependency lockfile, and
fingerprint algorithm. A discovery implementation change therefore cannot
silently reuse an older physical-profile identity merely because the crate
version was not manually changed.

Consequently, identical GPUs share one capability class even when their
runtime bindings differ. Compiler packages may target that class without
hardcoding model placement, while runtime placement still selects a concrete
device. The compiler derives its current SPIR-V lowering view from these
profiles and persists the complete inventory in `compiler_target.v2`; there is
no separate model-family or legacy device-capability path.

## Re-lowering

An alternative representation may change signals, state, parameters, or the
physical topology across one or more semantic scopes. Its re-lowering request
identifies:

- the candidate and replaced scopes;
- the representation contract digest;
- ordinary lowering passes that must run again; and
- boundary representations that those passes must preserve.

Re-lowering is a compiler request, not a runtime placement decision. The output
must retain provenance to the replaced semantic scopes, after which ordinary
physical optimization and package validation run again.
