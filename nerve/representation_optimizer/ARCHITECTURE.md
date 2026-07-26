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
| `source_behavior_contract.v1` | Observable source behavior and exact implementation |
| `algebraic_evidence.v1` | Analyzer identity, structural claims, and evidence artifacts |
| `hardware_process_profile.v1` | Target identity, processes, measurements, and provenance |
| `representation_descriptor.v1` | Open representation-family vocabulary, evidence requirements, physical forms, costs, and correction paths |
| `representation_candidate.v1` | Proposed representation, target predicate, and error contract |
| `candidate_construction.v1` | Isolated construction result, artifacts, resources, and diagnostics |
| `benchmark_record.v1` | Matched workload, raw samples, summary, and speed decision |
| `validation_record.v1` | Proof or behavioral-validation stages and counterexamples |
| `promotion_decision.v1` | Benchmark/validation evidence and guarded implementation decision |
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

Descriptor JSON is included in the package compiler fingerprint. Changing the
available representation vocabulary therefore cannot silently reuse a compiled
package fingerprint from different optimizer semantics.

## Candidate lifecycle

A candidate moves only through the following evidence-carrying sequence:

```text
synthesized
    -> staged
    -> statically_validated
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

## Failure isolation

An optimization session contains the immutable exact-baseline digest and
independent candidate lifecycles. Failing or rejecting one candidate changes
only that candidate's new session value. It cannot:

- mutate the exact semantic graph;
- mutate another candidate;
- publish partial artifacts;
- mark another candidate as evaluated; or
- turn a failed experiment into a runtime implementation.

Filesystem staging and atomic publication build on this contract in the
candidate-construction and promotion milestones.

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
