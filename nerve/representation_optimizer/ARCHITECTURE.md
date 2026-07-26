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
| `representation_candidate.v1` | Proposed representation, target predicate, and error contract |
| `candidate_construction.v1` | Isolated construction result, artifacts, resources, and diagnostics |
| `benchmark_record.v1` | Matched workload, raw samples, summary, and speed decision |
| `validation_record.v1` | Proof or behavioral-validation stages and counterexamples |
| `promotion_decision.v1` | Benchmark/validation evidence and guarded implementation decision |
| `relowering_request.v1` | Representation-aware request to repeat ordinary lowering passes |

Every contract round-trips through the same canonical serializer. The canonical
SHA-256 contract digest is for compiler identity and evidence linkage; package
artifact integrity independently protects the bytes published on disk.

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
