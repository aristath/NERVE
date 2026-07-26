# TODO — Behavioral Representation Optimizer

## Goal

Build a target-aware post-compilation optimization system that treats the
compiled source model as a behavioral specification rather than a mandatory
execution plan.

The optimizer must traverse the model's semantic hierarchy—individual
operators, coupled operator regions, layer subcomponents, complete layers,
stateful subsystems, and useful cross-layer groups—and:

1. analyze the source algebra, parameters, state, and reachable behavior;
2. identify exact optimizations and plausible alternative representations;
3. synthesize executable candidates for the actual CPU or GPU processes exposed
   by the target hardware;
4. reject invalid candidates cheaply;
5. benchmark viable candidates against the exact implementation under matched
   conditions;
6. fully verify candidates that are materially faster; and
7. permanently promote only implementations that are both faster and
   behaviorally valid in their declared operating regime.

The resulting compiled model remains a self-contained package. Model-specific
discoveries belong to that package; generic analyzers, representation providers,
proof systems, benchmarking machinery, and selection logic belong to NERVE.

## Architectural invariants

- Begin with the semantic responsibility of a component, not with its source
  matrix operation or an arbitrarily selected hardware feature.
- Matrices remain valid candidate representations. The optimizer is trying to
  discover better expressions where structure justifies them, not prohibit
  matrix or dot-product instructions.
- Analyze coupled systems as well as isolated matrices. Q/K transforms,
  gate/up/down projections, routers and expert banks, state writers/readers, and
  adjacent representation boundaries can contain structure invisible in one
  tensor at a time.
- Prefer exact algebraic transformations and proofs. Behavioral approximation
  is allowed only with an explicit error contract and complete validation.
- Behavioral traces validate candidates; exhaustive input/output sampling is
  not the primary discovery method.
- Hardware support and hardware performance are different facts. Both must be
  measured.
- Compiler optimization must not hardcode runtime placement. A package may
  contain implementations for several hardware capability classes; runtime
  placement selects among compatible verified implementations.
- A semantic component remains editable, placeable, duplicable, bypassable, and
  inspectable even when its physical implementation is fused, split, generated,
  or represented in another form.
- Alternative representations may span several adjacent components so their
  signals can remain in a native representation without repeatedly converting
  back to the source tensor format.
- Compilation, candidate construction, setup, warmup, steady-state execution,
  validation, and teardown are separate measured phases.
- No candidate is promoted from one lucky timing or one convenient prompt.
- GPU experiments remain sequential and obey all device-residency requirements
  in `AGENTS.md`. NVIDIA is not used for NERVE workloads.

## Work plan

### 8. Extend the IR for alternative representations and representation islands

- Represent non-dense signals, parameters, state, and topology explicitly.
- Describe transducers between representations as ordinary semantic entities
  with measurable cost.
- Allow adjacent scopes to share one alternative representation without
  materializing the source activation format between them.
- Allow the optimizer to absorb a basis or encoding change into adjacent
  parameters where algebraically valid.
- Preserve logical shapes and public port contracts separately from physical
  representations.
- Represent confidence, unresolved candidates, and correction requests where a
  candidate uses verified approximation.
- Keep provenance from every new node, state object, parameter artifact, and
  physical kernel back to the replaced semantic scope.

Completion requires validation and planning tests for heterogeneous
representation graphs, explicit rejection of incompatible connections, and
round-trip inspection through the runtime/editor schema.

### 9. Implement candidate synthesis and isolated staging

- Construct candidate parameters, indexes, fields, graphs, programs, codebooks,
  geometry, state layouts, correction artifacts, and target-specific code in an
  isolated staging area.
- Never mutate the source package or exact baseline in place.
- Measure construction time, temporary memory, final permanent memory, and
  generated artifact size.
- Support cancellation and clean removal of incomplete candidates.
- Re-run ordinary lowering and physical optimization after a semantic
  representation change.
- Validate every staged artifact and its integrity before execution.

Completion requires atomic staging tests, cancellation tests, corruption tests,
and proof that failed candidates leave no published or partially referenced
artifacts.

### 10. Build the matched candidate benchmark engine

- Benchmark a candidate and its exact reference scope with identical inputs,
  state, randomness, control, placement, and device conditions.
- Exercise the regimes in which the candidate claims validity, including
  relevant combinations of:
  - decode and prefill;
  - narrow and wide activation batches;
  - context and state sizes;
  - single-stream and multi-stream execution;
  - cold mount and resident reuse; and
  - local and cross-device boundaries.
- Measure useful work separately from speculative, cancelled, discarded, or
  corrective work.
- Report latency, throughput, permanent bytes, transient bytes,
  representation-conversion traffic, construction cost, setup cost, device
  utilization, synchronization, and memory residency.
- Use warmup, repeated measurements, confidence intervals, and a declared
  material-improvement threshold.
- Verify fixed-seed reproducibility at the distribution, token, and runtime
  scheduling levels. When identical inputs and seeds diverge, preserve both
  traces and classify whether the cause is permitted sampling variance,
  numerical nondeterminism, speculative scheduling, or a correctness defect.
- Report throughput slope over the output window and surface queue-wait,
  timeout, synchronization, and transport counters that explain degradation
  during sustained generation.
- Reject benchmark-only shortcuts, arbitrary output caps, convenient seeds, and
  comparisons against mismatched source work.

Completion requires deterministic benchmark plans, auditable raw results,
noise-sensitive tests, and reuse of normal runtime execution and default
statistics rather than a special inference path.

### 11. Build the proof and behavioral-validation funnel

Apply validation in this order:

1. static contract and artifact validation;
2. exact algebraic proof where available;
3. cheap numerical and state-transition sanity checks before benchmarking;
4. matched performance measurement;
5. full local validation only for materially faster candidates; and
6. whole-model free-running validation before promotion.

Full validation must cover, as applicable:

- component output error and state-transition consistency;
- output-distribution divergence;
- top-k overlap and rank stability;
- route, memory, and candidate recall;
- confidence and correction calibration;
- teacher-forced sequences;
- free-running long-horizon behavior;
- multiple fixed seeds;
- reasoning-enabled real conversations;
- long context and long output;
- interruption, snapshot, fork, rollback, and resumption;
- graph edits and alternative placements; and
- adversarial counterexamples collected during earlier trials.

Exact generated text is neither required nor sufficient by itself. Validation
thresholds must belong to an explicit behavioral error contract.

Completion requires tests demonstrating correct acceptance of proven exact
rewrites and rejection of faster but behaviorally invalid approximations.

### 12. Implement target-guarded promotion and package publication

- Promote a candidate only when its benchmark and validation records both pass.
- Bind the promoted implementation to explicit predicates such as:
  - hardware capability class;
  - device/API requirements;
  - prefill or decode regime;
  - activation batch range;
  - context or state range; and
  - local or distributed execution requirements.
- Permit several verified implementations of one semantic component when
  different representations win in different regimes.
- Retain the exact semantic source contract and transformation provenance.
- Include only complete, integrity-checked implementation artifacts in the
  self-contained compiled model.
- Record why a candidate won, where it is valid, and which exact implementation
  it was compared against.
- Make publication atomic and rebuild package integrity after promotion.

Completion requires tests proving that failed, slower, noisy, or inaccurate
candidates cannot be published and that promoted packages remain relocatable
and independently loadable.

### 13. Implement runtime implementation selection

- Select among promoted implementations using actual runtime device bindings
  and execution conditions.
- Keep placement under runtime/UI control.
- Refuse execution when no implementation satisfies its declared hardware and
  behavioral contract; never silently run an incompatible variant.
- Avoid repeated representation conversion when a compatible multi-component
  representation island is available.
- Expose the selected implementation, target predicate, representation,
  provenance, benchmark evidence, and validation status through normal runtime
  inspection and the TUI.
- Report selection and representation-boundary costs in default execution
  statistics.

Completion requires runtime tests across CPU, one-GPU, multi-GPU, and mixed
logical placements without model-specific selection code.

### 14. Automate the analyze–synthesize–benchmark–verify loop

- Walk optimization scopes deterministically.
- Invoke compatible analyzers and representation providers.
- Deduplicate equivalent candidates.
- Apply declared construction, memory, execution-time, and experimental budgets
  without weakening correctness or benchmark quality.
- Run one GPU workload at a time and verify residency before and after it.
- Isolate failures and continue with other valid scopes and candidates.
- Emit structured progress, evidence, counterexamples, benchmark results,
  validation results, and promotion decisions.
- Produce a final report explaining:
  - which scopes were analyzed;
  - which structures were found;
  - which candidates were generated;
  - why candidates were rejected;
  - which candidates were faster;
  - which faster candidates failed validation; and
  - which candidates were promoted.

Completion requires a full unattended run that can safely finish without
leaving devices, staging artifacts, or package publication in an ambiguous
state.

### 15. Establish the first generic representation providers

- Use the analysis evidence from real components to choose the first providers;
  do not preselect them solely because they appear interesting.
- Begin with exact representations where the source structure permits an
  equivalence proof.
- Add approximate providers only with explicit correction and validation
  contracts.
- Ensure that at least one provider changes the computational representation,
  rather than merely changing matrix tiling, fusion, quantization, or scheduling.
- Keep every provider generic across compatible structures and model families.

Completion requires both:

- a candidate that is correctly rejected because it is slower or inaccurate;
  and
- a genuinely alternative representation that is promoted after a measured
  hardware win and complete behavioral validation.

### 16. Run end-to-end model and hardware qualification

- Run the optimizer over every supported compiled Safetensors model.
- Exercise every available compatible CPU and idle AMD GPU capability class.
- Re-run complete package validation and real resident multi-turn conversations
  after every promoted compiler transformation.
- Re-run the canonical warmed benchmark conditions and compare against the
  exact NERVE implementation and an equivalent mature-engine reference where
  available.
- Confirm graph editing, component duplication, bypass, rewiring, transient
  state, runtime placement, and multi-device execution still work.
- Confirm no analyzer or provider introduces model-family facts into the core
  runtime.
- Record whole-model improvements and any component-level win hidden by
  conversion or downstream costs.

Completion requires correct real conversations for the complete supported model
set, no regression in existing explicit performance floors, clean device
release after every run, and independently loadable optimized packages.

### 17. Update project documentation and remove stale design-status claims

- Update `EXPERIMENTS.md` to distinguish the implemented semantic-module
  foundation from remaining representation-optimizer research.
- Document the hardware-process profile, provider contract, candidate lifecycle,
  benchmark rules, validation funnel, promotion policy, and runtime selection in
  `README.md`.
- Cross-reference the final implementation against `CONCEPT.md`.
- Document how to add a representation provider without modifying model-specific
  runtime code.
- Remove completed items from this file as each item is reviewed and confirmed
  complete; do not retain historical progress as TODO entries.

Completion requires documentation that matches the live schemas, commands,
packages, runtime behavior, and tests.

## Overall completion criteria

This goal is complete only when:

- the optimizer can discover hardware capabilities and calibrated performance;
- it can enumerate semantic and coupled optimization scopes;
- it can analyze algebraic structure without model-name rules;
- alternative representations are supplied through an extensible provider
  interface;
- candidates are staged, compiled, benchmarked, and validated automatically;
- inaccurate or slower candidates are rejected with auditable evidence;
- at least one genuinely non-matrix representation is permanently promoted
  because it is faster and behaviorally valid;
- promoted implementations are target-guarded and selected from runtime
  conditions rather than compiler-owned placement;
- optimized compiled models remain self-contained and relocatable;
- all supported models continue to produce correct real conversations;
- existing graph, state, placement, multi-device, and performance guarantees
  remain intact; and
- this TODO file contains no remaining work items.
