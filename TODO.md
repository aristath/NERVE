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
- Microbenchmarks answer one bounded binary question—whether the candidate is
  faster—using one discarded warmup and one matched measured call per role.
  Behavioral validation, not repeated timing samples, decides correctness and
  operating-regime coverage.
- GPU experiments remain sequential and obey all device-residency requirements
  in `AGENTS.md`. NVIDIA is not used for NERVE workloads.

## Work plan

### 16. Run end-to-end model and hardware qualification

- Run the optimizer over the compiled Qwen Safetensors model used by the
  established performance baseline. Laguna is explicitly out of scope.
- Exercise CPU capability discovery and the idle AMD GPU capability class used
  by the Qwen package; do not force CPU model execution when the compiled
  implementation is Vulkan-only.
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
- Use default sustained-decode windows and execution counters to distinguish
  exact-runtime effects—such as changing speculative acceptance, bounded
  completion waits, and cross-device transfers—from representation-candidate
  costs, and reject candidates that amplify those slow paths.

Completion requires a correct real Qwen conversation, no regression in the
established Qwen performance baseline, clean device release after every run,
and independently loadable optimized Qwen packages.

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
- Qwen continues to produce correct real conversations;
- existing graph, state, placement, multi-device, and performance guarantees
  remain intact; and
- this TODO file contains no remaining work items.
