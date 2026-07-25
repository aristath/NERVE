# TODO

## Direction and constraints

NERVE is a continuous-stream execution engine:

```text
running stream =
    runtime graph
  + compiled permanent component circuits
  + mutable transient circuit state
```

The compiled package is a neutral component catalog plus canonical topology.
Runtime configuration owns graph edits, component placement, stream scheduling,
and device selection.

The logical source-layer component remains the graph-editing and placement
boundary. Backends may fuse, split, tile, or otherwise lower work inside that
boundary, but optimization must not erase the user's ability to place, duplicate,
bypass, replace, migrate, or reconnect the component.

Use llama.cpp, vLLM, and other mature engines as engineering references, not as
architectural templates. Do not add model-name-specific runtime behavior,
compiler-time placement, obsolete-format compatibility layers, arbitrary small
context/output limits, or benchmark-only shortcuts.

## Remaining work, in priority order

### 1. Complete route-native MoE execution

Sparse components and selected-route kernels exist, but routing is not yet a
fully optimized runtime signal path.

- Group and batch selected routes across tokens and streams.
- Execute only selected experts and prove this with runtime work counters.
- Place or shard experts across devices without dense all-expert work.
- Keep route weights and reduction on the device.
- Make route signals participate in resident execution templates and feedback
  control.
- Validate output correctness and active-expert scaling on real MoE packages.
- Make the 35B MoE model's performance reflect its active parameter count rather
  than its full declared size.

### 2. Integrate MTP into the steady-state scheduler and device loop

MTP compilation and transactional verification work, but speculative execution
is not yet part of the optimized steady-state path.

- Add scheduler-native lookahead slots and multi-draft routing.
- Keep draft proposal, target verification, acceptance, rollback, and catch-up on
  the device where practical.
- Ensure enabling MTP does not disable resident feedback execution or introduce a
  host synchronization point per token.
- Keep thinking/reasoning behavior enabled normally during validation.
- Report proposal count, acceptance, rollback, useful tokens, and timing in
  normal chat output.
- Enable MTP by default only where warmed, realistic workloads show a net
  improvement.

### 3. Finish long-context prefill and mixed-workload scheduling

- Interleave prefill and decode fairly under memory pressure.
- Pipeline independent streams across placed devices so serial layer placement
  amortizes cross-queue handoffs. Under matched seed-1 16K conversations, the
  first three valid turns averaged 16.395 decode tokens/second on one GPU and
  12.896 on two GPUs. Direct shared-host edges reduced command count relative to
  two-copy device-local staging but improved throughput by only 0.4%, proving
  that the remaining 21.3% gap is dominated by the serial placed schedule rather
  than edge-copy mechanics.
- Derive prefill chunk size from available memory, device execution limits, and
  selected kernel shape.
- Batch compatible prefill work across streams.
- Parallelize the stable online attention softmax and value reduction as context
  grows. The current 256-wide attention-head kernel still executes tile score,
  exponential, and carry updates through a serial lane-zero region; preserve
  numerically stable online semantics while distributing that work.
- Measure the device-page translation cost introduced by physical transient-state
  paging. Hoist or specialize invariant page metadata and mapping reads without
  returning to host-resolved flat offsets. On the first post-paging 27B-FP8
  conversation, the four completed measured turns averaged 12.495 decode
  tokens/second, below the pre-paging 15.862-token/second observation, although
  the different generated lengths and accumulated context make this a lead to
  isolate rather than a causal attribution.
- Make resident prefix checkpoint capture asynchronous or incrementally
  copy-on-write instead of synchronously copying the complete retained state. In
  the first post-prefix-admission 27B-FP8 run, one checkpoint retained
  167,510,016 device bytes and introduced two blocking resident copy waits.
- Preallocate, reclaim, and compact physical state pages safely around long
  prompts.
- Validate 64K/128K context and long agentic outputs without arbitrary low token
  limits.
- Report prefill and decode throughput separately by default.

### 4. Maintain adversarial correctness and performance gates

Every meaningful compiler, runtime, state, graph, or kernel change must be tested
against the supported model set rather than optimized around one model.

Correctness coverage must include:

- Teacher-forced source-versus-compiled comparisons where a source runner is
  available.
- Real free-running, multi-turn conversations.
- Thinking/reasoning enabled for thinking models.
- Graph duplication, bypass, rewiring, and placement changes.
- State reset, snapshot, fork, shared prefix, copy-on-write, and reclamation.
- EOS/cancellation tests proving unused feedback-window tail work did not execute.
- Long-context and long-output operation.

Performance runs must:

- Keep the model resident for the full run.
- Use `hi` only as the discarded warmup request.
- Average the following five measured conversation turns:
  1. `Who are you?`
  2. `what is the capital of Greece?`
  3. `How many cities named "Corinth" are there?`
  4. `What is your knowledge cutoff date?`
  5. `I asked you earlier to tell me the capital of a country. Which country was that?`
- Use a 65,536-token output allowance and a realistic context allocation unless
  the test explicitly measures another context size.
- Report setup separately; report prefill and decode throughput, useful versus
  executed ticks, placement, device identities, kernel variants, and MTP state.
- Compare equivalent warmed settings with llama.cpp or vLLM where applicable.
- Exercise one device when the model fits and only the devices actually required
  when it does not.
- Treat 20 decode tokens/second on Qwen3.6-27B-FP8 as the current minimum target,
  not the final optimization ceiling.
- Exercise multiple fixed seeds for stochastic samplers. A seed-1 27B run
  completed the full conversation correctly, including cross-turn recall, while
  seed 0 entered a multi-thousand-token repetition on the Corinth question;
  neither arbitrary output caps nor a single convenient seed is a valid
  correctness gate.
- Make fixed-seed sampling and execution reproducible. A fresh seed-1 package
  whose tensors, shaders, and executable artifacts were byte-identical to the
  earlier successful package entered an unbounded emoji loop on the second
  measured turn. In the post-kernel-family review, seed 0 answered the Corinth
  turn with the preceding Athens answer and then repeated indefinitely on the
  next turn; seed 1 produced a correct Athens turn but repeated city names for
  hundreds of tokens while answering Corinth. Fail the gate on repeated final
  segments, malformed thinking boundaries, turn contamination, or failure to
  terminate after a valid answer; generating some meaningful text is
  insufficient.
- The post-multi-stream seed-1 27B-FP8 run again entered a repeated Corinth
  reasoning loop on measured turn three. The two valid turns decoded at 16.041
  and 15.682 tokens/second (15.862 average), confirming no measurable
  single-stream regression but leaving the full conversation gate failed.
- The first post-physical-paging seed-1 run completed the Corinth turn but entered
  an unbounded repeated-history loop on the fifth recall turn instead of
  answering Greece. The four completed measured turns decoded at 13.735, 13.158,
  11.908, and 11.178 tokens/second (12.495 average). Treat that run as a failed
  correctness gate, not a five-turn performance result.
- The first post-prefix-admission seed-1 run entered the repeated Corinth city
  list on measured turn three. The two completed measured turns decoded at
  13.667 and 16.667 tokens/second (15.167 average); the full benchmark is invalid,
  and the implementation remains below the 20-token/second floor.
- The first post-canonical-template seed-1 run answered the first three measured
  turns correctly, including the accumulated conversation history, but the
  knowledge-cutoff turn entered an exact repeated `Output matches / Done /
  Proceed` final segment and had to be stopped. The three completed turns
  decoded at 13.319, 12.952, and 11.973 tokens/second (12.748 average). Treat it
  as another failed correctness gate and not as a complete performance result.
- The matched 16K single-device and two-device runs produced identical valid
  outputs through the first three measured turns, then both entered the same
  exact `Output matches / Done / Proceed` loop on the knowledge-cutoff turn.
  Placement is therefore not the source of that correctness failure.
- An explicitly selected Vulkan test device must make a test run or fail; it must
  never silently turn a device-open error into a passing skip. The prefix,
  cancellation, and physical-page tests now enforce this, and the remaining
  Vulkan tests need the same contract.
- Finish migrating structural Rust tests to the checked-in deterministic tiny
  compiled package. Prompt-engine batching, wait-set, fairness, and cancellation
  tests now use it; remaining tests must stop depending on the deleted external
  230M lowered-model fixture.

Repository safety requirements remain mandatory: run tests sequentially, select
Vulkan tests individually, never run a NERVE workload on the NVIDIA GPU, and
verify AMD device residency before and after every GPU workload.
