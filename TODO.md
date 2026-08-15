# TODO

## Goal

Make `/mnt/models/models/qwen3.8/27b/FP8` a correct, practical, and maximally
fast NERVE model on this workstation.

NERVE must:

- execute real tensor-parallel work end to end, not merely compile or mount TP
  contracts;
- discover legal tensor partitions from the compiled graph and tensor roles,
  without model-name checks;
- measure the available hardware and automatically choose the fastest valid
  combination of single-device, serialized layer placement, tensor-parallel
  islands, and hybrid placement;
- preserve the standalone-component and editable-wiring architecture in
  `CONCEPT.md`; and
- remain a general inference engine. Qwen3.8 is the optimization target and
  acceptance workload, not a reason to hardcode Qwen behavior into the runtime.

## Performance target

The equivalent llama.cpp baseline uses:

- `Qwen3.8-27B-UD-Q8_K_XL.gguf`, the closest available GGUF to the FP8 source;
- two R9700 GPUs at PCI `07:00.0` and `21:00.0`;
- serialized layer placement, not llama.cpp tensor parallelism;
- 128K context, a 65,536-token output allowance, Q8 KV, flash attention, and
  model-owned reasoning/chat behavior; and
- one persistent server, one warmup turn, then five measured conversation
  turns with retained history.

The measured llama.cpp decode baseline is **15.4318 tok/s**, averaged across
the five post-warmup turns. NERVE's required bands are therefore:

- **23.1477 tok/s**: minimum 1.5x result;
- **30 tok/s**: concrete completion target; and
- **30.8636 tok/s or higher**: 2x result and preferred outcome.

Generated text must remain coherent, reasoning must remain enabled, and the
final recall question must correctly identify Greece. Throughput from broken,
truncated, non-thinking, or otherwise degraded output does not count.

## Product gate

Use one persistent model instance and this conversation:

1. Warmup, excluded from averages: `hi`
2. `Who are you?`
3. `What is the capital of Greece?`
4. `How many cities named "Corinth" are there?`
5. `What is your knowledge cutoff date?`
6. `I asked you earlier to tell me the capital of a country. Which country was that?`

Run with 128K context and `--max-new-tokens 65536`. Report prefill and decode
performance by default. Do not unload or reset the model between turns.

## Work queue

### 1. Compile and validate the FP8 model

- Compile the Safetensors source into one self-contained NERVE package while
  preserving native FP8 tensors wherever the selected GPUs support them.
- Confirm architecture, tokenizer, chat template, reasoning controls, tensor
  shapes, layer count, and output head from source artifacts rather than a
  model-name adapter.
- Run a scalar or serialized conversation first and compare behavior with the
  llama.cpp baseline before enabling TP.

### 2. Complete real tensor-parallel inference

- Select the smallest useful dense transformer island and partition its
  compiler-declared projections across two R9700s.
- Execute a complete conversation turn through that TP island. Require runtime
  submission counters to prove that every selected participant performed real
  shard work and that the collective/reduction completed.
- Compare tokens, logits within the compiled numerical tolerance, and
  persistent state against canonical execution.
- Extend the proven path to every profitable projection and layer. A mounted
  island, a resolved contract, or a component-only microbenchmark is not
  completion.
- Keep parameters resident in their owning shard. Do not retain redundant full
  tensors on the coordinator, reconstruct them per token, or round-trip shard
  intermediates through the CPU.

### 3. Make tensor placement automatic and measurement-driven

- Enumerate legal per-operation strategies from typed compiler contracts:
  local execution, serialized component placement, output-row/input-column TP,
  and compatible fused islands.
- Measure complete candidate transactions on the actual selected devices,
  including compute, synchronization, transfers, collectives, boundary cost,
  current reservations, and memory capacity.
- Optimize decode and prefill separately while selecting one compatible model
  placement. Account for KV/state ownership and the cost of transitions between
  adjacent strategies.
- Choose the minimum measured end-to-end plan. Do not hardcode device counts,
  equal splits, model names, or the assumption that TP always wins.
- Report the selected tensor ownership and physical execution plan clearly in
  normal runtime output.

### 4. Optimize the complete Qwen3.8 workload

- Profile ordinary chats, not a special benchmark execution path.
- Remove host orchestration from the per-token critical path: retain command
  buffers, descriptors, pipelines, tensor shards, synchronization objects, and
  KV/state allocations across tokens and turns.
- Fuse graph regions when doing so preserves the component boundary and proves
  a complete-stream win. Prioritize FP8 matrix paths, attention/KV work, dense
  FFN projections, reductions, and cross-device collectives according to
  measured critical-path time.
- Compare single-device, serialized two-device, TP two-device, and measured
  hybrid candidates with equivalent model work.
- After every meaningful optimization, rerun correctness first and reject any
  candidate that changes behavior or slows the product gate.

### 5. Final acceptance

- Complete the product gate with real TP submissions on every measured turn.
- Achieve at least 23.1477 tok/s, continue to the 30 tok/s concrete target, and
  pursue 30.8636 tok/s or better while material improvements remain.
- Run sequential unit and integration tests for the compiler, physical-plan
  resolver, TP execution, automatic placement, cancellation, and teardown.
- Confirm every selected GPU returns to its exact pre-run reservation and that
  unrelated workloads remain intact.
- Remove obsolete compiled packages, commit each completed milestone
  atomically, and push `main`.

## Non-negotiable guardrails

- Never run tests in parallel.
- Never substitute planner, mount, or microbenchmark evidence for a completed
  real-model conversation.
- Never improve reported speed by lowering context, output allowance, reasoning
  quality, quantization quality, or conversation correctness.
- Never add Qwen3.8-specific runtime branches when the behavior can be derived
  from graph, tensor, representation, or execution contracts.
- Never use tensor parallelism merely because it exists; use it where complete
  measured execution wins.
