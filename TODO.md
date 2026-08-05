# TODO

## Goal

Compile and run `/mnt/models/models/deepseek-v4/flash-0731/safetensors` as a
daily-driver NERVE model. Preserve source behavior, keep sparse routed experts
independently demand-resident, expose runtime per-component representation
selection, and optimize real multi-turn agentic inference past 30 tok/s and
toward 50 tok/s without regressing supported Qwen models.

## Work queue

1. Complete capability-driven per-component representation selection. Preserve
   a native source representation when it is optimal on the selected device;
   otherwise measure only structurally valid INT4 2:4, dense INT4, FP8 2:4,
   dense FP8/INT8, and FP16 candidates. Promote a candidate only after hardware
   measurement and behavioral-equivalence checks. Keep representation and
   placement as runtime choices, never model-family switches. SINT8 matrix
   calibration alone is insufficient: the previous inline MXFP4-to-SINT8
   reconstruction lost to native MXFP4 because activation quantization and
   packed-weight reconstruction were not amortized.

2. Extend the generic quality gate with a representative tool-call round trip
   and a long-stream continuity case. Use package-owned chat behavior, official
   thinking and sampling defaults, the 65,536-token output allowance, agentic
   context, complete-conversation warmup, coherent final answers, turn recall,
   and exact teardown. Exercise both successful canonical commits and malformed
   output rollback without model-name branches.

3. Raise DeepSeek's accepted decode rate past 30 tok/s and continue until no
   material bottleneck remains. The authoritative thinking-enabled product gate
   uses 128K context, a 65,536-token output allowance, a requested speculative
   window of seven clamped to the checkpoint's trained five-token block,
   contiguous placement across all five AMD GPUs, and one complete discarded
   conversation before the truth conversation in the same resident process.
   It terminates coherently, preserves Greece recall, releases all acquired
   capacity, and reaches 7.900 decode tok/s and 8.832 prefill tok/s. This is a
   44.6% decode improvement over the former fixed-window 5.462 tok/s gate, but
   remains far below the 30 tok/s floor.

   Compile the complete speculative cycle as a device-resident transaction:
   generate the draft window without a host wait per proposal, append target
   projection and sampling, compare draft and target tokens on-device, produce
   the committed-prefix count, select and commit causal state snapshots
   indirectly, publish the retained source frame, and catch draft state up
   without per-token host mediation. The host should regain control only for a
   real residency miss or a completed emitted block. Preserve exact rollback,
   cancellation, and canonical commit semantics.

   The adaptive selector now measures actual execution shapes, excludes their
   first warmups and residency-loading cycles, and chooses by useful emitted
   tokens per speculative-cycle nanosecond. Session reset also clears every
   target and draft recurrent state, sampler, and auxiliary buffer. The
   remaining truth set still spent most speculative time in target verification
   (237.45 seconds of the long Corinth turn), while demand residency performed
   367 loads, 221 evictions, and 195 reloads across the measured conversation.
   The rejected parallel-latent-lane candidate proved that a shader-local
   speedup is not acceptable evidence: it regressed truth throughput and twice
   emitted a second reserved thinking terminator after a coherent answer.
   Restore reproducible execution before using token digests for attribution.
   With the same package, seed, prompts, placement, and sampler, fresh processes
   reproduce the first turn but can diverge later. Audit radix/top-k tie order,
   speculative sampler RNG consumption, accepted-prefix commit, rollback, and
   reset state. Require repeated-process equality for committed tokens, routed
   experts, accepted prefixes, and state digests; do not weaken the product
   quality gate to accommodate nondeterminism.

   Separate target compute from residency churn with device timestamps, then:

   - compile draft, target projection, comparison, state selection, commit, and
     draft catch-up into one device-resident transaction with no host wait until
     a residency miss or completed emitted block;
   - eliminate reload churn using route-aware expert residency decisions that
     remain capacity- and evidence-driven rather than model-specific;
   - optimize independently material MXFP4 sparse-expert kernels with fused or
     amortized representations, including measured native INT4/FP8 candidates;
   - use the causal component harness to replace the serialized indexed-
     attention batch transaction only when a staged candidate preserves exact
     output and committed-state digests at the declared context horizon;
   - close the remaining speculative acceptance gap without sacrificing model
     quality.

   Do not revive the rejected standalone intermediate-FP8 quantization
   dispatch, incomplete replay signatures, or naive workgroup-Y temporal-lane
   split. Every candidate must win exact microbenchmarks, output-and-state
   equivalence, repeated-process determinism, and the complete product gate.

4. Before every runtime-performance commit, run Qwen3.6-35B-A3B and Qwen3.5-9B
   quality/performance gates sequentially on equivalent healthy AMD placement.
   The latest schema-v10 thinking-enabled gates each discarded one complete
   in-process warmup conversation, reset model state without unloading, passed
   correct Greece recall, and released their exact recorded reservations:
   Qwen3.6-35B-A3B reaches 100.34 decode tok/s and 161.34 prefill tok/s;
   Qwen3.5-9B reaches 54.52 decode tok/s and 116.38 prefill tok/s. Keep
   repetition, structured-protocol, conversation, and teardown checks active so
   throughput alone cannot pass.

5. Perform a final adversarial review against `CONCEPT.md`: compiled artifacts
   remain self-contained and model-specific, while compiler discovery, runtime
   operators, graph wiring, placement, representation, residency, and stream
   transactions remain capability-driven and reusable by unseen models. Finish
   with an empty TODO, a clean worktree, and every milestone committed and
   pushed.
