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
   material bottleneck remains. The current complete thinking-enabled gate uses
   128K context, a 65,536-token output allowance, DSpark-7, contiguous placement
   across all five AMD GPUs, and one full conversation-set warmup. It passes
   behavior and teardown at 3.987 decode tok/s and 8.288 prefill tok/s. A prior
   equivalent run reached 4.242 decode tok/s; DSpark acceptance changed from
   25.64% to 20.87%, confirming that accepted-token throughput and target
   verification remain the dominant variables rather than host drafting.

   Compile the complete speculative cycle as a device-resident transaction:
   generate the draft window without a host wait per proposal, append target
   projection and sampling, compare draft and target tokens on-device, produce
   the committed-prefix count, select and commit causal state snapshots
   indirectly, publish the retained source frame, and catch draft state up
   without per-token host mediation. The host should regain control only for a
   real residency miss or a completed emitted block. Preserve exact rollback,
   cancellation, and canonical commit semantics.

   Close the semantic acceptance gap against the equivalent llama.cpp DSpark
   reference, then reduce target verification cost. Optimize the independently
   material six-lane MXFP4 sparse expert kernels with fused or amortized
   representations; do not revive the rejected standalone intermediate-FP8
   quantization dispatch, incomplete replay signatures, or temporal-width
   experiment. Every candidate must win exact microbenchmarks, behavioral
   equivalence, and the complete product gate.

4. Before every runtime-performance commit, run Qwen3.6-35B-A3B and Qwen3.5-9B
   quality/performance gates sequentially on equivalent healthy AMD placement.
   The latest thinking-enabled gates passed with correct Greece recall and exact
   reservation release: Qwen3.6-35B-A3B reached 101.98 decode tok/s and 152.21
   prefill tok/s; Qwen3.5-9B reached 50.47 decode tok/s and 167.29 prefill tok/s.
   Keep repetition, structured-protocol, conversation, and teardown checks
   active so throughput alone cannot pass.

5. Perform a final adversarial review against `CONCEPT.md`: compiled artifacts
   remain self-contained and model-specific, while compiler discovery, runtime
   operators, graph wiring, placement, representation, residency, and stream
   transactions remain capability-driven and reusable by unseen models. Finish
   with an empty TODO, a clean worktree, and every milestone committed and
   pushed.
