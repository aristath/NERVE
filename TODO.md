# TODO

## Goal

Compile and run `/mnt/models/models/deepseek-v4/flash-0731/safetensors` as a
daily-driver NERVE model. Preserve source behavior, keep sparse routed experts
independently demand-resident, expose runtime per-component representation
selection, and optimize real multi-turn agentic inference to at least 25 tok/s,
with roughly 50 tok/s as the primary target and roughly 70 tok/s as a
speculative-decoding stretch target. Preserve supported Qwen models throughout.

## Work queue

1. Finish the identity-independent `parallel_backbone_markov` speculative
   decoder. The compiler and runtime already discover it from graph, tensor,
   and state contracts without family-name dispatch. The rebuilt v20 package
   still reached an SQC data-read GPUVM fault during its first real turn on a
   previously healthy AMD GPU, despite the current MXFP4 route/address guards.
   `VK_EXT_device_fault` reporting and live addressable-buffer attribution are
   now capability-driven runtime facilities. Reproduce safely enough to capture
   an attributed fault, identify and fix the invalid lifetime, range, or address
   transition, and complete a multi-cycle conversation before enabling the
   decoder by default. The guards and passing direct-buffer microtests are
   defense in depth, not proof that the real execution path is correct.

2. Complete runtime per-layer/per-component representation selection. Preserve
   a native source representation when it is optimal on the selected device;
   otherwise evaluate only structurally valid candidates from INT4 2:4, dense
   INT4, FP8 2:4, dense FP8/INT8, and FP16. Promote a candidate only after
   hardware measurement and behavioral equivalence checks. Keep representation
   and placement as runtime choices, never model-family switches.

3. Complete a real DeepSeek multi-turn quality gate with official thinking and
   sampling defaults, a 65,536-or-larger output allowance, and agentic context.
   Require coherent reasoning and answers, turn recall, tool-call syntax,
   long-stream continuity, and clean teardown.

4. Raise DeepSeek steady-state decode from the current retained-session range
   of roughly 6.4--7.2 tok/s past the 25 tok/s floor and toward 50 tok/s, then
   continue until no material bottleneck remains. The next measured target is
   the common single-frame FP8 parallel-linear path, followed by synchronization,
   adaptive-tier exchange, and speculative acceptance overhead. Benchmark
   warmup plus repeated short real requests and report prefill and decode by
   default; do not optimize synthetic scores at the expense of behavior.

5. Before every runtime-performance commit, run Qwen3.6-35B-A3B and
   Qwen3.5-9B quality/performance gates sequentially on equivalent healthy AMD
   placement. Qwen3.6-35B-A3B currently passes the full five-turn gate at 74.51
   mean decode tok/s and 209.70 mean prefill tok/s. Qwen3.5-9B now passes the
   complete sampled, thinking-enabled five-turn gate at 50.08 mean decode tok/s
   and 123.79 mean prefill tok/s, including recall of the earlier Greece turn.
   Keep the live gate's repetition and conversation checks active so throughput
   alone cannot pass. Do not use faulted AMD devices merely to satisfy this
   gate; defer a model when no verified-healthy placement exists.

6. Perform a final adversarial review against `CONCEPT.md`: compiled artifacts
   remain self-contained and model-specific, while compiler discovery, runtime
   operators, graph wiring, placement, representation, and residency remain
   capability-driven and reusable by unseen models. Finish with an empty TODO,
   a clean worktree, and every milestone committed and pushed.
