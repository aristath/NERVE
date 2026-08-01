# TODO

## Goal

Compile and run `/mnt/models/models/deepseek-v4/flash-0731/safetensors` as a
daily-driver NERVE model. Preserve its source behavior, make its sparse routed
experts independently demand-resident, expose runtime per-layer representation
selection, and optimize real multi-turn agentic inference to an absolute floor
of 25 tok/s, a primary target of roughly 50 tok/s for its approximately 13B
active-parameter path, and a DSpark stretch target of roughly 70 tok/s without
regressing supported Qwen models. Reaching the floor does not end optimization.

## Work queue

1. Compile the supplied `encoding/encoding_dsv4.py` behavior into a portable,
   model-owned chat codec. Support multi-turn reasoning (`low`, `high`, `max`),
   tool calls/results, stop tokens, and assistant response parsing without
   embedding DeepSeek-specific rules in the generic runtime.

2. Discover and compile the attached DSpark speculative module, including its
   Markov/confidence state and seven-token draft contract. Validate acceptance,
   rejection, and transient-state rollback before enabling it by default.

3. Execute the compiled expert-granular demand-retained residency contract.
   Route before admission;
   load only selected expert resources, retain hot experts, use a bounded and
   observable eviction policy when necessary, and never require a whole MoE
   layer to become resident. Demonstrate that the model can run on the four
   currently workload-free AMD GPUs without using the display-resident GPU or
   any NVIDIA device.

4. Add runtime per-layer/per-component representation selection and optimized
   kernels for the hardware's real throughput hierarchy. Preserve a native
   source representation when it is the best verified path; otherwise explore
   INT4 2:4 structured (1531 TOPS), dense INT4 (766 TOPS), FP8 2:4 structured
   (766 TFLOPs), dense FP8/INT8 (383 TFLOPs/TOPS), then FP16 matrix paths.
   Structured candidates may be promoted only when their exact sparsity
   contract is satisfied and behavioral/quality validation passes; never prune
   silently for a benchmark. Benchmark the actual NERVE kernels, record the
   chosen representation per component, and keep placement a runtime concern.

5. Run real multi-turn DeepSeek conversations with the official thinking and
   sampling defaults, 64K-or-larger output allowance, and enough context for
   agentic coding. Validate coherent answers, reasoning, turn recall, tool-call
   syntax, long-stream continuity, and clean teardown before benchmarking.

6. Optimize measured inference until steady-state decode clears the absolute
   25 tok/s usability floor, reaches roughly 50 tok/s on the approximately 13B
   active path, and then continue until no material bottleneck remains. Treat
   roughly 70 tok/s as the DSpark stretch target rather than a promise based on
   theoretical throughput alone. Use warmup plus repeated short real requests,
   report prefill and decode by default, compare equivalent llama.cpp/vLLM
   implementations, and profile routing, expert admission, kernels,
   synchronization, transfers, and DSpark acceptance.

7. Before every runtime-performance commit, run the established Qwen3.6-35B
    A3B and Qwen3.5-9B quality/performance checks sequentially on equivalent AMD
    placement. Reject or revise any change that regresses correctness or causes
    a material performance loss.

8. Perform a final adversarial review against `CONCEPT.md`: compiled artifacts
    stay self-contained and model-specific, runtime graph/placement/residency
    remain editable, no model-family facts leak into the core engine, TODO is
    empty, the worktree is clean, and every milestone is committed and pushed.
