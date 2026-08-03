# TODO

## Goal

Compile and run `/mnt/models/models/deepseek-v4/flash-0731/safetensors` as a
daily-driver NERVE model. Preserve source behavior, keep sparse routed experts
independently demand-resident, expose runtime per-component representation
selection, and optimize real multi-turn agentic inference to at least 25 tok/s,
with roughly 50 tok/s as the primary target and roughly 70 tok/s as a
speculative-decoding stretch target. Preserve supported Qwen models throughout.

## Work queue

1. Coalesce each checkpoint's missing selected-resource ranges by backing
   artifact and destination allocation, and preserve those coalesced waves
   through asynchronous read and upload. Overlap independent residency work
   with execution and split a command batch only at a real unresolved data or
   synchronization dependency. The two-set 128K-context gate still loaded 188
   cohorts (2.51 GB) and blocked for 3.99 seconds in the measured conversation;
   require the gate's second-set residency deltas to be negligible before
   calling the result steady state.

2. Finish the identity-independent `parallel_backbone_markov` speculative
   decoder. CPU-side SPIR-V reflection exposed that raw speculative graphs lost
   compiled descriptor bindings during device planning; device planning now
   materializes every executable component contract, while dedicated phases
   remain outside the generic graph based on their compiled execution role.
   This eliminated the deterministic SQC fault, and the full Qwen3.5-9B gate
   passes at 59.74 mean decode tok/s. Three-device target-only DeepSeek output
   is coherent, while enabling the parallel draft emits incoherent proposals.
   Trace proposal tokens, confidences, target verification, acceptance, and
   state commit as structural block contracts; complete a multi-cycle
   conversation before enabling the decoder by default.

3. Complete runtime per-layer/per-component representation selection. Preserve
   a native source representation when it is optimal on the selected device;
   otherwise evaluate only structurally valid candidates from INT4 2:4, dense
   INT4, FP8 2:4, dense FP8/INT8, and FP16. Promote a candidate only after
   hardware measurement and behavioral equivalence checks. Keep representation
   and placement as runtime choices, never model-family switches. Production
   compiler/runtime control flow currently contains no DeepSeek, Qwen, Llama,
   Gemma, LFM, or other model-family dispatch; source spelling is normalized at
   the compiler boundary and the execution ABI is structural. Vulkan capability
   discovery and calibration now include cooperative SINT8 matrix shapes. On the
   healthy RDNA4 device, the focused 16x16x16 calibration measured 16.3 us for
   SINT8 versus 22.6 us for F16 over the same 8,192-tile workload (about 1.39x
   faster). MXFP4 E2M1 weights now also have an identity-independent exact
   reconstruction proof into SINT8 codes: multiply each finite E2M1 value by
   two and halve its E8M0 group scale. A structurally generic inline
   reconstruction prototype preserved the known finite kernel outputs, but a
   real 4096x2048 six-expert decode microbenchmark measured 0.679 ms versus
   0.416 ms for native MXFP4 (1.63x slower). The rejected implementation was
   removed. Do not promote SINT8 merely because matrix calibration is faster;
   revisit it only with a representation-level design that amortizes activation
   quantization and packed-weight reconstruction while retaining MXFP4 as the
   compact backing store. Add other SINT8-backed implementations only for
   compatible structural scopes and retain the normal benchmark/equivalence
   promotion gate.

4. Extend the generic quality gate with a representative tool-call round trip
   and long-stream continuity case, using package-owned chat-template behavior
   rather than a model-name branch. Keep official thinking and sampling
   defaults, the 65,536-token output allowance, agentic context, coherent final
   answers, turn recall, and clean teardown. Apply the same complete-conversation
   warmup rule to these cases before accepting their performance measurements.

5. Raise DeepSeek decode from the current partially warm two-set mean of 7.93
   tok/s past the 25 tok/s floor and toward 50 tok/s, then continue until no
   material bottleneck remains. The successful five-device run selected zero
   optimized implementations, emitted no resident queue batches, and performed
   thousands of queue submit/wait pairs per turn. First make the common decode
   execution graph dependency-aware and submission-resident; then optimize the
   structurally selected MXFP4/FP8 parallel-linear paths, adaptive-tier exchange,
   and speculative acceptance overhead. Benchmark complete warmup plus repeated
   short real requests and report prefill and decode by default; do not optimize
   synthetic scores at the expense of behavior.

6. Before every runtime-performance commit, run Qwen3.6-35B-A3B and
   Qwen3.5-9B quality/performance gates sequentially on equivalent healthy AMD
   placement. The latest current-schema packages pass the full sampled,
   thinking-enabled five-turn gate with correct Greece recall: Qwen3.6-35B-A3B
   averages 105.48 decode tok/s and 202.96 prefill tok/s, while Qwen3.5-9B
   averages 60.43 decode tok/s and 176.32 prefill tok/s. Run both gates again
   before every runtime-performance commit.
   Keep the live gate's repetition and conversation checks active so throughput
   alone cannot pass. Do not use faulted AMD devices merely to satisfy this
   gate; defer a model when no verified-healthy placement exists.

7. Perform a final adversarial review against `CONCEPT.md`: compiled artifacts
   remain self-contained and model-specific, while compiler discovery, runtime
   operators, graph wiring, placement, representation, and residency remain
   capability-driven and reusable by unseen models. Finish with an empty TODO,
   a clean worktree, and every milestone committed and pushed.
