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
   previously healthy AMD GPU. The runtime was violating the declared
   `demand-retained` policy by silently evicting inactive groups, clearing their
   stable-address publications, and reusing physical allocations. That bounded
   cache path is now removed: an observed working set that exceeds capacity
   fails admission atomically while every previously loaded component and
   address remains resident. A second generic fault was found in local
   component-batch execution: compiled indirect-dispatch contracts were being
   discarded, so routed-expert kernels launched their maximum direct grid. The
   batch step now retains a typed fixed, batch-width, or indirect dispatch
   contract, including indirect execution under a demand-residency conditional
   replay. The exact contract tests pass, but the corrected v20 package still
   faulted at the identical `0x8000c74c7000` SQC data-read address after its
   first output fragment. Therefore neither eviction nor direct over-dispatch
   was the complete cause. Trace that deterministic address through stable-slot
   publication and its exact consuming dispatch, then complete a multi-cycle
   conversation before enabling the decoder by default. `VK_EXT_device_fault`
   reporting and live addressable-buffer attribution must cover every queue
   submission path so this class of failure names the allocation rather than
   surfacing later through an unrelated copy or transfer operation. The guards,
   retained-capacity regression, and passing batched MXFP4 tests are necessary
   evidence, not proof that the real execution path is correct.

2. Complete runtime per-layer/per-component representation selection. Preserve
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
   complete sampled, thinking-enabled five-turn gate at 50.11 mean decode tok/s
   and 122.95 mean prefill tok/s after the demand-retained correction, including
   recall of the earlier Greece turn.
   Keep the live gate's repetition and conversation checks active so throughput
   alone cannot pass. Do not use faulted AMD devices merely to satisfy this
   gate; defer a model when no verified-healthy placement exists.

6. Perform a final adversarial review against `CONCEPT.md`: compiled artifacts
   remain self-contained and model-specific, while compiler discovery, runtime
   operators, graph wiring, placement, representation, and residency remain
   capability-driven and reusable by unseen models. Finish with an empty TODO,
   a clean worktree, and every milestone committed and pushed.
