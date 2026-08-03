# TODO

## Goal

Compile and run `/mnt/models/models/deepseek-v4/flash-0731/safetensors` as a
daily-driver NERVE model. Preserve source behavior, keep sparse routed experts
independently demand-resident, expose runtime per-component representation
selection, and optimize real multi-turn agentic inference to at least 25 tok/s,
with roughly 50 tok/s as the primary target and roughly 70 tok/s as a
speculative-decoding stretch target. Preserve supported Qwen models throughout.

## Work queue

1. Add an explicit bounded on-demand residency policy for packages whose full
   addressable resource set exceeds aggregate VRAM. Preserve `demand-retained`
   exactly: a first-use load stays resident and a full store fails admission
   atomically. Add a separately named runtime policy that evicts only inactive
   allocation cohorts at synchronous checkpoint boundaries, clears their
   stable-address publications before physical reuse, and cannot overlap an
   execution lease. The 157 GB DeepSeek package exhausted retained stores on
   one, two, and three 32 GB execution GPUs during real target-only generation;
   moving component boundaries merely delayed the same cumulative-working-set
   limit. The separate `demand-paged` policy now passes exact reload,
   cohort-eviction, selector-fairness, address-publication, and AMD Vulkan store
   tests. The real 157 GB package mounts target-only across three AMD devices in
   17.89 seconds and releases its acquired capacity back to the recorded
   pre-run reservations. Its first cold
   prefill then exposed a compiler storage-layout defect: 72,317 independently
   addressable tensors were emitted as 72,317 one-tensor files, forcing a top-6
   expert wave through 36 files per layer. The compiler now derives artifact
   affinity from selector topology and plans 47 contiguous banks for this
   package (one execution spine plus 46 selected-resource domains), with no
   model-name dispatch and no intermediate per-tensor writes for compatible
   native sources. The fresh 47-bank package mounts and produces a coherent
   target-only greeting, but its cold turn loads 4,719 of 11,008 expert cohorts
   (63.09 GB), spends 66.14 seconds blocked on residency, and reaches only 1.364
   decode tok/s. It also performs 3,023 sequence submit/wait pairs and 5,069
   copy submissions with no resident queue batches. Coalesce selected ranges by
   artifact and destination, overlap read/upload with independent execution,
   and split command batches only at a real unresolved dependency. Re-measure
   with a complete discarded conversation followed by an identical measured
   conversation in one resident process before removing this item.

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

4. Complete a real DeepSeek multi-turn quality gate with official thinking and
   sampling defaults, a 65,536-or-larger output allowance, and agentic context.
   Require coherent reasoning and answers, turn recall, tool-call syntax,
   long-stream continuity, and clean teardown. Because routed resources are
   demand-resident, run two identical canonical conversation sets without a
   reset: validate but discard the complete first set and use only the second
   set as performance truth. Report cumulative residency deltas for both sets;
   if the measured set still has misses, blocking loads, or reloads, identify it
   as partially warm rather than presenting it as steady state. The generic
   conversation gate now supports this through
   `--warmup-conversation-sets 1` without any model-family branch.

5. Raise DeepSeek steady-state decode from the current retained-session range
   of roughly 6.4--7.2 tok/s past the 25 tok/s floor and toward 50 tok/s, then
   continue until no material bottleneck remains. The next measured target is
   the common single-frame FP8 parallel-linear path, followed by synchronization,
   adaptive-tier exchange, and speculative acceptance overhead. Benchmark
   warmup plus repeated short real requests and report prefill and decode by
   default; do not optimize synthetic scores at the expense of behavior.

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
