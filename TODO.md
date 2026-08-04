# TODO

## Goal

Compile and run `/mnt/models/models/deepseek-v4/flash-0731/safetensors` as a
daily-driver NERVE model. Preserve source behavior, keep sparse routed experts
independently demand-resident, expose runtime per-component representation
selection, and optimize real multi-turn agentic inference to at least 30 tok/s,
with roughly 50 tok/s as the primary target and roughly 70 tok/s as a
speculative-decoding stretch target. Preserve supported Qwen models throughout.

## Work queue

1. Complete runtime per-layer/per-component representation selection. Preserve
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

2. Extend the generic quality gate with a representative tool-call round trip
   and long-stream continuity case, using package-owned chat-template behavior
   rather than a model-name branch. Keep official thinking and sampling
   defaults, the 65,536-token output allowance, agentic context, coherent final
   answers, turn recall, and clean teardown. Apply the same complete-conversation
   warmup rule to these cases before accepting their performance measurements.

3. Raise DeepSeek's user-visible accepted decode rate past 30 tok/s and toward
   50 tok/s, then continue until no material bottleneck remains. The accepted
   target-only NERVE gate is 7.93 raw tok/s; equivalent non-speculative llama.cpp
   gates measured 8.50 tok/s for Q4 and 9.50 tok/s for Q2, so the workstation
   evidence does not support a 25 tok/s raw target-pass floor. Preserve and
   improve raw target performance, but reach the product goal through the
   package-owned DSpark path rather than redefining success. Stateless and
   stateful target operators now have exact dependency-preserving width-N
   implementations, including rolling state, latent compression and indexing,
   and indexed sparse attention. The complete package has causal implementations
   for 1,558 of 1,560 target decode kernels; only the stateless stream repeater
   and Sinkhorn head remain serial.

   Produced-port fan-out now aliases every local and remote consumer to one
   physical source, and `parallel_backbone_markov` prompt/verification catch-up
   consumes retained target frames through one compiled causal state-ingestion
   graph instead of replaying scalar adapter/state work per token. Dynamic tier
   admission now reserves the entire incoming load wave before
   eviction, uses exact physical allocation requirements, and treats stable slot
   layouts as versionable physical storage. Adaptive retiering atomically moves
   payload, address publication, residency ownership, allocation cohort, and tier
   assignment. The complete thinking-enabled two-set DeepSeek gate remained
   coherent, preserved turn recall, and released all five GPU reservations. The
   truth set averaged 4.111 decode tok/s and 7.301 prefill tok/s, incurred 283
   additional misses across 1.42 million expert selections, and ended with
   128.06 GB of device payload plus 18.14 GB of host-visible payload. Retiering
   performed 2,894 promotions and copied about 77.9 GB during the truth set,
   consuming about 30.6 seconds; target verification and overly broad tier churn
   are therefore the next measured costs to eliminate.

   Compile and submit the complete speculative cycle as a device-resident
   transaction: generate the full demand-resident draft window without a host
   wait per proposal, append target projection and sampling to the target lane,
   compare draft and target tokens on-device, produce the committed-prefix count,
   select and commit causal state snapshots indirectly, publish the retained
   source frame, and catch the draft state up without per-token host mediation.
   The host should regain control only for a real residency miss or a completed
   block of emitted tokens. Preserve exact rollback/commit semantics. Then add
   structurally selected MXFP4/FP8 parallel-linear implementations, improve the
   retained-frame DSpark schedule and acceptance, and use adaptive tier exchange
   so hot experts remain in VRAM while cold experts use the smallest viable lower
   tier. Benchmark complete warmup plus repeated short real requests and report
   prefill, raw target passes, accepted tokens, acceptance, residency misses, and
   host-visible spill by default; do not optimize synthetic scores at the expense
   of behavior.

4. Before every runtime-performance commit, run Qwen3.6-35B-A3B and
   Qwen3.5-9B quality/performance gates sequentially on equivalent healthy AMD
   placement. The latest current-schema packages pass the full sampled,
   thinking-enabled five-turn gate with correct Greece recall. After the
   retained-frame ingestion milestone, Qwen3.6-35B-A3B averaged 106.76 decode
   tok/s and 210.53 prefill tok/s versus 109.29 and 211.66 before, while
   Qwen3.5-9B averaged 59.97 decode tok/s and 183.45 prefill tok/s versus 59.58
   and 180.20 before. Both produced byte-identical responses to their baselines,
   passed every turn, and released their NERVE GPU reservations exactly. The
   post-retiering regression gates also passed: Qwen3.6-35B-A3B averaged 101.81
   decode tok/s and 158.59 prefill tok/s, while Qwen3.5-9B averaged 50.91 decode
   tok/s and 174.03 prefill tok/s. Run both gates again before every
   runtime-performance commit.
   Keep the live gate's repetition and conversation checks active so throughput
   alone cannot pass. Do not use faulted AMD devices merely to satisfy this
   gate; defer a model when no verified-healthy placement exists.

5. Perform a final adversarial review against `CONCEPT.md`: compiled artifacts
   remain self-contained and model-specific, while compiler discovery, runtime
   operators, graph wiring, placement, representation, and residency remain
   capability-driven and reusable by unseen models. Finish with an empty TODO,
   a clean worktree, and every milestone committed and pushed.
