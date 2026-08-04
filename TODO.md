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
   assignment. The original complete thinking-enabled two-set DeepSeek gate
   remained coherent, preserved turn recall, and released all five GPU
   reservations while averaging 4.111 decode tok/s and 7.301 prefill tok/s.
   Adaptive tiering now uses cumulative LFU working-set evidence and requires
   the observed access advantage to repay both full-payload copies before an
   equal-layout exchange. Runtime counters describe the tier where the completed
   interval actually executed rather than retroactively attributing it to the
   next interval's placement. The complete replacement gate passed unchanged
   behavioral checks and exact teardown. User-visible aggregate throughput rose
   from 3.186 to 3.641 tok/s (14.3%), decode rose to 4.419 tok/s, and prefill rose
   to 8.364 tok/s. Across the five measured requests, promotions fell from 2,520
   to 140, copied bytes from about 67.4 GB to 3.74 GB, and tier-exchange time
   from about 26.6 seconds to 1.60 seconds. The truth set made 218 load-required
   misses across 1.39 million expert selections and ended with 129.05 GB of
   device payload plus 18.42 GB of host-visible payload. This removes broad tier
   churn as a material bottleneck. Target verification still consumed 318.36
   seconds across 691 speculative cycles, versus 14.28 seconds drafting and
   0.58 seconds of draft catch-up, and is now the dominant measured cost.

   A fresh equivalent llama.cpp build-10257 reference now exercises the
   package's DSpark sidecar rather than plain target decoding. With the Q2
   target at 131,072 context, five-token trained draft blocks, thinking enabled,
   65,536 allowed output tokens, and the same warmup plus five real conversation
   turns, llama.cpp averaged 10.762 decode tok/s (11.331 weighted aggregate),
   83.152 prefill tok/s, and accepted 1,733 of 5,175 drafted tokens (33.49%).
   The conversation remained coherent and recalled Greece correctly. This is a
   stronger direct reference than the earlier non-speculative 9.50 tok/s Q2
   run, but it also demonstrates that the original 25--30 tok/s expectation is
   not delivered by current llama.cpp DSpark on this Vulkan workstation. NERVE
   must nevertheless continue toward the product target: first close the
   semantic acceptance gap from 23.55% to at least the reference's 33.49%, then
   surpass the reference through cheaper target verification and transaction
   fusion rather than assuming DSpark alone supplies 30 tok/s.

   The exact six-lane target-verification MXFP4 microbenchmark now matches the
   anchor plus five trained DSpark proposals. On an AMD device it measured
   1.654 ms for gate/up and 0.907 ms for down, or 2.561 ms per 36-route layer.
   Across 43 target layers that is roughly 110 ms before attention, routing,
   synchronization, or transport, so the structurally generic MXFP4 sparse
   expert kernels are independently material and require optimization. A
   temporal-history dispatch-width experiment was rejected after a complete
   gate fell from 4.419 to 4.178 decode tok/s; an attempted replay shortcut also
   produced a repeated-answer quality failure. Both implementation paths and
   the invalid compiled package were removed. Do not revive partial command
   signatures: future replay work must prove equivalence against every field in
   the recorded Vulkan sequence and pass the complete conversation gate.

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
   tok/s and 174.03 prefill tok/s. After the cost-aware LFU milestone,
   Qwen3.6-35B-A3B passed at 104.06 decode tok/s and 163.91 prefill tok/s, while
   Qwen3.5-9B passed at 50.78 decode tok/s and 175.26 prefill tok/s. Run both
   gates again before every
   runtime-performance commit.
   Keep the live gate's repetition and conversation checks active so throughput
   alone cannot pass. Do not use faulted AMD devices merely to satisfy this
   gate; defer a model when no verified-healthy placement exists.

5. Perform a final adversarial review against `CONCEPT.md`: compiled artifacts
   remain self-contained and model-specific, while compiler discovery, runtime
   operators, graph wiring, placement, representation, and residency remain
   capability-driven and reusable by unseen models. Finish with an empty TODO,
   a clean worktree, and every milestone committed and pushed.
