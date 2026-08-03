# TODO

## Goal

Compile and run `/mnt/models/models/deepseek-v4/flash-0731/safetensors` as a
daily-driver NERVE model. Preserve source behavior, keep sparse routed experts
independently demand-resident, expose runtime per-component representation
selection, and optimize real multi-turn agentic inference to at least 30 tok/s,
with roughly 50 tok/s as the primary target and roughly 70 tok/s as a
speculative-decoding stretch target. Preserve supported Qwen models throughout.

## Work queue

1. Finish the identity-independent `parallel_backbone_markov` speculative
   decoder. CPU-side SPIR-V reflection exposed that raw speculative graphs lost
   compiled descriptor bindings during device planning; device planning now
   materializes every executable component contract, while dedicated phases
   remain outside the generic graph based on their compiled execution role.
   The compiler now preserves the trained physical backbone width of five
   independently from the recommended user-visible window of seven and records
   the source-context tick offset. Runtime fan-out aliases every local sibling
   edge to one produced-port buffer, derives each processor's
   `committed_target_only` state dependency cone, excludes that cone from every
   speculative lane, and advances it once per retained prompt or accepted target
   frame. A four-cycle device-state proof showed occupied KV slots advancing
   5 -> 7 -> 10 -> 13 instead of being contaminated by all five proposal lanes;
   coherent multi-turn inference then reached 26.82% acceptance on its first
   response and 15.85% on the next. Complete the same produced-port fan-out
   contract for mixed local/outgoing and multiple cross-device consumers,
   replace repeated source-tap recapture with a compiled retained-frame
   state-ingestion schedule, and pass the complete sampled, thinking-enabled
   two-set DeepSeek conversation gate before enabling this decoder by default.

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

3. Extend the generic quality gate with a representative tool-call round trip
   and long-stream continuity case, using package-owned chat-template behavior
   rather than a model-name branch. Keep official thinking and sampling
   defaults, the 65,536-token output allowance, agentic context, coherent final
   answers, turn recall, and clean teardown. Apply the same complete-conversation
   warmup rule to these cases before accepting their performance measurements.

4. Raise DeepSeek's user-visible accepted decode rate past 30 tok/s and toward
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

   The first complete thinking-enabled two-set gate for this graph remained
   coherent and retained the model in one process. Its discarded warmup averaged
   4.500 decode tok/s and 7.755 prefill tok/s while loading 10,850 units / 145.06
   GB. Its truth set averaged only 3.933 decode tok/s and 7.251 prefill tok/s even
   though it loaded just 169 additional units / 2.259 GB across 1,335,468 expert
   selections. Truth-turn DSpark acceptance fell from 25.00% to 22.25% on the
   final turns, and the retained model ended with 137.62 GB device payload plus
   9.69 GB host-visible payload. The final truth turn still performed 5,767
   sequence submissions, 6,137 sequence fence waits, 4,733 copy submissions,
   and 3,831 copy waits. It selected zero optimized representations. The causal
   kernels are therefore complete enough to expose the actual bottleneck: host
   orchestration and residency synchronization dominate target verification even
   when nearly every selected expert is already resident.

   The resident-hit path now uses the GPU address table, presence map, compact
   miss queue, and one cross-device continuation predicate. A contiguous causal
   pipeline submits every demand-resident device slice before waiting only for
   the terminal slice; the first real miss suppresses downstream work, the owning
   slice resolves it, and only the affected suffix is resubmitted. Exact tests
   cover local guarding, shared initial guarding, resume ordering, and invalid
   gate layouts. The complete five-GPU gate remained coherent and cleanly
   released every NERVE allocation, but its truth result was 3.908 decode tok/s
   and 7.175 prefill tok/s versus 3.933 and 7.251 before: correct but
   performance-neutral. On the final truth turn, sequence submissions fell from
   5,767 to 5,580, sequence waits from 6,137 to 5,546, copy submissions from
   4,733 to 4,495, and copy waits from 3,831 to 3,657. Yet 79 target-verification
   cycles still consumed 43.695 seconds (about 553 ms per cycle), while DSpark
   accepted only 25.84% of proposals. The remaining synchronization is above the
   layer pipeline rather than inside it.

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

5. Before every runtime-performance commit, run Qwen3.6-35B-A3B and
   Qwen3.5-9B quality/performance gates sequentially on equivalent healthy AMD
   placement. The latest current-schema packages pass the full sampled,
   thinking-enabled five-turn gate with correct Greece recall. After the
   cross-device deferred-demand milestone, Qwen3.6-35B-A3B averaged 109.29 decode
   tok/s and 211.66 prefill tok/s versus 109.88 and 216.45 before, while
   Qwen3.5-9B averaged 59.58 decode tok/s and 180.20 prefill tok/s versus 59.81
   and 184.31 before. Both decode deltas are below 0.6%, both gates passed, and
   both released their NERVE GPU reservations exactly. Run both gates again
   before every runtime-performance commit.
   Keep the live gate's repetition and conversation checks active so throughput
   alone cannot pass. Do not use faulted AMD devices merely to satisfy this
   gate; defer a model when no verified-healthy placement exists.

6. Perform a final adversarial review against `CONCEPT.md`: compiled artifacts
   remain self-contained and model-specific, while compiler discovery, runtime
   operators, graph wiring, placement, representation, and residency remain
   capability-driven and reusable by unseen models. Finish with an empty TODO,
   a clean worktree, and every milestone committed and pushed.
