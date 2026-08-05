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
   packed-weight reconstruction were not amortized. Hardware capability is
   candidate discovery, not promotion: the rejected schema-v4 experiment
   selected expanded FP8 merely because every target exposed the required
   features. It doubled sparse-expert residency, reduced the first DeepSeek
   turn to 0.719 tok/s, and caused 207.3 GB of uploads for 143 decode tokens.
   The baseline compiler now keeps native MXFP4; attach any derived resident
   representation only through measured optimizer evidence that includes the
   whole routed working set, footprint, reload cost, and device headroom. The
   package/runtime contract can now materialize one explicitly selected MXFP4
   to FP8 derivation without changing source artifacts. Complete the remaining
   alternative-set contract, device-local measurement evidence, and runtime
   choice so native and derived representations can coexist and be selected per
   component on the component's actual target device.

2. Extend the generic quality gate with a representative tool-call round trip
   and a long-stream continuity case. Use package-owned chat behavior, official
   thinking and sampling defaults, the 65,536-token output allowance, agentic
   context, complete-conversation warmup, coherent final answers, turn recall,
   and exact teardown. Exercise both successful canonical commits and malformed
   output rollback without model-name branches.

3. Raise DeepSeek's accepted decode rate past 30 tok/s and continue until no
   material bottleneck remains. First compile a fresh package from the source
   checkpoint using native MXFP4 as the sparse-expert baseline; never rerun the
   rejected expanded-FP8 package. The authoritative thinking-enabled product
   gate uses 128K context, a 65,536-token output allowance, a requested
   speculative window of seven clamped to the checkpoint's trained five-token
   block, contiguous placement across all five explicitly bound AMD GPUs, and
   one complete discarded conversation before the truth conversation in the
   same resident process. The previous safe native package terminated
   coherently, preserved Greece recall, released all acquired capacity, and
   reached 8.818 decode tok/s and 8.942 prefill tok/s. This remains far below
   the 30 tok/s floor and must be remeasured on the current runtime before
   attribution work continues.

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
   full-horizon causal component harness proved that the indexed-attention
   temporal lanes are independent when each lane reads its declared state
   snapshot. The promoted generic workgroup-Y implementation preserves exact
   output and committed-state digests, is 2.40x faster in the 128K component
   benchmark, and passes the complete product gate without malformed output.
   Target verification still consumes 269.44 seconds, about 92% of the measured
   decode interval. Demand residency still performs 254 loads, 42 evictions,
   and 40 reloads across the truth conversation. Restore reproducible execution
   before using whole-process token digests for attribution: with the same
   package, seed, prompts, placement, and sampler, fresh processes can diverge
   after the first turn. Audit radix/top-k tie order, speculative sampler RNG
   consumption, accepted-prefix commit, rollback, and reset state. Require
   repeated-process equality for committed tokens, routed experts, accepted
   prefixes, and state digests; do not weaken the product quality gate to
   accommodate nondeterminism.

   Separate target compute from residency churn with device timestamps, then:

   - compile draft, target projection, comparison, state selection, commit, and
     draft catch-up into one device-resident transaction with no host wait until
     a residency miss or completed emitted block;
   - eliminate reload churn using route-aware expert residency decisions that
     remain capacity- and evidence-driven rather than model-specific;
   - optimize independently material MXFP4 sparse-expert kernels with fused or
     amortized representations, including measured native INT4/FP8 candidates;
   - close the remaining speculative acceptance gap without sacrificing model
     quality.

   Do not revive the rejected standalone intermediate-FP8 quantization
   dispatch, incomplete replay signatures, or naive workgroup-Y temporal-lane
   split. Every candidate must win exact microbenchmarks, output-and-state
   equivalence, repeated-process determinism, and the complete product gate.

4. Before every runtime-performance commit, run Qwen3.6-35B-A3B and Qwen3.5-9B
   quality/performance gates sequentially on equivalent healthy AMD placement.
   Restrict discovery to the AMD Vulkan ICD and bind PCI-derived UUIDs so an
   added Intel or NVIDIA adapter cannot silently change the comparison. The
   current thinking-enabled gates each discard one complete in-process warmup
   conversation, reset model state without unloading, pass correct Greece
   recall, and release their exact recorded reservations. On the current
   pressure-safe runtime, Qwen3.6-35B-A3B reaches 101.05 decode tok/s and 196.34
   prefill tok/s with a contiguous two-AMD split. Qwen3.5-9B reaches 48.96
   decode tok/s and 123.65 prefill tok/s on one AMD using the official
   temperature 1.0, top-k 20, top-p 0.95, min-p 0, presence-penalty 1.5,
   repetition-penalty 1.0 thinking profile. Keep repetition,
   structured-protocol, conversation, and teardown checks active so throughput
   alone cannot pass.

5. Perform a final adversarial review against `CONCEPT.md`: compiled artifacts
   remain self-contained and model-specific, while compiler discovery, runtime
   operators, graph wiring, placement, representation, residency, and stream
   transactions remain capability-driven and reusable by unseen models. Finish
   with an empty TODO, a clean worktree, and every milestone committed and
   pushed.
