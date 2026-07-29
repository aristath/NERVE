# Qwen3.6-35B-A3B eager residency and route coverage

This is the milestone-1 evidence for the generic demand-resident resource
work. It records the eager baseline and the selector working set that the later
residency implementation must reproduce. The model is a proof workload, not a
runtime special case.

## Reproduction contract

- Source model:
  `/mnt/models/models/qwen3.6/35b-a3b-fp8/safetensors`
- Compiled package:
  `/mnt/models/models/compiled/nerve-baselines/qwen3_6_35b_a3b_fp8-selection-telemetry`
- Package schema: `nerve.vulkan_resident_model_package.v4`
- Package id: `model_3ddc71b0_vulkan_resident`
- Package tree bytes: `36,635,987,419`
- Manifest SHA-256:
  `0ce82c99ab50c1001753fbe2a70070e9a2df80dabd2a0442a94319d394dd8684`
- Context: 131,072 activations
- Output allowance: 65,536 tokens
- Thinking: enabled
- Sampler seed: 0
- Placement: every main component internally sharded across `gpu0,gpu1`
- `gpu0`: `vulkan-uuid:00000000070000000000000000000000`
- `gpu1`: `vulkan-uuid:000000000a0000000000000000000000`
- Vulkan ICD:
  `/usr/share/vulkan/icd.d/radeon_icd.x86_64.json`

The RADV-only ICD and explicit UUID allowlist exclude the NVIDIA device. Both
AMD devices were verified at 59,973,632 bytes VRAM and 0% busy before each
workload and after every unload.

Compile with:

```bash
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
python -m nerve \
  --compile-model /mnt/models/models/qwen3.6/35b-a3b-fp8/safetensors \
  --compiled-model-dir /mnt/models/models/compiled/nerve-baselines/qwen3_6_35b_a3b_fp8-selection-telemetry \
  --runtime-bin runtime-rs/target/release/nerve-runtime \
  --allow-physical-device vulkan-uuid:00000000070000000000000000000000 \
  --allow-physical-device vulkan-uuid:000000000a0000000000000000000000 \
  --compiler-events-jsonl
```

Run the conversation gate with the normal `--chat` runtime. Construct one
`--shard-component layer_NN=gpu0,gpu1` argument for each component from
`layer_00` through `layer_39`, bind the two UUIDs above, and use
`--chat-template-var enable_thinking=true --context-size 131072
--max-new-tokens 65536`. The gate owns the warmup and five measured prompts.

Route evidence is regenerated from an ordinary chat transcript with:

```bash
python scripts/analyze_selection_coverage.py TRANSCRIPT \
  --resource-bytes 3146112 \
  --turn-label warmup_hi \
  --turn-label who_are_you \
  --turn-label capital_of_greece \
  --turn-label corinth_count \
  --turn-label knowledge_cutoff \
  --turn-label cross_turn_recall \
  --compact --output COVERAGE.json
```

The checked-in compact reports retain aligned per-domain arrays for turn
coverage, new resources, reuse, cumulative coverage, route counts, and the
three hottest resources. The domain order is explicit in each report.

## Smallest useful atomic resource

Every selected routed expert is consumed immediately by both physical expert
stages. The compiled circuit binds:

| Required slice | Dtype | Bytes per expert |
| --- | ---: | ---: |
| gate/up weights | FP8 E4M3 | 2,097,152 |
| gate/up inverse block scales | BF16 | 256 |
| down weights | FP8 E4M3 | 1,048,576 |
| down inverse block scales | BF16 | 128 |
| **Complete expert bundle** |  | **3,146,112** |

The selector produces one expert identity used by both stages; neither stage is
optional and both execute before reduction. Splitting this bundle would create
multiple inseparable loads for every miss without reducing retained memory.
The complete four-slice bundle is therefore the smallest useful atomic
residency unit for this workload. The 40 target domains expose 10,240 bundles
(`32,216,186,880` bytes). The one MTP domain exposes another 256 bundles
(`805,404,672` bytes).

## Eager mount

The paused normal chat mount reported `setup_ms=33,189.343` and had:

| Device | Absolute VRAM | Increase over idle |
| --- | ---: | ---: |
| gpu0 | 22,277,849,088 | 22,217,875,456 |
| gpu1 | 16,348,246,016 | 16,288,272,384 |
| **Total** |  | **38,506,147,840** |

After `/exit`, both devices returned exactly to 59,973,632 bytes and 0% busy.

## Performance and behavior

All rows are real thinking-enabled conversations with a 65,536-token output
allowance. The five measured turns are averaged after the warmup.

| Runtime | MTP draft tokens | Setup ms | Mean prefill tok/s | Mean decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| eager before telemetry | 0 | 32,897.677 | 70.015 | 30.305 |
| eager with selection telemetry and phase checkpoints | 0 | 33,375.916 | 75.250 | 29.502 |
| eager with selection telemetry | 2 | 33,885.812 | 83.830 | 43.521 |

Exact routing telemetry costs 2.65% mean decode throughput versus the prior
unobserved run. Phase checkpoints add no further measurable hot-path cost
(29.540 before versus 29.502 after); they execute only after an engine phase is
already idle. The regression is recorded rather than rounded away;
MTP-enabled normal execution remains well above the 30 tok/s goal. The
MTP-disabled warmup decoded at 32.350 tok/s.

Every gate passed response structure, non-repetition, Athens correctness,
Corinth relevance, and cross-turn recall of Greece. Teardown returned both AMD
devices to the exact idle baseline.

## Dense-model control

The instrumented runtime also mounted the compiled dense
Qwen3.6-27B-FP8 package across the same two AMD devices at a 131,072-token
context and a 65,536-token output allowance. A thinking-enabled normal chat
answered the capital-of-Greece prompt correctly, emitted no selection-domain
report, and returned both devices to exactly 59,973,632 bytes and 0% busy.
This proves that a package with no compiled selection domains incurs no
telemetry allocation or model-family fallback.

## First-output frontier

A separate one-token diagnostic was used only to locate the residency
frontier; it was not used as a performance or quality benchmark. Normal phase
checkpoints distinguish canonical user prefill, generation-prompt execution,
and assistant-state commit.

Before assistant commit, the first real chat prompt and first sampled output
had cumulatively touched 1,612 target bundles, or `5,071,532,544` bytes. The
per-layer selected counts, in `layer_00` through `layer_39` order, were:

```text
61,45,41,52,47,34,31,42,42,38,39,53,43,40,35,41,39,39,38,38,
38,37,39,48,40,37,32,42,40,40,38,36,36,37,32,34,35,43,44,46
```

This is the strict prompt-to-first-output demand frontier. Assistant commit is
reported separately and is not included.

## Retained working-set curve

### MTP disabled

| Turn | New bundles | Reused bundles | Cumulative bundles | Dynamic bytes |
| --- | ---: | ---: | ---: | ---: |
| warmup `hi` | 6,189 | 0 | 6,189 | 19,471,287,168 |
| Who are you? | 1,186 | 5,733 | 7,375 | 23,202,576,000 |
| capital of Greece | 523 | 5,541 | 7,898 | 24,847,992,576 |
| Corinth count | 1,250 | 7,543 | 9,148 | 28,780,632,576 |
| knowledge cutoff | 39 | 6,186 | 9,187 | 28,903,330,944 |
| cross-turn recall | 70 | 7,275 | 9,257 | 29,123,558,784 |

The final retained target working set is 90.40% of all target bundles. Demand
residency therefore provides a large mount-time and short-session benefit, but
this particular long conversation converges toward eager residency. The policy
must report this honestly rather than imply an unbounded saving.

### MTP enabled

| Turn | Target cumulative | Target bytes | Draft cumulative | Draft bytes |
| --- | ---: | ---: | ---: | ---: |
| warmup `hi` | 6,506 | 20,468,604,672 | 200 | 629,222,400 |
| Who are you? | 7,413 | 23,322,128,256 | 217 | 682,706,304 |
| capital of Greece | 7,844 | 24,678,102,528 | 222 | 698,436,864 |
| Corinth count | 9,002 | 28,321,300,224 | 239 | 751,920,768 |
| knowledge cutoff | 9,065 | 28,519,505,280 | 239 | 751,920,768 |
| cross-turn recall | 9,164 | 28,830,970,368 | 243 | 764,505,216 |

The draft domain is independent from the 40 target domains. This proves that
MTP resources can use the same generic selection-domain mechanism while
retaining separate accounting.

## Evidence hashes

| Artifact | SHA-256 |
| --- | --- |
| pre-telemetry gate report | `81fd63e94fedce6656730dbf241fd84147d5d25d17b23647cf28be9cf009ff20` |
| final MTP-off gate report | `265af3e3a8b367a0e7adeefc23001c1ad24a385f6328b6b8f2aa4d61a8b9cb89` |
| MTP-on gate report | `38e0786fb084ad92e72424c4b5a34f2de92a33ec79d42009676d97c7d503c626` |
| final MTP-off transcript | `2646693a79088786390d08267b8767cdbbc3d19cec36b35dbafb2de05630ca11` |
| MTP-on transcript | `d8c7ea1c6da633b61ad47e98666e8adef657efb88aaab2868fa2552b9e367361` |
| first-output phase transcript | `9659e8f1cc23b1bb63caee3d8301ecd72458dcca553b0f55a222bad513ecf12e` |
| dense-model control transcript | `aa6df0f4482aef58e01f7dd464e6755179448ac0ec8555ea3eabb85782b588a8` |
