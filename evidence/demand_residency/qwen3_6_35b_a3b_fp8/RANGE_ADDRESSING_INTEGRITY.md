# Range Addressing and Independent Integrity Evidence

## Acceptance result

Milestone 4 is complete.

The compiled package keeps large tensor banks while exposing exact,
alignment-constrained byte ranges for independently selectable resources.
Every dynamic range resolves to its own SHA-256 entry, so the runtime can read
and verify one selected atomic group without reading or hashing the other
partitions in the bank. Concrete and compact dynamic resources share one
strict package-validation path in Python and Rust; there is no legacy package
fallback.

## Fresh compiled-package proof

Source:

`/mnt/models/models/qwen3.6/35b-a3b-fp8/safetensors`

Compiled package:

`/mnt/models/models/compiled/nerve-milestones/qwen3_6_35b_a3b_fp8-residency-analysis-v1`

Manifest SHA-256:

`e2dc26027a37366030f89fae6f6e3ae686fea63ed245b4aa4f1bc62fc7091e95`

The fresh package contains:

- 805 concrete always-resident resources;
- 41 compact dynamic partition templates;
- 164 dynamic member templates;
- 256 addressable partitions per template; and
- one 1,343,488-byte digest table covering all 41,984 dynamic range digests
  exactly, without gaps or an unreferenced suffix.

As a direct independent-read proof, main routed partition 17 resolved to one
atomic group with four resources and four ranges. Only 3,146,112 bytes were
read and verified:

| Range | Artifact | Offset | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `weights/tensor_4b4bb68072c8a121.safetensors` | 17,825,984 | 1,048,576 | `7d5f259ea7ce70cc9f5e8ba97c52c084f2acc6909da31670c76dbae876ea81de` |
| 2 | `weights/tensor_e8bede7d5e25b8fd.safetensors` | 35,651,784 | 2,097,152 | `03b3dbfc84f4a36f9a2457f8d268f1571ded2964f790fc81cfd24898e520ee31` |
| 3 | `weights/tensor_52ed053e06c4a175.safetensors` | 4,552 | 256 | `afafae4e8bb638f02838ca550779d419807a61ca0464675145868f5144d3ad27` |
| 4 | `weights/tensor_c8752289ef4a4f05.safetensors` | 2,368 | 128 | `52c57180b27325c7760b38d8666512f97a3c474b922fef1439de2bbb2df5f11b` |

Corrupting partition 0 did not affect the independently resolved and verified
partition 1 read. Requesting partition 0 then failed its range digest, proving
that verification is selected-range-local rather than whole-bank-local.

## Validation and adversarial proof

The validators fail closed on:

- data or digest-table truncation;
- selected-range and digest-table corruption;
- unsafe package paths;
- misalignment and invalid strides;
- concrete/concrete, dynamic/dynamic, and concrete/dynamic overlap;
- digest gaps, aliases, conflicting contracts, and unreferenced suffixes;
- incomplete or duplicate atomic membership;
- a group seed that does not commit to every member; and
- concrete or partition members omitted from semantic bindings.

Relocating and renaming the package preserves resolution, identities, reads,
and verification because all storage addresses are package-relative.

Sequential verification:

- Python package, residency-contract, and planning suites: 69 passed.
- Rust range resolver/integrity tests: 5 individually selected tests passed.
- Rust identity, binding, package-loader, state-machine, and compatible-template
  tests: 6 individually selected tests passed.
- `cargo check`: passed.
- Rust formatting, Python lint, and `git diff --check`: passed.

## Eager inference and performance

The freshly compiled package completed the canonical thinking-enabled
conversation on the two AMD Radeon AI PRO R9700 devices. The model remained
mounted for the warmup and all five measured turns. The warmup timing was
discarded.

- Context capacity: 131,072 tokens.
- Maximum generated tokens: 65,536.
- MTP draft tokens: 2.
- Warmup: `hi`, 44.658 decode tok/s.
- Mean of five measured turns: **41.510 decode tok/s**.
- Mean prefill: **80.855 tok/s**.
- Measured-turn decode rates: 46.185, 41.932, 37.812, 37.356, and 44.265
  tok/s.

The responses were coherent, answered Athens correctly, qualified the
ambiguous Corinth count, reported the model's cutoff, and recalled Greece in
the final turn. This is above the 30 tok/s product floor and does not regress
the preceding 41.067 tok/s compiler-residency baseline.

Benchmark report SHA-256:

`c59b85de7c9e37878d05ce48b355714e731682aa8d0fce321d920b0665062b3b`

Transcript SHA-256:

`931b461282e8f94b0787afdc887c3d5b1b7ba9c35160c35e886b19c4dfe2dc87`

Both AMD devices returned to the exact pre-workload idle baseline after
execution: 59,973,632 bytes of VRAM in use and 0% busy on each device.
