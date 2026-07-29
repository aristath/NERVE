# Compiler residency analysis

This records completion of the compiler-analysis milestone for generic
demand-resident immutable resources. Qwen3.6-35B-A3B is the real proof
workload, not a model-family branch in the implementation.

## Implemented contract

The compiler now discovers selectable resources from physical graph semantics:

- a producer declares an exact `selection_domain`;
- physical consumers declare exact `selected_parameter_accesses`;
- each access names the selection signal, the contiguous tensor partition axis,
  and the complete sorted parameter set it selects;
- all selected accesses attached to one selector form one atomic resource
  bundle; and
- every referenced tensor is classified as part of the always-resident spine or
  as a partitioned dynamic resource.

The analysis does not inspect operation, component, tensor, architecture, or
model names. It validates physical signal production and ordering, direct
signal consumption, parameter ownership, tensor geometry, row-major
contiguity, complete atomic membership, and exact access coverage. Ambiguous
metadata, repeated parameters, selected/unconditional overlap, incompatible
partitioning, missing selectors, missing parameters, non-contiguous axes, and
unproven packaging transforms fail compilation.

Compatible immutable partitions reused by independent selectors are
content-addressed once and share one compact partition template. This preserves
the later requirement that duplicated or rewired graph instances share
immutable resources while their mutable state remains independent.

The compiled contract contains no device choice, placement, memory capacity,
initial working set, prefetch, or eviction policy. Those remain runtime facts.

## Compact packaging

Dynamic tensor banks remain large Safetensors artifacts. During the existing
copy pass, the compiler also computes one SHA-256 digest per independently
addressable partition. All partition digests are stored in one bounded table:

```text
integrity/resource_partitions.sha256
```

The manifest describes regular resources with affine range templates:
base offset, byte stride, byte count, alignment, digest-table offset, and digest
stride. It does not emit one manifest object or one file per expert.

Composite source tensors are streamed in logical output order, and partition
digests remain correct even when a partition crosses source-part boundaries.
Transforms that have not proven partition preservation fail closed.

## Compiler fixtures and adversarial review

The sequential compiler tests cover two structurally different patterns:

1. a selector whose atomic unit spans four tensors used across two physical
   compute stages; and
2. an optional projection bank with an unconditional bias, including reuse of
   the same immutable partitions by two independent selectors.

Adversarial cases cover missing access metadata, unknown or repeated
parameters, malformed metadata fields, non-contiguous partition axes,
selected/unconditional overlap, incompatible geometry, transform boundaries,
composite source parts, incomplete integrity output, and compiler-authored
runtime policy fields.

The review found and fixed one architectural issue before publication: an
earlier draft rejected compatible reuse of a partitioned tensor across
selectors. The final compiler deduplicates the identical partition template,
while Python and Rust validation require every template to have at least one
selector and every selector to have exactly one checkpoint.

Validation from the final source:

```text
Python compiler/package tests: 154 passed sequentially
Focused planner tests:          14 passed sequentially
Rust contract tests:             5 exact tests passed, one thread each
Rust feature check:              vulkan,tokenizers passed
Rust formatting:                 passed
Focused Ruff check:              passed
```

## Fresh real-model compile

- Source:
  `/mnt/models/models/qwen3.6/35b-a3b-fp8/safetensors`
- Compiled package:
  `/mnt/models/models/compiled/nerve-milestones/qwen3_6_35b_a3b_fp8-residency-analysis-v1`
- Model type discovered from structure: `qwen3_5_moe_text`
- Package id: `model_3ddc71b0_vulkan_resident`
- Circuits: 46
- Shaders: 113
- Package tree bytes: 36,638,909,856
- Manifest bytes: 19,074,375
- Manifest SHA-256:
  `268f8250149b713b0f7013de31080acc5fd511b30496e9260f57b79ff171dd47`

Compiled residency inventory:

| Fact | Result |
| --- | ---: |
| Concrete spine resources | 805 |
| Always-resident atomic groups | 1 |
| Dynamic partition templates | 41 |
| Selectors / checkpoints | 41 / 41 |
| Semantic parameter bindings | 970 |
| Partitions per template | 256 |
| Members per atomic partition | 4 |
| Dynamic tensors | 164 |
| Dynamically addressable bytes | 33,021,591,552 |
| Partition digest-table bytes | 1,343,488 |

The digest-table SHA-256 is
`8d094bbd05a21b7e4f49cf70442517df1676bb63a066e841392636eb4cb25c34`.
A direct hash of a compiled partition range exactly matched its independently
indexed table digest.

## Conversation and performance proof

The newly compiled package was loaded once on the two explicitly allowlisted
AMD GPUs. The normal chat runtime used a 131,072-token context, a 65,536-token
output allowance, thinking enabled, two MTP draft tokens, seed 0, one discarded
warmup, and the five canonical measured turns.

| Metric | Result |
| --- | ---: |
| Warmup decode | 43.716 tok/s |
| Mean measured prefill | 66.385 tok/s |
| Mean measured decode | 41.067 tok/s |
| Required decode floor | 30.000 tok/s |

Measured decode rates were `46.199`, `46.725`, `36.689`, `33.079`, and
`42.643` tok/s. The prior contract milestone measured 43.139 tok/s. The current
run generated materially different reasoning lengths, while its warmup was
within 1.5% of the prior warmup and this milestone changes compilation metadata,
not the eager execution kernels. No material execution regression was found.

The gate passed visible thinking, termination, non-repetition, identity,
Athens correctness, a meaningful qualified Corinth answer, and cross-turn
recall of Greece.

- Gate report SHA-256:
  `359715fef784864828ff03a23b73da5f7b6570f05d9583b406f8557e19fec07e`
- Transcript SHA-256:
  `7e5b21ec1dc68e33b3bb853243a03c683dc5ae2ea2055158fd16b1d340278cb2`

Compilation and inference used the RADV-only Vulkan ICD and explicit AMD UUID
allowlists. Before and after each operation, both AMD devices were at exactly
59,973,632 bytes VRAM and 0% busy. NVIDIA was neither enumerated nor used.
