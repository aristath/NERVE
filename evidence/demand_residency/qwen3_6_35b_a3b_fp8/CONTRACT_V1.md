# Generic compiled-resource residency contract

This records completion of the versioned residency-contract milestone. The
contract is generic compiler/runtime infrastructure; Qwen3.6-35B-A3B is only
the real-model proof workload.

## Contract

- Outer package schema: `nerve.vulkan_resident_model_package.v5`
- Residency schema: `nerve.compiled_resource_residency.v1`
- Identity algorithm: `nerve.resource_identity_sha256.v1`
- State machine: `nerve.resource_residency_state_machine.v1`
- Declared policies: `demand_retained`, `eager`
- Lifetimes: `always_resident`, `dynamic`
- Per-device states: `absent`, `requested`, `loading`, `resident`, `failed`

The compiler and Rust runtime share strict types and validation for immutable
resources, bounded byte ranges, integrity evidence, compatibility, atomic
groups, dependencies, compact affine partition templates, selectors, and
physical resume checkpoints. Unknown fields and unknown schema versions fail
closed.

Concrete resource identities include content integrity and compiled
compatibility semantics but exclude package paths, byte offsets, tensor names,
model names, and runtime placement. Atomic-group, selector, template, and
checkpoint identities are derived from their compiled semantics. The same
derived partition identity is verified byte-for-byte in Python and Rust.

The current compiler emits an exact eager baseline: 969 referenced immutable
resources in one always-resident atomic spine, with 970 semantic access
bindings. It deliberately does not manufacture one residency group per tensor.
Compiler residency analysis will replace this conservative spine with dynamic
groups and compact partition templates in the next milestone.

## Validation

Sequential Python validation:

```text
tests/test_resource_residency_contract.py
tests/test_package_integrity.py
tests/test_optimizer_runtime_target.py

58 passed
```

The broader model-package suite passed 150 tests before the final three
contract-adversarial cases were added. Exact sequential Rust checks covered:

- explicit terminal-state clearing;
- cross-language derived partition identities;
- path-independent concrete resource identities;
- rejection of unknown outer and nested contract fields;
- rejection of unsupported package and residency schema versions; and
- loading a package with self-contained integrity artifacts.

The adversarial review also covered concrete dynamic group selection, compact
partition selection, selector/count and checkpoint-boundary drift, incomplete
digest-table coverage, duplicate atomic membership, malformed alignment,
semantic-binding drift, relocation, and the absence of Qwen, MoE, expert,
model-type, or architecture branches in either production contract
implementation.

## Real package

- Source:
  `/mnt/models/models/qwen3.6/35b-a3b-fp8/safetensors`
- Compiled package:
  `/mnt/models/models/compiled/nerve-current/qwen3_6_35b_a3b_fp8`
- Package id: `model_3ddc71b0_vulkan_resident`
- Package tree bytes: `36,637,077,217`
- Manifest bytes: `18,847,156`
- Manifest SHA-256:
  `ada167009b0d47446a5750eec1e9baca400f135946d8b6c4c0c115e75e9f34b2`

The newly compiled package passed compiler validation with 46 circuits and 113
shaders.

## Conversation and performance proof

The normal runtime chat used both explicitly allowlisted AMD GPUs, a 131,072
context, a 65,536-token output allowance, thinking enabled, two MTP draft
tokens, seed 0, and the canonical warmup plus five measured turns. The first
turn was excluded from the means.

| Metric | Result |
| --- | ---: |
| Setup | 34,267.719 ms |
| Warmup decode | 44.341 tok/s |
| Mean measured prefill | 83.951 tok/s |
| Mean measured decode | 43.139 tok/s |
| Required decode floor | 30.000 tok/s |

Measured decode rates were `47.739`, `47.539`, `37.492`, `41.732`, and
`41.193` tok/s. The prior eager MTP baseline was 43.521 tok/s, so the
contract-only change was 0.88% lower in this single conversational sample and
introduced no material regression. The gate passed visible thinking,
non-repetition, Athens correctness, a meaningful Corinth answer, and cross-turn
recall of Greece.

- Gate report SHA-256:
  `0e023c401866d28876c88e0009dacfd81f15ecb5230f6ceccb4889729c469c12`
- Transcript SHA-256:
  `cb22bf9a3406da0c6983f5482694bf17082ae3a4ed87ff7ee9a174372f94865e`

Before compilation, before inference, after compilation, and after runtime
teardown, both allowlisted AMD GPUs were at the exact 59,973,632-byte VRAM idle
floor and 0% busy. No NVIDIA device was enumerated or used.
