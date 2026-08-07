# JSON Schema Notes

The benchmark output schema is currently identified as
`nerve.gpu_benchmark_run.v1`.

This document is descriptive rather than a formal JSON Schema file. The Rust
types in `src/model.rs` are the source of truth for the first implementation.

## Top-Level Fields

- `schema`: output document schema name.
- `started_at_unix_ms`, `finished_at_unix_ms`: wall-clock run timestamps.
- `implementation`: benchmark package name, version, and backend status.
- `policy`: user-selected run policy, including payload size and exclusions.
- `discovered_targets`: every target discovered before policy filtering.
- `selected_target_ids`: targets selected for this run.
- `skipped_targets`: discovered targets skipped by explicit policy.
- `workload_specs`: logical benchmark contracts generated for the selected
  payload size.
- `measurements`: single-target measurements.
- `pair_measurements`: ordered pair and two-target synthetic placement
  measurements.
- `diagnostics`: non-fatal implementation notes.

## Workload Specs

The default payload is 5 MiB. Workload specs scale with
`--payload-bytes`, but stay intentionally small.

Current workload specs:

- `single_target_gpu_small_payload`: full logical payload on one target.
- `ordered_activation_transfer`: activation-sized movement from one target to
  another.
- `synthetic_layer_split_small_payload`: first half of the logical payload on
  source, activation transfer, second half on destination.
- `synthetic_tensor_split_small_payload`: split logical payload across two
  targets, broadcast activation, compute shards, collect output.

GPU-backed specs are emitted before the Vulkan backend exists. Their
measurements use `status: "unmeasured"` until a backend can execute them.

## Target Policy

The schema records exclusions as run policy. It does not encode permanent bans
for CPU, integrated GPU, discrete GPU, AMD, NVIDIA, Intel, or any other target
class.
