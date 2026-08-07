# JSON Schema Notes

The benchmark output schema is currently identified as
`nerve.gpu_benchmark_run.v1`.

This document is descriptive rather than a formal JSON Schema file. The Rust
types in `src/model.rs` are the source of truth for the first implementation.
Use `nerve-gpu-bench validate --input <path>` to parse and run basic structural
validation on a saved result.
Use `nerve-gpu-bench summarize --input <path>` for a compact status and target
summary of the same saved result.
The summary groups status counts by `(comparison_group, placement_strategy)` so
missing one-target, two-target-serial, or two-target-parallel evidence is easy
to spot. Coverage warnings are emitted when a selected run has no completed
measurement for an expected placement strategy. It also resolves every
`comparison_sets[].candidates[]` entry against the recorded measurements and
reports whether that candidate is completed, unmeasured, missing, failed,
unsupported, or skipped.

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
- `comparison_sets`: planner-facing candidate sets. For each selected pair, the
  set currently compares first-target-only, second-target-only, serial
  first-to-second, serial second-to-first, and parallel two-target execution.
- `measurements`: single-target measurements.
- `pair_measurements`: ordered pair and two-target synthetic placement
  measurements.
- `group_measurements`: three-target synthetic placement measurements when
  enough targets are selected and `--max-group-size 3` is active.
- `diagnostics`: non-fatal implementation notes.

## Workload Specs

The default payload is 5 MiB. Workload specs scale with
`--payload-bytes`, but stay intentionally small.

Placement decisions must compare alternatives inside the same
`comparison_group`. The initial group is
`small_payload_placement_comparison`, which explicitly separates:

- `single_target_serial`: the whole payload on one target;
- `two_target_serial`: the same logical payload split into ordered stages across
  two targets;
- `two_target_parallel`: the same logical payload split into parallel shards
  across two targets; and
- three-target serial/parallel variants when triplet records are requested.

That distinction is intentional: a single GPU can beat two-GPU parallelism, and
two-GPU serial placement can beat or lose to both depending on compute cost,
transfer cost, synchronization, and format path.

For two-target comparisons, serial placement is directional. A result should
preserve both A-to-B and B-to-A candidates because the activation transfer path,
target speed, and stage ownership may not be symmetric.

Current workload specs:

- `single_target_gpu_small_payload`: full logical payload on one target.
- `cpu_reference_serialized_small_payload`: CPU reference execution of the full
  logical payload as one serialized pattern.
- `cpu_reference_layer_split_small_payload`: CPU reference execution of the same
  logical payload using the layer-split dataflow shape.
- `cpu_reference_tensor_split_small_payload`: CPU reference execution of the
  same logical payload using the tensor-split dataflow shape.
- `ordered_activation_transfer`: activation-sized movement from one target to
  another.
- `synthetic_layer_split_small_payload`: first half of the logical payload on
  source, activation transfer, second half on destination.
- `synthetic_tensor_split_small_payload`: split logical payload across two
  targets, broadcast activation, compute shards, collect output.
- `synthetic_layer_split_group_small_payload`: split logical payload across
  three ordered targets with activation movement between stages.
- `synthetic_tensor_split_group_small_payload`: split logical payload across
  three targets, broadcast activation, compute shards, collect output.

GPU-backed specs are emitted before the Vulkan backend exists. Their
measurements use `status: "unmeasured"` until a backend can execute them. CPU
reference compound measurements use the same small payload to keep the
serialized/layer-split/tensor-split semantics executable.

## Target Policy

The schema records exclusions as run policy. It does not encode permanent bans
for CPU, integrated GPU, discrete GPU, AMD, NVIDIA, Intel, or any other target
class.
