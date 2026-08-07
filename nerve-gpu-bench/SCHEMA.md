# JSON Schema Notes

The benchmark output schema is currently identified as
`nerve.gpu_benchmark_run.v1`.
Dry-plan output from `run --dry-plan` is identified as
`nerve.gpu_benchmark_plan.v1`.

This document is descriptive rather than a formal JSON Schema file. The Rust
types in `src/model.rs` are the source of truth for the first implementation.
Use `nerve-gpu-bench validate --input <path>` to parse and run basic structural
validation on a saved result.
Use `nerve-gpu-bench summarize --input <path>` for a compact status and target
summary of the same saved result.
The summary groups status counts by `(comparison_group, workload_class,
placement_strategy, format)` so missing one-target, two-target-serial, or
two-target-parallel evidence is visible for each operation/format path.
Coverage warnings are emitted when a selected run has no completed measurement
for an expected placement strategy on a requested workload/format axis. It also
resolves every `comparison_sets[].candidates[]` entry against the recorded
measurements and reports whether that candidate is completed, unmeasured,
missing, failed, unsupported, or skipped. Completed candidates also expose the
best matched min/median duration so a planner can rank candidates without
rejoining the full sample list first.

## Top-Level Fields

- `schema`: output document schema name.
- `started_at_unix_ms`, `finished_at_unix_ms`: wall-clock run timestamps.
- `implementation`: benchmark package name, version, and backend status.
- `policy`: user-selected run policy, including payload size, requested
  `benchmark_formats`, requested `benchmark_workloads`, and exclusions.
- `discovered_targets`: every target discovered before policy filtering.
  Targets include `format_capabilities`, which record native, emulated,
  fallback, unsupported, or unmeasured support per format. PCI targets may also
  include `pci_link`, a passive sysfs estimate of current/max link width and
  one-way byte rate. Vulkan targets may include `vulkan`, with physical-device
  properties, memory heaps, queue families, extension names, and probed feature
  flags.
- `selected_target_ids`: targets selected for this run.
- `skipped_targets`: discovered targets skipped by explicit policy.
- `workload_specs`: logical benchmark contracts generated for the selected
  payload size.
- `comparison_sets`: planner-facing candidate sets. For each selected pair, the
  set currently compares first-target-only, second-target-only, serial
  first-to-second, serial second-to-first, and parallel two-target execution.
  When triplets are active, selected triplets compare each single-target option,
  three-target serial, and three-target parallel.
- `measurements`: single-target measurements.
- `pair_measurements`: ordered pair and two-target placement measurements.
  Current Vulkan execution can emit ordered activation transfer, ordered serial
  pair, and parallel pair rows when `policy.execute` is true.
- `group_measurements`: three-target placement measurements when enough targets
  are selected and `--max-group-size 3` is active. Current Vulkan execution can
  emit ordered three-stage serial and three-target parallel rows when
  `policy.execute` is true.
- `diagnostics`: non-fatal implementation notes.

## Dry Plan Fields

`run --dry-plan` emits a JSON plan instead of measurements. It uses the same
policy and target selection path, then reports selected targets, skipped
targets, requested format/workload counts, estimated single/pair/group
measurement counts, estimated comparison set count, total estimated measurement
count, and the maximum payload bytes per measurement. It does not execute CPU
or GPU measurements.

`policy.execute` records whether the run was allowed to open Vulkan
logical devices. When true, selected Vulkan targets may attempt the execution
boundary. Current Vulkan single-target measurements can complete
`dense_projection`, `moe_expert`, and `router_reduction` paths with timestamped
compute samples for F32, feature-gated native F16, and packed-emulated
lower-precision or quantized storage formats. Raw INT8 uses native 8-bit integer
arithmetic when `shaderInt8` is available; block-quantized Q8 remains a separate
format path. Current Vulkan pair measurements can complete ordered host-staged
activation transfer, two-target serial, and two-target parallel rows for those
executable workload/format axes. Current Vulkan triplet measurements can
complete three-target serial and three-target parallel rows for the same
executable axes. Other Vulkan workload or format paths are omitted from
measurements until their kernels are implemented. Comparison candidate summaries
report those absent records as `missing`.

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
Comparison sets are also workload- and format-specific. The same target pair can
have a different answer for dense projection, MoE expert, router reduction, F16,
BF16, FP8 variants, FP4/MXFP4/NVFP4, INT formats, GGUF Q/IQ formats, or F32.

Current workload spec families:

- `single_target_small_payload:<workload_class>:<format>`: full logical
  payload on one target.
- `cpu_reference_serialized_small_payload`: CPU reference execution of the full
  logical payload as one serialized pattern.
- `cpu_reference_layer_split_small_payload`: CPU reference execution of the same
  logical payload using the layer-split dataflow shape.
- `cpu_reference_tensor_split_small_payload`: CPU reference execution of the
  same logical payload using the tensor-split dataflow shape.
- `ordered_activation_transfer:<workload_class>:<format>`: activation-sized
  movement from one target to another.
- `synthetic_layer_split_small_payload:<workload_class>:<format>`: first half of
  the logical payload on source, activation transfer, second half on
  destination.
- `synthetic_tensor_split_small_payload:<workload_class>:<format>`: split
  logical payload across two targets, broadcast activation, compute shards,
  collect output.
- `synthetic_layer_split_group_small_payload:<workload_class>:<format>`: split
  logical payload across three ordered targets with activation movement between
  stages.
- `synthetic_tensor_split_group_small_payload:<workload_class>:<format>`: split
  logical payload across three targets, broadcast activation, compute shards,
  collect output.

Device-backed specs are emitted even before every backend path exists. Vulkan
single-target, two-target pair, and three-target group measurements can complete
for executable workload/format axes when `--execute` is active. Other
device-backed specs have no measurement row until a backend can execute them.
CPU reference compound measurements use the same small payload to keep the
serialized/layer-split/tensor-split semantics executable.
CPU F32 single-target measurements use
`single_target_small_payload:<workload_class>:f32` so they can resolve the same
candidate family as GPU or accelerator targets. CPU formats that are not
implemented yet are emitted as `status: "unsupported"`.

## Target Policy

The schema records exclusions as run policy. It does not encode permanent bans
for CPU, integrated GPU, discrete GPU, AMD, NVIDIA, Intel, or any other target
class.

Vulkan targets are detected targets. Their stable IDs use
`vulkan:pci:<address>` when `VK_EXT_pci_bus_info` is available. Otherwise they
include the physical-device index because Vulkan device order is driver-defined.
Discovery creates a Vulkan instance only; it does not create logical devices or
run GPU workloads.
F16 capability uses the shaderFloat16 feature bit. BF16 and FP8 variant
capabilities use extension presence as conservative native-path probes. Raw INT8
capability uses the shaderInt8 feature bit. FP4, MXFP4, NVFP4, GGUF Q-family,
and IQ-family formats can be marked
`emulated` when the generic packed-u32 baseline can run, but native
format-specific math/dequant kernels beyond F16 and raw INT8 still require
separate measurements.

Format capability is not a speed claim. A target can lack native FP4/MXFP4 but
still win F16, BF16, FP8, routing, logits, sampler, or fallback/dequantized
work. Placement must use measurements, not support flags alone.

Workload class is also not a capability claim. It is the benchmark axis that
keeps dense projection, MoE expert, router/reduction, and future operation
families from being merged into one misleading score.

`pci_link.current_one_way_bytes_per_second` and
`pci_link.max_one_way_bytes_per_second` are passive estimates from link
speed/width. They are useful priors for candidate ordering, but measured pair
activation transfer must override them.
