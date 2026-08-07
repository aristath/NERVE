# nerve-gpu-bench

`nerve-gpu-bench` is a standalone Rust package for small, placement-oriented
hardware benchmarks.

The package is intentionally independent from `nerve-runtime`. Its output is a
versioned JSON document that can later be consumed by the inference engine
without coupling the benchmark implementation to runtime placement code.

## Current Scope

This first implementation provides:

- CPU and PCI accelerator target discovery from the host;
- optional Vulkan physical-device discovery behind the `vulkan` feature;
- explicit include/exclude run policy;
- no hard-coded global bans for CPU, integrated GPU, discrete GPU, AMD, NVIDIA,
  Intel, or other target classes;
- small CPU-only synthetic measurements using a default 5 MiB payload,
  including serialized, layer-split, and tensor-split reference patterns;
- requested CPU F32 single-target workload measurements for dense projection,
  MoE expert, and router/reduction classes, with unsupported records for
  unimplemented CPU formats;
- placeholder `unmeasured` GPU and pair records for the upcoming Vulkan backend;
  group placeholders up to triplets;
  and
- JSON output with discovered targets, selected targets, skipped targets,
  measurements, pair/group candidates, and provenance.

The package does not initialize Vulkan yet. GPU targets are discovered and
reported, but GPU compute, peer transfer, tensor-split, and layer-split GPU
measurements are deliberately marked `unmeasured` until backend kernels exist.
CPU reference versions of the small compound patterns execute today so the
logical workload contracts are testable before GPU backend work starts.
For CPU single-target comparisons, requested F32 workload records already use
the same workload IDs as device-backed single-target candidates.

With `--features vulkan`, discovery creates a Vulkan instance and enumerates
physical devices. It records device name/type, API and driver versions, memory
heaps, queue families, advertised device extensions, and conservative format
capabilities. F16 support is classified from the Vulkan shaderFloat16 feature
bit, and INT4 is marked as an emulated path when shaderInt8 is available.
BF16/FP8 are still extension-presence probes until their feature bits are
queried. When `VK_EXT_pci_bus_info` is available, Vulkan targets use the PCI
address in their stable target ID. It does not create logical devices, allocate
GPU memory, submit queues, or run kernels.

The benchmark schema treats placement strategy as first-class data. The initial
small-payload comparison records distinguish one-target serialized execution,
two-target serialized execution, and two-target parallel execution. A future
planner should compare those alternatives within the same comparison group,
instead of assuming any two-target placement is automatically better or worse.
Each selected pair also gets a `comparison_sets` entry that lists the concrete
candidates: left-only, right-only, left-to-right serial, right-to-left serial,
and pair-parallel. Selected triplets also get comparison sets with each
single-target candidate, three-stage serial, and three-target parallel
candidates.

Targets also report format capabilities separately from measurements. Capability
flags are only filters; they do not imply a target is fast for that format.
The run policy records requested benchmark formats, and backend workload IDs
are format-specific so F32, BF16, FP8, INT4, and FP4 evidence cannot collapse
into one generic path.

The workload matrix is separate from the format matrix. Defaults include dense
projection, MoE expert, and router/reduction shapes because one device may win
F32 dense math while losing FP8 expert work or low-byte routing. Backend
workload IDs include both workload class and format.

## Commands

List discovered targets:

```sh
cargo run --manifest-path nerve-gpu-bench/Cargo.toml -- list --json
```

List Vulkan physical devices as targets:

```sh
cargo run --manifest-path nerve-gpu-bench/Cargo.toml --features vulkan -- list --json
```

The JSON target list includes `pci_link` when sysfs exposes PCIe speed/width.
Those values include parsed current/max width and an estimated one-way byte rate
for placement priors. They are not a peer-transfer benchmark.

Run small benchmarks and write JSON:

```sh
cargo run --manifest-path nerve-gpu-bench/Cargo.toml -- run \
  --output benchmark-results.json
```

Preview the selected benchmark matrix without executing measurements:

```sh
cargo run --manifest-path nerve-gpu-bench/Cargo.toml -- run \
  --dry-plan \
  --output benchmark-plan.json
```

Limit the requested format matrix:

```sh
cargo run --manifest-path nerve-gpu-bench/Cargo.toml -- run \
  --format f32 \
  --format fp4 \
  --output benchmark-results.json
```

Limit the requested workload matrix:

```sh
cargo run --manifest-path nerve-gpu-bench/Cargo.toml -- run \
  --workload dense_projection \
  --workload router_reduction \
  --output benchmark-results.json
```

Validate a saved result:

```sh
cargo run --manifest-path nerve-gpu-bench/Cargo.toml -- validate \
  --input benchmark-results.json
```

Summarize a saved result:

```sh
cargo run --manifest-path nerve-gpu-bench/Cargo.toml -- summarize \
  --input benchmark-results.json
```

The summary includes placement-strategy coverage so a result can be checked for
the key alternatives: one-target serialized, two-target serialized, and
two-target parallel. It also reports coverage warnings when an expected
strategy has no completed measurement. Comparison candidates are resolved
against concrete measurements, so missing A-only, B-only, A-to-B serial,
B-to-A serial, or pair-parallel evidence is visible directly.

Select or exclude targets:

```sh
cargo run --manifest-path nerve-gpu-bench/Cargo.toml -- run \
  --include-target cpu:host \
  --exclude-pci 0000:8a:00.0 \
  --exclude-kind integrated_gpu \
  --output benchmark-results.json
```

The default payload is `5242880` bytes. Override it with
`--payload-bytes <bytes>` for shorter or larger synthetic workloads.
Two-target and three-target synthetic placement records are emitted by default
when enough selected targets exist. Use `--max-group-size 1`, `2`, or `3` to
bound those records, and `--no-pairs` to skip pair and group records entirely.

See `SCHEMA.md` for the current JSON shape and workload contract.
