# nerve-gpu-bench

`nerve-gpu-bench` is a standalone Rust package for small, placement-oriented
hardware benchmarks.

The package is intentionally independent from `nerve-runtime`. Its output is a
versioned JSON document that can later be consumed by the inference engine
without coupling the benchmark implementation to runtime placement code.

## Current Scope

This first implementation provides:

- CPU and PCI accelerator target discovery from the host;
- explicit include/exclude run policy;
- no hard-coded global bans for CPU, integrated GPU, discrete GPU, AMD, NVIDIA,
  Intel, or other target classes;
- small CPU-only synthetic measurements using a default 5 MiB payload;
- placeholder `unmeasured` GPU and pair records for the upcoming Vulkan backend;
  group placeholders up to triplets;
  and
- JSON output with discovered targets, selected targets, skipped targets,
  measurements, pair/group candidates, and provenance.

The package does not initialize Vulkan yet. GPU targets are discovered and
reported, but GPU compute, peer transfer, tensor-split, and layer-split
measurements are deliberately marked `unmeasured` until backend kernels exist.

## Commands

List discovered targets:

```sh
cargo run --manifest-path nerve-gpu-bench/Cargo.toml -- list --json
```

Run small benchmarks and write JSON:

```sh
cargo run --manifest-path nerve-gpu-bench/Cargo.toml -- run \
  --output benchmark-results.json
```

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
