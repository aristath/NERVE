# nerve-gpu-bench

`nerve-gpu-bench` is a standalone Rust/Vulkan comparison benchmark for NERVE's
placement planner. It generates a small deterministic dense projection and does
not read or download models.

The output answers two questions for every supported weight format:

1. Which single target or tensor-parallel target group runs one equivalent
   projection fastest?
2. Which target runs two equivalent stages fastest, and how do directed serial
   pairs such as `A -> B` and `B -> A` compare with that baseline?
3. When a model region must span a particular target set, is an equivalent
   serialized split or tensor-parallel split faster?

The default logical parameter budget is 5 MiB. Parameters remain resident while
the runner performs one untimed warmup and one timed execution. The artifact
retains the measured median duration for direct comparison; it is not a hardware
characterization report.

## Comparison Matrix

For every selected Vulkan target and format, the runner measures:

- one projection on that target; and
- an equivalent two-stage projection chain on that target.

For every directed pair, it measures the two-stage serial path including both
computations, synchronization, and the activation transfer between targets.

For every viable target subset from two through the selected maximum group
size, it measures real tensor-parallel execution. Parameters are sharded, every
participant computes disjoint output rows, and the timed path includes shared
activation/output access and synchronization. Each group member is tried as the
shared-buffer owner; only the fastest valid owner is retained in the final
ranking.

The forced-split comparison uses the same total parameters and logical stages
on both sides. Serialization places one equal parameter share on each target
and includes every inter-target activation boundary. TP shards every stage
across the same targets and includes its synchronization and shared-output
work. Pair comparisons measure both serial directions. Larger target sets are
expanded one participant at a time from the pair stage. The directed pair
measurements select the lowest-cost serial order, which is then executed end to
end; every TP owner is executed, and the fastest valid serial and TP paths are
compared in the final `combinations` section.

The group limit defaults to all selected targets. `--max-group-size` is an
optional workload bound, not an architectural device-count ceiling.

The runner tries the executable Vulkan sharing routes and validates output
before accepting a TP result. A route that cannot produce valid cross-device
output is absent from the ranking.

## Formats

The default formats are `f16`, `bf16`, `fp8_e4m3`, `fp8_e5m2`, `fp4`,
`mxfp4`, `int8`, `int4`, `q8_0`, and `f32`. Native format instructions are used
when every participant supports them; otherwise the same packed storage format
is decoded by the fallback shader.

## Commands

List discovered targets:

```sh
cargo run --manifest-path nerve-gpu-bench/Cargo.toml -- list --json
```

Preview the comparison count without executing workloads:

```sh
cargo run --manifest-path nerve-gpu-bench/Cargo.toml -- run --dry-plan
```

Run the benchmark:

```sh
cargo run --release --manifest-path nerve-gpu-bench/Cargo.toml -- run \
  --execute \
  --output nerve-gpu-bench/placement-benchmark.json
```

All discovered Vulkan targets are selected by default. Repeated selection
options can narrow the run:

```sh
cargo run --release --manifest-path nerve-gpu-bench/Cargo.toml -- run \
  --execute \
  --include-target vulkan-uuid:00112233445566778899aabbccddeeff \
  --include-target vulkan-uuid:ffeeddccbbaa99887766554433221100 \
  --exclude-kind integrated_gpu \
  --max-group-size 2 \
  --output nerve-gpu-bench/placement-benchmark.json
```

`--format` narrows the format set. `--payload-bytes` and `--samples` override
the small defaults. `--max-group-size` accepts any positive number and is
clamped to the number of selected targets.

Vulkan targets use the same `vulkan-uuid:<32 lowercase hex digits>` identity as
the NERVE runtime. PCI addresses remain topology metadata and can be selected
through `--exclude-pci`; they are not stable execution identities.

Validate or summarize an artifact:

```sh
cargo run --manifest-path nerve-gpu-bench/Cargo.toml -- validate \
  --input nerve-gpu-bench/placement-benchmark.json
cargo run --manifest-path nerve-gpu-bench/Cargo.toml -- summarize \
  --input nerve-gpu-bench/placement-benchmark.json
```

The final JSON contains completed, validated comparisons only. A missing target
or combination was not a usable measured path.
