# Repository rules

## Test execution

- Never run tests in parallel in this repository. Reliability is more important than speed.
- Every test command must explicitly select sequential execution, even when the test runner is sequential by default.
- Rust tests must use `-- --test-threads=1` (plus `--exact` when targeting a specific test).
- Do not run a broad test filter that can initialize Vulkan. Vulkan tests must be selected individually and run sequentially.
- If a test runner cannot guarantee sequential execution, do not run it until a safe sequential invocation is established.

## GPU residency

- Do not use the NVIDIA GPU for any NERVE workload. This includes model execution, tests, benchmarks, compilation probes, device enumeration, and diagnostic probes.
- Do not use the AMD integrated GPU at PCI `0000:8a:00.0` for any NERVE workload. Only explicitly allowlisted discrete AMD GPUs may be used for execution, tests, benchmarks, compilation probes, device enumeration, or diagnostics.
- Inspect every selected AMD GPU immediately before a workload and record its
  current VRAM allocation, usable capacity, and activity. Existing allocations
  are reservations, not a reason to discard the device.
- NERVE may share an AMD GPU with an existing workload when the measured safe
  remaining capacity is sufficient. It must preserve the existing allocation,
  stay within the unreserved budget, and must not unload or disrupt unrelated
  work merely to obtain an idle device.
- Placement must consider all compatible AMD GPUs and distribute work according
  to their measured remaining capacity and communication cost. Do not hardcode
  a device count or require an idle baseline.
- After a workload, verify that NERVE released the capacity it acquired and that
  pre-existing allocations remain present. Compare against the recorded
  pre-workload reservation, not against zero or an idle baseline.

## Multi-GPU placement

- Never use tensor parallelism on this workstation. Its GPU interconnect does not have enough lanes for tensor-parallel execution to be a useful or fair configuration.
- Multi-GPU NERVE workloads must use contiguous component/layer placement unless the user explicitly requests another non-tensor-parallel wiring.
- Performance comparisons with another inference engine must use equivalent execution settings and placement strategies. Do not compare a tensor-parallel run with a layer/model-parallel run.
