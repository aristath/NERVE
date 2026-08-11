# Repository rules

## Test execution

- Never run tests in parallel in this repository. Reliability is more important than speed.
- Every test command must explicitly select sequential execution, even when the test runner is sequential by default.
- Rust tests must use `-- --test-threads=1` (plus `--exact` when targeting a specific test).
- Do not run a broad test filter that can initialize Vulkan. Vulkan tests must be selected individually and run sequentially.
- If a test runner cannot guarantee sequential execution, do not run it until a safe sequential invocation is established.

## GPU residency

- All detected compute targets are eligible for NERVE workloads, including
  integrated GPUs, discrete GPUs from any vendor, and the CPU. A caller may
  explicitly exclude targets for a particular run.
- Inspect every selected target immediately before a workload and record its
  current VRAM allocation, usable capacity, and activity. Existing allocations
  are reservations, not a reason to discard the device.
- NERVE may share a target with an existing workload when the measured safe
  remaining capacity is sufficient. It must preserve the existing allocation,
  stay within the unreserved budget, and must not unload or disrupt unrelated
  work merely to obtain an idle device.
- Placement must consider all compatible targets and distribute work according
  to their measured remaining capacity and communication cost. Do not hardcode
  a device count or require an idle baseline.
- After a workload, verify that NERVE released the capacity it acquired and that
  pre-existing allocations remain present. Compare against the recorded
  pre-workload reservation, not against zero or an idle baseline.

## Multi-GPU placement

- Benchmark single-target, serialized multi-target, and tensor-parallel execution
  with equivalent model work so placement decisions can be based on measured
  end-to-end cost.
- Multi-target benchmarks must include the computation, synchronization,
  transfers, and collectives required by the measured placement strategy.
