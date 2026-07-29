# Per-Device Residency Manager Acceptance

Milestone 7 is complete.

The manager is generic over an immutable device-resource payload. It contains
no Qwen, MoE, expert, layer, or model-family logic. Resolved concrete and
partition groups both enter through the same atomic
`DeviceResourceGroupDescriptor`.

## Ownership and state

- Each manager owns exactly one logical/physical execution-device directory.
- A request returns one of three nonblocking outcomes:
  - an immediate resident lease;
  - the unique load-leader permit; or
  - a follower waiting on the same single-flight operation.
- The first request validates the compiled descriptor, reserves the complete
  group byte count, and performs the `absent -> requested -> loading`
  transitions before exposing a load permit.
- Concurrent requests for the same content identity and descriptor join the
  existing operation and cannot start another physical load.
- A content identity presented with a different resource layout,
  compatibility, or size fails deterministically.
- Publication accepts only a complete group whose resource identities,
  compatibility records, payload sizes, and total bytes match the reservation.
  The whole group changes from loading to resident under one manager lock.
- Corruption, I/O failure, invalid publication, cancellation, and dropped load
  permits release the full reservation and wake every follower.
- A failed load remains failed until an explicit lifecycle reset. Cancellation
  returns the group to absent so it can be requested again.

## Capacity and sharing

- Capacity includes the explicitly supplied always-resident byte floor,
  committed dynamic bytes, and every in-flight atomic reservation.
- A request that cannot fit its complete group fails before creating a loading
  state or performing I/O/allocation.
- Compatible immutable publications are shared across streams and duplicated
  or rewired graph instances.
- Mutable stream state is not stored in the residency manager and remains
  independently owned.
- Owners identify model/device slices. Residency remains retained after
  execution leases end and is released only by explicit owner or device unload.
- Active execution leases are counted per owner. Unloading an owner with a live
  lease fails transactionally even when another owner shares the group.
- Unloading a pending follower prevents that follower from receiving a lease
  without cancelling another owner's physical load.
- Unloading the load leader while a follower remains permits the single
  publication to finish for the follower but refuses a lease to the unloaded
  leader.
- Device unload atomically checks live leases, releases all resident groups,
  cancels all loading groups, clears failures, and restores dynamic/reserved
  accounting to zero.

The directory uses an explicit typed location:
`local { device_id }` today, with a `remote { device_id }` representation
reserved for later distributed execution rather than overloading local state.

## Vulkan bridge

`upload_loaded_compiled_resource_group` converts a verified host group into one
contiguous Vulkan allocation per immutable resource. All member ranges are
packed into one timeline transfer submission. The function waits only for that
transfer timeline, validates the resulting payloads against the reserved
descriptor, and returns a complete group suitable for atomic manager
publication. No partial group is ever visible.

## Sequential acceptance

The following exact library tests passed individually with
`CARGO_BUILD_JOBS=1`, `--features vulkan`, and
`-- --exact --test-threads=1`:

- `per_device_residency_single_flight_shares_one_atomic_publication`
- `per_device_residency_shares_parameters_but_not_mutable_stream_state`
- `per_device_residency_rejects_capacity_before_atomic_loading_and_rolls_back`
- `per_device_residency_cancellation_and_failure_wake_waiters_cleanly`
- `per_device_residency_owner_unload_cancels_only_that_pending_request`
- `per_device_residency_explicit_unload_refuses_live_leases_and_leaks_nothing`
- `per_device_residency_device_unload_releases_resident_and_loading_groups`
- `per_device_residency_directories_never_alias_physical_devices`

The concurrency test launches eight callers against one manager, proves exactly
one load permit and seven single-flight joins, verifies that every lease points
to the same publication, then explicitly unloads all eight owners and observes
one payload destruction.

The real-device test
`external_compiled_group_has_one_device_load_and_explicit_release` also passed
in release mode using RADV and one verified-idle AMD GPU. It:

1. resolves and verifies one real Qwen3.6-35B-A3B FP8 compiled group;
2. obtains the only device load permit;
3. allocates and timeline-uploads the complete group;
4. atomically publishes it;
5. proves a second owner is a resident hit;
6. reads and verifies every packed device range;
7. explicitly unloads both owners; and
8. restores dynamic residency to zero.

The GPU returned to its exact pre-test baseline of 59,973,632 bytes VRAM used
and 0% busy. The second AMD GPU was not used and remained at the same baseline.

## Matched conversation benchmark

The canonical two-AMD, thinking-enabled Qwen3.6-35B-A3B FP8 conversation used a
131,072-token context, 65,536 maximum new tokens, two MTP draft tokens, the
discarded `hi` warmup, and all five measured turns in one continuously resident
process.

Results:

- mean decode: **41.9426 tok/s**
- mean prefill: **90.2450 tok/s**
- milestone 6 mean decode: 40.8372 tok/s
- decode delta: +2.71%
- throughput floor: passed (30 tok/s)
- quality gate: passed, including thinking, Athens, a relevant Corinth answer,
  knowledge-cutoff response, and cross-turn Greece recall
- report:
  `/tmp/nerve-m7-device-residency-mtp2-v3/report.json`
- transcript:
  `/tmp/nerve-m7-device-residency-mtp2-v3/conversation-seed-0.log`
- report SHA-256:
  `645882d22330444c0b3af2c7c15b2472c7dd5fb4fd1dee647f154c1cb636f689`
- transcript SHA-256:
  `fdd2a6f93677800fba792588220e10c2e4caee91b0980f30e02644327beb8c7b`

Both AMD GPUs returned to their exact pre-benchmark baselines:
59,973,632 bytes VRAM used and 0% busy on each device.
