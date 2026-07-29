# Asynchronous Backing-Store Acceptance

Milestone 6 is complete.

The implementation is generic compiled-resource infrastructure. It does not
identify Qwen, MoE architectures, experts, model names, or a particular package
layout. Both concrete atomic groups and derived partition groups resolve to the
same immutable resource-range contract.

## Host path

- `CompiledResourceBackingStore` uses a fixed worker pool and a bounded request
  channel.
- Workers issue position-independent bounded range reads with `pread`
  semantics. No shared file cursor or caller-thread I/O is used.
- Exact duplicate physical ranges are read and verified once per atomic load
  and share one immutable payload.
- Nearby and adjacent ranges in one atomic group are coalesced when they belong
  to the same artifact and fit the configured gap and maximum-read bounds.
- Every logical range is verified independently with its compiled SHA-256
  digest before the group can be returned.
- Host payload retention, logical group size, range count, coalesced read size,
  worker count, and queued request count are all bounded.
- Cancellation, corruption, and I/O failure return deterministic errors. A
  failed group publishes no partial result, its host reservation is released,
  and later work remains usable.

Cross-request single-flight publication is intentionally not owned by the
backing store. It requires a physical-device/resource identity and therefore
belongs to the per-device residency manager in milestone 7. The backing store
does remove duplicate reads inside one requested atomic group.

## Device upload path

- Each Vulkan logical device acquires a distinct same-family transfer queue when
  its selected queue family exposes at least two queues. It safely uses the
  compute queue otherwise.
- `VulkanResidentTransferStream` owns a fixed ring of host-visible,
  persistently mapped staging buffers and reusable command buffers.
- A packed group upload is submitted with `vkQueueSubmit2` and signals a
  monotonic timeline semaphore.
- The completion point can be attached as a narrowly scoped dependency of later
  compute submissions.
- Reusing a saturated staging slot waits only for that slot's previous timeline
  value. Loading never calls `deviceWaitIdle` or drains unrelated queues.
- Staging bytes, staging slots, and therefore outstanding transfers are bounded.
  Teardown waits only for the stream's final timeline value before reclaiming
  its resources.

## Sequential acceptance tests

The following exact library tests passed individually with
`CARGO_BUILD_JOBS=1`, `--features vulkan`, and
`-- --exact --test-threads=1`:

- `isolated_and_adjacent_ranges_are_verified_and_coalesced`
- `duplicate_physical_ranges_share_one_verified_payload`
- `concrete_atomic_groups_use_the_same_verified_backing_store`
- `cancellation_and_queue_bounds_release_workers`
- `retained_payload_and_request_queue_limits_apply_backpressure`
- `corrupt_and_failed_requests_do_not_poison_following_work`
- `workers_execute_requests_asynchronously_with_a_bounded_queue`
- `resolves_and_verifies_only_the_requested_partition`
- `resident_transfer_stream_bounds_staging_and_completes_with_a_timeline`

The transfer test performs three packed uploads through two staging slots,
proves timeline reuse/backpressure, rejects an undersized slot, and verifies
every destination byte. It was run in release mode on a verified-idle AMD GPU
using the RADV ICD only.

## Real compiled-resource measurement

The external acceptance test used partition 0 of template 0 in the compiled
Qwen3.6-35B-A3B FP8 proof package. This is test data, not a runtime special
case.

- group:
  `sha256:d5d1704c3dc10b5eca81bce2f66f1e6918f7b3f607297f56179e8e99945746d8`
- four verified ranges: 256 B, 128 B, 2,097,152 B, and 1,048,576 B
- atomic payload: 3,146,112 B
- cold read: 12.501 ms, 0.234 GiB/s
- timeline upload: 0.792 ms, 3.699 GiB/s
- distinct transfer queue: yes

The test explicitly discarded the relevant page-cache ranges before timing,
loaded and verified the complete group asynchronously, uploaded all four
members as one transfer submission, waited on the transfer timeline, and
read-verified every device destination.

The AMD device returned to its exact pre-test idle baseline:
59,973,632 bytes VRAM used and 0% GPU busy.

## Matched conversation benchmark

The canonical two-AMD, thinking-enabled Qwen3.6-35B-A3B FP8 conversation used:

- 131,072-token context
- 65,536 maximum new tokens
- two MTP draft tokens
- the canonical warmup and five-turn conversation gate
- the first warmup timing discarded
- one continuously resident model

Results:

- mean decode: **40.8372 tok/s**
- mean prefill: **66.5322 tok/s**
- previous milestone mean decode: 41.1288 tok/s
- delta: -0.71%, not a material regression
- quality gate: passed, including thinking and cross-turn recall
- report:
  `/tmp/nerve-m6-async-backing-store-mtp2-v1/report.json`
- transcript:
  `/tmp/nerve-m6-async-backing-store-mtp2-v1/conversation-seed-0.log`
- report SHA-256:
  `cdfbfaa84df2f1675e641777bb899538c6d94828ccfe8963ff8754cf8b3ef50b`
- transcript SHA-256:
  `3dc3566ddb65e2f3d64d3382505d8cdda4b163e0c1c646203b6c0f5fee91bde8`

Both AMD GPUs returned to their exact pre-benchmark idle baselines:
59,973,632 bytes VRAM used and 0% GPU busy on each device.
