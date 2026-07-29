# Stable GPU Addressing Acceptance

Milestone 8 is complete.

## Addressing architecture

NERVE now exposes Vulkan buffer device addresses through the same
discover-enable-validate contract used for every other shader capability:

- physical-device discovery queries
  `PhysicalDeviceBufferDeviceAddressFeatures`;
- logical-device creation enables the feature only when reported;
- addressable buffers use both `SHADER_DEVICE_ADDRESS` and
  `MemoryAllocateFlags::DEVICE_ADDRESS`;
- SPIR-V capability 5347 (`PhysicalStorageBufferAddresses`) maps to the typed
  `buffer_device_address` feature and fails closed on unsupported devices; and
- every addressable allocation must receive a nonzero Vulkan device address.

The immutable resource arena is demand-created rather than sized to the model.
It:

- creates device-local addressable chunks only when an allocation cannot fit
  an existing free range;
- enforces a deterministic committed-byte ceiling before allocation;
- aligns the absolute GPU address, not merely the buffer-relative offset;
- uses best-fit placement with split and coalesced free ranges;
- keeps each address and allocation alive for its full published lifetime; and
- releases an empty chunk immediately, returning its committed capacity rather
  than retaining a hidden high-water allocation.

Capacity failure is transactional. It neither creates a resource publication
nor changes committed, allocated, chunk, or allocation counters.

## GPU-visible table and lifetime

Compiled resource ordinals map to fixed 32-byte GPU table records:

| Offset | Field | Type |
| ---: | --- | --- |
| 0 | device address | `u64` |
| 8 | byte count | `u64` |
| 16 | generation | `u64` |
| 24 | resident | `u32` |
| 28 | reserved | `u32` |

Publication and clearing are generation-checked atomic group operations.
Duplicate slots, out-of-range slots, already-resident slots, stale clears, and
cross-device addresses fail before GPU mutation.

Bulk resource data remains on the asynchronous transfer queue. The small table
commit is copied on the compute queue after earlier readers and before later
readers, with exact-range Vulkan synchronization. This makes a group
observationally atomic without draining the device or serializing the bulk
upload path.

The table retains a strong reference to every published allocation. Dropping a
graph/loader reference therefore cannot make a live GPU address dangle. A
generation-matched clear completes on the compute queue before the table
releases that reference. If synchronization fails after Vulkan accepted a
submission, the allocation remains retained until table teardown.

`upload_loaded_compiled_resource_group_to_stable_address_space` connects the
verified compiled-resource backing store to the arena and table. It packs each
resource's ranges into one stable allocation, performs one bulk timeline
upload, constructs the complete typed resident group, and only then publishes
all address slots. Explicit retirement clears the group before releasing its
allocations.

The mechanism contains no Qwen, MoE, expert, layer, or model-family branch.

## Direct versus indirect addressing

NERVE retains two materially different paths:

- direct descriptor binding for monolithic and always-resident resources; and
- stable buffer-device-address table lookup for independently resident
  resources.

The final release-mode microbenchmark used 1 MiB of identical data and
identical arithmetic, one warmup per path, and two measured calls per path:

- direct durations: 7,480 ns and 7,560 ns;
- address-table durations: 8,240 ns and 8,120 ns;
- direct mean: 7,520 ns;
- table mean: 8,180 ns; and
- table/direct ratio: 1.087766.

The table lookup was 8.78% slower in this deliberately lookup-sensitive
microbenchmark. It is therefore not used ceremonially where direct binding is
possible. Its value is stable dynamic resolution for demand-resident
resources; the faster direct representation remains available to compiled
always-resident implementations.

The entire microbenchmark, including shader compilation, allocation, uploads,
six timed dispatches, byte-for-byte output verification, clearing, and
teardown, completed in 0.18 seconds.

## Sequential acceptance

The following exact tests passed individually with `CARGO_BUILD_JOBS=1` and
`-- --exact --test-threads=1`:

- `stable_resource_address_contract_validates_alignment_and_layout`
- `stable_resource_free_ranges_coalesce_on_both_sides`
- `spirv_contract_extracts_buffer_device_address_feature`

The following release-mode tests passed individually using RADV and one
verified-idle AMD GPU:

- `stable_resource_address_space_is_visible_stable_and_transactional`
  - proves demand allocation, absolute alignment, stable addresses, bounded
    capacity, transactional publication errors, shader-visible data,
    generation-safe clearing, table-retained lifetimes, reuse, and full arena
    teardown;
- `external_compiled_group_uses_stable_address_slots_and_explicit_retirement`
  - resolves and verifies a real Qwen3.6-35B-A3B compiled resource group,
    uploads it through the generic stable path, checks every resource slot, and
    returns arena accounting to zero; and
- `resident_transfer_stream_bounds_staging_and_completes_with_a_timeline`
  - confirms the pre-existing bounded asynchronous transfer behavior remains
    correct after cross-queue visibility was made explicit.

`cargo fmt --check`, `cargo check --all-targets --features "vulkan tokenizers"`,
and `git diff --check` passed. All source files touched by this milestone remain
below the repository's 2,000-line review threshold and retain a single concern.

## Matched conversation benchmark

The canonical two-AMD, thinking-enabled Qwen3.6-35B-A3B FP8 conversation used a
131,072-token context, 65,536 maximum new tokens, two MTP draft tokens, seed 0,
the discarded `hi` warmup, and all five measured turns in one continuously
resident process.

Results:

- mean decode: **41.4644 tok/s**
- mean prefill: **66.3466 tok/s**
- warmup decode: 41.282 tok/s
- milestone 7 mean decode: 41.9426 tok/s
- decode delta: -1.14% (not material)
- throughput floor: passed (30 tok/s)
- quality gate: passed, including thinking, Qwen identity, Athens, a relevant
  qualified Corinth answer, the knowledge-cutoff response, and cross-turn
  Greece recall
- report:
  `/tmp/nerve-m8-stable-addressing-mtp2-v1/report.json`
- transcript:
  `/tmp/nerve-m8-stable-addressing-mtp2-v1/conversation-seed-0.log`
- report SHA-256:
  `bf6d0c954aa3c29748ca14cf5fbeed7dd5d5223bf32005ade65c6894b96a072d`
- transcript SHA-256:
  `02e9b371f9525ef7f1b0f9ec9038974df5b68aee0ca897f9155b4c3d553a347c`

After every GPU test and the full model benchmark, both AMD GPUs returned to
their exact pre-workload baselines: 59,973,632 bytes VRAM used and 0% busy.
NVIDIA was neither enumerated nor used.
