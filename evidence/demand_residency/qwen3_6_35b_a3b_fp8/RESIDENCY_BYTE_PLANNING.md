# Initial, Current, and Maximum Residency Planning Evidence

## Acceptance result

Milestone 5 is complete.

Runtime residency plan schema v2 separates immutable parameter address space
from the bytes that must physically exist at mount time. It reports, per
physical device:

- always-resident parameter bytes;
- initial dynamic parameter bytes;
- current resident parameter bytes;
- maximum addressable parameter bytes;
- reusable staging headroom;
- transient state;
- activation and workspace headroom; and
- the complete initial device-resident requirement.

`demand_retained` and `eager` are explicit runtime policies. A
demand-retained plan begins with zero dynamic parameter bytes. An eager plan
begins with its entire dynamic address space resident. The optimization target
admits against `initial_device_resident_bytes`, not maximum addressable bytes.
There is no v1 compatibility path.

Every later atomic growth can be checked before I/O or allocation through the
same plan. The check rejects an invalid current state, a group larger than
staging headroom, growth beyond the compiled maximum, arithmetic overflow, or
a projected device working set beyond the safe capacity.

## Real one-device admission proof

The fresh Qwen3.6-35B-A3B FP8 package was planned for one AMD Radeon AI PRO
R9700, with 131,072 context activations, MTP mounted, and strict
demand-retained policy. Planning opened no Vulkan device.

| Category | Bytes |
| --- | ---: |
| Physical VRAM capacity | 34,208,743,424 |
| Always-resident parameters | 3,540,055,424 |
| Initial dynamic parameters | 0 |
| Current resident parameters at mount | 3,540,055,424 |
| Maximum addressable parameters | 36,561,646,976 |
| Staging headroom | 3,146,112 |
| Transient state | 3,147,570,984 |
| Activation/workspace headroom | 10,705,992 |
| Complete initial device requirement | 6,701,478,512 |

The 36.56 GB parameter address space is 2.35 GB larger than the GPU's physical
VRAM, but its complete 6.70 GB initial requirement fits. The planner therefore
admits the model without allocating storage proportional to its maximum
address space.

For a two-device whole-component placement, the same package planned:

| Device | Initial device bytes | Current parameters | Maximum parameters | Staging |
| --- | ---: | ---: | ---: | ---: |
| AMD `0000:07:00.0` | 3,190,389,908 | 1,745,893,760 | 17,853,987,200 | 3,146,112 |
| AMD `0000:0a:00.0` | 3,514,238,832 | 1,794,161,664 | 18,707,659,776 | 3,146,112 |

Disabling MTP reduced current parameters from 3,540,055,424 to
3,491,791,616 bytes and maximum addressable parameters from 36,561,646,976 to
35,707,978,496 bytes. Draft resources therefore enter the plan only when the
runtime mounts their execution graph.

## Adversarial verification

Sequential tests prove:

- demand mount admission succeeds when maximum parameters exceed safe device
  capacity but the complete initial set fits;
- eager planning includes all dynamic parameters at mount;
- dynamic maximum bytes do not inflate the initial allocation;
- compatible reused resources are counted once per physical device;
- staging is sized to the largest selected atomic group, not the whole model;
- context growth changes transient state and activation headroom independently
  from parameters;
- malformed category sums, policy invariants, working-set sums, device totals,
  and global totals fail closed;
- unplanned internal sharding fails closed; and
- atomic growth is rejected before load when staging, maximum-address, or
  device-capacity constraints would be violated.

Verification results:

- Python runtime-target and plan-contract suites: 21 passed.
- Rust physical-layout, internal-sharding, demand/eager, and growth-admission
  tests: 3 individually selected tests passed.
- Real v2 planner execution against the fresh 35B package: passed for one and
  two devices, with MTP enabled and disabled.
- Release runtime and planner builds, Rust formatting, Python lint, and
  `git diff --check`: passed.

## Inference impact

Planning is outside token execution, so this milestone should not change
throughput or model behavior. The final canonical, thinking-enabled,
warmup-discarded conversation measured:

- mean decode: **41.129 tok/s**;
- mean prefill: **68.359 tok/s**;
- warmup decode: 45.477 tok/s; and
- measured-turn decode: 44.045, 41.710, 39.337, 36.182, and 44.370 tok/s.

The 41.129 tok/s result is within 0.9% of the preceding 41.510 tok/s sample and
remains above the 30 tok/s floor. An earlier confirmation sample measured
39.179 tok/s; repeating from the final source state ruled out a systematic
regression. Both samples passed the quality gate.

Final benchmark report SHA-256:

`7f6eb4e76337d7448e7b2995ad4633f780ea3f76e2bf6d4a1a3d52400d290d3e`

Final transcript SHA-256:

`25091eb2d41c3fcc4569b31fdb1658bb7cfb5d18e8ddc8cc2254381a862ccf0e`

Both AMD devices returned to their exact idle baseline after each run:
59,973,632 bytes of VRAM in use and 0% busy.
