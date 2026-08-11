# Placement JSON

`run --execute` writes a compact, pretty-printed placement ranking:

```json
{
  "schema": "nerve.placement_bench",
  "payload_bytes": 5242880,
  "formats": {
    "fp8_e4m3": {
      "placements": [
        {
          "mode": "single",
          "targets": ["vulkan:pci:0000:03:00.0"],
          "duration_ns": 382941
        },
        {
          "mode": "tp",
          "targets": [
            "vulkan:pci:0000:03:00.0",
            "vulkan:pci:0000:07:00.0"
          ],
          "duration_ns": 451870,
          "owner": "vulkan:pci:0000:03:00.0",
          "transport": "shared_host"
        }
      ],
      "serial": [
        {
          "mode": "single",
          "targets": ["vulkan:pci:0000:03:00.0"],
          "duration_ns": 718522
        },
        {
          "mode": "serial",
          "targets": [
            "vulkan:pci:0000:03:00.0",
            "vulkan:pci:0000:07:00.0"
          ],
          "duration_ns": 912523,
          "transport": "external_device_local"
        }
      ]
    }
  },
  "combinations": {
    "fp8_e4m3": [
      {
        "targets": [
          "vulkan:pci:0000:03:00.0",
          "vulkan:pci:0000:07:00.0"
        ],
        "split": [1, 1],
        "serialized": {
          "duration_ns": 491205,
          "order": [
            "vulkan:pci:0000:03:00.0",
            "vulkan:pci:0000:07:00.0"
          ],
          "transport": "external_device_local"
        },
        "tp": {
          "duration_ns": 382941,
          "owner": "vulkan:pci:0000:03:00.0",
          "transport": "shared_host"
        }
      }
    ]
  }
}
```

`placements` ranks equivalent one-projection candidates:

- `single` has exactly one target;
- `tp` has two or more targets, sorted by stable ID;
- `owner` identifies the measured TP member that owned shared activation and
  output memory; and
- `transport` identifies the valid measured sharing route.

`serial` ranks equivalent two-stage candidates:

- `single` is a two-stage chain on one target; and
- `serial` contains two targets in execution order, so `A -> B` and `B -> A`
  are distinct candidates.

Within each list, candidates are ordered by measured median `duration_ns`.
Smaller is faster. The measured duration is retained without normalization or
threshold-based grouping.

`combinations` answers the forced-split question for each viable target set:

- `split` is the relative parameter residency on the sorted targets; `[1, 1]`,
  `[1, 1, 1]`, and `[1, 1, 1, 1]` are balanced splits;
- `serialized` is the fastest measured complete serial chain and preserves its
  execution `order`;
- `tp` is the fastest measured equal-work TP chain and records its owner; and
- both paths retain their measured median `duration_ns`, allowing paths from
  different target sets to be compared directly.

Both paths use the same total parameter budget, stage count, activation/output
shape, and operation count. A combination is omitted unless both complete,
output-validated paths exist.

Only completed and output-validated measurements appear. The formats and target
combinations present in the rankings are the measured usable set; no separate
inventory, missing-results list, or raw timing telemetry is included.
