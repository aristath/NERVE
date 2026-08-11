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
          "rank": 1,
          "mode": "single",
          "targets": ["vulkan:pci:0000:03:00.0"],
          "relative_cost": 1.0
        },
        {
          "rank": 2,
          "mode": "tp",
          "targets": [
            "vulkan:pci:0000:03:00.0",
            "vulkan:pci:0000:07:00.0"
          ],
          "relative_cost": 1.18,
          "owner": "vulkan:pci:0000:03:00.0",
          "transport": "shared_host"
        }
      ],
      "serial": [
        {
          "rank": 1,
          "mode": "single",
          "targets": ["vulkan:pci:0000:03:00.0"],
          "relative_cost": 1.0
        },
        {
          "rank": 2,
          "mode": "serial",
          "targets": [
            "vulkan:pci:0000:03:00.0",
            "vulkan:pci:0000:07:00.0"
          ],
          "relative_cost": 1.27,
          "transport": "external_device_local"
        }
      ]
    }
  }
}
```

`placements` ranks equivalent one-projection candidates:

- `single` has exactly one target;
- `tp` has two through four targets, sorted by stable ID;
- `owner` identifies the measured TP member that owned shared activation and
  output memory; and
- `transport` identifies the valid measured sharing route.

`serial` ranks equivalent two-stage candidates:

- `single` is a two-stage chain on one target; and
- `serial` contains two targets in execution order, so `A -> B` and `B -> A`
  are distinct candidates.

Within each list, `relative_cost` is the candidate's elapsed cost divided by
the fastest candidate's cost. The fastest value is `1.0`; smaller is better.
Results within one percent share a rank. Ordering within a tied rank remains
deterministic.

Only completed and output-validated measurements appear. The formats and target
combinations present in the rankings are the measured usable set; no separate
inventory, missing-results list, or raw timing telemetry is included.
