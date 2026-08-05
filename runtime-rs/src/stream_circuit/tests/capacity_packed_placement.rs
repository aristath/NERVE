fn weighted_component(id: &str, bytes: usize) -> CapacityPackedPlacementComponent {
    CapacityPackedPlacementComponent {
        component_id: id.to_string(),
        resident_weight_bytes: bytes,
    }
}

fn placement_device(id: &str, bytes: usize) -> CapacityPackedPlacementDevice {
    CapacityPackedPlacementDevice {
        device_id: id.to_string(),
        capacity_bytes: bytes,
    }
}

#[test]
fn capacity_packed_placement_fills_devices_in_caller_preference_order() {
    let placement = capacity_packed_component_placement(
        &[
            weighted_component("a", 40),
            weighted_component("b", 40),
            weighted_component("c", 20),
            weighted_component("d", 40),
        ],
        &[
            placement_device("preferred", 100),
            placement_device("spill", 100),
        ],
    )
    .unwrap();

    assert_eq!(placement["a"], "preferred");
    assert_eq!(placement["b"], "preferred");
    assert_eq!(placement["c"], "preferred");
    assert_eq!(placement["d"], "spill");
}

#[test]
fn capacity_packed_placement_respects_partial_remaining_capacity() {
    let placement = capacity_packed_component_placement(
        &[
            weighted_component("a", 40),
            weighted_component("b", 40),
            weighted_component("c", 40),
        ],
        &[
            placement_device("partially_reserved", 85),
            placement_device("next", 80),
        ],
    )
    .unwrap();

    assert_eq!(placement["a"], "partially_reserved");
    assert_eq!(placement["b"], "partially_reserved");
    assert_eq!(placement["c"], "next");
}

#[test]
fn capacity_packed_placement_rejects_an_oversized_contiguous_tail() {
    let error = capacity_packed_component_placement(
        &[
            weighted_component("a", 60),
            weighted_component("b", 60),
            weighted_component("c", 60),
        ],
        &[
            placement_device("first", 100),
            placement_device("second", 100),
        ],
    )
    .unwrap_err();

    assert!(error.0.contains("final segment requires 120 bytes"));
}

#[test]
fn capacity_packed_placement_does_not_sort_device_names() {
    let placement = capacity_packed_component_placement(
        &[weighted_component("a", 50), weighted_component("b", 50)],
        &[
            placement_device("z-fast", 50),
            placement_device("a-slow", 50),
        ],
    )
    .unwrap();

    assert_eq!(placement["a"], "z-fast");
    assert_eq!(placement["b"], "a-slow");
}
