#[test]
fn speculative_catch_up_shifts_verified_hidden_frames_by_one_lane() {
    let frame_byte_capacity = 8_192;

    assert_eq!(
        speculative_catch_up_preceding_target_bytes(1, frame_byte_capacity).unwrap(),
        0
    );
    assert_eq!(
        speculative_catch_up_preceding_target_bytes(2, frame_byte_capacity).unwrap(),
        frame_byte_capacity
    );
    assert_eq!(
        speculative_catch_up_preceding_target_bytes(3, frame_byte_capacity).unwrap(),
        2 * frame_byte_capacity
    );
}

#[test]
fn speculative_catch_up_rejects_zero_width_and_byte_overflow() {
    assert!(speculative_catch_up_preceding_target_bytes(0, 8_192).is_err());
    assert!(speculative_catch_up_preceding_target_bytes(3, usize::MAX).is_err());
}

#[test]
fn speculative_catch_up_uses_one_canonical_target_window_capacity() {
    assert_eq!(speculative_catch_up_lane_capacity(1).unwrap(), 2);
    assert_eq!(speculative_catch_up_lane_capacity(2).unwrap(), 4);
    assert_eq!(speculative_catch_up_lane_capacity(7).unwrap(), 8);
    assert_eq!(speculative_catch_up_lane_capacity(31).unwrap(), 32);
    assert_eq!(speculative_catch_up_lane_capacity(63).unwrap(), 64);
    assert!(speculative_catch_up_lane_capacity(64).is_err());
    assert!(speculative_catch_up_lane_capacity(usize::MAX).is_err());
}

#[test]
fn speculative_catch_up_source_identity_covers_device_buffer_and_frame_geometry() {
    let baseline = VulkanResidentSpeculativeCatchUpSourceIdentity {
        device_handle: 11,
        buffer_handle: 17,
        frame_byte_capacity: 8_192,
    };
    assert_ne!(
        baseline,
        VulkanResidentSpeculativeCatchUpSourceIdentity {
            device_handle: 12,
            ..baseline
        }
    );
    assert_ne!(
        baseline,
        VulkanResidentSpeculativeCatchUpSourceIdentity {
            buffer_handle: 18,
            ..baseline
        }
    );
    assert_ne!(
        baseline,
        VulkanResidentSpeculativeCatchUpSourceIdentity {
            frame_byte_capacity: 16_384,
            ..baseline
        }
    );
    assert!(baseline.binds_command_identity((11, 17)));
    assert!(!baseline.binds_command_identity((12, 17)));
    assert!(!baseline.binds_command_identity((11, 18)));
    assert!(
        VulkanResidentSpeculativeCatchUpSourceIdentity {
            frame_byte_capacity: 16_384,
            ..baseline
        }
        .binds_command_identity((11, 17)),
        "source invalidation must not depend on the old frame geometry",
    );
}
