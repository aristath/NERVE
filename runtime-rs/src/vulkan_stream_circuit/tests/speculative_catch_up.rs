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
