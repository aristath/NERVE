#[test]
fn targeted_component_quanta_cover_decode_work_exactly() {
    let quanta = targeted_execution_quanta(8_194, 1).unwrap();
    assert_eq!(quanta.len(), 129);
    assert!(quanta[..128].iter().all(|repetitions| *repetitions == 64));
    assert_eq!(quanta[128], 2);
    assert_eq!(quanta.iter().sum::<usize>(), 8_194);
}

#[test]
fn targeted_component_quanta_cover_prefill_work_exactly() {
    let quanta = targeted_execution_quanta(4_096, 64).unwrap();
    assert_eq!(quanta, vec![1; 64]);
    assert_eq!(
        quanta.iter().sum::<usize>() * 64,
        4_096,
    );
}

#[test]
fn targeted_component_quanta_reject_partial_activation_batches() {
    let error = targeted_execution_quanta(65, 64).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("is not divisible by activation width"),
        "{error}"
    );
}

#[test]
fn targeted_component_fixture_is_deterministic_bounded_bf16() {
    let first = targeted_fixture_bytes(4_096, 17, 2);
    let repeated = targeted_fixture_bytes(4_096, 17, 2);
    let other_seed = targeted_fixture_bytes(4_096, 18, 2);
    let other_binding = targeted_fixture_bytes(4_096, 17, 3);
    assert_eq!(first, repeated);
    assert_ne!(first, other_seed);
    assert_ne!(first, other_binding);
    assert_eq!(first.len(), 4_096);
    for bytes in first.chunks_exact(2) {
        let bf16 = u16::from_le_bytes([bytes[0], bytes[1]]);
        let value = f32::from_bits(u32::from(bf16) << 16);
        assert!(value.is_finite());
        assert!(value.abs() <= 4.031_25, "{value}");
    }
}
#[test]
fn targeted_output_identity_remains_an_artifact_digest() {
    let digest = targeted_finalized_artifact_digest(&[0xAB; 32]);

    assert_eq!(
        digest,
        format!("nerve.optimizer.artifact_sha256.v1:{}", "ab".repeat(32))
    );
}

#[test]
fn targeted_prefill_accepts_truthful_stateless_causal_scan_metadata() {
    assert!(targeted_prefill_batch_mode_is_supported(
        VulkanResidentComponentKernelBatchMode::CausalScan,
    ));
    assert!(targeted_prefill_batch_mode_is_supported(
        VulkanResidentComponentKernelBatchMode::WeightShared,
    ));
    assert!(!targeted_prefill_batch_mode_is_supported(
        VulkanResidentComponentKernelBatchMode::SerialLanes,
    ));
}
