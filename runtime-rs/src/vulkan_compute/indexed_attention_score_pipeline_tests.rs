#[cfg(test)]
mod indexed_attention_score_pipeline_tests {
    use super::*;
    use std::mem::size_of;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{Duration, Instant};

    const QUERY_HEADS: usize = 64;
    const HEAD_WIDTH: usize = 512;
    const LOCAL_WINDOW: usize = 128;
    const MAX_COMPRESSED_INDICES: usize = 8192;

    fn rendered_product_geometry_shader(
        compression_ratio: usize,
        max_compressed_indices: usize,
    ) -> String {
        std::fs::read_to_string(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("shaders")
                .join("indexed_sparse_attention_main_score_pipeline_bf16.comp.template"),
        )
        .expect("score-pipelined indexed-attention shader template must exist")
        .replace("{{LOCAL_SIZE}}", "576")
        .replace("{{QUERY_HEADS}}", "64")
        .replace("{{HEAD_WIDTH}}", "512")
        .replace("{{LOCAL_WINDOW}}", "128")
        .replace("{{COMPRESSION_RATIO}}", &compression_ratio.to_string())
        .replace(
            "{{MAX_COMPRESSED_INDICES}}",
            &max_compressed_indices.to_string(),
        )
        .replace("{{ATTENTION_SCALE}}", "0.04419417382415922")
    }

    fn rendered_baseline_shader(
        compression_ratio: usize,
        max_compressed_indices: usize,
    ) -> String {
        std::fs::read_to_string(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("shaders")
                .join("indexed_sparse_attention_main_bf16.comp.template"),
        )
        .unwrap()
        .replace("{{LOCAL_SIZE}}", "512")
        .replace("{{QUERY_HEADS}}", "64")
        .replace("{{KV_HEADS}}", "1")
        .replace("{{HEAD_WIDTH}}", "512")
        .replace("{{LOCAL_WINDOW}}", "128")
        .replace("{{COMPRESSION_RATIO}}", &compression_ratio.to_string())
        .replace(
            "{{MAX_COMPRESSED_INDICES}}",
            &max_compressed_indices.to_string(),
        )
        .replace("{{ATTENTION_SCALE}}", "0.04419417382415922")
        .replace("{{HAS_COMPRESSED}}", "1")
        .replace("{{OUTPUT_BINDING}}", "4")
        .replace("{{SINK_BINDING}}", "5")
        .replace("{{CONTROL_BINDING}}", "8")
    }

    fn compile_source(label: &str, source: String) -> Vec<u32> {
        static SOURCE_COUNTER: AtomicU64 = AtomicU64::new(0);
        assert!(!source.contains("{{"));
        let source_id = SOURCE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let source_path = std::env::temp_dir().join(format!(
            "nerve-{label}-{}-{source_id}.comp",
            std::process::id(),
        ));
        std::fs::write(&source_path, source).unwrap();
        let words = compile_shader_words_from_source_path(&source_path)
            .unwrap_or_else(|| panic!("{label} shader must compile"));
        let _ = std::fs::remove_file(source_path);
        words
    }

    fn bf16_bits(value: f32) -> u16 {
        let bits = value.to_bits();
        let lsb = (bits >> 16) & 1;
        ((bits + 0x7fff + lsb) >> 16) as u16
    }

    fn u16_bytes(values: &[u16]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn u32_bytes(values: &[u32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn fixed_state_view_bytes(values: &[u16]) -> Vec<u8> {
        let payload_bytes = u16_bytes(values);
        let mut header = [0u32; 9];
        header[2] = u32::try_from(payload_bytes.len()).unwrap();
        header[3] = u32::try_from(header.len() * size_of::<u32>()).unwrap();
        let mut bytes = u32_bytes(&header);
        bytes.extend_from_slice(&payload_bytes);
        bytes
    }

    fn binding<'a>(
        binding: u32,
        buffer: &'a VulkanResidentBuffer,
        access: VulkanResidentKernelBufferAccess,
    ) -> VulkanResidentKernelBufferBinding<'a> {
        VulkanResidentKernelBufferBinding::new(binding, buffer, buffer.byte_capacity())
            .with_access(access)
    }

    fn timestamped(
        device: &VulkanComputeDevice,
        dispatch: &VulkanResidentKernelDispatch,
    ) -> VulkanResidentKernelSequence {
        let sequence = device
            .create_timestamped_resident_kernel_sequence()
            .unwrap();
        device
            .record_resident_kernel_sequence(
                &sequence,
                &[VulkanResidentKernelSequenceStep::new(dispatch, &[])],
            )
            .unwrap();
        sequence
    }

    #[test]
    fn score_pipelined_indexed_attention_compiles_for_product_geometry() {
        let words = compile_source(
            "indexed-attention-score-pipeline",
            rendered_product_geometry_shader(128, 8192),
        );
        assert!(!words.is_empty());
    }

    #[test]
    fn score_pipelined_indexed_attention_is_exact_and_faster_at_product_geometry() {
        let started = Instant::now();
        let device = selected_test_vulkan_device().unwrap();
        assert_eq!(device.subgroup_size, 64);

        let query_values = (0..QUERY_HEADS * HEAD_WIDTH)
            .map(|index| bf16_bits(((index * 17 % 97) as f32 - 48.0) / 64.0))
            .collect::<Vec<_>>();
        let local_values = (0..LOCAL_WINDOW * HEAD_WIDTH)
            .map(|index| bf16_bits(((index * 13 % 61) as f32 - 30.0) / 128.0))
            .collect::<Vec<_>>();
        let compressed_values = (0..MAX_COMPRESSED_INDICES * HEAD_WIDTH)
            .map(|index| bf16_bits(((index * 29 % 127) as f32 - 63.0) / 256.0))
            .collect::<Vec<_>>();
        let indices = (0..MAX_COMPRESSED_INDICES)
            .map(|index| u32::try_from(LOCAL_WINDOW + index).unwrap())
            .collect::<Vec<_>>();
        let query = device
            .create_resident_buffer(query_values.len() * size_of::<u16>())
            .unwrap();
        query.write_bytes(&u16_bytes(&query_values)).unwrap();
        let local_bytes = fixed_state_view_bytes(&local_values);
        let local_state = device.create_resident_buffer(local_bytes.len()).unwrap();
        local_state.write_bytes(&local_bytes).unwrap();
        let compressed_bytes = fixed_state_view_bytes(&compressed_values);
        let compressed_state = device
            .create_resident_buffer(compressed_bytes.len())
            .unwrap();
        compressed_state.write_bytes(&compressed_bytes).unwrap();
        let compressed_indices = device
            .create_resident_buffer(indices.len() * size_of::<u32>())
            .unwrap();
        compressed_indices.write_bytes(&u32_bytes(&indices)).unwrap();
        let output_bytes = QUERY_HEADS * HEAD_WIDTH * size_of::<u16>();
        let baseline_output = device.create_resident_buffer(output_bytes).unwrap();
        baseline_output.write_bytes(&vec![0; output_bytes]).unwrap();
        let pipelined_output = device.create_resident_buffer(output_bytes).unwrap();
        pipelined_output.write_bytes(&vec![0; output_bytes]).unwrap();
        let sinks = device
            .create_resident_buffer(QUERY_HEADS * size_of::<f32>())
            .unwrap();
        sinks
            .write_bytes(&u32_bytes(&vec![(-0.5_f32).to_bits(); QUERY_HEADS]))
            .unwrap();
        let control = device.create_resident_buffer(9 * size_of::<u32>()).unwrap();
        let dispatch = |words: &[u32], output: &VulkanResidentBuffer, local_size: u32| {
            device
                .create_resident_kernel_dispatch(
                    words,
                    &[
                        binding(0, &query, VulkanResidentKernelBufferAccess::Read),
                        binding(1, &local_state, VulkanResidentKernelBufferAccess::Read),
                        binding(2, &compressed_state, VulkanResidentKernelBufferAccess::Read),
                        binding(3, &compressed_indices, VulkanResidentKernelBufferAccess::Read),
                        binding(4, output, VulkanResidentKernelBufferAccess::Write),
                        binding(5, &sinks, VulkanResidentKernelBufferAccess::Read),
                        binding(8, &control, VulkanResidentKernelBufferAccess::Read),
                    ],
                    QUERY_HEADS as u32,
                    local_size,
                    0,
                )
                .unwrap()
        };
        let timeout = Duration::from_secs(30);
        let measure = |sequence: &VulkanResidentKernelSequence| {
            (0..2)
                .map(|_| {
                    device
                        .run_timestamped_recorded_resident_kernel_sequence_for(sequence, timeout)
                        .unwrap()
                })
                .min()
                .unwrap()
        };
        for (compression_ratio, max_compressed_indices) in [(4usize, 512usize), (128, 8192)] {
            let stream_tick = compression_ratio * max_compressed_indices - 1;
            control
                .write_bytes(&u32_bytes(&[
                    0,
                    u32::try_from(stream_tick).unwrap(),
                    0,
                    0,
                    LOCAL_WINDOW as u32,
                    0,
                    0,
                    0,
                    0,
                ]))
                .unwrap();
            let baseline_words = compile_source(
                "indexed-attention-baseline",
                rendered_baseline_shader(compression_ratio, max_compressed_indices),
            );
            let pipelined_words = compile_source(
                "indexed-attention-score-pipeline",
                rendered_product_geometry_shader(
                    compression_ratio,
                    max_compressed_indices,
                ),
            );
            let baseline_dispatch = dispatch(&baseline_words, &baseline_output, 512);
            let pipelined_dispatch = dispatch(&pipelined_words, &pipelined_output, 576);

            device
                .run_resident_kernel_dispatch(&baseline_dispatch, &[])
                .unwrap();
            device
                .run_resident_kernel_dispatch(&pipelined_dispatch, &[])
                .unwrap();
            assert_eq!(
                pipelined_output.read_bytes(output_bytes).unwrap(),
                baseline_output.read_bytes(output_bytes).unwrap(),
                "score pipelining must preserve every BF16 output bit for r{compression_ratio}/k{max_compressed_indices}",
            );

            let baseline_sequence = timestamped(&device, &baseline_dispatch);
            let pipelined_sequence = timestamped(&device, &pipelined_dispatch);
            for sequence in [&baseline_sequence, &pipelined_sequence] {
                device
                    .run_timestamped_recorded_resident_kernel_sequence_for(sequence, timeout)
                    .unwrap();
            }
            let baseline_ns = measure(&baseline_sequence);
            let pipelined_ns = measure(&pipelined_sequence);
            eprintln!(
                "indexed_attention_score_pipeline q={QUERY_HEADS} d={HEAD_WIDTH} r={compression_ratio} k={max_compressed_indices} baseline_ms={:.6} pipelined_ms={:.6} ratio={:.6} elapsed_ms={:.3}",
                baseline_ns as f64 / 1_000_000.0,
                pipelined_ns as f64 / 1_000_000.0,
                pipelined_ns as f64 / baseline_ns as f64,
                started.elapsed().as_secs_f64() * 1_000.0,
            );
            assert!(
                pipelined_ns < baseline_ns,
                "score-pipelined attention must be faster for r{compression_ratio}/k{max_compressed_indices} before product promotion",
            );
        }
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "score-pipelined attention microbenchmark exceeded one minute",
        );
    }
}
