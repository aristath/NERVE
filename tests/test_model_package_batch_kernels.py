from model_package_layout_common import *
from nerve.model_package_validation import valid_batch_stage, valid_indirect_dispatch_pipeline
from nerve.model_package_shader_compiler import compile_shader_artifacts
from nerve.model_package_shader_templates import parallelize_temporal_batch_lanes


def test_indirect_batch_dispatch_requires_an_earlier_writable_producer() -> None:
    control = {
        "kind": "storage_buffer",
        "byte_count": 28,
        "binding": 31,
        "payload": "width_expert_range_indirect",
    }
    consumer = {
        "shader_path": "shaders/consume.spv",
        "control": control,
        "indirect_dispatch_byte_offset": 16,
    }

    assert not valid_indirect_dispatch_pipeline([consumer])
    assert not valid_indirect_dispatch_pipeline(
        [
            consumer,
            {
                "shader_path": "shaders/late-producer.spv",
                "control": {**control, "access": "read_write"},
            },
        ]
    )
    assert valid_indirect_dispatch_pipeline(
        [
            {
                "shader_path": "shaders/producer.spv",
                "control": {**control, "access": "read_write"},
            },
            consumer,
        ]
    )


def test_compiler_selects_only_compatible_weight_shared_batch_kernels() -> None:
    assert SCALAR_BATCH_LANE_TILE_WIDTH == 16
    assert (
        weight_shared_batch_shader_file("rms_norm_bf16_h5120_eps1e-06_offset1.comp")
        == "rms_norm_batch16_bf16_h5120_eps1e-06_offset1.comp"
    )
    assert (
        weight_shared_batch_shader_file("linear_fp8_e4m3_b128x128_5120x17408.comp")
        == "linear_batch16_fp8_e4m3_b128x128_5120x17408.comp"
    )
    assert (
        weight_shared_batch_shader_file("quantize_fp8_e4m3_b128_h5120.comp")
        == "quantize_batch16_fp8_e4m3_b128_h5120.comp"
    )
    assert (
        weight_shared_batch_shader_file("quantize_int8_symmetric_b32_h5120.comp")
        == "quantize_batch16_int8_symmetric_b32_h5120.comp"
    )
    assert (
        weight_shared_batch_shader_file(
            "linear_prequant_fp8_e4m3_b128x128_5120x17408.comp"
        )
        == "linear_prequant_batch16_fp8_e4m3_b128x128_5120x17408.comp"
    )
    assert (
        weight_shared_batch_shader_file(
            "linear_residual_fp8_e4m3_b128x128_17408x5120.comp"
        )
        == "linear_residual_batch16_fp8_e4m3_b128x128_17408x5120.comp"
    )
    assert (
        weight_shared_batch_shader_file("linear_int4_gptq_sf16_g128_5120x17408.comp")
        == "linear_batch16_int4_gptq_sf16_g128_5120x17408.comp"
    )
    assert (
        weight_shared_batch_shader_file(
            "linear_prequant_int4_gptq_sf16_g128_5120x17408.comp"
        )
        == "linear_prequant_batch16_int4_gptq_sf16_g128_5120x17408.comp"
    )
    assert (
        weight_shared_batch_shader_file(
            "quantize_int8_symmetric_pairpacked_b32_h5120.comp"
        )
        == "quantize_batch16_int8_symmetric_pairpacked_b32_h5120.comp"
    )
    assert weight_shared_batch_shader_file(
        "linear_prequant_pairpacked_int4_gptq_sf16_g128_5120x17408.comp"
    ) == ("linear_prequant_pairpacked_batch16_int4_gptq_sf16_g128_5120x17408.comp")
    assert weight_shared_batch_shader_file(
        "rms_norm_quantize_int8_pairpacked_b32_h5120_eps1e-06_offset1.comp"
    ) == ("rms_norm_quantize_batch16_int8_pairpacked_b32_h5120_eps1e-06_offset1.comp")
    assert weight_shared_batch_shader_file(
        "silu_multiply_quantize_int8_pairpacked_b32_h17408.comp"
    ) == ("silu_multiply_quantize_batch16_int8_pairpacked_b32_h17408.comp")
    assert (
        weight_shared_batch_shader_file(
            "linear_residual_int4_ct_sbf16_g32_16384x5376.comp"
        )
        == "linear_residual_batch16_int4_ct_sbf16_g32_16384x5376.comp"
    )
    assert (
        weight_shared_batch_shader_file("parallel_linear_2way_bf16_1024x2560_2560.comp")
        == "parallel_linear_batch16_2way_bf16_1024x2560_2560.comp"
    )
    assert (
        weight_shared_batch_shader_file(
            "parallel_linear_2way_fp8_e4m3_b128x128_5120x5120_1024.comp"
        )
        == "parallel_linear_batch16_2way_fp8_e4m3_b128x128_5120x5120_1024.comp"
    )
    assert weight_shared_batch_shader_file(
        "parallel_linear_silu_multiply_fp8_e4m3_b128x128_5120x17408.comp"
    ) == ("parallel_linear_silu_multiply_batch16_fp8_e4m3_b128x128_5120x17408.comp")
    assert weight_shared_batch_shader_file(
        "parallel_linear_silu_multiply_prequant_fp8_e4m3_b128x128_5120x17408.comp"
    ) == (
        "parallel_linear_silu_multiply_prequant_batch16_fp8_e4m3_"
        "b128x128_5120x17408.comp"
    )
    assert weight_shared_batch_shader_file(
        "parallel_linear_2way_prequant_fp8_e4m3_b128x128_5120x5120_1024.comp"
    ) == ("parallel_linear_batch16_2way_prequant_fp8_e4m3_b128x128_5120x5120_1024.comp")
    assert weight_shared_batch_shader_file(
        "mixed_parallel_linear_4way_prequant_fp8_e4m3_"
        "b128x128_bf16_2048x8192_4096_32_32.comp",
        tile_width=4,
    ) == (
        "mixed_parallel_linear_4way_prequant_batch4_fp8_e4m3_"
        "b128x128_bf16_2048x8192_4096_32_32.comp"
    )
    assert weight_shared_batch_shader_file(
        "contiguous_linear_swiglu_prequant_fp8_e4m3_b128x128_2048x512.comp",
        tile_width=4,
    ) == ("contiguous_linear_swiglu_prequant_batch4_fp8_e4m3_b128x128_2048x512.comp")
    assert (
        weight_shared_batch_shader_file("linear_bf16_1024x1024.comp")
        == "linear_batch16_bf16_1024x1024.comp"
    )
    assert (
        weight_shared_batch_shader_file("linear_bf16_1024x1024.comp", tile_width=4)
        == "linear_batch4_bf16_1024x1024.comp"
    )
    assert (
        weight_shared_batch_shader_file("linear_bf16_2048x1.comp", tile_width=4)
        == "linear_batch4_bf16_2048x1.comp"
    )
    assert (
        weight_shared_batch_shader_file("linear_residual_bf16_1024x1024.comp")
        == "linear_residual_batch16_bf16_1024x1024.comp"
    )
    assert (
        weight_shared_batch_shader_file(
            "parallel_linear_silu_multiply_bf16_1024x4096.comp"
        )
        == "parallel_linear_silu_multiply_batch16_bf16_1024x4096.comp"
    )
    assert (
        weight_shared_batch_shader_file("split_bf16_2x16x256_head_interleaved.comp")
        == "split_batch16_bf16_2x16x256_head_interleaved.comp"
    )
    assert (
        weight_shared_batch_shader_file("split_bf16_2x512.comp")
        == "split_batch16_bf16_2x512.comp"
    )
    assert weight_shared_batch_shader_file("split_bf16_2x511.comp") is None
    assert (
        weight_shared_batch_shader_file("silu_multiply_bf16_512.comp")
        == "silu_multiply_batch16_bf16_512.comp"
    )
    assert (
        weight_shared_batch_shader_file("sigmoid_scalar_multiply_bf16_2048.comp")
        == "sigmoid_scalar_multiply_batch16_bf16_2048.comp"
    )
    assert (
        weight_shared_batch_shader_file(
            "linear_sigmoid_scalar_multiply_bf16_2048x2048.comp"
        )
        == "linear_sigmoid_scalar_multiply_batch16_bf16_2048x2048.comp"
    )
    assert weight_shared_batch_shader_file(
        "linear_sigmoid_scalar_multiply_residual2_bf16_2048x2048.comp"
    ) == ("linear_sigmoid_scalar_multiply_residual2_batch16_bf16_2048x2048.comp")
    assert (
        weight_shared_batch_shader_file("add_bf16_2048.comp")
        == "add_batch16_bf16_2048.comp"
    )
    assert weight_shared_batch_shader_file(
        "hyper_connection_pre_m4_h4096_i20_neps1e-06_heps1e-06.comp",
        tile_width=8,
    ) == ("hyper_connection_pre_batch8_m4_h4096_i20_neps1e-06_heps1e-06.comp")
    assert weight_shared_batch_shader_file(
        "hyper_connection_post_pre_m4_h4096_i20_neps1e-06_heps1e-06.comp",
        tile_width=8,
    ) == ("hyper_connection_post_pre_batch8_m4_h4096_i20_neps1e-06_heps1e-06.comp")
    assert (
        weight_shared_batch_shader_file(
            "hyper_connection_post_m4_h4096.comp",
            tile_width=8,
        )
        == "hyper_connection_post_batch8_m4_h4096.comp"
    )
    assert (
        weight_shared_batch_shader_file(
            "rms_norm_per_head_unscaled_bf16_64x512_eps1e-06.comp",
            tile_width=8,
        )
        == "rms_norm_per_head_unscaled_batch8_bf16_64x512_eps1e-06.comp"
    )
    assert weight_shared_batch_shader_file(
        "grouped_linear_fp8_e4m3_se8m0_b128x128_g8_32768x8192.comp",
        tile_width=8,
    ) == ("grouped_linear_batch8_fp8_e4m3_se8m0_b128x128_g8_32768x8192.comp")
    assert (
        weight_shared_batch_shader_file(
            "bounded_silu_multiply_bf16_2048_limit10.comp",
            tile_width=8,
        )
        == "bounded_silu_multiply_batch8_bf16_2048_limit10.comp"
    )
    for lane_parallel_shader in (
        "linear_bf16_2048x1.comp",
        "split_bf16_2x512.comp",
        "silu_multiply_bf16_512.comp",
        "sigmoid_scalar_multiply_bf16_2048.comp",
        "linear_sigmoid_scalar_multiply_bf16_2048x2048.comp",
        "linear_sigmoid_scalar_multiply_residual2_bf16_2048x2048.comp",
        "add_bf16_2048.comp",
    ):
        assert (
            weight_shared_batch_workgroup_count_x(
                lane_parallel_shader,
                tile_width=4,
                scalar_workgroup_count_x=1,
            )
            == 4
        )
    assert (
        weight_shared_batch_workgroup_count_x(
            "linear_bf16_1024x4096.comp",
            tile_width=4,
            scalar_workgroup_count_x=2048,
        )
        == 2048
    )
    assert (
        weight_shared_batch_workgroup_count_x(
            "split_bf16_2x16x256_head_interleaved.comp",
            tile_width=4,
            scalar_workgroup_count_x=8,
        )
        == 8
    )
    assert (
        weight_shared_batch_shader_file("sigmoid_multiply_bf16.comp")
        == "sigmoid_multiply_batch16_bf16.comp"
    )
    assert (
        weight_shared_batch_shader_file("softplus_multiply_bf16_q72_d128_per_head.comp")
        == "softplus_multiply_batch16_bf16_q72_d128_per_head.comp"
    )
    assert (
        weight_shared_batch_shader_file("linear_fp8_e4m3_b127x128_5120x17408.comp")
        is None
    )
    assert weight_shared_batch_shader_file("linear_bf16_1023x1024.comp") is None
    assert (
        frame_parallel_batch_shader_file(
            "rms_norm_batch16_bf16_h4096_eps1e-06_offset1.comp"
        )
        == "rms_norm_batch1_bf16_h4096_eps1e-06_offset1.comp"
    )
    assert (
        frame_parallel_batch_shader_file(
            "split_batch16_bf16_2x16x256_head_interleaved.comp"
        )
        == "split_batch1_bf16_2x16x256_head_interleaved.comp"
    )
    assert (
        frame_parallel_batch_shader_file("linear_batch16_bf16_4096x4096.comp") is None
    )
    assert (
        frame_parallel_batch_shader_file("moe_topk_bf16_e256_k8.comp")
        == "moe_topk_batch1_bf16_e256_k8.comp"
    )
    assert (
        frame_parallel_batch_shader_file(
            "sparse_moe_gate_up_fp8_e4m3_b128x128_h2048_i512_e256_k8.comp"
        )
        == "sparse_moe_gate_up_batch1_fp8_e4m3_b128x128_h2048_i512_e256_k8.comp"
    )
    assert (
        frame_parallel_batch_shader_file("moe_reduce_bf16_h2048_k8_scale1.comp")
        == "moe_reduce_batch1_bf16_h2048_k8_scale1.comp"
    )
    fp8_cooperative = cooperative_float8_e4m3_batch_shader_file(
        "linear_residual_prequant_fp8_e4m3_b128x128_17408x5120.comp",
        shape=(16, 16, 16),
    )
    assert fp8_cooperative == (
        "linear_residual_prequant_batch64_cooperative_"
        "fp8_e4m3_m16n16k16_b128x128_17408x5120.comp"
    )
    assert (
        cooperative_float8_e4m3_workgroup_count_x(
            "linear_residual_prequant_fp8_e4m3_b128x128_17408x5120.comp",
            shape=(16, 16, 16),
        )
        == 80
    )
    assert (
        compact_cooperative_float8_e4m3_batch_shader_file(
            "linear_prequant_fp8_e4m3_b128x128_1024x32768.comp",
            shape=(16, 16, 16),
        )
        == "linear_prequant_batch16_cooperative_fp8_e4m3_"
        "m16n16k16_b128x128_1024x32768.comp"
    )
    assert (
        compact_cooperative_float8_e4m3_batch_shader_file(
            "linear_prequant_fp8_e4m3_b128x128_4096x2048.comp",
            shape=(16, 16, 16),
        )
        is None
    )
    parallel_fp8_cooperative = cooperative_float8_e4m3_batch_shader_file(
        "parallel_linear_3way_prequant_fp8_e4m3_b128x128_5120x6144_1024_1024.comp",
        shape=(16, 16, 16),
    )
    assert parallel_fp8_cooperative == (
        "parallel_linear_batch64_3way_prequant_cooperative_"
        "fp8_e4m3_m16n16k16_b128x128_5120x6144_1024_1024.comp"
    )
    assert (
        cooperative_float8_e4m3_workgroup_count_x(
            "parallel_linear_3way_prequant_fp8_e4m3_b128x128_5120x6144_1024_1024.comp",
            shape=(16, 16, 16),
        )
        == 96
    )
    fused_ffn_fp8_cooperative = cooperative_float8_e4m3_batch_shader_file(
        "parallel_linear_silu_multiply_prequant_fp8_e4m3_b128x128_5120x17408.comp",
        shape=(16, 16, 16),
    )
    assert fused_ffn_fp8_cooperative == (
        "parallel_linear_silu_multiply_prequant_batch64_cooperative_"
        "fp8_e4m3_m16n16k16_b128x128_5120x17408.comp"
    )
    assert (
        cooperative_float8_e4m3_workgroup_count_x(
            "parallel_linear_silu_multiply_prequant_fp8_e4m3_b128x128_5120x17408.comp",
            shape=(16, 16, 16),
        )
        == 272
    )
    contiguous_swiglu_cooperative = cooperative_float8_e4m3_batch_shader_file(
        "contiguous_linear_swiglu_prequant_fp8_e4m3_b128x128_2048x512.comp",
        shape=(16, 16, 16),
    )
    assert contiguous_swiglu_cooperative == (
        "contiguous_linear_swiglu_prequant_batch64_cooperative_"
        "fp8_e4m3_m16n16k16_b128x128_2048x512.comp"
    )
    assert (
        cooperative_float8_e4m3_batch_shader_file(
            "contiguous_linear_swiglu_prequant_fp8_e4m3_b128x128_2048x520.comp",
            shape=(16, 16, 16),
        )
        is None
    )
    assert (
        cooperative_float8_e4m3_workgroup_count_x(
            "contiguous_linear_swiglu_prequant_fp8_e4m3_b128x128_2048x512.comp",
            shape=(16, 16, 16),
        )
        == 16
    )


def test_compiler_orders_frame_parallel_before_portable_batch_implementation() -> None:
    spec = component_kernel_spec(
        execution_index=0,
        node={"id": "norm", "op": "rms_norm"},
        circuit={},
        shader_file="rms_norm_bf16_h4096_eps1e-06_offset1.comp",
        local_size_x=64,
        workgroup_count_x=1,
    )

    frame_parallel, *portable = spec["batch_implementations"]
    assert spec["execution_domain"] == "decode"
    assert frame_parallel["execution_domain"] == "decode_and_prefill"
    assert frame_parallel["lane_tile_width"] == 1
    assert frame_parallel["independent_candidate_compatible"] is True
    assert frame_parallel["causal_sequence_compatible"] is True
    assert frame_parallel["parallel_block_compatible"] is True
    assert frame_parallel["device_requirements"] == {
        "vulkan_device_extensions": [],
        "vulkan_features": [],
        "subgroup_operations": [],
        "subgroup_size": 64,
    }
    assert frame_parallel["stages"][0]["shader_path"] == (
        "shaders/rms_norm_batch1_bf16_h4096_eps1e-06_offset1__pbc31.comp"
    )
    assert frame_parallel["stages"][0]["dispatch_y_from_batch_width"] is True
    assert [implementation["lane_tile_width"] for implementation in portable] == [
        2,
        4,
        8,
        16,
    ]
    assert all(
        implementation["independent_candidate_compatible"] is True
        for implementation in portable
    )
    assert all(
        implementation["causal_sequence_compatible"] is True
        for implementation in portable
    )
    assert all(
        implementation["parallel_block_compatible"] is True
        for implementation in portable
    )


def test_fused_hyper_norm_decode_keeps_exact_two_stage_batch_execution() -> None:
    cases = (
        (
            "hyper_connection_pre_rms_norm",
            "hyper_connection_pre_rms_norm_quantize_fp8_e4m3_spow2_b128_"
            "m4_h4096_i20_neps1e-06_heps1e-06_reps1e-06_roffset0.comp",
            [0, 1, 2, 3, 6, 7, 8],
            [1, 4, 5, 9],
        ),
        (
            "hyper_connection_post_pre_rms_norm",
            "hyper_connection_post_pre_rms_norm_quantize_fp8_e4m3_spow2_b128_"
            "m4_h4096_i20_neps1e-06_heps1e-06_reps1e-06_roffset0.comp",
            [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12],
            [5, 8, 9, 13],
        ),
    )
    for operation, shader_file, hyper_sources, rms_sources in cases:
        spec = component_kernel_spec(
            execution_index=0,
            node={"id": "fused", "op": operation},
            circuit={},
            shader_file=shader_file,
            local_size_x=1024,
            workgroup_count_x=1,
        )

        assert spec["execution_domain"] == "decode"
        assert spec["batch_mode"] == "weight_shared"
        assert [
            implementation["lane_tile_width"]
            for implementation in spec["batch_implementations"]
        ] == [2, 4, 8, 16]
        for implementation in spec["batch_implementations"]:
            assert implementation["execution_domain"] == "decode_and_prefill"
            assert len(implementation["stages"]) == 2
            hyper_stage, rms_stage = implementation["stages"]
            assert valid_batch_stage(hyper_stage)
            assert valid_batch_stage(rms_stage)
            assert hyper_stage["workgroup_count_x"] == implementation["lane_tile_width"]
            assert rms_stage["workgroup_count_x"] == 1
            assert [
                mapping["source_binding"]
                for mapping in hyper_stage["descriptor_bindings"]
            ] == hyper_sources
            assert [
                mapping["source_binding"]
                for mapping in rms_stage["descriptor_bindings"]
            ] == rms_sources
            assert "rms_norm_quantize_in_place_batch" in rms_stage["shader_path"]


def test_compiler_marks_all_visible_indexed_attention_as_parallel_block_only() -> None:
    spec = component_kernel_spec(
        execution_index=0,
        node={"id": "attend", "op": "indexed_sparse_attention"},
        circuit={},
        shader_file=(
            "indexed_sparse_attention_bf16_q64_kv1_d512_w128_"
            "scale0.0441941738__sc6.comp"
        ),
        local_size_x=512,
        workgroup_count_x=64,
    )

    assert spec["batch_mode"] == "weight_shared"
    assert len(spec["batch_implementations"]) == 1
    [implementation] = spec["batch_implementations"]
    assert implementation["lane_tile_width"] == 64
    assert implementation["independent_candidate_compatible"] is False
    assert implementation["causal_sequence_compatible"] is False
    assert implementation["parallel_block_compatible"] is True
    assert implementation["stages"] == [
        {
            "shader_path": (
                "shaders/indexed_sparse_attention_parallel_bf16_q64_kv1_d512_"
                "w128_scale0.0441941738__pbc6.comp"
            ),
            "local_size_x": 512,
            "workgroup_count_x": 64,
            "control": {
                "kind": "storage_buffer",
                "byte_count": 16,
                "binding": 6,
                "payload": "temporal",
            },
            "dispatch_y_from_batch_width": True,
        }
    ]


def test_compiler_selects_stateful_causal_scan_kernels() -> None:
    assert CAUSAL_SCAN_LANE_TILE_WIDTH == 64
    assert (
        causal_scan_batch_shader_file("causal_conv1d_silu_bf16_c8192_k4.comp")
        == "causal_conv1d_silu_temporal_bf16_c8192_k4.comp"
    )
    assert (
        causal_scan_batch_shader_file(
            "gated_delta_step_k16x128_v32x128_af32_dtbf16_nf32_eps1e-06.comp"
        )
        == "gated_delta_scan_k16x128_v32x128_af32_dtbf16_nf32_eps1e-06.comp"
    )
    assert (
        causal_scan_batch_shader_file(
            "gated_delta_step_k16x128_v32x128_af32_dtbf16_nf32_eps1e-06_qfp8b128.comp"
        )
        == "gated_delta_scan_k16x128_v32x128_af32_dtbf16_nf32_"
        "eps1e-06_qfp8b128.comp"
    )
    assert causal_scan_batch_shader_file(
        "parallel_head_norm_rope_2way_bf16_h16_4_d256_r64_eps1e-06_"
        "offset1_theta10000000_half__sc6.comp"
    ) == (
        "parallel_head_norm_rope_2way_temporal_bf16_h16_4_d256_r64_"
        "eps1e-06_offset1_theta10000000_half.comp"
    )
    assert (
        causal_scan_batch_shader_file(
            "rotary_bf16_16x256_r64_theta10000000_half__sc2.comp"
        )
        == "rotary_temporal_bf16_16x256_r64_theta10000000_half.comp"
    )
    assert causal_scan_batch_shader_file(
        "inverse_rotary_bf16_64x512_r64_theta160000_"
        "yarn_f16_lo15_hi25_a1_interleaved_tail_po1__sc2.comp"
    ) == (
        "inverse_rotary_temporal_bf16_64x512_r64_theta160000_"
        "yarn_f16_lo15_hi25_a1_interleaved_tail_po1.comp"
    )
    assert causal_scan_batch_shader_file("linear_bf16_4096x4096.comp") is None
    assert causal_scan_workgroup_count_x("causal_conv1d_silu_bf16_c8192_k4.comp") == 64
    assert (
        causal_scan_workgroup_count_x(
            "gated_delta_step_k16x128_v32x128_af32_dtbf16_nf32_eps1e-06.comp"
        )
        == 32
    )
    assert (
        causal_scan_workgroup_count_x(
            "parallel_head_norm_rope_2way_bf16_h16_4_d256_r64_eps1e-06_"
            "offset1_theta10000000_half__sc6.comp"
        )
        == 20
    )
    assert (
        causal_scan_workgroup_count_x(
            "rotary_bf16_16x256_r64_theta10000000_half__sc2.comp"
        )
        == 16
    )
    assert (
        causal_scan_workgroup_count_x(
            "inverse_rotary_bf16_64x512_r64_theta160000_"
            "yarn_f16_lo15_hi25_a1_interleaved_tail__sc2.comp"
        )
        == 64
    )
    deepseek_temporal_kernels = {
        "rolling_state_ring_append_bf16_128x512__sc6.comp": (
            "rolling_state_ring_append_temporal_bf16_128x512.comp",
            1,
        ),
        "learned_gated_kv_pool_bf16_f32_h4096_d512_r4_c2__sc8.comp": (
            "learned_gated_kv_pool_temporal_bf16_f32_h4096_d512_r4_c2.comp",
            512,
        ),
        "compressed_kv_finalize_f32_bf16_d512_r64_eps1e-06_theta160000_"
        "yarn_f16_lo15_hi25_a1_interleaved_po-3_qfp8e4m3b64__sc3.comp": (
            "compressed_kv_finalize_temporal_f32_bf16_d512_r64_eps1e-06_"
            "theta160000_yarn_f16_lo15_hi25_a1_interleaved_po-3_"
            "qfp8e4m3b64.comp",
            1,
        ),
        "conditional_append_state_bf16_d512_p4__sc6.comp": (
            "conditional_append_state_temporal_bf16_d512_p4.comp",
            1,
        ),
        "index_vector_transform_bf16_h64_d128_r64_theta160000_"
        "yarn_f16_lo15_hi25_a1_interleaved_qfp4e2m1b32__sc2.comp": (
            "index_vector_transform_temporal_bf16_h64_d128_r64_theta160000_"
            "yarn_f16_lo15_hi25_a1_interleaved_qfp4e2m1b32.comp",
            64,
        ),
        "compressed_index_kv_finalize_f32_bf16_d128_r64_eps1e-06_"
        "theta160000_yarn_f16_lo15_hi25_a1_interleaved_po-3_"
        "qfp4e2m1b32__sc3.comp": (
            "compressed_index_kv_finalize_temporal_f32_bf16_d128_r64_"
            "eps1e-06_theta160000_yarn_f16_lo15_hi25_a1_interleaved_"
            "po-3_qfp4e2m1b32.comp",
            1,
        ),
        "learned_index_scores_bf16_f32_h64_d128_r4_m262144_c256_"
        "scale0.0110485435__sc5.comp": (
            "learned_index_scores_temporal_bf16_f32_h64_d128_r4_m262144_"
            "c256_scale0.0110485435.comp",
            1024,
        ),
        "radix_topk_index_f32_u32_m262144_k512_r4_o128__sc2.comp": (
            "radix_topk_index_temporal_f32_u32_m262144_k512_r4_o128.comp",
            1,
        ),
        "chronological_compressed_index_u32_m8192_r128_o128__sc3.comp": (
            "chronological_compressed_index_temporal_u32_m8192_r128_o128.comp",
            1,
        ),
        "indexed_sparse_attention_main_bf16_q64_kv1_d512_w128_r4_k512_"
        "scale0.0441941738__sc8.comp": (
            "indexed_sparse_attention_main_temporal_parallel_bf16_q64_kv1_d512_"
            "w128_r4_k512_scale0.0441941738.comp",
            64,
        ),
    }
    for scalar_shader, (
        temporal_shader,
        workgroups,
    ) in deepseek_temporal_kernels.items():
        assert causal_scan_batch_shader_file(scalar_shader) == temporal_shader
        assert causal_scan_workgroup_count_x(scalar_shader) == workgroups
    unsupported_multi_kv_attention = (
        "indexed_sparse_attention_main_bf16_q64_kv8_d512_w128_r4_k512_"
        "scale0.0441941738__sc8.comp"
    )
    assert causal_scan_batch_shader_file(unsupported_multi_kv_attention) is None
    pipelined_attention = (
        "indexed_sparse_attention_main_score_pipeline_bf16_"
        "q64_kv1_d512_w128_r4_k512_scale0.0441941738__sc8.comp"
    )
    assert causal_scan_batch_shader_file(pipelined_attention) == (
        "indexed_sparse_attention_main_score_pipeline_temporal_parallel_bf16_"
        "q64_kv1_d512_w128_r4_k512_scale0.0441941738.comp"
    )
    assert causal_scan_workgroup_count_x(pipelined_attention) == 64
    tile_overlapped_attention = (
        "indexed_sparse_attention_main_tile_overlap_bf16_"
        "q64_kv1_d512_w128_r4_k512_scale0.0441941738__sc8.comp"
    )
    assert causal_scan_batch_shader_file(tile_overlapped_attention) == (
        "indexed_sparse_attention_main_tile_overlap_temporal_parallel_bf16_"
        "q64_kv1_d512_w128_r4_k512_scale0.0441941738.comp"
    )
    assert causal_scan_workgroup_count_x(tile_overlapped_attention) == 64

    attention_local_size = attention_workgroup_shape(256)[0]
    assert causal_scan_batch_stages(
        "append_gqa_attention_bf16_q16_kv4_d256_scale0.0625__sc7.comp",
        attention_local_size,
    ) == [
        {
            "shader_path": (
                "shaders/append_gqa_attention_temporal_read_bf16_"
                "q16_kv4_d256_scale0.0625__pbc7.comp"
            ),
            "local_size_x": attention_local_size,
            "workgroup_count_x": 16 * 64,
            "control": {
                "kind": "storage_buffer",
                "byte_count": 16,
                "binding": 7,
                "payload": "temporal",
            },
        },
        {
            "shader_path": "shaders/append_kv_temporal_commit_bf16_kv4_d256_w0__pbc7.comp",
            "local_size_x": 64,
            "workgroup_count_x": 4,
            "control": {
                "kind": "storage_buffer",
                "byte_count": 16,
                "binding": 7,
                "payload": "temporal",
            },
        },
    ]
    sink_stages = causal_scan_batch_stages(
        "append_gqa_attention_bf16_q16_kv4_d256_scale0.0625_w32768_sinks__sc7.comp",
        attention_local_size,
    )
    assert sink_stages is not None
    assert [stage["control"] for stage in sink_stages] == [
        {
            "kind": "storage_buffer",
            "byte_count": 16,
            "binding": 8,
            "payload": "temporal",
        },
        {
            "kind": "storage_buffer",
            "byte_count": 16,
            "binding": 8,
            "payload": "temporal",
        },
    ]
    rope_stages = causal_scan_batch_stages(
        "parallel_head_norm_rope_2way_bf16_h16_4_d256_r64_eps1e-06_"
        "offset1_theta10000000_half__sc6.comp",
        64,
    )
    assert rope_stages is not None
    assert rope_stages[0]["control"] == {
        "kind": "storage_buffer",
        "byte_count": 16,
        "binding": 6,
        "payload": "temporal",
    }
    standalone_rope_stages = causal_scan_batch_stages(
        "rotary_bf16_16x256_r64_theta10000000_half__sc2.comp",
        64,
    )
    assert standalone_rope_stages == [
        {
            "shader_path": (
                "shaders/rotary_temporal_bf16_16x256_r64_theta10000000_half__pbc2.comp"
            ),
            "local_size_x": 64,
            "workgroup_count_x": 16,
            "control": {
                "kind": "storage_buffer",
                "byte_count": 16,
                "binding": 2,
                "payload": "temporal",
            },
        }
    ]
    inverse_rope_stages = causal_scan_batch_stages(
        "inverse_rotary_bf16_64x512_r64_theta160000_"
        "yarn_f16_lo15_hi25_a1_interleaved_tail__sc2.comp",
        64,
    )
    assert inverse_rope_stages == [
        {
            "shader_path": (
                "shaders/inverse_rotary_temporal_bf16_64x512_r64_theta160000_"
                "yarn_f16_lo15_hi25_a1_interleaved_tail__pbc2.comp"
            ),
            "local_size_x": 64,
            "workgroup_count_x": 64,
            "control": {
                "kind": "storage_buffer",
                "byte_count": 16,
                "binding": 2,
                "payload": "temporal",
            },
        }
    ]
    rolling_stages = causal_scan_batch_stages(
        "rolling_state_ring_append_bf16_128x512__sc6.comp",
        64,
    )
    assert rolling_stages == [
        {
            "shader_path": (
                "shaders/rolling_state_ring_append_temporal_bf16_128x512__pbc6.comp"
            ),
            "local_size_x": 64,
            "workgroup_count_x": 1,
            "control": {
                "kind": "storage_buffer",
                "byte_count": 20,
                "binding": 6,
                "payload": "temporal_state_snapshots",
            },
            "state_snapshot_binding": 30,
        }
    ]
    pool_stages = causal_scan_batch_stages(
        "learned_gated_kv_pool_bf16_f32_h4096_d512_r4_c2__sc8.comp",
        64,
    )
    assert pool_stages[0]["control"] == {
        "kind": "storage_buffer",
        "byte_count": 20,
        "binding": 8,
        "payload": "temporal_state_snapshots",
    }
    assert pool_stages[0]["state_snapshot_binding"] == 30
    attention_stages = causal_scan_batch_stages(
        "indexed_sparse_attention_main_bf16_q64_kv1_d512_w128_r4_k512_"
        "scale0.0441941738__sc8.comp",
        512,
    )
    assert attention_stages[0]["control"] == {
        "kind": "storage_buffer",
        "byte_count": 20,
        "binding": 8,
        "payload": "temporal_state_snapshots",
    }
    assert attention_stages[0]["state_snapshot_binding"] == 30
    assert attention_stages[0]["state_snapshot_source_binding"] == 1
    assert attention_stages[0]["dispatch_y_from_batch_width"] is True
    conv_stages = causal_scan_batch_stages(
        "causal_conv1d_silu_bf16_c8192_k4.comp",
        128,
    )
    assert conv_stages is not None
    assert conv_stages[0]["control"] == {
        "kind": "storage_buffer",
        "byte_count": 8,
        "binding": 31,
        "payload": "width_state_snapshots",
    }
    assert conv_stages[0]["state_snapshot_binding"] == 30
    attention_spec = component_kernel_spec(
        execution_index=0,
        node={"id": "attention", "op": "append_scaled_dot_product_attention"},
        circuit={},
        shader_file="append_gqa_attention_bf16_q16_kv4_d256_scale0.0625__sc7.comp",
        local_size_x=attention_local_size,
        workgroup_count_x=16,
    )
    temporal = attention_spec["batch_implementations"][0]
    assert temporal["execution_domain"] == "prefill"
    assert temporal["independent_candidate_compatible"] is False
    assert temporal["causal_sequence_compatible"] is True
    assert temporal["parallel_block_compatible"] is False


def test_stateful_causal_scans_expose_transactional_snapshots(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "causal_conv1d_silu_temporal_bf16_c8192_k4__pbc31.comp",
        "gated_delta_scan_k16x128_v32x128_af32_dtbf16_nf32_eps1e-06__pbc31.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    for shader_file in shader_files:
        source = (tmp_path / shader_file).read_text()
        assert "binding = 30) buffer StateSnapshots" in source
        assert "binding = 31) readonly buffer BatchControl" in source
        assert "uint state_snapshots_enabled;" in source
        assert "state_snapshots_enabled != 0u" in source
        assert "layout(push_constant) uniform BatchControl" not in source


def test_compiler_renders_temporal_attention_stages(tmp_path: Path) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "append_gqa_attention_temporal_read_bf16_"
        "q16_kv4_d256_scale0.0625_w32768_sinks.comp",
        "append_kv_temporal_commit_bf16_kv4_d256_w32768_sinks.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    read_source = next(
        tmp_path.glob("append_gqa_attention_temporal_read_*.comp")
    ).read_text()
    assert "layout(set = 0, binding = 6) readonly buffer KvStateRead" in read_source
    assert "layout(set = 0, binding = 8) readonly buffer BatchControl" in read_source
    assert "layout(push_constant) uniform BatchControl" not in read_source
    assert "const uint ATTENTION_WINDOW = 32768u;" in read_source
    assert "absolute_tick >= batch_control.start_stream_tick_low" in read_source
    assert "shared float tile_reduction[" in read_source
    assert "(absolute_tick % capacity) * SLOT_WORD_COUNT" in read_source
    assert "shared float tile_alpha[" not in read_source
    assert "shared float tile_beta[" not in read_source
    assert "uint query_head = gl_WorkGroupID.x % QUERY_HEADS;" in read_source
    assert "uint position = gl_WorkGroupID.x / QUERY_HEADS;" in read_source
    assert "if (position >= batch_control.batch_width) return;" in read_source
    commit_source = next(tmp_path.glob("append_kv_temporal_commit_*.comp")).read_text()
    assert "layout(set = 0, binding = 7) buffer KvStateWrite" in commit_source
    assert "layout(set = 0, binding = 8) readonly buffer BatchControl" in commit_source
    assert "layout(push_constant) uniform BatchControl" not in commit_source
    assert "const uint ATTENTION_WINDOW = 32768u;" in commit_source
    assert (
        "min(batch_control.dynamic_state_capacity, ATTENTION_WINDOW)" in commit_source
    )
    assert "position * KV_WORD_COUNT + head_word" in commit_source
    assert "{{" not in read_source
    assert "{{" not in commit_source


def test_compiler_renders_standalone_temporal_rope(tmp_path: Path) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_file = "rotary_temporal_bf16_16x256_r64_theta10000000_half__pbc2.comp"

    copy_shader_templates(shader_source_dir, tmp_path, {shader_file})

    source = (tmp_path / shader_file).read_text()
    assert "layout(set = 0, binding = 2) readonly buffer BatchControl" in source
    assert "layout(push_constant) uniform BatchControl" not in source
    assert "position < batch_control.batch_width" in source
    assert "batch_control.start_stream_tick_low + position" in source
    assert "position * FRAME_WORDS" in source
    assert "{{" not in source


def test_compiler_renders_deepseek_stateless_causal_batch_kernels(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "hyper_connection_pre_batch8_m4_h4096_i20_neps1e-06_heps1e-06__pbc31.comp",
        "hyper_connection_post_pre_batch8_m4_h4096_i20_neps1e-06_heps1e-06__pbc31.comp",
        "hyper_connection_post_batch8_m4_h4096__pbc31.comp",
        "rms_norm_per_head_unscaled_batch8_bf16_64x512_eps1e-06__pbc31.comp",
        "grouped_linear_batch8_fp8_e4m3_se8m0_b128x128_g8_32768x8192__pbc31.comp",
        "bounded_silu_multiply_batch8_bf16_2048_limit10__pbc31.comp",
        "rms_norm_quantize_in_place_batch8_fp8_e4m3_spow2_b128_h4096_"
        "eps1e-06_offset0__pbc31.comp",
        "inverse_rotary_temporal_bf16_64x512_r64_theta160000_"
        "yarn_f16_lo15_hi25_a1_interleaved_tail_po1__pbc2.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    sources = {path.name: path.read_text() for path in tmp_path.glob("*.comp")}
    assert sources.keys() == shader_files
    assert all("{{" not in source for source in sources.values())
    assert all(
        "layout(push_constant) uniform BatchControl" not in source
        for source in sources.values()
    )
    assert all("batch_control.batch_width" in source for source in sources.values())
    assert "ROPE_DIRECTION = -1.0" in next(
        source for name, source in sources.items() if name.startswith("inverse_rotary_")
    )
    assert "batch_index * HYPER_WORDS" in next(
        source
        for name, source in sources.items()
        if name.startswith("hyper_connection_pre_batch")
    )
    assert "batch_index * TOTAL_INPUT_WORDS" in next(
        source
        for name, source in sources.items()
        if name.startswith("grouped_linear_batch")
    )
    in_place_rms = next(
        source
        for name, source in sources.items()
        if name.startswith("rms_norm_quantize_in_place_batch")
    )
    assert "binding = 0) buffer OutputFrames" in in_place_rms
    assert "binding = 1) buffer QuantizedFrames" in in_place_rms
    assert "binding = 2) buffer Scales" in in_place_rms
    assert "binding = 3) readonly buffer Weight" in in_place_rms
    assert "output_frames.words[" in in_place_rms
    assert "memoryBarrierBuffer();" in in_place_rms
    compile_shader_artifacts(tmp_path)
    assert len(list(tmp_path.glob("*.spv"))) == len(shader_files)


def test_compiler_renders_deepseek_stateful_causal_scan_kernels(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "rolling_state_ring_append_temporal_bf16_128x512__pbc6.comp",
        "learned_gated_kv_pool_temporal_bf16_f32_h4096_d512_r4_c2__pbc8.comp",
        "learned_gated_kv_pool_temporal_bf16_f32_h4096_d512_r4_c1__pbc8.comp",
        "compressed_kv_finalize_temporal_f32_bf16_d512_r64_eps1e-06_"
        "theta160000_yarn_f16_lo15_hi25_a1_interleaved_po-3_"
        "qfp8e4m3b64__pbc3.comp",
        "conditional_append_state_temporal_bf16_d512_p4__pbc6.comp",
        "index_vector_transform_temporal_bf16_h64_d128_r64_theta160000_"
        "yarn_f16_lo15_hi25_a1_interleaved_qfp4e2m1b32__pbc2.comp",
        "compressed_index_kv_finalize_temporal_f32_bf16_d128_r64_eps1e-06_"
        "theta160000_yarn_f16_lo15_hi25_a1_interleaved_po-3_"
        "qfp4e2m1b32__pbc3.comp",
        "learned_index_scores_temporal_bf16_f32_h64_d128_r4_m262144_c256_"
        "scale0.0110485435__pbc5.comp",
        "radix_topk_index_temporal_f32_u32_m262144_k512_r4_o128__pbc2.comp",
        "chronological_compressed_index_temporal_u32_m8192_r128_o128__pbc3.comp",
        "indexed_sparse_attention_main_temporal_parallel_bf16_q64_kv1_d512_w128_"
        "r4_k512_scale0.0441941738__pbc8.comp",
        "indexed_sparse_attention_main_score_pipeline_temporal_parallel_bf16_"
        "q64_kv1_d512_w128_r4_k512_scale0.0441941738__pbc8.comp",
        "indexed_sparse_attention_main_tile_overlap_temporal_parallel_bf16_"
        "q64_kv1_d512_w128_r4_k512_scale0.0441941738__pbc8.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    sources = {path.name: path.read_text() for path in tmp_path.glob("*.comp")}
    assert sources.keys() == shader_files
    assert all("{{" not in source for source in sources.values())
    assert all(
        "layout(push_constant) uniform BatchControl" not in source
        for source in sources.values()
    )
    assert all("batch_control.batch_width" in source for source in sources.values())
    assert (
        "state_snapshots_enabled"
        in sources["rolling_state_ring_append_temporal_bf16_128x512__pbc6.comp"]
    )
    assert (
        "state_snapshots_enabled"
        in sources[
            "learned_gated_kv_pool_temporal_bf16_f32_h4096_d512_r4_c2__pbc8.comp"
        ]
    )
    assert (
        "if (LANE_COEFFICIENT == 2u)"
        in sources[
            "learned_gated_kv_pool_temporal_bf16_f32_h4096_d512_r4_c1__pbc8.comp"
        ]
    )
    attention = sources[
        "indexed_sparse_attention_main_temporal_parallel_bf16_q64_kv1_d512_w128_"
        "r4_k512_scale0.0441941738__pbc8.comp"
    ]
    assert "binding = 30) readonly buffer LocalStateSnapshots" in attention
    assert "batch_position * LOCAL_STATE_WORDS" in attention
    assert "uint batch_position = gl_WorkGroupID.y;" in attention
    assert "if (batch_position >= batch_control.batch_width)" in attention
    assert "for (uint batch_position" not in attention
    pipelined_attention = sources[
        "indexed_sparse_attention_main_score_pipeline_temporal_parallel_bf16_"
        "q64_kv1_d512_w128_r4_k512_scale0.0441941738__pbc8.comp"
    ]
    assert "binding = 30) readonly buffer LocalStateSnapshots" in pipelined_attention
    assert "uint batch_position = gl_WorkGroupID.y;" in pipelined_attention
    assert "uint score_subgroup_count = HEAD_WIDTH / gl_SubgroupSize;" in pipelined_attention
    assert "for (uint batch_position" not in pipelined_attention
    tile_overlapped_attention = sources[
        "indexed_sparse_attention_main_tile_overlap_temporal_parallel_bf16_"
        "q64_kv1_d512_w128_r4_k512_scale0.0441941738__pbc8.comp"
    ]
    assert (
        "binding = 30) readonly buffer LocalStateSnapshots"
        in tile_overlapped_attention
    )
    assert "uint batch_position = gl_WorkGroupID.y;" in tile_overlapped_attention
    assert "bool value_worker = invocation >= HEAD_WIDTH;" in tile_overlapped_attention
    assert "previous_tile_accumulator" in tile_overlapped_attention
    assert "for (uint batch_position" not in tile_overlapped_attention
    compile_shader_artifacts(tmp_path)
    assert len(list(tmp_path.glob("*.spv"))) == len(shader_files)


def test_temporal_lane_parallelization_fails_closed_on_template_drift() -> None:
    with pytest.raises(
        ModelCompileError,
        match="has 0 serial lane loops; expected one",
    ):
        parallelize_temporal_batch_lanes(
            "indexed_sparse_attention_main_temporal_parallel.comp",
            "void main() {}",
        )

    duplicated_loop = """    for (uint batch_position = 0u;
         batch_position < batch_control.batch_width;
         ++batch_position) {"""
    with pytest.raises(
        ModelCompileError,
        match="has 2 serial lane loops; expected one",
    ):
        parallelize_temporal_batch_lanes(
            "indexed_sparse_attention_main_temporal_parallel.comp",
            duplicated_loop + "\n}\n" + duplicated_loop,
        )


def test_compiler_lowers_component_batch_width_to_a_persistent_buffer(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_file = "linear_batch2_bf16_1024x4096__pbc31.comp"
    input_shader_file = "embedding_lookup_batch_bf16_32000x768_scale1__pbc3.comp"

    copy_shader_templates(
        shader_source_dir,
        tmp_path,
        {shader_file, input_shader_file},
    )

    source = (tmp_path / shader_file).read_text()
    assert "layout(set = 0, binding = 31) readonly buffer BatchControl" in source
    assert "layout(push_constant) uniform BatchControl" not in source
    assert "batch_control.batch_width" in source
    input_source = (tmp_path / input_shader_file).read_text()
    assert "layout(set = 0, binding = 3) readonly buffer BatchControl" in input_source
    assert "layout(push_constant) uniform BatchControl" not in input_source


def test_compiler_selects_cooperative_bfloat16_projection_kernels() -> None:
    assert (
        cooperative_bfloat16_batch_shader_file("linear_bf16_1024x4096.comp")
        == "linear_batch64_cooperative_bf16_1024x4096.comp"
    )
    assert (
        cooperative_bfloat16_batch_shader_file("linear_residual_bf16_4096x1024.comp")
        == "linear_residual_batch64_cooperative_bf16_4096x1024.comp"
    )
    assert cooperative_bfloat16_batch_shader_file(
        "parallel_linear_3way_bf16_1024x1024_256_256.comp"
    ) == ("parallel_linear_batch64_cooperative_3way_bf16_1024x1024_256_256.comp")
    assert cooperative_bfloat16_batch_shader_file(
        "parallel_linear_silu_multiply_bf16_1024x4096.comp"
    ) == ("parallel_linear_silu_multiply_batch64_cooperative_bf16_1024x4096.comp")
    assert (
        cooperative_bfloat16_workgroup_count_x(
            "parallel_linear_3way_bf16_1024x1024_256_256.comp"
        )
        == 24
    )
    assert (
        cooperative_bfloat16_workgroup_count_x(
            "parallel_linear_2way_bf16_1024x1024_256.comp"
        )
        == 20
    )
    assert (
        cooperative_bfloat16_workgroup_count_x(
            "parallel_linear_silu_multiply_bf16_1024x4096.comp"
        )
        == 64
    )
    assert cooperative_bfloat16_batch_shader_file(
        "linear_residual_int4_ct_sbf16_g128_5120x17408.comp"
    ) == ("linear_residual_batch64_cooperative_int4_ct_sbf16_g128_5120x17408.comp")
    assert cooperative_bfloat16_batch_shader_file("linear_bf16_2048x1.comp") is None
    assert (
        cooperative_bfloat16_workgroup_count_x(
            "linear_residual_int4_ct_sbf16_g128_5120x17408.comp"
        )
        == 272
    )
    assert (
        cooperative_bfloat16_batch_shader_file(
            "linear_fp8_e4m3_b128x128_1024x4096.comp"
        )
        is None
    )


def test_projection_component_compiles_ordered_target_native_and_scalar_implementations() -> (
    None
):
    spec = component_kernel_spec(
        execution_index=0,
        node={"id": "project", "op": "linear"},
        circuit={},
        shader_file="linear_bf16_1024x4096.comp",
        local_size_x=64,
        workgroup_count_x=2048,
    )

    assert spec["batch_mode"] == "weight_shared"
    assert "batch_shader_path" not in spec
    assert "batch_lane_tile_width" not in spec
    cooperative, *exact = spec["batch_implementations"]
    assert cooperative == {
        "execution_domain": "prefill",
        "lane_tile_width": 64,
        "selection_priority": 0,
        "independent_candidate_compatible": False,
        "causal_sequence_compatible": True,
        "parallel_block_compatible": False,
        "device_requirements": {
            "vulkan_device_extensions": [],
            "vulkan_features": [],
            "subgroup_operations": [],
            "cooperative_bfloat16_shape": [16, 16, 16],
            "subgroup_size": 64,
        },
        "stages": [
            {
                "shader_path": (
                    "shaders/linear_batch64_cooperative_bf16_1024x4096__pbc31.comp"
                ),
                "local_size_x": 256,
                "workgroup_count_x": 64,
                "control": {
                    "kind": "storage_buffer",
                    "byte_count": 4,
                    "binding": 31,
                    "payload": "width",
                },
            }
        ],
    }
    assert [implementation["lane_tile_width"] for implementation in exact] == [
        2,
        4,
        8,
        16,
    ]
    for implementation in exact:
        tile_width = implementation["lane_tile_width"]
        assert implementation == {
            "execution_domain": "decode_and_prefill",
            "lane_tile_width": tile_width,
            "selection_priority": 0,
            "independent_candidate_compatible": True,
            "causal_sequence_compatible": True,
            "parallel_block_compatible": True,
            "device_requirements": {
                "vulkan_device_extensions": [],
                "vulkan_features": [],
                "subgroup_operations": [],
            },
            "stages": [
                {
                    "shader_path": (
                        f"shaders/linear_batch{tile_width}_bf16_1024x4096__pbc31.comp"
                    ),
                    "local_size_x": 64,
                    "workgroup_count_x": 2048,
                    "control": {
                        "kind": "storage_buffer",
                        "byte_count": 4,
                        "binding": 31,
                        "payload": "width",
                    },
                }
            ],
        }


def test_lane_parallel_batch_kernels_expose_one_workgroup_per_lane() -> None:
    spec = component_kernel_spec(
        execution_index=0,
        node={"id": "gate", "op": "linear"},
        circuit={},
        shader_file="linear_bf16_2048x1.comp",
        local_size_x=64,
        workgroup_count_x=1,
    )

    assert spec["batch_mode"] == "weight_shared"
    assert [
        implementation["lane_tile_width"]
        for implementation in spec["batch_implementations"]
    ] == [2, 4, 8, 16]
    for implementation in spec["batch_implementations"]:
        tile_width = implementation["lane_tile_width"]
        assert implementation["stages"][0]["workgroup_count_x"] == tile_width


def test_compiler_selects_device_typed_cooperative_fp8_prefill() -> None:
    spec = component_kernel_spec(
        execution_index=0,
        node={"id": "down", "op": "linear_residual"},
        circuit={},
        shader_file=("linear_residual_prequant_fp8_e4m3_b128x128_17408x5120.comp"),
        local_size_x=1024,
        workgroup_count_x=320,
        cooperative_float8_e4m3_shapes=((16, 16, 16),),
    )

    cooperative, *exact = spec["batch_implementations"]
    assert cooperative == {
        "execution_domain": "prefill",
        "lane_tile_width": 64,
        "selection_priority": 0,
        "independent_candidate_compatible": False,
        "causal_sequence_compatible": True,
        "parallel_block_compatible": False,
        "device_requirements": {
            "vulkan_device_extensions": [],
            "vulkan_features": [],
            "subgroup_operations": [],
            "cooperative_float8_e4m3_shape": [16, 16, 16],
            "subgroup_size": 64,
        },
        "stages": [
            {
                "shader_path": (
                    "shaders/linear_residual_prequant_batch64_cooperative_"
                    "fp8_e4m3_m16n16k16_b128x128_17408x5120__pbc31.comp"
                ),
                "local_size_x": 256,
                "workgroup_count_x": 80,
                "control": {
                    "kind": "storage_buffer",
                    "byte_count": 4,
                    "binding": 31,
                    "payload": "width",
                },
            }
        ],
    }


def test_compiler_prefers_compact_cooperative_fp8_only_for_occupied_decode_grid() -> None:
    spec = component_kernel_spec(
        execution_index=0,
        node={"id": "project", "op": "linear"},
        circuit={},
        shader_file="linear_prequant_fp8_e4m3_b128x128_1024x32768.comp",
        local_size_x=1024,
        workgroup_count_x=2048,
        cooperative_float8_e4m3_shapes=((16, 16, 16),),
    )

    compact, wide, *exact = spec["batch_implementations"]
    assert compact["execution_domain"] == "decode_and_prefill"
    assert compact["lane_tile_width"] == 16
    assert compact["selection_priority"] == 1
    assert compact["independent_candidate_compatible"] is True
    assert compact["causal_sequence_compatible"] is True
    assert compact["parallel_block_compatible"] is True
    assert compact["stages"] == [
        {
            "shader_path": (
                "shaders/linear_prequant_batch16_cooperative_fp8_e4m3_"
                "m16n16k16_b128x128_1024x32768__pbc31.comp"
            ),
            "local_size_x": 256,
            "workgroup_count_x": 512,
            "control": {
                "kind": "storage_buffer",
                "byte_count": 4,
                "binding": 31,
                "payload": "width",
            },
        }
    ]
    assert wide["execution_domain"] == "prefill"
    assert wide["lane_tile_width"] == 64
    assert wide["selection_priority"] == 0
    assert [implementation["lane_tile_width"] for implementation in exact] == [
        2,
        4,
        8,
        16,
    ]
    assert all(implementation["selection_priority"] == 0 for implementation in exact)


def test_mixed_parallel_projection_uses_fused_decode_and_split_prefill() -> None:
    shader_file = (
        "mixed_parallel_linear_4way_prequant_fp8_e4m3_"
        "b128x128_bf16_2048x8192_4096_32_32.comp"
    )
    spec = component_kernel_spec(
        execution_index=1,
        node={
            "id": "mixed_projection",
            "op": "mixed_parallel_linear_4way",
            "attrs": {"compiled_from": ["a", "b", "c", "d"]},
        },
        circuit={},
        shader_file=shader_file,
        local_size_x=1024,
        workgroup_count_x=512,
        cooperative_float8_e4m3_shapes=((16, 16, 16),),
    )

    cooperative, *exact = spec["batch_implementations"]
    assert cooperative["execution_domain"] == "prefill"
    assert cooperative["lane_tile_width"] == 64
    assert len(cooperative["stages"]) == 2
    assert cooperative["stages"][0]["descriptor_bindings"] == [
        {"binding": 0, "source_binding": 0},
        {"binding": 1, "source_binding": 1},
        {"binding": 2, "source_binding": 3},
        {"binding": 3, "source_binding": 4},
        {"binding": 4, "source_binding": 7},
        {"binding": 5, "source_binding": 8},
        {"binding": 6, "source_binding": 9},
        {"binding": 7, "source_binding": 10},
    ]
    assert cooperative["stages"][1]["descriptor_bindings"] == [
        {"binding": 0, "source_binding": 2},
        {"binding": 1, "source_binding": 5},
        {"binding": 2, "source_binding": 6},
        {"binding": 3, "source_binding": 11},
        {"binding": 4, "source_binding": 12},
    ]
    assert [implementation["lane_tile_width"] for implementation in exact] == [
        2,
        4,
        8,
        16,
    ]
    assert all(
        implementation["stages"][0]["workgroup_count_x"] == 512
        for implementation in exact
    )
    assert [implementation["lane_tile_width"] for implementation in exact] == [
        2,
        4,
        8,
        16,
    ]


def test_compiler_renders_cooperative_fp8_prefill_shader(tmp_path: Path) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "linear_prequant_batch64_cooperative_"
        "fp8_e4m3_m16n16k16_b128x128_5120x17408.comp",
        "linear_residual_prequant_batch64_cooperative_"
        "fp8_e4m3_m16n16k16_b128x128_17408x5120.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    linear = next(
        (tmp_path / name).read_text()
        for name in shader_files
        if name.startswith("linear_prequant_")
    )
    residual = next(
        (tmp_path / name).read_text()
        for name in shader_files
        if name.startswith("linear_residual_")
    )
    for source in (linear, residual):
        assert "coopmat<floate4m3_t" in source
        assert "const uint MATRIX_M = 16u;" in source
        assert "const uint MATRIX_N = 16u;" in source
        assert "const uint MATRIX_K = 16u;" in source
        assert "const uint BLOCK_COLUMNS = 128u;" in source
        assert "shared float result_tile[OUTPUT_TILE * BATCH_TILE];" in source
        assert "{{" not in source
    assert "binding = 2) buffer OutputFrames" in linear
    assert "binding = 2) readonly buffer ResidualFrames" in residual
    assert "binding = 3) buffer OutputFrames" in residual


def test_compiler_renders_cooperative_parallel_fp8_prefill_shaders(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    parallel = (
        "parallel_linear_batch64_3way_prequant_cooperative_"
        "fp8_e4m3_m16n16k16_b128x128_5120x6144_1024_1024.comp"
    )
    fused_ffn = (
        "parallel_linear_silu_multiply_prequant_batch64_cooperative_"
        "fp8_e4m3_m16n16k16_b128x128_5120x17408.comp"
    )

    copy_shader_templates(shader_source_dir, tmp_path, {parallel, fused_ffn})

    parallel_source = (tmp_path / parallel).read_text()
    assert "const uint BRANCH_COUNT = 3u;" in parallel_source
    assert "const uint OUTPUT_A_SIZE = 6144u;" in parallel_source
    assert "const uint OUTPUT_B_SIZE = 1024u;" in parallel_source
    assert "const uint OUTPUT_C_SIZE = 1024u;" in parallel_source
    assert "binding = 2) buffer OutputA" in parallel_source
    assert "binding = 4) buffer OutputC" in parallel_source
    assert "binding = 10) readonly buffer WeightScaleInvC" in parallel_source
    assert "weight_a.values," in parallel_source
    assert "weight_b.values," in parallel_source
    assert "weight_c.values," in parallel_source
    assert "bool stage_weight" in parallel_source
    assert "output_start + OUTPUT_TILE" in parallel_source
    assert "coopmat<floate4m3_t" in parallel_source
    assert "{{" not in parallel_source

    fused_source = (tmp_path / fused_ffn).read_text()
    assert "shared float result_tiles[2]" in fused_source
    assert "const uint OUTPUT_SIZE = 17408u;" in fused_source
    assert "binding = 3) readonly buffer GateWeight" in fused_source
    assert "binding = 5) readonly buffer UpWeight" in fused_source
    assert "gate_weight.values," in fused_source
    assert "up_weight.values," in fused_source
    assert "bool stage_weight" in fused_source
    assert "output_start + OUTPUT_TILE > OUTPUT_SIZE" in fused_source
    assert "rounded_silu(gate) * up" in fused_source
    assert "coopmat<floate4m3_t" in fused_source
    assert "{{" not in fused_source


def test_compiler_renders_weight_shared_component_batch_shaders(tmp_path: Path) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "rms_norm_batch16_bf16_h5120_eps1e-06_offset1.comp",
        "linear_batch16_fp8_e4m3_b128x128_5120x17408.comp",
        "linear_residual_batch16_fp8_e4m3_b128x128_17408x5120.comp",
        "parallel_linear_batch16_2way_bf16_1024x2560_2560.comp",
        "parallel_linear_silu_multiply_batch16_fp8_e4m3_b128x128_5120x17408.comp",
        "linear_batch16_bf16_1024x4096.comp",
        "linear_batch16_bf16_2048x1.comp",
        "linear_residual_batch16_bf16_4096x1024.comp",
        "parallel_linear_silu_multiply_batch16_bf16_1024x4096.comp",
        "split_batch16_bf16_2x16x256_head_interleaved.comp",
        "split_batch16_bf16_2x512.comp",
        "silu_multiply_batch16_bf16_512.comp",
        "sigmoid_scalar_multiply_batch16_bf16_2048.comp",
        "linear_sigmoid_scalar_multiply_batch16_bf16_2048x2048.comp",
        "linear_sigmoid_scalar_multiply_residual2_batch16_bf16_2048x2048.comp",
        "add_batch16_bf16_2048.comp",
        "sigmoid_multiply_batch16_bf16.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    odd_linear_source = (tmp_path / "linear_batch16_bf16_2048x1.comp").read_text()
    assert "uint16_t values[]" in odd_linear_source
    assert "batch_index * OUTPUT_SIZE" in odd_linear_source
    assert "output_frames.words" not in odd_linear_source
    assert "gl_WorkGroupID.x / BATCH_TILE_WIDTH" in odd_linear_source
    even_linear_source = (tmp_path / "linear_batch16_bf16_1024x4096.comp").read_text()
    assert "uint words[]" in even_linear_source
    assert "output_frames.words" in even_linear_source
    scalar_gate_source = (
        tmp_path / "sigmoid_scalar_multiply_batch16_bf16_2048.comp"
    ).read_text()
    assert "uint16_t values[]" in scalar_gate_source
    assert "gate_logits.values[batch_index]" in scalar_gate_source
    assert "uint batch_lane = gl_WorkGroupID.x;" in scalar_gate_source
    fused_scalar_gate_source = (
        tmp_path / "linear_sigmoid_scalar_multiply_batch16_bf16_2048x2048.comp"
    ).read_text()
    assert "float rounded_gate = bf16_to_f32(f32_to_bf16(gate_sum));" in (
        fused_scalar_gate_source
    )
    assert "uint batch_lane = gl_WorkGroupID.x;" in fused_scalar_gate_source
    fused_scalar_gate_residual_source = (
        tmp_path
        / "linear_sigmoid_scalar_multiply_residual2_batch16_bf16_2048x2048.comp"
    ).read_text()
    assert "uint combined_lo = f32_to_bf16(" in (fused_scalar_gate_residual_source)
    assert "bf16_to_f32(combined_lo) + bf16_to_f32(second)" in (
        fused_scalar_gate_residual_source
    )

    for shader_file in shader_files:
        source = (tmp_path / shader_file).read_text()
        assert "const uint BATCH_TILE_WIDTH = 16u;" in source
        assert "layout(push_constant) uniform BatchControl" in source
        assert "gl_WorkGroupID.y * BATCH_TILE_WIDTH" in source
        if "fp8_e4m3" in shader_file:
            assert "#extension GL_EXT_float_e4m3 : require" in source
            assert "uintBitsToFloate4m3EXT" in source
            assert "SPV_VALVE_mixed_float_dot_product" in source
            assert "fp8_dot4_acc32" in source
            assert "shared fe4m3vec4 quantized_input" in source
        assert "{{" not in source
    head_interleaved_split = (
        tmp_path / "split_batch16_bf16_2x16x256_head_interleaved.comp"
    ).read_text()
    assert "layout(local_size_x = 256" in head_interleaved_split
    assert "uint output_word = gl_GlobalInvocationID.x;" in head_interleaved_split
    assert "gl_NumWorkGroups.x * gl_WorkGroupSize.x" in head_interleaved_split
    assert required_vulkan_device_extensions(tmp_path, shader_files) == [
        "VK_EXT_shader_float8",
        "VK_VALVE_shader_mixed_float_dot_product",
    ]


def test_compiler_renders_position_aware_temporal_head_norm_rope(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_file = (
        "parallel_head_norm_rope_2way_temporal_bf16_h16_4_d256_r64_"
        "eps1e-06_offset1_theta10000000_half.comp"
    )

    copy_shader_templates(shader_source_dir, tmp_path, {shader_file})

    source = (tmp_path / shader_file).read_text()
    assert "layout(set = 0, binding = 6) readonly buffer BatchControl" in source
    assert "layout(push_constant) uniform BatchControl" not in source
    assert "uint start_stream_tick_low;" in source
    assert "position < batch_control.batch_width" in source
    assert "start_stream_tick_low + position" in source
    assert "StreamControl" not in source
    assert "{{" not in source


def test_compiler_renders_cooperative_bfloat16_batch_shaders(tmp_path: Path) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "linear_batch64_cooperative_bf16_1024x4096.comp",
        "linear_residual_batch64_cooperative_bf16_4096x1024.comp",
        "parallel_linear_batch64_cooperative_3way_bf16_1024x1024_256_256.comp",
        "parallel_linear_silu_multiply_batch64_cooperative_bf16_1024x4096.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    for shader_file in shader_files:
        source = (tmp_path / shader_file).read_text()
        assert "coopMatMulAdd" in source
        assert "coopmat<bfloat16_t" in source
        assert "#extension GL_EXT_bfloat16 : require" in source
        assert "#extension GL_KHR_cooperative_matrix : require" in source
        assert "layout(local_size_x = 256" in source
        expected_output_tile = 64
        assert f"const uint OUTPUT_TILE = {expected_output_tile}u;" in source
        assert "const uint BATCH_TILE = 64u;" in source
        expected_result_tile = (
            "BRANCH_COUNT * OUTPUT_TILE * BATCH_TILE"
            if "silu_multiply" in shader_file
            else "OUTPUT_TILE * MATRIX_TILE"
        )
        assert f"shared float result_tile[{expected_result_tile}];" in source
        assert "{{" not in source
    direct_linear = (
        tmp_path / "linear_residual_batch64_cooperative_bf16_4096x1024.comp"
    ).read_text()
    direct_parallel = (
        tmp_path / "parallel_linear_batch64_cooperative_3way_bf16_"
        "1024x1024_256_256.comp"
    ).read_text()
    direct_fused = (
        tmp_path / "parallel_linear_silu_multiply_batch64_cooperative_bf16_"
        "1024x4096.comp"
    ).read_text()
    assert "weight.values," in direct_linear
    assert "weight_a.values," in direct_parallel
    assert "weight_b.values," in direct_parallel
    assert "weight_c.values," in direct_parallel
    assert "gate_weight.values," in direct_fused
    assert "up_weight.values," in direct_fused
    assert "const uint BRANCH_SUBGROUPS = 2u;" in direct_fused
    assert "sums[OUTPUT_SUBTILES_PER_SUBGROUP * BATCH_SUBTILES]" in direct_fused
    assert "branch * OUTPUT_TILE * BATCH_TILE" in direct_fused
    assert "coopmat<bfloat16_t" in direct_linear
    assert "uintBitsToBFloat16EXT(uint16_t(f32_to_bf16" in direct_linear
    assert "residual_frames.values," in direct_linear
    assert "gl_CooperativeMatrixLayoutColumnMajor" in direct_linear
    assert "uintBitsToBFloat16EXT" in direct_parallel
    assert "gl_CooperativeMatrixLayoutColumnMajor" in direct_parallel
    assert required_vulkan_device_extensions(tmp_path, shader_files) == [
        "VK_KHR_cooperative_matrix",
        "VK_KHR_shader_bfloat16",
    ]


def test_compiler_renders_cooperative_int4_prefill_shaders(tmp_path: Path) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    gptq = "linear_bias_batch64_cooperative_int4_gptq_sf16_g128_5120x17408.comp"
    ct = "linear_residual_batch64_cooperative_int4_ct_sbf16_g128_17408x5120.comp"

    copy_shader_templates(shader_source_dir, tmp_path, {gptq, ct})

    gptq_source = (tmp_path / gptq).read_text()
    assert "binding = 2) readonly buffer QWeight" in gptq_source
    assert "binding = 3) readonly buffer Scales" in gptq_source
    assert "binding = 4) readonly buffer Bias" in gptq_source
    assert "uint index = group * OUTPUT_SIZE + row;" in gptq_source
    assert "QZeros" not in gptq_source
    assert "int zero =" not in gptq_source
    assert "& 15u) - 8;" in gptq_source
    assert "coopmat<bfloat16_t" in gptq_source
    assert "{{" not in gptq_source

    ct_source = (tmp_path / ct).read_text()
    assert "binding = 1) readonly buffer ResidualFrames" in ct_source
    assert "binding = 3) readonly buffer QWeight" in ct_source
    assert "binding = 4) readonly buffer Scales" in ct_source
    assert "QZeros" not in ct_source
    assert "uint index = row * SCALE_COLUMNS + group;" in ct_source
    assert "result_tile[index] = fma(" in ct_source
    assert "coopmat<bfloat16_t" in ct_source
    assert "{{" not in ct_source


@pytest.mark.parametrize(
    "head_width",
    [32, 64, 80, 96, 128, 192, 256, 320, 384, 512, 768, 1024],
)
def test_attention_tile_stays_within_portable_shared_memory_budget(
    head_width: int,
) -> None:
    local_size, tile_tokens = attention_workgroup_shape(head_width)
    padded_head_width = ((head_width + 63) // 64) * 64
    physical_tile_tokens = 1024 // padded_head_width
    shared_floats = (
        2 * head_width
        + tile_tokens * ((head_width + 31) // 32)
        + 3 * tile_tokens
        + tile_tokens * head_width
        + 4
    )

    assert local_size == padded_head_width * physical_tile_tokens
    assert local_size <= 1024
    assert tile_tokens > physical_tile_tokens
    assert shared_floats * 4 <= 32 * 1024
