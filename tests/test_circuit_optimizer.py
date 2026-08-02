from __future__ import annotations

import unittest

from nerve.circuit_optimizer import optimize_circuit_for_vulkan


class VulkanCircuitOptimizerTest(unittest.TestCase):
    def test_fuses_kv_append_into_attention_read(self) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "append",
                    "op": "append_state_update",
                    "inputs": ["k", "v", "kv_memory"],
                    "outputs": ["k_memory", "v_memory"],
                    "state_reads": ["kv_memory"],
                    "state_writes": ["kv_memory"],
                    "attrs": {"growth": "per_activation"},
                },
                {
                    "id": "attention",
                    "op": "scaled_dot_product_attention",
                    "inputs": ["q", "k_memory", "v_memory"],
                    "outputs": ["attention_out"],
                    "params": ["attention_sinks"],
                    "attrs": {"causal": True},
                },
            ]
        }

        optimized = optimize_circuit_for_vulkan(
            circuit,
            can_fuse_append_attention=lambda append, attention: (
                append["outputs"] == attention["inputs"][1:]
            ),
        )

        self.assertEqual(1, len(optimized["nodes"]))
        fused = optimized["nodes"][0]
        self.assertEqual("append_scaled_dot_product_attention", fused["op"])
        self.assertEqual(["q", "k", "v", "kv_memory"], fused["inputs"])
        self.assertEqual(["attention_out"], fused["outputs"])
        self.assertEqual(["attention_sinks"], fused["params"])
        self.assertEqual(["kv_memory"], fused["state_reads"])
        self.assertEqual(["kv_memory"], fused["state_writes"])
        self.assertEqual("direct_bf16_input", fused["attrs"]["current_kv_source"])

    def test_does_not_fuse_kv_append_with_shared_state_view(self) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "append",
                    "op": "append_state_update",
                    "inputs": ["k", "v", "kv_memory"],
                    "outputs": ["k_memory", "v_memory"],
                    "state_reads": ["kv_memory"],
                    "state_writes": ["kv_memory"],
                },
                {
                    "id": "attention",
                    "op": "scaled_dot_product_attention",
                    "inputs": ["q", "k_memory", "v_memory"],
                    "outputs": ["attention_out"],
                },
                {
                    "id": "extra",
                    "op": "silu",
                    "inputs": ["k_memory"],
                    "outputs": ["extra_out"],
                },
            ]
        }

        optimized = optimize_circuit_for_vulkan(
            circuit,
            can_fuse_append_attention=lambda _append, _attention: True,
        )

        self.assertEqual(circuit, optimized)

    def test_does_not_fuse_kv_append_exposed_at_circuit_boundary(self) -> None:
        circuit = {
            "boundary": {
                "outputs": [
                    {"id": "exported_k", "source": "k_memory"},
                ]
            },
            "nodes": [
                {
                    "id": "append",
                    "op": "append_state_update",
                    "inputs": ["k", "v", "kv_memory"],
                    "outputs": ["k_memory", "v_memory"],
                    "state_reads": ["kv_memory"],
                    "state_writes": ["kv_memory"],
                },
                {
                    "id": "attention",
                    "op": "scaled_dot_product_attention",
                    "inputs": ["q", "k_memory", "v_memory"],
                    "outputs": ["attention_out"],
                },
            ],
        }

        optimized = optimize_circuit_for_vulkan(
            circuit,
            can_fuse_append_attention=lambda _append, _attention: True,
        )

        self.assertEqual(circuit, optimized)

    def test_fuses_three_way_projection_into_recurrent_depthwise_gate(self) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "project__split",
                    "op": "linear_split_3way",
                    "inputs": ["normalized"],
                    "outputs": ["gate_b", "gate_c", "projected"],
                    "params": ["projection_weight"],
                    "attrs": {
                        "part_widths": [16, 16, 16],
                        "compiled_from": ["project", "split"],
                    },
                },
                {
                    "id": "recurrent",
                    "op": "multiply_rolling_depthwise_gate",
                    "inputs": ["gate_b", "projected", "memory", "gate_c"],
                    "outputs": ["gated_conv"],
                    "params": ["conv_kernel"],
                    "state_reads": ["memory"],
                    "state_writes": ["memory"],
                    "attrs": {"compiled_from": ["gate", "shift", "conv"]},
                },
            ]
        }

        optimized = optimize_circuit_for_vulkan(
            circuit,
            can_fuse_linear_split_recurrent=lambda projection, recurrent: (
                set(projection["outputs"]).issubset(recurrent["inputs"])
            ),
        )

        self.assertEqual(1, len(optimized["nodes"]))
        fused = optimized["nodes"][0]
        self.assertEqual("linear_split_recurrent_depthwise_gate", fused["op"])
        self.assertEqual(["normalized", "memory"], fused["inputs"])
        self.assertEqual(["gated_conv"], fused["outputs"])
        self.assertEqual(
            ["projection_weight", "conv_kernel"], fused["params"]
        )
        self.assertEqual([0, 2], fused["attrs"]["input_gate_branch_indices"])
        self.assertEqual(1, fused["attrs"]["output_gate_branch_index"])
        self.assertEqual("BF16", fused["attrs"]["projection_rounding"])

    def test_does_not_fuse_three_way_projection_with_shared_branch(self) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "project__split",
                    "op": "linear_split_3way",
                    "inputs": ["normalized"],
                    "outputs": ["gate_b", "gate_c", "projected"],
                    "params": ["projection_weight"],
                },
                {
                    "id": "recurrent",
                    "op": "multiply_rolling_depthwise_gate",
                    "inputs": ["gate_b", "projected", "memory", "gate_c"],
                    "outputs": ["gated_conv"],
                    "params": ["conv_kernel"],
                    "state_reads": ["memory"],
                    "state_writes": ["memory"],
                },
                {
                    "id": "extra",
                    "op": "silu",
                    "inputs": ["gate_c"],
                    "outputs": ["extra_out"],
                },
            ]
        }

        optimized = optimize_circuit_for_vulkan(
            circuit,
            can_fuse_linear_split_recurrent=lambda _projection, _recurrent: True,
        )

        self.assertEqual(circuit, optimized)

    def test_fuses_recurrent_depthwise_result_into_output_gate(self) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "recurrent",
                    "op": "multiply_rolling_depthwise",
                    "inputs": ["gate_b", "projected", "memory"],
                    "outputs": ["conv_out"],
                    "params": ["kernel"],
                    "state_reads": ["memory"],
                    "state_writes": ["memory"],
                    "attrs": {"compiled_from": ["multiply", "shift", "conv"]},
                },
                {
                    "id": "output_gate",
                    "op": "multiply",
                    "inputs": ["gate_c", "conv_out"],
                    "outputs": ["gated_conv"],
                },
            ]
        }

        optimized = optimize_circuit_for_vulkan(
            circuit,
            can_fuse_recurrent_output_gate=lambda recurrent, gate: (
                recurrent["outputs"][0] in gate["inputs"]
            ),
        )

        self.assertEqual(1, len(optimized["nodes"]))
        fused = optimized["nodes"][0]
        self.assertEqual("multiply_rolling_depthwise_gate", fused["op"])
        self.assertEqual(
            ["gate_b", "projected", "memory", "gate_c"], fused["inputs"]
        )
        self.assertEqual(["gated_conv"], fused["outputs"])
        self.assertEqual(["kernel"], fused["params"])
        self.assertEqual(["memory"], fused["state_reads"])
        self.assertEqual("BF16", fused["attrs"]["output_gate_rounding"])
        self.assertEqual(
            ["multiply", "shift", "conv", "output_gate"],
            fused["attrs"]["compiled_from"],
        )

    def test_does_not_fuse_shared_recurrent_output_into_gate(self) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "recurrent",
                    "op": "multiply_rolling_depthwise",
                    "inputs": ["gate_b", "projected", "memory"],
                    "outputs": ["conv_out"],
                    "params": ["kernel"],
                    "state_reads": ["memory"],
                    "state_writes": ["memory"],
                },
                {
                    "id": "output_gate",
                    "op": "multiply",
                    "inputs": ["gate_c", "conv_out"],
                    "outputs": ["gated_conv"],
                },
                {
                    "id": "extra",
                    "op": "silu",
                    "inputs": ["conv_out"],
                    "outputs": ["extra_out"],
                },
            ]
        }

        optimized = optimize_circuit_for_vulkan(
            circuit,
            can_fuse_recurrent_output_gate=lambda _recurrent, _gate: True,
        )

        self.assertEqual(circuit, optimized)

    def test_fuses_multiply_rolling_state_and_depthwise_convolution(self) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "gate",
                    "op": "multiply",
                    "inputs": ["gate_value", "projected"],
                    "outputs": ["gated"],
                },
                {
                    "id": "shift",
                    "op": "rolling_state_update",
                    "inputs": ["gated", "temporal_memory"],
                    "outputs": ["window"],
                    "state_reads": ["temporal_memory"],
                    "state_writes": ["temporal_memory"],
                },
                {
                    "id": "convolve",
                    "op": "depthwise_conv1d",
                    "inputs": ["window"],
                    "outputs": ["convolved"],
                    "params": ["kernel"],
                },
            ]
        }

        optimized = optimize_circuit_for_vulkan(
            circuit,
            can_fuse_multiply_rolling_depthwise=lambda multiply, rolling, depthwise: (
                multiply["outputs"] == rolling["inputs"][:1]
                and rolling["outputs"] == depthwise["inputs"]
            ),
        )

        self.assertEqual(1, len(optimized["nodes"]))
        fused = optimized["nodes"][0]
        self.assertEqual("multiply_rolling_depthwise", fused["op"])
        self.assertEqual(
            ["gate_value", "projected", "temporal_memory"], fused["inputs"]
        )
        self.assertEqual(["convolved"], fused["outputs"])
        self.assertEqual(["kernel"], fused["params"])
        self.assertEqual(["temporal_memory"], fused["state_reads"])
        self.assertEqual(["temporal_memory"], fused["state_writes"])
        self.assertEqual("BF16", fused["attrs"]["intermediate_rounding"])

    def test_does_not_fuse_multiply_rolling_state_with_shared_window(self) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "gate",
                    "op": "multiply",
                    "inputs": ["gate_value", "projected"],
                    "outputs": ["gated"],
                },
                {
                    "id": "shift",
                    "op": "rolling_state_update",
                    "inputs": ["gated", "temporal_memory"],
                    "outputs": ["window"],
                    "state_reads": ["temporal_memory"],
                    "state_writes": ["temporal_memory"],
                },
                {
                    "id": "convolve",
                    "op": "depthwise_conv1d",
                    "inputs": ["window"],
                    "outputs": ["convolved"],
                    "params": ["kernel"],
                },
                {
                    "id": "extra",
                    "op": "silu",
                    "inputs": ["window"],
                    "outputs": ["extra_out"],
                },
            ]
        }

        optimized = optimize_circuit_for_vulkan(
            circuit,
            can_fuse_multiply_rolling_depthwise=lambda _multiply, _rolling, _depthwise: True,
        )

        self.assertEqual(circuit, optimized)

    def test_preserves_parallel_linear_to_silu_multiply_precision_boundary(self) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "gate__up",
                    "op": "parallel_linear_2way",
                    "inputs": ["hidden"],
                    "outputs": ["gate", "up"],
                    "params": ["gate_weight", "up_weight"],
                    "attrs": {"branch_count": 2},
                },
                {
                    "id": "activate__multiply",
                    "op": "silu_multiply",
                    "inputs": ["gate", "up"],
                    "outputs": ["ffn_hidden"],
                    "attrs": {"intermediate_rounding": "BF16"},
                },
            ]
        }

        optimized = optimize_circuit_for_vulkan(circuit)

        self.assertEqual(circuit, optimized)

    def test_fuses_parallel_ffn_projection_activation_when_backend_supports_it(
        self,
    ) -> None:
        circuit = {
            "boundary": {
                "inputs": [{"id": "input", "source": "hidden"}],
                "outputs": [{"id": "output", "source": "ffn_hidden"}],
            },
            "nodes": [
                {
                    "id": "gate",
                    "op": "linear",
                    "inputs": ["hidden"],
                    "outputs": ["gate_projection"],
                    "params": ["gate_weight"],
                },
                {
                    "id": "up",
                    "op": "linear",
                    "inputs": ["hidden"],
                    "outputs": ["up_projection"],
                    "params": ["up_weight"],
                },
                {
                    "id": "activation",
                    "op": "silu",
                    "inputs": ["gate_projection"],
                    "outputs": ["activated"],
                    "attrs": {"element_count": 2560},
                },
                {
                    "id": "multiply",
                    "op": "multiply",
                    "inputs": ["activated", "up_projection"],
                    "outputs": ["ffn_hidden"],
                },
            ],
        }

        optimized = optimize_circuit_for_vulkan(
            circuit,
            can_fuse_parallel_linears=lambda group: len(group) == 2,
            can_fuse_parallel_linear_silu_multiply=lambda _projection, _activation: True,
        )

        self.assertEqual(1, len(optimized["nodes"]))
        fused = optimized["nodes"][0]
        self.assertEqual("parallel_linear_silu_multiply", fused["op"])
        self.assertEqual(["hidden"], fused["inputs"])
        self.assertEqual(["ffn_hidden"], fused["outputs"])
        self.assertEqual(["gate_weight", "up_weight"], fused["params"])
        self.assertEqual(
            ["gate", "up", "activation", "multiply"],
            fused["attrs"]["compiled_from"],
        )
        self.assertEqual("BF16", fused["attrs"]["intermediate_rounding"])
        self.assertEqual(2560, fused["attrs"]["element_count"])

    def test_fuses_block_scaled_fp8_ffn_without_intermediate_parallel_node(
        self,
    ) -> None:
        circuit = {
            "boundary": {
                "inputs": [{"id": "input", "source": "hidden"}],
                "outputs": [{"id": "output", "source": "ffn_hidden"}],
            },
            "nodes": [
                {
                    "id": "gate",
                    "op": "linear",
                    "inputs": ["hidden"],
                    "outputs": ["gate_projection"],
                    "params": ["gate_weight", "gate_weight_scale_inv"],
                },
                {
                    "id": "up",
                    "op": "linear",
                    "inputs": ["hidden"],
                    "outputs": ["up_projection"],
                    "params": ["up_weight", "up_weight_scale_inv"],
                },
                {
                    "id": "activation",
                    "op": "silu",
                    "inputs": ["gate_projection"],
                    "outputs": ["activated"],
                    "attrs": {"element_count": 2560},
                },
                {
                    "id": "multiply",
                    "op": "multiply",
                    "inputs": ["activated", "up_projection"],
                    "outputs": ["ffn_hidden"],
                },
            ],
        }

        optimized = optimize_circuit_for_vulkan(
            circuit,
            can_fuse_parallel_linear_silu_multiply=lambda _projection, _activation: True,
        )

        self.assertEqual(1, len(optimized["nodes"]))
        fused = optimized["nodes"][0]
        self.assertEqual("parallel_linear_silu_multiply", fused["op"])
        self.assertEqual(
            [
                "gate_weight",
                "gate_weight_scale_inv",
                "up_weight",
                "up_weight_scale_inv",
            ],
            fused["params"],
        )
        self.assertEqual(
            ["gate", "up", "activation", "multiply"],
            fused["attrs"]["compiled_from"],
        )

    def test_fuses_parallel_head_norm_rope_branches_across_independent_nodes(
        self,
    ) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "first_norm",
                    "op": "rms_norm_per_head",
                    "inputs": ["first_projected"],
                    "outputs": ["first_normed"],
                    "params": ["first_weight"],
                    "attrs": {"head_count": 8},
                },
                {
                    "id": "second_norm",
                    "op": "rms_norm_per_head",
                    "inputs": ["second_projected"],
                    "outputs": ["second_normed"],
                    "params": ["second_weight"],
                    "attrs": {"head_count": 2},
                },
                {
                    "id": "first_rope",
                    "op": "rotary_position_embedding",
                    "inputs": ["first_normed"],
                    "outputs": ["first_positioned"],
                    "attrs": {"head_count": 8},
                },
                {
                    "id": "second_rope",
                    "op": "rotary_position_embedding",
                    "inputs": ["second_normed"],
                    "outputs": ["second_positioned"],
                    "attrs": {"head_count": 2},
                },
            ]
        }

        optimized = optimize_circuit_for_vulkan(
            circuit,
            can_fuse_parallel_head_norm_rope=lambda branches: len(branches) == 2,
        )

        self.assertEqual(1, len(optimized["nodes"]))
        fused = optimized["nodes"][0]
        self.assertEqual("parallel_head_norm_rope_2way", fused["op"])
        self.assertEqual(
            ["first_projected", "second_projected"], fused["inputs"]
        )
        self.assertEqual(
            ["first_positioned", "second_positioned"], fused["outputs"]
        )
        self.assertEqual(["first_weight", "second_weight"], fused["params"])
        self.assertEqual("BF16", fused["attrs"]["intermediate_rounding"])

    def test_does_not_fuse_head_norm_with_multiple_consumers(self) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "first_norm",
                    "op": "rms_norm_per_head",
                    "inputs": ["first_projected"],
                    "outputs": ["first_normed"],
                    "params": ["first_weight"],
                },
                {
                    "id": "second_norm",
                    "op": "rms_norm_per_head",
                    "inputs": ["second_projected"],
                    "outputs": ["second_normed"],
                    "params": ["second_weight"],
                },
                {
                    "id": "extra_consumer",
                    "op": "silu",
                    "inputs": ["first_normed"],
                    "outputs": ["extra"],
                },
                {
                    "id": "first_rope",
                    "op": "rotary_position_embedding",
                    "inputs": ["first_normed"],
                    "outputs": ["first_positioned"],
                },
                {
                    "id": "second_rope",
                    "op": "rotary_position_embedding",
                    "inputs": ["second_normed"],
                    "outputs": ["second_positioned"],
                },
            ]
        }

        optimized = optimize_circuit_for_vulkan(
            circuit,
            can_fuse_parallel_head_norm_rope=lambda _branches: True,
        )

        self.assertEqual(circuit, optimized)

    def test_fuses_two_or_three_independent_linears_with_one_input(self) -> None:
        nodes = [
            {
                "id": branch,
                "op": "linear",
                "inputs": ["hidden"],
                "outputs": [f"{branch}_out"],
                "params": [f"{branch}_weight"],
            }
            for branch in ("a", "b", "c")
        ]

        optimized = optimize_circuit_for_vulkan(
            {"nodes": nodes},
            can_fuse_parallel_linears=lambda group: len(group) in {2, 3},
        )

        self.assertEqual(1, len(optimized["nodes"]))
        fused = optimized["nodes"][0]
        self.assertEqual("parallel_linear_3way", fused["op"])
        self.assertEqual(["hidden"], fused["inputs"])
        self.assertEqual(["a_out", "b_out", "c_out"], fused["outputs"])
        self.assertEqual(
            ["a_weight", "b_weight", "c_weight"], fused["params"]
        )
        self.assertNotIn("branch_parameter_counts", fused["attrs"])

        pair = optimize_circuit_for_vulkan(
            {"nodes": nodes[:2]},
            can_fuse_parallel_linears=lambda group: len(group) == 2,
        )
        self.assertEqual("parallel_linear_2way", pair["nodes"][0]["op"])

    def test_fuses_fp8_parallel_linears_with_weight_scale_pairs(self) -> None:
        nodes = [
            {
                "id": branch,
                "op": "linear",
                "inputs": ["hidden"],
                "outputs": [f"{branch}_out"],
                "params": [f"{branch}_weight", f"{branch}_weight_scale_inv"],
            }
            for branch in ("a", "b")
        ]

        optimized = optimize_circuit_for_vulkan(
            {"nodes": nodes},
            can_fuse_parallel_linears=lambda group: len(group) == 2,
        )

        self.assertEqual(1, len(optimized["nodes"]))
        fused = optimized["nodes"][0]
        self.assertEqual("parallel_linear_2way", fused["op"])
        self.assertEqual(
            [
                "a_weight",
                "a_weight_scale_inv",
                "b_weight",
                "b_weight_scale_inv",
            ],
            fused["params"],
        )
        self.assertEqual([2, 2], fused["attrs"]["branch_parameter_counts"])

    def test_parallel_linear_fusion_does_not_depend_on_scale_parameter_names(
        self,
    ) -> None:
        nodes = [
            {
                "id": "gate_projection",
                "op": "linear",
                "inputs": ["hidden"],
                "outputs": ["gate"],
                "params": ["first_matrix", "first_quantization_metadata"],
            },
            {
                "id": "up_projection",
                "op": "linear",
                "inputs": ["hidden"],
                "outputs": ["up"],
                "params": ["second_matrix", "second_scale"],
            },
        ]

        optimized = optimize_circuit_for_vulkan(
            {"nodes": nodes},
            can_fuse_parallel_linears=lambda group: len(group) == 2,
        )

        self.assertEqual(1, len(optimized["nodes"]))
        fused = optimized["nodes"][0]
        self.assertEqual("parallel_linear_2way", fused["op"])
        self.assertEqual(
            [
                "first_matrix",
                "first_quantization_metadata",
                "second_matrix",
                "second_scale",
            ],
            fused["params"],
        )
        self.assertEqual([2, 2], fused["attrs"]["branch_parameter_counts"])

    def test_lowers_fp8_input_quantization_to_one_reusable_physical_signal(self) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "projection",
                    "op": "linear_residual",
                    "inputs": ["normalized", "residual"],
                    "outputs": ["output"],
                    "params": ["weight", "weight_scale_inv"],
                    "attrs": {
                        "compiled_from": ["linear", "residual_add"],
                        "intermediate_rounding": "BF16",
                    },
                }
            ]
        }

        optimized = optimize_circuit_for_vulkan(
            circuit,
            prequantization_spec=lambda _node: {
                "contract": "bf16_blockwise_fp8_e4m3_f32_scale.v1",
                "input_size": 5120,
                "block_rows": 128,
                "block_columns": 128,
            },
        )

        self.assertEqual(2, len(optimized["nodes"]))
        quantize, projection = optimized["nodes"]
        self.assertEqual("quantize_fp8_e4m3", quantize["op"])
        self.assertEqual(["normalized"], quantize["inputs"])
        self.assertEqual([1, 4], quantize["attrs"]["output_element_bytes"])
        self.assertEqual(
            ["projection"], quantize["attrs"]["consumer_node_ids"]
        )
        self.assertEqual(
            ["linear", "residual_add"],
            quantize["attrs"]["semantic_source_node_ids"],
        )
        self.assertEqual(
            [*quantize["outputs"], "residual"],
            projection["inputs"],
        )
        self.assertEqual(
            ["normalized", "residual"],
            projection["attrs"]["physical_logical_inputs"],
        )
        self.assertEqual([2], projection["attrs"]["output_element_bytes"])

    def test_sparse_gate_up_emits_down_representation_without_helper_dispatch(
        self,
    ) -> None:
        contract = (
            "bf16_sparse_moe_intermediate_blockwise_fp8_e4m3_f32_scale_"
            "u32_route_map.v1"
        )
        circuit = {
            "nodes": [
                {
                    "id": "sparse_moe_gate_up",
                    "op": "sparse_moe_gate_up",
                    "inputs": ["normalized", "routes"],
                    "outputs": ["expert_intermediates"],
                    "params": ["gate_up_weight", "gate_up_scale"],
                    "attrs": {
                        "hidden_size": 2048,
                        "intermediate_size": 512,
                        "num_experts": 256,
                        "experts_per_token": 8,
                    },
                },
                {
                    "id": "sparse_moe_down",
                    "op": "sparse_moe_down",
                    "inputs": ["expert_intermediates", "routes"],
                    "outputs": ["expert_outputs"],
                    "params": ["down_weight", "down_scale"],
                    "attrs": {
                        "hidden_size": 2048,
                        "intermediate_size": 512,
                        "num_experts": 256,
                        "experts_per_token": 8,
                    },
                },
            ]
        }

        def describe(node):
            if node["op"] == "sparse_moe_gate_up":
                return {
                    "contract": "bf16_blockwise_fp8_e4m3_f32_scale.v1",
                    "input_size": 2048,
                    "block_columns": 128,
                }
            if node["op"] == "sparse_moe_down":
                return {
                    "contract": contract,
                    "input_size": 4096,
                    "block_columns": 128,
                    "experts_per_token": 8,
                }
            return None

        optimized = optimize_circuit_for_vulkan(
            circuit,
            prequantization_spec=describe,
            can_emit_representation=lambda producer, scope: (
                producer["op"] == "sparse_moe_gate_up"
                and scope["contract"] == contract
            ),
        )

        self.assertEqual(
            [
                "quantize_fp8_e4m3",
                "sparse_moe_gate_up",
                "sparse_moe_down",
            ],
            [node["op"] for node in optimized["nodes"]],
        )
        _, gate_up, down = optimized["nodes"]
        representation = gate_up["attrs"]["physical_output_representations"][0]
        self.assertEqual(contract, representation["contract"])
        self.assertEqual(8, representation["experts_per_token"])
        self.assertEqual(
            [
                "expert_intermediates",
                *representation["outputs"],
            ],
            gate_up["outputs"],
        )
        self.assertEqual(
            [*representation["outputs"], "routes"],
            down["inputs"],
        )
        self.assertEqual(
            "sparse_moe_gate_up",
            down["attrs"]["physical_input_provider_id"],
        )

    def test_sparse_representation_without_fused_producer_stays_logical(
        self,
    ) -> None:
        contract = (
            "bf16_sparse_moe_intermediate_blockwise_fp8_e4m3_f32_scale_"
            "u32_route_map.v1"
        )
        down = {
            "id": "sparse_moe_down",
            "op": "sparse_moe_down",
            "inputs": ["external_intermediates", "routes"],
            "outputs": ["expert_outputs"],
            "params": ["down_weight", "down_scale"],
            "attrs": {
                "hidden_size": 2048,
                "intermediate_size": 512,
                "num_experts": 256,
                "experts_per_token": 8,
            },
        }

        optimized = optimize_circuit_for_vulkan(
            {"nodes": [down]},
            prequantization_spec=lambda _node: {
                "contract": contract,
                "input_size": 4096,
                "block_columns": 128,
                "experts_per_token": 8,
            },
            can_emit_representation=lambda _producer, _scope: False,
        )

        self.assertEqual(1, len(optimized["nodes"]))
        optimized_down = optimized["nodes"][0]
        self.assertEqual("sparse_moe_down", optimized_down["op"])
        self.assertEqual(
            ["external_intermediates", "routes"],
            optimized_down["inputs"],
        )
        self.assertNotIn(
            "physical_input_contract",
            optimized_down["attrs"],
        )

    def test_shares_fp8_input_quantization_across_compatible_consumers(self) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "wide_projection",
                    "op": "linear",
                    "inputs": ["normalized"],
                    "outputs": ["wide"],
                    "params": ["wide_weight", "wide_scale"],
                },
                {
                    "id": "narrow_projection",
                    "op": "linear",
                    "inputs": ["normalized"],
                    "outputs": ["narrow"],
                    "params": ["narrow_weight", "narrow_scale"],
                },
            ]
        }

        optimized = optimize_circuit_for_vulkan(
            circuit,
            prequantization_spec=lambda _node: {
                "contract": "bf16_blockwise_fp8_e4m3_f32_scale.v1",
                "input_size": 5120,
                "block_rows": 128,
                "block_columns": 128,
            },
        )

        self.assertEqual(3, len(optimized["nodes"]))
        quantize, wide, narrow = optimized["nodes"]
        self.assertEqual(
            ["wide_projection", "narrow_projection"],
            quantize["attrs"]["consumer_node_ids"],
        )
        self.assertEqual(quantize["outputs"], wide["inputs"])
        self.assertEqual(quantize["outputs"], narrow["inputs"])
        self.assertEqual(
            wide["attrs"]["physical_input_provider_id"],
            narrow["attrs"]["physical_input_provider_id"],
        )

    def test_lowers_declared_int8_representation_without_format_specific_logic(
        self,
    ) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "projection",
                    "op": "linear",
                    "inputs": ["normalized"],
                    "outputs": ["projected"],
                    "params": ["weight", "weight_qzeros", "weight_scales"],
                }
            ]
        }

        optimized = optimize_circuit_for_vulkan(
            circuit,
            prequantization_spec=lambda _node: {
                "contract": "bf16_blockwise_symmetric_int8_f32_scale.v1",
                "input_size": 5120,
                "block_columns": 32,
            },
        )

        quantize, projection = optimized["nodes"]
        self.assertEqual("quantize_int8_symmetric", quantize["op"])
        self.assertEqual(
            ["projection__input_int8", "projection__input_scale_f32"],
            quantize["outputs"],
        )
        self.assertEqual(
            "bf16_blockwise_symmetric_int8_f32_scale.v1",
            projection["attrs"]["physical_input_contract"],
        )
        self.assertEqual(quantize["outputs"], projection["inputs"])

    def test_lowers_declared_pairpacked_int8_representation_with_block_sums(
        self,
    ) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "projection",
                    "op": "linear",
                    "inputs": ["normalized"],
                    "outputs": ["projected"],
                    "params": ["weight", "weight_scales"],
                }
            ]
        }

        optimized = optimize_circuit_for_vulkan(
            circuit,
            prequantization_spec=lambda _node: {
                "contract": (
                    "bf16_blockwise_symmetric_int8_pairpacked_"
                    "f32_scale_i32_sum.v1"
                ),
                "input_size": 5120,
                "block_columns": 32,
            },
        )

        quantize, projection = optimized["nodes"]
        self.assertEqual("quantize_int8_symmetric_pairpacked", quantize["op"])
        self.assertEqual(
            [
                "projection__input_int8_pairpacked",
                "projection__input_scale_f32",
                "projection__input_sum_i32",
            ],
            quantize["outputs"],
        )
        self.assertEqual([1, 4, 4], quantize["attrs"]["output_element_bytes"])
        self.assertEqual(quantize["outputs"], projection["inputs"])

    def test_fuses_reusable_fp8_representation_into_eligible_producer(self) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "normalization",
                    "op": "rms_norm",
                    "inputs": ["hidden"],
                    "outputs": ["normalized"],
                    "params": ["weight"],
                    "attrs": {"eps": 1e-6, "weight_offset": 1.0},
                },
                {
                    "id": "projection",
                    "op": "linear",
                    "inputs": ["normalized"],
                    "outputs": ["projected"],
                    "params": ["projection_weight", "projection_scale"],
                },
            ]
        }

        optimized = optimize_circuit_for_vulkan(
            circuit,
            prequantization_spec=lambda node: (
                {
                    "contract": "bf16_blockwise_fp8_e4m3_f32_scale.v1",
                    "input_size": 5120,
                    "block_rows": 128,
                    "block_columns": 128,
                }
                if node["id"] == "projection"
                else None
            ),
            can_emit_representation=lambda producer, _scope: (
                producer["op"] == "rms_norm"
            ),
        )

        self.assertEqual(2, len(optimized["nodes"]))
        normalization, projection = optimized["nodes"]
        self.assertEqual(
            [
                "normalized",
                "projection__input_fp8_e4m3",
                "projection__input_scale_f32",
            ],
            normalization["outputs"],
        )
        self.assertEqual([2, 1, 4], normalization["attrs"]["output_element_bytes"])
        self.assertEqual(
            [
                {
                    "contract": "bf16_blockwise_fp8_e4m3_f32_scale.v1",
                    "logical_signal": "normalized",
                    "outputs": [
                        "projection__input_fp8_e4m3",
                        "projection__input_scale_f32",
                    ],
                    "consumer_node_ids": ["projection"],
                    "element_count": 5120,
                    "block_columns": 128,
                }
            ],
            normalization["attrs"]["physical_output_representations"],
        )
        self.assertEqual(normalization["outputs"][1:], projection["inputs"])
        self.assertEqual(
            "normalization",
            projection["attrs"]["physical_input_provider_id"],
        )

    def test_does_not_fuse_linears_with_different_inputs(self) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "a",
                    "op": "linear",
                    "inputs": ["first"],
                    "outputs": ["a_out"],
                    "params": ["a_weight"],
                },
                {
                    "id": "b",
                    "op": "linear",
                    "inputs": ["second"],
                    "outputs": ["b_out"],
                    "params": ["b_weight"],
                },
            ]
        }

        optimized = optimize_circuit_for_vulkan(
            circuit,
            can_fuse_parallel_linears=lambda _group: True,
        )

        self.assertEqual(circuit, optimized)

    def test_fuses_compatible_linear_into_contiguous_three_way_split(self) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "projection",
                    "op": "linear",
                    "inputs": ["hidden"],
                    "outputs": ["projected"],
                    "params": ["weight"],
                },
                {
                    "id": "partition",
                    "op": "split",
                    "inputs": ["projected"],
                    "outputs": ["a", "b", "c"],
                    "attrs": {"part_width": 16},
                },
            ]
        }

        optimized = optimize_circuit_for_vulkan(
            circuit,
            can_fuse_linear_split=lambda node: node["params"] == ["weight"],
        )

        self.assertEqual(1, len(optimized["nodes"]))
        fused = optimized["nodes"][0]
        self.assertEqual("linear_split_3way", fused["op"])
        self.assertEqual(["hidden"], fused["inputs"])
        self.assertEqual(["a", "b", "c"], fused["outputs"])
        self.assertEqual([16, 16, 16], fused["attrs"]["part_widths"])
        self.assertEqual(["projection", "partition"], fused["attrs"]["compiled_from"])

    def test_keeps_linear_split_when_backend_layout_is_not_fusible(self) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "projection",
                    "op": "linear",
                    "inputs": ["hidden"],
                    "outputs": ["projected"],
                    "params": ["weight"],
                },
                {
                    "id": "partition",
                    "op": "split",
                    "inputs": ["projected"],
                    "outputs": ["a", "b", "c"],
                    "attrs": {"part_width": 16},
                },
            ]
        }

        optimized = optimize_circuit_for_vulkan(
            circuit,
            can_fuse_linear_split=lambda _node: False,
        )

        self.assertEqual(circuit, optimized)

    def test_fuses_discovered_regions_without_layer_or_node_names(self) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "projection_a",
                    "op": "linear",
                    "inputs": ["hidden"],
                    "outputs": ["projected"],
                    "params": ["weight_a"],
                },
                {
                    "id": "skip_a",
                    "op": "residual_add",
                    "inputs": ["input_frame", "projected"],
                    "outputs": ["residual_out"],
                },
                {
                    "id": "activation_a",
                    "op": "silu",
                    "inputs": ["gate"],
                    "outputs": ["activated"],
                    "attrs": {"element_count": 16},
                },
                {
                    "id": "product_a",
                    "op": "multiply",
                    "inputs": ["up", "activated"],
                    "outputs": ["output_frame"],
                },
            ]
        }

        optimized = optimize_circuit_for_vulkan(circuit)

        self.assertEqual(
            ["linear_residual", "silu_multiply"],
            [node["op"] for node in optimized["nodes"]],
        )
        self.assertEqual(
            ["hidden", "input_frame"], optimized["nodes"][0]["inputs"]
        )
        self.assertEqual(["weight_a"], optimized["nodes"][0]["params"])
        self.assertEqual(["gate", "up"], optimized["nodes"][1]["inputs"])
        self.assertEqual("BF16", optimized["nodes"][1]["attrs"]["intermediate_rounding"])
        self.assertEqual(4, len(circuit["nodes"]))

    def test_does_not_fuse_an_intermediate_with_multiple_consumers(self) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "activation",
                    "op": "silu",
                    "inputs": ["gate"],
                    "outputs": ["activated"],
                },
                {
                    "id": "product",
                    "op": "multiply",
                    "inputs": ["activated", "up"],
                    "outputs": ["product"],
                },
                {
                    "id": "extra_consumer",
                    "op": "multiply",
                    "inputs": ["activated", "other"],
                    "outputs": ["output_frame"],
                },
            ]
        }

        optimized = optimize_circuit_for_vulkan(circuit)

        self.assertEqual(circuit, optimized)

    def test_fuses_block_scaled_fp8_linear_with_residual(self) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "projection",
                    "op": "linear",
                    "inputs": ["hidden"],
                    "outputs": ["projected"],
                    "params": ["weight", "weight_scale_inv"],
                },
                {
                    "id": "skip",
                    "op": "residual_add",
                    "inputs": ["residual", "projected"],
                    "outputs": ["output"],
                },
            ]
        }

        optimized = optimize_circuit_for_vulkan(circuit)

        self.assertEqual("linear_residual", optimized["nodes"][0]["op"])
        self.assertEqual(
            ["weight", "weight_scale_inv"], optimized["nodes"][0]["params"]
        )

    def test_fuses_scalar_gate_projection_with_its_only_multiply(self) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "gate_projection",
                    "op": "linear",
                    "inputs": ["normalized"],
                    "outputs": ["gate_logit"],
                    "params": ["gate_weight"],
                },
                {
                    "id": "apply_gate",
                    "op": "sigmoid_scalar_multiply",
                    "inputs": ["value", "gate_logit"],
                    "outputs": ["gated_value"],
                },
            ]
        }

        optimized = optimize_circuit_for_vulkan(
            circuit,
            can_fuse_linear_sigmoid_scalar_multiply=lambda _linear, _multiply: True,
        )

        self.assertEqual(1, len(optimized["nodes"]))
        fused = optimized["nodes"][0]
        self.assertEqual("linear_sigmoid_scalar_multiply", fused["op"])
        self.assertEqual(["normalized", "value"], fused["inputs"])
        self.assertEqual(["gated_value"], fused["outputs"])
        self.assertEqual(["gate_weight"], fused["params"])
        self.assertEqual(
            ["gate_projection", "apply_gate"],
            fused["attrs"]["compiled_from"],
        )
        self.assertEqual("BF16", fused["attrs"]["intermediate_rounding"])

    def test_keeps_scalar_gate_projection_with_an_additional_consumer(self) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "gate_projection",
                    "op": "linear",
                    "inputs": ["normalized"],
                    "outputs": ["gate_logit"],
                    "params": ["gate_weight"],
                },
                {
                    "id": "apply_gate",
                    "op": "sigmoid_scalar_multiply",
                    "inputs": ["value", "gate_logit"],
                    "outputs": ["gated_value"],
                },
                {
                    "id": "observe_gate",
                    "op": "silu",
                    "inputs": ["gate_logit"],
                    "outputs": ["observed_gate"],
                },
            ]
        }

        optimized = optimize_circuit_for_vulkan(
            circuit,
            can_fuse_linear_sigmoid_scalar_multiply=lambda _linear, _multiply: True,
        )

        self.assertEqual(circuit, optimized)

    def test_fuses_scalar_gate_and_two_exact_residual_stages(self) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "gate_projection",
                    "op": "linear",
                    "inputs": ["normalized"],
                    "outputs": ["gate_logit"],
                    "params": ["gate_weight"],
                },
                {
                    "id": "apply_gate",
                    "op": "sigmoid_scalar_multiply",
                    "inputs": ["shared_value", "gate_logit"],
                    "outputs": ["gated_value"],
                },
                {
                    "id": "add_sparse",
                    "op": "residual_add",
                    "inputs": ["sparse_value", "gated_value"],
                    "outputs": ["combined_value"],
                },
                {
                    "id": "add_layer_residual",
                    "op": "residual_add",
                    "inputs": ["layer_residual", "combined_value"],
                    "outputs": ["output"],
                },
            ]
        }

        optimized = optimize_circuit_for_vulkan(
            circuit,
            can_fuse_linear_sigmoid_scalar_multiply=lambda _linear, _multiply: True,
        )

        self.assertEqual(1, len(optimized["nodes"]))
        fused = optimized["nodes"][0]
        self.assertEqual(
            "linear_sigmoid_scalar_multiply_residual2",
            fused["op"],
        )
        self.assertEqual(
            [
                "normalized",
                "shared_value",
                "sparse_value",
                "layer_residual",
            ],
            fused["inputs"],
        )
        self.assertEqual(["output"], fused["outputs"])
        self.assertEqual(["gate_weight"], fused["params"])
        self.assertEqual(
            [
                "gate_projection",
                "apply_gate",
                "add_sparse",
                "add_layer_residual",
            ],
            fused["attrs"]["compiled_from"],
        )

    def test_keeps_scalar_gate_residual_chain_with_an_observed_intermediate(self) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "gate_projection",
                    "op": "linear",
                    "inputs": ["normalized"],
                    "outputs": ["gate_logit"],
                    "params": ["gate_weight"],
                },
                {
                    "id": "apply_gate",
                    "op": "sigmoid_scalar_multiply",
                    "inputs": ["shared_value", "gate_logit"],
                    "outputs": ["gated_value"],
                },
                {
                    "id": "add_sparse",
                    "op": "residual_add",
                    "inputs": ["sparse_value", "gated_value"],
                    "outputs": ["combined_value"],
                },
                {
                    "id": "add_layer_residual",
                    "op": "residual_add",
                    "inputs": ["layer_residual", "combined_value"],
                    "outputs": ["output"],
                },
                {
                    "id": "observe_combined",
                    "op": "silu",
                    "inputs": ["combined_value"],
                    "outputs": ["observed"],
                },
            ]
        }

        optimized = optimize_circuit_for_vulkan(
            circuit,
            can_fuse_linear_sigmoid_scalar_multiply=lambda _linear, _multiply: True,
        )

        self.assertEqual(
            [
                "linear_sigmoid_scalar_multiply",
                "residual_add",
                "residual_add",
                "silu",
            ],
            [node["op"] for node in optimized["nodes"]],
        )

    def test_lowers_fused_attention_to_partitioned_physical_stages(self) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "kv_memory_append",
                    "op": "append_state_update",
                    "inputs": ["k", "v", "kv_memory"],
                    "outputs": ["all_k", "all_v"],
                    "params": [],
                    "state_reads": ["kv_memory"],
                    "state_writes": ["kv_memory"],
                    "attrs": {
                        "growth": "per_activation",
                        "query_heads": 16,
                        "key_value_heads": 2,
                        "head_width": 256,
                        "query_groups_per_kv_head": 8,
                    },
                },
                {
                    "id": "attention_read",
                    "op": "scaled_dot_product_attention",
                    "inputs": ["q", "all_k", "all_v"],
                    "outputs": ["out"],
                    "params": [],
                    "state_reads": [],
                    "state_writes": [],
                    "attrs": {
                        "causal": True,
                        "scale": 0.0625,
                        "window_size": None,
                        "attention_sinks": False,
                        "query_heads": 16,
                        "key_value_heads": 2,
                        "head_width": 256,
                        "query_groups_per_kv_head": 8,
                    },
                },
            ]
        }

        optimized = optimize_circuit_for_vulkan(
            circuit,
            can_fuse_append_attention=lambda _append, _attention: True,
            attention_partition_count=8,
            prequantization_spec=lambda _node: None,
        )

        self.assertEqual(
            [
                "attention_partition_partials",
                "append_scaled_dot_product_attention",
            ],
            [node["op"] for node in optimized["nodes"]],
        )
        helper, reduction = optimized["nodes"]
        self.assertEqual(["q", "k", "v", "kv_memory"], helper["inputs"])
        self.assertEqual(["kv_memory"], helper["state_reads"])
        self.assertEqual([], helper["state_writes"])
        self.assertEqual(8, helper["attrs"]["partition_count"])
        self.assertEqual([4], helper["attrs"]["output_element_bytes"])
        self.assertEqual(
            "bf16_attention_partition_partials_f32.v1",
            helper["attrs"]["physical_representation_contract"],
        )
        self.assertEqual(
            [
                helper["outputs"][0],
                "k",
                "v",
                "kv_memory",
            ],
            reduction["inputs"],
        )
        self.assertEqual(
            ["q", "k", "v", "kv_memory"],
            reduction["attrs"]["physical_logical_inputs"],
        )
        self.assertEqual(8, reduction["attrs"]["attention_partition_count"])

    def test_fuses_contiguous_gate_up_projection_split_and_swiglu(self) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "gate_up_projection",
                    "op": "linear",
                    "inputs": ["normalized"],
                    "outputs": ["gate_up"],
                    "params": ["weight", "weight_scale_inv"],
                },
                {
                    "id": "gate_up_split",
                    "op": "split",
                    "inputs": ["gate_up"],
                    "outputs": ["gate", "up"],
                    "attrs": {"part_width": 512},
                },
                {
                    "id": "activation",
                    "op": "silu_multiply",
                    "inputs": ["gate", "up"],
                    "outputs": ["activated"],
                    "attrs": {"element_count": 512},
                },
            ]
        }

        optimized = optimize_circuit_for_vulkan(
            circuit,
            can_fuse_contiguous_linear_swiglu=lambda _projection, _split, _activation: True,
        )

        self.assertEqual(1, len(optimized["nodes"]))
        fused = optimized["nodes"][0]
        self.assertEqual("contiguous_linear_swiglu", fused["op"])
        self.assertEqual(["normalized"], fused["inputs"])
        self.assertEqual(["activated"], fused["outputs"])
        self.assertEqual(["weight", "weight_scale_inv"], fused["params"])
        self.assertEqual(
            ["gate_up_projection", "gate_up_split", "activation"],
            fused["attrs"]["compiled_from"],
        )
        self.assertEqual(512, fused["attrs"]["part_width"])
        self.assertEqual("contiguous_gate_up", fused["attrs"]["weight_partition"])
        self.assertEqual("BF16", fused["attrs"]["intermediate_rounding"])

    def test_fuses_adjacent_fp8_and_bf16_parallel_projections(self) -> None:
        circuit = {
            "nodes": [
                {
                    "id": "large_a__large_b",
                    "op": "parallel_linear_2way",
                    "inputs": ["hidden_fp8", "hidden_scale"],
                    "outputs": ["large_a_out", "large_b_out"],
                    "params": ["large_a", "large_a_scale", "large_b", "large_b_scale"],
                    "attrs": {
                        "compiled_from": ["large_a", "large_b"],
                        "branch_count": 2,
                        "branch_parameter_counts": [2, 2],
                        "output_element_bytes": [2, 2],
                        "physical_input_contract": (
                            "bf16_blockwise_fp8_e4m3_f32_scale.v1"
                        ),
                        "physical_input_provider_id": "norm",
                        "physical_logical_inputs": ["hidden"],
                    },
                },
                {
                    "id": "small_c__small_d",
                    "op": "parallel_linear_2way",
                    "inputs": ["hidden"],
                    "outputs": ["small_c_out", "small_d_out"],
                    "params": ["small_c", "small_d"],
                    "attrs": {
                        "compiled_from": ["small_c", "small_d"],
                        "branch_count": 2,
                        "output_element_bytes": [2, 2],
                    },
                },
            ]
        }

        optimized = optimize_circuit_for_vulkan(
            circuit,
            can_fuse_mixed_precision_parallel_linears=lambda _fp8, _bf16: True,
        )

        self.assertEqual(1, len(optimized["nodes"]))
        fused = optimized["nodes"][0]
        self.assertEqual("mixed_parallel_linear_4way", fused["op"])
        self.assertEqual(
            ["hidden_fp8", "hidden_scale", "hidden"], fused["inputs"]
        )
        self.assertEqual(
            ["large_a_out", "large_b_out", "small_c_out", "small_d_out"],
            fused["outputs"],
        )
        self.assertEqual(
            [
                "large_a",
                "large_a_scale",
                "large_b",
                "large_b_scale",
                "small_c",
                "small_d",
            ],
            fused["params"],
        )
        self.assertEqual([2, 2, 1, 1], fused["attrs"]["branch_parameter_counts"])
        self.assertEqual(
            ["large_a", "large_b", "small_c", "small_d"],
            fused["attrs"]["compiled_from"],
        )

    def test_fuses_hyper_connection_pre_and_post_pre_regions(self) -> None:
        pre = [
            {
                "id": "attention_function",
                "op": "normalized_linear",
                "inputs": ["input_frame"],
                "outputs": ["attention_mixes"],
                "params": ["attention_function_weight"],
                "attrs": {
                    "normalization": "root_mean_square",
                    "normalization_epsilon": 1e-6,
                    "multiplicity": 4,
                    "output_element_bytes": [4],
                },
            },
            {
                "id": "attention_sinkhorn",
                "op": "hyper_connection_sinkhorn",
                "inputs": ["attention_mixes"],
                "outputs": ["attention_pre", "attention_post", "attention_comb"],
                "params": ["attention_scale", "attention_base"],
                "attrs": {
                    "multiplicity": 4,
                    "sinkhorn_iterations": 20,
                    "epsilon": 1e-6,
                    "output_element_bytes": [4, 4, 4],
                },
            },
            {
                "id": "attention_reduce",
                "op": "hyper_connection_reduce",
                "inputs": ["input_frame", "attention_pre"],
                "outputs": ["operator_input"],
                "attrs": {"multiplicity": 4, "output_element_bytes": [2]},
            },
        ]
        post = {
            "id": "attention_post",
            "op": "hyper_connection_post",
            "inputs": [
                "operator_out",
                "input_frame",
                "attention_post",
                "attention_comb",
            ],
            "outputs": ["attention_residual"],
            "attrs": {
                "multiplicity": 4,
                "sinkhorn_iterations": 20,
                "epsilon": 1e-6,
                "output_element_bytes": [2],
            },
        }
        feed_forward = []
        for node in pre:
            clone = {
                **node,
                "id": node["id"].replace("attention", "feed_forward"),
                "inputs": [
                    signal.replace("input_frame", "attention_residual").replace(
                        "attention_", "feed_forward_"
                    )
                    for signal in node["inputs"]
                ],
                "outputs": [
                    signal.replace("attention_", "feed_forward_").replace(
                        "operator_input", "ffn_input"
                    )
                    for signal in node["outputs"]
                ],
                "params": [
                    parameter.replace("attention_", "feed_forward_")
                    for parameter in node.get("params", [])
                ],
                "attrs": dict(node["attrs"]),
            }
            feed_forward.append(clone)
        feed_forward[0]["inputs"] = ["attention_residual"]
        feed_forward[2]["inputs"] = ["attention_residual", "feed_forward_pre"]

        optimized = optimize_circuit_for_vulkan(
            {"nodes": [*pre, {"id": "operator", "op": "operator"}, post, *feed_forward]}
        )

        self.assertEqual(
            ["hyper_connection_pre", "operator", "hyper_connection_post_pre"],
            [node["op"] for node in optimized["nodes"]],
        )
        first = optimized["nodes"][0]
        self.assertEqual(["input_frame"], first["inputs"])
        self.assertEqual(
            ["operator_input", "attention_post", "attention_comb"],
            first["outputs"],
        )
        self.assertEqual(
            ["attention_function_weight", "attention_scale", "attention_base"],
            first["params"],
        )
        fused = optimized["nodes"][2]
        self.assertEqual(post["inputs"], fused["inputs"])
        self.assertEqual(
            [
                "attention_residual",
                "ffn_input",
                "feed_forward_post",
                "feed_forward_comb",
            ],
            fused["outputs"],
        )
        self.assertEqual([2, 2, 4, 4], fused["attrs"]["output_element_bytes"])
        self.assertEqual(
            [
                "attention_post",
                "feed_forward_function",
                "feed_forward_sinkhorn",
                "feed_forward_reduce",
            ],
            fused["attrs"]["compiled_from"],
        )


if __name__ == "__main__":
    unittest.main()
