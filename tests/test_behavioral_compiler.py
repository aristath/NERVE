from __future__ import annotations

from copy import deepcopy

import pytest

from nerve.behavioral_compiler import (
    build_behavioral_validation,
    model_contract_digest,
    prove_exact_circuit_candidate,
    validate_behavioral_validation_artifact,
)
from nerve.compilation import ModelCompileError


def source_circuit() -> dict:
    return {
        "schema": "nerve.stream_circuit.v1",
        "source": {"component_id": "layer_00"},
        "boundary": {
            "inputs": [{"id": "input_frame", "source": "x"}],
            "outputs": [{"id": "output_frame", "source": "y"}],
        },
        "state_ports": [],
        "parameters": {"refs": {}},
        "behavioral_error_contract": {"mode": "source_reference_circuit"},
        "nodes": [
            {
                "id": "activation",
                "op": "silu",
                "inputs": ["x"],
                "outputs": ["activated"],
                "attrs": {"element_count": 4},
            },
            {
                "id": "multiply",
                "op": "multiply",
                "inputs": ["activated", "gate"],
                "outputs": ["y"],
            },
        ],
    }


def empirical_evidence(*, free_running_status: str = "passed") -> dict:
    return {
        "schema": "nerve.behavioral_empirical_evidence.v1",
        "model_contract_digest": "a" * 64,
        "teacher_forced": {
            "status": "passed",
            "sample_count": 128,
            "metrics": {"maximum_logit_error": 0.01},
        },
        "free_running": {
            "status": free_running_status,
            "sample_count": 64,
            "metrics": {"distribution_similarity": 0.99},
        },
    }


def fused_candidate(source: dict) -> dict:
    candidate = deepcopy(source)
    candidate["nodes"] = [
        {
            "id": "activation__multiply",
            "op": "silu_multiply",
            "inputs": ["x", "gate"],
            "outputs": ["y"],
            "attrs": {
                "compiled_from": ["activation", "multiply"],
                "intermediate_rounding": "BF16",
                "element_count": 4,
            },
        }
    ]
    return candidate


def hyper_connection_source_and_candidate() -> tuple[dict, dict]:
    source = {
        "schema": "nerve.layer_circuit.v1",
        "source": {"component_id": "layer_00"},
        "runtime_role": "signal_processor",
        "boundary": {
            "inputs": [{"id": "input_frame"}],
            "outputs": [
                {"id": "operator_input", "source": "operator_input"},
                {"id": "post", "source": "post"},
                {"id": "combination", "source": "combination"},
            ],
        },
        "state_ports": [],
        "parameters": {"refs": {}},
        "behavioral_error_contract": {"kind": "exact"},
        "nodes": [
            {
                "id": "function",
                "op": "normalized_linear",
                "inputs": ["input_frame"],
                "outputs": ["mixes"],
                "params": ["function"],
                "attrs": {
                    "multiplicity": 4,
                    "normalization": "root_mean_square",
                    "normalization_epsilon": 1e-6,
                },
            },
            {
                "id": "sinkhorn",
                "op": "hyper_connection_sinkhorn",
                "inputs": ["mixes"],
                "outputs": ["pre", "post", "combination"],
                "params": ["scale", "base"],
                "attrs": {
                    "epsilon": 1e-6,
                    "multiplicity": 4,
                    "sinkhorn_iterations": 20,
                },
            },
            {
                "id": "reduce",
                "op": "hyper_connection_reduce",
                "inputs": ["input_frame", "pre"],
                "outputs": ["operator_input"],
                "params": [],
                "attrs": {"multiplicity": 4},
            },
        ],
    }
    candidate = deepcopy(source)
    candidate["nodes"] = [
        {
            "id": "function__sinkhorn__reduce",
            "op": "hyper_connection_pre",
            "inputs": ["input_frame"],
            "outputs": ["operator_input", "post", "combination"],
            "params": ["function", "scale", "base"],
            "attrs": {
                "compiled_from": ["function", "sinkhorn", "reduce"],
                "epsilon": 1e-6,
                "intermediate_rounding": "BF16",
                "multiplicity": 4,
                "normalization_epsilon": 1e-6,
                "sinkhorn_iterations": 20,
            },
        }
    ]
    return source, candidate


def test_exact_candidate_gate_proves_complete_fusion_coverage() -> None:
    source = source_circuit()
    candidate = fused_candidate(source)

    evidence = prove_exact_circuit_candidate(
        component_id="layer_00", source=source, candidate=candidate
    )

    assert evidence["status"] == "passed"
    assert evidence["candidate_kind"] == "exact_reference"
    assert evidence["covered_source_node_count"] == 2
    assert evidence["rewrites"][0]["proof_contract"] == "silu_multiply_exact_bf16.v1"


def test_exact_candidate_proves_mixed_query_kv_norm_rope_transaction() -> None:
    source = {
        "schema": "nerve.layer_circuit.v1",
        "source": {"component_id": "layer_00"},
        "boundary": {
            "inputs": [{"id": "query"}, {"id": "kv"}],
            "outputs": [
                {"id": "query_positioned", "source": "query_positioned"},
                {"id": "kv_positioned", "source": "kv_positioned"},
            ],
        },
        "state_ports": [],
        "parameters": {"refs": {"kv_weight": {"tensor": "kv.weight"}}},
        "behavioral_error_contract": {"kind": "exact"},
        "nodes": [
            {
                "id": "query_norm",
                "op": "rms_norm_per_head_unscaled",
                "inputs": ["query"],
                "outputs": ["query_normed"],
                "attrs": {"eps": 1e-6, "head_count": 64, "head_width": 512},
            },
            {
                "id": "query_rope",
                "op": "rotary_position_embedding",
                "inputs": ["query_normed"],
                "outputs": ["query_positioned"],
                "attrs": {"head_count": 64, "head_width": 512, "rotary_width": 64},
            },
            {
                "id": "kv_norm",
                "op": "rms_norm",
                "inputs": ["kv"],
                "outputs": ["kv_normed"],
                "params": ["kv_weight"],
                "attrs": {"eps": 1e-6, "weight_offset": 0.0},
            },
            {
                "id": "kv_rope",
                "op": "rotary_position_embedding",
                "inputs": ["kv_normed"],
                "outputs": ["kv_positioned"],
                "attrs": {"head_count": 1, "head_width": 512, "rotary_width": 64},
            },
        ],
    }
    candidate = deepcopy(source)
    candidate["nodes"] = [
        {
            "id": "query_norm__query_rope__kv_norm__kv_rope",
            "op": "parallel_mixed_head_norm_rope_2way",
            "inputs": ["query", "kv"],
            "outputs": ["query_positioned", "kv_positioned"],
            "params": ["kv_weight"],
            "attrs": {
                "compiled_from": [
                    "query_norm",
                    "query_rope",
                    "kv_norm",
                    "kv_rope",
                ],
                "branches": [
                    {
                        "norm_op": "rms_norm_per_head_unscaled",
                        "norm": deepcopy(source["nodes"][0]["attrs"]),
                        "rope": deepcopy(source["nodes"][1]["attrs"]),
                    },
                    {
                        "norm_op": "rms_norm",
                        "norm": deepcopy(source["nodes"][2]["attrs"]),
                        "rope": deepcopy(source["nodes"][3]["attrs"]),
                    },
                ],
                "branch_parameter_counts": [0, 1],
                "intermediate_rounding": "BF16",
                "output_element_bytes": [2, 2],
            },
        }
    ]

    evidence = prove_exact_circuit_candidate(
        component_id="layer_00", source=source, candidate=candidate
    )

    assert evidence["status"] == "passed"
    assert evidence["rewrites"][0]["proof_contract"] == (
        "parallel_mixed_head_norm_rope_exact_bf16.v1"
    )


def test_exact_candidate_gate_proves_fused_linear_scalar_gate() -> None:
    source = {
        "schema": "nerve.stream_circuit.v1",
        "source": {"component_id": "layer_00"},
        "boundary": {
            "inputs": [
                {"id": "normalized", "source": "normalized"},
                {"id": "value", "source": "value"},
            ],
            "outputs": [{"id": "gated", "source": "gated"}],
        },
        "state_ports": [],
        "parameters": {
            "refs": {"gate_weight": {"tensor": "gate.weight"}}
        },
        "behavioral_error_contract": {"mode": "source_reference_circuit"},
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
                "outputs": ["gated"],
            },
        ],
    }
    candidate = deepcopy(source)
    candidate["nodes"] = [
        {
            "id": "gate_projection__apply_gate",
            "op": "linear_sigmoid_scalar_multiply",
            "inputs": ["normalized", "value"],
            "outputs": ["gated"],
            "params": ["gate_weight"],
            "attrs": {
                "compiled_from": ["gate_projection", "apply_gate"],
                "intermediate_rounding": "BF16",
            },
        }
    ]

    evidence = prove_exact_circuit_candidate(
        component_id="layer_00",
        source=source,
        candidate=candidate,
    )

    assert evidence["status"] == "passed"
    assert (
        evidence["rewrites"][0]["proof_contract"]
        == "linear_sigmoid_scalar_multiply_exact_bf16.v1"
    )


def test_exact_candidate_gate_proves_fused_scalar_gate_residual_chain() -> None:
    source = {
        "schema": "nerve.stream_circuit.v1",
        "source": {"component_id": "layer_00"},
        "boundary": {
            "inputs": [
                {"id": signal, "source": signal}
                for signal in (
                    "normalized",
                    "shared_value",
                    "sparse_value",
                    "layer_residual",
                )
            ],
            "outputs": [{"id": "output", "source": "output"}],
        },
        "state_ports": [],
        "parameters": {
            "refs": {"gate_weight": {"tensor": "gate.weight"}}
        },
        "behavioral_error_contract": {"mode": "source_reference_circuit"},
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
        ],
    }
    candidate = deepcopy(source)
    candidate["nodes"] = [
        {
            "id": (
                "gate_projection__apply_gate__add_sparse__"
                "add_layer_residual"
            ),
            "op": "linear_sigmoid_scalar_multiply_residual2",
            "inputs": [
                "normalized",
                "shared_value",
                "sparse_value",
                "layer_residual",
            ],
            "outputs": ["output"],
            "params": ["gate_weight"],
            "attrs": {
                "compiled_from": [
                    "gate_projection",
                    "apply_gate",
                    "add_sparse",
                    "add_layer_residual",
                ],
                "intermediate_rounding": "BF16",
            },
        }
    ]

    evidence = prove_exact_circuit_candidate(
        component_id="layer_00",
        source=source,
        candidate=candidate,
    )

    assert evidence["status"] == "passed"
    assert (
        evidence["rewrites"][0]["proof_contract"]
        == "linear_sigmoid_scalar_multiply_residual2_exact_bf16.v1"
    )


def test_exact_candidate_gate_proves_fused_parallel_ffn_projection() -> None:
    source = {
        "schema": "nerve.stream_circuit.v1",
        "source": {"component_id": "layer_00"},
        "boundary": {
            "inputs": [{"id": "input", "source": "x"}],
            "outputs": [{"id": "output", "source": "y"}],
        },
        "state_ports": [],
        "parameters": {
            "refs": {
                "gate_weight": {"tensor": "gate.weight"},
                "up_weight": {"tensor": "up.weight"},
            }
        },
        "behavioral_error_contract": {"mode": "source_reference_circuit"},
        "nodes": [
            {
                "id": "gate",
                "op": "linear",
                "inputs": ["x"],
                "outputs": ["gate_projection"],
                "params": ["gate_weight"],
            },
            {
                "id": "up",
                "op": "linear",
                "inputs": ["x"],
                "outputs": ["up_projection"],
                "params": ["up_weight"],
            },
            {
                "id": "activation",
                "op": "silu",
                "inputs": ["gate_projection"],
                "outputs": ["activated"],
                "attrs": {"element_count": 4},
            },
            {
                "id": "multiply",
                "op": "multiply",
                "inputs": ["activated", "up_projection"],
                "outputs": ["y"],
            },
        ],
    }
    candidate = deepcopy(source)
    candidate["nodes"] = [
        {
            "id": "fused_ffn",
            "op": "parallel_linear_silu_multiply",
            "inputs": ["x"],
            "outputs": ["y"],
            "params": ["gate_weight", "up_weight"],
            "attrs": {
                "compiled_from": ["gate", "up", "activation", "multiply"],
                "branch_count": 2,
                "intermediate_rounding": "BF16",
                "element_count": 4,
            },
        }
    ]

    evidence = prove_exact_circuit_candidate(
        component_id="layer_00", source=source, candidate=candidate
    )

    assert evidence["status"] == "passed"
    assert evidence["candidate_kind"] == "exact_reference"
    assert (
        evidence["rewrites"][0]["proof_contract"]
        == "parallel_linear_silu_multiply_exact_bf16.v1"
    )


def test_exact_candidate_gate_proves_contiguous_linear_swiglu() -> None:
    source = {
        "schema": "nerve.stream_circuit.v1",
        "source": {"component_id": "layer_00"},
        "boundary": {
            "inputs": [{"id": "input", "source": "x"}],
            "outputs": [{"id": "output", "source": "y"}],
        },
        "state_ports": [],
        "parameters": {
            "refs": {
                "weight": {"tensor": "gate_up.weight"},
                "scale": {"tensor": "gate_up.weight_scale_inv"},
            }
        },
        "behavioral_error_contract": {"mode": "source_reference_circuit"},
        "nodes": [
            {
                "id": "projection",
                "op": "linear",
                "inputs": ["x"],
                "outputs": ["gate_up"],
                "params": ["weight", "scale"],
            },
            {
                "id": "split",
                "op": "split",
                "inputs": ["gate_up"],
                "outputs": ["gate", "up"],
                "attrs": {"part_width": 8},
            },
            {
                "id": "activation",
                "op": "silu_multiply",
                "inputs": ["gate", "up"],
                "outputs": ["y"],
                "attrs": {"element_count": 8},
            },
        ],
    }
    candidate = deepcopy(source)
    candidate["nodes"] = [
        {
            "id": "fused_swiglu",
            "op": "contiguous_linear_swiglu",
            "inputs": ["x"],
            "outputs": ["y"],
            "params": ["weight", "scale"],
            "attrs": {
                "compiled_from": ["projection", "split", "activation"],
                "part_width": 8,
                "weight_partition": "contiguous_gate_up",
                "intermediate_rounding": "BF16",
            },
        }
    ]

    evidence = prove_exact_circuit_candidate(
        component_id="layer_00", source=source, candidate=candidate
    )

    assert evidence["status"] == "passed"
    assert (
        evidence["rewrites"][0]["proof_contract"]
        == "contiguous_linear_swiglu_exact_bf16.v1"
    )


def test_exact_candidate_gate_proves_fp8_parallel_linear_parameter_pairs() -> None:
    source = {
        "schema": "nerve.stream_circuit.v1",
        "source": {"component_id": "layer_00"},
        "boundary": {
            "inputs": [{"id": "input", "source": "x"}],
            "outputs": [
                {"id": "query", "source": "q"},
                {"id": "key", "source": "k"},
            ],
        },
        "state_ports": [],
        "parameters": {
            "refs": {
                "q_weight": {"tensor": "q.weight"},
                "q_weight_scale_inv": {"tensor": "q.weight_scale_inv"},
                "k_weight": {"tensor": "k.weight"},
                "k_weight_scale_inv": {"tensor": "k.weight_scale_inv"},
            }
        },
        "behavioral_error_contract": {"mode": "source_reference_circuit"},
        "nodes": [
            {
                "id": "q_projection",
                "op": "linear",
                "inputs": ["x"],
                "outputs": ["q"],
                "params": ["q_weight", "q_weight_scale_inv"],
            },
            {
                "id": "k_projection",
                "op": "linear",
                "inputs": ["x"],
                "outputs": ["k"],
                "params": ["k_weight", "k_weight_scale_inv"],
            },
        ],
    }
    candidate = deepcopy(source)
    candidate["nodes"] = [
        {
            "id": "q_projection__k_projection__quantize_input",
            "op": "quantize_fp8_e4m3",
            "inputs": ["x"],
            "outputs": ["x_fp8", "x_scale"],
            "attrs": {
                "physical_representation_contract": (
                    "bf16_blockwise_fp8_e4m3_f32_scale.v1"
                ),
                "consumer_node_ids": ["q_projection__k_projection"],
                "semantic_source_node_ids": ["q_projection", "k_projection"],
                "element_count": 5120,
                "block_columns": 128,
                "output_element_bytes": [1, 4],
            },
        },
        {
            "id": "q_projection__k_projection",
            "op": "parallel_linear_2way",
            "inputs": ["x_fp8", "x_scale"],
            "outputs": ["q", "k"],
            "params": [
                "q_weight",
                "q_weight_scale_inv",
                "k_weight",
                "k_weight_scale_inv",
            ],
            "attrs": {
                "compiled_from": ["q_projection", "k_projection"],
                "branch_count": 2,
                "branch_parameter_counts": [2, 2],
                "physical_input_contract": (
                    "bf16_blockwise_fp8_e4m3_f32_scale.v1"
                ),
                "physical_input_provider_id": (
                    "q_projection__k_projection__quantize_input"
                ),
                "physical_logical_inputs": ["x"],
                "output_element_bytes": [2, 2],
            },
        }
    ]

    evidence = prove_exact_circuit_candidate(
        component_id="layer_00", source=source, candidate=candidate
    )

    assert evidence["status"] == "passed"
    assert evidence["candidate_kind"] == "exact_reference"
    assert evidence["physical_representation_count"] == 1
    assert evidence["rewrites"][0]["proof_contract"] == "parallel_linear_exact_bf16.v1"


def test_exact_candidate_gate_accepts_fused_physical_representation_provider() -> None:
    source = {
        "nodes": [
            {
                "id": "normalization",
                "op": "rms_norm",
                "inputs": ["hidden"],
                "outputs": ["normalized"],
                "params": ["norm_weight"],
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
    candidate = deepcopy(source)
    quantized_outputs = ["normalized_fp8", "normalized_scale"]
    candidate["nodes"][0]["outputs"].extend(quantized_outputs)
    candidate["nodes"][0]["attrs"].update(
        {
            "output_element_bytes": [2, 1, 4],
            "physical_output_representations": [
                {
                    "contract": "bf16_blockwise_fp8_e4m3_f32_scale.v1",
                    "logical_signal": "normalized",
                    "outputs": quantized_outputs,
                    "consumer_node_ids": ["projection"],
                    "element_count": 5120,
                    "block_columns": 128,
                }
            ],
        }
    )
    candidate["nodes"][1]["inputs"] = [*quantized_outputs, "normalized"]
    candidate["nodes"][1]["attrs"] = {
        "output_element_bytes": [2],
        "physical_input_contract": "bf16_blockwise_fp8_e4m3_f32_scale.v1",
        "physical_input_provider_id": "normalization",
        "physical_logical_inputs": ["normalized"],
        "physical_passthrough_inputs": ["normalized"],
    }

    evidence = prove_exact_circuit_candidate(
        component_id="layer_00", source=source, candidate=candidate
    )

    assert evidence["status"] == "passed"
    assert evidence["physical_representation_count"] == 1

    candidate["nodes"][1]["inputs"].append("normalized")
    candidate["nodes"][1]["attrs"]["physical_passthrough_inputs"].append("normalized")
    with pytest.raises(ModelCompileError, match="invalid physical representation provider"):
        prove_exact_circuit_candidate(
            component_id="layer_00", source=source, candidate=candidate
        )


def test_exact_candidate_gate_accepts_route_aware_sparse_moe_representation() -> None:
    contract = (
        "bf16_sparse_moe_intermediate_blockwise_fp8_e4m3_f32_scale_"
        "u32_route_map.v1"
    )
    source = {
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
            }
        ]
    }
    physical_outputs = [
        "expert_intermediate_fp8",
        "expert_intermediate_scale",
        "expert_route_map",
    ]
    candidate = deepcopy(source)
    candidate["nodes"][0]["outputs"].extend(physical_outputs)
    candidate["nodes"][0]["attrs"].update(
        {
            "output_element_bytes": [2, 1, 4, 4],
            "physical_output_representations": [
                {
                    "contract": contract,
                    "logical_signal": "expert_intermediates",
                    "outputs": physical_outputs,
                    "consumer_node_ids": ["sparse_moe_down"],
                    "element_count": 4096,
                    "block_columns": 128,
                    "experts_per_token": 8,
                }
            ],
        }
    )
    candidate["nodes"][1] = {
        **candidate["nodes"][1],
        "inputs": [*physical_outputs, "routes"],
        "attrs": {
            **candidate["nodes"][1]["attrs"],
            "physical_input_contract": contract,
            "physical_input_provider_id": "sparse_moe_gate_up",
            "physical_logical_inputs": [
                "expert_intermediates",
                "routes",
            ],
            "output_element_bytes": [2],
        },
    }

    evidence = prove_exact_circuit_candidate(
        component_id="layer_00", source=source, candidate=candidate
    )

    assert evidence["status"] == "passed"
    assert evidence["physical_representation_count"] == 1


def test_exact_candidate_gate_rejects_dropped_source_behavior() -> None:
    source = source_circuit()
    candidate = deepcopy(source)
    candidate["nodes"] = [deepcopy(source["nodes"][0])]

    with pytest.raises(ModelCompileError, match="does not exactly cover"):
        prove_exact_circuit_candidate(
            component_id="layer_00", source=source, candidate=candidate
        )


def test_exact_candidate_gate_rejects_reordered_interface_and_specialization() -> None:
    source = source_circuit()
    candidate = deepcopy(source)
    candidate["nodes"] = [
        {
            "id": "activation__multiply",
            "op": "silu_multiply",
            "inputs": ["gate", "x"],
            "outputs": ["y"],
            "attrs": {
                "compiled_from": ["activation", "multiply"],
                "intermediate_rounding": "BF16",
                "element_count": 4,
            },
        }
    ]
    with pytest.raises(ModelCompileError, match="observable region interface"):
        prove_exact_circuit_candidate(
            component_id="layer_00", source=source, candidate=candidate
        )

    candidate["nodes"][0]["inputs"] = ["x", "gate"]
    candidate["nodes"][0]["attrs"]["intermediate_rounding"] = "F32"
    with pytest.raises(ModelCompileError, match="exact rewrite attributes"):
        prove_exact_circuit_candidate(
            component_id="layer_00", source=source, candidate=candidate
        )


def test_hyper_connection_fusion_has_a_strict_exactness_proof() -> None:
    source, candidate = hyper_connection_source_and_candidate()
    proof = prove_exact_circuit_candidate(
        component_id="layer_00", source=source, candidate=candidate
    )
    assert proof["candidate_kind"] == "exact_reference"
    assert proof["rewrites"][0]["proof_contract"] == (
        "hyper_connection_pre_exact_bf16.v1"
    )

    candidate["nodes"][0]["attrs"]["sinkhorn_iterations"] = 19
    with pytest.raises(ModelCompileError, match="changes exact rewrite attributes"):
        prove_exact_circuit_candidate(
            component_id="layer_00", source=source, candidate=candidate
        )


def test_hyper_connection_norm_transaction_has_a_strict_exactness_proof() -> None:
    source, _candidate = hyper_connection_source_and_candidate()
    source["boundary"]["outputs"] = [
        {"id": "normalized", "source": "normalized"},
        {"id": "post", "source": "post"},
        {"id": "combination", "source": "combination"},
    ]
    source["parameters"]["refs"]["norm_weight"] = {"tensor": "norm.weight"}
    source["nodes"].append(
        {
            "id": "norm",
            "op": "rms_norm",
            "inputs": ["operator_input"],
            "outputs": ["normalized"],
            "params": ["norm_weight"],
            "attrs": {"eps": 1e-6, "weight_offset": 0.0},
        }
    )
    candidate = deepcopy(source)
    candidate["nodes"] = [
        {
            "id": "function__sinkhorn__reduce__norm",
            "op": "hyper_connection_pre_rms_norm",
            "inputs": ["input_frame"],
            "outputs": ["normalized", "post", "combination"],
            "params": ["function", "scale", "base", "norm_weight"],
            "attrs": {
                "compiled_from": ["function", "sinkhorn", "reduce", "norm"],
                "epsilon": 1e-6,
                "intermediate_rounding": "BF16",
                "multiplicity": 4,
                "normalization_epsilon": 1e-6,
                "sinkhorn_iterations": 20,
                "rms_norm_eps": 1e-6,
                "rms_norm_weight_offset": 0.0,
                "rms_norm_intermediate_rounding": "BF16",
            },
        }
    ]

    proof = prove_exact_circuit_candidate(
        component_id="layer_00", source=source, candidate=candidate
    )
    assert proof["candidate_kind"] == "exact_reference"
    assert proof["rewrites"][0]["proof_contract"] == (
        "hyper_connection_pre_rms_norm_exact_bf16.v1"
    )

    source_with_representation = deepcopy(source)
    source_with_representation["nodes"].append(
        {
            "id": "projection",
            "op": "linear",
            "inputs": ["normalized"],
            "outputs": ["projected"],
            "params": ["projection_weight", "projection_scale"],
        }
    )
    source_with_representation["boundary"]["outputs"].append(
        {"id": "projected", "source": "projected"}
    )
    represented_candidate = deepcopy(candidate)
    represented_candidate["boundary"] = deepcopy(
        source_with_representation["boundary"]
    )
    represented_candidate["nodes"][0]["outputs"].extend(
        ["normalized_fp8", "normalized_scale"]
    )
    represented_candidate["nodes"][0]["attrs"].update(
        {
            "output_element_bytes": [2, 4, 4, 1, 4],
            "physical_output_representations": [
                {
                    "contract": (
                        "bf16_blockwise_fp8_e4m3_e8m0_scale_f32.v1"
                    ),
                    "logical_signal": "normalized",
                    "outputs": ["normalized_fp8", "normalized_scale"],
                    "consumer_node_ids": ["projection"],
                    "element_count": 4096,
                    "block_columns": 128,
                }
            ],
        }
    )
    represented_candidate["nodes"].append(
        {
            "id": "projection",
            "op": "linear",
            "inputs": ["normalized_fp8", "normalized_scale"],
            "outputs": ["projected"],
            "params": ["projection_weight", "projection_scale"],
            "attrs": {
                "output_element_bytes": [2],
                "physical_input_contract": (
                    "bf16_blockwise_fp8_e4m3_e8m0_scale_f32.v1"
                ),
                "physical_input_provider_id": (
                    "function__sinkhorn__reduce__norm"
                ),
                "physical_logical_inputs": ["normalized"],
            },
        }
    )
    represented_proof = prove_exact_circuit_candidate(
        component_id="layer_00",
        source=source_with_representation,
        candidate=represented_candidate,
    )
    assert represented_proof["physical_representation_count"] == 1

    candidate["nodes"][0]["attrs"]["rms_norm_intermediate_rounding"] = "F32"
    with pytest.raises(ModelCompileError, match="changes exact rewrite attributes"):
        prove_exact_circuit_candidate(
            component_id="layer_00", source=source, candidate=candidate
        )


def test_approximate_candidate_requires_both_closed_loop_evidence_modes() -> None:
    source = source_circuit()
    candidate = deepcopy(source)
    candidate["boundary"]["outputs"][0]["source"] = "approximate_y"

    with pytest.raises(ModelCompileError, match="without source-oracle evidence"):
        prove_exact_circuit_candidate(
            component_id="layer_00", source=source, candidate=candidate
        )
    with pytest.raises(ModelCompileError, match="versioned source-oracle"):
        prove_exact_circuit_candidate(
            component_id="layer_00",
            source=source,
            candidate=candidate,
            empirical_evidence={
                "teacher_forced": {"status": "passed"},
                "free_running": {"status": "failed"},
            },
        )

    with pytest.raises(ModelCompileError, match="free-running"):
        prove_exact_circuit_candidate(
            component_id="layer_00",
            source=source,
            candidate=candidate,
            empirical_evidence=empirical_evidence(free_running_status="failed"),
        )

    evidence = prove_exact_circuit_candidate(
        component_id="layer_00",
        source=source,
        candidate=candidate,
        empirical_evidence=empirical_evidence(),
    )
    assert evidence["candidate_kind"] == "approximate"
    assert len(evidence["candidate_contract_digest"]) == 64


def test_behavioral_validation_accepts_mixed_exact_and_approximate_components() -> None:
    model_graph = {
        "architecture": {"family": "fixture"},
        "dimensions": {"hidden_size": 4},
        "numerics": {"activation_dtype": "BF16"},
        "graph": {"topology": "series"},
    }
    tensor_index = {
        "tensors": {},
        "totals": {"parameter_count": 0, "byte_count": 0},
    }
    source_exact = source_circuit()
    source_approximate = deepcopy(source_exact)
    source_approximate["source"]["component_id"] = "layer_01"
    candidate_exact = fused_candidate(source_exact)
    candidate_approximate = deepcopy(source_approximate)
    candidate_approximate["boundary"]["outputs"][0]["source"] = "approximate_y"
    empirical = empirical_evidence()
    empirical["model_contract_digest"] = model_contract_digest(
        model_graph, tensor_index
    )

    validation = build_behavioral_validation(
        model_graph=model_graph,
        tensor_index=tensor_index,
        lowered_index={
            "graph": {
                "circuits": [
                    {"id": "layer_00"},
                    {"id": "layer_01"},
                ]
            }
        },
        source_circuits={
            "layer_00": source_exact,
            "layer_01": source_approximate,
        },
        candidate_circuits={
            "layer_00": candidate_exact,
            "layer_01": candidate_approximate,
        },
        empirical_evidence=empirical,
    )

    assert validation["candidate_kind"] == "approximate"
    assert [proof["candidate_kind"] for proof in validation["circuits"]] == [
        "exact_reference",
        "approximate",
    ]
    validate_behavioral_validation_artifact(
        validation,
        {"layer_00": candidate_exact, "layer_01": candidate_approximate},
    )

    mislabeled = deepcopy(validation)
    mislabeled["circuits"][1].update(
        {
            "candidate_kind": "exact_reference",
            "source_node_count": 2,
            "covered_source_node_count": 2,
        }
    )
    with pytest.raises(ModelCompileError, match="no approximate component proof"):
        validate_behavioral_validation_artifact(
            mislabeled,
            {"layer_00": candidate_exact, "layer_01": candidate_approximate},
        )
