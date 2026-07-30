from __future__ import annotations

from nerve.compilation import Json
from nerve.representation_optimizer.providers.output_fp8.artifacts import (
    BATCH_SHADER_PATH,
    DECODE_SHADER_PATH,
    ERROR_REPORT_PATH,
    SCALE_PATH,
    WEIGHT_PATH,
)
from nerve.representation_optimizer.providers.output_fp8.discovery import (
    OutputProjectionOpportunity,
)
from nerve.representation_optimizer.representation_ir import (
    REPRESENTATION_GRAPH_SCHEMA,
    finalize_representation_graph,
)


def output_projection_representation_graph(
    *,
    candidate: Json,
    opportunity: OutputProjectionOpportunity,
    capability_class: str,
) -> Json:
    scope_id = opportunity.scope_id
    evidence_refs = list(opportunity.evidence_ids)

    def provenance() -> Json:
        return {
            "scope_ids": [scope_id],
            "source_node_ids": list(opportunity.source_node_ids),
            "evidence_refs": evidence_refs,
            "transform_refs": ["transform.block_scaled_parameter"],
        }

    graph = {
        "schema": REPRESENTATION_GRAPH_SCHEMA,
        "graph_id": "",
        "candidate_id": candidate["candidate_id"],
        "scope_ids": [scope_id],
        "source_contract_digests": {
            scope_id: opportunity.source_contract_digest,
        },
        "logical_contracts": [
            {
                "id": "logical.input_frame",
                "signal": "output_transducer_input_frame",
                "shape": [opportunity.hidden_size],
                "dtype": "BF16",
            },
            {
                "id": "logical.logits",
                "signal": "output_transducer_logits",
                "shape": [opportunity.vocabulary_size],
                "dtype": "F32",
            },
            {
                "id": "logical.projection_scale",
                "signal": "output_projection_inverse_scale",
                "shape": list(opportunity.scale_shape),
                "dtype": "BF16",
            },
            {
                "id": "logical.projection_weight",
                "signal": "output_projection_weight",
                "shape": [
                    opportunity.vocabulary_size,
                    opportunity.hidden_size,
                ],
                "dtype": "BF16",
            },
        ],
        "physical_representations": [
            {
                "id": "repr.parameter.block_scale_bf16",
                "kind": "block_inverse_scale",
                "domain": "parameter",
                "physical_shape": list(opportunity.scale_shape),
                "encoding": {"dtype": "BF16"},
                "storage": {"layout": "row_major"},
            },
            {
                "id": "repr.parameter.dense_bf16",
                "kind": "dense_tensor",
                "domain": "parameter",
                "physical_shape": [
                    opportunity.vocabulary_size,
                    opportunity.hidden_size,
                ],
                "encoding": {"dtype": "BF16"},
                "storage": {"layout": "row_major"},
            },
            {
                "id": "repr.parameter.fp8_e4m3",
                "kind": "block_scaled_dense_tensor",
                "domain": "parameter",
                "physical_shape": [
                    opportunity.vocabulary_size,
                    opportunity.hidden_size,
                ],
                "encoding": {
                    "dtype": "F8_E4M3",
                    "block_rows": opportunity.block_rows,
                    "block_columns": opportunity.block_columns,
                },
                "storage": {"layout": "row_major"},
            },
            {
                "id": "repr.signal.bf16",
                "kind": "dense_numeric_vector",
                "domain": "signal",
                "physical_shape": [opportunity.hidden_size],
                "encoding": {"dtype": "BF16"},
                "storage": {"layout": "contiguous"},
            },
            {
                "id": "repr.signal.f32",
                "kind": "dense_numeric_vector",
                "domain": "signal",
                "physical_shape": [opportunity.vocabulary_size],
                "encoding": {"dtype": "F32"},
                "storage": {"layout": "contiguous"},
            },
        ],
        "signals": [
            {
                "id": "signal.input",
                "logical_contract_id": "logical.input_frame",
                "physical_representation_id": "repr.signal.bf16",
                "provenance": provenance(),
            },
            {
                "id": "signal.logits",
                "logical_contract_id": "logical.logits",
                "physical_representation_id": "repr.signal.f32",
                "provenance": provenance(),
            },
            {
                "id": "signal.quantized_logits",
                "logical_contract_id": "logical.logits",
                "physical_representation_id": "repr.signal.f32",
                "provenance": provenance(),
            },
        ],
        "resources": [
            {
                "id": "resource.projection_scale",
                "kind": "parameter",
                "logical_contract_id": "logical.projection_scale",
                "physical_representation_id": (
                    "repr.parameter.block_scale_bf16"
                ),
                "artifact": {
                    "path": SCALE_PATH,
                    "format": "safetensors",
                },
                "provenance": provenance(),
            },
            {
                "id": "resource.projection_weight",
                "kind": "parameter",
                "logical_contract_id": "logical.projection_weight",
                "physical_representation_id": "repr.parameter.fp8_e4m3",
                "artifact": {
                    "path": WEIGHT_PATH,
                    "format": "safetensors",
                },
                "provenance": provenance(),
            },
        ],
        "nodes": [
            {
                "id": "node.correction",
                "kind": "correction",
                "operation": "reject_candidate_and_retain_source",
                "inputs": [
                    {
                        "id": "input",
                        "signal_id": "signal.quantized_logits",
                        "physical_representation_id": "repr.signal.f32",
                    }
                ],
                "outputs": [
                    {
                        "id": "output",
                        "signal_id": "signal.logits",
                        "physical_representation_id": "repr.signal.f32",
                    }
                ],
                "resource_ids": [],
                "state_read_ids": [],
                "state_write_ids": [],
                "cost": None,
                "provenance": provenance(),
            },
            {
                "id": "node.projection",
                "kind": "operator",
                "operation": "block_scaled_fp8_e4m3_output_projection",
                "inputs": [
                    {
                        "id": "input",
                        "signal_id": "signal.input",
                        "physical_representation_id": "repr.signal.bf16",
                    }
                ],
                "outputs": [
                    {
                        "id": "output",
                        "signal_id": "signal.quantized_logits",
                        "physical_representation_id": "repr.signal.f32",
                    }
                ],
                "resource_ids": [
                    "resource.projection_scale",
                    "resource.projection_weight",
                ],
                "state_read_ids": [],
                "state_write_ids": [],
                "cost": {
                    "status": "estimated",
                    "metrics": {
                        "parameter_bytes": float(
                            opportunity.vocabulary_size
                            * opportunity.hidden_size
                            + opportunity.scale_shape[0]
                            * opportunity.scale_shape[1]
                            * 2
                        )
                    },
                },
                "provenance": provenance(),
            },
        ],
        "connections": [
            {
                "id": "connection.projected_logits",
                "producer": {
                    "node_id": "node.projection",
                    "port_id": "output",
                },
                "consumer": {
                    "node_id": "node.correction",
                    "port_id": "input",
                },
                "signal_id": "signal.quantized_logits",
                "materializes_source": False,
            }
        ],
        "public_ports": [
            {
                "id": "port.input",
                "direction": "input",
                "logical_contract_id": "logical.input_frame",
                "signal_id": "signal.input",
                "node_id": "node.projection",
                "node_port_id": "input",
            },
            {
                "id": "port.output",
                "direction": "output",
                "logical_contract_id": "logical.logits",
                "signal_id": "signal.logits",
                "node_id": "node.correction",
                "node_port_id": "output",
            },
        ],
        "islands": [],
        "absorbed_transforms": [
            {
                "id": "transform.block_scaled_parameter",
                "kind": "approximate_block_scaled_numeric_encoding",
                "source_representation_id": "repr.parameter.dense_bf16",
                "target_representation_id": "repr.parameter.fp8_e4m3",
                "adjacent_node_ids": ["node.projection"],
                "parameter_resource_ids": [
                    "resource.projection_scale",
                    "resource.projection_weight",
                ],
                "proof_ref": ERROR_REPORT_PATH,
                "evidence_refs": evidence_refs,
                "provenance": provenance(),
            }
        ],
        "physical_kernels": [
            {
                "id": "kernel.batch",
                "node_ids": ["node.projection"],
                "artifact": {"path": BATCH_SHADER_PATH, "format": "spirv"},
                "target_predicate": {
                    "capability_class": capability_class,
                    "execution_phase": "decode_batch",
                },
                "cost": {
                    "status": "estimated",
                    "metrics": {"parameter_reads": 1.0},
                },
                "provenance": provenance(),
            },
            {
                "id": "kernel.decode",
                "node_ids": ["node.projection"],
                "artifact": {"path": DECODE_SHADER_PATH, "format": "spirv"},
                "target_predicate": {
                    "capability_class": capability_class,
                    "execution_phase": "decode",
                },
                "cost": {
                    "status": "estimated",
                    "metrics": {"parameter_reads": 1.0},
                },
                "provenance": provenance(),
            },
        ],
        "confidence": {
            "mode": "verified_approximation",
            "score": 0.9,
            "basis": (
                "blockwise reconstruction error plus component and "
                "whole-model behavioral validation"
            ),
            "evidence_refs": evidence_refs,
        },
        "unresolved": [],
        "correction_requests": [
            {
                "id": "correction.retain_source",
                "trigger": {
                    "kind": "behavioral_error_contract_exceeded",
                },
                "correction_node_id": "node.correction",
                "fallback_scope_ids": [scope_id],
                "output_port_ids": ["port.output"],
                "error_contract": {
                    "action": "reject_candidate",
                    "fallback": "source_implementation",
                },
                "provenance": provenance(),
            }
        ],
    }
    return finalize_representation_graph(graph)
