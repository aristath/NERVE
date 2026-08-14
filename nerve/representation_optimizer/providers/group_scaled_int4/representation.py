from __future__ import annotations

from nerve.compilation import Json
from nerve.representation_optimizer.providers.group_scaled_int4.artifacts import (
    REPORT_PATH,
    scale_artifact_path,
    weight_artifact_path,
)
from nerve.representation_optimizer.providers.group_scaled_int4.discovery import (
    GroupScaledInt4Opportunity,
)
from nerve.representation_optimizer.providers.group_scaled_int4.physical import (
    ShaderArtifact,
)
from nerve.representation_optimizer.representation_ir import (
    REPRESENTATION_GRAPH_SCHEMA,
    finalize_representation_graph,
)


def group_scaled_int4_representation_graph(
    *,
    candidate: Json,
    opportunities: tuple[GroupScaledInt4Opportunity, ...],
    shader_artifacts: tuple[ShaderArtifact, ...],
    capability_class: str,
) -> Json:
    logical_contracts = []
    physical_representations = []
    signals = []
    resources = []
    nodes = []
    connections = []
    public_ports = []
    absorbed_transforms = []
    correction_requests = []
    projection_node_ids = []

    for opportunity in opportunities:
        token = opportunity.region_id
        prefix = f"region.{token}"
        scope_id = opportunity.scope_id

        def provenance() -> Json:
            return {
                "scope_ids": [scope_id],
                "source_node_ids": [opportunity.node_id],
                "evidence_refs": list(opportunity.evidence_ids),
                "transform_refs": [f"transform.{token}.quantize"],
            }

        logical_contracts.extend(
            (
                {
                    "id": f"logical.{token}.input",
                    "signal": f"{prefix}.input",
                    "shape": [opportunity.input_features],
                    "dtype": "BF16",
                },
                {
                    "id": f"logical.{token}.output",
                    "signal": f"{prefix}.output",
                    "shape": [opportunity.output_features],
                    "dtype": "BF16",
                },
                {
                    "id": f"logical.{token}.scale",
                    "signal": f"{prefix}.parameter_scale",
                    "shape": list(opportunity.scale_shape),
                    "dtype": "BF16",
                },
                {
                    "id": f"logical.{token}.weight",
                    "signal": f"{prefix}.parameter_weight",
                    "shape": [
                        opportunity.output_features,
                        opportunity.input_features,
                    ],
                    "dtype": "BF16",
                },
            )
        )
        physical_representations.extend(
            (
                {
                    "id": f"repr.{token}.input_bf16",
                    "kind": "dense_numeric_vector",
                    "domain": "signal",
                    "physical_shape": [opportunity.input_features],
                    "encoding": {"dtype": "BF16"},
                    "storage": {"layout": "contiguous"},
                },
                {
                    "id": f"repr.{token}.output_bf16",
                    "kind": "dense_numeric_vector",
                    "domain": "signal",
                    "physical_shape": [opportunity.output_features],
                    "encoding": {"dtype": "BF16"},
                    "storage": {"layout": "contiguous"},
                },
                {
                    "id": f"repr.{token}.scale_bf16",
                    "kind": "group_scale",
                    "domain": "parameter",
                    "physical_shape": list(opportunity.scale_shape),
                    "encoding": {"dtype": "BF16"},
                    "storage": {"layout": "row_major"},
                },
                {
                    "id": f"repr.{token}.source_bf16",
                    "kind": "dense_tensor",
                    "domain": "parameter",
                    "physical_shape": [
                        opportunity.output_features,
                        opportunity.input_features,
                    ],
                    "encoding": {"dtype": "BF16"},
                    "storage": {"layout": "row_major"},
                },
                {
                    "id": f"repr.{token}.weight_int4",
                    "kind": "group_scaled_signed_integer_tensor",
                    "domain": "parameter",
                    "physical_shape": list(opportunity.packed_shape),
                    "encoding": {
                        "dtype": "I4",
                        "group_size": opportunity.group_size,
                        "signed_offset": 8,
                        "symmetric": True,
                    },
                    "storage": {
                        "dtype": "I32",
                        "layout": "row_major_eight_input_columns_per_word",
                    },
                },
            )
        )
        input_signal = f"signal.{token}.input"
        candidate_output_signal = f"signal.{token}.candidate_output"
        output_signal = f"signal.{token}.output"
        signals.extend(
            (
                {
                    "id": input_signal,
                    "logical_contract_id": f"logical.{token}.input",
                    "physical_representation_id": f"repr.{token}.input_bf16",
                    "provenance": provenance(),
                },
                {
                    "id": candidate_output_signal,
                    "logical_contract_id": f"logical.{token}.output",
                    "physical_representation_id": f"repr.{token}.output_bf16",
                    "provenance": provenance(),
                },
                {
                    "id": output_signal,
                    "logical_contract_id": f"logical.{token}.output",
                    "physical_representation_id": f"repr.{token}.output_bf16",
                    "provenance": provenance(),
                },
            )
        )
        scale_resource = f"resource.{token}.scale"
        weight_resource = f"resource.{token}.weight"
        resources.extend(
            (
                {
                    "id": scale_resource,
                    "kind": "parameter",
                    "logical_contract_id": f"logical.{token}.scale",
                    "physical_representation_id": f"repr.{token}.scale_bf16",
                    "artifact": {
                        "path": scale_artifact_path(
                            opportunity.component_id,
                            opportunity.node_id,
                        ),
                        "format": "safetensors",
                    },
                    "provenance": provenance(),
                },
                {
                    "id": weight_resource,
                    "kind": "parameter",
                    "logical_contract_id": f"logical.{token}.weight",
                    "physical_representation_id": f"repr.{token}.weight_int4",
                    "artifact": {
                        "path": weight_artifact_path(
                            opportunity.component_id,
                            opportunity.node_id,
                        ),
                        "format": "safetensors",
                    },
                    "provenance": provenance(),
                },
            )
        )
        projection_node = f"node.{token}.projection"
        correction_node = f"node.{token}.correction"
        projection_node_ids.append(projection_node)
        candidate_bytes = (
            opportunity.output_features * opportunity.input_features // 2
            + opportunity.output_features
            * (opportunity.input_features // opportunity.group_size)
            * 2
        )
        nodes.extend(
            (
                {
                    "id": correction_node,
                    "kind": "correction",
                    "operation": "reject_candidate_and_retain_source",
                    "inputs": [
                        {
                            "id": "input",
                            "signal_id": candidate_output_signal,
                            "physical_representation_id": (
                                f"repr.{token}.output_bf16"
                            ),
                        }
                    ],
                    "outputs": [
                        {
                            "id": "output",
                            "signal_id": output_signal,
                            "physical_representation_id": (
                                f"repr.{token}.output_bf16"
                            ),
                        }
                    ],
                    "resource_ids": [],
                    "state_read_ids": [],
                    "state_write_ids": [],
                    "cost": None,
                    "provenance": provenance(),
                },
                {
                    "id": projection_node,
                    "kind": "operator",
                    "operation": "group_scaled_signed_int4_linear",
                    "inputs": [
                        {
                            "id": "input",
                            "signal_id": input_signal,
                            "physical_representation_id": (
                                f"repr.{token}.input_bf16"
                            ),
                        }
                    ],
                    "outputs": [
                        {
                            "id": "output",
                            "signal_id": candidate_output_signal,
                            "physical_representation_id": (
                                f"repr.{token}.output_bf16"
                            ),
                        }
                    ],
                    "resource_ids": [scale_resource, weight_resource],
                    "state_read_ids": [],
                    "state_write_ids": [],
                    "cost": {
                        "status": "estimated",
                        "metrics": {"parameter_bytes": float(candidate_bytes)},
                    },
                    "provenance": provenance(),
                },
            )
        )
        connections.append(
            {
                "id": f"connection.{token}.candidate_output",
                "producer": {"node_id": projection_node, "port_id": "output"},
                "consumer": {"node_id": correction_node, "port_id": "input"},
                "signal_id": candidate_output_signal,
                "materializes_source": False,
            }
        )
        public_ports.extend(
            (
                {
                    "id": f"port.{token}.input",
                    "direction": "input",
                    "logical_contract_id": f"logical.{token}.input",
                    "signal_id": input_signal,
                    "node_id": projection_node,
                    "node_port_id": "input",
                },
                {
                    "id": f"port.{token}.output",
                    "direction": "output",
                    "logical_contract_id": f"logical.{token}.output",
                    "signal_id": output_signal,
                    "node_id": correction_node,
                    "node_port_id": "output",
                },
            )
        )
        transform_id = f"transform.{token}.quantize"
        absorbed_transforms.append(
            {
                "id": transform_id,
                "kind": "approximate_group_scaled_integer_encoding",
                "source_representation_id": f"repr.{token}.source_bf16",
                "target_representation_id": f"repr.{token}.weight_int4",
                "adjacent_node_ids": [projection_node],
                "parameter_resource_ids": [scale_resource, weight_resource],
                "proof_ref": REPORT_PATH,
                "evidence_refs": list(opportunity.evidence_ids),
                "provenance": provenance(),
            }
        )
        correction_requests.append(
            {
                "id": f"correction.{token}.retain_source",
                "trigger": {"kind": "behavioral_error_contract_exceeded"},
                "correction_node_id": correction_node,
                "fallback_scope_ids": [scope_id],
                "output_port_ids": [f"port.{token}.output"],
                "error_contract": {
                    "action": "reject_candidate",
                    "fallback": "source_implementation",
                },
                "provenance": provenance(),
            }
        )

    scope_digests = {
        opportunity.scope_id: opportunity.source_contract_digest
        for opportunity in opportunities
    }
    physical_kernels = [
        {
            "id": f"kernel.{index:03d}",
            "node_ids": sorted(projection_node_ids),
            "artifact": {"path": shader.artifact_path, "format": "spirv"},
            "target_predicate": {
                "capability_class": capability_class,
                "execution_phase": (
                    "decode_and_prefill"
                    if "_batch" in shader.template_name
                    else "decode"
                ),
            },
            "cost": {
                "status": "estimated",
                "metrics": {"parameter_reads": 1.0},
            },
            "provenance": {
                "scope_ids": sorted(scope_digests),
                "source_node_ids": sorted(
                    opportunity.node_id for opportunity in opportunities
                ),
                "evidence_refs": sorted(
                    {
                        evidence_id
                        for opportunity in opportunities
                        for evidence_id in opportunity.evidence_ids
                    }
                ),
                "transform_refs": sorted(
                    f"transform.{opportunity.region_id}.quantize"
                    for opportunity in opportunities
                ),
            },
        }
        for index, shader in enumerate(shader_artifacts)
    ]
    graph = {
        "schema": REPRESENTATION_GRAPH_SCHEMA,
        "graph_id": "",
        "candidate_id": candidate["candidate_id"],
        "scope_ids": sorted(scope_digests),
        "source_contract_digests": {
            scope_id: scope_digests[scope_id] for scope_id in sorted(scope_digests)
        },
        "logical_contracts": sorted(logical_contracts, key=lambda item: item["id"]),
        "physical_representations": sorted(
            physical_representations,
            key=lambda item: item["id"],
        ),
        "signals": sorted(signals, key=lambda item: item["id"]),
        "resources": sorted(resources, key=lambda item: item["id"]),
        "nodes": sorted(nodes, key=lambda item: item["id"]),
        "connections": sorted(connections, key=lambda item: item["id"]),
        "public_ports": sorted(public_ports, key=lambda item: item["id"]),
        "islands": [],
        "absorbed_transforms": sorted(
            absorbed_transforms,
            key=lambda item: item["id"],
        ),
        "physical_kernels": physical_kernels,
        "confidence": {
            "mode": "verified_approximation",
            "score": 0.8,
            "basis": (
                "groupwise reconstruction error, component route stability, "
                "and whole-model behavioral validation"
            ),
            "evidence_refs": sorted(
                {
                    evidence_id
                    for opportunity in opportunities
                    for evidence_id in opportunity.evidence_ids
                }
            ),
        },
        "unresolved": [],
        "correction_requests": sorted(
            correction_requests,
            key=lambda item: item["id"],
        ),
    }
    return finalize_representation_graph(graph)


__all__ = ["group_scaled_int4_representation_graph"]
