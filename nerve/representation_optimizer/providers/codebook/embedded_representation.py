from __future__ import annotations

from nerve.compilation import Json
from nerve.representation_optimizer.providers.codebook.discovery import (
    HeadNormCodebookOpportunity,
)
from nerve.representation_optimizer.providers.codebook.embedded_artifacts import (
    DECODE_SHADER_PATH,
    PROOF_PATH,
)
from nerve.representation_optimizer.representation_ir import (
    REPRESENTATION_GRAPH_SCHEMA,
    finalize_representation_graph,
)


def embedded_parameter_program_representation_graph(
    *,
    candidate: Json,
    opportunity: HeadNormCodebookOpportunity,
    capability_class: str,
) -> Json:
    scope_id = opportunity.scope_id
    evidence_ids = list(opportunity.evidence_ids)
    node_id = "node.embedded_parameter_head_norm_rope"
    transform_id = "transform.embed_exact_head_norm_parameter_program"
    branch_widths = [
        int(branch.attrs["head_count"]) * opportunity.head_width
        for branch in opportunity.branches
    ]

    def provenance(*source_nodes: str) -> Json:
        return {
            "scope_ids": [scope_id],
            "source_node_ids": sorted(set(source_nodes)),
            "evidence_refs": evidence_ids,
            "transform_refs": [transform_id],
        }

    logical_contracts = [
        {
            "id": "logical.embedded_parameter_program",
            "signal": "exact_embedded_bf16_parameter_program",
            "shape": [sum(len(branch.raw_values) for branch in opportunity.branches)],
            "dtype": "BF16",
        }
    ]
    signals = []
    public_ports = []
    node_inputs = []
    node_outputs = []
    for index, (branch, width) in enumerate(
        zip(opportunity.branches, branch_widths, strict=True)
    ):
        for direction in ("input", "output"):
            logical_id = f"logical.branch_{index}_{direction}"
            signal_id = f"signal.branch_{index}_{direction}"
            port_id = f"{direction}_{index}"
            logical_contracts.append(
                {
                    "id": logical_id,
                    "signal": f"branch_{index}_{direction}",
                    "shape": [width],
                    "dtype": "BF16",
                }
            )
            signals.append(
                {
                    "id": signal_id,
                    "logical_contract_id": logical_id,
                    "physical_representation_id": "repr.signal.dense_bf16",
                    "provenance": provenance(branch.source_node_id),
                }
            )
            node_port = {
                "id": port_id,
                "signal_id": signal_id,
                "physical_representation_id": "repr.signal.dense_bf16",
            }
            (node_inputs if direction == "input" else node_outputs).append(node_port)
            public_ports.append(
                {
                    "id": f"port.branch_{index}_{direction}",
                    "direction": direction,
                    "logical_contract_id": logical_id,
                    "signal_id": signal_id,
                    "node_id": node_id,
                    "node_port_id": port_id,
                }
            )

    physical_representations = [
        {
            "id": "repr.parameter.dense_bf16",
            "kind": "dense_tensor",
            "domain": "parameter",
            "physical_shape": [2, opportunity.head_width],
            "encoding": {"dtype": "BF16"},
            "storage": {"layout": "row_major"},
        },
        {
            "id": "repr.signal.dense_bf16",
            "kind": "dense_tensor",
            "domain": "signal",
            "physical_shape": [sum(branch_widths)],
            "encoding": {"dtype": "BF16"},
            "storage": {"layout": "branch_contiguous"},
        },
        {
            "id": "repr.parameter.embedded_parameter_program",
            "kind": "target_shader_constant_program",
            "domain": "parameter",
            "physical_shape": [
                sum(len(branch.raw_values) for branch in opportunity.branches)
            ],
            "encoding": {
                "entry_dtype": "BF16",
                "branch_count": len(opportunity.branches),
                "elements_per_branch": opportunity.head_width,
            },
            "storage": {
                "layout": "spirv_constant_program",
                "lifetime": "pipeline",
            },
        },
    ]
    resource = {
        "id": "resource.embedded_parameter_program",
        "kind": "parameter",
        "logical_contract_id": "logical.embedded_parameter_program",
        "physical_representation_id": "repr.parameter.embedded_parameter_program",
        "artifact": {
            "path": DECODE_SHADER_PATH,
            "format": "spirv_with_embedded_exact_bf16_parameters",
        },
        "provenance": provenance(
            *(branch.source_node_id for branch in opportunity.branches)
        ),
    }
    graph = {
        "schema": REPRESENTATION_GRAPH_SCHEMA,
        "graph_id": "",
        "candidate_id": candidate["candidate_id"],
        "scope_ids": [scope_id],
        "source_contract_digests": {
            scope_id: opportunity.source_contract_digest,
        },
        "logical_contracts": sorted(logical_contracts, key=lambda item: item["id"]),
        "physical_representations": sorted(
            physical_representations, key=lambda item: item["id"]
        ),
        "signals": sorted(signals, key=lambda item: item["id"]),
        "resources": [resource],
        "nodes": [
            {
                "id": node_id,
                "kind": "operator",
                "operation": (
                    "decode_phase_parallel_head_norm_rope_with_embedded_exact_"
                    "bf16_parameters"
                ),
                "inputs": sorted(node_inputs, key=lambda item: item["id"]),
                "outputs": sorted(node_outputs, key=lambda item: item["id"]),
                "resource_ids": [resource["id"]],
                "state_read_ids": [],
                "state_write_ids": [],
                "cost": {
                    "status": "estimated",
                    "metrics": {
                        "decode_parameter_buffer_reads": 0.0,
                        "prefill_parameter_buffer_reads": 2.0,
                        "embedded_bf16_elements": float(
                            sum(
                                len(branch.raw_values)
                                for branch in opportunity.branches
                            )
                        ),
                    },
                },
                "provenance": provenance(
                    *(branch.source_node_id for branch in opportunity.branches)
                ),
            }
        ],
        "connections": [],
        "public_ports": sorted(public_ports, key=lambda item: item["id"]),
        "islands": [],
        "absorbed_transforms": [
            {
                "id": transform_id,
                "kind": "decode_only_exact_parameters_to_target_program",
                "source_representation_id": "repr.parameter.dense_bf16",
                "target_representation_id": (
                    "repr.parameter.embedded_parameter_program"
                ),
                "adjacent_node_ids": [node_id],
                "parameter_resource_ids": [resource["id"]],
                "proof_ref": PROOF_PATH,
                "evidence_refs": evidence_ids,
                "provenance": provenance(
                    *(branch.source_node_id for branch in opportunity.branches)
                ),
            }
        ],
        "physical_kernels": [
            _physical_kernel(
                kernel_id="kernel.decode",
                node_id=node_id,
                shader_path=DECODE_SHADER_PATH,
                execution_phase="decode",
                capability_class=capability_class,
                provenance=provenance(
                    *(branch.source_node_id for branch in opportunity.branches)
                ),
            ),
        ],
        "confidence": {
            "mode": "exact",
            "score": 1.0,
            "basis": (
                "exhaustive BF16 reconstruction and deterministic target-program "
                "lowering"
            ),
            "evidence_refs": evidence_ids,
        },
        "unresolved": [],
        "correction_requests": [],
    }
    return finalize_representation_graph(graph)


def _physical_kernel(
    *,
    kernel_id: str,
    node_id: str,
    shader_path: str,
    execution_phase: str,
    capability_class: str,
    provenance: Json,
) -> Json:
    return {
        "id": kernel_id,
        "node_ids": [node_id],
        "artifact": {"path": shader_path, "format": "spirv"},
        "target_predicate": {
            "capability_class": capability_class,
            "execution_phase": execution_phase,
        },
        "cost": {
            "status": "estimated",
            "metrics": {"runtime_parameter_buffer_reads": 0.0},
        },
        "provenance": provenance,
    }
