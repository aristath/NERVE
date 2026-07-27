from __future__ import annotations

from nerve.compilation import Json
from nerve.representation_optimizer.providers.codebook.artifacts import (
    BRANCH_INDEX_PATHS,
    CODEBOOK_TENSOR_PATH,
    DECODE_SHADER_PATH,
    PREFILL_SHADER_PATH,
)
from nerve.representation_optimizer.providers.codebook.discovery import (
    HeadNormCodebookOpportunity,
)
from nerve.representation_optimizer.representation_ir import (
    REPRESENTATION_GRAPH_SCHEMA,
    finalize_representation_graph,
)


def codebook_representation_graph(
    *,
    candidate: Json,
    opportunity: HeadNormCodebookOpportunity,
    capability_class: str,
) -> Json:
    scope_id = opportunity.scope_id
    evidence_ids = list(opportunity.evidence_ids)

    def provenance(*source_nodes: str) -> Json:
        return {
            "scope_ids": [scope_id],
            "source_node_ids": sorted(set(source_nodes)),
            "evidence_refs": evidence_ids,
            "transform_refs": ["transform.exact_codebook"],
        }

    branch_widths = [
        int(branch.attrs["head_count"]) * opportunity.head_width
        for branch in opportunity.branches
    ]
    logical_contracts = []
    physical_representations = [
        {
            "id": "repr.parameter.codebook_bf16",
            "kind": "exact_lookup_entries",
            "domain": "parameter",
            "physical_shape": [len(opportunity.codebook_storage_payload) // 2],
            "encoding": {
                "dtype": "BF16",
                "ordering": "raw_bits_ascending",
                "logical_entry_count": len(opportunity.codebook_values),
                "padding_entry_count": (
                    len(opportunity.codebook_storage_payload) // 2
                    - len(opportunity.codebook_values)
                ),
            },
            "storage": {"layout": "contiguous"},
        },
        {
            "id": "repr.parameter.dense_bf16",
            "kind": "dense_tensor",
            "domain": "parameter",
            "physical_shape": [2, opportunity.head_width],
            "encoding": {"dtype": "BF16"},
            "storage": {"layout": "row_major"},
        },
        {
            "id": "repr.parameter.indices_u8",
            "kind": "codebook_addresses",
            "domain": "parameter",
            "physical_shape": [
                2,
                len(opportunity.branches[0].index_storage_payload),
            ],
            "encoding": {
                "dtype": "U8",
                "entry_dtype": "BF16",
                "entry_count": len(opportunity.codebook_values),
                "logical_elements_per_branch": opportunity.head_width,
                "padding_elements_per_branch": (
                    len(opportunity.branches[0].index_storage_payload)
                    - opportunity.head_width
                ),
            },
            "storage": {"layout": "packed_u8"},
        },
        {
            "id": "repr.signal.dense_bf16",
            "kind": "dense_tensor",
            "domain": "signal",
            "physical_shape": [sum(branch_widths)],
            "encoding": {"dtype": "BF16"},
            "storage": {"layout": "branch_contiguous"},
        },
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
            port = {
                "id": port_id,
                "signal_id": signal_id,
                "physical_representation_id": "repr.signal.dense_bf16",
            }
            (node_inputs if direction == "input" else node_outputs).append(port)
            public_ports.append(
                {
                    "id": f"port.branch_{index}_{direction}",
                    "direction": direction,
                    "logical_contract_id": logical_id,
                    "signal_id": signal_id,
                    "node_id": "node.codebook_head_norm_rope",
                    "node_port_id": port_id,
                }
            )
        logical_contracts.append(
            {
                "id": f"logical.branch_{index}_weight",
                "signal": f"branch_{index}_normalization_weight",
                "shape": [opportunity.head_width],
                "dtype": "BF16",
            }
        )
    logical_contracts.append(
        {
            "id": "logical.codebook_entries",
            "signal": "shared_exact_codebook_entries",
            "shape": [len(opportunity.codebook_values)],
            "dtype": "BF16",
        }
    )
    resources = [
        {
            "id": f"resource.branch_{index}_indices",
            "kind": "parameter",
            "logical_contract_id": f"logical.branch_{index}_weight",
            "physical_representation_id": "repr.parameter.indices_u8",
            "artifact": {
                "path": BRANCH_INDEX_PATHS[index],
                "format": "safetensors_u8_codebook_addresses.v1",
            },
            "provenance": provenance(branch.source_node_id),
        }
        for index, branch in enumerate(opportunity.branches)
    ]
    resources.append(
        {
            "id": "resource.codebook",
            "kind": "parameter",
            "logical_contract_id": "logical.codebook_entries",
            "physical_representation_id": "repr.parameter.codebook_bf16",
            "artifact": {
                "path": CODEBOOK_TENSOR_PATH,
                "format": "safetensors_bf16_codebook.v1",
            },
            "provenance": provenance(
                *(branch.source_node_id for branch in opportunity.branches)
            ),
        }
    )
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
        "resources": sorted(resources, key=lambda item: item["id"]),
        "nodes": [
            {
                "id": "node.codebook_head_norm_rope",
                "kind": "operator",
                "operation": (
                    "parallel_head_norm_rope_with_exact_u8_codebook_parameters"
                ),
                "inputs": sorted(node_inputs, key=lambda item: item["id"]),
                "outputs": sorted(node_outputs, key=lambda item: item["id"]),
                "resource_ids": sorted(resource["id"] for resource in resources),
                "state_read_ids": [],
                "state_write_ids": [],
                "cost": {
                    "status": "estimated",
                    "metrics": {
                        "indexed_reads": float(2 * opportunity.head_width),
                        "parameter_bytes": float(opportunity.codebook_parameter_bytes),
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
                "id": "transform.exact_codebook",
                "kind": "exact_parameter_codebook_encoding",
                "source_representation_id": "repr.parameter.dense_bf16",
                "target_representation_id": "repr.parameter.indices_u8",
                "adjacent_node_ids": ["node.codebook_head_norm_rope"],
                "parameter_resource_ids": sorted(
                    resource["id"] for resource in resources
                ),
                "proof_ref": "proofs/codebook_equivalence.json",
                "evidence_refs": evidence_ids,
                "provenance": provenance(
                    *(branch.source_node_id for branch in opportunity.branches)
                ),
            }
        ],
        "physical_kernels": [
            {
                "id": "kernel.decode",
                "node_ids": ["node.codebook_head_norm_rope"],
                "artifact": {"path": DECODE_SHADER_PATH, "format": "spirv"},
                "target_predicate": {
                    "capability_class": capability_class,
                    "execution_phase": "decode",
                },
                "cost": {
                    "status": "estimated",
                    "metrics": {
                        "parameter_bytes": float(opportunity.codebook_parameter_bytes)
                    },
                },
                "provenance": provenance(
                    *(branch.source_node_id for branch in opportunity.branches)
                ),
            },
            {
                "id": "kernel.prefill",
                "node_ids": ["node.codebook_head_norm_rope"],
                "artifact": {"path": PREFILL_SHADER_PATH, "format": "spirv"},
                "target_predicate": {
                    "capability_class": capability_class,
                    "execution_phase": "prefill",
                },
                "cost": {
                    "status": "estimated",
                    "metrics": {
                        "parameter_bytes": float(opportunity.codebook_parameter_bytes)
                    },
                },
                "provenance": provenance(
                    *(branch.source_node_id for branch in opportunity.branches)
                ),
            },
        ],
        "confidence": {
            "mode": "exact",
            "score": 1.0,
            "basis": "exhaustive BF16 bit-pattern reconstruction",
            "evidence_refs": evidence_ids,
        },
        "unresolved": [],
        "correction_requests": [],
    }
    return finalize_representation_graph(graph)
