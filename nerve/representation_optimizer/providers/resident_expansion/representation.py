from __future__ import annotations

from nerve.compilation import Json
from nerve.representation_optimizer.providers.resident_expansion.artifacts import (
    PROOF_PATH,
)
from nerve.representation_optimizer.providers.resident_expansion.discovery import (
    ResidentExpansionOpportunity,
)
from nerve.representation_optimizer.representation_ir import (
    REPRESENTATION_GRAPH_SCHEMA,
    finalize_representation_graph,
)


def resident_expansion_representation_graph(
    *,
    candidate: Json,
    opportunity: ResidentExpansionOpportunity,
    capability_class: str,
) -> Json:
    scope_ids = list(opportunity.scope_ids)
    evidence_ids = list(opportunity.evidence_ids)
    source_node_ids = list(opportunity.node_ids)

    def provenance() -> Json:
        return {
            "scope_ids": scope_ids,
            "source_node_ids": source_node_ids,
            "evidence_refs": evidence_ids,
            "transform_refs": ["transform.exact_resident_expansion"],
        }

    logical_contracts = [
        {
            "id": "logical.input",
            "signal": "expert_bank_input",
            "shape": [opportunity.hidden_size],
            "dtype": "BF16",
        },
        {
            "id": "logical.output",
            "signal": "expert_bank_output",
            "shape": [opportunity.hidden_size],
            "dtype": "BF16",
        },
        {
            "id": "logical.expert_parameters",
            "signal": "selected_exact_expert_parameters",
            "shape": [opportunity.source_weight_bytes * 2],
            "dtype": "MXFP4_E2M1",
        },
    ]
    physical_representations = [
        {
            "id": "repr.signal.bf16",
            "kind": "dense_tensor",
            "domain": "signal",
            "physical_shape": [opportunity.hidden_size],
            "encoding": {"dtype": "BF16"},
            "storage": {"layout": "contiguous"},
        },
        {
            "id": "repr.parameter.compact_mxfp4",
            "kind": "selector_addressed_compact_parameter_bank",
            "domain": "parameter",
            "physical_shape": [opportunity.source_weight_bytes],
            "encoding": {
                "dtype": "packed_mxfp4_e2m1",
                "values_per_byte": 2,
                "group_size": 32,
                "scale_dtype": "F8_E8M0",
            },
            "storage": {"layout": "source_resource_ranges"},
        },
        {
            "id": "repr.parameter.resident_fp8",
            "kind": "selector_addressed_derived_parameter_bank",
            "domain": "parameter",
            "physical_shape": [opportunity.resident_weight_bytes],
            "encoding": {
                "dtype": "F8_E4M3",
                "source_dtype": "packed_mxfp4_e2m1",
                "mapping": "exact_finite_code_expansion",
            },
            "storage": {"layout": "demand_retained_component_local_resources"},
        },
    ]
    signals = [
        {
            "id": "signal.input",
            "logical_contract_id": "logical.input",
            "physical_representation_id": "repr.signal.bf16",
            "provenance": provenance(),
        },
        {
            "id": "signal.output",
            "logical_contract_id": "logical.output",
            "physical_representation_id": "repr.signal.bf16",
            "provenance": provenance(),
        },
    ]
    resources = [
        {
            "id": "resource.resident_expert_bank",
            "kind": "parameter",
            "logical_contract_id": "logical.expert_parameters",
            "physical_representation_id": "repr.parameter.resident_fp8",
            "artifact": {
                "path": opportunity.manifest_ref,
                "format": "runtime_derived_resident_resources.v1",
            },
            "provenance": provenance(),
        }
    ]
    node = {
        "id": "node.independent_sparse_expert_bank",
        "kind": "operator",
        "operation": "selector_addressed_exact_resident_sparse_expert_bank",
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
                "signal_id": "signal.output",
                "physical_representation_id": "repr.signal.bf16",
            }
        ],
        "resource_ids": ["resource.resident_expert_bank"],
        "state_read_ids": [],
        "state_write_ids": [],
        "cost": {
            "status": "estimated",
            "metrics": {
                "source_parameter_bytes": float(opportunity.source_weight_bytes),
                "fully_resident_parameter_bytes": float(
                    opportunity.resident_weight_bytes
                ),
                "experts_per_activation": float(opportunity.experts_per_token),
            },
        },
        "provenance": provenance(),
    }
    physical_kernels = [
        {
            "id": f"kernel.{index:02d}",
            "node_ids": ["node.independent_sparse_expert_bank"],
            "artifact": {
                "path": replacement.artifact_path,
                "format": "spirv",
            },
            "target_predicate": {
                "capability_class": capability_class,
                "execution_kind": replacement.execution_kind,
                "source_node_id": replacement.node_id,
            },
            "cost": {
                "status": "estimated",
                "metrics": {
                    "resident_parameter_byte_ratio": 2.0,
                },
            },
            "provenance": provenance(),
        }
        for index, replacement in enumerate(opportunity.shader_replacements)
    ]
    return finalize_representation_graph(
        {
            "schema": REPRESENTATION_GRAPH_SCHEMA,
            "graph_id": "",
            "candidate_id": candidate["candidate_id"],
            "scope_ids": scope_ids,
            "source_contract_digests": {
                scope_id: digest
                for scope_id, digest in zip(
                    opportunity.scope_ids,
                    opportunity.source_contract_digests,
                    strict=True,
                )
            },
            "logical_contracts": sorted(
                logical_contracts,
                key=lambda item: item["id"],
            ),
            "physical_representations": sorted(
                physical_representations,
                key=lambda item: item["id"],
            ),
            "signals": sorted(signals, key=lambda item: item["id"]),
            "resources": sorted(resources, key=lambda item: item["id"]),
            "nodes": [node],
            "connections": [],
            "public_ports": [
                {
                    "id": "port.input",
                    "direction": "input",
                    "logical_contract_id": "logical.input",
                    "signal_id": "signal.input",
                    "node_id": node["id"],
                    "node_port_id": "input",
                },
                {
                    "id": "port.output",
                    "direction": "output",
                    "logical_contract_id": "logical.output",
                    "signal_id": "signal.output",
                    "node_id": node["id"],
                    "node_port_id": "output",
                },
            ],
            "islands": [],
            "absorbed_transforms": [
                {
                    "id": "transform.exact_resident_expansion",
                    "kind": "exact_on_demand_mxfp4_e2m1_to_fp8_e4m3",
                    "source_representation_id": "repr.parameter.compact_mxfp4",
                    "target_representation_id": "repr.parameter.resident_fp8",
                    "adjacent_node_ids": [node["id"]],
                    "parameter_resource_ids": ["resource.resident_expert_bank"],
                    "proof_ref": PROOF_PATH,
                    "evidence_refs": evidence_ids,
                    "provenance": provenance(),
                }
            ],
            "physical_kernels": physical_kernels,
            "confidence": {
                "mode": "exact",
                "score": 1.0,
                "basis": (
                    "exhaustive finite MXFP4 code-domain equivalence and "
                    "sealed component-local resource coverage"
                ),
                "evidence_refs": evidence_ids,
            },
            "unresolved": [],
            "correction_requests": [],
        }
    )
