from __future__ import annotations

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.providers.resident_expansion.artifacts import (
    PROOF_PATH,
    component_overlay_path,
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
    opportunities: tuple[ResidentExpansionOpportunity, ...],
    capability_class: str,
) -> Json:
    if not opportunities:
        raise ModelCompileError(
            "resident expansion representation requires component regions"
        )
    representative = opportunities[0]
    logical_contracts = []
    signals = []
    resources = []
    nodes = []
    public_ports = []
    transforms = []
    physical_kernels = []
    all_evidence_ids = set()

    for region_index, opportunity in enumerate(opportunities):
        prefix = f"region.{region_index:03d}"
        scope_ids = list(opportunity.scope_ids)
        evidence_ids = list(opportunity.evidence_ids)
        source_node_ids = list(opportunity.node_ids)
        all_evidence_ids.update(evidence_ids)

        def provenance() -> Json:
            return {
                "scope_ids": scope_ids,
                "source_node_ids": source_node_ids,
                "evidence_refs": evidence_ids,
                "transform_refs": [f"transform.{prefix}"],
            }

        logical_input = f"logical.input.{prefix}"
        logical_output = f"logical.output.{prefix}"
        logical_parameters = f"logical.parameters.{prefix}"
        input_signal = f"signal.input.{prefix}"
        output_signal = f"signal.output.{prefix}"
        resource_id = f"resource.parameters.{prefix}"
        node_id = f"node.expert_bank.{prefix}"
        logical_contracts.extend(
            (
                {
                    "id": logical_input,
                    "signal": "expert_bank_input",
                    "shape": [opportunity.hidden_size],
                    "dtype": "BF16",
                },
                {
                    "id": logical_output,
                    "signal": "expert_bank_output",
                    "shape": [opportunity.hidden_size],
                    "dtype": "BF16",
                },
                {
                    "id": logical_parameters,
                    "signal": "selected_exact_expert_parameters",
                    "shape": [opportunity.source_weight_bytes * 2],
                    "dtype": "MXFP4_E2M1",
                },
            )
        )
        signals.extend(
            (
                {
                    "id": input_signal,
                    "logical_contract_id": logical_input,
                    "physical_representation_id": "repr.signal.bf16",
                    "provenance": provenance(),
                },
                {
                    "id": output_signal,
                    "logical_contract_id": logical_output,
                    "physical_representation_id": "repr.signal.bf16",
                    "provenance": provenance(),
                },
            )
        )
        resources.append(
            {
                "id": resource_id,
                "kind": "parameter",
                "logical_contract_id": logical_parameters,
                "physical_representation_id": "repr.parameter.resident_fp8",
                "artifact": {
                    "path": component_overlay_path(opportunity.component_id),
                    "format": "nerve.optimizer.vulkan_component_overlay.v2",
                },
                "provenance": provenance(),
            }
        )
        nodes.append(
            {
                "id": node_id,
                "kind": "operator",
                "operation": ("selector_addressed_exact_resident_sparse_expert_bank"),
                "inputs": [
                    {
                        "id": "input",
                        "signal_id": input_signal,
                        "physical_representation_id": "repr.signal.bf16",
                    }
                ],
                "outputs": [
                    {
                        "id": "output",
                        "signal_id": output_signal,
                        "physical_representation_id": "repr.signal.bf16",
                    }
                ],
                "resource_ids": [resource_id],
                "state_read_ids": [],
                "state_write_ids": [],
                "cost": {
                    "status": "estimated",
                    "metrics": {
                        "source_parameter_bytes": float(
                            opportunity.source_weight_bytes
                        ),
                        "fully_resident_parameter_bytes": float(
                            opportunity.resident_weight_bytes
                        ),
                        "experts_per_activation": float(opportunity.experts_per_token),
                    },
                },
                "provenance": provenance(),
            }
        )
        public_ports.extend(
            (
                {
                    "id": f"port.input.{prefix}",
                    "direction": "input",
                    "logical_contract_id": logical_input,
                    "signal_id": input_signal,
                    "node_id": node_id,
                    "node_port_id": "input",
                },
                {
                    "id": f"port.output.{prefix}",
                    "direction": "output",
                    "logical_contract_id": logical_output,
                    "signal_id": output_signal,
                    "node_id": node_id,
                    "node_port_id": "output",
                },
            )
        )
        transforms.append(
            {
                "id": f"transform.{prefix}",
                "kind": "exact_on_demand_mxfp4_e2m1_to_fp8_e4m3",
                "source_representation_id": "repr.parameter.compact_mxfp4",
                "target_representation_id": "repr.parameter.resident_fp8",
                "adjacent_node_ids": [node_id],
                "parameter_resource_ids": [resource_id],
                "proof_ref": PROOF_PATH,
                "evidence_refs": evidence_ids,
                "provenance": provenance(),
            }
        )
        physical_kernels.extend(
            {
                "id": f"kernel.{region_index:03d}.{kernel_index:02d}",
                "node_ids": [node_id],
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
                    "metrics": {"resident_parameter_byte_ratio": 2.0},
                },
                "provenance": provenance(),
            }
            for kernel_index, replacement in enumerate(opportunity.shader_replacements)
        )

    return finalize_representation_graph(
        {
            "schema": REPRESENTATION_GRAPH_SCHEMA,
            "graph_id": "",
            "candidate_id": candidate["candidate_id"],
            "scope_ids": list(candidate["scope_ids"]),
            "source_contract_digests": {
                scope_id: digest
                for scope_id, digest in zip(
                    candidate["scope_ids"],
                    candidate["source_contract_digests"],
                    strict=True,
                )
            },
            "logical_contracts": sorted(logical_contracts, key=lambda item: item["id"]),
            "physical_representations": [
                {
                    "id": "repr.parameter.compact_mxfp4",
                    "kind": "selector_addressed_compact_parameter_bank",
                    "domain": "parameter",
                    "physical_shape": [representative.source_weight_bytes],
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
                    "physical_shape": [representative.resident_weight_bytes],
                    "encoding": {
                        "dtype": "F8_E4M3",
                        "source_dtype": "packed_mxfp4_e2m1",
                        "mapping": "exact_finite_code_expansion",
                    },
                    "storage": {"layout": "demand_retained_component_local_resources"},
                },
                {
                    "id": "repr.signal.bf16",
                    "kind": "dense_tensor",
                    "domain": "signal",
                    "physical_shape": [representative.hidden_size],
                    "encoding": {"dtype": "BF16"},
                    "storage": {"layout": "contiguous"},
                },
            ],
            "signals": sorted(signals, key=lambda item: item["id"]),
            "resources": sorted(resources, key=lambda item: item["id"]),
            "nodes": sorted(nodes, key=lambda item: item["id"]),
            "connections": [],
            "public_ports": sorted(public_ports, key=lambda item: item["id"]),
            "islands": [],
            "absorbed_transforms": sorted(transforms, key=lambda item: item["id"]),
            "physical_kernels": sorted(physical_kernels, key=lambda item: item["id"]),
            "confidence": {
                "mode": "exact",
                "score": 1.0,
                "basis": (
                    "exhaustive finite MXFP4 code-domain equivalence and "
                    "sealed component-local resource coverage"
                ),
                "evidence_refs": sorted(all_evidence_ids),
            },
            "unresolved": [],
            "correction_requests": [],
        }
    )
