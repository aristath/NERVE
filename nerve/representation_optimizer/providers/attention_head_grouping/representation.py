from __future__ import annotations

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.providers.attention_head_grouping.artifacts import (
    component_overlay_path,
)
from nerve.representation_optimizer.providers.attention_head_grouping.discovery import (
    AttentionHeadGroupingOpportunity,
)
from nerve.representation_optimizer.providers.attention_head_grouping.physical import (
    PreparedGroupedAttention,
)
from nerve.representation_optimizer.representation_ir import (
    REPRESENTATION_GRAPH_SCHEMA,
    finalize_representation_graph,
)


def attention_head_grouping_representation_graph(
    *,
    candidate: Json,
    opportunities: tuple[AttentionHeadGroupingOpportunity, ...],
    prepared: tuple[PreparedGroupedAttention, ...],
    capability_class: str,
) -> Json:
    if not opportunities or len(opportunities) != len(prepared):
        raise ModelCompileError(
            "grouped-attention representation requires matching component alternatives"
        )
    query_heads = opportunities[0].query_heads
    head_width = opportunities[0].head_width
    logical_contracts = []
    signals = []
    resources = []
    nodes = []
    public_ports = []
    physical_kernels = []
    evidence_ids = set()
    for index, (opportunity, component) in enumerate(
        zip(opportunities, prepared, strict=True)
    ):
        prefix = f"component.{index:03d}"
        evidence = list(opportunity.evidence_ids)
        evidence_ids.update(evidence)

        def provenance() -> Json:
            return {
                "scope_ids": [opportunity.scope_id],
                "source_node_ids": [opportunity.source_node_id],
                "evidence_refs": evidence,
                "transform_refs": [],
            }

        input_contract = f"logical.input.{prefix}"
        local_state_contract = f"logical.local_state.{prefix}"
        output_contract = f"logical.output.{prefix}"
        topology_contract = f"logical.topology.{prefix}"
        input_signal = f"signal.input.{prefix}"
        local_state_signal = f"signal.local_state.{prefix}"
        output_signal = f"signal.output.{prefix}"
        topology_resource = f"resource.topology.{prefix}"
        node_id = f"node.grouped_attention.{prefix}"
        logical_contracts.extend(
            (
                {
                    "id": input_contract,
                    "signal": "positioned_query_heads",
                    "shape": [query_heads, head_width],
                    "dtype": "BF16",
                },
                {
                    "id": local_state_contract,
                    "signal": "local_attention_state",
                    "shape": [opportunity.local_window, head_width],
                    "dtype": "BF16",
                },
                {
                    "id": output_contract,
                    "signal": "attention_heads",
                    "shape": [query_heads, head_width],
                    "dtype": "BF16",
                },
                {
                    "id": topology_contract,
                    "signal": "exact_grouped_head_schedule",
                    "shape": [opportunity.head_group],
                    "dtype": "TOPOLOGY",
                },
            )
        )
        signals.extend(
            (
                {
                    "id": input_signal,
                    "logical_contract_id": input_contract,
                    "physical_representation_id": "repr.signal.query_bf16",
                    "provenance": provenance(),
                },
                {
                    "id": local_state_signal,
                    "logical_contract_id": local_state_contract,
                    "physical_representation_id": "repr.signal.local_state_bf16",
                    "provenance": provenance(),
                },
                {
                    "id": output_signal,
                    "logical_contract_id": output_contract,
                    "physical_representation_id": "repr.signal.query_bf16",
                    "provenance": provenance(),
                },
            )
        )
        resources.append(
            {
                "id": topology_resource,
                "kind": "topology",
                "logical_contract_id": topology_contract,
                "physical_representation_id": "repr.topology.component_region",
                "artifact": {
                    "path": component_overlay_path(opportunity.component_id),
                    "format": "nerve.optimizer.vulkan_component_region_overlay.v1",
                },
                "provenance": provenance(),
            }
        )
        node_inputs = [
            {
                "id": "query",
                "signal_id": input_signal,
                "physical_representation_id": "repr.signal.query_bf16",
            },
            {
                "id": "local_state",
                "signal_id": local_state_signal,
                "physical_representation_id": "repr.signal.local_state_bf16",
            },
        ]
        if opportunity.max_compressed_indices:
            compressed_state_contract = f"logical.compressed_state.{prefix}"
            compressed_indices_contract = f"logical.compressed_indices.{prefix}"
            compressed_state_signal = f"signal.compressed_state.{prefix}"
            compressed_indices_signal = f"signal.compressed_indices.{prefix}"
            logical_contracts.extend(
                (
                    {
                        "id": compressed_state_contract,
                        "signal": "compressed_attention_state",
                        "shape": [
                            opportunity.max_compressed_indices,
                            head_width,
                        ],
                        "dtype": "BF16",
                    },
                    {
                        "id": compressed_indices_contract,
                        "signal": "compressed_attention_indices",
                        "shape": [opportunity.max_compressed_indices],
                        "dtype": "U32",
                    },
                )
            )
            signals.extend(
                (
                    {
                        "id": compressed_state_signal,
                        "logical_contract_id": compressed_state_contract,
                        "physical_representation_id": (
                            "repr.signal.compressed_state_bf16"
                        ),
                        "provenance": provenance(),
                    },
                    {
                        "id": compressed_indices_signal,
                        "logical_contract_id": compressed_indices_contract,
                        "physical_representation_id": (
                            "repr.signal.compressed_indices_u32"
                        ),
                        "provenance": provenance(),
                    },
                )
            )
            node_inputs.extend(
                (
                    {
                        "id": "compressed_state",
                        "signal_id": compressed_state_signal,
                        "physical_representation_id": (
                            "repr.signal.compressed_state_bf16"
                        ),
                    },
                    {
                        "id": "compressed_indices",
                        "signal_id": compressed_indices_signal,
                        "physical_representation_id": (
                            "repr.signal.compressed_indices_u32"
                        ),
                    },
                )
            )
            public_ports.extend(
                (
                    {
                        "id": f"port.compressed_state.{prefix}",
                        "direction": "input",
                        "logical_contract_id": compressed_state_contract,
                        "signal_id": compressed_state_signal,
                        "node_id": node_id,
                        "node_port_id": "compressed_state",
                    },
                    {
                        "id": f"port.compressed_indices.{prefix}",
                        "direction": "input",
                        "logical_contract_id": compressed_indices_contract,
                        "signal_id": compressed_indices_signal,
                        "node_id": node_id,
                        "node_port_id": "compressed_indices",
                    },
                )
            )
        nodes.append(
            {
                "id": node_id,
                "kind": "operator",
                "operation": "exact_grouped_multi_query_attention_schedule",
                "inputs": sorted(node_inputs, key=lambda item: item["id"]),
                "outputs": [
                    {
                        "id": "output",
                        "signal_id": output_signal,
                        "physical_representation_id": "repr.signal.query_bf16",
                    }
                ],
                "resource_ids": [topology_resource],
                "state_read_ids": [],
                "state_write_ids": [],
                "cost": {
                    "status": "estimated",
                    "metrics": {
                        "source_state_reads_per_latent": float(query_heads * 2),
                        "candidate_state_reads_per_latent": float(
                            query_heads * 2 // opportunity.head_group
                        ),
                    },
                },
                "provenance": provenance(),
            }
        )
        public_ports.extend(
            (
                {
                    "id": f"port.query.{prefix}",
                    "direction": "input",
                    "logical_contract_id": input_contract,
                    "signal_id": input_signal,
                    "node_id": node_id,
                    "node_port_id": "query",
                },
                {
                    "id": f"port.local_state.{prefix}",
                    "direction": "input",
                    "logical_contract_id": local_state_contract,
                    "signal_id": local_state_signal,
                    "node_id": node_id,
                    "node_port_id": "local_state",
                },
                {
                    "id": f"port.output.{prefix}",
                    "direction": "output",
                    "logical_contract_id": output_contract,
                    "signal_id": output_signal,
                    "node_id": node_id,
                    "node_port_id": "output",
                },
            )
        )
        physical_kernels.extend(
            {
                "id": f"kernel.{index:03d}.{shader_index:02d}",
                "node_ids": [node_id],
                "artifact": {
                    "path": shader.artifact_path,
                    "format": "spirv",
                },
                "target_predicate": {
                    "capability_class": capability_class,
                    "template_name": shader.template_name,
                },
                "cost": {
                    "status": "estimated",
                    "metrics": {
                        "latent_state_read_reuse": float(opportunity.head_group),
                    },
                },
                "provenance": provenance(),
            }
            for shader_index, shader in enumerate(component.shader_artifacts)
        )
    physical_representations = [
        {
            "id": "repr.signal.query_bf16",
            "kind": "dense_tensor",
            "domain": "signal",
            "physical_shape": [query_heads, head_width],
            "encoding": {"dtype": "BF16"},
            "storage": {"layout": "per_head_row_major"},
        },
        {
            "id": "repr.signal.local_state_bf16",
            "kind": "dense_tensor",
            "domain": "signal",
            "physical_shape": [opportunities[0].local_window, head_width],
            "encoding": {"dtype": "BF16"},
            "storage": {"layout": "rolling_attention_memory"},
        },
        {
            "id": "repr.topology.component_region",
            "kind": "source_anchored_component_region",
            "domain": "topology",
            "physical_shape": [1],
            "encoding": {"schema": "vulkan_component_region_overlay.v1"},
            "storage": {"layout": "candidate_artifact"},
        },
    ]
    if opportunities[0].max_compressed_indices:
        physical_representations.extend(
            (
                {
                    "id": "repr.signal.compressed_state_bf16",
                    "kind": "dense_tensor",
                    "domain": "signal",
                    "physical_shape": [
                        opportunities[0].max_compressed_indices,
                        head_width,
                    ],
                    "encoding": {"dtype": "BF16"},
                    "storage": {"layout": "indexed_compressed_attention_memory"},
                },
                {
                    "id": "repr.signal.compressed_indices_u32",
                    "kind": "dense_tensor",
                    "domain": "signal",
                    "physical_shape": [
                        opportunities[0].max_compressed_indices
                    ],
                    "encoding": {"dtype": "U32"},
                    "storage": {"layout": "compressed_attention_index_order"},
                },
            )
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
            "nodes": sorted(nodes, key=lambda item: item["id"]),
            "connections": [],
            "public_ports": sorted(public_ports, key=lambda item: item["id"]),
            "islands": [],
            "absorbed_transforms": [],
            "physical_kernels": sorted(
                physical_kernels,
                key=lambda item: item["id"],
            ),
            "confidence": {
                "mode": "exact",
                "score": 1.0,
                "basis": (
                    "unchanged source operation and per-head arithmetic order; "
                    "only shared latent-state reads and workgroup scheduling change"
                ),
                "evidence_refs": sorted(evidence_ids),
            },
            "unresolved": [],
            "correction_requests": [],
        }
    )
