from __future__ import annotations

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.providers.parallel_projection_fusion.artifacts import (
    component_overlay_path,
)
from nerve.representation_optimizer.providers.parallel_projection_fusion.discovery import (
    ParallelProjectionFusionOpportunity,
)
from nerve.representation_optimizer.providers.parallel_projection_fusion.physical import (
    PreparedFusedComponent,
)
from nerve.representation_optimizer.representation_ir import (
    REPRESENTATION_GRAPH_SCHEMA,
    finalize_representation_graph,
)


def parallel_projection_representation_graph(
    *,
    candidate: Json,
    opportunities: tuple[ParallelProjectionFusionOpportunity, ...],
    prepared: tuple[PreparedFusedComponent, ...],
    capability_class: str,
) -> Json:
    if not opportunities or len(opportunities) != len(prepared):
        raise ModelCompileError(
            "parallel projection representation requires matching component alternatives"
        )
    hidden_size = opportunities[0].hidden_size
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
        scope_ids = sorted(opportunity.scope_ids)
        component_evidence = list(opportunity.evidence_ids)
        evidence_ids.update(component_evidence)
        source_node_ids = sorted(opportunity.region.semantic_source_node_ids)

        def provenance() -> Json:
            return {
                "scope_ids": scope_ids,
                "source_node_ids": source_node_ids,
                "evidence_refs": component_evidence,
                "transform_refs": [],
            }

        input_contract = f"logical.input.{prefix}"
        output_contract = f"logical.output.{prefix}"
        topology_contract = f"logical.topology.{prefix}"
        input_signal = f"signal.input.{prefix}"
        output_signal = f"signal.output.{prefix}"
        topology_resource = f"resource.topology.{prefix}"
        node_id = f"node.parallel_projection_island.{prefix}"
        logical_contracts.extend(
            (
                {
                    "id": input_contract,
                    "signal": "component_input_frame",
                    "shape": [4, hidden_size],
                    "dtype": "BF16",
                },
                {
                    "id": output_contract,
                    "signal": "component_output_frame",
                    "shape": [4, hidden_size],
                    "dtype": "BF16",
                },
                {
                    "id": topology_contract,
                    "signal": "exact_fused_component_region",
                    "shape": [1],
                    "dtype": "TOPOLOGY",
                },
            )
        )
        signals.extend(
            (
                {
                    "id": input_signal,
                    "logical_contract_id": input_contract,
                    "physical_representation_id": "repr.signal.bf16",
                    "provenance": provenance(),
                },
                {
                    "id": output_signal,
                    "logical_contract_id": output_contract,
                    "physical_representation_id": "repr.signal.bf16",
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
                    "path": component_overlay_path(
                        opportunity.component_id,
                        opportunity.physical_node_id,
                    ),
                    "format": "nerve.optimizer.vulkan_component_region_overlay.v1",
                },
                "provenance": provenance(),
            }
        )
        nodes.append(
            {
                "id": node_id,
                "kind": "operator",
                "operation": "exact_shared_input_parallel_projection_island",
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
                "resource_ids": [topology_resource],
                "state_read_ids": [],
                "state_write_ids": [],
                "cost": {
                    "status": "estimated",
                    "metrics": {
                        "source_dispatches": float(
                            1 + len(opportunity.region.linear_node_ids)
                        ),
                        "candidate_dispatches": 2.0,
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
                    "logical_contract_id": input_contract,
                    "signal_id": input_signal,
                    "node_id": node_id,
                    "node_port_id": "input",
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
                        "dispatch_reduction": float(
                            len(opportunity.region.linear_node_ids) - 1
                        )
                    },
                },
                "provenance": provenance(),
            }
            for shader_index, shader in enumerate(component.shader_artifacts)
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
                    "id": "repr.signal.bf16",
                    "kind": "dense_tensor",
                    "domain": "signal",
                    "physical_shape": [4, hidden_size],
                    "encoding": {"dtype": "BF16"},
                    "storage": {"layout": "component_native"},
                },
                {
                    "id": "repr.topology.component_region",
                    "kind": "source_anchored_component_region",
                    "domain": "topology",
                    "physical_shape": [1],
                    "encoding": {"schema": "vulkan_component_region_overlay.v1"},
                    "storage": {"layout": "candidate_artifact"},
                },
            ],
            "signals": sorted(signals, key=lambda item: item["id"]),
            "resources": sorted(resources, key=lambda item: item["id"]),
            "nodes": sorted(nodes, key=lambda item: item["id"]),
            "connections": [],
            "public_ports": sorted(public_ports, key=lambda item: item["id"]),
            "islands": [],
            "absorbed_transforms": [],
            "physical_kernels": sorted(physical_kernels, key=lambda item: item["id"]),
            "confidence": {
                "mode": "exact",
                "score": 1.0,
                "basis": (
                    "source-circuit exact-reference rewrite proof and sealed "
                    "component-region source anchors"
                ),
                "evidence_refs": sorted(evidence_ids),
            },
            "unresolved": [],
            "correction_requests": [],
        }
    )
