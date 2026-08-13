from __future__ import annotations

from copy import deepcopy

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.contracts import representation_candidate_id
from nerve.representation_optimizer.providers.attention_head_grouping.artifacts import (
    COMPONENT_FIXTURE_PATH,
    CONVERSATION_FIXTURE_PATH,
    MODEL_LIMITS_PATH,
    PRODUCT_CONVERSATION_FIXTURE_PATH,
    PROOF_PATH,
    artifact_paths,
    component_overlay_path,
)
from nerve.representation_optimizer.providers.attention_head_grouping.contracts import (
    EXACT_HEAD_GROUPING_OBLIGATIONS,
    HETEROGENEOUS_COMPOSITE_ISLAND_DESCRIPTOR_ID,
    PROOF_SCHEMA,
    TARGET_LOWERING_SCHEMA,
)
from nerve.representation_optimizer.providers.attention_head_grouping.discovery import (
    AttentionHeadGroupingOpportunity,
    discover_attention_head_groupings,
    discovery_result,
    is_attention_operator_scope,
    source_inputs,
)
from nerve.representation_optimizer.providers.attention_head_grouping.physical import (
    prepare_grouped_attention,
)
from nerve.representation_optimizer.providers.attention_head_grouping.representation import (
    attention_head_grouping_representation_graph,
)
from nerve.representation_optimizer.providers.attention_head_grouping.workloads import (
    attention_benchmark_workloads,
    attention_validation_requirements,
)
from nerve.representation_optimizer.providers.types import (
    EvidenceAssessment,
    MatchAssessment,
    ProviderContext,
    ProviderIdentity,
    StaticEstimate,
)
from nerve.representation_optimizer.validation.conversation_semantics import (
    VALIDATION_CONVERSATION_SCHEMA,
)


class ExactAttentionHeadGroupingProvider:
    identity = ProviderIdentity("nerve.exact_attention_head_grouping", "1")
    descriptor_id = HETEROGENEOUS_COMPOSITE_ISLAND_DESCRIPTOR_ID

    def may_optimize_scope(self, scope: Json, source_contract: Json) -> bool:
        return is_attention_operator_scope(scope, source_contract)

    def required_analyzer_ids(
        self,
        scope: Json,
        source_contract: Json,
    ) -> tuple[str, ...]:
        del scope, source_contract
        return ("semantic_graph_structure",)

    def match_semantics(self, context: ProviderContext) -> MatchAssessment:
        matched = any(
            is_attention_operator_scope(scope, contract)
            for scope, contract in zip(
                context.scopes,
                context.source_contracts,
                strict=True,
            )
        )
        return MatchAssessment(
            matched=matched,
            reasons=(
                "standalone indexed-attention operators may reuse shared latent reads"
                if matched
                else "no standalone indexed-attention operator is present",
            ),
        )

    def match_structure(self, context: ProviderContext) -> MatchAssessment:
        result = discovery_result(context)
        return MatchAssessment(
            matched=bool(result.opportunities),
            reasons=result.reasons,
            evidence_ids=result.evidence_ids,
        )

    def analyze_evidence(self, context: ProviderContext) -> EvidenceAssessment:
        result = discovery_result(context)
        if not result.opportunities:
            return EvidenceAssessment(
                accepted=False,
                evidence_ids=(),
                facts={},
                reasons=result.reasons,
            )
        return EvidenceAssessment(
            accepted=True,
            evidence_ids=result.evidence_ids,
            facts={
                "component_count": len(
                    {item.component_id for item in result.opportunities}
                ),
                "candidate_family_count": len(
                    {item.performance_signature for item in result.opportunities}
                ),
                "head_groups": sorted(
                    {item.head_group for item in result.opportunities}
                ),
                "source_execution": "one_workgroup_per_query_head",
                "candidate_execution": "one_workgroup_per_query_head_group",
                "selection_boundary": "component_region",
            },
            reasons=(
                "query-head geometry, shared KV geometry, exact source contract, and target capabilities agree",
            ),
        )

    def synthesize_candidates(
        self,
        context: ProviderContext,
        evidence: EvidenceAssessment,
    ) -> tuple[Json, ...]:
        accepted = set(evidence.evidence_ids)
        candidates = []
        for opportunities in _opportunity_groups(context):
            representative = opportunities[0]
            prepared = prepare_grouped_attention(context, representative)
            scope_contracts = sorted(
                (
                    (item.scope_id, item.source_contract_digest)
                    for item in opportunities
                ),
                key=lambda item: item[0],
            )
            overlay_paths = tuple(
                component_overlay_path(item.component_id) for item in opportunities
            )
            kernel_paths = tuple(
                shader.artifact_path for shader in prepared.shader_artifacts
            )
            candidate = {
                "schema": "nerve.optimizer.representation_candidate.v1",
                "candidate_id": "",
                "scope_ids": [scope_id for scope_id, _digest in scope_contracts],
                "source_contract_digests": [
                    digest for _scope_id, digest in scope_contracts
                ],
                "provider": self.identity.to_json(),
                "descriptor_id": self.descriptor_id,
                "evidence_refs": [
                    evidence_id
                    for evidence_id in sorted(
                        {
                            evidence_id
                            for item in opportunities
                            for evidence_id in item.evidence_ids
                        }
                    )
                    if evidence_id in accepted
                ],
                "representation": {
                    "kind": "exact_capability_scoped_attention_head_grouping",
                    "signal_formats": [
                        {"name": "source_bf16_query_and_latent_state"}
                    ],
                    "parameter_format": {"kind": "source_parameters_unchanged"},
                    "state_format": {"kind": "source_state_unchanged"},
                    "topology": {
                        "kind": "source_anchored_component_regions",
                        "component_ids": [
                            item.component_id for item in opportunities
                        ],
                        "region_count_per_component": 1,
                        "head_group": representative.head_group,
                        "performance_equivalence_class": (
                            representative.performance_signature
                        ),
                    },
                },
                "target_predicate": _target_predicate(context, representative),
                "behavioral_contract": {
                    "mode": "exact",
                    "proof_obligations": list(EXACT_HEAD_GROUPING_OBLIGATIONS),
                    "error_contract": None,
                },
                "artifact_declarations": [
                    {"path": path}
                    for path in artifact_paths(
                        overlay_paths=overlay_paths,
                        kernel_paths=kernel_paths,
                    )
                ],
            }
            candidate["candidate_id"] = representation_candidate_id(candidate)
            candidates.append(candidate)
        return tuple(candidates)

    def emit_representation_ir(
        self,
        context: ProviderContext,
        candidate: Json,
    ) -> Json:
        opportunities = _candidate_opportunities(context, candidate)
        representative = prepare_grouped_attention(context, opportunities[0])
        return attention_head_grouping_representation_graph(
            candidate=candidate,
            opportunities=opportunities,
            prepared=tuple(representative for _item in opportunities),
            capability_class=str(context.hardware_profile["capability_class"]),
        )

    def lower_for_target(
        self,
        context: ProviderContext,
        candidate: Json,
        representation_ir: Json,
    ) -> Json:
        opportunities = _candidate_opportunities(context, candidate)
        representative = prepare_grouped_attention(context, opportunities[0])
        return {
            "schema": TARGET_LOWERING_SCHEMA,
            "candidate_id": candidate["candidate_id"],
            "representation_graph_id": representation_ir["graph_id"],
            "scope_ids": list(candidate["scope_ids"]),
            "capability_class": context.hardware_profile["capability_class"],
            "regions": [_lowered_component(item) for item in opportunities],
            "shader_artifacts": [
                {
                    "artifact_path": shader.artifact_path,
                    "template_name": shader.template_name,
                }
                for shader in representative.shader_artifacts
            ],
            "artifacts": {
                "proof_path": PROOF_PATH,
                "component_fixture_path": COMPONENT_FIXTURE_PATH,
                "conversation_fixture_path": CONVERSATION_FIXTURE_PATH,
                "product_conversation_fixture_path": (
                    PRODUCT_CONVERSATION_FIXTURE_PATH
                ),
                "model_limits_path": MODEL_LIMITS_PATH,
            },
            "runtime": {
                "max_context_activations": opportunities[0].max_context_activations,
                "required_vulkan_version": "1.4",
                "fallback": "exact_source_attention_region",
            },
        }

    def estimate_static_cost(
        self,
        context: ProviderContext,
        candidate: Json,
        representation_ir: Json,
        target_lowering: Json,
    ) -> StaticEstimate:
        del representation_ir, target_lowering
        opportunities = _candidate_opportunities(context, candidate)
        representative = opportunities[0]
        return StaticEstimate(
            feasible=True,
            permanent_bytes=0,
            transient_bytes=0,
            construction_nanoseconds=None,
            steady_state_work={
                "kind": "exact_grouped_attention_transaction",
                "component_count": len(opportunities),
                "head_group": representative.head_group,
                "source_workgroups_per_component": representative.query_heads,
                "candidate_workgroups_per_component": (
                    representative.query_heads // representative.head_group
                ),
                "latent_state_read_reuse": representative.head_group,
            },
            reasons=(
                "the candidate preserves parameters, state, and per-head arithmetic while sharing each latent-state read across query heads",
            ),
        )

    def construction_requirements(
        self,
        context: ProviderContext,
        candidate: Json,
    ) -> Json:
        opportunities = _candidate_opportunities(context, candidate)
        representative = prepare_grouped_attention(context, opportunities[0])
        outputs = [
            _output(
                shader.artifact_path,
                "vulkan_shader",
                "residency",
                "physical_optimization",
                "spirv_module",
                {"minimum_version": 0x00010600},
            )
            for shader in representative.shader_artifacts
        ]
        outputs.extend(
            _output(
                component_overlay_path(item.component_id),
                "runtime_overlay",
                "mount",
                "physical_optimization",
                "json_contract",
                {
                    "schema": "nerve.optimizer.vulkan_component_region_overlay.v1",
                    "object_required": True,
                },
            )
            for item in opportunities
        )
        for path, kind, schema in (
            (
                COMPONENT_FIXTURE_PATH,
                "validation_fixture",
                "nerve.optimizer.attention_component_fixture.v1",
            ),
            (
                CONVERSATION_FIXTURE_PATH,
                "validation_fixture",
                VALIDATION_CONVERSATION_SCHEMA,
            ),
            (
                MODEL_LIMITS_PATH,
                "validation_fixture",
                "nerve.optimizer.model_limits_fixture.v1",
            ),
            (
                PRODUCT_CONVERSATION_FIXTURE_PATH,
                "validation_fixture",
                VALIDATION_CONVERSATION_SCHEMA,
            ),
            (PROOF_PATH, "equivalence_proof", PROOF_SCHEMA),
        ):
            outputs.append(
                _output(
                    path,
                    kind,
                    "compile",
                    (
                        "physical_optimization"
                        if path == PROOF_PATH
                        else "semantic_construction"
                    ),
                    "json_contract",
                    {"schema": schema, "object_required": True},
                )
            )
        source_records = {
            record["path"]: record
            for item in opportunities
            for record in source_inputs(context, item)
        }
        return {
            "schema": "nerve.optimizer.candidate_build_plan.v1",
            "phases": [
                "semantic_construction",
                "ordinary_lowering",
                "physical_optimization",
            ],
            "source_inputs": [
                source_records[path] for path in sorted(source_records)
            ],
            "outputs": sorted(outputs, key=lambda item: item["path"]),
            "resource_limits": {
                "maximum_construction_time_ns": None,
                "maximum_temporary_bytes": None,
                "maximum_staging_bytes": None,
            },
        }

    def mount_requirements(
        self,
        context: ProviderContext,
        candidate: Json,
    ) -> Json:
        return {
            "schema": "nerve.optimizer.runtime_mount_plan.v3",
            "candidate_id": candidate["candidate_id"],
            "adapter_id": "vulkan_stream_circuit_overlay.v2",
            "regions": [
                {
                    "replacements": [
                        {
                            "kind": "component_region",
                            "source_component_id": item.component_id,
                            "overlay_ref": component_overlay_path(item.component_id),
                        }
                    ]
                }
                for item in _candidate_opportunities(context, candidate)
            ],
            "tensor_index_refs": [],
        }

    def proof_or_error_contract(
        self,
        context: ProviderContext,
        candidate: Json,
    ) -> Json:
        del context
        return deepcopy(candidate["behavioral_contract"])

    def benchmark_workloads(
        self,
        context: ProviderContext,
        candidate: Json,
    ) -> tuple[Json, ...]:
        return attention_benchmark_workloads(
            _candidate_opportunities(context, candidate)[0]
        )

    def validation_requirements(
        self,
        context: ProviderContext,
        candidate: Json,
    ) -> Json:
        return attention_validation_requirements(
            candidate=candidate,
            opportunity=_candidate_opportunities(context, candidate)[0],
            speculative_draft_tokens=(
                context.qualification_regime.speculative_draft_tokens
            ),
        )


def _opportunity_groups(
    context: ProviderContext,
) -> tuple[tuple[AttentionHeadGroupingOpportunity, ...], ...]:
    grouped: dict[str, list[AttentionHeadGroupingOpportunity]] = {}
    for opportunity in discover_attention_head_groupings(context):
        grouped.setdefault(opportunity.performance_signature, []).append(opportunity)
    return tuple(
        tuple(sorted(group, key=lambda item: item.component_id))
        for _signature, group in sorted(grouped.items())
    )


def _candidate_opportunities(
    context: ProviderContext,
    candidate: Json,
) -> tuple[AttentionHeadGroupingOpportunity, ...]:
    requested = set(candidate["scope_ids"])
    topology = candidate["representation"]["topology"]
    head_group = int(topology["head_group"])
    matches = tuple(
        item
        for item in discover_attention_head_groupings(context)
        if item.scope_id in requested and item.head_group == head_group
    )
    if not matches or {item.scope_id for item in matches} != requested:
        raise ModelCompileError(
            "grouped-attention candidate does not map to exact component regions"
        )
    matches = tuple(sorted(matches, key=lambda item: item.component_id))
    expected = sorted(
        (item.scope_id, item.source_contract_digest) for item in matches
    )
    if list(
        zip(
            candidate["scope_ids"],
            candidate["source_contract_digests"],
            strict=True,
        )
    ) != expected:
        raise ModelCompileError("grouped-attention candidate source contracts drifted")
    signatures = {item.performance_signature for item in matches}
    expected_topology = {
        "kind": "source_anchored_component_regions",
        "component_ids": [item.component_id for item in matches],
        "region_count_per_component": 1,
        "head_group": head_group,
        "performance_equivalence_class": matches[0].performance_signature,
    }
    if len(signatures) != 1 or topology != expected_topology:
        raise ModelCompileError(
            "grouped-attention candidate crosses physical performance classes"
        )
    return matches


def _target_predicate(
    context: ProviderContext,
    opportunity: AttentionHeadGroupingOpportunity,
) -> Json:
    device = opportunity.compiler_device
    return {
        "capability_class": context.hardware_profile["capability_class"],
        "device_kind": "gpu",
        "api": "vulkan",
        "required_subgroup_operations": ["basic", "arithmetic"],
        "required_subgroup_size": int(device["subgroup_size"]),
        "minimum_workgroup_invocations": 1024,
        "minimum_workgroup_size_x": 1024,
        "subgroup_compute_required": True,
        "compiler_capabilities": deepcopy(device),
        "execution_envelope": {
            "phases": ["decode", "prefill"],
            "activation_batch": {
                "minimum": 1,
                "maximum": opportunity.max_context_activations,
            },
            "context_activations": {
                "minimum": 0,
                "maximum": opportunity.max_context_activations,
            },
        },
    }


def _lowered_component(opportunity: AttentionHeadGroupingOpportunity) -> Json:
    return {
        "component_id": opportunity.component_id,
        "scope_id": opportunity.scope_id,
        "source_contract_digest": opportunity.source_contract_digest,
        "source_node_id": opportunity.source_node_id,
        "physical_node_id": opportunity.physical_node_id,
        "terminal_node_id": opportunity.terminal_node_id,
        "evidence_ids": list(opportunity.evidence_ids),
        "source_artifact_refs": list(opportunity.source_artifact_refs),
        "manifest_ref": opportunity.manifest_ref,
        "circuit_ref": opportunity.circuit_ref,
        "tensor_index_ref": opportunity.tensor_index_ref,
        "query_heads": opportunity.query_heads,
        "key_value_heads": opportunity.key_value_heads,
        "head_width": opportunity.head_width,
        "local_window": opportunity.local_window,
        "compression_ratio": opportunity.compression_ratio,
        "max_compressed_indices": opportunity.max_compressed_indices,
        "head_group": opportunity.head_group,
        "shader_suffix": opportunity.shader_suffix,
        "max_context_activations": opportunity.max_context_activations,
        "compiler_device": deepcopy(opportunity.compiler_device),
        "performance_signature": opportunity.performance_signature,
        "overlay_path": component_overlay_path(opportunity.component_id),
    }


def _output(
    path: str,
    kind: str,
    lifetime: str,
    phase: str,
    validator_id: str,
    validation_contract: Json,
) -> Json:
    return {
        "path": path,
        "kind": kind,
        "lifetime": lifetime,
        "producer_phase": phase,
        "resident_bytes": 0,
        "validator_id": validator_id,
        "validation_contract": validation_contract,
    }
