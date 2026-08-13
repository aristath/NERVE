from __future__ import annotations

from copy import deepcopy

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.contracts import representation_candidate_id
from nerve.representation_optimizer.providers.hyper_norm_fusion.artifacts import (
    COMPONENT_FIXTURE_PATH,
    CONVERSATION_FIXTURE_PATH,
    MODEL_LIMITS_PATH,
    PRODUCT_CONVERSATION_FIXTURE_PATH,
    PROOF_PATH,
    artifact_paths,
    component_overlay_path,
)
from nerve.representation_optimizer.providers.hyper_norm_fusion.contracts import (
    COMPONENT_FIXTURE_SCHEMA,
    EXACT_FUSION_OBLIGATIONS,
    HETEROGENEOUS_COMPOSITE_ISLAND_DESCRIPTOR_ID,
    PROOF_SCHEMA,
    TARGET_LOWERING_SCHEMA,
)
from nerve.representation_optimizer.providers.hyper_norm_fusion.discovery import (
    HyperNormFusionOpportunity,
    discover_hyper_norm_fusions,
    discovery_result,
    is_hyper_norm_scope,
    source_inputs,
)
from nerve.representation_optimizer.providers.hyper_norm_fusion.physical import (
    prepare_fused_component,
)
from nerve.representation_optimizer.providers.hyper_norm_fusion.representation import (
    hyper_norm_representation_graph,
)
from nerve.representation_optimizer.providers.hyper_norm_fusion.workloads import (
    hyper_norm_benchmark_workloads,
    hyper_norm_validation_requirements,
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


class ExactHyperNormFusionProvider:
    identity = ProviderIdentity("nerve.exact_hyper_norm_fusion", "1")
    descriptor_id = HETEROGENEOUS_COMPOSITE_ISLAND_DESCRIPTOR_ID

    def may_optimize_scope(self, scope: Json, source_contract: Json) -> bool:
        return is_hyper_norm_scope(scope, source_contract)

    def required_analyzer_ids(
        self,
        scope: Json,
        source_contract: Json,
    ) -> tuple[str, ...]:
        del scope, source_contract
        return ("semantic_graph_structure",)

    def match_semantics(self, context: ProviderContext) -> MatchAssessment:
        matched = any(
            is_hyper_norm_scope(scope, contract)
            for scope, contract in zip(
                context.scopes,
                context.source_contracts,
                strict=True,
            )
        )
        return MatchAssessment(
            matched=matched,
            reasons=(
                "adjacent semantic representation islands may absorb exact local transducers"
                if matched
                else "no adjacent semantic representation island is present",
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
                    {
                        opportunity.component_id
                        for opportunity in result.opportunities
                    }
                ),
                "alternative_family_count": len(
                    {
                        opportunity.performance_signature
                        for opportunity in result.opportunities
                    }
                ),
                "region_count": sum(
                    len(opportunity.regions)
                    for opportunity in result.opportunities
                ),
                "source_execution": "hyper_reduce_then_rms_then_fp8_prequantize",
                "candidate_execution": "one_exact_fused_transaction",
                "selection_boundary": "component_region",
            },
            reasons=(
                "source algebra, compiled physical representation, and exact target capabilities agree",
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
            prepared = prepare_fused_component(context, representative)
            scope_contracts = sorted(
                (
                    (scope_id, digest)
                    for opportunity in opportunities
                    for scope_id, digest in zip(
                        opportunity.scope_ids,
                        opportunity.source_contract_digests,
                        strict=True,
                    )
                ),
                key=lambda item: item[0],
            )
            overlay_paths = tuple(
                component_overlay_path(opportunity.component_id)
                for opportunity in opportunities
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
                            item
                            for opportunity in opportunities
                            for item in opportunity.evidence_ids
                        }
                    )
                    if evidence_id in accepted
                ],
                "representation": {
                    "kind": "exact_capability_scoped_hyper_norm_fusion",
                    "signal_formats": [{"name": "source_bf16_and_fp8_signals"}],
                    "parameter_format": {"kind": "source_parameters_unchanged"},
                    "state_format": {"kind": "source_state_unchanged"},
                    "topology": {
                        "kind": "source_anchored_component_regions",
                        "component_ids": [
                            opportunity.component_id for opportunity in opportunities
                        ],
                        "region_count_per_component": len(representative.regions),
                        "performance_equivalence_class": (
                            representative.performance_signature
                        ),
                    },
                },
                "target_predicate": _target_predicate(context, representative),
                "behavioral_contract": {
                    "mode": "exact",
                    "proof_obligations": list(EXACT_FUSION_OBLIGATIONS),
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
        representative = prepare_fused_component(context, opportunities[0])
        return hyper_norm_representation_graph(
            candidate=candidate,
            opportunities=opportunities,
            prepared=tuple(representative for _ in opportunities),
            capability_class=str(context.hardware_profile["capability_class"]),
        )

    def lower_for_target(
        self,
        context: ProviderContext,
        candidate: Json,
        representation_ir: Json,
    ) -> Json:
        opportunities = _candidate_opportunities(context, candidate)
        representative = prepare_fused_component(context, opportunities[0])
        return {
            "schema": TARGET_LOWERING_SCHEMA,
            "candidate_id": candidate["candidate_id"],
            "representation_graph_id": representation_ir["graph_id"],
            "scope_ids": list(candidate["scope_ids"]),
            "capability_class": context.hardware_profile["capability_class"],
            "regions": [
                _lowered_component(opportunity) for opportunity in opportunities
            ],
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
                "max_context_activations": representative_opportunity(
                    opportunities
                ).max_context_activations,
                "required_vulkan_version": "1.4",
                "fallback": "exact_unfused_component_region",
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
        region_count = sum(len(item.regions) for item in opportunities)
        return StaticEstimate(
            feasible=True,
            permanent_bytes=0,
            transient_bytes=0,
            construction_nanoseconds=None,
            steady_state_work={
                "kind": "exact_fused_hyper_norm_transaction",
                "component_count": len(opportunities),
                "region_count": region_count,
                "source_dispatch_count": region_count * 3,
                "candidate_dispatch_count": region_count,
                "removed_intermediate_publications": region_count * 2,
            },
            reasons=(
                "the candidate keeps source parameters and state unchanged while reducing three local dispatches to one",
            ),
        )

    def construction_requirements(
        self,
        context: ProviderContext,
        candidate: Json,
    ) -> Json:
        opportunities = _candidate_opportunities(context, candidate)
        representative = prepare_fused_component(context, opportunities[0])
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
                component_overlay_path(opportunity.component_id),
                "runtime_overlay",
                "mount",
                "physical_optimization",
                "json_contract",
                {
                    "schema": "nerve.optimizer.vulkan_component_region_overlay.v1",
                    "object_required": True,
                },
            )
            for opportunity in opportunities
        )
        for path, kind, schema in (
            (COMPONENT_FIXTURE_PATH, "validation_fixture", COMPONENT_FIXTURE_SCHEMA),
            (CONVERSATION_FIXTURE_PATH, "validation_fixture", VALIDATION_CONVERSATION_SCHEMA),
            (MODEL_LIMITS_PATH, "validation_fixture", "nerve.optimizer.model_limits_fixture.v1"),
            (PRODUCT_CONVERSATION_FIXTURE_PATH, "validation_fixture", VALIDATION_CONVERSATION_SCHEMA),
            (PROOF_PATH, "equivalence_proof", PROOF_SCHEMA),
        ):
            producer_phase = (
                "physical_optimization"
                if path == PROOF_PATH
                else "semantic_construction"
            )
            outputs.append(
                _output(
                    path,
                    kind,
                    "compile",
                    producer_phase,
                    "json_contract",
                    {"schema": schema, "object_required": True},
                )
            )
        source_records = {
            record["path"]: record
            for opportunity in opportunities
            for record in source_inputs(context, opportunity)
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
        opportunities = _candidate_opportunities(context, candidate)
        return {
            "schema": "nerve.optimizer.runtime_mount_plan.v3",
            "candidate_id": candidate["candidate_id"],
            "adapter_id": "vulkan_stream_circuit_overlay.v2",
            "regions": [
                {
                    "replacements": [
                        {
                            "kind": "component_region",
                            "source_component_id": opportunity.component_id,
                            "overlay_ref": component_overlay_path(
                                opportunity.component_id
                            ),
                        }
                    ]
                }
                for opportunity in opportunities
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
        return hyper_norm_benchmark_workloads(
            _candidate_opportunities(context, candidate)[0]
        )

    def validation_requirements(
        self,
        context: ProviderContext,
        candidate: Json,
    ) -> Json:
        return hyper_norm_validation_requirements(
            candidate=candidate,
            opportunity=_candidate_opportunities(context, candidate)[0],
            speculative_draft_tokens=(
                context.qualification_regime.speculative_draft_tokens
            ),
        )


def _opportunity_groups(
    context: ProviderContext,
) -> tuple[tuple[HyperNormFusionOpportunity, ...], ...]:
    grouped: dict[str, list[HyperNormFusionOpportunity]] = {}
    for opportunity in discover_hyper_norm_fusions(context):
        grouped.setdefault(opportunity.performance_signature, []).append(opportunity)
    return tuple(
        tuple(sorted(group, key=lambda item: item.component_id))
        for _signature, group in sorted(grouped.items())
    )


def _candidate_opportunities(
    context: ProviderContext,
    candidate: Json,
) -> tuple[HyperNormFusionOpportunity, ...]:
    requested = set(candidate["scope_ids"])
    matches = tuple(
        opportunity
        for opportunity in discover_hyper_norm_fusions(context)
        if set(opportunity.scope_ids).issubset(requested)
    )
    covered = {
        scope_id for opportunity in matches for scope_id in opportunity.scope_ids
    }
    if not matches or covered != requested:
        raise ModelCompileError(
            "hyper/RMS candidate does not map to exact component regions"
        )
    matches = tuple(sorted(matches, key=lambda item: item.component_id))
    expected = sorted(
        (
            (scope_id, digest)
            for opportunity in matches
            for scope_id, digest in zip(
                opportunity.scope_ids,
                opportunity.source_contract_digests,
                strict=True,
            )
        ),
        key=lambda item: item[0],
    )
    if list(
        zip(
            candidate["scope_ids"],
            candidate["source_contract_digests"],
            strict=True,
        )
    ) != expected:
        raise ModelCompileError("hyper/RMS candidate source contracts drifted")
    signatures = {opportunity.performance_signature for opportunity in matches}
    expected_topology = {
        "kind": "source_anchored_component_regions",
        "component_ids": [opportunity.component_id for opportunity in matches],
        "region_count_per_component": len(matches[0].regions),
        "performance_equivalence_class": matches[0].performance_signature,
    }
    if (
        len(signatures) != 1
        or candidate["representation"]["topology"] != expected_topology
    ):
        raise ModelCompileError(
            "hyper/RMS candidate crosses physical performance classes"
        )
    return matches


def _target_predicate(
    context: ProviderContext,
    opportunity: HyperNormFusionOpportunity,
) -> Json:
    device = opportunity.compiler_device
    return {
        "capability_class": context.hardware_profile["capability_class"],
        "device_kind": "gpu",
        "api": "vulkan",
        "required_features": ["shader_float8", "shader_int8"],
        "required_subgroup_operations": ["arithmetic"],
        "required_subgroup_size": 64,
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


def _lowered_component(opportunity: HyperNormFusionOpportunity) -> Json:
    return {
        "component_id": opportunity.component_id,
        "scope_ids": list(opportunity.scope_ids),
        "source_contract_digests": list(opportunity.source_contract_digests),
        "evidence_ids": list(opportunity.evidence_ids),
        "source_artifact_refs": list(opportunity.source_artifact_refs),
        "manifest_ref": opportunity.manifest_ref,
        "circuit_ref": opportunity.circuit_ref,
        "tensor_index_ref": opportunity.tensor_index_ref,
        "terminal_node_id": opportunity.terminal_node_id,
        "hidden_size": opportunity.hidden_size,
        "max_context_activations": opportunity.max_context_activations,
        "compiler_device": deepcopy(opportunity.compiler_device),
        "performance_signature": opportunity.performance_signature,
        "regions": [
            {
                "scope_id": region.scope_id,
                "source_contract_digest": region.source_contract_digest,
                "semantic_source_node_ids": list(region.semantic_source_node_ids),
                "hyper_node_id": region.hyper_node_id,
                "norm_node_id": region.norm_node_id,
                "quantizer_node_id": region.quantizer_node_id,
            }
            for region in opportunity.regions
        ],
        "overlay_path": component_overlay_path(opportunity.component_id),
    }


def representative_opportunity(
    opportunities: tuple[HyperNormFusionOpportunity, ...],
) -> HyperNormFusionOpportunity:
    if not opportunities:
        raise ModelCompileError("hyper/RMS lowering has no opportunities")
    return opportunities[0]


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
