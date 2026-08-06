from __future__ import annotations

from copy import deepcopy

from nerve.compilation import Json
from nerve.representation_optimizer.contracts import representation_candidate_id
from nerve.representation_optimizer.providers.resident_expansion.artifacts import (
    COMPONENT_FIXTURE_PATH,
    CONVERSATION_FIXTURE_PATH,
    MODEL_LIMITS_PATH,
    OVERLAY_PATH,
    PRODUCT_CONVERSATION_FIXTURE_PATH,
    PROOF_PATH,
    artifact_paths,
)
from nerve.representation_optimizer.providers.resident_expansion.contracts import (
    COMPONENT_FIXTURE_SCHEMA,
    EXACT_EXPANSION_OBLIGATIONS,
    PROOF_SCHEMA,
    RECONSTRUCTED_PARAMETER_STREAM_DESCRIPTOR_ID,
    TARGET_LOWERING_SCHEMA,
)
from nerve.representation_optimizer.providers.resident_expansion.discovery import (
    ResidentExpansionOpportunity,
    discover_resident_expansions,
    discovery_result,
    is_resident_expansion_scope,
    require_resident_expansion,
    source_inputs,
)
from nerve.representation_optimizer.providers.resident_expansion.representation import (
    resident_expansion_representation_graph,
)
from nerve.representation_optimizer.providers.resident_expansion.workloads import (
    resident_expansion_benchmark_workloads,
    resident_expansion_validation_requirements,
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


class ExactResidentExpertExpansionProvider:
    identity = ProviderIdentity(
        "nerve.exact_resident_expert_parameter_expansion",
        "1",
    )
    descriptor_id = RECONSTRUCTED_PARAMETER_STREAM_DESCRIPTOR_ID

    def may_optimize_scope(self, scope: Json, source_contract: Json) -> bool:
        return is_resident_expansion_scope(scope, source_contract)

    def match_semantics(self, context: ProviderContext) -> MatchAssessment:
        matched = any(
            is_resident_expansion_scope(scope, contract)
            for scope, contract in zip(
                context.scopes,
                context.source_contracts,
                strict=True,
            )
        )
        return MatchAssessment(
            matched=matched,
            reasons=(
                (
                    "independently addressable sparse expert projections can "
                    "retain compact source storage and derive a resident form"
                )
                if matched
                else "no independently addressable sparse expert projection scope",
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
                "component_count": len(result.opportunities),
                "source_format": "packed_mxfp4_e2m1_with_f8_e8m0_scales",
                "resident_format": "fp8_e4m3_with_source_f8_e8m0_scales",
                "construction": "exact_on_demand_code_expansion",
                "selection_boundary": "component_local",
            },
            reasons=(
                "source metadata, selector-addressed resources, exact code "
                "mapping, and target-native FP8 execution all agree",
            ),
        )

    def synthesize_candidates(
        self,
        context: ProviderContext,
        evidence: EvidenceAssessment,
    ) -> tuple[Json, ...]:
        accepted = set(evidence.evidence_ids)
        candidates = []
        for opportunity in discover_resident_expansions(context):
            candidate = {
                "schema": "nerve.optimizer.representation_candidate.v1",
                "candidate_id": "",
                "scope_ids": list(opportunity.scope_ids),
                "source_contract_digests": list(opportunity.source_contract_digests),
                "provider": self.identity.to_json(),
                "descriptor_id": self.descriptor_id,
                "evidence_refs": [
                    evidence_id
                    for evidence_id in opportunity.evidence_ids
                    if evidence_id in accepted
                ],
                "representation": {
                    "kind": "exact_on_demand_mxfp4_to_fp8_e4m3",
                    "signal_formats": [{"name": "source_bf16_signals"}],
                    "parameter_format": {
                        "kind": "compact_source_with_derived_resident_form",
                        "source": "packed_mxfp4_e2m1",
                        "resident": "fp8_e4m3",
                        "scale": "f8_e8m0_power_of_two_per_group32",
                        "derivation_lifetime": "demand_retained",
                    },
                    "state_format": {"kind": "source_state_unchanged"},
                    "topology": {
                        "kind": "component_local_sparse_expert_bank",
                        "component_ids": [opportunity.component_id],
                        "node_ids": list(opportunity.node_ids),
                    },
                },
                "target_predicate": {
                    "capability_class": context.hardware_profile["capability_class"],
                    "device_kind": "gpu",
                    "api": "vulkan",
                    "required_processes": [
                        "packed_dot_product",
                        "shader_vector",
                    ],
                    "execution_envelope": {
                        "phases": ["decode", "prefill"],
                        "alternative_phases": ["decode", "prefill"],
                        "source_retained_phases": [],
                        "activation_batch": {
                            "minimum": 1,
                            "maximum": opportunity.max_context_activations,
                        },
                        "context_activations": {
                            "minimum": 0,
                            "maximum": opportunity.max_context_activations,
                        },
                        "state_activations": {
                            "minimum": 0,
                            "maximum": opportunity.max_context_activations,
                        },
                    },
                },
                "behavioral_contract": {
                    "mode": "exact",
                    "proof_obligations": list(EXACT_EXPANSION_OBLIGATIONS),
                    "error_contract": None,
                },
                "artifact_declarations": [
                    {"path": path}
                    for path in artifact_paths(opportunity.shader_artifact_paths)
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
        opportunity = _candidate_opportunity(context, candidate)
        return resident_expansion_representation_graph(
            candidate=candidate,
            opportunity=opportunity,
            capability_class=str(context.hardware_profile["capability_class"]),
        )

    def lower_for_target(
        self,
        context: ProviderContext,
        candidate: Json,
        representation_ir: Json,
    ) -> Json:
        opportunity = _candidate_opportunity(context, candidate)
        return {
            "schema": TARGET_LOWERING_SCHEMA,
            "candidate_id": candidate["candidate_id"],
            "representation_graph_id": representation_ir["graph_id"],
            "scope_ids": list(opportunity.scope_ids),
            "capability_class": context.hardware_profile["capability_class"],
            "source": {
                "component_id": opportunity.component_id,
                "node_ids": list(opportunity.node_ids),
                "manifest_ref": opportunity.manifest_ref,
                "source_inputs": source_inputs(context, opportunity),
            },
            "geometry": {
                "hidden_size": opportunity.hidden_size,
                "intermediate_size": opportunity.intermediate_size,
                "expert_count": opportunity.expert_count,
                "experts_per_token": opportunity.experts_per_token,
            },
            "resident_derivations": [
                item.to_json() for item in opportunity.weight_derivations
            ],
            "shader_replacements": [
                item.to_json() for item in opportunity.shader_replacements
            ],
            "artifacts": {
                "overlay_path": OVERLAY_PATH,
                "proof_path": PROOF_PATH,
                "component_fixture_path": COMPONENT_FIXTURE_PATH,
                "conversation_fixture_path": CONVERSATION_FIXTURE_PATH,
                "product_conversation_fixture_path": (
                    PRODUCT_CONVERSATION_FIXTURE_PATH
                ),
                "model_limits_path": MODEL_LIMITS_PATH,
            },
            "runtime": {
                "max_context_activations": (opportunity.max_context_activations),
                "required_vulkan_version": "1.4",
                "residency_lifetime": "demand_retained",
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
        opportunity = _candidate_opportunity(context, candidate)
        return StaticEstimate(
            feasible=True,
            permanent_bytes=0,
            transient_bytes=(
                opportunity.resident_weight_bytes - opportunity.source_weight_bytes
            ),
            construction_nanoseconds=None,
            steady_state_work={
                "kind": "native_fp8_dot4_acc32",
                "source_parameter_bytes": opportunity.source_weight_bytes,
                "fully_resident_parameter_bytes": (opportunity.resident_weight_bytes),
                "parameter_byte_ratio": (
                    opportunity.resident_weight_bytes / opportunity.source_weight_bytes
                ),
                "expert_count": opportunity.expert_count,
                "experts_per_activation": opportunity.experts_per_token,
                "materialization": "only_selected_resources_on_demand",
            },
            reasons=(
                "the source package stays compact; selected expert resources "
                "expand exactly on first device demand and remain resident "
                "subject to the shared cache policy",
            ),
        )

    def construction_requirements(
        self,
        context: ProviderContext,
        candidate: Json,
    ) -> Json:
        opportunity = _candidate_opportunity(context, candidate)
        return _build_plan(context, opportunity)

    def mount_requirements(
        self,
        context: ProviderContext,
        candidate: Json,
    ) -> Json:
        opportunity = _candidate_opportunity(context, candidate)
        return {
            "schema": "nerve.optimizer.runtime_mount_plan.v3",
            "candidate_id": candidate["candidate_id"],
            "adapter_id": "vulkan_stream_circuit_overlay.v2",
            "regions": [
                {
                    "replacements": [
                        {
                            "kind": "component",
                            "source_component_id": opportunity.component_id,
                            "overlay_ref": OVERLAY_PATH,
                        }
                    ]
                }
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
        return resident_expansion_benchmark_workloads(
            _candidate_opportunity(context, candidate)
        )

    def validation_requirements(
        self,
        context: ProviderContext,
        candidate: Json,
    ) -> Json:
        return resident_expansion_validation_requirements(
            candidate=candidate,
            opportunity=_candidate_opportunity(context, candidate),
            speculative_draft_tokens=(
                context.qualification_regime.speculative_draft_tokens
            ),
        )


def _candidate_opportunity(
    context: ProviderContext,
    candidate: Json,
) -> ResidentExpansionOpportunity:
    return require_resident_expansion(
        context,
        tuple(candidate["scope_ids"]),
    )


def _build_plan(
    context: ProviderContext,
    opportunity: ResidentExpansionOpportunity,
) -> Json:
    outputs = []
    for path in opportunity.shader_artifact_paths:
        outputs.append(
            _output(
                path,
                "vulkan_shader",
                "residency",
                "physical_optimization",
                {
                    "validator_id": "spirv_module",
                    "validation_contract": {"minimum_version": 0x00010600},
                },
            )
        )
    for path, kind, schema, phase in (
        (
            COMPONENT_FIXTURE_PATH,
            "validation_fixture",
            COMPONENT_FIXTURE_SCHEMA,
            "semantic_construction",
        ),
        (
            CONVERSATION_FIXTURE_PATH,
            "validation_fixture",
            VALIDATION_CONVERSATION_SCHEMA,
            "semantic_construction",
        ),
        (
            MODEL_LIMITS_PATH,
            "validation_fixture",
            "nerve.optimizer.model_limits_fixture.v1",
            "semantic_construction",
        ),
        (
            OVERLAY_PATH,
            "runtime_overlay",
            "nerve.optimizer.vulkan_component_overlay.v2",
            "ordinary_lowering",
        ),
        (
            PRODUCT_CONVERSATION_FIXTURE_PATH,
            "validation_fixture",
            VALIDATION_CONVERSATION_SCHEMA,
            "semantic_construction",
        ),
        (
            PROOF_PATH,
            "equivalence_proof",
            PROOF_SCHEMA,
            "semantic_construction",
        ),
    ):
        outputs.append(
            _output(
                path,
                kind,
                "mount" if kind == "runtime_overlay" else "compile",
                phase,
                {
                    "validator_id": "json_contract",
                    "validation_contract": {
                        "schema": schema,
                        "object_required": True,
                    },
                },
            )
        )
    outputs.sort(key=lambda item: item["path"])
    return {
        "schema": "nerve.optimizer.candidate_build_plan.v1",
        "phases": [
            "semantic_construction",
            "ordinary_lowering",
            "physical_optimization",
        ],
        "source_inputs": source_inputs(context, opportunity),
        "outputs": outputs,
        "resource_limits": {
            "maximum_construction_time_ns": None,
            "maximum_temporary_bytes": None,
            "maximum_staging_bytes": None,
        },
    }


def _output(
    path: str,
    kind: str,
    lifetime: str,
    phase: str,
    validation: Json,
) -> Json:
    return {
        "path": path,
        "kind": kind,
        "lifetime": lifetime,
        "producer_phase": phase,
        "resident_bytes": 0,
        **validation,
    }
