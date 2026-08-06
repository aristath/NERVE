from __future__ import annotations

from copy import deepcopy

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.contracts import (
    representation_candidate_id,
    stable_contract_id,
)
from nerve.representation_optimizer.providers.resident_expansion.artifacts import (
    COMPONENT_FIXTURE_PATH,
    CONVERSATION_FIXTURE_PATH,
    MODEL_LIMITS_PATH,
    PRODUCT_CONVERSATION_FIXTURE_PATH,
    PROOF_PATH,
    artifact_paths,
    component_overlay_path,
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

    def required_analyzer_ids(
        self,
        scope: Json,
        source_contract: Json,
    ) -> tuple[str, ...]:
        del scope, source_contract
        return ("semantic_graph_structure",)

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
        for opportunities in _opportunity_groups(context):
            representative = opportunities[0]
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
            component_ids = tuple(
                opportunity.component_id for opportunity in opportunities
            )
            shader_paths = tuple(
                sorted(
                    {
                        path
                        for opportunity in opportunities
                        for path in opportunity.shader_artifact_paths
                    }
                )
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
                        "kind": "independently_selectable_component_regions",
                        "component_ids": list(component_ids),
                        "node_ids": sorted(
                            {
                                node_id
                                for opportunity in opportunities
                                for node_id in opportunity.node_ids
                            }
                        ),
                        "performance_equivalence_class": _opportunity_performance_key(
                            representative
                        ),
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
                            "maximum": representative.max_context_activations,
                        },
                        "context_activations": {
                            "minimum": 0,
                            "maximum": representative.max_context_activations,
                        },
                        "state_activations": {
                            "minimum": 0,
                            "maximum": representative.max_context_activations,
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
                    for path in artifact_paths(shader_paths, component_ids)
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
        return resident_expansion_representation_graph(
            candidate=candidate,
            opportunities=opportunities,
            capability_class=str(context.hardware_profile["capability_class"]),
        )

    def lower_for_target(
        self,
        context: ProviderContext,
        candidate: Json,
        representation_ir: Json,
    ) -> Json:
        opportunities = _candidate_opportunities(context, candidate)
        return {
            "schema": TARGET_LOWERING_SCHEMA,
            "candidate_id": candidate["candidate_id"],
            "representation_graph_id": representation_ir["graph_id"],
            "scope_ids": list(candidate["scope_ids"]),
            "capability_class": context.hardware_profile["capability_class"],
            "regions": [
                _lowered_region(context, opportunity) for opportunity in opportunities
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
                "max_context_activations": (opportunities[0].max_context_activations),
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
        opportunities = _candidate_opportunities(context, candidate)
        representative = opportunities[0]
        total_source_bytes = sum(
            opportunity.source_weight_bytes for opportunity in opportunities
        )
        total_resident_bytes = sum(
            opportunity.resident_weight_bytes for opportunity in opportunities
        )
        return StaticEstimate(
            feasible=True,
            permanent_bytes=0,
            transient_bytes=total_resident_bytes - total_source_bytes,
            construction_nanoseconds=None,
            steady_state_work={
                "kind": "native_fp8_dot4_acc32",
                "source_parameter_bytes": representative.source_weight_bytes,
                "fully_resident_parameter_bytes": (
                    representative.resident_weight_bytes
                ),
                "maximum_selected_source_parameter_bytes": total_source_bytes,
                "maximum_selected_resident_parameter_bytes": total_resident_bytes,
                "parameter_byte_ratio": (
                    representative.resident_weight_bytes
                    / representative.source_weight_bytes
                ),
                "expert_count": representative.expert_count,
                "experts_per_activation": representative.experts_per_token,
                "independent_region_count": len(opportunities),
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
        return _build_plan(
            context,
            _candidate_opportunities(context, candidate),
        )

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
                            "kind": "component",
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
        return resident_expansion_benchmark_workloads(
            _candidate_opportunities(context, candidate)[0]
        )

    def validation_requirements(
        self,
        context: ProviderContext,
        candidate: Json,
    ) -> Json:
        return resident_expansion_validation_requirements(
            candidate=candidate,
            opportunity=_candidate_opportunities(context, candidate)[0],
            speculative_draft_tokens=(
                context.qualification_regime.speculative_draft_tokens
            ),
        )


def _opportunity_performance_key(
    opportunity: ResidentExpansionOpportunity,
) -> str:
    return stable_contract_id(
        "resident_performance_class",
        {
            "geometry": {
                "hidden_size": opportunity.hidden_size,
                "intermediate_size": opportunity.intermediate_size,
                "expert_count": opportunity.expert_count,
                "experts_per_token": opportunity.experts_per_token,
                "max_context_activations": opportunity.max_context_activations,
            },
            "derivations": [
                {
                    "node_id": item.node_id,
                    "parameter_id": item.parameter_id,
                    "source_byte_count": item.source_byte_count,
                    "resident_byte_count": item.derivation["resident_byte_count"],
                    "schema": item.derivation["schema"],
                    "kind": item.derivation["kind"],
                    "required_features": item.derivation["required_features"],
                }
                for item in opportunity.weight_derivations
            ],
            "shaders": [
                {
                    "node_id": item.node_id,
                    "template_name": item.template_name,
                    "execution_kind": item.execution_kind,
                }
                for item in opportunity.shader_replacements
            ],
        },
    )


def _opportunity_groups(
    context: ProviderContext,
) -> tuple[tuple[ResidentExpansionOpportunity, ...], ...]:
    grouped: dict[str, list[ResidentExpansionOpportunity]] = {}
    for opportunity in discover_resident_expansions(context):
        grouped.setdefault(_opportunity_performance_key(opportunity), []).append(
            opportunity
        )
    return tuple(
        tuple(sorted(group, key=lambda item: item.component_id))
        for _key, group in sorted(grouped.items())
    )


def _lowered_region(
    context: ProviderContext,
    opportunity: ResidentExpansionOpportunity,
) -> Json:
    return {
        "scope_ids": list(opportunity.scope_ids),
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
            "overlay_path": component_overlay_path(opportunity.component_id),
        },
    }


def _group_source_inputs(
    context: ProviderContext,
    opportunities: tuple[ResidentExpansionOpportunity, ...],
) -> list[Json]:
    inputs = {
        item["path"]: item
        for opportunity in opportunities
        for item in source_inputs(context, opportunity)
    }
    return [inputs[path] for path in sorted(inputs)]


def _candidate_opportunities(
    context: ProviderContext,
    candidate: Json,
) -> tuple[ResidentExpansionOpportunity, ...]:
    requested = set(candidate["scope_ids"])
    matches = tuple(
        opportunity
        for opportunity in discover_resident_expansions(context)
        if set(opportunity.scope_ids).issubset(requested)
    )
    covered = {
        scope_id for opportunity in matches for scope_id in opportunity.scope_ids
    }
    if not matches or covered != requested:
        raise ModelCompileError(
            "resident parameter expansion candidate does not map to one exact "
            "set of component regions"
        )
    matches = tuple(sorted(matches, key=lambda item: item.component_id))
    expected_scope_contracts = sorted(
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
    candidate_scope_contracts = list(
        zip(
            candidate["scope_ids"],
            candidate["source_contract_digests"],
            strict=True,
        )
    )
    if candidate_scope_contracts != expected_scope_contracts:
        raise ModelCompileError(
            "resident parameter expansion candidate source contracts do not "
            "match discovered component regions"
        )
    performance_keys = {
        _opportunity_performance_key(opportunity) for opportunity in matches
    }
    if len(performance_keys) != 1:
        raise ModelCompileError(
            "resident parameter expansion candidate crosses physical performance classes"
        )
    topology = candidate["representation"]["topology"]
    expected_topology = {
        "kind": "independently_selectable_component_regions",
        "component_ids": [opportunity.component_id for opportunity in matches],
        "node_ids": sorted(
            {node_id for opportunity in matches for node_id in opportunity.node_ids}
        ),
        "performance_equivalence_class": next(iter(performance_keys)),
    }
    if topology != expected_topology:
        raise ModelCompileError(
            "resident parameter expansion candidate topology does not match "
            "discovered component regions and performance class"
        )
    return matches


def _build_plan(
    context: ProviderContext,
    opportunities: tuple[ResidentExpansionOpportunity, ...],
) -> Json:
    outputs = []
    for path in sorted(
        {
            path
            for opportunity in opportunities
            for path in opportunity.shader_artifact_paths
        }
    ):
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
    for opportunity in opportunities:
        outputs.append(
            _output(
                component_overlay_path(opportunity.component_id),
                "runtime_overlay",
                "mount",
                "ordinary_lowering",
                {
                    "validator_id": "json_contract",
                    "validation_contract": {
                        "schema": "nerve.optimizer.vulkan_component_overlay.v2",
                        "object_required": True,
                    },
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
        "source_inputs": _group_source_inputs(context, opportunities),
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
