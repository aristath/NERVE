from __future__ import annotations

import json
from copy import deepcopy

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.contracts import representation_candidate_id
from nerve.representation_optimizer.providers.group_scaled_int4.artifacts import (
    COMPONENT_FIXTURE_PATH,
    CONVERSATION_FIXTURE_PATH,
    MODEL_LIMITS_PATH,
    PRODUCT_CONVERSATION_FIXTURE_PATH,
    REPORT_PATH,
    TENSOR_FRAGMENT_PATH,
    component_overlay_path,
    kernel_artifact_path,
    scale_artifact_path,
    weight_artifact_path,
)
from nerve.representation_optimizer.providers.group_scaled_int4.contracts import (
    COMPONENT_FIXTURE_SCHEMA,
    GROUP_SCALED_INTEGER_DESCRIPTOR_ID,
    QUANTIZATION_REPORT_SCHEMA,
    TARGET_LOWERING_SCHEMA,
)
from nerve.representation_optimizer.providers.group_scaled_int4.discovery import (
    GroupScaledInt4Opportunity,
    discover_group_scaled_int4_linears,
    discovery_result,
    is_group_scaled_int4_scope,
    source_inputs,
)
from nerve.representation_optimizer.providers.group_scaled_int4.physical import (
    PreparedInt4Region,
    prepare_group_scaled_int4_component_from_documents,
)
from nerve.representation_optimizer.providers.group_scaled_int4.representation import (
    group_scaled_int4_representation_graph,
)
from nerve.representation_optimizer.providers.group_scaled_int4.workloads import (
    group_scaled_int4_benchmark_workloads,
    group_scaled_int4_error_contract,
    group_scaled_int4_validation_requirements,
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


class GroupScaledInt4LinearProvider:
    identity = ProviderIdentity("nerve.group_scaled_int4_linear", "1")
    descriptor_id = GROUP_SCALED_INTEGER_DESCRIPTOR_ID

    def may_optimize_scope(self, scope: Json, source_contract: Json) -> bool:
        return is_group_scaled_int4_scope(scope, source_contract)

    def required_analyzer_ids(
        self,
        scope: Json,
        source_contract: Json,
    ) -> tuple[str, ...]:
        del scope, source_contract
        return ("semantic_graph_structure",)

    def match_semantics(self, context: ProviderContext) -> MatchAssessment:
        matched = any(
            is_group_scaled_int4_scope(scope, contract)
            for scope, contract in zip(
                context.scopes,
                context.source_contracts,
                strict=True,
            )
        )
        return MatchAssessment(
            matched=matched,
            reasons=(
                "plain linear operators may replace a private dense parameter "
                "at their component-region boundary"
                if matched
                else "no plain linear operator scope is present",
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
                "region_count": len(result.opportunities),
                "source_format": "bf16_row_major",
                "candidate_format": "signed_int4_group32_with_bf16_scales",
                "selection_boundary": "parameterized_component_region",
                "source_binding_ownership": "private_to_one_linear_operator",
            },
            reasons=(
                "source tensor geometry, private parameter ownership, target "
                "subgroup execution, and transactional region mounting agree",
            ),
        )

    def synthesize_candidates(
        self,
        context: ProviderContext,
        evidence: EvidenceAssessment,
    ) -> tuple[Json, ...]:
        accepted = set(evidence.evidence_ids)
        candidates = []
        for opportunities in _candidate_opportunity_sets(context):
            representative = opportunities[0]
            prepared = _prepare(context, representative)
            scope_contracts = sorted(
                (
                    (item.scope_id, item.source_contract_digest)
                    for item in opportunities
                ),
                key=lambda item: item[0],
            )
            artifact_paths = {
                COMPONENT_FIXTURE_PATH,
                CONVERSATION_FIXTURE_PATH,
                MODEL_LIMITS_PATH,
                PRODUCT_CONVERSATION_FIXTURE_PATH,
                REPORT_PATH,
                TENSOR_FRAGMENT_PATH,
                *(
                    shader.artifact_path for shader in prepared.shader_artifacts
                ),
            }
            for opportunity in opportunities:
                artifact_paths.update(
                    {
                        component_overlay_path(
                            opportunity.component_id,
                            opportunity.node_id,
                        ),
                        scale_artifact_path(
                            opportunity.component_id,
                            opportunity.node_id,
                        ),
                        weight_artifact_path(
                            opportunity.component_id,
                            opportunity.node_id,
                        ),
                    }
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
                            for opportunity in opportunities
                            for evidence_id in opportunity.evidence_ids
                        }
                    )
                    if evidence_id in accepted
                ],
                "representation": {
                    "kind": "group_scaled_signed_int4_linear_parameter",
                    "signal_formats": [{"name": "source_bf16_signals"}],
                    "parameter_format": {
                        "kind": "compressed_tensors_signed_int4",
                        "storage_dtype": "I32",
                        "bits": 4,
                        "group_size": representative.group_size,
                        "scale_dtype": "BF16",
                        "symmetric": True,
                        "signed_offset": 8,
                    },
                    "state_format": {"kind": "source_state_unchanged"},
                    "topology": {
                        "kind": "source_anchored_parameterized_component_regions",
                        "component_region_ids": [
                            {
                                "component_id": item.component_id,
                                "physical_node_id": item.node_id,
                                "source_parameter_ref_id": item.source_weight_ref_id,
                            }
                            for item in opportunities
                        ],
                        "region_count": len(opportunities),
                        "performance_equivalence_class": (
                            representative.performance_signature
                        ),
                    },
                },
                "target_predicate": _target_predicate(
                    context,
                    representative,
                ),
                "behavioral_contract": {
                    "mode": "approximate",
                    "proof_obligations": [],
                    "error_contract": group_scaled_int4_error_contract(
                        representative.group_size
                    ),
                },
                "artifact_declarations": [
                    {"path": path} for path in sorted(artifact_paths)
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
        prepared = _prepare(context, opportunities[0])
        return group_scaled_int4_representation_graph(
            candidate=candidate,
            opportunities=opportunities,
            shader_artifacts=prepared.shader_artifacts,
            capability_class=str(context.hardware_profile["capability_class"]),
        )

    def lower_for_target(
        self,
        context: ProviderContext,
        candidate: Json,
        representation_ir: Json,
    ) -> Json:
        opportunities = _candidate_opportunities(context, candidate)
        prepared = _prepare(context, opportunities[0])
        return {
            "schema": TARGET_LOWERING_SCHEMA,
            "candidate_id": candidate["candidate_id"],
            "representation_graph_id": representation_ir["graph_id"],
            "scope_ids": list(candidate["scope_ids"]),
            "capability_class": context.hardware_profile["capability_class"],
            "regions": [_lowered_region(item) for item in opportunities],
            "shader_artifacts": [
                {
                    "artifact_path": shader.artifact_path,
                    "template_name": shader.template_name,
                }
                for shader in prepared.shader_artifacts
            ],
            "artifacts": {
                "report_path": REPORT_PATH,
                "tensor_fragment_path": TENSOR_FRAGMENT_PATH,
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
                "fallback": "native_source_component_region",
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
        source_bytes = sum(
            item.source_weight.payload_byte_count for item in opportunities
        )
        candidate_bytes = sum(
            item.output_features * item.input_features // 2
            + item.output_features
            * (item.input_features // item.group_size)
            * 2
            for item in opportunities
        )
        return StaticEstimate(
            feasible=True,
            permanent_bytes=candidate_bytes,
            transient_bytes=max(
                item.input_features * 256 * 8 for item in opportunities
            ),
            construction_nanoseconds=None,
            steady_state_work={
                "kind": "group_scaled_int4_linear_region",
                "region_count": len(opportunities),
                "source_parameter_bytes": source_bytes,
                "candidate_parameter_bytes": candidate_bytes,
                "parameter_byte_ratio": candidate_bytes / source_bytes,
                "group_size": opportunities[0].group_size,
                "selection_basis": "measured_complete_region",
            },
            reasons=(
                "candidate reduces parameter traffic but remains selectable "
                "only after complete-region performance and behavior validation",
            ),
        )

    def construction_requirements(
        self,
        context: ProviderContext,
        candidate: Json,
    ) -> Json:
        opportunities = _candidate_opportunities(context, candidate)
        prepared = _prepare(context, opportunities[0])
        outputs = [
            _output(
                shader.artifact_path,
                "vulkan_shader",
                "residency",
                "physical_optimization",
                0,
                "spirv_module",
                {"minimum_version": 0x00010600},
            )
            for shader in prepared.shader_artifacts
        ]
        for opportunity in opportunities:
            outputs.extend(
                (
                    _output(
                        component_overlay_path(
                            opportunity.component_id,
                            opportunity.node_id,
                        ),
                        "runtime_overlay",
                        "mount",
                        "physical_optimization",
                        0,
                        "json_contract",
                        {
                            "schema": (
                                "nerve.optimizer."
                                "vulkan_component_region_overlay.v2"
                            ),
                            "object_required": True,
                        },
                    ),
                    _output(
                        scale_artifact_path(
                            opportunity.component_id,
                            opportunity.node_id,
                        ),
                        "group_scale_parameter",
                        "residency",
                        "semantic_construction",
                        opportunity.output_features
                        * (opportunity.input_features // opportunity.group_size)
                        * 2,
                        "nonempty_binary",
                        {"minimum_byte_count": 1, "byte_multiple": 1},
                    ),
                    _output(
                        weight_artifact_path(
                            opportunity.component_id,
                            opportunity.node_id,
                        ),
                        "packed_int4_parameter",
                        "residency",
                        "semantic_construction",
                        opportunity.output_features
                        * opportunity.input_features
                        // 2,
                        "nonempty_binary",
                        {"minimum_byte_count": 1, "byte_multiple": 1},
                    ),
                )
            )
        for path, kind, phase, schema in (
            (
                COMPONENT_FIXTURE_PATH,
                "validation_fixture",
                "semantic_construction",
                COMPONENT_FIXTURE_SCHEMA,
            ),
            (
                CONVERSATION_FIXTURE_PATH,
                "validation_fixture",
                "semantic_construction",
                VALIDATION_CONVERSATION_SCHEMA,
            ),
            (
                MODEL_LIMITS_PATH,
                "validation_fixture",
                "semantic_construction",
                "nerve.optimizer.model_limits_fixture.v1",
            ),
            (
                PRODUCT_CONVERSATION_FIXTURE_PATH,
                "validation_fixture",
                "semantic_construction",
                VALIDATION_CONVERSATION_SCHEMA,
            ),
            (
                REPORT_PATH,
                "quantization_error_report",
                "semantic_construction",
                QUANTIZATION_REPORT_SCHEMA,
            ),
            (
                TENSOR_FRAGMENT_PATH,
                "tensor_index_fragment",
                "semantic_construction",
                "nerve.tensor_index.v1",
            ),
        ):
            outputs.append(
                _output(
                    path,
                    kind,
                    "mount" if path == TENSOR_FRAGMENT_PATH else "compile",
                    phase,
                    0,
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
                                opportunity.component_id,
                                opportunity.node_id,
                            ),
                        }
                    ]
                }
                for opportunity in opportunities
            ],
            "tensor_index_refs": [TENSOR_FRAGMENT_PATH],
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
        return group_scaled_int4_benchmark_workloads(
            _candidate_opportunities(context, candidate)[0]
        )

    def validation_requirements(
        self,
        context: ProviderContext,
        candidate: Json,
    ) -> Json:
        return group_scaled_int4_validation_requirements(
            candidate=candidate,
            opportunity=_candidate_opportunities(context, candidate)[0],
            speculative_draft_tokens=(
                context.qualification_regime.speculative_draft_tokens
            ),
        )


def _prepare(
    context: ProviderContext,
    opportunity: GroupScaledInt4Opportunity,
) -> PreparedInt4Region:
    key = (
        "group_scaled_int4.prepared.v1:"
        f"{opportunity.component_id}:{opportunity.node_id}:"
        f"{opportunity.performance_signature}:"
        f"{context.hardware_profile['capability_class']}"
    )

    def prepare() -> PreparedInt4Region:
        manifest = _source_json(context, opportunity.manifest_ref)
        tensor_index = _source_json(context, opportunity.tensor_index_ref)
        return prepare_group_scaled_int4_component_from_documents(
            opportunity=opportunity,
            manifest=manifest,
            tensor_index=tensor_index,
        )

    return context.memoized(key, prepare)  # type: ignore[return-value]


def _candidate_opportunity_sets(
    context: ProviderContext,
) -> tuple[tuple[GroupScaledInt4Opportunity, ...], ...]:
    return tuple(
        (opportunity,)
        for opportunity in discover_group_scaled_int4_linears(context)
    )


def _candidate_opportunities(
    context: ProviderContext,
    candidate: Json,
) -> tuple[GroupScaledInt4Opportunity, ...]:
    requested = set(candidate["scope_ids"])
    topology = candidate["representation"]["topology"]
    performance_signature = topology["performance_equivalence_class"]
    requested_regions = {
        (
            str(record["component_id"]),
            str(record["physical_node_id"]),
            str(record["source_parameter_ref_id"]),
        )
        for record in topology["component_region_ids"]
    }
    matches = tuple(
        opportunity
        for opportunity in discover_group_scaled_int4_linears(context)
        if opportunity.scope_id in requested
        and opportunity.performance_signature == performance_signature
        and (
            opportunity.component_id,
            opportunity.node_id,
            opportunity.source_weight_ref_id,
        )
        in requested_regions
    )
    matches = tuple(
        sorted(matches, key=lambda item: (item.component_id, item.node_id))
    )
    if {item.scope_id for item in matches} != requested or not matches:
        raise ModelCompileError(
            "group-scaled INT4 candidate does not map to exact component regions"
        )
    expected = sorted(
        (
            (item.scope_id, item.source_contract_digest)
            for item in matches
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
        raise ModelCompileError(
            "group-scaled INT4 candidate source contracts drifted"
        )
    expected_topology = {
        "kind": "source_anchored_parameterized_component_regions",
        "component_region_ids": [
            {
                "component_id": item.component_id,
                "physical_node_id": item.node_id,
                "source_parameter_ref_id": item.source_weight_ref_id,
            }
            for item in matches
        ],
        "region_count": len(matches),
        "performance_equivalence_class": matches[0].performance_signature,
    }
    if topology != expected_topology:
        raise ModelCompileError(
            "group-scaled INT4 candidate crosses physical performance classes"
        )
    return matches


def _target_predicate(
    context: ProviderContext,
    opportunity: GroupScaledInt4Opportunity,
) -> Json:
    return {
        "capability_class": context.hardware_profile["capability_class"],
        "device_kind": "gpu",
        "api": "vulkan",
        "required_processes": ["shader_vector"],
        "required_subgroup_operations": ["arithmetic", "basic"],
        "permitted_subgroup_sizes": [
            size for size in (1, 2, 4, 8, 16, 32, 64) if 64 % size == 0
        ],
        "minimum_workgroup_invocations": 64,
        "minimum_workgroup_size_x": 64,
        "subgroup_compute_required": True,
        "compiler_capabilities": deepcopy(opportunity.compiler_device),
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
    }


def _lowered_region(opportunity: GroupScaledInt4Opportunity) -> Json:
    tensor = opportunity.source_weight
    return {
        "scope_id": opportunity.scope_id,
        "source_contract_digest": opportunity.source_contract_digest,
        "component_id": opportunity.component_id,
        "node_id": opportunity.node_id,
        "evidence_ids": list(opportunity.evidence_ids),
        "source_artifact_refs": list(opportunity.source_artifact_refs),
        "manifest_ref": opportunity.manifest_ref,
        "circuit_ref": opportunity.circuit_ref,
        "tensor_index_ref": opportunity.tensor_index_ref,
        "source_weight_ref_id": opportunity.source_weight_ref_id,
        "source_weight_ref": deepcopy(opportunity.source_weight_ref),
        "source_weight": {
            "name": tensor.tensor_name,
            "metadata": tensor.metadata,
            "tensor_index": tensor.tensor_index.to_json(),
            "storage": tensor.storage.to_json(),
            "safetensors_header_bytes": tensor.safetensors_header_bytes,
            "payload_byte_offset": tensor.payload_byte_offset,
            "payload_byte_count": tensor.payload_byte_count,
        },
        "candidate": {
            "weight_tensor_name": opportunity.candidate_weight_name,
            "scale_tensor_name": opportunity.candidate_scale_name,
            "weight_ref_id": opportunity.replacement_weight_ref_id,
            "scale_ref_id": opportunity.replacement_scale_ref_id,
            "weight_path": weight_artifact_path(
                opportunity.component_id,
                opportunity.node_id,
            ),
            "scale_path": scale_artifact_path(
                opportunity.component_id,
                opportunity.node_id,
            ),
            "overlay_path": component_overlay_path(
                opportunity.component_id,
                opportunity.node_id,
            ),
        },
        "geometry": {
            "input_features": opportunity.input_features,
            "output_features": opportunity.output_features,
            "group_size": opportunity.group_size,
            "packed_shape": list(opportunity.packed_shape),
            "scale_shape": list(opportunity.scale_shape),
        },
        "compiler_device": deepcopy(opportunity.compiler_device),
        "max_context_activations": opportunity.max_context_activations,
        "performance_signature": opportunity.performance_signature,
    }


def _source_json(context: ProviderContext, path: str) -> Json:
    key = f"group_scaled_int4.source_json.v1:{path}"

    def load() -> Json:
        try:
            value = json.loads(context.source_artifacts.read_path(path))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ModelCompileError(f"{path} is not valid JSON") from error
        if not isinstance(value, dict):
            raise ModelCompileError(f"{path} must contain a JSON object")
        return value

    return context.memoized(key, load)  # type: ignore[return-value]


def _output(
    path: str,
    kind: str,
    lifetime: str,
    producer_phase: str,
    resident_bytes: int,
    validator_id: str,
    validation_contract: Json,
) -> Json:
    return {
        "path": path,
        "kind": kind,
        "lifetime": lifetime,
        "producer_phase": producer_phase,
        "resident_bytes": resident_bytes,
        "validator_id": validator_id,
        "validation_contract": validation_contract,
    }


__all__ = ["GroupScaledInt4LinearProvider"]
