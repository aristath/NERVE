from __future__ import annotations

from copy import deepcopy

from nerve.compilation import Json
from nerve.representation_optimizer.contracts import (
    representation_candidate_id,
)
from nerve.representation_optimizer.providers.output_fp8.artifacts import (
    BATCH_SHADER_PATH,
    COMPONENT_FIXTURE_PATH,
    CONVERSATION_FIXTURE_PATH,
    DECODE_SHADER_PATH,
    DRAFT_DECODE_SHADER_PATH,
    DRAFT_SCALE_PATH,
    DRAFT_WEIGHT_PATH,
    ERROR_REPORT_PATH,
    MODEL_LIMITS_PATH,
    OVERLAY_PATH,
    PRODUCT_CONVERSATION_FIXTURE_PATH,
    SCALE_PATH,
    TENSOR_FRAGMENT_PATH,
    WEIGHT_PATH,
    artifact_paths,
)
from nerve.representation_optimizer.providers.output_fp8.contracts import (
    BLOCK_SCALED_OUTPUT_DESCRIPTOR_ID,
    TARGET_LOWERING_SCHEMA,
)
from nerve.representation_optimizer.providers.output_fp8.discovery import (
    OutputProjectionOpportunity,
    discover_output_projection,
    discover_output_projections,
    require_output_projection,
    source_inputs,
)
from nerve.representation_optimizer.providers.output_fp8.representation import (
    output_projection_representation_graph,
)
from nerve.representation_optimizer.providers.output_fp8.workloads import (
    output_projection_benchmark_workloads,
    output_projection_error_contract,
    output_projection_validation_requirements,
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


def _is_standalone_output_scope(scope: Json) -> bool:
    return (
        scope["kind"] == "output_transducer"
        and len(scope["members"]["component_ids"]) == 1
    )


class BlockScaledOutputProjectionProvider:
    identity = ProviderIdentity(
        "nerve.block_scaled_output_projection",
        "1",
    )
    descriptor_id = BLOCK_SCALED_OUTPUT_DESCRIPTOR_ID

    def may_optimize_scope(
        self,
        scope: Json,
        source_contract: Json,
    ) -> bool:
        del source_contract
        return _is_standalone_output_scope(scope)

    def required_analyzer_ids(
        self,
        scope: Json,
        source_contract: Json,
    ) -> tuple[str, ...]:
        del scope, source_contract
        return ("semantic_graph_structure",)

    def match_semantics(self, context: ProviderContext) -> MatchAssessment:
        eligible = [
            scope for scope in context.scopes if _is_standalone_output_scope(scope)
        ]
        return MatchAssessment(
            matched=bool(eligible),
            reasons=(
                ("scope exposes one independently mountable output transducer")
                if eligible
                else "no standalone output-transducer scope is present",
            ),
        )

    def match_structure(self, context: ProviderContext) -> MatchAssessment:
        opportunities = discover_output_projections(context)
        return MatchAssessment(
            matched=bool(opportunities),
            reasons=(
                (
                    f"discovered {len(opportunities)} compatible standalone "
                    "output projection"
                )
                if opportunities
                else _no_match_reason(context),
            ),
            evidence_ids=tuple(
                sorted(
                    {
                        evidence_id
                        for opportunity in opportunities
                        for evidence_id in opportunity.evidence_ids
                    }
                )
            ),
        )

    def analyze_evidence(
        self,
        context: ProviderContext,
    ) -> EvidenceAssessment:
        opportunities = discover_output_projections(context)
        if not opportunities:
            return EvidenceAssessment(
                accepted=False,
                evidence_ids=(),
                facts={},
                reasons=(_no_match_reason(context),),
            )
        if len(opportunities) != 1:
            return EvidenceAssessment(
                accepted=False,
                evidence_ids=tuple(
                    sorted(
                        {
                            evidence_id
                            for opportunity in opportunities
                            for evidence_id in opportunity.evidence_ids
                        }
                    )
                ),
                facts={},
                reasons=(
                    "multiple independently mountable output projections "
                    "require separate candidates",
                ),
            )
        opportunity = opportunities[0]
        return EvidenceAssessment(
            accepted=True,
            evidence_ids=opportunity.evidence_ids,
            facts={
                "component_id": opportunity.component_id,
                "source_tensor": opportunity.tensor.tensor_name,
                "source_dtype": "BF16",
                "source_shape": [
                    opportunity.vocabulary_size,
                    opportunity.hidden_size,
                ],
                "target_dtype": "F8_E4M3",
                "block_shape": [
                    opportunity.block_rows,
                    opportunity.block_columns,
                ],
                "native_fp8_processes": list(opportunity.fp8_process_names),
            },
            reasons=(
                "source geometry, hardware-native F8 packed dot products, and "
                "the output-transducer mount boundary are compatible",
            ),
        )

    def synthesize_candidates(
        self,
        context: ProviderContext,
        evidence: EvidenceAssessment,
    ) -> tuple[Json, ...]:
        opportunity = require_output_projection(context)
        error_contract = output_projection_error_contract()
        candidate = {
            "schema": "nerve.optimizer.representation_candidate.v1",
            "candidate_id": "",
            "scope_ids": [opportunity.scope_id],
            "source_contract_digests": [opportunity.source_contract_digest],
            "provider": self.identity.to_json(),
            "descriptor_id": self.descriptor_id,
            "evidence_refs": list(evidence.evidence_ids),
            "representation": {
                "kind": "block_scaled_fp8_e4m3_output_projection",
                "signal_formats": [
                    {"name": "dense_bf16_input"},
                    {"name": "dense_f32_logits"},
                ],
                "parameter_format": {
                    "kind": "fp8_e4m3_with_bf16_inverse_scales",
                    "block_rows": opportunity.block_rows,
                    "block_columns": opportunity.block_columns,
                    "source_dtype": "BF16",
                },
                "state_format": {"kind": "source_state_unchanged"},
                "topology": {
                    "kind": "output_transducer_projection_replacement",
                    "component_ids": [opportunity.component_id],
                },
                "extensions": {
                    "role_specializations": (
                        [
                            {
                                "role": "speculative_draft",
                                "decoder_ids": list(
                                    opportunity.speculative_decoder_ids
                                ),
                                "parameter_format": {
                                    "kind": "packed_signed_int4_with_bf16_scales",
                                    "group_columns": (opportunity.draft_group_columns),
                                },
                                "correctness_boundary": ("target_model_verification"),
                            }
                        ]
                        if opportunity.has_role_specialized_draft
                        else []
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
                "mode": "approximate",
                "proof_obligations": [],
                "error_contract": error_contract,
            },
            "artifact_declarations": [
                {"path": path}
                for path in artifact_paths(
                    role_specialized_draft=(opportunity.has_role_specialized_draft)
                )
            ],
        }
        candidate["candidate_id"] = representation_candidate_id(candidate)
        return (candidate,)

    def emit_representation_ir(
        self,
        context: ProviderContext,
        candidate: Json,
    ) -> Json:
        return output_projection_representation_graph(
            candidate=candidate,
            opportunity=require_output_projection(context),
            capability_class=str(context.hardware_profile["capability_class"]),
        )

    def lower_for_target(
        self,
        context: ProviderContext,
        candidate: Json,
        representation_ir: Json,
    ) -> Json:
        opportunity = require_output_projection(context)
        tensor = opportunity.tensor
        return {
            "schema": TARGET_LOWERING_SCHEMA,
            "candidate_id": candidate["candidate_id"],
            "representation_graph_id": representation_ir["graph_id"],
            "scope_id": opportunity.scope_id,
            "capability_class": context.hardware_profile["capability_class"],
            "source": {
                "component_id": opportunity.component_id,
                "physical_node_id": opportunity.physical_node_id,
                "norm_parameter_ref_id": (opportunity.norm_parameter_ref_id),
                "projection_parameter_ref_id": (
                    opportunity.projection_parameter_ref_id
                ),
                "projection_scale_parameter_ref_id": (
                    opportunity.projection_scale_parameter_ref_id
                ),
                "source_node_ids": list(opportunity.source_node_ids),
                "manifest_ref": opportunity.manifest_ref,
                "circuit_ref": opportunity.circuit_ref,
                "artifact_refs": list(opportunity.source_artifact_refs),
                "source_inputs": source_inputs(context, opportunity),
                "projection": {
                    "name": tensor.tensor_name,
                    "metadata": tensor.metadata,
                    "storage": tensor.storage.to_json(),
                    "payload_byte_offset": tensor.payload_byte_offset,
                    "payload_byte_count": tensor.payload_byte_count,
                },
                "norm_tensor_name": opportunity.norm_tensor_name,
            },
            "geometry": {
                "hidden_size": opportunity.hidden_size,
                "vocabulary_size": opportunity.vocabulary_size,
                "block_rows": opportunity.block_rows,
                "block_columns": opportunity.block_columns,
                "scale_shape": list(opportunity.scale_shape),
                "draft_scale_shape": list(opportunity.draft_scale_shape),
            },
            "parameters": {
                "weight_tensor_name": (opportunity.candidate_weight_name),
                "scale_tensor_name": (opportunity.candidate_scale_name),
                "draft_weight_tensor_name": (opportunity.draft_weight_name),
                "draft_scale_tensor_name": (opportunity.draft_scale_name),
            },
            "artifacts": {
                "weight_path": WEIGHT_PATH,
                "scale_path": SCALE_PATH,
                "tensor_fragment_path": TENSOR_FRAGMENT_PATH,
                "overlay_path": OVERLAY_PATH,
                "decode_shader_path": DECODE_SHADER_PATH,
                "batch_shader_path": BATCH_SHADER_PATH,
                "error_report_path": ERROR_REPORT_PATH,
                "component_fixture_path": COMPONENT_FIXTURE_PATH,
                "conversation_fixture_path": CONVERSATION_FIXTURE_PATH,
                "product_conversation_fixture_path": (
                    PRODUCT_CONVERSATION_FIXTURE_PATH
                ),
                "model_limits_path": MODEL_LIMITS_PATH,
            },
            "runtime": {
                "output_scale_token": opportunity.output_scale_token,
                "batch_lane_tile_width": 4,
                "max_context_activations": (opportunity.max_context_activations),
                "required_vulkan_version": "1.4",
                "role_specialized_draft": (opportunity.has_role_specialized_draft),
                "speculative_decoder_ids": list(opportunity.speculative_decoder_ids),
            },
        }

    def estimate_static_cost(
        self,
        context: ProviderContext,
        candidate: Json,
        representation_ir: Json,
        target_lowering: Json,
    ) -> StaticEstimate:
        opportunity = require_output_projection(context)
        scale_elements = opportunity.scale_shape[0] * opportunity.scale_shape[1]
        candidate_bytes = (
            opportunity.vocabulary_size * opportunity.hidden_size + scale_elements * 2
        )
        draft_bytes = 0
        if opportunity.has_role_specialized_draft:
            draft_bytes = (
                opportunity.vocabulary_size * opportunity.hidden_size // 2
                + opportunity.draft_scale_shape[0]
                * opportunity.draft_scale_shape[1]
                * 2
            )
        return StaticEstimate(
            feasible=True,
            permanent_bytes=candidate_bytes + draft_bytes,
            transient_bytes=(
                opportunity.block_rows * opportunity.hidden_size * 3
                + opportunity.scale_shape[1] * 4
            ),
            construction_nanoseconds=None,
            steady_state_work={
                "kind": "native_fp8_dot4_acc32",
                "source_parameter_bytes": (opportunity.tensor.payload_byte_count),
                "candidate_parameter_bytes": candidate_bytes,
                "parameter_byte_ratio": (
                    (candidate_bytes + draft_bytes)
                    / opportunity.tensor.payload_byte_count
                ),
                "role_specialized_draft_parameter_bytes": draft_bytes,
            },
            reasons=(
                "candidate halves authoritative projection traffic with FP8; "
                "speculative decoders additionally use packed INT4 when "
                "target verification preserves final-token correctness",
            ),
        )

    def construction_requirements(
        self,
        context: ProviderContext,
        candidate: Json,
    ) -> Json:
        return _build_plan(context, require_output_projection(context))

    def mount_requirements(
        self,
        context: ProviderContext,
        candidate: Json,
    ) -> Json:
        opportunity = require_output_projection(context)
        return {
            "schema": "nerve.optimizer.runtime_mount_plan.v3",
            "candidate_id": candidate["candidate_id"],
            "adapter_id": "vulkan_stream_circuit_overlay.v2",
            "regions": [
                {
                    "replacements": [
                        {
                            "kind": "output_transducer",
                            "source_component_id": (opportunity.component_id),
                            "overlay_ref": OVERLAY_PATH,
                        }
                    ]
                }
            ],
            "tensor_index_refs": [TENSOR_FRAGMENT_PATH],
        }

    def proof_or_error_contract(
        self,
        context: ProviderContext,
        candidate: Json,
    ) -> Json:
        return deepcopy(candidate["behavioral_contract"])

    def benchmark_workloads(
        self,
        context: ProviderContext,
        candidate: Json,
    ) -> tuple[Json, ...]:
        return output_projection_benchmark_workloads(require_output_projection(context))

    def validation_requirements(
        self,
        context: ProviderContext,
        candidate: Json,
    ) -> Json:
        return output_projection_validation_requirements(
            candidate=candidate,
            opportunity=require_output_projection(context),
            speculative_draft_tokens=(
                context.qualification_regime.speculative_draft_tokens
            ),
        )


def _build_plan(
    context: ProviderContext,
    opportunity: OutputProjectionOpportunity,
) -> Json:
    binary = {
        "validator_id": "nonempty_binary",
        "validation_contract": {
            "minimum_byte_count": 1,
            "byte_multiple": 1,
        },
    }

    def json_contract(schema: str) -> Json:
        return {
            "validator_id": "json_contract",
            "validation_contract": {
                "schema": schema,
                "object_required": True,
            },
        }

    outputs = [
        _output(
            BATCH_SHADER_PATH,
            "vulkan_shader",
            "residency",
            "physical_optimization",
            0,
            {
                "validator_id": "spirv_module",
                "validation_contract": {"minimum_version": 0x00010600},
            },
        ),
        _output(
            COMPONENT_FIXTURE_PATH,
            "validation_fixture",
            "compile",
            "semantic_construction",
            0,
            json_contract("nerve.optimizer.output_projection_fixture.v1"),
        ),
        _output(
            CONVERSATION_FIXTURE_PATH,
            "validation_fixture",
            "compile",
            "semantic_construction",
            0,
            json_contract(VALIDATION_CONVERSATION_SCHEMA),
        ),
        _output(
            DECODE_SHADER_PATH,
            "vulkan_shader",
            "residency",
            "physical_optimization",
            0,
            {
                "validator_id": "spirv_module",
                "validation_contract": {"minimum_version": 0x00010600},
            },
        ),
        _output(
            ERROR_REPORT_PATH,
            "quantization_error_report",
            "compile",
            "semantic_construction",
            0,
            json_contract("nerve.optimizer.block_scaled_quantization_report.v1"),
        ),
        _output(
            MODEL_LIMITS_PATH,
            "validation_fixture",
            "compile",
            "semantic_construction",
            0,
            json_contract("nerve.optimizer.model_limits_fixture.v1"),
        ),
        _output(
            OVERLAY_PATH,
            "runtime_overlay",
            "mount",
            "ordinary_lowering",
            0,
            json_contract("nerve.optimizer.vulkan_output_transducer_overlay.v1"),
        ),
        _output(
            PRODUCT_CONVERSATION_FIXTURE_PATH,
            "validation_fixture",
            "compile",
            "semantic_construction",
            0,
            json_contract(VALIDATION_CONVERSATION_SCHEMA),
        ),
        _output(
            SCALE_PATH,
            "block_scale_parameter",
            "residency",
            "semantic_construction",
            (opportunity.scale_shape[0] * opportunity.scale_shape[1] * 2),
            binary,
        ),
        _output(
            TENSOR_FRAGMENT_PATH,
            "tensor_index_fragment",
            "mount",
            "semantic_construction",
            0,
            json_contract("nerve.tensor_index.v1"),
        ),
        _output(
            WEIGHT_PATH,
            "block_scaled_parameter",
            "residency",
            "semantic_construction",
            opportunity.vocabulary_size * opportunity.hidden_size,
            binary,
        ),
    ]
    if opportunity.has_role_specialized_draft:
        outputs.extend(
            [
                _output(
                    DRAFT_DECODE_SHADER_PATH,
                    "vulkan_shader",
                    "residency",
                    "physical_optimization",
                    0,
                    {
                        "validator_id": "spirv_module",
                        "validation_contract": {"minimum_version": 0x00010600},
                    },
                ),
                _output(
                    DRAFT_SCALE_PATH,
                    "block_scale_parameter",
                    "residency",
                    "semantic_construction",
                    (
                        opportunity.draft_scale_shape[0]
                        * opportunity.draft_scale_shape[1]
                        * 2
                    ),
                    binary,
                ),
                _output(
                    DRAFT_WEIGHT_PATH,
                    "packed_int4_parameter",
                    "residency",
                    "semantic_construction",
                    (opportunity.vocabulary_size * opportunity.hidden_size // 2),
                    binary,
                ),
            ]
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


def _no_match_reason(context: ProviderContext) -> str:
    if len(context.scope_ids) == 1:
        return discover_output_projection(context).reasons[0]
    return "no compatible standalone BF16 output projection was found"


def _output(
    path: str,
    kind: str,
    lifetime: str,
    phase: str,
    resident_bytes: int,
    validation: Json,
) -> Json:
    return {
        "path": path,
        "kind": kind,
        "lifetime": lifetime,
        "producer_phase": phase,
        "resident_bytes": resident_bytes,
        **validation,
    }
