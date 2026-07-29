from __future__ import annotations

import json
from copy import deepcopy

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.contracts import (
    canonical_json_bytes,
    representation_candidate_id,
)
from nerve.representation_optimizer.providers.codebook.discovery import (
    HeadNormCodebookOpportunity,
    discover_head_norm_codebook,
    discover_head_norm_codebooks,
)
from nerve.representation_optimizer.providers.codebook.embedded_artifacts import (
    COMPONENT_FIXTURE_PATH,
    CONVERSATION_FIXTURE_PATH,
    DECODE_SHADER_PATH,
    MODEL_LIMITS_PATH,
    OVERLAY_PATH,
    PROOF_PATH,
    embedded_artifact_paths,
)
from nerve.representation_optimizer.providers.codebook.embedded_contracts import (
    EMBEDDED_PARAMETER_PROGRAM_PROOF_SCHEMA,
    EMBEDDED_PARAMETER_PROGRAM_PROOF_VERIFIER_ID,
    EMBEDDED_TARGET_LOWERING_SCHEMA,
)
from nerve.representation_optimizer.providers.codebook.embedded_representation import (
    embedded_parameter_program_representation_graph,
)
from nerve.representation_optimizer.providers.codebook.member_paths import (
    member_path,
)
from nerve.representation_optimizer.providers.codebook.provider import (
    LOOKUP_CODEBOOK_DESCRIPTOR_ID,
)
from nerve.representation_optimizer.providers.codebook.representation_bundle import (
    bundle_representation_graphs,
)
from nerve.representation_optimizer.providers.codebook.workloads import (
    bundled_head_norm_validation_requirements,
    head_norm_benchmark_workloads,
)
from nerve.representation_optimizer.providers.types import (
    EvidenceAssessment,
    MatchAssessment,
    ProviderContext,
    ProviderIdentity,
    StaticEstimate,
)


class ExactEmbeddedHeadNormParameterProgramProvider:
    identity = ProviderIdentity(
        "nerve.exact_embedded_head_norm_parameter_program",
        "2",
    )
    descriptor_id = LOOKUP_CODEBOOK_DESCRIPTOR_ID

    def match_semantics(self, context: ProviderContext) -> MatchAssessment:
        eligible = [
            scope
            for scope in context.scopes
            if len(scope["members"]["component_ids"]) == 1
            and len(scope["boundary"]["parameters"]) == 2
        ]
        matched = bool(eligible)
        return MatchAssessment(
            matched=matched,
            reasons=(
                (
                    f"{len(eligible)} scopes expose independently mountable "
                    "components with two immutable parameter bindings"
                )
                if matched
                else (
                    "no scope exposes one component with two immutable "
                    "parameter bindings"
                ),
            ),
        )

    def match_structure(self, context: ProviderContext) -> MatchAssessment:
        opportunities = _opportunities(context, required=False)
        return MatchAssessment(
            matched=bool(opportunities),
            reasons=(
                (
                    f"discovered {len(opportunities)} non-overlapping exact "
                    "head-normalization parameter programs"
                )
                if opportunities
                else _no_match_reason(context)
            ,),
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

    def analyze_evidence(self, context: ProviderContext) -> EvidenceAssessment:
        opportunities = _opportunities(context, required=False)
        evidence_ids = tuple(
            sorted(
                {
                    evidence_id
                    for opportunity in opportunities
                    for evidence_id in opportunity.evidence_ids
                }
            )
        )
        if not opportunities:
            return EvidenceAssessment(
                accepted=False,
                evidence_ids=evidence_ids,
                facts={},
                reasons=("no exact compatible parameter program was found",),
            )
        return EvidenceAssessment(
            accepted=True,
            evidence_ids=evidence_ids,
            facts={
                "member_count": len(opportunities),
                "members": [
                    {
                        "scope_id": opportunity.scope_id,
                        "component_id": opportunity.component_id,
                        "physical_node_id": opportunity.physical_node_id,
                        "branch_count": len(opportunity.branches),
                        "head_width": opportunity.head_width,
                        "source_tensor_names": [
                            branch.tensor_name for branch in opportunity.branches
                        ],
                        "source_tensor_data_sha256": [
                            branch.tensor.metadata["data_sha256"]
                            for branch in opportunity.branches
                        ],
                    }
                    for opportunity in opportunities
                ],
                "embedded_address_count": sum(
                    len(branch.indices)
                    for opportunity in opportunities
                    for branch in opportunity.branches
                ),
                "proof_domain": "all stored BF16 bit patterns",
            },
            reasons=(
                "exhaustive codebook analysis proves every stored BF16 value, "
                "allowing both exact branch tensors to be compiled directly into "
                "the target program",
            ),
        )

    def synthesize_candidates(
        self,
        context: ProviderContext,
        evidence: EvidenceAssessment,
    ) -> tuple[Json, ...]:
        opportunities = _opportunities(context)
        manifest = _source_json(context, opportunities[0].manifest_ref)
        max_context_activations = int(manifest["max_context_activations"])
        candidate = {
            "schema": "nerve.optimizer.representation_candidate.v1",
            "candidate_id": "",
            "scope_ids": [item.scope_id for item in opportunities],
            "source_contract_digests": [
                item.source_contract_digest for item in opportunities
            ],
            "provider": self.identity.to_json(),
            "descriptor_id": self.descriptor_id,
            "evidence_refs": list(evidence.evidence_ids),
            "representation": {
                "kind": "exact_embedded_bf16_head_norm_parameter_program",
                "signal_formats": [
                    {"name": "dense_bf16_component_boundary"},
                ],
                "parameter_format": {
                    "kind": "spirv_constant_bf16_parameter_program",
                    "execution_phases": ["decode"],
                    "source_retained_execution_phases": ["prefill"],
                    "member_count": len(opportunities),
                    "entry_dtype": "BF16",
                    "members": [
                        {
                            "scope_id": opportunity.scope_id,
                            "branch_count": len(opportunity.branches),
                            "elements_per_branch": opportunity.head_width,
                            "source_data_sha256": [
                                branch.tensor.metadata["data_sha256"]
                                for branch in opportunity.branches
                            ],
                        }
                        for opportunity in opportunities
                    ],
                },
                "state_format": {"kind": "source_state_unchanged"},
                "topology": {
                    "kind": (
                        "phase_selective_non_overlapping_fused_head_norm_rope_"
                        "dispatch_set"
                    ),
                    "component_ids": [
                        opportunity.component_id
                        for opportunity in opportunities
                    ],
                },
            },
            "target_predicate": {
                "capability_class": context.hardware_profile["capability_class"],
                "device_kind": "gpu",
                "api": "vulkan",
                "required_processes": [
                    "device_cache_hierarchy",
                    "shader_scalar",
                ],
                "execution_envelope": {
                    "phases": ["decode", "prefill"],
                    "alternative_phases": ["decode"],
                    "source_retained_phases": ["prefill"],
                    "activation_batch": {
                        "minimum": 1,
                        "maximum": max_context_activations,
                    },
                    "context_activations": {
                        "minimum": 0,
                        "maximum": max_context_activations,
                    },
                    "state_activations": {
                        "minimum": 0,
                        "maximum": max_context_activations,
                    },
                },
            },
            "behavioral_contract": {
                "mode": "exact",
                "proof_obligations": [
                    "embedded_parameter_program_preserves_source_rounding",
                    "embedded_parameter_program_reconstructs_source_bf16_bits",
                ],
                "error_contract": None,
            },
            "artifact_declarations": [
                {
                    "path": member_path(opportunity.scope_id, path),
                }
                for opportunity in opportunities
                for path in embedded_artifact_paths()
            ],
        }
        candidate["candidate_id"] = representation_candidate_id(candidate)
        return (candidate,)

    def emit_representation_ir(
        self,
        context: ProviderContext,
        candidate: Json,
    ) -> Json:
        return bundle_representation_graphs(
            candidate=candidate,
            members=tuple(
                (
                    opportunity.scope_id,
                    embedded_parameter_program_representation_graph(
                        candidate=candidate,
                        opportunity=opportunity,
                        capability_class=str(
                            context.hardware_profile["capability_class"]
                        ),
                    ),
                )
                for opportunity in _opportunities(context)
            ),
        )

    def lower_for_target(
        self,
        context: ProviderContext,
        candidate: Json,
        representation_ir: Json,
    ) -> Json:
        return {
            "schema": EMBEDDED_TARGET_LOWERING_SCHEMA,
            "candidate_id": candidate["candidate_id"],
            "representation_graph_id": representation_ir["graph_id"],
            "capability_class": context.hardware_profile["capability_class"],
            "members": [
                _member_lowering(context, candidate, opportunity)
                for opportunity in _opportunities(context)
            ],
        }

    def estimate_static_cost(
        self,
        context: ProviderContext,
        candidate: Json,
        representation_ir: Json,
        target_lowering: Json,
    ) -> StaticEstimate:
        opportunities = _opportunities(context)
        return StaticEstimate(
            feasible=True,
            permanent_bytes=sum(
                item.original_parameter_bytes for item in opportunities
            ),
            transient_bytes=max(
                item.original_parameter_bytes for item in opportunities
            ),
            construction_nanoseconds=None,
            steady_state_work={
                "decode_parameter_buffer_reads_per_dispatch": 0,
                "prefill_parameter_buffer_reads_per_dispatch": 2,
                "embedded_bf16_elements": sum(
                    len(branch.raw_values)
                    for opportunity in opportunities
                    for branch in opportunity.branches
                ),
                "qualified_component_count": len(opportunities),
                "dispatch_count_change": 0,
            },
            reasons=(
                "candidate embeds exact BF16 constants only in the decode "
                "program while retaining the source parameter buffers and "
                "source batch kernels for prefill; construction temporarily "
                "reads the parameters to prove the embedded program and decode "
                "dispatches perform no parameter-buffer reads",
            ),
        )

    def construction_requirements(
        self,
        context: ProviderContext,
        candidate: Json,
    ) -> Json:
        return _build_plan(context, _opportunities(context))

    def mount_requirements(
        self,
        context: ProviderContext,
        candidate: Json,
    ) -> Json:
        opportunities = _opportunities(context)
        return {
            "schema": "nerve.optimizer.runtime_mount_plan.v2",
            "candidate_id": candidate["candidate_id"],
            "adapter_id": "vulkan_stream_circuit_component_overlay.v1",
            "regions": [
                {
                    "component_replacements": [
                        {
                            "source_component_id": opportunity.component_id,
                            "overlay_ref": member_path(
                                opportunity.scope_id, OVERLAY_PATH
                            ),
                        }
                    ],
                }
                for opportunity in sorted(
                    opportunities,
                    key=lambda item: item.component_id,
                )
            ],
            "tensor_index_refs": [],
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
        opportunities = _opportunities(context)
        qualified = tuple(item.component_id for item in opportunities)
        return tuple(
            workload
            for opportunity in _execution_class_representatives(opportunities)
            for workload in head_norm_benchmark_workloads(
                opportunity,
                representation_name="embedded BF16 parameter program",
                artifact_scope_id=opportunity.scope_id,
                qualified_component_ids=qualified,
                execution_phases=("decode",),
            )
        )

    def validation_requirements(
        self,
        context: ProviderContext,
        candidate: Json,
    ) -> Json:
        opportunities = _opportunities(context)
        manifest = _source_json(context, opportunities[0].manifest_ref)
        return bundled_head_norm_validation_requirements(
            candidate=candidate,
            opportunities=opportunities,
            max_context_activations=int(manifest["max_context_activations"]),
            proof_verifier_id=EMBEDDED_PARAMETER_PROGRAM_PROOF_VERIFIER_ID,
            representation_name="embedded BF16 parameter program",
            execution_phases=("decode",),
            speculative_draft_tokens=(
                3 if manifest.get("speculative_decoders") else 0
            ),
        )


def _opportunities(
    context: ProviderContext,
    *,
    required: bool = True,
) -> tuple[HeadNormCodebookOpportunity, ...]:
    opportunities = discover_head_norm_codebooks(context)
    if required and not opportunities:
        raise ModelCompileError(
            "no exact embedded head-normalization parameter program was found"
        )
    return opportunities


def _no_match_reason(context: ProviderContext) -> str:
    if len(context.scope_ids) == 1:
        return discover_head_norm_codebook(context).reasons[0]
    return "no exact compatible head-normalization parameter program was found"


def _source_json(context: ProviderContext, path: str) -> Json:
    try:
        document = json.loads(context.source_artifacts.read_path(path))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelCompileError(
            f"provider source {path!r} is not valid JSON"
        ) from error
    if not isinstance(document, dict):
        raise ModelCompileError(f"provider source {path!r} must be a JSON object")
    return document


def _source_inputs(
    context: ProviderContext,
    opportunity: HeadNormCodebookOpportunity,
) -> list[Json]:
    artifacts = {}
    for branch in opportunity.branches:
        for source_input in branch.tensor.source_inputs:
            artifacts[source_input["path"]] = source_input
    for path in (
        *opportunity.source_artifact_refs,
        opportunity.manifest_ref,
    ):
        artifact = context.source_artifacts.resolve_path(path)
        artifacts[artifact.path] = artifact.source_input()
    return [artifacts[path] for path in sorted(artifacts)]


def _member_lowering(
    context: ProviderContext,
    candidate: Json,
    opportunity: HeadNormCodebookOpportunity,
) -> Json:
    manifest = _source_json(context, opportunity.manifest_ref)
    return {
        "schema": EMBEDDED_TARGET_LOWERING_SCHEMA,
        "candidate_id": candidate["candidate_id"],
        "scope_id": opportunity.scope_id,
        "source": {
            "component_id": opportunity.component_id,
            "circuit_ref": opportunity.circuit_ref,
            "manifest_ref": opportunity.manifest_ref,
            "artifact_refs": list(opportunity.source_artifact_refs),
            "physical_node_id": opportunity.physical_node_id,
            "physical_source_node_ids": list(opportunity.physical_source_node_ids),
            "source_inputs": _source_inputs(context, opportunity),
        },
        "geometry": {
            "head_width": opportunity.head_width,
            "branch_head_counts": [
                int(branch.attrs["head_count"])
                for branch in opportunity.branches
            ],
            "physical_attrs": deepcopy(opportunity.physical_attrs),
        },
        "parameters": {
            "source_tensors": [
                {
                    "name": branch.tensor_name,
                    "parameter_ref_id": branch.parameter_ref_id,
                    "metadata": branch.tensor.metadata,
                    "storage": branch.tensor.storage.to_json(),
                    "payload_byte_offset": branch.tensor.payload_byte_offset,
                    "payload_byte_count": branch.tensor.payload_byte_count,
                }
                for branch in opportunity.branches
            ],
            "branch_values_u16": [
                list(branch.raw_values) for branch in opportunity.branches
            ],
        },
        "artifacts": {
            "overlay_path": OVERLAY_PATH,
            "decode_shader_path": DECODE_SHADER_PATH,
            "proof_path": PROOF_PATH,
            "component_fixture_path": COMPONENT_FIXTURE_PATH,
            "conversation_fixture_path": CONVERSATION_FIXTURE_PATH,
            "model_limits_path": MODEL_LIMITS_PATH,
        },
        "runtime": {
            "source_abi_op": "parallel_head_norm_rope_2way",
            "implementation": (
                "exact_embedded_head_norm_parameter_program_v2"
            ),
            "alternative_execution_phases": ["decode"],
            "source_retained_execution_phases": ["prefill"],
            "max_context_activations": int(manifest["max_context_activations"]),
            "required_vulkan_version": "1.4",
        },
    }


def _execution_class_representatives(
    opportunities: tuple[HeadNormCodebookOpportunity, ...],
) -> tuple[HeadNormCodebookOpportunity, ...]:
    representatives: dict[bytes, HeadNormCodebookOpportunity] = {}
    for opportunity in opportunities:
        key = canonical_json_bytes(
            {
                "physical_attrs": opportunity.physical_attrs,
                "head_width": opportunity.head_width,
                "branch_head_counts": [
                    int(branch.attrs["head_count"])
                    for branch in opportunity.branches
                ],
            }
        )
        representatives.setdefault(key, opportunity)
    return tuple(
        sorted(
            representatives.values(),
            key=lambda item: (item.scope_id, item.component_id),
        )
    )


def _build_plan(
    context: ProviderContext,
    opportunities: tuple[HeadNormCodebookOpportunity, ...],
) -> Json:
    def json_contract(schema: str) -> Json:
        return {
            "validator_id": "json_contract",
            "validation_contract": {
                "schema": schema,
                "object_required": True,
            },
        }

    outputs = []
    for opportunity in opportunities:
        prefix = lambda path: member_path(opportunity.scope_id, path)
        outputs.extend(
            [
        _output(
            prefix(COMPONENT_FIXTURE_PATH),
            "validation_fixture",
            "compile",
            "semantic_construction",
            0,
            json_contract("nerve.optimizer.head_norm_fixture.v1"),
        ),
        _output(
            prefix(CONVERSATION_FIXTURE_PATH),
            "validation_fixture",
            "compile",
            "semantic_construction",
            0,
            json_contract("nerve.optimizer.validation_conversation.v1"),
        ),
        _output(
            prefix(DECODE_SHADER_PATH),
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
            prefix(MODEL_LIMITS_PATH),
            "validation_fixture",
            "compile",
            "semantic_construction",
            0,
            json_contract("nerve.optimizer.model_limits_fixture.v1"),
        ),
        _output(
            prefix(OVERLAY_PATH),
            "runtime_component_overlay",
            "mount",
            "ordinary_lowering",
            0,
            json_contract("nerve.optimizer.vulkan_component_overlay.v1"),
        ),
        _output(
            prefix(PROOF_PATH),
            "equivalence_proof",
            "compile",
            "physical_optimization",
            0,
            json_contract(EMBEDDED_PARAMETER_PROGRAM_PROOF_SCHEMA),
        ),
            ]
        )
    outputs.sort(key=lambda item: item["path"])
    source_inputs = {
        item["path"]: item
        for opportunity in opportunities
        for item in _source_inputs(context, opportunity)
    }
    return {
        "schema": "nerve.optimizer.candidate_build_plan.v1",
        "phases": [
            "semantic_construction",
            "ordinary_lowering",
            "physical_optimization",
        ],
        "source_inputs": [source_inputs[path] for path in sorted(source_inputs)],
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
