from __future__ import annotations

from types import SimpleNamespace

from nerve.representation_optimizer.contracts import validate_contract
from nerve.representation_optimizer.providers.output_fp8.discovery import (
    OutputProjectionOpportunity,
)
from nerve.representation_optimizer.providers.output_fp8.provider import (
    BlockScaledOutputProjectionProvider,
)
from nerve.representation_optimizer.providers.output_fp8.workloads import (
    output_projection_benchmark_workloads,
)
from nerve.representation_optimizer.providers.source_artifacts import (
    SourceArtifact,
    SourceTensorArtifact,
)
from nerve.representation_optimizer.providers.types import (
    EvidenceAssessment,
)


def _opportunity() -> OutputProjectionOpportunity:
    index = SourceArtifact("parameters/tensors.json", "digest:index", 1)
    storage = SourceArtifact("parameters/model.safetensors", "digest:data", 1)
    tensor = SourceTensorArtifact(
        tensor_name="lm_head.weight",
        _metadata={
            "dtype": "BF16",
            "shape": [248_320, 2_048],
            "data_sha256": "0" * 64,
        },
        tensor_index=index,
        storage=storage,
        safetensors_header_bytes=8,
        payload_byte_offset=0,
        payload_byte_count=1_017_118_720,
    )
    return OutputProjectionOpportunity(
        scope_id=f"scope_{'0' * 32}",
        source_contract_digest=(f"nerve.optimizer.canonical_json_sha256.v1:{'1' * 64}"),
        component_id="output_transducer",
        physical_node_id="output_projection",
        norm_parameter_ref_id="output_norm.weight",
        projection_parameter_ref_id="output_projection.weight",
        projection_scale_parameter_ref_id=("output_projection.weight_scale_inv"),
        source_node_ids=("output_norm", "output_projection"),
        evidence_ids=(f"evidence_{'2' * 32}",),
        source_artifact_refs=("vulkan_resident_package.json",),
        manifest_ref="vulkan_resident_package.json",
        circuit_ref="circuits/output_transducer.json",
        tensor=tensor,
        norm_tensor_name="model.language_model.norm.weight",
        hidden_size=2_048,
        vocabulary_size=248_320,
        output_scale_token="1",
        fp8_process_names=("packed_dot_product", "shader_vector"),
        speculative_decoder_ids=("draft_00",),
        max_context_activations=131_072,
    )


def test_output_fp8_candidate_covers_decode_and_batched_prefill(
    monkeypatch,
) -> None:
    opportunity = _opportunity()
    monkeypatch.setattr(
        "nerve.representation_optimizer.providers.output_fp8.provider."
        "require_output_projection",
        lambda context: opportunity,
    )
    context = SimpleNamespace(
        hardware_profile={"capability_class": "hardware_capability_fixture"}
    )
    evidence = EvidenceAssessment(
        accepted=True,
        evidence_ids=(f"evidence_{'2' * 32}",),
        facts={"output_projection": True},
        reasons=("fixture",),
    )

    candidate = BlockScaledOutputProjectionProvider().synthesize_candidates(
        context,
        evidence,
    )[0]
    validate_contract(candidate)

    assert candidate["target_predicate"]["execution_envelope"] == {
        "phases": ["decode", "prefill"],
        "alternative_phases": ["decode", "prefill"],
        "source_retained_phases": [],
        "activation_batch": {"minimum": 1, "maximum": 131_072},
        "context_activations": {"minimum": 0, "maximum": 131_072},
        "state_activations": {"minimum": 0, "maximum": 131_072},
    }
    assert candidate["representation"]["extensions"]["role_specializations"] == [
        {
            "role": "speculative_draft",
            "decoder_ids": ["draft_00"],
            "parameter_format": {
                "kind": "packed_signed_int4_with_bf16_scales",
                "group_columns": 128,
            },
            "correctness_boundary": "target_model_verification",
        }
    ]


def test_output_fp8_microbenchmarks_each_replaced_execution_path() -> None:
    workloads = list(output_projection_benchmark_workloads(_opportunity()))

    assert [
        (
            workload["regime"]["execution_phase"],
            workload["regime"]["activation_batch_width"],
        )
        for workload in workloads
    ] == [("decode", 1), ("prefill", 4)]
    assert {workload["useful_work"]["minimum_units"] for workload in workloads} == {128}
