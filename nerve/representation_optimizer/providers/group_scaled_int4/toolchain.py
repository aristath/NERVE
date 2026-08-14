from __future__ import annotations

import json
import struct
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import numpy as np

from nerve.compilation import Json, ModelCompileError
from nerve.model_package_shader_templates import render_shader_source
from nerve.model_package_tensors import (
    bf16_bytes_to_f32,
    bf16_bytes_to_f32_matrix,
    compiled_safetensors_header,
    f32_to_bf16_bytes,
)
from nerve.representation_optimizer.automation.target import CandidateToolchain
from nerve.representation_optimizer.contracts import contract_digest
from nerve.representation_optimizer.providers.codebook.shaders import compile_spirv
from nerve.representation_optimizer.providers.group_scaled_int4.artifacts import (
    COMPONENT_FIXTURE_PATH,
    CONVERSATION_FIXTURE_PATH,
    MODEL_LIMITS_PATH,
    PRODUCT_CONVERSATION_FIXTURE_PATH,
    REPORT_PATH,
    TENSOR_FRAGMENT_PATH,
    component_fixture,
    conversation_fixture,
    model_limits_fixture,
    product_conversation_fixture,
)
from nerve.representation_optimizer.providers.group_scaled_int4.contracts import (
    QUANTIZATION_REPORT_SCHEMA,
    TARGET_LOWERING_SCHEMA,
)
from nerve.representation_optimizer.providers.group_scaled_int4.discovery import (
    GroupScaledInt4Opportunity,
)
from nerve.representation_optimizer.providers.group_scaled_int4.physical import (
    finalize_group_scaled_int4_kernel,
    prepare_group_scaled_int4_component_from_documents,
)
from nerve.representation_optimizer.providers.source_artifacts import (
    SourceArtifact,
    SourceTensorArtifact,
)
from nerve.representation_optimizer.providers.types import ProviderCandidatePlan
from nerve.representation_optimizer.staging.workspace import (
    CandidateConstructionContext,
)


_SHADER_ROOT = Path(__file__).resolve().parents[4] / "runtime-rs" / "shaders"
_CONSTRUCTION_ROW_BATCH = 256


class GroupScaledInt4ToolchainResolver:
    def resolve(self, plan: ProviderCandidatePlan) -> CandidateToolchain:
        if (
            plan.provider.provider_id != "nerve.group_scaled_int4_linear"
            or plan.target_lowering.get("schema") != TARGET_LOWERING_SCHEMA
        ):
            raise ModelCompileError(
                "group-scaled INT4 toolchain cannot construct provider "
                f"{plan.provider.provider_id!r}"
            )
        return CandidateToolchain(
            semantic_constructor=GroupScaledInt4SemanticConstructor(),
            ordinary_relowerer=GroupScaledInt4OrdinaryRelowerer(),
            physical_optimizer=GroupScaledInt4PhysicalOptimizer(),
        )


class GroupScaledInt4SemanticConstructor:
    def construct_semantic_artifacts(
        self,
        context: CandidateConstructionContext,
    ) -> None:
        lowering = _lowering(context)
        opportunities = opportunities_from_lowering(lowering)
        tensor_index = _json_object(
            context.read_source_artifact("tensors.json"),
            "tensors.json",
        )
        tensors = {}
        reports = []
        total_squared_source = 0.0
        total_squared_error = 0.0
        maximum_absolute_error = 0.0
        total_elements = 0
        for opportunity in opportunities:
            source = opportunity.source_weight
            if (
                tensor_index.get("tensors", {}).get(source.tensor_name)
                != source.metadata
            ):
                raise ModelCompileError(
                    f"group-scaled INT4 source tensor {source.tensor_name!r} drifted"
                )
            state = _QuantizationState()
            weight_path = str(
                _region(lowering, opportunity)["candidate"]["weight_path"]
            )
            scale_path = str(
                _region(lowering, opportunity)["candidate"]["scale_path"]
            )
            weight_header = compiled_safetensors_header(
                opportunity.candidate_weight_name,
                dtype="I32",
                shape=list(opportunity.packed_shape),
                byte_count=(
                    opportunity.output_features * opportunity.input_features // 2
                ),
                layout="row_major",
            )
            scale_header = compiled_safetensors_header(
                opportunity.candidate_scale_name,
                dtype="BF16",
                shape=list(opportunity.scale_shape),
                byte_count=(
                    opportunity.output_features
                    * (opportunity.input_features // opportunity.group_size)
                    * 2
                ),
                layout="row_major",
            )
            context.write_artifact_streams(
                (weight_path, scale_path),
                _quantized_chunks(
                    context=context,
                    opportunity=opportunity,
                    weight_header=weight_header,
                    scale_header=scale_header,
                    state=state,
                ),
            )
            weight_bytes = (
                opportunity.output_features * opportunity.input_features // 2
            )
            scale_bytes = (
                opportunity.output_features
                * (opportunity.input_features // opportunity.group_size)
                * 2
            )
            tensors[opportunity.candidate_weight_name] = _tensor_metadata(
                dtype="I32",
                shape=list(opportunity.packed_shape),
                logical_shape=[
                    opportunity.output_features,
                    opportunity.input_features,
                ],
                source_file=weight_path,
                payload_byte_count=weight_bytes,
                payload_sha256=state.weight_digest.hexdigest(),
                quantization={
                    "format": "compressed_tensors_pack_quantized",
                    "bits": 4,
                    "group_size": opportunity.group_size,
                    "symmetric": True,
                    "signed_offset": 8,
                    "scales": opportunity.candidate_scale_name,
                },
            )
            tensors[opportunity.candidate_scale_name] = _tensor_metadata(
                dtype="BF16",
                shape=list(opportunity.scale_shape),
                source_file=scale_path,
                payload_byte_count=scale_bytes,
                payload_sha256=state.scale_digest.hexdigest(),
            )
            normalized_rms_error = (
                state.squared_error / max(state.squared_source, 1e-30)
            ) ** 0.5
            reports.append(
                {
                    "component_id": opportunity.component_id,
                    "node_id": opportunity.node_id,
                    "scope_id": opportunity.scope_id,
                    "source": {
                        "tensor": source.tensor_name,
                        "dtype": source.metadata["dtype"],
                        "shape": source.metadata["shape"],
                        "data_sha256": source.metadata["data_sha256"],
                    },
                    "candidate": {
                        "weight_tensor": opportunity.candidate_weight_name,
                        "weight_dtype": "SIGNED_INT4_PACKED_I32",
                        "weight_data_sha256": state.weight_digest.hexdigest(),
                        "scale_tensor": opportunity.candidate_scale_name,
                        "scale_dtype": "BF16",
                        "scale_data_sha256": state.scale_digest.hexdigest(),
                        "group_size": opportunity.group_size,
                    },
                    "reconstruction": {
                        "element_count": state.element_count,
                        "normalized_rms_error": normalized_rms_error,
                        "maximum_absolute_error": state.maximum_absolute_error,
                        "finite": True,
                    },
                }
            )
            total_squared_source += state.squared_source
            total_squared_error += state.squared_error
            maximum_absolute_error = max(
                maximum_absolute_error,
                state.maximum_absolute_error,
            )
            total_elements += state.element_count
        context.write_json_artifact(
            TENSOR_FRAGMENT_PATH,
            {"schema": "nerve.tensor_index.v1", "tensors": tensors},
        )
        context.write_json_artifact(
            REPORT_PATH,
            {
                "schema": QUANTIZATION_REPORT_SCHEMA,
                "candidate_id": lowering["candidate_id"],
                "regions": reports,
                "aggregate": {
                    "element_count": total_elements,
                    "normalized_rms_error": (
                        total_squared_error / max(total_squared_source, 1e-30)
                    )
                    ** 0.5,
                    "maximum_absolute_error": maximum_absolute_error,
                    "finite": True,
                },
                "correction": {
                    "policy": "reject_candidate",
                    "fallback": "source_implementation",
                },
            },
        )
        representative = opportunities[0]
        context.write_json_artifact(
            COMPONENT_FIXTURE_PATH,
            component_fixture(
                component_id=representative.component_id,
                node_id=representative.node_id,
                input_features=representative.input_features,
                output_features=representative.output_features,
            ),
        )
        context.write_json_artifact(CONVERSATION_FIXTURE_PATH, conversation_fixture())
        context.write_json_artifact(
            PRODUCT_CONVERSATION_FIXTURE_PATH,
            product_conversation_fixture(),
        )
        context.write_json_artifact(
            MODEL_LIMITS_PATH,
            model_limits_fixture(representative.max_context_activations),
        )
        context.account_transient_bytes(0)


class GroupScaledInt4OrdinaryRelowerer:
    def run_ordinary_lowering(
        self,
        context: CandidateConstructionContext,
    ) -> None:
        _lowering(context)
        context.account_transient_bytes(0)


class GroupScaledInt4PhysicalOptimizer:
    def optimize_physical_artifacts(
        self,
        context: CandidateConstructionContext,
    ) -> None:
        lowering = _lowering(context)
        opportunities = opportunities_from_lowering(lowering)
        required_documents = {
            opportunity.manifest_ref for opportunity in opportunities
        } | {opportunity.tensor_index_ref for opportunity in opportunities}
        documents = {}
        declared_paths = sorted(
            {
                path
                for opportunity in opportunities
                for path in opportunity.source_artifact_refs
            }
        )
        for path in declared_paths:
            payload = context.read_source_artifact(path)
            if path in required_documents:
                documents[path] = _json_object(payload, path)
        prepared = [
            prepare_group_scaled_int4_component_from_documents(
                opportunity=opportunity,
                manifest=documents[opportunity.manifest_ref],
                tensor_index=documents[opportunity.tensor_index_ref],
            )
            for opportunity in opportunities
        ]
        expected_shaders = tuple(
            (record["artifact_path"], record["template_name"])
            for record in lowering["shader_artifacts"]
        )
        for region in prepared:
            actual = tuple(
                (shader.artifact_path, shader.template_name)
                for shader in region.shader_artifacts
            )
            if actual != expected_shaders:
                raise ModelCompileError(
                    "group-scaled INT4 performance-equivalent regions lower "
                    "to different shaders"
                )
        artifact_payloads = {}
        for artifact_path, template_name in expected_shaders:
            source = render_shader_source(_SHADER_ROOT, template_name)
            payload = compile_spirv(source, template_name)
            artifact_payloads[artifact_path] = payload
            context.write_artifact(artifact_path, payload)
        for opportunity, region in zip(opportunities, prepared, strict=True):
            replacement_kernel = finalize_group_scaled_int4_kernel(
                region.replacement_kernel,
                prepared=region,
                artifact_payloads=artifact_payloads,
            )
            overlay = {
                "schema": "nerve.optimizer.vulkan_component_region_overlay.v2",
                "source_component_id": opportunity.component_id,
                "source": {
                    "nodes": [region.source_node],
                    "kernels": [region.source_kernel],
                    "parameter_refs": region.source_parameter_refs,
                },
                "replacement": {
                    "nodes": [region.replacement_node],
                    "kernels": [replacement_kernel],
                    "parameter_refs": region.replacement_parameter_refs,
                },
            }
            expected_source_digest = contract_digest(
                {
                    "nodes": [region.source_node],
                    "kernels": [region.source_kernel],
                    "parameter_refs": region.source_parameter_refs,
                }
            )
            if contract_digest(overlay["source"]) != expected_source_digest:
                raise ModelCompileError(
                    "group-scaled INT4 source overlay changed during construction"
                )
            context.write_json_artifact(
                str(_region(lowering, opportunity)["candidate"]["overlay_path"]),
                overlay,
            )
        context.account_transient_bytes(0)


class _QuantizationState:
    def __init__(self) -> None:
        self.weight_digest = sha256()
        self.scale_digest = sha256()
        self.squared_source = 0.0
        self.squared_error = 0.0
        self.maximum_absolute_error = 0.0
        self.element_count = 0


def _quantized_chunks(
    *,
    context: CandidateConstructionContext,
    opportunity: GroupScaledInt4Opportunity,
    weight_header: bytes,
    scale_header: bytes,
    state: _QuantizationState,
):
    yield (
        struct.pack("<Q", len(weight_header)) + weight_header,
        struct.pack("<Q", len(scale_header)) + scale_header,
    )
    source = opportunity.source_weight
    row_bytes = opportunity.input_features * 2
    groups = opportunity.input_features // opportunity.group_size
    shifts = (np.arange(8, dtype=np.uint32) * np.uint32(4)).reshape(1, 1, 8)
    for row_start in range(0, opportunity.output_features, _CONSTRUCTION_ROW_BATCH):
        row_count = min(
            _CONSTRUCTION_ROW_BATCH,
            opportunity.output_features - row_start,
        )
        payload = context.read_source_artifact_region(
            source.storage.path,
            source.payload_byte_offset + row_start * row_bytes,
            row_count * row_bytes,
        )
        values = bf16_bytes_to_f32_matrix(
            payload,
            row_count,
            opportunity.input_features,
        )
        if not np.all(np.isfinite(values)):
            raise ModelCompileError(
                f"group-scaled INT4 source tensor {source.tensor_name!r} is not finite"
            )
        blocks = values.reshape(row_count, groups, opportunity.group_size)
        maximum = np.max(np.abs(blocks), axis=2)
        scales = np.where(maximum > 0.0, maximum / 7.0, 1.0)
        scale_payload = f32_to_bf16_bytes(scales.reshape(-1))
        stored_scales = bf16_bytes_to_f32(scale_payload, list(scales.shape))
        if not np.all(np.isfinite(stored_scales)) or np.any(stored_scales <= 0.0):
            raise ModelCompileError(
                f"group-scaled INT4 scales for {source.tensor_name!r} are invalid"
            )
        quantized = np.clip(
            np.rint(blocks / stored_scales[:, :, None]),
            -7,
            7,
        ).astype(np.int8)
        encoded = (quantized.astype(np.int16) + 8).astype(np.uint32)
        packed = np.bitwise_or.reduce(
            encoded.reshape(row_count, opportunity.input_features // 8, 8)
            << shifts,
            axis=2,
        ).astype("<u4", copy=False)
        weight_payload = packed.tobytes(order="C")
        reconstructed = (
            quantized.astype(np.float32) * stored_scales[:, :, None]
        ).reshape(row_count, opportunity.input_features)
        difference = reconstructed - values
        state.squared_source += float(np.sum(values * values, dtype=np.float64))
        state.squared_error += float(
            np.sum(difference * difference, dtype=np.float64)
        )
        state.maximum_absolute_error = max(
            state.maximum_absolute_error,
            float(np.max(np.abs(difference))),
        )
        state.element_count += int(values.size)
        state.weight_digest.update(weight_payload)
        state.scale_digest.update(scale_payload)
        context.account_transient_bytes(
            len(payload)
            + int(values.nbytes)
            + int(blocks.nbytes)
            + int(quantized.nbytes)
            + int(encoded.nbytes)
            + int(packed.nbytes)
            + len(scale_payload)
        )
        yield (weight_payload, scale_payload)
    context.account_transient_bytes(0)


def opportunities_from_lowering(
    lowering: Json,
) -> tuple[GroupScaledInt4Opportunity, ...]:
    records = lowering.get("regions")
    if not isinstance(records, list) or not records:
        raise ModelCompileError("group-scaled INT4 lowering has no component regions")
    opportunities = tuple(_opportunity(record) for record in records)
    identities = [(item.component_id, item.node_id) for item in opportunities]
    if identities != sorted(set(identities)):
        raise ModelCompileError(
            "group-scaled INT4 lowering regions must be sorted and unique"
        )
    signatures = {item.performance_signature for item in opportunities}
    if len(signatures) != 1:
        raise ModelCompileError(
            "group-scaled INT4 lowering crosses performance-equivalence classes"
        )
    return opportunities


def _opportunity(record: Json) -> GroupScaledInt4Opportunity:
    source = record["source_weight"]
    storage = source["storage"]
    tensor_index = source["tensor_index"]
    geometry = record["geometry"]
    return GroupScaledInt4Opportunity(
        scope_id=str(record["scope_id"]),
        source_contract_digest=str(record["source_contract_digest"]),
        component_id=str(record["component_id"]),
        node_id=str(record["node_id"]),
        evidence_ids=tuple(record["evidence_ids"]),  # type: ignore[arg-type]
        source_artifact_refs=tuple(
            record["source_artifact_refs"]  # type: ignore[arg-type]
        ),
        manifest_ref=str(record["manifest_ref"]),
        circuit_ref=str(record["circuit_ref"]),
        tensor_index_ref=str(record["tensor_index_ref"]),
        source_weight_ref_id=str(record["source_weight_ref_id"]),
        source_weight_ref=deepcopy(record["source_weight_ref"]),
        source_weight=SourceTensorArtifact(
            tensor_name=str(source["name"]),
            _metadata=deepcopy(source["metadata"]),
            tensor_index=SourceArtifact(
                path=str(tensor_index["path"]),
                digest=str(tensor_index["digest"]),
                byte_count=int(tensor_index["byte_count"]),
            ),
            storage=SourceArtifact(
                path=str(storage["path"]),
                digest=str(storage["digest"]),
                byte_count=int(storage["byte_count"]),
            ),
            safetensors_header_bytes=int(source["safetensors_header_bytes"]),
            payload_byte_offset=int(source["payload_byte_offset"]),
            payload_byte_count=int(source["payload_byte_count"]),
        ),
        input_features=int(geometry["input_features"]),
        output_features=int(geometry["output_features"]),
        group_size=int(geometry["group_size"]),
        max_context_activations=int(record["max_context_activations"]),
        compiler_device=deepcopy(record["compiler_device"]),
        performance_signature=str(record["performance_signature"]),
    )


def _region(lowering: Json, opportunity: GroupScaledInt4Opportunity) -> Json:
    matches = [
        record
        for record in lowering["regions"]
        if record.get("component_id") == opportunity.component_id
        and record.get("node_id") == opportunity.node_id
    ]
    if len(matches) != 1:
        raise ModelCompileError(
            "group-scaled INT4 lowering has no unique component region"
        )
    return matches[0]


def _tensor_metadata(
    *,
    dtype: str,
    shape: list[int],
    source_file: str,
    payload_byte_count: int,
    payload_sha256: str,
    logical_shape: list[int] | None = None,
    quantization: Json | None = None,
) -> Json:
    metadata = {
        "dtype": dtype,
        "shape": shape,
        "parameter_count": int(np.prod(shape, dtype=np.int64)),
        "byte_count": payload_byte_count,
        "data_offsets": [0, payload_byte_count],
        "source_file": source_file,
        "data_sha256": payload_sha256,
        "layout": "row_major",
    }
    if logical_shape is not None:
        metadata["logical_shape"] = logical_shape
    if quantization is not None:
        metadata["quantization"] = quantization
    return metadata


def _lowering(context: CandidateConstructionContext) -> Json:
    lowering = context.target_lowering
    if lowering.get("schema") != TARGET_LOWERING_SCHEMA:
        raise ModelCompileError(
            "group-scaled INT4 toolchain received incompatible lowering"
        )
    if lowering.get("candidate_id") != context.candidate["candidate_id"]:
        raise ModelCompileError(
            "group-scaled INT4 lowering belongs to another candidate"
        )
    return lowering


def _json_object(payload: bytes, label: str) -> Json:
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelCompileError(f"{label} is not valid JSON") from error
    if not isinstance(document, dict):
        raise ModelCompileError(f"{label} must be a JSON object")
    return document


__all__ = [
    "GroupScaledInt4ToolchainResolver",
    "opportunities_from_lowering",
]
