from __future__ import annotations

import json
import struct
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import numpy as np

from nerve.compilation import Json, ModelCompileError
from nerve.model_package_common import FP8_OUTPUT_PROJECTION_TILE_ROWS
from nerve.model_package_shader_templates import (
    render_shader_source,
    render_shader_template,
)
from nerve.model_package_tensors import (
    bf16_bytes_to_f32,
    bf16_bytes_to_f32_matrix,
    compiled_safetensors_header,
    e4m3fn_to_f32,
    f32_to_bf16_bytes,
    f32_to_e4m3fn,
)
from nerve.representation_optimizer.automation.target import CandidateToolchain
from nerve.representation_optimizer.providers.codebook.shaders import (
    compile_spirv,
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
    component_fixture,
    conversation_fixture,
    model_limits_fixture,
    product_conversation_fixture,
)
from nerve.representation_optimizer.providers.output_fp8.contracts import (
    QUANTIZATION_REPORT_SCHEMA,
    TARGET_LOWERING_SCHEMA,
)
from nerve.representation_optimizer.providers.types import ProviderCandidatePlan
from nerve.representation_optimizer.staging.workspace import (
    CandidateConstructionContext,
)


_SHADER_ROOT = Path(__file__).resolve().parents[4] / "runtime-rs" / "shaders"
_CONSTRUCTION_ROW_BATCH = 256
_DRAFT_INT4_GROUP_COLUMNS = 128


class BlockScaledOutputToolchainResolver:
    def resolve(self, plan: ProviderCandidatePlan) -> CandidateToolchain:
        if (
            plan.provider.provider_id != "nerve.block_scaled_output_projection"
            or plan.target_lowering.get("schema") != TARGET_LOWERING_SCHEMA
        ):
            raise ModelCompileError(
                "block-scaled output toolchain cannot construct provider "
                f"{plan.provider.provider_id!r}"
            )
        return CandidateToolchain(
            semantic_constructor=BlockScaledOutputSemanticConstructor(),
            ordinary_relowerer=BlockScaledOutputOrdinaryRelowerer(),
            physical_optimizer=BlockScaledOutputPhysicalOptimizer(),
        )


class BlockScaledOutputSemanticConstructor:
    def construct_semantic_artifacts(
        self,
        context: CandidateConstructionContext,
    ) -> None:
        lowering = _lowering(context)
        source = lowering["source"]["projection"]
        geometry = lowering["geometry"]
        parameters = lowering["parameters"]
        tensor_index = _json_object(
            context.read_source_artifact("tensors.json"),
            "tensors.json",
        )
        if tensor_index.get("tensors", {}).get(source["name"]) != source["metadata"]:
            raise ModelCompileError(
                "block-scaled output source tensor metadata drifted"
            )
        rows = int(geometry["vocabulary_size"])
        columns = int(geometry["hidden_size"])
        block_rows = int(geometry["block_rows"])
        block_columns = int(geometry["block_columns"])
        if (
            block_rows != 16
            or block_columns != 128
            or columns % block_columns
            or source["payload_byte_count"] != rows * columns * 2
        ):
            raise ModelCompileError(
                "block-scaled output lowering has unsupported geometry"
            )
        scale_shape = [int(value) for value in geometry["scale_shape"]]
        state = _QuantizationState(
            scale_payload=bytearray(),
            weight_digest=sha256(),
            squared_source=0.0,
            squared_error=0.0,
            maximum_absolute_error=0.0,
            element_count=0,
        )
        weight_header = compiled_safetensors_header(
            parameters["weight_tensor_name"],
            dtype="F8_E4M3",
            shape=[rows, columns],
            byte_count=rows * columns,
            layout="row_major",
        )
        draft_enabled = bool(lowering["runtime"]["role_specialized_draft"])
        draft_state = (
            _QuantizationState(
                scale_payload=bytearray(),
                weight_digest=sha256(),
                squared_source=0.0,
                squared_error=0.0,
                maximum_absolute_error=0.0,
                element_count=0,
            )
            if draft_enabled
            else None
        )
        draft_weight_header = (
            compiled_safetensors_header(
                parameters["draft_weight_tensor_name"],
                dtype="U8",
                shape=[rows, columns // 2],
                byte_count=rows * columns // 2,
                layout="packed_signed_int4_row_major",
            )
            if draft_enabled
            else None
        )
        output_paths = (
            (WEIGHT_PATH, DRAFT_WEIGHT_PATH) if draft_enabled else (WEIGHT_PATH,)
        )
        context.write_artifact_streams(
            output_paths,
            _quantized_weight_chunks(
                context=context,
                source=source,
                rows=rows,
                columns=columns,
                block_rows=block_rows,
                block_columns=block_columns,
                header=weight_header,
                state=state,
                draft_header=draft_weight_header,
                draft_state=draft_state,
            ),
        )
        expected_scale_bytes = scale_shape[0] * scale_shape[1] * 2
        if len(state.scale_payload) != expected_scale_bytes:
            raise ModelCompileError(
                "block-scaled output construction emitted the wrong scale payload size"
            )
        scale_payload = bytes(state.scale_payload)
        scale_header = compiled_safetensors_header(
            parameters["scale_tensor_name"],
            dtype="BF16",
            shape=scale_shape,
            byte_count=expected_scale_bytes,
            layout="row_major",
        )
        context.write_artifact(
            SCALE_PATH,
            struct.pack("<Q", len(scale_header)) + scale_header + scale_payload,
        )
        tensors = {
            parameters["scale_tensor_name"]: _tensor_metadata(
                dtype="BF16",
                shape=scale_shape,
                source_file=SCALE_PATH,
                payload=scale_payload,
            ),
            parameters["weight_tensor_name"]: _tensor_metadata(
                dtype="F8_E4M3",
                shape=[rows, columns],
                source_file=WEIGHT_PATH,
                payload_byte_count=rows * columns,
                payload_sha256=state.weight_digest.hexdigest(),
            ),
        }
        if draft_state is not None:
            draft_scale_shape = [int(value) for value in geometry["draft_scale_shape"]]
            expected_draft_scale_bytes = draft_scale_shape[0] * draft_scale_shape[1] * 2
            if len(draft_state.scale_payload) != expected_draft_scale_bytes:
                raise ModelCompileError(
                    "role-specialized draft output construction emitted the "
                    "wrong scale payload size"
                )
            draft_scale_payload = bytes(draft_state.scale_payload)
            draft_scale_header = compiled_safetensors_header(
                parameters["draft_scale_tensor_name"],
                dtype="BF16",
                shape=draft_scale_shape,
                byte_count=expected_draft_scale_bytes,
                layout="row_major",
            )
            context.write_artifact(
                DRAFT_SCALE_PATH,
                struct.pack("<Q", len(draft_scale_header))
                + draft_scale_header
                + draft_scale_payload,
            )
            tensors[parameters["draft_scale_tensor_name"]] = _tensor_metadata(
                dtype="BF16",
                shape=draft_scale_shape,
                source_file=DRAFT_SCALE_PATH,
                payload=draft_scale_payload,
            )
            tensors[parameters["draft_weight_tensor_name"]] = _tensor_metadata(
                dtype="U8",
                shape=[rows, columns // 2],
                logical_shape=[rows, columns],
                layout="packed_signed_int4_row_major",
                source_file=DRAFT_WEIGHT_PATH,
                payload_byte_count=rows * columns // 2,
                payload_sha256=draft_state.weight_digest.hexdigest(),
            )
        context.write_json_artifact(
            TENSOR_FRAGMENT_PATH,
            {
                "schema": "nerve.tensor_index.v1",
                "tensors": tensors,
            },
        )
        context.write_json_artifact(
            COMPONENT_FIXTURE_PATH,
            component_fixture(
                component_id=lowering["source"]["component_id"],
                physical_node_id=lowering["source"]["physical_node_id"],
                hidden_size=columns,
                vocabulary_size=rows,
            ),
        )
        context.write_json_artifact(
            CONVERSATION_FIXTURE_PATH,
            conversation_fixture(),
        )
        context.write_json_artifact(
            PRODUCT_CONVERSATION_FIXTURE_PATH,
            product_conversation_fixture(),
        )
        context.write_json_artifact(
            MODEL_LIMITS_PATH,
            model_limits_fixture(int(lowering["runtime"]["max_context_activations"])),
        )
        normalized_rms_error = (
            state.squared_error / max(state.squared_source, 1e-30)
        ) ** 0.5
        role_specializations = []
        if draft_state is not None:
            role_specializations.append(
                {
                    "role": "speculative_draft",
                    "decoder_ids": list(lowering["runtime"]["speculative_decoder_ids"]),
                    "weight_tensor": parameters["draft_weight_tensor_name"],
                    "weight_dtype": "SIGNED_INT4_PACKED_U8",
                    "weight_data_sha256": (draft_state.weight_digest.hexdigest()),
                    "scale_tensor": parameters["draft_scale_tensor_name"],
                    "scale_dtype": "BF16",
                    "scale_data_sha256": sha256(
                        bytes(draft_state.scale_payload)
                    ).hexdigest(),
                    "group_columns": _DRAFT_INT4_GROUP_COLUMNS,
                    "reconstruction": {
                        "element_count": draft_state.element_count,
                        "normalized_rms_error": (
                            draft_state.squared_error
                            / max(draft_state.squared_source, 1e-30)
                        )
                        ** 0.5,
                        "maximum_absolute_error": (draft_state.maximum_absolute_error),
                        "finite": True,
                    },
                    "correctness_boundary": ("target_model_verification"),
                }
            )
        context.write_json_artifact(
            ERROR_REPORT_PATH,
            {
                "schema": QUANTIZATION_REPORT_SCHEMA,
                "candidate_id": lowering["candidate_id"],
                "scope_id": lowering["scope_id"],
                "source": {
                    "tensor": source["name"],
                    "dtype": source["metadata"]["dtype"],
                    "shape": source["metadata"]["shape"],
                    "data_sha256": source["metadata"]["data_sha256"],
                },
                "candidate": {
                    "weight_tensor": parameters["weight_tensor_name"],
                    "weight_dtype": "F8_E4M3",
                    "weight_data_sha256": (state.weight_digest.hexdigest()),
                    "scale_tensor": parameters["scale_tensor_name"],
                    "scale_dtype": "BF16",
                    "scale_data_sha256": sha256(scale_payload).hexdigest(),
                    "block_shape": [block_rows, block_columns],
                },
                "reconstruction": {
                    "element_count": state.element_count,
                    "normalized_rms_error": normalized_rms_error,
                    "maximum_absolute_error": (state.maximum_absolute_error),
                    "finite": True,
                },
                "correction": {
                    "policy": "reject_candidate",
                    "fallback": "source_implementation",
                },
                "role_specializations": role_specializations,
            },
        )
        context.account_transient_bytes(0)


class BlockScaledOutputOrdinaryRelowerer:
    def run_ordinary_lowering(
        self,
        context: CandidateConstructionContext,
    ) -> None:
        lowering = _lowering(context)
        source = lowering["source"]
        manifest = _json_object(
            context.read_source_artifact(source["manifest_ref"]),
            source["manifest_ref"],
        )
        exact_documents = {
            path: _json_object(
                context.read_source_artifact(path),
                path,
            )
            for path in source["artifact_refs"]
        }
        circuit = exact_documents.get(source["circuit_ref"])
        if (
            circuit is None
            or circuit.get("source", {}).get("component_id") != source["component_id"]
        ):
            raise ModelCompileError(
                "block-scaled output exact circuit identity drifted"
            )
        component = deepcopy(
            _unique_record(
                manifest["circuit_graph"]["components"],
                "component_id",
                source["component_id"],
                "resident output component",
            )
        )
        if component.get("runtime_role") != "output_transducer":
            raise ModelCompileError(
                "block-scaled candidate target is no longer an output transducer"
            )
        parameters = lowering["parameters"]
        source_projection_ref_id = source["projection_parameter_ref_id"]
        projection_ref_id = f"{source_projection_ref_id}.fp8_e4m3"
        scale_ref_id = source["projection_scale_parameter_ref_id"]
        replacement_refs = {
            projection_ref_id: {
                "tensor": parameters["weight_tensor_name"],
                "role": (
                    f"{source['component_id']}.{source['physical_node_id']}.weight"
                ),
            },
            scale_ref_id: {
                "tensor": parameters["scale_tensor_name"],
                "role": (
                    f"{source['component_id']}."
                    f"{source['physical_node_id']}.weight_scale_inv"
                ),
            },
        }
        circuit_refs = component["circuit"]["parameters"]["refs"]
        component_refs = component["params"]["refs"]
        if (
            circuit_refs.get(source_projection_ref_id, {}).get("tensor")
            != source["projection"]["name"]
            or component_refs.get(source_projection_ref_id, {}).get("tensor")
            != source["projection"]["name"]
        ):
            raise ModelCompileError("block-scaled output projection binding drifted")
        circuit_refs.pop(source_projection_ref_id)
        component_refs.pop(source_projection_ref_id)
        circuit_refs.update(deepcopy(replacement_refs))
        component_refs.update(deepcopy(replacement_refs))
        projection_node = _unique_record(
            component["circuit"]["nodes"],
            "id",
            source["physical_node_id"],
            "output projection node",
        )
        projection_node["params"] = [
            projection_ref_id,
            scale_ref_id,
        ]
        projection_node["attrs"] = {
            **projection_node["attrs"],
            "parameter_representation": {
                "kind": "fp8_e4m3_with_bf16_inverse_scales",
                "block_rows": lowering["geometry"]["block_rows"],
                "block_columns": lowering["geometry"]["block_columns"],
                "source_tensor": source["projection"]["name"],
                "source_parameter_ids": [source_projection_ref_id],
                "descriptor_abi": "source_parameters_replaced",
                "alternative_execution_phases": [
                    "decode",
                    "prefill",
                ],
                "source_retained_execution_phases": [],
            },
        }
        implementation = "optimized_block_scaled_fp8_output_projection_v1"
        component["implementation"] = implementation
        component["circuit"]["implementation"] = implementation
        component["circuit"]["behavioral_error_contract"] = {
            "mode": "validated_approximation",
            "candidate_id": lowering["candidate_id"],
            "error_report": context.artifact_reference(ERROR_REPORT_PATH),
            "fallback": "source_implementation",
        }

        output = deepcopy(manifest["output_transducer"])
        _lower_output_spec(
            output["spec"],
            lowering,
        )
        output["projection_shader_path"] = context.artifact_reference(
            DECODE_SHADER_PATH
        )
        output["projection_batch_shader_path"] = context.artifact_reference(
            BATCH_SHADER_PATH
        )
        output["projection_batch_lane_tile_width"] = int(
            lowering["runtime"]["batch_lane_tile_width"]
        )

        draft_outputs = []
        role_specialized_decoders = set(lowering["runtime"]["speculative_decoder_ids"])
        for decoder in sorted(
            manifest.get("speculative_decoders", []),
            key=lambda item: item["id"],
        ):
            draft = deepcopy(decoder["output_transducer"])
            if draft.get("projection_parameter_tensor") == source["projection"]["name"]:
                if decoder["id"] in role_specialized_decoders:
                    _lower_draft_output_spec(draft, lowering)
                    draft["projection_shader_path"] = context.artifact_reference(
                        DRAFT_DECODE_SHADER_PATH
                    )
                else:
                    _lower_output_spec(draft, lowering)
                    draft["projection_shader_path"] = context.artifact_reference(
                        DECODE_SHADER_PATH
                    )
            draft_outputs.append(
                {
                    "decoder_id": decoder["id"],
                    "output_transducer": draft,
                }
            )
        context.write_json_artifact(
            OVERLAY_PATH,
            {
                "schema": ("nerve.optimizer.vulkan_output_transducer_overlay.v1"),
                "source_component_id": source["component_id"],
                "component": component,
                "output_transducer": output,
                "speculative_output_transducers": draft_outputs,
            },
        )
        context.account_transient_bytes(0)


class BlockScaledOutputPhysicalOptimizer:
    def optimize_physical_artifacts(
        self,
        context: CandidateConstructionContext,
    ) -> None:
        lowering = _lowering(context)
        geometry = lowering["geometry"]
        scale = lowering["runtime"]["output_scale_token"]
        batch_width = int(lowering["runtime"]["batch_lane_tile_width"])
        stem = (
            f"fp8_e4m3_b{geometry['block_rows']}x"
            f"{geometry['block_columns']}_"
            f"{geometry['vocabulary_size']}x{geometry['hidden_size']}_"
            f"scale{scale}_to_f32"
        )
        decode_name = f"tied_output_projection_{stem}.comp"
        batch_name = f"tied_output_projection_batch{batch_width}_{stem}.comp"
        decode_source = render_shader_source(_SHADER_ROOT, decode_name)
        batch_source = render_shader_source(_SHADER_ROOT, batch_name)
        decode_spirv = compile_spirv(
            decode_source,
            "block_scaled_output_decode.comp",
        )
        batch_spirv = compile_spirv(
            batch_source,
            "block_scaled_output_batch.comp",
        )
        draft_source = None
        draft_spirv = None
        if lowering["runtime"]["role_specialized_draft"]:
            draft_source = render_shader_template(
                _SHADER_ROOT,
                "tied_output_projection_int4_draft.comp.template",
                {
                    "GROUP_COLUMNS": str(_DRAFT_INT4_GROUP_COLUMNS),
                    "INPUT_SIZE": str(geometry["hidden_size"]),
                    "VOCAB_SIZE": str(geometry["vocabulary_size"]),
                    "OUTPUT_TILE_ROWS": str(FP8_OUTPUT_PROJECTION_TILE_ROWS),
                    "OUTPUT_SCALE": str(scale),
                },
            )
            draft_spirv = compile_spirv(
                draft_source,
                "role_specialized_draft_output.comp",
            )
        context.account_transient_bytes(
            len(decode_source.encode("utf-8"))
            + len(batch_source.encode("utf-8"))
            + len(decode_spirv)
            + len(batch_spirv)
            + (
                len(draft_source.encode("utf-8")) + len(draft_spirv)
                if draft_source is not None and draft_spirv is not None
                else 0
            )
        )
        context.write_artifact(DECODE_SHADER_PATH, decode_spirv)
        context.write_artifact(BATCH_SHADER_PATH, batch_spirv)
        if draft_spirv is not None:
            context.write_artifact(
                DRAFT_DECODE_SHADER_PATH,
                draft_spirv,
            )
        context.account_transient_bytes(0)


class _QuantizationState:
    def __init__(
        self,
        *,
        scale_payload: bytearray,
        weight_digest,
        squared_source: float,
        squared_error: float,
        maximum_absolute_error: float,
        element_count: int,
    ) -> None:
        self.scale_payload = scale_payload
        self.weight_digest = weight_digest
        self.squared_source = squared_source
        self.squared_error = squared_error
        self.maximum_absolute_error = maximum_absolute_error
        self.element_count = element_count


def _quantized_weight_chunks(
    *,
    context: CandidateConstructionContext,
    source: Json,
    rows: int,
    columns: int,
    block_rows: int,
    block_columns: int,
    header: bytes,
    state: _QuantizationState,
    draft_header: bytes | None,
    draft_state: _QuantizationState | None,
):
    if (draft_header is None) != (draft_state is None):
        raise ModelCompileError(
            "role-specialized draft output header and state must agree"
        )
    prefix = struct.pack("<Q", len(header)) + header
    if draft_header is None:
        yield (prefix,)
    else:
        yield (
            prefix,
            struct.pack("<Q", len(draft_header)) + draft_header,
        )
    storage_path = source["storage"]["path"]
    source_start = int(source["payload_byte_offset"])
    row_bytes = columns * 2
    scale_columns = columns // block_columns
    for group_start in range(0, rows, _CONSTRUCTION_ROW_BATCH):
        group_rows = min(_CONSTRUCTION_ROW_BATCH, rows - group_start)
        payload = context.read_source_artifact_region(
            storage_path,
            source_start + group_start * row_bytes,
            group_rows * row_bytes,
        )
        values = bf16_bytes_to_f32_matrix(
            payload,
            group_rows,
            columns,
        )
        padded_rows = ((group_rows + block_rows - 1) // block_rows) * block_rows
        padded = np.zeros((padded_rows, columns), dtype=np.float32)
        padded[:group_rows] = values
        blocks = padded.reshape(
            padded_rows // block_rows,
            block_rows,
            scale_columns,
            block_columns,
        )
        block_max = np.max(np.abs(blocks), axis=(1, 3))
        scales = np.where(block_max > 0.0, block_max / 448.0, 1.0)
        scale_bytes = f32_to_bf16_bytes(scales.reshape(-1))
        state.scale_payload.extend(scale_bytes)
        stored_scales = bf16_bytes_to_f32(
            scale_bytes,
            list(scales.shape),
        )
        quantized_blocks = f32_to_e4m3fn(blocks / stored_scales[:, None, :, None])
        quantized_group = quantized_blocks.reshape(
            padded_rows,
            columns,
        )[:group_rows]
        reconstructed = (
            e4m3fn_to_f32(quantized_blocks) * stored_scales[:, None, :, None]
        ).reshape(padded_rows, columns)[:group_rows]
        difference = reconstructed - values
        state.squared_source += float(np.sum(values * values, dtype=np.float64))
        state.squared_error += float(np.sum(difference * difference, dtype=np.float64))
        state.maximum_absolute_error = max(
            state.maximum_absolute_error,
            float(np.max(np.abs(difference))),
        )
        state.element_count += int(values.size)
        quantized_bytes = quantized_group.tobytes(order="C")
        state.weight_digest.update(quantized_bytes)
        draft_bytes = None
        if draft_state is not None:
            draft_blocks = values.reshape(
                group_rows,
                columns // _DRAFT_INT4_GROUP_COLUMNS,
                _DRAFT_INT4_GROUP_COLUMNS,
            )
            draft_max = np.max(np.abs(draft_blocks), axis=2)
            draft_scales = np.where(
                draft_max > 0.0,
                draft_max / 7.0,
                1.0,
            )
            draft_scale_bytes = f32_to_bf16_bytes(draft_scales.reshape(-1))
            draft_state.scale_payload.extend(draft_scale_bytes)
            stored_draft_scales = bf16_bytes_to_f32(
                draft_scale_bytes,
                list(draft_scales.shape),
            )
            quantized_draft = np.clip(
                np.rint(draft_blocks / stored_draft_scales[:, :, None]),
                -7,
                7,
            ).astype(np.int8)
            encoded_draft = (quantized_draft.astype(np.int16) + 8).astype(np.uint8)
            packed_draft = (
                encoded_draft[:, :, 0::2] | (encoded_draft[:, :, 1::2] << np.uint8(4))
            ).reshape(group_rows, columns // 2)
            draft_bytes = packed_draft.tobytes(order="C")
            draft_state.weight_digest.update(draft_bytes)
            draft_reconstructed = (
                quantized_draft.astype(np.float32) * stored_draft_scales[:, :, None]
            ).reshape(group_rows, columns)
            draft_difference = draft_reconstructed - values
            draft_state.squared_source += float(
                np.sum(values * values, dtype=np.float64)
            )
            draft_state.squared_error += float(
                np.sum(
                    draft_difference * draft_difference,
                    dtype=np.float64,
                )
            )
            draft_state.maximum_absolute_error = max(
                draft_state.maximum_absolute_error,
                float(np.max(np.abs(draft_difference))),
            )
            draft_state.element_count += int(values.size)
        context.account_transient_bytes(
            len(payload)
            + int(values.nbytes)
            + int(quantized_group.nbytes)
            + (len(draft_bytes) if draft_bytes is not None else 0)
        )
        if draft_bytes is None:
            yield (quantized_bytes,)
        else:
            yield (quantized_bytes, draft_bytes)
    context.account_transient_bytes(0)


def _lower_output_spec(spec: Json, lowering: Json) -> None:
    geometry = lowering["geometry"]
    parameters = lowering["parameters"]
    rows = int(geometry["vocabulary_size"])
    columns = int(geometry["hidden_size"])
    scale_shape = [int(value) for value in geometry["scale_shape"]]
    spec["projection_parameter_tensor"] = parameters["weight_tensor_name"]
    spec["projection_parameter_dtype"] = "F8_E4M3"
    spec["projection_parameter_byte_capacity"] = rows * columns
    spec["projection_scale_parameter_tensor"] = parameters["scale_tensor_name"]
    spec["projection_scale_parameter_dtype"] = "BF16"
    spec["projection_scale_parameter_shape"] = scale_shape
    spec["projection_scale_parameter_byte_capacity"] = (
        scale_shape[0] * scale_shape[1] * 2
    )
    spec["projection_workgroup_count_x"] = (rows + 31) // 32
    spec["projection_local_size_x"] = 1024


def _lower_draft_output_spec(spec: Json, lowering: Json) -> None:
    geometry = lowering["geometry"]
    parameters = lowering["parameters"]
    rows = int(geometry["vocabulary_size"])
    columns = int(geometry["hidden_size"])
    scale_shape = [int(value) for value in geometry["draft_scale_shape"]]
    spec["projection_parameter_tensor"] = parameters["draft_weight_tensor_name"]
    spec["projection_parameter_dtype"] = "U8"
    spec["projection_parameter_shape"] = [rows, columns // 2]
    spec["projection_parameter_byte_capacity"] = rows * columns // 2
    spec["projection_scale_parameter_tensor"] = parameters["draft_scale_tensor_name"]
    spec["projection_scale_parameter_dtype"] = "BF16"
    spec["projection_scale_parameter_shape"] = scale_shape
    spec["projection_scale_parameter_byte_capacity"] = (
        scale_shape[0] * scale_shape[1] * 2
    )
    spec["projection_workgroup_count_x"] = (
        rows + FP8_OUTPUT_PROJECTION_TILE_ROWS - 1
    ) // FP8_OUTPUT_PROJECTION_TILE_ROWS
    spec["projection_local_size_x"] = 1024


def _tensor_metadata(
    *,
    dtype: str,
    shape: list[int],
    source_file: str,
    payload: bytes | None = None,
    payload_byte_count: int | None = None,
    payload_sha256: str | None = None,
    logical_shape: list[int] | None = None,
    layout: str = "row_major",
) -> Json:
    if payload is not None:
        payload_byte_count = len(payload)
        payload_sha256 = sha256(payload).hexdigest()
    if payload_byte_count is None or payload_sha256 is None:
        raise ModelCompileError(
            "candidate tensor metadata requires payload size and digest"
        )
    parameter_count = int(np.prod(shape, dtype=np.int64))
    metadata = {
        "dtype": dtype,
        "shape": shape,
        "parameter_count": parameter_count,
        "byte_count": payload_byte_count,
        "data_offsets": [0, payload_byte_count],
        "source_file": source_file,
        "data_sha256": payload_sha256,
        "layout": layout,
    }
    if logical_shape is not None:
        metadata["logical_shape"] = logical_shape
    return metadata


def _lowering(context: CandidateConstructionContext) -> Json:
    lowering = context.target_lowering
    if lowering.get("schema") != TARGET_LOWERING_SCHEMA:
        raise ModelCompileError(
            "block-scaled output toolchain received incompatible lowering"
        )
    if lowering.get("candidate_id") != context.candidate["candidate_id"]:
        raise ModelCompileError(
            "block-scaled output lowering belongs to another candidate"
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


def _unique_record(
    records: list[Json],
    field: str,
    expected: str,
    label: str,
) -> Json:
    matches = [record for record in records if record.get(field) == expected]
    if len(matches) != 1:
        raise ModelCompileError(f"{label} {expected!r} is not unique")
    return matches[0]
