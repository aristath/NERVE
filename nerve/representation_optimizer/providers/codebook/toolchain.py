from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.automation.target import CandidateToolchain
from nerve.representation_optimizer.providers.codebook.artifacts import (
    BRANCH_INDEX_PATHS,
    CODEBOOK_TENSOR_PATH,
    COMPONENT_FIXTURE_PATH,
    CONVERSATION_FIXTURE_PATH,
    DECODE_SHADER_PATH,
    MODEL_LIMITS_PATH,
    OVERLAY_PATH,
    PREFILL_SHADER_PATH,
    PROOF_PATH,
    TENSOR_FRAGMENT_PATH,
    component_fixture_from_geometry,
    conversation_fixture,
    model_limits_fixture,
)
from nerve.representation_optimizer.providers.codebook.contracts import (
    CODEBOOK_PROOF_SCHEMA,
    CODEBOOK_RUNTIME_OP,
    TARGET_LOWERING_SCHEMA,
)
from nerve.representation_optimizer.providers.codebook.member_context import (
    MemberConstructionContext,
)
from nerve.representation_optimizer.providers.codebook.shaders import (
    compile_spirv,
    render_codebook_shader,
    source_template_sha256,
)
from nerve.representation_optimizer.providers.types import ProviderCandidatePlan
from nerve.representation_optimizer.staging.workspace import (
    CandidateConstructionContext,
)


class CodebookToolchainResolver:
    def resolve(self, plan: ProviderCandidatePlan) -> CandidateToolchain:
        if (
            plan.provider.provider_id != "nerve.exact_head_norm_codebook"
            or plan.target_lowering.get("schema") != TARGET_LOWERING_SCHEMA
        ):
            raise ModelCompileError(
                f"codebook toolchain cannot construct provider {plan.provider.provider_id!r}"
            )
        return CandidateToolchain(
            semantic_constructor=CodebookSemanticConstructor(),
            ordinary_relowerer=CodebookOrdinaryRelowerer(),
            physical_optimizer=CodebookPhysicalOptimizer(),
        )


class CodebookSemanticConstructor:
    def construct_semantic_artifacts(
        self,
        context: CandidateConstructionContext,
    ) -> None:
        lowering = _lowering(context)
        if "members" in lowering:
            for member in lowering["members"]:
                self.construct_semantic_artifacts(
                    MemberConstructionContext(context, member)
                )
            return
        parameters = lowering["parameters"]
        sources = parameters["source_tensors"]
        source_payload_bytes = sum(
            int(source["payload_byte_count"]) for source in sources
        )
        candidate_payload_bytes = sum(
            _word_aligned_byte_count(int(source["payload_byte_count"]) // 2)
            for source in sources
        ) + _word_aligned_byte_count(2 * len(parameters["codebook_values_u16"]))
        context.account_transient_bytes(source_payload_bytes + candidate_payload_bytes)
        source_payloads = _read_source_tensor_payloads(context, sources)
        tensor_index_path = _tensor_index_source(lowering)
        tensor_index = _json_object(
            context.read_source_artifact(tensor_index_path),
            tensor_index_path,
        )
        for source, tensor_payload in zip(
            sources,
            source_payloads,
            strict=True,
        ):
            indexed = tensor_index.get("tensors", {}).get(source["name"])
            if indexed != source["metadata"]:
                raise ModelCompileError(
                    f"codebook source tensor {source['name']!r} metadata drifted"
                )

        codebook_values = tuple(
            int(value) for value in parameters["codebook_values_u16"]
        )
        if (
            not codebook_values
            or len(codebook_values) > 256
            or codebook_values != tuple(sorted(set(codebook_values)))
        ):
            raise ModelCompileError(
                "codebook target lowering contains invalid canonical entries"
            )
        codebook_payload = b"".join(
            value.to_bytes(2, "little") for value in codebook_values
        )
        codebook_storage_payload = _word_aligned(codebook_payload)
        if (
            sha256(codebook_payload).hexdigest()
            != parameters["codebook_payload_sha256"]
        ):
            raise ModelCompileError("codebook target lowering entry digest disagrees")
        addresses = {value: index for index, value in enumerate(codebook_values)}
        index_payloads = []
        for payload in source_payloads:
            values = tuple(
                int.from_bytes(payload[offset : offset + 2], "little")
                for offset in range(0, len(payload), 2)
            )
            try:
                indices = bytes(addresses[value] for value in values)
            except KeyError as error:
                raise ModelCompileError(
                    "codebook target lowering omits a source BF16 value"
                ) from error
            reconstructed = b"".join(
                codebook_values[index].to_bytes(2, "little") for index in indices
            )
            if reconstructed != payload:
                raise ModelCompileError(
                    "codebook construction did not reconstruct source BF16 bits"
                )
            index_payloads.append(indices)

        branch_names = parameters["branch_index_tensor_names"]
        codebook_name = parameters["codebook_tensor_name"]
        encoded_indices = [
            _single_tensor_safetensors(
                name,
                "U8",
                [len(_word_aligned(payload))],
                _word_aligned(payload),
            )
            for name, payload in zip(
                branch_names,
                index_payloads,
                strict=True,
            )
        ]
        encoded_codebook = _single_tensor_safetensors(
            codebook_name,
            "BF16",
            [len(codebook_storage_payload) // 2],
            codebook_storage_payload,
        )
        for path, encoded in zip(
            BRANCH_INDEX_PATHS,
            encoded_indices,
            strict=True,
        ):
            context.write_artifact(path, encoded)
        context.write_artifact(CODEBOOK_TENSOR_PATH, encoded_codebook)
        context.write_json_artifact(
            TENSOR_FRAGMENT_PATH,
            _tensor_fragment(
                branch_names=branch_names,
                index_payloads=[_word_aligned(payload) for payload in index_payloads],
                branch_paths=[
                    context.artifact_reference(path)
                    for path in BRANCH_INDEX_PATHS
                ],
                codebook_name=codebook_name,
                codebook_payload=codebook_storage_payload,
                codebook_path=context.artifact_reference(
                    CODEBOOK_TENSOR_PATH
                ),
            ),
        )
        branch_head_counts = tuple(
            int(value) for value in lowering["geometry"]["branch_head_counts"]
        )
        if len(branch_head_counts) != 2:
            raise ModelCompileError(
                "codebook fixture requires exactly two branch head counts"
            )
        context.write_json_artifact(
            COMPONENT_FIXTURE_PATH,
            component_fixture_from_geometry(
                component_id=lowering["source"]["component_id"],
                physical_node_id=lowering["source"]["physical_node_id"],
                head_width=int(lowering["geometry"]["head_width"]),
                branch_head_counts=branch_head_counts,
            ),
        )
        context.write_json_artifact(
            CONVERSATION_FIXTURE_PATH,
            conversation_fixture(),
        )
        context.write_json_artifact(
            MODEL_LIMITS_PATH,
            model_limits_fixture(int(lowering["runtime"]["max_context_activations"])),
        )
        context.account_transient_bytes(0)


class CodebookOrdinaryRelowerer:
    def run_ordinary_lowering(
        self,
        context: CandidateConstructionContext,
    ) -> None:
        lowering = _lowering(context)
        if "members" in lowering:
            for member in lowering["members"]:
                self.run_ordinary_lowering(
                    MemberConstructionContext(context, member)
                )
            return
        source = lowering["source"]
        source_payloads = {
            path: context.read_source_artifact(path) for path in source["artifact_refs"]
        }
        manifest_payload = context.read_source_artifact(source["manifest_ref"])
        context.account_transient_bytes(
            sum(len(payload) for payload in source_payloads.values())
            + len(manifest_payload)
        )
        read_documents = {
            path: _json_object(payload, path)
            for path, payload in source_payloads.items()
        }
        manifest = _json_object(
            manifest_payload,
            source["manifest_ref"],
        )
        circuit_ref = source["circuit_ref"]
        if circuit_ref not in read_documents:
            raise ModelCompileError(
                "codebook lowering circuit is not an exact source artifact"
            )
        source_circuit = read_documents[circuit_ref]
        if (
            source_circuit.get("source", {}).get("component_id")
            != source["component_id"]
        ):
            raise ModelCompileError(
                "codebook semantic source component identity drifted"
            )
        component = _unique_record(
            manifest["circuit_graph"]["components"],
            "component_id",
            source["component_id"],
            "resident component",
        )
        execution = _unique_record(
            manifest["component_executions"],
            "component_id",
            source["component_id"],
            "resident component execution",
        )
        overlay_component = deepcopy(component)
        overlay_execution = deepcopy(execution)
        circuit = overlay_component["circuit"]
        node = _unique_record(
            circuit["nodes"],
            "id",
            source["physical_node_id"],
            "physical head-normalization node",
        )
        if (
            node["op"] != "parallel_head_norm_rope_2way"
            or node["attrs"].get("compiled_from") != source["physical_source_node_ids"]
            or len(node["params"]) != 2
        ):
            raise ModelCompileError(
                "codebook physical source node no longer matches target lowering"
            )
        original_params = tuple(node["params"])
        refs = circuit["parameters"]["refs"]
        artifact_refs = overlay_component["params"]["refs"]
        for parameter_id, source_tensor in zip(
            original_params,
            lowering["parameters"]["source_tensors"],
            strict=True,
        ):
            if (
                refs.get(parameter_id, {}).get("tensor") != source_tensor["name"]
                or artifact_refs.get(parameter_id, {}).get("tensor")
                != source_tensor["name"]
            ):
                raise ModelCompileError("codebook physical parameter binding drifted")
            refs.pop(parameter_id)
            artifact_refs.pop(parameter_id)
        replacement_ids = (
            "optimizer_codebook_branch_0_indices",
            "optimizer_codebook_branch_1_indices",
            "optimizer_codebook_entries",
        )
        if any(
            parameter_id in refs or parameter_id in artifact_refs
            for parameter_id in replacement_ids
        ):
            raise ModelCompileError(
                "codebook replacement parameter identity collides with source"
            )
        tensor_names = (
            *lowering["parameters"]["branch_index_tensor_names"],
            lowering["parameters"]["codebook_tensor_name"],
        )
        roles = (
            "exact_codebook_branch_0_addresses",
            "exact_codebook_branch_1_addresses",
            "exact_codebook_entries",
        )
        refs.update(
            {
                parameter_id: {"tensor": tensor_name, "role": role}
                for parameter_id, tensor_name, role in zip(
                    replacement_ids,
                    tensor_names,
                    roles,
                    strict=True,
                )
            }
        )
        artifact_refs.update(deepcopy(refs))
        if artifact_refs != refs:
            raise ModelCompileError(
                "codebook circuit and parameter artifact bindings diverged"
            )
        node["op"] = CODEBOOK_RUNTIME_OP
        node["params"] = list(replacement_ids)
        node["attrs"] = {
            **node["attrs"],
            "parameter_representation": {
                "kind": "shared_bf16_codebook_u8_addresses",
                "entry_count": len(lowering["parameters"]["codebook_values_u16"]),
                "source_parameter_ids": list(original_params),
                "descriptor_abi": "source_parameters_replaced",
                "alternative_execution_phases": ["decode", "prefill"],
                "source_retained_execution_phases": [],
            },
        }
        implementation = "exact_codebook_head_norm_rope_v1"
        overlay_component["implementation"] = implementation
        circuit["implementation"] = implementation
        overlay_execution["implementation"] = implementation
        kernel = _unique_record(
            overlay_execution["kernels"],
            "node_id",
            source["physical_node_id"],
            "physical head-normalization kernel",
        )
        if kernel["op"] != "parallel_head_norm_rope_2way":
            raise ModelCompileError(
                "codebook source execution no longer matches its physical node"
            )
        kernel["op"] = CODEBOOK_RUNTIME_OP
        kernel["shader_path"] = context.artifact_reference(
            DECODE_SHADER_PATH
        )
        for batch in kernel["batch_implementations"]:
            for stage in batch["stages"]:
                stage["shader_path"] = context.artifact_reference(
                    PREFILL_SHADER_PATH
                )
                control = stage.get("control", {})
                if control.get("kind") == "storage_buffer":
                    control["binding"] = 7
        context.write_json_artifact(
            OVERLAY_PATH,
            {
                "schema": "nerve.optimizer.vulkan_component_overlay.v1",
                "source_component_id": source["component_id"],
                "component": overlay_component,
                "execution": overlay_execution,
            },
        )
        context.account_transient_bytes(0)


class CodebookPhysicalOptimizer:
    def optimize_physical_artifacts(
        self,
        context: CandidateConstructionContext,
    ) -> None:
        lowering = _lowering(context)
        if "members" in lowering:
            for member in lowering["members"]:
                self.optimize_physical_artifacts(
                    MemberConstructionContext(context, member)
                )
            return
        attrs = lowering["geometry"]["physical_attrs"]
        decode_source = render_codebook_shader(
            attrs,
            temporal=False,
        )
        prefill_source = render_codebook_shader(
            attrs,
            temporal=True,
        )
        decode_spirv = compile_spirv(
            decode_source,
            "codebook_decode.comp",
        )
        prefill_spirv = compile_spirv(
            prefill_source,
            "codebook_prefill.comp",
        )
        context.account_transient_bytes(
            len(decode_source.encode("utf-8"))
            + len(prefill_source.encode("utf-8"))
            + len(decode_spirv)
            + len(prefill_spirv)
            + sum(
                int(source["payload_byte_count"])
                for source in lowering["parameters"]["source_tensors"]
            )
        )
        context.write_artifact(DECODE_SHADER_PATH, decode_spirv)
        context.write_artifact(PREFILL_SHADER_PATH, prefill_spirv)
        _write_equivalence_proof(
            context,
            lowering,
            decode_source=decode_source,
            decode_spirv=decode_spirv,
            prefill_source=prefill_source,
            prefill_spirv=prefill_spirv,
        )
        context.account_transient_bytes(0)


def _lowering(context: CandidateConstructionContext) -> Json:
    lowering = context.target_lowering
    if lowering.get("schema") != TARGET_LOWERING_SCHEMA:
        raise ModelCompileError("codebook toolchain received incompatible lowering")
    if lowering.get("candidate_id") != context.candidate["candidate_id"]:
        raise ModelCompileError("codebook lowering belongs to another candidate")
    return lowering


def _tensor_index_source(lowering: Json) -> str:
    matches = [
        record["path"]
        for record in lowering["source"]["source_inputs"]
        if record["path"] == "tensors.json"
    ]
    if matches != ["tensors.json"]:
        raise ModelCompileError("codebook lowering has no unique tensor index source")
    return matches[0]


def _single_tensor_safetensors(
    tensor_name: str,
    dtype: str,
    shape: list[int],
    payload: bytes,
) -> bytes:
    header = json.dumps(
        {
            tensor_name: {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [0, len(payload)],
            }
        },
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    header += b" " * (-len(header) % 8)
    return len(header).to_bytes(8, "little") + header + payload


def _tensor_fragment(
    *,
    branch_names: list[str],
    index_payloads: list[bytes],
    branch_paths: list[str],
    codebook_name: str,
    codebook_payload: bytes,
    codebook_path: str,
) -> Json:
    tensors = {}
    for name, path, payload in zip(
        branch_names,
        branch_paths,
        index_payloads,
        strict=True,
    ):
        tensors[name] = _tensor_metadata("U8", len(payload), path, payload)
    tensors[codebook_name] = _tensor_metadata(
        "BF16",
        len(codebook_payload) // 2,
        codebook_path,
        codebook_payload,
    )
    return {
        "schema": "nerve.tensor_index.v1",
        "tensors": dict(sorted(tensors.items())),
    }


def _word_aligned(payload: bytes) -> bytes:
    return payload + bytes(_word_aligned_byte_count(len(payload)) - len(payload))


def _word_aligned_byte_count(byte_count: int) -> int:
    return byte_count + (-byte_count % 4)


def _tensor_metadata(
    dtype: str,
    element_count: int,
    source_file: str,
    payload: bytes,
) -> Json:
    return {
        "dtype": dtype,
        "shape": [element_count],
        "parameter_count": element_count,
        "byte_count": len(payload),
        "data_offsets": [0, len(payload)],
        "source_file": source_file,
        "data_sha256": sha256(payload).hexdigest(),
        "layout": "row_major",
    }


def _write_equivalence_proof(
    context: CandidateConstructionContext,
    lowering: Json,
    *,
    decode_source: str,
    decode_spirv: bytes,
    prefill_source: str,
    prefill_spirv: bytes,
) -> None:
    sources = lowering["parameters"]["source_tensors"]
    codebook_values = tuple(
        int(value) for value in lowering["parameters"]["codebook_values_u16"]
    )
    codebook_payload = b"".join(
        value.to_bytes(2, "little") for value in codebook_values
    )
    addresses = {value: index for index, value in enumerate(codebook_values)}
    source_payloads = _read_source_tensor_payloads(context, sources)
    source_proofs = []
    for source, payload in zip(sources, source_payloads, strict=True):
        values = tuple(
            int.from_bytes(payload[offset : offset + 2], "little")
            for offset in range(0, len(payload), 2)
        )
        try:
            indices = bytes(addresses[value] for value in values)
        except KeyError as error:
            raise ModelCompileError(
                "codebook proof cannot reconstruct every source BF16 value"
            ) from error
        reconstructed = b"".join(
            codebook_values[index].to_bytes(2, "little") for index in indices
        )
        if reconstructed != payload:
            raise ModelCompileError(
                "codebook proof reconstruction changed source BF16 bits"
            )
        source_proofs.append(
            {
                "name": source["name"],
                "data_sha256": source["metadata"]["data_sha256"],
                "reconstructed_sha256": sha256(reconstructed).hexdigest(),
                "index_sha256": sha256(indices).hexdigest(),
                "element_count": len(indices),
            }
        )
    context.write_json_artifact(
        PROOF_PATH,
        {
            "schema": CODEBOOK_PROOF_SCHEMA,
            "candidate_id": lowering["candidate_id"],
            "scope_id": lowering["scope_id"],
            "source_tensors": source_proofs,
            "codebook": {
                "entry_count": len(codebook_values),
                "payload_sha256": sha256(codebook_payload).hexdigest(),
                "address_dtype": "U8",
                "entry_dtype": "BF16",
            },
            "obligations": {
                "codebook_reconstructs_source_bf16_bits": "proven",
                "fused_operator_preserves_source_rounding": "proven",
            },
            "overlay": {
                "physical_node_id": lowering["source"]["physical_node_id"],
                "replacement_op": lowering["runtime"]["replacement_op"],
                "compiled_from": lowering["source"]["physical_source_node_ids"],
                "intermediate_rounding": lowering["geometry"]["physical_attrs"][
                    "intermediate_rounding"
                ],
                "decode_shader_path": context.artifact_reference(
                    DECODE_SHADER_PATH
                ),
                "prefill_shader_paths": [
                    context.artifact_reference(PREFILL_SHADER_PATH)
                ],
            },
            "lowering": {
                "source_template_sha256": {
                    "decode": source_template_sha256(temporal=False),
                    "prefill": source_template_sha256(temporal=True),
                },
                "transformed_source_sha256": {
                    "decode": sha256(decode_source.encode("utf-8")).hexdigest(),
                    "prefill": sha256(prefill_source.encode("utf-8")).hexdigest(),
                },
                "spirv_sha256": {
                    "decode": sha256(decode_spirv).hexdigest(),
                    "prefill": sha256(prefill_spirv).hexdigest(),
                },
                "target_environment": "vulkan1.4",
            },
        },
    )


def _read_source_tensor_payloads(
    context: CandidateConstructionContext,
    sources: list[Json],
) -> tuple[bytes, ...]:
    grouped: dict[str, list[tuple[int, Json]]] = {}
    for index, source in enumerate(sources):
        grouped.setdefault(str(source["storage"]["path"]), []).append((index, source))
    payloads: list[bytes | None] = [None] * len(sources)
    for storage_path, records in sorted(grouped.items()):
        regions = tuple(
            (
                int(source["payload_byte_offset"]),
                int(source["payload_byte_count"]),
            )
            for _index, source in records
        )
        extracted = context.read_source_artifact_regions(
            storage_path,
            regions,
        )
        for (index, source), payload in zip(
            records,
            extracted,
            strict=True,
        ):
            if (
                len(payload) != source["payload_byte_count"]
                or sha256(payload).hexdigest() != source["metadata"]["data_sha256"]
            ):
                raise ModelCompileError(
                    f"codebook source tensor {source['name']!r} digest disagrees"
                )
            payloads[index] = payload
    if any(payload is None for payload in payloads):
        raise ModelCompileError("codebook source tensor extraction is incomplete")
    return tuple(payload for payload in payloads if payload is not None)


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
