from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.automation.target import CandidateToolchain
from nerve.representation_optimizer.providers.codebook.artifacts import (
    component_fixture_from_geometry,
    conversation_fixture,
    model_limits_fixture,
    product_conversation_fixture,
)
from nerve.representation_optimizer.providers.codebook.embedded_artifacts import (
    COMPONENT_FIXTURE_PATH,
    CONVERSATION_FIXTURE_PATH,
    DECODE_SHADER_PATH,
    MODEL_LIMITS_PATH,
    OVERLAY_PATH,
    PRODUCT_CONVERSATION_FIXTURE_PATH,
    PROOF_PATH,
)
from nerve.representation_optimizer.providers.codebook.embedded_contracts import (
    EMBEDDED_PARAMETER_PROGRAM_PROOF_SCHEMA,
    EMBEDDED_TARGET_LOWERING_SCHEMA,
)
from nerve.representation_optimizer.providers.codebook.embedded_identity import (
    embedded_parameter_program_digest,
)
from nerve.representation_optimizer.providers.codebook.member_context import (
    MemberConstructionContext,
)
from nerve.representation_optimizer.providers.codebook.shaders import (
    compile_spirv,
    render_embedded_parameter_shader,
    source_template_sha256,
)
from nerve.representation_optimizer.providers.types import ProviderCandidatePlan
from nerve.representation_optimizer.staging.workspace import (
    CandidateConstructionContext,
)


class EmbeddedParameterProgramToolchainResolver:
    def resolve(self, plan: ProviderCandidatePlan) -> CandidateToolchain:
        if (
            plan.provider.provider_id
            != "nerve.exact_embedded_head_norm_parameter_program"
            or plan.target_lowering.get("schema") != EMBEDDED_TARGET_LOWERING_SCHEMA
        ):
            raise ModelCompileError(
                "embedded parameter-program toolchain cannot construct provider "
                f"{plan.provider.provider_id!r}"
            )
        return CandidateToolchain(
            semantic_constructor=EmbeddedParameterProgramSemanticConstructor(),
            ordinary_relowerer=EmbeddedParameterProgramOrdinaryRelowerer(),
            physical_optimizer=EmbeddedParameterProgramPhysicalOptimizer(),
        )


class EmbeddedParameterProgramSemanticConstructor:
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
        source_payloads = _read_source_tensor_payloads(
            context,
            lowering["parameters"]["source_tensors"],
        )
        tensor_index_path = _tensor_index_source(lowering)
        tensor_index = _json_object(
            context.read_source_artifact(tensor_index_path),
            tensor_index_path,
        )
        for source in lowering["parameters"]["source_tensors"]:
            if (
                tensor_index.get("tensors", {}).get(source["name"])
                != source["metadata"]
            ):
                raise ModelCompileError(
                    f"embedded parameter-program source tensor {source['name']!r} "
                    "metadata drifted"
                )
        _verify_embedded_reconstruction(lowering, source_payloads)
        context.account_transient_bytes(sum(len(item) for item in source_payloads))
        branch_head_counts = tuple(
            int(value) for value in lowering["geometry"]["branch_head_counts"]
        )
        if len(branch_head_counts) != 2:
            raise ModelCompileError(
                "embedded parameter-program fixture requires two normalization branches"
            )
        context.write_json_artifact(
            COMPONENT_FIXTURE_PATH,
            component_fixture_from_geometry(
                component_id=lowering["source"]["component_id"],
                physical_node_id=lowering["source"]["physical_node_id"],
                head_width=int(lowering["geometry"]["head_width"]),
                branch_head_counts=(
                    branch_head_counts[0],
                    branch_head_counts[1],
                ),
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
        context.account_transient_bytes(0)


class EmbeddedParameterProgramOrdinaryRelowerer:
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
        source_artifacts = {
            path: context.read_source_artifact(path) for path in source["artifact_refs"]
        }
        manifest_payload = context.read_source_artifact(source["manifest_ref"])
        context.account_transient_bytes(
            len(manifest_payload)
            + sum(len(payload) for payload in source_artifacts.values())
        )
        manifest = _json_object(
            manifest_payload,
            source["manifest_ref"],
        )
        circuit = _json_object(
            source_artifacts[source["circuit_ref"]],
            source["circuit_ref"],
        )
        if circuit.get("source", {}).get("component_id") != source["component_id"]:
            raise ModelCompileError(
                "embedded parameter-program source component identity drifted"
            )
        source_component = _unique_record(
            manifest["circuit_graph"]["components"],
            "component_id",
            source["component_id"],
            "resident component",
        )
        source_execution = _unique_record(
            manifest["component_executions"],
            "component_id",
            source["component_id"],
            "resident component execution",
        )
        overlay_component = deepcopy(source_component)
        overlay_execution = deepcopy(source_execution)
        node = _unique_record(
            overlay_component["circuit"]["nodes"],
            "id",
            source["physical_node_id"],
            "physical head-normalization node",
        )
        source_parameter_ids = [
            item["parameter_ref_id"]
            for item in lowering["parameters"]["source_tensors"]
        ]
        if (
            node["op"] != "parallel_head_norm_rope_2way"
            or node["attrs"].get("compiled_from") != source["physical_source_node_ids"]
            or node["params"] != source_parameter_ids
        ):
            raise ModelCompileError(
                "embedded parameter-program source node no longer matches lowering"
            )
        node["attrs"] = {
            **node["attrs"],
            "parameter_representation": {
                "kind": "spirv_constant_bf16_parameter_program",
                "program_digest": embedded_parameter_program_digest(
                    lowering["parameters"]["branch_values_u16"]
                ),
                "branch_count": len(
                    lowering["parameters"]["branch_values_u16"]
                ),
                "elements_per_branch": int(lowering["geometry"]["head_width"]),
                "source_parameter_ids": source_parameter_ids,
                "alternative_execution_phases": ["decode"],
                "source_retained_execution_phases": ["prefill"],
                "descriptor_abi": "source_parameters_retained",
            },
        }
        implementation = lowering["runtime"]["implementation"]
        overlay_component["implementation"] = implementation
        overlay_component["circuit"]["implementation"] = implementation
        overlay_execution["implementation"] = implementation
        kernel = _unique_record(
            overlay_execution["kernels"],
            "node_id",
            source["physical_node_id"],
            "physical head-normalization kernel",
        )
        if (
            kernel["op"] != "parallel_head_norm_rope_2way"
            or kernel["shader_path"] == DECODE_SHADER_PATH
        ):
            raise ModelCompileError(
                "embedded parameter-program source execution is not an exact baseline"
            )
        kernel["shader_path"] = context.artifact_reference(
            DECODE_SHADER_PATH
        )
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


class EmbeddedParameterProgramPhysicalOptimizer:
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
        parameters = lowering["parameters"]
        branch_values = tuple(
            tuple(int(value) for value in branch)
            for branch in parameters["branch_values_u16"]
        )
        if len(branch_values) != 2:
            raise ModelCompileError(
                "embedded parameter lowering requires two BF16 branches"
            )
        attrs = lowering["geometry"]["physical_attrs"]
        decode_source = render_embedded_parameter_shader(
            attrs,
            branch_values=(branch_values[0], branch_values[1]),
            temporal=False,
            retain_parameter_abi=True,
        )
        decode_spirv = compile_spirv(
            decode_source,
            "embedded_parameters_decode.comp",
        )
        source_payloads = _read_source_tensor_payloads(
            context,
            parameters["source_tensors"],
        )
        _verify_embedded_reconstruction(lowering, source_payloads)
        context.account_transient_bytes(
            len(decode_source.encode())
            + len(decode_spirv)
            + sum(len(item) for item in source_payloads)
        )
        context.write_artifact(DECODE_SHADER_PATH, decode_spirv)
        _write_equivalence_proof(
            context,
            lowering,
            source_payloads=source_payloads,
            decode_source=decode_source,
            decode_spirv=decode_spirv,
        )
        context.account_transient_bytes(0)


def _lowering(context: CandidateConstructionContext) -> Json:
    lowering = context.target_lowering
    if lowering.get("schema") != EMBEDDED_TARGET_LOWERING_SCHEMA:
        raise ModelCompileError(
            "embedded parameter-program toolchain received incompatible lowering"
        )
    if lowering.get("candidate_id") != context.candidate["candidate_id"]:
        raise ModelCompileError(
            "embedded parameter-program lowering belongs to another candidate"
        )
    return lowering


def _verify_embedded_reconstruction(
    lowering: Json,
    source_payloads: tuple[bytes, ...],
) -> None:
    parameters = lowering["parameters"]
    branch_payloads = _branch_program_payloads(parameters)
    if len(source_payloads) != 2:
        raise ModelCompileError(
            "embedded parameter lowering requires two source tensors"
        )
    for payload, reconstructed in zip(
        source_payloads, branch_payloads, strict=True
    ):
        if len(reconstructed) != len(payload):
            raise ModelCompileError(
                "embedded parameter count disagrees with source tensor"
            )
        if reconstructed != payload:
            raise ModelCompileError(
                "embedded parameter program does not reconstruct source BF16 bits"
            )


def _branch_program_payloads(parameters: Json) -> tuple[bytes, bytes]:
    raw_branches = parameters.get("branch_values_u16")
    if not isinstance(raw_branches, list) or len(raw_branches) != 2:
        raise ModelCompileError(
            "embedded parameter lowering requires two BF16 branches"
        )
    payloads = []
    for raw_values in raw_branches:
        if not isinstance(raw_values, list) or not raw_values:
            raise ModelCompileError(
                "embedded parameter branch must be a non-empty BF16 sequence"
            )
        values = []
        for value in raw_values:
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value > 0xFFFF
            ):
                raise ModelCompileError(
                    "embedded parameter branch contains a non-BF16 bit pattern"
                )
            values.append(value)
        payloads.append(
            b"".join(value.to_bytes(2, "little") for value in values)
        )
    return payloads[0], payloads[1]


def _tensor_index_source(lowering: Json) -> str:
    matches = [
        source["path"]
        for source in lowering["source"]["source_inputs"]
        if source["path"] == "tensors.json"
    ]
    if matches != ["tensors.json"]:
        raise ModelCompileError(
            "embedded parameter lowering has no unique tensor index"
        )
    return matches[0]


def _write_equivalence_proof(
    context: CandidateConstructionContext,
    lowering: Json,
    *,
    source_payloads: tuple[bytes, ...],
    decode_source: str,
    decode_spirv: bytes,
) -> None:
    parameters = lowering["parameters"]
    source_certificates = []
    branch_payloads = _branch_program_payloads(parameters)
    for source, payload, reconstructed in zip(
        parameters["source_tensors"],
        source_payloads,
        branch_payloads,
        strict=True,
    ):
        source_certificates.append(
            {
                "name": source["name"],
                "data_sha256": sha256(payload).hexdigest(),
                "reconstructed_sha256": sha256(reconstructed).hexdigest(),
                "element_count": len(reconstructed) // 2,
            }
        )
    context.write_json_artifact(
        PROOF_PATH,
        {
            "schema": EMBEDDED_PARAMETER_PROGRAM_PROOF_SCHEMA,
            "candidate_id": lowering["candidate_id"],
            "scope_id": lowering["scope_id"],
            "source_tensors": source_certificates,
            "parameter_program": {
                "program_digest": embedded_parameter_program_digest(
                    parameters["branch_values_u16"]
                ),
                "branch_count": len(branch_payloads),
                "element_count": sum(len(payload) // 2 for payload in branch_payloads),
                "source_payload_sha256": [
                    sha256(payload).hexdigest() for payload in branch_payloads
                ],
                "entry_dtype": "BF16",
                "storage": "spirv_constant_program",
            },
            "obligations": {
                "embedded_parameter_program_reconstructs_source_bf16_bits": (
                    "proven"
                ),
                "embedded_parameter_program_preserves_source_rounding": "proven",
            },
            "overlay": {
                "physical_node_id": lowering["source"]["physical_node_id"],
                "source_abi_op": lowering["runtime"]["source_abi_op"],
                "implementation": lowering["runtime"]["implementation"],
                "alternative_execution_phases": lowering["runtime"][
                    "alternative_execution_phases"
                ],
                "source_retained_execution_phases": lowering["runtime"][
                    "source_retained_execution_phases"
                ],
                "descriptor_abi": "source_parameters_retained",
                "compiled_from": lowering["source"]["physical_source_node_ids"],
                "intermediate_rounding": lowering["geometry"]["physical_attrs"][
                    "intermediate_rounding"
                ],
                "decode_shader_path": context.artifact_reference(
                    DECODE_SHADER_PATH
                ),
            },
            "lowering": {
                "source_template_sha256": {
                    "decode": source_template_sha256(temporal=False),
                },
                "transformed_source_sha256": {
                    "decode": sha256(decode_source.encode()).hexdigest(),
                },
                "spirv_sha256": {
                    "decode": sha256(decode_spirv).hexdigest(),
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
        grouped.setdefault(
            str(source["storage"]["path"]),
            [],
        ).append((index, source))
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
                    f"embedded parameter-program source tensor {source['name']!r} "
                    "digest disagrees"
                )
            payloads[index] = payload
    if any(payload is None for payload in payloads):
        raise ModelCompileError(
            "embedded parameter-program source tensor extraction is incomplete"
        )
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
