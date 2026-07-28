from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.providers.codebook.artifacts import (
    BRANCH_INDEX_PATHS,
    CODEBOOK_TENSOR_PATH,
    DECODE_SHADER_PATH,
    OVERLAY_PATH,
    PREFILL_SHADER_PATH,
    PROOF_PATH,
)
from nerve.representation_optimizer.providers.codebook.contracts import (
    CODEBOOK_PROOF_SCHEMA,
    CODEBOOK_RUNTIME_OP,
    TARGET_LOWERING_SCHEMA,
)
from nerve.representation_optimizer.providers.codebook.member_paths import (
    member_path,
    member_root,
)
from nerve.representation_optimizer.providers.codebook.shaders import (
    compile_spirv,
    render_codebook_shader,
    source_template_sha256,
)
from nerve.representation_optimizer.providers.source_artifacts import (
    PackageSourceArtifactResolver,
)
from nerve.representation_optimizer.staging.contracts import (
    staged_file_digest,
)
from nerve.representation_optimizer.validation.contracts import (
    PROOF_RESULT_SCHEMA,
    ProofResult,
    proof_result_id,
)
from nerve.representation_optimizer.validation.protocols import ProofRequest


CODEBOOK_PROOF_VERIFIER_ID = "nerve.exact_codebook_reconstruction.v1"
_SUPPORTED_OBLIGATIONS = frozenset(
    {
        "codebook_reconstructs_source_bf16_bits",
        "fused_operator_preserves_source_rounding",
    }
)


@dataclass(frozen=True)
class ExactCodebookProofVerifier:
    source_artifacts: PackageSourceArtifactResolver
    candidate_workspace_root: Path
    verifier_id: str = CODEBOOK_PROOF_VERIFIER_ID

    def verify(self, request: ProofRequest) -> Json:
        diagnostics = []
        facts: Json = {}
        artifacts = []
        try:
            if request.obligation not in _SUPPORTED_OBLIGATIONS:
                raise ModelCompileError(
                    f"unsupported codebook proof obligation {request.obligation!r}"
                )
            root = self._candidate_root(request.candidate_id)
            lowering = _json_file(_regular_file(root, "contracts/target_lowering.json"))
            if lowering.get("schema") != TARGET_LOWERING_SCHEMA or lowering.get(
                "candidate_id"
            ) != request.candidate_id:
                raise ModelCompileError(
                    "codebook proof contracts belong to another candidate"
                )
            members = lowering.get("members", [lowering])
            member_facts = []
            for member in members:
                scope_id = str(member["scope_id"])
                relative_proof = member_path(scope_id, PROOF_PATH)
                member_candidate_root = _regular_directory(
                    root, member_root(scope_id)
                )
                proof_path = _regular_file(root, relative_proof)
                artifact_digest = staged_file_digest(proof_path)
                _require_candidate_artifact(
                    request.candidate_implementation,
                    relative_proof,
                    artifact_digest,
                )
                proof = _json_file(proof_path)
                if (
                    proof.get("schema") != CODEBOOK_PROOF_SCHEMA
                    or proof.get("candidate_id") != request.candidate_id
                    or proof.get("scope_id") != scope_id
                ):
                    raise ModelCompileError(
                        "codebook member proof belongs to another candidate or scope"
                    )
                member_facts.append(
                    {
                        "scope_id": scope_id,
                        "exact_bf16_reconstruction": _verify_reconstruction(
                            self.source_artifacts,
                            member_candidate_root,
                            member,
                            proof,
                        ),
                        "source_overlay_difference_is_declared": _verify_overlay(
                            self.source_artifacts.package_root,
                            member_candidate_root,
                            member,
                            proof,
                        ),
                        "deterministic_lowering_matches": _verify_shaders(
                            member_candidate_root, member, proof
                        ),
                    }
                )
                artifacts.append(
                    {
                        "path": f"{request.candidate_id}/{relative_proof}",
                        "digest": artifact_digest,
                    }
                )
            facts = {
                "member_count": len(member_facts),
                "members": member_facts,
            }
        except (
            KeyError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            ModelCompileError,
        ) as error:
            diagnostics.append(str(error))
        document = {
            "schema": PROOF_RESULT_SCHEMA,
            "proof_id": "",
            "plan_id": request.plan_id,
            "candidate_id": request.candidate_id,
            "obligation": request.obligation,
            "verifier_id": request.verifier_id,
            "source_contract_digests": list(request.source_contract_digests),
            "construction_record_digest": (request.construction_record_digest),
            "status": "proven" if not diagnostics else "inconclusive",
            "facts": facts,
            "artifacts": artifacts,
            "diagnostics": diagnostics,
        }
        document["proof_id"] = proof_result_id(document)
        return ProofResult.from_json(document).to_json()

    def iter_proof_artifact(
        self,
        relative_path: str,
        *,
        chunk_bytes: int = 8 * 1024 * 1024,
    ):
        if (
            isinstance(chunk_bytes, bool)
            or not isinstance(chunk_bytes, int)
            or chunk_bytes <= 0
        ):
            raise ModelCompileError(
                "proof artifact chunk size must be a positive integer"
            )
        candidate_id, separator, artifact_path = relative_path.partition("/")
        if (
            not separator
            or not candidate_id.startswith("candidate_")
            or not artifact_path.startswith("members/scope_")
            or not artifact_path.endswith(f"/{PROOF_PATH}")
        ):
            raise ModelCompileError("codebook proof artifact reference is invalid")
        path = _regular_file(
            self._candidate_root(candidate_id),
            artifact_path,
        )
        with path.open("rb") as stream:
            while chunk := stream.read(chunk_bytes):
                yield chunk

    def _candidate_root(self, candidate_id: str) -> Path:
        if (
            not candidate_id.startswith("candidate_")
            or "/" in candidate_id
            or "\\" in candidate_id
        ):
            raise ModelCompileError("codebook proof candidate identity is unsafe")
        workspace = self.candidate_workspace_root.resolve()
        root = (workspace / "ready" / candidate_id).resolve()
        if not root.is_relative_to(workspace) or root.is_symlink() or not root.is_dir():
            raise ModelCompileError("codebook proof candidate bundle is unavailable")
        return root


def _regular_directory(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    if (
        not path.is_relative_to(root.resolve())
        or path.is_symlink()
        or not path.is_dir()
    ):
        raise ModelCompileError(
            f"codebook candidate directory is unavailable: {relative_path!r}"
        )
    return path


def _verify_reconstruction(
    resolver: PackageSourceArtifactResolver,
    candidate_root: Path,
    lowering: Json,
    proof: Json,
) -> Json:
    parameters = lowering["parameters"]
    codebook_values = tuple(int(value) for value in parameters["codebook_values_u16"])
    codebook_payload = b"".join(
        value.to_bytes(2, "little") for value in codebook_values
    )
    if (
        codebook_values != tuple(sorted(set(codebook_values)))
        or len(codebook_values) > 256
        or sha256(codebook_payload).hexdigest() != parameters["codebook_payload_sha256"]
    ):
        raise ModelCompileError(
            "codebook lowering does not contain a canonical U8-addressable table"
        )
    stored_codebook = _single_tensor_payload(
        _regular_file(candidate_root, CODEBOOK_TENSOR_PATH),
        parameters["codebook_tensor_name"],
        "BF16",
    )
    if (
        stored_codebook[: len(codebook_payload)] != codebook_payload
        or any(stored_codebook[len(codebook_payload) :])
        or len(stored_codebook) % 4
    ):
        raise ModelCompileError(
            "candidate codebook storage is not the exact aligned BF16 table"
        )
    addresses = {value: index for index, value in enumerate(codebook_values)}
    proof_sources = {record["name"]: record for record in proof["source_tensors"]}
    reconstructed = []
    for index, source in enumerate(parameters["source_tensors"]):
        tensor = resolver.resolve_tensor(source["name"])
        source_payload = resolver.read_tensor_storage(source["name"])
        if (
            tensor.metadata != source["metadata"]
            or tensor.storage.to_json() != source["storage"]
            or tensor.payload_byte_offset != source["payload_byte_offset"]
            or tensor.payload_byte_count != source["payload_byte_count"]
        ):
            raise ModelCompileError(
                f"source tensor {source['name']!r} no longer matches lowering"
            )
        index_payload = _single_tensor_payload(
            _regular_file(candidate_root, BRANCH_INDEX_PATHS[index]),
            parameters["branch_index_tensor_names"][index],
            "U8",
        )
        expected_indices = bytes(
            addresses[
                int.from_bytes(
                    source_payload[offset : offset + 2],
                    "little",
                )
            ]
            for offset in range(0, len(source_payload), 2)
        )
        if (
            index_payload[: len(expected_indices)] != expected_indices
            or any(index_payload[len(expected_indices) :])
            or len(index_payload) % 4
        ):
            raise ModelCompileError(
                f"candidate address storage for {source['name']!r} "
                "is not the exact aligned U8 sequence"
            )
        logical_indices = index_payload[: len(expected_indices)]
        decoded = b"".join(
            codebook_values[address].to_bytes(2, "little")
            for address in logical_indices
        )
        if decoded != source_payload:
            raise ModelCompileError(
                f"candidate addresses do not reconstruct {source['name']!r}"
            )
        certificate = proof_sources.get(source["name"])
        if certificate != {
            "name": source["name"],
            "data_sha256": sha256(source_payload).hexdigest(),
            "reconstructed_sha256": sha256(decoded).hexdigest(),
            "index_sha256": sha256(logical_indices).hexdigest(),
            "element_count": len(logical_indices),
        }:
            raise ModelCompileError(
                f"proof certificate for {source['name']!r} is inconsistent"
            )
        reconstructed.append(source["name"])
    if set(proof_sources) != set(reconstructed):
        raise ModelCompileError(
            "codebook proof has unexpected source tensor certificates"
        )
    if proof["codebook"] != {
        "entry_count": len(codebook_values),
        "payload_sha256": sha256(codebook_payload).hexdigest(),
        "address_dtype": "U8",
        "entry_dtype": "BF16",
    }:
        raise ModelCompileError("codebook proof table facts are inconsistent")
    return {
        "tensor_count": len(reconstructed),
        "logical_entry_count": len(codebook_values),
        "storage_byte_count": len(stored_codebook),
    }


def _verify_overlay(
    package_root: Path,
    candidate_root: Path,
    lowering: Json,
    proof: Json,
) -> bool:
    manifest = _json_file(
        _regular_file(
            package_root.resolve(),
            lowering["source"]["manifest_ref"],
        )
    )
    component_id = lowering["source"]["component_id"]
    source_component = _unique(
        manifest["circuit_graph"]["components"],
        "component_id",
        component_id,
    )
    source_execution = _unique(
        manifest["component_executions"],
        "component_id",
        component_id,
    )
    overlay = _json_file(_regular_file(candidate_root, OVERLAY_PATH))
    if (
        overlay.get("schema") != "nerve.optimizer.vulkan_component_overlay.v1"
        or overlay.get("source_component_id") != component_id
    ):
        raise ModelCompileError("candidate overlay identity is invalid")
    restored_component = deepcopy(overlay["component"])
    restored_execution = deepcopy(overlay["execution"])
    source_node = _unique(
        source_component["circuit"]["nodes"],
        "id",
        lowering["source"]["physical_node_id"],
    )
    node = _unique(
        restored_component["circuit"]["nodes"],
        "id",
        lowering["source"]["physical_node_id"],
    )
    replacement_ids = tuple(node["params"])
    if (
        node["op"] != CODEBOOK_RUNTIME_OP
        or len(replacement_ids) != 3
        or node["attrs"].get("parameter_representation")
        != {
            "kind": "shared_bf16_codebook_u8_addresses",
            "entry_count": len(lowering["parameters"]["codebook_values_u16"]),
            "source_parameter_ids": source_node["params"],
            "descriptor_abi": "source_parameters_replaced",
            "alternative_execution_phases": ["decode", "prefill"],
            "source_retained_execution_phases": [],
        }
    ):
        raise ModelCompileError("candidate overlay node rewrite is invalid")
    node["op"] = source_node["op"]
    node["params"] = deepcopy(source_node["params"])
    node["attrs"].pop("parameter_representation")
    for refs in (
        restored_component["circuit"]["parameters"]["refs"],
        restored_component["params"]["refs"],
    ):
        for parameter_id in replacement_ids:
            if refs.pop(parameter_id, None) is None:
                raise ModelCompileError(
                    "candidate overlay replacement parameter is absent"
                )
        refs.update(
            {
                parameter_id: deepcopy(
                    source_component["circuit"]["parameters"]["refs"][parameter_id]
                )
                for parameter_id in source_node["params"]
            }
        )
    restored_component["implementation"] = source_component["implementation"]
    restored_component["circuit"]["implementation"] = source_component["circuit"][
        "implementation"
    ]
    if restored_component != source_component:
        raise ModelCompileError(
            "candidate overlay changes undeclared component semantics"
        )

    source_kernel = _unique(
        source_execution["kernels"],
        "node_id",
        lowering["source"]["physical_node_id"],
    )
    kernel = _unique(
        restored_execution["kernels"],
        "node_id",
        lowering["source"]["physical_node_id"],
    )
    if (
        kernel["op"] != CODEBOOK_RUNTIME_OP
        or kernel["shader_path"]
        != member_path(lowering["scope_id"], DECODE_SHADER_PATH)
        or len(kernel["batch_implementations"])
        != len(source_kernel["batch_implementations"])
    ):
        raise ModelCompileError("candidate overlay kernel rewrite is invalid")
    kernel["op"] = source_kernel["op"]
    kernel["shader_path"] = source_kernel["shader_path"]
    for candidate_batch, source_batch in zip(
        kernel["batch_implementations"],
        source_kernel["batch_implementations"],
        strict=True,
    ):
        if len(candidate_batch["stages"]) != len(source_batch["stages"]):
            raise ModelCompileError("candidate overlay changes batch-stage topology")
        for candidate_stage, source_stage in zip(
            candidate_batch["stages"],
            source_batch["stages"],
            strict=True,
        ):
            if candidate_stage["shader_path"] != member_path(
                lowering["scope_id"], PREFILL_SHADER_PATH
            ):
                raise ModelCompileError(
                    "candidate overlay batch shader is inconsistent"
                )
            candidate_stage["shader_path"] = source_stage["shader_path"]
            candidate_stage["control"] = deepcopy(source_stage["control"])
    restored_execution["implementation"] = source_execution["implementation"]
    if restored_execution != source_execution:
        raise ModelCompileError(
            "candidate overlay changes undeclared execution semantics"
        )
    expected_overlay = {
        "physical_node_id": source_node["id"],
        "replacement_op": CODEBOOK_RUNTIME_OP,
        "compiled_from": source_node["attrs"]["compiled_from"],
        "intermediate_rounding": source_node["attrs"]["intermediate_rounding"],
        "decode_shader_path": member_path(
            lowering["scope_id"], DECODE_SHADER_PATH
        ),
        "prefill_shader_paths": [
            member_path(lowering["scope_id"], PREFILL_SHADER_PATH)
        ],
    }
    if proof["overlay"] != expected_overlay:
        raise ModelCompileError("codebook overlay proof facts are inconsistent")
    if proof["obligations"] != {
        "codebook_reconstructs_source_bf16_bits": "proven",
        "fused_operator_preserves_source_rounding": "proven",
    }:
        raise ModelCompileError("codebook proof has an unproven obligation")
    return True


def _verify_shaders(
    candidate_root: Path,
    lowering: Json,
    proof: Json,
) -> Json:
    attrs = lowering["geometry"]["physical_attrs"]
    decode_source = render_codebook_shader(attrs, temporal=False)
    prefill_source = render_codebook_shader(attrs, temporal=True)
    expected_decode = compile_spirv(decode_source, "proof_decode.comp")
    expected_prefill = compile_spirv(
        prefill_source,
        "proof_prefill.comp",
    )
    actual_decode = _regular_file(
        candidate_root,
        DECODE_SHADER_PATH,
    ).read_bytes()
    actual_prefill = _regular_file(
        candidate_root,
        PREFILL_SHADER_PATH,
    ).read_bytes()
    expected = {
        "source_template_sha256": {
            "decode": source_template_sha256(temporal=False),
            "prefill": source_template_sha256(temporal=True),
        },
        "transformed_source_sha256": {
            "decode": sha256(decode_source.encode("utf-8")).hexdigest(),
            "prefill": sha256(prefill_source.encode("utf-8")).hexdigest(),
        },
        "spirv_sha256": {
            "decode": sha256(actual_decode).hexdigest(),
            "prefill": sha256(actual_prefill).hexdigest(),
        },
        "target_environment": "vulkan1.4",
    }
    if (
        proof["lowering"] != expected
        or actual_decode != expected_decode
        or actual_prefill != expected_prefill
    ):
        raise ModelCompileError(
            "candidate SPIR-V does not match deterministic codebook lowering"
        )
    return {
        "decode_spirv_sha256": expected["spirv_sha256"]["decode"],
        "prefill_spirv_sha256": expected["spirv_sha256"]["prefill"],
    }


def _single_tensor_payload(
    path: Path,
    tensor_name: str,
    dtype: str,
) -> bytes:
    payload = path.read_bytes()
    if len(payload) < 8:
        raise ModelCompileError(f"candidate tensor file is truncated: {path}")
    header_bytes = int.from_bytes(payload[:8], "little")
    header_end = 8 + header_bytes
    if header_end > len(payload):
        raise ModelCompileError(f"candidate tensor header is truncated: {path}")
    header = json.loads(payload[8:header_end])
    if set(header) != {tensor_name}:
        raise ModelCompileError(f"candidate tensor file has unexpected tensors: {path}")
    metadata = header[tensor_name]
    offsets = metadata.get("data_offsets")
    if (
        metadata.get("dtype") != dtype
        or not isinstance(offsets, list)
        or len(offsets) != 2
        or offsets[0] != 0
        or offsets[1] < 0
    ):
        raise ModelCompileError(
            f"candidate tensor metadata is invalid: {tensor_name!r}"
        )
    tensor_payload = payload[header_end + offsets[0] : header_end + offsets[1]]
    if len(tensor_payload) != offsets[1] - offsets[0]:
        raise ModelCompileError(
            f"candidate tensor payload is truncated: {tensor_name!r}"
        )
    return tensor_payload


def _require_candidate_artifact(
    implementation: Json,
    path: str,
    digest: str,
) -> None:
    matches = [
        artifact
        for artifact in implementation["artifact_refs"]
        if artifact.get("path") == path
    ]
    if len(matches) != 1 or matches[0].get("digest") != digest:
        raise ModelCompileError(
            f"candidate implementation does not seal proof artifact {path!r}"
        )


def _regular_file(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if (
        relative_path == ""
        or relative.is_absolute()
        or relative.as_posix() != relative_path
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ModelCompileError(f"proof artifact path is unsafe: {relative_path!r}")
    path = root
    for part in relative.parts:
        path /= part
        if path.is_symlink():
            raise ModelCompileError(
                f"proof artifact path crosses a symlink: {relative_path!r}"
            )
    resolved = path.resolve()
    root = root.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ModelCompileError(
            f"proof artifact is not a confined regular file: {relative_path!r}"
        )
    return resolved


def _json_file(path: Path) -> Json:
    document = json.loads(path.read_bytes())
    if not isinstance(document, dict):
        raise ModelCompileError(f"proof input must be a JSON object: {path}")
    return document


def _unique(records: list[Json], field: str, value: str) -> Json:
    matches = [record for record in records if record.get(field) == value]
    if len(matches) != 1:
        raise ModelCompileError(f"proof source {field} {value!r} is not unique")
    return matches[0]
