from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.providers.codebook.embedded_artifacts import (
    DECODE_SHADER_PATH,
    OVERLAY_PATH,
    PROOF_PATH,
)
from nerve.representation_optimizer.providers.codebook.embedded_contracts import (
    EMBEDDED_PARAMETER_PROGRAM_PROOF_SCHEMA,
    EMBEDDED_PARAMETER_PROGRAM_PROOF_VERIFIER_ID,
    EMBEDDED_TARGET_LOWERING_SCHEMA,
)
from nerve.representation_optimizer.providers.codebook.embedded_identity import (
    embedded_parameter_program_digest,
)
from nerve.representation_optimizer.providers.codebook.member_paths import (
    member_path,
    member_root,
)
from nerve.representation_optimizer.providers.codebook.shaders import (
    compile_spirv,
    render_embedded_parameter_shader,
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


_SUPPORTED_OBLIGATIONS = frozenset(
    {
        "embedded_parameter_program_reconstructs_source_bf16_bits",
        "embedded_parameter_program_preserves_source_rounding",
    }
)


@dataclass(frozen=True)
class ExactEmbeddedParameterProgramProofVerifier:
    source_artifacts: PackageSourceArtifactResolver
    candidate_workspace_root: Path
    verifier_id: str = EMBEDDED_PARAMETER_PROGRAM_PROOF_VERIFIER_ID

    def verify(self, request: ProofRequest) -> Json:
        diagnostics = []
        facts: Json = {}
        artifacts = []
        try:
            if request.obligation not in _SUPPORTED_OBLIGATIONS:
                raise ModelCompileError(
                    "unsupported embedded parameter-program proof obligation "
                    f"{request.obligation!r}"
                )
            root = self._candidate_root(request.candidate_id)
            lowering = _json_file(_regular_file(root, "contracts/target_lowering.json"))
            if (
                lowering.get("schema") != EMBEDDED_TARGET_LOWERING_SCHEMA
                or lowering.get("candidate_id") != request.candidate_id
            ):
                raise ModelCompileError(
                    "embedded parameter-program proof contracts belong to another "
                    "candidate"
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
                    proof.get("schema")
                    != EMBEDDED_PARAMETER_PROGRAM_PROOF_SCHEMA
                    or proof.get("candidate_id") != request.candidate_id
                    or proof.get("scope_id") != scope_id
                ):
                    raise ModelCompileError(
                        "embedded member proof belongs to another candidate or scope"
                    )
                member_facts.append(
                    {
                        "scope_id": scope_id,
                        "exact_bf16_reconstruction": _verify_reconstruction(
                            self.source_artifacts, member, proof
                        ),
                        "source_overlay_difference_is_declared": _verify_overlay(
                            self.source_artifacts.package_root,
                            member_candidate_root,
                            member,
                            proof,
                        ),
                        "deterministic_embedded_lowering_matches": _verify_shaders(
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
            raise ModelCompileError(
                "embedded parameter-program proof artifact reference is invalid"
            )
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
            raise ModelCompileError(
                "embedded parameter-program proof candidate identity is unsafe"
            )
        workspace = self.candidate_workspace_root.resolve()
        root = (workspace / "ready" / candidate_id).resolve()
        if not root.is_relative_to(workspace) or root.is_symlink() or not root.is_dir():
            raise ModelCompileError(
                "embedded parameter-program proof candidate bundle is unavailable"
            )
        return root


def _regular_directory(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    if (
        not path.is_relative_to(root.resolve())
        or path.is_symlink()
        or not path.is_dir()
    ):
        raise ModelCompileError(
            "embedded parameter-program candidate member directory is unavailable"
        )
    return path


def _verify_reconstruction(
    resolver: PackageSourceArtifactResolver,
    lowering: Json,
    proof: Json,
) -> Json:
    parameters = lowering["parameters"]
    branch_payloads = _branch_program_payloads(parameters)
    if len(parameters["source_tensors"]) != 2:
        raise ModelCompileError(
            "embedded parameter lowering requires two source tensors"
        )
    certificates = {record["name"]: record for record in proof["source_tensors"]}
    reconstructed_names = []
    for source, reconstructed in zip(
        parameters["source_tensors"],
        branch_payloads,
        strict=True,
    ):
        tensor = resolver.resolve_tensor(source["name"])
        payload = resolver.read_tensor_storage(source["name"])
        if (
            tensor.metadata != source["metadata"]
            or tensor.storage.to_json() != source["storage"]
            or tensor.payload_byte_offset != source["payload_byte_offset"]
            or tensor.payload_byte_count != source["payload_byte_count"]
        ):
            raise ModelCompileError(
                f"source tensor {source['name']!r} no longer matches lowering"
            )
        if len(reconstructed) != len(payload):
            raise ModelCompileError(
                "embedded parameter count disagrees with source tensor"
            )
        if reconstructed != payload:
            raise ModelCompileError(
                f"embedded parameter program does not reconstruct {source['name']!r}"
            )
        certificate = certificates.get(source["name"])
        if certificate != {
            "name": source["name"],
            "data_sha256": sha256(payload).hexdigest(),
            "reconstructed_sha256": sha256(reconstructed).hexdigest(),
            "element_count": len(reconstructed) // 2,
        }:
            raise ModelCompileError(
                f"embedded proof certificate for {source['name']!r} is inconsistent"
            )
        reconstructed_names.append(source["name"])
    if set(certificates) != set(reconstructed_names):
        raise ModelCompileError(
            "embedded parameter-program proof has unexpected tensor certificates"
        )
    if proof["parameter_program"] != {
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
    }:
        raise ModelCompileError(
            "embedded parameter-program proof facts are inconsistent"
        )
    return {
        "tensor_count": len(reconstructed_names),
        "branch_count": len(branch_payloads),
        "embedded_element_count": sum(
            len(payload) // 2 for payload in branch_payloads
        ),
    }


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
        raise ModelCompileError(
            "embedded parameter-program overlay identity is invalid"
        )
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
    expected_parameter_representation = {
        "kind": "spirv_constant_bf16_parameter_program",
        "program_digest": embedded_parameter_program_digest(
            lowering["parameters"]["branch_values_u16"]
        ),
        "branch_count": len(lowering["parameters"]["branch_values_u16"]),
        "elements_per_branch": int(lowering["geometry"]["head_width"]),
        "source_parameter_ids": source_node["params"],
        "alternative_execution_phases": ["decode"],
        "source_retained_execution_phases": ["prefill"],
        "descriptor_abi": "source_parameters_retained",
    }
    if (
        node["op"] != source_node["op"]
        or node["params"] != source_node["params"]
        or node["attrs"].get("parameter_representation")
        != expected_parameter_representation
    ):
        raise ModelCompileError(
            "embedded parameter-program overlay node rewrite is invalid"
        )
    node["attrs"].pop("parameter_representation")
    if (
        restored_component["circuit"]["parameters"]
        != source_component["circuit"]["parameters"]
        or restored_component["params"] != source_component["params"]
    ):
        raise ModelCompileError(
            "embedded parameter-program overlay does not retain the exact source "
            "parameter declarations required by prefill"
        )
    implementation = lowering["runtime"]["implementation"]
    if (
        restored_component["implementation"] != implementation
        or restored_component["circuit"]["implementation"] != implementation
        or restored_execution["implementation"] != implementation
    ):
        raise ModelCompileError(
            "embedded parameter-program overlay implementation identity is invalid"
        )
    restored_component["implementation"] = source_component["implementation"]
    restored_component["circuit"]["implementation"] = source_component["circuit"][
        "implementation"
    ]
    restored_execution["implementation"] = source_execution["implementation"]
    if restored_component != source_component:
        raise ModelCompileError(
            "embedded overlay changes undeclared component semantics"
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
        kernel["op"] != source_kernel["op"]
        or kernel["shader_path"]
        != member_path(lowering["scope_id"], DECODE_SHADER_PATH)
        or kernel["batch_implementations"]
        != source_kernel["batch_implementations"]
    ):
        raise ModelCompileError(
            "embedded parameter-program overlay kernel rewrite is invalid"
        )
    kernel["shader_path"] = source_kernel["shader_path"]
    if restored_execution != source_execution:
        raise ModelCompileError(
            "embedded overlay changes undeclared execution semantics"
        )
    if proof["overlay"] != {
        "physical_node_id": source_node["id"],
        "source_abi_op": lowering["runtime"]["source_abi_op"],
        "implementation": implementation,
        "alternative_execution_phases": ["decode"],
        "source_retained_execution_phases": ["prefill"],
        "descriptor_abi": "source_parameters_retained",
        "compiled_from": source_node["attrs"]["compiled_from"],
        "intermediate_rounding": source_node["attrs"]["intermediate_rounding"],
        "decode_shader_path": member_path(
            lowering["scope_id"], DECODE_SHADER_PATH
        ),
    }:
        raise ModelCompileError(
            "embedded parameter-program overlay proof facts are inconsistent"
        )
    if proof["obligations"] != {
        "embedded_parameter_program_reconstructs_source_bf16_bits": "proven",
        "embedded_parameter_program_preserves_source_rounding": "proven",
    }:
        raise ModelCompileError(
            "embedded parameter-program proof has an unproven obligation"
        )
    return True


def _verify_shaders(
    candidate_root: Path,
    lowering: Json,
    proof: Json,
) -> Json:
    parameters = lowering["parameters"]
    raw_branches = tuple(
        tuple(int(value) for value in branch)
        for branch in parameters["branch_values_u16"]
    )
    if len(raw_branches) != 2:
        raise ModelCompileError(
            "embedded parameter-program proof requires two BF16 branches"
        )
    branches = (raw_branches[0], raw_branches[1])
    attrs = lowering["geometry"]["physical_attrs"]
    decode_source = render_embedded_parameter_shader(
        attrs,
        branch_values=branches,
        temporal=False,
        retain_parameter_abi=True,
    )
    expected_decode = compile_spirv(
        decode_source,
        "embedded_proof_decode.comp",
    )
    actual_decode = _regular_file(
        candidate_root,
        DECODE_SHADER_PATH,
    ).read_bytes()
    expected = {
        "source_template_sha256": {
            "decode": source_template_sha256(temporal=False),
        },
        "transformed_source_sha256": {
            "decode": sha256(decode_source.encode()).hexdigest(),
        },
        "spirv_sha256": {
            "decode": sha256(actual_decode).hexdigest(),
        },
        "target_environment": "vulkan1.4",
    }
    if (
        proof["lowering"] != expected
        or actual_decode != expected_decode
    ):
        raise ModelCompileError(
            "embedded candidate SPIR-V does not match deterministic lowering"
        )
    return {
        "decode_spirv_sha256": expected["spirv_sha256"]["decode"],
    }


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
            f"candidate proof artifact {path!r} is not integrity-bound"
        )


def _regular_file(root: Path, relative: str) -> Path:
    base = root.resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base) or path.is_symlink() or not path.is_file():
        raise ModelCompileError(
            f"embedded parameter-program artifact is unavailable: {relative}"
        )
    return path


def _json_file(path: Path) -> Json:
    document = json.loads(path.read_bytes())
    if not isinstance(document, dict):
        raise ModelCompileError(
            f"embedded parameter-program JSON artifact must be an object: {path}"
        )
    return document


def _unique(
    records: list[Json],
    field: str,
    expected: str,
) -> Json:
    matches = [record for record in records if record.get(field) == expected]
    if len(matches) != 1:
        raise ModelCompileError(
            f"embedded parameter-program record {expected!r} is not unique"
        )
    return matches[0]
