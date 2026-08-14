from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from nerve.compilation import Json, ModelCompileError
from nerve.model_package_shader_templates import render_shader_source
from nerve.representation_optimizer.contracts import contract_digest
from nerve.representation_optimizer.providers.attention_head_grouping.artifacts import (
    PROOF_PATH,
)
from nerve.representation_optimizer.providers.attention_head_grouping.contracts import (
    EXACT_HEAD_GROUPING_OBLIGATIONS,
    PROOF_SCHEMA,
    PROOF_VERIFIER_ID,
    TARGET_LOWERING_SCHEMA,
)
from nerve.representation_optimizer.providers.attention_head_grouping.physical import (
    prepare_grouped_attention_from_documents,
)
from nerve.representation_optimizer.providers.attention_head_grouping.toolchain import (
    finalize_grouped_attention_kernel,
    opportunities_from_lowering,
)
from nerve.representation_optimizer.providers.codebook.shaders import compile_spirv
from nerve.representation_optimizer.providers.source_artifacts import (
    PackageSourceArtifactResolver,
)
from nerve.representation_optimizer.staging.contracts import staged_file_digest
from nerve.representation_optimizer.validation.contracts import (
    PROOF_RESULT_SCHEMA,
    ProofResult,
    proof_result_id,
)
from nerve.representation_optimizer.validation.protocols import ProofRequest


_SHADER_ROOT = Path(__file__).resolve().parents[4] / "runtime-rs" / "shaders"


@dataclass(frozen=True)
class ExactAttentionHeadGroupingProofVerifier:
    source_artifacts: PackageSourceArtifactResolver
    candidate_workspace_root: Path
    verifier_id: str = PROOF_VERIFIER_ID

    def verify(self, request: ProofRequest) -> Json:
        diagnostics = []
        facts: Json = {}
        artifacts = []
        try:
            if request.obligation not in EXACT_HEAD_GROUPING_OBLIGATIONS:
                raise ModelCompileError(
                    f"unsupported grouped-attention proof obligation {request.obligation!r}"
                )
            root = self._candidate_root(request.candidate_id)
            lowering = _json_file(
                _regular_file(root, "contracts/target_lowering.json")
            )
            if (
                lowering.get("schema") != TARGET_LOWERING_SCHEMA
                or lowering.get("candidate_id") != request.candidate_id
            ):
                raise ModelCompileError(
                    "grouped-attention proof contracts belong to another candidate"
                )
            proof_path = _regular_file(root, PROOF_PATH)
            proof_digest = staged_file_digest(proof_path)
            _require_candidate_artifact(
                request.candidate_implementation,
                PROOF_PATH,
                proof_digest,
            )
            proof = _json_file(proof_path)
            verified = _verify_candidate(
                resolver=self.source_artifacts,
                candidate_root=root,
                lowering=lowering,
                proof=proof,
            )
            facts = {
                "component_count": len(verified),
                "components": verified,
                "source_anchored": True,
                "per_head_reduction_order_preserved": True,
                "deterministic_spirv": True,
            }
            artifacts.append(
                {
                    "path": f"{request.candidate_id}/{PROOF_PATH}",
                    "digest": proof_digest,
                }
            )
        except (
            KeyError,
            OSError,
            TypeError,
            UnicodeDecodeError,
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
            "construction_record_digest": request.construction_record_digest,
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
                "grouped-attention proof artifact chunk size must be positive"
            )
        candidate_id, separator, artifact_path = relative_path.partition("/")
        if (
            not separator
            or not candidate_id.startswith("candidate_")
            or artifact_path != PROOF_PATH
        ):
            raise ModelCompileError(
                "grouped-attention proof artifact reference is invalid"
            )
        with _regular_file(
            self._candidate_root(candidate_id),
            artifact_path,
        ).open("rb") as stream:
            while chunk := stream.read(chunk_bytes):
                yield chunk

    def _candidate_root(self, candidate_id: str) -> Path:
        if (
            not candidate_id.startswith("candidate_")
            or "/" in candidate_id
            or "\\" in candidate_id
        ):
            raise ModelCompileError(
                "grouped-attention proof candidate identity is unsafe"
            )
        workspace = self.candidate_workspace_root.resolve()
        root = (workspace / "ready" / candidate_id).resolve()
        if not root.is_relative_to(workspace) or root.is_symlink() or not root.is_dir():
            raise ModelCompileError(
                "grouped-attention proof candidate bundle is unavailable"
            )
        return root


def _verify_candidate(
    *,
    resolver: PackageSourceArtifactResolver,
    candidate_root: Path,
    lowering: Json,
    proof: Json,
) -> list[Json]:
    opportunities = opportunities_from_lowering(lowering)
    documents: dict[str, Json] = {}

    def source_json(path: str) -> Json:
        if path not in documents:
            documents[path] = _json_bytes(resolver.read_path(path), path)
        return documents[path]

    shader_payloads = {}
    for record in lowering["shader_artifacts"]:
        path = str(record["artifact_path"])
        template = str(record["template_name"])
        expected = compile_spirv(
            render_shader_source(_SHADER_ROOT, template),
            template,
        )
        actual = _regular_file(candidate_root, path).read_bytes()
        if actual != expected:
            raise ModelCompileError(
                f"grouped-attention shader artifact {path!r} is not deterministic"
            )
        shader_payloads[path] = expected

    expected_components = []
    verified = []
    lowered_regions = {
        str(record["component_id"]): record for record in lowering["regions"]
    }
    for opportunity in opportunities:
        prepared = prepare_grouped_attention_from_documents(
            opportunity=opportunity,
            manifest=source_json(opportunity.manifest_ref),
            source_circuit=source_json(opportunity.circuit_ref),
        )
        finalized = tuple(
            finalize_grouped_attention_kernel(
                kernel,
                component=prepared,
                tensor_index=source_json(opportunity.tensor_index_ref),
                artifact_payloads=shader_payloads,
            )
            for kernel in prepared.replacement_kernels
        )
        expected_overlay = {
            "schema": "nerve.optimizer.vulkan_component_region_overlay.v2",
            "source_component_id": opportunity.component_id,
            "source": {
                "nodes": list(prepared.source_nodes),
                "kernels": list(prepared.source_kernels),
                "parameter_refs": {},
            },
            "replacement": {
                "nodes": list(prepared.replacement_nodes),
                "kernels": list(finalized),
                "parameter_refs": {},
            },
        }
        overlay = _json_file(
            _regular_file(
                candidate_root,
                str(lowered_regions[opportunity.component_id]["overlay_path"]),
            )
        )
        if overlay != expected_overlay:
            raise ModelCompileError(
                f"grouped-attention overlay for {opportunity.component_id!r} changed outside its proof"
            )
        expected_components.append(
            {
                "component_id": opportunity.component_id,
                "scope_id": opportunity.scope_id,
                "head_group": opportunity.head_group,
                "source_region_digest": contract_digest(expected_overlay["source"]),
                "replacement_region_digest": contract_digest(
                    expected_overlay["replacement"]
                ),
                "exact_rewrite": prepared.proof,
            }
        )
        verified.append(
            {
                "component_id": opportunity.component_id,
                "head_group": opportunity.head_group,
                "source_workgroups": opportunity.query_heads,
                "replacement_workgroups": (
                    opportunity.query_heads // opportunity.head_group
                ),
                "exact_reference": True,
            }
        )
    expected_proof = {
        "schema": PROOF_SCHEMA,
        "candidate_id": lowering["candidate_id"],
        "scope_ids": lowering["scope_ids"],
        "components": expected_components,
        "shader_artifacts": [
            {
                "path": path,
                "sha256": f"sha256:{sha256(payload).hexdigest()}",
            }
            for path, payload in sorted(shader_payloads.items())
        ],
    }
    if proof != expected_proof:
        raise ModelCompileError(
            "grouped-attention proof document does not reconstruct exactly"
        )
    return verified


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
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root.resolve()) or path.is_symlink() or not path.is_file():
        raise ModelCompileError(
            f"grouped-attention candidate artifact {relative_path!r} is unavailable"
        )
    return path


def _json_file(path: Path) -> Json:
    return _json_bytes(path.read_bytes(), str(path))


def _json_bytes(payload: bytes, label: str) -> Json:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelCompileError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ModelCompileError(f"{label} must contain a JSON object")
    return deepcopy(value)
