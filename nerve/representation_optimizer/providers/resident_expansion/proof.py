from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from nerve.compilation import Json, ModelCompileError
from nerve.model_package_shader_templates import render_shader_source
from nerve.physical_representations import (
    independent_expert_resource_representation_dispatch,
)
from nerve.quantized_transforms import (
    MXFP4_E2M1_FP8_E4M3_BITS,
    mxfp4_value,
    resident_mxfp4_value,
)
from nerve.representation_optimizer.providers.resident_expansion.artifacts import (
    component_overlay_path,
    adaptive_shader_artifact_path,
)
from nerve.representation_optimizer.contracts import contract_digest
from nerve.representation_optimizer.providers.codebook.shaders import compile_spirv
from nerve.representation_optimizer.providers.resident_expansion.artifacts import (
    PROOF_PATH,
)
from nerve.representation_optimizer.providers.resident_expansion.contracts import (
    EXACT_EXPANSION_OBLIGATIONS,
    PROOF_SCHEMA,
    PROOF_VERIFIER_ID,
    TARGET_LOWERING_SCHEMA,
)
from nerve.representation_optimizer.providers.resident_expansion.discovery import (
    _tensor_pair_representation,
)
from nerve.representation_optimizer.providers.source_artifacts import (
    PackageSourceArtifactResolver,
)
from nerve.resident_representations import (
    MXFP4_TO_FP8_REQUIRED_FEATURES,
    mxfp4_to_fp8_resident_derivation,
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
class ExactResidentExpansionProofVerifier:
    source_artifacts: PackageSourceArtifactResolver
    candidate_workspace_root: Path
    verifier_id: str = PROOF_VERIFIER_ID

    def verify(self, request: ProofRequest) -> Json:
        diagnostics = []
        facts: Json = {}
        artifacts = []
        try:
            if request.obligation not in EXACT_EXPANSION_OBLIGATIONS:
                raise ModelCompileError(
                    f"unsupported resident expansion obligation {request.obligation!r}"
                )
            root = self._candidate_root(request.candidate_id)
            lowering = _json_file(_regular_file(root, "contracts/target_lowering.json"))
            if (
                lowering.get("schema") != TARGET_LOWERING_SCHEMA
                or lowering.get("candidate_id") != request.candidate_id
            ):
                raise ModelCompileError(
                    "resident expansion proof contracts belong to another candidate"
                )
            proof_path = _regular_file(root, PROOF_PATH)
            proof_digest = staged_file_digest(proof_path)
            _require_candidate_artifact(
                request.candidate_implementation,
                PROOF_PATH,
                proof_digest,
            )
            proof = _json_file(proof_path)
            code_domain = _verify_code_domain(lowering, proof)
            source_documents: dict[str, Json] = {}
            source_coverage = [
                _verify_source_coverage(
                    self.source_artifacts,
                    region,
                    source_documents=source_documents,
                )
                for region in _regions(lowering)
            ]
            resource_boundaries = [
                _verify_overlay(
                    self.source_artifacts,
                    root,
                    region,
                    source_documents=source_documents,
                )
                for region in _regions(lowering)
            ]
            deterministic_shaders = _verify_shaders(root, lowering)
            facts = {
                "code_domain": code_domain,
                "source_coverage": source_coverage,
                "resource_boundaries": resource_boundaries,
                "deterministic_shaders": deterministic_shaders,
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
                "proof artifact chunk size must be a positive integer"
            )
        candidate_id, separator, artifact_path = relative_path.partition("/")
        if (
            not separator
            or not candidate_id.startswith("candidate_")
            or artifact_path != PROOF_PATH
        ):
            raise ModelCompileError(
                "resident expansion proof artifact reference is invalid"
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
                "resident expansion proof candidate identity is unsafe"
            )
        workspace = self.candidate_workspace_root.resolve()
        root = (workspace / "ready" / candidate_id).resolve()
        if not root.is_relative_to(workspace) or root.is_symlink() or not root.is_dir():
            raise ModelCompileError(
                "resident expansion proof candidate bundle is unavailable"
            )
        return root


def _verify_code_domain(lowering: Json, proof: Json) -> Json:
    expected_mapping = [
        {"source_nibble": nibble, "resident_e4m3_bits": bits}
        for nibble, bits in enumerate(MXFP4_E2M1_FP8_E4M3_BITS)
    ]
    regions = _regions(lowering)
    expected_regions = [
        {
            "component_id": region["source"]["component_id"],
            "derivation_count": len(region["resident_derivations"]),
            "derivations_digest": contract_digest(
                {"resident_derivations": region["resident_derivations"]}
            ),
            "source_weight_bytes": sum(
                int(item["source_byte_count"])
                for item in region["resident_derivations"]
            ),
            "resident_weight_bytes": sum(
                int(item["derivation"]["resident_byte_count"])
                for item in region["resident_derivations"]
            ),
        }
        for region in regions
    ]
    if (
        proof.get("schema") != PROOF_SCHEMA
        or proof.get("candidate_id") != lowering["candidate_id"]
        or proof.get("scope_ids") != lowering["scope_ids"]
        or proof.get("mapping") != expected_mapping
        or proof.get("regions") != expected_regions
    ):
        raise ModelCompileError("resident expansion proof certificate is inconsistent")
    for nibble in range(16):
        for scale_byte in range(0xFF):
            if resident_mxfp4_value(nibble, scale_byte) != mxfp4_value(
                nibble,
                scale_byte,
            ):
                raise ModelCompileError(
                    f"resident expansion changes source code {nibble:#x}"
                )
    if any(
        region["resident_weight_bytes"] != region["source_weight_bytes"] * 2
        for region in expected_regions
    ):
        raise ModelCompileError("resident expansion proof sizes are inconsistent")
    return {
        "source_code_count": 16,
        "finite_scale_code_count": 255,
        "region_count": len(regions),
        "derivation_count": sum(
            len(region["resident_derivations"]) for region in regions
        ),
    }


def _verify_overlay(
    resolver: PackageSourceArtifactResolver,
    root: Path,
    lowering: Json,
    *,
    source_documents: dict[str, Json] | None = None,
) -> Json:
    manifest = _source_json(
        resolver,
        lowering["source"]["manifest_ref"],
        source_documents,
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
    overlay_path = lowering["artifacts"]["overlay_path"]
    if overlay_path != component_overlay_path(component_id):
        raise ModelCompileError(
            "resident expansion overlay path is not component-derived"
        )
    overlay = _json_file(_regular_file(root, overlay_path))
    expected_derivations = [
        {
            "node_id": item["node_id"],
            "parameter_id": item["parameter_id"],
            "derivation": item["derivation"],
        }
        for item in lowering["resident_derivations"]
    ]
    expected_derivations.sort(key=lambda item: (item["node_id"], item["parameter_id"]))
    if (
        overlay.get("schema") != "nerve.optimizer.vulkan_component_overlay.v2"
        or overlay.get("source_component_id") != component_id
        or overlay.get("resident_derivations") != expected_derivations
    ):
        raise ModelCompileError(
            "resident expansion overlay derivation boundary is inconsistent"
        )
    restored_component = deepcopy(overlay["component"])
    restored_execution = deepcopy(overlay["execution"])
    restored_component["implementation"] = source_component["implementation"]
    restored_component["circuit"]["implementation"] = source_component["circuit"][
        "implementation"
    ]
    restored_execution["implementation"] = source_execution["implementation"]
    resident_node_ids = sorted(
        {item["node_id"] for item in lowering["resident_derivations"]}
    )
    for node_id in resident_node_ids:
        target_kernel = _unique(restored_execution["kernels"], "node_id", node_id)
        source_kernel = _unique(source_execution["kernels"], "node_id", node_id)
        source_shader_path = str(source_kernel.get("shader_path", ""))
        if target_kernel.get("resource_representation_dispatch") != (
            independent_expert_resource_representation_dispatch(
                source_shader_path,
                adaptive=True,
            )
        ) or source_kernel.get("resource_representation_dispatch") != (
            independent_expert_resource_representation_dispatch(source_shader_path)
        ):
            raise ModelCompileError(
                "resident expansion overlay has an inconsistent explicit "
                "resource-representation dispatch boundary"
            )
        target_kernel["resource_representation_dispatch"] = deepcopy(
            source_kernel["resource_representation_dispatch"]
        )
    for replacement in lowering["shader_replacements"]:
        target_path = replacement["artifact_path"]
        source_path = replacement["source_path"]
        kernel = _unique(
            restored_execution["kernels"],
            "node_id",
            replacement["node_id"],
        )
        replaced = 0
        if replacement["execution_kind"] == "scalar":
            if kernel.get("shader_path") == target_path:
                kernel["shader_path"] = source_path
                replaced = 1
        else:
            for batch in kernel.get("batch_implementations", []):
                for stage in batch.get("stages", []):
                    if stage.get("shader_path") == target_path:
                        stage["shader_path"] = source_path
                        replaced += 1
        if replaced == 0:
            raise ModelCompileError(
                f"resident expansion target shader {target_path!r} is absent"
            )
    if restored_component != source_component or restored_execution != source_execution:
        raise ModelCompileError(
            "resident expansion overlay changes behavior outside its declared boundary"
        )
    return {
        "component_id": component_id,
        "derived_resource_count": len(expected_derivations),
        "source_component_restored": True,
    }


def _verify_source_coverage(
    resolver: PackageSourceArtifactResolver,
    lowering: Json,
    *,
    source_documents: dict[str, Json] | None = None,
) -> Json:
    manifest = _source_json(
        resolver,
        lowering["source"]["manifest_ref"],
        source_documents,
    )
    tensor_index_path = manifest.get("tensor_index_path")
    if tensor_index_path != "tensors.json":
        raise ModelCompileError(
            "resident expansion source tensor index is not canonical"
        )
    tensor_index = _source_json(
        resolver,
        tensor_index_path,
        source_documents,
    )
    tensors = tensor_index.get("tensors")
    if not isinstance(tensors, dict):
        raise ModelCompileError("resident expansion source tensor map is missing")
    component_id = lowering["source"]["component_id"]
    node_ids = lowering["source"]["node_ids"]
    if not isinstance(node_ids, list) or len(node_ids) != 2 or len(set(node_ids)) != 2:
        raise ModelCompileError("resident expansion source node boundary is invalid")
    component = _unique(
        manifest["circuit_graph"]["components"],
        "component_id",
        component_id,
    )
    execution = _unique(
        manifest["component_executions"],
        "component_id",
        component_id,
    )
    nodes = {
        node_id: _unique(component["circuit"]["nodes"], "id", node_id)
        for node_id in node_ids
    }
    kernels = {
        node_id: _unique(execution["kernels"], "node_id", node_id)
        for node_id in node_ids
    }
    roles = {node.get("op") for node in nodes.values()}
    if roles != {
        "independent_sparse_moe_down",
        "independent_sparse_moe_gate_up",
    }:
        raise ModelCompileError(
            "resident expansion source nodes are not one complete sparse expert pair"
        )
    refs = component.get("params", {}).get("refs")
    if not isinstance(refs, dict):
        raise ModelCompileError(
            "resident expansion source parameter references are missing"
        )
    residency = manifest.get("resource_residency")
    if not isinstance(residency, dict):
        raise ModelCompileError(
            "resident expansion source residency contract is missing"
        )
    bindings = {
        (
            item.get("component_id"),
            item.get("node_id"),
            item.get("parameter_id"),
        ): item
        for item in residency.get("bindings", [])
        if isinstance(item, dict)
    }
    resources = {
        item.get("id"): item
        for item in residency.get("resources", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    resource_consumers: dict[str, set[str]] = {}
    for binding in residency.get("bindings", []):
        mapping = binding.get("mapping") if isinstance(binding, dict) else None
        resource_id = mapping.get("resource_id") if isinstance(mapping, dict) else None
        consumer = binding.get("component_id") if isinstance(binding, dict) else None
        if isinstance(resource_id, str) and isinstance(consumer, str):
            resource_consumers.setdefault(resource_id, set()).add(consumer)
    expected_derivations = []
    for node_id, node in nodes.items():
        accesses = node.get("attrs", {}).get("selected_parameter_accesses")
        if not isinstance(accesses, list) or len(accesses) != 1:
            raise ModelCompileError(
                f"resident expansion node {node_id!r} has no exact selector map"
            )
        mapping = accesses[0].get("mapping")
        if not isinstance(mapping, list) or not mapping:
            raise ModelCompileError(
                f"resident expansion node {node_id!r} selector map is missing"
            )
        for selector, record in enumerate(mapping):
            parameter_ids = (
                record.get("parameter_ids") if isinstance(record, dict) else None
            )
            stride = 2 if node["op"].endswith("down") else 4
            if (
                record.get("selector") != selector
                or not isinstance(parameter_ids, list)
                or len(parameter_ids) != stride
            ):
                raise ModelCompileError(
                    f"resident expansion node {node_id!r} selector map drifted"
                )
            for parameter_slot in range(0, stride, 2):
                parameter_id = parameter_ids[parameter_slot]
                scale_parameter_id = parameter_ids[parameter_slot + 1]
                tensor_ref = refs.get(parameter_id)
                scale_ref = refs.get(scale_parameter_id)
                tensor_name = (
                    tensor_ref.get("tensor") if isinstance(tensor_ref, dict) else None
                )
                scale_name = (
                    scale_ref.get("tensor") if isinstance(scale_ref, dict) else None
                )
                tensor = tensors.get(tensor_name)
                scale = tensors.get(scale_name)
                binding = bindings.get((component_id, node_id, parameter_id))
                binding_mapping = (
                    binding.get("mapping") if isinstance(binding, dict) else None
                )
                if (
                    not isinstance(tensor_name, str)
                    or not isinstance(tensor, dict)
                    or not isinstance(binding_mapping, dict)
                    or binding_mapping.get("kind") != "selected_atomic_group"
                    or binding_mapping.get("selector_index") != selector
                    or binding_mapping.get("parameter_slot") != parameter_slot
                ):
                    raise ModelCompileError(
                        f"resident expansion source binding {node_id!r}/"
                        f"{parameter_id!r} drifted"
                    )
                resource_id = binding_mapping.get("resource_id")
                resource = resources.get(resource_id)
                source_byte_count = tensor.get("byte_count")
                if (
                    not isinstance(resource_id, str)
                    or not isinstance(resource, dict)
                    or resource.get("lifetime") != "dynamic"
                    or "resident_derivation" in resource
                    or _resource_byte_count(resource) != source_byte_count
                    or resource_consumers.get(resource_id) != {component_id}
                ):
                    raise ModelCompileError(
                        f"resident expansion source resource {resource_id!r} drifted"
                    )
                attrs = node.get("attrs", {})
                source_representation = _tensor_pair_representation(
                    tensor,
                    scale,
                    weight_name=tensor_name,
                    scale_name=scale_name,
                    output_size=int(
                        attrs["hidden_size"]
                        if node["op"].endswith("down")
                        else attrs["intermediate_size"]
                    ),
                    input_size=int(
                        attrs["intermediate_size"]
                        if node["op"].endswith("down")
                        else attrs["hidden_size"]
                    ),
                )
                if source_representation == "native_fp8_e4m3_e8m0_b128":
                    continue
                derivation = mxfp4_to_fp8_resident_derivation(
                    tensor,
                    {
                        "devices": [
                            {"shader_features": list(MXFP4_TO_FP8_REQUIRED_FEATURES)}
                        ]
                    },
                )
                if derivation is None:
                    raise ModelCompileError(
                        f"resident expansion tensor {tensor_name!r} is not exact MXFP4"
                    )
                expected_derivations.append(
                    {
                        "node_id": node_id,
                        "parameter_id": parameter_id,
                        "tensor_name": tensor_name,
                        "source_resource_id": resource_id,
                        "source_byte_count": source_byte_count,
                        "derivation": derivation,
                    }
                )
    expected_derivations.sort(key=lambda item: (item["node_id"], item["parameter_id"]))
    if lowering["resident_derivations"] != expected_derivations:
        raise ModelCompileError(
            "resident expansion lowering does not cover every selected source weight"
        )

    expected_replacements = []
    for node_id, kernel in kernels.items():
        expected_replacements.append((node_id, kernel.get("shader_path"), "scalar"))
        batches = kernel.get("batch_implementations")
        if not isinstance(batches, list) or not batches:
            raise ModelCompileError(
                f"resident expansion node {node_id!r} has no source batch path"
            )
        for batch in batches:
            stages = batch.get("stages") if isinstance(batch, dict) else None
            expert_paths = [
                stage.get("shader_path")
                for stage in stages or []
                if isinstance(stage, dict)
                and "_mxfp4_e2m1_" in str(stage.get("shader_path", ""))
            ]
            if len(expert_paths) != 1:
                raise ModelCompileError(
                    f"resident expansion node {node_id!r} batch path drifted"
                )
            expected_replacements.append((node_id, expert_paths[0], "batch"))
    expected_replacements = sorted(set(expected_replacements))
    actual_replacements = sorted(
        (
            item.get("node_id"),
            item.get("source_path"),
            item.get("execution_kind"),
        )
        for item in lowering["shader_replacements"]
    )
    if actual_replacements != expected_replacements or any(
        item.get("artifact_path")
        != adaptive_shader_artifact_path(item.get("source_path"))
        for item in lowering["shader_replacements"]
    ):
        raise ModelCompileError(
            "resident expansion lowering does not cover every source execution path"
        )
    return {
        "component_id": component_id,
        "selected_weight_count": len(expected_derivations),
        "execution_path_count": len(expected_replacements),
    }


def _verify_shaders(root: Path, lowering: Json) -> Json:
    verified = []
    by_artifact = {}
    for region in _regions(lowering):
        for replacement in region["shader_replacements"]:
            by_artifact.setdefault(replacement["artifact_path"], replacement)
    for path, replacement in sorted(by_artifact.items()):
        expected = compile_spirv(
            render_shader_source(_SHADER_ROOT, replacement["template_name"]),
            replacement["template_name"],
        )
        if _regular_file(root, path).read_bytes() != expected:
            raise ModelCompileError(
                f"resident expansion shader {path!r} is not deterministic"
            )
        verified.append(path)
    return {"shader_count": len(verified), "paths": verified}


def _regions(lowering: Json) -> list[Json]:
    regions = lowering.get("regions")
    if not isinstance(regions, list) or not regions:
        raise ModelCompileError("resident expansion proof has no component regions")
    component_ids = [
        region.get("source", {}).get("component_id")
        if isinstance(region, dict)
        else None
        for region in regions
    ]
    if any(
        not isinstance(component_id, str) or not component_id
        for component_id in component_ids
    ) or component_ids != sorted(set(component_ids)):
        raise ModelCompileError(
            "resident expansion proof regions must have sorted unique components"
        )
    return regions


def _resource_byte_count(resource: Json) -> int:
    ranges = resource.get("ranges")
    if not isinstance(ranges, list) or not ranges:
        return -1
    total = 0
    for record in ranges:
        byte_count = record.get("byte_count") if isinstance(record, dict) else None
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count <= 0
        ):
            return -1
        total += byte_count
    return total


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
        raise ModelCompileError(
            f"resident proof artifact path is unsafe: {relative_path!r}"
        )
    path = root
    for part in relative.parts:
        path /= part
        if path.is_symlink():
            raise ModelCompileError(
                f"resident proof path crosses a symlink: {relative_path!r}"
            )
    root = root.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ModelCompileError(
            f"resident proof artifact is unavailable: {relative_path!r}"
        )
    return resolved


def _json_file(path: Path) -> Json:
    document = json.loads(path.read_bytes())
    if not isinstance(document, dict):
        raise ModelCompileError(f"proof input must be a JSON object: {path}")
    return document


def _source_json(
    resolver: PackageSourceArtifactResolver,
    relative_path: str,
    documents: dict[str, Json] | None,
) -> Json:
    if documents is not None and relative_path in documents:
        return documents[relative_path]
    document = _json_file(
        _regular_file(resolver.package_root, relative_path)
    )
    if documents is not None:
        documents[relative_path] = document
    return document


def _unique(records: object, field: str, value: str) -> Json:
    if not isinstance(records, list):
        raise ModelCompileError(f"resident proof {field} records are missing")
    matches = [
        record
        for record in records
        if isinstance(record, dict) and record.get(field) == value
    ]
    if len(matches) != 1:
        raise ModelCompileError(f"resident proof has no unique {field}={value!r}")
    return matches[0]
