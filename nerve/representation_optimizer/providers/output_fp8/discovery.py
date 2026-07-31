from __future__ import annotations

import json
import re
from dataclasses import dataclass

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.providers.source_artifacts import (
    SourceTensorArtifact,
)
from nerve.representation_optimizer.providers.types import ProviderContext


_SOURCE_SHADER = re.compile(
    r"tied_output_projection(?:_dot2)?_bf16_\d+x\d+_"
    r"scale([A-Za-z0-9_]+)_to_f32\.spv"
)


@dataclass(frozen=True)
class OutputProjectionOpportunity:
    scope_id: str
    source_contract_digest: str
    component_id: str
    physical_node_id: str
    norm_parameter_ref_id: str
    projection_parameter_ref_id: str
    projection_scale_parameter_ref_id: str
    source_node_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    source_artifact_refs: tuple[str, ...]
    manifest_ref: str
    circuit_ref: str
    tensor: SourceTensorArtifact
    norm_tensor_name: str
    hidden_size: int
    vocabulary_size: int
    output_scale_token: str
    fp8_process_names: tuple[str, ...]
    speculative_decoder_ids: tuple[str, ...]
    max_context_activations: int

    @property
    def block_rows(self) -> int:
        return 16

    @property
    def block_columns(self) -> int:
        return 128

    @property
    def scale_shape(self) -> tuple[int, int]:
        return (
            (self.vocabulary_size + self.block_rows - 1) // self.block_rows,
            self.hidden_size // self.block_columns,
        )

    @property
    def candidate_weight_name(self) -> str:
        return f"nerve.optimizer.output_fp8.{self.scope_id}.weight"

    @property
    def candidate_scale_name(self) -> str:
        return f"nerve.optimizer.output_fp8.{self.scope_id}.scale_inv"

    @property
    def has_role_specialized_draft(self) -> bool:
        return bool(self.speculative_decoder_ids)

    @property
    def draft_group_columns(self) -> int:
        return 128

    @property
    def draft_scale_shape(self) -> tuple[int, int]:
        return (
            self.vocabulary_size,
            self.hidden_size // self.draft_group_columns,
        )

    @property
    def draft_weight_name(self) -> str:
        return f"nerve.optimizer.output_fp8.{self.scope_id}.draft_int4_weight"

    @property
    def draft_scale_name(self) -> str:
        return f"nerve.optimizer.output_fp8.{self.scope_id}.draft_int4_scale"


@dataclass(frozen=True)
class DiscoveryResult:
    opportunity: OutputProjectionOpportunity | None
    reasons: tuple[str, ...]
    evidence_ids: tuple[str, ...] = ()


def discover_output_projection(
    context: ProviderContext,
) -> DiscoveryResult:
    context.checkpoint()
    if len(context.scopes) != 1 or len(context.source_contracts) != 1:
        return DiscoveryResult(
            None,
            ("provider requires one independently mountable semantic scope",),
        )
    scope = context.scopes[0]
    contract = context.source_contracts[0]
    component_ids = scope["members"]["component_ids"]
    if (
        scope["kind"] != "output_transducer"
        or len(component_ids) != 1
        or contract["semantic_role"] != "output_transducer"
    ):
        return DiscoveryResult(
            None,
            ("scope is not the standalone output transducer",),
        )
    profile = context.hardware_profile
    if (
        profile["hardware_identity"]["device_kind"] != "gpu"
        or profile["provenance"]["api"] != "vulkan"
    ):
        return DiscoveryResult(
            None,
            ("block-scaled output execution currently requires a Vulkan GPU",),
        )
    fp8_processes = tuple(
        sorted(
            process["name"]
            for process in profile["processes"]
            if process["availability"] == "available"
            and process["programmability"] != "none"
            and "f8_e4m3" in process["numeric_formats"]
            and process["name"]
            in {
                "packed_dot_product",
                "shader_vector",
            }
        )
    )
    if "packed_dot_product" not in fp8_processes:
        return DiscoveryResult(
            None,
            ("target does not expose programmable native F8 E4M3 packed dot products",),
        )
    evidence_ids = tuple(
        sorted(
            str(evidence["evidence_id"])
            for evidence in context.evidence
            if any(
                claim.get("status") == "supported"
                for claim in evidence.get("claims", [])
            )
        )
    )
    if not evidence_ids:
        return DiscoveryResult(
            None,
            ("scope has no supported algebraic evidence",),
        )

    resolver = context.source_artifacts
    manifest_ref = "vulkan_resident_package.json"
    manifest = _json_object(
        resolver.read_path(manifest_ref),
        manifest_ref,
    )
    output = manifest.get("output_transducer")
    if not isinstance(output, dict) or not isinstance(output.get("spec"), dict):
        return DiscoveryResult(
            None,
            ("compiled package has no resident output-transducer contract",),
            evidence_ids,
        )
    spec = output["spec"]
    component_id = str(spec.get("transducer_id", ""))
    scoped_component_id = str(component_ids[0])
    node_ids = spec.get("node_ids")
    shape = spec.get("projection_parameter_shape")
    tensor_name = spec.get("projection_parameter_tensor")
    shader_path = output.get("projection_shader_path")
    shader_name = shader_path.rsplit("/", 1)[-1] if isinstance(shader_path, str) else ""
    shader_match = _SOURCE_SHADER.fullmatch(shader_name)
    if (
        component_id != scoped_component_id
        or not isinstance(node_ids, list)
        or not node_ids
        or any(not isinstance(node_id, str) or not node_id for node_id in node_ids)
        or spec.get("projection_parameter_dtype") != "BF16"
        or not isinstance(shape, list)
        or len(shape) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in shape
        )
        or not isinstance(tensor_name, str)
        or not tensor_name
        or shader_match is None
    ):
        return DiscoveryResult(
            None,
            ("output transducer is not a supported BF16 tied linear projection",),
            evidence_ids,
        )
    vocabulary_size, hidden_size = map(int, shape)
    tensor = resolver.resolve_tensor(tensor_name)
    if (
        tensor.metadata["dtype"] != "BF16"
        or tensor.metadata["shape"] != shape
        or tensor.metadata.get("layout") != "row_major"
        or tensor.payload_byte_count != vocabulary_size * hidden_size * 2
        or hidden_size % 128
    ):
        return DiscoveryResult(
            None,
            ("output projection tensor has unsupported storage geometry",),
            evidence_ids,
        )
    parameter_tensors = {
        record["parameter_ref_id"]: record["definition"].get("tensor")
        for record in scope["boundary"]["parameters"]
    }
    exact_refs = tuple(
        sorted(str(path) for path in contract["exact_reference"]["artifact_refs"])
    )
    circuit_matches = []
    for path in exact_refs:
        if not path.endswith("/circuit.json"):
            continue
        document = _json_object(resolver.read_path(path), path)
        if document.get("source", {}).get("component_id") == component_id:
            circuit_matches.append((path, document))
    if len(circuit_matches) != 1:
        raise ModelCompileError(
            "output transducer has no unique exact circuit artifact"
        )
    circuit_ref, circuit = circuit_matches[0]
    refs = circuit.get("parameters", {}).get("refs", {})
    nodes = circuit.get("nodes")
    if not isinstance(refs, dict) or not isinstance(nodes, list):
        return DiscoveryResult(
            None,
            ("output transducer circuit has no structural parameter graph",),
            evidence_ids,
        )
    projection_parameter_ref_id = _unique_parameter_ref_for_tensor(
        refs,
        tensor_name,
    )
    norm_parameter_ref_id = _unique_parameter_ref_for_tensor(
        refs,
        str(spec.get("norm_parameter_tensor", "")),
    )
    projection_nodes = [
        node
        for node in nodes
        if node.get("op") == "linear_projection"
        and projection_parameter_ref_id in node.get("params", [])
    ]
    if (
        len(projection_nodes) != 1
        or projection_nodes[0].get("id") not in node_ids
        or parameter_tensors.get(projection_parameter_ref_id) != tensor_name
        or parameter_tensors.get(norm_parameter_ref_id)
        != spec.get("norm_parameter_tensor")
    ):
        return DiscoveryResult(
            None,
            ("output scope parameter graph disagrees with its package ABI",),
            evidence_ids,
        )
    physical_node_id = str(projection_nodes[0]["id"])
    projection_scale_parameter_ref_id = (
        f"{projection_parameter_ref_id[:-7]}.weight_scale_inv"
        if projection_parameter_ref_id.endswith(".weight")
        else f"{projection_parameter_ref_id}_scale_inv"
    )
    speculative_decoder_ids = tuple(
        sorted(
            str(decoder["id"])
            for decoder in manifest.get("speculative_decoders", [])
            if (
                isinstance(decoder, dict)
                and isinstance(decoder.get("id"), str)
                and isinstance(decoder.get("output_transducer"), dict)
                and decoder["output_transducer"].get("projection_parameter_tensor")
                == tensor_name
            )
        )
    )
    return DiscoveryResult(
        OutputProjectionOpportunity(
            scope_id=str(scope["scope_id"]),
            source_contract_digest=str(contract["contract_digest"]),
            component_id=component_id,
            physical_node_id=physical_node_id,
            norm_parameter_ref_id=norm_parameter_ref_id,
            projection_parameter_ref_id=projection_parameter_ref_id,
            projection_scale_parameter_ref_id=(projection_scale_parameter_ref_id),
            source_node_ids=tuple(scope["members"]["source_node_ids"]),
            evidence_ids=evidence_ids,
            source_artifact_refs=exact_refs,
            manifest_ref=manifest_ref,
            circuit_ref=circuit_ref,
            tensor=tensor,
            norm_tensor_name=str(spec["norm_parameter_tensor"]),
            hidden_size=hidden_size,
            vocabulary_size=vocabulary_size,
            output_scale_token=shader_match.group(1),
            fp8_process_names=fp8_processes,
            speculative_decoder_ids=speculative_decoder_ids,
            max_context_activations=int(manifest["max_context_activations"]),
        ),
        (
            "discovered a standalone BF16 output projection on a target "
            "with native F8 E4M3 packed dot products",
        ),
        evidence_ids,
    )


def require_output_projection(
    context: ProviderContext,
) -> OutputProjectionOpportunity:
    opportunities = discover_output_projections(context)
    if len(opportunities) != 1:
        raise ModelCompileError(
            "block-scaled output provider requires exactly one compatible "
            f"output transducer, found {len(opportunities)}"
        )
    return opportunities[0]


def discover_output_projections(
    context: ProviderContext,
) -> tuple[OutputProjectionOpportunity, ...]:
    key = "output_fp8.v1:" + ",".join(context.scope_ids)
    return context.memoized(
        key,
        lambda: tuple(
            opportunity
            for scoped_context in context.single_scope_contexts()
            if (opportunity := discover_output_projection(scoped_context).opportunity)
            is not None
        ),
    )  # type: ignore[return-value]


def source_inputs(
    context: ProviderContext,
    opportunity: OutputProjectionOpportunity,
) -> list[Json]:
    resolver = context.source_artifacts
    artifacts = {source["path"]: source for source in opportunity.tensor.source_inputs}
    for path in (
        *opportunity.source_artifact_refs,
        opportunity.manifest_ref,
    ):
        artifact = resolver.resolve_path(path)
        artifacts[artifact.path] = artifact.source_input()
    return [artifacts[path] for path in sorted(artifacts)]


def _json_object(payload: bytes, label: str) -> Json:
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelCompileError(f"{label} is not valid JSON") from error
    if not isinstance(document, dict):
        raise ModelCompileError(f"{label} must be a JSON object")
    return document


def _unique_parameter_ref_for_tensor(
    refs: Json,
    tensor_name: str,
) -> str:
    matches = [
        parameter_ref_id
        for parameter_ref_id, definition in refs.items()
        if definition.get("tensor") == tensor_name
    ]
    if len(matches) != 1:
        raise ModelCompileError(
            f"output transducer tensor {tensor_name!r} has no unique parameter binding"
        )
    return str(matches[0])
