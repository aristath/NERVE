from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.providers.types import ProviderContext
from nerve.representation_optimizer.providers.source_artifacts import (
    SourceTensorArtifact,
)


@dataclass(frozen=True)
class CodebookBranch:
    source_node_id: str
    parameter_ref_id: str
    tensor_name: str
    attrs: Json
    tensor: SourceTensorArtifact
    raw_values: tuple[int, ...]
    indices: bytes

    @property
    def index_storage_payload(self) -> bytes:
        return self.indices + bytes(-len(self.indices) % 4)


@dataclass(frozen=True)
class HeadNormCodebookOpportunity:
    scope_id: str
    component_id: str
    structure_evidence_id: str
    codebook_evidence_ids: tuple[str, ...]
    source_contract_digest: str
    source_artifact_refs: tuple[str, ...]
    circuit_ref: str
    manifest_ref: str
    physical_node_id: str
    physical_source_node_ids: tuple[str, ...]
    physical_attrs: Json
    branches: tuple[CodebookBranch, CodebookBranch]
    codebook_values: tuple[int, ...]
    codebook_payload: bytes
    codebook_payload_sha256: str

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    self.structure_evidence_id,
                    *self.codebook_evidence_ids,
                }
            )
        )

    @property
    def head_width(self) -> int:
        return len(self.branches[0].raw_values)

    @property
    def original_parameter_bytes(self) -> int:
        return sum(branch.tensor.payload_byte_count for branch in self.branches)

    @property
    def codebook_parameter_bytes(self) -> int:
        return len(self.codebook_storage_payload) + sum(
            len(branch.index_storage_payload) for branch in self.branches
        )

    @property
    def codebook_storage_payload(self) -> bytes:
        return self.codebook_payload + bytes(-len(self.codebook_payload) % 4)


@dataclass(frozen=True)
class DiscoveryResult:
    opportunity: HeadNormCodebookOpportunity | None
    reasons: tuple[str, ...]
    evidence_ids: tuple[str, ...] = ()


def discover_head_norm_codebook(context: ProviderContext) -> DiscoveryResult:
    context.checkpoint()
    if len(context.scopes) != 1 or len(context.source_contracts) != 1:
        return DiscoveryResult(
            None,
            ("provider requires one independently mountable semantic scope",),
        )
    scope = context.scopes[0]
    contract = context.source_contracts[0]
    component_ids = scope["members"]["component_ids"]
    if len(component_ids) != 1:
        return DiscoveryResult(
            None,
            ("provider requires all operators to belong to one component",),
        )
    profile = context.hardware_profile
    if (
        profile["hardware_identity"]["device_kind"] != "gpu"
        or profile["provenance"]["api"] != "vulkan"
    ):
        return DiscoveryResult(
            None,
            ("provider currently emits a Vulkan GPU implementation",),
        )
    structure = _supported_claim(context.evidence, "operator_structure")
    if structure is None:
        return DiscoveryResult(
            None,
            ("exact operator-structure evidence is absent",),
        )
    operators = structure["claim"]["facts"]["operators"]
    if len(operators) != 2 or any(
        operator["op"] != "rms_norm_per_head" for operator in operators
    ):
        return DiscoveryResult(
            None,
            ("scope is not a two-branch per-head RMS normalization module",),
            (structure["evidence_id"],),
        )
    component_id = str(component_ids[0])
    if any(operator.get("component_id") != component_id for operator in operators):
        return DiscoveryResult(
            None,
            ("operator evidence does not preserve one component identity",),
            (structure["evidence_id"],),
        )
    common = ("head_width", "eps", "weight_offset")
    if any(
        operators[0]["attrs"].get(field) != operators[1]["attrs"].get(field)
        for field in common
    ):
        return DiscoveryResult(
            None,
            ("head-normalization branches do not share numerical semantics",),
            (structure["evidence_id"],),
        )
    head_width = operators[0]["attrs"].get("head_width")
    if (
        isinstance(head_width, bool)
        or not isinstance(head_width, int)
        or head_width <= 0
        or head_width % 2
    ):
        return DiscoveryResult(
            None,
            ("head-normalization width is not a positive even integer",),
            (structure["evidence_id"],),
        )

    parameter_tensors = {
        str(record["parameter_ref_id"]): str(record["definition"]["tensor"])
        for record in scope["boundary"]["parameters"]
        if isinstance(record["definition"].get("tensor"), str)
    }
    if any(
        len(operator["params"]) != 1 or operator["params"][0] not in parameter_tensors
        for operator in operators
    ):
        return DiscoveryResult(
            None,
            ("operator parameters do not map exactly to scope tensor bindings",),
            (structure["evidence_id"],),
        )

    codebook = _supported_codebook_claims(context.evidence)
    tensor_names = {parameter_tensors[operator["params"][0]] for operator in operators}
    if set(codebook["claims"]) != tensor_names:
        cited = tuple(
            sorted(
                {
                    structure["evidence_id"],
                    *codebook["evidence_ids"],
                }
            )
        )
        return DiscoveryResult(
            None,
            ("every branch requires exact exhaustive low-entropy codebook evidence",),
            cited,
        )
    if any(
        claim["facts"]["observation"]["storage_dtype"] != "BF16"
        or claim["facts"]["observation"]["mode"] != "exhaustive"
        or not claim["exact"]
        for claim in codebook["claims"].values()
    ):
        return DiscoveryResult(
            None,
            ("codebook evidence is not exact exhaustive BF16 evidence",),
            tuple(
                sorted(
                    {
                        structure["evidence_id"],
                        *codebook["evidence_ids"],
                    }
                )
            ),
        )

    resolver = context.source_artifacts
    branch_inputs = []
    for operator in operators:
        context.checkpoint()
        parameter_ref_id = str(operator["params"][0])
        tensor_name = parameter_tensors[parameter_ref_id]
        tensor = resolver.resolve_tensor(tensor_name)
        if (
            tensor.metadata["dtype"] != "BF16"
            or tensor.metadata["shape"] != [head_width]
            or tensor.payload_byte_count != head_width * 2
        ):
            return DiscoveryResult(
                None,
                ("source tensor storage disagrees with operator geometry",),
                tuple(
                    sorted(
                        {
                            structure["evidence_id"],
                            *codebook["evidence_ids"],
                        }
                    )
                ),
            )
        payload = resolver.read_tensor_storage(tensor_name)
        values = tuple(
            int.from_bytes(payload[offset : offset + 2], "little")
            for offset in range(0, len(payload), 2)
        )
        branch_inputs.append((operator, parameter_ref_id, tensor_name, tensor, values))

    context.checkpoint()
    codebook_values = tuple(
        sorted({value for item in branch_inputs for value in item[4]})
    )
    if len(codebook_values) > 256:
        return DiscoveryResult(
            None,
            ("combined exact codebook does not fit in U8 addresses",),
            tuple(
                sorted(
                    {
                        structure["evidence_id"],
                        *codebook["evidence_ids"],
                    }
                )
            ),
        )
    addresses = {value: index for index, value in enumerate(codebook_values)}
    branch_by_local_id = {
        _local_node_id(component_id, str(item[0]["node_id"])): (
            item,
            bytes(addresses[value] for value in item[4]),
        )
        for item in branch_inputs
    }

    refs = tuple(
        sorted(str(value) for value in contract["exact_reference"]["artifact_refs"])
    )
    circuit_ref = _component_circuit_ref(resolver, refs, component_id)
    manifest_ref = "vulkan_resident_package.json"
    manifest = _json_object(resolver.read_path(manifest_ref), manifest_ref)
    component = _manifest_component(manifest, component_id)
    source_circuit = _json_object(resolver.read_path(circuit_ref), circuit_ref)
    if (
        source_circuit.get("source", {}).get("component_id") != component_id
        or component["circuit"].get("source", {}).get("component_id") != component_id
    ):
        raise ModelCompileError(
            "semantic and physical source circuits disagree on component identity"
        )
    physical_node = _physical_codebook_node(
        component["circuit"]["nodes"],
        set(branch_by_local_id),
    )
    if physical_node is None:
        return DiscoveryResult(
            None,
            (
                "source physical graph does not retain the two branches in one "
                "head-normalization/rope circuit",
            ),
            tuple(
                sorted(
                    {
                        structure["evidence_id"],
                        *codebook["evidence_ids"],
                    }
                )
            ),
        )
    ordered_local_ids = tuple(
        source_id
        for source_id in physical_node["attrs"]["compiled_from"]
        if source_id in branch_by_local_id
    )
    if len(ordered_local_ids) != 2:
        raise ModelCompileError(
            "physical head-normalization circuit branch mapping is ambiguous"
        )
    branches = []
    for local_id in ordered_local_ids:
        item, indices = branch_by_local_id[local_id]
        operator, parameter_ref_id, tensor_name, tensor, values = item
        branches.append(
            CodebookBranch(
                source_node_id=str(operator["node_id"]),
                parameter_ref_id=parameter_ref_id,
                tensor_name=tensor_name,
                attrs=dict(operator["attrs"]),
                tensor=tensor,
                raw_values=values,
                indices=indices,
            )
        )
    if list(physical_node["params"]) != [
        branch.parameter_ref_id for branch in branches
    ]:
        raise ModelCompileError(
            "physical head-normalization parameter order disagrees with "
            "semantic branch order"
        )
    payload = b"".join(value.to_bytes(2, "little") for value in codebook_values)
    return DiscoveryResult(
        opportunity=HeadNormCodebookOpportunity(
            scope_id=str(scope["scope_id"]),
            component_id=component_id,
            structure_evidence_id=str(structure["evidence_id"]),
            codebook_evidence_ids=tuple(codebook["evidence_ids"]),
            source_contract_digest=str(contract["contract_digest"]),
            source_artifact_refs=refs,
            circuit_ref=circuit_ref,
            manifest_ref=manifest_ref,
            physical_node_id=str(physical_node["id"]),
            physical_source_node_ids=tuple(
                str(value) for value in physical_node["attrs"]["compiled_from"]
            ),
            physical_attrs=dict(physical_node["attrs"]),
            branches=(branches[0], branches[1]),
            codebook_values=codebook_values,
            codebook_payload=payload,
            codebook_payload_sha256=sha256(payload).hexdigest(),
        ),
        reasons=(
            "two exact BF16 head-normalization parameters share one U8-addressable codebook",
        ),
        evidence_ids=tuple(
            sorted(
                {
                    structure["evidence_id"],
                    *codebook["evidence_ids"],
                }
            )
        ),
    )


def discover_head_norm_codebooks(
    context: ProviderContext,
) -> tuple[HeadNormCodebookOpportunity, ...]:
    """Discover each non-overlapping compatible component in a problem."""

    key = "head_norm_codebooks.v1:" + ",".join(context.scope_ids)
    return context.memoized(
        key,
        lambda: _discover_head_norm_codebooks_uncached(context),
    )  # type: ignore[return-value]


def _discover_head_norm_codebooks_uncached(
    context: ProviderContext,
) -> tuple[HeadNormCodebookOpportunity, ...]:
    by_component: dict[str, HeadNormCodebookOpportunity] = {}
    for scoped_context in context.single_scope_contexts():
        context.checkpoint()
        result = discover_head_norm_codebook(scoped_context)
        opportunity = result.opportunity
        if opportunity is None:
            continue
        previous = by_component.get(opportunity.component_id)
        if previous is None or opportunity.scope_id < previous.scope_id:
            by_component[opportunity.component_id] = opportunity
    return tuple(
        sorted(
            by_component.values(),
            key=lambda item: (item.scope_id, item.component_id),
        )
    )


def _supported_claim(evidence: tuple[Json, ...], kind: str) -> Json | None:
    matches = []
    for document in evidence:
        for claim in document["claims"]:
            if (
                claim["kind"] == kind
                and claim["status"] == "supported"
                and claim["exact"] is True
            ):
                matches.append(
                    {
                        "evidence_id": document["evidence_id"],
                        "claim": claim,
                    }
                )
    if len(matches) != 1:
        return None
    return matches[0]


def _supported_codebook_claims(evidence: tuple[Json, ...]) -> Json:
    claims = {}
    evidence_ids = []
    for document in evidence:
        found = False
        for claim in document["claims"]:
            if (
                claim["kind"] == "low_entropy_codebook"
                and claim["status"] == "supported"
                and claim["exact"] is True
            ):
                tensor = str(claim["facts"]["tensor"])
                if tensor in claims:
                    raise ModelCompileError(
                        f"duplicate exact codebook evidence for {tensor!r}"
                    )
                claims[tensor] = claim
                found = True
        if found:
            evidence_ids.append(str(document["evidence_id"]))
    return {
        "claims": claims,
        "evidence_ids": tuple(sorted(evidence_ids)),
    }


def _local_node_id(component_id: str, qualified_id: str) -> str:
    prefix = f"{component_id}/"
    if not qualified_id.startswith(prefix):
        raise ModelCompileError(
            f"operator {qualified_id!r} is not qualified by {component_id!r}"
        )
    return qualified_id[len(prefix) :]


def _component_circuit_ref(
    resolver,
    refs: tuple[str, ...],
    component_id: str,
) -> str:
    matches = []
    for reference in refs:
        if not reference.endswith("/circuit.json"):
            continue
        document = _json_object(resolver.read_path(reference), reference)
        if document.get("source", {}).get("component_id") == component_id:
            matches.append(reference)
    if len(matches) != 1:
        raise ModelCompileError(
            f"source contract does not identify one circuit for {component_id!r}"
        )
    return matches[0]


def _manifest_component(manifest: Json, component_id: str) -> Json:
    matches = [
        component
        for component in manifest.get("circuit_graph", {}).get("components", [])
        if component.get("component_id") == component_id
    ]
    if len(matches) != 1:
        raise ModelCompileError(
            f"resident package manifest has no unique component {component_id!r}"
        )
    return matches[0]


def _physical_codebook_node(
    nodes: list[Json],
    norm_node_ids: set[str],
) -> Json | None:
    matches = []
    for node in nodes:
        compiled_from = node.get("attrs", {}).get("compiled_from", [])
        if (
            node.get("op") == "parallel_head_norm_rope_2way"
            and norm_node_ids <= set(compiled_from)
            and len(node.get("params", [])) == 2
        ):
            matches.append(node)
    if len(matches) > 1:
        raise ModelCompileError(
            "source circuit contains ambiguous fused head-normalization nodes"
        )
    return matches[0] if matches else None


def _json_object(payload: bytes, label: str) -> Json:
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelCompileError(f"{label} is not valid JSON") from error
    if not isinstance(document, dict):
        raise ModelCompileError(f"{label} must be a JSON object")
    return document
