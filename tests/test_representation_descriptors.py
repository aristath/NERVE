from __future__ import annotations

import json
from copy import deepcopy
from typing import Callable

import pytest

from nerve.representation_optimizer import (
    REPRESENTATION_DESCRIPTOR_SCHEMA,
    ContractDocument,
    ContractValidationError,
    RepresentationDescriptorRegistry,
    load_builtin_representation_descriptors,
    representation_descriptor_id,
    validate_contract,
)


def builtin_documents() -> list[dict[str, object]]:
    return load_builtin_representation_descriptors().to_json()


def named_descriptor(name: str) -> dict[str, object]:
    for descriptor in builtin_documents():
        if descriptor["identity"]["name"] == name:
            return descriptor
    raise AssertionError(f"missing built-in representation descriptor {name!r}")


def with_current_id(document: dict[str, object]) -> dict[str, object]:
    document["descriptor_id"] = representation_descriptor_id(document)
    return document


def test_builtin_descriptor_catalog_is_canonical_and_covers_open_expression_space() -> None:
    documents = builtin_documents()
    names = {str(document["identity"]["name"]) for document in documents}

    assert names == {
        "block_scaled_numeric_parameter",
        "bounded_multiscale_state",
        "coarse_to_fine_evaluation",
        "generated_program_with_exceptions",
        "group_scaled_integer_parameter",
        "heterogeneous_composite_island",
        "hierarchical_output_construction",
        "indexed_search_with_exact_refinement",
        "lookup_codebook_circuit",
        "packed_symbolic_logic",
        "reconstructed_parameter_stream",
        "sampled_field_with_residual",
        "sparse_event_graph",
        "structured_transform_with_exceptions",
        "verified_correction_circuit",
    }
    for document in documents:
        parsed = ContractDocument.from_json(
            document,
            expected_schema=REPRESENTATION_DESCRIPTOR_SCHEMA,
        )
        assert parsed.to_json() == document
        assert document["descriptor_id"] == representation_descriptor_id(document)

    exactness = {str(document["behavioral"]["exactness"]) for document in documents}
    assert exactness == {"approximate", "exact"}


def test_structurally_different_descriptors_are_not_matrix_specific() -> None:
    lookup = named_descriptor("lookup_codebook_circuit")
    field = named_descriptor("sampled_field_with_residual")
    events = named_descriptor("sparse_event_graph")
    state = named_descriptor("bounded_multiscale_state")
    output = named_descriptor("hierarchical_output_construction")

    assert lookup["representations"]["parameters"][1]["kind"] == "lookup_entries"
    assert field["representations"]["parameters"][1]["kind"] == "sampled_grid_mesh_or_basis"
    assert events["representations"]["signals"][0]["kind"] == "active_coordinate_delta_or_message"
    assert state["representations"]["states"][0]["kind"] == "bounded_recurrent_state"
    assert output["execution"]["topologies"] == [
        "hierarchical_tree",
        "product_code_graph",
        "token_trie",
    ]

    examples = (lookup, field, events, state, output)
    process_names = {
        process["name"]
        for document in examples
        for process in document["hardware"]["compatible_processes"]
    }
    assert {
        "atomics",
        "cache_hierarchy",
        "rasterization",
        "resident_program",
        "scalar_integer",
        "texture_sampling",
    } <= process_names
    assert all("matrix" not in document["identity"]["name"] for document in examples)


def test_descriptor_registry_accepts_external_expression_and_future_responsibility() -> None:
    registry = load_builtin_representation_descriptors()
    external = deepcopy(named_descriptor("packed_symbolic_logic"))
    external["identity"] = {
        "namespace": "third.party",
        "name": "temporal_symbol_wave",
        "version": "2026.1",
    }
    external["summary"] = "A separately registered symbolic temporal-wave expression."
    external["responsibilities"]["may_express"] = sorted(
        [
            *external["responsibilities"]["may_express"],
            "future_temporal_resonance",
        ]
    )
    external["tags"] = sorted([*external["tags"], "third_party"])
    with_current_id(external)

    extended = registry.register(external)

    assert len(extended.descriptors) == len(registry.descriptors) + 1
    assert registry.matching_responsibility("future_temporal_resonance") == ()
    assert [
        item.to_json()["identity"]["name"]
        for item in extended.matching_responsibility("future_temporal_resonance")
    ] == ["temporal_symbol_wave"]
    assert extended.get(str(external["descriptor_id"])).to_json() == external


def test_registry_rejects_duplicate_id_and_identity_drift() -> None:
    registry = load_builtin_representation_descriptors()
    descriptor = named_descriptor("lookup_codebook_circuit")
    with pytest.raises(ContractValidationError, match="already registered"):
        registry.register(descriptor)

    drifted = deepcopy(descriptor)
    drifted["summary"] = "Changed content under an existing identity."
    with_current_id(drifted)
    with pytest.raises(ContractValidationError, match="identity .* different content"):
        registry.register(drifted)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda document: document["responsibilities"].__setitem__(
                "may_express",
                list(reversed(document["responsibilities"]["may_express"])),
            ),
            "unique sorted",
        ),
        (
            lambda document: document["construction"].__setitem__("phases", []),
            "construction phases are required",
        ),
        (
            lambda document: document["boundaries"]["cost_terms"][0].__setitem__(
                "directions",
                ["teleport"],
            ),
            "unsupported boundary direction",
        ),
        (
            lambda document: document["representations"]["signals"].append(
                deepcopy(document["representations"]["signals"][0])
            ),
            "unique sorted names",
        ),
    ],
)
def test_descriptor_schema_rejects_semantically_ambiguous_contracts(
    mutation: Callable[[dict[str, object]], None],
    message: str,
) -> None:
    document = deepcopy(named_descriptor("sampled_field_with_residual"))
    mutation(document)
    with_current_id(document)

    with pytest.raises(ContractValidationError, match=message):
        validate_contract(document)


def test_descriptor_content_drift_requires_a_new_canonical_identity() -> None:
    descriptor = named_descriptor("structured_transform_with_exceptions")
    descriptor["summary"] = "Semantically changed without updating descriptor identity."

    with pytest.raises(ContractValidationError, match="canonical descriptor content"):
        validate_contract(descriptor)


def test_approximate_descriptor_requires_error_and_correction_contracts() -> None:
    descriptor = deepcopy(named_descriptor("sampled_field_with_residual"))
    descriptor["behavioral"]["error_contract"] = None
    descriptor["correction_paths"] = []
    with_current_id(descriptor)
    with pytest.raises(ContractValidationError, match="requires an error contract"):
        validate_contract(descriptor)

    descriptor = deepcopy(named_descriptor("sampled_field_with_residual"))
    descriptor["correction_paths"] = []
    with_current_id(descriptor)
    with pytest.raises(ContractValidationError, match="requires a correction path"):
        validate_contract(descriptor)


def test_directory_loader_is_deterministic_and_fails_closed(tmp_path) -> None:
    first = deepcopy(named_descriptor("lookup_codebook_circuit"))
    second = deepcopy(named_descriptor("sparse_event_graph"))
    (tmp_path / "z.json").write_text(json.dumps(first))
    (tmp_path / "a.json").write_text(json.dumps(second))

    registry = RepresentationDescriptorRegistry.from_directory(tmp_path)
    assert [item["descriptor_id"] for item in registry.to_json()] == sorted(
        [str(first["descriptor_id"]), str(second["descriptor_id"])]
    )

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ContractValidationError, match="is empty"):
        RepresentationDescriptorRegistry.from_directory(empty)

    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / "bad.json").write_text("[1, 2, 3]")
    with pytest.raises(ContractValidationError, match="JSON object"):
        RepresentationDescriptorRegistry.from_directory(malformed)
