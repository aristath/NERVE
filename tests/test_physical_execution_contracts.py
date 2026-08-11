from copy import deepcopy
import json
from pathlib import Path

import pytest

from nerve.physical_execution_contracts import (
    PhysicalExecutionContractError,
    artifact_sha256,
    implementation_digest,
    seal_physical_execution_contract,
    validate_physical_execution_contract,
)


FIXTURE = (
    Path(__file__).parents[1]
    / "execution-contracts"
    / "fixtures"
    / "tensor_parallel_projection.json"
)


def fixture_contract() -> dict[str, object]:
    return json.loads(FIXTURE.read_text())


def test_shared_tensor_parallel_contract_is_valid() -> None:
    validate_physical_execution_contract(fixture_contract())


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(strategy="model_named_deepseek"), "strategy"),
        (lambda value: value.update(parameter_partitions=[]), "partition"),
        (
            lambda value: value["inputs"][0].update(
                distribution="sharded", dimension=0
            ),
            "alignment",
        ),
        (
            lambda value: value["outputs"][0].update(
                collection="reduced", reduction=None
            ),
            "dimension",
        ),
        (lambda value: value.update(model_name="deepseek"), "unknown"),
    ],
)
def test_invalid_or_model_specific_distribution_fails_closed(mutate, message: str) -> None:
    contract = fixture_contract()
    mutate(contract)
    with pytest.raises(PhysicalExecutionContractError, match=message):
        validate_physical_execution_contract(contract)


def test_lazy_resources_cannot_be_declared_permanent() -> None:
    contract = fixture_contract()
    contract["resources"][0].update(kind="lazy_resource")
    with pytest.raises(PhysicalExecutionContractError, match="demand resident"):
        validate_physical_execution_contract(contract)


def test_contract_sealing_is_deterministic_and_covers_semantics() -> None:
    contract = fixture_contract()
    contract.pop("contract_id")
    first = seal_physical_execution_contract(contract)
    second = seal_physical_execution_contract(deepcopy(contract))
    assert first["contract_id"] == second["contract_id"]

    changed = deepcopy(contract)
    changed["geometry"]["dimensions"]["output_rows"] += 128
    assert seal_physical_execution_contract(changed)["contract_id"] != first["contract_id"]


def test_implementation_digest_covers_artifact_phase_format_geometry_and_strategy() -> None:
    contract = fixture_contract()
    digest = implementation_digest(
        artifact=contract["artifact"],
        phases=contract["phases"],
        formats=contract["formats"],
        geometry=contract["geometry"],
        strategy=contract["strategy"],
    )
    assert digest.startswith("sha256:")
    assert digest != implementation_digest(
        artifact={
            **contract["artifact"],
            "sha256": artifact_sha256(b"different artifact"),
        },
        phases=contract["phases"],
        formats=contract["formats"],
        geometry=contract["geometry"],
        strategy=contract["strategy"],
    )
