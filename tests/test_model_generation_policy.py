from __future__ import annotations

import json
from pathlib import Path

from nerve.model_transpiler_generation import load_source_generation_config


def _write_generation_config(model_dir: Path, value: dict[str, object]) -> None:
    (model_dir / "generation_config.json").write_text(
        json.dumps(value), encoding="utf-8"
    )


def test_explicit_machine_readable_sampling_policy_has_precedence(
    tmp_path: Path,
) -> None:
    _write_generation_config(
        tmp_path,
        {"do_sample": False, "eos_token_id": 7},
    )
    (tmp_path / "README.md").write_text(
        '--gen_kwargs "do_sample=True,temperature=1.0,top_p=0.95,top_k=20"',
        encoding="utf-8",
    )

    assert load_source_generation_config(tmp_path) == {
        "do_sample": False,
        "eos_token_id": 7,
    }


def test_unambiguous_documented_policy_fills_omitted_sampling_metadata(
    tmp_path: Path,
) -> None:
    _write_generation_config(
        tmp_path,
        {"_from_model_config": True, "eos_token_id": 248044},
    )
    documented = (
        '--gen_kwargs "do_sample=True,temperature=1.0,top_p=0.95,top_k=20,'
        "min_p=0.0,presence_penalty=1.5,repetition_penalty=1.0,"
        'max_gen_toks=65536,seed=<SEED>"'
    )
    (tmp_path / "README.md").write_text(
        f"first reproduction\n{documented}\nsecond reproduction\n{documented}\n",
        encoding="utf-8",
    )

    assert load_source_generation_config(tmp_path) == {
        "_from_model_config": True,
        "eos_token_id": 248044,
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    }


def test_generation_parameters_form_is_discovered_without_model_names(
    tmp_path: Path,
) -> None:
    (tmp_path / "MODEL_CARD.md").write_text(
        "generation_parameters={temperature:0.6,top_p:0.9,top_k:40,"
        "min_p:0.02,presence_penalty:0.5,repetition_penalty:1.05,seed:<SEED>}",
        encoding="utf-8",
    )

    assert load_source_generation_config(tmp_path) == {
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.9,
        "top_k": 40,
        "min_p": 0.02,
        "presence_penalty": 0.5,
        "repetition_penalty": 1.05,
    }


def test_conflicting_documented_profiles_are_not_guessed(tmp_path: Path) -> None:
    _write_generation_config(tmp_path, {"eos_token_id": 9})
    (tmp_path / "README.md").write_text(
        "\n".join(
            [
                '--gen_kwargs "do_sample=True,temperature=1.0,top_p=0.95,top_k=20"',
                '--gen_kwargs "do_sample=True,temperature=0.6,top_p=0.95,top_k=20"',
            ]
        ),
        encoding="utf-8",
    )

    assert load_source_generation_config(tmp_path) == {"eos_token_id": 9}


def test_malformed_or_non_sampling_examples_do_not_become_defaults(
    tmp_path: Path,
) -> None:
    (tmp_path / "README.md").write_text(
        "\n".join(
            [
                '--gen_kwargs "temperature=fast,top_p=0.95"',
                '--gen_kwargs "max_gen_toks=65536,seed=42"',
                "generation_parameters={top_k:-1,temperature:1.0}",
            ]
        ),
        encoding="utf-8",
    )

    assert load_source_generation_config(tmp_path) == {}
