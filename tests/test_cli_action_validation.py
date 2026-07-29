from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from nerve.cli import main


def discoverable_source(root: Path) -> None:
    (root / "config.json").write_text(json.dumps({"model_type": "synthetic"}))
    (root / "model.safetensors").write_bytes(b"weights")
    (root / "tokenizer.json").write_text("{}")


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--discover-model", "{source}", "--chat"], "--chat is only supported with --run"),
        (
            ["--compile-model", "{source}", "--prompt", "ignored"],
            "--prompt is only supported with --run",
        ),
        (
            [
                "--discover-model",
                "{source}",
                "--speculative-draft-tokens",
                "2",
            ],
            (
                "--speculative-draft-tokens is only supported with --run "
                "or --optimize-model"
            ),
        ),
        (
            [
                "--run",
                "{source}/missing-package",
                "--prompt",
                "hello",
                "--compiled-model-dir",
                "{source}/ignored",
            ],
            "--compiled-model-dir is only supported with --compile-model",
        ),
    ],
)
def test_cli_rejects_options_owned_by_a_different_action_before_running_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    message: str,
) -> None:
    discoverable_source(tmp_path)
    rendered = [argument.replace("{source}", str(tmp_path)) for argument in arguments]
    monkeypatch.setattr(sys, "argv", ["nerve", *rendered])

    with pytest.raises(SystemExit) as exit_info:
        main()

    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


def test_cli_forwards_runtime_regime_to_optimizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def optimize(package: Path, **kwargs: object) -> object:
        captured["package"] = package
        captured.update(kwargs)
        return SimpleNamespace(
            optimization=SimpleNamespace(
                report={"status": "completed"},
                report_path=tmp_path / "run" / "report.json",
                output_package_dir=tmp_path / "optimized",
            ),
            targets=SimpleNamespace(
                targets=(SimpleNamespace(target_id="target"),),
            ),
        )

    monkeypatch.setattr(
        "nerve.cli.optimize_compiled_package",
        optimize,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nerve",
            "--optimize-model",
            str(tmp_path),
            "--speculative-draft-tokens",
            "2",
        ],
    )

    main()

    assert captured["package"] == tmp_path
    assert captured["speculative_draft_tokens"] == 2
