from __future__ import annotations

import ast
import re
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parents[1]
PYTHON_PRODUCTION_ROOT = REPOSITORY_ROOT / "nerve"
RUST_PRODUCTION_ROOT = REPOSITORY_ROOT / "runtime-rs" / "src"

# These are source-model identities, not reusable operator or representation
# names.  A new checkpoint may carry any of them as model-owned metadata, but
# NERVE production code must never use them to select compiler or runtime work.
MODEL_FAMILY_IDENTITIES = (
    "deepseek",
    "falcon",
    "gemma",
    "granite",
    "lfm",
    "llama",
    "mistral",
    "mixtral",
    "phi",
    "qwen",
    "smollm",
)


def _production_sources() -> tuple[Path, ...]:
    python_sources = tuple(sorted(PYTHON_PRODUCTION_ROOT.rglob("*.py")))
    rust_sources = tuple(
        path
        for path in sorted(RUST_PRODUCTION_ROOT.rglob("*.rs"))
        if path.name != "tests.rs" and "tests" not in path.parts
    )
    return python_sources + rust_sources


def test_production_sources_do_not_name_model_families() -> None:
    family_pattern = re.compile(
        r"(?<![a-z0-9])(?:"
        + "|".join(re.escape(name) for name in MODEL_FAMILY_IDENTITIES)
        + r")(?![a-z0-9])",
        re.IGNORECASE,
    )
    violations = []
    for path in _production_sources():
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            match = family_pattern.search(line)
            if match is not None:
                violations.append(
                    f"{path.relative_to(REPOSITORY_ROOT)}:{line_number}:"
                    f"{match.group(0)}"
                )

    assert violations == [], (
        "production behavior must be selected from structural contracts and "
        "hardware capabilities, never source-model identity:\n"
        + "\n".join(violations)
    )


def test_python_production_does_not_branch_on_identity_metadata() -> None:
    violations = []
    for path in sorted(PYTHON_PRODUCTION_ROOT.rglob("*.py")):
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            predicate = None
            if isinstance(node, (ast.If, ast.IfExp, ast.While)):
                predicate = node.test
            elif isinstance(node, ast.Match):
                predicate = node.subject
            if predicate is None:
                continue
            rendered = ast.unparse(predicate)
            if "model_type" in rendered or "architectures" in rendered:
                violations.append(
                    f"{path.relative_to(REPOSITORY_ROOT)}:{node.lineno}:"
                    f"{rendered}"
                )

    assert violations == [], (
        "model_type and architectures are package metadata, not compiler "
        "dispatch inputs:\n" + "\n".join(violations)
    )
