from __future__ import annotations

import ast
import shutil
from pathlib import Path

from nerve.compiler_fingerprint import (
    COMPILER_FINGERPRINT_SCHEMA,
    compiler_source_inputs,
    package_compiler_fingerprint,
)


def _transitive_compiler_sources(compiler_dir: Path) -> set[str]:
    pending = ["model_compiler"]
    visited = set()
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        source = compiler_dir / f"{module}.py"
        assert source.is_file(), f"compiler dependency nerve.{module} is missing"
        visited.add(module)
        tree = ast.parse(source.read_text(), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported = (node.module,)
            elif isinstance(node, ast.Import):
                imported = tuple(alias.name for alias in node.names)
            else:
                continue
            for name in imported:
                if name.startswith("nerve."):
                    dependency = name.split(".", 1)[1].split(".", 1)[0]
                    if dependency not in visited:
                        pending.append(dependency)
    return {f"nerve/{module}.py" for module in visited}


def test_compiler_source_manifest_is_the_exact_transitive_compile_closure() -> None:
    compiler_dir = Path(__file__).parents[1] / "nerve"
    declared = {relative for relative, _path in compiler_source_inputs(compiler_dir)}

    assert declared == _transitive_compiler_sources(compiler_dir)
    assert "nerve/conversation_gate.py" not in declared
    assert "nerve/cli.py" not in declared


def test_fingerprint_ignores_noncompiler_modules_but_tracks_compiler_dependencies(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).parents[1]
    compiler_dir = tmp_path / "nerve"
    shader_dir = tmp_path / "shaders"
    compiler_dir.mkdir()
    shader_dir.mkdir()
    shutil.copy(
        repository_root / "nerve" / "compiler_sources.txt",
        compiler_dir / "compiler_sources.txt",
    )
    for _relative, source in compiler_source_inputs():
        shutil.copy(source, compiler_dir / source.name)
    (shader_dir / "kernel.comp").write_text("original shader")
    (compiler_dir / "conversation_gate.py").write_text("unrelated = 1\n")

    original = package_compiler_fingerprint(shader_dir, compiler_dir=compiler_dir)
    (compiler_dir / "conversation_gate.py").write_text("unrelated = 2\n")
    unrelated_change = package_compiler_fingerprint(
        shader_dir, compiler_dir=compiler_dir
    )
    compiler_source = compiler_dir / "model_compiler.py"
    compiler_source.write_text(compiler_source.read_text() + "\n# compiler change\n")
    compiler_change = package_compiler_fingerprint(
        shader_dir, compiler_dir=compiler_dir
    )

    assert original.startswith(f"{COMPILER_FINGERPRINT_SCHEMA}:")
    assert unrelated_change == original
    assert compiler_change != original


def test_fingerprint_tracks_shader_source_changes(tmp_path: Path) -> None:
    shader_dir = tmp_path / "shaders"
    shader_dir.mkdir()
    shader = shader_dir / "kernel.comp"
    shader.write_text("first")
    first = package_compiler_fingerprint(shader_dir)

    shader.write_text("second")
    second = package_compiler_fingerprint(shader_dir)

    assert second != first
