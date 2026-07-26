from __future__ import annotations

from hashlib import sha256
from pathlib import Path


COMPILER_FINGERPRINT_SCHEMA = "nerve.package_compiler_sha256.v2"
COMPILER_SOURCE_MANIFEST = "compiler_sources.txt"


def compiler_source_inputs(compiler_dir: Path | None = None) -> tuple[tuple[str, Path], ...]:
    compiler_dir = compiler_dir or Path(__file__).resolve().parent
    repository_root = compiler_dir.parent
    manifest_path = compiler_dir / COMPILER_SOURCE_MANIFEST
    try:
        relative_paths = tuple(
            line.strip()
            for line in manifest_path.read_text().splitlines()
            if line.strip()
        )
    except OSError as error:
        raise RuntimeError(
            f"could not read compiler source manifest {manifest_path}: {error}"
        ) from error
    if not relative_paths:
        raise RuntimeError(f"compiler source manifest {manifest_path} is empty")
    if tuple(sorted(set(relative_paths))) != relative_paths:
        raise RuntimeError(
            f"compiler source manifest {manifest_path} must contain unique sorted paths"
        )

    inputs = []
    for relative_path in relative_paths:
        parts = Path(relative_path).parts
        if (
            len(parts) < 2
            or parts[0] != "nerve"
            or parts[-1] in {"", ".", ".."}
            or not parts[-1].endswith(".py")
            or any(part in {"", ".", ".."} for part in parts)
        ):
            raise RuntimeError(
                f"invalid compiler source path {relative_path!r} in {manifest_path}"
            )
        source_path = repository_root / relative_path
        if not source_path.is_file():
            raise RuntimeError(
                f"compiler source {relative_path!r} declared by {manifest_path} is missing"
            )
        inputs.append((relative_path, source_path))
    return tuple(inputs)


def package_compiler_fingerprint(
    shader_source_dir: Path,
    *,
    compiler_dir: Path | None = None,
) -> str:
    inputs = [
        *compiler_source_inputs(compiler_dir),
        *(
            (f"runtime-rs/shaders/{path.name}", path)
            for path in shader_source_dir.iterdir()
            if path.is_file()
        ),
    ]
    digest = sha256()
    for relative_path, source_path in sorted(inputs):
        path_bytes = relative_path.encode("utf-8")
        source_bytes = source_path.read_bytes()
        digest.update(len(path_bytes).to_bytes(8, "little"))
        digest.update(path_bytes)
        digest.update(len(source_bytes).to_bytes(8, "little"))
        digest.update(source_bytes)
    return f"{COMPILER_FINGERPRINT_SCHEMA}:{digest.hexdigest()}"
