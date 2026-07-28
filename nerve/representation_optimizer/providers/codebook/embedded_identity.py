from __future__ import annotations

from hashlib import sha256

from nerve.compilation import Json, ModelCompileError


EMBEDDED_PARAMETER_PROGRAM_DIGEST_PREFIX = (
    "nerve.optimizer.embedded_parameter_program_sha256.v1:"
)


def embedded_parameter_program_digest(branch_values: Json) -> str:
    if not isinstance(branch_values, list) or not branch_values:
        raise ModelCompileError(
            "embedded parameter program requires at least one BF16 branch"
        )
    digest = sha256(b"nerve.optimizer.embedded_parameter_program.v1\0")
    for branch in branch_values:
        if not isinstance(branch, list) or not branch:
            raise ModelCompileError(
                "embedded parameter program branch must be a non-empty sequence"
            )
        digest.update(len(branch).to_bytes(8, "little"))
        for value in branch:
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value > 0xFFFF
            ):
                raise ModelCompileError(
                    "embedded parameter program contains a non-BF16 bit pattern"
                )
            digest.update(value.to_bytes(2, "little"))
    return EMBEDDED_PARAMETER_PROGRAM_DIGEST_PREFIX + digest.hexdigest()
