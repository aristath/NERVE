from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from nerve.compilation import ModelCompileError
from nerve.representation_optimizer.validation.contracts import (
    ProofResult,
    ValidationPlan,
)
from nerve.representation_optimizer.validation.protocols import (
    ExactProofVerifier,
    ProofRequest,
)


@dataclass(frozen=True)
class ProofVerifierRegistry:
    _verifiers: tuple[ExactProofVerifier, ...]

    @classmethod
    def from_verifiers(
        cls,
        verifiers: Iterable[ExactProofVerifier],
    ) -> ProofVerifierRegistry:
        ordered = sorted(verifiers, key=lambda verifier: verifier.verifier_id)
        identities = [verifier.verifier_id for verifier in ordered]
        if (
            any(not identity for identity in identities)
            or len(identities) != len(set(identities))
        ):
            raise ModelCompileError(
                "proof verifier identities must be non-empty and unique"
            )
        return cls(tuple(ordered))

    def prove(self, plan: ValidationPlan) -> tuple[ProofResult, ...]:
        document = plan.to_json()
        by_id = {
            verifier.verifier_id: verifier
            for verifier in self._verifiers
        }
        results = []
        for requirement in plan.proofs:
            verifier_id = str(requirement["verifier_id"])
            verifier = by_id.get(verifier_id)
            if verifier is None:
                raise ModelCompileError(
                    f"validation proof verifier {verifier_id!r} is unavailable"
                )
            request = ProofRequest(
                plan_id=plan.plan_id,
                candidate_id=plan.candidate_id,
                obligation=str(requirement["obligation"]),
                verifier_id=verifier_id,
                source_contract_digests=tuple(
                    document["source_contract_digests"]
                ),
                construction_record_digest=str(
                    document["construction_record_digest"]
                ),
                reference_implementation=plan.implementation("reference"),
                candidate_implementation=plan.implementation("candidate"),
            )
            result = ProofResult.from_json(verifier.verify(request))
            result_document = result.to_json()
            if (
                result_document["plan_id"] != request.plan_id
                or result_document["candidate_id"] != request.candidate_id
                or result_document["obligation"] != request.obligation
                or result_document["verifier_id"] != request.verifier_id
                or result_document["source_contract_digests"]
                != list(request.source_contract_digests)
                or result_document["construction_record_digest"]
                != request.construction_record_digest
            ):
                raise ModelCompileError(
                    "proof verifier returned evidence for a different request"
                )
            results.append(result)
        return tuple(results)

    def iter_proof_artifact(
        self,
        verifier_id: str,
        relative_path: str,
        *,
        chunk_bytes: int = 8 * 1024 * 1024,
    ):
        matching = [
            verifier
            for verifier in self._verifiers
            if verifier.verifier_id == verifier_id
        ]
        if len(matching) != 1:
            raise ModelCompileError(
                f"proof verifier {verifier_id!r} is unavailable"
            )
        reader = getattr(matching[0], "iter_proof_artifact", None)
        if not callable(reader):
            raise ModelCompileError(
                f"proof verifier {verifier_id!r} cannot stream its "
                "declared proof artifacts"
            )
        yield from reader(relative_path, chunk_bytes=chunk_bytes)
