from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from nerve.compilation import Json
from nerve.representation_optimizer.contracts import (
    REPRESENTATION_DESCRIPTOR_SCHEMA,
    ContractDocument,
    ContractValidationError,
)


BUILTIN_DESCRIPTOR_DIR = Path(__file__).with_name("descriptors")


@dataclass(frozen=True)
class RepresentationDescriptorRegistry:
    """Immutable registry of data-defined representation families."""

    descriptors: tuple[ContractDocument, ...] = ()

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Json | ContractDocument],
    ) -> RepresentationDescriptorRegistry:
        registry = cls()
        for document in documents:
            registry = registry.register(document)
        return registry

    @classmethod
    def from_directory(
        cls,
        directory: Path,
    ) -> RepresentationDescriptorRegistry:
        if not directory.is_dir():
            raise ContractValidationError(
                f"representation descriptor directory does not exist: {directory}"
            )
        documents = []
        for path in sorted(directory.glob("*.json")):
            try:
                document = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                raise ContractValidationError(
                    f"could not load representation descriptor {path}: {error}"
                ) from error
            if not isinstance(document, dict):
                raise ContractValidationError(
                    f"representation descriptor {path} must contain a JSON object"
                )
            documents.append(document)
        if not documents:
            raise ContractValidationError(
                f"representation descriptor directory is empty: {directory}"
            )
        return cls.from_documents(documents)

    def register(
        self,
        document: Json | ContractDocument,
    ) -> RepresentationDescriptorRegistry:
        descriptor = (
            document
            if isinstance(document, ContractDocument)
            else ContractDocument.from_json(
                document,
                expected_schema=REPRESENTATION_DESCRIPTOR_SCHEMA,
            )
        )
        if descriptor.schema != REPRESENTATION_DESCRIPTOR_SCHEMA:
            raise ContractValidationError(
                "registry accepts only representation descriptor contracts"
            )
        descriptor_json = descriptor.to_json()
        descriptor_id = str(descriptor_json["descriptor_id"])
        identity = _identity_key(descriptor_json)
        for existing in self.descriptors:
            existing_json = existing.to_json()
            if existing_json["descriptor_id"] == descriptor_id:
                raise ContractValidationError(
                    f"representation descriptor {descriptor_id!r} is already registered"
                )
            if _identity_key(existing_json) == identity:
                raise ContractValidationError(
                    "representation descriptor identity "
                    f"{identity!r} is already registered with different content"
                )
        return RepresentationDescriptorRegistry(
            descriptors=tuple(
                sorted(
                    (*self.descriptors, descriptor),
                    key=lambda item: str(item.to_json()["descriptor_id"]),
                )
            )
        )

    def get(self, descriptor_id: str) -> ContractDocument:
        for descriptor in self.descriptors:
            if descriptor.to_json()["descriptor_id"] == descriptor_id:
                return descriptor
        raise KeyError(descriptor_id)

    def matching_responsibility(
        self,
        responsibility: str,
    ) -> tuple[ContractDocument, ...]:
        return tuple(
            descriptor
            for descriptor in self.descriptors
            if responsibility
            in descriptor.to_json()["responsibilities"]["may_express"]
        )

    def to_json(self) -> list[Json]:
        return [descriptor.to_json() for descriptor in self.descriptors]


def load_builtin_representation_descriptors() -> RepresentationDescriptorRegistry:
    return RepresentationDescriptorRegistry.from_directory(BUILTIN_DESCRIPTOR_DIR)


def _identity_key(document: Json) -> str:
    identity = document["identity"]
    return (
        f"{identity['namespace']}:{identity['name']}@{identity['version']}"
    )
