from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from nerve.compilation import (
    Json,
    ModelCompileCancelled,
    ModelCompileError,
    check_compile_cancelled,
)
from nerve.representation_optimizer.contracts import (
    HARDWARE_PROCESS_PROFILE_SCHEMA,
    ContractDocument,
    stable_contract_id,
)


HARDWARE_PROCESS_INVENTORY_SCHEMA = "nerve.hardware_process_inventory.v1"
COMPILER_TARGET_SCHEMA = "nerve.compiler_target.v2"


@dataclass(frozen=True)
class CompilerTargetDevice:
    physical_device_index: int
    physical_device_id: str
    device_name: str
    device_type: str
    pci_address: str | None
    vendor_id: int
    device_id: int
    shader_features: frozenset[str]
    subgroup_operations: frozenset[str]
    subgroup_compute_supported: bool
    subgroup_size: int
    max_compute_work_group_invocations: int
    max_compute_work_group_size_x: int
    cooperative_float16_shapes: tuple[tuple[int, int, int], ...]
    cooperative_bfloat16_shapes: tuple[tuple[int, int, int], ...]
    cooperative_float8_e4m3_shapes: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        if self.physical_device_index < 0:
            raise ModelCompileError("Vulkan physical-device index must be non-negative")
        if not self.physical_device_id or not self.device_name or not self.device_type:
            raise ModelCompileError("Vulkan compiler target identity is incomplete")
        if self.vendor_id < 0 or self.device_id < 0:
            raise ModelCompileError("Vulkan vendor and device identifiers must be unsigned")
        if self.subgroup_size <= 0 or self.subgroup_size & (self.subgroup_size - 1):
            raise ModelCompileError(
                f"runtime returned invalid Vulkan subgroup size {self.subgroup_size}"
            )
        if (
            self.max_compute_work_group_invocations <= 0
            or self.max_compute_work_group_size_x <= 0
        ):
            raise ModelCompileError("Vulkan workgroup limits must be positive")

    @classmethod
    def from_json(cls, payload: Json) -> CompilerTargetDevice:
        expected_fields = {
            "physical_device_index",
            "physical_device_id",
            "device_name",
            "device_type",
            "pci_address",
            "vendor_id",
            "device_id",
            "shader_features",
            "subgroup_operations",
            "subgroup_compute_supported",
            "subgroup_size",
            "max_compute_work_group_invocations",
            "max_compute_work_group_size_x",
            "cooperative_float16_shapes",
            "cooperative_bfloat16_shapes",
            "cooperative_float8_e4m3_shapes",
        }
        if set(payload) != expected_fields:
            raise ModelCompileError(
                "runtime returned an invalid compiler target device: "
                f"expected fields {sorted(expected_fields)}, found {sorted(payload)}"
            )
        if not isinstance(payload.get("subgroup_compute_supported"), bool):
            raise ModelCompileError(
                "runtime returned an invalid compiler target device: "
                "subgroup_compute_supported must be boolean"
            )
        try:
            return cls(
                physical_device_index=int(payload["physical_device_index"]),
                physical_device_id=str(payload["physical_device_id"]),
                device_name=str(payload["device_name"]),
                device_type=str(payload["device_type"]),
                pci_address=(
                    str(payload["pci_address"])
                    if payload["pci_address"] is not None
                    else None
                ),
                vendor_id=int(payload["vendor_id"]),
                device_id=int(payload["device_id"]),
                shader_features=frozenset(
                    unique_string_values(
                        payload["shader_features"],
                        "shader_features",
                    )
                ),
                subgroup_operations=frozenset(
                    unique_string_values(
                        payload["subgroup_operations"],
                        "subgroup_operations",
                    )
                ),
                subgroup_compute_supported=payload["subgroup_compute_supported"],
                subgroup_size=int(payload["subgroup_size"]),
                max_compute_work_group_invocations=int(
                    payload["max_compute_work_group_invocations"]
                ),
                max_compute_work_group_size_x=int(
                    payload["max_compute_work_group_size_x"]
                ),
                cooperative_float16_shapes=cooperative_matrix_shapes(
                    payload, "cooperative_float16_shapes"
                ),
                cooperative_bfloat16_shapes=cooperative_matrix_shapes(
                    payload, "cooperative_bfloat16_shapes"
                ),
                cooperative_float8_e4m3_shapes=cooperative_matrix_shapes(
                    payload, "cooperative_float8_e4m3_shapes"
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ModelCompileError(
                f"runtime returned an invalid compiler target device: {payload!r}"
            ) from error

    def supports_native_dtype(self, dtype: str) -> bool:
        requirements = native_dtype_shader_features(dtype)
        return requirements is not None and requirements <= self.shader_features

    def to_json(self) -> Json:
        return {
            "physical_device_index": self.physical_device_index,
            "physical_device_id": self.physical_device_id,
            "device_name": self.device_name,
            "device_type": self.device_type,
            "pci_address": self.pci_address,
            "vendor_id": self.vendor_id,
            "device_id": self.device_id,
            "shader_features": sorted(self.shader_features),
            "subgroup_operations": sorted(self.subgroup_operations),
            "subgroup_compute_supported": self.subgroup_compute_supported,
            "subgroup_size": self.subgroup_size,
            "max_compute_work_group_invocations": (
                self.max_compute_work_group_invocations
            ),
            "max_compute_work_group_size_x": self.max_compute_work_group_size_x,
            "cooperative_float16_shapes": [
                list(shape) for shape in self.cooperative_float16_shapes
            ],
            "cooperative_bfloat16_shapes": [
                list(shape) for shape in self.cooperative_bfloat16_shapes
            ],
            "cooperative_float8_e4m3_shapes": [
                list(shape) for shape in self.cooperative_float8_e4m3_shapes
            ],
        }


@dataclass(frozen=True)
class CompilerTarget:
    devices: tuple[CompilerTargetDevice, ...]
    hardware_profiles: tuple[ContractDocument, ...]

    @classmethod
    def from_json(cls, payload: Json) -> CompilerTarget:
        if payload.get("schema") != COMPILER_TARGET_SCHEMA:
            raise ModelCompileError(
                f"unsupported compiler target schema {payload.get('schema')!r}"
            )
        target = cls._from_hardware_profiles(payload.get("hardware_profiles"))
        raw_devices = payload.get("devices")
        if not isinstance(raw_devices, list):
            raise ModelCompileError("compiler target has no derived device list")
        if raw_devices != [device.to_json() for device in target.devices]:
            raise ModelCompileError(
                "compiler target device list does not match its hardware profiles"
            )
        return target

    @classmethod
    def from_hardware_inventory_json(cls, payload: Json) -> CompilerTarget:
        if payload.get("schema") != HARDWARE_PROCESS_INVENTORY_SCHEMA:
            raise ModelCompileError(
                "unsupported hardware-process inventory schema "
                f"{payload.get('schema')!r}"
            )
        return cls._from_hardware_profiles(payload.get("profiles"))

    @classmethod
    def _from_hardware_profiles(
        cls,
        raw_profiles: object,
    ) -> CompilerTarget:
        if not isinstance(raw_profiles, list):
            raise ModelCompileError(
                "hardware-process inventory has no profile list"
            )
        profiles: list[ContractDocument] = []
        devices: list[CompilerTargetDevice] = []
        stable_device_ids: list[str] = []
        for raw_profile in raw_profiles:
            if not isinstance(raw_profile, dict):
                raise ModelCompileError(
                    f"runtime returned an invalid hardware profile: {raw_profile!r}"
                )
            profile = ContractDocument.from_json(
                raw_profile,
                expected_schema=HARDWARE_PROCESS_PROFILE_SCHEMA,
            )
            profile_json = profile.to_json()
            stable_device_ids.append(
                str(profile_json["hardware_identity"]["stable_device_id"])
            )
            profiles.append(profile)
            if profile_json["hardware_identity"]["device_kind"] != "gpu":
                continue
            raw_capabilities = profile_json["capability_extensions"].get(
                "vulkan_compiler_capabilities"
            )
            raw_binding = profile_json["runtime_bindings"].get(
                "vulkan_runtime_binding"
            )
            if not isinstance(raw_capabilities, dict) or not isinstance(
                raw_binding, dict
            ):
                raise ModelCompileError(
                    "GPU hardware profile has no Vulkan compiler capability or "
                    "runtime-binding view"
                )
            raw_device = {**raw_capabilities, **raw_binding}
            device = CompilerTargetDevice.from_json(raw_device)
            if (
                device.physical_device_id
                != profile_json["hardware_identity"]["stable_device_id"]
            ):
                raise ModelCompileError(
                    "GPU compiler target identity does not match its hardware profile"
                )
            if device.device_type in {
                "discrete_gpu",
                "integrated_gpu",
                "virtual_gpu",
            }:
                devices.append(device)
        if stable_device_ids != sorted(set(stable_device_ids)):
            raise ModelCompileError(
                "hardware profiles must have unique sorted stable device identities"
            )
        if not devices:
            raise ModelCompileError(
                "model compilation requires at least one Vulkan GPU target"
            )
        devices.sort(key=lambda device: device.physical_device_index)
        physical_indices = [
            device.physical_device_index for device in devices
        ]
        if physical_indices != sorted(set(physical_indices)):
            raise ModelCompileError(
                "GPU hardware profiles contain duplicate Vulkan physical indices"
            )
        return cls(devices=tuple(devices), hardware_profiles=tuple(profiles))

    @classmethod
    def for_features(cls, *feature_sets: Iterable[str]) -> CompilerTarget:
        devices = tuple(
            CompilerTargetDevice(
                physical_device_index=index,
                physical_device_id=f"test-device-{index}",
                device_name=f"test device {index}",
                device_type="discrete_gpu",
                pci_address=None,
                vendor_id=0,
                device_id=index,
                shader_features=frozenset(features),
                subgroup_operations=frozenset(),
                subgroup_compute_supported=True,
                subgroup_size=64,
                max_compute_work_group_invocations=1024,
                max_compute_work_group_size_x=1024,
                cooperative_float16_shapes=(),
                cooperative_bfloat16_shapes=(),
                cooperative_float8_e4m3_shapes=(),
            )
            for index, features in enumerate(feature_sets)
        )
        if not devices:
            raise ValueError("a compiler target requires at least one device")
        profiles = tuple(
            ContractDocument.from_json(synthetic_hardware_profile(device))
            for device in devices
        )
        return cls(devices=devices, hardware_profiles=profiles)

    def supports_native_dtype(self, dtype: str) -> bool:
        return any(device.supports_native_dtype(dtype) for device in self.devices)

    def to_json(self) -> Json:
        return {
            "schema": COMPILER_TARGET_SCHEMA,
            "hardware_profiles": [
                profile.to_json() for profile in self.hardware_profiles
            ],
            "devices": [device.to_json() for device in self.devices],
        }


def native_dtype_shader_features(dtype: str) -> frozenset[str] | None:
    return {
        "F32": frozenset(),
        "F16": frozenset({"shader_float16"}),
        "BF16": frozenset({"shader_bfloat16_type"}),
        "F8_E4M3": frozenset(
            {
                "shader_float8",
                "shader_mixed_float_dot_product_float8_acc_float32",
            }
        ),
    }.get(dtype)


def cooperative_matrix_shapes(
    payload: Json, field: str
) -> tuple[tuple[int, int, int], ...]:
    raw_shapes = payload[field]
    if not isinstance(raw_shapes, list):
        raise ModelCompileError(
            f"runtime compiler target device {field!r} must be a list"
        )
    shapes: list[tuple[int, int, int]] = []
    for raw_shape in raw_shapes:
        if (
            not isinstance(raw_shape, list)
            or len(raw_shape) != 3
            or any(
                not isinstance(dimension, int)
                or isinstance(dimension, bool)
                or dimension <= 0
                for dimension in raw_shape
            )
        ):
            raise ModelCompileError(
                f"runtime compiler target device has invalid {field!r}: "
                f"{raw_shapes!r}"
            )
        shapes.append(tuple(raw_shape))
    if shapes != sorted(set(shapes)):
        raise ModelCompileError(
            f"runtime compiler target device {field!r} must be unique and sorted"
        )
    return tuple(shapes)


def unique_string_values(payload: object, field: str) -> list[str]:
    if (
        not isinstance(payload, list)
        or any(not isinstance(value, str) or not value for value in payload)
        or len(payload) != len(set(payload))
    ):
        raise ModelCompileError(
            f"runtime compiler target device {field!r} must contain unique strings"
        )
    return payload


def synthetic_hardware_profile(device: CompilerTargetDevice) -> Json:
    is_gpu = device.device_type in {
        "discrete_gpu",
        "integrated_gpu",
        "virtual_gpu",
    }
    identity: Json = {
        "device_kind": "gpu" if is_gpu else "cpu",
        "stable_device_id": device.physical_device_id,
        "name": device.device_name,
        "vendor_id": f"0x{device.vendor_id:04x}",
        "device_id": f"0x{device.device_id:04x}",
        "architecture": (
            f"synthetic_vendor_{device.vendor_id:04x}_device_{device.device_id:04x}"
        ),
        "physical_location": (
            f"pci:{device.pci_address}"
            if device.pci_address is not None
            else device.physical_device_id
        ),
    }
    process: Json = {
        "name": "shader_arithmetic",
        "category": "arithmetic",
        "availability": "available",
        "programmability": "direct",
        "api": "vulkan",
        "operations": ["add", "multiply"],
        "numeric_formats": synthetic_numeric_formats(device.shader_features),
        "required_extensions": [],
        "required_features": sorted(device.shader_features),
        "limits": {
            "max_compute_work_group_invocations": (
                device.max_compute_work_group_invocations
            ),
            "max_compute_work_group_size_x": device.max_compute_work_group_size_x,
            "subgroup_size": device.subgroup_size,
        },
        "properties": {},
    }
    memory_domain: Json = {
        "name": "synthetic_device_memory",
        "kind": "device_local_heap",
        "capacity_bytes": 1,
        "host_visible": False,
        "device_local": True,
        "coherent": False,
        "cached": False,
        "minimum_alignment_bytes": 1,
        "properties": {},
    }
    provenance: Json = {
        "api": "vulkan",
        "api_version": "1.4.0",
        "driver": "synthetic",
        "driver_version": "test",
        "compiler": "nerve-test",
        "operating_system": "synthetic",
        "discovery_backend": "test_fixture",
    }
    raw_device = device.to_json()
    identity_fields = {
        "physical_device_index",
        "physical_device_id",
        "device_name",
        "device_type",
        "vendor_id",
        "device_id",
        "pci_address",
    }
    capability_extensions: Json = (
        {
            "vulkan_compiler_capabilities": {
                key: value
                for key, value in raw_device.items()
                if key not in identity_fields
            }
        }
        if is_gpu
        else {}
    )
    runtime_bindings: Json = (
        {
            "vulkan_runtime_binding": {
                key: value
                for key, value in raw_device.items()
                if key in identity_fields
            }
        }
        if is_gpu
        else {}
    )
    identity_extensions: Json = {}
    capability_class = stable_contract_id(
        "hardware_capability",
        {
            "device_kind": identity["device_kind"],
            "architecture": identity["architecture"],
            "processes": [process],
            "memory_domains": [memory_domain],
            "interconnects": [],
            "api": provenance["api"],
            "api_version": provenance["api_version"],
            "capability_extensions": capability_extensions,
        },
    )
    return {
        "schema": HARDWARE_PROCESS_PROFILE_SCHEMA,
        "profile_id": stable_contract_id(
            "hardware_profile",
            [
                identity,
                capability_class,
                provenance,
                identity_extensions,
                [],
            ],
        ),
        "hardware_identity": identity,
        "capability_class": capability_class,
        "processes": [process],
        "memory_domains": [memory_domain],
        "interconnects": [],
        "measurements": [],
        "provenance": provenance,
        "capability_extensions": capability_extensions,
        "identity_extensions": identity_extensions,
        "runtime_bindings": runtime_bindings,
    }


def synthetic_numeric_formats(features: frozenset[str]) -> list[str]:
    formats = {"f32", "i32", "u32"}
    for feature, values in {
        "shader_float16": {"f16"},
        "shader_float64": {"f64"},
        "shader_int8": {"i8", "u8"},
        "shader_int16": {"i16", "u16"},
        "shader_int64": {"i64", "u64"},
        "shader_bfloat16_type": {"bf16"},
        "shader_float8": {"f8_e4m3"},
    }.items():
        if feature in features:
            formats.update(values)
    return sorted(formats)


def discover_compiler_target(
    *,
    runtime_bin: Path | None = None,
    allowed_physical_device_ids: Iterable[str] = (),
    environment: dict[str, str] | None = None,
    initialize_device_contexts: bool = False,
    cancel_requested: Callable[[], bool] | None = None,
) -> CompilerTarget:
    check_compile_cancelled(cancel_requested)
    command = compiler_device_probe_command(
        runtime_bin=runtime_bin,
        allowed_physical_device_ids=allowed_physical_device_ids,
        initialize_device_contexts=initialize_device_contexts,
    )
    if cancel_requested is None:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
    else:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
        while True:
            try:
                stdout, stderr = process.communicate(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                try:
                    check_compile_cancelled(cancel_requested)
                except ModelCompileCancelled:
                    process.kill()
                    process.communicate()
                    raise
        completed = subprocess.CompletedProcess(
            command,
            process.returncode,
            stdout,
            stderr,
        )
    check_compile_cancelled(cancel_requested)
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip() or completed.stdout.strip()
        raise ModelCompileError(
            "could not discover GPU compiler capabilities"
            + (f": {diagnostic}" if diagnostic else "")
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ModelCompileError(
            "runtime returned invalid JSON while discovering compiler capabilities"
        ) from error
    if not isinstance(payload, dict):
        raise ModelCompileError(
            "runtime returned a non-object compiler capability report"
        )
    return CompilerTarget.from_hardware_inventory_json(payload)


def compiler_device_probe_command(
    *,
    runtime_bin: Path | None = None,
    allowed_physical_device_ids: Iterable[str] = (),
    initialize_device_contexts: bool = False,
) -> list[str]:
    allowed = tuple(sorted(set(allowed_physical_device_ids)))
    if any(not value.startswith("vulkan-uuid:") for value in allowed):
        raise ModelCompileError(
            "compiler device discovery requires stable Vulkan device identities"
        )
    configured = runtime_bin or runtime_bin_from_env()
    if configured is not None:
        return [
            str(configured),
            "--inspect-devices",
            "--json",
            *(["--initialize-device-contexts"] if initialize_device_contexts else []),
            *[
                value
                for device_id in allowed
                for value in ("--allow-physical-device", device_id)
            ],
        ]

    repo_root = Path(__file__).resolve().parents[1]
    cargo_manifest = repo_root / "runtime-rs" / "Cargo.toml"
    if cargo_manifest.is_file():
        command = [
            "cargo",
            "run",
            "--release",
            "--quiet",
            "--manifest-path",
            str(cargo_manifest),
            "--features",
            "vulkan tokenizers",
            "--bin",
            "nerve-runtime",
            "--",
            "--inspect-devices",
            "--json",
        ]
        if initialize_device_contexts:
            command.append("--initialize-device-contexts")
        command.extend(
            value
            for device_id in allowed
            for value in ("--allow-physical-device", device_id)
        )
        return command

    installed = shutil.which("nerve-runtime")
    if installed:
        return [
            installed,
            "--inspect-devices",
            "--json",
            *(["--initialize-device-contexts"] if initialize_device_contexts else []),
            *[
                value
                for device_id in allowed
                for value in ("--allow-physical-device", device_id)
            ],
        ]
    raise ModelCompileError(
        "could not find nerve-runtime for GPU compiler-capability discovery"
    )


def runtime_bin_from_env() -> Path | None:
    raw = os.environ.get("NERVE_RUNTIME_BIN")
    return Path(raw).expanduser() if raw else None
