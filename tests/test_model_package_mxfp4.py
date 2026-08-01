from __future__ import annotations

import json
import struct
from pathlib import Path

from nerve.model_package_assets import copy_tensor_package
from nerve.model_transpiler_tensor_index import make_tensor_index


def _payload(path: Path) -> bytes:
    data = path.read_bytes()
    header_bytes = struct.unpack("<Q", data[:8])[0]
    return data[8 + header_bytes :]


def test_packages_mxfp4_experts_as_independent_byte_exact_resources(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "config.json").write_text('{"expert_dtype":"fp4"}')
    tensor_payloads = {
        "layers.0.ffn.experts.0.w1.weight": bytes(range(32)),
        "layers.0.ffn.experts.0.w1.scale": bytes(range(2)),
        "layers.0.ffn.experts.1.w1.weight": bytes(range(32, 64)),
        "layers.0.ffn.experts.1.w1.scale": bytes(range(2, 4)),
    }
    header: dict[str, object] = {}
    payload = bytearray()
    for name, values in tensor_payloads.items():
        start = len(payload)
        payload.extend(values)
        is_scale = name.endswith(".scale")
        header[name] = {
            "dtype": "F8_E8M0" if is_scale else "I8",
            "shape": [1, 2] if is_scale else [1, 32],
            "data_offsets": [start, len(payload)],
        }
    header_payload = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_payload += b" " * (-len(header_payload) % 8)
    (source_dir / "model.safetensors").write_bytes(
        struct.pack("<Q", len(header_payload)) + header_payload + payload
    )

    source_index = make_tensor_index(source_dir)
    assert source_index["totals"] == {
        "tensor_count": 4,
        "parameter_count": 132,
        "byte_count": 68,
    }
    package_dir = tmp_path / "package"
    packaged = copy_tensor_package(source_index, package_dir)

    packaged_paths: set[str] = set()
    for name, expected_payload in tensor_payloads.items():
        info = packaged["tensors"][name]
        packaged_paths.add(str(info["source_file"]))
        assert _payload(package_dir / str(info["source_file"])) == expected_payload
        if name.endswith(".weight"):
            assert info["dtype"] == "I8"
            assert info["shape"] == [1, 32]
            assert info["logical_shape"] == [1, 64]
            assert info["quantization"]["format"] == "mxfp4_e2m1"
            assert info["quantization"]["scales"] == name.replace(".weight", ".scale")
    assert len(packaged_paths) == len(tensor_payloads)
