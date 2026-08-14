from nerve.model_package_common import *
from nerve.model_package_tensors import *
from nerve.model_package_packed_tensors import *
from nerve.model_package_artifact_layout import (
    pack_tensor_artifacts_by_affinity,
    validate_artifact_affinity_groups,
    write_atomic_tensor_affinity_bank,
)
from nerve.resource_residency_planning import (
    TENSOR_PARTITION_INTEGRITY_SCHEMA,
)
from nerve.chat_codec import CHAT_CODEC_FILE, compile_model_chat_codec


def stream_control_binding_for_node(circuit: Json, node: Json) -> int:
    state_view_signals = {
        output
        for producer in circuit["nodes"]
        if producer.get("op")
        in {
            "append_state_update",
            "conditional_append_state_update",
            "rolling_state_update",
        }
        for output in producer.get("outputs", [])
    }
    signal_bindings = [*node.get("inputs", []), *node.get("outputs", [])]
    state_view_binding_count = sum(
        signal in state_view_signals for signal in signal_bindings
    )
    return (
        len(node.get("inputs", []))
        + len(node.get("outputs", []))
        + len(node.get("params", []))
        + len(node.get("state_reads", []))
        + len(node.get("state_writes", []))
        + state_view_binding_count
    )


def stream_control_binding_from_artifact_path(path: str) -> int | None:
    match = re.search(r"__sc(\d+)\.(?:comp|spv)$", path)
    return int(match.group(1)) if match is not None else None


def copy_tokenizer_package(model_dir: Path, dest_dir: Path) -> Json:
    tokenizer_json = model_dir / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise ModelCompileError(
            f"source model does not contain required tokenizer file {tokenizer_json}"
        )

    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied_files = []
    for filename in TOKENIZER_PACKAGE_FILES:
        source = model_dir / filename
        if source.is_file():
            shutil.copy2(source, dest_dir / filename)
            copied_files.append(filename)

    if "chat_template.jinja" not in copied_files:
        tokenizer_config_path = model_dir / "tokenizer_config.json"
        if tokenizer_config_path.is_file():
            tokenizer_config = read_json(tokenizer_config_path)
            inline_template = tokenizer_config.get("chat_template")
            if isinstance(inline_template, str):
                (dest_dir / "chat_template.jinja").write_text(inline_template)
                copied_files.append("chat_template.jinja")

    chat_codec = compile_model_chat_codec(model_dir, dest_dir)
    if chat_codec is not None:
        copied_files.append(CHAT_CODEC_FILE)

    return {
        "path": TOKENIZER_PACKAGE_DIR,
        "files": copied_files,
        **({"chat_codec": CHAT_CODEC_FILE} if chat_codec is not None else {}),
    }


def write_runtime_config_package(model_graph: Json, package_dir: Path) -> None:
    token_ids = model_graph["token_ids"]
    write_json(
        package_dir / CONFIG_PACKAGE_FILE,
        {
            "schema": "nerve.runtime_model_config.v1",
            "bos_token_id": token_ids["bos"],
            "eos_token_id": token_ids["eos"],
            "pad_token_id": token_ids["pad"],
            "dimensions": model_graph["dimensions"],
            "numerics": model_graph["numerics"],
        },
    )


def referenced_tensor_index(
    tensor_index: Json,
    *,
    model_graph: Json,
    lowered_index: Json,
    lowered_dir: Path,
) -> Json:
    runtime_referenced = {
        model_graph["graph"]["input_transducer"]["params"]["weight"]["tensor"]
    }
    for component in model_graph["graph"]["output_transducer"]["components"]:
        runtime_referenced.update(
            ref["tensor"] for ref in component.get("params", {}).values()
        )
    for circuit_ref in all_lowered_circuit_refs(lowered_index):
        circuit = read_json(lowered_dir / circuit_ref["circuit"])
        runtime_referenced.update(
            ref["tensor"] for ref in circuit["parameters"]["refs"].values()
        )
    runtime_referenced.update(
        tensor_name
        for tensor_name, info in tensor_index["tensors"].items()
        if isinstance(info, dict) and info.get("physical_execution_only") is True
    )

    compile_dependencies: set[str] = set()
    for tensor_name in runtime_referenced:
        info = tensor_index["tensors"].get(tensor_name)
        quantization = info.get("quantization") if isinstance(info, dict) else None
        if not isinstance(quantization, dict):
            continue
        if (
            quantization.get("format") == "auto_gptq"
            and auto_gptq_zero_encoding(info) == AUTO_GPTQ_PER_GROUP_ZERO
        ):
            qzeros = quantization.get("qzeros")
            if isinstance(qzeros, str) and qzeros:
                compile_dependencies.add(qzeros)

    referenced = runtime_referenced | compile_dependencies
    missing = sorted(referenced - set(tensor_index["tensors"]))
    if missing:
        raise ModelCompileError(
            f"compiled circuit graph references missing tensors: {', '.join(missing)}"
        )
    selected = deepcopy(tensor_index)
    selected["tensors"] = {
        name: deepcopy(tensor_index["tensors"][name]) for name in sorted(referenced)
    }
    for tensor_name in compile_dependencies - runtime_referenced:
        selected["tensors"][tensor_name]["compile_only"] = True
    selected["totals"] = {
        "tensor_count": len(selected["tensors"]),
        "parameter_count": sum(
            int(info["parameter_count"]) for info in selected["tensors"].values()
        ),
        "byte_count": sum(
            int(info["byte_count"]) for info in selected["tensors"].values()
        ),
    }
    return selected


def all_lowered_circuit_refs(lowered_index: Json) -> list[Json]:
    refs = list(lowered_index["graph"]["circuits"])
    refs.extend(
        circuit_ref
        for draft in lowered_index.get("draft_execution_graphs", [])
        for circuit_ref in draft["circuits"]
    )
    return refs


def _supports_atomic_affinity_packaging(info: Json) -> bool:
    if info.get("source_parts"):
        return False
    derivation = info.get("derived")
    if derivation is not None and (
        not isinstance(derivation, dict)
        or derivation.get("kind")
        not in {
            BROADCAST_COLUMNS_TRANSPOSE_DERIVATION,
            "matrix_to_input_block_major",
            "transpose_2d",
        }
    ):
        return False
    quantization = info.get("quantization")
    return not (
        isinstance(quantization, dict)
        and quantization.get("format") == "auto_gptq"
        and auto_gptq_packing(info) == AUTO_GPTQ_INPUT_MAJOR_PACKING
        and auto_gptq_zero_encoding(info) == AUTO_GPTQ_PER_GROUP_ZERO
    )


def copy_tensor_package(
    tensor_index: Json,
    package_dir: Path,
    *,
    partition_counts: dict[str, int] | None = None,
    artifact_affinity_groups: list[list[str]] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> Json:
    weights_dir = package_dir / WEIGHTS_PACKAGE_DIR
    if weights_dir.exists():
        shutil.rmtree(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)

    if not tensor_index["tensors"]:
        raise ModelCompileError("tensor index does not declare any source_file entries")

    packaged = deepcopy(tensor_index)
    compiled_sources = []
    tensors = sorted(packaged["tensors"].items())
    total = len(tensors)
    derived_groups_written: set[str] = set()
    derived_tensors_written: set[str] = set()
    partition_counts = dict(partition_counts or {})
    unknown_partition_tensors = set(partition_counts).difference(packaged["tensors"])
    if unknown_partition_tensors:
        raise ModelCompileError(
            "partition plan references unknown tensors: "
            + ", ".join(sorted(unknown_partition_tensors))
        )
    affinity_groups = validate_artifact_affinity_groups(
        packaged["tensors"], artifact_affinity_groups
    )
    atomic_affinity_groups = [
        group
        for group in affinity_groups
        if all(
            _supports_atomic_affinity_packaging(packaged["tensors"][tensor_name])
            for tensor_name in group
        )
    ]
    deferred_affinity_groups = [
        group for group in affinity_groups if group not in atomic_affinity_groups
    ]
    atomic_affinity_group_by_tensor = {
        tensor_name: group for group in atomic_affinity_groups for tensor_name in group
    }
    atomic_affinity_groups_written: set[tuple[str, ...]] = set()
    partition_digest_payload = bytearray()
    partition_integrity_records: list[tuple[Json, int, int, int]] = []

    def record_partition_integrity(
        tensor_name: str,
        info: Json,
        partition_count: int,
        partition_digests: list[bytes],
    ) -> None:
        if len(partition_digests) != partition_count:
            raise ModelCompileError(
                f"selected tensor {tensor_name!r} emitted incomplete partition integrity"
            )
        digest_offset = len(partition_digest_payload)
        for partition_digest in partition_digests:
            if len(partition_digest) != 32:
                raise ModelCompileError(
                    f"selected tensor {tensor_name!r} emitted an invalid digest"
                )
            partition_digest_payload.extend(partition_digest)
        partition_integrity_records.append(
            (
                info,
                partition_count,
                int(info["byte_count"]) // partition_count,
                digest_offset,
            )
        )

    for index, (tensor_name, info) in enumerate(tensors, start=1):
        check_compile_cancelled(cancel_requested)
        atomic_group = atomic_affinity_group_by_tensor.get(tensor_name)
        if atomic_group is not None:
            group_key = tuple(atomic_group)
            if group_key not in atomic_affinity_groups_written:
                source_record, emitted, emitted_partition_digests = (
                    write_atomic_tensor_affinity_bank(
                        package_dir=package_dir,
                        tensor_names=atomic_group,
                        tensors=packaged["tensors"],
                        partition_counts=partition_counts,
                        cancel_requested=cancel_requested,
                    )
                )
                compiled_sources.append(source_record)
                for emitted_name, emitted_metadata in emitted.items():
                    emitted_info = packaged["tensors"][emitted_name]
                    emitted_info.update(emitted_metadata)
                    emitted_info["layout"] = ROW_MAJOR_LAYOUT
                    emitted_info.pop("derived", None)
                    emitted_info.pop("source_parts", None)
                    emitted_info.pop("source_header_bytes", None)
                    emitted_info.pop("layout_hint", None)
                    emitted_partition_count = partition_counts.get(emitted_name)
                    if emitted_partition_count is not None:
                        record_partition_integrity(
                            emitted_name,
                            emitted_info,
                            emitted_partition_count,
                            emitted_partition_digests[emitted_name],
                        )
                atomic_affinity_groups_written.add(group_key)
            if progress is not None:
                progress(index, total, tensor_name)
            continue
        if tensor_name in derived_tensors_written:
            continue
        if progress is not None:
            progress(index, total, tensor_name)
        if info.get("compile_only") is True:
            continue
        layout = ROW_MAJOR_LAYOUT
        digest = blake2s(tensor_name.encode("utf-8"), digest_size=8).hexdigest()
        destination = weights_dir / f"tensor_{digest}.safetensors"
        derivation = info.get("derived")
        partition_count = partition_counts.get(tensor_name)
        quantization = info.get("quantization")
        partition_digests: list[bytes] = []
        matrix_transform = (
            isinstance(derivation, dict)
            and derivation.get("kind")
            in {
                BROADCAST_COLUMNS_TRANSPOSE_DERIVATION,
                "matrix_to_input_block_major",
                "transpose_2d",
            }
        )
        if partition_count is not None and (
            (isinstance(derivation, dict) and not matrix_transform)
            or (
                isinstance(quantization, dict)
                and quantization.get("format") == "auto_gptq"
                and auto_gptq_packing(info) == AUTO_GPTQ_INPUT_MAJOR_PACKING
                and auto_gptq_zero_encoding(info) == AUTO_GPTQ_PER_GROUP_ZERO
            )
        ):
            raise ModelCompileError(
                f"selected tensor {tensor_name!r} requires a packaging transform "
                "that does not preserve independently verifiable partitions"
            )
        if (
            isinstance(derivation, dict)
            and derivation.get("kind") == "bf16_to_fp8_e4m3_scale"
        ):
            if str(derivation["group"]) not in derived_groups_written:
                raise ModelCompileError(
                    f"derived FP8 scale tensor {tensor_name!r} was visited before "
                    "its weight tensor"
                )
            continue
        if (
            isinstance(derivation, dict)
            and derivation.get("kind") == "bf16_to_fp8_e4m3"
        ):
            scale_tensor_name = str(derivation["scale_tensor"])
            scale_info = packaged["tensors"].get(scale_tensor_name)
            if not isinstance(scale_info, dict):
                raise ModelCompileError(
                    f"derived FP8 weight tensor {tensor_name!r} references missing "
                    f"scale tensor {scale_tensor_name!r}"
                )
            scale_digest = blake2s(
                scale_tensor_name.encode("utf-8"), digest_size=8
            ).hexdigest()
            scale_destination = weights_dir / f"tensor_{scale_digest}.safetensors"
            group_headers_and_digests = (
                write_compiled_derived_fp8_e4m3_output_projection(
                    weight_tensor_name=tensor_name,
                    weight_info=info,
                    weight_destination=destination,
                    scale_tensor_name=scale_tensor_name,
                    scale_info=scale_info,
                    scale_destination=scale_destination,
                    layout=layout,
                )
            )
            for emitted_name, emitted_destination in (
                (tensor_name, destination),
                (scale_tensor_name, scale_destination),
            ):
                emitted_info = packaged["tensors"][emitted_name]
                header_bytes, data_sha256 = group_headers_and_digests[emitted_name]
                relative_destination = relative_json_path(
                    package_dir, emitted_destination
                )
                emitted_info["source_file"] = relative_destination
                emitted_info["data_offsets"] = [0, int(emitted_info["byte_count"])]
                emitted_info["data_sha256"] = data_sha256
                emitted_info["layout"] = layout
                emitted_info.pop("derived", None)
                compiled_sources.append(
                    {
                        "path": relative_destination,
                        "safetensors_header_bytes": header_bytes,
                        "metadata": {
                            "format": "nerve",
                            "layout": layout,
                        },
                    }
                )
            derived_groups_written.add(str(derivation["group"]))
            derived_tensors_written.update({tensor_name, scale_tensor_name})
            continue
        if (
            isinstance(derivation, dict)
            and derivation.get("kind") == "fp8_e4m3_to_bf16"
        ):
            header_bytes, data_sha256 = write_compiled_derived_bf16_from_fp8_e4m3(
                tensor_name=tensor_name,
                info=info,
                destination=destination,
                layout=layout,
            )
            relative_destination = relative_json_path(package_dir, destination)
            info["source_file"] = relative_destination
            info["data_offsets"] = [0, int(info["byte_count"])]
            info["data_sha256"] = data_sha256
            info["layout"] = layout
            info.pop("derived", None)
            compiled_sources.append(
                {
                    "path": relative_destination,
                    "safetensors_header_bytes": header_bytes,
                    "metadata": {
                        "format": "nerve",
                        "layout": layout,
                    },
                }
            )
            continue
        if (
            isinstance(derivation, dict)
            and derivation.get("kind") == "fp8_e4m3_to_q8_0"
        ):
            header_bytes, data_sha256 = write_compiled_derived_q8_0_from_fp8_e4m3(
                tensor_name=tensor_name,
                info=info,
                destination=destination,
                layout=layout,
            )
            relative_destination = relative_json_path(package_dir, destination)
            info["source_file"] = relative_destination
            info["data_offsets"] = [0, int(info["byte_count"])]
            info["data_sha256"] = data_sha256
            info["layout"] = layout
            info.pop("derived", None)
            compiled_sources.append(
                {
                    "path": relative_destination,
                    "safetensors_header_bytes": header_bytes,
                    "metadata": {
                        "format": "nerve",
                        "layout": layout,
                    },
                }
            )
            continue
        if (
            isinstance(derivation, dict)
            and derivation.get("kind") == "fp8_channel_scale_to_block_grid"
        ):
            header_bytes, data_sha256 = write_compiled_block_grid_from_channel_scales(
                tensor_name=tensor_name,
                info=info,
                destination=destination,
                layout=layout,
            )
            relative_destination = relative_json_path(package_dir, destination)
            info["source_file"] = relative_destination
            info["data_offsets"] = [0, int(info["byte_count"])]
            info["data_sha256"] = data_sha256
            info["layout"] = layout
            info["safetensors_header_bytes"] = header_bytes
            info.pop("derived", None)
            compiled_sources.append(
                {
                    "path": relative_destination,
                    "safetensors_header_bytes": header_bytes,
                    "metadata": {
                        "format": "nerve",
                        "layout": layout,
                    },
                }
            )
            continue
        if matrix_transform:
            writer = (
                write_compiled_derived_broadcast_transpose
                if derivation.get("kind")
                == BROADCAST_COLUMNS_TRANSPOSE_DERIVATION
                else write_compiled_derived_matrix_reorder
            )
            header_bytes, data_sha256, partition_digests = (
                writer(
                    tensor_name=tensor_name,
                    info=info,
                    destination=destination,
                    layout=layout,
                    partition_count=partition_count,
                )
            )
        elif (
            isinstance(quantization, dict)
            and quantization.get("format") == "auto_gptq"
            and auto_gptq_packing(info) == AUTO_GPTQ_INPUT_MAJOR_PACKING
            and auto_gptq_zero_encoding(info) == AUTO_GPTQ_PER_GROUP_ZERO
        ):
            source = Path(info["source_file"])
            if not source.is_file():
                raise ModelCompileError(f"tensor source file does not exist: {source}")
            zero_name = str(quantization.get("qzeros") or "")
            zero_info = packaged["tensors"].get(zero_name)
            if not isinstance(zero_info, dict):
                raise ModelCompileError(
                    f"AutoGPTQ tensor {tensor_name!r} references missing zero "
                    f"tensor {zero_name!r}"
                )
            zero_source = Path(zero_info["source_file"])
            if not zero_source.is_file():
                raise ModelCompileError(
                    f"AutoGPTQ zero source file does not exist: {zero_source}"
                )
            header_bytes, data_sha256 = write_compiled_auto_gptq_fixed_zero_8(
                tensor_name=tensor_name,
                info=info,
                zero_info=zero_info,
                source=source,
                zero_source=zero_source,
                destination=destination,
                layout=layout,
                cancel_requested=cancel_requested,
            )
            quantization["zero_point_encoding"] = AUTO_GPTQ_FIXED_ZERO_8
            quantization.pop("execution_zero_point_encoding", None)
            quantization.pop("zero_point_add", None)
            quantization.pop("qzeros", None)
        elif info.get("source_parts"):
            header_bytes, data_sha256, partition_digests = (
                write_compiled_composite_tensor(
                    tensor_name=tensor_name,
                    info=info,
                    destination=destination,
                    layout=layout,
                    partition_count=partition_count,
                )
            )
        else:
            source = Path(info["source_file"])
            if not source.is_file():
                raise ModelCompileError(f"tensor source file does not exist: {source}")
            header_bytes, data_sha256, partition_digests = write_compiled_tensor(
                tensor_name=tensor_name,
                info=info,
                source=source,
                destination=destination,
                layout=layout,
                partition_count=partition_count,
            )
        if partition_count is not None:
            record_partition_integrity(
                tensor_name, info, partition_count, partition_digests
            )
        if isinstance(quantization, dict):
            quantization.pop("execution_zero_point_encoding", None)
        relative_destination = relative_json_path(package_dir, destination)
        info["source_file"] = relative_destination
        info["data_offsets"] = [0, int(info["byte_count"])]
        info["data_sha256"] = data_sha256
        info["layout"] = layout
        info["safetensors_header_bytes"] = header_bytes
        info.pop("derived", None)
        info.pop("source_parts", None)
        info.pop("source_header_bytes", None)
        info.pop("layout_hint", None)
        source_metadata = {
            "format": "nerve",
            "layout": layout,
        }
        if isinstance(quantization, dict) and quantization.get("format") == "auto_gptq":
            source_metadata["packing_layout"] = auto_gptq_packing(info)
            source_metadata["zero_point_encoding"] = auto_gptq_zero_encoding(info)
        compiled_sources.append(
            {
                "path": relative_destination,
                "safetensors_header_bytes": header_bytes,
                "metadata": source_metadata,
            }
        )

    if partition_integrity_records:
        integrity_path = package_dir / "integrity" / "resource_partitions.sha256"
        integrity_path.parent.mkdir(parents=True, exist_ok=True)
        integrity_path.write_bytes(bytes(partition_digest_payload))
        table_sha256 = sha256(partition_digest_payload).hexdigest()
        relative_integrity_path = relative_json_path(package_dir, integrity_path)
        for (
            info,
            partition_count,
            partition_byte_count,
            digest_offset,
        ) in partition_integrity_records:
            info["partition_integrity"] = {
                "schema": TENSOR_PARTITION_INTEGRITY_SCHEMA,
                "partition_axis": 0,
                "partition_count": partition_count,
                "partition_byte_count": partition_byte_count,
                "digest_table_path": relative_integrity_path,
                "digest_table_byte_offset": digest_offset,
                "digest_stride_bytes": 32,
                "table_sha256": table_sha256,
            }

    compiled_sources = pack_tensor_artifacts_by_affinity(
        package_dir=package_dir,
        tensors=packaged["tensors"],
        compiled_sources=compiled_sources,
        affinity_groups=deferred_affinity_groups,
        cancel_requested=cancel_requested,
    )

    packaged["tensors"] = {
        name: info
        for name, info in packaged["tensors"].items()
        if info.get("compile_only") is not True
    }
    packaged["totals"] = {
        "tensor_count": len(packaged["tensors"]),
        "parameter_count": sum(
            int(info["parameter_count"]) for info in packaged["tensors"].values()
        ),
        "byte_count": sum(
            int(info["byte_count"]) for info in packaged["tensors"].values()
        ),
    }
    packaged["source"] = {
        "packaged": True,
        "compiled": True,
        "weights_dir": WEIGHTS_PACKAGE_DIR,
        "weights_file": compiled_sources[0]["path"],
        "weights_files": compiled_sources,
    }

    write_json(package_dir / "tensors.json", packaged)
    return packaged
