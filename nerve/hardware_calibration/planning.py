from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Iterable

from nerve.compilation import Json
from nerve.representation_optimizer.contracts import (
    HARDWARE_PROCESS_PROFILE_SCHEMA,
    stable_contract_id,
    validate_contract,
)

from .contracts import CALIBRATION_PLAN_SCHEMA, validate_calibration_plan


@dataclass(frozen=True)
class CalibrationPolicy:
    warmup_iterations: int = 3
    steady_iterations: int = 11
    minimum_sample_duration_ns: int = 50_000_000
    sustained_window_duration_ms: int = 1_000
    sustained_window_count: int = 8
    confidence_level_ppm: int = 950_000
    maximum_relative_ci_width_ppm: int = 100_000

    def to_json(self) -> Json:
        return asdict(self)


@dataclass(frozen=True)
class Work:
    items_per_iteration: int
    operations_per_iteration: int
    bytes_read_per_iteration: int
    bytes_written_per_iteration: int

    def to_json(self) -> Json:
        return asdict(self)


@dataclass(frozen=True)
class WorkloadSpec:
    executor: str
    operation: str
    process_names: tuple[str, ...]
    regime: tuple[tuple[str, str], ...]
    work: Work
    artifacts: tuple[tuple[str, str], ...] = ()
    validation_mode: str = "digest"
    maximum_error_ppm: int = 0

    def to_json(self) -> Json:
        process_names = sorted(set(self.process_names))
        regime = dict(sorted(self.regime))
        artifacts = [
            {"name": name, "kind": kind, "digest": None}
            for name, kind in sorted(set(self.artifacts))
        ]
        validation = {
            "mode": self.validation_mode,
            "expected_digest": None,
            "maximum_error_ppm": self.maximum_error_ppm,
        }
        work = self.work.to_json()
        workload_id = stable_contract_id(
            "calibration_workload",
            process_names,
            self.executor,
            self.operation,
            regime,
            work,
            artifacts,
            validation,
        )
        return {
            "workload_id": workload_id,
            "process_names": process_names,
            "executor": self.executor,
            "operation": self.operation,
            "regime": regime,
            "work": work,
            "artifacts": artifacts,
            "validation": validation,
        }


ProcessMatcher = Callable[[Json], bool]
WorkloadBuilder = Callable[[Json, Json], Iterable[WorkloadSpec]]


@dataclass(frozen=True)
class CalibrationProvider:
    name: str
    matcher: ProcessMatcher
    builder: WorkloadBuilder


def build_calibration_plan(
    profile: Json,
    *,
    implementation_fingerprint: str,
    policy: CalibrationPolicy | None = None,
) -> Json:
    validate_contract(profile, expected_schema=HARDWARE_PROCESS_PROFILE_SCHEMA)
    selected_policy = policy or CalibrationPolicy()
    implementation = {
        "name": "nerve-hardware-calibrator",
        "version": "1",
        "fingerprint": implementation_fingerprint,
    }
    provided: dict[tuple[object, ...], WorkloadSpec] = {}
    covered_processes: set[str] = set()
    excluded: list[Json] = []
    for process in profile["processes"]:
        if (
            process["availability"] == "unavailable"
            or process["programmability"] == "none"
        ):
            excluded.append(
                {
                    "process_name": process["name"],
                    "reason": (
                        "unavailable"
                        if process["availability"] == "unavailable"
                        else "not_programmable"
                    ),
                }
            )
            continue
        provider = next(
            (candidate for candidate in _PROVIDERS if candidate.matcher(process)),
            None,
        )
        if provider is None:
            raise ValueError(
                "no calibration provider covers exposed hardware process "
                f"{process['name']!r} ({process['category']!r})"
            )
        specs = list(provider.builder(process, profile))
        if not specs:
            raise ValueError(
                f"calibration provider {provider.name!r} produced no workloads for "
                f"{process['name']!r}"
            )
        for spec in specs:
            if process["name"] not in spec.process_names:
                raise ValueError(
                    f"calibration provider {provider.name!r} did not associate "
                    f"workload {spec.operation!r} with {process['name']!r}"
                )
            key = _workload_equivalence_key(spec)
            if key in provided:
                existing = provided[key]
                provided[key] = WorkloadSpec(
                    executor=existing.executor,
                    operation=existing.operation,
                    process_names=tuple(
                        sorted(set(existing.process_names) | set(spec.process_names))
                    ),
                    regime=existing.regime,
                    work=existing.work,
                    artifacts=existing.artifacts,
                    validation_mode=existing.validation_mode,
                    maximum_error_ppm=existing.maximum_error_ppm,
                )
            else:
                provided[key] = spec
            covered_processes.add(process["name"])

    required_processes = {
        process["name"]
        for process in profile["processes"]
        if process["availability"] != "unavailable"
        and process["programmability"] != "none"
    }
    missing = sorted(required_processes - covered_processes)
    if missing:
        raise ValueError(f"calibration plan leaves hardware processes uncovered: {missing}")
    workloads = sorted(
        (spec.to_json() for spec in provided.values()),
        key=lambda workload: workload["workload_id"],
    )
    exclusions = sorted(excluded, key=lambda exclusion: exclusion["process_name"])
    policy_document = selected_policy.to_json()
    plan_id = stable_contract_id(
        "calibration_plan",
        profile["profile_id"],
        profile["capability_class"],
        implementation,
        policy_document,
        workloads,
        exclusions,
    )
    plan = {
        "schema": CALIBRATION_PLAN_SCHEMA,
        "plan_id": plan_id,
        "hardware_profile_id": profile["profile_id"],
        "capability_class": profile["capability_class"],
        "implementation": implementation,
        "policy": policy_document,
        "workloads": workloads,
        "excluded_processes": exclusions,
    }
    validate_calibration_plan(plan)
    return plan


def _workload_equivalence_key(spec: WorkloadSpec) -> tuple[object, ...]:
    return (
        spec.executor,
        spec.operation,
        spec.regime,
        spec.work,
        spec.artifacts,
        spec.validation_mode,
        spec.maximum_error_ppm,
    )


def _named(*names: str) -> ProcessMatcher:
    accepted = frozenset(names)
    return lambda process: process["name"] in accepted


def _cpu_scalar(process: Json, _profile: Json) -> Iterable[WorkloadSpec]:
    yield WorkloadSpec(
        executor="cpu",
        operation=process["name"],
        process_names=(process["name"],),
        regime=(("dependency", "independent_chains"), ("working_set", "registers")),
        work=Work(1_048_576, 8_388_608, 0, 8),
        validation_mode="exact",
    )


def _cpu_branch(process: Json, _profile: Json) -> Iterable[WorkloadSpec]:
    for predictability in ("predictable", "alternating", "data_dependent"):
        yield WorkloadSpec(
            executor="cpu",
            operation="branch_dispatch",
            process_names=(process["name"],),
            regime=(
                ("predictability", predictability),
                ("working_set", "l1_resident"),
            ),
            work=Work(1_048_576, 1_048_576, 1_048_576, 8),
            validation_mode="exact",
        )


def _cpu_vector(process: Json, _profile: Json) -> Iterable[WorkloadSpec]:
    formats = process["numeric_formats"] or ["f32"]
    width = str(process["limits"].get("maximum_vector_width_bits", 0))
    for numeric_format in formats:
        element_bytes = {
            "bf16": 2,
            "f16": 2,
            "f32": 4,
            "f64": 8,
            "i8": 1,
            "u8": 1,
            "i16": 2,
            "u16": 2,
            "i32": 4,
            "u32": 4,
            "i64": 8,
            "u64": 8,
        }.get(numeric_format)
        if element_bytes is None:
            raise ValueError(f"unsupported CPU SIMD format {numeric_format!r}")
        yield WorkloadSpec(
            executor="cpu",
            operation="vector_fused_arithmetic",
            process_names=(process["name"],),
            regime=(("format", numeric_format), ("vector_width_bits", width)),
            work=Work(
                1_048_576,
                2_097_152,
                1_048_576 * element_bytes,
                1_048_576 * element_bytes,
            ),
            validation_mode="tolerance" if "f" in numeric_format else "exact",
            maximum_error_ppm=100 if "f" in numeric_format else 0,
        )


def _cpu_matrix(process: Json, _profile: Json) -> Iterable[WorkloadSpec]:
    for numeric_format in process["numeric_formats"] or ["f32"]:
        yield WorkloadSpec(
            executor="cpu",
            operation="blocked_matrix_multiply",
            process_names=(process["name"],),
            regime=(("format", numeric_format), ("shape", "64x64x64")),
            work=Work(262_144, 524_288, 32_768, 16_384),
            validation_mode="tolerance",
            maximum_error_ppm=250,
        )


def _cpu_bit(process: Json, _profile: Json) -> Iterable[WorkloadSpec]:
    yield WorkloadSpec(
        executor="cpu",
        operation="bit_population_mix",
        process_names=(process["name"],),
        regime=(("word_width_bits", "64"),),
        work=Work(1_048_576, 4_194_304, 8_388_608, 8),
        validation_mode="exact",
    )


def _memory_working_sets(profile: Json) -> list[tuple[str, int]]:
    cache_sizes = sorted(
        {
            int(domain["capacity_bytes"])
            for domain in profile["memory_domains"]
            if domain["kind"].endswith("cache")
        }
    )
    regimes: list[tuple[str, int]] = []
    if cache_sizes:
        regimes.append(("smallest_cache", max(4_096, cache_sizes[0] // 2)))
        regimes.append(("largest_cache", max(4_096, cache_sizes[-1] // 2)))
        regimes.append(("beyond_cache", max(64 * 1024 * 1024, cache_sizes[-1] * 2)))
    else:
        regimes.extend(
            [
                ("l1_scale", 32 * 1024),
                ("llc_scale", 8 * 1024 * 1024),
                ("dram_scale", 64 * 1024 * 1024),
            ]
        )
    return regimes


def _cpu_memory(process: Json, profile: Json) -> Iterable[WorkloadSpec]:
    patterns = (
        ("sequential_read", 1, 0),
        ("sequential_copy", 1, 1),
        ("strided_read", 1, 0),
        ("pointer_chase", 1, 0),
        ("gather_scatter", 1, 1),
    )
    for working_set_name, working_set_bytes in _memory_working_sets(profile):
        for pattern, read_multiplier, write_multiplier in patterns:
            yield WorkloadSpec(
                executor="cpu",
                operation=pattern,
                process_names=(process["name"],),
                regime=(
                    ("working_set", working_set_name),
                    ("working_set_bytes", str(working_set_bytes)),
                ),
                work=Work(
                    working_set_bytes // 8,
                    working_set_bytes // 8,
                    working_set_bytes * read_multiplier,
                    working_set_bytes * write_multiplier,
                ),
                validation_mode="digest",
            )


def _cpu_code(process: Json, _profile: Json) -> Iterable[WorkloadSpec]:
    for footprint in ("small", "large"):
        yield WorkloadSpec(
            executor="cpu",
            operation="generated_code_dispatch",
            process_names=(process["name"],),
            regime=(("instruction_footprint", footprint),),
            work=Work(262_144, 2_097_152, 2_097_152, 8),
            artifacts=(("generated_branch_program", "generated_code"),),
            validation_mode="exact",
        )


def _cpu_atomic(process: Json, _profile: Json) -> Iterable[WorkloadSpec]:
    for contention in ("independent", "shared"):
        yield WorkloadSpec(
            executor="cpu",
            operation="atomic_fetch_add",
            process_names=(process["name"],),
            regime=(("contention", contention),),
            work=Work(1_048_576, 1_048_576, 8_388_608, 8_388_608),
            validation_mode="exact",
        )


def _cpu_numa(process: Json, profile: Json) -> Iterable[WorkloadSpec]:
    node_count = str(
        process["limits"].get(
            "numa_node_count",
            profile["capability_extensions"].get("numa_node_count", 1),
        )
    )
    yield WorkloadSpec(
        executor="cpu",
        operation="numa_local_copy",
        process_names=(process["name"],),
        regime=(("numa_node_count", node_count), ("placement", "local")),
        work=Work(8_388_608, 8_388_608, 67_108_864, 67_108_864),
        validation_mode="digest",
    )
    if node_count != "1":
        yield WorkloadSpec(
            executor="cpu",
            operation="numa_remote_copy",
            process_names=(process["name"],),
            regime=(("numa_node_count", node_count), ("placement", "remote")),
            work=Work(8_388_608, 8_388_608, 67_108_864, 67_108_864),
            validation_mode="digest",
        )


def _gpu_formats(process: Json, fallback: tuple[str, ...] = ("f32",)) -> list[str]:
    return list(process["numeric_formats"] or fallback)


def _gpu_arithmetic(process: Json, _profile: Json) -> Iterable[WorkloadSpec]:
    item_count = 1_048_576
    for numeric_format in _gpu_formats(process):
        packed_dot = process["name"] == "packed_dot_product"
        if packed_dot:
            operations_per_item = (
                512 if numeric_format in {"i8", "u8", "f8_e4m3"} else 256
            )
            bytes_read_per_item = 8
        else:
            operations_per_item = 64 if numeric_format in {"i64", "u64"} else 256
            bytes_read_per_item = (
                4
                if numeric_format in {"i8", "u8", "i16", "u16", "i64", "u64"}
                else 8
                if numeric_format == "f64"
                else 16
            )
        yield WorkloadSpec(
            executor="vulkan_compute",
            operation=process["name"],
            process_names=(process["name"],),
            regime=(
                ("format", numeric_format),
                ("dependency", "independent_chains"),
            ),
            work=Work(
                item_count,
                item_count * operations_per_item,
                item_count * bytes_read_per_item,
                item_count * 4,
            ),
            artifacts=((f"{process['name']}_{numeric_format}", "spirv_compute"),),
            validation_mode="tolerance" if "f" in numeric_format else "exact",
            maximum_error_ppm=250 if "f" in numeric_format else 0,
        )


def _gpu_matrix(process: Json, _profile: Json) -> Iterable[WorkloadSpec]:
    properties = process["properties"]
    for numeric_format, property_name in (
        ("f16", "float16_shapes"),
        ("bf16", "bfloat16_shapes"),
        ("f8_e4m3", "float8_e4m3_shapes"),
    ):
        shapes = properties.get(property_name, "")
        for shape in filter(None, shapes.split(",")):
            dimensions = [int(value) for value in shape.split("x")]
            tile_count = 8_192
            operations = 2 * dimensions[0] * dimensions[1] * dimensions[2] * tile_count
            yield WorkloadSpec(
                executor="vulkan_compute",
                operation="cooperative_matrix_multiply",
                process_names=(process["name"],),
                regime=(("format", numeric_format), ("shape", shape)),
                work=Work(
                    tile_count,
                    operations,
                    tile_count * 512 * (1 if numeric_format == "f8_e4m3" else 2),
                    tile_count * 256 * 4,
                ),
                artifacts=(
                    (
                        f"cooperative_matrix_{numeric_format}_{shape}",
                        "spirv_compute",
                    ),
                ),
                validation_mode="tolerance",
                maximum_error_ppm=500,
            )


def _gpu_subgroup(process: Json, _profile: Json) -> Iterable[WorkloadSpec]:
    for operation in ("reduce", "scan", "shuffle", "ballot"):
        yield WorkloadSpec(
            executor="vulkan_compute",
            operation=f"subgroup_{operation}",
            process_names=(process["name"],),
            regime=(("operation", operation),),
            work=Work(16_777_216, 16_777_216, 67_108_864, 67_108_864),
            artifacts=((f"subgroup_{operation}", "spirv_compute"),),
            validation_mode="exact",
        )


def _gpu_memory(process: Json, _profile: Json) -> Iterable[WorkloadSpec]:
    patterns = (
        "sequential_copy",
        "strided_read",
        "gather_scatter",
        "packed_decode",
        "register_pressure_sweep",
        "shared_memory_tiled_copy",
    )
    for pattern in patterns:
        item_count = (
            1_048_576
            if pattern == "register_pressure_sweep"
            else 4_194_304
            if pattern == "shared_memory_tiled_copy"
            else 16_777_216
        )
        operation_multiplier = (
            320
            if pattern == "register_pressure_sweep"
            else 7
            if pattern == "packed_decode"
            else 2
            if pattern == "gather_scatter"
            else 1
        )
        yield WorkloadSpec(
            executor="vulkan_compute",
            operation=pattern,
            process_names=(process["name"],),
            regime=(("working_set_bytes", str(item_count * 4)),),
            work=Work(
                item_count,
                item_count * operation_multiplier,
                item_count * (8 if pattern == "gather_scatter" else 4),
                item_count * 4,
            ),
            artifacts=((pattern, "spirv_compute"),),
            validation_mode="digest",
        )


def _gpu_atomic(process: Json, _profile: Json) -> Iterable[WorkloadSpec]:
    item_count = 1_048_576
    for contention in ("independent", "workgroup", "global"):
        yield WorkloadSpec(
            executor="vulkan_compute",
            operation="atomic_add",
            process_names=(process["name"],),
            regime=(("contention", contention),),
            work=Work(item_count, item_count, item_count * 4, item_count * 4),
            artifacts=(("atomic_add", "spirv_compute"),),
            validation_mode="exact",
        )


def _gpu_scheduling(process: Json, _profile: Json) -> Iterable[WorkloadSpec]:
    for dispatch_count in (1, 16, 256):
        yield WorkloadSpec(
            executor=(
                "vulkan_dgc"
                if process["name"] == "device_generated_commands"
                else "vulkan_compute"
            ),
            operation=process["name"],
            process_names=(process["name"],),
            regime=(
                ("dispatch_count", str(dispatch_count)),
                ("command_reuse", "resident"),
            ),
            work=Work(
                dispatch_count,
                dispatch_count,
                dispatch_count * 4,
                dispatch_count * 4,
            ),
            artifacts=((f"scheduling_{process['name']}_{dispatch_count}", "spirv_compute"),),
            validation_mode="exact",
        )


def _gpu_transfer(process: Json, _profile: Json) -> Iterable[WorkloadSpec]:
    for direction in ("host_to_device", "device_to_host", "device_to_device"):
        yield WorkloadSpec(
            executor="vulkan_transfer",
            operation="buffer_copy",
            process_names=(process["name"],),
            regime=(("direction", direction), ("bytes", "268435456")),
            work=Work(67_108_864, 67_108_864, 268_435_456, 268_435_456),
            validation_mode="digest",
        )


def _gpu_sync(process: Json, _profile: Json) -> Iterable[WorkloadSpec]:
    for primitive in ("pipeline_barrier", "fence", "timeline_semaphore"):
        yield WorkloadSpec(
            executor="vulkan_synchronization",
            operation="synchronization_round_trip",
            process_names=(process["name"],),
            regime=(("primitive", primitive), ("round_trips", "4096")),
            work=Work(4_096, 4_096, 16_384, 16_384),
            validation_mode="exact",
        )


def _gpu_texture(process: Json, _profile: Json) -> Iterable[WorkloadSpec]:
    for filter_mode in ("nearest", "linear"):
        yield WorkloadSpec(
            executor="vulkan_graphics",
            operation="texture_sampling",
            process_names=(process["name"],),
            regime=(
                ("filter", filter_mode),
                ("format", "rgba16f"),
                ("addressing", "random"),
            ),
            work=Work(16_777_216, 16_777_216, 134_217_728, 67_108_864),
            artifacts=(("texture_sampling", "spirv_compute"),),
            validation_mode="tolerance",
            maximum_error_ppm=250,
        )


def _gpu_graphics(process: Json, _profile: Json) -> Iterable[WorkloadSpec]:
    yield WorkloadSpec(
        executor="vulkan_graphics",
        operation=process["name"],
        process_names=(process["name"],),
        regime=(
            ("render_target", "4096x4096"),
            ("format", "rgba16f"),
            ("overdraw", "4"),
        ),
        work=Work(67_108_864, 67_108_864, 268_435_456, 134_217_728),
        artifacts=(
            (f"graphics_{process['name']}_vertex", "spirv_vertex"),
            (f"graphics_{process['name']}_fragment", "spirv_fragment"),
        ),
        validation_mode="digest",
    )


def _gpu_ray(process: Json, _profile: Json) -> Iterable[WorkloadSpec]:
    operation = (
        "build_acceleration_structure"
        if process["name"] == "acceleration_structure_construction"
        else "ray_query_traversal"
    )
    artifacts = [("ray_scene", "procedural_ray_scene")]
    if operation == "ray_query_traversal":
        artifacts.append(("ray_query_shader", "spirv_compute"))
    yield WorkloadSpec(
        executor="vulkan_ray",
        operation=operation,
        process_names=(process["name"],),
        regime=(("primitives", "1048576"), ("rays", "16777216")),
        work=Work(16_777_216, 16_777_216, 268_435_456, 67_108_864),
        artifacts=tuple(artifacts),
        validation_mode="digest",
    )


def _gpu_video(process: Json, _profile: Json) -> Iterable[WorkloadSpec]:
    yield WorkloadSpec(
        executor="vulkan_video",
        operation=process["name"],
        process_names=(process["name"],),
        regime=(
            ("codec", "av1"),
            ("resolution", "3840x2160"),
            ("frames", "120"),
            ("timeout_ms", "30000"),
        ),
        work=Work(120, 120, 3_981_312_000, 3_981_312_000),
        artifacts=(
            ("video_backend_manifest", "external_backend_manifest"),
            ("video_bitstream", "video_fixture_av1"),
        ),
        validation_mode="digest",
    )


_PROVIDERS = (
    CalibrationProvider(
        "cpu_scalar",
        _named(
            "scalar_integer",
            "scalar_floating_point",
            "out_of_order_control_flow",
        ),
        _cpu_scalar,
    ),
    CalibrationProvider(
        "cpu_branch",
        _named("branch_execution"),
        _cpu_branch,
    ),
    CalibrationProvider("cpu_vector", _named("simd_vector"), _cpu_vector),
    CalibrationProvider("cpu_matrix", _named("matrix_extension"), _cpu_matrix),
    CalibrationProvider(
        "cpu_bit",
        _named("bit_manipulation"),
        _cpu_bit,
    ),
    CalibrationProvider(
        "cpu_memory",
        _named(
            "cache_hierarchy",
            "hardware_prefetch",
            "main_memory",
            "memory_bandwidth",
        ),
        _cpu_memory,
    ),
    CalibrationProvider(
        "cpu_code",
        _named("instruction_cache", "micro_op_cache"),
        _cpu_code,
    ),
    CalibrationProvider("cpu_atomic", _named("atomics"), _cpu_atomic),
    CalibrationProvider(
        "cpu_copy",
        _named("dma_engines", "host_memory_copy"),
        _cpu_memory,
    ),
    CalibrationProvider("cpu_numa", _named("numa_memory_policy"), _cpu_numa),
    CalibrationProvider(
        "gpu_arithmetic",
        _named("shader_scalar", "shader_vector", "packed_dot_product"),
        _gpu_arithmetic,
    ),
    CalibrationProvider(
        "gpu_matrix",
        _named("cooperative_matrix"),
        _gpu_matrix,
    ),
    CalibrationProvider(
        "gpu_subgroup",
        _named("subgroup_collectives", "parallel_collective_algorithms"),
        _gpu_subgroup,
    ),
    CalibrationProvider(
        "gpu_memory",
        _named(
            "device_cache_hierarchy",
            "device_memory_bandwidth",
            "occupancy_constraints",
            "register_file",
            "workgroup_shared_memory",
        ),
        _gpu_memory,
    ),
    CalibrationProvider("gpu_atomic", _named("shader_atomics"), _gpu_atomic),
    CalibrationProvider(
        "gpu_scheduling",
        _named(
            "command_queues",
            "device_generated_commands",
            "indirect_work_generation",
            "resident_command_replay",
        ),
        _gpu_scheduling,
    ),
    CalibrationProvider("gpu_transfer", _named("copy_engines"), _gpu_transfer),
    CalibrationProvider("gpu_sync", _named("synchronization"), _gpu_sync),
    CalibrationProvider("gpu_texture", _named("texture_sampling"), _gpu_texture),
    CalibrationProvider(
        "gpu_graphics",
        _named(
            "blending",
            "depth_stencil",
            "fixed_function_interpolation",
            "rasterization",
        ),
        _gpu_graphics,
    ),
    CalibrationProvider(
        "gpu_ray",
        _named("acceleration_structure_construction", "ray_traversal"),
        _gpu_ray,
    ),
    CalibrationProvider(
        "gpu_video",
        _named("video_decode", "video_encode"),
        _gpu_video,
    ),
)
