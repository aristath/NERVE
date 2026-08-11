use std::collections::BTreeMap;
use std::ffi::CString;
use std::hint::black_box;
use std::mem;
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd};
use std::ptr;
use std::sync::Arc;
use std::time::Instant;

use ash::{Entry, vk};

use crate::benchmark::{
    MULTI_TARGET_TENSOR_PARALLEL_STRATEGY, activation_bytes_for_payload, component_chain_regime,
    format_workload_id, output_bytes_for_payload, single_target_status_measurement,
    single_target_status_measurements, target_index_combinations, targets_support_tensor_parallel,
    tensor_parallel_group_pattern, tensor_parallel_group_workload_id,
};
use crate::model::{
    GroupMeasurement, MAX_PLACEMENT_GROUP_SIZE, Measurement, PairMeasurement, Sample, Summary,
    Target,
};
use crate::vulkan_features::{
    MIXED_FLOAT_DOT_PRODUCT_NAME, MixedFloatDotProductFeatures, NATIVE_FP8_DOT_FEATURE,
    SHADER_FLOAT8_NAME, ShaderFloat8Features, external_timeline_semaphore_supported,
};

const F32_ROUTER_REDUCTION_SHADER_SPV: &[u8] = include_bytes!("shaders/f32_router_reduction.spv");
const F32_GEMM_SHADER_SPV: &[u8] = include_bytes!("shaders/f32_gemm.spv");
const F16_GEMM_SHADER_SPV: &[u8] = include_bytes!("shaders/f16_gemm.spv");
const F16_ROUTER_REDUCTION_SHADER_SPV: &[u8] = include_bytes!("shaders/f16_router_reduction.spv");
const INT8_ROUTER_REDUCTION_SHADER_SPV: &[u8] = include_bytes!("shaders/int8_router_reduction.spv");
const FORMAT_DEQUANT_SHADER_SPV: &[u8] = include_bytes!("shaders/format_dequant.spv");
const FORMAT_DEQUANT_GEMM_SHADER_SPV: &[u8] = include_bytes!("shaders/format_dequant_gemm.spv");
const KV_CACHE_SHADER_SPV: &[u8] = include_bytes!("shaders/kv_cache.spv");
const TP_FORMAT_GEMM_SHADER_SPV: &[u8] = include_bytes!("shaders/tp_format_gemm.spv");
const TP_F16_GEMM_SHADER_SPV: &[u8] = include_bytes!("shaders/tp_f16_gemm.spv");
const NATIVE_FP8_GEMM_SHADER_SPV: &[u8] = include_bytes!("shaders/native_fp8_gemm.spv");
const TP_NATIVE_FP8_GEMM_SHADER_SPV: &[u8] = include_bytes!("shaders/tp_native_fp8_gemm.spv");
const MAX_VULKAN_WAIT_NS: u64 = 10_000_000_000;
const TENSOR_PARALLEL_PAIR_PATTERN: &str = "synthetic_tensor_parallel_small_payload";
const TENSOR_PARALLEL_CHAIN_PATTERN: &str = "synthetic_tensor_parallel_forced_split_2";
const TWO_TARGET_TENSOR_PARALLEL_STRATEGY: &str = "two_target_tensor_parallel";
const SINGLE_COMPONENT_REGIME: &str = "single_component";
const TWO_COMPONENT_CHAIN_REGIME: &str = "two_component_chain";
const MOE_SELECTED_EXPERTS: usize = 6;

#[derive(Clone, Copy)]
enum ShaderCode {
    Bytes(&'static [u8]),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KernelShape {
    RouterReduction,
    F32Gemm,
    PackedGemm,
    KvCache,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KernelWorkload {
    DenseProjection,
    MoeExpert,
    KvCache,
    RouterReduction,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WeightLayout {
    Plain,
    Bf16Scaled { group_size: usize },
    E8m0Scaled { group_size: usize },
    NerveQ8_0,
}

#[derive(Clone)]
struct DenseFormatKernel {
    format: String,
    shader: ShaderCode,
    execution_path: &'static str,
    bytes_per_storage_element: usize,
    logical_elements_per_storage_element: u64,
    pattern: &'static str,
    required_feature: Option<&'static str>,
    format_kind: u32,
    shape: KernelShape,
    workload: KernelWorkload,
    weight_layout: WeightLayout,
    phase: &'static str,
    batch_size: usize,
}

fn kernel_identity(kernel: &DenseFormatKernel) -> String {
    let base = format!(
        "{}:{}:phase={}",
        kernel.execution_path, kernel.pattern, kernel.phase
    );
    if matches!(
        kernel.shape,
        KernelShape::F32Gemm | KernelShape::PackedGemm | KernelShape::KvCache
    ) {
        format!("{base}:activation=bf16:output=bf16")
    } else {
        base
    }
}

fn workload_format_kernel(workload_class: &str, format: &str) -> Option<DenseFormatKernel> {
    let family = workload_family(workload_class)?;
    let mut kernel = match family {
        "dense_projection" => dense_format_kernel(format),
        "moe_expert" => moe_expert_kernel(format),
        "kv_cache" => kv_cache_kernel(format),
        "router_reduction" => router_reduction_kernel(format),
        _ => None,
    }?;
    kernel.workload = match family {
        "dense_projection" => KernelWorkload::DenseProjection,
        "moe_expert" => KernelWorkload::MoeExpert,
        "kv_cache" => KernelWorkload::KvCache,
        "router_reduction" => KernelWorkload::RouterReduction,
        _ => return None,
    };
    kernel.phase = workload_phase(workload_class)?;
    kernel.batch_size = workload_batch_size(workload_class)?;
    Some(kernel)
}

fn workload_format_kernel_for_devices(
    workload_class: &str,
    format: &str,
    devices: &[&OpenVulkanComputeDevice],
) -> Option<DenseFormatKernel> {
    let family = workload_family(workload_class)?;
    let phase = workload_phase(workload_class)?;
    let batch_size = workload_batch_size(workload_class)?;
    if family == "router_reduction"
        && format == "f16"
        && devices
            .iter()
            .any(|device| !feature_flags_include(&device.feature_flags, "shader_float16"))
    {
        let mut kernel =
            format_dequant_workload_kernel("f16", "router_reduction_f16_dequant_compute")?;
        kernel.phase = phase;
        kernel.batch_size = batch_size;
        return Some(kernel);
    }
    if family == "router_reduction"
        && format == "int8"
        && devices
            .iter()
            .any(|device| !feature_flags_include(&device.feature_flags, "shader_int8"))
    {
        let mut kernel =
            format_dequant_workload_kernel("int8", "router_reduction_int8_dequant_compute")?;
        kernel.phase = phase;
        kernel.batch_size = batch_size;
        return Some(kernel);
    }
    if format == "f16"
        && matches!(family, "dense_projection" | "moe_expert")
        && devices
            .iter()
            .all(|device| feature_flags_include(&device.feature_flags, "shader_float16"))
    {
        return Some(DenseFormatKernel {
            format: "f16".to_string(),
            shader: ShaderCode::Bytes(F16_GEMM_SHADER_SPV),
            execution_path: "native_f16",
            bytes_per_storage_element: mem::size_of::<u32>(),
            logical_elements_per_storage_element: 2,
            pattern: if family == "moe_expert" {
                "moe_expert_f16_native_gemm_compute"
            } else {
                "dense_projection_f16_native_gemm_compute"
            },
            required_feature: Some("shader_float16"),
            format_kind: 17,
            shape: KernelShape::PackedGemm,
            workload: if family == "moe_expert" {
                KernelWorkload::MoeExpert
            } else {
                KernelWorkload::DenseProjection
            },
            weight_layout: WeightLayout::Plain,
            phase,
            batch_size,
        });
    }
    if matches!(format, "fp8" | "fp8_e4m3" | "mxfp4")
        && matches!(family, "dense_projection" | "moe_expert")
        && devices
            .iter()
            .all(|device| feature_flags_include(&device.feature_flags, NATIVE_FP8_DOT_FEATURE))
    {
        let mut kernel = match family {
            "dense_projection" => dense_format_kernel(format),
            "moe_expert" => moe_expert_kernel(format),
            _ => None,
        }?;
        kernel.shader = ShaderCode::Bytes(NATIVE_FP8_GEMM_SHADER_SPV);
        kernel.execution_path = NATIVE_FP8_DOT_FEATURE;
        kernel.pattern = native_fp8_pattern(family, format)?;
        kernel.required_feature = Some(NATIVE_FP8_DOT_FEATURE);
        kernel.phase = phase;
        kernel.batch_size = batch_size;
        return Some(kernel);
    }
    workload_format_kernel(workload_class, format)
}

fn native_fp8_pattern(family: &str, format: &str) -> Option<&'static str> {
    match (family, format) {
        ("dense_projection", "fp8" | "fp8_e4m3") => {
            Some("dense_projection_fp8_e4m3_native_dot_compute")
        }
        ("dense_projection", "mxfp4") => Some("dense_projection_mxfp4_native_dot_compute"),
        ("moe_expert", "fp8" | "fp8_e4m3") => Some("moe_expert_fp8_e4m3_native_dot_compute"),
        ("moe_expert", "mxfp4") => Some("moe_expert_mxfp4_native_dot_compute"),
        _ => None,
    }
}

fn workload_family(workload: &str) -> Option<&'static str> {
    [
        "dense_projection",
        "moe_expert",
        "kv_cache",
        "router_reduction",
    ]
    .into_iter()
    .find(|family| workload == *family || workload.starts_with(&format!("{family}_")))
}

fn workload_phase(workload: &str) -> Option<&'static str> {
    if workload.ends_with("_decode") {
        Some("decode")
    } else if workload.ends_with("_prefill") || workload_family(workload).is_some() {
        Some("prefill")
    } else {
        None
    }
}

fn workload_batch_size(workload: &str) -> Option<usize> {
    match workload_phase(workload)? {
        "decode" => Some(1),
        "prefill" => Some(16),
        _ => None,
    }
}

fn dense_format_kernel(format: &str) -> Option<DenseFormatKernel> {
    match format {
        "f32" => Some(DenseFormatKernel {
            format: "f32".to_string(),
            shader: ShaderCode::Bytes(F32_GEMM_SHADER_SPV),
            execution_path: "native_f32",
            bytes_per_storage_element: mem::size_of::<f32>(),
            logical_elements_per_storage_element: 1,
            pattern: "dense_projection_f32_gemm_compute",
            required_feature: None,
            format_kind: 0,
            shape: KernelShape::F32Gemm,
            workload: KernelWorkload::DenseProjection,
            weight_layout: WeightLayout::Plain,
            phase: "prefill",
            batch_size: 16,
        }),
        "f16" => Some(format_dequant_gemm_kernel(
            "f16",
            2,
            17,
            WeightLayout::Plain,
            "dense_projection_f16_gemm_compute",
        )),
        "bf16" => Some(format_dequant_gemm_kernel(
            "bf16",
            2,
            1,
            WeightLayout::Plain,
            "dense_projection_bf16_gemm_compute",
        )),
        "fp8" | "fp8_e4m3" => Some(format_dequant_gemm_kernel(
            format,
            4,
            2,
            WeightLayout::Bf16Scaled { group_size: 128 },
            "dense_projection_fp8_e4m3_gemm_compute",
        )),
        "fp8_e5m2" => Some(format_dequant_gemm_kernel(
            format,
            4,
            3,
            WeightLayout::Bf16Scaled { group_size: 128 },
            "dense_projection_fp8_e5m2_gemm_compute",
        )),
        "int8" => Some(format_dequant_gemm_kernel(
            "int8",
            4,
            8,
            WeightLayout::Plain,
            "dense_projection_int8_gemm_compute",
        )),
        "q8_0" => Some(format_dequant_gemm_kernel(
            format,
            32,
            18,
            WeightLayout::NerveQ8_0,
            "dense_projection_q8_0_gemm_compute",
        )),
        "int4" => Some(format_dequant_gemm_kernel(
            format,
            8,
            7,
            WeightLayout::Bf16Scaled { group_size: 128 },
            "dense_projection_int4_gemm_compute",
        )),
        "fp4" => Some(format_dequant_gemm_kernel(
            "fp4",
            8,
            4,
            WeightLayout::Plain,
            "dense_projection_fp4_gemm_compute",
        )),
        "mxfp4" => Some(format_dequant_gemm_kernel(
            format,
            8,
            5,
            WeightLayout::E8m0Scaled { group_size: 32 },
            "dense_projection_mxfp4_gemm_compute",
        )),
        _ => None,
    }
}

fn format_dequant_kernel(
    format: &str,
    logical_elements_per_storage_element: u64,
    format_kind: u32,
) -> DenseFormatKernel {
    DenseFormatKernel {
        format: format.to_string(),
        shader: ShaderCode::Bytes(FORMAT_DEQUANT_SHADER_SPV),
        execution_path: "dequant_f32",
        bytes_per_storage_element: mem::size_of::<u32>(),
        logical_elements_per_storage_element,
        pattern: "single_target_format_dequant_compute",
        required_feature: None,
        format_kind,
        shape: KernelShape::RouterReduction,
        workload: KernelWorkload::RouterReduction,
        weight_layout: weight_layout_for_format(format),
        phase: "prefill",
        batch_size: 16,
    }
}

fn weight_layout_for_format(format: &str) -> WeightLayout {
    match format {
        "fp8" | "fp8_e4m3" | "fp8_e5m2" => WeightLayout::Bf16Scaled { group_size: 128 },
        "mxfp4" => WeightLayout::E8m0Scaled { group_size: 32 },
        "int4" => WeightLayout::Bf16Scaled { group_size: 128 },
        "q8_0" => WeightLayout::NerveQ8_0,
        _ => WeightLayout::Plain,
    }
}

fn format_dequant_gemm_kernel(
    format: &str,
    logical_elements_per_storage_element: u64,
    format_kind: u32,
    weight_layout: WeightLayout,
    pattern: &'static str,
) -> DenseFormatKernel {
    DenseFormatKernel {
        format: format.to_string(),
        shader: ShaderCode::Bytes(FORMAT_DEQUANT_GEMM_SHADER_SPV),
        execution_path: "dequant_f32",
        bytes_per_storage_element: mem::size_of::<u32>(),
        logical_elements_per_storage_element,
        pattern,
        required_feature: None,
        format_kind,
        shape: KernelShape::PackedGemm,
        workload: KernelWorkload::DenseProjection,
        weight_layout,
        phase: "prefill",
        batch_size: 16,
    }
}

fn moe_expert_kernel(format: &str) -> Option<DenseFormatKernel> {
    let mut kernel = match format {
        "f32" => Some(DenseFormatKernel {
            format: "f32".to_string(),
            shader: ShaderCode::Bytes(F32_GEMM_SHADER_SPV),
            execution_path: "native_f32",
            bytes_per_storage_element: mem::size_of::<f32>(),
            logical_elements_per_storage_element: 1,
            pattern: "moe_expert_f32_gemm_compute",
            required_feature: None,
            format_kind: 0,
            shape: KernelShape::F32Gemm,
            workload: KernelWorkload::MoeExpert,
            weight_layout: WeightLayout::Plain,
            phase: "prefill",
            batch_size: 16,
        }),
        "f16" => Some(format_dequant_gemm_kernel(
            "f16",
            2,
            17,
            WeightLayout::Plain,
            "moe_expert_f16_gemm_compute",
        )),
        "int8" => Some(format_dequant_gemm_kernel(
            "int8",
            4,
            8,
            WeightLayout::Plain,
            "moe_expert_int8_gemm_compute",
        )),
        _ => dense_format_kernel(format).map(|base| DenseFormatKernel {
            pattern: match format {
                "bf16" => "moe_expert_bf16_gemm_compute",
                "fp8" | "fp8_e4m3" => "moe_expert_fp8_e4m3_gemm_compute",
                "fp8_e5m2" => "moe_expert_fp8_e5m2_gemm_compute",
                "fp4" => "moe_expert_fp4_gemm_compute",
                "mxfp4" => "moe_expert_mxfp4_gemm_compute",
                "int4" => "moe_expert_int4_gemm_compute",
                "q8_0" => "moe_expert_q8_0_gemm_compute",
                _ => base.pattern,
            },
            ..base
        }),
    }?;
    kernel.workload = KernelWorkload::MoeExpert;
    Some(kernel)
}

fn kv_cache_kernel(format: &str) -> Option<DenseFormatKernel> {
    (format == "bf16").then(|| DenseFormatKernel {
        format: "bf16".to_string(),
        shader: ShaderCode::Bytes(KV_CACHE_SHADER_SPV),
        execution_path: "native_bf16_state",
        bytes_per_storage_element: mem::size_of::<u32>(),
        logical_elements_per_storage_element: 2,
        pattern: "kv_cache_context_read_append",
        required_feature: None,
        format_kind: 1,
        shape: KernelShape::KvCache,
        workload: KernelWorkload::KvCache,
        weight_layout: WeightLayout::Plain,
        phase: "prefill",
        batch_size: 16,
    })
}

fn router_reduction_kernel(format: &str) -> Option<DenseFormatKernel> {
    match format {
        "f32" => Some(DenseFormatKernel {
            format: "f32".to_string(),
            shader: ShaderCode::Bytes(F32_ROUTER_REDUCTION_SHADER_SPV),
            execution_path: "native_f32",
            bytes_per_storage_element: mem::size_of::<f32>(),
            logical_elements_per_storage_element: 1,
            pattern: "router_reduction_compute",
            required_feature: None,
            format_kind: 0,
            shape: KernelShape::RouterReduction,
            workload: KernelWorkload::RouterReduction,
            weight_layout: WeightLayout::Plain,
            phase: "prefill",
            batch_size: 16,
        }),
        "f16" => Some(DenseFormatKernel {
            format: "f16".to_string(),
            shader: ShaderCode::Bytes(F16_ROUTER_REDUCTION_SHADER_SPV),
            execution_path: "native_f16",
            bytes_per_storage_element: mem::size_of::<u32>(),
            logical_elements_per_storage_element: 2,
            pattern: "router_reduction_f16_native_compute",
            required_feature: Some("shader_float16"),
            format_kind: 0,
            shape: KernelShape::RouterReduction,
            workload: KernelWorkload::RouterReduction,
            weight_layout: WeightLayout::Plain,
            phase: "prefill",
            batch_size: 16,
        }),
        "int8" => Some(DenseFormatKernel {
            format: "int8".to_string(),
            shader: ShaderCode::Bytes(INT8_ROUTER_REDUCTION_SHADER_SPV),
            execution_path: "native_int8",
            bytes_per_storage_element: mem::size_of::<u32>(),
            logical_elements_per_storage_element: 4,
            pattern: "router_reduction_int8_native_compute",
            required_feature: Some("shader_int8"),
            format_kind: 0,
            shape: KernelShape::RouterReduction,
            workload: KernelWorkload::RouterReduction,
            weight_layout: WeightLayout::Plain,
            phase: "prefill",
            batch_size: 16,
        }),
        _ => format_dequant_workload_kernel(format, "router_reduction_format_dequant_compute"),
    }
}

fn format_dequant_workload_kernel(
    format: &str,
    pattern: &'static str,
) -> Option<DenseFormatKernel> {
    format_dequant_elementwise_kernel(format).map(|base| DenseFormatKernel { pattern, ..base })
}

fn format_dequant_elementwise_kernel(format: &str) -> Option<DenseFormatKernel> {
    match format {
        "f16" => Some(format_dequant_kernel("f16", 2, 17)),
        "bf16" => Some(format_dequant_kernel("bf16", 2, 1)),
        "fp8" | "fp8_e4m3" => Some(format_dequant_kernel(format, 4, 2)),
        "fp8_e5m2" => Some(format_dequant_kernel(format, 4, 3)),
        "int8" => Some(format_dequant_kernel("int8", 4, 8)),
        "q8_0" => Some(format_dequant_kernel(format, 32, 18)),
        "int4" => Some(format_dequant_kernel(format, 8, 7)),
        "fp4" => Some(format_dequant_kernel("fp4", 8, 4)),
        "mxfp4" => Some(format_dequant_kernel(format, 8, 5)),
        _ => None,
    }
}

pub struct VulkanBenchmarkResults {
    pub measurements: Vec<Measurement>,
    pub pair_measurements: Vec<PairMeasurement>,
    pub group_measurements: Vec<GroupMeasurement>,
}

fn measured_serial_edge_costs(
    measurements: &[Measurement],
    pair_measurements: &[PairMeasurement],
    format: &str,
    workload: &str,
) -> BTreeMap<(String, String), u128> {
    let local_stage_costs = measurements
        .iter()
        .filter(|measurement| {
            measurement.status == "completed"
                && measurement.regime == TWO_COMPONENT_CHAIN_REGIME
                && measurement.format == format
                && measurement.workload_class == workload
        })
        .filter_map(|measurement| {
            measurement.summary.as_ref().map(|summary| {
                (
                    measurement.target_id.clone(),
                    summary.median_duration_ns / 2,
                )
            })
        })
        .collect::<BTreeMap<_, _>>();
    pair_measurements
        .iter()
        .filter(|measurement| {
            measurement.status == "completed"
                && measurement.placement_strategy == "two_target_serial"
                && measurement.regime == TWO_COMPONENT_CHAIN_REGIME
                && measurement.format == format
                && measurement.workload_class == workload
        })
        .filter_map(|measurement| {
            let summary = measurement.summary.as_ref()?;
            let source = local_stage_costs.get(&measurement.source_target_id)?;
            let destination = local_stage_costs.get(&measurement.destination_target_id)?;
            Some((
                (
                    measurement.source_target_id.clone(),
                    measurement.destination_target_id.clone(),
                ),
                summary
                    .median_duration_ns
                    .saturating_sub(*source)
                    .saturating_sub(*destination),
            ))
        })
        .collect()
}

fn best_serial_order(
    target_ids: &[String],
    edge_costs: &BTreeMap<(String, String), u128>,
) -> Vec<String> {
    let mut orders = Vec::new();
    extend_target_orders(target_ids, &mut Vec::new(), &mut orders);
    orders
        .into_iter()
        .min_by(|left, right| {
            serial_order_cost(left, edge_costs)
                .cmp(&serial_order_cost(right, edge_costs))
                .then_with(|| left.cmp(right))
        })
        .unwrap_or_default()
}

fn extend_target_orders(
    target_ids: &[String],
    current: &mut Vec<String>,
    orders: &mut Vec<Vec<String>>,
) {
    if current.len() == target_ids.len() {
        orders.push(current.clone());
        return;
    }
    for target_id in target_ids {
        if current.contains(target_id) {
            continue;
        }
        current.push(target_id.clone());
        extend_target_orders(target_ids, current, orders);
        current.pop();
    }
}

fn serial_order_cost(order: &[String], edge_costs: &BTreeMap<(String, String), u128>) -> u128 {
    order
        .windows(2)
        .try_fold(0_u128, |cost, edge| {
            edge_costs
                .get(&(edge[0].clone(), edge[1].clone()))
                .and_then(|edge_cost| cost.checked_add(*edge_cost))
        })
        .unwrap_or(u128::MAX)
}

pub fn run_vulkan_benchmarks(
    targets: &[&Target],
    payload_bytes: usize,
    samples: usize,
    formats: &[String],
    workloads: &[String],
    max_group_size: usize,
) -> VulkanBenchmarkResults {
    let opened = targets
        .iter()
        .map(|target| open_compute_device(target))
        .collect::<Vec<_>>();
    let mut measurements = Vec::new();
    let mut pair_measurements = Vec::new();
    let mut group_measurements = Vec::new();

    for (index, target) in targets.iter().enumerate() {
        match &opened[index] {
            Ok(device) => {
                measurements.extend(vulkan_measurements(
                    device,
                    &target.stable_target_id,
                    payload_bytes,
                    samples,
                    formats,
                    workloads,
                    max_group_size,
                ));
            }
            Err(message) => {
                measurements.extend(single_target_status_measurements(
                    &target.stable_target_id,
                    payload_bytes,
                    formats,
                    workloads,
                    "failed",
                    message,
                ));
                measurements.extend(formats.iter().flat_map(|format| {
                    workloads.iter().filter_map(|workload| {
                        supports_component_chain(workload).then(|| {
                            single_target_status_measurement_for_regime(
                                &target.stable_target_id,
                                payload_bytes,
                                workload,
                                format,
                                TWO_COMPONENT_CHAIN_REGIME,
                                "failed",
                                message,
                            )
                        })
                    })
                }));
            }
        }
    }

    if max_group_size >= 2 {
        for source_index in 0..targets.len() {
            for destination_index in 0..targets.len() {
                if source_index == destination_index {
                    continue;
                }
                let source = targets[source_index];
                let destination = targets[destination_index];
                let tensor_parallel = targets_support_tensor_parallel(&[source, destination]);
                match (&opened[source_index], &opened[destination_index]) {
                    (Ok(source_device), Ok(destination_device)) => {
                        pair_measurements.extend(opened_serial_pair_measurements(
                            source_device,
                            destination_device,
                            &source.stable_target_id,
                            &destination.stable_target_id,
                            payload_bytes,
                            samples,
                            formats,
                            workloads,
                        ));
                        if tensor_parallel {
                            pair_measurements.extend(opened_tensor_parallel_pair_measurements(
                                source_device,
                                destination_device,
                                &source.stable_target_id,
                                &destination.stable_target_id,
                                payload_bytes,
                                samples,
                                formats,
                                workloads,
                            ));
                            pair_measurements.extend(
                                opened_tensor_parallel_pair_chain_measurements(
                                    source_device,
                                    destination_device,
                                    &source.stable_target_id,
                                    &destination.stable_target_id,
                                    payload_bytes,
                                    samples,
                                    formats,
                                    workloads,
                                ),
                            );
                        }
                    }
                    (Err(message), _) | (_, Err(message)) => {
                        pair_measurements.extend(failed_dense_pair_measurements(
                            &source.stable_target_id,
                            &destination.stable_target_id,
                            "synthetic_layer_split_small_payload",
                            "two_target_serial",
                            payload_bytes,
                            formats,
                            workloads,
                            message,
                        ));
                        if tensor_parallel {
                            pair_measurements.extend(failed_dense_pair_measurements(
                                &source.stable_target_id,
                                &destination.stable_target_id,
                                TENSOR_PARALLEL_PAIR_PATTERN,
                                TWO_TARGET_TENSOR_PARALLEL_STRATEGY,
                                payload_bytes,
                                formats,
                                workloads,
                                message,
                            ));
                            pair_measurements.extend(formats.iter().flat_map(|format| {
                                workloads.iter().filter_map(|workload| {
                                    supports_tensor_parallel(workload).then(|| {
                                        dense_pair_status_measurement_for_regime(
                                            &source.stable_target_id,
                                            &destination.stable_target_id,
                                            TENSOR_PARALLEL_CHAIN_PATTERN,
                                            TWO_TARGET_TENSOR_PARALLEL_STRATEGY,
                                            TWO_COMPONENT_CHAIN_REGIME,
                                            "failed",
                                            payload_bytes,
                                            workload,
                                            format,
                                            message,
                                        )
                                    })
                                })
                            }));
                        }
                    }
                }
            }
        }
    }

    let max_group_size = max_group_size
        .min(MAX_PLACEMENT_GROUP_SIZE)
        .min(targets.len());
    if max_group_size >= 3 {
        let serial_edge_costs = formats
            .iter()
            .flat_map(|format| {
                workloads.iter().map(|workload| {
                    (
                        (format.clone(), workload.clone()),
                        measured_serial_edge_costs(
                            &measurements,
                            &pair_measurements,
                            format,
                            workload,
                        ),
                    )
                })
            })
            .collect::<BTreeMap<_, _>>();
        drop(opened);
        for group_size in 3..=max_group_size {
            for indices in target_index_combinations(targets.len(), group_size) {
                let group = indices
                    .iter()
                    .map(|index| targets[*index])
                    .collect::<Vec<_>>();
                if !targets_support_tensor_parallel(&group) {
                    continue;
                }
                let group_target_ids = group
                    .iter()
                    .map(|target| target.stable_target_id.clone())
                    .collect::<Vec<_>>();
                for format in formats {
                    for workload in workloads {
                        let edge_costs = &serial_edge_costs[&(format.clone(), workload.clone())];
                        let order = best_serial_order(&group_target_ids, edge_costs);
                        let ordered_targets = order
                            .iter()
                            .map(|target_id| {
                                *group
                                    .iter()
                                    .find(|target| target.stable_target_id == *target_id)
                                    .expect("serial order must contain only group targets")
                            })
                            .collect::<Vec<_>>();
                        let candidate_devices = ordered_targets
                            .iter()
                            .map(|target| open_compute_device(target))
                            .collect::<Vec<_>>();
                        if let Some(measurement) = opened_serial_group_measurement(
                            &candidate_devices,
                            &ordered_targets,
                            payload_bytes,
                            samples,
                            format,
                            workload,
                        ) {
                            group_measurements.push(measurement);
                        }
                    }
                }
                for owner in 0..group_size {
                    let mut ordered_targets = Vec::with_capacity(group_size);
                    ordered_targets.push(targets[indices[owner]]);
                    ordered_targets.extend(
                        indices
                            .iter()
                            .copied()
                            .enumerate()
                            .filter(|(index, _)| *index != owner)
                            .map(|(_, target_index)| targets[target_index]),
                    );
                    // Bound driver object lifetime to one placement candidate. Some drivers retain
                    // destroyed command resources until the logical device itself is destroyed.
                    let candidate_devices = ordered_targets
                        .iter()
                        .map(|target| open_compute_device(target))
                        .collect::<Vec<_>>();
                    let order = (0..group_size).collect::<Vec<_>>();
                    group_measurements.extend(opened_tensor_parallel_group_measurements(
                        &candidate_devices,
                        &ordered_targets,
                        &order,
                        payload_bytes,
                        samples,
                        formats,
                        workloads,
                    ));
                    group_measurements.extend(opened_tensor_parallel_group_chain_measurements(
                        &candidate_devices,
                        &ordered_targets,
                        &order,
                        payload_bytes,
                        samples,
                        formats,
                        workloads,
                    ));
                }
            }
        }
    }

    VulkanBenchmarkResults {
        measurements,
        pair_measurements,
        group_measurements,
    }
}

#[allow(clippy::too_many_arguments)]
fn failed_dense_pair_measurements(
    source_id: &str,
    destination_id: &str,
    workload_prefix: &str,
    placement_strategy: &str,
    payload_bytes: usize,
    formats: &[String],
    workloads: &[String],
    reason: &str,
) -> Vec<PairMeasurement> {
    formats
        .iter()
        .flat_map(|format| {
            workloads.iter().filter_map(move |workload| {
                let executable = if placement_strategy == "two_target_serial" {
                    supports_component_chain(workload)
                } else {
                    supports_tensor_parallel(workload)
                };
                if !executable {
                    return None;
                }
                workload_format_kernel(workload, format).map(|_| {
                    failed_dense_pair_measurement(
                        source_id,
                        destination_id,
                        workload_prefix,
                        placement_strategy,
                        payload_bytes,
                        workload,
                        format,
                        reason,
                    )
                })
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn failed_dense_pair_measurement(
    source_id: &str,
    destination_id: &str,
    workload_prefix: &str,
    placement_strategy: &str,
    payload_bytes: usize,
    workload_class: &str,
    format: &str,
    reason: &str,
) -> PairMeasurement {
    dense_pair_status_measurement(
        source_id,
        destination_id,
        workload_prefix,
        placement_strategy,
        "failed",
        payload_bytes,
        workload_class,
        format,
        reason,
    )
}

#[allow(clippy::too_many_arguments)]
fn dense_pair_status_measurement(
    source_id: &str,
    destination_id: &str,
    workload_prefix: &str,
    placement_strategy: &str,
    status: &str,
    payload_bytes: usize,
    workload_class: &str,
    format: &str,
    reason: &str,
) -> PairMeasurement {
    let regime = if placement_strategy == "two_target_serial" {
        TWO_COMPONENT_CHAIN_REGIME
    } else {
        SINGLE_COMPONENT_REGIME
    };
    dense_pair_status_measurement_for_regime(
        source_id,
        destination_id,
        workload_prefix,
        placement_strategy,
        regime,
        status,
        payload_bytes,
        workload_class,
        format,
        reason,
    )
}

#[allow(clippy::too_many_arguments)]
fn dense_pair_status_measurement_for_regime(
    source_id: &str,
    destination_id: &str,
    workload_prefix: &str,
    placement_strategy: &str,
    regime: &str,
    status: &str,
    payload_bytes: usize,
    workload_class: &str,
    format: &str,
    reason: &str,
) -> PairMeasurement {
    let source_payload_bytes = payload_bytes / 2;
    let destination_payload_bytes = payload_bytes - source_payload_bytes;
    PairMeasurement {
        workload_id: format_workload_id(workload_prefix, workload_class, format),
        comparison_group: "small_payload_placement_comparison".to_string(),
        workload_class: workload_class.to_string(),
        placement_strategy: placement_strategy.to_string(),
        source_target_id: source_id.to_string(),
        destination_target_id: destination_id.to_string(),
        pattern: workload_prefix.to_string(),
        operation_family: workload_class.to_string(),
        regime: regime.to_string(),
        format: format.to_string(),
        status: status.to_string(),
        reason: Some(reason.to_string()),
        payload_bytes,
        source_payload_bytes,
        destination_payload_bytes,
        activation_bytes: activation_bytes_for_payload(payload_bytes),
        output_bytes: output_bytes_for_payload(payload_bytes),
        samples: Vec::new(),
        summary: None,
    }
}

fn opened_serial_pair_measurements(
    source_device: &OpenVulkanComputeDevice,
    destination_device: &OpenVulkanComputeDevice,
    source_id: &str,
    destination_id: &str,
    payload_bytes: usize,
    samples: usize,
    formats: &[String],
    workloads: &[String],
) -> Vec<PairMeasurement> {
    formats
        .iter()
        .flat_map(|format| {
            workloads.iter().filter_map(|workload| {
                if !supports_component_chain(workload) {
                    return None;
                }
                workload_format_kernel_for_devices(
                    workload,
                    format,
                    &[source_device, destination_device],
                )
                .map(|kernel| {
                    if let Some(reason) =
                        unsupported_kernel_reason(&kernel, &[source_device, destination_device])
                    {
                        return dense_pair_status_measurement(
                            source_id,
                            destination_id,
                            "synthetic_layer_split_small_payload",
                            "two_target_serial",
                            "unsupported",
                            payload_bytes,
                            workload,
                            format,
                            &reason,
                        );
                    }
                    run_vulkan_dense_serial_pair(
                        source_device,
                        destination_device,
                        source_id,
                        destination_id,
                        payload_bytes,
                        samples,
                        workload,
                        kernel,
                    )
                    .unwrap_or_else(|message| {
                        failed_dense_pair_measurement(
                            source_id,
                            destination_id,
                            "synthetic_layer_split_small_payload",
                            "two_target_serial",
                            payload_bytes,
                            workload,
                            format,
                            &message,
                        )
                    })
                })
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn opened_tensor_parallel_pair_measurements(
    left_device: &OpenVulkanComputeDevice,
    right_device: &OpenVulkanComputeDevice,
    left_id: &str,
    right_id: &str,
    payload_bytes: usize,
    samples: usize,
    formats: &[String],
    workloads: &[String],
) -> Vec<PairMeasurement> {
    formats
        .iter()
        .flat_map(|format| {
            workloads.iter().filter_map(|workload| {
                if !supports_tensor_parallel(workload) {
                    return None;
                }
                workload_format_kernel_for_devices(workload, format, &[left_device, right_device])
                    .map(|kernel| {
                        if let Some(reason) =
                            unsupported_kernel_reason(&kernel, &[left_device, right_device])
                        {
                            return dense_pair_status_measurement(
                                left_id,
                                right_id,
                                TENSOR_PARALLEL_PAIR_PATTERN,
                                TWO_TARGET_TENSOR_PARALLEL_STRATEGY,
                                "unsupported",
                                payload_bytes,
                                workload,
                                format,
                                &reason,
                            );
                        }
                        run_vulkan_dense_tensor_parallel_pair(
                            left_device,
                            right_device,
                            left_id,
                            right_id,
                            payload_bytes,
                            samples,
                            workload,
                            kernel,
                        )
                        .unwrap_or_else(|message| {
                            failed_dense_pair_measurement(
                                left_id,
                                right_id,
                                TENSOR_PARALLEL_PAIR_PATTERN,
                                TWO_TARGET_TENSOR_PARALLEL_STRATEGY,
                                payload_bytes,
                                workload,
                                format,
                                &message,
                            )
                        })
                    })
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn opened_tensor_parallel_pair_chain_measurements(
    left_device: &OpenVulkanComputeDevice,
    right_device: &OpenVulkanComputeDevice,
    left_id: &str,
    right_id: &str,
    payload_bytes: usize,
    samples: usize,
    formats: &[String],
    workloads: &[String],
) -> Vec<PairMeasurement> {
    formats
        .iter()
        .flat_map(|format| {
            workloads.iter().filter_map(|workload| {
                if !supports_tensor_parallel(workload) {
                    return None;
                }
                workload_format_kernel_for_devices(workload, format, &[left_device, right_device])
                    .map(|kernel| {
                        if let Some(reason) =
                            unsupported_kernel_reason(&kernel, &[left_device, right_device])
                        {
                            return dense_pair_status_measurement_for_regime(
                                left_id,
                                right_id,
                                TENSOR_PARALLEL_CHAIN_PATTERN,
                                TWO_TARGET_TENSOR_PARALLEL_STRATEGY,
                                TWO_COMPONENT_CHAIN_REGIME,
                                "unsupported",
                                payload_bytes,
                                workload,
                                format,
                                &reason,
                            );
                        }
                        run_vulkan_dense_tensor_parallel_pair_chain(
                            left_device,
                            right_device,
                            left_id,
                            right_id,
                            payload_bytes,
                            samples,
                            workload,
                            kernel,
                        )
                        .unwrap_or_else(|message| {
                            dense_pair_status_measurement_for_regime(
                                left_id,
                                right_id,
                                TENSOR_PARALLEL_CHAIN_PATTERN,
                                TWO_TARGET_TENSOR_PARALLEL_STRATEGY,
                                TWO_COMPONENT_CHAIN_REGIME,
                                "failed",
                                payload_bytes,
                                workload,
                                format,
                                &message,
                            )
                        })
                    })
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn opened_tensor_parallel_group_measurements(
    opened: &[Result<OpenVulkanComputeDevice, String>],
    targets: &[&Target],
    order: &[usize],
    payload_bytes: usize,
    samples: usize,
    formats: &[String],
    workloads: &[String],
) -> Vec<GroupMeasurement> {
    let target_ids = order
        .iter()
        .map(|index| targets[*index].stable_target_id.clone())
        .collect::<Vec<_>>();
    let mut devices = Vec::with_capacity(order.len());
    let mut open_error = None;
    for index in order {
        match &opened[*index] {
            Ok(device) => devices.push(device),
            Err(message) => {
                open_error = Some(message.as_str());
                break;
            }
        }
    }

    formats
        .iter()
        .flat_map(|format| {
            workloads.iter().filter_map(|workload| {
                if !supports_tensor_parallel(workload) {
                    return None;
                }
                if let Some(message) = open_error {
                    return workload_format_kernel(workload, format).map(|_| {
                        tensor_parallel_group_status_measurement(
                            &target_ids,
                            "failed",
                            payload_bytes,
                            workload,
                            format,
                            message,
                        )
                    });
                }
                workload_format_kernel_for_devices(workload, format, &devices).map(|kernel| {
                    if let Some(reason) = unsupported_kernel_reason(&kernel, &devices) {
                        return tensor_parallel_group_status_measurement(
                            &target_ids,
                            "unsupported",
                            payload_bytes,
                            workload,
                            format,
                            &reason,
                        );
                    }
                    run_vulkan_dense_tensor_parallel_group(
                        &devices,
                        &target_ids,
                        payload_bytes,
                        samples,
                        workload,
                        kernel,
                    )
                    .unwrap_or_else(|message| {
                        tensor_parallel_group_status_measurement(
                            &target_ids,
                            "failed",
                            payload_bytes,
                            workload,
                            format,
                            &message,
                        )
                    })
                })
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn opened_tensor_parallel_group_chain_measurements(
    opened: &[Result<OpenVulkanComputeDevice, String>],
    targets: &[&Target],
    order: &[usize],
    payload_bytes: usize,
    samples: usize,
    formats: &[String],
    workloads: &[String],
) -> Vec<GroupMeasurement> {
    let target_ids = order
        .iter()
        .map(|index| targets[*index].stable_target_id.clone())
        .collect::<Vec<_>>();
    let mut devices = Vec::with_capacity(order.len());
    let mut open_error = None;
    for index in order {
        match &opened[*index] {
            Ok(device) => devices.push(device),
            Err(message) => {
                open_error = Some(message.as_str());
                break;
            }
        }
    }
    formats
        .iter()
        .flat_map(|format| {
            workloads.iter().filter_map(|workload| {
                if !supports_tensor_parallel(workload) {
                    return None;
                }
                if let Some(message) = open_error {
                    return workload_format_kernel(workload, format).map(|_| {
                        forced_split_group_status_measurement(
                            &target_ids,
                            MULTI_TARGET_TENSOR_PARALLEL_STRATEGY,
                            "failed",
                            payload_bytes,
                            workload,
                            format,
                            message,
                        )
                    });
                }
                workload_format_kernel_for_devices(workload, format, &devices).map(|kernel| {
                    if let Some(reason) = unsupported_kernel_reason(&kernel, &devices) {
                        return forced_split_group_status_measurement(
                            &target_ids,
                            MULTI_TARGET_TENSOR_PARALLEL_STRATEGY,
                            "unsupported",
                            payload_bytes,
                            workload,
                            format,
                            &reason,
                        );
                    }
                    run_vulkan_dense_tensor_parallel_group_chain(
                        &devices,
                        &target_ids,
                        payload_bytes,
                        samples,
                        workload,
                        kernel,
                    )
                    .unwrap_or_else(|message| {
                        forced_split_group_status_measurement(
                            &target_ids,
                            MULTI_TARGET_TENSOR_PARALLEL_STRATEGY,
                            "failed",
                            payload_bytes,
                            workload,
                            format,
                            &message,
                        )
                    })
                })
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn opened_serial_group_measurement(
    opened: &[Result<OpenVulkanComputeDevice, String>],
    targets: &[&Target],
    payload_bytes: usize,
    samples: usize,
    format: &str,
    workload: &str,
) -> Option<GroupMeasurement> {
    if !supports_component_chain(workload) {
        return None;
    }
    let target_ids = targets
        .iter()
        .map(|target| target.stable_target_id.clone())
        .collect::<Vec<_>>();
    let mut devices = Vec::with_capacity(opened.len());
    for device in opened {
        match device {
            Ok(device) => devices.push(device),
            Err(message) => {
                return workload_format_kernel(workload, format).map(|_| {
                    forced_split_group_status_measurement(
                        &target_ids,
                        "multi_target_serial",
                        "failed",
                        payload_bytes,
                        workload,
                        format,
                        message,
                    )
                });
            }
        }
    }
    workload_format_kernel_for_devices(workload, format, &devices).map(|kernel| {
        if let Some(reason) = unsupported_kernel_reason(&kernel, &devices) {
            return forced_split_group_status_measurement(
                &target_ids,
                "multi_target_serial",
                "unsupported",
                payload_bytes,
                workload,
                format,
                &reason,
            );
        }
        run_vulkan_dense_serial_group(
            &devices,
            &target_ids,
            payload_bytes,
            samples,
            workload,
            kernel,
        )
        .unwrap_or_else(|message| {
            forced_split_group_status_measurement(
                &target_ids,
                "multi_target_serial",
                "failed",
                payload_bytes,
                workload,
                format,
                &message,
            )
        })
    })
}

#[allow(clippy::too_many_arguments)]
fn vulkan_measurements(
    device: &OpenVulkanComputeDevice,
    target_id: &str,
    payload_bytes: usize,
    samples: usize,
    formats: &[String],
    workloads: &[String],
    _max_group_size: usize,
) -> Vec<Measurement> {
    let mut measurements = formats
        .iter()
        .flat_map(|format| {
            workloads.iter().filter_map(move |workload| {
                workload_format_kernel_for_devices(workload, format, &[device]).map(|kernel| {
                    if let Some(reason) = unsupported_kernel_reason(&kernel, &[device]) {
                        return single_target_status_measurement(
                            target_id,
                            payload_bytes,
                            workload,
                            format,
                            "unsupported",
                            &reason,
                        );
                    }
                    match run_vulkan_dense_projection(
                        device,
                        target_id,
                        payload_bytes,
                        samples,
                        workload,
                        kernel,
                    ) {
                        Ok(measurement) => measurement,
                        Err(message) => single_target_status_measurement(
                            target_id,
                            payload_bytes,
                            workload,
                            format,
                            "failed",
                            &message,
                        ),
                    }
                })
            })
        })
        .collect::<Vec<_>>();

    measurements.extend(vulkan_chain_measurements(
        device,
        target_id,
        payload_bytes,
        samples,
        formats,
        workloads,
        2,
    ));
    measurements
}

#[allow(clippy::too_many_arguments)]
fn vulkan_chain_measurements(
    device: &OpenVulkanComputeDevice,
    target_id: &str,
    payload_bytes: usize,
    samples: usize,
    formats: &[String],
    workloads: &[String],
    stages: usize,
) -> Vec<Measurement> {
    formats
        .iter()
        .flat_map(|format| {
            workloads.iter().filter_map(|workload| {
                if !supports_component_chain(workload) {
                    return None;
                }
                workload_format_kernel_for_devices(workload, format, &[device]).map(|kernel| {
                    if let Some(reason) = unsupported_kernel_reason(&kernel, &[device]) {
                        return single_target_status_measurement_for_regime(
                            target_id,
                            payload_bytes,
                            workload,
                            format,
                            component_chain_regime(stages),
                            "unsupported",
                            &reason,
                        );
                    }
                    run_vulkan_dense_single_target_chain(
                        device,
                        target_id,
                        payload_bytes,
                        samples,
                        workload,
                        kernel,
                        stages,
                    )
                    .unwrap_or_else(|message| {
                        single_target_status_measurement_for_regime(
                            target_id,
                            payload_bytes,
                            workload,
                            format,
                            component_chain_regime(stages),
                            "failed",
                            &message,
                        )
                    })
                })
            })
        })
        .collect()
}

fn supports_component_chain(workload_class: &str) -> bool {
    workload_family(workload_class) == Some("dense_projection")
}

fn supports_tensor_parallel(workload_class: &str) -> bool {
    matches!(
        workload_family(workload_class),
        Some("dense_projection" | "moe_expert")
    )
}

#[cfg(test)]
fn skipped_vulkan_axes<'a>(
    formats: &'a [String],
    workloads: &'a [String],
) -> Vec<(&'a str, &'a str)> {
    formats
        .iter()
        .flat_map(|format| {
            workloads.iter().filter_map(move |workload| {
                if workload_format_kernel(workload, format).is_some() {
                    None
                } else {
                    Some((format.as_str(), workload.as_str()))
                }
            })
        })
        .collect()
}

fn unsupported_kernel_reason(
    kernel: &DenseFormatKernel,
    devices: &[&OpenVulkanComputeDevice],
) -> Option<String> {
    kernel.required_feature.and_then(|required_feature| {
        devices
            .iter()
            .any(|device| !feature_flags_include(&device.feature_flags, required_feature))
            .then(|| format!("required Vulkan feature {required_feature} is not available"))
    })
}

fn feature_flags_include(feature_flags: &[String], required_feature: &str) -> bool {
    feature_flags
        .iter()
        .any(|feature| feature == required_feature)
}

struct OpenVulkanComputeDevice {
    _entry: Entry,
    device: ash::Device,
    instance: ash::Instance,
    compute_queue_family_index: u32,
    queue: vk::Queue,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    timestamp_period_ns: f32,
    timestamp_valid_bits: u32,
    feature_flags: Vec<String>,
    external_memory_fd: bool,
    external_memory_host: bool,
    shared_host_alignment: Option<usize>,
    external_timeline_semaphore: bool,
}

impl Drop for OpenVulkanComputeDevice {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

struct VulkanBuffer {
    device: ash::Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
    _shared_host: Option<Arc<SharedHostAllocation>>,
}

impl Drop for VulkanBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

struct SharedHostAllocation {
    pointer: ptr::NonNull<u8>,
    layout: std::alloc::Layout,
}

struct UnboundExternalBuffer {
    device: ash::Device,
    buffer: Option<vk::Buffer>,
    size: vk::DeviceSize,
    requirements: vk::MemoryRequirements,
    requires_dedicated: bool,
}

impl UnboundExternalBuffer {
    fn into_bound(
        mut self,
        memory: vk::DeviceMemory,
        shared_host: Option<Arc<SharedHostAllocation>>,
    ) -> VulkanBuffer {
        let buffer = self.buffer.take().unwrap();
        VulkanBuffer {
            device: self.device.clone(),
            buffer,
            memory,
            size: self.size,
            _shared_host: shared_host,
        }
    }
}

impl Drop for UnboundExternalBuffer {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            unsafe { self.device.destroy_buffer(buffer, None) };
        }
    }
}

unsafe impl Send for SharedHostAllocation {}
unsafe impl Sync for SharedHostAllocation {}

impl Drop for SharedHostAllocation {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.pointer.as_ptr(), self.layout) };
    }
}

struct TransferContext {
    device: ash::Device,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
}

impl Drop for TransferContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_fence(self.fence, None);
            self.device
                .free_command_buffers(self.command_pool, &[self.command_buffer]);
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}

struct DenseComputeContext {
    device: ash::Device,
    resources: ComputeResources,
    upload: VulkanBuffer,
    readback: VulkanBuffer,
    storage: VulkanBuffer,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    query_pool: vk::QueryPool,
    fence: vk::Fence,
    kernel: DenseFormatKernel,
    plan: ComputePlan,
    buffer_size: vk::DeviceSize,
}

#[derive(Clone)]
struct ComputePlan {
    storage_elements: usize,
    buffer_size: vk::DeviceSize,
    activation_size: vk::DeviceSize,
    output_offset: vk::DeviceSize,
    output_size: vk::DeviceSize,
    dispatch: [u32; 3],
    push_constants: Vec<u32>,
    operations: u64,
}

fn compute_plan_for_payload(payload_bytes: usize, kernel: &DenseFormatKernel) -> ComputePlan {
    match kernel.shape {
        KernelShape::RouterReduction => router_compute_plan(payload_bytes, kernel),
        KernelShape::F32Gemm | KernelShape::PackedGemm
            if kernel.workload == KernelWorkload::MoeExpert =>
        {
            moe_compute_plan(payload_bytes, kernel)
        }
        KernelShape::F32Gemm | KernelShape::PackedGemm => gemm_compute_plan(payload_bytes, kernel),
        KernelShape::KvCache => kv_cache_compute_plan(payload_bytes, kernel),
    }
}

fn kv_cache_compute_plan(payload_bytes: usize, kernel: &DenseFormatKernel) -> ComputePlan {
    let width = 256_usize;
    let query_tokens = kernel.batch_size;
    let words_per_token = width.div_ceil(2);
    let context_tokens = (payload_bytes / (words_per_token * mem::size_of::<u32>())).max(1);
    let state_words = context_tokens * words_per_token;
    let output_words = query_tokens * words_per_token;
    ComputePlan {
        storage_elements: state_words + output_words,
        buffer_size: ((state_words + output_words) * mem::size_of::<u32>()) as vk::DeviceSize,
        activation_size: (state_words * mem::size_of::<u32>()) as vk::DeviceSize,
        output_offset: (state_words * mem::size_of::<u32>()) as vk::DeviceSize,
        output_size: (output_words * mem::size_of::<u32>()) as vk::DeviceSize,
        dispatch: [
            width.div_ceil(2).div_ceil(256) as u32,
            query_tokens as u32,
            1,
        ],
        push_constants: vec![
            width as u32,
            context_tokens as u32,
            query_tokens as u32,
            state_words as u32,
        ],
        operations: 2 * width as u64 * context_tokens as u64 * query_tokens as u64,
    }
}

fn router_compute_plan(payload_bytes: usize, kernel: &DenseFormatKernel) -> ComputePlan {
    let logical_per_storage = kernel.logical_elements_per_storage_element.max(1) as usize;
    let tokens = kernel.batch_size;
    let logical_values = (payload_bytes / mem::size_of::<u32>())
        .max(1)
        .saturating_mul(logical_per_storage);
    let mut experts = (logical_values / tokens).clamp(6, 4092);
    experts -= experts % 6;
    router_compute_plan_for_dimensions(tokens, experts, kernel)
}

fn router_compute_plan_for_dimensions(
    tokens: usize,
    experts: usize,
    kernel: &DenseFormatKernel,
) -> ComputePlan {
    let logical_per_storage = kernel.logical_elements_per_storage_element.max(1) as usize;
    let (input_words, scale_words) = parameter_storage_words(tokens, experts, kernel);
    let scale_offset_words = input_words;
    let output_offset_words = input_words + scale_words;
    let output_words = tokens;
    let storage_elements = output_offset_words + output_words;
    ComputePlan {
        storage_elements,
        buffer_size: (storage_elements * mem::size_of::<u32>()) as vk::DeviceSize,
        activation_size: (input_words * mem::size_of::<u32>()) as vk::DeviceSize,
        output_offset: (output_offset_words * mem::size_of::<u32>()) as vk::DeviceSize,
        output_size: (output_words * mem::size_of::<u32>()) as vk::DeviceSize,
        dispatch: [tokens as u32, 1, 1],
        push_constants: vec![
            tokens as u32,
            experts as u32,
            0,
            output_offset_words as u32,
            logical_per_storage as u32,
            kernel.format_kind,
            scale_offset_words as u32,
            weight_group_size(kernel) as u32,
        ],
        operations: tokens as u64 * experts as u64 * 2,
    }
}

fn gemm_compute_plan(payload_bytes: usize, kernel: &DenseFormatKernel) -> ComputePlan {
    let logical_per_storage = kernel.logical_elements_per_storage_element.clamp(1, 8) as usize;
    let upper = ((payload_bytes / mem::size_of::<u32>()).saturating_mul(logical_per_storage) as f64)
        .sqrt()
        .floor() as usize;
    let mut low = 1_usize;
    let mut high = (upper / 12).max(1);
    while low < high {
        let midpoint = (low + high).div_ceil(2);
        let hidden = midpoint * 12;
        let words = parameter_storage_words(hidden, hidden, kernel);
        if (words.0 + words.1) * mem::size_of::<u32>() <= payload_bytes {
            low = midpoint;
        } else {
            high = midpoint - 1;
        }
    }
    let hidden = (low * 12).max(12);
    let m = kernel.batch_size;
    gemm_compute_plan_for_dimensions(m, hidden, hidden, kernel)
}

fn moe_compute_plan(payload_bytes: usize, kernel: &DenseFormatKernel) -> ComputePlan {
    let logical_per_storage = kernel.logical_elements_per_storage_element.clamp(1, 8) as usize;
    let logical_budget = (payload_bytes / mem::size_of::<u32>())
        .saturating_mul(logical_per_storage)
        / MOE_SELECTED_EXPERTS;
    let upper = (logical_budget as f64).sqrt().floor() as usize;
    let mut low = 1_usize;
    let mut high = (upper / 12).max(1);
    while low < high {
        let midpoint = (low + high).div_ceil(2);
        let hidden = midpoint * 12;
        let words = parameter_storage_words(MOE_SELECTED_EXPERTS * hidden, hidden, kernel);
        if (words.0 + words.1) * mem::size_of::<u32>() <= payload_bytes {
            low = midpoint;
        } else {
            high = midpoint - 1;
        }
    }
    let hidden = (low * 12).max(12);
    gemm_compute_plan_for_dimensions(
        kernel.batch_size,
        MOE_SELECTED_EXPERTS * hidden,
        hidden,
        kernel,
    )
}

fn gemm_compute_plan_for_dimensions(
    m: usize,
    n: usize,
    k: usize,
    kernel: &DenseFormatKernel,
) -> ComputePlan {
    let activation_words = (m * k).div_ceil(2);
    let (weight_words, scale_words) = parameter_storage_words(n, k, kernel);
    let weight_storage_offset = activation_words;
    let scale_storage_offset = weight_storage_offset + weight_words;
    let c_storage_offset = scale_storage_offset + scale_words;
    let output_words = (m * n).div_ceil(2);
    let storage_elements = c_storage_offset + output_words;
    ComputePlan {
        storage_elements,
        buffer_size: (storage_elements * kernel.bytes_per_storage_element) as vk::DeviceSize,
        activation_size: (activation_words * mem::size_of::<u32>()) as vk::DeviceSize,
        output_offset: (c_storage_offset * mem::size_of::<u32>()) as vk::DeviceSize,
        output_size: (output_words * mem::size_of::<u32>()) as vk::DeviceSize,
        dispatch: [n.div_ceil(32) as u32, m.div_ceil(16) as u32, 1],
        push_constants: vec![
            m as u32,
            n as u32,
            k as u32,
            0,
            weight_storage_offset as u32,
            c_storage_offset as u32,
            kernel.logical_elements_per_storage_element as u32,
            kernel.format_kind,
            scale_storage_offset as u32,
            weight_group_size(kernel) as u32,
        ],
        operations: 2 * m as u64 * n as u64 * k as u64,
    }
}

fn parameter_storage_words(
    rows: usize,
    columns: usize,
    kernel: &DenseFormatKernel,
) -> (usize, usize) {
    match kernel.weight_layout {
        WeightLayout::Plain => (
            (rows * columns).div_ceil(kernel.logical_elements_per_storage_element.max(1) as usize),
            0,
        ),
        WeightLayout::Bf16Scaled { group_size } => {
            let value_words = (rows * columns)
                .div_ceil(kernel.logical_elements_per_storage_element.max(1) as usize);
            let scale_count = rows * columns.div_ceil(group_size);
            (value_words, scale_count.div_ceil(2))
        }
        WeightLayout::E8m0Scaled { group_size } => {
            let value_words = (rows * columns)
                .div_ceil(kernel.logical_elements_per_storage_element.max(1) as usize);
            let scale_count = rows * columns.div_ceil(group_size);
            (value_words, scale_count.div_ceil(4))
        }
        WeightLayout::NerveQ8_0 => (rows * columns.div_ceil(32) * 9, 0),
    }
}

fn weight_group_size(kernel: &DenseFormatKernel) -> usize {
    match kernel.weight_layout {
        WeightLayout::Bf16Scaled { group_size } | WeightLayout::E8m0Scaled { group_size } => {
            group_size
        }
        WeightLayout::NerveQ8_0 => 32,
        WeightLayout::Plain => 1,
    }
}

fn split_extent(extent: usize, parts: usize) -> Vec<usize> {
    let base = extent / parts;
    let remainder = extent % parts;
    (0..parts)
        .map(|index| base + usize::from(index < remainder))
        .collect()
}

fn split_extent_aligned(
    extent: usize,
    parts: usize,
    alignment: usize,
) -> Result<Vec<usize>, String> {
    if parts == 0 || alignment == 0 || !extent.is_multiple_of(alignment) {
        return Err(format!(
            "extent {extent} cannot be split into {parts} parts aligned to {alignment}"
        ));
    }
    let units = extent / alignment;
    if units < parts {
        return Err(format!(
            "extent {extent} has fewer than {parts} aligned shard units"
        ));
    }
    Ok(split_extent(units, parts)
        .into_iter()
        .map(|units| units * alignment)
        .collect())
}

impl Drop for DenseComputeContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_query_pool(self.query_pool, None);
            self.device
                .free_command_buffers(self.command_pool, &[self.command_buffer]);
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SharedTransferRoute {
    ExternalDeviceLocal,
    SharedHost,
}

impl SharedTransferRoute {
    fn name(self) -> &'static str {
        match self {
            Self::ExternalDeviceLocal => "external_device_local",
            Self::SharedHost => "shared_host",
        }
    }
}

struct SharedTransferBuffers {
    route: SharedTransferRoute,
    source: VulkanBuffer,
    destination: VulkanBuffer,
    timeline: Option<ExternalTimelineSemaphore>,
}

struct SharedMultiBuffer {
    buffers: Vec<VulkanBuffer>,
}

struct ExternalTimelineSemaphore {
    source_device: ash::Device,
    destination_device: ash::Device,
    source: vk::Semaphore,
    destination: vk::Semaphore,
    next_value: u64,
}

impl Drop for ExternalTimelineSemaphore {
    fn drop(&mut self) {
        unsafe {
            self.source_device.destroy_semaphore(self.source, None);
            self.destination_device
                .destroy_semaphore(self.destination, None);
        }
    }
}

struct TransferSampleMetrics {
    bytes_read: u64,
    bytes_written: u64,
}

enum ComputeOutputTransferRoute {
    Shared {
        buffers: SharedTransferBuffers,
        source_transfer: TransferContext,
        destination_transfer: TransferContext,
    },
    HostStaged {
        destination_transfer: TransferContext,
        host_stage: Vec<u8>,
    },
}

struct ComputeOutputTransfer {
    route: ComputeOutputTransferRoute,
    size: vk::DeviceSize,
}

impl ComputeOutputTransfer {
    fn route_name(&self) -> &'static str {
        match &self.route {
            ComputeOutputTransferRoute::Shared { buffers, .. } => buffers.route.name(),
            ComputeOutputTransferRoute::HostStaged { .. } => "host_staged",
        }
    }

    fn requires_compute_readback(&self) -> bool {
        matches!(self.route, ComputeOutputTransferRoute::HostStaged { .. })
    }
}

fn create_compute_output_transfer(
    source_device: &OpenVulkanComputeDevice,
    destination_device: &OpenVulkanComputeDevice,
    source_context: &DenseComputeContext,
    destination_context: &DenseComputeContext,
    requested_bytes: usize,
) -> Result<ComputeOutputTransfer, String> {
    let size = requested_bytes
        .min(source_context.plan.output_size as usize)
        .min(destination_context.upload.size as usize)
        .min(destination_context.storage.size as usize)
        .max(mem::size_of::<u32>());
    let size = size as vk::DeviceSize;
    let route = match create_shared_transfer_buffers(source_device, destination_device, size) {
        Ok(buffers) => ComputeOutputTransferRoute::Shared {
            buffers,
            source_transfer: create_transfer_context(source_device)?,
            destination_transfer: create_transfer_context(destination_device)?,
        },
        Err(_) => ComputeOutputTransferRoute::HostStaged {
            destination_transfer: create_transfer_context(destination_device)?,
            host_stage: vec![0_u8; size as usize],
        },
    };
    Ok(ComputeOutputTransfer { route, size })
}

fn transfer_compute_output_to_input(
    source_device: &OpenVulkanComputeDevice,
    destination_device: &OpenVulkanComputeDevice,
    source_context: &DenseComputeContext,
    destination_context: &DenseComputeContext,
    transfer: &mut ComputeOutputTransfer,
    destination_offset: vk::DeviceSize,
) -> Result<TransferSampleMetrics, String> {
    match &mut transfer.route {
        ComputeOutputTransferRoute::Shared {
            buffers,
            source_transfer,
            destination_transfer,
        } => {
            submit_shared_buffer_transfer(
                source_device,
                destination_device,
                source_transfer,
                destination_transfer,
                source_context.storage.buffer,
                source_context.plan.output_offset,
                destination_context.storage.buffer,
                destination_offset,
                buffers,
                transfer.size,
            )?;
        }
        ComputeOutputTransferRoute::HostStaged {
            destination_transfer,
            host_stage,
        } => {
            read_buffer_bytes(&source_device.device, &source_context.readback, host_stage)?;
            write_buffer_bytes(
                &destination_device.device,
                &destination_context.upload,
                host_stage,
            )?;
            submit_copy_buffer_region(
                destination_device,
                destination_transfer,
                destination_context.upload.buffer,
                0,
                destination_context.storage.buffer,
                destination_offset,
                transfer.size,
            )?;
            black_box(checksum_bytes(host_stage));
        }
    }
    Ok(TransferSampleMetrics {
        bytes_read: transfer.size as u64,
        bytes_written: transfer.size as u64,
    })
}

fn create_transfer_context(
    compute_device: &OpenVulkanComputeDevice,
) -> Result<TransferContext, String> {
    let command_pool = unsafe {
        compute_device.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(compute_device.compute_queue_family_index),
            None,
        )
    }
    .map_err(|error| format!("could not create Vulkan transfer command pool: {error:?}"))?;
    let command_buffer = match unsafe {
        compute_device.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
    } {
        Ok(command_buffers) => command_buffers[0],
        Err(error) => {
            unsafe {
                compute_device
                    .device
                    .destroy_command_pool(command_pool, None)
            };
            return Err(format!(
                "could not allocate Vulkan transfer command buffer: {error:?}"
            ));
        }
    };
    let fence = match unsafe {
        compute_device
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
    } {
        Ok(fence) => fence,
        Err(error) => {
            unsafe {
                compute_device
                    .device
                    .free_command_buffers(command_pool, &[command_buffer]);
                compute_device
                    .device
                    .destroy_command_pool(command_pool, None);
            }
            return Err(format!("could not create Vulkan transfer fence: {error:?}"));
        }
    };
    Ok(TransferContext {
        device: compute_device.device.clone(),
        command_pool,
        command_buffer,
        fence,
    })
}

fn submit_copy_buffer(
    compute_device: &OpenVulkanComputeDevice,
    context: &TransferContext,
    source: vk::Buffer,
    destination: vk::Buffer,
    size: vk::DeviceSize,
) -> Result<(), String> {
    submit_copy_buffer_region(compute_device, context, source, 0, destination, 0, size)
}

#[allow(clippy::too_many_arguments)]
fn submit_copy_buffer_region(
    compute_device: &OpenVulkanComputeDevice,
    context: &TransferContext,
    source: vk::Buffer,
    source_offset: vk::DeviceSize,
    destination: vk::Buffer,
    destination_offset: vk::DeviceSize,
    size: vk::DeviceSize,
) -> Result<(), String> {
    unsafe {
        compute_device
            .device
            .reset_command_buffer(context.command_buffer, vk::CommandBufferResetFlags::empty())
            .map_err(|error| {
                format!("could not reset Vulkan transfer command buffer: {error:?}")
            })?;
        compute_device
            .device
            .begin_command_buffer(
                context.command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(|error| {
                format!("could not begin Vulkan transfer command buffer: {error:?}")
            })?;
        let source_visibility = [vk::BufferMemoryBarrier::default()
            .src_access_mask(
                vk::AccessFlags::HOST_WRITE
                    | vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::TRANSFER_WRITE,
            )
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(source)
            .offset(source_offset)
            .size(size)];
        compute_device.device.cmd_pipeline_barrier(
            context.command_buffer,
            vk::PipelineStageFlags::HOST
                | vk::PipelineStageFlags::COMPUTE_SHADER
                | vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &source_visibility,
            &[],
        );
        compute_device.device.cmd_copy_buffer(
            context.command_buffer,
            source,
            destination,
            &[vk::BufferCopy::default()
                .src_offset(source_offset)
                .dst_offset(destination_offset)
                .size(size)],
        );
        let destination_visibility = [vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(
                vk::AccessFlags::HOST_READ
                    | vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::TRANSFER_READ
                    | vk::AccessFlags::TRANSFER_WRITE,
            )
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(destination)
            .offset(destination_offset)
            .size(size)];
        compute_device.device.cmd_pipeline_barrier(
            context.command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST
                | vk::PipelineStageFlags::COMPUTE_SHADER
                | vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &destination_visibility,
            &[],
        );
        compute_device
            .device
            .end_command_buffer(context.command_buffer)
            .map_err(|error| format!("could not end Vulkan transfer command buffer: {error:?}"))?;
        compute_device
            .device
            .reset_fences(&[context.fence])
            .map_err(|error| format!("could not reset Vulkan transfer fence: {error:?}"))?;
        compute_device
            .device
            .queue_submit(
                compute_device.queue,
                &[vk::SubmitInfo::default().command_buffers(&[context.command_buffer])],
                context.fence,
            )
            .map_err(|error| format!("could not submit Vulkan transfer work: {error:?}"))?;
        compute_device
            .device
            .wait_for_fences(&[context.fence], true, MAX_VULKAN_WAIT_NS)
            .map_err(|error| format!("could not wait for Vulkan transfer work: {error:?}"))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn record_copy_buffer_region(
    compute_device: &OpenVulkanComputeDevice,
    context: &TransferContext,
    source: vk::Buffer,
    source_offset: vk::DeviceSize,
    destination: vk::Buffer,
    destination_offset: vk::DeviceSize,
    size: vk::DeviceSize,
) -> Result<(), String> {
    unsafe {
        compute_device
            .device
            .reset_command_buffer(context.command_buffer, vk::CommandBufferResetFlags::empty())
            .map_err(|error| {
                format!("could not reset Vulkan transfer command buffer: {error:?}")
            })?;
        compute_device
            .device
            .begin_command_buffer(
                context.command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(|error| {
                format!("could not begin Vulkan transfer command buffer: {error:?}")
            })?;
        let source_visibility = [vk::BufferMemoryBarrier::default()
            .src_access_mask(
                vk::AccessFlags::HOST_WRITE
                    | vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::TRANSFER_WRITE,
            )
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(source)
            .offset(source_offset)
            .size(size)];
        compute_device.device.cmd_pipeline_barrier(
            context.command_buffer,
            vk::PipelineStageFlags::HOST
                | vk::PipelineStageFlags::COMPUTE_SHADER
                | vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &source_visibility,
            &[],
        );
        compute_device.device.cmd_copy_buffer(
            context.command_buffer,
            source,
            destination,
            &[vk::BufferCopy::default()
                .src_offset(source_offset)
                .dst_offset(destination_offset)
                .size(size)],
        );
        let destination_visibility = [vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(
                vk::AccessFlags::HOST_READ
                    | vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::TRANSFER_READ
                    | vk::AccessFlags::TRANSFER_WRITE,
            )
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(destination)
            .offset(destination_offset)
            .size(size)];
        compute_device.device.cmd_pipeline_barrier(
            context.command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST
                | vk::PipelineStageFlags::COMPUTE_SHADER
                | vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &destination_visibility,
            &[],
        );
        compute_device
            .device
            .end_command_buffer(context.command_buffer)
            .map_err(|error| format!("could not end Vulkan transfer command buffer: {error:?}"))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn submit_shared_buffer_transfer(
    source_device: &OpenVulkanComputeDevice,
    destination_device: &OpenVulkanComputeDevice,
    source_context: &TransferContext,
    destination_context: &TransferContext,
    source_buffer: vk::Buffer,
    source_offset: vk::DeviceSize,
    destination_buffer: vk::Buffer,
    destination_offset: vk::DeviceSize,
    shared: &mut SharedTransferBuffers,
    size: vk::DeviceSize,
) -> Result<(), String> {
    let Some(timeline) = &mut shared.timeline else {
        submit_copy_buffer_region(
            source_device,
            source_context,
            source_buffer,
            source_offset,
            shared.source.buffer,
            0,
            size,
        )?;
        return submit_copy_buffer_region(
            destination_device,
            destination_context,
            shared.destination.buffer,
            0,
            destination_buffer,
            destination_offset,
            size,
        );
    };

    record_copy_buffer_region(
        source_device,
        source_context,
        source_buffer,
        source_offset,
        shared.source.buffer,
        0,
        size,
    )?;
    record_copy_buffer_region(
        destination_device,
        destination_context,
        shared.destination.buffer,
        0,
        destination_buffer,
        destination_offset,
        size,
    )?;
    timeline.next_value = timeline
        .next_value
        .checked_add(1)
        .ok_or_else(|| "external timeline semaphore value overflowed".to_string())?;
    let value = timeline.next_value;
    let command_buffers = [source_context.command_buffer];
    let signal_semaphores = [timeline.source];
    let signal_values = [value];
    let mut signal_timeline =
        vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&signal_values);
    let source_submit = [vk::SubmitInfo::default()
        .command_buffers(&command_buffers)
        .signal_semaphores(&signal_semaphores)
        .push_next(&mut signal_timeline)];
    let destination_command_buffers = [destination_context.command_buffer];
    let wait_semaphores = [timeline.destination];
    let wait_values = [value];
    let wait_stages = [vk::PipelineStageFlags::TRANSFER];
    let mut wait_timeline =
        vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&wait_values);
    let destination_submit = [vk::SubmitInfo::default()
        .command_buffers(&destination_command_buffers)
        .wait_semaphores(&wait_semaphores)
        .wait_dst_stage_mask(&wait_stages)
        .push_next(&mut wait_timeline)];
    unsafe {
        source_device
            .device
            .reset_fences(&[source_context.fence])
            .map_err(|error| format!("could not reset source transfer fence: {error:?}"))?;
        destination_device
            .device
            .reset_fences(&[destination_context.fence])
            .map_err(|error| format!("could not reset destination transfer fence: {error:?}"))?;
        source_device
            .device
            .queue_submit(source_device.queue, &source_submit, source_context.fence)
            .map_err(|error| format!("could not submit source timeline transfer: {error:?}"))?;
        destination_device
            .device
            .queue_submit(
                destination_device.queue,
                &destination_submit,
                destination_context.fence,
            )
            .map_err(|error| {
                format!("could not submit destination timeline transfer: {error:?}")
            })?;
        destination_device
            .device
            .wait_for_fences(&[destination_context.fence], true, MAX_VULKAN_WAIT_NS)
            .map_err(|error| format!("could not wait for timeline transfer: {error:?}"))?;
    }
    Ok(())
}

fn run_vulkan_dense_projection(
    compute_device: &OpenVulkanComputeDevice,
    target_id: &str,
    payload_bytes: usize,
    samples: usize,
    workload_class: &str,
    kernel: DenseFormatKernel,
) -> Result<Measurement, String> {
    let context = create_dense_compute_context(compute_device, payload_bytes, kernel)?;
    let mut measured_samples = Vec::with_capacity(samples);
    for iteration in 0..=samples {
        let sample_index = iteration.saturating_sub(1);
        let sample = submit_dense_compute_sample(compute_device, &context, sample_index, false)?;
        if iteration == 0 {
            continue;
        }
        measured_samples.push(sample);
    }
    validate_compute_output(compute_device, &context)?;

    Ok(Measurement {
        workload_id: format!(
            "single_target_small_payload:{workload_class}:{}",
            context.kernel.format
        ),
        comparison_group: "small_payload_placement_comparison".to_string(),
        workload_class: workload_class.to_string(),
        placement_strategy: "single_target_serial".to_string(),
        target_id: target_id.to_string(),
        pattern: kernel_identity(&context.kernel),
        operation_family: workload_class.to_string(),
        regime: SINGLE_COMPONENT_REGIME.to_string(),
        format: context.kernel.format.to_string(),
        status: "completed".to_string(),
        reason: None,
        payload_bytes,
        working_set_bytes: context.buffer_size as usize,
        activation_bytes: context.plan.activation_size as usize,
        output_bytes: context.plan.output_size as usize,
        summary: summarize_samples(&measured_samples),
        samples: measured_samples,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_vulkan_dense_single_target_chain(
    device: &OpenVulkanComputeDevice,
    target_id: &str,
    payload_bytes: usize,
    samples: usize,
    workload_class: &str,
    kernel: DenseFormatKernel,
    stages: usize,
) -> Result<Measurement, String> {
    let result = run_direct_local_serial_chain(device, payload_bytes, samples, &kernel, stages)?;
    let summary = summarize_samples(&result.samples);
    Ok(Measurement {
        workload_id: format!(
            "single_target_{stages}_component_chain:{workload_class}:{}",
            kernel.format
        ),
        comparison_group: "small_payload_placement_comparison".to_string(),
        workload_class: workload_class.to_string(),
        placement_strategy: "single_target_serial".to_string(),
        target_id: target_id.to_string(),
        pattern: format!(
            "{}:{stages}_component_chain:transport={}",
            kernel_identity(&kernel),
            result.transport
        ),
        operation_family: workload_class.to_string(),
        regime: component_chain_regime(stages).to_string(),
        format: kernel.format,
        status: "completed".to_string(),
        reason: None,
        payload_bytes,
        working_set_bytes: result.parameter_bytes_per_stage.iter().sum::<usize>()
            + result.activation_bytes * (stages + 1),
        activation_bytes: result.activation_bytes,
        output_bytes: result.output_bytes,
        summary,
        samples: result.samples,
    })
}

fn single_target_status_measurement_for_regime(
    target_id: &str,
    payload_bytes: usize,
    workload_class: &str,
    format: &str,
    regime: &str,
    status: &str,
    reason: &str,
) -> Measurement {
    let mut measurement = single_target_status_measurement(
        target_id,
        payload_bytes,
        workload_class,
        format,
        status,
        reason,
    );
    measurement.regime = regime.to_string();
    measurement.workload_id = format!("single_target_{regime}:{workload_class}:{format}");
    measurement
}

fn create_dense_compute_context(
    compute_device: &OpenVulkanComputeDevice,
    payload_bytes: usize,
    kernel: DenseFormatKernel,
) -> Result<DenseComputeContext, String> {
    let plan = compute_plan_for_payload(payload_bytes, &kernel);
    create_dense_compute_context_with_plan(compute_device, kernel, plan)
}

fn create_dense_compute_context_with_plan(
    compute_device: &OpenVulkanComputeDevice,
    kernel: DenseFormatKernel,
    plan: ComputePlan,
) -> Result<DenseComputeContext, String> {
    if let Some(required_feature) = kernel.required_feature {
        if !compute_device
            .feature_flags
            .iter()
            .any(|feature| feature == required_feature)
        {
            return Err(format!(
                "selected Vulkan device does not support required feature {required_feature}"
            ));
        }
    }
    if compute_device.timestamp_valid_bits == 0 || compute_device.timestamp_period_ns <= 0.0 {
        return Err("selected Vulkan compute queue does not expose usable timestamps".to_string());
    }

    let buffer_size = plan.buffer_size;
    let upload = create_buffer(
        compute_device,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let readback = create_buffer(
        compute_device,
        plan.output_size.max(plan.activation_size),
        vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let storage = create_buffer(
        compute_device,
        buffer_size,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    fill_upload_buffer(&compute_device.device, &upload, &plan, &kernel)?;
    let initialization = create_transfer_context(compute_device)?;
    submit_copy_buffer(
        compute_device,
        &initialization,
        upload.buffer,
        storage.buffer,
        buffer_size,
    )?;

    let resources =
        create_compute_resources(compute_device, storage.buffer, buffer_size, kernel.shader)?;
    let command_pool = unsafe {
        compute_device.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(compute_device.compute_queue_family_index),
            None,
        )
    }
    .map_err(|error| format!("could not create Vulkan command pool: {error:?}"))?;
    let command_buffer = unsafe {
        compute_device.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
    }
    .map_err(|error| format!("could not allocate Vulkan command buffer: {error:?}"))?[0];
    let query_pool = unsafe {
        compute_device.device.create_query_pool(
            &vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(2),
            None,
        )
    }
    .map_err(|error| format!("could not create Vulkan timestamp query pool: {error:?}"))?;
    let fence = unsafe {
        compute_device
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
    }
    .map_err(|error| format!("could not create Vulkan fence: {error:?}"))?;
    Ok(DenseComputeContext {
        device: compute_device.device.clone(),
        resources,
        upload,
        readback,
        storage,
        command_pool,
        command_buffer,
        query_pool,
        fence,
        kernel,
        plan,
        buffer_size,
    })
}

fn submit_dense_compute_sample(
    compute_device: &OpenVulkanComputeDevice,
    context: &DenseComputeContext,
    sample_index: usize,
    readback_output: bool,
) -> Result<Sample, String> {
    record_compute_dispatch(
        compute_device,
        &context.resources,
        context.command_buffer,
        context.query_pool,
        context.storage.buffer,
        context.readback.buffer,
        &context.plan,
        readback_output,
    )?;
    let started = Instant::now();
    unsafe {
        compute_device
            .device
            .reset_fences(&[context.fence])
            .map_err(|error| format!("could not reset Vulkan fence: {error:?}"))?;
        compute_device
            .device
            .queue_submit(
                compute_device.queue,
                &[vk::SubmitInfo::default().command_buffers(&[context.command_buffer])],
                context.fence,
            )
            .map_err(|error| format!("could not submit Vulkan compute work: {error:?}"))?;
        compute_device
            .device
            .wait_for_fences(&[context.fence], true, MAX_VULKAN_WAIT_NS)
            .map_err(|error| format!("could not wait for Vulkan compute work: {error:?}"))?;
    }
    let wall_duration_ns = started.elapsed().as_nanos();
    let gpu_duration_ns = read_timestamp_duration_ns(compute_device, context.query_pool)?;
    Ok(Sample {
        sample_index,
        duration_ns: wall_duration_ns.max(gpu_duration_ns),
        iterations: 1,
        bytes_read: context.plan.output_offset as u64,
        bytes_written: context.plan.output_size as u64,
        operations: context.plan.operations,
    })
}

fn validate_compute_output(
    compute_device: &OpenVulkanComputeDevice,
    context: &DenseComputeContext,
) -> Result<(), String> {
    let transfer = create_transfer_context(compute_device)?;
    submit_copy_buffer_region(
        compute_device,
        &transfer,
        context.storage.buffer,
        context.plan.output_offset,
        context.readback.buffer,
        0,
        context.plan.output_size,
    )?;
    let mut bytes = vec![0_u8; context.plan.output_size as usize];
    read_buffer_bytes(&compute_device.device, &context.readback, &mut bytes)?;
    match context.kernel.shape {
        KernelShape::F32Gemm | KernelShape::PackedGemm | KernelShape::KvCache => {
            for (index, chunk) in bytes.chunks_exact(mem::size_of::<u16>()).enumerate() {
                let bits = u16::from_le_bytes(chunk.try_into().unwrap());
                let value = f32::from_bits(u32::from(bits) << 16);
                if !value.is_finite() {
                    return Err(format!(
                        "output value {index} was not written to a finite BF16 value"
                    ));
                }
            }
        }
        KernelShape::RouterReduction => {
            for (index, chunk) in bytes.chunks_exact(mem::size_of::<f32>()).enumerate() {
                let value = f32::from_le_bytes(chunk.try_into().unwrap());
                if !value.is_finite() {
                    return Err(format!(
                        "router output value {index} was not written to a finite F32 value"
                    ));
                }
            }
        }
    }
    black_box(checksum_bytes(&bytes));
    Ok(())
}

fn run_vulkan_dense_serial_pair(
    source_device: &OpenVulkanComputeDevice,
    destination_device: &OpenVulkanComputeDevice,
    source_id: &str,
    destination_id: &str,
    payload_bytes: usize,
    samples: usize,
    workload_class: &str,
    kernel: DenseFormatKernel,
) -> Result<PairMeasurement, String> {
    match run_direct_external_serial_chain(
        &[source_device, destination_device],
        payload_bytes,
        samples,
        &kernel,
    ) {
        Ok(result) => {
            let summary = summarize_samples(&result.samples);
            Ok(PairMeasurement {
                workload_id: format_workload_id(
                    "synthetic_layer_split_small_payload",
                    workload_class,
                    &kernel.format,
                ),
                comparison_group: "small_payload_placement_comparison".to_string(),
                workload_class: workload_class.to_string(),
                placement_strategy: "two_target_serial".to_string(),
                source_target_id: source_id.to_string(),
                destination_target_id: destination_id.to_string(),
                pattern: format!(
                    "{}:transport={}",
                    kernel_identity(&kernel),
                    result.transport
                ),
                operation_family: workload_class.to_string(),
                regime: TWO_COMPONENT_CHAIN_REGIME.to_string(),
                format: kernel.format,
                status: "completed".to_string(),
                reason: None,
                payload_bytes,
                source_payload_bytes: result.parameter_bytes_per_stage[0],
                destination_payload_bytes: result.parameter_bytes_per_stage[1],
                activation_bytes: result.activation_bytes,
                output_bytes: result.output_bytes,
                samples: result.samples,
                summary,
            })
        }
        Err(direct_error) => run_vulkan_dense_serial_pair_staged(
                source_device,
                destination_device,
                source_id,
                destination_id,
                payload_bytes,
                samples,
                workload_class,
                kernel,
            )
            .map_err(|staged_error| {
                format!(
                    "direct serial route failed ({direct_error}); staged serial route failed ({staged_error})"
                )
            }),
    }
}

fn run_vulkan_dense_serial_pair_staged(
    source_device: &OpenVulkanComputeDevice,
    destination_device: &OpenVulkanComputeDevice,
    source_id: &str,
    destination_id: &str,
    payload_bytes: usize,
    samples: usize,
    workload_class: &str,
    kernel: DenseFormatKernel,
) -> Result<PairMeasurement, String> {
    let source_payload_bytes = payload_bytes / 2;
    let destination_payload_bytes = payload_bytes - source_payload_bytes;
    let requested_activation_bytes = activation_bytes_for_payload(payload_bytes);
    let source_context =
        create_dense_compute_context(source_device, source_payload_bytes, kernel.clone())?;
    let destination_context = create_dense_compute_context(
        destination_device,
        destination_payload_bytes,
        kernel.clone(),
    )?;
    let mut transfer = create_compute_output_transfer(
        source_device,
        destination_device,
        &source_context,
        &destination_context,
        requested_activation_bytes,
    )?;
    let transfer_route = transfer.route_name();
    let activation_bytes = transfer.size as usize;
    let mut measured_samples = Vec::with_capacity(samples);
    for iteration in 0..=samples {
        let sample_index = iteration.saturating_sub(1);
        let started = Instant::now();
        let source_sample = submit_dense_compute_sample(
            source_device,
            &source_context,
            sample_index,
            transfer.requires_compute_readback(),
        )?;
        let transfer_sample = transfer_compute_output_to_input(
            source_device,
            destination_device,
            &source_context,
            &destination_context,
            &mut transfer,
            0,
        )?;
        let destination_sample = submit_dense_compute_sample(
            destination_device,
            &destination_context,
            sample_index,
            false,
        )?;
        let duration = started.elapsed();
        if iteration == 0 {
            continue;
        }
        measured_samples.push(Sample {
            sample_index,
            duration_ns: duration.as_nanos(),
            iterations: 1,
            bytes_read: source_sample.bytes_read
                + destination_sample.bytes_read
                + transfer_sample.bytes_read,
            bytes_written: source_sample.bytes_written
                + destination_sample.bytes_written
                + transfer_sample.bytes_written,
            operations: source_sample.operations + destination_sample.operations,
        });
    }
    validate_compute_output(destination_device, &destination_context)?;

    let summary = summarize_samples(&measured_samples);
    Ok(PairMeasurement {
        workload_id: format_workload_id(
            "synthetic_layer_split_small_payload",
            workload_class,
            &kernel.format,
        ),
        comparison_group: "small_payload_placement_comparison".to_string(),
        workload_class: workload_class.to_string(),
        placement_strategy: "two_target_serial".to_string(),
        source_target_id: source_id.to_string(),
        destination_target_id: destination_id.to_string(),
        pattern: format!("{}:transport={transfer_route}", kernel_identity(&kernel)),
        operation_family: workload_class.to_string(),
        regime: TWO_COMPONENT_CHAIN_REGIME.to_string(),
        format: kernel.format,
        status: "completed".to_string(),
        reason: None,
        payload_bytes,
        source_payload_bytes: (source_context.plan.output_offset
            - source_context.plan.activation_size) as usize,
        destination_payload_bytes: (destination_context.plan.output_offset
            - destination_context.plan.activation_size) as usize,
        activation_bytes,
        output_bytes: destination_context.plan.output_size as usize,
        samples: measured_samples,
        summary,
    })
}

#[derive(Clone)]
struct TensorParallelShardPlan {
    parameter_words: usize,
    scale_words: usize,
    parameter_word_offset: usize,
    dispatch: [u32; 3],
    push_constants: Vec<u32>,
    operations: u64,
}

struct TensorParallelShardContext {
    device: ash::Device,
    resources: ComputeResources,
    _upload: VulkanBuffer,
    weights: VulkanBuffer,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    query_pool: vk::QueryPool,
    fence: vk::Fence,
    plan: TensorParallelShardPlan,
}

impl Drop for TensorParallelShardContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_query_pool(self.query_pool, None);
            self.device
                .free_command_buffers(self.command_pool, &[self.command_buffer]);
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}

struct TensorParallelSynchronization {
    ready: ExternalTimelineSemaphore,
    done: ExternalTimelineSemaphore,
}

struct TensorParallelRunResult {
    samples: Vec<Sample>,
    parameter_bytes_per_participant: Vec<usize>,
    activation_bytes: usize,
    output_bytes: usize,
    transport: &'static str,
}

struct SerialChainRunResult {
    samples: Vec<Sample>,
    parameter_bytes_per_stage: Vec<usize>,
    activation_bytes: usize,
    output_bytes: usize,
    transport: &'static str,
}

fn run_staged_serial_chain(
    devices: &[&OpenVulkanComputeDevice],
    payload_bytes: usize,
    samples: usize,
    kernel: &DenseFormatKernel,
) -> Result<SerialChainRunResult, String> {
    if devices.len() < 2 {
        return Err("staged serial chain requires at least two devices".to_string());
    }
    let stage_payload = payload_bytes / devices.len();
    let contexts = devices
        .iter()
        .map(|device| create_dense_compute_context(device, stage_payload, kernel.clone()))
        .collect::<Result<Vec<_>, _>>()?;
    let requested_activation_bytes = activation_bytes_for_payload(payload_bytes);
    let mut transfers = (0..devices.len() - 1)
        .map(|stage| {
            create_compute_output_transfer(
                devices[stage],
                devices[stage + 1],
                &contexts[stage],
                &contexts[stage + 1],
                requested_activation_bytes,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let transport = if transfers
        .iter()
        .map(ComputeOutputTransfer::route_name)
        .all(|route| route == transfers[0].route_name())
    {
        transfers[0].route_name()
    } else {
        "mixed"
    };
    let mut measured_samples = Vec::with_capacity(samples);
    for iteration in 0..=samples {
        let sample_index = iteration.saturating_sub(1);
        let started = Instant::now();
        let mut bytes_read = 0_u64;
        let mut bytes_written = 0_u64;
        let mut operations = 0_u64;
        for stage in 0..devices.len() {
            let sample = submit_dense_compute_sample(
                devices[stage],
                &contexts[stage],
                sample_index,
                stage + 1 < devices.len() && transfers[stage].requires_compute_readback(),
            )?;
            bytes_read += sample.bytes_read;
            bytes_written += sample.bytes_written;
            operations += sample.operations;
            if stage + 1 < devices.len() {
                let transfer = transfer_compute_output_to_input(
                    devices[stage],
                    devices[stage + 1],
                    &contexts[stage],
                    &contexts[stage + 1],
                    &mut transfers[stage],
                    0,
                )?;
                bytes_read += transfer.bytes_read;
                bytes_written += transfer.bytes_written;
            }
        }
        if iteration == 0 {
            continue;
        }
        measured_samples.push(Sample {
            sample_index,
            duration_ns: started.elapsed().as_nanos(),
            iterations: 1,
            bytes_read,
            bytes_written,
            operations,
        });
    }
    validate_compute_output(devices[devices.len() - 1], &contexts[contexts.len() - 1])?;
    Ok(SerialChainRunResult {
        samples: measured_samples,
        parameter_bytes_per_stage: contexts
            .iter()
            .map(|context| (context.plan.output_offset - context.plan.activation_size) as usize)
            .collect(),
        activation_bytes: transfers[0].size as usize,
        output_bytes: contexts[contexts.len() - 1].plan.output_size as usize,
        transport,
    })
}

const SERIAL_DIRECT_ROUTE_UNAVAILABLE: &str = "serial direct route unavailable:";

fn tensor_parallel_shader(kernel: &DenseFormatKernel) -> ShaderCode {
    if kernel.execution_path == "native_f16" {
        ShaderCode::Bytes(TP_F16_GEMM_SHADER_SPV)
    } else if kernel.execution_path == NATIVE_FP8_DOT_FEATURE {
        ShaderCode::Bytes(TP_NATIVE_FP8_GEMM_SHADER_SPV)
    } else {
        ShaderCode::Bytes(TP_FORMAT_GEMM_SHADER_SPV)
    }
}

fn tensor_parallel_shard_plans(
    payload_bytes: usize,
    kernel: &DenseFormatKernel,
    participants: usize,
) -> Result<(usize, usize, usize, Vec<TensorParallelShardPlan>), String> {
    if !matches!(kernel.shape, KernelShape::F32Gemm | KernelShape::PackedGemm) {
        return Err(format!(
            "{} does not have a NERVE tensor-parallel execution contract",
            kernel.pattern
        ));
    }
    let full = compute_plan_for_payload(payload_bytes, kernel);
    let m = full.push_constants[0] as usize;
    let n = full.push_constants[1] as usize;
    let k = full.push_constants[2] as usize;
    let shard_alignment = 2;
    let widths = split_extent_aligned(n, participants, shard_alignment)?;
    if widths.contains(&0) {
        return Err(format!(
            "output width {n} cannot be sharded across {participants} devices"
        ));
    }
    let mut column_offset = 0_usize;
    let plans = widths
        .into_iter()
        .map(|local_n| {
            let (parameter_words, scale_words) = parameter_storage_words(local_n, k, kernel);
            let (parameter_word_offset, _) = parameter_storage_words(column_offset, k, kernel);
            let plan = TensorParallelShardPlan {
                parameter_words,
                scale_words,
                parameter_word_offset,
                dispatch: [local_n.div_ceil(32) as u32, m.div_ceil(16) as u32, 1],
                push_constants: vec![
                    m as u32,
                    local_n as u32,
                    k as u32,
                    n as u32,
                    column_offset as u32,
                    kernel.logical_elements_per_storage_element as u32,
                    kernel.format_kind,
                    parameter_words as u32,
                    weight_group_size(kernel) as u32,
                ],
                operations: 2 * m as u64 * local_n as u64 * k as u64,
            };
            column_offset += local_n;
            plan
        })
        .collect();
    Ok((m, n, k, plans))
}

fn create_tensor_parallel_shard_context(
    device: &OpenVulkanComputeDevice,
    kernel: &DenseFormatKernel,
    plan: TensorParallelShardPlan,
    activation: &VulkanBuffer,
    output: &VulkanBuffer,
) -> Result<TensorParallelShardContext, String> {
    if let Some(required_feature) = kernel.required_feature
        && !feature_flags_include(&device.feature_flags, required_feature)
    {
        return Err(format!(
            "selected Vulkan device does not support required feature {required_feature}"
        ));
    }
    if device.timestamp_valid_bits == 0 || device.timestamp_period_ns <= 0.0 {
        return Err("selected Vulkan compute queue does not expose usable timestamps".to_string());
    }
    let weight_words = plan.parameter_words + plan.scale_words;
    let weight_size = (weight_words.max(1) * mem::size_of::<u32>()) as vk::DeviceSize;
    let upload = create_buffer(
        device,
        weight_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    fill_tensor_parallel_weights(&device.device, &upload, kernel, &plan)?;
    let weights = create_buffer(
        device,
        weight_size,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let initialization = create_transfer_context(device)?;
    submit_copy_buffer(
        device,
        &initialization,
        upload.buffer,
        weights.buffer,
        weight_size,
    )?;
    let resources = create_compute_resources_for_buffers(
        device,
        &[
            (activation.buffer, activation.size),
            (weights.buffer, weights.size),
            (output.buffer, output.size),
        ],
        tensor_parallel_shader(kernel),
    )?;
    let command_pool = unsafe {
        device.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(device.compute_queue_family_index),
            None,
        )
    }
    .map_err(|error| format!("could not create TP command pool: {error:?}"))?;
    let command_buffer = unsafe {
        device.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
    }
    .map_err(|error| format!("could not allocate TP command buffer: {error:?}"))?[0];
    let query_pool = unsafe {
        device.device.create_query_pool(
            &vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(2),
            None,
        )
    }
    .map_err(|error| format!("could not create TP timestamp query pool: {error:?}"))?;
    let fence = unsafe {
        device
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
    }
    .map_err(|error| format!("could not create TP completion fence: {error:?}"))?;
    Ok(TensorParallelShardContext {
        device: device.device.clone(),
        resources,
        _upload: upload,
        weights,
        command_pool,
        command_buffer,
        query_pool,
        fence,
        plan,
    })
}

fn fill_tensor_parallel_weights(
    device: &ash::Device,
    buffer: &VulkanBuffer,
    kernel: &DenseFormatKernel,
    plan: &TensorParallelShardPlan,
) -> Result<(), String> {
    let mapped = unsafe {
        device
            .map_memory(buffer.memory, 0, buffer.size, vk::MemoryMapFlags::empty())
            .map_err(|error| format!("could not map TP parameter upload: {error:?}"))?
    };
    let words = mapped.cast::<u32>();
    for index in 0..plan.parameter_words {
        unsafe {
            words
                .add(index)
                .write(parameter_word(kernel, plan.parameter_word_offset + index))
        };
    }
    for index in 0..plan.scale_words {
        unsafe {
            words
                .add(plan.parameter_words + index)
                .write(scale_word(kernel))
        };
    }
    unsafe { device.unmap_memory(buffer.memory) };
    Ok(())
}

fn initialize_tensor_parallel_buffer(
    owner: &OpenVulkanComputeDevice,
    buffer: &VulkanBuffer,
    activation: bool,
) -> Result<(), String> {
    let upload = create_buffer(
        owner,
        buffer.size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let mapped = unsafe {
        owner
            .device
            .map_memory(upload.memory, 0, upload.size, vk::MemoryMapFlags::empty())
            .map_err(|error| format!("could not map TP activation upload: {error:?}"))?
    };
    let words = mapped.cast::<u32>();
    for index in 0..(upload.size as usize / mem::size_of::<u32>()) {
        let value = if activation {
            let first = f32_to_bf16_bits(0.5 + (((index * 2) % 251) as f32) / 512.0);
            let second = f32_to_bf16_bits(0.5 + (((index * 2 + 1) % 251) as f32) / 512.0);
            u32::from(first) | (u32::from(second) << 16)
        } else {
            0x7fc0_7fc0
        };
        unsafe { words.add(index).write(value) };
    }
    unsafe { owner.device.unmap_memory(upload.memory) };
    let transfer = create_transfer_context(owner)?;
    submit_copy_buffer(owner, &transfer, upload.buffer, buffer.buffer, buffer.size)
}

fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn record_tensor_parallel_shard(
    device: &OpenVulkanComputeDevice,
    context: &TensorParallelShardContext,
    activation: &VulkanBuffer,
    output: &VulkanBuffer,
) -> Result<(), String> {
    unsafe {
        device
            .device
            .reset_command_buffer(context.command_buffer, vk::CommandBufferResetFlags::empty())
            .map_err(|error| format!("could not reset TP command buffer: {error:?}"))?;
        device
            .device
            .begin_command_buffer(
                context.command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(|error| format!("could not begin TP command buffer: {error:?}"))?;
        device
            .device
            .cmd_reset_query_pool(context.command_buffer, context.query_pool, 0, 2);
        let barriers = [
            vk::BufferMemoryBarrier::default()
                .src_access_mask(
                    vk::AccessFlags::HOST_WRITE
                        | vk::AccessFlags::TRANSFER_WRITE
                        | vk::AccessFlags::SHADER_WRITE,
                )
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(activation.buffer)
                .offset(0)
                .size(activation.size),
            vk::BufferMemoryBarrier::default()
                .src_access_mask(
                    vk::AccessFlags::HOST_WRITE
                        | vk::AccessFlags::TRANSFER_WRITE
                        | vk::AccessFlags::SHADER_WRITE,
                )
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(output.buffer)
                .offset(0)
                .size(output.size),
        ];
        device.device.cmd_pipeline_barrier(
            context.command_buffer,
            vk::PipelineStageFlags::HOST
                | vk::PipelineStageFlags::TRANSFER
                | vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &barriers,
            &[],
        );
        device.device.cmd_write_timestamp(
            context.command_buffer,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            context.query_pool,
            0,
        );
        device.device.cmd_bind_pipeline(
            context.command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            context.resources.pipeline,
        );
        device.device.cmd_bind_descriptor_sets(
            context.command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            context.resources.pipeline_layout,
            0,
            &[context.resources.descriptor_set],
            &[],
        );
        let push_bytes = std::slice::from_raw_parts(
            context.plan.push_constants.as_ptr().cast::<u8>(),
            context.plan.push_constants.len() * mem::size_of::<u32>(),
        );
        device.device.cmd_push_constants(
            context.command_buffer,
            context.resources.pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            push_bytes,
        );
        device.device.cmd_dispatch(
            context.command_buffer,
            context.plan.dispatch[0],
            context.plan.dispatch[1],
            context.plan.dispatch[2],
        );
        device.device.cmd_write_timestamp(
            context.command_buffer,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            context.query_pool,
            1,
        );
        device
            .device
            .end_command_buffer(context.command_buffer)
            .map_err(|error| format!("could not end TP command buffer: {error:?}"))?;
    }
    Ok(())
}

fn reserve_tensor_parallel_values(
    synchronizations: &mut [TensorParallelSynchronization],
) -> Result<Vec<(u64, u64)>, String> {
    synchronizations
        .iter_mut()
        .map(|synchronization| {
            synchronization.ready.next_value = synchronization
                .ready
                .next_value
                .checked_add(1)
                .ok_or_else(|| "TP ready timeline value overflowed".to_string())?;
            synchronization.done.next_value = synchronization
                .done
                .next_value
                .checked_add(1)
                .ok_or_else(|| "TP done timeline value overflowed".to_string())?;
            Ok((
                synchronization.ready.next_value,
                synchronization.done.next_value,
            ))
        })
        .collect()
}

fn submit_owner_timeline_bridge(
    owner: &OpenVulkanComputeDevice,
    synchronizations: &[TensorParallelSynchronization],
    values: &[(u64, u64)],
    previous_done_values: Option<&[(u64, u64)]>,
) -> Result<(), String> {
    let signal_semaphores = synchronizations
        .iter()
        .map(|sync| sync.ready.source)
        .collect::<Vec<_>>();
    let signal_values = values.iter().map(|(ready, _)| *ready).collect::<Vec<_>>();
    let wait_semaphores = previous_done_values
        .map(|_| {
            synchronizations
                .iter()
                .map(|sync| sync.done.destination)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let wait_values = previous_done_values
        .map(|values| values.iter().map(|(_, done)| *done).collect::<Vec<_>>())
        .unwrap_or_default();
    let wait_stages = vec![vk::PipelineStageFlags::COMPUTE_SHADER; wait_semaphores.len()];
    let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
        .wait_semaphore_values(&wait_values)
        .signal_semaphore_values(&signal_values);
    let submit = [vk::SubmitInfo::default()
        .wait_semaphores(&wait_semaphores)
        .wait_dst_stage_mask(&wait_stages)
        .signal_semaphores(&signal_semaphores)
        .push_next(&mut timeline)];
    unsafe {
        owner
            .device
            .queue_submit(owner.queue, &submit, vk::Fence::null())
    }
    .map_err(|error| format!("could not submit TP owner dependency bridge: {error:?}"))
}

fn submit_tensor_parallel_stage(
    devices: &[&OpenVulkanComputeDevice],
    contexts: &[TensorParallelShardContext],
    synchronizations: &[TensorParallelSynchronization],
    values: &[(u64, u64)],
) -> Result<(), String> {
    for participant in 1..devices.len() {
        let context = &contexts[participant];
        let synchronization = &synchronizations[participant - 1];
        let (ready_value, done_value) = values[participant - 1];
        let command_buffers = [context.command_buffer];
        let wait_semaphores = [synchronization.ready.destination];
        let wait_values = [ready_value];
        let wait_stages = [vk::PipelineStageFlags::COMPUTE_SHADER];
        let signal_semaphores = [synchronization.done.source];
        let signal_values = [done_value];
        let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&wait_values)
            .signal_semaphore_values(&signal_values);
        let submit = [vk::SubmitInfo::default()
            .command_buffers(&command_buffers)
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .signal_semaphores(&signal_semaphores)
            .push_next(&mut timeline)];
        unsafe {
            devices[participant].device.queue_submit(
                devices[participant].queue,
                &submit,
                vk::Fence::null(),
            )
        }
        .map_err(|error| format!("could not submit TP helper {participant} compute: {error:?}"))?;
    }
    let owner_commands = [contexts[0].command_buffer];
    unsafe {
        devices[0].device.queue_submit(
            devices[0].queue,
            &[vk::SubmitInfo::default().command_buffers(&owner_commands)],
            vk::Fence::null(),
        )
    }
    .map_err(|error| format!("could not submit TP owner compute: {error:?}"))
}

fn submit_tensor_parallel_completion(
    owner: &OpenVulkanComputeDevice,
    owner_context: &TensorParallelShardContext,
    synchronizations: &[TensorParallelSynchronization],
    values: &[(u64, u64)],
) -> Result<(), String> {
    let wait_semaphores = synchronizations
        .iter()
        .map(|sync| sync.done.destination)
        .collect::<Vec<_>>();
    let wait_values = values.iter().map(|(_, done)| *done).collect::<Vec<_>>();
    let wait_stages = vec![vk::PipelineStageFlags::COMPUTE_SHADER; wait_semaphores.len()];
    let mut timeline =
        vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&wait_values);
    let submit = [vk::SubmitInfo::default()
        .wait_semaphores(&wait_semaphores)
        .wait_dst_stage_mask(&wait_stages)
        .push_next(&mut timeline)];
    unsafe {
        owner
            .device
            .reset_fences(&[owner_context.fence])
            .map_err(|error| format!("could not reset TP completion fence: {error:?}"))?;
        owner
            .device
            .queue_submit(owner.queue, &submit, owner_context.fence)
            .map_err(|error| format!("could not submit TP completion wait: {error:?}"))?;
        owner
            .device
            .wait_for_fences(&[owner_context.fence], true, MAX_VULKAN_WAIT_NS)
            .map_err(|error| format!("could not wait for TP completion: {error:?}"))?;
    }
    Ok(())
}

fn validate_tensor_parallel_output(
    owner: &OpenVulkanComputeDevice,
    output: &VulkanBuffer,
) -> Result<(), String> {
    let readback = create_buffer(
        owner,
        output.size,
        vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let transfer = create_transfer_context(owner)?;
    submit_copy_buffer(
        owner,
        &transfer,
        output.buffer,
        readback.buffer,
        output.size,
    )?;
    let mut bytes = vec![0_u8; output.size as usize];
    read_buffer_bytes(&owner.device, &readback, &mut bytes)?;
    for (index, chunk) in bytes.chunks_exact(mem::size_of::<u16>()).enumerate() {
        let bits = u16::from_le_bytes(chunk.try_into().unwrap());
        let value = f32::from_bits(u32::from(bits) << 16);
        if !value.is_finite() {
            return Err(format!(
                "distributed output value {index} was not written to a finite BF16 value"
            ));
        }
    }
    black_box(checksum_bytes(&bytes));
    Ok(())
}

fn run_shared_tensor_parallel(
    devices: &[&OpenVulkanComputeDevice],
    payload_bytes: usize,
    samples: usize,
    kernel: &DenseFormatKernel,
    stages: usize,
) -> Result<TensorParallelRunResult, String> {
    if devices.len() < 2 || stages == 0 {
        return Err("TP execution requires multiple devices and at least one stage".to_string());
    }
    if devices
        .iter()
        .any(|device| !device.external_timeline_semaphore)
    {
        return Err(
            "NERVE TP requires opaque-FD timeline semaphores on every participant".to_string(),
        );
    }
    let device_local_available = devices.iter().all(|device| device.external_memory_fd);
    let shared_host_available = devices.iter().all(|device| device.external_memory_host);
    if device_local_available {
        match run_shared_tensor_parallel_with_route(
            devices,
            payload_bytes,
            samples,
            kernel,
            stages,
            SharedTransferRoute::ExternalDeviceLocal,
        ) {
            Ok(result) => return Ok(result),
            Err(device_local_error) if shared_host_available => {
                return run_shared_tensor_parallel_with_route(
                    devices,
                    payload_bytes,
                    samples,
                    kernel,
                    stages,
                    SharedTransferRoute::SharedHost,
                )
                .map_err(|host_error| {
                    format!(
                        "device-local TP route failed ({device_local_error}); shared-host TP route failed ({host_error})"
                    )
                });
            }
            Err(error) => return Err(error),
        }
    }
    if shared_host_available {
        return run_shared_tensor_parallel_with_route(
            devices,
            payload_bytes,
            samples,
            kernel,
            stages,
            SharedTransferRoute::SharedHost,
        );
    }
    Err("TP execution has no common shared-buffer route".to_string())
}

fn run_shared_tensor_parallel_with_route(
    devices: &[&OpenVulkanComputeDevice],
    payload_bytes: usize,
    samples: usize,
    kernel: &DenseFormatKernel,
    stages: usize,
    route: SharedTransferRoute,
) -> Result<TensorParallelRunResult, String> {
    let stage_payload = payload_bytes / stages;
    if stage_payload == 0 {
        return Err("payload is too small for the requested TP chain".to_string());
    }
    let (m, n, k, shard_plans) = tensor_parallel_shard_plans(stage_payload, kernel, devices.len())?;
    let activation_size = ((m * k).div_ceil(2) * mem::size_of::<u32>()) as vk::DeviceSize;
    let output_size = ((m * n).div_ceil(2) * mem::size_of::<u32>()) as vk::DeviceSize;
    if stages > 1 && activation_size != output_size {
        return Err("TP component chain requires equal activation and output shapes".to_string());
    }
    let shared_buffers = (0..=stages)
        .map(|stage| {
            create_shared_multi_buffer(
                devices,
                if stage == 0 {
                    activation_size
                } else {
                    output_size
                },
                route,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    initialize_tensor_parallel_buffer(devices[0], &shared_buffers[0].buffers[0], true)?;
    for shared in &shared_buffers[1..] {
        initialize_tensor_parallel_buffer(devices[0], &shared.buffers[0], false)?;
    }
    let mut stage_contexts = Vec::with_capacity(stages);
    for stage in 0..stages {
        let contexts = devices
            .iter()
            .enumerate()
            .map(|(participant, device)| {
                create_tensor_parallel_shard_context(
                    device,
                    kernel,
                    shard_plans[participant].clone(),
                    &shared_buffers[stage].buffers[participant],
                    &shared_buffers[stage + 1].buffers[participant],
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        stage_contexts.push(contexts);
    }
    let mut synchronizations = devices[1..]
        .iter()
        .map(|helper| {
            Ok(TensorParallelSynchronization {
                ready: create_external_timeline_semaphore(devices[0], helper)?,
                done: create_external_timeline_semaphore(helper, devices[0])?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut measured_samples = Vec::with_capacity(samples);
    for iteration in 0..=samples {
        for stage in 0..stages {
            for participant in 0..devices.len() {
                record_tensor_parallel_shard(
                    devices[participant],
                    &stage_contexts[stage][participant],
                    &shared_buffers[stage].buffers[participant],
                    &shared_buffers[stage + 1].buffers[participant],
                )?;
            }
        }
        let started = Instant::now();
        let mut previous_values: Option<Vec<(u64, u64)>> = None;
        let mut final_values = Vec::new();
        for contexts in &stage_contexts {
            let values = reserve_tensor_parallel_values(&mut synchronizations)?;
            submit_owner_timeline_bridge(
                devices[0],
                &synchronizations,
                &values,
                previous_values.as_deref(),
            )?;
            submit_tensor_parallel_stage(devices, contexts, &synchronizations, &values)?;
            previous_values = Some(values.clone());
            final_values = values;
        }
        submit_tensor_parallel_completion(
            devices[0],
            &stage_contexts.last().unwrap()[0],
            &synchronizations,
            &final_values,
        )?;
        let wall_duration_ns = started.elapsed().as_nanos();
        if iteration == 0 {
            continue;
        }
        let mut gpu_duration_ns = 0_u128;
        for contexts in &stage_contexts {
            let stage_duration = contexts
                .iter()
                .enumerate()
                .map(|(participant, context)| {
                    read_timestamp_duration_ns(devices[participant], context.query_pool)
                })
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .max()
                .unwrap_or_default();
            gpu_duration_ns += stage_duration;
        }
        measured_samples.push(Sample {
            sample_index: iteration - 1,
            duration_ns: wall_duration_ns.max(gpu_duration_ns),
            iterations: 1,
            bytes_read: stage_contexts
                .iter()
                .flat_map(|contexts| contexts.iter())
                .map(|context| activation_size as u64 + context.weights.size as u64)
                .sum(),
            bytes_written: (output_size as u64) * stages as u64,
            operations: stage_contexts
                .iter()
                .flat_map(|contexts| contexts.iter())
                .map(|context| context.plan.operations)
                .sum(),
        });
    }
    validate_tensor_parallel_output(devices[0], &shared_buffers.last().unwrap().buffers[0])?;
    let parameter_bytes_per_participant = (0..devices.len())
        .map(|participant| {
            stage_contexts
                .iter()
                .map(|contexts| contexts[participant].weights.size as usize)
                .sum()
        })
        .collect();
    let transport = route.name();
    Ok(TensorParallelRunResult {
        samples: measured_samples,
        parameter_bytes_per_participant,
        activation_bytes: activation_size as usize,
        output_bytes: output_size as usize,
        transport,
    })
}

fn serial_chain_plan(
    payload_bytes: usize,
    kernel: &DenseFormatKernel,
) -> Result<(usize, usize, usize, TensorParallelShardPlan), String> {
    let (m, n, k, mut plans) = tensor_parallel_shard_plans(payload_bytes, kernel, 1)?;
    Ok((m, n, k, plans.pop().unwrap()))
}

fn serial_chain_sample(
    duration_ns: u128,
    sample_index: usize,
    contexts: &[TensorParallelShardContext],
    activation_size: vk::DeviceSize,
    output_size: vk::DeviceSize,
) -> Sample {
    Sample {
        sample_index,
        duration_ns,
        iterations: 1,
        bytes_read: contexts
            .iter()
            .map(|context| activation_size as u64 + context.weights.size as u64)
            .sum(),
        bytes_written: output_size as u64 * contexts.len() as u64,
        operations: contexts.iter().map(|context| context.plan.operations).sum(),
    }
}

fn run_direct_local_serial_chain(
    device: &OpenVulkanComputeDevice,
    payload_bytes: usize,
    samples: usize,
    kernel: &DenseFormatKernel,
    stages: usize,
) -> Result<SerialChainRunResult, String> {
    let stage_payload = payload_bytes / stages;
    let (m, n, k, plan) = serial_chain_plan(stage_payload, kernel)?;
    let activation_size = ((m * k).div_ceil(2) * mem::size_of::<u32>()) as vk::DeviceSize;
    let output_size = ((m * n).div_ceil(2) * mem::size_of::<u32>()) as vk::DeviceSize;
    if activation_size != output_size {
        return Err("component chain requires equal BF16 activation and output shapes".to_string());
    }
    let slots = (0..=stages)
        .map(|_| {
            create_buffer(
                device,
                activation_size,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    initialize_tensor_parallel_buffer(device, &slots[0], true)?;
    for slot in &slots[1..] {
        initialize_tensor_parallel_buffer(device, slot, false)?;
    }
    let contexts = (0..stages)
        .map(|stage| {
            create_tensor_parallel_shard_context(
                device,
                kernel,
                plan.clone(),
                &slots[stage],
                &slots[stage + 1],
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut measured_samples = Vec::with_capacity(samples);
    for iteration in 0..=samples {
        for stage in 0..stages {
            record_tensor_parallel_shard(
                device,
                &contexts[stage],
                &slots[stage],
                &slots[stage + 1],
            )?;
        }
        let command_buffers = contexts
            .iter()
            .map(|context| context.command_buffer)
            .collect::<Vec<_>>();
        let completion = &contexts[stages - 1];
        let started = Instant::now();
        unsafe {
            device
                .device
                .reset_fences(&[completion.fence])
                .map_err(|error| format!("could not reset local chain fence: {error:?}"))?;
            device
                .device
                .queue_submit(
                    device.queue,
                    &[vk::SubmitInfo::default().command_buffers(&command_buffers)],
                    completion.fence,
                )
                .map_err(|error| format!("could not submit local component chain: {error:?}"))?;
            device
                .device
                .wait_for_fences(&[completion.fence], true, MAX_VULKAN_WAIT_NS)
                .map_err(|error| format!("could not wait for local component chain: {error:?}"))?;
        }
        let wall_duration_ns = started.elapsed().as_nanos();
        if iteration == 0 {
            continue;
        }
        let gpu_duration_ns = contexts
            .iter()
            .map(|context| read_timestamp_duration_ns(device, context.query_pool))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum();
        measured_samples.push(serial_chain_sample(
            wall_duration_ns.max(gpu_duration_ns),
            iteration - 1,
            &contexts,
            activation_size,
            output_size,
        ));
    }
    validate_tensor_parallel_output(device, &slots[stages])?;
    Ok(SerialChainRunResult {
        samples: measured_samples,
        parameter_bytes_per_stage: contexts
            .iter()
            .map(|context| context.weights.size as usize)
            .collect(),
        activation_bytes: activation_size as usize,
        output_bytes: output_size as usize,
        transport: "device_local",
    })
}

fn run_direct_external_serial_chain(
    devices: &[&OpenVulkanComputeDevice],
    payload_bytes: usize,
    samples: usize,
    kernel: &DenseFormatKernel,
) -> Result<SerialChainRunResult, String> {
    if devices.len() < 2 {
        return Err("external serial chain requires at least two devices".to_string());
    }
    if devices
        .iter()
        .any(|device| !device.external_timeline_semaphore)
    {
        return Err(format!(
            "{SERIAL_DIRECT_ROUTE_UNAVAILABLE} opaque-FD timeline semaphores are unavailable"
        ));
    }
    let stage_payload = payload_bytes / devices.len();
    let (m, n, k, plan) = serial_chain_plan(stage_payload, kernel)?;
    let activation_size = ((m * k).div_ceil(2) * mem::size_of::<u32>()) as vk::DeviceSize;
    let output_size = ((m * n).div_ceil(2) * mem::size_of::<u32>()) as vk::DeviceSize;
    if activation_size != output_size {
        return Err("component chain requires equal BF16 activation and output shapes".to_string());
    }
    let input = create_buffer(
        devices[0],
        activation_size,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let output = create_buffer(
        devices[devices.len() - 1],
        output_size,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    initialize_tensor_parallel_buffer(devices[0], &input, true)?;
    initialize_tensor_parallel_buffer(devices[devices.len() - 1], &output, false)?;
    let boundaries = devices
        .windows(2)
        .map(|pair| {
            let shared = create_shared_multi_buffer(
                &[pair[0], pair[1]],
                output_size,
                SharedTransferRoute::ExternalDeviceLocal,
            )
            .map_err(|error| format!("{SERIAL_DIRECT_ROUTE_UNAVAILABLE} {error}"))?;
            Ok(shared)
        })
        .collect::<Result<Vec<_>, String>>()?;
    for (stage, boundary) in boundaries.iter().enumerate() {
        initialize_tensor_parallel_buffer(devices[stage], &boundary.buffers[0], false)?;
    }
    let mut synchronizations = devices
        .windows(2)
        .map(|pair| create_external_timeline_semaphore(pair[0], pair[1]))
        .collect::<Result<Vec<_>, _>>()?;
    let contexts = (0..devices.len())
        .map(|stage| {
            let activation = if stage == 0 {
                &input
            } else {
                &boundaries[stage - 1].buffers[1]
            };
            let stage_output = if stage + 1 == devices.len() {
                &output
            } else {
                &boundaries[stage].buffers[0]
            };
            create_tensor_parallel_shard_context(
                devices[stage],
                kernel,
                plan.clone(),
                activation,
                stage_output,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut measured_samples = Vec::with_capacity(samples);
    for iteration in 0..=samples {
        for stage in 0..devices.len() {
            let activation = if stage == 0 {
                &input
            } else {
                &boundaries[stage - 1].buffers[1]
            };
            let stage_output = if stage + 1 == devices.len() {
                &output
            } else {
                &boundaries[stage].buffers[0]
            };
            record_tensor_parallel_shard(
                devices[stage],
                &contexts[stage],
                activation,
                stage_output,
            )?;
        }
        let values = synchronizations
            .iter_mut()
            .map(|synchronization| {
                synchronization.next_value = synchronization
                    .next_value
                    .checked_add(1)
                    .ok_or_else(|| "serial timeline value overflowed".to_string())?;
                Ok(synchronization.next_value)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let completion = &contexts[contexts.len() - 1];
        unsafe {
            devices[devices.len() - 1]
                .device
                .reset_fences(&[completion.fence])
                .map_err(|error| format!("could not reset serial completion fence: {error:?}"))?;
        }
        let started = Instant::now();
        for stage in 0..devices.len() {
            let command_buffers = [contexts[stage].command_buffer];
            let wait_semaphores = if stage == 0 {
                Vec::new()
            } else {
                vec![synchronizations[stage - 1].destination]
            };
            let wait_values = if stage == 0 {
                Vec::new()
            } else {
                vec![values[stage - 1]]
            };
            let wait_stages = vec![vk::PipelineStageFlags::COMPUTE_SHADER; wait_values.len()];
            let signal_semaphores = if stage + 1 == devices.len() {
                Vec::new()
            } else {
                vec![synchronizations[stage].source]
            };
            let signal_values = if stage + 1 == devices.len() {
                Vec::new()
            } else {
                vec![values[stage]]
            };
            let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
                .wait_semaphore_values(&wait_values)
                .signal_semaphore_values(&signal_values);
            let submit = [vk::SubmitInfo::default()
                .command_buffers(&command_buffers)
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .signal_semaphores(&signal_semaphores)
                .push_next(&mut timeline)];
            let fence = if stage + 1 == devices.len() {
                completion.fence
            } else {
                vk::Fence::null()
            };
            unsafe {
                devices[stage]
                    .device
                    .queue_submit(devices[stage].queue, &submit, fence)
            }
            .map_err(|error| format!("could not submit serial stage {stage}: {error:?}"))?;
        }
        unsafe {
            devices[devices.len() - 1]
                .device
                .wait_for_fences(&[completion.fence], true, MAX_VULKAN_WAIT_NS)
                .map_err(|error| format!("could not wait for serial chain: {error:?}"))?;
        }
        let wall_duration_ns = started.elapsed().as_nanos();
        if iteration == 0 {
            continue;
        }
        let gpu_duration_ns = contexts
            .iter()
            .enumerate()
            .map(|(stage, context)| read_timestamp_duration_ns(devices[stage], context.query_pool))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum();
        measured_samples.push(serial_chain_sample(
            wall_duration_ns.max(gpu_duration_ns),
            iteration - 1,
            &contexts,
            activation_size,
            output_size,
        ));
    }
    validate_tensor_parallel_output(devices[devices.len() - 1], &output)?;
    Ok(SerialChainRunResult {
        samples: measured_samples,
        parameter_bytes_per_stage: contexts
            .iter()
            .map(|context| context.weights.size as usize)
            .collect(),
        activation_bytes: activation_size as usize,
        output_bytes: output_size as usize,
        transport: "external_device_local",
    })
}

#[allow(clippy::too_many_arguments)]
fn run_vulkan_dense_serial_group(
    devices: &[&OpenVulkanComputeDevice],
    target_ids: &[String],
    payload_bytes: usize,
    samples: usize,
    workload_class: &str,
    kernel: DenseFormatKernel,
) -> Result<GroupMeasurement, String> {
    let result = match run_direct_external_serial_chain(devices, payload_bytes, samples, &kernel) {
        Ok(result) => result,
        Err(direct_error) => run_staged_serial_chain(devices, payload_bytes, samples, &kernel)
            .map_err(|staged_error| {
                format!(
                    "direct serial route failed ({direct_error}); staged serial route failed ({staged_error})"
                )
            })?,
    };
    let summary = summarize_samples(&result.samples);
    Ok(GroupMeasurement {
        workload_id: format_workload_id(
            &format!("synthetic_serialized_forced_split_{}", devices.len()),
            workload_class,
            &kernel.format,
        ),
        comparison_group: "forced_split_tp_vs_serialized".to_string(),
        workload_class: workload_class.to_string(),
        placement_strategy: "multi_target_serial".to_string(),
        target_ids: target_ids.to_vec(),
        pattern: format!(
            "{}:transport={}",
            kernel_identity(&kernel),
            result.transport
        ),
        operation_family: workload_class.to_string(),
        regime: component_chain_regime(devices.len()).to_string(),
        format: kernel.format,
        status: "completed".to_string(),
        reason: None,
        participant_count: devices.len(),
        payload_bytes,
        payload_bytes_per_participant: result.parameter_bytes_per_stage,
        activation_bytes: result.activation_bytes,
        output_bytes: result.output_bytes,
        samples: result.samples,
        summary,
    })
}

fn run_vulkan_dense_tensor_parallel_pair(
    left_device: &OpenVulkanComputeDevice,
    right_device: &OpenVulkanComputeDevice,
    left_id: &str,
    right_id: &str,
    payload_bytes: usize,
    samples: usize,
    workload_class: &str,
    kernel: DenseFormatKernel,
) -> Result<PairMeasurement, String> {
    let result = run_shared_tensor_parallel(
        &[left_device, right_device],
        payload_bytes,
        samples,
        &kernel,
        1,
    )?;
    let summary = summarize_samples(&result.samples);
    Ok(PairMeasurement {
        workload_id: format_workload_id(
            TENSOR_PARALLEL_PAIR_PATTERN,
            workload_class,
            &kernel.format,
        ),
        comparison_group: "small_payload_placement_comparison".to_string(),
        workload_class: workload_class.to_string(),
        placement_strategy: TWO_TARGET_TENSOR_PARALLEL_STRATEGY.to_string(),
        source_target_id: left_id.to_string(),
        destination_target_id: right_id.to_string(),
        pattern: format!(
            "{}:shared_resident:transport={}",
            kernel_identity(&kernel),
            result.transport
        ),
        operation_family: workload_class.to_string(),
        regime: SINGLE_COMPONENT_REGIME.to_string(),
        format: kernel.format,
        status: "completed".to_string(),
        reason: None,
        payload_bytes,
        source_payload_bytes: result.parameter_bytes_per_participant[0],
        destination_payload_bytes: result.parameter_bytes_per_participant[1],
        activation_bytes: result.activation_bytes,
        output_bytes: result.output_bytes,
        samples: result.samples,
        summary,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_vulkan_dense_tensor_parallel_pair_chain(
    left_device: &OpenVulkanComputeDevice,
    right_device: &OpenVulkanComputeDevice,
    left_id: &str,
    right_id: &str,
    payload_bytes: usize,
    samples: usize,
    workload_class: &str,
    kernel: DenseFormatKernel,
) -> Result<PairMeasurement, String> {
    let result = run_shared_tensor_parallel(
        &[left_device, right_device],
        payload_bytes,
        samples,
        &kernel,
        2,
    )?;
    let summary = summarize_samples(&result.samples);
    Ok(PairMeasurement {
        workload_id: format_workload_id(
            TENSOR_PARALLEL_CHAIN_PATTERN,
            workload_class,
            &kernel.format,
        ),
        comparison_group: "forced_split_tp_vs_serialized".to_string(),
        workload_class: workload_class.to_string(),
        placement_strategy: TWO_TARGET_TENSOR_PARALLEL_STRATEGY.to_string(),
        source_target_id: left_id.to_string(),
        destination_target_id: right_id.to_string(),
        pattern: format!(
            "{}:shared_resident:transport={}",
            kernel_identity(&kernel),
            result.transport
        ),
        operation_family: workload_class.to_string(),
        regime: TWO_COMPONENT_CHAIN_REGIME.to_string(),
        format: kernel.format,
        status: "completed".to_string(),
        reason: None,
        payload_bytes,
        source_payload_bytes: result.parameter_bytes_per_participant[0],
        destination_payload_bytes: result.parameter_bytes_per_participant[1],
        activation_bytes: result.activation_bytes,
        output_bytes: result.output_bytes,
        samples: result.samples,
        summary,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_vulkan_dense_tensor_parallel_group(
    devices: &[&OpenVulkanComputeDevice],
    target_ids: &[String],
    payload_bytes: usize,
    samples: usize,
    workload_class: &str,
    kernel: DenseFormatKernel,
) -> Result<GroupMeasurement, String> {
    let result = run_shared_tensor_parallel(devices, payload_bytes, samples, &kernel, 1)?;
    let summary = summarize_samples(&result.samples);
    Ok(GroupMeasurement {
        workload_id: tensor_parallel_group_workload_id(
            devices.len(),
            workload_class,
            &kernel.format,
        ),
        comparison_group: "small_payload_placement_comparison".to_string(),
        workload_class: workload_class.to_string(),
        placement_strategy: MULTI_TARGET_TENSOR_PARALLEL_STRATEGY.to_string(),
        target_ids: target_ids.to_vec(),
        pattern: format!(
            "{}:shared_resident:transport={}",
            kernel_identity(&kernel),
            result.transport
        ),
        operation_family: workload_class.to_string(),
        regime: SINGLE_COMPONENT_REGIME.to_string(),
        format: kernel.format,
        status: "completed".to_string(),
        reason: None,
        participant_count: devices.len(),
        payload_bytes,
        payload_bytes_per_participant: result.parameter_bytes_per_participant,
        activation_bytes: result.activation_bytes,
        output_bytes: result.output_bytes,
        samples: result.samples,
        summary,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_vulkan_dense_tensor_parallel_group_chain(
    devices: &[&OpenVulkanComputeDevice],
    target_ids: &[String],
    payload_bytes: usize,
    samples: usize,
    workload_class: &str,
    kernel: DenseFormatKernel,
) -> Result<GroupMeasurement, String> {
    let result =
        run_shared_tensor_parallel(devices, payload_bytes, samples, &kernel, devices.len())?;
    let summary = summarize_samples(&result.samples);
    Ok(GroupMeasurement {
        workload_id: format_workload_id(
            TENSOR_PARALLEL_CHAIN_PATTERN,
            workload_class,
            &kernel.format,
        ),
        comparison_group: "forced_split_tp_vs_serialized".to_string(),
        workload_class: workload_class.to_string(),
        placement_strategy: MULTI_TARGET_TENSOR_PARALLEL_STRATEGY.to_string(),
        target_ids: target_ids.to_vec(),
        pattern: format!(
            "{}:shared_resident:transport={}",
            kernel_identity(&kernel),
            result.transport
        ),
        operation_family: workload_class.to_string(),
        regime: component_chain_regime(devices.len()).to_string(),
        format: kernel.format,
        status: "completed".to_string(),
        reason: None,
        participant_count: devices.len(),
        payload_bytes,
        payload_bytes_per_participant: result.parameter_bytes_per_participant,
        activation_bytes: result.activation_bytes,
        output_bytes: result.output_bytes,
        samples: result.samples,
        summary,
    })
}

fn tensor_parallel_group_status_measurement(
    target_ids: &[String],
    status: &str,
    payload_bytes: usize,
    workload_class: &str,
    format: &str,
    reason: &str,
) -> GroupMeasurement {
    GroupMeasurement {
        workload_id: tensor_parallel_group_workload_id(target_ids.len(), workload_class, format),
        comparison_group: "small_payload_placement_comparison".to_string(),
        workload_class: workload_class.to_string(),
        placement_strategy: MULTI_TARGET_TENSOR_PARALLEL_STRATEGY.to_string(),
        target_ids: target_ids.to_vec(),
        pattern: tensor_parallel_group_pattern(target_ids.len()),
        operation_family: workload_class.to_string(),
        regime: SINGLE_COMPONENT_REGIME.to_string(),
        format: format.to_string(),
        status: status.to_string(),
        reason: Some(reason.to_string()),
        participant_count: target_ids.len(),
        payload_bytes,
        payload_bytes_per_participant: split_payload_bytes(payload_bytes, target_ids.len()),
        activation_bytes: activation_bytes_for_payload(payload_bytes),
        output_bytes: output_bytes_for_payload(payload_bytes),
        samples: Vec::new(),
        summary: None,
    }
}

#[allow(clippy::too_many_arguments)]
fn forced_split_group_status_measurement(
    target_ids: &[String],
    placement_strategy: &str,
    status: &str,
    payload_bytes: usize,
    workload_class: &str,
    format: &str,
    reason: &str,
) -> GroupMeasurement {
    let pattern = if placement_strategy == "multi_target_serial" {
        format!("synthetic_serialized_forced_split_{}", target_ids.len())
    } else {
        format!(
            "synthetic_tensor_parallel_forced_split_{}",
            target_ids.len()
        )
    };
    GroupMeasurement {
        workload_id: format_workload_id(&pattern, workload_class, format),
        comparison_group: "forced_split_tp_vs_serialized".to_string(),
        workload_class: workload_class.to_string(),
        placement_strategy: placement_strategy.to_string(),
        target_ids: target_ids.to_vec(),
        pattern,
        operation_family: workload_class.to_string(),
        regime: component_chain_regime(target_ids.len()).to_string(),
        format: format.to_string(),
        status: status.to_string(),
        reason: Some(reason.to_string()),
        participant_count: target_ids.len(),
        payload_bytes,
        payload_bytes_per_participant: split_payload_bytes(payload_bytes, target_ids.len()),
        activation_bytes: activation_bytes_for_payload(payload_bytes),
        output_bytes: output_bytes_for_payload(payload_bytes),
        samples: Vec::new(),
        summary: None,
    }
}

fn split_payload_bytes(payload_bytes: usize, parts: usize) -> Vec<usize> {
    split_extent(payload_bytes, parts)
}

struct ComputeResources {
    device: ash::Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_set: vk::DescriptorSet,
    pipeline_layout: vk::PipelineLayout,
    descriptor_pool: vk::DescriptorPool,
    pipeline: vk::Pipeline,
}

impl Drop for ComputeResources {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

fn create_compute_resources(
    compute_device: &OpenVulkanComputeDevice,
    storage_buffer: vk::Buffer,
    buffer_size: vk::DeviceSize,
    shader: ShaderCode,
) -> Result<ComputeResources, String> {
    create_compute_resources_for_buffers(compute_device, &[(storage_buffer, buffer_size)], shader)
}

fn create_compute_resources_for_buffers(
    compute_device: &OpenVulkanComputeDevice,
    buffers: &[(vk::Buffer, vk::DeviceSize)],
    shader: ShaderCode,
) -> Result<ComputeResources, String> {
    let device = &compute_device.device;
    let bindings = buffers
        .iter()
        .enumerate()
        .map(|(binding, _)| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect::<Vec<_>>();
    let descriptor_set_layout = unsafe {
        device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )
    }
    .map_err(|error| format!("could not create Vulkan descriptor set layout: {error:?}"))?;
    let push_ranges = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size((mem::size_of::<u32>() * 10) as u32)];
    let set_layouts = [descriptor_set_layout];
    let pipeline_layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&set_layouts)
                .push_constant_ranges(&push_ranges),
            None,
        )
    }
    .map_err(|error| format!("could not create Vulkan pipeline layout: {error:?}"))?;
    let shader_words = shader_words(shader)?;
    let shader_module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&shader_words),
            None,
        )
    }
    .map_err(|error| format!("could not create Vulkan shader module: {error:?}"))?;
    let entry_name = CString::new("main").expect("static string has no nul");
    let shader_stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(&entry_name);
    let pipeline = unsafe {
        device.create_compute_pipelines(
            vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo::default()
                .stage(shader_stage)
                .layout(pipeline_layout)],
            None,
        )
    }
    .map_err(|(_, error)| format!("could not create Vulkan compute pipeline: {error:?}"))?[0];
    unsafe { device.destroy_shader_module(shader_module, None) };
    let pool_sizes = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(buffers.len() as u32)];
    let descriptor_pool = unsafe {
        device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&pool_sizes),
            None,
        )
    }
    .map_err(|error| format!("could not create Vulkan descriptor pool: {error:?}"))?;
    let descriptor_set = unsafe {
        device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&set_layouts),
        )
    }
    .map_err(|error| format!("could not allocate Vulkan descriptor set: {error:?}"))?[0];
    let buffer_infos = buffers
        .iter()
        .map(|(buffer, size)| {
            vec![
                vk::DescriptorBufferInfo::default()
                    .buffer(*buffer)
                    .offset(0)
                    .range(*size),
            ]
        })
        .collect::<Vec<_>>();
    let descriptor_writes = buffer_infos
        .iter()
        .enumerate()
        .map(|(binding, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(binding as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(info)
        })
        .collect::<Vec<_>>();
    unsafe {
        device.update_descriptor_sets(&descriptor_writes, &[]);
    }
    Ok(ComputeResources {
        device: device.clone(),
        descriptor_set_layout,
        descriptor_set,
        pipeline_layout,
        descriptor_pool,
        pipeline,
    })
}

fn shader_words(shader: ShaderCode) -> Result<Vec<u32>, String> {
    match shader {
        ShaderCode::Bytes(bytes) => {
            if bytes.len() % mem::size_of::<u32>() != 0 {
                return Err("SPIR-V bytecode length is not u32-aligned".to_string());
            }
            Ok(bytes
                .chunks_exact(mem::size_of::<u32>())
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect())
        }
    }
}

fn record_compute_dispatch(
    compute_device: &OpenVulkanComputeDevice,
    resources: &ComputeResources,
    command_buffer: vk::CommandBuffer,
    query_pool: vk::QueryPool,
    storage_buffer: vk::Buffer,
    readback_buffer: vk::Buffer,
    plan: &ComputePlan,
    readback_output: bool,
) -> Result<(), String> {
    let device = &compute_device.device;
    unsafe {
        device
            .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
            .map_err(|error| format!("could not reset Vulkan command buffer: {error:?}"))?;
        device
            .begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(|error| format!("could not begin Vulkan command buffer: {error:?}"))?;
        device.cmd_reset_query_pool(command_buffer, query_pool, 0, 2);
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                .buffer(storage_buffer)
                .offset(0)
                .size(plan.buffer_size)],
            &[],
        );
        device.cmd_write_timestamp(
            command_buffer,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            query_pool,
            0,
        );
        device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            resources.pipeline,
        );
        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            resources.pipeline_layout,
            0,
            &[resources.descriptor_set],
            &[],
        );
        let push_bytes = std::slice::from_raw_parts(
            plan.push_constants.as_ptr().cast::<u8>(),
            plan.push_constants.len() * mem::size_of::<u32>(),
        );
        device.cmd_push_constants(
            command_buffer,
            resources.pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            push_bytes,
        );
        device.cmd_dispatch(
            command_buffer,
            plan.dispatch[0],
            plan.dispatch[1],
            plan.dispatch[2],
        );
        device.cmd_write_timestamp(
            command_buffer,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            query_pool,
            1,
        );
        if readback_output {
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .buffer(storage_buffer)
                    .offset(plan.output_offset)
                    .size(plan.output_size)],
                &[],
            );
            device.cmd_copy_buffer(
                command_buffer,
                storage_buffer,
                readback_buffer,
                &[vk::BufferCopy::default()
                    .src_offset(plan.output_offset)
                    .size(plan.output_size)],
            );
        }
        device
            .end_command_buffer(command_buffer)
            .map_err(|error| format!("could not end Vulkan command buffer: {error:?}"))?;
    }
    Ok(())
}

fn read_timestamp_duration_ns(
    compute_device: &OpenVulkanComputeDevice,
    query_pool: vk::QueryPool,
) -> Result<u128, String> {
    let mut timestamps = [0_u64; 2];
    unsafe {
        compute_device
            .device
            .get_query_pool_results(
                query_pool,
                0,
                &mut timestamps,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
            .map_err(|error| format!("could not read Vulkan timestamp queries: {error:?}"))?;
    }
    let mask = if compute_device.timestamp_valid_bits >= 64 {
        u64::MAX
    } else {
        (1_u64 << compute_device.timestamp_valid_bits) - 1
    };
    let ticks = timestamps[1].wrapping_sub(timestamps[0]) & mask;
    Ok((ticks as f64 * compute_device.timestamp_period_ns as f64).round() as u128)
}

fn create_buffer(
    compute_device: &OpenVulkanComputeDevice,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<VulkanBuffer, String> {
    let device = &compute_device.device;
    let buffer = unsafe {
        device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )
    }
    .map_err(|error| format!("could not create Vulkan buffer: {error:?}"))?;
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let memory_type_index = memory_type_index(
        &compute_device.memory_properties,
        requirements.memory_type_bits,
        properties,
    )
    .ok_or_else(|| format!("could not find Vulkan memory type with flags {properties:?}"))?;
    let memory = unsafe {
        device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type_index),
            None,
        )
    }
    .map_err(|error| format!("could not allocate Vulkan buffer memory: {error:?}"))?;
    unsafe {
        device
            .bind_buffer_memory(buffer, memory, 0)
            .map_err(|error| format!("could not bind Vulkan buffer memory: {error:?}"))?;
    }
    Ok(VulkanBuffer {
        device: device.clone(),
        buffer,
        memory,
        size,
        _shared_host: None,
    })
}

fn create_shared_transfer_buffers(
    source: &OpenVulkanComputeDevice,
    destination: &OpenVulkanComputeDevice,
    size: vk::DeviceSize,
) -> Result<SharedTransferBuffers, String> {
    match create_external_device_local_buffers(source, destination, size) {
        Ok((source_buffer, destination_buffer)) => {
            match create_external_timeline_semaphore(source, destination) {
                Ok(timeline) => Ok(SharedTransferBuffers {
                    route: SharedTransferRoute::ExternalDeviceLocal,
                    source: source_buffer,
                    destination: destination_buffer,
                    timeline: Some(timeline),
                }),
                Err(timeline_error) => create_shared_host_buffers(source, destination, size)
                    .map(|(source_buffer, destination_buffer)| SharedTransferBuffers {
                        route: SharedTransferRoute::SharedHost,
                        source: source_buffer,
                        destination: destination_buffer,
                        timeline: create_external_timeline_semaphore(source, destination).ok(),
                    })
                    .map_err(|host_error| {
                        format!(
                            "device-local memory exists but external timeline synchronization is unavailable ({timeline_error}); shared-host route unavailable ({host_error})"
                        )
                    }),
            }
        }
        Err(device_local_error) => {
            create_shared_host_buffers(source, destination, size)
                .map(|(source_buffer, destination_buffer)| SharedTransferBuffers {
                    route: SharedTransferRoute::SharedHost,
                    source: source_buffer,
                    destination: destination_buffer,
                    timeline: create_external_timeline_semaphore(source, destination).ok(),
                })
                .map_err(|host_error| {
                    format!(
                        "external device-local route unavailable ({device_local_error}); shared-host route unavailable ({host_error})"
                    )
                })
        }
    }
}

fn create_shared_multi_buffer(
    devices: &[&OpenVulkanComputeDevice],
    size: vk::DeviceSize,
    route: SharedTransferRoute,
) -> Result<SharedMultiBuffer, String> {
    if devices.len() < 2 {
        return Err("shared multi-device buffer requires at least two devices".to_string());
    }
    let buffers = match route {
        SharedTransferRoute::ExternalDeviceLocal => {
            create_external_device_local_multi_buffers(devices, size)
        }
        SharedTransferRoute::SharedHost => create_shared_host_multi_buffers(devices, size),
    }?;
    Ok(SharedMultiBuffer { buffers })
}

fn create_external_timeline_semaphore(
    source: &OpenVulkanComputeDevice,
    destination: &OpenVulkanComputeDevice,
) -> Result<ExternalTimelineSemaphore, String> {
    if !source.external_timeline_semaphore || !destination.external_timeline_semaphore {
        return Err("opaque-FD timeline semaphores are not available on both devices".to_string());
    }
    let handle_type = vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD;
    let mut timeline_info = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
    let mut export_info = vk::ExportSemaphoreCreateInfo::default().handle_types(handle_type);
    let source_semaphore = unsafe {
        source.device.create_semaphore(
            &vk::SemaphoreCreateInfo::default()
                .push_next(&mut timeline_info)
                .push_next(&mut export_info),
            None,
        )
    }
    .map_err(|error| format!("could not create exportable timeline semaphore: {error:?}"))?;
    let source_loader =
        ash::khr::external_semaphore_fd::Device::new(&source.instance, &source.device);
    let raw_fd = match unsafe {
        source_loader.get_semaphore_fd(
            &vk::SemaphoreGetFdInfoKHR::default()
                .semaphore(source_semaphore)
                .handle_type(handle_type),
        )
    } {
        Ok(fd) => fd,
        Err(error) => {
            unsafe { source.device.destroy_semaphore(source_semaphore, None) };
            return Err(format!("could not export timeline semaphore: {error:?}"));
        }
    };
    let fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };
    let mut destination_timeline_info = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
    let destination_semaphore = match unsafe {
        destination.device.create_semaphore(
            &vk::SemaphoreCreateInfo::default().push_next(&mut destination_timeline_info),
            None,
        )
    } {
        Ok(semaphore) => semaphore,
        Err(error) => {
            unsafe { source.device.destroy_semaphore(source_semaphore, None) };
            return Err(format!(
                "could not create imported timeline semaphore: {error:?}"
            ));
        }
    };
    let destination_loader =
        ash::khr::external_semaphore_fd::Device::new(&destination.instance, &destination.device);
    let import_info = vk::ImportSemaphoreFdInfoKHR::default()
        .semaphore(destination_semaphore)
        .handle_type(handle_type)
        .fd(fd.as_raw_fd());
    if let Err(error) = unsafe { destination_loader.import_semaphore_fd(&import_info) } {
        unsafe {
            destination
                .device
                .destroy_semaphore(destination_semaphore, None);
            source.device.destroy_semaphore(source_semaphore, None);
        }
        return Err(format!("could not import timeline semaphore: {error:?}"));
    }
    let _fd_owned_by_vulkan = fd.into_raw_fd();
    Ok(ExternalTimelineSemaphore {
        source_device: source.device.clone(),
        destination_device: destination.device.clone(),
        source: source_semaphore,
        destination: destination_semaphore,
        next_value: 0,
    })
}

fn create_unbound_external_buffer(
    compute_device: &OpenVulkanComputeDevice,
    size: vk::DeviceSize,
    handle_type: vk::ExternalMemoryHandleTypeFlags,
) -> Result<UnboundExternalBuffer, String> {
    let usage = vk::BufferUsageFlags::TRANSFER_SRC
        | vk::BufferUsageFlags::TRANSFER_DST
        | vk::BufferUsageFlags::STORAGE_BUFFER;
    let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle_type);
    let buffer = unsafe {
        compute_device.device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .push_next(&mut external),
            None,
        )
    }
    .map_err(|error| format!("could not create external Vulkan buffer: {error:?}"))?;
    let mut dedicated = vk::MemoryDedicatedRequirements::default();
    let mut requirements = vk::MemoryRequirements2::default().push_next(&mut dedicated);
    unsafe {
        compute_device.device.get_buffer_memory_requirements2(
            &vk::BufferMemoryRequirementsInfo2::default().buffer(buffer),
            &mut requirements,
        );
    }
    Ok(UnboundExternalBuffer {
        device: compute_device.device.clone(),
        buffer: Some(buffer),
        size,
        requirements: requirements.memory_requirements,
        requires_dedicated: dedicated.requires_dedicated_allocation == vk::TRUE,
    })
}

fn create_external_device_local_buffers(
    source: &OpenVulkanComputeDevice,
    destination: &OpenVulkanComputeDevice,
    size: vk::DeviceSize,
) -> Result<(VulkanBuffer, VulkanBuffer), String> {
    if !source.external_memory_fd || !destination.external_memory_fd {
        return Err(
            "VK_KHR_external_memory_fd with DMA-BUF is not available on both devices".to_string(),
        );
    }
    let handle_type = vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;
    let source_raw = create_unbound_external_buffer(source, size, handle_type)?;
    let destination_raw = create_unbound_external_buffer(destination, size, handle_type)?;
    let dedicated = source_raw.requires_dedicated || destination_raw.requires_dedicated;
    if dedicated && source_raw.requirements.size != destination_raw.requirements.size {
        return Err(format!(
            "dedicated cross-device requirements disagree: source={} destination={}",
            source_raw.requirements.size, destination_raw.requirements.size
        ));
    }
    let shared_allocation_size = source_raw
        .requirements
        .size
        .max(destination_raw.requirements.size);
    let source_memory_type = memory_type_index(
        &source.memory_properties,
        source_raw.requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .ok_or_else(|| "source has no exportable device-local memory type".to_string())?;
    let source_buffer_handle = source_raw.buffer.unwrap();
    let mut export = vk::ExportMemoryAllocateInfo::default().handle_types(handle_type);
    let mut source_dedicated =
        vk::MemoryDedicatedAllocateInfo::default().buffer(source_buffer_handle);
    let mut source_allocate = vk::MemoryAllocateInfo::default()
        .allocation_size(shared_allocation_size)
        .memory_type_index(source_memory_type)
        .push_next(&mut export);
    if dedicated {
        source_allocate = source_allocate.push_next(&mut source_dedicated);
    }
    let source_memory = unsafe { source.device.allocate_memory(&source_allocate, None) }
        .map_err(|error| format!("could not allocate exportable device-local memory: {error:?}"))?;
    if let Err(error) = unsafe {
        source
            .device
            .bind_buffer_memory(source_buffer_handle, source_memory, 0)
    } {
        unsafe { source.device.free_memory(source_memory, None) };
        return Err(format!(
            "could not bind exportable device-local memory: {error:?}"
        ));
    }
    let source_buffer = source_raw.into_bound(source_memory, None);

    let source_fd_loader =
        ash::khr::external_memory_fd::Device::new(&source.instance, &source.device);
    let raw_fd = unsafe {
        source_fd_loader.get_memory_fd(
            &vk::MemoryGetFdInfoKHR::default()
                .memory(source_buffer.memory)
                .handle_type(handle_type),
        )
    }
    .map_err(|error| format!("could not export device-local DMA-BUF: {error:?}"))?;
    let fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };
    let destination_fd_loader =
        ash::khr::external_memory_fd::Device::new(&destination.instance, &destination.device);
    let mut fd_properties = vk::MemoryFdPropertiesKHR::default();
    unsafe {
        destination_fd_loader.get_memory_fd_properties(
            handle_type,
            fd.as_raw_fd(),
            &mut fd_properties,
        )
    }
    .map_err(|error| format!("could not inspect imported DMA-BUF: {error:?}"))?;
    let compatible_types =
        destination_raw.requirements.memory_type_bits & fd_properties.memory_type_bits;
    let destination_memory_type = memory_type_index(
        &destination.memory_properties,
        compatible_types,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .ok_or_else(|| {
        "destination has no device-local memory type for imported DMA-BUF".to_string()
    })?;
    let destination_buffer_handle = destination_raw.buffer.unwrap();
    let mut import = vk::ImportMemoryFdInfoKHR::default()
        .handle_type(handle_type)
        .fd(fd.as_raw_fd());
    let mut destination_dedicated =
        vk::MemoryDedicatedAllocateInfo::default().buffer(destination_buffer_handle);
    let mut destination_allocate = vk::MemoryAllocateInfo::default()
        .allocation_size(shared_allocation_size)
        .memory_type_index(destination_memory_type)
        .push_next(&mut import);
    if dedicated {
        destination_allocate = destination_allocate.push_next(&mut destination_dedicated);
    }
    let destination_memory = unsafe {
        destination
            .device
            .allocate_memory(&destination_allocate, None)
    }
    .map_err(|error| format!("could not import device-local DMA-BUF: {error:?}"))?;
    let _vulkan_owned_fd = fd.into_raw_fd();
    if let Err(error) = unsafe {
        destination
            .device
            .bind_buffer_memory(destination_buffer_handle, destination_memory, 0)
    } {
        unsafe { destination.device.free_memory(destination_memory, None) };
        return Err(format!(
            "could not bind imported device-local DMA-BUF: {error:?}"
        ));
    }
    let destination_buffer = destination_raw.into_bound(destination_memory, None);
    Ok((source_buffer, destination_buffer))
}

fn create_external_device_local_multi_buffers(
    devices: &[&OpenVulkanComputeDevice],
    size: vk::DeviceSize,
) -> Result<Vec<VulkanBuffer>, String> {
    if devices.iter().any(|device| !device.external_memory_fd) {
        return Err(
            "VK_KHR_external_memory_fd with DMA-BUF is not available on every device".to_string(),
        );
    }
    let owner = devices[0];
    let handle_type = vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;
    let raw_buffers = devices
        .iter()
        .map(|device| create_unbound_external_buffer(device, size, handle_type))
        .collect::<Result<Vec<_>, _>>()?;
    let dedicated = raw_buffers.iter().any(|raw| raw.requires_dedicated);
    let allocation_size = if dedicated {
        let required_size = raw_buffers[0].requirements.size;
        if raw_buffers
            .iter()
            .any(|raw| raw.requirements.size != required_size)
        {
            return Err(
                "dedicated cross-device buffer requirements disagree on allocation size"
                    .to_string(),
            );
        }
        required_size
    } else {
        raw_buffers
            .iter()
            .map(|raw| raw.requirements.size)
            .max()
            .unwrap_or(size)
    };
    let owner_raw = &raw_buffers[0];
    let owner_memory_type = memory_type_index(
        &owner.memory_properties,
        owner_raw.requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .ok_or_else(|| "owner has no exportable device-local memory type".to_string())?;
    let owner_buffer_handle = owner_raw.buffer.unwrap();
    let mut export = vk::ExportMemoryAllocateInfo::default().handle_types(handle_type);
    let mut owner_dedicated =
        vk::MemoryDedicatedAllocateInfo::default().buffer(owner_buffer_handle);
    let mut owner_allocate = vk::MemoryAllocateInfo::default()
        .allocation_size(allocation_size)
        .memory_type_index(owner_memory_type)
        .push_next(&mut export);
    if dedicated {
        owner_allocate = owner_allocate.push_next(&mut owner_dedicated);
    }
    let owner_memory = unsafe { owner.device.allocate_memory(&owner_allocate, None) }
        .map_err(|error| format!("could not allocate shared owner memory: {error:?}"))?;
    if let Err(error) = unsafe {
        owner
            .device
            .bind_buffer_memory(owner_buffer_handle, owner_memory, 0)
    } {
        unsafe { owner.device.free_memory(owner_memory, None) };
        return Err(format!("could not bind shared owner memory: {error:?}"));
    }

    let mut raw_buffers = raw_buffers.into_iter();
    let owner_raw = raw_buffers.next().unwrap();
    let owner_buffer = owner_raw.into_bound(owner_memory, None);
    let owner_loader = ash::khr::external_memory_fd::Device::new(&owner.instance, &owner.device);
    let mut buffers = Vec::with_capacity(devices.len());
    buffers.push(owner_buffer);

    for (peer, peer_raw) in devices[1..].iter().zip(raw_buffers) {
        let raw_fd = unsafe {
            owner_loader.get_memory_fd(
                &vk::MemoryGetFdInfoKHR::default()
                    .memory(buffers[0].memory)
                    .handle_type(handle_type),
            )
        }
        .map_err(|error| format!("could not export owner DMA-BUF: {error:?}"))?;
        let fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };
        let peer_loader = ash::khr::external_memory_fd::Device::new(&peer.instance, &peer.device);
        let mut fd_properties = vk::MemoryFdPropertiesKHR::default();
        unsafe {
            peer_loader.get_memory_fd_properties(handle_type, fd.as_raw_fd(), &mut fd_properties)
        }
        .map_err(|error| format!("could not inspect owner DMA-BUF on peer: {error:?}"))?;
        let compatible_types =
            peer_raw.requirements.memory_type_bits & fd_properties.memory_type_bits;
        let peer_memory_type = memory_type_index(
            &peer.memory_properties,
            compatible_types,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .ok_or_else(|| "peer has no device-local memory type for owner DMA-BUF".to_string())?;
        let peer_buffer_handle = peer_raw.buffer.unwrap();
        let mut import = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(handle_type)
            .fd(fd.as_raw_fd());
        let mut peer_dedicated =
            vk::MemoryDedicatedAllocateInfo::default().buffer(peer_buffer_handle);
        let mut peer_allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(allocation_size)
            .memory_type_index(peer_memory_type)
            .push_next(&mut import);
        if dedicated {
            peer_allocate = peer_allocate.push_next(&mut peer_dedicated);
        }
        let peer_memory = unsafe { peer.device.allocate_memory(&peer_allocate, None) }
            .map_err(|error| format!("could not import owner DMA-BUF on peer: {error:?}"))?;
        let _vulkan_owned_fd = fd.into_raw_fd();
        if let Err(error) = unsafe {
            peer.device
                .bind_buffer_memory(peer_buffer_handle, peer_memory, 0)
        } {
            unsafe { peer.device.free_memory(peer_memory, None) };
            return Err(format!("could not bind owner DMA-BUF on peer: {error:?}"));
        }
        buffers.push(peer_raw.into_bound(peer_memory, None));
    }
    Ok(buffers)
}

fn create_shared_host_buffers(
    source: &OpenVulkanComputeDevice,
    destination: &OpenVulkanComputeDevice,
    size: vk::DeviceSize,
) -> Result<(VulkanBuffer, VulkanBuffer), String> {
    if !source.external_memory_host || !destination.external_memory_host {
        return Err("VK_EXT_external_memory_host is not available on both devices".to_string());
    }
    let handle_type = vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
    let source_raw = create_unbound_external_buffer(source, size, handle_type)?;
    let destination_raw = create_unbound_external_buffer(destination, size, handle_type)?;
    let alignment = source
        .shared_host_alignment
        .into_iter()
        .chain(destination.shared_host_alignment)
        .chain([
            source_raw.requirements.alignment as usize,
            destination_raw.requirements.alignment as usize,
        ])
        .max()
        .ok_or_else(|| "shared-host alignment is unavailable".to_string())?;
    if !alignment.is_power_of_two() {
        return Err(format!(
            "shared-host alignment {alignment} is not a power of two"
        ));
    }
    let required_size = (source_raw
        .requirements
        .size
        .max(destination_raw.requirements.size) as usize)
        .max(size as usize);
    let allocation_size = required_size
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| "shared-host allocation size overflowed".to_string())?;
    let layout = std::alloc::Layout::from_size_align(allocation_size, alignment)
        .map_err(|error| format!("invalid shared-host allocation layout: {error}"))?;
    let pointer = unsafe { std::alloc::alloc_zeroed(layout) };
    let pointer = ptr::NonNull::new(pointer)
        .ok_or_else(|| format!("could not allocate {allocation_size} shared-host bytes"))?;
    let allocation = Arc::new(SharedHostAllocation { pointer, layout });
    let source_buffer = import_shared_host_buffer(source, source_raw, Arc::clone(&allocation))?;
    let destination_buffer = import_shared_host_buffer(destination, destination_raw, allocation)?;
    Ok((source_buffer, destination_buffer))
}

fn create_shared_host_multi_buffers(
    devices: &[&OpenVulkanComputeDevice],
    size: vk::DeviceSize,
) -> Result<Vec<VulkanBuffer>, String> {
    if devices.iter().any(|device| !device.external_memory_host) {
        return Err("VK_EXT_external_memory_host is not available on every device".to_string());
    }
    let handle_type = vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
    let raw_buffers = devices
        .iter()
        .map(|device| create_unbound_external_buffer(device, size, handle_type))
        .collect::<Result<Vec<_>, _>>()?;
    let alignment = devices
        .iter()
        .filter_map(|device| device.shared_host_alignment)
        .chain(
            raw_buffers
                .iter()
                .map(|raw| raw.requirements.alignment as usize),
        )
        .max()
        .ok_or_else(|| "shared-host alignment is unavailable".to_string())?;
    if !alignment.is_power_of_two() {
        return Err(format!(
            "shared-host alignment {alignment} is not a power of two"
        ));
    }
    let required_size = raw_buffers
        .iter()
        .map(|raw| raw.requirements.size as usize)
        .max()
        .unwrap_or(size as usize)
        .max(size as usize);
    let allocation_size = required_size
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| "shared-host allocation size overflowed".to_string())?;
    let layout = std::alloc::Layout::from_size_align(allocation_size, alignment)
        .map_err(|error| format!("invalid shared-host allocation layout: {error}"))?;
    let pointer = unsafe { std::alloc::alloc_zeroed(layout) };
    let pointer = ptr::NonNull::new(pointer)
        .ok_or_else(|| format!("could not allocate {allocation_size} shared-host bytes"))?;
    let allocation = Arc::new(SharedHostAllocation { pointer, layout });
    devices
        .iter()
        .zip(raw_buffers)
        .map(|(device, raw)| import_shared_host_buffer(device, raw, Arc::clone(&allocation)))
        .collect()
}

fn import_shared_host_buffer(
    compute_device: &OpenVulkanComputeDevice,
    raw: UnboundExternalBuffer,
    allocation: Arc<SharedHostAllocation>,
) -> Result<VulkanBuffer, String> {
    let loader = ash::ext::external_memory_host::Device::new(
        &compute_device.instance,
        &compute_device.device,
    );
    let handle_type = vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
    let mut host_properties = vk::MemoryHostPointerPropertiesEXT::default();
    let result = unsafe {
        (loader.fp().get_memory_host_pointer_properties_ext)(
            loader.device(),
            handle_type,
            allocation.pointer.as_ptr().cast(),
            &mut host_properties,
        )
    };
    if result != vk::Result::SUCCESS {
        return Err(format!(
            "could not query shared-host memory types: {result:?}"
        ));
    }
    let compatible_types = raw.requirements.memory_type_bits & host_properties.memory_type_bits;
    let memory_type = memory_type_index(
        &compute_device.memory_properties,
        compatible_types,
        vk::MemoryPropertyFlags::HOST_VISIBLE,
    )
    .ok_or_else(|| "no host-visible type can import the shared-host allocation".to_string())?;
    let mut import = vk::ImportMemoryHostPointerInfoEXT::default()
        .handle_type(handle_type)
        .host_pointer(allocation.pointer.as_ptr().cast());
    let memory = unsafe {
        compute_device.device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(allocation.layout.size() as vk::DeviceSize)
                .memory_type_index(memory_type)
                .push_next(&mut import),
            None,
        )
    }
    .map_err(|error| format!("could not import shared-host memory: {error:?}"))?;
    if let Err(error) = unsafe {
        compute_device
            .device
            .bind_buffer_memory(raw.buffer.unwrap(), memory, 0)
    } {
        unsafe { compute_device.device.free_memory(memory, None) };
        return Err(format!("could not bind shared-host memory: {error:?}"));
    }
    Ok(raw.into_bound(memory, Some(allocation)))
}

fn fill_upload_buffer(
    device: &ash::Device,
    buffer: &VulkanBuffer,
    plan: &ComputePlan,
    kernel: &DenseFormatKernel,
) -> Result<(), String> {
    let ptr = unsafe {
        device
            .map_memory(buffer.memory, 0, buffer.size, vk::MemoryMapFlags::empty())
            .map_err(|error| format!("could not map Vulkan upload buffer: {error:?}"))?
    };
    let values = ptr.cast::<u32>();
    if matches!(kernel.shape, KernelShape::KvCache) {
        let output_offset = plan.push_constants[3] as usize;
        for index in 0..output_offset {
            let first = f32_to_bf16_bits(0.25 + (((index * 2) % 251) as f32) / 512.0);
            let second = f32_to_bf16_bits(0.25 + (((index * 2 + 1) % 251) as f32) / 512.0);
            unsafe {
                values
                    .add(index)
                    .write(u32::from(first) | (u32::from(second) << 16))
            };
        }
        for index in output_offset..plan.storage_elements {
            unsafe { values.add(index).write(0x7fc0_7fc0) };
        }
    } else if matches!(kernel.shape, KernelShape::F32Gemm | KernelShape::PackedGemm) {
        let weight_offset = plan.push_constants[4] as usize;
        let output_offset = plan.push_constants[5] as usize;
        let scale_offset = plan.push_constants[8] as usize;
        for index in 0..weight_offset {
            let first = f32_to_bf16_bits(0.5 + (((index * 2) % 251) as f32) / 512.0);
            let second = f32_to_bf16_bits(0.5 + (((index * 2 + 1) % 251) as f32) / 512.0);
            unsafe {
                values
                    .add(index)
                    .write(u32::from(first) | (u32::from(second) << 16))
            };
        }
        for index in weight_offset..scale_offset {
            let value = parameter_word(kernel, index - weight_offset);
            unsafe { values.add(index).write(value) };
        }
        for index in scale_offset..output_offset {
            unsafe { values.add(index).write(scale_word(kernel)) };
        }
        for index in output_offset..plan.storage_elements {
            unsafe { values.add(index).write(0x7fc0_7fc0) };
        }
    } else if matches!(kernel.shape, KernelShape::RouterReduction) {
        let output_offset = plan.push_constants[3] as usize;
        let scale_offset = plan.push_constants[6] as usize;
        for index in 0..scale_offset {
            let value = parameter_word(kernel, index);
            unsafe { values.add(index).write(value) };
        }
        for index in scale_offset..output_offset {
            unsafe { values.add(index).write(scale_word(kernel)) };
        }
        for index in output_offset..plan.storage_elements {
            unsafe { values.add(index).write(f32::NAN.to_bits()) };
        }
    } else {
        for index in 0..plan.storage_elements {
            unsafe {
                values
                    .add(index)
                    .write((index as u32).wrapping_mul(2_654_435_761) ^ 0xa5a5_5a5a);
            }
        }
    }
    unsafe { device.unmap_memory(buffer.memory) };
    Ok(())
}

fn parameter_word(kernel: &DenseFormatKernel, relative_word: usize) -> u32 {
    match kernel.format.as_str() {
        "f32" => {
            let magnitude = (1.0 + ((relative_word % 7) as f32) / 8.0) / 4096.0;
            if relative_word.is_multiple_of(2) {
                magnitude.to_bits()
            } else {
                (-magnitude).to_bits()
            }
        }
        "f16" => 0x9400_1400,
        "bf16" => 0xba80_3a80,
        "fp8" | "fp8_e4m3" => 0x3834_b8b4,
        "fp8_e5m2" => 0x3c38_bcb8,
        "fp4" | "mxfp4" => 0xba32_9810,
        "int8" => 0x7f40_c080,
        "int4" => 0xfdb9_7531,
        "q8_0" if relative_word.is_multiple_of(9) => 0x0000_3a80,
        "q8_0" => 0x6040_c0a0,
        _ => 0,
    }
}

fn scale_word(kernel: &DenseFormatKernel) -> u32 {
    match kernel.weight_layout {
        WeightLayout::Bf16Scaled { .. } => 0x3a80_3a80,
        WeightLayout::E8m0Scaled { .. } => 0x7575_7575,
        WeightLayout::Plain | WeightLayout::NerveQ8_0 => 0,
    }
}

fn write_buffer_bytes(
    device: &ash::Device,
    buffer: &VulkanBuffer,
    bytes: &[u8],
) -> Result<(), String> {
    if bytes.len() > buffer.size as usize {
        return Err("host write exceeds Vulkan buffer size".to_string());
    }
    let mapped = unsafe {
        device
            .map_memory(buffer.memory, 0, buffer.size, vk::MemoryMapFlags::empty())
            .map_err(|error| format!("could not map Vulkan host-write buffer: {error:?}"))?
    };
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.cast::<u8>(), bytes.len());
        device.unmap_memory(buffer.memory);
    }
    Ok(())
}

fn read_buffer_bytes(
    device: &ash::Device,
    buffer: &VulkanBuffer,
    bytes: &mut [u8],
) -> Result<(), String> {
    if bytes.len() > buffer.size as usize {
        return Err("host read exceeds Vulkan buffer size".to_string());
    }
    let mapped = unsafe {
        device
            .map_memory(buffer.memory, 0, buffer.size, vk::MemoryMapFlags::empty())
            .map_err(|error| format!("could not map Vulkan host-read buffer: {error:?}"))?
    };
    unsafe {
        ptr::copy_nonoverlapping(mapped.cast::<u8>(), bytes.as_mut_ptr(), bytes.len());
        device.unmap_memory(buffer.memory);
    }
    Ok(())
}

fn checksum_bytes(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0_u64, |sum, byte| {
        sum.rotate_left(5).wrapping_add(u64::from(*byte))
    })
}

fn memory_type_index(
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    allowed_types: u32,
    required_flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    memory_properties.memory_types[..memory_properties.memory_type_count as usize]
        .iter()
        .enumerate()
        .find(|(index, memory_type)| {
            (allowed_types & (1 << index)) != 0
                && memory_type.property_flags.contains(required_flags)
        })
        .map(|(index, _)| index as u32)
}

fn summarize_samples(samples: &[Sample]) -> Option<Summary> {
    if samples.is_empty() {
        return None;
    }
    let mut durations = samples
        .iter()
        .map(|sample| sample.duration_ns)
        .collect::<Vec<_>>();
    durations.sort_unstable();
    let total_bytes = samples
        .iter()
        .map(|sample| sample.bytes_read + sample.bytes_written)
        .sum::<u64>() as f64;
    let total_operations = samples.iter().map(|sample| sample.operations).sum::<u64>() as f64;
    let total_seconds = samples
        .iter()
        .map(|sample| sample.duration_ns as f64 / 1_000_000_000.0)
        .sum::<f64>();
    Some(Summary {
        min_duration_ns: durations[0],
        median_duration_ns: durations[durations.len() / 2],
        bytes_per_second: total_bytes / total_seconds,
        operations_per_second: total_operations / total_seconds,
    })
}

fn open_compute_device(target: &Target) -> Result<OpenVulkanComputeDevice, String> {
    let vulkan = target
        .vulkan
        .as_ref()
        .ok_or_else(|| "target has no Vulkan physical-device metadata".to_string())?;
    let entry = unsafe { Entry::load() }
        .map_err(|error| format!("could not load Vulkan loader: {error}"))?;
    let app_name = CString::new("nerve-gpu-bench").expect("static string has no nul");
    let engine_name = CString::new("nerve").expect("static string has no nul");
    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(1)
        .engine_name(&engine_name)
        .engine_version(1)
        .api_version(vk::make_api_version(0, 1, 3, 0));
    let instance = unsafe {
        entry.create_instance(
            &vk::InstanceCreateInfo::default().application_info(&app_info),
            None,
        )
    }
    .map_err(|error| format!("could not create Vulkan instance: {error:?}"))?;
    let result = unsafe {
        open_compute_device_from_instance(
            entry,
            instance,
            vulkan.physical_device_index,
            vulkan.feature_flags.clone(),
            vulkan.extension_names.clone(),
        )
    };
    match result {
        Ok(device) => Ok(device),
        Err((instance, message)) => {
            unsafe { instance.destroy_instance(None) };
            Err(message)
        }
    }
}

unsafe fn open_compute_device_from_instance(
    entry: Entry,
    instance: ash::Instance,
    physical_device_index: usize,
    feature_flags: Vec<String>,
    extension_names: Vec<String>,
) -> Result<OpenVulkanComputeDevice, (ash::Instance, String)> {
    let physical_devices = unsafe { instance.enumerate_physical_devices() }.map_err(|error| {
        (
            instance.clone(),
            format!("could not enumerate Vulkan physical devices: {error:?}"),
        )
    })?;
    let physical_device = *physical_devices.get(physical_device_index).ok_or_else(|| {
        (
            instance.clone(),
            format!("Vulkan physical device index {physical_device_index} is no longer available"),
        )
    })?;
    let queue_families =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let compute_queue_family_index =
        compute_queue_family_index(&queue_families).ok_or_else(|| {
            (
                instance.clone(),
                format!(
                    "Vulkan physical device index {physical_device_index} has no compute queue"
                ),
            )
        })?;
    let timestamp_valid_bits =
        queue_families[compute_queue_family_index as usize].timestamp_valid_bits;
    let physical_device_properties =
        unsafe { instance.get_physical_device_properties(physical_device) };
    let memory_properties =
        unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let priorities = [1.0_f32];
    let queue_info = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(compute_queue_family_index)
        .queue_priorities(&priorities)];
    let mut shader_float16_int8 = vk::PhysicalDeviceShaderFloat16Int8Features::default();
    if feature_flags
        .iter()
        .any(|feature| feature == "shader_float16")
    {
        shader_float16_int8.shader_float16 = vk::TRUE;
    }
    if feature_flags.iter().any(|feature| feature == "shader_int8") {
        shader_float16_int8.shader_int8 = vk::TRUE;
    }
    let external_memory_fd = extension_names
        .iter()
        .any(|name| name == ash::khr::external_memory_fd::NAME.to_str().unwrap());
    let external_memory_dma_buf = extension_names
        .iter()
        .any(|name| name == ash::ext::external_memory_dma_buf::NAME.to_str().unwrap());
    let external_memory_host = extension_names
        .iter()
        .any(|name| name == ash::ext::external_memory_host::NAME.to_str().unwrap());
    let external_timeline_semaphore = unsafe {
        external_timeline_semaphore_supported(&instance, physical_device, &extension_names)
    };
    let shared_host_alignment = if external_memory_host {
        let mut host_properties = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
        let mut properties =
            vk::PhysicalDeviceProperties2::default().push_next(&mut host_properties);
        unsafe { instance.get_physical_device_properties2(physical_device, &mut properties) };
        usize::try_from(host_properties.min_imported_host_pointer_alignment)
            .ok()
            .filter(|alignment| alignment.is_power_of_two())
    } else {
        None
    };
    let mut enabled_extensions = Vec::new();
    if external_memory_fd {
        enabled_extensions.push(ash::khr::external_memory_fd::NAME.as_ptr());
    }
    if external_memory_dma_buf {
        enabled_extensions.push(ash::ext::external_memory_dma_buf::NAME.as_ptr());
    }
    if external_memory_host {
        enabled_extensions.push(ash::ext::external_memory_host::NAME.as_ptr());
    }
    if external_timeline_semaphore {
        enabled_extensions.push(ash::khr::external_semaphore_fd::NAME.as_ptr());
    }
    let native_fp8_dot = feature_flags_include(&feature_flags, NATIVE_FP8_DOT_FEATURE);
    let mut shader_float8 = ShaderFloat8Features::disabled();
    let mut mixed_float_dot = MixedFloatDotProductFeatures::disabled();
    if native_fp8_dot {
        shader_float8.shader_float8 = vk::TRUE;
        mixed_float_dot.shader_float8_acc_float32 = vk::TRUE;
        enabled_extensions.push(SHADER_FLOAT8_NAME.as_ptr());
        enabled_extensions.push(MIXED_FLOAT_DOT_PRODUCT_NAME.as_ptr());
    }
    let mut timeline_features = vk::PhysicalDeviceTimelineSemaphoreFeatures::default()
        .timeline_semaphore(external_timeline_semaphore);
    let mut device_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_info)
        .enabled_extension_names(&enabled_extensions)
        .push_next(&mut shader_float16_int8)
        .push_next(&mut timeline_features);
    if native_fp8_dot {
        mixed_float_dot.p_next = device_info.p_next.cast_mut();
        device_info.p_next = std::ptr::from_ref(&mixed_float_dot).cast();
        shader_float8.p_next = device_info.p_next.cast_mut();
        device_info.p_next = std::ptr::from_ref(&shader_float8).cast();
    }
    let device = unsafe { instance.create_device(physical_device, &device_info, None) }.map_err(
        |error| {
            (
                instance.clone(),
                format!("could not create Vulkan logical device: {error:?}"),
            )
        },
    )?;
    let queue = unsafe { device.get_device_queue(compute_queue_family_index, 0) };
    Ok(OpenVulkanComputeDevice {
        _entry: entry,
        device,
        instance,
        compute_queue_family_index,
        queue,
        memory_properties,
        timestamp_period_ns: physical_device_properties.limits.timestamp_period,
        timestamp_valid_bits,
        feature_flags,
        external_memory_fd: external_memory_fd && external_memory_dma_buf,
        external_memory_host,
        shared_host_alignment,
        external_timeline_semaphore,
    })
}

fn compute_queue_family_index(queue_families: &[vk::QueueFamilyProperties]) -> Option<u32> {
    queue_families
        .iter()
        .enumerate()
        .filter(|(_, family)| {
            family.queue_count > 0 && family.queue_flags.contains(vk::QueueFlags::COMPUTE)
        })
        .min_by_key(|(_, family)| family.queue_flags.contains(vk::QueueFlags::GRAPHICS))
        .map(|(index, _)| index as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefers_compute_only_queue_family() {
        let queue_families = [
            vk::QueueFamilyProperties {
                queue_flags: vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
                queue_count: 1,
                ..Default::default()
            },
            vk::QueueFamilyProperties {
                queue_flags: vk::QueueFlags::COMPUTE,
                queue_count: 1,
                ..Default::default()
            },
        ];
        assert_eq!(compute_queue_family_index(&queue_families), Some(1));
    }

    #[test]
    fn ignores_queue_families_without_compute_or_queues() {
        let queue_families = [
            vk::QueueFamilyProperties {
                queue_flags: vk::QueueFlags::GRAPHICS,
                queue_count: 1,
                ..Default::default()
            },
            vk::QueueFamilyProperties {
                queue_flags: vk::QueueFlags::COMPUTE,
                queue_count: 0,
                ..Default::default()
            },
        ];
        assert_eq!(compute_queue_family_index(&queue_families), None);
    }

    #[test]
    fn finds_memory_type_with_required_flags() {
        let mut memory_properties = vk::PhysicalDeviceMemoryProperties {
            memory_type_count: 2,
            ..Default::default()
        };
        memory_properties.memory_types[0].property_flags = vk::MemoryPropertyFlags::HOST_VISIBLE;
        memory_properties.memory_types[1].property_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        assert_eq!(
            memory_type_index(
                &memory_properties,
                0b11,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ),
            Some(1)
        );
    }

    #[test]
    fn maps_model_storage_formats_to_dense_gemm_kernels() {
        for (format, logical_elements, format_kind) in [
            ("bf16", 2, 1),
            ("fp8_e4m3", 4, 2),
            ("fp8_e5m2", 4, 3),
            ("fp4", 8, 4),
            ("mxfp4", 8, 5),
            ("int4", 8, 7),
            ("q8_0", 32, 18),
        ] {
            let kernel = dense_format_kernel(format).unwrap();
            assert_eq!(kernel.format, format);
            assert_eq!(kernel.shape, KernelShape::PackedGemm);
            assert!(kernel.pattern.contains("gemm_compute"));
            assert_eq!(kernel.format_kind, format_kind);
            assert_eq!(
                kernel.logical_elements_per_storage_element,
                logical_elements
            );
        }
    }

    #[test]
    fn maps_f16_to_gemm_for_dense_and_native_for_router() {
        for (workload, pattern, shape, required_feature) in [
            (
                "dense_projection",
                "dense_projection_f16_gemm_compute",
                KernelShape::PackedGemm,
                None,
            ),
            (
                "moe_expert",
                "moe_expert_f16_gemm_compute",
                KernelShape::PackedGemm,
                None,
            ),
            (
                "router_reduction",
                "router_reduction_f16_native_compute",
                KernelShape::RouterReduction,
                Some("shader_float16"),
            ),
        ] {
            let kernel = workload_format_kernel(workload, "f16").unwrap();
            assert_eq!(kernel.format, "f16");
            assert_eq!(kernel.pattern, pattern);
            assert_eq!(kernel.shape, shape);
            assert_eq!(kernel.required_feature, required_feature);
            assert_eq!(kernel.bytes_per_storage_element, mem::size_of::<u32>());
            assert_eq!(kernel.logical_elements_per_storage_element, 2);
        }
        assert!(feature_flags_include(
            &["shader_float16".to_string()],
            "shader_float16"
        ));
        assert!(!feature_flags_include(&[], "shader_float16"));
    }

    #[test]
    fn maps_int8_to_gemm_for_dense_and_native_for_router() {
        for (workload, pattern, shape, required_feature) in [
            (
                "dense_projection",
                "dense_projection_int8_gemm_compute",
                KernelShape::PackedGemm,
                None,
            ),
            (
                "moe_expert",
                "moe_expert_int8_gemm_compute",
                KernelShape::PackedGemm,
                None,
            ),
            (
                "router_reduction",
                "router_reduction_int8_native_compute",
                KernelShape::RouterReduction,
                Some("shader_int8"),
            ),
        ] {
            let kernel = workload_format_kernel(workload, "int8").unwrap();
            assert_eq!(kernel.format, "int8");
            assert_eq!(kernel.pattern, pattern);
            assert_eq!(kernel.shape, shape);
            assert_eq!(kernel.required_feature, required_feature);
            assert_eq!(kernel.bytes_per_storage_element, mem::size_of::<u32>());
            assert_eq!(kernel.logical_elements_per_storage_element, 4);
        }
        assert!(feature_flags_include(
            &["shader_int8".to_string()],
            "shader_int8"
        ));
        assert!(!feature_flags_include(&[], "shader_int8"));
    }

    #[test]
    fn maps_model_storage_formats_to_workload_specific_kernels() {
        for workload in ["dense_projection", "moe_expert"] {
            let f32_kernel = workload_format_kernel(workload, "f32").unwrap();
            assert_eq!(f32_kernel.format, "f32");
            assert!(matches!(f32_kernel.shape, KernelShape::F32Gemm));

            let dequant_kernel = workload_format_kernel(workload, "mxfp4").unwrap();
            assert_eq!(dequant_kernel.format, "mxfp4");
            assert_eq!(dequant_kernel.logical_elements_per_storage_element, 8);
            assert!(matches!(dequant_kernel.shape, KernelShape::PackedGemm));
            assert!(dequant_kernel.pattern.contains("gemm_compute"));
        }
        let router_kernel = workload_format_kernel("router_reduction", "mxfp4").unwrap();
        assert_eq!(router_kernel.shape, KernelShape::RouterReduction);
        assert_eq!(
            router_kernel.pattern,
            "router_reduction_format_dequant_compute"
        );
        assert!(workload_format_kernel("unknown_workload", "f32").is_none());
        assert!(workload_format_kernel("dense_projection", "unknown_format").is_none());
    }

    #[test]
    fn native_fp8_dot_patterns_cover_runtime_dense_and_expert_families() {
        assert_eq!(
            native_fp8_pattern("dense_projection", "fp8_e4m3"),
            Some("dense_projection_fp8_e4m3_native_dot_compute")
        );
        assert_eq!(
            native_fp8_pattern("dense_projection", "mxfp4"),
            Some("dense_projection_mxfp4_native_dot_compute")
        );
        assert_eq!(
            native_fp8_pattern("moe_expert", "fp8_e4m3"),
            Some("moe_expert_fp8_e4m3_native_dot_compute")
        );
        assert_eq!(native_fp8_pattern("dense_projection", "fp8_e5m2"), None);
        assert_eq!(native_fp8_pattern("router_reduction", "fp8_e4m3"), None);
    }

    #[test]
    fn gemm_plan_uses_payload_for_resident_weights_and_chainable_activations() {
        let kernel = workload_format_kernel("dense_projection", "f32").unwrap();
        let full = compute_plan_for_payload(5 * 1024 * 1024, &kernel);
        let weight_words = full.push_constants[5] - full.push_constants[4];
        assert!(weight_words as usize * mem::size_of::<u32>() <= 5 * 1024 * 1024);
        assert_eq!(full.push_constants[0], 16);
        assert_eq!(full.push_constants[1], full.push_constants[2]);
        assert_eq!(full.activation_size, full.output_size);
        assert!(full.operations > 0);
    }

    #[test]
    fn generated_gemm_geometry_is_native_fp8_vector_aligned() {
        for format in ["fp8_e4m3", "mxfp4"] {
            let kernel = workload_format_kernel("dense_projection_decode", format).unwrap();
            for payload in [4 * 1024, 5 * 1024 * 1024, 64 * 1024 * 1024] {
                let plan = compute_plan_for_payload(payload, &kernel);
                assert!((plan.push_constants[1] as usize).is_multiple_of(4));
                assert!((plan.push_constants[2] as usize).is_multiple_of(4));
            }
        }
    }

    #[test]
    fn generated_weights_are_balanced_and_chain_safe() {
        let f32_kernel = workload_format_kernel("dense_projection_decode", "f32").unwrap();
        let positive = f32::from_bits(parameter_word(&f32_kernel, 0));
        let negative = f32::from_bits(parameter_word(&f32_kernel, 1));
        assert!(positive.is_finite() && positive > 0.0 && positive < 0.001);
        assert!(negative.is_finite() && negative < 0.0 && negative > -0.001);

        let f16_kernel = workload_format_kernel("dense_projection_decode", "f16").unwrap();
        assert_eq!(parameter_word(&f16_kernel, 0), 0x9400_1400);
        let bf16_kernel = workload_format_kernel("dense_projection_decode", "bf16").unwrap();
        assert_eq!(parameter_word(&bf16_kernel, 0), 0xba80_3a80);

        let fp8_kernel = workload_format_kernel("dense_projection_decode", "fp8_e4m3").unwrap();
        assert_eq!(scale_word(&fp8_kernel), 0x3a80_3a80);
        let mxfp4_kernel = workload_format_kernel("dense_projection_decode", "mxfp4").unwrap();
        assert_eq!(parameter_word(&mxfp4_kernel, 0), 0xba32_9810);
        assert_eq!(scale_word(&mxfp4_kernel), 0x7575_7575);
    }

    #[test]
    fn tensor_parallel_gemm_shards_exactly_match_full_work_and_output() {
        let kernel = workload_format_kernel("dense_projection", "f32").unwrap();
        let full = compute_plan_for_payload(5 * 1024 * 1024, &kernel);
        for participants in 2..=8 {
            let (m, n, k, shards) =
                tensor_parallel_shard_plans(5 * 1024 * 1024, &kernel, participants).unwrap();
            assert_eq!(shards.len(), participants);
            assert_eq!(
                shards.iter().map(|plan| plan.operations).sum::<u64>(),
                full.operations
            );
            assert_eq!(
                shards
                    .iter()
                    .map(|plan| plan.push_constants[1] as usize)
                    .sum::<usize>(),
                n
            );
            assert_eq!(m, full.push_constants[0] as usize);
            assert_eq!(k, full.push_constants[2] as usize);
            let mut expected_offset = 0_u32;
            for shard in shards {
                assert_eq!(shard.push_constants[4], expected_offset);
                expected_offset += shard.push_constants[1];
            }
        }
    }

    #[test]
    fn packed_gemm_plan_records_format_decode_parameters() {
        let kernel = workload_format_kernel("moe_expert", "mxfp4").unwrap();
        let plan = compute_plan_for_payload(5 * 1024 * 1024, &kernel);
        assert_eq!(kernel.shape, KernelShape::PackedGemm);
        assert_eq!(plan.push_constants[6], 8);
        assert_eq!(plan.push_constants[7], 5);
        let weight_words = plan.push_constants[5] - plan.push_constants[4];
        assert!(weight_words as usize * mem::size_of::<u32>() <= 5 * 1024 * 1024);
        assert_eq!(
            plan.output_size,
            plan.activation_size * MOE_SELECTED_EXPERTS as u64
        );
        assert_eq!(
            plan.push_constants[1],
            plan.push_constants[2] * MOE_SELECTED_EXPERTS as u32
        );
        assert!(plan.operations > 0);
    }

    #[test]
    fn scaled_router_plan_allocates_parameters_scales_and_output() {
        let kernel = workload_format_kernel("router_reduction_prefill", "mxfp4").unwrap();
        let plan = compute_plan_for_payload(5 * 1024 * 1024, &kernel);
        assert!(plan.push_constants[6] < plan.push_constants[3]);
        assert!(plan.output_offset + plan.output_size <= plan.buffer_size);
        assert_eq!(plan.output_size, 16 * mem::size_of::<f32>() as u64);
    }

    #[test]
    fn kv_cache_plan_uses_bf16_context_and_phase_query_counts() {
        let payload_bytes = 64 * 1024;
        let decode = workload_format_kernel("kv_cache_decode", "bf16").unwrap();
        let prefill = workload_format_kernel("kv_cache_prefill", "bf16").unwrap();
        let decode_plan = compute_plan_for_payload(payload_bytes, &decode);
        let prefill_plan = compute_plan_for_payload(payload_bytes, &prefill);
        assert_eq!(decode_plan.push_constants[0], 256);
        assert_eq!(decode_plan.push_constants[2], 1);
        assert_eq!(prefill_plan.push_constants[2], 16);
        assert_eq!(
            decode_plan.push_constants[1],
            prefill_plan.push_constants[1]
        );
        assert_eq!(decode_plan.output_size, 256 * mem::size_of::<u16>() as u64);
        assert_eq!(
            prefill_plan.output_size,
            16 * 256 * mem::size_of::<u16>() as u64
        );
    }

    #[test]
    fn tensor_parallel_is_limited_to_runtime_supported_workloads() {
        assert!(supports_tensor_parallel("dense_projection_decode"));
        assert!(supports_tensor_parallel("dense_projection_prefill"));
        assert!(supports_tensor_parallel("moe_expert_decode"));
        assert!(supports_tensor_parallel("moe_expert_prefill"));
        assert!(!supports_tensor_parallel("router_reduction_decode"));
        assert!(!supports_tensor_parallel("kv_cache_prefill"));
    }

    #[test]
    fn unknown_vulkan_axes_are_skipped() {
        let formats = vec!["mxfp4".to_string(), "unknown_format".to_string()];
        let workloads = vec![
            "dense_projection".to_string(),
            "moe_expert".to_string(),
            "router_reduction".to_string(),
        ];
        let skipped = skipped_vulkan_axes(&formats, &workloads);
        assert!(!skipped.contains(&("mxfp4", "dense_projection")));
        assert!(!skipped.contains(&("mxfp4", "moe_expert")));
        assert!(!skipped.contains(&("mxfp4", "router_reduction")));
        assert!(skipped.contains(&("unknown_format", "dense_projection")));
        assert!(skipped.contains(&("unknown_format", "moe_expert")));
        assert!(skipped.contains(&("unknown_format", "router_reduction")));
    }

    #[test]
    fn dense_pair_failure_rows_cover_only_executable_axes() {
        let formats = vec![
            "f32".to_string(),
            "mxfp4".to_string(),
            "unknown_format".to_string(),
        ];
        let workloads = vec![
            "dense_projection".to_string(),
            "router_reduction".to_string(),
        ];
        let measurements = failed_dense_pair_measurements(
            "left",
            "right",
            TENSOR_PARALLEL_PAIR_PATTERN,
            TWO_TARGET_TENSOR_PARALLEL_STRATEGY,
            64 * 1024,
            &formats,
            &workloads,
            "no device",
        );
        let ids = measurements
            .iter()
            .map(|measurement| measurement.workload_id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            ids,
            [
                "synthetic_tensor_parallel_small_payload:dense_projection:f32",
                "synthetic_tensor_parallel_small_payload:dense_projection:mxfp4",
            ]
        );
        assert!(measurements.iter().all(|measurement| {
            measurement.placement_strategy == TWO_TARGET_TENSOR_PARALLEL_STRATEGY
                && measurement.status == "failed"
                && measurement.source_target_id == "left"
                && measurement.destination_target_id == "right"
        }));
    }

    #[test]
    fn group_payload_split_preserves_total_bytes() {
        assert_eq!(split_payload_bytes(10, 3), [4, 3, 3]);
        assert_eq!(split_payload_bytes(11, 3), [4, 4, 3]);
        assert_eq!(split_payload_bytes(12, 3), [4, 4, 4]);
        assert_eq!(split_payload_bytes(10, 4), [3, 3, 2, 2]);
    }

    #[test]
    fn serial_order_uses_measured_directed_edge_costs() {
        let targets = vec![
            "gpu:c".to_string(),
            "gpu:a".to_string(),
            "gpu:b".to_string(),
        ];
        let costs = BTreeMap::from([
            (("gpu:a".to_string(), "gpu:b".to_string()), 10),
            (("gpu:b".to_string(), "gpu:c".to_string()), 10),
            (("gpu:a".to_string(), "gpu:c".to_string()), 50),
            (("gpu:c".to_string(), "gpu:b".to_string()), 50),
            (("gpu:b".to_string(), "gpu:a".to_string()), 50),
            (("gpu:c".to_string(), "gpu:a".to_string()), 50),
        ]);
        assert_eq!(
            best_serial_order(&targets, &costs),
            ["gpu:a", "gpu:b", "gpu:c"]
        );
        assert_eq!(
            best_serial_order(&targets, &BTreeMap::new()),
            ["gpu:a", "gpu:b", "gpu:c"]
        );
    }

    #[test]
    fn failed_tensor_parallel_group_preserves_order_and_strategy() {
        let target_ids = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ];
        let measurement = tensor_parallel_group_status_measurement(
            &target_ids,
            "failed",
            10,
            "moe_expert",
            "f32",
            "no device",
        );
        assert_eq!(measurement.target_ids, target_ids);
        assert_eq!(
            measurement.placement_strategy,
            MULTI_TARGET_TENSOR_PARALLEL_STRATEGY
        );
        assert_eq!(measurement.participant_count, 4);
        assert_eq!(measurement.payload_bytes_per_participant, [3, 3, 2, 2]);
        assert_eq!(measurement.status, "failed");
    }

    #[test]
    fn tensor_parallel_shards_keep_parameters_local_and_output_offsets_global() {
        let kernel = workload_format_kernel("moe_expert", "mxfp4").unwrap();
        let (_, n, k, shards) = tensor_parallel_shard_plans(5 * 1024 * 1024, &kernel, 3).unwrap();
        let parameter_bytes = shards
            .iter()
            .map(|shard| (shard.parameter_words + shard.scale_words) * mem::size_of::<u32>())
            .sum::<usize>();
        assert!(parameter_bytes <= 5 * 1024 * 1024);
        let full = compute_plan_for_payload(5 * 1024 * 1024, &kernel);
        assert_eq!(
            shards.iter().map(|shard| shard.operations).sum::<u64>(),
            full.operations
        );
        assert_eq!(
            shards.last().unwrap().push_constants[4] + shards.last().unwrap().push_constants[1],
            n as u32
        );
        assert!(
            shards
                .iter()
                .all(|shard| (shard.push_constants[1] as usize).is_multiple_of(k))
        );
        assert_eq!(
            shards
                .iter()
                .map(|shard| shard.push_constants[1] as usize / k)
                .sum::<usize>(),
            MOE_SELECTED_EXPERTS
        );
        assert!(shards[1].parameter_word_offset > 0);
    }

    #[test]
    fn moe_tensor_parallel_can_shard_expert_rows_across_seven_devices() {
        let kernel = workload_format_kernel("moe_expert", "mxfp4").unwrap();
        let (_, n, _, shards) = tensor_parallel_shard_plans(5 * 1024 * 1024, &kernel, 7).unwrap();
        assert_eq!(shards.len(), 7);
        assert!(
            shards
                .iter()
                .all(|shard| shard.push_constants[1] > 0
                    && shard.push_constants[1].is_multiple_of(2))
        );
        assert_eq!(
            shards
                .iter()
                .map(|shard| shard.push_constants[1] as usize)
                .sum::<usize>(),
            n
        );
    }

    #[test]
    fn gemm_plan_identifies_only_the_output_region_for_readback() {
        let kernel = workload_format_kernel("dense_projection", "mxfp4").unwrap();
        let plan = compute_plan_for_payload(5 * 1024 * 1024, &kernel);
        assert!(plan.output_offset > 0);
        assert!(plan.output_size > 0);
        assert!(plan.output_offset + plan.output_size <= plan.buffer_size);
        assert!(plan.output_size < plan.buffer_size);
    }

    #[test]
    fn summarizes_vulkan_samples() {
        let samples = [
            Sample {
                sample_index: 0,
                duration_ns: 10,
                iterations: 1,
                bytes_read: 4,
                bytes_written: 4,
                operations: 2,
            },
            Sample {
                sample_index: 1,
                duration_ns: 20,
                iterations: 1,
                bytes_read: 4,
                bytes_written: 4,
                operations: 2,
            },
        ];
        let summary = summarize_samples(&samples).unwrap();
        assert_eq!(summary.min_duration_ns, 10);
        assert_eq!(summary.median_duration_ns, 20);
        assert!(summary.bytes_per_second > 0.0);
        assert!(summary.operations_per_second > 0.0);
    }
}
