use super::schema::HardwareCalibrationWorkload;

pub(super) fn compute_shader_source(
    workload: &HardwareCalibrationWorkload,
) -> Result<String, String> {
    match workload.operation.as_str() {
        "shader_scalar" | "shader_vector" => arithmetic_source(
            workload
                .regime
                .get("format")
                .map(String::as_str)
                .unwrap_or("f32"),
            false,
        ),
        "packed_dot_product" => arithmetic_source(
            workload
                .regime
                .get("format")
                .map(String::as_str)
                .unwrap_or("i8"),
            true,
        ),
        "cooperative_matrix_multiply" => cooperative_matrix_source(
            workload
                .regime
                .get("format")
                .map(String::as_str)
                .unwrap_or("bf16"),
        ),
        "subgroup_reduce" | "subgroup_scan" | "subgroup_shuffle" | "subgroup_ballot" => {
            subgroup_source(&workload.operation)
        }
        "sparse_compaction" => sparse_compaction_source(),
        "bitfield_mix" => bitfield_mix_source(),
        "sequential_copy"
        | "strided_read"
        | "gather_scatter"
        | "packed_decode"
        | "register_pressure_sweep"
        | "shared_memory_tiled_copy" => memory_source(&workload.operation),
        "atomic_add" => atomic_source(
            workload
                .regime
                .get("contention")
                .map(String::as_str)
                .unwrap_or("global"),
        ),
        "command_queues" | "indirect_work_generation" | "resident_command_replay" => {
            scheduling_source()
        }
        unsupported => Err(format!(
            "Vulkan compute calibrator has no shader for {unsupported:?}"
        )),
    }
}

fn arithmetic_source(format: &str, packed_dot: bool) -> Result<String, String> {
    let body = match (format, packed_dot) {
        ("f32", false) => {
            r#"
    vec4 value = vec4(
        uintBitsToFloat(input_words.words[index * 4u]),
        uintBitsToFloat(input_words.words[index * 4u + 1u]),
        uintBitsToFloat(input_words.words[index * 4u + 2u]),
        uintBitsToFloat(input_words.words[index * 4u + 3u])
    );
    for (uint iteration = 0u; iteration < 32u; iteration++) {
        value = fma(value, vec4(1.0001220703125), vec4(0.0009765625));
    }
    output_words.words[index] = floatBitsToUint(dot(value, vec4(1.0)));
"#
        }
        ("f64", false) => {
            r#"
    dvec2 value = dvec2(
        double(input_words.words[index * 4u]) * 0.00000011920928955078125,
        double(input_words.words[index * 4u + 1u]) * 0.00000011920928955078125
    );
    for (uint iteration = 0u; iteration < 32u; iteration++) {
        value = fma(value, dvec2(1.0001220703125), dvec2(0.0009765625));
    }
    output_words.words[index] = uint(value.x + value.y);
"#
        }
        ("f16", false) => {
            r#"
    vec4 source = vec4(
        uintBitsToFloat(input_words.words[index * 4u]),
        uintBitsToFloat(input_words.words[index * 4u + 1u]),
        uintBitsToFloat(input_words.words[index * 4u + 2u]),
        uintBitsToFloat(input_words.words[index * 4u + 3u])
    );
    f16vec4 value = f16vec4(source);
    for (uint iteration = 0u; iteration < 32u; iteration++) {
        value = value * f16vec4(1.001) + f16vec4(0.0009);
    }
    output_words.words[index] = floatBitsToUint(float(value.x + value.y + value.z + value.w));
"#
        }
        ("bf16", true) => {
            r#"
    uint first = input_words.words[index * 2u];
    uint second = input_words.words[index * 2u + 1u];
    bf16vec2 a = bf16vec2(
        bfloat16_t(uintBitsToFloat(first << 16u)),
        bfloat16_t(uintBitsToFloat(first & 0xffff0000u))
    );
    bf16vec2 b = bf16vec2(
        bfloat16_t(uintBitsToFloat(second << 16u)),
        bfloat16_t(uintBitsToFloat(second & 0xffff0000u))
    );
    float sum = 0.0;
    for (uint iteration = 0u; iteration < 64u; iteration++) {
        sum = bf16_dot2_acc32(a, b, sum);
    }
    output_words.words[index] = floatBitsToUint(sum);
"#
        }
        ("f16", true) => {
            r#"
    f16vec2 a = f16vec2(unpackHalf2x16(input_words.words[index * 2u]));
    f16vec2 b = f16vec2(unpackHalf2x16(input_words.words[index * 2u + 1u]));
    float sum = 0.0;
    for (uint iteration = 0u; iteration < 64u; iteration++) {
        sum = f16_dot2_acc32(a, b, sum);
    }
    output_words.words[index] = floatBitsToUint(sum);
"#
        }
        ("f8_e4m3", true) => {
            r#"
    uint first = input_words.words[index * 2u];
    uint second = input_words.words[index * 2u + 1u];
    fe4m3vec4 a = uintBitsToFloate4m3EXT(u8vec4(
        uint8_t(first), uint8_t(first >> 8u), uint8_t(first >> 16u), uint8_t(first >> 24u)
    ));
    fe4m3vec4 b = uintBitsToFloate4m3EXT(u8vec4(
        uint8_t(second), uint8_t(second >> 8u), uint8_t(second >> 16u), uint8_t(second >> 24u)
    ));
    float sum = 0.0;
    for (uint iteration = 0u; iteration < 64u; iteration++) {
        sum = fp8_dot4_acc32(a, b, sum);
    }
    output_words.words[index] = floatBitsToUint(sum);
"#
        }
        ("i8" | "u8", true) => {
            r#"
    int32_t first = int32_t(input_words.words[index * 2u]);
    int32_t second = int32_t(input_words.words[index * 2u + 1u]);
    int sum = 0;
    for (uint iteration = 0u; iteration < 64u; iteration++) {
        sum += dotPacked4x8EXT(first, second);
    }
    output_words.words[index] = uint(sum);
"#
        }
        ("i8", false) => {
            r#"
    i8vec4 value = i8vec4(unpack8(int(input_words.words[index])));
    for (uint iteration = 0u; iteration < 32u; iteration++) {
        value = value * i8vec4(3) + i8vec4(1);
    }
    output_words.words[index] = uint(uint8_t(value.x)) | (uint(uint8_t(value.y)) << 8u)
        | (uint(uint8_t(value.z)) << 16u) | (uint(uint8_t(value.w)) << 24u);
"#
        }
        ("u8", false) => {
            r#"
    u8vec4 value = u8vec4(unpack8(input_words.words[index]));
    for (uint iteration = 0u; iteration < 32u; iteration++) {
        value = value * u8vec4(3) + u8vec4(1);
    }
    output_words.words[index] = uint(value.x) | (uint(value.y) << 8u)
        | (uint(value.z) << 16u) | (uint(value.w) << 24u);
"#
        }
        ("i16", false) => {
            r#"
    i16vec2 value = i16vec2(unpack16(int(input_words.words[index])));
    for (uint iteration = 0u; iteration < 32u; iteration++) {
        value = value * i16vec2(17) + i16vec2(3);
    }
    output_words.words[index] = uint(uint16_t(value.x)) | (uint(uint16_t(value.y)) << 16u);
"#
        }
        ("u16", false) => {
            r#"
    u16vec2 value = u16vec2(unpack16(input_words.words[index]));
    for (uint iteration = 0u; iteration < 32u; iteration++) {
        value = value * u16vec2(17) + u16vec2(3);
    }
    output_words.words[index] = uint(value.x) | (uint(value.y) << 16u);
"#
        }
        ("i32", false) => {
            r#"
    ivec4 value = ivec4(input_words.words[index * 4u], input_words.words[index * 4u + 1u],
        input_words.words[index * 4u + 2u], input_words.words[index * 4u + 3u]);
    for (uint iteration = 0u; iteration < 32u; iteration++) {
        value = value * ivec4(1664525) + ivec4(1013904223);
    }
    output_words.words[index] = uint(value.x ^ value.y ^ value.z ^ value.w);
"#
        }
        ("u32", false) => {
            r#"
    uvec4 value = uvec4(input_words.words[index * 4u], input_words.words[index * 4u + 1u],
        input_words.words[index * 4u + 2u], input_words.words[index * 4u + 3u]);
    for (uint iteration = 0u; iteration < 32u; iteration++) {
        value = value * uvec4(1664525u) + uvec4(1013904223u);
    }
    output_words.words[index] = value.x ^ value.y ^ value.z ^ value.w;
"#
        }
        ("i64", false) => {
            r#"
    int64_t value = int64_t(input_words.words[index]);
    for (uint iteration = 0u; iteration < 32u; iteration++) {
        value = value * int64_t(6364136223846793005L) + int64_t(1442695040888963407L);
    }
    output_words.words[index] = uint(value);
"#
        }
        ("u64", false) => {
            r#"
    uint64_t value = uint64_t(input_words.words[index]);
    for (uint iteration = 0u; iteration < 32u; iteration++) {
        value = value * uint64_t(6364136223846793005UL) + uint64_t(1442695040888963407UL);
    }
    output_words.words[index] = uint(value);
"#
        }
        _ => {
            return Err(format!(
                "Vulkan arithmetic calibration does not support format {format:?}"
            ));
        }
    };
    let extensions = match (format, packed_dot) {
        ("f64", false) => "#extension GL_ARB_gpu_shader_fp64 : require\n",
        ("f16", false) => "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n",
        ("i8" | "u8", false) => {
            concat!("#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require\n",)
        }
        ("i16" | "u16", false) => {
            "#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require\n"
        }
        ("i64" | "u64", false) => {
            "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n"
        }
        ("i8" | "u8", true) => concat!(
            "#extension GL_EXT_integer_dot_product : require\n",
            "#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require\n"
        ),
        ("bf16", true) => concat!(
            "#extension GL_EXT_bfloat16 : require\n",
            "#extension GL_EXT_spirv_intrinsics : require\n",
            "spirv_instruction(extensions=[\"SPV_VALVE_mixed_float_dot_product\"],",
            " capabilities=[6914], id=6916)\n",
            "float bf16_dot2_acc32(bf16vec2 a, bf16vec2 b, float accumulator);\n"
        ),
        ("f16", true) => concat!(
            "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n",
            "#extension GL_EXT_spirv_intrinsics : require\n",
            "spirv_instruction(extensions=[\"SPV_VALVE_mixed_float_dot_product\"],",
            " capabilities=[6912], id=6916)\n",
            "float f16_dot2_acc32(f16vec2 a, f16vec2 b, float accumulator);\n"
        ),
        ("f8_e4m3", true) => concat!(
            "#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require\n",
            "#extension GL_EXT_float_e4m3 : require\n",
            "#extension GL_EXT_spirv_intrinsics : require\n",
            "spirv_instruction(extensions=[\"SPV_VALVE_mixed_float_dot_product\"],",
            " capabilities=[6915], id=6918)\n",
            "float fp8_dot4_acc32(fe4m3vec4 a, fe4m3vec4 b, float accumulator);\n"
        ),
        _ => "",
    };
    Ok(format!(
        r#"#version 460
{extensions}
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) readonly buffer InputWords {{ uint words[]; }} input_words;
layout(set = 0, binding = 1) writeonly buffer OutputWords {{ uint words[]; }} output_words;
layout(push_constant) uniform Control {{ uint output_count; }} control;
void main() {{
    uint index = gl_GlobalInvocationID.x;
    if (index >= control.output_count) {{ return; }}
{body}
}}
"#
    ))
}

fn cooperative_matrix_source(format: &str) -> Result<String, String> {
    let (extensions, value_type, accumulator_type, zero) = match format {
        "bf16" => (
            concat!(
                "#extension GL_EXT_shader_16bit_storage : require\n",
                "#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require\n",
                "#extension GL_EXT_bfloat16 : require\n"
            ),
            "bfloat16_t",
            "float",
            "0.0",
        ),
        "f16" => (
            concat!(
                "#extension GL_EXT_shader_16bit_storage : require\n",
                "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n"
            ),
            "float16_t",
            "float",
            "0.0",
        ),
        "f8_e4m3" => (
            concat!(
                "#extension GL_EXT_shader_8bit_storage : require\n",
                "#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require\n",
                "#extension GL_EXT_float_e4m3 : require\n"
            ),
            "floate4m3_t",
            "float",
            "0.0",
        ),
        "i8" => (
            concat!(
                "#extension GL_EXT_shader_8bit_storage : require\n",
                "#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require\n"
            ),
            "int8_t",
            "int",
            "0",
        ),
        _ => {
            return Err(format!(
                "cooperative-matrix calibration does not support {format:?}"
            ));
        }
    };
    Ok(format!(
        r#"#version 460
{extensions}
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) readonly buffer InputValues {{ {value_type} values[]; }} input_values;
layout(set = 0, binding = 1) buffer OutputValues {{ {accumulator_type} values[]; }} output_values;
layout(push_constant) uniform Control {{ uint tile_count; }} control;
shared {value_type} a_values[256];
shared {value_type} b_values[256];
shared {accumulator_type} result_values[256];
void main() {{
    uint tile = gl_WorkGroupID.x;
    if (tile >= control.tile_count) {{ return; }}
    uint lane = gl_LocalInvocationID.x;
    for (uint index = lane; index < 256u; index += 64u) {{
        a_values[index] = input_values.values[(tile * 512u + index) % 33554432u];
        b_values[index] = input_values.values[(tile * 512u + 256u + index) % 33554432u];
    }}
    barrier();
    coopmat<{value_type}, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> a;
    coopmat<{value_type}, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> b;
    coopmat<{accumulator_type}, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> result =
        coopmat<{accumulator_type}, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>({zero});
    coopMatLoad(a, a_values, 0u, 16u, gl_CooperativeMatrixLayoutRowMajor);
    coopMatLoad(b, b_values, 0u, 16u, gl_CooperativeMatrixLayoutRowMajor);
    result = coopMatMulAdd(a, b, result);
    coopMatStore(result, result_values, 0u, 16u, gl_CooperativeMatrixLayoutRowMajor);
    barrier();
    for (uint index = lane; index < 256u; index += 64u) {{
        output_values.values[tile * 256u + index] = result_values[index];
    }}
}}
"#
    ))
}

fn subgroup_source(operation: &str) -> Result<String, String> {
    let expression = match operation {
        "subgroup_reduce" => "subgroupAdd(value)",
        "subgroup_scan" => "subgroupInclusiveAdd(value)",
        "subgroup_shuffle" => "subgroupShuffleXor(value, 1u)",
        "subgroup_ballot" => "float(subgroupBallotBitCount(subgroupBallot(value > 0.5)))",
        _ => return Err(format!("unsupported subgroup operation {operation:?}")),
    };
    Ok(format!(
        r#"#version 460
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) readonly buffer InputWords {{ uint words[]; }} input_words;
layout(set = 0, binding = 1) writeonly buffer OutputWords {{ uint words[]; }} output_words;
layout(push_constant) uniform Control {{ uint output_count; }} control;
void main() {{
    uint index = gl_GlobalInvocationID.x;
    if (index >= control.output_count) {{ return; }}
    float value = uintBitsToFloat(input_words.words[index]);
    output_words.words[index] = floatBitsToUint({expression});
}}
"#
    ))
}

fn memory_source(operation: &str) -> Result<String, String> {
    let body = match operation {
        "sequential_copy" => "uint value = input_words.words[index];",
        "strided_read" => {
            "uint source = (index * 4093u) % control.output_count; uint value = input_words.words[source];"
        }
        "gather_scatter" => {
            "uint source = input_words.words[index] % control.output_count; uint value = input_words.words[source];"
        }
        "packed_decode" => {
            "uint packed = input_words.words[index]; uint value = (packed & 255u) + ((packed >> 8u) & 255u) + ((packed >> 16u) & 255u) + (packed >> 24u);"
        }
        "register_pressure_sweep" => {
            "uint value = input_words.words[index]; for (uint i=0u; i<64u; i++) { value = value * 1664525u + i + 1013904223u; value = (value << 7u) | (value >> 25u); }"
        }
        "shared_memory_tiled_copy" => {
            "shared_words[gl_LocalInvocationID.x] = input_words.words[index]; barrier(); uint value = shared_words[(gl_LocalInvocationID.x * 17u) & 255u];"
        }
        _ => return Err(format!("unsupported memory calibration {operation:?}")),
    };
    Ok(format!(
        r#"#version 460
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) readonly buffer InputWords {{ uint words[]; }} input_words;
layout(set = 0, binding = 1) writeonly buffer OutputWords {{ uint words[]; }} output_words;
layout(push_constant) uniform Control {{ uint output_count; }} control;
shared uint shared_words[256];
void main() {{
    uint index = gl_GlobalInvocationID.x;
    if (index >= control.output_count) {{ return; }}
    {body}
    output_words.words[index] = value;
}}
"#
    ))
}

fn atomic_source(contention: &str) -> Result<String, String> {
    let target = match contention {
        "independent" => "index",
        "workgroup" => "gl_WorkGroupID.x",
        "global" => "0u",
        _ => return Err(format!("unsupported atomic contention {contention:?}")),
    };
    Ok(format!(
        r#"#version 460
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) readonly buffer InputWords {{ uint words[]; }} input_words;
layout(set = 0, binding = 1) buffer OutputWords {{ uint words[]; }} output_words;
layout(push_constant) uniform Control {{ uint output_count; }} control;
void main() {{
    uint index = gl_GlobalInvocationID.x;
    if (index >= control.output_count) {{ return; }}
    atomicAdd(output_words.words[{target}], (input_words.words[index] & 1u) + 1u);
}}
"#
    ))
}

fn sparse_compaction_source() -> Result<String, String> {
    Ok(r#"#version 460
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) readonly buffer InputWords { uint words[]; } input_words;
layout(set = 0, binding = 1) buffer OutputWords { uint words[]; } output_words;
layout(push_constant) uniform Control { uint output_count; } control;
void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= control.output_count) { return; }
    uint value = input_words.words[index];
    if ((value & 7u) == 0u) {
        uint slot = atomicAdd(output_words.words[0], 1u);
        uint capacity = max(control.output_count - 1u, 1u);
        output_words.words[1u + slot % capacity] = value ^ index;
    }
}
"#
    .to_string())
}

fn bitfield_mix_source() -> Result<String, String> {
    Ok(r#"#version 460
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) readonly buffer InputWords { uint words[]; } input_words;
layout(set = 0, binding = 1) writeonly buffer OutputWords { uint words[]; } output_words;
layout(push_constant) uniform Control { uint output_count; } control;
void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= control.output_count) { return; }
    uint value = input_words.words[index];
    value = bitfieldReverse(value);
    value = bitfieldInsert(value, value ^ 0x9e3779b9u, 7, 13);
    value ^= bitCount(value) * 0x45d9f3bu;
    value ^= uint(findMSB(value)) << 24u;
    value ^= uint(findLSB(value)) << 16u;
    output_words.words[index] = value;
}
"#
    .to_string())
}

fn scheduling_source() -> Result<String, String> {
    Ok(r#"#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) readonly buffer InputWords { uint words[]; } input_words;
layout(set = 0, binding = 1) buffer OutputWords { uint words[]; } output_words;
layout(push_constant) uniform Control { uint output_count; } control;
void main() {
    if (gl_GlobalInvocationID.x == 0u) {
        output_words.words[0] = input_words.words[0] + control.output_count;
    }
}
"#
    .to_string())
}
