use super::schema::HardwareCalibrationWorkload;
use sha2::{Digest, Sha256};
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};

pub(super) struct CpuCalibrationWorkload {
    operation: String,
    regime: std::collections::BTreeMap<String, String>,
    bytes: Vec<u8>,
    output: Vec<u8>,
    words: Vec<u64>,
    floating32: Vec<f32>,
    floating: Vec<f64>,
    bfloat16: Vec<u16>,
    atomics: Vec<AtomicU64>,
    atomic_operation_count: usize,
    generated_program: Option<GeneratedProgram>,
    checksum: u64,
}

impl CpuCalibrationWorkload {
    pub(super) fn prepare(workload: &HardwareCalibrationWorkload) -> Result<Self, String> {
        let requested_bytes = workload
            .regime
            .get("working_set_bytes")
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or_else(|| {
                usize::try_from(
                    workload
                        .work
                        .bytes_read_per_iteration
                        .max(workload.work.bytes_written_per_iteration),
                )
                .unwrap_or(usize::MAX)
            });
        let item_count = usize::try_from(workload.work.items_per_iteration)
            .map_err(|_| "CPU calibration item count exceeds usize".to_string())?;
        let needs_bytes = matches!(
            workload.operation.as_str(),
            "branch_dispatch"
                | "sequential_read"
                | "sequential_copy"
                | "strided_read"
                | "pointer_chase"
                | "gather_scatter"
                | "numa_local_copy"
                | "numa_remote_copy"
        );
        let needs_words = matches!(
            workload.operation.as_str(),
            "scalar_integer"
                | "out_of_order_control_flow"
                | "vector_fused_arithmetic"
                | "bit_population_mix"
                | "generated_code_dispatch"
                | "pointer_chase"
        );
        let needs_floating = matches!(
            workload.operation.as_str(),
            "scalar_floating_point" | "blocked_matrix_multiply"
        ) || workload.operation == "vector_fused_arithmetic"
            && workload
                .regime
                .get("format")
                .is_some_and(|format| matches!(format.as_str(), "f32" | "f64" | "bf16"));
        let needs_atomics = workload.operation == "atomic_fetch_add";
        let byte_count = if needs_bytes {
            requested_bytes.max(item_count).max(4_096)
        } else {
            0
        };
        let word_count = if needs_words {
            item_count.max(requested_bytes / 8).max(1)
        } else {
            0
        };
        let floating_count = if needs_floating { item_count.max(1) } else { 0 };
        let floating32_count = if workload.operation == "vector_fused_arithmetic"
            && workload
                .regime
                .get("format")
                .is_some_and(|format| format == "f32")
        {
            item_count.max(1)
        } else {
            0
        };
        let bfloat16_count = if workload.operation == "vector_fused_arithmetic"
            && workload
                .regime
                .get("format")
                .is_some_and(|format| format == "bf16")
        {
            item_count.max(2)
        } else {
            0
        };
        let atomic_count = if needs_atomics
            && workload
                .regime
                .get("contention")
                .is_some_and(|value| value == "independent")
        {
            item_count.max(1)
        } else if needs_atomics {
            1
        } else {
            0
        };
        let bytes = (0..byte_count)
            .map(|index| (index.wrapping_mul(131).wrapping_add(17) & 0xff) as u8)
            .collect::<Vec<_>>();
        let output = if matches!(
            workload.operation.as_str(),
            "sequential_copy" | "gather_scatter" | "numa_local_copy" | "numa_remote_copy"
        ) {
            vec![0u8; byte_count]
        } else {
            Vec::new()
        };
        let words = (0..word_count)
            .map(|index| {
                (index as u64)
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    .rotate_left((index & 63) as u32)
            })
            .collect();
        let floating = (0..floating_count)
            .map(|index| (index % 1_009) as f64 / 1_009.0 + 0.25)
            .collect();
        let floating32 = (0..floating32_count)
            .map(|index| (index % 1_009) as f32 / 1_009.0 + 0.25)
            .collect();
        let bfloat16 = (0..bfloat16_count)
            .map(|index| {
                let bits = ((index % 251) as f32 / 251.0 + 0.25).to_bits();
                let rounding_bias = 0x7fff + ((bits >> 16) & 1);
                ((bits + rounding_bias) >> 16) as u16
            })
            .collect();
        let atomics = (0..atomic_count)
            .map(|index| AtomicU64::new(index as u64))
            .collect();
        let generated_program = if workload.operation == "generated_code_dispatch" {
            Some(GeneratedProgram::compile(
                workload
                    .regime
                    .get("instruction_footprint")
                    .map(String::as_str)
                    .unwrap_or("small"),
                workload.work.operations_per_iteration,
            )?)
        } else {
            None
        };
        Ok(Self {
            operation: workload.operation.clone(),
            regime: workload.regime.clone(),
            bytes,
            output,
            words,
            floating32,
            floating,
            bfloat16,
            atomics,
            atomic_operation_count: item_count,
            generated_program,
            checksum: 0,
        })
    }

    pub(super) fn execute_once(&mut self) -> Result<u64, String> {
        let value = match self.operation.as_str() {
            "scalar_integer" => scalar_integer(&self.words),
            "scalar_floating_point" => scalar_floating_point(&self.floating),
            "out_of_order_control_flow" => out_of_order(&self.words),
            "branch_dispatch" => branch_dispatch(
                &self.bytes,
                self.regime
                    .get("predictability")
                    .map(String::as_str)
                    .unwrap_or("data_dependent"),
            ),
            "vector_fused_arithmetic" => vector_arithmetic(
                &mut self.words,
                &mut self.floating32,
                &mut self.floating,
                &mut self.bfloat16,
                self.regime
                    .get("format")
                    .map(String::as_str)
                    .unwrap_or("f32"),
            )?,
            "blocked_matrix_multiply" => blocked_matrix_multiply(&mut self.floating),
            "bit_population_mix" => bit_population_mix(&self.words),
            "sequential_read" => sequential_read(&self.bytes),
            "sequential_copy" | "numa_local_copy" | "numa_remote_copy" => {
                sequential_copy(&self.bytes, &mut self.output)
            }
            "strided_read" => strided_read(&self.bytes),
            "pointer_chase" => pointer_chase(&self.words),
            "gather_scatter" => gather_scatter(&self.bytes, &mut self.output),
            "generated_code_dispatch" => self
                .generated_program
                .as_ref()
                .ok_or_else(|| "generated program was not constructed".to_string())?
                .execute(&self.words),
            "atomic_fetch_add" => atomic_fetch_add(
                &self.atomics,
                self.atomic_operation_count,
                self.regime
                    .get("contention")
                    .map(String::as_str)
                    .unwrap_or("shared"),
            ),
            unsupported => {
                return Err(format!(
                    "CPU calibrator does not implement operation {unsupported:?}"
                ));
            }
        };
        self.checksum = self.checksum.rotate_left(7) ^ value;
        black_box(self.checksum);
        Ok(value)
    }

    pub(super) fn observed_digest(&self) -> String {
        let mut digest = Sha256::new();
        digest.update(self.operation.as_bytes());
        digest.update(self.checksum.to_le_bytes());
        digest.update((self.bytes.len() as u64).to_le_bytes());
        digest.update((self.words.len() as u64).to_le_bytes());
        digest.update((self.floating.len() as u64).to_le_bytes());
        digest.update((self.floating32.len() as u64).to_le_bytes());
        digest.update((self.bfloat16.len() as u64).to_le_bytes());
        if !self.output.is_empty() {
            digest.update(&self.output);
        }
        for atomic in &self.atomics {
            digest.update(atomic.load(Ordering::Relaxed).to_le_bytes());
        }
        format!("nerve.calibration_output_sha256.v1:{:x}", digest.finalize())
    }

    pub(super) fn generated_artifact(&self) -> Option<(&[u8], &str)> {
        self.generated_program
            .as_ref()
            .map(|program| (program.machine_code.as_slice(), "generated_code"))
    }
}

fn scalar_integer(words: &[u64]) -> u64 {
    let iterations = words.len().max(1);
    let mut a = 0x1234_5678_9abc_def0u64;
    let mut b = 0xfedc_ba98_7654_3210u64;
    let mut c = 0x9e37_79b9_7f4a_7c15u64;
    let mut d = 0xd1b5_4a32_d192_ed03u64;
    for index in 0..iterations {
        let value = words.get(index).copied().unwrap_or(index as u64);
        a = a.wrapping_add(value).rotate_left(13);
        b = b.wrapping_mul(3).wrapping_add(value ^ a);
        c = c.wrapping_sub(value).rotate_right(11);
        d = d.wrapping_mul(5).wrapping_add(value ^ c);
    }
    black_box(a ^ b ^ c ^ d)
}

fn scalar_floating_point(values: &[f64]) -> u64 {
    let mut a = 0.75f64;
    let mut b = 1.25f64;
    let mut c = 1.75f64;
    let mut d = 2.25f64;
    for value in values {
        a = value.mul_add(a, b);
        b = value.mul_add(b, c);
        c = value.mul_add(c, d);
        d = value.mul_add(d, a);
        if a > 1.0e100 {
            a *= 1.0e-100;
            b *= 1.0e-100;
            c *= 1.0e-100;
            d *= 1.0e-100;
        }
    }
    black_box((a + b + c + d).to_bits())
}

fn out_of_order(words: &[u64]) -> u64 {
    let mut lanes = [
        0x1234u64, 0x2345, 0x3456, 0x4567, 0x5678, 0x6789, 0x789a, 0x89ab,
    ];
    for (index, value) in words.iter().enumerate() {
        let lane = index & 7;
        lanes[lane] = lanes[lane]
            .wrapping_mul(0x9e37_79b1)
            .wrapping_add(*value)
            .rotate_left(((lane * 7 + 3) & 63) as u32);
    }
    black_box(lanes.into_iter().fold(0, u64::wrapping_add))
}

fn branch_dispatch(bytes: &[u8], predictability: &str) -> u64 {
    let mut sum = 0u64;
    for (index, byte) in bytes.iter().enumerate() {
        let branch = match predictability {
            "predictable" => true,
            "alternating" => index & 1 == 0,
            _ => byte & 0x80 != 0,
        };
        if black_box(branch) {
            sum = sum.wrapping_add(u64::from(*byte).wrapping_mul(3));
        } else {
            sum = sum.wrapping_sub(u64::from(*byte).wrapping_mul(5));
        }
    }
    black_box(sum)
}

fn vector_arithmetic(
    words: &mut [u64],
    floating32: &mut [f32],
    floating64: &mut [f64],
    bfloat16: &mut [u16],
    format: &str,
) -> Result<u64, String> {
    #[cfg(target_arch = "x86_64")]
    {
        match format {
            "f32"
                if std::is_x86_feature_detected!("avx512f")
                    && std::is_x86_feature_detected!("fma") =>
            {
                return Ok(unsafe { x86_vector_f32(floating32) });
            }
            "f64"
                if std::is_x86_feature_detected!("avx512f")
                    && std::is_x86_feature_detected!("fma") =>
            {
                return Ok(unsafe { x86_vector_f64(floating64) });
            }
            "bf16" if std::is_x86_feature_detected!("avx512bf16") => {
                return Ok(unsafe { x86_vector_bf16(bfloat16) });
            }
            "i8" | "u8" | "i16" | "u16" | "i32" | "u32" | "i64" | "u64"
                if std::is_x86_feature_detected!("avx2") =>
            {
                return Ok(unsafe { x86_vector_integer(words, format) });
            }
            _ => {}
        }
    }
    Err(format!(
        "CPU calibration has no native SIMD implementation for {format:?} on this target"
    ))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn x86_vector_f32(values: &mut [f32]) -> u64 {
    use std::arch::x86_64::*;
    let multiplier = _mm512_set1_ps(1.000_122_1);
    let addend = _mm512_set1_ps(0.000_976_562_5);
    let mut chunks = values.chunks_exact_mut(16);
    for chunk in &mut chunks {
        let value = unsafe { _mm512_loadu_ps(chunk.as_ptr()) };
        let result = _mm512_fmadd_ps(value, multiplier, addend);
        unsafe { _mm512_storeu_ps(chunk.as_mut_ptr(), result) };
    }
    for value in chunks.into_remainder() {
        *value = value.mul_add(1.000_122_1, 0.000_976_562_5);
    }
    black_box(
        values
            .iter()
            .step_by(257)
            .fold(0u64, |sum, value| sum ^ u64::from(value.to_bits())),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn x86_vector_f64(values: &mut [f64]) -> u64 {
    use std::arch::x86_64::*;
    let multiplier = _mm512_set1_pd(1.000_000_119_209_289_6);
    let addend = _mm512_set1_pd(0.000_000_953_674_316_406_25);
    let mut chunks = values.chunks_exact_mut(8);
    for chunk in &mut chunks {
        let value = unsafe { _mm512_loadu_pd(chunk.as_ptr()) };
        let result = _mm512_fmadd_pd(value, multiplier, addend);
        unsafe { _mm512_storeu_pd(chunk.as_mut_ptr(), result) };
    }
    for value in chunks.into_remainder() {
        *value = value.mul_add(1.000_000_119_209_289_6, 0.000_000_953_674_316_406_25);
    }
    black_box(
        values
            .iter()
            .step_by(257)
            .fold(0u64, |sum, value| sum ^ value.to_bits()),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bf16")]
unsafe fn x86_vector_bf16(values: &mut [u16]) -> u64 {
    use std::arch::x86_64::*;
    let mut accumulator = _mm512_setzero_ps();
    for chunk in values.chunks_exact(32) {
        let raw = unsafe { _mm512_loadu_si512(chunk.as_ptr().cast()) };
        let packed = unsafe { std::mem::transmute::<__m512i, __m512bh>(raw) };
        accumulator = _mm512_dpbf16_ps(accumulator, packed, packed);
    }
    let mut output = [0f32; 16];
    unsafe { _mm512_storeu_ps(output.as_mut_ptr(), accumulator) };
    black_box(
        output
            .into_iter()
            .fold(0u64, |sum, value| sum ^ u64::from(value.to_bits())),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn x86_vector_integer(words: &mut [u64], format: &str) -> u64 {
    use std::arch::x86_64::*;
    let byte_length = std::mem::size_of_val(words);
    let vector_count = byte_length / std::mem::size_of::<__m256i>();
    let pointer = words.as_mut_ptr().cast::<__m256i>();
    for index in 0..vector_count {
        let value = unsafe { _mm256_loadu_si256(pointer.add(index)) };
        let result = match format {
            "i8" | "u8" => _mm256_xor_si256(_mm256_add_epi8(value, _mm256_set1_epi8(3)), value),
            "i16" | "u16" => _mm256_add_epi16(
                _mm256_mullo_epi16(value, _mm256_set1_epi16(17)),
                _mm256_set1_epi16(3),
            ),
            "i32" | "u32" => _mm256_add_epi32(
                _mm256_mullo_epi32(value, _mm256_set1_epi32(1_664_525)),
                _mm256_set1_epi32(1_013_904_223),
            ),
            "i64" | "u64" => {
                _mm256_xor_si256(_mm256_add_epi64(value, _mm256_set1_epi64x(17)), value)
            }
            _ => unreachable!("integer SIMD format was validated"),
        };
        unsafe { _mm256_storeu_si256(pointer.add(index), result) };
    }
    black_box(
        words
            .iter()
            .step_by(257)
            .fold(0u64, |sum, value| sum ^ value),
    )
}

fn blocked_matrix_multiply(values: &mut [f64]) -> u64 {
    const SIDE: usize = 64;
    if values.len() < SIDE * SIDE * 3 {
        return scalar_floating_point(values);
    }
    let (left, rest) = values.split_at_mut(SIDE * SIDE);
    let (right, output) = rest.split_at_mut(SIDE * SIDE);
    output.fill(0.0);
    for block_row in (0..SIDE).step_by(8) {
        for block_column in (0..SIDE).step_by(8) {
            for block_inner in (0..SIDE).step_by(8) {
                for row in block_row..block_row + 8 {
                    for inner in block_inner..block_inner + 8 {
                        let left_value = left[row * SIDE + inner];
                        for column in block_column..block_column + 8 {
                            output[row * SIDE + column] = left_value
                                .mul_add(right[inner * SIDE + column], output[row * SIDE + column]);
                        }
                    }
                }
            }
        }
    }
    black_box(output.iter().fold(0u64, |sum, value| sum ^ value.to_bits()))
}

fn bit_population_mix(words: &[u64]) -> u64 {
    black_box(words.iter().fold(0u64, |sum, value| {
        sum.wrapping_add(u64::from(value.count_ones()))
            ^ value.leading_zeros() as u64
            ^ value.trailing_zeros() as u64
            ^ value.reverse_bits().rotate_left(17)
    }))
}

fn sequential_read(bytes: &[u8]) -> u64 {
    black_box(bytes.chunks_exact(8).fold(0u64, |sum, chunk| {
        sum.wrapping_add(u64::from_le_bytes(chunk.try_into().unwrap()))
    }))
}

fn sequential_copy(source: &[u8], destination: &mut [u8]) -> u64 {
    destination.copy_from_slice(source);
    black_box(
        destination
            .iter()
            .step_by(4_096)
            .fold(0u64, |sum, value| sum.wrapping_add(u64::from(*value))),
    )
}

fn strided_read(bytes: &[u8]) -> u64 {
    let mut sum = 0u64;
    for offset in 0..64 {
        for index in (offset..bytes.len()).step_by(64) {
            sum = sum.wrapping_add(u64::from(bytes[index]));
        }
    }
    black_box(sum)
}

fn pointer_chase(words: &[u64]) -> u64 {
    if words.is_empty() {
        return 0;
    }
    let mut index = 0usize;
    let mask = words.len().next_power_of_two() - 1;
    for _ in 0..words.len() {
        index = ((words[index] as usize).wrapping_mul(17).wrapping_add(13)) & mask;
        if index >= words.len() {
            index %= words.len();
        }
    }
    black_box(index as u64)
}

fn gather_scatter(source: &[u8], destination: &mut [u8]) -> u64 {
    if source.is_empty() {
        return 0;
    }
    let mut checksum = 0u64;
    for index in 0..source.len() {
        let target = index.wrapping_mul(17).wrapping_add(13) % source.len();
        destination[target] = source[index];
        checksum = checksum.wrapping_add(u64::from(destination[target]));
    }
    black_box(checksum)
}

fn atomic_fetch_add(atomics: &[AtomicU64], operation_count: usize, contention: &str) -> u64 {
    let worker_count = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(8)
        .min(operation_count.max(1));
    std::thread::scope(|scope| {
        if contention == "shared" {
            let shared = &atomics[0];
            let operations_per_worker = operation_count.div_ceil(worker_count);
            for worker_index in 0..worker_count {
                scope.spawn(move || {
                    let begin = worker_index.saturating_mul(operations_per_worker);
                    let end = begin
                        .saturating_add(operations_per_worker)
                        .min(operation_count);
                    for logical_index in begin..end {
                        shared.fetch_add((logical_index as u64) | 1, Ordering::Relaxed);
                    }
                });
            }
        } else {
            let chunk_size = atomics.len().div_ceil(worker_count);
            for (worker_index, chunk) in atomics.chunks(chunk_size).enumerate() {
                scope.spawn(move || {
                    for (local_index, atomic) in chunk.iter().enumerate() {
                        let logical_index = worker_index
                            .saturating_mul(chunk_size)
                            .saturating_add(local_index);
                        atomic.fetch_add((logical_index as u64) | 1, Ordering::Relaxed);
                    }
                });
            }
        }
    });
    black_box(
        atomics
            .iter()
            .fold(0u64, |sum, atomic| sum ^ atomic.load(Ordering::Relaxed)),
    )
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
struct GeneratedProgram {
    address: *mut libc::c_void,
    allocation_bytes: usize,
    machine_code: Vec<u8>,
    operations_per_call: usize,
    target_operations: u64,
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
impl GeneratedProgram {
    fn compile(footprint: &str, target_operations: u64) -> Result<Self, String> {
        let operations_per_call = match footprint {
            "small" => 32usize,
            "large" => 8_192usize,
            unsupported => {
                return Err(format!(
                    "unsupported generated-code footprint {unsupported:?}"
                ));
            }
        };
        let mut machine_code = Vec::with_capacity(operations_per_call * 8 + 4);
        // System V: move the first u64 argument from RDI to RAX.
        machine_code.extend_from_slice(&[0x48, 0x89, 0xf8]);
        for operation in 0..operations_per_call {
            match operation % 3 {
                0 => {
                    machine_code.extend_from_slice(&[0x48, 0x05]);
                    machine_code.extend_from_slice(
                        &(0x1f12_3bb5u32.wrapping_add(operation as u32)).to_le_bytes(),
                    );
                }
                1 => {
                    machine_code.extend_from_slice(&[0x48, 0x69, 0xc0]);
                    machine_code.extend_from_slice(
                        &(3u32.wrapping_add((operation as u32) & 14)).to_le_bytes(),
                    );
                }
                _ => {
                    machine_code.extend_from_slice(&[
                        0x48,
                        0xc1,
                        0xc0,
                        ((operation % 63) + 1) as u8,
                    ]);
                }
            }
        }
        machine_code.push(0xc3);
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if page_size <= 0 {
            return Err("could not determine executable-memory page size".to_string());
        }
        let page_size = page_size as usize;
        let allocation_bytes = machine_code.len().div_ceil(page_size) * page_size;
        let address = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                allocation_bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if address == libc::MAP_FAILED {
            return Err(format!(
                "could not allocate generated code: {}",
                std::io::Error::last_os_error()
            ));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                machine_code.as_ptr(),
                address.cast::<u8>(),
                machine_code.len(),
            );
        }
        if unsafe { libc::mprotect(address, allocation_bytes, libc::PROT_READ | libc::PROT_EXEC) }
            != 0
        {
            let error = std::io::Error::last_os_error();
            unsafe {
                libc::munmap(address, allocation_bytes);
            }
            return Err(format!("could not make generated code executable: {error}"));
        }
        Ok(Self {
            address,
            allocation_bytes,
            machine_code,
            operations_per_call,
            target_operations,
        })
    }

    fn execute(&self, words: &[u64]) -> u64 {
        let function = unsafe {
            std::mem::transmute::<*mut libc::c_void, unsafe extern "C" fn(u64) -> u64>(self.address)
        };
        let call_count = usize::try_from(
            self.target_operations
                .div_ceil(self.operations_per_call as u64),
        )
        .unwrap_or(usize::MAX);
        let mut checksum = 0u64;
        for call_index in 0..call_count {
            let input = words
                .get(call_index % words.len().max(1))
                .copied()
                .unwrap_or(call_index as u64);
            checksum ^= unsafe { function(input) };
        }
        black_box(checksum)
    }
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
impl Drop for GeneratedProgram {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.address, self.allocation_bytes);
        }
    }
}

#[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
struct GeneratedProgram {
    machine_code: Vec<u8>,
}

#[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
impl GeneratedProgram {
    fn compile(_footprint: &str, _target_operations: u64) -> Result<Self, String> {
        Err("native generated-code calibration is unavailable on this target".to_string())
    }

    fn execute(&self, _words: &[u64]) -> u64 {
        unreachable!("unsupported generated code cannot be constructed")
    }
}
