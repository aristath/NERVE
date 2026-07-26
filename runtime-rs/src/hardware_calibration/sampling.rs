use super::schema::{CalibrationSamplePhase, HardwareCalibrationPolicy, HardwareCalibrationSample};

const T_CRITICAL_95: [f64; 30] = [
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
    2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
    2.052, 2.048, 2.045, 2.042,
];

pub fn collect_adaptive_samples<F>(
    samples: &mut Vec<HardwareCalibrationSample>,
    policy: &HardwareCalibrationPolicy,
    mut measure: F,
) -> Result<(), String>
where
    F: FnMut(CalibrationSamplePhase, usize) -> Result<HardwareCalibrationSample, String>,
{
    while !warmup_complete(samples, policy) {
        samples.push(measure(CalibrationSamplePhase::Warmup, samples.len())?);
    }
    while !steady_complete(samples, policy) {
        samples.push(measure(CalibrationSamplePhase::Steady, samples.len())?);
    }
    Ok(())
}

pub fn warmup_complete(
    samples: &[HardwareCalibrationSample],
    policy: &HardwareCalibrationPolicy,
) -> bool {
    let durations = normalized_durations(samples, CalibrationSamplePhase::Warmup);
    if durations.len() >= policy.maximum_warmup_iterations {
        return true;
    }
    warmup_converged(samples, policy)
}

pub fn warmup_converged(
    samples: &[HardwareCalibrationSample],
    policy: &HardwareCalibrationPolicy,
) -> bool {
    let durations = normalized_durations(samples, CalibrationSamplePhase::Warmup);
    durations.len() >= policy.warmup_iterations
        && durations.len() >= policy.warmup_stability_window
        && warmup_relative_range_ppm(&durations, policy.warmup_stability_window)
            <= policy.maximum_warmup_relative_range_ppm
}

pub fn steady_complete(
    samples: &[HardwareCalibrationSample],
    policy: &HardwareCalibrationPolicy,
) -> bool {
    let durations = normalized_durations(samples, CalibrationSamplePhase::Steady);
    if durations.len() < policy.steady_iterations {
        return false;
    }
    if durations.len() >= policy.maximum_steady_iterations {
        return true;
    }
    relative_ci_width_ppm(&durations, policy.confidence_level_ppm)
        <= policy.maximum_relative_ci_width_ppm.saturating_mul(99) / 100
}

fn normalized_durations(
    samples: &[HardwareCalibrationSample],
    phase: CalibrationSamplePhase,
) -> Vec<u64> {
    samples
        .iter()
        .filter(|sample| sample.phase == phase && sample.valid && sample.iterations > 0)
        .map(|sample| {
            let duration = sample.device_duration_ns.unwrap_or(sample.duration_ns);
            duration
                .saturating_add(sample.iterations / 2)
                .checked_div(sample.iterations)
                .unwrap_or(u64::MAX)
                .max(1)
        })
        .collect()
}

fn warmup_relative_range_ppm(durations: &[u64], window: usize) -> u64 {
    if durations.len() < window || window == 0 {
        return u64::MAX;
    }
    let mut values = durations[durations.len() - window..].to_vec();
    values.sort_unstable();
    let minimum = values[0];
    let maximum = values[values.len() - 1];
    let median = if values.len() % 2 == 0 {
        values[values.len() / 2 - 1].saturating_add(values[values.len() / 2]) / 2
    } else {
        values[values.len() / 2]
    };
    if median == 0 {
        return if maximum == minimum { 0 } else { u64::MAX };
    }
    maximum
        .saturating_sub(minimum)
        .saturating_mul(1_000_000)
        .checked_div(median)
        .unwrap_or(u64::MAX)
}

fn relative_ci_width_ppm(durations: &[u64], confidence_level_ppm: u64) -> u64 {
    if durations.len() < 2 {
        return u64::MAX;
    }
    let count = durations.len() as f64;
    let mean = durations.iter().map(|value| *value as f64).sum::<f64>() / count;
    if mean <= 0.0 {
        return 0;
    }
    let variance = durations
        .iter()
        .map(|value| {
            let difference = *value as f64 - mean;
            difference * difference
        })
        .sum::<f64>()
        / (count - 1.0);
    let standard_deviation = variance.sqrt();
    let critical = t_critical(
        durations.len() - 1,
        confidence_level_ppm as f64 / 1_000_000.0,
    );
    let width = 2.0 * critical * standard_deviation / count.sqrt();
    ((width * 1_000_000.0 / mean).round()).clamp(0.0, u64::MAX as f64) as u64
}

fn t_critical(degrees_of_freedom: usize, confidence_level: f64) -> f64 {
    if degrees_of_freedom == 0 {
        return 0.0;
    }
    if (confidence_level - 0.95).abs() < f64::EPSILON && degrees_of_freedom <= 30 {
        return T_CRITICAL_95[degrees_of_freedom - 1];
    }
    let normal = inverse_normal_cdf(0.5 + confidence_level / 2.0);
    let degrees = degrees_of_freedom as f64;
    let z2 = normal * normal;
    let z3 = z2 * normal;
    let z5 = z3 * z2;
    let z7 = z5 * z2;
    normal
        + (z3 + normal) / (4.0 * degrees)
        + (5.0 * z5 + 16.0 * z3 + 3.0 * normal) / (96.0 * degrees.powi(2))
        + (3.0 * z7 + 19.0 * z5 + 17.0 * z3 - 15.0 * normal) / (384.0 * degrees.powi(3))
}

// Peter J. Acklam's rational approximation. Its error is far below the
// resolution of nanosecond calibration samples.
fn inverse_normal_cdf(probability: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const LOW: f64 = 0.024_25;
    const HIGH: f64 = 1.0 - LOW;
    if probability <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if probability >= 1.0 {
        return f64::INFINITY;
    }
    if probability < LOW {
        let q = (-2.0 * probability.ln()).sqrt();
        return (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    }
    if probability <= HIGH {
        let q = probability - 0.5;
        let r = q * q;
        return (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0);
    }
    let q = (-2.0 * (1.0 - probability).ln()).sqrt();
    -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
        / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> HardwareCalibrationPolicy {
        HardwareCalibrationPolicy {
            warmup_iterations: 5,
            maximum_warmup_iterations: 12,
            warmup_stability_window: 3,
            maximum_warmup_relative_range_ppm: 20_000,
            steady_iterations: 5,
            maximum_steady_iterations: 25,
            minimum_sample_duration_ns: 1,
            sustained_window_duration_ms: 1,
            sustained_window_count: 1,
            confidence_level_ppm: 950_000,
            maximum_relative_ci_width_ppm: 100_000,
        }
    }

    fn sample(
        index: usize,
        phase: CalibrationSamplePhase,
        duration_ns: u64,
    ) -> HardwareCalibrationSample {
        HardwareCalibrationSample {
            sample_index: index,
            phase,
            duration_ns,
            device_duration_ns: None,
            iterations: 1,
            window_index: None,
            thermal_millidegrees_celsius: None,
            valid: true,
        }
    }

    #[test]
    fn warmup_waits_for_a_stable_window_after_the_minimum() {
        let policy = policy();
        let durations = [100, 100, 95, 90, 85, 82, 81, 81];
        let mut samples = Vec::new();
        for (index, duration) in durations.into_iter().enumerate() {
            samples.push(sample(index, CalibrationSamplePhase::Warmup, duration));
            assert_eq!(warmup_complete(&samples, &policy), index == 7);
        }
        assert!(warmup_converged(&samples, &policy));
    }

    #[test]
    fn steady_sampling_absorbs_an_outlier_before_stopping() {
        let mut policy = policy();
        policy.maximum_steady_iterations = 101;
        let mut samples = Vec::new();
        let mut stopping_count = None;
        for index in 0..policy.maximum_steady_iterations {
            let duration = if index == 4 {
                200
            } else {
                99 + (index % 3) as u64
            };
            samples.push(sample(index, CalibrationSamplePhase::Steady, duration));
            if steady_complete(&samples, &policy) {
                stopping_count = Some(samples.len());
                break;
            }
        }
        let stopping_count = stopping_count.expect("stable samples must overcome one outlier");
        assert!(stopping_count > 20);
        assert!(stopping_count < policy.maximum_steady_iterations);
    }

    #[test]
    fn sampling_caps_terminate_without_claiming_warmup_convergence() {
        let policy = policy();
        let warmup = (0..policy.maximum_warmup_iterations)
            .map(|index| {
                sample(
                    index,
                    CalibrationSamplePhase::Warmup,
                    100 + index as u64 * 10,
                )
            })
            .collect::<Vec<_>>();
        assert!(warmup_complete(&warmup, &policy));
        assert!(!warmup_converged(&warmup, &policy));

        let steady = (0..policy.maximum_steady_iterations)
            .map(|index| {
                sample(
                    index,
                    CalibrationSamplePhase::Steady,
                    if index % 2 == 0 { 100 } else { 200 },
                )
            })
            .collect::<Vec<_>>();
        assert!(steady_complete(&steady, &policy));
    }
}
