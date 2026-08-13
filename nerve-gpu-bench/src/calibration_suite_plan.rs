use std::collections::BTreeSet;
use std::io;

use nerve_runtime::{
    VulkanResidentRuntimeModel, VulkanRuntimePlacementCalibrationTarget,
    VulkanTargetedComponentExecutionPhase, vulkan_runtime_placement_calibration_targets_for_phase,
    vulkan_runtime_placement_transfer_byte_counts,
};

use crate::cli::PackageCalibrationPhase;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CalibrationSuitePlan {
    pub component_cases: Vec<ComponentCalibrationCase>,
    pub boundary_cases: Vec<BoundaryCalibrationCase>,
    pub initial_target_orders: Vec<Vec<String>>,
    pub maximum_group_size: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComponentCalibrationCase {
    pub phase: PackageCalibrationPhase,
    pub target: VulkanRuntimePlacementCalibrationTarget,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundaryCalibrationCase {
    pub phase: PackageCalibrationPhase,
    pub source_target_id: String,
    pub destination_target_id: String,
}

pub fn plan_calibration_suite(
    runtime_model: &VulkanResidentRuntimeModel,
    target_ids: &[String],
    prefill_widths: &[usize],
    maximum_group_size: Option<usize>,
) -> Result<CalibrationSuitePlan, io::Error> {
    validate_distinct_nonempty_target_ids(target_ids)?;
    let maximum_group_size = maximum_group_size
        .unwrap_or(target_ids.len())
        .min(target_ids.len());
    if maximum_group_size == 0 {
        return Err(invalid_input(
            "calibration suite requires a positive maximum group size",
        ));
    }
    let phases = calibration_phases(prefill_widths)?;
    let mut supported_phases = Vec::new();
    let mut component_cases = Vec::new();
    for phase in &phases {
        let runtime_phase = targeted_phase(*phase);
        let targets =
            vulkan_runtime_placement_calibration_targets_for_phase(runtime_model, runtime_phase)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?;
        if targets.is_empty() {
            continue;
        }
        supported_phases.push(*phase);
        component_cases.extend(targets.into_iter().map(|target| ComponentCalibrationCase {
            phase: *phase,
            target,
        }));
    }

    let boundary_cases = if vulkan_runtime_placement_transfer_byte_counts(runtime_model)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?
        .is_empty()
    {
        Vec::new()
    } else {
        supported_phases
            .iter()
            .flat_map(|phase| {
                ordered_target_pairs(target_ids)
                    .into_iter()
                    .map(move |pair| BoundaryCalibrationCase {
                        phase: *phase,
                        source_target_id: pair[0].clone(),
                        destination_target_id: pair[1].clone(),
                    })
            })
            .collect()
    };

    Ok(CalibrationSuitePlan {
        component_cases,
        boundary_cases,
        initial_target_orders: initial_target_orders(target_ids, maximum_group_size),
        maximum_group_size,
    })
}

pub fn expand_target_orders(
    promising_prefixes: &[Vec<String>],
    eligible_target_ids: &[String],
    maximum_group_size: usize,
) -> Result<Vec<Vec<String>>, io::Error> {
    validate_distinct_nonempty_target_ids(eligible_target_ids)?;
    if maximum_group_size == 0 || maximum_group_size > eligible_target_ids.len() {
        return Err(invalid_input(
            "calibration expansion requires a valid maximum group size",
        ));
    }
    let eligible = eligible_target_ids.iter().collect::<BTreeSet<_>>();
    let mut expanded = BTreeSet::new();
    for prefix in promising_prefixes {
        if prefix.len() < 2 || prefix.len() >= maximum_group_size {
            continue;
        }
        let mut seen = BTreeSet::new();
        if prefix
            .iter()
            .any(|target| !eligible.contains(target) || !seen.insert(target))
        {
            return Err(invalid_input(
                "calibration expansion received an invalid candidate prefix",
            ));
        }
        for target in eligible_target_ids {
            if !seen.contains(target) {
                let mut candidate = prefix.clone();
                candidate.push(target.clone());
                expanded.insert(candidate);
            }
        }
    }
    Ok(expanded.into_iter().collect())
}

fn calibration_phases(prefill_widths: &[usize]) -> Result<Vec<PackageCalibrationPhase>, io::Error> {
    if prefill_widths.contains(&0) {
        return Err(invalid_input(
            "calibration suite prefill widths must be positive",
        ));
    }
    let mut widths = prefill_widths.to_vec();
    widths.sort_unstable();
    widths.dedup();
    let mut phases = vec![PackageCalibrationPhase::Decode];
    phases.extend(widths.into_iter().map(|activation_batch_width| {
        PackageCalibrationPhase::Prefill {
            activation_batch_width,
        }
    }));
    Ok(phases)
}

fn targeted_phase(phase: PackageCalibrationPhase) -> VulkanTargetedComponentExecutionPhase {
    match phase {
        PackageCalibrationPhase::Decode => VulkanTargetedComponentExecutionPhase::Decode,
        PackageCalibrationPhase::Prefill {
            activation_batch_width,
        } => VulkanTargetedComponentExecutionPhase::Prefill {
            activation_batch_width,
        },
    }
}

fn initial_target_orders(target_ids: &[String], maximum_group_size: usize) -> Vec<Vec<String>> {
    let mut orders = target_ids
        .iter()
        .map(|target| vec![target.clone()])
        .collect::<Vec<_>>();
    if maximum_group_size >= 2 {
        orders.extend(ordered_target_pairs(target_ids));
    }
    orders
}

fn ordered_target_pairs(target_ids: &[String]) -> Vec<Vec<String>> {
    target_ids
        .iter()
        .enumerate()
        .flat_map(|(source_index, source)| {
            target_ids
                .iter()
                .enumerate()
                .filter(move |(target_index, _)| *target_index != source_index)
                .map(move |(_, target)| vec![source.clone(), target.clone()])
        })
        .collect()
}

fn validate_distinct_nonempty_target_ids(target_ids: &[String]) -> Result<(), io::Error> {
    let mut distinct = BTreeSet::new();
    if target_ids.is_empty()
        || target_ids
            .iter()
            .any(|target| target.is_empty() || !distinct.insert(target))
    {
        return Err(invalid_input(
            "calibration suite requires distinct nonempty target identities",
        ));
    }
    Ok(())
}

fn invalid_input(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_coverage_contains_every_single_and_directed_pair() {
        let targets = ["a", "b", "c"].map(str::to_string).to_vec();
        assert_eq!(
            initial_target_orders(&targets, targets.len()),
            vec![
                vec!["a"],
                vec!["b"],
                vec!["c"],
                vec!["a", "b"],
                vec!["a", "c"],
                vec!["b", "a"],
                vec!["b", "c"],
                vec!["c", "a"],
                vec!["c", "b"],
            ]
            .into_iter()
            .map(|order| order.into_iter().map(str::to_string).collect())
            .collect::<Vec<Vec<String>>>(),
        );
    }

    #[test]
    fn staged_expansion_has_no_architectural_device_count_limit() {
        let eligible = (0..7)
            .map(|index| format!("gpu-{index}"))
            .collect::<Vec<_>>();
        let mut frontier = vec![vec!["gpu-0".to_string(), "gpu-1".to_string()]];
        for expected_width in 3..=eligible.len() {
            let expanded = expand_target_orders(&frontier, &eligible, eligible.len()).unwrap();
            assert!(
                expanded
                    .iter()
                    .all(|candidate| candidate.len() == expected_width)
            );
            frontier = vec![expanded[0].clone()];
        }
        assert_eq!(frontier[0].len(), 7);
        assert!(
            expand_target_orders(&frontier, &eligible, eligible.len())
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn expansion_rejects_unknown_repeated_and_invalid_limits() {
        let eligible = ["a", "b", "c"].map(str::to_string).to_vec();
        assert!(expand_target_orders(&[vec!["a".into(), "x".into()]], &eligible, 3).is_err());
        assert!(expand_target_orders(&[vec!["a".into(), "a".into()]], &eligible, 3).is_err());
        assert!(expand_target_orders(&[], &eligible, 0).is_err());
        assert!(expand_target_orders(&[], &eligible, 4).is_err());
    }

    #[test]
    fn phases_are_canonical_and_reject_zero_width() {
        assert_eq!(
            calibration_phases(&[64, 8, 64]).unwrap(),
            vec![
                PackageCalibrationPhase::Decode,
                PackageCalibrationPhase::Prefill {
                    activation_batch_width: 8,
                },
                PackageCalibrationPhase::Prefill {
                    activation_batch_width: 64,
                },
            ],
        );
        assert!(calibration_phases(&[0]).is_err());
    }
}
