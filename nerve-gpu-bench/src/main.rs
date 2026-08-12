mod benchmark;
mod boundary_calibration;
mod calibration_device_state;
mod calibration_package;
mod calibration_suite;
mod calibration_suite_plan;
mod catalog_merge;
mod cli;
mod discovery;
mod load_wave_calibration;
mod model;
mod output;
mod package_calibration;
mod policy;
mod region_calibration;
mod selected_resource_calibration;
mod vulkan_exec;
mod vulkan_features;
mod vulkan_probe;

use std::error::Error;
use std::fs;
use std::io::{self, Write};

use benchmark::{plan_benchmarks, run_benchmarks, validate_execution_coverage};
use cli::{Command, parse_args};
use discovery::discover_targets;
use nerve_execution_contracts::{PHYSICAL_EXECUTION_CONTRACT_SCHEMA, PhysicalExecutionContract};
use nerve_runtime::{
    VULKAN_PLACEMENT_CALIBRATION_CATALOG_SCHEMA, VulkanPlacementCalibrationCatalog,
};
use policy::apply_selection_policy;

fn main() {
    if let Err(error) = run() {
        eprintln!("nerve-gpu-bench: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let command = parse_args(std::env::args().skip(1))?;
    match command {
        Command::Help => {
            print!("{}", cli::usage());
        }
        Command::List { json } => {
            let targets = discover_targets();
            if json {
                println!("{}", model::targets_to_json(&targets)?);
            } else {
                print_target_table(&targets)?;
            }
        }
        Command::Validate { input } => {
            let payload = fs::read_to_string(&input)?;
            validate_json_document(&payload)
                .map_err(|message| format!("invalid benchmark JSON: {message}"))?;
            println!("valid {}", input.display());
        }
        Command::Summarize { input } => {
            let payload = fs::read_to_string(&input)?;
            print_json_summary(&payload)?;
        }
        Command::CalibrateSuite {
            package,
            mut target_ids,
            prefill_widths,
            maximum_group_size,
            runtime,
            output,
        } => {
            if target_ids.is_empty() {
                target_ids = executable_vulkan_target_ids(discover_targets());
            }
            calibration_suite::run_calibration_suite(
                &package,
                &target_ids,
                &prefill_widths,
                maximum_group_size,
                runtime,
                &output,
            )?;
        }
        Command::CalibratePackage {
            package,
            component,
            phase,
            target_ids,
            runtime,
            output,
        } => {
            package_calibration::run_package_calibration(
                &package,
                &component,
                phase,
                &target_ids,
                runtime,
                &output,
            )?;
        }
        Command::CalibrateBoundaries {
            package,
            phase,
            source_id,
            target_id,
            runtime,
            output,
        } => {
            boundary_calibration::run_boundary_calibration(
                &package, phase, &source_id, &target_id, runtime, &output,
            )?;
        }
        Command::CalibrateLoadWave {
            package,
            component,
            selector,
            phase,
            resource_indices,
            target_id,
            runtime,
            output,
        } => {
            load_wave_calibration::run_load_wave_calibration(
                &package,
                &component,
                &selector,
                phase,
                &resource_indices,
                &target_id,
                runtime,
                &output,
            )?;
        }
        Command::MergeCatalogs { inputs, output } => {
            catalog_merge::merge_catalog_files(&inputs, &output)?;
        }
        Command::Run {
            output,
            payload_bytes,
            samples,
            benchmark_formats,
            benchmark_workloads,
            include_targets,
            exclude_targets,
            exclude_pci,
            exclude_kinds,
            pairs,
            max_group_size,
            dry_plan,
            execute,
        } => {
            let targets = discover_targets();
            let mut policy = model::RunPolicy {
                payload_bytes,
                samples,
                benchmark_formats,
                benchmark_workloads,
                include_targets,
                exclude_targets,
                exclude_pci,
                exclude_kinds,
                pair_measurements: pairs,
                max_group_size: max_group_size.unwrap_or(1),
                execute,
            };
            let selection = apply_selection_policy(&targets, &policy);
            policy.max_group_size =
                resolve_max_group_size(max_group_size, selection.selected_target_ids.len());
            let payload = if dry_plan {
                plan_benchmarks(targets, selection, policy).to_json_pretty()?
            } else {
                let run = run_benchmarks(targets, selection, policy);
                validate_execution_coverage(&run)?;
                let placement = run.to_placement_benchmark()?;
                placement.validate_basic()?;
                placement.to_json()?
            };
            if let Some(path) = output {
                output::write_atomic(&path, payload.as_bytes())?;
            } else {
                println!("{payload}");
            }
        }
    }
    Ok(())
}

fn executable_vulkan_target_ids(targets: Vec<model::Target>) -> Vec<String> {
    targets
        .into_iter()
        .filter(|target| {
            target.backend == "vulkan"
                && target.vulkan.is_some()
                && target.stable_target_id.starts_with("vulkan-uuid:")
        })
        .map(|target| target.stable_target_id)
        .collect()
}

fn resolve_max_group_size(requested: Option<usize>, selected_target_count: usize) -> usize {
    requested
        .unwrap_or(selected_target_count)
        .min(selected_target_count)
        .max(1)
}

fn print_json_summary(payload: &str) -> Result<(), Box<dyn Error>> {
    let value = serde_json::from_str::<serde_json::Value>(payload)?;
    match value.get("schema").and_then(serde_json::Value::as_str) {
        Some(model::RUN_SCHEMA) => {
            let run = model::parse_benchmark_run_json(payload)?;
            run.validate_basic()
                .map_err(|message| format!("invalid benchmark JSON: {message}"))?;
            print_run_summary(&run)?;
        }
        Some(model::PLACEMENT_SCHEMA) => {
            let run = model::parse_placement_benchmark_json(payload)?;
            run.validate_basic()
                .map_err(|message| format!("invalid benchmark JSON: {message}"))?;
            print_placement_summary(&run)?;
        }
        Some(VULKAN_PLACEMENT_CALIBRATION_CATALOG_SCHEMA) => {
            let catalog = VulkanPlacementCalibrationCatalog::from_json_slice(payload.as_bytes())?;
            print_exact_calibration_summary(&catalog)?;
        }
        Some(schema) => {
            return Err(format!("unsupported benchmark JSON schema {schema:?}").into());
        }
        None => return Err("missing benchmark JSON schema".into()),
    }
    Ok(())
}

fn validate_json_document(payload: &str) -> Result<(), Box<dyn Error>> {
    let value = serde_json::from_str::<serde_json::Value>(payload)?;
    match value.get("schema").and_then(serde_json::Value::as_str) {
        Some(model::RUN_SCHEMA) => {
            let run = model::parse_benchmark_run_json(payload)?;
            run.validate_basic()?;
        }
        Some(model::PLACEMENT_SCHEMA) => {
            let run = model::parse_placement_benchmark_json(payload)?;
            run.validate_basic()?;
        }
        Some(model::PLAN_SCHEMA) => {
            let plan = model::parse_benchmark_plan_json(payload)?;
            plan.validate_basic()?;
        }
        Some(PHYSICAL_EXECUTION_CONTRACT_SCHEMA) => {
            let contract = serde_json::from_str::<PhysicalExecutionContract>(payload)?;
            contract.validate()?;
        }
        Some(VULKAN_PLACEMENT_CALIBRATION_CATALOG_SCHEMA) => {
            VulkanPlacementCalibrationCatalog::from_json_slice(payload.as_bytes())?;
        }
        Some(schema) => {
            return Err(format!("unsupported benchmark JSON schema {schema:?}").into());
        }
        None => return Err("missing benchmark JSON schema".into()),
    }
    Ok(())
}

fn print_exact_calibration_summary(
    catalog: &VulkanPlacementCalibrationCatalog,
) -> Result<(), Box<dyn Error>> {
    let mut stdout = io::stdout().lock();
    writeln!(
        stdout,
        "schema: {}",
        VULKAN_PLACEMENT_CALIBRATION_CATALOG_SCHEMA
    )?;
    writeln!(
        stdout,
        "canonical_references: {}",
        catalog.reference_count()
    )?;
    writeln!(
        stdout,
        "output_valid_observations: {}",
        catalog.observation_count()
    )?;
    Ok(())
}

fn print_placement_summary(run: &model::PlacementBenchmark) -> Result<(), Box<dyn Error>> {
    let mut stdout = io::stdout().lock();
    writeln!(stdout, "schema: {}", run.schema)?;
    writeln!(stdout, "payload_bytes: {}", run.payload_bytes)?;
    writeln!(stdout, "formats: {}", run.formats.len())?;
    for (format, ranking) in &run.formats {
        let combination_count = run.combinations.get(format).map_or(0, Vec::len);
        writeln!(
            stdout,
            "  {format}: placements={} serial={} combinations={combination_count}",
            ranking.placements.len(),
            ranking.serial.len()
        )?;
    }
    Ok(())
}

fn print_run_summary(run: &model::BenchmarkRun) -> Result<(), Box<dyn Error>> {
    let summary = run.summary();
    let mut stdout = io::stdout().lock();
    writeln!(stdout, "schema: {}", run.schema)?;
    writeln!(stdout, "payload_bytes: {}", run.policy.payload_bytes)?;
    writeln!(stdout, "samples: {}", run.policy.samples)?;
    writeln!(
        stdout,
        "targets: discovered={} selected={} skipped={}",
        summary.discovered_target_count,
        summary.selected_target_count,
        summary.skipped_target_count
    )?;
    writeln!(
        stdout,
        "measurements: comparison_sets={} single={} pair={} group={}",
        summary.comparison_set_count,
        summary.single_measurement_count,
        summary.pair_measurement_count,
        summary.group_measurement_count
    )?;
    writeln!(
        stdout,
        "statuses: completed={} unmeasured={} failed={} unsupported={} skipped={}",
        summary.completed_count,
        summary.unmeasured_count,
        summary.failed_count,
        summary.unsupported_count,
        summary.skipped_count
    )?;
    if !summary.strategy_statuses.is_empty() {
        writeln!(stdout, "placement_strategies:")?;
        for strategy in &summary.strategy_statuses {
            writeln!(
                stdout,
                "  {} / {} / {} / {}: completed={} unmeasured={} failed={} unsupported={} skipped={}",
                strategy.comparison_group,
                strategy.workload_class,
                strategy.placement_strategy,
                strategy.format,
                strategy.completed_count,
                strategy.unmeasured_count,
                strategy.failed_count,
                strategy.unsupported_count,
                strategy.skipped_count
            )?;
        }
    }
    if !summary.candidate_statuses.is_empty() {
        writeln!(stdout, "comparison_candidates:")?;
        for candidate in &summary.candidate_statuses {
            writeln!(
                stdout,
                "  {} / {}: workload={} format={} strategy={} kind={} status={} matches={} best_median_ns={}",
                candidate.comparison_id,
                candidate.candidate_id,
                candidate.workload_class,
                candidate.format,
                candidate.placement_strategy,
                candidate.measurement_kind,
                candidate.status,
                candidate.matched_measurement_count,
                candidate
                    .best_median_duration_ns
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_string())
            )?;
        }
    }
    if !summary.coverage_warnings.is_empty() {
        writeln!(stdout, "coverage_warnings:")?;
        for warning in &summary.coverage_warnings {
            writeln!(stdout, "  {warning}")?;
        }
    }
    if !run.selected_target_ids.is_empty() {
        writeln!(stdout, "selected_targets:")?;
        for target_id in &run.selected_target_ids {
            writeln!(stdout, "  {target_id}")?;
        }
    }
    Ok(())
}

fn print_target_table(targets: &[model::Target]) -> Result<(), Box<dyn Error>> {
    let mut stdout = io::stdout().lock();
    writeln!(
        stdout,
        "{:<30} {:<16} {:<12} {:<12} {:<18} {}",
        "TARGET", "KIND", "BACKEND", "VENDOR", "LINK", "LOCATION"
    )?;
    for target in targets {
        writeln!(
            stdout,
            "{:<30} {:<16} {:<12} {:<12} {:<18} {}",
            target.stable_target_id,
            target.kind,
            target.backend,
            target.vendor_name.as_deref().unwrap_or("unknown"),
            format_link(target),
            target
                .pci_address
                .as_deref()
                .or(target.physical_location.as_deref())
                .unwrap_or("host")
        )?;
    }
    Ok(())
}

fn format_link(target: &model::Target) -> String {
    let Some(link) = &target.pci_link else {
        return "-".to_string();
    };
    match (
        link.current_link_speed.as_deref(),
        link.current_link_width,
        link.current_one_way_bytes_per_second,
    ) {
        (Some(speed), Some(width), Some(bytes_per_second)) => {
            let gib_per_second = bytes_per_second as f64 / 1_073_741_824.0;
            format!("{speed} x{width} {gib_per_second:.1}GiB/s")
        }
        (Some(speed), Some(width), None) => format!("{speed} x{width}"),
        (Some(speed), None, _) => speed.to_string(),
        (None, Some(width), _) => format!("x{width}"),
        _ => "-".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::{resolve_max_group_size, validate_json_document};
    use nerve_runtime::VulkanPlacementCalibrationCatalog;

    #[test]
    fn omitted_group_cap_uses_every_selected_target() {
        assert_eq!(resolve_max_group_size(None, 8), 8);
        assert_eq!(resolve_max_group_size(None, 3), 3);
    }

    #[test]
    fn explicit_group_cap_is_bounded_by_selected_targets() {
        assert_eq!(resolve_max_group_size(Some(4), 8), 4);
        assert_eq!(resolve_max_group_size(Some(12), 8), 8);
    }

    #[test]
    fn benchmark_accepts_the_shared_physical_execution_contract() {
        validate_json_document(include_str!(
            "../../execution-contracts/fixtures/tensor_parallel_projection.json"
        ))
        .unwrap();
    }

    #[test]
    fn benchmark_accepts_the_runtime_exact_calibration_catalog() {
        let payload = VulkanPlacementCalibrationCatalog::default()
            .to_json_bytes()
            .unwrap();
        validate_json_document(std::str::from_utf8(&payload).unwrap()).unwrap();
    }

    #[test]
    fn benchmark_rejects_a_stale_runtime_calibration_schema() {
        let payload = VulkanPlacementCalibrationCatalog::default()
            .to_json_bytes()
            .unwrap();
        let mut document = serde_json::from_slice::<serde_json::Value>(&payload).unwrap();
        document["schema"] =
            serde_json::Value::String("nerve.vulkan_placement_calibration_catalog.v1".to_string());

        assert!(
            validate_json_document(&serde_json::to_string(&document).unwrap())
                .unwrap_err()
                .to_string()
                .contains("unsupported")
        );
    }
}
