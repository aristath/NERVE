mod benchmark;
mod cli;
mod discovery;
mod model;
mod policy;
mod vulkan_exec;
mod vulkan_probe;

use std::error::Error;
use std::fs;
use std::io::{self, Write};

use benchmark::{plan_benchmarks, run_benchmarks};
use cli::{Command, parse_args};
use discovery::discover_targets;
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
            let run = model::parse_benchmark_run_json(&payload)?;
            run.validate_basic()
                .map_err(|message| format!("invalid benchmark JSON: {message}"))?;
            print_run_summary(&run)?;
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
            let policy = model::RunPolicy {
                payload_bytes,
                samples,
                benchmark_formats,
                benchmark_workloads,
                include_targets,
                exclude_targets,
                exclude_pci,
                exclude_kinds,
                pair_measurements: pairs,
                max_group_size,
                execute,
            };
            let selection = apply_selection_policy(&targets, &policy);
            let payload = if dry_plan {
                plan_benchmarks(targets, selection, policy).to_json_pretty()?
            } else {
                run_benchmarks(targets, selection, policy).to_json_pretty()?
            };
            if let Some(path) = output {
                fs::write(path, payload.as_bytes())?;
            } else {
                println!("{payload}");
            }
        }
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
        Some(model::PLAN_SCHEMA) => {
            let plan = model::parse_benchmark_plan_json(payload)?;
            plan.validate_basic()?;
        }
        Some(schema) => {
            return Err(format!("unsupported benchmark JSON schema {schema:?}").into());
        }
        None => return Err("missing benchmark JSON schema".into()),
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
