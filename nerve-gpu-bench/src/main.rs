mod benchmark;
mod cli;
mod discovery;
mod model;
mod policy;

use std::error::Error;
use std::fs;
use std::io::{self, Write};

use benchmark::run_benchmarks;
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
            let run = model::parse_benchmark_run_json(&payload)?;
            run.validate_basic()
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
            include_targets,
            exclude_targets,
            exclude_pci,
            exclude_kinds,
            pairs,
            max_group_size,
        } => {
            let targets = discover_targets();
            let policy = model::RunPolicy {
                payload_bytes,
                samples,
                include_targets,
                exclude_targets,
                exclude_pci,
                exclude_kinds,
                pair_measurements: pairs,
                max_group_size,
            };
            let selection = apply_selection_policy(&targets, &policy);
            let run = run_benchmarks(targets, selection, policy);
            let payload = run.to_json_pretty()?;
            if let Some(path) = output {
                fs::write(path, payload.as_bytes())?;
            } else {
                println!("{payload}");
            }
        }
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
        "measurements: single={} pair={} group={}",
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
        "{:<30} {:<16} {:<12} {:<12} {}",
        "TARGET", "KIND", "BACKEND", "VENDOR", "LOCATION"
    )?;
    for target in targets {
        writeln!(
            stdout,
            "{:<30} {:<16} {:<12} {:<12} {}",
            target.stable_target_id,
            target.kind,
            target.backend,
            target.vendor_name.as_deref().unwrap_or("unknown"),
            target
                .pci_address
                .as_deref()
                .or(target.physical_location.as_deref())
                .unwrap_or("host")
        )?;
    }
    Ok(())
}
