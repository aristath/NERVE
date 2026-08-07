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
