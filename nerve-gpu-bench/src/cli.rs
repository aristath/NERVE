use std::error::Error;
use std::fmt;
use std::path::PathBuf;

use crate::model::MAX_PLACEMENT_GROUP_SIZE;

const DEFAULT_PAYLOAD_BYTES: usize = 5 * 1024 * 1024;
const DEFAULT_SAMPLES: usize = 1;
const DEFAULT_BENCHMARK_FORMATS: &[&str] = &[
    "f16", "bf16", "fp8_e4m3", "fp8_e5m2", "fp4", "mxfp4", "int8", "int4", "q8_0", "f32",
];
const PLACEMENT_WORKLOAD: &str = "dense_projection_decode";
const MAX_PAYLOAD_BYTES: usize = 64 * 1024 * 1024;
const MIN_PAYLOAD_BYTES: usize = 4 * 1024;
const MAX_SAMPLES: usize = 30;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Help,
    List {
        json: bool,
    },
    Validate {
        input: PathBuf,
    },
    Summarize {
        input: PathBuf,
    },
    Run {
        output: Option<PathBuf>,
        payload_bytes: usize,
        samples: usize,
        benchmark_formats: Vec<String>,
        benchmark_workloads: Vec<String>,
        include_targets: Vec<String>,
        exclude_targets: Vec<String>,
        exclude_pci: Vec<String>,
        exclude_kinds: Vec<String>,
        pairs: bool,
        max_group_size: Option<usize>,
        dry_plan: bool,
        execute: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CliError(String);

impl fmt::Display for CliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for CliError {}

pub fn parse_args<I>(arguments: I) -> Result<Command, CliError>
where
    I: IntoIterator<Item = String>,
{
    let mut args = arguments.into_iter();
    let Some(command) = args.next() else {
        return Ok(Command::Help);
    };
    match command.as_str() {
        "-h" | "--help" | "help" => Ok(Command::Help),
        "list" => parse_list(args.collect()),
        "summarize" => parse_input_file_command("summarize", args.collect()),
        "validate" => parse_validate(args.collect()),
        "run" => parse_run(args.collect()),
        other => Err(CliError(format!(
            "unknown command {other:?}\n\n{}",
            usage()
        ))),
    }
}

fn parse_validate(arguments: Vec<String>) -> Result<Command, CliError> {
    parse_input_file_command("validate", arguments)
}

fn parse_input_file_command(command: &str, arguments: Vec<String>) -> Result<Command, CliError> {
    let mut input = None;
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "--input" => {
                input = Some(PathBuf::from(required_value(
                    &arguments, &mut index, "--input",
                )?));
            }
            other => {
                return Err(CliError(format!(
                    "unknown {command} argument {other:?}\n\n{}",
                    usage()
                )));
            }
        }
        index += 1;
    }
    let input = input.ok_or_else(|| CliError(format!("{command} requires --input PATH")))?;
    match command {
        "summarize" => Ok(Command::Summarize { input }),
        "validate" => Ok(Command::Validate { input }),
        _ => unreachable!("unsupported input file command"),
    }
}

fn parse_list(arguments: Vec<String>) -> Result<Command, CliError> {
    let mut json = false;
    for argument in arguments {
        match argument.as_str() {
            "--json" => json = true,
            "-h" | "--help" => return Ok(Command::Help),
            other => {
                return Err(CliError(format!(
                    "unknown list argument {other:?}\n\n{}",
                    usage()
                )));
            }
        }
    }
    Ok(Command::List { json })
}

fn parse_run(arguments: Vec<String>) -> Result<Command, CliError> {
    let mut output = None;
    let mut payload_bytes = DEFAULT_PAYLOAD_BYTES;
    let mut samples = DEFAULT_SAMPLES;
    let mut include_targets = Vec::new();
    let mut exclude_targets = Vec::new();
    let mut exclude_pci = Vec::new();
    let mut exclude_kinds = Vec::new();
    let mut benchmark_formats = Vec::new();
    let benchmark_workloads = vec![PLACEMENT_WORKLOAD.to_string()];
    let mut pairs = true;
    let mut max_group_size = None;
    let mut dry_plan = false;
    let mut execute = false;

    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "--output" => {
                output = Some(PathBuf::from(required_value(
                    &arguments, &mut index, "--output",
                )?));
            }
            "--payload-bytes" => {
                payload_bytes = parse_usize(
                    &required_value(&arguments, &mut index, "--payload-bytes")?,
                    "--payload-bytes",
                )?;
            }
            "--samples" => {
                samples = parse_usize(
                    &required_value(&arguments, &mut index, "--samples")?,
                    "--samples",
                )?;
            }
            "--max-group-size" => {
                max_group_size = Some(parse_usize(
                    &required_value(&arguments, &mut index, "--max-group-size")?,
                    "--max-group-size",
                )?);
            }
            "--format" => {
                benchmark_formats.push(required_value(&arguments, &mut index, "--format")?);
            }
            "--include-target" => {
                include_targets.push(required_value(&arguments, &mut index, "--include-target")?);
            }
            "--exclude-target" => {
                exclude_targets.push(required_value(&arguments, &mut index, "--exclude-target")?);
            }
            "--exclude-pci" => {
                exclude_pci.push(required_value(&arguments, &mut index, "--exclude-pci")?);
            }
            "--exclude-kind" => {
                exclude_kinds.push(required_value(&arguments, &mut index, "--exclude-kind")?);
            }
            "--no-pairs" => pairs = false,
            "--dry-plan" => dry_plan = true,
            "--execute" => execute = true,
            other => {
                return Err(CliError(format!(
                    "unknown run argument {other:?}\n\n{}",
                    usage()
                )));
            }
        }
        index += 1;
    }

    if !(MIN_PAYLOAD_BYTES..=MAX_PAYLOAD_BYTES).contains(&payload_bytes) {
        return Err(CliError(format!(
            "--payload-bytes must be between {MIN_PAYLOAD_BYTES} and {MAX_PAYLOAD_BYTES}"
        )));
    }
    if samples == 0 || samples > MAX_SAMPLES {
        return Err(CliError(format!(
            "--samples must be between 1 and {MAX_SAMPLES}"
        )));
    }
    if let Some(max_group_size) = max_group_size {
        if !(1..=MAX_PLACEMENT_GROUP_SIZE).contains(&max_group_size) {
            return Err(CliError(format!(
                "--max-group-size must be between 1 and {MAX_PLACEMENT_GROUP_SIZE}"
            )));
        }
    }
    if benchmark_formats.is_empty() {
        benchmark_formats = DEFAULT_BENCHMARK_FORMATS
            .iter()
            .map(|format| (*format).to_string())
            .collect();
    }
    benchmark_formats.sort();
    benchmark_formats.dedup();
    if let Some(format) = benchmark_formats
        .iter()
        .find(|format| !DEFAULT_BENCHMARK_FORMATS.contains(&format.as_str()))
    {
        return Err(CliError(format!("unsupported --format {format:?}")));
    }
    Ok(Command::Run {
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
    })
}

fn required_value(
    arguments: &[String],
    index: &mut usize,
    option: &str,
) -> Result<String, CliError> {
    *index += 1;
    arguments
        .get(*index)
        .cloned()
        .ok_or_else(|| CliError(format!("{option} requires a value")))
}

fn parse_usize(value: &str, option: &str) -> Result<usize, CliError> {
    value
        .parse::<usize>()
        .map_err(|_| CliError(format!("{option} requires a positive integer")))
}

pub fn usage() -> &'static str {
    "Usage:\n  nerve-gpu-bench list [--json]\n  nerve-gpu-bench run [--output PATH] [--payload-bytes BYTES] [--samples N] [--format FORMAT ...] [--max-group-size N] [--include-target ID ...] [--exclude-target ID ...] [--exclude-pci PCI ...] [--exclude-kind KIND ...] [--no-pairs] [--dry-plan] [--execute]\n  nerve-gpu-bench summarize --input PATH\n  nerve-gpu-bench validate --input PATH\n"
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_benchmark_formats() -> Vec<String> {
        let mut formats = DEFAULT_BENCHMARK_FORMATS
            .iter()
            .map(|format| (*format).to_string())
            .collect::<Vec<_>>();
        formats.sort();
        formats.dedup();
        formats
    }

    #[test]
    fn parses_run_defaults() {
        let command = parse_args(["run".to_string()]).unwrap();
        let default_formats = default_benchmark_formats();
        for expected in ["mxfp4", "fp8_e4m3", "int4", "q8_0", "f32"] {
            assert!(default_formats.contains(&expected.to_string()));
        }
        assert_eq!(
            command,
            Command::Run {
                output: None,
                payload_bytes: DEFAULT_PAYLOAD_BYTES,
                samples: DEFAULT_SAMPLES,
                benchmark_formats: default_formats,
                benchmark_workloads: vec![PLACEMENT_WORKLOAD.to_string()],
                include_targets: Vec::new(),
                exclude_targets: Vec::new(),
                exclude_pci: Vec::new(),
                exclude_kinds: Vec::new(),
                pairs: true,
                max_group_size: None,
                dry_plan: false,
                execute: false,
            }
        );
    }

    #[test]
    fn rejects_payloads_too_small_for_aligned_pair_and_triplet_geometry() {
        let error = parse_args([
            "run".to_string(),
            "--payload-bytes".to_string(),
            (MIN_PAYLOAD_BYTES - 1).to_string(),
        ])
        .unwrap_err();
        assert!(error.to_string().contains(&MIN_PAYLOAD_BYTES.to_string()));
    }

    #[test]
    fn parses_selection_policy() {
        let command = parse_args([
            "run".to_string(),
            "--include-target".to_string(),
            "cpu:host".to_string(),
            "--exclude-kind".to_string(),
            "integrated_gpu".to_string(),
            "--no-pairs".to_string(),
        ])
        .unwrap();
        match command {
            Command::Run {
                include_targets,
                exclude_kinds,
                benchmark_formats,
                benchmark_workloads,
                pairs,
                max_group_size,
                dry_plan,
                execute,
                ..
            } => {
                assert_eq!(include_targets, ["cpu:host"]);
                assert_eq!(exclude_kinds, ["integrated_gpu"]);
                assert_eq!(benchmark_formats, default_benchmark_formats());
                assert_eq!(benchmark_workloads, [PLACEMENT_WORKLOAD.to_string()]);
                assert!(!pairs);
                assert_eq!(max_group_size, None);
                assert!(!dry_plan);
                assert!(!execute);
            }
            _ => panic!("expected run command"),
        }
    }

    #[test]
    fn parses_max_group_size() {
        let command = parse_args([
            "run".to_string(),
            "--max-group-size".to_string(),
            "2".to_string(),
        ])
        .unwrap();
        match command {
            Command::Run { max_group_size, .. } => assert_eq!(max_group_size, Some(2)),
            _ => panic!("expected run command"),
        }
    }

    #[test]
    fn accepts_four_device_group_size() {
        let command = parse_args([
            "run".to_string(),
            "--max-group-size".to_string(),
            "4".to_string(),
        ])
        .unwrap();
        match command {
            Command::Run { max_group_size, .. } => assert_eq!(max_group_size, Some(4)),
            _ => panic!("expected run command"),
        }
    }

    #[test]
    fn rejects_group_sizes_above_four() {
        let error = parse_args([
            "run".to_string(),
            "--max-group-size".to_string(),
            "5".to_string(),
        ])
        .unwrap_err();
        assert!(error.to_string().contains("between 1 and 4"));
    }

    #[test]
    fn rejects_zero_group_size() {
        let error = parse_args([
            "run".to_string(),
            "--max-group-size".to_string(),
            "0".to_string(),
        ])
        .unwrap_err();
        assert!(error.to_string().contains("between 1 and 4"));
    }

    #[test]
    fn parses_dry_plan() {
        let command = parse_args(["run".to_string(), "--dry-plan".to_string()]).unwrap();
        match command {
            Command::Run { dry_plan, .. } => assert!(dry_plan),
            _ => panic!("expected run command"),
        }
    }

    #[test]
    fn parses_execute() {
        let command = parse_args(["run".to_string(), "--execute".to_string()]).unwrap();
        match command {
            Command::Run { execute, .. } => assert!(execute),
            _ => panic!("expected run command"),
        }
    }

    #[test]
    fn parses_benchmark_formats() {
        let command = parse_args([
            "run".to_string(),
            "--format".to_string(),
            "f32".to_string(),
            "--format".to_string(),
            "fp4".to_string(),
        ])
        .unwrap();
        match command {
            Command::Run {
                benchmark_formats, ..
            } => assert_eq!(benchmark_formats, ["f32", "fp4"]),
            _ => panic!("expected run command"),
        }
    }

    #[test]
    fn parses_validate() {
        let command = parse_args([
            "validate".to_string(),
            "--input".to_string(),
            "result.json".to_string(),
        ])
        .unwrap();
        match command {
            Command::Validate { input } => assert_eq!(input, PathBuf::from("result.json")),
            _ => panic!("expected validate command"),
        }
    }

    #[test]
    fn parses_summarize() {
        let command = parse_args([
            "summarize".to_string(),
            "--input".to_string(),
            "result.json".to_string(),
        ])
        .unwrap();
        match command {
            Command::Summarize { input } => assert_eq!(input, PathBuf::from("result.json")),
            _ => panic!("expected summarize command"),
        }
    }
}
