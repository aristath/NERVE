use std::error::Error;
use std::fmt;
use std::path::PathBuf;

const DEFAULT_PAYLOAD_BYTES: usize = 5 * 1024 * 1024;
const DEFAULT_SAMPLES: usize = 5;
const DEFAULT_MAX_GROUP_SIZE: usize = 3;
const DEFAULT_BENCHMARK_FORMATS: &[&str] = &["bf16", "f32", "fp4", "fp8", "int4"];
const DEFAULT_BENCHMARK_WORKLOADS: &[&str] =
    &["dense_projection", "moe_expert", "router_reduction"];
const MAX_PAYLOAD_BYTES: usize = 64 * 1024 * 1024;
const MAX_SAMPLES: usize = 30;
const MAX_GROUP_SIZE: usize = 3;

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
        max_group_size: usize,
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
    let mut benchmark_workloads = Vec::new();
    let mut pairs = true;
    let mut max_group_size = DEFAULT_MAX_GROUP_SIZE;

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
                max_group_size = parse_usize(
                    &required_value(&arguments, &mut index, "--max-group-size")?,
                    "--max-group-size",
                )?;
            }
            "--format" => {
                benchmark_formats.push(required_value(&arguments, &mut index, "--format")?);
            }
            "--workload" => {
                benchmark_workloads.push(required_value(&arguments, &mut index, "--workload")?);
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
            other => {
                return Err(CliError(format!(
                    "unknown run argument {other:?}\n\n{}",
                    usage()
                )));
            }
        }
        index += 1;
    }

    if payload_bytes == 0 || payload_bytes > MAX_PAYLOAD_BYTES {
        return Err(CliError(format!(
            "--payload-bytes must be between 1 and {MAX_PAYLOAD_BYTES}"
        )));
    }
    if samples == 0 || samples > MAX_SAMPLES {
        return Err(CliError(format!(
            "--samples must be between 1 and {MAX_SAMPLES}"
        )));
    }
    if max_group_size == 0 || max_group_size > MAX_GROUP_SIZE {
        return Err(CliError(format!(
            "--max-group-size must be between 1 and {MAX_GROUP_SIZE}"
        )));
    }
    if benchmark_formats.is_empty() {
        benchmark_formats = DEFAULT_BENCHMARK_FORMATS
            .iter()
            .map(|format| (*format).to_string())
            .collect();
    }
    benchmark_formats.sort();
    benchmark_formats.dedup();
    if benchmark_formats.iter().any(|format| format.is_empty()) {
        return Err(CliError("--format cannot be empty".to_string()));
    }
    if benchmark_workloads.is_empty() {
        benchmark_workloads = DEFAULT_BENCHMARK_WORKLOADS
            .iter()
            .map(|workload| (*workload).to_string())
            .collect();
    }
    benchmark_workloads.sort();
    benchmark_workloads.dedup();
    if benchmark_workloads
        .iter()
        .any(|workload| workload.is_empty())
    {
        return Err(CliError("--workload cannot be empty".to_string()));
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
    "Usage:\n  nerve-gpu-bench list [--json]\n  nerve-gpu-bench run [--output PATH] [--payload-bytes BYTES] [--samples N] [--format FORMAT ...] [--workload WORKLOAD ...] [--max-group-size N] [--include-target ID ...] [--exclude-target ID ...] [--exclude-pci PCI ...] [--exclude-kind KIND ...] [--no-pairs]\n  nerve-gpu-bench summarize --input PATH\n  nerve-gpu-bench validate --input PATH\n"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_run_defaults() {
        let command = parse_args(["run".to_string()]).unwrap();
        assert_eq!(
            command,
            Command::Run {
                output: None,
                payload_bytes: DEFAULT_PAYLOAD_BYTES,
                samples: DEFAULT_SAMPLES,
                benchmark_formats: DEFAULT_BENCHMARK_FORMATS
                    .iter()
                    .map(|format| (*format).to_string())
                    .collect(),
                benchmark_workloads: DEFAULT_BENCHMARK_WORKLOADS
                    .iter()
                    .map(|workload| (*workload).to_string())
                    .collect(),
                include_targets: Vec::new(),
                exclude_targets: Vec::new(),
                exclude_pci: Vec::new(),
                exclude_kinds: Vec::new(),
                pairs: true,
                max_group_size: DEFAULT_MAX_GROUP_SIZE,
            }
        );
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
                ..
            } => {
                assert_eq!(include_targets, ["cpu:host"]);
                assert_eq!(exclude_kinds, ["integrated_gpu"]);
                assert_eq!(
                    benchmark_formats,
                    DEFAULT_BENCHMARK_FORMATS
                        .iter()
                        .map(|format| (*format).to_string())
                        .collect::<Vec<_>>()
                );
                assert_eq!(
                    benchmark_workloads,
                    DEFAULT_BENCHMARK_WORKLOADS
                        .iter()
                        .map(|workload| (*workload).to_string())
                        .collect::<Vec<_>>()
                );
                assert!(!pairs);
                assert_eq!(max_group_size, DEFAULT_MAX_GROUP_SIZE);
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
            Command::Run { max_group_size, .. } => assert_eq!(max_group_size, 2),
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
    fn parses_benchmark_workloads() {
        let command = parse_args([
            "run".to_string(),
            "--workload".to_string(),
            "dense_projection".to_string(),
            "--workload".to_string(),
            "moe_expert".to_string(),
        ])
        .unwrap();
        match command {
            Command::Run {
                benchmark_workloads,
                ..
            } => assert_eq!(benchmark_workloads, ["dense_projection", "moe_expert"]),
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
