use std::error::Error;
use std::fmt;
use std::path::PathBuf;

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
    CalibratePackage {
        package: PathBuf,
        component: String,
        phase: PackageCalibrationPhase,
        target_ids: Vec<String>,
        output: PathBuf,
    },
    CalibrateBoundaries {
        package: PathBuf,
        phase: PackageCalibrationPhase,
        source_id: String,
        target_id: String,
        output: PathBuf,
    },
    MergeCatalogs {
        inputs: Vec<PathBuf>,
        output: PathBuf,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackageCalibrationPhase {
    Decode,
    Prefill { activation_batch_width: usize },
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
        "calibrate-package" => parse_calibrate_package(args.collect()),
        "calibrate-boundaries" => parse_calibrate_boundaries(args.collect()),
        "merge-catalogs" => parse_merge_catalogs(args.collect()),
        "run" => parse_run(args.collect()),
        other => Err(CliError(format!(
            "unknown command {other:?}\n\n{}",
            usage()
        ))),
    }
}

fn parse_calibrate_boundaries(arguments: Vec<String>) -> Result<Command, CliError> {
    let mut package = None;
    let mut phase = None;
    let mut activation_batch_width = None;
    let mut source_id = None;
    let mut target_id = None;
    let mut output = None;
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "--package" => set_once_path(&mut package, &arguments, &mut index, "--package")?,
            "--phase" => set_once_string(&mut phase, &arguments, &mut index, "--phase")?,
            "--batch-width" => {
                let value = parse_usize(
                    &required_value(&arguments, &mut index, "--batch-width")?,
                    "--batch-width",
                )?;
                if activation_batch_width.replace(value).is_some() {
                    return Err(CliError(
                        "--batch-width may only be specified once".to_string(),
                    ));
                }
            }
            "--source" => set_once_string(&mut source_id, &arguments, &mut index, "--source")?,
            "--target" => set_once_string(&mut target_id, &arguments, &mut index, "--target")?,
            "--output" => set_once_path(&mut output, &arguments, &mut index, "--output")?,
            other => {
                return Err(CliError(format!(
                    "unknown calibrate-boundaries argument {other:?}\n\n{}",
                    usage()
                )));
            }
        }
        index += 1;
    }
    let package = package
        .ok_or_else(|| CliError("calibrate-boundaries requires --package PATH".to_string()))?;
    let phase =
        parse_package_calibration_phase(phase, activation_batch_width, "calibrate-boundaries")?;
    let source_id = source_id
        .filter(|id| is_canonical_vulkan_target_id(id))
        .ok_or_else(|| {
            CliError("calibrate-boundaries requires canonical --source vulkan-uuid:ID".to_string())
        })?;
    let target_id = target_id
        .filter(|id| is_canonical_vulkan_target_id(id))
        .ok_or_else(|| {
            CliError("calibrate-boundaries requires canonical --target vulkan-uuid:ID".to_string())
        })?;
    if source_id == target_id {
        return Err(CliError(
            "calibrate-boundaries requires distinct source and target identities".to_string(),
        ));
    }
    let output = output
        .ok_or_else(|| CliError("calibrate-boundaries requires --output PATH".to_string()))?;
    Ok(Command::CalibrateBoundaries {
        package,
        phase,
        source_id,
        target_id,
        output,
    })
}

fn set_once_path(
    slot: &mut Option<PathBuf>,
    arguments: &[String],
    index: &mut usize,
    option: &str,
) -> Result<(), CliError> {
    let value = PathBuf::from(required_value(arguments, index, option)?);
    if slot.replace(value).is_some() {
        return Err(CliError(format!("{option} may only be specified once")));
    }
    Ok(())
}

fn set_once_string(
    slot: &mut Option<String>,
    arguments: &[String],
    index: &mut usize,
    option: &str,
) -> Result<(), CliError> {
    let value = required_value(arguments, index, option)?;
    if slot.replace(value).is_some() {
        return Err(CliError(format!("{option} may only be specified once")));
    }
    Ok(())
}

fn parse_merge_catalogs(arguments: Vec<String>) -> Result<Command, CliError> {
    let mut inputs = Vec::new();
    let mut output = None;
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "--input" => inputs.push(PathBuf::from(required_value(
                &arguments, &mut index, "--input",
            )?)),
            "--output" => {
                let value = PathBuf::from(required_value(&arguments, &mut index, "--output")?);
                if output.replace(value).is_some() {
                    return Err(CliError("--output may only be specified once".to_string()));
                }
            }
            other => {
                return Err(CliError(format!(
                    "unknown merge-catalogs argument {other:?}\n\n{}",
                    usage()
                )));
            }
        }
        index += 1;
    }
    if inputs.len() < 2 {
        return Err(CliError(
            "merge-catalogs requires at least two --input paths".to_string(),
        ));
    }
    let mut distinct_inputs = inputs.clone();
    distinct_inputs.sort();
    distinct_inputs.dedup();
    if distinct_inputs.len() != inputs.len() {
        return Err(CliError(
            "merge-catalogs requires distinct input paths".to_string(),
        ));
    }
    let output =
        output.ok_or_else(|| CliError("merge-catalogs requires --output PATH".to_string()))?;
    Ok(Command::MergeCatalogs { inputs, output })
}

fn parse_calibrate_package(arguments: Vec<String>) -> Result<Command, CliError> {
    let mut package = None;
    let mut component = None;
    let mut phase = None;
    let mut activation_batch_width = None;
    let mut target_ids = Vec::new();
    let mut output = None;
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "--package" => {
                let value = PathBuf::from(required_value(&arguments, &mut index, "--package")?);
                if package.replace(value).is_some() {
                    return Err(CliError("--package may only be specified once".to_string()));
                }
            }
            "--component" => {
                let value = required_value(&arguments, &mut index, "--component")?;
                if component.replace(value).is_some() {
                    return Err(CliError(
                        "--component may only be specified once".to_string(),
                    ));
                }
            }
            "--phase" => {
                let value = required_value(&arguments, &mut index, "--phase")?;
                if phase.replace(value).is_some() {
                    return Err(CliError("--phase may only be specified once".to_string()));
                }
            }
            "--batch-width" => {
                let value = parse_usize(
                    &required_value(&arguments, &mut index, "--batch-width")?,
                    "--batch-width",
                )?;
                if activation_batch_width.replace(value).is_some() {
                    return Err(CliError(
                        "--batch-width may only be specified once".to_string(),
                    ));
                }
            }
            "--target" => {
                target_ids.push(required_value(&arguments, &mut index, "--target")?);
            }
            "--output" => {
                let value = PathBuf::from(required_value(&arguments, &mut index, "--output")?);
                if output.replace(value).is_some() {
                    return Err(CliError("--output may only be specified once".to_string()));
                }
            }
            other => {
                return Err(CliError(format!(
                    "unknown calibrate-package argument {other:?}\n\n{}",
                    usage()
                )));
            }
        }
        index += 1;
    }
    let package =
        package.ok_or_else(|| CliError("calibrate-package requires --package PATH".to_string()))?;
    let component = component
        .filter(|component| !component.is_empty())
        .ok_or_else(|| CliError("calibrate-package requires --component ID".to_string()))?;
    let phase =
        parse_package_calibration_phase(phase, activation_batch_width, "calibrate-package")?;
    if target_ids.is_empty() {
        return Err(CliError(
            "calibrate-package requires at least one ordered --target vulkan-uuid:ID".to_string(),
        ));
    }
    if let Some(target_id) = target_ids
        .iter()
        .find(|target_id| !is_canonical_vulkan_target_id(target_id))
    {
        return Err(CliError(format!(
            "calibrate-package target {target_id:?} is not a canonical vulkan-uuid identity"
        )));
    }
    let mut distinct_target_ids = target_ids.clone();
    distinct_target_ids.sort();
    distinct_target_ids.dedup();
    if distinct_target_ids.len() != target_ids.len() {
        return Err(CliError(
            "calibrate-package requires distinct ordered target identities".to_string(),
        ));
    }
    let output =
        output.ok_or_else(|| CliError("calibrate-package requires --output PATH".to_string()))?;
    Ok(Command::CalibratePackage {
        package,
        component,
        phase,
        target_ids,
        output,
    })
}

fn parse_package_calibration_phase(
    phase: Option<String>,
    activation_batch_width: Option<usize>,
    command: &str,
) -> Result<PackageCalibrationPhase, CliError> {
    let phase_name = phase
        .filter(|phase| !phase.is_empty())
        .ok_or_else(|| CliError(format!("{command} requires --phase decode|prefill")))?;
    match (phase_name.as_str(), activation_batch_width) {
        ("decode", None) => Ok(PackageCalibrationPhase::Decode),
        ("decode", Some(_)) => {
            return Err(CliError(
                "decode calibration must not specify --batch-width".to_string(),
            ));
        }
        ("prefill", Some(activation_batch_width)) if activation_batch_width > 0 => {
            Ok(PackageCalibrationPhase::Prefill {
                activation_batch_width,
            })
        }
        ("prefill", _) => {
            return Err(CliError(
                "prefill calibration requires a positive --batch-width".to_string(),
            ));
        }
        _ => {
            return Err(CliError(format!(
                "{command} --phase must be decode or prefill"
            )));
        }
    }
}

fn is_canonical_vulkan_target_id(target_id: &str) -> bool {
    target_id.strip_prefix("vulkan-uuid:").is_some_and(|uuid| {
        uuid.len() == 32
            && uuid
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
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
        if max_group_size == 0 {
            return Err(CliError(
                "--max-group-size must be greater than zero".to_string(),
            ));
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
    "Usage:\n  nerve-gpu-bench list [--json]\n  nerve-gpu-bench run [--output PATH] [--payload-bytes BYTES] [--samples N] [--format FORMAT ...] [--max-group-size N] [--include-target ID ...] [--exclude-target ID ...] [--exclude-pci PCI ...] [--exclude-kind KIND ...] [--no-pairs] [--dry-plan] [--execute]\n  nerve-gpu-bench calibrate-package --package PACKAGE.json --component ID --phase decode|prefill [--batch-width N] --target VULKAN_UUID ... --output CATALOG.json\n  nerve-gpu-bench calibrate-boundaries --package PACKAGE.json --phase decode|prefill [--batch-width N] --source VULKAN_UUID --target VULKAN_UUID --output CATALOG.json\n  nerve-gpu-bench merge-catalogs --input CATALOG.json --input CATALOG.json ... --output MERGED.json\n  nerve-gpu-bench summarize --input PATH\n  nerve-gpu-bench validate --input PATH\n"
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
    fn accepts_unbounded_positive_group_size() {
        let command = parse_args([
            "run".to_string(),
            "--max-group-size".to_string(),
            "17".to_string(),
        ])
        .unwrap();
        match command {
            Command::Run { max_group_size, .. } => assert_eq!(max_group_size, Some(17)),
            _ => panic!("expected run command"),
        }
    }

    #[test]
    fn rejects_zero_group_size() {
        let error = parse_args([
            "run".to_string(),
            "--max-group-size".to_string(),
            "0".to_string(),
        ])
        .unwrap_err();
        assert!(error.to_string().contains("greater than zero"));
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

    #[test]
    fn parses_ordered_decode_package_calibration_candidate() {
        let owner = "vulkan-uuid:00112233445566778899aabbccddeeff";
        let worker = "vulkan-uuid:ffeeddccbbaa99887766554433221100";
        assert_eq!(
            parse_args(
                [
                    "calibrate-package",
                    "--package",
                    "compiled/vulkan_resident_package.json",
                    "--component",
                    "transformer.block.7",
                    "--phase",
                    "decode",
                    "--target",
                    owner,
                    "--target",
                    worker,
                    "--output",
                    "placement.json",
                ]
                .map(str::to_string)
            )
            .unwrap(),
            Command::CalibratePackage {
                package: PathBuf::from("compiled/vulkan_resident_package.json"),
                component: "transformer.block.7".to_string(),
                phase: PackageCalibrationPhase::Decode,
                target_ids: vec![owner.to_string(), worker.to_string()],
                output: PathBuf::from("placement.json"),
            },
        );
    }

    #[test]
    fn parses_exact_prefill_package_calibration_shape() {
        let target = "vulkan-uuid:00112233445566778899aabbccddeeff";
        let command = parse_args(
            [
                "calibrate-package",
                "--package",
                "package.json",
                "--component",
                "block",
                "--phase",
                "prefill",
                "--batch-width",
                "64",
                "--target",
                target,
                "--output",
                "placement.json",
            ]
            .map(str::to_string),
        )
        .unwrap();
        assert!(matches!(
            command,
            Command::CalibratePackage {
                phase: PackageCalibrationPhase::Prefill {
                    activation_batch_width: 64
                },
                ..
            }
        ));
    }

    #[test]
    fn parses_exact_prefill_boundary_calibration_shape() {
        let source = "vulkan-uuid:00112233445566778899aabbccddeeff";
        let target = "vulkan-uuid:ffeeddccbbaa99887766554433221100";
        assert_eq!(
            parse_args(
                [
                    "calibrate-boundaries",
                    "--package",
                    "compiled/vulkan_resident_package.json",
                    "--phase",
                    "prefill",
                    "--batch-width",
                    "64",
                    "--source",
                    source,
                    "--target",
                    target,
                    "--output",
                    "boundaries.json",
                ]
                .map(str::to_string)
            )
            .unwrap(),
            Command::CalibrateBoundaries {
                package: PathBuf::from("compiled/vulkan_resident_package.json"),
                phase: PackageCalibrationPhase::Prefill {
                    activation_batch_width: 64,
                },
                source_id: source.to_string(),
                target_id: target.to_string(),
                output: PathBuf::from("boundaries.json"),
            },
        );
    }

    #[test]
    fn boundary_calibration_rejects_ambiguous_endpoints_and_phase() {
        let target = "vulkan-uuid:00112233445566778899aabbccddeeff";
        let base = [
            "calibrate-boundaries",
            "--package",
            "package.json",
            "--phase",
            "decode",
            "--source",
            target,
            "--target",
            target,
            "--output",
            "boundaries.json",
        ];
        assert!(
            parse_args(base.map(str::to_string))
                .unwrap_err()
                .to_string()
                .contains("distinct source and target")
        );

        let prefill_without_width = base
            .iter()
            .copied()
            .map(|argument| {
                if argument == "decode" {
                    "prefill"
                } else {
                    argument
                }
            })
            .map(str::to_string);
        assert!(
            parse_args(prefill_without_width)
                .unwrap_err()
                .to_string()
                .contains("positive --batch-width")
        );

        let noncanonical = base
            .iter()
            .copied()
            .map(|argument| if argument == target { "gpu0" } else { argument })
            .map(str::to_string);
        assert!(
            parse_args(noncanonical)
                .unwrap_err()
                .to_string()
                .contains("canonical --source")
        );
    }

    #[test]
    fn boundary_calibration_rejects_repeated_scalar_options() {
        let source = "vulkan-uuid:00112233445566778899aabbccddeeff";
        let target = "vulkan-uuid:ffeeddccbbaa99887766554433221100";
        for (option, second) in [
            ("--package", "other.json"),
            ("--phase", "decode"),
            ("--batch-width", "32"),
            ("--source", target),
            ("--target", source),
            ("--output", "other.json"),
        ] {
            let mut arguments = [
                "calibrate-boundaries",
                "--package",
                "package.json",
                "--phase",
                "prefill",
                "--batch-width",
                "64",
                "--source",
                source,
                "--target",
                target,
                "--output",
                "boundaries.json",
            ]
            .map(str::to_string)
            .to_vec();
            arguments.extend([option.to_string(), second.to_string()]);
            let error = parse_args(arguments).unwrap_err();
            assert!(
                error.to_string().contains("may only be specified once"),
                "{option} produced {error}"
            );
        }
    }

    #[test]
    fn rejects_ambiguous_or_noncanonical_package_calibration_requests() {
        let base = [
            "calibrate-package",
            "--package",
            "package.json",
            "--component",
            "block",
            "--phase",
            "decode",
            "--target",
            "vulkan-uuid:00112233445566778899aabbccddeeff",
            "--output",
            "placement.json",
        ];
        let with_decode_width = base
            .iter()
            .copied()
            .chain(["--batch-width", "2"])
            .map(str::to_string);
        assert!(
            parse_args(with_decode_width)
                .unwrap_err()
                .to_string()
                .contains("must not specify")
        );

        for invalid_target in [
            "gpu0",
            "vulkan-uuid:0011",
            "vulkan-uuid:00112233445566778899AABBCCDDEEFF",
        ] {
            let arguments = base
                .iter()
                .copied()
                .map(|argument| {
                    if argument == "vulkan-uuid:00112233445566778899aabbccddeeff" {
                        invalid_target
                    } else {
                        argument
                    }
                })
                .map(str::to_string);
            assert!(
                parse_args(arguments)
                    .unwrap_err()
                    .to_string()
                    .contains("canonical vulkan-uuid")
            );
        }
    }

    #[test]
    fn rejects_missing_width_and_duplicate_package_targets() {
        let target = "vulkan-uuid:00112233445566778899aabbccddeeff";
        let missing_width = parse_args(
            [
                "calibrate-package",
                "--package",
                "package.json",
                "--component",
                "block",
                "--phase",
                "prefill",
                "--target",
                target,
                "--output",
                "placement.json",
            ]
            .map(str::to_string),
        )
        .unwrap_err();
        assert!(missing_width.to_string().contains("positive --batch-width"));

        let repeated_target = parse_args(
            [
                "calibrate-package",
                "--package",
                "package.json",
                "--component",
                "block",
                "--phase",
                "decode",
                "--target",
                target,
                "--target",
                target,
                "--output",
                "placement.json",
            ]
            .map(str::to_string),
        )
        .unwrap_err();
        assert!(repeated_target.to_string().contains("distinct ordered"));
    }

    #[test]
    fn rejects_repeated_scalar_package_calibration_options() {
        let target = "vulkan-uuid:00112233445566778899aabbccddeeff";
        for (option, second) in [
            ("--package", "other.json"),
            ("--component", "other"),
            ("--phase", "prefill"),
            ("--batch-width", "32"),
            ("--output", "other.json"),
        ] {
            let mut arguments = [
                "calibrate-package",
                "--package",
                "package.json",
                "--component",
                "block",
                "--phase",
                "prefill",
                "--batch-width",
                "64",
                "--target",
                target,
                "--output",
                "placement.json",
            ]
            .map(str::to_string)
            .to_vec();
            arguments.extend([option.to_string(), second.to_string()]);
            let error = parse_args(arguments).unwrap_err();
            assert!(
                error.to_string().contains("may only be specified once"),
                "{option} produced {error}"
            );
        }
    }

    #[test]
    fn parses_exact_catalog_merge() {
        assert_eq!(
            parse_args(
                [
                    "merge-catalogs",
                    "--input",
                    "owner-a.json",
                    "--input",
                    "owner-b.json",
                    "--output",
                    "merged.json",
                ]
                .map(str::to_string),
            )
            .unwrap(),
            Command::MergeCatalogs {
                inputs: vec![PathBuf::from("owner-a.json"), PathBuf::from("owner-b.json"),],
                output: PathBuf::from("merged.json"),
            }
        );
    }

    #[test]
    fn rejects_underspecified_or_duplicate_catalog_merge() {
        let one_input = parse_args(
            [
                "merge-catalogs",
                "--input",
                "one.json",
                "--output",
                "merged.json",
            ]
            .map(str::to_string),
        )
        .unwrap_err();
        assert!(one_input.to_string().contains("at least two"));

        let duplicate = parse_args(
            [
                "merge-catalogs",
                "--input",
                "one.json",
                "--input",
                "one.json",
                "--output",
                "merged.json",
            ]
            .map(str::to_string),
        )
        .unwrap_err();
        assert!(duplicate.to_string().contains("distinct input"));
    }
}
