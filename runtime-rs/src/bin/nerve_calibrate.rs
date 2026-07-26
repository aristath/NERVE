use nerve_runtime::{CalibrationRunnerOptions, read_calibration_plan, run_calibration_plan};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("nerve-calibrate error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    if arguments == ["--fingerprint"] {
        println!("{}", env!("NERVE_HARDWARE_CALIBRATOR_FINGERPRINT"));
        return Ok(());
    }
    if arguments.iter().any(|argument| argument == "--help") {
        println!(
            "Usage: nerve-calibrate --plan PLAN.json --output RUN.json \
             [--artifacts DIRECTORY] [--vulkan-device-index INDEX]\n\
             Executes one validated hardware-calibration plan sequentially.\n\
             nerve-calibrate --fingerprint prints its implementation identity."
        );
        return Ok(());
    }
    let plan_path =
        option_path(&arguments, "--plan")?.ok_or_else(|| "--plan is required".to_string())?;
    let output_path =
        option_path(&arguments, "--output")?.ok_or_else(|| "--output is required".to_string())?;
    let artifact_directory = option_path(&arguments, "--artifacts")?
        .unwrap_or_else(|| output_path.with_extension("artifacts"));
    let vulkan_physical_device_index = option_value(&arguments, "--vulkan-device-index")?
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|error| format!("invalid --vulkan-device-index {value:?}: {error}"))
        })
        .transpose()?;
    reject_unknown_arguments(&arguments)?;
    let plan = read_calibration_plan(&plan_path)?;
    let options = CalibrationRunnerOptions {
        artifact_directory,
        vulkan_physical_device_index,
        ..CalibrationRunnerOptions::default()
    };
    let run = run_calibration_plan(&plan, &options)?;
    let payload = serde_json::to_vec_pretty(&run)
        .map_err(|error| format!("could not serialize calibration run: {error}"))?;
    write_atomic(&output_path, &payload)?;
    println!(
        "calibrated {} workloads for {} into {}",
        run.workloads.len(),
        run.hardware_profile_id,
        output_path.display()
    );
    Ok(())
}

fn option_path(arguments: &[String], name: &str) -> Result<Option<PathBuf>, String> {
    option_value(arguments, name).map(|value| value.map(PathBuf::from))
}

fn option_value(arguments: &[String], name: &str) -> Result<Option<String>, String> {
    let positions = arguments
        .iter()
        .enumerate()
        .filter(|(_, argument)| argument.as_str() == name)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if positions.len() > 1 {
        return Err(format!("{name} may only be provided once"));
    }
    let Some(index) = positions.first().copied() else {
        return Ok(None);
    };
    let value = arguments
        .get(index + 1)
        .ok_or_else(|| format!("{name} requires a path"))?;
    if value.starts_with("--") {
        return Err(format!("{name} requires a path"));
    }
    Ok(Some(value.clone()))
}

fn reject_unknown_arguments(arguments: &[String]) -> Result<(), String> {
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--plan" | "--output" | "--artifacts" | "--vulkan-device-index" => {
                if index + 1 >= arguments.len() {
                    return Err(format!("{} requires a value", arguments[index]));
                }
                index += 2;
            }
            unsupported => return Err(format!("unsupported argument {unsupported:?}")),
        }
    }
    Ok(())
}

fn write_atomic(path: &Path, payload: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .map_err(|error| format!("could not create output directory {parent:?}: {error}"))?;
    let temporary = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("calibration"),
        std::process::id()
    ));
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|error| format!("could not create temporary output {temporary:?}: {error}"))?;
    output
        .write_all(payload)
        .and_then(|()| output.write_all(b"\n"))
        .and_then(|()| output.sync_all())
        .map_err(|error| format!("could not write temporary output {temporary:?}: {error}"))?;
    drop(output);
    fs::rename(&temporary, path)
        .map_err(|error| format!("could not publish calibration output {path:?}: {error}"))?;
    Ok(())
}
