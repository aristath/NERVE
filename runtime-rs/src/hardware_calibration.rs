mod cpu;
mod runner;
mod schema;
#[cfg(feature = "vulkan")]
mod shader_compiler;
#[cfg(feature = "vulkan")]
mod vulkan_compute;
#[cfg(feature = "vulkan")]
mod vulkan_compute_shaders;
#[cfg(feature = "vulkan")]
mod vulkan_graphics;
#[cfg(feature = "vulkan")]
mod vulkan_transfer;

pub use runner::{CalibrationRunnerOptions, read_calibration_plan, run_calibration_plan};
pub use schema::{
    HardwareCalibrationPlan, HardwareCalibrationRun, HardwareCalibrationSample,
    HardwareCalibrationWorkloadResult,
};

#[cfg(test)]
mod tests;
