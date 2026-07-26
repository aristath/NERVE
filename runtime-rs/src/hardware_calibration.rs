mod cpu;
mod runner;
mod sampling;
mod schema;
#[cfg(feature = "vulkan")]
mod shader_compiler;
mod telemetry;
#[cfg(feature = "vulkan")]
mod vulkan_compute;
#[cfg(feature = "vulkan")]
mod vulkan_compute_shaders;
#[cfg(feature = "vulkan")]
mod vulkan_dgc;
#[cfg(feature = "vulkan")]
mod vulkan_graphics;
#[cfg(feature = "vulkan")]
mod vulkan_ray;
#[cfg(feature = "vulkan")]
mod vulkan_specialized;
#[cfg(feature = "vulkan")]
mod vulkan_synchronization;
#[cfg(feature = "vulkan")]
mod vulkan_transfer;
#[cfg(feature = "vulkan")]
mod vulkan_video;

pub use runner::{CalibrationRunnerOptions, read_calibration_plan, run_calibration_plan};
pub use schema::{
    HardwareCalibrationPlan, HardwareCalibrationRun, HardwareCalibrationSample,
    HardwareCalibrationWorkloadResult,
};

#[cfg(test)]
mod tests;
