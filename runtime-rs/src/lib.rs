#[cfg(all(feature = "vulkan", feature = "tokenizers"))]
pub mod chat;
pub mod critical_path;
#[cfg(feature = "vulkan")]
pub mod editor;
pub mod execution_schedule;
pub mod hardware_calibration;
pub mod hardware_profile;
pub mod implementation_selection;
pub use nerve_execution_contracts as execution_contracts;
pub mod representation_graph;
pub mod stream_circuit;
pub mod stream_plan;
pub mod stream_prefix_cache;
pub mod stream_runtime;
pub mod stream_state;
pub mod tensor_storage;
#[cfg(test)]
pub(crate) mod test_support;
#[cfg(feature = "tui")]
pub mod tui;
pub mod vulkan;
#[cfg(feature = "vulkan")]
pub mod vulkan_compute;
#[cfg(feature = "vulkan")]
pub mod vulkan_distributed;
#[cfg(feature = "vulkan")]
pub mod vulkan_stream_circuit;

#[cfg(all(feature = "vulkan", feature = "tokenizers"))]
pub use chat::*;
pub use critical_path::*;
#[cfg(feature = "vulkan")]
pub use editor::*;
pub use execution_schedule::*;
pub use hardware_calibration::*;
pub use hardware_profile::*;
pub use implementation_selection::*;
pub use representation_graph::*;
pub use stream_circuit::*;
pub use stream_plan::*;
pub use stream_prefix_cache::*;
pub use stream_runtime::*;
pub use stream_state::*;
pub use tensor_storage::*;
pub use vulkan::*;
#[cfg(feature = "vulkan")]
pub use vulkan_compute::*;
#[cfg(feature = "vulkan")]
pub use vulkan_distributed::*;
#[cfg(feature = "vulkan")]
pub use vulkan_stream_circuit::*;

pub const RUNTIME_IMPLEMENTATION_FINGERPRINT: &str =
    env!("NERVE_RUNTIME_IMPLEMENTATION_FINGERPRINT");
