use std::ffi::{CStr, c_void};

use ash::vk;

pub const SHADER_FLOAT8_NAME: &CStr = c"VK_EXT_shader_float8";
pub const MIXED_FLOAT_DOT_PRODUCT_NAME: &CStr = c"VK_VALVE_shader_mixed_float_dot_product";
pub const NATIVE_FP8_DOT_FEATURE: &str = "native_fp8_dot_f32_accum";
pub const EXTERNAL_TIMELINE_SEMAPHORE_FEATURE: &str = "external_timeline_semaphore_opaque_fd";
pub const EXTERNAL_MEMORY_DMA_BUF_FEATURE: &str = "external_memory_dma_buf";
pub const EXTERNAL_MEMORY_HOST_FEATURE: &str = "external_memory_host";

const SHADER_FLOAT8_FEATURES_STRUCTURE_TYPE: i32 = 1_000_567_000;
const MIXED_FLOAT_DOT_PRODUCT_FEATURES_STRUCTURE_TYPE: i32 = 1_000_673_000;

#[repr(C)]
pub struct ShaderFloat8Features {
    pub s_type: vk::StructureType,
    pub p_next: *mut c_void,
    pub shader_float8: vk::Bool32,
    pub shader_float8_cooperative_matrix: vk::Bool32,
}

impl ShaderFloat8Features {
    pub fn disabled() -> Self {
        Self {
            s_type: vk::StructureType::from_raw(SHADER_FLOAT8_FEATURES_STRUCTURE_TYPE),
            p_next: std::ptr::null_mut(),
            shader_float8: vk::FALSE,
            shader_float8_cooperative_matrix: vk::FALSE,
        }
    }
}

#[repr(C)]
pub struct MixedFloatDotProductFeatures {
    pub s_type: vk::StructureType,
    pub p_next: *mut c_void,
    pub shader_float16_acc_float32: vk::Bool32,
    pub shader_float16_acc_float16: vk::Bool32,
    pub shader_bfloat16_acc: vk::Bool32,
    pub shader_float8_acc_float32: vk::Bool32,
}

impl MixedFloatDotProductFeatures {
    pub fn disabled() -> Self {
        Self {
            s_type: vk::StructureType::from_raw(MIXED_FLOAT_DOT_PRODUCT_FEATURES_STRUCTURE_TYPE),
            p_next: std::ptr::null_mut(),
            shader_float16_acc_float32: vk::FALSE,
            shader_float16_acc_float16: vk::FALSE,
            shader_bfloat16_acc: vk::FALSE,
            shader_float8_acc_float32: vk::FALSE,
        }
    }
}

pub unsafe fn native_fp8_dot_supported(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    extension_names: &[String],
) -> bool {
    if !extension_names
        .iter()
        .any(|name| name == SHADER_FLOAT8_NAME.to_str().unwrap())
        || !extension_names
            .iter()
            .any(|name| name == MIXED_FLOAT_DOT_PRODUCT_NAME.to_str().unwrap())
    {
        return false;
    }

    let mut mixed = MixedFloatDotProductFeatures::disabled();
    let mut float8 = ShaderFloat8Features::disabled();
    float8.p_next = std::ptr::from_mut(&mut mixed).cast();
    let mut features = vk::PhysicalDeviceFeatures2 {
        p_next: std::ptr::from_mut(&mut float8).cast(),
        ..Default::default()
    };
    unsafe { instance.get_physical_device_features2(physical_device, &mut features) };
    float8.shader_float8 == vk::TRUE && mixed.shader_float8_acc_float32 == vk::TRUE
}

pub unsafe fn external_timeline_semaphore_supported(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    extension_names: &[String],
) -> bool {
    if !extension_names
        .iter()
        .any(|name| name == ash::khr::external_semaphore_fd::NAME.to_str().unwrap())
    {
        return false;
    }

    let mut timeline = vk::PhysicalDeviceTimelineSemaphoreFeatures::default();
    let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut timeline);
    unsafe { instance.get_physical_device_features2(physical_device, &mut features) };

    let mut external = vk::ExternalSemaphoreProperties::default();
    unsafe {
        instance.get_physical_device_external_semaphore_properties(
            physical_device,
            &vk::PhysicalDeviceExternalSemaphoreInfo::default()
                .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD),
            &mut external,
        )
    };
    timeline.timeline_semaphore == vk::TRUE
        && external_semaphore_properties_support_import_export(&external)
}

pub fn external_semaphore_properties_support_import_export(
    properties: &vk::ExternalSemaphoreProperties<'_>,
) -> bool {
    properties.external_semaphore_features.contains(
        vk::ExternalSemaphoreFeatureFlags::EXPORTABLE
            | vk::ExternalSemaphoreFeatureFlags::IMPORTABLE,
    )
}

pub fn extension_names_include_dma_buf_shared_memory(extension_names: &[String]) -> bool {
    extension_names
        .iter()
        .any(|name| name == ash::khr::external_memory_fd::NAME.to_str().unwrap())
        && extension_names
            .iter()
            .any(|name| name == ash::ext::external_memory_dma_buf::NAME.to_str().unwrap())
}

pub fn extension_names_include_host_shared_memory(extension_names: &[String]) -> bool {
    extension_names
        .iter()
        .any(|name| name == ash::ext::external_memory_host::NAME.to_str().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_feature_structures_use_published_vulkan_abis() {
        let float8 = ShaderFloat8Features::disabled();
        let mixed = MixedFloatDotProductFeatures::disabled();
        assert_eq!(
            float8.s_type.as_raw(),
            SHADER_FLOAT8_FEATURES_STRUCTURE_TYPE
        );
        assert_eq!(
            mixed.s_type.as_raw(),
            MIXED_FLOAT_DOT_PRODUCT_FEATURES_STRUCTURE_TYPE
        );
        assert_eq!(float8.shader_float8, vk::FALSE);
        assert_eq!(mixed.shader_float8_acc_float32, vk::FALSE);
    }

    #[test]
    fn external_semaphore_requires_both_import_and_export() {
        let mut properties = vk::ExternalSemaphoreProperties::default();
        assert!(!external_semaphore_properties_support_import_export(
            &properties
        ));
        properties.external_semaphore_features = vk::ExternalSemaphoreFeatureFlags::EXPORTABLE;
        assert!(!external_semaphore_properties_support_import_export(
            &properties
        ));
        properties.external_semaphore_features = vk::ExternalSemaphoreFeatureFlags::EXPORTABLE
            | vk::ExternalSemaphoreFeatureFlags::IMPORTABLE;
        assert!(external_semaphore_properties_support_import_export(
            &properties
        ));
    }

    #[test]
    fn shared_memory_routes_require_complete_extension_sets() {
        let dma = ash::ext::external_memory_dma_buf::NAME
            .to_str()
            .unwrap()
            .to_string();
        let fd = ash::khr::external_memory_fd::NAME
            .to_str()
            .unwrap()
            .to_string();
        let host = ash::ext::external_memory_host::NAME
            .to_str()
            .unwrap()
            .to_string();
        assert!(!extension_names_include_dma_buf_shared_memory(&[
            dma.clone()
        ]));
        assert!(extension_names_include_dma_buf_shared_memory(&[dma, fd]));
        assert!(extension_names_include_host_shared_memory(&[host]));
    }
}
