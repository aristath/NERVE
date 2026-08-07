use std::ffi::CString;

use ash::{Entry, vk};

use crate::benchmark::single_target_status_measurements;
use crate::model::{Measurement, Target};

pub fn run_vulkan_single_target_measurements(
    target: &Target,
    payload_bytes: usize,
    formats: &[String],
    workloads: &[String],
) -> Vec<Measurement> {
    match open_compute_device(target) {
        Ok(device) => single_target_status_measurements(
            &target.stable_target_id,
            payload_bytes,
            formats,
            workloads,
            "unmeasured",
            &format!(
                "vulkan_compute_kernel_not_implemented_after_opening_physical_device_{}_queue_family_{}",
                device.physical_device_index, device.compute_queue_family_index
            ),
        ),
        Err(message) => single_target_status_measurements(
            &target.stable_target_id,
            payload_bytes,
            formats,
            workloads,
            "failed",
            &message,
        ),
    }
}

struct OpenVulkanComputeDevice {
    device: ash::Device,
    instance: ash::Instance,
    physical_device_index: usize,
    compute_queue_family_index: u32,
}

impl Drop for OpenVulkanComputeDevice {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn open_compute_device(target: &Target) -> Result<OpenVulkanComputeDevice, String> {
    let vulkan = target
        .vulkan
        .as_ref()
        .ok_or_else(|| "target has no Vulkan physical-device metadata".to_string())?;
    let entry = unsafe { Entry::load() }
        .map_err(|error| format!("could not load Vulkan loader: {error}"))?;
    let app_name = CString::new("nerve-gpu-bench").expect("static string has no nul");
    let engine_name = CString::new("nerve").expect("static string has no nul");
    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(1)
        .engine_name(&engine_name)
        .engine_version(1)
        .api_version(vk::make_api_version(0, 1, 3, 0));
    let instance = unsafe {
        entry.create_instance(
            &vk::InstanceCreateInfo::default().application_info(&app_info),
            None,
        )
    }
    .map_err(|error| format!("could not create Vulkan instance: {error:?}"))?;
    let result =
        unsafe { open_compute_device_from_instance(instance, vulkan.physical_device_index) };
    match result {
        Ok(device) => Ok(device),
        Err((instance, message)) => {
            unsafe { instance.destroy_instance(None) };
            Err(message)
        }
    }
}

unsafe fn open_compute_device_from_instance(
    instance: ash::Instance,
    physical_device_index: usize,
) -> Result<OpenVulkanComputeDevice, (ash::Instance, String)> {
    let physical_devices = unsafe { instance.enumerate_physical_devices() }.map_err(|error| {
        (
            instance.clone(),
            format!("could not enumerate Vulkan physical devices: {error:?}"),
        )
    })?;
    let physical_device = *physical_devices.get(physical_device_index).ok_or_else(|| {
        (
            instance.clone(),
            format!("Vulkan physical device index {physical_device_index} is no longer available"),
        )
    })?;
    let queue_families =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let compute_queue_family_index =
        compute_queue_family_index(&queue_families).ok_or_else(|| {
            (
                instance.clone(),
                format!(
                    "Vulkan physical device index {physical_device_index} has no compute queue"
                ),
            )
        })?;
    let priorities = [1.0_f32];
    let queue_info = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(compute_queue_family_index)
        .queue_priorities(&priorities)];
    let device_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_info);
    let device = unsafe { instance.create_device(physical_device, &device_info, None) }.map_err(
        |error| {
            (
                instance.clone(),
                format!("could not create Vulkan logical device: {error:?}"),
            )
        },
    )?;
    Ok(OpenVulkanComputeDevice {
        device,
        instance,
        physical_device_index,
        compute_queue_family_index,
    })
}

fn compute_queue_family_index(queue_families: &[vk::QueueFamilyProperties]) -> Option<u32> {
    queue_families
        .iter()
        .enumerate()
        .filter(|(_, family)| {
            family.queue_count > 0 && family.queue_flags.contains(vk::QueueFlags::COMPUTE)
        })
        .min_by_key(|(_, family)| family.queue_flags.contains(vk::QueueFlags::GRAPHICS))
        .map(|(index, _)| index as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefers_compute_only_queue_family() {
        let queue_families = [
            vk::QueueFamilyProperties {
                queue_flags: vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
                queue_count: 1,
                ..Default::default()
            },
            vk::QueueFamilyProperties {
                queue_flags: vk::QueueFlags::COMPUTE,
                queue_count: 1,
                ..Default::default()
            },
        ];
        assert_eq!(compute_queue_family_index(&queue_families), Some(1));
    }

    #[test]
    fn ignores_queue_families_without_compute_or_queues() {
        let queue_families = [
            vk::QueueFamilyProperties {
                queue_flags: vk::QueueFlags::GRAPHICS,
                queue_count: 1,
                ..Default::default()
            },
            vk::QueueFamilyProperties {
                queue_flags: vk::QueueFlags::COMPUTE,
                queue_count: 0,
                ..Default::default()
            },
        ];
        assert_eq!(compute_queue_family_index(&queue_families), None);
    }
}
