use super::{SpecializedBuffer, SpecializedVulkanContext, SpecializedVulkanResources};
use ash::vk;
use sha2::{Digest, Sha256};
use std::cell::Cell;
use std::rc::Rc;

pub(crate) struct PreparedSynchronizationCalibration {
    context: Rc<SpecializedVulkanContext>,
    resources: SpecializedVulkanResources,
    output: SpecializedBuffer,
    primitive: SynchronizationPrimitive,
    round_trips: u32,
    timeline: Option<vk::Semaphore>,
    timeline_value: Cell<u64>,
}

#[derive(Clone, Copy)]
enum SynchronizationPrimitive {
    PipelineBarrier,
    Fence,
    TimelineSemaphore,
}

impl PreparedSynchronizationCalibration {
    pub(in crate::hardware_calibration) fn new(
        context: Rc<SpecializedVulkanContext>,
        primitive: &str,
        round_trips: u32,
    ) -> Result<Self, String> {
        if round_trips == 0 {
            return Err("synchronization round trips must be nonzero".to_string());
        }
        let primitive = match primitive {
            "pipeline_barrier" => SynchronizationPrimitive::PipelineBarrier,
            "fence" => SynchronizationPrimitive::Fence,
            "timeline_semaphore" => SynchronizationPrimitive::TimelineSemaphore,
            other => {
                return Err(format!(
                    "synchronization calibrator does not implement {other:?}"
                ));
            }
        };
        let output = context.create_buffer(
            4096,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL
                | vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
            true,
        )?;
        output.write(&vec![0; 4096])?;
        let resources =
            SpecializedVulkanResources::new(Rc::clone(&context), context.compute_queue_family()?)?;
        resources.begin()?;
        let command_round_trips = if matches!(primitive, SynchronizationPrimitive::PipelineBarrier)
        {
            round_trips
        } else {
            1
        };
        for index in 0..command_round_trips {
            unsafe {
                context.device().cmd_fill_buffer(
                    resources.command_buffer,
                    output.buffer,
                    0,
                    4096,
                    0x9e37_79b9u32.wrapping_add(index),
                );
                context.device().cmd_pipeline_barrier2(
                    resources.command_buffer,
                    &vk::DependencyInfo::default().buffer_memory_barriers(&[
                        vk::BufferMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .buffer(output.buffer)
                            .offset(0)
                            .size(4096),
                    ]),
                );
            }
        }
        unsafe {
            context.device().cmd_pipeline_barrier2(
                resources.command_buffer,
                &vk::DependencyInfo::default().buffer_memory_barriers(&[
                    vk::BufferMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                        .dst_access_mask(vk::AccessFlags2::HOST_READ)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .buffer(output.buffer)
                        .offset(0)
                        .size(4096),
                ]),
            );
        }
        resources.finish_recording()?;
        let timeline = if matches!(primitive, SynchronizationPrimitive::TimelineSemaphore) {
            let mut type_info = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            Some(
                unsafe {
                    context.device().create_semaphore(
                        &vk::SemaphoreCreateInfo::default().push_next(&mut type_info),
                        None,
                    )
                }
                .map_err(|error| format!("could not create calibration timeline: {error:?}"))?,
            )
        } else {
            None
        };
        Ok(Self {
            context,
            resources,
            output,
            primitive,
            round_trips,
            timeline,
            timeline_value: Cell::new(0),
        })
    }

    pub(in crate::hardware_calibration) fn run(&self) -> Result<u64, String> {
        match self.primitive {
            SynchronizationPrimitive::PipelineBarrier => self.resources.run(1_000_000_000),
            SynchronizationPrimitive::Fence => {
                let mut duration = 0u64;
                for _ in 0..self.round_trips {
                    duration = duration.saturating_add(self.resources.run(1_000_000_000)?);
                }
                Ok(duration)
            }
            SynchronizationPrimitive::TimelineSemaphore => {
                let semaphore = self
                    .timeline
                    .ok_or_else(|| "timeline semaphore was not created".to_string())?;
                let mut duration = 0u64;
                for _ in 0..self.round_trips {
                    let value = self.timeline_value.get().saturating_add(1);
                    self.timeline_value.set(value);
                    duration = duration.saturating_add(self.resources.run_with_timeline(
                        semaphore,
                        value,
                        1_000_000_000,
                    )?);
                }
                Ok(duration)
            }
        }
    }

    pub(in crate::hardware_calibration) fn observed_digest(&self) -> Result<String, String> {
        let bytes = self.output.read(4096)?;
        if bytes.iter().all(|byte| *byte == 0) {
            return Err("synchronization calibration produced unchanged output".to_string());
        }
        Ok(format!(
            "nerve.calibration_output_sha256.v1:{:x}",
            Sha256::digest(bytes)
        ))
    }
}

impl Drop for PreparedSynchronizationCalibration {
    fn drop(&mut self) {
        if let Some(timeline) = self.timeline {
            unsafe {
                let _ = self.context.device().device_wait_idle();
                self.context.device().destroy_semaphore(timeline, None);
            }
        }
    }
}
