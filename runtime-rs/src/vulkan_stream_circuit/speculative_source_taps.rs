enum VulkanSpeculativeSourceTapTransfer {
    DeviceLocal(VulkanResidentBufferCopy),
    HostStaged {
        source_copy: VulkanResidentBufferCopy,
        source_staging: VulkanResidentBuffer,
        destination_staging: VulkanResidentBuffer,
        destination_copy: VulkanResidentBufferCopy,
        byte_len: usize,
    },
}

struct VulkanResolvedSpeculativeSourceTap<'a> {
    device_id: &'a str,
    scalar_buffer: &'a VulkanResidentBuffer,
    scalar_buffer_owner: Arc<VulkanResidentBuffer>,
    batch_signal_key: VulkanComponentBatchSignalKey,
    frame_byte_capacity: usize,
}

impl VulkanSpeculativeSourceTapTransfer {
    fn new(
        source_device: &VulkanComputeDevice,
        destination_device: &VulkanComputeDevice,
        source: &VulkanResidentBuffer,
        destination: &VulkanResidentBuffer,
        byte_len: usize,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        if source_device.shares_logical_device_with(destination_device) {
            return destination_device
                .create_resident_buffer_copy(source, destination, byte_len)
                .map(Self::DeviceLocal)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop);
        }

        let mut source_staging = source_device
            .create_host_visible_resident_buffer(byte_len)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        source_staging
            .persistently_map()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let mut destination_staging = destination_device
            .create_host_visible_resident_buffer(byte_len)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        destination_staging
            .persistently_map()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let source_copy = source_device
            .create_resident_buffer_copy(source, &source_staging, byte_len)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let destination_copy = destination_device
            .create_resident_buffer_copy(&destination_staging, destination, byte_len)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        Ok(Self::HostStaged {
            source_copy,
            source_staging,
            destination_staging,
            destination_copy,
            byte_len,
        })
    }

    fn run(&self) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        match self {
            Self::DeviceLocal(copy) => copy
                .run(copy.byte_len())
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop),
            Self::HostStaged {
                source_copy,
                source_staging,
                destination_staging,
                destination_copy,
                byte_len,
            } => {
                source_copy
                    .run(*byte_len)
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                let bytes = source_staging
                    .read_bytes(*byte_len)
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                destination_staging
                    .write_bytes(&bytes)
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                destination_copy
                    .run(*byte_len)
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
            }
        }
    }
}

fn resolve_speculative_source_tap_instance<'a>(
    instances: &'a [VulkanRuntimeComponentInstance],
    tap: &StreamCircuitGraphSourceTap,
) -> Result<&'a VulkanRuntimeComponentInstance, VulkanResidentInProcessPlacedRuntimeError> {
    match tap.instance_selection {
        StreamCircuitGraphSourceTapInstanceSelection::LastInExecutionOrder => instances
            .iter()
            .filter(|instance| instance.source_component_id == tap.component_id)
            .max_by_key(|instance| instance.execution_index)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(format!(
                        "speculative source tap references absent target component {:?}",
                        tap.component_id
                    )),
                )
            }),
    }
}

fn resolved_speculative_source_tap_buffer<'a>(
    model: &VulkanResidentInProcessPlacedModelPackage,
    target_slices: &'a [VulkanResidentInProcessPlacedStreamProcessorDevice],
    tap: &StreamCircuitGraphSourceTap,
) -> Result<VulkanResolvedSpeculativeSourceTap<'a>, VulkanResidentInProcessPlacedRuntimeError> {
    let instance =
        resolve_speculative_source_tap_instance(&model.runtime_component_instances, tap)?;
    let slice = target_slices
        .iter()
        .find(|slice| slice.device_id == instance.device_id)
        .ok_or_else(|| VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
            device_id: instance.device_id.clone(),
        })?;
    let circuit = slice
        .mounted
        .placed_plan
        .binding_plan
        .circuit(&instance.instance_id)
        .ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "speculative source tap target instance {:?} is not mounted on {:?}",
                    instance.instance_id, instance.device_id
                )),
            )
        })?;
    let port = circuit
        .output_ports
        .iter()
        .find(|port| port.id == tap.port_id)
        .ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "speculative source tap target {:?} has no output port {:?}",
                    instance.instance_id, tap.port_id
                )),
            )
        })?;
    let source_signal = port.source.as_deref().unwrap_or(port.id.as_str());
    let descriptor = slice
        .mounted_bound
        .dispatches
        .iter()
        .filter(|dispatch| dispatch.component_id == instance.instance_id)
        .flat_map(|dispatch| &dispatch.descriptors)
        .find(|descriptor| {
            descriptor.usage == VulkanKernelDescriptorUsage::OutputSignal
                && descriptor.name == source_signal
        })
        .ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "speculative source tap target {}.{} has no mounted output descriptor",
                    instance.instance_id, tap.port_id
                )),
            )
        })?;
    let (buffer, buffer_owner, byte_len) = match &descriptor.target {
        VulkanMountedPlacedBoundDescriptorTarget::Resident {
            target:
                VulkanBoundDescriptorTarget::ActivationSlot {
                    buffer_index,
                    signal_byte_capacity,
                    ..
                },
        } => {
            let allocation = slice
                .mounted
                .buffers
                .activation_slot_buffers
                .get(*buffer_index)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                        "speculative source tap activation slot {buffer_index} is absent"
                    )))
                })?;
            (
                allocation.buffer.as_ref(),
                Arc::clone(&allocation.buffer),
                *signal_byte_capacity,
            )
        }
        VulkanMountedPlacedBoundDescriptorTarget::ModelOutput { signal_id } => {
            let allocation = slice
                .mounted
                .boundary_io
                .output_buffer(signal_id)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                        "speculative source tap model output {signal_id:?} is absent"
                    )))
                })?;
            (
                allocation.buffer.as_ref(),
                Arc::clone(&allocation.buffer),
                allocation.byte_capacity,
            )
        }
        VulkanMountedPlacedBoundDescriptorTarget::ProducedPortBuffer { port } => {
            let allocation = port.buffer(&slice.mounted.edge_io).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "speculative source tap produced port is absent".to_string(),
                ))
            })?;
            (
                allocation.as_ref(),
                Arc::clone(allocation),
                port.byte_capacity,
            )
        }
        _ => {
            return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "speculative source tap target {}.{} is not a produced output buffer",
                    instance.instance_id, tap.port_id
                )),
            ))
        }
    };
    let (batch_signal_key, batch_frame_byte_capacity) =
        component_batch_signal_target_with_mounted(&slice.mounted, descriptor)?
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(format!(
                        "speculative source tap target {}.{} is not addressable in a component batch",
                        instance.instance_id, tap.port_id
                    )),
                )
            })?;
    if batch_frame_byte_capacity != byte_len {
        return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
            VulkanResidentTokenModelPackageError::new(format!(
                "speculative source tap target {}.{} has scalar frame bytes {byte_len} but batch frame bytes {batch_frame_byte_capacity}",
                instance.instance_id, tap.port_id
            )),
        ));
    }
    Ok(VulkanResolvedSpeculativeSourceTap {
        device_id: slice.device_id.as_str(),
        scalar_buffer: buffer,
        scalar_buffer_owner: buffer_owner,
        batch_signal_key,
        frame_byte_capacity: byte_len,
    })
}
