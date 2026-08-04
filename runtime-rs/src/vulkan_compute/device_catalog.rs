impl VulkanResidentKernelSequence {
    pub fn has_recorded_commands(&self) -> bool {
        self.recorded_input_copies.borrow().is_some()
            && self.recorded_steps.borrow().is_some()
            && self.recorded_snapshot_copies.borrow().is_some()
    }
}

impl VulkanComputeDeviceCatalog {
    pub fn discover() -> Result<Self, VulkanError> {
        Self::discover_with_allowed_physical_device_ids(None)
    }

    pub fn discover_allowed_physical_device_ids(
        allowed_physical_device_ids: &BTreeSet<String>,
    ) -> Result<Self, VulkanError> {
        if allowed_physical_device_ids.is_empty() {
            return Err(VulkanError(
                "the Vulkan physical-device allowlist must not be empty".to_string(),
            ));
        }
        Self::discover_with_allowed_physical_device_ids(Some(allowed_physical_device_ids))
    }

    fn discover_with_allowed_physical_device_ids(
        allowed_physical_device_ids: Option<&BTreeSet<String>>,
    ) -> Result<Self, VulkanError> {
        unsafe {
            let entry = Entry::load()
                .map_err(|error| VulkanError(format!("failed to load Vulkan: {error}")))?;
            let instance = create_nerve_vulkan_instance(&entry)?;
            let physical_devices = instance.enumerate_physical_devices().map_err(|error| {
                instance.destroy_instance(None);
                VulkanError(format!("failed to enumerate Vulkan devices: {error:?}"))
            })?;

            let discovered_physical_device_ids = physical_devices
                .iter()
                .enumerate()
                .map(|(physical_device_index, physical_device)| {
                    (
                        physical_device_index,
                        format!(
                            "vulkan-uuid:{}",
                            format_device_uuid(&physical_device_uuid(
                                &instance,
                                *physical_device,
                            ))
                        ),
                    )
                })
                .collect::<Vec<_>>();
            let allowed_physical_device_indices = match allowed_physical_device_indices(
                &discovered_physical_device_ids,
                allowed_physical_device_ids,
            ) {
                Ok(indices) => indices,
                Err(error) => {
                    instance.destroy_instance(None);
                    return Err(error);
                }
            };

            let allowed_physical_devices = allowed_physical_device_indices
                .iter()
                .map(|physical_device_index| physical_devices[*physical_device_index])
                .collect::<Vec<_>>();
            let selected_allowed_index =
                select_compute_device_index(&instance, &allowed_physical_devices);
            let selected_physical_device_index = selected_allowed_index
                .map(|allowed_index| allowed_physical_device_indices[allowed_index]);
            let available_devices = allowed_physical_device_indices
                .iter()
                .filter_map(|physical_device_index| {
                    inspect_compute_device(
                        &instance,
                        *physical_device_index,
                        physical_devices[*physical_device_index],
                        Some(*physical_device_index) == selected_physical_device_index,
                    )
                })
                .collect::<Vec<_>>();

            if let Some(allowed) = allowed_physical_device_ids {
                let compute_capable = available_devices
                    .iter()
                    .map(|device| device.physical_device_id.clone())
                    .collect::<BTreeSet<_>>();
                let unavailable = allowed
                    .difference(&compute_capable)
                    .cloned()
                    .collect::<Vec<_>>();
                if !unavailable.is_empty() {
                    instance.destroy_instance(None);
                    return Err(VulkanError(format!(
                        "allowed Vulkan physical devices are not compute-capable: {}",
                        unavailable.join(", ")
                    )));
                }
            }

            Ok(Self {
                context: Arc::new(VulkanInstanceContext {
                    _entry: entry,
                    instance,
                }),
                physical_devices,
                available_devices,
            })
        }
    }

    pub fn available_compute_devices(&self) -> &[VulkanComputeDeviceInfo] {
        &self.available_devices
    }

    pub fn available_target_capabilities(
        &self,
    ) -> Result<Vec<VulkanComputeTargetCapabilities>, VulkanError> {
        self.available_devices
            .iter()
            .map(|device| {
                let physical_device = self.physical_devices[device.physical_device_index];
                let properties = unsafe {
                    self.context
                        .instance
                        .get_physical_device_properties(physical_device)
                };
                let subgroup = physical_device_subgroup_support(
                    &self.context.instance,
                    physical_device,
                );
                let shader_features = physical_device_supported_shader_features(
                    &self.context.instance,
                    physical_device,
                )?;
                let cooperative_bfloat16_shapes = if BTreeSet::from([
                    VulkanShaderFeature::CooperativeMatrix,
                    VulkanShaderFeature::ShaderBfloat16Type,
                    VulkanShaderFeature::ShaderBfloat16CooperativeMatrix,
                ])
                .is_subset(&shader_features)
                {
                    physical_device_cooperative_bfloat16_shapes(
                        &self.context._entry,
                        &self.context.instance,
                        physical_device,
                    )?
                } else {
                    BTreeSet::new()
                };
                let cooperative_float16_shapes = if BTreeSet::from([
                    VulkanShaderFeature::CooperativeMatrix,
                    VulkanShaderFeature::ShaderFloat16,
                ])
                .is_subset(&shader_features)
                {
                    physical_device_cooperative_float16_shapes(
                        &self.context._entry,
                        &self.context.instance,
                        physical_device,
                    )?
                } else {
                    BTreeSet::new()
                };
                let cooperative_float8_e4m3_shapes = if BTreeSet::from([
                    VulkanShaderFeature::CooperativeMatrix,
                    VulkanShaderFeature::ShaderFloat8,
                    VulkanShaderFeature::ShaderFloat8CooperativeMatrix,
                ])
                .is_subset(&shader_features)
                {
                    physical_device_cooperative_float8_e4m3_shapes(
                        &self.context._entry,
                        &self.context.instance,
                        physical_device,
                    )?
                } else {
                    BTreeSet::new()
                };
                let cooperative_sint8_shapes = if BTreeSet::from([
                    VulkanShaderFeature::CooperativeMatrix,
                    VulkanShaderFeature::ShaderInt8,
                ])
                .is_subset(&shader_features)
                {
                    physical_device_cooperative_sint8_shapes(
                        &self.context._entry,
                        &self.context.instance,
                        physical_device,
                    )?
                } else {
                    BTreeSet::new()
                };
                Ok(VulkanComputeTargetCapabilities {
                    physical_device_index: device.physical_device_index,
                    physical_device_id: device.physical_device_id.clone(),
                    device_name: device.device_name.clone(),
                    device_type: device.device_type.clone(),
                    vendor_id: device.vendor_id,
                    device_id: device.device_id,
                    shader_features,
                    subgroup_operations: subgroup_operations(
                        subgroup.supported_operations,
                    ),
                    subgroup_compute_supported: subgroup
                        .supported_stages
                        .contains(vk::ShaderStageFlags::COMPUTE),
                    subgroup_size: subgroup.subgroup_size,
                    max_compute_work_group_invocations: properties
                        .limits
                        .max_compute_work_group_invocations,
                    max_compute_work_group_size_x: properties
                        .limits
                        .max_compute_work_group_size[0],
                    cooperative_float16_shapes,
                    cooperative_bfloat16_shapes,
                    cooperative_float8_e4m3_shapes,
                    cooperative_sint8_shapes,
                })
            })
            .collect()
    }

    pub fn open_device_uuid(
        &self,
        device_uuid: [u8; vk::UUID_SIZE],
    ) -> Result<VulkanComputeDevice, VulkanError> {
        self.open_device(None, Some(device_uuid))
    }

    pub fn open_physical_device_index(
        &self,
        physical_device_index: usize,
    ) -> Result<VulkanComputeDevice, VulkanError> {
        self.open_device(Some(physical_device_index), None)
    }

    fn open_device(
        &self,
        requested_physical_device_index: Option<usize>,
        requested_device_uuid: Option<[u8; vk::UUID_SIZE]>,
    ) -> Result<VulkanComputeDevice, VulkanError> {
        unsafe {
            let instance = &self.context.instance;
            let permitted_device = if let Some(device_uuid) = requested_device_uuid {
                self.available_devices
                    .iter()
                    .find(|device| device.device_uuid == device_uuid)
            } else if let Some(physical_device_index) = requested_physical_device_index {
                self.available_devices
                    .iter()
                    .find(|device| device.physical_device_index == physical_device_index)
            } else {
                self.available_devices
                    .iter()
                    .find(|device| device.selected_by_default)
                    .or_else(|| self.available_devices.first())
            }
            .ok_or_else(|| {
                VulkanError(
                    "the requested Vulkan device is unavailable or outside this catalog's physical-device allowlist"
                        .to_string(),
                )
            })?;
            let (physical_device, queue_family_index, device_name) =
                select_compute_device_by_index(
                    instance,
                    &self.physical_devices,
                    permitted_device.physical_device_index,
                )?;

            let queue_family = instance
                .get_physical_device_queue_family_properties(physical_device)
                .get(queue_family_index as usize)
                .copied()
                .ok_or_else(|| {
                    VulkanError(format!(
                        "selected compute queue family {queue_family_index} disappeared"
                    ))
                })?;
            let queue_priorities = if queue_family.queue_count >= 2 {
                vec![1.0_f32, 1.0_f32]
            } else {
                vec![1.0_f32]
            };
            let queue_info = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&queue_priorities)];
            let memory_properties =
                instance.get_physical_device_memory_properties(physical_device);
            let device_local_memory_bytes = (0..memory_properties.memory_heap_count)
                .map(|heap_index| memory_properties.memory_heaps[heap_index as usize])
                .filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
                .map(|heap| heap.size)
                .max()
                .unwrap_or(0);
            let memory_budget_supported = physical_device_supports_extension(
                instance,
                physical_device,
                ash::ext::memory_budget::NAME,
            )?;
            let device_coherent_memory_supported =
                physical_device_supports_device_coherent_memory(
                    instance,
                    physical_device,
                )?;
            let conditional_rendering_supported =
                physical_device_supports_conditional_rendering(
                    instance,
                    physical_device,
                )?;
            let device_fault_supported =
                physical_device_supports_device_fault(instance, physical_device)?;
            let enabled_shader_features =
                physical_device_supported_shader_features(instance, physical_device)?;
            let shader_float8_support = VulkanShaderFloat8Support {
                shader_float8: enabled_shader_features
                    .contains(&VulkanShaderFeature::ShaderFloat8),
                shader_float8_cooperative_matrix: enabled_shader_features
                    .contains(&VulkanShaderFeature::ShaderFloat8CooperativeMatrix),
            };
            let cooperative_matrix_supported =
                enabled_shader_features.contains(&VulkanShaderFeature::CooperativeMatrix);
            let shader_bfloat16_support = VulkanShaderBfloat16Support {
                shader_bfloat16_type: enabled_shader_features
                    .contains(&VulkanShaderFeature::ShaderBfloat16Type),
                shader_bfloat16_dot_product: enabled_shader_features
                    .contains(&VulkanShaderFeature::ShaderBfloat16DotProduct),
                shader_bfloat16_cooperative_matrix: enabled_shader_features
                    .contains(&VulkanShaderFeature::ShaderBfloat16CooperativeMatrix),
            };
            let mixed_float_dot_product_support = VulkanShaderMixedFloatDotProductSupport {
                shader_float16_acc_float32: enabled_shader_features.contains(
                    &VulkanShaderFeature::ShaderMixedFloatDotProductFloat16AccFloat32,
                ),
                shader_float16_acc_float16: enabled_shader_features.contains(
                    &VulkanShaderFeature::ShaderMixedFloatDotProductFloat16AccFloat16,
                ),
                shader_bfloat16_acc: enabled_shader_features
                    .contains(&VulkanShaderFeature::ShaderMixedFloatDotProductBfloat16Acc),
                shader_float8_acc_float32: enabled_shader_features.contains(
                    &VulkanShaderFeature::ShaderMixedFloatDotProductFloat8AccFloat32,
                ),
            };
            let cooperative_bfloat16_features_supported = cooperative_matrix_supported
                && shader_bfloat16_support.shader_bfloat16_type
                && shader_bfloat16_support.shader_bfloat16_cooperative_matrix;
            let cooperative_bfloat16_shapes = if cooperative_bfloat16_features_supported {
                physical_device_cooperative_bfloat16_shapes(
                    &self.context._entry,
                    instance,
                    physical_device,
                )?
            } else {
                BTreeSet::new()
            };
            let cooperative_float8_e4m3_features_supported =
                cooperative_matrix_supported
                    && shader_float8_support.shader_float8
                    && shader_float8_support.shader_float8_cooperative_matrix;
            let cooperative_float8_e4m3_shapes =
                if cooperative_float8_e4m3_features_supported {
                    physical_device_cooperative_float8_e4m3_shapes(
                        &self.context._entry,
                        instance,
                        physical_device,
                    )?
                } else {
                    BTreeSet::new()
                };
            let cooperative_sint8_shapes = if cooperative_matrix_supported
                && enabled_shader_features.contains(&VulkanShaderFeature::ShaderInt8)
            {
                physical_device_cooperative_sint8_shapes(
                    &self.context._entry,
                    instance,
                    physical_device,
                )?
            } else {
                BTreeSet::new()
            };
            let shared_host_memory_alignment =
                if physical_device_supports_extension(
                    instance,
                    physical_device,
                    ash::ext::external_memory_host::NAME,
                )? && physical_device_supports_shared_host_buffer(instance, physical_device)
                {
                    Some(physical_device_shared_host_memory_alignment(
                        instance,
                        physical_device,
                    )?)
                } else {
                    None
                };
            let shared_device_memory_supported = physical_device_supports_extension(
                instance,
                physical_device,
                ash::khr::external_memory_fd::NAME,
            )? && physical_device_supports_extension(
                instance,
                physical_device,
                ash::ext::external_memory_dma_buf::NAME,
            )? && physical_device_supports_shared_device_buffer(instance, physical_device);
            let opaque_fd_timeline_semaphore_supported = physical_device_supports_extension(
                instance,
                physical_device,
                ash::khr::external_semaphore_fd::NAME,
            )?
                && physical_device_supports_opaque_fd_timeline_semaphore(instance, physical_device);
            let (timeline_semaphore_supported, synchronization2_supported) =
                physical_device_supports_modern_submission(instance, physical_device);
            let buffer_device_address_supported = enabled_shader_features
                .contains(&VulkanShaderFeature::BufferDeviceAddress);
            if !timeline_semaphore_supported || !synchronization2_supported {
                return Err(VulkanError(format!(
                    "Vulkan device {device_name:?} does not support the required timeline-semaphore and synchronization2 execution contract"
                )));
            }
            // Logical-device features cannot be added later. Enable every supported
            // feature in the runtime's SPIR-V contract so this device can safely
            // host different compiled component packages without being recreated.
            let enabled_core_features = vk::PhysicalDeviceFeatures {
                shader_float64: bool32(
                    enabled_shader_features.contains(&VulkanShaderFeature::ShaderFloat64),
                ),
                shader_int16: bool32(
                    enabled_shader_features.contains(&VulkanShaderFeature::ShaderInt16),
                ),
                shader_int64: bool32(
                    enabled_shader_features.contains(&VulkanShaderFeature::ShaderInt64),
                ),
                ..Default::default()
            };
            let mut shader_float16_int8_features =
                vk::PhysicalDeviceShaderFloat16Int8Features::default()
                    .shader_float16(
                        enabled_shader_features.contains(&VulkanShaderFeature::ShaderFloat16),
                    )
                    .shader_int8(
                        enabled_shader_features.contains(&VulkanShaderFeature::ShaderInt8),
                    );
            let mut storage16_features = vk::PhysicalDevice16BitStorageFeatures::default()
                .storage_buffer16_bit_access(
                    enabled_shader_features
                        .contains(&VulkanShaderFeature::StorageBuffer16BitAccess),
                )
                .uniform_and_storage_buffer16_bit_access(
                    enabled_shader_features
                        .contains(&VulkanShaderFeature::UniformAndStorageBuffer16BitAccess),
                )
                .storage_push_constant16(
                    enabled_shader_features.contains(&VulkanShaderFeature::StoragePushConstant16),
                )
                .storage_input_output16(
                    enabled_shader_features.contains(&VulkanShaderFeature::StorageInputOutput16),
                );
            let mut storage8_features = vk::PhysicalDevice8BitStorageFeatures::default()
                .storage_buffer8_bit_access(
                    enabled_shader_features.contains(&VulkanShaderFeature::StorageBuffer8BitAccess),
                )
                .uniform_and_storage_buffer8_bit_access(
                    enabled_shader_features
                        .contains(&VulkanShaderFeature::UniformAndStorageBuffer8BitAccess),
                )
                .storage_push_constant8(
                    enabled_shader_features.contains(&VulkanShaderFeature::StoragePushConstant8),
                );
            let mut integer_dot_product_features =
                vk::PhysicalDeviceShaderIntegerDotProductFeatures::default()
                    .shader_integer_dot_product(
                        enabled_shader_features
                            .contains(&VulkanShaderFeature::ShaderIntegerDotProduct),
                    );
            let mut vulkan_memory_model_features =
                vk::PhysicalDeviceVulkanMemoryModelFeatures::default()
                    .vulkan_memory_model(
                        enabled_shader_features.contains(&VulkanShaderFeature::VulkanMemoryModel),
                    )
                    .vulkan_memory_model_device_scope(
                        enabled_shader_features
                            .contains(&VulkanShaderFeature::VulkanMemoryModelDeviceScope),
                    );
            let mut shader_float8_features =
                VulkanPhysicalDeviceShaderFloat8FeaturesExt::disabled();
            let mut shader_bfloat16_features =
                VulkanPhysicalDeviceShaderBfloat16FeaturesKhr::disabled();
            let mut mixed_float_dot_product_features =
                VulkanPhysicalDeviceShaderMixedFloatDotProductFeaturesValve::disabled();
            let mut cooperative_matrix_features =
                vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
            let mut timeline_semaphore_features =
                vk::PhysicalDeviceTimelineSemaphoreFeatures::default().timeline_semaphore(true);
            let mut synchronization2_features =
                vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
            let mut buffer_device_address_features =
                vk::PhysicalDeviceBufferDeviceAddressFeatures::default()
                    .buffer_device_address(buffer_device_address_supported);
            let mut conditional_rendering_features =
                vk::PhysicalDeviceConditionalRenderingFeaturesEXT::default()
                    .conditional_rendering(conditional_rendering_supported);
            let mut device_fault_features =
                vk::PhysicalDeviceFaultFeaturesEXT::default()
                    .device_fault(device_fault_supported);
            let mut device_coherent_memory_features =
                vk::PhysicalDeviceCoherentMemoryFeaturesAMD::default()
                    .device_coherent_memory(device_coherent_memory_supported);
            let mut extension_names = Vec::new();
            let mut enabled_device_extensions = BTreeSet::new();
            let mut device_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_info)
                .enabled_features(&enabled_core_features)
                .push_next(&mut timeline_semaphore_features)
                .push_next(&mut synchronization2_features)
                .push_next(&mut buffer_device_address_features)
                .push_next(&mut shader_float16_int8_features)
                .push_next(&mut storage16_features)
                .push_next(&mut storage8_features)
                .push_next(&mut integer_dot_product_features)
                .push_next(&mut vulkan_memory_model_features);
            if shader_float8_support.shader_float8
                || shader_float8_support.shader_float8_cooperative_matrix
            {
                shader_float8_features.shader_float8 = bool32(shader_float8_support.shader_float8);
                shader_float8_features.shader_float8_cooperative_matrix = bool32(
                    shader_float8_support.shader_float8_cooperative_matrix
                        && cooperative_matrix_supported,
                );
                extension_names.push(VK_EXT_SHADER_FLOAT8_NAME.as_ptr());
                enabled_device_extensions
                    .insert(VK_EXT_SHADER_FLOAT8_NAME.to_string_lossy().into_owned());
            }
            if cooperative_matrix_supported {
                cooperative_matrix_features.cooperative_matrix = vk::TRUE;
                extension_names.push(ash::khr::cooperative_matrix::NAME.as_ptr());
                enabled_device_extensions.insert(
                    ash::khr::cooperative_matrix::NAME
                        .to_string_lossy()
                        .into_owned(),
                );
            }
            if shader_bfloat16_support.shader_bfloat16_type
                || shader_bfloat16_support.shader_bfloat16_dot_product
                || shader_bfloat16_support.shader_bfloat16_cooperative_matrix
            {
                shader_bfloat16_features.shader_bfloat16_type =
                    bool32(shader_bfloat16_support.shader_bfloat16_type);
                shader_bfloat16_features.shader_bfloat16_dot_product =
                    bool32(shader_bfloat16_support.shader_bfloat16_dot_product);
                shader_bfloat16_features.shader_bfloat16_cooperative_matrix = bool32(
                    shader_bfloat16_support.shader_bfloat16_cooperative_matrix
                        && cooperative_matrix_supported,
                );
                extension_names.push(VK_KHR_SHADER_BFLOAT16_NAME.as_ptr());
                enabled_device_extensions
                    .insert(VK_KHR_SHADER_BFLOAT16_NAME.to_string_lossy().into_owned());
            }
            if mixed_float_dot_product_support.shader_float16_acc_float32
                || mixed_float_dot_product_support.shader_float16_acc_float16
                || mixed_float_dot_product_support.shader_bfloat16_acc
                || mixed_float_dot_product_support.shader_float8_acc_float32
            {
                mixed_float_dot_product_features.shader_float16_acc_float32 =
                    bool32(mixed_float_dot_product_support.shader_float16_acc_float32);
                mixed_float_dot_product_features.shader_float16_acc_float16 =
                    bool32(mixed_float_dot_product_support.shader_float16_acc_float16);
                mixed_float_dot_product_features.shader_bfloat16_acc =
                    bool32(mixed_float_dot_product_support.shader_bfloat16_acc);
                mixed_float_dot_product_features.shader_float8_acc_float32 =
                    bool32(mixed_float_dot_product_support.shader_float8_acc_float32);
                extension_names.push(VK_VALVE_SHADER_MIXED_FLOAT_DOT_PRODUCT_NAME.as_ptr());
                enabled_device_extensions.insert(
                    VK_VALVE_SHADER_MIXED_FLOAT_DOT_PRODUCT_NAME
                        .to_string_lossy()
                        .into_owned(),
                );
            }
            if shared_host_memory_alignment.is_some() {
                extension_names.push(ash::ext::external_memory_host::NAME.as_ptr());
                enabled_device_extensions.insert(
                    ash::ext::external_memory_host::NAME
                        .to_string_lossy()
                        .into_owned(),
                );
            }
            if shared_device_memory_supported {
                extension_names.push(ash::khr::external_memory_fd::NAME.as_ptr());
                extension_names.push(ash::ext::external_memory_dma_buf::NAME.as_ptr());
                enabled_device_extensions.insert(
                    ash::khr::external_memory_fd::NAME
                        .to_string_lossy()
                        .into_owned(),
                );
                enabled_device_extensions.insert(
                    ash::ext::external_memory_dma_buf::NAME
                        .to_string_lossy()
                        .into_owned(),
                );
            }
            if opaque_fd_timeline_semaphore_supported {
                extension_names.push(ash::khr::external_semaphore_fd::NAME.as_ptr());
                enabled_device_extensions.insert(
                    ash::khr::external_semaphore_fd::NAME
                        .to_string_lossy()
                        .into_owned(),
                );
            }
            if conditional_rendering_supported {
                extension_names
                    .push(ash::ext::conditional_rendering::NAME.as_ptr());
                enabled_device_extensions.insert(
                    ash::ext::conditional_rendering::NAME
                        .to_string_lossy()
                        .into_owned(),
                );
                device_info =
                    device_info.push_next(&mut conditional_rendering_features);
            }
            if device_fault_supported {
                extension_names.push(ash::ext::device_fault::NAME.as_ptr());
                enabled_device_extensions.insert(
                    ash::ext::device_fault::NAME.to_string_lossy().into_owned(),
                );
                device_info = device_info.push_next(&mut device_fault_features);
            }
            if device_coherent_memory_supported {
                extension_names
                    .push(ash::amd::device_coherent_memory::NAME.as_ptr());
                enabled_device_extensions.insert(
                    ash::amd::device_coherent_memory::NAME
                        .to_string_lossy()
                        .into_owned(),
                );
                device_info =
                    device_info.push_next(&mut device_coherent_memory_features);
            }
            if shader_float8_support.shader_float8
                || shader_float8_support.shader_float8_cooperative_matrix
            {
                shader_float8_features.p_next = device_info.p_next.cast_mut();
                device_info.p_next = std::ptr::from_ref(&shader_float8_features).cast();
            }
            if shader_bfloat16_support.shader_bfloat16_type
                || shader_bfloat16_support.shader_bfloat16_dot_product
                || shader_bfloat16_support.shader_bfloat16_cooperative_matrix
            {
                shader_bfloat16_features.p_next = device_info.p_next.cast_mut();
                device_info.p_next = std::ptr::from_ref(&shader_bfloat16_features).cast();
            }
            if mixed_float_dot_product_support.shader_float16_acc_float32
                || mixed_float_dot_product_support.shader_float16_acc_float16
                || mixed_float_dot_product_support.shader_bfloat16_acc
                || mixed_float_dot_product_support.shader_float8_acc_float32
            {
                mixed_float_dot_product_features.p_next = device_info.p_next.cast_mut();
                device_info.p_next = std::ptr::from_ref(&mixed_float_dot_product_features).cast();
            }
            if cooperative_matrix_supported {
                cooperative_matrix_features.p_next = device_info.p_next.cast_mut();
                device_info.p_next = std::ptr::from_ref(&cooperative_matrix_features).cast();
            }
            device_info = device_info.enabled_extension_names(&extension_names);
            let device = instance
                .create_device(physical_device, &device_info, None)
                .map_err(|error| {
                    VulkanError(format!("failed to create Vulkan device: {error:?}"))
                })?;
            let conditional_rendering = conditional_rendering_supported.then(|| {
                ash::ext::conditional_rendering::Device::new(instance, &device)
            });
            let device_fault = device_fault_supported.then(|| {
                ash::ext::device_fault::Device::new(instance, &device)
            });
            let queue = device.get_device_queue(queue_family_index, 0);
            let transfer_queue_is_distinct = queue_priorities.len() >= 2;
            let transfer_queue = device.get_device_queue(
                queue_family_index,
                u32::from(transfer_queue_is_distinct),
            );
            let physical_device_properties =
                instance.get_physical_device_properties(physical_device);
            let limits = physical_device_properties.limits;
            let min_storage_buffer_offset_alignment =
                usize::try_from(limits.min_storage_buffer_offset_alignment).map_err(|_| {
                    VulkanError("Vulkan storage-buffer offset alignment exceeds usize".to_string())
                })?;
            let subgroup_support = physical_device_subgroup_support(instance, physical_device);
            let subgroup_size = subgroup_support.subgroup_size;
            let pci_address =
                physical_device_pci_address(instance, physical_device);
            let (activity_lease, activity_lease_health) =
                if permitted_device.vendor_id == AMD_PCI_VENDOR_ID {
                    let (render_major, render_minor) =
                        match physical_device_drm_render_node(instance, physical_device) {
                            Ok(render_node) => render_node,
                            Err(error) => {
                                device.destroy_device(None);
                                return Err(error);
                            }
                        };
                    let lease = match VulkanDeviceActivityLease::start_linux_drm(
                        Arc::<str>::from(permitted_device.physical_device_id.clone()),
                        render_major,
                        render_minor,
                    ) {
                        Ok(lease) => lease,
                        Err(error) => {
                            device.destroy_device(None);
                            return Err(error);
                        }
                    };
                    let health = lease.health();
                    (Some(lease), health)
                } else {
                    (
                        None,
                        VulkanDeviceActivityLeaseHealth::inactive(Arc::<str>::from(
                            permitted_device.physical_device_id.clone(),
                        )),
                    )
                };

            Ok(VulkanComputeDevice {
                context: Arc::clone(&self.context),
                physical_device,
                device,
                queue_family_index,
                queue,
                transfer_queue,
                transfer_queue_is_distinct,
                activity_lease: RefCell::new(activity_lease),
                activity_lease_health,
                buffer_device_address_supported,
                api_version: physical_device_properties.api_version,
                physical_device_id: permitted_device.physical_device_id.clone(),
                device_name,
                pci_address,
                enabled_device_extensions,
                enabled_shader_features,
                shared_host_memory_alignment,
                shared_device_memory_supported,
                opaque_fd_timeline_semaphore_supported,
                cooperative_bfloat16_shapes,
                cooperative_float8_e4m3_shapes,
                cooperative_sint8_shapes,
                subgroup_size,
                subgroup_supported_stages: subgroup_support.supported_stages,
                subgroup_supported_operations: subgroup_support.supported_operations,
                max_compute_work_group_invocations: limits.max_compute_work_group_invocations,
                max_compute_work_group_size_x: limits.max_compute_work_group_size[0],
                max_compute_work_group_count_x: limits.max_compute_work_group_count[0],
                min_storage_buffer_offset_alignment,
                device_local_memory_bytes,
                memory_budget_supported,
                timestamp_period_ns: limits.timestamp_period,
                conditional_rendering,
                device_fault,
                device_address_registry: Arc::new(Mutex::new(
                    VulkanDeviceAddressRegistry::default(),
                )),
                generic_storage_pipelines: RefCell::new(HashMap::new()),
                immediate_kernel_sequence: RefCell::new(None),
            })
        }
    }
}

fn allowed_physical_device_indices(
    discovered_physical_device_ids: &[(usize, String)],
    allowed_physical_device_ids: Option<&BTreeSet<String>>,
) -> Result<Vec<usize>, VulkanError> {
    let Some(allowed) = allowed_physical_device_ids else {
        return Ok(discovered_physical_device_ids
            .iter()
            .map(|(physical_device_index, _)| *physical_device_index)
            .collect());
    };
    let discovered = discovered_physical_device_ids
        .iter()
        .map(|(_, physical_device_id)| physical_device_id.clone())
        .collect::<BTreeSet<_>>();
    let missing = allowed.difference(&discovered).cloned().collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(VulkanError(format!(
            "allowed Vulkan physical devices are not present: {}",
            missing.join(", ")
        )));
    }
    Ok(discovered_physical_device_ids
        .iter()
        .filter_map(|(physical_device_index, physical_device_id)| {
            allowed
                .contains(physical_device_id)
                .then_some(*physical_device_index)
        })
        .collect())
}
