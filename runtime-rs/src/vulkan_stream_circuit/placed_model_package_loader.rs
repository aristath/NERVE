#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct VulkanDemandPagedSharedTierCapacities {
    store_payload_bytes: usize,
    device_payload_bytes: usize,
    host_visible_payload_bytes: usize,
}

#[cfg(test)]
fn physical_execution_stream_working_set_bytes(
    plan: &VulkanRuntimePhysicalExecutionResidencyPlan,
    logical_device_ids: &BTreeSet<String>,
) -> Result<usize, VulkanResidentTokenModelPackageError> {
    logical_device_ids.iter().try_fold(0usize, |total, device_id| {
        let device = plan
            .device_plans
            .iter()
            .find(|device| device.device_id == *device_id)
            .ok_or_else(|| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "physical execution residency has no stream plan for selected-resource slice {device_id:?}",
                ))
            })?;
        total
            .checked_add(device.stream_device_local_bytes)
            .ok_or_else(|| {
                VulkanResidentTokenModelPackageError::new(
                    "physical stream working-set accounting overflowed",
                )
            })
    })
}

fn physical_execution_store_pending_fixed_bytes(
    physical_stream_requirement_bytes_by_physical_device: &BTreeMap<String, usize>,
    physical_device_id: &str,
    store_fixed_bytes: usize,
) -> Result<usize, VulkanResidentTokenModelPackageError> {
    let stream_bytes = physical_stream_requirement_bytes_by_physical_device
        .get(physical_device_id)
        .copied()
        .ok_or_else(|| {
            VulkanResidentTokenModelPackageError::new(format!(
                "physical selected-resource store {physical_device_id:?} has no exact stream requirement",
            ))
        })?;
    stream_bytes.checked_add(store_fixed_bytes).ok_or_else(|| {
        VulkanResidentTokenModelPackageError::new(
            "pending physical residency accounting overflowed",
        )
    })
}

fn demand_paged_shared_tier_capacities(
    maximum_store_payload_bytes: usize,
    device_payload_bytes: usize,
    shared_host_physical_bytes: usize,
) -> Result<VulkanDemandPagedSharedTierCapacities, VulkanResidentTokenModelPackageError> {
    if maximum_store_payload_bytes == 0
        || device_payload_bytes == 0
        || device_payload_bytes >= maximum_store_payload_bytes
        || shared_host_physical_bytes == 0
    {
        return Err(VulkanResidentTokenModelPackageError::new(
            "demand-paged shared-tier capacity has an invalid store, device, or host bound",
        ));
    }
    let store_payload_bytes = device_payload_bytes
        .checked_add(shared_host_physical_bytes)
        .unwrap_or(usize::MAX)
        .min(maximum_store_payload_bytes);
    Ok(VulkanDemandPagedSharedTierCapacities {
        store_payload_bytes,
        device_payload_bytes,
        // This is a logical per-store admission limit, not a reservation. A
        // store may need more than its nominal overflow when physical device
        // fragmentation leaves part of its device payload budget unusable.
        // The shared cache independently enforces the one physical host-memory
        // hard bound across every store.
        host_visible_payload_bytes: maximum_store_payload_bytes.min(shared_host_physical_bytes),
    })
}

impl VulkanResidentInProcessPlacedModelPackage {
    fn from_runtime_model_for_device_resolver<'a, F>(
        manifest_dir: impl AsRef<Path>,
        runtime_model: VulkanResidentRuntimeModel,
        physical_execution_plan: Option<VulkanRuntimePhysicalExecutionPlan>,
        placement_calibration_catalog: Option<&VulkanPlacementCalibrationCatalog>,
        dynamic_state_capacity_activations: Option<usize>,
        speculative_draft_tokens: usize,
        resource_residency_policy: ResourceResidencyPolicy,
        parameter_pool: Option<&VulkanResidentBufferPool>,
        retained_stores: Option<&VulkanRetainedCompiledResourceStores>,
        device_for: F,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError>
    where
        F: Fn(&str) -> Result<&'a VulkanComputeDevice, VulkanResidentInProcessPlacedRuntimeError>,
    {
        let manifest_dir = manifest_dir.as_ref();
        let mount_speculative_decoders = speculative_draft_tokens > 0;
        let package_id = runtime_model.package.package_id.clone();
        let activation_element_bytes = runtime_model
            .package
            .activation_element_bytes
            .filter(|bytes| *bytes > 0)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(
                        "distributed execution requires a compiled activation element width",
                    ),
                )
            })?;
        let execution_scope = runtime_model.execution_scope.clone();
        if execution_scope.trim().is_empty() {
            return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(
                    "resident runtime execution scope must not be empty",
                ),
            ));
        }
        let (input_processor_id, output_processor_id) = runtime_model
            .circuit_graph
            .signal_processor_endpoint_component_ids()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?;
        let input_device_id = runtime_model
            .placement
            .device_for_component(&input_processor_id)
            .to_string();
        let output_device_id = runtime_model
            .placement
            .device_for_component(&output_processor_id)
            .to_string();
        let capacity = dynamic_state_capacity_activations
            .unwrap_or(runtime_model.package.max_context_activations);
        if capacity == 0 {
            return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(
                    "resident dynamic state capacity must be at least 1 activation",
                ),
            ));
        }
        let physical_execution_plan = physical_execution_plan
            .unwrap_or_else(|| VulkanRuntimePhysicalExecutionPlan::uniform(&runtime_model));
        physical_execution_plan
            .validate(&runtime_model)
            .map_err(|error| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(error.to_string()),
                )
            })?;
        let base_runtime_execution_identity = canonical_runtime_execution_identity(
            &runtime_model,
            &physical_execution_plan,
            capacity,
            mount_speculative_decoders,
            resource_residency_policy,
        )
        .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?;
        let tensor_index = runtime_model.load_runtime_tensor_index(manifest_dir)?;
        let compiled_resource_contract = Arc::new(
            instantiate_runtime_resource_contract(&runtime_model).map_err(|error| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(format!(
                        "failed to instantiate runtime resource contract: {error}"
                    )),
                )
            })?,
        );
        let residency_plan = plan_vulkan_runtime_residency_with_contract(
            manifest_dir,
            &runtime_model,
            &tensor_index,
            capacity,
            speculative_draft_tokens,
            resource_residency_policy,
            &compiled_resource_contract,
        )
        .map_err(|error| {
            VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to plan compiled resources for {resource_residency_policy:?} residency: {error}"
                )),
            )
        })?;
        let (resource_plan, placement_plan, _boundary_placed_plan) =
            plan_resident_package_placed_stream_circuit_with_tensor_index(
                &input_device_id,
                &runtime_model.placement,
                &runtime_model.circuit_graph,
                manifest_dir,
                &tensor_index,
                runtime_model.package.activation_element_bytes,
            )?;
        let input_transducer_spirv_words = load_required_resident_model_package_shader(
            manifest_dir,
            &runtime_model.package.input_transducer.shader_path,
        )?;
        let input_transducer_batch_spirv_words = load_required_resident_model_package_shader(
            manifest_dir,
            &runtime_model.package.input_transducer.batch_shader_path,
        )?;
        let embedding_norm_spirv_words = load_required_resident_model_package_shader(
            manifest_dir,
            &runtime_model
                .package
                .output_transducer
                .embedding_norm_shader_path,
        )?;
        let embedding_norm_batch_spirv_words = load_required_resident_model_package_shader(
            manifest_dir,
            &runtime_model
                .package
                .output_transducer
                .embedding_norm_batch_shader_path,
        )?;
        let tied_projection_spirv_words = load_required_resident_model_package_shader(
            manifest_dir,
            &runtime_model
                .package
                .output_transducer
                .projection_shader_path,
        )?;
        let tied_projection_batch_spirv_words = load_required_resident_model_package_shader(
            manifest_dir,
            &runtime_model
                .package
                .output_transducer
                .projection_batch_shader_path,
        )?;
        let sampler_kernels =
            load_resident_sampler_kernels(manifest_dir, &runtime_model.package.sampler)?;
        let device_ids = physical_execution_plan.device_ids(&runtime_model);
        let owner_device_ids = runtime_model
            .circuit_graph
            .signal_processor_owner_device_ids(&runtime_model.placement);
        let mut device_slice_plans = Vec::with_capacity(owner_device_ids.len());
        let mut hosted_component_count = 0usize;

        for device_id in &owner_device_ids {
            let slice_device = device_for(device_id)?;
            let package_slice = VulkanResidentModelPackageDeviceSlicePlan::prepare(
                slice_device,
                manifest_dir,
                &runtime_model,
                &compiled_resource_contract,
                &tensor_index,
                device_id,
                capacity,
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?;
            hosted_component_count = hosted_component_count
                .checked_add(package_slice.hosted_component_count)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::Package(
                        VulkanResidentTokenModelPackageError::new(
                            "placed package hosted component count overflowed",
                        ),
                    )
                })?;
            device_slice_plans.push(package_slice);
        }
        validate_physical_residency_schedule_coverage(
            &compiled_resource_contract,
            &execution_scope,
            device_slice_plans
                .iter()
                .map(|slice| &slice.physical_residency_schedule),
        )
        .map_err(|error| {
            VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to validate placed physical residency checkpoints: {error}"
                )),
            )
        })?;

        let mut speculative_decoder_slice_plans = BTreeMap::new();
        if mount_speculative_decoders {
            let decoder_device = device_for(&output_device_id)?;
            for decoder in &runtime_model.package.speculative_decoders {
                let plan = VulkanResidentSpeculativeDecoderModelPackage::prepare_device_slice(
                    decoder_device,
                    manifest_dir,
                    &runtime_model,
                    &tensor_index,
                    decoder,
                    &output_device_id,
                    capacity,
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?;
                if speculative_decoder_slice_plans
                    .insert(decoder.id.clone(), plan)
                    .is_some()
                {
                    return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                        VulkanResidentTokenModelPackageError::new(format!(
                            "resident package repeats speculative decoder {:?}",
                            decoder.id,
                        )),
                    ));
                }
            }
        }

        let prepared_plans = device_slice_plans
            .iter()
            .map(|slice| (slice.device_id.as_str(), &slice.prepared_plan))
            .collect::<Vec<_>>();
        let distributed_loaded_manifest =
            resident_package_loaded_kernel_manifest_for_slice_plans(&device_slice_plans)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?;
        let distributed_artifact_manifest = VulkanPhysicalKernelArtifactManifest::new(
            distributed_loaded_manifest
                .physical_artifacts
                .iter()
                .map(|artifact| artifact.artifact.clone())
                .collect(),
        );
        let storage_buffer_offset_alignment = device_ids
            .iter()
            .map(|device_id| {
                device_for(device_id).map(VulkanComputeDevice::min_storage_buffer_offset_alignment)
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .unwrap_or(1);
        let device_execution_identity_by_logical_device = device_ids
            .iter()
            .map(|device_id| {
                let device = device_for(device_id)?;
                Ok((
                    device_id.clone(),
                    VulkanPlacementDeviceExecutionIdentity {
                        physical_device_id: device.physical_device_id().to_string(),
                        api_version: device.api_version(),
                        driver_version: device.driver_version(),
                    },
                ))
            })
            .collect::<Result<BTreeMap<_, _>, VulkanResidentInProcessPlacedRuntimeError>>()?;
        physical_execution_plan
            .validate_bound_boundary_device_identities(
                &device_execution_identity_by_logical_device,
            )
            .map_err(|error| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(error.to_string()),
                )
            })?;
        let mut distributed_execution_plans =
            VulkanDistributedExecutionPlanSet::from_prepared_plans_with_resource_contract_and_execution_cases(
                &prepared_plans,
                &tensor_index,
                &distributed_artifact_manifest,
                &physical_execution_plan.component_device_pools,
                &placement_plan.edges,
                storage_buffer_offset_alignment,
                activation_element_bytes,
                &execution_scope,
                &compiled_resource_contract,
                &physical_execution_plan.decode_execution_cases_by_component,
                &physical_execution_plan.decode_batch_execution_cases_by_component,
                &physical_execution_plan.prefill_execution_cases_by_component,
                &physical_execution_plan.decode_contract_ids_by_component,
                &physical_execution_plan.decode_batch_contract_ids_by_component,
                &physical_execution_plan.prefill_contract_ids_by_component,
            )
            .map_err(|error| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(format!(
                        "failed to lower distributed Vulkan phase and shape plans: {error}"
                    )),
                )
            })?;
        distributed_execution_plans
            .apply_exact_execution_cases(
                &physical_execution_plan.decode_execution_cases_by_component,
                &physical_execution_plan.decode_batch_execution_cases_by_component,
                &physical_execution_plan.prefill_execution_cases_by_component,
                &device_execution_identity_by_logical_device,
                &distributed_loaded_manifest,
            )
            .map_err(|error| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(format!(
                        "failed to replay exact calibrated physical execution cases: {error}"
                    )),
                )
            })?;
        let mut physical_device_by_logical_device = BTreeMap::new();
        let mut safe_capacity_by_physical_device = BTreeMap::new();
        let mut selected_resource_mount_devices = Vec::with_capacity(device_ids.len());
        for device_id in &device_ids {
            let device = device_for(device_id)?;
            let physical_device_id = device.physical_device_id().to_string();
            physical_device_by_logical_device.insert(device_id.clone(), physical_device_id.clone());
            let safe_capacity =
                usize::try_from(device.device_local_memory_budget().reservable_bytes)
                    .unwrap_or(usize::MAX);
            safe_capacity_by_physical_device
                .entry(physical_device_id)
                .and_modify(|capacity: &mut usize| *capacity = (*capacity).min(safe_capacity))
                .or_insert(safe_capacity);
            let upload_alignment =
                compiled_resource_upload_alignment(&compiled_resource_contract, device)
                    .map_err(|error| {
                        VulkanResidentInProcessPlacedRuntimeError::Package(
                            VulkanResidentTokenModelPackageError::new(error.to_string()),
                        )
                    })?;
            selected_resource_mount_devices.push(VulkanRuntimeSelectedResourceMountDevice {
                logical_device_id: device_id.clone(),
                physical_device_id: device.physical_device_id().to_string(),
                execution_identity: device_execution_identity_by_logical_device[device_id].clone(),
                live_safe_capacity_bytes: safe_capacity,
                upload_alignment,
            });
        }
        let host_safe_capacity_bytes = vulkan_safe_host_available_bytes()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?;
        let edge_plans = device_slice_plans
            .iter()
            .map(|slice| {
                VulkanPlacedEdgeIoPlan::from_placed_resident_plan(
                    &slice.placed_plan.placed_resident_plan,
                )
                .map_err(|error| {
                    VulkanResidentInProcessPlacedRuntimeError::Package(
                        VulkanResidentTokenModelPackageError::new(format!(
                            "failed to plan exact graph-edge residency: {error}",
                        )),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let workload_free_boundary_routes = plan_vulkan_workload_free_graph_edge_routes(
            &edge_plans,
            &physical_device_by_logical_device,
        )
        .map_err(|error| {
            VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to plan workload-free graph-edge routes: {error}",
                )),
            )
        })?;
        let exact_boundary_routes = physical_execution_plan
            .mounted_boundary_routes()
            .map_err(|error| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(error.to_string()),
                )
            })?;
        let mounted_boundary_routes = compose_vulkan_runtime_mounted_boundary_routes(
            workload_free_boundary_routes,
            exact_boundary_routes,
        );
        let Some(transient_resolution) =
            try_resolve_vulkan_runtime_selected_resources_with_exact_execution_transients(
            &runtime_model,
            &compiled_resource_contract,
            &distributed_loaded_manifest,
            &distributed_execution_plans,
            &residency_plan,
            &device_ids,
            &device_slice_plans,
            &speculative_decoder_slice_plans,
            &edge_plans,
            &mounted_boundary_routes,
            &tensor_index,
            &selected_resource_mount_devices,
            &input_device_id,
            &output_device_id,
            physical_execution_plan.prefill_activation_batch_width,
            speculative_draft_tokens,
            resource_residency_policy,
            placement_calibration_catalog,
            None,
            host_safe_capacity_bytes,
        )
        .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?
        else {
            return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(
                    "no compiled normal-prompt lane capacity fits the live physical mount",
                ),
            ));
        };
        let normal_prefill_lane_capacity = transient_resolution.normal_prefill_lane_capacity;
        let selected_resource_resolution = transient_resolution.resolution;
        let selected_resource_placements = selected_resource_resolution.placements;
        let runtime_execution_identity = canonical_mounted_runtime_execution_identity(
            &base_runtime_execution_identity,
            &selected_resource_placements,
        )?;
        let VulkanRuntimeDistributedMountPlans {
            execution_plans: distributed_execution_plans,
            activation_plan: distributed_activation_plan,
            parameter_allocation_plan: distributed_parameter_allocation_plan,
            parameter_exclusion_plan: distributed_parameter_exclusion_plan,
            selected_resource_execution_ownership_plan:
                distributed_selected_resource_execution_ownership_plan,
            selected_resource_store_plan: distributed_selected_resource_store_plan,
            physical_execution_residency_plan,
        } = selected_resource_resolution.plans;
        admit_vulkan_runtime_physical_execution_mount(
            &physical_execution_residency_plan,
            &physical_device_by_logical_device,
            &safe_capacity_by_physical_device,
        )
        .map_err(|error| {
            VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed exact physical execution mount admission: {error}"
                )),
            )
        })?;
        let physical_stream_requirement_bytes_by_physical_device =
            physical_execution_stream_device_requirement_bytes_by_physical_device(
                &physical_execution_residency_plan,
                &device_for,
            )?;

        let input_device = device_for(&input_device_id)?;
        let output_device = device_for(&output_device_id)?;
        let (input_transducer_parameter_buffers, output_transducer_parameter_buffers) =
            if input_device_id == output_device_id {
                let shared = Arc::new(load_resident_package_transducer_parameter_buffers(
                    input_device,
                    &input_device_id,
                    &resource_plan,
                    &tensor_index,
                    parameter_pool,
                )?);
                (shared.clone(), shared)
            } else {
                (
                    Arc::new(load_resident_package_transducer_parameter_buffers_for(
                        input_device,
                        &input_device_id,
                        &resource_plan,
                        &tensor_index,
                        "input_transducer",
                        parameter_pool,
                    )?),
                    Arc::new(load_resident_package_transducer_parameter_buffers_for(
                        output_device,
                        &output_device_id,
                        &resource_plan,
                        &tensor_index,
                        "output_transducer",
                        parameter_pool,
                    )?),
                )
            };
        let distributed_parameter_buffers = Arc::new(
            match parameter_pool {
                Some(pool) => VulkanDistributedParameterBuffers::allocate_and_load_from_pool(
                    &distributed_parameter_allocation_plan,
                    &tensor_index,
                    pool,
                ),
                None => VulkanDistributedParameterBuffers::allocate_and_load(
                    &distributed_parameter_allocation_plan,
                    &tensor_index,
                    |device_id| device_for(device_id),
                ),
            }
            .map_err(|error| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(format!(
                        "failed to allocate distributed Vulkan parameter shards: {error}"
                    )),
                )
            })?,
        );

        let mut device_slices = Vec::with_capacity(device_slice_plans.len());
        for package_slice in device_slice_plans {
            let slice_device = device_for(&package_slice.device_id)?;
            let excluded_tensors =
                distributed_parameter_exclusion_plan.tensors_for_device(&package_slice.device_id);
            let package_slice = package_slice
                .materialize(
                    slice_device,
                    &tensor_index,
                    &excluded_tensors,
                    parameter_pool,
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?;
            device_slices.push(package_slice);
        }

        let compiled_resource_layout = Arc::new(
            VulkanCompiledResourceAddressLayout::from_contract(&compiled_resource_contract)
                .map_err(|error| {
                    VulkanResidentInProcessPlacedRuntimeError::Package(
                        VulkanResidentTokenModelPackageError::new(format!(
                            "failed to lower compiled resource address layout: {error}"
                        )),
                    )
                })?,
        );
        let maximum_ranges_per_group = compiled_resource_maximum_ranges_per_group(
            &compiled_resource_contract,
        )
        .map_err(|error| {
            VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(error.to_string()),
            )
        })?;
        let logical_devices = device_ids
            .iter()
            .map(|device_id| device_for(device_id).map(|device| (device_id.clone(), device)))
            .collect::<Result<Vec<_>, _>>()?;
        let physical_device_groups = group_compiled_resource_logical_devices_by_physical(
            &logical_devices,
        )
        .map_err(|error| {
            VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(error.to_string()),
            )
        })?;

        let mut compiled_resource_device_stores = BTreeMap::new();
        let mut compiled_resource_physical_placements = Vec::new();
        let mut planned_selector_physical_placements =
            Vec::<(VulkanCompiledSelectorAddressMapping, String)>::new();
        let mut remaining_safe_host_visible_payload_bytes = None;
        let mut shared_host_cache = retained_stores
            .map(VulkanRetainedCompiledResourceStores::shared_host_cache)
            .transpose()
            .map_err(|error| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(error),
                )
            })?
            .flatten();
        let mut distributed_dynamic_resource_buffers = BTreeMap::new();
        for logical_device_id_list in physical_device_groups {
            let logical_device_ids = logical_device_id_list
                .iter()
                .cloned()
                .collect::<BTreeSet<_>>();
            let Some(selector_ownership) = compiled_resource_selector_ownership_for_device_set(
                &runtime_model,
                &compiled_resource_contract,
                &input_device_id,
                &output_device_id,
                &logical_device_ids,
                mount_speculative_decoders,
                &distributed_selected_resource_store_plan,
            )
            .map_err(|error| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(format!(
                        "failed to plan physical compiled-resource ownership: {error}"
                    )),
                )
            })?
            else {
                continue;
            };
            let physical_parameters =
                plan_compiled_parameter_residency_for_device_set_with_selector_ownership(
                    &runtime_model,
                    &compiled_resource_contract,
                    &input_device_id,
                    &output_device_id,
                    &logical_device_ids,
                    mount_speculative_decoders,
                    resource_residency_policy,
                    &selector_ownership,
                )
                .map_err(|error| {
                    VulkanResidentInProcessPlacedRuntimeError::Package(
                        VulkanResidentTokenModelPackageError::new(format!(
                            "failed to size physical compiled-resource ownership: {error}"
                        )),
                    )
                })?;
            let maximum_dynamic_bytes = physical_parameters
                .maximum_addressable_bytes
                .checked_sub(physical_parameters.always_resident_bytes)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::Package(
                        VulkanResidentTokenModelPackageError::new(
                            "physical dynamic parameter accounting underflowed",
                        ),
                    )
                })?;
            if maximum_dynamic_bytes == 0 {
                continue;
            }
            if physical_parameters.staging_headroom_bytes == 0 {
                return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(format!(
                        "physical device for logical slices {logical_device_ids:?} has selected parameters but no dynamic residency capacity"
                    )),
                ));
            }
            let allowed_selector_ids = selector_ownership.selector_ids();
            let representative_device_id = logical_device_id_list[0].clone();
            let physical_device = device_for(&representative_device_id)?;
            let physical_device_id = physical_device.physical_device_id().to_string();
            let working_set_bytes = physical_stream_requirement_bytes_by_physical_device
                .get(&physical_device_id)
                .copied()
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::Package(
                        VulkanResidentTokenModelPackageError::new(format!(
                            "physical selected-resource store {physical_device_id:?} has no exact stream requirement",
                        )),
                    )
                })?;
            let upload_alignment =
                compiled_resource_upload_alignment(&compiled_resource_contract, physical_device)
                    .map_err(|error| {
                        VulkanResidentInProcessPlacedRuntimeError::Package(
                            VulkanResidentTokenModelPackageError::new(error.to_string()),
                        )
                    })?;
            let store_residency = plan_compiled_resource_store_residency_for_ownership(
                &compiled_resource_contract,
                &compiled_resource_layout,
                &selector_ownership,
                physical_parameters.staging_headroom_bytes,
                upload_alignment,
            )
            .map_err(|error| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(error.to_string()),
                )
            })?;
            let components_by_scope = compiled_resource_store_components_by_scope(
                &compiled_resource_contract,
                &allowed_selector_ids,
            )
            .map_err(|error| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(error.to_string()),
                )
            })?;
            let pending_fixed_bytes = physical_execution_store_pending_fixed_bytes(
                &physical_stream_requirement_bytes_by_physical_device,
                &physical_device_id,
                store_residency.fixed_device_bytes().map_err(|error| {
                    VulkanResidentInProcessPlacedRuntimeError::Package(
                        VulkanResidentTokenModelPackageError::new(error.to_string()),
                    )
                })?,
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?;
            let store_id =
                format!("{package_id}:physical_store:{physical_device_id}:{execution_scope}");
            let logical_device_id_list = logical_device_ids.iter().cloned().collect::<Vec<_>>();
            let retained_store = retained_stores
                .map(|stores| stores.store_for_logical_devices(&logical_device_id_list))
                .transpose()
                .map_err(|error| {
                    VulkanResidentInProcessPlacedRuntimeError::Package(
                        VulkanResidentTokenModelPackageError::new(error),
                    )
                })?
                .flatten();
            let (store, reused_store) = if let Some(store) = retained_store {
                let compatible = store
                    .is_compatible_with_mount(
                        resource_residency_policy,
                        &store_id,
                        &physical_device_id,
                        &logical_device_id_list,
                        manifest_dir,
                        &compiled_resource_contract,
                        &compiled_resource_layout,
                        &selector_ownership,
                        physical_parameters.staging_headroom_bytes,
                        maximum_ranges_per_group,
                        physical_parameters.always_resident_bytes,
                        working_set_bytes,
                        store_residency.metadata_device_bytes,
                    )
                    .map_err(|error| {
                        VulkanResidentInProcessPlacedRuntimeError::Package(
                            VulkanResidentTokenModelPackageError::new(error.to_string()),
                        )
                    })?;
                if !compatible {
                    return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                        VulkanResidentTokenModelPackageError::new(format!(
                            "retained compiled-resource store {store_id:?} is incompatible with the requested remount",
                        )),
                    ));
                }
                let capacities = store.retained_mount_capacities();
                if capacities.store_payload_bytes == 0
                    || capacities.device_payload_bytes == 0
                    || capacities.available_dynamic_device_bytes == 0
                {
                    return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                        VulkanResidentTokenModelPackageError::new(format!(
                            "retained compiled-resource store {store_id:?} has invalid capacities",
                        )),
                    ));
                }
                (store, true)
            } else {
                let admission = physical_device
                    .admit_device_local_memory(
                        u64::try_from(pending_fixed_bytes).unwrap_or(u64::MAX),
                    )
                    .map_err(|error| {
                        VulkanResidentInProcessPlacedRuntimeError::Package(
                            VulkanResidentTokenModelPackageError::new(format!(
                                "failed stable capacity admission for physical slices {logical_device_ids:?}: {error}"
                            )),
                        )
                    })?;
                let safe_dynamic_bytes =
                    usize::try_from(admission.allocatable_bytes).unwrap_or(usize::MAX);
                let maximum_alignment_padding =
                    store_residency.maximum_dynamic_allocation_padding_bytes;
                let retained_representation_cache_bytes = store_residency
                    .retained_representation_cache_device_bytes()
                    .map_err(|error| {
                        VulkanResidentInProcessPlacedRuntimeError::Package(
                            VulkanResidentTokenModelPackageError::new(error.to_string()),
                        )
                    })?;
                let resident_payload_capacity = compiled_resource_source_payload_capacity(
                    maximum_dynamic_bytes,
                    safe_dynamic_bytes,
                    &store_residency,
                )
                .map_err(|error| {
                    VulkanResidentInProcessPlacedRuntimeError::Package(
                        VulkanResidentTokenModelPackageError::new(error.to_string()),
                    )
                })?;
                if resident_payload_capacity < store_residency.maximum_load_wave_payload_bytes {
                    return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                        VulkanResidentTokenModelPackageError::new(format!(
                            "stable device-local capacity for physical slices {logical_device_ids:?} admits {resident_payload_capacity} source payload bytes after reserving {retained_representation_cache_bytes} representation-cache bytes, but one complete selector load wave needs {} source bytes",
                            store_residency.maximum_load_wave_payload_bytes,
                        )),
                    ));
                }
                let (
                    store_payload_capacity,
                    device_payload_capacity,
                    host_visible_payload_capacity,
                    store_shared_host_cache,
                ) = if maximum_dynamic_bytes > resident_payload_capacity {
                    if remaining_safe_host_visible_payload_bytes.is_none() {
                        let capacity = read_vulkan_host_memory_capacity()
                            .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?;
                        remaining_safe_host_visible_payload_bytes = Some(
                            remaining_vulkan_runtime_host_cache_bytes(
                                capacity.safe_tiered_payload_bytes(),
                                physical_execution_residency_plan
                                    .total_stream_shared_host_bytes,
                            )
                            .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?,
                        );
                    }
                    let remaining = remaining_safe_host_visible_payload_bytes
                        .as_mut()
                        .expect("tiered host capacity was established");
                    if resource_residency_policy == ResourceResidencyPolicy::Eager {
                        let minimum_host_payload_bytes = maximum_dynamic_bytes
                            .saturating_sub(resident_payload_capacity)
                            .checked_add(physical_parameters.staging_headroom_bytes)
                            .ok_or_else(|| {
                                VulkanResidentInProcessPlacedRuntimeError::Package(
                                    VulkanResidentTokenModelPackageError::new(
                                        "tiered host-visible payload capacity overflowed",
                                    ),
                                )
                            })?;
                        let maximum_host_allocation_bytes = minimum_host_payload_bytes
                            .checked_add(maximum_alignment_padding)
                            .ok_or_else(|| {
                                VulkanResidentInProcessPlacedRuntimeError::Package(
                                    VulkanResidentTokenModelPackageError::new(
                                        "tiered host-visible allocation capacity overflowed",
                                    ),
                                )
                            })?;
                        if maximum_host_allocation_bytes > *remaining {
                            return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                                VulkanResidentTokenModelPackageError::new(format!(
                                    "eager tiered resources for physical slices {logical_device_ids:?} need up to {maximum_host_allocation_bytes} host-visible allocation bytes after exhausting device-local capacity, but only {remaining} safe host bytes remain after system headroom"
                                )),
                            ));
                        }
                        *remaining -= maximum_host_allocation_bytes;
                        (
                            maximum_dynamic_bytes,
                            resident_payload_capacity,
                            minimum_host_payload_bytes,
                            None,
                        )
                    } else {
                        let cache = match &shared_host_cache {
                            Some(cache) => Arc::clone(cache),
                            None => {
                                let cache = Arc::new(
                                    VulkanCompiledResourceSharedHostCache::new(
                                        format!("{package_id}:{execution_scope}:host_cache"),
                                        *remaining,
                                    )
                                    .map_err(|error| {
                                        VulkanResidentInProcessPlacedRuntimeError::Package(
                                            VulkanResidentTokenModelPackageError::new(
                                                error.to_string(),
                                            ),
                                        )
                                    })?,
                                );
                                shared_host_cache = Some(Arc::clone(&cache));
                                cache
                            }
                        };
                        let capacities = demand_paged_shared_tier_capacities(
                            maximum_dynamic_bytes,
                            resident_payload_capacity,
                            cache.capacity_bytes(),
                        )
                        .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?;
                        (
                            capacities.store_payload_bytes,
                            capacities.device_payload_bytes,
                            capacities.host_visible_payload_bytes,
                            Some(cache),
                        )
                    }
                } else {
                    (
                        resident_payload_capacity,
                        resident_payload_capacity,
                        0,
                        None,
                    )
                };
                let store = Arc::new(
                    VulkanCompiledResourceDeviceStore::new_tiered_with_selector_ownership(
                        physical_device,
                        resource_residency_policy,
                        store_id.clone(),
                        physical_device_id.clone(),
                        logical_device_id_list.clone(),
                        manifest_dir,
                        Arc::clone(&compiled_resource_contract),
                        Arc::clone(&compiled_resource_layout),
                        selector_ownership.clone(),
                        store_payload_capacity,
                        device_payload_capacity,
                        host_visible_payload_capacity,
                        safe_dynamic_bytes,
                        physical_parameters.staging_headroom_bytes,
                        maximum_ranges_per_group,
                        physical_parameters.always_resident_bytes,
                        working_set_bytes,
                        store_residency.metadata_device_bytes,
                        store_shared_host_cache.clone(),
                    )
                    .map_err(|error| {
                        VulkanResidentInProcessPlacedRuntimeError::Package(
                            VulkanResidentTokenModelPackageError::new(format!(
                                "failed to create compiled resource store for physical slices {logical_device_ids:?}: {error}"
                            )),
                        )
                    })?,
                );
                if let Some(cache) = &store_shared_host_cache {
                    cache.register_store(&store_id).map_err(|error| {
                        VulkanResidentInProcessPlacedRuntimeError::Package(
                            VulkanResidentTokenModelPackageError::new(format!(
                                "failed to register shared host-cache store for physical slices {logical_device_ids:?}: {error}"
                            )),
                        )
                    })?;
                }
                store
                    .register_device_memory_reclaimer(physical_device)
                    .map_err(|error| {
                        VulkanResidentInProcessPlacedRuntimeError::Package(
                            VulkanResidentTokenModelPackageError::new(format!(
                                "failed to register compiled-resource capacity reclamation for physical slices {logical_device_ids:?}: {error}"
                            )),
                        )
                    })?;
                (store, false)
            };
            for logical_device_id in &logical_device_id_list {
                let logical_execution_device_ids =
                    BTreeSet::from([logical_device_id.clone()]);
                let Some(execution_ownership) =
                    compiled_resource_selector_ownership_for_device_set(
                        &runtime_model,
                        &compiled_resource_contract,
                        &input_device_id,
                        &output_device_id,
                        &logical_execution_device_ids,
                        mount_speculative_decoders,
                        &distributed_selected_resource_execution_ownership_plan,
                    )
                    .map_err(|error| {
                        VulkanResidentInProcessPlacedRuntimeError::Package(
                            VulkanResidentTokenModelPackageError::new(format!(
                                "failed to plan logical compiled-resource execution ownership: {error}"
                            )),
                        )
                    })?
                else {
                    continue;
                };
                let owner_slice_index = device_slices
                    .iter()
                    .position(|slice| slice.device_id == *logical_device_id);
                let component_ids =
                    compiled_resource_component_ids_for_selector_ownership(
                        &compiled_resource_contract,
                        &execution_ownership,
                    )
                    .map_err(|error| {
                        VulkanResidentInProcessPlacedRuntimeError::Package(
                            VulkanResidentTokenModelPackageError::new(format!(
                                "failed to resolve dynamic-resource components for device {logical_device_id:?}: {error}",
                            )),
                        )
                    })?;
                if component_ids.is_empty() {
                    continue;
                }
                let logical_device = device_for(logical_device_id)?;
                let dynamic_buffers = store
                    .dynamic_buffers_for_components_with_execution_ownership(
                        logical_device,
                        &execution_scope,
                        &component_ids,
                        &execution_ownership,
                    )
                    .map_err(|error| {
                        VulkanResidentInProcessPlacedRuntimeError::Package(
                            VulkanResidentTokenModelPackageError::new(format!(
                                "failed to bind dynamic resources for device {logical_device_id:?}: {error}"
                            )),
                        )
                    })?;
                if distributed_dynamic_resource_buffers
                    .insert(logical_device_id.clone(), Arc::clone(&dynamic_buffers))
                    .is_some()
                {
                    return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                        VulkanResidentTokenModelPackageError::new(format!(
                            "dynamic resource buffers for logical device {logical_device_id:?} were assigned twice"
                        )),
                    ));
                }
                if let Some(slice_index) = owner_slice_index {
                    device_slices[slice_index].dynamic_resource_buffers = Some(dynamic_buffers);
                }
            }
            if !reused_store && resource_residency_policy == ResourceResidencyPolicy::Eager {
                let owner = DeviceResourceResidencyOwnerId::new(format!("{store_id}:eager"))
                    .map_err(|error| {
                        VulkanResidentInProcessPlacedRuntimeError::Package(
                            VulkanResidentTokenModelPackageError::new(format!(
                                "failed to create physical compiled resource owner: {error}"
                            )),
                        )
                    })?;
                store
                    .load_all_allowed(physical_device, owner)
                    .map_err(|error| {
                        VulkanResidentInProcessPlacedRuntimeError::Package(
                            VulkanResidentTokenModelPackageError::new(format!(
                                "failed to load eager compiled resources for physical slices {logical_device_ids:?}: {error}"
                            )),
                        )
                    })?;
            }
            if !reused_store {
                store.mark_mount_complete().map_err(|error| {
                    VulkanResidentInProcessPlacedRuntimeError::Package(
                        VulkanResidentTokenModelPackageError::new(format!(
                            "failed to seal initial compiled resource state for physical slices {logical_device_ids:?}: {error}"
                        )),
                    )
                })?;
            }
            let executing_component_ids = components_by_scope
                .into_values()
                .flatten()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            let mut selector_placements = Vec::new();
            for selector_id in &allowed_selector_ids {
                let selector_layout = compiled_resource_layout
                    .selectors
                    .iter()
                    .find(|selector| selector.selector_id == *selector_id)
                    .expect("owned selector was validated against the address layout");
                let previously_resident_physical_device_ids = planned_selector_physical_placements
                    .iter()
                    .map(|(mapping, device_id)| {
                        selector_layout
                            .mapping
                            .overlaps(mapping)
                            .map(|overlaps| overlaps.then(|| device_id.clone()))
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|error| {
                        VulkanResidentInProcessPlacedRuntimeError::Package(
                            VulkanResidentTokenModelPackageError::new(error.to_string()),
                        )
                    })?
                    .into_iter()
                    .flatten()
                    .filter(|device_id| device_id != &physical_device_id)
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>();
                let cross_device_choice = if previously_resident_physical_device_ids.is_empty() {
                    None
                } else {
                    Some(
                        require_explicit_compiled_resource_cross_device_choice(
                            &VulkanCompiledResourceCrossDeviceAccessRequest {
                                selector_id: selector_id.clone(),
                                execution_physical_device_id: physical_device_id.clone(),
                                resident_physical_device_ids:
                                    previously_resident_physical_device_ids.clone(),
                            },
                            Some(VulkanCompiledResourceCrossDeviceAccessChoice::SecondResidentCopy),
                        )
                        .map_err(|error| {
                            VulkanResidentInProcessPlacedRuntimeError::Package(
                                VulkanResidentTokenModelPackageError::new(error.to_string()),
                            )
                        })?,
                    )
                };
                planned_selector_physical_placements
                    .push((selector_layout.mapping.clone(), physical_device_id.clone()));
                selector_placements.push(VulkanCompiledResourceSelectorPhysicalPlacement {
                    selector_id: selector_id.clone(),
                    cross_device_choice,
                    previously_resident_physical_device_ids,
                });
            }
            let logical_device_ids = logical_device_ids.into_iter().collect::<Vec<_>>();
            compiled_resource_physical_placements.push(VulkanCompiledResourcePhysicalPlacement {
                store_id,
                physical_device_id,
                action: VulkanCompiledResourcePlacementAction::LocalToExecutionDevice,
                logical_device_ids: logical_device_ids.clone(),
                executing_component_ids,
                selector_ids: allowed_selector_ids.into_iter().collect(),
                selector_placements,
                maximum_dynamic_payload_bytes: maximum_dynamic_bytes,
                maximum_atomic_group_bytes: physical_parameters.staging_headroom_bytes,
            });
            for logical_device_id in logical_device_ids {
                if compiled_resource_device_stores
                    .insert(logical_device_id.clone(), Arc::clone(&store))
                    .is_some()
                {
                    return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                        VulkanResidentTokenModelPackageError::new(format!(
                            "compiled resource store for logical device {logical_device_id:?} was assigned twice"
                        )),
                    ));
                }
            }
        }
        if resource_residency_policy.is_demand_loaded() {
            attach_distributed_compiled_resource_cohorts(
                &distributed_selected_resource_execution_ownership_plan,
                &compiled_resource_device_stores,
            )
            .map_err(|error| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(format!(
                        "failed to attach distributed selected-resource residency cohorts: {error}",
                    )),
                )
            })?;
        }
        let selected_resource_reconfiguration_context =
            build_vulkan_runtime_selected_resource_reconfiguration_context(
                &runtime_model,
                &compiled_resource_contract,
                &distributed_loaded_manifest,
                &distributed_execution_plans,
                &distributed_selected_resource_execution_ownership_plan,
                &compiled_resource_device_stores,
                &device_execution_identity_by_logical_device,
                resource_residency_policy,
                placement_calibration_catalog,
            )?;
        let device_slices = device_slices.into_iter().map(Arc::new).collect::<Vec<_>>();

        let transducer_parameter_count = if Arc::ptr_eq(
            &input_transducer_parameter_buffers,
            &output_transducer_parameter_buffers,
        ) {
            input_transducer_parameter_buffers.plan.parameter_count
        } else {
            input_transducer_parameter_buffers.plan.parameter_count
                + output_transducer_parameter_buffers.plan.parameter_count
        };
        let transducer_parameter_bytes = if Arc::ptr_eq(
            &input_transducer_parameter_buffers,
            &output_transducer_parameter_buffers,
        ) {
            input_transducer_parameter_buffers.total_byte_capacity
        } else {
            input_transducer_parameter_buffers.total_byte_capacity
                + output_transducer_parameter_buffers.total_byte_capacity
        };

        let speculative_decoder_count = if mount_speculative_decoders {
            runtime_model.package.speculative_decoders.len()
        } else {
            0
        };
        let mut speculative_decoders = Vec::with_capacity(speculative_decoder_count);
        let speculative_decoder_context = VulkanResidentSpeculativeDecoderLoadContext {
            manifest_dir,
            runtime_model: &runtime_model,
            capacity,
            tensor_index: &tensor_index,
            target_output_parameters: &output_transducer_parameter_buffers,
            input_embedding_spec: &runtime_model.package.input_transducer.spec,
            input_embedding_spirv_words: &input_transducer_spirv_words,
            input_embedding_batch_spirv_words: &input_transducer_batch_spirv_words,
            input_embedding_batch_control: runtime_model.package.input_transducer.batch_control,
            compiled_resource_device_stores: &compiled_resource_device_stores,
            resource_residency_policy,
        };
        for decoder in runtime_model
            .package
            .speculative_decoders
            .iter()
            .take(speculative_decoder_count)
        {
            let device_slice_plan = speculative_decoder_slice_plans
                .remove(&decoder.id)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::Package(
                        VulkanResidentTokenModelPackageError::new(format!(
                            "speculative decoder {:?} has no prepared device slice",
                            decoder.id,
                        )),
                    )
                })?;
            speculative_decoders.push(
                VulkanResidentSpeculativeDecoderModelPackage::from_runtime_model(
                    output_device,
                    decoder,
                    &output_device_id,
                    device_slice_plan,
                    &speculative_decoder_context,
                )?,
            );
        }
        if !speculative_decoder_slice_plans.is_empty() {
            return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "prepared speculative decoder slices were not consumed: {:?}",
                    speculative_decoder_slice_plans.keys().collect::<Vec<_>>(),
                )),
            ));
        }

        let runtime_instance_by_id = runtime_model
            .runtime_graph
            .instances
            .iter()
            .map(|instance| (instance.instance_id.as_str(), instance))
            .collect::<BTreeMap<_, _>>();
        let runtime_component_instances = runtime_model
            .circuit_graph
            .components
            .iter()
            .enumerate()
            .map(|(execution_index, component)| {
                let instance = runtime_instance_by_id
                    .get(component.component_id.as_str())
                    .expect(
                        "mounted circuit graph components come from validated runtime instances",
                    );
                VulkanRuntimeComponentInstance {
                    instance_id: instance.instance_id.clone(),
                    source_component_id: instance.source_component_id.clone(),
                    device_id: instance.device_id.clone(),
                    execution_index,
                }
            })
            .collect();

        Ok(Self {
            package_id,
            execution_scope,
            runtime_execution_identity,
            resource_residency_policy,
            input_device_id,
            output_device_id,
            dynamic_state_capacity_activations: capacity,
            speculative_draft_tokens,
            normal_prefill_lane_capacity,
            device_count: device_ids.len(),
            device_ids,
            hosted_component_count,
            transducer_parameter_count,
            transducer_parameter_bytes,
            input_transducer_parameter_buffers,
            output_transducer_parameter_buffers,
            input_transducer_spirv_words,
            input_transducer_batch_spirv_words,
            input_transducer_batch_control: runtime_model.package.input_transducer.batch_control,
            embedding_norm_spirv_words,
            embedding_norm_batch_spirv_words,
            embedding_norm_batch_lane_tile_width: runtime_model
                .package
                .output_transducer
                .embedding_norm_batch_lane_tile_width,
            tied_projection_spirv_words,
            tied_projection_batch_spirv_words,
            projection_batch_lane_tile_width: runtime_model
                .package
                .output_transducer
                .projection_batch_lane_tile_width,
            sampler_kernels,
            input_transducer_spec: runtime_model.package.input_transducer.spec.clone(),
            output_transducer_spec: runtime_model.package.output_transducer.spec.clone(),
            sampler_spec: runtime_model.package.sampler.spec.clone(),
            device_slices,
            speculative_decoders,
            distributed_execution_plans,
            distributed_activation_plan,
            distributed_parameter_allocation_plan,
            distributed_parameter_exclusion_plan,
            physical_execution_residency_plan,
            mounted_boundary_routes,
            selected_resource_placements,
            selected_resource_reconfiguration_context,
            distributed_selected_resource_execution_ownership_plan,
            distributed_selected_resource_store_plan,
            distributed_loaded_manifest,
            distributed_parameter_buffers,
            distributed_dynamic_resource_buffers,
            compiled_resource_device_stores,
            compiled_resource_physical_placements,
            runtime_component_instances,
        })
    }

    pub fn device_slice(&self, device_id: &str) -> Option<&VulkanResidentModelPackageDeviceSlice> {
        self.device_slices
            .iter()
            .find(|slice| slice.device_id == device_id)
            .map(Arc::as_ref)
    }

    pub fn decode_distributed_execution_plan(&self) -> &VulkanDistributedExecutionPlan {
        &self.distributed_execution_plans.decode
    }

    pub fn prefill_distributed_execution_plan(&self) -> &VulkanDistributedExecutionPlan {
        &self.distributed_execution_plans.prefill
    }

    pub fn decode_batch_distributed_execution_plan(&self) -> &VulkanDistributedExecutionPlan {
        &self.distributed_execution_plans.decode_batch
    }

    pub fn distributed_activation_plan(&self) -> &VulkanDistributedActivationBufferPlan {
        &self.distributed_activation_plan
    }

    pub fn distributed_parameter_allocation_plan(
        &self,
    ) -> &VulkanDistributedParameterAllocationPlan {
        &self.distributed_parameter_allocation_plan
    }

    pub fn distributed_parameter_exclusion_plan(&self) -> &VulkanDistributedParameterExclusionPlan {
        &self.distributed_parameter_exclusion_plan
    }

    pub fn physical_execution_residency_plan(
        &self,
    ) -> &VulkanRuntimePhysicalExecutionResidencyPlan {
        &self.physical_execution_residency_plan
    }

    pub fn create_stream_processor_for_devices(
        self: &Arc<Self>,
        device: &VulkanComputeDevice,
        random_seed: u32,
    ) -> Result<
        VulkanResidentInProcessPlacedStreamProcessor,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        self.create_stream_processor_for_device_resolver(random_seed, None, |_| Ok(device))
    }

    pub fn create_stream_processor_for_bound_devices(
        self: &Arc<Self>,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        random_seed: u32,
    ) -> Result<
        VulkanResidentInProcessPlacedStreamProcessor,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        self.create_stream_processor_for_device_resolver(random_seed, None, |device_id| {
            devices
                .get(device_id)
                .map(|device| device.as_ref())
                .ok_or_else(
                    || VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                        device_id: device_id.to_string(),
                    },
                )
        })
    }

    pub fn create_stream_processor_inheriting_state_for_devices(
        self: &Arc<Self>,
        device: &VulkanComputeDevice,
        random_seed: u32,
        source: &VulkanResidentInProcessPlacedStreamProcessor,
    ) -> Result<
        VulkanResidentInProcessPlacedStreamProcessor,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        self.create_stream_processor_for_device_resolver(random_seed, Some(source), |_| Ok(device))
    }

    pub fn create_stream_processor_inheriting_state_for_bound_devices(
        self: &Arc<Self>,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        random_seed: u32,
        source: &VulkanResidentInProcessPlacedStreamProcessor,
    ) -> Result<
        VulkanResidentInProcessPlacedStreamProcessor,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        self.create_stream_processor_for_device_resolver(random_seed, Some(source), |device_id| {
            devices
                .get(device_id)
                .map(|device| device.as_ref())
                .ok_or_else(
                    || VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                        device_id: device_id.to_string(),
                    },
                )
        })
    }

    fn create_stream_processor_for_device_resolver<'a, F>(
        self: &Arc<Self>,
        random_seed: u32,
        source: Option<&VulkanResidentInProcessPlacedStreamProcessor>,
        device_for: F,
    ) -> Result<
        VulkanResidentInProcessPlacedStreamProcessor,
        VulkanResidentInProcessPlacedRuntimeError,
    >
    where
        F: Fn(&str) -> Result<&'a VulkanComputeDevice, VulkanResidentInProcessPlacedRuntimeError>,
    {
        if let Some(source) = source
            && source.model.package_id != self.package_id
        {
            return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "cannot inherit stream state from package {:?} into package {:?}",
                    source.model.package_id, self.package_id
                )),
            ));
        }
        let selected_resource_adaptation = self
            .selected_resource_reconfiguration_context
            .as_ref()
            .map(|context| {
                initial_vulkan_runtime_selected_resource_adaptation_state(
                    Arc::clone(context),
                    &self.distributed_execution_plans,
                    source.and_then(|source| source.selected_resource_adaptation.as_ref()),
                )
            });
        let selected_resource_cache_registration = selected_resource_adaptation
            .as_ref()
            .map(|state| state.context.cache_arbiter.register())
            .transpose()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?;
        let stream_distributed_execution_plans = selected_resource_adaptation
            .as_ref()
            .map(|state| &state.execution_plans)
            .unwrap_or(&self.distributed_execution_plans);
        let stream_memory_admission =
            reserve_vulkan_runtime_physical_execution_stream_memory(self, &device_for)?;
        let _stream_memory_scope = stream_memory_admission.enter();
        validate_vulkan_runtime_private_activation_residency(
            &self.physical_execution_residency_plan,
            &self.distributed_activation_plan,
        )
        .map_err(|error| {
            VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "invalid exact private activation allocation ledger: {error}",
                )),
            )
        })?;
        let distributed_dynamic_resource_buffers = self
            .distributed_dynamic_resource_buffers
            .iter()
            .map(|(device_id, template)| {
                let device = device_for(device_id)?;
                template
                    .fork_for_stream(device)
                    .map(|buffers| (device_id.clone(), buffers))
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        initialize_vulkan_stream_selected_resource_execution_ownership(
            &distributed_dynamic_resource_buffers,
            &stream_distributed_execution_plans.decode,
        )
        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let mut distributed_activation_buffers =
            VulkanDistributedActivationBuffers::allocate_deferring_graph_edges(
                &self.distributed_activation_plan,
                |device_id| device_for(device_id),
            )
        .map_err(|error| {
            VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to allocate distributed Vulkan activation edges: {error}"
                )),
            )
        })?;
        let VulkanPlacedDeviceLinks {
            local_edge_overrides: shared_local_edge_overrides,
            endpoint_overrides: shared_edge_endpoint_overrides,
            synchronizations: edge_synchronizations,
            stream_control_buffers,
            every_edge_is_resident_replayable,
            feedback_stream_control_is_resident_replayable,
        } = create_placed_device_links(
            &self.device_slices,
            &mut distributed_activation_buffers,
            &self.mounted_boundary_routes,
            &self.physical_execution_residency_plan,
            &device_for,
        )?;
        let mut distributed_dispatch_indices = BTreeMap::<&str, BTreeSet<usize>>::new();
        for group in &stream_distributed_execution_plans.decode.execution_islands {
            distributed_dispatch_indices
                .entry(group.owner_device_id.as_str())
                .or_default()
                .extend(group.dispatch_indices());
        }
        let distributed_islands_by_owner = stream_distributed_execution_plans
            .decode
            .execution_islands
            .iter()
            .fold(BTreeMap::<&str, Vec<Vec<usize>>>::new(), |mut groups, island| {
                groups
                    .entry(island.owner_device_id.as_str())
                    .or_default()
                    .push(island.dispatch_indices());
                groups
            });
        let local_dispatch_count = self.device_slices.iter().try_fold(0usize, |total, slice| {
            let distributed = distributed_dispatch_indices.get(slice.device_id.as_str());
            let model_dispatch_count = slice
                .prepared_plan()
                .dispatches
                .iter()
                .filter(|dispatch| {
                    distributed.is_none_or(|indices| !indices.contains(&dispatch.dispatch_index))
                })
                .count();
            let checkpoint_count = slice
                .physical_residency_schedule()
                .local_demand_gate_count(
                    self.resource_residency_policy,
                    distributed_islands_by_owner
                        .get(slice.device_id.as_str())
                        .map(Vec::as_slice)
                        .unwrap_or_default(),
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            total
                .checked_add(model_dispatch_count)
                .and_then(|count| count.checked_add(checkpoint_count))
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "resident feedback planned local dispatch count overflowed".to_string(),
                    ))
                })
        })?;
        let demand_feedback_predicates =
            create_demand_feedback_pipeline_predicates(self, &device_for)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let mut distributed_dispatch_runners = VulkanDistributedDispatchRunners::create(
            &stream_distributed_execution_plans.decode,
            VulkanResidentDistributedExecutionPhase::Decode,
            &self.distributed_parameter_buffers,
            &distributed_dynamic_resource_buffers,
            &self.compiled_resource_device_stores,
            demand_feedback_predicates.as_ref(),
            &self.execution_scope,
            &distributed_activation_buffers,
            &self.distributed_loaded_manifest,
            |device_id| device_for(device_id),
        )
        .map_err(|error| {
            VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to mount distributed Vulkan dispatches: {error}"
                )),
            )
        })?;
        let distributed_feedback_dispatch_count = distributed_dispatch_runners
            .feedback_dispatch_count()
            .map_err(|error| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(format!(
                        "failed to count distributed feedback dispatches: {error}"
                    )),
                )
            })?;
        let sampler_dispatch_count = VulkanResidentSamplerRunner::feedback_dispatch_count_for_spec(
            &self.sampler_kernels,
            &self.sampler_spec,
        );
        let feedback_dispatch_capacity = local_dispatch_count
            .checked_add(distributed_feedback_dispatch_count)
            .and_then(|count| count.checked_add(1))
            .and_then(|count| count.checked_add(2))
            .and_then(|count| count.checked_add(sampler_dispatch_count))
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "resident feedback dispatch capacity overflowed".to_string(),
                ))
            })?;
        let vocabulary_size = self.sampler_spec.logits_byte_capacity / size_of::<f32>();
        let planned_feedback_control_bytes = self
            .physical_execution_residency_plan
            .feedback_control_resident_byte_capacity()
            .map_err(|error| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(format!(
                        "failed exact feedback-control residency validation: {error}",
                    )),
                )
            })?;
        let mounted_feedback_control_bytes = resident_feedback_control_byte_capacity(
            vocabulary_size,
            feedback_dispatch_capacity,
        )
        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        if planned_feedback_control_bytes != mounted_feedback_control_bytes {
            return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "planned feedback control needs {planned_feedback_control_bytes} bytes but the exact mounted dispatch set needs {mounted_feedback_control_bytes} bytes",
                )),
            ));
        }
        let feedback_control_shared_allocations = self
            .physical_execution_residency_plan
            .resident_shared_host_allocations
            .iter()
            .filter(|allocation| {
                matches!(
                    allocation.kind,
                    VulkanRuntimeSharedHostResidentAllocationKind::FeedbackControl { .. }
                )
            })
            .collect::<Vec<_>>();
        let [planned_feedback_control_shared_allocation] =
            feedback_control_shared_allocations.as_slice()
        else {
            return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "physical feedback control resolves {} shared-host allocation ledgers, expected one",
                    feedback_control_shared_allocations.len(),
                )),
            ));
        };
        let mut feedback_control = VulkanResidentFeedbackControlPlane::new(
            &self.device_ids,
            &self.output_device_id,
            vocabulary_size,
            feedback_dispatch_capacity,
            planned_feedback_control_shared_allocation,
            &device_for,
        )
        .map_err(|error| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                "failed to mount exact resident feedback control: {error}",
            )))
        })?;
        let mut devices = Vec::with_capacity(self.device_slices.len());
        for package_slice in &self.device_slices {
            let device = device_for(&package_slice.device_id)?;
            let activation_overrides = distributed_activation_buffers
                .activation_overrides_for_owner_device(&package_slice.device_id);
            let boundary_overrides = distributed_activation_buffers
                .boundary_overrides_for_owner_device(&package_slice.device_id);
            let shared_local_edge_ids = shared_local_edge_overrides
                .get(&package_slice.device_id)
                .into_iter()
                .flat_map(|overrides| overrides.iter().map(|override_| override_.edge_index))
                .collect::<BTreeSet<_>>();
            let local_edge_overrides = package_slice
                .placed_plan
                .placed_resident_plan
                .local_edges
                .iter()
                .filter(|edge| !shared_local_edge_ids.contains(&edge.edge_index))
                .filter_map(|edge| {
                    distributed_activation_buffers
                        .edge_buffer(edge.edge_index, &package_slice.device_id)
                        .map(|buffer| VulkanPlacedLocalEdgeBufferOverride {
                            edge_index: edge.edge_index,
                            buffer: Arc::clone(buffer),
                        })
                })
                .chain(
                    shared_local_edge_overrides
                        .get(&package_slice.device_id)
                        .into_iter()
                        .flat_map(|overrides| overrides.iter())
                        .map(|override_| VulkanPlacedLocalEdgeBufferOverride {
                            edge_index: override_.edge_index,
                            buffer: Arc::clone(&override_.buffer),
                        }),
                )
                .collect::<Vec<_>>();
            let mounted = package_slice
                .create_mounted_stream_circuit_with_all_buffer_and_dynamic_resource_overrides(
                    device,
                    &activation_overrides,
                    &local_edge_overrides,
                    shared_edge_endpoint_overrides
                        .get(&package_slice.device_id)
                        .map(Vec::as_slice)
                        .unwrap_or_default(),
                    &boundary_overrides,
                    stream_control_buffers
                        .get(&package_slice.device_id)
                        .cloned(),
                    distributed_dynamic_resource_buffers
                        .get(&package_slice.device_id)
                        .cloned(),
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?;
            mounted
                .buffers
                .initialize_state_buffers(device)
                .map_err(|error| {
                    VulkanResidentInProcessPlacedRuntimeError::Package(
                        VulkanResidentTokenModelPackageError::new(format!(
                            "failed to initialize stream state buffers for placed device {:?}: {error}",
                            package_slice.device_id
                        )),
                    )
                })?;
            let reusable_manifest = resident_package_reusable_kernel_manifest(&mounted.placed_plan);
            let physical_execution_islands = stream_distributed_execution_plans
                .decode
                .execution_islands
                .iter()
                .filter(|group| group.owner_device_id == package_slice.device_id)
                .map(|group| group.dispatch_indices())
                .collect::<Vec<_>>();
            let distributed_dispatch_indices = physical_execution_islands
                .iter()
                .flatten()
                .copied()
                .collect::<BTreeSet<_>>();
            let mounted_bound = mounted
                .mounted_placed_bound_dispatch_plan_with_distributed_dispatches(
                    &reusable_manifest,
                    &distributed_dispatch_indices,
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BoundDispatchPlan)?;
            let tick_plan =
                VulkanMountedPlacedStreamTickPlan::from_mounted_bound_plan(&mounted_bound);
            let demand_context = if package_slice
                .physical_residency_schedule()
                .requires_demand_execution(self.resource_residency_policy)
            {
                let store = self
                    .compiled_resource_device_stores
                    .get(&package_slice.device_id)
                    .cloned()
                    .ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::Package(
                            VulkanResidentTokenModelPackageError::new(format!(
                                "demand-resident device {:?} has no compiled resource store",
                                package_slice.device_id
                            )),
                        )
                    })?;
                Some(VulkanDemandResidencyExecutionContext {
                    execution_scope: self.execution_scope.clone(),
                    contract: Arc::clone(&store.contract),
                    layout: Arc::clone(&store.layout),
                    store,
                    owner: DeviceResourceResidencyOwnerId::new(format!(
                        "{}:{}:{}",
                        self.package_id, package_slice.device_id, self.execution_scope
                    ))
                    .map_err(|error| {
                        VulkanResidentInProcessPlacedRuntimeError::Package(
                            VulkanResidentTokenModelPackageError::new(error.to_string()),
                        )
                    })?,
                })
            } else {
                None
            };
            let resident_execution_plan =
                VulkanMountedPlacedResidentStreamTickExecutionPlan::from_tick_plan_with_physical_execution_islands_and_demand(
                    device,
                    &mounted,
                    &mounted_bound,
                    package_slice.loaded_manifest(),
                    tick_plan,
                    &physical_execution_islands,
                    demand_context
                        .as_ref()
                        .map(|_| package_slice.physical_residency_schedule()),
                    demand_context.as_ref(),
                    demand_feedback_predicates
                        .as_ref()
                        .and_then(|predicates| predicates.get(&package_slice.device_id))
                        .cloned(),
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::ResidentDispatch)?;
            devices.push(VulkanResidentInProcessPlacedStreamProcessorDevice {
                device_id: package_slice.device_id.clone(),
                hosted_component_count: package_slice.hosted_component_count,
                incoming_edge_count: package_slice.incoming_edge_count,
                outgoing_edge_count: package_slice.outgoing_edge_count,
                dispatch_count: mounted_bound.dispatches.len(),
                package_slice: package_slice.clone(),
                mounted,
                mounted_bound,
                resident_execution_plan,
                demand_residency_context: demand_context,
            });
        }
        let inherited = source
            .map(|source| inherit_matching_placed_stream_state(&devices, &source.device_slices))
            .transpose()
            .map_err(|error| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(format!(
                        "failed to inherit mounted stream state: {error}"
                    )),
                )
            })?
            .map(|(_, copied)| copied)
            .unwrap_or_default();
        apply_placed_clone_state_policies(&devices, &inherited).map_err(|error| {
            VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to initialize cloned stream state: {error}"
                )),
            )
        })?;
        let activation_tick_plans = devices
            .iter()
            .map(|slice| slice.resident_execution_plan.tick_plan.as_ref())
            .collect::<Vec<_>>();
        let activation_schedule =
            VulkanMountedPlacedResidentInProcessSchedule::from_tick_plans(&activation_tick_plans)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::Schedule)?;
        let input_slice = devices
            .iter()
            .find(|slice| slice.device_id == self.input_device_id)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(format!(
                        "placed package {:?} has no mounted input device slice {:?}",
                        self.package_id, self.input_device_id
                    )),
                )
            })?;
        let output_slice = devices
            .iter()
            .find(|slice| slice.device_id == self.output_device_id)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(format!(
                        "placed package {:?} has no mounted output device slice {:?}",
                        self.package_id, self.output_device_id
                    )),
                )
            })?;
        let input_device = device_for(&self.input_device_id)?;
        let output_device = device_for(&self.output_device_id)?;
        let input_transducer =
            VulkanResidentInputEmbeddingTransducerRunner::from_mounted_token_embedding(
                input_device,
                &input_slice.mounted,
                &self.input_transducer_parameter_buffers,
                &self.input_transducer_spirv_words,
                &self.input_transducer_spec,
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::InputTransducer)?;
        let output_transducer =
            VulkanResidentOutputTransducerRunner::from_mounted_output_transducer(
                output_device,
                &output_slice.mounted,
                &self.output_transducer_parameter_buffers,
                &self.embedding_norm_spirv_words,
                &self.tied_projection_spirv_words,
                &self.output_transducer_spec,
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::OutputTransducer)?;
        let mounted_local_dispatch_count = devices.iter().try_fold(0usize, |total, slice| {
            total
                .checked_add(
                    slice
                        .resident_execution_plan
                        .feedback_dispatch_count()
                        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
                )
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "resident feedback local dispatch count overflowed".to_string(),
                    ))
                })
        })?;
        if mounted_local_dispatch_count != local_dispatch_count {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "resident feedback planned {local_dispatch_count} local dispatches but mounted {mounted_local_dispatch_count}"
                )),
            ));
        }
        let sampler_host_allocations = self
            .physical_execution_residency_plan
            .resident_shared_host_allocations
            .iter()
            .filter(|allocation| {
                matches!(
                    allocation.kind,
                    VulkanRuntimeSharedHostResidentAllocationKind::HostVisibleRuntimeBuffer {
                        scope: VulkanRuntimeResidentStreamAllocationScope::Target,
                        class: VulkanRuntimeResidentBufferClass::SamplerWorkspace,
                        ..
                    }
                )
            })
            .collect::<Vec<_>>();
        let sampler =
            VulkanResidentSamplerRunner::from_output_transducer_with_spec_and_feedback_control(
                output_device,
                &self.output_device_id,
                &output_slice.mounted,
                &output_transducer,
                &self.sampler_kernels,
                &self.sampler_spec,
                random_seed,
                feedback_control
                    .sampler_bindings()
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
                &sampler_host_allocations,
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)?;
        let feedback_lane_capacity =
            VULKAN_BACKEND_LOOP_MAX_WINDOW.min(sampler.history_capacity_activations.max(1));
        for slice in &mut devices {
            let mut prefix_dispatches = SmallVec::<[&VulkanResidentKernelDispatch; 2]>::new();
            let mut suffix_dispatches = SmallVec::<[&VulkanResidentKernelDispatch; 5]>::new();
            if slice.device_id == self.input_device_id {
                prefix_dispatches.push(&input_transducer.resident_dispatch);
            }
            if slice.device_id == self.output_device_id {
                prefix_dispatches.extend(sampler.input_tracking_dispatches());
                suffix_dispatches.push(&output_transducer.embedding_norm_dispatch);
                suffix_dispatches.push(&output_transducer.tied_projection_dispatch);
                suffix_dispatches.extend(sampler.resident_dispatches());
                suffix_dispatches.push(sampler.feedback_control_dispatch());
            }
            let generation_tail_dispatch_count = (slice.device_id == self.output_device_id)
                .then_some(
                    2usize
                        .checked_add(sampler.resident_dispatches().len())
                        .ok_or_else(|| {
                            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                "resident feedback generation tail count overflowed".to_string(),
                            ))
                        })?,
                );
            slice
                .resident_execution_plan
                .configure_feedback_indirect_dispatches(
                    device_for(&slice.device_id)?,
                    &mut feedback_control,
                    &slice.device_id,
                    &prefix_dispatches,
                    &suffix_dispatches,
                    generation_tail_dispatch_count,
                    feedback_lane_capacity,
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        }
        distributed_dispatch_runners
            .configure_feedback_indirect_dispatches(
                &mut feedback_control,
                feedback_lane_capacity,
                |device_id| device_for(device_id),
            )
            .map_err(|error| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(format!(
                        "failed to configure distributed feedback dispatches: {error}"
                    )),
                )
            })?;
        feedback_control
            .finish_registration()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let output_synchronization =
            VulkanResidentPlacedOutputTimelineSynchronization::new(output_device)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let mut speculative_decoders = Vec::with_capacity(self.speculative_decoders.len());
        for decoder in &self.speculative_decoders {
            let draft_device = device_for(&decoder.device_id)?;
            let planned_stream_controls = self
                .physical_execution_residency_plan
                .resident_shared_host_allocations
                .iter()
                .filter(|allocation| {
                    matches!(
                        &allocation.kind,
                        VulkanRuntimeSharedHostResidentAllocationKind::HostVisibleRuntimeBuffer {
                            scope: VulkanRuntimeResidentStreamAllocationScope::SpeculativeDecoder {
                                decoder_id,
                            },
                            class: VulkanRuntimeResidentBufferClass::SpeculativeDecoderState,
                            buffer_id,
                            ..
                        } if decoder_id == &decoder.id && buffer_id == "stream_control"
                    )
                })
                .collect::<Vec<_>>();
            let [planned_stream_control] = planned_stream_controls.as_slice() else {
                return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(format!(
                        "speculative decoder {:?} resolves {} host-visible stream-control ledgers, expected one",
                        decoder.id,
                        planned_stream_controls.len(),
                    )),
                ));
            };
            speculative_decoders.push(VulkanResidentSpeculativeDecoderProcessor::from_model(
                draft_device,
                decoder,
                planned_stream_control,
                self,
                &devices,
                output_transducer.normalized_frame_buffer(),
                &self.output_transducer_parameter_buffers,
                &self.sampler_kernels,
                &self.sampler_spec,
                random_seed,
                &device_for,
            )?);
        }
        let speculative_state_is_resident_replayable =
            parallel_speculative_feedback_state_is_replayable(&speculative_decoders, &device_for)?;
        let resident_feedback_loop = VulkanResidentInProcessPlacedFeedbackLoop::new_if_supported(
            self,
            &devices,
            &activation_schedule,
            every_edge_is_resident_replayable,
            feedback_stream_control_is_resident_replayable,
            VulkanResidentPlacedFeedbackMount {
                input_transducer: &input_transducer,
                output_transducer: &output_transducer,
                sampler: &sampler,
                control: feedback_control,
                demand_pipeline_predicates: demand_feedback_predicates,
                speculative_state_is_resident_replayable,
            },
            &device_for,
        )
        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let speculative_target_frame_history = resident_feedback_loop
            .as_ref()
            .map(|_| {
                VulkanResidentSpeculativeTargetFrameHistory::new_if_needed(
                    self,
                    output_device,
                    &output_transducer,
                    &sampler,
                )
            })
            .transpose()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?
            .flatten();
        let parallel_speculative_feedback_memory_admission = resident_feedback_loop
            .as_ref()
            .map(|feedback_loop| {
                reserve_parallel_speculative_feedback_state_memory(
                    &speculative_decoders,
                    feedback_loop.window_policy.maximum_tick_count,
                    &device_for,
                )
            })
            .transpose()?
            .flatten();
        let parallel_speculative_feedback_state = if let Some(feedback_loop) = &resident_feedback_loop
        {
            let expected = VulkanParallelSpeculativeFeedbackAllocationPlan::from_decoders(
                &speculative_decoders,
                feedback_loop.window_policy.maximum_tick_count,
            )?
            .logical_bytes_by_device()?;
            let _feedback_memory_scope = parallel_speculative_feedback_memory_admission
                .as_ref()
                .map(|admission| admission.enter());
            let state = VulkanResidentParallelSpeculativeFeedbackState::new_if_needed(
                &speculative_decoders,
                feedback_loop.window_policy.maximum_tick_count,
                &device_for,
            )?;
            let actual = state
                .as_ref()
                .map(VulkanResidentParallelSpeculativeFeedbackState::logical_allocation_bytes_by_device)
                .transpose()?
                .unwrap_or_default();
            validate_parallel_speculative_feedback_allocation_totals(&expected, &actual)?;
            if let Some(admission) = &parallel_speculative_feedback_memory_admission {
                admission
                    .ensure_fully_consumed("parallel speculative feedback")
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            }
            state
        } else {
            None
        };
        stream_memory_admission
            .ensure_class_fully_consumed(
                VulkanMemoryAdmissionAllocationClass::Permanent,
                "permanent stream mount",
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let execution_quantum_calibrators = devices
            .iter()
            .map(|slice| {
                (
                    slice.device_id.clone(),
                    Rc::new(RefCell::new(RuntimeExecutionQuantumCalibrator::default())),
                )
            })
            .collect();
        Ok(VulkanResidentInProcessPlacedStreamProcessor {
            stream_memory_admission,
            _parallel_speculative_feedback_memory_admission:
                parallel_speculative_feedback_memory_admission,
            model: self.clone(),
            distributed_dispatch_runners,
            distributed_dynamic_resource_buffers,
            selected_resource_adaptation,
            selected_resource_cache_registration,
            _distributed_activation_buffers: distributed_activation_buffers,
            edge_synchronizations,
            input_transducer,
            output_transducer,
            sampler,
            output_synchronization,
            resident_feedback_loop,
            speculative_target_frame_history,
            parallel_speculative_feedback_state,
            activation_schedule,
            device_slices: devices,
            execution_quantum_calibrators,
            speculative_decoders,
            verification_state_transactions: RefCell::new(None),
            temporal_block_executions: RefCell::new(BTreeMap::new()),
        })
    }
}
