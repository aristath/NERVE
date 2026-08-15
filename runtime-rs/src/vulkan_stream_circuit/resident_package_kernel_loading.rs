fn resident_package_reusable_kernel_manifest(
    placed_plan: &VulkanPlacedStreamCircuitPlan,
) -> VulkanReusableKernelArtifactManifest {
    VulkanReusableKernelArtifactManifest::new(
        placed_plan
            .reusable_kernel_plan
            .families
            .iter()
            .map(|family| {
                VulkanReusableKernelArtifact::from_family(
                    family,
                    format!("kernels/{}.spv", family.family_id),
                )
            })
            .collect(),
    )
}

fn resident_package_loaded_kernel_manifest_for_slice_plans(
    slice_plans: &[VulkanResidentModelPackageDeviceSlicePlan],
) -> Result<VulkanLoadedKernelArtifactCatalog, VulkanResidentTokenModelPackageError> {
    let mut artifacts_by_family = BTreeMap::<String, VulkanLoadedReusableKernelArtifact>::new();
    let mut physical_artifacts_by_id =
        BTreeMap::<String, VulkanLoadedPhysicalKernelArtifact>::new();
    for slice in slice_plans {
        for artifact in &slice.loaded_manifest.reusable_artifacts {
            if let Some(existing) = artifacts_by_family.get(&artifact.artifact.family_id) {
                let mut existing_contract = existing.artifact.clone();
                existing_contract.path.clear();
                let mut candidate_contract = artifact.artifact.clone();
                candidate_contract.path.clear();
                if existing_contract != candidate_contract || existing.words != artifact.words {
                    return Err(VulkanResidentTokenModelPackageError::new(format!(
                        "loaded reusable Vulkan family {:?} conflicts across device slices",
                        artifact.artifact.family_id
                    )));
                }
            } else {
                artifacts_by_family.insert(artifact.artifact.family_id.clone(), artifact.clone());
            }
        }
        for artifact in &slice.loaded_manifest.physical_artifacts {
            if let Some(existing) =
                physical_artifacts_by_id.get(&artifact.artifact.artifact_id)
            {
                if existing.artifact != artifact.artifact || existing.words != artifact.words {
                    return Err(VulkanResidentTokenModelPackageError::new(format!(
                        "loaded physical Vulkan artifact {:?} conflicts across device slices",
                        artifact.artifact.artifact_id
                    )));
                }
            } else {
                physical_artifacts_by_id
                    .insert(artifact.artifact.artifact_id.clone(), artifact.clone());
            }
        }
    }
    let reusable_artifacts = artifacts_by_family.into_values().collect::<Vec<_>>();
    let physical_artifacts = physical_artifacts_by_id.into_values().collect::<Vec<_>>();
    let reusable_word_count = reusable_artifacts
        .iter()
        .map(|artifact| artifact.words.len())
        .try_fold(0usize, |total, words| {
            total.checked_add(words).ok_or_else(|| {
                VulkanResidentTokenModelPackageError::new(
                    "combined reusable Vulkan kernel word count overflowed",
                )
            })
        })?;
    let physical_word_count = physical_artifacts
        .iter()
        .map(|artifact| artifact.words.len())
        .try_fold(0usize, |total, words| {
            total.checked_add(words).ok_or_else(|| {
                VulkanResidentTokenModelPackageError::new(
                    "combined physical Vulkan kernel word count overflowed",
                )
            })
        })?;
    Ok(VulkanLoadedKernelArtifactCatalog {
        reusable_artifacts,
        physical_artifacts,
        reusable_word_count,
        physical_word_count,
    })
}

fn resident_package_component_kernel_shader_refs(
    component_executions: &[VulkanResidentComponentExecutionSpec],
) -> Vec<VulkanResidentComponentKernelShaderRef> {
    component_executions
        .iter()
        .flat_map(|component| {
            component
                .kernels
                .iter()
                .map(|kernel| VulkanResidentComponentKernelShaderRef {
                    component_id: component.component_id.clone(),
                    node_id: kernel.node_id.clone(),
                    shader_path: kernel.shader_path.clone(),
                    local_size_x: kernel.local_size_x,
                    workgroup_count_x: kernel.workgroup_count_x,
                    physical_execution_contracts: kernel.physical_execution_contracts.clone(),
                    resource_representation_dispatch: kernel
                        .resource_representation_dispatch
                        .clone(),
                })
        })
        .collect()
}

fn resident_package_component_kernel_shader_refs_for_prepared_dispatches(
    component_executions: &[VulkanResidentComponentExecutionSpec],
    prepared_plan: &VulkanPreparedDispatchPlan,
) -> Vec<VulkanResidentComponentKernelShaderRef> {
    resident_package_component_kernel_shader_refs(component_executions)
        .into_iter()
        .filter(|shader| {
            prepared_plan
                .dispatch(&shader.component_id, &shader.node_id)
                .is_some()
        })
        .collect()
}

fn attach_resident_package_physical_execution_contracts(
    prepared_plan: &mut VulkanPreparedDispatchPlan,
    dispatch_shaders: &[VulkanResidentComponentKernelShaderRef],
) -> Result<(), VulkanResidentTokenModelPackageError> {
    let mut contracts_by_dispatch = BTreeMap::new();
    for shader in dispatch_shaders {
        let key = (shader.component_id.as_str(), shader.node_id.as_str());
        if contracts_by_dispatch
            .insert(
                key,
                (
                    shader.physical_execution_contracts.as_slice(),
                    shader.resource_representation_dispatch.as_ref(),
                ),
            )
            .is_some()
        {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "resident package repeats kernel contract source {}.{}",
                shader.component_id, shader.node_id,
            )));
        }
    }
    for dispatch in &mut prepared_plan.dispatches {
        let (contracts, representation_dispatch) = contracts_by_dispatch
            .get(&(dispatch.component_id.as_str(), dispatch.node_id.as_str()))
            .ok_or_else(|| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "resident package has no physical contracts for prepared dispatch {}.{}",
                    dispatch.component_id, dispatch.node_id,
                ))
            })?;
        if contracts.is_empty() {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "resident package has an empty physical contract set for prepared dispatch {}.{}",
                dispatch.component_id, dispatch.node_id,
            )));
        }
        dispatch.physical_execution_contracts = contracts.to_vec();
        dispatch.resource_representation_dispatch = representation_dispatch.cloned();
    }
    Ok(())
}

fn loaded_kernel_pack_from_package_shader_refs(
    manifest_dir: &Path,
    placed_plan: &VulkanPlacedStreamCircuitPlan,
    prepared_plan: &VulkanPreparedDispatchPlan,
    dispatch_shaders: &[VulkanResidentComponentKernelShaderRef],
) -> Result<VulkanLoadedKernelArtifactCatalog, VulkanResidentTokenModelPackageError> {
    let mut loaded_artifacts = Vec::new();
    let mut loaded_physical_artifacts = Vec::new();
    let mut loaded_families = BTreeSet::new();
    let mut loaded_physical_metadata = BTreeMap::new();
    let mut reusable_word_count = 0usize;
    let mut physical_word_count = 0usize;

    for shader in dispatch_shaders {
        let dispatch = prepared_plan
            .dispatch(&shader.component_id, &shader.node_id)
            .ok_or_else(|| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "mounted dispatch {}.{} declared by resident model package is missing",
                    shader.component_id, shader.node_id
                ))
            })?;
        let family = placed_plan
            .reusable_kernel_plan
            .family(&dispatch.reusable_family_id)
            .ok_or_else(|| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "reusable kernel family {:?} declared by mounted dispatch {}.{} is missing",
                    dispatch.reusable_family_id, shader.component_id, shader.node_id
                ))
            })?;
        if loaded_families.insert(dispatch.reusable_family_id.clone()) {
            let spirv_words =
                load_required_resident_model_package_shader(manifest_dir, &shader.shader_path)?;
            reusable_word_count = reusable_word_count
                .checked_add(spirv_words.len())
                .ok_or_else(|| {
                    VulkanResidentTokenModelPackageError::new(
                        "reusable kernel artifact word count overflowed",
                    )
                })?;
            loaded_artifacts.push(VulkanLoadedReusableKernelArtifact {
                artifact: VulkanReusableKernelArtifact::from_family(
                    family,
                    shader.shader_path.clone(),
                )
                .with_local_size_x(shader.local_size_x)
                .with_workgroup_count_x(shader.workgroup_count_x),
                resolved_path: resolve_resident_model_package_path(
                    manifest_dir,
                    &shader.shader_path,
                ),
                words: spirv_words,
            });
        }
        for contract in shader
            .physical_execution_contracts
            .iter()
            .filter(|contract| contract.strategy.is_distributed())
        {
            for (artifact_index, identity) in contract.artifacts.iter().enumerate() {
                if identity.role != nerve_execution_contracts::ArtifactRole::Primary {
                    continue;
                }
                let artifact = physical_contract_kernel_artifact(
                    family,
                    contract,
                    artifact_index,
                    identity,
                )?;
                if let Some(existing) =
                    loaded_physical_metadata.get(&artifact.artifact_id)
                {
                    if existing != &artifact {
                        return Err(VulkanResidentTokenModelPackageError::new(format!(
                            "physical kernel artifact {:?} has conflicting metadata",
                            artifact.artifact_id
                        )));
                    }
                    continue;
                }
                loaded_physical_metadata
                    .insert(artifact.artifact_id.clone(), artifact.clone());
                let spirv_words = load_required_resident_model_package_shader(
                    manifest_dir,
                    &identity.path,
                )?;
                physical_word_count = physical_word_count
                    .checked_add(spirv_words.len())
                    .ok_or_else(|| {
                        VulkanResidentTokenModelPackageError::new(
                            "physical kernel artifact word count overflowed",
                        )
                    })?;
                loaded_physical_artifacts.push(VulkanLoadedPhysicalKernelArtifact {
                    artifact,
                    resolved_path: resolve_resident_model_package_path(
                        manifest_dir,
                        &identity.path,
                    ),
                    words: spirv_words,
                });
            }
        }
    }

    let required_families: BTreeSet<&str> = placed_plan
        .reusable_kernel_plan
        .families
        .iter()
        .map(|family| family.family_id.as_str())
        .collect();
    let loaded_family_ids: BTreeSet<&str> = loaded_artifacts
        .iter()
        .map(|artifact| artifact.artifact.family_id.as_str())
        .collect();
    let missing_families = required_families
        .difference(&loaded_family_ids)
        .copied()
        .collect::<Vec<_>>();
    if !missing_families.is_empty() {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "resident model package is missing shaders for reusable kernel families: {}",
            missing_families.join(", ")
        )));
    }

    Ok(VulkanLoadedKernelArtifactCatalog {
        reusable_artifacts: loaded_artifacts,
        physical_artifacts: loaded_physical_artifacts,
        reusable_word_count,
        physical_word_count,
    })
}

fn physical_contract_kernel_artifact(
    family: &VulkanReusableKernelFamily,
    contract: &nerve_execution_contracts::PhysicalExecutionContract,
    artifact_index: usize,
    identity: &nerve_execution_contracts::ArtifactIdentity,
) -> Result<VulkanPhysicalKernelArtifact, VulkanResidentTokenModelPackageError> {
    Ok(VulkanPhysicalKernelArtifact {
        artifact_id: physical_execution_artifact_id(&contract.contract_id, artifact_index),
        op: contract.operation_family.clone(),
        path: identity.path.clone(),
        entry_point: identity.entry_point.clone(),
        local_size_x: physical_contract_geometry_u32(contract, "local_size_x")?,
        workgroup_count_x: physical_contract_geometry_u32(contract, "workgroup_count_x")?,
        descriptor_signature: physical_contract_descriptor_signature(family, contract)?,
        push_constants: physical_contract_push_constants(contract)?,
        stream_control_binding: None,
    })
}

fn physical_contract_descriptor_signature(
    family: &VulkanReusableKernelFamily,
    contract: &nerve_execution_contracts::PhysicalExecutionContract,
) -> Result<Vec<VulkanKernelDescriptorSlotSignature>, VulkanResidentTokenModelPackageError> {
    let mut required = BTreeMap::<usize, VulkanKernelDescriptorUsage>::new();
    let mut insert = |binding: u32, usage: VulkanKernelDescriptorUsage| {
        let binding = usize::try_from(binding).map_err(|_| {
            VulkanResidentTokenModelPackageError::new(format!(
                "physical contract {:?} descriptor binding exceeds usize",
                contract.contract_id
            ))
        })?;
        if required.insert(binding, usage).is_some() {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "physical contract {:?} reuses descriptor binding {binding}",
                contract.contract_id
            )));
        }
        Ok(())
    };
    for input in &contract.inputs {
        insert(input.binding, VulkanKernelDescriptorUsage::InputSignal)?;
    }
    for output in &contract.outputs {
        insert(output.binding, VulkanKernelDescriptorUsage::OutputSignal)?;
    }
    for parameter in &contract.parameter_partitions {
        insert(parameter.binding, VulkanKernelDescriptorUsage::Parameter)?;
    }
    for selected in &contract.selected_resource_partitions {
        insert(
            selected.address_table_binding,
            VulkanKernelDescriptorUsage::DynamicResourceAddressTable,
        )?;
        insert(
            selected.parameter_slots_binding,
            VulkanKernelDescriptorUsage::DynamicResourceParameterSlots,
        )?;
    }

    let mut signature = Vec::with_capacity(required.len());
    for (binding, usage) in required {
        let candidates = family
            .descriptor_signature
            .iter()
            .filter(|slot| slot.binding == binding)
            .collect::<Vec<_>>();
        let [slot] = candidates.as_slice() else {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "physical contract {:?} binding {binding} does not identify exactly one canonical descriptor slot",
                contract.contract_id
            )));
        };
        if slot.usage != usage {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "physical contract {:?} binding {binding} requires {usage:?}, canonical slot declares {:?}",
                contract.contract_id, slot.usage
            )));
        }
        signature.push((*slot).clone());
    }
    Ok(signature)
}

fn physical_contract_geometry_u32(
    contract: &nerve_execution_contracts::PhysicalExecutionContract,
    dimension: &str,
) -> Result<u32, VulkanResidentTokenModelPackageError> {
    contract
        .geometry
        .dimensions
        .get(dimension)
        .copied()
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| *value > 0)
        .ok_or_else(|| {
            VulkanResidentTokenModelPackageError::new(format!(
                "physical contract {:?} has no positive u32 {dimension} geometry",
                contract.contract_id
            ))
        })
}

fn physical_contract_push_constants(
    contract: &nerve_execution_contracts::PhysicalExecutionContract,
) -> Result<Vec<VulkanKernelScalarBinding>, VulkanResidentTokenModelPackageError> {
    let launch = contract.partition_launch.as_ref().ok_or_else(|| {
        VulkanResidentTokenModelPackageError::new(format!(
            "distributed physical contract {:?} has no partition launch",
            contract.contract_id
        ))
    })?;
    if launch.origin == nerve_execution_contracts::PartitionOrigin::LocalZero {
        return Ok(Vec::new());
    }
    let origin = launch.origin_push_constant.as_ref().ok_or_else(|| {
        VulkanResidentTokenModelPackageError::new(format!(
            "distributed physical contract {:?} has no partition-origin control",
            contract.contract_id
        ))
    })?;
    let mut controls = vec![VulkanKernelScalarBinding {
        name: origin.clone(),
        scalar_type: "u32".to_string(),
        source: VulkanKernelScalarSource::PushConstant,
        canonical_u32: None,
    }];
    if launch.workgroup_x == nerve_execution_contracts::WorkgroupXMapping::Repeated {
        let count = launch.count_push_constant.as_ref().ok_or_else(|| {
            VulkanResidentTokenModelPackageError::new(format!(
                "distributed physical contract {:?} has no partition-count control",
                contract.contract_id
            ))
        })?;
        controls.push(VulkanKernelScalarBinding {
            name: count.clone(),
            scalar_type: "u32".to_string(),
            source: VulkanKernelScalarSource::PushConstant,
            canonical_u32: None,
        });
    }
    Ok(controls)
}

fn load_resident_component_batch_kernels(
    device: &VulkanComputeDevice,
    manifest_dir: &Path,
    component_executions: &[VulkanResidentComponentExecutionSpec],
    prepared_plan: &VulkanPreparedDispatchPlan,
) -> Result<Vec<VulkanResidentComponentBatchKernelArtifact>, VulkanResidentTokenModelPackageError> {
    let mut artifacts = Vec::new();
    for component in component_executions {
        for kernel in &component.kernels {
            if !matches!(
                kernel.batch_mode,
                VulkanResidentComponentKernelBatchMode::WeightShared
                    | VulkanResidentComponentKernelBatchMode::CausalScan
            ) || prepared_plan
                .dispatch(&component.component_id, &kernel.node_id)
                .is_none()
            {
                continue;
            }
            let supported = kernel
                .batch_implementations
                .iter()
                .filter(|implementation| batch_implementation_is_supported(device, implementation))
                .collect::<Vec<_>>();
            for implementation in supported {
                artifacts.push(VulkanResidentComponentBatchKernelArtifact {
                    component_id: component.component_id.clone(),
                    node_id: kernel.node_id.clone(),
                    execution_domain: implementation.execution_domain,
                    batch_mode: kernel.batch_mode,
                    lane_tile_width: implementation.lane_tile_width as usize,
                    selection_priority: implementation.selection_priority,
                    independent_candidate_compatible: implementation
                        .independent_candidate_compatible,
                    causal_sequence_compatible: implementation.causal_sequence_compatible,
                    parallel_block_compatible: implementation.parallel_block_compatible,
                    device_requirements: implementation.device_requirements.clone(),
                    stages: implementation
                        .stages
                        .iter()
                        .map(|stage| {
                            Ok(VulkanResidentComponentBatchStageArtifact {
                                shader_path: stage.shader_path.clone(),
                                spirv_words: load_required_resident_model_package_shader(
                                    manifest_dir,
                                    &stage.shader_path,
                                )?,
                                local_size_x: stage.local_size_x,
                                workgroup_count_x: stage.workgroup_count_x,
                                descriptor_bindings: stage.descriptor_bindings.clone(),
                                state_snapshot_binding: stage.state_snapshot_binding,
                                state_snapshot_source_binding: stage
                                    .state_snapshot_source_binding,
                                control: stage.control,
                                indirect_dispatch_byte_offset: stage
                                    .indirect_dispatch_byte_offset,
                                dispatch_y_from_batch_width: stage
                                    .dispatch_y_from_batch_width,
                            })
                        })
                        .collect::<Result<Vec<_>, VulkanResidentTokenModelPackageError>>()?,
                });
            }
        }
    }
    Ok(artifacts)
}

fn batch_implementation_is_supported(
    device: &VulkanComputeDevice,
    implementation: &VulkanResidentComponentBatchImplementationSpec,
) -> bool {
    batch_device_requirements_are_supported(
        device,
        &implementation.device_requirements,
        implementation.stages.iter().map(|stage| stage.local_size_x),
    )
}

fn batch_kernel_artifact_is_supported(
    device: &VulkanComputeDevice,
    artifact: &VulkanResidentComponentBatchKernelArtifact,
) -> bool {
    batch_device_requirements_are_supported(
        device,
        &artifact.device_requirements,
        artifact.stages.iter().map(|stage| stage.local_size_x),
    )
}

fn batch_device_requirements_are_supported(
    device: &VulkanComputeDevice,
    requirements: &VulkanResidentVulkanDeviceRequirements,
    local_size_x_values: impl IntoIterator<Item = u32>,
) -> bool {
    local_size_x_values
        .into_iter()
        .all(|local_size_x| device.supports_compute_local_size_x(local_size_x))
        && requirements
            .vulkan_device_extensions
            .iter()
            .all(|extension| device.has_enabled_device_extension(extension))
        && requirements
            .vulkan_features
            .iter()
            .all(|feature| device.has_enabled_shader_feature(*feature))
        && requirements
            .subgroup_operations
            .iter()
            .all(|operation| device.supports_subgroup_operation(*operation))
        && requirements
            .cooperative_bfloat16_shape
            .is_none_or(|[m, n, k]| device.supports_cooperative_bfloat16_shape(m, n, k))
        && requirements
            .cooperative_float8_e4m3_shape
            .is_none_or(|[m, n, k]| device.supports_cooperative_float8_e4m3_shape(m, n, k))
        && requirements
            .subgroup_size
            .is_none_or(|subgroup_size| device.subgroup_size() == subgroup_size)
}

fn load_required_resident_model_package_shader(
    manifest_dir: &Path,
    shader_path: &str,
) -> Result<Vec<u32>, VulkanResidentTokenModelPackageError> {
    let resolved_path = resolve_resident_model_package_path(manifest_dir, shader_path);
    if resolved_path
        .extension()
        .and_then(|extension| extension.to_str())
        != Some("spv")
    {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "resident model package shader {:?} is not a compiled SPIR-V artifact",
            resolved_path
        )));
    }
    read_spirv_words(&resolved_path).map_err(|error| {
        VulkanResidentTokenModelPackageError::new(format!(
            "failed to load compiled Vulkan shader {:?}: {error}",
            resolved_path
        ))
    })
}

fn load_resident_sampler_kernels(
    manifest_dir: &Path,
    package: &VulkanResidentSamplerPackageSpec,
) -> Result<Vec<VulkanResidentSamplerKernelArtifact>, VulkanResidentTokenModelPackageError> {
    package
        .kernels
        .iter()
        .map(|kernel| {
            Ok(VulkanResidentSamplerKernelArtifact {
                role: kernel.role.clone(),
                spirv_words: load_required_resident_model_package_shader(
                    manifest_dir,
                    &kernel.shader_path,
                )?,
                local_size_x: kernel.local_size_x,
                workgroup_count_x: kernel.workgroup_count_x,
            })
        })
        .collect()
}
