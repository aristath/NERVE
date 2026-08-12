pub struct VulkanResidentInProcessPlacedModelPackage {
    pub package_id: String,
    pub execution_scope: String,
    pub runtime_execution_identity: String,
    pub resource_residency_policy: ResourceResidencyPolicy,
    pub input_device_id: String,
    pub output_device_id: String,
    pub dynamic_state_capacity_activations: usize,
    pub device_ids: Vec<String>,
    pub device_count: usize,
    pub hosted_component_count: usize,
    pub transducer_parameter_count: usize,
    pub transducer_parameter_bytes: usize,
    input_transducer_parameter_buffers: Arc<VulkanPermanentParameterBuffers>,
    output_transducer_parameter_buffers: Arc<VulkanPermanentParameterBuffers>,
    input_transducer_spirv_words: Vec<u32>,
    input_transducer_batch_spirv_words: Vec<u32>,
    input_transducer_batch_control: VulkanResidentComponentBatchControlSpec,
    embedding_norm_spirv_words: Vec<u32>,
    embedding_norm_batch_spirv_words: Vec<u32>,
    embedding_norm_batch_lane_tile_width: u32,
    tied_projection_spirv_words: Vec<u32>,
    tied_projection_batch_spirv_words: Vec<u32>,
    projection_batch_lane_tile_width: u32,
    sampler_kernels: Vec<VulkanResidentSamplerKernelArtifact>,
    input_transducer_spec: VulkanResidentInputEmbeddingTransducerSpec,
    output_transducer_spec: VulkanResidentOutputTransducerSpec,
    sampler_spec: VulkanResidentSamplerSpec,
    device_slices: Vec<Arc<VulkanResidentModelPackageDeviceSlice>>,
    speculative_decoders: Vec<VulkanResidentSpeculativeDecoderModelPackage>,
    distributed_execution_plans: VulkanDistributedExecutionPlanSet,
    distributed_activation_plan: VulkanDistributedActivationBufferPlan,
    distributed_parameter_allocation_plan: VulkanDistributedParameterAllocationPlan,
    distributed_parameter_exclusion_plan: VulkanDistributedParameterExclusionPlan,
    physical_execution_residency_plan: VulkanRuntimePhysicalExecutionResidencyPlan,
    mounted_boundary_routes: BTreeMap<usize, VulkanRuntimeMountedBoundaryRoute>,
    selected_resource_placements: Vec<VulkanSelectedResourcePlacementPlan>,
    distributed_selected_resource_execution_ownership_plan:
        VulkanDistributedSelectedResourceStorePlan,
    distributed_selected_resource_store_plan: VulkanDistributedSelectedResourceStorePlan,
    distributed_loaded_manifest: VulkanLoadedKernelArtifactCatalog,
    distributed_parameter_buffers: Arc<VulkanDistributedParameterBuffers>,
    distributed_dynamic_resource_buffers: BTreeMap<String, Arc<VulkanDynamicResourceBuffers>>,
    compiled_resource_device_stores: BTreeMap<String, Arc<VulkanCompiledResourceDeviceStore>>,
    compiled_resource_physical_placements: Vec<VulkanCompiledResourcePhysicalPlacement>,
    runtime_component_instances: Vec<VulkanRuntimeComponentInstance>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VulkanMountedPhysicalExecutionSummary {
    pub tensor_parallel_island_count: usize,
    pub whole_expert_parallel_island_count: usize,
    pub intra_expert_tensor_parallel_island_count: usize,
    pub hybrid_island_count: usize,
    pub selected_resource_placement_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanRuntimeComponentInstance {
    instance_id: String,
    source_component_id: String,
    device_id: String,
    execution_index: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentStreamStateDeclaration {
    pub key: TransientStateKey,
    pub block_shape: TransientStateBlockShape,
}

impl VulkanResidentInProcessPlacedModelPackage {
    pub fn stream_execution_class_id(&self) -> String {
        self.runtime_execution_identity.clone()
    }

    pub fn compiled_resource_device_store(
        &self,
        device_id: &str,
    ) -> Option<&VulkanCompiledResourceDeviceStore> {
        self.compiled_resource_device_stores
            .get(device_id)
            .map(Arc::as_ref)
    }

    fn adaptive_resource_stores(&self) -> Vec<Arc<VulkanCompiledResourceDeviceStore>> {
        let mut stores = BTreeMap::new();
        for store in self.compiled_resource_device_stores.values() {
            if store.supports_adaptive_retiering() || store.supports_adaptive_representations() {
                stores
                    .entry(store.device_id().to_string())
                    .or_insert_with(|| Arc::clone(store));
            }
        }
        stores.into_values().collect()
    }

    pub fn compiled_resource_physical_placements(
        &self,
    ) -> &[VulkanCompiledResourcePhysicalPlacement] {
        &self.compiled_resource_physical_placements
    }

    pub fn distributed_selected_resource_store_plan(
        &self,
    ) -> &VulkanDistributedSelectedResourceStorePlan {
        &self.distributed_selected_resource_store_plan
    }

    pub fn distributed_selected_resource_execution_ownership_plan(
        &self,
    ) -> &VulkanDistributedSelectedResourceStorePlan {
        &self.distributed_selected_resource_execution_ownership_plan
    }

    pub fn selected_resource_placements(&self) -> &[VulkanSelectedResourcePlacementPlan] {
        &self.selected_resource_placements
    }

    pub fn decode_physical_execution_summary(&self) -> VulkanMountedPhysicalExecutionSummary {
        mounted_physical_execution_summary(
            &self.distributed_execution_plans.decode.execution_islands,
            self.selected_resource_placements.len(),
        )
    }

    fn compiled_resource_load_required_count(
        &self,
    ) -> Result<u64, VulkanCompiledResourceDeviceStoreError> {
        self.compiled_resource_physical_placements
            .iter()
            .try_fold(0u64, |total, placement| {
                let logical_device_id = placement.logical_device_ids.first().ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled resource physical placement has no logical device",
                    )
                })?;
                let store = self
                    .compiled_resource_device_stores
                    .get(logical_device_id)
                    .ok_or_else(|| {
                        VulkanCompiledResourceDeviceStoreError::new(format!(
                            "compiled resource physical placement {:?} has no device store",
                            placement.store_id
                        ))
                    })?;
                total
                    .checked_add(store.statistics()?.miss_count)
                    .ok_or_else(|| {
                        VulkanCompiledResourceDeviceStoreError::new(
                            "compiled resource load-required count overflowed",
                        )
                    })
            })
    }

    pub fn resident_state_buffer(
        &self,
        key: &TransientStateKey,
    ) -> Option<&VulkanResidentStateBuffer> {
        self.device_slices
            .iter()
            .flat_map(|device_slice| {
                device_slice
                    .placed_plan
                    .placed_resident_plan
                    .resident_plan
                    .stream_state_buffers
                    .iter()
            })
            .find(|state| {
                state.component_id == key.node_instance_id && state.state_id == key.state_id
            })
    }

    pub fn transient_state_declarations(
        &self,
    ) -> Result<Vec<VulkanResidentStreamStateDeclaration>, VulkanResidentTokenModelPackageError>
    {
        let mut declarations = BTreeMap::new();
        for device_slice in &self.device_slices {
            for state in &device_slice
                .placed_plan
                .placed_resident_plan
                .resident_plan
                .stream_state_buffers
            {
                if let Some(declaration) = transient_state_declaration_for_resident_state_buffer(
                    state,
                    self.dynamic_state_capacity_activations,
                )? {
                    declarations.insert(declaration.key.clone(), declaration);
                }
            }
        }
        Ok(declarations.into_values().collect())
    }
}

pub(crate) fn mounted_physical_execution_summary(
    islands: &[VulkanPhysicalExecutionIslandPlan],
    selected_resource_placement_count: usize,
) -> VulkanMountedPhysicalExecutionSummary {
    let mut summary = VulkanMountedPhysicalExecutionSummary {
        selected_resource_placement_count,
        ..VulkanMountedPhysicalExecutionSummary::default()
    };
    for island in islands {
        use nerve_execution_contracts::ExecutionStrategy;
        let saw_tensor_parallel = island
            .dispatches
            .iter()
            .any(|dispatch| dispatch.execution_strategy == ExecutionStrategy::TensorParallel);
        let saw_whole_expert = island
            .dispatches
            .iter()
            .any(|dispatch| dispatch.execution_strategy == ExecutionStrategy::ExpertParallel);
        let saw_intra_expert = island.dispatches.iter().any(|dispatch| {
            dispatch.execution_strategy == ExecutionStrategy::TensorParallelExpert
        });
        match (saw_tensor_parallel, saw_whole_expert, saw_intra_expert) {
            (true, false, false) => summary.tensor_parallel_island_count += 1,
            (false, true, false) => summary.whole_expert_parallel_island_count += 1,
            (false, false, true) => summary.intra_expert_tensor_parallel_island_count += 1,
            (false, false, false) => {}
            _ => summary.hybrid_island_count += 1,
        }
    }
    summary
}

fn transient_state_declaration_for_resident_state_buffer(
    state: &VulkanResidentStateBuffer,
    package_dynamic_state_capacity_activations: usize,
) -> Result<Option<VulkanResidentStreamStateDeclaration>, VulkanResidentTokenModelPackageError> {
    let block_shape = match state.bytes_per_activation {
        Some(bytes_per_activation) => {
            let maximum_activation_count = state
                .max_dynamic_activations
                .map(|limit| limit.min(package_dynamic_state_capacity_activations))
                .unwrap_or(package_dynamic_state_capacity_activations);
            let activation_capacity = maximum_activation_count.min(VULKAN_BACKEND_LOOP_MAX_WINDOW);
            TransientStateBlockShape::new(bytes_per_activation, activation_capacity)
                .and_then(|shape| shape.with_maximum_activation_count(maximum_activation_count))
        }
        None => match state.static_bytes {
            Some(static_bytes) => TransientStateBlockShape::mutable_singleton(static_bytes),
            None => return Ok(None),
        },
    }
    .map_err(|error| {
        VulkanResidentTokenModelPackageError::new(format!(
            "failed to declare transient state for {}.{}: {error}",
            state.component_id, state.state_id
        ))
    })?;
    Ok(Some(VulkanResidentStreamStateDeclaration {
        key: TransientStateKey::new(state.component_id.clone(), state.state_id.clone()),
        block_shape,
    }))
}
