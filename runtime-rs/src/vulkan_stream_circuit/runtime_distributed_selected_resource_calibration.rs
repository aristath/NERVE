struct VulkanCalibrationSelectedResourceMount {
    stores: BTreeMap<String, Arc<VulkanCompiledResourceDeviceStore>>,
    dynamic_buffers: BTreeMap<String, Arc<VulkanDynamicResourceBuffers>>,
    transaction_predicates: BTreeMap<String, Arc<VulkanResidentBuffer>>,
    resident_transient_bytes_by_device: BTreeMap<String, usize>,
    resident_host_transient_bytes: usize,
}

impl VulkanCalibrationSelectedResourceMount {
    fn empty() -> Self {
        Self {
            stores: BTreeMap::new(),
            dynamic_buffers: BTreeMap::new(),
            transaction_predicates: BTreeMap::new(),
            resident_transient_bytes_by_device: BTreeMap::new(),
            resident_host_transient_bytes: 0,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn mount_distributed_calibration_selected_resources(
    manifest_dir: &Path,
    execution_scope: &str,
    contract: &Arc<CompiledResourceResidencyContract>,
    execution_plan: &VulkanDistributedExecutionPlan,
    logical_devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
    maximum_total_payload_bytes: usize,
    maximum_payload_bytes_by_device: &BTreeMap<String, usize>,
) -> Result<Option<VulkanCalibrationSelectedResourceMount>, VulkanResidentTokenModelPackageError> {
    let has_selected_resources = execution_plan
        .dispatches
        .iter()
        .any(|dispatch| !dispatch.selected_resource_partitions.is_empty());
    if !has_selected_resources {
        return Ok(Some(VulkanCalibrationSelectedResourceMount::empty()));
    }
    let store_plan = VulkanDistributedSelectedResourceStorePlan::from_execution_plan(execution_plan)
        .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
    if store_plan.devices.len() != logical_devices.len()
        || maximum_payload_bytes_by_device.len() != logical_devices.len()
        || store_plan
            .devices
            .iter()
            .any(|plan| {
                !logical_devices.contains_key(&plan.device_id)
                    || !maximum_payload_bytes_by_device.contains_key(&plan.device_id)
            })
    {
        return distributed_calibration_error(
            "distributed selected-resource calibration does not cover every participant",
        );
    }
    let minimum_load_wave_bytes = store_plan.devices.iter().try_fold(
        0usize,
        |total, plan| total.checked_add(plan.maximum_load_wave_bytes),
    ).ok_or_else(|| {
        distributed_calibration_error_value(
            "distributed selected-resource minimum load-wave bytes overflowed",
        )
    })?;
    if minimum_load_wave_bytes > maximum_total_payload_bytes {
        return Ok(None);
    }

    let layout = Arc::new(
        VulkanCompiledResourceAddressLayout::from_contract(contract)
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?,
    );
    let maximum_ranges_per_group = compiled_resource_maximum_ranges_per_group(contract)
        .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
    let mut remaining_payload_budget = maximum_total_payload_bytes;
    let mut remaining_minimum_wave_bytes = minimum_load_wave_bytes;
    let mut mounted = VulkanCalibrationSelectedResourceMount::empty();

    for device_plan in &store_plan.devices {
        remaining_minimum_wave_bytes = remaining_minimum_wave_bytes
            .checked_sub(device_plan.maximum_load_wave_bytes)
            .expect("selected-resource minimum wave was accumulated above");
        let device = logical_devices.get(&device_plan.device_id).ok_or_else(|| {
            distributed_calibration_error_value(format!(
                "distributed selected-resource calibration has no device {:?}",
                device_plan.device_id,
            ))
        })?;
        for selector in &device_plan.selectors {
            if selector.execution_scope != execution_scope {
                return distributed_calibration_error(format!(
                    "distributed selector {:?} belongs to execution scope {:?}, expected {execution_scope:?}",
                    selector.selector_id, selector.execution_scope,
                ));
            }
        }
        let ownership = compiled_resource_selector_ownership_from_distributed_device_plan(
            contract,
            device_plan,
        )
        .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let upload_alignment = compiled_resource_upload_alignment(contract, device)
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let store_residency = plan_compiled_resource_store_residency_for_ownership(
            contract,
            &layout,
            &ownership,
            device_plan.maximum_atomic_group_bytes,
            upload_alignment,
        )
        .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        if store_residency.maximum_load_wave_payload_bytes
            != device_plan.maximum_load_wave_bytes
        {
            return distributed_calibration_error(format!(
                "distributed selected-resource load-wave contract changed from {} to {} bytes on {:?}",
                device_plan.maximum_load_wave_bytes,
                store_residency.maximum_load_wave_payload_bytes,
                device_plan.device_id,
            ));
        }
        let fixed_device_bytes = store_residency
            .fixed_device_bytes()
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let admission = device
            .admit_device_local_memory(u64::try_from(fixed_device_bytes).unwrap_or(u64::MAX))
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let safe_dynamic_bytes = usize::try_from(admission.allocatable_bytes).unwrap_or(usize::MAX);
        let maximum_alignment_padding =
            store_residency.maximum_dynamic_allocation_padding_bytes;
        let retained_representation_cache_bytes = store_residency
            .retained_representation_cache_device_bytes()
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let payload_budget_for_device = remaining_payload_budget
            .checked_sub(remaining_minimum_wave_bytes)
            .expect("minimum wave budget was validated above");
        let device_payload_budget = maximum_payload_bytes_by_device[&device_plan.device_id];
        let resident_payload_capacity = device_plan
            .total_addressable_bytes
            .min(
                compiled_resource_source_payload_capacity(
                    device_plan.total_addressable_bytes,
                    safe_dynamic_bytes,
                    &store_residency,
                )
                .map_err(|error| distributed_calibration_error_value(error.to_string()))?,
            )
            .min(payload_budget_for_device);
        let resident_payload_capacity = resident_payload_capacity.min(device_payload_budget);
        if resident_payload_capacity < device_plan.maximum_load_wave_bytes {
            return Ok(None);
        }
        let allocation_capacity = resident_payload_capacity
            .checked_add(maximum_alignment_padding)
            .and_then(|bytes| bytes.checked_add(retained_representation_cache_bytes))
            .ok_or_else(|| {
                distributed_calibration_error_value(
                    "distributed selected-resource allocation capacity overflowed",
                )
            })?;
        let store_id = format!(
            "calibration:selected_resources:{}:{}",
            device.physical_device_id(), device_plan.device_id,
        );
        let store = Arc::new(
            VulkanCompiledResourceDeviceStore::new_tiered_with_selector_ownership(
                device,
                ResourceResidencyPolicy::DemandPaged,
                store_id,
                device.physical_device_id(),
                vec![device_plan.device_id.clone()],
                manifest_dir,
                Arc::clone(contract),
                Arc::clone(&layout),
                ownership,
                resident_payload_capacity,
                resident_payload_capacity,
                0,
                allocation_capacity,
                device_plan.maximum_atomic_group_bytes,
                maximum_ranges_per_group,
                0,
                0,
                store_residency.metadata_device_bytes,
                None,
            )
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?,
        );
        store
            .register_device_memory_reclaimer(device)
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let component_ids = device_plan
            .selectors
            .iter()
            .map(|selector| selector.component_id.clone())
            .collect::<BTreeSet<_>>();
        let dynamic_buffers = store
            .dynamic_buffers_for_components(device, execution_scope, &component_ids)
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        store
            .mark_mount_complete()
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let predicate = Arc::new(
            device
                .create_conditional_resident_buffer(size_of::<u32>())
                .map_err(|error| distributed_calibration_error_value(error.to_string()))?,
        );
        predicate
            .write_bytes(&1u32.to_le_bytes())
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let store_report = store
            .residency_report()
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let device_transient_bytes = store_report
            .metadata_device_bytes
            .checked_add(predicate.byte_capacity())
            .ok_or_else(|| {
                distributed_calibration_error_value(
                    "distributed selected-resource transient bytes overflowed",
                )
            })?;
        mounted.resident_host_transient_bytes = mounted
            .resident_host_transient_bytes
            .checked_add(store_report.transfer_staging_host_bytes)
            .ok_or_else(|| {
                distributed_calibration_error_value(
                    "distributed selected-resource host transient bytes overflowed",
                )
            })?;
        remaining_payload_budget = remaining_payload_budget
            .checked_sub(resident_payload_capacity)
            .expect("resident payload was capped by the remaining budget");
        mounted
            .stores
            .insert(device_plan.device_id.clone(), Arc::clone(&store));
        mounted
            .dynamic_buffers
            .insert(device_plan.device_id.clone(), dynamic_buffers);
        mounted
            .transaction_predicates
            .insert(device_plan.device_id.clone(), predicate);
        mounted
            .resident_transient_bytes_by_device
            .insert(device_plan.device_id.clone(), device_transient_bytes);
    }
    Ok(Some(mounted))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanRuntimeSelectedResourceExecutionCalibrationReport {
    pub physical_device_id: String,
    pub target: VulkanRuntimeSelectedResourceExecutionCalibrationTarget,
    pub resource_execution_class_id: String,
    pub resource_payload_byte_count: usize,
    pub selector_selection_count: usize,
    pub warmup_execution_ns: u64,
    pub measured_execution_ns: u64,
    pub warmup_output_digest: String,
    pub warmup_output_artifact: Option<VulkanPlacementOutputArtifact>,
    pub measured_output_digest: String,
    pub measured_output_artifact: Option<VulkanPlacementOutputArtifact>,
    pub output_equivalence: VulkanPlacementOutputEquivalenceEvidence,
    pub state_digest: String,
    pub resident_parameter_bytes: usize,
    pub transient_peak_device_bytes: usize,
    pub transient_host_bytes: usize,
    pub execution_case: VulkanPlacementExecutionCaseIdentity,
}

impl VulkanRuntimeSelectedResourceExecutionCalibrationReport {
    fn validate(&self) -> Result<(), VulkanPlacementCalibrationCatalogError> {
        let [device] = self.execution_case.devices.as_slice() else {
            return Err(VulkanPlacementCalibrationCatalogError(
                "selected-resource execution calibration requires one physical device"
                    .to_string(),
            ));
        };
        let transaction_geometries = self
            .execution_case
            .operations
            .iter()
            .filter_map(|operation| match operation {
                VulkanPlacementOperationGeometry::SelectedResourceTransaction {
                    resource_execution_class_id,
                    selector_selection_count,
                    executed_resource_occurrence_count,
                    ..
                } => Some((
                    resource_execution_class_id,
                    *selector_selection_count,
                    *executed_resource_occurrence_count,
                )),
                _ => None,
            })
            .collect::<Vec<_>>();
        let expected_phase = match self.target.phase {
            VulkanTargetedComponentExecutionPhase::Decode => {
                nerve_execution_contracts::ExecutionPhase::Decode
            }
            VulkanTargetedComponentExecutionPhase::Prefill { .. } => {
                nerve_execution_contracts::ExecutionPhase::Prefill
            }
        };
        if self.physical_device_id.is_empty()
            || self.target.component.component_id.is_empty()
            || self.target.selector_id.is_empty()
            || self.target.selected_contract_ids.is_empty()
            || !valid_sha256_digest(&self.resource_execution_class_id)
            || self.resource_execution_class_id != self.target.resource_execution_class_id
            || self.resource_payload_byte_count == 0
            || self.selector_selection_count == 0
            || self.warmup_execution_ns == 0
            || self.measured_execution_ns == 0
            || self.warmup_output_digest.is_empty()
            || self.measured_output_digest.is_empty()
            || self.state_digest.is_empty()
            || self.resident_parameter_bytes < self.resource_payload_byte_count
            || self.transient_peak_device_bytes == 0
            || self.execution_case.strategy
                != VulkanPlacementExecutionStrategy::SelectedResourceTransaction
            || self.execution_case.behavior.phase != expected_phase
            || self.execution_case.behavior.shape.activation_batch_width
                != self.target.phase.activation_batch_width()
            || self.execution_case.contract_ids.iter().collect::<BTreeSet<_>>()
                != self.target.selected_contract_ids.iter().collect::<BTreeSet<_>>()
            || self.execution_case.input_physical_device_id != self.physical_device_id
            || self.execution_case.output_physical_device_id != self.physical_device_id
            || self.execution_case.owner_physical_device_id != self.physical_device_id
            || device.physical_device_id != self.physical_device_id
            || !self.execution_case.shards.is_empty()
            || !self.execution_case.transports.is_empty()
            || !matches!(
                transaction_geometries.as_slice(),
                [(class_id, selection_count, 1)]
                    if *class_id == &self.resource_execution_class_id
                        && *selection_count == self.selector_selection_count
            )
        {
            return Err(VulkanPlacementCalibrationCatalogError(
                "selected-resource execution calibration report is internally inconsistent"
                    .to_string(),
            ));
        }
        let output_equivalence = validate_vulkan_placement_output_equivalence(
            &self.execution_case.equivalence,
            &self.warmup_output_digest,
            self.warmup_output_artifact.as_ref(),
            &self.measured_output_digest,
            self.measured_output_artifact.as_ref(),
        )?;
        if output_equivalence != self.output_equivalence {
            return Err(VulkanPlacementCalibrationCatalogError(
                "selected-resource execution calibration carries stale output-equivalence evidence"
                    .to_string(),
            ));
        }
        Ok(())
    }

    fn canonical_reference(&self) -> VulkanPlacementCanonicalReference {
        VulkanPlacementCanonicalReference {
            behavior: self.execution_case.behavior.clone(),
            output_digest: self.warmup_output_digest.clone(),
            output_artifact: self.warmup_output_artifact.clone(),
            state_digest: self.state_digest.clone(),
        }
    }

    fn calibration_observation(&self) -> VulkanPlacementCalibrationObservation {
        VulkanPlacementCalibrationObservation {
            execution_case: self.execution_case.clone(),
            warmup_call_count: 1,
            measured_call_count: 1,
            complete_transaction: true,
            duration_ns: self.measured_execution_ns,
            useful_activation_count: 1,
            output_digest: self.measured_output_digest.clone(),
            output_artifact: self.measured_output_artifact.clone(),
            output_equivalence: self.output_equivalence.clone(),
            state_digest: self.state_digest.clone(),
            resident_bytes_by_physical_device: BTreeMap::from([(
                self.physical_device_id.clone(),
                self.resident_parameter_bytes,
            )]),
            transient_peak_bytes_by_physical_device: BTreeMap::from([(
                self.physical_device_id.clone(),
                self.transient_peak_device_bytes,
            )]),
            host_resident_bytes: 0,
            host_transient_peak_bytes: self.transient_host_bytes,
        }
    }

    pub fn execution_class_calibration(
        &self,
        load_wave: &VulkanRuntimeLoadWaveCalibrationReport,
    ) -> Result<
        VulkanPlacementSelectedResourceExecutionClassCalibration,
        VulkanPlacementCalibrationCatalogError,
    > {
        self.validate()?;
        load_wave.validate()?;
        if load_wave.physical_device_id != self.physical_device_id
            || load_wave.component_id != self.target.component.component_id
            || load_wave.selector_id != self.target.selector_id
            || load_wave.resource_indices != [self.target.resource_index]
            || load_wave.phase != self.execution_case.behavior.phase
            || load_wave.activation_batch_width
                != self.execution_case.behavior.shape.activation_batch_width
            || load_wave.loaded_group_count != 1
            || load_wave.loaded_byte_count != self.resource_payload_byte_count
        {
            return Err(VulkanPlacementCalibrationCatalogError(
                "selected-resource execution and load-wave reports do not describe the same exact resource transaction"
                    .to_string(),
            ));
        }
        Ok(VulkanPlacementSelectedResourceExecutionClassCalibration {
            resource_execution_class_id: self.resource_execution_class_id.clone(),
            resource_payload_byte_count: self.resource_payload_byte_count,
            execution_case: self.execution_case.clone(),
            lazy_load_wave_case: load_wave.execution_case.clone(),
        })
    }

    pub fn requirement(&self) -> VulkanPlacementSelectedResourceExecutionClassRequirement {
        VulkanPlacementSelectedResourceExecutionClassRequirement {
            resource_execution_class_id: self.resource_execution_class_id.clone(),
            compiled_execution_signature: self
                .execution_case
                .behavior
                .compiled_execution_signature
                .clone(),
            runtime_implementation_fingerprint: self
                .execution_case
                .behavior
                .runtime_implementation_fingerprint
                .clone(),
            phase: self.execution_case.behavior.phase,
            shape: self.execution_case.behavior.shape.clone(),
            artifact_digest: self.execution_case.artifact_digest.clone(),
            execution_graph_digest: self.execution_case.execution_graph_digest.clone(),
        }
    }
}

pub fn record_vulkan_runtime_selected_resource_execution_calibration_report(
    catalog: &mut VulkanPlacementCalibrationCatalog,
    report: &VulkanRuntimeSelectedResourceExecutionCalibrationReport,
) -> Result<(), VulkanPlacementCalibrationCatalogError> {
    report.validate()?;
    let mut updated = catalog.clone();
    updated.record_reference(report.canonical_reference())?;
    updated.record_observation(report.calibration_observation())?;
    *catalog = updated;
    Ok(())
}

struct VulkanRuntimeSelectedResourceExecutionRun {
    duration_ns: u64,
    output_digest: String,
    output_artifact: Option<VulkanPlacementOutputArtifact>,
}

struct VulkanRuntimeSelectedResourceExecutionSession {
    physical_device_id: String,
    logical_device_id: String,
    device: Rc<VulkanComputeDevice>,
    target: VulkanRuntimeSelectedResourceExecutionCalibrationTarget,
    selector: CompiledResourceSelector,
    resource_execution_class_id: String,
    resource_payload_byte_count: usize,
    output_scalar_format: VulkanPlacementScalarFormat,
    execution_plan: VulkanDistributedExecutionPlan,
    execution_case: VulkanPlacementExecutionCaseIdentity,
    runners: VulkanDistributedDispatchRunners,
    activation_buffers: VulkanDistributedActivationBuffers,
    resource_stores: BTreeMap<String, Arc<VulkanCompiledResourceDeviceStore>>,
    dynamic_resource_buffers: BTreeMap<String, Arc<VulkanDynamicResourceBuffers>>,
    transaction_predicates: BTreeMap<String, Arc<VulkanResidentBuffer>>,
    parameter_buffers: VulkanDistributedParameterBuffers,
    parameter_pool: VulkanResidentBufferPool,
    static_parameter_bytes: usize,
    transient_peak_device_bytes: usize,
    transient_host_bytes: usize,
}

impl VulkanRuntimeSelectedResourceExecutionSession {
    #[allow(clippy::too_many_arguments)]
    fn prepare(
        physical_device_id: &str,
        device: Rc<VulkanComputeDevice>,
        manifest_dir: &Path,
        runtime_model: &VulkanResidentRuntimeModel,
        target: &VulkanRuntimeSelectedResourceExecutionCalibrationTarget,
        maximum_total_resident_parameter_bytes: usize,
    ) -> Result<Option<Self>, VulkanResidentTokenModelPackageError> {
        let blueprint = VulkanRuntimeSelectedResourceExecutionBlueprint::prepare(
            &device,
            manifest_dir,
            runtime_model,
            &target.component,
            &target.selector_id,
            target.phase,
            &target.selected_contract_ids,
        )?;
        let VulkanRuntimeSelectedResourceExecutionBlueprint {
            logical_device_id,
            placed_model,
            tensor_index,
            contract,
            selector,
            loaded_manifest,
            contract_phase,
            full_execution_plan,
            resource_execution_class_ids,
        } = blueprint;
        if target.resource_index >= selector.resource_count {
            return distributed_calibration_error(
                "selected-resource execution selector does not own the requested component resource",
            );
        }
        let planned_resource_execution_class_id = resource_execution_class_ids
            .get(target.resource_index)
            .ok_or_else(|| {
                distributed_calibration_error_value(
                    "selected-resource execution class omits the requested resource",
                )
            })?;
        if planned_resource_execution_class_id != &target.resource_execution_class_id {
            return distributed_calibration_error(
                "selected-resource execution class changed after calibration target discovery",
            );
        }
        let resource_execution_class_id = target.resource_execution_class_id.clone();
        let execution_plan = full_execution_plan
            .isolated_selected_resource_transaction(
                &target.selector_id,
                target.resource_index,
                &logical_device_id,
                contract_phase,
            )
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let store_plan = VulkanDistributedSelectedResourceStorePlan::from_execution_plan(
            &execution_plan,
        )
        .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        if store_plan.device_count != 1
            || store_plan.unique_atomic_group_count != 1
            || store_plan.total_addressable_bytes == 0
        {
            return distributed_calibration_error(
                "isolated selected-resource execution did not produce one complete atomic resource",
            );
        }
        let resource_payload_byte_count = store_plan.total_addressable_bytes;
        let parameter_plan = VulkanDistributedParameterAllocationPlan::from_sampled_execution_plan(
            &execution_plan,
            &tensor_index,
        )
        .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let static_parameter_bytes = parameter_plan.allocations.iter().try_fold(
            0usize,
            |total, allocation| {
                total.checked_add(allocation.byte_count).ok_or_else(|| {
                    distributed_calibration_error_value(
                        "selected-resource static parameter bytes overflowed",
                    )
                })
            },
        )?;
        let Some(dynamic_parameter_capacity) = maximum_total_resident_parameter_bytes
            .checked_sub(static_parameter_bytes)
        else {
            return Ok(None);
        };
        if dynamic_parameter_capacity < resource_payload_byte_count {
            return Ok(None);
        }
        let parameter_pool = VulkanResidentBufferPool::default();
        parameter_pool
            .register_device(&logical_device_id, Rc::clone(&device))
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let parameter_buffers = VulkanDistributedParameterBuffers::allocate_and_load_from_pool(
            &parameter_plan,
            &tensor_index,
            &parameter_pool,
        )
        .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let logical_devices =
            BTreeMap::from([(logical_device_id.clone(), Rc::clone(&device))]);
        let Some(selected_resource_mount) = mount_distributed_calibration_selected_resources(
            manifest_dir,
            &placed_model.execution_scope,
            &contract,
            &execution_plan,
            &logical_devices,
            dynamic_parameter_capacity,
            &BTreeMap::from([(
                logical_device_id.clone(),
                dynamic_parameter_capacity,
            )]),
        )?
        else {
            return Ok(None);
        };
        let activation_plan =
            VulkanDistributedActivationBufferPlan::from_execution_plan(&execution_plan)
                .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let activation_buffers = VulkanDistributedActivationBuffers::allocate_for_lanes(
            &activation_plan,
            target.phase.activation_batch_width(),
            |device_id| {
                logical_devices
                    .get(device_id)
                    .map(Rc::as_ref)
                    .ok_or_else(|| format!("missing selected-resource device {device_id:?}"))
            },
        )
        .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let runners = VulkanDistributedDispatchRunners::create(
            &execution_plan,
            match target.phase {
                VulkanTargetedComponentExecutionPhase::Decode => {
                    VulkanResidentDistributedExecutionPhase::Decode
                }
                VulkanTargetedComponentExecutionPhase::Prefill { .. } => {
                    VulkanResidentDistributedExecutionPhase::Prefill
                }
            },
            &parameter_buffers,
            &selected_resource_mount.dynamic_buffers,
            &selected_resource_mount.stores,
            Some(&selected_resource_mount.transaction_predicates),
            &placed_model.execution_scope,
            &activation_buffers,
            &loaded_manifest,
            |device_id| {
                logical_devices
                    .get(device_id)
                    .map(Rc::as_ref)
                    .ok_or_else(|| format!("missing selected-resource device {device_id:?}"))
            },
        )
        .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let dispatch_work = execution_plan
            .dispatches
            .iter()
            .map(|dispatch| {
                let rows = dispatch.shards.iter().try_fold(0usize, |total, shard| {
                    total.checked_add(shard.row_count).ok_or_else(|| {
                        distributed_calibration_error_value(
                            "selected-resource dispatch rows overflowed",
                        )
                    })
                })?;
                Ok(VulkanRuntimeDistributedPlacementDispatchWork {
                    component_id: dispatch.component_id.clone(),
                    node_id: dispatch.node_id.clone(),
                    sampled_rows: rows,
                    full_rows: dispatch.output_rows,
                })
            })
            .collect::<Result<Vec<_>, VulkanResidentTokenModelPackageError>>()?;
        let requirement = selected_resource_execution_requirement(
            &target.component.signature_id,
            &selector,
            &resource_execution_class_id,
            &execution_plan,
            &loaded_manifest,
            target.phase,
        )?;
        let mut execution_case = distributed_calibration_execution_case(
            &[(physical_device_id.to_string(), Rc::clone(&device))],
            std::slice::from_ref(&logical_device_id),
            &execution_plan,
            &loaded_manifest,
            requirement.compiled_execution_signature,
            requirement.artifact_digest,
            requirement.execution_graph_digest,
            target.phase,
            &dispatch_work,
        )?;
        let contract_id = execution_case
            .contract_ids
            .first()
            .cloned()
            .ok_or_else(|| {
                distributed_calibration_error_value(
                    "selected-resource execution case has no physical contract",
                )
            })?;
        execution_case.operations.insert(
            0,
            VulkanPlacementOperationGeometry::SelectedResourceTransaction {
                contract_id,
                resource_execution_class_id: resource_execution_class_id.clone(),
                selector_selection_count: selector
                    .encoding
                    .selection_count_per_activation,
                executed_resource_occurrence_count: 1,
            },
        );
        execution_case.strategy =
            VulkanPlacementExecutionStrategy::SelectedResourceTransaction;
        execution_case.shards.clear();
        execution_case.transports.clear();
        execution_case.behavior.input_fixture_digest = selected_resource_fixture_digest(
            &selector,
            target.resource_index,
            target.phase.activation_batch_width(),
        )?;
        let output_scalar_format = match placed_model.package.activation_element_bytes {
            Some(2) => VulkanPlacementScalarFormat::Bf16,
            Some(4) => VulkanPlacementScalarFormat::F32,
            bytes => {
                return distributed_calibration_error(format!(
                    "selected-resource execution cannot validate {bytes:?}-byte activation scalars",
                ));
            }
        };
        if execution_case
            .equivalence
            .output_scalar_format
            .is_some_and(|format| format != output_scalar_format)
        {
            return distributed_calibration_error(
                "selected-resource output scalar format disagrees with its compiled activation layout",
            );
        }
        let (mut transient_device_bytes, transient_host_bytes) =
            distributed_calibration_transient_backing_bytes(
                std::slice::from_ref(&logical_device_id),
                &activation_plan,
            )?;
        let lane_count = target.phase.activation_batch_width();
        for bytes in transient_device_bytes.values_mut() {
            *bytes = bytes.checked_mul(lane_count).ok_or_else(|| {
                distributed_calibration_error_value(
                    "selected-resource activation bytes overflowed",
                )
            })?;
        }
        let transient_host_bytes = transient_host_bytes
            .checked_mul(lane_count)
            .and_then(|bytes| {
                bytes.checked_add(selected_resource_mount.resident_host_transient_bytes)
            })
            .ok_or_else(|| {
                distributed_calibration_error_value(
                    "selected-resource host transient bytes overflowed",
                )
            })?;
        let gate_transient_bytes = runners
            .selected_resource_transient_device_bytes_by_device()
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let transient_peak_device_bytes = transient_device_bytes[&logical_device_id]
            .checked_add(
                selected_resource_mount.resident_transient_bytes_by_device[&logical_device_id],
            )
            .and_then(|bytes| {
                bytes.checked_add(
                    gate_transient_bytes
                        .get(&logical_device_id)
                        .copied()
                        .unwrap_or(0),
                )
            })
            .ok_or_else(|| {
                distributed_calibration_error_value(
                    "selected-resource device transient bytes overflowed",
                )
            })?;
        Ok(Some(Self {
            physical_device_id: physical_device_id.to_string(),
            logical_device_id,
            device,
            target: target.clone(),
            selector,
            resource_execution_class_id,
            resource_payload_byte_count,
            output_scalar_format,
            execution_plan,
            execution_case,
            runners,
            activation_buffers,
            resource_stores: selected_resource_mount.stores,
            dynamic_resource_buffers: selected_resource_mount.dynamic_buffers,
            transaction_predicates: selected_resource_mount.transaction_predicates,
            parameter_buffers,
            parameter_pool,
            static_parameter_bytes,
            transient_peak_device_bytes,
            transient_host_bytes,
        }))
    }

    fn preload(&self) -> Result<(), VulkanResidentTokenModelPackageError> {
        let store = self
            .resource_stores
            .get(&self.logical_device_id)
            .ok_or_else(|| {
                distributed_calibration_error_value(
                    "selected-resource execution has no device store",
                )
            })?;
        let owner = DeviceResourceResidencyOwnerId::new(format!(
            "selected-resource-calibration:{}:{}",
            self.target.selector_id, self.target.resource_index,
        ))
        .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        store
            .load_selector_resource(
                &self.device,
                &self.target.selector_id,
                self.target.resource_index,
                owner,
            )
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let validation = store
            .validate_selector_resources_readback(
                &self.device,
                &self.target.selector_id,
                &[self.target.resource_index],
            )
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        if validation.resource_count != 1
            || validation.byte_count != self.resource_payload_byte_count
        {
            return distributed_calibration_error(
                "selected-resource execution preload did not publish the exact resource payload",
            );
        }
        Ok(())
    }

    fn resident_parameter_bytes(&self) -> Result<usize, VulkanResidentTokenModelPackageError> {
        let store = self.resource_stores[&self.logical_device_id]
            .residency_report()
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let dynamic_bytes = store
            .current_device_bytes
            .checked_sub(store.metadata_device_bytes)
            .ok_or_else(|| {
                distributed_calibration_error_value(
                    "selected-resource resident parameter accounting underflowed",
                )
            })?;
        self.static_parameter_bytes
            .checked_add(dynamic_bytes)
            .ok_or_else(|| {
                distributed_calibration_error_value(
                    "selected-resource resident parameter accounting overflowed",
                )
            })
    }

    fn execute_once(
        &self,
        maximum_duration: Duration,
    ) -> Result<VulkanRuntimeSelectedResourceExecutionRun, VulkanResidentTokenModelPackageError>
    {
        if maximum_duration.is_zero() {
            return distributed_calibration_error(
                "selected-resource execution requires a positive duration bound",
            );
        }
        self.write_fixture()?;
        let started = Instant::now();
        for runner in &self.runners.dispatches {
            if started.elapsed() >= maximum_duration {
                return distributed_calibration_error(
                    "selected-resource execution exceeded its duration bound",
                );
            }
            self.runners
                .run_dispatch(
                    &runner.planned.owner_device_id,
                    runner.planned.leader().dispatch_index,
                    |device_id| {
                        if device_id == self.logical_device_id {
                            Ok::<_, String>(self.device.as_ref())
                        } else {
                            Err(format!(
                                "selected-resource execution references unknown device {device_id:?}",
                            ))
                        }
                    },
                )
                .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        }
        let duration_ns = elapsed_nanoseconds(started).max(1);
        if started.elapsed() > maximum_duration {
            return distributed_calibration_error(
                "selected-resource execution exceeded its duration bound",
            );
        }
        let tail = self.execution_plan.dispatches.last().ok_or_else(|| {
            distributed_calibration_error_value(
                "selected-resource execution has no terminal operation",
            )
        })?;
        let output = self
            .activation_buffers
            .activation_buffer(
                &tail.owner_device_id,
                &tail.output_activation,
                &self.logical_device_id,
            )
            .ok_or_else(|| {
                distributed_calibration_error_value(
                    "selected-resource execution has no terminal activation buffer",
                )
            })?
            .read_bytes(
                tail.output_activation
                    .signal_byte_capacity
                    .checked_mul(self.target.phase.activation_batch_width())
                    .ok_or_else(|| {
                        distributed_calibration_error_value(
                            "selected-resource output byte count overflowed",
                        )
                    })?,
            )
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        validate_selected_resource_finite_output(&output, self.output_scalar_format)?;
        let (output_digest, output_artifact) = match self
            .execution_case
            .equivalence
            .output_scalar_format
        {
            Some(scalar_format) => {
                let artifact = VulkanPlacementOutputArtifact {
                    scalar_format,
                    segments: vec![VulkanPlacementOutputSegment {
                        binding: tail.output_activation.binding,
                        name: tail.output_activation.signal_id.clone(),
                        bytes: output,
                    }],
                };
                let digest = vulkan_placement_output_artifact_digest(&artifact)
                    .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
                (digest, Some(artifact))
            }
            None => (
                targeted_finalized_artifact_digest(Sha256::digest(output).as_slice()),
                None,
            ),
        };
        Ok(VulkanRuntimeSelectedResourceExecutionRun {
            duration_ns,
            output_digest,
            output_artifact,
        })
    }

    fn write_fixture(&self) -> Result<(), VulkanResidentTokenModelPackageError> {
        for allocation in &self.activation_buffers.allocations {
            let buffer = allocation
                .device_buffers
                .get(&self.logical_device_id)
                .ok_or_else(|| {
                    distributed_calibration_error_value(
                        "selected-resource activation allocation omits its device view",
                    )
                })?;
            buffer
                .write_bytes(&vec![0; buffer.byte_capacity()])
                .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        }
        for allocation in &self.activation_buffers.reduction_allocations {
            let buffer = allocation
                .device_buffers
                .get(&self.logical_device_id)
                .ok_or_else(|| {
                    distributed_calibration_error_value(
                        "selected-resource reduction allocation omits its device view",
                    )
                })?;
            buffer
                .write_bytes(&vec![0; buffer.byte_capacity()])
                .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        }
        for allocation in &self.activation_buffers.private_intermediate_allocations {
            let buffer = allocation
                .device_buffers
                .get(&self.logical_device_id)
                .ok_or_else(|| {
                    distributed_calibration_error_value(
                        "selected-resource private allocation omits its device view",
                    )
                })?;
            buffer
                .write_bytes(&vec![0; buffer.byte_capacity()])
                .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        }
        for predicate in self.transaction_predicates.values() {
            predicate
                .write_bytes(&1u32.to_le_bytes())
                .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        }

        let produced_signals = self
            .execution_plan
            .dispatches
            .iter()
            .map(|dispatch| dispatch.output_activation.signal_id.as_str())
            .collect::<BTreeSet<_>>();
        let mut fixtures_by_buffer = BTreeMap::<usize, Vec<u8>>::new();
        for dispatch in &self.execution_plan.dispatches {
            for activation in std::iter::once(&dispatch.input_activation)
                .chain(dispatch.auxiliary_input_activations.iter())
                .filter(|activation| !produced_signals.contains(activation.signal_id.as_str()))
            {
                let buffer = self
                    .activation_buffers
                    .activation_buffer(
                        &dispatch.owner_device_id,
                        activation,
                        &self.logical_device_id,
                    )
                    .ok_or_else(|| {
                        distributed_calibration_error_value(format!(
                            "selected-resource input signal {:?} has no activation buffer",
                            activation.signal_id,
                        ))
                    })?;
                let fixture = self.fixture_for_activation(activation, buffer.byte_capacity())?;
                let key = Arc::as_ptr(buffer) as usize;
                if let Some(existing) = fixtures_by_buffer.insert(key, fixture.clone())
                    && existing != fixture
                {
                    return distributed_calibration_error(format!(
                        "selected-resource input signals alias one buffer with incompatible fixtures at {}.slot_{}",
                        activation.component_id, activation.slot,
                    ));
                }
                buffer
                    .write_bytes(&fixture)
                    .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
            }
        }
        Ok(())
    }

    fn fixture_for_activation(
        &self,
        activation: &VulkanDistributedActivationSlot,
        buffer_byte_capacity: usize,
    ) -> Result<Vec<u8>, VulkanResidentTokenModelPackageError> {
        let base = if activation.signal_id == self.selector.selection_signal {
            Some(self.selector.encoding.calibration_word_base)
        } else if activation.signal_id == self.selector.execution_signal {
            Some(self.selector.execution_calibration_word_base)
        } else {
            None
        };
        let Some(base) = base else {
            return Ok(targeted_fixture_bytes(
                buffer_byte_capacity,
                0,
                activation.binding,
            ));
        };
        let lane_count = self.target.phase.activation_batch_width();
        let words = selected_resource_fixture_words(
            &self.selector,
            self.target.resource_index,
            lane_count,
            base,
        )?;
        let required_signal_bytes = self
            .selector
            .encoding
            .selection_count_per_activation
            .checked_mul(size_of::<u32>())
            .ok_or_else(|| {
                distributed_calibration_error_value(
                    "selected-resource fixture signal bytes overflowed",
                )
            })?;
        if required_signal_bytes > activation.signal_byte_capacity
            || activation
                .signal_byte_capacity
                .checked_mul(lane_count)
                .is_none_or(|bytes| bytes > buffer_byte_capacity)
        {
            return distributed_calibration_error(
                "selected-resource fixture does not fit its declared activation layout",
            );
        }
        let mut bytes = vec![0; buffer_byte_capacity];
        for lane in 0..lane_count {
            let lane_word_start = lane * self.selector.encoding.selection_count_per_activation;
            let lane_byte_start = lane * activation.signal_byte_capacity;
            for (ordinal, word) in words[lane_word_start
                ..lane_word_start + self.selector.encoding.selection_count_per_activation]
                .iter()
                .enumerate()
            {
                let offset = lane_byte_start + ordinal * size_of::<u32>();
                bytes[offset..offset + size_of::<u32>()]
                    .copy_from_slice(&word.to_le_bytes());
            }
        }
        Ok(bytes)
    }

    fn cleanup(self) -> Result<(), VulkanResidentTokenModelPackageError> {
        let Self {
            physical_device_id: _,
            logical_device_id,
            device,
            target: _,
            selector: _,
            resource_execution_class_id: _,
            resource_payload_byte_count: _,
            output_scalar_format: _,
            execution_plan: _,
            execution_case: _,
            runners,
            activation_buffers,
            resource_stores,
            dynamic_resource_buffers,
            transaction_predicates,
            parameter_buffers,
            parameter_pool,
            static_parameter_bytes: _,
            transient_peak_device_bytes: _,
            transient_host_bytes: _,
        } = self;
        let mut errors = Vec::new();
        if let Err(error) = device.quiesce() {
            errors.push(error.to_string());
        }
        drop(runners);
        drop(dynamic_resource_buffers);
        drop(transaction_predicates);
        let mut unloaded = BTreeSet::new();
        for store in resource_stores.values() {
            if unloaded.insert(Arc::as_ptr(store) as usize)
                && let Err(error) = store.unload()
            {
                errors.push(error.to_string());
            }
        }
        drop(resource_stores);
        drop(activation_buffers);
        drop(parameter_buffers);
        if let Err(error) = parameter_pool.release_device(&logical_device_id) {
            errors.push(error.to_string());
        }
        let stats = parameter_pool.stats();
        if parameter_pool.registered_device_count() != 0
            || stats.resident_allocation_count != 0
            || stats.resident_buffer_count != 0
            || stats.resident_bytes != 0
        {
            errors.push("selected-resource calibration retained parameter-pool state".to_string());
        }
        if let Err(error) = device.quiesce() {
            errors.push(error.to_string());
        }
        match device.device_local_memory_accounting() {
            Ok(accounting)
                if accounting.tracked_allocation_bytes == 0
                    && accounting.pending_reservation_bytes == 0 => {}
            Ok(accounting) => errors.push(format!(
                "selected-resource calibration retained {} tracked and {} pending device bytes",
                accounting.tracked_allocation_bytes, accounting.pending_reservation_bytes,
            )),
            Err(error) => errors.push(error.to_string()),
        }
        if errors.is_empty() {
            Ok(())
        } else {
            distributed_calibration_error(format!(
                "selected-resource execution calibration cleanup failed: {}",
                errors.join("; "),
            ))
        }
    }
}

pub fn calibrate_vulkan_runtime_selected_resource_execution(
    physical_device_id: &str,
    device: Rc<VulkanComputeDevice>,
    manifest_dir: impl AsRef<Path>,
    runtime_model: &VulkanResidentRuntimeModel,
    target: &VulkanRuntimeSelectedResourceExecutionCalibrationTarget,
    policy: VulkanRuntimePlacementCalibrationPolicy,
) -> Result<
    Option<VulkanRuntimeSelectedResourceExecutionCalibrationReport>,
    VulkanResidentTokenModelPackageError,
> {
    if physical_device_id.is_empty()
        || device.physical_device_id() != physical_device_id
        || target.component.signature_id.is_empty()
        || target.component.component_id.is_empty()
        || target.selector_id.is_empty()
        || !valid_sha256_digest(&target.resource_execution_class_id)
        || target.selected_contract_ids.is_empty()
        || target.phase.activation_batch_width() == 0
        || policy.warmup_units != 1
        || policy.measured_units != 1
        || policy.maximum_duration.is_zero()
        || policy.maximum_duration > VULKAN_RUNTIME_PLACEMENT_CALIBRATION_MAXIMUM_DURATION
    {
        return distributed_calibration_error(
            "selected-resource execution calibration requires one exact device, target, contract set, warmup, measurement, and at most one minute",
        );
    }
    let capacity = policy
        .parameter_capacity_for_physical_device(physical_device_id)?
        .min(policy.maximum_total_resident_parameter_bytes);
    let started = Instant::now();
    let Some(session) = VulkanRuntimeSelectedResourceExecutionSession::prepare(
        physical_device_id,
        device,
        manifest_dir.as_ref(),
        runtime_model,
        target,
        capacity,
    )?
    else {
        return Ok(None);
    };
    let execution_result = (|| {
        session.preload()?;
        let warmup = session.execute_once(remaining_calibration_duration(
            started,
            policy.maximum_duration,
        )?)?;
        let measured = session.execute_once(remaining_calibration_duration(
            started,
            policy.maximum_duration,
        )?)?;
        let output_equivalence = validate_vulkan_placement_output_equivalence(
            &session.execution_case.equivalence,
            &warmup.output_digest,
            warmup.output_artifact.as_ref(),
            &measured.output_digest,
            measured.output_artifact.as_ref(),
        )
        .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let resident_parameter_bytes = session.resident_parameter_bytes()?;
        Ok(VulkanRuntimeSelectedResourceExecutionCalibrationReport {
            physical_device_id: session.physical_device_id.clone(),
            target: session.target.clone(),
            resource_execution_class_id: session.resource_execution_class_id.clone(),
            resource_payload_byte_count: session.resource_payload_byte_count,
            selector_selection_count: session
                .selector
                .encoding
                .selection_count_per_activation,
            warmup_execution_ns: warmup.duration_ns,
            measured_execution_ns: measured.duration_ns,
            warmup_output_digest: warmup.output_digest,
            warmup_output_artifact: warmup.output_artifact,
            measured_output_digest: measured.output_digest,
            measured_output_artifact: measured.output_artifact,
            output_equivalence,
            state_digest: targeted_finalized_artifact_digest(Sha256::digest([]).as_slice()),
            resident_parameter_bytes,
            transient_peak_device_bytes: session.transient_peak_device_bytes,
            transient_host_bytes: session.transient_host_bytes,
            execution_case: session.execution_case.clone(),
        })
    })();
    let cleanup_result = session.cleanup();
    match (execution_result, cleanup_result) {
        (Ok(report), Ok(())) => Ok(Some(report)),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Err(error), Err(cleanup)) => Err(distributed_calibration_error_value(format!(
            "{error}; cleanup also failed: {cleanup}",
        ))),
    }
}

fn selected_resource_fixture_words(
    selector: &CompiledResourceSelector,
    selected_resource_index: usize,
    lane_count: usize,
    calibration_word_base: u32,
) -> Result<Vec<u32>, VulkanResidentTokenModelPackageError> {
    let selection_count = selector.encoding.selection_count_per_activation;
    if lane_count == 0
        || selection_count == 0
        || selected_resource_index >= selector.resource_count
        || selector.resource_count < selection_count
        || (lane_count > 1 && selector.resource_count <= selection_count)
    {
        return distributed_calibration_error(
            "selected-resource fixture cannot preserve selector width with one local occurrence",
        );
    }
    let encoded = |resource_index: usize| {
        calibration_word_base
            | ((u32::try_from(resource_index).unwrap_or(u32::MAX)
                & selector.encoding.index_mask)
                << selector.encoding.index_shift)
    };
    let non_local = (0..selector.resource_count)
        .filter(|index| *index != selected_resource_index)
        .collect::<Vec<_>>();
    let mut words = Vec::with_capacity(selection_count * lane_count);
    for lane in 0..lane_count {
        if lane == 0 {
            words.push(encoded(selected_resource_index));
            words.extend(non_local.iter().take(selection_count - 1).map(|index| encoded(*index)));
        } else {
            let start = (lane * selection_count) % non_local.len();
            for offset in 0..selection_count {
                words.push(encoded(non_local[(start + offset) % non_local.len()]));
            }
        }
    }
    if words.len() != selection_count * lane_count
        || words
            .iter()
            .filter(|word| {
                ((**word >> selector.encoding.index_shift) & selector.encoding.index_mask)
                    == u32::try_from(selected_resource_index).unwrap_or(u32::MAX)
            })
            .count()
            != 1
    {
        return distributed_calibration_error(
            "selected-resource fixture did not encode exactly one local occurrence",
        );
    }
    Ok(words)
}

fn selected_resource_fixture_digest(
    selector: &CompiledResourceSelector,
    selected_resource_index: usize,
    lane_count: usize,
) -> Result<String, VulkanResidentTokenModelPackageError> {
    let selection = selected_resource_fixture_words(
        selector,
        selected_resource_index,
        lane_count,
        selector.encoding.calibration_word_base,
    )?;
    let execution = selected_resource_fixture_words(
        selector,
        selected_resource_index,
        lane_count,
        selector.execution_calibration_word_base,
    )?;
    let payload = serde_json::to_vec(&(
        "nerve.selected_resource_execution_fixture.v1",
        selector.selection_signal == selector.execution_signal,
        selector.encoding.index_shift,
        selector.encoding.index_mask,
        lane_count,
        selection,
        execution,
    ))
    .map_err(|error| {
        distributed_calibration_error_value(format!(
            "could not encode selected-resource execution fixture: {error}",
        ))
    })?;
    Ok(format!("sha256:{:x}", Sha256::digest(payload)))
}

fn selected_resource_compiled_execution_signature(
    component_signature: &str,
    selector: &CompiledResourceSelector,
    resource_execution_class_id: &str,
    execution_plan: &VulkanDistributedExecutionPlan,
) -> Result<String, VulkanResidentTokenModelPackageError> {
    let contract_ids = execution_plan
        .execution_islands
        .iter()
        .flat_map(|island| island.contract_ids.iter().cloned())
        .collect::<BTreeSet<_>>();
    let mapping_kind = match &selector.mapping {
        CompiledResourceSelectorMapping::GroupTable { .. } => "group_table",
        CompiledResourceSelectorMapping::PartitionTemplate { .. } => "partition_template",
    };
    let payload = serde_json::to_vec(&serde_json::json!({
        "schema": "nerve.selected_resource_compiled_execution.v1",
        "component_signature": component_signature,
        "resource_execution_class_id": resource_execution_class_id,
        "contract_ids": contract_ids,
        "selector": {
            "resource_count": selector.resource_count,
            "selection_count_per_activation": selector.encoding.selection_count_per_activation,
            "selection_and_execution_share_signal": selector.selection_signal == selector.execution_signal,
            "index_shift": selector.encoding.index_shift,
            "index_mask": selector.encoding.index_mask,
            "mapping_kind": mapping_kind,
        },
    }))
    .map_err(|error| {
        distributed_calibration_error_value(format!(
            "could not encode selected-resource compiled execution: {error}",
        ))
    })?;
    Ok(format!("sha256:{:x}", Sha256::digest(payload)))
}

fn selected_resource_execution_graph_digest(
    component_signature: &str,
    selector: &CompiledResourceSelector,
    resource_execution_class_id: &str,
    execution_plan: &VulkanDistributedExecutionPlan,
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"nerve.selected_resource_execution_graph.v1\0");
    digest.update(component_signature.as_bytes());
    digest.update([0]);
    digest.update([u8::from(
        selector.selection_signal == selector.execution_signal,
    )]);
    digest.update(selector.resource_count.to_le_bytes());
    digest.update(
        selector
            .encoding
            .selection_count_per_activation
            .to_le_bytes(),
    );
    digest.update(resource_execution_class_id.as_bytes());
    digest.update(execution_plan.dispatches.len().to_le_bytes());
    let dispatch_ordinal_by_index = execution_plan
        .dispatches
        .iter()
        .enumerate()
        .map(|(ordinal, dispatch)| (dispatch.dispatch_index, ordinal))
        .collect::<BTreeMap<_, _>>();
    for dispatch in &execution_plan.dispatches {
        digest.update(dispatch.physical_execution_contract_id.as_bytes());
        digest.update([0]);
        digest.update(dispatch.input_width.to_le_bytes());
        digest.update(dispatch.output_rows.to_le_bytes());
        digest.update(dispatch.input_byte_capacity.to_le_bytes());
        digest.update(dispatch.output_byte_capacity.to_le_bytes());
        digest.update(match dispatch.distribution {
            VulkanDistributedDispatchDistribution::OutputRows => [0],
            VulkanDistributedDispatchDistribution::InputColumns => [1],
            VulkanDistributedDispatchDistribution::ExpertRange => [2],
        });
        digest.update(dispatch.local_intermediates.len().to_le_bytes());
        digest.update([u8::from(dispatch.reduction.is_some())]);
    }
    for island in &execution_plan.execution_islands {
        digest.update(island.dispatch_indices().len().to_le_bytes());
        for dispatch_index in island.dispatch_indices() {
            digest.update(
                dispatch_ordinal_by_index
                    .get(&dispatch_index)
                    .copied()
                    .unwrap_or(usize::MAX)
                    .to_le_bytes(),
            );
        }
    }
    format!("sha256:{:x}", digest.finalize())
}

fn validate_selected_resource_finite_output(
    bytes: &[u8],
    scalar_format: VulkanPlacementScalarFormat,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    let finite = match scalar_format {
        VulkanPlacementScalarFormat::Bf16 => {
            !bytes.is_empty()
                && bytes.len().is_multiple_of(size_of::<u16>())
                && bytes.chunks_exact(size_of::<u16>()).all(|bytes| {
                    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
                    bits & 0x7f80 != 0x7f80
                })
        }
        VulkanPlacementScalarFormat::F32 => {
            !bytes.is_empty()
                && bytes.len().is_multiple_of(size_of::<f32>())
                && bytes.chunks_exact(size_of::<f32>()).all(|bytes| {
                    f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]).is_finite()
                })
        }
    };
    if finite {
        Ok(())
    } else {
        distributed_calibration_error(
            "selected-resource execution produced empty, unaligned, or non-finite output",
        )
    }
}

#[cfg(test)]
mod runtime_selected_resource_execution_calibration_tests {
    use super::*;

    fn digest(byte: char) -> String {
        format!("sha256:{}", byte.to_string().repeat(64))
    }

    fn selector() -> CompiledResourceSelector {
        CompiledResourceSelector {
            id: "experts".to_string(),
            execution_scope: "model".to_string(),
            component_id: "block".to_string(),
            node_id: "router".to_string(),
            domain_id: "experts".to_string(),
            resource_count: 8,
            selection_signal: "preselected".to_string(),
            execution_signal: "routes".to_string(),
            execution_calibration_word_base: 0x3f80_0000,
            encoding: CompiledResourceSelectionEncoding {
                element_type: CompiledResourceSelectionElementType::U32,
                selection_count_per_activation: 3,
                index_shift: 0,
                index_mask: 0xff,
                calibration_word_base: 0,
            },
            mapping: CompiledResourceSelectorMapping::GroupTable {
                atomic_group_ids: (0..8).map(|index| format!("expert-{index}")).collect(),
            },
        }
    }

    fn report() -> VulkanRuntimeSelectedResourceExecutionCalibrationReport {
        let physical_device_id = "gpu0".to_string();
        let resource_execution_class_id = digest('c');
        let behavior = VulkanPlacementBehaviorIdentity {
            compiled_execution_signature: digest('a'),
            runtime_implementation_fingerprint: "runtime".to_string(),
            phase: nerve_execution_contracts::ExecutionPhase::Decode,
            shape: VulkanPlacementShapeClass {
                activation_batch_width: 1,
                input_byte_capacity: 128,
                output_byte_capacity: 128,
            },
            input_fixture_digest: digest('b'),
        };
        VulkanRuntimeSelectedResourceExecutionCalibrationReport {
            physical_device_id: physical_device_id.clone(),
            target: VulkanRuntimeSelectedResourceExecutionCalibrationTarget {
                component: VulkanRuntimePlacementCalibrationTarget {
                    signature_id: digest('1'),
                    component_id: "block".to_string(),
                    component_ids: vec!["block".to_string()],
                    terminal_node_id: "down".to_string(),
                    implementation: "sparse_ffn".to_string(),
                    planned_resident_parameter_bytes: 4096,
                },
                selector_id: "experts".to_string(),
                resource_index: 4,
                resource_execution_class_id: resource_execution_class_id.clone(),
                phase: VulkanTargetedComponentExecutionPhase::Decode,
                selected_contract_ids: BTreeSet::from(["contract".to_string()]),
            },
            resource_execution_class_id: resource_execution_class_id.clone(),
            resource_payload_byte_count: 4096,
            selector_selection_count: 6,
            warmup_execution_ns: 100,
            measured_execution_ns: 90,
            warmup_output_digest: digest('d'),
            warmup_output_artifact: None,
            measured_output_digest: digest('d'),
            measured_output_artifact: None,
            output_equivalence: VulkanPlacementOutputEquivalenceEvidence::BitExact,
            state_digest: digest('e'),
            resident_parameter_bytes: 4096,
            transient_peak_device_bytes: 1024,
            transient_host_bytes: 256,
            execution_case: VulkanPlacementExecutionCaseIdentity {
                behavior,
                contract_ids: vec!["contract".to_string()],
                implementation_digests: vec![digest('f')],
                artifact_digest: digest('6'),
                execution_graph_digest: digest('7'),
                operations: vec![
                    VulkanPlacementOperationGeometry::SelectedResourceTransaction {
                        contract_id: "contract".to_string(),
                        resource_execution_class_id,
                        selector_selection_count: 6,
                        executed_resource_occurrence_count: 1,
                    },
                    VulkanPlacementOperationGeometry::Dispatch {
                        geometry: VulkanPlacementDispatchGeometry {
                            contract_id: "contract".to_string(),
                            logical_extent: 64,
                            sampled_extent: 64,
                            input_width: 128,
                            workgroup_count_x: 1,
                            local_size_x: 64,
                        },
                    },
                ],
                equivalence: VulkanPlacementEquivalenceIdentity::bit_exact(),
                strategy: VulkanPlacementExecutionStrategy::SelectedResourceTransaction,
                devices: vec![VulkanPlacementDeviceExecutionIdentity {
                    physical_device_id: physical_device_id.clone(),
                    api_version: 14,
                    driver_version: 27,
                }],
                shards: Vec::new(),
                input_physical_device_id: physical_device_id.clone(),
                output_physical_device_id: physical_device_id.clone(),
                owner_physical_device_id: physical_device_id,
                transports: Vec::new(),
            },
        }
    }

    fn load_wave_report() -> VulkanRuntimeLoadWaveCalibrationReport {
        let physical_device_id = "gpu0".to_string();
        let payload_bytes = 4096;
        VulkanRuntimeLoadWaveCalibrationReport {
            physical_device_id: physical_device_id.clone(),
            api_version: 14,
            driver_version: 27,
            component_id: "block".to_string(),
            selector_id: "experts".to_string(),
            resource_indices: vec![4],
            group_ids: vec!["expert-4".to_string()],
            phase: nerve_execution_contracts::ExecutionPhase::Decode,
            activation_batch_width: 1,
            loaded_group_count: 1,
            loaded_resource_count: 1,
            loaded_byte_count: payload_bytes,
            warmup_ns: 200,
            measured_ns: 180,
            output_digest: digest('8'),
            state_digest: digest('9'),
            resident_device_bytes: payload_bytes,
            transient_peak_device_bytes: payload_bytes + 256,
            transient_host_bytes: 256,
            execution_case: VulkanPlacementExecutionCaseIdentity {
                behavior: VulkanPlacementBehaviorIdentity {
                    compiled_execution_signature: digest('2'),
                    runtime_implementation_fingerprint: "runtime".to_string(),
                    phase: nerve_execution_contracts::ExecutionPhase::Decode,
                    shape: VulkanPlacementShapeClass {
                        activation_batch_width: 1,
                        input_byte_capacity: payload_bytes,
                        output_byte_capacity: payload_bytes,
                    },
                    input_fixture_digest: digest('3'),
                },
                contract_ids: vec!["experts".to_string()],
                implementation_digests: vec![digest('4')],
                artifact_digest: digest('5'),
                execution_graph_digest: digest('0'),
                operations: vec![VulkanPlacementOperationGeometry::LazyLoadWave {
                    contract_id: "experts".to_string(),
                    resource_count: 1,
                    byte_count: payload_bytes,
                }],
                equivalence: VulkanPlacementEquivalenceIdentity::bit_exact(),
                strategy: VulkanPlacementExecutionStrategy::LazyLoadWave,
                devices: vec![VulkanPlacementDeviceExecutionIdentity {
                    physical_device_id: physical_device_id.clone(),
                    api_version: 14,
                    driver_version: 27,
                }],
                shards: Vec::new(),
                input_physical_device_id: physical_device_id.clone(),
                output_physical_device_id: physical_device_id.clone(),
                owner_physical_device_id: physical_device_id,
                transports: Vec::new(),
            },
        }
    }

    #[test]
    fn fixture_preserves_real_selector_width_and_one_local_occurrence() {
        let selector = selector();
        let words = selected_resource_fixture_words(&selector, 6, 4, 0x3f80_0000).unwrap();
        assert_eq!(words.len(), 12);
        assert_eq!(words[0] & selector.encoding.index_mask, 6);
        assert_eq!(
            words
                .iter()
                .filter(|word| **word & selector.encoding.index_mask == 6)
                .count(),
            1,
        );
        for lane in words.chunks_exact(3) {
            assert_eq!(lane.iter().collect::<BTreeSet<_>>().len(), 3);
            assert!(lane.iter().all(|word| *word & 0x3f80_0000 == 0x3f80_0000));
        }
    }

    #[test]
    fn fixture_rejects_a_shape_that_cannot_isolate_one_occurrence() {
        let mut selector = selector();
        selector.resource_count = 3;
        selector.mapping = CompiledResourceSelectorMapping::GroupTable {
            atomic_group_ids: (0..3).map(|index| format!("expert-{index}")).collect(),
        };
        assert!(
            selected_resource_fixture_words(&selector, 1, 2, 0)
                .unwrap_err()
                .to_string()
                .contains("cannot preserve selector width")
        );
    }

    #[test]
    fn output_validation_rejects_non_finite_bf16_and_f32() {
        validate_selected_resource_finite_output(
            &0x3f80u16.to_le_bytes(),
            VulkanPlacementScalarFormat::Bf16,
        )
        .unwrap();
        assert!(
            validate_selected_resource_finite_output(
                &0x7f80u16.to_le_bytes(),
                VulkanPlacementScalarFormat::Bf16,
            )
            .is_err()
        );
        assert!(
            validate_selected_resource_finite_output(
                &f32::NAN.to_le_bytes(),
                VulkanPlacementScalarFormat::F32,
            )
            .is_err()
        );
    }

    #[test]
    fn exact_execution_report_records_transactionally() {
        let report = report();
        let mut catalog = VulkanPlacementCalibrationCatalog::default();
        record_vulkan_runtime_selected_resource_execution_calibration_report(
            &mut catalog,
            &report,
        )
        .unwrap();
        assert!(catalog
            .canonical_reference(&report.execution_case.behavior)
            .is_some());
        assert!(catalog
            .exact_observation(&report.execution_case)
            .is_some());

        let snapshot = catalog.clone();
        let mut invalid = report;
        invalid.measured_output_digest = digest('9');
        assert!(
            record_vulkan_runtime_selected_resource_execution_calibration_report(
                &mut catalog,
                &invalid,
            )
            .is_err()
        );
        assert_eq!(catalog, snapshot);
    }

    #[test]
    fn fragmented_resource_identity_does_not_require_whole_resource_ownership() {
        assert!(
            distributed_calibration_normalized_selected_resource_indices(
                std::iter::empty::<&str>(),
                &BTreeMap::new(),
                "gpu0",
            )
            .unwrap()
            .is_empty()
        );
        let fragment = VulkanDistributedSelectedResourceFragmentPlan {
            resource_index: 3,
            atomic_group_id: "expert-3".to_string(),
            logical_start: 0,
            logical_count: 64,
            parameters: vec![
                crate::vulkan_distributed::VulkanDistributedSelectedResourceParameterFragmentPlan {
                parameter_slot: 0,
                resource_id: "expert-3-gate".to_string(),
                resource_byte_count: 128,
                byte_offset: 0,
                byte_count: 128,
            }],
        };
        let normalized = distributed_calibration_normalized_selected_resource_fragments(
            ["runtime-selector"].into_iter(),
            &BTreeMap::from([("runtime-selector".to_string(), vec![fragment])]),
            "gpu0",
        )
        .unwrap();
        assert_eq!(normalized[&0][0].resource_index, 3);
        assert_eq!(normalized[&0][0].logical_count, 64);
    }

    #[test]
    fn execution_and_load_reports_publish_one_exact_planner_class() {
        let execution = report();
        let load_wave = load_wave_report();
        let mut catalog = VulkanPlacementCalibrationCatalog::default();
        record_vulkan_runtime_selected_resource_execution_calibration_report(
            &mut catalog,
            &execution,
        )
        .unwrap();
        record_vulkan_runtime_load_wave_calibration_report(&mut catalog, &load_wave).unwrap();
        catalog
            .record_selected_resource_execution_class(
                execution.execution_class_calibration(&load_wave).unwrap(),
            )
            .unwrap();

        let requirement = execution.requirement();
        let devices = catalog
            .selected_resource_placement_devices(
                &[requirement],
                &[VulkanPlacementSelectedResourceDeviceCapacity {
                    device_id: "logical0".to_string(),
                    identity: VulkanPlacementDeviceExecutionIdentity {
                        physical_device_id: "gpu0".to_string(),
                        api_version: 14,
                        driver_version: 27,
                    },
                    resident_payload_capacity_bytes: 8192,
                }],
            )
            .unwrap();
        assert_eq!(devices.len(), 1);
        assert_eq!(
            devices[0].measured_costs_by_execution_class[&execution.resource_execution_class_id]
                .execution_duration_ns,
            90,
        );
        assert_eq!(
            devices[0].measured_costs_by_execution_class[&execution.resource_execution_class_id]
                .lazy_load_wave_duration_ns,
            180,
        );
    }
}
