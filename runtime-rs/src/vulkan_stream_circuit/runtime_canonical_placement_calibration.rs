#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanRuntimeCanonicalPlacementCalibration {
    pub reference: VulkanPlacementCanonicalReference,
    pub observation: VulkanPlacementCalibrationObservation,
}

pub fn record_vulkan_runtime_canonical_placement_calibration(
    catalog: &mut VulkanPlacementCalibrationCatalog,
    calibration: VulkanRuntimeCanonicalPlacementCalibration,
) -> Result<(), VulkanPlacementCalibrationCatalogError> {
    let mut updated = catalog.clone();
    if let Some(existing) = updated.canonical_reference(&calibration.reference.behavior) {
        if existing != &calibration.reference {
            return Err(VulkanPlacementCalibrationCatalogError(
                "canonical placement devices produced different reference evidence".to_string(),
            ));
        }
    } else {
        updated.record_reference(calibration.reference)?;
    }
    updated.record_observation(calibration.observation)?;
    *catalog = updated;
    Ok(())
}

pub fn calibrate_vulkan_runtime_canonical_placement_candidate_with_policy(
    physical_device_id: &str,
    device: Rc<VulkanComputeDevice>,
    manifest_dir: impl AsRef<Path>,
    runtime_model: &VulkanResidentRuntimeModel,
    target: &VulkanRuntimePlacementCalibrationTarget,
    phase: VulkanTargetedComponentExecutionPhase,
    policy: VulkanRuntimePlacementCalibrationPolicy,
) -> Result<Option<VulkanRuntimeCanonicalPlacementCalibration>, VulkanResidentTokenModelPackageError>
{
    if physical_device_id.is_empty() || device.physical_device_id() != physical_device_id {
        return canonical_calibration_error(
            "canonical placement calibration requires the exact physical device identity",
        );
    }
    let behavior = canonical_component_boundary_behavior(runtime_model, target, phase)?;
    calibrate_vulkan_runtime_canonical_component(
        physical_device_id,
        device,
        manifest_dir.as_ref(),
        runtime_model,
        target,
        phase,
        &behavior,
        policy,
    )
}

impl VulkanRuntimePlacementCalibrationSuite {
    /// Converts a scalar calibration report already measured by this suite
    /// into the exact canonical evidence consumed by hybrid placement. This
    /// reuses the mounted executable plan and measured output instead of
    /// repeating the same GPU work in a second calibration pass.
    pub fn canonical_calibration_from_report(
        &mut self,
        device: &VulkanComputeDevice,
        capability_class: &str,
        manifest_dir: &Path,
        runtime_model: &VulkanResidentRuntimeModel,
        report: &VulkanRuntimePlacementCalibrationReport,
        phase: VulkanTargetedComponentExecutionPhase,
        warmup_call_count: usize,
        measured_call_count: usize,
    ) -> Result<VulkanRuntimeCanonicalPlacementCalibration, VulkanResidentTokenModelPackageError>
    {
        let target_index = self
            .targets()
            .iter()
            .position(|target| {
                target.signature_id == report.target.signature_id
                    && target.component_id == report.target.component_id
            })
            .ok_or_else(|| {
                canonical_calibration_error_value(
                    "scalar calibration report does not belong to its calibration suite",
                )
            })?;
        if report.physical_device_id != device.physical_device_id()
            || report.activation_batch_width != phase.activation_batch_width()
        {
            return canonical_calibration_error(
                "scalar calibration report changed its physical device or phase identity",
            );
        }
        let target = self.targets()[target_index].clone();
        let plans = self
            .plans_for_device(
                device,
                capability_class,
                manifest_dir,
                &[target_index],
            )?
            .0;
        let [cached_plan] = plans.as_slice() else {
            return canonical_calibration_error(
                "scalar calibration suite did not retain exactly one executable plan",
            );
        };
        let behavior = canonical_component_boundary_behavior(runtime_model, &target, phase)?;
        canonical_calibration_from_report_and_plan(
            device,
            &target,
            phase,
            &behavior,
            report,
            cached_plan,
            warmup_call_count,
            measured_call_count,
        )
    }
}

fn canonical_component_boundary_behavior(
    runtime_model: &VulkanResidentRuntimeModel,
    target: &VulkanRuntimePlacementCalibrationTarget,
    phase: VulkanTargetedComponentExecutionPhase,
) -> Result<VulkanPlacementBehaviorIdentity, VulkanResidentTokenModelPackageError> {
    let component = runtime_model
        .circuit_graph
        .components
        .iter()
        .find(|component| {
            component.component_id == target.component_id
                && component.runtime_role.is_signal_processor()
        })
        .ok_or_else(|| {
            canonical_calibration_error_value(format!(
                "canonical placement calibration found no signal processor {:?}",
                target.component_id,
            ))
        })?;
    let fallback_element_bytes = runtime_model.package.activation_element_bytes.ok_or_else(|| {
        canonical_calibration_error_value(
            "canonical placement calibration requires a compiled activation element width",
        )
    })?;
    let input_byte_capacity = canonical_boundary_ports_byte_capacity(
        &component.circuit.boundary.inputs,
        fallback_element_bytes,
    )?;
    let output_byte_capacity = canonical_boundary_ports_byte_capacity(
        &component.circuit.boundary.outputs,
        fallback_element_bytes,
    )?;
    let execution_phase = match phase {
        VulkanTargetedComponentExecutionPhase::Decode => {
            nerve_execution_contracts::ExecutionPhase::Decode
        }
        VulkanTargetedComponentExecutionPhase::Prefill { .. } => {
            nerve_execution_contracts::ExecutionPhase::Prefill
        }
    };
    let shape = VulkanPlacementShapeClass {
        activation_batch_width: phase.activation_batch_width(),
        input_byte_capacity,
        output_byte_capacity,
    };
    Ok(VulkanPlacementBehaviorIdentity {
        compiled_execution_signature: target.signature_id.clone(),
        runtime_implementation_fingerprint: crate::RUNTIME_IMPLEMENTATION_FINGERPRINT.to_string(),
        phase: execution_phase,
        input_fixture_digest: distributed_calibration_fixture_identity(execution_phase, &shape, 0)?,
        shape,
    })
}

fn canonical_boundary_ports_byte_capacity(
    ports: &[CircuitPort],
    fallback_element_bytes: usize,
) -> Result<usize, VulkanResidentTokenModelPackageError> {
    if ports.is_empty() || fallback_element_bytes == 0 {
        return canonical_calibration_error(
            "canonical placement calibration requires nonempty typed component boundaries",
        );
    }
    let byte_capacity = ports.iter().try_fold(0usize, |total, port| {
        let element_count = port.shape.iter().try_fold(1usize, |elements, dimension| {
            elements.checked_mul(*dimension).ok_or_else(|| {
                canonical_calibration_error_value(
                    "canonical component boundary shape overflowed",
                )
            })
        })?;
        let element_bytes = port
            .dtype
            .as_deref()
            .map(crate::stream_plan::circuit_dtype_bytes)
            .transpose()
            .map_err(|error| canonical_calibration_error_value(error.to_string()))?
            .unwrap_or(fallback_element_bytes);
        let port_bytes = element_count.checked_mul(element_bytes).ok_or_else(|| {
            canonical_calibration_error_value(
                "canonical component boundary byte capacity overflowed",
            )
        })?;
        total.checked_add(port_bytes).ok_or_else(|| {
            canonical_calibration_error_value(
                "canonical component boundary aggregate capacity overflowed",
            )
        })
    })?;
    if byte_capacity == 0 {
        return canonical_calibration_error(
            "canonical component boundary byte capacity must be positive",
        );
    }
    Ok(byte_capacity)
}

#[allow(clippy::too_many_arguments)]
fn calibrate_vulkan_runtime_canonical_component(
    physical_device_id: &str,
    device: Rc<VulkanComputeDevice>,
    manifest_dir: &Path,
    runtime_model: &VulkanResidentRuntimeModel,
    target: &VulkanRuntimePlacementCalibrationTarget,
    phase: VulkanTargetedComponentExecutionPhase,
    behavior: &VulkanPlacementBehaviorIdentity,
    policy: VulkanRuntimePlacementCalibrationPolicy,
) -> Result<Option<VulkanRuntimeCanonicalPlacementCalibration>, VulkanResidentTokenModelPackageError>
{
    let warmup_call_count = policy.warmup_units;
    let measured_call_count = policy.measured_units;
    let capacity = runtime_model
        .package
        .max_context_activations
        .max(1)
        .min(VULKAN_RUNTIME_PLACEMENT_CALIBRATION_MAXIMUM_STATE_ACTIVATIONS);
    let mut suite = VulkanRuntimePlacementCalibrationSuite::prepare_target(
        manifest_dir,
        runtime_model,
        capacity,
        target.clone(),
    )?;
    let scope = match phase {
        VulkanTargetedComponentExecutionPhase::Decode => {
            VulkanTargetedComponentExecutionScope::DecodeComponentPrefix
        }
        VulkanTargetedComponentExecutionPhase::Prefill { .. } => {
            VulkanTargetedComponentExecutionScope::Component
        }
    };
    let reports = calibrate_vulkan_runtime_placement_phase_candidate_with_policy(
        Rc::clone(&device),
        manifest_dir,
        physical_device_id,
        &mut suite,
        phase,
        scope,
        policy,
        None,
    )?;
    let report = match reports.as_slice() {
        [] => return Ok(None),
        [report] => report,
        _ => {
        return canonical_calibration_error(format!(
            "canonical component calibration produced {} reports for {:?}",
            reports.len(),
            target.component_id,
        ));
        }
    };
    if report.target.signature_id != target.signature_id
        || report.target.component_id != target.component_id
        || report.activation_batch_width != phase.activation_batch_width()
    {
        return canonical_calibration_error(
            "canonical component calibration changed its requested transaction identity",
        );
    }
    let plans = suite
        .plans_for_device(&device, physical_device_id, manifest_dir, &[0])?
        .0;
    let [cached_plan] = plans.as_slice() else {
        return canonical_calibration_error(
            "canonical component calibration did not retain exactly one executable plan",
        );
    };
    canonical_calibration_from_report_and_plan(
        &device,
        target,
        phase,
        behavior,
        report,
        cached_plan,
        warmup_call_count,
        measured_call_count,
    )
    .map(Some)
}

#[allow(clippy::too_many_arguments)]
fn canonical_calibration_from_report_and_plan(
    device: &VulkanComputeDevice,
    target: &VulkanRuntimePlacementCalibrationTarget,
    phase: VulkanTargetedComponentExecutionPhase,
    behavior: &VulkanPlacementBehaviorIdentity,
    report: &VulkanRuntimePlacementCalibrationReport,
    cached_plan: &VulkanRuntimePlacementCalibrationCachedPlan,
    warmup_call_count: usize,
    measured_call_count: usize,
) -> Result<VulkanRuntimeCanonicalPlacementCalibration, VulkanResidentTokenModelPackageError> {
    let contracts = canonical_component_execution_contracts(cached_plan, target, phase)?;
    let output_artifact = canonical_calibration_output_artifact(report)?;
    let physical_device_id = device.physical_device_id();
    let execution_case = canonical_component_execution_case(
        VulkanPlacementDeviceExecutionIdentity {
            physical_device_id: physical_device_id.to_string(),
            api_version: device.api_version(),
            driver_version: device.driver_version(),
        },
        target,
        behavior,
        &contracts,
    )?;
    let reference = VulkanPlacementCanonicalReference {
        behavior: behavior.clone(),
        output_digest: report.output_digest.clone(),
        output_artifact: Some(output_artifact),
        state_digest: report.state_digest.clone(),
    };
    let observation = VulkanPlacementCalibrationObservation {
        execution_case,
        warmup_call_count,
        measured_call_count,
        complete_transaction: true,
        duration_ns: report.measured_execution_ns,
        useful_activation_count: measured_call_count
            .checked_mul(phase.activation_batch_width())
            .ok_or_else(|| {
                canonical_calibration_error_value(
                    "canonical component useful activation count overflowed",
                )
            })?,
        output_digest: report.output_digest.clone(),
        output_artifact: None,
        output_equivalence: VulkanPlacementOutputEquivalenceEvidence::BitExact,
        state_digest: report.state_digest.clone(),
        resident_bytes_by_physical_device: BTreeMap::from([(
            physical_device_id.to_string(),
            report.resident_parameter_bytes,
        )]),
        transient_peak_bytes_by_physical_device: BTreeMap::from([(
            physical_device_id.to_string(),
            report.resident_transient_bytes,
        )]),
        host_resident_bytes: 0,
        host_transient_peak_bytes: 0,
    };
    Ok(VulkanRuntimeCanonicalPlacementCalibration {
        reference,
        observation,
    })
}

fn canonical_loaded_reusable_artifact_path<'a>(
    loaded_manifest: &'a VulkanLoadedKernelArtifactCatalog,
    family_id: &str,
    component_id: &str,
    node_id: &str,
) -> Result<&'a str, VulkanResidentTokenModelPackageError> {
    loaded_manifest
        .reusable_artifact(family_id)
        .map(|loaded| loaded.artifact.path.as_str())
        .ok_or_else(|| {
            canonical_calibration_error_value(format!(
                "canonical component has no loaded reusable artifact for {component_id}.{node_id} family {family_id:?}",
            ))
        })
}

fn canonical_component_execution_contracts(
    cached_plan: &VulkanRuntimePlacementCalibrationCachedPlan,
    target: &VulkanRuntimePlacementCalibrationTarget,
    phase: VulkanTargetedComponentExecutionPhase,
) -> Result<Vec<nerve_execution_contracts::PhysicalExecutionContract>, VulkanResidentTokenModelPackageError>
{
    let slice_plan = &cached_plan.plan.slice_plan;
    let target_dispatch = slice_plan
        .prepared_plan
        .dispatch(&target.component_id, &target.terminal_node_id)
        .ok_or_else(|| {
            canonical_calibration_error_value(format!(
                "canonical component plan has no terminal dispatch {}.{}",
                target.component_id, target.terminal_node_id,
            ))
        })?;
    let terminal_dispatch_index = match phase {
        VulkanTargetedComponentExecutionPhase::Decode => slice_plan
            .physical_residency_schedule
            .checkpoints
            .iter()
            .find_map(|checkpoint| {
                if checkpoint.component_id != target.component_id {
                    return None;
                }
                let target_is_checkpoint_work =
                    target_dispatch.dispatch_index == checkpoint.selection_dispatch_index
                        || checkpoint
                            .selected_computation_dispatch_indices
                            .contains(&target_dispatch.dispatch_index)
                        || checkpoint.selected_result_continuation_dispatch_index
                            == Some(target_dispatch.dispatch_index);
                target_is_checkpoint_work.then(|| {
                    checkpoint
                        .selected_result_continuation_dispatch_index
                        .or_else(|| {
                            checkpoint
                                .selected_computation_dispatch_indices
                                .last()
                                .copied()
                        })
                        .unwrap_or(checkpoint.selection_dispatch_index)
                })
            })
            .unwrap_or(target_dispatch.dispatch_index),
        VulkanTargetedComponentExecutionPhase::Prefill { .. } => target_dispatch.dispatch_index,
    };
    let dispatches = slice_plan
        .prepared_plan
        .dispatches
        .iter()
        .filter(|dispatch| {
            dispatch.component_id == target.component_id
                && dispatch.dispatch_index <= terminal_dispatch_index
        })
        .collect::<Vec<_>>();
    if dispatches
        .last()
        .is_none_or(|dispatch| dispatch.dispatch_index != terminal_dispatch_index)
    {
        return canonical_calibration_error(
            "canonical component contract chain does not reach its terminal dispatch",
        );
    }
    dispatches
        .into_iter()
        .map(|dispatch| match phase {
            VulkanTargetedComponentExecutionPhase::Decode => {
                let artifact_path = canonical_loaded_reusable_artifact_path(
                    &slice_plan.loaded_manifest,
                    &dispatch.reusable_family_id,
                    &dispatch.component_id,
                    &dispatch.node_id,
                )?;
                canonical_single_device_contract_for_artifacts(
                    &dispatch.physical_execution_contracts,
                    &dispatch.op,
                    &dispatch.node_id,
                    nerve_execution_contracts::ExecutionPhase::Decode,
                    nerve_execution_contracts::ExecutionShape::SingleLane,
                    &[artifact_path],
                )
                .map(|contract| vec![contract])
            }
            VulkanTargetedComponentExecutionPhase::Prefill {
                activation_batch_width,
            } => {
                let selected = select_component_batch_kernel_artifact(
                    &slice_plan.batch_kernels,
                    &dispatch.component_id,
                    &dispatch.node_id,
                    VulkanComponentBatchExecutionMode::CausalSequence,
                    activation_batch_width,
                );
                if selected.is_some_and(|artifact| {
                    artifact.batch_mode == VulkanResidentComponentKernelBatchMode::CausalScan
                        && activation_batch_width > artifact.lane_tile_width
                }) {
                    return canonical_calibration_error(format!(
                        "canonical prefill dispatch {}.{} exceeds its causal scan tile width",
                        dispatch.component_id, dispatch.node_id,
                    ));
                }
                let executable_batch = selected
                    .filter(|artifact| {
                        component_batch_stages_replace_push_constants(
                            &artifact.stages,
                            &dispatch.push_constants,
                        )
                    })
                    .filter(|artifact| {
                        targeted_prefill_batch_mode_is_supported(artifact.batch_mode)
                    });
                if let Some(artifact) = executable_batch {
                    let artifact_paths = artifact
                        .stages
                        .iter()
                        .map(|stage| stage.shader_path.as_str())
                        .collect::<Vec<_>>();
                    return canonical_single_device_contract_for_artifacts(
                        &dispatch.physical_execution_contracts,
                        &dispatch.op,
                        &dispatch.node_id,
                        nerve_execution_contracts::ExecutionPhase::Prefill,
                        nerve_execution_contracts::ExecutionShape::MultiLane,
                        &artifact_paths,
                    )
                    .map(|contract| vec![contract]);
                }

                // This is the exact component-batch fallback: execute the
                // scalar primary dispatch once for every causal lane.
                let artifact_path = canonical_loaded_reusable_artifact_path(
                    &slice_plan.loaded_manifest,
                    &dispatch.reusable_family_id,
                    &dispatch.component_id,
                    &dispatch.node_id,
                )?;
                canonical_scalar_lane_prefill_contracts(
                    &dispatch.physical_execution_contracts,
                    &dispatch.op,
                    &dispatch.node_id,
                    artifact_path,
                    activation_batch_width,
                )
            }
        })
        .collect::<Result<Vec<_>, _>>()
        .map(|contracts| contracts.into_iter().flatten().collect())
}

fn canonical_scalar_lane_prefill_contracts(
    contracts: &[nerve_execution_contracts::PhysicalExecutionContract],
    operation_family: &str,
    node_id: &str,
    artifact_path: &str,
    activation_batch_width: usize,
) -> Result<Vec<nerve_execution_contracts::PhysicalExecutionContract>, VulkanResidentTokenModelPackageError>
{
    if activation_batch_width == 0 {
        return canonical_calibration_error(
            "canonical scalar-lane prefill requires a positive activation batch width",
        );
    }
    canonical_single_device_contract_for_artifacts(
        contracts,
        operation_family,
        node_id,
        nerve_execution_contracts::ExecutionPhase::Decode,
        nerve_execution_contracts::ExecutionShape::SingleLane,
        &[artifact_path],
    )
    .map(|contract| vec![contract; activation_batch_width])
}

fn canonical_single_device_contract_for_artifacts(
    contracts: &[nerve_execution_contracts::PhysicalExecutionContract],
    operation_family: &str,
    node_id: &str,
    phase: nerve_execution_contracts::ExecutionPhase,
    execution_shape: nerve_execution_contracts::ExecutionShape,
    artifact_paths: &[&str],
) -> Result<nerve_execution_contracts::PhysicalExecutionContract, VulkanResidentTokenModelPackageError>
{
    let matches = contracts
        .iter()
        .filter(|contract| {
            contract.strategy == nerve_execution_contracts::ExecutionStrategy::SingleDevice
                && contract.operation_family == operation_family
                && contract.member_node_ids.iter().any(|member| member == node_id)
                && contract.phases.contains(&phase)
                && contract.execution_shape.supports(execution_shape)
                && contract
                    .artifacts
                    .iter()
                    .map(|artifact| artifact.path.as_str())
                    .eq(artifact_paths.iter().copied())
        })
        .collect::<Vec<_>>();
    let [contract] = matches.as_slice() else {
        return canonical_calibration_error(format!(
            "canonical dispatch {node_id:?} resolved {} matching single-device contracts for artifacts {artifact_paths:?}",
            matches.len(),
        ));
    };
    contract
        .validate()
        .map_err(|error| canonical_calibration_error_value(error.to_string()))?;
    Ok((*contract).clone())
}

fn canonical_component_execution_case(
    device: VulkanPlacementDeviceExecutionIdentity,
    target: &VulkanRuntimePlacementCalibrationTarget,
    behavior: &VulkanPlacementBehaviorIdentity,
    contracts: &[nerve_execution_contracts::PhysicalExecutionContract],
) -> Result<VulkanPlacementExecutionCaseIdentity, VulkanResidentTokenModelPackageError> {
    if contracts.is_empty() {
        return canonical_calibration_error(
            "canonical component execution case has no compiler-declared contracts",
        );
    }
    let mut contract_digests = BTreeMap::new();
    let mut operations = Vec::with_capacity(contracts.len());
    for contract in contracts {
        if let Some(existing) = contract_digests.insert(
            contract.contract_id.clone(),
            contract.implementation_digest.clone(),
        ) && existing != contract.implementation_digest
        {
            return canonical_calibration_error(
                "canonical component contract has conflicting implementation digests",
            );
        }
        let local_size_x = canonical_contract_dimension_u32(contract, "local_size_x")?;
        let workgroup_count_x =
            canonical_contract_dimension_u32(contract, "workgroup_count_x")?;
        let logical_extent = usize::try_from(local_size_x)
            .ok()
            .and_then(|local| {
                usize::try_from(workgroup_count_x)
                    .ok()
                    .and_then(|groups| local.checked_mul(groups))
            })
            .ok_or_else(|| {
                canonical_calibration_error_value(
                    "canonical component dispatch logical extent overflowed",
                )
            })?;
        operations.push(VulkanPlacementOperationGeometry::Dispatch {
            geometry: VulkanPlacementDispatchGeometry {
                contract_id: contract.contract_id.clone(),
                logical_extent,
                sampled_extent: logical_extent,
                input_width: behavior.shape.input_byte_capacity,
                workgroup_count_x,
                local_size_x,
            },
        });
    }
    let contract_ids = contract_digests.keys().cloned().collect::<Vec<_>>();
    let implementation_digests = contract_digests.values().cloned().collect::<Vec<_>>();
    let artifact_digest = canonical_calibration_digest(
        b"nerve.canonical_component_artifacts.v1\0",
        &contracts
            .iter()
            .flat_map(|contract| &contract.artifacts)
            .collect::<Vec<_>>(),
    )?;
    let execution_graph_digest = canonical_calibration_digest(
        b"nerve.canonical_component_execution_graph.v1\0",
        &(
            target.signature_id.as_str(),
            behavior.phase,
            &behavior.shape,
            &contract_ids,
            &operations,
        ),
    )?;
    let physical_device_id = device.physical_device_id.clone();
    Ok(VulkanPlacementExecutionCaseIdentity {
        behavior: behavior.clone(),
        contract_ids,
        implementation_digests,
        artifact_digest,
        execution_graph_digest,
        operations,
        equivalence: VulkanPlacementEquivalenceIdentity::bit_exact(),
        strategy: VulkanPlacementExecutionStrategy::SingleDevice,
        devices: vec![device],
        shards: Vec::new(),
        input_physical_device_id: physical_device_id.clone(),
        output_physical_device_id: physical_device_id.clone(),
        owner_physical_device_id: physical_device_id,
        transports: Vec::new(),
    })
}

fn canonical_contract_dimension_u32(
    contract: &nerve_execution_contracts::PhysicalExecutionContract,
    name: &str,
) -> Result<u32, VulkanResidentTokenModelPackageError> {
    contract
        .geometry
        .dimensions
        .get(name)
        .copied()
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| *value > 0)
        .ok_or_else(|| {
            canonical_calibration_error_value(format!(
                "canonical contract {:?} has no positive u32 {name:?} dimension",
                contract.contract_id,
            ))
        })
}

fn canonical_calibration_output_artifact(
    report: &VulkanRuntimePlacementCalibrationReport,
) -> Result<VulkanPlacementOutputArtifact, VulkanResidentTokenModelPackageError> {
    let mut segments = report
        .captured_outputs
        .iter()
        .map(|output| {
            let bytes = decode_canonical_calibration_hex(&output.bytes_le_hex)?;
            if bytes.len() != output.byte_count {
                return canonical_calibration_error(format!(
                    "canonical output {:?} captured {} bytes but declared {}",
                    output.name,
                    bytes.len(),
                    output.byte_count,
                ));
            }
            Ok(VulkanPlacementOutputSegment {
                binding: output.binding,
                name: output.name.clone(),
                bytes,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    segments.sort_by(|left, right| {
        (left.binding, left.name.as_str()).cmp(&(right.binding, right.name.as_str()))
    });
    if segments.is_empty() {
        return canonical_calibration_error(
            "canonical component calibration captured no output signals",
        );
    }
    let output_artifact = VulkanPlacementOutputArtifact {
        scalar_format: VulkanPlacementScalarFormat::Bf16,
        segments,
    };
    let digest = vulkan_placement_output_artifact_digest(&output_artifact)
        .map_err(|error| canonical_calibration_error_value(error.to_string()))?;
    if digest != report.output_digest {
        return canonical_calibration_error(
            "canonical component output capture disagrees with its measured digest",
        );
    }
    Ok(output_artifact)
}

fn canonical_calibration_digest(
    tag: &[u8],
    value: &impl Serialize,
) -> Result<String, VulkanResidentTokenModelPackageError> {
    let payload = serde_json::to_vec(value).map_err(|error| {
        canonical_calibration_error_value(format!(
            "canonical calibration could not encode execution identity: {error}",
        ))
    })?;
    let mut digest = Sha256::new();
    digest.update(tag);
    digest.update(payload);
    Ok(format!("sha256:{:x}", digest.finalize()))
}

fn decode_canonical_calibration_hex(
    value: &str,
) -> Result<Vec<u8>, VulkanResidentTokenModelPackageError> {
    if !value.len().is_multiple_of(2) {
        return canonical_calibration_error(
            "canonical output contains odd-length hexadecimal data",
        );
    }
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            std::str::from_utf8(pair)
                .ok()
                .and_then(|encoded| u8::from_str_radix(encoded, 16).ok())
                .ok_or_else(|| {
                    canonical_calibration_error_value(
                        "canonical output contains invalid hexadecimal data",
                    )
                })
        })
        .collect()
}

fn canonical_calibration_error<T>(
    message: impl Into<String>,
) -> Result<T, VulkanResidentTokenModelPackageError> {
    Err(canonical_calibration_error_value(message))
}

fn canonical_calibration_error_value(
    message: impl Into<String>,
) -> VulkanResidentTokenModelPackageError {
    VulkanResidentTokenModelPackageError::new(message.into())
}

#[cfg(test)]
mod runtime_canonical_placement_calibration_tests {
    use super::*;

    #[test]
    fn canonical_output_hex_is_decoded_strictly() {
        assert_eq!(
            decode_canonical_calibration_hex("00a5ff").unwrap(),
            [0, 0xa5, 0xff]
        );
        assert!(decode_canonical_calibration_hex("0").is_err());
        assert!(decode_canonical_calibration_hex("0x").is_err());
    }

    #[test]
    fn canonical_contract_selection_rejects_distributed_and_wrong_artifacts() {
        let model = tests::tiny_fixture_model_runtime_model_with_placement(
            StreamCircuitPlacementSpec::new("gpu0"),
        );
        let kernel = &model.component_executions[0].kernels[0];
        let selected = canonical_single_device_contract_for_artifacts(
            &kernel.physical_execution_contracts,
            &kernel.op,
            &kernel.node_id,
            nerve_execution_contracts::ExecutionPhase::Decode,
            nerve_execution_contracts::ExecutionShape::SingleLane,
            &[kernel.shader_path.as_str()],
        )
        .unwrap();
        assert_eq!(
            selected.strategy,
            nerve_execution_contracts::ExecutionStrategy::SingleDevice
        );
        assert!(
            canonical_single_device_contract_for_artifacts(
                &kernel.physical_execution_contracts,
                &kernel.op,
                &kernel.node_id,
                nerve_execution_contracts::ExecutionPhase::Decode,
                nerve_execution_contracts::ExecutionShape::SingleLane,
                &["shaders/not-the-executed-artifact.spv"],
            )
            .is_err()
        );
    }

    #[test]
    fn canonical_decode_uses_the_loaded_reusable_artifact_not_the_plan_placeholder() {
        let loaded = VulkanLoadedKernelArtifactCatalog {
            reusable_artifacts: vec![VulkanLoadedReusableKernelArtifact {
                artifact: VulkanReusableKernelArtifact {
                    family_id: "family".to_string(),
                    op: "op".to_string(),
                    path: "shaders/selected.spv".to_string(),
                    entry_point: "main".to_string(),
                    local_size_x: 64,
                    workgroup_count_x: 1,
                    descriptor_signature: Vec::new(),
                    push_constants: Vec::new(),
                    stream_control_binding: None,
                },
                resolved_path: PathBuf::from("/compiled/shaders/selected.spv"),
                words: vec![0x0723_0203],
            }],
            physical_artifacts: Vec::new(),
            reusable_word_count: 1,
            physical_word_count: 0,
        };

        assert_eq!(
            canonical_loaded_reusable_artifact_path(
                &loaded,
                "family",
                "component",
                "node",
            )
            .unwrap(),
            "shaders/selected.spv",
        );
        assert!(
            canonical_loaded_reusable_artifact_path(
                &loaded,
                "missing",
                "component",
                "node",
            )
            .unwrap_err()
            .to_string()
            .contains("loaded reusable artifact")
        );
    }

    #[test]
    fn canonical_scalar_lane_prefill_repeats_the_exact_primary_contract() {
        let model = tests::tiny_fixture_model_runtime_model_with_placement(
            StreamCircuitPlacementSpec::new("gpu0"),
        );
        let kernel = &model.component_executions[0].kernels[0];

        let contracts = canonical_scalar_lane_prefill_contracts(
            &kernel.physical_execution_contracts,
            &kernel.op,
            &kernel.node_id,
            &kernel.shader_path,
            4,
        )
        .unwrap();

        assert_eq!(contracts.len(), 4);
        assert!(contracts.windows(2).all(|pair| pair[0] == pair[1]));
        assert!(canonical_scalar_lane_prefill_contracts(
            &kernel.physical_execution_contracts,
            &kernel.op,
            &kernel.node_id,
            &kernel.shader_path,
            0,
        )
        .is_err());
    }

    #[test]
    fn canonical_only_behavior_uses_the_typed_component_boundary() {
        let model = tests::tiny_fixture_model_runtime_model_with_placement(
            StreamCircuitPlacementSpec::new("gpu0"),
        );
        let target = vulkan_runtime_placement_calibration_targets(&model)
            .unwrap()
            .remove(0);
        let behavior = canonical_component_boundary_behavior(
            &model,
            &target,
            VulkanTargetedComponentExecutionPhase::Decode,
        )
        .unwrap();

        assert_eq!(behavior.compiled_execution_signature, target.signature_id);
        assert_eq!(behavior.shape.activation_batch_width, 1);
        assert!(behavior.shape.input_byte_capacity > 0);
        assert!(behavior.shape.output_byte_capacity > 0);
        assert!(behavior.input_fixture_digest.starts_with("sha256:"));

        let mut invalid = model.clone();
        invalid
            .circuit_graph
            .components
            .iter_mut()
            .find(|component| component.component_id == target.component_id)
            .unwrap()
            .circuit
            .boundary
            .inputs[0]
            .dtype = Some("NOT_A_DTYPE".to_string());
        assert!(
            canonical_component_boundary_behavior(
                &invalid,
                &target,
                VulkanTargetedComponentExecutionPhase::Decode,
            )
            .unwrap_err()
            .to_string()
            .contains("unsupported circuit dtype")
        );
    }

    #[test]
    fn canonical_single_device_case_is_valid_catalog_evidence() {
        let model = tests::tiny_fixture_model_runtime_model_with_placement(
            StreamCircuitPlacementSpec::new("gpu0"),
        );
        let target = vulkan_runtime_placement_calibration_targets(&model)
            .unwrap()
            .remove(0);
        let execution = model
            .component_executions
            .iter()
            .find(|execution| execution.component_id == target.component_id)
            .unwrap();
        let contracts = execution
            .kernels
            .iter()
            .filter(|kernel| kernel.execution_domain.supports_decode())
            .map(|kernel| {
                canonical_single_device_contract_for_artifacts(
                    &kernel.physical_execution_contracts,
                    &kernel.op,
                    &kernel.node_id,
                    nerve_execution_contracts::ExecutionPhase::Decode,
                    nerve_execution_contracts::ExecutionShape::SingleLane,
                    &[kernel.shader_path.as_str()],
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let behavior = VulkanPlacementBehaviorIdentity {
            compiled_execution_signature: target.signature_id.clone(),
            runtime_implementation_fingerprint: "runtime".to_string(),
            phase: nerve_execution_contracts::ExecutionPhase::Decode,
            shape: VulkanPlacementShapeClass {
                activation_batch_width: 1,
                input_byte_capacity: 16,
                output_byte_capacity: 16,
            },
            input_fixture_digest: format!("sha256:{}", "a".repeat(64)),
        };
        let execution_case = canonical_component_execution_case(
            VulkanPlacementDeviceExecutionIdentity {
                physical_device_id: "gpu0".to_string(),
                api_version: 1,
                driver_version: 2,
            },
            &target,
            &behavior,
            &contracts,
        )
        .unwrap();
        assert!(execution_case.shards.is_empty());
        assert_eq!(
            execution_case.strategy,
            VulkanPlacementExecutionStrategy::SingleDevice
        );

        let calibration = VulkanRuntimeCanonicalPlacementCalibration {
            reference: VulkanPlacementCanonicalReference {
                behavior,
                output_digest: "output".to_string(),
                output_artifact: None,
                state_digest: "state".to_string(),
            },
            observation: VulkanPlacementCalibrationObservation {
                execution_case,
                warmup_call_count: 1,
                measured_call_count: 1,
                complete_transaction: true,
                duration_ns: 10,
                useful_activation_count: 1,
                output_digest: "output".to_string(),
                output_artifact: None,
                output_equivalence: VulkanPlacementOutputEquivalenceEvidence::BitExact,
                state_digest: "state".to_string(),
                resident_bytes_by_physical_device: BTreeMap::from([(
                    "gpu0".to_string(),
                    16,
                )]),
                transient_peak_bytes_by_physical_device: BTreeMap::from([(
                    "gpu0".to_string(),
                    8,
                )]),
                host_resident_bytes: 0,
                host_transient_peak_bytes: 0,
            },
        };
        let mut catalog = VulkanPlacementCalibrationCatalog::default();
        record_vulkan_runtime_canonical_placement_calibration(
            &mut catalog,
            calibration.clone(),
        )
        .unwrap();
        assert_eq!(catalog.observation_count(), 1);

        let before = catalog.clone();
        let mut conflicting = calibration;
        conflicting.reference.output_digest = "different".to_string();
        conflicting.observation.execution_case.devices[0].driver_version += 1;
        assert!(
            record_vulkan_runtime_canonical_placement_calibration(
                &mut catalog,
                conflicting,
            )
            .unwrap_err()
            .0
            .contains("different reference evidence")
        );
        assert_eq!(catalog, before);
    }
}
