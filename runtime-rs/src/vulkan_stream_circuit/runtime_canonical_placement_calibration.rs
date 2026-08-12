struct VulkanRuntimeCanonicalPlacementCalibration {
    reference: VulkanPlacementCanonicalReference,
    observation: VulkanPlacementCalibrationObservation,
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
        .plans_for_device(&device, physical_device_id, manifest_dir)?
        .0;
    let [cached_plan] = plans.as_slice() else {
        return canonical_calibration_error(
            "canonical component calibration did not retain exactly one executable plan",
        );
    };
    let contracts = canonical_component_execution_contracts(cached_plan, target, phase)?;
    let output_artifact = canonical_calibration_output_artifact(report)?;
    let device_identity = VulkanPlacementDeviceExecutionIdentity {
        physical_device_id: device.physical_device_id().to_string(),
        api_version: device.api_version(),
        driver_version: device.driver_version(),
    };
    let execution_case = canonical_component_execution_case(
        device_identity,
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
    Ok(Some(VulkanRuntimeCanonicalPlacementCalibration {
        reference,
        observation,
    }))
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
        .map(|dispatch| {
            let (contract_phase, execution_shape, artifact_paths) = match phase {
                VulkanTargetedComponentExecutionPhase::Decode => (
                    nerve_execution_contracts::ExecutionPhase::Decode,
                    nerve_execution_contracts::ExecutionShape::SingleLane,
                    vec![dispatch.artifact_path.as_str()],
                ),
                VulkanTargetedComponentExecutionPhase::Prefill {
                    activation_batch_width,
                } => {
                    let artifact = select_component_batch_kernel_artifact(
                        &slice_plan.batch_kernels,
                        &dispatch.component_id,
                        &dispatch.node_id,
                        VulkanComponentBatchExecutionMode::CausalSequence,
                        activation_batch_width,
                    )
                    .filter(|artifact| {
                        component_batch_stages_replace_push_constants(
                            &artifact.stages,
                            &dispatch.push_constants,
                        )
                    })
                    .filter(|artifact| {
                        targeted_prefill_batch_mode_is_supported(artifact.batch_mode)
                    })
                    .ok_or_else(|| {
                        canonical_calibration_error_value(format!(
                            "canonical prefill dispatch {}.{} has no compiler-declared executable batch artifact",
                            dispatch.component_id, dispatch.node_id,
                        ))
                    })?;
                    (
                        nerve_execution_contracts::ExecutionPhase::Prefill,
                        nerve_execution_contracts::ExecutionShape::MultiLane,
                        artifact
                            .stages
                            .iter()
                            .map(|stage| stage.shader_path.as_str())
                            .collect(),
                    )
                }
            };
            canonical_single_device_contract_for_artifacts(
                &dispatch.physical_execution_contracts,
                &dispatch.op,
                &dispatch.node_id,
                contract_phase,
                execution_shape,
                &artifact_paths,
            )
        })
        .collect()
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

        let mut catalog = VulkanPlacementCalibrationCatalog::default();
        catalog
            .record_reference(VulkanPlacementCanonicalReference {
                behavior,
                output_digest: "output".to_string(),
                output_artifact: None,
                state_digest: "state".to_string(),
            })
            .unwrap();
        catalog
            .record_observation(VulkanPlacementCalibrationObservation {
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
            })
            .unwrap();
        assert_eq!(catalog.observation_count(), 1);
    }
}
