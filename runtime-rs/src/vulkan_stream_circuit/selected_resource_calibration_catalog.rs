/// Exact evidence joining one complete selected-resource transaction with the
/// corresponding one-resource lazy-load transaction. The two source
/// observations remain canonical catalog records; this record is only a typed
/// relationship between them and a compiler-declared execution class.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct VulkanPlacementSelectedResourceExecutionClassCalibration {
    pub resource_execution_class_id: String,
    pub resource_payload_byte_count: usize,
    pub execution_case: VulkanPlacementExecutionCaseIdentity,
    pub lazy_load_wave_case: VulkanPlacementExecutionCaseIdentity,
}

/// Exact current-runtime identity that a selected-resource cost must match.
/// The execution-class digest alone is insufficient because shader bytes,
/// graph topology, runtime ABI, phase, and activation shape can all change the
/// measured cost or invalidate the result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPlacementSelectedResourceExecutionClassRequirement {
    pub resource_execution_class_id: String,
    pub compiled_execution_signature: String,
    pub runtime_implementation_fingerprint: String,
    pub phase: nerve_execution_contracts::ExecutionPhase,
    pub shape: VulkanPlacementShapeClass,
    pub artifact_digest: String,
    pub execution_graph_digest: String,
}

/// Reservation-aware capacity supplied at the instant placement is solved.
/// Calibration never supplies capacity: it describes measured work on an
/// exact target identity, while this structure describes what the current
/// process may safely acquire now.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPlacementSelectedResourceDeviceCapacity {
    pub device_id: String,
    pub identity: VulkanPlacementDeviceExecutionIdentity,
    pub resident_payload_capacity_bytes: usize,
}

impl VulkanPlacementCalibrationCatalog {
    pub fn record_selected_resource_execution_class(
        &mut self,
        calibration: VulkanPlacementSelectedResourceExecutionClassCalibration,
    ) -> Result<(), VulkanPlacementCalibrationCatalogError> {
        validate_selected_resource_execution_class_calibration(self, &calibration)?;
        let key = selected_resource_execution_class_calibration_key(&calibration);
        if let Some(existing) = self
            .selected_resource_execution_classes
            .iter()
            .find(|existing| selected_resource_execution_class_calibration_key(existing) == key)
        {
            if existing == &calibration {
                return Ok(());
            }
            return Err(VulkanPlacementCalibrationCatalogError(
                "selected-resource execution class has conflicting exact calibration evidence"
                    .to_string(),
            ));
        }
        let index = self
            .selected_resource_execution_classes
            .binary_search(&calibration)
            .unwrap_or_else(|index| index);
        self.selected_resource_execution_classes
            .insert(index, calibration);
        Ok(())
    }

    /// Resolves placement inputs only when every required class has one exact
    /// output-valid execution and load-wave observation on every candidate
    /// device. Missing or ambiguous evidence fails closed; no averages,
    /// advertised throughput, or neighboring representation are substituted.
    pub fn selected_resource_placement_devices(
        &self,
        requirements: &[VulkanPlacementSelectedResourceExecutionClassRequirement],
        capacities: &[VulkanPlacementSelectedResourceDeviceCapacity],
    ) -> Result<
        Vec<crate::vulkan_distributed::VulkanSelectedResourcePlacementDevice>,
        VulkanPlacementCalibrationCatalogError,
    > {
        self.try_selected_resource_placement_devices(requirements, capacities)?
            .ok_or_else(|| {
                VulkanPlacementCalibrationCatalogError(
                    "selected-resource placement lacks exact calibration coverage on at least one candidate device"
                        .to_string(),
                )
            })
    }

    /// Returns `None` only for absent exact evidence. Invalid, stale,
    /// conflicting, or ambiguous evidence remains an error and must not be
    /// treated as a harmless optimization miss.
    pub fn try_selected_resource_placement_devices(
        &self,
        requirements: &[VulkanPlacementSelectedResourceExecutionClassRequirement],
        capacities: &[VulkanPlacementSelectedResourceDeviceCapacity],
    ) -> Result<
        Option<Vec<crate::vulkan_distributed::VulkanSelectedResourcePlacementDevice>>,
        VulkanPlacementCalibrationCatalogError,
    > {
        validate_selected_resource_execution_class_requirements(requirements)?;
        validate_selected_resource_device_capacities(capacities)?;
        let mut devices = Vec::with_capacity(capacities.len());
        for capacity in capacities {
            let mut measured_costs_by_execution_class = BTreeMap::new();
            for requirement in requirements {
                let matching = self
                    .selected_resource_execution_classes
                    .iter()
                    .filter(|calibration| {
                        selected_resource_execution_class_matches(
                            self,
                            calibration,
                            requirement,
                            &capacity.identity,
                        )
                    })
                    .collect::<Vec<_>>();
                if matching.is_empty() {
                    return Ok(None);
                }
                let [calibration] = matching.as_slice() else {
                    return Err(VulkanPlacementCalibrationCatalogError(format!(
                        "selected-resource class {:?} has {} exact calibrations on physical device {:?}; expected one",
                        requirement.resource_execution_class_id,
                        matching.len(),
                        capacity.identity.physical_device_id,
                    )));
                };
                let execution = self
                    .exact_observation(&calibration.execution_case)
                    .expect("matching calibration was validated against this catalog");
                let load_wave = self
                    .exact_observation(&calibration.lazy_load_wave_case)
                    .expect("matching load wave was validated against this catalog");
                measured_costs_by_execution_class.insert(
                    requirement.resource_execution_class_id.clone(),
                    crate::vulkan_distributed::VulkanSelectedResourceExecutionClassCost {
                        phase: requirement.phase,
                        complete_transaction: execution.complete_transaction
                            && load_wave.complete_transaction,
                        output_valid: true,
                        warmup_call_count: execution.warmup_call_count,
                        measured_call_count: execution.measured_call_count,
                        execution_duration_ns: execution.duration_ns,
                        lazy_load_wave_duration_ns: load_wave.duration_ns,
                    },
                );
            }
            devices.push(
                crate::vulkan_distributed::VulkanSelectedResourcePlacementDevice {
                    device_id: capacity.device_id.clone(),
                    physical_device_id: capacity.identity.physical_device_id.clone(),
                    api_version: capacity.identity.api_version,
                    driver_version: capacity.identity.driver_version,
                    resident_payload_capacity_bytes: capacity.resident_payload_capacity_bytes,
                    measured_costs_by_execution_class,
                },
            );
        }
        Ok(Some(devices))
    }
}

fn validate_selected_resource_execution_class_calibration(
    catalog: &VulkanPlacementCalibrationCatalog,
    calibration: &VulkanPlacementSelectedResourceExecutionClassCalibration,
) -> Result<(), VulkanPlacementCalibrationCatalogError> {
    if !valid_sha256_digest(&calibration.resource_execution_class_id)
        || calibration.resource_payload_byte_count == 0
    {
        return Err(VulkanPlacementCalibrationCatalogError(
            "selected-resource calibration requires an exact class and positive payload"
                .to_string(),
        ));
    }
    let execution = catalog
        .exact_observation(&calibration.execution_case)
        .ok_or_else(|| {
            VulkanPlacementCalibrationCatalogError(
                "selected-resource calibration has no exact execution observation".to_string(),
            )
        })?;
    let load_wave = catalog
        .exact_observation(&calibration.lazy_load_wave_case)
        .ok_or_else(|| {
            VulkanPlacementCalibrationCatalogError(
                "selected-resource calibration has no exact lazy-load observation".to_string(),
            )
        })?;
    let execution_device = single_device_for_selected_resource_observation(
        execution,
        VulkanPlacementExecutionStrategy::SelectedResourceTransaction,
    )?;
    let load_wave_device = single_device_for_selected_resource_observation(
        load_wave,
        VulkanPlacementExecutionStrategy::LazyLoadWave,
    )?;
    let load_geometry = match load_wave.execution_case.operations.as_slice() {
        [VulkanPlacementOperationGeometry::LazyLoadWave {
            resource_count,
            byte_count,
            ..
        }] => Some((*resource_count, *byte_count)),
        _ => None,
    };
    let execution_geometry = execution
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
                resource_execution_class_id.as_str(),
                *selector_selection_count,
                *executed_resource_occurrence_count,
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    let exact_single_occurrence = matches!(
        execution_geometry.as_slice(),
        [(class_id, selector_selection_count, 1)]
            if *class_id == calibration.resource_execution_class_id
                && *selector_selection_count > 0
    );
    if execution_device != load_wave_device
        || execution.execution_case.behavior.phase
            != load_wave.execution_case.behavior.phase
        || execution
            .execution_case
            .behavior
            .runtime_implementation_fingerprint
            != load_wave
                .execution_case
                .behavior
                .runtime_implementation_fingerprint
        || execution
            .execution_case
            .behavior
            .shape
            .activation_batch_width
            != load_wave
                .execution_case
                .behavior
                .shape
                .activation_batch_width
        || execution.warmup_call_count != load_wave.warmup_call_count
        || execution.measured_call_count != load_wave.measured_call_count
        || load_geometry != Some((1, calibration.resource_payload_byte_count))
        || !exact_single_occurrence
    {
        return Err(VulkanPlacementCalibrationCatalogError(
            "selected-resource execution and lazy-load evidence do not describe one exact device, phase, shape, call shape, and executed resource occurrence"
                .to_string(),
        ));
    }
    Ok(())
}

fn single_device_for_selected_resource_observation(
    observation: &VulkanPlacementCalibrationObservation,
    expected_strategy: VulkanPlacementExecutionStrategy,
) -> Result<&VulkanPlacementDeviceExecutionIdentity, VulkanPlacementCalibrationCatalogError> {
    let [device] = observation.execution_case.devices.as_slice() else {
        return Err(VulkanPlacementCalibrationCatalogError(
            "selected-resource class calibration requires one physical device".to_string(),
        ));
    };
    if observation.execution_case.strategy != expected_strategy
        || observation.execution_case.input_physical_device_id != device.physical_device_id
        || observation.execution_case.output_physical_device_id != device.physical_device_id
        || observation.execution_case.owner_physical_device_id != device.physical_device_id
        || !observation.execution_case.shards.is_empty()
        || !observation.execution_case.transports.is_empty()
    {
        return Err(VulkanPlacementCalibrationCatalogError(
            "selected-resource class calibration uses an incompatible physical execution case"
                .to_string(),
        ));
    }
    Ok(device)
}

fn selected_resource_execution_class_calibration_key(
    calibration: &VulkanPlacementSelectedResourceExecutionClassCalibration,
) -> (
    &str,
    &str,
    &str,
    nerve_execution_contracts::ExecutionPhase,
    &VulkanPlacementShapeClass,
    &str,
    &str,
    &VulkanPlacementDeviceExecutionIdentity,
) {
    let behavior = &calibration.execution_case.behavior;
    (
        &calibration.resource_execution_class_id,
        &behavior.compiled_execution_signature,
        &behavior.runtime_implementation_fingerprint,
        behavior.phase,
        &behavior.shape,
        &calibration.execution_case.artifact_digest,
        &calibration.execution_case.execution_graph_digest,
        calibration
            .execution_case
            .devices
            .first()
            .expect("validated calibration has one device"),
    )
}

fn validate_selected_resource_execution_class_requirements(
    requirements: &[VulkanPlacementSelectedResourceExecutionClassRequirement],
) -> Result<(), VulkanPlacementCalibrationCatalogError> {
    if requirements.is_empty() {
        return Err(VulkanPlacementCalibrationCatalogError(
            "selected-resource placement requires at least one execution class".to_string(),
        ));
    }
    let mut class_ids = BTreeSet::new();
    if requirements.iter().any(|requirement| {
        !valid_sha256_digest(&requirement.resource_execution_class_id)
            || !class_ids.insert(requirement.resource_execution_class_id.as_str())
            || requirement.compiled_execution_signature.is_empty()
            || requirement.runtime_implementation_fingerprint.is_empty()
            || requirement.shape.activation_batch_width == 0
            || requirement.shape.input_byte_capacity == 0
            || requirement.shape.output_byte_capacity == 0
            || !valid_sha256_digest(&requirement.artifact_digest)
            || !valid_sha256_digest(&requirement.execution_graph_digest)
    }) {
        return Err(VulkanPlacementCalibrationCatalogError(
            "selected-resource placement requirements are incomplete or repeat a class"
                .to_string(),
        ));
    }
    Ok(())
}

fn validate_selected_resource_device_capacities(
    capacities: &[VulkanPlacementSelectedResourceDeviceCapacity],
) -> Result<(), VulkanPlacementCalibrationCatalogError> {
    if capacities.is_empty() {
        return Err(VulkanPlacementCalibrationCatalogError(
            "selected-resource placement requires at least one current device capacity"
                .to_string(),
        ));
    }
    let mut logical_ids = BTreeSet::new();
    let mut physical_ids = BTreeSet::new();
    if capacities.iter().any(|capacity| {
        capacity.device_id.is_empty()
            || !logical_ids.insert(capacity.device_id.as_str())
            || capacity.identity.physical_device_id.is_empty()
            || !physical_ids.insert(capacity.identity.physical_device_id.as_str())
            || capacity.identity.api_version == 0
            || capacity.identity.driver_version == 0
            || capacity.resident_payload_capacity_bytes == 0
    }) {
        return Err(VulkanPlacementCalibrationCatalogError(
            "selected-resource placement capacities require unique exact devices and positive remaining capacity"
                .to_string(),
        ));
    }
    Ok(())
}

fn selected_resource_execution_class_matches(
    catalog: &VulkanPlacementCalibrationCatalog,
    calibration: &VulkanPlacementSelectedResourceExecutionClassCalibration,
    requirement: &VulkanPlacementSelectedResourceExecutionClassRequirement,
    device: &VulkanPlacementDeviceExecutionIdentity,
) -> bool {
    if validate_selected_resource_execution_class_calibration(catalog, calibration).is_err() {
        return false;
    }
    let case = &calibration.execution_case;
    calibration.resource_execution_class_id == requirement.resource_execution_class_id
        && case.behavior.compiled_execution_signature
            == requirement.compiled_execution_signature
        && case.behavior.runtime_implementation_fingerprint
            == requirement.runtime_implementation_fingerprint
        && case.behavior.phase == requirement.phase
        && case.behavior.shape == requirement.shape
        && case.artifact_digest == requirement.artifact_digest
        && case.execution_graph_digest == requirement.execution_graph_digest
        && case.devices.as_slice() == std::slice::from_ref(device)
}

#[cfg(test)]
mod selected_resource_calibration_catalog_tests {
    use super::*;

    fn digest(byte: char) -> String {
        format!("sha256:{}", byte.to_string().repeat(64))
    }

    fn device(physical_device_id: &str) -> VulkanPlacementDeviceExecutionIdentity {
        VulkanPlacementDeviceExecutionIdentity {
            physical_device_id: physical_device_id.to_string(),
            api_version: 14,
            driver_version: 27,
        }
    }

    fn behavior(
        signature: &str,
        phase: nerve_execution_contracts::ExecutionPhase,
        shape: VulkanPlacementShapeClass,
        fixture: char,
    ) -> VulkanPlacementBehaviorIdentity {
        VulkanPlacementBehaviorIdentity {
            compiled_execution_signature: signature.to_string(),
            runtime_implementation_fingerprint: "runtime-fingerprint".to_string(),
            phase,
            shape,
            input_fixture_digest: digest(fixture),
        }
    }

    fn observation(
        physical_device_id: &str,
        behavior: VulkanPlacementBehaviorIdentity,
        strategy: VulkanPlacementExecutionStrategy,
        operations: Vec<VulkanPlacementOperationGeometry>,
        artifact: char,
        graph: char,
        duration_ns: u64,
        output: &str,
        state: &str,
    ) -> VulkanPlacementCalibrationObservation {
        let device = device(physical_device_id);
        VulkanPlacementCalibrationObservation {
            execution_case: VulkanPlacementExecutionCaseIdentity {
                behavior,
                contract_ids: vec!["contract".to_string()],
                implementation_digests: vec![digest('1')],
                artifact_digest: digest(artifact),
                execution_graph_digest: digest(graph),
                operations,
                equivalence: VulkanPlacementEquivalenceIdentity::bit_exact(),
                strategy,
                devices: vec![device],
                shards: Vec::new(),
                input_physical_device_id: physical_device_id.to_string(),
                output_physical_device_id: physical_device_id.to_string(),
                owner_physical_device_id: physical_device_id.to_string(),
                transports: Vec::new(),
            },
            warmup_call_count: 1,
            measured_call_count: 1,
            complete_transaction: true,
            duration_ns,
            useful_activation_count: 1,
            output_digest: output.to_string(),
            output_artifact: None,
            output_equivalence: VulkanPlacementOutputEquivalenceEvidence::BitExact,
            state_digest: state.to_string(),
            resident_bytes_by_physical_device: BTreeMap::from([(
                physical_device_id.to_string(),
                32,
            )]),
            transient_peak_bytes_by_physical_device: BTreeMap::from([(
                physical_device_id.to_string(),
                16,
            )]),
            host_resident_bytes: 0,
            host_transient_peak_bytes: 8,
        }
    }

    fn execution_observation(
        physical_device_id: &str,
    ) -> VulkanPlacementCalibrationObservation {
        observation(
            physical_device_id,
            behavior(
                "compiled-expert-signature",
                nerve_execution_contracts::ExecutionPhase::Decode,
                VulkanPlacementShapeClass {
                    activation_batch_width: 1,
                    input_byte_capacity: 8192,
                    output_byte_capacity: 8192,
                },
                'e',
            ),
            VulkanPlacementExecutionStrategy::SelectedResourceTransaction,
            vec![
                VulkanPlacementOperationGeometry::SelectedResourceTransaction {
                    contract_id: "contract".to_string(),
                    resource_execution_class_id: digest('f'),
                    selector_selection_count: 6,
                    executed_resource_occurrence_count: 1,
                },
                VulkanPlacementOperationGeometry::Dispatch {
                    geometry: VulkanPlacementDispatchGeometry {
                        contract_id: "contract".to_string(),
                        logical_extent: 4096,
                        sampled_extent: 4096,
                        input_width: 4096,
                        workgroup_count_x: 16,
                        local_size_x: 256,
                    },
                },
            ],
            'a',
            'b',
            1_500,
            "expert-output",
            "expert-state",
        )
    }

    fn load_wave_observation(
        physical_device_id: &str,
        payload_bytes: usize,
    ) -> VulkanPlacementCalibrationObservation {
        observation(
            physical_device_id,
            behavior(
                "compiled-load-signature",
                nerve_execution_contracts::ExecutionPhase::Decode,
                VulkanPlacementShapeClass {
                    activation_batch_width: 1,
                    input_byte_capacity: payload_bytes,
                    output_byte_capacity: payload_bytes,
                },
                '5',
            ),
            VulkanPlacementExecutionStrategy::LazyLoadWave,
            vec![VulkanPlacementOperationGeometry::LazyLoadWave {
                contract_id: "contract".to_string(),
                resource_count: 1,
                byte_count: payload_bytes,
            }],
            'c',
            'd',
            2_500,
            "load-output",
            "load-state",
        )
    }

    fn record_observation_with_reference(
        catalog: &mut VulkanPlacementCalibrationCatalog,
        observation: VulkanPlacementCalibrationObservation,
    ) {
        catalog
            .record_reference(VulkanPlacementCanonicalReference {
                behavior: observation.execution_case.behavior.clone(),
                output_digest: observation.output_digest.clone(),
                output_artifact: None,
                state_digest: observation.state_digest.clone(),
            })
            .unwrap();
        catalog.record_observation(observation).unwrap();
    }

    fn catalog_with_exact_class(
    ) -> (
        VulkanPlacementCalibrationCatalog,
        VulkanPlacementSelectedResourceExecutionClassRequirement,
    ) {
        let execution = execution_observation("gpu0");
        let load_wave = load_wave_observation("gpu0", 4096);
        let requirement = VulkanPlacementSelectedResourceExecutionClassRequirement {
            resource_execution_class_id: digest('f'),
            compiled_execution_signature: execution
                .execution_case
                .behavior
                .compiled_execution_signature
                .clone(),
            runtime_implementation_fingerprint: execution
                .execution_case
                .behavior
                .runtime_implementation_fingerprint
                .clone(),
            phase: execution.execution_case.behavior.phase,
            shape: execution.execution_case.behavior.shape.clone(),
            artifact_digest: execution.execution_case.artifact_digest.clone(),
            execution_graph_digest: execution.execution_case.execution_graph_digest.clone(),
        };
        let calibration = VulkanPlacementSelectedResourceExecutionClassCalibration {
            resource_execution_class_id: requirement.resource_execution_class_id.clone(),
            resource_payload_byte_count: 4096,
            execution_case: execution.execution_case.clone(),
            lazy_load_wave_case: load_wave.execution_case.clone(),
        };
        let mut catalog = VulkanPlacementCalibrationCatalog::default();
        record_observation_with_reference(&mut catalog, execution);
        record_observation_with_reference(&mut catalog, load_wave);
        catalog
            .record_selected_resource_execution_class(calibration)
            .unwrap();
        (catalog, requirement)
    }

    #[test]
    fn exact_class_join_produces_planner_costs_without_inference() {
        let (catalog, requirement) = catalog_with_exact_class();
        let devices = catalog
            .selected_resource_placement_devices(
                &[requirement],
                &[VulkanPlacementSelectedResourceDeviceCapacity {
                    device_id: "logical0".to_string(),
                    identity: device("gpu0"),
                    resident_payload_capacity_bytes: 1 << 20,
                }],
            )
            .unwrap();

        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].device_id, "logical0");
        let cost = devices[0]
            .measured_costs_by_execution_class
            .values()
            .next()
            .unwrap();
        assert_eq!(cost.execution_duration_ns, 1_500);
        assert_eq!(cost.lazy_load_wave_duration_ns, 2_500);
        assert!(cost.complete_transaction);
        assert!(cost.output_valid);
    }

    #[test]
    fn class_join_rejects_cross_device_and_multi_resource_load_evidence() {
        let execution = execution_observation("gpu0");
        let foreign_load = load_wave_observation("gpu1", 4096);
        let mut catalog = VulkanPlacementCalibrationCatalog::default();
        record_observation_with_reference(&mut catalog, execution.clone());
        record_observation_with_reference(&mut catalog, foreign_load.clone());
        let class_id = digest('f');
        let error = catalog
            .record_selected_resource_execution_class(
                VulkanPlacementSelectedResourceExecutionClassCalibration {
                    resource_execution_class_id: class_id.clone(),
                    resource_payload_byte_count: 4096,
                    execution_case: execution.execution_case.clone(),
                    lazy_load_wave_case: foreign_load.execution_case,
                },
            )
            .unwrap_err();
        assert!(error.to_string().contains("one exact device"));

        let mut multi_load = load_wave_observation("gpu0", 4096);
        multi_load.execution_case.operations =
            vec![VulkanPlacementOperationGeometry::LazyLoadWave {
                contract_id: "contract".to_string(),
                resource_count: 2,
                byte_count: 4096,
            }];
        let mut catalog = VulkanPlacementCalibrationCatalog::default();
        record_observation_with_reference(&mut catalog, execution.clone());
        record_observation_with_reference(&mut catalog, multi_load.clone());
        assert!(
            catalog
                .record_selected_resource_execution_class(
                    VulkanPlacementSelectedResourceExecutionClassCalibration {
                        resource_execution_class_id: class_id,
                        resource_payload_byte_count: 4096,
                        execution_case: execution.execution_case,
                        lazy_load_wave_case: multi_load.execution_case,
                    },
                )
                .unwrap_err()
                .to_string()
                .contains("one exact device, phase, shape, call shape, and executed resource occurrence")
        );

        let execution = execution_observation("gpu0");
        let mut mismatched_counts = load_wave_observation("gpu0", 4096);
        mismatched_counts.measured_call_count = 2;
        let mut catalog = VulkanPlacementCalibrationCatalog::default();
        record_observation_with_reference(&mut catalog, execution.clone());
        record_observation_with_reference(&mut catalog, mismatched_counts.clone());
        assert!(
            catalog
                .record_selected_resource_execution_class(
                    VulkanPlacementSelectedResourceExecutionClassCalibration {
                        resource_execution_class_id: digest('f'),
                        resource_payload_byte_count: 4096,
                        execution_case: execution.execution_case,
                        lazy_load_wave_case: mismatched_counts.execution_case,
                    },
                )
                .is_err()
        );
    }

    #[test]
    fn class_join_rejects_a_relabelled_full_selector_wave() {
        let mut execution = execution_observation("gpu0");
        let VulkanPlacementOperationGeometry::SelectedResourceTransaction {
            executed_resource_occurrence_count,
            ..
        } = &mut execution.execution_case.operations[0]
        else {
            panic!("fixture must begin with selected-resource geometry");
        };
        *executed_resource_occurrence_count = 6;
        let load_wave = load_wave_observation("gpu0", 4096);
        let mut catalog = VulkanPlacementCalibrationCatalog::default();
        record_observation_with_reference(&mut catalog, execution.clone());
        record_observation_with_reference(&mut catalog, load_wave.clone());

        let error = catalog
            .record_selected_resource_execution_class(
                VulkanPlacementSelectedResourceExecutionClassCalibration {
                    resource_execution_class_id: digest('f'),
                    resource_payload_byte_count: 4096,
                    execution_case: execution.execution_case,
                    lazy_load_wave_case: load_wave.execution_case,
                },
            )
            .unwrap_err();

        assert!(error.to_string().contains("executed resource occurrence"));
    }

    #[test]
    fn selected_resource_geometry_cannot_be_omitted_or_used_by_another_strategy() {
        let mut missing_geometry = execution_observation("gpu0");
        missing_geometry.execution_case.operations.remove(0);
        let mut catalog = VulkanPlacementCalibrationCatalog::default();
        catalog
            .record_reference(VulkanPlacementCanonicalReference {
                behavior: missing_geometry.execution_case.behavior.clone(),
                output_digest: missing_geometry.output_digest.clone(),
                output_artifact: None,
                state_digest: missing_geometry.state_digest.clone(),
            })
            .unwrap();
        assert!(catalog.record_observation(missing_geometry).is_err());

        let mut wrong_strategy = execution_observation("gpu0");
        wrong_strategy.execution_case.strategy = VulkanPlacementExecutionStrategy::SingleDevice;
        let mut catalog = VulkanPlacementCalibrationCatalog::default();
        catalog
            .record_reference(VulkanPlacementCanonicalReference {
                behavior: wrong_strategy.execution_case.behavior.clone(),
                output_digest: wrong_strategy.output_digest.clone(),
                output_artifact: None,
                state_digest: wrong_strategy.state_digest.clone(),
            })
            .unwrap();
        assert!(catalog.record_observation(wrong_strategy).is_err());
    }

    #[test]
    fn class_resolution_fails_closed_for_stale_or_missing_identity() {
        let (catalog, requirement) = catalog_with_exact_class();
        let capacity = VulkanPlacementSelectedResourceDeviceCapacity {
            device_id: "logical0".to_string(),
            identity: device("gpu0"),
            resident_payload_capacity_bytes: 1 << 20,
        };

        let mut stale = requirement.clone();
        stale.artifact_digest = digest('9');
        assert_eq!(
            catalog
                .try_selected_resource_placement_devices(
                    std::slice::from_ref(&stale),
                    std::slice::from_ref(&capacity),
                )
                .unwrap(),
            None,
        );
        assert!(
            catalog
                .selected_resource_placement_devices(&[stale], std::slice::from_ref(&capacity))
                .unwrap_err()
                .to_string()
                .contains("lacks exact calibration coverage")
        );

        let mut missing = requirement;
        missing.resource_execution_class_id = digest('8');
        assert!(
            catalog
                .selected_resource_placement_devices(&[missing], &[capacity])
                .unwrap_err()
                .to_string()
                .contains("lacks exact calibration coverage")
        );
    }

    #[test]
    fn class_links_survive_canonical_serialization_and_merge() {
        let (catalog, requirement) = catalog_with_exact_class();
        let decoded = VulkanPlacementCalibrationCatalog::from_json_slice(
            &catalog.to_json_bytes().unwrap(),
        )
        .unwrap();
        assert_eq!(decoded, catalog);
        assert_eq!(decoded.selected_resource_execution_class_count(), 1);

        let mut merged = VulkanPlacementCalibrationCatalog::default();
        merged.merge(&decoded).unwrap();
        merged.merge(&decoded).unwrap();
        assert_eq!(merged.selected_resource_execution_class_count(), 1);
        assert!(
            merged
                .selected_resource_placement_devices(
                    &[requirement],
                    &[VulkanPlacementSelectedResourceDeviceCapacity {
                        device_id: "logical0".to_string(),
                        identity: device("gpu0"),
                        resident_payload_capacity_bytes: 1 << 20,
                    }],
                )
                .is_ok()
        );
    }
}
