impl VulkanPlacementCalibrationCatalog {
    pub fn record_region_execution(
        &mut self,
        calibration: VulkanPlacementRegionExecutionCalibration,
    ) -> Result<(), VulkanPlacementCalibrationCatalogError> {
        validate_region_execution_calibration(self, &calibration)?;
        match self.region_executions.binary_search_by(|existing| {
            existing.execution_case.cmp(&calibration.execution_case)
        }) {
            Ok(index) if self.region_executions[index] == calibration => return Ok(()),
            Ok(_) => {
                return Err(VulkanPlacementCalibrationCatalogError(
                    "region execution case has conflicting exact replay evidence".to_string(),
                ));
            }
            Err(index) => self.region_executions.insert(index, calibration),
        }
        Ok(())
    }

    pub fn region_execution(
        &self,
        execution_case: &VulkanPlacementExecutionCaseIdentity,
    ) -> Option<&VulkanPlacementRegionExecutionCalibration> {
        self.region_executions
            .binary_search_by(|calibration| calibration.execution_case.cmp(execution_case))
            .ok()
            .map(|index| &self.region_executions[index])
    }

    pub fn region_executions_for_behavior(
        &self,
        behavior: &VulkanPlacementBehaviorIdentity,
    ) -> Vec<&VulkanPlacementRegionExecutionCalibration> {
        self.region_executions
            .iter()
            .filter(|calibration| &calibration.execution_case.behavior == behavior)
            .collect()
    }
}

pub fn vulkan_placement_region_compiled_execution_signature(
    component_compiled_execution_signatures: &[String],
    boundary_byte_counts: &[usize],
) -> Result<String, VulkanPlacementCalibrationCatalogError> {
    if component_compiled_execution_signatures.len() < 2
        || component_compiled_execution_signatures
            .iter()
            .any(String::is_empty)
        || boundary_byte_counts.len() + 1 != component_compiled_execution_signatures.len()
        || boundary_byte_counts.contains(&0)
    {
        return Err(VulkanPlacementCalibrationCatalogError(
            "region execution signature requires at least two ordered components and one positive byte count per boundary"
                .to_string(),
        ));
    }
    let payload = serde_json::to_vec(&(
        "nerve.vulkan_region_compiled_execution.v1",
        component_compiled_execution_signatures,
        boundary_byte_counts,
    ))
    .map_err(|error| {
        VulkanPlacementCalibrationCatalogError(format!(
            "could not encode region execution signature: {error}",
        ))
    })?;
    Ok(format!("{:x}", Sha256::digest(payload)))
}

fn validate_region_execution_calibration(
    catalog: &VulkanPlacementCalibrationCatalog,
    calibration: &VulkanPlacementRegionExecutionCalibration,
) -> Result<(), VulkanPlacementCalibrationCatalogError> {
    let outer = catalog
        .exact_observation(&calibration.execution_case)
        .ok_or_else(|| {
            VulkanPlacementCalibrationCatalogError(
                "region execution has no exact complete transaction observation".to_string(),
            )
        })?;
    if !matches!(
        outer.execution_case.strategy,
        VulkanPlacementExecutionStrategy::SerializedRegion
            | VulkanPlacementExecutionStrategy::HybridRegion
    ) || !outer.execution_case.shards.is_empty()
        || calibration.component_cases.len() < 2
        || calibration.boundary_byte_counts.len() + 1 != calibration.component_cases.len()
        || calibration.boundary_byte_counts.contains(&0)
        || calibration
            .boundary_cases
            .windows(2)
            .any(|pair| pair[0].boundary_ordinal >= pair[1].boundary_ordinal)
    {
        return Err(VulkanPlacementCalibrationCatalogError(
            "region execution calibration is incomplete or non-canonical".to_string(),
        ));
    }
    let outer_case = &outer.execution_case;
    let component_signatures = calibration
        .component_cases
        .iter()
        .map(|case| case.behavior.compiled_execution_signature.clone())
        .collect::<Vec<_>>();
    if outer_case.behavior.compiled_execution_signature
        != vulkan_placement_region_compiled_execution_signature(
            &component_signatures,
            &calibration.boundary_byte_counts,
        )?
    {
        return Err(VulkanPlacementCalibrationCatalogError(
            "region execution behavior does not identify its exact ordered component and boundary geometry"
                .to_string(),
        ));
    }

    for component in &calibration.component_cases {
        validate_region_nested_execution_case(component)?;
        if !matches!(
            component.strategy,
            VulkanPlacementExecutionStrategy::SingleDevice
                | VulkanPlacementExecutionStrategy::TensorParallel
                | VulkanPlacementExecutionStrategy::WholeExpertParallel
                | VulkanPlacementExecutionStrategy::IntraExpertTensorParallel
                | VulkanPlacementExecutionStrategy::Hybrid
        ) || component.behavior.runtime_implementation_fingerprint
            != outer_case.behavior.runtime_implementation_fingerprint
            || component.behavior.phase != outer_case.behavior.phase
            || component.behavior.shape.activation_batch_width
                != outer_case.behavior.shape.activation_batch_width
        {
            return Err(VulkanPlacementCalibrationCatalogError(
                "region component case is not an exact phase-compatible component transaction"
                    .to_string(),
            ));
        }
    }
    if calibration.component_cases[0].input_physical_device_id
        != outer_case.input_physical_device_id
        || calibration.component_cases.last().unwrap().output_physical_device_id
            != outer_case.output_physical_device_id
        || calibration.component_cases[0].owner_physical_device_id
            != outer_case.owner_physical_device_id
    {
        return Err(VulkanPlacementCalibrationCatalogError(
            "region transaction endpoints or coordinator disagree with its component sequence"
                .to_string(),
        ));
    }

    let boundary_by_ordinal = calibration
        .boundary_cases
        .iter()
        .map(|boundary| (boundary.boundary_ordinal, &boundary.execution_case))
        .collect::<BTreeMap<_, _>>();
    if boundary_by_ordinal.len() != calibration.boundary_cases.len()
        || boundary_by_ordinal
            .keys()
            .any(|ordinal| *ordinal >= calibration.component_cases.len() - 1)
    {
        return Err(VulkanPlacementCalibrationCatalogError(
            "region execution contains a duplicate or out-of-range physical boundary".to_string(),
        ));
    }
    for boundary_ordinal in 0..calibration.component_cases.len() - 1 {
        let source = &calibration.component_cases[boundary_ordinal];
        let destination = &calibration.component_cases[boundary_ordinal + 1];
        let crosses_devices = source.output_physical_device_id
            != destination.input_physical_device_id;
        match (crosses_devices, boundary_by_ordinal.get(&boundary_ordinal)) {
            (false, None) => {}
            (true, Some(boundary)) => validate_region_boundary_execution_case(
                outer_case,
                source,
                destination,
                calibration.boundary_byte_counts[boundary_ordinal],
                boundary,
            )?,
            _ => {
                return Err(VulkanPlacementCalibrationCatalogError(format!(
                    "region boundary {boundary_ordinal} does not exactly cover its component transition",
                )));
            }
        }
    }

    let nested_cases = region_execution_cases_in_order(calibration);
    let nested_devices = nested_cases
        .iter()
        .flat_map(|case| case.devices.iter().cloned())
        .collect::<BTreeSet<_>>();
    if nested_devices != outer_case.devices.iter().cloned().collect::<BTreeSet<_>>() {
        return Err(VulkanPlacementCalibrationCatalogError(
            "region transaction device identity set differs from its exact replay cases"
                .to_string(),
        ));
    }
    let mut contract_implementations = BTreeMap::<String, String>::new();
    let mut operations = Vec::new();
    let mut transports = Vec::new();
    for case in nested_cases {
        for (contract_id, implementation_digest) in case
            .contract_ids
            .iter()
            .cloned()
            .zip(case.implementation_digests.iter().cloned())
        {
            if contract_implementations
                .insert(contract_id, implementation_digest.clone())
                .is_some_and(|existing| existing != implementation_digest)
            {
                return Err(VulkanPlacementCalibrationCatalogError(
                    "region execution repeats one contract with conflicting implementations"
                        .to_string(),
                ));
            }
        }
        operations.extend(case.operations.iter().cloned());
        transports.extend(case.transports.iter().cloned());
    }
    transports.sort();
    transports.dedup();
    if outer_case.contract_ids
        != contract_implementations.keys().cloned().collect::<Vec<_>>()
        || outer_case.implementation_digests
            != contract_implementations.values().cloned().collect::<Vec<_>>()
        || outer_case.operations != operations
        || outer_case.transports != transports
    {
        return Err(VulkanPlacementCalibrationCatalogError(
            "region transaction does not preserve the exact contracts, operations, or transports of its replay plan"
                .to_string(),
        ));
    }
    Ok(())
}

fn validate_region_nested_execution_case(
    execution_case: &VulkanPlacementExecutionCaseIdentity,
) -> Result<(), VulkanPlacementCalibrationCatalogError> {
    let devices = execution_case
        .devices
        .iter()
        .map(|device| (device.physical_device_id.clone(), 0usize))
        .collect::<BTreeMap<_, _>>();
    validate_observation(&VulkanPlacementCalibrationObservation {
        execution_case: execution_case.clone(),
        warmup_call_count: 1,
        measured_call_count: 1,
        complete_transaction: true,
        duration_ns: 1,
        useful_activation_count: 1,
        output_digest: "region-nested-output".to_string(),
        output_artifact: None,
        output_equivalence: VulkanPlacementOutputEquivalenceEvidence::BitExact,
        state_digest: "region-nested-state".to_string(),
        resident_bytes_by_physical_device: devices.clone(),
        transient_peak_bytes_by_physical_device: devices,
        host_resident_bytes: 0,
        host_transient_peak_bytes: 0,
    })
}

fn validate_region_boundary_execution_case(
    outer: &VulkanPlacementExecutionCaseIdentity,
    source: &VulkanPlacementExecutionCaseIdentity,
    destination: &VulkanPlacementExecutionCaseIdentity,
    byte_count: usize,
    boundary: &VulkanPlacementExecutionCaseIdentity,
) -> Result<(), VulkanPlacementCalibrationCatalogError> {
    validate_region_nested_execution_case(boundary)?;
    if boundary.strategy != VulkanPlacementExecutionStrategy::DirectedBoundary
        || boundary.behavior.runtime_implementation_fingerprint
            != outer.behavior.runtime_implementation_fingerprint
        || boundary.behavior.phase != outer.behavior.phase
        || boundary.behavior.shape.activation_batch_width
            != outer.behavior.shape.activation_batch_width
        || boundary.input_physical_device_id != source.output_physical_device_id
        || boundary.output_physical_device_id != destination.input_physical_device_id
        || !matches!(
            boundary.operations.as_slice(),
            [VulkanPlacementOperationGeometry::DirectedTransfer {
                byte_count: measured,
                ..
            }] if *measured == byte_count
        )
    {
        return Err(VulkanPlacementCalibrationCatalogError(
            "region physical boundary is not the exact measured directed transition"
                .to_string(),
        ));
    }
    Ok(())
}

fn region_execution_cases_in_order(
    calibration: &VulkanPlacementRegionExecutionCalibration,
) -> Vec<&VulkanPlacementExecutionCaseIdentity> {
    let boundaries = calibration
        .boundary_cases
        .iter()
        .map(|boundary| (boundary.boundary_ordinal, &boundary.execution_case))
        .collect::<BTreeMap<_, _>>();
    let mut cases = Vec::with_capacity(
        calibration.component_cases.len() + calibration.boundary_cases.len(),
    );
    for (ordinal, component) in calibration.component_cases.iter().enumerate() {
        cases.push(component);
        if let Some(boundary) = boundaries.get(&ordinal) {
            cases.push(*boundary);
        }
    }
    cases
}

#[cfg(test)]
mod region_placement_calibration_catalog_tests {
    use super::*;

    fn digest(byte: char) -> String {
        format!("sha256:{}", byte.to_string().repeat(64))
    }

    fn device(id: &str) -> VulkanPlacementDeviceExecutionIdentity {
        VulkanPlacementDeviceExecutionIdentity {
            physical_device_id: id.to_string(),
            api_version: 1,
            driver_version: 2,
        }
    }

    fn behavior(signature: String, fixture: char) -> VulkanPlacementBehaviorIdentity {
        VulkanPlacementBehaviorIdentity {
            compiled_execution_signature: signature,
            runtime_implementation_fingerprint: "runtime".to_string(),
            phase: nerve_execution_contracts::ExecutionPhase::Decode,
            shape: VulkanPlacementShapeClass {
                activation_batch_width: 1,
                input_byte_capacity: 16,
                output_byte_capacity: 16,
            },
            input_fixture_digest: digest(fixture),
        }
    }

    fn component_case(
        signature: &str,
        contract_id: &str,
        device_id: &str,
        digest_byte: char,
    ) -> VulkanPlacementExecutionCaseIdentity {
        VulkanPlacementExecutionCaseIdentity {
            behavior: behavior(signature.to_string(), digest_byte),
            contract_ids: vec![contract_id.to_string()],
            implementation_digests: vec![digest(digest_byte)],
            artifact_digest: digest(digest_byte),
            execution_graph_digest: digest(digest_byte),
            operations: vec![VulkanPlacementOperationGeometry::Dispatch {
                geometry: VulkanPlacementDispatchGeometry {
                    contract_id: contract_id.to_string(),
                    logical_extent: 8,
                    sampled_extent: 8,
                    input_width: 8,
                    workgroup_count_x: 1,
                    local_size_x: 64,
                },
            }],
            equivalence: VulkanPlacementEquivalenceIdentity::bit_exact(),
            strategy: VulkanPlacementExecutionStrategy::SingleDevice,
            devices: vec![device(device_id)],
            shards: Vec::new(),
            input_physical_device_id: device_id.to_string(),
            output_physical_device_id: device_id.to_string(),
            owner_physical_device_id: device_id.to_string(),
            transports: Vec::new(),
        }
    }

    fn boundary_case() -> VulkanPlacementExecutionCaseIdentity {
        VulkanPlacementExecutionCaseIdentity {
            behavior: behavior("boundary-signature".to_string(), 'e'),
            contract_ids: vec!["boundary".to_string()],
            implementation_digests: vec![digest('e')],
            artifact_digest: digest('e'),
            execution_graph_digest: digest('e'),
            operations: vec![VulkanPlacementOperationGeometry::DirectedTransfer {
                contract_id: "boundary".to_string(),
                byte_count: 16,
            }],
            equivalence: VulkanPlacementEquivalenceIdentity::bit_exact(),
            strategy: VulkanPlacementExecutionStrategy::DirectedBoundary,
            devices: vec![device("gpu0"), device("gpu1")],
            shards: Vec::new(),
            input_physical_device_id: "gpu0".to_string(),
            output_physical_device_id: "gpu1".to_string(),
            owner_physical_device_id: "gpu0".to_string(),
            transports: vec![VulkanPlacementTransportIdentity {
                source_physical_device_id: "gpu0".to_string(),
                destination_physical_device_id: "gpu1".to_string(),
                byte_capacity: 16,
                route: "shared_host".to_string(),
            }],
        }
    }

    fn region_fixture() -> (
        VulkanPlacementCalibrationCatalog,
        VulkanPlacementRegionExecutionCalibration,
    ) {
        let components = vec![
            component_case("component-a", "component-a", "gpu0", 'a'),
            component_case("component-b", "component-b", "gpu1", 'b'),
        ];
        let boundary = boundary_case();
        let signature = vulkan_placement_region_compiled_execution_signature(
            &components
                .iter()
                .map(|case| case.behavior.compiled_execution_signature.clone())
                .collect::<Vec<_>>(),
            &[16],
        )
        .unwrap();
        let region_behavior = behavior(signature, 'f');
        let execution_case = VulkanPlacementExecutionCaseIdentity {
            behavior: region_behavior.clone(),
            contract_ids: vec![
                "boundary".to_string(),
                "component-a".to_string(),
                "component-b".to_string(),
            ],
            implementation_digests: vec![digest('e'), digest('a'), digest('b')],
            artifact_digest: digest('f'),
            execution_graph_digest: digest('f'),
            operations: vec![
                components[0].operations[0].clone(),
                boundary.operations[0].clone(),
                components[1].operations[0].clone(),
            ],
            equivalence: VulkanPlacementEquivalenceIdentity::bit_exact(),
            strategy: VulkanPlacementExecutionStrategy::SerializedRegion,
            devices: vec![device("gpu0"), device("gpu1")],
            shards: Vec::new(),
            input_physical_device_id: "gpu0".to_string(),
            output_physical_device_id: "gpu1".to_string(),
            owner_physical_device_id: "gpu0".to_string(),
            transports: boundary.transports.clone(),
        };
        let observation = VulkanPlacementCalibrationObservation {
            execution_case: execution_case.clone(),
            warmup_call_count: 1,
            measured_call_count: 1,
            complete_transaction: true,
            duration_ns: 10,
            useful_activation_count: 1,
            output_digest: "output".to_string(),
            output_artifact: None,
            output_equivalence: VulkanPlacementOutputEquivalenceEvidence::BitExact,
            state_digest: "state".to_string(),
            resident_bytes_by_physical_device: BTreeMap::from([
                ("gpu0".to_string(), 10),
                ("gpu1".to_string(), 10),
            ]),
            transient_peak_bytes_by_physical_device: BTreeMap::from([
                ("gpu0".to_string(), 2),
                ("gpu1".to_string(), 2),
            ]),
            host_resident_bytes: 0,
            host_transient_peak_bytes: 0,
        };
        let mut catalog = VulkanPlacementCalibrationCatalog::default();
        catalog
            .record_reference(VulkanPlacementCanonicalReference {
                behavior: region_behavior,
                output_digest: "output".to_string(),
                output_artifact: None,
                state_digest: "state".to_string(),
            })
            .unwrap();
        catalog.record_observation(observation).unwrap();
        (
            catalog,
            VulkanPlacementRegionExecutionCalibration {
                execution_case,
                boundary_byte_counts: vec![16],
                component_cases: components,
                boundary_cases: vec![VulkanPlacementRegionBoundaryExecutionCase {
                    boundary_ordinal: 0,
                    execution_case: boundary,
                }],
            },
        )
    }

    #[test]
    fn exact_region_replay_survives_catalog_serialization_and_merge() {
        let (mut catalog, calibration) = region_fixture();
        catalog.record_region_execution(calibration.clone()).unwrap();
        assert_eq!(catalog.region_execution_count(), 1);
        assert_eq!(
            catalog.region_execution(&calibration.execution_case),
            Some(&calibration),
        );

        let decoded = VulkanPlacementCalibrationCatalog::from_json_slice(
            &catalog.to_json_bytes().unwrap(),
        )
        .unwrap();
        assert_eq!(decoded, catalog);
        let mut merged = VulkanPlacementCalibrationCatalog::default();
        merged.merge(&catalog).unwrap();
        assert_eq!(merged, catalog);
    }

    #[test]
    fn region_replay_rejects_missing_boundaries_and_summed_component_costs() {
        let (catalog, calibration) = region_fixture();
        let mut missing_boundary = calibration.clone();
        missing_boundary.boundary_cases.clear();
        assert!(validate_region_execution_calibration(&catalog, &missing_boundary)
            .unwrap_err()
            .to_string()
            .contains("does not exactly cover"));

        let (mut catalog, _) = region_fixture();
        let mut summed_components = calibration;
        summed_components.execution_case.operations.remove(1);
        catalog.observations[0].execution_case = summed_components.execution_case.clone();
        assert!(validate_region_execution_calibration(&catalog, &summed_components)
            .unwrap_err()
            .to_string()
            .contains("contracts, operations, or transports"));
    }
}
