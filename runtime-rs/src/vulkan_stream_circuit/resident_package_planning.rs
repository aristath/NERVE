fn resolve_resident_model_package_path(manifest_dir: &Path, path: &str) -> PathBuf {
    let path = Path::new(path);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        manifest_dir.join(path)
    }
}

#[cfg(test)]
fn plan_resident_package_single_device_stream_circuit(
    device_id: &str,
    placement_spec: &StreamCircuitPlacementSpec,
    circuit_graph: &VulkanResidentPackageCircuitGraph,
    manifest_dir: &Path,
    tensor_index_path: &Path,
    activation_element_bytes: Option<usize>,
) -> Result<
    (
        TensorIndex,
        StreamCircuitResourcePlan,
        VulkanPlacedStreamCircuitPlan,
    ),
    VulkanResidentTokenModelPackageError,
> {
    let (tensor_index, resource_plan, placement_plan, placed_plan) =
        plan_resident_package_placed_stream_circuit(
            device_id,
            placement_spec,
            circuit_graph,
            manifest_dir,
            tensor_index_path,
            activation_element_bytes,
        )?;
    validate_single_device_resident_package_placement(device_id, &placement_plan)?;
    Ok((tensor_index, resource_plan, placed_plan))
}

#[cfg(test)]
fn plan_resident_package_placed_stream_circuit(
    device_id: &str,
    placement_spec: &StreamCircuitPlacementSpec,
    circuit_graph: &VulkanResidentPackageCircuitGraph,
    manifest_dir: &Path,
    tensor_index_path: &Path,
    activation_element_bytes: Option<usize>,
) -> Result<
    (
        TensorIndex,
        StreamCircuitResourcePlan,
        StreamCircuitPlacementPlan,
        VulkanPlacedStreamCircuitPlan,
    ),
    VulkanResidentTokenModelPackageError,
> {
    let tensor_index = TensorIndex::from_package_json_file(tensor_index_path).map_err(|error| {
        VulkanResidentTokenModelPackageError::new(format!(
            "failed to load tensor index {:?}: {error}",
            tensor_index_path
        ))
    })?;
    let (resource_plan, placement_plan, placed_plan) =
        plan_resident_package_placed_stream_circuit_with_tensor_index(
            device_id,
            placement_spec,
            circuit_graph,
            manifest_dir,
            &tensor_index,
            activation_element_bytes,
        )?;
    Ok((tensor_index, resource_plan, placement_plan, placed_plan))
}

fn plan_resident_package_placed_stream_circuit_with_tensor_index(
    device_id: &str,
    placement_spec: &StreamCircuitPlacementSpec,
    circuit_graph: &VulkanResidentPackageCircuitGraph,
    manifest_dir: &Path,
    tensor_index: &TensorIndex,
    activation_element_bytes: Option<usize>,
) -> Result<
    (
        StreamCircuitResourcePlan,
        StreamCircuitPlacementPlan,
        VulkanPlacedStreamCircuitPlan,
    ),
    VulkanResidentTokenModelPackageError,
> {
    let basis = prepare_resident_package_planning_basis(circuit_graph, manifest_dir, tensor_index)?;
    let (placement_plan, placed_plan) = plan_resident_package_from_planning_basis(
        device_id,
        placement_spec,
        circuit_graph,
        tensor_index,
        activation_element_bytes,
        &basis,
    )?;
    Ok((basis.resource_plan, placement_plan, placed_plan))
}

/// Placement-invariant execution and resource metadata. Large sparse models
/// can contain hundreds of megabytes of graph records; capacity search must
/// resolve those records once, not once per device subset and owner.
struct VulkanResidentPackagePlanningBasis {
    graph: ResolvedLoweredExecutionGraph,
    execution_plan: StreamCircuitExecutionPlan,
    resource_plan: StreamCircuitResourcePlan,
}

#[cfg(test)]
static RESIDENT_PACKAGE_PLANNING_BASIS_PREPARATIONS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
fn reset_resident_package_planning_basis_preparation_count() {
    RESIDENT_PACKAGE_PLANNING_BASIS_PREPARATIONS
        .store(0, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(test)]
fn resident_package_planning_basis_preparation_count() -> usize {
    RESIDENT_PACKAGE_PLANNING_BASIS_PREPARATIONS.load(std::sync::atomic::Ordering::Relaxed)
}

fn prepare_resident_package_planning_basis(
    circuit_graph: &VulkanResidentPackageCircuitGraph,
    manifest_dir: &Path,
    tensor_index: &TensorIndex,
) -> Result<VulkanResidentPackagePlanningBasis, VulkanResidentTokenModelPackageError> {
    #[cfg(test)]
    RESIDENT_PACKAGE_PLANNING_BASIS_PREPARATIONS
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let graph = circuit_graph.to_signal_processor_graph(manifest_dir.to_path_buf())?;
    let execution_plan = StreamCircuitExecutionPlan::from_graph_with_tensor_index(
        &graph,
        tensor_index,
    )
    .map_err(|error| {
        VulkanResidentTokenModelPackageError::new(format!(
            "failed to create stream execution plan: {error}"
        ))
    })?;
    let resource_plan = StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan)
        .map_err(|error| {
            VulkanResidentTokenModelPackageError::new(format!(
                "failed to create stream resource plan: {error}"
            ))
        })?;
    Ok(VulkanResidentPackagePlanningBasis {
        graph,
        execution_plan,
        resource_plan,
    })
}

fn plan_resident_package_from_planning_basis(
    device_id: &str,
    placement_spec: &StreamCircuitPlacementSpec,
    circuit_graph: &VulkanResidentPackageCircuitGraph,
    tensor_index: &TensorIndex,
    activation_element_bytes: Option<usize>,
    basis: &VulkanResidentPackagePlanningBasis,
) -> Result<
    (StreamCircuitPlacementPlan, VulkanPlacedStreamCircuitPlan),
    VulkanResidentTokenModelPackageError,
> {
    let placement_spec = circuit_graph.signal_processor_placement(placement_spec);
    let placement_plan = basis.graph.placement_plan(&placement_spec).map_err(|error| {
        VulkanResidentTokenModelPackageError::new(format!(
            "failed to create placement plan for {device_id:?}: {error}"
        ))
    })?;
    let resident = VulkanPlacedStreamCircuitResidentPlan::from_resource_plan_for_device(
        &basis.resource_plan,
        &placement_plan,
        device_id,
        Some(tensor_index),
        activation_element_bytes,
    )
    .map_err(|error| {
        VulkanResidentTokenModelPackageError::new(format!(
            "failed to create Vulkan resident plan for {device_id:?}: {error}"
        ))
    })?;
    let placed_plan = VulkanPlacedStreamCircuitPlan::from_plans(
        &basis.execution_plan,
        &basis.resource_plan,
        resident,
    )
    .map_err(|error| {
        VulkanResidentTokenModelPackageError::new(format!(
            "failed to create Vulkan placed stream circuit plan: {error}"
        ))
    })?;
    Ok((placement_plan, placed_plan))
}

fn validate_single_device_resident_package_placement(
    device_id: &str,
    placement_plan: &StreamCircuitPlacementPlan,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    let remote_components = placement_plan
        .components
        .iter()
        .filter(|component| component.device_id != device_id)
        .map(|component| format!("{}@{}", component.component_id, component.device_id))
        .collect::<Vec<_>>();
    if !remote_components.is_empty() {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "single-device resident package for {device_id:?} cannot host remote components: {}",
            remote_components.join(", ")
        )));
    }
    if placement_plan.cross_device_edge_count != 0 {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "single-device resident package for {device_id:?} cannot host {} cross-device edges",
            placement_plan.cross_device_edge_count
        )));
    }
    Ok(())
}
