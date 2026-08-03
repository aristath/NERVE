#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum VulkanComponentBatchSignalKey {
    Activation { component_id: String, signal_id: String },
    ProducedPort { component_id: String, port_id: String },
    ModelInput(String),
    ModelOutput(String),
    LocalEdge(usize),
    IncomingEdge(usize),
    OutgoingEdge(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VulkanComponentBatchCompletionMode {
    Blocking,
    Deferred,
}

struct VulkanComponentBatchSignalBuffer {
    frame_byte_capacity: usize,
    buffer: Arc<VulkanResidentBuffer>,
    shared_device_buffers: BTreeMap<String, Arc<VulkanResidentBuffer>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanComponentBatchSignalLifetime {
    key: VulkanComponentBatchSignalKey,
    frame_byte_capacity: usize,
    host_visible: bool,
    first_dispatch: usize,
    last_dispatch: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanComponentBatchSignalBufferPlan {
    frame_byte_capacity: usize,
    host_visible: bool,
    last_dispatch: usize,
}

fn allocate_component_batch_signal_lifetimes(
    mut lifetimes: Vec<VulkanComponentBatchSignalLifetime>,
) -> (
    BTreeMap<VulkanComponentBatchSignalKey, usize>,
    Vec<VulkanComponentBatchSignalBufferPlan>,
) {
    lifetimes.sort_by(|left, right| {
        left.first_dispatch
            .cmp(&right.first_dispatch)
            .then_with(|| left.last_dispatch.cmp(&right.last_dispatch))
            .then_with(|| left.key.cmp(&right.key))
    });
    let mut signal_buffer_indices = BTreeMap::new();
    let mut buffers = Vec::<VulkanComponentBatchSignalBufferPlan>::new();
    for lifetime in lifetimes {
        let buffer_index = buffers
            .iter()
            .position(|buffer| {
                buffer.frame_byte_capacity == lifetime.frame_byte_capacity
                    && buffer.host_visible == lifetime.host_visible
                    && buffer.last_dispatch < lifetime.first_dispatch
            })
            .unwrap_or_else(|| {
                buffers.push(VulkanComponentBatchSignalBufferPlan {
                    frame_byte_capacity: lifetime.frame_byte_capacity,
                    host_visible: lifetime.host_visible,
                    last_dispatch: lifetime.last_dispatch,
                });
                buffers.len() - 1
            });
        buffers[buffer_index].last_dispatch = lifetime.last_dispatch;
        signal_buffer_indices.insert(lifetime.key, buffer_index);
    }
    (signal_buffer_indices, buffers)
}

fn component_batch_signal_buffer_plan(
    mounted: &VulkanMountedPlacedStreamCircuit,
    dispatches: &[VulkanMountedPlacedBoundDispatch],
) -> Result<
    (
        BTreeMap<VulkanComponentBatchSignalKey, usize>,
        Vec<VulkanComponentBatchSignalBufferPlan>,
    ),
    VulkanResidentInProcessPlacedRuntimeError,
> {
    component_batch_signal_buffer_plan_for_dispatches_retaining(
        mounted,
        dispatches.iter(),
        &BTreeSet::new(),
    )
}

fn component_batch_signal_buffer_plan_for_dispatches_retaining<'a>(
    mounted: &VulkanMountedPlacedStreamCircuit,
    dispatches: impl IntoIterator<Item = &'a VulkanMountedPlacedBoundDispatch>,
    retained_signal_keys: &BTreeSet<VulkanComponentBatchSignalKey>,
) -> Result<
    (
        BTreeMap<VulkanComponentBatchSignalKey, usize>,
        Vec<VulkanComponentBatchSignalBufferPlan>,
    ),
    VulkanResidentInProcessPlacedRuntimeError,
> {
    let dispatches = dispatches.into_iter().collect::<Vec<_>>();
    let dispatch_count = dispatches.len();
    let mut lifetimes = BTreeMap::<VulkanComponentBatchSignalKey, (usize, bool, usize, usize)>::new();
    for (dispatch_index, dispatch) in dispatches.iter().enumerate() {
        for descriptor in &dispatch.descriptors {
            let Some((key, frame_byte_capacity)) =
                component_batch_signal_target_with_mounted(mounted, descriptor)?
            else {
                continue;
            };
            // Component-batch activations remain device-local. A cross-device
            // edge uses a dedicated transfer route rather than forcing every
            // producer and consumer dispatch to operate directly on system
            // memory.
            let host_visible = false;
            let external_source = matches!(
                key,
                VulkanComponentBatchSignalKey::ModelInput(_)
                    | VulkanComponentBatchSignalKey::IncomingEdge(_)
            );
            let external_sink = component_batch_signal_is_external_sink(mounted, &key);
            let first_dispatch = if external_source { 0 } else { dispatch_index };
            let last_dispatch = if external_sink {
                dispatch_count
            } else {
                dispatch_index
            };
            merge_component_batch_signal_lifetime(
                &mut lifetimes,
                key,
                frame_byte_capacity,
                host_visible,
                first_dispatch,
                last_dispatch,
            )?;
        }
    }
    let mut lifetimes = lifetimes
        .into_iter()
        .map(
            |(key, (frame_byte_capacity, host_visible, first_dispatch, last_dispatch))| {
                VulkanComponentBatchSignalLifetime {
                    key,
                    frame_byte_capacity,
                    host_visible,
                    first_dispatch,
                    last_dispatch,
                }
            },
        )
        .collect::<Vec<_>>();
    let canonical_retained_signal_keys = retained_signal_keys
        .iter()
        .map(|key| canonical_component_batch_signal_key(mounted, key))
        .collect::<Result<BTreeSet<_>, _>>()?;
    retain_component_batch_signal_lifetimes(
        &mut lifetimes,
        &canonical_retained_signal_keys,
        dispatch_count,
    )
    .map_err(|error| {
        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(error)
    })?;
    let (mut signal_buffer_indices, buffer_plan) =
        allocate_component_batch_signal_lifetimes(lifetimes);
    install_component_batch_edge_aliases(mounted, &mut signal_buffer_indices)?;
    Ok((signal_buffer_indices, buffer_plan))
}

fn merge_component_batch_signal_lifetime(
    lifetimes: &mut BTreeMap<VulkanComponentBatchSignalKey, (usize, bool, usize, usize)>,
    key: VulkanComponentBatchSignalKey,
    frame_byte_capacity: usize,
    host_visible: bool,
    first_dispatch: usize,
    last_dispatch: usize,
) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
    match lifetimes.entry(key.clone()) {
        std::collections::btree_map::Entry::Vacant(entry) => {
            entry.insert((
                frame_byte_capacity,
                host_visible,
                first_dispatch,
                last_dispatch,
            ));
        }
        std::collections::btree_map::Entry::Occupied(mut entry) => {
            let (existing_capacity, existing_visibility, first, last) = entry.get_mut();
            if *existing_capacity != frame_byte_capacity
                || *existing_visibility != host_visible
            {
                return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                    VulkanError(format!(
                        "component batch signal {key:?} has incompatible physical requirements"
                    )),
                ));
            }
            *first = (*first).min(first_dispatch);
            *last = (*last).max(last_dispatch);
        }
    }
    Ok(())
}

fn produced_port_signal_key(
    component_id: &str,
    port_id: &str,
) -> VulkanComponentBatchSignalKey {
    VulkanComponentBatchSignalKey::ProducedPort {
        component_id: component_id.to_string(),
        port_id: port_id.to_string(),
    }
}

fn canonical_component_batch_signal_key(
    mounted: &VulkanMountedPlacedStreamCircuit,
    key: &VulkanComponentBatchSignalKey,
) -> Result<VulkanComponentBatchSignalKey, VulkanResidentInProcessPlacedRuntimeError> {
    let resident_plan = &mounted.placed_plan.placed_resident_plan;
    match key {
        VulkanComponentBatchSignalKey::LocalEdge(edge_index) => {
            let edge = resident_plan
                .local_edges
                .iter()
                .find(|edge| edge.edge_index == *edge_index)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                        "component batch local edge alias references absent edge {edge_index}"
                    )))
                })?;
            Ok(produced_port_signal_key(
                &edge.source_component_id,
                &edge.source_port_id,
            ))
        }
        VulkanComponentBatchSignalKey::OutgoingEdge(edge_index) => {
            let edge = resident_plan
                .outgoing_edges
                .iter()
                .find(|edge| edge.edge_index == *edge_index)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                        "component batch outgoing edge alias references absent edge {edge_index}"
                    )))
                })?;
            Ok(produced_port_signal_key(
                &edge.source_component_id,
                &edge.source_port_id,
            ))
        }
        _ => Ok(key.clone()),
    }
}

fn component_batch_signal_is_external_sink(
    mounted: &VulkanMountedPlacedStreamCircuit,
    key: &VulkanComponentBatchSignalKey,
) -> bool {
    match key {
        VulkanComponentBatchSignalKey::ModelOutput(_) => true,
        VulkanComponentBatchSignalKey::ProducedPort {
            component_id,
            port_id,
        } => mounted
            .placed_plan
            .placed_resident_plan
            .outgoing_edges
            .iter()
            .any(|edge| {
                edge.source_component_id == *component_id
                    && edge.source_port_id == *port_id
            }),
        _ => false,
    }
}

fn install_component_batch_edge_aliases(
    mounted: &VulkanMountedPlacedStreamCircuit,
    signal_buffer_indices: &mut BTreeMap<VulkanComponentBatchSignalKey, usize>,
) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
    let mut aliases = Vec::new();
    for edge in &mounted.placed_plan.placed_resident_plan.local_edges {
        aliases.push((
            VulkanComponentBatchSignalKey::LocalEdge(edge.edge_index),
            produced_port_signal_key(&edge.source_component_id, &edge.source_port_id),
        ));
    }
    for edge in &mounted.placed_plan.placed_resident_plan.outgoing_edges {
        aliases.push((
            VulkanComponentBatchSignalKey::OutgoingEdge(edge.edge_index),
            produced_port_signal_key(&edge.source_component_id, &edge.source_port_id),
        ));
    }
    for (alias, canonical) in aliases {
        let Some(buffer_index) = signal_buffer_indices.get(&canonical).copied() else {
            continue;
        };
        if let Some(existing) = signal_buffer_indices.insert(alias.clone(), buffer_index)
            && existing != buffer_index
        {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "component batch signal alias {alias:?} maps to incompatible physical buffers {existing} and {buffer_index}"
                )),
            ));
        }
    }
    Ok(())
}

fn retain_component_batch_signal_lifetimes(
    lifetimes: &mut [VulkanComponentBatchSignalLifetime],
    retained_signal_keys: &BTreeSet<VulkanComponentBatchSignalKey>,
    retention_boundary: usize,
) -> Result<(), VulkanError> {
    for key in retained_signal_keys {
        let lifetime = lifetimes
            .iter_mut()
            .find(|lifetime| &lifetime.key == key)
            .ok_or_else(|| {
                VulkanError(format!(
                    "retained component batch signal {key:?} has no physical lifetime"
                ))
            })?;
        lifetime.last_dispatch = lifetime.last_dispatch.max(retention_boundary);
    }
    Ok(())
}
