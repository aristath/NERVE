fn pair_placed_edge_endpoints(
    plans: &[VulkanPlacedEdgeIoPlan],
) -> Result<Vec<(VulkanPlacedEdgeEndpoint, VulkanPlacedEdgeEndpoint)>, VulkanError> {
    let mut incoming_by_key = BTreeMap::new();
    for plan in plans {
        for endpoint in plan
            .endpoints
            .iter()
            .filter(|endpoint| endpoint.direction == VulkanPlacedEdgeDirection::Incoming)
        {
            let key = VulkanPlacedEdgePacketKey::from_incoming_endpoint(endpoint);
            if incoming_by_key
                .insert(key.clone(), endpoint.clone())
                .is_some()
            {
                return Err(VulkanError(format!(
                    "placed execution_graph repeats incoming edge endpoint {key:?}"
                )));
            }
        }
    }

    let mut pairs = Vec::with_capacity(incoming_by_key.len());
    let mut outgoing_keys = BTreeSet::new();
    for plan in plans {
        for outgoing in plan
            .endpoints
            .iter()
            .filter(|endpoint| endpoint.direction == VulkanPlacedEdgeDirection::Outgoing)
        {
            let key = VulkanPlacedEdgePacketKey::from_outgoing_endpoint(outgoing);
            if !outgoing_keys.insert(key.clone()) {
                return Err(VulkanError(format!(
                    "placed execution_graph repeats outgoing edge endpoint {key:?}"
                )));
            }
            let incoming = incoming_by_key.remove(&key).ok_or_else(|| {
                VulkanError(format!(
                    "placed execution_graph has no incoming endpoint for edge {key:?}"
                ))
            })?;
            let outgoing_byte_capacity = outgoing.byte_capacity.ok_or_else(|| {
                VulkanError(format!("outgoing edge {key:?} has unknown byte capacity"))
            })?;
            let incoming_byte_capacity = incoming.byte_capacity.ok_or_else(|| {
                VulkanError(format!("incoming edge {key:?} has unknown byte capacity"))
            })?;
            if outgoing_byte_capacity != incoming_byte_capacity {
                return Err(VulkanError(format!(
                    "placed edge {key:?} has outgoing capacity {outgoing_byte_capacity} and incoming capacity {incoming_byte_capacity}"
                )));
            }
            pairs.push((outgoing.clone(), incoming));
        }
    }
    if let Some(key) = incoming_by_key.keys().next() {
        return Err(VulkanError(format!(
            "placed execution_graph has no outgoing endpoint for edge {key:?}"
        )));
    }
    Ok(pairs)
}

#[derive(Clone, Debug)]
struct VulkanPlacedProducedPortEdgeGroup {
    source_device_id: String,
    source_component_id: String,
    source_port_id: String,
    byte_capacity: usize,
    edges: Vec<(VulkanPlacedEdgeEndpoint, VulkanPlacedEdgeEndpoint)>,
}

fn required_vulkan_boundary_route_for_edge_group(
    group: &VulkanPlacedProducedPortEdgeGroup,
    selected_boundary_routes: &BTreeMap<usize, VulkanRuntimeMountedBoundaryRoute>,
) -> Result<Option<VulkanPlacedEdgeTransferRoute>, VulkanError> {
    let mut required = BTreeSet::new();
    for (outgoing, incoming) in &group.edges {
        let Some(selected) = selected_boundary_routes.get(&outgoing.edge_index) else {
            continue;
        };
        if selected.edge_index != outgoing.edge_index
            || selected.source_device_id != outgoing.local_device_id
            || selected.destination_device_id != incoming.local_device_id
            || selected.frame_byte_count != group.byte_capacity
        {
            return Err(VulkanError(format!(
                "selected physical boundary for edge {} disagrees with its mounted endpoints or frame bytes",
                outgoing.edge_index,
            )));
        }
        if !matches!(
            selected.route,
            VulkanPlacedEdgeTransferRoute::ExternalDeviceLocal
                | VulkanPlacedEdgeTransferRoute::DeviceLocalStaging
        ) {
            return Err(VulkanError(format!(
                "selected physical boundary for edge {} uses unsupported resident route {:?}",
                outgoing.edge_index, selected.route,
            )));
        }
        required.insert(selected.route);
    }
    if required.len() > 1 {
        return Err(VulkanError(format!(
            "produced port {}.{} selects incompatible physical boundary routes",
            group.source_component_id, group.source_port_id,
        )));
    }
    Ok(required.into_iter().next())
}

fn resolve_vulkan_produced_port_resident_route(
    group: &VulkanPlacedProducedPortEdgeGroup,
    selected_route: Option<VulkanPlacedEdgeTransferRoute>,
    distributed_route: Option<VulkanPlacedEdgeTransferRoute>,
    physical_participant_count: usize,
) -> Result<Option<VulkanPlacedEdgeTransferRoute>, VulkanError> {
    if physical_participant_count == 0 {
        return Err(VulkanError(format!(
            "produced port {}.{} has no physical participant",
            group.source_component_id, group.source_port_id,
        )));
    }
    if selected_route.is_some()
        && distributed_route.is_some()
        && selected_route != distributed_route
    {
        return Err(VulkanError(format!(
            "produced port {}.{} has incompatible selected and distributed routes",
            group.source_component_id, group.source_port_id,
        )));
    }
    if physical_participant_count == 1 {
        if selected_route.is_some() {
            return Err(VulkanError(format!(
                "selected boundary route for {}.{} has no physical peer",
                group.source_component_id, group.source_port_id,
            )));
        }
        return Ok(None);
    }
    selected_route.or(distributed_route).map(Some).ok_or_else(|| {
        VulkanError(format!(
            "produced port {}.{} crosses physical devices without an exact mounted route",
            group.source_component_id, group.source_port_id,
        ))
    })
}

fn group_placed_edge_pairs_by_produced_port(
    edge_pairs: Vec<(VulkanPlacedEdgeEndpoint, VulkanPlacedEdgeEndpoint)>,
) -> Result<Vec<VulkanPlacedProducedPortEdgeGroup>, VulkanError> {
    let mut groups = BTreeMap::<
        (String, String, String),
        VulkanPlacedProducedPortEdgeGroup,
    >::new();
    for (outgoing, incoming) in edge_pairs {
        let byte_capacity = outgoing
            .byte_capacity
            .expect("paired outgoing edge capacity was validated");
        let key = (
            outgoing.local_device_id.clone(),
            outgoing.local_component_id.clone(),
            outgoing.local_port_id.clone(),
        );
        let group = groups
            .entry(key)
            .or_insert_with(|| VulkanPlacedProducedPortEdgeGroup {
                source_device_id: outgoing.local_device_id.clone(),
                source_component_id: outgoing.local_component_id.clone(),
                source_port_id: outgoing.local_port_id.clone(),
                byte_capacity,
                edges: Vec::new(),
            });
        if group.byte_capacity != byte_capacity {
            return Err(VulkanError(format!(
                "produced port {}.{} on {:?} has incompatible outgoing capacities {} and {byte_capacity}",
                group.source_component_id,
                group.source_port_id,
                group.source_device_id,
                group.byte_capacity,
            )));
        }
        group.edges.push((outgoing, incoming));
    }
    Ok(groups.into_values().collect())
}
