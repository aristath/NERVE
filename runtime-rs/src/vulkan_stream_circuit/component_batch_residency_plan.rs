#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum VulkanComponentBatchResidentAllocationKind {
    SignalBuffer { buffer_index: usize },
    StreamControl { lane: usize },
    RuntimeTokenIds,
    BatchControl {
        payload: VulkanResidentComponentBatchControlPayload,
    },
    CausalStateSnapshotDummy,
    CausalStateSnapshot {
        component_id: String,
        state_id: String,
    },
    DemandPipelinePredicate,
    DemandMissQueue,
    DemandGateConfiguration {
        checkpoint_id: String,
        selector_id: String,
    },
    DemandGateResourceGroups {
        checkpoint_id: String,
        selector_id: String,
    },
    DemandGateAddressSlots {
        checkpoint_id: String,
        selector_id: String,
    },
    DemandGateResolvedAddresses {
        checkpoint_id: String,
        selector_id: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct VulkanComponentBatchResidentAllocation {
    kind: VulkanComponentBatchResidentAllocationKind,
    byte_capacity: usize,
    host_visible: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanComponentBatchResidentAllocationPlan {
    signal_buffer_indices: BTreeMap<VulkanComponentBatchSignalKey, usize>,
    signal_buffer_plan: Vec<VulkanComponentBatchSignalBufferPlan>,
    allocations: Vec<VulkanComponentBatchResidentAllocation>,
    total_byte_capacity: usize,
}

#[derive(Clone, Debug)]
struct VulkanComponentBatchDemandGateGeometry {
    checkpoint_id: String,
    selector_id: String,
    selection_count_per_activation: usize,
    selection_index_shift: u32,
    selection_index_mask: u32,
    address_mapping: VulkanCompiledSelectorAddressMapping,
}

#[derive(Clone, Copy)]
struct VulkanComponentBatchDemandResidencyPlanContext<'a> {
    schedule: &'a VulkanPhysicalResidencySchedule,
    contract: &'a CompiledResourceResidencyContract,
    layout: &'a VulkanCompiledResourceAddressLayout,
}

impl VulkanComponentBatchResidentAllocationPlan {
    #[allow(clippy::too_many_arguments)]
    fn for_single_device(
        placed_plan: &VulkanPlacedStreamCircuitPlan,
        prepared_plan: &VulkanPreparedDispatchPlan,
        batch_kernels: &[VulkanResidentComponentBatchKernelArtifact],
        lane_capacity: usize,
        execution_mode: VulkanComponentBatchExecutionMode,
        execution_scope: &VulkanComponentBatchExecutionScope,
        retained_signal_keys: &BTreeSet<VulkanComponentBatchSignalKey>,
        capture_causal_state_snapshots: bool,
        demand_residency: Option<VulkanComponentBatchDemandResidencyPlanContext<'_>>,
    ) -> Result<Self, VulkanError> {
        if lane_capacity == 0 {
            return Err(VulkanError(
                "component batch residency lane capacity is zero".to_string(),
            ));
        }
        execution_scope.validate_dispatch_ids(
            prepared_plan
                .dispatches
                .iter()
                .map(|dispatch| (dispatch.component_id.as_str(), dispatch.node_id.as_str())),
        )?;
        let selected_dispatches = prepared_plan
            .dispatches
            .iter()
            .filter(|dispatch| {
                execution_scope.includes_dispatch(&dispatch.component_id, &dispatch.node_id)
            })
            .collect::<Vec<_>>();
        if selected_dispatches.is_empty() {
            return Err(VulkanError(
                "component batch residency scope selects no dispatches".to_string(),
            ));
        }
        let (signal_buffer_indices, signal_buffer_plan) =
            component_batch_signal_buffer_plan_from_prepared_dispatches_retaining(
                placed_plan,
                selected_dispatches.iter().copied(),
                retained_signal_keys,
            )?;
        let mut allocations = Vec::new();
        for (buffer_index, signal) in signal_buffer_plan.iter().enumerate() {
            allocations.push(VulkanComponentBatchResidentAllocation {
                kind: VulkanComponentBatchResidentAllocationKind::SignalBuffer { buffer_index },
                byte_capacity: signal
                    .frame_byte_capacity
                    .checked_mul(lane_capacity)
                    .ok_or_else(|| {
                        VulkanError("component batch signal capacity overflowed".to_string())
                    })?,
                host_visible: signal.host_visible,
            });
        }
        for lane in 0..lane_capacity {
            allocations.push(VulkanComponentBatchResidentAllocation {
                kind: VulkanComponentBatchResidentAllocationKind::StreamControl { lane },
                byte_capacity: VULKAN_STREAM_CONTROL_BYTE_CAPACITY,
                host_visible: true,
            });
        }
        allocations.push(VulkanComponentBatchResidentAllocation {
            kind: VulkanComponentBatchResidentAllocationKind::RuntimeTokenIds,
            byte_capacity: lane_capacity.checked_mul(size_of::<u32>()).ok_or_else(|| {
                VulkanError("component batch runtime token-id capacity overflowed".to_string())
            })?,
            host_visible: true,
        });
        for payload in component_batch_control_payloads() {
            allocations.push(VulkanComponentBatchResidentAllocation {
                kind: VulkanComponentBatchResidentAllocationKind::BatchControl { payload },
                byte_capacity: payload.byte_count() as usize,
                host_visible: true,
            });
        }
        allocations.extend(component_batch_causal_snapshot_allocations(
            placed_plan,
            batch_kernels,
            &selected_dispatches,
            lane_capacity,
            execution_mode,
            capture_causal_state_snapshots,
        )?);
        if let Some(demand_residency) = demand_residency {
            let gate_geometries = component_batch_single_device_demand_gate_geometries(
                prepared_plan,
                execution_scope,
                demand_residency,
            )?;
            allocations.extend(component_batch_demand_segment_allocations(
                &gate_geometries,
                lane_capacity,
                true,
            )?);
        }
        allocations.sort();
        if allocations
            .windows(2)
            .any(|window| window[0].kind == window[1].kind)
        {
            return Err(VulkanError(
                "component batch residency repeats a physical allocation identity".to_string(),
            ));
        }
        let total_byte_capacity = allocations.iter().try_fold(0usize, |total, allocation| {
            if allocation.byte_capacity == 0 {
                return Err(VulkanError(
                    "component batch residency contains an empty allocation".to_string(),
                ));
            }
            total.checked_add(allocation.byte_capacity).ok_or_else(|| {
                VulkanError("component batch residency capacity overflowed".to_string())
            })
        })?;
        Ok(Self {
            signal_buffer_indices,
            signal_buffer_plan,
            allocations,
            total_byte_capacity,
        })
    }
}

fn component_batch_single_device_demand_gate_geometries(
    prepared_plan: &VulkanPreparedDispatchPlan,
    execution_scope: &VulkanComponentBatchExecutionScope,
    demand_residency: VulkanComponentBatchDemandResidencyPlanContext<'_>,
) -> Result<Vec<VulkanComponentBatchDemandGateGeometry>, VulkanError> {
    let selected_dispatch_indices = prepared_plan
        .dispatches
        .iter()
        .filter(|dispatch| {
            execution_scope.includes_dispatch(&dispatch.component_id, &dispatch.node_id)
        })
        .map(|dispatch| dispatch.dispatch_index)
        .collect::<BTreeSet<_>>();
    let mut geometries = Vec::new();
    for checkpoint in &demand_residency.schedule.checkpoints {
        if !selected_dispatch_indices.contains(&checkpoint.selection_dispatch_index) {
            continue;
        }
        let selected_computation_is_complete = checkpoint
            .selected_computation_dispatch_indices
            .iter()
            .copied()
            .chain(checkpoint.selected_result_continuation_dispatch_index)
            .all(|dispatch_index| selected_dispatch_indices.contains(&dispatch_index));
        if !selected_computation_is_complete {
            return Err(VulkanError(format!(
                "component batch demand checkpoint {:?} selects inside the execution scope but its selected computation crosses the scope boundary",
                checkpoint.id,
            )));
        }
        for selector_id in &checkpoint.selector_ids {
            let selector = demand_residency
                .contract
                .selectors
                .iter()
                .find(|selector| selector.id == *selector_id)
                .ok_or_else(|| {
                    VulkanError(format!(
                        "component batch demand checkpoint {:?} references unknown selector {selector_id:?}",
                        checkpoint.id,
                    ))
                })?;
            if selector.execution_scope != demand_residency.schedule.execution_scope
                || selector.component_id != checkpoint.component_id
            {
                return Err(VulkanError(format!(
                    "component batch demand selector {selector_id:?} does not belong to checkpoint {:?}",
                    checkpoint.id,
                )));
            }
            let address_mapping = demand_residency
                .layout
                .selectors
                .iter()
                .find(|layout| layout.selector_id == *selector_id)
                .map(|layout| layout.mapping.clone())
                .ok_or_else(|| {
                    VulkanError(format!(
                        "component batch demand selector {selector_id:?} has no stable-address layout",
                    ))
                })?;
            geometries.push(VulkanComponentBatchDemandGateGeometry {
                checkpoint_id: checkpoint.id.clone(),
                selector_id: selector.id.clone(),
                selection_count_per_activation: selector.encoding.selection_count_per_activation,
                selection_index_shift: selector.encoding.index_shift,
                selection_index_mask: selector.encoding.index_mask,
                address_mapping,
            });
        }
    }
    Ok(geometries)
}

fn component_batch_demand_segment_allocations(
    gate_geometries: &[VulkanComponentBatchDemandGateGeometry],
    lane_capacity: usize,
    owns_continuation_predicate: bool,
) -> Result<Vec<VulkanComponentBatchResidentAllocation>, VulkanError> {
    if gate_geometries.is_empty() {
        return Ok(Vec::new());
    }
    let mut allocations = Vec::new();
    if owns_continuation_predicate {
        allocations.push(VulkanComponentBatchResidentAllocation {
            kind: VulkanComponentBatchResidentAllocationKind::DemandPipelinePredicate,
            byte_capacity: size_of::<u32>(),
            host_visible: false,
        });
    }
    let missing_capacity = gate_geometries
        .iter()
        .map(|geometry| {
            geometry
                .selection_count_per_activation
                .checked_mul(lane_capacity)
                .ok_or_else(|| {
                    VulkanError("component batch demand queue capacity overflowed".to_string())
                })
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .expect("component batch demand geometries are non-empty");
    allocations.push(VulkanComponentBatchResidentAllocation {
        kind: VulkanComponentBatchResidentAllocationKind::DemandMissQueue,
        byte_capacity: VulkanGpuResidencyMissQueue::device_bytes_for_capacity(missing_capacity)?
            .byte_count,
        host_visible: true,
    });
    for geometry in gate_geometries {
        let selection_count = geometry
            .selection_count_per_activation
            .checked_mul(lane_capacity)
            .ok_or_else(|| {
                VulkanError("component batch demand gate capacity overflowed".to_string())
            })?;
        let private = VulkanGpuResidencyGateConfig {
            maximum_selection_count: selection_count,
            selection_count_per_lane: geometry.selection_count_per_activation,
            selection_lane_stride_words: geometry.selection_count_per_activation,
            selection_index_shift: geometry.selection_index_shift,
            selection_index_mask: geometry.selection_index_mask,
            address_mapping: vulkan_gpu_residency_address_mapping_from_compiled(
                &geometry.address_mapping,
            ),
            owned_resource_indices: None,
        }
        .private_device_bytes()?;
        for (kind, byte_capacity) in [
            (
                VulkanComponentBatchResidentAllocationKind::DemandGateConfiguration {
                    checkpoint_id: geometry.checkpoint_id.clone(),
                    selector_id: geometry.selector_id.clone(),
                },
                private.configuration_bytes,
            ),
            (
                VulkanComponentBatchResidentAllocationKind::DemandGateResourceGroups {
                    checkpoint_id: geometry.checkpoint_id.clone(),
                    selector_id: geometry.selector_id.clone(),
                },
                private.resource_group_record_bytes,
            ),
            (
                VulkanComponentBatchResidentAllocationKind::DemandGateAddressSlots {
                    checkpoint_id: geometry.checkpoint_id.clone(),
                    selector_id: geometry.selector_id.clone(),
                },
                private.resource_address_slot_bytes,
            ),
            (
                VulkanComponentBatchResidentAllocationKind::DemandGateResolvedAddresses {
                    checkpoint_id: geometry.checkpoint_id.clone(),
                    selector_id: geometry.selector_id.clone(),
                },
                private.resolved_address_bytes,
            ),
        ] {
            allocations.push(VulkanComponentBatchResidentAllocation {
                kind,
                byte_capacity,
                host_visible: false,
            });
        }
    }
    Ok(allocations)
}

fn vulkan_gpu_residency_address_mapping_from_compiled(
    mapping: &VulkanCompiledSelectorAddressMapping,
) -> VulkanGpuResidencyAddressMapping {
    match mapping {
        VulkanCompiledSelectorAddressMapping::GroupTable {
            resource_address_slots,
            resource_address_slot_offsets,
        } => VulkanGpuResidencyAddressMapping::GroupTable {
            resource_address_slots: resource_address_slots.clone(),
            resource_address_slot_offsets: resource_address_slot_offsets.clone(),
        },
        VulkanCompiledSelectorAddressMapping::PartitionTemplate {
            member_slot_bases,
            resource_count,
            ..
        } => VulkanGpuResidencyAddressMapping::Partitioned {
            member_slot_bases: member_slot_bases.clone(),
            resource_count: *resource_count,
        },
    }
}

fn component_batch_control_payloads(
) -> [VulkanResidentComponentBatchControlPayload; 6] {
    [
        VulkanResidentComponentBatchControlPayload::Width,
        VulkanResidentComponentBatchControlPayload::WidthStateSnapshots,
        VulkanResidentComponentBatchControlPayload::WidthExpertStart,
        VulkanResidentComponentBatchControlPayload::WidthExpertRangeIndirect,
        VulkanResidentComponentBatchControlPayload::Temporal,
        VulkanResidentComponentBatchControlPayload::TemporalStateSnapshots,
    ]
}

fn component_batch_causal_snapshot_allocations(
    placed_plan: &VulkanPlacedStreamCircuitPlan,
    batch_kernels: &[VulkanResidentComponentBatchKernelArtifact],
    selected_dispatches: &[&VulkanPreparedDispatch],
    lane_capacity: usize,
    execution_mode: VulkanComponentBatchExecutionMode,
    capture_causal_state_snapshots: bool,
) -> Result<Vec<VulkanComponentBatchResidentAllocation>, VulkanError> {
    let selected_artifacts = selected_dispatches
        .iter()
        .map(|dispatch| {
            let artifact = selected_component_batch_kernel_artifact_for_prepared_dispatch(
                batch_kernels,
                dispatch,
                execution_mode,
                lane_capacity,
            );
            if artifact.is_some_and(|artifact| {
                artifact.batch_mode == VulkanResidentComponentKernelBatchMode::CausalScan
                    && lane_capacity > artifact.lane_tile_width
            }) {
                return Err(VulkanError(format!(
                    "causal scan kernel {}.{} cannot execute {lane_capacity} lanes",
                    dispatch.component_id, dispatch.node_id,
                )));
            }
            Ok((*dispatch, artifact))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let enabled = capture_causal_state_snapshots
        || selected_artifacts.iter().any(|(_, artifact)| {
            artifact.is_some_and(component_batch_artifact_reads_state_snapshots)
        });
    let mut allocations = vec![VulkanComponentBatchResidentAllocation {
        kind: VulkanComponentBatchResidentAllocationKind::CausalStateSnapshotDummy,
        byte_capacity: size_of::<u32>(),
        host_visible: false,
    }];
    if !enabled {
        return Ok(allocations);
    }
    let state_layouts = placed_plan
        .placed_resident_plan
        .resident_plan
        .stream_state_buffers
        .iter()
        .map(|state| {
            (
                (state.component_id.as_str(), state.state_id.as_str()),
                state.static_bytes,
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut snapshots = BTreeMap::<(String, String), usize>::new();
    for (dispatch, artifact) in selected_artifacts {
        let Some(artifact) = artifact else {
            continue;
        };
        for stage in artifact
            .stages
            .iter()
            .filter(|stage| stage.state_snapshot_binding.is_some())
        {
            let state = if let Some(source_binding) = stage.state_snapshot_source_binding {
                component_batch_prepared_static_state_at_binding(dispatch, source_binding)?
            } else {
                let writers = component_batch_prepared_static_state_writers(dispatch)?;
                let [writer] = writers.as_slice() else {
                    return Err(VulkanError(format!(
                        "component batch stage {:?} snapshots {} static state writers; expected one",
                        stage.shader_path,
                        writers.len(),
                    )));
                };
                writer.clone()
            };
            let static_byte_capacity = state_layouts
                .get(&(state.0.as_str(), state.1.as_str()))
                .copied()
                .flatten()
                .ok_or_else(|| {
                    VulkanError(format!(
                        "component batch snapshot references absent or dynamic-only state {}.{}",
                        state.0, state.1,
                    ))
                })?;
            match snapshots.entry(state) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(static_byte_capacity);
                }
                std::collections::btree_map::Entry::Occupied(entry)
                    if *entry.get() != static_byte_capacity =>
                {
                    return Err(VulkanError(
                        "component batch snapshot state has inconsistent static capacity"
                            .to_string(),
                    ));
                }
                std::collections::btree_map::Entry::Occupied(_) => {}
            }
        }
    }
    allocations.extend(snapshots.into_iter().map(
        |((component_id, state_id), static_byte_capacity)| {
            Ok(VulkanComponentBatchResidentAllocation {
                kind: VulkanComponentBatchResidentAllocationKind::CausalStateSnapshot {
                    component_id,
                    state_id,
                },
                byte_capacity: static_byte_capacity.checked_mul(lane_capacity).ok_or_else(
                    || VulkanError("component batch snapshot capacity overflowed".to_string()),
                )?,
                host_visible: false,
            })
        },
    ).collect::<Result<Vec<_>, VulkanError>>()?);
    Ok(allocations)
}

fn component_batch_prepared_static_state_writers(
    dispatch: &VulkanPreparedDispatch,
) -> Result<Vec<(String, String)>, VulkanError> {
    let mut writers = BTreeSet::new();
    for descriptor in &dispatch.descriptors {
        if !matches!(
            descriptor.usage,
            VulkanKernelDescriptorUsage::StateWrite | VulkanKernelDescriptorUsage::StateView
        ) {
            continue;
        }
        if let Some(state) = component_batch_prepared_static_state(descriptor)? {
            writers.insert(state);
        }
    }
    Ok(writers.into_iter().collect())
}

fn component_batch_prepared_static_state_at_binding(
    dispatch: &VulkanPreparedDispatch,
    source_binding: u32,
) -> Result<(String, String), VulkanError> {
    let descriptor = dispatch
        .descriptors
        .iter()
        .find(|descriptor| u32::try_from(descriptor.binding).ok() == Some(source_binding))
        .ok_or_else(|| {
            VulkanError(format!(
                "component batch snapshot reader {}.{} references absent descriptor binding {source_binding}",
                dispatch.component_id, dispatch.node_id,
            ))
        })?;
    component_batch_prepared_static_state(descriptor)?.ok_or_else(|| {
        VulkanError(format!(
            "component batch snapshot reader {}.{} descriptor {source_binding} is not static stream state",
            dispatch.component_id, dispatch.node_id,
        ))
    })
}

fn component_batch_prepared_static_state(
    descriptor: &VulkanResolvedDescriptorBinding,
) -> Result<Option<(String, String)>, VulkanError> {
    let (component_id, state_id, static_bytes) = match &descriptor.resource {
        VulkanDescriptorResourceAddress::StateBuffer {
            component_id,
            state_id,
            static_bytes,
            ..
        }
        | VulkanDescriptorResourceAddress::StateView {
            component_id,
            state_id,
            static_bytes,
            ..
        } => (component_id, state_id, static_bytes),
        _ => return Ok(None),
    };
    if static_bytes.is_some_and(|bytes| bytes == 0) {
        return Err(VulkanError(format!(
            "component batch state {component_id}.{state_id} has empty static storage",
        )));
    }
    Ok(static_bytes.map(|_| (component_id.clone(), state_id.clone())))
}
