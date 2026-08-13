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
        uses_demand_residency: bool,
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
        if execution_mode == VulkanComponentBatchExecutionMode::CausalSequence
            && uses_demand_residency
        {
            allocations.push(VulkanComponentBatchResidentAllocation {
                kind: VulkanComponentBatchResidentAllocationKind::DemandPipelinePredicate,
                byte_capacity: size_of::<u32>(),
                host_visible: false,
            });
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
