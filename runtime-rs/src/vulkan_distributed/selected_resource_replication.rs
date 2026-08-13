#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanSelectedResourceConcurrentStreamDemand {
    pub stream_id: String,
    /// Relative activation rate for this stream. Selection telemetry defines
    /// the resource distribution; this weight defines how much concurrent
    /// service demand the stream contributes.
    pub activation_rate_weight: u64,
    pub telemetry: crate::vulkan_stream_circuit::VulkanSelectionTelemetryDomainSnapshot,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanSelectedResourceConcurrentAssignment {
    pub stream_id: String,
    pub resource_index: usize,
    pub device_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanSelectedResourceReplicatedResidency {
    pub resource_index: usize,
    /// Every listed target has one immutable physical copy. Arithmetic still
    /// executes exactly once for each stream through `assignments`.
    pub execution_device_ids: Vec<String>,
    pub payload_bytes_per_copy: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanSelectedResourceConcurrentDeviceLoad {
    pub device_id: String,
    /// Unique immutable resources addressable on this physical target. Copies
    /// shared by multiple streams are counted exactly once. Eager and retained
    /// policies keep this complete set resident; paged policy admits the exact
    /// maximum wave below.
    pub addressable_resource_indices: Vec<usize>,
    pub addressable_payload_bytes: usize,
    /// Conservative simultaneous demand-loaded wave across all streams
    /// assigned to this target.
    pub maximum_load_wave_bytes: usize,
    /// Costs are scaled by the plan's `activation_normalization_denominator`.
    pub first_moment_ns: u128,
    pub second_moment_ns2: u128,
    pub stream_resource_indices: BTreeMap<String, Vec<usize>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanSelectedResourceConcurrentPlacementPlan {
    pub selector_id: String,
    pub residency_policy: crate::vulkan_stream_circuit::ResourceResidencyPolicy,
    pub stream_ids: Vec<String>,
    pub assignments: Vec<VulkanSelectedResourceConcurrentAssignment>,
    pub replicated_resources: Vec<VulkanSelectedResourceReplicatedResidency>,
    pub device_loads: Vec<VulkanSelectedResourceConcurrentDeviceLoad>,
    /// The common denominator used to combine selection frequencies from
    /// telemetry windows with different activation counts without rounding.
    pub activation_normalization_denominator: u128,
    pub maximum_first_moment_ns: u128,
    pub maximum_second_moment_ns2: u128,
    /// Conservative cold-load critical path assembled from exact measured
    /// resource-class loads. Per-device waves may overlap, while the groups
    /// conservatively admitted together on one device add.
    pub maximum_cold_load_wave_duration_ns: u128,
}

impl VulkanSelectedResourceConcurrentPlacementPlan {
    pub fn execution_ownership_by_stream(
        &self,
        resource_count: usize,
    ) -> Result<BTreeMap<String, BTreeMap<String, BTreeSet<usize>>>, VulkanDistributedPlanError>
    {
        if self.selector_id.trim().is_empty()
            || self.stream_ids.is_empty()
            || self.assignments.len()
                != self.stream_ids.len().checked_mul(resource_count).ok_or_else(|| {
                    VulkanDistributedPlanError(
                        "concurrent selected-resource ownership size overflowed".to_string(),
                    )
                })?
        {
            return Err(VulkanDistributedPlanError(
                "concurrent selected-resource placement does not cover its complete stream/resource domain"
                    .to_string(),
            ));
        }
        let stream_ids = self.stream_ids.iter().map(String::as_str).collect::<BTreeSet<_>>();
        if stream_ids.len() != self.stream_ids.len()
            || self
                .stream_ids
                .iter()
                .any(|stream_id| stream_id.trim().is_empty())
        {
            return Err(VulkanDistributedPlanError(
                "concurrent selected-resource placement repeats a stream ID".to_string(),
            ));
        }
        let mut covered = BTreeSet::new();
        let mut ownership = BTreeMap::<String, BTreeMap<String, BTreeSet<usize>>>::new();
        for assignment in &self.assignments {
            if !stream_ids.contains(assignment.stream_id.as_str())
                || assignment.device_id.trim().is_empty()
                || assignment.resource_index >= resource_count
                || !covered.insert((assignment.stream_id.as_str(), assignment.resource_index))
            {
                return Err(VulkanDistributedPlanError(
                    "concurrent selected-resource placement repeats arithmetic ownership or references an invalid stream, resource, or device"
                        .to_string(),
                ));
            }
            ownership
                .entry(assignment.stream_id.clone())
                .or_default()
                .entry(assignment.device_id.clone())
                .or_default()
                .insert(assignment.resource_index);
        }
        for stream_id in &self.stream_ids {
            if (0..resource_count)
                .any(|resource_index| !covered.contains(&(stream_id.as_str(), resource_index)))
            {
                return Err(VulkanDistributedPlanError(format!(
                    "concurrent selected-resource placement leaves stream {stream_id:?} with an unowned resource",
                )));
            }
        }
        Ok(ownership)
    }
}

#[derive(Clone, Debug)]
struct VulkanSelectedResourceConcurrentMutableDeviceLoad {
    device_id: String,
    resident_payload_capacity_bytes: usize,
    addressable_resource_indices: BTreeSet<usize>,
    first_moment_ns: u128,
    second_moment_ns2: u128,
    stream_resource_indices: BTreeMap<String, Vec<usize>>,
}

#[derive(Clone, Debug)]
struct VulkanSelectedResourceNormalizedStreamDemand<'a> {
    stream_id: &'a str,
    selection_weights: Vec<u128>,
    co_selection_weights: Vec<u128>,
}

/// Jointly places one selector for multiple concurrent streams.
///
/// Replication is a residency consequence of distinct, exactly-once stream
/// ownership: when two streams execute the same resource on different
/// targets, each target retains one immutable copy. This deliberately does
/// not introduce a "replicated execution" strategy or duplicate arithmetic.
/// Every execution and lazy-load cost comes from the exact compiler-declared
/// resource class measured on the target that receives the work.
#[allow(clippy::too_many_arguments)]
pub fn try_plan_concurrent_selected_resource_placement(
    component_id: &str,
    partition: &VulkanDistributedSelectedResourcePartitionPlan,
    execution_classes: &VulkanSelectedResourceExecutionClassPlan,
    streams: &[VulkanSelectedResourceConcurrentStreamDemand],
    devices: &[VulkanSelectedResourcePlacementDevice],
    residency_policy: crate::vulkan_stream_circuit::ResourceResidencyPolicy,
    phase: nerve_execution_contracts::ExecutionPhase,
) -> Result<Option<VulkanSelectedResourceConcurrentPlacementPlan>, VulkanDistributedPlanError> {
    try_plan_concurrent_selected_resource_placement_with_fixed_assignments(
        component_id,
        partition,
        execution_classes,
        streams,
        devices,
        residency_policy,
        phase,
        &[],
    )
}

/// Plans one stream against already-mounted peer ownership. Fixed assignments
/// remain exact; only uncovered stream/resource occurrences may move. This is
/// the package-safe best-response primitive used when one stream reaches a
/// quiescent prompt boundary but sibling streams are still live.
#[allow(clippy::too_many_arguments)]
pub fn try_plan_concurrent_selected_resource_placement_with_fixed_assignments(
    component_id: &str,
    partition: &VulkanDistributedSelectedResourcePartitionPlan,
    execution_classes: &VulkanSelectedResourceExecutionClassPlan,
    streams: &[VulkanSelectedResourceConcurrentStreamDemand],
    devices: &[VulkanSelectedResourcePlacementDevice],
    residency_policy: crate::vulkan_stream_circuit::ResourceResidencyPolicy,
    phase: nerve_execution_contracts::ExecutionPhase,
    fixed_assignments: &[VulkanSelectedResourceConcurrentAssignment],
) -> Result<Option<VulkanSelectedResourceConcurrentPlacementPlan>, VulkanDistributedPlanError> {
    if streams.is_empty() {
        return Err(VulkanDistributedPlanError(
            "concurrent selected-resource placement requires at least one stream".to_string(),
        ));
    }
    let mut stream_ids = BTreeSet::new();
    let mut activation_counts = Vec::with_capacity(streams.len());
    for stream in streams {
        if stream.stream_id.trim().is_empty()
            || !stream_ids.insert(stream.stream_id.as_str())
            || stream.activation_rate_weight == 0
        {
            return Err(VulkanDistributedPlanError(
                "concurrent selected-resource placement requires unique stream IDs and positive activation-rate weights"
                    .to_string(),
            ));
        }
        validate_selected_resource_placement_problem(
            component_id,
            partition,
            execution_classes,
            &stream.telemetry,
            devices,
            phase,
        )?;
        let selection_total = stream
            .telemetry
            .selection_counts
            .iter()
            .try_fold(0u64, |total, count| total.checked_add(*count))
            .ok_or_else(|| {
                VulkanDistributedPlanError(
                    "concurrent selected-resource selection total overflowed".to_string(),
                )
            })?;
        let selection_width = u64::try_from(partition.selection_count_per_activation)
            .map_err(|_| {
                VulkanDistributedPlanError(
                    "concurrent selected-resource selection width exceeds u64".to_string(),
                )
            })?;
        if selection_width == 0
            || selection_total == 0
            || !selection_total.is_multiple_of(selection_width)
        {
            return Err(VulkanDistributedPlanError(format!(
                "concurrent selected-resource telemetry for stream {:?} does not contain complete activations",
                stream.stream_id,
            )));
        }
        let activation_count = selection_total / selection_width;
        let pairs_per_activation = selection_width
            .checked_mul(selection_width.saturating_sub(1))
            .and_then(|pairs| pairs.checked_div(2))
            .ok_or_else(|| {
                VulkanDistributedPlanError(
                    "concurrent selected-resource pair width overflowed".to_string(),
                )
            })?;
        let expected_pair_total = activation_count
            .checked_mul(pairs_per_activation)
            .ok_or_else(|| {
                VulkanDistributedPlanError(
                    "concurrent selected-resource expected pair total overflowed".to_string(),
                )
            })?;
        let observed_pair_total = stream
            .telemetry
            .co_selection_counts
            .iter()
            .try_fold(0u64, |total, count| total.checked_add(*count))
            .ok_or_else(|| {
                VulkanDistributedPlanError(
                    "concurrent selected-resource pair total overflowed".to_string(),
                )
            })?;
        if observed_pair_total != expected_pair_total {
            return Err(VulkanDistributedPlanError(format!(
                "concurrent selected-resource telemetry for stream {:?} has {observed_pair_total} co-selections, expected {expected_pair_total}",
                stream.stream_id,
            )));
        }
        for resource_index in 0..partition.resource_count {
            let observed_degree = (0..partition.resource_count)
                .filter(|other| *other != resource_index)
                .try_fold(0u64, |total, other| {
                    total
                        .checked_add(stream.telemetry.co_selection_count(resource_index, other).unwrap_or(0))
                        .ok_or_else(|| {
                            VulkanDistributedPlanError(
                                "concurrent selected-resource co-selection degree overflowed"
                                    .to_string(),
                            )
                        })
                })?;
            let expected_degree = stream.telemetry.selection_counts[resource_index]
                .checked_mul(selection_width.saturating_sub(1))
                .ok_or_else(|| {
                    VulkanDistributedPlanError(
                        "concurrent selected-resource expected co-selection degree overflowed"
                            .to_string(),
                    )
                })?;
            if observed_degree != expected_degree {
                return Err(VulkanDistributedPlanError(format!(
                    "concurrent selected-resource telemetry for stream {:?} resource {resource_index} has co-selection degree {observed_degree}, expected {expected_degree}",
                    stream.stream_id,
                )));
            }
        }
        activation_counts.push(activation_count);
    }

    let activation_normalization_denominator = activation_counts
        .iter()
        .try_fold(1u128, |denominator, count| {
            checked_least_common_multiple(denominator, u128::from(*count))
        })?;
    let normalized_streams = streams
        .iter()
        .zip(&activation_counts)
        .map(|(stream, activation_count)| {
            let scale = activation_normalization_denominator / u128::from(*activation_count);
            let rate = u128::from(stream.activation_rate_weight);
            let normalize = |count: u64| {
                u128::from(count)
                    .checked_mul(scale)
                    .and_then(|value| value.checked_mul(rate))
                    .ok_or_else(|| {
                        VulkanDistributedPlanError(
                            "concurrent selected-resource normalized telemetry overflowed"
                                .to_string(),
                        )
                    })
            };
            Ok(VulkanSelectedResourceNormalizedStreamDemand {
                stream_id: &stream.stream_id,
                selection_weights: stream
                    .telemetry
                    .selection_counts
                    .iter()
                    .copied()
                    .map(normalize)
                    .collect::<Result<Vec<_>, _>>()?,
                co_selection_weights: stream
                    .telemetry
                    .co_selection_counts
                    .iter()
                    .copied()
                    .map(normalize)
                    .collect::<Result<Vec<_>, _>>()?,
            })
        })
        .collect::<Result<Vec<_>, VulkanDistributedPlanError>>()?;
    let stream_index_by_id = normalized_streams
        .iter()
        .enumerate()
        .map(|(index, stream)| (stream.stream_id, index))
        .collect::<BTreeMap<_, _>>();
    let device_index_by_id = devices
        .iter()
        .enumerate()
        .map(|(index, device)| (device.device_id.as_str(), index))
        .collect::<BTreeMap<_, _>>();
    let mut fixed_device_by_occurrence = BTreeMap::new();
    for assignment in fixed_assignments {
        let Some(stream_index) = stream_index_by_id.get(assignment.stream_id.as_str()).copied()
        else {
            return Err(VulkanDistributedPlanError(format!(
                "fixed concurrent selected-resource assignment references unknown stream {:?}",
                assignment.stream_id,
            )));
        };
        let Some(device_index) = device_index_by_id.get(assignment.device_id.as_str()).copied()
        else {
            return Err(VulkanDistributedPlanError(format!(
                "fixed concurrent selected-resource assignment references unknown device {:?}",
                assignment.device_id,
            )));
        };
        if assignment.resource_index >= partition.resource_count
            || fixed_device_by_occurrence
                .insert((stream_index, assignment.resource_index), device_index)
                .is_some()
        {
            return Err(VulkanDistributedPlanError(
                "fixed concurrent selected-resource assignments repeat or exceed the stream/resource domain"
                    .to_string(),
            ));
        }
    }

    let mut occurrences = normalized_streams
        .iter()
        .enumerate()
        .flat_map(|(stream_index, stream)| {
            (0..partition.resource_count).map(move |resource_index| {
                let joint_weight = (0..partition.resource_count)
                    .filter(|other| *other != resource_index)
                    .try_fold(0u128, |total, other| {
                        total.checked_add(concurrent_co_selection_weight(
                            &stream.co_selection_weights,
                            partition.resource_count,
                            resource_index,
                            other,
                        )).ok_or_else(|| VulkanDistributedPlanError(
                            "concurrent selected-resource normalized joint weight overflowed"
                                .to_string(),
                        ))
                    });
                joint_weight.map(|joint_weight| (
                    stream_index,
                    resource_index,
                    stream.selection_weights[resource_index],
                    joint_weight,
                ))
            })
        })
        .collect::<Result<Vec<_>, VulkanDistributedPlanError>>()?;
    occurrences.sort_by(|left, right| {
        let left_fixed = fixed_device_by_occurrence.contains_key(&(left.0, left.1));
        let right_fixed = fixed_device_by_occurrence.contains_key(&(right.0, right.1));
        right_fixed
            .cmp(&left_fixed)
            .then_with(|| right.2.cmp(&left.2))
            .then_with(|| right.3.cmp(&left.3))
            .then_with(|| {
                partition.atomic_group_byte_counts[right.1]
                    .cmp(&partition.atomic_group_byte_counts[left.1])
            })
            .then_with(|| normalized_streams[left.0].stream_id.cmp(normalized_streams[right.0].stream_id))
            .then_with(|| left.1.cmp(&right.1))
    });

    let mut loads = devices
        .iter()
        .map(|device| VulkanSelectedResourceConcurrentMutableDeviceLoad {
            device_id: device.device_id.clone(),
            resident_payload_capacity_bytes: device.resident_payload_capacity_bytes,
            addressable_resource_indices: BTreeSet::new(),
            first_moment_ns: 0,
            second_moment_ns2: 0,
            stream_resource_indices: BTreeMap::new(),
        })
        .collect::<Vec<_>>();

    for (stream_index, resource_index, selection_weight, _) in occurrences {
        let stream = &normalized_streams[stream_index];
        let class_id = &execution_classes.resource_execution_class_ids[resource_index];
        let resource_bytes = partition.atomic_group_byte_counts[resource_index];
        let mut candidates = Vec::with_capacity(devices.len());
        for (device_index, (device, load)) in devices.iter().zip(&loads).enumerate() {
            if fixed_device_by_occurrence
                .get(&(stream_index, resource_index))
                .is_some_and(|fixed| *fixed != device_index)
            {
                continue;
            }
            let duration = u128::from(
                device.measured_costs_by_execution_class[class_id].execution_duration_ns,
            );
            let first_contribution = selection_weight.checked_mul(duration).ok_or_else(|| {
                VulkanDistributedPlanError(
                    "concurrent selected-resource first moment overflowed".to_string(),
                )
            })?;
            let self_second = selection_weight
                .checked_mul(duration)
                .and_then(|value| value.checked_mul(duration))
                .ok_or_else(|| {
                    VulkanDistributedPlanError(
                        "concurrent selected-resource second moment overflowed".to_string(),
                    )
                })?;
            let pair_second = load
                .stream_resource_indices
                .get(stream.stream_id)
                .into_iter()
                .flatten()
                .try_fold(0u128, |total, other_resource| {
                    let joint = concurrent_co_selection_weight(
                        &stream.co_selection_weights,
                        partition.resource_count,
                        resource_index,
                        *other_resource,
                    );
                    let other_class =
                        &execution_classes.resource_execution_class_ids[*other_resource];
                    let other_duration = u128::from(
                        device.measured_costs_by_execution_class[other_class]
                            .execution_duration_ns,
                    );
                    joint
                        .checked_mul(duration)
                        .and_then(|value| value.checked_mul(other_duration))
                        .and_then(|value| value.checked_mul(2))
                        .and_then(|value| total.checked_add(value))
                        .ok_or_else(|| {
                            VulkanDistributedPlanError(
                                "concurrent selected-resource joint second moment overflowed"
                                    .to_string(),
                            )
                        })
                })?;
            let projected_first = load
                .first_moment_ns
                .checked_add(first_contribution)
                .ok_or_else(|| {
                    VulkanDistributedPlanError(
                        "concurrent selected-resource projected first moment overflowed"
                            .to_string(),
                    )
                })?;
            let projected_second = load
                .second_moment_ns2
                .checked_add(self_second)
                .and_then(|value| value.checked_add(pair_second))
                .ok_or_else(|| {
                    VulkanDistributedPlanError(
                        "concurrent selected-resource projected second moment overflowed"
                            .to_string(),
                    )
                })?;
            let mut addressable_resources = load.addressable_resource_indices.clone();
            let added_copy_bytes = if addressable_resources.insert(resource_index) {
                resource_bytes
            } else {
                0
            };
            let addressable_payload_bytes = concurrent_selected_resource_payload_bytes(
                partition,
                addressable_resources.iter().copied(),
            )?;
            let mut stream_resources = load.stream_resource_indices.clone();
            stream_resources
                .entry(stream.stream_id.to_string())
                .or_default()
                .push(resource_index);
            let maximum_load_wave_bytes = concurrent_selected_resource_maximum_load_wave_bytes(
                partition,
                stream_resources.values(),
                addressable_payload_bytes,
            )?;
            let required_resident_bytes = match residency_policy {
                crate::vulkan_stream_circuit::ResourceResidencyPolicy::Eager
                | crate::vulkan_stream_circuit::ResourceResidencyPolicy::DemandRetained => {
                    addressable_payload_bytes
                }
                crate::vulkan_stream_circuit::ResourceResidencyPolicy::DemandPaged => {
                    maximum_load_wave_bytes
                }
            };
            if required_resident_bytes > load.resident_payload_capacity_bytes {
                continue;
            }
            let maximum_first = loads
                .iter()
                .enumerate()
                .map(|(index, candidate)| {
                    if index == device_index {
                        projected_first
                    } else {
                        candidate.first_moment_ns
                    }
                })
                .max()
                .unwrap_or(projected_first);
            let maximum_second = loads
                .iter()
                .enumerate()
                .map(|(index, candidate)| {
                    if index == device_index {
                        projected_second
                    } else {
                        candidate.second_moment_ns2
                    }
                })
                .max()
                .unwrap_or(projected_second);
            candidates.push((
                maximum_first,
                maximum_second,
                added_copy_bytes,
                std::cmp::Reverse(
                    load.resident_payload_capacity_bytes - required_resident_bytes,
                ),
                device.device_id.as_str(),
                device_index,
                addressable_resources,
                stream_resources,
                projected_first,
                projected_second,
            ));
        }
        candidates.sort_by(|left, right| {
            (&left.0, &left.1, &left.2, &left.3, &left.4)
                .cmp(&(&right.0, &right.1, &right.2, &right.3, &right.4))
        });
        let Some((
            _,
            _,
            _,
            _,
            _,
            device_index,
            addressable_resources,
            stream_resources,
            projected_first,
            projected_second,
        )) = candidates.into_iter().next()
        else {
            return Ok(None);
        };
        let load = &mut loads[device_index];
        load.addressable_resource_indices = addressable_resources;
        load.stream_resource_indices = stream_resources;
        load.first_moment_ns = projected_first;
        load.second_moment_ns2 = projected_second;
    }

    let mut assignments = loads
        .iter()
        .flat_map(|load| {
            load.stream_resource_indices
                .iter()
                .flat_map(|(stream_id, resources)| {
                    resources.iter().map(|resource_index| {
                        VulkanSelectedResourceConcurrentAssignment {
                            stream_id: stream_id.clone(),
                            resource_index: *resource_index,
                            device_id: load.device_id.clone(),
                        }
                    })
                })
        })
        .collect::<Vec<_>>();
    assignments.sort_by(|left, right| {
        left.stream_id
            .cmp(&right.stream_id)
            .then_with(|| left.resource_index.cmp(&right.resource_index))
    });
    let execution_devices_by_resource = assignments.iter().fold(
        vec![BTreeSet::<String>::new(); partition.resource_count],
        |mut devices, assignment| {
            devices[assignment.resource_index].insert(assignment.device_id.clone());
            devices
        },
    );
    let replicated_resources = execution_devices_by_resource
        .into_iter()
        .enumerate()
        .filter(|(_, devices)| devices.len() > 1)
        .map(|(resource_index, devices)| VulkanSelectedResourceReplicatedResidency {
            resource_index,
            execution_device_ids: devices.into_iter().collect(),
            payload_bytes_per_copy: partition.atomic_group_byte_counts[resource_index],
        })
        .collect::<Vec<_>>();
    let maximum_first_moment_ns = loads
        .iter()
        .map(|load| load.first_moment_ns)
        .max()
        .unwrap_or(0);
    let maximum_second_moment_ns2 = loads
        .iter()
        .map(|load| load.second_moment_ns2)
        .max()
        .unwrap_or(0);
    let mut cold_load_wave_ns_by_device = BTreeMap::<String, u128>::new();
    let device_loads = loads
        .into_iter()
        .map(|mut load| {
            for resources in load.stream_resource_indices.values_mut() {
                resources.sort_unstable();
            }
            let addressable_payload_bytes = concurrent_selected_resource_payload_bytes(
                partition,
                load.addressable_resource_indices.iter().copied(),
            )?;
            let maximum_load_wave_bytes = concurrent_selected_resource_maximum_load_wave_bytes(
                partition,
                load.stream_resource_indices.values(),
                addressable_payload_bytes,
            )?;
            let device = devices
                .iter()
                .find(|device| device.device_id == load.device_id)
                .expect("mutable concurrent load came from a validated device");
            let all_addressable_load_ns = load.addressable_resource_indices.iter().try_fold(
                0u128,
                |total, resource_index| {
                    let class_id = &execution_classes.resource_execution_class_ids[*resource_index];
                    total
                        .checked_add(u128::from(
                            device.measured_costs_by_execution_class[class_id]
                                .lazy_load_wave_duration_ns,
                        ))
                        .ok_or_else(|| {
                            VulkanDistributedPlanError(
                                "concurrent selected-resource cold-load duration overflowed"
                                    .to_string(),
                            )
                        })
                },
            )?;
            let cold_load_wave_ns = match residency_policy {
                crate::vulkan_stream_circuit::ResourceResidencyPolicy::Eager
                | crate::vulkan_stream_circuit::ResourceResidencyPolicy::DemandRetained => {
                    all_addressable_load_ns
                }
                crate::vulkan_stream_circuit::ResourceResidencyPolicy::DemandPaged => {
                    let summed_stream_waves = load.stream_resource_indices.values().try_fold(
                        0u128,
                        |total, resources| {
                            let mut durations = resources
                                .iter()
                                .map(|resource_index| {
                                    let class_id = &execution_classes.resource_execution_class_ids
                                        [*resource_index];
                                    u128::from(
                                        device.measured_costs_by_execution_class[class_id]
                                            .lazy_load_wave_duration_ns,
                                    )
                                })
                                .collect::<Vec<_>>();
                            durations.sort_unstable_by(|left, right| right.cmp(left));
                            let wave = durations
                                .into_iter()
                                .take(partition.selection_count_per_activation)
                                .try_fold(0u128, |wave, duration| {
                                    wave.checked_add(duration).ok_or_else(|| {
                                        VulkanDistributedPlanError(
                                            "concurrent selected-resource cold-load wave overflowed"
                                                .to_string(),
                                        )
                                    })
                                })?;
                            total.checked_add(wave).ok_or_else(|| {
                                VulkanDistributedPlanError(
                                    "concurrent selected-resource cross-stream cold-load wave overflowed"
                                        .to_string(),
                                )
                            })
                        },
                    )?;
                    summed_stream_waves.min(all_addressable_load_ns)
                }
            };
            cold_load_wave_ns_by_device.insert(load.device_id.clone(), cold_load_wave_ns);
            Ok(VulkanSelectedResourceConcurrentDeviceLoad {
                device_id: load.device_id,
                addressable_resource_indices: load
                    .addressable_resource_indices
                    .into_iter()
                    .collect(),
                addressable_payload_bytes,
                maximum_load_wave_bytes,
                first_moment_ns: load.first_moment_ns,
                second_moment_ns2: load.second_moment_ns2,
                stream_resource_indices: load.stream_resource_indices,
            })
        })
        .collect::<Result<Vec<_>, VulkanDistributedPlanError>>()?;
    let plan = VulkanSelectedResourceConcurrentPlacementPlan {
        selector_id: partition.selector_id.clone(),
        residency_policy,
        stream_ids: stream_ids.into_iter().map(str::to_string).collect(),
        assignments,
        replicated_resources,
        device_loads,
        activation_normalization_denominator,
        maximum_first_moment_ns,
        maximum_second_moment_ns2,
        maximum_cold_load_wave_duration_ns: cold_load_wave_ns_by_device
            .values()
            .copied()
            .max()
            .unwrap_or(0),
    };
    plan.execution_ownership_by_stream(partition.resource_count)?;
    Ok(Some(plan))
}

/// Produces one stream-local ownership change while treating every live peer
/// assignment as immutable. The accepted move must reduce the measured joint
/// device makespan, preserve the participant set, and repay its exact cold
/// destination loads within another observation window.
#[allow(clippy::too_many_arguments)]
pub fn try_plan_concurrent_selected_resource_reconfiguration(
    component_id: &str,
    partition: &VulkanDistributedSelectedResourcePartitionPlan,
    execution_classes: &VulkanSelectedResourceExecutionClassPlan,
    current_stream_id: &str,
    streams: &[VulkanSelectedResourceConcurrentStreamDemand],
    fixed_peer_assignments: &[VulkanSelectedResourceConcurrentAssignment],
    devices: &[VulkanSelectedResourcePlacementDevice],
    residency_policy: crate::vulkan_stream_circuit::ResourceResidencyPolicy,
    phase: nerve_execution_contracts::ExecutionPhase,
    current: &VulkanSelectedResourcePlacementPlan,
) -> Result<Option<VulkanSelectedResourceReconfigurationPlan>, VulkanDistributedPlanError> {
    if current_stream_id.trim().is_empty()
        || current.selector_id != partition.selector_id
        || !streams
            .iter()
            .any(|stream| stream.stream_id == current_stream_id)
        || fixed_peer_assignments
            .iter()
            .any(|assignment| assignment.stream_id == current_stream_id)
    {
        return Err(VulkanDistributedPlanError(
            "concurrent selected-resource reconfiguration has incompatible current-stream identity, placement, or peer ownership"
                .to_string(),
        ));
    }
    let current_stream = streams
        .iter()
        .find(|stream| stream.stream_id == current_stream_id)
        .expect("current stream identity was checked above");
    let observed_selection_count = current_stream
        .telemetry
        .selection_counts
        .iter()
        .try_fold(0u64, |total, count| total.checked_add(*count))
        .ok_or_else(|| {
            VulkanDistributedPlanError(
                "concurrent selected-resource observed selection count overflowed".to_string(),
            )
        })?;
    let selection_width = u64::try_from(partition.selection_count_per_activation).map_err(|_| {
        VulkanDistributedPlanError(
            "concurrent selected-resource selection width exceeds u64".to_string(),
        )
    })?;
    if selection_width == 0
        || observed_selection_count == 0
        || !observed_selection_count.is_multiple_of(selection_width)
    {
        return Err(VulkanDistributedPlanError(
            "concurrent selected-resource current stream has incomplete activation telemetry"
                .to_string(),
        ));
    }
    let observed_activation_count = observed_selection_count / selection_width;
    current.execution_ownership_by_device(partition.resource_count)?;
    let current_fixed = current
        .assignments
        .iter()
        .map(|assignment| VulkanSelectedResourceConcurrentAssignment {
            stream_id: current_stream_id.to_string(),
            resource_index: assignment.resource_index,
            device_id: assignment.device_id.clone(),
        })
        .collect::<Vec<_>>();
    let baseline_fixed = fixed_peer_assignments
        .iter()
        .cloned()
        .chain(current_fixed)
        .collect::<Vec<_>>();
    let Some(baseline) = try_plan_concurrent_selected_resource_placement_with_fixed_assignments(
        component_id,
        partition,
        execution_classes,
        streams,
        devices,
        residency_policy,
        phase,
        &baseline_fixed,
    )?
    else {
        return Ok(None);
    };
    let Some(joint_proposed) =
        try_plan_concurrent_selected_resource_placement_with_fixed_assignments(
            component_id,
            partition,
            execution_classes,
            streams,
            devices,
            residency_policy,
            phase,
            fixed_peer_assignments,
        )?
    else {
        return Ok(None);
    };
    if baseline.activation_normalization_denominator
        != joint_proposed.activation_normalization_denominator
        || baseline.maximum_first_moment_ns <= joint_proposed.maximum_first_moment_ns
    {
        return Ok(None);
    }
    let proposed_assignments = joint_proposed
        .assignments
        .iter()
        .filter(|assignment| assignment.stream_id == current_stream_id)
        .map(|assignment| VulkanSelectedResourceAssignment {
            resource_index: assignment.resource_index,
            device_id: assignment.device_id.clone(),
        })
        .collect::<Vec<_>>();
    if proposed_assignments == current.assignments {
        return Ok(None);
    }
    let proposed = score_selected_resource_assignments(
        component_id,
        partition,
        execution_classes,
        &current_stream.telemetry,
        devices,
        residency_policy,
        phase,
        &proposed_assignments,
    )?;
    let current_participants = current.execution_ownership_by_device(partition.resource_count)?;
    let proposed_participants = proposed.execution_ownership_by_device(partition.resource_count)?;
    if current_participants.keys().ne(proposed_participants.keys()) {
        return Ok(None);
    }
    let normalization = baseline.activation_normalization_denominator;
    let current_duration_ns_per_activation = baseline.maximum_first_moment_ns.div_ceil(normalization);
    let proposed_duration_ns_per_activation =
        joint_proposed.maximum_first_moment_ns.div_ceil(normalization);
    let Some(improvement_ns_per_activation) = current_duration_ns_per_activation
        .checked_sub(proposed_duration_ns_per_activation)
        .filter(|improvement| *improvement > 0)
    else {
        return Ok(None);
    };
    let current_owners = current
        .assignments
        .iter()
        .map(|assignment| (assignment.resource_index, assignment.device_id.as_str()))
        .collect::<BTreeMap<_, _>>();
    let devices_by_id = devices
        .iter()
        .map(|device| (device.device_id.as_str(), device))
        .collect::<BTreeMap<_, _>>();
    let mut moves = Vec::new();
    let mut load_ns_by_destination = BTreeMap::<String, u128>::new();
    for assignment in &proposed.assignments {
        let source_device_id = current_owners[&assignment.resource_index];
        if source_device_id == assignment.device_id {
            continue;
        }
        let destination = devices_by_id[assignment.device_id.as_str()];
        let class_id = &execution_classes.resource_execution_class_ids[assignment.resource_index];
        let load_duration = destination.measured_costs_by_execution_class[class_id]
            .lazy_load_wave_duration_ns;
        let destination_total = load_ns_by_destination
            .entry(assignment.device_id.clone())
            .or_default();
        *destination_total = destination_total
            .checked_add(u128::from(load_duration))
            .ok_or_else(|| {
                VulkanDistributedPlanError(
                    "concurrent selected-resource migration duration overflowed".to_string(),
                )
            })?;
        moves.push(VulkanSelectedResourcePlacementMove {
            resource_index: assignment.resource_index,
            source_device_id: source_device_id.to_string(),
            destination_device_id: assignment.device_id.clone(),
            payload_bytes: partition.atomic_group_byte_counts[assignment.resource_index],
            destination_load_duration_ns: load_duration,
        });
    }
    if moves.is_empty() {
        return Ok(None);
    }
    moves.sort_by_key(|movement| movement.resource_index);
    let migration_critical_path_ns = load_ns_by_destination
        .values()
        .copied()
        .max()
        .unwrap_or(0);
    let break_even_activation_count =
        migration_critical_path_ns.div_ceil(improvement_ns_per_activation);
    if break_even_activation_count > u128::from(observed_activation_count) {
        return Ok(None);
    }
    Ok(Some(VulkanSelectedResourceReconfigurationPlan {
        selector_id: partition.selector_id.clone(),
        observed_activation_count,
        current_duration_ns_per_activation,
        proposed_duration_ns_per_activation,
        migration_critical_path_ns,
        break_even_activation_count,
        moves,
        proposed,
    }))
}

fn checked_least_common_multiple(
    left: u128,
    right: u128,
) -> Result<u128, VulkanDistributedPlanError> {
    if left == 0 || right == 0 {
        return Err(VulkanDistributedPlanError(
            "concurrent selected-resource normalization has a zero activation count".to_string(),
        ));
    }
    let mut a = left;
    let mut b = right;
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    (left / a).checked_mul(right).ok_or_else(|| {
        VulkanDistributedPlanError(
            "concurrent selected-resource activation normalization overflowed".to_string(),
        )
    })
}

fn concurrent_co_selection_weight(
    co_selection_weights: &[u128],
    resource_count: usize,
    left: usize,
    right: usize,
) -> u128 {
    if left == right || co_selection_weights.is_empty() {
        return 0;
    }
    let (left, right) = if left < right { (left, right) } else { (right, left) };
    let preceding = left * (2 * resource_count - left - 1) / 2;
    co_selection_weights[preceding + right - left - 1]
}

fn concurrent_selected_resource_payload_bytes(
    partition: &VulkanDistributedSelectedResourcePartitionPlan,
    resource_indices: impl IntoIterator<Item = usize>,
) -> Result<usize, VulkanDistributedPlanError> {
    resource_indices.into_iter().try_fold(0usize, |total, resource_index| {
        total
            .checked_add(partition.atomic_group_byte_counts[resource_index])
            .ok_or_else(|| {
                VulkanDistributedPlanError(
                    "concurrent selected-resource payload bytes overflowed".to_string(),
                )
            })
    })
}

fn concurrent_selected_resource_maximum_load_wave_bytes<'a>(
    partition: &VulkanDistributedSelectedResourcePartitionPlan,
    stream_resources: impl IntoIterator<Item = &'a Vec<usize>>,
    resident_payload_bytes: usize,
) -> Result<usize, VulkanDistributedPlanError> {
    let summed_stream_waves = stream_resources
        .into_iter()
        .try_fold(0usize, |total, resources| {
            let wave = selected_resource_maximum_load_wave_bytes(
                partition,
                resources.iter().copied(),
            )?;
            total.checked_add(wave).ok_or_else(|| {
                VulkanDistributedPlanError(
                    "concurrent selected-resource load-wave bytes overflowed".to_string(),
                )
            })
        })?;
    Ok(summed_stream_waves.min(resident_payload_bytes))
}

#[cfg(test)]
mod selected_resource_replication_tests {
    use super::*;

    fn class_id() -> String {
        format!("sha256:{}", "a".repeat(64))
    }

    fn partition(resource_count: usize, selected: usize) -> VulkanDistributedSelectedResourcePartitionPlan {
        VulkanDistributedSelectedResourcePartitionPlan {
            execution_scope: "target".to_string(),
            selector_id: "experts".to_string(),
            node_id: "router".to_string(),
            domain_id: "expert-bank".to_string(),
            selection_signal: "routes".to_string(),
            address_table_binding: 3,
            parameter_slots_binding: 4,
            resource_count,
            parameters_per_resource: 1,
            parameter_partitions: Vec::new(),
            selection_count_per_activation: selected,
            resource_operation_class_ids: vec![class_id(); resource_count],
            atomic_group_ids: (0..resource_count).map(|index| format!("expert-{index}")).collect(),
            atomic_group_byte_counts: vec![10; resource_count],
            atomic_group_resource_ids: (0..resource_count).map(|index| vec![format!("weight-{index}")]).collect(),
            parameter_resource_ids: (0..resource_count).map(|index| vec![format!("weight-{index}")]).collect(),
            parameter_resource_byte_counts: vec![vec![10]; resource_count],
        }
    }

    fn classes(resource_count: usize) -> VulkanSelectedResourceExecutionClassPlan {
        VulkanSelectedResourceExecutionClassPlan {
            component_id: "layer".to_string(),
            selector_id: "experts".to_string(),
            resource_execution_class_ids: vec![class_id(); resource_count],
        }
    }

    fn devices(count: usize, capacity: usize) -> Vec<VulkanSelectedResourcePlacementDevice> {
        (0..count)
            .map(|index| VulkanSelectedResourcePlacementDevice {
                device_id: format!("gpu{index}"),
                physical_device_id: format!("physical-gpu{index}"),
                api_version: 1,
                driver_version: 1,
                resident_payload_capacity_bytes: capacity,
                measured_costs_by_execution_class: BTreeMap::from([(
                    class_id(),
                    VulkanSelectedResourceExecutionClassCost {
                        phase: nerve_execution_contracts::ExecutionPhase::Decode,
                        complete_transaction: true,
                        output_valid: true,
                        warmup_call_count: 1,
                        measured_call_count: 1,
                        execution_duration_ns: 10,
                        lazy_load_wave_duration_ns: 20,
                    },
                )]),
            })
            .collect()
    }

    fn stream(
        stream_id: &str,
        selection_counts: Vec<u64>,
        co_selection_counts: Vec<u64>,
    ) -> VulkanSelectedResourceConcurrentStreamDemand {
        VulkanSelectedResourceConcurrentStreamDemand {
            stream_id: stream_id.to_string(),
            activation_rate_weight: 1,
            telemetry: crate::vulkan_stream_circuit::VulkanSelectionTelemetryDomainSnapshot {
                execution_scope: "target".to_string(),
                component_id: "layer".to_string(),
                node_id: "router".to_string(),
                domain_id: "expert-bank".to_string(),
                resource_count: selection_counts.len(),
                selection_counts,
                co_selection_counts,
            },
        }
    }

    #[test]
    fn concurrent_hot_resource_is_replicated_without_duplicate_stream_arithmetic() {
        let partition = partition(2, 1);
        let plan = try_plan_concurrent_selected_resource_placement(
            "layer",
            &partition,
            &classes(2),
            &[
                stream("stream-a", vec![100, 0], Vec::new()),
                stream("stream-b", vec![100, 0], Vec::new()),
            ],
            &devices(2, 20),
            crate::vulkan_stream_circuit::ResourceResidencyPolicy::DemandRetained,
            nerve_execution_contracts::ExecutionPhase::Decode,
        )
        .unwrap()
        .unwrap();

        let ownership = plan.execution_ownership_by_stream(2).unwrap();
        assert_eq!(ownership.len(), 2);
        for stream_ownership in ownership.values() {
            assert_eq!(
                stream_ownership.values().map(BTreeSet::len).sum::<usize>(),
                2,
            );
        }
        assert_eq!(plan.replicated_resources.len(), 1);
        assert_eq!(plan.replicated_resources[0].resource_index, 0);
        assert_eq!(
            plan.replicated_resources[0].execution_device_ids,
            ["gpu0".to_string(), "gpu1".to_string()],
        );
        assert_eq!(plan.device_loads.iter().map(|load| load.addressable_payload_bytes).sum::<usize>(), 30);
        assert_eq!(plan.maximum_first_moment_ns, 1_000);
    }

    #[test]
    fn one_stream_never_creates_a_residency_replica() {
        let partition = partition(2, 1);
        let plan = try_plan_concurrent_selected_resource_placement(
            "layer",
            &partition,
            &classes(2),
            &[stream("only", vec![50, 50], Vec::new())],
            &devices(2, 10),
            crate::vulkan_stream_circuit::ResourceResidencyPolicy::DemandRetained,
            nerve_execution_contracts::ExecutionPhase::Decode,
        )
        .unwrap()
        .unwrap();

        assert!(plan.replicated_resources.is_empty());
        assert_eq!(plan.assignments.len(), 2);
        assert_eq!(plan.maximum_cold_load_wave_duration_ns, 20);
    }

    #[test]
    fn shared_copy_counts_once_and_demand_paged_waves_cover_concurrent_streams() {
        let partition = partition(2, 1);
        let plan = try_plan_concurrent_selected_resource_placement(
            "layer",
            &partition,
            &classes(2),
            &[
                stream("a", vec![1, 0], Vec::new()),
                stream("b", vec![1, 0], Vec::new()),
            ],
            &devices(1, 20),
            crate::vulkan_stream_circuit::ResourceResidencyPolicy::DemandPaged,
            nerve_execution_contracts::ExecutionPhase::Decode,
        )
        .unwrap()
        .unwrap();

        assert!(plan.replicated_resources.is_empty());
        assert_eq!(plan.device_loads[0].addressable_payload_bytes, 20);
        assert_eq!(plan.device_loads[0].maximum_load_wave_bytes, 20);
    }

    #[test]
    fn retained_joint_plan_fails_when_unique_resource_union_cannot_fit() {
        let partition = partition(3, 1);
        assert!(
            try_plan_concurrent_selected_resource_placement(
                "layer",
                &partition,
                &classes(3),
                &[stream("only", vec![1, 1, 1], Vec::new())],
                &devices(2, 10),
                crate::vulkan_stream_circuit::ResourceResidencyPolicy::DemandRetained,
                nerve_execution_contracts::ExecutionPhase::Decode,
            )
            .unwrap()
            .is_none()
        );
    }

    #[test]
    fn concurrent_plan_rejects_incomplete_activation_and_pair_histories() {
        let partition = partition(3, 2);
        let incomplete = try_plan_concurrent_selected_resource_placement(
            "layer",
            &partition,
            &classes(3),
            &[stream("bad", vec![1, 0, 0], vec![0, 0, 0])],
            &devices(2, 30),
            crate::vulkan_stream_circuit::ResourceResidencyPolicy::DemandPaged,
            nerve_execution_contracts::ExecutionPhase::Decode,
        )
        .unwrap_err();
        assert!(incomplete.0.contains("complete activations"));

        let malformed_pairs = try_plan_concurrent_selected_resource_placement(
            "layer",
            &partition,
            &classes(3),
            &[stream("bad", vec![1, 1, 0], vec![0, 0, 0])],
            &devices(2, 30),
            crate::vulkan_stream_circuit::ResourceResidencyPolicy::DemandPaged,
            nerve_execution_contracts::ExecutionPhase::Decode,
        )
        .unwrap_err();
        assert!(malformed_pairs.0.contains("co-selections"));

        let malformed_degrees = try_plan_concurrent_selected_resource_placement(
            "layer",
            &partition,
            &classes(3),
            &[stream("bad", vec![2, 1, 1], vec![2, 0, 0])],
            &devices(2, 30),
            crate::vulkan_stream_circuit::ResourceResidencyPolicy::DemandPaged,
            nerve_execution_contracts::ExecutionPhase::Decode,
        )
        .unwrap_err();
        assert!(malformed_degrees.0.contains("co-selection degree"));
    }

    #[test]
    fn different_window_lengths_are_normalized_exactly() {
        let partition = partition(2, 1);
        let plan = try_plan_concurrent_selected_resource_placement(
            "layer",
            &partition,
            &classes(2),
            &[
                stream("short", vec![2, 0], Vec::new()),
                stream("long", vec![6, 0], Vec::new()),
            ],
            &devices(2, 20),
            crate::vulkan_stream_circuit::ResourceResidencyPolicy::DemandRetained,
            nerve_execution_contracts::ExecutionPhase::Decode,
        )
        .unwrap()
        .unwrap();

        assert_eq!(plan.activation_normalization_denominator, 6);
        assert_eq!(plan.maximum_first_moment_ns, 60);
        assert_eq!(plan.replicated_resources.len(), 1);
    }

    #[test]
    fn fixed_peer_ownership_is_preserved_while_new_stream_uses_a_replica() {
        let partition = partition(2, 1);
        let streams = [
            stream("mounted-peer", vec![100, 0], Vec::new()),
            stream("current", vec![100, 0], Vec::new()),
        ];
        let fixed = [
            VulkanSelectedResourceConcurrentAssignment {
                stream_id: "mounted-peer".to_string(),
                resource_index: 0,
                device_id: "gpu0".to_string(),
            },
            VulkanSelectedResourceConcurrentAssignment {
                stream_id: "mounted-peer".to_string(),
                resource_index: 1,
                device_id: "gpu0".to_string(),
            },
        ];
        let plan = try_plan_concurrent_selected_resource_placement_with_fixed_assignments(
            "layer",
            &partition,
            &classes(2),
            &streams,
            &devices(2, 20),
            crate::vulkan_stream_circuit::ResourceResidencyPolicy::DemandRetained,
            nerve_execution_contracts::ExecutionPhase::Decode,
            &fixed,
        )
        .unwrap()
        .unwrap();

        for assignment in &fixed {
            assert!(plan.assignments.contains(assignment));
        }
        assert_eq!(
            plan.assignments
                .iter()
                .find(|assignment| {
                    assignment.stream_id == "current" && assignment.resource_index == 0
                })
                .unwrap()
                .device_id,
            "gpu1",
        );
        assert_eq!(plan.replicated_resources.len(), 1);
    }

    #[test]
    fn fixed_ownership_rejects_unknown_and_duplicate_coordinates() {
        let partition = partition(2, 1);
        let streams = [stream("known", vec![1, 0], Vec::new())];
        let unknown = [VulkanSelectedResourceConcurrentAssignment {
            stream_id: "missing".to_string(),
            resource_index: 0,
            device_id: "gpu0".to_string(),
        }];
        assert!(
            try_plan_concurrent_selected_resource_placement_with_fixed_assignments(
                "layer",
                &partition,
                &classes(2),
                &streams,
                &devices(2, 20),
                crate::vulkan_stream_circuit::ResourceResidencyPolicy::DemandRetained,
                nerve_execution_contracts::ExecutionPhase::Decode,
                &unknown,
            )
            .unwrap_err()
            .0
            .contains("unknown stream")
        );

        let repeated = [
            VulkanSelectedResourceConcurrentAssignment {
                stream_id: "known".to_string(),
                resource_index: 0,
                device_id: "gpu0".to_string(),
            },
            VulkanSelectedResourceConcurrentAssignment {
                stream_id: "known".to_string(),
                resource_index: 0,
                device_id: "gpu1".to_string(),
            },
        ];
        assert!(
            try_plan_concurrent_selected_resource_placement_with_fixed_assignments(
                "layer",
                &partition,
                &classes(2),
                &streams,
                &devices(2, 20),
                crate::vulkan_stream_circuit::ResourceResidencyPolicy::DemandRetained,
                nerve_execution_contracts::ExecutionPhase::Decode,
                &repeated,
            )
            .unwrap_err()
            .0
            .contains("repeat")
        );
    }

    #[test]
    fn concurrent_reconfiguration_reduces_joint_makespan_without_moving_peer_ownership() {
        let partition = partition(3, 1);
        let execution_classes = classes(3);
        // Two-copy capacity keeps one cold route on gpu0 after hot resource 0
        // is replicated to gpu1, so the mounted participant set is preserved.
        let devices = devices(2, 20);
        let current_telemetry = stream("current", vec![100, 0, 10], Vec::new());
        let peer_telemetry = stream("peer", vec![100, 0, 0], Vec::new());
        let current = score_selected_resource_assignments(
            "layer",
            &partition,
            &execution_classes,
            &current_telemetry.telemetry,
            &devices,
            crate::vulkan_stream_circuit::ResourceResidencyPolicy::DemandRetained,
            nerve_execution_contracts::ExecutionPhase::Decode,
            &[
                VulkanSelectedResourceAssignment {
                    resource_index: 0,
                    device_id: "gpu0".to_string(),
                },
                VulkanSelectedResourceAssignment {
                    resource_index: 1,
                    device_id: "gpu1".to_string(),
                },
                VulkanSelectedResourceAssignment {
                    resource_index: 2,
                    device_id: "gpu0".to_string(),
                },
            ],
        )
        .unwrap();
        let fixed_peer_assignments = [
            VulkanSelectedResourceConcurrentAssignment {
                stream_id: "peer".to_string(),
                resource_index: 0,
                device_id: "gpu0".to_string(),
            },
            VulkanSelectedResourceConcurrentAssignment {
                stream_id: "peer".to_string(),
                resource_index: 1,
                device_id: "gpu1".to_string(),
            },
            VulkanSelectedResourceConcurrentAssignment {
                stream_id: "peer".to_string(),
                resource_index: 2,
                device_id: "gpu0".to_string(),
            },
        ];
        let reconfiguration = try_plan_concurrent_selected_resource_reconfiguration(
            "layer",
            &partition,
            &execution_classes,
            "current",
            &[current_telemetry, peer_telemetry],
            &fixed_peer_assignments,
            &devices,
            crate::vulkan_stream_circuit::ResourceResidencyPolicy::DemandRetained,
            nerve_execution_contracts::ExecutionPhase::Decode,
            &current,
        )
        .unwrap()
        .expect("joint hot-resource contention should make one stream use a replica");

        assert!(
            reconfiguration.proposed_duration_ns_per_activation
                < reconfiguration.current_duration_ns_per_activation
        );
        assert!(
            reconfiguration.break_even_activation_count
                <= u128::from(reconfiguration.observed_activation_count)
        );
        assert!(reconfiguration.moves.iter().any(|movement| {
            movement.resource_index == 0
                && movement.source_device_id == "gpu0"
                && movement.destination_device_id == "gpu1"
        }));
        assert_eq!(
            fixed_peer_assignments
                .iter()
                .find(|assignment| assignment.resource_index == 0)
                .unwrap()
                .device_id,
            "gpu0",
        );
    }

    #[test]
    fn concurrent_reconfiguration_rejects_a_move_that_cannot_repay_its_load() {
        let partition = partition(3, 1);
        let execution_classes = classes(3);
        let mut devices = devices(2, 20);
        for device in &mut devices {
            device
                .measured_costs_by_execution_class
                .get_mut(&class_id())
                .unwrap()
                .lazy_load_wave_duration_ns = 10_000;
        }
        let current_telemetry = stream("current", vec![10, 0, 1], Vec::new());
        let peer_telemetry = stream("peer", vec![10, 0, 0], Vec::new());
        let current = score_selected_resource_assignments(
            "layer",
            &partition,
            &execution_classes,
            &current_telemetry.telemetry,
            &devices,
            crate::vulkan_stream_circuit::ResourceResidencyPolicy::DemandRetained,
            nerve_execution_contracts::ExecutionPhase::Decode,
            &[
                VulkanSelectedResourceAssignment {
                    resource_index: 0,
                    device_id: "gpu0".to_string(),
                },
                VulkanSelectedResourceAssignment {
                    resource_index: 1,
                    device_id: "gpu1".to_string(),
                },
                VulkanSelectedResourceAssignment {
                    resource_index: 2,
                    device_id: "gpu0".to_string(),
                },
            ],
        )
        .unwrap();
        let fixed_peer_assignments = (0..3)
            .map(|resource_index| VulkanSelectedResourceConcurrentAssignment {
                stream_id: "peer".to_string(),
                resource_index,
                device_id: if resource_index == 1 {
                    "gpu1"
                } else {
                    "gpu0"
                }
                .to_string(),
            })
            .collect::<Vec<_>>();
        assert!(
            try_plan_concurrent_selected_resource_reconfiguration(
                "layer",
                &partition,
                &execution_classes,
                "current",
                &[current_telemetry, peer_telemetry],
                &fixed_peer_assignments,
                &devices,
                crate::vulkan_stream_circuit::ResourceResidencyPolicy::DemandRetained,
                nerve_execution_contracts::ExecutionPhase::Decode,
                &current,
            )
            .unwrap()
            .is_none()
        );
    }
}
