#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use sha2::{Digest, Sha256};

    use nerve_execution_contracts::{
        ArtifactIdentity, EquivalenceKind, EquivalenceRequirement, ExecutionForm,
        ExecutionGeometry, ExecutionPhase, ExecutionStrategy, InputContract,
        InputDistribution, OutputCollection, OutputContract, ParameterPartition,
        ParameterPartitionKind, PartitionExtent, PartitionLaunch, PartitionOrigin,
        PhysicalExecutionContract, PhysicalFormats, ReductionContract,
        ReductionFinalization, ReductionOperation, ResourceAccess, ResourceKind,
        ResourceRequirement, ResidencyRequirement, WorkgroupXMapping,
        PHYSICAL_EXECUTION_CONTRACT_SCHEMA,
    };

    use super::*;
    use crate::stream_plan::TensorMetadata;
    use crate::vulkan_stream_circuit::{
        VulkanKernelDescriptorUsage, VulkanKernelScalarBinding, VulkanKernelScalarSource,
        VulkanPhysicalKernelArtifact, VulkanResolvedDescriptorBinding,
        VulkanReusableKernelArtifact,
        physical_execution_artifact_id,
    };

    #[test]
    fn placed_components_do_not_implicitly_shard_their_internal_dispatches() {
        let prepared_plan = fixture_prepared_plan();
        let plan = VulkanDistributedExecutionPlan::from_prepared_plans(
            &[("gpu0", &prepared_plan)],
            &fixture_tensor_index("row_major"),
            &fixture_artifact_manifest(),
            &BTreeMap::new(),
            &[],
            256,
        )
        .unwrap();

        assert!(plan.device_ids.is_empty());
        assert!(plan.dispatches.is_empty());
        assert!(plan.execution_islands.is_empty());
        assert_eq!(plan.shared_input_byte_capacity, 0);
        assert_eq!(plan.shared_output_byte_capacity, 0);
        assert_eq!(plan.distributed_parameter_byte_count, 0);
    }

    #[test]
    fn timeline_dependency_clock_is_monotonic_and_refuses_wraparound() {
        let clock = VulkanDistributedDependencyClock::new();

        assert_eq!(clock.reserve("owner", 7).unwrap(), 1);
        assert_eq!(clock.reserve("owner", 7).unwrap(), 2);
        clock.validate_advance(64, "owner", 7).unwrap();
        clock.advance(64);
        assert_eq!(clock.reserve("owner", 7).unwrap(), 67);

        clock.next_value.set(u64::MAX);
        let error = clock.reserve("owner", 7).unwrap_err();
        assert!(error.to_string().contains("exhausted its timeline"));
        assert_eq!(clock.next_value.get(), u64::MAX);
    }

    #[test]
    fn plans_balanced_parameter_and_output_shards_from_compiled_contracts() {
        let plan = fixture_plan("row_major");

        assert_eq!(plan.dispatches.len(), 1);
        assert_eq!(plan.shared_input_byte_capacity, 8);
        assert_eq!(plan.shared_output_byte_capacity, 24);
        assert_eq!(plan.storage_buffer_offset_alignment, 4);
        assert_eq!(plan.distributed_parameter_byte_count, 192);
        assert_eq!(
            plan.shared_activation_route,
            VulkanSharedResidentBufferRoute::SharedHost,
        );
        assert_eq!(plan.execution_islands.len(), 1);
        let dispatch = &plan.dispatches[0];
        assert_eq!(dispatch.owner_device_id, "owner");
        assert_eq!(dispatch.row_alignment, 2);
        assert_eq!(dispatch.input_activation.component_id, "component");
        assert_eq!(dispatch.input_activation.signal_id, "normalized");
        assert_eq!(dispatch.input_activation.slot, 0);
        assert_eq!(dispatch.output_activation.component_id, "component");
        assert_eq!(dispatch.output_activation.signal_id, "hidden");
        assert_eq!(dispatch.output_activation.slot, 1);
        assert_eq!(
            dispatch
                .shards
                .iter()
                .map(|shard| (
                    shard.device_id.as_str(),
                    shard.row_start,
                    shard.row_count,
                    shard.workgroup_count_x,
                    shard.output_byte_offset,
                    shard.output_byte_count,
                ))
                .collect::<Vec<_>>(),
            vec![
                ("owner", 0, 4, 2, 0, 8),
                ("helper-a", 4, 4, 2, 8, 8),
                ("helper-b", 8, 2, 1, 16, 4),
                ("helper-c", 10, 2, 1, 20, 4),
            ]
        );
        assert_eq!(
            dispatch.shards[1]
                .parameters
                .iter()
                .map(|fragment| (
                    fragment.binding,
                    fragment.tensor.as_str(),
                    fragment.byte_offset,
                    fragment.byte_count,
                ))
                .collect::<Vec<_>>(),
            vec![(2, "gate", 32, 32), (3, "up", 32, 32)]
        );
        let island = &plan.execution_islands[0];
        assert_eq!(island.component_id, "component");
        assert_eq!(island.phase_schedules.len(), 1);
        assert_eq!(island.phase_schedules[0].phase, ExecutionPhase::Decode);
        assert_eq!(island.entry_device_id, "owner");
        assert_eq!(island.exit_device_id, "owner");
        assert_eq!(island.owner_device_id, "owner");
        assert_eq!(island.member_node_ids, ["ffn"]);
        assert_eq!(island.contract_ids, [dispatch.physical_execution_contract_id.clone()]);
        assert_eq!(island.implementation_digests, [dispatch.implementation_digest.clone()]);
        assert_eq!(island.participants.len(), 4);
        assert!(island.participants.iter().any(|participant| {
            participant.device_id == "owner"
                && participant
                    .roles
                    .contains(&VulkanPhysicalExecutionParticipantRole::Coordinator)
                && participant
                    .roles
                    .contains(&VulkanPhysicalExecutionParticipantRole::ShardWorker)
        }));
        assert_eq!(island.shard_assignments.len(), 4);
        assert_eq!(
            island
                .shard_assignments
                .iter()
                .map(|shard| shard.parameter_bytes)
                .sum::<usize>(),
            192,
        );
        assert!(island.transient_memory.iter().all(|requirement| {
            requirement.fixed_byte_capacity == 0
                && requirement.per_lane_byte_capacity > 0
        }));
        assert_eq!(island.transport_routes.len(), 6);
        assert!(island.transport_routes.iter().all(|route| {
            route.kind == VulkanPhysicalExecutionTransportKind::SharedHost
        }));
        assert_eq!(island.synchronization_routes.len(), 6);
        assert!(island.synchronization_routes.iter().all(|route| {
            route.kind == VulkanPhysicalExecutionSynchronizationKind::TimelineSemaphore
        }));
        assert_eq!(
            island
                .phase_schedules[0]
                .steps
                .iter()
                .map(|step| step.kind)
                .collect::<Vec<_>>(),
            vec![
                VulkanPhysicalExecutionScheduleKind::PublishInputs,
                VulkanPhysicalExecutionScheduleKind::ExecuteShards,
                VulkanPhysicalExecutionScheduleKind::CollectOutputs,
            ],
        );
        assert_eq!(
            island
                .residency
                .iter()
                .filter(|requirement| {
                    requirement.kind
                        == VulkanPhysicalExecutionResidencyKind::PermanentParameterShard
                })
                .map(|requirement| requirement.byte_capacity)
                .sum::<usize>(),
            192,
        );
    }

    #[test]
    fn resolved_island_rejects_unplanned_lazy_resource_residency() {
        let mut dispatch = fixture_plan("row_major").dispatches.remove(0);
        dispatch.has_lazy_resource_requirements = true;

        let error = resolved_physical_execution_islands(
            &[dispatch],
            VulkanSharedResidentBufferRoute::SharedHost,
        )
        .unwrap_err();

        assert!(error.to_string().contains("without a resolved atomic residency plan"));
    }

    #[test]
    fn resolved_island_preserves_edge_endpoints_and_owner_state() {
        let mut dispatch = fixture_plan("row_major").dispatches.remove(0);
        dispatch.input_activation.storage = VulkanDistributedActivationStorage::Edge {
            edge_index: 3,
            owner_device_id: "upstream".to_string(),
        };
        dispatch.output_activation.storage = VulkanDistributedActivationStorage::Edge {
            edge_index: 4,
            owner_device_id: "downstream".to_string(),
        };
        dispatch.owner_residency_requirements = vec![
            VulkanPhysicalExecutionResidencyRequirement {
                device_id: "owner".to_string(),
                kind: VulkanPhysicalExecutionResidencyKind::OwnerState,
                resource_id: "state:component:kv".to_string(),
                byte_capacity: 4096,
            },
        ];

        let islands = resolved_physical_execution_islands(
            &[dispatch],
            VulkanSharedResidentBufferRoute::ExternalDeviceLocal,
        )
        .unwrap();
        let island = &islands[0];

        assert_eq!(island.entry_device_id, "upstream");
        assert_eq!(island.exit_device_id, "downstream");
        assert!(island.transport_routes.iter().any(|route| {
            route.source_device_id == "upstream"
                && route.destination_device_id == "owner"
        }));
        assert!(island.transport_routes.iter().any(|route| {
            route.source_device_id == "owner"
                && route.destination_device_id == "downstream"
        }));
        assert!(island.residency.iter().any(|requirement| {
            requirement.kind == VulkanPhysicalExecutionResidencyKind::OwnerState
                && requirement.resource_id == "state:component:kv"
                && requirement.byte_capacity == 4096
        }));
        assert!(island.transport_routes.iter().all(|route| {
            route.kind == VulkanPhysicalExecutionTransportKind::ExternalDeviceLocal
        }));
    }

    #[test]
    fn samples_real_dispatch_fragments_under_one_total_parameter_budget() {
        let tensor_index = fixture_tensor_index("row_major");
        let sampled = fixture_plan("row_major")
            .sampled_for_parameter_budget(
                &tensor_index,
                &["owner".to_string(), "helper-a".to_string()],
                96,
            )
            .unwrap()
            .unwrap();
        let allocations = VulkanDistributedParameterAllocationPlan::
            from_sampled_execution_plan(&sampled, &tensor_index)
            .unwrap();

        assert!(allocations.total_byte_capacity <= 96);
        assert_eq!(sampled.device_ids, ["owner", "helper-a"]);
        assert!(sampled.dispatches.iter().all(|dispatch| {
            dispatch.shards.len() == 2
                && dispatch.shards.iter().all(|shard| {
                    shard.row_count > 0
                        && shard.row_count.is_multiple_of(dispatch.row_alignment)
                        && shard.workgroup_count_x > 0
                })
        }));
        assert!(
            VulkanDistributedParameterAllocationPlan::from_execution_plan(
                &sampled,
                &tensor_index,
            )
            .is_err()
        );
    }

    #[test]
    fn samples_the_same_real_dispatch_for_a_single_participant() {
        let tensor_index = fixture_tensor_index("row_major");
        let sampled = fixture_plan("row_major")
            .sampled_for_parameter_budget(
                &tensor_index,
                &["owner".to_string()],
                64,
            )
            .unwrap()
            .unwrap();
        let allocations = VulkanDistributedParameterAllocationPlan::
            from_sampled_execution_plan(&sampled, &tensor_index)
            .unwrap();

        assert_eq!(sampled.device_ids, ["owner"]);
        assert_eq!(sampled.dispatches[0].shards.len(), 1);
        assert_eq!(sampled.dispatches[0].shards[0].device_id, "owner");
        assert!(allocations.total_byte_capacity <= 64);
    }

    #[test]
    fn sampled_expert_ranges_keep_route_launch_and_output_geometry() {
        let mut dispatch = fixture_plan("row_major").dispatches.remove(0);
        dispatch.distribution = VulkanDistributedDispatchDistribution::ExpertRange;
        let source = dispatch.shards[0].clone();
        let sampled = sampled_distributed_dispatch_shard(
            &dispatch,
            &source,
            "owner",
            1,
            2,
        )
        .unwrap();

        assert_eq!(sampled.row_count, source.row_count / 2);
        assert_eq!(sampled.workgroup_count_x, source.workgroup_count_x);
        assert_eq!(sampled.output_byte_count, source.output_byte_count);
        assert_eq!(sampled.auxiliary_input_ranges, source.auxiliary_input_ranges);
        assert!(sampled
            .parameters
            .iter()
            .zip(&source.parameters)
            .all(|(sampled, source)| sampled.byte_count == source.byte_count / 2));
    }

    #[test]
    fn expert_range_push_constants_include_start_and_count() {
        let mut expert_dispatch = fixture_plan("row_major").dispatches.remove(0);
        expert_dispatch.distribution = VulkanDistributedDispatchDistribution::ExpertRange;
        let mut expert_shard = expert_dispatch.shards[1].clone();
        expert_shard.base_workgroup_z = 7;
        let bytes = distributed_shard_push_constants(&expert_dispatch, &expert_shard).unwrap();
        assert_eq!(bytes.len(), 8);
        assert_eq!(u32::from_le_bytes(bytes[..4].try_into().unwrap()), 7);
        assert_eq!(
            u32::from_le_bytes(bytes[4..].try_into().unwrap()),
            u32::try_from(expert_shard.row_count).unwrap(),
        );

        let output_rows = fixture_plan("row_major").dispatches.remove(0);
        let bytes =
            distributed_shard_push_constants(&output_rows, &output_rows.shards[1]).unwrap();
        assert!(bytes.is_empty());
    }

    #[test]
    fn distributed_sequence_selection_matches_the_submitted_variant() {
        let direct = "direct";
        let feedback = "feedback";
        assert_eq!(
            VulkanDistributedDispatchSequenceKind::for_feedback_lane(None),
            VulkanDistributedDispatchSequenceKind::Direct,
        );
        assert_eq!(
            VulkanDistributedDispatchSequenceKind::for_feedback_lane(Some(7)),
            VulkanDistributedDispatchSequenceKind::FeedbackIndirect,
        );
        assert_eq!(
            distributed_sequence_for_kind(
                &direct,
                Some(&feedback),
                VulkanDistributedDispatchSequenceKind::Direct,
                "device",
            )
            .unwrap(),
            &direct,
        );
        assert_eq!(
            distributed_sequence_for_kind(
                &direct,
                Some(&feedback),
                VulkanDistributedDispatchSequenceKind::FeedbackIndirect,
                "device",
            )
            .unwrap(),
            &feedback,
        );
        let error = distributed_sequence_for_kind(
            &direct,
            None,
            VulkanDistributedDispatchSequenceKind::FeedbackIndirect,
            "helper",
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("feedback shard on \"helper\" has no indirect sequence"));
    }

    #[test]
    fn embedded_distributed_sum_has_exact_typed_control() {
        let words = distributed_sum_f32_spirv_words().unwrap();
        assert_eq!(words.first().copied(), Some(0x0723_0203));
        let bf16_words = distributed_sum_f32_add_bf16_residual_spirv_words().unwrap();
        assert_eq!(bf16_words.first().copied(), Some(0x0723_0203));
        let reduction = VulkanDistributedReductionPlan {
            operation: ReductionOperation::SumF32,
            element_count: 4096,
            partial_byte_capacity: 4096 * 4,
            finalization: VulkanDistributedReductionFinalizationPlan::StoreF32,
        };
        let bytes = distributed_sum_f32_push_constants(&reduction, 5, 3).unwrap();
        assert_eq!(bytes.len(), 12);
        assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 4096);
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 5);
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 3);
    }

    #[test]
    fn plans_input_column_partials_with_typed_f32_reduction() {
        let activation = |binding, usage, signal: &str, bytes| VulkanResolvedDescriptorBinding {
            binding,
            usage,
            name: signal.to_string(),
            resource: VulkanDescriptorResourceAddress::ActivationSlot {
                component_id: "component".to_string(),
                signal_id: signal.to_string(),
                slot: binding,
                byte_capacity: bytes,
                signal_byte_capacity: bytes,
            },
        };
        let mut contract = test_physical_contract(
            "linear_partial",
            "down-partial",
            "down-partial.spv",
            ExecutionStrategy::TensorParallel,
            ExecutionForm::PartitionedInputPartialOutput,
            12,
            2,
            2,
            WorkgroupXMapping::Repeated,
            PartitionOrigin::PushConstantU32,
            Some("input_start"),
            Some("input_count"),
            vec![test_partition(
                "down-input-major",
                2,
                ParameterPartitionKind::Contiguous,
                2,
                1,
            )],
            vec![test_input(0, InputDistribution::Sharded, Some(2))],
            OutputContract {
                binding: 1,
                collection: OutputCollection::Reduced,
                dimension: None,
                alignment_elements: None,
                reduction: Some(ReductionContract {
                    operation: ReductionOperation::SumF32,
                    dimension_name: "output_elements".to_string(),
                    finalization: ReductionFinalization::StoreF32,
                }),
            },
        );
        contract
            .geometry
            .dimensions
            .insert("output_elements".to_string(), 4);
        let dispatch = VulkanPreparedDispatch {
            dispatch_index: 3,
            kernel_id: "component.down-partial".to_string(),
            component_id: "component".to_string(),
            circuit_id: "circuit".to_string(),
            node_index: 2,
            node_id: "down-partial".to_string(),
            op: "linear_partial".to_string(),
            reusable_family_id: "down-partial-family".to_string(),
            artifact_path: "canonical-down.spv".to_string(),
            entry_point: "main".to_string(),
            local_size_x: 64,
            descriptors: vec![
                activation(0, VulkanKernelDescriptorUsage::InputSignal, "input", 24),
                activation(1, VulkanKernelDescriptorUsage::OutputSignal, "output", 16),
                VulkanResolvedDescriptorBinding {
                    binding: 2,
                    usage: VulkanKernelDescriptorUsage::Parameter,
                    name: "down-input-major".to_string(),
                    resource: VulkanDescriptorResourceAddress::PermanentParameter {
                        param_id: "down-input-major".to_string(),
                        tensor: "canonical-down".to_string(),
                        byte_count: Some(96),
                    },
                },
            ],
            push_constants: vec![
                VulkanKernelScalarBinding {
                    name: "input_start".to_string(),
                    scalar_type: "u32".to_string(),
                    source: VulkanKernelScalarSource::PushConstant,
                },
                VulkanKernelScalarBinding {
                    name: "input_count".to_string(),
                    scalar_type: "u32".to_string(),
                    source: VulkanKernelScalarSource::PushConstant,
                },
            ],
            stream_control_binding: None,
            physical_execution_contracts: vec![contract.clone()],
        };
        let prepared = VulkanPreparedDispatchPlan {
            backend_id: "vulkan_stream_circuit".to_string(),
            reusable_family_count: 1,
            dispatches: vec![dispatch.clone()],
            total_descriptor_count: 3,
        };
        let tensor_index = TensorIndex {
            schema: "nerve.tensor_index.v1".to_string(),
            tensors: BTreeMap::from([(
                "down-input-major".to_string(),
                TensorMetadata {
                    dtype: "BF16".to_string(),
                    shape: vec![12, 4],
                    logical_shape: Some(vec![4, 12]),
                    parameter_count: Some(48),
                    byte_count: Some(96),
                    data_offsets: Some(vec![0, 96]),
                    source_file: Some("weights.safetensors".to_string()),
                    data_sha256: None,
                    layout: Some("row_major".to_string()),
                },
            )]),
        };
        let artifact = VulkanReusableKernelArtifact {
            family_id: "down-partial-family".to_string(),
            op: "linear_partial".to_string(),
            path: "canonical-down.spv".to_string(),
            entry_point: "main".to_string(),
            local_size_x: 64,
            workgroup_count_x: 2,
            descriptor_signature: Vec::new(),
            push_constants: dispatch.push_constants.clone(),
            stream_control_binding: None,
        };
        let mut artifacts = test_artifact_manifest_with_physical(artifact.clone());
        artifacts.artifacts[0].path = "down-partial.spv".to_string();
        let plan = VulkanDistributedExecutionPlan::from_prepared_plans(
            &[("owner", &prepared)],
            &tensor_index,
            &artifacts,
            &component_device_pools("component", &["owner", "helper-a", "helper-b"]),
            &[],
            4,
        )
        .unwrap();

        assert_eq!(plan.distributed_parameter_byte_count, 96);
        let planned = &plan.dispatches[0];
        assert_eq!(
            planned.physical_artifact_id,
            physical_execution_artifact_id(&contract.contract_id, 0)
        );
        assert_eq!(
            artifacts
                .artifacts
                .iter()
                .find(|artifact| artifact.artifact_id == planned.physical_artifact_id)
                .unwrap()
                .path,
            "down-partial.spv"
        );
        assert_eq!(
            planned.distribution,
            VulkanDistributedDispatchDistribution::InputColumns
        );
        assert_eq!(planned.input_distribution, InputDistribution::Sharded);
        assert_eq!(planned.output_collection, OutputCollection::Reduced);
        assert!(planned.shards.iter().all(|shard| {
            shard.parameters.len() == 1
                && shard.parameters[0].tensor == "down-input-major"
        }));
        assert_eq!(
            planned.reduction,
            Some(VulkanDistributedReductionPlan {
                operation: ReductionOperation::SumF32,
                element_count: 4,
                partial_byte_capacity: 16,
                finalization: VulkanDistributedReductionFinalizationPlan::StoreF32,
            })
        );

        let mut residual_finalized = prepared.clone();
        let residual_contract =
            &mut residual_finalized.dispatches[0].physical_execution_contracts[0];
        residual_contract
            .inputs
            .push(test_input(3, InputDistribution::Replicated, None));
        residual_contract.outputs[0]
            .reduction
            .as_mut()
            .unwrap()
            .finalization = ReductionFinalization::AddBf16ResidualToBf16 {
            residual_binding: 3,
        };
        let VulkanDescriptorResourceAddress::ActivationSlot {
            byte_capacity,
            signal_byte_capacity,
            ..
        } = &mut residual_finalized.dispatches[0].descriptors[1].resource
        else {
            panic!("fixture output is an activation slot");
        };
        *byte_capacity = 8;
        *signal_byte_capacity = 8;
        residual_finalized.dispatches[0].descriptors.push(activation(
            3,
            VulkanKernelDescriptorUsage::InputSignal,
            "residual",
            8,
        ));
        residual_finalized.total_descriptor_count = 4;
        let residual_plan = VulkanDistributedExecutionPlan::from_prepared_plans(
            &[("owner", &residual_finalized)],
            &tensor_index,
            &artifacts,
            &component_device_pools("component", &["owner", "helper-a"]),
            &[],
            4,
        )
        .unwrap();
        assert_eq!(
            residual_plan.dispatches[0].reduction,
            Some(VulkanDistributedReductionPlan {
                operation: ReductionOperation::SumF32,
                element_count: 4,
                partial_byte_capacity: 16,
                finalization:
                    VulkanDistributedReductionFinalizationPlan::AddBf16ResidualToBf16 {
                        residual_input_index: 1,
                    },
            })
        );
        assert_eq!(
            residual_plan.dispatches[0].auxiliary_input_activations[0].signal_id,
            "residual"
        );

        let buffer_plan =
            VulkanDistributedActivationBufferPlan::from_execution_plan(&plan).unwrap();
        assert_eq!(buffer_plan.allocation_count, 3);
        assert_eq!(buffer_plan.import_count, 7);
        assert_eq!(buffer_plan.reference_count, 6);
        assert_eq!(buffer_plan.total_shared_byte_capacity, 88);
        assert_eq!(
            buffer_plan
                .allocation("owner", "component", 1)
                .unwrap()
                .device_ids,
            ["owner"]
        );
        assert_eq!(
            buffer_plan.reduction_allocation("owner", 3).unwrap(),
            &VulkanDistributedReductionBufferAllocation {
                owner_device_id: "owner".to_string(),
                dispatch_index: 3,
                component_id: "component".to_string(),
                node_id: "down-partial".to_string(),
                plane_byte_capacity: 16,
                byte_capacity: 48,
                device_ids: vec![
                    "owner".to_string(),
                    "helper-a".to_string(),
                    "helper-b".to_string(),
                ],
            }
        );
        assert_eq!(
            planned
                .shards
                .iter()
                .map(|shard| (
                    shard.device_id.as_str(),
                    shard.row_start,
                    shard.row_count,
                    shard.workgroup_count_x,
                    shard.input_range.byte_offset,
                    shard.input_range.byte_count,
                    shard.output_byte_offset,
                    shard.output_byte_count,
                    shard.parameters[0].byte_offset,
                    shard.parameters[0].byte_count,
                ))
                .collect::<Vec<_>>(),
            vec![
                ("owner", 0, 4, 2, 0, 8, 0, 16, 0, 32),
                ("helper-a", 4, 4, 2, 8, 8, 0, 16, 32, 32),
                ("helper-b", 8, 4, 2, 16, 8, 0, 16, 64, 32),
            ]
        );
        let push = distributed_shard_push_constants(planned, &planned.shards[2]).unwrap();
        assert_eq!(u32::from_le_bytes(push[..4].try_into().unwrap()), 8);
        assert_eq!(u32::from_le_bytes(push[4..].try_into().unwrap()), 4);

        let mut invalid_accumulation = prepared.clone();
        invalid_accumulation.dispatches[0].physical_execution_contracts[0]
            .formats
            .accumulation = "bf16".to_string();
        let error = VulkanDistributedExecutionPlan::from_prepared_plans(
            &[("owner", &invalid_accumulation)],
            &tensor_index,
            &artifacts,
            &component_device_pools("component", &["owner", "helper-a"]),
            &[],
            4,
        )
        .unwrap_err();
        assert!(error.to_string().contains("requires f32 accumulation"));

        let missing_physical_resource = TensorIndex {
            schema: tensor_index.schema.clone(),
            tensors: BTreeMap::new(),
        };
        let error = VulkanDistributedExecutionPlan::from_prepared_plans(
            &[("owner", &prepared)],
            &missing_physical_resource,
            &artifacts,
            &component_device_pools("component", &["owner", "helper-a"]),
            &[],
            4,
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("no tensor metadata for \"down-input-major\""));

        let mut wrong_output_size = prepared.clone();
        let VulkanDescriptorResourceAddress::ActivationSlot {
            byte_capacity,
            signal_byte_capacity,
            ..
        } = &mut wrong_output_size.dispatches[0].descriptors[1].resource
        else {
            panic!("fixture output is an activation slot");
        };
        *byte_capacity = 8;
        *signal_byte_capacity = 8;
        let error = VulkanDistributedExecutionPlan::from_prepared_plans(
            &[("owner", &wrong_output_size)],
            &tensor_index,
            &artifacts,
            &component_device_pools("component", &["owner", "helper-a"]),
            &[],
            4,
        )
        .unwrap_err();
        assert!(error.to_string().contains("produces 16 bytes"));

        let mut wrong_abi = artifacts.clone();
        wrong_abi.artifacts[0].push_constants.pop();
        let error = VulkanDistributedExecutionPlan::from_prepared_plans(
            &[("owner", &prepared)],
            &tensor_index,
            &wrong_abi,
            &component_device_pools("component", &["owner", "helper-a"]),
            &[],
            4,
        )
        .unwrap_err();
        assert!(error.to_string().contains("exact push-constant ABI"));

        let mut strided_partition = prepared;
        strided_partition.dispatches[0].physical_execution_contracts[0]
            .parameter_partitions[0]
            .dimension = 1;
        let error = VulkanDistributedExecutionPlan::from_prepared_plans(
            &[("owner", &strided_partition)],
            &tensor_index,
            &artifacts,
            &component_device_pools("component", &["owner", "helper-a"]),
            &[],
            4,
        )
        .unwrap_err();
        assert!(error.to_string().contains("unsupported dimension 1"));
    }

    #[test]
    fn plans_sparse_expert_ranges_with_shared_routes_and_full_outputs() {
        let activation = |binding, signal: &str, slot, bytes| VulkanResolvedDescriptorBinding {
            binding,
            usage: if binding == 2 {
                VulkanKernelDescriptorUsage::OutputSignal
            } else {
                VulkanKernelDescriptorUsage::InputSignal
            },
            name: signal.to_string(),
            resource: VulkanDescriptorResourceAddress::ActivationSlot {
                component_id: "moe".to_string(),
                signal_id: signal.to_string(),
                slot,
                byte_capacity: bytes,
                signal_byte_capacity: bytes,
            },
        };
        let parameter = |binding, tensor: &str, bytes| VulkanResolvedDescriptorBinding {
            binding,
            usage: VulkanKernelDescriptorUsage::Parameter,
            name: tensor.to_string(),
            resource: VulkanDescriptorResourceAddress::PermanentParameter {
                param_id: tensor.to_string(),
                tensor: tensor.to_string(),
                byte_count: Some(bytes),
            },
        };
        let prepared = VulkanPreparedDispatchPlan {
            backend_id: "vulkan_stream_circuit".to_string(),
            reusable_family_count: 1,
            dispatches: vec![VulkanPreparedDispatch {
                dispatch_index: 9,
                kernel_id: "moe.sparse-down".to_string(),
                component_id: "moe".to_string(),
                circuit_id: "moe-circuit".to_string(),
                node_index: 4,
                node_id: "sparse-down".to_string(),
                op: "sparse_moe_down".to_string(),
                reusable_family_id: "sparse-family".to_string(),
                artifact_path: "sparse.spv".to_string(),
                entry_point: "main".to_string(),
                local_size_x: 64,
                descriptors: vec![
                    activation(0, "intermediates", 0, 8192),
                    activation(1, "routes", 1, 32),
                    activation(2, "outputs", 2, 32768),
                    parameter(3, "expert-weight", 256 * 2048 * 512),
                    parameter(4, "expert-scale", 256 * 16 * 4 * 2),
                ],
                push_constants: vec![
                    VulkanKernelScalarBinding {
                        name: "expert_start".to_string(),
                        scalar_type: "u32".to_string(),
                        source: VulkanKernelScalarSource::PushConstant,
                    },
                    VulkanKernelScalarBinding {
                        name: "expert_count".to_string(),
                        scalar_type: "u32".to_string(),
                        source: VulkanKernelScalarSource::PushConstant,
                    },
                ],
                stream_control_binding: None,
                physical_execution_contracts: vec![test_physical_contract(
                    "sparse_moe_down",
                    "sparse-down",
                    "sparse.spv",
                    ExecutionStrategy::ExpertParallel,
                    ExecutionForm::WholeExpertOwnership,
                    256,
                    1,
                    8192,
                    WorkgroupXMapping::Repeated,
                    PartitionOrigin::PushConstantU32,
                    Some("expert_start"),
                    Some("expert_count"),
                    vec![
                        test_partition(
                            "expert-weight",
                            3,
                            ParameterPartitionKind::ExpertRange,
                            1,
                            1,
                        ),
                        test_partition(
                            "expert-scale",
                            4,
                            ParameterPartitionKind::ExpertRange,
                            1,
                            1,
                        ),
                    ],
                    vec![
                        test_input(0, InputDistribution::Replicated, None),
                        test_input(1, InputDistribution::Routed, Some(1)),
                    ],
                    test_output(2, OutputCollection::Routed, Some(1)),
                )],
            }],
            total_descriptor_count: 5,
        };
        let tensor_index = TensorIndex {
            schema: "nerve.tensor_index.v1".to_string(),
            tensors: BTreeMap::from([
                (
                    "expert-weight".to_string(),
                    TensorMetadata {
                        dtype: "F8_E4M3".to_string(),
                        shape: vec![256, 2048, 512],
                        logical_shape: None,
                        parameter_count: Some(256 * 2048 * 512),
                        byte_count: Some(256 * 2048 * 512),
                        data_offsets: Some(vec![0, 256 * 2048 * 512]),
                        source_file: Some("weights.safetensors".to_string()),
                        data_sha256: None,
                        layout: Some("row_major".to_string()),
                    },
                ),
                (
                    "expert-scale".to_string(),
                    TensorMetadata {
                        dtype: "BF16".to_string(),
                        shape: vec![256, 16, 4],
                        logical_shape: None,
                        parameter_count: Some(256 * 16 * 4),
                        byte_count: Some(256 * 16 * 4 * 2),
                        data_offsets: Some(vec![0, 256 * 16 * 4 * 2]),
                        source_file: Some("weights.safetensors".to_string()),
                        data_sha256: None,
                        layout: Some("row_major".to_string()),
                    },
                ),
            ]),
        };
        let artifacts = test_artifact_manifest_with_physical(VulkanReusableKernelArtifact {
                family_id: "sparse-family".to_string(),
                op: "sparse_moe_down".to_string(),
                path: "sparse.spv".to_string(),
                entry_point: "main".to_string(),
                local_size_x: 64,
                workgroup_count_x: 8192,
                descriptor_signature: Vec::new(),
                push_constants: vec![
                    VulkanKernelScalarBinding {
                        name: "expert_start".to_string(),
                        scalar_type: "u32".to_string(),
                        source: VulkanKernelScalarSource::PushConstant,
                    },
                    VulkanKernelScalarBinding {
                        name: "expert_count".to_string(),
                        scalar_type: "u32".to_string(),
                        source: VulkanKernelScalarSource::PushConstant,
                    },
                ],
                stream_control_binding: None,
            });

        let plan = VulkanDistributedExecutionPlan::from_prepared_plans(
            &[("owner", &prepared)],
            &tensor_index,
            &artifacts,
            &component_device_pools("moe", &["owner", "helper"]),
            &[],
            256,
        )
        .unwrap();

        assert_eq!(plan.dispatches.len(), 1);
        let dispatch = &plan.dispatches[0];
        assert_eq!(
            dispatch.distribution,
            VulkanDistributedDispatchDistribution::ExpertRange
        );
        assert_eq!(dispatch.input_activation.binding, 0);
        assert_eq!(dispatch.auxiliary_input_activations[0].binding, 1);
        assert_eq!(dispatch.output_activation.binding, 2);
        assert_eq!(dispatch.shards.len(), 2);
        assert_eq!(dispatch.shards[0].device_id, "owner");
        assert_eq!(dispatch.shards[0].row_start, 0);
        assert_eq!(dispatch.shards[0].row_count, 128);
        assert_eq!(dispatch.shards[0].base_workgroup_z, 0);
        assert_eq!(dispatch.shards[1].device_id, "helper");
        assert_eq!(dispatch.shards[1].row_start, 128);
        assert_eq!(dispatch.shards[1].row_count, 128);
        assert_eq!(dispatch.shards[1].base_workgroup_z, 128);
        assert!(
            dispatch
                .shards
                .iter()
                .all(|shard| shard.workgroup_count_x == 8192
                    && shard.output_byte_offset == 0
                    && shard.output_byte_count == 32768)
        );
        assert_eq!(
            dispatch.shards[1].parameters[0].byte_offset,
            128 * 2048 * 512
        );
        assert_eq!(
            dispatch.shards[1].parameters[0].byte_count,
            128 * 2048 * 512
        );
        assert_eq!(
            dispatch.shards[1].parameters[1].byte_offset,
            128 * 16 * 4 * 2
        );
        assert_eq!(
            dispatch.shards[1].parameters[1].byte_count,
            128 * 16 * 4 * 2
        );

        let mut prequant = prepared.clone();
        for descriptor in &mut prequant.dispatches[0].descriptors {
            if descriptor.binding >= 1 {
                descriptor.binding += 1;
            }
        }
        prequant.dispatches[0].descriptors.insert(
            1,
            VulkanResolvedDescriptorBinding {
                binding: 1,
                usage: VulkanKernelDescriptorUsage::InputSignal,
                name: "input-scale".to_string(),
                resource: VulkanDescriptorResourceAddress::ActivationSlot {
                    component_id: "moe".to_string(),
                    signal_id: "input-scale".to_string(),
                    slot: 3,
                    byte_capacity: 64,
                    signal_byte_capacity: 64,
                },
            },
        );
        prequant.total_descriptor_count = 6;
        prequant.dispatches[0].physical_execution_contracts = vec![
            test_physical_contract(
                "sparse_moe_down",
                "sparse-down",
                "sparse.spv",
                ExecutionStrategy::ExpertParallel,
                ExecutionForm::WholeExpertOwnership,
                256,
                1,
                8192,
                WorkgroupXMapping::Repeated,
                PartitionOrigin::PushConstantU32,
                Some("expert_start"),
                Some("expert_count"),
                vec![
                    test_partition(
                        "expert-weight",
                        4,
                        ParameterPartitionKind::ExpertRange,
                        1,
                        1,
                    ),
                    test_partition(
                        "expert-scale",
                        5,
                        ParameterPartitionKind::ExpertRange,
                        1,
                        1,
                    ),
                ],
                vec![
                    test_input(0, InputDistribution::Replicated, None),
                    test_input(1, InputDistribution::Replicated, None),
                    test_input(2, InputDistribution::Routed, Some(1)),
                ],
                test_output(3, OutputCollection::Routed, Some(1)),
            ),
        ];
        let prequant_plan = VulkanDistributedExecutionPlan::from_prepared_plans(
            &[("owner", &prequant)],
            &tensor_index,
            &artifacts,
            &component_device_pools("moe", &["owner", "helper"]),
            &[],
            256,
        )
        .unwrap();
        let prequant_dispatch = &prequant_plan.dispatches[0];
        assert_eq!(prequant_dispatch.input_activation.binding, 0);
        assert_eq!(
            prequant_dispatch
                .auxiliary_input_activations
                .iter()
                .map(|activation| activation.binding)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(prequant_dispatch.output_activation.binding, 3);
        assert_eq!(
            prequant_dispatch.shards[0]
                .parameters
                .iter()
                .map(|parameter| parameter.binding)
                .collect::<Vec<_>>(),
            vec![4, 5]
        );

        let mut custom_range_abi = prepared.clone();
        custom_range_abi.dispatches[0].physical_execution_contracts[0]
            .partition_launch
            .as_mut()
            .unwrap()
            .count_push_constant = Some("owned_expert_count".to_string());
        let mut custom_range_artifacts = artifacts.clone();
        custom_range_artifacts.artifacts[0].push_constants[1].name =
            "owned_expert_count".to_string();
        VulkanDistributedExecutionPlan::from_prepared_plans(
            &[("owner", &custom_range_abi)],
            &tensor_index,
            &custom_range_artifacts,
            &component_device_pools("moe", &["owner", "helper"]),
            &[],
            256,
        )
        .unwrap();

        let mut stale_artifacts = artifacts.clone();
        stale_artifacts.artifacts[0].push_constants.truncate(1);
        let stale_abi_plan = VulkanDistributedExecutionPlan::from_prepared_plans(
            &[("owner", &prepared)],
            &tensor_index,
            &stale_artifacts,
            &component_device_pools("moe", &["owner", "helper"]),
            &[],
            256,
        )
        .unwrap_err();
        assert!(stale_abi_plan
            .to_string()
            .contains("requires exact push-constant ABI"));
        assert!(stale_abi_plan.to_string().contains("expert_count"));
    }

    #[test]
    fn islands_contain_only_adjacent_dataflow_compatible_expert_dispatches() {
        let mut producer = fixture_plan("row_major").dispatches.remove(0);
        producer.dispatch_index = 7;
        producer.distribution = VulkanDistributedDispatchDistribution::ExpertRange;
        producer.output_activation = producer.input_activation.clone();
        producer.output_activation.binding = 1;
        let mut consumer = producer.clone();
        consumer.dispatch_index = 8;
        consumer.node_id = "consumer".to_string();
        consumer.input_activation = producer.output_activation.clone();
        consumer.input_activation.binding = 0;

        let groups = resolved_physical_execution_islands(
            &[producer.clone(), consumer.clone()],
            VulkanSharedResidentBufferRoute::SharedHost,
        )
        .unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].dispatch_indices(), vec![7, 8]);
        assert!(groups[0].transient_memory.iter().all(|requirement| {
            requirement.kind
                == VulkanPhysicalExecutionTransientMemoryKind::SharedActivationAllocation
        }));

        let mut non_adjacent = consumer.clone();
        non_adjacent.dispatch_index = 9;
        assert_eq!(
            resolved_physical_execution_islands(
                &[producer.clone(), non_adjacent],
                VulkanSharedResidentBufferRoute::SharedHost,
            )
                .unwrap()
                .len(),
            2
        );

        let mut different_dataflow = consumer.clone();
        different_dataflow.input_activation.signal_id = "another-signal".to_string();
        assert_eq!(
            resolved_physical_execution_islands(
                &[producer.clone(), different_dataflow],
                VulkanSharedResidentBufferRoute::SharedHost,
            )
                .unwrap()
                .len(),
            2
        );

        let mut different_shards = consumer.clone();
        different_shards.shards[1].row_start += 1;
        assert_eq!(
            resolved_physical_execution_islands(
                &[producer.clone(), different_shards],
                VulkanSharedResidentBufferRoute::SharedHost,
            )
                .unwrap()
                .len(),
            2
        );

        let mut row_distributed = consumer;
        row_distributed.distribution = VulkanDistributedDispatchDistribution::OutputRows;
        assert_eq!(
            resolved_physical_execution_islands(
                &[producer, row_distributed],
                VulkanSharedResidentBufferRoute::SharedHost,
            )
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn contract_declared_output_rows_flow_into_local_input_column_shards() {
        let mut producer = fixture_plan("row_major").dispatches.remove(0);
        producer.dispatch_index = 7;
        producer.output_activation.signal_id = "activated".to_string();
        producer.local_intermediates = vec![
            nerve_execution_contracts::LocalIntermediateContract {
                signal: producer.output_activation.signal_id.clone(),
                producer_binding: u32::try_from(producer.output_activation.binding).unwrap(),
                consumer_binding: 0,
                format: "bf16".to_string(),
            },
        ];
        let mut consumer = producer.clone();
        consumer.dispatch_index = 8;
        consumer.node_id = "down".to_string();
        consumer.distribution = VulkanDistributedDispatchDistribution::InputColumns;
        consumer.input_distribution = InputDistribution::Sharded;
        consumer.input_activation = producer.output_activation.clone();
        consumer.input_activation.binding = 0;
        consumer.output_activation.signal_id = "hidden".to_string();
        consumer.output_activation.slot += 1;
        for (producer_shard, consumer_shard) in
            producer.shards.iter().zip(&mut consumer.shards)
        {
            consumer_shard.input_range.byte_offset = producer_shard.output_byte_offset;
            consumer_shard.input_range.byte_count = producer_shard.output_byte_count;
            consumer_shard.base_workgroup_z =
                u32::try_from(consumer_shard.row_start).unwrap();
        }

        let mut islands = resolved_physical_execution_islands(
            &[producer.clone(), consumer.clone()],
            VulkanSharedResidentBufferRoute::SharedHost,
        )
        .unwrap();
        assert_eq!(islands.len(), 1);
        assert_eq!(islands[0].dispatch_indices(), [7, 8]);
        let private_requirements = islands[0]
            .transient_memory
            .iter()
            .filter(|requirement| {
                requirement.kind
                    == VulkanPhysicalExecutionTransientMemoryKind::PrivateShardIntermediate
            })
            .collect::<Vec<_>>();
        assert_eq!(private_requirements.len(), producer.shards.len());
        assert_eq!(
            private_requirements
                .iter()
                .map(|requirement| requirement.per_lane_byte_capacity)
                .sum::<usize>(),
            producer
                .shards
                .iter()
                .map(|shard| shard.output_byte_count)
                .sum::<usize>()
        );
        assert!(islands[0].transient_memory.iter().all(|requirement| {
            requirement.kind
                != VulkanPhysicalExecutionTransientMemoryKind::SharedActivationAllocation
                || !requirement
                    .resource_id
                    .contains(&format!("slot_{}", producer.output_activation.slot))
        }));
        assert_eq!(
            islands[0]
                .phase_schedules[0]
                .steps
                .iter()
                .map(|step| step.kind)
                .collect::<Vec<_>>(),
            [
                VulkanPhysicalExecutionScheduleKind::PublishInputs,
                VulkanPhysicalExecutionScheduleKind::ExecuteShards,
                VulkanPhysicalExecutionScheduleKind::ExecuteShards,
                VulkanPhysicalExecutionScheduleKind::CollectOutputs,
            ]
        );
        let execution_plan = VulkanDistributedExecutionPlan {
            device_ids: producer
                .shards
                .iter()
                .map(|shard| shard.device_id.clone())
                .collect(),
            storage_buffer_offset_alignment: 4,
            dispatches: vec![producer.clone(), consumer.clone()],
            execution_islands: islands.clone(),
            shared_activation_route: VulkanSharedResidentBufferRoute::SharedHost,
            shared_input_byte_capacity: producer.input_byte_capacity,
            shared_output_byte_capacity: consumer.output_byte_capacity,
            distributed_parameter_byte_count: producer
                .distributed_parameter_byte_count
                + consumer.distributed_parameter_byte_count,
        };
        let activation_plan =
            VulkanDistributedActivationBufferPlan::from_execution_plan(&execution_plan)
                .unwrap();
        assert_eq!(activation_plan.private_intermediate_allocations.len(), 1);
        let private = &activation_plan.private_intermediate_allocations[0];
        assert_eq!(private.producer_dispatch_index, 7);
        assert_eq!(private.consumer_dispatch_index, 8);
        assert_eq!(private.signal_id, producer.output_activation.signal_id);
        assert_eq!(private.devices.len(), producer.shards.len());
        assert_eq!(
            activation_plan.total_private_byte_capacity,
            producer
                .shards
                .iter()
                .map(|shard| shard.output_byte_count)
                .sum::<usize>()
        );
        assert!(activation_plan.allocations.iter().all(|allocation| {
            !allocation
                .signal_ids
                .contains(&producer.output_activation.signal_id)
        }), "shared allocations retained private signal: {:?}", activation_plan.allocations);
        islands[0].dispatches[1].reduction = Some(VulkanDistributedReductionPlan {
            operation: ReductionOperation::SumF32,
            element_count: 12,
            partial_byte_capacity: 48,
            finalization: VulkanDistributedReductionFinalizationPlan::StoreF32,
        });
        assert_eq!(
            physical_island_reduction_dispatch(&islands[0])
                .unwrap()
                .unwrap()
                .dispatch_index,
            8
        );
        islands[0].dispatches[0].reduction = islands[0].dispatches[1].reduction.clone();
        assert!(
            physical_island_reduction_dispatch(&islands[0])
                .unwrap_err()
                .to_string()
                .contains("only one tail reduction is legal")
        );

        let mut undeclared = producer.clone();
        undeclared.local_intermediates.clear();
        assert_eq!(
            resolved_physical_execution_islands(
                &[undeclared, consumer.clone()],
                VulkanSharedResidentBufferRoute::SharedHost,
            )
            .unwrap()
            .len(),
            2
        );

        let mut format_mismatch = consumer.clone();
        format_mismatch.local_intermediates[0].format = "f32".to_string();
        assert_eq!(
            resolved_physical_execution_islands(
                &[producer.clone(), format_mismatch],
                VulkanSharedResidentBufferRoute::SharedHost,
            )
            .unwrap()
            .len(),
            2
        );

        let mut mismatched_range = consumer;
        mismatched_range.shards[1].input_range.byte_offset += 2;
        assert_eq!(
            resolved_physical_execution_islands(
                &[producer, mismatched_range],
                VulkanSharedResidentBufferRoute::SharedHost,
            )
            .unwrap()
            .len(),
            2
        );
    }

    #[test]
    fn distributed_shards_always_start_with_the_dispatch_owner() {
        let tensor_index = fixture_tensor_index("row_major");
        let prepared_plan = fixture_prepared_plan();
        let artifact_manifest = fixture_artifact_manifest();
        let plan = VulkanDistributedExecutionPlan::from_prepared_plans(
            &[("owner", &prepared_plan)],
            &tensor_index,
            &artifact_manifest,
            &component_device_pools("component", &["helper-a", "helper-b", "owner"]),
            &[],
            4,
        )
        .unwrap();

        assert_eq!(plan.dispatches.len(), 1);
        assert_eq!(plan.dispatches[0].shards[0].device_id, "owner");
        assert!(
            plan.dispatches[0]
                .shards
                .iter()
                .any(|shard| shard.device_id != "owner")
        );
    }

    #[test]
    fn distributed_planner_rejects_requested_unsplittable_dispatch() {
        let tensor_index = fixture_tensor_index("row_major");
        let prepared_plan = fixture_prepared_plan();
        let artifact_manifest = fixture_artifact_manifest();
        let plan = VulkanDistributedExecutionPlan::from_prepared_plans(
            &[("owner", &prepared_plan)],
            &tensor_index,
            &artifact_manifest,
            &component_device_pools("component", &["helper", "owner"]),
            &[],
            1024,
        )
        .unwrap_err();

        assert!(
            plan.to_string()
                .contains("has no compatible distributable dispatch")
        );
    }

    #[test]
    fn requested_distribution_without_a_dispatch_contract_fails_closed() {
        let mut prepared_plan = fixture_prepared_plan();
        prepared_plan.dispatches[0].physical_execution_contracts.clear();

        let error = VulkanDistributedExecutionPlan::from_prepared_plans(
            &[("owner", &prepared_plan)],
            &fixture_tensor_index("row_major"),
            &fixture_artifact_manifest(),
            &component_device_pools("component", &["owner", "helper"]),
            &[],
            4,
        )
        .unwrap_err();

        assert!(error
            .to_string()
            .contains("has no compatible distributable dispatch"));
    }

    #[test]
    fn ambiguous_dispatch_distribution_contracts_are_rejected() {
        let mut prepared_plan = fixture_prepared_plan();
        let mut duplicate = prepared_plan.dispatches[0].physical_execution_contracts[0].clone();
        duplicate.contract_id = format!("sha256:{}", "d".repeat(64));
        prepared_plan.dispatches[0]
            .physical_execution_contracts
            .push(duplicate);
        let mut artifact_manifest = fixture_artifact_manifest();
        let mut duplicate_artifact = artifact_manifest.artifacts[0].clone();
        duplicate_artifact.artifact_id = physical_execution_artifact_id(
            &format!("sha256:{}", "d".repeat(64)),
            0,
        );
        artifact_manifest.artifacts.push(duplicate_artifact);

        let error = VulkanDistributedExecutionPlan::from_prepared_plans(
            &[("owner", &prepared_plan)],
            &fixture_tensor_index("row_major"),
            &artifact_manifest,
            &component_device_pools("component", &["owner", "helper"]),
            &[],
            4,
        )
        .unwrap_err();

        assert!(error.to_string().contains("ambiguous decode distribution contracts"));
    }

    #[test]
    fn an_unseen_operation_family_uses_its_compiled_contract_without_runtime_changes() {
        let mut prepared_plan = fixture_prepared_plan();
        prepared_plan.dispatches[0].op = "future_fused_projection".to_string();
        let mut artifacts = fixture_artifact_manifest();
        for artifact in &mut artifacts.artifacts {
            artifact.op = "future_fused_projection".to_string();
        }
        prepared_plan.dispatches[0].physical_execution_contracts[0].operation_family =
            "future_fused_projection".to_string();

        let plan = VulkanDistributedExecutionPlan::from_prepared_plans(
            &[("owner", &prepared_plan)],
            &fixture_tensor_index("row_major"),
            &artifacts,
            &component_device_pools("component", &["owner", "helper"]),
            &[],
            4,
        )
        .unwrap();

        assert_eq!(plan.dispatches.len(), 1);
        assert_eq!(plan.dispatches[0].shards.len(), 2);
    }

    #[test]
    fn a_stale_contract_cannot_partially_cover_the_artifact_abi() {
        let mut prepared_plan = fixture_prepared_plan();
        prepared_plan.dispatches[0].physical_execution_contracts[0]
            .parameter_partitions
            .pop();

        let error = VulkanDistributedExecutionPlan::from_prepared_plans(
            &[("owner", &prepared_plan)],
            &fixture_tensor_index("row_major"),
            &fixture_artifact_manifest(),
            &component_device_pools("component", &["owner", "helper"]),
            &[],
            4,
        )
        .unwrap_err();

        assert!(error.to_string().contains("disagree with artifact ABI"));
    }

    #[test]
    fn preserves_packed_row_pairs_at_shard_boundaries() {
        let plan = fixture_plan("vulkan_bf16_row_pair_u32");

        assert_eq!(plan.dispatches[0].row_alignment, 2);
        assert!(
            plan.dispatches[0]
                .shards
                .iter()
                .all(|shard| shard.row_start % 2 == 0 && shard.row_count % 2 == 0)
        );
    }

    #[test]
    fn aligns_shared_output_offsets_and_keeps_a_workgroup_aligned_tail() {
        let plan = fixture_plan_result_with_alignment("row_major", 16).unwrap();
        let dispatch = &plan.dispatches[0];

        assert_eq!(dispatch.row_alignment, 8);
        assert_eq!(
            dispatch
                .shards
                .iter()
                .map(|shard| (
                    shard.device_id.as_str(),
                    shard.row_start,
                    shard.row_count,
                    shard.workgroup_count_x,
                    shard.output_byte_offset,
                ))
                .collect::<Vec<_>>(),
            vec![("owner", 0, 8, 4, 0), ("helper-a", 8, 4, 2, 16)]
        );
        assert!(dispatch.shards.iter().all(|shard| {
            shard
                .output_byte_offset
                .is_multiple_of(plan.storage_buffer_offset_alignment)
        }));
    }

    #[test]
    fn plans_one_shared_allocation_per_owner_activation_slot() {
        let execution_plan = fixture_plan("row_major");

        let activation_plan =
            VulkanDistributedActivationBufferPlan::from_execution_plan(&execution_plan).unwrap();

        assert_eq!(activation_plan.allocation_count, 2);
        assert_eq!(activation_plan.import_count, 8);
        assert_eq!(activation_plan.reference_count, 2);
        assert_eq!(activation_plan.total_shared_byte_capacity, 32);
        assert_eq!(
            activation_plan.allocation("owner", "component", 0).unwrap(),
            &VulkanDistributedActivationBufferAllocation {
                storage: VulkanDistributedActivationStorage::ActivationSlot,
                owner_device_id: "owner".to_string(),
                component_id: "component".to_string(),
                slot: 0,
                byte_capacity: 8,
                signal_ids: vec!["normalized".to_string()],
                device_ids: vec![
                    "helper-a".to_string(),
                    "helper-b".to_string(),
                    "helper-c".to_string(),
                    "owner".to_string(),
                ],
                input_use_count: 1,
                output_use_count: 0,
            }
        );
        assert_eq!(
            activation_plan
                .allocation("owner", "component", 1)
                .unwrap()
                .output_use_count,
            1
        );
    }

    #[test]
    fn rejects_zero_lane_shared_activation_allocations_before_device_access() {
        let execution_plan = fixture_plan("row_major");
        let activation_plan =
            VulkanDistributedActivationBufferPlan::from_execution_plan(&execution_plan).unwrap();

        let error =
            VulkanDistributedActivationBuffers::allocate_for_lanes(&activation_plan, 0, |_| {
                Err::<&VulkanComputeDevice, _>("device resolver must not run")
            })
            .err()
            .unwrap();

        assert_eq!(
            error.to_string(),
            "distributed activation lane capacity must not be zero"
        );
    }

    #[test]
    fn reuses_shared_activation_allocations_across_repeated_dispatches() {
        let mut execution_plan = fixture_plan("row_major");
        let mut repeated = execution_plan.dispatches[0].clone();
        repeated.dispatch_index = 8;
        repeated.input_activation.signal_id = "normalized-again".to_string();
        execution_plan.dispatches.push(repeated);

        let activation_plan =
            VulkanDistributedActivationBufferPlan::from_execution_plan(&execution_plan).unwrap();

        assert_eq!(activation_plan.allocation_count, 2);
        assert_eq!(activation_plan.import_count, 8);
        assert_eq!(activation_plan.reference_count, 4);
        assert_eq!(activation_plan.total_shared_byte_capacity, 32);
        let input = activation_plan.allocation("owner", "component", 0).unwrap();
        assert_eq!(input.input_use_count, 2);
        assert_eq!(
            input.signal_ids,
            vec!["normalized".to_string(), "normalized-again".to_string()]
        );
    }

    #[test]
    fn deduplicates_distributed_component_boundaries_by_graph_edge() {
        let mut execution_plan = fixture_plan("row_major");
        let edge_storage = VulkanDistributedActivationStorage::Edge {
            edge_index: 7,
            owner_device_id: "owner".to_string(),
        };
        execution_plan.dispatches[0].output_activation.storage = edge_storage.clone();
        let mut consumer = execution_plan.dispatches[0].clone();
        consumer.dispatch_index = 8;
        consumer.component_id = "consumer".to_string();
        consumer.input_activation = execution_plan.dispatches[0].output_activation.clone();
        consumer.input_activation.component_id = "consumer".to_string();
        consumer.input_activation.signal_id = "consumer_input".to_string();
        consumer.output_activation.component_id = "consumer".to_string();
        consumer.output_activation.slot = 4;
        consumer.output_activation.signal_id = "consumer_output".to_string();
        consumer.output_activation.storage =
            VulkanDistributedActivationStorage::ActivationSlot;
        execution_plan.dispatches.push(consumer);

        let activation_plan =
            VulkanDistributedActivationBufferPlan::from_execution_plan(&execution_plan).unwrap();

        assert_eq!(activation_plan.allocation_count, 3);
        let edge = activation_plan.edge_allocation(7).unwrap();
        assert_eq!(edge.owner_device_id, "owner");
        assert_eq!(edge.input_use_count, 1);
        assert_eq!(edge.output_use_count, 1);
        assert_eq!(
            edge.signal_ids,
            vec!["consumer_input".to_string(), "hidden".to_string()]
        );
    }

    #[test]
    fn rejects_conflicting_capacities_for_the_same_activation_slot() {
        let mut execution_plan = fixture_plan("row_major");
        let mut repeated = execution_plan.dispatches[0].clone();
        repeated.dispatch_index = 8;
        repeated.input_activation.byte_capacity = 16;
        execution_plan.dispatches.push(repeated);

        let error = VulkanDistributedActivationBufferPlan::from_execution_plan(&execution_plan)
            .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("activation component.slot_0 has conflicting capacities 8 and 16")
        );
    }

    #[test]
    fn rejects_non_contiguous_projection_layouts() {
        let error = fixture_plan_result("column_major").unwrap_err();

        assert!(
            error
                .to_string()
                .contains("tensor \"gate\" has non-shardable layout Some(\"column_major\")")
        );
    }

    #[test]
    fn rejects_physical_artifact_push_constants_outside_the_contract() {
        let prepared_plan = fixture_prepared_plan();
        let mut artifact_manifest = fixture_artifact_manifest();
        artifact_manifest.artifacts[0].push_constants = vec![VulkanKernelScalarBinding {
            name: "stream_tick".to_string(),
            scalar_type: "u64".to_string(),
            source: VulkanKernelScalarSource::PushConstant,
        }];

        let plan = VulkanDistributedExecutionPlan::from_prepared_plans(
            &[("owner", &prepared_plan)],
            &fixture_tensor_index("row_major"),
            &artifact_manifest,
            &component_device_pools("component", &["owner", "helper"]),
            &[],
            4,
        )
        .unwrap_err();

        assert!(plan
            .to_string()
            .contains("local-zero partition contract forbids push constants"));
    }

    #[test]
    fn plans_block_scaled_fp8_parallel_projection_ranges() {
        let mut prepared_plan = fixture_prepared_plan();
        let activation =
            |binding, usage, signal: &str, slot, bytes| VulkanResolvedDescriptorBinding {
                binding,
                usage,
                name: signal.to_string(),
                resource: VulkanDescriptorResourceAddress::ActivationSlot {
                    component_id: "component".to_string(),
                    signal_id: signal.to_string(),
                    slot,
                    byte_capacity: bytes,
                    signal_byte_capacity: bytes,
                },
            };
        let parameter =
            |binding, param_id: &str, tensor: &str, byte_count| VulkanResolvedDescriptorBinding {
                binding,
                usage: VulkanKernelDescriptorUsage::Parameter,
                name: param_id.to_string(),
                resource: VulkanDescriptorResourceAddress::PermanentParameter {
                    param_id: param_id.to_string(),
                    tensor: tensor.to_string(),
                    byte_count: Some(byte_count),
                },
            };
        prepared_plan.dispatches[0].descriptors = vec![
            activation(
                0,
                VulkanKernelDescriptorUsage::InputSignal,
                "normalized_fp8",
                0,
                4,
            ),
            activation(
                1,
                VulkanKernelDescriptorUsage::InputSignal,
                "normalized_scale",
                1,
                4,
            ),
            activation(
                2,
                VulkanKernelDescriptorUsage::OutputSignal,
                "hidden",
                2,
                24,
            ),
            parameter(3, "gate", "gate", 48),
            parameter(4, "gate_scale", "gate_scale", 6),
            parameter(5, "up", "up", 48),
            parameter(6, "up_scale", "up_scale", 6),
        ];
        prepared_plan.dispatches[0].physical_execution_contracts = vec![
            test_physical_contract(
                "parallel_linear_silu_multiply",
                "ffn",
                "ffn.spv",
                ExecutionStrategy::TensorParallel,
                ExecutionForm::ReplicatedInputPartitionedOutput,
                12,
                4,
                6,
                WorkgroupXMapping::Proportional,
                PartitionOrigin::LocalZero,
                None,
                None,
                vec![
                    test_partition("gate", 3, ParameterPartitionKind::Contiguous, 4, 1),
                    test_partition(
                        "gate_scale",
                        4,
                        ParameterPartitionKind::Contiguous,
                        1,
                        4,
                    ),
                    test_partition("up", 5, ParameterPartitionKind::Contiguous, 4, 1),
                    test_partition(
                        "up_scale",
                        6,
                        ParameterPartitionKind::Contiguous,
                        1,
                        4,
                    ),
                ],
                vec![
                    test_input(0, InputDistribution::Replicated, None),
                    test_input(1, InputDistribution::Replicated, None),
                ],
                test_output(2, OutputCollection::Concatenated, Some(2)),
            ),
        ];
        let mut tensor_index = fixture_tensor_index("row_major");
        for tensor in ["gate", "up"] {
            let metadata = tensor_index.tensors.get_mut(tensor).unwrap();
            metadata.dtype = "F8_E4M3".to_string();
            metadata.byte_count = Some(48);
        }
        let scale = TensorMetadata {
            dtype: "BF16".to_string(),
            shape: vec![3, 1],
            logical_shape: None,
            parameter_count: Some(3),
            byte_count: Some(6),
            data_offsets: Some(vec![0, 6]),
            source_file: Some("weights.safetensors".to_string()),
            data_sha256: None,
            layout: Some("row_major".to_string()),
        };
        tensor_index
            .tensors
            .insert("gate_scale".to_string(), scale.clone());
        tensor_index.tensors.insert("up_scale".to_string(), scale);

        let artifacts = test_artifact_manifest(
            "family",
            "parallel_linear_silu_multiply",
            "ffn.spv",
            6,
        );
        let plan = VulkanDistributedExecutionPlan::from_prepared_plans(
            &[("owner", &prepared_plan)],
            &tensor_index,
            &artifacts,
            &component_device_pools("component", &["owner", "helper"]),
            &[],
            4,
        )
        .unwrap();

        assert_eq!(plan.distributed_parameter_byte_count, 108);
        let dispatch = &plan.dispatches[0];
        assert_eq!(dispatch.row_alignment, 4);
        assert_eq!(dispatch.input_byte_capacity, 4);
        assert_eq!(dispatch.output_byte_capacity, 24);
        assert_eq!(dispatch.auxiliary_input_activations.len(), 1);
        assert_eq!(
            dispatch
                .shards
                .iter()
                .map(|shard| (
                    shard.device_id.as_str(),
                    shard.row_start,
                    shard.row_count,
                    shard.output_byte_offset,
                    shard.output_byte_count,
                ))
                .collect::<Vec<_>>(),
            vec![("owner", 0, 8, 0, 16), ("helper", 8, 4, 16, 8)]
        );
        assert_eq!(
            dispatch.shards[1].auxiliary_input_ranges,
            vec![VulkanDistributedActivationRange {
                byte_offset: 0,
                byte_count: 4,
            }]
        );
        assert_eq!(
            dispatch.shards[1]
                .parameters
                .iter()
                .map(|fragment| (
                    fragment.binding,
                    fragment.tensor.as_str(),
                    fragment.byte_offset,
                    fragment.byte_count,
                ))
                .collect::<Vec<_>>(),
            vec![
                (3, "gate", 32, 16),
                (4, "gate_scale", 4, 2),
                (5, "up", 32, 16),
                (6, "up_scale", 4, 2),
            ]
        );
    }

    #[test]
    fn plans_block_scaled_fp8_residual_projection_with_strided_residual_ranges() {
        let activation =
            |binding, usage, signal: &str, slot, bytes| VulkanResolvedDescriptorBinding {
                binding,
                usage,
                name: signal.to_string(),
                resource: VulkanDescriptorResourceAddress::ActivationSlot {
                    component_id: "component".to_string(),
                    signal_id: signal.to_string(),
                    slot,
                    byte_capacity: bytes,
                    signal_byte_capacity: bytes,
                },
            };
        let parameter =
            |binding, tensor: &str, byte_count| VulkanResolvedDescriptorBinding {
                binding,
                usage: VulkanKernelDescriptorUsage::Parameter,
                name: tensor.to_string(),
                resource: VulkanDescriptorResourceAddress::PermanentParameter {
                    param_id: tensor.to_string(),
                    tensor: tensor.to_string(),
                    byte_count: Some(byte_count),
                },
            };
        let prepared = VulkanPreparedDispatchPlan {
            backend_id: "vulkan_stream_circuit".to_string(),
            reusable_family_count: 1,
            dispatches: vec![VulkanPreparedDispatch {
                dispatch_index: 9,
                kernel_id: "component.residual".to_string(),
                component_id: "component".to_string(),
                circuit_id: "circuit".to_string(),
                node_index: 4,
                node_id: "residual".to_string(),
                op: "linear_residual".to_string(),
                reusable_family_id: "residual-family".to_string(),
                artifact_path: "residual.spv".to_string(),
                entry_point: "main".to_string(),
                local_size_x: 64,
                descriptors: vec![
                    activation(
                        0,
                        VulkanKernelDescriptorUsage::InputSignal,
                        "hidden_fp8",
                        0,
                        4,
                    ),
                    activation(
                        1,
                        VulkanKernelDescriptorUsage::InputSignal,
                        "hidden_scale",
                        1,
                        4,
                    ),
                    activation(
                        2,
                        VulkanKernelDescriptorUsage::InputSignal,
                        "residual",
                        2,
                        24,
                    ),
                    activation(
                        3,
                        VulkanKernelDescriptorUsage::OutputSignal,
                        "output",
                        3,
                        24,
                    ),
                    parameter(4, "weight", 48),
                    parameter(5, "weight_scale", 6),
                ],
                push_constants: Vec::new(),
                stream_control_binding: None,
                physical_execution_contracts: vec![test_physical_contract(
                    "linear_residual",
                    "residual",
                    "residual.spv",
                    ExecutionStrategy::TensorParallel,
                    ExecutionForm::ReplicatedInputPartitionedOutput,
                    12,
                    4,
                    6,
                    WorkgroupXMapping::Proportional,
                    PartitionOrigin::LocalZero,
                    None,
                    None,
                    vec![
                        test_partition(
                            "weight",
                            4,
                            ParameterPartitionKind::Contiguous,
                            4,
                            1,
                        ),
                        test_partition(
                            "weight_scale",
                            5,
                            ParameterPartitionKind::Contiguous,
                            1,
                            4,
                        ),
                    ],
                    vec![
                        test_input(0, InputDistribution::Replicated, None),
                        test_input(1, InputDistribution::Replicated, None),
                        test_input(2, InputDistribution::Sharded, Some(2)),
                    ],
                    test_output(3, OutputCollection::Concatenated, Some(2)),
                )],
            }],
            total_descriptor_count: 6,
        };
        let weight = TensorMetadata {
            dtype: "F8_E4M3".to_string(),
            shape: vec![12, 4],
            logical_shape: None,
            parameter_count: Some(48),
            byte_count: Some(48),
            data_offsets: Some(vec![0, 48]),
            source_file: Some("weights.safetensors".to_string()),
            data_sha256: None,
            layout: Some("row_major".to_string()),
        };
        let scale = TensorMetadata {
            dtype: "BF16".to_string(),
            shape: vec![3, 1],
            logical_shape: None,
            parameter_count: Some(3),
            byte_count: Some(6),
            data_offsets: Some(vec![0, 6]),
            source_file: Some("weights.safetensors".to_string()),
            data_sha256: None,
            layout: Some("row_major".to_string()),
        };
        let tensor_index = TensorIndex {
            schema: "nerve.tensor_index.v1".to_string(),
            tensors: BTreeMap::from([
                ("weight".to_string(), weight),
                ("weight_scale".to_string(), scale),
            ]),
        };
        let artifacts = test_artifact_manifest_with_physical(VulkanReusableKernelArtifact {
                family_id: "residual-family".to_string(),
                op: "linear_residual".to_string(),
                path: "residual.spv".to_string(),
                entry_point: "main".to_string(),
                local_size_x: 64,
                workgroup_count_x: 6,
                descriptor_signature: Vec::new(),
                push_constants: Vec::new(),
                stream_control_binding: None,
            });

        let plan = VulkanDistributedExecutionPlan::from_prepared_plans(
            &[("owner", &prepared)],
            &tensor_index,
            &artifacts,
            &component_device_pools("component", &["owner", "helper"]),
            &[],
            4,
        )
        .unwrap();

        assert_eq!(plan.distributed_parameter_byte_count, 54);
        let dispatch = &plan.dispatches[0];
        assert_eq!(dispatch.auxiliary_input_activations.len(), 2);
        assert_eq!(
            dispatch.shards[1].auxiliary_input_ranges,
            vec![
                VulkanDistributedActivationRange {
                    byte_offset: 0,
                    byte_count: 4,
                },
                VulkanDistributedActivationRange {
                    byte_offset: 16,
                    byte_count: 8,
                },
            ]
        );
        assert_eq!(
            dispatch.shards[1]
                .parameters
                .iter()
                .map(|fragment| (
                    fragment.binding,
                    fragment.tensor.as_str(),
                    fragment.byte_offset,
                    fragment.byte_count,
                ))
                .collect::<Vec<_>>(),
            vec![(4, "weight", 32, 16), (5, "weight_scale", 4, 2)]
        );
    }

    #[test]
    fn immutable_parameter_shards_are_reused_by_duplicated_components() {
        let mut execution_plan = fixture_plan("row_major");
        let mut duplicate = execution_plan.dispatches[0].clone();
        duplicate.dispatch_index = 8;
        duplicate.component_id = "duplicated-component".to_string();
        duplicate.node_id = "duplicated-ffn".to_string();
        execution_plan.dispatches.push(duplicate);

        let allocation_plan = VulkanDistributedParameterAllocationPlan::from_execution_plan(
            &execution_plan,
            &fixture_tensor_index("row_major"),
        )
        .unwrap();

        assert_eq!(allocation_plan.allocation_count, 8);
        assert_eq!(allocation_plan.tensor_count, 2);
        assert_eq!(allocation_plan.total_byte_capacity, 192);
        assert!(
            allocation_plan
                .allocations
                .iter()
                .all(|allocation| allocation.use_count == 2)
        );
    }

    #[test]
    fn loads_each_tensor_once_and_streams_verified_shards_to_devices() {
        let execution_plan = fixture_plan("row_major");
        let fixture = DistributedStorageFixture::new();
        let allocation_plan = VulkanDistributedParameterAllocationPlan::from_execution_plan(
            &execution_plan,
            &fixture.tensor_index,
        )
        .unwrap();
        let mut writes = Vec::new();

        let report = allocation_plan
            .load_from_tensor_index(&fixture.tensor_index, |allocation, bytes| {
                writes.push((allocation.clone(), bytes.to_vec()));
                Ok(())
            })
            .unwrap();

        assert_eq!(report.tensor_count, 2);
        assert_eq!(report.source_file_count, 1);
        assert_eq!(report.allocation_count, 8);
        assert_eq!(report.write_count, 8);
        assert_eq!(report.total_bytes_read, 192);
        assert_eq!(report.total_bytes_written, 192);
        let (allocation, bytes) = writes
            .iter()
            .find(|(allocation, _)| {
                allocation.device_id == "helper-a" && allocation.tensor == "gate"
            })
            .unwrap();
        assert_eq!(allocation.byte_offset, 32);
        assert_eq!(allocation.byte_count, 32);
        assert_eq!(bytes, &fixture.gate_bytes[32..64]);
    }

    #[test]
    fn excludes_full_parameters_only_when_all_prepared_uses_are_distributed() {
        let execution_plan = fixture_plan("row_major");
        let prepared_plan = fixture_prepared_plan();

        let exclusions =
            VulkanDistributedParameterExclusionPlan::from_execution_and_prepared_plans(
                &execution_plan,
                &[("owner", &prepared_plan)],
                &fixture_tensor_index("row_major"),
            )
            .unwrap();

        assert_eq!(exclusions.device_count, 1);
        assert_eq!(exclusions.unique_tensor_count, 2);
        assert_eq!(exclusions.excluded_full_allocation_count, 2);
        assert_eq!(exclusions.excluded_full_byte_capacity, 192);
        assert_eq!(
            exclusions.tensors_for_device("owner"),
            BTreeSet::from(["gate".to_string(), "up".to_string()])
        );
        assert!(exclusions.tensors_for_device("helper-a").is_empty());
    }

    #[test]
    fn refuses_to_exclude_a_tensor_still_used_by_a_canonical_dispatch() {
        let execution_plan = fixture_plan("row_major");
        let mut prepared_plan = fixture_prepared_plan();
        let mut canonical = prepared_plan.dispatches[0].clone();
        canonical.dispatch_index = 8;
        canonical.node_index = 4;
        canonical.node_id = "canonical-use".to_string();
        canonical.op = "linear".to_string();
        canonical.descriptors.retain(|descriptor| {
            matches!(
                descriptor.resource,
                VulkanDescriptorResourceAddress::PermanentParameter { .. }
            )
        });
        prepared_plan.dispatches.push(canonical);

        let error = VulkanDistributedParameterExclusionPlan::from_execution_and_prepared_plans(
            &execution_plan,
            &[("owner", &prepared_plan)],
            &fixture_tensor_index("row_major"),
        )
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("canonical dispatch component.canonical-use still uses it")
        );
    }

    fn fixture_plan(layout: &str) -> VulkanDistributedExecutionPlan {
        fixture_plan_result(layout).unwrap()
    }

    fn fixture_plan_result(
        layout: &str,
    ) -> Result<VulkanDistributedExecutionPlan, VulkanDistributedPlanError> {
        fixture_plan_result_with_alignment(layout, 4)
    }

    fn fixture_plan_result_with_alignment(
        layout: &str,
        storage_buffer_offset_alignment: usize,
    ) -> Result<VulkanDistributedExecutionPlan, VulkanDistributedPlanError> {
        let tensor_index = fixture_tensor_index(layout);
        let prepared_plan = fixture_prepared_plan();
        let artifact_manifest = fixture_artifact_manifest();
        VulkanDistributedExecutionPlan::from_prepared_plans(
            &[("owner", &prepared_plan)],
            &tensor_index,
            &artifact_manifest,
            &component_device_pools(
                "component",
                &["owner", "helper-a", "helper-b", "helper-c"],
            ),
            &[],
            storage_buffer_offset_alignment,
        )
    }

    fn component_device_pools(
        component_id: &str,
        device_ids: &[&str],
    ) -> BTreeMap<String, Vec<String>> {
        BTreeMap::from([(
            component_id.to_string(),
            device_ids
                .iter()
                .map(|device_id| (*device_id).to_string())
                .collect(),
        )])
    }

    fn fixture_prepared_plan() -> VulkanPreparedDispatchPlan {
        let activation =
            |binding, name: &str, signal: &str, bytes| VulkanResolvedDescriptorBinding {
                binding,
                usage: if binding == 0 {
                    VulkanKernelDescriptorUsage::InputSignal
                } else {
                    VulkanKernelDescriptorUsage::OutputSignal
                },
                name: name.to_string(),
                resource: VulkanDescriptorResourceAddress::ActivationSlot {
                    component_id: "component".to_string(),
                    signal_id: signal.to_string(),
                    slot: binding,
                    byte_capacity: bytes,
                    signal_byte_capacity: bytes,
                },
            };
        let parameter = |binding, tensor: &str| VulkanResolvedDescriptorBinding {
            binding,
            usage: VulkanKernelDescriptorUsage::Parameter,
            name: tensor.to_string(),
            resource: VulkanDescriptorResourceAddress::PermanentParameter {
                param_id: tensor.to_string(),
                tensor: tensor.to_string(),
                byte_count: Some(96),
            },
        };
        VulkanPreparedDispatchPlan {
            backend_id: "vulkan_stream_circuit".to_string(),
            reusable_family_count: 1,
            dispatches: vec![VulkanPreparedDispatch {
                dispatch_index: 7,
                kernel_id: "component.ffn".to_string(),
                component_id: "component".to_string(),
                circuit_id: "circuit".to_string(),
                node_index: 3,
                node_id: "ffn".to_string(),
                op: "parallel_linear_silu_multiply".to_string(),
                reusable_family_id: "family".to_string(),
                artifact_path: "ffn.spv".to_string(),
                entry_point: "main".to_string(),
                local_size_x: 64,
                descriptors: vec![
                    activation(0, "input", "normalized", 8),
                    activation(1, "output", "hidden", 24),
                    parameter(2, "gate"),
                    parameter(3, "up"),
                ],
                push_constants: Vec::new(),
                stream_control_binding: None,
                physical_execution_contracts: vec![test_physical_contract(
                    "parallel_linear_silu_multiply",
                    "ffn",
                    "ffn.spv",
                    ExecutionStrategy::TensorParallel,
                    ExecutionForm::ReplicatedInputPartitionedOutput,
                    12,
                    2,
                    6,
                    WorkgroupXMapping::Proportional,
                    PartitionOrigin::LocalZero,
                    None,
                    None,
                    vec![
                        test_partition("gate", 2, ParameterPartitionKind::Contiguous, 2, 1),
                        test_partition("up", 3, ParameterPartitionKind::Contiguous, 2, 1),
                    ],
                    vec![test_input(0, InputDistribution::Replicated, None)],
                    test_output(1, OutputCollection::Concatenated, Some(2)),
                )],
            }],
            total_descriptor_count: 4,
        }
    }

    fn fixture_artifact_manifest() -> VulkanPhysicalKernelArtifactManifest {
        test_artifact_manifest(
            "family",
            "parallel_linear_silu_multiply",
            "ffn.spv",
            6,
        )
    }

    fn test_artifact_manifest(
        family_id: &str,
        op: &str,
        path: &str,
        workgroup_count_x: u32,
    ) -> VulkanPhysicalKernelArtifactManifest {
        test_artifact_manifest_with_physical(VulkanReusableKernelArtifact {
            family_id: family_id.to_string(),
            op: op.to_string(),
            path: path.to_string(),
            entry_point: "main".to_string(),
            local_size_x: 64,
            workgroup_count_x,
            descriptor_signature: Vec::new(),
            push_constants: Vec::new(),
            stream_control_binding: None,
        })
    }

    fn test_artifact_manifest_with_physical(
        canonical: VulkanReusableKernelArtifact,
    ) -> VulkanPhysicalKernelArtifactManifest {
        VulkanPhysicalKernelArtifactManifest::new(vec![VulkanPhysicalKernelArtifact {
            artifact_id: physical_execution_artifact_id(
                &format!("sha256:{}", "a".repeat(64)),
                0,
            ),
            op: canonical.op,
            path: canonical.path,
            entry_point: canonical.entry_point,
            local_size_x: canonical.local_size_x,
            workgroup_count_x: canonical.workgroup_count_x,
            descriptor_signature: canonical.descriptor_signature,
            push_constants: canonical.push_constants,
            stream_control_binding: canonical.stream_control_binding,
        }])
    }

    #[allow(clippy::too_many_arguments)]
    fn test_physical_contract(
        op: &str,
        node_id: &str,
        path: &str,
        strategy: ExecutionStrategy,
        execution_form: ExecutionForm,
        extent: u64,
        alignment: u64,
        workgroup_count_x: u32,
        workgroup_x: WorkgroupXMapping,
        origin: PartitionOrigin,
        origin_push_constant: Option<&str>,
        count_push_constant: Option<&str>,
        parameter_partitions: Vec<ParameterPartition>,
        inputs: Vec<InputContract>,
        output: OutputContract,
    ) -> PhysicalExecutionContract {
        let resources = parameter_partitions
            .iter()
            .map(|partition| ResourceRequirement {
                resource: partition.resource.clone(),
                kind: ResourceKind::PersistentParameter,
                residency: ResidencyRequirement::Permanent,
                access: ResourceAccess::Read,
                binding: Some(partition.binding),
                atomic_group: None,
            })
            .collect();
        PhysicalExecutionContract {
            schema: PHYSICAL_EXECUTION_CONTRACT_SCHEMA.to_string(),
            contract_id: format!("sha256:{}", "a".repeat(64)),
            operation_family: op.to_string(),
            region_family: None,
            member_node_ids: vec![node_id.to_string()],
            artifacts: vec![ArtifactIdentity {
                path: path.to_string(),
                sha256: format!("sha256:{}", "b".repeat(64)),
                entry_point: "main".to_string(),
            }],
            implementation_digest: format!("sha256:{}", "c".repeat(64)),
            phases: vec![ExecutionPhase::Decode],
            formats: PhysicalFormats {
                storage: "test".to_string(),
                compute: "test".to_string(),
                accumulation: "f32".to_string(),
            },
            geometry: ExecutionGeometry {
                shape_class: "test-shape".to_string(),
                dimensions: BTreeMap::from([
                    ("partition".to_string(), extent),
                    ("local_size_x".to_string(), 64),
                    ("workgroup_count_x".to_string(), u64::from(workgroup_count_x)),
                ]),
                dynamic_dimensions: Vec::new(),
            },
            strategy,
            execution_form,
            partition_extent: Some(PartitionExtent {
                dimension_name: "partition".to_string(),
                elements: extent,
                alignment_elements: alignment,
            }),
            partition_launch: Some(PartitionLaunch {
                workgroup_x,
                origin,
                origin_push_constant: origin_push_constant.map(str::to_string),
                count_push_constant: count_push_constant.map(str::to_string),
            }),
            parameter_partitions,
            inputs,
            outputs: vec![output],
            local_intermediates: Vec::new(),
            resources,
            equivalence: EquivalenceRequirement {
                output: EquivalenceKind::BitExact,
                state: EquivalenceKind::BitExact,
                absolute_tolerance: None,
                relative_tolerance: None,
            },
        }
    }

    fn test_partition(
        resource: &str,
        binding: u32,
        kind: ParameterPartitionKind,
        alignment_elements: u64,
        logical_elements_per_index: u64,
    ) -> ParameterPartition {
        ParameterPartition {
            binding,
            resource: resource.to_string(),
            dimension: 0,
            kind,
            alignment_elements,
            logical_elements_per_index,
        }
    }

    fn test_input(
        binding: u32,
        distribution: InputDistribution,
        alignment_elements: Option<u64>,
    ) -> InputContract {
        InputContract {
            binding,
            distribution,
            dimension: alignment_elements.map(|_| 0),
            alignment_elements,
        }
    }

    fn test_output(
        binding: u32,
        collection: OutputCollection,
        alignment_elements: Option<u64>,
    ) -> OutputContract {
        OutputContract {
            binding,
            collection,
            dimension: alignment_elements.map(|_| 0),
            alignment_elements,
            reduction: None,
        }
    }

    fn fixture_tensor_index(layout: &str) -> TensorIndex {
        let metadata = |layout: &str| TensorMetadata {
            dtype: "BF16".to_string(),
            shape: vec![12, 4],
            logical_shape: None,
            parameter_count: Some(48),
            byte_count: Some(96),
            data_offsets: Some(vec![0, 96]),
            source_file: Some("weights.safetensors".to_string()),
            data_sha256: None,
            layout: Some(layout.to_string()),
        };
        TensorIndex {
            schema: "nerve.tensor_index.v1".to_string(),
            tensors: BTreeMap::from([
                ("gate".to_string(), metadata(layout)),
                ("up".to_string(), metadata(layout)),
            ]),
        }
    }

    struct DistributedStorageFixture {
        root: PathBuf,
        tensor_index: TensorIndex,
        gate_bytes: Vec<u8>,
    }

    impl DistributedStorageFixture {
        fn new() -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let root = std::env::temp_dir().join(format!(
                "nerve-distributed-storage-{}-{unique}",
                std::process::id()
            ));
            fs::create_dir_all(&root).unwrap();
            let source = root.join("weights.safetensors");
            let gate_bytes = (0..96).map(|value| value as u8).collect::<Vec<_>>();
            let up_bytes = (0..96)
                .map(|value| 255u8.wrapping_sub(value as u8))
                .collect::<Vec<_>>();
            let header = b"{}";
            let mut file_bytes = Vec::new();
            file_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
            file_bytes.extend_from_slice(header);
            file_bytes.extend_from_slice(&gate_bytes);
            file_bytes.extend_from_slice(&up_bytes);
            fs::write(&source, file_bytes).unwrap();
            let metadata = |data_offsets: Vec<usize>, bytes: &[u8]| TensorMetadata {
                dtype: "BF16".to_string(),
                shape: vec![12, 4],
                logical_shape: None,
                parameter_count: Some(48),
                byte_count: Some(96),
                data_offsets: Some(data_offsets),
                source_file: Some(source.to_string_lossy().into_owned()),
                data_sha256: Some(
                    Sha256::digest(bytes)
                        .iter()
                        .map(|byte| format!("{byte:02x}"))
                        .collect(),
                ),
                layout: Some("row_major".to_string()),
            };
            let tensor_index = TensorIndex {
                schema: "nerve.tensor_index.v1".to_string(),
                tensors: BTreeMap::from([
                    ("gate".to_string(), metadata(vec![0, 96], &gate_bytes)),
                    ("up".to_string(), metadata(vec![96, 192], &up_bytes)),
                ]),
            };
            Self {
                root,
                tensor_index,
                gate_bytes,
            }
        }
    }

    impl Drop for DistributedStorageFixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }
}
