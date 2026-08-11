#[test]
fn kernel_interfaces_describe_fixture_model_compiled_component_abi() {
    let graph = fixture_model_execution_graph();
    let tensor_index = TensorIndex::from_json_file(fixture_model_tensor_index_path()).unwrap();
    let execution_plan =
        StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index).unwrap();
    let resource_plan =
        StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
    let resident_plan = VulkanStreamCircuitResidentPlan::from_resource_plan(
        &resource_plan,
        Some(&tensor_index),
        Some(2),
    )
    .unwrap();
    let binding_plan =
        VulkanStreamCircuitBindingPlan::from_plans(&execution_plan, &resource_plan, &resident_plan)
            .unwrap();

    let kernel_plan = VulkanKernelInterfacePlan::from_binding_plan(&binding_plan);

    assert_eq!(kernel_plan.backend_id, VULKAN_STREAM_CIRCUIT_BACKEND_ID);
    assert_eq!(kernel_plan.circuits.len(), 1);
    assert_eq!(kernel_plan.total_kernel_count(), 10);

    let q_projection = kernel_plan
        .kernel("layer_00", "q_projection__k_projection__v_projection")
        .unwrap();
    assert_eq!(
        q_projection.kernel_id,
        "layer_00.q_projection__k_projection__v_projection"
    );
    assert_eq!(q_projection.op, "parallel_linear_3way");
    assert_eq!(q_projection.inputs.len(), 1);
    assert_eq!(q_projection.outputs.len(), 3);
    assert_eq!(q_projection.parameters.len(), 3);
    assert_eq!(q_projection.stream_metadata.stream_control_binding, None);
    assert_eq!(
        q_projection.parameters[0],
        VulkanParameterBinding {
            param_id: "q_projection".to_string(),
            tensor: "model.layers.0.self_attn.q_proj.weight".to_string(),
            byte_count: Some(512),
            shape: Some(vec![16, 16]),
        }
    );
    assert!(q_projection.outputs.iter().all(|output| matches!(
        output.resource,
        VulkanSignalResource::ActivationSlot {
            ref component_id,
            signal_bytes: Some(32) | Some(16),
            ..
        } if component_id == "layer_00"
    )));

    let q_rope = kernel_plan.kernel("layer_00", "q_rope").unwrap();
    assert_eq!(q_rope.op, "rotary_position_embedding");
    assert_eq!(q_rope.stream_metadata.stream_control_binding, Some(2));
    assert_eq!(
        q_rope.stream_metadata.stream_tick,
        VulkanKernelScalarBinding {
            name: "stream_tick".to_string(),
            scalar_type: "u64".to_string(),
            source: VulkanKernelScalarSource::PushConstant,
        }
    );
    assert_eq!(q_rope.stream_metadata.control_flags.name, "control_flags");
    assert!(matches!(
        q_rope.outputs[0].resource,
        VulkanSignalResource::ActivationSlot {
            ref component_id,
            signal_bytes: Some(32),
            ..
        } if component_id == "layer_00"
    ));

    let kv_append = kernel_plan
        .kernel("layer_00", "kv_memory_append__attention_read")
        .unwrap();
    assert_eq!(kv_append.op, "append_scaled_dot_product_attention");
    assert_eq!(kv_append.stream_metadata.stream_control_binding, Some(7));
    assert_eq!(kv_append.inputs.len(), 4);
    assert_eq!(kv_append.outputs.len(), 1);
    assert_eq!(kv_append.state_reads.len(), 1);
    assert_eq!(kv_append.state_writes.len(), 1);
    assert!(kv_append.state_views.is_empty());
    assert!(matches!(
        kv_append.inputs[3].resource,
        VulkanSignalResource::StateBuffer {
            ref component_id,
            ref state_id,
            bytes_per_activation: Some(32),
            ..
        } if component_id == "layer_00" && state_id == "kv_memory"
    ));
    assert_eq!(
        kv_append
            .stream_metadata
            .dynamic_state_capacity_activations
            .name,
        "dynamic_state_capacity_activations"
    );
}

#[test]
fn kernel_stream_control_is_driven_only_by_the_compiled_node_contract() {
    let mut graph = fixture_model_execution_graph();
    let circuit = &mut graph.circuits[0].circuit;
    let operator_norm = circuit
        .nodes
        .iter_mut()
        .find(|node| node.id == "operator_norm")
        .unwrap();
    operator_norm.attrs["stream_control_binding"] = serde_json::json!(3);
    let q_rope = circuit
        .nodes
        .iter_mut()
        .find(|node| node.id == "q_rope")
        .unwrap();
    q_rope.attrs["stream_control_binding"] = serde_json::Value::Null;
    let tensor_index = TensorIndex::from_json_file(fixture_model_tensor_index_path()).unwrap();
    let execution_plan =
        StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index).unwrap();
    let resource_plan =
        StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
    let resident_plan = VulkanStreamCircuitResidentPlan::from_resource_plan(
        &resource_plan,
        Some(&tensor_index),
        Some(2),
    )
    .unwrap();
    let binding_plan =
        VulkanStreamCircuitBindingPlan::from_plans(&execution_plan, &resource_plan, &resident_plan)
            .unwrap();
    let kernels = VulkanKernelInterfacePlan::from_binding_plan(&binding_plan);

    assert_eq!(
        kernels
            .kernel("layer_00", "operator_norm")
            .unwrap()
            .stream_metadata
            .stream_control_binding,
        Some(3)
    );
    assert_eq!(
        kernels
            .kernel("layer_00", "q_rope")
            .unwrap()
            .stream_metadata
            .stream_control_binding,
        None
    );
}

#[test]
fn kernel_interfaces_keep_runtime_controls_out_of_graph_boundary_buffers() {
    let mut graph = fixture_model_execution_graph();
    let component = &mut graph.circuits[0];
    component.circuit.boundary.controls = vec![
        serde_json::from_value(serde_json::json!({
            "id": "token_id",
            "signal": "token_id",
            "shape": [],
            "dtype": "U32",
            "runtime_source": "input_token_id"
        }))
        .unwrap(),
    ];
    component
        .circuit
        .nodes
        .iter_mut()
        .find(|node| node.id == "q_projection__k_projection__v_projection")
        .unwrap()
        .inputs
        .push("token_id".to_string());
    component.circuit.validate_contract().unwrap();

    let tensor_index = TensorIndex::from_json_file(fixture_model_tensor_index_path()).unwrap();
    let execution_plan =
        StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index).unwrap();
    let resource_plan =
        StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
    let resident_plan = VulkanStreamCircuitResidentPlan::from_resource_plan(
        &resource_plan,
        Some(&tensor_index),
        Some(2),
    )
    .unwrap();
    let binding_plan =
        VulkanStreamCircuitBindingPlan::from_plans(&execution_plan, &resource_plan, &resident_plan)
            .unwrap();
    let kernel_plan = VulkanKernelInterfacePlan::from_binding_plan(&binding_plan);
    let projection = kernel_plan
        .kernel("layer_00", "q_projection__k_projection__v_projection")
        .unwrap();
    let token_id = projection
        .inputs
        .iter()
        .find(|input| input.signal_id == "token_id")
        .unwrap();

    assert!(matches!(
        token_id.resource,
        VulkanSignalResource::RuntimeControl {
            ref runtime_source,
            byte_capacity: 4,
        } if runtime_source == "input_token_id"
    ));
    assert!(
        !binding_plan.circuits[0]
            .input_ports
            .iter()
            .any(|port| port.id == "token_id")
    );
}

#[test]
fn boundary_input_passthrough_remains_a_readable_input_resource() {
    let mut graph = fixture_model_execution_graph();
    let component = &mut graph.circuits[0];
    component.circuit.boundary.outputs.push(
        serde_json::from_value(serde_json::json!({
            "id": "input_passthrough",
            "signal": "frame",
            "shape": [16],
            "source": "input_frame",
            "component_port": "input_passthrough"
        }))
        .unwrap(),
    );
    component.circuit.validate_contract().unwrap();

    let tensor_index = TensorIndex::from_json_file(fixture_model_tensor_index_path()).unwrap();
    let execution_plan =
        StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index).unwrap();
    let resource_plan =
        StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
    let resident_plan = VulkanStreamCircuitResidentPlan::from_resource_plan(
        &resource_plan,
        Some(&tensor_index),
        Some(2),
    )
    .unwrap();
    let binding_plan =
        VulkanStreamCircuitBindingPlan::from_plans(&execution_plan, &resource_plan, &resident_plan)
            .unwrap();
    let kernel_plan = VulkanKernelInterfacePlan::from_binding_plan(&binding_plan);
    let projection = kernel_plan.kernel("layer_00", "operator_norm").unwrap();

    assert!(matches!(
        projection.inputs[0].resource,
        VulkanSignalResource::BoundaryInput
    ));
    let planned_input = execution_plan.circuits[0].signal("input_frame").unwrap();
    assert!(planned_input.is_boundary_output);
    assert_eq!(planned_input.producer, SignalProducer::BoundaryInput);
}
#[test]
fn stream_control_buffer_bytes_follow_kernel_abi_order() {
    let push_constants =
        VulkanKernelStreamMetadata::from_compiled_contract(
            "rotary_position_embedding",
            Some(2),
        )
        .push_constants();
    assert!(push_constants.is_empty());
    let control = VulkanMountedPlacedStreamControl {
        stream_tick: 42,
        control_flags: 7,
        dynamic_state_capacity_activations: 4,
    };
    let push_bytes = stream_control_push_constant_bytes(&push_constants, control).unwrap();
    assert!(push_bytes.is_empty());

    let bytes = stream_control_bytes(11, control);
    assert_eq!(&bytes[0..4], &11u32.to_le_bytes());
    assert_eq!(&bytes[4..12], &42u64.to_le_bytes());
    assert_eq!(&bytes[12..16], &7u32.to_le_bytes());
    assert_eq!(&bytes[16..20], &4u32.to_le_bytes());
}

#[test]
fn component_batch_lane_controls_preserve_each_token_identity() {
    let controls = component_batch_lane_stream_control_bytes(&[9259, 1902], 41, 65_536).unwrap();

    assert_eq!(controls.len(), 2);
    assert_eq!(&controls[0][0..4], &9259u32.to_le_bytes());
    assert_eq!(&controls[0][4..12], &41u64.to_le_bytes());
    assert_eq!(&controls[1][0..4], &1902u32.to_le_bytes());
    assert_eq!(&controls[1][4..12], &42u64.to_le_bytes());
    assert_eq!(&controls[0][16..20], &65_536u32.to_le_bytes());
    assert_eq!(&controls[1][16..20], &65_536u32.to_le_bytes());
}

#[test]
fn component_batch_runtime_token_ids_are_a_contiguous_u32_vector() {
    let bytes = component_batch_runtime_token_id_bytes(&[9259, 1902, u32::MAX]).unwrap();

    assert_eq!(bytes.len(), 3 * size_of::<u32>());
    assert_eq!(&bytes[0..4], &9259u32.to_le_bytes());
    assert_eq!(&bytes[4..8], &1902u32.to_le_bytes());
    assert_eq!(&bytes[8..12], &u32::MAX.to_le_bytes());
}

#[test]
fn independent_stream_batch_controls_preserve_nonconsecutive_stream_ticks() {
    let controls =
        component_batch_lane_stream_control_bytes_for_ticks(&[9259, 1902], &[41, 907], 65_536)
            .unwrap();

    assert_eq!(controls.len(), 2);
    assert_eq!(&controls[0][0..4], &9259u32.to_le_bytes());
    assert_eq!(&controls[0][4..12], &41u64.to_le_bytes());
    assert_eq!(&controls[1][0..4], &1902u32.to_le_bytes());
    assert_eq!(&controls[1][4..12], &907u64.to_le_bytes());
}

#[test]
fn compiled_contract_can_require_stream_control_for_any_operation() {
    let metadata = VulkanKernelStreamMetadata::from_compiled_contract(
        "architecture_defined_temporal_operation",
        Some(6),
    );

    assert_eq!(metadata.stream_control_binding, Some(6));
    assert!(metadata.push_constants().is_empty());
}

#[test]
fn operation_name_does_not_implicitly_require_stream_control() {
    let metadata = VulkanKernelStreamMetadata::from_compiled_contract(
        "inverse_rotary_position_embedding",
        None,
    );

    assert_eq!(metadata.stream_control_binding, None);
    assert!(metadata.push_constants().is_empty());
}

#[test]
fn sparse_moe_kernels_receive_an_explicit_expert_range() {
    for op in ["sparse_moe_gate_up", "sparse_moe_down"] {
        let metadata = VulkanKernelStreamMetadata::from_compiled_contract(op, None);
        let push_constants = metadata.push_constants();

        assert_eq!(
            push_constants,
            vec![
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
            ]
        );
        assert_eq!(
            stream_control_push_constant_bytes(
                &push_constants,
                VulkanMountedPlacedStreamControl {
                    stream_tick: 42,
                    control_flags: 7,
                    dynamic_state_capacity_activations: 65_536,
                },
            )
            .unwrap(),
            [0u8; 8]
        );
    }
}

#[test]
fn selected_parameters_lower_to_generic_dynamic_resource_descriptors() {
    let node = VulkanNodeBinding {
        node_index: 7,
        node_id: "selected_compute".to_string(),
        op: "generic_compute".to_string(),
        specialization: String::new(),
        stream_control_binding: None,
        inputs: Vec::new(),
        outputs: Vec::new(),
        parameters: ["bank", "scale"]
            .into_iter()
            .map(|param_id| VulkanParameterBinding {
                param_id: param_id.to_string(),
                tensor: format!("tensor.{param_id}"),
                byte_count: Some(128),
                shape: Some(vec![4, 32]),
            })
            .collect(),
        state_reads: Vec::new(),
        state_writes: Vec::new(),
        selection_domain: None,
        selected_parameter_accesses: vec![
            VulkanSelectedParameterAccessBinding {
                component_id: "component".to_string(),
                node_id: "selected_compute".to_string(),
                selection_signal: "selected".to_string(),
                layout: PlannedSelectedParameterLayout::Partitioned {
                    partition_axis: 0,
                },
                parameter_ids: vec!["bank".to_string(), "scale".to_string()],
            },
        ],
    };
    let kernel = VulkanKernelInterface::from_node_binding("component", &node);
    let descriptors = descriptor_bindings_for_kernel(&kernel);

    assert_eq!(descriptors.len(), 2);
    assert_eq!(
        descriptors[0].usage,
        VulkanKernelDescriptorUsage::DynamicResourceAddressTable
    );
    assert_eq!(
        descriptors[1].usage,
        VulkanKernelDescriptorUsage::DynamicResourceParameterSlots
    );
    assert!(descriptors.iter().all(|descriptor| {
        descriptor.usage != VulkanKernelDescriptorUsage::Parameter
    }));
}

#[test]
fn selected_parameter_tensors_cannot_alias_permanent_parameters() {
    let selected_parameter = VulkanParameterBinding {
        param_id: "selected".to_string(),
        tensor: "weights.shared".to_string(),
        byte_count: Some(16),
        shape: Some(vec![2, 2]),
    };
    let permanent_parameter = VulkanParameterBinding {
        param_id: "permanent".to_string(),
        tensor: "weights.shared".to_string(),
        byte_count: Some(16),
        shape: Some(vec![2, 2]),
    };
    let node = |
        node_index: usize,
        node_id: &str,
        parameters: Vec<VulkanParameterBinding>,
        selected_parameter_accesses: Vec<VulkanSelectedParameterAccessBinding>,
    | {
        VulkanNodeBinding {
            node_index,
            node_id: node_id.to_string(),
            op: "compute".to_string(),
            specialization: "generic".to_string(),
            stream_control_binding: None,
            inputs: Vec::new(),
            outputs: Vec::new(),
            parameters,
            state_reads: Vec::new(),
            state_writes: Vec::new(),
            selection_domain: None,
            selected_parameter_accesses,
        }
    };
    let plan = VulkanStreamCircuitBindingPlan {
        backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
        circuits: vec![VulkanCircuitBindingPlan {
            component_id: "component".to_string(),
            circuit_id: "circuit".to_string(),
            input_ports: Vec::new(),
            output_ports: Vec::new(),
            nodes: vec![
                node(
                    0,
                    "selected_compute",
                    vec![selected_parameter],
                    vec![VulkanSelectedParameterAccessBinding {
                        component_id: "component".to_string(),
                        node_id: "selected_compute".to_string(),
                        selection_signal: "selected_resources".to_string(),
                        layout: PlannedSelectedParameterLayout::Partitioned {
                            partition_axis: 0,
                        },
                        parameter_ids: vec!["selected".to_string()],
                    }],
                ),
                node(
                    1,
                    "permanent_compute",
                    vec![permanent_parameter],
                    Vec::new(),
                ),
            ],
        }],
    };

    let error = plan.selected_parameter_tensors().unwrap_err();

    assert!(error.to_string().contains("both dynamic and permanent"));
}

#[test]
fn fused_head_norm_rope_kernel_receives_compiled_stream_control_metadata() {
    let metadata = VulkanKernelStreamMetadata::from_compiled_contract(
        "parallel_head_norm_rope_2way",
        Some(5),
    );

    assert_eq!(metadata.stream_control_binding, Some(5));
    assert!(metadata.push_constants().is_empty());

    let codebook_metadata = VulkanKernelStreamMetadata::from_compiled_contract(
        "parallel_head_norm_rope_2way_codebook_u8",
        Some(5),
    );
    assert_eq!(codebook_metadata.stream_control_binding, Some(5));
    assert!(codebook_metadata.push_constants().is_empty());

    let embedded_metadata = VulkanKernelStreamMetadata::from_compiled_contract(
        "parallel_head_norm_rope_2way_embedded_parameters",
        Some(5),
    );
    assert_eq!(embedded_metadata.stream_control_binding, Some(5));
    assert!(embedded_metadata.push_constants().is_empty());
}

#[test]
fn fused_append_attention_kernel_receives_stream_control_metadata() {
    let metadata = VulkanKernelStreamMetadata::from_compiled_contract(
        "append_scaled_dot_product_attention",
        Some(7),
    );

    assert_eq!(metadata.stream_control_binding, Some(7));
    assert!(metadata.push_constants().is_empty());
}

#[test]
fn dispatch_plan_orders_fixture_model_kernel_commands_for_stream_ticks() {
    let graph = fixture_model_execution_graph();
    let tensor_index = TensorIndex::from_json_file(fixture_model_tensor_index_path()).unwrap();
    let execution_plan =
        StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index).unwrap();
    let resource_plan =
        StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
    let resident_plan = VulkanStreamCircuitResidentPlan::from_resource_plan(
        &resource_plan,
        Some(&tensor_index),
        Some(2),
    )
    .unwrap();
    let binding_plan =
        VulkanStreamCircuitBindingPlan::from_plans(&execution_plan, &resource_plan, &resident_plan)
            .unwrap();

    let dispatch_plan = VulkanKernelDispatchPlan::from_binding_plan(&binding_plan);

    assert_eq!(dispatch_plan.backend_id, VULKAN_STREAM_CIRCUIT_BACKEND_ID);
    assert_eq!(dispatch_plan.total_dispatch_count(), 10);
    assert_eq!(dispatch_plan.op_counts().get("rms_norm"), Some(&2));
    assert_eq!(
        dispatch_plan.op_counts().get("rotary_position_embedding"),
        Some(&2)
    );

    let first = &dispatch_plan.commands[0];
    assert_eq!(first.dispatch_index, 0);
    assert_eq!(first.circuit_index, 0);
    assert_eq!(first.kernel_id, "layer_00.operator_norm");
    assert_eq!(first.component_id, "layer_00");
    assert_eq!(first.node_index, 0);
    assert_eq!(first.op, "rms_norm");
    assert_eq!(first.descriptor_bindings.len(), 3);
    assert_eq!(
        first
            .descriptor_bindings
            .iter()
            .map(|binding| binding.usage.clone())
            .collect::<Vec<_>>(),
        vec![
            VulkanKernelDescriptorUsage::InputSignal,
            VulkanKernelDescriptorUsage::OutputSignal,
            VulkanKernelDescriptorUsage::Parameter,
        ]
    );
    assert!(first.push_constants.is_empty());
    assert_eq!(first.stream_control_binding, None);

    let kv_append = dispatch_plan
        .command("layer_00", "kv_memory_append__attention_read")
        .unwrap();
    assert_eq!(kv_append.dispatch_index, 5);
    assert_eq!(kv_append.circuit_index, 0);
    assert_eq!(kv_append.node_index, 5);
    assert_eq!(kv_append.op, "append_scaled_dot_product_attention");
    assert_eq!(kv_append.stream_control_binding, Some(7));
    assert_eq!(
        kv_append
            .descriptor_bindings
            .iter()
            .map(|binding| (
                binding.binding,
                binding.usage.clone(),
                binding.name.as_str()
            ))
            .collect::<Vec<_>>(),
        vec![
            (
                0,
                VulkanKernelDescriptorUsage::InputSignal,
                "kv_memory_append__attention_read__attention_partials_f32",
            ),
            (1, VulkanKernelDescriptorUsage::InputSignal, "k_positioned"),
            (2, VulkanKernelDescriptorUsage::InputSignal, "v_projected"),
            (3, VulkanKernelDescriptorUsage::InputSignal, "kv_memory"),
            (4, VulkanKernelDescriptorUsage::OutputSignal, "attention_out"),
            (5, VulkanKernelDescriptorUsage::StateRead, "kv_memory"),
            (6, VulkanKernelDescriptorUsage::StateWrite, "kv_memory"),
        ]
    );
    assert!(matches!(
        kv_append.descriptor_bindings[3].resource,
        VulkanKernelDescriptorResource::Signal(VulkanSignalBinding {
            resource: VulkanSignalResource::StateBuffer {
                ref component_id,
                ref state_id,
                bytes_per_activation: Some(32),
                ..
            },
            ..
        }) if component_id == "layer_00" && state_id == "kv_memory"
    ));
    assert!(matches!(
        kv_append.descriptor_bindings[6].resource,
        VulkanKernelDescriptorResource::State {
            ref component_id,
            binding: VulkanStateBinding {
                ref state_id,
                bytes_per_activation: Some(32),
                ..
            },
        } if component_id == "layer_00" && state_id == "kv_memory"
    ));

    let last = dispatch_plan.commands.last().unwrap();
    assert_eq!(last.dispatch_index, 9);
    assert_eq!(last.circuit_index, 0);
    assert_eq!(last.kernel_id, "layer_00.ffn_down_projection__ffn_residual");
    assert_eq!(last.node_index, 9);
    let output = last
        .descriptor_bindings
        .iter()
        .find(|binding| binding.usage == VulkanKernelDescriptorUsage::OutputSignal)
        .unwrap();
    assert_eq!(
        output.resource,
        VulkanKernelDescriptorResource::Signal(VulkanSignalBinding {
            signal_id: "output_frame".to_string(),
            resource: VulkanSignalResource::BoundaryOutput,
        })
    );
}
