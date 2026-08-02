    #[test]
    fn circuit_contract_rejects_ambiguous_boundary_ports() {
        let circuit: StreamCircuit = serde_json::from_value(serde_json::json!({
            "schema": STREAM_CIRCUIT_SCHEMA,
            "id": "fixture_circuit",
            "source": {
                "component_id": "fixture_component",
                "source_layer_index": null,
                "source_operator_type": "fixture"
            },
            "runtime_role": "input_transducer",
            "behavioral_role": "fixture",
            "implementation": "fixture",
            "boundary": {
                "inputs": [{
                    "id": "input_frame",
                    "signal": "frame",
                    "shape": [8],
                    "component_port": "input"
                }],
                "outputs": [{
                    "id": "output_frame",
                    "signal": "frame",
                    "shape": [8],
                    "source": "output_frame",
                    "component_port": "output"
                }]
            },
            "parameters": {
                "layout": "row_major",
                "storage": "safetensors",
                "refs": {}
            },
            "nodes": [{
                "id": "identity",
                "op": "identity",
                "inputs": ["input_frame"],
                "outputs": ["output_frame"]
            }]
        }))
        .unwrap();

        assert_eq!(circuit.source.source_layer_index, None);
        assert_eq!(circuit.runtime_role, CircuitRuntimeRole::InputTransducer);

        let mut duplicate_input = circuit.clone();
        duplicate_input
            .boundary
            .inputs
            .push(duplicate_input.boundary.inputs[0].clone());
        let input_error = duplicate_input.validate_contract().unwrap_err();
        assert!(
            input_error
                .to_string()
                .contains("duplicate boundary input port id")
        );

        let mut duplicate_output = circuit.clone();
        duplicate_output
            .boundary
            .outputs
            .push(duplicate_output.boundary.outputs[0].clone());
        let output_error = duplicate_output.validate_contract().unwrap_err();
        assert!(
            output_error
                .to_string()
                .contains("duplicate boundary output port id")
        );

        let mut malformed = circuit.clone();
        malformed.boundary.inputs[0].shape.clear();
        malformed.boundary.outputs[0].component_port = Some(String::new());
        let malformed_error = malformed.validate_contract().unwrap_err().to_string();
        assert!(malformed_error.contains("shape must contain positive dimensions"));
        assert!(malformed_error.contains("must map to a non-empty component_port"));
    }

    #[test]
    fn circuit_contract_accepts_typed_runtime_controls_as_node_inputs() {
        let circuit: StreamCircuit = serde_json::from_value(serde_json::json!({
            "schema": STREAM_CIRCUIT_SCHEMA,
            "id": "runtime_control_fixture",
            "source": {
                "component_id": "layer_00",
                "source_layer_index": 0,
                "source_operator_type": "latent_sparse_attention"
            },
            "runtime_role": "signal_processor",
            "behavioral_role": "fixture",
            "implementation": "fixture",
            "boundary": {
                "inputs": [{
                    "id": "input_frame",
                    "signal": "frame",
                    "shape": [8],
                    "component_port": "input"
                }],
                "outputs": [{
                    "id": "output_frame",
                    "signal": "frame",
                    "shape": [8],
                    "source": "output_frame",
                    "component_port": "output"
                }],
                "controls": [{
                    "id": "token_id",
                    "signal": "token_id",
                    "shape": [],
                    "dtype": "U32",
                    "runtime_source": "input_token_id"
                }]
            },
            "parameters": {
                "layout": "row_major",
                "storage": "safetensors",
                "refs": {}
            },
            "nodes": [{
                "id": "route",
                "op": "moe_route",
                "inputs": ["input_frame", "token_id"],
                "outputs": ["output_frame"]
            }]
        }))
        .unwrap();

        circuit.validate_contract().unwrap();
        let plan = crate::stream_plan::CircuitActivationPlan::from_circuit(
            "layer_00",
            &circuit,
        )
        .unwrap();
        let token_id = plan.signal("token_id").unwrap();
        assert_eq!(token_id.shape, Some(Vec::new()));
        assert_eq!(token_id.element_bytes, Some(4));
        assert!(matches!(
            token_id.producer,
            crate::stream_plan::SignalProducer::RuntimeControl {
                ref runtime_source
            } if runtime_source == "input_token_id"
        ));
        assert_eq!(
            token_id.storage,
            crate::stream_plan::SignalStorage::RuntimeControl
        );

        let mut unsupported = circuit.clone();
        unsupported.boundary.controls[0].runtime_source = "wall_clock".to_string();
        assert!(
            unsupported
                .validate_contract()
                .unwrap_err()
                .to_string()
                .contains("unsupported runtime source")
        );
    }

    #[test]
    fn circuit_contract_types_single_vector_temporal_state_geometry() {
        let mut circuit: StreamCircuit = serde_json::from_value(serde_json::json!({
            "schema": STREAM_CIRCUIT_SCHEMA,
            "id": "temporal_state_fixture",
            "source": {
                "component_id": "layer_00",
                "source_layer_index": 0,
                "source_operator_type": "latent_sparse_attention"
            },
            "runtime_role": "signal_processor",
            "behavioral_role": "fixture",
            "implementation": "fixture",
            "boundary": {
                "inputs": [{
                    "id": "input_frame",
                    "signal": "frame",
                    "shape": [8],
                    "component_port": "input"
                }],
                "outputs": [{
                    "id": "output_frame",
                    "signal": "frame",
                    "shape": [8],
                    "source": "output_frame",
                    "component_port": "output"
                }]
            },
            "state_ports": [{
                "id": "temporal_memory",
                "type": "rolling_attention_memory",
                "shape_per_token": [2, 8],
                "capacity": 128,
                "max_dynamic_activations": 64,
                "dtype": "BF16",
                "update": "ring_append"
            }],
            "parameters": {
                "layout": "row_major",
                "storage": "safetensors",
                "refs": {}
            },
            "nodes": [{
                "id": "identity",
                "op": "identity",
                "inputs": ["input_frame"],
                "outputs": ["output_frame"]
            }]
        }))
        .unwrap();

        circuit.validate_contract().unwrap();
        let state = &circuit.state_ports[0];
        assert_eq!(state.elements_per_activation(), Some(16));
        assert_eq!(state.dynamic_activation_capacity(), Some(64));
        assert_eq!(state.dtype.as_deref(), Some("BF16"));
        assert!(!state.extra.contains_key("shape_per_token"));
        assert!(!state.extra.contains_key("capacity"));
        assert!(!state.extra.contains_key("dtype"));

        circuit.state_ports[0].capacity = Some(0);
        assert!(circuit
            .validate_contract()
            .unwrap_err()
            .to_string()
            .contains("capacity must be positive"));

        circuit.state_ports[0].capacity = Some(128);
        circuit.state_ports[0].key_shape_per_token = Some(vec![2, 8]);
        let ambiguity = circuit.validate_contract().unwrap_err().to_string();
        assert!(ambiguity.contains("cannot combine shape_per_token with key/value shapes"));
    }

    #[test]
    fn circuit_contract_requires_exact_semantic_module_ownership() {
        let circuit: StreamCircuit = serde_json::from_value(serde_json::json!({
            "schema": STREAM_CIRCUIT_SCHEMA,
            "id": "semantic_fixture",
            "source": {
                "component_id": "layer_00",
                "source_layer_index": 0,
                "source_operator_type": "conv"
            },
            "runtime_role": "signal_processor",
            "behavioral_role": "fixture",
            "implementation": "fixture",
            "boundary": {
                "inputs": [{
                    "id": "input_frame",
                    "signal": "frame",
                    "shape": [8],
                    "component_port": "input"
                }],
                "outputs": [{
                    "id": "output_frame",
                    "signal": "frame",
                    "shape": [8],
                    "source": "output_frame",
                    "component_port": "output"
                }]
            },
            "state_ports": [{
                "id": "memory",
                "type": "rolling_memory",
                "shape": [2, 8],
                "update": "replace"
            }],
            "parameters": {
                "layout": "row_major",
                "storage": "safetensors",
                "refs": {"weight": {"tensor": "layer.weight"}}
            },
            "semantic_module_tree": {
                "schema": SEMANTIC_MODULE_TREE_SCHEMA,
                "root_module_id": "layer",
                "modules": [{
                    "id": "layer",
                    "role": "layer",
                    "responsibility": "Editable layer",
                    "parent_id": null,
                    "child_ids": ["layer.token_mixer"],
                    "source_node_ids": [],
                    "parameter_ref_ids": [],
                    "owned_state_port_ids": [],
                    "input_signals": ["input_frame"],
                    "output_signals": ["output_frame"]
                }, {
                    "id": "layer.token_mixer",
                    "role": "token_mixer",
                    "responsibility": "Stateful projection",
                    "parent_id": "layer",
                    "child_ids": [],
                    "source_node_ids": ["project"],
                    "parameter_ref_ids": ["weight"],
                    "owned_state_port_ids": ["memory"],
                    "input_signals": ["input_frame"],
                    "output_signals": ["output_frame"]
                }]
            },
            "nodes": [{
                "id": "project",
                "op": "linear",
                "inputs": ["input_frame", "memory"],
                "outputs": ["output_frame"],
                "params": ["weight"],
                "state_reads": ["memory"],
                "state_writes": ["memory"]
            }]
        }))
        .unwrap();

        circuit.validate_contract().unwrap();

        let mut duplicate_node = circuit.clone();
        duplicate_node
            .semantic_module_tree
            .as_mut()
            .unwrap()
            .modules[0]
            .source_node_ids
            .push("project".to_string());
        assert!(
            duplicate_node
                .validate_contract()
                .unwrap_err()
                .to_string()
                .contains("belongs to semantic modules")
        );

        let mut represented = circuit.clone();
        represented.semantic_execution_nodes = represented.nodes.clone();
        represented.parameters.refs.remove("weight");
        represented.parameters.refs.insert(
            "weight_codebook".to_string(),
            ParameterRef {
                tensor: Some("layer.weight.codebook".to_string()),
                role: Some("exact_codebook".to_string()),
                extra: Map::new(),
            },
        );
        represented.nodes[0].params = vec!["weight_codebook".to_string()];
        represented.nodes[0].attrs = serde_json::json!({
            "parameter_representation": {
                "kind": "shared_codebook",
                "source_parameter_ids": ["weight"],
                "descriptor_abi": "source_parameters_replaced",
                "alternative_execution_phases": ["decode", "prefill"],
                "source_retained_execution_phases": []
            }
        });
        represented.validate_contract().unwrap();

        let mut unknown_representation_source = represented.clone();
        unknown_representation_source.nodes[0].attrs = serde_json::json!({
            "parameter_representation": {
                "kind": "shared_codebook",
                "source_parameter_ids": ["unknown_weight"],
                "descriptor_abi": "source_parameters_replaced",
                "alternative_execution_phases": ["decode", "prefill"],
                "source_retained_execution_phases": []
            }
        });
        assert!(
            unknown_representation_source
                .validate_contract()
                .unwrap_err()
                .to_string()
                .contains("absent from semantic execution")
        );

        let mut duplicate_physical_storage = represented;
        duplicate_physical_storage.parameters.refs.insert(
            "weight".to_string(),
            ParameterRef {
                tensor: Some("layer.weight".to_string()),
                role: Some("source_weight".to_string()),
                extra: Map::new(),
            },
        );
        assert!(
            duplicate_physical_storage
                .validate_contract()
                .unwrap_err()
                .to_string()
                .contains("remains physically bound")
        );

        let mut phase_selective = circuit.clone();
        phase_selective.semantic_execution_nodes = phase_selective.nodes.clone();
        phase_selective.nodes[0].attrs = serde_json::json!({
            "parameter_representation": {
                "kind": "embedded_exact_program",
                "source_parameter_ids": ["weight"],
                "descriptor_abi": "source_parameters_retained",
                "alternative_execution_phases": ["decode"],
                "source_retained_execution_phases": ["prefill"]
            }
        });
        phase_selective.validate_contract().unwrap();

        let mut missing_retained_binding = phase_selective;
        missing_retained_binding.parameters.refs.remove("weight");
        assert!(
            missing_retained_binding
                .validate_contract()
                .unwrap_err()
                .to_string()
                .contains("retained source parameter")
        );

        let mut missing_state = circuit;
        missing_state
            .semantic_module_tree
            .as_mut()
            .unwrap()
            .modules[1]
            .owned_state_port_ids
            .clear();
        assert!(
            missing_state
                .validate_contract()
                .unwrap_err()
                .to_string()
                .contains("does not own every state port exactly once")
        );
    }
    #[test]
    fn graph_boundary_deserializes_typed_source_tap() {
        let boundary: StreamCircuitGraphBoundary =
            serde_json::from_value(serde_json::json!({
                "external_inputs": [{
                    "id": "target_hidden",
                    "endpoint": {
                        "component_id": "draft_input",
                        "port_id": "target_hidden"
                    },
                    "source_tap": {
                        "component_id": "target_processor",
                        "port_id": "output_frame",
                        "instance_selection": "last_in_execution_order"
                    }
                }],
                "public_outputs": [{
                    "id": "draft_tokens",
                    "endpoint": {
                        "component_id": "draft_output",
                        "port_id": "draft_tokens"
                    }
                }]
            }))
            .unwrap();

        let source_tap = boundary.external_inputs[0].source_tap.as_ref().unwrap();
        assert_eq!(source_tap.component_id, "target_processor");
        assert_eq!(source_tap.port_id, "output_frame");
        assert_eq!(
            source_tap.instance_selection,
            StreamCircuitGraphSourceTapInstanceSelection::LastInExecutionOrder
        );
        assert!(boundary.public_outputs[0].source_tap.is_none());
    }
