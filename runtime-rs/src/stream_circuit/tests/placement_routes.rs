    #[test]
    fn placement_plan_keeps_every_stream_entity_as_a_deployable_component() {
        let resolved =
            ResolvedLoweredExecutionGraph::from_index_file(fixture_model_index_path()).unwrap();

        let placement = resolved.single_device_placement_plan("gpu0").unwrap();

        assert_eq!(placement.schema, STREAM_CIRCUIT_PLACEMENT_SCHEMA);
        assert_eq!(placement.topology, "explicit_graph");
        assert_eq!(placement.components.len(), 4);
        assert_eq!(placement.edges.len(), 4);
        assert_eq!(placement.local_edge_count, 4);
        assert_eq!(placement.cross_device_edge_count, 0);
        assert_eq!(
            placement.component("layer_00").unwrap(),
            &ComponentPlacement {
                component_index: 1,
                component_id: "layer_00".to_string(),
                circuit_id: "layer_00_gqa_attention_circuit_v1".to_string(),
                operator_type: "full_attention".to_string(),
                device_id: "gpu0".to_string(),
            }
        );

        let first_edge = &placement.edges[0];
        assert_eq!(first_edge.source_component_id, "input_transducer");
        assert_eq!(first_edge.destination_component_id, "layer_00");
        assert_eq!(first_edge.signal, "frame");
        assert_eq!(first_edge.shape, vec![16]);
        assert_eq!(first_edge.source_port_id, "output_frame");
        assert_eq!(first_edge.destination_port_id, "input_frame");
        assert_eq!(first_edge.source_component_port.as_deref(), Some("frame"));
        assert_eq!(first_edge.destination_component_port.as_deref(), Some("input"));
        assert_eq!(
            first_edge.transport,
            EdgeTransport::LocalBuffer {
                device_id: "gpu0".to_string(),
            }
        );
    }

    #[test]
    fn bypassed_instance_is_retained_in_draft_and_removed_from_execution_graph() {
        let source =
            ResolvedLoweredExecutionGraph::from_index_file(fixture_model_index_path()).unwrap();
        let runtime_graph = StreamCircuitRuntimeGraph::from_source_series(&source, "gpu0")
            .unwrap()
            .with_instance_enabled("layer_00", false)
            .unwrap();

        let effective = source.instantiate_runtime_graph(&runtime_graph).unwrap();
        let placement = effective.placement_plan(&runtime_graph.placement_spec()).unwrap();

        assert_eq!(runtime_graph.instances.len(), 4);
        assert!(!runtime_graph.instances[1].enabled);
        assert_eq!(effective.circuits.len(), 3);
        assert!(
            effective
                .circuits
                .iter()
                .all(|circuit| circuit.component.id != "layer_00")
        );
        let bypass = placement
            .edges
            .iter()
            .find(|edge| edge.source_component_id == "input_transducer")
            .unwrap();
        assert_eq!(bypass.destination_component_id, "output_transducer");
    }

    #[test]
    fn placement_plan_changes_edges_not_execution_graph_when_devices_differ() {
        let resolved =
            ResolvedLoweredExecutionGraph::from_index_file(fixture_model_index_path()).unwrap();
        let spec = StreamCircuitPlacementSpec::new("gpu0")
            .with_component_device("layer_00", "cpu0")
            .with_component_device("output_transducer", "gpu1")
            .with_component_device("sampler", "lan:worker-a");

        let placement = resolved.placement_plan(&spec).unwrap();

        assert_eq!(placement.components.len(), 4);
        assert_eq!(placement.edges.len(), 4);
        assert_eq!(placement.local_edge_count, 0);
        assert_eq!(placement.cross_device_edge_count, 4);
        assert_eq!(
            placement
                .component("layer_00")
                .map(|component| component.device_id.as_str()),
            Some("cpu0")
        );
        assert_eq!(
            placement
                .component("output_transducer")
                .map(|component| component.device_id.as_str()),
            Some("gpu1")
        );
        assert_eq!(
            placement
                .component("sampler")
                .map(|component| component.device_id.as_str()),
            Some("lan:worker-a")
        );
        assert_eq!(
            placement.cross_device_edges().len(),
            4
        );
        let cross = placement.cross_device_edges();
        assert_eq!(cross.len(), 4);
        assert_eq!(
            cross
                .iter()
                .map(|edge| (
                    edge.source_component_id.as_str(),
                    edge.source_device_id.as_str(),
                    edge.destination_component_id.as_str(),
                    edge.destination_device_id.as_str()
                ))
                .collect::<Vec<_>>(),
            vec![
                ("input_transducer", "gpu0", "layer_00", "cpu0"),
                ("layer_00", "cpu0", "output_transducer", "gpu1"),
                ("output_transducer", "gpu1", "sampler", "lan:worker-a"),
                ("sampler", "lan:worker-a", "input_transducer", "gpu0"),
            ]
        );
        assert_eq!(
            cross[2].transport,
            EdgeTransport::CrossDevice {
                from_device_id: "gpu1".to_string(),
                to_device_id: "lan:worker-a".to_string(),
            }
        );
    }

    #[test]
    fn runtime_edge_routes_classify_logical_and_physical_routes() {
        let resolved =
            ResolvedLoweredExecutionGraph::from_index_file(fixture_model_index_path()).unwrap();
        let spec = StreamCircuitPlacementSpec::new("gpu0")
            .with_component_device("layer_00", "gpu1")
            .with_component_device("output_transducer", "gpu2");
        let placement = resolved.placement_plan(&spec).unwrap();

        let routes = placement.runtime_edge_routes(|device_id| {
            let (target, physical_device_index) = match device_id {
                "gpu0" | "gpu1" => (Some("vulkan:0".to_string()), Some(0)),
                "gpu2" => (Some("vulkan:1".to_string()), Some(1)),
                _ => (None, None),
            };
            RuntimeEdgeRouteTarget {
                target,
                physical_device_index,
                binding_source: "test".to_string(),
            }
        });

        assert_eq!(routes.schema, RUNTIME_EDGE_ROUTES_SCHEMA);
        assert_eq!(routes.edge_count, 4);
        assert_eq!(routes.logical_local_edge_count, 1);
        assert_eq!(routes.logical_cross_device_edge_count, 3);
        assert_eq!(routes.same_physical_target_edge_count, 1);
        assert_eq!(routes.cross_physical_target_edge_count, 2);
        assert_eq!(routes.unresolved_target_edge_count, 0);
        assert_eq!(
            routes.routes[0].route_kind,
            RuntimeEdgeRouteKind::SamePhysicalTarget
        );
        assert_eq!(
            routes.routes[1].route_kind,
            RuntimeEdgeRouteKind::CrossPhysicalTarget
        );
        assert_eq!(
            routes.routes[3].route_kind,
            RuntimeEdgeRouteKind::LogicalLocal
        );

        let payload = serde_json::to_value(&routes).unwrap();
        assert_eq!(payload["routes"][0]["route_kind"], "same_physical_target");
        assert_eq!(payload["routes"][1]["route_kind"], "cross_physical_target");
        assert_eq!(payload["routes"][3]["route_kind"], "logical_local");
    }

    #[test]
    fn runtime_device_bindings_capture_runtime_target_contract() {
        let logical_device_ids = vec![
            "gpu0".to_string(),
            "gpu1".to_string(),
            "vulkan:7".to_string(),
        ];
        let mut explicit_bindings = BTreeMap::new();
        explicit_bindings.insert("gpu1".to_string(), "vulkan:5".to_string());
        explicit_bindings.insert("remote0".to_string(), "lan:worker-a".to_string());

        let bindings = RuntimeDeviceBindings::from_vulkan_targets(
            &logical_device_ids,
            &explicit_bindings,
            Some(0),
            |target| {
                if let Some(index) = target.strip_prefix("vulkan:") {
                    return index.parse::<usize>().map(Some).map_err(|error| {
                        format!("invalid Vulkan physical device reference {target:?}: {error}")
                    });
                }
                Ok(None)
            },
        );

        assert_eq!(bindings.schema, RUNTIME_DEVICE_BINDINGS_SCHEMA);
        assert_eq!(bindings.process_vulkan_device_index, Some(0));
        assert_eq!(bindings.default_vulkan_device_index, Some(0));
        assert_eq!(bindings.requested_vulkan_device_indices, vec![0, 5, 7]);
        assert!(!bindings.can_mount_in_process);
        assert_eq!(bindings.mounting_model, "unsupported_targets");
        assert_eq!(bindings.unsupported_targets, vec!["remote0=lan:worker-a"]);
        assert_eq!(
            bindings
                .logical_devices
                .iter()
                .map(|device| (
                    device.device_id.as_str(),
                    device.target.as_deref(),
                    device.binding_source.as_str()
                ))
                .collect::<Vec<_>>(),
            vec![
                ("gpu0", Some("vulkan:0"), "process_default"),
                ("gpu1", Some("vulkan:5"), "explicit"),
                ("remote0", Some("lan:worker-a"), "explicit"),
                ("vulkan:7", Some("vulkan:7"), "device_id"),
            ]
        );

        let payload = serde_json::to_value(&bindings).unwrap();
        assert_eq!(payload["schema"], RUNTIME_DEVICE_BINDINGS_SCHEMA);
        assert_eq!(payload["logical_devices"][0]["device_id"], "gpu0");
        assert_eq!(
            payload["logical_devices"][0]["binding_source"],
            "process_default"
        );
        assert_eq!(payload["unsupported_targets"][0], "remote0=lan:worker-a");
    }
    #[test]
    fn parallel_block_connections_require_exact_scatter_and_gather_geometry() {
        let scatter = StreamCircuitConnection::ParallelBlockScatter { width: 7 };
        assert!(scatter.connects_port_geometry(
            "stream_frame_block",
            &[7, 4, 4096],
            "frame",
            &[4, 4096],
        ));
        assert!(!scatter.connects_port_geometry(
            "stream_frame_block",
            &[5, 4, 4096],
            "frame",
            &[4, 4096],
        ));

        let gather = StreamCircuitConnection::ParallelBlockGather { width: 7 };
        assert!(gather.connects_port_geometry(
            "frame",
            &[4, 4096],
            "stream_frame_block",
            &[7, 4, 4096],
        ));
        assert!(!gather.connects_port_geometry(
            "frame",
            &[4, 4096],
            "stream_frame_block",
            &[7, 4096],
        ));
    }
