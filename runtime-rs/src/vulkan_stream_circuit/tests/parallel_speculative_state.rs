fn parallel_state_circuit() -> StreamCircuit {
    serde_json::from_value(serde_json::json!({
        "schema": "nerve.stream_circuit.v1",
        "id": "draft_processor",
        "source": {
            "component_id": "draft_processor",
            "source_layer_index": null,
            "source_operator_type": "synthetic"
        },
        "runtime_role": "draft_processor",
        "behavioral_role": "stream_generation_circuit",
        "implementation": "synthetic",
        "boundary": {
            "inputs": [
                {"id": "query", "signal": "frame", "shape": [8]},
                {"id": "main_context", "signal": "frame", "shape": [8]}
            ],
            "outputs": [
                {"id": "output", "signal": "frame", "shape": [8], "source": "query_out"}
            ],
            "controls": []
        },
        "state_ports": [
            {"id": "context_memory", "type": "rolling", "shape": [4, 8]},
            {"id": "query_memory", "type": "rolling", "shape": [4, 8]}
        ],
        "parameters": {"layout": "none", "storage": "none", "refs": {}},
        "nodes": [
            {
                "id": "query_projection",
                "op": "linear",
                "inputs": ["query"],
                "outputs": ["query_out"]
            },
            {
                "id": "context_quantize",
                "op": "quantize",
                "inputs": ["main_context"],
                "outputs": ["context_quantized"]
            },
            {
                "id": "context_projection",
                "op": "linear",
                "inputs": ["context_quantized"],
                "outputs": ["context_projected"]
            },
            {
                "id": "context_state_update",
                "op": "rolling_state_update",
                "inputs": ["context_projected", "context_memory"],
                "outputs": ["context_values"],
                "state_reads": ["context_memory"],
                "state_writes": ["context_memory"]
            },
            {
                "id": "query_state_update",
                "op": "rolling_state_update",
                "inputs": ["query_out", "query_memory"],
                "outputs": ["query_values"],
                "state_reads": ["query_memory"],
                "state_writes": ["query_memory"]
            }
        ],
        "behavioral_error_contract": {},
        "lowering_notes": []
    }))
    .expect("synthetic parallel state circuit must deserialize")
}

#[test]
fn parallel_state_ingestion_selects_only_the_committed_context_dependency_cone() {
    let selected = committed_context_state_node_ids(
        &parallel_state_circuit(),
        "main_context",
    )
    .unwrap();

    assert_eq!(
        selected,
        BTreeSet::from([
            "context_quantize".to_string(),
            "context_projection".to_string(),
            "context_state_update".to_string(),
        ])
    );
}

#[test]
fn parallel_proposal_excludes_every_committed_context_state_node() {
    let circuit = parallel_state_circuit();
    let committed = committed_context_state_node_ids(&circuit, "main_context").unwrap();
    let proposal = proposal_node_ids(&circuit, &committed).unwrap();

    assert_eq!(
        committed,
        BTreeSet::from([
            "context_quantize".to_string(),
            "context_projection".to_string(),
            "context_state_update".to_string(),
        ])
    );
    assert_eq!(
        proposal,
        BTreeSet::from([
            "query_projection".to_string(),
            "query_state_update".to_string(),
        ])
    );
    assert!(committed.is_disjoint(&proposal));
}

#[test]
fn parallel_state_ingestion_rejects_a_signal_without_a_state_sink() {
    let error = committed_context_state_node_ids(&parallel_state_circuit(), "absent_context")
        .unwrap_err();

    assert!(error.0.contains("no state update derived from committed context"));
}

#[test]
fn node_scoped_component_batch_execution_rejects_missing_dispatches() {
    let scope = VulkanComponentBatchExecutionScope::nodes(BTreeMap::from([(
        "draft_processor".to_string(),
        BTreeSet::from([
            "context_projection".to_string(),
            "context_state_update".to_string(),
        ]),
    )]))
    .unwrap();

    assert!(scope.includes_dispatch("draft_processor", "context_projection"));
    assert!(!scope.includes_dispatch("draft_processor", "query_projection"));
    assert!(
        scope
            .validate_dispatch_ids([
                ("draft_processor", "context_projection"),
                ("draft_processor", "context_state_update"),
            ])
            .is_ok()
    );
    assert!(
        scope
            .validate_dispatch_ids([("draft_processor", "context_projection")])
            .unwrap_err()
            .0
            .contains("context_state_update")
    );
}

#[test]
fn compiled_source_context_tick_offsets_are_checked() {
    assert_eq!(offset_stream_tick(17, -1).expect("preceding tick"), 16);
    assert_eq!(offset_stream_tick(17, 0).expect("same tick"), 17);
    assert_eq!(offset_stream_tick(17, 2).expect("future tick"), 19);
    assert!(offset_stream_tick(0, -1).is_err());
    assert!(offset_stream_tick(u64::MAX, 1).is_err());
}
