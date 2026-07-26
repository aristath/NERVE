pub fn inspect_representation_graph(
    graph: &crate::RepresentationGraph,
) -> Result<RuntimeEditorRepresentationGraph, RuntimeEditorError> {
    graph.validate().map_err(RuntimeEditorError)?;
    Ok(RuntimeEditorRepresentationGraph {
        schema: graph.schema.clone(),
        graph_id: graph.graph_id.clone(),
        candidate_id: graph.candidate_id.clone(),
        scope_ids: graph.scope_ids.clone(),
        source_contract_digests: graph.source_contract_digests.clone(),
        logical_contracts: graph.logical_contracts.clone(),
        physical_representations: graph.physical_representations.clone(),
        signals: graph.signals.clone(),
        resources: graph.resources.clone(),
        nodes: graph.nodes.clone(),
        connections: graph.connections.clone(),
        public_ports: graph.public_ports.clone(),
        islands: graph.islands.clone(),
        absorbed_transforms: graph.absorbed_transforms.clone(),
        physical_kernels: graph.physical_kernels.clone(),
        confidence: graph.confidence.clone(),
        unresolved: graph.unresolved.clone(),
        correction_requests: graph.correction_requests.clone(),
    })
}

#[cfg(test)]
mod representation_inspection_tests {
    use super::*;

    #[test]
    fn editor_schema_round_trips_logical_and_physical_representation_inspection() {
        let graph = crate::representation_graph::representation_graph_test_fixture();
        let inspection = inspect_representation_graph(&graph).unwrap();
        let encoded = serde_json::to_vec(&inspection).unwrap();
        let decoded: RuntimeEditorRepresentationGraph =
            serde_json::from_slice(&encoded).unwrap();

        assert_eq!(decoded, inspection);
        assert_eq!(decoded.logical_contracts[0].shape, vec![8]);
        assert_eq!(
            decoded.physical_representations[1].physical_shape,
            vec![5]
        );
        assert_eq!(
            decoded.signals[0].provenance.scope_ids,
            vec!["scope_a", "scope_b"]
        );
        assert!(!decoded.connections[0].materializes_source);
    }
}
