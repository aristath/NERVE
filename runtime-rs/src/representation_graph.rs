use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const REPRESENTATION_GRAPH_SCHEMA: &str = "nerve.optimizer.representation_graph.v1";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RepresentationGraph {
    pub schema: String,
    pub graph_id: String,
    pub candidate_id: String,
    pub scope_ids: Vec<String>,
    pub source_contract_digests: BTreeMap<String, String>,
    pub logical_contracts: Vec<RepresentationLogicalContract>,
    pub physical_representations: Vec<PhysicalRepresentation>,
    pub signals: Vec<RepresentationSignal>,
    pub resources: Vec<RepresentationResource>,
    pub nodes: Vec<RepresentationNode>,
    pub connections: Vec<RepresentationConnection>,
    pub public_ports: Vec<RepresentationPublicPort>,
    pub islands: Vec<RepresentationIsland>,
    pub absorbed_transforms: Vec<RepresentationAbsorbedTransform>,
    pub physical_kernels: Vec<RepresentationPhysicalKernel>,
    pub confidence: RepresentationConfidence,
    pub unresolved: Vec<RepresentationUnresolved>,
    pub correction_requests: Vec<RepresentationCorrectionRequest>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepresentationLogicalContract {
    pub id: String,
    pub signal: String,
    pub shape: Vec<usize>,
    pub dtype: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PhysicalRepresentation {
    pub id: String,
    pub kind: String,
    pub domain: String,
    pub physical_shape: Vec<usize>,
    pub encoding: Value,
    pub storage: Value,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepresentationProvenance {
    pub scope_ids: Vec<String>,
    pub source_node_ids: Vec<String>,
    pub evidence_refs: Vec<String>,
    pub transform_refs: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepresentationSignal {
    pub id: String,
    pub logical_contract_id: String,
    pub physical_representation_id: String,
    pub provenance: RepresentationProvenance,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RepresentationResource {
    pub id: String,
    pub kind: String,
    pub logical_contract_id: String,
    pub physical_representation_id: String,
    pub artifact: Value,
    pub provenance: RepresentationProvenance,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepresentationNodePort {
    pub id: String,
    pub signal_id: String,
    pub physical_representation_id: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RepresentationCost {
    pub status: String,
    pub metrics: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RepresentationNode {
    pub id: String,
    pub kind: String,
    pub operation: String,
    pub inputs: Vec<RepresentationNodePort>,
    pub outputs: Vec<RepresentationNodePort>,
    pub resource_ids: Vec<String>,
    pub state_read_ids: Vec<String>,
    pub state_write_ids: Vec<String>,
    pub cost: Option<RepresentationCost>,
    pub provenance: RepresentationProvenance,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepresentationEndpoint {
    pub node_id: String,
    pub port_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepresentationConnection {
    pub id: String,
    pub producer: RepresentationEndpoint,
    pub consumer: RepresentationEndpoint,
    pub signal_id: String,
    pub materializes_source: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepresentationPublicPort {
    pub id: String,
    pub direction: String,
    pub logical_contract_id: String,
    pub signal_id: String,
    pub node_id: String,
    pub node_port_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepresentationIsland {
    pub id: String,
    pub scope_ids: Vec<String>,
    pub node_ids: Vec<String>,
    pub connection_ids: Vec<String>,
    pub representation_ids: Vec<String>,
    pub boundary_port_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepresentationAbsorbedTransform {
    pub id: String,
    pub kind: String,
    pub source_representation_id: String,
    pub target_representation_id: String,
    pub adjacent_node_ids: Vec<String>,
    pub parameter_resource_ids: Vec<String>,
    pub proof_ref: String,
    pub evidence_refs: Vec<String>,
    pub provenance: RepresentationProvenance,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RepresentationPhysicalKernel {
    pub id: String,
    pub node_ids: Vec<String>,
    pub artifact: Value,
    pub target_predicate: Value,
    pub cost: RepresentationCost,
    pub provenance: RepresentationProvenance,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RepresentationConfidence {
    pub mode: String,
    pub score: f64,
    pub basis: String,
    pub evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepresentationUnresolved {
    pub id: String,
    pub subject_ids: Vec<String>,
    pub reason: String,
    pub evidence_refs: Vec<String>,
    pub provenance: RepresentationProvenance,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RepresentationCorrectionRequest {
    pub id: String,
    pub trigger: Value,
    pub correction_node_id: String,
    pub fallback_scope_ids: Vec<String>,
    pub output_port_ids: Vec<String>,
    pub error_contract: Value,
    pub provenance: RepresentationProvenance,
}

impl RepresentationGraph {
    pub fn from_json_slice(bytes: &[u8]) -> Result<Self, String> {
        let graph: Self = serde_json::from_slice(bytes)
            .map_err(|error| format!("invalid representation graph JSON: {error}"))?;
        graph.validate()?;
        Ok(graph)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != REPRESENTATION_GRAPH_SCHEMA {
            return Err(format!(
                "unsupported representation graph schema {:?}",
                self.schema
            ));
        }
        require_text(&self.graph_id, "graph_id")?;
        require_text(&self.candidate_id, "candidate_id")?;
        let graph_scopes = unique_ids(
            self.scope_ids.iter().map(String::as_str),
            "representation graph scope_ids",
            true,
        )?;
        if self
            .source_contract_digests
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>()
            != graph_scopes
        {
            return Err(
                "source_contract_digests must cover exactly the graph scope_ids".to_string(),
            );
        }
        let logical = indexed(
            self.logical_contracts.iter().map(|item| item.id.as_str()),
            "logical contracts",
        )?;
        let representations = indexed(
            self.physical_representations
                .iter()
                .map(|item| item.id.as_str()),
            "physical representations",
        )?;
        let signals = indexed(
            self.signals.iter().map(|item| item.id.as_str()),
            "representation signals",
        )?;
        let resources = indexed(
            self.resources.iter().map(|item| item.id.as_str()),
            "representation resources",
        )?;
        let nodes = indexed(
            self.nodes.iter().map(|item| item.id.as_str()),
            "representation nodes",
        )?;
        let public_ports = indexed(
            self.public_ports.iter().map(|item| item.id.as_str()),
            "representation public ports",
        )?;
        for signal in &self.signals {
            require_reference(
                &signal.logical_contract_id,
                &logical,
                "signal logical contract",
            )?;
            require_reference(
                &signal.physical_representation_id,
                &representations,
                "signal physical representation",
            )?;
            validate_provenance(&signal.provenance, &graph_scopes, "signal")?;
        }
        for resource in &self.resources {
            require_reference(
                &resource.logical_contract_id,
                &logical,
                "resource logical contract",
            )?;
            require_reference(
                &resource.physical_representation_id,
                &representations,
                "resource physical representation",
            )?;
            validate_provenance(&resource.provenance, &graph_scopes, "resource")?;
        }
        for node in &self.nodes {
            validate_provenance(&node.provenance, &graph_scopes, "node")?;
            for port in node.inputs.iter().chain(&node.outputs) {
                require_reference(&port.signal_id, &signals, "node signal")?;
                require_reference(
                    &port.physical_representation_id,
                    &representations,
                    "node physical representation",
                )?;
                let signal = self
                    .signals
                    .iter()
                    .find(|signal| signal.id == port.signal_id)
                    .expect("signal reference was checked");
                if signal.physical_representation_id != port.physical_representation_id {
                    return Err(format!(
                        "node port {:?} has an incompatible physical representation",
                        port.id
                    ));
                }
            }
            for resource_id in node
                .resource_ids
                .iter()
                .chain(&node.state_read_ids)
                .chain(&node.state_write_ids)
            {
                require_reference(resource_id, &resources, "node resource")?;
            }
        }
        for connection in &self.connections {
            require_reference(&connection.signal_id, &signals, "connection signal")?;
            let producer = self
                .nodes
                .iter()
                .find(|node| node.id == connection.producer.node_id)
                .ok_or_else(|| format!("connection {:?} has unknown producer", connection.id))?;
            let consumer = self
                .nodes
                .iter()
                .find(|node| node.id == connection.consumer.node_id)
                .ok_or_else(|| format!("connection {:?} has unknown consumer", connection.id))?;
            let output = producer
                .outputs
                .iter()
                .find(|port| port.id == connection.producer.port_id)
                .ok_or_else(|| format!("connection {:?} has unknown output port", connection.id))?;
            let input = consumer
                .inputs
                .iter()
                .find(|port| port.id == connection.consumer.port_id)
                .ok_or_else(|| format!("connection {:?} has unknown input port", connection.id))?;
            if output.signal_id != connection.signal_id || input.signal_id != connection.signal_id {
                return Err(format!(
                    "connection {:?} endpoint signal mismatch",
                    connection.id
                ));
            }
            if output.physical_representation_id != input.physical_representation_id {
                return Err(format!(
                    "connection {:?} has incompatible physical representations",
                    connection.id
                ));
            }
        }
        for port in &self.public_ports {
            require_reference(
                &port.logical_contract_id,
                &logical,
                "public port logical contract",
            )?;
            require_reference(&port.signal_id, &signals, "public port signal")?;
            require_reference(&port.node_id, &nodes, "public port node")?;
            let signal = self
                .signals
                .iter()
                .find(|signal| signal.id == port.signal_id)
                .expect("signal reference was checked");
            if signal.logical_contract_id != port.logical_contract_id {
                return Err(format!(
                    "public port {:?} changes its logical contract",
                    port.id
                ));
            }
        }
        for kernel in &self.physical_kernels {
            validate_provenance(&kernel.provenance, &graph_scopes, "physical kernel")?;
            for node_id in &kernel.node_ids {
                require_reference(node_id, &nodes, "physical kernel node")?;
            }
        }
        for island in &self.islands {
            for node_id in &island.node_ids {
                require_reference(node_id, &nodes, "island node")?;
            }
            for representation_id in &island.representation_ids {
                require_reference(representation_id, &representations, "island representation")?;
            }
            for port_id in &island.boundary_port_ids {
                require_reference(port_id, &public_ports, "island boundary port")?;
            }
        }
        Ok(())
    }
}

fn require_text(value: &str, label: &str) -> Result<(), String> {
    if value.is_empty() {
        Err(format!("{label} must not be empty"))
    } else {
        Ok(())
    }
}

fn indexed<'a>(
    values: impl Iterator<Item = &'a str>,
    label: &str,
) -> Result<BTreeSet<String>, String> {
    unique_ids(values, label, false)
}

fn unique_ids<'a>(
    values: impl Iterator<Item = &'a str>,
    label: &str,
    nonempty: bool,
) -> Result<BTreeSet<String>, String> {
    let mut result = BTreeSet::new();
    for value in values {
        require_text(value, label)?;
        if !result.insert(value.to_string()) {
            return Err(format!("{label} contains duplicate id {value:?}"));
        }
    }
    if nonempty && result.is_empty() {
        return Err(format!("{label} must not be empty"));
    }
    Ok(result)
}

fn require_reference(value: &str, records: &BTreeSet<String>, label: &str) -> Result<(), String> {
    if records.contains(value) {
        Ok(())
    } else {
        Err(format!("{label} references unknown id {value:?}"))
    }
}

fn validate_provenance(
    provenance: &RepresentationProvenance,
    graph_scopes: &BTreeSet<String>,
    label: &str,
) -> Result<(), String> {
    let scopes = unique_ids(
        provenance.scope_ids.iter().map(String::as_str),
        &format!("{label} provenance"),
        true,
    )?;
    if !scopes.is_subset(graph_scopes) {
        return Err(format!("{label} provenance leaves graph scopes"));
    }
    if provenance.evidence_refs.is_empty() {
        return Err(format!("{label} provenance requires evidence"));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn representation_graph_test_fixture() -> RepresentationGraph {
    serde_json::from_value(serde_json::json!({
            "schema": REPRESENTATION_GRAPH_SCHEMA,
            "graph_id": "representation_graph_fixture",
            "candidate_id": "candidate_fixture",
            "scope_ids": ["scope_a", "scope_b"],
            "source_contract_digests": {
                "scope_a": "digest_a",
                "scope_b": "digest_b"
            },
            "logical_contracts": [{
                "id": "logical.hidden",
                "signal": "hidden",
                "shape": [8],
                "dtype": "BF16"
            }],
            "physical_representations": [{
                "id": "repr.dense",
                "kind": "dense",
                "domain": "signal",
                "physical_shape": [8],
                "encoding": {"dtype": "BF16"},
                "storage": {"layout": "contiguous"}
            }, {
                "id": "repr.spectral",
                "kind": "spectral",
                "domain": "signal",
                "physical_shape": [5],
                "encoding": {"basis": "fft"},
                "storage": {"layout": "packed"}
            }],
            "signals": [{
                "id": "signal.native",
                "logical_contract_id": "logical.hidden",
                "physical_representation_id": "repr.spectral",
                "provenance": {
                    "scope_ids": ["scope_a", "scope_b"],
                    "source_node_ids": ["source_a", "source_b"],
                    "evidence_refs": ["evidence"],
                    "transform_refs": []
                }
            }],
            "resources": [],
            "nodes": [{
                "id": "node.a",
                "kind": "operator",
                "operation": "native_a",
                "inputs": [],
                "outputs": [{
                    "id": "output",
                    "signal_id": "signal.native",
                    "physical_representation_id": "repr.spectral"
                }],
                "resource_ids": [],
                "state_read_ids": [],
                "state_write_ids": [],
                "cost": null,
                "provenance": {
                    "scope_ids": ["scope_a"],
                    "source_node_ids": ["source_a"],
                    "evidence_refs": ["evidence"],
                    "transform_refs": []
                }
            }, {
                "id": "node.b",
                "kind": "operator",
                "operation": "native_b",
                "inputs": [{
                    "id": "input",
                    "signal_id": "signal.native",
                    "physical_representation_id": "repr.spectral"
                }],
                "outputs": [],
                "resource_ids": [],
                "state_read_ids": [],
                "state_write_ids": [],
                "cost": null,
                "provenance": {
                    "scope_ids": ["scope_b"],
                    "source_node_ids": ["source_b"],
                    "evidence_refs": ["evidence"],
                    "transform_refs": []
                }
            }],
            "connections": [{
                "id": "connection.native",
                "producer": {"node_id": "node.a", "port_id": "output"},
                "consumer": {"node_id": "node.b", "port_id": "input"},
                "signal_id": "signal.native",
                "materializes_source": false
            }],
            "public_ports": [],
            "islands": [],
            "absorbed_transforms": [],
            "physical_kernels": [],
            "confidence": {
                "mode": "exact",
                "score": 1.0,
                "basis": "proof",
                "evidence_refs": ["evidence"]
            },
            "unresolved": [],
            "correction_requests": []
    }))
    .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn representation_graph_round_trips_without_losing_physical_shape() {
        let graph = representation_graph_test_fixture();
        graph.validate().unwrap();
        let encoded = serde_json::to_vec(&graph).unwrap();
        let decoded = RepresentationGraph::from_json_slice(&encoded).unwrap();

        assert_eq!(decoded, graph);
        assert_eq!(decoded.logical_contracts[0].shape, vec![8]);
        assert_eq!(decoded.physical_representations[1].physical_shape, vec![5]);
    }

    #[test]
    fn runtime_rejects_incompatible_physical_connection() {
        let mut graph = representation_graph_test_fixture();
        graph.nodes[1].inputs[0].physical_representation_id = "repr.dense".to_string();

        assert!(
            graph
                .validate()
                .unwrap_err()
                .contains("incompatible physical representation")
        );
    }
}
