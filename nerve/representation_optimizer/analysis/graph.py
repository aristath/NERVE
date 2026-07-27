from __future__ import annotations

from collections import Counter, defaultdict, deque

from nerve.representation_optimizer.analysis.claims import AnalyzerResult, claim
from nerve.representation_optimizer.analysis.context import ScopeAnalysisContext
from nerve.representation_optimizer.contracts import contract_digest


class GraphStructureAnalyzer:
    analyzer_id = "semantic_graph_structure"
    version = "2"

    def analyze(self, context: ScopeAnalysisContext) -> AnalyzerResult:
        nodes = {str(node["id"]): node for node in context.nodes}
        operators = _operator_inventory(nodes)
        producer: dict[str, str] = {}
        consumers: dict[str, list[str]] = defaultdict(list)
        for node_id, node in nodes.items():
            for output in node.get("outputs", []):
                producer[str(output)] = node_id
            for input_id in node.get("inputs", []):
                consumers[str(input_id)].append(node_id)

        adjacency: dict[str, set[str]] = {node_id: set() for node_id in nodes}
        edges = []
        for signal_id, source in producer.items():
            for destination in consumers.get(signal_id, []):
                adjacency[source].add(destination)
                adjacency[destination].add(source)
                edges.append((source, destination, signal_id))

        communities = _connected_components(adjacency)
        routes = _routing_nodes(nodes)
        signatures = Counter(_node_signature(node) for node in nodes.values())
        repeated_signatures = {
            signature: count for signature, count in signatures.items() if count > 1
        }
        fanout = {
            signal: len(destinations)
            for signal, destinations in consumers.items()
            if len(destinations) > 1
        }
        claims = (
            claim(
                kind="operator_structure",
                status="supported" if operators else "rejected",
                exact=True,
                facts={
                    "node_count": len(operators),
                    "operators": operators,
                },
            ),
            claim(
                kind="graph_communities",
                status="supported" if len(communities) > 1 else "rejected",
                exact=True,
                facts={
                    "community_count": len(communities),
                    "communities": communities,
                    "node_count": len(nodes),
                    "internal_edge_count": len(edges),
                },
            ),
            claim(
                kind="routing_structure",
                status="supported" if routes or fanout else "rejected",
                exact=True,
                facts={
                    "routing_nodes": routes,
                    "fanout_signals": fanout,
                },
            ),
            claim(
                kind="graph_common_subexpression",
                status=("supported" if repeated_signatures else "rejected"),
                exact=True,
                facts={"repeated_node_signatures": repeated_signatures},
            ),
        )
        return AnalyzerResult(
            claims=claims,
            details={
                "nodes": sorted(nodes),
                "edges": [list(edge) for edge in sorted(edges)],
                "communities": communities,
                "operators": operators,
                "routing_nodes": routes,
                "repeated_node_signatures": repeated_signatures,
            },
        )


def _connected_components(
    adjacency: dict[str, set[str]],
) -> list[list[str]]:
    remaining = set(adjacency)
    result = []
    while remaining:
        start = min(remaining)
        queue = deque([start])
        component = []
        remaining.remove(start)
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        result.append(sorted(component))
    return sorted(result)


def _routing_nodes(nodes: dict[str, dict]) -> list[dict]:
    routes = []
    vocabulary = ("route", "select", "dispatch", "expert", "switch", "topk")
    for node_id, node in nodes.items():
        semantic_text = " ".join(
            (
                str(node.get("op", "")),
                str(node.get("attrs", "")),
                str(node.get("semantic_role", "")),
            )
        ).casefold()
        matched = [term for term in vocabulary if term in semantic_text]
        if matched:
            routes.append(
                {
                    "node_id": node_id,
                    "semantic_terms": matched,
                    "output_count": len(node.get("outputs", [])),
                }
            )
    return sorted(routes, key=lambda route: route["node_id"])


def _operator_inventory(nodes: dict[str, dict]) -> list[dict]:
    inventory = []
    for node_id in sorted(nodes):
        node = nodes[node_id]
        record = {
            "node_id": node_id,
            "op": str(node.get("op", "")),
            "inputs": [str(item) for item in node.get("inputs", [])],
            "outputs": [str(item) for item in node.get("outputs", [])],
            "params": [str(item) for item in node.get("params", [])],
            "attrs": node.get("attrs", {}),
        }
        for optional in ("component_id", "semantic_role"):
            value = node.get(optional)
            if value is not None:
                record[optional] = str(value)
        inventory.append(record)
    return inventory


def _node_signature(node: dict) -> str:
    return contract_digest(
        {
            "op": node.get("op"),
            "inputs": node.get("inputs", []),
            "params": node.get("params", []),
            "attrs": node.get("attrs", {}),
        }
    )
