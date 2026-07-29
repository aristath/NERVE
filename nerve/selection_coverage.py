from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


_COVERAGE_LABELS = {
    "selection_coverage:": "turn",
    "cumulative_selection_coverage:": "cumulative",
}
_SUMMARY_LINE = re.compile(r"^  ([a-z][a-z0-9_]*)=(.+)$")
_DOMAIN_LINE = re.compile(
    r"^  domain=(?P<identity>\S+) "
    r"scope=(?P<scope>\S+) "
    r"selected=(?P<selected>\d+)/(?P<resources>\d+) "
    r"selections=(?P<selections>\d+) "
    r"resources=\[(?P<counts>[^\]]*)\]$"
)
_RESOURCE_COUNT = re.compile(r"^(?P<resource>\d+):(?P<count>\d+)$")
_PHASE_LINE = re.compile(
    r"^  phase=(?P<phase>[a-z][a-z0-9_]*) "
    r"selected=(?P<selected>\d+)/(?P<resources>\d+) "
    r"selections=(?P<selections>\d+) "
    r"domains=\[(?P<domains>[^\]]*)\]$"
)
_PHASE_DOMAIN = re.compile(
    r"^(?P<scope>[^@]+)@"
    r"(?P<component>[^/]+)/(?P<node>[^/]+)/(?P<domain>[^:]+):"
    r"(?P<selected>\d+)/(?P<resources>\d+)$"
)


class SelectionCoverageError(ValueError):
    pass


@dataclass(frozen=True)
class ResourceSelectionCount:
    resource_id: int
    selection_count: int


@dataclass(frozen=True)
class SelectionDomainCoverage:
    execution_scope: str
    component_id: str
    node_id: str
    domain_id: str
    resource_count: int
    selected_resource_count: int
    selection_count: int
    selected_resources: tuple[ResourceSelectionCount, ...]

    @property
    def identity(self) -> tuple[str, str, str, str]:
        return (
            self.execution_scope,
            self.component_id,
            self.node_id,
            self.domain_id,
        )

    @property
    def selected_resource_ids(self) -> frozenset[int]:
        return frozenset(item.resource_id for item in self.selected_resources)


@dataclass(frozen=True)
class SelectionCoverageBlock:
    kind: str
    domain_count: int
    selected_resource_count: int
    addressable_resource_count: int
    selection_count: int
    domains: tuple[SelectionDomainCoverage, ...]


@dataclass(frozen=True)
class SelectionPhaseDomainCoverage:
    execution_scope: str
    component_id: str
    node_id: str
    domain_id: str
    selected_resource_count: int
    resource_count: int


@dataclass(frozen=True)
class SelectionPhaseCoverage:
    phase: str
    selected_resource_count: int
    addressable_resource_count: int
    selection_count: int
    domains: tuple[SelectionPhaseDomainCoverage, ...]


def parse_selection_coverage_transcript(
    transcript: str,
) -> list[SelectionCoverageBlock]:
    lines = transcript.splitlines()
    blocks: list[SelectionCoverageBlock] = []
    cursor = 0
    while cursor < len(lines):
        kind = _COVERAGE_LABELS.get(lines[cursor])
        if kind is None:
            cursor += 1
            continue
        block, cursor = _parse_block(lines, cursor + 1, kind)
        blocks.append(block)
    if not blocks:
        raise SelectionCoverageError(
            "transcript contains no selection coverage reports"
        )
    return blocks


def parse_selection_phase_coverage_transcript(
    transcript: str,
) -> list[tuple[SelectionPhaseCoverage, ...]]:
    lines = transcript.splitlines()
    groups: list[tuple[SelectionPhaseCoverage, ...]] = []
    cursor = 0
    while cursor < len(lines):
        if lines[cursor] != "selection_phases:":
            cursor += 1
            continue
        cursor += 1
        phases = []
        while cursor < len(lines):
            match = _PHASE_LINE.fullmatch(lines[cursor])
            if match is None:
                break
            phases.append(_parse_phase(match))
            cursor += 1
        if not phases:
            raise SelectionCoverageError(
                "selection_phases report contains no phase lines"
            )
        names = [phase.phase for phase in phases]
        if len(set(names)) != len(names):
            raise SelectionCoverageError(
                "selection_phases report contains duplicate phase names"
            )
        groups.append(tuple(phases))
    if not groups:
        raise SelectionCoverageError(
            "transcript contains no selection phase reports"
        )
    return groups


def _parse_phase(match: re.Match[str]) -> SelectionPhaseCoverage:
    selected_resource_count = _parse_non_negative(
        match.group("selected"), "phase selected resource count"
    )
    addressable_resource_count = _parse_non_negative(
        match.group("resources"), "phase addressable resource count"
    )
    selection_count = _parse_non_negative(
        match.group("selections"), "phase selection count"
    )
    domains = []
    encoded_domains = match.group("domains")
    if encoded_domains:
        for encoded in encoded_domains.split(","):
            domain_match = _PHASE_DOMAIN.fullmatch(encoded)
            if domain_match is None:
                raise SelectionCoverageError(
                    f"invalid selection phase domain {encoded!r}"
                )
            domain_selected = _parse_non_negative(
                domain_match.group("selected"),
                "phase domain selected resource count",
            )
            domain_resources = _parse_non_negative(
                domain_match.group("resources"),
                "phase domain resource count",
            )
            if domain_selected > domain_resources:
                raise SelectionCoverageError(
                    f"phase domain {encoded!r} selects more resources than it "
                    "addresses"
                )
            domains.append(
                SelectionPhaseDomainCoverage(
                    execution_scope=domain_match.group("scope"),
                    component_id=domain_match.group("component"),
                    node_id=domain_match.group("node"),
                    domain_id=domain_match.group("domain"),
                    selected_resource_count=domain_selected,
                    resource_count=domain_resources,
                )
            )
    identities = [
        (
            domain.execution_scope,
            domain.component_id,
            domain.node_id,
            domain.domain_id,
        )
        for domain in domains
    ]
    if len(set(identities)) != len(identities):
        raise SelectionCoverageError(
            f"phase {match.group('phase')} contains duplicate domain identities"
        )
    if selected_resource_count != sum(
        domain.selected_resource_count for domain in domains
    ):
        raise SelectionCoverageError(
            f"phase {match.group('phase')} selected-resource total does not "
            "match its domains"
        )
    if addressable_resource_count != sum(
        domain.resource_count for domain in domains
    ):
        raise SelectionCoverageError(
            f"phase {match.group('phase')} addressable-resource total does not "
            "match its domains"
        )
    return SelectionPhaseCoverage(
        phase=match.group("phase"),
        selected_resource_count=selected_resource_count,
        addressable_resource_count=addressable_resource_count,
        selection_count=selection_count,
        domains=tuple(domains),
    )


def _parse_block(
    lines: Sequence[str], cursor: int, kind: str
) -> tuple[SelectionCoverageBlock, int]:
    summary: dict[str, str] = {}
    domains: list[SelectionDomainCoverage] = []
    while cursor < len(lines):
        line = lines[cursor]
        domain_match = _DOMAIN_LINE.fullmatch(line)
        if domain_match is not None:
            domains.append(_parse_domain(domain_match))
            cursor += 1
            continue
        summary_match = _SUMMARY_LINE.fullmatch(line)
        if summary_match is not None and not domains:
            key, value = summary_match.groups()
            if key in summary:
                raise SelectionCoverageError(
                    f"{kind} coverage repeats summary field {key!r}"
                )
            summary[key] = value
            cursor += 1
            continue
        break

    expected_summary = {
        "domain_count",
        "selected_resources",
        "selection_count",
    }
    if set(summary) != expected_summary:
        raise SelectionCoverageError(
            f"{kind} coverage summary fields are {sorted(summary)}, "
            f"expected {sorted(expected_summary)}"
        )
    selected_resource_count, addressable_resource_count = _parse_ratio(
        summary["selected_resources"], f"{kind} selected_resources"
    )
    domain_count = _parse_non_negative(summary["domain_count"], "domain_count")
    selection_count = _parse_non_negative(
        summary["selection_count"], "selection_count"
    )
    if domain_count != len(domains):
        raise SelectionCoverageError(
            f"{kind} coverage declares {domain_count} domains but contains "
            f"{len(domains)}"
        )
    identities = [domain.identity for domain in domains]
    if len(set(identities)) != len(identities):
        raise SelectionCoverageError(
            f"{kind} coverage contains duplicate domain identities"
        )
    if selected_resource_count != sum(
        domain.selected_resource_count for domain in domains
    ):
        raise SelectionCoverageError(
            f"{kind} selected-resource total does not match its domains"
        )
    if addressable_resource_count != sum(
        domain.resource_count for domain in domains
    ):
        raise SelectionCoverageError(
            f"{kind} addressable-resource total does not match its domains"
        )
    if selection_count != sum(domain.selection_count for domain in domains):
        raise SelectionCoverageError(
            f"{kind} selection total does not match its domains"
        )
    return (
        SelectionCoverageBlock(
            kind=kind,
            domain_count=domain_count,
            selected_resource_count=selected_resource_count,
            addressable_resource_count=addressable_resource_count,
            selection_count=selection_count,
            domains=tuple(domains),
        ),
        cursor,
    )


def _parse_domain(match: re.Match[str]) -> SelectionDomainCoverage:
    identity = match.group("identity").rsplit(".", 2)
    if len(identity) != 3 or any(not part for part in identity):
        raise SelectionCoverageError(
            f"invalid selection domain identity {match.group('identity')!r}"
        )
    component_id, node_id, domain_id = identity
    selected_resource_count = _parse_non_negative(
        match.group("selected"), "selected resource count"
    )
    resource_count = _parse_non_negative(
        match.group("resources"), "resource count"
    )
    selection_count = _parse_non_negative(
        match.group("selections"), "selection count"
    )
    selected_resources = _parse_resource_counts(
        match.group("counts"), resource_count
    )
    if selected_resource_count != len(selected_resources):
        raise SelectionCoverageError(
            f"{match.group('identity')} declares {selected_resource_count} "
            f"selected resources but lists {len(selected_resources)}"
        )
    if selection_count != sum(
        resource.selection_count for resource in selected_resources
    ):
        raise SelectionCoverageError(
            f"{match.group('identity')} selection total does not match its "
            "resource counts"
        )
    return SelectionDomainCoverage(
        execution_scope=match.group("scope"),
        component_id=component_id,
        node_id=node_id,
        domain_id=domain_id,
        resource_count=resource_count,
        selected_resource_count=selected_resource_count,
        selection_count=selection_count,
        selected_resources=selected_resources,
    )


def _parse_resource_counts(
    encoded: str, resource_count: int
) -> tuple[ResourceSelectionCount, ...]:
    if not encoded:
        return ()
    result: list[ResourceSelectionCount] = []
    previous_id = -1
    for item in encoded.split(","):
        match = _RESOURCE_COUNT.fullmatch(item)
        if match is None:
            raise SelectionCoverageError(
                f"invalid selected-resource count {item!r}"
            )
        resource_id = _parse_non_negative(
            match.group("resource"), "resource id"
        )
        selection_count = _parse_non_negative(
            match.group("count"), "resource selection count"
        )
        if resource_id >= resource_count:
            raise SelectionCoverageError(
                f"resource id {resource_id} exceeds domain size {resource_count}"
            )
        if resource_id <= previous_id:
            raise SelectionCoverageError(
                "selected resource ids must be strictly increasing"
            )
        if selection_count == 0:
            raise SelectionCoverageError(
                f"selected resource {resource_id} has a zero count"
            )
        result.append(ResourceSelectionCount(resource_id, selection_count))
        previous_id = resource_id
    return tuple(result)


def _parse_ratio(raw: str, label: str) -> tuple[int, int]:
    parts = raw.split("/")
    if len(parts) != 2:
        raise SelectionCoverageError(f"invalid {label} ratio {raw!r}")
    numerator = _parse_non_negative(parts[0], label)
    denominator = _parse_non_negative(parts[1], label)
    if numerator > denominator:
        raise SelectionCoverageError(
            f"{label} numerator {numerator} exceeds denominator {denominator}"
        )
    return numerator, denominator


def _parse_non_negative(raw: str, label: str) -> int:
    try:
        value = int(raw)
    except ValueError as error:
        raise SelectionCoverageError(f"invalid {label} {raw!r}") from error
    if value < 0:
        raise SelectionCoverageError(f"{label} must be non-negative")
    return value


def analyze_selection_coverage(
    blocks: Sequence[SelectionCoverageBlock],
    *,
    resource_bytes: int,
    turn_labels: Sequence[str] | None = None,
) -> dict[str, object]:
    if resource_bytes <= 0:
        raise SelectionCoverageError("resource_bytes must be positive")
    if len(blocks) % 2 != 0:
        raise SelectionCoverageError(
            "coverage evidence must contain one turn and one cumulative block "
            "per transaction"
        )
    turns = []
    previous_cumulative: dict[
        tuple[str, str, str, str], frozenset[int]
    ] = {}
    for turn_index in range(len(blocks) // 2):
        turn_block = blocks[turn_index * 2]
        cumulative_block = blocks[turn_index * 2 + 1]
        if turn_block.kind != "turn" or cumulative_block.kind != "cumulative":
            raise SelectionCoverageError(
                "coverage blocks must alternate turn then cumulative"
            )
        turn_domains = {domain.identity: domain for domain in turn_block.domains}
        cumulative_domains = {
            domain.identity: domain for domain in cumulative_block.domains
        }
        if set(turn_domains) != set(cumulative_domains):
            raise SelectionCoverageError(
                f"turn {turn_index} domain identities differ between turn and "
                "cumulative reports"
            )
        domain_reports = []
        for identity in sorted(turn_domains):
            turn_domain = turn_domains[identity]
            cumulative_domain = cumulative_domains[identity]
            current = cumulative_domain.selected_resource_ids
            prior = previous_cumulative.get(identity, frozenset())
            if not prior.issubset(current):
                raise SelectionCoverageError(
                    f"turn {turn_index} cumulative coverage regressed for "
                    f"{identity}"
                )
            selected = turn_domain.selected_resource_ids
            if not selected.issubset(current):
                raise SelectionCoverageError(
                    f"turn {turn_index} selected resources are absent from "
                    f"cumulative coverage for {identity}"
                )
            new = current - prior
            reused = selected & prior
            hottest = sorted(
                turn_domain.selected_resources,
                key=lambda item: (-item.selection_count, item.resource_id),
            )[:8]
            domain_reports.append(
                {
                    "execution_scope": turn_domain.execution_scope,
                    "component_id": turn_domain.component_id,
                    "node_id": turn_domain.node_id,
                    "domain_id": turn_domain.domain_id,
                    "resource_count": turn_domain.resource_count,
                    "turn_selected_resource_count": len(selected),
                    "new_resource_count": len(new),
                    "reused_resource_count": len(reused),
                    "cumulative_resource_count": len(current),
                    "turn_selection_count": turn_domain.selection_count,
                    "hot_resources": [asdict(item) for item in hottest],
                }
            )
            previous_cumulative[identity] = current
        scope_reports = []
        scopes = sorted(
            {domain.execution_scope for domain in cumulative_block.domains}
        )
        for scope in scopes:
            matching = [
                domain
                for domain in domain_reports
                if domain["execution_scope"] == scope
            ]
            cumulative_count = sum(
                int(domain["cumulative_resource_count"]) for domain in matching
            )
            scope_reports.append(
                {
                    "execution_scope": scope,
                    "domain_count": len(matching),
                    "addressable_resource_count": sum(
                        int(domain["resource_count"]) for domain in matching
                    ),
                    "turn_selected_resource_count": sum(
                        int(domain["turn_selected_resource_count"])
                        for domain in matching
                    ),
                    "new_resource_count": sum(
                        int(domain["new_resource_count"]) for domain in matching
                    ),
                    "reused_resource_count": sum(
                        int(domain["reused_resource_count"]) for domain in matching
                    ),
                    "cumulative_resource_count": cumulative_count,
                    "cumulative_resource_bytes": cumulative_count
                    * resource_bytes,
                    "turn_selection_count": sum(
                        int(domain["turn_selection_count"])
                        for domain in matching
                    ),
                }
            )
        turns.append(
            {
                "turn_index": turn_index,
                "label": (
                    turn_labels[turn_index]
                    if turn_labels is not None and turn_index < len(turn_labels)
                    else f"turn_{turn_index}"
                ),
                "scope_summaries": scope_reports,
                "domains": domain_reports,
            }
        )
    return {
        "schema": "nerve.selection_coverage_evidence.v1",
        "resource_bytes": resource_bytes,
        "turn_count": len(turns),
        "turns": turns,
    }


def compact_selection_coverage_evidence(
    report: dict[str, object],
) -> dict[str, object]:
    raw_turns = report.get("turns")
    if not isinstance(raw_turns, list) or not raw_turns:
        raise SelectionCoverageError(
            "selection coverage report contains no turns"
        )
    domain_ids_by_scope: dict[str, list[str]] = {}
    for raw_turn in raw_turns:
        if not isinstance(raw_turn, dict):
            raise SelectionCoverageError("selection coverage turn is not an object")
        raw_domains = raw_turn.get("domains")
        if not isinstance(raw_domains, list):
            raise SelectionCoverageError(
                "selection coverage turn contains no domains"
            )
        current: dict[str, list[str]] = {}
        for domain in raw_domains:
            if not isinstance(domain, dict):
                raise SelectionCoverageError(
                    "selection coverage domain is not an object"
                )
            scope = str(domain["execution_scope"])
            identity = "/".join(
                (
                    str(domain["component_id"]),
                    str(domain["node_id"]),
                    str(domain["domain_id"]),
                )
            )
            current.setdefault(scope, []).append(identity)
        for scope in current:
            current[scope].sort()
        if not domain_ids_by_scope:
            domain_ids_by_scope = current
        elif current != domain_ids_by_scope:
            raise SelectionCoverageError(
                "selection coverage domain order changes between turns"
            )

    compact_turns = []
    for raw_turn in raw_turns:
        raw_domains = raw_turn["domains"]
        scope_summaries = {
            str(summary["execution_scope"]): summary
            for summary in raw_turn["scope_summaries"]
        }
        compact_scopes = []
        for scope, domain_ids in sorted(domain_ids_by_scope.items()):
            domains = {
                "/".join(
                    (
                        str(domain["component_id"]),
                        str(domain["node_id"]),
                        str(domain["domain_id"]),
                    )
                ): domain
                for domain in raw_domains
                if str(domain["execution_scope"]) == scope
            }
            ordered = [domains[domain_id] for domain_id in domain_ids]
            summary = dict(scope_summaries[scope])
            summary.pop("execution_scope", None)
            compact_scopes.append(
                {
                    "execution_scope": scope,
                    "summary": summary,
                    "turn_selected_resource_counts": [
                        domain["turn_selected_resource_count"]
                        for domain in ordered
                    ],
                    "new_resource_counts": [
                        domain["new_resource_count"] for domain in ordered
                    ],
                    "reused_resource_counts": [
                        domain["reused_resource_count"] for domain in ordered
                    ],
                    "cumulative_resource_counts": [
                        domain["cumulative_resource_count"] for domain in ordered
                    ],
                    "turn_selection_counts": [
                        domain["turn_selection_count"] for domain in ordered
                    ],
                    "hottest_resources": [
                        [
                            [
                                resource["resource_id"],
                                resource["selection_count"],
                            ]
                            for resource in domain["hot_resources"][:3]
                        ]
                        for domain in ordered
                    ],
                }
            )
        compact_turns.append(
            {
                "turn_index": raw_turn["turn_index"],
                "label": raw_turn["label"],
                "scopes": compact_scopes,
            }
        )
    return {
        "schema": "nerve.selection_coverage_evidence.compact.v1",
        "resource_bytes": report["resource_bytes"],
        "turn_count": report["turn_count"],
        "domain_ids_by_scope": domain_ids_by_scope,
        "turns": compact_turns,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze normal NERVE chat selection-coverage reports."
    )
    parser.add_argument("transcript", type=Path)
    parser.add_argument("--resource-bytes", type=int, required=True)
    parser.add_argument("--turn-label", action="append", default=[])
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    blocks = parse_selection_coverage_transcript(
        args.transcript.read_text(errors="replace")
    )
    report = analyze_selection_coverage(
        blocks,
        resource_bytes=args.resource_bytes,
        turn_labels=args.turn_label or None,
    )
    if args.compact:
        report = compact_selection_coverage_evidence(report)
    if args.compact:
        encoded = json.dumps(
            report, sort_keys=True, separators=(",", ":")
        ) + "\n"
    else:
        encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
