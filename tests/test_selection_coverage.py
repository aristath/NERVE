from __future__ import annotations

import pytest

from nerve.selection_coverage import (
    SelectionCoverageError,
    analyze_selection_coverage,
    compact_selection_coverage_evidence,
    parse_selection_coverage_transcript,
    parse_selection_phase_coverage_transcript,
)


def _block(
    label: str,
    *,
    first_counts: str,
    second_counts: str,
) -> str:
    first_total = sum(
        int(item.split(":")[1]) for item in first_counts.split(",") if item
    )
    second_total = sum(
        int(item.split(":")[1]) for item in second_counts.split(",") if item
    )
    first_selected = 0 if not first_counts else len(first_counts.split(","))
    second_selected = 0 if not second_counts else len(second_counts.split(","))
    return "\n".join(
        (
            f"{label}:",
            "  domain_count=2",
            f"  selected_resources={first_selected + second_selected}/8",
            f"  selection_count={first_total + second_total}",
            "  domain=component_0.selector.resources "
            f"scope=target selected={first_selected}/4 "
            f"selections={first_total} resources=[{first_counts}]",
            "  domain=component_1.selector.resources "
            f"scope=target selected={second_selected}/4 "
            f"selections={second_total} resources=[{second_counts}]",
        )
    )


def test_analyzes_new_reused_hot_and_cumulative_resources() -> None:
    transcript = "\n".join(
        (
            _block(
                "selection_coverage",
                first_counts="1:3",
                second_counts="2:2",
            ),
            _block(
                "cumulative_selection_coverage",
                first_counts="1:3",
                second_counts="2:2",
            ),
            _block(
                "selection_coverage",
                first_counts="1:5,3:1",
                second_counts="0:4",
            ),
            _block(
                "cumulative_selection_coverage",
                first_counts="1:8,3:1",
                second_counts="0:4,2:2",
            ),
        )
    )

    report = analyze_selection_coverage(
        parse_selection_coverage_transcript(transcript),
        resource_bytes=1024,
        turn_labels=("warmup", "measured"),
    )

    assert report["turn_count"] == 2
    second = report["turns"][1]
    assert second["label"] == "measured"
    assert second["scope_summaries"] == [
        {
            "execution_scope": "target",
            "domain_count": 2,
            "addressable_resource_count": 8,
            "turn_selected_resource_count": 3,
            "new_resource_count": 2,
            "reused_resource_count": 1,
            "cumulative_resource_count": 4,
            "cumulative_resource_bytes": 4096,
            "turn_selection_count": 10,
        }
    ]
    assert second["domains"][0]["new_resource_count"] == 1
    assert second["domains"][0]["reused_resource_count"] == 1
    assert second["domains"][0]["hot_resources"][0] == {
        "resource_id": 1,
        "selection_count": 5,
    }


@pytest.mark.parametrize(
    "replacement, message",
    (
        ("selected_resources=3/8", "selected-resource total"),
        ("selection_count=99", "selection total"),
        ("selected=2/4 selections=3 resources=[1:3]", "declares 2"),
        ("resources=[1:2,1:1]", "strictly increasing"),
        ("resources=[4:3]", "exceeds domain size"),
    ),
)
def test_rejects_internally_inconsistent_coverage(
    replacement: str, message: str
) -> None:
    valid = _block(
        "selection_coverage",
        first_counts="1:3",
        second_counts="2:2",
    )
    if replacement.startswith("selected_resources"):
        invalid = valid.replace("selected_resources=2/8", replacement)
    elif replacement.startswith("selection_count"):
        invalid = valid.replace("selection_count=5", replacement)
    elif replacement.startswith("selected="):
        invalid = valid.replace(
            "selected=1/4 selections=3 resources=[1:3]", replacement
        )
    else:
        invalid = valid.replace("resources=[1:3]", replacement, 1)

    with pytest.raises(SelectionCoverageError, match=message):
        parse_selection_coverage_transcript(invalid)


def test_rejects_non_alternating_turn_and_cumulative_blocks() -> None:
    block = _block(
        "selection_coverage",
        first_counts="1:3",
        second_counts="2:2",
    )

    with pytest.raises(SelectionCoverageError, match="alternate"):
        analyze_selection_coverage(
            parse_selection_coverage_transcript("\n".join((block, block))),
            resource_bytes=1,
        )


def test_parses_compact_selection_phase_frontiers() -> None:
    transcript = "\n".join(
        (
            "selection_phases:",
            "  phase=user selected=2/8 selections=5 "
            "domains=[target@component_0/selector/resources:1/4,"
            "target@component_1/selector/resources:1/4]",
            "  phase=post_generation_cumulative selected=3/8 selections=9 "
            "domains=[target@component_0/selector/resources:2/4,"
            "target@component_1/selector/resources:1/4]",
        )
    )

    phases = parse_selection_phase_coverage_transcript(transcript)

    assert len(phases) == 1
    assert phases[0][0].phase == "user"
    assert phases[0][1].selected_resource_count == 3
    assert phases[0][1].domains[0].component_id == "component_0"


def test_rejects_selection_phase_summary_mismatch() -> None:
    transcript = "\n".join(
        (
            "selection_phases:",
            "  phase=user selected=2/4 selections=5 "
            "domains=[target@component_0/selector/resources:1/4]",
        )
    )

    with pytest.raises(SelectionCoverageError, match="selected-resource total"):
        parse_selection_phase_coverage_transcript(transcript)


def test_compacts_domains_into_stable_aligned_arrays() -> None:
    transcript = "\n".join(
        (
            _block(
                "selection_coverage",
                first_counts="1:3",
                second_counts="2:2",
            ),
            _block(
                "cumulative_selection_coverage",
                first_counts="1:3",
                second_counts="2:2",
            ),
        )
    )
    full = analyze_selection_coverage(
        parse_selection_coverage_transcript(transcript),
        resource_bytes=1024,
    )

    compact = compact_selection_coverage_evidence(full)

    assert compact["domain_ids_by_scope"] == {
        "target": [
            "component_0/selector/resources",
            "component_1/selector/resources",
        ]
    }
    scope = compact["turns"][0]["scopes"][0]
    assert scope["turn_selected_resource_counts"] == [1, 1]
    assert scope["cumulative_resource_counts"] == [1, 1]
    assert scope["hottest_resources"] == [[[1, 3]], [[2, 2]]]
