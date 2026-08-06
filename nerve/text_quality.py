from __future__ import annotations

import re
from collections import defaultdict


_WORD = re.compile(r"\S+")
_LONG_REPEAT_WINDOW = 32_768
_LONG_REPEAT_ANCHOR = 32
_LONG_REPEAT_MINIMUM_UNIT = 80
_LONG_REPEAT_MAXIMUM_UNIT = 8_192
_DIAGNOSTIC_LIMIT = 256


def _diagnostic_excerpt(segment: str) -> str:
    segment = segment.strip()
    if len(segment) <= _DIAGNOSTIC_LIMIT:
        return segment
    return segment[: _DIAGNOSTIC_LIMIT - 3].rstrip() + "..."


def _long_repeated_segment(text: str, minimum_repeats: int) -> str | None:
    recent = text[-_LONG_REPEAT_WINDOW:]
    if len(recent) < _LONG_REPEAT_MINIMUM_UNIT * minimum_repeats:
        return None

    maximum_unit = min(
        _LONG_REPEAT_MAXIMUM_UNIT,
        len(recent) // minimum_repeats,
    )
    anchor_positions: dict[str, list[int]] = defaultdict(list)
    last_anchor_start = len(recent) - _LONG_REPEAT_ANCHOR
    for start in range(last_anchor_start + 1):
        anchor_positions[recent[start : start + _LONG_REPEAT_ANCHOR]].append(start)

    for positions in anchor_positions.values():
        if len(positions) < minimum_repeats:
            continue
        position_set = set(positions)
        for position_index, start in enumerate(positions[:-minimum_repeats + 1]):
            for second in positions[position_index + 1 :]:
                unit_width = second - start
                if unit_width < _LONG_REPEAT_MINIMUM_UNIT:
                    continue
                if unit_width > maximum_unit:
                    break
                if any(
                    start + unit_width * repeat not in position_set
                    for repeat in range(2, minimum_repeats)
                ):
                    continue
                end = start + unit_width * minimum_repeats
                if end > len(recent):
                    break
                unit = recent[start:second]
                if recent[second:end] == unit * (minimum_repeats - 1):
                    return _diagnostic_excerpt(unit)
    return None


def repeated_segment(text: str, minimum_repeats: int = 4) -> str | None:
    if minimum_repeats < 2:
        raise ValueError("minimum_repeats must be at least two")

    normalized = re.sub(r"\s+", " ", text).strip()
    for width in (8, 12, 16, 24, 32, 48, 64, 96, 128, 192):
        if len(normalized) < width * minimum_repeats:
            continue
        suffix = normalized[-width:]
        if suffix.strip() and normalized.endswith(suffix * minimum_repeats):
            return _diagnostic_excerpt(suffix)

    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    maximum_line_width = min(len(lines) // minimum_repeats, 512)
    for width in range(1, maximum_line_width + 1):
        suffix = lines[-width:]
        if all(
            lines[-width * repeat : -width * (repeat - 1)] == suffix
            for repeat in range(2, minimum_repeats + 1)
        ):
            return _diagnostic_excerpt("\n".join(suffix))

    words = _WORD.findall(normalized)
    for width in (*range(4, 17), 24, 32, 48, 64):
        if len(words) < width * minimum_repeats:
            continue
        suffix = words[-width:]
        if all(
            words[-width * repeat : -width * (repeat - 1)] == suffix
            for repeat in range(2, minimum_repeats + 1)
        ):
            return _diagnostic_excerpt(" ".join(suffix))

    return _long_repeated_segment(normalized, minimum_repeats)
