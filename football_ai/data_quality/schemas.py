"""The `QualityIssue` type every data_quality check produces."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class QualityIssue:
    """One integrity-check finding, always traceable back to its source row(s).

    `severity` is "error" for a factually-impossible value (negative
    corners, shots < shots_on_target, a team playing itself) versus
    "warning" for something that's suspicious but not provably wrong (a
    duplicate-looking row, a source disagreement) -- mirrors this project's
    existing HIGH/MEDIUM_HIGH/LOW/UNKNOWN confidence vocabulary in spirit,
    kept binary here because every check below already knows for certain
    which bucket it's in.

    `subject` is deliberately loosely typed (an int match_id, a tuple key,
    or a string) -- different checks key on different things (one match,
    one event, one (match, statistic, source) triple), and forcing a single
    shape here would just push the type-narrowing problem onto every caller.
    """

    check_name: str
    severity: str
    message: str
    subject: object
    details: dict[str, object] = field(default_factory=dict)
