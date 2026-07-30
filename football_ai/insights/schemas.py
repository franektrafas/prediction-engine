"""The `Fact` type every insights submodule produces."""

from __future__ import annotations

from dataclasses import dataclass, field


class MissingProvenanceError(ValueError):
    """Raised when a `Fact` would be constructed without match_id provenance."""


@dataclass(slots=True, frozen=True)
class Fact:
    """A single, provenance-carrying piece of match/team history.

    Non-negotiable per PLAN_SILNIKA.md: a `Fact` without `provenance` must
    never leave this package. Enforced here, at construction time, not left
    as a convention callers might forget -- `__post_init__` raises rather
    than silently allowing an empty list through.

    `summary` is a short, plain-English/Polish description for debugging and
    golden tests -- e.g. "Poland's longest unbeaten run is 12 matches
    (2021-2023)". It is deliberately *not* the polished, template-rendered
    narrative sentence Krok 4 (`narrative/`) produces later; this layer only
    guarantees the fact is true and reproducible, not that it reads well.

    `as_of_match_id` records the point-in-time cutoff a fact was computed
    relative to (the same match_id space as `provenance`) -- most detector
    functions accept an `as_of` cutoff so a preview generated before match N
    can never surface information only known after N. `None` means the fact
    reflects the full dataset as of whenever it was computed (e.g. an
    all-time record), not scoped to a specific upcoming match.
    """

    fact_type: str
    subject_team: str
    summary: str
    data: dict[str, object]
    weight: float
    provenance: tuple[int, ...]
    as_of_match_id: int | None = None
    related_teams: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Enforce the provenance invariant at construction time."""
        if not self.provenance:
            raise MissingProvenanceError(
                f"Fact(fact_type={self.fact_type!r}, subject_team={self.subject_team!r}) "
                "has no provenance -- every Fact must carry the match_ids it was derived from."
            )
