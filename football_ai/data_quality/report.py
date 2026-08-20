"""Data Quality Report: coverage by period/competition/source + issue summary.

Matches the Data Engine plan's example format:

    1980-1989: shots 12%  corners 18%  cards 72%
    2010-2019: shots 87%  corners 91%  cards 98%
    2020-2026: shots 99%  corners 99%  cards 100%  xG 94%

A stat is "covered" for a row only when BOTH sides are known -- a lone
`home_shots` with a missing `away_shots` isn't usable for a home/away
feature pair, so counting it as covered would overstate real usability.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import pandas as pd

from football_ai.data_quality import provenance_checks, validators
from football_ai.data_quality.schemas import QualityIssue

# display name -> (home column, away column). Every pair is optional --
# a records_frame missing a pair (e.g. no xg columns) just skips it rather
# than raising, per this project's "not every field needs every source" rule.
DEFAULT_STAT_PAIRS: dict[str, tuple[str, str]] = {
    "shots": ("home_shots", "away_shots"),
    "shots_on_target": ("home_shots_on_target", "away_shots_on_target"),
    "corners": ("home_corners", "away_corners"),
    "cards": ("home_yellow_cards", "away_yellow_cards"),
    "possession": ("home_possession_pct", "away_possession_pct"),
    "xg": ("home_xg", "away_xg"),
}


@dataclass(slots=True, frozen=True)
class DataQualityReport:
    """The full output of `generate_report`."""

    total_matches: int
    coverage_by_period: pd.DataFrame
    coverage_by_tournament: pd.DataFrame
    issues: list[QualityIssue]
    issue_counts_by_check: dict[str, int] = field(default_factory=dict)

    def error_count(self) -> int:
        """Number of "error"-severity issues found."""
        return sum(1 for issue in self.issues if issue.severity == "error")

    def warning_count(self) -> int:
        """Number of "warning"-severity issues found."""
        return sum(1 for issue in self.issues if issue.severity == "warning")


def _period_label(year: int, period_years: int) -> str:
    """Bucket a year into a period label, e.g. (1985, 10) -> "1980-1989"."""
    start = (year // period_years) * period_years
    return f"{start}-{start + period_years - 1}"


def coverage_by_period(
    records_frame: pd.DataFrame,
    stat_pairs: dict[str, tuple[str, str]] | None = None,
    period_years: int = 10,
) -> pd.DataFrame:
    """Coverage % per stat, bucketed into `period_years`-year periods.

    Returns a DataFrame indexed by period label, one column per stat present
    in `stat_pairs` AND actually found in `records_frame` (columns for pairs
    the frame doesn't have are simply absent from the result, not zero).
    """
    stat_pairs = stat_pairs or DEFAULT_STAT_PAIRS
    frame = records_frame.copy()
    frame["_period"] = frame["match_date"].dt.year.map(
        lambda year: _period_label(year, period_years)
    )
    return _coverage_by_group(frame, "_period", stat_pairs)


def coverage_by_tournament(
    records_frame: pd.DataFrame, stat_pairs: dict[str, tuple[str, str]] | None = None
) -> pd.DataFrame:
    """Coverage % per stat, grouped by `tournament`, largest tournaments first."""
    stat_pairs = stat_pairs or DEFAULT_STAT_PAIRS
    if "tournament" not in records_frame.columns:
        return pd.DataFrame()
    coverage = _coverage_by_group(records_frame, "tournament", stat_pairs)
    return coverage.sort_values("match_count", ascending=False)


def _coverage_by_group(
    frame: pd.DataFrame, group_column: str, stat_pairs: dict[str, tuple[str, str]]
) -> pd.DataFrame:
    """Build the coverage table -- always carries a `match_count` metadata column.

    `match_count` isn't a coverage percentage -- `format_coverage_report`
    treats it specially rather than printing it as a fake "match_count 830%".
    """
    available_pairs = {
        name: (home_col, away_col)
        for name, (home_col, away_col) in stat_pairs.items()
        if home_col in frame.columns and away_col in frame.columns
    }
    rows: dict[str, dict[str, float]] = {}
    for group_key, group in frame.groupby(group_column, dropna=False):
        row: dict[str, float] = {"match_count": len(group)}
        for stat_name, (home_col, away_col) in available_pairs.items():
            both_known = group[home_col].notna() & group[away_col].notna()
            row[stat_name] = 100.0 * both_known.mean() if len(group) else 0.0
        rows[group_key] = row
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def format_coverage_report(coverage: pd.DataFrame) -> str:
    """Render a coverage-by-period table as the plan's example text format."""
    lines = []
    for period, row in coverage.iterrows():
        parts = [
            f"{stat} {value:.0f}%"
            for stat, value in row.items()
            if stat != "match_count"
        ]
        lines.append(f"{period}: " + "  ".join(parts))
    return "\n".join(lines)


def generate_report(
    records_frame: pd.DataFrame,
    goalscorers_frame: pd.DataFrame | None = None,
    provenance_frame: pd.DataFrame | None = None,
    as_of: pd.Timestamp | None = None,
    period_years: int = 10,
) -> DataQualityReport:
    """Run every integrity check plus the coverage breakdowns, in one report."""
    issues = validators.run_all_checks(records_frame, goalscorers_frame, as_of)
    if provenance_frame is not None:
        issues.extend(provenance_checks.run_all_checks(provenance_frame))

    issue_counts: dict[str, int] = {}
    for issue in issues:
        issue_counts[issue.check_name] = issue_counts.get(issue.check_name, 0) + 1

    return DataQualityReport(
        total_matches=len(records_frame),
        coverage_by_period=coverage_by_period(records_frame, period_years=period_years),
        coverage_by_tournament=coverage_by_tournament(records_frame),
        issues=issues,
        issue_counts_by_check=issue_counts,
    )


def print_report(
    quality_report: DataQualityReport, top_n_tournaments: int = 15
) -> None:
    """Print a human-readable rendering of a `DataQualityReport` to stdout.

    Meant to run after every import, per the Data Engine plan -- a quick,
    skimmable summary a human reviews before trusting a new ingest, not a
    replacement for inspecting `quality_report.issues` programmatically.
    """
    print(f"Total matches: {quality_report.total_matches}")
    print(
        f"Errors: {quality_report.error_count()}  "
        f"Warnings: {quality_report.warning_count()}"
    )
    print()
    print("Coverage by period:")
    print(format_coverage_report(quality_report.coverage_by_period))
    print()
    print(f"Coverage by tournament (top {top_n_tournaments} by row count):")
    by_tournament = quality_report.coverage_by_tournament
    if not by_tournament.empty:
        print(format_coverage_report(by_tournament.head(top_n_tournaments)))
    print()
    print("Issues by check:")
    for check_name, count in sorted(
        quality_report.issue_counts_by_check.items(), key=lambda item: -item[1]
    ):
        print(f"  {check_name}: {count}")


if __name__ == "__main__":
    from football_ai.data.io import read_table
    from football_ai.scrapers.sources.international_results import (
        InternationalResultsSource,
    )

    parser = argparse.ArgumentParser(
        description="Generate a Data Quality Report for a records_frame."
    )
    parser.add_argument(
        "--records-frame",
        default="football_ai/data/processed/international_results_full_history.parquet",
    )
    parser.add_argument(
        "--include-goalscorers",
        action="store_true",
        help="Also fetch and check martj42's goalscorers.csv (network access).",
    )
    arguments = parser.parse_args()

    records = read_table(arguments.records_frame)
    goalscorers = None
    if arguments.include_goalscorers:
        source = InternationalResultsSource()
        goalscorers = source.normalize_goalscorers(source.fetch_goalscorers(), records)

    generated_report = generate_report(
        records, goalscorers_frame=goalscorers, as_of=pd.Timestamp.now(tz="UTC")
    )
    print_report(generated_report)
