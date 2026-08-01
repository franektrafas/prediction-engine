"""Merge supplemental advanced-stat sources into a base wide matches frame."""

from __future__ import annotations

import pandas as pd

from football_ai.data.normalization import canonicalize_team_name


_MERGE_KEY_COLUMNS = ("_merge_home_team", "_merge_away_team", "match_date")
_IDENTITY_COLUMNS = ("home_team_name", "away_team_name", "home_score", "away_score")


def merge_advanced_stats(base_matches: pd.DataFrame, supplemental_matches: pd.DataFrame) -> pd.DataFrame:
    """Left-merge a supplemental wide matches frame's stat columns onto a base frame.

    The base frame is always the source of truth for match identity (date,
    teams, score) -- supplemental columns only fill in where the base frame
    doesn't already have a non-null value for that column, so layering a
    richer future source on top can never silently override a value the base
    data already provides. Matching is on (canonicalized home team,
    canonicalized away team, match date) rather than raw team-name strings,
    since different sources spell team names differently (see
    `config.team_aliases`). Not specific to any one source -- reusable for
    any future supplemental frame (StatsBomb, FBref, ...) with the same wide
    `home_*`/`away_*` column convention `features.builder` expects.

    Also matches with home/away swapped: sources disagree on which side of a
    neutral-venue match counts as "home" (verified in real data -- StatsBomb
    has 2018 World Cup Uruguay v Russia as Uruguay-home, the base file has it
    Russia-home). A supplemental row that doesn't match the base's own
    home/away order still merges, with its `home_*`/`away_*` stat columns
    swapped to match the base's orientation.

    Falls back to a +/-1 calendar day tolerance for rows the exact date
    doesn't match -- CONFIRMED REAL (found 2026-08-01 diagnosing why only
    285 of 511 real, fetched StatsBomb international matches were merging):
    Copa America 2024 was hosted in the US, and a match kicking off in the
    evening US local time lands on the *next* UTC calendar day -- StatsBomb
    records e.g. Argentina v Canada as 2024-06-21, this project's own base
    results have it as 2024-06-20, both correct under a different date
    convention for the same real match. The fallback only fires for rows the
    exact-date pass genuinely found no merge-key match for (tracked via
    pandas' own merge indicator, not by checking whether the stat columns
    are still null -- CONFIRMED REAL BUG caught testing this fix: a
    supplemental frame can carry a column name the base frame already
    populates from its own data for every row regardless of any merge, e.g.
    `tournament`/`neutral_venue` -- "null-check" would have reported every
    row as already matched and the fallback would silently never run). Never
    overwrites a row the exact pass already found, so this can only fill
    genuine gaps, never introduce ambiguity between two real matches.
    """
    if base_matches.empty or supplemental_matches.empty:
        return base_matches.copy()

    base = base_matches.copy()
    base["match_date"] = pd.to_datetime(base["match_date"], utc=True)
    base["_merge_home_team"] = base["home_team_name"].map(canonicalize_team_name)
    base["_merge_away_team"] = base["away_team_name"].map(canonicalize_team_name)

    supplemental = supplemental_matches.copy()
    supplemental["match_date"] = pd.to_datetime(supplemental["match_date"], utc=True)
    supplemental["_merge_home_team"] = supplemental["home_team_name"].map(canonicalize_team_name)
    supplemental["_merge_away_team"] = supplemental["away_team_name"].map(canonicalize_team_name)

    supplemental_stat_columns = [
        column
        for column in supplemental.columns
        if column not in _IDENTITY_COLUMNS and column not in _MERGE_KEY_COLUMNS
    ]
    supplemental_slice = supplemental[[*_MERGE_KEY_COLUMNS, *supplemental_stat_columns]].drop_duplicates(
        subset=list(_MERGE_KEY_COLUMNS), keep="last"
    )

    swapped_slice = supplemental_slice.rename(
        columns={"_merge_home_team": "_merge_away_team", "_merge_away_team": "_merge_home_team"}
        | {
            column: ("away_" + column[len("home_"):] if column.startswith("home_") else "home_" + column[len("away_"):])
            for column in supplemental_stat_columns
            if column.startswith("home_") or column.startswith("away_")
        }
    )
    candidate_slice = pd.concat([supplemental_slice, swapped_slice], ignore_index=True).drop_duplicates(
        subset=list(_MERGE_KEY_COLUMNS), keep="first"
    )

    merged, found = _left_merge_stats(base, candidate_slice, supplemental_stat_columns)

    still_unmatched = ~found
    for offset_days in (-1, 1):
        if not still_unmatched.any():
            break
        shifted_base = base.loc[still_unmatched].copy()
        shifted_base["match_date"] = shifted_base["match_date"] + pd.Timedelta(days=offset_days)
        fallback, fallback_found = _left_merge_stats(shifted_base, candidate_slice, supplemental_stat_columns)
        newly_found = fallback_found[fallback_found].index
        for column in supplemental_stat_columns:
            merged.loc[newly_found, column] = fallback.loc[newly_found, column].to_numpy()
        still_unmatched.loc[newly_found] = False

    # Only drop the synthetic team-key helper columns -- match_date is also a
    # merge key but is a real output column too (unlike the `_merge_*` pair),
    # so dropping the whole _MERGE_KEY_COLUMNS tuple here would silently strip
    # match_date from every caller's result.
    return merged.drop(columns=["_merge_home_team", "_merge_away_team"])


def _left_merge_stats(
    base: pd.DataFrame, candidate_slice: pd.DataFrame, supplemental_stat_columns: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    """Left-merge `candidate_slice`'s stat columns onto `base` by the merge key, `base`'s own values winning ties.

    Preserves `base`'s original index (`pd.merge` otherwise resets it), so a
    caller merging a row-subset (e.g. the +/-1 day fallback pass) can align
    the result back against the original frame by label.

    Also returns a boolean Series (same index as `base`) marking which rows
    the merge key actually matched a supplemental row for -- via pandas' own
    `indicator=`, not by inspecting the stat columns afterwards (CONFIRMED
    REAL BUG this replaced: a stat column can already be non-null in `base`
    from its own unrelated data, e.g. `tournament`/`neutral_venue`, which
    made every row look "already matched" regardless of whether this merge
    found anything).
    """
    merged = base.merge(
        candidate_slice, on=list(_MERGE_KEY_COLUMNS), how="left", suffixes=("", "_supplemental"), indicator="_merge_found"
    )
    merged.index = base.index
    found = merged["_merge_found"] == "both"
    merged = merged.drop(columns=["_merge_found"])
    for column in supplemental_stat_columns:
        supplemental_column = f"{column}_supplemental"
        if supplemental_column not in merged.columns:
            continue  # no name collision -- the supplemental column already landed under its own name
        if column in merged.columns:
            merged[column] = merged[column].where(merged[column].notna(), merged[supplemental_column])
        else:
            merged[column] = merged[supplemental_column]
        merged = merged.drop(columns=[supplemental_column])
    return merged, found
