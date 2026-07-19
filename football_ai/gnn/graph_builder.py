"""Experimental graph builder for player and coach relations."""

from __future__ import annotations

import pandas as pd

from football_ai.gnn.schemas import GraphBuildConfig, GraphBuildResult


def build_player_relation_graph(
    player_match_history: pd.DataFrame,
    club_relationships: pd.DataFrame | None = None,
    coach_relationships: pd.DataFrame | None = None,
    config: GraphBuildConfig | None = None,
) -> GraphBuildResult:
    """Build a tabular graph payload for later GNN experiments.

    This function currently prepares a leak-safe graph scaffold:
    nodes represent players or coaches, while edges represent relationships
    observed strictly before the target match date. The tensor conversion and
    neural model training will live in separate modules.
    """
    graph_config = config or GraphBuildConfig()
    history = player_match_history.copy()
    required_columns = {"player_id", "player_name", "match_date"}
    missing_columns = required_columns.difference(history.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Player match history is missing required columns: {missing_list}")

    node_columns = [column for column in history.columns if column in {"player_id", "player_name", "team_name", "position", "age", "market_value_eur"}]
    node_frame = (
        history.sort_values("match_date")
        .drop_duplicates(subset=["player_id"], keep="last")[node_columns]
        .reset_index(drop=True)
    )

    edge_frames: list[pd.DataFrame] = []
    if graph_config.include_national_team_edges:
        national_edges = history.copy()
        if "shared_matches" not in national_edges.columns:
            national_edges["shared_matches"] = 0
        national_edges["edge_type"] = "national_team"
        edge_frames.append(national_edges)

    if graph_config.include_club_edges and club_relationships is not None:
        club_edges = club_relationships.copy()
        club_edges["edge_type"] = "club"
        edge_frames.append(club_edges)

    if graph_config.include_coach_node and coach_relationships is not None:
        coach_edges = coach_relationships.copy()
        coach_edges["edge_type"] = "coach"
        edge_frames.append(coach_edges)

    edge_frame = (
        pd.concat(edge_frames, ignore_index=True)
        if edge_frames
        else pd.DataFrame(columns=["source_node_id", "target_node_id", "edge_type"])
    )
    return GraphBuildResult(
        node_frame=node_frame,
        edge_frame=edge_frame,
        graph_metadata={
            "embedding_dimension": graph_config.embedding_dimension,
            "include_club_edges": graph_config.include_club_edges,
            "include_national_team_edges": graph_config.include_national_team_edges,
            "include_coach_node": graph_config.include_coach_node,
        },
    )
