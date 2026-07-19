"""Schemas for experimental GNN graph construction."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(slots=True)
class GraphBuildConfig:
    """Configuration for building player-relation graphs."""

    min_shared_matches: int = 1
    min_shared_minutes: int = 0
    include_club_edges: bool = True
    include_national_team_edges: bool = True
    include_coach_node: bool = True
    embedding_dimension: int = 32


@dataclass(slots=True)
class GraphBuildResult:
    """Tabular graph payload before GNN tensor conversion."""

    node_frame: pd.DataFrame
    edge_frame: pd.DataFrame
    graph_metadata: dict[str, object] = field(default_factory=dict)
