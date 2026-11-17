"""Diagnostyka (nie zmiana kodu silnika): sprawdza, czy podmiana home_*_venue
na away-split tej samej drużyny -- na meczu neutralnym -- zmniejsza swing_neutral
na JUŻ WYTRENOWANYM modelu match_outcome, bez żadnego retreningu.

Kontekst: features/builder.py's _add_rolling_features liczy dwa rolling
feature'y per zespół -- ogólny i rozdzielony na venue (`_venue`, przez
groupby(["team_name","is_home"])). is_home jest czysto pozycyjne (kolumna
home_team_name zawsze dostaje is_home=1), więc na neutralnym gruncie zespół
wpisany jako "home" i tak dostaje swój prawdziwy historyczny split "u siebie".

Ten skrypt NIE zmienia features/builder.py. Zamiast tego buduje oba warianty
(A1: Polska "home", A2: Niemcy "home", oba na neutralnym gruncie), i dla A1
podmienia home_*_venue Polski na jej WŁASNY away split -- wzięty wprost z
frame'u A2, gdzie Polska jest już poprawnie away_team_name (więc away_*_venue
tam JEST jej prawdziwym away splitem, nic nie trzeba przeliczać). Analogicznie
dla A2/Niemiec. Efekt: widzimy, czy ten swap w ogóle rusza predykcję
już-wytrenowanego modelu, i w którą stronę -- zanim zdecydujemy, czy retrening
jest tego wart.

Uruchom z korzenia repo, w środowisku z requirements-local.txt/requirements.txt
zainstalowanym (potrzebne: joblib, scikit-learn, lightgbm, catboost, xgboost,
pydantic, pandas):

    python -m neutral_swap_check
    # albo
    python neutral_swap_check.py
"""

from __future__ import annotations

import pandas as pd

from football_ai.predictions.service import PredictionService
from football_ai.predictions.schemas import MatchContextPayload
from football_ai.features.schemas import FeatureBuilderConfig


def _payload(
    home: str, away: str, neutral: bool, country: str, home_elo: float, away_elo: float
) -> MatchContextPayload:
    return MatchContextPayload(
        match_date=pd.Timestamp("2026-11-20", tz="UTC"),
        home_team_name=home,
        away_team_name=away,
        tournament="FIFA World Cup",
        neutral_venue=neutral,
        country=country,
        home_elo=home_elo,
        away_elo=away_elo,
    )


def main() -> None:
    service = PredictionService.create()
    cfg = FeatureBuilderConfig()

    POLAND_ELO, GERMANY_ELO = 1700.0, 1750.0

    p_a1 = _payload("Poland", "Germany", True, "Qatar", POLAND_ELO, GERMANY_ELO)
    p_a2 = _payload("Germany", "Poland", True, "Qatar", GERMANY_ELO, POLAND_ELO)

    frame_a1 = service._build_feature_frame(p_a1)
    frame_a2 = service._build_feature_frame(p_a2)

    # Dokładne nazwy kolumn _venue z FeatureBuilderConfig (nie zgadywanie regexem --
    # inne kolumny, np. home_neutral_venue, też kończą się na "_venue" i nie są
    # tym samym mechanizmem).
    venue_suffixes = [
        f"{metric}_{agg}_{window}_venue"
        for metric in cfg.rolling_metrics
        for window in cfg.rolling_windows
        for agg in cfg.rolling_aggregations
    ]

    mod_a1 = frame_a1.copy()
    mod_a2 = frame_a2.copy()
    swapped = 0
    for suffix in venue_suffixes:
        home_col, away_col = f"home_{suffix}", f"away_{suffix}"
        if home_col in mod_a1.columns and away_col in frame_a2.columns:
            # a1: home_team = Polska. Jej prawdziwy away-split = away_col z frame_a2
            # (tam Polska jest away_team_name, wiec away_col to jej realny away split).
            mod_a1.loc[:, home_col] = frame_a2[away_col].values
            swapped += 1
        if home_col in mod_a2.columns and away_col in frame_a1.columns:
            mod_a2.loc[:, home_col] = frame_a1[away_col].values

    print(f"Podmieniono {swapped}/{len(venue_suffixes)} kolumn home_*_venue na frame.\n")

    registry_entry = service.registry.get_latest("match_outcome")
    if registry_entry is None:
        raise SystemExit("Brak zarejestrowanego modelu match_outcome -- nie ma czego testowac.")
    print("model match_outcome:", registry_entry.version)

    def outcome_probs(frame: pd.DataFrame) -> dict[str, float]:
        return service._predict_classification(
            artifact_path=registry_entry.artifact_path,
            frame=frame,
            labels=["away_win", "draw", "home_win"],
        )

    baseline_a1 = outcome_probs(frame_a1)
    baseline_a2 = outcome_probs(frame_a2)
    modified_a1 = outcome_probs(mod_a1)
    modified_a2 = outcome_probs(mod_a2)

    print("\n=== BASELINE (obecny bug, bez zmian) ===")
    print("A1 (Polska 'home' na arkuszu):", baseline_a1)
    print("A2 (Niemcy 'home' na arkuszu):", baseline_a2)
    swing_baseline = abs(baseline_a1["home_win"] - baseline_a2["away_win"])
    print(f"swing_neutral (baseline): {swing_baseline:.4f}")

    print("\n=== PO SWAPIE (home_*_venue -> away-split tej samej druzyny) ===")
    print("A1 zmodyfikowany:", modified_a1)
    print("A2 zmodyfikowany:", modified_a2)
    swing_modified = abs(modified_a1["home_win"] - modified_a2["away_win"])
    print(f"swing_neutral (po swapie): {swing_modified:.4f}")

    print(f"\nZmiana: {swing_baseline:.4f} -> {swing_modified:.4f} (delta {swing_modified - swing_baseline:+.4f})")
    print(
        "Interpretacja: jesli delta jest wyraznie ujemna (swing sie skurczyl),"
        " swap cos daje jako prowizorka -- ale to wciaz nie to samo co model"
        " NAUCZONY na poprawionych cechach (patrz nasza wczesniejsza rozmowa"
        " o train/serve skew). Jesli delta jest bliska zeru albo dodatnia,"
        " swap sam w sobie nie pomaga i retrening jest jedyna droga."
    )


if __name__ == "__main__":
    main()
