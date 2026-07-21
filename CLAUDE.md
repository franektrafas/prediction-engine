# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`football_ai` — a modular platform for predicting international football matches (World Cup, Euro,
Nations League, qualifiers, Copa América, friendlies), built to grow into a generic
"Universal Football Prediction Engine" that can later cover club football without renaming
core entities. Target architecture: automated data collection → feature engineering → ML
(LightGBM/CatBoost/XGBoost + stacking ensemble) → GNN (experimental, optional) → FastAPI →
PostgreSQL → commercial B2B prediction API. Python 3.12, hybrid workflow: light dev/API work
locally, heavier feature engineering/tuning/training in Google Colab (`football_ai/notebooks/00_colab_bootstrap.ipynb`).

Naming is intentionally generic (`Match`, `Team`, `Competition`, `competition_type` = `international`/`club`)
so club football can be added later without a schema/class rename.

## Commands

Setup (local, lightweight — code/API/tests only, no training):
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-local.txt
```

Full dev setup (training, tuning, notebooks, tests):
```bash
pip install -r requirements.txt   # -e .[dev]
cp .env.example .env
createdb football_ai
python -m football_ai.database.session   # creates tables from database/models.py via SQLAlchemy metadata
```

Data collection (each source is its own runnable module):
```bash
python -m football_ai.scrapers.sources.statsbomb
python -m football_ai.scrapers.sources.football_data
python -m football_ai.scrapers.sources.elo
python -m football_ai.scrapers.sources.open_meteo
python -m football_ai.scrapers.sources.fbref
python -m football_ai.scrapers.sources.football_api --date 2026-07-10 --next-count 10   # needs FOOTBALL_API_KEY
```

Training / API:
```bash
python -m football_ai.training.orchestrator     # build features, tune, train, evaluate, register all targets
uvicorn football_ai.api.main:app --reload       # POST /predict, GET /health
```

Tests:
```bash
python -m unittest discover football_ai/tests   # requires football_ai/tests/**/__init__.py — do not delete them
pytest football_ai/tests                        # equivalent, and supports single-test selection:
pytest football_ai/tests/unit/test_feature_builder.py::FeatureBuilderTestCase::test_build_features_generates_large_feature_space_and_targets
```
Some tests `skipUnless` on optional heavy deps (joblib/sklearn/fastapi/pydantic/sqlalchemy) — a
"skipped" result there means the dependency isn't installed, not a real pass.

Lint: `ruff check .` (config in `pyproject.toml`, line-length 88, py312 target, pydocstyle google convention).

## Architecture

**Feature pipeline (`features/builder.py`) — read this before touching features.** `build_features()`
takes a wide `matches` DataFrame (one row per match, `home_*`/`away_*` columns) and:
1. `build_team_match_frame` reshapes it into a long, one-row-per-team-per-match frame.
2. `_add_rolling_features` / `_add_recency_features` / `_add_h2h_features` compute historical,
   `.shift(1)`-based rolling/recency/head-to-head aggregates on that long frame — this is where
   "leak-safe" history features are built.
3. `_wide_merge` merges the engineered long frame back onto the original wide `matches` frame,
   prefixed `home_`/`away_`.
4. `_add_relative_features` builds home-minus-away diffs/ratios from the merged frame.

**Critical invariant:** `matches` is always the source of truth for any column it already defines.
`_wide_merge` explicitly drops engineered columns that would collide with an existing `matches`
column (`_drop_columns_shadowing_source`). Without that guard, `pandas.merge` silently renames
both copies to `*_x`/`*_y` on a name collision — which both breaks every downstream exact-name
lookup (targets, categorical columns, relative-feature diffs) *and* leaks the current match's own
post-match stats (xG, shots, possession, ...) into the feature set. If you add a new per-team raw
stat anywhere in this pipeline, make sure its name either only exists in the engineered frame or
is deliberately excluded — don't reintroduce a same-name pass-through of a raw `matches` column.
`build_features`'s `excluded_columns` also hard-excludes the full raw `team_statistics` stat family
(`RAW_MATCH_OUTCOME_FIELDS`) so a richer training frame can't leak fouls/offsides/saves/passes/etc.
as features even though they aren't explicit prediction targets.

**No random splits, anywhere — this is a hard project rule, not a preference.** All chronological
splitting lives in `training/splitters.py`: `chronological_train_validation_test_split` (single
70/15/15 cut) and `rolling_positional_splits`/`rolling_time_series_splits` (rolling CV folds,
position-based vs. date-based). `training/tuning.py`'s Optuna objective and
`models/stacking_ensemble.py`'s `StackingClassifier`/`StackingRegressor` both build their `cv=`
from `rolling_positional_splits` instead of scikit-learn's default (Stratified)KFold — the sklearn
default is a random split and will quietly reintroduce time-series leakage if it ever comes back
into either of those two files. `rolling_positional_splits` assumes the frame it's given is already
sorted chronologically (earliest row first); callers are responsible for that.

**Training pipeline (`training/orchestrator.py`).** `train_target()` builds one target at a time
(`TARGET_TASK_MAPPING`): tune each of RandomForest/LightGBM/CatBoost/XGBoost with Optuna on
train+validation (rolling CV), fit on train, score on validation, refit on train+validation, score
on test, save via `models/base.py`'s joblib artifact format, register in `ModelRegistry`. Then a
`StackingEnsembleModel` is built from the top base models and goes through the same
validation/test scoring. The best model per target (by F1 for classification / RMSE for
regression) is registered again under the alias `"selected_model"`.

**Prediction pipeline (`predictions/service.py` + `predictions/inference_feature_builder.py`).**
Multi-stage: match_outcome (1X2) → goals/xG → corners/cards/shots/possession → BTTS/Over-Under,
each an independently-registered target that falls back to Elo-based heuristics
(`_heuristic_prediction`) when no trained model is registered yet, so `/predict` always returns a
response even before any training has happened. `inference_feature_builder.py` re-runs
`features.builder.build_features` on `historical_matches + the single upcoming match row` and
slices out just that match's feature row — inference and training must go through the same
feature code path for consistency.

**Model registry (`models/registry.py`) / DB (`database/models.py`, `database/sql/schema.sql`).**
Tables are created via SQLAlchemy `Base.metadata.create_all()` (`database/session.py`), not by
running `schema.sql` directly — keep the two in sync manually when changing the schema.
`schema.sql` has no migration mechanism; there's no Alembic in this project yet.

**GNN (`gnn/`) is an intentional, early scaffold**, not wired into training/inference. Its
docstrings describe leak-safety goals ("edges built only from history strictly before the target
match") that the current code does not yet enforce — don't trust the docstring over the code here.
