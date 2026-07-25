# 8SI UFC Predictor

ML-powered UFC fight prediction with a FastAPI backend and React frontend.

## Folder Structure

```
ufc-predictor/
├── backend/          FastAPI server (main.py, port 8000/8002)
├── frontend/         React app (port 3000)
├── data/             All CSV/JSON data files
├── model/            Trained model artifacts (.pkl, .json)
│   └── v2/           Phase 1-3 remediated model (not yet cut over to serving — see docs/DECISIONS.md)
├── features/          Shared, point-in-time-correct feature-computation pieces
│   ├── constants.py  WC_ORDER, layoff buckets, small formulas
│   └── build.py      get_fighter_state_asof() — as-of-date fighter feature lookup
├── training/         Production trainer scripts
│   ├── train_model1.py
│   ├── style_stats.py
│   ├── walk_forward.py
│   ├── backfill_clv.py
│   └── clv_report.py
├── tests/            pytest suite (smoke, leakage, train/serve parity)
├── docs/             Architecture documentation
│   ├── DATA_SOURCES.md
│   ├── FEATURE_REFERENCE.md
│   ├── DECISIONS.md
│   └── REBASELINE.md
└── experiments/      Experiment logs (not used in production)
    ├── archive/      Experiment scripts
    └── results/      Experiment output logs
```

## Models

### Model 1 — currently serving: 70% LR + 30% XGB blend
- **129 features**: fight record, career stats (shift-1), QA stats, Elo, style stats
- **Temporal accuracy**: 72.81% (2024+ men's holdout) — see the caveat below
- **Artifacts**: `model/ufc_model_best.pkl`, `model/ufc_model_xgb.pkl`, `model/feature_columns_best.pkl`
- **Metadata**: `model/model_metadata.json`

### Model 1 v2 — remediated, not yet serving: 133 features
A leakage-fix and honest-evaluation pass (see `docs/REBASELINE.md` and
`docs/DECISIONS.md` for the full account — this is worth reading before
trusting any single accuracy number in this repo) found and fixed two real
temporal leaks in the currently-serving model above (a style-stat snapshot
that let historical fights see a fighter's future averages, and an
opponent-quality feature that used opponents' full-career win rate instead
of their win rate as of the fight date), added symmetric inference and
proper walk-forward validation, and a market-shrinkage + unified-Kelly pass
on the betting layer.

- **133 features** (129 + 4 missingness indicators), artifacts in `model/v2/`
- **72.81% single-holdout accuracy is NOT the honest number** — pooled
  walk-forward across 5 independent test years (2021-2025) is **66.08%**
  (`training/walk_forward.py`). The single holdout has been reused across
  many experiment sprints and is a biased target for model selection.
- Elo now uses method-weighted K + layoff regression (8SI Phase 7 — see
  `docs/DECISIONS.md`), promoted after measurably improving pooled
  walk-forward log loss (0.6200 → 0.6152).
- **Not yet cut over to serving.** `backend/main.py` still computes
  features with its own separate pipeline and loads the original 129-feature
  model — a tested, parity-checked replacement (`features/build.py`,
  `tests/test_train_serve_parity.py`) exists but wiring it into live
  serving is a distinct, not-yet-made decision.

### Model 2 — Odds-aware LR
- **23 features**: Model 1 probability + opening odds features
- **Purpose**: Value-bet detection — triggers on a market-shrunk probability
  gap (see `docs/DECISIONS.md` "8SI Phase 4" — the trigger threshold and
  Kelly fraction were re-derived, not the original untouched values)
- **Kelly sizing**: 1/3 Kelly fraction, shared `kelly_stake()`/`kelly_fraction()`
  implementation (`backend/main.py`) — previously two divergent implementations
- **Artifact**: `model/ufc_model2_best.pkl`

## Retrain Model 1

```bash
# From project root — writes to model/v2/ by default, NOT the serving
# paths (model/ufc_model_best.pkl etc.) — see --out-dir below.
python training/train_model1.py

# Promote a retrain to serving (a deliberate act, not automatic):
python training/train_model1.py --out-dir model
```

Expected output: temporal accuracy in the low-to-high 60s% (post-leakage-fix,
honest number — see the Models section above for why this is lower than
the older 129-feature model's reported 72.81%), all 133 features present,
7 files saved (6 + `calibrator.pkl`).

## Walk-forward evaluation

```bash
python training/walk_forward.py
```

Trains and evaluates 5 independent folds (test years 2021-2025). This is
the honest accuracy number, not the single 2024+ holdout `train_model1.py`
reports — see `docs/REBASELINE.md`.

## Run tests

```bash
pip install -r requirements.txt
pytest tests/
```

Covers: trainer smoke test, temporal-leakage regression tests (career
stats, QA stats, Elo, style stats, opp_quality — all checked against
truncated-source recomputation), and train/serve feature parity.

## Run Locally

```bash
# Backend
cd backend && uvicorn main:app --reload --port 8000

# Frontend
cd frontend && npm start
```

## Key Data Files

| File | Purpose |
|---|---|
| `data/ufc-master.csv` | Primary fight history (7,183 fights) |
| `data/career_fights_updated.csv` | Per-fighter fight log for career stat computation |
| `data/ufc_gold_dataset_final.csv` | Per-fight strike/TD/sub totals (both corners) — source for as-of style stats (`training/style_stats.py`) |
| `data/ufc_fighters_final_updated.csv` | Fighter style-stat snapshot — still used by `backend/main.py` and the women's trainer; superseded for Model 1 v2 by the as-of computation above |
| `data/elo_current.csv`, `data/elo_ratings_history.csv` | Elo ratings (generated by the trainer) |
| `data/value_bet_log.csv` | Model 2B's training data (not a live bet log — see `docs/DECISIONS.md` "8SI Phase 4") |
| `data/upcoming.csv` | Upcoming card data |

See `docs/DATA_SOURCES.md` for column details, gotchas, and how to obtain
or regenerate each file.

## Data is not fully tracked in git

`__pycache__/`, `.DS_Store`, and `catboost_info/` (build/experiment
artifacts) are gitignored and untracked. Existing `data/`/`model/` files
already in the repo remain tracked for now; `.gitignore` prevents *new*
large CSVs and `.pkl` files from being added going forward. See
`docs/DATA_SOURCES.md` for regeneration instructions per file.
