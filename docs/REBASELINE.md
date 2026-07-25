# Rebaseline Log — Model 1 Remediation

Tracks temporal accuracy / log loss after each phase of `8si_remediation_plan.md`,
so the effect of each leakage fix is visible rather than just trusting a
single final number. Evaluated on the same holdout throughout this log
(`>= 2024-01-01`, men's fights only) — per the plan, this holdout is
considered "burned" for model-selection purposes starting Phase 3, at which
point `training/walk_forward.py` becomes the honest evaluation.

**Read this top-to-bottom as a sequence, not as a leaderboard.** A drop in
accuracy after Phases 1–2 means a real leak was removed, not that the model
got worse — see "What NOT to do" in `8si_remediation_plan.md`.

## Phase 0 — safety nets (no feature/pipeline changes)

- Fixed the `corner_flip` signature bug that crashed the trainer before it
  could complete.
- Added `requirements.txt`, `tests/test_smoke.py`, `tests/test_no_leakage.py`.
- No feature logic changed, so this number is the actual as-shipped baseline
  of the (still leaky) pipeline documented in the module docstring's
  "Variant V2" block.

| Metric | Value |
|---|---|
| Temporal accuracy (2024+, men's) | **72.50%** |
| Log loss | not tracked yet (added in Phase 1) |
| Features | 129 |

## Phase 1 — kill the style-stat leak

- `R_SLpM`/`B_SLpM`/etc. (8 stats × 2 corners = 16 raw + 6 diff = 22 columns)
  now come from `training/style_stats.compute_style_stats_asof`, an as-of
  (`merge_asof`, backward) join sourced from `data/ufc_gold_dataset_final.csv`
  — cumulative per-fighter rates using ONLY fights strictly before the fight
  being evaluated.
- **Source changed from the plan's text**: the plan pointed at
  `data/ufc_stats_fights.csv`, but that file has no strike/takedown numeric
  columns at all (only fighter/opponent/date/result/method/round/time).
  `data/ufc_gold_dataset_final.csv` has both fighters' landed/attempted
  counts per fight, which is also what makes `Str_Def`/`TD_Def` computable
  (they need the opponent's attempts against this fighter) — nothing had to
  be dropped from the 8 style-stat columns, unlike the plan's fallback
  language anticipated for a column-poor source.
- Missingness: rows where the fighter has no prior fight in
  `ufc_gold_dataset_final.csv` (their UFC debut, or a name spelling not
  present in that source — ~5.3% of `career_fights_updated.csv` fighter
  names have no match, checked directly) get `R_style_missing`/
  `B_style_missing` = 1, and their 8 style stats are imputed to the
  **training-window** (`date < train_cutoff`), **weight-class** median
  (falling back to the training-window global median for classes with no
  observed values) — replacing the previous blind `fillna(0.0)`.
- Same treatment applied to `got_finished_rate` (previously hardcoded to 0.5
  for any fighter with no prior loss, conflating "truly unknown" with "0
  finish-losses out of some prior losses"): now NaN + `R_gf_missing`/
  `B_gf_missing` indicators, imputed the same way.
- Net: 129 → 133 features (+4 missingness indicators).
- `tests/test_no_leakage.py::test_style_stats_no_leakage` — was `xfail`,
  now genuinely passes (15 sampled fighter/date pairs, truncated-source
  recomputation matches the pipeline's merged value within `1e-6`).
- `opp_quality` leak (Phase 2) is untouched — still `xfail`.

| Metric | Value | Δ vs. Phase 0 |
|---|---|---|
| Temporal accuracy (2024+, men's) | **68.96%** | −3.54 pts |
| Log loss | **0.5961** | (new metric, no Phase 0 comparison) |
| Features | 133 | +4 |

The accuracy drop is expected and is the point of this phase: the previous
style-stat features let 2016 fights see the fighter's 2026 lifetime
averages (including the outcome of that fight and everything after). Per
the plan's ground rules, **do not** try to recover this by reintroducing
snapshot-based style stats.

## Infra — serving isolation + schema tripwire (between Phase 1 and Phase 2)

Both of these landed before Phase 2's retrain, so Phase 2's numbers below
are the first to actually exercise them:

- **Serving artifacts restored to pre-remediation state.** The Phase 0 and
  Phase 1 retrains above had been writing directly to the paths
  `backend/main.py` loads (`model/ufc_model_best.pkl`,
  `model/ufc_model_xgb.pkl`, `model/feature_columns_best.pkl`,
  `model/model_metadata.json`, `data/elo_current.csv`,
  `data/elo_ratings_history.csv`), silently swapping the served model out
  from under the backend. These were restored to their original
  pre-remediation content (72.81% accuracy, 129 features) via `git checkout`
  — the git index still held them staged-but-uncommitted from before any of
  this work started, so no separate backup was needed. **Backend serving is
  now pinned to the original model until Phase 5** unifies the train/serve
  feature pipeline (backend computes its own features from
  `ufc_fighters_final_updated.csv` independently of the trainer, so it isn't
  safe to point it at a Phase 1/2 model yet — see the Phase 1 report for
  detail).
- **Trainer now writes versioned output.** `training/train_model1.py`'s
  `if __name__ == '__main__':` block gained a `--out-dir` flag, defaulting
  to `model/v2/` — never the serving paths. Promoting a new model to serving
  is now a separate, deliberate `--out-dir model` (or manual copy) step, not
  a side effect of running the trainer.
- **`feature_schema_version` tripwire.** `model_metadata.json` now carries a
  `feature_schema_version` int (manually set to `1` on the restored
  pre-remediation metadata; the trainer writes `FEATURE_SCHEMA_VERSION = 2`
  — bumped only when the feature *list* changes, not when a feature's
  formula changes — see the constant's comment in train_model1.py).
  `backend/main.py` asserts `metadata['feature_schema_version'] ==
  EXPECTED_FEATURE_SCHEMA_VERSION` right after loading it, so pointing the
  backend at a schema-incompatible model now fails immediately and loudly
  at import time instead of silently producing skewed predictions.

## Phase 2 — kill the opp_quality leak

- `opp_quality` (mean win rate of a fighter's last ≤5 opponents) now scores
  each opponent by THEIR OWN cumulative win rate as of the target fight's
  date, instead of their full-career (all-time) win rate. Implementation:
  `compute_career_stats()` already computes a shift(1)/pre-fight
  `career_win_rate` per fighter per fight; opp_quality needed the opponent's
  *post*-fight cumulative rate through their last fight strictly before the
  target date (a fight that already happened is not future information),
  computed as `(_cs_won + won) / (cum_fights + 1)` at that row and looked up
  via `np.searchsorted` per opponent (`_win_rate_asof()`).
  - First implementation attempt reused the existing `career_win_rate`
    (pre-fight) column for this lookup and was off by exactly one fight per
    opponent — caught by `tests/test_no_leakage.py::test_opp_quality_no_leakage`
    (8 of 10 sampled cases failed) before being fixed.
- Removed the `all_win_rates` dict and its full-career computation in
  `main()` entirely — `compute_career_stats()` no longer takes it as a
  parameter; opp_quality is now fully self-contained within the function.
- `tests/test_no_leakage.py::test_opp_quality_no_leakage` — was `xfail`,
  now genuinely passes (10 sampled fighter/date pairs, `1e-6` tolerance).
  **No `xfail` markers remain in `tests/test_no_leakage.py`.**
- Trained into `model/v2/` (see infra section above) — does not touch
  serving paths.

| Metric | Value | Δ vs. Phase 1 |
|---|---|---|
| Temporal accuracy (2024+, men's) | **68.75%** | −0.21 pts |
| Log loss | **0.5965** | +0.0004 |
| Features | 133 | 0 (opp_quality's formula changed, not the column list) |

Small further drop, as expected — `opp_quality` was leaking less than the
style-stat snapshot did (it only affected one feature, windowed to ≤5
opponents), but it was still future information. Both leaks identified in
`8si_remediation_plan.md` are now fixed; the trainer's temporal accuracy
(68.75%) reflects an honestly-evaluated Model 1 for the first time in this
log, though the plan itself flags this holdout as compromised for
model-selection purposes starting Phase 3 (`training/walk_forward.py` is
the honest evaluation going forward).

## Phase 3 — honest evaluation

### 3.1 Symmetric inference

`predict_symmetric(model_lr, model_xgb, X, ...)` (train_model1.py) now
provides corner-invariant P(Red wins): `0.5 * (blend(X) + (1 -
blend(corner_swapped(X))))`. Corner-flip augmentation at training time
doesn't make the model perfectly symmetric — the raw blend can (and does)
give a different prediction for the same fight depending on which fighter
is drawn Red vs. Blue; averaging with the corner-swapped pass cancels that
out. This is now used for ALL test-set evaluation (single-holdout, walk-
forward, and calibration fitting), replacing the raw blend used through
Phase 2. **This alone moved single-holdout accuracy from 68.75% (Phase 2,
raw blend) to 69.27% (Phase 3, symmetric)** — not a feature or leakage
change, just a less biased way of reading out the same trained model.

### 3.2 Metrics + baselines

Trainer now reports and persists to `model_metadata.json`: accuracy, log
loss, Brier score, and a 10-bin calibration table (mean predicted vs.
observed win rate, with counts — see `calibration_table()` /
`print_calibration_table()`). Two baselines are computed on the same test
set:
- **Pick higher Elo**: 55.21%.
- **Always pick favorite** (opening `R_odds`/`B_odds`, American format —
  more negative = bigger favorite): 68.63%, at 88.6% odds coverage of the
  test set.

The model (69.27%) beats both, but only barely beats the odds-favorite
baseline (+0.64 pts) — a useful, humbling number that the single-holdout
accuracy alone wouldn't have surfaced. `docs/FEATURE_REFERENCE.md`/model
docstring claims of "beats X%" should be read against these baselines, not
against 50%.

### 3.3 Walk-forward evaluation

New `training/walk_forward.py`: 5 independent folds, train on
`[2015-01-01, N)` / test on year `N`, for `N` in `{2021, 2022, 2023, 2024,
2025}`. Each fold retrains a fresh blend from scratch via the same
`build_dataset()` used by `main()` (necessary since style-stat/
got_finished_rate median imputation is itself training-window-dependent —
see `build_dataset()`'s docstring). No hyperparameter was changed based on
any fold, including the last one. Results also written to
`model/v2/walk_forward_results.json`.

| Fold (test year) | N test | Accuracy | Log loss | Brier |
|---|---|---|---|---|
| 2021 | 398 | 60.55% | 0.6503 | 0.2293 |
| 2022 | 418 | 64.35% | 0.6311 | 0.2206 |
| 2023 | 413 | 65.86% | 0.6252 | 0.2180 |
| 2024 | 424 | 69.10% | 0.5987 | 0.2056 |
| 2025 | 434 | 67.97% | 0.5975 | 0.2057 |
| **POOLED** | **2,087** | **65.64%** | **0.6200** | **0.2156** |

**This — 65.64% pooled accuracy, 0.6200 log loss — is the honest number,
not the 69.27% single-holdout figure above.** The clear upward trend across
folds (60.55% → ~68%) is mostly a training-set-size effect (the 2021 fold
trains on only ~2,300 rows vs. ~4,050 for 2025), not evidence the model
itself improved — expected, and exactly why the plan calls the single
2024+ holdout compromised for model-selection: it's the two most-favorable,
most-tuned-against folds in this pool taken alone.

### 3.4 Probability calibration

`fit_calibrator()` fits a calibrator mapping raw symmetric blended
probability → observed win rate, on a held-out validation window (last 18
months of the training period), using a SEPARATE pair of blend models
trained excluding that window (so the calibrator learns from genuinely
out-of-sample predictions, not the deployed model's memorized training
data). Saved as `model/v2/calibrator.pkl`.

**Deviated from the plan's literal suggestion of `IsotonicRegression`.**
Tried it first as specified; on this holdout it made things WORSE (log
loss 0.5960 → 0.8005, Brier 0.2048 → 0.2054) instead of better. Isolated
the cause empirically (see conversation, not reproduced as a script here):
even applying isotonic to the SAME restricted model used to fit it (ruling
out deployed-vs-calibration-model mismatch as the cause), isotonic still
hurt (0.618 → 0.719 log loss on that restricted model's own test
predictions). Root cause: the calibration validation window is only ~600
rows, and isotonic regression's many-knot step function is known (per
sklearn's own docs) to overfit noise at that sample size. Switched to
`PlattCalibrator` (2-parameter logistic/sigmoid fit) — the plan explicitly
allows "IsotonicRegression, or CalibratedClassifierCV equivalent," and
Platt/sigmoid is the standard CalibratedClassifierCV alternative for
limited calibration data.

Even with Platt scaling, calibration is a wash on this holdout — **slightly
worse**, not better: log loss 0.5960 (raw) → 0.6058 (calibrated), Brier
0.2048 → 0.2092. The pre-calibration calibration table (Phase 3.2) already
shows the raw symmetric blend is reasonably well-calibrated (predicted vs.
observed track within ~5-10 points across most bins), so there's little
calibration error left for a calibrator fit on ~600 rows to correct without
just adding its own estimation noise. **The `calibrator.pkl` artifact is
saved per the plan's acceptance criteria, but is not applied anywhere in
this pass** (not to the trainer's own reported "official" metrics above,
which use raw symmetric probabilities; not in the backend, which still
serves the pre-remediation model entirely — see the infra section). Worth
revisiting once more calibration data accumulates (e.g. fit on pooled
walk-forward validation folds instead of a single 600-row window) or if a
future feature/model change measurably worsens raw calibration.

## Phase 4 — betting-layer hardening (backend/main.py)

No trainer/accuracy changes in this phase — it's entirely `backend/main.py`
(live betting decisions) and `data/value_bet_log.csv` (Model 2B's training
data). **Two live `uvicorn` processes are running `backend/main.py` right
now** (ports 8000/8002) — code was edited, but those processes were left
running; changes take effect on the user's own next restart, not
automatically.

### 4.1 + 4.2 — market shrinkage + unified Kelly

`market_shrink()`, `kelly_fraction()`/`kelly_stake()` added to
`backend/main.py`, applied to both live betting-decision paths
(`model2a_predict`, `bet_recommendation`). Full reasoning in
`docs/DECISIONS.md` ("8SI Phase 4"); summary:

- `SHRINKAGE_W = 0.30` is a **documented default, not a grid-searched
  value**. The plan calls for fitting `w` on walk-forward validation, which
  `training/walk_forward.py` can do properly for the NEW model — but
  backend still serves the OLD (pre-remediation) model, whose own feature
  lookups (`get_career_stats()` etc.) always compute "stats as of right
  now" with no way to score a historical date without leaking each
  fighter's future career into the "historical" prediction. Building a
  point-in-time-correct backtest specifically for a model Phase 5 is going
  to replace anyway was judged not worth it. 0.30 is the midpoint of the
  plan's own stated 0.2–0.4 range.
- `GAP_THRESHOLD = 0.03`, derived algebraically (`gap_final = w *
  gap_raw`, exactly) from the original 10% trigger, not re-fit — replaces
  two previously-inconsistent thresholds (10% in one Kelly path, an
  undocumented 5% in the other) with one.
- `KELLY_FRACTION = 1/3` standardized across both paths (previously 1/3 in
  one, an inline unnamed 1/4 in the other).
- **`model2b_predict`'s own gap/probability computation was deliberately
  left untouched** — it feeds a separately-trained RandomForest
  (`model/ufc_model2b.pkl`) as raw model features on the ORIGINAL unshrunk
  scale; shrinking it would have fed that frozen model a distribution it
  was never trained on, silently corrupting its predictions. This was
  caught by reading through to a third, easy-to-miss consumer of the same
  gap-computation pattern before editing anything — see `docs/DECISIONS.md`
  for the full explanation and the `_gap_zone(gap_size / SHRINKAGE_W)`
  trick used to keep `bet_recommendation`'s human-readable zone label
  consistent without touching the shared (Model-2B-feeding) `_gap_zone()`
  function.
- Sanity-checked end-to-end (not unit tests — none exist for `backend/`
  yet): called `model2a_predict()` and `bet_recommendation()` directly with
  realistic inputs and hand-verified the shrinkage/gap/Kelly arithmetic
  matched the code, and confirmed `model2b_predict()`'s output is on the
  same raw scale as before (untouched).

### 4.3 — CLV tracking

- `data/value_bet_log.csv` extended with `odds_taken`, `novig_prob_taken`,
  `novig_prob_close`, `clv_pct` — **purely additive**. This file turned out
  to be Model 2B's actual training data (`split` column = train/test), not
  a live P&L log — none of the existing columns (`gap`, `m1_prob`,
  `m2a_prob`, `pick_novig`, `closing_odds`, ...) were touched, since that
  model was fit on their exact values.
- `training/backfill_clv.py`: backfills `odds_taken`/`novig_prob_taken`
  from `data/odds_snapshots.json`'s earliest snapshot per matchup where one
  exists; falls back to the existing `closing_odds`/`pick_novig` (i.e.
  `clv_pct = 0`, not fabricated) otherwise. **Current real coverage: 0 of
  3,007 rows** — the 13 matchups in `odds_snapshots.json` are all one
  upcoming/recent card not yet present in the historical log. Script is
  correct and will pick up real matches automatically as
  `odds_snapshots.json` and `value_bet_log.csv` both accumulate more
  history — verified the zero-match result directly (checked matchup-pair
  overlap independently of the script), not assumed.
- `training/clv_report.py`: prints per-archetype (CONFIRM_DOG/CONFIRM_FAV,
  derived from existing `m1_m2a_agree`/`vegas_agree`/`gap_direction`
  columns) mean CLV and % beating close, plus the plan-mandated caveat that
  ROI/win-rate at this sample size (CONFIRM_DOG n=314, CONFIRM_FAV n=1,410)
  isn't evidence of edge. Runs cleanly; current output is honest
  all-zero-CLV given the coverage gap above.

### 4.4 — line movement feature

**Skipped**, per the plan's own stated condition: `odds_snapshots.json`
covers 13 matchups from one card, nowhere near the "~60% of M2A training
rows" coverage threshold the plan sets for adding this feature.

## Phase 5 — unify train/serve feature pipeline

No trainer accuracy changes — full reasoning and design in
`docs/DECISIONS.md` ("8SI Phase 5"). Two independent pieces:

### Part A — QA-stats / got_finished_rate bug fix (live, shipped)

`backend/main.py`'s `predict()`/`predict_method()` had never computed real
QA stats or `got_finished_rate` for the live model — 14 of its 129 trained
features got placeholder/wrong values on every prediction, independent of
any leakage issue. Fixed by calling `train_model1.compute_qa_stats()` /
the newly-extracted `compute_got_finished_rate()` directly, against a
synthetic "as of today" row per fighter. Old model, old architecture —
just correct inputs now.

**Independent discovery, not fixed:** `compute_qa_stats()`'s own formula
has `qa_SLpM == qa_win_rate` and `qa_SApM == 1 - qa_win_rate` exactly
(verified numerically) — it was never actually fed striking-volume data,
just the win/loss flag. Predates this entire remediation project, present
in both the original and the Phase 1–3 model's training data. Flagged for
a future pass; fixing it means retraining, out of scope here.

### Part B — shared feature module + parity test (built, not wired into serving)

Per explicit user choice: proved the point-in-time pipeline can be unified,
did NOT cut backend over to serve the new model. `features/constants.py`
(dependency-free shared constants) + `features/build.py`
(`get_fighter_state_asof()`, built on the trainer's own `compute_*()`
functions) + `tests/test_train_serve_parity.py` (10 historical fights,
matches `build_dataset()` within `1e-6`).

Getting the parity test green surfaced 3 real issues along the way (one my
own bug, two pre-existing data-quality gaps in the source CSVs) — see
`docs/DECISIONS.md` for detail on each. None required changing any
trained model or leakage-relevant logic.

`train_model1.py`/`backend/main.py`'s duplicate `WC_ORDER`/`WOMENS_CLASSES`/
layoff-bucket definitions were consolidated into `features/constants.py` —
confirmed behavior-identical via full test suite + a retrain producing
byte-identical accuracy/log-loss/Brier numbers to before the refactor
(69.27% / 0.5960 / 0.2048, unchanged).

**Backend still serves the original pre-remediation model** —
`get_fighter_state_asof()` exists, is tested, and is not called from any
live endpoint. The cutover (swap `backend/main.py` to `model/v2/`, wire the
new pipeline into `/predict`/`bet_recommendation`, bump
`EXPECTED_FEATURE_SCHEMA_VERSION` to 2) remains a distinct future decision.

## Phase 6 — repo hygiene

No trainer/accuracy changes. `.gitignore` added (`__pycache__/`, `*.pyc`,
`.DS_Store`, `catboost_info/`, `node_modules/`, `*.pkl` — deliberately
narrower than the plan's literal "large generated CSVs," see
`docs/DECISIONS.md` for why a blanket `data/*.csv` ignore would have been
a footgun, not a hygiene win). `git rm -r --cached` on every currently-
tracked `__pycache__/`/`.DS_Store`/`catboost_info/` path (working tree
left intact, verified). README.md's pre-existing, unrelated merge conflict
(flagged back in Phase 0, left unresolved since) finally resolved —
blocked the plan's own README-update requirement, so had to be dealt with
here. `docs/DATA_SOURCES.md` got a provenance/regeneration line per file,
including an honest "no producer script found" for `ufc_gold_dataset_final.csv`
rather than a guessed origin.

## Phase 7 — Elo upgrades

Promoted method-weighted K (×1.25 KO/TKO/Sub, ×0.75 split decision) + 25%
layoff regression (>365 days inactive) to `training/train_model1.py`
defaults (`ELO_METHOD_MULTIPLIERS`, `ELO_LAYOFF_REGRESSION`), gated on
walk-forward log loss per the plan's own rule. K-sensitivity grid
{24,32,40,48} confirmed K=48 already optimal — no change there. Full
methodology, the winner-combining bug caught along the way, and the
cross-module Elo-consistency bug `tests/test_train_serve_parity.py` caught
(a real fighter's post-layoff `elo_before` diverging between
`build_dataset()` and Phase 5's `features/build.py`) are in
`docs/DECISIONS.md` ("8SI Phase 7").

| Metric | Before Phase 7 | After Phase 7 |
|---|---|---|
| Single-holdout accuracy | 69.27% | 68.96% |
| Single-holdout log loss | 0.5960 | 0.5932 |
| Single-holdout Brier | 0.2048 | 0.2035 |
| **Pooled walk-forward accuracy** | **65.64%** | **66.08%** |
| **Pooled walk-forward log loss** | **0.6200** | **0.6152** |
| **Pooled walk-forward Brier** | **0.2156** | **0.2134** |
| Higher-Elo baseline (single-holdout) | 55.21% | 59.69% |

The single-holdout accuracy dip (69.27% → 68.96%) is noise, not a
regression — pooled walk-forward (the metric this whole log has been
building toward treating as the honest one) improved on every axis.
Retrained into `model/v2/`; serving paths untouched, same as every prior
phase.

## Next

All 7 phases of `8si_remediation_plan.md` have now been addressed (Phase
5's full live cutover deliberately deferred, per the plan's own acceptance
criteria being "parity test passes," not "cut over"; Phase 7's Glicko-2
stretch goal skipped). Open items for a future pass, in rough priority
order: (1) the `qa_SLpM`/`qa_SApM` formula bug found in Phase 5 Part A,
(2) the actual serving cutover decision, (3) the git-lfs migration path
Phase 6 marked optional and didn't do, (4) Glicko-2 if there's appetite
for further Elo-side gains.
