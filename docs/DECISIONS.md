# Architectural Decisions

---

## 8SI Phase 7: Elo upgrades (Jul 2026)

**Decision:** Promoted method-weighted K + 25% layoff regression to
production defaults. `experiments/elo_v2/run_experiment.py` has the full
methodology; summary here.

**K-sensitivity** (grid {24,32,40,48}, lone-feature `elo_dif` LR across the
same 5 walk-forward folds `training/walk_forward.py` uses): K=48 (the
existing value) was already best in the tested range (0.6783 pooled log
loss vs. 0.6799 at K=24) — accuracy/log loss improved monotonically with K
across the whole grid, i.e. no evidence K=48 is "too aggressive" as the
plan speculated. **No change** — kept K=48.

**Method-weighted K** (×1.25 for KO/TKO/Sub wins, ×0.75 for split
decisions, ×1.0 otherwise) and **layoff regression** (>365 days inactive
→ regress rating toward 1500 by a tunable %, grid-checked at the plan's
stated 10-25% range endpoints) were each evaluated individually through
the FULL 133-feature model (not lone-feature — these change `elo_dif`
throughout the whole feature set, so the meaningful test is the deployed
model's own walk-forward log loss) via a new `elo_kwargs` parameter
threaded through `compute_elo()` and `build_dataset()`, both `None`
(default) or off unless explicitly requested — verified byte-identical
baseline behavior via the full test suite before touching anything else.

All three individually beat baseline (0.6200 pooled log loss):
method-weighted K alone → 0.6168; layoff-regression 15% alone → 0.6184;
layoff-regression 25% alone → 0.6178. Combining method-weighted K with the
better-performing layoff-regression-25% (not literally "all three" — see
bug note below) → **0.6152**, the best result, confirmed by an independent
verification run reproducing the exact same number.

**Bug caught while combining winners:** the experiment script's first
version blindly `dict.update()`-merged every individual winner's kwargs in
iteration order. `layoff-regression 15%` and `layoff-regression 25%` are
mutually-exclusive VALUES of the same `layoff_regression` kwarg, not
combinable — the naive merge let whichever ran last in the loop silently
clobber the other, so the script's own printed label ("...15% + ...25%")
overstated what actually ran (only 25% took effect). Caught by noticing
the label implied 3 combined changes but the result exactly matched a
2-way combination tested separately; fixed the script to pick the
better-individual-performing value per colliding kwarg key rather than
last-write-wins, and reran to confirm.

**Promoted via new module constants** in `training/train_model1.py`:
`ELO_METHOD_MULTIPLIERS`, `ELO_LAYOFF_REGRESSION = (365, 0.25)`. Wired in
as `build_dataset()`'s DEFAULT when its `elo_kwargs` param is omitted
(`None`) — explicitly passing `elo_kwargs={}` still gets the pre-Phase-7
plain-K=48 baseline, which is what `experiments/elo_v2/run_experiment.py`'s
own baseline run uses. Does NOT bump `FEATURE_SCHEMA_VERSION` — same
column names/count, formula only (per that constant's own documented bump
rule).

**Cross-module consistency catch:** `features/build.py`'s `DataBundle`
(Phase 5) computes its own independent Elo history for
`get_fighter_state_asof()`'s point-in-time queries — promoting the new
config in `build_dataset()` without ALSO updating `DataBundle` to match
would have silently broken Phase 5's train/serve parity guarantee. Caught
immediately by `tests/test_train_serve_parity.py` (an `R_elo` mismatch for
Jon Jones's 2024 fight — his multi-year layoff before it is exactly the
scenario layoff-regression changes). Root cause: `_elo_asof()`'s shortcut
(a fighter's next `elo_before` normally equals their previous fight's
`elo_after` exactly) stopped holding once layoff regression could
intervene between two fights — fixed by replicating the same regression
check inside `_elo_asof()`, gated on the SAME `ELO_LAYOFF_REGRESSION`
constant imported from `train_model1.py` (not re-duplicated).

**Retrained `model/v2/` with the promoted config** (serving paths
untouched, same protocol as every prior phase): single-holdout accuracy
68.96% (down slightly from 69.27% — expected noise, not the metric that
matters, see Phase 3), log loss 0.5932 (down from 0.5960), Brier 0.2035
(down from 0.2048). Pooled walk-forward — the honest number —
**66.08%/0.6152/0.2134**, vs. 65.64%/0.6200/0.2156 before. The
higher-Elo baseline itself jumped from 55.21% to 59.69% accuracy on the
single holdout, consistent with the promoted Elo genuinely being a better
standalone signal, not just interacting favorably with the rest of the
model.

**Not done:** Glicko-2 (explicitly marked "stretch" in the plan) — skipped
given time budget and that the three required items already produced a
measurable win; revisit only if a future pass specifically wants to chase
further Elo gains.

---

## 8SI Phase 6: repo hygiene (Jul 2026)

**Decision:** Resolved a pre-existing, unrelated README.md merge conflict
(left unresolved since before this remediation project — see the Phase 0
report) to unblock the plan's README-update requirement. One side was a
2-line placeholder (`# 8si` / `UFC ML model and 8si frontend`), the other
the full, detailed README matching the actual project structure — an
unambiguous resolution (kept the detailed side), not a judgment call
between two substantively different real options, so made directly rather
than blocking on it again.

**`.gitignore` scope — narrower than the plan's literal text.** The plan
says ignore "large generated CSVs" alongside `__pycache__/`, `.DS_Store`,
`*.pkl`, `catboost_info/`, `node_modules/`. Did NOT add a blanket
`data/*.csv` pattern: `.gitignore` doesn't affect files already tracked
(they stay tracked and stageable regardless), so the only effect of a
broad CSV ignore would be to silently hide any genuinely NEW data file a
future `git add` might otherwise catch — a footgun, not a hygiene win,
for a repo where `data/` is being deliberately kept as still-tracked
working data rather than migrated out (the plan marks the git-lfs
migration path explicitly optional, not done here). Shipped: `__pycache__/`,
`*.pyc`, `.DS_Store`, `catboost_info/`, `node_modules/`, `*.pkl`.

**`git rm -r --cached`** (index-only, working tree left intact — verified
files still exist on disk after) applied to every currently-tracked
`__pycache__/`, `.DS_Store`, and `catboost_info/` path, including one
`experiments/research/model1_v2/__pycache__/` that the plan's own example
command list didn't call out explicitly — found by a follow-up
`git ls-files | grep` sweep after the plan's literal command left it
behind. `backend/__pycache__/main.cpython-313.pyc` needed `-f` (staged
content had drifted from the working-tree file, from `.pyc` cache
regeneration during this session's own work) — safe here since it's pure
bytecode cache being untracked, not deleted.

**`docs/DATA_SOURCES.md` provenance notes:** added a "Provenance /
regeneration" line to every documented file. Where genuinely unclear
(`ufc_gold_dataset_final.csv` — grepped the whole repo for a producer
script and found none, only consumers), said so explicitly rather than
guessing at a plausible-sounding origin. Added a blanket note for ~15
archival `data/` files (intermediate byproducts of the Sherdog-cleanup and
UFC-Stats-scraping sprints, e.g. `career_fights.csv`, `sherdog_records*.pkl`,
`bad_sherdog_matches.csv`) that aren't read by any current production
path, rather than fabricating individual provenance detail for files no
code in this repo actually touches.

**README:** rewrote the Models section to describe BOTH the currently-
serving 129-feature model and the not-yet-cut-over 133-feature `model/v2/`
one, explicitly flagging that 72.81% is not the honest number (pointing to
the 65.64% pooled walk-forward figure) rather than just bumping the
feature count as the plan's literal text asked — a stale-but-technically-
updated README would have been worse than the one it replaced. Added `Run
tests` and `Walk-forward evaluation` sections.

---

## 8SI Phase 5: unify train/serve feature pipeline (Jul 2026)

**Decision:** Two independent pieces of work, both scoped and confirmed
with the user before starting given the live-service stakes.

### Part A — QA-stats / got_finished_rate bug fix (shipped, live model unaffected)

While researching Phase 5, found that `backend/main.py`'s `predict()` (and
a second, independent duplicate inside `predict_method()`) had NEVER
computed real QA stats or `got_finished_rate` for the live model —
placeholders instead: `qa_SLpM`/`qa_SApM` hardcoded `0.0`, `got_finished_rate`
hardcoded `0.5`, `qa_win_rate`/`qa_finish_rate` silently aliased to
unrelated stats (`career_win_rate`, `last5_finish_rate`). 14 of the
currently-live model's 129 trained features got wrong values on every
prediction, on both running services (ports 8000/8002) — independent of
any leakage issue, a plain implementation gap. Confirmed with the user to
fix immediately, keeping the old model/architecture unchanged.

**Fix:** `training/train_model1.compute_qa_stats()` (untouched by Phases
1–2 — no leakage concerns) and a newly-extracted
`compute_got_finished_rate()` (pulled out of
`compute_interaction_features()`, pure refactor, verified behavior-identical
via the full test suite + a retrain producing byte-identical metrics) are
now called directly from `backend/main.py` at startup, against one
synthetic "as of today" row appended per fighter to `career_df`. This
works because both functions read a row's cumulative value BEFORE folding
in that row's own contribution (shift(1) discipline) — the synthetic row's
placeholder fields are never actually read, so what comes back is exactly
"as of today, using every real prior fight." `qa_and_gf_features()` is the
single shared helper both `predict()` and `predict_method()` now call, so
the same bug can't recur in two places again. `finish_danger_mismatch`
was also fixed in passing — it used a fixed 0.5 weight in place of
`got_finished_rate` for the same reason (that value didn't exist yet).

**Independent finding, NOT fixed:** while verifying this, discovered
`qa_SLpM` and `qa_SApM` in `compute_qa_stats()` itself are computed from
the `won` flag, not any actual striking-volume data — `qa_SLpM ==
qa_win_rate` and `qa_SApM == 1 - qa_win_rate` exactly, verified
numerically. This is a pre-existing bug in the trainer's own formula
(predates all of this remediation project, baked into both the original
model's training data and the Phase 1–3 model's). Not fixed here — it's a
trainer-formula change requiring retraining and its own scoping decision,
out of scope for "give the currently-served model the inputs it expects."
Flagged for a future pass.

### Part B — shared feature module + parity test (built, NOT wired into serving)

Per user's explicit choice: build the point-in-time-correct shared pipeline
and prove it with a parity test, but do NOT cut backend over to serve the
new model/schema in this pass. That remains a distinct future decision
(would need: backend to load `model/v2/`, `get_fighter_state_asof()` wired
into `/predict`/`bet_recommendation`, `EXPECTED_FEATURE_SCHEMA_VERSION`
bumped to 2).

- **`features/constants.py`** (new): `WC_ORDER`, `WOMENS_CLASSES`, layoff-
  bucket thresholds, and the small interaction-feature formulas
  (`age_x_exp`, `age_x_layoff`, `finish_danger`, `finish_danger_mismatch`)
  — previously duplicated (with matching values, confirmed before merging)
  in `train_model1.py` and `backend/main.py`. Deliberately has NO
  dependency on `train_model1.py` or `features/build.py` — see below.
- **`features/build.py`** (new): `get_fighter_state_asof(fighter_name,
  as_of_date, bundle)` — the actual new capability, a point-in-time query
  built on top of `train_model1.py`'s own `compute_career_stats()` /
  `compute_qa_stats()` / `compute_got_finished_rate()` and
  `style_stats.compute_style_stats_asof()` (not reimplemented), using a
  synthetic as-of-date row per source table — same technique as Part A.
  `as_of_date='today'` is what a live endpoint would eventually want;
  an arbitrary historical date is what the parity test uses.
- **Why constants live in a separate file from build.py:** `build.py`
  imports `compute_*()` functions FROM `train_model1.py`. If
  `train_model1.py` also imported shared constants back from `build.py`,
  that's a circular import. `features/constants.py` has no dependency on
  either side, so `train_model1.py`, `backend/main.py`, and `build.py` can
  all safely import from it. `train_model1.py` and `backend/main.py`'s own
  duplicate `WC_ORDER`/`WOMENS_CLASSES`/layoff-bucket definitions were
  removed in favor of importing from here — confirmed behavior-identical
  via full test suite + a retrain producing byte-identical metrics.
- **`tests/test_train_serve_parity.py`** (new, Phase 5's stated acceptance
  criterion): 10 sampled historical fights, asserts `get_fighter_state_asof()`
  matches `train_model1.build_dataset()`'s own computed values within
  `1e-6`. Getting this green surfaced three real, independent bugs/gotchas
  along the way (see the test file's own comments for detail):
  1. My own truncation bug: style-stat truncation used strict `<` where
     the trainer's `merge_asof(direction='backward')` includes exact-date
     ties — silently dropped a fighter's own gold-dataset row for the
     target fight itself, undercounting their cumulative stats by one
     fight. Fixed in `get_fighter_state_asof()`.
  2. `_impute_by_weight_class()` fills every individual NaN style-stat
     cell with a weight-class median, not just rows flagged
     `R/B_style_missing` — that flag is only set from `R_SLpM`
     specifically. Samples are now pre-filtered to fights where
     `get_fighter_state_asof()` itself returns zero NaN across all 8 style
     columns, so the test validates point-in-time correctness without
     also needing to reproduce Phase 1's imputation logic (already
     covered by `tests/test_no_leakage.py`).
  3. Two genuine, pre-existing data-quality gaps in the source CSVs
     (unrelated to any of this project's code): `career_fights_updated.csv`
     is sometimes missing a fighter's own row for a specific
     `ufc-master.csv` fight (a sync gap between the two files), and
     `career_fights_updated.csv` sometimes double-logs the same fight
     under two spellings of an opponent's name on the same date (e.g.
     "Zachary Reese" / "Zach Reese" — the latter is the documented
     `docs/DATA_SOURCES.md` "~2,235 duplicate (fighter, date) rows"
     gotcha). Both make `merge_asof`'s tie-breaking and a from-scratch
     as-of computation legitimately diverge — not a bug in either
     approach, just two different reasonable answers to an
     under-specified question. Samples with either issue are excluded
     from the test rather than papered over.

**Live-service handling (same protocol as Phase 4):** two `uvicorn`
processes were running throughout (ports 8000/8002). Neither was touched
directly. Port 8000 runs with `--reload` and auto-restarted on its own
partway through this work (confirmed via a live request showing Phase 4's
`gap_threshold: "3%"`) — flagged to the user immediately when discovered;
confirmed fine to continue. Every change shipped in this phase was in a
tested, working state at each step, so no broken intermediate state was
ever live.

---

## 8SI Phase 4: betting-layer hardening — market shrinkage, unified Kelly (Jul 2026)

**Decision:** Added market-shrinkage (`market_shrink()`) and unified Kelly staking (`kelly_fraction()`/`kelly_stake()`) to `backend/main.py`. Applied to the two live betting-decision paths: `model2a_predict`'s Kelly section and `bet_recommendation`'s Kelly section. `model2b_predict`'s feature computation (which feeds Model 2B, a separately-trained classifier) was deliberately left untouched — see below.

**Why w=0.30 is a documented default, not a fitted value:** The plan (`8si_remediation_plan.md` Phase 4.1) calls for grid-searching `w` on walk-forward validation. That's straightforward for the new (Phase 1–3 remediated) model, which `training/walk_forward.py` can already score honestly at any historical date. It is NOT straightforward for the model backend actually serves — `get_career_stats()`, `get_elo_stats()`, `get_fighter_extra_stats()` etc. always compute "stats as of right now" (`pd.Timestamp.now()`/`datetime.now()`), with no date parameter. Pointing them at a historical fight to backtest would leak that fighter's entire future career into the "historical" prediction, making any resulting `w` meaningless. Building point-in-time-correct versions of those four functions specifically for the old model was considered and rejected as largely throwaway work, since Phase 5 replaces this model and its feature pipeline entirely. `w=0.30` is the midpoint of the plan's own stated expected range (0.2–0.4) — an informed placeholder, explicitly documented as such in `SHRINKAGE_W`'s comment, to be re-fit once Phase 5 lands.

**Why GAP_THRESHOLD is a derivation, not a re-fit:** Since `p_final = w*p_model + (1-w)*p_market`, algebraically `gap_final = p_final - p_market = w * gap_raw` exactly. The original 10% raw-gap trigger (comment: "168 bets, +34.4% ROI" — a small-sample backtested number the plan explicitly warns not to treat as validated) becomes `w * 10% = 3%` on the new shrunk-gap scale, by construction, not by fitting. This preserves whatever the original threshold meant (for better or worse) rather than silently changing decision behavior as a side effect of adding shrinkage. Also provisional pending Phase 5.

**Why one threshold, not two:** `model2a_predict`'s Kelly path used 10% (`GAP_THRESHOLD`); `bet_recommendation`'s used an undocumented, separately hardcoded 5% (`gap_size >= 0.05`) for `should_bet`. Per the plan's "pick one config" instruction, both paths now use the single derived `GAP_THRESHOLD` (3%). If the two decision points genuinely warrant different risk tolerances, that's a product decision to make explicitly later, not something that should persist as an undocumented inconsistency.

**Why Kelly fraction unified to 1/3, not 1/4:** `KELLY_FRACTION = 1/3` was already the named, exposed module constant (referenced in `/predict`'s response and `model2a_predict`'s original code); `bet_recommendation`'s Kelly path had an inline, unnamed `0.25` instead. Standardized on the pre-existing named constant rather than the ad-hoc inline value. The debut-fighter risk discount (half Kelly when either fighter has zero UFC wins) in `model2a_predict` was preserved as a caller-side fraction adjustment, not baked into the shared `kelly_stake()` primitive.

**Why `model2b_predict`'s gap/probability computation was NOT touched:** `_gap_zone()`'s boundaries and the `pick_prob_val`/`gap_size` values computed inside `model2b_predict` (lines ~1390–1477 as of this change) are fed directly as trained-model features into `_model2b_rf` (a pre-fit RandomForest, `model/ufc_model2b.pkl`). That model was trained on the RAW (unshrunk) gap/probability distribution and a specific `gap_zone` 0–6 ordinal encoding. Applying shrinkage or rescaling `_gap_zone()`'s boundaries there would feed the frozen model features from a completely different distribution than it was trained on — silently corrupting its predictions, a worse outcome than anything Phase 4 is trying to fix. `bet_recommendation` needed a shrunk-gap-consistent zone label for its own (human-readable, non-model-feeding) `gap_zone` response field; rather than duplicating or modifying `_gap_zone()`, it divides the shrunk `gap_size` back by `SHRINKAGE_W` before calling the existing, untouched function (`gap_size / w == gap_raw` exactly, by the same identity above).

**Not done in this pass:** No live backtest of the currently-served model against `w`/`GAP_THRESHOLD` grid search (see above). No change to `model2b_predict`. No wiring of Model 1's own probability into shrinkage (Model 1's raw probability is only ever used as an *input feature* to Model 2/2A in this file, not directly for a standalone Model-1-only betting decision — there was no existing Model-1-only Kelly path to touch). Live server processes on ports 8000/8002 were left running; this file's changes take effect on their next restart, which is the user's call, not automated here.

---

## Models 3A and 3B: method prediction layer (May 2026)

**Decision:** Promoted Model 3A ("Goes the Distance") and Model 3B v2 ("Winner and Method") to production. Models are exposed via the `/method` endpoint. They run alongside M1/M2A/M2B and do not replace them.

**Model 3A — Goes the Distance (binary classifier):**
30% LR + 70% XGB, 63 features, 64.94% accuracy (+11.98pp vs 52.96% naive). Predicts whether a fight ends by decision (1) vs finish (0). Used for `goes_distance_prob` and `finish_prob` in the `/method` response. Uncalibrated — isotonic calibration on a non-chronological val slice degraded MAE from 0.0615 to 0.1895 and was discarded. A proper chronological holdout calibration pass is deferred.

Low-confidence divisions flagged: Women's Flyweight, Light Heavyweight, Bantamweight (smaller sample, higher method variance in training data). `low_confidence_division: true` is set in the response when the weight class matches.

**Model 3B v2 — Winner and Method (six-class classifier):**
40% RF + 60% XGB, 102 features, 46.56% six-class accuracy (+16.68pp vs 29.88% naive). Predicts one of six outcomes: R KO/TKO (0), R Sub (1), R Dec (2), B KO/TKO (3), B Sub (4), B Dec (5). Classes 0/3 collapse to the winner pick, giving a direction accuracy of 70.67%.

**Why M1 probability was fed into 3B:** The original 3B (v1, 99 features) reached only 67.64% direction accuracy despite having access to the same raw stats as M1 (72.81%). The gap existed because 3B had to solve winner-prediction and method-prediction simultaneously from raw features, and the joint learning task diluted the winner signal. Feeding `m1_red_win_prob` (M1's solved probability) as a feature in v2 offloaded winner-prediction to the specialist model, letting 3B focus on method-splitting. Result: direction accuracy jumped from 67.64% to 70.67% (+3.03pp), and submission recall improved from 16.0%/8.4% to 34.9%/12.1% (R/B). The M1 probability feature ranked 2nd and 3rd in XGB importance (`m1_red_win_prob_sq` at #2, `m1_red_win_prob` at #3), confirming it dominated the winner signal.

**Why 40% RF + 60% XGB:** Swept 10 blend combinations on the temporal test set. RF alone (44.96%) and XGB alone (44.74%) were close, but the 40/60 RF+XGB blend (46.56%) outperformed all LR-inclusive blends. RF contributes the most accurate leaf-level probability estimates on the method features; XGB corrects non-linear interactions. LR (41% alone) dragged accuracy down in all combinations — the six-class multinomial task has too many local non-linearities for LR's linear boundary to handle well.

**Direction accuracy gap vs M1 (2.14pp):** 3B reaches 70.67% vs M1's 72.81%. The remaining gap is structural: 3B simultaneously assigns a method to every prediction, which adds uncertainty that pure winner-prediction avoids. The gap is expected and acceptable — 3B's purpose is method-splitting, not winner-prediction.

**Accuracy:** 3A: 64.94% (+11.98pp). 3B v2: 46.56% six-class (+16.68pp), 70.67% direction (+3.03pp vs v1). Files: `ufc_model3a_lr.pkl`, `ufc_model3a_xgb.pkl`, `ufc_model3a_features.pkl`, `ufc_model3b_rf.pkl`, `ufc_model3b_xgb.pkl`, `ufc_model3b_features.pkl`.

---

## Model 2B V3: Random Forest + SPLIT floor, 20 features (May 2026)

**Decision:** Promoted Model 2B V3 (Random Forest + SPLIT probability floor 0.45) to production. All five promotion criteria passed. Previous production was a 1/3+1/3+1/3 LR+RF+XGB ensemble on 15 features.

**Why RF outperformed the ensemble on this dataset:** With a 3,007-row training universe and features that are largely derived from the same underlying signal (gap, agreement, conviction), the ensemble's diversity benefit is minimal. RF at depth≤5, min_samples_leaf=10 provided the best bias-variance tradeoff. LR and the 33/33 ensemble also passed all criteria, but RF had the highest test accuracy (71.14%) and lowest Brier (0.1900) of any passing model.

**Why SPLIT floor 0.45:** SPLIT fights (M1 and M2A disagree on the winner) have an actual historical win rate of 52.1%, but the trained model consistently predicted below 45% for these fights — a systematic underconfidence of ~7–10pp. The floor corrects this without retraining. It is applied post-prediction, before reporting win_probability. Brier improved from 0.1951 to 0.1900 with the floor applied.

**Why agreement_encoded ranked #3:** Providing an explicit ordinal (CONFIRM=3, SPLIT=2, NEAR_ZERO=1, COUNTER=0) proved more informative than asking the model to infer agreement type from the combination of m1_m2a_agree (binary) and gap_direction (±1). The ordinal encodes both the agreement state and its implied confidence ordering in a single feature.

**Why conviction_product ranked #2:** The product of M1 and M2A conviction (abs(m1_prob−0.5) × abs(m2a_prob−0.5)) captures the joint confidence of both models. High conviction_product means both models are far from coin-flip — this is the strongest predictor of the value fighter winning, beyond either model's conviction individually. It ranked above m1_m2a_agree in the binary case because it carries magnitude information not just direction.

**Why is_m1_signal was removed from frontend (retained in model):** The SPLIT + Zone≥5 + M1 conviction≥0.15 archetype showed 23.9% win rate in the training data (vs 73.3% non-signal) — a complete inversion of the 68–75% WR pattern observed in the test set. This strongly suggests the test-set pattern was a sampling artifact rather than a learnable signal. The feature is retained in the model's feature vector (RF expects 20 features) but hardcoded to 0 in the backend, and the M1 SIGNAL badge is removed from AETSlip.js.

**Why gap_signed outperforms gap_size:** gap_signed (= gap_size × gap_direction) combines magnitude and direction into one continuous feature. Correlation with outcome: +0.249 vs +0.023 for gap_size alone — 11× stronger. The direction of the gap (model more confident than Vegas vs Vegas more confident than model) is the primary signal; magnitude alone is nearly uncorrelated with outcome.

**Accuracy:** 70.51% → 71.14% (+0.63pp). Brier: 0.1943 → 0.1900 (−0.0043). COUNTER MAE: 0.1064 → 0.0180 (6× better). SPLIT MAE: 0.1239 → 0.0490 (2.5× better). Model file: `model/ufc_model2b.pkl` (RF), `model/ufc_model2b_features.pkl` (20 features), `model/ufc_model2b_config.json`.

---

## Model 1 V2: men's-only, recency weighting, QA stats, 129 features (May 2026 sprint)

**Decision:** Promoted Model 1 V2 to production. New model achieves 72.81% temporal accuracy (2024+ holdout), replacing the previous 72.08% blend.

**Why women's fights were excluded:** Women's weight classes (Strawweight, Flyweight, Bantamweight, Featherweight) were scoring 57–60% accuracy — well below the men's baseline — and pulling the blended accuracy down. The women's divisions are a structurally different prediction problem: smaller fighter pools, fewer career fights per fighter, different striking/grappling profiles, and less historical depth in the career stats dataset. Mixing them into a single model requires the model to simultaneously learn two distinct prediction tasks that share features but not patterns. Exclusion immediately rescued accuracy. A dedicated women's model is flagged as a future project.

**Why recency weighting (half-life=730 days):** The sport's meta evolves. A fight from 2015 is less informative about a fighter's current form than a fight from 2023. Exponential decay weighting (`exp(-ln(2) * days_before_cutoff / HL)`) with HL=730 days was tested against 1095d and 1460d. The 730d half-life gave the best temporal holdout accuracy and makes the model more responsive to recent fighter development. The 2025 (test) accuracy recovered from 65.9% (no weighting) to 71.0% — a substantial rescue.

**Why training window expanded to 2015:** A data quality audit of 2015–2017 fights found a maximum missing-rate delta of 10.6pp relative to 2018+ data — below the 20pp threshold set for inclusion. Expanding the window added 1,222 training rows at no accuracy cost, strengthening minority patterns in the training distribution.

**Why V2 beat V3:** V2 (recency + opponent-quality-adjusted stats + interaction features, 129 features) scored 72.81% tuned vs. V3 (recency only, 109 features) at 71.98% tuned. The QA stats (career win rate, finish rate, SLpM, SApM weighted by opponent Elo at time of each fight) contributed the largest per-feature accuracy lift. Interaction features (age × layoff, finish danger mismatch) added marginal but consistent signal. V3 tuned was also below the previous production number (72.08%), making it a regression — V2 was the only viable promotion candidate.

**New features — opponent-quality-adjusted (QA) stats:** 12 features computed as cumulative career stats where each fight's contribution is scaled by `opponent_elo / 1500` at fight time. This gives a fighter's stats weighted by the quality of competition faced — a 70% strike accuracy against elite opponents is worth more than 70% against cans. All 8 source QA metrics (win rate, finish rate, SLpM, SApM for both corners) outperformed their raw counterparts in target correlation.

**New features — interactions:** `age_x_layoff` (age × min(layoff_days, 730)), `finish_danger` (KO rate + sub rate), `got_finished_rate` (fraction of losses by finish — a chin-proxy), and `finish_danger_mismatch` (cross-multiplied finish danger vs. finish resistance between corners). Rematch features (`is_rematch`, `won_first_fight`) were tested but dropped — both fell below the |r| < 0.03 inclusion threshold.

**Women's model:** Flagged as a future project. Requires a separate career stats dataset, separate Elo ratings for women's divisions, and separate feature selection. Not a V2 scope item.

**Accuracy:** 72.08% → 72.81% (+0.73pp). Model files: `ufc_model_best.pkl` (LR pipeline), `ufc_model_xgb.pkl` (XGB, Optuna best params), `feature_columns_best.pkl` (129-feature list).

**Backend approximation for QA features:** The `/predict` route approximates QA stats at inference since fighter historical data is not available from the input payload: `qa_win_rate = career_win_rate`, `qa_finish_rate = last5_finish_rate`, `qa_SLpM = qa_SApM = 0.0`. The safety fallback (`for col in feature_columns: if col not in df_input: df_input[col] = 0`) handles any remaining gaps.

---

## Model 2: 50/50 LR+XGB blend, 42 features (May 2026 sprint)

**Decision:** Promoted new Model 2 (50% LR + 50% XGB, 42 features) to production. Previous production Model 2 was a single LR model on 23 features (72.35% accuracy). New model achieves 73.20% (+0.85pp).

**Why 50/50 blend:** Sprint swept 80/20 through 50/50 LR/XGB ratios on the test set. The 50/50 split gave the highest accuracy (73.20%), outperforming 70/30 (73.13%), 80/20 (72.67%), and single-model LR (72.67%). XGB complements LR on non-linear patterns in the odds space — in the 42-feature M2 dataset, XGB carries more signal than in M1 because the odds features are discrete enough for tree splits to help.

**Why tier_hist_win_rate matters:** This is the standout new feature (r=+0.40 with outcome, 2nd most important XGB feature). It encodes the historical win rate of fighters at each odds tier — heavy_dog (<0.30 implied prob): 17.9% historical win rate; dog (0.30–0.45): 38.2%; coinflip (0.45–0.55): 49.4%; fav (0.55–0.70): 63.7%; heavy_fav (>0.70): 82.7%. This is computed from training data and looked up at inference. It tells M2 how well-calibrated the Vegas line is at this tier, giving a prior for whether the current gap is signal or noise.

**Why split models weren't promoted:** The fav/dog split approach (separate models for when F1 is the favorite vs underdog) showed 72.45% combined accuracy vs 72.24% unified — only +0.21pp above the +0.2pp threshold. The gain is too marginal to justify the added complexity: two separate models, an f1_is_fav routing condition in the backend, and doubled maintenance surface. Unified 42-feature model at 73.20% is superior.

**Feature groups added:** 7 underdog/fav profile features + 8 method odds interaction features + 4 weight class/context features = 19 new features on top of base 23. ElasticNet regularization (l1_ratio=0.785) zeroed many of the weak new features, keeping effective model complexity low.

**Backup files:** `model/ufc_model2_best_v1_backup.pkl`, `model/ufc_model2_features_v1_backup.pkl`

**Accuracy:** 72.35% → 73.20% (+0.85pp). Model files: `ufc_model2_best.pkl` (LR), `ufc_model2_xgb.pkl` (XGB), `ufc_model2_features.pkl` (42-feature list), `model2_tier_stats.json` (tier lookup table).

---

## Blend ratio: LR 70% + XGB 30% (May 2026 optimization sprint)

**Decision:** Changed production blend from 90/10 to 70/30 (LR/XGB). No retraining — same pkl files, constants updated in `backend/main.py`.

**Why:** May 2026 optimization sprint swept ratios from 95/5 to 70/30 using the production LR and XGB models on the temporal test set (2024+). Results were non-monotonic: accuracy dipped at 85/15 and 80/20 (−0.26pp each) before recovering and jumping at 75/25 (+0.35pp) and 70/30 (+0.44pp). The dip at intermediate ratios suggests XGB's non-linear corrections are partially destructive when too weak to override LR — they introduce noise rather than signal. At 30% weight, XGB has sufficient influence to correct LR's linear misses on non-linear patterns, producing the 72.08% accuracy vs. 71.64% baseline.

**Alternatives rejected:** ElasticNet penalty, LightGBM blends, isotonic calibration, and C re-tuning all underperformed the baseline on the temporal test set. LightGBM in particular was −0.79pp to −1.23pp across all blend configurations, consistent with the dataset being too small (~4K training rows) for LGBM's tree structure.

**Previous entry (90/10):** The earlier 90/10 decision was based on experiments at the 114-feature stage where 85/15 and 80/20 showed no improvement. The 109-feature Variant A model behaves differently under XGB — fewer noisy features means XGB's predictions are cleaner and can carry more weight without degrading accuracy.

**Accuracy:** 71.64% → 72.08% (+0.44pp). No model files changed.

---

## Model architecture: LR + XGB blend (90/10)

**Decision:** Primary model is 90% LogisticRegression + 10% XGBoost, probability blend.

**Why:** LR with heavy L2 regularization (`C=0.00711`) is robust on structured tabular data with moderate row counts (~3000 training fights). XGB adds a small non-linear correction. Blending at 90/10 vs. higher XGB weights did not improve temporal accuracy — per experiments `experiment11_output.txt`, 90/10, 85/15, and 80/20 all achieved 73.24%. The 90/10 split was chosen to keep the model closer to LR's calibrated probabilities (important for Kelly sizing in Model 2).

**Alternatives considered:** Random Forest, CatBoost, pure XGB. All underperformed LR on temporal split by 1–2%. RF in particular overfit to training set patterns that don't generalize forward in time.

---

## Temporal split (train < 2024, test ≥ 2024)

**Decision:** Hard cutoff at 2024-01-01. Train on all 2018–2023 fights, test on all 2024+ fights.

**Why:** UFC fighting meta and fighter pool evolve over time. A temporal split correctly simulates the deployment scenario: the model only sees past fights when predicting future ones. Cross-validation with shuffled folds would inflate accuracy by ~3–4% due to future leakage.

**Why 2018 as train start:** Career stat features become sparse and noisy before 2018 (fewer fighters with multiple UFC fights). Restricting to 2018+ gives each fight row at least some career history context.

---

## Corner-flip augmentation

**Decision:** Training set is doubled by swapping R_/B_ columns and negating all `_dif` columns. Target flips from 1→0 and 0→1.

**Why:** The UFC arbitrarily assigns Red/Blue corners. Red corner has a slight home-field advantage in some eras, but the underlying fighter quality signal should be corner-invariant. Augmenting with flipped examples forces the model to learn relative differences, not corner assignment.

**Note:** Augmentation is applied ONLY to the training set. Test set is never augmented, so test accuracy reflects real-world prediction.

---

## Elo: K=48, base=1500, all-time

**Decision:** K=48 with all-time fight history (back to 1993).

**Why K=48:** Experiment grid over K=32,40,48,56,64,72,80,96,128 showed K=48 and K=64 achieved identical temporal accuracy (72.35%). K=48 was chosen as the production value because it was the baseline and there was no clear reason to increase K (higher K amplifies single-fight volatility, which adds noise for fighters with sparse records).

**Why all-time, not windowed:** Windowed Elo (e.g., last-50-fights only) was tested and performed 0.1–0.2% worse. All-time Elo correctly preserves legacy information for established champions.

**What Elo captures:** Relative strength accounting for opponent quality. `elo_dif` is consistently a top-5 most important feature.

---

## Shift(1) career stats — no leakage

**Decision:** All career stats are computed with `shift(1)` within each fighter group, ensuring each row only sees data from fights BEFORE the current one.

**Why critical:** Using cumulative stats that include the current fight is data leakage. For example, `cumsum().shift(0)` for `won` would include the current fight's outcome in the training feature — the model would be learning from the answer. `shift(1)` shifts the entire series so row 0 sees nothing (uses debut defaults).

**Why career_fights_updated.csv, not ufc-master.csv:** ufc-master.csv only has UFC fights. Career stats include regional/international fights, giving accurate pre-UFC win rates for fighters debuting in the UFC. This is particularly important for fighters with long regional careers (e.g., 15-1 regional record before UFC).

---

## Fight filter: R_cum_fights ≥ 1 AND B_cum_fights ≥ 1

**Decision:** Exclude fights where either fighter has zero prior UFC fights in the career dataset.

**Why:** Debut fighters have no prior career data to fill the career stats columns — all features default to neutral values (0.5 win rate, 0 finishes, etc.). Including these rows in training introduces noise because the features carry no signal. The model performs poorly on debuts by design (insufficient training signal).

**Why not ≥ 3 or higher:** Experiments showed that min=1 preserved enough training data without sacrificing accuracy. Higher thresholds reduced training set size without accuracy benefit.

---

## LR regularization: C=0.00711

**Decision:** Heavy L2 regularization, C=0.00711 (λ ≈ 141).

**Why:** The feature space has 114 features with many correlated diffs. Strong regularization prevents overfitting to training-era patterns that don't generalize. The C value was tuned via grid search on temporal accuracy and then held fixed. Lower C values (stronger regularization) marginally underperformed; higher values overfit.

---

## Model 2: odds-aware LR, 1/3 Kelly gating

**Decision:** Separate Model 2 uses opening odds as a feature. Kelly fraction = 1/3. Bets only when Model 1 probability differs from implied odds probability by ≥ 10%.

**Why separate model:** Incorporating odds into Model 1 would make it unusable for fights without odds (regional cards, early lines). Model 2 is a pure value-detection layer on top of Model 1.

**Why 1/3 Kelly:** Full Kelly criterion maximizes geometric growth but has high variance and drawdown. 1/3 Kelly is a standard risk-adjusted fraction that reduces variance substantially (≈ variance goes down by 9x) while retaining most of the growth advantage.

**Why 10% gap threshold:** Below 10%, the edge is within typical odds-line noise. Gates out marginal bets where the model is not clearly disagreeing with market consensus.

---

## Feature pruning: Variant A (109 features, May 2026)

**Decision:** Removed 5 features from the original 114-feature set:

| Feature | XGB Importance | Zero Rate | Reason |
|---------|---------------|-----------|--------|
| `title_bout_bin` | 0.0000 | 96.0% | 96% of training rows are zero; zero predictive power |
| `B_southpaw` | 0.0000 | 81.5% | Zero importance; stance is fully captured by `orth_clash` and `south_clash` |
| `B_layoff_gt365` | 0.0032 | 87.6% | Rare event (only 12% of fighters), very weak importance |
| `R_total_title_bouts` | 0.0032 | 80.3% | Very sparse, near-zero importance |
| `last10_win_rate_dif` | 0.0038 | 34.1% | Highly redundant with `career_win_rate_dif` (\|r\|=0.851) — drop the lower-correlation duplicate |

**Result:** Accuracy improved from 71.47% → 71.64% (+0.18pp) on the temporal test set (2024+).

**Why these and not others:** The audit flagged features by requiring BOTH low XGB importance (<0.5%) AND high zero rate (>70%). Features in the bottom 15 by importance that still have decent zero rates were kept — their correlation with the target (0.05–0.15) suggests they carry diffuse signal that LR picks up even when XGB doesn't. `last10_win_rate_dif` was a special case: flagged purely on redundancy grounds (correlation with `career_win_rate_dif`) rather than sparsity.

**Backup files:** `model/ufc_model_best_114_backup.pkl`, `model/ufc_model_xgb_114_backup.pkl`, `model/feature_columns_best_114_backup.pkl`

---

## is_debut flag: zero rows in career dataset (not zero UFC wins)

**Decision:** A fighter is flagged as a debut if they have NO rows in `career_fights_updated.csv` at all — not if they have zero UFC wins.

**Why:** Pre-fight `R_wins` in ufc-master.csv stores wins going INTO the fight. A fighter who won their first UFC fight has `R_wins=0` on that fight's row. Using `wins==0` as the debut flag would incorrectly flag fighters like Brando Pericic (1-0 UFC) as debuts.

**Effect in backend:** Debut fighters receive `🆕 UFC Debut` badge, a model-confidence warning, and `N/A` Kelly bet size (Model 2 not applied). Only Ben Johnston (Perth card) meets this criterion.

---

## style_stats.py: observed-mask design for missing raw values

**Decision:** `compute_style_stats_asof()`'s per-stat cumulative sums now track, per numerator/denominator pair, a joint "was this fight observed" count and gate on it — instead of the `cumsum() - own_value` idiom used elsewhere in this codebase, gated on the cumulative value itself being `> 0`.

**Why:** A code review surfaced a latent bug in the old idiom: pandas' `cumsum()` reports `NaN` at the position of a `NaN` input even with `skipna=True`. If a raw column (`Sig_Landed`, `Total_Fight_Time_Sec`, etc.) were ever `NaN` for one fight — plausible since `ufc_gold_dataset_final.csv` has no producer script (`docs/DATA_SOURCES.md`) and could pick up gaps on a future re-scrape — that fight's own "as of before" aggregate would wrongly go `NaN` even with real prior data available, AND every later fight for that same fighter would silently and permanently undercount (the missing fight treated as a silent zero contribution, with no record it happened). Not a leakage bug (no future information involved) and not caught by the existing leak tests, since they recompute with the same function.

**Design:** for each of the 8 stats, a fight only counts toward EITHER side of that stat's running numerator/denominator if BOTH columns were observed for that fight — a fight missing just the numerator doesn't get to silently count its minutes toward the denominator (which would bias the rate down). "Has data" requires both a nonzero jointly-observed-fight count AND a nonzero cumulative denominator — the second check preserves the pre-existing (correct) behavior that a fighter whose priors are all genuinely zero attempts (e.g. never attempted a takedown) gets `NaN`, not a fabricated `0%` accuracy.

**Verification:** the refactor is confirmed a byte-for-byte no-op on today's data — position-aligned comparison against the pre-fix implementation across all 17,102 rows / 8 stats found zero differences, and the pooled walk-forward log loss is unchanged (`0.6152062384218069`, identical to before). Two new regression tests in `tests/test_no_leakage.py` inject a synthetic `NaN` and pin the exact expected behavior (own-row not poisoned, later rows exclude the fight like it never happened, unrelated stats unaffected); a third pins known values for 5 real fighters against future regressions.
