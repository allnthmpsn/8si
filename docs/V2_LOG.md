# V2 Log — 8SI v2 Build Spec

Every experiment that could plausibly change `training/train_model1.py`'s
defaults gets one entry here: what changed, the pooled walk-forward log
loss before/after, and the decision. Per the v2 spec's own governing
rules, **pooled walk-forward log loss (5 folds, test years 2021–2025,
symmetric inference, via `training/walk_forward.py`) is the only metric
that decides anything** — not the single 2024+ holdout (compromised for
model-selection since `docs/REBASELINE.md` Phase 2/3), not accuracy, not a
lone-feature proxy signal. A change ships only if it improves the pooled
number, no exceptions for features that "should" work. All prior v1 ROI
figures in `data/value_bet_log.csv` are formally void (see Stage 0.4).

## Market benchmark (permanent reference line)

`training/market_baseline.py` — no-vig market performance on the exact
same test population `training/walk_forward.py` evaluates the model
against (same 5 folds, same `build_dataset()` row filtering). Every
experiment's pooled log loss below should be read against this line, not
against 50/50.

| Metric | Value |
|---|---|
| Test years | 2021–2025 |
| Total test fights (men's, 2015+ window, `cum_fights>=1`) | 2,087 |
| Odds coverage | 1,899/2,087 (91.0%) |
| Pick-favorite accuracy | 67.72% |
| **Market log loss** | **0.5991** |
| Market Brier | 0.2063 |

**The market currently beats the model on every axis** (model pooled log
loss at Stage 0 start: 0.6152 vs. market 0.5991). Closing this gap is the
whole point of v2 — nothing here should be read as "the model is good,"
only "here is how big the gap is and whether a given change narrows it."

## Stage 0.1 — Market baseline

Landed above. No trainer change.

## Stage 0.2 — Drop duplicate QA-stat features

`compute_qa_stats()` (train_model1.py:478) computes `qa_SLpM`/`qa_SApM`
from `cum_off`/`cum_def`, which are incremented by exactly `ew*w` and
`ew*(1-w)` each fight — algebraically identical to `cum_eww`/`cum_ew -
cum_eww`, the same accumulators `qa_win_rate` is built from. Verified
numerically: `qa_SLpM == qa_win_rate` and `qa_SApM == 1 - qa_win_rate`
exactly, for every row. The function was never actually fed striking-
volume data — this predates the whole remediation project (flagged but
not fixed in `docs/REBASELINE.md` Phase 5 Part A) and is a plain
implementation bug, not a leakage issue. Dropped `qa_SLpM`/`qa_SApM` and
their two `_dif` columns (6 columns total: `R/B_qa_SLpM`, `R/B_qa_SApM`,
`qa_SLpM_dif`, `qa_SApM_dif`) from `FEAT_QA`. `compute_qa_stats()` itself
is untouched — still computes and returns all 4 columns, since
`features/build.py`'s `get_fighter_state_asof()` is a general lookup
utility, not tied to the trainer's exact feature list.

Features: 133 → 127. `FEATURE_SCHEMA_VERSION` bumped 2 → 3.

| | Features | Pooled log loss | Δ |
|---|---|---|---|
| Before (Phase 7 baseline, `docs/REBASELINE.md`) | 133 | 0.6152 | — |
| After (QA dup drop only, K=48+method+layoff unchanged) | 127 | **0.6166** | **+0.0014** |

**Regression, shipped anyway.** This is in tension with the spec's own
rule #3 at face value, resolved as follows: (a) this is an explicit,
specific plan instruction motivated by removing a known formula bug, not
a speculative "should work" feature — the columns were never real signal,
just `qa_win_rate` copied under two other names; (b) the regression
(0.0014) is below the plan's own stated 0.002 "matters" threshold used
elsewhere (RAPM gating in the spec's Stage 2.6); (c) Stage 2's RAPM work
is explicitly meant to properly replace this signal with something real,
not leave the gap unfilled. Reporting the actual (negative) delta here
rather than assuming "~no change."

## Stage 0.3 — Elo K-grid extension

Phase 7's original grid `{24,32,40,48}` found K=48 optimal in that range
and promoted it alongside method-weighted K (×1.25 KO/TKO/Sub, ×0.75
split decision) and 25% layoff regression (>365 days inactive). v2 Stage
0.3 asks whether a wider grid still prefers K=48 now that the QA dup
features are gone.

**Step 1 — lone-feature (`elo_dif` only) LR diagnostic**, cheap, wide
range, method-weighting/layoff-regression held at zero (isolates K only):

| K | 48 | 64 | 80 | 96 | 112 | 128 | 160 | 192 | 224 | 256 | 320 | 384 | 448 | 512 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Pooled log loss | 0.6783 | 0.6774 | 0.6766 | 0.6760 | 0.6754 | 0.6750 | 0.6745 | 0.6742 | **0.6741** | 0.6742 | 0.6747 | 0.6752 | 0.6758 | 0.6762 |

Monotonically improving all the way out to K≈224 before reversing — the
diagnostic's own "best" is K=224.

**Step 2 — full-model (127-feature) validation**, method-weighting +
layoff-regression held at production values, only K varying (this is the
actual apples-to-apples comparison for a promotion decision):

| K | 48 | 80 | **96** | 112 | 192 | 256 |
|---|---|---|---|---|---|---|
| Pooled log loss | 0.6166 | 0.6163 | **0.6152** | 0.6164 | 0.6165 | 0.6175 |

**The lone-feature diagnostic's global suggestion (K≈192–256) is worse
for the full model than K=96** — a genuine, non-monotonic local optimum
the cheap proxy signal missed entirely, because it only measures how much
signal `elo_dif` carries in isolation, not how that signal interacts with
the other 126 features once blended. This is the spec's own governing
rule #3 playing out directly: the diagnostic is useful for cheaply
narrowing a wide search space, but only the full-model pooled metric
decides anything. K=96 confirmed against both neighbors (K=80, K=112) and
both wide-range candidates (K=192, K=256) — all four worse.

**Promoted `ELO_K = 96`** in `training/train_model1.py`, and mirrored into
`features/build.py`'s independent `compute_elo()` call (`DataBundle.__init__`)
so the two computations can't silently diverge — this exact class of bug
(a promoted Elo config updated in `build_dataset()` but not in
`features/build.py`) broke `tests/test_train_serve_parity.py` once already
during Phase 7 (`docs/REBASELINE.md`). Verified: `pytest tests/` (86/86
pass, parity test included) after the promotion.

## Stage 0.2 + 0.3 combined (confirmed via `training/walk_forward.py`)

| Metric | Value |
|---|---|
| **Pooled log loss** | **0.6152** |
| Pooled accuracy | 65.93% |
| Pooled Brier | 0.2134 |

Nets out to the same pooled log loss as the pre-Stage-0 Phase 7 baseline
(0.6152) — the QA dup-feature removal's small regression (+0.0014) is
fully offset by the K retune (−0.0014). Net effect on the headline number
is a wash; the actual gain is hygiene (6 dead/duplicate columns removed,
a stale hyperparameter corrected) that Stage 2's real feature work builds
on cleanly.

| Fold (test year) | N test | Accuracy | Log loss | Brier |
|---|---|---|---|---|
| 2021 | 398 | 60.55% | 0.6438 | 0.2263 |
| 2022 | 418 | 64.35% | 0.6281 | 0.2191 |
| 2023 | 413 | 64.65% | 0.6253 | 0.2182 |
| 2024 | 424 | 70.52% | 0.5940 | 0.2035 |
| 2025 | 434 | 69.12% | 0.5877 | 0.2012 |
| **POOLED** | **2,087** | **65.93%** | **0.6152** | **0.2134** |

Still well short of the market's 0.5991 (Stage 0.1) — expected, that gap
is what Stages 1–2's real feature work (round-level scraping, RAPM, etc.)
is for.

## Stage 0.4 — Void the v1 bet log

`data/value_bet_log.csv` (3,007 rows) turned out to be Model 2B's
training data, not a live P&L log (`docs/REBASELINE.md` Phase 4.3) — none
of the existing columns were touched, only a new `provenance` column
added, set to `v1_void` for every existing row. `training/clv_report.py`
now excludes `provenance == 'v1_void'` rows from its headline CLV numbers
by default (`--include-void` / `include_void=True` to see the full
history anyway) — per the spec's rule that all prior v1 ROI/CLV figures
are formally void, a stale number can no longer surface from the log
just by running the report normally. Since every row is currently void,
`clv_report.py`'s default output correctly reports zero non-void bets
until real v2-era bets are logged.

## Stage 0 acceptance criteria

- [x] Market baseline numbers in this log (Stage 0.1).
- [x] 6 dead features removed (`qa_SLpM`/`qa_SApM` + their 4 `_dif`
      columns — the spec anticipated 4, actual count is 6).
- [x] K adopted (96) with parity green (`tests/test_train_serve_parity.py`
      passes against `features/build.py`'s mirrored config).
- [x] Bet log partitioned (`provenance` column, all-void, `clv_report.py`
      updated).
- [x] All tests pass (86/86, `pytest tests/ -q`).
- [x] Serving paths untouched — `model/` (root), `data/elo_current.csv`,
      `data/elo_ratings_history.csv` not touched by this stage's retrains,
      which wrote to `model/v2/` per the Phase-2-era infra convention
      (`docs/REBASELINE.md`). Backend continues to serve the original
      pre-remediation model, unchanged.

## Next

Stage 0 is complete. Stage 1 (data ingestion — round-level scraping, name
reconciliation, odds pipeline) is a distinct scope with external network
dependencies (scraping, dataset downloads, a live odds API) not present
in Stage 0's self-contained work — a deliberate checkpoint before starting
it.
