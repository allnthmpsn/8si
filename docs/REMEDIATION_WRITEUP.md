# 8SI Model 1 Remediation — Phases 0–7

A full account of the leakage audit, honest-evaluation rebuild, betting-layer
hardening, train/serve unification, repo hygiene pass, and Elo tuning carried
out on the 8SI UFC fight predictor — what changed, what broke along the way,
and what's still deliberately left undone.

Two numbers matter more than any other in this document. The originally
reported **72.81%** single-holdout accuracy was never the honest number — it
survived two confirmed temporal leaks and a biased evaluation window. The
number that replaced it, **pooled 5-fold walk-forward accuracy**, ended this
project at **66.08%** (log loss **0.6152**), up from **65.64%** (0.6200) once
the leaks were closed and the Elo model was retuned.

Both live serving processes (ports 8000/8002) ran throughout this entire
project and were never directly touched. They still serve the original
pre-remediation model — the cutover to the remediated one is a deliberate,
separate decision, not something that happened as a side effect of this work.

## Metrics across phases

| Phase | Single-holdout acc. | Log loss | Brier | Pooled walk-fwd acc. | Pooled log loss |
|---|---|---|---|---|---|
| 0 — Safety nets | 72.50% | — | — | — | — |
| 1 — Style-stat fix | 68.96% | 0.5961 | — | — | — |
| 2 — opp_quality fix | 68.75% | 0.5965 | — | — | — |
| 3 — Honest evaluation | 69.27%* | 0.5960 | 0.2048 | **65.64%** | **0.6200** |
| 7 — Elo upgrades (final) | 68.96% | 0.5932 | 0.2035 | **66.08%** | **0.6152** |

\* The Phase 2→3 jump is a measurement change (symmetric inference), not a
model change. Phases 4–6 made no trainer changes. See Phase 3 below for why
pooled walk-forward, not single-holdout, is the number to trust.

---

## Phase 0 — Safety nets

**Status: shipped.** Make the trainer actually run, and build the tests every
later phase leans on.

The committed trainer could not complete a run. `corner_flip` was defined to
take two arguments and return two values, but was called with three arguments
and unpacked into three variables — a hard `TypeError` on every invocation.
Fixed by extending the function to thread a sample-weight series through the
corner-flip augmentation alongside `X` and `y`.

Beyond the fix: a pinned `requirements.txt` (catboost excluded after
confirming it's unused anywhere in the production path); `main()` refactored
to accept `train_start`/`train_cutoff`/`data_dir`/`output_dir` as parameters
instead of hardcoded module constants, with defaults that reproduce the exact
original behavior; a smoke test; and a leakage-detection test suite that
recomputes each stat from source data truncated to strictly before a sampled
fight's date, and compares against the pipeline's own merged value.

That leakage suite immediately proved its worth: written against the
then-current pipeline, it passed for career stats, QA stats, and Elo (already
correctly point-in-time), and failed — as designed — for style stats and
`opp_quality`, reproducing both leaks the rest of this project exists to fix.

- Temporal accuracy: 72.50%
- Features: 129
- Known leaks: 2 (style stats, opp_quality)

**Defect caught:** `corner_flip` TypeError — the trainer could not run at
all. 2-parameter definition, 3-argument call site. Fixed by extending the
function's signature and return tuple to match its own call site.

---

## Phase 1 — Kill the style-stat leak

**Status: shipped.** Style stats were merged from a career-to-date snapshot
with no date dimension — every historical fight saw the fighter's future
averages.

The plan's own text pointed at `data/ufc_stats_fights.csv` as the as-of
source. It doesn't have the columns needed — no strike or takedown counts at
all, only fighter/opponent/date/result/method/round/time. The actual source
with both fighters' attempt counts per fight (needed to compute defensive
rates like `Str_Def`/`TD_Def`, which require knowing what the *opponent*
attempted) turned out to be `data/ufc_gold_dataset_final.csv`, undocumented
for this purpose until this phase.

New module `training/style_stats.py` computes all eight style stats as
cumulative, pre-fight (shift-1) rates from that source. The previous
snapshot-based merge is gone from the trainer entirely — replaced with an
as-of `merge_asof`, matching the pattern already used for career stats, QA
stats, and Elo. Missingness (fighters with no prior gold-dataset fight) now
gets a `style_missing` indicator and training-window weight-class-median
imputation, replacing a blind `fillna(0.0)`; `got_finished_rate` got the same
treatment, going from a hardcoded 0.5 to a real computed value with its own
missingness flag.

**Result:** accuracy 72.50% → 68.96%; features 129 → 133. The drop is the
expected, correct outcome — 2016 fights could previously see a fighter's 2026
lifetime averages. A dropping accuracy number was the actual acceptance
criterion for this phase, not a regression to explain away.

---

## Phase 2 — Kill the opp_quality leak

**Status: shipped.** Opponent quality was scored by each opponent's
full-career win rate, not their win rate as of the fight being evaluated.

`opp_quality` — the mean win rate of a fighter's last five opponents —
correctly windowed *which* five opponents by position, but scored each one
using a dictionary of full-career win rates built once over the entire
dataset. Fixed by scoring each opponent using their own cumulative win rate
strictly as of the target fight's date, via `np.searchsorted` against each
opponent's own fight history.

**Defect caught before shipping:** the first implementation reused the
existing pre-fight (shift-1) `career_win_rate` column for the opponent
lookup, which excludes that opponent's own most recent result. What's
actually needed is the opponent's *post*-fight cumulative rate through their
last fight strictly before the target date — a fight that already happened is
not future information. The leakage test caught this immediately: 8 of 10
sampled cases failed before the fix, all passed after.

The `all_win_rates` dictionary and its full-career computation were removed
from `main()` entirely — `compute_career_stats()` no longer takes it as a
parameter. No feature columns changed, only the formula behind one of them.

**Result:** accuracy 68.96% → 68.75%. No `xfail` markers remain in the
leakage suite.

---

## Phase 3 — Honest evaluation

**Status: shipped.** Symmetric inference, real baselines, walk-forward
validation, and a calibration attempt that made things worse — kept anyway,
as a documented negative result.

**Symmetric inference.** `predict_symmetric()` averages the model's direct
prediction with the complement of its prediction on the corner-swapped input.
Corner-flip augmentation at training time doesn't make a model perfectly
symmetric — it can still answer differently depending on which fighter is
drawn red versus blue. This one change moved single-holdout accuracy from
68.75% to 69.27%: a change in how the model is *read*, not in the model
itself.

**Metrics and baselines.** Log loss, Brier score, and a 10-bin calibration
table were added to every evaluation. Two baselines were computed on the same
test set: picking the higher-Elo fighter (55.21%) and always picking the
market favorite from opening odds (68.63%, at 88.6% odds coverage). The model
beat the favorite baseline by only 0.64 points — a number the single accuracy
figure alone would never have surfaced.

**Walk-forward evaluation.** New `training/walk_forward.py`: five independent
folds, training on `[2015, N)` and testing on year `N` for `N` in 2021–2025,
each retrained from scratch. Pooled result — **65.64%** accuracy, **0.6200**
log loss — became the number this whole project treats as honest, replacing
the single 2024+ holdout, which had been reused across enough experiment
sprints to be a biased target for model selection.

**Calibration — tried, and undone.** Isotonic regression, per the plan's
literal suggestion, made things measurably *worse*: log loss went from 0.5960
to 0.8005. Diagnosed rather than assumed — isolated the deployed-vs-
calibration-model mismatch as a non-cause by testing isotonic against its own
matched model directly, and confirmed the real issue was sample size: the
~600-row calibration window is too small for isotonic's flexible, many-knot
fit, which is exactly what sklearn's own documentation warns about. Switched
to Platt (2-parameter sigmoid) scaling instead — still a small net negative
(0.5960 → 0.6058), because the raw blend was already reasonably calibrated
and there wasn't much error left for ~600 rows to usefully correct. The
calibrator artifact is saved (`calibrator.pkl`) per the phase's acceptance
criteria, but is not applied to any reported metric or wired into serving.

**Result:** single-holdout accuracy 68.75% → 69.27%; pooled walk-forward
65.64% accuracy / 0.6200 log loss.

---

## Phase 4 — Betting-layer hardening

**Status: shipped.** Market shrinkage, one Kelly implementation instead of
two, and CLV tracking — on `backend/main.py`, discovered live and running on
two ports mid-investigation.

**Defect caught:** two uvicorn processes were already running this code —
ports 8000 and 8002, running since May and July respectively, with real
bankroll/bet-sizing constants wired in. Work paused to confirm handling
before any edit: keep the code changes going in, but never touch the running
processes directly — restart timing is the user's call. Held to that for the
rest of the project.

`market_shrink()` blends the raw model probability toward the no-vig market
consensus (`w = 0.30`, a documented default rather than a grid-searched
value — the model actually being served can't be backtested against
historical dates without leaking each fighter's future career into the past,
since its own feature lookups always compute "as of right now").
`GAP_THRESHOLD` was re-derived algebraically rather than re-fit: since
`gap_final = w × gap_raw` exactly, the original 10% trigger becomes 3% on the
new scale by construction. Two divergent Kelly implementations (1/3 fraction
with one gate, 1/4 with another) were unified into shared `kelly_fraction()`/
`kelly_stake()` functions.

**Defect caught:** "two Kelly implementations" were actually two different
model ensembles. What looked like simple duplication was Model 2's own Kelly
path and a separate Model 1 + Model 2A agreement-based Kelly path —
genuinely different logic, not a copy-paste. Unified the sizing math without
collapsing the two decision contexts into one.

**Defect caught:** a shared gap function was secretly a trained model's input
feature. `model2b_predict()` feeds its own gap/probability computation
directly into a separately-trained RandomForest (Model 2B) as raw features,
on the original unshrunk scale. Applying the new market-shrinkage logic there
would have fed that frozen model a distribution it was never trained on,
silently corrupting its predictions — worse than anything this phase was
trying to fix. Left it untouched; used a division trick (`gap_size / w`) so
`bet_recommendation()`'s human-readable zone label stays consistent without
touching the function Model 2B depends on.

**Defect caught:** the "value bet log" is actually Model 2B's training data.
`data/value_bet_log.csv` has a `split` column of `train`/`test` — it's not a
live P&L log at all. CLV columns (`odds_taken`, `novig_prob_taken`,
`novig_prob_close`, `clv_pct`) were added purely additively via a new
backfill script; none of the existing columns Model 2B was fit on were
touched. Real snapshot-backed CLV coverage as of this pass: 0 of 3,007 rows —
the only available odds-snapshot data covers one card not yet reflected in
the historical log. Reported honestly rather than papered over.

The line-movement feature (4.4) was explicitly skipped: the plan's own stated
coverage bar is ~60% of training rows, and available snapshot data covers 13
matchups from one card.

---

## Phase 5 — Unify train/serve feature pipeline

**Status: shipped.** Two independent pieces: a live bug fix that shipped
immediately, and a proven-but-not-deployed shared feature pipeline.

### Part A — a bug older than this entire project

**Defect caught:** 14 of the live model's 129 features were placeholders, on
every prediction. `backend/main.py`'s `predict()` — and a second,
independently duplicated copy inside `predict_method()` — had never computed
real QA stats or `got_finished_rate`. `qa_SLpM`/`qa_SApM` were hardcoded to
0.0, `got_finished_rate` hardcoded to 0.5, and `qa_win_rate`/`qa_finish_rate`
were silently aliased to unrelated stats. Independent of any leakage issue —
a plain implementation gap, live on both running services.

Fixed by calling the trainer's own `compute_qa_stats()` and a newly-extracted
`compute_got_finished_rate()` directly, against a synthetic "as of today" row
appended per fighter — the same shift-1 discipline used everywhere else in
this project means the synthetic row's placeholder fields are never actually
read. `finish_danger_mismatch` was fixed in passing for the same reason: it
had been using a fixed 0.5 weight in place of the `got_finished_rate` that
didn't exist yet.

**Defect caught, flagged not fixed:** a second, unrelated formula bug. While
verifying the fix above: `qa_SLpM == qa_win_rate` and
`qa_SApM == 1 − qa_win_rate`, exactly, verified numerically.
`compute_qa_stats()`'s own formula never actually consumed striking-volume
data — it reused the win/loss flag as a stand-in. This predates the entire
remediation project and is baked into both the original model's training
data and every model retrained since. Not fixed here: it's a trainer-formula
change that requires a retraining decision, out of scope for "give the
currently-served model the inputs it was trained to expect."

### Part B — built, deliberately not deployed

Per an explicit choice made before starting: prove the unified pipeline
works, but do not cut the live services over to it. `features/constants.py`
(dependency-free shared constants) and `features/build.py`
(`get_fighter_state_asof()`, a genuine point-in-time query built on the
trainer's own functions, not reimplemented) plus
`tests/test_train_serve_parity.py` — the phase's actual acceptance criterion.

Getting that test to pass surfaced three more real issues:

- **Own bug:** an off-by-one in the new truncation logic. Style-stat
  truncation used strict `<` where the trainer's own
  `merge_asof(direction='backward')` includes exact-date ties — silently
  dropping a fighter's own gold-dataset row for the target fight itself,
  undercounting their cumulative stats by one fight.
- **Imputation fires on more rows than its own flag suggests.**
  `_impute_by_weight_class()` fills every individually-missing style-stat
  cell with a weight-class median, not just rows flagged `style_missing` —
  that flag is only set from `R_SLpM` specifically. Rather than reproduce
  Phase 1's imputation logic inside a test meant to check something else,
  sample selection was adjusted to avoid the imputation path entirely.
- **Two genuine, pre-existing data-quality gaps**, unrelated to any code in
  this project: `career_fights_updated.csv` is sometimes missing a fighter's
  own row for a specific fight (a sync gap against `ufc-master.csv`), and
  sometimes double-logs the same fight under two spellings of an opponent's
  name on the same date — the documented "~2,235 duplicate rows" gotcha,
  caught concretely as "Zachary Reese" and "Zach Reese" being the same
  person. Samples hitting either issue are excluded from the test rather
  than papered over.

Duplicate `WC_ORDER`/`WOMENS_CLASSES`/layoff-bucket definitions across the
trainer and backend were consolidated into the new constants module —
confirmed behavior-identical via a full retrain producing byte-identical
metrics.

**Still true after this phase:** backend serves the original pre-remediation
model. `get_fighter_state_asof()` exists, is tested, and is called from
nowhere live. The cutover remains a distinct, future decision.

---

## Phase 6 — Repo hygiene

**Status: shipped.** A merge conflict, an untracked ten months of build
junk, and a README that hadn't caught up with anything above.

**Defect caught:** `README.md` had an unresolved merge conflict the whole
time. Flagged and deliberately left alone back in Phase 0. Phase 6's own
requirement to update the README forced the issue. One side was a two-line
placeholder; the other was the full, detailed README matching the actual
project — an unambiguous resolution, not a real judgment call, so made
directly.

Added `.gitignore` — deliberately narrower than the plan's literal "large
generated CSVs": a blanket `data/*.csv` ignore doesn't untrack anything
already tracked, it only silently hides genuinely new files from
`git status` later. A footgun, not a hygiene win, left out. `git rm -r
--cached` ran against every currently-tracked `__pycache__/`, `.DS_Store`,
and `catboost_info/` path — including one nested under `experiments/` that
the plan's own example command list didn't call out, found by a follow-up
sweep. Working tree verified intact throughout.

`docs/DATA_SOURCES.md` got a provenance/regeneration line per file —
including an honest "no producer script found in this repo" for
`ufc_gold_dataset_final.csv` rather than a plausible-sounding guess. The
README's Models section was rewritten to describe both the currently-serving
and not-yet-cut-over models honestly, rather than just bumping the feature
count as the plan's literal text asked for.

---

## Phase 7 — Elo upgrades

**Status: shipped.** Method-weighted K and layoff regression, gated on
walk-forward log loss — and a promotion that broke Phase 5's parity
guarantee until a real fighter's real layoff caught it.

K-sensitivity grid (24/32/40/48) via a lone-feature `elo_dif` logistic
regression across the same walk-forward folds: K=48 was already best in the
tested range — accuracy and log loss improved monotonically with K across
the whole grid, no evidence the current value is "too aggressive" as the
plan speculated. No change made there.

Method-weighted K (×1.25 for KO/TKO/Sub wins, ×0.75 for split decisions) and
layoff regression (regress toward 1500 after 365+ days inactive, checked at
15% and 25%) were each evaluated individually through the full 133-feature
model, since they change `elo_dif` throughout the whole feature set, not just
as a standalone signal. All three beat baseline individually. The winning
combination — method-weighted K plus 25% layoff regression — took pooled log
loss from 0.6200 to **0.6152**.

**Defect caught:** own bug, silently combining mutually-exclusive winners.
The first version of the combining logic blindly merged every individual
winner's parameters in iteration order. 15% and 25% layoff regression are two
different values of the *same* parameter, not combinable — the naive merge
let whichever ran last silently overwrite the other, so the script's own
printed label ("...15% + ...25%") overstated what had actually run. Caught by
noticing the label implied three combined changes but the result exactly
matched a two-way combination tested separately; fixed to keep the
better-performing value per colliding key.

**Defect caught:** promotion broke Phase 5's parity test — caught by a real
fighter's real layoff. Promoting the new Elo config to `build_dataset()`'s
defaults without also updating `features/build.py`'s independent Elo
computation would have silently broken the train/serve parity guarantee from
Phase 5. Caught immediately: an `R_elo` mismatch on Jon Jones's 2024
heavyweight comeback fight, exactly the kind of multi-year layoff the new
regression logic changes. Root cause: a shortcut in `_elo_asof()` — a
fighter's next fight's pre-fight rating normally equals their previous
fight's post-fight rating exactly — stopped holding once layoff regression
could intervene between the two. Fixed by replicating the same regression
check inside `_elo_asof()`, reading from the same shared constant the trainer
uses.

Glicko-2, the plan's explicitly-marked stretch goal, was skipped — the three
required items already produced a measurable win within the time available.

**Final result:**
- Pooled walk-forward accuracy: 65.64% → 66.08%
- Pooled log loss: 0.6200 → 0.6152
- Pooled Brier: 0.2156 → 0.2134
- Higher-Elo baseline (single-holdout): 55.21% → 59.69%

---

## Open items

Everything in `8si_remediation_plan.md` has now been addressed. What's left
is deliberate, not forgotten.

1. **`qa_SLpM`/`qa_SApM` formula bug.** Found in Phase 5 Part A. Fixing it
   changes trainer output and needs a retraining decision — deliberately not
   bundled into a live-service bug fix.
2. **Serving cutover to the remediated model.** Infrastructure is built and
   parity-tested (Phase 5 Part B). Actually pointing the live services at it
   is a distinct decision, made explicitly, not automatically.
3. **Git-LFS migration for large data/model files.** The plan marked this
   explicitly optional (Phase 6). Not attempted.
4. **Glicko-2.** Phase 7's stated stretch goal. Skipped after the required
   items already produced a measurable win.
