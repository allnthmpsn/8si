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

> **Current baseline for every Stage 2+ family's delta: 0.6134525**, not
> 0.6152. The 0.6152 figure above is Stage 0's starting point, kept for
> historical record. The fighter-identity data fix (documented under
> Stage 1.1 below, full writeup in `docs/DECISIONS.md`) moved pooled log
> loss to 0.6134525 *before* any Stage 2 feature work began — that
> improvement belongs to the identity fix, not to whichever family
> happens to be tested next. Every Stage 2 entry's "before" number must
> read 0.6134525 (or the prior family's own "after," once one ships) —
> never 0.6152 — or the identity fix's gain gets double-counted into a
> feature that didn't earn it.

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

## Stage 1 — Data ingestion

### 1.1 Round-level UFCStats data

Downloaded the Greco1899/scrape_ufc_stats pre-scraped CSVs (not scraped —
that project runs its own rate-limited, daily-automated scraper against
ufcstats.com and publishes plain CSVs, exactly per the spec's "download,
don't scrape" instruction) into `data/raw/ufcstats_rounds/`.
`training/ingest_rounds.py` parses `ufc_fight_stats.csv` (one row per
fight/round/fighter) into `data/round_stats.parquet`: knockdowns, sig
strikes landed/attempted by target (head/body/leg) and position
(distance/clinch/ground), total strikes, takedowns, sub attempts,
reversals, control time in seconds — 41,298 rows, 8,762 fights, 2,702
fighters, 1994–2026. Handles a real live-source edge case gracefully
(one event's fight_stats rows briefly outpacing its own event_details
row, since the two CSVs aren't refreshed atomically) by dropping and
warning rather than crashing.

`training/build_name_map.py` reconciles ufcstats.com's own fighter-name
spelling against this project's canonical names (`ufc-master.csv` ∪
`career_fights_updated.csv`): exact match, then rapidfuzz
(`token_sort_ratio`, threshold 92) for the remainder, with a
`manual_override` column for human-reviewed cases (22 applied — real
fighters ufcstats records under a nickname/ring name the canonical set
doesn't, e.g. Mirko Filipovic → Mirko Cro Cop, Antonio Rodrigo Nogueira →
Minotauro Nogueira — each verified against the *specific* real person,
not just a plausible-looking string match; see docs/DECISIONS.md for why
that distinction matters). **1.27% of 2015+ fights have an unmatched
fighter — passes the <2% bar.**

`training/scrape_rounds_update.py` is the "incremental updater" — but
deliberately re-downloads the source's already-fresh CSVs rather than
scraping ufcstats.com directly a second time, since Greco1899's own daily
job already does that scrape responsibly; running it ourselves too would
just double the load on the actual site for no benefit. Verified it
end-to-end: a live re-run picked up 12 new fights and handled a source
sync-lag gracefully.

**Side discovery, fixed as its own checkpoint:** cross-referencing
`round_stats.parquet` against `ufc-master.csv` surfaced a serious,
pre-existing, unrelated bug in `career_fights_updated.csv` — 69 fighter
names (Cris Cyborg, Rampage Jackson, and other prominent fighters among
them) had 100% fabricated fight histories, and 24 more pairs were real
fighters split across two name spellings. Full writeup in
`docs/DECISIONS.md`. Fixing it **improved** pooled walk-forward log loss
0.6152062 → 0.6134525 (confirming it was restoring real signal, not just
cleanup) and raised round_stats coverage from 95.96% to 97.24%.

**Acceptance:** round_stats.parquet coverage of 2015+ `ufc-master.csv`
fights is **97.24%**, short of the spec's 98% target. The residual gap is
fighters genuinely absent from ufcstats.com's own scrape (e.g. Katlyn
Chookagian: zero rows under any spelling in round_stats.parquet at all —
confirmed this is a source data gap, not a reconciliation failure) —
not fixable from this side without a deeper investigation into the
external source itself. Documented and accepted short of the bar rather
than silently claimed.

### 1.2 Odds pipeline

Historical backfill coverage: **95.26%** of 2015+ `ufc-master.csv` fights
have opening-line odds — comfortably above the spec's 85% bar, so per the
spec's own rule, no BestFightOdds scraper was built (ToS-gray, fragile,
and unnecessary given this coverage).

`training/collect_odds.py`: forward snapshot collector, meant to run on a
schedule (cron/manual per card). Appends to the *existing*
`data/odds_snapshots.json` (list of `{timestamp, fights: [{f1, f2,
f1_price, f2_price}]}`) rather than the spec's literal
`data/odds_snapshots/` directory suggestion — that file already has two
established consumers (`training/backfill_clv.py` reads it,
`backend/main.py`'s `/odds` endpoint writes it), so forking the format
would fragment it for no benefit. Requires `ODDS_API_KEY` as an
environment variable.

**Live-tested 2026-07-26**: one real snapshot captured (16 fights, e.g.
Aleksandar Rakic/Marcin Tybura, Ian Garry/Islam Makhachev), confirmed
correctly appended to `data/odds_snapshots.json` with the right schema
including `commence_time`. `backfill_clv.py`'s `_earliest_snapshot_prices
()` is already per-fight-pair scoped (keyed by matchup, not "first entry
in the whole file"), so this collector can run indefinitely across many
cards without the open/close proxies degrading over time.

**Cadence, now settled**: run once when a card's lines first appear
(open proxy — whichever snapshot is earliest for a given matchup is what
`backfill_clv.py` already picks up automatically) and once more as close
to event start as reliably achievable (close proxy — consistency in
*when* this second run happens matters more than getting arbitrarily
close to first pitch). No third mode or special "final" flag needed;
running it more often near fight night is sufficient on its own.

**Also found and fixed while investigating this:** `backend/main.py` had
a live Odds API key hardcoded in source (predates this session), already
pushed to the public repo in an earlier commit this session. Per explicit
user direction: the key is free-tier/no billing risk, so left as-is
rather than rotated; git history is not being rewritten for a dead
credential. Code now reads `ODDS_API_KEY` from the environment with the
old literal as a fallback, so it no longer regresses.

### 1.3 Leakage tests for new data

`tests/test_no_leakage.py` gained a reusable harness
(`assert_round_feature_no_leakage()`) for any Stage 2 feature function
satisfying the contract `feature_fn(round_stats_df) -> DataFrame[fighter,
date, ...]`, one row per fight, shift(1) discipline — mirrors
`test_style_stats_no_leakage`'s truncate-and-recompute strategy exactly.
Exercised now with a proof-of-concept function (cumulative knockdowns
landed) since no real Stage 2 feature exists yet; 15 parametrized cases,
all passing, ready for Stage 2's real feature functions to reuse
directly.

**Acceptance:** round_stats.parquet coverage 97.24% (target 98%, gap
explained and documented above); name-map unmatched 1.27% (target <2%,
pass); forward odds collection built, unit-tested, and live-tested
against a real card (one real snapshot captured and verified); leakage
harness extended and passing (104/104 tests total). All work committed
in three checkpoints: the fighter-identity data fix, Stage 1's own
deliverables, and this live-verification snapshot.

## Stage 2 — Feature build

Baseline for every family's delta below: **0.6134525** (post
fighter-identity-fix, not the 0.6152 Stage-0-start figure — see the
callout under "Market benchmark" above).

### 2.1 Knockdowns & damage — DROPPED

`training/features_kd.py`: `kd_per15_for`, `kd_per15_against`,
`kd_absorbed_per_sig_str`, `damage_ratio` — as-of, shift(1), computed
from `data/round_stats.parquet` (round-level rows summed to fight-level,
then the same per-stat joint numerator/denominator observed-mask design
`training/style_stats.py` uses). `tests/test_no_leakage.py::
test_kd_features_no_leakage` (8 cases) verifies no leakage via the same
truncate-and-recompute strategy as every other as-of feature in this
codebase, adapted for this family's extra dependencies (opponent stats
via a same-fight self-join, `ufc_fight_results.csv` for fight duration,
`data/name_map.csv` for the canonical-name join).

**A real bug caught before evaluating anything**: the first pass omitted
the `_csd_{feat} > 0` gate `style_stats.py`'s own fix already established
— the exact same class of bug, reintroduced in new code by not copying
the earlier fix. `damage_ratio` blew up to a mean of 43 million (max 48
billion) whenever a fighter's cumulative opponent-sig-strikes-landed
denominator was genuinely (not missing, just actually) zero — division
by an EPS-clipped near-zero instead of the correct `NaN`. Fixed
identically to the earlier style_stats.py fix. Re-verified
`kd_per15_for`'s own remaining outliers (max 128.57) separately — that
one's real, not a bug: `R_SLpM` (the existing, already-shipped feature)
has the identical small-sample-tiny-denominator pattern (max 56 against
a mean of 3.9), from very short early-career fights. Consistent with an
existing, accepted characteristic of per-time-rate features in this
codebase, not something specific to this new family.

`experiments/kd_v2/run_experiment.py` (same behind-a-flag pattern as
`experiments/elo_v2/`) tested two variants via `training/train_model1.py`'s
own `build_dataset()`/training loop, unmodified:

| Variant | Pooled log loss | Δ vs. baseline |
|---|---|---|
| Baseline (current FEAT_114, no KD) | 0.6135 | — (sanity-check match to 0.6134525) |
| + KD family (12 new columns) | 0.6172 | **+0.0038** |
| + KD family, retire `got_finished_rate`/`finish_danger_mismatch` | 0.6164 | **+0.0029** |

**Neither variant beats baseline — dropped, per the spec's own rule with
no exception.** The spec calls this family "highest expected value"
because it was the top SHAP family in someone else's model; that isn't
evidence for this one, and isn't grounds for a discretionary pass — the
pooled number is the only thing that decides, and it says no. Retiring
`got_finished_rate`/`finish_danger_mismatch` alongside KD is closer to
breakeven than adding KD on top of the full existing set, meaning those
two retired features were carrying at least as much signal as KD adds —
another data point against the family, not for it. `training/train_model1
.py`'s `FEAT_114` is unchanged. `training/features_kd.py` and
`experiments/kd_v2/run_experiment.py` stay in the repo (code kept,
shipped disabled) as a record and in case a future family (e.g. RAPM)
changes the calculus by interacting with it.

### 2.2 Control & grappling exposure — DROPPED

`training/features_grappling.py`: `ctrl_pct_for`, `ctrl_pct_against`,
`ground_share_landed`, `td_landed_per15`, `td_absorbed_per15` — same
architecture as 2.1 (fight-level aggregation, opponent self-join, joint
observed-mask design, `_csd>0` gate applied correctly from the start this
time). One scoping note: the spec's `ground_time_share` is approximated
as `ground_share_landed` (share of a fighter's own landed significant
strikes thrown from ground position) — ufcstats.com has no direct
ground-TIME stat, only ground-STRIKE counts and an overall control-time
figure that mixes clinch and ground control together, so literal ground
*time* isn't available; documented as an approximation rather than
silently presented as the real thing. Verified `distance_landed +
clinch_landed + ground_landed == sig_str_landed` for every 2015+ row
before relying on it. Leakage-tested (8 cases,
`test_grappling_features_no_leakage`, reusing the same truncation helper
2.1's test introduced).

`experiments/grappling_v2/run_experiment.py` tested two variants against
the 0.6134525 baseline:

| Variant | Pooled log loss | Δ vs. baseline |
|---|---|---|
| Baseline (current FEAT_114) | 0.6135 | — (sanity-check match) |
| + grappling family (10 new columns) | 0.6144 | **+0.0009** |
| + grappling family, retire `TD_Avg`/`TD_Def` lineage | 0.6162 | **+0.0027** |

**Neither beats baseline — dropped.** The plain add is close to
breakeven (+0.0009, well under the spec's 0.002 "matters" threshold used
elsewhere) but still doesn't clear the bar, and retiring the existing
`TD_Avg`/`TD_Def` style-stat features in favor of the round-derived
takedown rates makes things worse, not better — the older snapshot-era
features aren't strictly dominated here, contrary to what the spec
anticipated for this family. `FEAT_114` unchanged; code kept, shipped
disabled.

### 2.3 Cardio / late-round profile — DROPPED

`training/features_cardio.py`: `r3_output_ratio`, `late_round_finish_rate`,
`r1_finish_rate`. Two sources, not one: round-level strike volume
(`data/round_stats.parquet`, for the round-1-vs-round-3+ output
comparison) and win/method/finish-round (`data/raw/ufcstats_rounds/
ufc_fight_results.csv`, parsed via `BOUT`'s " vs. " delimiter — verified
zero non-matching rows before relying on it — since `round_stats.parquet`
has no notion of how a fight ended). "Finish" reuses
`compute_career_stats()`'s own `KO|TKO` / `Sub` string-match definition
rather than inventing a second one. `r3_output_ratio` honors the spec's
own `>=3` prior round-3-reaching-fights minimum-sample guard. Leakage-
tested (8 cases) — this family needed a *second* truncation helper
(`_truncated_raw_dir`) since it depends on `ufc_fight_results.csv` too,
not just `round_stats.parquet`.

`experiments/cardio_v2/run_experiment.py` — no existing feature to test
retiring here (unlike 2.1/2.2, this family has no snapshot-era
predecessor in `FEAT_114`), so one variant:

| Variant | Pooled log loss | Δ vs. baseline |
|---|---|---|
| Baseline (current FEAT_114) | 0.6135 | — |
| + cardio family (9 new columns) | 0.6162 | **+0.0028** |

**Does not beat baseline — dropped.** Third consecutive round-derived
family to fail the gate (2.1 +0.0038, 2.2 +0.0009, 2.3 +0.0028) — each
individually leakage-tested, bug-checked (2.1's `_csd>0` catch applied
correctly to 2.2 and 2.3 from the start), and value-distribution-sanity-
checked before evaluation, so this reads as a real pattern (round-level
volume/timing signal not adding much on top of what career-level Elo/
win-rate/style-stat features already capture at this fight count), not
three repeats of the same bug. `FEAT_114` unchanged; code kept, shipped
disabled.

### 2.4 Style vectors (position/target mix) — DROPPED; interactions not built

`training/features_style_vectors.py`: `dist_share`, `clinch_share`,
`ground_share` (position mix) and `head_share`, `body_share`, `leg_share`
(target mix) — career as-of shares of a fighter's own landed significant
strikes. Simplest family so far: every input column already lives in
`round_stats.parquet`, no second source needed. Both partitions verified
exact (`distance+clinch+ground == sig_str_landed`, `head+body+leg ==
sig_str_landed`, every 2015+ row) before relying on them. Leakage-tested
(8 cases).

`experiments/style_v2/run_experiment.py` — one variant (no existing
feature to test retiring; this is new signal, nothing like it is in
`FEAT_114` today):

| Variant | Pooled log loss | Δ vs. baseline |
|---|---|---|
| Baseline (current FEAT_114) | 0.6135 | — |
| + style vector family (18 new columns) | 0.6150 | **+0.0016** |

**Does not beat baseline — dropped.** Fourth consecutive round-derived
family to fail the gate. The spec's matchup-interaction sub-bullet (own
`leg_share` × opp stance; own `ground_share` × opp `ctrl_pct_against`;
own `td_landed_per15` × opp `td_absorbed_per15`) was explicitly gated
"only after mains are in" — the mains didn't clear the bar, and two of
the three named interactions are built on Stage 2.2's `ctrl_pct_against`/
`td_landed_per15`, which already failed independently — so the
interactions were not built at all rather than tested and dropped.
`FEAT_114` unchanged; code kept, shipped disabled.

### 2.5 Situational & priors — DROPPED; travel proxy not built

`training/features_situational.py`: `age_x_division` (row-level, no
leakage risk — same category as the existing `age_x_layoff`),
`five_round_bout`, `catchweight` (both row-level), and `division_change`
(the one genuinely history-dependent piece — shift(1) on a fighter's own
`ufc-master.csv` weight-class timeline, no `round_stats.parquet`
dependency at all, unlike every other Stage 2 family so far). Leakage-
tested only where there's leakage risk to test (`division_change`, 8
cases) — the three row-level features can't leak by construction (both
inputs are properties of the fight itself, known at prediction time).

**Travel proxy not built**: the spec wants fighter-country vs.
event-country. `ufc-master.csv` has event `country`; nothing in this
codebase carries fighter *nationality* — checked
`data/ufc_fighters_final_updated.csv` and ufcstats.com's own
`ufc_fighter_details.csv`/`ufc_fighter_tott.csv` directly, none has it.
Approximating a fighter's home country from their own fight-location
history would be circular (most fights are wherever the promotion
scheduled them), so this was left out rather than built on a fabricated
signal.

`experiments/situational_v2/run_experiment.py` — one variant (no
existing feature to retire):

| Variant | Pooled log loss | Δ vs. baseline |
|---|---|---|
| Baseline (current FEAT_114) | 0.6135 | — |
| + situational family (7 new columns) | 0.6145 | **+0.0010** |

**Does not beat baseline — dropped.** Fifth consecutive family to fail
the gate. `FEAT_114` unchanged; code kept, shipped disabled.

### 2.6 RAPM (opponent-adjusted ridge regression) — SHIPPED DISABLED

`training/rapm.py`: for each walk-forward fold's `[train_start,
train_cutoff)` window, a ridge regression (`RidgeCV`, alpha grid
1–300) across every fight in the window at once — a sparse
`2×n_fighters`-column design matrix (offense + defense per fighter),
+1/-1 per fight-row, response = the fighter's own per-minute output
(significant strikes for `rapm_off`/`rapm_def`/`rapm_net`; takedowns for
`rapm_grap_off`/`rapm_grap_def`/`rapm_grap_net`). Structurally different
from every other Stage 2 family: not a per-fighter as-of timeline, a
single fold-level fit, refit fresh per fold, never seeing a fight from
the test period or later (leakage-tested by proving
`fit_rapm(train_cutoff=X)` on the full data matches a fit on data
pre-truncated to `date < X`, rather than the per-row truncate-and-compare
every other family's test uses — there's no single per-row value to
truncate-and-compare against here).

**Two real bugs caught before trusting any output:**
1. The first formulation used each fighter's *differential* (own minus
   opponent) as the ridge response. That makes the two rows any single
   fight contributes (one per fighter's perspective) exact negatives of
   each other — a perfect antisymmetry that ridge's L2 penalty (symmetric
   in offense/defense) resolves by forcing `rapm_off == rapm_def` for
   every fighter, silently. Caught empirically (every fighter's off/def
   were identical to 6 decimal places — an unmistakable tell, not a
   subtle one) before it reached the experiment stage. Fixed by using
   each fighter's own *raw* per-minute rate as the response instead (the
   standard APM formulation) — offense and defense separated correctly
   after the fix; `rapm_net` (their sum) was unaffected by the bug either
   way, since the degenerate split still summed to the same total.
2. The experiment script's merge-then-impute step passed a stale-index
   boolean mask into `_impute_by_weight_class()` after `df.merge()` reset
   the index — `pandas.errors.IndexingError`, not a silent bug, but worth
   noting since it blocked the first corrected run.

**A real methodology gap, also fixed before evaluating**: the initial
merge used a blind `fillna(0.0)` for fighters absent from a fold's RAPM
fit (no fights before that fold's `train_cutoff`) instead of this
project's established weight-class-median imputation. Measured directly:
missingness is 10x higher in the test period (8.1%) than the train
period (0.8%) for the obvious reason (every train-period fighter fought
at least once before `train_cutoff` by construction; test-period debuts
often haven't). Fixed to match the pattern every other Stage 2 family
already uses.

`experiments/rapm_v2/run_experiment.py`, after both fixes:

| Variant | Pooled log loss | Δ vs. baseline |
|---|---|---|
| Baseline (current FEAT_114) | 0.6135 | — |
| + RAPM (striking + grappling, 24 new columns) | 0.6479 | **+0.0345** |

**A large, genuine regression — not a leftover bug.** Re-verified after
fixing both the offense/defense degeneracy and the imputation gap; the
regression persisted at essentially the same magnitude (+0.0330 before
the imputation fix, +0.0345 after), which is what settled it as real
rather than an artifact still being chased. Plausible cause, not proven:
many fighters in a 5-6-year training window have only 1-3 fights, giving
high-variance individual RAPM estimates despite regularization — noise
that may be actively confusing the LR+XGB blend rather than adding
signal, especially given Elo and the QA stats already provide some
opponent-adjustment along a related axis.

Per the spec's own explicit rule for this family (unlike 2.1-2.5's plain
drop-on-failure): keep the code, ship it disabled, log it here rather
than delete it — RAPM is the spec's "most novel component" and the
magnitude here, while much larger than the spec's own <0.002 threshold
anticipated, doesn't change that the code itself is correct (both real
bugs are fixed, the leakage test is genuine and passing) and may be
worth revisiting with different regularization or a minimum-fights
floor before writing off opponent-adjustment entirely. `FEAT_114`
unchanged.

## Stage 2 summary

All six families attempted; none shipped. 2.1-2.5 failed the plain
pooled-log-loss gate by small margins (+0.0009 to +0.0038); 2.6 (RAPM)
failed by a much larger margin (+0.0345) but is kept disabled per its
own explicit spec rule rather than dropped. Pooled walk-forward log loss
remains **0.6134525** — unchanged from the post-fighter-identity-fix
baseline going into Stage 2, since nothing here cleared the bar. Every
family was independently leakage-tested and bug-checked before its
result was trusted; two real implementation bugs were caught and fixed
along the way (`training/style_stats.py`'s `_csd>0` gate reused
correctly from 2.1 onward, and RAPM's offense/defense degeneracy) —
this stage's five-and-a-half failures are a genuine result of this
discipline being followed consistently, not five bugs that would have
looked like wins if left unchecked.

## Stage 3 — Model consolidation

### 3.1a Pooled men's+women's model — NOT PROMOTED

`training/train_model1.py`'s `build_dataset()` gained an
`include_womens` parameter (default `False`, current production
behavior unchanged — same safe-default pattern as `elo_kwargs`).
Elo/career-stats/QA-stats were never actually computed men's-only (only
the final training-row filter excluded women's fights), so flipping this
flag is low-risk: women's fights get their already-correct upstream
histories, no separate code path needed.

`experiments/pooled_v2/run_experiment.py` trained men's+women's pooled
vs. the current men's-only baseline, **scoring both on the exact same
men's-only test fights** (apples-to-apples per the spec's own framing —
pooling adds training data, the question is whether it helps or hurts
predictions on the fights the current model already makes), with
`is_womens`, `is_womens_x_elo_dif`, `is_womens_x_age_dif` added for the
pooled variant:

| Variant | Men's pooled log loss | Δ vs. baseline |
|---|---|---|
| Men's-only baseline | 0.6135 | — |
| Pooled (scored on men's subset) | 0.6146 | **+0.0011** |

**Misses the spec's own ≤0.001 degradation bar** — by one ten-thousandth,
essentially within noise of the boundary, and not what the spec
anticipated ("pooling wins or ties"). Applying the gate on its number,
not its closeness to the number: **not promoted**, `include_womens`
stays `False` in production. Women's-fight log loss on the pooled model
was 0.6406 (n=424, informational — not a gate), for reference if this is
revisited.

### 3.1b Blend re-tune + monotonic constraint — PROMOTED

`experiments/retune_v2/run_experiment.py`: Optuna (TPE sampler, seed 42,
25 trials) searching `lr_weight`, `lr_c`, and 9 XGB hyperparameters
(`n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`,
`subsample`, `colsample_bytree`, `gamma`, `reg_alpha`, `reg_lambda`),
objective = pooled walk-forward log loss across the same 5 time-ordered
folds every other v2 experiment uses (never shuffled CV, per the spec).
Fold datasets cached across trials (only the model hyperparameters
change between trials, not the feature-engineering pipeline) so 25
trials × 5 folds stayed fast. An XGBoost `monotone_constraints` on
`elo_dif` (+1: higher Elo differential can never predict a *lower*
P(Red wins)) was applied unconditionally in every trial — a correctness
constraint, not something to search over.

Two sanity checks run before trusting the search: the constraint alone
(old hyperparameters, monotonic elo_dif) was ~neutral (0.6135 → 0.6137,
-0.0002) — confirming the eventual gain comes from the retuned
hyperparameters, not the constraint riding along; the constraint's own
regularizing effect (shallower, more conservative trees dominate the
found optimum — `max_depth` 6 → 3) does track with genuinely better
generalization, not just a fluke of this particular 25-trial draw.

| Variant | Pooled log loss | Δ vs. baseline |
|---|---|---|
| Baseline (0.6134525) | 0.6135 | — |
| Current params + monotonic constraint only | 0.6137 | +0.0002 (constraint alone, ~neutral) |
| Optuna best (25 trials) | 0.6099 | **-0.0036** |
| **Re-verified via `training/walk_forward.py` directly** | **0.6095** | **-0.0040** |

**The first change in this entire v2 pass to clear the gate outright**,
not by exemption. Promoted into `training/train_model1.py`: `LR_WEIGHT`
0.70→0.5247, `XGB_WEIGHT` 0.30→0.4753, a new named `LR_C` constant
(0.00711→0.04123, replacing three separate hardcoded literals in
`train_model1.py`'s two `LogisticRegression` call sites and
`walk_forward.py`'s own copy — a real, if minor, duplication risk fixed
along the way), `XGB_PARAMS` retuned, and `XGB_MONOTONE_CONSTRAINTS`
added as its own named tuple. Retrained: single-holdout accuracy
70.00%→70.21%, log loss 0.5883→0.5852. All 145 tests pass. Serving
paths and live processes confirmed untouched (writes to `model/v2/`
only, per the established convention since Phase 2).

## Next

3.2 (retire Model 2A/2B/CONFIRM_DOG-FAV taxonomy/women's fork from the
serving path) is next.
