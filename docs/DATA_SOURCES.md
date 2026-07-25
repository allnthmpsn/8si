# Data Sources

## data/ufc-master.csv
**Role:** Primary fight-level dataset. One row per fight (both corners).

**Key columns:**
- `date` — fight date (ISO format). Used as the merge key for all asof joins.
- `R_fighter` / `B_fighter` — canonical fighter names (must match other files exactly).
- `Winner` — `"Red"`, `"Blue"`, or draw string. Draw rows are dropped before training.
- `R_wins`, `R_losses`, `R_current_win_streak`, `R_longest_win_streak` — pre-fight cumulative stats supplied by the data provider (not re-derived). These represent what was known going INTO the fight.
- `R_avg_SIG_STR_landed`, `R_avg_TD_landed`, `R_avg_SIG_STR_pct`, `R_avg_SUB_ATT`, `R_avg_TD_pct` — rolling career averages up to that fight (provider-computed, pre-fight).
- `R_Height_cms`, `R_Reach_cms`, `R_age`, `R_Stance` — physical attributes at fight time.
- `weight_class` — string weight class name mapped to ordinal in training.
- `title_bout` — boolean-like field, coerced to int (0/1).

**Gotchas:**
- Some older rows have `--` or empty strings in numeric columns. Use `pd.to_numeric(..., errors='coerce').fillna(0)` throughout.
- Draws and NC rows have `Winner` values other than `"Red"`/`"Blue"` — always filter to those two.
- The file contains historical fights back to 1993. Training is restricted to 2018+ to ensure dense career stat coverage.

**Provenance / regeneration:** Base fight-history dataset (originally
sourced from a public UFC-stats Kaggle-style export) plus locally-added
odds columns (`R_odds`/`B_odds`, `r_dec_odds`/etc.). No in-repo script
regenerates this from scratch — treat it as the primary source-of-truth
input, not a derived/generated file.

---

## data/career_fights_updated.csv
**Role:** Per-fighter fight log, one row per fighter per fight. Used to compute shift(1) career stats with no leakage.

**Key columns:**
- `fighter` — fighter name (must match ufc-master.csv).
- `date` — fight date.
- `won` — 1 if the fighter won, 0 if lost.
- `method` — finish method string (e.g., "KO/TKO", "Submission", "Decision").
- `opponent` — opponent name (used for opp_quality lookback).

**Computed from this file (in training):**
`cum_fights`, `career_win_rate`, `ko_finish_rate`, `sub_finish_rate`, `last3_win_rate`, `last10_win_rate`, `last5_won`, `last5_finish_rate`, `trend_score`, `layoff_days`, `opp_quality`.

**Gotchas:**
- Contains ~50k rows including non-UFC regional fights (3.5× the UFC fight count). This is intentional — it gives accurate pre-UFC career win rates.
- ~2,235 rows share a (fighter, date) combination (fighter fought twice on same day in a regional promotion). `merge_asof` handles this correctly by selecting the last matching row.
- Stats are computed with `shift(1)` inside each fighter group, so row 0 for each fighter always has 0 prior fights. Debut rows are filled with neutral defaults (`career_win_rate=0.5`, streaks=0, etc.).

**Provenance / regeneration:** Built from `data/career_fights.csv` (the
Sherdog-derived base log, rebuilt by `experiments/archive/sherdog_fix.py`)
patched with UFC Stats data via `experiments/archive/ufc_stats_scraper.py`
(step 4 of that script: "Patch career_fights.csv → career_fights_updated.csv").
Both scripts are archival/one-off — not part of the production trainer
path, and not re-run automatically. Re-run only if onboarding new fighters
or fixing a data-quality issue; both scripts document themselves as
resumable and non-destructive to their inputs.

---

## data/ufc_fighters_final_updated.csv
**Role (as of 8SI Phase 1):** No longer used by `training/train_model1.py` —
its style-stat columns were a career-to-date snapshot with no date
dimension, so every historical training fight saw the fighter's future
averages (see docs/REBASELINE.md, Phase 1). Style stats are now computed
as-of by `training/style_stats.py` from `ufc_gold_dataset_final.csv`
instead. This file is still read by `backend/main.py` and the women's
trainer (`train_model1_womens.py`), which have not been migrated yet —
Phase 5 of `8si_remediation_plan.md` (unify train/serve pipeline) is where
that's addressed.

**Key columns:**
- `Fighter_Name` — canonical name. Must match ufc-master.csv fighter names.
- `SLpM`, `SApM` — strikes landed/absorbed per minute.
- `Str_Acc`, `Str_Def`, `TD_Acc`, `TD_Def` — stored as `"46%"` strings. Strip `%` and divide by 100 at load time.
- `TD_Avg`, `Sub_Avg` — takedowns per 15 min, submission attempts per 15 min.
- `Height`, `Reach` — stored in cm (numeric). May contain `--` for unknown values (use `_safe_float()` in backend).
- `DOB` — date of birth string.
- `Wins`, `Losses` — total MMA record (manually curated, overrides scraped sherdog values in backend).

**Gotchas:**
- Six fighter names have duplicate rows (two different fighters sharing the same name, or a stale earlier entry): Mike Davis, Joey Gomez, Tony Johnson, Michael McDonald, Jean Silva, Bruno Silva. Always `drop_duplicates(subset=['Fighter_Name'], keep='last')` before merging. The later row has the more complete/correct stats.
- `--` appears in Height/Reach for some fighters (e.g., Ben Johnston). The backend uses `_safe_float()` to handle this; training uses `pd.to_numeric(..., errors='coerce').fillna(0)`.
- Percentage columns (`Str_Acc`, etc.) stored as `"46%"` — a plain `float()` call will raise `ValueError`. Always strip `%` first.

**Provenance / regeneration:** Built from `data/ufc_fighters_final.csv`,
updated by `experiments/archive/ufc_stats_scraper.py` (step 6: "Update
ufc_fighters_final.csv → ufc_fighters_final_updated.csv") via UFC Stats
scraping. Archival script, not part of the production trainer path.

---

## data/ufc_gold_dataset_final.csv
**Role (added 8SI Phase 1):** Per-fight strike/takedown/submission totals
for BOTH corners. One row per fight, `F1_*`/`F2_*` column pairs. Used by
`training/style_stats.compute_style_stats_asof()` to build the as-of style
stats (`SLpM`, `SApM`, `Str_Acc`, `Str_Def`, `TD_Avg`, `TD_Acc`, `TD_Def`,
`Sub_Avg`) that replaced the `ufc_fighters_final_updated.csv` snapshot
merge. This is the only file in the repo with both fighters' attempt counts
per fight, which is required to compute defensive rates (`Str_Def`,
`TD_Def` need to know what the OPPONENT attempted against this fighter).

**Key columns:**
- `Fighter_1`, `Fighter_2` — fighter names. ~94.7% overlap with
  `career_fights_updated.csv` fighter names (checked directly); the
  remainder are name-spelling mismatches (e.g. typos, alternate
  romanizations) rather than a systematic gap. Fighters with no match get
  `R_style_missing`/`B_style_missing` = 1 and are median-imputed — this is
  treated as ordinary missingness, not chased down further.
- `Event_Date` — fight date.
- `F1_Sig_Landed`, `F1_Sig_Att`, `F2_Sig_Landed`, `F2_Sig_Att` — significant strikes.
- `F1_TD_Landed`, `F1_TD_Att`, `F2_TD_Landed`, `F2_TD_Att` — takedowns.
- `F1_Sub_Att`, `F2_Sub_Att` — submission attempts.
- `Total_Fight_Time_Sec` — used to normalize per-minute / per-15-min rates.

**Gotchas:**
- 8,551 fights, 1994-03-11 to 2026-03-07, no nulls in any of the stat columns checked.
- `compute_style_stats_asof()` converts this wide (one row per fight) shape
  to long (one row per fighter per fight) internally before computing
  cumulative shift(1) rates — see the module docstring in
  `training/style_stats.py`.

**Provenance / regeneration:** No producer script found in this repo (only
consumers: `train_model1.py`, `style_stats.py`, and
`experiments/archive/model_trainer.ipynb` reference it) — flagging as a
real gap rather than guessing. If this file needs to be regenerated or
extended with new fights, that process isn't currently documented or
scripted anywhere in-repo.

---

## data/ufc-master.csv — diff columns used for features

Several diff features (`win_dif`, `height_dif`, `reach_dif`, etc.) are computed directly in training from the Red/Blue column pairs in ufc-master.csv. See FEATURE_REFERENCE.md for the full list.

---

## data/elo_ratings_history.csv (generated)
**Role:** Per-fight Elo snapshots. One row per fighter per fight.

**Key columns:** `fighter`, `opponent`, `date`, `elo_before`, `elo_after`, `result`, `elo_trend`.

`elo_trend = elo_before - elo_before.shift(3)` per fighter (3-fight Elo momentum).

**Provenance / regeneration:** Fully regenerated by
`python training/train_model1.py`. Note the default `--out-dir` is
`model/v2/`, not `data/` — pass `--out-dir data` explicitly to overwrite
the tracked copy in `data/`. Also regenerated internally by
`training/walk_forward.py` per fold (not persisted outside that script).
Purely computed from `data/ufc-master.csv` — no manual curation.

---

## data/elo_current.csv (generated)
**Role:** Current (post-last-fight) Elo for every fighter. Used by the backend for upcoming-fight predictions.

**Key columns:** `fighter`, `current_elo`, `last_fight_date`, `total_fights`.

**Provenance / regeneration:** Same as `elo_ratings_history.csv` above —
generated alongside it by the trainer.

---

## data/upcoming.csv
**Role:** Upcoming card data fetched and stored by the backend. Not used in training.

**Provenance / regeneration:** Written by `backend/main.py` at request
time (not a static input file) — check `backend/main.py` for the exact
write path if this needs to be regenerated manually.

---

## data/odds_snapshots.json
**Role:** Betting odds history used by Model 2 (odds-aware LR). Not used in Model 1 training.

**Provenance / regeneration:** Appended to by `backend/main.py`'s
`GET /odds` endpoint (`_ODDS_SNAPSHOT_FILE`), which polls a live odds API
(`the-odds-api.com`) and fuzzy-matches against a hardcoded upcoming-card
fight list. Not regenerable from historical data — it's a live-polling log,
and current coverage is limited to whatever cards that endpoint was polled
for (see `docs/DECISIONS.md` "8SI Phase 4" CLV section for current
coverage: 13 matchups, one card).

---

## Other files in data/ (archival, not part of the production pipeline)

`career_fights.csv` (pre-patch base for `career_fights_updated.csv`),
`sherdog_records*.pkl` (`sherdog_records.pkl` →
`sherdog_records_fixed.pkl` → `sherdog_records_patched.pkl`, each a stage
in `experiments/archive/sherdog_fix.py`'s cleanup pipeline),
`bad_sherdog_matches.csv` / `still_unfixed.csv` / `sherdog_fix_log.txt`
(byproducts of that same cleanup run), `ufc_fighters_final.csv` /
`ufc_fighters_scraped.csv` (pre-patch stages for
`ufc_fighters_final_updated.csv`), `wiki_fighter_records.pkl`,
`active_fighters_status.csv`, `mega_fights.csv`, `ufc_stats_fights.csv`
(fighter/opponent/date/result/method only — no strike/TD numeric columns,
see `training/style_stats.py`'s module docstring for why it's not used for
style stats despite the name), `ufc_training_data.csv` — none of these are
read by `training/train_model1.py`, `training/walk_forward.py`, or
`backend/main.py`'s Model 1 path as of this writing. Left in the working
tree as intermediate artifacts from earlier data-cleaning sprints; not
individually documented here beyond this note. If one turns out to still
be a live dependency somewhere, move its documentation up to its own
section rather than trusting this blanket note.
