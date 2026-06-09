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

---

## data/ufc_fighters_final_updated.csv
**Role:** Fighter-level style stats from UFC Stats. One row per fighter name (after deduplication).

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

---

## data/ufc-master.csv — diff columns used for features

Several diff features (`win_dif`, `height_dif`, `reach_dif`, etc.) are computed directly in training from the Red/Blue column pairs in ufc-master.csv. See FEATURE_REFERENCE.md for the full list.

---

## data/elo_ratings_history.csv (generated)
**Role:** Per-fight Elo snapshots. One row per fighter per fight.

**Key columns:** `fighter`, `opponent`, `date`, `elo_before`, `elo_after`, `result`, `elo_trend`.

`elo_trend = elo_before - elo_before.shift(3)` per fighter (3-fight Elo momentum).

---

## data/elo_current.csv (generated)
**Role:** Current (post-last-fight) Elo for every fighter. Used by the backend for upcoming-fight predictions.

**Key columns:** `fighter`, `current_elo`, `last_fight_date`, `total_fights`.

---

## data/upcoming.csv
**Role:** Upcoming card data fetched and stored by the backend. Not used in training.

---

## data/odds_snapshots.json
**Role:** Betting odds history used by Model 2 (odds-aware LR). Not used in Model 1 training.
