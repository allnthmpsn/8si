# Feature Reference — Model 1 (114 features)

All features are numeric. Diffs are always `R_value - B_value` (positive = Red advantage).
Corner-flip augmentation negates all `_dif` columns and swaps R_/B_ pairs, so the model is corner-invariant.

---

## Group 1 — Raw fight-record stats from ufc-master.csv (28 features)

These are pre-fight cumulative values supplied by the data provider. Each represents what was known going INTO the fight.

| Feature | Source column | Description |
|---|---|---|
| `R_wins` | `R_wins` | Red fighter's total UFC wins before this fight |
| `R_losses` | `R_losses` | Red fighter's total UFC losses before this fight |
| `B_wins` | `B_wins` | Blue fighter's total UFC wins |
| `B_losses` | `B_losses` | Blue fighter's total UFC losses |
| `R_Height_cms` | `R_Height_cms` | Height in cm |
| `B_Height_cms` | `B_Height_cms` | Height in cm |
| `R_Reach_cms` | `R_Reach_cms` | Reach in cm |
| `B_Reach_cms` | `B_Reach_cms` | Reach in cm |
| `R_age` | `R_age` | Age at fight time |
| `B_age` | `B_age` | Age at fight time |
| `R_avg_SIG_STR_landed` | `R_avg_SIG_STR_landed` | Career avg significant strikes landed |
| `B_avg_SIG_STR_landed` | `B_avg_SIG_STR_landed` | |
| `R_avg_TD_landed` | `R_avg_TD_landed` | Career avg takedowns landed |
| `B_avg_TD_landed` | `B_avg_TD_landed` | |
| `R_current_win_streak` | `R_current_win_streak` | Win streak entering fight |
| `B_current_win_streak` | `B_current_win_streak` | |
| `R_current_lose_streak` | `R_current_lose_streak` | Loss streak entering fight |
| `B_current_lose_streak` | `B_current_lose_streak` | |
| `R_longest_win_streak` | `R_longest_win_streak` | Career longest win streak |
| `B_longest_win_streak` | `B_longest_win_streak` | |
| `R_avg_SIG_STR_pct` | `R_avg_SIG_STR_pct` | Career avg sig str accuracy % |
| `B_avg_SIG_STR_pct` | `B_avg_SIG_STR_pct` | |
| `R_avg_SUB_ATT` | `R_avg_SUB_ATT` | Career avg submission attempts |
| `B_avg_SUB_ATT` | `B_avg_SUB_ATT` | |
| `R_avg_TD_pct` | `R_avg_TD_pct` | Career avg takedown accuracy % |
| `B_avg_TD_pct` | `B_avg_TD_pct` | |
| `R_total_title_bouts` | `R_total_title_bouts` | Total UFC title bouts |
| `B_total_title_bouts` | `B_total_title_bouts` | |

---

## Group 2 — Derived diffs from ufc-master.csv (12 features)

Computed directly: `R_col - B_col`.

| Feature | R col | B col |
|---|---|---|
| `win_dif` | `R_wins` | `B_wins` |
| `loss_dif` | `R_losses` | `B_losses` |
| `win_streak_dif` | `R_current_win_streak` | `B_current_win_streak` |
| `lose_streak_dif` | `R_current_lose_streak` | `B_current_lose_streak` |
| `height_dif` | `R_Height_cms` | `B_Height_cms` |
| `reach_dif` | `R_Reach_cms` | `B_Reach_cms` |
| `age_dif` | `R_age` | `B_age` |
| `sig_str_dif` | `R_avg_SIG_STR_landed` | `B_avg_SIG_STR_landed` |
| `avg_td_dif` | `R_avg_TD_landed` | `B_avg_TD_landed` |
| `ko_dif` | computed from wins/method in master | |
| `sub_dif` | computed from wins/method in master | |
| `total_title_bout_dif` | `R_total_title_bouts` | `B_total_title_bouts` |

---

## Group 3 — Bout context (2 features)

| Feature | Source | Description |
|---|---|---|
| `weight_class_ord` | `weight_class` | Ordinal 0–11 (Women's Straw=0, HW=11) |
| `title_bout_bin` | `title_bout` | 1 if title fight, 0 otherwise |

---

## Group 4 — Stance (4 features)

| Feature | Description |
|---|---|
| `R_southpaw` | 1 if Red is southpaw |
| `B_southpaw` | 1 if Blue is southpaw |
| `orth_clash` | 1 if both orthodox |
| `south_clash` | 1 if both southpaw |

---

## Group 5 — Career stats from career_fights_updated.csv, shift(1) (26 features)

Computed by `compute_career_stats()` in train_model1.py. Every value represents pre-fight accumulated stats — the fighter's history BEFORE stepping in for this fight.

| Feature | Window | Description |
|---|---|---|
| `R_cum_fights` | all-time | Number of prior career fights |
| `B_cum_fights` | all-time | |
| `R_career_win_rate` | all-time | All-career win rate (cum wins / cum fights, 0.5 default for debut) |
| `B_career_win_rate` | all-time | |
| `career_win_rate_dif` | | R - B |
| `R_last5_won` | last 5 | Rolling 5-fight win rate |
| `B_last5_won` | last 5 | |
| `last5_won_dif` | | R - B |
| `R_last5_finish_rate` | last 5 | Rolling 5-fight finish rate |
| `B_last5_finish_rate` | last 5 | |
| `last5_finish_rate_dif` | | R - B |
| `R_opp_quality` | last 5 | Avg career win rate of last 5 opponents |
| `B_opp_quality` | last 5 | |
| `opp_quality_dif` | | R - B |
| `R_trend_score` | 3 vs 10 | `last3_win_rate - last10_win_rate` (momentum) |
| `B_trend_score` | 3 vs 10 | |
| `trend_score_dif` | | R - B |
| `R_ko_finish_rate` | all-time | KO/TKO wins / total fights |
| `B_ko_finish_rate` | all-time | |
| `ko_finish_rate_dif` | | R - B |
| `R_sub_finish_rate` | all-time | Submission wins / total fights |
| `B_sub_finish_rate` | all-time | |
| `sub_finish_rate_dif` | | R - B |
| `R_last3_win_rate` | last 3 | |
| `B_last3_win_rate` | last 3 | |
| `last3_win_rate_dif` | | |
| `R_last10_win_rate` | last 10 | |
| `B_last10_win_rate` | last 10 | |
| `last10_win_rate_dif` | | |

---

## Group 6 — Age × Experience interaction (3 features)

| Feature | Formula | Description |
|---|---|---|
| `R_age_x_exp` | `R_age × R_cum_fights` | Experience-weighted age |
| `B_age_x_exp` | `B_age × B_cum_fights` | |
| `age_x_exp_dif` | R - B | |

---

## Group 7 — Layoff buckets (8 features)

Days since last fight, bucketed. Default 180 days if no prior fight.

| Feature | Condition |
|---|---|
| `R_layoff_lt90` | days < 90 |
| `R_layoff_90_180` | 90 ≤ days < 180 |
| `R_layoff_180_365` | 180 ≤ days < 365 |
| `R_layoff_gt365` | days ≥ 365 |
| `B_layoff_*` | same for Blue |

---

## Group 8 — Style stats from ufc_fighters_final_updated.csv (14 features)

Joined as a left merge on fighter name (after deduplication). No temporal component — reflects career-to-date UFC Stats values.

| Feature | Description |
|---|---|
| `R_SLpM`, `B_SLpM` | Strikes landed per minute |
| `R_SApM`, `B_SApM` | Strikes absorbed per minute |
| `R_Str_Acc`, `B_Str_Acc` | Striking accuracy (0–1) |
| `R_Str_Def`, `B_Str_Def` | Striking defense (0–1) |
| `R_TD_Avg`, `B_TD_Avg` | Takedowns per 15 min |
| `R_TD_Acc`, `B_TD_Acc` | Takedown accuracy (0–1) |
| `R_TD_Def`, `B_TD_Def` | Takedown defense (0–1) |
| `R_Sub_Avg`, `B_Sub_Avg` | Submission attempts per 15 min |
| `SLpM_dif` | R_SLpM - B_SLpM |
| `SApM_dif` | R_SApM - B_SApM |
| `Str_Def_dif` | R_Str_Def - B_Str_Def |
| `TD_Def_dif` | R_TD_Def - B_TD_Def |
| `Sub_Avg_dif` | R_Sub_Avg - B_Sub_Avg |
| `TD_Avg_dif` | R_TD_Avg - B_TD_Avg |

---

## Group 9 — Elo (6 features)

Computed all-time from ufc-master.csv with K=48, base=1500. `elo_before` is the fighter's rating going INTO the fight (not updated until after).

| Feature | Description |
|---|---|
| `R_elo`, `B_elo` | Pre-fight Elo rating |
| `elo_dif` | R_elo - B_elo |
| `R_elo_trend`, `B_elo_trend` | elo_before - elo_before.shift(3) per fighter |
| `elo_trend_dif` | R_elo_trend - B_elo_trend |
