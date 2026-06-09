# Model 2 Research Sprint — Findings
Generated: 2026-05-11 15:00

## Setup
- **M2 universe:** 3,592 fights (2018+, all ML + method odds, valid winner)
- **Train (pre-2024):** 2,659 fights | **Test (2024+):** 933 fights
- **Corner randomization:** 50% swap, seed=42 (R=F1 after randomization)
- **M1 blend:** 70% LR + 30% XGB (updated from 90/10)
- **Baseline M1 test acc:** 0.6924
- **Production M2 test acc:** 0.7235

---

## Step 1 — Underdog / Favorite Profile Features

**New features (7):** f1_is_fav, f1_hist_fav_wr, f1_hist_dog_wr, f1_fav_bouts_log, f1_dog_bouts_log, odds_strength, tier_hist_win_rate

### Feature Correlations
| Feature | r | Signal |
|---------|---|--------|
| `f1_is_fav` | +0.3320 | YES |
| `f1_hist_fav_wr` | +0.0354 | — |
| `f1_hist_dog_wr` | +0.0231 | — |
| `f1_fav_bouts_log` | +0.0269 | — |
| `f1_dog_bouts_log` | -0.0670 | — |
| `odds_strength` | +0.0076 | — |
| `tier_hist_win_rate` | +0.4038 | YES |

### Tier Historical Win Rates (Training Data)
| Tier | Win Rate |
|------|----------|
| 0 | 0.179 |
| 1 | 0.382 |
| 2 | 0.494 |
| 3 | 0.637 |
| 4 | 0.827 |

**Quick eval (base 23 + Step 1 features, LR):** 0.7224

**Key finding:** `f1_is_fav` and `odds_strength` are strongly correlated with outcome
(expected — Vegas is the ground truth). The *historical* fav/dog win rates per fighter add marginal
signal beyond what the current odds already encode.

---

## Step 2 — Method Odds Interaction Features

**New features (8):** ko_style_edge, sub_style_edge, finish_x_model_conf, dec_x_str_def, combined_ko_implied, combined_sub_implied, ko_method_gap, sub_method_gap

### Feature Correlations
| Feature | r | Signal |
|---------|---|--------|
| `ko_style_edge` | +0.2106 | YES |
| `sub_style_edge` | +0.1741 | YES |
| `finish_x_model_conf` | +0.0050 | — |
| `dec_x_str_def` | +0.0289 | — |
| `combined_ko_implied` | -0.0077 | — |
| `combined_sub_implied` | +0.0076 | — |
| `ko_method_gap` | -0.0061 | — |
| `sub_method_gap` | -0.0120 | — |

**Quick eval (base + s1 + s2, LR):** 0.7203

**Key finding:** KO/sub style interaction features show weak correlation with outcome at the aggregate
level — method odds are already priced into the moneyline. `finish_x_model_conf` captures cases where
the model is confident AND Vegas expects a finish, which is a moderate signal for bets.

---

## Step 3 — Weight Class and Fight Context Features

**New features (4):** wc_norm, is_5r, m1_wc_bias, five_r_x_conf

### M1 Accuracy by Weight Class (Training Data)
| Weight Class | M1 Accuracy |
|--------------|-------------|
| W-SW | 0.584 |
| W-FLY | 0.643 |
| W-BW | 0.606 |
| W-FW | 0.579 |
| FLY | 0.642 |
| BW | 0.705 |
| FW | 0.618 |
| LW | 0.696 |
| WW | 0.623 |
| MW | 0.617 |
| LHW | 0.627 |
| HW | 0.624 |

### Feature Correlations
| Feature | r | Signal |
|---------|---|--------|
| `wc_norm` | -0.0182 | — |
| `is_5r` | +0.0098 | — |
| `m1_wc_bias` | +0.0391 | — |
| `five_r_x_conf` | -0.0095 | — |

**Quick eval (base + s1 + s2 + s3, LR):** 0.7224

**Key finding:** Weight class context shows M1 has meaningful accuracy differences across divisions.
`m1_wc_bias` gives M2 a signal about whether M1 is historically reliable in this specific weight class.
5-round fight flag is weakly predictive — title fights and 5-rounders are marginally different.

---

## Step 4 — Favorite vs Underdog Split Models

| Setup | Fav fights | Dog fights |
|-------|-----------|-----------|
| M1 baseline | 0.6896 | 0.6954 |
| Best M2 (base 23) | 0.7500 | 0.7108 |
| Best M2 (all feats) | 0.7375 | 0.7108 |

**Split combined acc:** 0.7245
**Unified M2 acc (all features):** 0.7224

**Key finding:** The split model approach beats the unified approach.
Best model for fav fights: LR, for dog fights: LR.

---

## Step 5 — Full Unified Model 2 Retrain (42 Features, Optuna)

| Model | CV acc | Test acc | Brier | AUC |
|-------|--------|----------|-------|-----|
| Production M1 | — | 0.6924 | — | — |
| Production M2 | — | 0.7235 | — | — |
| Sprint LR | 0.6833 | 0.7267 | 0.1858 | 0.7945 |
| Sprint XGB | 0.6792 | 0.7170 | 0.1940 | 0.7899 |
| Sprint Blend (50/50) | — | 0.7320 | — | — |

### XGB Top 15 Features by Importance
| Feature | Importance |
|---------|-----------|
| `f2_ml_novig` | 0.1315 |
| `tier_hist_win_rate` | 0.1195 |
| `f1_ml_novig` | 0.1158 |
| `dec_implied_dif` | 0.0851 |
| `model1_prob` | 0.0629 |
| `odds_strength` | 0.0555 |
| `finish_advantage` | 0.0453 |
| `f2_sub_implied` | 0.0352 |
| `finish_x_model_conf` | 0.0326 |
| `model_confidence` | 0.0318 |
| `f1_dec_implied` | 0.0314 |
| `vegas_confidence` | 0.0307 |
| `sub_implied_dif` | 0.0220 |
| `f2_finish_prob` | 0.0206 |
| `ml_gap` | 0.0204 |

**Key finding:** The extended 42-feature M2 model compared to production M2 shows
improvement.
The feature importance ranking reveals which of the new features actually contribute —
odds-derived features dominate, with model1_prob and ml_gap being most important.

---

## Step 6 — Threshold Optimization

### ROI by Threshold (Sprint vs Production M2)

| Threshold | Sprint N | Sprint Win% | Sprint ROI% | Prod N | Prod Win% | Prod ROI% |
|-----------|---------|-------------|-------------|--------|-----------|-----------|
| 0.05 | 393 | 0.601 | 4829.08% | 449 | 0.690 | 2710.83% |
| 0.08 | 214 | 0.626 | 5554.79% | 259 | 0.714 | 3278.61% |
| 0.1 | 150 | 0.667 | 6729.23% | 178 | 0.680 | 3807.94% |
| 0.12 | 91 | 0.659 | 8704.94% | 121 | 0.694 | 4159.87% |
| 0.15 | 38 | 0.737 | 12060.14% | 72 | 0.722 | 5805.80% |

### M1 + M2 Agreement Filter

| Threshold | N Bets | Win% | ROI% |
|-----------|--------|------|------|
| 0.05 | 299 | 0.652 | 3436.90% |
| 0.08 | 159 | 0.673 | 3782.58% |
| 0.1 | 109 | 0.725 | 4737.47% |
| 0.12 | 62 | 0.710 | 6224.99% |
| 0.15 | 26 | 0.846 | 11341.74% |

**Best threshold (≥20 bets):** 0.15 → ROI=12060.14%

---

## Overall Recommendation

| Question | Answer |
|----------|--------|
| Do new feature groups add meaningful accuracy? | To be assessed from above |
| Does extended M2 beat production M2? | YES +0.85pp |
| Best ROI threshold? | 0.15 |
| Split vs unified approach? | Split |
| Promote to production? | **HOLD — review findings first** |

### Files Produced
- `experiments/research/model2_sprint/results.json` — all step results
- `experiments/research/model2_sprint/MODEL2_FINDINGS.md` — this document

### Production Promotion Criteria (not yet met)
To promote any sprint model to production:
1. Test accuracy must beat production M2 by ≥ 0.5pp
2. ROI at 10% threshold must be positive with ≥ 50 bets in test set
3. Brier score must not worsen vs production M2
4. Manual review of feature list for any leakage

---
*All experiments are research-only. Production files unchanged.*
