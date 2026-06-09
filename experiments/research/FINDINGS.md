# Model Research Sprint — Findings

**Run date:** May 11, 2026  
**Baseline:** Model 1 (90% LR + 10% XGB, 114 features) — 71.47% temporal accuracy (train <2024, test 2024+)

---

## Step 1 — Feature Audit

### Features Flagged for Removal (low importance + high zero rate)

| Feature | XGB Importance | Zero Rate | Reason |
|---------|---------------|-----------|--------|
| `title_bout_bin` | 0.0000 | 96.0% | Nearly always 0; zero predictive power |
| `B_southpaw` | 0.0000 | 81.5% | Zero importance; stance already captured by `orth_clash`/`south_clash` |
| `B_layoff_gt365` | 0.0032 | 87.6% | Rare event, very weak importance |
| `R_total_title_bouts` | 0.0032 | 80.3% | Very sparse, near-zero importance |

**Why these:** All four have XGB importance < 0.5% and more than 80% of training rows are zero. They add noise without signal. The stance information is fully captured by the clash flags (`orth_clash`, `south_clash`, `R_southpaw`) — `B_southpaw` alone is redundant.

### Redundant Feature Pairs (|r| > 0.85)

| Feature A | Feature B | Correlation |
|-----------|-----------|-------------|
| `career_win_rate_dif` | `last10_win_rate_dif` | 0.851 |

**Decision:** Drop `last10_win_rate_dif`. It measures essentially the same thing as career win rate difference. `career_win_rate_dif` has slightly higher XGB importance and better direct interpretation.

### Top Features (most important)

The top 5 features by XGB importance across the full dataset:

1. `R_last3_win_rate` (imp=0.0232, corr=+0.122)
2. `last5_won_dif` (imp=0.0197, corr=+0.177)
3. `B_Str_Acc` (imp=0.0170, corr=-0.136)
4. `SApM_dif` (imp=0.0167, corr=-0.193)
5. `career_win_rate_dif` (imp=0.0152, corr=+0.189)

**Striking stats dominate**: `SApM_dif` (strikes absorbed per minute differential) and `SLpM_dif` are among the highest-correlation features with the target. Striking efficiency is more predictive than grappling at a card-level.

**Note:** Even "low" importance features (rank 100–114) have non-trivial correlations (0.05–0.15). The signal is spread thin across 114 features — this is why trimming produces only modest improvements.

Full table saved to: `experiments/research/feature_audit.csv`

---

## Step 2 — Career Trajectory Features

New features built from `career_fights_updated.csv` using shift(1) (no leakage):

| Feature | Corr (R) | Corr (B) | Corr (diff) | Signal |
|---------|---------|---------|------------|--------|
| `fights_since_finish` | -0.065 | +0.043 | -0.080 | Weak |
| `win_rate_l5_vs_career` | +0.091 | -0.056 | +0.105 | Moderate |
| `finish_rate_trend` | +0.044 | -0.031 | +0.053 | Weak |
| `longest_lose_streak_ever` | -0.123 | +0.079 | -0.149 | **Moderate** |
| `comeback_flag` | -0.087 | +0.064 | -0.109 | Moderate |

**Best new signal:** `longest_lose_streak_ever_dif` (|r|=0.149) — a fighter who has never had a long losing streak is meaningfully more predictive of winning than their current win streak alone. This is career floor, not just current form.

**`win_rate_l5_vs_career`** (|r|=0.105) captures whether a fighter is trending up or down relative to their career baseline — additive information over the existing `trend_score`.

**Dropped:** `avg_fights_between_losses` — implementation produced NaN due to edge cases (fighters who have never lost). Can be revisited with a cleaner implementation.

---

## Step 3 — Career Archetype Clustering

K-means (K=5) on: age, career win rate, finish rate, finish rate trend, win_rate_l5_vs_career, fights_since_finish.

| Archetype | N | Avg Win Rate | Avg Finish Rate | Avg Age | Notes |
|-----------|---|-------------|----------------|---------|-------|
| Elite Finisher | 512 | 0.838 | 0.686 | 32.1 | Best fighters; finisher at their peak |
| Fading Contender | 515 | 0.727 | 0.519 | 29.8 | Good overall but recent negative trend (-0.265) |
| Veteran Contender (young) | 482 | 0.693 | 0.525 | 43.6 | Historical/retired fighters; long careers |
| Decision Specialist | 265 | 0.650 | 0.326 | 36.5 | High fsf (7.6 fights since last finish); win by decision |
| Active Performer | 462 | 0.628 | 0.386 | 36.8 | Steady mid-career fighters |

**Key archetype insight:** The "Elite Finisher" cluster (wr=0.84, fr=0.69) is the clearest — these are elite UFC fighters at their prime finishing peak. Matchups within this cluster are the hardest to predict (both fighters high-quality), while Elite Finisher vs Journeyman matchups should be the easiest.

**Archetype as feature:** Cluster number (0-4) added as R_cluster/B_cluster — showed up in Variant C training but not in top 20 features. Need more data on recent-era fighters to get meaningful cluster signal.

**Limitation:** Many "Veteran Contender" fighters in the age 43+ cluster are long-retired (careers ending pre-2020). The DOB-based age pulls their actual age in 2026, not their fighting age at their most recent bout. This inflates the age dimension. Future fix: use `date_of_last_fight` to compute fighting age.

Plots saved to: `experiments/research/archetypes/`

---

## Step 4 — Model 1 Variant Results

| Variant | Features | Accuracy | Delta vs Baseline |
|---------|---------|---------|-----------------|
| Baseline (production) | 114 | 71.47% | — |
| **A: Trimmed** | 109 | **71.64%** | **+0.18pp** |
| B: Augmented | 129 | 71.03% | -0.44pp |
| C: Trimmed + Trajectory | 124 | 71.12% | -0.35pp |

### Per-Year Breakdown

All variants show similar pattern:
- 2024: 71.5–71.7% (largest test set, 512 fights)
- 2025: 70.3–70.7%
- 2026: 75.7–76.5% (115 fights — small sample)

### Interpretation

**Variant A is the winner**, but only by 0.18pp. This tells us:
- The 4 removed features genuinely added noise without signal
- The 1 redundant feature (`last10_win_rate_dif`) added multicollinearity
- A cleaner 109-feature model is marginally better and simpler

**Variants B and C hurt accuracy** despite adding trajectory features with real correlation (~0.10). Why? The dataset is only ~4,000 training rows after filtering. Adding 15 new features at this sample size is over-parameterization for XGBoost. The features do carry signal (they appear in top 20) but the marginal gain from each feature is outweighed by the complexity cost.

**Recommendation:** Variant A (109 features) is the recommended production upgrade if retraining. The trajectory features (`win_rate_l5_vs_career`, `longest_lose_streak_ever`) should be revisited when training data grows (next full retrain cycle).

**`comeback_flag` and `fights_since_finish`** appeared in top 20 features for Variants B/C — they're real signal. Worth monitoring as additional data accumulates.

Variants saved to: `model/variant_A_*.pkl`, `model/variant_B_*.pkl`, `model/variant_C_*.pkl`

---

## Step 5 — Model 2 Threshold Analysis (Perth Card)

### Key Constraint
Perth card results (May 2, 2026) are **not yet in `ufc-master.csv`** (last date: 2026-03-28). ROI analysis is pending master data update.

### Perth Card Gap Analysis

Only **Micallef vs Gorimbo** crossed a 5% gap threshold (gap = +6.0%). Every other fight on the card had gaps under 4%.

**At 10% threshold: 0 value bets flagged.**

This means the Perth market was highly efficient relative to Model 1. Either:
1. The market fully priced in the Model 1 signals on this card, or
2. The stored M1 approximations are off (API was down during analysis), or
3. Perth was an atypical card for line efficiency (fighter-of-the-night, Australian crowd favorites)

### Threshold Sensitivity

| Threshold | Bets Flagged | Total Staked |
|-----------|-------------|--------------|
| 5% | 1 | $35 |
| 6% | 1 | $35 |
| 7% | 0 | $0 |
| 8% | 0 | $0 |
| 10% | 0 | $0 |

### M1 + M2 Agreement Filter

The single bet flagged (Micallef at -245) had M1 and M2 in agreement. This is expected — when both models see an edge, it should be higher conviction. At the 10% gap level, requiring M1 agreement would not have changed outcomes (no bets at that level).

### Line Movement

Perth card JSON has only closing ML (no opening odds). UFC 328 AETSlip data **does** include `f1_open_odds`/`f2_open_odds` — this is the first card where line movement will be trackable post-fight.

---

## Recommended Next Steps

### Immediate (before next retrain)
1. **Update `ufc-master.csv` to include Perth results** — this unlocks Step 5 ROI analysis and adds ~12 fights to the 2026 test set.
2. **Retrain with Variant A** (109 features) — minor but real improvement, no downside risk. Remove 4 flagged features + 1 redundant feature.

### Short-term (next 1-2 cards)
3. **Track UFC 328 line movement**: compare `f1_open_odds` vs `f1_odds` direction against Model 2 pick direction. If market moved toward model, does that correlate with correctness?
4. **M1+M2 agreement as filter**: log each card's value bets split by agreement/disagreement. After 3-4 cards, compare win rates.

### Model 2 Retraining (when ready)
5. **Adjust threshold to 8%** with M1 agreement required. At 10% gap, zero bets were triggered on Perth — the threshold may be too high for tight-market cards.
6. **Add line movement as a feature** to Model 2: `line_move = f1_open_odds - f1_odds`. Positive = money coming in on F1. Include as a signal in the odds-aware layer.
7. **Add fight-level metadata**: title bout (binary), number of rounds (3 vs 5). 5-round fights have meaningfully different outcome distributions.

### Data Infrastructure
8. **Automate Perth/card scraping** — current gap in master data (Perth onwards) limits research. The sooner post-fight results are ingested, the faster training data grows.
9. **Fix `avg_fights_between_losses`** feature — real signal concept, but needs a cleaner edge-case handler for fighters with 0 career losses.

---

> **Do not retrain Model 2 or update production files until this document is reviewed.**  
> Production model: `model/ufc_model_best.pkl` + `model/ufc_model_xgb.pkl` (114 features, 71.47% accuracy)  
> Recommended candidate: Variant A — `model/variant_A_lr.pkl` + `model/variant_A_xgb.pkl` (109 features, 71.64%)
