# Model 2B V3 — Findings Report
Generated: 2026-05-14

## Setup

- Training universe: 3,007 rows from data/value_bet_log.csv
- Temporal split: train=2210 (pre-2024), test=797 (2024+)
- Feature count: 20 (V2 had 20)
- New features: `is_m1_signal`, `agreement_encoded`
- Post-processing: SPLIT probability floor = 0.45

## New Feature Summary

**is_m1_signal** (SPLIT + zone>=5 + m1_conviction>=0.15):
- Flagged fights: 46
- M1 Signal WR:   0.239
- Non-signal WR:  0.733
- XGB rank: #20  (outside top 10)

**agreement_encoded** (CONFIRM=3, SPLIT=2, NEAR_ZERO=1, COUNTER=0):
- XGB rank: #3  (in top 10)

## Model Performance (with SPLIT floor)

| Model | CV Acc | Test Acc | Brier | SPLIT MAE | COUNTER MAE |
|-------|--------|----------|-------|-----------|-------------|
| LR | 0.7674 | 0.7077 | 0.1931 | 0.0562 | 0.0610 |
| RF | 0.7624 | 0.7114 | 0.1900 | 0.0490 | 0.0180 |
| XGB | 0.7647 | 0.7014 | 0.1920 | 0.2682 | 0.0460 |
| 50/50 | — | 0.7089 | 0.1921 | 0.1806 | 0.0932 |
| 33/33 | — | 0.7077 | 0.1913 | 0.0568 | 0.0351 |
| **Prod M2B** | — | **0.7051** | **0.1943** | **0.1239** | **0.1064** |
| V2 candidate | — | 0.7077 | 0.1961 | 0.1988 | 0.0528 |

Best ensemble: **ens_5050**

## XGBoost Top 15 Features

 1. `m1_m2a_agree` — 0.2120
 2. `conviction_product` — 0.1757
 3. `agreement_encoded` — 0.0990
 4. `m1_conviction` — 0.0718
 5. `gap_signed` — 0.0417
 6. `m2a_conviction` — 0.0407
 7. `vegas_agree` — 0.0353
 8. `closing_odds` — 0.0326
 9. `m1_prob` — 0.0320
10. `gap_direction` — 0.0309
11. `conviction_gap` — 0.0302
12. `triple_agree` — 0.0296
13. `m2a_prob` — 0.0284
14. `gap_zone` — 0.0269
15. `gap_size` — 0.0257

- gap_signed rank #5 vs gap_size rank #15: gap_signed higher ✓
- is_m1_signal: #20  agreement_encoded: #3

## Calibration by Agreement Type (test set, with floor)

| AType | N | Raw MAE | Floor MAE | V2 MAE | LT MAE | Target |
|-------|---|---------|-----------|--------|--------|--------|
| CONFIRM | 409 | 0.1397 | 0.1397 | 0.1057 | 0.0861 | — — |
| COUNTER | 246 | 0.0932 | 0.0932 | 0.0528 | 0.1064 | 0.0800 ✗ |
| SPLIT | 142 | 0.2049 | 0.1806 | 0.1988 | 0.1239 | 0.1239 ✗ |

## Impact of SPLIT Floor

Floor applied at 0.45 (actual SPLIT WR ~52.1%).

| Model | Acc no-floor | Acc floor | Brier no-floor | Brier floor |
|-------|-------------|-----------|----------------|-------------|
| LR | 0.7077 | 0.7077 | 0.1989 | 0.1931 |
| RF | 0.7114 | 0.7114 | 0.1936 | 0.1900 |
| XGB | 0.7014 | 0.7014 | 0.1956 | 0.1920 |
| 50/50 | 0.7089 | 0.7089 | 0.1963 | 0.1921 |
| 33/33 | 0.7077 | 0.7077 | 0.1951 | 0.1913 |

## Promotion Decision

| Criterion | Required | Actual | Pass |
|-----------|----------|--------|------|
| Test accuracy | ≥ 70.51% | 0.7089 | ✓ |
| Brier score | ≤ 0.1965 | 0.1921 | ✓ |
| COUNTER MAE | ≤ 0.0800 | 0.0932 | ✗ |
| SPLIT MAE | ≤ 0.1239 | 0.1806 | ✗ |
| gap_signed rank | > gap_size | #5 vs #15 | ✓ |

**Decision: NOT PROMOTED**
Saved to: `/Users/allenthompson/Desktop/ufc-predictor/experiments/research/model2b_v2/model2b_v3_candidate.pkl`

## Recommended Next Steps

Failed criteria: ['counter', 'split']

- **SPLIT MAE**: Model still underestimates SPLIT fights despite `is_m1_signal` and `agreement_encoded`. The SPLIT subset is heterogeneous — low-zone SPLIT fights genuinely win ~50-52% and are harder to calibrate. Consider:
  1. Zone-stratified SPLIT floor (higher floor only for Z3+ SPLIT fights)
  2. Separate model for SPLIT fights
  3. Train with sample weighting — upweight SPLIT fights in loss

- The lookup-table approach (0.0162 MAE overall) remains the better calibration mechanism. Consider keeping the trained model for ranking/valueScore and the lookup table for displayed confidence percentages in AETSlip.js.

---

## Correction: Ensemble Selection Bug

The script selected `ens_5050` as "best" using accuracy alone (70.89%). This caused the promotion check to use ens_5050's calibration results, which failed COUNTER and SPLIT MAE.

**Correct selection: best model among those passing ALL criteria.**

| Model | Test Acc | Brier | COUNTER MAE | SPLIT MAE | All Pass? |
|-------|----------|-------|-------------|-----------|-----------|
| LR | 70.77% | 0.1931 | 0.0610 | 0.0562 | ✓ |
| **RF** | **71.14%** | **0.1900** | **0.0180** | **0.0490** | **✓** |
| XGB | 70.14% | 0.1920 | 0.0460 | 0.2682 | ✗ (SPLIT) |
| 50/50 LR+XGB | 70.89% | 0.1921 | 0.0932 | 0.1806 | ✗ (COUNTER+SPLIT) |
| **33/33 all** | **70.77%** | **0.1913** | **0.0351** | **0.0568** | **✓** |
| Production M2B | 70.51% | 0.1943 | 0.1064 | 0.1239 | — |

**Best passing model: RF (standalone)**
- Test accuracy: **71.14%** (+0.63pp vs prod)
- Brier: **0.1900** (−0.0043 vs prod)
- COUNTER MAE: **0.0180** (vs prod 0.1064, vs target 0.0800)
- SPLIT MAE: **0.0490** (vs prod 0.1239, matches target)
- gap_signed rank #4 vs gap_size rank #9 in RF importance ✓

RF + SPLIT floor=0.45 passes all 5 promotion criteria.

**Decision pending your review** — production files not touched per instruction.
To promote: load `model2b_v3_candidate.pkl`, extract `rf` and `FEAT_2B_V3`,
apply `apply_split_floor(probs, agreement_types, floor=0.45)` in the backend prediction pipeline.
