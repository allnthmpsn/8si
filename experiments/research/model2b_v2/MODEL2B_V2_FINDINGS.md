# Model 2B V2 — Findings Report
Generated: 2026-05-14

## Setup

- Training universe: 3,007 rows from data/value_bet_log.csv
- Temporal split: train=2210 (pre-2024), test=797 (2024+)
- Target: `pick_won`
- Feature count: 20 (was 15–16 in production)

## New Feature Correlations with `pick_won`

| Feature | Correlation | Note |
|---------|------------|------|
| gap_signed | +0.2489 | continuous signed gap |
| gap_size (ref) | +0.0226 | unsigned reference |
| m1_conviction | +0.3106 | abs(m1_prob - 0.5) |
| m2a_conviction | +0.2877 | abs(m2a_prob - 0.5) |
| conviction_product | +0.3083 | m1 × m2a conviction |
| conviction_gap | -0.1247 | \|m1_prob - m2a_prob\| |
| m1_m2a_agree (ref) | +0.3176 | binary reference |

gap_signed vs gap_size: gap_signed stronger (directional signal validated)

## Model Performance

| Model | CV Acc | Test Acc | Brier |
|-------|--------|----------|-------|
| LR (isotonic) | 0.7652 | 0.7051 | 0.2013 |
| Random Forest | 0.7638 | 0.7014 | 0.1944 |
| XGBoost | 0.7638 | 0.7051 | 0.1957 |
| 50/50 LR+XGB | — | 0.7051 | 0.1976 |
| 33/33 LR+RF+XGB | — | 0.7077 | 0.1961 |
| **Production M2B** | — | **0.7051** | **0.1943** |

Best ensemble: **ens_333** (acc=0.7077, Brier=0.1961)

## XGBoost Best Params (Optuna, 15 trials)

```json
{
  "n_estimators": 321,
  "learning_rate": 0.014702023890826074,
  "max_depth": 2,
  "min_child_weight": 11,
  "subsample": 0.6636655611866618,
  "colsample_bytree": 0.7181897632552066,
  "reg_alpha": 0.8186654304763922,
  "reg_lambda": 1.9500817996846322,
  "random_state": 42,
  "n_jobs": 1,
  "eval_metric": "logloss"
}
```

## Feature Importance (XGBoost, full dataset)

Top 15:

 1. m1_m2a_agree              0.1518
 2. conviction_product        0.1485
 3. gap_direction             0.0957
 4. m1_confidence             0.0888
 5. triple_agree              0.0825
 6. m1_conviction             0.0690
 7. m2a_conviction            0.0541
 8. gap_signed                0.0524
 9. vegas_agree               0.0377
10. m2a_confidence            0.0295
11. gap_zone                  0.0238
12. closing_odds              0.0228
13. conviction_gap            0.0216
14. gap_size                  0.0213
15. m2a_prob                  0.0211

gap_signed rank: #8  gap_size rank: #14
→ gap_signed ranks higher, directional signal validated

## Calibration vs Lookup Table

Production lookup table MAE: 0.0162 (by construction — built from same data)

Model 2B V2 calibration MAE by agreement type (test set):
- COUNTER: 0.0528  (target < 0.1064)
- SPLIT:   0.1988  (target < 0.1239)

Note: The lookup table achieves 0.0162 MAE by construction (it's the actual WR per cell).
A trained model will not match this in-sample precision but should generalize better
to unseen combinations.

## Agreement Type Analysis

Model 2B V2 mean predicted probability by agreement type (test set):
- CONFIRM: 0.842
- SPLIT:   0.384
- COUNTER: 0.766

Model correctly assigns lower confidence to COUNTER and SPLIT vs CONFIRM.

## Promotion Decision

| Criterion | Required | Actual | Pass? |
|-----------|----------|--------|-------|
| Test accuracy | ≥ 70.51% | 0.7077 | ✓ |
| Brier score | ≤ 0.1943 | 0.1961 | ✗ |
| COUNTER calib MAE | < 0.1064 | 0.0528 | ✓ |
| SPLIT calib MAE | < 0.1239 | 0.1988 | ✗ |
| gap_signed > gap_size | rank higher | #8 vs #14 | ✓ |

**Decision: NOT PROMOTED — saved as candidate**
Saved to: `/Users/allenthompson/Desktop/ufc-predictor/experiments/research/model2b_v2/model2b_v2_candidate.pkl`

Failed criteria: ['brier_pass', 'split_cal_pass']

## Recommended Next Steps

1. If not promoted: analyze which cells in the COUNTER/SPLIT calibration are worst and
   target those specifically with interaction features.
2. Gap direction trivariate encoding (1/0/-1 where 0 = near_zero) is already in the
   log — this is the 'gap_direction' column. Confirm it's using the correct encoding
   before next retrain.
3. Consider whether `vig` should be computed from opponent odds rather than approximated
   from one side only — this requires adding opponent odds to the value_bet_log.
4. If calibration MAE targets are met but accuracy isn't, consider a production hybrid:
   keep the lookup table for calibration but use V2 predictions for ranking.
5. Retrain with trivariate gap_direction analysis per model_metadata.json note
   (pos_gap+agree 83.2% WR vs neg_gap+agree 71.4% WR deserves explicit feature).
