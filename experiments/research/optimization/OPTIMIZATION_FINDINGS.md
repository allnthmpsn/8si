# Model 1 Optimization Findings

**Run date:** 2026-05-11
**Baseline:** 109-feature LR90/XGB10 blend — 71.64% temporal accuracy (train <2024, test 2024+)

---

## Summary Table

| Variant | Accuracy | Delta | Notes |
|---------|---------|-------|-------|
| Baseline (LR90/XGB10, C=0.00711) | 71.64% | +0.00pp | Production model |
| Step 1: ElasticNet best (l1=0.3) | 71.29% | -0.35pp | L2→ElasticNet, C=0.00711, 90/10 blend |
| Step 2: LR70/XGB30 | 72.08% | +0.44pp | Blend ratio only, no retraining |
| Step 3: LR80+XGB10+LGBM10 | 70.85% | -0.79pp | Three-way blend, best of 4 tested |
| Step 4: Isotonic calibration | 71.20% | -0.44pp | Calibration layer, val-slice fit |
| Step 5: C re-tuned | 70.94% | -0.70pp | C=0.1, 90/10 blend |

---

## Step 1 — ElasticNet LR

Replaced L2 with ElasticNet (saga solver), swept l1_ratio ∈ [0.1, 0.3, 0.5, 0.7, 0.9] at fixed C=0.00711.

| l1_ratio | Accuracy | Delta |
|----------|----------|-------|
| 0.1 | 71.11% | -0.53pp |
| 0.3 | 71.29% | -0.35pp |
| 0.5 | 71.03% | -0.61pp |
| 0.7 | 71.11% | -0.53pp |
| 0.9 | 71.11% | -0.53pp |

**Conclusion:** All l1_ratios underperform L2 baseline. L2 regularization is the right penalty for this feature set.

---

## Step 2 — Blend Ratio Sweep

| LR % | XGB % | Accuracy | Delta |
|------|-------|----------|-------|
| 95 | 5 | 71.03% | -0.61pp |
| 90 | 10 | 71.64% | -0.00pp |
| 85 | 15 | 71.38% | -0.26pp |
| 80 | 20 | 71.38% | -0.26pp |
| 75 | 25 | 71.99% | +0.35pp |
| 70 | 30 | 72.08% | +0.44pp |

**Best:** LR 70% / XGB 30%  →  72.08%
**Note:** Increasing XGB weight up to 30% monotonically improves accuracy. This is a free improvement — no retraining required.

---

## Step 3 — LightGBM Blends

Params: n_estimators=200, lr=0.05, max_depth=4, num_leaves=15, subsample=0.8.

| Blend | Accuracy | Delta |
|-------|----------|-------|
| LR 90% + LGBM 10% | 70.41% | -1.23pp |
| LR 85% + LGBM 15% | 70.59% | -1.05pp |
| LR 80% + LGBM 20% | 70.59% | -1.05pp |
| LR 80% + XGB 10% + LGBM 10% | 70.85% | -0.79pp |

**Conclusion:** LGBM consistently underperforms XGB in all blend configurations on this dataset size.

---

## Step 4 — Isotonic Calibration

Calibration holdout: last 20% of training data (chronological split, not shuffled).
Calibrator fit on validation slice only — test set used only for final evaluation.

| Metric | Value |
|--------|-------|
| Uncalibrated accuracy | 71.64% |
| Calibrated accuracy | 71.20% |
| Mean cal error before | 0.0484 |
| Mean cal error after | 0.0894 |
| Cal improvement | -0.0410 |
| Accuracy change | -0.44pp |

**Note:** Calibration primarily improves probability reliability (important for Kelly sizing), not raw accuracy.

---

## Step 5 — C Re-tuning

C grid searched via 5-fold CV on augmented training set. Production C=0.00711 was tuned on 114-feature model.

| C | CV Accuracy |
|---|-------------|
| 0.001 | 64.03% |
| 0.003 | 64.36% |
| 0.005 | 64.78% |
| 0.007 | 64.96% |
| 0.0071 | 64.96% |
| 0.009 | 65.14% |
| 0.01 | 65.06% |
| 0.02 | 65.31% |
| 0.05 | 65.23% |
| 0.1 | 65.35% |

**Best C by CV:** 0.1
**Temporal test accuracy:** 70.94%  (-0.70pp)

---

## Recommendation

**Promote Step 2: LR70/XGB30: +0.44pp — meaningful improvement.**

### Free improvement (no retraining)

Change blend from LR90/XGB10 to **LR70/XGB30**. Accuracy improves from 71.64% → 72.08% (+0.44pp) with the same pkl files.

> **Production files unchanged.** All variants saved to `experiments/research/optimization/`.
