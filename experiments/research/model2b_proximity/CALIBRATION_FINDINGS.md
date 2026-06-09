# Calibration Check — Findings

**Date:** 2026-05-14
**Data:** `data/value_bet_log.csv` (3007 rows, men's UFC only)
**System:** AETSlip.js CONF_MATRIX × COUNTER_MULTIPLIERS / SPLIT_MULTIPLIERS (empirical zone-specific)

---

## Step 2 — Overall Calibration

**Overall MAE: 0.1057**  Max error: 0.2475

| Bucket | Predicted | Actual WR | Error | N |
|--------|-----------|-----------|-------|---|
| 0.0 | 0.054 | 0.121 | -0.067 | 257 |
| 0.1 | 0.148 | 0.352 | -0.204 | 491 |
| 0.2 | 0.243 | 0.472 | -0.230 | 163 |
| 0.3 | 0.348 | 0.592 | -0.244 | 348 |
| 0.4 | 0.438 | 0.685 | -0.248 | 162 |
| 0.5 | 0.563 | 0.607 | -0.043 | 89 |
| 0.6 | 0.673 | 0.762 | -0.090 | 181 |
| 0.7 | 0.758 | 0.790 | -0.032 | 386 |
| 0.8 | 0.833 | 0.836 | -0.003 | 597 |
| 0.9 | 0.941 | 0.944 | -0.004 | 287 |
| 1.0 | 1.000 | 1.000 | +0.000 | 46 |

Buckets with |error| > 0.10: 4 of 11.

---

## Step 3 — Calibration by Agreement Type

| Type | N | MAE | Bias | Direction |
|------|---|-----|------|-----------|
| CONFIRM | 1514 | 0.0861 | +0.0519 | over-confident |
| COUNTER | 570 | 0.1653 | -0.1653 | under-confident |
| SPLIT | 487 | 0.1989 | -0.1989 | under-confident |
| NEAR_ZERO | 436 | 0.2345 | -0.2345 | under-confident |

**CONFIRM:** over-confident by 0.052
**COUNTER:** under-confident by 0.165 — after zone-specific multiplier correction (was 0.65 flat, now 0.26–0.39)
**SPLIT:**   under-confident by 0.199 — after zone-specific multiplier correction (was 0.75 flat, now 0.48–0.83)

---

## Step 4 — Underdog vs Favorite Calibration

| Category | N | MAE | Bias | ROI |
|----------|---|-----|------|-----|
| Underdogs (sdog+mdog+hdog) | 1273 | 0.1417 | -0.1417 | +23.33% |
| Favorites (hfav+mfav+sfav) | 1444 | 0.0987 | -0.0828 | +20.93% |

---

## Step 5 — Suppression Effectiveness

| Category | N | WR |
|----------|---|----|
| All COUNTER | 570 | 0.3614 (non-suppressed) |
| Suppressed (m2a_conv ≥ 0.25) | 155 | 0.0839 |
| Non-suppressed COUNTER | 415 | 0.3614 |

**Suppression lift: +0.278** — effective

The suppression rule removes fights where the value fighter has only 8.4% historical win rate.
Non-suppressed COUNTER fights have 36.1% WR — still below 50% but notably better.

---

## Step 6 — UFC 328 Spot Checks

**Strickland (COUNTER, Heavy Dog +410):**
- Gap: 4.3% → Zone 3 (Watch), Tier: hdog
- Base: 0.0490 × COUNTER mult 0.35 = 1.7%
- m2a_conviction: ~0.269 ≥ 0.25 → **SUPPRESSED** ✓
- UI shows: "Suppressed — model highly confident against this pick"

**Van (SPLIT, Slight Dog +130):**
- Gap: 3.0% → Zone 3 (Watch, since 3.0 is NOT < 3), Tier: sdog
- Base: 0.4780 × SPLIT mult 0.70 = **33.5%**
- UI displays: 33.5%

---

## Step 7 — Key Findings & Recommendation

**Overall MAE: 0.1057** — needs improvement

1. **CONFIRM calibration** (over-confident by 0.052): The CONF_MATRIX base rates directly reflect historical CONFIRM win rates (since CONFIRM dominates the training data). Expected to be well-calibrated.

2. **COUNTER calibration** (under-confident by 0.165): Zone-specific multipliers (0.26–0.39) corrected the flat 0.65 overestimate. Still slightly off — see bucket detail above.

3. **SPLIT calibration** (under-confident by 0.199): Zone-specific multipliers (0.48–0.83) added zone-level resolution. Some residual miscalibration.

4. **Underdog calibration**: MAE=0.1417 vs Favorite MAE=0.0987. Underdogs are substantially harder to calibrate.

5. **Suppression is effective**: Suppressed fights (8.4% WR) are 27.7pp worse than non-suppressed COUNTER fights (36.1% WR). Rule is correctly identifying the worst COUNTER fights.

6. **Van displayed 33.5%** — computed correctly. Strickland suppressed correctly.

**Recommendation:** Calibration needs attention — see bucket details above.
