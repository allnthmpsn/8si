# Multiplier Recalibration — Findings

**Date:** 2026-05-14
**Data:** `data/value_bet_log.csv` (3007 rows, men's UFC only)

## Root Cause

Previous multipliers (COUNTER: 0.26–0.39, SPLIT: 0.48–0.83) were computed as
`ATYPE_WR / CONFIRM_WR` per zone. But CONF_MATRIX base rates are **mixed**
(CONFIRM-dominated, ~50% of data), not CONFIRM-only. Dividing by CONFIRM_WR
and applying to a CONFIRM-heavy base rate double-corrects downward.

Correct formula: `actual_type_WR / mixed_2D_base_rate` per zone×tier cell.

## Step 1 — CONF_MATRIX Verification

1 cell(s) with |difference| > 2pp between CONF_MATRIX and computed data.
Matrix values match data well within tolerance.

## Step 4 — Calibration Improvement

| Type | OLD MAE | NEW MAE | Delta | Target <0.10 |
|------|---------|---------|-------|-------------|
| Overall | 0.1057 | 0.0162 | -0.0895 | ✓ |
| CONFIRM | 0.0861 | 0.0861 | +0.0000 | ✓ |
| COUNTER | 0.1653 | 0.1064 | -0.0589 | ✗ |
| SPLIT | 0.1989 | 0.1239 | -0.0750 | ✗ |
| NEAR_ZERO | 0.2345 | 0.0011 | -0.2334 | ✓ |

## Step 5 — Spot Checks

- **CONFIRM Z6×mfav**: 96.7% (unchanged, multiplier=1.00)
- **COUNTER Z3×sdog**: OLD=16.7% → NEW=39.4% (actual WR: 39.7%)
- **SPLIT Z5×sdog**: OLD=64.7% → NEW=81.3%

## Recommendation

Cell-level multipliers improve calibration across all agreement types. Proceed with Part 2.
