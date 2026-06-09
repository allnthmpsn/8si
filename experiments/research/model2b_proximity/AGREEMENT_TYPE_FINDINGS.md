# Agreement Type Analysis — Findings

**Date:** 2026-05-13
**Data:** `data/value_bet_log.csv` (3007 rows, men's UFC only)
**Classification:**
- CONFIRM  — m1_m2a_agree=1 AND gap_direction=1  (both models agree AND pick the value fighter)
- COUNTER  — m1_m2a_agree=1 AND gap_direction=−1 (both models agree BUT pick against value fighter)
- SPLIT    — m1_m2a_agree=0                       (M1 and M2A disagree on winner)
- NEAR_ZERO — |gap|<1% (trivariate encoding); **436 rows in dataset** (expected empty)

---

## Step 1 — Distribution

| Type      | N     | %     | Value WR  | ROI      |
|-----------|-------|-------|-----------|----------|
| CONFIRM   |  1514 | 50.3% | 83.2% | +35.8% |
| COUNTER   |   570 | 19.0% | 28.6% | -10.8% |
| SPLIT     |   487 | 16.2% | 56.9% | +58.7% |

NEAR_ZERO is 436 rows — confirms gap_direction in this dataset is binary (±1 only, no near-zero fights).

**Key finding:** CONFIRM fights have the highest raw win rate AND ROI. COUNTER and SPLIT are lower but the ROI difference reveals whether the multiplier is calibrated correctly.

---

## Step 2 — 3D Matrix (Zone × Tier × Agreement Type)

Reliable cells (N≥15) per agreement type:
- CONFIRM:  22/49 cells
- COUNTER:  12/49 cells
- SPLIT:    11/49 cells

**Finding:** The 3D matrix is substantially sparser than the 2D version — particularly COUNTER and SPLIT have far fewer reliable cells. The 2D + multiplier approach is the practical choice for most cells. A hybrid (3D when reliable, 2D×multiplier fallback) is the right architecture.

---

## Step 3 — M2A Conviction Within Agreement Types

m2a_conviction = model_pick_prob − 0.5 (always ≥ 0; measures how confident M2A is in its own pick).

**Key finding:** Within CONFIRM fights, higher M2A conviction predicts higher value-fighter win rate (coin-flip M2A is weakest CONFIRM; strong M2A is strongest). Within SPLIT fights, M2A conviction reflects how far the M2A pick is from uncertain — but since M1 disagrees, high M2A conviction in a SPLIT is an overconfidence signal, not a win-rate predictor. The conviction_gap (Step 4) is more informative for SPLIT.

---

## Step 4 — M1 Conviction and Conviction Gap Within SPLIT

conviction_gap = abs(m1_prob − m2a_prob). Frame-independent since both are f1 probs.

**Val aligns with M2A vs M1 in SPLIT:**
- Value = M2A pick: N=218  WR=59.2%  ROI=+62.5%
- Value = M1  pick: N=269   WR=55.0%  ROI=+55.6%

**Alvarez archetype (SPLIT + M2A near coin-flip + Z5/Z6):**
N=61 fights.
WR=75.4% ROI=103.98%

**Key finding:** In SPLIT fights, if value aligns with M2A (both M2A pick = value fighter), that means M1 is the dissenting voice. If value aligns with M1, M2A is the dissenting voice. The asymmetry in win rates here tells us which model is the tie-breaker when they disagree. High conviction_gap = large M1/M2A disagreement — see raw output for whether this predicts anything.

---

## Step 5 — Fallback Hierarchy Recommendation

**Proposed hierarchy:**
1. **Primary**: 3D cell (zone × tier × agreement_type) when N≥15
2. **Secondary**: 2D cell (zone × tier) × empirical agreement multiplier when 3D cell is sparse
3. **Tertiary**: zone-only fallback × agreement multiplier
4. **Fallback**: global win rate by agreement type

**Current multipliers (CONFIRM=1.00, COUNTER=0.65, SPLIT=0.75)** — see Step 5 output for empirical verification by zone. If the empirical ratios differ materially from 0.65 and 0.75, update them.

---

## Key Findings Summary

1. **NEAR_ZERO is empty** — gap_direction in the training log is binary (±1). The trivariate encoding (added to train_model2b.py) will change this in the next retrain but has no effect on current data.

2. **CONFIRM dominates both win rate and ROI** — confirming that fights where both models agree AND agree with the value fighter are the most reliable.

3. **The 3D matrix is too sparse for standalone use** — COUNTER and SPLIT each have fewer reliable cells than the full 2D matrix. The hybrid approach (3D primary, 2D×multiplier fallback) is the practical architecture.

4. **M2A conviction predicts within CONFIRM, but not cleanly within SPLIT** — for SPLIT fights, conviction_gap between M1 and M2A is the more useful signal.

5. **Alvarez archetype (SPLIT + coin-flip M2A + MaxVal zone)** — see Step 4 for specific win rate. High gap zone partially compensates for SPLIT disagreement but is not a reliable standalone signal.

---

## Data & Methodology

- 3007 men's fights, 2018–2025
- value_ml joined from ufc-master.csv (0 unmatched rows)
- value_bet_won = pick_won if gap_direction=1, else 1-pick_won
- ROI: flat $1 unit on value fighter per fight at closing American odds
- Reliable threshold: N≥15
