# Positive vs Negative Gap Split — Findings

**Date:** 2026-05-13  
**Data:** `data/value_bet_log.csv` (3,007 rows, 2018–present)  
**Definition:** `gap = m2a_prob - pick_novig` (fraction, signed). Positive = model sees the pick as undervalued by Vegas. Negative = Vegas more confident than our model.  
**Groups (exclusive):** pos_gap = gap ≥ +1% (1,591 fights), neg_gap = gap ≤ -1% (980 fights), near_zero = |gap| < 1% (436 fights).

---

## Top-Line: Direction Is the Biggest Split in the Dataset

| Group     | N     | Overall WR | Agree WR | Disagree WR | ROI (agree) |
|-----------|-------|-----------|----------|-------------|-------------|
| pos_gap   | 1,591 | **81.3%**  | **83.2%**| 42.9%       | **+35.8%**  |
| near_zero | 436   | 72.5%      | 74.4%    | 53.7%       | +5.6%       |
| neg_gap   | 980   | 58.5%      | 71.4%    | 40.5%       | **-4.9%**   |

Gap direction creates a 11.8pp spread in agree-group win rate (83.2% vs 71.4%) and a 40pp spread in ROI (+35.8% vs -4.9%). This is larger than the difference between any two proximity buckets from the prior analysis.

---

## Key Questions

### Q1 — For neg_gap + agree: does WR exceed 79.1%?

**No. neg_gap + agree = 71.4% (N=570).**

Agreement is still predictive — 71.4% is well above random — but it's 7.7pp below the overall agree benchmark. When Vegas is more confident in the pick than our model is, even agreement doesn't fully recover.

---

### Q2 — Chimaev-style: large neg_gap (>10%) + agree + heavy favorite (<-300 odds) — WR?

**76.5% (N=17). Hypothesis of 85%+ not confirmed.**

Broader cut (neg >10% + agree, no odds threshold): **65.4% (N=26).**

Heavy favorites with large negative gaps don't outperform the agree group — in fact they're the weakest cell in the neg_gap segment. The "heavy fave + model consensus" intuition doesn't hold when Vegas is significantly more confident than our model. The model's lower probability is a real signal, not noise.

---

### Q3 — Do neg_gap + agree fights warrant different UI treatment?

**Yes, for suppression — not for separate surfacing.**

- pos_gap + agree ROI: **+35.8%** — the true edge cases
- neg_gap + agree ROI: **-4.9%** — near break-even to slightly negative
- neg_gap disagree ROI: **-40.5%** — strong avoid signal

Within neg_gap, agreement rescues win rate to 71.4% but does not rescue ROI. The market is already pricing in the edge that our model thinks it sees. These fights should not be surfaced as value picks. The current UI's 10% gap threshold already excludes most of them — but any neg_gap fight surfaced as a "value pick" is misleading regardless of gap size.

**Concrete implication:** The `valueGap = abs(gap)` used in BetSummary scoring doesn't distinguish direction. A neg_gap fight with large magnitude will score high. Consider whether to apply a direction penalty (multiply neg_gap valueScore by 0, or filter out entirely). Report only — do not change UI.

---

### Q4 — Neg gap by magnitude × agreement: does WR recover at high magnitudes?

**No — it degrades.**

| Neg magnitude | Agree WR | Agree N | Agree ROI | Disagree WR | Disagree N |
|---------------|----------|---------|-----------|-------------|------------|
| 0–3%          | 68.8%    | 263     | -4.5%     | 56.2%       | 73         |
| 3–5%          | 72.1%    | 154     | -4.9%     | 44.0%       | 84         |
| 5–10%         | **77.2%**| 127     | -2.7%     | 35.5%       | 172        |
| >10%          | 65.4%    | 26      | -20.2%    | 33.3%       | 81         |

There's a partial rebound at 5–10% (77.2%) but it crashes back at >10% (65.4%). ROI is negative at every magnitude even within the agree group. **Large negative gaps don't signal value — they signal market correctness.** The bigger the gap against us, the more the market knows something our model doesn't.

The 5–10% bump may be a small-N artifact or fighters who are genuinely misrated by our model. It doesn't warrant a different rule — the ROI is still -2.7%.

---

## Critical Asymmetry: Direction Changes Everything

```
pos_gap + agree:   83.2% WR, +35.8% ROI  ← strong edge
near_zero + agree: 74.4% WR,  +5.6% ROI  ← slight edge
neg_gap + agree:   71.4% WR,  -4.9% ROI  ← no edge
neg_gap + disagree: 40.5% WR, -40.5% ROI ← strong fade signal
```

The 79.1% overall agree benchmark obscures this. Agree fights that happen to be pos_gap (the majority: 1,514 of 2,479 agree fights) are the source of that benchmark's alpha. The 570 neg_gap agree fights bring the average down. Treating all agree fights equally leaves a significant amount of signal on the table.

---

## Recommendations (report only — no model or UI changes)

1. **BetSummary valueScore**: The current formula uses `abs(gap)` for magnitude, which treats pos and neg gaps symmetrarily. A neg_gap fight with 8% magnitude scores equivalently to a pos_gap fight with 8% magnitude — but their ROIs are +35.8% vs -4.9%. Direction should enter the scoring.

2. **Value Picks sections**: The 10% threshold already filters out most neg_gap fights from the "8SI VALUE PICKS" section since very few neg_gap fights have magnitude ≥ 10% AND agree (N=26). But those 26 fights have 65.4% WR — they should probably be filtered out explicitly.

3. **Proximity interaction (from prior analysis)**: The prior finding that `within_1% + triple_agree = 81.2%` may be confounded by direction. If near-zero proximity fights are split by direction, the true signal may differ.

4. **Future M2B input**: `gap_direction` (binary: 1 or -1) should be a candidate feature for M2B retraining. It carries 11.8pp of WR signal within the agree group and 40pp of ROI signal overall.

---

## Data & Methodology

- 3,007 fights, 2018–2025, men's UFC only  
- `gap = m2a_prob − pick_novig` (both in fraction form, same fighter frame)  
- Exclusive groups: pos_gap = gap ≥ +0.01, neg_gap = gap ≤ -0.01, near_zero = |gap| < 0.01  
- ROI: flat $1 unit per fight, American odds conversion, only fights with non-null odds  
- `pick_won = 1` if M2A's predicted winner actually won  
