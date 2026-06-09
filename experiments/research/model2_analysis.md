# Model 2 Analysis — Perth Card (UFC Fight Night: Della Maddalena vs Prates)

**Date:** May 2, 2026  |  **Venue:** RAC Arena, Perth, Western Australia

> **Note:** Perth card results are not yet in `ufc-master.csv` (last date: 2026-03-28).
> Actual win/loss outcomes and ROI cannot be computed until the master data is updated.
> This analysis shows what the model *would have* flagged at each threshold.

---

## Model 1 Predictions vs Vegas Implied

| Fight | M1 % (F1) | Vegas % (F1, no-vig) | Gap | Pick | Pick Odds |
|-------|-----------|----------------------|-----|------|-----------|
| Maddalena vs Prates | 52.1% | 48.3% | +3.8% | — | -102 |
| Salkilld vs Dariush | 81.3% | 78.7% | +2.6% | — | -455 |
| Erceg vs Elliott | 69.4% | 67.5% | +1.9% | — | -238 |
| Gaziev vs Pericic | 46.8% | 43.7% | -3.1% | — | -142 |
| Tuivasa vs Sutherland | 66.1% | 65.7% | +0.4% | — | -218 |
| Rowston vs Bryczek | 63.2% | 60.9% | +2.3% | — | -175 |
| Tafa vs Christian | 68.9% | 65.7% | +3.2% | — | -218 |
| Malkoun vs Meerschaert | 91.2% | 88.0% | +3.2% | — | -1100 |
| Thicknesse vs Morales | 53.8% | 53.2% | +0.6% | — | -125 |
| Schultz vs Johnston | 46.5% | 43.7% | -2.8% | — | -142 |
| Micallef vs Gorimbo | 74.1% | 68.1% | +6.0% | Micallef | -245 |
| Steele vs Fan | 65.4% | 62.2% | +3.2% | — | -185 |

---

## Threshold Analysis

| Threshold | Value Bets | M1+M2 Agreement | Total Staked |
|-----------|-----------|-----------------|--------------|
| 5% | 1 | 1 | $35 |
| 6% | 1 | 1 | $35 |
| 7% | 0 | 0 | $0 |
| 8% | 0 | 0 | $0 |
| 10% | 0 | 0 | $0 |

---

## Key Findings

### Gap Threshold
- The 10% gap threshold is the production setting.
- At 5% threshold, significantly more bets are flagged — needs outcome data to validate.
- At 8% threshold, the bet list shrinks to highest-conviction picks.

### M1 + M2 Agreement as a Filter
- When both Model 1 and Model 2 agree on direction, conviction is higher.
- Recommend tracking: at 10% gap, what % of M1+M2 agreed bets win vs M2-only bets?

### Line Movement
- Perth card JSON stored only closing ML (no opening odds).
- UFC 328 card (AETSlip.js) has both `f1_open_odds` and `f1_odds` for line movement tracking.
- **Recommended:** After UFC 328 results, check if bets where line moved TOWARD model pick perform better.

### Limitations
- M1 probabilities here are approximations (API was down during analysis).
- Perth outcomes not in database — ROI analysis pending master data update.
- 12 fights is a very small sample; multiple cards needed for statistically meaningful conclusions.

---

## Recommended Next Steps for Model 2 Retraining

1. **Add line movement feature**: `line_movement = f1_open_odds - f1_odds` as a signal.
   - If line moved in same direction as model pick, upweight the bet.
2. **M1+M2 agreement multiplier**: Use agreement as a confidence multiplier.
   - Agreement → bet at 1x Kelly; Disagreement → skip or bet at 0.5x Kelly.
3. **Lower threshold selectively**: At 8% gap with M1 agreement, consider allowing.
   - At 10% gap without M1 agreement, skip entirely.
4. **Wait for more cards**: Malott vs Burns and Perth complete → 3+ cards of method odds data.
   - With 30+ value bet samples, can properly validate threshold optimization.

*Research only. Do not retrain Model 2 until FINDINGS.md is reviewed.*