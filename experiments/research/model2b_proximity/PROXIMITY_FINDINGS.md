# M1 vs M2A Proximity Analysis — Findings

**Date:** 2026-05-13  
**Data:** `data/value_bet_log.csv` (3,007 rows, 2018–present)  
**Metric:** `proximity = abs(m1_prob - m2a_prob)` — always frame-independent, same either fighter

---

## Key Questions

### Q1 — Does within-3% proximity + agreement outperform just agreement (79.1%)?

**No.** Within-3% proximity + agree = **78.5% (N=641)**. Overall agree = **79.1% (N=2,479)**.

Proximity filtering *removes* agree fights, and the fights removed happen to perform at roughly the same rate. Tightening to within-3% recovers 26% of agree fights at a cost of 0.6pp. Not worth it as a filter.

---

### Q2 — At what proximity bucket does agree win rate drop off?

**It doesn't, in any meaningful way.**

| Bucket  | Agree WR | N   |
|---------|----------|-----|
| ≤1%     | 79.1%    | 234 |
| 1–3%    | 78.1%    | 407 |
| 3–5%    | 76.5%    | 400 |
| 5–10%   | 79.3%    | 767 |
| 10–20%  | 81.0%    | 589 |
| 20%+    | **81.7%**| 82  |

Agree WR is effectively flat from ≤1% through 20%+. The 3–5% bucket dips to 76.5% but rebounds immediately. **The agreement flag dominates; proximity within the agree group is noise.**

---

### Q3 — When models are 20%+ apart but agree, is the signal still meaningful?

**Yes — and it's the strongest agree bucket: 81.7% (N=82).**

This is counterintuitive. When M1 gives F1 80% and M2A gives F1 40% but both still predict the same winner (F2), the winner still comes through at 81.7%. The agreement direction matters far more than the magnitude of the underlying probabilities. These are rare fights (82 out of 3,007) but the signal holds.

---

### Q4 — Within-1% proximity + triple_agree win rate?

**81.2% (N=202).** Compare to the overall Lock zone (triple_agree + gap<1%) rate of **74.3% (N=438)**.

Adding proximity ≤1% on top of triple_agree filters from 438 to 202 fights (+10.9pp on win rate). The tighter the proximity between M1 and M2A *and* Vegas, the higher the convergence signal. This is the strongest multi-condition combo in the dataset.

---

## Cross-Tab: Proximity × Gap Zone (win rate % / N)

| Bucket  | Z0 Lock | Z1 Strong | Z2 Lean | Z3 Watch | Z4 Value | Z5 StrongVal | Z6 MaxVal |
|---------|---------|-----------|---------|----------|----------|--------------|-----------|
| ≤1%     | 88.5/26 | 73.0/37   | 81.2/32 | 71.1/45  | 70.7/41  | 90.0/20      | **91.2/34** |
| 1–3%    | 73.6/72 | 82.1/56   | 80.7/57 | 69.2/91  | 75.7/74  | 85.7/35      | 82.1/39   |
| 3–5%    | 72.9/59 | 69.8/53   | 72.2/54 | 72.3/94  | 81.5/92  | 75.0/40      | 78.1/32   |
| 5–10%   | 67.4/129| 66.9/136  | 76.4/110| 76.2/172 | 77.7/179 | 80.4/51      | 84.8/99   |
| 10–20%  | 76.0/121| 68.5/108  | 70.5/105| 69.2/143 | 62.6/155 | 56.5/62      | 75.6/131  |
| 20%+    | 64.5/31 | 65.0/20   | 55.6/18 | 65.2/23  | 54.1/37  | 50.0/20      | **51.4/74** |

Notable patterns:
- **≤1% + MaxVal (Z6)**: 91.2% at 34 fights — highest cell in the table.
- **≤1% + StrongVal (Z5)**: 90.0% at 20 fights — very small N but striking.
- **20%+ + MaxVal (Z6)**: 51.4% — model divergence fully erases the gap-zone signal.
- The 10–20% bucket is where Value/StrongVal zones start degrading (62.6%, 56.5%). **When models are 10%+ apart and Vegas shows a large gap, be cautious.**

---

## Recommendation: Add `m1_m2a_proximity` as Continuous Feature to M2B?

**No — not as a standalone feature.**

Correlation breakdown:
- Overall: `corr(proximity, pick_won) = -0.125` — looks large but is driven by confounding
- Within agree group: `+0.033` — not meaningful
- Within disagree group: uncorrelated or mild

The overall -0.125 is a confounding artifact: low proximity (models close together) is *more common* in disagree fights (both models near 50/50, just on different sides), which have low win rates. Within the agree group — where M2B actually operates — proximity adds essentially nothing beyond what the binary `m1_m2a_agree` flag already captures.

**What to do instead:**
- The `m1_m2a_agree` binary feature (already in M2B at 28.2% importance) captures the key split.
- The interaction `within_1pct + triple_agree` is interesting (81.2% WR) but samples are too small (202) to retrain on stably.
- If retraining M2B, consider adding `proximity` only as an interaction term: `proximity × m1_m2a_agree`. That interaction captures "how confidently do the models agree" without the confounding.

**Bottom line:** Proximity is not an independent predictor. It's a proxy for agreement confidence, which `m1_m2a_agree` already captures. Skip for now; revisit if M2B sample size grows past 5,000 rows.

---

## Data & Methodology

- 3,007 fights, 2018–2025, men's UFC only
- Temporal split preserved: 2,473 train / 534 test
- `pick_won` = 1 if M2A's predicted winner actually won
- `proximity` = abs(m1_prob_f1 − m2a_prob_f1); frame-invariant since both models output F1 probability
- Gap zones: 0=Lock<1%, 1=Strong<2%, 2=Lean<3%, 3=Watch<5%, 4=Value<8%, 5=StrongVal<10%, 6=MaxVal>10%
