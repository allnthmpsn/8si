# M1 vs M2A ROI Comparison — Findings

**Source:** value_bet_log.csv joined to ufc-master.csv for two-sided odds  
**Fights:** 3,007  |  **Date range:** 2018-01-14 to 2026-03-28

---

## Overall ROI (flat $1 per pick)

| Model | Bets | Accuracy | Profit | ROI% |
|---|---|---|---|---|
| M1  | 3,007 | 0.512 | -302.38 | -10.06% |
| M2A | 3,007 | 0.514 | -609.10 | -20.26% |

**ROI advantage:** M1 by 10.20pp

---

## ROI by Agreement / Disagreement

| Group | N | M1 WR | M1 ROI% | M2A WR | M2A ROI% | Winner |
|---|---|---|---|---|---|---|
| Both agree (same pick) | 2,478 | 0.516 | -19.85% | 0.516 | -19.85% | M2A |
| M1 picks F1 / M2A picks F2 | 308 | 0.536 | +41.92% | 0.464 | -26.13% | M1 |
| M1 picks F2 / M2A picks F1 | 221 | 0.434 | +27.32% | 0.566 | -16.62% | M1 |
| All disagreements | 529 | 0.493 | +35.82% | 0.507 | -22.16% | M1 |

---

## ROI by Odds Tier

### M1 picks

| Tier | N | WR | ROI% |
|---|---|---|---|
| Heavy Fav | 652 | 0.471 | -42.63% |
| Mod Fav | 1,086 | 0.552 | -18.06% |
| Slight Fav | 453 | 0.486 | -14.33% |
| Pick'em | 230 | 0.504 | +0.61% |
| Slight Dog | 430 | 0.526 | +30.32% |
| Mod Dog | 146 | 0.445 | +58.84% |

### M2A picks

| Tier | N | WR | ROI% |
|---|---|---|---|
| Heavy Fav | 733 | 0.482 | -41.18% |
| Mod Fav | 1,395 | 0.541 | -19.71% |
| Slight Fav | 503 | 0.485 | -14.61% |
| Pick'em | 233 | 0.515 | +2.79% |
| Slight Dog | 139 | 0.504 | +18.03% |

---

## Strategy Comparison

| Strategy | N | Accuracy | ROI% |
|---|---|---|---|
| A: Always M2A | 3,007 | 0.514 | -20.26% |
| B: Always M1 | 3,007 | 0.512 | -10.06% |
| C: Agree only (M2A pick) | 2,478 | 0.516 | -19.85% |
| D: M1 on splits, M2A on agrees | 3,007 | 0.512 | -10.06% |
| E: M2A on splits, M1 on agrees | 3,007 | 0.514 | -20.26% |
| F: Agree + dog picks only ← | 252 | 0.504 | +13.35% |
| G: Agree + fav picks only | 2,226 | 0.517 | -23.61% |

---

## Questions Answered

**Q1 — Overall ROI winner:** M1 (M1: -10.06%, M2A: -20.26%, gap: 10.20pp)

**Q2 — Disagreement fights winner:** M1

**Q3/Q4 — Tier where M1 outperforms most / M2A outperforms most:** M1 best at Slight Dog (+12.29pp); M2A best at Pick'em (+2.18pp)

**Q5 — Best strategy:** F: Agree + dog picks only (ROI +13.35%)

**Q6 — Which model to display?** M1 has higher overall ROI — consider elevating M1 as primary.

**Q7 — Does M2A-primary architecture make sense?** Debatable — M1 ROI is higher overall.

---

_Research only — no model, frontend, or backend files were modified._
