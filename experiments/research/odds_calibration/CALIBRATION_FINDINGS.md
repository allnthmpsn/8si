# Odds Calibration Analysis — Findings

**Date:** 2026-05-19  
**Fights analyzed:** 3,007  
**Overall win rate:** 72.6%  
**Avg vig:** 4.12%  

---

## Summary

- **Systematic bias:** +0.2856 (positive = model picks win more than implied)
- **Mean absolute calibration error:** 0.2948 (29.48 pp)
- **Max absolute calibration error:** 0.7665 (76.65 pp)
- **Smoothed MAE:** 0.2872
- **Reliable ML buckets (N≥15):** 19

---

## Linear Calibration Fit

```
WR = -0.0587 × novig + 0.8158
R² = 0.0235  |  p = 0.5307  |  Residual MAE = 0.0802
```

Slope < 1 (-0.059): heavy favorites underperform their implied probability; underdogs outperform. This is consistent with the well-known longshot bias in combat sports.

Power fit: WR = 0.7464 × novig^-0.0608

---

## By ML Category

| Category | N | WinRate | NoVig | CalErr |
|---|---|---|---|---|
| Heavy Fav (≤-600) | 64 | 0.922 | 0.856 | +0.066 |
| Big Fav (-400 to -600) | 142 | 0.817 | 0.792 | +0.025 |
| Fav (-200 to -400) | 573 | 0.702 | 0.695 | +0.007 |
| Slight Fav (-100 to -200) | 812 | 0.696 | 0.563 | +0.133 |
| Pick'em (-100 to +150) | 538 | 0.673 | 0.430 | +0.242 |
| Dog (+150 to +300) | 649 | 0.733 | 0.317 | +0.417 |
| Big Dog (>+300) | 229 | 0.882 | 0.184 | +0.698 |

---

## By Weight Class

| Weight Class | N | WinRate | NoVig | CalErr | AvgVig |
|---|---|---|---|---|---|
| Lightweight | 492 | 0.732 | 0.495 | +0.237 | 0.0392 |
| Welterweight | 469 | 0.708 | 0.503 | +0.205 | 0.0428 |
| Featherweight | 431 | 0.719 | 0.486 | +0.233 | 0.0412 |
| Middleweight | 421 | 0.722 | 0.505 | +0.217 | 0.0423 |
| Bantamweight | 408 | 0.774 | 0.508 | +0.267 | 0.0415 |
| Light Heavyweight | 265 | 0.698 | 0.508 | +0.190 | 0.0415 |
| Heavyweight | 262 | 0.718 | 0.510 | +0.208 | 0.0422 |
| Flyweight | 210 | 0.729 | 0.477 | +0.251 | 0.0389 |
| Catch Weight | 49 | 0.694 | 0.497 | +0.197 | 0.0369 |

---

## Feature Recommendation

**Proposed feature:** `odds_calibration_adjustment`  
**Verdict:** RECOMMENDED

Construction: assign a signed calibration adjustment based on the ML category of the value pick.  
Positive = historical over-performance vs implied; negative = under-performance.

| Category | Adjustment |
|---|---|
| Heavy Fav (≤-600) | +0.0662 |
| Big Fav (-400 to -600) | +0.0252 |
| Fav (-200 to -400) | +0.0067 |
| Slight Fav (-100 to -200) | +0.1331 |
| Pick'em (-100 to +150) | +0.2424 |
| Dog (+150 to +300) | +0.4167 |
| Big Dog (>+300) | +0.6980 |

---

_Research only — no model, frontend, or backend files were modified._
