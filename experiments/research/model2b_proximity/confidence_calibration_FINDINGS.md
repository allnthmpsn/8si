# Confidence Calibration Analysis — Findings

**Date:** 2026-05-13  
**Data:** `data/value_bet_log.csv` (3,007 rows, 2018–present)  
**Method:** Value fighter = fighter with positive gap (model more confident than Vegas).  
- gap_direction=1: value fighter = M2A pick → `value_fighter_won = pick_won`
- gap_direction=-1: value fighter = other fighter → `value_fighter_won = 1 - pick_won`
- Other fighter's odds joined from `ufc-master.csv` (0 unmatched rows).

---

## Cross-Tab: Gap Zone × Value Fighter Odds Tier

Win rate% (N fights). `*` = N < 10. ROI at closing odds.

| Zone \ Tier           | Heavy Fav <-300 | Mod Fav -300–-150 | Slight Fav -150–-110 | Pick'em | Slight Dog +110–+200 | Mod Dog +200–+400 | Heavy Dog +400+ |
|-----------------------|-----------------|-------------------|---------------------|---------|---------------------|-------------------|-----------------|
| Z0 Lock (<1%)         | **81.9%** (72)   | 65.8% (120)        | 61.3% (31)           | 42.4% (33) | 31.9% (91)         | 15.5% (71)        | 10.0% (20)      |
| Z1 Strong (1-2%)      | **80.5%** (77)   | 71.6% (102)        | 68.1% (47)           | 53.6% (28) | 38.6% (88)         | 29.4% (51)        | 11.8% (17)      |
| Z2 Lean (2-3%)        | **82.8%** (58)   | 79.2% (101)        | 76.2% (42)           | 69.6% (23) | 43.0% (86)         | 25.0% (36)        | 10.0% (30)      |
| Z3 Watch (3-5%)       | **93.0%** (86)   | 82.3% (147)        | 67.2% (58)           | 53.5% (43) | 47.8% (136)        | 35.1% (57)        | **4.9%** (41)   |
| Z4 Value (5-8%)       | **100%** (46)    | 86.5% (141)        | 80.5% (87)           | 68.3% (60) | 56.7% (134)        | 40.0% (75)        | 8.6% (35)       |
| Z5 StrongVal (8-10%)  | 100%\* (7)       | **96.9%** (65)     | 81.1% (37)           | 71.4% (28) | 78.0% (50)         | 48.3% (29)        | 8.3% (12)       |
| Z6 MaxVal (>10%)      | 100%\* (2)       | **96.7%** (61)     | 87.7% (57)           | **90.7%** (75) | 77.9% (145)    | 65.3% (49)        | **20.0%** (20)  |

**46 of 49 cells reliable (N ≥ 15). Only 3 small cells: Z5×HeavyFav (7), Z5×HeavyDog (12), Z6×HeavyFav (2).**

---

## Q1 — Heavy Underdog Value Picks (+400 and above): Does Zone Matter?

**Overall: 9.7% win rate, -41.4% ROI (N=175). Zone does not rescue this.**

| Zone              | WR%   | N  | ROI%   |
|-------------------|-------|----|--------|
| Z0 Lock (<1%)     | 10.0% | 20 | -47.0% |
| Z1 Strong (1-2%)  | 11.8% | 17 | -36.8% |
| Z2 Lean (2-3%)    | 10.0% | 30 | -38.2% |
| Z3 Watch (3-5%)   | 4.9%  | 41 | -72.2% |
| Z4 Value (5-8%)   | 8.6%  | 35 | -39.7% |
| Z5 StrongVal (8-10%) | 8.3% | 12 | -50.8% |
| Z6 MaxVal (>10%)  | **20.0%** | 20 | **+21.3%** |

Z0–Z5: all below 12%, all deeply negative ROI. **Z3 is worst: 4.9%, -72.2%.** The gap zone carries almost no signal for heavy underdogs. The only exception is Z6 (MaxVal) at 20% / +21.3% ROI, but N=20 is small.

**Bottom line: Heavy dog value picks are a losing proposition in 6 of 7 zones. The confidence percentage shown for heavy dog Z4–Z5 picks (83%, 84% per CONF_RATES) is not reflecting value-fighter win rate — it's reflecting M2A pick win rate (the favorite). This is a framing mismatch, not a calibration error per se, but it creates user confusion.**

---

## Q2 — What Dominates Value Fighter Win Rate: Zone or Odds Tier?

**Odds tier dominates raw win rate. Zone dominates ROI (market efficiency).**

### Marginal win rate by gap zone (across all tiers):
| Zone              | WR%   | N   | ROI%    |
|-------------------|-------|-----|---------|
| Z0 Lock (<1%)     | 48.6% | 438 | -13.8%  |
| Z1 Strong (1-2%)  | 56.8% | 410 | +3.2%   |
| Z2 Lean (2-3%)    | 59.8% | 376 | +9.9%   |
| Z3 Watch (3-5%)   | 61.6% | 568 | +12.2%  |
| Z4 Value (5-8%)   | 67.1% | 578 | **+32.5%**  |
| Z5 StrongVal (8-10%) | 76.3% | 228 | **+53.9%** |
| Z6 MaxVal (>10%)  | 80.2% | 409 | **+82.0%** |

**Zone WR spread: 31.6pp** (48.6% → 80.2%)

### Marginal win rate by odds tier (across all zones):
| Tier                      | WR%   | N   | ROI%    |
|---------------------------|-------|-----|---------|
| Heavy Fav (<-300)         | 87.4% | 348 | +9.6%   |
| Mod Fav (-300 to -150)    | 81.0% | 737 | +20.3%  |
| Slight Fav (-150 to -110) | 75.8% | 359 | +33.3%  |
| Pick'em (-110 to +110)    | 67.9% | 290 | +36.5%  |
| Slight Dog (+110 to +200) | 53.8% | 730 | +35.5%  |
| Mod Dog (+200 to +400)    | 35.6% | 368 | +29.9%  |
| Heavy Dog (+400+)         | 9.7%  | 175 | -41.4%  |

**Tier WR spread: 77.7pp** (9.7% → 87.4%)

**Pearson r (zone_number → value WR): +0.984**  
**Pearson r (tier_rank → value WR): -0.958**

Both are near-perfect linear correlates. But the tier spread (77.7pp) is 2.5× the zone spread (31.6pp). **Odds tier is the dominant predictor of raw win rate. Zone is the dominant predictor of ROI.**

This makes complete sense: zone measures how far M2A's probability departs from Vegas. A large positive zone means Vegas is significantly mispricing the fight — which generates ROI regardless of which fighter you're betting on. But whether the fighter actually wins is mostly determined by the odds (who's actually favored).

**The CONFIDENCE section currently shows zone-based win rates that ignore this 77.7pp odds-tier spread. A +500 dog in Z4 shows 83% confidence; their actual win rate is 8.6%. The number displayed is technically M2A pick's win rate (not value fighter win rate), but the value fighter IS the underdog — the displayed confidence doesn't describe what the user is evaluating.**

---

## Q3 — Value Fighter = Model's Pick vs Model's Underdog Pick

The value fighter is M2A's pick when gap_direction=1 (1,820 fights) and the OTHER fighter when gap_direction=-1 (1,187 fights).

| Group                              | WR%   | N     | ROI%    |
|------------------------------------|-------|-------|---------|
| pos_gap (value = model pick)       | **79.8%** | 1,820 | **+29.0%** |
| neg_gap (value = other fighter)    | **38.6%** | 1,187 | **+15.0%** |

A 41pp gap in win rate, but neg_gap still shows +15% ROI because value underdogs pay out much more when they win.

### Within pos_gap by zone (value fighter = model pick):
| Zone              | WR%   | N   | ROI%   |
|-------------------|-------|-----|--------|
| Z0 Lock (<1%)     | 70.1% | 231 | +0.05% |
| Z1 Strong (1-2%)  | 73.6% | 235 | +7.2%  |
| Z2 Lean (2-3%)    | 79.9% | 214 | +20.3% |
| Z3 Watch (3-5%)   | 78.8% | 330 | +17.5% |
| Z4 Value (5-8%)   | 81.9% | 354 | +33.1% |
| Z5 StrongVal 8-10% | 85.7% | 154 | +45.3% |
| Z6 MaxVal (>10%)  | 87.7% | 302 | +73.5% |

### Within neg_gap by zone (value fighter = other fighter):
| Zone              | WR%   | N   | ROI%    |
|-------------------|-------|-----|---------|
| Z0 Lock (<1%)     | 24.6% | 207 | -29.2%  |
| Z1 Strong (1-2%)  | 34.3% | 175 | -2.3%   |
| Z2 Lean (2-3%)    | 33.3% | 162 | -3.9%   |
| Z3 Watch (3-5%)   | 37.8% | 238 | +4.8%   |
| Z4 Value (5-8%)   | 43.8% | 224 | +31.6%  |
| Z5 StrongVal 8-10% | 56.8% | 74 | +71.8%  |
| Z6 MaxVal (>10%)  | 58.9% | 107 | +105.8% |

Two very different regimes. For neg_gap fights, value fighter win rates are 24-59%, but ROI turns positive at Z3+ and is exceptional at Z5/Z6 (the ones the UI's 10% threshold already captures).

---

## Q4 — Proposed Confidence Lookup (Zone × Tier)

**46 of 49 cells reliable (N ≥ 15).** Selected cells to illustrate the range:

| Zone × Tier                          | WR%   | N   | ROI%    |
|--------------------------------------|-------|-----|---------|
| Z6 MaxVal × Mod Dog (+200–+400)      | **65.3%** | 49 | **+150.0%** |
| Z6 MaxVal × Pick'em (-110–+110)      | 90.7% | 75 | +82.7%  |
| Z6 MaxVal × Slight Dog (+110–+200)   | 77.9% | 145 | +93.3%  |
| Z4 Value × Heavy Fav (<-300)         | **100%** | 46 | +28.2%  |
| Z4 Value × Slight Dog (+110–+200)    | 56.7% | 134 | +43.1%  |
| Z4 Value × Heavy Dog (+400+)         | 8.6%  | 35  | -39.7%  |
| Z0 Lock × Heavy Fav (<-300)          | 81.9% | 72  | +0.46%  |
| Z0 Lock × Slight Dog (+110–+200)     | 31.9% | 91  | -18.1%  |
| Z3 Watch × Heavy Fav (<-300)         | 93.0% | 86  | +18.4%  |
| Z3 Watch × Heavy Dog (+400+)         | **4.9%** | 41 | **-72.2%** |

Within the same gap zone, the WR range spans from 5% to 100% across tiers. **The zone-only confidence number is meaningless without knowing who the value fighter is at the odds level.**

---

## Key Findings Summary

### 1. Heavy underdogs are systematically bad regardless of zone
175 heavy dog value picks across all zones: 9.7% WR, -41.4% ROI. Only Z6 is marginally positive (+21.3%, N=20). This is not a market inefficiency — these fighters genuinely lose 90% of the time and the odds usually reflect it correctly. Zone gives essentially no lift.

### 2. Odds tier dominates win rate; zone dominates ROI
Tier explains 2.5× more variance in raw win rate (77.7pp spread vs 31.6pp). But zone is where the actual edge lives: Z4–Z6 deliver positive ROI across ALL tiers except Heavy Dog. The market mispricing captured by zone is real and consistent — it just doesn't change who's likely to win.

### 3. The current confidence display has a framing mismatch
The CONFIDENCE section shows M2A pick win rate by zone (e.g. 83% for Z4 agree). But when the value fighter is a heavy underdog, the displayed confidence refers to the FAVORITE'S win rate (not the underdog's). This is technically accurate from M2A's perspective, but a user evaluating a +500 underdog pick in the value section will interpret 83% as that pick's win probability. It's not.

### 4. Zone × tier matrix works — 46 of 49 cells reliable
There's enough data (N≥15) to support a full zone×tier confidence lookup. The matrix is stable and shows large, meaningful variation within zones (4.9% to 100% within Z3 alone across tiers).

---

## Recommendation

**Three options, ordered by complexity:**

**Option A — Filter, don't recalibrate** (simplest, most impactful)  
Remove heavy dog picks (>+400) from BetSummary TopPicks entirely. They lose 90% of the time. Leave the current confidence % as-is since it's describing M2A pick win rate correctly. Add a caveat label when value fighter odds > +300.

**Option B — Replace confidence % with zone × tier lookup** (accurate, more complex)  
Requires knowing the value fighter's odds in the component (which the UI already computes). The matrix is reliable for 46/49 cells. Cells with N<15 could fall back to zone-only. This accurately shows the value fighter's historical win rate for the specific combination.

**Option C — Remove the confidence % from the card display** (simplest but loses signal)  
The number is meaningful for favorable tiers (83% on a Z4 favorite is real) but misleading for underdog tiers. A consistent zone label (VALUE, STRONG VALUE, MAX VALUE) without a percentage avoids the miscalibration problem. The ROI signal is in the label; the % adds noise when it refers to the wrong fighter's probability.

**Recommendation: Option A as the immediate fix, Option B as the next iteration.**  
The current confidence % is structurally correct for pos_gap fights (value=M2A pick, % = pick's win rate). The problem is neg_gap and heavy underdog cells where the displayed % is the opponent's win rate. Option A handles heavy dogs (which are just bad bets anyway). Option B handles neg_gap underdog value picks correctly.

---

## Data & Methodology

- 3,007 fights, 2018–2025, men's UFC only  
- Value fighter = fighter with positive gap (model more confident than Vegas)  
- Other fighter's odds joined from `ufc-master.csv` on (date + m2a_pick name) — 0 unmatched rows  
- ROI: flat $1 unit on value fighter per fight, American odds closing line  
- Reliable cells: N ≥ 15  
