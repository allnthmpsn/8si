"""
ROI comparison: Model 1 vs Model 2A on historical data.
Research only. No model, frontend, or backend files are modified.

value_bet_log.csv has m1_prob and m2a_prob for every fight the log tracks.
  - closing_odds  = the value fighter's ML (one side only)
  - gap_direction = 1 if value fighter is f1, -1 if value fighter is f2
  - pick_won      = 1 if the value fighter (M2A pick) won

We join ufc-master.csv to get both fighters' closing lines (R_odds / B_odds)
so each model's pick can be priced accurately regardless of which side it's on.
"""

import os, json, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, '../../../data')
OUT  = BASE

LOG_PATH    = os.path.join(DATA, 'value_bet_log.csv')
MASTER_PATH = os.path.join(DATA, 'ufc-master.csv')

# ─────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────
print("=" * 64)
print("SETUP")
print("=" * 64)

log    = pd.read_csv(LOG_PATH)
master = pd.read_csv(MASTER_PATH, low_memory=False)

log['date']    = pd.to_datetime(log['date'])
master['date'] = pd.to_datetime(master['date'])

print(f"  value_bet_log:  {len(log):,} rows")
print(f"  Columns: {log.columns.tolist()}")
print(f"  Date range: {log['date'].min().date()} to {log['date'].max().date()}")
print(f"  NOTE: actual outcome column is 'pick_won' (not 'value_bet_won')")

# Join both fighters' odds from master (f1=Red, f2=Blue in the log)
master_slim = master[['date','R_fighter','B_fighter','R_odds','B_odds']].dropna(
    subset=['R_odds','B_odds'])

# Orientation A: f1=Red
join_r = log.merge(
    master_slim.rename(columns={'R_fighter':'f1_name','B_fighter':'f2_name',
                                 'R_odds':'f1_odds','B_odds':'f2_odds'}),
    on=['date','f1_name','f2_name'], how='inner')
join_r['join_orient'] = 'f1=Red'

# Orientation B: f1=Blue
join_b = log.merge(
    master_slim.rename(columns={'B_fighter':'f1_name','R_fighter':'f2_name',
                                 'B_odds':'f1_odds','R_odds':'f2_odds'}),
    on=['date','f1_name','f2_name'], how='inner')
join_b['join_orient'] = 'f1=Blue'

df = pd.concat([join_r, join_b], ignore_index=True).drop_duplicates(
    subset=['date','f1_name','f2_name'])

print(f"\n  join_r: {len(join_r):,}  join_b: {len(join_b):,}  union: {len(df):,} / {len(log):,}")
assert len(df) == len(log), f"Union missed {len(log)-len(df)} rows"
print("  All rows matched via master join.")

def to_implied(ml):
    if pd.isna(ml) or ml == 0: return np.nan
    return abs(ml)/(abs(ml)+100) if ml < 0 else 100/(ml+100)

def to_ml_from_implied(p):
    if p >= 1:   return -10000.0
    if p <= 0:   return  10000.0
    return -(p/(1-p))*100 if p > 0.5 else ((1-p)/p)*100

def compute_roi(pick_won_series, pick_ml_series):
    profits = []
    for won, ml in zip(pick_won_series, pick_ml_series):
        if pd.isna(ml): continue
        profit = (ml/100 if ml >= 0 else 100/abs(ml)) if won else -1.0
        profits.append(profit)
    n = len(profits)
    total = sum(profits)
    return total, n, total/n if n else 0.0

# ─────────────────────────────────────────────────────────
# STEP 1 — Reconstruct M1 and M2A picks independently
# ─────────────────────────────────────────────────────────
print()
print("=" * 64)
print("STEP 1: Reconstruct independent picks")
print("=" * 64)

# m1_prob and m2a_prob = P(f1 wins); f1 is always the Red/first corner
df['m1_picks_f1']  = df['m1_prob']  > 0.5
df['m2a_picks_f1'] = df['m2a_prob'] > 0.5

# gap_direction=1 → value fighter is f1; gap_direction=-1 → value fighter is f2
# pick_won=1 → value fighter won
df['value_is_f1'] = df['gap_direction'] == 1

# Did M1's pick win?
# Same side as value fighter → outcome = pick_won; opposite → 1 - pick_won
df['m1_same_side'] = df['m1_picks_f1'] == df['value_is_f1']
df['m1_pick_won']  = np.where(df['m1_same_side'], df['pick_won'], 1 - df['pick_won'])

# Did M2A's pick win? (M2A picks the value fighter by construction)
# → if m2a_picks_f1 == value_is_f1, outcome = pick_won; else 1-pick_won
# (should almost always equal pick_won since M2A IS the value picker)
df['m2a_same_side'] = df['m2a_picks_f1'] == df['value_is_f1']
df['m2a_pick_won']  = np.where(df['m2a_same_side'], df['pick_won'], 1 - df['pick_won'])

# Verify: M2A same_side should be true whenever m2a_prob != 0.5
print(f"  M2A same_side rate: {df['m2a_same_side'].mean():.4f}  "
      f"(should be ~1.0 — M2A IS the value picker)")
print(f"  M1 overall accuracy:  {df['m1_pick_won'].mean():.4f}")
print(f"  M2A overall accuracy: {df['m2a_pick_won'].mean():.4f}")
print(f"  M1/M2A agree:         {(df['m1_picks_f1'] == df['m2a_picks_f1']).mean():.4f} "
      f"({(df['m1_picks_f1']==df['m2a_picks_f1']).sum():,} fights)")
print(f"  Disagree:             {(df['m1_picks_f1'] != df['m2a_picks_f1']).sum():,} fights")

# ─────────────────────────────────────────────────────────
# STEP 2 — Assign closing odds to each model's pick
# ─────────────────────────────────────────────────────────
print()
print("=" * 64)
print("STEP 2: Assign closing ML to each model's pick")
print("=" * 64)

# f1_odds and f2_odds are now in df from the master join
df['m1_pick_ml']  = np.where(df['m1_picks_f1'],  df['f1_odds'], df['f2_odds'])
df['m2a_pick_ml'] = np.where(df['m2a_picks_f1'], df['f1_odds'], df['f2_odds'])

valid_m1  = df['m1_pick_ml'].notna().sum()
valid_m2a = df['m2a_pick_ml'].notna().sum()
print(f"  Fights with valid M1 odds:  {valid_m1:,}")
print(f"  Fights with valid M2A odds: {valid_m2a:,}")
print(f"  M1 pick avg ML:   {df['m1_pick_ml'].mean():.1f}")
print(f"  M2A pick avg ML:  {df['m2a_pick_ml'].mean():.1f}")
print(f"  M1  favs: {(df['m1_pick_ml']  < 0).sum():,}  dogs: {(df['m1_pick_ml']  > 0).sum():,}")
print(f"  M2A favs: {(df['m2a_pick_ml'] < 0).sum():,}  dogs: {(df['m2a_pick_ml'] > 0).sum():,}")

# ─────────────────────────────────────────────────────────
# STEP 3 — Overall ROI
# ─────────────────────────────────────────────────────────
print()
print("=" * 64)
print("STEP 3: Overall ROI — flat $1 per pick")
print("=" * 64)

m1_profit,  m1_bets,  m1_roi  = compute_roi(df['m1_pick_won'],  df['m1_pick_ml'])
m2a_profit, m2a_bets, m2a_roi = compute_roi(df['m2a_pick_won'], df['m2a_pick_ml'])

print(f"  M1  — Bets: {m1_bets:,}  Profit: {m1_profit:+.2f}  ROI: {m1_roi:+.4f} ({m1_roi*100:+.2f}%)")
print(f"  M2A — Bets: {m2a_bets:,}  Profit: {m2a_profit:+.2f}  ROI: {m2a_roi:+.4f} ({m2a_roi*100:+.2f}%)")
print(f"  ROI advantage (M1−M2A): {(m1_roi-m2a_roi)*100:+.2f}pp")

# ─────────────────────────────────────────────────────────
# STEP 4 — ROI by agreement type
# ─────────────────────────────────────────────────────────
print()
print("=" * 64)
print("STEP 4: ROI by agreement / disagreement")
print("=" * 64)

agree_mask    = df['m1_picks_f1'] == df['m2a_picks_f1']
dis_m1f1_mask = (df['m1_picks_f1'] == True)  & (df['m2a_picks_f1'] == False)
dis_m2f1_mask = (df['m1_picks_f1'] == False) & (df['m2a_picks_f1'] == True)
disagree_mask = ~agree_mask

agreement_groups = {
    'Both agree (same pick)':    agree_mask,
    'M1 picks F1 / M2A picks F2': dis_m1f1_mask,
    'M1 picks F2 / M2A picks F1': dis_m2f1_mask,
    'All disagreements':          disagree_mask,
}

agree_results = {}
print(f"\n  {'Group':<35} {'N':>5}  {'M1 WR':>7} {'M1 ROI':>8}  {'M2A WR':>7} {'M2A ROI':>8}  {'Winner':>6}")
print(f"  {'-'*35} {'-'*5}  {'-'*7} {'-'*8}  {'-'*7} {'-'*8}  {'-'*6}")
for label, mask in agreement_groups.items():
    sub = df[mask]
    if len(sub) < 5:
        continue
    _, _, m1_r  = compute_roi(sub['m1_pick_won'],  sub['m1_pick_ml'])
    _, _, m2a_r = compute_roi(sub['m2a_pick_won'], sub['m2a_pick_ml'])
    m1_wr  = sub['m1_pick_won'].mean()
    m2a_wr = sub['m2a_pick_won'].mean()
    winner = 'M1' if m1_r > m2a_r else 'M2A'
    print(f"  {label:<35} {len(sub):>5}  {m1_wr:>7.3f} {m1_r:>+8.4f}  {m2a_wr:>7.3f} {m2a_r:>+8.4f}  {winner:>6}")
    agree_results[label] = {
        'n': len(sub), 'm1_wr': round(m1_wr,4), 'm1_roi': round(m1_r,4),
        'm2a_wr': round(m2a_wr,4), 'm2a_roi': round(m2a_r,4), 'winner': winner
    }

# ─────────────────────────────────────────────────────────
# STEP 5 — ROI by odds tier
# ─────────────────────────────────────────────────────────
print()
print("=" * 64)
print("STEP 5: ROI by odds tier")
print("=" * 64)

def get_tier(ml):
    if pd.isna(ml): return None
    if ml < -300:  return 'Heavy Fav'
    if ml < -150:  return 'Mod Fav'
    if ml < -110:  return 'Slight Fav'
    if ml <= 110:  return "Pick'em"
    if ml <= 200:  return 'Slight Dog'
    if ml <= 400:  return 'Mod Dog'
    return 'Heavy Dog'

TIER_ORDER = ['Heavy Fav','Mod Fav','Slight Fav',"Pick'em",'Slight Dog','Mod Dog','Heavy Dog']

df['m1_tier']  = df['m1_pick_ml'].apply(get_tier)
df['m2a_tier'] = df['m2a_pick_ml'].apply(get_tier)

m1_tier_results  = {}
m2a_tier_results = {}

print(f"\n  M1 ROI by odds tier of M1's pick:")
print(f"  {'Tier':<14} {'N':>5}  {'WR':>7} {'ROI%':>8}")
print(f"  {'-'*14} {'-'*5}  {'-'*7} {'-'*8}")
for tier in TIER_ORDER:
    sub = df[df['m1_tier'] == tier]
    if len(sub) < 20: continue
    p, b, r = compute_roi(sub['m1_pick_won'], sub['m1_pick_ml'])
    wr = sub['m1_pick_won'].mean()
    print(f"  {tier:<14} {b:>5}  {wr:>7.3f} {r*100:>+8.2f}%")
    m1_tier_results[tier] = {'n': b, 'wr': round(wr,4), 'roi': round(r,4)}

print(f"\n  M2A ROI by odds tier of M2A's pick:")
print(f"  {'Tier':<14} {'N':>5}  {'WR':>7} {'ROI%':>8}")
print(f"  {'-'*14} {'-'*5}  {'-'*7} {'-'*8}")
for tier in TIER_ORDER:
    sub = df[df['m2a_tier'] == tier]
    if len(sub) < 20: continue
    p, b, r = compute_roi(sub['m2a_pick_won'], sub['m2a_pick_ml'])
    wr = sub['m2a_pick_won'].mean()
    print(f"  {tier:<14} {b:>5}  {wr:>7.3f} {r*100:>+8.2f}%")
    m2a_tier_results[tier] = {'n': b, 'wr': round(wr,4), 'roi': round(r,4)}

# Side-by-side tier comparison
print(f"\n  Head-to-head by tier (M1 ROI% vs M2A ROI%):")
print(f"  {'Tier':<14} {'M1 N':>6} {'M1 ROI%':>9}  {'M2A N':>6} {'M2A ROI%':>10}  {'Advantage':>10}")
print(f"  {'-'*14} {'-'*6} {'-'*9}  {'-'*6} {'-'*10}  {'-'*10}")
for tier in TIER_ORDER:
    m1r  = m1_tier_results.get(tier)
    m2ar = m2a_tier_results.get(tier)
    if not m1r and not m2ar: continue
    m1_str  = f"{m1r['roi']*100:>+9.2f}%" if m1r else f"{'N/A':>9}"
    m2a_str = f"{m2ar['roi']*100:>+10.2f}%" if m2ar else f"{'N/A':>10}"
    m1n  = str(m1r['n'])  if m1r  else '-'
    m2an = str(m2ar['n']) if m2ar else '-'
    if m1r and m2ar:
        diff = (m1r['roi'] - m2ar['roi']) * 100
        adv = f"M1 +{diff:.2f}pp" if diff > 0 else f"M2A +{-diff:.2f}pp"
    else:
        adv = '-'
    print(f"  {tier:<14} {m1n:>6} {m1_str}  {m2an:>6} {m2a_str}  {adv:>10}")

# ─────────────────────────────────────────────────────────
# STEP 6 — Temporal analysis
# ─────────────────────────────────────────────────────────
print()
print("=" * 64)
print("STEP 6: ROI by year")
print("=" * 64)

df['year'] = df['date'].dt.year
yr_results = []
print(f"\n  {'Year':>5} {'N':>5}  {'M1 WR':>7} {'M1 ROI%':>9}  {'M2A WR':>7} {'M2A ROI%':>10}  {'Gap pp':>7}")
print(f"  {'-'*5} {'-'*5}  {'-'*7} {'-'*9}  {'-'*7} {'-'*10}  {'-'*7}")
for yr in sorted(df['year'].unique()):
    sub = df[df['year'] == yr]
    if len(sub) < 30: continue
    _, _, m1_r  = compute_roi(sub['m1_pick_won'],  sub['m1_pick_ml'])
    _, _, m2a_r = compute_roi(sub['m2a_pick_won'], sub['m2a_pick_ml'])
    m1_wr  = sub['m1_pick_won'].mean()
    m2a_wr = sub['m2a_pick_won'].mean()
    gap = (m1_r - m2a_r) * 100
    print(f"  {yr:>5} {len(sub):>5}  {m1_wr:>7.3f} {m1_r*100:>+9.2f}%  {m2a_wr:>7.3f} {m2a_r*100:>+10.2f}%  {gap:>+7.2f}")
    yr_results.append({'year': int(yr), 'n': len(sub),
                        'm1_wr': round(m1_wr,4), 'm1_roi': round(m1_r,4),
                        'm2a_wr': round(m2a_wr,4), 'm2a_roi': round(m2a_r,4),
                        'gap_pp': round(gap,2)})

# ─────────────────────────────────────────────────────────
# STEP 7 — Strategy comparison
# ─────────────────────────────────────────────────────────
print()
print("=" * 64)
print("STEP 7: Strategy comparison")
print("=" * 64)

agree_df    = df[agree_mask]
disagree_df = df[disagree_mask]

# Strategy A: Always M2A
_, n_a, roi_a = compute_roi(df['m2a_pick_won'], df['m2a_pick_ml'])
acc_a = df['m2a_pick_won'].mean()

# Strategy B: Always M1
_, n_b, roi_b = compute_roi(df['m1_pick_won'], df['m1_pick_ml'])
acc_b = df['m1_pick_won'].mean()

# Strategy C: Only bet when they agree (follow M2A pick)
_, n_c, roi_c = compute_roi(agree_df['m2a_pick_won'], agree_df['m2a_pick_ml'])
acc_c = agree_df['m2a_pick_won'].mean()

# Strategy D: M1 on disagreements, M2A when they agree
d_won = pd.concat([disagree_df['m1_pick_won'],  agree_df['m2a_pick_won']])
d_ml  = pd.concat([disagree_df['m1_pick_ml'],   agree_df['m2a_pick_ml']])
_, n_d, roi_d = compute_roi(d_won, d_ml)
acc_d = d_won.mean()

# Strategy E: M2A on disagreements, M1 when they agree
e_won = pd.concat([disagree_df['m2a_pick_won'], agree_df['m1_pick_won']])
e_ml  = pd.concat([disagree_df['m2a_pick_ml'],  agree_df['m1_pick_ml']])
_, n_e, roi_e = compute_roi(e_won, e_ml)
acc_e = e_won.mean()

# Strategy F: Only bet when they agree AND pick is on underdog (|ML| < 0 or > 0?)
# Specifically: agree + dog side
agree_dog = agree_df[agree_df['m2a_pick_ml'] > 0]
_, n_f, roi_f = compute_roi(agree_dog['m2a_pick_won'], agree_dog['m2a_pick_ml'])
acc_f = agree_dog['m2a_pick_won'].mean() if len(agree_dog) else 0.0

# Strategy G: Only bet when they agree AND pick is on favorite
agree_fav = agree_df[agree_df['m2a_pick_ml'] < 0]
_, n_g, roi_g = compute_roi(agree_fav['m2a_pick_won'], agree_fav['m2a_pick_ml'])
acc_g = agree_fav['m2a_pick_won'].mean() if len(agree_fav) else 0.0

strategies = [
    ('A: Always M2A',                 n_a, acc_a, roi_a),
    ('B: Always M1',                  n_b, acc_b, roi_b),
    ('C: Agree only (M2A pick)',       n_c, acc_c, roi_c),
    ('D: M1 on splits, M2A on agrees', n_d, acc_d, roi_d),
    ('E: M2A on splits, M1 on agrees', n_e, acc_e, roi_e),
    ('F: Agree + dog picks only',      n_f, acc_f, roi_f),
    ('G: Agree + fav picks only',      n_g, acc_g, roi_g),
]

print(f"\n  {'Strategy':<38} {'N':>5}  {'Accuracy':>9} {'ROI%':>9}")
print(f"  {'-'*38} {'-'*5}  {'-'*9} {'-'*9}")
for name, n, acc, roi in strategies:
    print(f"  {name:<38} {n:>5}  {acc:>9.3f} {roi*100:>+9.2f}%")

best = max(strategies, key=lambda x: x[3])
print(f"\n  Best strategy: {best[0]}  (ROI {best[3]*100:+.2f}%)")

# ─────────────────────────────────────────────────────────
# STEP 8 — Questions answered + Save
# ─────────────────────────────────────────────────────────
print()
print("=" * 64)
print("STEP 8: Specific questions answered")
print("=" * 64)

print(f"""
Q1 — Overall ROI: M1 vs M2A
   M1:  {m1_roi*100:+.2f}%  (on {m1_bets:,} bets)
   M2A: {m2a_roi*100:+.2f}%  (on {m2a_bets:,} bets)
   Advantage: {"M1" if m1_roi > m2a_roi else "M2A"} by {abs(m1_roi-m2a_roi)*100:.2f}pp

Q2 — On disagreement fights, which model has better ROI?
   Disagree N = {disagree_df.shape[0]:,}
   M1  ROI on disagreements: {agree_results.get("All disagreements", {}).get("m1_roi", 0)*100:+.2f}%
   M2A ROI on disagreements: {agree_results.get("All disagreements", {}).get("m2a_roi", 0)*100:+.2f}%
   Winner: {agree_results.get("All disagreements", {}).get("winner", "?")}

Q3 — Which tier does M1 outperform M2A most?""")

tier_gaps = []
for tier in TIER_ORDER:
    m1r  = m1_tier_results.get(tier)
    m2ar = m2a_tier_results.get(tier)
    if m1r and m2ar:
        gap = (m1r['roi'] - m2ar['roi']) * 100
        tier_gaps.append((tier, gap))

if tier_gaps:
    best_m1_tier = max(tier_gaps, key=lambda x: x[1])
    best_m2a_tier = min(tier_gaps, key=lambda x: x[1])
    print(f"   M1 outperforms M2A most at: {best_m1_tier[0]} (M1 advantage: +{best_m1_tier[1]:.2f}pp)")
    print(f"   M2A outperforms M1 most at: {best_m2a_tier[0]} (M2A advantage: +{-best_m2a_tier[1]:.2f}pp)")

print(f"""
Q5 — Best strategy across all five options (extended to seven):
   Winner: {best[0]}  ROI={best[3]*100:+.2f}%

Q6 — Should the displayed pick be M1 or M2A based on ROI?
   Overall ROI: {"M1" if m1_roi > m2a_roi else "M2A"} wins
   On agreements: same pick, same ROI direction
   On disagreements: {agree_results.get("All disagreements", {}).get("winner", "?")} has better ROI

Q7 — Does current architecture (M2A primary, M1 as analyst) make sense?""")

if m2a_roi > m1_roi:
    print("   YES — M2A has higher overall ROI, appropriate as primary display.")
else:
    print("   PARTIALLY — M1 has higher overall ROI, but M2A likely wins on agreement fights.")
    print("   Disagreement fights are the key diagnostic.")

# Save results
results = {
    "meta": {
        "total_fights": int(len(df)),
        "date_range": [str(df['date'].min().date()), str(df['date'].max().date())],
        "m1_accuracy": round(float(df['m1_pick_won'].mean()), 4),
        "m2a_accuracy": round(float(df['m2a_pick_won'].mean()), 4),
    },
    "overall_roi": {
        "m1_bets": m1_bets, "m1_profit": round(m1_profit, 2), "m1_roi": round(m1_roi, 4),
        "m2a_bets": m2a_bets, "m2a_profit": round(m2a_profit, 2), "m2a_roi": round(m2a_roi, 4),
        "roi_gap_pp": round((m1_roi - m2a_roi) * 100, 2),
        "winner": "M1" if m1_roi > m2a_roi else "M2A",
    },
    "by_agreement": agree_results,
    "m1_by_tier": m1_tier_results,
    "m2a_by_tier": m2a_tier_results,
    "by_year": yr_results,
    "strategies": [{"name": n, "bets": bn, "accuracy": round(acc,4), "roi": round(roi,4)}
                   for n, bn, acc, roi in strategies],
}

json_path = os.path.join(OUT, 'roi_comparison_results.json')
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Saved: {json_path}")

# ── Markdown findings ──────────────────────────────────────
md_lines = [
    "# M1 vs M2A ROI Comparison — Findings",
    "",
    f"**Source:** value_bet_log.csv joined to ufc-master.csv for two-sided odds  ",
    f"**Fights:** {len(df):,}  |  **Date range:** {df['date'].min().date()} to {df['date'].max().date()}",
    "",
    "---",
    "",
    "## Overall ROI (flat $1 per pick)",
    "",
    f"| Model | Bets | Accuracy | Profit | ROI% |",
    f"|---|---|---|---|---|",
    f"| M1  | {m1_bets:,} | {df['m1_pick_won'].mean():.3f} | {m1_profit:+.2f} | {m1_roi*100:+.2f}% |",
    f"| M2A | {m2a_bets:,} | {df['m2a_pick_won'].mean():.3f} | {m2a_profit:+.2f} | {m2a_roi*100:+.2f}% |",
    f"",
    f"**ROI advantage:** {'M1' if m1_roi > m2a_roi else 'M2A'} by {abs(m1_roi-m2a_roi)*100:.2f}pp",
    "",
    "---",
    "",
    "## ROI by Agreement / Disagreement",
    "",
    "| Group | N | M1 WR | M1 ROI% | M2A WR | M2A ROI% | Winner |",
    "|---|---|---|---|---|---|---|",
]
for label, res in agree_results.items():
    md_lines.append(
        f"| {label} | {res['n']:,} | {res['m1_wr']:.3f} | {res['m1_roi']*100:+.2f}% | "
        f"{res['m2a_wr']:.3f} | {res['m2a_roi']*100:+.2f}% | {res['winner']} |"
    )

md_lines += [
    "",
    "---",
    "",
    "## ROI by Odds Tier",
    "",
    "### M1 picks",
    "",
    "| Tier | N | WR | ROI% |",
    "|---|---|---|---|",
]
for tier in TIER_ORDER:
    r = m1_tier_results.get(tier)
    if r: md_lines.append(f"| {tier} | {r['n']:,} | {r['wr']:.3f} | {r['roi']*100:+.2f}% |")

md_lines += [
    "",
    "### M2A picks",
    "",
    "| Tier | N | WR | ROI% |",
    "|---|---|---|---|",
]
for tier in TIER_ORDER:
    r = m2a_tier_results.get(tier)
    if r: md_lines.append(f"| {tier} | {r['n']:,} | {r['wr']:.3f} | {r['roi']*100:+.2f}% |")

md_lines += [
    "",
    "---",
    "",
    "## Strategy Comparison",
    "",
    "| Strategy | N | Accuracy | ROI% |",
    "|---|---|---|---|",
]
for name, n, acc, roi in strategies:
    star = " ←" if (name, n, acc, roi) == best else ""
    md_lines.append(f"| {name}{star} | {n:,} | {acc:.3f} | {roi*100:+.2f}% |")

md_lines += [
    "",
    "---",
    "",
    "## Questions Answered",
    "",
    f"**Q1 — Overall ROI winner:** {'M1' if m1_roi > m2a_roi else 'M2A'} "
    f"(M1: {m1_roi*100:+.2f}%, M2A: {m2a_roi*100:+.2f}%, gap: {abs(m1_roi-m2a_roi)*100:.2f}pp)",
    "",
    f"**Q2 — Disagreement fights winner:** {agree_results.get('All disagreements',{}).get('winner','?')}",
    "",
    f"**Q3/Q4 — Tier where M1 outperforms most / M2A outperforms most:** "
    f"M1 best at {best_m1_tier[0] if tier_gaps else 'N/A'} (+{best_m1_tier[1]:.2f}pp); "
    f"M2A best at {best_m2a_tier[0] if tier_gaps else 'N/A'} (+{-best_m2a_tier[1]:.2f}pp)",
    "",
    f"**Q5 — Best strategy:** {best[0]} (ROI {best[3]*100:+.2f}%)",
    "",
    f"**Q6 — Which model to display?** "
    f"{'M2A has higher overall ROI — current primary display is correct.' if m2a_roi > m1_roi else 'M1 has higher overall ROI — consider elevating M1 as primary.'}",
    "",
    f"**Q7 — Does M2A-primary architecture make sense?** "
    f"{'Yes — confirmed by ROI.' if m2a_roi >= m1_roi else 'Debatable — M1 ROI is higher overall.'}",
    "",
    "---",
    "",
    "_Research only — no model, frontend, or backend files were modified._",
]

md_path = os.path.join(OUT, 'ROI_COMPARISON_FINDINGS.md')
with open(md_path, 'w') as f:
    f.write('\n'.join(md_lines) + '\n')
print(f"  Saved: {md_path}")
print("\nAll steps complete.")
