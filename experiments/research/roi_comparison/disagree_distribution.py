import pandas as pd
import numpy as np

# ── Load and join both-sided odds from master ──────────────────────────
df_log    = pd.read_csv('data/value_bet_log.csv')
df_master = pd.read_csv('data/ufc-master.csv', low_memory=False)
df_log['date']    = pd.to_datetime(df_log['date'])
df_master['date'] = pd.to_datetime(df_master['date'])

master_slim = df_master[['date','R_fighter','B_fighter','R_odds','B_odds']].dropna(
    subset=['R_odds','B_odds'])

join_r = df_log.merge(
    master_slim.rename(columns={'R_fighter':'f1_name','B_fighter':'f2_name',
                                 'R_odds':'f1_odds','B_odds':'f2_odds'}),
    on=['date','f1_name','f2_name'], how='inner')
join_b = df_log.merge(
    master_slim.rename(columns={'B_fighter':'f1_name','R_fighter':'f2_name',
                                 'B_odds':'f1_odds','R_odds':'f2_odds'}),
    on=['date','f1_name','f2_name'], how='inner')
df = pd.concat([join_r, join_b], ignore_index=True).drop_duplicates(
    subset=['date','f1_name','f2_name'])
assert len(df) == len(df_log)

# ── Reconstruct picks ──────────────────────────────────────────────────
df['m1_picks_f1']  = df['m1_prob']  > 0.5
df['m2a_picks_f1'] = df['m2a_prob'] > 0.5
df['models_agree'] = df['m1_picks_f1'] == df['m2a_picks_f1']

disagree = df[~df['models_agree']].copy()
print(f"Disagreement fights: {len(disagree)}")

# Agreement type label
disagree['agreement_type'] = np.where(
    disagree['m1_picks_f1'],
    'M1 picks F1 / M2A picks F2',
    'M1 picks F2 / M2A picks F1'
)

# ── Did M1's pick win? ─────────────────────────────────────────────────
# pick_won = 1 means VALUE fighter (gap_direction=1→f1, -1→f2) won
disagree['value_is_f1'] = disagree['gap_direction'] == 1
disagree['m1_same_as_value'] = disagree['m1_picks_f1'] == disagree['value_is_f1']
disagree['m1_pick_won'] = np.where(
    disagree['m1_same_as_value'], disagree['pick_won'], 1 - disagree['pick_won'])

# ── M1 pick odds from master join ─────────────────────────────────────
disagree['m1_pick_ml'] = np.where(
    disagree['m1_picks_f1'], disagree['f1_odds'], disagree['f2_odds'])

# ── Per-fight profit ──────────────────────────────────────────────────
def fight_profit(won, ml):
    if pd.isna(ml): return None
    if ml >= 0: return (ml / 100) if won else -1.0
    return (100 / abs(ml)) if won else -1.0

disagree['m1_profit'] = disagree.apply(
    lambda r: fight_profit(r['m1_pick_won'], r['m1_pick_ml']), axis=1)
disagree = disagree.dropna(subset=['m1_profit'])

# ══════════════════════════════════════════════════════════════════════
print(f"\n=== M1 DISAGREE PROFIT DISTRIBUTION ===")
print(f"Total fights:     {len(disagree)}")
print(f"Total profit:     {disagree['m1_profit'].sum():+.2f}")
print(f"ROI:              {disagree['m1_profit'].mean():+.4f} ({disagree['m1_profit'].mean()*100:+.2f}%)")
print(f"Win rate:         {disagree['m1_pick_won'].mean():.4f}")
print()
print(f"Profit stats:")
print(f"  Mean:           {disagree['m1_profit'].mean():+.4f}")
print(f"  Median:         {disagree['m1_profit'].median():+.4f}")
print(f"  Std dev:        {disagree['m1_profit'].std():.4f}")
print(f"  Min:            {disagree['m1_profit'].min():+.4f}")
print(f"  Max:            {disagree['m1_profit'].max():+.4f}")
print(f"  25th pct:       {disagree['m1_profit'].quantile(0.25):+.4f}")
print(f"  75th pct:       {disagree['m1_profit'].quantile(0.75):+.4f}")
print()

# ══════════════════════════════════════════════════════════════════════
print(f"=== BIG WIN ANALYSIS ===")
big_wins = disagree[disagree['m1_profit'] > 2.0]
print(f"Fights with profit > 2.0 (big dog wins): {len(big_wins)}")
print(f"Their total profit:  {big_wins['m1_profit'].sum():+.2f}")
no_big = disagree[disagree['m1_profit'] <= 2.0]
print(f"ROI without big wins: {no_big['m1_profit'].mean():+.4f} ({no_big['m1_profit'].mean()*100:+.2f}%)")
print()

print(f"Top 10 biggest wins:")
top10 = disagree.nlargest(10, 'm1_profit')[
    ['f1_name','f2_name','m1_profit','m1_pick_won','m1_pick_ml','gap_direction','agreement_type','date']]
print(top10.to_string())
print()

# ══════════════════════════════════════════════════════════════════════
print(f"=== PROFIT DISTRIBUTION BUCKETS ===")
buckets = [
    ('Big loss   (< -0.5)',       disagree['m1_profit'] < -0.5),
    ('Small loss (-0.5 to 0)',   (disagree['m1_profit'] >= -0.5) & (disagree['m1_profit'] < 0)),
    ('Small win  (0 to 0.5)',    (disagree['m1_profit'] >= 0) & (disagree['m1_profit'] < 0.5)),
    ('Med win    (0.5 to 1.0)',  (disagree['m1_profit'] >= 0.5) & (disagree['m1_profit'] < 1.0)),
    ('Big win    (1.0 to 2.0)',  (disagree['m1_profit'] >= 1.0) & (disagree['m1_profit'] < 2.0)),
    ('Huge win   (> 2.0)',        disagree['m1_profit'] >= 2.0),
]
for label, mask in buckets:
    sub = disagree[mask]
    pct = len(sub) / len(disagree) * 100
    print(f"  {label}: N={len(sub):3d} ({pct:4.1f}%)  Total profit={sub['m1_profit'].sum():+7.2f}")
print()

# ══════════════════════════════════════════════════════════════════════
print(f"=== BY AGREEMENT TYPE ===")
for atype in ['M1 picks F1 / M2A picks F2', 'M1 picks F2 / M2A picks F1']:
    sub = disagree[disagree['agreement_type'] == atype]
    if len(sub) < 10: continue
    p, b, roi = sub['m1_profit'].sum(), len(sub), sub['m1_profit'].mean()
    print(f"  {atype}")
    print(f"    N={len(sub)}  WR={sub['m1_pick_won'].mean():.3f}  ROI={roi:+.4f} ({roi*100:+.2f}%)")
    print(f"    Avg M1 pick ML: {sub['m1_pick_ml'].mean():.1f}")
    print(f"    Favs: {(sub['m1_pick_ml']<0).sum()}  Dogs: {(sub['m1_pick_ml']>0).sum()}")
print()

# ══════════════════════════════════════════════════════════════════════
print(f"=== BY M1 PICK ODDS TIER ===")
def get_tier(ml):
    if pd.isna(ml): return None
    if ml < -300:  return 'Heavy Fav'
    if ml < -150:  return 'Mod Fav'
    if ml < -110:  return 'Slight Fav'
    if ml <= 110:  return "Pick'em"
    if ml <= 200:  return 'Slight Dog'
    if ml <= 400:  return 'Mod Dog'
    return 'Heavy Dog'

tier_order = ['Heavy Fav','Mod Fav','Slight Fav',"Pick'em",'Slight Dog','Mod Dog','Heavy Dog']
disagree['m1_tier'] = disagree['m1_pick_ml'].apply(get_tier)
for tier in tier_order:
    sub = disagree[disagree['m1_tier'] == tier]
    if len(sub) < 5: continue
    print(f"  {tier:<14} N={len(sub):3d}  WR={sub['m1_pick_won'].mean():.3f}  "
          f"ROI={sub['m1_profit'].mean():+.4f} ({sub['m1_profit'].mean()*100:+.2f}%)  "
          f"Total={sub['m1_profit'].sum():+.2f}")
print()

# ══════════════════════════════════════════════════════════════════════
# Sensitivity: remove top N wins and watch ROI
print(f"=== ROI SENSITIVITY — remove top N wins ===")
sorted_profits = disagree['m1_profit'].sort_values(ascending=False).reset_index(drop=True)
for n_remove in [0, 1, 3, 5, 10, 20]:
    remaining = sorted_profits.iloc[n_remove:]
    roi = remaining.mean()
    total = remaining.sum()
    print(f"  Remove top {n_remove:2d}: N={len(remaining):3d}  Total={total:+.2f}  ROI={roi:+.4f} ({roi*100:+.2f}%)")
print()

# ══════════════════════════════════════════════════════════════════════
print(f"=== CONCLUSION ===")
total = disagree['m1_profit'].sum()
big_win_total = big_wins['m1_profit'].sum()
pct_from_big = big_win_total / total * 100 if total != 0 else 0
print(f"Total profit:                {total:+.2f}")
print(f"From big wins (>2.0):        {big_win_total:+.2f}  ({pct_from_big:.1f}% of total)")
print(f"From everything else:        {total-big_win_total:+.2f}  ({100-pct_from_big:.1f}% of total)")
if pct_from_big > 50:
    print("VERDICT: ROI is concentrated in a small number of big wins — NOISE, not stable signal")
else:
    print("VERDICT: ROI is broadly distributed — more likely to be STABLE signal")
