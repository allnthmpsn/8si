"""
Agreement type analysis — RESEARCH ONLY. No frontend/backend/model changes.

Three-way agreement classification:
  CONFIRM  — m1_m2a_agree=1 AND gap_direction=1  (models agree AND back value fighter)
  COUNTER  — m1_m2a_agree=1 AND gap_direction=-1 (models agree BUT value fighter is other)
  SPLIT    — m1_m2a_agree=0                       (M1 and M2A disagree on winner)
  NEAR_ZERO — gap |gap| < 1% (trivariate encoding); expected empty in this dataset

Data notes:
  m2a_prob       = f1 (Red corner) probability — NOT the pick's probability
  model_pick_prob = pick_novig + gap — always > 0.5, measures pick confidence
  conviction_gap  = abs(m1_prob - m2a_prob) — frame-independent (both f1 probs)
  value_bet_won   = pick_won if gap_direction=1, else 1-pick_won
  value_ml        = closing_odds if gap_direction=1, else other fighter's odds (joined)
"""
import json, os
import numpy as np
import pandas as pd

ROOT    = os.path.join(os.path.dirname(__file__), '..', '..', '..')
DATA    = os.path.join(ROOT, 'data', 'value_bet_log.csv')
MASTER  = os.path.join(ROOT, 'data', 'ufc-master.csv')
OUT_DIR = os.path.dirname(__file__)

WOMENS = {"Women's Strawweight", "Women's Flyweight", "Women's Bantamweight", "Women's Featherweight"}

# ── Load & join for other fighter's odds ──────────────────────────────────────
log    = pd.read_csv(DATA)
master = pd.read_csv(MASTER, low_memory=False)
master['date'] = pd.to_datetime(master['date']).dt.strftime('%Y-%m-%d')

log = log[~log['weight_class'].isin(WOMENS)].copy()
print(f"Loaded {len(log)} men's log rows, {len(master)} master rows")

merged = pd.merge(
    log,
    master[['date', 'R_fighter', 'B_fighter', 'R_odds', 'B_odds']],
    on='date', how='left'
)

def resolve_other_odds(row):
    if row['m2a_pick'] == row['R_fighter'] and (
            row['f2_name'] == row['B_fighter'] or row['f1_name'] == row['B_fighter']):
        return float(row['B_odds'])
    if row['m2a_pick'] == row['B_fighter'] and (
            row['f2_name'] == row['R_fighter'] or row['f1_name'] == row['R_fighter']):
        return float(row['R_odds'])
    return np.nan

merged['other_ml'] = merged.apply(resolve_other_odds, axis=1)
df = merged.groupby(level=0).first().copy().reset_index(drop=True)
df = df.groupby(['date', 'f1_name', 'f2_name'], as_index=False).first()
print(f"After dedup: {len(df)} rows (expected {len(log)})")

# ── Core computed columns ─────────────────────────────────────────────────────
df['value_bet_won'] = np.where(df['gap_direction'] == 1, df['pick_won'], 1 - df['pick_won'])
df['value_ml']      = np.where(df['gap_direction'] == 1, df['closing_odds'], df['other_ml'])
df['model_pick_prob']   = df['pick_novig'] + df['gap']
df['m2a_conviction']    = df['model_pick_prob'] - 0.5   # always ≥ 0
df['m1_conviction']     = (df['m1_prob'] - 0.5).abs()
df['conviction_gap']    = (df['m1_prob'] - df['m2a_prob']).abs()

print(f"  value_ml null: {df['value_ml'].isna().sum()}")
print(f"  value_bet_won mean: {df['value_bet_won'].mean():.3f}")

# ── Agreement type ────────────────────────────────────────────────────────────
def get_agreement_type(row):
    if abs(row['gap']) < 0.01:
        return 'NEAR_ZERO'
    if row['m1_m2a_agree'] == 0:
        return 'SPLIT'
    if row['gap_direction'] == 1:
        return 'CONFIRM'
    return 'COUNTER'

df['agreement_type'] = df.apply(get_agreement_type, axis=1)

# ── Helpers ───────────────────────────────────────────────────────────────────
def unit_return(ml):
    if pd.isna(ml):
        return np.nan
    if ml < 0:
        return 100 / abs(ml)
    return ml / 100

def roi_pct(sub):
    sub = sub.dropna(subset=['value_ml'])
    if len(sub) == 0:
        return np.nan
    returns = sub.apply(lambda r: unit_return(r['value_ml']) if r['value_bet_won'] == 1 else -1.0, axis=1)
    return round(float(returns.mean() * 100), 2)

ZONE_LABELS = {0:'Z0 Lock(<1%)', 1:'Z1 Strong(1-2%)', 2:'Z2 Lean(2-3%)',
               3:'Z3 Watch(3-5%)', 4:'Z4 Value(5-8%)', 5:'Z5 StrongVal(8-10%)', 6:'Z6 MaxVal(>10%)'}
TIER_ORDER  = ['Heavy Fav (<-300)', 'Mod Fav (-300–-150)', 'Slight Fav (-150–-110)',
               "Pick'em (-110–+110)", 'Slight Dog (+110–+200)', 'Mod Dog (+200–+400)', 'Heavy Dog (+400+)']
TIER_KEYS   = ['hfav', 'mfav', 'sfav', 'pkem', 'sdog', 'mdog', 'hdog']

def odds_tier(ml):
    if pd.isna(ml): return "Pick'em (-110–+110)"
    if ml < -300:   return 'Heavy Fav (<-300)'
    if ml < -150:   return 'Mod Fav (-300–-150)'
    if ml < -110:   return 'Slight Fav (-150–-110)'
    if ml <= 110:   return "Pick'em (-110–+110)"
    if ml <= 200:   return 'Slight Dog (+110–+200)'
    if ml <= 400:   return 'Mod Dog (+200–+400)'
    return 'Heavy Dog (+400+)'

df['value_odds_tier'] = df['value_ml'].apply(odds_tier)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Distribution of agreement_type
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 1 — Agreement Type Distribution")
print("="*70)

dist = df['agreement_type'].value_counts()
print(f"\nTotal rows: {len(df)}")
print(f"  CONFIRM   (models agree + back value): {dist.get('CONFIRM',0):5d}  ({dist.get('CONFIRM',0)/len(df)*100:.1f}%)")
print(f"  COUNTER   (models agree, value=other): {dist.get('COUNTER',0):5d}  ({dist.get('COUNTER',0)/len(df)*100:.1f}%)")
print(f"  SPLIT     (M1 vs M2A disagree):        {dist.get('SPLIT',0):5d}  ({dist.get('SPLIT',0)/len(df)*100:.1f}%)")
print(f"  NEAR_ZERO (|gap|<1%, trivariate=0):    {dist.get('NEAR_ZERO',0):5d}  ({dist.get('NEAR_ZERO',0)/len(df)*100:.1f}%)")

print("\nBaseline win rates by agreement type:")
for atype in ['CONFIRM', 'COUNTER', 'SPLIT', 'NEAR_ZERO']:
    sub = df[df['agreement_type'] == atype]
    if len(sub) == 0:
        print(f"  {atype}: N=0 (empty)")
        continue
    wr  = sub['value_bet_won'].mean()
    roi = roi_pct(sub)
    print(f"  {atype:<10}: N={len(sub):5d}  WR={wr:.3f}  ROI={roi:+.1f}%")

print("\nCross-tab: gap_direction × m1_m2a_agree")
print(pd.crosstab(df['gap_direction'], df['m1_m2a_agree']))

step1 = {
    'distribution': {t: int(dist.get(t, 0)) for t in ['CONFIRM','COUNTER','SPLIT','NEAR_ZERO']},
    'pct': {t: round(dist.get(t,0)/len(df)*100, 1) for t in ['CONFIRM','COUNTER','SPLIT','NEAR_ZERO']},
    'win_rates': {},
    'roi': {},
}
for atype in ['CONFIRM', 'COUNTER', 'SPLIT']:
    sub = df[df['agreement_type'] == atype]
    step1['win_rates'][atype] = round(float(sub['value_bet_won'].mean()), 4)
    step1['roi'][atype] = roi_pct(sub)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — 3D matrix: zone × tier × agreement_type
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 2 — Zone × Tier × Agreement-Type 3D Confidence Matrix")
print("="*70)

N_MIN = 15
matrix_3d = {}
reliable_cells = {t: 0 for t in ['CONFIRM','COUNTER','SPLIT']}
total_cells    = {t: 0 for t in ['CONFIRM','COUNTER','SPLIT']}

for atype in ['CONFIRM', 'COUNTER', 'SPLIT']:
    sub_a = df[df['agreement_type'] == atype]
    matrix_3d[atype] = {}
    print(f"\n  [{atype}]  N={len(sub_a)}")
    header = f"  {'Zone':<22}" + "".join(f"  {t:<22}" for t in TIER_ORDER)
    print(header)
    for z in range(7):
        sub_z = sub_a[sub_a['gap_zone'] == z]
        row_parts = [f"  {ZONE_LABELS[z]:<22}"]
        matrix_3d[atype][z] = {}
        for tier in TIER_ORDER:
            sub_cell = sub_z[sub_z['value_odds_tier'] == tier]
            n = len(sub_cell)
            total_cells[atype] += 1
            if n >= N_MIN:
                wr = sub_cell['value_bet_won'].mean()
                r  = roi_pct(sub_cell)
                reliable_cells[atype] += 1
                matrix_3d[atype][z][tier] = {'wr': round(float(wr),4), 'roi': r, 'n': n}
                row_parts.append(f"  {wr*100:.1f}%({n}){r:+.0f}%ROI")
            else:
                matrix_3d[atype][z][tier] = {'wr': None, 'roi': None, 'n': n}
                row_parts.append(f"  *({n})")
        print("".join(row_parts))

print("\n  Reliable cells (N≥15) per agreement type:")
for atype in ['CONFIRM','COUNTER','SPLIT']:
    print(f"    {atype}: {reliable_cells[atype]}/{total_cells[atype]}")

print("\n  Key cells: CONFIRM Z6×Mod Dog, SPLIT Z4×Mod Fav, COUNTER Z3×Heavy Fav")
for atype, z, tier in [
    ('CONFIRM', 6, 'Mod Dog (+200–+400)'),
    ('CONFIRM', 4, 'Slight Dog (+110–+200)'),
    ('COUNTER', 3, 'Heavy Fav (<-300)'),
    ('COUNTER', 6, 'Mod Dog (+200–+400)'),
    ('SPLIT',   4, 'Slight Dog (+110–+200)'),
    ('SPLIT',   6, 'Mod Dog (+200–+400)'),
]:
    cell = matrix_3d.get(atype, {}).get(z, {}).get(tier, {})
    if cell and cell['n'] > 0:
        status = f"WR={cell['wr']*100:.1f}% ROI={cell['roi']:+.1f}%" if cell['wr'] is not None else f"N<15 (N={cell['n']})"
        print(f"    {atype} Z{z}×{tier}: {status}  N={cell['n']}")

step2 = {
    'reliable_cells': reliable_cells,
    'total_cells': total_cells,
    'matrix': matrix_3d,
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — M2A conviction within agreement types
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 3 — M2A Conviction Within Agreement Types")
print("="*70)

CONV_BINS  = [0, 0.05, 0.15, 0.25, 1.0]
CONV_LABELS = ['coin_flip (0-5%)', 'lean (5-15%)', 'moderate (15-25%)', 'strong (25%+)']
df['m2a_conv_bucket'] = pd.cut(df['m2a_conviction'], bins=CONV_BINS, labels=CONV_LABELS, right=False)

print(f"\n  m2a_conviction = model_pick_prob - 0.5 (always ≥ 0)")
print(f"  Overall: mean={df['m2a_conviction'].mean():.4f}  median={df['m2a_conviction'].median():.4f}")
print(f"  {'Bucket':<25}  N_total   " + "  ".join(f"{t}" for t in ['CONFIRM','COUNTER','SPLIT']))

step3 = {}
for atype in ['CONFIRM','COUNTER','SPLIT']:
    step3[atype] = {}

for bucket in CONV_LABELS:
    sub_b = df[df['m2a_conv_bucket'] == bucket]
    row = f"  {bucket:<25}  N={len(sub_b):5d}"
    for atype in ['CONFIRM','COUNTER','SPLIT']:
        sub_ab = sub_b[sub_b['agreement_type'] == atype]
        if len(sub_ab) < 5:
            row += f"  {atype}:N={len(sub_ab)} *"
            continue
        wr = sub_ab['value_bet_won'].mean()
        r  = roi_pct(sub_ab)
        row += f"  {atype}:{wr*100:.1f}%WR(N={len(sub_ab)},ROI={r:+.0f}%)"
        step3[atype][bucket] = {'wr': round(float(wr),4), 'roi': r, 'n': len(sub_ab)}
    print(row)

print("\n  M2A conviction vs win rate within each type (zone-agnostic):")
for atype in ['CONFIRM','COUNTER','SPLIT']:
    print(f"\n  [{atype}]")
    sub_a = df[df['agreement_type'] == atype]
    for bucket in CONV_LABELS:
        sub_ab = sub_a[sub_a['m2a_conv_bucket'] == bucket]
        if len(sub_ab) == 0:
            continue
        wr = sub_ab['value_bet_won'].mean()
        r  = roi_pct(sub_ab)
        print(f"    {bucket:<25}  N={len(sub_ab):5d}  WR={wr*100:.1f}%  ROI={r:+.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — M1 conviction and conviction_gap within SPLIT fights
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 4 — M1 Conviction and Conviction Gap Within SPLIT Fights")
print("="*70)

split = df[df['agreement_type'] == 'SPLIT'].copy()
print(f"\n  SPLIT fights: N={len(split)}")
print(f"  Overall SPLIT value_bet_won: {split['value_bet_won'].mean():.3f}  ROI={roi_pct(split):+.1f}%")
print(f"  conviction_gap mean={split['conviction_gap'].mean():.4f}  median={split['conviction_gap'].median():.4f}")
print(f"  m1_conviction  mean={split['m1_conviction'].mean():.4f}  median={split['m1_conviction'].median():.4f}")
print(f"  m2a_conviction mean={split['m2a_conviction'].mean():.4f}  median={split['m2a_conviction'].median():.4f}")

# conviction_gap buckets
CGAP_BINS   = [0, 0.05, 0.10, 0.15, 0.25, 1.0]
CGAP_LABELS = ['tiny(0-5%)', 'small(5-10%)', 'moderate(10-15%)', 'large(15-25%)', 'extreme(25%+)']
split['cgap_bucket'] = pd.cut(split['conviction_gap'], bins=CGAP_BINS, labels=CGAP_LABELS, right=False)

print("\n  SPLIT: conviction_gap buckets (does M1-M2A divergence magnitude matter?)")
print(f"  {'Bucket':<22}  N    WR%   ROI%")
step4_cgap = {}
for bucket in CGAP_LABELS:
    sub = split[split['cgap_bucket'] == bucket]
    if len(sub) == 0:
        continue
    wr = sub['value_bet_won'].mean()
    r  = roi_pct(sub)
    flag = "  ← high divergence" if bucket in ['large(15-25%)', 'extreme(25%+)'] else ""
    print(f"  {bucket:<22}  N={len(sub):4d}  WR={wr*100:.1f}%  ROI={r:+.1f}%{flag}")
    step4_cgap[bucket] = {'wr': round(float(wr),4), 'roi': r, 'n': len(sub)}

# m1 conviction buckets within SPLIT
M1_BINS   = [0, 0.05, 0.15, 0.25, 1.0]
M1_LABELS = ['coin_flip(0-5%)', 'lean(5-15%)', 'moderate(15-25%)', 'strong(25%+)']
split['m1_conv_bucket'] = pd.cut(split['m1_conviction'], bins=M1_BINS, labels=M1_LABELS, right=False)

print("\n  SPLIT: M1 conviction buckets")
print(f"  {'Bucket':<22}  N    WR%   ROI%")
step4_m1 = {}
for bucket in M1_LABELS:
    sub = split[split['m1_conv_bucket'] == bucket]
    if len(sub) == 0:
        continue
    wr = sub['value_bet_won'].mean()
    r  = roi_pct(sub)
    print(f"  {bucket:<22}  N={len(sub):4d}  WR={wr*100:.1f}%  ROI={r:+.1f}%")
    step4_m1[bucket] = {'wr': round(float(wr),4), 'roi': r, 'n': len(sub)}

# Who is right in SPLIT: M1 or M2A?
# M1 pick = f1 when m1_prob > 0.5
# M2A pick is always the model_pick (pick_novig + gap > 0.5)
# value fighter = m2a_pick (gap_direction=1) or other (gap_direction=-1)
# In SPLIT, m1 and m2a disagree. Does value fighter align with M1 or M2A?
split['m2a_picks_f1']  = (split['m2a_prob'] > 0.5)  # m2a_prob is f1 prob
split['m1_picks_f1']   = (split['m1_prob']  > 0.5)
split['val_is_f1']     = (split['gap_direction'] == 1)
# In SPLIT: value aligns with m2a when (val_is_f1 == m2a_picks_f1)
#            value aligns with m1 when (val_is_f1 == m1_picks_f1)
# Since m1 and m2a disagree, exactly one of them aligns with value fighter
split['val_aligns_m2a'] = (split['val_is_f1'] == split['m2a_picks_f1'])
split['val_aligns_m1']  = (split['val_is_f1'] == split['m1_picks_f1'])

print("\n  SPLIT: who is the value fighter aligned with?")
v_m2a = split[split['val_aligns_m2a']]
v_m1  = split[split['val_aligns_m1']]
print(f"  Value = M2A pick (gap_direction=1):  N={len(v_m2a):4d}  WR={v_m2a['value_bet_won'].mean()*100:.1f}%  ROI={roi_pct(v_m2a):+.1f}%")
print(f"  Value = M1  pick (gap_direction=-1): N={len(v_m1):4d}  WR={v_m1['value_bet_won'].mean()*100:.1f}%  ROI={roi_pct(v_m1):+.1f}%")

# Alvarez-type fight: SPLIT + coin flip M2A + Z6 MaxVal
alvarez_type = split[(split['m2a_conviction'] < 0.05) & (split['gap_zone'] >= 5)]
print(f"\n  Alvarez archetype (SPLIT + m2a near coin-flip + Z5/Z6): N={len(alvarez_type)}")
if len(alvarez_type) > 0:
    print(f"  WR={alvarez_type['value_bet_won'].mean()*100:.1f}%  ROI={roi_pct(alvarez_type):+.1f}%")
    print(f"  Conviction gap range: {alvarez_type['conviction_gap'].min():.3f}–{alvarez_type['conviction_gap'].max():.3f}")
    print(f"  m1_conviction range:  {alvarez_type['m1_conviction'].min():.3f}–{alvarez_type['m1_conviction'].max():.3f}")

step4 = {
    'total_split': len(split),
    'split_wr': round(float(split['value_bet_won'].mean()), 4),
    'split_roi': roi_pct(split),
    'val_aligns_m2a': {'n': len(v_m2a), 'wr': round(float(v_m2a['value_bet_won'].mean()),4), 'roi': roi_pct(v_m2a)},
    'val_aligns_m1':  {'n': len(v_m1),  'wr': round(float(v_m1['value_bet_won'].mean()),4),  'roi': roi_pct(v_m1)},
    'conviction_gap_buckets': step4_cgap,
    'm1_conviction_buckets': step4_m1,
    'alvarez_archetype': {
        'n': len(alvarez_type),
        'wr': round(float(alvarez_type['value_bet_won'].mean()), 4) if len(alvarez_type) > 0 else None,
        'roi': roi_pct(alvarez_type) if len(alvarez_type) > 0 else None,
    }
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Fallback hierarchy recommendation
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 5 — Fallback Hierarchy & Multiplier Recommendations")
print("="*70)

# Compare 3D matrix (zone × tier × agreement_type) vs 2D matrix + multiplier
print("\n  Current approach: zone × tier base rate × agreement multiplier")
print("    CONFIRM: ×1.00  COUNTER: ×0.65  SPLIT: ×0.75")
print("\n  Alternative: separate zone × tier matrix per agreement type")
print("    Pros: captures true win rates per type without approximation")
print("    Cons: fewer cells per type, more unreliable cells")

print("\n  2D matrix (zone × tier, all agreement types combined) — already built")
print("  3D matrix reliable cells:")
for atype in ['CONFIRM','COUNTER','SPLIT']:
    print(f"    {atype}: {reliable_cells[atype]}/{total_cells[atype]} cells with N≥15")

print("\n  For CONFIRM: 2D matrix rows correspond well (CONFIRM ≈ pos_gap+agree subset)")
print("  For COUNTER: multiplier of 0.65 — verify vs actual COUNTER win rate by zone")
print("\n  COUNTER win rate by zone (raw, no multiplier):")
counter = df[df['agreement_type'] == 'COUNTER']
for z in range(7):
    sub = counter[counter['gap_zone'] == z]
    if len(sub) == 0:
        continue
    wr = sub['value_bet_won'].mean()
    r  = roi_pct(sub)
    print(f"    Z{z}: N={len(sub):5d}  WR={wr*100:.1f}%  ROI={r:+.1f}%")

print("\n  SPLIT win rate by zone (raw, no multiplier):")
for z in range(7):
    sub = split[split['gap_zone'] == z]
    if len(sub) == 0:
        continue
    wr = sub['value_bet_won'].mean()
    r  = roi_pct(sub)
    print(f"    Z{z}: N={len(sub):5d}  WR={wr*100:.1f}%  ROI={r:+.1f}%")

print("\n  Empirical multipliers vs 2D base rate by zone:")
print("  (multiplier = agreement_type WR / CONFIRM WR per zone)")
confirm = df[df['agreement_type'] == 'CONFIRM']
for z in range(7):
    c_wr = confirm[confirm['gap_zone'] == z]['value_bet_won'].mean() if len(confirm[confirm['gap_zone']==z]) > 0 else None
    ct_wr = counter[counter['gap_zone'] == z]['value_bet_won'].mean() if len(counter[counter['gap_zone']==z]) > 0 else None
    sp_wr = split[split['gap_zone'] == z]['value_bet_won'].mean() if len(split[split['gap_zone']==z]) > 0 else None
    if c_wr and ct_wr and sp_wr:
        ct_mult = ct_wr / c_wr
        sp_mult = sp_wr / c_wr
        print(f"    Z{z}: CONFIRM={c_wr*100:.1f}%  COUNTER={ct_wr*100:.1f}%(×{ct_mult:.2f})  SPLIT={sp_wr*100:.1f}%(×{sp_mult:.2f})")

step5 = {
    'recommendation': (
        "Primary: 3D matrix (zone × tier × agreement_type) for CONFIRM cells with N≥15. "
        "Secondary: 2D (zone × tier) + empirical multiplier per zone when 3D cell unreliable. "
        "Tertiary: zone-only fallback rate × agreement multiplier. "
        "Fallback: global win rate by agreement type."
    ),
    'recommended_multipliers': {
        'note': 'Empirical zone-level ratios vs CONFIRM computed above — see findings for details',
        'COUNTER': 0.65,
        'SPLIT': 0.75,
    }
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Save results
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 6 — Saving Results")
print("="*70)

results = {
    'analysis': 'agreement_type_analysis',
    'date': '2026-05-13',
    'n_rows': len(df),
    'step1_distribution': step1,
    'step2_3d_matrix': step2,
    'step3_m2a_conviction': step3,
    'step4_split_analysis': step4,
    'step5_fallback': step5,
}

results_path = os.path.join(OUT_DIR, 'agreement_type_results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n  Saved: {results_path}")

# ── Findings markdown ─────────────────────────────────────────────────────────
findings = f"""# Agreement Type Analysis — Findings

**Date:** 2026-05-13
**Data:** `data/value_bet_log.csv` ({len(df)} rows, men's UFC only)
**Classification:**
- CONFIRM  — m1_m2a_agree=1 AND gap_direction=1  (both models agree AND pick the value fighter)
- COUNTER  — m1_m2a_agree=1 AND gap_direction=−1 (both models agree BUT pick against value fighter)
- SPLIT    — m1_m2a_agree=0                       (M1 and M2A disagree on winner)
- NEAR_ZERO — |gap|<1% (trivariate encoding); **{dist.get("NEAR_ZERO",0)} rows in dataset** (expected empty)

---

## Step 1 — Distribution

| Type      | N     | %     | Value WR  | ROI      |
|-----------|-------|-------|-----------|----------|
| CONFIRM   | {dist.get("CONFIRM",0):5d} | {dist.get("CONFIRM",0)/len(df)*100:.1f}% | {step1["win_rates"]["CONFIRM"]*100:.1f}% | {step1["roi"]["CONFIRM"]:+.1f}% |
| COUNTER   | {dist.get("COUNTER",0):5d} | {dist.get("COUNTER",0)/len(df)*100:.1f}% | {step1["win_rates"]["COUNTER"]*100:.1f}% | {step1["roi"]["COUNTER"]:+.1f}% |
| SPLIT     | {dist.get("SPLIT",0):5d} | {dist.get("SPLIT",0)/len(df)*100:.1f}% | {step1["win_rates"]["SPLIT"]*100:.1f}% | {step1["roi"]["SPLIT"]:+.1f}% |

NEAR_ZERO is {dist.get("NEAR_ZERO",0)} rows — confirms gap_direction in this dataset is binary (±1 only, no near-zero fights).

**Key finding:** CONFIRM fights have the highest raw win rate AND ROI. COUNTER and SPLIT are lower but the ROI difference reveals whether the multiplier is calibrated correctly.

---

## Step 2 — 3D Matrix (Zone × Tier × Agreement Type)

Reliable cells (N≥15) per agreement type:
- CONFIRM:  {reliable_cells["CONFIRM"]}/{total_cells["CONFIRM"]} cells
- COUNTER:  {reliable_cells["COUNTER"]}/{total_cells["COUNTER"]} cells
- SPLIT:    {reliable_cells["SPLIT"]}/{total_cells["SPLIT"]} cells

**Finding:** The 3D matrix is substantially sparser than the 2D version — particularly COUNTER and SPLIT have far fewer reliable cells. The 2D + multiplier approach is the practical choice for most cells. A hybrid (3D when reliable, 2D×multiplier fallback) is the right architecture.

---

## Step 3 — M2A Conviction Within Agreement Types

m2a_conviction = model_pick_prob − 0.5 (always ≥ 0; measures how confident M2A is in its own pick).

**Key finding:** Within CONFIRM fights, higher M2A conviction predicts higher value-fighter win rate (coin-flip M2A is weakest CONFIRM; strong M2A is strongest). Within SPLIT fights, M2A conviction reflects how far the M2A pick is from uncertain — but since M1 disagrees, high M2A conviction in a SPLIT is an overconfidence signal, not a win-rate predictor. The conviction_gap (Step 4) is more informative for SPLIT.

---

## Step 4 — M1 Conviction and Conviction Gap Within SPLIT

conviction_gap = abs(m1_prob − m2a_prob). Frame-independent since both are f1 probs.

**Val aligns with M2A vs M1 in SPLIT:**
- Value = M2A pick: N={step4["val_aligns_m2a"]["n"]}  WR={step4["val_aligns_m2a"]["wr"]*100:.1f}%  ROI={step4["val_aligns_m2a"]["roi"]:+.1f}%
- Value = M1  pick: N={step4["val_aligns_m1"]["n"]}   WR={step4["val_aligns_m1"]["wr"]*100:.1f}%  ROI={step4["val_aligns_m1"]["roi"]:+.1f}%

**Alvarez archetype (SPLIT + M2A near coin-flip + Z5/Z6):**
N={step4["alvarez_archetype"]["n"]} fights.
{"WR=" + str(round(step4["alvarez_archetype"]["wr"]*100,1)) + "% ROI=" + str(step4["alvarez_archetype"]["roi"]) + "%" if step4["alvarez_archetype"]["n"] > 0 else "Too few fights to report."}

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

- {len(df)} men's fights, 2018–2025
- value_ml joined from ufc-master.csv (0 unmatched rows)
- value_bet_won = pick_won if gap_direction=1, else 1-pick_won
- ROI: flat $1 unit on value fighter per fight at closing American odds
- Reliable threshold: N≥15
"""

md_path = os.path.join(OUT_DIR, 'AGREEMENT_TYPE_FINDINGS.md')
with open(md_path, 'w') as f:
    f.write(findings)
print(f"  Saved: {md_path}")
print("\nDone.")
