"""
Confidence calibration analysis: gap zone × value-fighter odds tier.

Value fighter = the fighter with the POSITIVE gap (model more confident than Vegas).
  gap_direction = 1  → value fighter = M2A pick   → value_fighter_won = pick_won
  gap_direction = -1 → value fighter = other fighter → value_fighter_won = 1 - pick_won

Other fighter's odds: joined from ufc-master.csv using date + m2a_pick name match.

Key questions:
  Q1 — Heavy underdog value picks (+400+): actual win rate by zone?
  Q2 — Is gap zone or odds tier the dominant predictor of value fighter win rate?
  Q3 — Value fighter = model's pick vs value fighter = underdog pick (model picks other fighter)?
  Q4 — Zone × tier confidence lookup: which cells are reliable (N >= 15)?
"""
import json, os
import numpy as np
import pandas as pd

ROOT    = os.path.join(os.path.dirname(__file__), '..', '..', '..')
DATA    = os.path.join(ROOT, 'data', 'value_bet_log.csv')
MASTER  = os.path.join(ROOT, 'data', 'ufc-master.csv')
OUT_DIR = os.path.dirname(__file__)

# ── Load & join ───────────────────────────────────────────────────────────────
log = pd.read_csv(DATA)
master = pd.read_csv(MASTER, low_memory=False)
master['date'] = pd.to_datetime(master['date']).dt.strftime('%Y-%m-%d')
print(f"Loaded {len(log)} log rows, {len(master)} master rows")

# Join to get both fighters' odds
merged = pd.merge(
    log,
    master[['date','R_fighter','B_fighter','R_odds','B_odds']],
    on='date', how='left'
)

# Match: m2a_pick must be R_fighter or B_fighter, and the other log fighter in the other slot
def resolve_other_odds(row):
    if row['m2a_pick'] == row['R_fighter'] and (
            row['f2_name'] == row['B_fighter'] or row['f1_name'] == row['B_fighter']):
        return float(row['B_odds'])
    if row['m2a_pick'] == row['B_fighter'] and (
            row['f2_name'] == row['R_fighter'] or row['f1_name'] == row['R_fighter']):
        return float(row['R_odds'])
    return np.nan

merged['other_ml'] = merged.apply(resolve_other_odds, axis=1)

# Keep one match per log row (take first)
df = merged.groupby(level=0).first().copy()
df = df.reset_index(drop=True)

# Deduplicate to original log length (join may have multiplied some rows)
df = df.groupby(['date','f1_name','f2_name'], as_index=False).first()
print(f"After dedup: {len(df)} rows (expected {len(log)})")

# ── Value fighter features ────────────────────────────────────────────────────
# gap_direction: 1 → value fighter = m2a_pick; -1 → value fighter = other
df['value_fighter_won'] = np.where(
    df['gap_direction'] == 1,
    df['pick_won'],
    1 - df['pick_won']
)
# Value fighter's closing odds: exact from master join
df['value_ml'] = np.where(
    df['gap_direction'] == 1,
    df['closing_odds'],   # M2A pick's odds (exact)
    df['other_ml']        # other fighter's odds (exact from master join)
)
# Whether value fighter is also M2A's predicted winner (gap_direction=1 always)
df['value_is_pick'] = (df['gap_direction'] == 1).astype(int)

print(f"  Value fighter won overall: {df['value_fighter_won'].mean():.3f}")
print(f"  Value fighter odds null: {df['value_ml'].isna().sum()}")
print(f"  Rows with value_ml: {df['value_ml'].notna().sum()}")

# ── Odds tier ─────────────────────────────────────────────────────────────────
TIER_ORDER = [
    'Heavy Fav (<-300)',
    'Mod Fav (-300 to -150)',
    'Slight Fav (-150 to -110)',
    "Pick'em (-110 to +110)",
    'Slight Dog (+110 to +200)',
    'Mod Dog (+200 to +400)',
    'Heavy Dog (+400+)',
]

def odds_tier_label(ml):
    if pd.isna(ml):      return 'Unknown'
    ml = float(ml)
    if ml < -300:        return 'Heavy Fav (<-300)'
    elif ml < -150:      return 'Mod Fav (-300 to -150)'
    elif ml < -110:      return 'Slight Fav (-150 to -110)'
    elif ml <= 110:      return "Pick'em (-110 to +110)"
    elif ml <= 200:      return 'Slight Dog (+110 to +200)'
    elif ml <= 400:      return 'Mod Dog (+200 to +400)'
    else:                return 'Heavy Dog (+400+)'

df['value_odds_tier'] = df['value_ml'].apply(odds_tier_label)

print("\nOdds tier distribution for value fighter:")
for tier in TIER_ORDER:
    n = (df['value_odds_tier'] == tier).sum()
    print(f"  {tier:35s}: {n:4d}")

# ── ROI helper ────────────────────────────────────────────────────────────────
def unit_return(ml):
    """Net profit per $1 unit bet (positive = win, negative = loss per unit)."""
    ml = float(ml)
    if ml > 0:  return ml / 100.0
    else:       return 100.0 / abs(ml)

def roi_pct(sub):
    valid = sub[sub['value_ml'].notna()]
    if len(valid) == 0: return None
    profits = valid.apply(
        lambda r: unit_return(r['value_ml']) if r['value_fighter_won'] == 1 else -1.0,
        axis=1
    )
    return round(float(profits.mean()) * 100, 2)

def wr(sub):
    if len(sub) == 0: return None
    return round(float(sub['value_fighter_won'].mean()) * 100, 1)

def avg_odds(sub):
    valid = sub[sub['value_ml'].notna()]
    if len(valid) == 0: return None
    return round(float(valid['value_ml'].median()), 0)

# ── Gap zone labels ───────────────────────────────────────────────────────────
ZONE_LABELS = {
    0: 'Lock (<1%)',
    1: 'Strong (1-2%)',
    2: 'Lean (2-3%)',
    3: 'Watch (3-5%)',
    4: 'Value (5-8%)',
    5: 'StrongVal (8-10%)',
    6: 'MaxVal (>10%)',
}

# ── Cross-tab: gap zone × odds tier ──────────────────────────────────────────
print("\n" + "="*80)
print("CROSS-TAB: gap zone × value fighter odds tier")
print("Format: WR% (N) [ROI%]  — cells with N<10 marked *")
print("="*80)

cross_results = {}
for zone in range(7):
    zone_sub = df[df['gap_zone'] == zone]
    cross_results[zone] = {}
    row_parts = [f"Z{zone} {ZONE_LABELS[zone]:20s}"]
    for tier in TIER_ORDER:
        cell = zone_sub[zone_sub['value_odds_tier'] == tier]
        n = len(cell)
        cell_wr = wr(cell)
        cell_roi = roi_pct(cell)
        cell_odds = avg_odds(cell)
        reliable = n >= 10
        flag = '' if reliable else '*'
        cross_results[zone][tier] = {
            'N':          n,
            'win_rate':   cell_wr,
            'roi_pct':    cell_roi,
            'avg_odds':   cell_odds,
            'reliable':   reliable,
        }
        if n > 0:
            row_parts.append(f"{cell_wr}%({n}){flag}")
        else:
            row_parts.append("—")
    print("  " + "  |  ".join(row_parts))

# ── Q1: Heavy underdogs (+400+) across zones ─────────────────────────────────
print("\n" + "="*80)
print("Q1: Heavy Dog (+400+) value picks — win rate by zone")
print("="*80)
heavy_dog = df[df['value_odds_tier'] == 'Heavy Dog (+400+)']
print(f"  Overall Heavy Dog: WR={wr(heavy_dog)}%  N={len(heavy_dog)}  ROI={roi_pct(heavy_dog)}%")
q1_rows = {}
for zone in range(7):
    cell = heavy_dog[heavy_dog['gap_zone'] == zone]
    q1_rows[zone] = {'zone_label': ZONE_LABELS[zone], 'N': len(cell), 'win_rate': wr(cell), 'roi_pct': roi_pct(cell)}
    if len(cell) > 0:
        print(f"  Z{zone} {ZONE_LABELS[zone]:20s}: WR={wr(cell)}%  N={len(cell)}  ROI={roi_pct(cell)}%")

# ── Q2: Is gap zone or odds tier dominant? ────────────────────────────────────
print("\n" + "="*80)
print("Q2: Marginal WR by gap zone (all tiers) vs by odds tier (all zones)")
print("="*80)
print("\n  By gap zone:")
zone_marginal = {}
for zone in range(7):
    sub = df[df['gap_zone'] == zone]
    zone_marginal[zone] = {'N': len(sub), 'win_rate': wr(sub), 'roi_pct': roi_pct(sub)}
    print(f"  Z{zone} {ZONE_LABELS[zone]:20s}: WR={wr(sub)}%  N={len(sub)}  ROI={roi_pct(sub)}%")

print("\n  By odds tier:")
tier_marginal = {}
for tier in TIER_ORDER:
    sub = df[df['value_odds_tier'] == tier]
    tier_marginal[tier] = {'N': len(sub), 'win_rate': wr(sub), 'roi_pct': roi_pct(sub), 'avg_odds': avg_odds(sub)}
    print(f"  {tier:35s}: WR={wr(sub)}%  N={len(sub)}  ROI={roi_pct(sub)}%  med_odds={avg_odds(sub)}")

# Range of WRs
zone_wrs = [v['win_rate'] for v in zone_marginal.values() if v['win_rate'] is not None]
tier_wrs = [v['win_rate'] for v in tier_marginal.values() if v['win_rate'] is not None and tier_marginal[list(tier_marginal.keys())[list(tier_marginal.values()).index(v)]]['N'] > 20]

print(f"\n  Gap zone WR range: {min(zone_wrs):.1f}% – {max(zone_wrs):.1f}%  (spread: {max(zone_wrs)-min(zone_wrs):.1f}pp)")

tier_wrs2 = [(t, v['win_rate']) for t,v in tier_marginal.items() if v['win_rate'] is not None and v['N'] > 20]
tier_wr_vals = [x[1] for x in tier_wrs2]
print(f"  Odds tier WR range: {min(tier_wr_vals):.1f}% – {max(tier_wr_vals):.1f}%  (spread: {max(tier_wr_vals)-min(tier_wr_vals):.1f}pp, tiers with N>20 only)")

# Pearson correlation: WR ~ zone_number and WR ~ tier_implied_prob
zone_corr_data = [(zone, df[df['gap_zone']==zone]['value_fighter_won'].mean())
                  for zone in range(7) if (df['gap_zone']==zone).sum() > 20]
tier_prob_corr = []
for tier in TIER_ORDER:
    sub = df[df['value_odds_tier'] == tier]
    if len(sub) > 20:
        tier_prob_corr.append((TIER_ORDER.index(tier), sub['value_fighter_won'].mean()))

if len(zone_corr_data) > 2:
    zx = [x[0] for x in zone_corr_data]; zy = [x[1] for x in zone_corr_data]
    zone_r = np.corrcoef(zx, zy)[0,1]
    print(f"\n  Pearson r (zone_number → value WR): {zone_r:.3f}")

if len(tier_prob_corr) > 2:
    tx = [x[0] for x in tier_prob_corr]; ty = [x[1] for x in tier_prob_corr]
    tier_r = np.corrcoef(tx, ty)[0,1]
    print(f"  Pearson r (tier_rank → value WR): {tier_r:.3f}")
    print("  (tier_rank 0=HeavyFav, 6=HeavyDog — positive r = underdogs win more)")

# ── Q3: Value fighter = model's pick vs model's underdog pick ─────────────────
print("\n" + "="*80)
print("Q3: Value fighter = model's pick (gap_dir=1) vs model's other pick (gap_dir=-1)")
print("="*80)
pos_gap = df[df['gap_direction'] == 1]
neg_gap = df[df['gap_direction'] == -1]
near_z  = df[df['gap_direction'] == 0]

print(f"  pos_gap (value=model pick):   WR={wr(pos_gap)}%  N={len(pos_gap)}  ROI={roi_pct(pos_gap)}%")
print(f"  neg_gap (value=other fighter): WR={wr(neg_gap)}%  N={len(neg_gap)}  ROI={roi_pct(neg_gap)}%")
print(f"  near_zero (|gap|<1%):          WR={wr(near_z)}%  N={len(near_z)}  ROI={roi_pct(near_z)}%")

print("\n  By zone within pos_gap (value=model pick):")
pos_by_zone = {}
for zone in range(7):
    sub = pos_gap[pos_gap['gap_zone'] == zone]
    pos_by_zone[zone] = {'N': len(sub), 'win_rate': wr(sub), 'roi_pct': roi_pct(sub)}
    if len(sub) > 0:
        print(f"    Z{zone} {ZONE_LABELS[zone]:20s}: WR={wr(sub)}%  N={len(sub)}  ROI={roi_pct(sub)}%")

print("\n  By zone within neg_gap (value=other fighter):")
neg_by_zone = {}
for zone in range(7):
    sub = neg_gap[neg_gap['gap_zone'] == zone]
    neg_by_zone[zone] = {'N': len(sub), 'win_rate': wr(sub), 'roi_pct': roi_pct(sub)}
    if len(sub) > 0:
        print(f"    Z{zone} {ZONE_LABELS[zone]:20s}: WR={wr(sub)}%  N={len(sub)}  ROI={roi_pct(sub)}%")

# ── Q4: Confidence lookup matrix (zone × tier, N>=15) ────────────────────────
print("\n" + "="*80)
print("Q4: Confidence matrix (zone × tier, N>=15 = reliable)")
print("="*80)
confidence_matrix = {}
reliable_count = 0
for zone in range(7):
    zone_sub = df[df['gap_zone'] == zone]
    for tier in TIER_ORDER:
        cell = zone_sub[zone_sub['value_odds_tier'] == tier]
        n = len(cell)
        key = f"Z{zone}_{tier}"
        confidence_matrix[key] = {
            'zone': zone,
            'zone_label': ZONE_LABELS[zone],
            'tier': tier,
            'win_rate': wr(cell),
            'N': n,
            'roi_pct': roi_pct(cell),
            'avg_odds': avg_odds(cell),
            'reliable': n >= 15,
        }
        if n >= 15:
            reliable_count += 1
            print(f"  Z{zone} {ZONE_LABELS[zone]:20s} | {tier:35s}: WR={wr(cell)}%  N={n}  ROI={roi_pct(cell)}%  ✓")
        elif n >= 5:
            print(f"  Z{zone} {ZONE_LABELS[zone]:20s} | {tier:35s}: WR={wr(cell)}%  N={n}  ROI={roi_pct(cell)}%  (small N)")

print(f"\n  Reliable cells (N>=15): {reliable_count} of {7*7} = {reliable_count/(7*7)*100:.0f}%")

# ── Save JSON ─────────────────────────────────────────────────────────────────
results = {
    'dataset': {
        'source': 'data/value_bet_log.csv',
        'total_rows': len(df),
        'date_range': f"{df['date'].min()} to {df['date'].max()}",
    },
    'cross_tab_zone_x_tier': {
        f"Z{z}_{ZONE_LABELS[z]}": {
            tier: cross_results[z][tier] for tier in TIER_ORDER
        }
        for z in range(7)
    },
    'zone_marginal':    zone_marginal,
    'tier_marginal':    tier_marginal,
    'q1_heavy_dog_by_zone': {
        'overall': {'N': len(heavy_dog), 'win_rate': wr(heavy_dog), 'roi_pct': roi_pct(heavy_dog)},
        'by_zone': q1_rows,
    },
    'q2_dominance': {
        'zone_wr_range_pp': round(max(zone_wrs) - min(zone_wrs), 1),
        'tier_wr_range_pp': round(max(tier_wr_vals) - min(tier_wr_vals), 1) if tier_wr_vals else None,
    },
    'q3_direction': {
        'pos_gap': {'N': len(pos_gap), 'win_rate': wr(pos_gap), 'roi_pct': roi_pct(pos_gap), 'by_zone': pos_by_zone},
        'neg_gap': {'N': len(neg_gap), 'win_rate': wr(neg_gap), 'roi_pct': roi_pct(neg_gap), 'by_zone': neg_by_zone},
        'near_zero': {'N': len(near_z), 'win_rate': wr(near_z), 'roi_pct': roi_pct(near_z)},
    },
    'q4_confidence_matrix': confidence_matrix,
}

out_json = os.path.join(OUT_DIR, 'confidence_calibration_results.json')
with open(out_json, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_json}")
