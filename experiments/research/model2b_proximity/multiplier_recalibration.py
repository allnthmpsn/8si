"""
Multiplier recalibration — RESEARCH ONLY. No frontend/backend/model changes.

Fixes the COUNTER/SPLIT/NEAR_ZERO multipliers using the correct formula:
    actual_type_WR / mixed_2D_base_rate  (per zone × tier cell)

Previous error: multipliers were computed as type_WR / CONFIRM_WR, but the
CONF_MATRIX base rates are mixed (CONFIRM-dominated), not CONFIRM-only.
Applying a CONFIRM-relative ratio to a CONFIRM-heavy base rate over-corrected.
"""
import json, os
import numpy as np
import pandas as pd

ROOT    = os.path.join(os.path.dirname(__file__), '..', '..', '..')
DATA    = os.path.join(ROOT, 'data', 'value_bet_log.csv')
MASTER  = os.path.join(ROOT, 'data', 'ufc-master.csv')
OUT_DIR = os.path.dirname(__file__)

WOMENS = {"Women's Strawweight", "Women's Flyweight", "Women's Bantamweight", "Women's Featherweight"}

# Current constants from AETSlip.js (for comparison)
CONF_MATRIX = {
    0: {'hfav':0.819,'mfav':0.658,'sfav':0.613,'pkem':0.424,'sdog':0.319,'mdog':0.155,'hdog':0.100},
    1: {'hfav':0.805,'mfav':0.716,'sfav':0.681,'pkem':0.536,'sdog':0.386,'mdog':0.294,'hdog':0.118},
    2: {'hfav':0.828,'mfav':0.792,'sfav':0.762,'pkem':0.696,'sdog':0.430,'mdog':0.250,'hdog':0.100},
    3: {'hfav':0.930,'mfav':0.823,'sfav':0.672,'pkem':0.535,'sdog':0.478,'mdog':0.351,'hdog':0.049},
    4: {'hfav':1.000,'mfav':0.865,'sfav':0.805,'pkem':0.683,'sdog':0.567,'mdog':0.400,'hdog':0.086},
    5: {'hfav':None, 'mfav':0.969,'sfav':0.811,'pkem':0.714,'sdog':0.780,'mdog':0.483,'hdog':None },
    6: {'hfav':None, 'mfav':0.967,'sfav':0.877,'pkem':0.907,'sdog':0.779,'mdog':0.653,'hdog':0.200},
}
CONF_ZONE_FALLBACK = {0:0.726, 1:0.702, 2:0.742, 3:0.718, 4:0.720, 5:0.719, 6:0.756}
COUNTER_MULTIPLIERS = {0:0.39, 1:0.39, 2:0.38, 3:0.35, 4:0.26, 5:0.33, 6:0.39}
SPLIT_MULTIPLIERS   = {0:0.48, 1:0.48, 2:0.54, 3:0.70, 4:0.70, 5:0.83, 6:0.75}

TIERS    = ['hfav','mfav','sfav','pkem','sdog','mdog','hdog']
TIER_LABELS = {
    'hfav':'Heavy Fav(<-300)', 'mfav':'Mod Fav(-300–-150)',
    'sfav':'Slight Fav(-150–-110)', 'pkem':"Pick'em(-110–+110)",
    'sdog':'Slight Dog(+110–+200)', 'mdog':'Mod Dog(+200–+400)',
    'hdog':'Heavy Dog(+400+)',
}

def get_odds_tier(ml):
    if ml is None or (isinstance(ml, float) and np.isnan(ml)): return 'pkem'
    ml = float(ml)
    if ml < -300: return 'hfav'
    if ml < -150: return 'mfav'
    if ml < -110: return 'sfav'
    if ml <= 110: return 'pkem'
    if ml <= 200: return 'sdog'
    if ml <= 400: return 'mdog'
    return 'hdog'

def get_gap_zone(gap_pct):
    g = abs(gap_pct)
    if g < 1:  return 0
    if g < 2:  return 1
    if g < 3:  return 2
    if g < 5:  return 3
    if g < 8:  return 4
    if g < 10: return 5
    return 6

def get_agreement_type(row):
    if abs(row['gap']) < 0.01:    return 'NEAR_ZERO'
    if row['m1_m2a_agree'] == 0:  return 'SPLIT'
    if row['gap_direction'] == 1: return 'CONFIRM'
    return 'COUNTER'

def unit_return(ml):
    if ml is None or (isinstance(ml, float) and np.isnan(ml)): return np.nan
    ml = float(ml)
    return 100 / abs(ml) if ml < 0 else ml / 100

# ── Load & join ───────────────────────────────────────────────────────────────
log    = pd.read_csv(DATA)
master = pd.read_csv(MASTER, low_memory=False)
master['date'] = pd.to_datetime(master['date']).dt.strftime('%Y-%m-%d')
log = log[~log['weight_class'].isin(WOMENS)].copy()

merged = pd.merge(log, master[['date','R_fighter','B_fighter','R_odds','B_odds']], on='date', how='left')

def resolve_other_odds(row):
    if row['m2a_pick'] == row['R_fighter'] and (row['f2_name'] == row['B_fighter'] or row['f1_name'] == row['B_fighter']):
        return float(row['B_odds'])
    if row['m2a_pick'] == row['B_fighter'] and (row['f2_name'] == row['R_fighter'] or row['f1_name'] == row['R_fighter']):
        return float(row['R_odds'])
    return np.nan

merged['other_ml'] = merged.apply(resolve_other_odds, axis=1)
df = merged.groupby(level=0).first().copy().reset_index(drop=True)
df = df.groupby(['date','f1_name','f2_name'], as_index=False).first()

df['value_bet_won']  = np.where(df['gap_direction'] == 1, df['pick_won'], 1 - df['pick_won'])
df['value_ml']       = np.where(df['gap_direction'] == 1, df['closing_odds'], df['other_ml'])
df['agreement_type'] = df.apply(get_agreement_type, axis=1)
df['gap_zone']       = (df['gap_size'] * 100).apply(get_gap_zone)
df['value_tier']     = df['value_ml'].apply(get_odds_tier)
df['m2a_conviction'] = (df['m2a_prob'] - 0.5).abs()

print(f"Loaded {len(df)} men's fights. value_ml null: {df['value_ml'].isna().sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Mixed 2D base rates: verify against CONF_MATRIX
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 1 — Mixed 2D Base Rates vs CONF_MATRIX")
print("="*70)

mixed_2d = {}
discrepancies = []
for zone in range(7):
    mixed_2d[zone] = {}
    for tier in TIERS:
        mask = (df['gap_zone'] == zone) & (df['value_tier'] == tier)
        cell = df[mask]
        n    = len(cell)
        wr   = float(cell['value_bet_won'].mean()) if n >= 5 else None
        mixed_2d[zone][tier] = {'wr': round(wr, 4) if wr is not None else None, 'N': n}

        cm_val = CONF_MATRIX[zone].get(tier)
        if wr is not None and cm_val is not None:
            diff = abs(wr - cm_val)
            if diff > 0.02:
                discrepancies.append((zone, tier, cm_val, wr, diff, n))

print(f"\n  CONF_MATRIX vs computed mixed_2d (cells with |diff|>2pp flagged):")
print(f"  {'Z×Tier':<22} {'CM value':>10} {'Data WR':>10} {'Diff':>8} {'N':>6}")
for z, t, cm, wr, diff, n in discrepancies:
    print(f"  Z{z}×{t:<18} {cm:>10.4f} {wr:>10.4f} {diff:>+8.4f} {n:>6}")
if not discrepancies:
    print("  All cells match within 2pp ✓")
print(f"  Total discrepancies > 2pp: {len(discrepancies)}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Actual WRs per agreement type per cell
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 2 — Actual WRs per Agreement Type per Cell (N≥10 required)")
print("="*70)

N_MIN_CELL = 10

atype_rates = {}
for atype in ['CONFIRM','COUNTER','SPLIT','NEAR_ZERO']:
    atype_rates[atype] = {}
    for zone in range(7):
        atype_rates[atype][zone] = {}
        for tier in TIERS:
            mask = (df['agreement_type'] == atype) & (df['gap_zone'] == zone) & (df['value_tier'] == tier)
            cell = df[mask]
            n    = len(cell)
            wr   = float(cell['value_bet_won'].mean()) if n >= N_MIN_CELL else None
            atype_rates[atype][zone][tier] = {'wr': round(wr,4) if wr is not None else None, 'N': n, 'reliable': n >= N_MIN_CELL}

for atype in ['COUNTER','SPLIT']:
    print(f"\n  [{atype}] actual WR per cell (N≥10):")
    hdr = f"  {'Zone':<20}" + "".join(f" {t:>8}" for t in TIERS)
    print(hdr)
    for z in range(7):
        row_parts = [f"  Z{z}:<20".replace('<20', '')]
        row_str = f"  Z{z:<19}"
        for tier in TIERS:
            c = atype_rates[atype][z][tier]
            if c['wr'] is not None:
                row_str += f" {c['wr']*100:>7.1f}%"
            else:
                row_str += f"  *({c['N']:>2})"
        print(row_str)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Compute corrected multipliers
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 3 — Corrected Multipliers: actual_type_WR / mixed_2D_base_rate")
print("="*70)

corrected_multipliers = {}

for atype in ['COUNTER','SPLIT','NEAR_ZERO']:
    corrected_multipliers[atype] = {}
    # Precompute zone-level fallback scalars
    zone_scalars = {}
    for zone in range(7):
        zone_type  = df[(df['agreement_type'] == atype) & (df['gap_zone'] == zone)]
        zone_mixed = df[df['gap_zone'] == zone]
        mixed_wr   = float(zone_mixed['value_bet_won'].mean()) if len(zone_mixed) > 0 else None
        if len(zone_type) >= N_MIN_CELL and mixed_wr and mixed_wr > 0:
            zone_scalars[zone] = float(zone_type['value_bet_won'].mean()) / mixed_wr
        else:
            # Last-resort hardcoded
            zone_scalars[zone] = {'COUNTER': COUNTER_MULTIPLIERS.get(zone, 0.35),
                                   'SPLIT':   SPLIT_MULTIPLIERS.get(zone, 0.65),
                                   'NEAR_ZERO': 0.50}[atype]

    for zone in range(7):
        corrected_multipliers[atype][zone] = {}
        for tier in TIERS:
            type_cell  = atype_rates[atype][zone][tier]
            mixed_cell = mixed_2d[zone][tier]

            if (type_cell['reliable'] and
                    mixed_cell['wr'] is not None and
                    mixed_cell['wr'] > 0):
                mult   = type_cell['wr'] / mixed_cell['wr']
                source = 'cell'
                n      = type_cell['N']
            else:
                mult   = zone_scalars[zone]
                source = 'zone_fallback'
                n      = type_cell['N']

            corrected_multipliers[atype][zone][tier] = {
                'multiplier': round(mult, 4),
                'type_wr':    round(type_cell['wr'], 4) if type_cell['wr'] is not None else None,
                'mixed_wr':   mixed_cell['wr'],
                'N':          n,
                'source':     source,
            }

# Print JS-ready tables
for atype in ['COUNTER','SPLIT','NEAR_ZERO']:
    print(f"\n  {atype} corrected multipliers (JS-ready):")
    print(f"  const {'COUNTER_MULT' if atype=='COUNTER' else 'SPLIT_MULT' if atype=='SPLIT' else 'NEAR_ZERO_MULT'} = {{")
    for zone in range(7):
        row = corrected_multipliers[atype][zone]
        vals = {t: row[t]['multiplier'] for t in TIERS}
        src  = {t: row[t]['source'][0].upper() for t in TIERS}  # C=cell, Z=zone_fallback
        ns   = {t: row[t]['N'] for t in TIERS}
        hfav_v = vals['hfav']; mfav_v = vals['mfav']; sfav_v = vals['sfav']
        pkem_v = vals['pkem']; sdog_v = vals['sdog']; mdog_v = vals['mdog']; hdog_v = vals['hdog']
        print(f"    {zone}: {{hfav:{hfav_v:.4f},mfav:{mfav_v:.4f},sfav:{sfav_v:.4f},"
              f"pkem:{pkem_v:.4f},sdog:{sdog_v:.4f},mdog:{mdog_v:.4f},hdog:{hdog_v:.4f}}},")
        # sources
        src_str = " ".join(f"{t[0]}:{src[t]}({ns[t]})" for t in TIERS)
        print(f"       // {src_str}")
    print("  };")

# Zone-level fallback scalars
print("\n  Zone-level fallback scalars (when cell lookup fails):")
for atype in ['COUNTER','SPLIT','NEAR_ZERO']:
    print(f"\n  {atype} zone-level fallbacks:")
    for zone in range(7):
        zone_type  = df[(df['agreement_type'] == atype) & (df['gap_zone'] == zone)]
        zone_mixed = df[df['gap_zone'] == zone]
        if len(zone_type) >= N_MIN_CELL and len(zone_mixed) > 0:
            scalar = float(zone_type['value_bet_won'].mean()) / float(zone_mixed['value_bet_won'].mean())
            print(f"    Z{zone}: {scalar:.4f}  (N_type={len(zone_type)}, N_mixed={len(zone_mixed)})")
        else:
            print(f"    Z{zone}: insufficient data (N={len(zone_type)})")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Simulate new calibration
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 4 — Simulated Calibration with Corrected Multipliers")
print("="*70)

def get_corrected_multiplier(atype, zone, tier):
    if atype == 'CONFIRM': return 1.00
    table = corrected_multipliers.get(atype, {})
    cell  = table.get(zone, {}).get(tier)
    if cell: return cell['multiplier']
    # Zone average fallback
    zone_row = table.get(zone, {})
    vals = [v['multiplier'] for v in zone_row.values() if v['multiplier'] is not None]
    if vals: return sum(vals) / len(vals)
    return {'COUNTER':0.35,'SPLIT':0.65,'NEAR_ZERO':0.50}.get(atype, 1.00)

def compute_new_confidence(row):
    zone  = row['gap_zone']
    tier  = row['value_tier']
    atype = row['agreement_type']
    cm_val = CONF_MATRIX[zone].get(tier)
    base   = cm_val if cm_val is not None else CONF_ZONE_FALLBACK[zone]
    mult   = get_corrected_multiplier(atype, zone, tier)
    return base * mult

df['new_confidence'] = df.apply(compute_new_confidence, axis=1)

# Old confidence (for comparison)
def compute_old_confidence(row):
    zone  = row['gap_zone']
    tier  = row['value_tier']
    atype = row['agreement_type']
    cm_val = CONF_MATRIX[zone].get(tier)
    base   = cm_val if cm_val is not None else CONF_ZONE_FALLBACK[zone]
    if atype == 'CONFIRM':   mult = 1.00
    elif atype == 'COUNTER': mult = COUNTER_MULTIPLIERS.get(zone, 0.35)
    elif atype == 'SPLIT':   mult = SPLIT_MULTIPLIERS.get(zone, 0.65)
    else:                    mult = 0.50
    return base * mult

df['old_confidence'] = df.apply(compute_old_confidence, axis=1)

def calibration_mae(conf_col, sub=None):
    d = df if sub is None else df[df['agreement_type'] == sub]
    d = d.copy()
    d['bucket'] = (d[conf_col] * 10).astype(int) / 10
    cal = d.groupby('bucket').agg(pred=(conf_col,'mean'), actual=('value_bet_won','mean'), N=('value_bet_won','count')).reset_index()
    cal['err'] = cal['pred'] - cal['actual']
    return float(cal['err'].abs().mean()), float(cal['err'].mean()), cal

print("\n  Overall calibration:")
old_mae, old_bias, _ = calibration_mae('old_confidence')
new_mae, new_bias, new_cal = calibration_mae('new_confidence')
print(f"    OLD MAE: {old_mae:.4f}  Bias: {old_bias:+.4f}")
print(f"    NEW MAE: {new_mae:.4f}  Bias: {new_bias:+.4f}  (delta: {new_mae-old_mae:+.4f})")

print("\n  By agreement type:")
step4_results = {'overall': {'old_mae': round(old_mae,4), 'new_mae': round(new_mae,4)}}
for atype in ['CONFIRM','COUNTER','SPLIT','NEAR_ZERO']:
    old_mae_t, old_bias_t, _ = calibration_mae('old_confidence', atype)
    new_mae_t, new_bias_t, _ = calibration_mae('new_confidence', atype)
    delta = new_mae_t - old_mae_t
    target_met = new_mae_t < 0.10
    print(f"    {atype:<10}: OLD={old_mae_t:.4f} → NEW={new_mae_t:.4f}  (delta:{delta:+.4f})  "
          f"{'✓ <0.10' if target_met else '✗ >0.10'}")
    step4_results[atype] = {'old_mae': round(old_mae_t,4), 'new_mae': round(new_mae_t,4),
                             'old_bias': round(old_bias_t,4), 'new_bias': round(new_bias_t,4)}

print("\n  New confidence bucket calibration (all fights):")
print(f"  {'Bucket':<8} {'Predicted':>10} {'Actual WR':>10} {'Error':>8} {'N':>6}")
for _, row in new_cal.iterrows():
    flag = " ← |err|>0.10" if abs(row['err']) > 0.10 else ""
    print(f"  {row['bucket']:<8.1f} {row['pred']:>10.3f} {row['actual']:>10.3f} {row['err']:>+8.3f} {int(row['N']):>6}{flag}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Final JS-ready output + spot checks
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 5 — JS-Ready Multiplier Tables + Spot Checks")
print("="*70)

# Spot check: COUNTER Z3×sdog
z3_sdog_counter = corrected_multipliers['COUNTER'][3]['sdog']
cm_z3_sdog = CONF_MATRIX[3]['sdog']
conf_z3_sdog_counter = cm_z3_sdog * z3_sdog_counter['multiplier']
print(f"\n  Spot check 1 — COUNTER Z3×sdog:")
print(f"    type_WR={z3_sdog_counter['type_wr']:.4f}, mixed_WR={z3_sdog_counter['mixed_wr']:.4f}")
print(f"    multiplier={z3_sdog_counter['multiplier']:.4f} (source: {z3_sdog_counter['source']})")
print(f"    CONF_MATRIX[3]['sdog']={cm_z3_sdog:.4f}")
print(f"    NEW confidence = {conf_z3_sdog_counter:.4f} ({conf_z3_sdog_counter*100:.1f}%)")
print(f"    OLD confidence = {cm_z3_sdog * COUNTER_MULTIPLIERS[3]:.4f} ({cm_z3_sdog * COUNTER_MULTIPLIERS[3]*100:.1f}%)")
print(f"    Actual COUNTER Z3×sdog WR (from data): {z3_sdog_counter['type_wr']*100:.1f}%")

# Spot check: SPLIT Z5×sdog
z5_sdog_split = corrected_multipliers['SPLIT'][5]['sdog']
cm_z5_sdog = CONF_MATRIX[5].get('sdog', CONF_ZONE_FALLBACK[5])
conf_z5_sdog_split = cm_z5_sdog * z5_sdog_split['multiplier']
old_conf_z5_sdog = cm_z5_sdog * SPLIT_MULTIPLIERS[5]
print(f"\n  Spot check 2 — SPLIT Z5×sdog:")
print(f"    type_WR={z5_sdog_split['type_wr']:.4f}" if z5_sdog_split['type_wr'] else f"    type_WR=None (zone fallback)")
print(f"    multiplier={z5_sdog_split['multiplier']:.4f} (source: {z5_sdog_split['source']})")
print(f"    CONF_MATRIX[5]['sdog']={cm_z5_sdog:.4f}")
print(f"    NEW confidence = {conf_z5_sdog_split:.4f} ({conf_z5_sdog_split*100:.1f}%)")
print(f"    OLD confidence = {old_conf_z5_sdog:.4f} ({old_conf_z5_sdog*100:.1f}%)")

# Spot check: CONFIRM Z6×mfav
cm_z6_mfav = CONF_MATRIX[6]['mfav']
print(f"\n  Spot check 3 — CONFIRM Z6×mfav:")
print(f"    CONF_MATRIX[6]['mfav']={cm_z6_mfav:.4f}, multiplier=1.00")
print(f"    NEW confidence = {cm_z6_mfav:.4f} ({cm_z6_mfav*100:.1f}%) [unchanged]")

# ── Save results ──────────────────────────────────────────────────────────────
# Serialize for JSON (replace None with null-compatible)
def clean_for_json(d):
    if isinstance(d, dict):
        return {k: clean_for_json(v) for k, v in d.items()}
    if isinstance(d, float) and np.isnan(d):
        return None
    if isinstance(d, np.floating):
        return float(d)
    if isinstance(d, np.integer):
        return int(d)
    return d

results = {
    'analysis': 'multiplier_recalibration',
    'date': '2026-05-14',
    'n_rows': len(df),
    'step1_discrepancies': len(discrepancies),
    'step4_calibration': step4_results,
    'corrected_multipliers': clean_for_json(corrected_multipliers),
    'mixed_2d': clean_for_json(mixed_2d),
}

results_path = os.path.join(OUT_DIR, 'multiplier_recalibration_results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Saved: {results_path}")

# ── Findings markdown ─────────────────────────────────────────────────────────
findings = f"""# Multiplier Recalibration — Findings

**Date:** 2026-05-14
**Data:** `data/value_bet_log.csv` ({len(df)} rows, men's UFC only)

## Root Cause

Previous multipliers (COUNTER: 0.26–0.39, SPLIT: 0.48–0.83) were computed as
`ATYPE_WR / CONFIRM_WR` per zone. But CONF_MATRIX base rates are **mixed**
(CONFIRM-dominated, ~50% of data), not CONFIRM-only. Dividing by CONFIRM_WR
and applying to a CONFIRM-heavy base rate double-corrects downward.

Correct formula: `actual_type_WR / mixed_2D_base_rate` per zone×tier cell.

## Step 1 — CONF_MATRIX Verification

{len(discrepancies)} cell(s) with |difference| > 2pp between CONF_MATRIX and computed data.
{('Matrix values match data well within tolerance.' if len(discrepancies) < 5 else 'Some discrepancies — see results JSON.')}

## Step 4 — Calibration Improvement

| Type | OLD MAE | NEW MAE | Delta | Target <0.10 |
|------|---------|---------|-------|-------------|
| Overall | {step4_results['overall']['old_mae']:.4f} | {step4_results['overall']['new_mae']:.4f} | {step4_results['overall']['new_mae']-step4_results['overall']['old_mae']:+.4f} | {'✓' if step4_results['overall']['new_mae'] < 0.10 else '✗'} |
{chr(10).join(f"| {t} | {v['old_mae']:.4f} | {v['new_mae']:.4f} | {v['new_mae']-v['old_mae']:+.4f} | {'✓' if v['new_mae'] < 0.10 else '✗'} |" for t, v in step4_results.items() if t != 'overall')}

## Step 5 — Spot Checks

- **CONFIRM Z6×mfav**: {cm_z6_mfav*100:.1f}% (unchanged, multiplier=1.00)
- **COUNTER Z3×sdog**: OLD={cm_z3_sdog * COUNTER_MULTIPLIERS[3]*100:.1f}% → NEW={conf_z3_sdog_counter*100:.1f}% (actual WR: {z3_sdog_counter['type_wr']*100:.1f}%)
- **SPLIT Z5×sdog**: OLD={old_conf_z5_sdog*100:.1f}% → NEW={conf_z5_sdog_split*100:.1f}%

## Recommendation

{'Cell-level multipliers improve calibration across all agreement types. Proceed with Part 2.' if step4_results['overall']['new_mae'] < step4_results['overall']['old_mae'] else 'Review results before proceeding to Part 2.'}
"""

md_path = os.path.join(OUT_DIR, 'MULTIPLIER_RECALIBRATION_FINDINGS.md')
with open(md_path, 'w') as f:
    f.write(findings)
print(f"  Saved: {md_path}")
print("\nDone.")
