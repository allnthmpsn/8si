"""
Positive vs negative gap split analysis.

gap = m2a_prob - pick_novig (fraction, signed):
  > 0  → model sees the pick as undervalued by Vegas (value bet territory)
  < 0  → model sees the pick as overvalued by Vegas (Vegas more confident)
  near 0 → model and Vegas roughly agree on probability

Groups:
  pos_gap   : gap > 0 (exclusive of near_zero)
  neg_gap   : gap < 0 (exclusive of near_zero)
  near_zero : abs(gap) < 0.01

Key questions answered:
  Q1 — For neg_gap + M1/M2A agree: is WR > 79.1%?
  Q2 — Chimaev-style (neg_gap > 10% + agree + heavy fav): WR? Hypothesis 85%+
  Q3 — Should neg_gap + agree fights be surfaced in UI differently?
  Q4 — Does gap direction affect WR within the agree group?
"""
import os
import json
import pandas as pd
import numpy as np

ROOT    = os.path.join(os.path.dirname(__file__), '..', '..', '..')
DATA    = os.path.join(ROOT, 'data', 'value_bet_log.csv')
OUT_DIR = os.path.dirname(__file__)

df = pd.read_csv(DATA)
print(f"Loaded {len(df)} rows")

# ── Direction masks (exclusive buckets) ─────────────────────────────────────
near_zero_mask = df['gap'].abs() < 0.01
pos_mask       = (df['gap'] > 0) & ~near_zero_mask    # gap >= +1%
neg_mask       = (df['gap'] < 0) & ~near_zero_mask    # gap <= -1%

df['direction'] = 'near_zero'
df.loc[pos_mask, 'direction'] = 'pos_gap'
df.loc[neg_mask, 'direction'] = 'neg_gap'

print(f"  pos_gap: {pos_mask.sum()}, neg_gap: {neg_mask.sum()}, near_zero: {near_zero_mask.sum()}")

# ── ROI helper ───────────────────────────────────────────────────────────────
def roi_pct(subset):
    valid = subset[subset['closing_odds'].notna() & (subset['closing_odds'] != 0)]
    if len(valid) == 0:
        return None
    def net(row):
        o = row['closing_odds']
        profit = o / 100 if o > 0 else 100 / abs(o)
        return profit if row['pick_won'] == 1 else -1.0
    return round(float(valid.apply(net, axis=1).mean()) * 100, 2)

# ── Group stats ──────────────────────────────────────────────────────────────
def group_stats(subset, label):
    if len(subset) == 0:
        return {'label': label, 'N': 0}
    agree     = subset[subset['m1_m2a_agree'] == 1]
    disagree  = subset[subset['m1_m2a_agree'] == 0]
    triple    = subset[subset['triple_agree']  == 1]

    def wr(s):
        return round(float(s['pick_won'].mean()) * 100, 1) if len(s) > 0 else None

    return {
        'label':              label,
        'N':                  len(subset),
        'overall_wr':         wr(subset),
        'agree_N':            len(agree),
        'agree_wr':           wr(agree),
        'disagree_N':         len(disagree),
        'disagree_wr':        wr(disagree),
        'triple_agree_N':     len(triple),
        'triple_agree_wr':    wr(triple),
        'avg_gap_size_pct':   round(float(subset['gap_size'].mean()) * 100, 2),
        'avg_closing_odds':   round(float(subset['closing_odds'].mean()), 1),
        'roi_pct':            roi_pct(subset),
    }

direction_stats = {
    'pos_gap':   group_stats(df[pos_mask],       'pos_gap (gap >= +1%)'),
    'neg_gap':   group_stats(df[neg_mask],        'neg_gap (gap <= -1%)'),
    'near_zero': group_stats(df[near_zero_mask],  'near_zero (|gap| < 1%)'),
    'all':       group_stats(df,                  'all fights'),
}

print("\nDirection breakdown:")
for k, v in direction_stats.items():
    print(f"  {k:12s}: N={v['N']:4d}  overall_wr={v['overall_wr']}%  agree_wr={v['agree_wr']}%  "
          f"triple_wr={v['triple_agree_wr']}%  roi={v['roi_pct']}%")

# ── Negative gap by magnitude ─────────────────────────────────────────────
neg_df  = df[neg_mask].copy()
neg_df['neg_mag'] = neg_df['gap'].abs()
p = neg_df['neg_mag']

neg_buckets = {
    '0–3%':   neg_df[p < 0.03],
    '3–5%':   neg_df[(p >= 0.03) & (p < 0.05)],
    '5–10%':  neg_df[(p >= 0.05) & (p < 0.10)],
    '>10%':   neg_df[p >= 0.10],
}

neg_magnitude_stats = {}
print("\nNegative gap by magnitude:")
for label, sub in neg_buckets.items():
    stats = group_stats(sub, f'neg_{label}')
    neg_magnitude_stats[label] = stats
    print(f"  neg {label:6s}: N={stats['N']:4d}  overall_wr={stats['overall_wr']}%  "
          f"agree_wr={stats['agree_wr']}%  triple_wr={stats['triple_agree_wr']}%")

# ── Cross-tab: direction × agreement ─────────────────────────────────────
cross = {}
for dir_label, dir_mask_val in [('pos_gap', pos_mask), ('neg_gap', neg_mask), ('near_zero', near_zero_mask)]:
    sub = df[dir_mask_val]
    agree_s = sub[sub['m1_m2a_agree'] == 1]
    disagree_s = sub[sub['m1_m2a_agree'] == 0]
    cross[dir_label] = {
        'agree':    {'N': len(agree_s),    'wr': round(float(agree_s['pick_won'].mean()) * 100, 1) if len(agree_s) > 0 else None,
                     'roi': roi_pct(agree_s)},
        'disagree': {'N': len(disagree_s), 'wr': round(float(disagree_s['pick_won'].mean()) * 100, 1) if len(disagree_s) > 0 else None,
                     'roi': roi_pct(disagree_s)},
    }

print("\nCross-tab (direction × agreement) — WR% (N) [ROI%]:")
for dir_label, cells in cross.items():
    print(f"  {dir_label:12s}  agree={cells['agree']['wr']}% (N={cells['agree']['N']}, roi={cells['agree']['roi']}%)  "
          f"disagree={cells['disagree']['wr']}% (N={cells['disagree']['N']}, roi={cells['disagree']['roi']}%)")

# ── Key Questions ─────────────────────────────────────────────────────────

# Q1: neg_gap + agree — does WR exceed 79.1%?
neg_agree = df[neg_mask & (df['m1_m2a_agree'] == 1)]
q1 = {
    'question':   'For neg_gap + M1/M2A agree — is WR > 79.1% benchmark?',
    'neg_agree_N':  len(neg_agree),
    'neg_agree_wr': round(float(neg_agree['pick_won'].mean()) * 100, 1) if len(neg_agree) > 0 else None,
    'beats_benchmark': None,
}
if q1['neg_agree_wr'] is not None:
    q1['beats_benchmark'] = q1['neg_agree_wr'] > 79.1
print(f"\nQ1: neg_gap+agree WR = {q1['neg_agree_wr']}% (N={q1['neg_agree_N']})  beats_79.1%={q1['beats_benchmark']}")

# Q2: Chimaev-style — large neg_gap (>10%) + agree + heavy fav (closing_odds < -300)
chimaev_mask = neg_mask & (df['gap'].abs() >= 0.10) & (df['m1_m2a_agree'] == 1) & (df['closing_odds'] < -300)
chimaev = df[chimaev_mask]
q2 = {
    'question': 'Large neg_gap (>10%) + agree + heavy fav (<-300 odds) — WR? Hypothesis: 85%+',
    'N':        len(chimaev),
    'wr':       round(float(chimaev['pick_won'].mean()) * 100, 1) if len(chimaev) > 0 else None,
    'roi':      roi_pct(chimaev),
    'hypothesis_confirmed': None,
}
if q2['wr'] is not None:
    q2['hypothesis_confirmed'] = q2['wr'] >= 85.0

# Broader: neg_gap >10% + agree (no odds threshold)
chimaev_broad = df[neg_mask & (df['gap'].abs() >= 0.10) & (df['m1_m2a_agree'] == 1)]
q2['broad_N']  = len(chimaev_broad)
q2['broad_wr'] = round(float(chimaev_broad['pick_won'].mean()) * 100, 1) if len(chimaev_broad) > 0 else None

print(f"Q2: Chimaev-style (neg>10%+agree+<-300) WR = {q2['wr']}% (N={q2['N']})  hypothesis_confirmed={q2['hypothesis_confirmed']}")
print(f"    Broad (neg>10%+agree no odds filter): {q2['broad_wr']}% (N={q2['broad_N']})")

# Q3: Should neg_gap + agree be surfaced differently in UI?
# Compare: pos_gap+agree vs neg_gap+agree vs near_zero+agree
pos_agree   = df[pos_mask      & (df['m1_m2a_agree'] == 1)]
nz_agree    = df[near_zero_mask & (df['m1_m2a_agree'] == 1)]
q3 = {
    'question': 'Does gap direction affect WR within agree group — warrant different UI treatment?',
    'pos_agree_wr':  round(float(pos_agree['pick_won'].mean()) * 100, 1) if len(pos_agree) > 0 else None,
    'pos_agree_N':   len(pos_agree),
    'neg_agree_wr':  q1['neg_agree_wr'],
    'neg_agree_N':   q1['neg_agree_N'],
    'near_zero_agree_wr': round(float(nz_agree['pick_won'].mean()) * 100, 1) if len(nz_agree) > 0 else None,
    'near_zero_agree_N':  len(nz_agree),
    'pos_agree_roi': roi_pct(pos_agree),
    'neg_agree_roi': roi_pct(neg_agree),
}
print(f"Q3: WR within agree group — pos={q3['pos_agree_wr']}%(N={q3['pos_agree_N']})  "
      f"neg={q3['neg_agree_wr']}%(N={q3['neg_agree_N']})  near_zero={q3['near_zero_agree_wr']}%(N={q3['near_zero_agree_N']})")
print(f"    ROI within agree group — pos={q3['pos_agree_roi']}%  neg={q3['neg_agree_roi']}%")

# Q4: Gap direction interaction with gap zone — does neg direction degrade value zones?
print("\nQ4: Neg gap by magnitude × agreement:")
neg_mag_agree_cross = {}
for label, sub in neg_buckets.items():
    ag = sub[sub['m1_m2a_agree'] == 1]
    di = sub[sub['m1_m2a_agree'] == 0]
    def wr(s): return round(float(s['pick_won'].mean()) * 100, 1) if len(s) > 0 else None
    neg_mag_agree_cross[label] = {
        'agree_wr': wr(ag), 'agree_N': len(ag),
        'disagree_wr': wr(di), 'disagree_N': len(di),
        'roi_agree': roi_pct(ag),
    }
    print(f"  neg {label:6s}: agree={wr(ag)}%(N={len(ag)}, roi={roi_pct(ag)}%)  disagree={wr(di)}%(N={len(di)})")

# ── Save ──────────────────────────────────────────────────────────────────
results = {
    'dataset': {
        'source':     'data/value_bet_log.csv',
        'total_rows': len(df),
        'date_range': f"{df['date'].min()} to {df['date'].max()}",
    },
    'direction_stats':          direction_stats,
    'neg_magnitude_stats':      neg_magnitude_stats,
    'cross_tab_direction_x_agreement': cross,
    'neg_magnitude_x_agreement': neg_mag_agree_cross,
    'key_questions': {
        'Q1_neg_agree_vs_benchmark':      q1,
        'Q2_chimaev_style':               q2,
        'Q3_direction_within_agree':      q3,
        'Q4_neg_mag_x_agree':             {'table': neg_mag_agree_cross},
    },
}

out_json = os.path.join(OUT_DIR, 'gap_direction_results.json')
with open(out_json, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {out_json}")
