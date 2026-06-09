"""
M1 vs M2A proximity analysis.

For each row in data/value_bet_log.csv, proximity = abs(m1_prob - m2a_prob).
This is the same regardless of which fighter M2A picked, since both models
output F1 probability and (1 - F1 prob) for F2.

Answers 4 key questions and recommends whether to add proximity as a
continuous feature to M2B.
"""
import os
import json
import pandas as pd
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..', '..', '..')
DATA = os.path.join(ROOT, 'data', 'value_bet_log.csv')
OUT_DIR = os.path.dirname(__file__)

df = pd.read_csv(DATA)
print(f"Loaded {len(df)} rows from {DATA}")

# ── Proximity ────────────────────────────────────────────────────────────────
df['proximity'] = (df['m1_prob'] - df['m2a_prob']).abs()

ZONE_LABELS = {
    0: 'Lock(<1%)', 1: 'Strong(<2%)', 2: 'Lean(<3%)',
    3: 'Watch(<5%)', 4: 'Value(<8%)', 5: 'StrongVal(<10%)', 6: 'MaxVal(>10%)',
}

# ── ROI helper ───────────────────────────────────────────────────────────────
def roi_on_subset(subset):
    valid = subset[subset['closing_odds'].notna() & (subset['closing_odds'] != 0)]
    if len(valid) == 0:
        return None
    def net(row):
        o = row['closing_odds']
        profit = o / 100 if o > 0 else 100 / abs(o)
        return profit if row['pick_won'] == 1 else -1.0
    return round(float(valid.apply(net, axis=1).mean()) * 100, 2)

# ── Bucket stats helper ──────────────────────────────────────────────────────
def bucket_stats(subset):
    n = len(subset)
    if n == 0:
        return {'N': 0}
    agree_s    = subset[subset['m1_m2a_agree'] == 1]
    disagree_s = subset[subset['m1_m2a_agree'] == 0]
    triple_s   = subset[subset['triple_agree']  == 1]
    mode_val   = subset['gap_zone'].mode()

    def wr(s):
        return round(float(s['pick_won'].mean()) * 100, 1) if len(s) > 0 else None

    return {
        'N':                  n,
        'overall_wr':         wr(subset),
        'agree_N':            len(agree_s),
        'agree_wr':           wr(agree_s),
        'disagree_N':         len(disagree_s),
        'disagree_wr':        wr(disagree_s),
        'triple_agree_N':     len(triple_s),
        'triple_agree_wr':    wr(triple_s),
        'avg_gap_vs_vegas_pct': round(float(subset['gap_size'].mean()) * 100, 2),
        'roi_pct':            roi_on_subset(subset),
        'most_common_gap_zone': int(mode_val[0]) if len(mode_val) > 0 else None,
    }

# ── 10 proximity buckets ─────────────────────────────────────────────────────
p = df['proximity']

CUMULATIVE_BUCKETS = {
    'within_1pct':   p <= 0.01,
    'within_3pct':   p <= 0.03,
    'within_5pct':   p <= 0.05,
    'within_10pct':  p <= 0.10,
    'within_20pct':  p <= 0.20,
    'gt_20pct':      p >  0.20,
}
EXCLUSIVE_BUCKETS = {
    '1_to_3pct_only':   (p > 0.01) & (p <= 0.03),
    '3_to_5pct_only':   (p > 0.03) & (p <= 0.05),
    '5_to_10pct_only':  (p > 0.05) & (p <= 0.10),
    '10_to_20pct_only': (p > 0.10) & (p <= 0.20),
}
ALL_BUCKETS = {**CUMULATIVE_BUCKETS, **EXCLUSIVE_BUCKETS}

proximity_results = {}
for name, mask in ALL_BUCKETS.items():
    proximity_results[name] = bucket_stats(df[mask])

print("\nProximity bucket breakdown:")
for name, stats in proximity_results.items():
    print(f"  {name:22s}: N={stats['N']:4d}  overall_wr={stats['overall_wr']}%  agree_wr={stats['agree_wr']}%  triple_wr={stats['triple_agree_wr']}%")

# ── Cross-tab: proximity (exclusive) × gap zone ──────────────────────────────
EXCL_ORDERED = [
    ('≤1%',    p <= 0.01),
    ('1-3%',   (p > 0.01) & (p <= 0.03)),
    ('3-5%',   (p > 0.03) & (p <= 0.05)),
    ('5-10%',  (p > 0.05) & (p <= 0.10)),
    ('10-20%', (p > 0.10) & (p <= 0.20)),
    ('20%+',   p > 0.20),
]

crosstab = {}
for bucket_label, mask in EXCL_ORDERED:
    row = {}
    sub = df[mask]
    for zone in range(7):
        zone_sub = sub[sub['gap_zone'] == zone]
        if len(zone_sub) > 0:
            wr_val = round(float(zone_sub['pick_won'].mean()) * 100, 1)
        else:
            wr_val = None
        row[f'z{zone}_{ZONE_LABELS[zone]}'] = {'wr': wr_val, 'N': len(zone_sub)}
    crosstab[bucket_label] = row

print("\nCross-tab (proximity × gap zone) — win rate% (N):")
header = "Bucket    " + "".join(f"  Z{z:<16}" for z in range(7))
print(header)
for bucket_label, row in crosstab.items():
    line = f"{bucket_label:<10}"
    for zone in range(7):
        cell = row[f'z{zone}_{ZONE_LABELS[zone]}']
        if cell['N'] > 0:
            line += f"  {cell['wr']}%({cell['N']})"
        else:
            line += "  —        "
    print(line)

# ── 4 Key Questions ──────────────────────────────────────────────────────────

# Q1: within-3% + agreement vs just agreement
w3_agree  = df[(p <= 0.03) & (df['m1_m2a_agree'] == 1)]
all_agree = df[df['m1_m2a_agree'] == 1]
q1 = {
    'question': 'Does within-3% proximity + agreement outperform just agreement (79.1% benchmark)?',
    'within_3pct_agree_wr':  round(float(w3_agree['pick_won'].mean()) * 100, 1) if len(w3_agree) > 0 else None,
    'within_3pct_agree_N':   len(w3_agree),
    'just_agree_wr':         round(float(all_agree['pick_won'].mean()) * 100, 1) if len(all_agree) > 0 else None,
    'just_agree_N':          len(all_agree),
    'outperforms_benchmark': None,  # filled below
}
if q1['within_3pct_agree_wr'] is not None and q1['just_agree_wr'] is not None:
    q1['outperforms_benchmark'] = q1['within_3pct_agree_wr'] > 79.1

# Q2: At what bucket does agree WR drop off?
q2_rows = {}
for bucket_label, mask in EXCL_ORDERED:
    sub = df[mask & (df['m1_m2a_agree'] == 1)]
    q2_rows[bucket_label] = {
        'agree_wr': round(float(sub['pick_won'].mean()) * 100, 1) if len(sub) > 0 else None,
        'N': len(sub),
    }
q2 = {
    'question': 'At what proximity bucket does agree win rate drop off?',
    'agree_wr_by_exclusive_bucket': q2_rows,
}

# Q3: 20%+ apart + agree — still meaningful?
gt20_agree = df[(p > 0.20) & (df['m1_m2a_agree'] == 1)]
q3 = {
    'question': 'When models are 20%+ apart but agree, is the signal still meaningful?',
    'gt20pct_agree_wr': round(float(gt20_agree['pick_won'].mean()) * 100, 1) if len(gt20_agree) > 0 else None,
    'gt20pct_agree_N':  len(gt20_agree),
    'baseline_agree_wr': q1['just_agree_wr'],
}
if q3['gt20pct_agree_wr'] is not None:
    q3['still_meaningful'] = q3['gt20pct_agree_wr'] >= 70.0

# Q4: within-1% + triple_agree win rate?
w1_triple = df[(p <= 0.01) & (df['triple_agree'] == 1)]
q4 = {
    'question': 'What is the win rate for within-1% proximity + triple_agree?',
    'within_1pct_triple_agree_wr': round(float(w1_triple['pick_won'].mean()) * 100, 1) if len(w1_triple) > 0 else None,
    'within_1pct_triple_agree_N':  len(w1_triple),
}

# Q5: Recommendation — add proximity as M2B feature?
corr_overall = df['proximity'].corr(df['pick_won'])
corr_agree   = df[df['m1_m2a_agree'] == 1]['proximity'].corr(
                   df[df['m1_m2a_agree'] == 1]['pick_won'])
corr_disagree= df[df['m1_m2a_agree'] == 0]['proximity'].corr(
                   df[df['m1_m2a_agree'] == 0]['pick_won'])
recommend_add = abs(float(corr_overall)) > 0.04 or abs(float(corr_agree)) > 0.04

q5 = {
    'question': 'Recommend adding m1_m2a_proximity as continuous feature to M2B?',
    'correlation_overall': round(float(corr_overall), 4),
    'correlation_in_agree_group': round(float(corr_agree), 4),
    'correlation_in_disagree_group': round(float(corr_disagree), 4),
    'recommend': recommend_add,
    'reasoning': (
        'Add if correlation within agree/disagree groups is meaningful (|r| > 0.04). '
        'Proximity captures how confidently the models converge, which is different from '
        'the binary m1_m2a_agree flag already in M2B.'
    ),
}

print(f"\nQ1: within-3%+agree WR = {q1['within_3pct_agree_wr']}% (N={q1['within_3pct_agree_N']}) vs all-agree WR = {q1['just_agree_wr']}% (N={q1['just_agree_N']})")
print(f"Q2: agree WR by exclusive bucket:")
for b, v in q2_rows.items():
    print(f"   {b}: {v['agree_wr']}% (N={v['N']})")
print(f"Q3: 20%+ apart + agree WR = {q3['gt20pct_agree_wr']}% (N={q3['gt20pct_agree_N']})")
print(f"Q4: within-1%+triple_agree WR = {q4['within_1pct_triple_agree_wr']}% (N={q4['within_1pct_triple_agree_N']})")
print(f"Rec: corr(proximity, pick_won) = {q5['correlation_overall']:.4f}  agree_group = {q5['correlation_in_agree_group']:.4f}  → add={q5['recommend']}")

# ── Save results ─────────────────────────────────────────────────────────────
results = {
    'dataset': {
        'source': 'data/value_bet_log.csv',
        'total_rows': len(df),
        'date_range': f"{df['date'].min()} to {df['date'].max()}",
    },
    'proximity_bucket_stats': proximity_results,
    'crosstab_proximity_x_gap_zone': crosstab,
    'key_questions': {
        'Q1_within3pct_plus_agreement': q1,
        'Q2_dropoff_threshold':         q2,
        'Q3_far_apart_agree':           q3,
        'Q4_within1pct_triple_agree':   q4,
        'Q5_recommendation':            q5,
    },
}

out_json = os.path.join(OUT_DIR, 'proximity_results.json')
with open(out_json, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {out_json}")
