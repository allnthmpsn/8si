#!/usr/bin/env python3
"""
training/features_grappling.py — 8SI v2 Stage 2.2, control & grappling exposure

Computes five as-of (shift(1), no-leakage) career features per fighter
from data/round_stats.parquet, same architecture as
training/features_kd.py (which see for the fight-level aggregation +
opponent self-join + joint numerator/denominator observed-mask design —
not repeated verbatim here):

  ctrl_pct_for       — share of fight time this fighter spent controlling
  ctrl_pct_against    — share of fight time spent BEING controlled
  ground_share_landed — share of this fighter's own landed significant
                         strikes thrown from ground position (approximates
                         the spec's "ground_time_share": ufcstats.com has
                         no direct ground-TIME stat, only ground-STRIKE
                         counts and overall control time, which mixes
                         clinch and ground control together — this is the
                         closest available proxy, documented as such
                         rather than silently presented as literal time)
  td_landed_per15     — takedowns landed per 15 min, career as-of
  td_absorbed_per15   — takedowns absorbed per 15 min, career as-of

Verified round_stats.parquet's position-strike columns are internally
consistent for the 2015+ window: distance_landed + clinch_landed +
ground_landed == sig_str_landed for every row (checked directly, not
assumed) — so ground_share_landed is a clean ratio, not an approximation
stacked on an approximation.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from training.train_model1 import DATA
from training.features_kd import RAW_DIR, ROUND_STATS_PATH, NAME_MAP_PATH, _load_fight_minutes, _load_name_map

GRAPPLING_FEATURES = ['ctrl_pct_for', 'ctrl_pct_against', 'ground_share_landed',
                       'td_landed_per15', 'td_absorbed_per15']

EPS = 1e-9


def _fight_level(round_stats_path=ROUND_STATS_PATH, raw_dir=RAW_DIR, name_map_path=NAME_MAP_PATH):
    rs = pd.read_parquet(round_stats_path)
    fight_level = rs.groupby(['event', 'bout', 'date', 'fighter'], as_index=False)[
        ['ctrl_sec', 'sig_str_landed', 'ground_landed', 'td_landed']
    ].sum()

    name_map = _load_name_map(name_map_path)
    fight_level['canonical'] = fight_level['fighter'].map(name_map)
    fight_level = fight_level.dropna(subset=['canonical'])

    fight_min = _load_fight_minutes(raw_dir)
    fight_level = fight_level.merge(fight_min, on=['event', 'bout'], how='left')
    fight_level['fight_sec'] = fight_level['fight_min'] * 60.0

    pair = fight_level.merge(
        fight_level[['event', 'bout', 'canonical', 'ctrl_sec', 'td_landed']],
        on=['event', 'bout'], suffixes=('', '_opp'),
    )
    pair = pair[pair['canonical'] != pair['canonical_opp']]
    pair = pair.rename(columns={'ctrl_sec_opp': 'opp_ctrl_sec', 'td_landed_opp': 'opp_td_landed'})
    pair = pair.drop_duplicates(subset=['event', 'bout', 'canonical'])
    return pair[['canonical', 'date', 'ctrl_sec', 'opp_ctrl_sec', 'sig_str_landed', 'ground_landed',
                 'td_landed', 'opp_td_landed', 'fight_sec', 'fight_min']].rename(columns={'canonical': 'fighter'})


def compute_grappling_features_asof(round_stats_path=ROUND_STATS_PATH, raw_dir=RAW_DIR, name_map_path=NAME_MAP_PATH):
    """Returns DataFrame[fighter, date] + GRAPPLING_FEATURES, one row per
    (canonical fighter, fight), each value computed from ONLY that
    fighter's fights strictly before `date`."""
    df = _fight_level(round_stats_path, raw_dir, name_map_path)
    df = df.sort_values(['fighter', 'date']).reset_index(drop=True)

    # (feature, numerator col, denominator col, per15, is_complement)
    PAIRS = [
        ('ctrl_pct_for', 'ctrl_sec', 'fight_sec', False, False),
        ('ctrl_pct_against', 'opp_ctrl_sec', 'fight_sec', False, False),
        ('ground_share_landed', 'ground_landed', 'sig_str_landed', False, False),
        ('td_landed_per15', 'td_landed', 'fight_min', True, False),
        ('td_absorbed_per15', 'opp_td_landed', 'fight_min', True, False),
    ]

    for feat, ncol, dcol, _, _ in PAIRS:
        joint = df[ncol].notna() & df[dcol].notna()
        df[f'_jn_{feat}'] = df[ncol].where(joint, 0.0)
        df[f'_jd_{feat}'] = df[dcol].where(joint, 0.0)
        df[f'_jo_{feat}'] = joint.astype(float)

    g = df.groupby('fighter', sort=False)
    for feat, *_ in PAIRS:
        df[f'_csn_{feat}'] = (g[f'_jn_{feat}'].cumsum() - df[f'_jn_{feat}']).astype(float)
        df[f'_csd_{feat}'] = (g[f'_jd_{feat}'].cumsum() - df[f'_jd_{feat}']).astype(float)
        df[f'_cso_{feat}'] = (g[f'_jo_{feat}'].cumsum() - df[f'_jo_{feat}']).astype(float)

    nan = np.nan
    for feat, ncol, dcol, per15, is_complement in PAIRS:
        has = (df[f'_cso_{feat}'] > 0).to_numpy() & (df[f'_csd_{feat}'] > 0).to_numpy()
        denom = df[f'_csd_{feat}'] / 15.0 if per15 else df[f'_csd_{feat}']
        ratio = df[f'_csn_{feat}'].to_numpy() / denom.clip(lower=EPS).to_numpy()
        if is_complement:
            ratio = 1.0 - ratio
        df[feat] = np.where(has, ratio, nan)

    return df[['fighter', 'date'] + GRAPPLING_FEATURES]


if __name__ == '__main__':
    out = compute_grappling_features_asof()
    print(f'Rows: {len(out):,}  |  Unique fighters: {out["fighter"].nunique():,}')
    print(out.dropna(subset=['ctrl_pct_for']).describe())
