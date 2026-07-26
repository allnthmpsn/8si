#!/usr/bin/env python3
"""
training/features_style_vectors.py — 8SI v2 Stage 2.4, position/target mix

Six as-of (shift(1), no-leakage) career features per fighter, from
data/round_stats.parquet only (no second source needed — unlike 2.1-2.3,
every column here already lives in round_stats.parquet):

  Position mix (share of this fighter's own landed sig strikes thrown
  from each position): dist_share, clinch_share, ground_share
  Target mix (share landed at each target): head_share, body_share,
  leg_share

Both breakdowns are exact partitions of sig_str_landed — verified
directly (not assumed) that distance_landed+clinch_landed+ground_landed
== sig_str_landed and head_landed+body_landed+leg_landed ==
sig_str_landed for every 2015+ row, so these are clean shares, not
approximations.

Matchup interactions (own leg_share x opp stance, own ground_share x opp
ctrl_pct_against, own td_landed_per15 x opp td_absorbed_per15) are the
spec's Stage 2.4 second half, explicitly gated "only after mains are
in" — see docs/V2_LOG.md for why they weren't built: the position/target
mix "mains" themselves didn't clear the walk-forward gate, and two of
the three named interactions are built on Stage 2.2's td_landed_per15/
ctrl_pct_against, which already failed its own gate independently.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from training.train_model1 import DATA
from training.features_kd import RAW_DIR, ROUND_STATS_PATH, NAME_MAP_PATH, _load_name_map

STYLE_VECTOR_FEATURES = ['dist_share', 'clinch_share', 'ground_share', 'head_share', 'body_share', 'leg_share']

EPS = 1e-9


def _fight_level(round_stats_path=ROUND_STATS_PATH, name_map_path=NAME_MAP_PATH):
    rs = pd.read_parquet(round_stats_path)
    fight_level = rs.groupby(['event', 'bout', 'date', 'fighter'], as_index=False)[
        ['sig_str_landed', 'distance_landed', 'clinch_landed', 'ground_landed',
         'head_landed', 'body_landed', 'leg_landed']
    ].sum()

    name_map = _load_name_map(name_map_path)
    fight_level['canonical'] = fight_level['fighter'].map(name_map)
    fight_level = fight_level.dropna(subset=['canonical'])
    fight_level = fight_level.drop_duplicates(subset=['event', 'bout', 'canonical'])
    fight_level = fight_level.drop(columns=['fighter']).rename(columns={'canonical': 'fighter'})
    return fight_level[
        ['fighter', 'date', 'sig_str_landed', 'distance_landed', 'clinch_landed', 'ground_landed',
         'head_landed', 'body_landed', 'leg_landed']
    ]


def compute_style_vector_features_asof(round_stats_path=ROUND_STATS_PATH, name_map_path=NAME_MAP_PATH):
    """Returns DataFrame[fighter, date] + STYLE_VECTOR_FEATURES, one row
    per (canonical fighter, fight), each value computed from ONLY that
    fighter's fights strictly before `date`."""
    df = _fight_level(round_stats_path, name_map_path)
    df = df.sort_values(['fighter', 'date']).reset_index(drop=True)

    PAIRS = [
        ('dist_share', 'distance_landed', 'sig_str_landed'),
        ('clinch_share', 'clinch_landed', 'sig_str_landed'),
        ('ground_share', 'ground_landed', 'sig_str_landed'),
        ('head_share', 'head_landed', 'sig_str_landed'),
        ('body_share', 'body_landed', 'sig_str_landed'),
        ('leg_share', 'leg_landed', 'sig_str_landed'),
    ]

    for feat, ncol, dcol in PAIRS:
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
    for feat, ncol, dcol in PAIRS:
        has = (df[f'_cso_{feat}'] > 0).to_numpy() & (df[f'_csd_{feat}'] > 0).to_numpy()
        ratio = df[f'_csn_{feat}'].to_numpy() / df[f'_csd_{feat}'].clip(lower=EPS).to_numpy()
        df[feat] = np.where(has, ratio, nan)

    return df[['fighter', 'date'] + STYLE_VECTOR_FEATURES]


if __name__ == '__main__':
    out = compute_style_vector_features_asof()
    print(f'Rows: {len(out):,}  |  Unique fighters: {out["fighter"].nunique():,}')
    print(out.dropna(subset=['dist_share']).describe())
