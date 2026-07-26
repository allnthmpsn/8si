#!/usr/bin/env python3
"""
training/features_cardio.py — 8SI v2 Stage 2.3, cardio / late-round profile

Three as-of (shift(1), no-leakage) career features per fighter:

  r3_output_ratio      — (sig strike attempts per min, rounds 3+) /
                          (sig strike attempts per min, round 1), career
                          as-of. NaN unless the fighter has >=3 PRIOR
                          fights that reached round 3 (the spec's own
                          minimum-sample guard, since a single early
                          fight's round-3 output is noisy).
  late_round_finish_rate — share of career fights won by finish
                            (KO/TKO/Submission, same definition
                            compute_career_stats() already uses) in round
                            3 or later, career as-of.
  r1_finish_rate          — share of career fights won by finish in
                             round 1, career as-of.

r3_output_ratio needs round-level strike volume (data/round_stats.parquet
— round_stats.parquet has no notion of "how did the fight end", so it
can't answer this on its own). The finish-timing features need win/
method/round-of-finish, which round_stats.parquet also lacks (it's pure
per-round volume stats) — sourced instead from
data/raw/ufcstats_rounds/ufc_fight_results.csv (BOUT, OUTCOME, METHOD,
ROUND), parsed and joined through data/name_map.csv the same as every
other round-derived family. "Finish" uses the exact same KO/TKO/Sub
definition training/train_model1.py's compute_career_stats() already
uses (str.contains('KO|TKO') / str.contains('Sub')), not a new one.
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

CARDIO_FEATURES = ['r3_output_ratio', 'late_round_finish_rate', 'r1_finish_rate']

EPS = 1e-9


def _load_results(raw_dir=RAW_DIR):
    """One row per (event, bout, canonical fighter): won (0/1), is_finish
    (0/1), finish_round (int, NaN if not a finish or fighter lost)."""
    fr = pd.read_csv(os.path.join(raw_dir, 'ufc_fight_results.csv'))
    fr['EVENT'] = fr['EVENT'].str.strip()
    ev = pd.read_csv(os.path.join(raw_dir, 'ufc_event_details.csv'))
    ev['date'] = pd.to_datetime(ev['DATE'], format='%B %d, %Y')
    fr = fr.merge(ev[['EVENT', 'date']], on='EVENT', how='left')
    fr = fr.dropna(subset=['date'])

    fr = fr[fr['OUTCOME'].isin(['W/L', 'L/W'])].copy()
    fr[['f1', 'f2']] = fr['BOUT'].str.split(' vs. ', n=1, expand=True)
    method = fr['METHOD'].str.strip()
    is_finish = method.str.contains('KO|TKO', case=False, na=False) | method.str.contains('Sub', case=False, na=False)

    rows = []
    for corner, name_col, win_flag in (('f1', 'f1', fr['OUTCOME'] == 'W/L'), ('f2', 'f2', fr['OUTCOME'] == 'L/W')):
        rows.append(pd.DataFrame({
            'event': fr['EVENT'], 'bout': fr['BOUT'], 'date': fr['date'],
            'fighter': fr[name_col], 'won': win_flag.astype(int),
            'is_finish': is_finish.astype(int), 'round': fr['ROUND'].astype(int),
        }))
    long_df = pd.concat(rows, ignore_index=True)

    name_map = _load_name_map(NAME_MAP_PATH)
    long_df['canonical'] = long_df['fighter'].map(name_map)
    long_df = long_df.dropna(subset=['canonical'])
    long_df = long_df.drop_duplicates(subset=['event', 'bout', 'canonical'])
    return long_df.rename(columns={'canonical': 'canon_fighter'})


def _round_output(round_stats_path=ROUND_STATS_PATH, name_map_path=NAME_MAP_PATH):
    """One row per (event, bout, canonical fighter): r1_att (round-1 sig
    strike attempts), r1_min (always 5, but computed defensively as
    max round-1 duration isn't tracked per-round — round length is
    assumed 5 min, true for every 2015+ fight per training/features_kd.py's
    own TIME FORMAT check), r3plus_att, r3plus_min (rounds actually
    fought, 3+), reached_r3 (bool)."""
    rs = pd.read_parquet(round_stats_path)
    name_map = _load_name_map(name_map_path)
    rs['canonical'] = rs['fighter'].map(name_map)
    rs = rs.dropna(subset=['canonical'])

    r1 = rs[rs['round'] == 1].groupby(['event', 'bout', 'date', 'canonical'], as_index=False)['sig_str_att'].sum()
    r1 = r1.rename(columns={'sig_str_att': 'r1_att'})

    r3plus = rs[rs['round'] >= 3]
    r3plus_agg = r3plus.groupby(['event', 'bout', 'canonical'], as_index=False).agg(
        r3plus_att=('sig_str_att', 'sum'), r3plus_rounds=('round', 'nunique'))

    out = r1.merge(r3plus_agg, on=['event', 'bout', 'canonical'], how='left')
    out['reached_r3'] = out['r3plus_rounds'].notna()
    out['r3plus_att'] = out['r3plus_att'].fillna(0.0)
    out['r3plus_rounds'] = out['r3plus_rounds'].fillna(0.0)
    out = out.drop(columns=['date'])  # results already carries date; avoid a merge suffix collision
    return out.rename(columns={'canonical': 'canon_fighter'})


def compute_cardio_features_asof(round_stats_path=ROUND_STATS_PATH, raw_dir=RAW_DIR, name_map_path=NAME_MAP_PATH):
    """Returns DataFrame[fighter, date] + CARDIO_FEATURES, one row per
    (canonical fighter, fight), each value computed from ONLY that
    fighter's fights strictly before `date`."""
    results = _load_results(raw_dir)
    output = _round_output(round_stats_path, name_map_path)

    df = results.merge(output, on=['event', 'bout', 'canon_fighter'], how='left')
    df = df.sort_values(['canon_fighter', 'date']).reset_index(drop=True)

    df['r1_rate'] = df['r1_att'] / 5.0  # round length is 5 min for every 2015+ fight (verified in features_kd.py)
    df['r3plus_rate'] = np.where(df['r3plus_rounds'] > 0, df['r3plus_att'] / (df['r3plus_rounds'] * 5.0), np.nan)
    df['r1_finish'] = ((df['won'] == 1) & (df['is_finish'] == 1) & (df['round'] == 1)).astype(float)
    df['late_finish'] = ((df['won'] == 1) & (df['is_finish'] == 1) & (df['round'] >= 3)).astype(float)

    g = df.groupby('canon_fighter', sort=False)
    df['_cs_fights'] = g.cumcount()
    df['_cs_r1_finish'] = g['r1_finish'].cumsum() - df['r1_finish']
    df['_cs_late_finish'] = g['late_finish'].cumsum() - df['late_finish']
    df['r1_finish_rate'] = np.where(df['_cs_fights'] > 0, df['_cs_r1_finish'] / df['_cs_fights'], np.nan)
    df['late_round_finish_rate'] = np.where(df['_cs_fights'] > 0, df['_cs_late_finish'] / df['_cs_fights'], np.nan)

    # r3_output_ratio: joint observed-mask on (r1_rate always observed for
    # any fight; r3plus_rate only observed for fights reaching round 3) —
    # PLUS the spec's own >=3-prior-round-3-fights minimum sample guard.
    has_r3 = df['reached_r3'].fillna(False)
    df['_jn_r1'] = df['r1_rate']
    df['_jn_r3'] = df['r3plus_rate'].where(has_r3, 0.0)
    df['_jo_r3'] = has_r3.astype(float)
    df['_csn_r1'] = g['_jn_r1'].cumsum() - df['_jn_r1']
    df['_csn_r3'] = g['_jn_r3'].cumsum() - df['_jn_r3']
    df['_cso_r3'] = g['_jo_r3'].cumsum() - df['_jo_r3']
    ratio_has = (df['_cso_r3'] >= 3) & (df['_cs_fights'] > 0)
    ratio = df['_csn_r3'] / df['_cso_r3'].clip(lower=EPS) / (df['_csn_r1'] / df['_cs_fights'].clip(lower=EPS)).clip(lower=EPS)
    df['r3_output_ratio'] = np.where(ratio_has, ratio, np.nan)

    return df[['canon_fighter', 'date'] + CARDIO_FEATURES].rename(columns={'canon_fighter': 'fighter'})


if __name__ == '__main__':
    out = compute_cardio_features_asof()
    print(f'Rows: {len(out):,}  |  Unique fighters: {out["fighter"].nunique():,}')
    for c in CARDIO_FEATURES:
        print(c, out[c].describe())
        print()
