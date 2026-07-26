#!/usr/bin/env python3
"""
training/features_kd.py — 8SI v2 Stage 2.1, knockdowns & damage family

Computes, from data/round_stats.parquet (round-level, ufcstats.com
spelling), four as-of (shift(1), no-leakage) career features per fighter:

  kd_per15_for            — knockdowns landed per 15 min, career as-of
  kd_per15_against         — knockdowns absorbed per 15 min, career as-of
  kd_absorbed_per_sig_str  — knockdowns absorbed per significant strike
                              absorbed (durability)
  damage_ratio             — significant strikes landed / absorbed,
                              career as-of

Same two-step pattern training/style_stats.py established: round-level
rows are first summed to fight-level (one row per fighter per fight),
then a per-fighter cumulative pass computes each stat using ONLY that
fighter's fights strictly before the target date. Uses the same
observed-mask, joint numerator/denominator design as
training/style_stats.py's compute_style_stats_asof() (see docs/DECISIONS.md
for why the naive cumsum()-minus-own-value idiom is unsafe against
missing raw values) rather than repeating that bug on new data.

Total fight time comes from data/raw/ufcstats_rounds/ufc_fight_results.csv
(ROUND + TIME + TIME FORMAT), not round_stats.parquet itself, which has
no explicit per-round duration column. ufc_fight_results.csv's EVENT
column has a systematic trailing-space quirk (every row) that
ufc_fight_stats.csv's doesn't — stripped here before any merge.

Output is keyed by CANONICAL fighter name (via data/name_map.csv) so it
merges directly against ufc-master.csv's R_fighter/B_fighter, same as
every other as-of feature in training/train_model1.py.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from training.train_model1 import DATA

RAW_DIR = os.path.join(DATA, 'raw', 'ufcstats_rounds')
ROUND_STATS_PATH = os.path.join(DATA, 'round_stats.parquet')
NAME_MAP_PATH = os.path.join(DATA, 'name_map.csv')

KD_FEATURES = ['kd_per15_for', 'kd_per15_against', 'kd_absorbed_per_sig_str', 'damage_ratio']

EPS = 1e-9


def _round_len_map(time_format):
    """'5 Rnd (5-5-5-5-5)' -> [5,5,5,5,5]; '3 Rnd + OT (5-5-5-5)' ->
    [5,5,5,5]; unparseable (e.g. 'No Time Limit', pre-2000 exotic formats)
    -> None."""
    m = re.search(r'\(([\d\-]+)\)', str(time_format))
    if not m:
        return None
    try:
        return [int(x) for x in m.group(1).split('-')]
    except ValueError:
        return None


def _total_fight_minutes(row):
    lens = _round_len_map(row['TIME FORMAT'])
    if lens is None:
        return np.nan
    final_round = int(row['ROUND'])
    if final_round < 1 or final_round > len(lens):
        return np.nan
    prior_minutes = sum(lens[:final_round - 1])
    mm, ss = str(row['TIME']).split(':')
    return prior_minutes + int(mm) + int(ss) / 60.0


def _load_fight_minutes(raw_dir=RAW_DIR):
    fr = pd.read_csv(os.path.join(raw_dir, 'ufc_fight_results.csv'))
    fr['EVENT'] = fr['EVENT'].str.strip()
    ev = pd.read_csv(os.path.join(raw_dir, 'ufc_event_details.csv'))
    ev['date'] = pd.to_datetime(ev['DATE'], format='%B %d, %Y')
    fr = fr.merge(ev[['EVENT', 'date']], on='EVENT', how='left')
    fr = fr.dropna(subset=['date'])
    fr['fight_min'] = fr.apply(_total_fight_minutes, axis=1)
    return fr[['EVENT', 'BOUT', 'fight_min']].rename(columns={'EVENT': 'event', 'BOUT': 'bout'})


def _load_name_map(path=NAME_MAP_PATH):
    nm = pd.read_csv(path)
    nm['effective'] = nm['manual_override'].where(
        nm['manual_override'].notna() & (nm['manual_override'] != ''), nm['canonical_name']
    )
    return dict(zip(nm['ufcstats_name'], nm['effective']))


def _fight_level(round_stats_path=ROUND_STATS_PATH, raw_dir=RAW_DIR, name_map_path=NAME_MAP_PATH):
    """One row per (event, bout, canonical fighter): kd, sig_str_landed,
    sig_str_att summed across rounds, fight_min attached, opponent's kd/
    sig_str_landed attached (self-join on event+bout) for the *_against
    stats. Rows whose fighter has no canonical mapping are dropped."""
    rs = pd.read_parquet(round_stats_path)
    fight_level = rs.groupby(['event', 'bout', 'date', 'fighter'], as_index=False)[
        ['kd', 'sig_str_landed', 'sig_str_att']
    ].sum()

    name_map = _load_name_map(name_map_path)
    fight_level['canonical'] = fight_level['fighter'].map(name_map)
    fight_level = fight_level.dropna(subset=['canonical'])

    fight_min = _load_fight_minutes(raw_dir)
    fight_level = fight_level.merge(fight_min, on=['event', 'bout'], how='left')

    # Attach the opponent's kd/sig_str_landed for this same (event, bout) —
    # exactly two canonical fighters per fight; self-join and drop self-pairs.
    pair = fight_level.merge(
        fight_level[['event', 'bout', 'canonical', 'kd', 'sig_str_landed']],
        on=['event', 'bout'], suffixes=('', '_opp'),
    )
    pair = pair[pair['canonical'] != pair['canonical_opp']]
    pair = pair.rename(columns={'kd_opp': 'opp_kd', 'sig_str_landed_opp': 'opp_sig_str_landed'})
    # A fight can appear more than once per fighter pair if ufcstats has a
    # data-entry duplicate; keep the first occurrence per (event, bout, fighter).
    pair = pair.drop_duplicates(subset=['event', 'bout', 'canonical'])
    return pair[['canonical', 'date', 'kd', 'sig_str_landed', 'sig_str_att',
                 'opp_kd', 'opp_sig_str_landed', 'fight_min']].rename(columns={'canonical': 'fighter'})


def compute_kd_features_asof(round_stats_path=ROUND_STATS_PATH, raw_dir=RAW_DIR, name_map_path=NAME_MAP_PATH):
    """Returns DataFrame[fighter, date, kd_per15_for, kd_per15_against,
    kd_absorbed_per_sig_str, damage_ratio] — one row per (canonical
    fighter, fight), values computed from ONLY that fighter's fights
    strictly before `date`. NaN where no prior fight has the relevant
    pair of columns both observed (see the joint observed-mask design in
    training/style_stats.py's compute_style_stats_asof, mirrored here)."""
    df = _fight_level(round_stats_path, raw_dir, name_map_path)
    df = df.sort_values(['fighter', 'date']).reset_index(drop=True)

    # (feature, numerator col, denominator col, per15, is_complement)
    PAIRS = [
        ('kd_per15_for', 'kd', 'fight_min', True, False),
        ('kd_per15_against', 'opp_kd', 'fight_min', True, False),
        ('kd_absorbed_per_sig_str', 'opp_kd', 'opp_sig_str_landed', False, False),
        ('damage_ratio', 'sig_str_landed', 'opp_sig_str_landed', False, False),
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
        # Both checks required (see training/style_stats.py's identical
        # fix): _cso>0 alone isn't enough — a fighter can have jointly-
        # observed prior fights where the denominator (e.g. opponent's
        # sig strikes landed) was genuinely zero for every one of them,
        # which without this second check would EPS-clip into a wildly
        # inflated ratio instead of the correct "undefined" NaN.
        has = (df[f'_cso_{feat}'] > 0).to_numpy() & (df[f'_csd_{feat}'] > 0).to_numpy()
        denom = df[f'_csd_{feat}'] / 15.0 if per15 else df[f'_csd_{feat}']
        ratio = df[f'_csn_{feat}'].to_numpy() / denom.clip(lower=EPS).to_numpy()
        if is_complement:
            ratio = 1.0 - ratio
        df[feat] = np.where(has, ratio, nan)

    return df[['fighter', 'date'] + KD_FEATURES]


if __name__ == '__main__':
    out = compute_kd_features_asof()
    print(f'Rows: {len(out):,}  |  Unique fighters: {out["fighter"].nunique():,}')
    print(out.dropna(subset=['damage_ratio']).tail(10).to_string())
