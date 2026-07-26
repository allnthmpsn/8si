#!/usr/bin/env python3
"""
training/features_situational.py — 8SI v2 Stage 2.5, situational & priors

Four features (the spec's fifth, a travel proxy from fighter-country vs.
event-country, is NOT built here — see "Skipped" below):

  age_x_division   — R_age * weight_class_ord, row-level (no leakage
                      risk: both inputs are properties of the CURRENT
                      fight, known at prediction time, not derived from
                      history — same category as the existing
                      age_x_layoff interaction already in FEAT_114)
  five_round_bout   — no_of_rounds == 5, row-level
  catchweight       — weight_class == 'Catch Weight', row-level
  division_change   — this fighter's PREVIOUS fight (career as-of,
                       shift(1)) was in a different weight class than
                       THIS one. The one genuinely history-dependent
                       feature in this family, computed directly from
                       data/ufc-master.csv's own per-fighter weight_class
                       timeline (no round_stats.parquet needed — this
                       family, unlike 2.1-2.4, has no round-level
                       dependency at all).

Skipped: the spec's travel proxy (fighter country != event country).
data/ufc-master.csv has an event `country` column, but no source in this
codebase carries fighter NATIONALITY — not data/ufc_fighters_final_updated
.csv, not ufcstats.com's own ufc_fighter_details.csv/ufc_fighter_tott.csv
(checked directly: neither has a country/nationality field). Approximating
a fighter's "home country" from their own fight-location history would be
circular (most of their fights are wherever the promotion scheduled them,
not their nationality) rather than a genuine proxy, so this was left out
entirely rather than built on a fabricated signal.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from training.train_model1 import DATA

SITUATIONAL_FEATURES = ['age_x_division', 'five_round_bout', 'catchweight', 'division_change']


def _division_change_asof(master_path=None, data_dir=DATA):
    """Long-format (fighter, date, weight_class) history from
    ufc-master.csv's R/B corners, then shift(1): was the fighter's PRIOR
    fight in a different weight class than the one about to be looked up?
    Returns DataFrame[fighter, date, division_change]."""
    master = pd.read_csv(master_path or os.path.join(data_dir, 'ufc-master.csv'), low_memory=False)
    master['date'] = pd.to_datetime(master['date'])
    master = master[master['Winner'].isin(['Red', 'Blue'])]

    r = master[['R_fighter', 'date', 'weight_class']].rename(columns={'R_fighter': 'fighter'})
    b = master[['B_fighter', 'date', 'weight_class']].rename(columns={'B_fighter': 'fighter'})
    long_df = pd.concat([r, b], ignore_index=True).sort_values(['fighter', 'date']).reset_index(drop=True)
    long_df = long_df.drop_duplicates(subset=['fighter', 'date'])

    g = long_df.groupby('fighter', sort=False)
    prev_wc = g['weight_class'].shift(1)
    long_df['division_change'] = np.where(prev_wc.notna(), (prev_wc != long_df['weight_class']).astype(float), np.nan)
    return long_df[['fighter', 'date', 'division_change']]


def compute_situational_features(df):
    """Row-level features computed directly on build_dataset()'s own
    output columns (age, weight_class, weight_class_ord, no_of_rounds) —
    no merge needed for these three. Returns df with
    R_/B_age_x_division, R_/B_five_round_bout, R_/B_catchweight added
    (the same value for both corners on the fight-level ones, since
    five_round_bout/catchweight are properties of the FIGHT, not the
    fighter — matching how title_bout_bin already works in FEAT_BASE)."""
    df = df.copy()
    df['R_age_x_division'] = df['R_age'] * df['weight_class_ord']
    df['B_age_x_division'] = df['B_age'] * df['weight_class_ord']
    df['five_round_bout'] = (df['no_of_rounds'] == 5).astype(int)
    df['catchweight'] = (df['weight_class'] == 'Catch Weight').astype(int)
    return df


if __name__ == '__main__':
    dc = _division_change_asof()
    print(f'division_change rows: {len(dc):,}  |  fighters: {dc["fighter"].nunique():,}')
    print(dc.dropna(subset=['division_change']).describe())
