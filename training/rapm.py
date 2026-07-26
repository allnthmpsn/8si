#!/usr/bin/env python3
"""
training/rapm.py — 8SI v2 Stage 2.6, opponent-adjusted ridge RAPM

Regularized Adjusted Plus-Minus: instead of a per-fighter cumulative
average (which conflates "landed a lot" with "fought weak opponents"),
fits ONE ridge regression across every fight in a window simultaneously,
with two columns per fighter (offense, defense) and +1/-1 design-matrix
entries per fight — isolating each fighter's own offensive/defensive
contribution net of who they actually fought. Standard technique from
basketball analytics (APM/RAPM), applied here to sig-strike differential
(rapm_off/def/net) and takedown differential (rapm_grap_off/def/net).

Structurally different from every other Stage 2 family: this is NOT a
per-fighter as-of timeline (compute_*_features_asof(round_stats) ->
DataFrame[fighter, date, ...]). A ridge fit needs the WHOLE training
window's fights at once, so "as-of" here means fold-level, not
fight-level: fit ONCE per walk-forward fold using ONLY fights strictly
before that fold's train_cutoff, producing a SINGLE rating per fighter
for the whole fold — never refit mid-fold, never see a fight from the
test period or later. fit_rapm() takes an already-windowed DataFrame; the
caller (an experiment script, eventually build_dataset() if promoted) is
responsible for the windowing, exactly like compute_elo()'s own
[train_start, train_cutoff) discipline.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import RidgeCV

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from training.train_model1 import DATA
from training.features_kd import RAW_DIR, ROUND_STATS_PATH, NAME_MAP_PATH, _load_fight_minutes, _load_name_map

ALPHA_GRID = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)


def _fight_level_diffs(round_stats_path=ROUND_STATS_PATH, raw_dir=RAW_DIR, name_map_path=NAME_MAP_PATH):
    """One row per (event, bout, fighter): sig_str_rate_per_min and
    td_rate_per_min — THIS fighter's OWN raw landed rate (not a
    differential against the opponent). Response must be the fighter's
    own output, not own-minus-opponent: using the differential makes the
    two rows of every single fight exact negatives of each other (same
    two fighters, y_B = -y_A), which is a perfect antisymmetry ridge
    regression resolves by setting off_f == def_f for every fighter (the
    penalty is symmetric in off/def, so it collapses onto the
    unpenalized-preferred symmetric solution) — caught empirically before
    trusting any RAPM output, not a hypothetical concern (see
    docs/DECISIONS.md). The standard APM formulation avoids this because
    it's each player's own raw rate, only netted against the opponent
    through the design matrix's own_offense minus their_defense
    structure, not pre-subtracted in the response itself."""
    rs = pd.read_parquet(round_stats_path)
    fight_level = rs.groupby(['event', 'bout', 'date', 'fighter'], as_index=False)[
        ['sig_str_landed', 'td_landed']
    ].sum()

    name_map = _load_name_map(name_map_path)
    fight_level['canonical'] = fight_level['fighter'].map(name_map)
    fight_level = fight_level.dropna(subset=['canonical'])
    fight_level = fight_level.drop(columns=['fighter']).rename(columns={'canonical': 'fighter'})

    fight_min = _load_fight_minutes(raw_dir)
    fight_level = fight_level.merge(fight_min, on=['event', 'bout'], how='left')

    pair = fight_level.merge(
        fight_level[['event', 'bout', 'fighter']],
        on=['event', 'bout'], suffixes=('', '_opp'),
    )
    pair = pair[pair['fighter'] != pair['fighter_opp']]
    pair = pair.drop_duplicates(subset=['event', 'bout', 'fighter'])
    pair = pair[pair['fight_min'] > 0]

    pair['sig_str_rate_per_min'] = pair['sig_str_landed'] / pair['fight_min']
    pair['td_rate_per_min'] = pair['td_landed'] / pair['fight_min']
    return pair.rename(columns={'fighter_opp': 'opp_fighter'})[
        ['event', 'bout', 'date', 'fighter', 'opp_fighter', 'sig_str_rate_per_min', 'td_rate_per_min']
    ]


def _design_matrix(fight_level_long, response_col):
    fighters = sorted(set(fight_level_long['fighter']) | set(fight_level_long['opp_fighter']))
    idx = {f: i for i, f in enumerate(fighters)}
    n = len(fighters)
    n_rows = len(fight_level_long)

    off_idx = fight_level_long['fighter'].map(idx).to_numpy()
    def_idx = fight_level_long['opp_fighter'].map(idx).to_numpy()
    row_ids = np.arange(n_rows)

    rows = np.concatenate([row_ids, row_ids])
    cols = np.concatenate([off_idx, n + def_idx])
    data = np.concatenate([np.ones(n_rows), -np.ones(n_rows)])
    X = sparse.csr_matrix((data, (rows, cols)), shape=(n_rows, 2 * n))
    y = fight_level_long[response_col].to_numpy()
    return X, y, fighters


def _fit_one(fight_level_long, response_col, alpha_grid=ALPHA_GRID):
    X, y, fighters = _design_matrix(fight_level_long, response_col)
    model = RidgeCV(alphas=alpha_grid, fit_intercept=True)
    model.fit(X, y)
    n = len(fighters)
    off = model.coef_[:n]
    deff = model.coef_[n:]
    return pd.DataFrame({'fighter': fighters, 'off': off, 'def': deff, 'net': off + deff}), model.alpha_


def fit_rapm(round_stats_path=ROUND_STATS_PATH, raw_dir=RAW_DIR, name_map_path=NAME_MAP_PATH,
             train_start=None, train_cutoff=None, alpha_grid=ALPHA_GRID):
    """Fit both RAPM families (striking, grappling) on fights in
    [train_start, train_cutoff) — the caller's window, this function
    enforces nothing beyond filtering to it. Returns
    (striking_df[fighter, rapm_off, rapm_def, rapm_net],
     grappling_df[fighter, rapm_grap_off, rapm_grap_def, rapm_grap_net],
     alphas_used)."""
    diffs = _fight_level_diffs(round_stats_path, raw_dir, name_map_path)
    if train_start is not None:
        diffs = diffs[diffs['date'] >= pd.Timestamp(train_start)]
    if train_cutoff is not None:
        diffs = diffs[diffs['date'] < pd.Timestamp(train_cutoff)]

    striking, alpha_s = _fit_one(diffs, 'sig_str_rate_per_min', alpha_grid)
    striking = striking.rename(columns={'off': 'rapm_off', 'def': 'rapm_def', 'net': 'rapm_net'})

    grappling, alpha_g = _fit_one(diffs, 'td_rate_per_min', alpha_grid)
    grappling = grappling.rename(columns={'off': 'rapm_grap_off', 'def': 'rapm_grap_def', 'net': 'rapm_grap_net'})

    return striking, grappling, {'striking_alpha': alpha_s, 'grappling_alpha': alpha_g}


if __name__ == '__main__':
    striking, grappling, alphas = fit_rapm(train_start='2015-01-01', train_cutoff='2021-01-01')
    print(f'Fit on [2015-01-01, 2021-01-01): {len(striking):,} fighters, alphas={alphas}')
    print(striking.sort_values('rapm_net', ascending=False).head(10).to_string())
    print()
    print(grappling.sort_values('rapm_grap_net', ascending=False).head(10).to_string())
