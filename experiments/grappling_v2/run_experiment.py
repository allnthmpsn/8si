#!/usr/bin/env python3
"""
experiments/grappling_v2/run_experiment.py — 8SI v2 Stage 2.2

Tests the control/grappling feature family (training/features_grappling.py)
against the current production feature set on walk-forward pooled log
loss. Same behind-a-flag pattern as experiments/elo_v2/ and
experiments/kd_v2/ — nothing here changes training/train_model1.py.

Baseline (post fighter-identity-fix, Stage 2.1 didn't ship): 0.6134525.

Candidates:
  1. FEAT_114 + grappling family (10 new columns: 5 stats x R/B, +5 dif)
  2. FEAT_114 + grappling family, with R/B_TD_Avg, R/B_TD_Def, TD_Avg_dif,
     TD_Def_dif retired — the spec's explicit "replacing the snapshot-era
     TD_Avg lineage where the new versions dominate" instruction.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from training.train_model1 import (
    DATA, FEAT_114, LR_WEIGHT, XGB_WEIGHT, HL_DAYS, XGB_PARAMS,
    build_dataset, corner_flip, compute_weights, predict_symmetric, _impute_by_weight_class,
)
from training.features_grappling import compute_grappling_features_asof, GRAPPLING_FEATURES

FOLD_YEARS = [2021, 2022, 2023, 2024, 2025]
TRAIN_START = '2015-01-01'

RETIRE_ON_GRAPPLING = ['R_TD_Avg', 'B_TD_Avg', 'TD_Avg_dif', 'R_TD_Def', 'B_TD_Def', 'TD_Def_dif']


def _merge_grappling_features(df, train_cutoff, gr_full):
    r = gr_full.rename(columns={'fighter': 'R_fighter', **{f: f'R_{f}' for f in GRAPPLING_FEATURES}})
    b = gr_full.rename(columns={'fighter': 'B_fighter', **{f: f'B_{f}' for f in GRAPPLING_FEATURES}})
    df = pd.merge_asof(df.sort_values('date'), r.sort_values('date'),
                        on='date', by='R_fighter', direction='backward')
    df = pd.merge_asof(df.sort_values('date'), b.sort_values('date'),
                        on='date', by='B_fighter', direction='backward')

    df['R_grap_missing'] = df['R_ctrl_pct_for'].isna().astype(int)
    df['B_grap_missing'] = df['B_ctrl_pct_for'].isna().astype(int)

    train_mask = df['date'] < train_cutoff
    _impute_by_weight_class(df, GRAPPLING_FEATURES, 'weight_class', train_mask)

    for feat in GRAPPLING_FEATURES:
        df[f'{feat}_dif'] = df[f'R_{feat}'] - df[f'B_{feat}']
    return df


GRAP_COLS = ([f'R_{f}' for f in GRAPPLING_FEATURES] + [f'B_{f}' for f in GRAPPLING_FEATURES]
             + [f'{f}_dif' for f in GRAPPLING_FEATURES])


def run_full_fold(year, feat_list, gr_full=None):
    train_cutoff = f'{year}-01-01'
    test_end = f'{year + 1}-01-01'

    df, _, _ = build_dataset(TRAIN_START, train_cutoff, DATA)
    df['target'] = (df['Winner'] == 'Red').astype(int)
    if gr_full is not None:
        df = _merge_grappling_features(df, train_cutoff, gr_full)

    train_mask = df['date'] < train_cutoff
    test_mask = (df['date'] >= train_cutoff) & (df['date'] < test_end)

    X_train_raw = df.loc[train_mask, feat_list].reset_index(drop=True)
    y_train_raw = df.loc[train_mask, 'target'].reset_index(drop=True)
    d_train_raw = df.loc[train_mask, 'date'].reset_index(drop=True)
    X_test = df.loc[test_mask, feat_list].reset_index(drop=True)
    y_test = df.loc[test_mask, 'target'].reset_index(drop=True)

    w_raw = pd.Series(compute_weights(d_train_raw, half_life_days=HL_DAYS), index=y_train_raw.index)
    X_aug, y_aug, w_aug = corner_flip(X_train_raw, y_train_raw, w_raw)

    model_lr = Pipeline([
        ('sc', RobustScaler()),
        ('lr', LogisticRegression(penalty='l2', C=0.00711, solver='liblinear',
                                   max_iter=2000, random_state=42, n_jobs=1)),
    ])
    model_lr.fit(X_aug, y_aug, lr__sample_weight=w_aug.values)
    model_xgb = XGBClassifier(**XGB_PARAMS)
    model_xgb.fit(X_aug, y_aug, sample_weight=w_aug.values)

    p_test = predict_symmetric(model_lr, model_xgb, X_test, LR_WEIGHT, XGB_WEIGHT)
    return y_test.to_numpy(), p_test


def run_full_walk_forward(feat_list, label, gr_full=None):
    y_all, p_all = [], []
    for year in FOLD_YEARS:
        y, p = run_full_fold(year, feat_list, gr_full)
        y_all.append(y)
        p_all.append(p)
    y_all = np.concatenate(y_all)
    p_all = np.concatenate(p_all)
    ll = log_loss(y_all, p_all)
    acc = accuracy_score(y_all, (p_all > 0.5).astype(int))
    brier = brier_score_loss(y_all, p_all)
    print(f'  [{label:<45}] pooled log loss={ll:.4f}  accuracy={acc:.4f}  brier={brier:.4f}')
    return ll, acc, brier


def main():
    gr_full = compute_grappling_features_asof()
    print(f'Grappling family: {len(gr_full):,} rows, {gr_full["fighter"].nunique():,} fighters')

    print('\n=== Sanity: baseline (current FEAT_114, no grappling) ===')
    baseline_ll, baseline_acc, baseline_brier = run_full_walk_forward(FEAT_114, 'baseline (no grappling)')

    print('\n=== Candidate 1: FEAT_114 + grappling family ===')
    feat_with = FEAT_114 + GRAP_COLS
    ll1, acc1, brier1 = run_full_walk_forward(feat_with, '+grappling family', gr_full)

    print('\n=== Candidate 2: FEAT_114 + grappling family, retire TD_Avg/TD_Def lineage ===')
    feat_retire = [f for f in FEAT_114 if f not in RETIRE_ON_GRAPPLING] + GRAP_COLS
    ll2, acc2, brier2 = run_full_walk_forward(feat_retire, '+grappling, retire TD_Avg/TD_Def', gr_full)

    print(f'\n{"=" * 70}')
    print('  SUMMARY (pooled walk-forward log loss, lower is better)')
    print(f'{"=" * 70}')
    results = {
        'baseline (no grappling)': (baseline_ll, baseline_acc, baseline_brier),
        '+grappling family': (ll1, acc1, brier1),
        '+grappling, retire TD_Avg/TD_Def': (ll2, acc2, brier2),
    }
    for label, (ll, acc, brier) in sorted(results.items(), key=lambda kv: kv[1][0]):
        delta = ll - baseline_ll
        marker = '  <-- baseline' if label == 'baseline (no grappling)' else (
            f'  ({delta:+.4f} vs baseline)' if delta < 0 else f'  ({delta:+.4f}, does not beat baseline)')
        print(f'  {label:<38} log_loss={ll:.4f}  acc={acc:.4f}  brier={brier:.4f}{marker}')

    best_label = min(results, key=lambda k: results[k][0])
    if best_label == 'baseline (no grappling)':
        print('\n  RESULT: grappling family does not beat baseline. Recommend NOT promoting.')
    else:
        print(f'\n  RESULT: "{best_label}" beat baseline by {baseline_ll - results[best_label][0]:.4f} log loss.')
        print('  Review before promoting — this script does not modify train_model1.py.')


if __name__ == '__main__':
    main()
