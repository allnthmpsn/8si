#!/usr/bin/env python3
"""
experiments/situational_v2/run_experiment.py — 8SI v2 Stage 2.5

Tests the situational/priors feature family (training/features_situational
.py: age_x_division, five_round_bout, catchweight, division_change)
against the current production feature set on walk-forward pooled log
loss. Same behind-a-flag pattern as the other Stage 2 experiment scripts.

Baseline (post fighter-identity-fix; 2.1-2.4 didn't ship): 0.6134525.
No existing feature to test retiring — this family has no snapshot-era
predecessor in FEAT_114.
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
    build_dataset, corner_flip, compute_weights, predict_symmetric,
)
from training.features_situational import compute_situational_features, _division_change_asof

FOLD_YEARS = [2021, 2022, 2023, 2024, 2025]
TRAIN_START = '2015-01-01'

SITUATIONAL_COLS = ['R_age_x_division', 'B_age_x_division', 'age_x_division_dif',
                     'five_round_bout', 'catchweight', 'R_division_change', 'B_division_change']


def _merge_situational_features(df, dc_full):
    df = compute_situational_features(df)
    df['age_x_division_dif'] = df['R_age_x_division'] - df['B_age_x_division']

    r = dc_full.rename(columns={'fighter': 'R_fighter', 'division_change': 'R_division_change'})
    b = dc_full.rename(columns={'fighter': 'B_fighter', 'division_change': 'B_division_change'})
    df = pd.merge_asof(df.sort_values('date'), r.sort_values('date'),
                        on='date', by='R_fighter', direction='backward')
    df = pd.merge_asof(df.sort_values('date'), b.sort_values('date'),
                        on='date', by='B_fighter', direction='backward')
    df['R_division_change'] = df['R_division_change'].fillna(0.0)
    df['B_division_change'] = df['B_division_change'].fillna(0.0)
    return df


def run_full_fold(year, feat_list, dc_full=None):
    train_cutoff = f'{year}-01-01'
    test_end = f'{year + 1}-01-01'

    df, _, _ = build_dataset(TRAIN_START, train_cutoff, DATA)
    df['target'] = (df['Winner'] == 'Red').astype(int)
    if dc_full is not None:
        df = _merge_situational_features(df, dc_full)

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


def run_full_walk_forward(feat_list, label, dc_full=None):
    y_all, p_all = [], []
    for year in FOLD_YEARS:
        y, p = run_full_fold(year, feat_list, dc_full)
        y_all.append(y)
        p_all.append(p)
    y_all = np.concatenate(y_all)
    p_all = np.concatenate(p_all)
    ll = log_loss(y_all, p_all)
    acc = accuracy_score(y_all, (p_all > 0.5).astype(int))
    brier = brier_score_loss(y_all, p_all)
    print(f'  [{label:<40}] pooled log loss={ll:.4f}  accuracy={acc:.4f}  brier={brier:.4f}')
    return ll, acc, brier


def main():
    dc_full = _division_change_asof()
    print(f'division_change: {len(dc_full):,} rows, {dc_full["fighter"].nunique():,} fighters')

    print('\n=== Sanity: baseline (current FEAT_114, no situational) ===')
    baseline_ll, baseline_acc, baseline_brier = run_full_walk_forward(FEAT_114, 'baseline (no situational)')

    print('\n=== Candidate: FEAT_114 + situational family ===')
    feat_with = FEAT_114 + SITUATIONAL_COLS
    ll1, acc1, brier1 = run_full_walk_forward(feat_with, '+situational family', dc_full)

    print(f'\n{"=" * 70}')
    print('  SUMMARY (pooled walk-forward log loss, lower is better)')
    print(f'{"=" * 70}')
    delta = ll1 - baseline_ll
    marker = f'({delta:+.4f} vs baseline)' if delta < 0 else f'({delta:+.4f}, does not beat baseline)'
    print(f'  baseline (no situational)      log_loss={baseline_ll:.4f}  acc={baseline_acc:.4f}  brier={baseline_brier:.4f}  <-- baseline')
    print(f'  +situational family             log_loss={ll1:.4f}  acc={acc1:.4f}  brier={brier1:.4f}  {marker}')

    if ll1 < baseline_ll:
        print(f'\n  RESULT: situational family beat baseline by {baseline_ll - ll1:.4f} log loss.')
        print('  Review before promoting — this script does not modify train_model1.py.')
    else:
        print('\n  RESULT: situational family does not beat baseline. Recommend NOT promoting.')


if __name__ == '__main__':
    main()
