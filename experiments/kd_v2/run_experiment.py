#!/usr/bin/env python3
"""
experiments/kd_v2/run_experiment.py — 8SI v2 Stage 2.1

Tests the knockdowns/damage feature family (training/features_kd.py)
against the current production feature set on walk-forward pooled log
loss. Behind an experiment first, per the v2 spec's own governing rule 3
("every feature ships only if it improves pooled walk-forward log loss")
and this project's established pattern (experiments/elo_v2/): nothing
here changes training/train_model1.py's actual FEAT_114 list unless a
variant wins; promotion is a separate, explicit edit after reviewing
this script's output.

Current Stage 2 baseline (post fighter-identity-fix, see docs/V2_LOG.md):
pooled log loss 0.6134525. This script's own "baseline" run should
reproduce that number as a sanity check that nothing about the harness
itself has drifted.

Candidates tested:
  1. FEAT_114 + KD family (12 new columns: 4 stats x R/B/dif)
  2. FEAT_114 + KD family, WITH got_finished_rate/finish_danger_mismatch
     retired (spec's explicit instruction: "retire got_finished_rate and
     finish_danger_mismatch if the KD family strictly dominates them —
     test with and without")
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
from training.features_kd import compute_kd_features_asof, KD_FEATURES

FOLD_YEARS = [2021, 2022, 2023, 2024, 2025]
TRAIN_START = '2015-01-01'

RETIRE_ON_KD = ['R_finish_danger', 'B_finish_danger', 'finish_danger_mismatch',
                'R_got_finished_rate', 'B_got_finished_rate', 'R_gf_missing', 'B_gf_missing']


def _merge_kd_features(df, train_cutoff, kd_full):
    r_kd = kd_full.rename(columns={'fighter': 'R_fighter', **{f: f'R_{f}' for f in KD_FEATURES}})
    b_kd = kd_full.rename(columns={'fighter': 'B_fighter', **{f: f'B_{f}' for f in KD_FEATURES}})
    df = pd.merge_asof(df.sort_values('date'), r_kd.sort_values('date'),
                        on='date', by='R_fighter', direction='backward')
    df = pd.merge_asof(df.sort_values('date'), b_kd.sort_values('date'),
                        on='date', by='B_fighter', direction='backward')

    df['R_kd_missing'] = df['R_damage_ratio'].isna().astype(int)
    df['B_kd_missing'] = df['B_damage_ratio'].isna().astype(int)

    train_mask = df['date'] < train_cutoff
    _impute_by_weight_class(df, KD_FEATURES, 'weight_class', train_mask)

    for feat in KD_FEATURES:
        df[f'{feat}_dif'] = df[f'R_{feat}'] - df[f'B_{feat}']
    return df


KD_COLS = [f'R_{f}' for f in KD_FEATURES] + [f'B_{f}' for f in KD_FEATURES] + [f'{f}_dif' for f in KD_FEATURES]


def run_full_fold(year, feat_list, kd_full=None):
    train_cutoff = f'{year}-01-01'
    test_end = f'{year + 1}-01-01'

    df, _, _ = build_dataset(TRAIN_START, train_cutoff, DATA)
    df['target'] = (df['Winner'] == 'Red').astype(int)
    if kd_full is not None:
        df = _merge_kd_features(df, train_cutoff, kd_full)

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


def run_full_walk_forward(feat_list, label, kd_full=None):
    y_all, p_all = [], []
    for year in FOLD_YEARS:
        y, p = run_full_fold(year, feat_list, kd_full)
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
    kd_full = compute_kd_features_asof()
    print(f'KD family: {len(kd_full):,} rows, {kd_full["fighter"].nunique():,} fighters')

    print('\n=== Sanity: baseline (current FEAT_114, no KD) ===')
    baseline_ll, baseline_acc, baseline_brier = run_full_walk_forward(FEAT_114, 'baseline (no KD)')

    print('\n=== Candidate 1: FEAT_114 + KD family ===')
    feat_with_kd = FEAT_114 + KD_COLS
    ll1, acc1, brier1 = run_full_walk_forward(feat_with_kd, '+KD family', kd_full)

    print('\n=== Candidate 2: FEAT_114 + KD family, retire got_finished_rate/finish_danger_mismatch ===')
    feat_kd_retire = [f for f in FEAT_114 if f not in RETIRE_ON_KD] + KD_COLS
    ll2, acc2, brier2 = run_full_walk_forward(feat_kd_retire, '+KD, retire gf/finish_danger', kd_full)

    print(f'\n{"=" * 70}')
    print('  SUMMARY (pooled walk-forward log loss, lower is better)')
    print(f'{"=" * 70}')
    results = {
        'baseline (no KD)': (baseline_ll, baseline_acc, baseline_brier),
        '+KD family': (ll1, acc1, brier1),
        '+KD, retire gf/finish_danger': (ll2, acc2, brier2),
    }
    for label, (ll, acc, brier) in sorted(results.items(), key=lambda kv: kv[1][0]):
        delta = ll - baseline_ll
        marker = '  <-- baseline' if label == 'baseline (no KD)' else (
            f'  ({delta:+.4f} vs baseline)' if delta < 0 else f'  ({delta:+.4f}, does not beat baseline)')
        print(f'  {label:<32} log_loss={ll:.4f}  acc={acc:.4f}  brier={brier:.4f}{marker}')

    best_label = min(results, key=lambda k: results[k][0])
    if best_label == 'baseline (no KD)':
        print('\n  RESULT: KD family does not beat baseline. Recommend NOT promoting.')
    else:
        print(f'\n  RESULT: "{best_label}" beat baseline by {baseline_ll - results[best_label][0]:.4f} log loss.')
        print('  Review before promoting — this script does not modify train_model1.py.')


if __name__ == '__main__':
    main()
