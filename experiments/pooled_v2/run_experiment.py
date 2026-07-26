#!/usr/bin/env python3
"""
experiments/pooled_v2/run_experiment.py — 8SI v2 Stage 3.1

Tests pooling women's fights back into the trainer (removing the
WOMENS_CLASSES exclusion — training/train_model1.py's build_dataset()
now takes include_womens=True/False, defaulting to False/current
behavior; nothing is promoted here) against the current men's-only
production feature set on walk-forward pooled log loss, per the spec's
own acceptance bar: pooling must not degrade MEN'S log loss by more than
0.001 while adding women's coverage.

Comparison is apples-to-apples: both variants are SCORED on the exact
same set of men's test fights (the pooled model is trained on more data
— men's + women's — but only its predictions on the men's subset of each
fold's test period are used for the log-loss comparison). Women's-fight
performance is reported separately, for completeness, not as a gate.

Adds is_womens, is_womens_x_elo_dif, is_womens_x_age_dif to the pooled
variant's feature list (computed here, not inside build_dataset() —
trivially derivable from columns build_dataset() already outputs, so no
need to bake this into the trainer before it's proven).
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
from features.constants import WOMENS_CLASSES

FOLD_YEARS = [2021, 2022, 2023, 2024, 2025]
TRAIN_START = '2015-01-01'

POOLED_COLS = ['is_womens', 'is_womens_x_elo_dif', 'is_womens_x_age_dif']


def _add_womens_interactions(df):
    df = df.copy()
    df['is_womens'] = df['weight_class'].isin(WOMENS_CLASSES).astype(int)
    df['is_womens_x_elo_dif'] = df['is_womens'] * df['elo_dif']
    df['is_womens_x_age_dif'] = df['is_womens'] * df['age_dif']
    return df


def run_full_fold(year, feat_list, pooled=False):
    train_cutoff = f'{year}-01-01'
    test_end = f'{year + 1}-01-01'

    df, _, _ = build_dataset(TRAIN_START, train_cutoff, DATA, include_womens=pooled)
    df['target'] = (df['Winner'] == 'Red').astype(int)
    if pooled:
        df = _add_womens_interactions(df)

    train_mask = df['date'] < train_cutoff
    test_mask = (df['date'] >= train_cutoff) & (df['date'] < test_end)

    X_train_raw = df.loc[train_mask, feat_list].reset_index(drop=True)
    y_train_raw = df.loc[train_mask, 'target'].reset_index(drop=True)
    d_train_raw = df.loc[train_mask, 'date'].reset_index(drop=True)
    X_test = df.loc[test_mask, feat_list].reset_index(drop=True)
    y_test = df.loc[test_mask, 'target'].reset_index(drop=True)
    is_womens_test = df.loc[test_mask, 'weight_class'].isin(WOMENS_CLASSES).to_numpy() if pooled else \
        np.zeros(test_mask.sum(), dtype=bool)

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
    return y_test.to_numpy(), p_test, is_womens_test


def run_full_walk_forward(feat_list, label, pooled=False):
    y_all, p_all, is_w_all = [], [], []
    for year in FOLD_YEARS:
        y, p, is_w = run_full_fold(year, feat_list, pooled)
        y_all.append(y)
        p_all.append(p)
        is_w_all.append(is_w)
    y_all = np.concatenate(y_all)
    p_all = np.concatenate(p_all)
    is_w_all = np.concatenate(is_w_all)

    mens_mask = ~is_w_all
    ll_mens = log_loss(y_all[mens_mask], p_all[mens_mask])
    acc_mens = accuracy_score(y_all[mens_mask], (p_all[mens_mask] > 0.5).astype(int))
    brier_mens = brier_score_loss(y_all[mens_mask], p_all[mens_mask])
    print(f'  [{label:<32}] MEN\'S log loss={ll_mens:.4f}  acc={acc_mens:.4f}  brier={brier_mens:.4f}  (n={mens_mask.sum()})')

    if is_w_all.sum() > 0:
        ll_w = log_loss(y_all[is_w_all], p_all[is_w_all])
        acc_w = accuracy_score(y_all[is_w_all], (p_all[is_w_all] > 0.5).astype(int))
        print(f'  [{label:<32}] WOMEN\'S log loss={ll_w:.4f}  acc={acc_w:.4f}  (n={is_w_all.sum()}, informational only)')

    return ll_mens, acc_mens, brier_mens


def main():
    print('=== Baseline: men\'s-only (current production) ===')
    baseline_ll, baseline_acc, baseline_brier = run_full_walk_forward(FEAT_114, 'men\'s-only baseline', pooled=False)

    print('\n=== Candidate: pooled (men\'s + women\'s), scored on men\'s fights only ===')
    pooled_feat = FEAT_114 + POOLED_COLS
    pooled_ll, pooled_acc, pooled_brier = run_full_walk_forward(pooled_feat, 'pooled', pooled=True)

    print(f'\n{"=" * 70}')
    print('  SUMMARY (men\'s-fight pooled walk-forward log loss, lower is better)')
    print(f'{"=" * 70}')
    delta = pooled_ll - baseline_ll
    print(f'  men\'s-only baseline   log_loss={baseline_ll:.4f}  acc={baseline_acc:.4f}  brier={baseline_brier:.4f}')
    print(f'  pooled (men\'s subset)  log_loss={pooled_ll:.4f}  acc={pooled_acc:.4f}  brier={pooled_brier:.4f}  ({delta:+.4f})')

    print(f'\n  Men\'s log-loss delta: {delta:+.4f}  (spec bar: must not exceed +0.001)')
    if delta <= 0.001:
        print('  RESULT: within the spec\'s degradation bar — pooling adds women\'s coverage safely. Promote.')
    else:
        print('  RESULT: exceeds the spec\'s +0.001 degradation bar — do NOT promote, stay men\'s-only.')


if __name__ == '__main__':
    main()
