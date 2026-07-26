#!/usr/bin/env python3
"""
experiments/glicko_v2/run_experiment.py — 8SI v2 Stage 3.3

Tests rd_max = max(R_rd, B_rd) (training/ratings.py's Glicko-2 rating
deviation, an uncertainty measure — not a skill rating) as a model
FEATURE against the current production feature set on walk-forward
pooled log loss. Baseline is now 0.6094540 (post 3.1b's promoted
blend/hyperparameter retune, not the pre-3.1 0.6134525 figure) — this is
the first Stage 3 experiment to run after a real promotion, so it uses
current train_model1.py params directly rather than a separately-passed
baseline config.

Per the spec: ship the FEATURE only if walk-forward improves. The BET
GATE use (Stage 4, gating on rd_max being below some percentile of
historical RD) needs no training and ships regardless of this result —
not built here, since Stage 4 (betting layer) hasn't started yet.
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
    DATA, FEAT_114, LR_WEIGHT, XGB_WEIGHT, LR_C, HL_DAYS, XGB_PARAMS,
    build_dataset, corner_flip, compute_weights, predict_symmetric, _impute_by_weight_class,
)
from training.ratings import compute_glicko2

FOLD_YEARS = [2021, 2022, 2023, 2024, 2025]
TRAIN_START = '2015-01-01'


def _merge_glicko(df, train_cutoff, glicko_hist):
    r = glicko_hist[['fighter', 'date', 'rd_before']].rename(columns={'fighter': 'R_fighter', 'rd_before': 'R_rd'})
    b = glicko_hist[['fighter', 'date', 'rd_before']].rename(columns={'fighter': 'B_fighter', 'rd_before': 'B_rd'})
    df = pd.merge_asof(df.sort_values('date'), r.sort_values('date'),
                        on='date', by='R_fighter', direction='backward').reset_index(drop=True)
    df = pd.merge_asof(df.sort_values('date'), b.sort_values('date'),
                        on='date', by='B_fighter', direction='backward').reset_index(drop=True)
    train_mask = df['date'] < train_cutoff
    _impute_by_weight_class(df, ['rd'], 'weight_class', train_mask)
    df['rd_max'] = df[['R_rd', 'B_rd']].max(axis=1)
    return df


def run_full_fold(year, feat_list, with_glicko=False, glicko_hist=None):
    train_cutoff = f'{year}-01-01'
    test_end = f'{year + 1}-01-01'

    df, _, _ = build_dataset(TRAIN_START, train_cutoff, DATA)
    df['target'] = (df['Winner'] == 'Red').astype(int)
    if with_glicko:
        df = _merge_glicko(df, train_cutoff, glicko_hist)

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
        ('lr', LogisticRegression(penalty='l2', C=LR_C, solver='liblinear',
                                   max_iter=2000, random_state=42, n_jobs=1)),
    ])
    model_lr.fit(X_aug, y_aug, lr__sample_weight=w_aug.values)

    xgb_params = dict(XGB_PARAMS)
    if with_glicko:
        # monotone_constraints length must match the feature list — the
        # promoted tuple only covers FEAT_114; extend with a neutral 0
        # for rd_max (no monotonic assumption about uncertainty itself).
        xgb_params['monotone_constraints'] = xgb_params['monotone_constraints'] + (0,)
    model_xgb = XGBClassifier(**xgb_params)
    model_xgb.fit(X_aug, y_aug, sample_weight=w_aug.values)

    p_test = predict_symmetric(model_lr, model_xgb, X_test, LR_WEIGHT, XGB_WEIGHT)
    return y_test.to_numpy(), p_test


def run_full_walk_forward(feat_list, label, with_glicko=False, glicko_hist=None):
    y_all, p_all = [], []
    for year in FOLD_YEARS:
        y, p = run_full_fold(year, feat_list, with_glicko, glicko_hist)
        y_all.append(y)
        p_all.append(p)
    y_all = np.concatenate(y_all)
    p_all = np.concatenate(p_all)
    ll = log_loss(y_all, p_all)
    acc = accuracy_score(y_all, (p_all > 0.5).astype(int))
    brier = brier_score_loss(y_all, p_all)
    print(f'  [{label:<30}] pooled log loss={ll:.4f}  accuracy={acc:.4f}  brier={brier:.4f}')
    return ll, acc, brier


def main():
    print('=== Sanity: baseline (current FEAT_114, no rd_max) ===')
    baseline_ll, baseline_acc, baseline_brier = run_full_walk_forward(FEAT_114, 'baseline (no rd_max)')

    print('\n=== Computing Glicko-2 history ===')
    master = pd.read_csv(os.path.join(DATA, 'ufc-master.csv'), low_memory=False)
    master['date'] = pd.to_datetime(master['date'])
    glicko_hist = compute_glicko2(master)
    print(f'  {len(glicko_hist):,} rows, {glicko_hist["fighter"].nunique():,} fighters')

    print('\n=== Candidate: FEAT_114 + rd_max ===')
    feat_with = FEAT_114 + ['rd_max']
    ll1, acc1, brier1 = run_full_walk_forward(feat_with, '+rd_max', with_glicko=True, glicko_hist=glicko_hist)

    print(f'\n{"=" * 70}')
    print('  SUMMARY (pooled walk-forward log loss, lower is better)')
    print(f'{"=" * 70}')
    delta = ll1 - baseline_ll
    marker = f'({delta:+.4f} vs baseline)' if delta < 0 else f'({delta:+.4f}, does not beat baseline)'
    print(f'  baseline (no rd_max)  log_loss={baseline_ll:.4f}  acc={baseline_acc:.4f}  brier={baseline_brier:.4f}  <-- baseline')
    print(f'  +rd_max                log_loss={ll1:.4f}  acc={acc1:.4f}  brier={brier1:.4f}  {marker}')

    if ll1 < baseline_ll:
        print(f'\n  RESULT: rd_max beats baseline by {baseline_ll - ll1:.4f} log loss — ship as a feature.')
    else:
        print('\n  RESULT: rd_max does not beat baseline as a feature — do not ship as a feature.')
        print('  The BET GATE use (Stage 4) is unaffected — it ships regardless, needs no training.')


if __name__ == '__main__':
    main()
