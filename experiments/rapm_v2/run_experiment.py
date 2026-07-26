#!/usr/bin/env python3
"""
experiments/rapm_v2/run_experiment.py — 8SI v2 Stage 2.6

Tests RAPM (training/rapm.py) against the current production feature set
on walk-forward pooled log loss. Same behind-a-flag pattern as every
other Stage 2 experiment script, with one structural difference: RAPM is
refit PER FOLD (each fold's own [train_start, train_cutoff) window, per
training/rapm.py's own fold-level "as-of" discipline — never a per-fight
merge_asof like every other Stage 2 family), and the resulting flat
per-fighter rating table is merged onto every row (train and test alike)
in that fold via a plain merge, not a date-aware one.

Baseline (post fighter-identity-fix; 2.1-2.5 didn't ship): 0.6134525.
Per the spec's own explicit rule for this family: if the pooled log-loss
gain is < 0.002, ship the code disabled rather than drop it outright —
a different bar than 2.1-2.5 got (those just needed to not lose).
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
from training.rapm import fit_rapm

FOLD_YEARS = [2021, 2022, 2023, 2024, 2025]
TRAIN_START = '2015-01-01'

RAPM_COLS_STRIKE = ['R_rapm_off', 'R_rapm_def', 'R_rapm_net', 'B_rapm_off', 'B_rapm_def', 'B_rapm_net',
                     'rapm_off_dif', 'rapm_def_dif', 'rapm_net_dif']
RAPM_COLS_GRAP = ['R_rapm_grap_off', 'R_rapm_grap_def', 'R_rapm_grap_net',
                   'B_rapm_grap_off', 'B_rapm_grap_def', 'B_rapm_grap_net',
                   'rapm_grap_off_dif', 'rapm_grap_def_dif', 'rapm_grap_net_dif']


def _merge_rapm(df, train_cutoff):
    striking, grappling, alphas = fit_rapm(train_start=TRAIN_START, train_cutoff=train_cutoff)
    print(f'    RAPM fit for train_cutoff={train_cutoff}: {len(striking)} fighters, alphas={alphas}')

    for stat_df, cols in ((striking, ['rapm_off', 'rapm_def', 'rapm_net']),
                          (grappling, ['rapm_grap_off', 'rapm_grap_def', 'rapm_grap_net'])):
        r = stat_df.rename(columns={'fighter': 'R_fighter', **{c: f'R_{c}' for c in cols}})
        b = stat_df.rename(columns={'fighter': 'B_fighter', **{c: f'B_{c}' for c in cols}})
        # merge() resets the index each time — recompute train_mask fresh
        # against the CURRENT df rather than reuse a stale-index Series
        # (pandas raises IndexingError on a misaligned boolean indexer,
        # it doesn't silently fall back to positional).
        df = df.merge(r, on='R_fighter', how='left').merge(b, on='B_fighter', how='left').reset_index(drop=True)
        train_mask = df['date'] < train_cutoff
        # Same weight-class-median imputation every other Stage 2 family
        # uses (a fighter debuting inside the test window has no prior
        # fights to fit RAPM on at all — 10x more common in the test period
        # than the train period, since every train-period fighter fought at
        # least once before train_cutoff by construction) — NOT a blind
        # fillna(0.0), which doesn't distinguish weight classes.
        _impute_by_weight_class(df, cols, 'weight_class', train_mask)
        for c in cols:
            df[f'{c}_dif'] = df[f'R_{c}'] - df[f'B_{c}']
    return df


def run_full_fold(year, feat_list, with_rapm=False):
    train_cutoff = f'{year}-01-01'
    test_end = f'{year + 1}-01-01'

    df, _, _ = build_dataset(TRAIN_START, train_cutoff, DATA)
    df['target'] = (df['Winner'] == 'Red').astype(int)
    if with_rapm:
        df = _merge_rapm(df, train_cutoff)

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


def run_full_walk_forward(feat_list, label, with_rapm=False):
    y_all, p_all = [], []
    for year in FOLD_YEARS:
        print(f'  fold {year}...')
        y, p = run_full_fold(year, feat_list, with_rapm)
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
    print('\n=== Sanity: baseline (current FEAT_114, no RAPM) ===')
    baseline_ll, baseline_acc, baseline_brier = run_full_walk_forward(FEAT_114, 'baseline (no RAPM)')

    print('\n=== Candidate: FEAT_114 + RAPM (striking + grappling) ===')
    feat_with = FEAT_114 + RAPM_COLS_STRIKE + RAPM_COLS_GRAP
    ll1, acc1, brier1 = run_full_walk_forward(feat_with, '+RAPM', with_rapm=True)

    print(f'\n{"=" * 70}')
    print('  SUMMARY (pooled walk-forward log loss, lower is better)')
    print(f'{"=" * 70}')
    delta = ll1 - baseline_ll
    marker = f'({delta:+.4f} vs baseline)' if delta < 0 else f'({delta:+.4f}, does not beat baseline)'
    print(f'  baseline (no RAPM)   log_loss={baseline_ll:.4f}  acc={baseline_acc:.4f}  brier={baseline_brier:.4f}  <-- baseline')
    print(f'  +RAPM                 log_loss={ll1:.4f}  acc={acc1:.4f}  brier={brier1:.4f}  {marker}')

    gain = baseline_ll - ll1
    print(f'\n  Gain vs baseline: {gain:+.4f}')
    if gain > 0:
        if gain >= 0.002:
            print('  RESULT: RAPM beats baseline by >= 0.002 — promote.')
        else:
            print('  RESULT: RAPM beats baseline but by < 0.002 — per the spec, ship the code disabled, note in V2_LOG.md.')
    else:
        print('  RESULT: RAPM does not beat baseline at all — ship the code disabled, note in V2_LOG.md.')


if __name__ == '__main__':
    main()
