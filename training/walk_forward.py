#!/usr/bin/env python3
"""
training/walk_forward.py — 8SI Model 1 walk-forward evaluation (Phase 3.3)

Folds: train on [2015-01-01, N), test on year N, for N in
{2021, 2022, 2023, 2024, 2025}. Each fold is a fresh, independently trained
LR+XGB blend (same hyperparameters as train_model1.main()) — nothing here
reuses a fold's trained model in another fold, and no hyperparameter is
tuned based on any fold's result (including the last one).

Why this script exists: the single 2024+ holdout that train_model1.py
reports has been reused across many experiment sprints while iterating on
this model, which makes it a biased number for model-SELECTION purposes
(you can always find a change that happens to help that one slice). Walk-
forward evaluation pools 5 independent out-of-sample test years, which is a
much harder target to accidentally overfit to. Treat the POOLED metrics
below as the honest number going forward — not the single-holdout number in
model_metadata.json.

Nothing is saved except an optional --out-dir JSON summary; this script
does not write model artifacts.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from training.train_model1 import (
    DATA, MODEL, FEAT_114, LR_WEIGHT, XGB_WEIGHT, HL_DAYS, XGB_PARAMS,
    build_dataset, corner_flip, compute_weights, predict_symmetric,
    calibration_table, print_calibration_table,
)

FOLD_YEARS = [2021, 2022, 2023, 2024, 2025]
TRAIN_START = '2015-01-01'


def run_fold(year, data_dir):
    train_cutoff = f'{year}-01-01'
    test_end = f'{year + 1}-01-01'

    df, _, _ = build_dataset(TRAIN_START, train_cutoff, data_dir)
    df['target'] = (df['Winner'] == 'Red').astype(int)

    train_mask = df['date'] < train_cutoff
    test_mask  = (df['date'] >= train_cutoff) & (df['date'] < test_end)

    X_train_raw = df.loc[train_mask, FEAT_114].reset_index(drop=True)
    y_train_raw = df.loc[train_mask, 'target'].reset_index(drop=True)
    d_train_raw = df.loc[train_mask, 'date'].reset_index(drop=True)
    X_test = df.loc[test_mask, FEAT_114].reset_index(drop=True)
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
    y_pred = (p_test > 0.5).astype(int)

    return {
        'year': year,
        'n_train': len(X_train_raw),
        'n_test': len(X_test),
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'log_loss': float(log_loss(y_test, p_test)),
        'brier': float(brier_score_loss(y_test, p_test)),
        'y_test': y_test.to_numpy(),
        'p_test': p_test,
    }


def main(data_dir=DATA, out_dir=None):
    print('=' * 62)
    print('  8SI Model 1 — Walk-Forward Evaluation (Phase 3.3)')
    print(f'  Folds: train [{TRAIN_START}, N) / test year N, N in {FOLD_YEARS}')
    print('=' * 62)

    fold_results = []
    for year in FOLD_YEARS:
        print(f'\n[Fold {year}] training on [{TRAIN_START}, {year}-01-01), testing on {year}...')
        r = run_fold(year, data_dir)
        fold_results.append(r)
        print(f'  n_train={r["n_train"]:,}  n_test={r["n_test"]:,}  '
              f'accuracy={r["accuracy"]:.4f}  log_loss={r["log_loss"]:.4f}  brier={r["brier"]:.4f}')

    print(f'\n{"=" * 62}')
    print('  Per-fold results')
    print(f'{"=" * 62}')
    print(f'  {"Year":<6}{"N Test":>8}{"Accuracy":>12}{"Log Loss":>12}{"Brier":>10}')
    for r in fold_results:
        print(f'  {r["year"]:<6}{r["n_test"]:>8,}{r["accuracy"]:>12.4f}{r["log_loss"]:>12.4f}{r["brier"]:>10.4f}')

    y_pooled = np.concatenate([r['y_test'] for r in fold_results])
    p_pooled = np.concatenate([r['p_test'] for r in fold_results])
    pooled_acc   = float(accuracy_score(y_pooled, (p_pooled > 0.5).astype(int)))
    pooled_ll    = float(log_loss(y_pooled, p_pooled))
    pooled_brier = float(brier_score_loss(y_pooled, p_pooled))

    print(f'\n  {"POOLED":<6}{len(y_pooled):>8,}{pooled_acc:>12.4f}{pooled_ll:>12.4f}{pooled_brier:>10.4f}')

    print_calibration_table(calibration_table(y_pooled, p_pooled), title='Pooled calibration table')

    print(f'\n{"=" * 62}')
    print('  NOTE: the 2024+ holdout in train_model1.py has been reused across')
    print('  many experiment sprints and is compromised for model-selection')
    print('  purposes. Treat the POOLED metrics above as the honest number.')
    print('  No hyperparameter was changed based on any fold above, including')
    print('  the last one.')
    print(f'{"=" * 62}\n')

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        summary = {
            'fold_years': FOLD_YEARS,
            'train_start': TRAIN_START,
            'per_fold': [
                {k: v for k, v in r.items() if k not in ('y_test', 'p_test')}
                for r in fold_results
            ],
            'pooled_accuracy': pooled_acc,
            'pooled_log_loss': pooled_ll,
            'pooled_brier': pooled_brier,
            'pooled_calibration_table': calibration_table(y_pooled, p_pooled),
        }
        out_path = os.path.join(out_dir, 'walk_forward_results.json')
        with open(out_path, 'w') as fh:
            json.dump(summary, fh, indent=2)
        print(f'Results written to: {out_path}')

    return fold_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Walk-forward evaluation for Model 1')
    parser.add_argument('--out-dir', default=os.path.join(MODEL, 'v2'),
                         help='Where to write walk_forward_results.json (default: model/v2/). '
                              'Pass empty string to skip writing a file.')
    args = parser.parse_args()
    main(out_dir=args.out_dir or None)
