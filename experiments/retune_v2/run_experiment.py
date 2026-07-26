#!/usr/bin/env python3
"""
experiments/retune_v2/run_experiment.py — 8SI v2 Stage 3.1b

Re-tunes the LR/XGB blend weight and XGB hyperparameters via Optuna,
using the walk-forward folds themselves as the search's evaluation
metric (time-ordered by construction — train on [2015, N), test on N —
never a shuffled k-fold, per the spec's explicit instruction). Also adds
an XGBoost monotonic constraint on elo_dif (higher Elo differential ->
strictly higher or equal P(Red wins), never lower) unconditionally, per
the spec — not itself a search parameter, since it's a correctness
constraint (a model that thinks a HIGHER-rated fighter is LESS likely to
win, even occasionally, is capturing noise, not signal), not a
performance lever to tune.

3.1a (pooling women's fights) was not promoted, so this stays on the
current men's-only FEAT_114 — the same feature set the 0.6134525
baseline was measured on.

Optuna objective = pooled walk-forward log loss (5 folds), minimized.
Nothing here changes training/train_model1.py's actual LR_WEIGHT/
XGB_WEIGHT/XGB_PARAMS defaults — promotion is a separate, explicit edit
after reviewing this script's output, same as every other Stage 2/3
experiment.
"""
import os
import sys

import numpy as np
import optuna
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

optuna.logging.set_verbosity(optuna.logging.WARNING)

FOLD_YEARS = [2021, 2022, 2023, 2024, 2025]
TRAIN_START = '2015-01-01'
N_TRIALS = 25

MONOTONE_CONSTRAINTS = tuple(1 if f == 'elo_dif' else 0 for f in FEAT_114)

# Cache built datasets across trials — the feature set never changes
# between trials (only model hyperparameters do), so rebuilding
# build_dataset() 5x per trial x 25 trials would be pure waste.
_FOLD_CACHE = {}


def _get_fold_data(year):
    if year not in _FOLD_CACHE:
        train_cutoff = f'{year}-01-01'
        test_end = f'{year + 1}-01-01'
        df, _, _ = build_dataset(TRAIN_START, train_cutoff, DATA)
        df['target'] = (df['Winner'] == 'Red').astype(int)
        train_mask = df['date'] < train_cutoff
        test_mask = (df['date'] >= train_cutoff) & (df['date'] < test_end)
        _FOLD_CACHE[year] = (
            df.loc[train_mask, FEAT_114].reset_index(drop=True),
            df.loc[train_mask, 'target'].reset_index(drop=True),
            df.loc[train_mask, 'date'].reset_index(drop=True),
            df.loc[test_mask, FEAT_114].reset_index(drop=True),
            df.loc[test_mask, 'target'].reset_index(drop=True),
        )
    return _FOLD_CACHE[year]


def run_fold(year, lr_weight, lr_c, xgb_params, use_monotone):
    X_train_raw, y_train_raw, d_train_raw, X_test, y_test = _get_fold_data(year)

    w_raw = pd.Series(compute_weights(d_train_raw, half_life_days=HL_DAYS), index=y_train_raw.index)
    X_aug, y_aug, w_aug = corner_flip(X_train_raw, y_train_raw, w_raw)

    model_lr = Pipeline([
        ('sc', RobustScaler()),
        ('lr', LogisticRegression(penalty='l2', C=lr_c, solver='liblinear',
                                   max_iter=2000, random_state=42, n_jobs=1)),
    ])
    model_lr.fit(X_aug, y_aug, lr__sample_weight=w_aug.values)

    params = dict(xgb_params)
    if use_monotone:
        params['monotone_constraints'] = MONOTONE_CONSTRAINTS
    model_xgb = XGBClassifier(**params)
    model_xgb.fit(X_aug, y_aug, sample_weight=w_aug.values)

    p_test = predict_symmetric(model_lr, model_xgb, X_test, lr_weight, 1.0 - lr_weight)
    return y_test.to_numpy(), p_test


def pooled_log_loss(lr_weight, lr_c, xgb_params, use_monotone=True):
    y_all, p_all = [], []
    for year in FOLD_YEARS:
        y, p = run_fold(year, lr_weight, lr_c, xgb_params, use_monotone)
        y_all.append(y)
        p_all.append(p)
    y_all = np.concatenate(y_all)
    p_all = np.concatenate(p_all)
    return log_loss(y_all, p_all), accuracy_score(y_all, (p_all > 0.5).astype(int)), brier_score_loss(y_all, p_all)


def objective(trial):
    lr_weight = trial.suggest_float('lr_weight', 0.3, 0.9)
    lr_c = trial.suggest_float('lr_c', 0.001, 0.05, log=True)
    xgb_params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),
        'random_state': 42, 'eval_metric': 'logloss', 'verbosity': 0, 'n_jobs': 1,
    }
    ll, _, _ = pooled_log_loss(lr_weight, lr_c, xgb_params, use_monotone=True)
    return ll


def main():
    print('=== Sanity: current production params, no monotonic constraint ===')
    ll0, acc0, brier0 = pooled_log_loss(LR_WEIGHT, 0.00711, XGB_PARAMS, use_monotone=False)
    print(f'  current production (unconstrained): log_loss={ll0:.4f}  acc={acc0:.4f}  brier={brier0:.4f}')

    print('\n=== Sanity: current production params, WITH monotonic constraint on elo_dif ===')
    ll_mono, acc_mono, brier_mono = pooled_log_loss(LR_WEIGHT, 0.00711, XGB_PARAMS, use_monotone=True)
    print(f'  current production + monotone:        log_loss={ll_mono:.4f}  acc={acc_mono:.4f}  brier={brier_mono:.4f}')

    print(f'\n=== Optuna search ({N_TRIALS} trials, time-ordered walk-forward folds, monotone always on) ===')
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best = study.best_params
    best_xgb = {
        'n_estimators': best['n_estimators'], 'learning_rate': best['learning_rate'],
        'max_depth': best['max_depth'], 'min_child_weight': best['min_child_weight'],
        'subsample': best['subsample'], 'colsample_bytree': best['colsample_bytree'],
        'gamma': best['gamma'], 'reg_alpha': best['reg_alpha'], 'reg_lambda': best['reg_lambda'],
        'random_state': 42, 'eval_metric': 'logloss', 'verbosity': 0, 'n_jobs': 1,
    }
    ll_best, acc_best, brier_best = pooled_log_loss(best['lr_weight'], best['lr_c'], best_xgb, use_monotone=True)

    print(f'\n{"=" * 70}')
    print('  SUMMARY (pooled walk-forward log loss, lower is better)')
    print(f'{"=" * 70}')
    print(f'  baseline (0.6134525, no monotone)     log_loss=0.6135')
    print(f'  current params + monotone constraint  log_loss={ll_mono:.4f}  ({ll_mono - ll0:+.4f} vs unconstrained)')
    print(f'  Optuna best (found during search)     log_loss={study.best_value:.4f}')
    print(f'  Optuna best (re-verified)              log_loss={ll_best:.4f}  acc={acc_best:.4f}  brier={brier_best:.4f}')
    print(f'\n  Best params: {best}')
    print(f'\n  Gain vs 0.6134525 baseline: {0.6134525 - ll_best:+.4f}')
    if ll_best < ll0:
        print('  RESULT: retuned params beat the current-params-with-monotone baseline. Review before promoting.')
    else:
        print('  RESULT: retuning did not beat current params. Recommend NOT promoting the retuned hyperparameters.')
    print(f'\n  Monotonic constraint alone (current params): {ll0 - ll_mono:+.4f}')


if __name__ == '__main__':
    main()
