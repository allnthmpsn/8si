#!/usr/bin/env python3
"""
experiments/elo_v2/run_experiment.py — 8SI Phase 7 Elo upgrades.

Tests Elo variants against the production K=48 / no-method-weighting /
no-layoff-regression baseline, gated on walk-forward log loss (not the
single 2024+ holdout — see docs/REBASELINE.md for why). Behind a flag/
experiment first, per 8si_remediation_plan.md Phase 7: nothing here
changes training/train_model1.py's actual defaults unless a variant wins;
promotion is a separate, explicit edit to that file after reviewing this
script's output.

1. K-sensitivity: grid {24, 32, 40, 48}, evaluated as a LONE-FEATURE
   (elo_dif only) logistic regression across the same 5 walk-forward folds
   training/walk_forward.py uses — cheap enough to run standalone rather
   than through the full 133-feature model.
2. Method-weighted K: K x1.25 for KO/TKO/Sub wins, x0.75 for split
   decisions, x1.0 otherwise. Evaluated through the FULL model (all 133
   features) via the same walk-forward folds, since this changes elo_dif's
   values throughout the whole feature set, not just as a standalone signal.
3. Layoff regression: after >365 days inactive, regress the pre-fight
   rating toward 1500 by a tunable percentage. Same full-model evaluation,
   grid over the plan's stated 10-25% range (checked at the two endpoints;
   narrow further only if one clearly wins).
4. Winners combined, if more than one measurably helps alone.

No hyperparameter here is tuned against any single fold's own result in
isolation — always the pooled walk-forward metric across all 5 folds.
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
    compute_elo, build_dataset, corner_flip, compute_weights, predict_symmetric,
)

FOLD_YEARS = [2021, 2022, 2023, 2024, 2025]
TRAIN_START = '2015-01-01'
K_GRID = (24, 32, 40, 48)   # original Phase 7 grid — kept for the module's own record
K_GRID_V2 = (48, 56, 64, 80, 96)   # 8SI v2 Stage 0.3 extension — see docs/V2_LOG.md
METHOD_MULTIPLIERS = {'KO/TKO': 1.25, 'SUB': 1.25, 'S-DEC': 0.75}
LAYOFF_PCT_GRID = (0.15, 0.25)
LAYOFF_DAYS_THRESHOLD = 365


# ─────────────────────────────────────────────────────────────────────────────
# Part 1: K-sensitivity — lone-feature (elo_dif) LR, cheap
# ─────────────────────────────────────────────────────────────────────────────
def run_k_grid(grid=K_GRID):
    master = pd.read_csv(os.path.join(DATA, 'ufc-master.csv'), low_memory=False)
    master['date'] = pd.to_datetime(master['date'])
    master = master[master['Winner'].isin(['Red', 'Blue'])].copy()

    print(f'=== K-sensitivity grid (lone-feature elo_dif LR), grid={grid} ===')
    pooled_ll_by_k = {}
    for k in grid:
        elo_hist, _ = compute_elo(master, K=k, base=1500.0)
        elo_cols = elo_hist[['fighter', 'date', 'elo_before']]
        r = elo_cols.rename(columns={'fighter': 'R_fighter', 'elo_before': 'R_elo'})
        b = elo_cols.rename(columns={'fighter': 'B_fighter', 'elo_before': 'B_elo'})
        df = pd.merge_asof(master.sort_values('date'), r.sort_values('date'),
                            on='date', by='R_fighter', direction='backward')
        df = pd.merge_asof(df.sort_values('date'), b.sort_values('date'),
                            on='date', by='B_fighter', direction='backward')
        df['R_elo'] = df['R_elo'].fillna(1500.0)
        df['B_elo'] = df['B_elo'].fillna(1500.0)
        df['elo_dif'] = df['R_elo'] - df['B_elo']
        df['target'] = (df['Winner'] == 'Red').astype(int)
        df = df[df['date'] >= TRAIN_START]

        y_pooled, p_pooled = [], []
        for year in FOLD_YEARS:
            train_mask = df['date'] < f'{year}-01-01'
            test_mask = (df['date'] >= f'{year}-01-01') & (df['date'] < f'{year + 1}-01-01')
            X_train = df.loc[train_mask, ['elo_dif']].values
            y_train = df.loc[train_mask, 'target'].values
            X_test = df.loc[test_mask, ['elo_dif']].values
            y_test = df.loc[test_mask, 'target'].values
            lr = LogisticRegression()
            lr.fit(X_train, y_train)
            p_pooled.append(lr.predict_proba(X_test)[:, 1])
            y_pooled.append(y_test)

        y_pooled = np.concatenate(y_pooled)
        p_pooled = np.concatenate(p_pooled)
        pooled_ll = log_loss(y_pooled, p_pooled)
        pooled_acc = accuracy_score(y_pooled, (p_pooled > 0.5).astype(int))
        pooled_ll_by_k[k] = pooled_ll
        print(f'  K={k:>3}: pooled log loss={pooled_ll:.4f}  pooled accuracy={pooled_acc:.4f}')

    best_k = min(pooled_ll_by_k, key=pooled_ll_by_k.get)
    baseline_note = f' vs K=48 {pooled_ll_by_k[48]:.4f}' if 48 in pooled_ll_by_k else ''
    print(f'  Best K (lone-feature signal): {best_k} (log loss {pooled_ll_by_k[best_k]:.4f}{baseline_note})')
    return best_k, pooled_ll_by_k


# ─────────────────────────────────────────────────────────────────────────────
# Part 2/3/4: full-model (133-feature) walk-forward, for variants that
# change elo_dif throughout the whole feature set
# ─────────────────────────────────────────────────────────────────────────────
def run_full_fold(year, elo_kwargs):
    train_cutoff = f'{year}-01-01'
    test_end = f'{year + 1}-01-01'

    df, _, _ = build_dataset(TRAIN_START, train_cutoff, DATA, elo_kwargs=elo_kwargs)
    df['target'] = (df['Winner'] == 'Red').astype(int)

    train_mask = df['date'] < train_cutoff
    test_mask = (df['date'] >= train_cutoff) & (df['date'] < test_end)

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
    return y_test.to_numpy(), p_test


def run_full_walk_forward(elo_kwargs, label):
    y_all, p_all = [], []
    for year in FOLD_YEARS:
        y, p = run_full_fold(year, elo_kwargs)
        y_all.append(y)
        p_all.append(p)
    y_all = np.concatenate(y_all)
    p_all = np.concatenate(p_all)
    ll = log_loss(y_all, p_all)
    acc = accuracy_score(y_all, (p_all > 0.5).astype(int))
    brier = brier_score_loss(y_all, p_all)
    print(f'  [{label:<32}] pooled log loss={ll:.4f}  accuracy={acc:.4f}  brier={brier:.4f}')
    return ll, acc, brier


def main():
    best_k, k_results = run_k_grid()

    print('\n=== Part 2/3: full-model (133-feature) walk-forward ===')
    baseline_ll, baseline_acc, baseline_brier = run_full_walk_forward({}, 'baseline (K=48)')

    candidates = {}
    if best_k != 48:
        candidates[f'K={best_k} only'] = {'K': best_k}
    candidates['method-weighted K'] = {'method_multipliers': METHOD_MULTIPLIERS}
    for pct in LAYOFF_PCT_GRID:
        candidates[f'layoff-regression {int(pct * 100)}%'] = {'layoff_regression': (LAYOFF_DAYS_THRESHOLD, pct)}

    results = {'baseline': (baseline_ll, baseline_acc, baseline_brier)}
    for label, kwargs in candidates.items():
        results[label] = run_full_walk_forward(kwargs, label)

    winners = {label: v for label, v in results.items() if label != 'baseline' and v[0] < baseline_ll}

    print('\n=== Part 4: combining individual winners ===')
    # Winners can collide on the same kwarg key (e.g. two different
    # layoff_regression percentages are mutually exclusive, not
    # combinable) — for each kwarg key contributed by any winner, keep
    # only the value from whichever winning candidate had the BEST
    # (lowest) individual log loss, rather than blindly dict.update()-ing
    # every winner's kwargs (which would silently let the last one in
    # iteration order clobber an earlier, possibly-better, same-key value).
    if len(winners) >= 2:
        winners_by_key = {}
        for label, (ll, _, _) in sorted(winners.items(), key=lambda kv: kv[1][0], reverse=True):
            for key, val in candidates[label].items():
                winners_by_key[key] = (label, val)  # later (better, since sorted worst->best) overwrites
        combined_kwargs = {key: val for key, (_, val) in winners_by_key.items()}
        combined_label = ' + '.join(sorted({label for label, _ in winners_by_key.values()}))
        if combined_kwargs == candidates.get(combined_label):
            print(f'  Best winners collapse to a single already-tested candidate ({combined_label}) — nothing new to combine.')
        else:
            results[combined_label] = run_full_walk_forward(combined_kwargs, combined_label)
    else:
        print('  Fewer than 2 individual winners beat baseline — nothing to combine.')

    print(f'\n{"=" * 70}')
    print('  SUMMARY (pooled walk-forward log loss, lower is better)')
    print(f'{"=" * 70}')
    for label, (ll, acc, brier) in sorted(results.items(), key=lambda kv: kv[1][0]):
        delta = ll - baseline_ll
        marker = '  <-- baseline' if label == 'baseline' else (f'  ({delta:+.4f} vs baseline)' if delta < 0 else f'  ({delta:+.4f}, does not beat baseline)')
        print(f'  {label:<40} log_loss={ll:.4f}  acc={acc:.4f}  brier={brier:.4f}{marker}')

    best_label = min(results, key=lambda k: results[k][0])
    if best_label == 'baseline':
        print('\n  RESULT: no variant beat the baseline. Recommend NOT promoting any Phase 7 change.')
    else:
        print(f'\n  RESULT: "{best_label}" beat baseline by {baseline_ll - results[best_label][0]:.4f} log loss.')
        print('  Review before promoting — this script does not modify train_model1.py.')


if __name__ == '__main__':
    main()
