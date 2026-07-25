#!/usr/bin/env python3
"""
training/market_baseline.py — 8SI v2 Stage 0.1

Computes the no-vig market's own pooled performance on the EXACT SAME test
population training/walk_forward.py evaluates the model against (same 5
folds, same build_dataset() row filtering — men's fights, R/B_cum_fights
>= 1, 2015+ window) — the permanent benchmark line every subsequent v2
experiment's pooled log loss gets reported next to (see docs/V2_LOG.md).

Vig removal is proportional (implied_R = 1/dec_odds_R, normalized against
implied_B so the two sum to 1), matching backend/main.py's own _implied()/
market_shrink() convention from 8SI Phase 4.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from training.train_model1 import DATA, build_dataset

FOLD_YEARS = [2021, 2022, 2023, 2024, 2025]
TRAIN_START = '2015-01-01'


def _implied(odds):
    if pd.isna(odds) or odds == 0:
        return np.nan
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)


def _fold_test_rows(year, data_dir):
    """Same test population walk_forward.py's fold for this year uses —
    built via build_dataset() so filtering (men's, cum_fights>=1, date
    window) is identical, not separately re-implemented."""
    train_cutoff = f'{year}-01-01'
    test_end = f'{year + 1}-01-01'
    df, _, _ = build_dataset(TRAIN_START, train_cutoff, data_dir)
    test_mask = (df['date'] >= train_cutoff) & (df['date'] < test_end)
    test_df = df.loc[test_mask, ['date', 'R_odds', 'B_odds', 'Winner']].copy()
    test_df['target'] = (test_df['Winner'] == 'Red').astype(int)
    return test_df


def main(data_dir=DATA):
    print('=' * 62)
    print('  8SI v2 Stage 0.1 — Market baseline (walk-forward test years)')
    print('=' * 62)

    fold_frames = []
    for year in FOLD_YEARS:
        print(f'  Building fold {year} test population...')
        fold_frames.append(_fold_test_rows(year, data_dir))
    test_df = pd.concat(fold_frames, ignore_index=True)

    total_n = len(test_df)
    has_odds = (
        test_df['R_odds'].notna() & test_df['B_odds'].notna()
        & (test_df['R_odds'] != 0) & (test_df['B_odds'] != 0)
    )
    coverage = float(has_odds.mean())
    odds_df = test_df.loc[has_odds].copy()

    odds_df['R_implied'] = odds_df['R_odds'].apply(_implied)
    odds_df['B_implied'] = odds_df['B_odds'].apply(_implied)
    total_implied = odds_df['R_implied'] + odds_df['B_implied']
    odds_df['R_novig'] = odds_df['R_implied'] / total_implied

    fav_pred = (odds_df['R_novig'] > 0.5).astype(int)
    fav_acc = float(accuracy_score(odds_df['target'], fav_pred))
    market_ll = float(log_loss(odds_df['target'], odds_df['R_novig']))
    market_brier = float(brier_score_loss(odds_df['target'], odds_df['R_novig']))

    print(f'\n  Test years: {FOLD_YEARS}')
    print(f'  Total test fights (men\'s, 2015+ window, cum_fights>=1): {total_n:,}')
    print(f'  Odds coverage: {int(has_odds.sum()):,}/{total_n:,} ({coverage * 100:.1f}%)')
    print(f'\n  Pick-favorite accuracy: {fav_acc:.4f}  ({fav_acc * 100:.2f}%)')
    print(f'  Market log loss:        {market_ll:.4f}')
    print(f'  Market Brier:           {market_brier:.4f}')
    print('=' * 62)

    return {
        'test_years': FOLD_YEARS,
        'n_test_total': total_n,
        'n_with_odds': int(has_odds.sum()),
        'odds_coverage': round(coverage, 6),
        'favorite_accuracy': round(fav_acc, 6),
        'market_log_loss': round(market_ll, 6),
        'market_brier': round(market_brier, 6),
    }


if __name__ == '__main__':
    main()
