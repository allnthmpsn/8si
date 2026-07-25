#!/usr/bin/env python3
"""
training/clv_report.py — 8SI Phase 4.3, CLV report

Reports per-archetype (CONFIRM_DOG / CONFIRM_FAV) mean closing-line value
(CLV) and % of bets that beat the close, from data/value_bet_log.csv (after
training/backfill_clv.py has populated the CLV columns).

Why CLV, not ROI: with well under 300 bets in each archetype slice here,
ROI is noise-dominated — a handful of coin-flip outcomes swings it wildly.
CLV (did we get a better price than the market eventually settled on) is
measurable per-bet and doesn't require waiting for outcomes to resolve,
making it the more reliable signal at this sample size. See
8si_remediation_plan.md Phase 4.3 and its "What NOT to do" section.
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
LOG_PATH = os.path.join(DATA, 'value_bet_log.csv')


def _archetype(row):
    if row['m1_m2a_agree'] == 0:
        return 'SPLIT'
    if row['vegas_agree'] == 0 and row['gap_direction'] >= 0:
        return 'CONFIRM_DOG'
    if row['vegas_agree'] == 1 and row['gap_direction'] >= 0:
        return 'CONFIRM_FAV'
    return 'NO_EDGE'


def main(log_path=LOG_PATH, include_void=False):
    df = pd.read_csv(log_path)
    if 'clv_pct' not in df.columns:
        raise RuntimeError(f'{log_path} has no clv_pct column — run training/backfill_clv.py first.')

    # 8SI v2 Stage 0.4: all pre-v2 bets are formally void (see docs/V2_LOG.md
    # and the v2 build spec's governing rules — "all prior v1 ROI numbers are
    # formally void"). Headline numbers below default to v2-provenance rows
    # only, so old, already-discredited CLV/ROI figures can't leak back in
    # just because they're still sitting in the log. Pass include_void=True
    # (or --include-void) to see the full history anyway.
    n_void = 0
    if 'provenance' in df.columns:
        n_void = int((df['provenance'] == 'v1_void').sum())
        if not include_void:
            df = df[df['provenance'] != 'v1_void']

    df['archetype'] = df.apply(_archetype, axis=1)

    print('=' * 62)
    print('  8SI CLV Report — data/value_bet_log.csv')
    print('=' * 62)
    if n_void and not include_void:
        print(f'\n  {n_void:,} v1_void row(s) excluded (pre-v2, formally void — '
              f'pass include_void=True to include).')

    for archetype in ('CONFIRM_DOG', 'CONFIRM_FAV'):
        sub = df[df['archetype'] == archetype]
        n = len(sub)
        if n == 0:
            print(f'\n{archetype}: no bets logged.')
            continue

        mean_clv    = sub['clv_pct'].mean()
        n_snap      = int((sub['odds_taken'] != sub['closing_odds']).sum())
        beat_close  = int((sub['clv_pct'] > 0).sum())
        pct_beat    = beat_close / n * 100

        print(f'\n{archetype}  (n={n:,}, {n_snap} with genuine snapshot-backed CLV)')
        print(f'  Mean CLV:        {mean_clv * 100:+.2f}pp')
        print(f'  % beating close: {pct_beat:.1f}%  ({beat_close}/{n})')
        if 'pick_won' in sub.columns:
            print(f'  Win rate (informational, NOT ROI): {sub["pick_won"].mean() * 100:.1f}%')

    n_snap_total = int((df['odds_taken'] != df['closing_odds']).sum())
    print(f'\n{"=" * 62}')
    print(f'  Total logged bets: {len(df):,} — CONFIRM_DOG + CONFIRM_FAV are the')
    print(f'  archetypes flagged as bettable by bet_recommendation().')
    if len(df) == 0:
        print('  Snapshot-backed CLV coverage: n/a — no non-void bets logged yet.')
    else:
        print(f'  Snapshot-backed CLV coverage: {n_snap_total}/{len(df):,} rows '
              f'({n_snap_total / len(df) * 100:.1f}%) — see training/backfill_clv.py.')
    print('  NOTE: ROI/win-rate on this many bets is not evidence of edge — a')
    print('  handful of coin-flip outcomes swings it. Do not treat any historical')
    print('  ROI figure in this codebase (e.g. "CONFIRM_DOG +90.8% ROI") as')
    print('  validated. CLV is the arbiter going forward: it is measurable')
    print('  per-bet without waiting on outcomes, and does not require a large')
    print('  sample to be directionally meaningful.')
    print(f'{"=" * 62}\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-void', action='store_true',
                         help='Include pre-v2 (v1_void) rows in the headline numbers.')
    args = parser.parse_args()
    main(include_void=args.include_void)
