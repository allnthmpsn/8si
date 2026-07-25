#!/usr/bin/env python3
"""
training/backfill_clv.py — 8SI Phase 4.3, CLV column backfill

Extends data/value_bet_log.csv with odds_taken, novig_prob_taken,
novig_prob_close, clv_pct — PURELY ADDITIVE. No existing column is
modified. This matters because value_bet_log.csv is not a live P&L log —
its `split` column (train/test) shows it's Model 2B's (ufc_model2b.pkl)
training data. gap/gap_size/m1_prob/m2a_prob/pick_novig/closing_odds are
features that model was fit on; touching their values would silently
invalidate that already-trained model.

Semantics (see 8si_remediation_plan.md Phase 4.3 and docs/DECISIONS.md):
  - closing_odds (existing column) is treated as the CLOSE reference —
    it's already what this dataset recorded per bet.
  - odds_taken is backfilled from data/odds_snapshots.json's EARLIEST
    snapshot for that exact matchup, where one exists — a proxy for the
    price available when the bet would have been placed, distinct from
    closing_odds. Where no snapshot covers a row's matchup (the vast
    majority — snapshots only exist for one recent card, not the
    2018-2026 span this log covers), odds_taken falls back to
    closing_odds itself, and clv_pct is 0 (no measurable line movement
    captured, not a fabricated value).
  - novig_prob_close reuses the existing pick_novig column (already the
    no-vig probability at closing_odds) under its Phase 4.3 name.
  - clv_pct = novig_prob_close - novig_prob_taken, from the bet's own
    picked side.

Idempotent — rerunning recomputes all four columns from scratch each time
(safe to rerun as odds_snapshots.json accumulates more history).
"""
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
LOG_PATH = os.path.join(DATA, 'value_bet_log.csv')
SNAPSHOT_PATH = os.path.join(DATA, 'odds_snapshots.json')


def _decimal_to_american(dec):
    if dec is None or dec <= 1:
        return None
    if dec >= 2.0:
        return round((dec - 1) * 100)
    return round(-100 / (dec - 1))


def _earliest_snapshot_prices(snapshots):
    """fighter-pair (frozenset of two names) -> {name: earliest decimal price}
    from the earliest snapshot timestamp that contains that exact pair."""
    by_pair = {}
    for snap in sorted(snapshots, key=lambda s: s['timestamp']):
        for fight in snap['fights']:
            pair = frozenset((fight['f1'], fight['f2']))
            if pair not in by_pair:
                by_pair[pair] = {fight['f1']: fight['f1_price'], fight['f2']: fight['f2_price']}
    return by_pair


def backfill(log_path=LOG_PATH, snapshot_path=SNAPSHOT_PATH):
    df = pd.read_csv(log_path)

    with open(snapshot_path) as fh:
        snapshots = json.load(fh)
    earliest = _earliest_snapshot_prices(snapshots)

    odds_taken, novig_taken = [], []
    n_matched = 0
    for _, row in df.iterrows():
        pair = frozenset((row['f1_name'], row['f2_name']))
        pick = row['m2a_pick']
        prices = earliest.get(pair)

        if prices is not None and pick in prices:
            other = next(n for n in prices if n != pick)
            pick_dec, other_dec = prices[pick], prices[other]
            pick_implied, other_implied = 1.0 / pick_dec, 1.0 / other_dec
            total = pick_implied + other_implied
            novig_pick = pick_implied / total
            odds_taken.append(_decimal_to_american(pick_dec))
            novig_taken.append(novig_pick)
            n_matched += 1
        else:
            # No snapshot covers this matchup — fall back to the existing
            # closing line itself (no measurable movement captured).
            odds_taken.append(row['closing_odds'])
            novig_taken.append(row['pick_novig'])

    df['odds_taken'] = odds_taken
    df['novig_prob_taken'] = novig_taken
    df['novig_prob_close'] = df['pick_novig']
    df['clv_pct'] = df['novig_prob_close'] - df['novig_prob_taken']

    df.to_csv(log_path, index=False)
    print(f'Backfilled {len(df):,} rows ({n_matched} with a genuine snapshot match, '
          f'{len(df) - n_matched} fell back to closing_odds — no snapshot coverage).')
    print(f'Wrote: {log_path}')


if __name__ == '__main__':
    backfill()
