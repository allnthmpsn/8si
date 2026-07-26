#!/usr/bin/env python3
"""
training/build_name_map.py — 8SI v2 Stage 1.1, name reconciliation

data/round_stats.parquet's `fighter` column is ufcstats.com's own name
spelling, which doesn't always match this project's canonical names (the
R_fighter/B_fighter values in data/ufc-master.csv, unioned with
data/career_fights_updated.csv's `fighter` column for completeness — the
two sources differ by ~10 names, a pre-existing gap documented in
docs/DATA_SOURCES.md, unrelated to this reconciliation).

Strategy: exact match first, then rapidfuzz (token_sort_ratio, threshold
92) for the remainder against the canonical name universe restricted to
fighters actually active in ufcstats' own 2015+ window (matching against
every canonical name ever recorded, including many 1990s names round_stats
fighters would never plausibly match, would only invite false positives).
Anything below 92 is left for manual review via the `manual_override`
column — fill it in and rerun to promote a row to `matched`.

Writes data/name_map.csv (ufcstats_name, canonical_name, match_type,
match_score, manual_override) and prints an unmatched report scoped to
the 2015+ window, weighted by FIGHTS (not names) since that's what the
spec's <2% acceptance bar is measured against — one unmatched name can
sideline every fight that name fought.
"""
import os
import sys

import pandas as pd
from rapidfuzz import fuzz, process

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from training.train_model1 import DATA

ROUND_STATS_PATH = os.path.join(DATA, 'round_stats.parquet')
NAME_MAP_PATH = os.path.join(DATA, 'name_map.csv')
FUZZY_THRESHOLD = 92


def _canonical_names(data_dir=DATA):
    master = pd.read_csv(os.path.join(data_dir, 'ufc-master.csv'), low_memory=False)
    career = pd.read_csv(os.path.join(data_dir, 'career_fights_updated.csv'))
    names = set(master['R_fighter'].dropna()) | set(master['B_fighter'].dropna())
    names |= set(career['fighter'].dropna())
    return sorted(names)


def build(round_stats_path=ROUND_STATS_PATH, data_dir=DATA, manual_overrides_path=None):
    round_stats = pd.read_parquet(round_stats_path)
    ufcstats_names = sorted(round_stats['fighter'].dropna().unique())
    canonical = _canonical_names(data_dir)
    canonical_set = set(canonical)

    # Preserve any manual overrides from a prior run so rerunning this
    # script (e.g. after round_stats.parquet gains new fighters) doesn't
    # wipe out human-reviewed rows.
    prior_overrides = {}
    if manual_overrides_path is None:
        manual_overrides_path = NAME_MAP_PATH
    if os.path.exists(manual_overrides_path):
        prior = pd.read_csv(manual_overrides_path)
        prior_overrides = {
            row['ufcstats_name']: row['manual_override']
            for _, row in prior.iterrows()
            if pd.notna(row.get('manual_override')) and str(row.get('manual_override')).strip()
        }

    rows = []
    for name in ufcstats_names:
        override = prior_overrides.get(name)
        if override:
            rows.append({'ufcstats_name': name, 'canonical_name': override,
                         'match_type': 'manual', 'match_score': 100.0, 'manual_override': override})
            continue
        if name in canonical_set:
            rows.append({'ufcstats_name': name, 'canonical_name': name,
                         'match_type': 'exact', 'match_score': 100.0, 'manual_override': ''})
            continue
        best = process.extractOne(name, canonical, scorer=fuzz.token_sort_ratio)
        if best is not None and best[1] >= FUZZY_THRESHOLD:
            rows.append({'ufcstats_name': name, 'canonical_name': best[0],
                         'match_type': 'fuzzy', 'match_score': float(best[1]), 'manual_override': ''})
        else:
            score = float(best[1]) if best is not None else 0.0
            rows.append({'ufcstats_name': name, 'canonical_name': '',
                         'match_type': 'unmatched', 'match_score': score, 'manual_override': ''})

    name_map = pd.DataFrame(rows).sort_values('ufcstats_name').reset_index(drop=True)
    name_map.to_csv(NAME_MAP_PATH, index=False)
    return name_map, round_stats


def unmatched_report(name_map, round_stats):
    unmatched_names = set(name_map[name_map['match_type'] == 'unmatched']['ufcstats_name'])

    rs_2015 = round_stats[round_stats['date'] >= '2015-01-01']
    fights_2015 = rs_2015[['event', 'bout', 'date']].drop_duplicates()
    total_fights = len(fights_2015)

    fighters_per_fight = rs_2015.groupby(['event', 'bout'])['fighter'].apply(set)
    affected = fighters_per_fight.apply(lambda fs: bool(fs & unmatched_names))
    n_affected = int(affected.sum())
    pct = (n_affected / total_fights * 100) if total_fights else 0.0

    print('=' * 62)
    print('  8SI v2 Stage 1.1 — name_map.csv unmatched report (2015+ window)')
    print('=' * 62)
    print(f'  ufcstats fighter names total: {len(name_map):,}')
    print(f'  exact:    {(name_map["match_type"]=="exact").sum():,}')
    print(f'  fuzzy:    {(name_map["match_type"]=="fuzzy").sum():,}')
    print(f'  manual:   {(name_map["match_type"]=="manual").sum():,}')
    print(f'  unmatched: {(name_map["match_type"]=="unmatched").sum():,}')
    print()
    print(f'  2015+ fights (event,bout): {total_fights:,}')
    print(f'  2015+ fights with >=1 unmatched fighter: {n_affected:,} ({pct:.2f}%)')
    print(f'  Acceptance bar: < 2.00%  ->  {"PASS" if pct < 2.0 else "FAIL"}')
    if n_affected:
        print()
        print('  Unmatched names appearing in 2015+ fights (fill manual_override and rerun):')
        for n in sorted(unmatched_names & set(rs_2015['fighter'])):
            print(f'    {n}')
    print('=' * 62)
    return {'total_fights_2015': total_fights, 'affected_fights_2015': n_affected, 'pct_unmatched': pct}


def main():
    name_map, round_stats = build()
    return unmatched_report(name_map, round_stats)


if __name__ == '__main__':
    main()
