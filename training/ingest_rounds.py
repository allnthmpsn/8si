#!/usr/bin/env python3
"""
training/ingest_rounds.py — 8SI v2 Stage 1.1

Parses the Greco1899/scrape_ufc_stats pre-scraped CSVs (downloaded, not
scraped — see training/scrape_rounds_update.py for how they're refreshed)
into data/round_stats.parquet: one row per (fight, fighter, round) with
knockdowns, significant-strike landed/attempted by target (head/body/leg)
and position (distance/clinch/ground), total strikes, takedowns, sub
attempts, reversals, and control time in seconds.

Source files (data/raw/ufcstats_rounds/, from
https://github.com/Greco1899/scrape_ufc_stats — its own README documents a
daily automated refresh, so re-running training/scrape_rounds_update.py
picks up new events without ever scraping ufcstats.com ourselves):
  ufc_fight_stats.csv    — one row per (fight, round, fighter), the source
                            for every stat column below
  ufc_event_details.csv  — EVENT -> DATE, needed since fight_stats has no
                            date of its own

Fighter names here are ufcstats.com's OWN spelling/formatting, NOT yet
reconciled against this project's canonical (ufc-master.csv) names — that
is data/name_map.csv's job (training/build_name_map.py), a deliberately
separate concern. Downstream (Stage 2) feature code joins round_stats.parquet
through the name map, not directly on `fighter`.
"""
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RAW_DIR = os.path.join(ROOT, 'data', 'raw', 'ufcstats_rounds')
OUT_PATH = os.path.join(ROOT, 'data', 'round_stats.parquet')

# (output prefix, source column)
LANDED_ATT_COLS = [
    ('sig_str',  'SIG.STR.'),
    ('total_str', 'TOTAL STR.'),
    ('td',       'TD'),
    ('head',     'HEAD'),
    ('body',     'BODY'),
    ('leg',      'LEG'),
    ('distance', 'DISTANCE'),
    ('clinch',   'CLINCH'),
    ('ground',   'GROUND'),
]

_OF_RE = re.compile(r'^(\d+) of (\d+)$')


def _split_of(series):
    """'29 of 62' -> (29, 62); non-matching/NaN -> (NaN, NaN)."""
    extracted = series.astype(str).str.extract(_OF_RE)
    landed = pd.to_numeric(extracted[0], errors='coerce')
    att = pd.to_numeric(extracted[1], errors='coerce')
    return landed, att


def _ctrl_to_seconds(series):
    """'2:19' -> 139; '--' (only ever seen pre-2000, outside this
    project's 2015+ window) or NaN -> NaN, not 0 — a fighter's control
    time not being TRACKED is not the same as it being zero."""
    extracted = series.astype(str).str.extract(r'^(\d+):(\d+)$')
    minutes = pd.to_numeric(extracted[0], errors='coerce')
    seconds = pd.to_numeric(extracted[1], errors='coerce')
    return minutes * 60 + seconds


def ingest(raw_dir=RAW_DIR, out_path=OUT_PATH):
    fs = pd.read_csv(os.path.join(raw_dir, 'ufc_fight_stats.csv'))
    ev = pd.read_csv(os.path.join(raw_dir, 'ufc_event_details.csv'))

    # 42 rows in the source are entirely empty placeholders (pre-1999
    # fights with no stats ever recorded) — EVENT/BOUT present, everything
    # else NaN. Drop before parsing rather than let them become all-NaN
    # rows downstream.
    fs = fs.dropna(subset=['ROUND', 'FIGHTER']).copy()

    ev = ev.copy()
    ev['date'] = pd.to_datetime(ev['DATE'], format='%B %d, %Y')
    fs = fs.merge(ev[['EVENT', 'date']], on='EVENT', how='left')
    missing = fs[fs['date'].isna()]
    if len(missing):
        # The source's two CSVs are refreshed by the same daily job but not
        # atomically — a just-completed event can briefly have fight_stats
        # rows before its event_details row lands. Drop and warn rather
        # than crash; a re-run after the next refresh picks it up.
        missing_events = sorted(missing['EVENT'].unique())
        print(f'  WARNING: dropping {len(missing)} row(s) with no matching event date '
              f'(likely a source sync lag): {missing_events}')
        fs = fs[fs['date'].notna()].copy()

    fs['round'] = fs['ROUND'].str.extract(r'^Round (\d+)$')[0].astype(int)

    out = pd.DataFrame({
        'event':   fs['EVENT'],
        'bout':    fs['BOUT'],
        'date':    fs['date'],
        'fighter': fs['FIGHTER'],
        'round':   fs['round'],
        'kd':      fs['KD'].astype(float),
        'sub_att': fs['SUB.ATT'].astype(float),
        'rev':     fs['REV.'].astype(float),
        'ctrl_sec': _ctrl_to_seconds(fs['CTRL']),
    })

    for prefix, src_col in LANDED_ATT_COLS:
        landed, att = _split_of(fs[src_col])
        out[f'{prefix}_landed'] = landed
        out[f'{prefix}_att'] = att

    out = out.sort_values(['fighter', 'date', 'round']).reset_index(drop=True)
    out.to_parquet(out_path, index=False)
    return out


def main():
    print('=' * 62)
    print('  8SI v2 Stage 1.1 — round_stats.parquet ingestion')
    print('=' * 62)
    out = ingest()
    print(f'  Rows: {len(out):,}')
    print(f'  Unique (event, bout): {out[["event","bout"]].drop_duplicates().shape[0]:,}')
    print(f'  Unique fighters (ufcstats spelling): {out["fighter"].nunique():,}')
    print(f'  Date range: {out["date"].min().date()} to {out["date"].max().date()}')
    n_2015 = out[out['date'] >= '2015-01-01'][['event', 'bout']].drop_duplicates().shape[0]
    print(f'  Fights (event,bout) 2015+: {n_2015:,}')
    print(f'  Written to: {OUT_PATH}')
    print('=' * 62)


if __name__ == '__main__':
    main()
