#!/usr/bin/env python3
"""
training/scrape_rounds_update.py — 8SI v2 Stage 1.1, incremental round-data refresh

Deliberately does NOT scrape ufcstats.com directly. Greco1899/scrape_ufc_stats
already runs a rate-limited scraper against ufcstats.com as a daily
scheduled job (its own README: "deployed to GCP as a Cloud Run Job... runs
daily via Cloud Scheduler... push the refreshed data files to this
repository") and publishes the result as plain CSVs. Re-scraping the same
site ourselves would duplicate that traffic for no benefit and is exactly
the kind of hammering the spec says to avoid — so "incremental update"
here means re-downloading that already-fresh CSV snapshot (a handful of
small HTTP GETs to GitHub, not a live scrape) and re-running the ingestion
pipeline on top of it.

Usage: run after each UFC event (or on any cadence) to pick up newly
completed fights into data/round_stats.parquet and data/name_map.csv.
Safe to run repeatedly — every step here is idempotent.
"""
import os
import sys
import time

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from training.ingest_rounds import RAW_DIR, OUT_PATH as ROUND_STATS_PATH, ingest
from training.build_name_map import build as build_name_map, unmatched_report

SOURCE_REPO = 'Greco1899/scrape_ufc_stats'
SOURCE_BRANCH = 'main'
FILES = [
    'ufc_event_details.csv', 'ufc_fight_details.csv', 'ufc_fight_results.csv',
    'ufc_fight_stats.csv', 'ufc_fighter_details.csv', 'ufc_fighter_tott.csv',
]
# Polite pacing between requests even though these are small static files
# on GitHub's CDN, not the live scraper's own traffic against ufcstats.com.
REQUEST_DELAY_SEC = 2.0
USER_AGENT = '8si-ufc-predictor/1.0 (data refresh; see training/scrape_rounds_update.py)'


def download_source_csvs(raw_dir=RAW_DIR, files=FILES, delay=REQUEST_DELAY_SEC):
    os.makedirs(raw_dir, exist_ok=True)
    headers = {'User-Agent': USER_AGENT}
    downloaded = {}
    for i, fname in enumerate(files):
        url = f'https://raw.githubusercontent.com/{SOURCE_REPO}/{SOURCE_BRANCH}/{fname}'
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        path = os.path.join(raw_dir, fname)
        with open(path, 'wb') as f:
            f.write(resp.content)
        downloaded[fname] = len(resp.content)
        print(f'  {fname}: {len(resp.content):,} bytes')
        if i < len(files) - 1:
            time.sleep(delay)
    return downloaded


def main():
    print('=' * 62)
    print('  8SI v2 Stage 1.1 — incremental round-data refresh')
    print('=' * 62)

    prior_fights = None
    if os.path.exists(ROUND_STATS_PATH):
        prior = pd.read_parquet(ROUND_STATS_PATH)
        prior_fights = set(map(tuple, prior[['event', 'bout']].drop_duplicates().to_numpy()))

    print('\n[1/3] Downloading latest source CSVs...')
    download_source_csvs()

    print('\n[2/3] Re-running ingest_rounds.py...')
    new = ingest()
    new_fights = set(map(tuple, new[['event', 'bout']].drop_duplicates().to_numpy()))
    if prior_fights is not None:
        added = new_fights - prior_fights
        removed = prior_fights - new_fights
        print(f'  Fights added since last refresh: {len(added):,}')
        if removed:
            print(f'  WARNING: {len(removed):,} previously-seen fights no longer present '
                  f'(source data correction upstream, or a stale local copy) — review before trusting downstream.')
    else:
        print(f'  First run — {len(new_fights):,} fights ingested.')

    print('\n[3/3] Re-running build_name_map.py (existing manual overrides preserved)...')
    name_map, round_stats = build_name_map()
    unmatched_report(name_map, round_stats)


if __name__ == '__main__':
    main()
