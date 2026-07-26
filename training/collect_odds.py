#!/usr/bin/env python3
"""
training/collect_odds.py — 8SI v2 Stage 1.2, forward odds snapshot collector

Standalone script meant to run on a schedule (cron, or manually per card)
and append one timestamped snapshot of current DraftKings MMA lines to
data/odds_snapshots.json — the SAME file/schema training/backfill_clv.py
already reads and backend/main.py's live /odds endpoint already writes
(list of {timestamp, fights: [{f1, f2, f1_price, f2_price}]}, decimal
odds). Deliberately kept on this existing schema rather than the spec's
literal "data/odds_snapshots/" directory suggestion — introducing a second,
differently-shaped store would fork the format two established consumers
already depend on for no real benefit.

Differs from backend/main.py's /odds endpoint in one deliberate way:
that endpoint only recognizes fights on a hardcoded `_CARD_FIGHTS` list
(one specific card, manually updated), which doesn't generalize to "run
this on a schedule forever." This script snapshots every MMA fight the
API returns, using the odds provider's own fighter-name spelling — the
same as-of reconciliation problem round_stats.parquet already has, and
the same tool (training/build_name_map.py) can resolve it later; this
collector's job is just to not lose a snapshot, not to pre-match names.

Running this repeatedly IS the open/close mechanism: the earliest
snapshot for a given fight is the open proxy (training/backfill_clv.py's
_earliest_snapshot_prices() already reads it this way), and whichever
snapshot lands closest to the event is the close proxy — no special
"final" mode needed, just run it more often as fight night approaches.

Requires ODDS_API_KEY as an environment variable — never hardcode it here
(see docs/DECISIONS.md for why backend/main.py's own hardcoded copy,
pre-dating this script, is being left alone rather than force-scrubbed
from git history).
"""
import json
import os
import sys
from datetime import datetime

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, 'data')
SNAPSHOT_PATH = os.path.join(DATA, 'odds_snapshots.json')
ODDS_API_URL = 'https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds/'


def fetch_current_odds(api_key, timeout=15):
    resp = requests.get(
        ODDS_API_URL,
        params={'apiKey': api_key, 'regions': 'us', 'markets': 'h2h', 'bookmakers': 'draftkings'},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def to_snapshot_fights(api_events):
    """API event -> this project's {f1, f2, f1_price, f2_price} shape.
    Uses the odds provider's own fighter-name spelling as-is — reconciled
    against canonical names later via training/build_name_map.py, same as
    round_stats.parquet."""
    fights = []
    for event in api_events:
        dk = next((b for b in event.get('bookmakers', []) if b['key'] == 'draftkings'), None)
        if not dk:
            continue
        h2h = next((m for m in dk.get('markets', []) if m['key'] == 'h2h'), None)
        if not h2h or len(h2h.get('outcomes', [])) < 2:
            continue
        outcomes = h2h['outcomes']
        fights.append({
            'f1': outcomes[0]['name'], 'f2': outcomes[1]['name'],
            'f1_price': outcomes[0]['price'], 'f2_price': outcomes[1]['price'],
            'commence_time': event.get('commence_time'),
        })
    return fights


def append_snapshot(fights, snapshot_path=SNAPSHOT_PATH, timestamp=None):
    timestamp = timestamp or datetime.now().isoformat(timespec='seconds')
    if os.path.exists(snapshot_path):
        with open(snapshot_path) as fp:
            snapshots = json.load(fp)
    else:
        snapshots = []
    snapshots.append({'timestamp': timestamp, 'fights': fights})
    with open(snapshot_path, 'w') as fp:
        json.dump(snapshots, fp)
    return snapshots


def main():
    api_key = os.environ.get('ODDS_API_KEY')
    if not api_key:
        print('ERROR: ODDS_API_KEY environment variable is not set.')
        print('  export ODDS_API_KEY=<your key>  (never commit it — see this file\'s docstring)')
        sys.exit(1)

    print('=' * 62)
    print('  8SI v2 Stage 1.2 — forward odds snapshot')
    print('=' * 62)
    api_events = fetch_current_odds(api_key)
    fights = to_snapshot_fights(api_events)
    if not fights:
        print('  No MMA fights with DraftKings h2h odds returned right now — nothing to snapshot.')
        return
    snapshots = append_snapshot(fights)
    print(f'  Snapshotted {len(fights)} fight(s) at {snapshots[-1]["timestamp"]}')
    for f in fights:
        print(f'    {f["f1"]} ({f["f1_price"]})  vs  {f["f2"]} ({f["f2_price"]})')
    print(f'  Total snapshots in {SNAPSHOT_PATH}: {len(snapshots)}')
    print('=' * 62)


if __name__ == '__main__':
    main()
