"""
sherdog_fix.py — Fix wrong Sherdog profiles using UFC record verification.

Step 1: Detect bad matches (sherdog wins < UFC wins, or dramatic win-rate divergence)
Step 2: Re-scrape only flagged fighters, pick the Sherdog profile closest to UFC win count
Step 3: Rebuild career_fights.csv with rolling last5_won / last5_finish_rate
Step 4: Verify key fighters
"""

import os
import re
import time
import pickle
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
SHERDOG_BASE    = "https://www.sherdog.com"
HEADERS         = {
    'User-Agent': (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    ),
    'Accept':          'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Referer':         'https://www.sherdog.com',
}
SLEEP           = 1.5   # seconds between every request
MAX_PAGES       = 25    # max search-result pages per fighter
MAX_CANDIDATES  = 5     # max profile candidates to verify per fighter
WIN_TOLERANCE   = 4     # ±4 wins from UFC record is acceptable

ORIG_PKL   = "data/sherdog_records.pkl"
FIXED_PKL  = "data/sherdog_records_fixed.pkl"
BAD_CSV    = "data/bad_sherdog_matches.csv"
UNFIXED_CSV= "data/still_unfixed.csv"
OUT_CAREER  = "data/career_fights.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Load source data
# ─────────────────────────────────────────────────────────────────────────────
print("Loading source data...")
with open(ORIG_PKL, 'rb') as f:
    sherdog_orig = pickle.load(f)

ufc_df = pd.read_csv("data/ufc_fighters_final.csv")
ufc_df['Wins']   = pd.to_numeric(ufc_df['Wins'],   errors='coerce').fillna(0).astype(int)
ufc_df['Losses'] = pd.to_numeric(ufc_df['Losses'], errors='coerce').fillna(0).astype(int)
ufc_lookup = {row['Fighter_Name']: row for _, row in ufc_df.iterrows()}

print(f"  sherdog_records: {len(sherdog_orig)} fighters")
print(f"  ufc_fighters_final: {len(ufc_df)} fighters")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Find bad matches
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1: Detecting bad Sherdog matches")
print("=" * 60)

def sherdog_summary(rec):
    fights  = rec.get('fights', [])
    wins    = sum(1 for f in fights if f.get('result') == 'win')
    losses  = sum(1 for f in fights if f.get('result') == 'loss')
    return wins, losses, len(fights)

bad_fighters = []
for name, rec in sherdog_orig.items():
    if name not in ufc_lookup:
        continue

    urow       = ufc_lookup[name]
    ufc_wins   = int(urow['Wins'])
    ufc_losses = int(urow['Losses'])

    sh_wins, sh_losses, sh_total = sherdog_summary(rec)

    reasons = []

    # Rule 1: sherdog wins < UFC wins  (Sherdog must have >= UFC wins)
    if sh_wins < ufc_wins:
        reasons.append(f"sh_wins({sh_wins})<ufc_wins({ufc_wins})")

    # Rule 2: too few total fights for a fighter with 5+ UFC wins
    if sh_total < 3 and ufc_wins >= 5:
        reasons.append(f"sh_total({sh_total})<3 with ufc_wins={ufc_wins}")

    # Rule 3: win-rate diverges by >30%  (need at least 5 fights each side)
    if sh_total >= 5 and (ufc_wins + ufc_losses) >= 5:
        sh_wr  = sh_wins  / sh_total
        ufc_wr = ufc_wins / (ufc_wins + ufc_losses)
        if abs(sh_wr - ufc_wr) > 0.30:
            reasons.append(
                f"wr_dif({sh_wr:.2f} vs {ufc_wr:.2f})"
            )

    if reasons:
        bad_fighters.append({
            'name':           name,
            'ufc_wins':       ufc_wins,
            'ufc_losses':     ufc_losses,
            'sherdog_wins':   sh_wins,
            'sherdog_losses': sh_losses,
            'sherdog_total':  sh_total,
            'reasons':        '; '.join(reasons),
        })

bad_df = pd.DataFrame(bad_fighters)
bad_df.to_csv(BAD_CSV, index=False)
print(f"Flagged {len(bad_fighters)} fighters — saved to {BAD_CSV}\n")
if len(bad_df):
    print(bad_df[['name','ufc_wins','ufc_losses',
                  'sherdog_wins','sherdog_losses','reasons']].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Re-scrape flagged fighters
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Re-scraping flagged fighters")
print("=" * 60)

session = requests.Session()
session.headers.update(HEADERS)
_consecutive_errors = 0


def safe_get(url: str):
    global _consecutive_errors
    time.sleep(SLEEP)
    try:
        r = session.get(url, timeout=20)
        r.raise_for_status()
        _consecutive_errors = 0
        return r
    except Exception as e:
        _consecutive_errors += 1
        print(f"    HTTP error ({_consecutive_errors}): {e}")
        if _consecutive_errors >= 3:
            print("    3 consecutive errors — pausing 60 s...")
            time.sleep(60)
            _consecutive_errors = 0
        return None


def search_candidates(name: str) -> list[tuple[str, str]]:
    """
    Paginate Sherdog search results and collect exact-name-match profile URLs.
    Returns list of (display_name, full_profile_url).
    Stops early when alphabetically past the target name or when MAX_CANDIDATES found.
    """
    target_lower   = name.strip().lower()
    target_sortkey = target_lower  # for alphabetical comparison
    candidates     = []

    for page in range(1, MAX_PAGES + 1):
        url = (
            f"{SHERDOG_BASE}/stats/fightfinder"
            f"?association=&weightclass="
            f"&SearchTxt={quote_plus(name)}&page={page}"
        )
        r = safe_get(url)
        if r is None:
            break

        soup   = BeautifulSoup(r.text, 'html.parser')
        tables = soup.find_all('table')
        if len(tables) < 2:
            break

        rows      = tables[1].find_all('tr')[1:]  # skip header
        if not rows:
            break

        page_names = []
        for row in rows:
            a = row.find('a', href=lambda h: h and '/fighter/' in h)
            if not a:
                continue
            cname = a.get_text(strip=True)
            href  = a['href']
            if not href.startswith('http'):
                href = SHERDOG_BASE + href
            page_names.append(cname.lower())

            if cname.strip().lower() == target_lower:
                candidates.append((cname.strip(), href))
                if len(candidates) >= MAX_CANDIDATES:
                    return candidates

        # Early stop: if all names on this page come after target alphabetically
        if page_names and all(pn > target_sortkey for pn in page_names):
            break

        # Stop if no "Next" link
        has_next = any(
            f'page={page + 1}' in (a.get('href', ''))
            for a in soup.find_all('a', href=True)
        )
        if not has_next:
            break

    return candidates


def parse_fighter_profile(url: str) -> dict | None:
    """
    Fetch Sherdog fighter profile. Return dict with keys:
      'fights': list of fight dicts, 'wins': int, 'losses': int
    Returns None on failure.
    """
    r = safe_get(url)
    if r is None:
        return None

    soup   = BeautifulSoup(r.text, 'html.parser')
    tables = soup.find_all('table')

    # Identify fight history table by its header row
    fight_table = None
    for t in tables:
        header_row = t.find('tr')
        if header_row is None:
            continue
        header_text = header_row.get_text(separator='|')
        if 'Result' in header_text and 'Fighter' in header_text:
            fight_table = t
            break

    if fight_table is None:
        return None

    fights = []
    for row in fight_table.find_all('tr')[1:]:  # skip header
        cells = row.find_all('td')
        if len(cells) < 4:
            continue

        result_raw = cells[0].get_text(strip=True).lower()
        if result_raw == 'win':
            result = 'win'
        elif result_raw == 'loss':
            result = 'loss'
        elif result_raw in ('draw', 'nc', 'no contest'):
            result = result_raw
        else:
            continue  # skip non-standard rows

        opponent   = cells[1].get_text(strip=True)

        # Cell 2: "Event Name  Mon / DD / YYYY"
        cell2      = cells[2].get_text(separator=' ', strip=True)
        date_match = re.search(r'(\b\w{3,9} / \d{1,2} / \d{4}\b)', cell2)
        date_obj   = None
        event_name = cell2
        if date_match:
            try:
                date_obj   = pd.Timestamp(
                    datetime.strptime(date_match.group(1), "%b / %d / %Y")
                )
                event_name = cell2.replace(date_match.group(1), '').strip()
            except Exception:
                pass

        # Cell 3: "Method Referee VIEW PLAY-BY-PLAY"
        cell3  = cells[3].get_text(separator=' ', strip=True)
        method = cell3.split('VIEW PLAY-BY-PLAY')[0].strip()

        fights.append({
            'result':   result,
            'opponent': opponent,
            'date':     date_obj,
            'method':   method,
            'event':    event_name,
        })

    wins   = sum(1 for f in fights if f['result'] == 'win')
    losses = sum(1 for f in fights if f['result'] == 'loss')
    return {'fights': fights, 'wins': wins, 'losses': losses}


# Load existing progress if resuming
if os.path.exists(FIXED_PKL):
    with open(FIXED_PKL, 'rb') as f:
        sherdog_fixed = pickle.load(f)
    print(f"Resuming: {len(sherdog_fixed)} fighters already in {FIXED_PKL}")
else:
    sherdog_fixed = {}

still_unfixed = []
n_bad = len(bad_fighters)

for idx, row in enumerate(bad_fighters):
    name     = row['name']
    ufc_wins = row['ufc_wins']

    if name in sherdog_fixed:
        print(f"[{idx+1}/{n_bad}] SKIP (already fixed): {name}")
        continue

    print(f"\n[{idx+1}/{n_bad}] {name}  (UFC wins={ufc_wins})")

    candidates = search_candidates(name)
    print(f"  Found {len(candidates)} exact-name candidates")

    best_profile = None
    best_diff    = float('inf')

    for cname, profile_url in candidates:
        profile = parse_fighter_profile(profile_url)
        if profile is None:
            print(f"    {profile_url} → fetch failed")
            continue

        diff = abs(profile['wins'] - ufc_wins)
        print(
            f"    {profile_url.split('/')[-1]}: "
            f"sherdog {profile['wins']}-{profile['losses']}, "
            f"diff={diff}"
        )

        if diff <= WIN_TOLERANCE and diff < best_diff:
            best_diff    = diff
            best_profile = profile

    if best_profile is not None:
        sherdog_fixed[name] = {'fights': best_profile['fights']}
        print(f"  ✓ Fixed {name}: {best_profile['wins']}-{best_profile['losses']}")
    else:
        still_unfixed.append(name)
        print(f"  ✗ No valid candidate for {name}")

    if (idx + 1) % 20 == 0:
        with open(FIXED_PKL, 'wb') as f:
            pickle.dump(sherdog_fixed, f)
        print(f"  [Saved progress: {len(sherdog_fixed)} fixed so far]")

# Final save of fixed records
with open(FIXED_PKL, 'wb') as f:
    pickle.dump(sherdog_fixed, f)

pd.DataFrame({'fighter': still_unfixed}).to_csv(UNFIXED_CSV, index=False)
print(f"\nStep 2 complete:")
print(f"  Fixed:          {len(sherdog_fixed)}")
print(f"  Still unfixed:  {len(still_unfixed)} → {UNFIXED_CSV}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Rebuild career_fights.csv
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Rebuilding career_fights.csv")
print("=" * 60)

# Merge: fixed records override originals, rest unchanged
merged_records = dict(sherdog_orig)   # start with all originals
merged_records.update(sherdog_fixed)  # override bad ones with fixed ones
print(f"  Total fighters in merged pkl: {len(merged_records)}")

rows_out = []
for fighter, rec in merged_records.items():
    for fight in rec.get('fights', []):
        result   = fight.get('result', '')
        method   = fight.get('method', '') or ''
        won      = 1 if result == 'win' else 0
        got_finish = int(
            won == 1 and
            bool(re.search(r'KO|TKO|Submission', method, re.IGNORECASE))
        )
        rows_out.append({
            'fighter':  fighter,
            'opponent': fight.get('opponent', ''),
            'date':     fight.get('date'),
            'result':   result,
            'method':   method,
            'won':      won,
            'got_finish': got_finish,
        })

career = pd.DataFrame(rows_out)
career['date'] = pd.to_datetime(career['date'], errors='coerce')

# Sort fighter → date ascending
career = career.sort_values(['fighter', 'date']).reset_index(drop=True)

# Rolling last5_won and last5_finish_rate with shift(1) to avoid leakage
def roll5_shift(x):
    return x.shift(1).rolling(5, min_periods=1).mean()

g = career.groupby('fighter', sort=False)
career['last5_won']          = g['won'].transform(roll5_shift)
career['last5_finish_rate']  = g['got_finish'].transform(roll5_shift)

career = career[[
    'fighter', 'opponent', 'date', 'result', 'method',
    'won', 'got_finish', 'last5_won', 'last5_finish_rate'
]]

career.to_csv(OUT_CAREER, index=False)
print(f"  Saved {len(career):,} rows to {OUT_CAREER}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Verify key fighters
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Verification")
print("=" * 60)

VERIFY = [
    ("Jon Jones",       "expected ~28-1"),
    ("Youssef Zalal",   "expected ~19-4"),
    ("Conor McGregor",  "expected ~22-6"),
    ("Islam Makhachev", "expected ~27-1"),
    ("Alex Pereira",    "expected ~12-2 MMA"),
]

for name, note in VERIFY:
    ff = career[career['fighter'] == name].sort_values('date')
    if len(ff) == 0:
        print(f"  {name}: NOT FOUND in career_fights")
        continue

    wins   = int((ff['won'] == 1).sum())
    losses = int((ff['won'] == 0).sum())
    total  = len(ff)
    last_date = ff['date'].max()
    print(f"  {name}: {wins}-{losses} ({total} total fights, last: {last_date.date() if pd.notna(last_date) else 'unknown'})  [{note}]")
    if len(ff) > 0:
        recent = ff.tail(5)[['date','opponent','result','method']].copy()
        recent['date'] = recent['date'].dt.date
        print(recent.to_string(index=False))

print("\nAll done.")
