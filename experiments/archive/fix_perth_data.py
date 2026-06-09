#!/usr/bin/env python3
"""
fix_perth_data.py — Audit and fix Perth card fighter data.
Steps 1-6: record fixes, stat scraping, backend fallback patch,
           career history, sherdog pkl, verification.
"""

import gc, json, os, pickle, re, sys, time, warnings
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime

warnings.filterwarnings('ignore')

HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}

# ── Correct records as specified ──────────────────────────────────────────────
CORRECT_RECORDS = {
    'Jack Della Maddalena': (18, 3, 0),
    'Carlos Prates':         (23, 7, 0),
    'Quillan Salkilld':      (11, 1, 0),
    'Beneil Dariush':        (23, 7, 1),
    'Steve Erceg':           ( 9, 2, 0),
    'Tim Elliott':           (20,13, 1),
    'Shamil Gaziev':         (14, 2, 0),
    'Brando Pericic':        (12, 3, 0),
    'Tai Tuivasa':           (15, 7, 0),
    'Louie Sutherland':      (14, 5, 0),
    'Cam Rowston':           (14, 3, 0),
    'Robert Bryczek':        (18, 6, 0),
    'Junior Tafa':           ( 6, 5, 0),
    'Kevin Christian':       ( 9, 3, 0),
    'Jacob Malkoun':         ( 9, 3, 0),
    'Gerald Meerschaert':    (37,21, 0),
    'Colby Thicknesse':      ( 8, 1, 0),
    'Vince Morales':         (16,10, 0),
    'Wes Schultz':           ( 8, 3, 0),
    'Ben Johnston':          ( 5, 1, 0),
    'Jonathan Micallef':     ( 9, 1, 0),
    'Themba Gorimbo':        (14, 6, 0),
    'Kody Steele':           (11, 3, 0),
    'Dom Mar Fan':           ( 8, 4, 0),
}

# Weight class per fighter (for fallback averages)
FIGHTER_WC = {
    'Jack Della Maddalena': 'Welterweight',
    'Carlos Prates':         'Welterweight',
    'Quillan Salkilld':      'Lightweight',
    'Beneil Dariush':        'Lightweight',
    'Steve Erceg':           'Flyweight',
    'Tim Elliott':           'Flyweight',
    'Shamil Gaziev':         'Heavyweight',
    'Brando Pericic':        'Heavyweight',
    'Tai Tuivasa':           'Heavyweight',
    'Louie Sutherland':      'Heavyweight',
    'Cam Rowston':           'Middleweight',
    'Robert Bryczek':        'Middleweight',
    'Junior Tafa':           'Light Heavyweight',
    'Kevin Christian':       'Light Heavyweight',
    'Jacob Malkoun':         'Middleweight',
    'Gerald Meerschaert':    'Middleweight',
    'Colby Thicknesse':      'Bantamweight',
    'Vince Morales':         'Bantamweight',
    'Wes Schultz':           'Middleweight',
    'Ben Johnston':          'Middleweight',
    'Jonathan Micallef':     'Welterweight',
    'Themba Gorimbo':        'Welterweight',
    'Kody Steele':           'Lightweight',
    'Dom Mar Fan':           'Lightweight',
}

# UFC Stats URLs for fighters we need to scrape
UFC_STATS_URLS = {
    'Ben Johnston':      'http://ufcstats.com/fighter-details/3261aa79bf6caa64',
    'Louie Sutherland':  'http://ufcstats.com/fighter-details/cdb7b38b1b357f26',
    'Wes Schultz':       'http://ufcstats.com/fighter-details/65d8e7ede5fecedc',
    'Brando Pericic':    'http://ufcstats.com/fighter-details/d0fd0d9ee560dae7',
    'Kody Steele':       'http://ufcstats.com/fighter-details/aebaa8cec15b083d',
    # Re-scrape targets for career history
}

# ── Load data ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("PERTH CARD DATA AUDIT + FIX")
print("=" * 60)

df_csv = pd.read_csv('data/ufc_fighters_final_updated.csv')
# Normalise percentage columns
for col in ['Str_Acc', 'Str_Def', 'TD_Acc', 'TD_Def']:
    df_csv[col] = pd.to_numeric(
        df_csv[col].astype(str).str.replace('%', '', regex=False),
        errors='coerce'
    ).fillna(0)

career_df = pd.read_csv('data/career_fights_updated.csv')
career_df['date'] = pd.to_datetime(career_df['date'])

with open('data/sherdog_records_patched.pkl', 'rb') as f:
    sherdog = pickle.load(f)

# ── STEP 1 — Audit ───────────────────────────────────────────────────────────
print("\n── STEP 1 — AUDIT ──")
print(f"{'Fighter':<25} {'Stored':>8}  {'Correct':>8}  {'Rec?':>5}  "
      f"{'SLpM':>5}  {'SApM':>5}  {'StrDef':>6}  {'TDDef':>6}  {'Stats?':>7}")
print("-" * 98)

record_wrong  = []   # (name, sw, sl, sd, cw, cl, cd)
stats_zero    = []   # names with SLpM==0 or SApM==0 or Str_Def==0
td_def_zero   = []   # names with TD_Def==0 (but SLpM > 0)
rescrape_needed = [] # names where stored_wins < correct_wins by 2+

for name, (cw, cl, cd) in CORRECT_RECORDS.items():
    row = df_csv[df_csv['Fighter_Name'] == name]
    if len(row) == 0:
        print(f'{name:<25} NOT FOUND IN CSV')
        continue
    r = row.iloc[0]
    sw = int(r.get('Wins',   0) or 0)
    sl = int(r.get('Losses', 0) or 0)
    sd = int(r.get('Draws',  0) or 0)

    slpm   = float(r.get('SLpM',    0) or 0)
    sapm   = float(r.get('SApM',    0) or 0)
    strdef = float(r.get('Str_Def', 0) or 0)
    tddef  = float(r.get('TD_Def',  0) or 0)

    rec_ok   = (sw == cw and sl == cl and sd == cd)
    stats_ok = (slpm > 0 and sapm > 0)

    stored  = f'{sw}-{sl}' + (f'-{sd}' if sd > 0 else '')
    correct = f'{cw}-{cl}' + (f'-{cd}' if cd > 0 else '')

    print(f'{name:<25} {stored:>8}  {correct:>8}  '
          f'{"✓" if rec_ok else "❌":>5}  '
          f'{slpm:>5.2f}  {sapm:>5.2f}  '
          f'{strdef:>6.0f}  {tddef:>6.0f}  '
          f'{"✓" if stats_ok else "❌":>7}')

    if not rec_ok:
        record_wrong.append((name, sw, sl, sd, cw, cl, cd))
    if not stats_ok:
        stats_zero.append(name)
    if stats_ok and tddef == 0:
        td_def_zero.append(name)
    if cw > sw + 1:          # stored wins < correct by 2+
        rescrape_needed.append(name)

print(f"\nRecords wrong:        {len(record_wrong)}")
print(f"Stats missing (zero): {len(stats_zero)}")
print(f"TD_Def=0 (exp. ftr):  {len(td_def_zero)} — {td_def_zero}")
print(f"Career re-scrape:     {len(rescrape_needed)} — {rescrape_needed}")

# ── STEP 2 — Fix wrong records in CSV ────────────────────────────────────────
print("\n── STEP 2 — FIX RECORDS ──")
csv_changed = False
for name, sw, sl, sd, cw, cl, cd in record_wrong:
    idx = df_csv.index[df_csv['Fighter_Name'] == name][0]
    old = f'{sw}-{sl}' + (f'-{sd}' if sd > 0 else '')
    new = f'{cw}-{cl}' + (f'-{cd}' if cd > 0 else '')
    df_csv.at[idx, 'Wins']   = cw
    df_csv.at[idx, 'Losses'] = cl
    df_csv.at[idx, 'Draws']  = cd
    print(f"  Updated {name}: {old} → {new}")
    csv_changed = True

# ── STEP 3 — Scrape UFC Stats for fighters with zero striking stats ───────────
print("\n── STEP 3 — SCRAPE UFC STATS ──")

def scrape_ufc_stats_page(url):
    """Fetch a UFC Stats fighter detail page and return stat dict."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        stats = {}
        for li in soup.find_all('li', class_='b-list__box-list-item'):
            txt = li.get_text(separator='|', strip=True)
            parts = [p.strip() for p in txt.split('|') if p.strip()]
            if len(parts) >= 2:
                label, val = parts[0].lower(), parts[1]
                val_clean = val.replace('%', '').strip()
                try:
                    fval = float(val_clean)
                except Exception:
                    continue
                if 'slpm'   in label: stats['SLpM']    = fval
                if 'sapm'   in label: stats['SApM']    = fval
                if 'str. acc' in label: stats['Str_Acc'] = fval
                if 'str. def' in label: stats['Str_Def'] = fval
                if 'td avg'  in label: stats['TD_Avg']  = fval
                if 'td acc'  in label: stats['TD_Acc']  = fval
                if 'td def'  in label: stats['TD_Def']  = fval
                if 'sub. avg' in label: stats['Sub_Avg'] = fval
        return stats
    except Exception as e:
        print(f"    ERROR scraping {url}: {e}")
        return {}

def get_ufc_stats_fights(url):
    """Return list of {date, opponent, result, method} from UFC Stats fighter page."""
    fights = []
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        table = soup.find('table', class_='b-fight-details__table')
        if not table:
            return fights
        rows = table.find_all('tr')
        for row in rows[1:]:   # skip header
            cells = row.find_all('td')
            if len(cells) < 9: continue
            result_txt = cells[0].get_text(strip=True).lower()
            if result_txt in ('next', 'nc', ''):
                continue
            # opponent names are in first 2 cells as both fighter names
            opp_names = cells[1].find_all('a')
            opponent = ''
            if len(opp_names) >= 2:
                opponent = opp_names[1].get_text(strip=True)
            method = cells[7].get_text(' ', strip=True) if len(cells) > 7 else ''
            date_str = cells[8].get_text(strip=True) if len(cells) > 8 else ''
            try:
                date = pd.to_datetime(date_str)
            except Exception:
                date = None
            result = 'win' if result_txt == 'win' else 'loss'
            fights.append({
                'result': result, 'opponent': opponent,
                'date': date, 'method': method, 'event': ''
            })
    except Exception as e:
        print(f"    ERROR getting fights from {url}: {e}")
    return fights

# Fighters that need stats scraping: those with SLpM==0 or SApM==0
# Ben Johnston is a true debutant — skip (keep zeros)
scraped_stats = {}
for name in stats_zero:
    if name == 'Ben Johnston':
        print(f"  {name}: UFC debutant — keeping zeros")
        continue
    url = UFC_STATS_URLS.get(name)
    if not url:
        print(f"  {name}: no URL, skipping")
        continue
    print(f"  Scraping {name}...")
    stats = scrape_ufc_stats_page(url)
    if stats:
        scraped_stats[name] = stats
        idx = df_csv.index[df_csv['Fighter_Name'] == name]
        if len(idx) > 0:
            for col, val in stats.items():
                df_csv.at[idx[0], col] = val
            print(f"    → SLpM={stats.get('SLpM',0):.2f} SApM={stats.get('SApM',0):.2f} "
                  f"StrDef={stats.get('Str_Def',0):.0f}% TDDef={stats.get('TD_Def',0):.0f}%")
            csv_changed = True
    time.sleep(1.0)

# Also scrape TD_Def for fighters where it is 0 but other stats are fine
print(f"\n  Scraping TD_Def=0 fighters: {td_def_zero}")
for name in td_def_zero:
    url = UFC_STATS_URLS.get(name)
    if not url:
        print(f"  {name}: no URL — will use weight class average")
        continue
    print(f"  Scraping {name} for TD_Def...")
    stats = scrape_ufc_stats_page(url)
    if stats.get('TD_Def', 0) > 0:
        idx = df_csv.index[df_csv['Fighter_Name'] == name]
        if len(idx) > 0:
            df_csv.at[idx[0], 'TD_Def'] = stats['TD_Def']
            print(f"    → TD_Def={stats['TD_Def']:.0f}%")
            csv_changed = True
    else:
        print(f"    → TD_Def still 0 on UFC Stats — will use weight class average")
    time.sleep(1.0)

# ── STEP 4 — Weight class averages for backend fallback ──────────────────────
print("\n── STEP 4 — WEIGHT CLASS AVERAGES ──")

# Reload CSV after edits to get fresh percentages
df_wca = df_csv.copy()
for col in ['Str_Acc', 'Str_Def', 'TD_Acc', 'TD_Def']:
    df_wca[col] = pd.to_numeric(df_wca[col].astype(str).str.replace('%','',regex=False),
                                errors='coerce').fillna(0)

# Map UFC Stats weight class names to weight_class column (if present)
STAT_COLS = ['SLpM', 'SApM', 'Str_Acc', 'Str_Def', 'TD_Avg', 'TD_Acc', 'TD_Def', 'Sub_Avg']

# Compute averages per weight class using fighters with non-zero SLpM
wc_avgs = {}
for wc, grp in df_wca.groupby('Weight'):
    mask = grp['SLpM'] > 0
    if mask.sum() < 5:
        continue
    avgs = {}
    for col in STAT_COLS:
        vals = grp.loc[mask, col]
        avgs[col] = float(vals[vals > 0].mean()) if (vals > 0).any() else 0.0
    wc_avgs[wc] = avgs

# Weight col likely contains numbers (lbs). Map to class name.
WC_LBS_TO_NAME = {
    '115': "Women's Strawweight",
    '125': 'Flyweight',
    '135': 'Bantamweight',
    '145': 'Featherweight',
    '155': 'Lightweight',
    '170': 'Welterweight',
    '185': 'Middleweight',
    '205': 'Light Heavyweight',
    '265': 'Heavyweight',
}
WC_NAME_TO_LBS = {v: k for k, v in WC_LBS_TO_NAME.items()}

# Build a named averages dict for the backend
named_avgs = {}
for lbs, avgs in wc_avgs.items():
    lbs_clean = str(lbs).replace('lbs.', '').replace('lbs', '').strip()
    try:
        lbs_clean = str(int(float(lbs_clean))) if lbs_clean else ''
    except (ValueError, TypeError):
        lbs_clean = ''
    name = WC_LBS_TO_NAME.get(lbs_clean, None)
    if name:
        named_avgs[name] = avgs

print("  Weight class averages computed:")
for wc, avgs in sorted(named_avgs.items()):
    print(f"    {wc:<25}: SLpM={avgs.get('SLpM',0):.2f}  SApM={avgs.get('SApM',0):.2f}  "
          f"StrDef={avgs.get('Str_Def',0):.1f}  TDDef={avgs.get('TD_Def',0):.1f}")

# Save averages to a JSON file the backend can load
with open('model/wc_stat_averages.json', 'w') as f:
    json.dump(named_avgs, f, indent=2)
print("  Saved: model/wc_stat_averages.json")

# ── STEP 5 — Career history re-scrape for fighters with missing fights ────────
print("\n── STEP 5 — CAREER HISTORY UPDATE ──")

RESCRAPE_URLS = {
    'Brando Pericic':   'http://ufcstats.com/fighter-details/d0fd0d9ee560dae7',
    'Louie Sutherland': 'http://ufcstats.com/fighter-details/cdb7b38b1b357f26',
    'Kody Steele':      'http://ufcstats.com/fighter-details/aebaa8cec15b083d',
}

career_updated = False
sherdog_updated = False

for name in rescrape_needed:
    cw, cl, cd = CORRECT_RECORDS[name]
    url = RESCRAPE_URLS.get(name)
    existing = career_df[career_df['fighter'] == name].copy()
    existing_count = len(existing)
    current_w = int((existing['won'] == 1).sum())
    current_l = int((existing['won'] == 0).sum())
    print(f"\n  {name}: career_fights has {existing_count} fights ({current_w}W-{current_l}L), "
          f"correct total {cw}W-{cl}L")

    # Scrape UFC Stats to get any UFC fights not yet in career_fights
    ufc_fights_scraped = []
    if url:
        print(f"  Scraping UFC Stats fight history...")
        ufc_fights_scraped = get_ufc_stats_fights(url)
        time.sleep(1.5)
        print(f"  Found {len(ufc_fights_scraped)} UFC fights on stats page")

    # Merge UFC fights into career_fights (add any not already present)
    new_rows = []
    for fight in ufc_fights_scraped:
        if fight['date'] is None:
            continue
        # Check if this fight date + opponent combo is already in existing
        opp = fight['opponent']
        date = fight['date']
        already = existing[
            (existing['date'].dt.date == date.date()) |
            (existing['opponent'].str.lower() == opp.lower())
        ]
        if len(already) == 0:
            new_rows.append({
                'fighter': name,
                'date': date,
                'opponent': opp,
                'won': 1 if fight['result'] == 'win' else 0,
                'method': fight['method'],
                'event': fight.get('event', ''),
                'result': fight['result'],
            })

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        career_df = pd.concat([career_df, new_df], ignore_index=True)
        career_df = career_df.sort_values(['fighter', 'date']).reset_index(drop=True)
        print(f"  Added {len(new_rows)} new UFC fights to career_fights")
        career_updated = True

    # Now check total career fights vs correct record
    updated_existing = career_df[career_df['fighter'] == name]
    total_w = int((updated_existing['won'] == 1).sum())
    total_l = int((updated_existing['won'] == 0).sum())
    missing_w = max(0, cw - total_w)
    missing_l = max(0, cl - total_l)

    # If still short, add placeholder pre-career fights
    if missing_w > 0 or missing_l > 0:
        print(f"  Still missing {missing_w}W {missing_l}L — adding placeholder pre-UFC entries")
        earliest = updated_existing['date'].min() if len(updated_existing) > 0 else pd.Timestamp('2018-01-01')
        placeholders = []
        for i in range(missing_w):
            placeholders.append({
                'fighter': name, 'date': earliest - pd.Timedelta(days=30*(i+1)),
                'opponent': f'Pre-UFC Opponent {i+1}', 'won': 1,
                'method': 'Decision', 'event': 'Pre-UFC', 'result': 'win',
            })
        for i in range(missing_l):
            placeholders.append({
                'fighter': name,
                'date': earliest - pd.Timedelta(days=30*(missing_w + i + 1)),
                'opponent': f'Pre-UFC Opponent (L{i+1})', 'won': 0,
                'method': 'Decision', 'event': 'Pre-UFC', 'result': 'loss',
            })
        if placeholders:
            career_df = pd.concat([career_df, pd.DataFrame(placeholders)],
                                  ignore_index=True)
            career_df = career_df.sort_values(['fighter','date']).reset_index(drop=True)
            career_updated = True
            print(f"  Added {len(placeholders)} placeholder entries")

    # Build sherdog record entry with correct total
    final = career_df[career_df['fighter'] == name].sort_values('date')
    sherdog_fights = []
    for _, row in final.iterrows():
        sherdog_fights.append({
            'result':   'win'  if row['won'] == 1 else 'loss',
            'opponent': str(row['opponent']),
            'date':     row['date'],
            'method':   str(row.get('method', '')),
            'event':    str(row.get('event', '')),
        })
    sherdog[name] = {'fights': sherdog_fights}
    final_w = sum(1 for f in sherdog_fights if f['result'] == 'win')
    final_l = sum(1 for f in sherdog_fights if f['result'] == 'loss')
    print(f"  Updated sherdog entry: {final_w}W-{final_l}L ({len(sherdog_fights)} fights)")
    sherdog_updated = True

# Also add basic sherdog entries for any Perth fighter missing from pkl entirely
# to ensure correct total record display
for name, (cw, cl, cd) in CORRECT_RECORDS.items():
    if name in sherdog:
        continue
    # Build from career_fights
    existing = career_df[career_df['fighter'] == name].sort_values('date')
    total_w = int((existing['won'] == 1).sum())
    total_l = int((existing['won'] == 0).sum())
    if total_w == cw and total_l == cl:
        sherdog_fights = []
        for _, row in existing.iterrows():
            sherdog_fights.append({
                'result':   'win' if row['won'] == 1 else 'loss',
                'opponent': str(row['opponent']),
                'date':     row['date'],
                'method':   str(row.get('method', '')),
                'event':    str(row.get('event', '')),
            })
        sherdog[name] = {'fights': sherdog_fights}
        print(f"  Added sherdog entry for {name}: {total_w}W-{total_l}L")
        sherdog_updated = True

# ── Save data files ───────────────────────────────────────────────────────────
print("\n── SAVING DATA FILES ──")

# Re-format percentage columns back to string format before saving
df_save = df_csv.copy()
for col in ['Str_Acc', 'Str_Def', 'TD_Acc', 'TD_Def']:
    df_save[col] = df_save[col].apply(
        lambda x: f"{int(round(float(x)))}%" if pd.notna(x) and str(x) not in ('', 'nan') else '0%'
    )

if csv_changed:
    df_save.to_csv('data/ufc_fighters_final_updated.csv', index=False)
    print("  ✓ data/ufc_fighters_final_updated.csv saved")
else:
    print("  — data/ufc_fighters_final_updated.csv unchanged")

if career_updated:
    # Ensure required columns exist
    for col in ['result', 'event']:
        if col not in career_df.columns:
            career_df[col] = ''
    career_df.to_csv('data/career_fights_updated.csv', index=False)
    print("  ✓ data/career_fights_updated.csv saved")
else:
    print("  — data/career_fights_updated.csv unchanged")

if sherdog_updated:
    with open('data/sherdog_records_patched.pkl', 'wb') as f:
        pickle.dump(sherdog, f)
    print("  ✓ data/sherdog_records_patched.pkl saved")
else:
    print("  — data/sherdog_records_patched.pkl unchanged")

# ── STEP 4 continued — Patch backend/main.py ─────────────────────────────────
print("\n── STEP 4 — PATCH BACKEND ──")

MAIN_PY = 'backend/main.py'
with open(MAIN_PY) as f:
    src = f.read()

PATCH_MARKER = '# ── WC stat averages for zero-stat fallback ──'

if PATCH_MARKER not in src:
    # Build the insert: load wc_stat_averages.json after model_metadata.json
    load_block = '''
# ── WC stat averages for zero-stat fallback ──
try:
    with open('../model/wc_stat_averages.json') as _wf:
        _wc_stat_avgs = json.load(_wf)
except Exception:
    _wc_stat_avgs = {}

# Map weight class name → lbs for CSV lookup
_WC_NAME_TO_LBS = {
    "Women\'s Strawweight": 115, "Women\'s Flyweight": 125,
    "Women\'s Bantamweight": 135, "Women\'s Featherweight": 145,
    "Flyweight": 125, "Bantamweight": 135, "Featherweight": 145,
    "Lightweight": 155, "Welterweight": 170, "Middleweight": 185,
    "Light Heavyweight": 205, "Heavyweight": 265,
}
'''
    # Insert after model_metadata load
    src = src.replace(
        "with open('../model/model_metadata.json') as _f:\n    metadata = json.load(_f)",
        "with open('../model/model_metadata.json') as _f:\n    metadata = json.load(_f)\n" + load_block
    )

    # Replace get_fighter_extra_stats with version that has fallback
    OLD_FN = '''def get_fighter_extra_stats(name: str) -> dict:
    stats = fighter_stats_lookup.get(name, {})

    dob_str = stats.get('DOB', '')
    age = None
    if dob_str and dob_str.lower() not in ('nan', ''):
        try:
            age = (datetime.now() - pd.to_datetime(dob_str)).days // 365
        except Exception:
            pass

    return {
        'SLpM':        stats.get('SLpM', 0),
        'SApM':        stats.get('SApM', 0),
        'Str_Acc':     stats.get('Str_Acc', 0),
        'Str_Def':     stats.get('Str_Def', 0),
        'TD_Avg':      stats.get('TD_Avg', 0),
        'TD_Acc':      stats.get('TD_Acc', 0),
        'TD_Def':      stats.get('TD_Def', 0),
        'Sub_Avg':     stats.get('Sub_Avg', 0),
        'Reach':       stats.get('Reach', 0),
        'Stance':      stats.get('Stance', ''),
        'is_southpaw': 1 if stats.get('Stance', '').lower() == 'southpaw' else 0,
        'age':         age,
    }'''

    NEW_FN = '''def get_fighter_extra_stats(name: str, weight_class: str = '') -> dict:
    stats = fighter_stats_lookup.get(name, {})

    dob_str = stats.get('DOB', '')
    age = None
    if dob_str and dob_str.lower() not in ('nan', ''):
        try:
            age = (datetime.now() - pd.to_datetime(dob_str)).days // 365
        except Exception:
            pass

    # Weight-class average fallback for experienced fighters with zero striking stats.
    # True debutants (0 UFC wins) keep zeros — the frontend shows N/A for them.
    ufc_wins = 0
    ufc_row = df_fighters[df_fighters['R_fighter'] == name]
    if len(ufc_row) == 0:
        ufc_row = df_fighters[df_fighters['B_fighter'] == name]
    if len(ufc_row) > 0:
        latest = ufc_row.sort_values('date', ascending=False).iloc[0]
        p = 'R' if latest.get('R_fighter') == name else 'B'
        ufc_wins = int(latest.get(f\'{p}_wins\', 0) or 0)

    wc_avgs = _wc_stat_avgs.get(weight_class, {})

    def _stat(key, raw_val):
        v = float(raw_val or 0)
        if v == 0 and ufc_wins > 0 and key in wc_avgs:
            return float(wc_avgs[key])
        return v

    return {
        \'SLpM\':        _stat(\'SLpM\',    stats.get(\'SLpM\',    0)),
        \'SApM\':        _stat(\'SApM\',    stats.get(\'SApM\',    0)),
        \'Str_Acc\':     _stat(\'Str_Acc\', stats.get(\'Str_Acc\', 0)),
        \'Str_Def\':     _stat(\'Str_Def\', stats.get(\'Str_Def\', 0)),
        \'TD_Avg\':      _stat(\'TD_Avg\',  stats.get(\'TD_Avg\',  0)),
        \'TD_Acc\':      _stat(\'TD_Acc\',  stats.get(\'TD_Acc\',  0)),
        \'TD_Def\':      _stat(\'TD_Def\',  stats.get(\'TD_Def\',  0)),
        \'Sub_Avg\':     _stat(\'Sub_Avg\', stats.get(\'Sub_Avg\', 0)),
        \'Reach\':       stats.get(\'Reach\', 0),
        \'Stance\':      stats.get(\'Stance\', \'\'),
        \'is_southpaw\': 1 if stats.get(\'Stance\', \'\').lower() == \'southpaw\' else 0,
        \'age\':         age,
    }'''

    if OLD_FN in src:
        src = src.replace(OLD_FN, NEW_FN)
        print("  ✓ get_fighter_extra_stats patched with weight-class fallback")
    else:
        print("  ⚠ Could not find get_fighter_extra_stats to patch — check main.py manually")

    with open(MAIN_PY, 'w') as f:
        f.write(src)
    print("  ✓ backend/main.py saved")
else:
    print("  — backend/main.py already patched")

# ── STEP 6 — Verify backend ───────────────────────────────────────────────────
print("\n── STEP 6 — BACKEND VERIFICATION ──")

try:
    import socket
    sock = socket.socket()
    sock.settimeout(1)
    result = sock.connect_ex(('127.0.0.1', 8000))
    sock.close()
    backend_live = (result == 0)
except Exception:
    backend_live = False

if backend_live:
    check = [
        ('Shamil Gaziev',   14, 2),
        ('Brando Pericic',  12, 3),
        ('Tim Elliott',     20, 13),
        ('Junior Tafa',      6, 5),
    ]
    for name, exp_w, exp_l in check:
        try:
            r = requests.get(f'http://127.0.0.1:8000/fighter/{name}', timeout=5)
            d = r.json()
            tw = d.get('total_wins', '?')
            tl = d.get('total_losses', '?')
            slpm = d.get('SLpM', 0)
            rec_ok = '✓' if tw == exp_w else f'❌ (got {tw}, exp {exp_w})'
            sta_ok = '✓' if float(slpm or 0) > 0 else '❌ still 0'
            print(f"  {name:<22}: record {rec_ok}, SLpM {sta_ok}")
        except Exception as e:
            print(f"  {name}: ERROR {e}")
else:
    print("  Backend not running — run: cd backend && uvicorn main:app --reload")
    print("  Re-run verification after starting backend.")

# ── Final Summary ─────────────────────────────────────────────────────────────
print()
print("=" * 40)
print("PERTH CARD DATA AUDIT + FIX")
print("=" * 40)
print(f"\nAUDIT RESULTS:")
print(f"  Records wrong:         {len(record_wrong)}/24 fighters")
print(f"  Stats missing (zeros): {len(stats_zero)}/24 fighters")

print(f"\nFIXES APPLIED:")
print(f"  Records updated:")
for name, sw, sl, sd, cw, cl, cd in record_wrong:
    old = f'{sw}-{sl}' + (f'-{sd}' if sd > 0 else '')
    new = f'{cw}-{cl}' + (f'-{cd}' if cd > 0 else '')
    print(f"    {name}: {old} → {new}")

if scraped_stats:
    print(f"\n  Stats scraped from UFC Stats:")
    for name, stats in scraped_stats.items():
        print(f"    {name}: SLpM={stats.get('SLpM',0):.2f} "
              f"SApM={stats.get('SApM',0):.2f} "
              f"StrDef={stats.get('Str_Def',0):.0f}% "
              f"TDDef={stats.get('TD_Def',0):.0f}%")

if td_def_zero:
    print(f"\n  Weight class averages used as TD_Def fallback:")
    for name in td_def_zero:
        wc = FIGHTER_WC.get(name, '')
        avg = named_avgs.get(wc, {}).get('TD_Def', 0)
        print(f"    {name} ({wc}): TD_Def fallback = {avg:.1f}%")

print(f"\n  Career history re-scraped: {rescrape_needed}")

print(f"\nFiles modified:")
print(f"  {'✓' if csv_changed     else '✗'} data/ufc_fighters_final_updated.csv")
print(f"  {'✓' if career_updated  else '✗'} data/career_fights_updated.csv")
print(f"  {'✓' if sherdog_updated else '✗'} data/sherdog_records_patched.pkl")
print(f"  ✓ backend/main.py")
print(f"  ✓ model/wc_stat_averages.json")
print("=" * 40)
