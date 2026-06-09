"""
ufc_stats_scraper.py — Scrape UFC Stats and patch career data for active fighters.

Steps:
  1  Build active fighter list (2023+ fights)
  2  Find missing UFC Stats URLs
  3  Scrape fight history + fighter stats
  4  Patch career_fights.csv → career_fights_updated.csv
  5  Patch sherdog_records_fixed.pkl → sherdog_records_patched.pkl
  6  Update ufc_fighters_final.csv → ufc_fighters_final_updated.csv
  7  Verify spot-check fighters and print summary

NEVER modifies original files. Saves progress frequently. Resumable.
"""

import os
import re
import time
import pickle
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR   = "data"
SLEEP      = 1.0          # seconds between requests
SAVE_EVERY = 25           # save progress every N fighters (URL search)
SAVE_EVERY_SCRAPE = 50    # save progress every N fighters (scraping)
MAX_CONSEC_ERRORS  = 3
PAUSE_ON_ERRORS    = 60   # seconds to pause after MAX_CONSEC_ERRORS
PAUSE_ON_BLOCK     = 90   # seconds to pause on 429 / timeout

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
}

# Output files
STATUS_CSV   = os.path.join(DATA_DIR, "active_fighters_status.csv")
NOT_FOUND    = os.path.join(DATA_DIR, "fighters_not_found.csv")
FIGHTS_CSV   = os.path.join(DATA_DIR, "ufc_stats_fights.csv")
SCRAPED_CSV  = os.path.join(DATA_DIR, "ufc_fighters_scraped.csv")
ERRORS_CSV   = os.path.join(DATA_DIR, "ufc_stats_scrape_errors.csv")
CAREER_OUT   = os.path.join(DATA_DIR, "career_fights_updated.csv")
SHERDOG_OUT  = os.path.join(DATA_DIR, "sherdog_records_patched.pkl")
FIGHTERS_OUT = os.path.join(DATA_DIR, "ufc_fighters_final_updated.csv")

# Input files
MASTER_CSV   = os.path.join(DATA_DIR, "ufc-master.csv")
FIGHTERS_CSV = os.path.join(DATA_DIR, "ufc_fighters_final.csv")
CAREER_CSV   = os.path.join(DATA_DIR, "career_fights.csv")
SHERDOG_PKL  = os.path.join(DATA_DIR, "sherdog_records_fixed.pkl")
UNFIXED_CSV  = os.path.join(DATA_DIR, "still_unfixed.csv")

session = requests.Session()
session.headers.update(HEADERS)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def log(msg=""):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def safe_get(url, retries=3):
    """GET with retry logic. Returns (response, error_str)."""
    consec = 0
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=20)
            if r.status_code == 429:
                log(f"  429 rate-limit — pausing {PAUSE_ON_BLOCK}s ...")
                time.sleep(PAUSE_ON_BLOCK)
                continue
            if r.status_code != 200:
                return None, f"HTTP {r.status_code}"
            return r, None
        except requests.exceptions.Timeout:
            log(f"  Timeout on {url} (attempt {attempt+1}/{retries}) — pausing {PAUSE_ON_BLOCK}s")
            time.sleep(PAUSE_ON_BLOCK)
        except Exception as e:
            log(f"  Error on {url}: {e}")
            time.sleep(SLEEP * 3)
    return None, "max retries exceeded"


def normalize_name(s):
    return " ".join(str(s).strip().lower().split())


def clean_text(td):
    return td.get_text(separator=" ", strip=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Build active fighter list
# ─────────────────────────────────────────────────────────────────────────────
def step1_build_active_list():
    log("=" * 60)
    log("STEP 1 — Build active fighter list")
    log("=" * 60)

    master = pd.read_csv(MASTER_CSV, low_memory=False)
    master["date"] = pd.to_datetime(master["date"])
    recent = master[master["date"] >= "2023-01-01"]
    red    = set(recent["R_fighter"].dropna().str.strip())
    blue   = set(recent["B_fighter"].dropna().str.strip())
    active = sorted(red | blue)
    log(f"Active fighters (2023+): {len(active)}")

    fighters_df = pd.read_csv(FIGHTERS_CSV)

    # Build lookup: name → URL
    name_to_url = {}
    for _, row in fighters_df.iterrows():
        name_to_url[str(row["Fighter_Name"]).strip()] = str(row.get("Fighter_URL", "")).strip()

    rows = []
    for name in active:
        url   = name_to_url.get(name, "")
        has   = bool(url and url.lower() != "nan")
        rows.append({"fighter_name": name, "has_ufc_stats_url": has, "ufc_stats_url": url if has else ""})

    status_df = pd.DataFrame(rows)
    status_df.to_csv(STATUS_CSV, index=False)
    log(f"Saved {STATUS_CSV}  ({len(status_df)} fighters)")

    with_url    = status_df["has_ufc_stats_url"].sum()
    without_url = (~status_df["has_ufc_stats_url"]).sum()
    log(f"  Already have URL: {with_url}")
    log(f"  Missing URL:      {without_url}")

    return status_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Find missing UFC Stats URLs
# ─────────────────────────────────────────────────────────────────────────────
def _parse_fighter_list_page(html, target_name):
    """Search a fighter-list page for target_name. Returns URL or None."""
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select("table.b-statistics__table tbody tr")
    if not rows:
        rows = soup.select("table tbody tr")
    tn   = normalize_name(target_name)
    # Also try with hyphens normalized to spaces (UFC Stats splits hyphenated names)
    tn_dehyphen = tn.replace("-", " ")
    target_parts = tn.split()

    candidates = []
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        a = cells[0].find("a") or cells[1].find("a")
        if not a:
            continue
        href = a.get("href", "")
        if "fighter-details" not in href:
            continue
        first = cells[0].get_text(strip=True).lower().strip()
        last  = cells[1].get_text(strip=True).lower().strip() if len(cells) > 1 else ""
        # UFC Stats format: first=given name, last=family name (may include spaces for hyphenated)
        full_ufc   = normalize_name(f"{first} {last}")
        # Also reconstruct as "first last" (matching our target format)
        candidates.append((full_ufc, href, first, last))

    # Exact full-name match (original or dehyphenated)
    for full, href, first, last in candidates:
        if full == tn or full == tn_dehyphen:
            return href

    # Match: target first + target last (normalized, dehyphenated)
    target_first = target_parts[0] if target_parts else ""
    target_last  = " ".join(target_parts[1:]).replace("-", " ") if len(target_parts) > 1 else ""

    last_matches = []
    for full, href, first, last in candidates:
        # Normalize last name for comparison
        last_norm = last.replace("-", " ")
        if last_norm == target_last and first == target_first:
            return href   # first+last exact match
        if last_norm == target_last:
            last_matches.append((full, href, first, last))

    if len(last_matches) == 1:
        return last_matches[0][1]
    if len(last_matches) > 1:
        # Among last-name matches, pick the one with matching first name
        for full, href, first, last in last_matches:
            if first == target_first:
                return href
        return last_matches[0][1]

    # First-name-only fallback
    first_matches = [(full, href) for full, href, first, last in candidates if first == target_first]
    if len(first_matches) == 1:
        return first_matches[0][1]

    return None


def step2_find_missing_urls(status_df):
    log()
    log("=" * 60)
    log("STEP 2 — Find UFC Stats URLs for fighters missing them")
    log("=" * 60)

    missing = status_df[~status_df["has_ufc_stats_url"]].copy()
    log(f"Fighters needing URL lookup: {len(missing)}")

    if len(missing) == 0:
        log("All active fighters already have URLs — skipping Step 2.")
        return status_df

    not_found_rows = []
    consec_errors  = 0
    letter_cache   = {}   # letter → html (avoid duplicate fetches)
    new_urls       = 0

    for idx, (_, row) in enumerate(missing.iterrows()):
        name   = row["fighter_name"]
        parts  = name.strip().split()
        letter = parts[-1][0].lower() if parts else "a"
        list_url = f"http://ufcstats.com/statistics/fighters?char={letter}&page=all"

        if letter not in letter_cache:
            time.sleep(SLEEP)
            resp, err = safe_get(list_url)
            if err:
                consec_errors += 1
                log(f"  [{idx+1}/{len(missing)}] {name} — list page error: {err}")
                not_found_rows.append({"fighter": name, "reason": f"list page error: {err}"})
                if consec_errors >= MAX_CONSEC_ERRORS:
                    log(f"  {MAX_CONSEC_ERRORS} consecutive errors — pausing {PAUSE_ON_ERRORS}s")
                    time.sleep(PAUSE_ON_ERRORS)
                    consec_errors = 0
                continue
            consec_errors = 0
            letter_cache[letter] = resp.text
        else:
            consec_errors = 0

        fighter_url = _parse_fighter_list_page(letter_cache[letter], name)

        if fighter_url:
            if not fighter_url.startswith("http"):
                fighter_url = "http://ufcstats.com" + fighter_url
            # Update status_df
            mask = status_df["fighter_name"] == name
            status_df.loc[mask, "has_ufc_stats_url"] = True
            status_df.loc[mask, "ufc_stats_url"]     = fighter_url
            new_urls += 1
            log(f"  [{idx+1}/{len(missing)}] {name} → {fighter_url}")
        else:
            not_found_rows.append({"fighter": name, "reason": "no match on list page"})
            log(f"  [{idx+1}/{len(missing)}] {name} — NOT FOUND")

        if (idx + 1) % SAVE_EVERY == 0:
            status_df.to_csv(STATUS_CSV, index=False)
            log(f"  Progress saved ({idx+1}/{len(missing)})")

    # Save
    status_df.to_csv(STATUS_CSV, index=False)
    log(f"Saved {STATUS_CSV}  (new URLs found: {new_urls})")

    if not_found_rows:
        pd.DataFrame(not_found_rows).to_csv(NOT_FOUND, index=False)
        log(f"Saved {NOT_FOUND}  ({len(not_found_rows)} fighters not found)")

    return status_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Scrape fight history for all active fighters
# ─────────────────────────────────────────────────────────────────────────────
def _parse_fighter_page(html, fighter_name):
    """
    Returns:
      fighter_stats: dict or None
      fight_rows:    list of dicts
    """
    soup = BeautifulSoup(html, "html.parser")

    # ── Fighter stats ──────────────────────────────────────────────────────
    # All li items in the two stat lists
    all_li_text = {}
    for li in soup.select("ul.b-list__box-list li"):
        txt = li.get_text(" ", strip=True)
        if ":" in txt:
            key, _, val = txt.partition(":")
            k = key.strip().lower()
            v = val.strip()
            if v and v != "--":
                all_li_text[k] = v

    def _li(label):
        return all_li_text.get(label.lower().rstrip(":"), None)

    def _to_cm(h_str):
        if not h_str:
            return None
        m = re.search(r"(\d+)'\s*(\d+)", str(h_str))
        if m:
            return round(int(m.group(1)) * 30.48 + int(m.group(2)) * 2.54, 1)
        return None

    def _reach_cm(r_str):
        if not r_str:
            return None
        m = re.search(r'([\d.]+)"', str(r_str))
        if m:
            return round(float(m.group(1)) * 2.54, 1)
        return None

    def _pct(val):
        if not val:
            return None
        return float(str(val).replace("%", "").strip()) / 100.0

    # Record W/L/D — "Record: 22-10-0"
    wins, losses, draws = None, None, None
    for span in soup.select("span.b-content__title-record"):
        txt = span.get_text(strip=True)
        m   = re.search(r"(\d+)-(\d+)-(\d+)", txt)
        if m:
            wins, losses, draws = int(m.group(1)), int(m.group(2)), int(m.group(3))
            break

    fighter_stats = {
        "fighter":   fighter_name,
        "height_cm": _to_cm(_li("height")),
        "reach_cm":  _reach_cm(_li("reach")),
        "stance":    _li("stance"),
        "dob":       _li("dob"),
        "wins":      wins,
        "losses":    losses,
        "draws":     draws,
        "slpm":      _to_float(_li("slpm")),
        "str_acc":   _pct(_li("str. acc.")),
        "sapm":      _to_float(_li("sapm")),
        "str_def":   _pct(_li("str. def")),
        "td_avg":    _to_float(_li("td avg.")),
        "td_acc":    _pct(_li("td acc.")),
        "td_def":    _pct(_li("td def.")),
        "sub_avg":   _to_float(_li("sub. avg.")),
    }

    # ── Fight history table ────────────────────────────────────────────────
    fight_rows = []
    fn_norm    = normalize_name(fighter_name)

    table = soup.find("table", class_="b-fight-details__table")
    if table:
        tbody = table.find("tbody")
        if tbody:
            for tr in tbody.find_all("tr"):
                cls_list = tr.get("class", [])
                # Skip header rows (class b-statistics__table-row)
                if "b-statistics__table-row" in cls_list:
                    continue
                tds = tr.find_all("td")
                if len(tds) < 8:
                    continue

                # Col 0: result
                result_raw = tds[0].get_text(strip=True).lower()
                result_map = {
                    "win": "win", "loss": "loss", "draw": "draw",
                    "nc": "no contest", "no contest": "no contest",
                }
                result = result_map.get(result_raw, result_raw)

                # Col 1: two <p> tags each with one <a> fighter link
                # First <p><a> = current fighter, second = opponent
                col1_paras = tds[1].find_all("p")
                links_in_col1 = [p.find("a") for p in col1_paras if p.find("a")]
                if len(links_in_col1) >= 2:
                    f_a = normalize_name(links_in_col1[0].get_text(strip=True))
                    f_b = normalize_name(links_in_col1[1].get_text(strip=True))
                    if f_a == fn_norm:
                        opp = links_in_col1[1].get_text(strip=True)
                    elif f_b == fn_norm:
                        opp = links_in_col1[0].get_text(strip=True)
                    else:
                        opp = links_in_col1[1].get_text(strip=True)
                elif len(links_in_col1) == 1:
                    opp = links_in_col1[0].get_text(strip=True)
                else:
                    # Fallback: all <a> tags in the cell
                    all_a = tds[1].find_all("a")
                    if len(all_a) >= 2:
                        opp = all_a[1].get_text(strip=True)
                    elif len(all_a) == 1:
                        opp = all_a[0].get_text(strip=True)
                    else:
                        opp = tds[1].get_text(" ", strip=True)

                # Col 6: two <p> tags — first has event link, second has date
                col6_paras = tds[6].find_all("p")
                event_name = ""
                event_date = None
                if col6_paras:
                    event_a = col6_paras[0].find("a")
                    if event_a:
                        event_name = event_a.get_text(strip=True)
                if len(col6_paras) >= 2:
                    date_raw = col6_paras[1].get_text(strip=True)
                    try:
                        event_date = pd.to_datetime(date_raw).strftime("%Y-%m-%d")
                    except Exception:
                        event_date = date_raw if date_raw else None
                if not event_date:
                    # Fallback: any span
                    date_span = tds[6].find("span")
                    if date_span:
                        try:
                            event_date = pd.to_datetime(date_span.get_text(strip=True)).strftime("%Y-%m-%d")
                        except Exception:
                            pass

                # Col 7: method
                method = tds[7].get_text(" ", strip=True) if len(tds) > 7 else ""

                # Col 8: round
                rnd = tds[8].get_text(strip=True) if len(tds) > 8 else ""

                # Col 9: time
                fight_time = tds[9].get_text(strip=True) if len(tds) > 9 else ""

                if not event_date or not opp:
                    continue

                fight_rows.append({
                    "fighter":  fighter_name,
                    "opponent": opp.strip(),
                    "date":     event_date,
                    "result":   result,
                    "method":   method.strip(),
                    "event":    event_name.strip(),
                    "round":    rnd,
                    "time":     fight_time,
                })

    return fighter_stats, fight_rows


def _to_float(s):
    if s is None:
        return None
    try:
        return float(str(s).replace(",", "").strip())
    except Exception:
        return None


def step3_scrape_fights(status_df):
    log()
    log("=" * 60)
    log("STEP 3 — Scrape fight history for all active fighters")
    log("=" * 60)

    fighters_with_url = status_df[status_df["has_ufc_stats_url"]].copy()
    log(f"Fighters to scrape: {len(fighters_with_url)}")

    # Load already-scraped fighters to allow resume
    already_done = set()
    existing_fights   = []
    existing_stats    = []
    existing_errors   = []

    if os.path.exists(FIGHTS_CSV):
        prior_fights = pd.read_csv(FIGHTS_CSV)
        already_done.update(prior_fights["fighter"].unique())
        existing_fights = prior_fights.to_dict("records")
        log(f"  Resume: {len(already_done)} fighters already scraped ({len(existing_fights)} fight rows)")

    if os.path.exists(SCRAPED_CSV):
        prior_stats = pd.read_csv(SCRAPED_CSV)
        existing_stats = prior_stats.to_dict("records")

    if os.path.exists(ERRORS_CSV):
        prior_errors = pd.read_csv(ERRORS_CSV)
        existing_errors = prior_errors.to_dict("records")

    new_fights = list(existing_fights)
    new_stats  = list(existing_stats)
    new_errors = list(existing_errors)
    consec_errors = 0
    scraped_this_run = 0

    todo = fighters_with_url[~fighters_with_url["fighter_name"].isin(already_done)]
    log(f"  Remaining to scrape this run: {len(todo)}")

    for i, (_, row) in enumerate(todo.iterrows()):
        name = row["fighter_name"]
        url  = str(row["ufc_stats_url"]).strip()

        if not url or url == "nan":
            continue

        time.sleep(SLEEP)
        resp, err = safe_get(url)

        if err or resp is None:
            consec_errors += 1
            new_errors.append({"fighter": name, "url": url, "error": err or "no response"})
            log(f"  [{i+1}/{len(todo)}] {name} — ERROR: {err}")
            if consec_errors >= MAX_CONSEC_ERRORS:
                log(f"  {MAX_CONSEC_ERRORS} consecutive errors — pausing {PAUSE_ON_ERRORS}s")
                time.sleep(PAUSE_ON_ERRORS)
                consec_errors = 0
            continue

        consec_errors = 0

        try:
            stats, fights = _parse_fighter_page(resp.text, name)
        except Exception as e:
            new_errors.append({"fighter": name, "url": url, "error": str(e)})
            log(f"  [{i+1}/{len(todo)}] {name} — PARSE ERROR: {e}")
            continue

        new_fights.extend(fights)
        if stats:
            # Remove prior entry for this fighter if resuming with updated stats
            new_stats = [s for s in new_stats if s.get("fighter") != name]
            new_stats.append(stats)

        scraped_this_run += 1

        if (i + 1) % SAVE_EVERY_SCRAPE == 0 or (i + 1) == len(todo):
            _save_scrape_progress(new_fights, new_stats, new_errors)
            log(f"  Progress saved — fighter {i+1}/{len(todo)}: {name} ({len(fights)} fights)")
        else:
            if (i + 1) % 25 == 0:
                log(f"  [{i+1}/{len(todo)}] {name} — {len(fights)} fights")

    _save_scrape_progress(new_fights, new_stats, new_errors)
    log(f"Scraping complete: {scraped_this_run} new fighters, {len(new_fights)} total fight rows")
    return new_fights, new_stats, new_errors


def _save_scrape_progress(fights, stats, errors):
    if fights:
        pd.DataFrame(fights).to_csv(FIGHTS_CSV, index=False)
    if stats:
        pd.DataFrame(stats).to_csv(SCRAPED_CSV, index=False)
    if errors:
        pd.DataFrame(errors).to_csv(ERRORS_CSV, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Patch career_fights.csv
# ─────────────────────────────────────────────────────────────────────────────
def _recompute_rolling(df):
    """Recompute won, got_finish, last5_won, last5_finish_rate for a fighter's sorted history."""
    df = df.sort_values("date").reset_index(drop=True)

    df["won"] = (df["result"].str.lower() == "win").astype(int)
    df["got_finish"] = (
        (df["won"] == 1) &
        df["method"].str.contains("KO|TKO|Submission|Sub", case=False, na=False)
    ).astype(int)

    df["last5_won"]          = df["won"].shift(1).rolling(5, min_periods=1).mean()
    df["last5_finish_rate"]  = df["got_finish"].shift(1).rolling(5, min_periods=1).mean()

    return df


def step4_patch_career_fights():
    log()
    log("=" * 60)
    log("STEP 4 — Patch career_fights.csv")
    log("=" * 60)

    career  = pd.read_csv(CAREER_CSV)
    career["date"] = pd.to_datetime(career["date"])
    original_rows = len(career)
    log(f"Original career_fights: {original_rows} rows")

    if not os.path.exists(FIGHTS_CSV):
        log(f"ERROR: {FIGHTS_CSV} not found — run Step 3 first")
        return career

    ufc_fights = pd.read_csv(FIGHTS_CSV)
    ufc_fights["date"] = pd.to_datetime(ufc_fights["date"], errors="coerce")
    ufc_fights = ufc_fights.dropna(subset=["date"])
    log(f"UFC Stats fights: {len(ufc_fights)} rows")

    unfixed_df  = pd.read_csv(UNFIXED_CSV)
    unfixed_set = set(unfixed_df["fighter"].str.strip())
    log(f"still_unfixed fighters: {len(unfixed_set)}")

    # Active fighters who have UFC Stats fight data
    fighters_in_ufc = set(ufc_fights["fighter"].unique())
    log(f"Fighters in ufc_stats_fights.csv: {len(fighters_in_ufc)}")

    patched_fighters = set()
    new_rows_added   = 0
    unfixed_patched  = 0
    groups_career    = career.groupby("fighter")

    all_updated = []

    # Process all fighters from ufc_stats_fights.csv
    for fighter, ufc_grp in ufc_fights.groupby("fighter"):
        ufc_grp = ufc_grp.copy().sort_values("date").reset_index(drop=True)

        # Build standardized fight records from UFC Stats
        ufc_records = []
        for _, fr in ufc_grp.iterrows():
            ufc_records.append({
                "fighter":  fighter,
                "opponent": fr["opponent"],
                "date":     fr["date"],
                "result":   str(fr["result"]).lower().strip(),
                "method":   str(fr["method"]).strip(),
                "won":      1 if str(fr["result"]).lower().strip() == "win" else 0,
                "got_finish": 0,  # will recompute
                "last5_won": np.nan,
                "last5_finish_rate": np.nan,
            })

        if fighter in unfixed_set:
            # Replace entirely with UFC Stats data
            existing_other = career[career["fighter"] != fighter].copy()
            new_df = pd.DataFrame(ufc_records)
            new_df = _recompute_rolling(new_df)
            all_updated.append(new_df)
            unfixed_patched += 1
            patched_fighters.add(fighter)
            continue

        # For normal fighters: find missing fights and append
        if fighter in groups_career.groups:
            existing = career[career["fighter"] == fighter].copy()
        else:
            existing = pd.DataFrame(columns=career.columns)

        existing["date"] = pd.to_datetime(existing["date"])

        # Match: fighter + opponent + date within 7 days
        def already_exists(opp, dt):
            if len(existing) == 0:
                return False
            opp_norm = normalize_name(opp)
            for _, er in existing.iterrows():
                if normalize_name(str(er["opponent"])) == opp_norm:
                    delta = abs((er["date"] - dt).days)
                    if delta <= 7:
                        return True
            return False

        missing_fights = []
        for rec in ufc_records:
            if not already_exists(rec["opponent"], rec["date"]):
                missing_fights.append(rec)

        if missing_fights:
            new_rows_added += len(missing_fights)
            patched_fighters.add(fighter)
            combined = pd.concat(
                [existing, pd.DataFrame(missing_fights)],
                ignore_index=True
            )
            combined = _recompute_rolling(combined)
            all_updated.append(combined)
        else:
            all_updated.append(existing)

    # Also keep career rows for fighters NOT in ufc_stats_fights
    already_handled = set(ufc_fights["fighter"].unique())
    untouched = career[~career["fighter"].isin(already_handled)].copy()
    all_updated.append(untouched)

    updated = pd.concat(all_updated, ignore_index=True)
    updated["date"] = pd.to_datetime(updated["date"])
    updated = updated.sort_values(["fighter", "date"]).reset_index(drop=True)

    # Ensure correct column order
    for col in ["won", "got_finish", "last5_won", "last5_finish_rate"]:
        if col not in updated.columns:
            updated[col] = np.nan

    updated = updated[["fighter", "opponent", "date", "result", "method",
                        "won", "got_finish", "last5_won", "last5_finish_rate"]]

    updated.to_csv(CAREER_OUT, index=False)
    log(f"Saved {CAREER_OUT}")
    log(f"  Original rows:   {original_rows}")
    log(f"  New rows added:  {new_rows_added}")
    log(f"  Total rows:      {len(updated)}")
    log(f"  Fighters patched: {len(patched_fighters)}")
    log(f"  still_unfixed patched: {unfixed_patched}/{len(unfixed_set)}")

    return updated, patched_fighters, new_rows_added, unfixed_patched


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Patch sherdog_records_fixed.pkl
# ─────────────────────────────────────────────────────────────────────────────
def step5_patch_sherdog():
    log()
    log("=" * 60)
    log("STEP 5 — Patch sherdog_records_fixed.pkl")
    log("=" * 60)

    with open(SHERDOG_PKL, "rb") as f:
        sherdog = pickle.load(f)
    log(f"Loaded sherdog_records_fixed.pkl: {len(sherdog)} entries")

    unfixed_df  = pd.read_csv(UNFIXED_CSV)
    unfixed_set = set(unfixed_df["fighter"].str.strip())

    if not os.path.exists(FIGHTS_CSV):
        log(f"ERROR: {FIGHTS_CSV} not found — run Step 3 first")
        return sherdog

    ufc_fights = pd.read_csv(FIGHTS_CSV)
    ufc_fights["date"] = pd.to_datetime(ufc_fights["date"], errors="coerce")
    ufc_fights = ufc_fights.dropna(subset=["date"])

    patched = 0
    for fighter in unfixed_set:
        grp = ufc_fights[ufc_fights["fighter"] == fighter]
        if len(grp) == 0:
            log(f"  {fighter} — no UFC Stats data found, skipping")
            continue

        fights_list = []
        for _, fr in grp.sort_values("date").iterrows():
            fights_list.append({
                "result":   str(fr["result"]).lower().strip(),
                "opponent": str(fr["opponent"]).strip(),
                "date":     fr["date"],
                "method":   str(fr["method"]).strip(),
                "event":    str(fr.get("event", "")).strip(),
            })

        sherdog[fighter] = {"fights": fights_list}
        patched += 1

    with open(SHERDOG_OUT, "wb") as f:
        pickle.dump(sherdog, f)
    log(f"Saved {SHERDOG_OUT}  (patched {patched}/{len(unfixed_set)} unfixed fighters)")

    return sherdog


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Update ufc_fighters_final.csv
# ─────────────────────────────────────────────────────────────────────────────
def step6_update_fighters_final():
    log()
    log("=" * 60)
    log("STEP 6 — Update ufc_fighters_final.csv")
    log("=" * 60)

    orig = pd.read_csv(FIGHTERS_CSV)
    orig_count = len(orig)
    log(f"Original ufc_fighters_final: {orig_count} fighters")

    if not os.path.exists(SCRAPED_CSV):
        log(f"WARNING: {SCRAPED_CSV} not found — skipping Step 6")
        orig.to_csv(FIGHTERS_OUT, index=False)
        return orig

    scraped = pd.read_csv(SCRAPED_CSV)
    log(f"Scraped stats: {len(scraped)} fighters")

    # Column mapping: scraped → ufc_fighters_final
    col_map = {
        "fighter":   "Fighter_Name",
        "height_cm": "Height",
        "reach_cm":  "Reach",
        "stance":    "Stance",
        "dob":       "DOB",
        "wins":      "Wins",
        "losses":    "Losses",
        "draws":     "Draws",
        "slpm":      "SLpM",
        "str_acc":   "Str_Acc",
        "sapm":      "SApM",
        "str_def":   "Str_Def",
        "td_avg":    "TD_Avg",
        "td_acc":    "TD_Acc",
        "td_def":    "TD_Def",
        "sub_avg":   "Sub_Avg",
    }

    # Build lookup from scraped data (fighter → row)
    scraped_lookup = {}
    for _, row in scraped.iterrows():
        name = str(row["fighter"]).strip()
        scraped_lookup[name] = row

    updated_rows = orig.to_dict("records")
    orig_names   = {str(r["Fighter_Name"]).strip() for r in updated_rows}

    # Update existing fighters
    updated_count = 0
    for rec in updated_rows:
        name = str(rec["Fighter_Name"]).strip()
        if name not in scraped_lookup:
            continue
        sr = scraped_lookup[name]
        for sc, oc in col_map.items():
            if sc == "fighter":
                continue
            val = sr.get(sc)
            if val is not None and str(val).lower() not in ("nan", "none", ""):
                # For str_acc/str_def/td_acc/td_def: convert fraction to pct string
                if sc in ("str_acc", "str_def", "td_acc", "td_def"):
                    try:
                        v = float(val)
                        # If already a fraction (< 1.5), convert to pct string
                        if v <= 1.5:
                            rec[oc] = f"{v*100:.0f}%"
                        else:
                            rec[oc] = f"{v:.0f}%"
                    except Exception:
                        rec[oc] = val
                else:
                    rec[oc] = val
        updated_count += 1

    # Add new fighters
    new_added = 0
    for name, sr in scraped_lookup.items():
        if name in orig_names:
            continue
        new_rec = {"Fighter_Name": name}
        for sc, oc in col_map.items():
            if sc == "fighter":
                continue
            val = sr.get(sc)
            if sc in ("str_acc", "str_def", "td_acc", "td_def") and val is not None:
                try:
                    v = float(val)
                    new_rec[oc] = f"{v*100:.0f}%" if v <= 1.5 else f"{v:.0f}%"
                except Exception:
                    new_rec[oc] = val
            else:
                new_rec[oc] = val
        # Set Fighter_URL from status if available
        new_rec["Fighter_URL"] = ""
        if os.path.exists(STATUS_CSV):
            status_df = pd.read_csv(STATUS_CSV)
            match = status_df[status_df["fighter_name"] == name]
            if len(match) > 0:
                new_rec["Fighter_URL"] = str(match.iloc[0]["ufc_stats_url"])
        # Fill missing cols
        for col in orig.columns:
            if col not in new_rec:
                new_rec[col] = np.nan
        updated_rows.append(new_rec)
        new_added += 1

    final_df = pd.DataFrame(updated_rows, columns=orig.columns)
    final_df.to_csv(FIGHTERS_OUT, index=False)
    log(f"Saved {FIGHTERS_OUT}")
    log(f"  Original fighters: {orig_count}")
    log(f"  Updated fighters:  {updated_count}")
    log(f"  New fighters added: {new_added}")
    log(f"  Total: {len(final_df)}")

    return final_df, orig_count, new_added


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Verify
# ─────────────────────────────────────────────────────────────────────────────
def step7_verify():
    log()
    log("=" * 60)
    log("STEP 7 — Verification")
    log("=" * 60)

    if not os.path.exists(CAREER_OUT):
        log("career_fights_updated.csv not found — skipping verify")
        return

    career = pd.read_csv(CAREER_OUT)
    career["date"] = pd.to_datetime(career["date"])

    spot_check = ["Josh Hokit", "Aaron Pico", "Carlos Ulberg", "Charles Radtke"]

    # Add one random still_unfixed fighter
    unfixed_df = pd.read_csv(UNFIXED_CSV)
    unfixed_set = list(unfixed_df["fighter"].str.strip())
    random_unfixed = None
    for f in unfixed_set:
        if f in set(career["fighter"]):
            random_unfixed = f
            break
    if random_unfixed:
        spot_check.append(random_unfixed)

    for name in spot_check:
        grp = career[career["fighter"] == name].sort_values("date")
        if len(grp) == 0:
            log(f"  {name}: NOT FOUND in career_fights_updated.csv")
            continue
        wins   = int(grp["won"].sum()) if "won" in grp.columns else "?"
        losses = len(grp) - wins if isinstance(wins, int) else "?"
        last5  = grp.tail(5)[["date", "result", "opponent"]].values
        log(f"\n  {name}: {len(grp)} fights ({wins}W-{losses}L)")
        log(f"  Last 5:")
        for row in last5:
            log(f"    {str(row[0])[:10]}  {str(row[1]):6}  vs {row[2]}")

    # Unfixed patch count
    in_updated = sum(1 for f in unfixed_set if f in set(career["fighter"]))
    log(f"\n  still_unfixed fighters in career_fights_updated: {in_updated}/{len(unfixed_set)}")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(status_df, new_rows_added, patched_fighters, unfixed_patched,
                  orig_career_rows, total_career_rows,
                  orig_fighter_count, new_fighters_added,
                  scrape_errors_count):
    log()
    log("=" * 40)
    log("UFC STATS SCRAPING — FINAL SUMMARY")
    log("=" * 40)

    total   = len(status_df)
    had_url = int(status_df["has_ufc_stats_url"].sum())
    # Load not_found to get count
    not_found_count = 0
    if os.path.exists(NOT_FOUND):
        not_found_count = len(pd.read_csv(NOT_FOUND))
    new_urls_found = had_url - int(pd.read_csv(STATUS_CSV)["has_ufc_stats_url"].sum() == had_url)
    # Recompute from saved file
    saved_status = pd.read_csv(STATUS_CSV)
    final_with_url = int(saved_status["has_ufc_stats_url"].sum())
    orig_with_url  = total - (total - final_with_url) - not_found_count
    new_urls_found = final_with_url - orig_with_url

    print()
    print("=" * 40)
    print("UFC STATS SCRAPING — FINAL SUMMARY")
    print("=" * 40)
    print(f"Active fighters identified: {total}")
    print(f"Fighters already had UFC Stats URL: {orig_with_url}")
    print(f"New URLs found this run: {max(0, new_urls_found)}")
    print(f"Fighters still not found: {not_found_count}")
    print()

    fights_total = 0
    if os.path.exists(FIGHTS_CSV):
        fights_total = len(pd.read_csv(FIGHTS_CSV))
    scraped_count = len(pd.read_csv(SCRAPED_CSV)) if os.path.exists(SCRAPED_CSV) else 0
    print(f"Fight history scraped:")
    print(f"  Total fighters scraped: {scraped_count}")
    print(f"  Total fight rows collected: {fights_total}")
    print(f"  Scrape errors/failures: {scrape_errors_count}")
    print()

    print(f"career_fights_updated.csv:")
    print(f"  Original rows: {orig_career_rows:,}")
    print(f"  New rows added: {new_rows_added}")
    print(f"  Total rows: {total_career_rows:,}")
    print(f"  Fighters with updated history: {len(patched_fighters)}")
    print(f"  still_unfixed fighters patched: {unfixed_patched} / 66")
    print()

    print(f"ufc_fighters_final_updated.csv:")
    print(f"  Original fighters: {orig_fighter_count}")
    print(f"  New fighters added: {new_fighters_added}")
    print(f"  Total fighters: {orig_fighter_count + new_fighters_added}")
    print()

    def check(path):
        return "✓" if os.path.exists(path) else "✗"

    print("Files saved:")
    print(f"  {check(FIGHTS_CSV)} {FIGHTS_CSV}")
    print(f"  {check(CAREER_OUT)} {CAREER_OUT}")
    print(f"  {check(FIGHTERS_OUT)} {FIGHTERS_OUT}")
    print(f"  {check(SHERDOG_OUT)} {SHERDOG_OUT}")
    print(f"  {check(NOT_FOUND)} {NOT_FOUND}")
    print(f"  {check(STATUS_CSV)} {STATUS_CSV}")
    print(f"  {check(SCRAPED_CSV)} {SCRAPED_CSV}")
    print(f"  {check(ERRORS_CSV)} {ERRORS_CSV}")
    print()

    # Spot checks
    if os.path.exists(CAREER_OUT):
        career = pd.read_csv(CAREER_OUT)
        career["date"] = pd.to_datetime(career["date"])
        print("Spot checks:")
        for name in ["Josh Hokit", "Aaron Pico", "Carlos Ulberg"]:
            grp = career[career["fighter"] == name].sort_values("date")
            if len(grp) == 0:
                print(f"  {name}: NOT FOUND")
                continue
            wins   = int(grp["won"].sum())
            losses = len(grp) - wins
            last   = grp.iloc[-1]
            print(f"  {name}: {len(grp)} fights ({wins}-{losses}), last fight: {str(last['date'])[:10]}")
    print("=" * 40)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    log("=" * 60)
    log("UFC STATS SCRAPER — START")
    log(f"Timestamp: {datetime.now().isoformat()}")
    log("=" * 60)

    # Step 1
    status_df = step1_build_active_list()
    orig_career_rows = len(pd.read_csv(CAREER_CSV))

    # Step 2
    status_df = step2_find_missing_urls(status_df)

    # Step 3
    new_fights, new_stats, new_errors = step3_scrape_fights(status_df)
    scrape_errors_count = len(new_errors)

    # Step 4
    result4 = step4_patch_career_fights()
    if isinstance(result4, tuple):
        updated_career, patched_fighters, new_rows_added, unfixed_patched = result4
        total_career_rows = len(updated_career)
    else:
        patched_fighters, new_rows_added, unfixed_patched = set(), 0, 0
        total_career_rows = orig_career_rows

    # Step 5
    step5_patch_sherdog()

    # Step 6
    result6 = step6_update_fighters_final()
    if isinstance(result6, tuple):
        _, orig_fighter_count, new_fighters_added = result6
    else:
        orig_fighter_count = 4455
        new_fighters_added = 0

    # Step 7
    step7_verify()

    # Summary
    print_summary(
        status_df=status_df,
        new_rows_added=new_rows_added,
        patched_fighters=patched_fighters,
        unfixed_patched=unfixed_patched,
        orig_career_rows=orig_career_rows,
        total_career_rows=total_career_rows,
        orig_fighter_count=orig_fighter_count,
        new_fighters_added=new_fighters_added,
        scrape_errors_count=scrape_errors_count,
    )


if __name__ == "__main__":
    main()
