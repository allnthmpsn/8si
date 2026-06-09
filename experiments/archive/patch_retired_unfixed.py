"""
patch_retired_unfixed.py — Scrape and patch the 37 still_unfixed retired fighters
that weren't covered by the main scraper (not in 2023+ active list).

Appends results to existing ufc_stats_fights.csv, then re-runs Steps 4 & 5
to fully patch career_fights_updated.csv and sherdog_records_patched.pkl.
"""

import os, re, time, pickle
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime

DATA_DIR   = "data"
SLEEP      = 1.0
PAUSE_ON_BLOCK = 90

FIGHTS_CSV   = os.path.join(DATA_DIR, "ufc_stats_fights.csv")
SCRAPED_CSV  = os.path.join(DATA_DIR, "ufc_fighters_scraped.csv")
ERRORS_CSV   = os.path.join(DATA_DIR, "ufc_stats_scrape_errors.csv")
CAREER_OUT   = os.path.join(DATA_DIR, "career_fights_updated.csv")
SHERDOG_OUT  = os.path.join(DATA_DIR, "sherdog_records_patched.pkl")
FIGHTERS_CSV = os.path.join(DATA_DIR, "ufc_fighters_final.csv")
CAREER_CSV   = os.path.join(DATA_DIR, "career_fights.csv")
UNFIXED_CSV  = os.path.join(DATA_DIR, "still_unfixed.csv")
SHERDOG_PKL  = os.path.join(DATA_DIR, "sherdog_records_fixed.pkl")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
}
session = requests.Session()
session.headers.update(HEADERS)

def log(msg=""):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def normalize_name(s):
    return " ".join(str(s).strip().lower().split())

def _to_float(s):
    if s is None: return None
    try: return float(str(s).replace(",","").strip())
    except: return None

def safe_get(url, retries=3):
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=20)
            if r.status_code == 429:
                log(f"  429 — pausing {PAUSE_ON_BLOCK}s")
                time.sleep(PAUSE_ON_BLOCK)
                continue
            if r.status_code != 200:
                return None, f"HTTP {r.status_code}"
            return r, None
        except requests.exceptions.Timeout:
            log(f"  Timeout (attempt {attempt+1}/{retries})")
            time.sleep(PAUSE_ON_BLOCK)
        except Exception as e:
            log(f"  Error: {e}")
            time.sleep(SLEEP * 3)
    return None, "max retries exceeded"

def _parse_fighter_page(html, fighter_name):
    soup = BeautifulSoup(html, "html.parser")
    fn_norm = normalize_name(fighter_name)

    all_li_text = {}
    for li in soup.select("ul.b-list__box-list li"):
        txt = li.get_text(" ", strip=True)
        if ":" in txt:
            key, _, val = txt.partition(":")
            k = key.strip().lower()
            v = val.strip()
            if v and v != "--":
                all_li_text[k] = v

    def _li(label): return all_li_text.get(label.lower().rstrip(":"), None)
    def _to_cm(h):
        if not h: return None
        m = re.search(r"(\d+)'\s*(\d+)", str(h))
        return round(int(m.group(1))*30.48+int(m.group(2))*2.54,1) if m else None
    def _reach_cm(r):
        if not r: return None
        m = re.search(r'([\d.]+)"', str(r))
        return round(float(m.group(1))*2.54,1) if m else None
    def _pct(v):
        if not v: return None
        return float(str(v).replace("%","").strip())/100.0

    wins=losses=draws=None
    for sp in soup.select("span.b-content__title-record"):
        m = re.search(r"(\d+)-(\d+)-(\d+)", sp.get_text(strip=True))
        if m:
            wins,losses,draws = int(m.group(1)),int(m.group(2)),int(m.group(3))
            break

    stats = {
        "fighter": fighter_name,
        "height_cm": _to_cm(_li("height")),
        "reach_cm": _reach_cm(_li("reach")),
        "stance": _li("stance"),
        "dob": _li("dob"),
        "wins": wins, "losses": losses, "draws": draws,
        "slpm": _to_float(_li("slpm")),
        "str_acc": _pct(_li("str. acc.")),
        "sapm": _to_float(_li("sapm")),
        "str_def": _pct(_li("str. def")),
        "td_avg": _to_float(_li("td avg.")),
        "td_acc": _pct(_li("td acc.")),
        "td_def": _pct(_li("td def.")),
        "sub_avg": _to_float(_li("sub. avg.")),
    }

    fight_rows = []
    table = soup.find("table", class_="b-fight-details__table")
    if table:
        tbody = table.find("tbody")
        if tbody:
            for tr in tbody.find_all("tr"):
                if "b-statistics__table-row" in tr.get("class", []):
                    continue
                tds = tr.find_all("td")
                if len(tds) < 8:
                    continue

                result_raw = tds[0].get_text(strip=True).lower()
                result_map = {"win":"win","loss":"loss","draw":"draw",
                              "nc":"no contest","no contest":"no contest"}
                result = result_map.get(result_raw, result_raw)

                col1_paras = tds[1].find_all("p")
                links = [p.find("a") for p in col1_paras if p.find("a")]
                if len(links) >= 2:
                    fa = normalize_name(links[0].get_text(strip=True))
                    opp = links[1].get_text(strip=True) if fa == fn_norm else links[0].get_text(strip=True)
                elif len(links) == 1:
                    opp = links[0].get_text(strip=True)
                else:
                    all_a = tds[1].find_all("a")
                    opp = all_a[1].get_text(strip=True) if len(all_a) >= 2 else tds[1].get_text(" ", strip=True)

                col6_paras = tds[6].find_all("p")
                event_name = ""
                event_date = None
                if col6_paras:
                    ea = col6_paras[0].find("a")
                    if ea: event_name = ea.get_text(strip=True)
                if len(col6_paras) >= 2:
                    dr = col6_paras[1].get_text(strip=True)
                    try: event_date = pd.to_datetime(dr).strftime("%Y-%m-%d")
                    except: event_date = dr if dr else None

                method    = tds[7].get_text(" ", strip=True) if len(tds) > 7 else ""
                rnd       = tds[8].get_text(strip=True) if len(tds) > 8 else ""
                fight_time = tds[9].get_text(strip=True) if len(tds) > 9 else ""

                if not event_date or not opp:
                    continue

                fight_rows.append({
                    "fighter": fighter_name, "opponent": opp.strip(),
                    "date": event_date, "result": result,
                    "method": method.strip(), "event": event_name,
                    "round": rnd, "time": fight_time,
                })

    return stats, fight_rows

def _recompute_rolling(df):
    df = df.sort_values("date").reset_index(drop=True)
    df["won"] = (df["result"].str.lower() == "win").astype(int)
    df["got_finish"] = (
        (df["won"] == 1) &
        df["method"].str.contains("KO|TKO|Submission|Sub", case=False, na=False)
    ).astype(int)
    df["last5_won"]         = df["won"].shift(1).rolling(5, min_periods=1).mean()
    df["last5_finish_rate"] = df["got_finish"].shift(1).rolling(5, min_periods=1).mean()
    return df

def main():
    log("=" * 60)
    log("PATCH RETIRED UNFIXED FIGHTERS")
    log("=" * 60)

    # Find the 37 not-yet-scraped still_unfixed fighters
    unfixed_df  = pd.read_csv(UNFIXED_CSV)
    unfixed_set = set(unfixed_df["fighter"].str.strip())

    existing_fights = pd.read_csv(FIGHTS_CSV) if os.path.exists(FIGHTS_CSV) else pd.DataFrame()
    already_scraped = set(existing_fights["fighter"].unique()) if len(existing_fights) > 0 else set()

    to_scrape = [f for f in unfixed_set if f not in already_scraped]
    log(f"still_unfixed fighters not yet scraped: {len(to_scrape)}")

    # Get URLs from ufc_fighters_final
    fighters_df = pd.read_csv(FIGHTERS_CSV)
    name_to_url = {str(r["Fighter_Name"]).strip(): str(r["Fighter_URL"]).strip()
                   for _, r in fighters_df.iterrows()}

    new_fights = []
    new_stats  = []
    new_errors = []
    patched    = 0

    for i, name in enumerate(to_scrape):
        url = name_to_url.get(name, "")
        if not url or url.lower() == "nan":
            log(f"  [{i+1}/{len(to_scrape)}] {name} — no URL found")
            new_errors.append({"fighter": name, "url": "", "error": "no URL in ufc_fighters_final"})
            continue

        time.sleep(SLEEP)
        resp, err = safe_get(url)
        if err or resp is None:
            log(f"  [{i+1}/{len(to_scrape)}] {name} — error: {err}")
            new_errors.append({"fighter": name, "url": url, "error": err or "no response"})
            continue

        try:
            stats, fights = _parse_fighter_page(resp.text, name)
        except Exception as e:
            log(f"  [{i+1}/{len(to_scrape)}] {name} — parse error: {e}")
            new_errors.append({"fighter": name, "url": url, "error": str(e)})
            continue

        new_fights.extend(fights)
        new_stats.append(stats)
        patched += 1
        log(f"  [{i+1}/{len(to_scrape)}] {name} — {len(fights)} fights")

    log(f"\nScraped {patched}/{len(to_scrape)} retired fighters")

    if not new_fights:
        log("No new fight data — nothing to patch")
        return

    # Append to ufc_stats_fights.csv
    all_fights = pd.concat([existing_fights, pd.DataFrame(new_fights)], ignore_index=True)
    all_fights.to_csv(FIGHTS_CSV, index=False)
    log(f"Updated {FIGHTS_CSV}: {len(all_fights)} total rows")

    # Append to ufc_fighters_scraped.csv
    if new_stats:
        existing_scraped = pd.read_csv(SCRAPED_CSV) if os.path.exists(SCRAPED_CSV) else pd.DataFrame()
        all_scraped = pd.concat([existing_scraped, pd.DataFrame(new_stats)], ignore_index=True)
        all_scraped.to_csv(SCRAPED_CSV, index=False)

    # Append errors
    if new_errors:
        existing_errors = pd.read_csv(ERRORS_CSV) if os.path.exists(ERRORS_CSV) else pd.DataFrame()
        all_errors = pd.concat([existing_errors, pd.DataFrame(new_errors)], ignore_index=True)
        all_errors.to_csv(ERRORS_CSV, index=False)

    # ── Re-run Step 4: patch career_fights_updated ───────────────────────────
    log()
    log("Re-patching career_fights_updated.csv...")

    career  = pd.read_csv(CAREER_CSV)
    career["date"] = pd.to_datetime(career["date"])
    ufc_fights = pd.read_csv(FIGHTS_CSV)
    ufc_fights["date"] = pd.to_datetime(ufc_fights["date"], errors="coerce")
    ufc_fights = ufc_fights.dropna(subset=["date"])

    # Load the already-patched career (from the main run)
    # We rebuild from the original to ensure consistency
    all_updated = []
    fighters_in_ufc = set(ufc_fights["fighter"].unique())
    new_rows_added = 0
    total_patched  = set()
    unfixed_patched = 0

    career_groups = career.groupby("fighter")

    for fighter, ufc_grp in ufc_fights.groupby("fighter"):
        ufc_grp = ufc_grp.copy().sort_values("date").reset_index(drop=True)

        ufc_records = []
        for _, fr in ufc_grp.iterrows():
            ufc_records.append({
                "fighter": fighter,
                "opponent": fr["opponent"],
                "date": fr["date"],
                "result": str(fr["result"]).lower().strip(),
                "method": str(fr["method"]).strip(),
                "won": 1 if str(fr["result"]).lower().strip() == "win" else 0,
                "got_finish": 0,
                "last5_won": np.nan,
                "last5_finish_rate": np.nan,
            })

        if fighter in unfixed_set:
            new_df = pd.DataFrame(ufc_records)
            new_df = _recompute_rolling(new_df)
            all_updated.append(new_df)
            unfixed_patched += 1
            total_patched.add(fighter)
            continue

        existing = career[career["fighter"] == fighter].copy() if fighter in career_groups.groups else pd.DataFrame(columns=career.columns)
        existing["date"] = pd.to_datetime(existing["date"])

        def already_exists(opp, dt):
            if len(existing) == 0: return False
            opp_norm = normalize_name(opp)
            for _, er in existing.iterrows():
                if normalize_name(str(er["opponent"])) == opp_norm:
                    if abs((er["date"] - dt).days) <= 7:
                        return True
            return False

        missing = [r for r in ufc_records if not already_exists(r["opponent"], r["date"])]
        if missing:
            new_rows_added += len(missing)
            total_patched.add(fighter)
            combined = pd.concat([existing, pd.DataFrame(missing)], ignore_index=True)
            combined = _recompute_rolling(combined)
            all_updated.append(combined)
        else:
            all_updated.append(existing)

    # Keep untouched fighters
    untouched = career[~career["fighter"].isin(fighters_in_ufc)].copy()
    all_updated.append(untouched)

    updated = pd.concat(all_updated, ignore_index=True)
    updated["date"] = pd.to_datetime(updated["date"])
    updated = updated.sort_values(["fighter","date"]).reset_index(drop=True)
    updated = updated[["fighter","opponent","date","result","method",
                        "won","got_finish","last5_won","last5_finish_rate"]]
    updated.to_csv(CAREER_OUT, index=False)

    log(f"Saved {CAREER_OUT}")
    log(f"  New rows added:  {new_rows_added}")
    log(f"  Total rows:      {len(updated)}")
    log(f"  still_unfixed patched: {unfixed_patched}/66")

    # ── Re-run Step 5: patch sherdog_records_patched ─────────────────────────
    log()
    log("Re-patching sherdog_records_patched.pkl...")

    # Load from the existing patched pkl (continue from previous work)
    sherdog_src = SHERDOG_OUT if os.path.exists(SHERDOG_OUT) else "data/sherdog_records_fixed.pkl"
    with open(sherdog_src, "rb") as f:
        sherdog = pickle.load(f)

    sherdog_patched = 0
    for fighter in unfixed_set:
        grp = ufc_fights[ufc_fights["fighter"] == fighter]
        if len(grp) == 0:
            continue
        fights_list = []
        for _, fr in grp.sort_values("date").iterrows():
            fights_list.append({
                "result": str(fr["result"]).lower().strip(),
                "opponent": str(fr["opponent"]).strip(),
                "date": fr["date"],
                "method": str(fr["method"]).strip(),
                "event": str(fr.get("event","")).strip(),
            })
        sherdog[fighter] = {"fights": fights_list}
        sherdog_patched += 1

    import os as _os
    with open(SHERDOG_OUT, "wb") as f:
        pickle.dump(sherdog, f)
    log(f"Saved {SHERDOG_OUT}  ({sherdog_patched}/66 still_unfixed patched)")

    # ── Final spot checks ─────────────────────────────────────────────────────
    log()
    log("Spot checks on patched retired fighters:")
    career_updated = pd.read_csv(CAREER_OUT)
    career_updated["date"] = pd.to_datetime(career_updated["date"])

    for name in ["Georges St-Pierre", "Aleksei Oleinik", "TJ Dillashaw"]:
        grp = career_updated[career_updated["fighter"] == name].sort_values("date")
        if len(grp) == 0:
            log(f"  {name}: NOT FOUND")
            continue
        wins   = int(grp["won"].sum())
        losses = len(grp) - wins
        last   = grp.iloc[-1]
        log(f"  {name}: {len(grp)} fights ({wins}W-{losses}L), last: {str(last['date'])[:10]} vs {last['opponent']}")

    log()
    log("=" * 40)
    log("PATCH SUMMARY")
    log("=" * 40)
    log(f"Retired fighters scraped:    {patched}/{len(to_scrape)}")
    log(f"Total ufc_stats_fights rows: {len(all_fights)}")
    log(f"career_fights_updated rows:  {len(updated)}")
    log(f"still_unfixed patched:       {unfixed_patched}/66")
    log("=" * 40)


if __name__ == "__main__":
    main()
