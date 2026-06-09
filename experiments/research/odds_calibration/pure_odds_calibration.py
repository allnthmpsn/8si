"""
Pure historical odds calibration — no model involvement.
Research only. No model, frontend, or backend files are modified.

Uses ufc-master.csv directly: R_odds / B_odds are American moneyline closing lines.
All 6,929 fights with both odds columns and a valid winner (Red/Blue) are used.
"""

import os, json, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, '../../../data')
OUT  = BASE

# ─────────────────────────────────────────────────────────
# SETUP — Load and build fight-level dataset
# ─────────────────────────────────────────────────────────
print("=" * 64)
print("SETUP: Load ufc-master.csv")
print("=" * 64)

master = pd.read_csv(os.path.join(DATA, 'ufc-master.csv'), low_memory=False)
master['date'] = pd.to_datetime(master['date'])

# Valid fights: known winner + both ML odds present
df = master[
    master['Winner'].isin(['Red', 'Blue']) &
    master['R_odds'].notna() &
    master['B_odds'].notna()
].copy()

df['r_ml'] = df['R_odds'].astype(float)
df['b_ml'] = df['B_odds'].astype(float)
df['red_won'] = (df['Winner'] == 'Red').astype(int)

print(f"  Total fights in master:         {len(master):,}")
print(f"  Fights with valid winner+odds:  {len(df):,}")
print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"  Red won: {df['red_won'].sum():,} / {len(df):,}  ({df['red_won'].mean():.3f})")

def to_implied(ml):
    """Raw (vig-inclusive) implied probability from American ML."""
    if pd.isna(ml) or ml == 0:
        return np.nan
    if ml < 0:
        return abs(ml) / (abs(ml) + 100)
    return 100 / (ml + 100)

df['r_implied'] = df['r_ml'].apply(to_implied)
df['b_implied'] = df['b_ml'].apply(to_implied)
df['vig']       = (df['r_implied'] + df['b_implied'] - 1.0)

# ─────────────────────────────────────────────────────────
# STEP 1 — Identify favorite / underdog per fight
# ─────────────────────────────────────────────────────────
print()
print("=" * 64)
print("STEP 1: Identify favorite and underdog per fight")
print("=" * 64)

df['fav_is_red'] = df['r_implied'] > df['b_implied']

df['fav_ml']  = np.where(df['fav_is_red'], df['r_ml'],  df['b_ml'])
df['dog_ml']  = np.where(df['fav_is_red'], df['b_ml'],  df['r_ml'])
df['fav_impl']= np.where(df['fav_is_red'], df['r_implied'], df['b_implied'])
df['dog_impl']= np.where(df['fav_is_red'], df['b_implied'], df['r_implied'])

df['fav_won'] = np.where(df['fav_is_red'], df['red_won'], 1 - df['red_won'])
df['dog_won'] = 1 - df['fav_won']

# No-vig probabilities
df['novig_total'] = df['fav_impl'] + df['dog_impl']
df['fav_novig']   = df['fav_impl'] / df['novig_total']
df['dog_novig']   = df['dog_impl'] / df['novig_total']

fav_wr = df['fav_won'].mean()
dog_wr = df['dog_won'].mean()

print(f"  Total fights:         {len(df):,}")
print(f"  Avg vig:              {df['vig'].mean():.4f} ({df['vig'].mean()*100:.2f}%)")
print(f"  Favorite win rate:    {fav_wr:.4f}  ({fav_wr*100:.2f}%)")
print(f"  Underdog win rate:    {dog_wr:.4f}  ({dog_wr*100:.2f}%)")
print(f"  Fav avg ML:           {df['fav_ml'].mean():.1f}")
print(f"  Dog avg ML:           {df['dog_ml'].mean():.1f}")
print(f"  Avg fav novig:        {df['fav_novig'].mean():.4f}")
print(f"  True-pick'em fights (fav_novig < 52%): {(df['fav_novig'] < 0.52).sum():,}")

# ─────────────────────────────────────────────────────────
# STEP 2 — Bucket analysis (every 50 ML points)
# ─────────────────────────────────────────────────────────
print()
print("=" * 64)
print("STEP 2: Bucket analysis — FAVORITES (fav ML ≤ -101)")
print("=" * 64)

def get_bucket(ml, step=50):
    return int(round(ml / step) * step)

fav_df = df[df['fav_ml'] < 0].copy()
fav_df['fav_bucket'] = fav_df['fav_ml'].apply(get_bucket)

fav_stats = fav_df.groupby('fav_bucket').agg(
    N=('fav_won', 'count'),
    actual_wr=('fav_won', 'mean'),
    avg_ml=('fav_ml', 'mean'),
).reset_index()

fav_stats['implied_prob'] = fav_stats['fav_bucket'].apply(to_implied)
# no-vig: approximate using avg vig from full dataset
avg_vig = df['vig'].mean()
fav_stats['novig_approx'] = fav_stats['implied_prob'] / (1 + avg_vig / 2)  # rough
# Use actual per-bucket no-vig instead
bucket_novig = fav_df.groupby('fav_bucket')['fav_novig'].mean().reset_index().rename(
    columns={'fav_novig': 'novig_avg'})
fav_stats = fav_stats.merge(bucket_novig, on='fav_bucket', how='left')

fav_stats['gap_vs_novig'] = fav_stats['actual_wr'] - fav_stats['novig_avg']
fav_stats['gap_vs_implied'] = fav_stats['actual_wr'] - fav_stats['implied_prob']
fav_stats['roi_pct'] = fav_stats.apply(
    lambda r: (r['actual_wr'] * (100 / abs(r['fav_bucket'])) - (1 - r['actual_wr'])) * 100
    if r['fav_bucket'] < 0 else np.nan, axis=1
)
fav_stats['reliable'] = fav_stats['N'] >= 15

rel_fav = fav_stats[fav_stats['reliable']].sort_values('fav_bucket')

print(f"\n  {'Bucket':>7} {'N':>5} {'AvgML':>7} {'ActWR':>7} {'NoVig':>7} {'Implied':>8} {'GapNV':>7} {'ROI%':>7}")
print(f"  {'-'*7} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*7} {'-'*7}")
for _, r in rel_fav.iterrows():
    print(f"  {r['fav_bucket']:>7.0f} {r['N']:>5.0f} {r['avg_ml']:>7.1f} {r['actual_wr']:>7.3f} "
          f"{r['novig_avg']:>7.3f} {r['implied_prob']:>8.3f} {r['gap_vs_novig']:>+7.3f} {r['roi_pct']:>7.2f}")

print()
print("=" * 64)
print("STEP 2: Bucket analysis — UNDERDOGS (dog ML ≥ +101)")
print("=" * 64)

dog_df = df[df['dog_ml'] > 0].copy()
dog_df['dog_bucket'] = dog_df['dog_ml'].apply(get_bucket)

dog_stats = dog_df.groupby('dog_bucket').agg(
    N=('dog_won', 'count'),
    actual_wr=('dog_won', 'mean'),
    avg_ml=('dog_ml', 'mean'),
).reset_index()

dog_stats['implied_prob'] = dog_stats['dog_bucket'].apply(to_implied)
bucket_dog_novig = dog_df.groupby('dog_bucket')['dog_novig'].mean().reset_index().rename(
    columns={'dog_novig': 'novig_avg'})
dog_stats = dog_stats.merge(bucket_dog_novig, on='dog_bucket', how='left')

dog_stats['gap_vs_novig'] = dog_stats['actual_wr'] - dog_stats['novig_avg']
dog_stats['gap_vs_implied'] = dog_stats['actual_wr'] - dog_stats['implied_prob']
dog_stats['roi_pct'] = dog_stats.apply(
    lambda r: (r['actual_wr'] * (r['dog_bucket'] / 100) - (1 - r['actual_wr'])) * 100
    if r['dog_bucket'] > 0 else np.nan, axis=1
)
dog_stats['reliable'] = dog_stats['N'] >= 15

rel_dog = dog_stats[dog_stats['reliable']].sort_values('dog_bucket')

print(f"\n  {'Bucket':>7} {'N':>5} {'AvgML':>7} {'ActWR':>7} {'NoVig':>7} {'Implied':>8} {'GapNV':>7} {'ROI%':>7}")
print(f"  {'-'*7} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*7} {'-'*7}")
for _, r in rel_dog.iterrows():
    print(f"  {r['dog_bucket']:>7.0f} {r['N']:>5.0f} {r['avg_ml']:>7.1f} {r['actual_wr']:>7.3f} "
          f"{r['novig_avg']:>7.3f} {r['implied_prob']:>8.3f} {r['gap_vs_novig']:>+7.3f} {r['roi_pct']:>7.2f}")

# ─────────────────────────────────────────────────────────
# STEP 3 — Broader category summary
# ─────────────────────────────────────────────────────────
print()
print("=" * 64)
print("STEP 3: Category summary")
print("=" * 64)

fav_categories = {
    'Heavy Fav (≤-600)':          fav_df['fav_ml'] <= -600,
    'Big Fav (-400 to -600)':    (fav_df['fav_ml'] > -600) & (fav_df['fav_ml'] <= -400),
    'Mod Fav (-250 to -400)':    (fav_df['fav_ml'] > -400) & (fav_df['fav_ml'] <= -250),
    'Slight Fav (-150 to -250)': (fav_df['fav_ml'] > -250) & (fav_df['fav_ml'] <= -150),
    'Small Fav (-110 to -150)':  (fav_df['fav_ml'] > -150) & (fav_df['fav_ml'] <= -110),
}

dog_categories = {
    'Small Dog (+110 to +150)':  (dog_df['dog_ml'] >= 110) & (dog_df['dog_ml'] < 150),
    'Slight Dog (+150 to +250)': (dog_df['dog_ml'] >= 150) & (dog_df['dog_ml'] < 250),
    'Mod Dog (+250 to +400)':    (dog_df['dog_ml'] >= 250) & (dog_df['dog_ml'] < 400),
    'Big Dog (+400 to +600)':    (dog_df['dog_ml'] >= 400) & (dog_df['dog_ml'] < 600),
    'Heavy Dog (>+600)':          dog_df['dog_ml'] >= 600,
}

cat_results = []

print(f"\n  {'Category':<32} {'N':>5} {'AvgML':>7} {'ActWR':>7} {'NoVig':>7} {'GapNV':>7} {'ROI%':>8}")
print(f"  {'-'*32} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")

for label, mask in fav_categories.items():
    sub = fav_df[mask]
    n = len(sub)
    if n < 15:
        print(f"  {label:<32} {n:>5}  (unreliable)")
        continue
    wr = sub['fav_won'].mean()
    avg_ml = sub['fav_ml'].mean()
    nv = sub['fav_novig'].mean()
    gap = wr - nv
    roi = (wr * (100 / abs(avg_ml)) - (1 - wr)) * 100
    print(f"  {label:<32} {n:>5} {avg_ml:>7.1f} {wr:>7.3f} {nv:>7.3f} {gap:>+7.3f} {roi:>8.2f}")
    cat_results.append({'category': label, 'side': 'fav', 'n': n, 'avg_ml': round(avg_ml,1),
                        'actual_wr': round(wr,4), 'novig': round(nv,4), 'gap': round(gap,4), 'roi_pct': round(roi,2)})

print()

for label, mask in dog_categories.items():
    sub = dog_df[mask]
    n = len(sub)
    if n < 15:
        print(f"  {label:<32} {n:>5}  (unreliable)")
        continue
    wr = sub['dog_won'].mean()
    avg_ml = sub['dog_ml'].mean()
    nv = sub['dog_novig'].mean()
    gap = wr - nv
    roi = (wr * (avg_ml / 100) - (1 - wr)) * 100
    print(f"  {label:<32} {n:>5} {avg_ml:>7.1f} {wr:>7.3f} {nv:>7.3f} {gap:>+7.3f} {roi:>8.2f}")
    cat_results.append({'category': label, 'side': 'dog', 'n': n, 'avg_ml': round(avg_ml,1),
                        'actual_wr': round(wr,4), 'novig': round(nv,4), 'gap': round(gap,4), 'roi_pct': round(roi,2)})

cat_df = pd.DataFrame(cat_results)

# ─────────────────────────────────────────────────────────
# STEP 4 — By weight class and by year
# ─────────────────────────────────────────────────────────
print()
print("=" * 64)
print("STEP 4A: Favorite win rate by weight class")
print("=" * 64)

wc_results = []
wc_rows = []
for wc in sorted(df['weight_class'].unique()):
    sub = df[df['weight_class'] == wc]
    if len(sub) < 30:
        continue
    fav_wr_wc = sub['fav_won'].mean()
    dog_wr_wc = sub['dog_won'].mean()
    nv_wc = sub['fav_novig'].mean()
    gap_wc = fav_wr_wc - nv_wc
    wc_rows.append((wc, len(sub), fav_wr_wc, nv_wc, gap_wc, dog_wr_wc))
    wc_results.append({'weight_class': wc, 'n': len(sub), 'fav_wr': round(fav_wr_wc,4),
                        'fav_novig': round(nv_wc,4), 'gap': round(gap_wc,4), 'dog_wr': round(dog_wr_wc,4)})

wc_rows.sort(key=lambda x: x[2], reverse=True)
print(f"\n  {'Weight Class':<30} {'N':>5} {'FavWR':>7} {'FavNoVig':>9} {'Gap':>7} {'DogWR':>7}")
print(f"  {'-'*30} {'-'*5} {'-'*7} {'-'*9} {'-'*7} {'-'*7}")
for row in wc_rows:
    wc, n, fwr, nv, gap, dwr = row
    print(f"  {wc:<30} {n:>5} {fwr:>7.3f} {nv:>9.3f} {gap:>+7.3f} {dwr:>7.3f}")

print()
print("=" * 64)
print("STEP 4B: Favorite win rate by year")
print("=" * 64)

df['year'] = df['date'].dt.year
yr_results = []
print(f"\n  {'Year':>5} {'N':>5} {'FavWR':>7} {'FavNoVig':>9} {'Gap':>7} {'DogWR':>7} {'AvgVig%':>9}")
print(f"  {'-'*5} {'-'*5} {'-'*7} {'-'*9} {'-'*7} {'-'*7} {'-'*9}")
for yr in sorted(df['year'].unique()):
    sub = df[df['year'] == yr]
    if len(sub) < 30:
        continue
    fwr = sub['fav_won'].mean()
    dwr = sub['dog_won'].mean()
    nv = sub['fav_novig'].mean()
    gap = fwr - nv
    avg_vig_yr = sub['vig'].mean() * 100
    print(f"  {yr:>5} {len(sub):>5} {fwr:>7.3f} {nv:>9.3f} {gap:>+7.3f} {dwr:>7.3f} {avg_vig_yr:>9.2f}")
    yr_results.append({'year': int(yr), 'n': len(sub), 'fav_wr': round(fwr,4),
                        'fav_novig': round(nv,4), 'gap': round(gap,4), 'dog_wr': round(dwr,4),
                        'avg_vig_pct': round(avg_vig_yr,2)})

# Trend check
yr_df = pd.DataFrame(yr_results)
if len(yr_df) >= 5:
    from scipy import stats as scipy_stats
    slope, intercept, r, p, _ = scipy_stats.linregress(yr_df['year'], yr_df['fav_wr'])
    print(f"\n  Trend (favorite WR vs year): slope={slope:+.4f}/yr  R²={r**2:.3f}  p={p:.3f}")
    if p < 0.05:
        direction = 'INCREASING' if slope > 0 else 'DECREASING'
        print(f"  → Statistically significant {direction} trend (p<0.05)")
    else:
        print(f"  → No statistically significant trend (p={p:.3f})")

# ─────────────────────────────────────────────────────────
# STEP 5 — Specific questions answered
# ─────────────────────────────────────────────────────────
print()
print("=" * 64)
print("STEP 5: Specific questions answered")
print("=" * 64)

# Q1: Where is the market well-calibrated (gap near zero)?
print("\nQ1: At what odds is the market well-calibrated (|gap| < 2pp vs novig)?")
rel_fav_reset = rel_fav.reset_index(drop=True)
rel_dog_reset = rel_dog.reset_index(drop=True)
well_cal_fav = rel_fav_reset[rel_fav_reset['gap_vs_novig'].abs() < 0.02]
well_cal_dog = rel_dog_reset[rel_dog_reset['gap_vs_novig'].abs() < 0.02]
if len(well_cal_fav):
    print(f"  Favorite buckets: {sorted(well_cal_fav['fav_bucket'].tolist())}")
else:
    print("  No favorite bucket within 2pp of novig")
if len(well_cal_dog):
    print(f"  Underdog buckets: {sorted(well_cal_dog['dog_bucket'].tolist())}")
else:
    print("  No underdog bucket within 2pp of novig")

# Q2: Underdogs consistently outperforming?
print("\nQ2: Do underdogs consistently outperform their implied probability?")
dog_over = rel_dog_reset[rel_dog_reset['gap_vs_novig'] > 0.02]
dog_under = rel_dog_reset[rel_dog_reset['gap_vs_novig'] < -0.02]
print(f"  Dog buckets overperforming (>+2pp):  {len(dog_over)} / {len(rel_dog_reset)}")
print(f"  Dog buckets underperforming (< -2pp): {len(dog_under)} / {len(rel_dog_reset)}")
if len(dog_over):
    rng = f"{rel_dog_reset[rel_dog_reset['gap_vs_novig']>0.02]['dog_bucket'].min():.0f} to {rel_dog_reset[rel_dog_reset['gap_vs_novig']>0.02]['dog_bucket'].max():.0f}"
    print(f"  Overperforming range: +{rng}")
if len(dog_under):
    rng2 = f"{rel_dog_reset[rel_dog_reset['gap_vs_novig']<-0.02]['dog_bucket'].min():.0f} to {rel_dog_reset[rel_dog_reset['gap_vs_novig']<-0.02]['dog_bucket'].max():.0f}"
    print(f"  Underperforming range: +{rng2}")

# Q3: Favorites consistently underperforming?
print("\nQ3: Do favorites consistently underperform their implied probability?")
fav_over = rel_fav_reset[rel_fav_reset['gap_vs_novig'] > 0.02]
fav_under = rel_fav_reset[rel_fav_reset['gap_vs_novig'] < -0.02]
print(f"  Fav buckets overperforming (>+2pp):   {len(fav_over)} / {len(rel_fav_reset)}")
print(f"  Fav buckets underperforming (< -2pp): {len(fav_under)} / {len(rel_fav_reset)}")
if len(fav_under):
    ml_range = f"{rel_fav_reset[rel_fav_reset['gap_vs_novig']<-0.02]['fav_bucket'].min():.0f} to {rel_fav_reset[rel_fav_reset['gap_vs_novig']<-0.02]['fav_bucket'].max():.0f}"
    print(f"  Underperforming ML range: {ml_range}")

# Q4: Weight class with highest/lowest fav win rate
print("\nQ4: Favorite win rate by weight class (range):")
wc_df2 = pd.DataFrame(wc_results)
if len(wc_df2):
    best = wc_df2.loc[wc_df2['fav_wr'].idxmax()]
    worst = wc_df2.loc[wc_df2['fav_wr'].idxmin()]
    print(f"  Highest fav WR: {best['weight_class']} ({best['fav_wr']:.3f}, N={best['n']})")
    print(f"  Lowest  fav WR: {worst['weight_class']} ({worst['fav_wr']:.3f}, N={worst['n']})")
    print(f"  Range: {wc_df2['fav_wr'].min():.3f} – {wc_df2['fav_wr'].max():.3f}")

# Q5: Trend over time
print("\nQ5: Is MMA becoming more predictable over time?")
if 'slope' in dir():
    if p < 0.05:
        print(f"  Slope = {slope:+.4f}/yr (significant at p={p:.3f})")
        print(f"  {'Yes — favorites are winning more often' if slope > 0 else 'No — if anything less predictable'}")
    else:
        print(f"  Slope = {slope:+.4f}/yr — NOT statistically significant (p={p:.3f})")
        print(f"  No clear trend. MMA predictability is roughly stable.")

# Q6: Overall favorite win rate
print(f"\nQ6: Overall favorite win rate: {df['fav_won'].mean():.4f} ({df['fav_won'].mean()*100:.2f}%)")
print(f"    Overall underdog win rate:  {df['dog_won'].mean():.4f} ({df['dog_won'].mean()*100:.2f}%)")
print(f"    Overall fav novig avg:      {df['fav_novig'].mean():.4f}")
print(f"    Overall fav gap vs novig:   {(df['fav_won'].mean() - df['fav_novig'].mean()):+.4f}")

# ─────────────────────────────────────────────────────────
# STEP 5 — Save results
# ─────────────────────────────────────────────────────────
print()
print("=" * 64)
print("Saving results")
print("=" * 64)

results = {
    "meta": {
        "total_fights": int(len(df)),
        "date_range": [str(df['date'].min().date()), str(df['date'].max().date())],
        "overall_fav_wr": round(float(df['fav_won'].mean()), 4),
        "overall_dog_wr": round(float(df['dog_won'].mean()), 4),
        "avg_vig": round(float(df['vig'].mean()), 4),
        "avg_fav_novig": round(float(df['fav_novig'].mean()), 4),
        "fav_gap_vs_novig": round(float(df['fav_won'].mean() - df['fav_novig'].mean()), 4),
    },
    "favorite_buckets": rel_fav[['fav_bucket','N','actual_wr','novig_avg','gap_vs_novig','roi_pct']].rename(
        columns={'fav_bucket':'bucket','actual_wr':'win_rate','novig_avg':'novig','gap_vs_novig':'gap'}).to_dict(orient='records'),
    "underdog_buckets": rel_dog[['dog_bucket','N','actual_wr','novig_avg','gap_vs_novig','roi_pct']].rename(
        columns={'dog_bucket':'bucket','actual_wr':'win_rate','novig_avg':'novig','gap_vs_novig':'gap'}).to_dict(orient='records'),
    "categories": cat_df.to_dict(orient='records') if len(cat_df) else [],
    "by_weight_class": wc_results,
    "by_year": yr_results,
}

json_path = os.path.join(OUT, 'pure_odds_calibration_results.json')
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"  Saved: {json_path}")

# ── Markdown findings ──────────────────────────────────────
fav_novig_overall = df['fav_novig'].mean()
fav_gap_overall = df['fav_won'].mean() - fav_novig_overall
well_cal_q = "No bucket within 2pp of novig (heavily non-linear)"
if len(well_cal_fav) or len(well_cal_dog):
    parts = []
    if len(well_cal_fav): parts.append(f"Favorites: {sorted(well_cal_fav['fav_bucket'].tolist())}")
    if len(well_cal_dog): parts.append(f"Underdogs: {sorted(well_cal_dog['dog_bucket'].tolist())}")
    well_cal_q = "; ".join(parts)

md_lines = [
    "# Pure Odds Calibration — Findings",
    "",
    f"**Source:** ufc-master.csv (R_odds / B_odds — closing moneylines)  ",
    f"**Fights:** {len(df):,} with valid winner + both ML odds  ",
    f"**Date range:** {df['date'].min().date()} to {df['date'].max().date()}  ",
    f"**Avg vig:** {df['vig'].mean()*100:.2f}%  ",
    "",
    "---",
    "",
    "## Overall",
    "",
    f"| Metric | Value |",
    f"|---|---|",
    f"| Favorite win rate | {df['fav_won'].mean():.3f} ({df['fav_won'].mean()*100:.1f}%) |",
    f"| Underdog win rate | {df['dog_won'].mean():.3f} ({df['dog_won'].mean()*100:.1f}%) |",
    f"| Avg fav no-vig probability | {df['fav_novig'].mean():.3f} ({df['fav_novig'].mean()*100:.1f}%) |",
    f"| Overall fav gap vs no-vig | {fav_gap_overall:+.4f} ({fav_gap_overall*100:+.1f}pp) |",
    "",
    "---",
    "",
    "## Favorite Bucket Analysis (N≥15)",
    "",
    "| Bucket | N | WinRate | NoVig | Gap vs NoVig | ROI% |",
    "|---|---|---|---|---|---|",
]
for _, r in rel_fav.iterrows():
    md_lines.append(f"| {r['fav_bucket']:.0f} | {r['N']:.0f} | {r['actual_wr']:.3f} | {r['novig_avg']:.3f} | {r['gap_vs_novig']:+.3f} | {r['roi_pct']:.2f}% |")

md_lines += [
    "",
    "## Underdog Bucket Analysis (N≥15)",
    "",
    "| Bucket | N | WinRate | NoVig | Gap vs NoVig | ROI% |",
    "|---|---|---|---|---|---|",
]
for _, r in rel_dog.iterrows():
    md_lines.append(f"| +{r['dog_bucket']:.0f} | {r['N']:.0f} | {r['actual_wr']:.3f} | {r['novig_avg']:.3f} | {r['gap_vs_novig']:+.3f} | {r['roi_pct']:.2f}% |")

md_lines += [
    "",
    "## Category Summary",
    "",
    "| Category | N | AvgML | WinRate | NoVig | Gap | ROI% |",
    "|---|---|---|---|---|---|---|",
]
for row in cat_results:
    md_lines.append(f"| {row['category']} | {row['n']} | {row['avg_ml']:.0f} | {row['actual_wr']:.3f} | {row['novig']:.3f} | {row['gap']:+.3f} | {row['roi_pct']:.2f}% |")

md_lines += [
    "",
    "## By Weight Class (sorted by fav WR, N≥30)",
    "",
    "| Weight Class | N | Fav WR | Fav NoVig | Gap | Dog WR |",
    "|---|---|---|---|---|---|",
]
for row in sorted(wc_results, key=lambda x: x['fav_wr'], reverse=True):
    md_lines.append(f"| {row['weight_class']} | {row['n']} | {row['fav_wr']:.3f} | {row['fav_novig']:.3f} | {row['gap']:+.3f} | {row['dog_wr']:.3f} |")

md_lines += [
    "",
    "## By Year",
    "",
    "| Year | N | Fav WR | Fav NoVig | Gap | Dog WR | Avg Vig% |",
    "|---|---|---|---|---|---|---|",
]
for row in yr_results:
    md_lines.append(f"| {row['year']} | {row['n']} | {row['fav_wr']:.3f} | {row['fav_novig']:.3f} | {row['gap']:+.3f} | {row['dog_wr']:.3f} | {row['avg_vig_pct']:.2f}% |")

md_lines += [
    "",
    "---",
    "",
    "## Questions Answered",
    "",
    f"**Q1 — Where is the market well-calibrated?** {well_cal_q}",
    "",
    f"**Q2 — Do underdogs consistently outperform implied?** "
    f"{'Yes' if len(dog_over) > len(dog_under) else 'No'}: "
    f"{len(dog_over)}/{len(rel_dog_reset)} reliable buckets overperform by >2pp.",
    "",
    f"**Q3 — Do favorites consistently underperform?** "
    f"{'Yes' if len(fav_under) > len(fav_over) else 'No / mixed'}: "
    f"{len(fav_under)}/{len(rel_fav_reset)} reliable buckets underperform by >2pp.",
    "",
    f"**Q4 — Weight class variation:** Range {wc_df2['fav_wr'].min():.3f}–{wc_df2['fav_wr'].max():.3f} "
    f"(highest: {best['weight_class']} {best['fav_wr']:.3f}, lowest: {worst['weight_class']} {worst['fav_wr']:.3f})",
    "",
    "**Q5 — Is MMA becoming more predictable?** "
    + (f"Slope = {slope:+.4f}/yr, p={p:.3f} — "
       + ("Yes, statistically significant." if p < 0.05 else "No significant trend.")),
    "",
    f"**Q6 — Overall favorite win rate:** {df['fav_won'].mean():.4f} ({df['fav_won'].mean()*100:.2f}%)",
    "",
    "---",
    "",
    "_Research only — no model, frontend, or backend files were modified._",
]

md_path = os.path.join(OUT, 'PURE_ODDS_CALIBRATION_FINDINGS.md')
with open(md_path, 'w') as f:
    f.write('\n'.join(md_lines) + '\n')
print(f"  Saved: {md_path}")
print("\nAll steps complete.")
