"""
Empirical odds calibration analysis.
Research only — no model, frontend, or backend files are modified.

Steps:
  1. Load value_bet_log.csv + ufc-master.csv; join for two-sided odds
  2. Build full fight-level odds dataset (implied prob, no-vig)
  3. Bucket analysis (-1500 to +1000, every 50 ML points, N>=15 reliable)
  4. Smoothed calibration curve (3-bucket rolling average)
  5. By-weight-class breakdown
  6. Favorite vs underdog summary (7 categories)
  7. Fit calibration curve (linear + power)
  8. Feature recommendation (odds_calibration_adjustment for Model 2B)
  9. Save calibration_results.json + CALIBRATION_FINDINGS.md
"""

import os, json, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

BASE  = os.path.dirname(os.path.abspath(__file__))
DATA  = os.path.join(BASE, '../../../data')
OUT   = BASE

LOG_PATH    = os.path.join(DATA, 'value_bet_log.csv')
MASTER_PATH = os.path.join(DATA, 'ufc-master.csv')

def ml_to_impl(ml):
    """Raw (vig-inclusive) implied probability from American odds."""
    if ml == 0:
        return 0.5
    if ml < 0:
        return abs(ml) / (abs(ml) + 100)
    return 100 / (ml + 100)

def novig_probs(ml_r, ml_b):
    """Return (p_r_novig, p_b_novig) removing the vig."""
    p_r = ml_to_impl(ml_r)
    p_b = ml_to_impl(ml_b)
    total = p_r + p_b
    if total == 0:
        return 0.5, 0.5
    return p_r / total, p_b / total

# ─────────────────────────────────────────────────────────
# STEP 1 — Load data and join for two-sided odds
# ─────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Load and join data")
print("=" * 60)

log = pd.read_csv(LOG_PATH)
master = pd.read_csv(MASTER_PATH, low_memory=False)

print(f"  value_bet_log:  {log.shape[0]:,} rows, {log.shape[1]} cols")
print(f"  ufc-master:     {master.shape[0]:,} rows, {master.shape[1]} cols")

# Normalize dates
log['date'] = pd.to_datetime(log['date'], errors='coerce')
master['date'] = pd.to_datetime(master['date'], errors='coerce')

# Keep only needed master cols
master_slim = master[['date','R_fighter','B_fighter','R_odds','B_odds','weight_class']].dropna(
    subset=['R_odds','B_odds']
)
print(f"  master rows with both odds: {master_slim.shape[0]:,}")

# Join orientation A: f1 = Red corner
join_r = log.merge(
    master_slim.rename(columns={'R_fighter':'f1_name','B_fighter':'f2_name',
                                 'R_odds':'f1_master_odds','B_odds':'f2_master_odds'}),
    on=['date','f1_name','f2_name'], how='inner', suffixes=('','_mr')
)
join_r['join_orientation'] = 'f1=Red'

# Join orientation B: f1 = Blue corner
join_b = log.merge(
    master_slim.rename(columns={'B_fighter':'f1_name','R_fighter':'f2_name',
                                 'B_odds':'f1_master_odds','R_odds':'f2_master_odds'}),
    on=['date','f1_name','f2_name'], how='inner', suffixes=('','_mb')
)
join_b['join_orientation'] = 'f1=Blue'

# Union (drop exact duplicates)
joined = pd.concat([join_r, join_b], ignore_index=True).drop_duplicates(
    subset=['date','f1_name','f2_name']
)
print(f"  join_r matches: {join_r.shape[0]:,}")
print(f"  join_b matches: {join_b.shape[0]:,}")
print(f"  union coverage: {joined.shape[0]:,} / {log.shape[0]:,} rows")
assert joined.shape[0] == log.shape[0], "Join did not cover all rows!"
print("  All rows matched. Step 1 OK.\n")

# ─────────────────────────────────────────────────────────
# STEP 2 — Build fight-level odds dataset
# ─────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 2: Build fight-level odds dataset")
print("=" * 60)

df = joined.copy()

# f1 ML from master (two-sided)
df['f1_ml']  = df['f1_master_odds'].astype(float)
df['f2_ml']  = df['f2_master_odds'].astype(float)

# Implied probs (raw)
df['f1_impl_raw'] = df['f1_ml'].apply(ml_to_impl)
df['f2_impl_raw'] = df['f2_ml'].apply(ml_to_impl)
df['vig']         = (df['f1_impl_raw'] + df['f2_impl_raw'] - 1.0).round(4)

# No-vig probs
nv = df.apply(lambda r: novig_probs(r['f1_ml'], r['f2_ml']), axis=1)
df['f1_novig'] = nv.apply(lambda x: x[0])
df['f2_novig'] = nv.apply(lambda x: x[1])

# Pick-level columns: f1 is the "value pick" in value_bet_log
df['pick_ml']    = df['f1_ml']
df['pick_novig_calc'] = df['f1_novig']
df['pick_impl']  = df['f1_impl_raw']
df['won']        = df['pick_won'].astype(int)

# Is the pick a favorite or underdog per Vegas?
df['is_fav'] = (df['f1_ml'] < 0).astype(int)

print(f"  Rows in analysis df: {df.shape[0]:,}")
print(f"  Wins: {df['won'].sum():,}  Losses: {(1-df['won']).sum():,}")
print(f"  Overall win rate: {df['won'].mean():.3f}")
print(f"  Avg vig: {df['vig'].mean():.4f} ({df['vig'].mean()*100:.2f}%)")
print(f"  f1_ml range: {df['f1_ml'].min():.0f} to {df['f1_ml'].max():.0f}")
print("  Step 2 OK.\n")

# ─────────────────────────────────────────────────────────
# STEP 3 — Bucket analysis
# ─────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 3: Bucket analysis (every 50 ML points)")
print("=" * 60)

def ml_bucket_label(ml, step=50):
    """Assign a bucket center."""
    lo = int(np.floor(ml / step) * step)
    return lo

df['ml_bucket'] = df['f1_ml'].apply(lambda x: ml_bucket_label(x))

bucket_stats = []
for bkt, grp in df.groupby('ml_bucket'):
    n = len(grp)
    wr = grp['won'].mean()
    impl_avg = grp['f1_impl_raw'].mean()
    novig_avg = grp['f1_novig'].mean()
    roi = wr * (1 + grp['f1_ml'].apply(
        lambda m: 100/m if m > 0 else 100/abs(m)*(-1)
    ).mean()) - 1  # simplified ROI
    # ROI: if you bet $100 on each pick in this bucket
    bets = grp.apply(lambda r: (r['won'] * (100/r['f1_ml'] if r['f1_ml'] > 0 else 100/abs(r['f1_ml'])) - (1-r['won'])) , axis=1)
    roi_pct = bets.mean() * 100
    reliable = n >= 15
    bucket_stats.append({
        'bucket': bkt,
        'n': n,
        'win_rate': round(wr, 4),
        'implied_prob': round(impl_avg, 4),
        'novig_prob': round(novig_avg, 4),
        'calibration_error': round(wr - novig_avg, 4),
        'roi_pct': round(roi_pct, 2),
        'reliable': reliable,
    })

buckets_df = pd.DataFrame(bucket_stats).sort_values('bucket')
reliable_df = buckets_df[buckets_df['reliable']]

print(f"  Total buckets: {len(buckets_df)}")
print(f"  Reliable buckets (N>=15): {len(reliable_df)}")
print()
print(f"  {'Bucket':>8} {'N':>5} {'WinRate':>8} {'NoVig':>8} {'CalErr':>8} {'ROI%':>8} {'Reliable':>9}")
print(f"  {'-'*8} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*9}")
for _, r in buckets_df.iterrows():
    flag = '*' if r['reliable'] else ''
    print(f"  {r['bucket']:>8.0f} {r['n']:>5} {r['win_rate']:>8.3f} {r['novig_prob']:>8.3f} "
          f"{r['calibration_error']:>8.3f} {r['roi_pct']:>8.2f} {flag:>9}")
print()
print("  (* = reliable, N>=15)")

# Summary stats on reliable buckets
if len(reliable_df):
    print(f"\n  Reliable buckets calibration error summary:")
    print(f"    Mean abs error: {reliable_df['calibration_error'].abs().mean():.4f}")
    print(f"    Max overestimate (WR > novig): {reliable_df['calibration_error'].max():.4f}")
    print(f"    Max underestimate (WR < novig): {reliable_df['calibration_error'].min():.4f}")
    print(f"    Positive bias buckets: {(reliable_df['calibration_error'] > 0).sum()}")
    print(f"    Negative bias buckets: {(reliable_df['calibration_error'] < 0).sum()}")
print("  Step 3 OK.\n")

# ─────────────────────────────────────────────────────────
# STEP 4 — Smoothed calibration curve (3-bucket rolling avg)
# ─────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 4: Smoothed calibration curve")
print("=" * 60)

smooth_df = reliable_df.sort_values('bucket').copy()
smooth_df['wr_smooth']    = smooth_df['win_rate'].rolling(3, center=True, min_periods=1).mean()
smooth_df['novig_smooth'] = smooth_df['novig_prob'].rolling(3, center=True, min_periods=1).mean()
smooth_df['smooth_err']   = (smooth_df['wr_smooth'] - smooth_df['novig_smooth']).round(4)

print(f"  {'Bucket':>8} {'WR_raw':>8} {'WR_smooth':>10} {'NoVig_s':>9} {'SmoothErr':>10}")
print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*9} {'-'*10}")
for _, r in smooth_df.iterrows():
    print(f"  {r['bucket']:>8.0f} {r['win_rate']:>8.3f} {r['wr_smooth']:>10.3f} "
          f"{r['novig_smooth']:>9.3f} {r['smooth_err']:>10.3f}")

overall_smooth_mae = smooth_df['smooth_err'].abs().mean()
print(f"\n  Smoothed MAE: {overall_smooth_mae:.4f}")
print("  Step 4 OK.\n")

# ─────────────────────────────────────────────────────────
# STEP 5 — By-weight-class breakdown
# ─────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 5: By-weight-class breakdown")
print("=" * 60)

wc_stats = []
for wc, grp in df.groupby('weight_class'):
    n = len(grp)
    wr = grp['won'].mean()
    nv_avg = grp['f1_novig'].mean()
    cal_err = wr - nv_avg
    avg_vig = grp['vig'].mean()
    wc_stats.append({
        'weight_class': wc, 'n': n,
        'win_rate': round(wr, 4),
        'novig_avg': round(nv_avg, 4),
        'calibration_error': round(cal_err, 4),
        'avg_vig': round(avg_vig, 4),
    })

wc_df = pd.DataFrame(wc_stats).sort_values('n', ascending=False)

print(f"  {'Weight Class':<30} {'N':>5} {'WinRate':>8} {'NoVig':>8} {'CalErr':>8} {'AvgVig':>8}")
print(f"  {'-'*30} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for _, r in wc_df.iterrows():
    print(f"  {r['weight_class']:<30} {r['n']:>5} {r['win_rate']:>8.3f} {r['novig_avg']:>8.3f} "
          f"{r['calibration_error']:>8.3f} {r['avg_vig']:>8.4f}")
print("  Step 5 OK.\n")

# ─────────────────────────────────────────────────────────
# STEP 6 — Favorite vs underdog summary (7 categories)
# ─────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 6: Favorite vs underdog summary")
print("=" * 60)

def ml_category(ml):
    if ml <= -600: return "Heavy Fav (≤-600)"
    if ml <= -400: return "Big Fav (-400 to -600)"
    if ml <= -200: return "Fav (-200 to -400)"
    if ml <= -100: return "Slight Fav (-100 to -200)"
    if ml <= 150:  return "Pick'em (-100 to +150)"
    if ml <= 300:  return "Dog (+150 to +300)"
    return "Big Dog (>+300)"

category_order = [
    "Heavy Fav (≤-600)",
    "Big Fav (-400 to -600)",
    "Fav (-200 to -400)",
    "Slight Fav (-100 to -200)",
    "Pick'em (-100 to +150)",
    "Dog (+150 to +300)",
    "Big Dog (>+300)",
]

df['category'] = df['f1_ml'].apply(ml_category)

cat_stats = []
for cat in category_order:
    grp = df[df['category'] == cat]
    n = len(grp)
    if n == 0:
        continue
    wr = grp['won'].mean()
    nv_avg = grp['f1_novig'].mean()
    cal_err = wr - nv_avg
    avg_ml = grp['f1_ml'].mean()
    cat_stats.append({
        'category': cat, 'n': n,
        'avg_ml': round(avg_ml, 1),
        'win_rate': round(wr, 4),
        'novig_avg': round(nv_avg, 4),
        'calibration_error': round(cal_err, 4),
    })

cat_df = pd.DataFrame(cat_stats)

print(f"  {'Category':<30} {'N':>5} {'AvgML':>7} {'WinRate':>8} {'NoVig':>8} {'CalErr':>8}")
print(f"  {'-'*30} {'-'*5} {'-'*7} {'-'*8} {'-'*8} {'-'*8}")
for _, r in cat_df.iterrows():
    print(f"  {r['category']:<30} {r['n']:>5} {r['avg_ml']:>7.0f} {r['win_rate']:>8.3f} "
          f"{r['novig_avg']:>8.3f} {r['calibration_error']:>8.3f}")

print()
fav_rows = cat_df[cat_df['calibration_error'] > 0.02]
dog_rows = cat_df[cat_df['calibration_error'] < -0.02]
if len(fav_rows):
    print(f"  Systematic OVER-performance (WR > novig by >2pp): {list(fav_rows['category'])}")
if len(dog_rows):
    print(f"  Systematic UNDER-performance (WR < novig by >2pp): {list(dog_rows['category'])}")
print("  Step 6 OK.\n")

# ─────────────────────────────────────────────────────────
# STEP 7 — Fit calibration curve
# ─────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 7: Fit calibration curve")
print("=" * 60)

fit_df = reliable_df.copy()
x = fit_df['novig_prob'].values
y = fit_df['win_rate'].values

# Linear fit: WR = a * novig + b
slope, intercept, r_val, p, se = stats.linregress(x, y)
print(f"  Linear fit: WR = {slope:.4f} * novig + {intercept:.4f}")
print(f"    R²={r_val**2:.4f}  p={p:.4f}  SE={se:.4f}")
lin_residuals = y - (slope * x + intercept)
print(f"    Residual MAE: {np.abs(lin_residuals).mean():.4f}")

# Power fit: WR = a * novig^b  (only for novig in (0,1))
try:
    from scipy.optimize import curve_fit
    def power_model(x, a, b):
        return a * np.power(x, b)
    popt, _ = curve_fit(power_model, x, y, p0=[1.0, 1.0], maxfev=5000)
    y_pow = power_model(x, *popt)
    pow_mae = np.abs(y - y_pow).mean()
    print(f"\n  Power fit: WR = {popt[0]:.4f} * novig^{popt[1]:.4f}")
    print(f"    MAE: {pow_mae:.4f}")
    power_params = {'a': round(float(popt[0]), 4), 'b': round(float(popt[1]), 4)}
except Exception as e:
    print(f"  Power fit failed: {e}")
    power_params = None

# Interpretation
print()
if slope < 1.0:
    print(f"  Slope < 1.0 ({slope:.3f}): Heavy favorites perform WORSE than implied; underdogs perform BETTER.")
elif slope > 1.0:
    print(f"  Slope > 1.0 ({slope:.3f}): Heavy favorites perform BETTER than implied.")
else:
    print(f"  Slope ≈ 1.0: Odds are approximately calibrated.")

print(f"  Intercept = {intercept:.4f}: At novig=0.5, predicted WR = {slope*0.5+intercept:.3f} vs ideal 0.5")
print(f"  R² = {r_val**2:.4f}")
print("  Step 7 OK.\n")

# ─────────────────────────────────────────────────────────
# STEP 8 — Feature recommendation
# ─────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 8: Feature recommendation")
print("=" * 60)

mean_abs_cal_err = reliable_df['calibration_error'].abs().mean()
max_cal_err = reliable_df['calibration_error'].abs().max()
systematic_bias = reliable_df['calibration_error'].mean()

print(f"  Overall systematic bias (mean cal_err): {systematic_bias:+.4f}")
print(f"  Mean absolute calibration error: {mean_abs_cal_err:.4f}")
print(f"  Max absolute calibration error: {max_cal_err:.4f}")

# Build adjustment lookup per category
adj_lookup = {}
for _, r in cat_df.iterrows():
    adj_lookup[r['category']] = round(float(r['calibration_error']), 4)

print()
print("  Recommended feature: odds_calibration_adjustment")
print("  Construction: for each fight, look up the ML category of the value pick,")
print("    assign the historical calibration error as a signed float.")
print("    (+) = model outperforms implied; (-) = underperforms.")
print()
print("  Category → adjustment:")
for cat, adj in adj_lookup.items():
    direction = '↑ over' if adj > 0 else ('↓ under' if adj < 0 else '≈ flat')
    print(f"    {cat:<30} {adj:+.4f}  [{direction}]")

print()
if mean_abs_cal_err >= 0.03:
    print("  VERDICT: Calibration error is MATERIAL (MAE ≥ 3pp).")
    print("  Adding odds_calibration_adjustment to Model 2B features is RECOMMENDED.")
elif mean_abs_cal_err >= 0.015:
    print("  VERDICT: Calibration error is MODERATE (1.5–3pp MAE).")
    print("  Feature may provide marginal lift — test in ablation before adding.")
else:
    print("  VERDICT: Calibration error is SMALL (<1.5pp MAE).")
    print("  Feature unlikely to provide meaningful lift. Low priority.")

print("  Step 8 OK.\n")

# ─────────────────────────────────────────────────────────
# STEP 9 — Save results
# ─────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 9: Save results")
print("=" * 60)

results = {
    "meta": {
        "analysis_date": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "n_fights": int(df.shape[0]),
        "n_wins": int(df['won'].sum()),
        "overall_win_rate": round(float(df['won'].mean()), 4),
        "avg_vig": round(float(df['vig'].mean()), 4),
    },
    "bucket_analysis": buckets_df.to_dict(orient='records'),
    "smoothed_curve": smooth_df[['bucket','wr_smooth','novig_smooth','smooth_err']].to_dict(orient='records'),
    "by_weight_class": wc_df.to_dict(orient='records'),
    "by_category": cat_df.to_dict(orient='records'),
    "linear_fit": {
        "slope": round(float(slope), 4),
        "intercept": round(float(intercept), 4),
        "r_squared": round(float(r_val**2), 4),
        "p_value": round(float(p), 6),
        "residual_mae": round(float(np.abs(lin_residuals).mean()), 4),
    },
    "power_fit": power_params,
    "calibration_summary": {
        "systematic_bias": round(float(systematic_bias), 4),
        "mean_abs_cal_err": round(float(mean_abs_cal_err), 4),
        "max_abs_cal_err": round(float(max_cal_err), 4),
        "smoothed_mae": round(float(overall_smooth_mae), 4),
        "reliable_buckets": int(len(reliable_df)),
    },
    "feature_recommendation": {
        "feature_name": "odds_calibration_adjustment",
        "category_adjustments": adj_lookup,
        "verdict": (
            "RECOMMENDED" if mean_abs_cal_err >= 0.03 else
            "TEST_IN_ABLATION" if mean_abs_cal_err >= 0.015 else
            "LOW_PRIORITY"
        ),
    }
}

json_path = os.path.join(OUT, 'calibration_results.json')
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"  Saved: {json_path}")

# Build markdown findings doc
findings_lines = [
    "# Odds Calibration Analysis — Findings",
    "",
    f"**Date:** {results['meta']['analysis_date']}  ",
    f"**Fights analyzed:** {results['meta']['n_fights']:,}  ",
    f"**Overall win rate:** {results['meta']['overall_win_rate']:.1%}  ",
    f"**Avg vig:** {results['meta']['avg_vig']:.2%}  ",
    "",
    "---",
    "",
    "## Summary",
    "",
    f"- **Systematic bias:** {systematic_bias:+.4f} (positive = model picks win more than implied)",
    f"- **Mean absolute calibration error:** {mean_abs_cal_err:.4f} ({mean_abs_cal_err*100:.2f} pp)",
    f"- **Max absolute calibration error:** {max_cal_err:.4f} ({max_cal_err*100:.2f} pp)",
    f"- **Smoothed MAE:** {overall_smooth_mae:.4f}",
    f"- **Reliable ML buckets (N≥15):** {len(reliable_df)}",
    "",
    "---",
    "",
    "## Linear Calibration Fit",
    "",
    f"```",
    f"WR = {slope:.4f} × novig + {intercept:.4f}",
    f"R² = {r_val**2:.4f}  |  p = {p:.4f}  |  Residual MAE = {np.abs(lin_residuals).mean():.4f}",
    f"```",
    "",
]

if slope < 1.0:
    findings_lines.append(
        f"Slope < 1 ({slope:.3f}): heavy favorites underperform their implied probability; "
        f"underdogs outperform. This is consistent with the well-known longshot bias in combat sports."
    )
elif slope > 1.0:
    findings_lines.append(
        f"Slope > 1 ({slope:.3f}): heavy favorites outperform their implied probability."
    )

if power_params:
    findings_lines += [
        "",
        f"Power fit: WR = {power_params['a']:.4f} × novig^{power_params['b']:.4f}",
    ]

findings_lines += [
    "",
    "---",
    "",
    "## By ML Category",
    "",
    "| Category | N | WinRate | NoVig | CalErr |",
    "|---|---|---|---|---|",
]
for _, row in cat_df.iterrows():
    findings_lines.append(
        f"| {row['category']} | {row['n']} | {row['win_rate']:.3f} | {row['novig_avg']:.3f} | {row['calibration_error']:+.3f} |"
    )

findings_lines += [
    "",
    "---",
    "",
    "## By Weight Class",
    "",
    "| Weight Class | N | WinRate | NoVig | CalErr | AvgVig |",
    "|---|---|---|---|---|---|",
]
for _, row in wc_df.iterrows():
    findings_lines.append(
        f"| {row['weight_class']} | {row['n']} | {row['win_rate']:.3f} | {row['novig_avg']:.3f} | {row['calibration_error']:+.3f} | {row['avg_vig']:.4f} |"
    )

findings_lines += [
    "",
    "---",
    "",
    "## Feature Recommendation",
    "",
    f"**Proposed feature:** `odds_calibration_adjustment`  ",
    f"**Verdict:** {results['feature_recommendation']['verdict']}",
    "",
    "Construction: assign a signed calibration adjustment based on the ML category of the value pick.  ",
    "Positive = historical over-performance vs implied; negative = under-performance.",
    "",
    "| Category | Adjustment |",
    "|---|---|",
]
for cat, adj in adj_lookup.items():
    findings_lines.append(f"| {cat} | {adj:+.4f} |")

findings_lines += [
    "",
    "---",
    "",
    "_Research only — no model, frontend, or backend files were modified._",
]

md_path = os.path.join(OUT, 'CALIBRATION_FINDINGS.md')
with open(md_path, 'w') as f:
    f.write('\n'.join(findings_lines) + '\n')
print(f"  Saved: {md_path}")
print("\nAll steps complete.")
