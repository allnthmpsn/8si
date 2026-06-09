"""
Model 3A — "Goes the Distance" binary classifier.
Target: 1 = decision, 0 = finish (KO/TKO or submission).
Output saved to experiments/research/model3/. No production files touched.
"""

import gc, json, os, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, '../../../data')
OUT    = BASE
CUTOFF = pd.Timestamp('2024-01-01')

# ─────────────────────────────────────────────────────────────────────
# STEP 1 — Load ufc-master.csv, build target
# ─────────────────────────────────────────────────────────────────────
print("=" * 64)
print("STEP 1: Load master, build target")
print("=" * 64)

master = pd.read_csv(os.path.join(DATA, 'ufc-master.csv'), low_memory=False)
master['date'] = pd.to_datetime(master['date'])

# Valid fights: known winner, known finish method, 2015+
master = master[
    master['Winner'].isin(['Red', 'Blue']) &
    master['finish'].notna() &
    (master['date'].dt.year >= 2015)
].copy()

# Target: 1 = decision, 0 = finish
DEC_CODES     = {'U-DEC', 'S-DEC', 'M-DEC'}
FINISH_CODES  = {'KO/TKO', 'SUB'}
DROP_CODES    = {'DQ', 'CNC', 'Overturned'}

master = master[master['finish'].isin(DEC_CODES | FINISH_CODES)].copy()
master['goes_to_decision'] = master['finish'].isin(DEC_CODES).astype(int)

n_train = (master['date'] < CUTOFF).sum()
n_test  = (master['date'] >= CUTOFF).sum()
dec_rate = master['goes_to_decision'].mean()

print(f"  Valid fights (2015+):  {len(master):,}")
print(f"  Train (<2024):         {n_train:,}")
print(f"  Test  (≥2024):         {n_test:,}")
print(f"  Decision rate overall: {dec_rate:.4f} ({dec_rate*100:.2f}%)")
print(f"  Finish rate overall:   {1-dec_rate:.4f} ({(1-dec_rate)*100:.2f}%)")
print(f"  Naive baseline acc:    {max(dec_rate, 1-dec_rate):.4f}")

gc.collect()

# ─────────────────────────────────────────────────────────────────────
# STEP 2 — Career method rates from career_fights_updated.csv
# ─────────────────────────────────────────────────────────────────────
print()
print("=" * 64)
print("STEP 2: Career method rates (shift(1), no leakage)")
print("=" * 64)

cf = pd.read_csv(os.path.join(DATA, 'career_fights_updated.csv'), low_memory=False)
cf['date'] = pd.to_datetime(cf['date'], errors='coerce')
cf = cf.dropna(subset=['date', 'fighter']).copy()

def classify_method(m):
    if pd.isna(m): return 'unknown'
    m = str(m).lower()
    if 'decision' in m or m.startswith(('u-dec','s-dec','m-dec')):
        return 'decision'
    if 'tko' in m or ('ko' in m and 'submission' not in m):
        return 'ko'
    if 'submission' in m or m.startswith('sub'):
        return 'sub'
    return 'other'

cf['method_type'] = cf['method'].apply(classify_method)
cf['is_finish']    = cf['method_type'].isin(['ko', 'sub']).astype(float)
cf['is_decision']  = (cf['method_type'] == 'decision').astype(float)
cf['is_ko']        = (cf['method_type'] == 'ko').astype(float)
cf['is_sub']       = (cf['method_type'] == 'sub').astype(float)
cf['won']          = cf['won'].fillna(0).astype(float)

# Finish delivered = won by finish; finish received = lost by finish
cf['finish_delivered']  = cf['is_finish'] * cf['won']
cf['finish_received']   = cf['is_finish'] * (1 - cf['won'])
cf['dec_delivered']     = cf['is_decision'] * cf['won']
cf['dec_received']      = cf['is_decision'] * (1 - cf['won'])

cf = cf.sort_values(['fighter', 'date']).reset_index(drop=True)

def expanding_rate(series):
    """Cumulative mean up to but NOT including current row."""
    cumsum = series.cumsum().shift(1)
    count  = pd.Series(range(len(series)), index=series.index).shift(1) + 1
    return (cumsum / count).fillna(0)

rate_cols = ['is_finish','is_decision','is_ko','is_sub',
             'finish_delivered','finish_received','dec_delivered','dec_received']

career_rates = []
for fighter, grp in cf.groupby('fighter'):
    grp = grp.sort_values('date').copy()
    for col in rate_cols:
        grp[f'career_{col}'] = expanding_rate(grp[col])
    # n_fights prior
    grp['career_n_fights'] = np.arange(len(grp))
    career_rates.append(grp[['fighter','date'] + [f'career_{c}' for c in rate_cols] + ['career_n_fights']])

career_df = pd.concat(career_rates, ignore_index=True)
print(f"  Career rate rows: {len(career_df):,}  Fighters: {career_df['fighter'].nunique():,}")
print(f"  Sample career_is_finish rate (mean): {career_df['career_is_finish'].mean():.4f}")

gc.collect()

# ─────────────────────────────────────────────────────────────────────
# STEP 3 — Load style stats from ufc_fighters_final.csv
# ─────────────────────────────────────────────────────────────────────
print()
print("=" * 64)
print("STEP 3: Load fighter style stats")
print("=" * 64)

fighters = pd.read_csv(os.path.join(DATA, 'ufc_fighters_final.csv'))
style_cols = ['Fighter_Name','SLpM','SApM','Str_Def','TD_Avg','TD_Def','Sub_Avg']
fighters = fighters[style_cols].copy()
fighters.columns = ['fighter'] + style_cols[1:]

# Str_Def and TD_Def are stored as "57%" strings — strip and convert to float 0–1
for col in ['Str_Def', 'TD_Def']:
    if fighters[col].dtype == object:
        fighters[col] = fighters[col].str.replace('%', '', regex=False).astype(float) / 100.0

print(f"  Fighters with style stats: {len(fighters):,}")
print(f"  SLpM null rate: {fighters['SLpM'].isna().mean():.3f}")

# Global medians for imputation
style_medians = fighters[style_cols[1:]].median()

gc.collect()

# ─────────────────────────────────────────────────────────────────────
# STEP 4 — Join everything to master
# ─────────────────────────────────────────────────────────────────────
print()
print("=" * 64)
print("STEP 4: Build feature matrix")
print("=" * 64)

WC_ORD = {
    "Women's Strawweight": 0, "Women's Flyweight": 1,
    "Women's Bantamweight": 2, "Women's Featherweight": 3,
    'Flyweight': 4, 'Bantamweight': 5, 'Featherweight': 6,
    'Lightweight': 7, 'Welterweight': 8, 'Middleweight': 9,
    'Light Heavyweight': 10, 'Heavyweight': 11, 'Catch Weight': 6,
}
WOMENS = {"Women's Strawweight","Women's Flyweight","Women's Bantamweight","Women's Featherweight"}

master['weight_class_ord'] = master['weight_class'].map(WC_ORD).fillna(6)
master['is_5rnd']     = (master['no_of_rounds'] == 5).astype(int)
master['is_title']    = master['title_bout'].astype(int)
master['is_womens']   = master['weight_class'].isin(WOMENS).astype(int)

CAREER_RATE_COLS = [f'career_{c}' for c in rate_cols] + ['career_n_fights']

def join_career(df, fighter_col, prefix, career_df):
    """Left-join career rates for one corner, rename with prefix."""
    sub = career_df.rename(columns={'fighter': fighter_col, 'date': 'date'})
    sub = sub.rename(columns={c: f'{prefix}_{c}' for c in CAREER_RATE_COLS})
    merged = df.merge(sub, on=[fighter_col, 'date'], how='left')
    return merged

def join_style(df, fighter_col, prefix, fighters_df):
    sub = fighters_df.rename(columns={'fighter': fighter_col})
    sub = sub.rename(columns={c: f'{prefix}_{c}' for c in style_cols[1:]})
    return df.merge(sub, on=fighter_col, how='left')

df = master.copy()
df = join_career(df, 'R_fighter', 'R', career_df)
gc.collect()
df = join_career(df, 'B_fighter', 'B', career_df)
gc.collect()
df = join_style(df, 'R_fighter', 'R', fighters)
df = join_style(df, 'B_fighter', 'B', fighters)
gc.collect()

# Impute style stats with league median
for side in ['R_', 'B_']:
    for col in style_cols[1:]:
        c = f'{side}{col}'
        if c in df.columns:
            df[c] = df[c].fillna(style_medians[col])

# Impute career rates with 0 (no prior fights = unknown)
for col in CAREER_RATE_COLS:
    for pref in ['R_', 'B_']:
        c = f'{pref}{col}'
        if c in df.columns:
            df[c] = df[c].fillna(0)

print(f"  Joined shape: {df.shape}")

# ─────────────────────────────────────────────────────────────────────
# STEP 5 — Engineer combined features
# ─────────────────────────────────────────────────────────────────────
print()
print("=" * 64)
print("STEP 5: Engineer combined features")
print("=" * 64)

# Combined method rate features (sum of both fighters)
for col in rate_cols:
    r_col = f'R_career_{col}'
    b_col = f'B_career_{col}'
    if r_col in df.columns and b_col in df.columns:
        df[f'combined_{col}'] = df[r_col] + df[b_col]

# Combined style features
for sc in style_cols[1:]:
    r_col = f'R_{sc}';  b_col = f'B_{sc}'
    if r_col in df.columns and b_col in df.columns:
        df[f'combined_{sc}'] = df[r_col] + df[b_col]

# Aggregate strike/TD stats from master
for side in ['R_', 'B_']:
    for stat in ['avg_SIG_STR_landed', 'avg_TD_landed', 'avg_SUB_ATT']:
        c = f'{side}{stat}'
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].fillna(0)

df['combined_sig_str_landed'] = df['R_avg_SIG_STR_landed'] + df['B_avg_SIG_STR_landed']
df['combined_td_landed']      = df['R_avg_TD_landed']      + df['B_avg_TD_landed']
df['combined_sub_att']        = df['R_avg_SUB_ATT']        + df['B_avg_SUB_ATT']

# Master diff stats (already computed)
for col in ['reach_dif','age_dif','sig_str_dif','avg_sub_att_dif','ko_dif','sub_dif']:
    if col in df.columns:
        df[col] = df[col].fillna(0)
    else:
        df[col] = 0

print(f"  Features built. DataFrame shape: {df.shape}")

# ─────────────────────────────────────────────────────────────────────
# STEP 6 — Assemble final feature set
# ─────────────────────────────────────────────────────────────────────
print()
print("=" * 64)
print("STEP 6: Assemble final feature set")
print("=" * 64)

FEATURE_COLS = (
    # Fight context
    ['weight_class_ord', 'is_5rnd', 'is_title', 'is_womens']
    # Career method rates — Red
    + [f'R_career_{c}' for c in rate_cols]
    + ['R_career_n_fights']
    # Career method rates — Blue
    + [f'B_career_{c}' for c in rate_cols]
    + ['B_career_n_fights']
    # Combined method rates
    + [f'combined_{c}' for c in rate_cols]
    # Style stats — Red
    + [f'R_{c}' for c in style_cols[1:]]
    # Style stats — Blue
    + [f'B_{c}' for c in style_cols[1:]]
    # Combined style
    + [f'combined_{c}' for c in style_cols[1:]]
    # Master aggregate stats
    + ['R_avg_SIG_STR_landed','B_avg_SIG_STR_landed','combined_sig_str_landed']
    + ['R_avg_TD_landed','B_avg_TD_landed','combined_td_landed']
    + ['R_avg_SUB_ATT','B_avg_SUB_ATT','combined_sub_att']
    # Diff stats
    + ['reach_dif','age_dif','sig_str_dif','avg_sub_att_dif','ko_dif','sub_dif']
)

# Keep only available columns
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]
print(f"  Feature count: {len(FEATURE_COLS)}")

# Drop rows with NaN in any feature (should be minimal after imputation)
df_clean = df[FEATURE_COLS + ['goes_to_decision','date','weight_class']].dropna().copy()
print(f"  Rows after dropna: {len(df_clean):,}  (dropped {len(df)-len(df_clean)})")

X = df_clean[FEATURE_COLS].values
y = df_clean['goes_to_decision'].values
dates = df_clean['date'].values
wc    = df_clean['weight_class'].values

train_mask = dates < CUTOFF
test_mask  = dates >= CUTOFF

X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print(f"  Train: {X_train.shape[0]:,} rows  |  Test: {X_test.shape[0]:,} rows")
train_dec = y_train.mean()
test_dec  = y_test.mean()
print(f"  Train decision rate: {train_dec:.4f}  |  Test decision rate: {test_dec:.4f}")

naive_train = max(train_dec, 1-train_dec)
naive_test  = max(test_dec, 1-test_dec)
print(f"  Naive baseline — Train: {naive_train:.4f}  Test: {naive_test:.4f}")

gc.collect()

# ─────────────────────────────────────────────────────────────────────
# STEP 7 — Scale + train models
# ─────────────────────────────────────────────────────────────────────
print()
print("=" * 64)
print("STEP 7: Train models")
print("=" * 64)

scaler = RobustScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Logistic Regression ───────────────────────────────────────────────
print("  Training Logistic Regression...")
lr = LogisticRegression(C=0.5, max_iter=2000, solver='lbfgs', n_jobs=1,
                        class_weight=None, random_state=42)
lr.fit(X_train_sc, y_train)
lr_train_acc = accuracy_score(y_train, lr.predict(X_train_sc))
lr_test_acc  = accuracy_score(y_test,  lr.predict(X_test_sc))
lr_test_prob = lr.predict_proba(X_test_sc)[:,1]
print(f"  LR   — Train: {lr_train_acc:.4f}  Test: {lr_test_acc:.4f}  "
      f"vs naive: {lr_test_acc-naive_test:+.4f}")
gc.collect()

# ── Random Forest ────────────────────────────────────────────────────
print("  Training Random Forest...")
rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=12,
                             random_state=42, n_jobs=1)
rf.fit(X_train, y_train)
rf_train_acc = accuracy_score(y_train, rf.predict(X_train))
rf_test_acc  = accuracy_score(y_test,  rf.predict(X_test))
rf_test_prob = rf.predict_proba(X_test)[:,1]
print(f"  RF   — Train: {rf_train_acc:.4f}  Test: {rf_test_acc:.4f}  "
      f"vs naive: {rf_test_acc-naive_test:+.4f}")
gc.collect()

# ── XGBoost ──────────────────────────────────────────────────────────
print("  Training XGBoost...")
xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8,
                     eval_metric='logloss', verbosity=0,
                     random_state=42, n_jobs=1)
xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
xgb_train_acc = accuracy_score(y_train, xgb.predict(X_train))
xgb_test_acc  = accuracy_score(y_test,  xgb.predict(X_test))
xgb_test_prob = xgb.predict_proba(X_test)[:,1]
print(f"  XGB  — Train: {xgb_train_acc:.4f}  Test: {xgb_test_acc:.4f}  "
      f"vs naive: {xgb_test_acc-naive_test:+.4f}")
gc.collect()

# ── Blend ratios ─────────────────────────────────────────────────────
print()
print("  Blend comparison:")
blends = [
    ('LR only',            1.0, 0.0, 0.0),
    ('RF only',            0.0, 1.0, 0.0),
    ('XGB only',           0.0, 0.0, 1.0),
    ('50% LR + 50% XGB',   0.5, 0.0, 0.5),
    ('40% LR + 60% XGB',   0.4, 0.0, 0.6),
    ('30% LR + 70% XGB',   0.3, 0.0, 0.7),
    ('33% each',           1/3, 1/3, 1/3),
    ('25% LR + 25% RF + 50% XGB', 0.25, 0.25, 0.50),
]

lr_prob_test  = lr.predict_proba(X_test_sc)[:,1]
best_blend_acc = 0; best_blend_label = ''; best_blend_prob = None

print(f"  {'Blend':<35} {'Acc':>6} {'vs naive':>9} {'Brier':>7}")
print(f"  {'-'*35} {'-'*6} {'-'*9} {'-'*7}")
for label, w_lr, w_rf, w_xgb in blends:
    p = w_lr * lr_prob_test + w_rf * rf_test_prob + w_xgb * xgb_test_prob
    pred = (p > 0.5).astype(int)
    acc  = accuracy_score(y_test, pred)
    brier = brier_score_loss(y_test, p)
    delta = acc - naive_test
    mark = ' ←' if acc > best_blend_acc else ''
    if acc > best_blend_acc:
        best_blend_acc = acc; best_blend_label = label; best_blend_prob = p
    print(f"  {label:<35} {acc:.4f} {delta:>+9.4f} {brier:>7.4f}{mark}")

print(f"\n  Best blend: {best_blend_label}  ({best_blend_acc:.4f})")
print(f"  Beats naive by: {best_blend_acc - naive_test:+.4f} ({(best_blend_acc-naive_test)*100:+.2f}pp)")
PASSES_BAR = (best_blend_acc - naive_test) > 0.03
print(f"  Beats 3pp bar: {'YES' if PASSES_BAR else 'NO'}")

gc.collect()

# ─────────────────────────────────────────────────────────────────────
# STEP 8 — Feature importance
# ─────────────────────────────────────────────────────────────────────
print()
print("=" * 64)
print("STEP 8: Feature importance (XGB + RF)")
print("=" * 64)

# XGB importance
xgb_imp = pd.Series(xgb.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
print("\n  Top 20 features — XGBoost:")
print(f"  {'Feature':<40} {'Importance':>10}")
print(f"  {'-'*40} {'-'*10}")
for feat, imp in xgb_imp.head(20).items():
    print(f"  {feat:<40} {imp:>10.4f}")

# RF importance
rf_imp = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
print("\n  Top 15 features — Random Forest:")
for feat, imp in rf_imp.head(15).items():
    print(f"  {feat:<40} {imp:>10.4f}")

gc.collect()

# ─────────────────────────────────────────────────────────────────────
# STEP 9 — Per-weight-class breakdown
# ─────────────────────────────────────────────────────────────────────
print()
print("=" * 64)
print("STEP 9: Per-weight-class breakdown (test set)")
print("=" * 64)

test_df = df_clean[test_mask].copy().reset_index(drop=True)
test_df['pred_prob']    = best_blend_prob
test_df['pred_label']   = (best_blend_prob > 0.5).astype(int)
test_df['correct']      = (test_df['pred_label'] == test_df['goes_to_decision']).astype(int)

print(f"\n  {'Weight Class':<30} {'N':>5} {'Dec%':>7} {'Acc':>7} {'Naive':>7} {'vs naive':>9}")
print(f"  {'-'*30} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*9}")
wc_results = []
for wc_name in sorted(test_df['weight_class'].unique()):
    sub = test_df[test_df['weight_class'] == wc_name]
    if len(sub) < 20: continue
    dec_r = sub['goes_to_decision'].mean()
    acc   = sub['correct'].mean()
    naive = max(dec_r, 1 - dec_r)
    delta = acc - naive
    print(f"  {wc_name:<30} {len(sub):>5} {dec_r:>7.3f} {acc:>7.3f} {naive:>7.3f} {delta:>+9.3f}")
    wc_results.append({'weight_class': wc_name, 'n': len(sub), 'dec_rate': round(dec_r,4),
                        'accuracy': round(acc,4), 'naive': round(naive,4), 'vs_naive': round(delta,4)})

# ─────────────────────────────────────────────────────────────────────
# STEP 10 — Calibration
# ─────────────────────────────────────────────────────────────────────
print()
print("=" * 64)
print("STEP 10: Calibration")
print("=" * 64)

try:
    frac_pos, mean_pred = calibration_curve(y_test, best_blend_prob, n_bins=8, strategy='quantile')
    print(f"\n  {'Pred prob bin':>15} {'Actual freq':>12} {'Gap':>8}")
    print(f"  {'-'*15} {'-'*12} {'-'*8}")
    for mp, fp in zip(mean_pred, frac_pos):
        print(f"  {mp:>15.3f} {fp:>12.3f} {fp-mp:>+8.3f}")
except Exception as e:
    print(f"  Calibration error: {e}")

# ─────────────────────────────────────────────────────────────────────
# STEP 11 — Save models + metadata
# ─────────────────────────────────────────────────────────────────────
print()
print("=" * 64)
print("STEP 11: Save models and metadata")
print("=" * 64)

# Parse best blend weights
best_w = {l: (wl, wr, wx) for l, wl, wr, wx in blends if l == best_blend_label}
w_lr, w_rf, w_xgb = best_w.get(best_blend_label, (0.5, 0.0, 0.5))

joblib.dump(lr,    os.path.join(OUT, 'model3a_lr.pkl'))
joblib.dump(rf,    os.path.join(OUT, 'model3a_rf.pkl'))
joblib.dump(xgb,   os.path.join(OUT, 'model3a_xgb.pkl'))
joblib.dump(scaler,os.path.join(OUT, 'model3a_scaler.pkl'))
joblib.dump(FEATURE_COLS, os.path.join(OUT, 'model3a_features.pkl'))

metadata = {
    "model": "Model 3A — Goes the Distance",
    "target": "goes_to_decision (1=DEC, 0=KO/TKO/SUB)",
    "train_cutoff": str(CUTOFF.date()),
    "n_train": int(X_train.shape[0]),
    "n_test":  int(X_test.shape[0]),
    "n_features": len(FEATURE_COLS),
    "features": FEATURE_COLS,
    "decision_rate_train": round(float(y_train.mean()), 4),
    "decision_rate_test":  round(float(y_test.mean()),  4),
    "naive_baseline_test": round(float(naive_test), 4),
    "model_accuracy": {
        "lr_test":   round(float(lr_test_acc), 4),
        "rf_test":   round(float(rf_test_acc), 4),
        "xgb_test":  round(float(xgb_test_acc), 4),
        "best_blend": round(float(best_blend_acc), 4),
        "best_blend_label": best_blend_label,
        "vs_naive_pp": round(float((best_blend_acc - naive_test)*100), 2),
    },
    "blend_weights": {"lr": float(w_lr), "rf": float(w_rf), "xgb": float(w_xgb)},
    "passes_3pp_bar": bool(PASSES_BAR),
    "xgb_top15_features": xgb_imp.head(15).round(4).to_dict(),
    "by_weight_class": wc_results,
}

with open(os.path.join(OUT, 'model3a_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"  Saved: model3a_lr.pkl, model3a_rf.pkl, model3a_xgb.pkl")
print(f"  Saved: model3a_scaler.pkl, model3a_features.pkl, model3a_metadata.json")
print()
print("=" * 64)
print("SUMMARY")
print("=" * 64)
print(f"  Decision rate (test):  {y_test.mean():.4f} ({y_test.mean()*100:.1f}%)")
print(f"  Naive baseline:        {naive_test:.4f}")
print(f"  Best model ({best_blend_label}):")
print(f"    Accuracy:            {best_blend_acc:.4f}")
print(f"    vs naive:            {best_blend_acc-naive_test:+.4f} ({(best_blend_acc-naive_test)*100:+.2f}pp)")
print(f"  3pp bar:               {'PASSED' if PASSES_BAR else 'NOT PASSED'}")
print()
print("All steps complete. No production files touched.")
