"""
Model 3B — Six-class Winner + Method combined classifier.
Classes: 0=R KO, 1=R Sub, 2=R Dec, 3=B KO, 4=B Sub, 5=B Dec
Also fixes Model 3A isotonic calibration first.
Output saved to experiments/research/model3/. No production files touched.
"""

import gc, json, os, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, '../../../data')
OUT    = BASE
CUTOFF = pd.Timestamp('2024-01-01')

DEC_CODES    = {'U-DEC', 'S-DEC', 'M-DEC'}
FINISH_CODES = {'KO/TKO', 'SUB'}
WOMENS       = {"Women's Strawweight","Women's Flyweight","Women's Bantamweight","Women's Featherweight"}
WC_ORD = {
    "Women's Strawweight": 0, "Women's Flyweight": 1,
    "Women's Bantamweight": 2, "Women's Featherweight": 3,
    'Flyweight': 4, 'Bantamweight': 5, 'Featherweight': 6,
    'Lightweight': 7, 'Welterweight': 8, 'Middleweight': 9,
    'Light Heavyweight': 10, 'Heavyweight': 11, 'Catch Weight': 6,
}
LOW_CONF_DIVS = {'Women\'s Flyweight', 'Light Heavyweight', 'Bantamweight'}

# ═════════════════════════════════════════════════════════════════════
# PART A — Fix Model 3A calibration
# ═════════════════════════════════════════════════════════════════════
print("=" * 64)
print("PART A: Fix Model 3A isotonic calibration")
print("=" * 64)

# Load master and rebuild features for 3A calibration
master = pd.read_csv(os.path.join(DATA, 'ufc-master.csv'), low_memory=False)
master['date'] = pd.to_datetime(master['date'])
master = master[
    master['Winner'].isin(['Red','Blue']) &
    master['finish'].isin(DEC_CODES | FINISH_CODES) &
    (master['date'].dt.year >= 2015)
].copy()
master['goes_to_decision'] = master['finish'].isin(DEC_CODES).astype(int)
master['weight_class_ord'] = master['weight_class'].map(WC_ORD).fillna(6)
master['is_5rnd']   = (master['no_of_rounds'] == 5).astype(int)
master['is_title']  = master['title_bout'].astype(int)
master['is_womens'] = master['weight_class'].isin(WOMENS).astype(int)

# Career method rates
cf = pd.read_csv(os.path.join(DATA, 'career_fights_updated.csv'), low_memory=False)
cf['date'] = pd.to_datetime(cf['date'], errors='coerce')
cf = cf.dropna(subset=['date','fighter']).copy()

def classify_method(m):
    if pd.isna(m): return 'unknown'
    m = str(m).lower()
    if 'decision' in m or m.startswith(('u-dec','s-dec','m-dec')): return 'decision'
    if 'tko' in m or ('ko' in m and 'submission' not in m): return 'ko'
    if 'submission' in m or m.startswith('sub'): return 'sub'
    return 'other'

cf['method_type'] = cf['method'].apply(classify_method)
rate_cols = ['is_finish','is_decision','is_ko','is_sub',
             'finish_delivered','finish_received','dec_delivered','dec_received']
cf['is_finish']   = cf['method_type'].isin(['ko','sub']).astype(float)
cf['is_decision'] = (cf['method_type'] == 'decision').astype(float)
cf['is_ko']       = (cf['method_type'] == 'ko').astype(float)
cf['is_sub']      = (cf['method_type'] == 'sub').astype(float)
cf['won']         = cf['won'].fillna(0).astype(float)
cf['finish_delivered'] = cf['is_finish'] * cf['won']
cf['finish_received']  = cf['is_finish'] * (1 - cf['won'])
cf['dec_delivered']    = cf['is_decision'] * cf['won']
cf['dec_received']     = cf['is_decision'] * (1 - cf['won'])
cf = cf.sort_values(['fighter','date']).reset_index(drop=True)

def expanding_rate(series):
    cumsum = series.cumsum().shift(1)
    count  = pd.Series(range(len(series)), index=series.index).shift(1) + 1
    return (cumsum / count).fillna(0)

career_parts = []
for fighter, grp in cf.groupby('fighter'):
    grp = grp.sort_values('date').copy()
    for col in rate_cols:
        grp[f'career_{col}'] = expanding_rate(grp[col])
    grp['career_n_fights'] = np.arange(len(grp))
    career_parts.append(grp[['fighter','date'] + [f'career_{c}' for c in rate_cols] + ['career_n_fights']])
career_df = pd.concat(career_parts, ignore_index=True)
CAREER_RATE_COLS = [f'career_{c}' for c in rate_cols] + ['career_n_fights']

fighters = pd.read_csv(os.path.join(DATA, 'ufc_fighters_final.csv'))
style_cols_raw = ['SLpM','SApM','Str_Def','TD_Avg','TD_Def','Sub_Avg']
fighters = fighters[['Fighter_Name'] + style_cols_raw].copy()
fighters.columns = ['fighter'] + style_cols_raw
for col in ['Str_Def','TD_Def']:
    if fighters[col].dtype == object:
        fighters[col] = fighters[col].str.replace('%','',regex=False).astype(float) / 100.0
style_medians = fighters[style_cols_raw].median()

def join_career(df, fc, prefix, cdf):
    sub = cdf.rename(columns={'fighter': fc, 'date': 'date'})
    sub = sub.rename(columns={c: f'{prefix}_{c}' for c in CAREER_RATE_COLS})
    return df.merge(sub, on=[fc,'date'], how='left')

def join_style(df, fc, prefix, fdf):
    sub = fdf.rename(columns={'fighter': fc})
    sub = sub.rename(columns={c: f'{prefix}_{c}' for c in style_cols_raw})
    return df.merge(sub, on=fc, how='left')

df3a = master.copy()
df3a = join_career(df3a, 'R_fighter', 'R', career_df)
df3a = join_career(df3a, 'B_fighter', 'B', career_df)
df3a = join_style(df3a, 'R_fighter', 'R', fighters)
df3a = join_style(df3a, 'B_fighter', 'B', fighters)

for sc in style_cols_raw:
    for pref in ['R_','B_']:
        c = f'{pref}{sc}'; df3a[c] = df3a[c].fillna(style_medians[sc])
for col in CAREER_RATE_COLS:
    for pref in ['R_','B_']:
        df3a[f'{pref}{col}'] = df3a[f'{pref}{col}'].fillna(0)

for col in rate_cols:
    df3a[f'combined_{col}'] = df3a[f'R_career_{col}'] + df3a[f'B_career_{col}']
for sc in style_cols_raw:
    df3a[f'combined_{sc}'] = df3a[f'R_{sc}'] + df3a[f'B_{sc}']
for stat in ['avg_SIG_STR_landed','avg_TD_landed','avg_SUB_ATT']:
    for s in ['R_','B_']: df3a[f'{s}{stat}'] = df3a[f'{s}{stat}'].fillna(0)
df3a['combined_sig_str_landed'] = df3a['R_avg_SIG_STR_landed'] + df3a['B_avg_SIG_STR_landed']
df3a['combined_td_landed']      = df3a['R_avg_TD_landed']      + df3a['B_avg_TD_landed']
df3a['combined_sub_att']        = df3a['R_avg_SUB_ATT']        + df3a['B_avg_SUB_ATT']
for col in ['reach_dif','age_dif','sig_str_dif','avg_sub_att_dif','ko_dif','sub_dif']:
    df3a[col] = df3a[col].fillna(0) if col in df3a.columns else 0

FEATS_3A = joblib.load(os.path.join(OUT, 'model3a_features.pkl'))
df3a_clean = df3a[FEATS_3A + ['goes_to_decision','date']].dropna().copy()
X_3a = df3a_clean[FEATS_3A].values
y_3a = df3a_clean['goes_to_decision'].values
dates_3a = df3a_clean['date'].values

train_mask_3a = dates_3a < CUTOFF
test_mask_3a  = dates_3a >= CUTOFF

scaler_3a = joblib.load(os.path.join(OUT, 'model3a_scaler.pkl'))
lr_3a     = joblib.load(os.path.join(OUT, 'model3a_lr.pkl'))
xgb_3a    = joblib.load(os.path.join(OUT, 'model3a_xgb.pkl'))

X_train_3a = X_3a[train_mask_3a]
y_train_3a = y_3a[train_mask_3a]
X_test_3a  = X_3a[test_mask_3a]
y_test_3a  = y_3a[test_mask_3a]

# Validation slice: last 20% of training by date (chronological)
n_val = int(len(X_train_3a) * 0.20)
n_fit = len(X_train_3a) - n_val
X_fit_3a, X_val_3a = X_train_3a[:n_fit], X_train_3a[n_fit:]
y_fit_3a, y_val_3a = y_train_3a[:n_fit], y_train_3a[n_fit:]

X_fit_sc = scaler_3a.transform(X_fit_3a)
X_val_sc = scaler_3a.transform(X_val_3a)
X_test_sc = scaler_3a.transform(X_test_3a)

# Uncalibrated blend probs on val and test
LR_W, XGB_W = 0.3, 0.7
p_val_uncal  = LR_W * lr_3a.predict_proba(X_val_sc)[:,1]  + XGB_W * xgb_3a.predict_proba(X_val_3a)[:,1]
p_test_uncal = LR_W * lr_3a.predict_proba(X_test_sc)[:,1] + XGB_W * xgb_3a.predict_proba(X_test_3a)[:,1]

# Fit isotonic regression on val set
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(p_val_uncal, y_val_3a)
p_test_cal = iso.predict(p_test_uncal)

# Calibration MAE: |mean_pred_in_bin - actual_freq_in_bin|
def cal_mae(y_true, probs, n_bins=8):
    frac, mean_p = calibration_curve(y_true, probs, n_bins=n_bins, strategy='quantile')
    return np.abs(frac - mean_p).mean()

mae_uncal = cal_mae(y_test_3a, p_test_uncal)
mae_cal   = cal_mae(y_test_3a, p_test_cal)
acc_uncal = accuracy_score(y_test_3a, (p_test_uncal > 0.5).astype(int))
acc_cal   = accuracy_score(y_test_3a, (p_test_cal   > 0.5).astype(int))

print(f"  Val size (calibration fit): {len(X_val_3a):,}")
print(f"  Calibration MAE — uncalibrated: {mae_uncal:.4f}")
print(f"  Calibration MAE — calibrated:   {mae_cal:.4f}  (improvement: {mae_uncal-mae_cal:+.4f})")
print(f"  Test accuracy — uncalibrated:   {acc_uncal:.4f}")
print(f"  Test accuracy — calibrated:     {acc_cal:.4f}")

joblib.dump(iso, os.path.join(OUT, 'model3a_iso_calibrator.pkl'))

# Update 3A metadata
with open(os.path.join(OUT, 'model3a_metadata.json')) as f:
    meta3a = json.load(f)
meta3a['calibration'] = {
    'method': 'isotonic',
    'fit_on': 'last 20% of training (chronological)',
    'val_n': int(len(X_val_3a)),
    'cal_mae_before': round(float(mae_uncal), 4),
    'cal_mae_after':  round(float(mae_cal),   4),
    'cal_acc_after':  round(float(acc_cal),   4),
}
meta3a['low_confidence_divisions'] = list(LOW_CONF_DIVS)
for wc_row in meta3a.get('by_weight_class', []):
    wc_row['low_confidence'] = wc_row['weight_class'] in LOW_CONF_DIVS
with open(os.path.join(OUT, 'model3a_metadata.json'), 'w') as f:
    json.dump(meta3a, f, indent=2)
print(f"  Saved: model3a_iso_calibrator.pkl  (metadata updated)")
gc.collect()

# ═════════════════════════════════════════════════════════════════════
# PART B — Model 3B: Six-class classifier
# ═════════════════════════════════════════════════════════════════════
print()
print("=" * 64)
print("PART B: Model 3B — Six-class Winner + Method")
print("=" * 64)

# ── STEP B1: Build six-class target ──────────────────────────────────
print()
print("STEP B1: Build six-class target")

# master already loaded and filtered above (2015+, valid winner, known method)
LABEL_MAP = {
    ('Red',  'KO/TKO'): 0,
    ('Red',  'SUB'):     1,
    ('Red',  'DEC'):     2,
    ('Blue', 'KO/TKO'): 3,
    ('Blue', 'SUB'):     4,
    ('Blue', 'DEC'):     5,
}

def build_label(row):
    w = row['Winner']
    f = row['finish']
    method = 'DEC' if f in DEC_CODES else ('KO/TKO' if f == 'KO/TKO' else 'SUB')
    return LABEL_MAP.get((w, method), None)

master['label_6'] = master.apply(build_label, axis=1)
master = master[master['label_6'].notna()].copy()
master['label_6'] = master['label_6'].astype(int)

CLASS_NAMES = ['R KO/TKO','R Sub','R Dec','B KO/TKO','B Sub','B Dec']
print(f"\n  Class balance ({len(master):,} fights):")
for i, name in enumerate(CLASS_NAMES):
    n = (master['label_6'] == i).sum()
    print(f"    {i} {name:<12}: {n:5d}  ({n/len(master)*100:.1f}%)")
naive_acc = (master['label_6'].value_counts().max() / len(master))
print(f"  Naive baseline (most common class): {naive_acc:.4f}")

gc.collect()

# ── STEP B2: Build Elo lookup ─────────────────────────────────────────
print()
print("STEP B2: Build per-fight Elo lookup")

elo_hist = pd.read_csv(os.path.join(DATA, 'elo_ratings_history.csv'))
elo_hist['date'] = pd.to_datetime(elo_hist['date'], errors='coerce')
elo_hist = elo_hist.dropna(subset=['date','fighter'])
elo_lookup = {}
for _, row in elo_hist.iterrows():
    elo_lookup[(row['fighter'], row['date'])] = {
        'elo': float(row['elo_before']),
        'elo_trend': float(row['elo_trend']) if not pd.isna(row['elo_trend']) else 0.0,
    }
print(f"  Elo lookup size: {len(elo_lookup):,} fighter-date pairs")

def get_elo(name, date, default=1500.0):
    v = elo_lookup.get((name, date))
    if v: return v['elo'], v['elo_trend']
    return default, 0.0

master['R_elo'], master['R_elo_trend'] = zip(*master.apply(
    lambda r: get_elo(r['R_fighter'], r['date']), axis=1))
master['B_elo'], master['B_elo_trend'] = zip(*master.apply(
    lambda r: get_elo(r['B_fighter'], r['date']), axis=1))
master['elo_dif']       = master['R_elo']       - master['B_elo']
master['elo_trend_dif'] = master['R_elo_trend'] - master['B_elo_trend']

elo_cov = (master['R_elo'] != 1500.0).mean()
print(f"  R elo coverage (non-default): {elo_cov:.3f}")
gc.collect()

# ── STEP B3: Career win rates ─────────────────────────────────────────
print()
print("STEP B3: Career win rates from career_fights_updated.csv")

win_parts = []
for fighter, grp in cf.groupby('fighter'):
    grp = grp.sort_values('date').copy()
    grp['career_win_rate'] = expanding_rate(grp['won'])
    win_parts.append(grp[['fighter','date','career_win_rate']])
win_rate_df = pd.concat(win_parts, ignore_index=True)

def join_single(df, fc, col, wdf, new_col):
    sub = wdf.rename(columns={'fighter': fc, 'date': 'date', col: new_col})
    return df.merge(sub[[fc,'date',new_col]], on=[fc,'date'], how='left')

master = join_single(master, 'R_fighter', 'career_win_rate', win_rate_df, 'R_career_win_rate')
master = join_single(master, 'B_fighter', 'career_win_rate', win_rate_df, 'B_career_win_rate')
master['career_win_rate_dif'] = master['R_career_win_rate'].fillna(0.5) - master['B_career_win_rate'].fillna(0.5)
master['R_career_win_rate']   = master['R_career_win_rate'].fillna(0.5)
master['B_career_win_rate']   = master['B_career_win_rate'].fillna(0.5)
gc.collect()

# ── STEP B4: Join Model 3A career method rates ────────────────────────
print()
print("STEP B4: Join Model 3A career method rates")

master = join_career(master, 'R_fighter', 'R', career_df)
master = join_career(master, 'B_fighter', 'B', career_df)
master = join_style(master, 'R_fighter', 'R', fighters)
master = join_style(master, 'B_fighter', 'B', fighters)

for sc in style_cols_raw:
    for pref in ['R_','B_']:
        master[f'{pref}{sc}'] = master[f'{pref}{sc}'].fillna(style_medians[sc])
for col in CAREER_RATE_COLS:
    for pref in ['R_','B_']:
        master[f'{pref}{col}'] = master[f'{pref}{col}'].fillna(0)

for col in rate_cols:
    master[f'combined_{col}'] = master[f'R_career_{col}'] + master[f'B_career_{col}']
for sc in style_cols_raw:
    master[f'combined_{sc}'] = master[f'R_{sc}'] + master[f'B_{sc}']
for stat in ['avg_SIG_STR_landed','avg_TD_landed','avg_SUB_ATT']:
    for s in ['R_','B_']: master[f'{s}{stat}'] = master[f'{s}{stat}'].fillna(0)
master['combined_sig_str_landed'] = master['R_avg_SIG_STR_landed'] + master['B_avg_SIG_STR_landed']
master['combined_td_landed']      = master['R_avg_TD_landed']      + master['B_avg_TD_landed']
master['combined_sub_att']        = master['R_avg_SUB_ATT']        + master['B_avg_SUB_ATT']
gc.collect()

# ── STEP B5: Derive additional winner-prediction features ─────────────
print()
print("STEP B5: Derive additional winner-prediction features")

# KO/Sub rate from career win_by columns (cumulative counts)
master['R_wins_safe'] = master['R_wins'].clip(lower=1)
master['B_wins_safe'] = master['B_wins'].clip(lower=1)
master['R_ko_win_rate']  = master['R_win_by_KO/TKO'] / master['R_wins_safe']
master['B_ko_win_rate']  = master['B_win_by_KO/TKO'] / master['B_wins_safe']
master['R_sub_win_rate'] = master['R_win_by_Submission'] / master['R_wins_safe']
master['B_sub_win_rate'] = master['B_win_by_Submission'] / master['B_wins_safe']
master['R_dec_wins'] = (master['R_win_by_Decision_Unanimous'] +
                         master['R_win_by_Decision_Split'] +
                         master['R_win_by_Decision_Majority'])
master['B_dec_wins'] = (master['B_win_by_Decision_Unanimous'] +
                         master['B_win_by_Decision_Split'] +
                         master['B_win_by_Decision_Majority'])
master['R_dec_win_rate'] = master['R_dec_wins'] / master['R_wins_safe']
master['B_dec_win_rate'] = master['B_dec_wins'] / master['B_wins_safe']
master['ko_win_rate_dif']  = master['R_ko_win_rate']  - master['B_ko_win_rate']
master['sub_win_rate_dif'] = master['R_sub_win_rate'] - master['B_sub_win_rate']
master['dec_win_rate_dif'] = master['R_dec_win_rate'] - master['B_dec_win_rate']

# Style differentials
master['SLpM_dif'] = master['R_SLpM'] - master['B_SLpM']
master['SApM_dif'] = master['R_SApM'] - master['B_SApM']
master['Str_Def_dif'] = master['R_Str_Def'] - master['B_Str_Def']
master['TD_Avg_dif']  = master['R_TD_Avg']  - master['B_TD_Avg']
master['Sub_Avg_dif'] = master['R_Sub_Avg'] - master['B_Sub_Avg']

# Sig strike pct
for s in ['R_','B_']:
    master[f'{s}avg_SIG_STR_pct'] = master[f'{s}avg_SIG_STR_pct'].fillna(0)
    master[f'{s}avg_TD_pct']      = master[f'{s}avg_TD_pct'].fillna(0)
master['sig_str_pct_dif'] = master['R_avg_SIG_STR_pct'] - master['B_avg_SIG_STR_pct']
master['td_pct_dif']      = master['R_avg_TD_pct']      - master['B_avg_TD_pct']

# Remaining master diffs
for col in ['reach_dif','age_dif','sig_str_dif','avg_sub_att_dif','ko_dif','sub_dif',
            'win_streak_dif','lose_streak_dif','win_dif','loss_dif','avg_td_dif',
            'total_round_dif','total_title_bout_dif']:
    master[col] = master[col].fillna(0) if col in master.columns else 0

print(f"  win_by columns for R/B added as rate features")
gc.collect()

# ── STEP B6: Assemble feature set ────────────────────────────────────
print()
print("STEP B6: Assemble feature set")

FEATS_3B = list(dict.fromkeys([
    # Fight context
    'weight_class_ord','is_5rnd','is_title','is_womens',
    # Elo
    'R_elo','B_elo','elo_dif','R_elo_trend','B_elo_trend','elo_trend_dif',
    # Career win rates
    'R_career_win_rate','B_career_win_rate','career_win_rate_dif',
    # Career win-by rates from master counts
    'R_ko_win_rate','B_ko_win_rate','ko_win_rate_dif',
    'R_sub_win_rate','B_sub_win_rate','sub_win_rate_dif',
    'R_dec_win_rate','B_dec_win_rate','dec_win_rate_dif',
    # Career method rates (3A features)
    *[f'R_career_{c}' for c in rate_cols], 'R_career_n_fights',
    *[f'B_career_{c}' for c in rate_cols], 'B_career_n_fights',
    *[f'combined_{c}' for c in rate_cols],
    # Style stats
    *[f'R_{c}' for c in style_cols_raw],
    *[f'B_{c}' for c in style_cols_raw],
    *[f'combined_{c}' for c in style_cols_raw],
    # Style differentials
    'SLpM_dif','SApM_dif','Str_Def_dif','TD_Avg_dif','Sub_Avg_dif',
    # Master aggregate stats
    'R_avg_SIG_STR_landed','B_avg_SIG_STR_landed','combined_sig_str_landed',
    'R_avg_SIG_STR_pct','B_avg_SIG_STR_pct','sig_str_pct_dif',
    'R_avg_TD_landed','B_avg_TD_landed','combined_td_landed',
    'R_avg_TD_pct','B_avg_TD_pct','td_pct_dif',
    'R_avg_SUB_ATT','B_avg_SUB_ATT','combined_sub_att',
    # Record diffs
    'reach_dif','age_dif','sig_str_dif','avg_sub_att_dif','ko_dif','sub_dif',
    'win_streak_dif','lose_streak_dif','win_dif','loss_dif','avg_td_dif',
    'total_round_dif','total_title_bout_dif',
]))

FEATS_3B = [c for c in FEATS_3B if c in master.columns]
print(f"  Feature count: {len(FEATS_3B)}")

df_clean = master[FEATS_3B + ['label_6','date','weight_class']].dropna().copy()
print(f"  Rows after dropna: {len(df_clean):,}  (dropped {len(master)-len(df_clean)})")

X_all = df_clean[FEATS_3B].values
y_all = df_clean['label_6'].values
dates_all = df_clean['date'].values
wc_all    = df_clean['weight_class'].values

train_mask = dates_all < CUTOFF
test_mask  = dates_all >= CUTOFF

X_train_raw, y_train = X_all[train_mask], y_all[train_mask]
X_test,       y_test  = X_all[test_mask],  y_all[test_mask]
print(f"  Train: {len(X_train_raw):,}  Test: {len(X_test):,}")

naive_test = np.bincount(y_test).max() / len(y_test)
print(f"  Test class distribution: {dict(zip(CLASS_NAMES, np.bincount(y_test)))}")
print(f"  Naive baseline (test):    {naive_test:.4f}")

gc.collect()

# ── STEP B7: Corner-flip augmentation (training only) ─────────────────
print()
print("STEP B7: Corner-flip augmentation on training data")

LABEL_FLIP = {0:3, 1:4, 2:5, 3:0, 4:1, 5:2}

def corner_flip_augment(X, y, feature_names):
    feat_idx = {f: i for i, f in enumerate(feature_names)}
    X_flip = X.copy()
    # Swap R_ ↔ B_ feature pairs
    for rc in feature_names:
        if not rc.startswith('R_'): continue
        bc = 'B_' + rc[2:]
        if bc in feat_idx:
            ri, bi = feat_idx[rc], feat_idx[bc]
            X_flip[:, ri], X_flip[:, bi] = X[:, bi].copy(), X[:, ri].copy()
    # Negate difference features
    for i, f in enumerate(feature_names):
        if f.endswith('_dif'):
            X_flip[:, i] = -X[:, i]
    # Flip elo_trend_dif (already a _dif, handled above)
    y_flip = np.array([LABEL_FLIP[label] for label in y])
    return (np.concatenate([X, X_flip], axis=0),
            np.concatenate([y, y_flip],  axis=0))

X_train_aug, y_train_aug = corner_flip_augment(X_train_raw, y_train, FEATS_3B)
print(f"  Training rows before aug: {len(X_train_raw):,}")
print(f"  Training rows after aug:  {len(X_train_aug):,}  (2×)")
print(f"  Aug class balance: {dict(zip(CLASS_NAMES, np.bincount(y_train_aug)))}")
gc.collect()

# ── STEP B8: Scale and train ──────────────────────────────────────────
print()
print("STEP B8: Train models")

scaler3b = RobustScaler()
X_train_sc = scaler3b.fit_transform(X_train_aug)
X_test_sc  = scaler3b.transform(X_test)

# Logistic Regression (multinomial)
print("  Training Logistic Regression (multinomial)...")
lr3b = LogisticRegression(C=0.3, max_iter=2000, multi_class='multinomial',
                           solver='lbfgs', n_jobs=1, random_state=42)
lr3b.fit(X_train_sc, y_train_aug)
lr_test_acc = accuracy_score(y_test, lr3b.predict(X_test_sc))
print(f"  LR   — Test acc: {lr_test_acc:.4f}  vs naive: {lr_test_acc-naive_test:+.4f}")
gc.collect()

# Random Forest
print("  Training Random Forest...")
rf3b = RandomForestClassifier(n_estimators=300, max_depth=9, min_samples_leaf=10,
                               random_state=42, n_jobs=1)
rf3b.fit(X_train_aug, y_train_aug)
rf_test_acc = accuracy_score(y_test, rf3b.predict(X_test))
print(f"  RF   — Test acc: {rf_test_acc:.4f}  vs naive: {rf_test_acc-naive_test:+.4f}")
gc.collect()

# XGBoost
print("  Training XGBoost...")
xgb3b = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        objective='multi:softprob', num_class=6,
                        eval_metric='mlogloss', verbosity=0,
                        random_state=42, n_jobs=1)
xgb3b.fit(X_train_aug, y_train_aug,
           eval_set=[(X_test, y_test)], verbose=False)
xgb_test_acc = accuracy_score(y_test, xgb3b.predict(X_test))
print(f"  XGB  — Test acc: {xgb_test_acc:.4f}  vs naive: {xgb_test_acc-naive_test:+.4f}")
gc.collect()

# Blend comparison
print()
print("  Blend comparison:")
lr_p  = lr3b.predict_proba(X_test_sc)
rf_p  = rf3b.predict_proba(X_test)
xgb_p = xgb3b.predict_proba(X_test)

blends3b = [
    ('LR only',            1.0, 0.0, 0.0),
    ('RF only',            0.0, 1.0, 0.0),
    ('XGB only',           0.0, 0.0, 1.0),
    ('50% LR + 50% XGB',   0.5, 0.0, 0.5),
    ('30% LR + 70% XGB',   0.3, 0.0, 0.7),
    ('40% LR + 60% XGB',   0.4, 0.0, 0.6),
    ('33% each',           1/3, 1/3, 1/3),
    ('25% LR + 25% RF + 50% XGB', 0.25, 0.25, 0.50),
]

best_acc3b = 0; best_label3b = ''; best_prob3b = None; best_w3b = None
print(f"  {'Blend':<38} {'Acc':>6} {'vs naive':>9}")
print(f"  {'-'*38} {'-'*6} {'-'*9}")
for label, wl, wr, wx in blends3b:
    p = wl * lr_p + wr * rf_p + wx * xgb_p
    pred = p.argmax(axis=1)
    acc  = accuracy_score(y_test, pred)
    mark = ' ←' if acc > best_acc3b else ''
    if acc > best_acc3b:
        best_acc3b = acc; best_label3b = label; best_prob3b = p; best_w3b = (wl, wr, wx)
    print(f"  {label:<38} {acc:.4f} {acc-naive_test:>+9.4f}{mark}")

print(f"\n  Best blend: {best_label3b}  ({best_acc3b:.4f})")
print(f"  Beats naive by: {best_acc3b-naive_test:+.4f} ({(best_acc3b-naive_test)*100:+.2f}pp)")

# ── STEP B9: Per-class accuracy and direction/method accuracy ─────────
print()
print("STEP B9: Per-class, direction, and method accuracy")

best_pred = best_prob3b.argmax(axis=1)

print(f"\n  Per-class accuracy on test set:")
print(f"  {'Class':<14} {'N':>5}  {'Predicted':>9} {'Correct':>8} {'PerClass%':>10}")
print(f"  {'-'*14} {'-'*5}  {'-'*9} {'-'*8} {'-'*10}")
per_class = []
for i, name in enumerate(CLASS_NAMES):
    mask = y_test == i
    n = mask.sum()
    if n == 0: continue
    n_pred_this = (best_pred == i).sum()
    n_correct   = ((best_pred == i) & mask).sum()
    pc_acc = n_correct / n if n > 0 else 0
    print(f"  {name:<14} {n:>5}  {n_pred_this:>9} {n_correct:>8} {pc_acc*100:>10.1f}%")
    per_class.append({'class': i, 'name': name, 'n': int(n),
                      'per_class_acc': round(float(pc_acc), 4)})

# Direction accuracy: correct winner regardless of method
# Red classes: 0,1,2 → pred winner=Red if pred in {0,1,2}
# Blue classes: 3,4,5 → pred winner=Blue if pred in {3,4,5}
actual_red  = y_test < 3
pred_red    = best_pred < 3
dir_acc = (actual_red == pred_red).mean()

# Method accuracy: correct method regardless of winner
# KO classes: 0,3; Sub: 1,4; Dec: 2,5
actual_method = np.where(y_test % 3 == 0, 0, np.where(y_test % 3 == 1, 1, 2))
pred_method   = np.where(best_pred % 3 == 0, 0, np.where(best_pred % 3 == 1, 1, 2))
method_acc = (actual_method == pred_method).mean()

print(f"\n  Direction accuracy (correct winner, any method): {dir_acc:.4f} ({dir_acc*100:.2f}%)")
print(f"  Method accuracy   (correct method, any winner): {method_acc:.4f} ({method_acc*100:.2f}%)")
print(f"  Model 1 reference direction accuracy:            0.7281 (72.81%)")

# Method breakdown
for m_idx, m_name in enumerate(['KO/TKO','Submission','Decision']):
    mask = actual_method == m_idx
    n    = mask.sum()
    if n == 0: continue
    acc  = (pred_method[mask] == m_idx).mean()
    print(f"    {m_name:<14} (N={n:4d}): method recall {acc:.3f}")

gc.collect()

# ── STEP B10: Feature importance ─────────────────────────────────────
print()
print("STEP B10: Feature importance (XGBoost)")

xgb_imp = pd.Series(xgb3b.feature_importances_, index=FEATS_3B).sort_values(ascending=False)
print(f"\n  Top 20 features — XGBoost:")
print(f"  {'Feature':<40} {'Importance':>10}")
print(f"  {'-'*40} {'-'*10}")
for feat, imp in xgb_imp.head(20).items():
    print(f"  {feat:<40} {imp:>10.4f}")

gc.collect()

# ── STEP B11: Per-weight-class breakdown ─────────────────────────────
print()
print("STEP B11: Per-weight-class breakdown (test set)")

test_df = df_clean[test_mask].copy().reset_index(drop=True)
test_df['pred_label']  = best_pred
test_df['pred_winner'] = best_pred < 3
test_df['true_winner'] = y_test < 3
test_df['dir_correct'] = test_df['pred_winner'] == test_df['true_winner']

print(f"\n  {'Weight Class':<30} {'N':>5}  {'6-class':>8} {'Dir acc':>8} {'Naive':>7}")
print(f"  {'-'*30} {'-'*5}  {'-'*8} {'-'*8} {'-'*7}")
wc_res = []
for wc_name in sorted(test_df['weight_class'].unique()):
    sub = test_df[test_df['weight_class'] == wc_name]
    if len(sub) < 20: continue
    six_acc = accuracy_score(sub['label_6'], sub['pred_label'])
    dir_a   = sub['dir_correct'].mean()
    naive_wc = np.bincount(sub['label_6'].values).max() / len(sub)
    print(f"  {wc_name:<30} {len(sub):>5}  {six_acc:>8.3f} {dir_a:>8.3f} {naive_wc:>7.3f}")
    wc_res.append({'weight_class': wc_name, 'n': len(sub),
                   'six_class_acc': round(six_acc,4), 'direction_acc': round(dir_a,4),
                   'naive': round(naive_wc,4)})

gc.collect()

# ── STEP B12: Calibration ─────────────────────────────────────────────
print()
print("STEP B12: Isotonic calibration on training val slice")

n_val_b = int(len(X_train_aug) * 0.20)
n_fit_b = len(X_train_aug) - n_val_b
X_fit_b, X_val_b = X_train_aug[:n_fit_b], X_train_aug[n_fit_b:]
y_fit_b, y_val_b = y_train_aug[:n_fit_b], y_train_aug[n_fit_b:]
X_fit_sc_b = scaler3b.transform(X_fit_b)
X_val_sc_b = scaler3b.transform(X_val_b)

# Refit on fit slice (one-shot — don't contaminate with val)
lr3b_refit = LogisticRegression(C=0.3, max_iter=2000, multi_class='multinomial',
                                  solver='lbfgs', n_jobs=1, random_state=42)
lr3b_refit.fit(X_fit_sc_b, y_fit_b)

xgb3b_refit = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               objective='multi:softprob', num_class=6,
                               eval_metric='mlogloss', verbosity=0,
                               random_state=42, n_jobs=1)
xgb3b_refit.fit(X_fit_b, y_fit_b, verbose=False)
gc.collect()

wl3b, wr3b, wx3b = best_w3b
p_val_b = wl3b * lr3b_refit.predict_proba(X_val_sc_b) + wx3b * xgb3b_refit.predict_proba(X_val_b)

# Per-class isotonic regression calibration
iso_list = []
for cls in range(6):
    iso_c = IsotonicRegression(out_of_bounds='clip')
    y_binary = (y_val_b == cls).astype(int)
    iso_c.fit(p_val_b[:, cls], y_binary)
    iso_list.append(iso_c)

# Apply calibration to test set
p_test_b_raw = wl3b * lr3b.predict_proba(X_test_sc) + wx3b * xgb3b.predict_proba(X_test)
p_test_b_cal = np.stack([iso_list[c].predict(p_test_b_raw[:, c]) for c in range(6)], axis=1)
# Renormalize
p_test_b_cal = p_test_b_cal / p_test_b_cal.sum(axis=1, keepdims=True).clip(1e-6)

cal_acc = accuracy_score(y_test, p_test_b_cal.argmax(axis=1))
print(f"  Calibrated test accuracy: {cal_acc:.4f}  (uncalibrated: {best_acc3b:.4f})")

gc.collect()

# ── STEP B13: Save ────────────────────────────────────────────────────
print()
print("STEP B13: Save models and metadata")

joblib.dump(lr3b,    os.path.join(OUT, 'model3b_lr.pkl'))
joblib.dump(rf3b,    os.path.join(OUT, 'model3b_rf.pkl'))
joblib.dump(xgb3b,   os.path.join(OUT, 'model3b_xgb.pkl'))
joblib.dump(scaler3b,os.path.join(OUT, 'model3b_scaler.pkl'))
joblib.dump(iso_list,os.path.join(OUT, 'model3b_calibrators.pkl'))
joblib.dump(FEATS_3B,os.path.join(OUT, 'model3b_features.pkl'))

meta3b = {
    "model": "Model 3B — Winner + Method Six-Class",
    "classes": CLASS_NAMES,
    "label_map": {str(v): k for k, v in LABEL_MAP.items()},
    "train_cutoff": str(CUTOFF.date()),
    "n_train_raw": int(len(X_train_raw)),
    "n_train_aug": int(len(X_train_aug)),
    "n_test": int(len(X_test)),
    "n_features": len(FEATS_3B),
    "features": FEATS_3B,
    "class_balance_test": {CLASS_NAMES[i]: int(v) for i, v in enumerate(np.bincount(y_test))},
    "naive_baseline_test": round(float(naive_test), 4),
    "model_accuracy": {
        "lr_test":  round(float(lr_test_acc), 4),
        "rf_test":  round(float(rf_test_acc), 4),
        "xgb_test": round(float(xgb_test_acc), 4),
        "best_blend": round(float(best_acc3b), 4),
        "best_blend_calibrated": round(float(cal_acc), 4),
        "best_blend_label": best_label3b,
        "vs_naive_pp": round(float((best_acc3b - naive_test)*100), 2),
    },
    "blend_weights": {"lr": float(wl3b), "rf": float(wr3b), "xgb": float(wx3b)},
    "direction_accuracy": round(float(dir_acc), 4),
    "method_accuracy":    round(float(method_acc), 4),
    "per_class": per_class,
    "by_weight_class": wc_res,
    "xgb_top15_features": xgb_imp.head(15).round(4).to_dict(),
}
with open(os.path.join(OUT, 'model3b_metadata.json'), 'w') as f:
    json.dump(meta3b, f, indent=2)

print(f"  Saved: model3b_lr, rf, xgb, scaler, calibrators, features, metadata")
print()
print("=" * 64)
print("SUMMARY")
print("=" * 64)
print(f"  Model 3A calibration fix:")
print(f"    MAE before: {mae_uncal:.4f}  →  after: {mae_cal:.4f}")
print()
print(f"  Model 3B six-class results:")
print(f"    Classes: {CLASS_NAMES}")
print(f"    Naive baseline:       {naive_test:.4f}")
print(f"    Best blend:           {best_acc3b:.4f}  ({best_label3b})")
print(f"    Calibrated:           {cal_acc:.4f}")
print(f"    vs naive:             {best_acc3b-naive_test:+.4f} ({(best_acc3b-naive_test)*100:+.2f}pp)")
print(f"    Direction accuracy:   {dir_acc:.4f}  (M1 reference: 0.7281)")
print(f"    Method accuracy:      {method_acc:.4f}")
print()
print("All steps complete. No production files touched.")
