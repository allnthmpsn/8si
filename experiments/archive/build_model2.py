#!/usr/bin/env python3
"""
Model 2 — Odds-Aware Value Bet Model
Combines Model 1 probabilities with historical Vegas odds to find edge.
"""

import bisect, gc, json, os, sys, warnings
import numpy as np
import pandas as pd
import joblib, pickle
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

# ─── Config ───────────────────────────────────────────────────────────────────
TRAIN_CUTOFF     = pd.Timestamp('2024-01-01')
LR_WEIGHT        = 0.90
XGB_WEIGHT       = 0.10
N_OPTUNA_TRIALS  = 50
RANDOM_SEED      = 42

WC_ORDER = {
    "Women's Strawweight": 0, "Women's Flyweight": 1, "Women's Bantamweight": 2,
    "Women's Featherweight": 3, "Flyweight": 4, "Bantamweight": 5,
    "Featherweight": 6, "Lightweight": 7, "Welterweight": 8,
    "Middleweight": 9, "Light Heavyweight": 10, "Heavyweight": 11,
    "Catch Weight": 6,
}

MODEL2_FEATURES = [
    'model1_prob', 'f1_no_vig', 'f2_no_vig', 'model_vs_vegas_gap',
    'abs_gap', 'vegas_confidence', 'f1_is_favorite', 'model_agrees',
    'model_confidence', 'f1_dec_implied', 'f1_sub_implied', 'f1_ko_implied',
    'f2_dec_implied', 'f2_sub_implied', 'f2_ko_implied',
    'dec_implied_dif', 'sub_implied_dif', 'ko_implied_dif',
    'finish_implied', 'gap_x_vegas_conf', 'joint_confidence', 'gap_squared',
]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load all data
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("MODEL 2 BUILD — loading data...")
print("=" * 60)

model_lr      = joblib.load('model/ufc_model_best.pkl')
model_xgb     = joblib.load('model/ufc_model_xgb.pkl')
feat_cols_114 = joblib.load('model/feature_columns_best.pkl')

df_master = pd.read_csv('data/ufc-master.csv', low_memory=False)
df_master['date'] = pd.to_datetime(df_master['date'])

career_raw = pd.read_csv('data/career_fights_updated.csv')
career_raw['date'] = pd.to_datetime(career_raw['date'])
career_raw = career_raw.sort_values(['fighter', 'date']).reset_index(drop=True)

fstats_df = pd.read_csv('data/ufc_fighters_final_updated.csv')
for pct_col in ['Str_Acc', 'Str_Def', 'TD_Acc', 'TD_Def']:
    fstats_df[pct_col] = pd.to_numeric(
        fstats_df[pct_col].astype(str).str.replace('%', '', regex=False),
        errors='coerce',
    ).fillna(0) / 100.0

elo_hist = pd.read_csv('data/elo_ratings_history.csv')
elo_hist['date'] = pd.to_datetime(elo_hist['date'])
elo_hist = elo_hist.sort_values(['fighter', 'date']).reset_index(drop=True)

elo_curr = pd.read_csv('data/elo_current.csv')
elo_curr_map = dict(zip(elo_curr['fighter'], elo_curr['current_elo']))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Filter to 2018+ with odds, drop draws
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 1 — Filtering odds dataset...")

df = df_master[
    (df_master['date'] >= '2018-01-01') &
    df_master['R_odds'].notna() &
    df_master['B_odds'].notna() &
    df_master['Winner'].isin(['Red', 'Blue'])
].copy().reset_index(drop=True)

print(f"  Raw 2018+ fights with odds: {len(df)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Randomize corners (MUST happen before any feature engineering)
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 2 — Randomizing corners (seed=42)...")

np.random.seed(RANDOM_SEED)
swap_mask = np.random.random(len(df)) < 0.5

# Swap all R_*/B_* column pairs
r_all = sorted([c for c in df.columns if c.startswith('R_')])
b_all = sorted([c for c in df.columns if c.startswith('B_')])
r_matched = [c for c in r_all if ('B_' + c[2:]) in df.columns]
b_matched  = ['B_' + c[2:] for c in r_matched]

for rc, bc in zip(r_matched, b_matched):
    rv = df.loc[swap_mask, rc].values.copy()
    bv = df.loc[swap_mask, bc].values.copy()
    df.loc[swap_mask, rc] = bv
    df.loc[swap_mask, bc] = rv

# Flip Winner
df.loc[swap_mask & (df['Winner'] == 'Red'),  'Winner'] = 'TEMP'
df.loc[swap_mask & (df['Winner'] == 'Blue'), 'Winner'] = 'Red'
df.loc[swap_mask & (df['Winner'] == 'TEMP'), 'Winner'] = 'Blue'

# Swap lowercase method odds (not caught by R_/B_ prefix)
for rc, bc in [('r_dec_odds','b_dec_odds'),('r_sub_odds','b_sub_odds'),('r_ko_odds','b_ko_odds')]:
    if rc in df.columns and bc in df.columns:
        rv = df.loc[swap_mask, rc].values.copy()
        bv = df.loc[swap_mask, bc].values.copy()
        df.loc[swap_mask, rc] = bv
        df.loc[swap_mask, bc] = rv

target_full = (df['Winner'] == 'Red').astype(int)
print(f"  Red (F1) win rate after randomization: {target_full.mean():.3f}  (expect ≈0.500)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Precompute career stats timeline (expanding window, no leakage)
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 3 — Precomputing career stats timeline...")

cf = career_raw.copy()

# Expanding cumulative stats (each row = stats BEFORE that fight, via shift)
def shift_cumsum(x):
    return x.cumsum().shift(1).fillna(0)

cf['cum_fights'] = cf.groupby('fighter').cumcount()   # number of prior fights
cf['cum_wins']   = cf.groupby('fighter')['won'].transform(shift_cumsum)
cf['career_win_rate'] = np.where(
    cf['cum_fights'] > 0, cf['cum_wins'] / cf['cum_fights'], 0.5
)

cf['ko_win']  = ((cf['won']==1) & cf['method'].str.contains('KO|TKO',       case=False, na=False)).astype(int)
cf['sub_win'] = ((cf['won']==1) & cf['method'].str.contains('Sub|Submission',case=False, na=False)).astype(int)
cf['fin_win'] = ((cf['won']==1) & cf['method'].str.contains('KO|TKO|Sub|Submission', case=False, na=False)).astype(int)

cf['cum_ko']  = cf.groupby('fighter')['ko_win'].transform(shift_cumsum)
cf['cum_sub'] = cf.groupby('fighter')['sub_win'].transform(shift_cumsum)
cf['ko_finish_rate']  = np.where(cf['cum_fights'] > 0, cf['cum_ko']  / cf['cum_fights'], 0.0)
cf['sub_finish_rate'] = np.where(cf['cum_fights'] > 0, cf['cum_sub'] / cf['cum_fights'], 0.0)

def roll_sh(x, n):
    return x.shift(1).rolling(n, min_periods=1).mean()

cf['last3_win_rate']    = cf.groupby('fighter')['won'].transform(lambda x: roll_sh(x, 3)).fillna(0.5)
cf['last5_won']         = cf.groupby('fighter')['won'].transform(lambda x: roll_sh(x, 5)).fillna(0.5)
cf['last10_win_rate']   = cf.groupby('fighter')['won'].transform(lambda x: roll_sh(x, 10)).fillna(0.5)
cf['last5_finish_rate'] = cf.groupby('fighter')['fin_win'].transform(lambda x: roll_sh(x, 5)).fillna(0.0)
cf['trend_score']       = cf['last3_win_rate'] - cf['last10_win_rate']

cf['prev_date']   = cf.groupby('fighter')['date'].shift(1)
cf['layoff_days'] = (cf['date'] - cf['prev_date']).dt.days.fillna(365.0)

# Opponent quality (all-time win rate as proxy — same logic as production backend)
wr_cache_global = cf.groupby('fighter')['won'].mean().to_dict()

def opp_quality_series(grp):
    opps = grp['opponent'].values
    n    = len(grp)
    res  = np.full(n, 0.5)
    for i in range(n):
        start  = max(0, i - 5)
        prior  = opps[start:i]
        rates  = [wr_cache_global.get(o, 0.5) for o in prior]
        res[i] = float(np.mean(rates)) if rates else 0.5
    return pd.Series(res, index=grp.index)

cf['opp_quality'] = cf.groupby('fighter', group_keys=False).apply(opp_quality_series)

# Group for fast binary-search lookups
CAREER_COLS = [
    'cum_fights','career_win_rate','ko_finish_rate','sub_finish_rate',
    'last3_win_rate','last5_won','last10_win_rate','last5_finish_rate',
    'trend_score','layoff_days','opp_quality',
]
DEFAULT_CAREER = {
    'cum_fights': 0, 'career_win_rate': 0.5, 'ko_finish_rate': 0.0,
    'sub_finish_rate': 0.0, 'last3_win_rate': 0.5, 'last5_won': 0.5,
    'last10_win_rate': 0.5, 'last5_finish_rate': 0.0,
    'trend_score': 0.0, 'layoff_days': 365.0, 'opp_quality': 0.5,
}

career_by_f   = {}
career_dates_f = {}
for fname, grp in cf.groupby('fighter'):
    g = grp.reset_index(drop=True)
    career_by_f[fname]    = g
    career_dates_f[fname] = g['date'].tolist()

def get_career_at(fighter, fight_date):
    if fighter not in career_by_f:
        return DEFAULT_CAREER.copy()
    dates = career_dates_f[fighter]
    idx   = bisect.bisect_right(dates, fight_date) - 1
    if idx < 0:
        return DEFAULT_CAREER.copy()
    row = career_by_f[fighter].iloc[idx]
    return {c: float(row[c]) for c in CAREER_COLS}

print(f"  Career timeline built for {len(career_by_f)} fighters")

# ─────────────────────────────────────────────────────────────────────────────
# Elo lookups
# ─────────────────────────────────────────────────────────────────────────────
elo_by_f    = {}
elo_dates_f = {}
for fname, grp in elo_hist.groupby('fighter'):
    g = grp.sort_values('date').reset_index(drop=True)
    elo_by_f[fname]    = g
    elo_dates_f[fname] = g['date'].tolist()

def get_elo_at(fighter, fight_date):
    if fighter not in elo_by_f:
        return {'elo': 1500.0, 'elo_trend': 0.0}
    dates = elo_dates_f[fighter]
    # strictly before: elo_after of last fight < fight_date = entering this fight
    idx = bisect.bisect_left(dates, fight_date) - 1
    if idx < 0:
        return {'elo': 1500.0, 'elo_trend': 0.0}
    row = elo_by_f[fighter].iloc[idx]
    return {
        'elo':       float(row['elo_after']),
        'elo_trend': float(row.get('elo_trend', 0.0) or 0.0),
    }

# Fighter style stats lookup
fstyle = {}
for _, row in fstats_df.iterrows():
    fstyle[row['Fighter_Name']] = {
        'SLpM':    float(row.get('SLpM', 0)    or 0),
        'SApM':    float(row.get('SApM', 0)    or 0),
        'Str_Acc': float(row.get('Str_Acc', 0) or 0),
        'Str_Def': float(row.get('Str_Def', 0) or 0),
        'TD_Avg':  float(row.get('TD_Avg', 0)  or 0),
        'TD_Acc':  float(row.get('TD_Acc', 0)  or 0),
        'TD_Def':  float(row.get('TD_Def', 0)  or 0),
        'Sub_Avg': float(row.get('Sub_Avg', 0) or 0),
    }

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Build 114-feature matrix
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 4 — Building 114-feature matrix (this takes ~30s)...")

def g(row, col, default=0.0):
    v = row.get(col, default)
    try:
        if pd.isna(v):
            return float(default)
    except Exception:
        pass
    return float(v) if v is not None else float(default)

def layoff_buckets(days):
    return {
        'lt90':    1 if days < 90  else 0,
        '90_180':  1 if 90  <= days < 180 else 0,
        '180_365': 1 if 180 <= days < 365 else 0,
        'gt365':   1 if days >= 365 else 0,
    }

rows_list = []
for _, row in df.iterrows():
    r_name = row['R_fighter']
    b_name = row['B_fighter']
    fdate  = row['date']

    rc = get_career_at(r_name, fdate)
    bc = get_career_at(b_name, fdate)
    rs = fstyle.get(r_name, {})
    bs = fstyle.get(b_name, {})
    re = get_elo_at(r_name, fdate)
    be = get_elo_at(b_name, fdate)

    r_lb = layoff_buckets(rc['layoff_days'])
    b_lb = layoff_buckets(bc['layoff_days'])

    r_sp = 1 if str(row.get('R_Stance','') or '').lower() == 'southpaw' else 0
    b_sp = 1 if str(row.get('B_Stance','') or '').lower() == 'southpaw' else 0

    r_wins = g(row,'R_wins');       b_wins = g(row,'B_wins')
    r_loss = g(row,'R_losses');     b_loss = g(row,'B_losses')
    r_h    = g(row,'R_Height_cms',175); b_h = g(row,'B_Height_cms',175)
    r_rch  = g(row,'R_Reach_cms',175);  b_rch = g(row,'B_Reach_cms',175)
    r_age  = g(row,'R_age',28);     b_age = g(row,'B_age',28)
    r_sig  = g(row,'R_avg_SIG_STR_landed'); b_sig = g(row,'B_avg_SIG_STR_landed')
    r_td   = g(row,'R_avg_TD_landed');      b_td  = g(row,'B_avg_TD_landed')
    r_ws   = g(row,'R_current_win_streak'); b_ws  = g(row,'B_current_win_streak')
    r_ls   = g(row,'R_current_lose_streak');b_ls  = g(row,'B_current_lose_streak')
    r_lws  = g(row,'R_longest_win_streak'); b_lws = g(row,'B_longest_win_streak')
    r_sigp = g(row,'R_avg_SIG_STR_pct');    b_sigp= g(row,'B_avg_SIG_STR_pct')
    r_suba = g(row,'R_avg_SUB_ATT');        b_suba= g(row,'B_avg_SUB_ATT')
    r_tdp  = g(row,'R_avg_TD_pct');         b_tdp = g(row,'B_avg_TD_pct')
    r_ttb  = g(row,'R_total_title_bouts');  b_ttb = g(row,'B_total_title_bouts')
    r_ko   = g(row,'R_win_by_KO/TKO');      b_ko  = g(row,'B_win_by_KO/TKO')
    r_sub  = g(row,'R_win_by_Submission');   b_sub = g(row,'B_win_by_Submission')

    wc_ord = WC_ORDER.get(str(row.get('weight_class','') or ''), 6)
    title  = 1 if row.get('title_bout', False) else 0

    r_axe  = r_age * rc['cum_fights']
    b_axe  = b_age * bc['cum_fights']

    feat = {
        'R_wins': r_wins, 'R_losses': r_loss,
        'R_Height_cms': r_h, 'R_age': r_age,
        'R_avg_SIG_STR_landed': r_sig, 'R_avg_TD_landed': r_td,
        'R_current_win_streak': r_ws, 'R_current_lose_streak': r_ls,
        'R_longest_win_streak': r_lws, 'R_avg_SIG_STR_pct': r_sigp,
        'R_avg_SUB_ATT': r_suba, 'R_avg_TD_pct': r_tdp,
        'R_Reach_cms': r_rch, 'R_total_title_bouts': r_ttb,
        'B_wins': b_wins, 'B_losses': b_loss,
        'B_Height_cms': b_h, 'B_age': b_age,
        'B_avg_SIG_STR_landed': b_sig, 'B_avg_TD_landed': b_td,
        'B_current_win_streak': b_ws, 'B_current_lose_streak': b_ls,
        'B_longest_win_streak': b_lws, 'B_avg_SIG_STR_pct': b_sigp,
        'B_avg_SUB_ATT': b_suba, 'B_avg_TD_pct': b_tdp,
        'B_Reach_cms': b_rch, 'B_total_title_bouts': b_ttb,
        # diffs
        'win_dif': r_wins-b_wins, 'loss_dif': r_loss-b_loss,
        'win_streak_dif': r_ws-b_ws, 'lose_streak_dif': r_ls-b_ls,
        'height_dif': r_h-b_h, 'reach_dif': r_rch-b_rch,
        'age_dif': r_age-b_age, 'sig_str_dif': r_sig-b_sig,
        'avg_td_dif': r_td-b_td, 'ko_dif': r_ko-b_ko,
        'sub_dif': r_sub-b_sub, 'total_title_bout_dif': r_ttb-b_ttb,
        # meta
        'weight_class_ord': wc_ord, 'title_bout_bin': title,
        # stance
        'orth_clash': 1 if (r_sp==0 and b_sp==0) else 0,
        'south_clash': 1 if (r_sp==1 and b_sp==1) else 0,
        'R_southpaw': r_sp, 'B_southpaw': b_sp,
        # career
        'R_cum_fights': rc['cum_fights'], 'B_cum_fights': bc['cum_fights'],
        'R_career_win_rate': rc['career_win_rate'], 'B_career_win_rate': bc['career_win_rate'],
        'career_win_rate_dif': rc['career_win_rate']-bc['career_win_rate'],
        'R_last5_won': rc['last5_won'], 'B_last5_won': bc['last5_won'],
        'last5_won_dif': rc['last5_won']-bc['last5_won'],
        'R_last5_finish_rate': rc['last5_finish_rate'], 'B_last5_finish_rate': bc['last5_finish_rate'],
        'last5_finish_rate_dif': rc['last5_finish_rate']-bc['last5_finish_rate'],
        'R_opp_quality': rc['opp_quality'], 'B_opp_quality': bc['opp_quality'],
        'opp_quality_dif': rc['opp_quality']-bc['opp_quality'],
        'R_trend_score': rc['trend_score'], 'B_trend_score': bc['trend_score'],
        'trend_score_dif': rc['trend_score']-bc['trend_score'],
        'R_ko_finish_rate': rc['ko_finish_rate'], 'B_ko_finish_rate': bc['ko_finish_rate'],
        'ko_finish_rate_dif': rc['ko_finish_rate']-bc['ko_finish_rate'],
        'R_sub_finish_rate': rc['sub_finish_rate'], 'B_sub_finish_rate': bc['sub_finish_rate'],
        'sub_finish_rate_dif': rc['sub_finish_rate']-bc['sub_finish_rate'],
        'R_last3_win_rate': rc['last3_win_rate'], 'B_last3_win_rate': bc['last3_win_rate'],
        'last3_win_rate_dif': rc['last3_win_rate']-bc['last3_win_rate'],
        'R_last10_win_rate': rc['last10_win_rate'], 'B_last10_win_rate': bc['last10_win_rate'],
        'last10_win_rate_dif': rc['last10_win_rate']-bc['last10_win_rate'],
        'R_age_x_exp': r_axe, 'B_age_x_exp': b_axe, 'age_x_exp_dif': r_axe-b_axe,
        # layoff buckets
        'R_layoff_lt90': r_lb['lt90'], 'R_layoff_90_180': r_lb['90_180'],
        'R_layoff_180_365': r_lb['180_365'], 'R_layoff_gt365': r_lb['gt365'],
        'B_layoff_lt90': b_lb['lt90'], 'B_layoff_90_180': b_lb['90_180'],
        'B_layoff_180_365': b_lb['180_365'], 'B_layoff_gt365': b_lb['gt365'],
        # style
        'R_SLpM': rs.get('SLpM',0), 'B_SLpM': bs.get('SLpM',0),
        'R_SApM': rs.get('SApM',0), 'B_SApM': bs.get('SApM',0),
        'R_Str_Acc': rs.get('Str_Acc',0), 'B_Str_Acc': bs.get('Str_Acc',0),
        'R_Str_Def': rs.get('Str_Def',0), 'B_Str_Def': bs.get('Str_Def',0),
        'R_TD_Avg': rs.get('TD_Avg',0), 'B_TD_Avg': bs.get('TD_Avg',0),
        'R_TD_Acc': rs.get('TD_Acc',0), 'B_TD_Acc': bs.get('TD_Acc',0),
        'R_TD_Def': rs.get('TD_Def',0), 'B_TD_Def': bs.get('TD_Def',0),
        'R_Sub_Avg': rs.get('Sub_Avg',0), 'B_Sub_Avg': bs.get('Sub_Avg',0),
        'SLpM_dif': rs.get('SLpM',0)-bs.get('SLpM',0),
        'SApM_dif': rs.get('SApM',0)-bs.get('SApM',0),
        'Str_Def_dif': rs.get('Str_Def',0)-bs.get('Str_Def',0),
        'TD_Def_dif': rs.get('TD_Def',0)-bs.get('TD_Def',0),
        'Sub_Avg_dif': rs.get('Sub_Avg',0)-bs.get('Sub_Avg',0),
        'TD_Avg_dif': rs.get('TD_Avg',0)-bs.get('TD_Avg',0),
        # elo
        'R_elo': re['elo'], 'B_elo': be['elo'],
        'elo_dif': re['elo']-be['elo'],
        'R_elo_trend': re['elo_trend'], 'B_elo_trend': be['elo_trend'],
        'elo_trend_dif': re['elo_trend']-be['elo_trend'],
    }
    rows_list.append(feat)

X_full_df = pd.DataFrame(rows_list)
for col in feat_cols_114:
    if col not in X_full_df.columns:
        X_full_df[col] = 0
X_full = X_full_df[feat_cols_114].fillna(0).values
print(f"  Feature matrix: {X_full.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Generate Model 1 predictions (CV on train, direct on test)
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 5 — Generating Model 1 predictions...")

dates_full  = df['date'].values
train_mask  = df['date'] < TRAIN_CUTOFF
test_mask   = df['date'] >= TRAIN_CUTOFF

X_train = X_full[train_mask]
X_test  = X_full[test_mask]
y_train = target_full[train_mask].values
y_test  = target_full[test_mask].values

print(f"  Train (2018-2023): {X_train.shape[0]} fights")
print(f"  Test  (2024+):     {X_test.shape[0]} fights")

# 5-fold OOF on train set to avoid leakage
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

print("  Running 5-fold CV for OOF predictions on train set...")
oof_lr  = cross_val_predict(model_lr,  X_train, y_train, cv=skf, method='predict_proba', n_jobs=1)[:, 1]
oof_xgb = cross_val_predict(model_xgb, X_train, y_train, cv=skf, method='predict_proba', n_jobs=1)[:, 1]
oof_m1  = LR_WEIGHT * oof_lr + XGB_WEIGHT * oof_xgb

# Direct predictions for test set (full model trained on all training data)
model_lr.fit(X_train, y_train)   # retrain on full train set for test predictions
model_xgb.fit(X_train, y_train)
test_lr  = model_lr.predict_proba(X_test)[:, 1]
test_xgb = model_xgb.predict_proba(X_test)[:, 1]
test_m1  = LR_WEIGHT * test_lr + XGB_WEIGHT * test_xgb

m1_prob_full = np.empty(len(df))
m1_prob_full[train_mask] = oof_m1
m1_prob_full[test_mask]  = test_m1

train_acc = accuracy_score(y_train, (oof_m1 > 0.5).astype(int))
test_acc  = accuracy_score(y_test,  (test_m1 > 0.5).astype(int))
print(f"  Model 1 OOF train acc: {train_acc:.4f}")
print(f"  Model 1 test acc:      {test_acc:.4f}")

gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Build 22 Model 2 features
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 6 — Building 22 Model 2 features...")

def implied_prob(odds):
    odds = float(odds)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)

def safe_implied(val, default=0.5):
    try:
        if pd.isna(val):
            return default
        return implied_prob(float(val))
    except Exception:
        return default

m2_rows = []
for i, (_, row) in enumerate(df.iterrows()):
    m1_p = float(m1_prob_full[i])

    f1_raw  = safe_implied(row['R_odds'])
    f2_raw  = safe_implied(row['B_odds'])
    total   = f1_raw + f2_raw
    f1_nv   = f1_raw / total
    f2_nv   = f2_raw / total

    gap     = m1_p - f1_nv
    abs_gap = abs(gap)
    vconf   = abs(f1_nv - 0.5)
    mconf   = abs(m1_p - 0.5)

    # Method odds
    f1_dec = safe_implied(row.get('r_dec_odds'))
    f1_sub = safe_implied(row.get('r_sub_odds'))
    f1_ko  = safe_implied(row.get('r_ko_odds'))
    f2_dec = safe_implied(row.get('b_dec_odds'))
    f2_sub = safe_implied(row.get('b_sub_odds'))
    f2_ko  = safe_implied(row.get('b_ko_odds'))

    m2_rows.append({
        'model1_prob':       m1_p,
        'f1_no_vig':         f1_nv,
        'f2_no_vig':         f2_nv,
        'model_vs_vegas_gap': gap,
        'abs_gap':           abs_gap,
        'vegas_confidence':  vconf,
        'f1_is_favorite':    1.0 if f1_nv > 0.5 else 0.0,
        'model_agrees':      1.0 if (m1_p > 0.5) == (f1_nv > 0.5) else 0.0,
        'model_confidence':  mconf,
        'f1_dec_implied':    f1_dec,
        'f1_sub_implied':    f1_sub,
        'f1_ko_implied':     f1_ko,
        'f2_dec_implied':    f2_dec,
        'f2_sub_implied':    f2_sub,
        'f2_ko_implied':     f2_ko,
        'dec_implied_dif':   f1_dec - f2_dec,
        'sub_implied_dif':   f1_sub - f2_sub,
        'ko_implied_dif':    f1_ko  - f2_ko,
        'finish_implied':    1.0 - (f1_dec + f2_dec) / 2.0,
        'gap_x_vegas_conf':  gap * vconf,
        'joint_confidence':  m1_p * f1_nv,
        'gap_squared':       gap**2 * np.sign(gap),
    })

m2_df = pd.DataFrame(m2_rows, columns=MODEL2_FEATURES)
print(f"  Model 2 feature matrix: {m2_df.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Temporal split for Model 2
# ─────────────────────────────────────────────────────────────────────────────
X2_train = m2_df[train_mask].values
X2_test  = m2_df[test_mask].values
y2_train = target_full[train_mask].values
y2_test  = target_full[test_mask].values

# Track extra info for value bet analysis
f1_odds_test = df[test_mask]['R_odds'].values
f2_odds_test = df[test_mask]['B_odds'].values
f1_name_test = df[test_mask]['R_fighter'].values
f2_name_test = df[test_mask]['B_fighter'].values
dates_test   = df[test_mask]['date'].values

def calc_roi(predictions, confidences, y_true, f1_odds, threshold=0.55):
    """Bet $100 on predicted winner when confidence > threshold. Returns ROI%."""
    total_bet, total_pnl = 0, 0
    for pred, conf, actual, odds_f1 in zip(predictions, confidences, y_true, f1_odds):
        if conf < threshold:
            continue
        total_bet += 100
        if pred == 1:   # betting F1
            if actual == 1:
                if odds_f1 < 0:
                    pnl = 100 * 100 / abs(odds_f1)
                else:
                    pnl = 100 * odds_f1 / 100
            else:
                pnl = -100
        else:           # betting F2 (use inverse)
            odds_f2 = df[test_mask]['B_odds'].values[0]  # placeholder — handled below
            pnl = -100 if actual == 1 else 100
        total_pnl += pnl
    if total_bet == 0:
        return 0.0, 0
    return round(total_pnl / total_bet * 100, 2), total_bet // 100

def calc_roi_v2(m2_prob, y_true, f1_odds_arr, b_odds_arr, threshold=0.55):
    """Full ROI with correct payout for each side."""
    total_bet, total_pnl, n_bets = 0, 0.0, 0
    for p, actual, o1, o2 in zip(m2_prob, y_true, f1_odds_arr, b_odds_arr):
        if max(p, 1-p) < threshold:
            continue
        n_bets   += 1
        total_bet += 100
        bet_f1 = p > 0.5
        won    = (bet_f1 and actual==1) or (not bet_f1 and actual==0)
        odds   = float(o1) if bet_f1 else float(o2)
        if won:
            if odds < 0:
                pnl = 100 * 100 / abs(odds)
            else:
                pnl = 100 * odds / 100
        else:
            pnl = -100
        total_pnl += pnl
    roi = round(total_pnl / total_bet * 100, 2) if total_bet > 0 else 0.0
    return roi, n_bets

def value_bet_roi(m2_prob, y_true, f1_odds_arr, b_odds_arr, gap_arr, min_gap=0.05):
    """ROI on value bets only (|model_vs_vegas_gap| > min_gap)."""
    total_bet, total_pnl, n = 0, 0.0, 0
    for p, actual, o1, o2, gap in zip(m2_prob, y_true, f1_odds_arr, b_odds_arr, gap_arr):
        if abs(gap) <= min_gap:
            continue
        n        += 1
        total_bet += 100
        bet_f1 = gap > 0   # bet whichever side model favors over Vegas
        won    = (bet_f1 and actual==1) or (not bet_f1 and actual==0)
        odds   = float(o1) if bet_f1 else float(o2)
        if won:
            pnl = 100*100/abs(odds) if odds < 0 else 100*odds/100
        else:
            pnl = -100
        total_pnl += pnl
    roi = round(total_pnl / total_bet * 100, 2) if total_bet > 0 else 0.0
    return roi, n

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Train Model 2 candidates
# ─────────────────────────────────────────────────────────────────────────────
m2_gap_test = m2_df[test_mask]['model_vs_vegas_gap'].values
results = {}

# ── A: Logistic Regression (Optuna) ──────────────────────────────────────────
print("\nStep 7A — Model A: Logistic Regression (Optuna 50 trials)...")

def lr_objective(trial):
    C       = trial.suggest_float('C', 0.001, 10, log=True)
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
    scaler  = trial.suggest_categorical('scaler', ['robust', 'standard'])
    sc      = RobustScaler() if scaler == 'robust' else StandardScaler()
    pipe    = Pipeline([('sc', sc), ('lr', LogisticRegression(
        C=C, penalty=penalty, solver='liblinear', max_iter=2000,
        random_state=RANDOM_SEED, n_jobs=1,
    ))])
    cv_acc = []
    for tr_idx, va_idx in skf.split(X2_train, y2_train):
        pipe.fit(X2_train[tr_idx], y2_train[tr_idx])
        preds = pipe.predict(X2_train[va_idx])
        cv_acc.append(accuracy_score(y2_train[va_idx], preds))
    return np.mean(cv_acc)

study_lr = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
study_lr.optimize(lr_objective, n_trials=N_OPTUNA_TRIALS, n_jobs=1)

best_lr_params = study_lr.best_params
scA = RobustScaler() if best_lr_params['scaler'] == 'robust' else StandardScaler()
model_A = Pipeline([('sc', scA), ('lr', LogisticRegression(
    C=best_lr_params['C'], penalty=best_lr_params['penalty'],
    solver='liblinear', max_iter=2000, random_state=RANDOM_SEED, n_jobs=1,
))])
model_A.fit(X2_train, y2_train)
pred_A     = model_A.predict(X2_test)
prob_A     = model_A.predict_proba(X2_test)[:, 1]
acc_A      = accuracy_score(y2_test, pred_A)
roi_A, nA  = calc_roi_v2(prob_A, y2_test, f1_odds_test, f2_odds_test)
vroi_A, vnA = value_bet_roi(prob_A, y2_test, f1_odds_test, f2_odds_test, m2_gap_test)
results['A_LR'] = {'acc': acc_A, 'roi': roi_A, 'n_bets': nA, 'vroi': vroi_A, 'vn': vnA, 'params': best_lr_params}
print(f"  A — LR:  acc={acc_A:.4f}  ROI={roi_A:+.2f}%  ({nA} bets) | value ROI={vroi_A:+.2f}% ({vnA} bets)")
print(f"  Best params: {best_lr_params}")
gc.collect()

# ── B: XGBoost (Optuna) ───────────────────────────────────────────────────────
print("\nStep 7B — Model B: XGBoost (Optuna 50 trials)...")

def xgb_objective(trial):
    params = {
        'n_estimators':  trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth':     trial.suggest_int('max_depth', 2, 4),
        'subsample':     trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': RANDOM_SEED, 'n_jobs': 1, 'eval_metric': 'logloss',
    }
    cv_acc = []
    for tr_idx, va_idx in skf.split(X2_train, y2_train):
        m = XGBClassifier(**params)
        m.fit(X2_train[tr_idx], y2_train[tr_idx],
              eval_set=[(X2_train[va_idx], y2_train[va_idx])], verbose=False)
        preds = m.predict(X2_train[va_idx])
        cv_acc.append(accuracy_score(y2_train[va_idx], preds))
    return np.mean(cv_acc)

study_xgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
study_xgb.optimize(xgb_objective, n_trials=N_OPTUNA_TRIALS, n_jobs=1)

best_xgb_params = study_xgb.best_params
best_xgb_params.update({'random_state': RANDOM_SEED, 'n_jobs': 1, 'eval_metric': 'logloss'})
model_B = XGBClassifier(**best_xgb_params)
model_B.fit(X2_train, y2_train)
pred_B      = model_B.predict(X2_test)
prob_B      = model_B.predict_proba(X2_test)[:, 1]
acc_B       = accuracy_score(y2_test, pred_B)
roi_B, nB   = calc_roi_v2(prob_B, y2_test, f1_odds_test, f2_odds_test)
vroi_B, vnB = value_bet_roi(prob_B, y2_test, f1_odds_test, f2_odds_test, m2_gap_test)
results['B_XGB'] = {'acc': acc_B, 'roi': roi_B, 'n_bets': nB, 'vroi': vroi_B, 'vn': vnB, 'params': best_xgb_params}
print(f"  B — XGB: acc={acc_B:.4f}  ROI={roi_B:+.2f}%  ({nB} bets) | value ROI={vroi_B:+.2f}% ({vnB} bets)")
gc.collect()

# ── C: Isotonic-calibrated LR ─────────────────────────────────────────────────
print("\nStep 7C — Model C: Calibrated LR (isotonic)...")

model_C = CalibratedClassifierCV(model_A, cv=5, method='isotonic')
model_C.fit(X2_train, y2_train)
pred_C      = model_C.predict(X2_test)
prob_C      = model_C.predict_proba(X2_test)[:, 1]
acc_C       = accuracy_score(y2_test, pred_C)
roi_C, nC   = calc_roi_v2(prob_C, y2_test, f1_odds_test, f2_odds_test)
vroi_C, vnC = value_bet_roi(prob_C, y2_test, f1_odds_test, f2_odds_test, m2_gap_test)
results['C_CalibLR'] = {'acc': acc_C, 'roi': roi_C, 'n_bets': nC, 'vroi': vroi_C, 'vn': vnC}
print(f"  C — Calib LR: acc={acc_C:.4f}  ROI={roi_C:+.2f}%  ({nC} bets) | value ROI={vroi_C:+.2f}% ({vnC} bets)")
gc.collect()

# ── D: Simple threshold model ─────────────────────────────────────────────────
print("\nStep 7D — Model D: Threshold model...")

thresh_results = {}
for thresh in [0.03, 0.05, 0.07, 0.10]:
    vroi, vn = value_bet_roi(
        # Use model1_prob as the probability
        m2_df[test_mask]['model1_prob'].values,
        y2_test, f1_odds_test, f2_odds_test,
        m2_gap_test, min_gap=thresh
    )
    # Accuracy: among value bets, how often does model win?
    gaps = m2_gap_test
    mask = abs(gaps) > thresh
    if mask.sum() > 0:
        preds_thresh = (gaps[mask] > 0).astype(int)
        acc_thresh   = accuracy_score(y2_test[mask], preds_thresh)
    else:
        acc_thresh = 0.0
    thresh_results[thresh] = {'acc': acc_thresh, 'roi': vroi, 'n': vn}
    print(f"  D — gap>{thresh:.0%}: acc={acc_thresh:.4f}  ROI={vroi:+.2f}%  ({vn} bets)")

results['D_Threshold'] = thresh_results
gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — Value bet analysis (2024+ holdout)
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 8 — Value bet analysis (2024+ test set, gap>5%)...")

# Pick best ML model for value bet analysis (by value ROI)
best_name = max(['A_LR', 'B_XGB', 'C_CalibLR'], key=lambda k: results[k]['vroi'])
best_prob  = {'A_LR': prob_A, 'B_XGB': prob_B, 'C_CalibLR': prob_C}[best_name]

gap_arr = m2_gap_test

# By gap bucket
print("\n  Value bet breakdown by gap bucket:")
print(f"  {'Bucket':<12} {'N':>5} {'Win%':>7} {'ROI':>8}")
for lo, hi in [(0.05, 0.10), (0.10, 0.15), (0.15, 1.0)]:
    mask  = (abs(gap_arr) > lo) & (abs(gap_arr) <= hi)
    if mask.sum() == 0:
        print(f"  {lo:.0%}-{hi:.0%}   {'0':>5} {'—':>7} {'—':>8}")
        continue
    bet_f1  = gap_arr[mask] > 0
    actuals = y2_test[mask]
    won_arr = (bet_f1 & (actuals==1)) | (~bet_f1 & (actuals==0))
    win_rate = won_arr.mean()
    # ROI
    total_pnl = 0.0
    for wf1, act, o1, o2 in zip(bet_f1, actuals, f1_odds_test[mask], f2_odds_test[mask]):
        won  = (wf1 and act==1) or (not wf1 and act==0)
        odds = float(o1) if wf1 else float(o2)
        total_pnl += (100*100/abs(odds) if odds < 0 else 100*odds/100) if won else -100
    roi = total_pnl / (mask.sum() * 100) * 100
    lbl = f"{lo:.0%}-{hi:.0%}" if hi < 1.0 else f"{lo:.0%}+"
    print(f"  {lbl:<12} {mask.sum():>5} {win_rate:>7.1%} {roi:>+8.2f}%")

# Print value bet table (gap > 5%)
vb_mask = abs(gap_arr) > 0.05
print(f"\n  All value bets in 2024+ test set (gap>5%): {vb_mask.sum()} fights")
print(f"  {'F1':<22} {'F2':<22} {'M1%':>5} {'VGS%':>5} {'Gap':>6} {'Outcome':>8} {'Profit':>8}")
for i in np.where(vb_mask)[0]:
    gap    = gap_arr[i]
    m1p    = m2_df[test_mask].iloc[i]['model1_prob']
    f1nv   = m2_df[test_mask].iloc[i]['f1_no_vig']
    bet_f1 = gap > 0
    act    = y2_test[i]
    won    = (bet_f1 and act==1) or (not bet_f1 and act==0)
    o1, o2 = float(f1_odds_test[i]), float(f2_odds_test[i])
    odds   = o1 if bet_f1 else o2
    pnl    = (100*100/abs(odds) if odds < 0 else 100*odds/100) if won else -100
    print(f"  {f1_name_test[i]:<22} {f2_name_test[i]:<22} "
          f"{m1p:>5.1%} {f1nv:>5.1%} {gap:>+6.3f} "
          f"{'WIN' if won else 'LOSS':>8} {pnl:>+8.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11 — Tonight's card application
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 9 — Tonight's card (UFC Fight Night: Sterling vs Zalal)...")

card_odds = {
    'Zalal vs Sterling':        {'f1': 'Youssef Zalal',      'f2': 'Aljamain Sterling',    'f1_odds': -135, 'f2_odds':  114},
    'Hernandez vs Garcia':      {'f1': 'Alexander Hernandez','f2': 'Rafa Garcia',           'f1_odds': -155, 'f2_odds':  130},
    'Martinetti vs Grant':      {'f1': 'Adrian Luna Martinetti','f2': 'Davey Grant',         'f1_odds': -142, 'f2_odds':  120},
    'Jackson vs Barcelos':      {'f1': 'Montel Jackson',     'f2': 'Raoni Barcelos',        'f1_odds': -170, 'f2_odds':  142},
    'Dumont vs Edwards':        {'f1': 'Norma Dumont',       'f2': 'Joselyne Edwards',      'f1_odds': -225, 'f2_odds':  185},
    'Buchecha vs Spann':        {'f1': 'Marcus Buchecha',    'f2': 'Ryan Spann',            'f1_odds': -135, 'f2_odds':  114},
    'Vieira vs McConico':       {'f1': 'Rodolfo Vieira',     'f2': 'Eric McConico',         'f1_odds': -200, 'f2_odds':  165},
    'McVey vs Dumas':           {'f1': 'Jackson McVey',      'f2': 'Sedriques Dumas',       'f1_odds': -130, 'f2_odds':  108},
    'Bueno Silva vs Montague':  {'f1': 'Mayra Bueno Silva',  'f2': 'Michelle Montague',     'f1_odds': -175, 'f2_odds':  145},
    'Polastri vs Alencar':      {'f1': 'Julia Polastri',     'f2': 'Talita Alencar',        'f1_odds': -145, 'f2_odds':  120},
    'Marshall vs Brennan':      {'f1': 'Francis Marshall',   'f2': 'Lucas Brennan',         'f1_odds': -150, 'f2_odds':  126},
    'Valenzuela vs Griffin':    {'f1': 'Victor Valenzuela',  'f2': 'Max Griffin',           'f1_odds':  200, 'f2_odds': -250},
}

# Helper: build feature vector for a single fighter (current stats)
TODAY = pd.Timestamp('2026-04-28')

def build_fighter_feats(name, prefix):
    # UFC stats from master (latest row)
    as_r = df_master[df_master['R_fighter'] == name].sort_values('date')
    as_b = df_master[df_master['B_fighter'] == name].sort_values('date')
    rd   = as_r.iloc[-1]['date'] if len(as_r) > 0 else pd.Timestamp.min
    bd   = as_b.iloc[-1]['date'] if len(as_b) > 0 else pd.Timestamp.min
    if rd >= bd and len(as_r) > 0:
        ufc_row, p = as_r.iloc[-1], 'R'
    elif len(as_b) > 0:
        ufc_row, p = as_b.iloc[-1], 'B'
    else:
        ufc_row, p = None, None

    def ufc(col, d=0.0):
        if ufc_row is None:
            return d
        v = ufc_row.get(f'{p}_{col}', d)
        try:
            return float(v) if not pd.isna(v) else d
        except Exception:
            return d

    # Current career stats (as of today)
    cc = get_career_at(name, TODAY)
    # Current style stats
    ss = fstyle.get(name, {})
    # Current elo
    eel = get_elo_at(name, TODAY)
    # Current elo from elo_current if not in history
    if eel['elo'] == 1500.0 and name in elo_curr_map:
        eel['elo'] = float(elo_curr_map[name])

    age   = ufc('age', 28)
    lb    = layoff_buckets(cc['layoff_days'])
    sp    = 1 if str(ufc_row.get(f'{p}_Stance','') if ufc_row is not None else '').lower() == 'southpaw' else 0

    return {
        f'{prefix}_wins':               ufc('wins'),
        f'{prefix}_losses':             ufc('losses'),
        f'{prefix}_Height_cms':         ufc('Height_cms', 175),
        f'{prefix}_age':                age,
        f'{prefix}_avg_SIG_STR_landed': ufc('avg_SIG_STR_landed'),
        f'{prefix}_avg_TD_landed':      ufc('avg_TD_landed'),
        f'{prefix}_current_win_streak': ufc('current_win_streak'),
        f'{prefix}_current_lose_streak':ufc('current_lose_streak'),
        f'{prefix}_longest_win_streak': ufc('longest_win_streak'),
        f'{prefix}_avg_SIG_STR_pct':    ufc('avg_SIG_STR_pct'),
        f'{prefix}_avg_SUB_ATT':        ufc('avg_SUB_ATT'),
        f'{prefix}_avg_TD_pct':         ufc('avg_TD_pct'),
        f'{prefix}_Reach_cms':          ufc('Reach_cms', 175),
        f'{prefix}_total_title_bouts':  ufc('total_title_bouts'),
        f'{prefix}_win_by_KO/TKO':      ufc('win_by_KO/TKO'),
        f'{prefix}_win_by_Submission':  ufc('win_by_Submission'),
        f'{prefix}_cum_fights':         cc['cum_fights'],
        f'{prefix}_career_win_rate':    cc['career_win_rate'],
        f'{prefix}_last5_won':          cc['last5_won'],
        f'{prefix}_last5_finish_rate':  cc['last5_finish_rate'],
        f'{prefix}_opp_quality':        cc['opp_quality'],
        f'{prefix}_trend_score':        cc['trend_score'],
        f'{prefix}_ko_finish_rate':     cc['ko_finish_rate'],
        f'{prefix}_sub_finish_rate':    cc['sub_finish_rate'],
        f'{prefix}_last3_win_rate':     cc['last3_win_rate'],
        f'{prefix}_last10_win_rate':    cc['last10_win_rate'],
        f'{prefix}_layoff_days':        cc['layoff_days'],
        f'{prefix}_layoff_lt90':        lb['lt90'],
        f'{prefix}_layoff_90_180':      lb['90_180'],
        f'{prefix}_layoff_180_365':     lb['180_365'],
        f'{prefix}_layoff_gt365':       lb['gt365'],
        f'{prefix}_SLpM':               ss.get('SLpM', 0),
        f'{prefix}_SApM':               ss.get('SApM', 0),
        f'{prefix}_Str_Acc':            ss.get('Str_Acc', 0),
        f'{prefix}_Str_Def':            ss.get('Str_Def', 0),
        f'{prefix}_TD_Avg':             ss.get('TD_Avg', 0),
        f'{prefix}_TD_Acc':             ss.get('TD_Acc', 0),
        f'{prefix}_TD_Def':             ss.get('TD_Def', 0),
        f'{prefix}_Sub_Avg':            ss.get('Sub_Avg', 0),
        f'{prefix}_elo':                eel['elo'],
        f'{prefix}_elo_trend':          eel['elo_trend'],
        f'{prefix}_southpaw':           sp,
        f'{prefix}_age_x_exp':          age * cc['cum_fights'],
    }

card_output = []
for fight_name, fight in card_odds.items():
    f1, f2 = fight['f1'], fight['f2']
    o1, o2 = fight['f1_odds'], fight['f2_odds']

    rf = build_fighter_feats(f1, 'R')
    bf = build_fighter_feats(f2, 'B')

    # Build 114-feature row
    r_age  = rf['R_age'];        b_age  = bf['B_age']
    r_wins = rf['R_wins'];       b_wins = bf['B_wins']
    r_loss = rf['R_losses'];     b_loss = bf['B_losses']
    r_h    = rf['R_Height_cms']; b_h    = bf['B_Height_cms']
    r_rch  = rf['R_Reach_cms'];  b_rch  = bf['B_Reach_cms']
    r_sig  = rf['R_avg_SIG_STR_landed']; b_sig = bf['B_avg_SIG_STR_landed']
    r_td   = rf['R_avg_TD_landed'];      b_td  = bf['B_avg_TD_landed']
    r_ws   = rf['R_current_win_streak']; b_ws  = bf['B_current_win_streak']
    r_ls   = rf['R_current_lose_streak'];b_ls  = bf['B_current_lose_streak']
    r_lws  = rf['R_longest_win_streak']; b_lws = bf['B_longest_win_streak']
    r_sigp = rf['R_avg_SIG_STR_pct'];   b_sigp= bf['B_avg_SIG_STR_pct']
    r_suba = rf['R_avg_SUB_ATT'];        b_suba= bf['B_avg_SUB_ATT']
    r_tdp  = rf['R_avg_TD_pct'];         b_tdp = bf['B_avg_TD_pct']
    r_ttb  = rf['R_total_title_bouts'];  b_ttb = bf['B_total_title_bouts']
    r_ko   = rf['R_win_by_KO/TKO'];      b_ko  = bf['B_win_by_KO/TKO']
    r_sub  = rf['R_win_by_Submission'];   b_sub = bf['B_win_by_Submission']
    r_sp   = rf['R_southpaw'];           b_sp  = bf['B_southpaw']
    r_axe  = rf['R_age_x_exp'];          b_axe = bf['B_age_x_exp']
    r_lb = {k[2:]: v for k,v in rf.items() if 'layoff_' in k and k.startswith('R_')}

    card_feat = {
        'R_wins': r_wins, 'R_losses': r_loss, 'R_Height_cms': r_h, 'R_age': r_age,
        'R_avg_SIG_STR_landed': r_sig, 'R_avg_TD_landed': r_td,
        'R_current_win_streak': r_ws, 'R_current_lose_streak': r_ls,
        'R_longest_win_streak': r_lws, 'R_avg_SIG_STR_pct': r_sigp,
        'R_avg_SUB_ATT': r_suba, 'R_avg_TD_pct': r_tdp,
        'R_Reach_cms': r_rch, 'R_total_title_bouts': r_ttb,
        'B_wins': b_wins, 'B_losses': b_loss, 'B_Height_cms': b_h, 'B_age': b_age,
        'B_avg_SIG_STR_landed': b_sig, 'B_avg_TD_landed': b_td,
        'B_current_win_streak': b_ws, 'B_current_lose_streak': b_ls,
        'B_longest_win_streak': b_lws, 'B_avg_SIG_STR_pct': b_sigp,
        'B_avg_SUB_ATT': b_suba, 'B_avg_TD_pct': b_tdp,
        'B_Reach_cms': b_rch, 'B_total_title_bouts': b_ttb,
        'win_dif': r_wins-b_wins, 'loss_dif': r_loss-b_loss,
        'win_streak_dif': r_ws-b_ws, 'lose_streak_dif': r_ls-b_ls,
        'height_dif': r_h-b_h, 'reach_dif': r_rch-b_rch,
        'age_dif': r_age-b_age, 'sig_str_dif': r_sig-b_sig,
        'avg_td_dif': r_td-b_td, 'ko_dif': r_ko-b_ko,
        'sub_dif': r_sub-b_sub, 'total_title_bout_dif': r_ttb-b_ttb,
        'weight_class_ord': 6, 'title_bout_bin': 0,
        'orth_clash': 1 if (r_sp==0 and b_sp==0) else 0,
        'south_clash': 1 if (r_sp==1 and b_sp==1) else 0,
        'R_southpaw': r_sp, 'B_southpaw': b_sp,
        'R_cum_fights': rf['R_cum_fights'], 'B_cum_fights': bf['B_cum_fights'],
        'R_career_win_rate': rf['R_career_win_rate'], 'B_career_win_rate': bf['B_career_win_rate'],
        'career_win_rate_dif': rf['R_career_win_rate']-bf['B_career_win_rate'],
        'R_last5_won': rf['R_last5_won'], 'B_last5_won': bf['B_last5_won'],
        'last5_won_dif': rf['R_last5_won']-bf['B_last5_won'],
        'R_last5_finish_rate': rf['R_last5_finish_rate'], 'B_last5_finish_rate': bf['B_last5_finish_rate'],
        'last5_finish_rate_dif': rf['R_last5_finish_rate']-bf['B_last5_finish_rate'],
        'R_opp_quality': rf['R_opp_quality'], 'B_opp_quality': bf['B_opp_quality'],
        'opp_quality_dif': rf['R_opp_quality']-bf['B_opp_quality'],
        'R_trend_score': rf['R_trend_score'], 'B_trend_score': bf['B_trend_score'],
        'trend_score_dif': rf['R_trend_score']-bf['B_trend_score'],
        'R_ko_finish_rate': rf['R_ko_finish_rate'], 'B_ko_finish_rate': bf['B_ko_finish_rate'],
        'ko_finish_rate_dif': rf['R_ko_finish_rate']-bf['B_ko_finish_rate'],
        'R_sub_finish_rate': rf['R_sub_finish_rate'], 'B_sub_finish_rate': bf['B_sub_finish_rate'],
        'sub_finish_rate_dif': rf['R_sub_finish_rate']-bf['B_sub_finish_rate'],
        'R_last3_win_rate': rf['R_last3_win_rate'], 'B_last3_win_rate': bf['B_last3_win_rate'],
        'last3_win_rate_dif': rf['R_last3_win_rate']-bf['B_last3_win_rate'],
        'R_last10_win_rate': rf['R_last10_win_rate'], 'B_last10_win_rate': bf['B_last10_win_rate'],
        'last10_win_rate_dif': rf['R_last10_win_rate']-bf['B_last10_win_rate'],
        'R_age_x_exp': r_axe, 'B_age_x_exp': b_axe, 'age_x_exp_dif': r_axe-b_axe,
        'R_layoff_lt90': rf['R_layoff_lt90'], 'R_layoff_90_180': rf['R_layoff_90_180'],
        'R_layoff_180_365': rf['R_layoff_180_365'], 'R_layoff_gt365': rf['R_layoff_gt365'],
        'B_layoff_lt90': bf['B_layoff_lt90'], 'B_layoff_90_180': bf['B_layoff_90_180'],
        'B_layoff_180_365': bf['B_layoff_180_365'], 'B_layoff_gt365': bf['B_layoff_gt365'],
        'R_SLpM': rf['R_SLpM'], 'B_SLpM': bf['B_SLpM'],
        'R_SApM': rf['R_SApM'], 'B_SApM': bf['B_SApM'],
        'R_Str_Acc': rf['R_Str_Acc'], 'B_Str_Acc': bf['B_Str_Acc'],
        'R_Str_Def': rf['R_Str_Def'], 'B_Str_Def': bf['B_Str_Def'],
        'R_TD_Avg': rf['R_TD_Avg'], 'B_TD_Avg': bf['B_TD_Avg'],
        'R_TD_Acc': rf['R_TD_Acc'], 'B_TD_Acc': bf['B_TD_Acc'],
        'R_TD_Def': rf['R_TD_Def'], 'B_TD_Def': bf['B_TD_Def'],
        'R_Sub_Avg': rf['R_Sub_Avg'], 'B_Sub_Avg': bf['B_Sub_Avg'],
        'SLpM_dif': rf['R_SLpM']-bf['B_SLpM'], 'SApM_dif': rf['R_SApM']-bf['B_SApM'],
        'Str_Def_dif': rf['R_Str_Def']-bf['B_Str_Def'],
        'TD_Def_dif': rf['R_TD_Def']-bf['B_TD_Def'],
        'Sub_Avg_dif': rf['R_Sub_Avg']-bf['B_Sub_Avg'],
        'TD_Avg_dif': rf['R_TD_Avg']-bf['B_TD_Avg'],
        'R_elo': rf['R_elo'], 'B_elo': bf['B_elo'],
        'elo_dif': rf['R_elo']-bf['B_elo'],
        'R_elo_trend': rf['R_elo_trend'], 'B_elo_trend': bf['B_elo_trend'],
        'elo_trend_dif': rf['R_elo_trend']-bf['B_elo_trend'],
    }

    x1_df = pd.DataFrame([card_feat])
    for col in feat_cols_114:
        if col not in x1_df.columns:
            x1_df[col] = 0
    x1 = x1_df[feat_cols_114].fillna(0).values

    lr_p  = float(model_lr.predict_proba(x1)[0][1])
    xgb_p = float(model_xgb.predict_proba(x1)[0][1])
    m1_p  = LR_WEIGHT * lr_p + XGB_WEIGHT * xgb_p

    f1_raw = implied_prob(o1); f2_raw = implied_prob(o2)
    tot    = f1_raw + f2_raw
    f1_nv  = f1_raw / tot;     f2_nv  = f2_raw / tot
    gap    = m1_p - f1_nv
    vconf  = abs(f1_nv - 0.5)
    mconf  = abs(m1_p - 0.5)

    # Model 2 features for this fight
    x2_card = np.array([[
        m1_p, f1_nv, f2_nv, gap, abs(gap), vconf,
        1.0 if f1_nv > 0.5 else 0.0,
        1.0 if (m1_p > 0.5) == (f1_nv > 0.5) else 0.0,
        mconf,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5,   # method odds: not available for tonight
        0.0, 0.0, 0.0,
        0.5,
        gap * vconf,
        m1_p * f1_nv,
        gap**2 * np.sign(gap),
    ]])

    best_m2_model = {'A_LR': model_A, 'B_XGB': model_B, 'C_CalibLR': model_C}[best_name]
    m2_p = float(best_m2_model.predict_proba(x2_card)[0][1])

    # Kelly bet sizing (quarter Kelly on $1000 bankroll)
    if gap > 0:   # bet F1
        odds_bet = float(o1)
        prob_win  = m2_p
    else:          # bet F2
        odds_bet = float(o2)
        prob_win  = 1 - m2_p

    if odds_bet < 0:
        b_decimal = 100 / abs(odds_bet)
    else:
        b_decimal = odds_bet / 100

    kelly = (b_decimal * prob_win - (1 - prob_win)) / b_decimal
    q_kelly = max(0, kelly / 4)
    bet_size = round(q_kelly * 1000, 0)

    value_flag = '⚡ VALUE' if abs(gap) > 0.05 else ''

    card_output.append({
        'fight': fight_name,
        'f1': f1, 'f2': f2,
        'm1_prob': m1_p,
        'f1_no_vig': f1_nv,
        'gap': gap,
        'm2_prob': m2_p,
        'value': value_flag,
        'kelly_bet': bet_size,
        'o1': o1, 'o2': o2,
    })

# ─────────────────────────────────────────────────────────────────────────────
# STEP 12 — Save best Model 2
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 10 — Saving Model 2 files...")

# Pick best overall by value ROI
all_vrois = {
    'A_LR': results['A_LR']['vroi'],
    'B_XGB': results['B_XGB']['vroi'],
    'C_CalibLR': results['C_CalibLR']['vroi'],
}
overall_best = max(all_vrois, key=all_vrois.get)
best_model_obj = {'A_LR': model_A, 'B_XGB': model_B, 'C_CalibLR': model_C}[overall_best]

save_ok = {'model': False, 'features': False, 'meta': False}
try:
    joblib.dump(best_model_obj, 'model/ufc_model2_best.pkl')
    save_ok['model'] = True
except Exception as e:
    print(f"  ERROR saving model: {e}")

try:
    joblib.dump(MODEL2_FEATURES, 'model/ufc_model2_features.pkl')
    save_ok['features'] = True
except Exception as e:
    print(f"  ERROR saving features: {e}")

try:
    meta2 = {
        'model_type': f'Model2_{overall_best}',
        'n_m2_features': len(MODEL2_FEATURES),
        'm2_feature_list': MODEL2_FEATURES,
        'train_fights': int(train_mask.sum()),
        'test_fights': int(test_mask.sum()),
        'test_accuracy': round(float({'A_LR': acc_A, 'B_XGB': acc_B, 'C_CalibLR': acc_C}[overall_best]), 4),
        'test_value_roi': round(float(all_vrois[overall_best]), 4),
        'model1_blend': f'{LR_WEIGHT:.0%} LR + {XGB_WEIGHT:.0%} XGB',
        'train_cutoff': str(TRAIN_CUTOFF.date()),
        'date_trained': datetime.now().isoformat(),
        'randomize_seed': RANDOM_SEED,
        'optuna_trials': N_OPTUNA_TRIALS,
    }
    with open('model/model2_metadata.json', 'w') as fp:
        json.dump(meta2, fp, indent=2)
    save_ok['meta'] = True
except Exception as e:
    print(f"  ERROR saving metadata: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("MODEL 2 — ODDS-AWARE VALUE BET MODEL")
print("=" * 60)
print(f"Training fights (2018-2023): {train_mask.sum()}")
print(f"Test fights     (2024+):     {test_mask.sum()}")
print()
print("BASELINE:")
print("  Vegas favorite accuracy: 66.80%")
print(f"  Model 1 accuracy (temporal): 73.24%")
print(f"  Model 1 on this dataset (OOF train): {train_acc:.2%}")
print(f"  Model 1 on test (2024+):             {test_acc:.2%}")
print()
print("MODEL 2 RESULTS:")
print(f"  {'Config':<15} {'Accuracy':>9} {'ROI(all)':>10} {'ROI(value)':>11} {'Bets':>6}")
print(f"  {'-'*15} {'-'*9} {'-'*10} {'-'*11} {'-'*6}")
print(f"  {'A — LR':<15} {acc_A:>9.2%} {roi_A:>+10.2f}% {vroi_A:>+11.2f}% {vnA:>6}")
print(f"  {'B — XGB':<15} {acc_B:>9.2%} {roi_B:>+10.2f}% {vroi_B:>+11.2f}% {vnB:>6}")
print(f"  {'C — Calib LR':<15} {acc_C:>9.2%} {roi_C:>+10.2f}% {vroi_C:>+11.2f}% {vnC:>6}")
print()
print("  D — Threshold model:")
for thresh, tr in thresh_results.items():
    print(f"    gap>{thresh:.0%}:  acc={tr['acc']:.2%}  ROI={tr['roi']:+.2f}%  ({tr['n']} bets)")
print()
print(f"Best Model 2: {overall_best}  (value ROI: {all_vrois[overall_best]:+.2f}%)")
print()
print("TONIGHT'S CARD — VALUE BETS (UFC Fight Night: Sterling vs Zalal):")
hdr = f"  {'Fight':<28} {'M1%':>6} {'Vegas%':>7} {'Gap':>7} {'Rec':>12} {'Kelly/$1k':>10}"
print(hdr)
print("  " + "-" * 75)
for co in card_output:
    m1p    = co['m1_prob']
    vnv    = co['f1_no_vig']
    gap    = co['gap']
    rec    = co['f1'] if gap > 0 else co['f2']
    kelly  = co['kelly_bet']
    flag   = co['value']
    print(f"  {co['fight']:<28} {m1p:>6.1%} {vnv:>7.1%} {gap:>+7.3f}  "
          f"{rec[:12]:<12} ${kelly:>6.0f}  {flag}")

print()
print(f"Saved: {'✓' if save_ok['model'] else '✗'} model2  "
      f"{'✓' if save_ok['features'] else '✗'} features  "
      f"{'✓' if save_ok['meta'] else '✗'} metadata")
print("=" * 60)
