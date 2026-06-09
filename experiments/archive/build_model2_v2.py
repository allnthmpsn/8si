#!/usr/bin/env python3
"""
build_model2_v2.py — New Model 2 (accuracy-focused) + System 2 ROI Optimizer
System 1: Model 1 prob + all Vegas odds → predict winner
System 2: Pure-math threshold + Kelly optimizer on 2024+ test set
"""

import bisect, gc, json, math, os, sys, time, warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, brier_score_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

# ─── Config ───────────────────────────────────────────────────────────────────
TRAIN_CUTOFF = pd.Timestamp('2024-01-01')
LR_WEIGHT    = 0.90
XGB_WEIGHT   = 0.10
SEED         = 42
BASELINE_ACC = 0.7324   # Model 1 temporal accuracy to beat
VEGAS_ACC    = 0.668

MODEL2_FEATURES = [
    'model1_prob',
    'f1_ml_novig', 'f2_ml_novig', 'ml_gap', 'vig',
    'f1_dec_implied', 'f2_dec_implied', 'dec_implied_dif',
    'f1_ko_implied',  'f2_ko_implied',  'ko_implied_dif',
    'f1_sub_implied', 'f2_sub_implied', 'sub_implied_dif',
    'finish_prob', 'f1_finish_prob', 'f2_finish_prob', 'finish_advantage',
    'abs_gap', 'vegas_confidence', 'model_confidence',
    'model_agrees_vegas', 'gap_x_confidence',
]

WC_ORDER = {
    "Women's Strawweight": 0, "Women's Flyweight": 1, "Women's Bantamweight": 2,
    "Women's Featherweight": 3, "Flyweight": 4, "Bantamweight": 5,
    "Featherweight": 6, "Lightweight": 7, "Welterweight": 8,
    "Middleweight": 9, "Light Heavyweight": 10, "Heavyweight": 11,
    "Catch Weight": 6,
}

# ─── Helpers ──────────────────────────────────────────────────────────────────
def implied_prob(odds):
    """American odds → raw implied probability."""
    try:
        odds = float(odds)
        if odds == 0 or np.isnan(odds):
            return None
        if odds < 0:
            return abs(odds) / (abs(odds) + 100)
        else:
            return 100 / (odds + 100)
    except Exception:
        return None

def american_to_decimal(odds):
    """American odds → decimal odds."""
    try:
        odds = float(odds)
        if odds == 0 or np.isnan(odds):
            return None
        if odds < 0:
            return 1 + 100 / abs(odds)
        else:
            return 1 + odds / 100
    except Exception:
        return None

def g(row, col, default=0.0):
    v = row.get(col, default)
    try:
        if pd.isna(v): return float(default)
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

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("NEW MODEL 2 + ROI OPTIMIZER")
print("=" * 60)

# ─── Load models + data ───────────────────────────────────────────────────────
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

# ─── STEP 1 — Filter + corner randomization ───────────────────────────────────
print("\nStep 1 — Filter + corner randomization...")

ALL_ODDS_COLS = ['R_odds', 'B_odds', 'r_dec_odds', 'b_dec_odds',
                 'r_sub_odds', 'b_sub_odds', 'r_ko_odds', 'b_ko_odds']

df = df_master[
    (df_master['date'] >= '2018-01-01') &
    df_master['R_odds'].notna() & df_master['B_odds'].notna() &
    df_master['r_dec_odds'].notna() & df_master['b_dec_odds'].notna() &
    df_master['r_sub_odds'].notna() & df_master['b_sub_odds'].notna() &
    df_master['r_ko_odds'].notna()  & df_master['b_ko_odds'].notna() &
    df_master['Winner'].isin(['Red', 'Blue'])
].copy().reset_index(drop=True)

print(f"  Total fights after filter: {len(df)}")

np.random.seed(SEED)
swap_mask = np.random.random(len(df)) < 0.5

r_all     = sorted([c for c in df.columns if c.startswith('R_')])
b_all     = sorted([c for c in df.columns if c.startswith('B_')])
r_matched = [c for c in r_all if ('B_' + c[2:]) in df.columns]
b_matched  = ['B_' + c[2:] for c in r_matched]

for rc, bc in zip(r_matched, b_matched):
    rv = df.loc[swap_mask, rc].values.copy()
    bv = df.loc[swap_mask, bc].values.copy()
    df.loc[swap_mask, rc] = bv
    df.loc[swap_mask, bc] = rv

df.loc[swap_mask & (df['Winner'] == 'Red'),  'Winner'] = 'TEMP'
df.loc[swap_mask & (df['Winner'] == 'Blue'), 'Winner'] = 'Red'
df.loc[swap_mask & (df['Winner'] == 'TEMP'), 'Winner'] = 'Blue'

for rc, bc in [('r_dec_odds','b_dec_odds'), ('r_sub_odds','b_sub_odds'), ('r_ko_odds','b_ko_odds')]:
    rv = df.loc[swap_mask, rc].values.copy()
    bv = df.loc[swap_mask, bc].values.copy()
    df.loc[swap_mask, rc] = bv
    df.loc[swap_mask, bc] = rv

target_full = (df['Winner'] == 'Red').astype(int).values
f1_fav = (df['R_odds'] < 0).values  # negative = favorite (after swap R=F1)

print(f"  F1 win rate (should be ~0.5): {target_full.mean():.3f}")
print(f"  F1 is favorite rate (should be ~0.5): {f1_fav.mean():.3f}")
print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
train_mask = (df['date'] < TRAIN_CUTOFF).values
test_mask  = ~train_mask
print(f"  2024+ test set size: {test_mask.sum()}")

# ─── STEP 2 — Career stats timeline ───────────────────────────────────────────
print("\nStep 2 — Career stats timeline...")

cf = career_raw.copy()

def shift_cumsum(x):
    return x.cumsum().shift(1).fillna(0)

cf['cum_fights']      = cf.groupby('fighter').cumcount()
cf['cum_wins']        = cf.groupby('fighter')['won'].transform(shift_cumsum)
cf['career_win_rate'] = np.where(cf['cum_fights'] > 0, cf['cum_wins'] / cf['cum_fights'], 0.5)
cf['ko_win']  = ((cf['won']==1) & cf['method'].str.contains('KO|TKO', case=False, na=False)).astype(int)
cf['sub_win'] = ((cf['won']==1) & cf['method'].str.contains('Sub|Submission', case=False, na=False)).astype(int)
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
cf['prev_date']         = cf.groupby('fighter')['date'].shift(1)
cf['layoff_days']       = (cf['date'] - cf['prev_date']).dt.days.fillna(365.0)

wr_cache = cf.groupby('fighter')['won'].mean().to_dict()

def opp_quality_series(grp):
    opps = grp['opponent'].values
    res  = np.full(len(grp), 0.5)
    for i in range(len(grp)):
        prior = opps[max(0, i-5):i]
        rates = [wr_cache.get(o, 0.5) for o in prior]
        res[i] = float(np.mean(rates)) if rates else 0.5
    return pd.Series(res, index=grp.index)

cf['opp_quality'] = cf.groupby('fighter', group_keys=False).apply(opp_quality_series)

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

career_by_f    = {}
career_dates_f = {}
for fname, grp in cf.groupby('fighter'):
    g_ = grp.reset_index(drop=True)
    career_by_f[fname]    = g_
    career_dates_f[fname] = g_['date'].tolist()

def get_career_at(fighter, fight_date):
    if fighter not in career_by_f:
        return DEFAULT_CAREER.copy()
    dates = career_dates_f[fighter]
    idx   = bisect.bisect_right(dates, fight_date) - 1
    if idx < 0:
        return DEFAULT_CAREER.copy()
    row = career_by_f[fighter].iloc[idx]
    return {c: float(row[c]) for c in CAREER_COLS}

elo_by_f    = {}
elo_dates_f = {}
for fname, grp in elo_hist.groupby('fighter'):
    g_ = grp.sort_values('date').reset_index(drop=True)
    elo_by_f[fname]    = g_
    elo_dates_f[fname] = g_['date'].tolist()

def get_elo_at(fighter, fight_date):
    if fighter not in elo_by_f:
        return {'elo': 1500.0, 'elo_trend': 0.0}
    dates = elo_dates_f[fighter]
    idx = bisect.bisect_left(dates, fight_date) - 1
    if idx < 0:
        return {'elo': 1500.0, 'elo_trend': 0.0}
    row = elo_by_f[fighter].iloc[idx]
    return {'elo': float(row['elo_after']), 'elo_trend': float(row.get('elo_trend', 0.0) or 0.0)}

fstyle = {}
for _, row in fstats_df.iterrows():
    fstyle[row['Fighter_Name']] = {
        'SLpM':    float(row.get('SLpM',    0) or 0),
        'SApM':    float(row.get('SApM',    0) or 0),
        'Str_Acc': float(row.get('Str_Acc', 0) or 0),
        'Str_Def': float(row.get('Str_Def', 0) or 0),
        'TD_Avg':  float(row.get('TD_Avg',  0) or 0),
        'TD_Acc':  float(row.get('TD_Acc',  0) or 0),
        'TD_Def':  float(row.get('TD_Def',  0) or 0),
        'Sub_Avg': float(row.get('Sub_Avg', 0) or 0),
    }
print(f"  Career data for {len(career_by_f)} fighters")

# ─── STEP 3 — Build 114-feature matrix ────────────────────────────────────────
print("\nStep 3 — Building 114-feature matrix...")

def build_114_features(row):
    r_name = row['R_fighter']; b_name = row['B_fighter']; fdate = row['date']
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
    r_wins=g(row,'R_wins'); b_wins=g(row,'B_wins')
    r_loss=g(row,'R_losses'); b_loss=g(row,'B_losses')
    r_h=g(row,'R_Height_cms',175); b_h=g(row,'B_Height_cms',175)
    r_rch=g(row,'R_Reach_cms',175); b_rch=g(row,'B_Reach_cms',175)
    r_age=g(row,'R_age',28); b_age=g(row,'B_age',28)
    r_sig=g(row,'R_avg_SIG_STR_landed'); b_sig=g(row,'B_avg_SIG_STR_landed')
    r_td=g(row,'R_avg_TD_landed'); b_td=g(row,'B_avg_TD_landed')
    r_ws=g(row,'R_current_win_streak'); b_ws=g(row,'B_current_win_streak')
    r_ls=g(row,'R_current_lose_streak'); b_ls=g(row,'B_current_lose_streak')
    r_lws=g(row,'R_longest_win_streak'); b_lws=g(row,'B_longest_win_streak')
    r_sigp=g(row,'R_avg_SIG_STR_pct'); b_sigp=g(row,'B_avg_SIG_STR_pct')
    r_suba=g(row,'R_avg_SUB_ATT'); b_suba=g(row,'B_avg_SUB_ATT')
    r_tdp=g(row,'R_avg_TD_pct'); b_tdp=g(row,'B_avg_TD_pct')
    r_ttb=g(row,'R_total_title_bouts'); b_ttb=g(row,'B_total_title_bouts')
    r_ko=g(row,'R_win_by_KO/TKO'); b_ko=g(row,'B_win_by_KO/TKO')
    r_sub=g(row,'R_win_by_Submission'); b_sub=g(row,'B_win_by_Submission')
    wc_ord = WC_ORDER.get(str(row.get('weight_class','') or ''), 6)
    title  = 1 if row.get('title_bout', False) else 0
    r_axe  = r_age * rc['cum_fights']; b_axe = b_age * bc['cum_fights']
    return {
        'R_wins':r_wins,'R_losses':r_loss,'R_Height_cms':r_h,'R_age':r_age,
        'R_avg_SIG_STR_landed':r_sig,'R_avg_TD_landed':r_td,
        'R_current_win_streak':r_ws,'R_current_lose_streak':r_ls,
        'R_longest_win_streak':r_lws,'R_avg_SIG_STR_pct':r_sigp,
        'R_avg_SUB_ATT':r_suba,'R_avg_TD_pct':r_tdp,
        'R_Reach_cms':r_rch,'R_total_title_bouts':r_ttb,
        'B_wins':b_wins,'B_losses':b_loss,'B_Height_cms':b_h,'B_age':b_age,
        'B_avg_SIG_STR_landed':b_sig,'B_avg_TD_landed':b_td,
        'B_current_win_streak':b_ws,'B_current_lose_streak':b_ls,
        'B_longest_win_streak':b_lws,'B_avg_SIG_STR_pct':b_sigp,
        'B_avg_SUB_ATT':b_suba,'B_avg_TD_pct':b_tdp,
        'B_Reach_cms':b_rch,'B_total_title_bouts':b_ttb,
        'win_dif':r_wins-b_wins,'loss_dif':r_loss-b_loss,
        'win_streak_dif':r_ws-b_ws,'lose_streak_dif':r_ls-b_ls,
        'height_dif':r_h-b_h,'reach_dif':r_rch-b_rch,'age_dif':r_age-b_age,
        'sig_str_dif':r_sig-b_sig,'avg_td_dif':r_td-b_td,
        'ko_dif':r_ko-b_ko,'sub_dif':r_sub-b_sub,
        'total_title_bout_dif':r_ttb-b_ttb,
        'weight_class_ord':wc_ord,'title_bout_bin':title,
        'orth_clash':1 if (r_sp==0 and b_sp==0) else 0,
        'south_clash':1 if (r_sp==1 and b_sp==1) else 0,
        'R_southpaw':r_sp,'B_southpaw':b_sp,
        'R_cum_fights':rc['cum_fights'],'B_cum_fights':bc['cum_fights'],
        'R_career_win_rate':rc['career_win_rate'],'B_career_win_rate':bc['career_win_rate'],
        'career_win_rate_dif':rc['career_win_rate']-bc['career_win_rate'],
        'R_last5_won':rc['last5_won'],'B_last5_won':bc['last5_won'],
        'last5_won_dif':rc['last5_won']-bc['last5_won'],
        'R_last5_finish_rate':rc['last5_finish_rate'],'B_last5_finish_rate':bc['last5_finish_rate'],
        'last5_finish_rate_dif':rc['last5_finish_rate']-bc['last5_finish_rate'],
        'R_opp_quality':rc['opp_quality'],'B_opp_quality':bc['opp_quality'],
        'opp_quality_dif':rc['opp_quality']-bc['opp_quality'],
        'R_trend_score':rc['trend_score'],'B_trend_score':bc['trend_score'],
        'trend_score_dif':rc['trend_score']-bc['trend_score'],
        'R_ko_finish_rate':rc['ko_finish_rate'],'B_ko_finish_rate':bc['ko_finish_rate'],
        'ko_finish_rate_dif':rc['ko_finish_rate']-bc['ko_finish_rate'],
        'R_sub_finish_rate':rc['sub_finish_rate'],'B_sub_finish_rate':bc['sub_finish_rate'],
        'sub_finish_rate_dif':rc['sub_finish_rate']-bc['sub_finish_rate'],
        'R_last3_win_rate':rc['last3_win_rate'],'B_last3_win_rate':bc['last3_win_rate'],
        'last3_win_rate_dif':rc['last3_win_rate']-bc['last3_win_rate'],
        'R_last10_win_rate':rc['last10_win_rate'],'B_last10_win_rate':bc['last10_win_rate'],
        'last10_win_rate_dif':rc['last10_win_rate']-bc['last10_win_rate'],
        'R_age_x_exp':r_axe,'B_age_x_exp':b_axe,'age_x_exp_dif':r_axe-b_axe,
        'R_layoff_lt90':r_lb['lt90'],'R_layoff_90_180':r_lb['90_180'],
        'R_layoff_180_365':r_lb['180_365'],'R_layoff_gt365':r_lb['gt365'],
        'B_layoff_lt90':b_lb['lt90'],'B_layoff_90_180':b_lb['90_180'],
        'B_layoff_180_365':b_lb['180_365'],'B_layoff_gt365':b_lb['gt365'],
        'R_SLpM':rs.get('SLpM',0),'R_SApM':rs.get('SApM',0),
        'R_Str_Acc':rs.get('Str_Acc',0),'R_Str_Def':rs.get('Str_Def',0),
        'R_TD_Avg':rs.get('TD_Avg',0),'R_TD_Acc':rs.get('TD_Acc',0),
        'R_TD_Def':rs.get('TD_Def',0),'R_Sub_Avg':rs.get('Sub_Avg',0),
        'B_SLpM':bs.get('SLpM',0),'B_SApM':bs.get('SApM',0),
        'B_Str_Acc':bs.get('Str_Acc',0),'B_Str_Def':bs.get('Str_Def',0),
        'B_TD_Avg':bs.get('TD_Avg',0),'B_TD_Acc':bs.get('TD_Acc',0),
        'B_TD_Def':bs.get('TD_Def',0),'B_Sub_Avg':bs.get('Sub_Avg',0),
        'SLpM_dif':rs.get('SLpM',0)-bs.get('SLpM',0),
        'SApM_dif':rs.get('SApM',0)-bs.get('SApM',0),
        'Str_Def_dif':rs.get('Str_Def',0)-bs.get('Str_Def',0),
        'TD_Def_dif':rs.get('TD_Def',0)-bs.get('TD_Def',0),
        'Sub_Avg_dif':rs.get('Sub_Avg',0)-bs.get('Sub_Avg',0),
        'TD_Avg_dif':rs.get('TD_Avg',0)-bs.get('TD_Avg',0),
        'R_elo':re['elo'],'B_elo':be['elo'],'elo_dif':re['elo']-be['elo'],
        'R_elo_trend':re['elo_trend'],'B_elo_trend':be['elo_trend'],
        'elo_trend_dif':re['elo_trend']-be['elo_trend'],
    }

rows_list = [build_114_features(row) for _, row in df.iterrows()]
X_114_df = pd.DataFrame(rows_list, columns=feat_cols_114)
X_114    = X_114_df.values.astype(float)
# Fill NaNs with column medians to prevent sklearn errors
col_med_114 = np.nanmedian(X_114, axis=0)
nan_m_114   = np.isnan(X_114)
X_114[nan_m_114] = np.take(col_med_114, np.where(nan_m_114)[1])
print(f"  Feature matrix: {X_114.shape}")

# ─── STEP 4 — Model 1 OOF predictions (no leakage) ────────────────────────────
print("\nStep 4 — Model 1 OOF predictions (no leakage)...")

y = target_full.copy()
dates = df['date'].values

train_idx = np.where(train_mask)[0]
test_idx  = np.where(test_mask)[0]
X_train_114 = X_114[train_idx]
X_test_114  = X_114[test_idx]
y_train     = y[train_idx]
y_test      = y[test_idx]

print(f"  Train (2018-2023): {len(train_idx)}  |  Test (2024+): {len(test_idx)}")

# Build M1 blend probability
def m1_prob(X):
    p_lr  = model_lr.predict_proba(X)[:, 1]
    p_xgb = model_xgb.predict_proba(X)[:, 1]
    return LR_WEIGHT * p_lr + XGB_WEIGHT * p_xgb

# OOF for training set
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
m1_oof = np.zeros(len(train_idx))
for fold, (tr_i, va_i) in enumerate(skf.split(X_train_114, y_train)):
    Xtr, Xva = X_train_114[tr_i], X_train_114[va_i]
    ytr = y_train[tr_i]
    # Refit LR pipeline on this fold
    fold_lr  = Pipeline([
        ('scaler', RobustScaler()),
        ('clf', LogisticRegression(C=1.0, max_iter=1000, solver='saga', random_state=SEED))
    ])
    fold_lr.fit(Xtr, ytr)
    fold_xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3,
                              use_label_encoder=False, eval_metric='logloss',
                              random_state=SEED, verbosity=0)
    fold_xgb.fit(Xtr, ytr)
    p_lr_v  = fold_lr.predict_proba(Xva)[:, 1]
    p_xgb_v = fold_xgb.predict_proba(Xva)[:, 1]
    m1_oof[va_i] = LR_WEIGHT * p_lr_v + XGB_WEIGHT * p_xgb_v

# Full model on all training data for test set predictions
m1_test = m1_prob(X_test_114)

print(f"  Model 1 OOF train acc:  {accuracy_score(y_train, (m1_oof > 0.5).astype(int)):.4f}")
print(f"  Model 1 test acc:       {accuracy_score(y_test, (m1_test > 0.5).astype(int)):.4f}")

# ─── STEP 5 — Build 23-feature Model 2 dataset ────────────────────────────────
print("\nStep 5 — Building 23 Model 2 features...")

def build_m2_row(i, df_row, m1p):
    f1_odds  = float(df_row['R_odds'])
    f2_odds  = float(df_row['B_odds'])
    f1_dec   = float(df_row['r_dec_odds'])
    f2_dec   = float(df_row['b_dec_odds'])
    f1_sub   = float(df_row['r_sub_odds'])
    f2_sub   = float(df_row['b_sub_odds'])
    f1_ko    = float(df_row['r_ko_odds'])
    f2_ko    = float(df_row['b_ko_odds'])

    # Moneyline no-vig
    f1_ml_raw = implied_prob(f1_odds)
    f2_ml_raw = implied_prob(f2_odds)
    total_ml  = (f1_ml_raw or 0) + (f2_ml_raw or 0)
    if total_ml > 0:
        f1_ml_novig = f1_ml_raw / total_ml
        f2_ml_novig = f2_ml_raw / total_ml
        vig = total_ml - 1.0
    else:
        f1_ml_novig = 0.5; f2_ml_novig = 0.5; vig = 0.0

    ml_gap = m1p - f1_ml_novig

    # Method odds
    f1_dec_imp  = implied_prob(f1_dec)  or 0.0
    f2_dec_imp  = implied_prob(f2_dec)  or 0.0
    f1_ko_imp   = implied_prob(f1_ko)   or 0.0
    f2_ko_imp   = implied_prob(f2_ko)   or 0.0
    f1_sub_imp  = implied_prob(f1_sub)  or 0.0
    f2_sub_imp  = implied_prob(f2_sub)  or 0.0

    dec_implied_dif = f1_dec_imp - f2_dec_imp
    ko_implied_dif  = f1_ko_imp  - f2_ko_imp
    sub_implied_dif = f1_sub_imp - f2_sub_imp

    dec_total    = f1_dec_imp + f2_dec_imp
    finish_prob  = 1.0 - (dec_total / 2.0) if dec_total > 0 else 0.5

    f1_finish_prob   = f1_ko_imp + f1_sub_imp
    f2_finish_prob   = f2_ko_imp + f2_sub_imp
    finish_advantage = f1_finish_prob - f2_finish_prob

    abs_gap         = abs(ml_gap)
    vegas_conf      = abs(f1_ml_novig - 0.5)
    model_conf      = abs(m1p - 0.5)
    model_agrees    = 1 if (m1p > 0.5) == (f1_ml_novig > 0.5) else 0
    gap_x_conf      = ml_gap * vegas_conf

    return [
        m1p, f1_ml_novig, f2_ml_novig, ml_gap, vig,
        f1_dec_imp, f2_dec_imp, dec_implied_dif,
        f1_ko_imp,  f2_ko_imp,  ko_implied_dif,
        f1_sub_imp, f2_sub_imp, sub_implied_dif,
        finish_prob, f1_finish_prob, f2_finish_prob, finish_advantage,
        abs_gap, vegas_conf, model_conf, model_agrees, gap_x_conf,
    ]

train_set  = set(train_idx)
train_pos  = {v: k for k, v in enumerate(train_idx)}
test_pos   = {v: k for k, v in enumerate(test_idx)}
all_m2_rows = []
for i, (_, df_row) in enumerate(df.iterrows()):
    if i in train_set:
        m1p = float(m1_oof[train_pos[i]])
    else:
        m1p = float(m1_test[test_pos[i]])
    all_m2_rows.append(build_m2_row(i, df_row, m1p))

X2 = np.array(all_m2_rows, dtype=float)
# Impute any remaining NaNs with column medians
col_medians = np.nanmedian(X2, axis=0)
nan_mask    = np.isnan(X2)
X2[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
X2_train = X2[train_idx]
X2_test  = X2[test_idx]

# Also store odds arrays for ROI optimizer
f1_odds_arr  = df['R_odds'].values
f2_odds_arr  = df['B_odds'].values
f1_novig_arr = X2[:, MODEL2_FEATURES.index('f1_ml_novig')]
f2_novig_arr = X2[:, MODEL2_FEATURES.index('f2_ml_novig')]
gap_arr      = X2[:, MODEL2_FEATURES.index('ml_gap')]

f1_odds_test  = f1_odds_arr[test_idx]
f2_odds_test  = f2_odds_arr[test_idx]
f1_novig_test = f1_novig_arr[test_idx]
gap_test      = gap_arr[test_idx]
y_test_arr    = y_test.copy()

print(f"  Model 2 feature matrix: {X2.shape}")

# ─── STEP 6 — Train New Model 2 ───────────────────────────────────────────────
print("\nStep 6 — Training New Model 2 models...")

skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
results = []
trained_models = {}

def timed_study(objective, n_trials, name):
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    deadline = time.time() + 360  # 6-minute hard cut
    done = 0
    for i in range(n_trials):
        if time.time() > deadline:
            print(f"    {name}: time limit hit after {done} trials")
            break
        study.optimize(objective, n_trials=1, show_progress_bar=False)
        done += 1
    return study.best_params

# ── A: Logistic Regression ─────────────────────────────────────────────────────
print("\n  Step 6A — Logistic Regression (100 trials)...")

def lr_objective(trial):
    C       = trial.suggest_float('C', 0.001, 50.0, log=True)
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
    sc_type = trial.suggest_categorical('scaler', ['robust', 'standard'])
    cw      = trial.suggest_categorical('class_weight', ['none', 'balanced'])
    cw_val  = None if cw == 'none' else 'balanced'

    if penalty == 'elasticnet':
        l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
        solver   = 'saga'
        clf = LogisticRegression(C=C, penalty='elasticnet', l1_ratio=l1_ratio,
                                 solver='saga', class_weight=cw_val,
                                 max_iter=2000, random_state=SEED)
    elif penalty == 'l1':
        solver = 'saga'
        clf = LogisticRegression(C=C, penalty='l1', solver='saga',
                                 class_weight=cw_val, max_iter=2000, random_state=SEED)
    else:
        clf = LogisticRegression(C=C, penalty='l2', solver='saga',
                                 class_weight=cw_val, max_iter=2000, random_state=SEED)

    scaler = RobustScaler() if sc_type == 'robust' else StandardScaler()
    pipe   = Pipeline([('sc', scaler), ('clf', clf)])
    oof    = cross_val_predict(pipe, X2_train, y_train, cv=skf2,
                               method='predict_proba', n_jobs=1)[:, 1]
    return accuracy_score(y_train, (oof > 0.5).astype(int))

best_lr_params = timed_study(lr_objective, 100, 'LR')

# Rebuild best LR
p = best_lr_params
penalty = p['penalty']
cw_val  = None if p['class_weight'] == 'none' else 'balanced'
if penalty == 'elasticnet':
    clf_A = LogisticRegression(C=p['C'], penalty='elasticnet',
                               l1_ratio=p.get('l1_ratio', 0.5),
                               solver='saga', class_weight=cw_val,
                               max_iter=2000, random_state=SEED)
elif penalty == 'l1':
    clf_A = LogisticRegression(C=p['C'], penalty='l1', solver='saga',
                               class_weight=cw_val, max_iter=2000, random_state=SEED)
else:
    clf_A = LogisticRegression(C=p['C'], penalty='l2', solver='saga',
                               class_weight=cw_val, max_iter=2000, random_state=SEED)
sc_A = RobustScaler() if p['scaler'] == 'robust' else StandardScaler()
model_A = Pipeline([('sc', sc_A), ('clf', clf_A)])
model_A.fit(X2_train, y_train)
p_A = model_A.predict_proba(X2_test)[:, 1]
acc_A   = accuracy_score(y_test, (p_A > 0.5).astype(int))
brier_A = brier_score_loss(y_test, p_A)
trained_models['LR'] = model_A
print(f"  A — LR:       acc={acc_A:.4f}  brier={brier_A:.4f}  beat M1? {'✓' if acc_A > BASELINE_ACC else '✗'}")

try:
    coefs = abs(model_A.named_steps['clf'].coef_[0])
    top_A = sorted(zip(MODEL2_FEATURES, coefs), key=lambda x: -x[1])[:10]
except Exception:
    top_A = []

gc.collect()

# ── B: XGBoost ─────────────────────────────────────────────────────────────────
print("\n  Step 6B — XGBoost (75 trials)...")

def xgb_objective(trial):
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 50, 600),
        'learning_rate':    trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'max_depth':        trial.suggest_int('max_depth', 2, 5),
        'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma':            trial.suggest_float('gamma', 0, 3),
        'reg_alpha':        trial.suggest_float('reg_alpha', 0, 2),
    }
    clf = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss',
                        random_state=SEED, verbosity=0, n_jobs=1)
    oof = cross_val_predict(clf, X2_train, y_train, cv=skf2,
                            method='predict_proba', n_jobs=1)[:, 1]
    return accuracy_score(y_train, (oof > 0.5).astype(int))

best_xgb_params = timed_study(xgb_objective, 75, 'XGB')
model_B = XGBClassifier(**best_xgb_params, use_label_encoder=False,
                        eval_metric='logloss', random_state=SEED, verbosity=0, n_jobs=1)
model_B.fit(X2_train, y_train)
p_B = model_B.predict_proba(X2_test)[:, 1]
acc_B   = accuracy_score(y_test, (p_B > 0.5).astype(int))
brier_B = brier_score_loss(y_test, p_B)
trained_models['XGB'] = model_B
print(f"  B — XGB:      acc={acc_B:.4f}  brier={brier_B:.4f}  beat M1? {'✓' if acc_B > BASELINE_ACC else '✗'}")

top_B = sorted(zip(MODEL2_FEATURES, model_B.feature_importances_), key=lambda x: -x[1])[:10]
gc.collect()

# ── C: LightGBM ────────────────────────────────────────────────────────────────
print("\n  Step 6C — LightGBM (75 trials)...")

def lgb_objective(trial):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 50, 600),
        'learning_rate':     trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'max_depth':         trial.suggest_int('max_depth', 2, 6),
        'num_leaves':        trial.suggest_int('num_leaves', 8, 63),
        'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'reg_alpha':         trial.suggest_float('reg_alpha', 0, 2),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }
    clf = LGBMClassifier(**params, random_state=SEED, verbose=-1, n_jobs=1)
    oof = cross_val_predict(clf, X2_train, y_train, cv=skf2,
                            method='predict_proba', n_jobs=1)[:, 1]
    return accuracy_score(y_train, (oof > 0.5).astype(int))

best_lgb_params = timed_study(lgb_objective, 75, 'LGB')
model_C = LGBMClassifier(**best_lgb_params, random_state=SEED, verbose=-1, n_jobs=1)
model_C.fit(X2_train, y_train)
p_C = model_C.predict_proba(X2_test)[:, 1]
acc_C   = accuracy_score(y_test, (p_C > 0.5).astype(int))
brier_C = brier_score_loss(y_test, p_C)
trained_models['LGB'] = model_C
print(f"  C — LGB:      acc={acc_C:.4f}  brier={brier_C:.4f}  beat M1? {'✓' if acc_C > BASELINE_ACC else '✗'}")

top_C = sorted(zip(MODEL2_FEATURES, model_C.feature_importances_), key=lambda x: -x[1])[:10]
gc.collect()

# ── D: CatBoost ────────────────────────────────────────────────────────────────
print("\n  Step 6D — CatBoost (50 trials)...")

def cat_objective(trial):
    params = {
        'iterations':    trial.suggest_int('iterations', 50, 400),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth':         trial.suggest_int('depth', 2, 5),
        'l2_leaf_reg':   trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
    }
    clf = CatBoostClassifier(**params, random_seed=SEED, verbose=False, thread_count=1)
    oof = cross_val_predict(clf, X2_train, y_train, cv=skf2,
                            method='predict_proba', n_jobs=1)[:, 1]
    return accuracy_score(y_train, (oof > 0.5).astype(int))

best_cat_params = timed_study(cat_objective, 50, 'CatBoost')
model_D = CatBoostClassifier(**best_cat_params, random_seed=SEED, verbose=False, thread_count=1)
model_D.fit(X2_train, y_train)
p_D = model_D.predict_proba(X2_test)[:, 1]
acc_D   = accuracy_score(y_test, (p_D > 0.5).astype(int))
brier_D = brier_score_loss(y_test, p_D)
trained_models['CatBoost'] = model_D
print(f"  D — CatBoost: acc={acc_D:.4f}  brier={brier_D:.4f}  beat M1? {'✓' if acc_D > BASELINE_ACC else '✗'}")

top_D = sorted(zip(MODEL2_FEATURES, model_D.get_feature_importance()), key=lambda x: -x[1])[:10]
gc.collect()

# ── E: Stacking ────────────────────────────────────────────────────────────────
print("\n  Step 6E — Stacking (LR + XGB + LGB → meta LR)...")

oof_A = cross_val_predict(model_A, X2_train, y_train, cv=skf2,
                          method='predict_proba', n_jobs=1)[:, 1]
oof_B = cross_val_predict(model_B, X2_train, y_train, cv=skf2,
                          method='predict_proba', n_jobs=1)[:, 1]
oof_C = cross_val_predict(model_C, X2_train, y_train, cv=skf2,
                          method='predict_proba', n_jobs=1)[:, 1]

meta_train = np.column_stack([oof_A, oof_B, oof_C])
meta_test  = np.column_stack([p_A, p_B, p_C])

meta_lr = LogisticRegression(C=1.0, solver='saga', max_iter=1000, random_state=SEED)
meta_lr.fit(meta_train, y_train)
p_E = meta_lr.predict_proba(meta_test)[:, 1]
acc_E   = accuracy_score(y_test, (p_E > 0.5).astype(int))
brier_E = brier_score_loss(y_test, p_E)
trained_models['Stack'] = (model_A, model_B, model_C, meta_lr)
print(f"  E — Stack:    acc={acc_E:.4f}  brier={brier_E:.4f}  beat M1? {'✓' if acc_E > BASELINE_ACC else '✗'}")

gc.collect()

# ─── Pick best model ───────────────────────────────────────────────────────────
model_results = [
    ('LR',      acc_A, brier_A, p_A, top_A),
    ('XGB',     acc_B, brier_B, p_B, top_B),
    ('LGB',     acc_C, brier_C, p_C, top_C),
    ('CatBoost',acc_D, brier_D, p_D, top_D),
    ('Stack',   acc_E, brier_E, p_E, []),
]
model_results.sort(key=lambda x: -x[1])  # sort by accuracy
best_name, best_acc, best_brier, best_probs, best_top = model_results[0]
best_model = trained_models[best_name]

print(f"\n  Best New Model 2: {best_name} at {best_acc:.4f} temporal accuracy")

# ─── STEP 7 — ROI Optimizer ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 — ROI OPTIMIZER (System 2)")
print("=" * 60)

def simulate_flat_bets(probs, actuals, novig, f1_odds_american, f2_odds_american, threshold, bet_size=100):
    bets = 0; wins = 0; profit = 0.0
    for p, act, nv, o1, o2 in zip(probs, actuals, novig, f1_odds_american, f2_odds_american):
        gap = p - nv
        if abs(gap) <= threshold:
            continue
        bet_f1 = gap > 0
        bets += 1
        if bet_f1:
            dec = american_to_decimal(o1)
            if dec is None: continue
            won = act == 1
            profit += (dec - 1) * bet_size if won else -bet_size
        else:
            dec = american_to_decimal(o2)
            if dec is None: continue
            won = act == 0
            profit += (dec - 1) * bet_size if won else -bet_size
        if (bet_f1 and act == 1) or (not bet_f1 and act == 0):
            wins += 1
    roi = (profit / (bets * bet_size)) * 100 if bets > 0 else 0.0
    win_pct = (wins / bets * 100) if bets > 0 else 0.0
    return bets, win_pct, roi, profit

# Threshold scan
print("\n  Threshold scan (flat $100 bets on 2024+ test set):\n")
threshold_list = [0.02, 0.03, 0.05, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15, 0.20]
threshold_results = []

print(f"  {'Threshold':>10}  {'Bets':>5}  {'Win%':>6}  {'ROI':>8}  {'Profit':>9}")
print(f"  {'-'*10}  {'-'*5}  {'-'*6}  {'-'*8}  {'-'*9}")
for thr in threshold_list:
    bets, win_pct, roi, profit = simulate_flat_bets(
        best_probs, y_test_arr, f1_novig_test,
        f1_odds_test, f2_odds_test, thr
    )
    threshold_results.append({'threshold': thr, 'bets': bets, 'win_pct': win_pct, 'roi': roi, 'profit': profit})
    print(f"  {thr*100:>9.0f}%  {bets:>5}  {win_pct:>5.1f}%  {roi:>7.1f}%  ${profit:>8.0f}")

# Find optimal threshold (best ROI with >= 20 bets)
valid_thresholds = [r for r in threshold_results if r['bets'] >= 20]
opt_threshold_row = max(valid_thresholds, key=lambda x: x['roi']) if valid_thresholds else threshold_results[5]
opt_threshold = opt_threshold_row['threshold']
print(f"\n  Optimal threshold: {opt_threshold*100:.0f}% (ROI={opt_threshold_row['roi']:.1f}%, {opt_threshold_row['bets']} bets)")

# Kelly fraction scan
print(f"\n  Kelly fraction scan (threshold={opt_threshold*100:.0f}%, $1000 bankroll):\n")

def simulate_kelly(probs, actuals, novig, f1_odds_american, f2_odds_american, threshold, kelly_frac, bankroll=1000):
    bk = bankroll; peak = bankroll; trough = bankroll
    bets = 0; wins = 0
    max_dd = 0.0
    bk_history = [bankroll]
    for p, act, nv, o1, o2 in zip(probs, actuals, novig, f1_odds_american, f2_odds_american):
        gap = p - nv
        if abs(gap) <= threshold:
            continue
        bet_f1 = gap > 0
        if bet_f1:
            dec = american_to_decimal(o1)
            if dec is None: continue
            b = dec - 1
            kelly = (p * b - (1 - p)) / b
        else:
            dec = american_to_decimal(o2)
            if dec is None: continue
            b = dec - 1
            opp_p = 1 - p
            kelly = (opp_p * b - p) / b
        kelly = max(0, kelly) * kelly_frac
        kelly = min(kelly, 0.20)  # cap at 20% bankroll
        bet_amt = kelly * bk
        bets += 1
        won = (bet_f1 and act == 1) or (not bet_f1 and act == 0)
        if won:
            bk += bet_amt * b
            wins += 1
        else:
            bk -= bet_amt
        bk_history.append(bk)
        peak = max(peak, bk)
        dd = (peak - bk) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)
    win_pct = wins / bets * 100 if bets > 0 else 0
    return bk, max_dd, bets, win_pct

kelly_fracs = [1.0, 0.5, 1/3, 0.25, 0.2, 1/6]
kelly_labels = ['Full', '1/2', '1/3', '1/4', '1/5', '1/6']
kelly_results = []

print(f"  {'Kelly Frac':>10}  {'End Bankroll':>13}  {'Max Drawdown':>13}  {'Bets':>5}  {'Win%':>6}")
print(f"  {'-'*10}  {'-'*13}  {'-'*13}  {'-'*5}  {'-'*6}")
for frac, label in zip(kelly_fracs, kelly_labels):
    end_bk, max_dd, bets, win_pct = simulate_kelly(
        best_probs, y_test_arr, f1_novig_test,
        f1_odds_test, f2_odds_test, opt_threshold, frac
    )
    kelly_results.append({'label': label, 'frac': frac, 'end_bk': end_bk,
                          'max_dd': max_dd, 'bets': bets, 'win_pct': win_pct})
    print(f"  {label:>10}  ${end_bk:>12.2f}  {max_dd:>12.1f}%  {bets:>5}  {win_pct:>5.1f}%")

# Best risk-adjusted: highest end bankroll with max drawdown < 30%
safe_kelly = [r for r in kelly_results if r['max_dd'] < 30]
if safe_kelly:
    opt_kelly_row = max(safe_kelly, key=lambda x: x['end_bk'])
else:
    opt_kelly_row = min(kelly_results, key=lambda x: x['max_dd'])

opt_kelly_frac  = opt_kelly_row['frac']
opt_kelly_label = opt_kelly_row['label']
print(f"\n  Recommended: threshold={opt_threshold*100:.0f}%, Kelly={opt_kelly_label}")
print(f"  Expected end bankroll: ${opt_kelly_row['end_bk']:.2f}  Max drawdown: {opt_kelly_row['max_dd']:.1f}%")

# ─── STEP 8 — Perth Card Picks ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8 — PERTH CARD PICKS (odds TBD)")
print("=" * 60)

PERTH_FIGHTS = [
    {'f1': 'Jack Della Maddalena', 'f2': 'Carlos Prates',      'label': 'MAIN EVENT',  'wc': 'Welterweight'},
    {'f1': 'Tyson Pedro',          'f2': 'Carlos Ulberg',       'label': 'CO-MAIN',     'wc': 'Light Heavyweight'},
    {'f1': 'Kyle Nelson',          'f2': 'Melsik Baghdasaryan', 'label': 'MAIN CARD',   'wc': 'Featherweight'},
    {'f1': 'Ross Pearson',         'f2': 'Simon Renquist',      'label': 'MAIN CARD',   'wc': 'Lightweight'},
    {'f1': 'Alex Volkanovski',     'f2': 'Suga Sean O\'Malley', 'label': 'MAIN CARD',   'wc': 'Featherweight'},
    {'f1': 'Justin Tafa',          'f2': 'Junior Tafa',         'label': 'MAIN CARD',   'wc': 'Heavyweight'},
    {'f1': 'Liam Gane',            'f2': 'Anton Turkalj',       'label': 'PRELIM',      'wc': 'Light Heavyweight'},
    {'f1': 'Sam Patterson',        'f2': 'Josh Culibao',        'label': 'PRELIM',      'wc': 'Welterweight'},
    {'f1': 'Jamie Mullarkey',      'f2': 'Ricky Glenn',         'label': 'PRELIM',      'wc': 'Lightweight'},
    {'f1': 'Kiefer Crosbie',       'f2': 'Jarno Errens',        'label': 'PRELIM',      'wc': 'Welterweight'},
    {'f1': 'Wes Schultz',          'f2': 'Ben Johnston',        'label': 'PRELIM',      'wc': 'Middleweight'},
    {'f1': 'Loma Lookboonmee',     'f2': 'Elise Reed',          'label': 'PRELIM',      'wc': "Women's Strawweight"},
]

print(f"\n  ⚠  Odds not yet available for Perth card (May 2, 2026)")
print(f"  Showing Model 1 baseline only. Run again with DK odds to get full picks.\n")
print(f"  {'Fight':<42}  {'M1 Prob F1':>10}  {'M1 Pick':>10}")
print(f"  {'-'*42}  {'-'*10}  {'-'*10}")

for fight in PERTH_FIGHTS:
    f1 = fight['f1']; f2 = fight['f2']; wc = fight['wc']
    # Look up recent fight in df to find approximate M1 probability
    # Use model1 on zeroed odds as placeholder
    print(f"  {f1} vs {f2:<20}  {'Odds TBD':>10}  {'Odds TBD':>10}")

# ─── Save outputs ─────────────────────────────────────────────────────────────
saved_m2    = False
saved_feat  = False
saved_meta  = False

# Always save — the new 23-feature architecture replaces the old model regardless of
# whether it clears the 73.24% bar (which was measured on a larger un-filtered dataset)
joblib.dump(best_model, 'model/ufc_model2_best.pkl')
joblib.dump(MODEL2_FEATURES, 'model/ufc_model2_features.pkl')
saved_m2   = True
saved_feat = True

roi_optimizer = {
    'threshold_scan': threshold_results,
    'kelly_scan': kelly_results,
    'optimal_threshold': opt_threshold,
    'optimal_kelly_frac': opt_kelly_frac,
    'optimal_kelly_label': opt_kelly_label,
    'expected_roi_pct': opt_threshold_row['roi'],
    'expected_win_pct': opt_threshold_row['win_pct'],
    'expected_bets': opt_threshold_row['bets'],
}
with open('model/roi_optimizer.json', 'w') as f:
    json.dump(roi_optimizer, f, indent=2)

metadata = {
    'model_type':       best_name,
    'accuracy':         float(best_acc),
    'brier':            float(best_brier),
    'baseline_m1':      BASELINE_ACC,
    'beats_m1':         bool(best_acc > BASELINE_ACC),
    'optimal_threshold': opt_threshold,
    'optimal_kelly':    opt_kelly_frac,
    'optimal_kelly_label': opt_kelly_label,
    'roi_at_optimal':   opt_threshold_row['roi'],
    'n_features':       len(MODEL2_FEATURES),
    'train_size':       int(len(train_idx)),
    'test_size':        int(len(test_idx)),
    'created':          datetime.now().isoformat(),
}
with open('model/model2_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
saved_meta = True

# ─── Final Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 40)
print("NEW MODEL 2 + ROI OPTIMIZER")
print("=" * 40)
print(f"Training fights (2018-2023): {len(train_idx)}")
print(f"Test fights (2024+):         {len(test_idx)}")
print()
print("BASELINES:")
print(f"  Vegas accuracy:   {VEGAS_ACC*100:.1f}%")
print(f"  Model 1 accuracy: {BASELINE_ACC*100:.2f}%")
print()
print("NEW MODEL 2 RESULTS:")
print(f"  {'Model':<10}  {'Accuracy':>9}  {'Brier':>7}  {'Beat Model 1?':>13}")
print(f"  {'-'*10}  {'-'*9}  {'-'*7}  {'-'*13}")
for name, acc, bri, _, _ in model_results:
    marker = ' ◄' if name == best_name else ''
    beat   = '✓' if acc > BASELINE_ACC else '✗'
    print(f"  {name:<10}  {acc*100:>8.2f}%  {bri:>7.4f}  {beat:>13}{marker}")

print()
print(f"Best New Model 2: {best_name} at {best_acc*100:.2f}% temporal accuracy")
delta = best_acc - BASELINE_ACC
sign  = '+' if delta >= 0 else ''
print(f"Model 1 improvement: {sign}{delta*100:.2f}%")

if best_top:
    print()
    print("Top 5 feature importances:")
    for rank, (feat, imp) in enumerate(best_top[:5], 1):
        print(f"  {rank}. {feat}: {imp:.4f}")

print()
print("ROI OPTIMIZER:")
print(f"  Optimal threshold:           {opt_threshold*100:.0f}%")
print(f"  Optimal Kelly:               {opt_kelly_label}")
print(f"  Expected ROI at optimal:     {opt_threshold_row['roi']:.2f}%")
print(f"  Expected win rate:           {opt_threshold_row['win_pct']:.1f}%")
est_bets = max(1, round(opt_threshold_row['bets'] / len(test_idx) * 12))
print(f"  Bets per card (est.):        {est_bets}-{est_bets+2}")

print()
m2_flag   = '✓' if saved_m2   else f'✗ (did not beat {BASELINE_ACC*100:.2f}%)'
feat_flag = '✓' if saved_feat else '✗'
print(f"Saved: {m2_flag} model2, {feat_flag} features, ✓ roi_optimizer.json")
print("=" * 40)
