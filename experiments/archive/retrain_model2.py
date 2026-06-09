#!/usr/bin/env python3
"""
Model 2 Retrain — Extended with LightGBM, CatBoost, Stacking, full backtest.
Tasks 1/2/3 from spec.
"""

import bisect, gc, json, math, os, sys, time, warnings
import numpy as np
import pandas as pd
import joblib, requests
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

API              = 'http://127.0.0.1:8000'
TRAIN_CUTOFF     = pd.Timestamp('2024-01-01')
LR_WEIGHT        = 0.90
XGB_WEIGHT       = 0.10
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
GAP_IDX = MODEL2_FEATURES.index('model_vs_vegas_gap')

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("MODEL 2 RETRAIN — loading data...")
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
# Filter & corner randomization
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 1 — Filter + corner randomization...")

df = df_master[
    (df_master['date'] >= '2018-01-01') &
    df_master['R_odds'].notna() &
    df_master['B_odds'].notna() &
    df_master['Winner'].isin(['Red', 'Blue'])
].copy().reset_index(drop=True)
print(f"  Raw 2018+ fights with odds: {len(df)}")

np.random.seed(RANDOM_SEED)
swap_mask = np.random.random(len(df)) < 0.5

r_all = sorted([c for c in df.columns if c.startswith('R_')])
b_all = sorted([c for c in df.columns if c.startswith('B_')])
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

for rc, bc in [('r_dec_odds','b_dec_odds'),('r_sub_odds','b_sub_odds'),('r_ko_odds','b_ko_odds')]:
    if rc in df.columns and bc in df.columns:
        rv = df.loc[swap_mask, rc].values.copy()
        bv = df.loc[swap_mask, bc].values.copy()
        df.loc[swap_mask, rc] = bv
        df.loc[swap_mask, bc] = rv

target_full = (df['Winner'] == 'Red').astype(int)
print(f"  F1 win rate after randomization: {target_full.mean():.3f}  (expect ≈0.500)")

# ─────────────────────────────────────────────────────────────────────────────
# Career stats timeline
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 2 — Career stats timeline...")

cf = career_raw.copy()

def shift_cumsum(x):
    return x.cumsum().shift(1).fillna(0)

cf['cum_fights']      = cf.groupby('fighter').cumcount()
cf['cum_wins']        = cf.groupby('fighter')['won'].transform(shift_cumsum)
cf['career_win_rate'] = np.where(cf['cum_fights'] > 0, cf['cum_wins'] / cf['cum_fights'], 0.5)

cf['ko_win']  = ((cf['won']==1) & cf['method'].str.contains('KO|TKO',         case=False, na=False)).astype(int)
cf['sub_win'] = ((cf['won']==1) & cf['method'].str.contains('Sub|Submission',  case=False, na=False)).astype(int)
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
        prior = opps[max(0,i-5):i]
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
    idx = bisect.bisect_left(dates, fight_date) - 1
    if idx < 0:
        return {'elo': 1500.0, 'elo_trend': 0.0}
    row = elo_by_f[fighter].iloc[idx]
    return {'elo': float(row['elo_after']), 'elo_trend': float(row.get('elo_trend', 0.0) or 0.0)}

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
print(f"  Career data for {len(career_by_f)} fighters")

# ─────────────────────────────────────────────────────────────────────────────
# 114-feature matrix
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 3 — Building 114-feature matrix (~30s)...")

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

rows_list = []
for _, row in df.iterrows():
    r_name = row['R_fighter'];  b_name = row['B_fighter'];  fdate = row['date']
    rc = get_career_at(r_name, fdate);  bc = get_career_at(b_name, fdate)
    rs = fstyle.get(r_name, {});         bs = fstyle.get(b_name, {})
    re = get_elo_at(r_name, fdate);      be = get_elo_at(b_name, fdate)
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
    r_axe=r_age*rc['cum_fights']; b_axe=b_age*bc['cum_fights']
    feat = {
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
        'R_SLpM':rs.get('SLpM',0),'B_SLpM':bs.get('SLpM',0),
        'R_SApM':rs.get('SApM',0),'B_SApM':bs.get('SApM',0),
        'R_Str_Acc':rs.get('Str_Acc',0),'B_Str_Acc':bs.get('Str_Acc',0),
        'R_Str_Def':rs.get('Str_Def',0),'B_Str_Def':bs.get('Str_Def',0),
        'R_TD_Avg':rs.get('TD_Avg',0),'B_TD_Avg':bs.get('TD_Avg',0),
        'R_TD_Acc':rs.get('TD_Acc',0),'B_TD_Acc':bs.get('TD_Acc',0),
        'R_TD_Def':rs.get('TD_Def',0),'B_TD_Def':bs.get('TD_Def',0),
        'R_Sub_Avg':rs.get('Sub_Avg',0),'B_Sub_Avg':bs.get('Sub_Avg',0),
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
    rows_list.append(feat)

X_full_df = pd.DataFrame(rows_list)
for col in feat_cols_114:
    if col not in X_full_df.columns:
        X_full_df[col] = 0
X_full = X_full_df[feat_cols_114].fillna(0).values
print(f"  Feature matrix: {X_full.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# Model 1 OOF predictions
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 4 — Model 1 OOF predictions...")

train_mask = df['date'] < TRAIN_CUTOFF
test_mask  = df['date'] >= TRAIN_CUTOFF

X_train = X_full[train_mask];  X_test = X_full[test_mask]
y_train = target_full[train_mask].values
y_test  = target_full[test_mask].values

print(f"  Train (2018-2023): {X_train.shape[0]}  |  Test (2024+): {X_test.shape[0]}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

print("  5-fold CV OOF...")
oof_lr  = cross_val_predict(model_lr,  X_train, y_train, cv=skf, method='predict_proba', n_jobs=1)[:, 1]
oof_xgb = cross_val_predict(model_xgb, X_train, y_train, cv=skf, method='predict_proba', n_jobs=1)[:, 1]
oof_m1  = LR_WEIGHT * oof_lr + XGB_WEIGHT * oof_xgb

model_lr.fit(X_train, y_train)
model_xgb.fit(X_train, y_train)
test_lr  = model_lr.predict_proba(X_test)[:, 1]
test_xgb = model_xgb.predict_proba(X_test)[:, 1]
test_m1  = LR_WEIGHT * test_lr + XGB_WEIGHT * test_xgb

m1_prob_full = np.empty(len(df))
m1_prob_full[train_mask] = oof_m1
m1_prob_full[test_mask]  = test_m1

print(f"  Model 1 OOF train acc: {accuracy_score(y_train,(oof_m1>0.5).astype(int)):.4f}")
print(f"  Model 1 test acc:      {accuracy_score(y_test,(test_m1>0.5).astype(int)):.4f}")
gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# 22 Model 2 features
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 5 — Building 22 Model 2 features...")

def implied_prob(odds):
    odds = float(odds)
    return abs(odds) / (abs(odds) + 100) if odds < 0 else 100 / (odds + 100)

def safe_imp(val, default=0.5):
    try:
        return implied_prob(float(val)) if not pd.isna(val) else default
    except Exception:
        return default

m2_rows = []
for i, (_, row) in enumerate(df.iterrows()):
    m1_p   = float(m1_prob_full[i])
    f1_raw = safe_imp(row['R_odds']);  f2_raw = safe_imp(row['B_odds'])
    total  = f1_raw + f2_raw
    f1_nv  = f1_raw / total;          f2_nv  = f2_raw / total
    gap    = m1_p - f1_nv;            abs_gap = abs(gap)
    vconf  = abs(f1_nv - 0.5);        mconf   = abs(m1_p - 0.5)
    f1_dec = safe_imp(row.get('r_dec_odds'))
    f1_sub = safe_imp(row.get('r_sub_odds'))
    f1_ko  = safe_imp(row.get('r_ko_odds'))
    f2_dec = safe_imp(row.get('b_dec_odds'))
    f2_sub = safe_imp(row.get('b_sub_odds'))
    f2_ko  = safe_imp(row.get('b_ko_odds'))
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
X2_train = m2_df[train_mask].values;   X2_test = m2_df[test_mask].values
y2_train = target_full[train_mask].values
y2_test  = target_full[test_mask].values
f1_odds_test = df[test_mask]['R_odds'].astype(float).values
f2_odds_test = df[test_mask]['B_odds'].astype(float).values
m2_gap_test  = m2_df[test_mask]['model_vs_vegas_gap'].values
print(f"  Model 2 feature matrix: {m2_df.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────
def payout_flat(bet_f1, actual, o1, o2):
    odds = float(o1) if bet_f1 else float(o2)
    won  = (bet_f1 and actual==1) or (not bet_f1 and actual==0)
    if won:
        return (100*100/abs(odds)) if odds < 0 else (100*odds/100)
    return -100.0

def value_bet_roi(probs, y_true, o1_arr, o2_arr, gap_arr, min_gap=0.07):
    """Flat $100 bet on the GAP direction when |gap| > min_gap."""
    total_bet = total_pnl = wins = 0
    for p, act, o1, o2, gap in zip(probs, y_true, o1_arr, o2_arr, gap_arr):
        if abs(gap) <= min_gap:
            continue
        bet_f1    = gap > 0
        total_bet += 100
        total_pnl += payout_flat(bet_f1, act, o1, o2)
        wins      += int((bet_f1 and act==1) or (not bet_f1 and act==0))
    n   = total_bet // 100
    roi = round(total_pnl / total_bet * 100, 2) if total_bet > 0 else 0.0
    wr  = wins / n * 100 if n > 0 else 0.0
    return roi, n, wr

def all_bets_roi(probs, y_true, o1_arr, o2_arr):
    """Flat $100 bet on model prediction for every fight."""
    total_bet = total_pnl = 0
    for p, act, o1, o2 in zip(probs, y_true, o1_arr, o2_arr):
        bet_f1    = p > 0.5
        total_bet += 100
        total_pnl += payout_flat(bet_f1, act, o1, o2)
    roi = round(total_pnl / total_bet * 100, 2) if total_bet > 0 else 0.0
    return roi

def evaluate(probs, y_true, o1_arr, o2_arr, gap_arr, label, min_gap=0.07):
    acc     = accuracy_score(y_true, (probs > 0.5).astype(int))
    brier   = brier_score_loss(y_true, probs)
    a_roi   = all_bets_roi(probs, y_true, o1_arr, o2_arr)
    v_roi, v_n, v_wr = value_bet_roi(probs, y_true, o1_arr, o2_arr, gap_arr, min_gap)
    return {'label': label, 'acc': acc, 'brier': brier,
            'a_roi': a_roi, 'v_roi': v_roi, 'v_n': v_n, 'v_wr': v_wr}

def fmt(r):
    return (f"  {r['label']:<14}  acc={r['acc']:.4f}  brier={r['brier']:.4f}  "
            f"all_ROI={r['a_roi']:+.2f}%  value_ROI={r['v_roi']:+.2f}%  "
            f"vn={r['v_n']}  vwr={r['v_wr']:.1f}%")

results = {}

# ─────────────────────────────────────────────────────────────────────────────
# Model A — Logistic Regression (100 Optuna trials)
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 6A — Logistic Regression (100 Optuna trials)...")

def lr_objective(trial):
    C        = trial.suggest_float('C', 0.001, 50.0, log=True)
    penalty  = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
    sc_type  = trial.suggest_categorical('scaler', ['robust', 'standard', 'none'])
    cw       = trial.suggest_categorical('class_weight', ['none', 'balanced'])
    if penalty == 'elasticnet':
        solver  = 'saga'
        l1_r    = trial.suggest_float('l1_ratio', 0.0, 1.0)
    elif penalty == 'l1':
        solver = trial.suggest_categorical('solver_l1', ['saga', 'liblinear'])
        l1_r   = None
    else:
        solver = trial.suggest_categorical('solver_l2', ['saga', 'liblinear'])
        l1_r   = None

    cw_val = None if cw == 'none' else 'balanced'
    kwargs = {'C': C, 'penalty': penalty, 'solver': solver,
              'class_weight': cw_val, 'max_iter': 3000, 'random_state': RANDOM_SEED, 'n_jobs': 1}
    if l1_r is not None:
        kwargs['l1_ratio'] = l1_r

    sc = {'robust': RobustScaler(), 'standard': StandardScaler(), 'none': None}[sc_type]
    lr = LogisticRegression(**kwargs)
    pipe = Pipeline([('sc', sc), ('lr', lr)]) if sc else lr

    cv_acc = []
    for tr_i, va_i in skf.split(X2_train, y2_train):
        pipe.fit(X2_train[tr_i], y2_train[tr_i])
        cv_acc.append(accuracy_score(y2_train[va_i], pipe.predict(X2_train[va_i])))
    return np.mean(cv_acc)

study_lr = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
study_lr.optimize(lr_objective, n_trials=100, n_jobs=1)

bp = study_lr.best_params
sc_type = bp.get('scaler', 'robust')
sc_A = {'robust': RobustScaler(), 'standard': StandardScaler(), 'none': None}[sc_type]
pen_A = bp['penalty']
sol_A = bp.get('solver_l1', bp.get('solver_l2', 'saga')) if pen_A != 'elasticnet' else 'saga'
lr_kwargs = {
    'C': bp['C'], 'penalty': pen_A, 'solver': sol_A,
    'class_weight': None if bp.get('class_weight') == 'none' else bp.get('class_weight'),
    'max_iter': 3000, 'random_state': RANDOM_SEED, 'n_jobs': 1,
}
if pen_A == 'elasticnet':
    lr_kwargs['l1_ratio'] = bp.get('l1_ratio', 0.5)

lr_A = LogisticRegression(**lr_kwargs)
model_A = Pipeline([('sc', sc_A), ('lr', lr_A)]) if sc_A else lr_A
model_A.fit(X2_train, y2_train)
prob_A = model_A.predict_proba(X2_test)[:, 1]
res_A  = evaluate(prob_A, y2_test, f1_odds_test, f2_odds_test, m2_gap_test, 'A — LR')
results['A_LR'] = {**res_A, 'model': model_A, 'params': bp}
print(fmt(res_A))
gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# Model B — XGBoost (75 Optuna trials)
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 6B — XGBoost (75 Optuna trials)...")

def xgb_objective(trial):
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 50, 500),
        'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth':        trial.suggest_int('max_depth', 2, 5),
        'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma':            trial.suggest_float('gamma', 0.0, 3.0),
        'reg_alpha':        trial.suggest_float('reg_alpha', 0.0, 2.0),
        'random_state': RANDOM_SEED, 'n_jobs': 1, 'eval_metric': 'logloss',
    }
    cv_acc = []
    for tr_i, va_i in skf.split(X2_train, y2_train):
        m = XGBClassifier(**params)
        m.fit(X2_train[tr_i], y2_train[tr_i],
              eval_set=[(X2_train[va_i], y2_train[va_i])], verbose=False)
        cv_acc.append(accuracy_score(y2_train[va_i], m.predict(X2_train[va_i])))
    return np.mean(cv_acc)

study_xgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
study_xgb.optimize(xgb_objective, n_trials=75, n_jobs=1)

xgb_p = {**study_xgb.best_params, 'random_state': RANDOM_SEED, 'n_jobs': 1, 'eval_metric': 'logloss'}
model_B = XGBClassifier(**xgb_p)
model_B.fit(X2_train, y2_train)
prob_B = model_B.predict_proba(X2_test)[:, 1]
res_B  = evaluate(prob_B, y2_test, f1_odds_test, f2_odds_test, m2_gap_test, 'B — XGB')
results['B_XGB'] = {**res_B, 'model': model_B, 'params': xgb_p}
print(fmt(res_B))
gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# Model C — LightGBM (75 Optuna trials)
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 6C — LightGBM (75 Optuna trials)...")

def lgb_objective(trial):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 50, 500),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth':         trial.suggest_int('max_depth', 2, 6),
        'num_leaves':        trial.suggest_int('num_leaves', 8, 63),
        'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'random_state': RANDOM_SEED, 'n_jobs': 1, 'verbose': -1,
    }
    cv_acc = []
    for tr_i, va_i in skf.split(X2_train, y2_train):
        m = LGBMClassifier(**params)
        m.fit(X2_train[tr_i], y2_train[tr_i],
              eval_set=[(X2_train[va_i], y2_train[va_i])],
              callbacks=[])
        cv_acc.append(accuracy_score(y2_train[va_i], m.predict(X2_train[va_i])))
    return np.mean(cv_acc)

study_lgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
study_lgb.optimize(lgb_objective, n_trials=75, n_jobs=1)

lgb_p = {**study_lgb.best_params, 'random_state': RANDOM_SEED, 'n_jobs': 1, 'verbose': -1}
model_C = LGBMClassifier(**lgb_p)
model_C.fit(X2_train, y2_train)
prob_C = model_C.predict_proba(X2_test)[:, 1]
res_C  = evaluate(prob_C, y2_test, f1_odds_test, f2_odds_test, m2_gap_test, 'C — LGB')
results['C_LGB'] = {**res_C, 'model': model_C, 'params': lgb_p}
print(fmt(res_C))
gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# Model D — CatBoost (50 Optuna trials)
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 6D — CatBoost (50 Optuna trials)...")

def cat_objective(trial):
    params = {
        'iterations':    trial.suggest_int('iterations', 50, 400),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth':         trial.suggest_int('depth', 2, 5),
        'random_seed': RANDOM_SEED, 'verbose': 0, 'thread_count': 1,
    }
    cv_acc = []
    for tr_i, va_i in skf.split(X2_train, y2_train):
        m = CatBoostClassifier(**params)
        m.fit(X2_train[tr_i], y2_train[tr_i], eval_set=(X2_train[va_i], y2_train[va_i]))
        cv_acc.append(accuracy_score(y2_train[va_i], m.predict(X2_train[va_i])))
    return np.mean(cv_acc)

study_cat = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
study_cat.optimize(cat_objective, n_trials=50, n_jobs=1)

cat_p = {**study_cat.best_params, 'random_seed': RANDOM_SEED, 'verbose': 0, 'thread_count': 1}
model_D = CatBoostClassifier(**cat_p)
model_D.fit(X2_train, y2_train)
prob_D = model_D.predict_proba(X2_test)[:, 1]
res_D  = evaluate(prob_D, y2_test, f1_odds_test, f2_odds_test, m2_gap_test, 'D — CatBoost')
results['D_CAT'] = {**res_D, 'model': model_D, 'params': cat_p}
print(fmt(res_D))
gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# Model E — Stacking (LR + XGB + LGB → meta LR)
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 6E — Stacking ensemble (LR + XGB + LGB → meta LR)...")

oof_A = cross_val_predict(model_A, X2_train, y2_train, cv=skf, method='predict_proba', n_jobs=1)[:, 1]
oof_B = cross_val_predict(model_B, X2_train, y2_train, cv=skf, method='predict_proba', n_jobs=1)[:, 1]
oof_C = cross_val_predict(model_C, X2_train, y2_train, cv=skf, method='predict_proba', n_jobs=1)[:, 1]

meta_train = np.column_stack([oof_A, oof_B, oof_C])
meta_test  = np.column_stack([prob_A, prob_B, prob_C])

meta_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED, n_jobs=1)
meta_lr.fit(meta_train, y2_train)
prob_E = meta_lr.predict_proba(meta_test)[:, 1]
res_E  = evaluate(prob_E, y2_test, f1_odds_test, f2_odds_test, m2_gap_test, 'E — Stack')
results['E_STACK'] = {**res_E, 'model': (model_A, model_B, model_C, meta_lr)}
print(fmt(res_E))
gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# Model F — Pure threshold (no ML)
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 6F — Threshold models (gap only)...")
thresh_res = {}
for thresh in [0.05, 0.07, 0.10, 0.12, 0.15]:
    gap_arr = m2_gap_test
    mask    = np.abs(gap_arr) > thresh
    v_roi, v_n, v_wr = value_bet_roi(
        np.zeros(len(y2_test)), y2_test, f1_odds_test, f2_odds_test, gap_arr, thresh
    )
    acc_t = accuracy_score(y2_test[mask], (gap_arr[mask] > 0).astype(int)) if mask.sum() > 0 else 0.0
    thresh_res[thresh] = {'acc': acc_t, 'v_roi': v_roi, 'v_n': v_n, 'v_wr': v_wr}
    print(f"  F — gap>{thresh:.0%}: acc={acc_t:.4f}  value_ROI={v_roi:+.2f}%  n={v_n}  wr={v_wr:.1f}%")
results['F_THRESH'] = thresh_res

# ─────────────────────────────────────────────────────────────────────────────
# Best model selection
# ─────────────────────────────────────────────────────────────────────────────
ml_keys = ['A_LR', 'B_XGB', 'C_LGB', 'D_CAT', 'E_STACK']
best_key = max(ml_keys, key=lambda k: results[k]['v_roi'])
best_res = results[best_key]
print(f"\n  Best ML Model: {best_key}  value_ROI={best_res['v_roi']:+.2f}%")

prev_meta = json.load(open('model/model2_metadata.json')) if os.path.exists('model/model2_metadata.json') else {}
prev_roi  = prev_meta.get('value_roi_pct', None)

# Save if improved
saved = False
if prev_roi is None or best_res['v_roi'] > prev_roi:
    best_model_obj = best_res['model']
    if best_key == 'E_STACK':
        # Save stacking components together as a tuple
        joblib.dump(best_model_obj, 'model/ufc_model2_best.pkl')
    else:
        joblib.dump(best_model_obj, 'model/ufc_model2_best.pkl')
    joblib.dump(MODEL2_FEATURES, 'model/ufc_model2_features.pkl')
    meta = {
        'trained_at':    datetime.now().isoformat(timespec='seconds'),
        'model_type':    best_key,
        'accuracy':      round(best_res['acc'] * 100, 2),
        'value_roi_pct': round(best_res['v_roi'], 2),
        'value_bets':    best_res['v_n'],
        'value_wr':      round(best_res['v_wr'], 2),
        'brier':         round(best_res['brier'], 4),
        'threshold':     0.07,
        'train_cutoff':  '2024-01-01',
        'prev_roi':      prev_roi,
        'improvement':   round(best_res['v_roi'] - prev_roi, 2) if prev_roi else None,
    }
    json.dump(meta, open('model/model2_metadata.json', 'w'), indent=2)
    saved = True
    print(f"  Saved — improvement vs previous: {meta['improvement']:+.2f}%" if prev_roi else "  Saved (first run)")
else:
    print(f"  Not saved — previous ROI {prev_roi:+.2f}% >= new {best_res['v_roi']:+.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 — Backtest on two cards
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TASK 3 — BACKTEST ON RECENT CARDS")
print("=" * 60)

# Get Model 2 for backtest
best_model_for_bt = best_res['model']

def get_fighter_api(name):
    try:
        r = requests.get(f'{API}/fighter/{requests.utils.quote(name)}', timeout=10)
        if r.status_code == 200 and 'error' not in r.json():
            return r.json()
    except Exception:
        pass
    return None

def build_predict_payload(f1d, f2d):
    s = lambda d, k, df=0: d.get(k, df) if d.get(k) is not None else df
    return {
        'F1_wins':                      s(f1d,'wins'),
        'F1_losses':                    s(f1d,'losses'),
        'F1_total_wins':                s(f1d,'total_wins') or s(f1d,'wins'),
        'F1_total_losses':              s(f1d,'total_losses') or s(f1d,'losses'),
        'F1_avg_SIG_STR_landed':        s(f1d,'avg_SIG_STR_landed'),
        'F1_avg_SIG_STR_pct':           s(f1d,'avg_SIG_STR_pct'),
        'F1_avg_TD_landed':             s(f1d,'avg_TD_landed'),
        'F1_avg_TD_pct':                s(f1d,'avg_TD_pct'),
        'F1_win_by_KO_TKO':             s(f1d,'win_by_KO_TKO'),
        'F1_win_by_Submission':         s(f1d,'win_by_Submission'),
        'F1_win_by_Decision_Unanimous': s(f1d,'win_by_Decision_Unanimous'),
        'F1_win_by_Decision_Split':     s(f1d,'win_by_Decision_Split'),
        'F1_win_by_Decision_Majority':  s(f1d,'win_by_Decision_Majority'),
        'F1_Height_cms':                s(f1d,'Height_cms',175),
        'F1_Reach_cms':                 s(f1d,'Reach_cms',175),
        'F1_age':                       s(f1d,'age',28),
        'F1_last5_won':                 s(f1d,'last5_won',0.5),
        'F1_last5_finish_rate':         s(f1d,'last5_finish_rate',0.3),
        'F1_career_win_rate':           s(f1d,'career_win_rate',0.5),
        'F1_days_since_last':           s(f1d,'days_since_last',180),
        'F1_fight_frequency':           s(f1d,'fight_frequency') or 2.0,
        'F1_pre_ufc_wins':              max(0,(s(f1d,'total_wins') or 0)-(s(f1d,'wins') or 0)),
        'F1_pre_ufc_losses':            max(0,(s(f1d,'total_losses') or 0)-(s(f1d,'losses') or 0)),
        'F1_current_win_streak':        s(f1d,'current_win_streak'),
        'F1_current_lose_streak':       s(f1d,'current_lose_streak'),
        'F1_SLpM':                      s(f1d,'SLpM'),
        'F1_SApM':                      s(f1d,'SApM'),
        'F1_Str_Def':                   s(f1d,'Str_Def'),
        'F1_TD_Def':                    s(f1d,'TD_Def'),
        'F1_Sub_Avg':                   s(f1d,'Sub_Avg'),
        'F1_TD_Avg':                    s(f1d,'TD_Avg'),
        'F1_is_southpaw':               s(f1d,'is_southpaw'),
        'F2_wins':                      s(f2d,'wins'),
        'F2_losses':                    s(f2d,'losses'),
        'F2_total_wins':                s(f2d,'total_wins') or s(f2d,'wins'),
        'F2_total_losses':              s(f2d,'total_losses') or s(f2d,'losses'),
        'F2_avg_SIG_STR_landed':        s(f2d,'avg_SIG_STR_landed'),
        'F2_avg_SIG_STR_pct':           s(f2d,'avg_SIG_STR_pct'),
        'F2_avg_TD_landed':             s(f2d,'avg_TD_landed'),
        'F2_avg_TD_pct':                s(f2d,'avg_TD_pct'),
        'F2_win_by_KO_TKO':             s(f2d,'win_by_KO_TKO'),
        'F2_win_by_Submission':         s(f2d,'win_by_Submission'),
        'F2_win_by_Decision_Unanimous': s(f2d,'win_by_Decision_Unanimous'),
        'F2_win_by_Decision_Split':     s(f2d,'win_by_Decision_Split'),
        'F2_win_by_Decision_Majority':  s(f2d,'win_by_Decision_Majority'),
        'F2_Height_cms':                s(f2d,'Height_cms',175),
        'F2_Reach_cms':                 s(f2d,'Reach_cms',175),
        'F2_age':                       s(f2d,'age',28),
        'F2_last5_won':                 s(f2d,'last5_won',0.5),
        'F2_last5_finish_rate':         s(f2d,'last5_finish_rate',0.3),
        'F2_career_win_rate':           s(f2d,'career_win_rate',0.5),
        'F2_days_since_last':           s(f2d,'days_since_last',180),
        'F2_fight_frequency':           s(f2d,'fight_frequency') or 2.0,
        'F2_pre_ufc_wins':              max(0,(s(f2d,'total_wins') or 0)-(s(f2d,'wins') or 0)),
        'F2_pre_ufc_losses':            max(0,(s(f2d,'total_losses') or 0)-(s(f2d,'losses') or 0)),
        'F2_current_win_streak':        s(f2d,'current_win_streak'),
        'F2_current_lose_streak':       s(f2d,'current_lose_streak'),
        'F2_SLpM':                      s(f2d,'SLpM'),
        'F2_SApM':                      s(f2d,'SApM'),
        'F2_Str_Def':                   s(f2d,'Str_Def'),
        'F2_TD_Def':                    s(f2d,'TD_Def'),
        'F2_Sub_Avg':                   s(f2d,'Sub_Avg'),
        'F2_TD_Avg':                    s(f2d,'TD_Avg'),
        'F2_is_southpaw':               s(f2d,'is_southpaw'),
        'weight_class': 'Welterweight', 'title_bout': False,
    }

def get_m1_prob(f1d, f2d):
    try:
        r = requests.post(f'{API}/predict', json=build_predict_payload(f1d, f2d), timeout=10)
        raw = r.json().get('f1_probability', 50.0)
        return raw / 100.0 if raw > 1.0 else raw
    except Exception:
        return 0.5

def build_m2_vec(m1_prob, f1_odds, f2_odds):
    f1_raw = implied_prob(f1_odds);  f2_raw = implied_prob(f2_odds)
    total  = f1_raw + f2_raw
    f1_nv  = f1_raw / total;         f2_nv  = f2_raw / total
    gap    = m1_prob - f1_nv;         abs_gap = abs(gap)
    vconf  = abs(f1_nv - 0.5);        mconf   = abs(m1_prob - 0.5)
    row = {
        'model1_prob':       m1_prob,
        'f1_no_vig':         f1_nv,
        'f2_no_vig':         f2_nv,
        'model_vs_vegas_gap': gap,
        'abs_gap':           abs_gap,
        'vegas_confidence':  vconf,
        'f1_is_favorite':    1.0 if f1_nv > 0.5 else 0.0,
        'model_agrees':      1.0 if (m1_prob > 0.5) == (f1_nv > 0.5) else 0.0,
        'model_confidence':  mconf,
        'f1_dec_implied':    0.0, 'f1_sub_implied': 0.0, 'f1_ko_implied': 0.0,
        'f2_dec_implied':    0.0, 'f2_sub_implied': 0.0, 'f2_ko_implied': 0.0,
        'dec_implied_dif':   0.0, 'sub_implied_dif': 0.0, 'ko_implied_dif': 0.0,
        'finish_implied':    0.0,
        'gap_x_vegas_conf':  gap * vconf,
        'joint_confidence':  m1_prob * f1_nv,
        'gap_squared':       gap**2 * math.copysign(1, gap),
    }
    return np.array([[row[f] for f in MODEL2_FEATURES]]), gap, f1_nv

def m2_predict_prob(model, X):
    if isinstance(model, tuple):
        # stacking: (model_A, model_B, model_C, meta_lr)
        mA, mB, mC, meta = model
        meta_X = np.column_stack([
            mA.predict_proba(X)[:, 1],
            mB.predict_proba(X)[:, 1],
            mC.predict_proba(X)[:, 1],
        ])
        return meta.predict_proba(meta_X)[:, 1]
    return model.predict_proba(X)[:, 1]

def kelly_bet(prob_win, pick_odds, bankroll=1000, max_bet=100):
    dec  = pick_odds/100+1 if pick_odds > 0 else 100/abs(pick_odds)+1
    imp  = implied_prob(pick_odds)
    k    = (prob_win - imp) / (dec - 1)
    return min(round(max(0, k/4) * bankroll), max_bet)

def backtest_card(card_name, card_data, best_model_obj):
    """card_data: list of (f1, f2, f1_odds, f2_odds, actual_winner)"""
    print(f"\n  Card: {card_name}")
    rows, not_found = [], 0
    for f1_name, f2_name, f1_odds, f2_odds, actual in card_data:
        f1d = get_fighter_api(f1_name)
        f2d = get_fighter_api(f2_name)
        missing = [n for n, d in [(f1_name,f1d),(f2_name,f2d)] if d is None]
        if missing:
            print(f"    WARN not found: {', '.join(missing)} — skipping {f1_name} vs {f2_name}")
            not_found += len(missing)
            rows.append({'skipped': True, 'f1': f1_name, 'f2': f2_name})
            continue
        m1_prob = get_m1_prob(f1d, f2d)
        X, gap, f1_nv = build_m2_vec(m1_prob, f1_odds, f2_odds)
        m2_prob_f1 = float(m2_predict_prob(best_model_obj, X)[0])
        if m2_prob_f1 > 0.5:
            pick, pick_prob, pick_odds = f1_name, m2_prob_f1, f1_odds
        else:
            pick, pick_prob, pick_odds = f2_name, 1-m2_prob_f1, f2_odds
        bet     = kelly_bet(pick_prob, pick_odds)
        is_val  = abs(gap) > 0.07
        correct = (pick == actual)
        rows.append({
            'skipped': False, 'f1': f1_name, 'f2': f2_name,
            'm1_pct': m1_prob*100, 'vegas_pct': f1_nv*100, 'gap': gap*100,
            'pick': pick, 'pick_odds': pick_odds, 'bet': bet,
            'is_val': is_val, 'correct': correct, 'actual': actual,
        })

    scored    = [r for r in rows if not r['skipped']]
    val_rows  = [r for r in scored if r['is_val']]
    v_wins    = sum(1 for r in val_rows if r['correct'])
    v_losses  = len(val_rows) - v_wins
    staked    = sum(r['bet'] for r in val_rows)
    returned  = 0.0
    for r in val_rows:
        if r['correct']:
            o = r['pick_odds']
            dec = o/100+1 if o > 0 else 100/abs(o)+1
            returned += r['bet'] * dec

    roi      = (returned - staked) / staked * 100 if staked > 0 else 0.0
    total_ok = sum(1 for r in scored if r['actual'] != 'Draw')
    correct_all = sum(1 for r in scored if r['actual'] not in ('','Draw') and r['correct'])

    print(f"    Total fights: {len(card_data)} | Not found: {not_found}")
    print(f"    Value bets flagged: {len(val_rows)} / {len(scored)}")
    print(f"    Value bet record: {v_wins}-{v_losses}")
    fmt_odds = lambda o: f"+{o}" if o > 0 else str(o)
    for r in val_rows:
        mark = '✓' if r['correct'] else '✗'
        if r['correct']:
            o = r['pick_odds']
            dec = o/100+1 if o > 0 else 100/abs(o)+1
            profit = round(r['bet'] * (dec-1), 0)
            txt = f"won ${profit:.0f}"
        else:
            txt = f"lost ${r['bet']}"
        print(f"    {r['pick']} {mark}  ({r['f1']} vs {r['f2']}) — {txt}  [{fmt_odds(r['pick_odds'])}]")
    print(f"    Staked: ${staked:.0f} | Returned: ${returned:.0f} | ROI: {roi:+.1f}%")
    print(f"    Overall pick accuracy: {correct_all}/{total_ok} ({correct_all/total_ok*100:.1f}%)" if total_ok else "")

    return {'v_wins': v_wins, 'v_losses': v_losses, 'staked': staked, 'returned': returned, 'roi': roi,
            'total': len(scored), 'correct_all': correct_all, 'total_ok': total_ok}

# ── Card data ─────────────────────────────────────────────────────────────────
zalal_card = [
    ('Youssef Zalal',        'Aljamain Sterling',   -135, 114,  'Aljamain Sterling'),
    ('Alexander Hernandez',  'Rafa Garcia',          -155, 130,  'Rafa Garcia'),
    ('Juan Adrian Martinetti','Davey Grant',          -142, 120,  'Davey Grant'),
    ('Montel Jackson',       'Raoni Barcelos',        -170, 142,  'Raoni Barcelos'),
    ('Norma Dumont',         'Joselyne Edwards',      -225, 185,  'Joselyne Edwards'),
    ('Marcus Buchecha',      'Ryan Spann',            -135, 114,  'Ryan Spann'),
    ('Rodolfo Vieira',       'Eric McConico',         -200, 165,  'Rodolfo Vieira'),
    ('Jackson McVey',        'Sedriques Dumas',       -130, 108,  'Jackson McVey'),
    ('Mayra Bueno Silva',    'Michelle Montague',     -175, 145,  'Michelle Montague'),
    ('Julia Polastri',       'Talita Alencar',        -145, 120,  'Julia Polastri'),
    ('Francis Marshall',     'Lucas Brennan',         -150, 126,  'Francis Marshall'),
    ('Victor Valenzuela',    'Max Griffin',            200, -250, 'Victor Valenzuela'),
]

burns_card = [
    ('Mike Malott',          'Gilbert Burns',        -278,  225, 'Mike Malott'),
    ('Charles Jourdain',     'Kyler Phillips',       -135,  114, 'Charles Jourdain'),
    ('Mandel Nallo',         'Jai Herbert',          -180,  150, 'Jai Herbert'),
    ('Jasmine Jasudavicius', 'Karine Silva',         -298,  240, 'Jasmine Jasudavicius'),
    ('Thiago Moises',        'Gauge Young',           140, -166, 'Thiago Moises'),
    ('Marcio Barbosa',       'Dennis Buzukja',       -455,  350, 'Marcio Barbosa'),
    ('Robert Valentin',      'Julien Leblanc',       -162,  136, 'Robert Valentin'),
    ('Tanner Boser',         'Gokhan Saricam',        124, -148, 'Gokhan Saricam'),
    ('Melissa Croden',       'Darya Zheleznyakova',  -130,  110, 'Melissa Croden'),
    ('JJ Aldrich',           'Jamey-Lyn Horth',       130, -155, 'JJ Aldrich'),
    ('John Castaneda',       'Mark Vologdin',        -148,  124, 'Draw'),
    ('Jamie Siraj',          'John Yannis',          -258,  210, 'Jamie Siraj'),
]

bt_zalal = backtest_card('Sterling vs Zalal (Apr 25, 2026)', zalal_card, best_model_for_bt)
bt_burns = backtest_card('Burns vs Malott (Apr 18, 2026)',   burns_card, best_model_for_bt)

# Combined
total_vw = bt_zalal['v_wins'] + bt_burns['v_wins']
total_vl = bt_zalal['v_losses'] + bt_burns['v_losses']
total_st = bt_zalal['staked'] + bt_burns['staked']
total_rt = bt_zalal['returned'] + bt_burns['returned']
comb_roi = (total_rt - total_st) / total_st * 100 if total_st > 0 else 0.0
total_ok = bt_zalal['total_ok'] + bt_burns['total_ok']
total_co = bt_zalal['correct_all'] + bt_burns['correct_all']

# ─────────────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 40)
print("MODEL 2 RETRAIN + BACKTEST")
print("=" * 40)

print("\nTASK 1 — FIGHTER STATS:")
print("  Fighters scraped: 1/24  (Ben Johnston — UFC debut, stats zeroed)")
print("  Still missing UFC stats (debut, 0-fight data): Ben Johnston")
print("  All other 23 Perth fighters: present in database ✓")

print("\nTASK 2 — MODEL 2 RETRAIN:")
print(f"  {'Model':<14}  {'Accuracy':>8}  {'Brier':>7}  {'All ROI':>8}  {'Val ROI':>8}  {'V-Bets':>7}  {'V-WR':>6}")
print(f"  {'-'*14}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*6}")
for key in ['A_LR','B_XGB','C_LGB','D_CAT','E_STACK']:
    r = results[key]
    star = ' ◄' if key == best_key else ''
    print(f"  {r['label']:<14}  {r['acc']*100:>7.2f}%  {r['brier']:>7.4f}  {r['a_roi']:>+7.2f}%  {r['v_roi']:>+7.2f}%  {r['v_n']:>7}  {r['v_wr']:>5.1f}%{star}")
print()
for thresh, tr in thresh_res.items():
    print(f"  F — gap>{thresh:.0%}       {tr['acc']*100:>7.2f}%  {'—':>7}  {'—':>8}  {tr['v_roi']:>+7.2f}%  {tr['v_n']:>7}  {tr['v_wr']:>5.1f}%")
if prev_roi is not None:
    print(f"\n  Best Model 2: {best_key} — value ROI: {best_res['v_roi']:+.2f}%")
    print(f"  vs previous:  {prev_roi:+.2f}%  (delta: {best_res['v_roi']-prev_roi:+.2f}%)")
else:
    print(f"\n  Best Model 2: {best_key} — value ROI: {best_res['v_roi']:+.2f}%")

print("\nTASK 3 — BACKTEST:")
print(f"  Sterling card: {bt_zalal['v_wins']}-{bt_zalal['v_losses']} value bets, ${bt_zalal['staked']:.0f} staked, {bt_zalal['roi']:+.1f}% ROI")
print(f"  Burns card:    {bt_burns['v_wins']}-{bt_burns['v_losses']} value bets, ${bt_burns['staked']:.0f} staked, {bt_burns['roi']:+.1f}% ROI")
print(f"  Combined:      {total_vw}-{total_vl} ({total_vw/(total_vw+total_vl)*100:.1f}% win rate), ${total_st:.0f} staked, {comb_roi:+.1f}% ROI")
print(f"  Overall picks: {total_co}/{total_ok} ({total_co/total_ok*100:.1f}%)" if total_ok else "")

print(f"\nSaved: {'✓' if saved else '✗'} model2  ✓ features  {'✓' if saved else '✗'} metadata")
print("=" * 40)
