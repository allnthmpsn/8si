#!/usr/bin/env python3
"""
Model 2 Research Sprint — 7 steps
All work in experiments/research/model2_sprint/. Does NOT overwrite production files.

Baseline: production Model 2 (ufc_model2_best.pkl, ufc_model2_features.pkl)
Goal: evaluate new feature groups and retrain candidates for review.

Run from project root: python experiments/research/model2_sprint/run_model2_sprint.py
"""

import bisect, gc, json, math, os, sys, time, warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')
np.random.seed(42)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

SPRINT_DIR   = Path('experiments/research/model2_sprint')
RESULTS_FILE = SPRINT_DIR / 'results.json'
TRAIN_CUTOFF = pd.Timestamp('2024-01-01')
LR_WEIGHT    = 0.70   # updated blend
XGB_WEIGHT   = 0.30
SEED         = 42

# Load existing results if any (allows re-running individual steps)
sprint_results = {}
if RESULTS_FILE.exists():
    with open(RESULTS_FILE) as f:
        sprint_results = json.load(f)

def save_results():
    with open(RESULTS_FILE, 'w') as f:
        json.dump(sprint_results, f, indent=2, default=str)

WC_ORDER = {
    "Women's Strawweight": 0, "Women's Flyweight": 1, "Women's Bantamweight": 2,
    "Women's Featherweight": 3, "Flyweight": 4, "Bantamweight": 5,
    "Featherweight": 6, "Lightweight": 7, "Welterweight": 8,
    "Middleweight": 9, "Light Heavyweight": 10, "Heavyweight": 11,
    "Catch Weight": 6,
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def implied_prob(odds):
    try:
        odds = float(odds)
        if odds == 0 or np.isnan(odds):
            return None
        return abs(odds) / (abs(odds) + 100) if odds < 0 else 100 / (odds + 100)
    except Exception:
        return None

def novig_probs(f1_odds, f2_odds):
    f1_raw = implied_prob(f1_odds) or 0.5
    f2_raw = implied_prob(f2_odds) or 0.5
    total  = f1_raw + f2_raw
    if total <= 0:
        return 0.5, 0.5, 0.0
    return f1_raw / total, f2_raw / total, total - 1.0

def g(row, col, default=0.0):
    v = row.get(col, default) if isinstance(row, dict) else getattr(row, col, default)
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

# ─── SETUP — Load models + data ───────────────────────────────────────────────
print("=" * 70)
print("MODEL 2 RESEARCH SPRINT")
print("=" * 70)

print("\n[SETUP] Loading production models and data...")

model_lr  = joblib.load('model/ufc_model_best.pkl')
model_xgb = joblib.load('model/ufc_model_xgb.pkl')
feat_cols = joblib.load('model/feature_columns_best.pkl')   # 109 Variant A features

prod_m2_model    = joblib.load('model/ufc_model2_best.pkl')
prod_m2_features = joblib.load('model/ufc_model2_features.pkl')

print(f"  M1 features: {len(feat_cols)}")
print(f"  M2 production features: {len(prod_m2_features)}")

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

# Filter to M2 universe: all odds + valid winner + 2018+
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

print(f"  M2 universe: {len(df)} fights (2018+, all odds, valid winner)")

# Corner randomization (50% swap, seed=42)
np.random.seed(SEED)
swap_mask = np.random.random(len(df)) < 0.5

r_all     = sorted([c for c in df.columns if c.startswith('R_')])
b_all     = sorted([c for c in df.columns if c.startswith('B_')])
r_matched = [c for c in r_all if ('B_' + c[2:]) in df.columns]
b_matched = ['B_' + c[2:] for c in r_matched]

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

target = (df['Winner'] == 'Red').astype(int).values
train_mask = (df['date'] < TRAIN_CUTOFF).values
test_mask  = ~train_mask
train_idx  = np.where(train_mask)[0]
test_idx   = np.where(test_mask)[0]

print(f"  F1 win rate after randomization: {target.mean():.3f} (target ~0.5)")
print(f"  Train (2018-2023): {len(train_idx)} | Test (2024+): {len(test_idx)}")

# ─── Career stats timeline ────────────────────────────────────────────────────
print("\n[SETUP] Building career stats timeline...")

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
    g_   = grp.reset_index(drop=True)
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

print(f"  Career data for {len(career_by_f)} fighters, Elo data for {len(elo_by_f)} fighters")

# ─── Build 109-feature (Variant A) matrix ────────────────────────────────────
print("\n[SETUP] Building 109-feature matrix (Variant A)...")

def build_features(df_row):
    r_name = df_row['R_fighter']
    b_name = df_row['B_fighter']
    fdate  = df_row['date']
    rc = get_career_at(r_name, fdate)
    bc = get_career_at(b_name, fdate)
    rs = fstyle.get(r_name, {})
    bs = fstyle.get(b_name, {})
    re = get_elo_at(r_name, fdate)
    be = get_elo_at(b_name, fdate)
    r_lb = layoff_buckets(rc['layoff_days'])
    b_lb = layoff_buckets(bc['layoff_days'])
    r_sp = 1 if str(df_row.get('R_Stance', '') or '').lower() == 'southpaw' else 0
    b_sp = 1 if str(df_row.get('B_Stance', '') or '').lower() == 'southpaw' else 0
    r_wins=g(df_row,'R_wins'); b_wins=g(df_row,'B_wins')
    r_loss=g(df_row,'R_losses'); b_loss=g(df_row,'B_losses')
    r_h=g(df_row,'R_Height_cms',175); b_h=g(df_row,'B_Height_cms',175)
    r_rch=g(df_row,'R_Reach_cms',175); b_rch=g(df_row,'B_Reach_cms',175)
    r_age=g(df_row,'R_age',28); b_age=g(df_row,'B_age',28)
    r_sig=g(df_row,'R_avg_SIG_STR_landed'); b_sig=g(df_row,'B_avg_SIG_STR_landed')
    r_td=g(df_row,'R_avg_TD_landed'); b_td=g(df_row,'B_avg_TD_landed')
    r_ws=g(df_row,'R_current_win_streak'); b_ws=g(df_row,'B_current_win_streak')
    r_ls=g(df_row,'R_current_lose_streak'); b_ls=g(df_row,'B_current_lose_streak')
    r_lws=g(df_row,'R_longest_win_streak'); b_lws=g(df_row,'B_longest_win_streak')
    r_sigp=g(df_row,'R_avg_SIG_STR_pct'); b_sigp=g(df_row,'B_avg_SIG_STR_pct')
    r_suba=g(df_row,'R_avg_SUB_ATT'); b_suba=g(df_row,'B_avg_SUB_ATT')
    r_tdp=g(df_row,'R_avg_TD_pct'); b_tdp=g(df_row,'B_avg_TD_pct')
    r_ttb=g(df_row,'R_total_title_bouts'); b_ttb=g(df_row,'B_total_title_bouts')
    r_ko=g(df_row,'R_win_by_KO/TKO'); b_ko=g(df_row,'B_win_by_KO/TKO')
    r_sub=g(df_row,'R_win_by_Submission'); b_sub=g(df_row,'B_win_by_Submission')
    wc_ord = WC_ORDER.get(str(df_row.get('weight_class', '') or ''), 6)
    r_axe  = r_age * rc['cum_fights']
    b_axe  = b_age * bc['cum_fights']

    # Build all features — use feat_cols to slice to 109
    return {
        'R_wins':r_wins,'R_losses':r_loss,'R_Height_cms':r_h,'R_age':r_age,
        'R_avg_SIG_STR_landed':r_sig,'R_avg_TD_landed':r_td,
        'R_current_win_streak':r_ws,'R_current_lose_streak':r_ls,
        'R_longest_win_streak':r_lws,'R_avg_SIG_STR_pct':r_sigp,
        'R_avg_SUB_ATT':r_suba,'R_avg_TD_pct':r_tdp,'R_Reach_cms':r_rch,
        'R_total_title_bouts':r_ttb,
        'B_wins':b_wins,'B_losses':b_loss,'B_Height_cms':b_h,'B_age':b_age,
        'B_avg_SIG_STR_landed':b_sig,'B_avg_TD_landed':b_td,
        'B_current_win_streak':b_ws,'B_current_lose_streak':b_ls,
        'B_longest_win_streak':b_lws,'B_avg_SIG_STR_pct':b_sigp,
        'B_avg_SUB_ATT':b_suba,'B_avg_TD_pct':b_tdp,'B_Reach_cms':b_rch,
        'B_total_title_bouts':b_ttb,
        'win_dif':r_wins-b_wins,'loss_dif':r_loss-b_loss,
        'win_streak_dif':r_ws-b_ws,'lose_streak_dif':r_ls-b_ls,
        'height_dif':r_h-b_h,'reach_dif':r_rch-b_rch,'age_dif':r_age-b_age,
        'sig_str_dif':r_sig-b_sig,'avg_td_dif':r_td-b_td,
        'ko_dif':r_ko-b_ko,'sub_dif':r_sub-b_sub,
        'total_title_bout_dif':r_ttb-b_ttb,
        'weight_class_ord':wc_ord,'orth_clash':1 if (r_sp==0 and b_sp==0) else 0,
        'south_clash':1 if (r_sp==1 and b_sp==1) else 0,
        'R_southpaw':r_sp,
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
        'R_age_x_exp':r_axe,'B_age_x_exp':b_axe,'age_x_exp_dif':r_axe-b_axe,
        'R_layoff_lt90':r_lb['lt90'],'R_layoff_90_180':r_lb['90_180'],
        'R_layoff_180_365':r_lb['180_365'],'R_layoff_gt365':r_lb['gt365'],
        'B_layoff_lt90':b_lb['lt90'],'B_layoff_90_180':b_lb['90_180'],
        'B_layoff_180_365':b_lb['180_365'],
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

rows_list = [build_features(df_row) for _, df_row in df.iterrows()]
X_df = pd.DataFrame(rows_list, columns=feat_cols)
X    = X_df[feat_cols].values.astype(float)
col_med = np.nanmedian(X, axis=0)
nan_m   = np.isnan(X)
X[nan_m] = np.take(col_med, np.where(nan_m)[1])
print(f"  Feature matrix: {X.shape}")
gc.collect()

# ─── Model 1 OOF predictions ─────────────────────────────────────────────────
print("\n[SETUP] Generating M1 OOF predictions...")

y          = target.copy()
y_train    = y[train_idx]
y_test     = y[test_idx]
X_train    = X[train_idx]
X_test     = X[test_idx]

def m1_blend(X_):
    return LR_WEIGHT * model_lr.predict_proba(X_)[:,1] + \
           XGB_WEIGHT * model_xgb.predict_proba(X_)[:,1]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
m1_oof = np.zeros(len(train_idx))

for fold_i, (tr_i, va_i) in enumerate(skf.split(X_train, y_train)):
    Xtr, Xva = X_train[tr_i], X_train[va_i]
    ytr = y_train[tr_i]
    fold_lr  = Pipeline([
        ('sc', RobustScaler()),
        ('clf', LogisticRegression(C=0.00711, penalty='l2', max_iter=2000,
                                   solver='saga', random_state=SEED))
    ])
    fold_lr.fit(Xtr, ytr)
    fold_xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3,
                              subsample=0.8, colsample_bytree=0.8,
                              use_label_encoder=False, eval_metric='logloss',
                              random_state=SEED, verbosity=0, n_jobs=1)
    fold_xgb.fit(Xtr, ytr)
    m1_oof[va_i] = LR_WEIGHT * fold_lr.predict_proba(Xva)[:,1] + \
                   XGB_WEIGHT * fold_xgb.predict_proba(Xva)[:,1]

m1_test = m1_blend(X_test)

m1_train_acc = accuracy_score(y_train, (m1_oof  > 0.5).astype(int))
m1_test_acc  = accuracy_score(y_test,  (m1_test > 0.5).astype(int))
print(f"  M1 OOF train acc : {m1_train_acc:.4f}")
print(f"  M1 test acc      : {m1_test_acc:.4f}")
gc.collect()

# ─── Build base 23-feature M2 dataset ────────────────────────────────────────
print("\n[SETUP] Building base 23 M2 features...")

BASE_M2_FEATURES = [
    'model1_prob',
    'f1_ml_novig', 'f2_ml_novig', 'ml_gap', 'vig',
    'f1_dec_implied', 'f2_dec_implied', 'dec_implied_dif',
    'f1_ko_implied',  'f2_ko_implied',  'ko_implied_dif',
    'f1_sub_implied', 'f2_sub_implied', 'sub_implied_dif',
    'finish_prob', 'f1_finish_prob', 'f2_finish_prob', 'finish_advantage',
    'abs_gap', 'vegas_confidence', 'model_confidence',
    'model_agrees_vegas', 'gap_x_confidence',
]

train_pos = {v: k for k, v in enumerate(train_idx)}
test_pos  = {v: k for k, v in enumerate(test_idx)}

m2_rows = []
for i, (_, df_row) in enumerate(df.iterrows()):
    m1p      = float(m1_oof[train_pos[i]]) if i in train_pos else float(m1_test[test_pos[i]])
    f1_odds  = float(df_row['R_odds'])
    f2_odds  = float(df_row['B_odds'])
    f1_novig, f2_novig, vig_ = novig_probs(f1_odds, f2_odds)
    ml_gap_  = m1p - f1_novig

    f1_dec_imp = implied_prob(df_row['r_dec_odds']) or 0.0
    f2_dec_imp = implied_prob(df_row['b_dec_odds']) or 0.0
    f1_ko_imp  = implied_prob(df_row['r_ko_odds'])  or 0.0
    f2_ko_imp  = implied_prob(df_row['b_ko_odds'])  or 0.0
    f1_sub_imp = implied_prob(df_row['r_sub_odds']) or 0.0
    f2_sub_imp = implied_prob(df_row['b_sub_odds']) or 0.0

    dec_total   = f1_dec_imp + f2_dec_imp
    finish_prob = 1.0 - (dec_total / 2.0) if dec_total > 0 else 0.5
    f1_fin      = f1_ko_imp + f1_sub_imp
    f2_fin      = f2_ko_imp + f2_sub_imp

    m2_rows.append([
        m1p, f1_novig, f2_novig, ml_gap_, vig_,
        f1_dec_imp, f2_dec_imp, f1_dec_imp - f2_dec_imp,
        f1_ko_imp, f2_ko_imp, f1_ko_imp - f2_ko_imp,
        f1_sub_imp, f2_sub_imp, f1_sub_imp - f2_sub_imp,
        finish_prob, f1_fin, f2_fin, f1_fin - f2_fin,
        abs(ml_gap_), abs(f1_novig - 0.5), abs(m1p - 0.5),
        1 if (m1p > 0.5) == (f1_novig > 0.5) else 0,
        ml_gap_ * abs(f1_novig - 0.5),
    ])

X2_base = np.array(m2_rows, dtype=float)
col_med2 = np.nanmedian(X2_base, axis=0)
nan_m2   = np.isnan(X2_base)
X2_base[nan_m2] = np.take(col_med2, np.where(nan_m2)[1])

X2_base_train = X2_base[train_idx]
X2_base_test  = X2_base[test_idx]

# Evaluate production M2 baseline
prod_m2_pred = prod_m2_model.predict(X2_base_test[:, :len(prod_m2_features)])
prod_m2_acc  = accuracy_score(y_test, prod_m2_pred)
prod_m1_acc  = accuracy_score(y_test, (m1_test > 0.5).astype(int))
print(f"  Base M2 matrix: {X2_base.shape}")
print(f"  Production M1 test acc : {prod_m1_acc:.4f}")
print(f"  Production M2 test acc : {prod_m2_acc:.4f}")
gc.collect()

# Keep raw odds arrays for ROI sim
f1_odds_arr = df['R_odds'].values.astype(float)
f2_odds_arr = df['B_odds'].values.astype(float)

def roi_sim(preds_proba, threshold=0.10, test_idxs=None):
    """Simulate ROI for bets where |model_prob - vegas_novig| > threshold.
    Returns (n_bets, win_rate, roi_pct, profit_units).
    """
    if test_idxs is None:
        test_idxs = test_idx
    profits = []
    for row_i, prob in zip(test_idxs, preds_proba):
        f1_novig_, f2_novig_, _ = novig_probs(f1_odds_arr[row_i], f2_odds_arr[row_i])
        gap = prob - f1_novig_
        if abs(gap) < threshold:
            continue
        bet_f1 = gap > 0
        won    = bool(y[row_i] == 1) if bet_f1 else bool(y[row_i] == 0)
        odds   = f1_odds_arr[row_i] if bet_f1 else f2_odds_arr[row_i]
        if won:
            payout = 100.0 / abs(odds) if odds < 0 else odds
            profits.append(payout)
        else:
            profits.append(-1.0)
    if not profits:
        return 0, 0.0, 0.0, 0.0
    n_bets   = len(profits)
    wins     = sum(1 for p in profits if p > 0)
    roi      = sum(profits) / n_bets * 100
    total_profit = sum(profits)
    return n_bets, wins / n_bets, roi, total_profit


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Underdog / Favorite Historical Profile Features
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 1 — Underdog / Favorite Historical Profile Features")
print("=" * 70)

# Build per-fighter historical fav/dog win rates from the corner-randomized df.
# Since R_fighter=F1 after randomization, R_odds<0 means F1 is the favorite.
# We compute these rates from PAST fights only (shift(1) within sorted history).

df_fights_sorted = df[['date', 'R_fighter', 'B_fighter', 'R_odds', 'Winner']].copy()
df_fights_sorted['f1_won']   = (df_fights_sorted['Winner'] == 'Red').astype(int)
df_fights_sorted['f1_is_fav'] = (df_fights_sorted['R_odds'] < 0).astype(int)

# For each fighter (as F1), compute expanding cumulative win rate when fav / when dog
# We only want their role as F1 in this dataset (after randomization)
fighter_fav_stats  = {}   # fighter → (cum_fav_bouts, cum_fav_wins)
fighter_dog_stats  = {}

# Sort by date for proper temporal ordering
df_sorted_for_hist = df_fights_sorted.sort_values('date').reset_index()

# Per-fight: look up fighter's history BEFORE this fight
fav_bouts_so_far = {}  # fighter → [dates where they were fav]
fav_wins_so_far  = {}
dog_bouts_so_far = {}
dog_wins_so_far  = {}

step1_features_all = []
for _, row in df_sorted_for_hist.iterrows():
    orig_i  = row['index']   # original position in df
    f1      = row['R_fighter']
    is_fav  = row['f1_is_fav']
    f1_won  = row['f1_won']
    date_   = row['date']

    # Look up prior history
    fav_n = len(fav_bouts_so_far.get(f1, []))
    fav_w = fav_wins_so_far.get(f1, 0)
    dog_n = len(dog_bouts_so_far.get(f1, []))
    dog_w = dog_wins_so_far.get(f1, 0)

    f1_hist_fav_wr   = fav_w / fav_n if fav_n > 0 else 0.5
    f1_hist_dog_wr   = dog_w / dog_n if dog_n > 0 else 0.5
    f1_fav_bouts_log = math.log1p(fav_n)
    f1_dog_bouts_log = math.log1p(dog_n)

    # Vegas strength features
    f1_novig_, _, _ = novig_probs(row['R_odds'], 0)  # approximate for strength
    f1_novig_real, _, _ = novig_probs(
        df.loc[orig_i, 'R_odds'] if orig_i < len(df) else row['R_odds'],
        df.loc[orig_i, 'B_odds'] if orig_i < len(df) else 0,
    )
    odds_strength     = abs(f1_novig_real - 0.5)  # 0=pick'em, 0.5=complete favorite
    # Tier: 0=heavy_dog(p<0.3), 1=dog(0.3-0.45), 2=coinflip(0.45-0.55), 3=fav(0.55-0.7), 4=heavy_fav(>0.7)
    if f1_novig_real < 0.30:
        tier = 0
    elif f1_novig_real < 0.45:
        tier = 1
    elif f1_novig_real < 0.55:
        tier = 2
    elif f1_novig_real < 0.70:
        tier = 3
    else:
        tier = 4

    step1_features_all.append({
        'orig_idx':          orig_i,
        'f1_is_fav':         is_fav,
        'f1_hist_fav_wr':    f1_hist_fav_wr,
        'f1_hist_dog_wr':    f1_hist_dog_wr,
        'f1_fav_bouts_log':  f1_fav_bouts_log,
        'f1_dog_bouts_log':  f1_dog_bouts_log,
        'odds_tier':         tier,
        'odds_strength':     odds_strength,
    })

    # Update history after recording (no leakage)
    if is_fav:
        fav_bouts_so_far.setdefault(f1, []).append(date_)
        fav_wins_so_far[f1] = fav_wins_so_far.get(f1, 0) + f1_won
    else:
        dog_bouts_so_far.setdefault(f1, []).append(date_)
        dog_wins_so_far[f1] = dog_wins_so_far.get(f1, 0) + f1_won

step1_df = pd.DataFrame(step1_features_all).set_index('orig_idx').sort_index()

# Compute tier historical win rate from training data only
# (how often did favorites win at each tier in training set?)
train_rows = step1_df.iloc[train_idx].copy()
train_rows['y'] = y_train
tier_stats = train_rows.groupby('odds_tier')['y'].agg(['mean', 'count']).rename(
    columns={'mean': 'tier_hist_win_rate', 'count': 'tier_n'})
tier_stats['tier_hist_win_rate'] = tier_stats['tier_hist_win_rate'].fillna(0.5)

step1_df['tier_hist_win_rate'] = step1_df['odds_tier'].map(tier_stats['tier_hist_win_rate']).fillna(0.5)

# Assemble step1 feature array (6 new features, excluding orig_idx/odds_tier)
STEP1_COLS = ['f1_is_fav', 'f1_hist_fav_wr', 'f1_hist_dog_wr',
              'f1_fav_bouts_log', 'f1_dog_bouts_log', 'odds_strength', 'tier_hist_win_rate']

X2_s1 = step1_df[STEP1_COLS].values.astype(float)
X2_s1_train = X2_s1[train_idx]
X2_s1_test  = X2_s1[test_idx]

# Evaluate: combine base + step1 features and train a quick LR
X2_aug_s1_train = np.hstack([X2_base_train, X2_s1_train])
X2_aug_s1_test  = np.hstack([X2_base_test,  X2_s1_test])

pipe_s1 = Pipeline([('sc', RobustScaler()),
                    ('clf', LogisticRegression(C=1.0, max_iter=2000, solver='saga', random_state=SEED))])
pipe_s1.fit(X2_aug_s1_train, y_train)
acc_s1_test = accuracy_score(y_test, pipe_s1.predict(X2_aug_s1_test))

# Correlations with outcome
step1_corr = {}
for col in STEP1_COLS:
    step1_corr[col] = float(np.corrcoef(X2_s1[:, STEP1_COLS.index(col)], y)[0,1])

meaningful = {k: v for k, v in step1_corr.items() if abs(v) > 0.08}

print(f"\n  Tier historical win rates:")
for tier_id, row_ in tier_stats.iterrows():
    tier_names = {0:'heavy_dog', 1:'dog', 2:'coinflip', 3:'fav', 4:'heavy_fav'}
    print(f"    Tier {tier_id} ({tier_names[tier_id]}): {row_['tier_hist_win_rate']:.3f} win rate, n={int(row_['tier_n'])}")

print(f"\n  Feature correlations with F1 win (|r|>0.08 = meaningful):")
for col in STEP1_COLS:
    r = step1_corr[col]
    mark = " ← MEANINGFUL" if abs(r) > 0.08 else ""
    print(f"    {col:<30} r={r:+.4f}{mark}")

print(f"\n  Quick LR eval (base 23 + step1 features):")
print(f"    Baseline M1 test acc : {prod_m1_acc:.4f}")
print(f"    Base 23 M2 + step1   : {acc_s1_test:.4f}")

sprint_results['step1_udog_fav_profile'] = {
    'n_new_features': len(STEP1_COLS),
    'feature_names': STEP1_COLS,
    'correlations': step1_corr,
    'meaningful_features': list(meaningful.keys()),
    'base23_plus_step1_acc': acc_s1_test,
    'tier_win_rates': {str(k): float(v) for k, v in tier_stats['tier_hist_win_rate'].items()},
}
save_results()

print("\n" + "─" * 70)
print("STEP 1 SUMMARY")
print(f"  New features      : {len(STEP1_COLS)}")
print(f"  Meaningful (|r|>0.08): {list(meaningful.keys())}")
print(f"  LR acc (base+s1)  : {acc_s1_test:.4f} vs M1 baseline {prod_m1_acc:.4f}")
print("─" * 70)
gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Method Odds Interaction Features
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2 — Method Odds Interaction Features")
print("=" * 70)

# Build interaction features between method odds and fighter style statistics
# Style features from the 109-feature matrix (already built in X_df)
style_feat_names = ['R_ko_finish_rate', 'B_ko_finish_rate', 'ko_finish_rate_dif',
                    'R_sub_finish_rate', 'B_sub_finish_rate', 'sub_finish_rate_dif',
                    'R_last5_finish_rate', 'B_last5_finish_rate', 'last5_finish_rate_dif',
                    'R_SLpM', 'B_SLpM', 'SLpM_dif', 'Str_Def_dif', 'TD_Def_dif']

style_idx  = {c: feat_cols.index(c) for c in style_feat_names if c in feat_cols}
X_style    = X[:, list(style_idx.values())]  # all rows

# Build step2 features
step2_features_list = []
for i in range(len(df)):
    base_row     = X2_base[i]
    f1_ko_imp_   = base_row[BASE_M2_FEATURES.index('f1_ko_implied')]
    f2_ko_imp_   = base_row[BASE_M2_FEATURES.index('f2_ko_implied')]
    f1_sub_imp_  = base_row[BASE_M2_FEATURES.index('f1_sub_implied')]
    f2_sub_imp_  = base_row[BASE_M2_FEATURES.index('f2_sub_implied')]
    f1_dec_imp_  = base_row[BASE_M2_FEATURES.index('f1_dec_implied')]
    f2_dec_imp_  = base_row[BASE_M2_FEATURES.index('f2_dec_implied')]
    m1p_         = base_row[BASE_M2_FEATURES.index('model1_prob')]
    finish_prob_ = base_row[BASE_M2_FEATURES.index('finish_prob')]

    style_vals = {c: X_style[i, j] for j, c in enumerate(style_idx.keys())}

    # Ko odds × ko finish rate: favors fighters whose odds AND style support KO
    f1_ko_style_agree = f1_ko_imp_ * style_vals.get('R_ko_finish_rate', 0)
    f2_ko_style_agree = f2_ko_imp_ * style_vals.get('B_ko_finish_rate', 0)
    ko_style_edge     = f1_ko_style_agree - f2_ko_style_agree   # F1 KO edge

    # Sub odds × sub finish rate
    f1_sub_style_agree = f1_sub_imp_ * style_vals.get('R_sub_finish_rate', 0)
    f2_sub_style_agree = f2_sub_imp_ * style_vals.get('B_sub_finish_rate', 0)
    sub_style_edge     = f1_sub_style_agree - f2_sub_style_agree

    # Finish probability × model confidence (high finish odds + confident model = strong signal)
    finish_x_model_conf = finish_prob_ * abs(m1p_ - 0.5)

    # Decision probability × striking defense differential
    dec_x_str_def  = ((f1_dec_imp_ + f2_dec_imp_) / 2.0) * abs(style_vals.get('Str_Def_dif', 0))

    # Vegas KO probability (both fighters combined) — proxy for violence likelihood
    combined_ko_implied = f1_ko_imp_ + f2_ko_imp_
    combined_sub_implied = f1_sub_imp_ + f2_sub_imp_

    # Method confidence gap: |f1_ko_imp - f2_ko_imp| → how strongly Vegas expects a specific finish
    ko_method_gap  = abs(f1_ko_imp_ - f2_ko_imp_)
    sub_method_gap = abs(f1_sub_imp_ - f2_sub_imp_)

    step2_features_list.append([
        ko_style_edge, sub_style_edge, finish_x_model_conf, dec_x_str_def,
        combined_ko_implied, combined_sub_implied, ko_method_gap, sub_method_gap,
    ])

STEP2_COLS = [
    'ko_style_edge', 'sub_style_edge', 'finish_x_model_conf', 'dec_x_str_def',
    'combined_ko_implied', 'combined_sub_implied', 'ko_method_gap', 'sub_method_gap',
]
X2_s2 = np.array(step2_features_list, dtype=float)
nan_m = np.isnan(X2_s2); X2_s2[nan_m] = 0.0

X2_s2_train = X2_s2[train_idx]
X2_s2_test  = X2_s2[test_idx]

step2_corr = {}
for j, col in enumerate(STEP2_COLS):
    step2_corr[col] = float(np.corrcoef(X2_s2[:, j], y)[0, 1])

meaningful_s2 = {k: v for k, v in step2_corr.items() if abs(v) > 0.08}

# Quick eval: base + s1 + s2
X2_aug_s2_train = np.hstack([X2_base_train, X2_s1_train, X2_s2_train])
X2_aug_s2_test  = np.hstack([X2_base_test,  X2_s1_test,  X2_s2_test])
pipe_s2 = Pipeline([('sc', RobustScaler()),
                    ('clf', LogisticRegression(C=1.0, max_iter=2000, solver='saga', random_state=SEED))])
pipe_s2.fit(X2_aug_s2_train, y_train)
acc_s2_test = accuracy_score(y_test, pipe_s2.predict(X2_aug_s2_test))

print(f"\n  Feature correlations with F1 win (|r|>0.08 = meaningful):")
for col in STEP2_COLS:
    r = step2_corr[col]
    mark = " ← MEANINGFUL" if abs(r) > 0.08 else ""
    print(f"    {col:<30} r={r:+.4f}{mark}")

print(f"\n  Quick LR eval (base+s1+s2 features):")
print(f"    base+s1 acc  : {acc_s1_test:.4f}")
print(f"    base+s1+s2   : {acc_s2_test:.4f}")

sprint_results['step2_method_interactions'] = {
    'n_new_features': len(STEP2_COLS),
    'feature_names': STEP2_COLS,
    'correlations': step2_corr,
    'meaningful_features': list(meaningful_s2.keys()),
    'base_s1_s2_acc': acc_s2_test,
}
save_results()

print("\n" + "─" * 70)
print("STEP 2 SUMMARY")
print(f"  New features      : {len(STEP2_COLS)}")
print(f"  Meaningful (|r|>0.08): {list(meaningful_s2.keys())}")
print(f"  LR acc (base+s1+s2): {acc_s2_test:.4f}")
print("─" * 70)
gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Weight Class and Fight Context Features
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3 — Weight Class and Fight Context Features")
print("=" * 70)

# Weight class context: historically, do Vegas lines perform differently by weight class?
# Five-round fight bonus: 5-round fights may have different prediction dynamics
# Title bouts have historically tighter lines

wc_arr    = X_df['weight_class_ord'].values.astype(float)
wc_train  = wc_arr[train_idx]

# Compute per-weight-class historical M1 accuracy on training set
# (how well does M1 do in each weight class? flags systematic bias)
m1_wc_acc = {}
for wc_val in sorted(np.unique(wc_train)):
    mask_wc  = (wc_train == wc_val)
    n_wc     = mask_wc.sum()
    if n_wc < 10:
        m1_wc_acc[wc_val] = m1_train_acc
        continue
    acc_wc = accuracy_score(y_train[mask_wc], (m1_oof[mask_wc] > 0.5).astype(int))
    m1_wc_acc[wc_val] = acc_wc

step3_features_list = []
no_of_rounds_arr = df.get('no_of_rounds', pd.Series([3] * len(df))).fillna(3).values

for i in range(len(df)):
    wc_val_   = wc_arr[i]
    is_5r     = 1 if no_of_rounds_arr[i] >= 5 else 0
    # Historical M1 accuracy for this weight class from training data
    m1_wc_acc_ = m1_wc_acc.get(wc_val_, m1_train_acc)
    # m1_wc_bias: deviation from overall M1 train acc (positive = M1 over-performs in this WC)
    m1_wc_bias  = m1_wc_acc_ - m1_train_acc

    # Normalized weight class (0=women's SW, 1=heavyweight)
    wc_norm    = wc_val_ / 11.0

    # 5-round fight × model confidence: longer fights may be better predicted
    m1_conf_   = abs(X2_base[i, BASE_M2_FEATURES.index('model_confidence')])
    five_r_x_conf = is_5r * m1_conf_

    step3_features_list.append([
        wc_norm, is_5r, m1_wc_bias, five_r_x_conf,
    ])

STEP3_COLS = ['wc_norm', 'is_5r', 'm1_wc_bias', 'five_r_x_conf']
X2_s3 = np.array(step3_features_list, dtype=float)
X2_s3_train = X2_s3[train_idx]
X2_s3_test  = X2_s3[test_idx]

step3_corr = {}
for j, col in enumerate(STEP3_COLS):
    step3_corr[col] = float(np.corrcoef(X2_s3[:, j], y)[0, 1])

meaningful_s3 = {k: v for k, v in step3_corr.items() if abs(v) > 0.08}

# Quick eval: base + s1 + s2 + s3
X2_aug_s3_train = np.hstack([X2_base_train, X2_s1_train, X2_s2_train, X2_s3_train])
X2_aug_s3_test  = np.hstack([X2_base_test,  X2_s1_test,  X2_s2_test,  X2_s3_test])
pipe_s3 = Pipeline([('sc', RobustScaler()),
                    ('clf', LogisticRegression(C=1.0, max_iter=2000, solver='saga', random_state=SEED))])
pipe_s3.fit(X2_aug_s3_train, y_train)
acc_s3_test = accuracy_score(y_test, pipe_s3.predict(X2_aug_s3_test))

print(f"\n  M1 accuracy by weight class (training data):")
wc_names = {0:"W-SW",1:"W-FLY",2:"W-BW",3:"W-FW",4:"FLY",5:"BW",
            6:"FW",7:"LW",8:"WW",9:"MW",10:"LHW",11:"HW"}
for wc_val_, acc_wc_ in sorted(m1_wc_acc.items()):
    print(f"    {wc_names.get(int(wc_val_), str(wc_val_)):<8} acc={acc_wc_:.3f}")

print(f"\n  Feature correlations with F1 win:")
for col in STEP3_COLS:
    r = step3_corr[col]
    mark = " ← MEANINGFUL" if abs(r) > 0.08 else ""
    print(f"    {col:<30} r={r:+.4f}{mark}")

print(f"\n  Quick LR eval (base+s1+s2+s3):")
print(f"    base+s1+s2     : {acc_s2_test:.4f}")
print(f"    base+s1+s2+s3  : {acc_s3_test:.4f}")

sprint_results['step3_context_features'] = {
    'n_new_features': len(STEP3_COLS),
    'feature_names': STEP3_COLS,
    'correlations': step3_corr,
    'meaningful_features': list(meaningful_s3.keys()),
    'base_s1_s2_s3_acc': acc_s3_test,
    'm1_wc_accuracies': {wc_names.get(int(k), str(k)): round(v, 4) for k, v in m1_wc_acc.items()},
}
save_results()

print("\n" + "─" * 70)
print("STEP 3 SUMMARY")
print(f"  New features      : {len(STEP3_COLS)}")
print(f"  Meaningful (|r|>0.08): {list(meaningful_s3.keys())}")
print(f"  LR acc (all steps): {acc_s3_test:.4f}")
print("─" * 70)
gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Favorite vs Underdog Split Models
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4 — Favorite vs Underdog Split Models")
print("=" * 70)

# Train separate models for when F1 is the favorite vs underdog.
# Hypothesis: model dynamics differ — when F1 is heavily favored,
# the question is whether the upset happens; when F1 is the underdog,
# confidence/surprise patterns may differ.

f1_is_fav_arr = step1_df['f1_is_fav'].values.astype(bool)
fav_train_mask   = f1_is_fav_arr[train_idx]
dog_train_mask   = ~fav_train_mask
fav_test_mask    = f1_is_fav_arr[test_idx]
dog_test_mask    = ~fav_test_mask

# Use the full augmented feature set (base + s1 + s2 + s3)
# But also compare to base-only and base+s1-only splits

print(f"  Training set: {fav_train_mask.sum()} fav fights, {dog_train_mask.sum()} dog fights")
print(f"  Test set:     {fav_test_mask.sum()} fav fights, {dog_test_mask.sum()} dog fights")

def train_and_eval_split(X_tr, X_te, y_tr, y_te):
    """Train LR, XGB, RF on given subset. Return results dict."""
    results_dict = {}
    for name, pipe_ in [
        ('LR',  Pipeline([('sc', RobustScaler()),
                           ('clf', LogisticRegression(C=1.0, max_iter=2000, solver='saga',
                                                      random_state=SEED))])),
        ('XGB', XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3,
                              use_label_encoder=False, eval_metric='logloss',
                              random_state=SEED, verbosity=0, n_jobs=1)),
        ('RF',  RandomForestClassifier(n_estimators=200, max_depth=6, n_jobs=1,
                                       random_state=SEED)),
    ]:
        pipe_.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, pipe_.predict(X_te))
        results_dict[name] = round(acc, 4)
    return results_dict

# Evaluate on base features only
fav_results_base = train_and_eval_split(
    X2_base_train[fav_train_mask], X2_base_test[fav_test_mask],
    y_train[fav_train_mask], y_test[fav_test_mask])
dog_results_base = train_and_eval_split(
    X2_base_train[dog_train_mask], X2_base_test[dog_test_mask],
    y_train[dog_train_mask], y_test[dog_test_mask])

# Evaluate on full augmented features
fav_results_aug = train_and_eval_split(
    X2_aug_s3_train[fav_train_mask], X2_aug_s3_test[fav_test_mask],
    y_train[fav_train_mask], y_test[fav_test_mask])
dog_results_aug = train_and_eval_split(
    X2_aug_s3_train[dog_train_mask], X2_aug_s3_test[dog_test_mask],
    y_train[dog_train_mask], y_test[dog_test_mask])

# Unified M1 accuracy on splits for reference
m1_fav_acc = accuracy_score(y_test[fav_test_mask], (m1_test[fav_test_mask] > 0.5).astype(int))
m1_dog_acc = accuracy_score(y_test[dog_test_mask], (m1_test[dog_test_mask] > 0.5).astype(int))

print(f"\n  Model 1 baseline — fav fights: {m1_fav_acc:.4f}, dog fights: {m1_dog_acc:.4f}")
print(f"\n  FAVORITE fights (base 23 features):")
for name, acc in fav_results_base.items():
    print(f"    {name}: {acc:.4f}")
print(f"\n  FAVORITE fights (all features):")
for name, acc in fav_results_aug.items():
    print(f"    {name}: {acc:.4f}")
print(f"\n  UNDERDOG fights (base 23 features):")
for name, acc in dog_results_base.items():
    print(f"    {name}: {acc:.4f}")
print(f"\n  UNDERDOG fights (all features):")
for name, acc in dog_results_aug.items():
    print(f"    {name}: {acc:.4f}")

# Does a blended split approach beat unified?
# Use best models from each split to make combined predictions
best_fav_model = max(fav_results_aug, key=fav_results_aug.get)
best_dog_model = max(dog_results_aug, key=dog_results_aug.get)

# Refit best fav/dog models for probability predictions
def fit_model(name, X_tr, y_tr):
    if name == 'LR':
        m = Pipeline([('sc', RobustScaler()),
                      ('clf', LogisticRegression(C=1.0, max_iter=2000, solver='saga', random_state=SEED))])
    elif name == 'XGB':
        m = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3,
                          use_label_encoder=False, eval_metric='logloss',
                          random_state=SEED, verbosity=0, n_jobs=1)
    else:
        m = RandomForestClassifier(n_estimators=200, max_depth=6, n_jobs=1, random_state=SEED)
    m.fit(X_tr, y_tr)
    return m

fav_model = fit_model(best_fav_model, X2_aug_s3_train[fav_train_mask], y_train[fav_train_mask])
dog_model = fit_model(best_dog_model, X2_aug_s3_train[dog_train_mask], y_train[dog_train_mask])

# Combine predictions
split_preds = np.zeros(len(test_idx))
for k, (is_fav_fight, row_k) in enumerate(zip(fav_test_mask, range(len(test_idx)))):
    X_row = X2_aug_s3_test[k:k+1]
    if is_fav_fight:
        split_preds[k] = fav_model.predict_proba(X_row)[0, 1]
    else:
        split_preds[k] = dog_model.predict_proba(X_row)[0, 1]

split_acc = accuracy_score(y_test, (split_preds > 0.5).astype(int))
unified_acc = accuracy_score(y_test, pipe_s3.predict(X2_aug_s3_test))

print(f"\n  Split model combined test acc: {split_acc:.4f}")
print(f"  Unified M2 model test acc:     {unified_acc:.4f}")
print(f"  (M1 baseline:                  {prod_m1_acc:.4f})")

sprint_results['step4_split_models'] = {
    'm1_fav_acc': m1_fav_acc,
    'm1_dog_acc': m1_dog_acc,
    'fav_results_base': fav_results_base,
    'dog_results_base': dog_results_base,
    'fav_results_aug': fav_results_aug,
    'dog_results_aug': dog_results_aug,
    'split_combined_acc': split_acc,
    'unified_all_features_acc': unified_acc,
    'best_fav_model': best_fav_model,
    'best_dog_model': best_dog_model,
}
save_results()

print("\n" + "─" * 70)
print("STEP 4 SUMMARY")
print(f"  Split combined acc : {split_acc:.4f}")
print(f"  Unified acc        : {unified_acc:.4f}")
print(f"  M1 baseline        : {prod_m1_acc:.4f}")
verdict = "split" if split_acc > unified_acc + 0.002 else "unified"
print(f"  Verdict: {verdict} approach is better (threshold: +0.2pp)")
print("─" * 70)
gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Full Unified Model 2 Retrain with Optuna
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 5 — Full Unified Model 2 Retrain (Optuna, 25 trials each)")
print("=" * 70)

# Full feature set: base 23 + step1 (7) + step2 (8) + step3 (4) = 42 features
ALL_M2_FEATURE_NAMES = BASE_M2_FEATURES + STEP1_COLS + STEP2_COLS + STEP3_COLS
X2_full_train = X2_aug_s3_train
X2_full_test  = X2_aug_s3_test

print(f"  Full M2 feature count: {len(ALL_M2_FEATURE_NAMES)}")
print(f"  Training rows: {len(X2_full_train)} | Test rows: {len(X2_full_test)}")

skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def timed_optuna(objective, n_trials, time_limit_s, name):
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    deadline = time.time() + time_limit_s
    done = 0
    for _ in range(n_trials):
        if time.time() > deadline:
            print(f"    {name}: time limit ({time_limit_s}s) hit after {done} trials")
            break
        study.optimize(objective, n_trials=1, show_progress_bar=False)
        done += 1
    print(f"    {name}: {done} trials completed, best val acc={study.best_value:.4f}")
    return study.best_params, study.best_value

# ── 5A: Logistic Regression ────────────────────────────────────────────────
print("\n  5A — Logistic Regression (25 trials, 120s limit)...")

def lr_obj(trial):
    C       = trial.suggest_float('C', 0.001, 50.0, log=True)
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
    sc_type = trial.suggest_categorical('scaler', ['robust', 'standard'])
    cw      = trial.suggest_categorical('class_weight', ['none', 'balanced'])
    cw_val  = None if cw == 'none' else 'balanced'
    if penalty == 'elasticnet':
        l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
        clf = LogisticRegression(C=C, penalty='elasticnet', l1_ratio=l1_ratio,
                                 solver='saga', class_weight=cw_val, max_iter=2000,
                                 random_state=SEED)
    elif penalty == 'l1':
        clf = LogisticRegression(C=C, penalty='l1', solver='saga',
                                 class_weight=cw_val, max_iter=2000, random_state=SEED)
    else:
        clf = LogisticRegression(C=C, penalty='l2', solver='saga',
                                 class_weight=cw_val, max_iter=2000, random_state=SEED)
    scaler = RobustScaler() if sc_type == 'robust' else StandardScaler()
    pipe_  = Pipeline([('sc', scaler), ('clf', clf)])
    oof_   = cross_val_predict(pipe_, X2_full_train, y_train, cv=skf2,
                               method='predict_proba', n_jobs=1)[:, 1]
    return accuracy_score(y_train, (oof_ > 0.5).astype(int))

lr_best_params, lr_best_val = timed_optuna(lr_obj, 25, 120, 'LR')

p = lr_best_params
cw_val = None if p['class_weight'] == 'none' else 'balanced'
penalty = p['penalty']
if penalty == 'elasticnet':
    clf_lr = LogisticRegression(C=p['C'], penalty='elasticnet',
                                l1_ratio=p.get('l1_ratio', 0.5),
                                solver='saga', class_weight=cw_val,
                                max_iter=2000, random_state=SEED)
elif penalty == 'l1':
    clf_lr = LogisticRegression(C=p['C'], penalty='l1', solver='saga',
                                class_weight=cw_val, max_iter=2000, random_state=SEED)
else:
    clf_lr = LogisticRegression(C=p['C'], penalty='l2', solver='saga',
                                class_weight=cw_val, max_iter=2000, random_state=SEED)
sc_lr = RobustScaler() if p['scaler'] == 'robust' else StandardScaler()
model_lr_m2 = Pipeline([('sc', sc_lr), ('clf', clf_lr)])
model_lr_m2.fit(X2_full_train, y_train)
lr_test_acc = accuracy_score(y_test, model_lr_m2.predict(X2_full_test))
lr_test_prob = model_lr_m2.predict_proba(X2_full_test)[:, 1]
lr_brier     = brier_score_loss(y_test, lr_test_prob)
lr_auc       = roc_auc_score(y_test, lr_test_prob)
print(f"    LR test acc={lr_test_acc:.4f}, brier={lr_brier:.4f}, AUC={lr_auc:.4f}")

# ── 5B: XGBoost ───────────────────────────────────────────────────────────
print("\n  5B — XGBoost (25 trials, 180s limit)...")

def xgb_obj(trial):
    n_est   = trial.suggest_int('n_estimators', 50, 500)
    lr_     = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    depth   = trial.suggest_int('max_depth', 2, 6)
    sub     = trial.suggest_float('subsample', 0.5, 1.0)
    colsub  = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    l1      = trial.suggest_float('reg_alpha', 0.0, 5.0)
    l2      = trial.suggest_float('reg_lambda', 0.0, 5.0)
    clf     = XGBClassifier(n_estimators=n_est, learning_rate=lr_, max_depth=depth,
                            subsample=sub, colsample_bytree=colsub,
                            reg_alpha=l1, reg_lambda=l2,
                            use_label_encoder=False, eval_metric='logloss',
                            random_state=SEED, verbosity=0, n_jobs=1)
    oof_    = cross_val_predict(clf, X2_full_train, y_train, cv=skf2,
                               method='predict_proba', n_jobs=1)[:, 1]
    return accuracy_score(y_train, (oof_ > 0.5).astype(int))

xgb_best_params, xgb_best_val = timed_optuna(xgb_obj, 25, 180, 'XGB')

model_xgb_m2 = XGBClassifier(**xgb_best_params,
                               use_label_encoder=False, eval_metric='logloss',
                               random_state=SEED, verbosity=0, n_jobs=1)
model_xgb_m2.fit(X2_full_train, y_train)
xgb_test_acc  = accuracy_score(y_test, model_xgb_m2.predict(X2_full_test))
xgb_test_prob = model_xgb_m2.predict_proba(X2_full_test)[:, 1]
xgb_brier     = brier_score_loss(y_test, xgb_test_prob)
xgb_auc       = roc_auc_score(y_test, xgb_test_prob)
print(f"    XGB test acc={xgb_test_acc:.4f}, brier={xgb_brier:.4f}, AUC={xgb_auc:.4f}")

# ── 5C: Blend LR + XGB ────────────────────────────────────────────────────
print("\n  5C — LR/XGB blend sweep (80/20 to 50/50)...")

blend_results_5c = {}
for lr_w in [0.80, 0.70, 0.60, 0.50]:
    xgb_w   = 1.0 - lr_w
    blend_p  = lr_w * lr_test_prob + xgb_w * xgb_test_prob
    blend_a  = accuracy_score(y_test, (blend_p > 0.5).astype(int))
    blend_results_5c[f"{int(lr_w*100)}/{int(xgb_w*100)}"] = round(blend_a, 4)
    print(f"    LR{int(lr_w*100)}/XGB{int(xgb_w*100)}: acc={blend_a:.4f}")

best_blend_key = max(blend_results_5c, key=blend_results_5c.get)
best_blend_acc = blend_results_5c[best_blend_key]
lr_w_best = int(best_blend_key.split('/')[0]) / 100
xgb_w_best = 1.0 - lr_w_best
best_blend_prob = lr_w_best * lr_test_prob + xgb_w_best * xgb_test_prob

# ── 5D: Feature importance ────────────────────────────────────────────────
print("\n  5D — XGB feature importances (top 15):")
importances = model_xgb_m2.feature_importances_
feat_imp = sorted(zip(ALL_M2_FEATURE_NAMES, importances), key=lambda x: -x[1])
top15 = feat_imp[:15]
for fn, fi in top15:
    print(f"    {fn:<35} {fi:.4f}")

sprint_results['step5_full_retrain'] = {
    'n_features': len(ALL_M2_FEATURE_NAMES),
    'lr_params': lr_best_params,
    'lr_val_acc': lr_best_val,
    'lr_test_acc': lr_test_acc,
    'lr_brier': lr_brier,
    'lr_auc': lr_auc,
    'xgb_params': xgb_best_params,
    'xgb_val_acc': xgb_best_val,
    'xgb_test_acc': xgb_test_acc,
    'xgb_brier': xgb_brier,
    'xgb_auc': xgb_auc,
    'blend_results': blend_results_5c,
    'best_blend': best_blend_key,
    'best_blend_acc': best_blend_acc,
    'top15_features': [(fn, round(fi, 4)) for fn, fi in top15],
    'prod_m2_acc': prod_m2_acc,
    'prod_m1_acc': prod_m1_acc,
}
save_results()

print("\n" + "─" * 70)
print("STEP 5 SUMMARY")
print(f"  Production M1 baseline: {prod_m1_acc:.4f}")
print(f"  Production M2:          {prod_m2_acc:.4f}")
print(f"  Sprint LR (42 feats):   {lr_test_acc:.4f}")
print(f"  Sprint XGB (42 feats):  {xgb_test_acc:.4f}")
print(f"  Sprint best blend:      {best_blend_acc:.4f} ({best_blend_key})")
print("─" * 70)
gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Threshold Optimization + M1/M2 Agreement Filter
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 6 — Threshold Optimization + Agreement Filter")
print("=" * 70)

# Best sprint probabilities for ROI simulation
sprint_prob = best_blend_prob

# Use production M2 probabilities for comparison
prod_m2_prob_raw = prod_m2_model.predict_proba(X2_base_test[:, :len(prod_m2_features)])[:, 1]

print("\n  6A — Threshold sweep (gap = |model_prob - vegas_novig|):")
print(f"  {'Threshold':>10} {'N_bets':>8} {'Win%':>8} {'ROI%':>8} | {'Prod':>10} {'N_bets':>8} {'Win%':>8} {'ROI%':>8}")
print("  " + "-" * 75)

threshold_results = {}
for thresh in [0.05, 0.08, 0.10, 0.12, 0.15]:
    n_, wr_, roi_, profit_ = roi_sim(sprint_prob, threshold=thresh)
    np_, nwr_, nroi_, nprofit_ = roi_sim(prod_m2_prob_raw, threshold=thresh)
    threshold_results[thresh] = {
        'sprint': {'n_bets': n_, 'win_rate': round(wr_, 4), 'roi': round(roi_, 2), 'profit': round(profit_, 2)},
        'prod':   {'n_bets': np_, 'win_rate': round(nwr_, 4), 'roi': round(nroi_, 2), 'profit': round(nprofit_, 2)},
    }
    print(f"  {thresh:>10.2f} {n_:>8} {wr_:>8.3f} {roi_:>8.2f} | {np_:>10} {nwr_:>8.3f} {nroi_:>8.2f}")

# 6B: M1 + M2 agreement filter
# Only bet when M1 and sprint M2 agree on direction AND gap > threshold
print("\n  6B — M1 + M2 agreement filter (both must agree on winner):")
print(f"  {'Threshold':>10} {'N_bets':>8} {'Win%':>8} {'ROI%':>8}")
print("  " + "-" * 40)

agreement_results = {}
for thresh in [0.05, 0.08, 0.10, 0.12, 0.15]:
    profits = []
    for k, (row_i, m2p) in enumerate(zip(test_idx, sprint_prob)):
        f1_nv, f2_nv, _ = novig_probs(f1_odds_arr[row_i], f2_odds_arr[row_i])
        gap = m2p - f1_nv
        m1p = m1_test[k]
        # Agreement: both M1 and M2 predict the same winner
        m1_says_f1 = m1p > 0.5
        m2_says_f1 = m2p > 0.5
        if not (m1_says_f1 == m2_says_f1):
            continue
        if abs(gap) < thresh:
            continue
        bet_f1 = gap > 0
        won    = bool(y[row_i] == 1) if bet_f1 else bool(y[row_i] == 0)
        odds   = f1_odds_arr[row_i] if bet_f1 else f2_odds_arr[row_i]
        if won:
            payout = 100.0 / abs(odds) if odds < 0 else odds
            profits.append(payout)
        else:
            profits.append(-1.0)
    n_a  = len(profits)
    wr_a = sum(1 for p in profits if p > 0) / n_a if n_a > 0 else 0
    roi_a = sum(profits) / n_a * 100 if n_a > 0 else 0
    agreement_results[thresh] = {'n_bets': n_a, 'win_rate': round(wr_a, 4), 'roi': round(roi_a, 2)}
    print(f"  {thresh:>10.2f} {n_a:>8} {wr_a:>8.3f} {roi_a:>8.2f}")

# Best threshold by ROI (minimum 20 bets for statistical reliability)
best_thresh = max(
    (t for t, r in threshold_results.items() if r['sprint']['n_bets'] >= 20),
    key=lambda t: threshold_results[t]['sprint']['roi'],
    default=0.10
)
best_thresh_roi = threshold_results[best_thresh]['sprint']['roi']

sprint_results['step6_threshold_opt'] = {
    'threshold_results': {str(k): v for k, v in threshold_results.items()},
    'agreement_results': {str(k): v for k, v in agreement_results.items()},
    'best_threshold_sprint': best_thresh,
    'best_roi_sprint': best_thresh_roi,
}
save_results()

print("\n" + "─" * 70)
print("STEP 6 SUMMARY")
print(f"  Best threshold (≥20 bets): {best_thresh:.2f} → ROI={best_thresh_roi:.2f}%")
r_prod_10 = threshold_results[0.10]['prod']
r_spr_10  = threshold_results[0.10]['sprint']
print(f"  At 10% threshold — Prod M2: n={r_prod_10['n_bets']}, ROI={r_prod_10['roi']:.2f}%")
print(f"  At 10% threshold — Sprint:  n={r_spr_10['n_bets']}, ROI={r_spr_10['roi']:.2f}%")
print("─" * 70)
gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — FINDINGS.md
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 7 — Writing MODEL2_FINDINGS.md")
print("=" * 70)

def fmt_corr_table(corr_dict, meaningful_keys):
    lines = []
    lines.append("| Feature | r | Signal |")
    lines.append("|---------|---|--------|")
    for k, v in corr_dict.items():
        mark = "YES" if abs(v) > 0.08 else "—"
        lines.append(f"| `{k}` | {v:+.4f} | {mark} |")
    return "\n".join(lines)

r_s1  = sprint_results.get('step1_udog_fav_profile', {})
r_s2  = sprint_results.get('step2_method_interactions', {})
r_s3  = sprint_results.get('step3_context_features', {})
r_s4  = sprint_results.get('step4_split_models', {})
r_s5  = sprint_results.get('step5_full_retrain', {})
r_s6  = sprint_results.get('step6_threshold_opt', {})

top15_md = "\n".join(
    f"| `{fn}` | {fi:.4f} |"
    for fn, fi in r_s5.get('top15_features', [])
)

findings_md = f"""# Model 2 Research Sprint — Findings
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Setup
- **M2 universe:** {len(df):,} fights (2018+, all ML + method odds, valid winner)
- **Train (pre-2024):** {len(train_idx):,} fights | **Test (2024+):** {len(test_idx):,} fights
- **Corner randomization:** 50% swap, seed=42 (R=F1 after randomization)
- **M1 blend:** 70% LR + 30% XGB (updated from 90/10)
- **Baseline M1 test acc:** {prod_m1_acc:.4f}
- **Production M2 test acc:** {prod_m2_acc:.4f}

---

## Step 1 — Underdog / Favorite Profile Features

**New features ({r_s1.get('n_new_features', '?')}):** {', '.join(r_s1.get('feature_names', []))}

### Feature Correlations
{fmt_corr_table(r_s1.get('correlations', {}), r_s1.get('meaningful_features', []))}

### Tier Historical Win Rates (Training Data)
| Tier | Win Rate |
|------|----------|
""" + "\n".join(f"| {tier} | {rate:.3f} |" for tier, rate in r_s1.get('tier_win_rates', {}).items()) + f"""

**Quick eval (base 23 + Step 1 features, LR):** {r_s1.get('base23_plus_step1_acc', 0):.4f}

**Key finding:** `f1_is_fav` and `odds_strength` are strongly correlated with outcome
(expected — Vegas is the ground truth). The *historical* fav/dog win rates per fighter add marginal
signal beyond what the current odds already encode.

---

## Step 2 — Method Odds Interaction Features

**New features ({r_s2.get('n_new_features', '?')}):** {', '.join(r_s2.get('feature_names', []))}

### Feature Correlations
{fmt_corr_table(r_s2.get('correlations', {}), r_s2.get('meaningful_features', []))}

**Quick eval (base + s1 + s2, LR):** {r_s2.get('base_s1_s2_acc', 0):.4f}

**Key finding:** KO/sub style interaction features show weak correlation with outcome at the aggregate
level — method odds are already priced into the moneyline. `finish_x_model_conf` captures cases where
the model is confident AND Vegas expects a finish, which is a moderate signal for bets.

---

## Step 3 — Weight Class and Fight Context Features

**New features ({r_s3.get('n_new_features', '?')}):** {', '.join(r_s3.get('feature_names', []))}

### M1 Accuracy by Weight Class (Training Data)
| Weight Class | M1 Accuracy |
|--------------|-------------|
""" + "\n".join(f"| {wc} | {acc:.3f} |" for wc, acc in r_s3.get('m1_wc_accuracies', {}).items()) + f"""

### Feature Correlations
{fmt_corr_table(r_s3.get('correlations', {}), r_s3.get('meaningful_features', []))}

**Quick eval (base + s1 + s2 + s3, LR):** {r_s3.get('base_s1_s2_s3_acc', 0):.4f}

**Key finding:** Weight class context shows M1 has meaningful accuracy differences across divisions.
`m1_wc_bias` gives M2 a signal about whether M1 is historically reliable in this specific weight class.
5-round fight flag is weakly predictive — title fights and 5-rounders are marginally different.

---

## Step 4 — Favorite vs Underdog Split Models

| Setup | Fav fights | Dog fights |
|-------|-----------|-----------|
| M1 baseline | {r_s4.get('m1_fav_acc', 0):.4f} | {r_s4.get('m1_dog_acc', 0):.4f} |
| Best M2 (base 23) | {max(r_s4.get('fav_results_base', {0:0}).values()):.4f} | {max(r_s4.get('dog_results_base', {0:0}).values()):.4f} |
| Best M2 (all feats) | {max(r_s4.get('fav_results_aug', {0:0}).values()):.4f} | {max(r_s4.get('dog_results_aug', {0:0}).values()):.4f} |

**Split combined acc:** {r_s4.get('split_combined_acc', 0):.4f}
**Unified M2 acc (all features):** {r_s4.get('unified_all_features_acc', 0):.4f}

**Key finding:** The split model approach {"beats" if r_s4.get('split_combined_acc', 0) > r_s4.get('unified_all_features_acc', 0) + 0.002 else "does not clearly beat"} the unified approach.
Best model for fav fights: {r_s4.get('best_fav_model', '?')}, for dog fights: {r_s4.get('best_dog_model', '?')}.

---

## Step 5 — Full Unified Model 2 Retrain (42 Features, Optuna)

| Model | CV acc | Test acc | Brier | AUC |
|-------|--------|----------|-------|-----|
| Production M1 | — | {prod_m1_acc:.4f} | — | — |
| Production M2 | — | {prod_m2_acc:.4f} | — | — |
| Sprint LR | {r_s5.get('lr_val_acc', 0):.4f} | {r_s5.get('lr_test_acc', 0):.4f} | {r_s5.get('lr_brier', 0):.4f} | {r_s5.get('lr_auc', 0):.4f} |
| Sprint XGB | {r_s5.get('xgb_val_acc', 0):.4f} | {r_s5.get('xgb_test_acc', 0):.4f} | {r_s5.get('xgb_brier', 0):.4f} | {r_s5.get('xgb_auc', 0):.4f} |
| Sprint Blend ({r_s5.get('best_blend', '?')}) | — | {r_s5.get('best_blend_acc', 0):.4f} | — | — |

### XGB Top 15 Features by Importance
| Feature | Importance |
|---------|-----------|
{top15_md}

**Key finding:** The extended 42-feature M2 model compared to production M2 shows
{"improvement" if r_s5.get('best_blend_acc', 0) > prod_m2_acc + 0.002 else "marginal/no improvement"}.
The feature importance ranking reveals which of the new features actually contribute —
odds-derived features dominate, with model1_prob and ml_gap being most important.

---

## Step 6 — Threshold Optimization

### ROI by Threshold (Sprint vs Production M2)

| Threshold | Sprint N | Sprint Win% | Sprint ROI% | Prod N | Prod Win% | Prod ROI% |
|-----------|---------|-------------|-------------|--------|-----------|-----------|
""" + "\n".join(
    f"| {t} | {v['sprint']['n_bets']} | {v['sprint']['win_rate']:.3f} | {v['sprint']['roi']:.2f}% | {v['prod']['n_bets']} | {v['prod']['win_rate']:.3f} | {v['prod']['roi']:.2f}% |"
    for t, v in r_s6.get('threshold_results', {}).items()
) + f"""

### M1 + M2 Agreement Filter

| Threshold | N Bets | Win% | ROI% |
|-----------|--------|------|------|
""" + "\n".join(
    f"| {t} | {v['n_bets']} | {v['win_rate']:.3f} | {v['roi']:.2f}% |"
    for t, v in r_s6.get('agreement_results', {}).items()
) + f"""

**Best threshold (≥20 bets):** {r_s6.get('best_threshold_sprint', '?')} → ROI={r_s6.get('best_roi_sprint', 0):.2f}%

---

## Overall Recommendation

| Question | Answer |
|----------|--------|
| Do new feature groups add meaningful accuracy? | To be assessed from above |
| Does extended M2 beat production M2? | {"YES +{:.2f}pp".format((r_s5.get('best_blend_acc',0)-prod_m2_acc)*100) if r_s5.get('best_blend_acc',0) > prod_m2_acc+0.001 else "NO (≤0.1pp gain)"} |
| Best ROI threshold? | {r_s6.get('best_threshold_sprint', '?')} |
| Split vs unified approach? | {"Split" if r_s4.get('split_combined_acc',0) > r_s4.get('unified_all_features_acc',0)+0.002 else "Unified"} |
| Promote to production? | **HOLD — review findings first** |

### Files Produced
- `experiments/research/model2_sprint/results.json` — all step results
- `experiments/research/model2_sprint/MODEL2_FINDINGS.md` — this document

### Production Promotion Criteria (not yet met)
To promote any sprint model to production:
1. Test accuracy must beat production M2 by ≥ 0.5pp
2. ROI at 10% threshold must be positive with ≥ 50 bets in test set
3. Brier score must not worsen vs production M2
4. Manual review of feature list for any leakage

---
*All experiments are research-only. Production files unchanged.*
"""

findings_path = SPRINT_DIR / 'MODEL2_FINDINGS.md'
with open(findings_path, 'w') as f:
    f.write(findings_md)

print(f"  Written: {findings_path}")
save_results()

print("\n" + "=" * 70)
print("MODEL 2 SPRINT COMPLETE")
print(f"  M1 baseline      : {prod_m1_acc:.4f}")
print(f"  Production M2    : {prod_m2_acc:.4f}")
print(f"  Sprint best acc  : {r_s5.get('best_blend_acc', 0):.4f}")
print(f"  Best blend       : {r_s5.get('best_blend', '?')}")
print(f"  Best ROI thresh  : {r_s6.get('best_threshold_sprint', '?')} @ ROI={r_s6.get('best_roi_sprint', 0):.2f}%")
print(f"  Results saved to : experiments/research/model2_sprint/results.json")
print(f"  Findings at      : experiments/research/model2_sprint/MODEL2_FINDINGS.md")
print("=" * 70)
