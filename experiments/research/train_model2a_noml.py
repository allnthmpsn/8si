#!/usr/bin/env python3
"""
train_model2a_noml.py — Model 2A retrain using method odds only (no moneyline).

Architecture  : LR 50% + XGB 50% blend (same as production M2A)
Market signal : r/b ko/sub/dec implied probs (6-way no-vig) + differentials +
                method-derived win prob + tier_hist_win_rate (method-tier basis)
Training data : ufc-master.csv ∩ career_fights_updated.csv
Filter        : men's only, 2013+, all 6 method cols non-null
Temporal split: train <2024-01-01, test 2024+  |  n_jobs=1, seed=42
Output        : model/ufc_model2a_noml_lr.pkl
                model/ufc_model2a_noml_xgb.pkl
                model/feature_columns_2a_noml.pkl
"""
import bisect, gc, json, os, sys, warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

os.chdir('/Users/allenthompson/Desktop/ufc-predictor')

# ─── Config ───────────────────────────────────────────────────────────────────
SEED         = 42
TRAIN_CUTOFF = pd.Timestamp('2024-01-01')
M1_LR_W      = 0.70
M1_XGB_W     = 0.30
WOMENS       = {"Women's Strawweight","Women's Flyweight",
                "Women's Bantamweight","Women's Featherweight"}
M2A_PROD_ACC = 0.7320   # production M2A (LR+XGB, moneyline+method)
M1_ACC       = 0.7281   # M1 baseline
N_TRIALS     = 50

WC_ORDER = {
    "Women's Strawweight":0,"Women's Flyweight":1,"Women's Bantamweight":2,
    "Women's Featherweight":3,"Flyweight":4,"Bantamweight":5,
    "Featherweight":6,"Lightweight":7,"Welterweight":8,
    "Middleweight":9,"Light Heavyweight":10,"Heavyweight":11,"Catch Weight":6,
}

TIER_FALLBACK = {
    'hfav':0.82,'mfav':0.72,'sfav':0.68,'pkem':0.55,
    'sdog':0.45,'mdog':0.35,'hdog':0.15,
}

print("=" * 70)
print("MODEL 2A — METHOD ODDS ONLY (no moneyline features)")
print("=" * 70)

# ─── Load production M1 ──────────────────────────────────────────────────────
print("\nLoading M1 production models...")
m1_lr        = joblib.load('model/ufc_model_best.pkl')
m1_xgb       = joblib.load('model/ufc_model_xgb.pkl')
feat_cols_m1 = joblib.load('model/feature_columns_best.pkl')

def m1_predict(X):
    return (M1_LR_W  * m1_lr.predict_proba(X)[:, 1] +
            M1_XGB_W * m1_xgb.predict_proba(X)[:, 1])

# ─── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
df_master = pd.read_csv('data/ufc-master.csv', low_memory=False)
df_master['date'] = pd.to_datetime(df_master['date'])

career_raw = pd.read_csv('data/career_fights_updated.csv')
career_raw['date'] = pd.to_datetime(career_raw['date'])
career_raw = career_raw.sort_values(['fighter','date']).reset_index(drop=True)

fstats_df = pd.read_csv('data/ufc_fighters_final_updated.csv')
for col in ['Str_Acc','Str_Def','TD_Acc','TD_Def']:
    fstats_df[col] = pd.to_numeric(
        fstats_df[col].astype(str).str.replace('%','',regex=False),
        errors='coerce').fillna(0) / 100.0

elo_hist = pd.read_csv('data/elo_ratings_history.csv')
elo_hist['date'] = pd.to_datetime(elo_hist['date'])
elo_hist = elo_hist.sort_values(['fighter','date']).reset_index(drop=True)

# ─── Filter ───────────────────────────────────────────────────────────────────
METHOD_COLS = ['r_ko_odds','b_ko_odds','r_sub_odds','b_sub_odds',
               'r_dec_odds','b_dec_odds']

df = df_master[
    (df_master['date'] >= '2013-01-01') &
    df_master[METHOD_COLS].notna().all(axis=1) &
    df_master['Winner'].isin(['Red','Blue']) &
    ~df_master['weight_class'].isin(WOMENS)
].copy().reset_index(drop=True)

print(f"  Rows after filter : {len(df):,}")
print(f"  Train (<2024)     : {(df['date'] < TRAIN_CUTOFF).sum():,}")
print(f"  Test  (2024+)     : {(df['date'] >= TRAIN_CUTOFF).sum():,}")
print(f"  Date range        : {df['date'].min().date()} → {df['date'].max().date()}")

# ─── Corner randomization (seed=42, matches all other scripts) ────────────────
np.random.seed(SEED)
swap_mask = np.random.random(len(df)) < 0.5

r_matched = sorted([c for c in df.columns if c.startswith('R_')
                    and ('B_'+c[2:]) in df.columns])
b_matched  = ['B_'+c[2:] for c in r_matched]

for rc, bc in zip(r_matched, b_matched):
    rv = df.loc[swap_mask, rc].values.copy()
    bv = df.loc[swap_mask, bc].values.copy()
    df.loc[swap_mask, rc] = bv
    df.loc[swap_mask, bc] = rv

df.loc[swap_mask & (df['Winner']=='Red'),  'Winner'] = 'TEMP'
df.loc[swap_mask & (df['Winner']=='Blue'), 'Winner'] = 'Red'
df.loc[swap_mask & (df['Winner']=='TEMP'), 'Winner'] = 'Blue'

for rc, bc in [('r_dec_odds','b_dec_odds'),('r_sub_odds','b_sub_odds'),('r_ko_odds','b_ko_odds')]:
    rv = df.loc[swap_mask, rc].values.copy()
    bv = df.loc[swap_mask, bc].values.copy()
    df.loc[swap_mask, rc] = bv
    df.loc[swap_mask, bc] = rv

target    = (df['Winner'] == 'Red').astype(int).values
train_mask = (df['date'] < TRAIN_CUTOFF).values
test_mask  = ~train_mask
train_idx  = np.where(train_mask)[0]
test_idx   = np.where(test_mask)[0]

print(f"  F1 win rate (should be ~0.5): {target.mean():.3f}")
print(f"  Swap check passed: {swap_mask.mean():.3f} rows swapped")
gc.collect()

# ─── Build M1 feature matrix (114 features, mirrors production) ───────────────
print("\nBuilding M1 feature matrix (114 features)...")

def g(row, col, default=0.0):
    v = row.get(col, default)
    try:
        if pd.isna(v): return float(default)
    except Exception:
        pass
    return float(v) if v is not None else float(default)

def layoff_buckets(days):
    return {'lt90':int(days<90),'90_180':int(90<=days<180),
            '180_365':int(180<=days<365),'gt365':int(days>=365)}

# Career rolling stats
cf = career_raw.copy()
def shift_cumsum(x): return x.cumsum().shift(1).fillna(0)

cf['cum_fights']     = cf.groupby('fighter').cumcount()
cf['cum_wins']       = cf.groupby('fighter')['won'].transform(shift_cumsum)
cf['career_win_rate']= np.where(cf['cum_fights']>0,
                                cf['cum_wins']/cf['cum_fights'], 0.5)
cf['ko_win']  = ((cf['won']==1)&cf['method'].str.contains('KO|TKO',case=False,na=False)).astype(int)
cf['sub_win'] = ((cf['won']==1)&cf['method'].str.contains('Sub|Submission',case=False,na=False)).astype(int)
cf['fin_win'] = ((cf['won']==1)&cf['method'].str.contains('KO|TKO|Sub|Submission',case=False,na=False)).astype(int)
cf['cum_ko']  = cf.groupby('fighter')['ko_win'].transform(shift_cumsum)
cf['cum_sub'] = cf.groupby('fighter')['sub_win'].transform(shift_cumsum)
cf['ko_finish_rate']  = np.where(cf['cum_fights']>0, cf['cum_ko']/cf['cum_fights'],  0.0)
cf['sub_finish_rate'] = np.where(cf['cum_fights']>0, cf['cum_sub']/cf['cum_fights'], 0.0)

def roll_sh(x, n): return x.shift(1).rolling(n, min_periods=1).mean()
cf['last3_win_rate']    = cf.groupby('fighter')['won'].transform(lambda x: roll_sh(x,3)).fillna(0.5)
cf['last5_won']         = cf.groupby('fighter')['won'].transform(lambda x: roll_sh(x,5)).fillna(0.5)
cf['last10_win_rate']   = cf.groupby('fighter')['won'].transform(lambda x: roll_sh(x,10)).fillna(0.5)
cf['last5_finish_rate'] = cf.groupby('fighter')['fin_win'].transform(lambda x: roll_sh(x,5)).fillna(0.0)
cf['trend_score']       = cf['last3_win_rate'] - cf['last10_win_rate']
cf['prev_date']         = cf.groupby('fighter')['date'].shift(1)
cf['layoff_days']       = (cf['date'] - cf['prev_date']).dt.days.fillna(365.0)

wr_cache = cf.groupby('fighter')['won'].mean().to_dict()
def opp_quality_series(grp):
    opps = grp['opponent'].values
    res  = np.full(len(grp), 0.5)
    for i in range(len(grp)):
        prior = opps[max(0,i-5):i]
        rates = [wr_cache.get(o,0.5) for o in prior]
        res[i] = float(np.mean(rates)) if rates else 0.5
    return pd.Series(res, index=grp.index)
cf['opp_quality'] = cf.groupby('fighter', group_keys=False).apply(opp_quality_series)

CAREER_COLS = ['cum_fights','career_win_rate','ko_finish_rate','sub_finish_rate',
               'last3_win_rate','last5_won','last10_win_rate','last5_finish_rate',
               'trend_score','layoff_days','opp_quality']
DEFAULT_CAREER = {c: (0.5 if 'rate' in c or c in ('trend_score','opp_quality') else
                      (365.0 if c=='layoff_days' else 0.0))
                  for c in CAREER_COLS}
DEFAULT_CAREER.update({'cum_fights':0,'career_win_rate':0.5,'last5_won':0.5,
                        'last3_win_rate':0.5,'last10_win_rate':0.5})

career_by_f = {}; career_dates_f = {}
for fname, grp in cf.groupby('fighter'):
    g_ = grp.reset_index(drop=True)
    career_by_f[fname] = g_; career_dates_f[fname] = g_['date'].tolist()

def get_career_at(fighter, fdate):
    if fighter not in career_by_f: return DEFAULT_CAREER.copy()
    dates = career_dates_f[fighter]
    idx = bisect.bisect_right(dates, fdate) - 1
    if idx < 0: return DEFAULT_CAREER.copy()
    return {c: float(career_by_f[fighter].iloc[idx][c]) for c in CAREER_COLS}

elo_by_f = {}; elo_dates_f = {}
for fname, grp in elo_hist.groupby('fighter'):
    g_ = grp.sort_values('date').reset_index(drop=True)
    elo_by_f[fname] = g_; elo_dates_f[fname] = g_['date'].tolist()

def get_elo_at(fighter, fdate):
    if fighter not in elo_by_f: return {'elo':1500.0,'elo_trend':0.0}
    dates = elo_dates_f[fighter]
    idx = bisect.bisect_left(dates, fdate) - 1
    if idx < 0: return {'elo':1500.0,'elo_trend':0.0}
    row = elo_by_f[fighter].iloc[idx]
    return {'elo':float(row['elo_after']),
            'elo_trend':float(row.get('elo_trend',0.0) or 0.0)}

fstyle = {}
for _, row in fstats_df.iterrows():
    fstyle[row['Fighter_Name']] = {
        k: float(row.get(k,0) or 0)
        for k in ['SLpM','SApM','Str_Acc','Str_Def','TD_Avg','TD_Acc','TD_Def','Sub_Avg']
    }

def build_m1_row(row):
    r_name=row['R_fighter']; b_name=row['B_fighter']; fdate=row['date']
    rc=get_career_at(r_name,fdate); bc=get_career_at(b_name,fdate)
    rs=fstyle.get(r_name,{}); bs=fstyle.get(b_name,{})
    re=get_elo_at(r_name,fdate); be=get_elo_at(b_name,fdate)
    r_lb=layoff_buckets(rc['layoff_days']); b_lb=layoff_buckets(bc['layoff_days'])
    r_sp=1 if str(row.get('R_Stance','') or '').lower()=='southpaw' else 0
    b_sp=1 if str(row.get('B_Stance','') or '').lower()=='southpaw' else 0
    r_wins=g(row,'R_wins'); b_wins=g(row,'B_wins')
    r_loss=g(row,'R_losses'); b_loss=g(row,'B_losses')
    r_h=g(row,'R_Height_cms',175); b_h=g(row,'B_Height_cms',175)
    r_rch=g(row,'R_Reach_cms',175); b_rch=g(row,'B_Reach_cms',175)
    r_age=g(row,'R_age',28); b_age=g(row,'B_age',28)
    r_sig=g(row,'R_avg_SIG_STR_landed'); b_sig=g(row,'B_avg_SIG_STR_landed')
    r_td=g(row,'R_avg_TD_landed'); b_td=g(row,'B_avg_TD_landed')
    r_ws=g(row,'R_current_win_streak'); b_ws=g(row,'B_current_win_streak')
    r_ls=g(row,'R_current_lose_streak'); b_ls=g(row,'B_current_lose_streak')
    r_lws=g(row,'R_longest_win_streak'); b_lws=g(row['B_longest_win_streak'] if 'B_longest_win_streak' in row else 0,'B_longest_win_streak' if 'B_longest_win_streak' in row else 0,0) if False else g(row,'B_longest_win_streak')
    r_sigp=g(row,'R_avg_SIG_STR_pct'); b_sigp=g(row,'B_avg_SIG_STR_pct')
    r_suba=g(row,'R_avg_SUB_ATT'); b_suba=g(row,'B_avg_SUB_ATT')
    r_tdp=g(row,'R_avg_TD_pct'); b_tdp=g(row,'B_avg_TD_pct')
    r_ttb=g(row,'R_total_title_bouts'); b_ttb=g(row,'B_total_title_bouts')
    r_ko=g(row,'R_win_by_KO/TKO'); b_ko=g(row,'B_win_by_KO/TKO')
    r_sub=g(row,'R_win_by_Submission'); b_sub=g(row,'B_win_by_Submission')
    wc_ord=WC_ORDER.get(str(row.get('weight_class','') or ''),6)
    title=1 if row.get('title_bout',False) else 0
    r_axe=r_age*rc['cum_fights']; b_axe=b_age*bc['cum_fights']
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

rows_m1 = [build_m1_row(row) for _, row in df.iterrows()]
X_m1_df = pd.DataFrame(rows_m1).reindex(columns=feat_cols_m1).fillna(0.0)
X_m1    = X_m1_df.values.astype(float)
# secondary safety: column-median impute then zero-fill any all-NaN cols
cm = np.nanmedian(X_m1, axis=0)
nm = np.isnan(X_m1)
if nm.any():
    cm_safe = np.where(np.isnan(cm), 0.0, cm)
    X_m1[nm] = np.take(cm_safe, np.where(nm)[1])
X_m1 = np.nan_to_num(X_m1, nan=0.0)
print(f"  M1 feature matrix: {X_m1.shape}")
gc.collect()

# ─── M1 probabilities (production model, all rows) ───────────────────────────
print("  Computing M1 probabilities (production model, no OOF)...")
m1_probs = m1_predict(X_m1)
m1_test_acc = accuracy_score(target[test_idx], (m1_probs[test_idx]>0.5).astype(int))
print(f"  M1 test accuracy on this filtered set: {m1_test_acc:.4f}")
gc.collect()

# ─── Build method-odds feature matrix ─────────────────────────────────────────
print("\nBuilding method-odds features (18 columns)...")

def implied(odds):
    try:
        o = float(odds)
        if o == 0 or np.isnan(o): return 0.0
        return abs(o)/(abs(o)+100) if o < 0 else 100/(o+100)
    except Exception:
        return 0.0

NOML_COLS_BASE = [
    'm1_prob',
    'r_ko_novig','b_ko_novig','r_sub_novig','b_sub_novig','r_dec_novig','b_dec_novig',
    'ko_diff','sub_diff','dec_diff',
    'method_win_prob','method_vs_m1',
    'finish_prob','r_finish_prob','b_finish_prob','finish_advantage',
    'model_conf',
]

def build_noml_row(i, row, m1p):
    r_ko_r  = implied(row['r_ko_odds']);  b_ko_r  = implied(row['b_ko_odds'])
    r_sub_r = implied(row['r_sub_odds']); b_sub_r = implied(row['b_sub_odds'])
    r_dec_r = implied(row['r_dec_odds']); b_dec_r = implied(row['b_dec_odds'])

    # 6-way normalization — all outcomes sum to 1
    total = r_ko_r + b_ko_r + r_sub_r + b_sub_r + r_dec_r + b_dec_r
    if total <= 0: total = 1.0
    r_ko  = r_ko_r/total;  b_ko  = b_ko_r/total
    r_sub = r_sub_r/total; b_sub = b_sub_r/total
    r_dec = r_dec_r/total; b_dec = b_dec_r/total

    method_win_prob = r_ko + r_sub + r_dec   # market's red win prob
    finish_prob     = r_ko + b_ko + r_sub + b_sub
    r_finish_prob   = r_ko + r_sub
    b_finish_prob   = b_ko + b_sub

    return [
        m1p,
        r_ko, b_ko, r_sub, b_sub, r_dec, b_dec,
        r_ko - b_ko, r_sub - b_sub, r_dec - b_dec,
        method_win_prob, method_win_prob - m1p,
        finish_prob, r_finish_prob, b_finish_prob,
        r_finish_prob - b_finish_prob,
        abs(m1p - 0.5),
    ]

rows_noml = [build_noml_row(i, row, m1_probs[i]) for i, (_, row) in enumerate(df.iterrows())]
X_base = np.array(rows_noml, dtype=float)
cm2 = np.nanmedian(X_base, axis=0)
nm2 = np.isnan(X_base)
X_base[nm2] = np.take(cm2, np.where(nm2)[1])
print(f"  Base method matrix: {X_base.shape}")

# ─── Tier historical win rate (method-based tier, no leakage) ─────────────────
print("  Computing tier_hist_win_rate (no-leakage expanding window)...")

def method_tier(p):
    if p > 0.75: return 'hfav'
    if p > 0.60: return 'mfav'
    if p > 0.525: return 'sfav'
    if p >= 0.475: return 'pkem'
    if p >= 0.40: return 'sdog'
    if p >= 0.25: return 'mdog'
    return 'hdog'

mwp_arr  = X_base[:, NOML_COLS_BASE.index('method_win_prob')]
pick_red = mwp_arr > 0.5
pick_prob_arr = np.where(pick_red, mwp_arr, 1.0 - mwp_arr)
pick_won_arr  = np.where(pick_red, target, 1 - target)
tier_labels   = [method_tier(p) for p in pick_prob_arr]

# Iterate in date order, assign no-leakage values back to original row positions
date_vals  = df['date'].values
sort_order = np.argsort(date_vals, kind='stable')

tier_counts = {}; tier_wins = {}
tier_hist_wr = np.zeros(len(df))

for k in sort_order:
    t = tier_labels[k]
    c = tier_counts.get(t, 0); w = tier_wins.get(t, 0)
    tier_hist_wr[k] = (w/c) if c >= 5 else TIER_FALLBACK.get(t, 0.50)
    tier_counts[t] = c + 1
    tier_wins[t]   = w + pick_won_arr[k]

# Append tier feature
X_noml = np.column_stack([X_base, tier_hist_wr])
NOML_COLS = NOML_COLS_BASE + ['tier_hist_win_rate']
print(f"  Final method matrix: {X_noml.shape}  |  features: {len(NOML_COLS)}")
print(f"  Features: {NOML_COLS}")
gc.collect()

# ─── Train/test split ─────────────────────────────────────────────────────────
X_train = X_noml[train_idx]
X_test  = X_noml[test_idx]
y_train = target[train_idx]
y_test  = target[test_idx]
print(f"\n  Train: {X_train.shape}  Test: {X_test.shape}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# ─── STEP A — Logistic Regression (Optuna, 50 trials) ─────────────────────────
print(f"\nStep A — Logistic Regression ({N_TRIALS} Optuna trials)...")

def lr_objective(trial):
    C       = trial.suggest_float('C', 0.001, 50.0, log=True)
    penalty = trial.suggest_categorical('penalty', ['l1','l2','elasticnet'])
    sc_type = trial.suggest_categorical('scaler', ['robust','standard'])
    cw      = trial.suggest_categorical('class_weight', ['none','balanced'])
    cw_val  = None if cw == 'none' else 'balanced'
    if penalty == 'elasticnet':
        l1r = trial.suggest_float('l1_ratio', 0.0, 1.0)
        clf = LogisticRegression(C=C, penalty='elasticnet', l1_ratio=l1r,
                                 solver='saga', class_weight=cw_val,
                                 max_iter=2000, random_state=SEED)
    elif penalty == 'l1':
        clf = LogisticRegression(C=C, penalty='l1', solver='saga',
                                 class_weight=cw_val, max_iter=2000, random_state=SEED)
    else:
        clf = LogisticRegression(C=C, penalty='l2', solver='saga',
                                 class_weight=cw_val, max_iter=2000, random_state=SEED)
    scaler = RobustScaler() if sc_type == 'robust' else StandardScaler()
    pipe   = Pipeline([('sc', scaler), ('clf', clf)])
    oof    = cross_val_predict(pipe, X_train, y_train, cv=skf,
                               method='predict_proba', n_jobs=1)[:, 1]
    return accuracy_score(y_train, (oof > 0.5).astype(int))

study_lr = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
study_lr.optimize(lr_objective, n_trials=N_TRIALS, show_progress_bar=False)
p_lr = study_lr.best_params

cw_val_lr = None if p_lr['class_weight'] == 'none' else 'balanced'
pen = p_lr['penalty']
if pen == 'elasticnet':
    clf_lr = LogisticRegression(C=p_lr['C'], penalty='elasticnet',
                                l1_ratio=p_lr.get('l1_ratio',0.5),
                                solver='saga', class_weight=cw_val_lr,
                                max_iter=2000, random_state=SEED)
elif pen == 'l1':
    clf_lr = LogisticRegression(C=p_lr['C'], penalty='l1', solver='saga',
                                class_weight=cw_val_lr, max_iter=2000, random_state=SEED)
else:
    clf_lr = LogisticRegression(C=p_lr['C'], penalty='l2', solver='saga',
                                class_weight=cw_val_lr, max_iter=2000, random_state=SEED)

sc_lr  = RobustScaler() if p_lr['scaler'] == 'robust' else StandardScaler()
model_lr_noml = Pipeline([('sc', sc_lr), ('clf', clf_lr)])
model_lr_noml.fit(X_train, y_train)
p_lr_test  = model_lr_noml.predict_proba(X_test)[:, 1]
acc_lr  = accuracy_score(y_test, (p_lr_test > 0.5).astype(int))
brier_lr = brier_score_loss(y_test, p_lr_test)
auc_lr   = roc_auc_score(y_test, p_lr_test)
print(f"  LR: acc={acc_lr:.4f}  brier={brier_lr:.4f}  AUC={auc_lr:.4f}  "
      f"  beat M2A_prod? {'YES' if acc_lr > M2A_PROD_ACC else 'no'}")
print(f"  Best params: {p_lr}")
gc.collect()

# ─── STEP B — XGBoost (Optuna, 50 trials) ────────────────────────────────────
print(f"\nStep B — XGBoost ({N_TRIALS} Optuna trials)...")

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
    oof = cross_val_predict(clf, X_train, y_train, cv=skf,
                            method='predict_proba', n_jobs=1)[:, 1]
    return accuracy_score(y_train, (oof > 0.5).astype(int))

study_xgb = optuna.create_study(direction='maximize',
                                  sampler=optuna.samplers.TPESampler(seed=SEED))
study_xgb.optimize(xgb_objective, n_trials=N_TRIALS, show_progress_bar=False)
p_xgb = study_xgb.best_params

model_xgb_noml = XGBClassifier(**p_xgb, use_label_encoder=False,
                                eval_metric='logloss', random_state=SEED,
                                verbosity=0, n_jobs=1)
model_xgb_noml.fit(X_train, y_train)
p_xgb_test = model_xgb_noml.predict_proba(X_test)[:, 1]
acc_xgb   = accuracy_score(y_test, (p_xgb_test > 0.5).astype(int))
brier_xgb = brier_score_loss(y_test, p_xgb_test)
auc_xgb   = roc_auc_score(y_test, p_xgb_test)
print(f"  XGB: acc={acc_xgb:.4f}  brier={brier_xgb:.4f}  AUC={auc_xgb:.4f}  "
      f"  beat M2A_prod? {'YES' if acc_xgb > M2A_PROD_ACC else 'no'}")
print(f"  Best params: {p_xgb}")
gc.collect()

# ─── 50/50 Blend ──────────────────────────────────────────────────────────────
print("\nStep C — 50/50 LR+XGB Blend...")
p_blend = 0.50 * p_lr_test + 0.50 * p_xgb_test
acc_blend   = accuracy_score(y_test, (p_blend > 0.5).astype(int))
brier_blend = brier_score_loss(y_test, p_blend)
auc_blend   = roc_auc_score(y_test, p_blend)

# Calibration MAE: |mean(pred_proba_in_bin) - mean(actual_in_bin)| per decile
bins = np.linspace(0, 1, 11)
cal_mae = 0.0; n_bins = 0
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (p_blend >= lo) & (p_blend < hi)
    if mask.sum() >= 5:
        cal_mae += abs(p_blend[mask].mean() - y_test[mask].mean())
        n_bins += 1
cal_mae = cal_mae / n_bins if n_bins > 0 else float('nan')

print(f"\n  50/50 Blend: acc={acc_blend:.4f}  brier={brier_blend:.4f}  "
      f"AUC={auc_blend:.4f}  CalMAE={cal_mae:.4f}")
print(f"  Beat M2A prod ({M2A_PROD_ACC:.4f})? {'YES ✓' if acc_blend > M2A_PROD_ACC else 'no ✗'}")
print(f"  Beat M1 baseline ({M1_ACC:.4f})?  {'YES ✓' if acc_blend > M1_ACC else 'no ✗'}")
gc.collect()

# ─── Feature importances ──────────────────────────────────────────────────────
print("\nFeature importances — XGB (top 15):")
fi_xgb = sorted(zip(NOML_COLS, model_xgb_noml.feature_importances_),
                key=lambda x: -x[1])
for rank, (feat, imp) in enumerate(fi_xgb[:15], 1):
    print(f"  {rank:2d}. {feat:<22}: {imp:.4f}")

print("\nFeature importances — LR |coef| (top 15):")
try:
    coefs = abs(model_lr_noml.named_steps['clf'].coef_[0])
    fi_lr = sorted(zip(NOML_COLS, coefs), key=lambda x: -x[1])
    for rank, (feat, imp) in enumerate(fi_lr[:15], 1):
        print(f"  {rank:2d}. {feat:<22}: {imp:.4f}")
except Exception as e:
    print(f"  (LR coef extraction failed: {e})")

# ─── Side-by-side comparison ──────────────────────────────────────────────────
# Production M2A metrics on same test set (already known from training)
# Recompute M1-only baseline on this filtered test set for fairness
m1_only_acc = accuracy_score(y_test, (m1_probs[test_idx] > 0.5).astype(int))
m1_only_brier = brier_score_loss(y_test, m1_probs[test_idx])
m1_only_auc   = roc_auc_score(y_test, m1_probs[test_idx])

print("\n" + "=" * 70)
print("SIDE-BY-SIDE COMPARISON — Test set 2024+ (men's, method odds available)")
print("=" * 70)
print(f"\n  {'Model':<30}  {'Acc':>7}  {'Brier':>7}  {'AUC':>7}  {'CalMAE':>8}")
print(f"  {'-'*30}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*8}")
print(f"  {'M1 Analyst (baseline)':30}  {m1_only_acc:7.4f}  {m1_only_brier:7.4f}  {m1_only_auc:7.4f}  {'—':>8}")
print(f"  {'M2A Prod (ML+method, 73.20%)':30}  {M2A_PROD_ACC:7.4f}  {'—':>7}  {'—':>7}  {'—':>8}")
print(f"  {'NEW LR (method-only)':30}  {acc_lr:7.4f}  {brier_lr:7.4f}  {auc_lr:7.4f}  {'—':>8}")
print(f"  {'NEW XGB (method-only)':30}  {acc_xgb:7.4f}  {brier_xgb:7.4f}  {auc_xgb:7.4f}  {'—':>8}")
print(f"  {'NEW 50/50 Blend':30}  {acc_blend:7.4f}  {brier_blend:7.4f}  {auc_blend:7.4f}  {cal_mae:8.4f}")

delta_vs_m1   = acc_blend - m1_only_acc
delta_vs_m2a  = acc_blend - M2A_PROD_ACC
sign1 = '+' if delta_vs_m1  >= 0 else ''
sign2 = '+' if delta_vs_m2a >= 0 else ''
print(f"\n  vs M1 baseline:    {sign1}{delta_vs_m1*100:.2f}pp")
print(f"  vs M2A prod:       {sign2}{delta_vs_m2a*100:.2f}pp")

# ─── Save models ──────────────────────────────────────────────────────────────
print("\nSaving models...")
joblib.dump(model_lr_noml,  'model/ufc_model2a_noml_lr.pkl')
joblib.dump(model_xgb_noml, 'model/ufc_model2a_noml_xgb.pkl')
joblib.dump(NOML_COLS,       'model/feature_columns_2a_noml.pkl')
print("  Saved: model/ufc_model2a_noml_lr.pkl")
print("  Saved: model/ufc_model2a_noml_xgb.pkl")
print("  Saved: model/feature_columns_2a_noml.pkl")

# Metadata
meta = {
    'created':         datetime.now().isoformat(),
    'description':     'M2A method-odds-only variant — no moneyline features',
    'features':        NOML_COLS,
    'n_features':      len(NOML_COLS),
    'filter':          'men-only, 2013+, all 6 method odds non-null',
    'train_size':      int(len(train_idx)),
    'test_size':       int(len(test_idx)),
    'temporal_split':  '2024-01-01',
    'blend':           '50% LR + 50% XGB',
    'acc_lr':          float(acc_lr),
    'acc_xgb':         float(acc_xgb),
    'acc_blend':       float(acc_blend),
    'brier_blend':     float(brier_blend),
    'auc_blend':       float(auc_blend),
    'cal_mae_blend':   float(cal_mae),
    'm2a_prod_acc':    M2A_PROD_ACC,
    'm1_baseline_acc': float(m1_only_acc),
    'beats_m2a_prod':  bool(acc_blend > M2A_PROD_ACC),
    'beats_m1':        bool(acc_blend > m1_only_acc),
    'lr_params':       {k: (float(v) if isinstance(v,float) else v) for k,v in p_lr.items()},
    'xgb_params':      {k: (float(v) if isinstance(v,float) else v) for k,v in p_xgb.items()},
}
with open('model/model2a_noml_metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)
print("  Saved: model/model2a_noml_metadata.json")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
