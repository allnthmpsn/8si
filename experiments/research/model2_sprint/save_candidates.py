#!/usr/bin/env python3
"""
Save Model 2 candidate pkl files using Optuna params from results.json.
Rebuilds training data, fits LR + XGB on full training set, saves to model2_sprint/.
Run from project root: python experiments/research/model2_sprint/save_candidates.py
"""

import bisect, gc, json, math, warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

warnings.filterwarnings('ignore')
np.random.seed(42)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

SPRINT_DIR   = Path('experiments/research/model2_sprint')
TRAIN_CUTOFF = pd.Timestamp('2024-01-01')
LR_WEIGHT    = 0.70
XGB_WEIGHT   = 0.30
SEED         = 42

with open(SPRINT_DIR / 'results.json') as f:
    sprint_results = json.load(f)

s5          = sprint_results['step5_full_retrain']
LR_PARAMS   = s5['lr_params']
XGB_PARAMS  = s5['xgb_params']

WC_ORDER = {
    "Women's Strawweight": 0, "Women's Flyweight": 1, "Women's Bantamweight": 2,
    "Women's Featherweight": 3, "Flyweight": 4, "Bantamweight": 5,
    "Featherweight": 6, "Lightweight": 7, "Welterweight": 8,
    "Middleweight": 9, "Light Heavyweight": 10, "Heavyweight": 11,
    "Catch Weight": 6,
}

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

print("=" * 60)
print("MODEL 2 CANDIDATE SAVE")
print("=" * 60)

# ── Load ─────────────────────────────────────────────────────────────────────
print("\nLoading models and data...")

model_lr_m1  = joblib.load('model/ufc_model_best.pkl')
model_xgb_m1 = joblib.load('model/ufc_model_xgb.pkl')
feat_cols     = joblib.load('model/feature_columns_best.pkl')

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

# ── Filter + corner randomization ────────────────────────────────────────────
df = df_master[
    (df_master['date'] >= '2018-01-01') &
    df_master['R_odds'].notna() & df_master['B_odds'].notna() &
    df_master['r_dec_odds'].notna() & df_master['b_dec_odds'].notna() &
    df_master['r_sub_odds'].notna() & df_master['b_sub_odds'].notna() &
    df_master['r_ko_odds'].notna()  & df_master['b_ko_odds'].notna() &
    df_master['Winner'].isin(['Red', 'Blue'])
].copy().reset_index(drop=True)

np.random.seed(SEED)
swap_mask = np.random.random(len(df)) < 0.5

r_matched = sorted([c for c in df.columns if c.startswith('R_') and ('B_' + c[2:]) in df.columns])
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

target     = (df['Winner'] == 'Red').astype(int).values
train_mask = (df['date'] < TRAIN_CUTOFF).values
test_mask  = ~train_mask
train_idx  = np.where(train_mask)[0]
test_idx   = np.where(test_mask)[0]
y_train    = target[train_idx]
y_test     = target[test_idx]

print(f"  M2 universe: {len(df)} | Train: {len(train_idx)} | Test: {len(test_idx)}")

# ── Career stats ──────────────────────────────────────────────────────────────
print("\nBuilding career stats...")

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

# ── 109-feature matrix ────────────────────────────────────────────────────────
print("\nBuilding 109-feature matrix...")

def build_features(df_row):
    r_name = df_row['R_fighter']; b_name = df_row['B_fighter']; fdate = df_row['date']
    rc = get_career_at(r_name, fdate); bc = get_career_at(b_name, fdate)
    rs = fstyle.get(r_name, {}); bs = fstyle.get(b_name, {})
    re = get_elo_at(r_name, fdate); be = get_elo_at(b_name, fdate)
    r_lb = layoff_buckets(rc['layoff_days']); b_lb = layoff_buckets(bc['layoff_days'])
    r_sp = 1 if str(df_row.get('R_Stance','') or '').lower() == 'southpaw' else 0
    b_sp = 1 if str(df_row.get('B_Stance','') or '').lower() == 'southpaw' else 0
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
    wc_ord = WC_ORDER.get(str(df_row.get('weight_class','') or ''), 6)
    r_axe  = r_age * rc['cum_fights']; b_axe = b_age * bc['cum_fights']
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
        'ko_dif':r_ko-b_ko,'sub_dif':r_sub-b_sub,'total_title_bout_dif':r_ttb-b_ttb,
        'weight_class_ord':wc_ord,'orth_clash':1 if (r_sp==0 and b_sp==0) else 0,
        'south_clash':1 if (r_sp==1 and b_sp==1) else 0,'R_southpaw':r_sp,
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
X_df  = pd.DataFrame(rows_list, columns=feat_cols)
X_m1  = X_df[feat_cols].values.astype(float)
cm    = np.nanmedian(X_m1, axis=0)
nm    = np.isnan(X_m1)
X_m1[nm] = np.take(cm, np.where(nm)[1])
print(f"  M1 matrix: {X_m1.shape}")
gc.collect()

# ── M1 OOF predictions ────────────────────────────────────────────────────────
print("\nGenerating M1 OOF predictions...")
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

X_m1_train = X_m1[train_idx]; X_m1_test = X_m1[test_idx]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
m1_oof = np.zeros(len(train_idx))

for fold_i, (tr_i, va_i) in enumerate(skf.split(X_m1_train, y_train)):
    Xtr, Xva = X_m1_train[tr_i], X_m1_train[va_i]
    ytr = y_train[tr_i]
    fold_lr  = Pipeline([('sc', RobustScaler()),
                         ('clf', LogisticRegression(C=0.00711, penalty='l2', max_iter=2000,
                                                    solver='saga', random_state=SEED))])
    fold_lr.fit(Xtr, ytr)
    fold_xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3,
                              subsample=0.8, colsample_bytree=0.8,
                              use_label_encoder=False, eval_metric='logloss',
                              random_state=SEED, verbosity=0, n_jobs=1)
    fold_xgb.fit(Xtr, ytr)
    m1_oof[va_i] = LR_WEIGHT * fold_lr.predict_proba(Xva)[:,1] + \
                   XGB_WEIGHT * fold_xgb.predict_proba(Xva)[:,1]

m1_test = LR_WEIGHT * model_lr_m1.predict_proba(X_m1_test)[:,1] + \
          XGB_WEIGHT * model_xgb_m1.predict_proba(X_m1_test)[:,1]

print(f"  M1 OOF train acc : {accuracy_score(y_train, (m1_oof>0.5).astype(int)):.4f}")
print(f"  M1 test acc      : {accuracy_score(y_test,  (m1_test>0.5).astype(int)):.4f}")
gc.collect()

# ── Base 23 M2 features ───────────────────────────────────────────────────────
print("\nBuilding 42-feature M2 dataset...")

BASE_M2_FEATURES = [
    'model1_prob','f1_ml_novig','f2_ml_novig','ml_gap','vig',
    'f1_dec_implied','f2_dec_implied','dec_implied_dif',
    'f1_ko_implied','f2_ko_implied','ko_implied_dif',
    'f1_sub_implied','f2_sub_implied','sub_implied_dif',
    'finish_prob','f1_finish_prob','f2_finish_prob','finish_advantage',
    'abs_gap','vegas_confidence','model_confidence','model_agrees_vegas','gap_x_confidence',
]

train_pos = {v: k for k, v in enumerate(train_idx)}
test_pos  = {v: k for k, v in enumerate(test_idx)}

m2_rows = []
for i, (_, df_row) in enumerate(df.iterrows()):
    m1p  = float(m1_oof[train_pos[i]]) if i in train_pos else float(m1_test[test_pos[i]])
    f1_n, f2_n, vig_ = novig_probs(df_row['R_odds'], df_row['B_odds'])
    ml_gap_ = m1p - f1_n
    f1_dec = implied_prob(df_row['r_dec_odds']) or 0.0
    f2_dec = implied_prob(df_row['b_dec_odds']) or 0.0
    f1_ko  = implied_prob(df_row['r_ko_odds'])  or 0.0
    f2_ko  = implied_prob(df_row['b_ko_odds'])  or 0.0
    f1_sub = implied_prob(df_row['r_sub_odds']) or 0.0
    f2_sub = implied_prob(df_row['b_sub_odds']) or 0.0
    dec_total = f1_dec + f2_dec
    fin_prob  = 1.0 - (dec_total / 2.0) if dec_total > 0 else 0.5
    f1_fin = f1_ko + f1_sub; f2_fin = f2_ko + f2_sub
    m2_rows.append([
        m1p, f1_n, f2_n, ml_gap_, vig_,
        f1_dec, f2_dec, f1_dec-f2_dec,
        f1_ko, f2_ko, f1_ko-f2_ko,
        f1_sub, f2_sub, f1_sub-f2_sub,
        fin_prob, f1_fin, f2_fin, f1_fin-f2_fin,
        abs(ml_gap_), abs(f1_n-0.5), abs(m1p-0.5),
        1 if (m1p>0.5)==(f1_n>0.5) else 0,
        ml_gap_ * abs(f1_n-0.5),
    ])

X2_base = np.array(m2_rows, dtype=float)
cm2 = np.nanmedian(X2_base, axis=0)
nm2 = np.isnan(X2_base); X2_base[nm2] = np.take(cm2, np.where(nm2)[1])

# ── Step 1 features ───────────────────────────────────────────────────────────
df_fh = df[['date','R_fighter','B_fighter','R_odds','B_odds','Winner']].copy()
df_fh['f1_won']    = (df_fh['Winner'] == 'Red').astype(int)
df_fh['f1_is_fav'] = (df_fh['R_odds'] < 0).astype(int)
df_sorted = df_fh.sort_values('date').reset_index()

fav_bouts = {}; fav_wins = {}; dog_bouts = {}; dog_wins = {}

step1_rows = [None] * len(df)
for _, row in df_sorted.iterrows():
    orig_i = row['index']
    f1     = row['R_fighter']
    is_fav = row['f1_is_fav']
    f1_won = row['f1_won']
    f1_nv, _, _ = novig_probs(df.loc[orig_i, 'R_odds'], df.loc[orig_i, 'B_odds'])

    fav_n = len(fav_bouts.get(f1, []))
    fav_w = fav_wins.get(f1, 0)
    dog_n = len(dog_bouts.get(f1, []))
    dog_w = dog_wins.get(f1, 0)

    if f1_nv < 0.30:   tier = 0
    elif f1_nv < 0.45: tier = 1
    elif f1_nv < 0.55: tier = 2
    elif f1_nv < 0.70: tier = 3
    else:              tier = 4

    step1_rows[orig_i] = [
        is_fav,
        fav_w/fav_n if fav_n>0 else 0.5,
        dog_w/dog_n if dog_n>0 else 0.5,
        math.log1p(fav_n),
        math.log1p(dog_n),
        abs(f1_nv - 0.5),
        tier,
    ]

    if is_fav:
        fav_bouts.setdefault(f1, []).append(row['date'])
        fav_wins[f1] = fav_wins.get(f1, 0) + f1_won
    else:
        dog_bouts.setdefault(f1, []).append(row['date'])
        dog_wins[f1] = dog_wins.get(f1, 0) + f1_won

step1_arr = np.array(step1_rows, dtype=float)

# Compute tier_hist_win_rate from training data
tier_col    = step1_arr[train_idx, 6]
train_wins  = target[train_idx]
tier_wr     = {}
for t in range(5):
    mask_t = (tier_col == t)
    tier_wr[t] = float(train_wins[mask_t].mean()) if mask_t.sum() > 0 else 0.5

tier_hist_wr = np.array([tier_wr.get(int(t), 0.5) for t in step1_arr[:, 6]])
# Replace tier col with computed win rate
step1_final = np.column_stack([step1_arr[:, :6], tier_hist_wr])  # 7 cols: drop raw tier, add win rate

STEP1_COLS = ['f1_is_fav','f1_hist_fav_wr','f1_hist_dog_wr','f1_fav_bouts_log',
              'f1_dog_bouts_log','odds_strength','tier_hist_win_rate']

# ── Step 2 features ───────────────────────────────────────────────────────────
style_feat_names = ['R_ko_finish_rate','B_ko_finish_rate','R_sub_finish_rate',
                    'B_sub_finish_rate','Str_Def_dif']
style_idx = {c: feat_cols.index(c) for c in style_feat_names if c in feat_cols}

step2_rows = []
for i in range(len(df)):
    br = X2_base[i]
    f1_ko_i  = br[BASE_M2_FEATURES.index('f1_ko_implied')]
    f2_ko_i  = br[BASE_M2_FEATURES.index('f2_ko_implied')]
    f1_sub_i = br[BASE_M2_FEATURES.index('f1_sub_implied')]
    f2_sub_i = br[BASE_M2_FEATURES.index('f2_sub_implied')]
    f1_dec_i = br[BASE_M2_FEATURES.index('f1_dec_implied')]
    f2_dec_i = br[BASE_M2_FEATURES.index('f2_dec_implied')]
    m1p_     = br[BASE_M2_FEATURES.index('model1_prob')]
    fin_p_   = br[BASE_M2_FEATURES.index('finish_prob')]
    r_ko_fr  = X_m1[i, feat_cols.index('R_ko_finish_rate')]
    b_ko_fr  = X_m1[i, feat_cols.index('B_ko_finish_rate')]
    r_sub_fr = X_m1[i, feat_cols.index('R_sub_finish_rate')]
    b_sub_fr = X_m1[i, feat_cols.index('B_sub_finish_rate')]
    str_def  = X_m1[i, feat_cols.index('Str_Def_dif')]
    step2_rows.append([
        f1_ko_i*r_ko_fr - f2_ko_i*b_ko_fr,   # ko_style_edge
        f1_sub_i*r_sub_fr - f2_sub_i*b_sub_fr, # sub_style_edge
        fin_p_ * abs(m1p_ - 0.5),              # finish_x_model_conf
        ((f1_dec_i+f2_dec_i)/2.0)*abs(str_def),# dec_x_str_def
        f1_ko_i + f2_ko_i,                     # combined_ko_implied
        f1_sub_i + f2_sub_i,                   # combined_sub_implied
        abs(f1_ko_i - f2_ko_i),                # ko_method_gap
        abs(f1_sub_i - f2_sub_i),              # sub_method_gap
    ])

STEP2_COLS = ['ko_style_edge','sub_style_edge','finish_x_model_conf','dec_x_str_def',
              'combined_ko_implied','combined_sub_implied','ko_method_gap','sub_method_gap']
step2_arr = np.array(step2_rows, dtype=float)

# ── Step 3 features ───────────────────────────────────────────────────────────
wc_arr = X_m1[:, feat_cols.index('weight_class_ord')]
no_rds = df.get('no_of_rounds', pd.Series([3]*len(df))).fillna(3).values

# m1_wc_bias from training
m1_train_acc = accuracy_score(y_train, (m1_oof > 0.5).astype(int))
m1_wc_acc = {}
for wc_v in np.unique(wc_arr[train_idx]):
    mask_ = (wc_arr[train_idx] == wc_v)
    if mask_.sum() >= 10:
        m1_wc_acc[wc_v] = accuracy_score(y_train[mask_], (m1_oof[mask_]>0.5).astype(int))
    else:
        m1_wc_acc[wc_v] = m1_train_acc

step3_rows = []
for i in range(len(df)):
    wc_v_   = wc_arr[i]
    is_5r   = 1 if no_rds[i] >= 5 else 0
    wc_acc_ = m1_wc_acc.get(wc_v_, m1_train_acc)
    m1_conf = abs(X2_base[i, BASE_M2_FEATURES.index('model_confidence')])
    step3_rows.append([wc_v_/11.0, is_5r, wc_acc_-m1_train_acc, is_5r*m1_conf])

STEP3_COLS = ['wc_norm','is_5r','m1_wc_bias','five_r_x_conf']
step3_arr = np.array(step3_rows, dtype=float)

ALL_M2_FEATURE_NAMES = BASE_M2_FEATURES + STEP1_COLS + STEP2_COLS + STEP3_COLS
X2_full = np.hstack([X2_base, step1_final, step2_arr, step3_arr])
X2_train = X2_full[train_idx]
X2_test  = X2_full[test_idx]

print(f"  Full M2 matrix: {X2_full.shape} (expected 42 features)")

# ── Fit candidate models ──────────────────────────────────────────────────────
print("\nFitting LR candidate...")
p = LR_PARAMS
cw_val  = None if p['class_weight'] == 'none' else 'balanced'
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
sc_lr = StandardScaler()
candidate_lr = Pipeline([('sc', sc_lr), ('clf', clf_lr)])
candidate_lr.fit(X2_train, y_train)
lr_prob  = candidate_lr.predict_proba(X2_test)[:, 1]
lr_acc   = accuracy_score(y_test, (lr_prob > 0.5).astype(int))
print(f"  LR test acc: {lr_acc:.4f}")

print("Fitting XGB candidate...")
candidate_xgb = XGBClassifier(
    **XGB_PARAMS,
    use_label_encoder=False, eval_metric='logloss',
    random_state=SEED, verbosity=0, n_jobs=1
)
candidate_xgb.fit(X2_train, y_train)
xgb_prob = candidate_xgb.predict_proba(X2_test)[:, 1]
xgb_acc  = accuracy_score(y_test, (xgb_prob > 0.5).astype(int))
print(f"  XGB test acc: {xgb_acc:.4f}")

blend_prob = 0.50 * lr_prob + 0.50 * xgb_prob
blend_acc  = accuracy_score(y_test, (blend_prob > 0.5).astype(int))
print(f"  50/50 blend test acc: {blend_acc:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
print("\nSaving candidate files...")
joblib.dump(candidate_lr,          SPRINT_DIR / 'model2_candidate_lr.pkl')
joblib.dump(candidate_xgb,         SPRINT_DIR / 'model2_candidate_xgb.pkl')
joblib.dump(ALL_M2_FEATURE_NAMES,  SPRINT_DIR / 'model2_candidate_features.pkl')
# Save tier stats for backend inference (needed for tier_hist_win_rate at prediction time)
with open(SPRINT_DIR / 'model2_tier_stats.json', 'w') as f:
    json.dump({'tier_win_rates': {str(k): v for k,v in tier_wr.items()},
               'm1_train_acc': m1_train_acc,
               'm1_wc_acc': {str(k): v for k,v in m1_wc_acc.items()},
               'feature_names': ALL_M2_FEATURE_NAMES,
               'blend_lr': 0.50, 'blend_xgb': 0.50,
               'test_acc': round(blend_acc, 4),
               'n_features': len(ALL_M2_FEATURE_NAMES)}, f, indent=2)

print(f"\n  Saved: model2_candidate_lr.pkl")
print(f"  Saved: model2_candidate_xgb.pkl")
print(f"  Saved: model2_candidate_features.pkl")
print(f"  Saved: model2_tier_stats.json")
print(f"\nVerification — 50/50 blend test acc: {blend_acc:.4f}")
print("Done.")
