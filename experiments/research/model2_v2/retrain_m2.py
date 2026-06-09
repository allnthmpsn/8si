#!/usr/bin/env python3
"""
Model 2 retrain — align M2 training to new M1 V2 (129 features, men's only).

Changes vs original sprint:
  1. Men's-only filter (exclude women's weight classes)
  2. M1 feature matrix extended to 129 features (QA approx + interactions)
  3. Retrain 50/50 LR+XGB M2 on updated model1_prob values
  4. Threshold analysis 5-15% with/without M1+M2 agreement filter
  5. Promote if new acc > 73.20% (old production); else hold

Run from project root:
    python experiments/research/model2_v2/retrain_m2.py
"""

import bisect, gc, json, math, warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

warnings.filterwarnings('ignore')
np.random.seed(42)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

RESULTS_DIR  = Path('experiments/research/model2_v2')
TRAIN_CUTOFF = pd.Timestamp('2024-01-01')
SEED         = 42
OLD_M2_ACC   = 0.7320   # production threshold to beat
WOMENS_CLASSES = {
    "Women's Strawweight", "Women's Flyweight",
    "Women's Bantamweight", "Women's Featherweight",
}

WC_ORDER = {
    "Women's Strawweight": 0, "Women's Flyweight": 1,
    "Women's Bantamweight": 2, "Women's Featherweight": 3,
    "Flyweight": 4, "Bantamweight": 5, "Featherweight": 6,
    "Lightweight": 7, "Welterweight": 8, "Middleweight": 9,
    "Light Heavyweight": 10, "Heavyweight": 11, "Catch Weight": 6,
}

print("=" * 65)
print("MODEL 2 RETRAIN — aligned to M1 V2 (129 features, men's only)")
print("=" * 65)

# ── Load M1 production models ─────────────────────────────────────────────────
print("\n[SETUP] Loading M1 production models...")
model_lr_m1  = joblib.load('model/ufc_model_best.pkl')
model_xgb_m1 = joblib.load('model/ufc_model_xgb.pkl')
feat_cols_m1 = joblib.load('model/feature_columns_best.pkl')
print(f"  M1 features: {len(feat_cols_m1)}")

# ── Load data ─────────────────────────────────────────────────────────────────
print("\n[SETUP] Loading data...")
df_master  = pd.read_csv('data/ufc-master.csv', low_memory=False)
df_master['date'] = pd.to_datetime(df_master['date'])

career_raw = pd.read_csv('data/career_fights_updated.csv')
career_raw['date'] = pd.to_datetime(career_raw['date'])
career_raw = career_raw.sort_values(['fighter', 'date']).reset_index(drop=True)

fstats_df  = pd.read_csv('data/ufc_fighters_final_updated.csv')
for c in ['Str_Acc', 'Str_Def', 'TD_Acc', 'TD_Def']:
    fstats_df[c] = pd.to_numeric(
        fstats_df[c].astype(str).str.replace('%', '', regex=False),
        errors='coerce').fillna(0) / 100.0

elo_hist   = pd.read_csv('data/elo_ratings_history.csv')
elo_hist['date'] = pd.to_datetime(elo_hist['date'])
elo_hist   = elo_hist.sort_values(['fighter', 'date']).reset_index(drop=True)

# ── Filter to M2 universe: 2018+, all method odds, valid winner, men's only ───
df = df_master[
    (df_master['date'] >= '2018-01-01') &
    df_master['R_odds'].notna() & df_master['B_odds'].notna() &
    df_master['r_dec_odds'].notna() & df_master['b_dec_odds'].notna() &
    df_master['r_sub_odds'].notna() & df_master['b_sub_odds'].notna() &
    df_master['r_ko_odds'].notna()  & df_master['b_ko_odds'].notna() &
    df_master['Winner'].isin(['Red', 'Blue']) &
    ~df_master['weight_class'].isin(WOMENS_CLASSES)
].copy().reset_index(drop=True)

print(f"  M2 universe (men's only): {len(df)} fights")

# Corner randomization — same seed=42 as original sprint
np.random.seed(SEED)
swap_mask = np.random.random(len(df)) < 0.5
r_matched = sorted([c for c in df.columns if c.startswith('R_') and ('B_'+c[2:]) in df.columns])
b_matched = ['B_'+c[2:] for c in r_matched]
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

print(f"  Train (2018-2023): {len(train_idx)} | Test (2024+): {len(test_idx)}")
print(f"  F1 win rate after randomization: {target.mean():.3f}")

# ── Career stats timeline ─────────────────────────────────────────────────────
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
    'cum_fights', 'career_win_rate', 'ko_finish_rate', 'sub_finish_rate',
    'last3_win_rate', 'last5_won', 'last10_win_rate', 'last5_finish_rate',
    'trend_score', 'layoff_days', 'opp_quality',
]
DEFAULT_CAREER = {
    'cum_fights': 0, 'career_win_rate': 0.5, 'ko_finish_rate': 0.0,
    'sub_finish_rate': 0.0, 'last3_win_rate': 0.5, 'last5_won': 0.5,
    'last10_win_rate': 0.5, 'last5_finish_rate': 0.0,
    'trend_score': 0.0, 'layoff_days': 365.0, 'opp_quality': 0.5,
}

career_by_f = {}; career_dates_f = {}
for fname, grp in cf.groupby('fighter'):
    g_ = grp.reset_index(drop=True)
    career_by_f[fname]    = g_
    career_dates_f[fname] = g_['date'].tolist()

def get_career_at(fighter, fight_date):
    if fighter not in career_by_f:
        return DEFAULT_CAREER.copy()
    idx = bisect.bisect_right(career_dates_f[fighter], fight_date) - 1
    if idx < 0:
        return DEFAULT_CAREER.copy()
    row = career_by_f[fighter].iloc[idx]
    return {c: float(row[c]) for c in CAREER_COLS}

elo_by_f = {}; elo_dates_f = {}
for fname, grp in elo_hist.groupby('fighter'):
    g_ = grp.sort_values('date').reset_index(drop=True)
    elo_by_f[fname]    = g_
    elo_dates_f[fname] = g_['date'].tolist()

def get_elo_at(fighter, fight_date):
    if fighter not in elo_by_f:
        return {'elo': 1500.0, 'elo_trend': 0.0}
    idx = bisect.bisect_left(elo_dates_f[fighter], fight_date) - 1
    if idx < 0:
        return {'elo': 1500.0, 'elo_trend': 0.0}
    row = elo_by_f[fighter].iloc[idx]
    return {'elo': float(row['elo_after']), 'elo_trend': float(row.get('elo_trend', 0.0) or 0.0)}

fstyle = {}
for _, row in fstats_df.iterrows():
    fstyle[row['Fighter_Name']] = {k: float(row.get(k, 0) or 0)
        for k in ['SLpM', 'SApM', 'Str_Acc', 'Str_Def', 'TD_Avg', 'TD_Acc', 'TD_Def', 'Sub_Avg']}

def g(row, col, default=0.0):
    v = row.get(col, default) if isinstance(row, dict) else getattr(row, col, default)
    try:
        if pd.isna(v): return float(default)
    except Exception: pass
    return float(v) if v is not None else float(default)

def layoff_buckets(days):
    return {'lt90': 1 if days < 90 else 0, '90_180': 1 if 90 <= days < 180 else 0,
            '180_365': 1 if 180 <= days < 365 else 0, 'gt365': 1 if days >= 365 else 0}

print(f"  Career data for {len(career_by_f)} fighters")

# ── Build 129-feature matrix (base 109 + QA approx + interactions) ────────────
print("\n[SETUP] Building 129-feature matrix...")

def build_features_129(df_row):
    r_name = df_row['R_fighter']; b_name = df_row['B_fighter']; fdate = df_row['date']
    rc = get_career_at(r_name, fdate); bc = get_career_at(b_name, fdate)
    rs = fstyle.get(r_name, {}); bs = fstyle.get(b_name, {})
    re = get_elo_at(r_name, fdate); be = get_elo_at(b_name, fdate)
    r_lb = layoff_buckets(rc['layoff_days']); b_lb = layoff_buckets(bc['layoff_days'])
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
    b_ttb=g(df_row,'B_total_title_bouts')
    r_ko=g(df_row,'R_win_by_KO/TKO'); b_ko=g(df_row,'B_win_by_KO/TKO')
    r_sub=g(df_row,'R_win_by_Submission'); b_sub=g(df_row,'B_win_by_Submission')
    wc_ord = WC_ORDER.get(str(df_row.get('weight_class', '') or ''), 6)
    r_axe  = r_age * rc['cum_fights']; b_axe = b_age * bc['cum_fights']

    # QA approximations (same as backend inference)
    r_qa_wr  = rc['career_win_rate']; b_qa_wr  = bc['career_win_rate']
    r_qa_fr  = rc['last5_finish_rate']; b_qa_fr  = bc['last5_finish_rate']

    # Interaction features
    r_layoff_cap = min(rc['layoff_days'], 730); b_layoff_cap = min(bc['layoff_days'], 730)
    r_fd = rc['ko_finish_rate'] + rc['sub_finish_rate']
    b_fd = bc['ko_finish_rate'] + bc['sub_finish_rate']

    total_r = r_wins + r_loss
    total_b = b_wins + b_loss
    r_gfr = (r_loss / total_r) * 0.5 if total_r > 0 else 0.5
    b_gfr = (b_loss / total_b) * 0.5 if total_b > 0 else 0.5

    fd_mismatch = r_fd * b_gfr - b_fd * r_gfr

    r_ttb_val = g(df_row, 'R_total_title_bouts')
    total_ttb_dif = r_ttb_val - b_ttb

    return {
        # Base 109
        'R_wins':r_wins, 'R_losses':r_loss, 'R_Height_cms':r_h, 'R_age':r_age,
        'R_avg_SIG_STR_landed':r_sig, 'R_avg_TD_landed':r_td,
        'R_current_win_streak':r_ws, 'R_current_lose_streak':r_ls,
        'R_longest_win_streak':r_lws, 'R_avg_SIG_STR_pct':r_sigp,
        'R_avg_SUB_ATT':r_suba, 'R_avg_TD_pct':r_tdp, 'R_Reach_cms':r_rch,
        'B_wins':b_wins, 'B_losses':b_loss, 'B_Height_cms':b_h, 'B_age':b_age,
        'B_avg_SIG_STR_landed':b_sig, 'B_avg_TD_landed':b_td,
        'B_current_win_streak':b_ws, 'B_current_lose_streak':b_ls,
        'B_longest_win_streak':b_lws, 'B_avg_SIG_STR_pct':b_sigp,
        'B_avg_SUB_ATT':b_suba, 'B_avg_TD_pct':b_tdp, 'B_Reach_cms':b_rch,
        'B_total_title_bouts':b_ttb,
        'win_dif':r_wins-b_wins, 'loss_dif':r_loss-b_loss,
        'win_streak_dif':r_ws-b_ws, 'lose_streak_dif':r_ls-b_ls,
        'height_dif':r_h-b_h, 'reach_dif':r_rch-b_rch, 'age_dif':r_age-b_age,
        'sig_str_dif':r_sig-b_sig, 'avg_td_dif':r_td-b_td,
        'ko_dif':r_ko-b_ko, 'sub_dif':r_sub-b_sub, 'total_title_bout_dif':total_ttb_dif,
        'weight_class_ord':wc_ord,
        'orth_clash':1 if (r_sp==0 and b_sp==0) else 0,
        'south_clash':1 if (r_sp==1 and b_sp==1) else 0,
        'R_southpaw':r_sp,
        'R_cum_fights':rc['cum_fights'], 'B_cum_fights':bc['cum_fights'],
        'R_career_win_rate':rc['career_win_rate'], 'B_career_win_rate':bc['career_win_rate'],
        'career_win_rate_dif':rc['career_win_rate']-bc['career_win_rate'],
        'R_last5_won':rc['last5_won'], 'B_last5_won':bc['last5_won'],
        'last5_won_dif':rc['last5_won']-bc['last5_won'],
        'R_last5_finish_rate':rc['last5_finish_rate'], 'B_last5_finish_rate':bc['last5_finish_rate'],
        'last5_finish_rate_dif':rc['last5_finish_rate']-bc['last5_finish_rate'],
        'R_opp_quality':rc['opp_quality'], 'B_opp_quality':bc['opp_quality'],
        'opp_quality_dif':rc['opp_quality']-bc['opp_quality'],
        'R_trend_score':rc['trend_score'], 'B_trend_score':bc['trend_score'],
        'trend_score_dif':rc['trend_score']-bc['trend_score'],
        'R_ko_finish_rate':rc['ko_finish_rate'], 'B_ko_finish_rate':bc['ko_finish_rate'],
        'ko_finish_rate_dif':rc['ko_finish_rate']-bc['ko_finish_rate'],
        'R_sub_finish_rate':rc['sub_finish_rate'], 'B_sub_finish_rate':bc['sub_finish_rate'],
        'sub_finish_rate_dif':rc['sub_finish_rate']-bc['sub_finish_rate'],
        'R_last3_win_rate':rc['last3_win_rate'], 'B_last3_win_rate':bc['last3_win_rate'],
        'last3_win_rate_dif':rc['last3_win_rate']-bc['last3_win_rate'],
        'R_last10_win_rate':rc['last10_win_rate'], 'B_last10_win_rate':bc['last10_win_rate'],
        'R_age_x_exp':r_axe, 'B_age_x_exp':b_axe, 'age_x_exp_dif':r_axe-b_axe,
        'R_layoff_lt90':r_lb['lt90'], 'R_layoff_90_180':r_lb['90_180'],
        'R_layoff_180_365':r_lb['180_365'], 'R_layoff_gt365':r_lb['gt365'],
        'B_layoff_lt90':b_lb['lt90'], 'B_layoff_90_180':b_lb['90_180'],
        'B_layoff_180_365':b_lb['180_365'],
        'R_SLpM':rs.get('SLpM',0), 'R_SApM':rs.get('SApM',0),
        'R_Str_Acc':rs.get('Str_Acc',0), 'R_Str_Def':rs.get('Str_Def',0),
        'R_TD_Avg':rs.get('TD_Avg',0), 'R_TD_Acc':rs.get('TD_Acc',0),
        'R_TD_Def':rs.get('TD_Def',0), 'R_Sub_Avg':rs.get('Sub_Avg',0),
        'B_SLpM':bs.get('SLpM',0), 'B_SApM':bs.get('SApM',0),
        'B_Str_Acc':bs.get('Str_Acc',0), 'B_Str_Def':bs.get('Str_Def',0),
        'B_TD_Avg':bs.get('TD_Avg',0), 'B_TD_Acc':bs.get('TD_Acc',0),
        'B_TD_Def':bs.get('TD_Def',0), 'B_Sub_Avg':bs.get('Sub_Avg',0),
        'SLpM_dif':rs.get('SLpM',0)-bs.get('SLpM',0),
        'SApM_dif':rs.get('SApM',0)-bs.get('SApM',0),
        'Str_Def_dif':rs.get('Str_Def',0)-bs.get('Str_Def',0),
        'TD_Def_dif':rs.get('TD_Def',0)-bs.get('TD_Def',0),
        'Sub_Avg_dif':rs.get('Sub_Avg',0)-bs.get('Sub_Avg',0),
        'TD_Avg_dif':rs.get('TD_Avg',0)-bs.get('TD_Avg',0),
        'R_elo':re['elo'], 'B_elo':be['elo'], 'elo_dif':re['elo']-be['elo'],
        'R_elo_trend':re['elo_trend'], 'B_elo_trend':be['elo_trend'],
        'elo_trend_dif':re['elo_trend']-be['elo_trend'],
        # QA features (approximated — consistent with production inference)
        'R_qa_win_rate':r_qa_wr, 'R_qa_finish_rate':r_qa_fr,
        'R_qa_SLpM':0.0, 'R_qa_SApM':0.0,
        'B_qa_win_rate':b_qa_wr, 'B_qa_finish_rate':b_qa_fr,
        'B_qa_SLpM':0.0, 'B_qa_SApM':0.0,
        'qa_win_rate_dif':r_qa_wr-b_qa_wr, 'qa_finish_rate_dif':r_qa_fr-b_qa_fr,
        'qa_SLpM_dif':0.0, 'qa_SApM_dif':0.0,
        # Interaction features
        'R_age_x_layoff':r_age * r_layoff_cap, 'B_age_x_layoff':b_age * b_layoff_cap,
        'age_x_layoff_dif':r_age * r_layoff_cap - b_age * b_layoff_cap,
        'R_finish_danger':r_fd, 'B_finish_danger':b_fd,
        'finish_danger_mismatch':fd_mismatch,
        'R_got_finished_rate':r_gfr, 'B_got_finished_rate':b_gfr,
    }

rows_list = [build_features_129(df_row) for _, df_row in df.iterrows()]
X_df = pd.DataFrame(rows_list, columns=feat_cols_m1)
X_m1 = X_df[feat_cols_m1].values.astype(float)
cm   = np.nanmedian(X_m1, axis=0)
nm   = np.isnan(X_m1)
X_m1[nm] = np.take(cm, np.where(nm)[1])
print(f"  Feature matrix: {X_m1.shape}")
gc.collect()

# ── M1 OOF predictions ────────────────────────────────────────────────────────
print("\n[M1] Generating OOF predictions on men's training set...")
X_m1_train = X_m1[train_idx]; X_m1_test = X_m1[test_idx]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
m1_oof = np.zeros(len(train_idx))

for fold_i, (tr_i, va_i) in enumerate(skf.split(X_m1_train, y_train)):
    fl = Pipeline([('sc', RobustScaler()),
                   ('clf', LogisticRegression(C=0.00711, penalty='l2',
                                              max_iter=2000, solver='saga',
                                              random_state=SEED))])
    fl.fit(X_m1_train[tr_i], y_train[tr_i])
    fx = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3,
                       subsample=0.8, colsample_bytree=0.8,
                       eval_metric='logloss', random_state=SEED,
                       verbosity=0, n_jobs=1)
    fx.fit(X_m1_train[tr_i], y_train[tr_i])
    m1_oof[va_i] = (0.70 * fl.predict_proba(X_m1_train[va_i])[:,1] +
                    0.30 * fx.predict_proba(X_m1_train[va_i])[:,1])

m1_test = (0.70 * model_lr_m1.predict_proba(X_m1_test)[:,1] +
           0.30 * model_xgb_m1.predict_proba(X_m1_test)[:,1])

m1_train_acc_oof = accuracy_score(y_train, (m1_oof > 0.5).astype(int))
m1_test_acc      = accuracy_score(y_test,  (m1_test > 0.5).astype(int))
print(f"  M1 OOF train acc: {m1_train_acc_oof:.4f}")
print(f"  M1 test acc:      {m1_test_acc:.4f}")
gc.collect()

# ── Build M2 feature matrix (42 features) ────────────────────────────────────
print("\n[M2] Building 42-feature dataset...")

def implied_prob(odds):
    try:
        odds = float(odds)
        if odds == 0 or np.isnan(odds): return None
        return abs(odds)/(abs(odds)+100) if odds < 0 else 100/(odds+100)
    except Exception: return None

def novig_probs(f1_odds, f2_odds):
    f1_raw = implied_prob(f1_odds) or 0.5
    f2_raw = implied_prob(f2_odds) or 0.5
    total  = f1_raw + f2_raw
    if total <= 0: return 0.5, 0.5, 0.0
    return f1_raw/total, f2_raw/total, total-1.0

BASE_M2 = ['model1_prob','f1_ml_novig','f2_ml_novig','ml_gap','vig',
           'f1_dec_implied','f2_dec_implied','dec_implied_dif',
           'f1_ko_implied','f2_ko_implied','ko_implied_dif',
           'f1_sub_implied','f2_sub_implied','sub_implied_dif',
           'finish_prob','f1_finish_prob','f2_finish_prob','finish_advantage',
           'abs_gap','vegas_confidence','model_confidence','model_agrees_vegas','gap_x_confidence']

train_pos = {v: k for k, v in enumerate(train_idx)}
test_pos  = {v: k for k, v in enumerate(test_idx)}

m2_rows = []
for i, (_, df_row) in enumerate(df.iterrows()):
    m1p = float(m1_oof[train_pos[i]]) if i in train_pos else float(m1_test[test_pos[i]])
    f1n, f2n, vig_ = novig_probs(df_row['R_odds'], df_row['B_odds'])
    ml_gap_ = m1p - f1n
    f1_dec = implied_prob(df_row['r_dec_odds']) or 0.0
    f2_dec = implied_prob(df_row['b_dec_odds']) or 0.0
    f1_ko  = implied_prob(df_row['r_ko_odds'])  or 0.0
    f2_ko  = implied_prob(df_row['b_ko_odds'])  or 0.0
    f1_sub = implied_prob(df_row['r_sub_odds']) or 0.0
    f2_sub = implied_prob(df_row['b_sub_odds']) or 0.0
    dec_tot = f1_dec + f2_dec
    fin_p   = 1.0 - (dec_tot / 2.0) if dec_tot > 0 else 0.5
    f1_fin  = f1_ko + f1_sub; f2_fin = f2_ko + f2_sub
    m2_rows.append([m1p, f1n, f2n, ml_gap_, vig_,
                    f1_dec, f2_dec, f1_dec-f2_dec,
                    f1_ko,  f2_ko,  f1_ko-f2_ko,
                    f1_sub, f2_sub, f1_sub-f2_sub,
                    fin_p, f1_fin, f2_fin, f1_fin-f2_fin,
                    abs(ml_gap_), abs(f1n-0.5), abs(m1p-0.5),
                    1 if (m1p > 0.5) == (f1n > 0.5) else 0,
                    ml_gap_ * abs(f1n-0.5)])

X2_base = np.array(m2_rows, dtype=float)
cm2 = np.nanmedian(X2_base, axis=0); nm2 = np.isnan(X2_base)
X2_base[nm2] = np.take(cm2, np.where(nm2)[1])

# Step 1: fav/dog profile + tier
df_fh = df[['date','R_fighter','B_fighter','R_odds','B_odds','Winner']].copy()
df_fh['f1_won']    = (df_fh['Winner'] == 'Red').astype(int)
df_fh['f1_is_fav'] = (df_fh['R_odds'] < 0).astype(int)
df_sorted = df_fh.sort_values('date').reset_index()
fav_bouts = {}; fav_wins = {}; dog_bouts = {}; dog_wins = {}
step1_rows = [None] * len(df)
tier_train_counts = {t: 0 for t in range(5)}
tier_train_wins   = {t: 0 for t in range(5)}

for _, row in df_sorted.iterrows():
    orig_i = row['index']; f1 = row['R_fighter']
    is_fav = row['f1_is_fav']; f1_won = row['f1_won']
    f1nv, _, _ = novig_probs(df.loc[orig_i, 'R_odds'], df.loc[orig_i, 'B_odds'])
    t = 0 if f1nv < 0.30 else (1 if f1nv < 0.45 else (2 if f1nv < 0.55 else (3 if f1nv < 0.70 else 4)))
    fav_n = len(fav_bouts.get(f1, [])); fav_w = fav_wins.get(f1, 0)
    dog_n = len(dog_bouts.get(f1, [])); dog_w = dog_wins.get(f1, 0)
    step1_rows[orig_i] = [is_fav,
                          fav_w/fav_n if fav_n > 0 else 0.5,
                          dog_w/dog_n if dog_n > 0 else 0.5,
                          math.log1p(fav_n), math.log1p(dog_n),
                          abs(f1nv - 0.5), t]
    # accumulate tier stats from training rows only
    if orig_i in train_pos:
        tier_train_counts[t] += 1
        tier_train_wins[t]   += f1_won
    if is_fav:
        fav_bouts.setdefault(f1, []).append(row['date'])
        fav_wins[f1] = fav_wins.get(f1, 0) + f1_won
    else:
        dog_bouts.setdefault(f1, []).append(row['date'])
        dog_wins[f1] = dog_wins.get(f1, 0) + f1_won

tier_wr_map = {t: tier_train_wins[t]/tier_train_counts[t]
               if tier_train_counts[t] > 0 else 0.5
               for t in range(5)}

step1_arr = np.array(step1_rows, dtype=float)
tier_hist_wr = np.array([float(tier_wr_map.get(int(t), 0.5)) for t in step1_arr[:, 6]])
step1_final  = np.column_stack([step1_arr[:, :6], tier_hist_wr])

# Step 2: method × style interactions
feat_idx = {f: i for i, f in enumerate(feat_cols_m1)}
step2_rows = []
for i in range(len(df)):
    br = X2_base[i]
    f1_ko_i  = br[BASE_M2.index('f1_ko_implied')];  f2_ko_i  = br[BASE_M2.index('f2_ko_implied')]
    f1_sub_i = br[BASE_M2.index('f1_sub_implied')]; f2_sub_i = br[BASE_M2.index('f2_sub_implied')]
    f1_dec_i = br[BASE_M2.index('f1_dec_implied')]; f2_dec_i = br[BASE_M2.index('f2_dec_implied')]
    m1p_     = br[BASE_M2.index('model1_prob')];    fin_p_   = br[BASE_M2.index('finish_prob')]
    r_ko_fr  = X_m1[i, feat_idx['R_ko_finish_rate']]
    b_ko_fr  = X_m1[i, feat_idx['B_ko_finish_rate']]
    r_sub_fr = X_m1[i, feat_idx['R_sub_finish_rate']]
    b_sub_fr = X_m1[i, feat_idx['B_sub_finish_rate']]
    str_def  = X_m1[i, feat_idx['Str_Def_dif']]
    step2_rows.append([f1_ko_i*r_ko_fr - f2_ko_i*b_ko_fr,
                       f1_sub_i*r_sub_fr - f2_sub_i*b_sub_fr,
                       fin_p_ * abs(m1p_ - 0.5),
                       ((f1_dec_i + f2_dec_i) / 2.0) * abs(str_def),
                       f1_ko_i + f2_ko_i, f1_sub_i + f2_sub_i,
                       abs(f1_ko_i - f2_ko_i), abs(f1_sub_i - f2_sub_i)])
step2_arr = np.array(step2_rows, dtype=float)

# Step 3: weight-class context
wc_arr = X_m1[:, feat_idx['weight_class_ord']]
no_rds = df.get('no_of_rounds', pd.Series([3]*len(df))).fillna(3).values
m1_train_acc_wc = {}
for wc_v in np.unique(wc_arr[train_mask]):
    mask = train_mask & (wc_arr == wc_v)
    if mask.sum() >= 5:
        m1_train_acc_wc[int(wc_v)] = accuracy_score(
            target[mask], (np.concatenate([m1_oof, m1_test])[
                np.concatenate([train_idx, test_idx]).argsort()[
                    np.concatenate([train_idx, test_idx])[np.concatenate([train_idx, test_idx]).argsort()
                        == np.where(mask)[0][:, None]].any(axis=1)
                ]
            ] > 0.5).astype(int)) if False else 0.5

# Simpler wc accuracy approach
m1_all_probs = np.empty(len(df))
m1_all_probs[train_idx] = m1_oof
m1_all_probs[test_idx]  = m1_test
m1_train_acc_wc = {}
for wc_v in np.unique(wc_arr[train_mask]):
    mask = train_mask & (wc_arr == wc_v)
    if mask.sum() >= 5:
        m1_train_acc_wc[int(wc_v)] = accuracy_score(
            target[mask], (m1_all_probs[mask] > 0.5).astype(int))

m1_train_acc_global = accuracy_score(y_train, (m1_oof > 0.5).astype(int))
step3_rows = []
for i in range(len(df)):
    wc_v    = wc_arr[i]; is_5r = 1 if no_rds[i] >= 5 else 0
    wc_a    = m1_train_acc_wc.get(int(wc_v), m1_train_acc_global)
    m1_conf = abs(X2_base[i, BASE_M2.index('model_confidence')])
    step3_rows.append([wc_v/11.0, is_5r, wc_a - m1_train_acc_global, is_5r * m1_conf])
step3_arr = np.array(step3_rows, dtype=float)

X2_full = np.hstack([X2_base, step1_final, step2_arr, step3_arr])
FEAT_NAMES = (BASE_M2 +
              ['f1_is_fav','f1_hist_fav_wr','f1_hist_dog_wr',
               'f1_fav_bouts_log','f1_dog_bouts_log','odds_strength','tier_hist_win_rate'] +
              ['ko_style_edge','sub_style_edge','finish_x_model_conf','dec_x_str_def',
               'combined_ko_implied','combined_sub_implied','ko_method_gap','sub_method_gap'] +
              ['wc_norm','is_5r','m1_wc_bias','five_r_x_conf'])
assert X2_full.shape[1] == 42, f"Expected 42 features, got {X2_full.shape[1]}"

X2_train = X2_full[train_idx]; X2_test = X2_full[test_idx]
print(f"  M2 feature matrix: {X2_full.shape}")
gc.collect()

# ── Train new M2 (50/50 LR+XGB) ──────────────────────────────────────────────
print("\n[M2] Training new 50/50 LR+XGB blend...")

from sklearn.model_selection import cross_val_score
import optuna

def train_lr(X, y):
    pipe = Pipeline([
        ('sc', RobustScaler()),
        ('clf', LogisticRegression(C=0.292291, penalty='l2', solver='lbfgs',
                                   max_iter=2000, random_state=SEED))
    ])
    pipe.fit(X, y)
    return pipe

def train_xgb(X, y):
    # Quick Optuna tune (15 trials) to find best XGB params on men's data
    def objective(trial):
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 100, 400),
            'max_depth':         trial.suggest_int('max_depth', 2, 6),
            'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight':  trial.suggest_int('min_child_weight', 1, 5),
            'gamma':             trial.suggest_float('gamma', 0.0, 0.5),
            'reg_lambda':        trial.suggest_float('reg_lambda', 0.5, 3.0),
        }
        clf = XGBClassifier(**params, eval_metric='logloss',
                            random_state=SEED, verbosity=0, n_jobs=1)
        cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy', n_jobs=1)
        return scores.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=15, show_progress_bar=False)
    best = study.best_params
    print(f"  XGB best CV acc: {study.best_value:.4f} | params: {best}")
    clf = XGBClassifier(**best, eval_metric='logloss',
                        random_state=SEED, verbosity=0, n_jobs=1)
    clf.fit(X, y)
    return clf

new_lr  = train_lr(X2_train, y_train)
new_xgb = train_xgb(X2_train, y_train)
gc.collect()

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\n[M2] Evaluating on test set (2024+, men's only)...")
lr_prob  = new_lr.predict_proba(X2_test)[:, 1]
xgb_prob = new_xgb.predict_proba(X2_test)[:, 1]
m2_probs = 0.50 * lr_prob + 0.50 * xgb_prob
m2_acc   = accuracy_score(y_test, (m2_probs > 0.5).astype(int))

print(f"  M1 test acc:        {m1_test_acc:.4f}")
print(f"  New M2 test acc:    {m2_acc:.4f}")
print(f"  Old M2 production:  {OLD_M2_ACC:.4f}")
delta = m2_acc - OLD_M2_ACC
print(f"  Delta vs old M2:    {delta:+.4f}")

# ── Threshold analysis ────────────────────────────────────────────────────────
print("\n[M2] Threshold analysis (5%, 8%, 10%, 12%, 15%)...")

f1_odds_arr = df['R_odds'].values.astype(float)
f2_odds_arr = df['B_odds'].values.astype(float)

def unit_return(odds):
    return 100.0 / abs(odds) if odds < 0 else odds / 100.0

def roi_sim(probs, threshold, use_agreement=False):
    profits = []
    for k, (row_i, m2p) in enumerate(zip(test_idx, probs)):
        f1n, f2n, _ = novig_probs(f1_odds_arr[row_i], f2_odds_arr[row_i])
        gap  = m2p - f1n
        m1p_ = float(m1_test[k])
        if use_agreement and (m2p > 0.5) != (m1p_ > 0.5):
            continue
        if abs(gap) < threshold:
            continue
        bet_f1 = gap > 0
        won    = bool(target[row_i] == 1) if bet_f1 else bool(target[row_i] == 0)
        odds   = f1_odds_arr[row_i] if bet_f1 else f2_odds_arr[row_i]
        profits.append(unit_return(float(odds)) if won else -1.0)
    n = len(profits)
    if n == 0:
        return 0, 0.0, 0.0, 0.0
    wins  = sum(1 for p in profits if p > 0)
    return n, wins/n, sum(profits)/n*100, sum(profits)

THRESHOLDS = [0.05, 0.08, 0.10, 0.12, 0.15]

print(f"\n  No agreement filter:")
print(f"  {'Thresh':>8} {'N':>6} {'Win%':>7} {'ROI%':>7} {'Profit':>10}")
print("  " + "-"*44)
std_results = {}
for t in THRESHOLDS:
    n, wr, roi, profit = roi_sim(m2_probs, t, use_agreement=False)
    std_results[t] = {'n_bets':n,'win_rate':round(wr,4),'roi_pct':round(roi,2),'total_profit_units':round(profit,2)}
    print(f"  {t:>8.2f} {n:>6} {wr:>7.3f} {roi:>7.2f}% {profit:>10.2f}")

print(f"\n  With M1+M2 agreement filter:")
print(f"  {'Thresh':>8} {'N':>6} {'Win%':>7} {'ROI%':>7} {'Profit':>10}")
print("  " + "-"*44)
agree_results = {}
for t in THRESHOLDS:
    n, wr, roi, profit = roi_sim(m2_probs, t, use_agreement=True)
    agree_results[t] = {'n_bets':n,'win_rate':round(wr,4),'roi_pct':round(roi,2),'total_profit_units':round(profit,2)}
    print(f"  {t:>8.2f} {n:>6} {wr:>7.3f} {roi:>7.2f}% {profit:>10.2f}")

# ── Save results ──────────────────────────────────────────────────────────────
output = {
    'new_m2_test_acc':     round(m2_acc, 4),
    'm1_test_acc':         round(m1_test_acc, 4),
    'old_m2_test_acc':     OLD_M2_ACC,
    'delta_vs_old':        round(delta, 4),
    'promoted':            m2_acc > OLD_M2_ACC,
    'n_train':             len(train_idx),
    'n_test':              len(test_idx),
    'filter':              'men_only_2018+',
    'standard_thresholds': {str(k): v for k, v in std_results.items()},
    'agreement_thresholds':{str(k): v for k, v in agree_results.items()},
    'tier_win_rates':      {str(k): round(v, 4) for k, v in tier_wr_map.items()},
    'm1_train_acc_oof':    round(m1_train_acc_oof, 4),
}
with open(RESULTS_DIR / 'model2_retrain_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n  Saved: experiments/research/model2_v2/model2_retrain_results.json")

# ── Promotion decision ────────────────────────────────────────────────────────
print("\n" + "=" * 65)
if m2_acc > OLD_M2_ACC:
    print(f"PROMOTING new M2: {m2_acc:.4f} > {OLD_M2_ACC:.4f} (+{delta:+.4f})")
    joblib.dump(new_lr,  'model/ufc_model2_best.pkl')
    joblib.dump(new_xgb, 'model/ufc_model2_xgb.pkl')
    joblib.dump(FEAT_NAMES, 'model/ufc_model2_features.pkl')
    # Update tier stats
    new_tier_stats = {
        'tier_win_rates':   {str(k): tier_wr_map[k] for k in sorted(tier_wr_map)},
        'm1_train_acc':     round(m1_train_acc_oof, 4),
        'm1_wc_acc':        {str(float(k)): round(v, 4) for k, v in m1_train_acc_wc.items()},
        'feature_names':    FEAT_NAMES,
        'blend_lr':         0.5,
        'blend_xgb':        0.5,
        'test_acc':         round(m2_acc, 4),
        'n_features':       42,
    }
    with open('model/model2_tier_stats.json', 'w') as f:
        json.dump(new_tier_stats, f, indent=2)
    # Update model_metadata.json
    with open('model/model_metadata.json') as f:
        meta = json.load(f)
    meta['model2']['temporal_accuracy'] = round(m2_acc, 4)
    meta['model2']['prev_accuracy']     = OLD_M2_ACC
    meta['model2']['delta_vs_prev']     = f'{delta:+.4f}'
    meta['model2']['training_universe'] = '2018+ men\'s fights with all ML + method odds'
    meta['model2']['n_train']           = len(train_idx)
    meta['model2']['n_test']            = len(test_idx)
    meta['model2']['date_trained']      = '2026-05-12'
    with open('model/model_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print("  Replaced: ufc_model2_best.pkl, ufc_model2_xgb.pkl, ufc_model2_features.pkl")
    print("  Updated:  model/model2_tier_stats.json, model/model_metadata.json")
else:
    print(f"HOLDING — new M2 {m2_acc:.4f} does not beat old {OLD_M2_ACC:.4f} (delta={delta:+.4f})")
    print("  Production files unchanged.")
print("=" * 65)
print("Done.")
